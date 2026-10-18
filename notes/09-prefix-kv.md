# Prefix KV reuse

The reference clip is prefilled once instead of once per chunk.

Implementation: `bin/prefix_kv.py`. Test: `bin/test_prefix_kv.py`.
A/B driver: `bin/prefix_ab.sh`. Enabled with `--prefix-kv` on `bin/narrate.py`.

## Why

From the time decomposition in `08-STATE-AND-NEXT.md` (fit over 234
single-attempt chunks of the real script):

```
decode  (producing speech)   25.06 min   80.3%
prefill (re-reading prompt)   5.97 min   19.1%
fixed per chunk               0.19 min    0.6%
```

Every prompt begins with the same system turn and the same reference block.
**Measured: 734 rows of a 768-row prompt — 96 %** — byte-identical in every
chunk. (`08` said 608 rows / 84 %; that estimate counted only the 614 audio
code rows and forgot the system text, the instruction, the transcript and the
markers. 734 is the measured figure.)

So the model was re-reading the same 28.5 seconds of audio once per chunk,
235 times.

## What it does

Prefill rows `[0, P)` once, snapshot every piece of state it produced, and
restore the snapshot before each chunk instead of recomputing it.

Three kinds of state, all of them required:

| | layers | what | size |
|---|---|---|---|
| `FullAttention.k_cache` / `v_cache` | 10 of 40 backbone | positional | 20 KiB/row |
| `GatedDeltaNet.conv_state` + `recurrent_state` | the other 30 | **recurrent** | 61 MiB flat |
| `TowerAttention.k_cache` / `v_cache` | 13 tower | positional | 26 KiB/row |

Plus one row of backbone output, `hidden[P-1]`.

**Total 94 MiB** for a 28.5 s reference. It lives on the GPU and comes out of
the 1.5 GiB of activation headroom `size_cache_to_free_vram` already holds
back, *not* out of `--vram-reserve`. Restoring is ~53 device-to-device copies.

The GDN state is the one that bites. It is not indexed by position — it is a
single accumulated state — so there is nothing to slice and nothing that
"just works". Leave it out and the model still generates fluent audio; it is
simply conditioned on the wrong thing.

### The off-by-one

`tower.prefill` teacher-forces row *p* with frame *p+1* as its label. Row
`P-1`'s label is `frames[P]`, the first row of the chunk-specific text, so
row `P-1` cannot be towered during the prefix build. The snapshot holds tower
rows `[0, P-1)` and carries `hidden[P-1]` forward; the resumed prefill
tower-fills row `P-1` once it knows `frames[P]`.

Get this wrong by one and the tower pairs the reference's last frame with
itself. It still generates.

## Correctness

`bin/test_prefix_kv.py` asks four separate questions. Conflating them is how
you end up arguing about whether 1e-3 is a bug.

```
0 determinism   two identical passes differ by 0.000e+00
A faithfulness  PASS  hidden 0.000e+00; 0/106 cache tensors above the floor
B drift         vs --prefill-chunk 1024: 1.3-1.8% of the hidden state's scale
C cost          prefix+suffix 1786 ms -> restore+suffix 491 ms (72% less)
E control       --prefill-chunk 128 vs 1024, same seed: divergence at frame 0
```

**A is the test that matters.** Build the state in one pass, versus snapshot
it, zero every cache, restore, and resume. All 106 cache tensors and the
carried hidden row come back **bit-identical**. The caches are zeroed first on
purpose: without that, A passes for the wrong reason, because the previous
pass left the right bytes in place and a restore that copies nothing looks
correct.

**B is expected, not a defect.** Resuming at row P moves where GatedDeltaNet's
64-row chunk boundaries fall and shifts the tower's batching by one row, so
the last bits of the hidden state move. The frames then differ from frame 0,
because a one-ulp bf16 change can flip a 4,097-way nucleus draw.

**E is why we believe that.** Move a prefill boundary with no prefix logic
involved at all — `--prefill-chunk 128` against `1024`, same seed — and the
output diverges at frame 0 too. The prefix path is not implicated; this model
is simply that sensitive to boundary placement. It always was.

## Cost measured end to end

A/B over the first 20 chunks of the real script, production configuration
(`--dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts
--vram-reserve 2 --prefill-chunk 1024`), two separate processes so neither arm
can warm the other:

```
arm    chunks   audio s     gen s    RTF  prefill s  rows kept  retries
off        20     244.6     248.0  1.014       0.00          0        0
on         20     234.7     222.7  0.949      22.41        734        0
```

**RTF 1.014 -> 0.949: 6.4 % faster per second of audio.**

⚠️ The raw "248.0 s -> 222.7 s, 10.2 % less" that the driver prints is
**not** the honest number, and the reason is worth keeping. The two arms do
not produce the same audio — sampling diverges from frame 0 (see B and E
above) — so the `on` arm happened to generate 9.9 s less speech. Some of its
25.3 s advantage is just having had less to say. RTF normalises that out;
quote RTF.

Prefill alone, straight from the two logs:

```
off   20 prefills, total 39.42 s, mean 1.971 s
on    20 prefills, total 22.41 s, mean 1.120 s      43 % less
```

Zero retries in either arm, so nothing was traded for the speed.

### The shape of what is left — this is the next job

Per-prefill timings from the `on` arm, rows actually prefilled against the
seconds they took:

```
 11 rows -> 0.28 s        453 rows -> 1.25 s
 71 rows -> 0.80 s        570 rows -> 1.25 s
147 rows -> 1.03 s        625 rows -> 1.26 s
```

57x the rows for 4.5x the time. And in the `off` arm the giveaway: a 745-row
prompt is one batch at `--prefill-chunk 1024` and costs 1.35 s, while an
1180-row prompt is **two** batches and costs 2.42 s — almost exactly double,
despite being only 58 % more rows.

So prefill cost tracks **batches**, not rows, and the per-batch cost is bytes:
a batch routes to somewhere between ~60 and ~200 distinct experts per layer,
and every one of them is fetched over PCIe at 3.001 MiB a piece.

```
 11 rows: ~60 experts x 40 layers x 3.0 MiB =  7.0 GiB / 23 GB/s = 0.31 s   (measured 0.28)
625 rows: ~200 experts x 40 layers x 3.0 MiB = 23.4 GiB / 23 GB/s = 1.04 s   (measured 1.26)
```

⚠️ **Prefill is PCIe-bound. Decode is not.** The headline finding of this
project — "the bottleneck was never PCIe, it was dispatch" — is true of
decode and false of prefill, and conflating the two will send the next
attempt at the wrong target. Decode touches 8 experts per layer per step and
the device cache serves most of them; a prefill batch touches 25-80 % of all
256 and the gather path (`offload.offloaded_forward`, `COPY_MODE=direct`)
**ignores the resident device cache entirely**, so its hit rate is zero by
construction. See task #25.

## What it unlocks

Reference length used to trade against speed — 56 s cost ~10 % on *every*
chunk, so quality was being rationed to pay for it. Paid once, that stops:

```
90 s reference, recomputed:  1938 rows x 1.91 ms x 235 = 14.5 min
90 s reference, reused:      1938 rows x 1.91 ms x   1 =  3.7 s
```

The user's judgement on reference length was 20 s "bad", 28.2 s "no longer a
different voice", 56 s "a better yet but not much". The current clip
(`source clips/test audio.mp3`) is **56 s long**, so its full length is now
free to use — `--ref-seconds 56`. Going past that needs a longer source clip.

## Guards, and why each exists

- **`PrefixKV.applies_to`** compares the prompt's first P rows against the
  snapshot's rows before resuming. Cheap (768 x 11 int64). Without it, a
  prefix that stops matching does not crash — it generates a chunk from
  somebody else's KV cache. `Drama3History.candidates()` can drop the pinned
  turn on a long prompt, so a non-matching prompt is reachable, and the
  fallback to a full prefill is recorded per chunk in the manifest.
- **`PrefixKV.is_valid`** compares cache `data_ptr`s against the ones the
  snapshot was taken from. CUDA-graph capture and `ensure_drama3_caches` both
  reallocate; the snapshot is then stale *and* pointing at freed buffers. On
  mismatch it rebuilds rather than restoring garbage.
- **`UPSTREAM_HASHES`** pins the source of the two upstream functions this
  module forks (`build_drama3_prompt`, `generate_drama3`). A copy goes stale
  silently; a hash does not.

## Rejected

- **Including chunk 0 in the prefix.** `Drama3History` pins chunk 0 and emits
  it as the first history turn of every later prompt, so the rows would line
  up. But the reference already covers 96 % of the prompt, leaving a few per
  cent, and it only holds while `candidates()` has not dropped the pinned
  turn. Not worth the extra failure mode.
- **Holding the snapshot in pinned host RAM.** Would cost ~8 ms of PCIe per
  chunk instead of ~0.3 ms, to save 94 MiB of VRAM that is not scarce. The
  flag was written and removed; add it back only if the snapshot grows past
  the activation headroom.

## Notes for whoever is next

- The remaining prefill is **1.12 s per chunk, ~10 % of runtime**, and it is
  bytes over PCIe, not launches. See "the shape of what is left" above. Task
  #25. ⚠️ My first reading of this was "a prefill batch routes to nearly all
  256 experts regardless of row count" — wrong. It does scale with rows
  (0.28 s at 11, 1.26 s at 625), it just saturates, and what saturates is the
  distinct-expert count. The fix is fewer bytes, not fewer kernel launches.
- The test ran with `--cuda-graph 0`. Graphs only touch decode, and leaving
  them off keeps A honest — capture runs three decode steps through the live
  caches, which is exactly the corruption `is_valid` exists to survive. Use
  `--cuda-graph 1` to exercise that recovery path on purpose.
