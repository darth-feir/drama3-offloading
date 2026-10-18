# State of play and what to do next

Written 2026-10-17 end of day, before a context compaction. This is the
pick-up-here document. Read `00-INDEX.md` for the rest.

## Where it got to

Drama 3 (35B MoE, FP8) narrates long documents on one 24 GB RTX 4090 with
all 30 GiB of routed experts in system RAM.

| | start of 2026-10-17 | end |
|---|---|---|
| long-form RTF | 3.90 | **0.85** on the full script (1.01 on 2026-10-17; see `09`-`12`) |
| wall clock, 31 min of audio | — | **27.0 min**, of which 32.5 s is startup |
| prefill share of runtime | — | 19.1 % -> **8.2 %** |
| compute for the 32-min script | 130 min | ~32 min |
| chunks losing content | 4 / 192 | **0 / 235** |
| dead air | 7.0 % of file | ~1 % |
| VRAM / host RAM | — | 19.8 GiB / 37.9 GiB |

### The working configuration

```bash
bin/run.sh bin/narrate.py --checkpoint "$DRAMA3_CKPT" \
  --text-file FILE --mode paragraph --out-dir DIR \
  --dev-cache --fetch-backend cpp --cuda-graph 1 \
  --pin-experts --vram-reserve 2 --prefill-chunk 1024 \
  --prefix-kv --slice-lm-head --offload-embed \
  --ref-seconds 56
```

Measured on 20 chunks: **RTF 0.872**, VRAM peak 19.76 GiB, host RAM
38.9 GiB. `--vram-reserve 3.9` instead of 2 gives the desktop 1.94 GiB back
and costs 2.5 % of RTF (0.872 -> 0.894); that is the whole exchange rate.

`--prefix-kv` added 2026-10-18 (`09-prefix-kv.md`). It also makes reference
length nearly free, so `--ref-seconds 56` — the full length of the current
clip, which the user judged "a better yet" — now costs almost nothing.

Defaults now carry the rest: temperature 1.0, top_p 0.9, top_k 4097,
`--ref-seconds 30` with silence snapping.

⚠️ This file used to say **do not add `--slice-lm-head` with
`--cuda-graph 1`, it NaNs**. That was true until 2026-10-18 and is now
wrong: the cause was the slice's own `-inf` padding tripping the graph's
finiteness check over the full 252,288-wide logits, not anything about the
slice. Fixed, and it is part of the working config above. See `10`.

### The two findings that mattered

1. **Dispatch, not PCIe.** The predicted failure (bandwidth) never happened.
   ~86 % of each step was the GPU idle waiting for kernel launches. A
   device-resident expert cache + a `uint4` fetch kernel
   (`csrc/fetch_experts.cu`) inside a captured CUDA graph took RTF 1.71 ->
   0.97.
2. **A sampler bug, not model behaviour.** `tower.py:57` drops the token
   crossing `top_p`. At the old default 0.7, EOS was drawn **0/4000 times at
   every temperature** -- end-of-speech was unreachable, so the model emitted
   silence until the cap. Changing to 1.0/0.9/4097 took content loss from
   5/235 chunks to **0/235**.

## Next, in priority order

### 1. Prefix KV reuse -- ✅ DONE 2026-10-18, see `09-prefix-kv.md`

Built as `bin/prefix_kv.py`, enabled with `--prefix-kv`. Restore is
bit-identical to a full prefill (all 106 cache tensors, zero difference).
Measured on 20 real chunks: **RTF 1.014 -> 0.949**, prefill 39.42 s -> 22.41 s
(43 % less), zero retries either arm.

Two corrections to what is written below:

- The prefix is **734 rows of a 768-row prompt (96 %)**, not 608 / 84 %. The
  old estimate counted only the 614 audio code rows and forgot the system
  text, the instruction, the transcript and the markers.
- The snapshot is **94 MiB**, not ~99 MB of GDN state alone -- 61 MiB of that
  is the recurrent state, the rest is the positional caches.

The prediction below of "~15 % off total runtime" came in at 6.4 %, because
the suffix after the prefix is bigger in real chunks than the arithmetic
assumed: real prompts carry history turns, so a median prompt is ~1,100 rows,
not 724. **What remains is now task #25 -- see `09`, the prefill that is left
is PCIe-bound and the gather ignores the resident cache.**

The original plan, kept for the record:

Measured decomposition of generation time (fit over 234 single-attempt
chunks):

```
decode  (producing speech)        25.06 min   80.3%    36.79 ms/frame
prefill (re-reading the prompt)    5.97 min   19.1%     1.91 ms/row
fixed per-chunk                    0.19 min    0.6%
```

The prefill is nearly all waste: **the reference is 608 rows, 84 % of a
median 724-row prompt, and byte-identical in all 235 chunks.** We re-read the
same 28 s of reference 235 times.

Cache the KV of the unchanging prefix (system + reference + pinned chunk 0)
once and resume from it. Plausibly ~15 % off total runtime.

**And it unlocks something better.** Reference length currently trades
against speed (56 s cost 10 %). With reuse it is paid once:

```
90 s reference, today:  1938 rows x 1.91 ms x 235 chunks = 14.5 min of prefill
90 s reference, reused: 1938 rows x 1.91 ms x 1          =  3.7 seconds
```

So a 90 s+ reference becomes free, and reference quality is the main lever
left on voice consistency (user: 20 s "bad", 28.2 s "no longer a different
voice", 56 s "better yet but not much").

Structure supports it: `FullAttention.k_cache/v_cache` (`layers.py:137-142`)
and `TowerAttention` (`tower.py:94-97`) are positional, so a prefix slice
stays valid. `GatedDeltaNet` (`layers.py:344-361`) is recurrent but its whole
state is `conv_state` (48 KiB) + `recurrent_state` (2.0 MiB fp32) per layer
-- ~99 MB total to snapshot. Snapshot at row P-1, not P: `tower.prefill`
pairs row p with frame p+1.

### 2. Understand why custom layers break graph capture -- ✅ DONE, see `10`

Not a capture problem at all. `Drama3DecodeGraphs.run` checks
`torch.isfinite(logits).all()` over the full 252,288-wide tensor, and the
slice fills 248,191 of those with `-inf` by construction. Padding with
`torch.finfo(dtype).min` instead fixes it. Measured: 0.95 GiB freed, cache
62 -> 70 experts/layer, RTF 0.919 -> 0.872.

⚠️ The judgement written here -- "low value today: VRAM is not the
constraint and more cache does not help a prefill-bound workload" -- was
**wrong on both halves**. VRAM is exactly the constraint on prefill, and
more cache helps precisely *because* prefill is bandwidth-bound. `embed_tokens`
is the same trick for another ~0.96 GiB and is still open; see `10` for the
shape it has to take (prefill needs the full table, so the table goes to
pinned host RAM rather than being sliced).

### 3. Do NOT finish the CPU codec offload

Tested: codec on CPU raised the cache 63 -> 96 experts/layer but made it
**50 % slower** (RTF 0.99 -> 1.49), because `decode_to_audio` is called
inline and CPU decode at 0.64x realtime lands straight in the critical path.
A background-thread version might win, but it only buys VRAM, which is not
binding, and it adds a way to acquire a new bottleneck silently.

## Open, lower value

- `bin/prebuild.py` is written and dry-runs correctly but has never been run
  for real -- needs `say -engine drama3 -chunks FILE` so the fingerprints
  carry the `drama3` prefix. See `05-say-integration.md`.
- Nobody has listened to `out/paragraph-v4/full.wav` end to end.
- A better/longer reference clip (the user mentioned sourcing one).

## Things I got wrong today, so they are not re-derived

- Residency "67 %" -> actually 41-50 % (forgot non-expert weights).
- "19.34 tok/s is the compute ceiling" -> it was the *dispatch* ceiling; the
  bandwidth roof is ~139 tok/s.
- "CUDA graphs are refused when experts are on CPU" -> false; FP8 experts are
  **buffers**, and `for_model` checks `parameters()`.
- "Pinning buys ~10 %" -> 1.47x on a clean measurement.
- "The sizing wastes 2 GiB of VRAM" -> no, the desktop and CUDA context use it.
- "lm_head slicing is bit-identical" -> only for the 1-row shape that occurs.
- "~30 % of chunks lose content" -> 2 % on real prose; the 30 % came from
  adversarially repetitive test sentences.
- "CPU codec decode will hide behind generation" -> only if pipelined, which
  it was not.
