# Prefill is PCIe-bound, and the gather ignores the cache it already has

Measured 2026-10-18 with `bin/narrate.py --prefill-probe`, 8 real chunks of
the production script in the production configuration. Analysis:
`bin/prefill_report.py`. Task #25.

## The measurement

```
320 layer-batches over 8 chunks

1 WHERE THE TIME GOES
  gather       7.62 s   96.5%
  compute      0.28 s    3.5%

2 IS IT BANDWIDTH?
  experts fetched      62,458 x 3.001 MiB = 183.0 GiB
  -> gather is running at 25.8 GB/s, at or above the pinned PCIe roof

3 HOW MUCH IS ALREADY IN VRAM
  device cache holds up to 62 experts/layer
  of 62,458 experts the gather fetched, 17,578 (28.1%) were already resident
  a PERFECT policy would reach 31.8%, against 28.1% now
  -> policy is worth 3.6 points; the rest is capacity

4 DO CONSECUTIVE CHUNKS WANT THE SAME EXPERTS?
  overlap 83.1% on average
  but one batch already touches 76% of all 256, so 76% is the chance floor
```

## What it means

**Prefill is 96.5 % weight transfer running at the hardware roof.** There is
nothing to win from better kernels, fewer launches or more overlap — the
GEMM it is feeding is 3.5 % of the time, so even perfect overlap buys 3.5 %.
The only lever is moving fewer bytes.

⚠️ **This is the opposite of decode, and the project's headline finding does
not transfer.** `02`/`06` established that the wall was *dispatch*, not
bandwidth — ~86 % of a decode step was the GPU idle waiting for kernel
launches, and CUDA graphs took RTF 1.71 to 0.97. That is true of decode,
where a step touches 8 experts per layer and the device cache serves most of
them. A prefill batch touches **~196 of 256 experts per layer** and moves
**23 GiB per chunk**. Bring the decode intuition here and you will optimise
launches that are not the problem.

**The gather re-fetches 28 % of what is already in VRAM.**
`offload.offloaded_forward` is the prefill path and it does not consult the
device cache at all — `devcache.dev_cached_forward` hands prefill straight
back to it (`devcache.py:258`, guarded on `topk_ids.shape[0] != 1`). So its
hit rate is zero *by construction*, not because the experts are absent. 62
experts per layer are sitting in VRAM and get pulled over PCIe again anyway.

**Policy is nearly worthless here; only capacity matters.** This is the
non-obvious one and it is why the 83.1 % cross-chunk overlap in #4 is a trap.
A batch wants 76 % of all 256 experts, so *almost any* cached expert is a
wanted one — the cache is already converting 55 of its 62 slots into hits.
An oracle that knew the future would reach 31.8 % against the 28.1 % a
decode-driven LRU achieves. **3.6 points.** Two batches overlapping by 83 %
when each covers 76 % of the space is arithmetic, not a reusable working set.

## Results

All arms 20 chunks of the real script, same configuration otherwise, RTF
computed the same way (sum of gen seconds / sum of audio seconds):

```
arm              n  audio s    gen s    RTF  prefill s  cache  gather  VRAMpk
baseline        20    244.6    248.0  1.014      39.42     62       -   19.67
+prefix-kv      20    234.7    222.7  0.949      22.41     62       -   19.76
+gather         20    240.8    221.3  0.919      17.74     62   26.6%   19.75
+lm_head        20    243.1    212.1  0.872      17.27     70   29.9%   19.81
+embed          20    237.7    207.2  0.872      17.05     78   33.4%   19.76
reserve 3.9     20    234.9    210.1  0.894      17.60     61   26.2%   17.82
```

**RTF 1.014 -> 0.872. Prefill 39.42 s -> 17.05 s, from 19.1 % of runtime to
8.2 %.** One retry across 120 chunk-generations.

The last row is not a step in the chain -- it is the same configuration as
`+embed` with the freed VRAM handed back to the desktop instead of to the
cache. See "what the freed VRAM is actually worth".

`gather off` (0.950) reproduces `prefix on` (0.949) in a separate process,
which is the cross-check that these differences are real and not run-to-run
noise.

Per change:

| | prefill | RTF | what did it |
|---|---|---|---|
| `--prefix-kv` | 39.42 -> 22.41 s | 1.014 -> 0.949 | stop re-reading the reference |
| hybrid gather | 22.41 -> 17.74 s | 0.950 -> 0.919 | stop re-fetching resident experts |
| `--slice-lm-head` | 17.74 -> 17.27 s | 0.919 -> 0.872 | mostly **decode**, see below |

Hybrid gather hit 26.6 % against the 28.1 % the probe predicted, and took
20.7 % off prefill against ~26 % predicted. The gap is the per-layer
`slot_of` readback and the device-to-device copies not being free. 138 GiB
per 20 chunks stopped crossing the bus.

⚠️ **`--slice-lm-head` is a decode win, not a prefill win, and the numbers
say so.** It took the prefill hit rate from 26.6 % to 29.9 % (the freed
0.95 GiB bought 62 -> 70 experts/layer) but prefill only moved 17.74 ->
17.27 s, 2.6 %. RTF moved 5.1 %. The rest is what the slice was always for:
`compute_logits` no longer reads a [252288, 2048] matrix out of VRAM for
every frame. Do not credit it to the cache.

## Decided: hybrid gather

For each distinct expert a prefill batch needs, copy from the device cache
(device-to-device, ~1.5 TB/s) when it is resident and over PCIe only when it
is not.

- **Ceiling: 2.15 s of 7.62 s of gather, ~28 %.** Over a full run that is
  ~0.31 s per chunk, ~2.8 % of total runtime.
- Costs no VRAM, changes no arithmetic, needs no policy work.
- The slot lookup needs a host sync, but `offloaded_forward` **already** does
  one (`torch.unique(...).cpu()` for the id remap), so the lookup rides along
  on a sync that is already paid.

**Read-only on purpose.** Prefill must not insert into the cache or bump LRU
stamps. It would evict decode's working set — 200 experts a layer against a
62-slot cache — and decode is 80 % of runtime. Prefill borrows what happens
to be there and leaves the ordering alone.

## Beyond that, it is a VRAM problem

Each +10 experts/layer is +1.17 GiB of VRAM and roughly +5 points of prefill
hit rate. Current state: 62/layer, 7.27 GiB, 2.09 GiB free at peak against a
2 GiB user reserve. So there is no slack.

`--slice-lm-head` (task #19) is now ✅ fixed and gave 0.95 GiB → 62 to 70
experts/layer → prefill hit rate 26.6 % to 29.9 %.

### Why it NaN'd under CUDA graphs, and the one-value fix

`Drama3DecodeGraphs.run` (`generate.py:376-377`) does

```python
logits = backbone.compute_logits(self.hidden)
self.finite.copy_(torch.isfinite(logits).all())
```

over the **full 252,288-wide** tensor. The slice scatters 4,097 real values
into a buffer filled with `-inf`, so 248,191 entries were non-finite *by
construction*, the check was False on the first step and every step after,
and every chunk died with "non-finite logits; refusing to sample invalid
audio". **The slice was never producing bad numbers. It was tripping a guard
with its own sentinel.**

Eager decode did not care because the guard that runs there lives in
`sample_top_k_top_p` and only ever sees `logits[:, candidates]`
(`tower.py:349-353`) — the 4,097 real values, all finite. That asymmetry is
why this read as a CUDA-graph bug for a day.

Fix: pad with `torch.finfo(dtype).min` instead of `-inf`. Softmaxes to
exactly 0 the same way, nothing reads those positions except the check, and
a genuine Inf/NaN out of the GEMM still trips it.

⚠️ The earlier attempted fix — pre-allocating the wide buffer instead of
allocating one per call — was addressing a real and separate hazard
(allocation inside a captured region) and is kept. It was not the cause,
which is why it did not help on its own. Two bugs, one symptom.

### `embed_tokens` — ✅ built, and it bought nothing directly

`--offload-embed`. Full [252288, 2048] table in host RAM, the 4,097 rows
decode can reach resident in VRAM (16.0 MiB) plus a 1.9 MiB id→row index.
**Freed 0.94 GiB.** `bin/test_embed_offload.py` checks the embeddings are
bit-identical on decode ids, prefill ids and a mix.

It delivered everything it promised and moved RTF **not at all**: cache
70 → 78 experts/layer, prefill gather 29.9 % → 33.4 %, prefill 17.27 →
17.05 s over 20 chunks, RTF 0.872 → 0.872. It also costs 1 GiB of host RAM
(37.9 → 38.9 GiB).

Two traps it had to avoid, both worth more than the change itself:

- **The branch is `torch.cuda.is_current_stream_capturing()`, not row
  count.** "One row means decode" is wrong: a prefill whose length is
  1 mod `--prefill-chunk` ends with a single-row batch carrying an arbitrary
  text id, which would read row −1 of the resident slice and return a
  plausible embedding for the wrong word. `test_embed_offload.py` exercises
  that case explicitly (id 248045).
- **The host table is held in a closure, not left as the module's
  Parameter.** `Drama3DecodeGraphs.for_model` (`generate.py:350-353`)
  disables CUDA graphs if `{p.device for p in backbone.parameters()}` is not
  a single CUDA device. One CPU parameter would silently cost RTF
  1.71 → 0.97 — twenty times what this saves. The experts get away with the
  host because they are **buffers**; an embedding weight is a Parameter.
  `offload_embed` raises if any parameter ends up off-device.

### What the freed VRAM is actually worth

`--slice-lm-head` and `--offload-embed` together free **1.89 GiB**,
losslessly. That gigabyte and a bit can go to the expert cache or back to
the desktop, and the exchange rate is now measured — same features, only
`--vram-reserve` differs:

```
arm              n  audio s    gen s    RTF  prefill s  cache  gather  VRAMpk
+embed          20    237.7    207.2  0.872      17.05     78   33.4%   19.76
reserve 3.9     20    234.9    210.1  0.894      17.60     61   26.2%   17.82
```

**1.94 GiB of VRAM peak costs 2.5 % of RTF.** 61 against 78 experts/layer,
everything else identical. The user's standing preference is a 2 GiB
reserve on a box that is not headless, so the working config keeps
`--vram-reserve 2` and spends the freed space on experts; `--vram-reserve
3.9` is there when the desktop needs it more than the clock does.

⚠️ I said earlier in this file that cache capacity had "hit diminishing
returns", on the strength of 70 → 78 experts/layer showing 0.872 → 0.872.
**That comparison was confounded** — those two arms differ in whether
`--offload-embed` is on *and* in how much audio they generated. The clean
A/B is the one above, and it says capacity is still worth ~0.15 % of RTF per
expert per layer over this range. Do not plan on the basis of the sentence I
had to delete.

Note also that peak VRAM barely moved across the whole day (19.67 GiB at
baseline, 19.76 now): the lossless savings did not *reduce* usage, they
moved 1.89 GiB out of weights and into cache.

⚠️ `08` ranked #19 "low value today: VRAM is not the constraint". **That
ranking was wrong** — VRAM is exactly the constraint on the largest
remaining cost.

## Rejected

- **More overlap / fewer launches.** Compute is 3.5 %. Ceiling is 3.5 %.
- **A smarter eviction policy for prefill.** 3.6 points, measured, above.
- **Bigger prefill batches.** Already helps and is already done
  (`--prefill-chunk 1024`); with `--prefix-kv` every observed suffix is a
  single batch, so there is nothing left to merge. It would still help a run
  *without* `--prefix-kv`, where an 1,180-row prompt becomes two batches and
  costs 2.42 s against 1.35 s for a 745-row one.
- **Extending the prefix over the pinned chunk-0 turn.** Measured at ~60
  rows (chunk 0 is the document title). Against a 453-row suffix that is 13 %
  of rows and, because distinct-expert count saturates, far less than 13 % of
  time. The `recent` history turns are the bulk of the suffix and they
  *slide*, so their KV is not positionally reusable without redesigning the
  prompt layout.

## A bug this found

`narrate.py` printed `!! a cache was requested but none was allocated` on
**every** `--dev-cache` run, including ones where a 7.27 GiB cache was
working perfectly. The check read `_cap`, the python cache's capacity, which
`--dev-cache` deliberately zeroes when it installs `_dc_cap`. Fixed to check
both, and to report the device cache's capacity rather than a hit rate it
does not collect on the host.

`bin/prefill_report.py` had the mirror-image bug: it read `cache_size` from
the first record, taken while the cache was still filling, and reported "12
experts/layer" against a measured 55 resident. Impossible on its face, which
is how it was caught. Now reads the high-water mark.
