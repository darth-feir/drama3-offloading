# First results — it runs, and the predicted failure mode did not happen

2026-10-17. Everything here is **measured on the box**. Reproduce with
`bin/run.sh bin/offload.py --checkpoint "$DRAMA3_CKPT" ...` and
`bin/simulate_cache.py`. Read `notes/01-static-findings.md` first.

## Headline

Drama 3 runs on the 4090 with every routed expert resident in CPU RAM.

| configuration | decode tok/s |
|---|---|
| naive gather + H2D (first working version) | 6.2 |
| **per-expert DMA, no cache at all** | **8.07** |
| compute floor — zero transfers, eager (ceiling for *any* cache) | **19.34** |

**It is not PCIe-bound.** That was the predicted failure mode and it did not
occur. The bus does 21.5–24.9 GiB/s; the thing in the way is the CPU-side
gather and, once that is gone, eager-mode compute.

## It loads, and the static accounting was right

```
FishOmni experts: block-FP8 W8A8
FishOmni backbone: loaded 62053 parameter slots      (62,245 - 192 tower = 62,053)
FishOmni tower: loaded 192 tensors on cuda:0
loaded in 10.4s
VRAM after load: 6.87 GiB allocated, 7.42 GiB peak
```

- **6.87 GiB allocated** is exactly the non-expert figure predicted from the
  safetensors headers. The budget table in `01-static-findings.md` holds.
- **7.42 GiB peak** confirms the per-layer materialize trick: the transient is
  one layer's experts (~0.55 GiB), not all forty. Peak never approached 24 GiB.
- Experts stayed **FP8** (`block-FP8 W8A8`), so the 30 GiB did not become 60.
- 10.4 s load is with the page cache hot from the download. ⚠️ A cold load will
  be disk-bound and much slower; unmeasured.

This leaves **~16 GiB of VRAM headroom** for an expert cache, slightly more
than the 13.3 GiB the static estimate assumed, because nothing loads
`codec.pth` on this path.

## Routing is strongly skewed — open question #1, answered

211 decode frames, `notes/route-trace-512.json`. Uniform over 256 experts
would be 8.000 bits; **measured mean entropy is 6.737 bits**.

| layer | experts seen | entropy | gini | top-8 share |
|---|---|---|---|---|
| 0 | 236 | 7.640 | 0.319 | 9.2 % |
| 2 | 243 | 7.706 | 0.306 | 9.2 % |
| 5 | 227 | 7.534 | 0.351 | 10.5 % |
| 37 | 126 | 6.137 | 0.580 | 29.3 % |
| 38 | 134 | 5.916 | 0.635 | 36.0 % |
| 39 | 159 | 6.174 | 0.656 | 28.4 % |

**The skew has structure: it grows with depth.** Early layers are nearly
uniform (gini 0.31, 236 of 256 experts touched); the last few concentrate hard
(gini 0.66, top 8 experts take 28–36 % of all traffic). A uniform per-layer
cache budget is therefore the wrong default — deep layers reach a high hit rate
cheaply and shallow layers never will. ⚠️ Unmeasured: whether reallocating
budget toward shallow layers beats a flat split. The trace is on disk, so this
costs nothing but CPU to answer.

### Hit rate by budget (per-layer LRU, 211-frame decode trace)

| expert budget | experts/layer | LRU | LFU | static (oracle) |
|---|---|---|---|---|
| 10.0 GiB | 85 | 70.6 % | 70.2 % | 81.0 % |
| 13.3 GiB | 113 | **77.9 %** | 78.4 % | 88.8 % |
| 15.0 GiB | 127 | 80.7 % | 81.1 % | 91.5 % |
| 20.0 GiB | 170 | 86.4 % | 86.6 % | 97.3 % |

LRU and LFU are within noise of each other. "static" peeks at the whole trace
and is a ceiling, not an implementation.

⚠️ **A 64-frame trace gave 70.3 % at 13.3 GiB, not 77.9 %** — short traces are
dominated by cold-start misses and *understate* the hit rate. Do not quote
numbers from a short run. Both traces are kept.

## Where the time actually goes (`bin/bench_copy.py`)

One layer = 8 experts = 24.0 MiB. A 40-layer token moves 0.938 GiB.

| step | ms | GiB/s | implied tok/s cap |
|---|---|---|---|
| CPU gather only, no transfer | 1.43 | 16.3 | 17.4 |
| gather + unpinned H2D *(first version)* | 2.74 | 8.6 | 9.1 |
| H2D only, unpinned | 1.09 | 21.5 | 22.9 |
| H2D only, pinned | 0.94 | 24.9 | 26.6 |
| **per-expert DMA, unpinned** *(the fix)* | **1.07** | **21.9** | 23.3 |
| per-expert DMA, pinned | 0.97 | 24.3 | 25.9 |

Two conclusions, both load-bearing:

**1. The gather was the problem, and it was unnecessary.** `select_fp8` runs an
advanced-index copy that materialises a `[8, 1024, 2048]` tensor in CPU RAM
*before* any transfer — 1.43 ms, more than the 1.09 ms transfer it feeds, and
the two do not overlap. But in `w13 [E, 2I, H]` **each expert is already
contiguous**, so a miss can DMA `w13[e]` straight into a GPU row. Removing the
gather took the copy path 2.74 ms → 1.07 ms and decode **6.2 → 8.07 tok/s**,
with no cache involved. Prefill went 3.68 s → 1.70 s.

**2. Pinning is not worth 30 GiB of locked memory.** Unpinned per-expert DMA
reaches 21.9 GiB/s against the pinned ceiling of 24.3 — **95 %** of the benefit
for none of the risk. Pin a small staging area if anything; do not pin the
model. (The brief's plan to pin the whole thing in RAM buys ~10 %.)

## What a cache is actually worth

Per token: compute floor 51.7 ms; no-cache transfer adds 72.2 ms (measured as
123.9 − 51.7, higher than the 42.8 ms the microbenchmark predicts because of
the per-layer `.cpu()` sync and allocator traffic).

```
hit rate      ms/token              tok/s
  0 %         51.7 + 72.2 = 123.9    8.07   (measured)
 78 %         51.7 + 15.9 =  67.6   ~14.8   (projected, 13.3 GiB LRU)
 86 %         51.7 + 10.1 =  61.8   ~16.2   (projected, 20 GiB LRU)
100 %         51.7                   19.34  (measured, --copy-mode none)
```

⚠️ The 78 %/86 % rows are **projections**, assuming a miss costs what it costs
today and a hit costs nothing. A real cache adds bookkeeping. Treat ~15 tok/s
as the target to beat, not a result.

So a cache is worth roughly **+80 %** over the current 8.07 tok/s — real, but
bounded by a 19.34 tok/s ceiling that has nothing to do with experts.

## The ceiling is eager mode, not the bus

19.34 tok/s with **zero** transfers is the whole budget any cache plays inside,
and it is low because offloading disables CUDA graphs (`generate.py:350-353`,
see `01-static-findings.md`). Eager decode is launch-bound by its own
docstring. ⚠️ Unmeasured: how much of the 51.7 ms is launch overhead versus
real matmul. That is now the most valuable unknown — it caps everything.

Measuring it needs a graph-captured run, which needs a fully resident model,
which does not fit. A smaller MoE, or capturing only the non-expert part of
the step, would isolate it.

## RTF — now measured, and an earlier figure here was wrong

**The codec runs at 21.53 semantic frames per second**, verified two ways on
real output: a 20.00 s reference encoded to 431 codes (21.55 Hz), and 183
generated frames decoded to 8.50 s of audio (21.53 Hz). So **RTF 1.0 = 21.53
tok/s**.

⚠️ **Correction.** An earlier draft of this file said the frame rate was ~86 Hz
and the RTF ~4.5. That was wrong: it took the encoder's `encoder_rates
[2,4,8,8]` = 512x and stopped there, missing that the quantizer applies a
further `downsample_factor: [2, 2]` = 4x. 44100 / 512 / 4 = 21.53.

It also means the brief's "21.5 tokens/sec" was **the realtime frame rate all
along**, not an assumed throughput as this file first claimed. The brief was
right and the correction was mine to make.

| configuration | tok/s | RTF |
|---|---|---|
| per-expert DMA, no cache (decode only) | 8.07 | **2.68** |
| same, including a 564-row prefill | 5.83 | 3.69 |
| projected with a 78 % LRU cache | ~15 | ~1.43 |
| compute floor, zero transfers | 19.34 | **1.11** |

So the ceiling is *just* above realtime, and a working cache lands around 1.4x
slower than realtime. That is far closer than "it will not be realtime"
suggested -- though still the wrong tool for the `say` job, which dots.tts
already does at RTF 0.28.

## Correctness: the output is right, not just fast

`bin/speak.py` generates a wav using the clip in `source clips/` as a reference
voice (transcribed by the resident Parakeet daemon). Round-tripping the
generated audio back through STT returns the input sentence verbatim.

This matters more than any throughput number: the offload path rewires how
expert weights reach the GPU, and a subtly wrong gather would produce fluent
garbage. It does not. ⚠️ Still unjudged by a human: timbre, prosody, and how
well the reference voice is actually cloned. STT only proves the words.

## Status of the four open questions

1. **What does the router pick?** — **answered.** Skewed, 6.737 bits vs 8.000
   uniform, and the skew deepens with layer index. LRU reaches 77.9 % at the
   real budget. The plan is worth *more* than near-uniform routing would imply.
2. **Does the loader OOM first?** — **answered** (statically, then confirmed:
   6.87 GiB allocated, 7.42 GiB peak). Hook, not fork.
3. **Hit rate of a per-layer LRU?** — **answered: 77.9 % at 13.3 GiB**,
   86.4 % at 20 GiB. Not yet implemented, only simulated.
4. **Can copies overlap compute?** — still open, and now lower priority: with
   the gather removed, transfers are only ~58 % of the step, and overlapping
   them perfectly buys at most what the cache already buys.

## What to do next, in order

1. **Implement the LRU cache** against the 16 GiB of real headroom. Expected
   ~15 tok/s. The trace makes this verifiable offline before it is written.
2. **Measure the eager-mode launch overhead.** It is the 19.34 tok/s ceiling
   and nothing else matters above it.
3. Settle RTF by decoding one generation to a wav.
4. Only then: non-uniform per-layer budgets, and stream overlap.
