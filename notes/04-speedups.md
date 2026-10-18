# Speedups: what is measured, and what it cost to get right

2026-10-17. Everything here is measured on this box unless marked otherwise.

## Scoreboard

| configuration | decode | long-form RTF | VRAM peak | RAM peak |
|---|---|---|---|---|
| naive gather + H2D | 6.2 tok/s | — | — | — |
| per-expert DMA, no cache | 8.07 tok/s | **3.90** | 11.53 GiB | 37.91 GiB |
| + expert cache, 80/layer | — | **2.85** | 21.25 GiB | 37.92 GiB |
| zero transfers (dispatch floor) | 19.34 tok/s | — | — | — |
| memory-bandwidth roof (arithmetic) | ~139 tok/s | ~0.16 | — | — |

**The cache is worth 27 % of compute time** (10.42 -> 7.63 min for the same
2.67 min of audio) for ~9.4 GiB of VRAM.

## Quality: unchanged, and checked two ways

The constraint was "no lower quality than it is", so speed alone is not a pass.

1. **Unit test, bit-exact.** `bin/test_cache.py` builds a small `MoEExperts`
   with random FP8 weights, runs the same routing through the stock kernel
   with all experts resident and through the cache path, and compares. At
   capacities 8, 12 and 32 of 32 experts, decode- and prefill-shaped calls:
   `max|difference| = 0.000000`. The cache is not approximately right, it is
   the same arithmetic.
2. **End-to-end, same content.** 11 paragraphs generated with and without the
   cache, every chunk transcribed: **11/11 identical** (one chunk differed
   only in where the transcriber put a comma). Durations track to within
   0.5 s per chunk. Audio is *not* bit-identical between runs and should not
   be — sampling is stochastic and the two runs drew independently.

## Hit rate: 41.8 %, not the simulated 77.9 %

Measured 830,601 hits of 1,986,204 lookups. The simulation in
`notes/02-first-results.md` said 77.9 % at a 13.3 GiB budget. Both are right;
they measure different things:

- The simulation replayed a **decode-only** trace. This run includes prefill.
- Prefill routes to 114-175 distinct experts per layer, far more than the 80
  the cache holds, so it takes the uncached fallback and **every one of those
  lookups is counted as a miss**.

⚠️ So 41.8 % is not the decode hit rate and should not be compared to 77.9 %.
The decode-only rate is not separately instrumented yet.

## VRAM is now the binding constraint

```
21.25 GiB peak of 23.52   (cache 80/layer = 9.4 GiB)
```

2.3 GiB of headroom, which is ~19 more experts per layer. Two pending changes
free more: slicing `lm_head` to the 4,097 reachable ids (~1.89 GiB,
bit-identical output) and keeping the codec off the GPU when only generating
semantic tokens (~1.74 GiB).

Host RAM is unchanged at 37.9 GiB whether the cache is on or off, as expected
— the experts live in RAM either way.

## ⚠️ How the cache was broken, and the real lesson

The first two versions produced **zero frames on every chunk**. Two wrong
diagnoses before the right one:

1. *"It evicts a slot already promised to the same step."* A real hazard, and
   the guard for it is worth keeping — but not the bug.
2. *"`non_blocking=True` from pageable memory corrupts via a recycled staging
   buffer."* The evidence looked strong (clean at 8 copies, broken at 22) and
   it is a genuine CUDA pitfall. Falsified by testing the shipped `direct`
   path, which uses the same flag and is **bit-exact up to 254 experts**.

The actual cause: **`bin/offload.py` contained two definitions of
`cached_forward`.** Repeated string-replacement patching had appended copies
rather than replacing, and Python keeps the last one — so every "fix" was
applied to a function that never ran. The live copy was the original 38-line
version with no guard and no fallback.

The lesson is about method, not about caches: *a file edited by text
substitution must be re-parsed and checked for duplicate definitions.* There
is now a check for this; the other eleven scripts were clean.

Two useful facts fell out of the wrong turns, both worth keeping:

- The `direct` path is bit-exact from 1 to 128 tokens (up to 254 distinct
  experts), so **the overnight 32-minute narration was never degraded**.
- `fp8_moe_forward` returns **identical** results given all 256 experts or a
  renumbered subset. That is the property the whole cache design rests on,
  and it is now verified rather than assumed.


## Prefill chunk size: the cheapest win of the lot

`generate_drama3` takes `prefill_chunk_size` as a keyword argument;
`DRAMA3_PREFILL_CHUNK_SIZE = 128` is only its default. With experts offloaded
that default is actively harmful: **each prefill batch fetches every distinct
expert it routes to**, and that count saturates near 256 regardless of batch
size, so small batches re-fetch the same experts over and over.

Same 10 paragraphs, cache 80/layer, only the chunk size changing:

| prefill rows | RTF | compute | cache hit | VRAM peak |
|---|---|---|---|---|
| 128 (upstream default) | 2.74 | 7.61 min | 39.4 % | 21.42 GiB |
| 512 | 2.25 | 6.09 min | 58.6 % | 21.36 GiB |
| **1024** | **2.05** | **5.58 min** | **60.8 %** | 21.35 GiB |

**27 % less compute than the default, from one keyword argument.** The hit
rate climbs with it for a mechanical reason: fewer prefill batches means fewer
calls that exceed the cache and take the uncached fallback, and every lookup
on that path is counted as a miss.

Correctness: 11/11 chunks succeeded at every setting, none truncated. The
GDN layers chunk internally over fixed 64-row blocks with a carried state and
full-attention masks are position-derived, so results are equivalent; ⚠️ SDPA
tile shapes change, so not guaranteed *bit*-identical. VRAM is flat — the
transient at 1024 rows is small next to the expert cache.

## Cumulative

| configuration | long-form RTF | compute for 2.7 min of audio |
|---|---|---|
| this morning (no cache, prefill 128) | 3.90 | 10.42 min |
| + expert cache 80/layer | 2.85 | 7.63 min |
| + prefill 1024 | **2.05** | **5.58 min** |

**47 % less compute than where the day started**, with output content
verified unchanged at each step. On the 32-minute script that is 130 minutes
down to roughly 68.


## lm_head slice

`compute_logits` runs the full [252288, 2048] output layer every frame;
`decode_step` then reads `logits[:, candidates]` -- eos plus the semantic
band, **4,097 of 252,288 ids (1.62 %)**. Slicing the weight to those rows and
scattering back into a full-width -inf buffer leaves `decode_step` untouched.

```
rows=1  identical=True   max|d|=0.000000   <-- the only shape this path uses
rows=2  identical=False  max|d|=0.000244 (1.0e-04 of peak)   informational
rows=4  identical=False  max|d|=0.007812 (3.6e-03 of peak)   informational
weight 0.96 GiB -> 0.016 GiB  (frees 0.95 GiB)
```

⚠️ **I first called this "bit-identical" without qualification, which was
wrong.** A wider GEMM reassociates differently in bf16, so rows>1 differ at
the 1e-4 level. It does not matter here only because `compute_logits` is
*always* called with exactly one row -- `generate.py:252` does
`hidden = hidden[-1:]` before the decode loop, and the CUDA-graph path uses a
`(1, hidden)` buffer (`generate.py:376`). Prefill never calls it at all. The
test now asserts on rows=1 and reports wider shapes as informational.

## Final configuration and cumulative result

```
--cache-experts 88 --prefill-chunk 2048 --slice-lm-head
```

| configuration | RTF | compute (2.7 min audio) | cache hit | VRAM peak |
|---|---|---|---|---|
| this morning | 3.90 | 10.42 min | — | 11.53 GiB |
| + expert cache 80 | 2.85 | 7.63 min | 41.8 % | 21.25 GiB |
| + prefill 1024 | 2.05 | 5.58 min | 60.8 % | 21.35 GiB |
| + lm_head slice, cache 88 | 1.90 | 5.23 min | 59.9 % | 21.30 GiB |
| + prefill 2048 | **1.88** | **5.14 min** | 65.1 % | 21.33 GiB |

**52 % less compute than this morning**, content verified unchanged at each
step. Host RAM is 37.9 GiB throughout and does not move.

⚠️ The final quality comparison rests on only **3** STT-checked chunks
(`--stt-every 4` on an 11-chunk run), which is thin. The cache's bit-exact
unit test is the stronger evidence; a full-document re-run would be better
still.

Prefill 2048 over 1024 is worth ~1 %, within noise. 1024 is the honest
recommendation; 2048 only helps because the longest prompt here is ~1300 rows
and it makes that a single batch.


## Speed vs VRAM — the curve, because the box is not headless

The user runs other work on this GPU and wants ~2 GiB left free, so "fastest"
is the wrong objective on its own. Same 10 paragraphs, prefill 1024 +
lm_head slice, only the cache changing:

| experts/layer | RTF | VRAM peak | cache hit |
|---|---|---|---|
| 0 | 2.97 | 11.16 GiB | — |
| 24 | 2.56 | 14.17 GiB | 38.0 % |
| 48 | 2.18 | 16.73 GiB | 45.5 % |
| 64 | 2.04 | 18.63 GiB | 51.9 % |
| 88 (+prefill 2048) | 1.88 | 21.33 GiB | 65.1 % |

Near-linear: roughly **0.3 RTF per 2.5 GiB**. Note the zero-cache row still
beats this morning's 3.90 *and* uses slightly less VRAM, because prefill 1024
and the lm_head slice cost nothing in memory.

`--vram-reserve N` (default 2.0) sizes the cache from VRAM actually free
after load, holding back N GiB for other programs plus 1.5 GiB for our own
activation peak (measured: 20.09 -> 21.33 GiB during generation). On an idle
card that resolves to ~87 experts/layer; if something else already holds
memory, the cache shrinks instead of the run failing.

## ⚠️ Pinning: the earlier 10 % figure was wrong

`notes/02` measured per-expert DMA at 21.9 GiB/s unpinned vs 24.3 pinned and
concluded pinning buys ~10 % and is not worth locking 30 GiB. A cleaner
measurement disagrees. One region at a time, nothing else held, four sizes:

| host region | pinned | unpinned | ratio |
|---|---|---|---|
| 1 GiB | 24.62 GiB/s | 16.63 GiB/s | 1.48x |
| 4 GiB | 24.63 | 16.65 | 1.48x |
| 16 GiB | 24.63 | 16.76 | 1.47x |
| 30 GiB | 24.63 | 16.73 | 1.47x |

**1.47x, not 1.10x.** ⚠️ The discrepancy with the earlier run is
*unexplained* — that one used a 768 MiB buffer and is not reproduced here.
The newer measurement varies one thing and agrees with itself across four
region sizes, so it is the one to trust, but both are recorded rather than
one quietly replacing the other.

Also established:

- **Pinning 30 GiB works on this box.** 63 GiB of RAM still available
  afterwards.
- **DMA rate is flat with region size** — a 30 GiB pinned region is no slower
  per expert than a 1 GiB one, so there is no penalty for pinning everything.
- **Setup cost is superlinear**: 0.5 s at 1 GiB, 1.2 s at 4, 6.7 s at 16,
  **44.6 s at 30**. Paid once at load; meaningful for a one-shot run, trivial
  for a resident server.

⚠️ Still unmeasured: the bandwidth a custom kernel gets reading
device-**mapped** host memory. That is what CUDA-graph capture would actually
use, and it needs a kernel taking a raw host pointer, which PyTorch will not
hand out. Do not assume it matches the DMA figures above.


## Pinning, measured end to end

Same 10 paragraphs, cache 64, prefill 1024, lm_head sliced:

| | RTF | VRAM peak | RAM peak | load |
|---|---|---|---|---|
| unpinned | 2.04 | 18.63 GiB | 37.92 GiB | 11.4 s |
| **pinned** | **1.72** | 18.62 GiB | 37.92 GiB | 26.9 s |

**16 % faster at identical VRAM and identical RAM.** Pinning does not add
memory -- the experts were already resident; it only locks them. Load grows
by ~15 s (pinning 30 GiB took 14 s, not the 45 s the synthetic benchmark
suggested: `pin_memory()` copies existing pages, while the benchmark
allocated and faulted fresh ones).

It also beats buying more cache: pinned at 64/layer (1.72, 18.62 GiB) is
faster than unpinned at 88/layer (1.88, 21.33 GiB) while using 2.7 GiB less.

Quality unchanged: 3/3 STT-checked chunks identical to the unpinned run, as
expected since pinning changes where memory lives, not what is in it.

## Recommended configuration

```
--vram-reserve 2 --prefill-chunk 1024 --slice-lm-head --pin-experts
```

```
RTF 1.69    69 experts/layer, 53.9 % hit
VRAM 19.19 GiB peak      RAM 37.91 GiB peak      load 25 s
```

| | RTF | compute for 2.7 min of audio |
|---|---|---|
| start of 2026-10-17 | 3.90 | 10.42 min |
| end of 2026-10-17 | **1.69** | **4.59 min** |

**57 % less compute.** On the 32-minute script that is ~130 min -> ~55 min.

## ⚠️ A wrong diagnosis worth recording

I claimed the sizing was "under-counting free VRAM by ~2 GiB because the
caching allocator holds unused blocks", and added `torch.cuda.empty_cache()`
before the measurement. **It changed nothing** -- still 69 experts/layer.

The 2.11 GiB gap (`23.52 total - 11.64 free - 9.77 ours`) is the **display
server plus the CUDA context**; `nvidia-smi` shows 1.2-1.6 GiB used with
nothing of ours running. `mem_get_info` already excludes memory other
processes hold, so the reserve was being honoured the whole time. My "4.3 GiB
free at peak" figure came from subtracting only our own peak from the card
total and forgetting everything else on the GPU.

The `empty_cache()` call stays -- it is correct hygiene before measuring --
but it fixed nothing, because nothing was broken.


## Negative result: per-layer cache budgets are not worth it

The traces show early layers touching ~236 distinct experts and deep layers
~159, which suggested a flat N-per-layer cache wastes slots. Tested offline
against the census by greedily assigning each additional slot to whichever
layer gains the most calls:

| slots/layer | flat coverage | greedy coverage | gain |
|---|---|---|---|
| 69 | 58.55 % | 58.66 % | +0.11 |
| 128 | 81.16 % | 81.57 % | +0.42 |
| 160 | 89.43 % | 90.22 % | +0.79 |

**0.1-0.8 percentage points.** Not worth the complexity. Flat allocation
stands.

The reasoning behind the hypothesis was also wrong, which is the more useful
part: I expected shallow layers to need *more* slots because they touch more
distinct experts. The optimiser gives them **fewer** (49-59 at a 69 budget)
and favours the middle layers (up to 87). Shallow layers route close to
uniformly, so the marginal slot there buys almost nothing; the skewed middle
layers are where a slot pays.

Cost of finding this out: a few minutes of CPU against data already on disk,
versus implementing per-layer budgets and discovering it in a GPU run.


## Full-document re-run, everything applied

`out/paragraph-v3/full.wav`, the same 32-minute SQL script as the overnight
run, with bullet-list line splitting, duration retry, silence trimming,
punctuation-aware pauses, expert cache, prefill 1024, lm_head slice and
pinned experts:

| | overnight | final |
|---|---|---|
| chunks | 192 | 235 |
| failed | 0 | 0 |
| **wrong after retries** | **4** | **0** |
| truncated mid-word | 2 | 1 |
| first draw accepted | 97 % | 99 % |
| audio | 32.3 min | 31.6 min |
| **compute** | **130.0 min** | **56.8 min** |
| **RTF** | **4.03** | **1.80** |
| VRAM peak | — | 19.57 GiB (2.0 GiB left free) |
| cache hit | — | 54.6 % |

The chunk count rose 192 -> 235 because splitting bullet lists into separate
generations is what stops them losing content. Verified directly on the three
lines that vanished overnight:

```
chunk 57 -> "Any potential migration blockers."
chunk 58 -> "Any breaking changes that require post-migration fixes."
chunk 59 -> "Azure features that the workloads can use."
```

Overnight, all three collapsed into the first one. 24 STT spot checks across
the run matched their source.

⚠️ One chunk stayed merged on purpose: chunk 8 is "Explain what data platform
modernization is. Review the stages of a data migration. Recommend..." --
those lines *do* end in terminal punctuation, so the splitting rule leaves
them joined. It generated cleanly (10.4 s, accepted first draw), which is
evidence the rule draws the line in the right place rather than splitting
everything.


## Reference clip: length and cut point (user-judged)

Every full run up to 2026-10-17 used `--ref-seconds 20`, a default set early
for a quick test and never revisited. It cuts **mid-word** at both ends of a
56 s clip (measured RMS 0.149 at the cut, i.e. full speech level), and the
reference transcript is ASR of that truncated audio -- so the last word the
model is told the reference says is probably wrong.

Same text generated three ways, user listened:

| reference | codes | RTF | verdict |
|---|---|---|---|
| 20.0 s (mid-word) | 431 | 1.02 | "bad" |
| **28.2 s** (ends on a sentence) | 608 | **1.00** | "lightly better, no longer a different voice for the most part" |
| 56.0 s (whole clip) | 1206 | 1.10 | "better yet but not much... still a bit odd, probably the model" |

**28.2 s is free** -- same RTF as 20 s despite 177 more prompt rows, because
the extra prefill disappears into the cache. 56 s costs ~10 %, which is a
1,206-row reference re-prefilled on every chunk, for a gain the user called
marginal.

### Implemented as a rule, not a constant

`snap_to_silence()` cuts at the last >=0.3 s silence at or before the
requested duration. Default is now `--ref-seconds 30`, which lands at 28.5 s
on this clip. Any new clip gets the same treatment:

```
--ref-seconds 20 -> 17.7s      --ref-seconds 45 -> 41.9s
--ref-seconds 30 -> 28.5s      --ref-seconds 56 -> 51.7s
```

The 56 s case snapping *down* to 51.7 s is correct -- the clip's last words
run to the end with no trailing silence, so a literal cut would chop a word.

⚠️ In this recording every gap >= 0.3 s was also a sentence boundary, so
snapping to silence snapped to punctuation for free. That is a property of
clean read speech and should not be assumed for other material.

## ⚠️ Codec on CPU: tested, and it loses

Moving the codec off the GPU frees 1.74 GiB and raised the cache from 63 to
96 experts/layer -- but it made things **50 % slower**:

| | cache | RTF |
|---|---|---|
| codec on GPU | 63/layer | **0.99** |
| codec on CPU | 96/layer | 1.49 |

The reasoning was wrong, not the measurement. I argued that CPU decode at
0.64x realtime would "hide behind" generation at RTF ~1.0 -- and then tested
an implementation with **no pipelining in it**. `decode_to_audio` is called
inline after generation, so moving it to the CPU simply inserts 0.64x of
serial work into the critical path, and 33 extra cached experts do not pay
for that.

A real background-thread implementation might win (torch releases the GIL
during the convolutions). Not built, because VRAM is not currently the
constraint -- 19.5 of 23.5 GiB -- and the thing limiting the full document is
prefill on short chunks, which more cache does not help.
