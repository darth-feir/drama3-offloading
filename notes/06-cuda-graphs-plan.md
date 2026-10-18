# CUDA graphs: DONE. Drama 3 runs at RTF 0.97 on one 4090.

2026-10-17. Originally a design note with no measurements. **Now measured.**

## The measurement

Four configurations, same prompt, **exactly 300 decode steps per chunk x 4
chunks** so both arms do identical work (`--ignore-eos`), with a sync-free
stub MoE (`--copy-mode graphstub`) because the real path cannot be captured:

| CUDA graphs | lm_head slice | result | RTF |
|---|---|---|---|
| off | yes | 4/4 ok | 1.03 |
| on | yes | **4/4 non-finite logits** | — |
| off | no | 4/4 ok | 1.08 |
| **on** | **no** | **4/4 ok** | **0.46** |

**2.35x** (1.08 -> 0.46), against the 2.36x reported in the literature for a
comparable 4090 / 8-of-256 FP8 configuration. Capture itself takes 1.4-1.5 s
for all four KV buckets.

Two things this settles:

1. **The dispatch-bound thesis is correct and is now our own measurement.**
   The eager stub at RTF 1.03-1.08 also independently reproduces the 19.34
   tok/s (RTF 1.11) dispatch floor measured separately this morning.
2. **`for_model` really does offer graphs with experts on the CPU**, as the
   correction below predicted. Capture succeeded.

### ⚠️ What this does NOT show

The MoE is **synthetic** — junk resident weights and `topk_ids % 8`. It
measures the dispatch overhead in attention, the 13-layer tower and the 10
sampling passes, which is real and is where the overhead lives. It does
**not** include any expert transfer, so RTF 0.46 is not a projection of the
finished system.

Applying it with Amdahl to the real configuration: measured RTF 1.69 total,
of which the compute/dispatch part is ~1.05 (the eager stub) and transfer is
therefore ~0.64. Graphs take the first term to ~0.45, giving **~1.1 overall**
— a large gain, and *borderline* realtime rather than comfortably past it.
Whether it goes below 1.0 depends on how well a capturable miss path
overlaps, which is unmeasured.

### ⚠️ The lm_head slice is incompatible with capture

`slice_lm_head` installs a Python closure that allocates a fresh
`torch.full(..., -inf)` buffer on **every call**, inside the captured region.
With graphs on, every chunk died with "non-finite logits"; without the slice,
the identical configuration worked. The slice is worth 0.95 GiB and ~5 %;
graphs are worth 2.35x, so graphs win.

**I tried that fix and it did not work.** Pre-allocating one `wide_buf`
outside the closure and doing `fill_` + index-assign in place — no allocation
during replay — still produced non-finite logits on all 4 chunks. So the
per-call allocation was not the cause, and my diagnosis was wrong.

⚠️ **Cause unresolved.** Candidates not yet eliminated: returning the same
buffer across replays while the consumer holds a reference; the `-inf` fill
interacting with capture; or something about replacing `backbone.lm_head`
with a different module after `Drama3DecodeGraphs` has been constructed.

**Decision: keep graphs, drop the slice.** Graphs are 2.35x; the slice is
~5 % and 0.95 GiB. They do not compose and the trade is not close. Chasing a
third theory was not worth the GPU time against a result the data already
decided.

## Reproducibility

The whole matrix was run twice, before and after the attempted fix:

| config | run 1 | run 2 |
|---|---|---|
| eager, slice | 1.03 | 1.02 |
| graphs, slice | NaN | NaN |
| eager, no slice | 1.08 | 1.08 |
| **graphs, no slice** | **0.46** | **0.46** |

Identical to the hundredth. The 2.35x is not a fluke of one run.

## Why it is the biggest remaining item

`notes/02` called 19.34 tok/s (zero transfers) a "compute floor". It is a
**dispatch floor**. Summing the weight bytes the GPU must actually read per
decode frame against the 4090's ~939 GiB/s gives ~7.2 ms/frame ≈ **139 tok/s**.
The measured 51.7 ms/frame is ~14 % of that, so **~86 % of a step is the GPU
idling between kernel launches**, not computing.

Published measurement on a near-identical configuration (Qwen3-30B-A3B-FP8 and
a 35B-A3B 8-of-256 model, RTX 4090, PCIe 4.0): eager 52.1 tok/s, CUDA graphs
**122.7 tok/s — 2.36x**. Also measured there: a configuration with *one host
sync per layer* gets only 74.9, i.e. **a single sync per layer throws away
60 % of the benefit**. We currently do exactly that sync.

⚠️ That is not our number. Ours is unmeasured, and getting it is task #14.

## Correction carried over

`notes/01` claimed offloading disables graphs because `for_model` sees the
model spanning devices. **False** — the FP8 path registers experts as
`register_buffer`, and `for_model` iterates `parameters()`, which excludes
buffers. Verified: 160 expert buffers, 0 expert parameters. Graphs are
disabled only by our own `FISH_DRAMA3_CUDA_GRAPH=0`.

## The design (user's framing, which is the standard solution)

The user asked, unprompted: *"can we have it so it always sets the ram agents
in the same way and it calls that address and it just gets a translation layer
from gpu to ram? same call to same address just with a modifier to place it to
the correct offset address?"*

That is exactly right, and the distinction that makes it work is **memcpy
versus kernel**:

- A captured **memcpy node** freezes its source pointer at capture time. There
  is no arithmetic in it. On replay it re-reads the same address forever —
  which would produce fluent, confident, *wrong* audio. This is the single
  most dangerous failure mode in the whole design.
- A captured **kernel** is code. It can read an expert index out of device
  memory and compute `base + index * EXPERT_BYTES` **at replay time**. The
  base is baked, which is fine because it never moves; the offset is live.

So: one stable base address, a modifier computed per replay. Same call, right
expert.

### What has to change

1. **One contiguous pinned, device-mapped host allocation** for the experts
   (`cudaHostAllocMapped`), replacing the 40 separate per-layer CPU tensors.
   A single global slot pool keeps the kernel's base address fixed.
2. **The expert->slot table moves to device memory**, updated by a small
   kernel. Today `cached_forward` keeps it in a python `OrderedDict`, which
   cannot be captured.
3. **Delete the per-layer host sync.** `topk_ids.reshape(-1).cpu()`
   (`offload.py`) is illegal during capture *and* is costing us now.
4. **Eviction must protect slots in flight.** A GEMM reading a slot while
   another expert is written into it is a silent corruption. The published
   design uses an explicit per-slot state machine
   (`Empty / Fetch / Compute / Used`) with the middle two eviction-protected.

## ⚠️ Pinning: a judgement that needs reversing

`notes/02` rejected pinning on the grounds that unpinned per-expert DMA
reaches 21.9 GiB/s against a pinned 24.3 — "95 % of the benefit for none of
the risk of locking 30 GiB". That was **measured correctly and concluded
wrongly for the new purpose**: pinning here is not a bandwidth tweak, it is
the *precondition* for graph capture, and for any genuinely asynchronous copy
(a `.copy_(non_blocking=True)` from pageable memory is not actually async —
it stages through a buffer).

30 GiB of 125 GiB, with no swap in the path, is cheap. The earlier conclusion
stands only for the question it was asked.

## Measure these first, before writing any of it

Cheap, no model needed:

1. **Does pinning 30 GiB work comfortably on this box**, and how long does it
   take to set up?
2. **What bandwidth does a kernel reading device-mapped host memory get**,
   versus the 21.9 GiB/s an explicit DMA achieves? If mapped reads are slow,
   the design changes shape entirely.
3. **The graph-vs-eager ratio on this model**, with a sync-free stub MoE and
   `--copy-mode none`. One number that decides whether any of the above is
   worth a week.

If (3) comes back near 1.0 rather than near 2.36, the dispatch-bound thesis is
wrong and everything in this file should be re-ranked below the cheap
op-removals in `notes/04`.


---

# Stage 0 MEASURED: a kernel reads pinned host memory at full speed

2026-10-17, `bin/bench_hostread.py`, 45 s wall. The question was whether a
CUDA-graph-capturable **kernel** can read pinned host memory, and at what
bandwidth. Answer: **yes, with no penalty at all.**

| mechanism | GiB/s @8 | @64 | bytes | capture | replay |
|---|---|---|---|---|---|
| `copy_` (copy engine, memcpy node) | 24.60 | 24.62 | ok | ok | **FROZEN** |
| triton, pinned CPU tensor (B=1024) | **24.87** | — | ok | ok | **DYNAMIC** |
| triton, raw host pointer (duck type) | 24.86 | — | ok | ok | DYNAMIC |
| **`index_select`, host ptr as fake CUDA tensor** | 22.84 | 24.32 | ok | ok | **DYNAMIC** |
| C++ `uint4` kernel via nvcc | 24.87 | 24.92 | ok | ok | DYNAMIC |
| triton reading VRAM (kernel-quality control) | 781.13 | — | n/a | — | — |
| `copy_` from **pageable** host | 12.16 | — | n/a | ok | **WRONG** |

## What this settles

1. **No bandwidth penalty.** A Triton kernel reading host memory hits
   24.87 GiB/s against the copy engine's 24.60 — it *matches* the control.
   The decision rule in the benchmark ("within ~25 % of the control") is met
   with room to spare, so the plan in this file stands.
2. **The FROZEN failure is now demonstrated, not asserted.** The `copy_`
   memcpy node captured fine and replayed the *capture-time* experts forever.
   That is exactly the fluent-wrong-audio trap this design exists to avoid,
   and it is no longer a theoretical worry — it is a measured row in a table.
3. **The kernel is not the bottleneck; PCIe is.** The same kernel reading VRAM
   does 781 GiB/s. 24 GiB/s over the bus is the ceiling, and we are at it.
4. ⚠️ **Pageable `copy_` produced WRONG BYTES** (slots held expert 0 when
   361/267/1330/124 were wanted), on top of running at half speed. Another
   silent-corruption path, now named.
5. **`index_select` works, which removes the hardest dependency.** Wrapping a
   host pointer as a fake CUDA tensor and using stock `torch.index_select`
   gets 22.84-24.32 GiB/s and is DYNAMIC under capture — **no Triton kernel,
   no nvcc, no C++ extension**. That is a far cheaper implementation path than
   this file originally assumed.

## Cost arithmetic (from the benchmark)

A decode step misses ~145 experts at the measured 53.9 % hit rate, moving
~425 MiB. Against the 46.4 ms frame budget: **17.3 ms/frame at 24 GiB/s.**

## ⚠️ The remaining risk, unmeasured

**SM contention.** A fetch *kernel* occupies streaming multiprocessors and
competes with the grouped GEMM; the copy engine does not and overlaps for
free. Today's ~0.64 of RTF spent on transfer is partly hidden by that
overlap, and a kernel may serialise against compute instead.

Matching the copy engine on raw bandwidth is **necessary but not
sufficient**. This cannot be isolated without the real model and is the main
open question for task #20.

## Also found while building this

- **Triton's vectorisation is conditional.** With the divisibility hint the
  JIT attaches automatically it emits 4-wide vector loads; without it the
  identical kernel compiles to 19 scalar loads — roughly 4x the bandwidth
  difference on a zero-copy path. Our stride (3,146,496) is divisible by 256
  so it is safe, but a stride not divisible by 16 would become a
  **misaligned-access fault**, not merely slow.
- ⚠️ **`pin_experts` in `bin/offload.py` needs rewriting for this.** Torch's
  host allocator rounds to powers of two and never splits blocks, so pinning
  per layer produces 160 separately-rounded allocations with 160 unrelated
  base addresses. A capturable kernel needs **one contiguous base**. This is a
  production problem independent of the measurement above.


---

# Stage 1 DONE: device-resident cache, both backends

`bin/devcache.py` + `csrc/fetch_experts.cu` + `bin/cppfetch.py`,
tested by `bin/test_devcache.py`.

The host sync is gone. `cached_forward`'s `topk_ids.reshape(-1).cpu()` is
replaced by three device tensors per layer:

```
slot_of[expert]    int32[256]   expert -> slot, -1 absent
slot_expert[slot]  int32[cap]   slot -> expert, -1 empty
slot_stamp[slot]   int64[cap]   step when last used
```

## The `promised` guard comes for free

In the Python cache, forgetting to protect a slot already handed out during
the same call produced fluent wrong audio, and it needed an explicit `set`.
On device it is structural: a slot touched this step carries the current
stamp, eviction picks the **minimum** stamp, so a promised slot can never be
chosen. There is no separate bookkeeping to forget.

## ⚠️ The step counter must live on device

The first version passed `step` as a Python int. Capture would have baked it
as a constant, so every replay stamps the same value, LRU ordering stops
advancing and every slot looks equally old forever. It is read and
incremented **inside** the kernel instead.

## ⚠️ Per-layer pinned tensors are fine — notes above were wrong

This file previously said a capturable kernel needs "one contiguous pinned
allocation" and that `pin_experts`'s 160 power-of-two-rounded allocations
were a production problem. **They are not, for correctness.** A captured
kernel bakes its *base* pointer, and a given layer's base never moves; the
expert offset is computed at replay. One global pool remains an optimisation
(fewer distinct bases, needed for a fused kernel) but is not a prerequisite.

## Results, both backends

| | bit-exact vs stock kernel | capture | replay | bytes after replay |
|---|---|---|---|---|
| Triton | 120 steps, 0 mismatches | ok | **DYNAMIC** | exact |
| C++ uint4 | 120 steps, 0 mismatches | ok | **DYNAMIC** | exact |

Tested at capacities 16, 32 and 64 of 64 experts with eviction genuinely
exercised (all slots occupied and recycling), at real model dimensions
(H=2048, I=512).

## Why the C++ backend exists

Measured bandwidth is identical (24.87 GiB/s both @8 experts), so this is not
a speed decision. Triton reaches that number only because its JIT
auto-attaches a divisibility hint; without it the same kernel drops to 19
scalar loads — roughly 4x slower, with nothing in the source to explain why.
`csrc/fetch_experts.cu` puts the `uint4` vectorisation in readable code, with
the `cuobjdump` command to check the SASS in a comment. That is worth having
when something is unexpectedly slow.

Build: ~45 s cold, cached after, in `torch-ext-cache/`. ⚠️ The nix store
paths it needs are **not gc roots**; `cppfetch._resolve()` re-globs if they
vanish.

## A bug the test caught

Toy test dimensions gave an 8-byte `w2_scale_inv` per expert, which is not
16 B aligned, and the `uint4` kernel correctly refused it. The real model's
scale blocks are 512 B and 256 B so this could never fire in production — but
the kernel now falls back to Triton per-buffer for anything not 16 B aligned,
rather than assuming its inputs are convenient.

## Not done

- Wiring into `MoEExperts.forward` in place of the Python cache.
- Prefill: routes to far more experts than the cache holds. It can keep the
  existing gather path, since prefill is not captured anyway.
- ⚠️ **SM contention, still unmeasured.** A fetch kernel occupies SMs and
  competes with the grouped GEMM; the copy engine overlaps for free. This is
  the open risk and it needs the real model. `blocks_per_job` and `threads`
  are exposed from the C++ side specifically so occupancy can be throttled
  without a rebuild.


---

# IT WORKS: RTF 0.97 on the real model

2026-10-17, 20:47. Device-resident cache + C++ `uint4` fetch kernel + CUDA
graph capture, on the real Drama 3 with 30 GiB of experts in host RAM.

| configuration | RTF | note |
|---|---|---|
| python cache (the day's best) | 1.71 | control, same prompt |
| + device cache, eager | 1.55 | host sync removed |
| **+ CUDA graphs** | **0.97** | **faster than realtime** |

**43 % less compute than the control**, and below the RTF 1.0 line that this
project's brief said was unreachable.

## Correctness

**4/4 STT-checked chunks identical to the control.** This is the check that
matters and it is not a formality: a frozen fetch would have been *equally
fast* and completely fluent while serving capture-time experts. Matching
words mean the right experts were fetched at replay.

Capture took 2.8 s for the four KV buckets, against 1.4 s for the synthetic
stub — the difference being the real fetch kernels now inside the graph.

11/11 chunks, 0 failed, 0 truncated. VRAM 19.73 GiB peak, host RAM 37.91 GiB,
2.0 GiB left free as requested.

## What the brief said

> "It will **run**, probably. It will not be realtime."
> "Drama 3 is 35B and will land well above RTF 1.0 on this card, so it can
> never keep ahead of a listener."

That was the right prediction from the information available, and it is now
wrong. The reasoning that overturned it: the bottleneck was never PCIe (the
predicted failure) and never arithmetic — it was **kernel-launch dispatch**,
~86 % of each step spent with the GPU idle waiting to be told what to do
next. Graphs remove that, and the expert streaming turns out to fit
comfortably inside the time the GPU was already wasting.

## ⚠️ Caveats, which are real

- **One 11-chunk sample.** Needs the full 32-minute document to confirm it
  holds at length, and a cold-cache run.
- **Only 4 chunks were STT-checked.** The cache's bit-exact unit test is
  stronger evidence than this end-to-end spot check.
- **Nobody has listened to it.** Content-correct is not the same as
  sounding right.
- **`--slice-lm-head` is OFF.** It still NaNs under capture for reasons never
  established, and it is worth 0.95 GiB. Graphs are worth far more, so it
  stays off, but the interaction is unexplained.
- **SM contention did not bite** — which was the main predicted risk. The
  fetch kernel competing with the GEMM evidently costs less than the
  dispatch overhead it removes. Not investigated further because it did not
  need to be.
