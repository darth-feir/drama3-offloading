# Running this on something that is not an Ada NVIDIA card

**Status: DESIGN ONLY. None of this has been run.** Everything else in
`notes/` is measured; this file is not, and the arithmetic in it is marked as
arithmetic. There is one GPU on the machine this was developed on and it is
an RTX 4090.

## The problem

`fp8_gemm_supported()` (`fp8_moe.py:22`) is a hard gate:

```python
if triton is None or not torch.cuda.is_available() or torch.version.hip:
    return False
return all(torch.cuda.get_device_capability(device) >= (8, 9) for device in devices)
```

Below sm_89, or on ROCm at all, it returns False — and then
`load_backbone_weights` **dequantises every expert at load time**. 30 GiB of
FP8 becomes ~60 GiB in bf16, which is more RAM than most people have and
removes the entire reason offloading works here.

So the practical requirement today is **Ada or newer**: RTX 40-series, Ada
workstation cards, Hopper. Not Ampere (a 3090's 24 GB is no help — sm_86 has
no FP8), not Turing, not AMD, not Apple.

## Why this is fixable

Two things in the upstream code make it much less work than it looks.

**FP8 storage already works everywhere.** `select_fp8` (`moe.py:41`) exists
precisely because *"advanced indexing has no float8 kernel"* — it views the
tensor as `uint8` and indexes that. FP8 weights are bytes; moving, caching
and indexing them needs no hardware support at all. Only the **matrix
multiply** needs sm_89.

**A dtype-agnostic compute path already exists and is already used.** When
`self.fp8` is False, `MoEExperts.forward` falls through to
`forward_gathered` / `forward_grouped` (`moe.py:83`, `:96`), which are plain
`torch.bmm`. They get their weights from `expert_weights` (`moe.py:116`),
which does exactly the right thing:

```python
return (
    dequantize_fp8_blocks(select_fp8(self.w13_weight, index),
                          self.w13_scale_inv[index], dtype),
    dequantize_fp8_blocks(select_fp8(self.w2_weight, index),
                          self.w2_scale_inv[index], dtype),
)
```

Per-expert, on demand, into whatever dtype you ask for. And
`dequantize_fp8_blocks` (`moe.py:21`) is four lines of ordinary PyTorch —
no Triton, no tensor cores.

**The mechanism is already written. It is wired to the wrong switch.**

## The actual change: split one flag into two

`MoEExperts.fp8` currently means two different things at once:

1. the weights are **stored** as `float8_e4m3fn`, and
2. the forward pass uses the **W8A8 Triton kernel**.

`fp8_gemm_supported()` controls both, so a card that cannot do (2) is also
denied (1), and that is what inflates 30 GiB to 60.

Split them:

| | gated by | effect |
|---|---|---|
| `fp8_storage` | the checkpoint's format | keep FP8 bytes in RAM and in the VRAM cache; dequantise per use |
| `fp8_compute` | `fp8_gemm_supported()` | use `fp8_moe_forward` instead of bmm |

On a 4090 both are true and nothing changes. On a 4070, a 3090 or a 7900 XTX
storage stays true and compute goes false, giving:

```
RAM (FP8, 30 GiB)  ->  PCIe (FP8)  ->  VRAM cache (FP8)
                   ->  dequantise only the experts this step routed to
                   ->  stock bmm
```

**Nothing upstream of the GEMM changes.** The 30 GiB stays 30 GiB. PCIe
traffic is byte-for-byte identical. The VRAM cache stays FP8, so its
capacity in experts is unchanged — which matters, because `10` establishes
that prefill cost is almost entirely cache capacity.

## Where the code is

Upstream (would need a patch, or a runtime hook like everything else here):

- `fp8_moe.py:22` — `fp8_gemm_supported`, gate `fp8_compute` only
- `moe.py:131` — `MoEExperts.forward`, pick the path on `fp8_compute`
- `backbone.py:330` — the load-time dequantise, skip it when `fp8_storage`

In this repo, `fp8_moe_forward` is called at six sites, all in two files:

- `offload.py:180, 254, 363, 411, 472` — the gather paths
- `devcache.py:263` — the device-cache path

Each already hands the kernel a **subset** tensor of shape `[n, ...]` with
renumbered ids, which is exactly the shape `expert_weights` wants. A helper
that takes `(w13, s13, w2, s2, ids, weights)` and dispatches to either
`fp8_moe_forward` or dequantise-then-bmm would cover all six.

## Cost, as arithmetic and not as measurement

One expert is 3.001 MiB of FP8 (`w13` [1024, 2048] + `w2` [2048, 512]).
Dequantising reads 3 MiB and writes 6 MiB of bf16, so ~9 MiB of traffic.

```
decode   8 experts x 40 layers x 9 MiB   = 2.9 GiB per step
         at ~1 TB/s                      ~ 2.9 ms
         against a 34.8 ms step (measured, on a 4090)   ~ 8 %

prefill  ~196 experts x 40 layers x 9 MiB = 70 GiB per batch
         at ~1 TB/s                       ~ 70 ms
         against ~1.1 s of prefill (measured)            ~ 6 %
```

⚠️ Those percentages are against **4090 step times**. On a slower card every
other term grows too, so the relative cost would be lower — and the honest
comparison is not "8 % slower" but "runs at all instead of not running".

⚠️ The prefill **scratch buffer** is the real constraint: 196 experts
expanded to bf16 is 1.18 GiB for one layer. Transient, but on a 12 GB card
that has to be chunked over experts rather than materialised whole.

## What to measure, in order

1. **Does `float8_e4m3fn → float32` have a CUDA kernel below sm_89?**
   One line to find out: `torch.zeros(4, dtype=torch.float8_e4m3fn,
   device="cuda").to(torch.float32)`. If it raises, the dequantise has to be
   written as Triton bit manipulation instead — e4m3 unpacking is about ten
   lines of shifts and masks, and Triton supports the older architectures.
   **Everything else depends on this answer.**
2. Does the rest of the model run? Attention, the tower, the codec. Nothing
   to do with FP8, and never tried below sm_89.
3. Then the memory floor, then the speed.

## The VRAM floor, for sizing

Measured on a 4090, resident after load, before any expert cache:

| | VRAM at load |
|---|---|
| plain | 10.72 GiB |
| `--slice-lm-head` | 9.77 GiB |
| `+ --offload-embed` | **8.83 GiB** |
| `+ --codec-cpu` | ~7.1 GiB (but measured 50 % slower overall) |

Add ~1.2–1.5 GiB of activation headroom. So **~10.3 GiB is the floor** with
both lossless slices and a zero-size expert cache, and 12 GB leaves about
1.7 GiB for cache — roughly 14 experts per layer, against the 80 this
project runs with. It would work. It would be slow.

## AMD

The same change fixes AMD, and for a slightly different reason than it fixes
Ampere.

FP8 is not one format. These weights are **e4m3fn** (NVIDIA / OCP). CDNA3
(MI300) implements **e4m3fnuz** — different exponent bias, no infinities,
different zero handling. Same width, incompatible meaning; feeding one to
the other gives wrong numbers rather than an error, which is why
`fp8_gemm_supported` rejects `torch.version.hip` outright rather than
checking the capability. RDNA3 has no FP8 matrix support at all.

Dequantising in software **sidesteps that completely** — you interpret the
bytes yourself instead of handing them to hardware that reads them
differently. What would remain:

- `csrc/fetch_experts.cu` — 99 lines of `uint4` copy from pinned host
  memory. hipify handles most of this; the host-pointer mapping needs
  checking.
- `devcache.py`'s Triton kernels — Triton supports ROCm.

⚠️ Note the irony before investing: the AMD card this would most obviously
suit, an MI300X, has 192 GB of HBM. The whole model fits and offloading
solves a problem it does not have. The case that matters is a 24 GB
RDNA3/RDNA4 consumer card — which is exactly where AMD's FP8 support is
weakest or absent.

## What not to do

**Do not just remove the capability check.** The gate is correct: below
sm_89 the W8A8 kernel does not have hardware to run on, and on ROCm it would
misinterpret the bytes. Bypassing it produces wrong audio, not an error. The
fix is a second code path, not a deleted guard.
