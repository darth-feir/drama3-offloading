"""Where does the naive offload's time actually go?

The 64-token run moved 72 GiB at an effective 4.6 GiB/s, far under PCIe 4.0
x16. If the bus were the limit, the whole "expert cache" plan would be
bandwidth-bound exactly as the brief predicted. If instead the CPU-side gather
dominates, the plan is implementation-bound -- a very different conclusion,
because that is fixable without touching the hit rate.

This isolates the three costs for one layer's worth of work (8 experts of 256).
"""
from __future__ import annotations
import sys, time
import paths  # resolves DRAMA3_FISH; see bin/paths.py
paths.install()
import torch
from fish_speech.models.text2semantic.fish_omni.moe import select_fp8

E, I, H = 256, 512, 2048
TOPK, LAYERS = 8, 40
dev = torch.device("cuda:0")

w13 = torch.empty((E, 2 * I, H), dtype=torch.float8_e4m3fn)
w2 = torch.empty((E, H, I), dtype=torch.float8_e4m3fn)
one_layer_bytes = TOPK * (2 * I * H + H * I)
print(f"8 experts = {one_layer_bytes/2**20:.1f} MiB; a 40-layer token = "
      f"{LAYERS*one_layer_bytes/2**30:.3f} GiB\n")

def timed(fn, n=30, warmup=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n

ids = torch.randperm(E)[:TOPK]

# 1. CPU gather only -- the advanced-index copy out of the 256-expert buffer.
def gather():
    return select_fp8(w13, ids), select_fp8(w2, ids)
t_gather = timed(gather)

# 2. gather + unpinned H2D (what bin/offload.py does today)
def gather_copy():
    a, b = select_fp8(w13, ids), select_fp8(w2, ids)
    return a.to(dev, non_blocking=True), b.to(dev, non_blocking=True)
t_both = timed(gather_copy)

# 3. H2D only, from an already-gathered PINNED staging buffer -- the ceiling
#    a well-implemented cache miss could reach.
stage13 = torch.empty((TOPK, 2 * I, H), dtype=torch.float8_e4m3fn).pin_memory()
stage2 = torch.empty((TOPK, H, I), dtype=torch.float8_e4m3fn).pin_memory()
def copy_pinned():
    return (stage13.to(dev, non_blocking=True), stage2.to(dev, non_blocking=True))
t_pinned = timed(copy_pinned)

# 4. H2D only, unpinned, already gathered -- isolates pinning's contribution.
flat13 = torch.empty((TOPK, 2 * I, H), dtype=torch.float8_e4m3fn)
flat2 = torch.empty((TOPK, H, I), dtype=torch.float8_e4m3fn)
def copy_unpinned():
    return flat13.to(dev, non_blocking=True), flat2.to(dev, non_blocking=True)
t_unpinned = timed(copy_unpinned)

# 5/6. THE PROPOSED FIX: no gather at all. In w13 [E, 2I, H] each expert is a
#      contiguous block, so a miss can DMA expert e straight into row i of a
#      preallocated GPU buffer. 8 small copies, zero CPU-side materialisation.
dst13 = torch.empty((TOPK, 2 * I, H), dtype=torch.float8_e4m3fn, device=dev)
dst2 = torch.empty((TOPK, H, I), dtype=torch.float8_e4m3fn, device=dev)

def per_expert(src13, src2):
    def go():
        for i, e in enumerate(ids.tolist()):
            dst13[i].copy_(src13[e], non_blocking=True)
            dst2[i].copy_(src2[e], non_blocking=True)
    return go

t_direct = timed(per_expert(w13, w2))
print("pinning one layer's experts (768 MiB)...", flush=True)
w13p, w2p = w13.pin_memory(), w2.pin_memory()
t_direct_pinned = timed(per_expert(w13p, w2p))

GiB = 2**30
rows = [
    ("CPU gather only (no transfer)", t_gather),
    ("gather + unpinned H2D  [current]", t_both),
    ("H2D only, unpinned", t_unpinned),
    ("H2D only, PINNED  [ceiling]", t_pinned),
    ("per-expert DMA, unpinned  [fix]", t_direct),
    ("per-expert DMA, PINNED    [fix]", t_direct_pinned),
]
print(f"{'step':>36} {'ms':>8} {'GiB/s':>8} {'tok/s if 40 layers':>20}")
for name, t in rows:
    print(f"{name:>36} {1000*t:>8.2f} {one_layer_bytes/GiB/t:>8.2f} "
          f"{1/(t*LAYERS):>20.1f}")
print("\nThe 'tok/s if 40 layers' column is an upper bound on decode rate from")
print("that cost alone, ignoring all compute. Compare to the measured rate.")
