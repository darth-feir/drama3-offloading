"""Can we pin 30 GiB, and is per-expert DMA still fast from a region that big?

Pinning is the precondition for CUDA-graph capture and for genuinely async
copies -- not the bandwidth tweak it was first judged as. Before designing
around it, two things need to be true on THIS box: the allocation has to
succeed without hurting the system, and per-expert DMA out of a 30 GiB pinned
region has to be as fast as out of a small one.

⚠️ What this does NOT measure: the bandwidth a custom kernel gets reading
device-MAPPED host memory. That needs a kernel taking a raw host pointer,
which PyTorch will not hand out. It stays unmeasured until someone writes it.
"""
from __future__ import annotations
import time, torch

EXPERT = 3_146_496
GiB = 2**30
dev = torch.device("cuda:0")

def timed(fn, n=20, warm=4):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n

print(f"{'region':>10} {'pin s':>8} {'8-expert DMA ms':>17} {'GiB/s':>8}  note")
held = []
for gib in (1, 4, 16, 30):
    n_exp = int(gib * GiB // EXPERT)
    t0 = time.perf_counter()
    try:
        buf = torch.empty((n_exp, EXPERT), dtype=torch.uint8, pin_memory=True)
    except RuntimeError as e:
        print(f"{gib:>9}G  FAILED: {str(e)[:60]}")
        break
    pin_s = time.perf_counter() - t0
    held.append(buf)
    dst = torch.empty((8, EXPERT), dtype=torch.uint8, device=dev)
    ids = torch.randperm(n_exp)[:8].tolist()
    def go():
        for i, e in enumerate(ids):
            dst[i].copy_(buf[e], non_blocking=True)
    ms = timed(go) * 1000
    gbs = 8 * EXPERT / GiB / (ms / 1000)
    free_ram = int([l.split()[1] for l in open("/proc/meminfo")
                    if l.startswith("MemAvailable")][0]) / 2**20
    print(f"{gib:>9}G {pin_s:>8.1f} {ms:>17.2f} {gbs:>8.2f}  "
          f"{free_ram:.0f} GiB RAM still available")

print("\n--- unpinned comparison at the same granularity ---")
up = torch.empty((int(4 * GiB // EXPERT), EXPERT), dtype=torch.uint8)
dst = torch.empty((8, EXPERT), dtype=torch.uint8, device=dev)
ids = torch.randperm(up.shape[0])[:8].tolist()
def go_up():
    for i, e in enumerate(ids):
        dst[i].copy_(up[e], non_blocking=True)
ms = timed(go_up) * 1000
print(f"  unpinned 4G region: {ms:.2f} ms  {8*EXPERT/GiB/(ms/1000):.2f} GiB/s")
print("\n(a 40-layer token needs 40 x this)")
