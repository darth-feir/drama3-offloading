"""Check the device-resident cache against the stock kernel. No model load.

Same shape of test that caught the OrderedDict cache's eviction bug: build a
small MoEExperts with random FP8 weights, run identical routing through (a)
the stock path with every expert resident on the GPU and (b) the device cache
with experts in pinned host memory, and compare.

Exercises eviction deliberately: capacity is smaller than the number of
distinct experts used across steps, so slots must be recycled. The bug this
is hunting is the one that shipped before -- a slot handed out twice in the
same step, which produces fluent wrong output rather than an error.
"""
import sys
from pathlib import Path
import paths  # resolves DRAMA3_FISH; see bin/paths.py
paths.install()
import torch
import devcache

BACKEND = sys.argv[1] if len(sys.argv) > 1 else "triton"
devcache.FETCH_BACKEND = BACKEND
print(f"backend: {BACKEND}")
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

dev = torch.device("cuda:0")
# Real model dimensions. Toy sizes gave an 8-byte w2_scale_inv per
# expert, which is not 16 B aligned and the uint4 kernel rejects --
# a bug in the TEST, but one that would have hidden the real path.
E, H, I, K = 64, 2048, 512, 8
torch.manual_seed(0)

ref = MoEExperts(E, H, I, fp8=True, device=dev)
ref.w13_weight.copy_((torch.randn(E, 2*I, H, device=dev) * 0.2).to(torch.float8_e4m3fn))
ref.w2_weight.copy_((torch.randn(E, H, I, device=dev) * 0.2).to(torch.float8_e4m3fn))
ref.w13_scale_inv.copy_(torch.rand(E, 2*I//128, H//128, device=dev) * 0.1 + 0.05)
ref.w2_scale_inv.copy_(torch.rand(E, H//128, I//128, device=dev) * 0.1 + 0.05)

host = MoEExperts(E, H, I, fp8=True, device=torch.device("cpu"))
for n in ("w13_weight", "w2_weight", "w13_scale_inv", "w2_scale_inv"):
    src = getattr(ref, n).cpu()
    # MUST be pinned: Triton rejects pageable pointers outright (verified in
    # bench_hostread -- "cannot be accessed from Triton"), which is the good
    # failure. A silent wrong read would be far worse.
    setattr(host, n, src.pin_memory())

for cap in (16, 32, E):
    devcache.init_dev_cache(host, dev, cap)
    print(f"\n=== capacity {cap} of {E} experts ===")
    bad = 0
    for step in range(40):
        x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
        ids = torch.randperm(E, device=dev)[:K].unsqueeze(0)
        w = torch.rand(1, K, device=dev, dtype=torch.bfloat16)
        w = w / w.sum(-1, keepdim=True)

        want = fp8_moe_forward(x, ids, w, ref.w13_weight, ref.w13_scale_inv,
                               ref.w2_weight, ref.w2_scale_inv)
        slots = devcache.dev_resolve_and_fetch(host, ids)
        got = fp8_moe_forward(x, slots.reshape(ids.shape).to(torch.int64), w,
                              host._dc_w13, host._dc_s13, host._dc_w2, host._dc_s2)
        d = (want.float() - got.float()).abs().max().item()
        if d != 0.0:
            bad += 1
            if bad <= 3:
                print(f"  step {step:2d}: MISMATCH max|d|={d:.6f}")
        elif step < 2:
            print(f"  step {step:2d}: exact")
    # did eviction actually happen?
    occupied = int((host._dc_slot_expert >= 0).sum())
    print(f"  40 steps, {bad} mismatches, {occupied}/{cap} slots occupied"
          f"{' (eviction exercised)' if cap < 40*K else ''}")
    if bad:
        print("  FAILED"); sys.exit(1)

print("\nPASS: device cache is bit-exact against the stock kernel")

# ---------------------------------------------------------------- capture

print("\n=== CUDA graph capture ===")
devcache.init_dev_cache(host, dev, 32)
ids_warm = torch.randperm(E, device=dev)[:K].unsqueeze(0)
ids_cap  = torch.randperm(E, device=dev)[:K].unsqueeze(0)   # set at capture time
ids_rep  = torch.randperm(E, device=dev)[:K].unsqueeze(0)   # set before replay
ids_live = ids_warm.clone()

def run():
    return devcache.dev_resolve_and_fetch(host, ids_live)

# warm up on a side stream, as torch requires before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        run()
torch.cuda.current_stream().wait_stream(s)

ids_live.copy_(ids_cap)
g = torch.cuda.CUDAGraph()
try:
    with torch.cuda.graph(g):
        out = run()
    print("  capture: ok")
except Exception as exc:
    print(f"  capture: FAILED {type(exc).__name__}: {exc}"); sys.exit(1)

# change the ids AFTER capture. A frozen implementation would still serve the
# capture-time experts; a correct one picks up the new ones.
ids_live.copy_(ids_rep)
g.replay()
torch.cuda.synchronize()

slots = out.clone()
occupant = host._dc_slot_expert[slots.to(torch.int64)]
wanted = ids_rep.reshape(-1).to(torch.int32)
if torch.equal(occupant, wanted):
    print(f"  replay : DYNAMIC -- slots hold the replay-time experts")
elif torch.equal(occupant, ids_cap.reshape(-1).to(torch.int32)):
    print(f"  replay : FROZEN -- still serving capture-time experts"); sys.exit(1)
else:
    print(f"  replay : WRONG\n    slots hold {occupant.tolist()}\n    wanted     {wanted.tolist()}")
    sys.exit(1)

# and are the BYTES right, not just the bookkeeping?
x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
w = torch.rand(1, K, device=dev, dtype=torch.bfloat16); w = w / w.sum(-1, keepdim=True)
want = fp8_moe_forward(x, ids_rep, w, ref.w13_weight, ref.w13_scale_inv,
                       ref.w2_weight, ref.w2_scale_inv)
got = fp8_moe_forward(x, slots.reshape(1, K).to(torch.int64), w,
                      host._dc_w13, host._dc_s13, host._dc_w2, host._dc_s2)
d = (want.float() - got.float()).abs().max().item()
print(f"  bytes  : {'exact' if d == 0 else f'MISMATCH max|d|={d:.6f}'}")
print("\nPASS: capturable and dynamic" if d == 0 else "\nFAILED on bytes")
