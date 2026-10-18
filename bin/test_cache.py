"""Unit-test the expert cache against the stock kernel. No model load.

Builds a tiny MoEExperts with random FP8 weights, runs the same routing
through (a) the stock fp8_moe_forward with all experts resident and (b) the
cache path, and compares. Any mismatch is the cache lying about which weights
belong to which expert.
"""
import sys
from pathlib import Path
import paths  # resolves DRAMA3_FISH; see bin/paths.py
paths.install()
import torch
import offload
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

dev = torch.device("cuda:0")
E, H, I, K = 32, 256, 128, 8
torch.manual_seed(0)

def make(device):
    m = MoEExperts(E, H, I, fp8=True, device=device)
    return m

# reference: everything on GPU, stock path
ref = make(dev)
ref.w13_weight.copy_((torch.randn(E, 2*I, H, device=dev) * 0.2).to(torch.float8_e4m3fn))
ref.w2_weight.copy_((torch.randn(E, H, I, device=dev) * 0.2).to(torch.float8_e4m3fn))
ref.w13_scale_inv.copy_(torch.rand(E, 2*I//128, H//128, device=dev) * 0.1 + 0.05)
ref.w2_scale_inv.copy_(torch.rand(E, H//128, I//128, device=dev) * 0.1 + 0.05)

# cached: same weights, stored on CPU, with a VRAM cache
cac = make(torch.device("cpu"))
for n in ("w13_weight", "w2_weight", "w13_scale_inv", "w2_scale_inv"):
    getattr(cac, n).copy_(getattr(ref, n).cpu())

fails = 0
for cap in (8, 12, E):
    offload.CACHE_EXPERTS = cap
    offload._init_cache(cac, dev, cap)
    print(f"\n=== capacity {cap} of {E} experts ===")
    for step in range(12):
        tokens = 1 if step % 3 else 5           # decode-ish and prefill-ish
        x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(tokens)])
        w = torch.rand(tokens, K, device=dev, dtype=torch.bfloat16)
        w = w / w.sum(-1, keepdim=True)
        want = fp8_moe_forward(x, ids, w, ref.w13_weight, ref.w13_scale_inv,
                               ref.w2_weight, ref.w2_scale_inv)
        got = offload.cached_forward(cac, x, ids, w)
        same = torch.allclose(want.float(), got.float(), atol=2e-2, rtol=2e-2)
        md = (want.float() - got.float()).abs().max().item()
        if not same:
            fails += 1
            u, iv = torch.unique(ids.reshape(-1).cpu(), return_inverse=True)
            n = int(u.numel())
            print(f"  step {step:2d} tokens={tokens} uniq={n:2d}  MISMATCH "
                  f"max|d|={md:.4f}  peak={want.float().abs().max().item():.4f}  "
                  f"fallback={n > cac._cap}")
            # rebuild the same gather inline and see whether IT matches
            g13 = torch.empty((n, 2*I, H), dtype=torch.float8_e4m3fn, device=dev)
            g2 = torch.empty((n, H, I), dtype=torch.float8_e4m3fn, device=dev)
            for i, e in enumerate(u.tolist()):
                g13[i].copy_(cac.w13_weight[e]); g2[i].copy_(cac.w2_weight[e])
            inline = fp8_moe_forward(x, iv.reshape(ids.shape).to(dev), w, g13,
                                     cac.w13_scale_inv[u].to(dev), g2,
                                     cac.w2_scale_inv[u].to(dev))
            print(f"      inline-gather vs ref : {(want.float()-inline.float()).abs().max().item():.6f}")
            print(f"      cached   vs inline   : {(got.float()-inline.float()).abs().max().item():.6f}")
            print(f"      got all zero?        : {bool((got==0).all())}   "
                  f"got peak={got.float().abs().max().item():.4f}")
        elif step < 3:
            print(f"  step {step:2d} tokens={tokens}  ok (max|d|={md:.4f})")
print(f"\n{'FAILED: ' + str(fails) + ' mismatches' if fails else 'PASS: cache matches the stock kernel'}")
