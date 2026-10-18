"""Prove the lm_head slice is bit-identical where it matters.

The slice keeps only the 4,097 rows decode can sample and scatters them back
into a full-width -inf buffer. Correct means: at every candidate position the
value equals the unsliced lm_head exactly, and nothing else is reachable.
"""
import sys
from pathlib import Path
import paths  # resolves DRAMA3_FISH; see bin/paths.py
paths.install()
import torch, torch.nn as nn

dev = torch.device("cuda:0")
VOCAB, HID = 252288, 2048
EOS, LO, HI = 248046, 248099, 252194

torch.manual_seed(0)
full = nn.Linear(HID, VOCAB, bias=False, device=dev, dtype=torch.bfloat16)
cand = torch.cat((torch.tensor([EOS]), torch.arange(LO, HI + 1))).sort().values.to(dev)
print(f"candidates: {cand.numel():,} of {VOCAB:,}  ({100*cand.numel()/VOCAB:.2f}%)")

head = nn.Linear(HID, cand.numel(), bias=False, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    head.weight.copy_(full.weight[cand])

def sliced_logits(h):
    out = head(h)
    wide = torch.full((out.shape[0], VOCAB), float("-inf"), device=out.device, dtype=out.dtype)
    wide[:, cand] = out
    return wide

# compute_logits is only ever called with ONE row: generate.py:252 does
# `hidden = hidden[-1:]` before the decode loop, and the CUDA-graph path uses
# a (1, hidden) buffer (generate.py:376). rows>1 is tested for information
# only -- a wider GEMM reassociates differently in bf16, which is expected and
# does not occur in this code path.
bad = 0
REAL_ROWS = 1
for rows in (1, 2, 4):
    h = torch.randn(rows, HID, device=dev, dtype=torch.bfloat16)
    ref, got = full(h), sliced_logits(h)
    same = torch.equal(ref[:, cand], got[:, cand])
    noncand = torch.ones(VOCAB, dtype=torch.bool, device=dev); noncand[cand] = False
    masked = bool(torch.isinf(got[:, noncand]).all())
    d = (ref[:, cand].float() - got[:, cand].float()).abs().max().item()
    pk = ref[:, cand].float().abs().max().item()
    real = rows == REAL_ROWS
    if real:
        bad += (not same) + (not masked)
    tag = "  <-- the only shape this code path uses" if real else "  (informational)"
    print(f"  rows={rows}: identical={same}  masked={masked}  "
          f"max|d|={d:.6f} ({d/max(pk,1e-9):.1e} of peak){tag}")

mb_full = full.weight.numel() * 2 / 2**20
mb_head = head.weight.numel() * 2 / 2**20
print(f"\nweight {mb_full/1024:.2f} GiB -> {mb_head/1024:.3f} GiB "
      f"(frees {(mb_full-mb_head)/1024:.2f} GiB)")
print("PASS (rows=1 bit-identical)" if not bad else f"FAILED ({bad} checks at rows=1)")
