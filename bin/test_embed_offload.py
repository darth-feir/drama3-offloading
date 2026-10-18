"""Check that moving embed_tokens to host RAM returns identical embeddings.

The failure mode this is guarding against is not a crash. If the resident
slice is indexed with a prefill token id, the model gets a *valid-looking*
embedding for the wrong word and narrates fluent nonsense -- the same shape
of bug as the tower off-by-one in prefix_kv. So compare bit-for-bit, on both
kinds of id, before and after.

Three cases, because they take different branches:

  decode   ids drawn from eos + the semantic band -- the resident slice
  prefill  real text ids from a real prompt -- the host table
  mixed    both in one call, which is what a prompt row vector actually is

Plus the landmine check: `Drama3DecodeGraphs.for_model` disables CUDA graphs
if ANY backbone parameter is off the GPU, and graphs are worth twenty times
what this change saves.

  bin/run.sh bin/test_embed_offload.py --checkpoint "$DRAMA3_CKPT"
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch
import offload

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())

    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.COPY_MODE = "direct"
    offload.CACHE_EXPERTS = 0
    offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda self, devices: (
        offload.materialize_offloaded(self, device))

    model = FishOmniModel.from_pretrained(
        args.checkpoint, devices=[str(device)], tower_device=str(device),
        dtype=torch.bfloat16)
    cfg = model.config

    prompt = build_drama3_prompt(
        "The migration strategy begins with an inventory of every database.",
        model.tokenizer, cfg)
    text_ids = prompt[:, 0].to(device)
    band = torch.arange(cfg.semantic_start_token_id,
                        cfg.semantic_start_token_id + 64, device=device)
    decode_ids = torch.cat((torch.tensor([cfg.eos_token_id], device=device), band))
    mixed = torch.cat((text_ids, decode_ids))

    cases = {"decode": decode_ids, "prefill": text_ids, "mixed": mixed}
    with torch.inference_mode():
        before = {k: model.backbone.embed(v).clone() for k, v in cases.items()}

    freed = offload.offload_embed(model, device)
    print(f"\nfreed {freed:.3f} GiB", flush=True)

    with torch.inference_mode():
        after = {k: model.backbone.embed(v).clone() for k, v in cases.items()}

    failures = 0
    for name, ids in cases.items():
        a, b = before[name], after[name]
        same = a.shape == b.shape and bool(torch.equal(a, b))
        difference = (a.float() - b.float()).abs().max().item() if not same else 0.0
        print(f"{name:8s} {len(ids):5d} ids  device {b.device}  "
              f"{'IDENTICAL' if same else f'DIFFERS by {difference:.3e}'}",
              flush=True)
        if not same or b.device.type != "cuda":
            failures += 1

    # The landmine: one CPU parameter turns CUDA graphs off silently.
    stray = sorted({str(p.device) for p in model.backbone.parameters()
                    if p.device.type != "cuda"})
    print(f"backbone parameters off the GPU: {stray or 'none'}", flush=True)
    if stray:
        failures += 1

    # And the branch itself: a one-row PREFILL batch must not be mistaken for
    # a decode step. This is the case a row-count test would get wrong.
    with torch.inference_mode():
        one = text_ids[:1]
        if int(one.item()) < cfg.semantic_start_token_id:
            got = model.backbone.embed(one)
            want = model.backbone.embed(torch.cat((one, one)))[:1]
            ok = bool(torch.equal(got, want))
            print(f"one-row prefill id {int(one.item())}: "
                  f"{'ok' if ok else 'WRONG ROW'}", flush=True)
            failures += 0 if ok else 1
        else:
            print("first prompt id is in-band; one-row case not exercised",
                  flush=True)

    print(f"\n{'FAILED' if failures else 'PASS'}: {failures} problem(s)",
          flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
