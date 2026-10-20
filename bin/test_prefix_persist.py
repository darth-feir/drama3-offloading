#!/usr/bin/env python
"""Can a prefix-KV snapshot survive leaving the process?

    bin/run.sh bin/test_prefix_persist.py --phase save   --ref CLIP ...
    bin/run.sh bin/test_prefix_persist.py --phase verify --ref CLIP ...

THE QUESTION
------------
`PrefixKV` prefills a reference clip once and restores it per chunk, which
is why reference length is nearly free. It is built fresh every boot and
lives only in memory. If a snapshot could be written to disk and reloaded,
a server could hold several NAMED voices and answer `voice: "..."` on the
OpenAI route without re-prefilling anything -- and `voice` is currently
accepted and ignored, so the protocol slot already exists.

Two things make that non-obvious, and both are what this measures:

  1. A snapshot holds LIVE MODULE REFERENCES -- `(module, k, v)` tuples --
     and `_buffer_ids` are `data_ptr()` values. Both are process-local.
     Persisting means saving tensors in module order and re-binding them,
     and `is_valid()` cannot be the check any more: pointers always differ
     in a new process. A persisted snapshot needs a BUILD FINGERPRINT.
  2. Whether the bytes even come back the same.

⚠️ THE CONTROL MATTERS MORE THAN THE TEST. If a freshly BUILT prefix in
process 2 does not already match process 1's, then "the reloaded one does
not match" says nothing about persistence -- it is cross-process
nondeterminism, and no amount of careful serialisation fixes it. An
earlier experiment in this project (`test_prefix_kv.py`, control E) was
saved by exactly this kind of control: `--prefill-chunk 128` vs `1024`
diverges at frame 0, so a naive A/B would have blamed the wrong thing.

So `--phase verify` reports three comparisons:

    built-here   vs saved     the CONTROL: is the build reproducible at all
    loaded       vs saved     did the bytes survive the round trip
    built-here   vs loaded    the question the feature actually depends on

EXIT 0 if persistence looks viable, 1 if not. Nothing here is a feature;
it decides whether the feature is worth writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import paths  # noqa: F401  -- resolves DRAMA3_FISH before fish_speech imports
import torch

import offload
import prefix_kv
from prefix_kv import PrefixKV

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, ensure_drama3_caches, Drama3DecodeGraphs,
)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, resolve_codec_path, encode_audio,
)

# narrate owns the clip preparation; encode_audio comes from upstream.
from narrate import snap_to_silence, to_pcm16, transcribe

TEXT = "The ledger was kept in pencil, because the sea is not a thing you write down in ink."


def digest(t: torch.Tensor) -> str:
    """Bit-exact fingerprint. .cpu() first: hashing device memory is not
    defined, and bfloat16 has no numpy view, so go through raw bytes."""
    return hashlib.sha256(
        t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def snapshot_tensors(prefix: PrefixKV) -> list[tuple[str, torch.Tensor]]:
    """Every tensor in the snapshot, in a STABLE order.

    Order is the contract: on reload the tensors are re-bound to modules by
    position, because the module objects themselves cannot be serialised.
    """
    out: list[tuple[str, torch.Tensor]] = [("carry", prefix.carry)]
    for i, (_, k, v) in enumerate(prefix.backbone_kv):
        out += [(f"backbone.{i}.k", k), (f"backbone.{i}.v", v)]
    for i, (_, c, r) in enumerate(prefix.gdn):
        out += [(f"gdn.{i}.conv", c), (f"gdn.{i}.recurrent", r)]
    for i, (_, k, v) in enumerate(prefix.tower_kv):
        out += [(f"tower.{i}.k", k), (f"tower.{i}.v", v)]
    return out


def fingerprint(args, model, prefix: PrefixKV) -> dict:
    """What must match for a saved snapshot to be safe to reuse.

    ⚠️ NOT data_ptr(). Those differ in every process and would reject every
    reload. These are the things that, if changed, make the restored state
    wrong rather than merely relocated -- a stale snapshot silently
    restored is the poisoned-cache failure this project already hit once
    with torch-ext-cache.
    """
    return {
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "rows_sha": digest(prefix.rows),
        "P": prefix.P,
        "cache_capacity": int(model.cache_capacity),
        "n_backbone": len(prefix.backbone_kv),
        "n_gdn": len(prefix.gdn),
        "n_tower": len(prefix.tower_kv),
        "prefill_chunk": args.prefill_chunk,
        "cuda_graph": args.cuda_graph,
        "dtype": str(prefix.carry.dtype),
    }


def build_everything(args):
    """Load the model and build a prefix from the clip. Shared by both phases."""
    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = args.cuda_graph
    offload.COPY_MODE = args.copy_mode
    offload.CACHE_EXPERTS = args.cache_experts
    offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda self, devices: (
        offload.materialize_offloaded(self, device))

    t0 = time.perf_counter()
    model = FishOmniModel.from_pretrained(
        args.checkpoint, devices=[str(device)], tower_device=str(device),
        dtype=torch.bfloat16)
    if args.pin_experts:
        offload.pin_experts(model, device)
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)),
                             device=device, precision=torch.bfloat16)
    if args.dev_cache:
        import devcache
        devcache.FETCH_BACKEND = args.fetch_backend
        cap = (offload.size_cache_to_free_vram(model, device, args.vram_reserve)
               if args.vram_reserve else args.cache_experts)
        for layer in model.backbone.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and getattr(experts, "_cap", 0):
                experts._cap = 0
                for name in ("_cache_w13", "_cache_w2", "_cache_s13", "_cache_s2"):
                    if hasattr(experts, name):
                        delattr(experts, name)
        torch.cuda.empty_cache()
        devcache.install_dev_cache(model, device, cap)
    print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "reference.wav"
    seconds = snap_to_silence(Path(args.ref), args.ref_seconds)
    to_pcm16(Path(args.ref), wav, seconds)
    ref_text = args.ref_text or transcribe(wav)
    prompt_texts = (ref_text,)
    prompt_tokens = (encode_audio(wav, codec, device),)
    print(f"reference: {seconds:.1f}s, {prompt_tokens[0].shape[1]} codes", flush=True)

    ensure_drama3_caches(model, Drama3DecodeGraphs.graph_capacity(model))
    rows = prefix_kv.prefix_rows(model.tokenizer, model.config,
                                 prompt_texts, prompt_tokens)
    prompt = build_drama3_prompt(TEXT, model.tokenizer, model.config,
                                 prompt_texts, prompt_tokens)
    prefix = PrefixKV(model, rows)
    prefix.build(prefill_chunk_size=args.prefill_chunk)
    return model, prefix, prompt


def phase_save(args) -> int:
    model, prefix, _ = build_everything(args)
    tensors = snapshot_tensors(prefix)
    store = Path(args.snapshot); store.parent.mkdir(parents=True, exist_ok=True)

    # .cpu() on save: a CUDA tensor pickled with its device would reload
    # onto whatever ordinal it was saved from, which is not portable.
    payload = {name: t.detach().to("cpu").clone() for name, t in tensors}
    torch.save(payload, store)
    meta = {
        "fingerprint": fingerprint(args, model, prefix),
        "order": [name for name, _ in tensors],
        "digests": {name: digest(t) for name, t in tensors},
        "bytes": prefix.bytes,
        "build_seconds": prefix.build_seconds,
    }
    store.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    size_mib = store.stat().st_size / 2**20
    print(f"\nsaved {len(tensors)} tensors, {size_mib:.0f} MiB on disk "
          f"({prefix.bytes / 2**20:.0f} MiB live), built in "
          f"{prefix.build_seconds:.2f}s")
    print(f"  -> {store}")
    return 0


def phase_verify(args) -> int:
    store = Path(args.snapshot)
    if not store.is_file():
        raise SystemExit(f"no snapshot at {store}; run --phase save first")
    saved_meta = json.loads(store.with_suffix(".json").read_text())
    saved = torch.load(store, map_location="cpu")

    model, prefix, _ = build_everything(args)
    here = snapshot_tensors(prefix)
    here_meta = fingerprint(args, model, prefix)

    print("\n=== fingerprint ===")
    ok_fp = True
    for key, want in saved_meta["fingerprint"].items():
        got = here_meta.get(key)
        same = got == want
        ok_fp &= same
        print(f"  {'ok  ' if same else 'DIFF'} {key:<16} {want} "
              f"{'' if same else f'-> {got}'}")

    # CONTROL: is the build reproducible across processes at all?
    print("\n=== built-here vs saved  (CONTROL: cross-process determinism) ===")
    control_same = sum(digest(t) == saved_meta["digests"][n] for n, t in here)
    print(f"  {control_same}/{len(here)} tensors bit-identical")

    # Did the round trip preserve the bytes?
    print("\n=== loaded vs saved  (did the round trip preserve bytes) ===")
    rt_same = sum(digest(saved[n]) == saved_meta["digests"][n]
                  for n, _ in here)
    print(f"  {rt_same}/{len(here)} tensors bit-identical")

    # What the feature depends on.
    print("\n=== built-here vs loaded  (what multi-voice would rely on) ===")
    feat_same = sum(digest(t) == digest(saved[n]) for n, t in here)
    print(f"  {feat_same}/{len(here)} tensors bit-identical")

    n = len(here)
    print("\n=== verdict ===")
    if rt_same != n:
        print("  FAIL: serialisation itself lost bytes. Nothing else matters.")
        return 1
    if control_same != n:
        print(f"  ⚠️  The CONTROL failed: only {control_same}/{n} match when")
        print("      rebuilt in a fresh process, so the prefix build is NOT")
        print("      bit-reproducible across processes. A reloaded snapshot")
        print("      cannot be judged against a rebuilt one -- but note the")
        print("      round trip above was clean, so persistence is still")
        print("      viable; it just cannot be validated this way.")
        return 1
    if feat_same == n:
        print("  PASS: a reloaded snapshot is bit-identical to a rebuilt one.")
        print("        Named multi-voice is plumbing from here.")
        return 0
    print(f"  FAIL: {n - feat_same}/{n} tensors differ after reload.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("save", "verify"), required=True)
    ap.add_argument("--snapshot", default="out/prefix-persist/voice.pt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-seconds", type=float, default=56.0)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--copy-mode", default="direct")
    ap.add_argument("--pin-experts", action="store_true")
    ap.add_argument("--dev-cache", action="store_true")
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--cache-experts", type=int, default=0)
    ap.add_argument("--cuda-graph", choices=("0", "1"), default="1")
    ap.add_argument("--out-dir", default="out/prefix-persist")
    args = ap.parse_args()
    if not args.ref:
        raise SystemExit("no reference clip: pass --ref")
    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    return phase_save(args) if args.phase == "save" else phase_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
