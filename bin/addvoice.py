#!/usr/bin/env python
"""Build a named voice from a reference clip and store it for reuse.

    bin/run.sh bin/addvoice.py NAME /path/to/clip.mp3 [--ref-text "..."]

Prefills the clip once, snapshots the state (118 MiB), and writes it to
voices/ so the server can offer it by name. After this, a request with
`"voice": "NAME"` on /v1/audio/speech speaks in that voice with no
re-prefill -- `voice` was accepted and ignored until now, so the standard
field finally means something.

⚠️ THIS LOADS THE MODEL, which takes ~20 GiB. The resident server holds
about the same, and two of them do not fit on a 24 GiB card. So it refuses
to start while the server is up rather than racing it into an OOM that
would look like a bug in whichever one lost.

⚠️ Duplication to be aware of: the model/codec/dev-cache setup below is
the third copy of the same sequence (narrate.py, server.py, here). They
have drifted apart once already. It wants factoring into one loader; it
is written out here rather than refactoring three callers in the same
change that introduces a feature.
"""

from __future__ import annotations

import argparse
import os
import time
import urllib.request
from pathlib import Path

import paths  # noqa: F401  -- resolves DRAMA3_FISH before fish_speech imports
import torch

import offload
import prefix_kv
import voices
from prefix_kv import PrefixKV

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    ensure_drama3_caches, Drama3DecodeGraphs,
)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, resolve_codec_path, encode_audio,
)

from narrate import snap_to_silence, to_pcm16, transcribe


def server_is_up() -> str | None:
    """The running server's URL, or None. It publishes where it bound."""
    url_file = Path(os.environ.get("DRAMA3_RUN",
                                   Path(__file__).resolve().parent.parent / "run"))
    url_file = url_file / "server.url"
    if not url_file.is_file():
        return None
    url = url_file.read_text().strip()
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2):
            return url
    except Exception:              # noqa: BLE001 -- unreachable means not up
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="what to call this voice")
    ap.add_argument("clip", help="30-60 s of one person speaking cleanly")
    ap.add_argument("--ref-text", default=None,
                    help="transcript of the clip; STT is used when omitted")
    ap.add_argument("--ref-seconds", type=float, default=56.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pin-experts", action="store_true")
    ap.add_argument("--dev-cache", action="store_true")
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--cache-experts", type=int, default=0)
    ap.add_argument("--default", action="store_true",
                    help="also make this the default voice")
    ap.add_argument("--force", action="store_true",
                    help="build even if the server is running (will likely OOM)")
    args = ap.parse_args()

    if not Path(args.clip).is_file():
        raise SystemExit(f"no such clip: {args.clip}")
    if args.name in voices.names():
        print(f"note: replacing the existing voice {args.name!r}")

    running = server_is_up()
    if running and not args.force:
        raise SystemExit(
            f"the server is running at {running} and holds ~20 GiB of the\n"
            f"GPU; building a voice needs about the same and a 24 GiB card\n"
            f"will not hold both.\n\n"
            f"  bin/drama3ctl stop            # then add the voice\n"
            f"  bin/drama3ctl --addvoice ...  # does this for you\n\n"
            f"--force skips this check if you know the card is big enough.")

    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    device = torch.device(args.device)
    # Graphs off: this process only prefills, never decodes, so capture
    # would cost ~45 s on a cold Triton cache and buy nothing.
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.COPY_MODE = "direct"
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

    work = voices.voices_dir() / ".build"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{args.name}.wav"
    seconds = snap_to_silence(Path(args.clip), args.ref_seconds)
    to_pcm16(Path(args.clip), wav, seconds)
    ref_text = args.ref_text or transcribe(wav)
    if not ref_text or ref_text.startswith("<stt failed"):
        raise SystemExit(
            f"no transcript for the clip ({ref_text!r}).\n"
            "Pass --ref-text, or start an STT server and set DRAMA3_STT.\n"
            "An approximate transcript is fine -- measured no worse on "
            "timbre or content (notes/17) -- but some text is required.")
    print(f"reference: {seconds:.1f}s, transcript {len(ref_text.split())} words",
          flush=True)

    prompt_tokens = (encode_audio(wav, codec, device),)
    ensure_drama3_caches(model, Drama3DecodeGraphs.graph_capacity(model))
    rows = prefix_kv.prefix_rows(model.tokenizer, model.config,
                                 (ref_text,), prompt_tokens)
    prefix = PrefixKV(model, rows)
    prefix.build(prefill_chunk_size=args.prefill_chunk)

    path = voices.save(args.name, prefix, model, args.checkpoint,
                       source=args.clip, ref_text=ref_text, seconds=seconds,
                       prefill_chunk=args.prefill_chunk,
                       prompt_tokens=prompt_tokens)
    print(f"\nvoice {args.name!r}: {prefix.bytes / 2**20:.0f} MiB, "
          f"{prefix.P} rows, prefilled in {prefix.build_seconds:.2f}s")
    print(f"  -> {path}")

    if args.default or len(voices.names()) == 1:
        voices.set_default(args.name)
        print(f"  default voice is now {args.name!r}")
    print(f"\nvoices: {', '.join(voices.names())}")
    print("Restart the server to pick it up:  bin/drama3ctl restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
