"""Measure which experts English actually uses, and how often.

Not "which experts are touched at all" -- the saturation curve says that
heads toward all 256 given enough text (one 211-frame document already hit
236/256 in layer 0). The useful question is the call *distribution*: what
fraction of routing decisions the top N experts account for. That is what
decides whether an English-only model could drop any, and by how much.

Writes no audio. A full id trace is ~4 MB per minute and we only aggregate
it; the census is 40x256 integers regardless of corpus length.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()
import torch, numpy as np, offload, narrate
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, generate_drama3, Drama3PromptTooLongError)
from fish_speech.models.text2semantic.fish_omni.chunking import Drama3History
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, resolve_codec_path)

SR, FRAME_RATE, BPS = 44100, 44100/512/4, 16.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--label", required=True, help="e.g. english, chinese")
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="stop after about this much generated audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    ap.add_argument("--ref-seconds", type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--pin-experts", action="store_true", default=True)
    ap.add_argument("--max-bytes", type=int, default=400)
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    out = Path(args.out or f"notes/census-{args.label}.json")

    chunks = []
    for f in args.corpus:
        text = Path(f).read_text(encoding="utf-8", errors="ignore")
        chunks += narrate.paragraphs(text, args.max_bytes)
    print(f"corpus: {len(chunks)} chunks, "
          f"{sum(len(c.encode()) for c in chunks)/1024:.0f} KB, "
          f"~{sum(len(c.encode()) for c in chunks)/BPS/60:.0f} min if all read")

    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.COPY_MODE = "direct"; offload.TRACE.enabled = False
    offload.enable_census()
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda s, d: offload.materialize_offloaded(s, device)

    model = FishOmniModel.from_pretrained(args.checkpoint, devices=[str(device)],
                                          tower_device=str(device), dtype=torch.bfloat16)
    if args.pin_experts:
        gib, secs = offload.pin_experts(model, device)
        print(f"pinned {gib:.1f} GiB in {secs:.0f}s")
    offload.slice_lm_head(model, device)
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)), device=device)
    cap = offload.size_cache_to_free_vram(model, device, args.vram_reserve)
    print(f"cache {cap}/layer", flush=True)

    prompt_texts, prompt_tokens = (), ()
    ref = Path(args.ref)
    if ref.is_file():
        wav = Path("out/census-ref.wav"); wav.parent.mkdir(exist_ok=True, parents=True)
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(ref),"-t",str(args.ref_seconds),
                        "-ac","1","-ar","44100","-c:a","pcm_s16le",str(wav)], check=True)
        import socket
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock = paths.stt_socket()
        if sock is None:
            return ("<stt unavailable: set DRAMA3_STT to a socket, "
                    "or pass --ref-text>")
        s.connect(sock)
        s.sendall((str(wav.resolve())+"\n").encode()); buf=b""
        while True:
            c = s.recv(65536)
            if not c or b"\n" in buf: break
            buf += c
        s.close(); rt = buf.decode().strip()
        if rt:
            prompt_texts, prompt_tokens = (rt,), (encode_audio(wav, codec, device),)

    history = Drama3History()
    audio_s = 0.0
    started = time.perf_counter()
    for i, chunk in enumerate(chunks):
        if audio_s / 60 >= args.minutes:
            break
        prompt = None
        for turns in history.candidates():
            try:
                prompt = build_drama3_prompt(chunk, model.tokenizer, model.config,
                                             prompt_texts, prompt_tokens, history=turns)
                break
            except Drama3PromptTooLongError:
                continue
        if prompt is None:
            continue
        f, _ = generate_drama3(model, prompt, max_new_tokens=narrate.frame_budget(chunk, 600),
                               prefill_chunk_size=args.prefill_chunk)
        if not len(f):
            continue
        history.push(chunk, f[:, 1:].T.cpu())
        audio_s += len(f) / FRAME_RATE
        if i % 20 == 0:
            el = time.perf_counter() - started
            print(f"  [{i}/{len(chunks)}] {audio_s/60:5.1f} min audio, "
                  f"{el/60:5.1f} min compute, RTF {el/max(audio_s,1e-9):.2f}", flush=True)
            out.write_text(json.dumps({"label": args.label, "audio_seconds": audio_s,
                                       "chunks_done": i, "census": offload.CENSUS.tolist(),
                                       "census_decode": offload.CENSUS_DECODE.tolist()}))

    out.write_text(json.dumps({"label": args.label, "audio_seconds": audio_s,
                               "chunks_done": len(chunks), "corpus": args.corpus,
                               "census": offload.CENSUS.tolist(),
                               "census_decode": offload.CENSUS_DECODE.tolist()}))
    print(f"\n{audio_s/60:.1f} min of audio, census -> {out}")


if __name__ == "__main__":
    main()
