"""Generate several short texts in one model load and report what happened.

Loading Drama 3 costs ~14s, so testing N hypotheses as N separate runs wastes
most of the time on loading. This loads once and sweeps a list, reporting for
each: frames, whether it stopped on EOS or ran into the token cap, how much of
the audio is actually speech rather than trailing silence, and what STT hears.

Used for two things: finding why a chunk fails to stop, and checking whether
[bracketed cues] and <whisper> tags do anything audible.
"""
from __future__ import annotations
import argparse, json, os, socket, subprocess, sys, time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()
import torch, numpy as np, offload
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, generate_drama3)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, decode_to_audio, resolve_codec_path)

FRAME_RATE = 44100 / 512 / 4
SR = 44100


def stt(path: Path) -> str:
    tmp = path.with_suffix(".16k.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1",
                    "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)], check=True)
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock = paths.stt_socket()
        if sock is None:
            return ("<stt unavailable: set DRAMA3_STT to a socket, "
                    "or pass --ref-text>")
        s.connect(sock)
        s.sendall((str(tmp.resolve()) + "\n").encode())
        buf = b""
        while True:
            c = s.recv(65536)
            if not c or b"\n" in buf: break
            buf += c
        s.close(); return buf.decode().strip()
    except Exception as e:
        return f"<stt failed: {e}>"
    finally:
        tmp.unlink(missing_ok=True)


def speech_span(d: np.ndarray, sr: int, thresh=0.004):
    win = sr // 10
    rms = np.array([np.sqrt((d[i:i+win]**2).mean()) for i in range(0, len(d), win)])
    loud = np.where(rms > thresh)[0]
    if not len(loud): return 0.0, 0.0
    return loud[0] / 10, (loud[-1] + 1) / 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--cases", required=True, help="json list of {name,text}")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    ap.add_argument("--ref-seconds", type=float, default=20.0)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--max-attempts", type=int, default=1,
                    help="redraw when the spoken duration implies a stall or babble")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.COPY_MODE = "direct"; offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda s, d: offload.materialize_offloaded(s, device)

    cases = json.loads(Path(args.cases).read_text())
    model = FishOmniModel.from_pretrained(args.checkpoint, devices=[str(device)],
                                          tower_device=str(device), dtype=torch.bfloat16)
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)), device=device)

    prompt_texts, prompt_tokens = (), ()
    ref = Path(args.ref)
    if ref.is_file():
        wav = out / "reference.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ref), "-t",
                        str(args.ref_seconds), "-ac", "1", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        rt = stt(wav)
        if rt and not rt.startswith("<stt"):
            prompt_texts, prompt_tokens = (rt,), (encode_audio(wav, codec, device),)

    results = []
    import soundfile as sf
    expanded = []
    for case in cases:
        for rep in range(case.get("repeat", 1)):
            c = dict(case)
            if case.get("repeat", 1) > 1:
                c["name"] = f"{case['name']}_{rep+1}"
            expanded.append(c)
    cases = expanded

    for case in cases:
        name, text = case["name"], case["text"]
        temp = float(case.get("temperature", args.temperature))
        prompt = build_drama3_prompt(text, model.tokenizer, model.config,
                                     prompt_texts, prompt_tokens)
        t0 = time.perf_counter()
        # Same duration sanity check as narrate.py: a draw whose spoken length
        # is far from what the text implies has stalled or babbled, and would
        # pollute a listening comparison with sampling noise rather than
        # telling us anything about the case under test.
        expected = len(text.encode()) / 16.0
        best = None
        for attempt in range(args.max_attempts):
            f, rsn = generate_drama3(model, prompt, temperature=temp,
                                     top_p=float(case.get("top_p", args.top_p)),
                                     top_k=int(case.get("top_k", args.top_k)),
                                     max_new_tokens=args.max_new_tokens)
            if not len(f):
                continue
            cd = f[:, 1:].T.to(device)
            au = decode_to_audio(cd, codec).float().cpu().numpy()
            _, end_i = speech_span(au, SR)
            ratio = end_i / expected if expected else 1.0
            score = abs(1.0 - ratio)
            if best is None or score < best[0]:
                best = (score, f, rsn, cd, au, attempt + 1)
            if expected < 1.5 or 0.55 <= ratio <= 2.2:
                break
        if best is None:
            print(f"{name}: NO FRAMES"); continue
        _, frames, reason, codes, audio, tries = best
        gen = time.perf_counter() - t0
        path = out / f"{name}.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        dur = len(audio) / SR
        a, b = speech_span(audio, SR)
        heard = stt(path)
        r = dict(name=name, text=text, temperature=temp,
                 top_p=float(case.get("top_p", args.top_p)),
                 top_k=int(case.get("top_k", args.top_k)),
                 frames=len(frames), reason=reason,
                 seconds=round(dur, 2), speech_end=round(b, 1),
                 trailing_silence=round(dur - b, 1), gen_seconds=round(gen, 1),
                 attempts=tries,
                 stt=heard)
        results.append(r)
        print(f"\n[{name}] {len(frames)}f {dur:.1f}s  stop={reason}  "
              f"speech ends {b:.1f}s  trailing silence {dur-b:.1f}s")
        print(f"   in : {text[:110]!r}")
        print(f"   out: {heard[:110]!r}", flush=True)

    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    main()
