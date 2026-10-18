"""Generate speech with Drama 3 (experts in RAM) and write a wav you can listen to.

This is the correctness check the speed work has been missing. Everything so
far was measured in tokens/sec, but the offload path rewires how expert weights
reach the GPU -- exactly the kind of change that can produce fast, confident
garbage. If this file sounds wrong, the throughput numbers mean nothing.

It also settles RTF, which the notes deliberately refused to quote: the codec
runs at 44100 / 512 / 4 = 21.53 semantic frames per second (encoder_rates
[2,4,8,8] = 512x, then the quantizer's downsample_factor [2,2] = 4x more), so
RTF 1.0 is 21.53 tok/s.
"""
from __future__ import annotations

import argparse, os, socket, subprocess, sys, time
import pathlib
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch

sys.path.insert(0, str(Path(__file__).parent))
import offload  # installs the expert-offload hook

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt,
    generate_drama3,
    Drama3PromptTooLongError,
)
from fish_speech.models.text2semantic.fish_omni.chunking import (
    split_drama3_text,
    Drama3History,
    DRAMA3_CHUNK_MAX_NEW_TOKENS,
)
from fish_speech.models.text2semantic.inference import (
    load_codec_model,
    encode_audio,
    decode_to_audio,
    resolve_codec_path,
)

FRAME_RATE = 44100 / 512 / 4   # 21.53 Hz -- see docstring


def to_pcm16(src: Path, dst: Path, seconds: float | None) -> None:
    """mp3 -> mono 44.1k PCM_16 wav. The STT daemon reads PCM_16 ONLY."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if seconds:
        cmd += ["-t", str(seconds)]
    cmd += ["-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def transcribe(path: Path) -> str:
    """Ask the resident Parakeet daemon. Empty result = one of its three traps."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock = paths.stt_socket()
    if sock is None:
        return ("<stt unavailable: set DRAMA3_STT to a socket, "
                "or pass --ref-text>")
    s.connect(sock)
    s.sendall((str(path.resolve()) + "\n").encode())
    buf = b""
    while True:
        c = s.recv(65536)
        if not c or b"\n" in buf:
            break
        buf += c
    s.close()
    return buf.decode().strip()


TEXT = (
    "This is Drama 3 running on a twenty four gigabyte card, with every routed "
    "expert living in system memory and fetched over the bus as it speaks."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    ap.add_argument("--ref-text", default=None, help="skip STT and use this transcript")
    ap.add_argument("--ref-seconds", type=float, default=20.0,
                    help="trim the reference; 0 uses all of it")
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--text-file", default=None,
                    help="read the text to speak from a file (UTF-8)")
    ap.add_argument("--no-chunk", action="store_true",
                    help="one shot; fails or truncates on long text")
    ap.add_argument("--out", default="out/drama3.wav")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--copy-mode", choices=("gather", "direct"), default="direct")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    if args.text_file:
        args.text = pathlib.Path(args.text_file).read_text(encoding="utf-8").strip()

    offload.COPY_MODE = args.copy_mode
    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.TRACE.enabled = False          # not tracing here; keep it cheap
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda self, devices: (
        offload.materialize_offloaded(self, device)
    )

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent

    # --- reference voice -------------------------------------------------
    ref_codes = None
    ref_text = args.ref_text
    ref = Path(args.ref)
    if ref.is_file():
        wav = work / "reference.wav"
        to_pcm16(ref, wav, args.ref_seconds or None)
        print(f"reference: {ref}  ->  {wav}")
        if ref_text is None:
            ref_text = transcribe(wav)
            print(f"transcript (Parakeet): {ref_text!r}")
            if not ref_text:
                print("  !! STT returned empty -- continuing WITHOUT a reference voice")
    else:
        print(f"no reference at {ref}; generating in the default voice")

    # --- model -----------------------------------------------------------
    t0 = time.perf_counter()
    model = FishOmniModel.from_pretrained(
        args.checkpoint, devices=[str(device)], tower_device=str(device),
        dtype=torch.bfloat16,
    )
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)), device=device)
    print(f"model + codec loaded in {time.perf_counter()-t0:.1f}s; "
          f"VRAM {torch.cuda.memory_allocated(device)/2**30:.2f} GiB")

    prompt_texts, prompt_tokens = (), ()
    if ref_text:
        ref_codes = encode_audio(work / "reference.wav", codec, device)
        print(f"reference codes: {tuple(ref_codes.shape)} "
              f"({ref_codes.shape[1]/FRAME_RATE:.1f}s at {FRAME_RATE:.2f} Hz)")
        prompt_texts, prompt_tokens = (ref_text,), (ref_codes,)

    prompt = build_drama3_prompt(
        args.text, model.tokenizer, model.config, prompt_texts, prompt_tokens
    )
    print(f"prompt rows: {len(prompt)}")

    # --- generate ---------------------------------------------------------
    import numpy as np
    chunks = [args.text] if args.no_chunk else split_drama3_text(args.text)
    print(f"{len(chunks)} chunk(s)")
    history = Drama3History()
    pieces, total_frames = [], 0
    t0 = time.perf_counter()

    for index, chunk in enumerate(chunks, 1):
        # candidates() yields histories longest-first; drop context until the
        # prompt fits the 8192-row cap rather than failing outright.
        for turns in history.candidates():
            try:
                prompt = build_drama3_prompt(
                    chunk, model.tokenizer, model.config,
                    prompt_texts, prompt_tokens, history=turns,
                )
                break
            except Drama3PromptTooLongError:
                continue
        else:
            print(f"  chunk {index}: cannot fit even with no history; skipped")
            continue

        budget = args.max_new_tokens if args.no_chunk else DRAMA3_CHUNK_MAX_NEW_TOKENS
        frames, reason = generate_drama3(model, prompt, max_new_tokens=budget)
        if not len(frames):
            print(f"  chunk {index}: produced nothing; skipped")
            continue
        codes = frames[:, 1:].T.to(device)       # [frames, 1+N] -> [N, frames]
        history.push(chunk, codes.cpu())
        pieces.append(decode_to_audio(codes, codec).float().cpu().numpy())
        total_frames += len(frames)
        print(f"  chunk {index}/{len(chunks)}: {len(frames)} frames "
              f"({len(frames)/FRAME_RATE:.1f}s, {reason}) {chunk[:60]!r}")

    gen_s = time.perf_counter() - t0
    if not pieces:
        print("!! nothing generated"); return
    audio = np.concatenate(pieces)
    frames = range(total_frames)
    reason = "chunked"

    import soundfile as sf
    sf.write(str(out), audio, 44100, subtype="PCM_16")

    dur = len(audio) / 44100
    print(f"\n{len(frames)} frames in {gen_s:.1f}s ({reason})")
    print(f"  {len(frames)/gen_s:.2f} tok/s   audio {dur:.2f}s")
    print(f"  RTF {gen_s/dur:.2f}  (1.0 = realtime; {FRAME_RATE:.2f} tok/s)")
    print(f"\nwrote {out.resolve()}")


if __name__ == "__main__":
    main()
