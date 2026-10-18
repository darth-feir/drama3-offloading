"""Generate a whole document ahead of time into say's audio cache.

`say` reuses any audio already saved for a paragraph before it calls a
backend, so prebuilding sidesteps the 8-second prefetch lead entirely: by the
time you press a key, every paragraph is already on disk. At RTF ~1.9 that is
the difference between usable and not.

Input is whatever `say -engine drama3 -chunks FILE` prints: one object per
paragraph carrying the EXACT text that was hashed and the absolute path the
audio belongs at. We never recompute the fingerprint or re-split the text --
say's chunker collapses whitespace and splits long paragraphs its own way,
and any disagreement would write files say never looks at.

⚠️ One wrinkle that matters. say's chunker joins a trailing heading onto the
end of a paragraph ("...objectives and goals. Use case scenario"), and that
is precisely the shape Drama 3 drops content on -- measured: it speaks the
sentence, omits the heading, and stalls into silence. We cannot change the
text, because the text IS the cache key. So we split it only INTERNALLY,
generate each piece, and concatenate them into the single file say expects.
Same bytes on the key, correct audio inside.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch, numpy as np, offload
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, generate_drama3, Drama3PromptTooLongError)
from fish_speech.models.text2semantic.fish_omni.chunking import (
    split_drama3_text, Drama3History)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, decode_to_audio, resolve_codec_path)

SR = 44100
FRAME_RATE = 44100 / 512 / 4
BYTES_PER_SECOND = 16.0

# narrate.py guards its main, so importing it is safe and far less brittle
# than slicing source text out of it (which silently lost frame_budget once).
import narrate
clamp_internal_silence = narrate.clamp_internal_silence
speech_end = narrate.speech_end
frame_budget = narrate.frame_budget


def internal_pieces(text: str, max_bytes: int) -> list[str]:
    """Split a say chunk for generation only; the file stays one file.

    say has already collapsed newlines, so the heading/bullet boundary that
    trips the model is no longer marked by \\n. Fall back to sentence-aware
    splitting when the chunk is long enough to be worth it.
    """
    if len(text.encode()) <= max_bytes:
        return [text]
    return split_drama3_text(text, max_bytes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True,
                    help="JSON from: say -engine drama3 -chunks FILE")
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    ap.add_argument("--ref-seconds", type=float, default=20.0)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--slice-lm-head", action="store_true", default=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--piece-bytes", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dest-root", default=None,
                    help="write under here instead of the absolute 'file' path "
                         "(for generating on this box for another machine)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())

    chunks = json.loads(Path(args.chunks).read_text())
    if args.limit:
        chunks = chunks[: args.limit]
    todo = [c for c in chunks if not c.get("exists")]
    backends = {c.get("backend") for c in chunks}
    print(f"{len(chunks)} chunks, {len(todo)} to generate, backend(s): {backends}")
    if "elevenlabs" in backends:
        print("  !! backend is 'elevenlabs' -- the fingerprint has NO prefix for it.")
        print("     Re-run say with -engine drama3 or the files will be written")
        print("     under the wrong key.")
    est = sum(len(c["text"].encode()) for c in todo) / BYTES_PER_SECOND
    print(f"  ~{est/60:.1f} min of audio, ~{est*1.9/60:.0f} min at RTF 1.9")

    def dest(c) -> Path:
        p = Path(c["file"])
        if not args.dest_root:
            return p
        # keep the say-<doc>/audio/<fp>.wav tail, re-root the rest
        return Path(args.dest_root) / Path(*p.parts[-3:])

    if args.dry_run:
        for c in todo[:5]:
            print(f"  [{c['n']}/{c['of']}] -> {dest(c)}")
            print(f"      {c['text'][:90]!r}")
        return

    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
    offload.COPY_MODE = "direct"; offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda s, d: offload.materialize_offloaded(s, device)

    t0 = time.perf_counter()
    model = FishOmniModel.from_pretrained(args.checkpoint, devices=[str(device)],
                                          tower_device=str(device), dtype=torch.bfloat16)
    if args.slice_lm_head:
        print(f"lm_head sliced, freed {offload.slice_lm_head(model, device):.2f} GiB")
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)), device=device)
    if args.vram_reserve:
        cap = offload.size_cache_to_free_vram(model, device, args.vram_reserve)
        print(f"expert cache {cap}/layer")
    print(f"loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    prompt_texts, prompt_tokens = (), ()
    ref = Path(args.ref)
    if ref.is_file():
        wav = Path("out") / "prebuild-reference.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ref), "-t",
                        str(args.ref_seconds), "-ac", "1", "-ar", "44100",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        rt = args.ref_text
        if rt is None:
            import socket
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock = paths.stt_socket()
            if sock is None:
                return ("<stt unavailable: set DRAMA3_STT to a socket, "
                        "or pass --ref-text>")
            s.connect(sock)
            s.sendall((str(wav.resolve()) + "\n").encode())
            buf = b""
            while True:
                ch = s.recv(65536)
                if not ch or b"\n" in buf: break
                buf += ch
            s.close(); rt = buf.decode().strip()
        if rt:
            prompt_texts, prompt_tokens = (rt,), (encode_audio(wav, codec, device),)
            print(f"reference: {prompt_tokens[0].shape[1]} codes "
                  f"({prompt_tokens[0].shape[1]/FRAME_RATE:.1f}s)")

    import soundfile as sf
    started, done, made = time.perf_counter(), 0, 0
    for c in todo:
        out = dest(c)
        if out.is_file():
            done += 1; continue
        out.parent.mkdir(parents=True, exist_ok=True)
        pieces_txt = internal_pieces(c["text"], args.piece_bytes)
        history = Drama3History()
        audio_parts = []
        ok = True
        for piece in pieces_txt:
            expected = len(piece.encode()) / BYTES_PER_SECOND
            best = None
            for attempt in range(args.max_attempts):
                prompt = None
                for turns in history.candidates():
                    try:
                        prompt = build_drama3_prompt(piece, model.tokenizer, model.config,
                                                     prompt_texts, prompt_tokens, history=turns)
                        break
                    except Drama3PromptTooLongError:
                        continue
                if prompt is None:
                    ok = False; break
                f, _ = generate_drama3(model, prompt, temperature=args.temperature,
                                       max_new_tokens=frame_budget(piece, 600),
                                       prefill_chunk_size=args.prefill_chunk)
                if not len(f):
                    continue
                cd = f[:, 1:].T.to(device)
                au = decode_to_audio(cd, codec).float().cpu().numpy()
                end = speech_end(au, SR)
                ratio = end / expected if expected else 1.0
                score = abs(1.0 - ratio)
                if best is None or score < best[0]:
                    best = (score, cd, au, end)
                if expected < 1.5 or 0.55 <= ratio <= 2.2:
                    break
            if best is None:
                ok = False; break
            _, cd, au, end = best
            history.push(piece, cd.cpu())
            au = au[: min(len(au), int((end + 0.15) * SR))] if end > 0 else au
            audio_parts.append(clamp_internal_silence(au, SR, args.max_gap))
        if not ok or not audio_parts:
            print(f"  [{c['n']}/{c['of']}] FAILED {c['text'][:50]!r}", flush=True)
            continue
        sf.write(str(out), np.concatenate(audio_parts), SR, subtype="PCM_16")
        made += 1; done += 1
        el = time.perf_counter() - started
        eta = (len(todo) - done) * el / max(made, 1)
        print(f"  [{c['n']}/{c['of']}] {len(pieces_txt)}pc "
              f"{sum(len(a) for a in audio_parts)/SR:5.1f}s  eta {eta/60:.0f}min  "
              f"{out.name}", flush=True)

    print(f"\n{made} files written, {len(todo)-made} skipped or failed")
    print(f"total {(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
