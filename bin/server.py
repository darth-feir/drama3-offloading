"""Resident HTTP server: one model load, many requests. Speaks `say`'s
generic `http` backend protocol.

WHY A SERVER NOW, WHEN notes/05 SAID PREBUILD
---------------------------------------------
`notes/05-say-integration.md` concluded that the live-backend path was
"gappy at RTF 1.9" and that prebuilding a whole document ahead of time was
the only option that fitted. **That was measured at RTF 1.9. It is now
0.851** (`notes/12-full-run.md`), so generation outruns playback and the
8-second prefetch lead `say` gives a backend is comfortable rather than
marginal. The old conclusion is stale; both paths now work.

  prebuild   bin/prebuild.py, a whole document ahead of time. Still the
             right answer for a long document you will read more than
             once, because nothing is generated at read time at all.
  this       ad-hoc text, no prepared cache, nothing to run first.

STARTUP IS THE WHOLE POINT
--------------------------
Measured: 32.5 s to load the model, page-lock 30 GiB of experts, load the
codec, encode the reference and build the prefix KV — plus ~2.5 s of
CUDA-graph capture on the first request. Paying that per paragraph would be
absurd, which is why this is a resident process rather than a CLI.

The prefix KV snapshot is built once at startup and restored per request,
so the reference clip is never re-read. See notes/09-prefix-kv.md.

PROTOCOL
--------
`say`'s `http` backend POSTs a JSON body built from a template and expects
**raw audio bytes** back; for `"format": "wav"` it reads the sample rate
from the header (`rest.go`). So:

    POST /tts   {"text": "...", "voice": "...", "speed": 1.0}
    -> 200, audio/wav, 44.1 kHz mono PCM_16

    GET  /health -> 200 {"ready": true, ...}

`voice` and `speed` are accepted and ignored: the voice is the reference
clip this process was started with, and speed is not a knob the model
exposes. They are accepted rather than rejected so a stock `say` config
template works unmodified.

  bin/run.sh bin/server.py --ref CLIP.wav --ref-text "what it says"
  bin/drama3ctl start|stop|status|log      <- normally use this instead
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import numpy as np
import torch

import narrate
import offload
import prefix_kv
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, decode_to_audio, resolve_codec_path,
)

SR = narrate.SR
STATE: dict = {"ready": False, "requests": 0, "audio_seconds": 0.0,
               "generate_seconds": 0.0, "started": time.time()}
# One GPU, one model, one set of KV caches. Two concurrent generations would
# interleave writes into the same cache buffers and produce two wrong
# answers, so requests are serialised. ThreadingHTTPServer is still used so
# a /health check cannot be blocked behind a long generation.
GPU = threading.Lock()


def internal_pieces(text: str, max_bytes: int) -> list[str]:
    """Split a paragraph for generation only; the caller's text is unchanged.

    ⚠️ Mirrors bin/prebuild.py. say's chunker joins a trailing heading onto
    the previous paragraph, which is exactly the shape Drama 3 drops content
    on (notes/03). The text is the cache key over there and the request body
    here, so it must not be altered -- split internally, concatenate the
    audio, return one wav.
    """
    if len(text.encode()) <= max_bytes:
        return [text]
    return split_drama3_text(text, max_bytes)


def to_wav_bytes(audio: np.ndarray) -> bytes:
    """44.1 kHz mono PCM_16 in a buffer, with a true size field.

    ⚠️ PCM_16 and a real header are both load-bearing. The STT daemon used
    for scoring reads PCM_16 only and trusts the header; a streamed wav with
    an indeterminate size (0xFFFFFFFF) transcribes as empty while identical
    audio with a true header is fine. say reads the header too.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Engine:
    """The model, loaded once."""

    def __init__(self, args) -> None:
        self.args = args
        device = torch.device(args.device)
        self.device = device
        os.environ["FISH_DRAMA3_CUDA_GRAPH"] = args.cuda_graph
        offload.COPY_MODE = "direct"
        offload.CACHE_EXPERTS = 0
        offload.HYBRID_GATHER = True
        offload.TRACE.enabled = False
        offload.install_offload()
        import fish_speech.models.text2semantic.fish_omni.backbone as bb
        bb.FishOmniBackbone.materialize = lambda self_, devices: (
            offload.materialize_offloaded(self_, device))

        started = time.perf_counter()
        self.model = FishOmniModel.from_pretrained(
            args.checkpoint, devices=[str(device)], tower_device=str(device),
            dtype=torch.bfloat16)
        if args.pin_experts:
            gib, secs = offload.pin_experts(self.model, device)
            log(f"pinned {gib:.1f} GiB in {secs:.0f}s")
        if args.slice_lm_head:
            log(f"lm_head sliced; freed {offload.slice_lm_head(self.model, device):.2f} GiB")
        if args.offload_embed:
            log(f"embed_tokens to host RAM; freed {offload.offload_embed(self.model, device):.2f} GiB")
        self.codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)),
                                      device=device, precision=torch.bfloat16)

        import devcache
        devcache.FETCH_BACKEND = args.fetch_backend
        cap = offload.size_cache_to_free_vram(self.model, device, args.vram_reserve)
        for layer in self.model.backbone.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and getattr(experts, "_cap", 0):
                experts._cap = 0
                for name in ("_cache_w13", "_cache_w2", "_cache_s13", "_cache_s2"):
                    if hasattr(experts, name):
                        delattr(experts, name)
        torch.cuda.empty_cache()
        cap = devcache.install_dev_cache(self.model, device, cap)
        log(f"device cache: {cap} experts/layer")

        # The reference, encoded once.
        wav = Path(args.state_dir) / "reference.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        seconds = args.ref_seconds
        if seconds and not args.no_snap:
            seconds = narrate.snap_to_silence(Path(args.ref), seconds)
        narrate.to_pcm16(Path(args.ref), wav, seconds)
        ref_text = args.ref_text or narrate.transcribe(wav)
        if not ref_text or ref_text.startswith("<stt"):
            raise SystemExit(
                f"no reference transcript ({ref_text!r}); pass --ref-text. "
                "It is worth writing by hand: ASR errors in the reference are "
                "inherited by every paragraph this server ever speaks.")
        self.prompt_texts = (ref_text,)
        self.prompt_tokens = (encode_audio(wav, self.codec, device),)
        log(f"reference: {self.prompt_tokens[0].shape[1]} codes "
            f"({self.prompt_tokens[0].shape[1]/narrate.FRAME_RATE:.1f}s)")

        self.prefix = None
        if args.prefix_kv:
            self.prefix = prefix_kv.make_prefix(
                self.model, self.prompt_texts, self.prompt_tokens, args.prefill_chunk)
            log(f"prefix KV: {self.prefix.P} rows reused, "
                f"{self.prefix.bytes/2**20:.0f} MiB")

        # ⚠️ Warm up HERE, on the main thread, before the server starts.
        #
        # CUDA-graph capture has to happen somewhere, and if it happens
        # inside a ThreadingHTTPServer worker it does not complete in any
        # reasonable time -- measured: >3 minutes and still going, at 75 %
        # CPU with the GPU at 8-12 %, against 2.2 s on the main thread. The
        # model and its caches are created here; capturing from a different
        # thread is not something to rely on.
        #
        # Doing it now also means the first real request is not three
        # minutes slower than the rest, which `say` would experience as the
        # backend hanging on the opening paragraph.
        log("warming up (CUDA-graph capture)...")
        warm = time.perf_counter()
        self.speak("Warm up.")
        log(f"warm-up done in {time.perf_counter()-warm:.1f}s")

        STATE["load_seconds"] = round(time.perf_counter() - started, 1)
        STATE["ready"] = True
        log(f"ready in {STATE['load_seconds']:.0f}s")

    def speak(self, text: str) -> np.ndarray:
        """One paragraph in, one waveform out. Caller holds GPU."""
        args = self.args
        pieces = internal_pieces(text, args.para_max_bytes)
        out = []
        for piece in pieces:
            prompt = build_drama3_prompt(piece, self.model.tokenizer,
                                         self.model.config,
                                         self.prompt_texts, self.prompt_tokens)
            budget = narrate.frame_budget(piece, args.max_new_tokens)
            expected = len(piece.encode()) / narrate.BYTES_PER_SECOND
            best = None
            for attempt in range(args.max_attempts):
                if attempt:
                    budget = args.max_new_tokens
                if self.prefix is not None:
                    frames, _ = prefix_kv.generate_drama3_prefix(
                        self.model, prompt, self.prefix, max_new_tokens=budget,
                        temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, prefill_chunk_size=args.prefill_chunk)
                else:
                    from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3
                    frames, _ = generate_drama3(
                        self.model, prompt, max_new_tokens=budget,
                        temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k, prefill_chunk_size=args.prefill_chunk)
                if not len(frames):
                    continue
                audio = decode_to_audio(frames[:, 1:].T.to(self.device),
                                        self.codec).float().cpu().numpy()
                end = narrate.speech_end(audio, SR)
                ratio = end / expected if expected else 1.0
                score = abs(1.0 - ratio)
                if best is None or score < best[0]:
                    best = (score, audio, end)
                if expected < 1.5 or args.min_ratio <= ratio <= args.max_ratio:
                    break
            if best is None:
                raise RuntimeError("no frames generated")
            _, audio, end = best
            if end > 0:
                audio = audio[: min(len(audio), int((end + 0.15) * SR))]
            audio = narrate.clamp_internal_silence(
                audio, SR, args.max_gap, args.long_gap, narrate.pause_budget(piece))
            out.append(audio)
        return np.concatenate(out) if len(out) > 1 else out[0]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # set in main

    def log_message(self, fmt, *a):      # quieter than the default
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/healthz", ""):
            uptime = time.time() - STATE["started"]
            rtf = (STATE["generate_seconds"] / STATE["audio_seconds"]
                   if STATE["audio_seconds"] else None)
            self._json(200, {**STATE, "uptime_seconds": round(uptime, 1),
                             "rtf": round(rtf, 3) if rtf else None})
        else:
            self._json(404, {"error": "GET /health, POST /tts"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/tts", "/tts_to_audio"):
            self._json(404, {"error": "POST /tts"})
            return
        try:
            length = int(self.headers.get("content-length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("text") or "").strip()
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return
        if not text:
            self._json(400, {"error": "no text"})
            return
        started = time.perf_counter()
        try:
            # Serialised: one model, one set of KV caches. Concurrent
            # generations would interleave into the same buffers.
            with GPU:
                audio = self.engine.speak(text)
        except Exception as exc:
            log(f"FAILED {len(text.encode())}B: {type(exc).__name__}: {exc}")
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        elapsed = time.perf_counter() - started
        seconds = len(audio) / SR
        STATE["requests"] += 1
        STATE["audio_seconds"] = round(STATE["audio_seconds"] + seconds, 2)
        STATE["generate_seconds"] = round(STATE["generate_seconds"] + elapsed, 2)
        log(f"{len(text.encode()):5d}B -> {seconds:6.2f}s audio in "
            f"{elapsed:6.2f}s  rtf {elapsed/max(seconds,1e-9):.2f}")
        self._send(200, to_wav_bytes(audio), "audio/wav")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--ref-seconds", type=float, default=56.0)
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--para-max-bytes", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--cuda-graph", choices=("0", "1"), default="1")
    ap.add_argument("--pin-experts", action="store_true", default=True)
    ap.add_argument("--no-pin-experts", dest="pin_experts", action="store_false")
    ap.add_argument("--slice-lm-head", action="store_true", default=True)
    ap.add_argument("--offload-embed", action="store_true", default=True)
    ap.add_argument("--prefix-kv", action="store_true", default=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=4097)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--min-ratio", type=float, default=0.55)
    ap.add_argument("--max-ratio", type=float, default=2.2)
    ap.add_argument("--max-gap", type=float, default=0.5)
    ap.add_argument("--long-gap", type=float, default=1.2)
    args = ap.parse_args()
    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    args.state_dir = args.state_dir or str(paths.HOME / "run")

    Handler.engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"listening on http://{args.host}:{args.port}  (POST /tts, GET /health)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")


if __name__ == "__main__":
    main()
