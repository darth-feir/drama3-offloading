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

  speed      honoured: ffmpeg atempo, 0.25-4.0, pitch preserved.
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

import config
import narrate
import offload
CORS = os.environ.get("DRAMA3_CORS", "").strip()

import prefix_kv
from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, decode_to_audio, resolve_codec_path,
)

SR = narrate.SR
STATE: dict = {"ready": False, "requests": 0, "audio_seconds": 0.0,
               "generate_seconds": 0.0, "verify_seconds": 0.0, "verify_calls": 0,
               "retries": 0, "gave_up": 0, "started": time.time()}
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


def stt_similarity(audio: np.ndarray, text: str, scratch: Path,
                   min_words: int = 8, tail_fraction: float = 0.25
                   ) -> tuple[float | None, str]:
    """Word-level similarity between what was asked for and what came out.

    Returns the LOWER of the whole-piece score and the tail score -- see the
    comment at the bottom for why the tail is scored separately.

    ⚠️ This exists because the duration check CANNOT catch the failure it is
    meant to catch. A bad draw speaks gibberish for about as long as the real
    words take, so `0.55 <= ratio <= 2.2` passes it on the first attempt and
    no retry happens. Observed on a book: "der pausbomwerk denisian" in place
    of "their world", at the right length, scoring a perfectly acceptable
    duration ratio.

    STT is not a transcript oracle -- it normalises numbers, drops
    punctuation and mangles proper nouns, so good audio lands around 0.95
    rather than 1.00. The threshold only has to separate that from a draw
    that has stopped speaking English at all, which scores ~0.6 or below.

    Returns None when no STT socket is configured, in which case the caller
    keeps the old duration-only behaviour.
    """
    if paths.stt_socket() is None:
        return None, ""
    # ⚠️ NOT /tmp. The STT daemon runs PrivateTmp=true, so a wav under /tmp
    # is invisible to it and comes back empty -- which would look exactly
    # like the gibberish this is trying to detect.
    wav = scratch / "_verify.wav"
    wav.write_bytes(to_wav_bytes(audio))
    out = scratch / "_verify16.wav"
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                        str(out)], check=True)
        heard = narrate.transcribe(out)
    except Exception:
        return None, ""
    finally:
        wav.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
    if not heard or heard.startswith("<stt"):
        return None, ""
    import re
    from difflib import SequenceMatcher
    w = re.findall(r"[a-z0-9]+", text.lower())
    g = re.findall(r"[a-z0-9]+", heard.lower())
    # ⚠️ Too short to judge. STT similarity on a couple of words is noise:
    # the 8-byte warm-up text "Warm up." scored 0.44 and triggered a
    # spurious regeneration. The duration check has the same bypass
    # (expected < 1.5 s) for the same reason -- a short utterance cannot
    # carry enough signal to separate a bad draw from ASR variance.
    if len(w) < min_words:
        return None, heard
    overall = SequenceMatcher(None, w, g).ratio()

    # ⚠️ Also score the TAIL on its own. Similarity over a whole piece is
    # diluted by length: a 196-word chunk from the book containing clear
    # gibberish ("a gotiwerkte lartes setting passanting", "tecutu") still
    # scored 0.87, because only ~25 words were wrong. And the tail is
    # exactly where this fails -- the report that started this was
    # "gibberish at the end of longer paragraphs".
    #
    # The last quarter, minimum 8 words, so a short piece is judged whole.
    tail_n = max(min_words, int(len(w) * tail_fraction))
    if len(w) > tail_n:
        tail = SequenceMatcher(None, w[-tail_n:], g[-tail_n:]).ratio()
        return min(overall, tail), heard
    return overall, heard


# What a client may ask for, and what ffmpeg needs to produce it. wav is
# native; everything else is a transcode of it.
FORMATS = {
    "wav": ("audio/wav", None),
    "pcm": ("audio/L16", None),           # headerless 16-bit LE, 44.1 kHz
    "mp3": ("audio/mpeg", ["-f", "mp3", "-b:a", "128k"]),
    "flac": ("audio/flac", ["-f", "flac"]),
    "opus": ("audio/opus", ["-f", "opus", "-b:a", "96k"]),
    "aac": ("audio/aac", ["-f", "adts", "-b:a", "128k"]),
}


def atempo_factors(speed: float) -> list[float]:
    """atempo accepts 0.5-2.0 per instance; OpenAI allows 0.25-4.0, so
    anything outside that range becomes a chain whose product is `speed`."""
    factors, remaining = [], speed
    while remaining > 2.0:
        factors.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5); remaining /= 0.5
    factors.append(remaining)
    return factors


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change playback rate, preserve pitch. Returns samples, not a file.

    ⚠️ Deliberately PCM in and PCM out. The obvious implementation -- ask
    ffmpeg for wav on stdout -- produces a wav whose size field is
    0xFFFFFFFF, because ffmpeg cannot seek back on a pipe to patch the
    header. Readers then report absurd durations (measured: 48695 s for a
    6.6 s clip) or treat it as empty. That failure is already written up in
    to_wav_bytes() and it was reintroduced here anyway.

    Staying in samples also means the format encoder below is unchanged
    whether or not speed was applied.
    """
    if abs(speed - 1.0) < 1e-3:
        return audio
    import subprocess
    pcm_in = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    done = subprocess.run(
        ["ffmpeg", "-v", "error",
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-filter:a", ",".join(f"atempo={f:.6f}" for f in atempo_factors(speed)),
         "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=pcm_in, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg speed={speed}: {done.stderr.decode()[:200]}")
    return np.frombuffer(done.stdout, dtype="<i2").astype(np.float32) / 32768.0


def encode(audio: np.ndarray, fmt: str, speed: float = 1.0) -> tuple[bytes, str]:
    """Return (bytes, content-type) in the requested format and speed.

    ⚠️ `speed` is APPLIED, not ignored. It used to be accepted and
    discarded, which is the worse half of the silent-failure category: a
    server that rejects an unimplemented parameter tells the client, one
    that accepts it does not. Measured by say's author against this
    server, the same text at 1.0 and 1.5 differed by 5 % -- run-to-run
    variance, not speed.

    The damage was not the missing feature. `speed` is part of a reading
    client's CACHE KEY, so changing it invalidated a whole document,
    regenerated every paragraph, and returned byte-identical audio.

    The model has no speed control, so this is a post-process. That is
    enough to make the parameter true, which is what a cache key needs.
    """
    audio = apply_speed(audio, speed)
    ctype, args = FORMATS[fmt]
    body = to_wav_bytes(audio)           # always a real header
    if fmt == "pcm":
        return body[44:], ctype          # strip the header we just wrote
    if args is None:
        return body, ctype
    import subprocess
    done = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", *args, "pipe:1"],
                          input=body, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg {fmt}: {done.stderr.decode()[:200]}")
    return done.stdout, ctype


class Engine:
    """The model, loaded once."""

    def __init__(self, args, cfg) -> None:
        self.args = args
        self.cfg = cfg
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
        seconds = cfg.voice.seconds
        if seconds and cfg.voice.snap_to_silence:
            seconds = narrate.snap_to_silence(Path(args.ref), seconds)
        narrate.to_pcm16(Path(args.ref), wav, seconds)
        ref_text = args.ref_text or cfg.voice.ref_text or narrate.transcribe(wav)
        if not ref_text or ref_text.startswith("<stt"):
            raise SystemExit(
                f"no reference transcript ({ref_text!r}); pass --ref-text, "
                "or set DRAMA3_STT and it will be transcribed for you. "
                "An approximate transcript is far better than none: a wrong "
                "one measured no worse on timbre or content (notes/17), but "
                "the model still needs some text to align the clip against.")
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
            # The clip this process was started with is itself a voice, so
            # one code path serves both; --ref is just the unnamed one.
            self.prefix.prompt_texts = self.prompt_texts
            self.prefix.prompt_tokens = self.prompt_tokens

        # Named voices from voices/. Each is ~118 MiB of host RAM against
        # 30 GiB of pinned experts, so holding several costs nothing.
        #
        # ⚠️ A voice that fails its fingerprint is SKIPPED WITH A REASON,
        # never silently restored. Restoring state built against different
        # weights or a different cache layout produces plausible audio in
        # the wrong voice, which no downstream check would catch.
        self.voices: dict = {}
        if args.prefix_kv and args.voices:
            import voices as voicelib
            for name in voicelib.names():
                loaded, why = voicelib.load_into(
                    name, self.model, prefix_kv.PrefixKV,
                    args.checkpoint, args.prefill_chunk)
                if loaded is None:
                    log(f"voice {name!r} SKIPPED -- {why}")
                    continue
                self.voices[name] = loaded
                log(f"voice {name!r}: {loaded.P} rows, "
                    f"{loaded.bytes/2**20:.0f} MiB")
            self.default_voice = voicelib.get_default()
            if self.voices:
                log(f"voices: {', '.join(sorted(self.voices))}"
                    f"  default={self.default_voice or '--ref'}")
        self.default_voice = getattr(self, "default_voice", None)

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

    def add_voice(self, name: str, clip: str, ref_text: str | None,
                  ref_seconds: float | None) -> dict:
        """Build and register a voice using the model already loaded here.

        ⚠️ The caller must hold the GPU lock. Prefilling writes into the
        same KV caches a generation uses, so this is not something to run
        beside one -- but it IS something to queue behind one, which is
        why it no longer needs a restart. Measured: ~3 s, against ~70 s to
        stop and start the server.
        """
        import voices as voicelib
        prefix, info = voicelib.build(
            self.model, self.codec, self.device, name, clip,
            ref_text=ref_text,
            ref_seconds=ref_seconds or self.cfg.voice.seconds,
            prefill_chunk=self.args.prefill_chunk,
            checkpoint=self.args.checkpoint,
            transcribe=narrate.transcribe)
        self.voices[name] = prefix
        if self.default_voice is None:
            self.default_voice = name
            voicelib.set_default(name)
        return info

    def drop_voice(self, name: str) -> dict:
        """Unload a voice and delete it from disk.

        ⚠️ Unloads from memory too. Deleting only the files would leave the
        running server still answering for a voice the user believes is
        gone -- which, for state derived from somebody's recorded voice, is
        the wrong way round to fail.
        """
        import voices as voicelib
        if name not in self.voices and name not in voicelib.names():
            raise ValueError(f"no voice named {name!r}; have: "
                             f"{', '.join(sorted(self.voices)) or '(none)'}")
        self.voices.pop(name, None)
        voicelib.remove(name)
        if self.default_voice == name:
            self.default_voice = voicelib.get_default()
        return {"removed": name,
                "voices": sorted(self.voices),
                "default": self.default_voice}

    def pick_voice(self, name: str | None):
        """Resolve a request's `voice` to a prefix, or raise for a bad name.

        Falls back to the process's own --ref clip, so a server started the
        old way behaves exactly as before and `voice` stays optional.
        """
        if name and name in self.voices:
            return self.voices[name]
        if name and name not in ("drama3", "default", ""):
            known = ", ".join(sorted(self.voices)) or "(none registered)"
            raise ValueError(f"unknown voice {name!r}; have: {known}")
        if self.default_voice and self.default_voice in self.voices:
            return self.voices[self.default_voice]
        # ⚠️ Mirror voices.get_default(): one registered voice IS the
        # default. Without this the server fell back to its --ref clip
        # whenever an explicit default had been cleared -- e.g. after
        # removing the voice that held it -- so /v1/voices advertised a
        # voice that requests were not actually using.
        if len(self.voices) == 1:
            return next(iter(self.voices.values()))
        return self.prefix

    def speak(self, text: str, voice: str | None = None) -> np.ndarray:
        """One paragraph in, one waveform out. Caller holds GPU."""
        args, cfg = self.args, self.cfg
        dur, con = cfg.checks.duration, cfg.checks.content
        chosen = self.pick_voice(voice)
        # ⚠️ The prompt MUST be built from the same reference as the prefix
        # it will resume from: generate_drama3_prefix checks that the
        # prompt's head equals the snapshot's rows, so mixing a voice's
        # prefix with another's prompt is a hard failure, not a subtle one.
        texts = getattr(chosen, "prompt_texts", self.prompt_texts)
        tokens = getattr(chosen, "prompt_tokens", self.prompt_tokens)
        pieces = internal_pieces(text, cfg.splitting.para_max_bytes)
        out = []
        for piece in pieces:
            prompt = build_drama3_prompt(piece, self.model.tokenizer,
                                         self.model.config, texts, tokens)
            budget = narrate.frame_budget(piece, cfg.splitting.max_new_tokens)
            expected = len(piece.encode()) / narrate.BYTES_PER_SECOND
            best = None
            for attempt in range(cfg.checks.max_attempts):
                if attempt:
                    budget = cfg.splitting.max_new_tokens
                if chosen is not None:
                    frames, _ = prefix_kv.generate_drama3_prefix(
                        self.model, prompt, chosen, max_new_tokens=budget,
                        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
                        top_k=cfg.sampling.top_k,
                        prefill_chunk_size=args.prefill_chunk)
                else:
                    from fish_speech.models.text2semantic.fish_omni.generate import generate_drama3
                    frames, _ = generate_drama3(
                        self.model, prompt, max_new_tokens=budget,
                        temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p,
                        top_k=cfg.sampling.top_k,
                        prefill_chunk_size=args.prefill_chunk)
                if not len(frames):
                    continue
                audio = decode_to_audio(frames[:, 1:].T.to(self.device),
                                        self.codec).float().cpu().numpy()
                end = narrate.speech_end(audio, SR)
                ratio = end / expected if expected else 1.0
                # Provisional score on duration alone; refined below if the
                # content check runs. LOWER IS BETTER.
                score = abs(1.0 - ratio)
                duration_ok = (not dur.enabled or expected < dur.min_seconds
                               or dur.min_ratio <= ratio <= dur.max_ratio)
                similarity = None
                if duration_ok and con.enabled:
                    t_v = time.perf_counter()
                    similarity, heard = stt_similarity(
                        audio, piece, Path(args.state_dir),
                        con.min_words, con.tail_fraction)
                    STATE["verify_seconds"] = round(
                        STATE["verify_seconds"] + time.perf_counter() - t_v, 2)
                    STATE["verify_calls"] += 1
                if similarity is not None:
                    # Content dominates duration: a draw that says the right
                    # words slightly too fast beats one that says the wrong
                    # words at the right pace.
                    score = 2.0 * (1.0 - similarity) + abs(1.0 - ratio)
                # ⚠️ COMPARE, do not overwrite. This assignment used to be
                # unconditional, so when every attempt failed verification the
                # LAST one was delivered rather than the best -- which is the
                # opposite of what you want from a give-up path.
                if best is None or score < best[0]:
                    best = (score, audio, end, similarity)
                if similarity is not None and similarity < con.min_similarity:
                    STATE["retries"] += 1
                    # ⚠️ Log BOTH tails. The score alone cannot tell a real
                    # failure from ASR mangling a technical term at the end of
                    # a piece -- "BACPAC" heard as "backpack" scores the same
                    # as genuine nonsense, and the only way to tell is to read
                    # them side by side.
                    log(f"  retry {attempt+1}: sim {similarity:.2f} "
                        f"({len(piece.encode())}B)")
                    log(f"     want ...{piece[-70:]!r}")
                    log(f"     got  ...{heard[-70:]!r}")
                    continue
                if duration_ok:
                    break
            if best is None:
                raise RuntimeError("no frames generated")
            _, audio, end, best_sim = best
            # Attempts are bounded (--max-attempts, default 3). When they are
            # all used up this DELIVERS the best draw rather than failing or
            # looping: an imperfect paragraph beats a hole in the audiobook.
            # It says so in the log, so a document full of these is visible
            # rather than silent.
            if (con.enabled and best_sim is not None
                    and best_sim < con.min_similarity):
                STATE["gave_up"] += 1
                log(f"  GAVE UP after {cfg.checks.max_attempts} attempts, best was "
                    f"{best_sim:.2f} -- delivering it anyway")
            trim = cfg.post.trim_trailing_silence
            if trim.enabled and end > 0:
                audio = audio[: min(len(audio), int((end + trim.pad_seconds) * SR))]
            clamp = cfg.post.clamp_internal_silence
            if clamp.enabled:
                audio = narrate.clamp_internal_silence(
                    audio, SR, clamp.max_gap, clamp.long_gap,
                    narrate.pause_budget(piece))
            out.append(audio)
        return np.concatenate(out) if len(out) > 1 else out[0]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None  # set in main
    # BaseHTTPRequestHandler defaults to HTTP/1.0, which means a new TCP
    # connection per request and no keep-alive. Every response here sends an
    # accurate Content-Length, which is the only requirement for 1.1, and
    # some clients and proxies are unhappy with 1.0.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):      # quieter than the default
        pass

    # ⚠️ CORS is OPT-IN and never applies to /admin.
    #
    # Browsers block any page from calling this server without it, so a
    # browser-based client cannot work at all -- that is the reason to
    # have it. But this server has NO AUTH, and /admin/voices can delete
    # a voice or build one from a server-side path. With
    # Access-Control-Allow-Origin: * on those routes, any page the user
    # happens to visit could delete their voices. The README already
    # warns against exposing the port; enabling CORS by default would
    # quietly undo that warning for every browser tab.
    #
    # So: off unless DRAMA3_CORS is set, and the header is attached only
    # to the public read/speech routes. Set it to a specific origin
    # rather than "*" if you can.
    def _cors_origin(self) -> str:
        if not CORS:
            return ""
        if self.path.rstrip("/").startswith("/admin"):
            return ""
        return CORS

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Preflight. Without this BaseHTTP answers 501 and the browser
        never sends the real request."""
        origin = self._cors_origin()
        if not origin:
            self.send_response(405)
            self.send_header("Allow", "GET, POST")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code: int, obj, param: str | None = None) -> None:
        """Send JSON, upgrading a bare error string to the OpenAI envelope.

        ⚠️ The spec wants `error` to be an OBJECT with message/type/param/
        code, and both official SDKs assume it. With a bare string:

          openai-python  _client.py unwraps body["error"] to the string,
                         APIError sees a non-dict, and the user is shown
                         `Error code: 400 - {'error': "..."}` -- a Python
                         dict repr.
          openai-node    `error?.message` is undefined on a string, so it
                         falls back to JSON.stringify and double-quotes it.

        The messages here are among the better things about this server and
        both SDKs mangled every one of them. Upgrading centrally rather
        than at all 17 call sites means no new error site can forget.
        """
        if isinstance(obj, dict) and isinstance(obj.get("error"), str):
            obj = {"error": {
                "message": obj["error"],
                "type": "invalid_request_error" if code < 500 else "server_error",
                "param": param,
                "code": None,
            }}
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            # Some OpenAI clients list models before their first call.
            # `created` is required by the spec and non-optional in the
            # SDK's Model type, so a strict client rejects the response
            # without it. Fixed: the project's first commit date.
            self._json(200, {"object": "list", "data": [
                {"id": "drama-3", "object": "model",
                 "created": 1760000000, "owned_by": "fishaudio"}]})
            return
        if self.path.rstrip("/") == "/v1/audio/voices":
            # ⚠️ /v1/audio/voices, NOT /v1/voices. Not in the OpenAI spec
            # either way, but Kokoro-FastAPI serves it under /v1/audio/,
            # and a client that only tries the conventional path would
            # find nothing here. This was /v1/voices for a few hours and
            # was moved rather than aliased: two paths for one resource
            # is how a "standard" interface quietly stops being one.
            #
            # (openedai-speech, named alongside Kokoro when this came up,
            # turns out to have no voices endpoint at all -- it configures
            # voices in YAML. The convention rests on Kokoro.)
            eng = self.engine
            names = sorted(eng.voices)
            self._json(200, {"voices": [
                {"id": n, "default": n == eng.default_voice} for n in names]
                or [{"id": "drama3", "default": True}]})
            return
        if self.path.rstrip("/") in ("/health", "/healthz", ""):
            STATE["voices"] = sorted(self.engine.voices) if self.engine else []
            STATE["default_voice"] = (self.engine.default_voice
                                      if self.engine else None)
            uptime = time.time() - STATE["started"]
            rtf = (STATE["generate_seconds"] / STATE["audio_seconds"]
                   if STATE["audio_seconds"] else None)
            self._json(200, {**STATE, "uptime_seconds": round(uptime, 1),
                             "rtf": round(rtf, 3) if rtf else None})
        else:
            # 405 + Allow, not 404, when the path exists for another verb.
            if self.path.rstrip("/") in ("/v1/audio/speech", "/tts",
                                         "/tts_to_audio"):
                self.send_response(405)
                self.send_header("Allow", "POST")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._json(404, {"error": "GET /health, /v1/models or "
                                      "/v1/audio/voices; "
                                      "POST /v1/audio/speech or /tts"})

    def do_POST(self):
        # Three routes, same engine. Nothing here is specific to any one
        # client -- the point is that a tool which already speaks to some
        # other TTS server can be pointed at this one unchanged.
        #
        #   /v1/audio/speech   OpenAI. Field "input", default format mp3.
        #   /tts_to_audio      XTTS-api-server. Field "text", wav.
        #   /tts               this project's own. Field "text", wav.
        route = self.path.rstrip("/") or "/"

        # ⚠️ An ADMIN route, deliberately not under /v1: OpenAI has no
        # voice-creation call, and inventing one under its namespace would
        # make a client think this is part of the standard. The public API
        # stays exactly what everyone else implements.
        if route == "/admin/voices":
            try:
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                name = (body.get("name") or "").strip()
                clip = (body.get("clip") or "").strip()
            except Exception as exc:                       # noqa: BLE001
                self._json(400, {"error": f"bad request: {exc}"})
                return
            if not name or not clip:
                self._json(400, {"error": "need 'name' and 'clip'"})
                return
            started = time.perf_counter()
            try:
                # Queued behind generation, not run beside it: prefilling
                # writes the same KV caches a decode reads.
                with GPU:
                    info = self.engine.add_voice(
                        name, clip, body.get("ref_text"),
                        body.get("ref_seconds"))
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            except Exception as exc:                       # noqa: BLE001
                log(f"addvoice {name!r} FAILED: {type(exc).__name__}: {exc}")
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            info["elapsed"] = round(time.perf_counter() - started, 2)
            log(f"voice {name!r} added live in {info['elapsed']:.1f}s "
                f"({info['bytes']/2**20:.0f} MiB, {info['rows']} rows)")
            self._json(200, info)
            return

        if route == "/admin/voices/remove":
            try:
                length = int(self.headers.get("content-length", 0))
                name = (json.loads(self.rfile.read(length) or b"{}")
                        .get("name") or "").strip()
            except Exception as exc:                       # noqa: BLE001
                self._json(400, {"error": f"bad request: {exc}"})
                return
            try:
                # No GPU work: dropping a reference and deleting files.
                info = self.engine.drop_voice(name)
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
                return
            log(f"voice {name!r} removed; now: "
                f"{', '.join(info['voices']) or '(none)'}")
            self._json(200, info)
            return

        if route == "/admin/voices/default":
            try:
                length = int(self.headers.get("content-length", 0))
                name = (json.loads(self.rfile.read(length) or b"{}")
                        .get("name") or "").strip()
            except Exception as exc:                       # noqa: BLE001
                self._json(400, {"error": f"bad request: {exc}"})
                return
            if name not in self.engine.voices:
                self._json(400, {"error": f"unknown voice {name!r}; have: "
                                 f"{', '.join(sorted(self.engine.voices)) or '(none)'}"})
                return
            import voices as voicelib
            voicelib.set_default(name)
            self.engine.default_voice = name
            log(f"default voice is now {name!r}")
            self._json(200, {"default": name})
            return

        if route not in ("/tts", "/tts_to_audio", "/v1/audio/speech"):
            self._json(404, {"error": "POST /v1/audio/speech, /tts or /tts_to_audio"})
            return
        openai = route == "/v1/audio/speech"
        try:
            length = int(self.headers.get("content-length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            # OpenAI calls it "input"; everyone else calls it "text". Accept
            # either on any route rather than being pedantic about it.
            text = (payload.get("input") or payload.get("text") or "").strip()
            fmt = (payload.get("response_format")
                   or payload.get("format")
                   or ("mp3" if openai else "wav")).lower()
            speed = float(payload.get("speed") or 1.0)
            # ⚠️ The spec allows `voice` as a STRING or as an object
            # {"id": "..."}. The documented object form used to reach
            # .strip() and 400 with "'dict' object has no attribute
            # 'strip'" -- a conforming client rejected, and a raw Python
            # AttributeError leaking into the response body.
            raw_voice = payload.get("voice")
            if isinstance(raw_voice, dict):
                raw_voice = raw_voice.get("id")
            voice = (str(raw_voice).strip() or None) if raw_voice else None
            stream_format = payload.get("stream_format")
        except Exception as exc:
            # ⚠️ Do not put exception text in the body. It leaks internals
            # and reads as a server bug when the fault is in the request.
            log(f"bad request: {type(exc).__name__}: {exc}")
            self._json(400, {"error": "malformed JSON body"})
            return

        # ⚠️ Reject rather than ignore. Accepting stream_format="sse" and
        # returning raw audio is the same silent-bite that `speed` had:
        # the client asked for one thing, got another, and fails somewhere
        # less obvious. Only "audio" is implemented.
        if stream_format not in (None, "audio"):
            self._json(400, {"error": f"stream_format {stream_format!r} is not "
                             "supported; this server returns complete audio "
                             "(stream_format 'audio')"}, "stream_format")
            return
        if not text:
            self._json(400, {"error": "no text: send 'input' or 'text'"}, "input")
            return
        if fmt not in FORMATS:
            self._json(400, {"error": f"unsupported response_format {fmt!r}; "
                                      f"have {', '.join(sorted(FORMATS))}"})
            return
        if not 0.25 <= speed <= 4.0:
            self._json(400, {"error": f"speed {speed} outside 0.25-4.0"}, "speed")
            return
        # ⚠️ Validate the voice BEFORE taking the GPU lock. An unknown name
        # is the client's typo, answerable in microseconds; discovering it
        # after queueing behind a 20 s generation would make a 400 look
        # like a performance problem.
        try:
            self.engine.pick_voice(voice)
        except ValueError as exc:
            self._json(400, {"error": str(exc)}, "voice")
            return
        started = time.perf_counter()
        try:
            # Serialised: one model, one set of KV caches. Concurrent
            # generations would interleave into the same buffers.
            with GPU:
                audio = self.engine.speak(text, voice)
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
            f"{elapsed:6.2f}s  rtf {elapsed/max(seconds,1e-9):.2f}  "
            f"[{fmt}{'' if abs(speed-1)<1e-3 else f' x{speed:g}'}]")
        try:
            body, ctype = encode(audio, fmt, speed)
        except Exception as exc:
            self._json(500, {"error": str(exc)})
            return
        self._send(200, body, ctype)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Resident Drama 3 TTS server. Settings come from "
                    "config.toml; any flag here overrides the file.")
    ap.add_argument("--config", default=None,
                    help="TOML settings file (default: ./config.toml if present)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--state-dir", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--cuda-graph", choices=("0", "1"), default="1")
    ap.add_argument("--pin-experts", action="store_true", default=True)
    ap.add_argument("--no-pin-experts", dest="pin_experts", action="store_false")
    ap.add_argument("--slice-lm-head", action="store_true", default=True)
    ap.add_argument("--offload-embed", action="store_true", default=True)
    ap.add_argument("--prefix-kv", action="store_true", default=True)
    ap.add_argument("--no-voices", dest="voices", action="store_false",
                    default=True, help="ignore voices/ and use --ref only")
    # Switches that override a config section wholesale, for a one-off run
    # without editing the file.
    ap.add_argument("--no-verify-stt", action="store_true",
                    help="disable the content check for this run")
    ap.add_argument("--no-checks", action="store_true",
                    help="disable every quality check; fastest, least safe")
    args = ap.parse_args()

    cfg = config.load(args.config)
    if args.no_verify_stt or args.no_checks:
        cfg.checks.content.enabled = False
    if args.no_checks:
        cfg.checks.duration.enabled = False
    args.checkpoint = args.checkpoint or str(paths.checkpoint())
    args.state_dir = args.state_dir or str(paths.HOME / "run")
    args.ref = args.ref or cfg.voice.ref
    args.host = args.host or cfg.server.host
    args.port = args.port or cfg.server.port
    if not args.ref:
        raise SystemExit("no reference clip: set --ref or voice.ref in config.toml")

    log(f"config: {cfg._path}")
    log(config.describe(cfg))

    Handler.engine = Engine(args, cfg)
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    # ⚠️ Publish the address we ACTUALLY bound, for anything that needs to
    # reach us. drama3ctl used to assume 127.0.0.1:8123 and pass it as a
    # flag, which silently beat `[server] port` in config.toml: the server
    # obeyed the config, the launcher health-checked its own guess, and
    # `drama3ctl config` handed clients a URL nothing was listening on.
    # A reader is never wrong in the way a guesser is.
    #
    # getsockname(), not args.port, so `port = 0` (ask the OS) reports the
    # real number rather than a zero.
    bound_host, bound_port = server.socket.getsockname()[:2]
    url = f"http://{args.host or bound_host}:{bound_port}"
    url_file = Path(args.state_dir) / "server.url"
    url_file.parent.mkdir(parents=True, exist_ok=True)
    url_file.write_text(url + "\n")

    log(f"listening on {url}  (POST /v1/audio/speech, GET /health)")

    # ⚠️ SIGTERM, not just Ctrl-C. `drama3ctl stop` signals the process
    # group, and without this the finally: below never ran -- server.url
    # outlived the server, so `drama3ctl config` handed clients a URL
    # with nothing listening on it. The cleanup existed and was simply
    # unreachable by the only shutdown path anyone actually uses.
    import signal

    def _terminate(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGINT, _terminate)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("stopping")
    finally:
        # Stale file here means a client connects to a dead port and waits,
        # which reads as a hang rather than as "not running".
        url_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
