"""Where everything lives. Import this before importing fish_speech.

Every script here used to open with

    FISH = Path("/mnt/unencrypted/ai/fish-s2")
    sys.path.insert(0, str(FISH))

which worked on exactly one machine. This resolves the same things from the
environment, with defaults that are right for a normal checkout, and fails
with an instruction rather than an ImportError when it cannot.

  DRAMA3_FISH   a checkout of github.com/fishaudio/fish-speech.
                ⚠️ NOT vendored and NOT forked -- everything in bin/ is
                additive and hooks upstream at runtime, so you want a clean
                checkout. Verified against 5415669.
  DRAMA3_CKPT   the fishaudio/drama-3 snapshot directory (the one containing
                config.json). Found under HF_HOME if not set.
  DRAMA3_STT    optional. A unix socket speaking the "write a wav path, read
                one line" protocol, used to transcribe the reference clip and
                to score output. Without it, pass --ref-text yourself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent


def _first_existing(candidates) -> Path | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate).resolve()
    return None


def fish_speech() -> Path:
    """The fish-speech checkout, as a directory to put on sys.path."""
    found = _first_existing([
        os.environ.get("DRAMA3_FISH"),
        HOME.parent / "fish-speech",
        HOME.parent / "fish-s2",
        HOME / "fish-speech",
    ])
    if found is None or not (found / "fish_speech").is_dir():
        raise SystemExit(
            "cannot find a fish-speech checkout.\n"
            "  git clone https://github.com/fishaudio/fish-speech\n"
            "  export DRAMA3_FISH=/path/to/fish-speech\n"
            "It is used unmodified; nothing here patches it."
        )
    return found


def checkpoint() -> Path:
    """The Drama 3 snapshot directory, i.e. the one holding config.json."""
    explicit = os.environ.get("DRAMA3_CKPT")
    if explicit:
        if not (Path(explicit) / "config.json").is_file():
            raise SystemExit(f"DRAMA3_CKPT={explicit} has no config.json")
        return Path(explicit)
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snapshots = hf / "hub/models--fishaudio--drama-3/snapshots"
    if snapshots.is_dir():
        for entry in sorted(snapshots.iterdir()):
            if (entry / "config.json").is_file():
                return entry
    raise SystemExit(
        "cannot find the Drama 3 weights.\n"
        "  hf download fishaudio/drama-3   # huggingface-cli in hub < 1.0\n"
        "  export DRAMA3_CKPT=/path/to/snapshots/<hash>\n"
        "36.9 GiB of weights plus a 1.87 GiB codec. Licence: Fish Audio\n"
        "Research, non-commercial -- this repo ships no weights."
    )


def stt_endpoint() -> str | None:
    """Where to send audio for transcription, or None. Optional on purpose.

    Two shapes are accepted, because one of them is portable and the other
    is what this was developed against:

      http://host:port    an OpenAI-compatible server -- POST
                          /v1/audio/transcriptions. whisper.cpp's server,
                          faster-whisper-server, LocalAI, openedai-whisper
                          and OpenAI all answer it.
      /path/to/sock       a unix socket speaking "write an absolute wav
                          path, read one line of text".

    ⚠️ The socket form is bespoke. It exists because the machine this was
    built on already had a resident Parakeet daemon with that protocol, and
    keeping it costs nothing -- but nobody else has one, so the HTTP form is
    the one to document.

    Without either, the content check silently does nothing and the
    reference transcript must be supplied with --ref-text.
    """
    value = os.environ.get("DRAMA3_STT")
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        return value
    return value if Path(value).exists() else None


# Older name, kept so nothing breaks.
stt_socket = stt_endpoint


def install() -> Path:
    """Put fish-speech and bin/ on sys.path. Returns the fish-speech path."""
    fish = fish_speech()
    for entry in (str(fish), str(HOME / "bin")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return fish
