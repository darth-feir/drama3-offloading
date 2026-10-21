#!/usr/bin/env bash
# OPTIONAL. Sets up a local speech-to-text server, for people who do not
# already run one.
#
#   ./bin/setup-stt.sh          # install into stt-venv/ and fetch the model
#   ./bin/setup-stt.sh --serve  # ...and then run it in the foreground
#
# SKIP THIS if you already have an OpenAI-compatible
# /v1/audio/transcriptions endpoint -- whisper.cpp's server,
# faster-whisper-server, LocalAI, anything. Point the TTS server at it:
#
#   export DRAMA3_STT=http://127.0.0.1:8080
#
# The endpoint is what matters, not the engine. This script exists so
# that "I don't have one" is not a dead end, not because this particular
# engine is required.
#
# WHAT IT INSTALLS
#   parakeet-api  (PyPI), serving NVIDIA Parakeet TDT 0.6B v2 int8 on
#   sherpa-onnx, CPU. Measured RTF ~0.04, roughly 27x realtime, ~0.17 s
#   per check. ~465 MB of model.
#
# ⚠️ CPU on purpose. ONNX Runtime's CUDA provider measured SLOWER than
# CPU for this model -- 520 ms against 426 ms -- because per-inference
# launch and transfer overhead exceeds the compute saved. It also means
# transcription keeps working while something else owns the GPU, which
# matters here: the TTS server holds ~20 GiB of it.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SERVE=0
PORT="${STT_PORT:-8816}"
for arg in "$@"; do
    case "$arg" in
        --serve)   SERVE=1 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mSTT SETUP FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# --- already serving? -------------------------------------------------
if curl -fsS --max-time 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    say "Something is already answering on port $PORT"
    note "Nothing to do. Point the TTS server at it:"
    note "  export DRAMA3_STT=http://127.0.0.1:$PORT"
    exit 0
fi

# --- python -----------------------------------------------------------
say "Looking for Python 3.12"
PY="${DRAMA3_PYTHON:-}"
if [ -z "$PY" ]; then
    for cand in python3.12 python3; do
        if command -v "$cand" >/dev/null 2>&1 &&
           "$cand" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3,12))' 2>/dev/null; then
            PY="$(command -v "$cand")"; break
        fi
    done
fi
if [ -z "$PY" ] && command -v nix-shell >/dev/null 2>&1; then
    PY="$(nix-shell -p python312 --run 'command -v python3.12' 2>/dev/null || true)"
fi
[ -n "$PY" ] || die "no Python 3.12. Install one or set DRAMA3_PYTHON."
note "using $PY"

# --- venv -------------------------------------------------------------
say "Installing parakeet-api"
VENV="$REPO/stt-venv"
if [ -x "$VENV/bin/parakeet-api" ]; then
    note "already installed at $VENV"
else
    [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || die "venv failed"
    export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"
    "$VENV/bin/pip" install --quiet --disable-pip-version-check parakeet-api ||
        die "pip install parakeet-api failed"
    note "installed"
fi

# ⚠️ NixOS: a venv built from a nix-store python cannot find libstdc++,
# and parakeet-api dies on import. bin/env.sh fixes it for processes it
# launches -- but this venv is not one of those, so borrow its answer.
if [ -e /etc/NIXOS ] || [ -d /run/opengl-driver/lib ]; then
    note "NixOS: borrowing LD_LIBRARY_PATH from bin/env.sh"
    # shellcheck disable=SC1091
    source "$REPO/bin/env.sh" >/dev/null 2>&1 || true
fi

# --- model ------------------------------------------------------------
say "Model (~465 MB, Parakeet TDT 0.6B v2 int8)"
if "$VENV/bin/parakeet-api" download sherpa 2>&1 | tail -2; then
    note "model ready"
else
    die "model download failed"
fi

# --- how to run -------------------------------------------------------
cat <<EOF

$(printf '\033[1m')Running it$(printf '\033[0m')
  $VENV/bin/parakeet-api serve

  then, for the TTS server:
    export DRAMA3_STT=http://127.0.0.1:$PORT

  To have the TTS server start it for you:
    export DRAMA3_STT_START="$VENV/bin/parakeet-api serve"

⚠️ Configure parakeet-api with STT__* ENVIRONMENT VARIABLES. Its --stt.*
   command-line flags are accepted without complaint and then ignored --
   the engine went looking for a default model while both flags sat on
   its command line. Worth an upstream issue.

⚠️ Its /health answers before the engine has loaded. It is liveness, not
   readiness; the first transcription is what initialises sherpa.
EOF

if [ "$SERVE" = 1 ]; then
    say "Serving on port $PORT (Ctrl-C to stop)"
    exec "$VENV/bin/parakeet-api" serve
fi
