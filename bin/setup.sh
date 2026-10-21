#!/usr/bin/env bash
# One command from a fresh clone to a working install.
#
#   ./bin/setup.sh                 # everything, prompting before the 38 GiB
#   ./bin/setup.sh --yes           # no prompts
#   ./bin/setup.sh --no-weights    # skip the download (already have them)
#
# WHY THIS EXISTS
# ---------------
# The README's Setup was twelve commands with traps between them, and
# three of those commands exist ONLY because of traps found the hard way:
#
#   git checkout 5415669     upstream moves; "verified against" is worth
#                            nothing without the checkout
#   pip install --no-deps    fish-speech declares pyaudio, which needs
#                            portaudio headers and is never used here
#   HF_HUB_DISABLE_XET=1     without it the download can sit at 4.5 kB/s
#                            with NO error on stdout -- 18 minutes of
#                            silence in a clean-room run
#
# A newcomer cannot tell which steps are load-bearing. Encoding them here
# turns knowledge-you-had-to-read into behaviour-you-get.
#
# Safe to re-run: every step checks before acting.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FISH_PIN=5415669
ASSUME_YES=0
WANT_WEIGHTS=1
for arg in "$@"; do
    case "$arg" in
        --yes|-y)     ASSUME_YES=1 ;;
        --no-weights) WANT_WEIGHTS=0 ;;
        -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mSETUP FAILED: %s\033[0m\n' "$*" >&2; exit 1; }

ask() {
    [ "$ASSUME_YES" = 1 ] && return 0
    [ -t 0 ] || return 1          # non-interactive: treat as "no"
    read -r -p "    $1 [y/N] " reply
    [[ "$reply" =~ ^[Yy] ]]
}

# --- 1. a python 3.12 -------------------------------------------------
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
# NixOS has no python3.12 on PATH; nix-shell can supply one.
if [ -z "$PY" ] && command -v nix-shell >/dev/null 2>&1; then
    note "none on PATH; asking nix-shell for python312"
    PY="$(nix-shell -p python312 --run 'command -v python3.12' 2>/dev/null || true)"
fi
[ -n "$PY" ] || die "no Python 3.12 found. Install one, or set DRAMA3_PYTHON
    to its path. 3.12 specifically: several upstream dependencies cap
    below 3.13."
note "using $PY"

# --- 2. fish-speech, at the pinned commit -----------------------------
say "fish-speech (used unmodified, not vendored)"
FISH="${DRAMA3_FISH:-$(cd "$REPO/.." && pwd)/fish-speech}"
if [ -d "$FISH/.git" ]; then
    note "already at $FISH"
else
    note "cloning into $FISH"
    git clone --quiet https://github.com/fishaudio/fish-speech "$FISH" ||
        die "clone failed. Network? Or clone it yourself and set DRAMA3_FISH."
fi
have="$(git -C "$FISH" rev-parse --short HEAD)"
if [ "$have" != "$FISH_PIN" ]; then
    if git -C "$FISH" diff --quiet && git -C "$FISH" diff --cached --quiet; then
        note "checking out the pinned $FISH_PIN (was $have)"
        git -C "$FISH" checkout --quiet "$FISH_PIN" 2>/dev/null ||
            die "could not check out $FISH_PIN in $FISH"
    else
        # ⚠️ Never discard someone's edits to make a version match.
        note "⚠️  $FISH is at $have, not the pinned $FISH_PIN, and has local"
        note "    changes. Leaving it alone. Everything here is tested"
        note "    against $FISH_PIN."
    fi
else
    note "at the pinned commit $FISH_PIN"
fi
export DRAMA3_FISH="$FISH"

# --- 3. the venv ------------------------------------------------------
say "Python environment"
VENV="${DRAMA3_VENV:-$REPO/venv}"
if [ -x "$VENV/bin/python" ]; then
    note "reusing $VENV"
else
    note "creating $VENV"
    "$PY" -m venv "$VENV" || die "could not create a venv at $VENV"
fi
export DRAMA3_VENV="$VENV"
# uv-managed CPythons ship no CA bundle; curl works, urllib does not.
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}"

note "installing fish-speech (--no-deps: it declares pyaudio, unused here)"
"$VENV/bin/pip" install --quiet --disable-pip-version-check \
    -e "$FISH" --no-deps || die "fish-speech install failed"

note "installing pinned requirements (this is the slow one)"
# ⚠️ Expect ~15 red "fish-speech 3.0.0 requires ..." lines here. pip
# resolves those only once requirements.txt lands on top of --no-deps.
# Correct, and alarming.
"$VENV/bin/pip" install --disable-pip-version-check \
    -r "$REPO/requirements.txt" 2>&1 | grep -vE '^\s*$' | tail -3 ||
    die "requirements install failed"

# --- 4. weights -------------------------------------------------------
say "Model weights (36.9 GiB + a 1.87 GiB codec)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
note "HF_HOME=$HF_HOME"
if "$VENV/bin/python" -c "
import sys; sys.path.insert(0, '$REPO/bin')
import paths
try:
    paths.checkpoint(); print('found')
except SystemExit:
    raise SystemExit(1)
" >/dev/null 2>&1; then
    note "already present"
elif [ "$WANT_WEIGHTS" = 0 ]; then
    note "skipped (--no-weights). Set DRAMA3_CKPT or re-run without it."
elif ask "Download 38 GiB now?"; then
    # ⚠️ xet off: it can stall at ~4.5 kB/s with no error on stdout.
    HF_HUB_DISABLE_XET=1 "$VENV/bin/hf" download fishaudio/drama-3 ||
        die "download failed. Re-run; hf resumes."
else
    note "skipped. When ready:"
    note "  HF_HUB_DISABLE_XET=1 $VENV/bin/hf download fishaudio/drama-3"
fi

# --- 5. hand over to doctor ------------------------------------------
say "Checking the machine"
set +e
"$REPO/bin/run.sh" bin/doctor.py
rc=$?
set -e

cat <<EOF

$(printf '\033[1m')Next$(printf '\033[0m')
  A reference clip is the one thing this repo cannot ship -- a voice
  belongs to somebody. 30-60 s of one person speaking cleanly:

    bin/drama3-offloading --addvoice myvoice /path/to/clip.mp3
    bin/drama3-offloading --start

  The content check that catches fluent gibberish needs speech-to-text.
  If you already run an OpenAI-compatible STT server:

    export DRAMA3_STT=http://127.0.0.1:8816

  If you do not:  ./bin/setup-stt.sh
EOF
exit $rc
