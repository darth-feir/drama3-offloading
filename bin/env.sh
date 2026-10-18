# Source this before running anything here:  source bin/env.sh
#
# Sensible defaults everywhere, plus a NixOS section that only fires on
# NixOS. Every line in that section is a trap that cost someone hours, so
# they are kept rather than deleted even though most people will never hit
# them.

DRAMA3_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DRAMA3_HOME

# --- required ---------------------------------------------------------
# A clean checkout of github.com/fishaudio/fish-speech. NOT forked and NOT
# patched: everything in bin/ hooks it at runtime. Verified at 5415669.
: "${DRAMA3_FISH:=$(cd "$DRAMA3_HOME/.." && pwd)/fish-speech}"
export DRAMA3_FISH
export PYTHONPATH="$DRAMA3_FISH${PYTHONPATH:+:$PYTHONPATH}"

# The python that has torch in it. A venv interpreter by default.
: "${DRAMA3_VENV:=$DRAMA3_HOME/venv}"
export DRAMA3_VENV

# --- optional ---------------------------------------------------------
# Weights. Left unset, bin/paths.py finds them under HF_HOME. 36.9 GiB plus
# a 1.87 GiB codec; licence is Fish Audio Research, non-commercial.
#   export DRAMA3_CKPT=/path/to/models--fishaudio--drama-3/snapshots/<hash>
#
# A speech-to-text unix socket: write an absolute wav path, read one line of
# text. Used to transcribe the reference clip and to score output. Without
# it, pass --ref-text yourself -- which is the better choice for a long
# reference anyway, since ASR errors in the transcript are inherited by
# every chunk of the document.
#   export DRAMA3_STT=/run/user/1000/stt.sock
#
# Toolchain for building csrc/fetch_experts.cu, if discovery guesses wrong:
#   export DRAMA3_CUDA_HOME=/usr/local/cuda   # needs bin/nvcc AND include/
#   export DRAMA3_CXX=/usr/bin/g++-14         # CUDA 12.9 errors above gcc 14
#   export DRAMA3_CUDA_ARCH=8.9               # Ada; 9.0 Hopper. Auto-detected.

# Compile caches. Persist them or you pay for kernel compilation every run.
: "${TORCHINDUCTOR_CACHE_DIR:=$DRAMA3_HOME/.cache/inductor}"
: "${TRITON_CACHE_DIR:=$DRAMA3_HOME/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR TRITON_CACHE_DIR
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# --- NixOS only -------------------------------------------------------
if [ -e /etc/NIXOS ] || [ -d /run/opengl-driver/lib ]; then
    # PyPI CUDA wheels are not patchelf'd. Without this,
    # torch.cuda.is_available() returns False SILENTLY and everything runs
    # on the CPU -- no error, just 50x slower.
    export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # torchcodec is mandatory for torch >= 2.9 (torchaudio.load routes
    # through it) and needs libavcodec on the library path.
    for _ff in /nix/store/*ffmpeg*-lib/lib; do
        [ -d "$_ff" ] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_ff"
    done
    unset _ff

    # triton HARDCODES /sbin/ldconfig, which does not exist on NixOS, and
    # dies with FileNotFoundError. This knob is checked BEFORE that call, so
    # no patching is needed.
    export TRITON_LIBCUDA_PATH=/run/opengl-driver/lib

    # uv-managed CPythons ship NO CA bundle, so urllib and requests die with
    # CERTIFICATE_VERIFY_FAILED while curl works fine on the same URL.
    if [ -z "${SSL_CERT_FILE:-}" ] && [ -r /etc/ssl/certs/ca-certificates.crt ]; then
        export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    fi
fi

# ⚠️ Must end with a success. This file is meant to be used as
# `source bin/env.sh && something`, and a sourced file returns the status of
# its LAST command. An `a && b && c` chain whose first test is false returns
# 1, so the && short-circuits and `something` silently never runs -- no
# error, no output, exit 1. That happened; this line is the fix.
true
