# Source this before running anything here:  source bin/env.sh
#
# Sensible defaults everywhere, plus a NixOS section that only fires on
# NixOS. Every line in that section is a trap that cost someone hours, so
# they are kept rather than deleted even though most people will never hit
# them.

DRAMA3_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DRAMA3_HOME

# --- your machine's answers, if you have written them down -------------
# bin/env.local is sourced FIRST so it wins: every default below uses
# := and therefore defers to anything already set.
#
# It exists because a checkout that does not sit beside its venv,
# fish-speech and weights needs several exports, and typing them before
# every start is the kind of friction that gets a project called fiddly.
# Write them once:
#
#   cp bin/env.local.example bin/env.local   # then edit
#
# ⚠️ GITIGNORED on purpose. It holds absolute paths from one machine --
# and possibly a path to somebody's voice recording.
if [ -f "$DRAMA3_HOME/bin/env.local" ]; then
    # shellcheck disable=SC1091
    source "$DRAMA3_HOME/bin/env.local"
fi

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
# Speech-to-text, for the content check that catches gibberish and for
# transcribing the reference clip. Either an OpenAI-compatible server or a
# unix socket; see bin/paths.py.
#   export DRAMA3_STT=http://127.0.0.1:8080      # /v1/audio/transcriptions
#   export DRAMA3_STT=/run/user/1000/stt.sock    # bespoke, see paths.py
# Without it, pass --ref-text yourself. ⚠️ This used to claim ASR errors
# were "inherited by every chunk"; that was asserted, never measured, and
# the measurement did not support it -- see notes/17. A transcript of the
# single word "placeholder" against 56 s of unrelated speech cost nothing
# detectable in timbre or content. It did speak ~8 % slower.
#
# Browser clients: set an origin to enable CORS. OFF by default, and
# never applied to /admin/* -- this server has no auth and /admin/voices
# can delete a voice, so "*" would let any page you visit administer it.
#   export DRAMA3_CORS=http://localhost:3000
#
# Where the compiled CUDA extension is cached. Defaults to
# torch-ext-cache/ beside the code, which needs a WRITABLE checkout:
#   export DRAMA3_EXT_CACHE=/var/tmp/drama3-ext
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
# ⚠️ /etc/NIXOS, and ONLY that. Two weaker tests were tried and both
# gave false positives inside a GPU container on Ubuntu:
#
#   /run/opengl-driver/lib  the nvidia-container-toolkit CREATES it in
#                           the container, on any distro
#   /nix/store              on NixOS the toolkit MOUNTS the driver libs
#                           from the store, so the container has one too
#
# Either way an Ubuntu container printed "no libstdc++.so.6 found in
# /nix/store; torch will not import" while torch imported perfectly
# well -- harmless, alarming, and exactly the sort of message that costs
# somebody an evening. /etc/NIXOS is present on the host and absent in
# the container, which is the distinction that actually matters.
if [ -e /etc/NIXOS ]; then
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

    # ⚠️ libstdc++.so.6. torch's own _C extension needs it, and a venv
    # built from a NIX-STORE python cannot find it: `import torch` dies
    # with "ImportError: libstdc++.so.6: cannot open shared object file"
    # from inside torch/__init__.py.
    #
    # This went undetected for the life of the project because the author's
    # venv is built on a UV-MANAGED CPython. That is a foreign binary, so
    # NixOS's programs.nix-ld fires and supplies libstdc++ transparently.
    # A nix-store python uses the nix loader, nix-ld never applies, and the
    # documented setup fails on the only machine this has ever run on.
    #
    # ⚠️ `nix-shell -p gcc` does NOT fix this -- verified. The compiler's
    # own lib output is not on the runtime loader path, so the obvious
    # guess wastes an hour.
    for _d in /nix/store/*-gcc-*-lib/lib; do
        [ -e "$_d/libstdc++.so.6" ] || continue
        case "$_d" in *mingw*) continue ;; esac   # matches the glob, wrong libc
        _v=$(printf '%s' "$_d" | sed -n 's|.*-gcc-\([0-9.]*\)-lib/lib$|\1|p')
        [ -n "$_v" ] || continue
        if [ -z "${_bestv:-}" ] || \
           [ "$(printf '%s\n%s\n' "$_bestv" "$_v" | sort -V | tail -1)" = "$_v" ]; then
            _bestv="$_v"; _best="$_d"
        fi
    done
    if [ -n "${_best:-}" ]; then
        export LD_LIBRARY_PATH="$_best${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    else
        echo "env.sh: no libstdc++.so.6 found in /nix/store; torch will not import" >&2
    fi
    unset _d _v _best _bestv

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
