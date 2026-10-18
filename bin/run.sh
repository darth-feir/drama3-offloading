#!/usr/bin/env bash
# Run one of these scripts with the environment set up.
#
#   bin/run.sh bin/narrate.py --text-file doc.txt --out-dir out/run1 ...
#
# ⚠️ Arguments are passed with "$@", not "$*". The original of this script
# ended in `--run "$PY $*"`, which word-splits, so any path containing a
# space silently became two arguments. Every long-form driver here had to
# work around it by quoting inside a nix-shell string.
set -euo pipefail
cd "$(dirname "$0")/.."
source bin/env.sh

# torch.compile / inductor and the CUDA extension build need a WRAPPED gcc.
# On NixOS a bare /nix/store/...-gcc/bin/gcc is the unwrapped compiler and
# fails to link; `nix-shell -p gcc` gives the wrapped one. Elsewhere the
# system compiler is already correct.
if command -v nix-shell >/dev/null 2>&1 && { [ -e /etc/NIXOS ] || [ -d /run/opengl-driver/lib ]; }; then
    exec nix-shell -p gcc --run "$(printf '%q ' "$DRAMA3_VENV/bin/python" "$@")"
fi
exec "$DRAMA3_VENV/bin/python" "$@"
