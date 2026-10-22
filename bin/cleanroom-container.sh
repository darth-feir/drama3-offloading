#!/usr/bin/env bash
# Build and smoke-test this project inside another distro's userspace.
#
#   ./bin/cleanroom-container.sh docker.io/library/ubuntu:24.04
#   ./bin/cleanroom-container.sh docker.io/library/fedora:42 --gpu
#
# The native clean room (bin/cleanroom.sh) proves the setup works HERE.
# This proves it works somewhere that is not NixOS -- the gap `16` left
# open, where every number came from one machine.
#
# Rootless podman via nix-shell: no install, no daemon, no system
# configuration. podman's state goes in a scratch HOME so nothing lands
# in ~/.config.
#
# ⚠️ CONTAINERS SHARE THE GPU, THEY DO NOT TAKE IT. Safe to run while a
# desktop is on the same card. A VM would need VFIO passthrough, which
# unbinds the card and kills the display. Do not do that.
#
# ⚠️ IPv6. If the host advertises IPv6 but cannot route it, pip inside a
# container crawls at ~30 KiB/s with no error, because the package host
# resolves to v6 only. The resolv.conf and gai.conf written below force
# IPv4 preference: measured 30 KiB/s -> 3.9 MB/s. See notes/19.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${1:-docker.io/library/ubuntu:24.04}"
shift 2>/dev/null || true
WANT_GPU=0
for a in "$@"; do [ "$a" = "--gpu" ] && WANT_GPU=1; done

SLUG=$(printf '%s' "$IMAGE" | tr '/:' '__')
WORK="${DRAMA3_CONTAINER_WORK:-/mnt/data/Git/_distrotest}"
RESULTS="$WORK/RESULTS.md"
T="$WORK/$SLUG"
mkdir -p "$T/home/.config/containers" "$WORK"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# podman needs these and will not invent them; ~/.config is off limits.
cat > "$T/home/.config/containers/policy.json" <<'EOF'
{"default": [{"type": "insecureAcceptAnything"}]}
EOF
HOSTNS=$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf 2>/dev/null)
cat > "$T/resolv.conf" <<EOF
nameserver ${HOSTNS:-1.1.1.1}
options edns0 timeout:2 attempts:2
EOF
cat > "$T/gai.conf" <<'EOF'
precedence ::ffff:0:0/96  100
EOF

say "Staging the repo"
rsync -a --exclude '.git' --exclude venv --exclude run --exclude voices \
      --exclude torch-ext-cache --exclude out --exclude __pycache__ \
      --exclude 'bin/env.local' "$REPO/" "$T/repo/"

# The in-container driver. Each distro names its python package
# differently, and some do not ship 3.12 at all -- which is itself a
# finding rather than a failure of this script.
cat > "$T/run.sh" <<'EOS'
set -uo pipefail
. /etc/os-release 2>/dev/null || true
echo "DISTRO: ${PRETTY_NAME:-unknown}"

if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -qq update >/dev/null 2>&1
    apt-get -qq install -y python3.12-venv git curl ffmpeg >/dev/null 2>&1 \
      || apt-get -qq install -y python3-venv git curl ffmpeg >/dev/null 2>&1
elif command -v dnf >/dev/null 2>&1; then
    dnf -q -y install python3.12 git curl ffmpeg-free >/dev/null 2>&1 \
      || dnf -q -y install python3 git curl >/dev/null 2>&1
elif command -v pacman >/dev/null 2>&1; then
    pacman -Sy --noconfirm --quiet python git curl ffmpeg >/dev/null 2>&1
fi

PY=$(command -v python3.12 || true)
if [ -z "$PY" ]; then
    echo "PYTHON312: NOT AVAILABLE ($(python3 --version 2>&1))"
    echo "VERDICT: BLOCKED -- this distro ships no python3.12 and the"
    echo "         project requires it. Not a bug in setup.sh."
    exit 20
fi
echo "PYTHON312: $($PY --version 2>&1)"

export PIP_CACHE_DIR=/work/pipcache HF_HOME=/hf
cd /work/repo
./bin/setup.sh --yes --no-weights
rc=$?
echo "SETUP_EXIT: $rc"
[ $rc -gt 1 ] && { echo "VERDICT: SETUP FAILED"; exit $rc; }

if [ "${WANT_GPU:-0}" = 1 ] && [ -e /clip/ref.mp3 ]; then
    echo "=== GPU SMOKE ==="
    ./bin/run.sh bin/narrate.py \
      --text-file "text for testing quote from Dune.txt" --out-dir out/smoke \
      --ref /clip/ref.mp3 \
      --ref-text "For any organization using the cloud, securing your Azure resources is a critical function." \
      --ref-seconds 56 --dev-cache --pin-experts --cuda-graph 1 \
      --vram-reserve 4 --prefill-chunk 1024 --prefix-kv --slice-lm-head \
      --offload-embed --limit 1 2>&1 | tail -12
    echo "SMOKE_EXIT: $?"
fi
echo "VERDICT: COMPLETED"
EOS

GPUFLAG=()
CLIPMOUNT=()
if [ "$WANT_GPU" = 1 ]; then
    GPUFLAG=(--device nvidia.com/gpu=all)
    [ -d "$WORK/clip" ] && CLIPMOUNT=(-v "$WORK/clip:/clip:ro")
fi
HFMOUNT=()
[ -n "${HF_HOME:-}" ] && [ -d "${HF_HOME:-}" ] && HFMOUNT=(-v "$HF_HOME:/hf:ro")

say "Running $IMAGE  (gpu=$WANT_GPU)"
LOG="$T/run.log"
nix-shell -p podman --run "HOME=$T/home WANT_GPU=$WANT_GPU podman run --rm --network=host \
  ${GPUFLAG[*]} ${CLIPMOUNT[*]} ${HFMOUNT[*]} \
  -e WANT_GPU=$WANT_GPU \
  -v $T/resolv.conf:/etc/resolv.conf:ro -v $T/gai.conf:/etc/gai.conf:ro \
  -v $T:/work:rw -w /work $IMAGE bash /work/run.sh" > "$LOG" 2>&1
rc=$?

{
  echo ""
  echo "## $IMAGE  (gpu=$WANT_GPU)  rc=$rc"
  echo '```'
  grep -E 'DISTRO:|PYTHON312:|SETUP_EXIT:|SMOKE_EXIT:|VERDICT:|^\[ *(ok|warn|FAIL)|ok,|failures' "$LOG" | tail -25
  echo '```'
} >> "$RESULTS"

say "rc=$rc  log: $LOG"
tail -5 "$LOG"
exit $rc
