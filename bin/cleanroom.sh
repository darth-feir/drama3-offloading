#!/usr/bin/env bash
# Build a clean room and run the README's setup in it, from scratch.
#
#   bin/cleanroom.sh DIR [pinned|latest]
#
#     pinned   (default) the versions every measurement here used
#     latest   whatever pip resolves today -- a forward-compatibility canary
#
# WHY THIS EXISTS
# ---------------
# The setup instructions were written twice, believed twice, and were wrong
# six ways: torch unpinned to a version fish-speech does not support,
# `pip install -e ../fish-speech` failing on pyaudio's C extension, four
# packages missing for import, one more missing at runtime behind a hydra
# error that named the wrong thing, and a CLI binary that has been renamed.
#
# Reading instructions does not find any of that. Executing them somewhere
# that does not already have the answers finds all of it in twenty minutes.
#
# Needs ~7 GiB of disk and a long download. The weights are NOT duplicated --
# point HF_HOME at a copy you already have.
set -euo pipefail

DIR="${1:?usage: bin/cleanroom.sh DIR [pinned|latest]}"
MODE="${2:-pinned}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${DRAMA3_PYTHON:-python3.12}"

command -v "$PYBIN" >/dev/null || {
    echo "no $PYBIN on PATH; set DRAMA3_PYTHON to a 3.12 interpreter" >&2
    exit 1
}
[ -n "${HF_HOME:-}" ] || echo "note: HF_HOME unset, weights will go to ~/.cache" >&2

mkdir -p "$DIR"
DIR="$(cd "$DIR" && pwd)"
echo "=== clean room at $DIR ($MODE) ==="

if [ ! -d "$DIR/fish-speech" ]; then
    # A local clone if one is to hand -- much faster, and gives the same
    # tracked tree. Note it deliberately does NOT copy checkpoints/, which
    # is untracked and is not something a new user would have.
    if [ -d "${DRAMA3_FISH:-}" ]; then
        git clone -q --local --no-hardlinks "$DRAMA3_FISH" "$DIR/fish-speech"
    else
        git clone -q https://github.com/fishaudio/fish-speech "$DIR/fish-speech"
    fi
fi
echo "  fish-speech: $(cd "$DIR/fish-speech" && git log --oneline -1)"

"$PYBIN" -m venv "$DIR/venv"
PIP="$DIR/venv/bin/pip"
"$PIP" install -q --upgrade pip

# ⚠️ --no-deps: fish-speech declares 27 dependencies, one of which is
# pyaudio. It needs a C compiler and portaudio headers, it fails to build
# without them, and nothing in this project records audio.
"$PIP" install -q -e "$DIR/fish-speech" --no-deps

COMMON=(torchcodec triton numpy soundfile loguru tokenizers tqdm
        safetensors hydra-core transformers einops
        descript-audio-codec huggingface_hub)
if [ "$MODE" = pinned ]; then
    "$PIP" install -q "torch==2.8.0" "torchaudio==2.8.0" "${COMMON[@]}"
else
    "$PIP" install -q torch torchaudio "${COMMON[@]}"
fi

echo "  venv: $("$PIP" list 2>/dev/null | wc -l) packages, $(du -sh "$DIR/venv" | cut -f1)"
"$DIR/venv/bin/python" - <<'EOF'
import torch
print(f"  torch {torch.__version__}")
EOF

cat > "$DIR/run-smoke.sh" <<EOF
#!/usr/bin/env bash
# Smoke-test the repo against this clean room. Pass a text file and a clip.
set -eu
cd "$REPO"
export DRAMA3_FISH="$DIR/fish-speech"
export DRAMA3_VENV="$DIR/venv"
source bin/env.sh
rm -rf torch-ext-cache          # force the CUDA kernel to rebuild
exec bin/run.sh bin/narrate.py \\
  --text-file "\${1:?need a text file}" --ref "\${2:?need a reference clip}" \\
  --ref-text "\${3:-placeholder reference transcript}" \\
  --out-dir "$DIR/out" --mode paragraph \\
  --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \\
  --vram-reserve 2 --prefill-chunk 1024 \\
  --prefix-kv --slice-lm-head --offload-embed --limit 2
EOF
chmod +x "$DIR/run-smoke.sh"

echo
echo "built. now:  $DIR/run-smoke.sh DOC.txt CLIP.mp3 'what the clip says'"
[ "$MODE" = latest ] && echo "⚠️  'latest' mode: do NOT quote timings from it as project numbers."
exit 0
