#!/usr/bin/env bash
# A/B the prefix-KV reuse on real chunks of the real script.
#
# Two separate processes, not two arms in one, because --prefix-kv changes
# what sits in the caches from load time onwards and sharing a process would
# let arm A warm something for arm B. The model load is paid twice; that is
# the price of an arm that cannot contaminate the other.
#
# ⚠️ Calls nix-shell directly rather than bin/run.sh. run.sh ends in
# `--run "$PY $*"`, which word-splits, and the script path has spaces in it.
# Every other long-form runner here does the same for the same reason.
#
# Everything else is the production configuration from notes/08-STATE-AND-NEXT.md.
set -u
cd "$(dirname "$0")/.."
source bin/env.sh
PY="$DRAMA3_VENV/bin/python"

LIMIT="${LIMIT:-20}"
REF_SECONDS="${REF_SECONDS:-30}"
SCRIPT="${SCRIPT:-test script/design a sql server migration strategy.txt}"
TAG="${TAG:-ab}"

for arm in off on; do
    dir="out/prefix-$TAG-$arm"
    rm -rf "$dir"                      # a stale manifest would skip every chunk
    flag=""
    [ "$arm" = on ] && flag="--prefix-kv"
    echo "######## arm $arm -> $dir :: $(date +%H:%M:%S) ########"
    nix-shell -p gcc --run "$PY bin/narrate.py \
      --checkpoint '$DRAMA3_CKPT' \
      --text-file '$SCRIPT' \
      --mode paragraph --out-dir '$dir' \
      --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \
      --vram-reserve 2 --prefill-chunk 1024 --ref-seconds $REF_SECONDS \
      --limit $LIMIT $flag" > "$dir.log" 2>&1
    echo "---- exit $? at $(date +%H:%M:%S) ----"
done

"$PY" - "$TAG" <<'EOF'
import json, sys
from pathlib import Path
tag = sys.argv[1]
print(f"\n{'arm':6s} {'chunks':>6s} {'audio s':>9s} {'gen s':>9s} {'RTF':>6s} "
      f"{'prefill s':>10s} {'rows kept':>10s} {'retries':>8s}")
rows = {}
for arm in ("off", "on"):
    path = Path(f"out/prefix-{tag}-{arm}/manifest.json")
    if not path.is_file():
        print(f"{arm:6s} (no manifest)"); continue
    chunks = [c for c in json.loads(path.read_text())["chunks"] if c.get("ok")]
    audio = sum(c["seconds"] for c in chunks)
    gen = sum(c["gen_seconds"] for c in chunks)
    prefill = sum(c.get("prefix", {}).get("prefill_seconds", 0.0) for c in chunks)
    kept = chunks[0].get("prefix", {}).get("prefix_rows", 0) if chunks else 0
    # A retry means the first draw's duration implied dropped or repeated
    # text. Counting them keeps a "faster" arm from hiding a quality loss.
    retries = sum(len(c.get("attempts", [])) - 1 for c in chunks)
    rows[arm] = (len(chunks), audio, gen)
    print(f"{arm:6s} {len(chunks):6d} {audio:9.1f} {gen:9.1f} {gen/max(audio,1e-9):6.3f} "
          f"{prefill:10.2f} {kept:10d} {retries:8d}")
if len(rows) == 2 and rows["off"][0] == rows["on"][0]:
    off, on = rows["off"][2], rows["on"][2]
    print(f"\ngeneration time {off:.1f}s -> {on:.1f}s: {100*(1-on/off):.1f}% less, "
          f"{(off-on)/rows['off'][0]*1000:.0f} ms saved per chunk")
else:
    print("\narms produced different chunk counts -- not comparable as totals")
EOF
