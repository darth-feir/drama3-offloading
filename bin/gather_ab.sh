#!/usr/bin/env bash
# A/B the hybrid prefill gather: does borrowing resident experts from the
# device cache actually cut prefill time by the predicted 28 %?
#
# Both arms carry --prefix-kv, because that is now the production config and
# the question is what the gather adds ON TOP of it.
#
# ⚠️ nix-shell directly, not bin/run.sh: run.sh ends in `--run "$PY $*"`,
# which word-splits, and the script path has spaces.
set -u
cd "$(dirname "$0")/.."
source bin/env.sh
PY="$DRAMA3_VENV/bin/python"

LIMIT="${LIMIT:-20}"
SCRIPT="${SCRIPT:-test script/design a sql server migration strategy.txt}"
TAG="${TAG:-gather}"

for arm in off on; do
    dir="out/$TAG-$arm"
    rm -rf "$dir"
    flag=""
    [ "$arm" = off ] && flag="--no-hybrid-gather"
    echo "######## hybrid gather $arm -> $dir :: $(date +%H:%M:%S) ########"
    nix-shell -p gcc --run "$PY bin/narrate.py \
      --checkpoint '$DRAMA3_CKPT' \
      --text-file '$SCRIPT' \
      --mode paragraph --out-dir '$dir' \
      --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \
      --vram-reserve 2 --prefill-chunk 1024 --prefix-kv \
      --limit $LIMIT $flag" > "$dir.log" 2>&1
    echo "---- exit $? at $(date +%H:%M:%S) ----"
    grep -E "prefill gather:|device cache:" "$dir.log" || true
done

"$PY" - "$TAG" <<'EOF'
import json, re, sys
from pathlib import Path
tag = sys.argv[1]
print(f"\n{'arm':6s} {'chunks':>6s} {'audio s':>9s} {'gen s':>9s} {'RTF':>6s} "
      f"{'prefill s':>10s} {'retries':>8s}")
rtf = {}
for arm in ("off", "on"):
    path = Path(f"out/{tag}-{arm}/manifest.json")
    if not path.is_file():
        print(f"{arm:6s} (no manifest)"); continue
    chunks = [c for c in json.loads(path.read_text())["chunks"] if c.get("ok")]
    audio = sum(c["seconds"] for c in chunks)
    gen = sum(c["gen_seconds"] for c in chunks)
    # The logged per-batch prefill seconds are the honest measure here: the
    # arms produce different audio (sampling diverges), so comparing total
    # gen seconds silently compares "how much it had to say" as well.
    log = Path(f"out/{tag}-{arm}.log").read_text()
    prefill = sum(float(m) for m in
                  re.findall(r"Drama 3 prefill:.*? in ([0-9.]+)s", log))
    retries = sum(len(c.get("attempts", [])) - 1 for c in chunks)
    rtf[arm] = (gen / max(audio, 1e-9), prefill, len(chunks))
    print(f"{arm:6s} {len(chunks):6d} {audio:9.1f} {gen:9.1f} "
          f"{gen/max(audio,1e-9):6.3f} {prefill:10.2f} {retries:8d}")
if len(rtf) == 2:
    (r0, p0, n0), (r1, p1, n1) = rtf["off"], rtf["on"]
    print(f"\nprefill  {p0:.2f}s -> {p1:.2f}s  ({100*(1-p1/max(p0,1e-9)):.1f}% less, "
          f"{(p0-p1)/max(n0,1)*1000:.0f} ms per chunk)")
    print(f"RTF      {r0:.3f} -> {r1:.3f}  ({100*(1-r1/max(r0,1e-9)):.1f}% faster "
          f"per second of audio)")
EOF
