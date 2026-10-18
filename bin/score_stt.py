"""Score a run's audio against its source text, chunk by chunk.

STT is not a transcript oracle -- it normalises numbers, drops punctuation,
and mishears proper nouns. So this reports a word-level similarity and flags
the tail, rather than demanding equality. What it is actually hunting is
DROPPED CONTENT: a chunk where the model skipped a clause reads as a sharp
drop in coverage, which is how the bullet-list bug was found.
"""
import json, re, sys, pathlib
from difflib import SequenceMatcher

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/paragraph-v4")
src_file = sys.argv[2] if len(sys.argv) > 2 else \
    "test script/design a sql server migration strategy.txt"
m = json.loads((d / "manifest.json").read_text())
cs = [c for c in m["chunks"] if c.get("ok") and c.get("stt")]

# ⚠️ manifest["text"] is chunk[:200] -- narrate.py truncates it for
# readability. Comparing full speech against a truncated source produced
# "coverage 2.00" and nonsense similarities on the first attempt. Rebuild
# the real chunk texts from the source with the same splitter.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import narrate
full = narrate.paragraphs(pathlib.Path(src_file).read_text(encoding="utf-8"), 400)
if len(full) >= max(c["index"] for c in cs) + 1:
    for c in cs:
        c["text"] = full[c["index"]]
else:
    print(f"!! splitter gave {len(full)} chunks, manifest has "
          f"{max(c['index'] for c in cs)+1}; falling back to truncated text")

def words(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

rows = []
for c in cs:
    want, got = words(c["text"]), words(c["stt"])
    if not want:
        continue
    ratio = SequenceMatcher(None, want, got).ratio()
    cover = len(got) / len(want)
    rows.append((ratio, cover, c))

rows.sort(key=lambda r: r[0])
n = len(rows)
print(f"{d}: {n} chunks with STT\n")
import statistics
rs = [r[0] for r in rows]
print(f"word-similarity  median {statistics.median(rs):.3f}  mean {statistics.mean(rs):.3f}")
for thr in (0.95, 0.90, 0.80, 0.60):
    print(f"  >= {thr:.2f}: {sum(1 for r in rs if r >= thr):>4}/{n}  "
          f"({100*sum(1 for r in rs if r >= thr)/n:.1f}%)")
print(f"\n--- worst 12 (likely dropped or mangled content) ---")
for ratio, cover, c in rows[:12]:
    print(f"  sim {ratio:.2f} cover {cover:.2f}  chunk {c['index']:>3}  {c['seconds']:>5.1f}s")
    print(f"      want: {c['text'][:88]}")
    print(f"      got : {c['stt'][:88]}")
