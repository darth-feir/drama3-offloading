"""Replay a router trace against cache policies. No GPU, no model needed.

The point of separating this from bin/offload.py: measuring routing is
expensive (it needs the 36.9 GiB model resident), but *evaluating a policy* is
arithmetic over a list of ints. Capture once, simulate as many policies as you
like.

Reads notes/route-trace.json as written by bin/offload.py.
"""

from __future__ import annotations

import argparse, json, math
from collections import Counter, OrderedDict
from pathlib import Path

EXPERT_BYTES = 3_146_496          # measured, see notes/01-static-findings.md
LAYERS = 40
EXPERTS = 256
GiB = 1024 ** 3


def load(path: Path, phase: str | None):
    data = json.loads(path.read_text())
    recs = data["records"]
    if phase:
        recs = [r for r in recs if r["phase"] == phase]
    return data["meta"], recs


def skew(recs) -> None:
    """Is routing uniform? If it is, the whole offloading plan is worth less."""
    per_layer = [Counter() for _ in range(LAYERS)]
    for r in recs:
        flat = [i for row in r["ids"] for i in (row if isinstance(row, list) else [row])]
        per_layer[r["layer"]].update(flat)

    print(f"{'layer':>5} {'seen':>5} {'entropy':>8} {'gini':>6} {'top8%':>7} {'top32%':>7}")
    ents = []
    for layer, c in enumerate(per_layer):
        total = sum(c.values())
        if not total:
            continue
        p = [v / total for v in c.values()]
        ent = -sum(x * math.log2(x) for x in p if x)
        ents.append(ent)
        vals = sorted(c.values(), reverse=True)
        cum = sum(vals)
        top8 = 100 * sum(vals[:8]) / cum
        top32 = 100 * sum(vals[:32]) / cum
        n = len(vals)
        srt = sorted(c.values())
        gini = (2 * sum((i + 1) * v for i, v in enumerate(srt))) / (n * sum(srt)) - (n + 1) / n
        if layer < 6 or layer >= LAYERS - 3:
            print(f"{layer:>5} {len(c):>5} {ent:>8.3f} {gini:>6.3f} {top8:>7.1f} {top32:>7.1f}")
    if ents:
        print(f"\nmean entropy {sum(ents)/len(ents):.3f} bits "
              f"(uniform over 256 = {math.log2(EXPERTS):.3f}; "
              f"over the {sum(len(c) for c in per_layer)/LAYERS:.0f} actually seen = "
              f"{math.log2(max(1,sum(len(c) for c in per_layer)/LAYERS)):.3f})")
        print("-> near-uniform entropy means poor cacheability; low entropy means skew.")


def simulate(recs, capacity: int, policy: str) -> tuple[int, int]:
    """Returns (hits, total) for a per-layer cache of `capacity` experts."""
    if policy == "static":
        freq = [Counter() for _ in range(LAYERS)]
        for r in recs:
            flat = [i for row in r["ids"] for i in (row if isinstance(row, list) else [row])]
            freq[r["layer"]].update(flat)
        keep = [set(e for e, _ in f.most_common(capacity)) for f in freq]

    caches = [OrderedDict() for _ in range(LAYERS)]
    hits = total = 0
    for r in recs:
        layer = r["layer"]
        flat = [i for row in r["ids"] for i in (row if isinstance(row, list) else [row])]
        for e in flat:
            total += 1
            if policy == "static":
                hits += e in keep[layer]
                continue
            c = caches[layer]
            if e in c:
                hits += 1
                if policy == "lru":
                    c.move_to_end(e)
                else:
                    c[e] += 1
            else:
                if len(c) >= capacity:
                    if policy == "lru":
                        c.popitem(last=False)
                    else:
                        c.pop(min(c, key=c.get))
                c[e] = 1
    return hits, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="notes/route-trace.json")
    ap.add_argument("--phase", default="decode", help="decode | prefill | all")
    ap.add_argument("--tok-per-sec", type=float, default=None,
                    help="rate to price PCIe demand at; default = measured decode rate")
    ap.add_argument("--budgets-gib", default="10,12,13.3,15,20",
                    help="VRAM budgets for expert storage")
    args = ap.parse_args()

    meta, recs = load(Path(args.trace), None if args.phase == "all" else args.phase)
    print(f"trace: {len(recs):,} records, phase={args.phase}")
    print(f"model ran at {meta.get('tokens_per_second')} tok/s\n")

    print("=== routing skew ===")
    skew(recs)

    print("\n=== cache hit rate ===")
    tps = (args.tok_per_sec or meta.get("decode_tokens_per_second")
           or meta.get("tokens_per_second") or 0)
    print(f"pricing PCIe demand at {tps:.2f} tok/s")
    per_token_gib = 8 * LAYERS * EXPERT_BYTES / GiB
    print(f"{'budget':>8} {'per-layer':>10} {'policy':>8} {'hit%':>7} {'GiB/s':>8}")
    for b in [float(x) for x in args.budgets_gib.split(",")]:
        cap = int(b * GiB / (LAYERS * EXPERT_BYTES))
        cap = max(1, min(EXPERTS, cap))
        for policy in ("lru", "lfu", "static"):
            hits, total = simulate(recs, cap, policy)
            hr = hits / total if total else 0
            print(f"{b:>7.1f}G {cap:>10} {policy:>8} {100*hr:>6.1f}% "
                  f"{tps*per_token_gib*(1-hr):>7.2f}")
    print("\n'static' keeps the globally hottest N per layer -- an ORACLE, it")
    print("peeks at the whole trace. It is the ceiling a frequency policy could")
    print("reach, not something implementable without a profiling pass.")


if __name__ == "__main__":
    main()
