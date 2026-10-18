"""Compare expert-usage censuses across languages.

The question: does Drama 3 route by LANGUAGE (so an English-only model could
drop the Mandarin experts) or by something else -- phonetics, syntax, prosody
-- in which case language-stripping buys nothing beyond the vocabulary.

Presence/absence cannot answer it: the saturation curve shows English alone
heads toward touching nearly all 256 experts per layer given enough text. The
answer is in the call DISTRIBUTION, and in how much of each language's
traffic the other language's hot set already covers.
"""
import json, sys, pathlib
import numpy as np

DECODE_ONLY = "--all" not in sys.argv

def load(p):
    d = json.loads(pathlib.Path(p).read_text())
    key = "census_decode" if (DECODE_ONLY and "census_decode" in d) else "census"
    return d["label"] + ("" if key == "census" else " [decode]"), \
           np.array(d[key], dtype=np.int64), d.get("audio_seconds", 0)

def coverage(c, keep):
    """Fraction of calls served if only the top `keep` experts per layer exist."""
    tot = c.sum()
    if not tot: return 0.0
    kept = sum(np.sort(c[L])[::-1][:keep].sum() for L in range(c.shape[0]))
    return kept / tot

paths = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(paths) < 1:
    print("usage: census_compare.py notes/census-english.json [notes/census-chinese.json]")
    sys.exit(1)

cs = [load(p) for p in paths]
for label, c, secs in cs:
    used = [(c[L] > 0).sum() for L in range(c.shape[0])]
    print(f"\n=== {label}: {secs/60:.1f} min of audio, {c.sum():,} routing decisions ===")
    print(f"  experts touched per layer: min {min(used)}, median {sorted(used)[len(used)//2]}, max {max(used)} of {c.shape[1]}")
    print(f"  {'keep top N':>11} {'% of calls served':>19} {'GiB saved':>11}")
    for keep in (256, 224, 192, 160, 128, 96, 64):
        cov = coverage(c, keep)
        print(f"  {keep:>11} {100*cov:>18.3f}% {(256-keep)*40*3_146_496/2**30:>10.1f}G")

if len(cs) == 2:
    (la, ca, _), (lb, cb, _) = cs
    print(f"\n=== {la} vs {lb} ===")
    # per-layer rank correlation of call counts
    from itertools import product
    rhos = []
    for L in range(ca.shape[0]):
        a, b = ca[L].astype(float), cb[L].astype(float)
        ra, rb = a.argsort().argsort(), b.argsort().argsort()
        rho = np.corrcoef(ra, rb)[0, 1]
        rhos.append(rho)
    print(f"  Spearman rank correlation of expert usage, per layer:")
    print(f"    mean {np.mean(rhos):.3f}, min {np.min(rhos):.3f}, max {np.max(rhos):.3f}")
    print(f"    (1.0 = identical ordering -> no language specialisation;")
    print(f"     0.0 = unrelated -> strong specialisation)")
    # how much of B's traffic do A's top experts cover, and vice versa?
    for (l1, c1), (l2, c2) in (((la, ca), (lb, cb)), ((lb, cb), (la, ca))):
        for keep in (128, 192):
            served = 0
            for L in range(c1.shape[0]):
                top = np.argsort(c1[L])[::-1][:keep]
                served += c2[L][top].sum()
            print(f"  {l1}'s top {keep} serve {100*served/max(c2.sum(),1):6.2f}% of {l2}'s calls")
    # experts one language uses and the other never does
    only_a = int(((ca > 0) & (cb == 0)).sum())
    only_b = int(((cb > 0) & (ca == 0)).sum())
    both = int(((ca > 0) & (cb > 0)).sum())
    print(f"\n  (layer,expert) slots used by {la} only: {only_a:,}")
    print(f"  (layer,expert) slots used by {lb} only: {only_b:,}")
    print(f"  used by both:                           {both:,}")
    print(f"  -> dropping {lb}-only slots would free "
          f"{only_b*3_146_496/2**30:.2f} GiB")
