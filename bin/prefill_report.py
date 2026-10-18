"""Read a --prefill-probe dump and answer: is prefill PCIe-bound, and would
using the resident device cache fix it?

  bin/run.sh bin/prefill_report.py out/prefill-probe.json

Four questions, in the order that decides what to build:

  1. Where does a prefill batch's time go -- gather or GEMM? If it is the
     GEMM there is nothing to win by moving bytes around.
  2. Does the measured gather time match bytes / PCIe bandwidth? If it does,
     prefill is bandwidth-bound and the only fix is fewer bytes. This is the
     OPPOSITE of decode, where dispatch was the wall.
  3. How much of each batch's working set is ALREADY in VRAM? The gather
     path ignores the device cache, so its hit rate is zero by construction
     rather than because the experts are absent. This number is the ceiling
     on what a hybrid gather (D2D for hits, PCIe for misses) could save.
  4. How much do consecutive chunks' prefill working sets overlap? That is
     the ceiling on what a prefill-aware eviction policy could add on top.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Measured on this box in bin/bench_pin.py: pinned host -> device over
# PCIe 4.0 x16. Pageable is ~1.47x slower, which is why --pin-experts exists.
PINNED_GB_PER_S = 23.0

data = json.loads(Path(sys.argv[1]).read_text())
records = data["records"]
expert_bytes = data["expert_bytes"]
if not records:
    sys.exit("no prefill records -- was --prefill-probe given a run with prefill?")

# ---------------------------------------------------------------- 1. split
gather = sum(r["gather_ms"] for r in records)
compute = sum(r["compute_ms"] for r in records)
total = gather + compute
batches = len({(r["chunk"], r["layer"]) for r in records}) // max(
    1, len({r["layer"] for r in records}))
print(f"{len(records)} layer-batches over {len({r['chunk'] for r in records})} chunks")
print(f"\n1 WHERE THE TIME GOES")
print(f"  gather   {gather/1000:8.2f} s  {100*gather/total:5.1f}%")
print(f"  compute  {compute/1000:8.2f} s  {100*compute/total:5.1f}%")

# ------------------------------------------------------------ 2. bandwidth
fetched = sum(r["distinct"] for r in records)
gib = fetched * expert_bytes / 2**30
predicted = gib * 2**30 / (PINNED_GB_PER_S * 1e9)
print(f"\n2 IS IT BANDWIDTH?")
print(f"  experts fetched      {fetched:,} x {expert_bytes/2**20:.3f} MiB = "
      f"{gib:.1f} GiB")
print(f"  at {PINNED_GB_PER_S:.0f} GB/s that is {predicted:.2f} s; "
      f"measured gather {gather/1000:.2f} s")
print(f"  -> gather is running at {gib*2**30/1e9/(gather/1000):.1f} GB/s "
      f"({100*predicted/(gather/1000):.0f}% of the pinned PCIe roof)")

# -------------------------------------------------------------- 3. resident
distinct = sum(r["distinct"] for r in records)
resident = sum(r["resident"] for r in records)
# records[0] is taken during the very first batch, when the cache is still
# filling, so it reports a near-empty cache. The capacity is the high-water
# mark. Reading [0] said "12 experts/layer" against a measured 55 resident,
# which is impossible and is how this bug was caught.
cache = max(r["cache_size"] for r in records)
print(f"\n3 HOW MUCH IS ALREADY IN VRAM")
print(f"  device cache holds up to {cache} experts/layer")
print(f"  of {distinct:,} experts the gather fetched, {resident:,} "
      f"({100*resident/max(distinct,1):.1f}%) were already resident")
print(f"  a hybrid gather would move {(distinct-resident)*expert_bytes/2**30:.1f} "
      f"GiB instead of {gib:.1f} GiB")
saved = gather/1000 * resident / max(distinct, 1)
print(f"  -> upper bound on the saving: {saved:.2f} s of {gather/1000:.2f} s "
      f"of gather")

# Is the cache doing badly, or is it just small? A batch wants most of the
# 256 experts, so almost any cached expert is a wanted one -- which means a
# cleverer eviction policy has almost nothing to win and only capacity does.
# Quantify it rather than asserting it.
batches = len({(r["chunk"], r["layer"]) for r in records})
perfect = min(cache, distinct / batches) * batches
print(f"  a PERFECT policy (every cached slot wanted by this batch) would "
      f"reach {100*perfect/max(distinct,1):.1f}%, against {100*resident/max(distinct,1):.1f}% now")
print(f"  -> policy is worth {100*(perfect-resident)/max(distinct,1):.1f} points; "
      f"the rest is capacity")

# --------------------------------------------------------------- 4. overlap
by_chunk: dict[int, dict[int, set]] = defaultdict(dict)
for r in records:
    by_chunk[r["chunk"]].setdefault(r["layer"], set()).update(r["experts"])
chunks = sorted(by_chunk)
overlaps = []
for previous, current in zip(chunks, chunks[1:]):
    shared = union = 0
    for layer, experts in by_chunk[current].items():
        before = by_chunk[previous].get(layer, set())
        shared += len(experts & before)
        union += len(experts)
    if union:
        overlaps.append(shared / union)
coverage = (distinct / batches) / 256
print(f"\n4 DO CONSECUTIVE CHUNKS WANT THE SAME EXPERTS?")
if overlaps:
    print(f"  chunk N's prefill set overlaps chunk N-1's by "
          f"{100*sum(overlaps)/len(overlaps):.1f}% on average "
          f"(min {100*min(overlaps):.1f}%, max {100*max(overlaps):.1f}%)")
    # ⚠️ Read this together with #3 before concluding anything. A single
    # batch already touches `coverage` of all 256 experts, so two batches
    # overlapping heavily is close to arithmetic rather than evidence of a
    # reusable working set. High overlap here does NOT mean a smarter
    # eviction policy would pay; #3 is the line that answers that.
    print(f"  but one batch already touches {100*coverage:.0f}% of all 256 "
          f"experts, so {100*coverage:.0f}% overlap is the floor from chance alone")
else:
    print("  only one chunk in the dump; run more to measure this")

# --------------------------------------------------------------- per-layer
print(f"\nPER-LAYER (mean over batches)")
per_layer = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
for r in records:
    row = per_layer[r["layer"]]
    row[0] += 1
    row[1] += r["distinct"]
    row[2] += r["resident"]
    row[3] += r["gather_ms"]
    row[4] += r["compute_ms"]
print(f"  {'layer':>5s} {'distinct':>9s} {'resident':>9s} {'gather ms':>10s} "
      f"{'compute ms':>11s}")
for layer in sorted(per_layer):
    n, d, res, g, c = per_layer[layer]
    print(f"  {layer:5d} {d/n:9.1f} {res/n:9.1f} {g/n:10.2f} {c/n:11.2f}")
