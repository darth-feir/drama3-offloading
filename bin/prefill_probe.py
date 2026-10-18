"""Measure where a prefill batch's time actually goes, per layer.

WHY THIS EXISTS
---------------
After `--prefix-kv` landed, prefill is still 1.12 s per chunk, ~10 % of
runtime, and the per-batch timings say it is not about row count:

     11 rows -> 0.28 s        453 rows -> 1.25 s
     71 rows -> 0.80 s        625 rows -> 1.26 s

57x the rows for 4.5x the time, and an 1180-row prompt (two batches at
--prefill-chunk 1024) costs almost exactly double a 745-row one (one batch).
Cost tracks BATCHES, and the arithmetic says the batch cost is bytes:

    ~200 distinct experts x 40 layers x 3.001 MiB = 23.4 GiB
    23.4 GiB / ~23 GB/s of pinned PCIe             =  1.04 s   (measured 1.26)

If that is right then prefill is PCIe-bound -- the opposite of decode, where
this project's headline finding is that dispatch, not bandwidth, was the
wall. Attacking it as a launch-overhead problem would be aiming at the wrong
target, so measure before writing anything.

WHAT IT MEASURES
----------------
Per (batch, layer), via CUDA events so the async copies are actually waited
on without a host sync storm:

  rows            rows in this prefill batch
  selections      rows x 8, the raw routing decisions
  distinct        distinct experts this layer must have resident
  resident        how many of those are ALREADY in VRAM in the device cache
  gather_ms       time to get the weights onto the GPU
  compute_ms      time in fp8_moe_forward afterwards

`resident` is the headline. `offload.offloaded_forward` is the prefill path
and it ignores the device cache completely -- it fetches all `distinct`
experts over PCIe every single batch, so its hit rate is zero BY
CONSTRUCTION, not because the experts are not there. If a useful fraction of
each batch's working set is already sitting in VRAM, a hybrid gather
(device-to-device for the hits at ~1.5 TB/s, PCIe only for the misses) is
most of a fix and costs no extra VRAM.

It also records the expert set per (batch, layer) so cross-batch overlap can
be computed offline: if consecutive chunks' prefill working sets overlap,
a prefill-aware eviction policy is worth more than a bigger cache.

Usage: bin/narrate.py --prefill-probe PATH.json ...
Analysis: bin/prefill_report.py PATH.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

RECORDS: list[dict] = []
ENABLED = False
# Bumped by the driver so a record can be attributed to the chunk that caused
# it. Batches within one chunk are told apart by their order.
CHUNK = -1

EXPERT_BYTES = 3.001 * 2**20


def resident_experts(experts) -> set[int]:
    """Expert ids currently held in this layer's device cache.

    Reads `_dc_slot_expert`, the devcache's slot -> expert map. Empty slots
    hold -1. This is a host sync, which is why the probe is opt-in: it would
    be a real cost in production.
    """
    table = getattr(experts, "_dc_slot_expert", None)
    if table is None:
        return set()
    return {int(e) for e in table.tolist() if e >= 0}


def install(model) -> None:
    """Wrap the prefill gather path with per-layer timing and census."""
    global ENABLED
    import offload

    original = offload.offloaded_forward

    def probed(self, x, topk_ids, topk_weights):
        if not ENABLED or topk_ids.shape[0] == 1:
            # One row is a decode step; it goes through the device cache and
            # is already understood. Only prefill batches are of interest.
            return original(self, x, topk_ids, topk_weights)
        ids = topk_ids.reshape(-1)
        distinct = torch.unique(ids).tolist()
        cached = resident_experts(self)
        start = torch.cuda.Event(enable_timing=True)
        gathered = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        offload.GATHER_DONE_HOOK = gathered.record
        start.record()
        try:
            out = original(self, x, topk_ids, topk_weights)
        finally:
            offload.GATHER_DONE_HOOK = None
        end.record()
        end.synchronize()
        RECORDS.append({
            "chunk": CHUNK,
            "layer": int(getattr(self, "_layer_index", -1)),
            "rows": int(topk_ids.shape[0]),
            "selections": int(ids.numel()),
            "distinct": len(distinct),
            "resident": len(set(distinct) & cached),
            "cache_size": len(cached),
            "gather_ms": float(start.elapsed_time(gathered)),
            "compute_ms": float(gathered.elapsed_time(end)),
            "total_ms": float(start.elapsed_time(end)),
            "experts": distinct,
        })
        return out

    offload.offloaded_forward = probed
    # devcache.dev_cached_forward holds its own reference to the module, so
    # rebinding the module attribute is enough -- it looks the name up at
    # call time. Verified: devcache.py:258 does `import offload` inside the
    # function and calls `offload.offloaded_forward(...)`.
    ENABLED = True


def save(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "expert_bytes": EXPERT_BYTES,
        "records": RECORDS,
    }))
    print(f"prefill probe: {len(RECORDS)} layer-batches -> {path}", flush=True)
