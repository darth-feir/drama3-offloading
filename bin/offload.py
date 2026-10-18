"""Run Drama 3 with routed experts resident in CPU RAM, and trace the router.

Why this shape and not a CPU-only run: a pure-CPU placement makes
`fp8_gemm_supported()` return False, which makes the loader DEQUANTIZE every
expert to bf16 on load -- 30 GiB of FP8 becomes 60 GiB. Keeping the compute
device CUDA keeps the experts FP8 and keeps the Triton W8A8 kernel on the fast
path. The only thing that moves to RAM is the expert weight storage.

The gather-and-remap trick: `fp8_moe_forward` reads its expert count from
`w13_weight.shape[0]`, so handing it only the experts this step actually hit,
with topk_ids renumbered into that subset, is legal and needs no kernel change.

This is deliberately the NAIVE version -- every miss is a fresh H2D copy, no
cache, no second stream. It exists to answer "what does the router pick?"
before any cache is written. Policies are simulated offline from the trace it
writes, so cache design never requires re-running the model.
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni import moe as moe_mod
from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts, select_fp8
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt,
    generate_drama3,
)


# ---------------------------------------------------------------- trace

class RouterTrace:
    """Every (layer, token) -> 8 expert ids, kept as plain python lists.

    Small enough not to matter: 40 layers x 8 ids x a few hundred tokens.
    Phase is recorded because prefill routes many tokens at once and decode
    routes one; a cache policy behaves very differently under the two.
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.phase = "prefill"
        self.enabled = True
        self.first_decode: float | None = None

    def add(self, layer: int, topk_ids: Tensor) -> None:
        if not self.enabled:
            return
        # Prefill routes many rows in one call, decode exactly one. Reading the
        # phase off the shape beats tracking it externally, which went wrong
        # the moment prefill chunking changed.
        rows = topk_ids.shape[0]
        if rows == 1 and self.first_decode is None:
            self.first_decode = time.perf_counter()
        self.records.append(
            {
                "phase": "decode" if rows == 1 else "prefill",
                "layer": layer,
                "ids": topk_ids.tolist(),
            }
        )

    def save(self, path: Path, meta: dict) -> None:
        path.write_text(json.dumps({"meta": meta, "records": self.records}))


TRACE = RouterTrace()


# ---------------------------------------------------------------- offload


def memory_snapshot(device) -> dict:
    """VRAM and host RAM high-water marks for this process.

    VmHWM is read from /proc rather than psutil (not installed) and is the
    peak resident set, which is what matters here: the experts live in host
    RAM, so a run's real cost is VRAM *and* RAM together.
    """
    import torch
    out = {}
    if torch.cuda.is_available():
        out["vram_allocated_gib"] = round(torch.cuda.memory_allocated(device) / 2**30, 2)
        out["vram_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        out["vram_reserved_gib"] = round(torch.cuda.memory_reserved(device) / 2**30, 2)
        free, total = torch.cuda.mem_get_info(device)
        out["vram_free_gib"] = round(free / 2**30, 2)
        out["vram_total_gib"] = round(total / 2**30, 2)
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                out["ram_peak_gib"] = round(int(line.split()[1]) / 2**20, 2)
            elif line.startswith("VmRSS:"):
                out["ram_rss_gib"] = round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return out

def materialize_offloaded(backbone, device: torch.device) -> None:
    """Like FishOmniBackbone.materialize([device]) but expert storage lands in RAM.

    Peak VRAM stays low because materialize is per-layer: a layer's expert
    buffers are allocated on the GPU by `to_empty`, then immediately
    reallocated on the CPU, freeing them before the next layer. The transient
    is one layer's experts (~768 MiB), not all forty.
    """
    backbone.model.embed_tokens.to_empty(device=device)
    backbone.model.norm.to_empty(device=device)
    backbone.lm_head.to_empty(device=device)
    for index, layer in enumerate(backbone.model.layers):
        layer.to_empty(device=device)
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:          # dense layer; nothing to offload
            continue
        experts.to_empty(device="cpu")
        experts._layer_index = index
        experts._compute_device = device
        if CACHE_EXPERTS:
            _init_cache(experts, device, min(CACHE_EXPERTS, experts.num_experts))
    backbone.inverse_frequencies.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def offloaded_forward(self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> Tensor:
    """Copy just the experts this step hit to the GPU, then run the stock kernel."""
    _count(getattr(self, "_layer_index", -1), topk_ids)
    TRACE.add(getattr(self, "_layer_index", -1), topk_ids)
    if self.w13_weight.device.type != "cpu":
        return MoEExperts._stock_forward(self, x, topk_ids, topk_weights)
    # Dispatch on the per-layer capacity alone. Checking the CACHE_EXPERTS
    # global as well was a bug: --vram-reserve allocates the cache without
    # setting that global, so 8.32 GiB of cache sat allocated and unused and
    # the run was silently no faster than having none. The missing hit-rate
    # line in the log was the only clue.
    # graphstub is a diagnostic that must override everything: it is the only
    # sync-free path, and a cache allocated by --vram-reserve would otherwise
    # shadow it and send capture straight into cached_forward's .cpu() sync.
    if COPY_MODE == "graphstub":
        return _graph_stub(self, x, topk_ids, topk_weights, x.device)
    if getattr(self, "_cap", 0):
        return cached_forward(self, x, topk_ids, topk_weights)

    device = x.device
    # unique() must run where the ids are, but indexing CPU storage needs CPU
    # ids -- this .cpu() is a host sync every layer and is a real cost of the
    # naive path. A cache removes most of it by making the copy conditional.
    uniq, inverse = torch.unique(topk_ids.reshape(-1).cpu(), return_inverse=True)
    local_ids = inverse.reshape(topk_ids.shape).to(device)
    n = int(uniq.numel())

    if COPY_MODE == "none":
        # TIMING ONLY -- THE AUDIO IS GARBAGE. Reuses one resident buffer of
        # junk weights instead of fetching the right experts, so the kernel
        # does identical work with zero transfer. This measures the compute
        # floor: the decode rate a PERFECT cache (100% hit) could never beat.
        buf = getattr(self, "_fake", None)
        if buf is None or buf[0].shape[0] < n:
            buf = (
                torch.zeros((n, *self.w13_weight.shape[1:]),
                            dtype=self.w13_weight.dtype, device=device),
                torch.zeros((n, *self.w2_weight.shape[1:]),
                            dtype=self.w2_weight.dtype, device=device),
                torch.ones((n, *self.w13_scale_inv.shape[1:]), device=device),
                torch.ones((n, *self.w2_scale_inv.shape[1:]), device=device),
            )
            self._fake = buf
        return fp8_moe_forward(x, local_ids, topk_weights,
                               buf[0][:n], buf[2][:n], buf[1][:n], buf[3][:n])

    if COPY_MODE == "gather":
        # select_fp8 materialises a [n, 2I, H] copy on the CPU first. Measured
        # at 1.43 ms/layer -- MORE than the 1.09 ms transfer it feeds, and the
        # two do not overlap. See bin/bench_copy.py.
        w13 = select_fp8(self.w13_weight, uniq).to(device, non_blocking=True)
        w2 = select_fp8(self.w2_weight, uniq).to(device, non_blocking=True)
    else:
        # Each expert is already contiguous in [E, 2I, H], so DMA it straight
        # into a GPU row. No CPU-side materialisation at all: 2.74 ms -> 1.07.
        w13 = torch.empty(
            (n, *self.w13_weight.shape[1:]), dtype=self.w13_weight.dtype, device=device
        )
        w2 = torch.empty(
            (n, *self.w2_weight.shape[1:]), dtype=self.w2_weight.dtype, device=device
        )
        # Take whatever the device cache already holds from VRAM instead of
        # pulling it over PCIe again.
        #
        # WHY: measured (notes/10) a prefill batch is 96.5 % weight transfer
        # running at 25.8 GB/s -- the pinned PCIe roof -- and 28.1 % of the
        # experts it fetches are ALREADY RESIDENT. This path never looked, so
        # its hit rate was zero by construction, not because the experts were
        # absent. A device-to-device copy is ~1.5 TB/s against ~25 GB/s.
        #
        # The lookup needs the slot table on the host, but `unique()` above
        # already paid a host sync, so this rides a sync that exists.
        #
        # ⚠️ READ-ONLY, DELIBERATELY. No insert, no stamp bump. A prefill
        # batch wants ~196 experts per layer against a 62-slot cache, so
        # letting it populate would evict decode's working set wholesale --
        # and decode is 80 % of runtime. Prefill borrows what happens to be
        # there and leaves the LRU ordering to decode. If you change this,
        # measure decode, not prefill.
        slot_of = getattr(self, "_dc_slot_of", None) if HYBRID_GATHER else None
        if slot_of is not None:
            # slot_of[e] >= 0 implies slot holds expert e: _resolve_slots
            # clears slot_of[victim] before reassigning (devcache.py:86-90),
            # and no decode step runs concurrently with a prefill batch.
            slots = slot_of.cpu().tolist()
            cache_w13, cache_w2 = self._dc_w13, self._dc_w2
            hits = 0
            for i, e in enumerate(uniq.tolist()):
                s = slots[e]
                if s >= 0:
                    w13[i].copy_(cache_w13[s])
                    w2[i].copy_(cache_w2[s])
                    hits += 1
                else:
                    w13[i].copy_(self.w13_weight[e], non_blocking=True)
                    w2[i].copy_(self.w2_weight[e], non_blocking=True)
            STATS["gather_hits"] += hits
            STATS["gather_misses"] += n - hits
        else:
            for i, e in enumerate(uniq.tolist()):
                w13[i].copy_(self.w13_weight[e], non_blocking=True)
                w2[i].copy_(self.w2_weight[e], non_blocking=True)
            STATS["gather_misses"] += n
    s13 = self.w13_scale_inv[uniq].to(device, non_blocking=True)
    s2 = self.w2_scale_inv[uniq].to(device, non_blocking=True)

    STATS["experts_copied"] += int(uniq.numel())
    STATS["bytes_copied"] += int(
        w13.numel() + w2.numel() + s13.numel() * 4 + s2.numel() * 4
    )
    # A seam for bin/prefill_probe.py to record a CUDA event between the
    # gather and the GEMM. Separating the two is the whole question there --
    # if prefill is PCIe-bound the fix is fewer bytes, if it is compute-bound
    # the fix is something else entirely -- and there is no other point in
    # this function where that boundary exists. None in production.
    if GATHER_DONE_HOOK is not None:
        GATHER_DONE_HOOK()
    return fp8_moe_forward(x, local_ids, topk_weights, w13, s13, w2, s2)


GATHER_DONE_HOOK = None


# hits/misses are the PYTHON cache (cached_forward). gather_hits/misses are
# the prefill gather borrowing from the DEVICE cache -- a different cache on
# a different path, kept separate because conflating them is how the "cache
# allocated but never used" check stayed wrong for a day.
STATS = {"experts_copied": 0, "bytes_copied": 0, "hits": 0, "misses": 0,
         "gather_hits": 0, "gather_misses": 0}
# Prefill borrows resident experts from the device cache instead of
# re-fetching them. See the comment at the gather. Off reproduces the old
# behaviour exactly, for A/B.
HYBRID_GATHER = True
# [layer, expert] call counts. A full id trace is ~4 MB per minute of audio
# and we only ever aggregate it; counts are 40x256 ints regardless of length.
CENSUS = None


CENSUS_DECODE = None


def enable_census(layers: int = 40, experts: int = 256) -> None:
    """Count prefill and decode separately.

    They are different regimes and mixing them is misleading: prefill routes
    over the PROMPT (reference transcript + history + the text to read) and,
    because a prompt is ~900 rows against one row per decode step, it
    outnumbers decode roughly 9:1 in raw call counts. Measured on a 1-minute
    smoke run: 3.82 M total decisions where decode alone accounts for ~413 k.

    Decode is the regime that matters for "what does speaking English use";
    prefill is text comprehension. Keep both, report both.
    """
    global CENSUS, CENSUS_DECODE
    CENSUS = np.zeros((layers, experts), dtype=np.int64)
    CENSUS_DECODE = np.zeros((layers, experts), dtype=np.int64)


def _count(layer: int, topk_ids: Tensor) -> None:
    if CENSUS is None or layer < 0:
        return
    flat = topk_ids.reshape(-1).to("cpu").numpy()
    counts = np.bincount(flat, minlength=CENSUS.shape[1])
    CENSUS[layer] += counts
    if topk_ids.shape[0] == 1:          # one row == a decode step
        CENSUS_DECODE[layer] += counts
COPY_MODE = "direct"
# Experts held in VRAM per layer. 0 disables the cache. Each costs 3.001 MiB,
# so N=96 is 96 * 40 * 3.001 MiB = 11.3 GiB on top of the 6.87 GiB of
# non-expert weights. Simulated hit rate at N=113 was 77.9%.
CACHE_EXPERTS = 0


def _init_cache(experts, device: torch.device, capacity: int) -> None:
    """Preallocate this layer's VRAM cache.

    Allocated up front rather than lazily so an over-large budget fails during
    load, not three chunks into a multi-hour narration.
    """
    from collections import OrderedDict
    experts._cap = capacity
    experts._cache_w13 = torch.zeros(
        (capacity, *experts.w13_weight.shape[1:]),
        dtype=experts.w13_weight.dtype, device=device)
    experts._cache_w2 = torch.zeros(
        (capacity, *experts.w2_weight.shape[1:]),
        dtype=experts.w2_weight.dtype, device=device)
    experts._cache_s13 = torch.ones(
        (capacity, *experts.w13_scale_inv.shape[1:]),
        dtype=torch.float32, device=device)
    experts._cache_s2 = torch.ones(
        (capacity, *experts.w2_scale_inv.shape[1:]),
        dtype=torch.float32, device=device)
    experts._slot_of = {}               # global expert id -> slot
    experts._lru = OrderedDict()        # slot -> expert id, oldest first
    experts._free = list(range(capacity))


STUB_EXPERTS = 8


def _graph_stub(self, x, topk_ids, topk_weights, device):
    # TIMING ONLY -- THE AUDIO IS GARBAGE.
    #
    # Measures CUDA-graph capture against eager on the real attention,
    # tower and sampling path, with a synthetic MoE that has NO host
    # sync. Graphs cannot capture the real path because every layer does
    # topk_ids.cpu() to decide what to fetch, which is illegal during
    # capture; this replaces that with `topk_ids % cap`, a pure GPU op.
    #
    # The expert weights are junk and resident, so the routing is wrong
    # and the output is noise. What stays real is the ~4,600 small kernel
    # launches per decode step across attention, the 13-layer tower and
    # 10 sampling passes -- which is where the dispatch overhead lives.
    cap = STUB_EXPERTS
    buf = getattr(self, "_stub", None)
    if buf is None:
        buf = (
            torch.zeros((cap, *self.w13_weight.shape[1:]),
                        dtype=self.w13_weight.dtype, device=device),
            torch.zeros((cap, *self.w2_weight.shape[1:]),
                        dtype=self.w2_weight.dtype, device=device),
            torch.ones((cap, *self.w13_scale_inv.shape[1:]), device=device),
            torch.ones((cap, *self.w2_scale_inv.shape[1:]), device=device),
        )
        self._stub = buf
    return fp8_moe_forward(x, topk_ids % cap, topk_weights,
                           buf[0], buf[2], buf[1], buf[3])


def init_graph_stub(model, device: torch.device) -> None:
    """Pre-allocate the stub buffers. Allocation inside a graph capture fails,
    and Drama3DecodeGraphs captures before the first real forward."""
    for layer in model.backbone.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:
            continue
        # ⚠️ NOT zeros. All-zero expert weights make the intermediate
        # activations all-zero, and quantising an all-zero tensor yields a
        # zero scale -> NaN. Eager tolerated it; graph capture did not, and
        # every chunk died with "non-finite logits". Small random values are
        # both non-degenerate and closer to real weight statistics.
        g = torch.Generator(device=device).manual_seed(0)
        experts._stub = (
            (torch.randn((STUB_EXPERTS, *experts.w13_weight.shape[1:]),
                         generator=g, device=device) * 0.2
             ).to(experts.w13_weight.dtype),
            (torch.randn((STUB_EXPERTS, *experts.w2_weight.shape[1:]),
                         generator=g, device=device) * 0.2
             ).to(experts.w2_weight.dtype),
            torch.full((STUB_EXPERTS, *experts.w13_scale_inv.shape[1:]), 0.1,
                       device=device),
            torch.full((STUB_EXPERTS, *experts.w2_scale_inv.shape[1:]), 0.1,
                       device=device),
        )


def _gather_to_gpu(self, x, topk_ids, topk_weights, uniq, inverse):
    """Uncached per-expert DMA into a throwaway buffer."""
    device = x.device
    n = int(uniq.numel())
    w13 = torch.empty((n, *self.w13_weight.shape[1:]),
                      dtype=self.w13_weight.dtype, device=device)
    w2 = torch.empty((n, *self.w2_weight.shape[1:]),
                     dtype=self.w2_weight.dtype, device=device)
    for i, e in enumerate(uniq.tolist()):
        w13[i].copy_(self.w13_weight[e], non_blocking=True)
        w2[i].copy_(self.w2_weight[e], non_blocking=True)
    s13 = self.w13_scale_inv[uniq].to(device, non_blocking=True)
    s2 = self.w2_scale_inv[uniq].to(device, non_blocking=True)
    STATS["misses"] += n
    STATS["experts_copied"] += n
    STATS["bytes_copied"] += n * 3_146_496
    local = inverse.reshape(topk_ids.shape).to(device)
    return fp8_moe_forward(x, local, topk_weights, w13, s13, w2, s2)


def cached_forward(self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> Tensor:
    """Resolve experts through a per-layer VRAM LRU, fetching only on a miss.

    The whole cache buffer is handed to fp8_moe_forward with topk_ids
    renumbered to slots, so a hit costs no copy at all -- not even a gather.

    Two hazards this has to avoid, both of which produce FLUENT WRONG AUDIO
    rather than an error, and one of which shipped in the first version:

    1. Evicting a slot already promised to THIS call. A prefill step routes to
       far more experts than the cache holds; the first version happily evicted
       slot 3 to make room, after slot 3 had already been recorded as expert A
       for this same call. Every id after that pointed at the wrong weights.
       Guarded by `promised` below, and by falling back when the call needs
       more experts than the cache can hold at once.
    2. Reading an uninitialised slot. Buffers are zeroed rather than left as
       torch.empty so a mistake degrades to silence, not to NaN.
    """
    uniq, inverse = torch.unique(topk_ids.reshape(-1).cpu(), return_inverse=True)
    n = int(uniq.numel())
    if n > self._cap:
        return _gather_to_gpu(self, x, topk_ids, topk_weights, uniq, inverse)

    device = x.device
    promised: set[int] = set()
    slots = []
    for e in uniq.tolist():
        slot = self._slot_of.get(e)
        if slot is None:
            STATS["misses"] += 1
            if self._free:
                slot = self._free.pop()
            else:
                for candidate in list(self._lru.keys()):
                    if candidate not in promised:
                        victim = self._lru.pop(candidate)
                        del self._slot_of[victim]
                        slot = candidate
                        break
                else:                       # every slot is in use this call
                    return _gather_to_gpu(self, x, topk_ids, topk_weights,
                                          uniq, inverse)
            self._cache_w13[slot].copy_(self.w13_weight[e], non_blocking=True)
            self._cache_w2[slot].copy_(self.w2_weight[e], non_blocking=True)
            self._cache_s13[slot].copy_(self.w13_scale_inv[e], non_blocking=True)
            self._cache_s2[slot].copy_(self.w2_scale_inv[e], non_blocking=True)
            self._slot_of[e] = slot
            STATS["experts_copied"] += 1
            STATS["bytes_copied"] += 3_146_496
        else:
            STATS["hits"] += 1
        self._lru[slot] = e
        self._lru.move_to_end(slot)
        promised.add(slot)
        slots.append(slot)

    slot_map = torch.tensor(slots, dtype=torch.long, device=device)
    local_ids = slot_map[inverse.to(device)].reshape(topk_ids.shape)
    return fp8_moe_forward(x, local_ids, topk_weights,
                           self._cache_w13, self._cache_s13,
                           self._cache_w2, self._cache_s2)


def slice_lm_head(model, device: torch.device) -> float:
    """Keep only the output rows decode can actually sample. Returns GiB freed.

    `compute_logits` runs the full [252288, 2048] lm_head every frame, but
    `decode_step` immediately does `logits[:, candidates]` where candidates is
    eos plus the semantic band -- 4,097 of 252,288 ids (tower.py:350-357).
    99.84 % of that GEMM is computed, read from VRAM, and discarded.

    Slicing the weight to those rows is **bit-identical**: the retained rows
    are unchanged and the discarded ones were never read. The scatter back
    into a full-width -inf buffer keeps `decode_step` untouched, and costs a
    0.5 MB write against the ~1 GiB of weight traffic it saves.

    Prefill never calls compute_logits at all (generate.py:232-247), so this
    affects decode only.
    """
    import torch.nn as nn
    cfg = model.config
    cand = torch.cat((
        torch.tensor([cfg.eos_token_id]),
        torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1),
    )).sort().values.to(device)

    backbone = model.backbone
    full = backbone.lm_head.weight
    vocab, hidden = full.shape
    before = full.numel() * full.element_size()
    sliced = full[cand].clone()

    head = nn.Linear(hidden, sliced.shape[0], bias=False,
                     device=device, dtype=sliced.dtype)
    with torch.no_grad():
        head.weight.copy_(sliced)
    backbone.lm_head = head
    del full, sliced
    torch.cuda.empty_cache()

    # ⚠️ Pre-allocated, not allocated per call. The first version created a
    # fresh torch.full(-inf) buffer inside compute_logits, which sits inside
    # the CUDA-graph captured region; with graphs on, every chunk died with
    # "non-finite logits" while the identical run without the slice was fine.
    # fill_ and index-assign are capturable; allocation is not safely so.
    #
    # compute_logits is only ever called with one row (generate.py:252 does
    # hidden = hidden[-1:], and the graph path uses a (1, hidden) buffer), so
    # one buffer suffices; anything wider falls back to allocating and is not
    # capture-safe, which is fine because it cannot occur in the decode path.
    # ⚠️ The padding is the most-negative FINITE value, not -inf, and that is
    # the whole reason --slice-lm-head now works under --cuda-graph 1.
    #
    # THE BUG, for anyone tempted to put -inf back. `Drama3DecodeGraphs.run`
    # (generate.py:376-377) does
    #
    #     logits = backbone.compute_logits(self.hidden)
    #     self.finite.copy_(torch.isfinite(logits).all())
    #
    # over the FULL 252,288-wide tensor. Fill 248,191 of those with -inf and
    # that check is False on the very first step, every step, forever --
    # which is exactly the "non-finite logits; refusing to sample invalid
    # audio" that killed every chunk. The slice was never producing bad
    # numbers; it was tripping a guard with its own sentinel.
    #
    # Eager decode did not care because the guard that runs there lives in
    # `sample_top_k_top_p` and only ever sees `logits[:, candidates]`
    # (tower.py:349-353) -- the 4,097 real values, all finite. That
    # difference is why this looked like a CUDA-graph bug for a day.
    #
    # Nothing reads the padded positions except that check: the only other
    # consumer indexes by `candidates`. finfo.min softmaxes to exactly 0 the
    # same way -inf does, so the sampled distribution is unchanged, while a
    # genuine Inf/NaN out of the GEMM still trips the guard.
    #
    # An earlier fix attempt -- pre-allocating this buffer instead of
    # allocating one per call -- was addressing a real and separate hazard
    # (allocation inside a captured region) and is kept. It was not the
    # cause, which is why it did not help on its own.
    pad = torch.finfo(head.weight.dtype).min

    wide_buf = torch.full((1, vocab), pad, device=device,
                          dtype=head.weight.dtype)

    def compute_logits(h: Tensor) -> Tensor:
        out = head(h.to(device))
        if out.shape[0] != 1:
            wide = torch.full((out.shape[0], vocab), pad,
                              device=out.device, dtype=out.dtype)
            wide[:, cand] = out
            return wide
        wide_buf.fill_(pad)
        wide_buf[:, cand] = out
        return wide_buf

    backbone.compute_logits = compute_logits
    freed = (before - head.weight.numel() * head.weight.element_size()) / 2**30
    return freed


def offload_embed(model, device: torch.device) -> float:
    """Move the input embedding table to host RAM, keep decode's slice in VRAM.

    The mirror of slice_lm_head on the input side, and worth about the same
    (0.96 GiB), but it cannot be a plain slice. Decode only ever embeds a
    token it just sampled, which `decode_step` draws from `main_candidates`
    -- eos plus the semantic band, 4,097 of 252,288 ids. **Prefill embeds
    real text ids and needs the whole table** (`generate.py:235` against
    `281` and `390`). Slice it and prefill reads the wrong rows, silently.

    So: full [252288, 2048] table on the host, the 4,097 rows decode can
    reach resident in VRAM (16.8 MiB), and prefill gathers what it needs.
    A prompt touches ~1,100 distinct text tokens, so that gather is ~4.5 MiB
    against the ~23 GiB of expert weights the same batch already moves.

    Returns GiB of VRAM freed.
    """
    cfg = model.config
    backbone = model.backbone
    table = backbone.model.embed_tokens.weight
    vocab, hidden = table.shape
    before = table.numel() * table.element_size()

    cand = torch.cat((
        torch.tensor([cfg.eos_token_id]),
        torch.arange(cfg.semantic_start_token_id, cfg.semantic_end_token_id + 1),
    )).sort().values

    host = table.detach().to("cpu")
    # id -> row in the resident slice, or -1 for the 248,191 ids only prefill
    # can see.
    row_of = torch.full((vocab,), -1, dtype=torch.long, device=device)
    row_of[cand.to(device)] = torch.arange(len(cand), device=device)

    # ⚠️ The host table is held in this closure, NOT left as the module's
    # parameter, and that is load-bearing. `Drama3DecodeGraphs.for_model`
    # (generate.py:350-353) does
    #
    #     devices = {p.device for p in model.backbone.parameters()}
    #     if len(devices) != 1 or ...: return None
    #
    # so one CPU parameter anywhere in the backbone silently turns CUDA
    # graphs OFF -- and graphs are worth RTF 1.71 -> 0.97, which is twenty
    # times this change. The experts get away with living on the host
    # because they are registered as *buffers*; an embedding weight is a
    # Parameter. Replacing the module with a 4,097-row one keeps every
    # backbone parameter on the GPU.
    import torch.nn as nn
    small = nn.Embedding(len(cand), hidden, device=device, dtype=host.dtype)
    with torch.no_grad():
        small.weight.copy_(host[cand].to(device))
    backbone.model.embed_tokens = small
    resident = small.weight
    del table
    torch.cuda.empty_cache()

    stray = {str(p.device) for p in backbone.parameters() if p.device.type != "cuda"}
    if stray:
        raise RuntimeError(
            f"offload_embed left backbone parameters on {sorted(stray)}; "
            "CUDA graphs would be silently disabled")

    def embed(self, input_ids: Tensor) -> Tensor:
        ids = input_ids.to(device)
        rows = row_of[ids]
        # ⚠️ The branch is decided by `is_current_stream_capturing`, NOT by
        # row count. The tempting test is "one row means decode", and it is
        # wrong: a prefill whose length is 1 mod --prefill-chunk ends with a
        # single-row batch carrying an arbitrary text id, which would then
        # read row -1 of the resident slice. Capture state is exact.
        #
        # Inside capture the ids are in-band by construction -- the only
        # thing decode ever embeds is `sampled[:, 0]`, and `decode_step`
        # draws it from `main_candidates` (tower.py:349-357), including
        # during the warm-up passes, where the zeroed buffers still go
        # through the sampler. Replay does not call this at all.
        if torch.cuda.is_current_stream_capturing():
            return torch.nn.functional.embedding(rows, resident)
        if bool((rows >= 0).all()):
            # Eager decode, and any prefill that happens to be all in-band.
            # The sync here is free next to the 40 the MoE gather already
            # does per step, and prefill is not captured.
            return torch.nn.functional.embedding(rows, resident)
        return torch.nn.functional.embedding(
            input_ids.to("cpu"), host).to(device)

    import types
    backbone.embed = types.MethodType(embed, backbone)
    freed = (before - resident.numel() * resident.element_size()
             - row_of.numel() * row_of.element_size()) / 2**30
    logger.info(
        f"embed_tokens: {vocab:,} rows to host RAM, {len(cand):,} resident "
        f"({resident.numel()*resident.element_size()/2**20:.1f} MiB) "
        f"+ {row_of.numel()*row_of.element_size()/2**20:.1f} MiB index; "
        f"freed {freed:.2f} GiB"
    )
    return freed


EXPERT_BYTES = 3_146_496        # measured, one expert incl. its fp32 scales


# Measured: VRAM peak during generation ran ~1.24 GiB above the post-load
# allocation (20.09 -> 21.33 GiB). 1.5 gives a little room over that.
ACTIVATION_HEADROOM_GIB = 1.5


def size_cache_to_free_vram(model, device: torch.device,
                            leave_free_gib: float = 2.0,
                            cap_per_layer: int = 256) -> int:
    """Allocate the expert cache from VRAM that is actually free right now.

    A fixed --cache-experts is wrong on a desktop: the box is not headless and
    other work takes varying amounts of VRAM, so a number that fits at launch
    can starve something else later. This measures free memory *after* the
    model and codec are resident, then holds back

        leave_free_gib  (for OTHER programs)
      + ACTIVATION_HEADROOM_GIB  (for this process's own peak during decode)

    and gives the rest to the cache. `leave_free_gib` therefore means what a
    user would expect: VRAM still available to everything else while a
    generation is running, not merely while it is idle.

    Returns the per-layer capacity chosen (0 means no cache).
    """
    layers = [l.mlp.experts for l in model.backbone.model.layers
              if getattr(l.mlp, "experts", None) is not None
              and l.mlp.experts.w13_weight.device.type == "cpu"]
    if not layers:
        return 0
    # Release blocks the caching allocator holds but is not using, or we
    # under-count free memory and hand back more headroom than was asked for
    # (measured: 4.3 GiB free at peak against a 2 GiB request, about 17
    # experts/layer of speed left unclaimed).
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    held = leave_free_gib + ACTIVATION_HEADROOM_GIB
    budget = free - int(held * 2**30)
    per_layer_cost = len(layers) * EXPERT_BYTES
    capacity = int(max(0, budget) // per_layer_cost)
    capacity = min(capacity, cap_per_layer, layers[0].num_experts)
    if capacity <= 0:
        logger.warning(
            f"no VRAM for an expert cache: {free/2**30:.2f} GiB free, "
            f"{leave_free_gib:.1f} GiB reserved for other programs + "
            f"{ACTIVATION_HEADROOM_GIB:.1f} GiB for activations")
        return 0
    for experts in layers:
        _init_cache(experts, device, capacity)
    torch.cuda.synchronize(device)
    after, _ = torch.cuda.mem_get_info(device)
    logger.info(
        f"expert cache: {capacity}/layer over {len(layers)} layers = "
        f"{capacity*per_layer_cost/2**30:.2f} GiB; {after/2**30:.2f} GiB free now, "
        f"~{after/2**30 - ACTIVATION_HEADROOM_GIB:.2f} GiB free at peak")
    return capacity


def pin_experts(model, device: torch.device) -> tuple[float, float]:
    """Page-lock the CPU-resident expert buffers. Returns (GiB, seconds).

    Measured on this box, per-expert DMA out of a host region, clean runs at
    four region sizes (1/4/16/30 GiB), all consistent:

        unpinned  16.6-16.8 GiB/s
        pinned    24.6 GiB/s          -> 1.47x

    ⚠️ This supersedes the "+10 %" figure in notes/02, which came from a
    768 MiB buffer and is not reproduced here. The discrepancy is unexplained;
    the newer measurement varies one thing and agrees with itself across four
    sizes, so it is the one to trust.

    Pinning is also the precondition for CUDA-graph capture and for genuinely
    async copies -- a non_blocking copy from pageable memory stages through a
    buffer and does not overlap.

    Cost: ~45 s at load for 30 GiB, and that much RAM locked. Done per layer
    so the transient is one layer (~768 MiB), not a second full copy.
    """
    started = time.perf_counter()
    total = 0
    for layer in model.backbone.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None or experts.w13_weight.device.type != "cpu":
            continue
        for name in ("w13_weight", "w2_weight", "w13_scale_inv", "w2_scale_inv"):
            buf = getattr(experts, name)
            if buf.is_pinned():
                continue
            pinned = buf.pin_memory()
            setattr(experts, name, pinned)
            total += pinned.numel() * pinned.element_size()
            del buf
    return total / 2**30, time.perf_counter() - started


def install_offload() -> None:
    MoEExperts._stock_forward = MoEExperts.forward
    MoEExperts.forward = offloaded_forward


# ---------------------------------------------------------------- main

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Routing is the only thing we are measuring here, so the words matter "
    "less than the length of the sequence they produce."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--text", default=TEXT)
    ap.add_argument("--out", default="notes/route-trace.json")
    ap.add_argument("--cache-experts", type=int, default=0,
                    help="experts held in VRAM per layer (0 = no cache); 96 ~ 11.3 GiB")
    ap.add_argument("--copy-mode", choices=("gather", "direct", "none", "graphstub"), default="direct",
                    help="gather = materialise on CPU first (slow); direct = per-expert DMA")
    ap.add_argument(
        "--no-offload",
        action="store_true",
        help="load straight onto the GPU; expected to OOM, which is itself a result",
    )
    args = ap.parse_args()

    global COPY_MODE, CACHE_EXPERTS
    COPY_MODE = args.copy_mode
    CACHE_EXPERTS = args.cache_experts
    device = torch.device(args.device)
    # Eager decode: CUDA graphs cannot capture the dynamic H2D copies, and
    # for_model() would refuse anyway once anything sits off the GPU.
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"

    if not args.no_offload:
        install_offload()
        import fish_speech.models.text2semantic.fish_omni.backbone as bb
        bb.FishOmniBackbone.materialize = lambda self, devices: (
            materialize_offloaded(self, device)
        )

    started = time.perf_counter()
    model = FishOmniModel.from_pretrained(
        args.checkpoint,
        devices=[str(device)],
        tower_device=str(device),
        dtype=torch.bfloat16,
    )
    load_s = time.perf_counter() - started
    print(f"loaded in {load_s:.1f}s", flush=True)
    if device.type == "cuda":
        print(
            f"VRAM after load: {torch.cuda.memory_allocated(device)/2**30:.2f} GiB "
            f"allocated, {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB peak",
            flush=True,
        )

    prompt = build_drama3_prompt(args.text, model.tokenizer, model.config)
    print(f"prompt rows: {len(prompt)}", flush=True)

    t0 = time.perf_counter()
    codes, reason = generate_drama3(
        model, prompt, max_new_tokens=args.max_new_tokens
    )
    elapsed = time.perf_counter() - t0

    frames = len(codes)
    # Decode rate is the number that matters: prefill is paid once, decode is
    # paid per frame and is what an expert cache would speed up.
    decode_s = (time.perf_counter() - TRACE.first_decode) if TRACE.first_decode else elapsed
    decode_tps = frames / decode_s if decode_s else 0
    print(f"\n{frames} frames, {elapsed:.1f}s total / {decode_s:.1f}s decode ({reason})")
    print(f"  blended {frames/elapsed:.2f} tok/s | DECODE {decode_tps:.2f} tok/s")
    total = STATS["hits"] + STATS["misses"]
    if total:
        print(f"cache: {STATS['hits']:,} hits / {total:,} = "
              f"{100*STATS['hits']/total:.1f}% hit rate")
    print(f"experts copied: {STATS['experts_copied']:,}")
    print(f"bytes copied:   {STATS['bytes_copied']/2**30:.2f} GiB")
    if elapsed > 0:
        print(f"effective H2D:  {STATS['bytes_copied']/2**30/elapsed:.2f} GiB/s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    TRACE.save(
        out,
        {
            "frames": frames,
            "seconds": elapsed,
            "tokens_per_second": frames / elapsed if elapsed else None,
            "decode_seconds": decode_s,
            "decode_tokens_per_second": decode_tps,
            "prompt_rows": len(prompt),
            "finish_reason": reason,
            "experts_copied": STATS["experts_copied"],
            "bytes_copied": STATS["bytes_copied"],
            "load_seconds": load_s,
            "text": args.text,
        },
    )
    print(f"trace -> {out} ({len(TRACE.records):,} records)")


if __name__ == "__main__":
    main()
