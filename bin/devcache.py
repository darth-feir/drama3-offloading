"""Device-resident expert cache: no host sync, CUDA-graph capturable.

Why this exists. `offload.cached_forward` keeps its LRU in a Python
OrderedDict and starts every layer with

    torch.unique(topk_ids.reshape(-1).cpu(), ...)

That `.cpu()` is a host sync on every one of 40 layers per decode step. It is
illegal during CUDA-graph capture, and it costs time even in eager mode --
published work on a comparable 4090 measured that one sync per layer throws
away 60 % of the graph benefit.

Stage 0 (bin/bench_hostread.py) established the piece this depends on: a
Triton kernel reading pinned host memory runs at 24.87 GiB/s against the copy
engine's 24.60, is byte-correct, and replays DYNAMIC under capture -- unlike a
memcpy node, which replays FROZEN and would serve capture-time experts
forever.

The design mirrors cached_forward's semantics exactly, because that function
is the reference this is checked against:

  slot_of[expert]  -> slot index, or -1 if absent         int32[256]
  slot_expert[slot]-> which expert occupies it, or -1     int32[cap]
  slot_stamp[slot] -> step number when last used          int64[cap]

The `promised` set that cached_forward needed -- the guard whose absence
produced fluent wrong audio -- falls out for free here: a slot touched during
this step carries the current stamp, and eviction picks the *minimum* stamp,
so a slot promised this step can never be chosen. No separate bookkeeping.

⚠️ Per-layer host tensors are fine for capture. A captured kernel bakes its
BASE pointer, not the offset, and the base of a given layer's pinned expert
tensor never moves. The "one global pool" idea in notes/06 is an optimisation
(fewer distinct bases, better for a fused kernel), not a correctness
requirement -- which removes the need to rewrite pin_experts for Stage 1.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _resolve_slots(
    ids_ptr,            # int32 [k]   expert ids wanted this call
    slot_of_ptr,        # int32 [E]   expert -> slot, -1 absent
    slot_expert_ptr,    # int32 [C]   slot -> expert, -1 empty
    slot_stamp_ptr,     # int64 [C]   slot -> step last used
    out_slot_ptr,       # int32 [k]   resolved slot per id
    fetch_slot_ptr,     # int32 [k]   slot to fill, -1 if hit
    fetch_expert_ptr,   # int32 [k]   expert to fetch, -1 if hit
    nmiss_ptr,          # int32 [1]   how many fetches were queued
    step_ptr,           # int64 [1]   step counter, incremented HERE
    K: tl.constexpr, C: tl.constexpr,
):
    """One program, sequential over K ids. Deliberately single-threaded.

    K is 8 for a decode step, so there is nothing to parallelise, and the
    allocation decisions are inherently sequential: two ids missing at once
    must not be handed the same slot. Doing it in one program is both correct
    and cheaper than any synchronisation scheme over 8 elements.
    """
    # ⚠️ The step counter is read and bumped on device, not passed as a Python
    # int. A scalar argument would be baked at capture time, so every replay
    # would stamp the same value and the LRU ordering would stop advancing --
    # slots would look equally old forever.
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)

    nmiss = 0
    for i in range(K):
        e = tl.load(ids_ptr + i).to(tl.int32)
        s = tl.load(slot_of_ptr + e).to(tl.int32)
        if s < 0:
            # choose the slot with the oldest stamp. A slot already used this
            # step carries `step` itself, which is the maximum, so it can
            # never win -- this is the `promised` guard, for free.
            best = 0
            best_stamp = tl.load(slot_stamp_ptr + 0)
            for c in range(1, C):
                st = tl.load(slot_stamp_ptr + c)
                if st < best_stamp:
                    best_stamp = st
                    best = c
            victim = tl.load(slot_expert_ptr + best).to(tl.int32)
            if victim >= 0:
                tl.store(slot_of_ptr + victim, -1)
            tl.store(slot_of_ptr + e, best)
            tl.store(slot_expert_ptr + best, e)
            tl.store(fetch_slot_ptr + nmiss, best)
            tl.store(fetch_expert_ptr + nmiss, e)
            nmiss += 1
            s = best
        tl.store(slot_stamp_ptr + s, step)
        tl.store(out_slot_ptr + i, s)
    tl.store(nmiss_ptr, nmiss)


@triton.jit
def _fetch_experts(
    host_ptr,           # pinned host tensor, flattened
    cache_ptr,          # device cache buffer, flattened
    fetch_slot_ptr,     # int32 [k]
    fetch_expert_ptr,   # int32 [k]
    nmiss_ptr,          # int32 [1]
    ELEMS: tl.constexpr,     # elements per expert
    BLOCK: tl.constexpr,
):
    """Copy queued experts host -> device cache slot.

    Grid is sized for the worst case (every id a miss); programs beyond the
    actual miss count retire without writing. That keeps the launch shape
    static, which is what capture requires -- the work done varies with data,
    the shape does not. bench_hostread verified surplus programs retire
    correctly ("njobs=honoured").
    """
    pid = tl.program_id(0)
    job = pid // tl.cdiv(ELEMS, BLOCK)
    nmiss = tl.load(nmiss_ptr)
    if job >= nmiss:
        return
    blk = pid % tl.cdiv(ELEMS, BLOCK)
    slot = tl.load(fetch_slot_ptr + job).to(tl.int64)
    expert = tl.load(fetch_expert_ptr + job).to(tl.int64)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ELEMS
    val = tl.load(host_ptr + expert * ELEMS + offs, mask=mask)
    tl.store(cache_ptr + slot * ELEMS + offs, val, mask=mask)


# ---------------------------------------------------------------- wiring

EXPERT_BYTES = 3_146_496


def init_dev_cache(experts, device: torch.device, capacity: int) -> None:
    """Allocate the device cache and its slot tables for one layer.

    Everything is byte-typed (uint8 views). The fetch is a pure byte copy, so
    it does not need to know that the weights are float8_e4m3fn -- and Triton
    has no float8 load anyway.
    """
    n_experts = experts.num_experts
    experts._dc_cap = capacity
    experts._dc_slot_of = torch.full((n_experts,), -1, dtype=torch.int32, device=device)
    experts._dc_slot_expert = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    # distinct negative stamps so the first `capacity` misses each take a
    # different empty slot instead of all colliding on slot 0
    experts._dc_stamp = -torch.arange(capacity, 0, -1, dtype=torch.int64, device=device)
    experts._dc_step = torch.zeros(1, dtype=torch.int64, device=device)

    def buf(src):
        return torch.zeros((capacity, *src.shape[1:]), dtype=src.dtype, device=device)

    experts._dc_w13 = buf(experts.w13_weight)
    experts._dc_w2 = buf(experts.w2_weight)
    experts._dc_s13 = torch.ones((capacity, *experts.w13_scale_inv.shape[1:]),
                                 dtype=torch.float32, device=device)
    experts._dc_s2 = torch.ones((capacity, *experts.w2_scale_inv.shape[1:]),
                                dtype=torch.float32, device=device)
    # scratch, allocated once so nothing allocates inside a captured region
    experts._dc_out = torch.zeros(8, dtype=torch.int32, device=device)
    experts._dc_fslot = torch.zeros(8, dtype=torch.int32, device=device)
    experts._dc_fexp = torch.zeros(8, dtype=torch.int32, device=device)
    experts._dc_nmiss = torch.zeros(1, dtype=torch.int32, device=device)


def _copy_pairs(experts):
    """(host, cache) byte views to move on a miss."""
    return (
        (experts.w13_weight.view(torch.uint8).reshape(-1),
         experts._dc_w13.view(torch.uint8).reshape(-1)),
        (experts.w2_weight.view(torch.uint8).reshape(-1),
         experts._dc_w2.view(torch.uint8).reshape(-1)),
        (experts.w13_scale_inv.view(torch.uint8).reshape(-1),
         experts._dc_s13.view(torch.uint8).reshape(-1)),
        (experts.w2_scale_inv.view(torch.uint8).reshape(-1),
         experts._dc_s2.view(torch.uint8).reshape(-1)),
    )


BLOCK = 4096

# "triton" or "cpp". Identical measured bandwidth (24.87 GiB/s @8 experts
# both); the C++ path exists because its uint4 vectorisation is explicit in
# csrc/fetch_experts.cu and can be inspected when something is unexpectedly
# slow, whereas Triton's depends on a JIT-attached alignment hint that can
# silently evaporate.
FETCH_BACKEND = "triton"
CPP_BLOCKS_PER_JOB = 64
_dev_ptr_cache: dict[int, int] = {}


def _cpp_fetch(experts, k: int) -> bool:
    """C++ uint4 fetch. Returns False if the extension is unavailable."""
    import cppfetch
    ext = cppfetch.load()
    if ext is None:
        return False
    for host, cache in _copy_pairs(experts):
        elems = host.numel() // experts.num_experts
        if elems % 16:
            # uint4 needs 16 B granularity. The real model's scale blocks are
            # 512 B and 256 B per expert so this never fires there, but a
            # kernel that silently assumes its inputs are convenient is how
            # you get a misaligned fault on someone else's config. Small
            # buffers go through Triton; they are far too small to matter.
            _triton_fetch_one(experts, host, cache, elems, k)
            continue
        key = host.data_ptr()
        base = _dev_ptr_cache.get(key)
        if base is None:
            base = ext.device_ptr_of(host)
            _dev_ptr_cache[key] = base
        ext.fetch_experts(base, cache, experts._dc_fexp, experts._dc_fslot,
                          experts._dc_nmiss, elems, k, CPP_BLOCKS_PER_JOB, 256)
    return True


def _triton_fetch_one(experts, host, cache, elems: int, k: int) -> None:
    grid = (k * triton.cdiv(elems, BLOCK),)
    _fetch_experts[grid](host, cache, experts._dc_fslot, experts._dc_fexp,
                         experts._dc_nmiss, ELEMS=elems, BLOCK=BLOCK,
                         num_warps=8)


def dev_resolve_and_fetch(experts, topk_ids: torch.Tensor) -> torch.Tensor:
    """Resolve ids to slots, fetching misses. Returns slot indices, no sync."""
    k = topk_ids.numel()
    ids = topk_ids.reshape(-1).to(torch.int32)
    _resolve_slots[(1,)](
        ids, experts._dc_slot_of, experts._dc_slot_expert, experts._dc_stamp,
        experts._dc_out, experts._dc_fslot, experts._dc_fexp,
        experts._dc_nmiss, experts._dc_step,
        K=k, C=experts._dc_cap, num_warps=1,
    )
    if FETCH_BACKEND == "cpp" and _cpp_fetch(experts, k):
        return experts._dc_out[:k]
    for host, cache in _copy_pairs(experts):
        _triton_fetch_one(experts, host, cache,
                          host.numel() // experts.num_experts, k)
    return experts._dc_out[:k]


def dev_cached_forward(self, x, topk_ids, topk_weights):
    """MoEExperts.forward using the device-resident cache.

    Decode only. Prefill routes to 114-175 distinct experts per layer against
    a cache of ~70, so it would thrash and the resolve kernel's sequential
    loop over hundreds of ids would be slow. Prefill is never captured
    anyway, so it keeps the existing gather path where a host sync is
    harmless.
    """
    import offload
    from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_moe_forward

    if topk_ids.shape[0] != 1 or not getattr(self, "_dc_cap", 0):
        return offload.offloaded_forward(self, x, topk_ids, topk_weights)

    slots = dev_resolve_and_fetch(self, topk_ids)
    offload.STATS["experts_copied"] += 0      # counted on device; see _dc_nmiss
    return fp8_moe_forward(x, slots.reshape(topk_ids.shape).to(torch.int64),
                           topk_weights, self._dc_w13, self._dc_s13,
                           self._dc_w2, self._dc_s2)


def install_dev_cache(model, device, capacity: int) -> int:
    """Allocate the device cache on every offloaded layer and swap in the
    forward. Returns the per-layer capacity used.

    ⚠️ Requires the host expert tensors to be PAGE-LOCKED. Triton rejects a
    pageable pointer outright ("cannot be accessed from Triton") and the C++
    path's device_ptr_of TORCH_CHECKs is_pinned, so this fails loudly rather
    than silently reading nonsense -- but call pin_experts first.
    """
    from fish_speech.models.text2semantic.fish_omni.moe import MoEExperts
    n = 0
    for layer in model.backbone.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is None or experts.w13_weight.device.type != "cpu":
            continue
        if not experts.w13_weight.is_pinned():
            raise RuntimeError(
                "device cache needs pinned expert tensors; run pin_experts first")
        init_dev_cache(experts, device, capacity)
        n += 1
    MoEExperts.forward = dev_cached_forward
    return n
