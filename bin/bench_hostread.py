"""Stage 0: can a GPU *kernel* read pinned host memory, how fast, and is it
CUDA-graph safe?  (task #21)

WHY THIS EXISTS
---------------
`notes/06-cuda-graphs-plan.md` measured CUDA graphs at **2.35x** (RTF 1.08 ->
0.46) with a synthetic MoE.  To get that on the real path, the fetch of a
missed expert has to happen INSIDE the captured region, and it has to read an
expert index that only exists at replay time.

The whole design turns on one distinction:

  * A captured **memcpy node** freezes its source pointer at capture.  There is
    no arithmetic in it.  On replay it re-reads the same address forever, which
    produces fluent, confident, WRONG audio -- the worst failure mode this
    project has.
  * A captured **kernel** is code.  It can load an expert index out of device
    memory and compute `base + index * EXPERT_BYTES` at replay time.

So: what bandwidth does a kernel get reading page-locked host memory, versus
the 24.6 GiB/s the copy engine gets?  `notes/04-speedups.md` has flagged this
as "still unmeasured" twice.  This script measures it.

It also measures the thing nobody has checked: *whether each mechanism is
actually capturable, and whether replay follows the data or the frozen
pointer.*  A mechanism that captures cleanly and then silently reads the
capture-time expert is worse than one that refuses to capture.

WHAT EACH OUTCOME MEANS
-----------------------
  kernel host-read >= ~20 GiB/s
      The plan in notes/06 is sound.  Build the device-side slot table
      (task #20) and keep the fetch as a kernel node inside the graph.
  kernel host-read ~10-20 GiB/s
      Workable but the transfer term grows.  Amdahl on notes/06's split
      (RTF 1.69 = ~1.05 compute + ~0.64 transfer) says graphs land near 1.1
      at copy-engine speed; a 2x slower fetch pushes transfer to ~1.3 and the
      whole thing back over realtime.  At that point partial capture (graph
      the attention/tower/sampling, leave the 40 MoE layers eager) is the
      better trade.
  kernel host-read < ~8 GiB/s
      The design changes shape.  Zero-copy is not a transport; fall back to
      keeping memcpy nodes outside the graph, or to raising the hit rate so
      misses are rare enough to pay a graph break for.
  ANY mechanism that captures but replays the capture-time expert
      Blacklist it in the notes, loudly.  That is the silent-corruption case.

CORRECTNESS IS NOT OPTIONAL HERE
--------------------------------
Every expert in the host region is filled with a pattern that encodes its own
index: as int64, element k of expert e is `(e << 32) | k`.  Every fetched byte
is checked against that.  A mechanism that is fast and reads the wrong expert
is exactly the bug this project is trying to design out, so "fast" is only
reported alongside a full-byte comparison.

CONSTRAINTS OBSERVED
--------------------
  * Another process may own the GPU.  Caps: ~4 GiB host (one power-of-two
    pinned allocation), <1 GiB VRAM.  Refuses to start if VRAM is tight.
  * Runs in well under 3 minutes once the Triton and nvcc caches are warm.
    First run pays ~10 s of Triton JIT and ~40 s of nvcc, both cached after.

RUN IT
------
    cd /mnt/unencrypted/ai/drama3 && bin/run.sh bin/bench_hostread.py

run.sh wraps nix-shell -p gcc, which Triton needs to build its launcher.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------- constants

# The real number, from bin/offload.py:352 -- w13 (2*512*2048) + w2 (2048*512)
# + both scale_inv blocks.  3.001 MiB.  Keep this exact: the alignment argument
# below depends on it.
EXPERT = 3_146_496

# 3_146_496 = 2^8 * 3 * 17 * 241, so every expert boundary is 256-byte aligned
# inside a page-aligned region.  That matters: EMOGI (VLDB 2021) measured
# zero-copy PCIe reads at 12.23 GB/s when merged and 128 B aligned versus
# 9.40 GB/s when off by 32 bytes, on the same link where cudaMemcpy got 12.36.
# If anyone ever changes the expert stride, keep it a multiple of 128 or the
# numbers below stop applying.
# 16 B divisibility is load-bearing, not cosmetic -- see _fetch_kernel.
assert EXPERT % 256 == 0
ELEMS_I32 = EXPERT // 4          # 786_624 -- what the Triton kernel indexes in
ELEMS_I64 = EXPERT // 8          # 393_312 -- the verification/index_select view

GiB = 2**30

# Nix store paths for the optional nvcc path.  VERIFIED present 2026-10-17 by
# compiling expert_fetch.cu against them.  They are NOT gc roots -- if
# nix-collect-garbage has run, these vanish and the extension mechanism skips.
# Resolved the same way as bin/cppfetch.py -- see the comment there.
import cppfetch as _cf
_cf._resolve()
CUDA_MERGED = _cf.CUDA_MERGED
NINJA_BIN = _cf.NINJA_BIN
# nvcc's own nvcc.profile already points compiler-bindir here.  Do NOT
# substitute `nix-shell -p gcc`: that is gcc 15.3 and CUDA 12.9's
# crt/host_config.h hard-errors above gcc 14.
GCC14_BIN = _cf.GCC14_BIN

DEV = torch.device("cuda:0")


# ---------------------------------------------------------------- results


@dataclass
class Result:
    name: str
    note: str = ""
    gbs_small: float | None = None     # GiB/s at 8 experts  (one decode layer)
    gbs_large: float | None = None     # GiB/s at 64 experts (asymptotic)
    correct: str = "-"                 # ok / N mismatched / skipped
    capture: str = "-"                 # ok / the exception
    replay: str = "-"                  # DYNAMIC / FROZEN / wrong / n/a
    skipped: str = ""                  # why, if skipped


RESULTS: list[Result] = []
KEEPALIVE: list[object] = []           # anything that must outlive a raw ptr


def emit(r: Result) -> None:
    RESULTS.append(r)
    if r.skipped:
        print(f"  SKIPPED: {r.skipped}", flush=True)
    else:
        small = f"{r.gbs_small:6.2f}" if r.gbs_small else "     -"
        large = f"{r.gbs_large:6.2f}" if r.gbs_large else "     -"
        print(f"  {small} GiB/s @8   {large} GiB/s @64   "
              f"bytes={r.correct}  capture={r.capture}  replay={r.replay}",
              flush=True)


# ---------------------------------------------------------------- timing


def timed(fn, iters: int, warm: int = 3) -> float:
    """Seconds per call, measured with CUDA events on the current stream."""
    for i in range(warm):
        fn(i)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for i in range(iters):
        fn(i)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 1000.0 / iters


# ---------------------------------------------------------------- preflight


def cudart():
    """ctypes handle on the already-loaded libcudart.  torch exposes neither
    cudaHostGetDevicePointer nor cudaPointerGetAttributes (checked: torch._C.
    _cudart has only 9 entries and neither is among them), and we want three
    facts about this box that have only ever been asserted, never measured."""
    cand = os.path.join(os.path.dirname(torch.__file__), "..", "nvidia",
                        "cuda_runtime", "lib", "libcudart.so.12")
    for p in (os.path.normpath(cand), "libcudart.so.12"):
        try:
            return ctypes.CDLL(p)
        except OSError:
            continue
    return None


def probe_device_facts() -> None:
    """Print the three attributes the investigations could not verify.

    cudaDevAttrUnifiedAddressing (41) is the gate on 'a host pointer from
    cudaHostAlloc is dereferenceable in a kernel at the same address'.  If it
    is 0, every kernel mechanism below is invalid and the numbers are noise.

    CanUseHostPointerForRegisteredMem (91) only matters if someone sets
    PYTORCH_CUDA_ALLOC_CONF=pinned_use_cuda_host_register:True, which switches
    torch from cudaHostAlloc to malloc+cudaHostRegister.  In that case the
    device pointer may differ from the host pointer and passing data_ptr()
    straight to a kernel becomes wrong.  Don't set that knob.
    """
    rt = cudart()
    if rt is None:
        print("  (could not dlopen libcudart; skipping attribute probe)")
        return
    val = ctypes.c_int(0)
    for attr, nm in ((41, "unifiedAddressing"),
                     (91, "canUseHostPointerForRegisteredMem"),
                     (99, "hostRegisterSupported"),
                     (113, "hostRegisterReadOnlySupported")):
        rc = rt.cudaDeviceGetAttribute(ctypes.byref(val), attr, 0)
        shown = val.value if rc == 0 else "<cudaError %d>" % rc
        print(f"  attr {attr:>3} {nm:<36} = {shown}")

    # And the one that actually decides whether data_ptr() is usable as-is.
    t = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
    KEEPALIVE.append(t)
    dptr = ctypes.c_void_p(0)
    rt.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                            ctypes.c_void_p, ctypes.c_uint]
    rc = rt.cudaHostGetDevicePointer(ctypes.byref(dptr),
                                     ctypes.c_void_p(t.data_ptr()), 0)
    same = (rc == 0 and dptr.value == t.data_ptr())
    print(f"  cudaHostGetDevicePointer(pin_memory ptr) rc={rc} "
          f"-> {'SAME address as host (good)' if same else 'DIFFERENT / failed'}")
    print(f"  pinned tensor base alignment: data_ptr % 4096 = "
          f"{t.data_ptr() % 4096}  (0 means page aligned)")


def preflight(host_bytes: int) -> None:
    print("=" * 78)
    print("bench_hostread.py -- Stage 0: kernel reads of page-locked host RAM")
    print("=" * 78)
    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable.  LD_LIBRARY_PATH=/run/opengl-driver/lib "
                 "missing?  Source bin/env.sh.")
    free, total = torch.cuda.mem_get_info()
    print(f"  torch {torch.__version__}   device {torch.cuda.get_device_name(0)}"
          f"   VRAM {free/GiB:.2f} GiB free of {total/GiB:.2f}")
    if free < 2.0 * GiB:
        sys.exit(f"Only {free/GiB:.2f} GiB VRAM free; another process is "
                 f"probably mid-run.  Refusing to start (this script needs "
                 f"<1 GiB but will not race an inference job).")
    avail = int([l.split()[1] for l in open("/proc/meminfo")
                 if l.startswith("MemAvailable")][0]) * 1024
    print(f"  host RAM available: {avail/GiB:.1f} GiB; "
          f"will page-lock {host_bytes/GiB:.0f} GiB")
    if avail < host_bytes + 8 * GiB:
        sys.exit("Not enough free host RAM to pin safely.")
    # /proc/cmdline has no iommu= on this box, so the IOMMU is in whatever mode
    # the kernel defaults to.  A translating IOMMU taxes scattered device reads
    # of host pages more than it taxes the copy engine, so if the kernel
    # numbers come back oddly low this is the first thing to look at.
    print(f"  /proc/cmdline iommu: "
          f"{[w for w in open('/proc/cmdline').read().split() if 'iommu' in w] or 'not specified'}")
    probe_device_facts()


# ---------------------------------------------------------------- host region


def build_host_region(host_bytes: int):
    """One contiguous page-locked region, filled with a self-identifying pattern.

    Two things here are deliberate and easy to get wrong:

    1. `host_bytes` is an exact power of two.  ATen's CachingHostAllocator does
       `roundSize = PowerOf2Ceil(size)` (CachingHostAllocator.h:206) and never
       splits blocks, so asking for 6 GiB really cudaHostAllocs 8 GiB.  Ask for
       2^32 and you get 2^32.
       (Corollary for the production loader: bin/offload.py:577 pins per layer,
       4 tensors x 40 layers = 160 separate power-of-two-rounded allocations
       with 160 unrelated base addresses.  That is wasteful AND incompatible
       with the single-base-pointer kernel design.)

    2. The pattern.  Element k of expert e, read as int64, is (e << 32) | k.
       That makes every single byte attributable to a specific expert and a
       specific offset, so a fetch that lands the right expert at the wrong
       offset, or the wrong expert entirely, both show up.
    """
    n_exp = host_bytes // EXPERT
    print(f"\n[setup] page-locking {host_bytes/GiB:.0f} GiB "
          f"({n_exp} experts of {EXPERT/2**20:.3f} MiB)...", flush=True)
    t0 = time.perf_counter()
    host = torch.empty(host_bytes, dtype=torch.uint8, pin_memory=True)
    t_pin = time.perf_counter() - t0
    assert host.data_ptr() % 4096 == 0, "pinned base is not page aligned"

    h64 = host.view(torch.int64)[: n_exp * ELEMS_I64].view(n_exp, ELEMS_I64)
    karange = torch.arange(ELEMS_I64, dtype=torch.int64)
    t0 = time.perf_counter()
    for e in range(n_exp):
        torch.add(karange, e << 32, out=h64[e])
    t_fill = time.perf_counter() - t0
    print(f"[setup] pinned in {t_pin:.1f} s, pattern written in {t_fill:.1f} s",
          flush=True)
    return host, h64, n_exp


# ---------------------------------------------------------------- verification


class Checker:
    """Full-byte verification that slot j holds expert ids[j]."""

    def __init__(self, dst: torch.Tensor):
        # dst is the uint8 VRAM slot pool [SLOTS, EXPERT]; view as int64.
        self.d64 = dst.view(torch.int64).view(dst.shape[0], ELEMS_I64)
        self.karange = torch.arange(ELEMS_I64, dtype=torch.int64, device=DEV)

    def mismatches(self, ids: list[int], base_slot: int = 0) -> int:
        """Count int64 elements that are not what expert ids[j] should put in
        slot base_slot+j.  Done one slot at a time and accumulated on device:
        the broadcast form would materialise a 64 x 3 MiB expectation tensor,
        and the VRAM budget here is deliberately tight."""
        acc = torch.zeros((), dtype=torch.int64, device=DEV)
        for j, e in enumerate(ids):
            acc += (self.d64[base_slot + j] != self.karange + (e << 32)).sum()
        return int(acc.item())

    def untouched(self, lo: int, hi: int, sentinel: int = -1) -> bool:
        """True if slots [lo, hi) still hold the scrub value -- i.e. nothing
        wrote them.  Used to check that a captured kernel honours a job count
        that shrank between capture and replay."""
        if hi <= lo:
            return True
        return bool((self.d64[lo:hi] == sentinel).all().item())

    def which_expert(self, slot: int) -> int:
        """Reverse the pattern: whatever is in this slot, what expert is it?
        Used by the replay test to say WHICH wrong expert came back."""
        return int(self.d64[slot, 1].item() >> 32)


# ---------------------------------------------------------------- mechanisms
#
# Every mechanism exposes the same tiny interface so the capture test is
# identical across them:
#
#   set_ids(ids)  -- make the next run fetch these experts into slots 0..J-1.
#                    For kernel mechanisms this writes a DEVICE tensor, which
#                    is the whole point: the graph must read it at replay.
#                    For the memcpy baseline it can only set a host variable,
#                    which is the whole problem.
#   run()         -- issue the fetch on the current stream.  No sync, no alloc.
#   resize(jobs)  -- set the launch geometry.  For kernel mechanisms this is
#                    the grid, and the grid is FROZEN at capture, so in
#                    production it would be nailed to the worst case (8
#                    misses per layer, one per expert in topk).
#   force_njobs(n) -- kernel mechanisms only: set the DEVICE-resident job
#                    count without touching the grid.  This is how a captured
#                    graph copes with a layer that missed on 3 experts instead
#                    of 8, and it is tested separately because getting it
#                    wrong means writing garbage into slots the GEMM will read.


class Mechanism:
    name = "?"
    jobs = 8
    dynamic_njobs = False              # does it have a device-side job count?

    def set_ids(self, ids: list[int]) -> None: ...
    def run(self) -> None: ...
    def resize(self, jobs: int) -> None:
        self.jobs = jobs
    def force_njobs(self, n: int) -> None: ...

    def prime(self, host_ids: list[int], dev_i32: "torch.Tensor",
              dev_i64: "torch.Tensor") -> None:
        """Timing-loop variant of set_ids, given the same ids three ways.

        WHY IT EXISTS: set_ids() copies a freshly built CPU tensor to the
        device, and a copy_ from PAGEABLE host memory is synchronous -- torch
        issues the cudaMemcpyAsync and then blocks.  Inside the timed loop
        that both serialises every iteration and charges its latency to the
        mechanism under test.  Kernel mechanisms override this to do an async
        device-to-device copy out of a pre-staged index tensor instead; the
        memcpy baselines fall through to set_ids because their 'index' is a
        python list and costs nothing."""
        self.set_ids(host_ids)


class MemcpyBaseline(Mechanism):
    """CONTROL: per-expert tensor.copy_ from the pinned region, i.e. exactly
    what bin/offload.py:404 does today.  Known to reach 24.6 GiB/s
    (notes/04).  Runs on the copy engine, so it does not contend for SMs --
    that is its real advantage over every kernel below, and the reason a
    kernel that merely *matches* it is still a step backwards for overlap.

    Its capture behaviour is the thing to watch.  Capture will probably
    SUCCEED, because a cudaMemcpyAsync from page-locked memory is a legal
    graph node.  But the source pointer is computed on the host, in this
    python loop, at capture time, and baked.  Replay will copy the
    capture-time experts forever.  That is the failure mode notes/06 names as
    the most dangerous in the design, and this is the test that demonstrates
    it rather than asserting it."""

    name = "copy_ (copy engine, memcpy node)"

    def __init__(self, host2d, dst2d):
        self.src, self.dst, self.ids = host2d, dst2d, [0] * 8

    def set_ids(self, ids):
        self.ids = list(ids)

    def run(self):
        for i, e in enumerate(self.ids):
            self.dst[i].copy_(self.src[e], non_blocking=True)


class PageableBaseline(Mechanism):
    """CONTROL 2: the same copy_ out of ordinary pageable memory.  Measured
    before at 16.7 GiB/s.  Included only so the pinned/unpinned ratio in this
    run can be compared against the 1.47x in notes/04 -- if it does not
    reproduce, something about this box changed and none of the other numbers
    should be trusted either.  Expected to FAIL capture."""

    name = "copy_ from PAGEABLE host"

    def __init__(self, src2d, dst2d):
        self.src, self.dst, self.ids = src2d, dst2d, [0] * 8

    def set_ids(self, ids):
        self.ids = [i % self.src.shape[0] for i in ids]

    def run(self):
        for i, e in enumerate(self.ids):
            self.dst[i].copy_(self.src[e], non_blocking=True)


# ---- Triton -----------------------------------------------------------------

TRITON_OK = True
TRITON_ERR = ""
try:
    import triton
    import triton.language as tl
except Exception as _e:                                    # pragma: no cover
    TRITON_OK = False
    TRITON_ERR = repr(_e)

if TRITON_OK:

    @triton.jit
    def _fetch_kernel(src_ptr,          # *i32  base of the expert region
                      dst_ptr,          # *i32  base of the VRAM slot pool
                      idx_ptr,          # *i32  [MAX_JOBS], DEVICE memory
                      slot_ptr,         # *i32  [MAX_JOBS], DEVICE memory
                      njobs_ptr,        # *i32  [1],        DEVICE memory
                      ELEMS: tl.constexpr,
                      BLOCK: tl.constexpr):
        """One job per blockIdx.y: copy expert idx[j] into slot slot[j].

        Capture discipline, which is the entire point of writing it this way:

          * Launch geometry is CONSTANT.  gridDim.y is MAX_JOBS, fixed at
            capture.  The real job count is read from device memory and
            surplus programs retire.  Anything that varies must be data, never
            a launch parameter, because launch parameters freeze at capture
            exactly like a memcpy node's source pointer does.
          * The expert index is loaded from device memory INSIDE the kernel.
            That is the one thing a memcpy node cannot do.
          * src_ptr is a host address.  It is baked at capture, which is fine:
            it never moves.  Only the offset is live.

        VECTORISATION -- VERIFIED, ahead-of-time, no GPU needed.  Compiling
        this kernel for sm_89 with tt.divisibility=16 on the pointer args
        emits `ld.global.v4.b32` / `st.global.v4.b32`: 16 B per thread, so a
        warp issues 512 contiguous bytes = four fully aligned 128 B PCIe
        reads.  At BLOCK=4096 / num_warps=8 that is 4 such loads per thread.

        Two things that result DEPENDS on, both easy to break:

          * Every pointer argument must be 16 B aligned, because that is what
            makes Triton attach tt.divisibility=16
            (backends/compiler.py:88 tests `data_ptr() % 16 == 0`).  WITHOUT
            it the same kernel compiles to scalar `ld.global.b32` -- measured:
            19 scalar loads instead of 4 vector ones.  That is a 4x drop in
            request size, and EMOGI's numbers say small requests cost ~2.5x on
            a PCIe zero-copy path.  torch's allocators always give 16 B
            alignment, so this holds, but a hand-rolled buffer need not.
          * `e * ELEMS` must stay 16 B aligned, because Triton proves
            alignment from the BASE pointer and then assumes the whole access
            is aligned.  EXPERT = 3_146_496 is divisible by 16 (indeed by
            256), so every expert boundary is aligned and a 128-bit load can
            never straddle.  Change the expert stride to something not
            divisible by 16 and this becomes a misaligned-access FAULT, not
            merely slow.  Hence the assert at the top of this file.
        """
        j = tl.program_id(1)
        n = tl.load(njobs_ptr)
        if j < n:
            e = tl.load(idx_ptr + j).to(tl.int64)
            s = tl.load(slot_ptr + j).to(tl.int64)
            src = src_ptr + e * ELEMS
            dst = dst_ptr + s * ELEMS
            start = tl.program_id(0) * BLOCK
            step = tl.num_programs(0) * BLOCK
            for off in range(start, ELEMS, step):
                idx = tl.max_contiguous(
                    tl.multiple_of(off + tl.arange(0, BLOCK), BLOCK), BLOCK)
                m = idx < ELEMS
                tl.store(dst + idx, tl.load(src + idx, mask=m, other=0), mask=m)


class RawPtr:
    """Duck type Triton accepts as a pointer argument.

    triton/runtime/jit.py:810 documents it: "arguments are implicitly
    converted to pointers if they have a .data_ptr() method and a .dtype
    attribute".  specialize_impl (jit.py:339) only reads .dtype, and
    get_arg_specialization (backends/compiler.py:88) only tests
    data_ptr() % 16 == 0.  Neither looks at .device.

    Why not just pass the int?  Because specialize_impl tests isinstance(arg,
    int) FIRST (jit.py:325) and would turn the address into an i64 scalar.
    """

    def __init__(self, ptr: int, dtype=torch.int32):
        self._p, self.dtype = int(ptr), dtype

    def data_ptr(self) -> int:
        return self._p


class TritonFetch(Mechanism):
    """THE MECHANISM THE DESIGN NEEDS, in its cheapest form: stock Triton
    3.4.0, no C++ extension, no nvcc, no cudaHostRegister.

    The source argument is a page-locked CPU tensor (or a bare address).
    Triton's generated launcher resolves it itself -- driver.py:396 calls
    cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_DEVICE_POINTER, p) on every
    pointer argument.  For cudaHostAlloc'd memory under UVA that succeeds and
    returns the identical value, which is why no cudaHostAllocMapped and no
    cudaHostGetDevicePointer are needed on the torch side.

    If this row is fast, correct and DYNAMIC on replay, the notes/06 design is
    reachable today with zero new build tooling and the remaining work is
    entirely the device-side slot table (task #20)."""

    def __init__(self, src_obj, dst: torch.Tensor, label: str,
                 slots: int, block: int, blocks_per_job: int, warps: int):
        self.name = label
        self.src = src_obj
        self.dst_ptr = RawPtr(dst.data_ptr(), torch.int32)
        self.block, self.bpj, self.warps = block, blocks_per_job, warps
        self.idx = torch.zeros(slots, dtype=torch.int32, device=DEV)
        self.slot = torch.arange(slots, dtype=torch.int32, device=DEV)
        self.njobs = torch.full((1,), slots, dtype=torch.int32, device=DEV)
        self.jobs = self.grid_y = slots
        self.dynamic_njobs = True

    def set_ids(self, ids):
        self.idx[: len(ids)].copy_(
            torch.tensor(ids, dtype=torch.int32), non_blocking=False)
        self.njobs.fill_(len(ids))
        self.jobs = len(ids)

    def resize(self, jobs):
        # grid_y is the launch geometry, which freezes at capture; njobs is
        # device data, which does not.  Setting them together here matches
        # production, where the grid would be nailed to the worst case.
        self.jobs = self.grid_y = jobs
        self.njobs.fill_(jobs)

    def force_njobs(self, n):
        self.njobs.fill_(n)

    def prime(self, host_ids, dev_i32, dev_i64):
        # async D2D; njobs is already correct from resize()
        self.idx[: dev_i32.numel()].copy_(dev_i32, non_blocking=True)

    def run(self):
        _fetch_kernel[(self.bpj, self.grid_y)](
            self.src, self.dst_ptr, self.idx, self.slot, self.njobs,
            ELEMS=ELEMS_I32, BLOCK=self.block, num_warps=self.warps)


class IndexSelectFetch(Mechanism):
    """The no-new-code option: stock torch index_select, where the SOURCE is a
    plain host pointer dressed up as a CUDA tensor.

    torch._C._construct_storage_from_data_pointer builds a c10::Storage with a
    NON-OWNING DataPtr (torch/csrc/Module.cpp) -- no deleter, so it will not
    try to cudaFree host memory -- and
    _construct_CUDA_Tensor_From_Storage_And_Metadata wraps it.  Both are used
    by inductor's own cudagraph machinery
    (torch/_inductor/cudagraph_trees.py:1647), so they are undocumented but
    not exotic.  The result is a tensor torch believes is on cuda:0 whose
    bytes live in page-locked host RAM.

    index_select IS a kernel, so it computes its addresses at replay time from
    the device-resident index tensor.  If this works at a usable rate, Stage 0
    is answered with no Triton and no nvcc at all, and task #20 gets simpler.

    Two caveats worth stating before anyone builds on it:
      * Only pass it to kernels.  Routing this tensor through .copy_() would
        dispatch cudaMemcpyAsync with kind=DeviceToDevice and a host source,
        which happens to work under UVA but is outside the documented
        contract.
      * int64 view, not uint8, so the element type gives the kernel something
        to vectorise.  EXPERT is divisible by 8, so the view is exact.
    """

    name = "index_select (host ptr as fake CUDA tensor)"

    def __init__(self, host: torch.Tensor, n_exp: int, dst: torch.Tensor,
                 max_jobs: int):
        nbytes = host.numel()
        storage = torch._C._construct_storage_from_data_pointer(
            host.data_ptr(), DEV, nbytes)
        meta = {
            "nbytes": nbytes,
            "data_ptr": host.data_ptr(),
            "size": (n_exp, ELEMS_I64),
            "stride": (ELEMS_I64, 1),
            "dtype": torch.int64,
            "device": DEV,
            "storage_offset": 0,
        }
        self.src = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
            meta, storage)
        KEEPALIVE.extend((host, storage, self.src))
        self.out = dst.view(torch.int64).view(dst.shape[0], ELEMS_I64)
        self.idx = torch.zeros(max_jobs, dtype=torch.int64, device=DEV)
        self.jobs = max_jobs

    def set_ids(self, ids):
        self.idx[: len(ids)].copy_(torch.tensor(ids, dtype=torch.int64))
        self.jobs = len(ids)

    def prime(self, host_ids, dev_i32, dev_i64):
        self.idx[: dev_i64.numel()].copy_(dev_i64, non_blocking=True)

    def run(self):
        # out= so nothing is allocated inside the captured region.
        torch.index_select(self.src, 0, self.idx[: self.jobs],
                           out=self.out[: self.jobs])


# ---- C++ / nvcc extension ---------------------------------------------------

EXT_SRC = r"""
// Hand-written uint4 fetch kernel: the bandwidth ceiling a kernel can reach,
// against which the Triton row is judged.  16 B per thread means a warp issues
// 512 contiguous bytes, which the coalescer turns into four fully aligned
// 128 B PCIe reads -- the "merged and aligned" case EMOGI measured at copy
// engine parity.  Verified in SASS (LDG.E.128.CONSTANT / STG.E.128) during the
// cpp-extension investigation.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

int64_t device_ptr_of(at::Tensor t) {
  TORCH_CHECK(t.device().is_cpu() && t.is_pinned() && t.is_contiguous(),
              "expected a contiguous page-locked CPU tensor");
  void* dptr = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&dptr, t.data_ptr(), 0);
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  return reinterpret_cast<int64_t>(dptr);
}

__global__ void fetch_kernel(const uint4* __restrict__ src_base,
                             uint4* __restrict__ dst_base,
                             const int32_t* __restrict__ idx,
                             const int32_t* __restrict__ slot,
                             const int32_t* __restrict__ njobs,
                             int32_t vec_per_expert) {
  const int j = blockIdx.y;
  if (j >= *njobs) return;                 // read at REPLAY, not at capture
  const uint4* __restrict__ s =
      src_base + static_cast<size_t>(idx[j]) * vec_per_expert;
  uint4* __restrict__ d =
      dst_base + static_cast<size_t>(slot[j]) * vec_per_expert;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_expert;
       i += stride)
    d[i] = s[i];
}

void fetch_experts(int64_t src_dev_base, at::Tensor dst, at::Tensor idx,
                   at::Tensor slot, at::Tensor njobs, int64_t expert_bytes,
                   int64_t max_jobs, int64_t blocks_per_job) {
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous());
  TORCH_CHECK(expert_bytes % 16 == 0 && src_dev_base % 16 == 0);
  const at::cuda::CUDAGuard guard(dst.device());
  const dim3 grid(static_cast<unsigned>(blocks_per_job),
                  static_cast<unsigned>(max_jobs));
  fetch_kernel<<<grid, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(src_dev_base),
      reinterpret_cast<uint4*>(dst.data_ptr()),
      idx.data_ptr<int32_t>(), slot.data_ptr<int32_t>(),
      njobs.data_ptr<int32_t>(), static_cast<int32_t>(expert_bytes / 16));
  // No cudaGetLastError + sync here on purpose: a sync is illegal during
  // capture.  C10_CUDA_KERNEL_LAUNCH_CHECK() is sync-free and could be added.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_ptr_of", &device_ptr_of);
  m.def("fetch_experts", &fetch_experts);
}
"""


@contextmanager
def nvcc_env():
    """Everything the cpp-extension investigation verified, applied narrowly
    and then put back.

    The traps, each of which was hit for real:
      * TORCH_CUDA_ARCH_LIST must be set or _get_cuda_arch_flags()
        (cpp_extension.py:2421) calls torch.cuda.get_device_capability() and
        the *build* touches the GPU.
      * CC must be UNSET.  cpp_extension.py:2715 does
        `if CC: cuda_flags = ['-ccbin', CC] + ...`, which overrides nvcc's own
        correct gcc-14 default from its nvcc.profile.  run.sh runs us inside
        nix-shell -p gcc, which may well have exported CC=gcc (15.3), and
        CUDA 12.9's crt/host_config.h hard-errors above gcc 14.
      * CXX must point somewhere real: torch's get_cxx_compiler() falls back to
        bare `c++`, and the nix-shell may or may not provide one.
      * cpp_extension caches CUDA_HOME in a module global at import time, so
        setting the env var is not enough -- patch the attribute too.
    """
    saved = {k: os.environ.get(k) for k in
             ("CC", "CXX", "PATH", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
              "TORCH_EXTENSIONS_DIR")}
    try:
        os.environ.pop("CC", None)
        os.environ["CXX"] = os.path.join(GCC14_BIN, "g++")
        os.environ["CUDA_HOME"] = CUDA_MERGED
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
        os.environ.setdefault("TORCH_EXTENSIONS_DIR",
                              str(Path(__file__).resolve().parent.parent / "torch-ext-cache"))
        os.environ["PATH"] = ":".join(
            [os.path.join(CUDA_MERGED, "bin"), NINJA_BIN, GCC14_BIN,
             os.environ.get("PATH", "")])
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def try_build_extension():
    """Returns (module, None) or (None, reason-to-skip).  Never raises."""
    for p in (os.path.join(CUDA_MERGED, "bin", "nvcc"),
              os.path.join(NINJA_BIN, "ninja"),
              os.path.join(GCC14_BIN, "g++")):
        if not os.path.exists(p):
            return None, (f"{p} is gone.  These are not gc roots; "
                          f"nix-collect-garbage removes them without warning. "
                          f"Ask the nix session for a devShell that pins them.")
    try:
        with nvcc_env():
            import torch.utils.cpp_extension as ce
            ce.CUDA_HOME = CUDA_MERGED          # cached at import; must patch
            d = os.environ["TORCH_EXTENSIONS_DIR"]
            os.makedirs(d, exist_ok=True)
            cu = os.path.join(d, "drama3_hostread.cu")
            if not os.path.exists(cu) or open(cu).read() != EXT_SRC:
                with open(cu, "w") as f:
                    f.write(EXT_SRC)
            print("[ext] building (first run ~40 s; cached afterwards)...",
                  flush=True)
            t0 = time.perf_counter()
            mod = ce.load(name="drama3_hostread", sources=[cu],
                          extra_cuda_cflags=["-O3", "-lineinfo"],
                          verbose=False)
            print(f"[ext] built in {time.perf_counter()-t0:.1f} s", flush=True)
            return mod, None
    except Exception as e:
        return None, f"build failed: {type(e).__name__}: {str(e)[:400]}"


class ExtFetch(Mechanism):
    """The uint4 kernel.  Same contract as TritonFetch; the only reason it
    exists is to separate 'zero-copy PCIe reads are slow' from 'Triton
    generated bad code'.  If this is fast and Triton is not, use this and
    accept the nvcc dependency.  If both are equally slow, the link is the
    limit and no amount of kernel tuning helps."""

    name = "C++ uint4 kernel (nvcc)"

    def __init__(self, mod, src_dev_base, dst, slots, blocks_per_job):
        self.mod, self.base, self.dst = mod, src_dev_base, dst
        self.bpj = blocks_per_job
        self.idx = torch.zeros(slots, dtype=torch.int32, device=DEV)
        self.slot = torch.arange(slots, dtype=torch.int32, device=DEV)
        self.njobs = torch.full((1,), slots, dtype=torch.int32, device=DEV)
        self.jobs = self.grid_y = slots
        self.dynamic_njobs = True

    def set_ids(self, ids):
        self.idx[: len(ids)].copy_(torch.tensor(ids, dtype=torch.int32))
        self.njobs.fill_(len(ids))
        self.jobs = len(ids)

    def resize(self, jobs):
        self.jobs = self.grid_y = jobs
        self.njobs.fill_(jobs)

    def force_njobs(self, n):
        self.njobs.fill_(n)

    def prime(self, host_ids, dev_i32, dev_i64):
        self.idx[: dev_i32.numel()].copy_(dev_i32, non_blocking=True)

    def run(self):
        self.mod.fetch_experts(self.base, self.dst, self.idx, self.slot,
                               self.njobs, EXPERT, self.grid_y, self.bpj)


# ---------------------------------------------------------------- the harness


def bench(mech: Mechanism, jobs: int, id_sets: list[list[int]],
          chk: Checker) -> tuple[float, int]:
    """Bandwidth over rotating expert sets, then a full-byte correctness check.

    WHY ROTATE: the 4090 has 72 MiB of L2 and zero-copy host reads go through
    it.  8 experts is 24 MiB, which FITS.  Re-fetching the same experts in a
    loop would measure L2, not PCIe, and would report a flattering number that
    the real decode path -- which touches a different 8 experts every layer --
    could never reach.  Rotating over hundreds of distinct experts keeps the
    working set an order of magnitude past L2.  (The copy-engine baseline is
    immune to this, since DMA bypasses L2; it rotates anyway, for fairness.)
    """
    mech.resize(jobs)

    # Stage every index set on the device BEFORE timing.  See Mechanism.prime:
    # doing it inside the loop would mean a synchronous pageable H2D per
    # iteration, which serialises the loop and bills its latency to whatever
    # mechanism is being measured.
    d32 = [torch.tensor(s, dtype=torch.int32, device=DEV) for s in id_sets]
    d64 = [torch.tensor(s, dtype=torch.int64, device=DEV) for s in id_sets]
    torch.cuda.synchronize()

    def one(i):
        k = i % len(id_sets)
        mech.prime(id_sets[k], d32[k], d64[k])
        mech.run()

    secs = timed(one, iters=max(8, min(40, 2 * len(id_sets))))
    gbs = jobs * EXPERT / GiB / secs

    probe = id_sets[-1]
    mech.set_ids(probe)
    mech.run()
    torch.cuda.synchronize()
    return gbs, chk.mismatches(probe)


def capture_test(mech: Mechanism, jobs: int, set_a: list[int],
                 set_c: list[int], set_b: list[int],
                 chk: Checker) -> tuple[str, str]:
    """Three distinct expert sets tell the three outcomes apart.

      A -- used for warmup, which actually executes.
      C -- set just before capture.  Stream capture does NOT execute, so the
           only way C's bytes can appear in the slots is if a node baked a
           pointer derived from C.
      B -- set after capture, before replay.  B's bytes mean the mechanism
           read the index at replay time.  This is the only acceptable result.

    So: result == B -> DYNAMIC (usable).  result == C -> FROZEN (the silent
    corruption case; blacklist it).  result == A -> the replay did nothing.
    """
    mech.resize(jobs)
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                mech.set_ids(set_a)
                mech.run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
    except Exception as e:
        return f"warmup failed: {type(e).__name__}", "n/a"

    g = torch.cuda.CUDAGraph()
    try:
        mech.set_ids(set_c)
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            mech.run()
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if torch.cuda.is_current_stream_capturing():
            msg += "  [!! stream still capturing -- later results suspect]"
        return msg, "n/a"

    # Scrub the slots so a replay that does nothing is distinguishable.
    chk.d64[:jobs].fill_(-1)
    try:
        mech.set_ids(set_b)
        g.replay()
        torch.cuda.synchronize()
    except Exception as e:
        return "ok", f"replay failed: {type(e).__name__}: {str(e)[:80]}"

    if chk.mismatches(set_b) == 0:
        verdict = "DYNAMIC"
    elif chk.mismatches(set_c) == 0:
        verdict = "FROZEN(=capture-time experts)"
    else:
        got = [chk.which_expert(s) for s in range(min(jobs, 4))]
        verdict = f"WRONG (slots 0-3 hold experts {got}, wanted {set_b[:4]})"

    # Second replay with a SHRUNKEN device-side job count.  In production the
    # number of misses per layer varies 0..8 while the grid is frozen at 8, so
    # the surplus programs must retire without writing anything.  If they do
    # write, they write expert idx[j] for a stale j into a slot the GEMM is
    # about to read -- silent corruption again, from the other direction.
    if verdict == "DYNAMIC" and mech.dynamic_njobs and jobs >= 2:
        half = jobs // 2
        chk.d64[:jobs].fill_(-1)
        try:
            mech.set_ids(set_b)
            mech.force_njobs(half)
            g.replay()
            torch.cuda.synchronize()
            ok = (chk.mismatches(set_b[:half]) == 0
                  and chk.untouched(half, jobs))
            verdict += "  njobs=" + ("honoured" if ok else "IGNORED(!)")
        except Exception as e:
            verdict += f"  njobs-test raised {type(e).__name__}"
        finally:
            mech.force_njobs(jobs)
    del g
    torch.cuda.synchronize()
    return "ok", verdict


# ---------------------------------------------------------------- main



@dataclass
class Entry:
    """One row of the final table, plus everything needed to produce it.

    Mechanisms are BUILT and BENCHED before any of them is handed to CUDA
    graph capture, and capture is then attempted in increasing order of risk.
    That ordering is not fussiness: a capture that fails badly can leave the
    stream in a capturing state or the context in an error state, and then
    every number after it is noise.  Measuring first means a poisoned context
    costs us the capture verdicts, not the bandwidths.
    """
    mech: "Mechanism"
    result: Result
    do_large: bool = True
    verify: bool = True        # does this source carry the self-ID pattern?
    sets_small: list | None = None
    sets_large: list | None = None
    do_capture: bool = True
    risk: int = 0              # capture attempted in ascending risk order


def gpu_healthy() -> bool:
    """Cheap proof the context still works after a capture attempt."""
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
        torch.zeros(4, device=DEV).add_(1)
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host-gib", type=int, default=4, choices=(1, 2, 4),
                    help="page-locked region.  A POWER OF TWO on purpose: "
                         "ATen's CachingHostAllocator does PowerOf2Ceil(size) "
                         "and never splits blocks, so asking for 3 GiB really "
                         "cudaHostAllocs 4")
    ap.add_argument("--small-jobs", type=int, default=8,
                    help="experts per launch; 8 = one decode layer, all-miss")
    ap.add_argument("--large-jobs", type=int, default=64,
                    help="experts per launch for the asymptotic number")
    ap.add_argument("--no-ext", action="store_true",
                    help="skip the nvcc extension (saves ~40 s on a cold cache)")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--no-capture", action="store_true",
                    help="bandwidth only; skip every graph capture attempt")
    args = ap.parse_args()

    t_start = time.perf_counter()
    host_bytes = args.host_gib * GiB
    preflight(host_bytes)

    host, _h64, n_exp = build_host_region(host_bytes)
    KEEPALIVE.append(host)
    host2d = host[: n_exp * EXPERT].view(n_exp, EXPERT)

    SLOTS = max(args.small_jobs, args.large_jobs)
    dst = torch.zeros(SLOTS, EXPERT, dtype=torch.uint8, device=DEV)
    chk = Checker(dst)
    print(f"[setup] VRAM slot pool {SLOTS} x {EXPERT/2**20:.3f} MiB = "
          f"{SLOTS*EXPERT/2**20:.0f} MiB")

    # Rotating expert sets, plus three disjoint sets reserved for the capture
    # test so warmup / capture-time / replay-time can never collide.
    rng = torch.Generator().manual_seed(20261017)
    perm = torch.randperm(n_exp, generator=rng).tolist()
    set_a = perm[:SLOTS]
    set_c = perm[SLOTS:2 * SLOTS]
    set_b = perm[2 * SLOTS:3 * SLOTS]
    pool = perm[3 * SLOTS:]

    def sets(jobs: int, src_n: int | None = None) -> list[list[int]]:
        k = max(1, len(pool) // jobs)
        out = [pool[i * jobs:(i + 1) * jobs] for i in range(k)]
        if src_n is not None:                      # remap into a smaller source
            out = [[e % src_n for e in s] for s in out[:8]]
        return out

    small_sets, large_sets = sets(args.small_jobs), sets(args.large_jobs)

    entries: list[Entry] = []

    def add(mech, *, large=True, verify=True, capture=True, risk=0,
            ss=None, ls=None, note="") -> None:
        entries.append(Entry(mech, Result(name=mech.name, note=note),
                             do_large=large, verify=verify,
                             sets_small=ss, sets_large=ls,
                             do_capture=capture and not args.no_capture,
                             risk=risk))

    def skip(name: str, why: str) -> None:
        emit(Result(name, skipped=why))

    # ================================================================ build
    # Nothing below touches the GPU beyond allocating; every measurement
    # happens in phase 1.

    # 1. The control the project already depends on.  copy_ on the copy
    #    engine, which is also the mechanism whose capture behaviour is the
    #    designed-against hazard.  risk=1: a captured H2D from the caching
    #    host allocator also records a cudaEvent, and host-side event queries
    #    during capture are not always legal -- if that errors, better it
    #    errors after the kernel verdicts are in.
    add(MemcpyBaseline(host2d, dst), risk=1)

    # 2. Triton.
    if args.no_triton:
        skip("triton", "--no-triton")
    elif not TRITON_OK:
        skip("triton", f"import failed: {TRITON_ERR}")
    else:
        host_i32 = host.view(torch.int32)

        # 2a. Hand Triton the page-locked CPU tensor itself.  THE question
        #     notes/04 has been deferring: does the launcher's
        #     cuPointerGetAttribute translate a host pointer?  Expected yes --
        #     CUDA's Unified Addressing reference guarantees cudaHostAlloc
        #     memory is kernel-addressable at the same address, and torch's
        #     pin_memory() is cudaHostAlloc(cudaHostAllocDefault).  If it
        #     raises, read the message: driver.py:397 has a specific
        #     "cannot be accessed from Triton (cpu tensor?)" ValueError.
        #
        #     Three shapes, because the right one is not obvious and costs one
        #     compile each.  EMOGI measured ~135 outstanding requests as
        #     enough to saturate the link, so more blocks is NOT automatically
        #     better -- and for a fetch that must share SMs with the grouped
        #     GEMM, the smallest shape that reaches full bandwidth is the one
        #     worth having.
        #     bpj=192 is the shape where one grid-stride pass covers a whole
        #     expert (786_624 / 4096 = 192), i.e. no loop.  bpj=8 is the
        #     opposite end -- 64 blocks for 8 jobs, well under the 4090's 128
        #     SMs -- and it is the interesting one, because a fetch kernel
        #     that reaches full bandwidth on a small SM footprint is one that
        #     might actually overlap with the grouped GEMM.
        for blk, bpj, w in ((1024, 32, 4), (4096, 8, 8),
                            (4096, 32, 8), (4096, 192, 8)):
            add(TritonFetch(host_i32, dst,
                            f"triton pinned-CPU-tensor B={blk} bpj={bpj} w={w}",
                            SLOTS, blk, bpj, w),
                large=(blk == 4096 and bpj == 192))

        # 2b. Same kernel, bare address in a duck type.  Matters if the
        #     experts ever come from mmap or a NUMA-aware malloc rather than
        #     torch.  Should be identical to 2a; a difference would mean torch
        #     tensor arguments are doing something we have not accounted for.
        add(TritonFetch(RawPtr(host.data_ptr(), torch.int32), dst,
                        "triton RAW host pointer (duck type)",
                        SLOTS, 4096, 32, 8), large=False)

        # 2c. CONTROL: the same kernel reading VRAM.  This is what separates
        #     "PCIe zero-copy is the limit" from "the kernel is badly written",
        #     and without it the host numbers cannot be interpreted at all.
        #     Hundreds of GiB/s here plus ~20 on host = the link is the
        #     ceiling, which is the good answer.  Slow here too = tune
        #     BLOCK/blocks-per-job before concluding anything about the design.
        vram_src = torch.empty(SLOTS * 2, EXPERT, dtype=torch.uint8, device=DEV)
        KEEPALIVE.append(vram_src)
        add(TritonFetch(RawPtr(vram_src.data_ptr(), torch.int32), dst,
                        "triton reading VRAM (kernel-quality control)",
                        SLOTS, 4096, 32, 8),
            large=False, verify=False, capture=False,
            ss=sets(args.small_jobs, SLOTS * 2),
            note="VRAM source, no pattern to verify")

    # 3. Stock torch, no Triton and no nvcc: index_select over a host pointer
    #    dressed as a CUDA tensor.  If this row is fast, correct and DYNAMIC,
    #    Stage 0 is answered with zero new build tooling.
    try:
        add(IndexSelectFetch(host, n_exp, dst, SLOTS))
    except Exception as e:
        skip("index_select (host ptr as fake CUDA tensor)",
             f"construction failed: {type(e).__name__}: {str(e)[:200]}")

    # 4. The hand-written uint4 kernel -- the bandwidth ceiling a kernel can
    #    reach, and the tiebreaker if Triton's codegen disappoints.
    if args.no_ext:
        skip("C++ uint4 kernel (nvcc)", "--no-ext")
    else:
        mod, why = try_build_extension()
        if mod is None:
            skip("C++ uint4 kernel (nvcc)", why)
        else:
            try:
                # Round-trip through cudaHostGetDevicePointer.  Under UVA with
                # cudaHostAlloc this returns data_ptr() unchanged; doing it
                # anyway costs nothing and is the only form that stays correct
                # if anyone ever sets PYTORCH_CUDA_ALLOC_CONF=
                # pinned_use_cuda_host_register:True, which switches torch to
                # malloc+cudaHostRegister where the two can differ.
                base = mod.device_ptr_of(host)
                print(f"[ext] device_ptr_of(host) == data_ptr(): "
                      f"{base == host.data_ptr()}")
                add(ExtFetch(mod, base, dst, SLOTS, 32))
            except Exception as e:
                skip("C++ uint4 kernel (nvcc)",
                     f"{type(e).__name__}: {str(e)[:200]}")

    # 5. Pinned-vs-pageable sanity, so this run can be compared against the
    #    1.47x in notes/04.  If the ratio does not reproduce, something about
    #    the box changed and none of the other numbers should be trusted.
    #    risk=2: expected to FAIL capture (a memcpy from pageable memory is
    #    not a legal graph node), so it goes last.
    pag2d = torch.empty(64, EXPERT, dtype=torch.uint8)
    KEEPALIVE.append(pag2d)
    add(PageableBaseline(pag2d, dst), large=False, verify=False, risk=2,
        ss=sets(args.small_jobs, 64), note="no pattern; ratio check only")

    # ------------------------------------------------- one-off: loud failure
    # Not a bandwidth row.  The point is to confirm that handing Triton an
    # UN-pinned buffer fails loudly instead of reading garbage -- driver.py:
    # 396-401 turns CUDA_ERROR_INVALID_VALUE into a ValueError.  That is the
    # guarantee that a forgotten pin_memory() can never silently produce
    # wrong audio.  A silent success here would be the alarming result.
    if TRITON_OK and not args.no_triton:
        print("\n[triton with PAGEABLE source -- expecting a loud ValueError]",
              flush=True)
        pag_i32 = torch.empty(4 * ELEMS_I32, dtype=torch.int32)   # 12 MiB
        try:
            mm = TritonFetch(pag_i32, dst, "triton pageable",
                             SLOTS, 1024, 8, 4)
            mm.resize(1)
            mm.set_ids([0])
            mm.run()
            torch.cuda.synchronize()
            emit(Result("triton with PAGEABLE source", capture="n/a",
                        replay="n/a",
                        correct="!! IT RAN -- investigate before shipping"))
        except Exception as e:
            emit(Result("triton with PAGEABLE source", capture="n/a",
                        replay="n/a",
                        correct=f"rejected: {type(e).__name__}: "
                                f"{str(e).splitlines()[0][:80]}"))
        del pag_i32

    # ============================================= phase 1: bandwidth + bytes
    print("\n" + "-" * 78)
    print("PHASE 1 -- bandwidth and byte-exactness (no capture yet)")
    print("-" * 78)
    for en in entries:
        m, r = en.mech, en.result
        print(f"\n[{m.name}]", flush=True)
        try:
            r.gbs_small, miss = bench(m, args.small_jobs,
                                      en.sets_small or small_sets, chk)
            r.correct = ("ok" if miss == 0 else f"{miss} BAD elems") \
                if en.verify else "n/a (no pattern)"
            if en.do_large:
                r.gbs_large, miss2 = bench(m, args.large_jobs,
                                           en.sets_large or large_sets, chk)
                if en.verify and miss2:
                    r.correct += f" /@64 {miss2} BAD"
        except Exception as e:
            r.skipped = f"bench raised {type(e).__name__}: {str(e)[:200]}"
            en.do_capture = False
        RESULTS.append(r)
        if r.skipped:
            print(f"  SKIPPED: {r.skipped}", flush=True)
        else:
            s = f"{r.gbs_small:6.2f}" if r.gbs_small else "     -"
            l = f"{r.gbs_large:6.2f}" if r.gbs_large else "     -"
            print(f"  {s} GiB/s @{args.small_jobs}   {l} GiB/s "
                  f"@{args.large_jobs}   bytes={r.correct}", flush=True)

    # ================================================ phase 2: graph capture
    print("\n" + "-" * 78)
    print("PHASE 2 -- CUDA graph capture, lowest-risk mechanisms first")
    print("-" * 78)
    poisoned = None
    for en in sorted(entries, key=lambda e: e.risk):
        r = en.result
        if not en.do_capture:
            r.capture = r.replay = "not attempted"
            continue
        if poisoned:
            r.capture = f"not attempted (context poisoned by {poisoned})"
            r.replay = "-"
            continue
        print(f"\n[capture] {en.mech.name}", flush=True)
        try:
            r.capture, r.replay = capture_test(
                en.mech, args.small_jobs, set_a[:args.small_jobs],
                set_c[:args.small_jobs], set_b[:args.small_jobs], chk)
        except Exception as e:
            r.capture = f"harness error {type(e).__name__}: {str(e)[:90]}"
            r.replay = "-"
        print(f"  capture={r.capture}\n  replay ={r.replay}", flush=True)
        if not gpu_healthy():
            poisoned = en.mech.name
            print(f"  !! CUDA context no longer healthy after this test; "
                  f"remaining capture tests skipped.", flush=True)

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 118)
    print(f"{'mechanism':<48} {'GiB/s@' + str(args.small_jobs):>9} "
          f"{'GiB/s@' + str(args.large_jobs):>10} {'bytes':>15}  "
          f"{'capture':<20} replay")
    print("-" * 118)
    for r in RESULTS:
        if r.skipped:
            print(f"{r.name:<48} {'-':>9} {'-':>10} {'skipped':>15}  "
                  f"{r.skipped[:55]}")
            continue
        s = f"{r.gbs_small:9.2f}" if r.gbs_small else f"{'-':>9}"
        l = f"{r.gbs_large:10.2f}" if r.gbs_large else f"{'-':>10}"
        print(f"{r.name:<48} {s} {l} {r.correct[:15]:>15}  "
              f"{r.capture[:20]:<20} {r.replay}")
    print("=" * 118)

    print("""
HOW TO READ THIS
  A decode step misses ~145 experts (notes/04: 53.9 % hit at 69 cached per
  layer), so it moves ~425 MiB.  Against the 46.4 ms frame budget (21.53 Hz,
  notes/02), serial transfer time is:

        24 GiB/s -> 17.3 ms/frame      10 GiB/s -> 41.5 ms/frame
        20 GiB/s -> 20.8 ms/frame       5 GiB/s -> 83.0 ms/frame

  Decision rule:
    * best DYNAMIC + byte-correct kernel row within ~25 % of the copy_ control
        -> notes/06 stands.  Next step is task #20: port cached_forward's
           OrderedDict LRU (offload.py:374-416) to device memory, keeping the
           `promised` eviction guard, which is the part that produced fluent
           wrong audio when it was missing.
    * kernel rows well below the control but the VRAM control row is fast
        -> PCIe zero-copy is the limit, not the code.  Re-rank PARTIAL capture
           (graph attention + the 13-layer tower + the 10 sampling passes,
           leave the 40 MoE layers eager) above full capture.
    * the VRAM control row is also slow
        -> the kernel is the problem.  Sweep BLOCK / blocks-per-job and check
           the SASS for LDG.E.128 before concluding anything about the design.
    * ANY row that captures and replays FROZEN
        -> name it in notes/06.  That is the fluent-wrong-audio failure, and
           the only defence is knowing which mechanisms do it.

  WHAT THIS DOES NOT MEASURE, and it is the main remaining risk:
  a fetch KERNEL occupies SMs, so it contends with the grouped GEMM in a way
  the copy engine does not.  Today's ~0.64 of RTF spent on transfer is partly
  overlapped by the copy engine; a kernel may serialise against compute
  instead.  Matching the control here is necessary but not sufficient -- see
  the overlap question under task #20.
""")
    print(f"total wall time {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
