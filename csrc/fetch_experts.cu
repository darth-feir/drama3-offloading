// Fetch MoE experts from pinned host memory into a GPU cache slot.
//
// WHY THIS IS A KERNEL AND NOT A MEMCPY
// A captured cudaMemcpyAsync freezes its source pointer, so on replay it
// copies the SAME expert forever. Measured, not assumed: bench_hostread.py
// shows the copy_ row replaying FROZEN(=capture-time experts) while every
// kernel row replays DYNAMIC. A frozen fetch produces fluent, confident,
// wrong audio -- the worst failure this model has.
//
// A kernel reads `idx[j]` from device memory at REPLAY time, so the same
// captured graph serves different experts each step.
//
// WHY uint4
// 16 B per thread means a warp issues 512 contiguous bytes, which the
// coalescer turns into four fully aligned 128 B PCIe reads. Measured at
// 24.87 GiB/s @8 experts against the copy engine's 24.60 -- parity.
//
// The vectorisation is explicit here on purpose. Triton reaches the same
// speed but only because its JIT auto-attaches a divisibility hint; without
// it the identical kernel drops to 19 scalar loads, roughly 4x slower, with
// nothing in the source to indicate why. If this ever gets slow you can read
// the reason. That is the whole argument for this file existing.
//
// DEBUGGING A SLOWDOWN
//   nvcc -arch=sm_89 -cubin -o /tmp/f.cubin csrc/fetch_experts.cu
//   cuobjdump -sass /tmp/f.cubin | grep -E 'LDG|STG'
// Expect LDG.E.128 / STG.E.128. Anything narrower means the alignment
// precondition broke -- check expert_bytes % 16 and the base pointer.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cstdint>

// Map a page-locked host allocation into the device address space.
// cudaHostAlloc'd memory is already device-addressable under UVA, so this
// returns the same value -- but going through the API means it still works
// if torch ever switches to malloc + cudaHostRegister.
int64_t device_ptr_of(at::Tensor t) {
  TORCH_CHECK(t.device().is_cpu() && t.is_pinned() && t.is_contiguous(),
              "expected a contiguous page-locked CPU tensor");
  void* dptr = nullptr;
  cudaError_t e = cudaHostGetDevicePointer(&dptr, t.data_ptr(), 0);
  TORCH_CHECK(e == cudaSuccess, cudaGetErrorString(e));
  return reinterpret_cast<int64_t>(dptr);
}

__global__ __launch_bounds__(256) void fetch_kernel(
    const uint4* __restrict__ src_base,
    uint4* __restrict__ dst_base,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ slot,
    const int32_t* __restrict__ njobs,
    int32_t vec_per_expert) {
  const int j = blockIdx.y;
  // Read at REPLAY, not at capture. The grid is sized for the worst case
  // (every id a miss) so the launch shape stays static -- which capture
  // requires -- while the work actually done varies with the data.
  if (j >= *njobs) return;
  const uint4* __restrict__ s =
      src_base + static_cast<size_t>(idx[j]) * vec_per_expert;
  uint4* __restrict__ d =
      dst_base + static_cast<size_t>(slot[j]) * vec_per_expert;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < vec_per_expert;
       i += stride)
    d[i] = s[i];
}

// blocks_per_job and threads are exposed rather than hard-coded so occupancy
// can be tuned from Python without a rebuild. That matters for the open
// question in notes/06: this kernel occupies SMs and competes with the MoE
// GEMM, where the copy engine overlaps for free. Throttling it may turn out
// to be worth more than its raw bandwidth.
void fetch_experts(int64_t src_dev_base, at::Tensor dst, at::Tensor idx,
                   at::Tensor slot, at::Tensor njobs, int64_t expert_bytes,
                   int64_t max_jobs, int64_t blocks_per_job, int64_t threads) {
  TORCH_CHECK(dst.is_cuda() && dst.is_contiguous(), "dst must be CUDA+contiguous");
  TORCH_CHECK(expert_bytes % 16 == 0, "expert_bytes must be 16 B aligned");
  TORCH_CHECK(src_dev_base % 16 == 0, "host base must be 16 B aligned");
  TORCH_CHECK(threads == 256, "kernel is compiled with __launch_bounds__(256)");
  const at::cuda::CUDAGuard guard(dst.device());
  const dim3 grid(static_cast<unsigned>(blocks_per_job),
                  static_cast<unsigned>(max_jobs));
  fetch_kernel<<<grid, static_cast<unsigned>(threads), 0,
                 c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(src_dev_base),
      reinterpret_cast<uint4*>(dst.data_ptr()),
      idx.data_ptr<int32_t>(), slot.data_ptr<int32_t>(),
      njobs.data_ptr<int32_t>(), static_cast<int32_t>(expert_bytes / 16));
  // Deliberately no cudaGetLastError()+sync: a sync is illegal during graph
  // capture. C10_CUDA_KERNEL_LAUNCH_CHECK() is sync-free if checking is wanted.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("device_ptr_of", &device_ptr_of, "device pointer for pinned host tensor");
  m.def("fetch_experts", &fetch_experts, "host->device expert fetch");
}
