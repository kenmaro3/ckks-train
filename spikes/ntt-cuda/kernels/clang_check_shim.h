// Offline compile check only (NOT used by NVRTC): lets Homebrew clang (NVPTX backend) compile
// kernels/ntt.cu to PTX without a CUDA toolkit, by providing the few declarations NVRTC has
// built in. Usage: see spikes/RESULTS.md, Task 3.
#pragma once
#include "__clang_cuda_builtin_vars.h" // threadIdx, blockIdx, blockDim (clang resource dir)
typedef __SIZE_TYPE__ size_t;
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
static __device__ __forceinline__ unsigned long long __umul64hi(unsigned long long a,
                                                                 unsigned long long b) {
    return __nvvm_mulhi_ull(a, b);
}
// __syncthreads is a clang builtin for NVPTX.
