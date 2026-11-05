// Offline compile check only (NOT used by NVRTC): lets Homebrew clang (NVPTX backend) compile
// the kernels in this directory to PTX without a CUDA toolkit, by declaring the few intrinsics
// NVRTC has built in. A syntax and type check, nothing more. Usage (per file, per sm_XY):
//   /opt/homebrew/opt/llvm/bin/clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 \
//     -nocudainc -nocudalib -O3 -Wall -Wextra -include kernels/clang_check_shim.h \
//     -S kernels/basis.cu -o /tmp/basis.ptx
// Extends spikes/ntt-cuda/kernels/clang_check_shim.h with the intrinsics of pointwise.cu and
// basis.cu.
#pragma once
#include "__clang_cuda_builtin_vars.h" // threadIdx, blockIdx, blockDim (clang resource dir)
typedef __SIZE_TYPE__ size_t;
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __restrict__ __restrict
static __device__ __forceinline__ unsigned long long __umul64hi(unsigned long long a,
                                                                 unsigned long long b) {
    return __nvvm_mulhi_ull(a, b);
}
static __device__ __forceinline__ unsigned int __brev(unsigned int x) { return __builtin_bitreverse32(x); }
static __device__ __forceinline__ double __ull2double_rn(unsigned long long x) {
    return __nvvm_ull2d_rn(x);
}
static __device__ __forceinline__ double __dmul_rn(double a, double b) {
    return __nvvm_mul_rn_d(a, b);
}
static __device__ __forceinline__ double __dadd_rn(double a, double b) {
    return __nvvm_add_rn_d(a, b);
}
static __device__ __forceinline__ double __longlong_as_double(long long x) {
    return __builtin_bit_cast(double, x);
}
static __device__ __forceinline__ unsigned long long __double2ull_rz(double x) {
    return __nvvm_d2ull_rz(x);
}
// __syncthreads is a clang builtin for NVPTX.
