# ckks-train-cuda

Part of **ckks-train**, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network
training (repository: <https://github.com/kenmaro3/ckks-train>). The workspace uses the
`ckks-train-*` crate names.

The CUDA backend of `ckks-train-ckks`. With the feature `cuda` (off by default):

- `CudaDevice`, `DevicePoly`, `DevicePlaintext`, `CudaCiphertext`, `DeviceKsk`: device memory;
- batched NTT, pointwise arithmetic, rescale, base conversion and hybrid key switching;
- `CudaEvaluator`: `HeOps` on the GPU, including BSGS matrix-vector products, so the
  `ckks-train-nn` trainers run on it unchanged;
- `CudaBootstrapper`: GPU bootstrapping (the code exists; its measurement on hardware is
  pending).

The kernels and the evaluator are bit-exact with the CPU code (the GPU tests, run on an RTX
4090). `cudarc` loads
`libcuda` and `libnvrtc` at run time, so the crate builds without a CUDA toolkit; running it
needs an NVIDIA driver supporting CUDA >= 13.1 and NVRTC 13.1 on the library path. Kernels are
compiled by NVRTC with `--fmad=false`.

Without the feature only host-side launch plans and tables are compiled, with CPU tests that
replay the kernels.

Research code, not audited: see the security note in the
[repository README](https://github.com/kenmaro3/ckks-train#security).

License: MIT. `cudarc` pulls in `libloading` (ISC); see `THIRD_PARTY_NOTICES`.
