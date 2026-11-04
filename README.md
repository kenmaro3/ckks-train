# hekit

hekit (temporary name) is an MIT-licensed Rust RNS-CKKS library for encrypted neural-network training (CPU first, CUDA later).

## Crates

- `hekit-core`: RNS arithmetic, NTT, sampling
- `hekit-ckks`: RNS-CKKS scheme
- `hekit-nn`: encrypted NN training
- `hekit-bench`: benchmarks and cost model

## License

MIT (see `LICENSE`). Dependencies must be MIT / Apache-2.0 / BSD-style only; `tfhe` and `concrete*` are banned.

## Status

M0 spikes (throwaway experiments under `spikes/`, not part of the workspace).
