# Changelog

All notable changes to ckks-train are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Finalized the M3 report using both full-setting encrypted training runs. The
  baseline-batch accuracy criterion passes, while one normalized key-switch count
  criterion and the common-security condition remain unmet; full-setting ReBoot
  CPU timing is unmeasured.
- Adopted the repaired R-47-3 GPU independent-product measurements, separating
  within-run latency comparisons, memory-skipped tests and pending GPU training.
- Updated the paper and arXiv metadata with those results. The new local 10-page
  PDF and 19-member source ZIP pass content review and isolated-source rebuild
  checks; the published preview assets remain unchanged and arXiv is not submitted.

## [1.0.0] - unreleased

Stable release preparation; the research preview below is already public.
Workspace version fields are `1.0.0`, but stable v1.0 has not been released.
The crate names are
`ckks-train-core`, `ckks-train-ckks`, `ckks-train-nn` and `ckks-train-cuda`.
The `ckks-train-bench` crate has `publish = false`.

### Added

- **M0, spikes** (outside the workspace): NTT and CUDA NTT prototypes, the key-switch cost
  model, the 2020 SEAL-based training reproduction used as the reference, and a separate
  environment that runs the ReBoot baseline (no ReBoot code is in the repository).
- **M1, CKKS core** (`ckks-train-core`, `ckks-train-ckks`):
  - RNS arithmetic with Barrett and Shoup reduction, NTT-friendly prime generation, the
    negacyclic NTT, fast base conversion, and ChaCha20-based sampling;
  - RNS-CKKS: the presets `seal_2020` (N = 2^15) and `n16_train` (N = 2^16), an encoder
    bit-exact with SEAL 4.1.2, encryption and decryption, hybrid key switching,
    relinearization, rescaling, rotations with hoisting and conjugation;
  - diagonal-encoded matrices and BSGS matrix-vector products;
  - `OpCounter` (rotations, relinearizations, SSE switches, multiplications, rescales,
    bootstraps) and a precision suite;
  - SEAL serialization (`seal_compat`) and known-answer tests against SEAL 4.1.2.
- **M2, interactive training** (`ckks-train-nn`, `ckks-train-bench`):
  - the batched slot layout, encrypted dense layers with transposition by rotations,
    polynomial activations, squared-error loss, SGD and Nesterov momentum;
  - `Trainer` with client-side refresh, and a plaintext simulator of levels and counts;
  - the `iris` reproduction of the 2020 work and the `mnist` eMLP-1 benchmark.
- **M3, bootstrapping and non-interactive training**:
  - `Bootstrapper` on the `n16_boot` preset with sparse-secret encapsulation, CoeffToSlot,
    a Chebyshev EvalMod (level-optimal, scale-invariant) and SlotToCoeff: 17.46 bits of
    precision, 111 key switches per bootstrap, 16 levels left;
  - `NonInteractiveTrainer` with compact repacking, a fused gradient/pack mask and the
    bootstrap placements `AfterUpdate`, `WhenNeeded` and `AfterActivation`;
  - closed-form prediction of every operation count, checked against the counters;
  - ReBoot-faithful configurations (input, initial weights, sample order, learning-rate
    schedule taken as data) and the ReBoot slot layout as a baseline;
  - lattice-estimator security estimates for every preset and key (`docs/security.md`).
- **M4, CUDA backend** (`ckks-train-cuda`):
  - device memory, batched NTT, pointwise arithmetic, base conversion and hybrid key
    switching, compiled at run time by NVRTC through `cudarc` with dynamic loading;
  - `HeOps` and `BootstrapOps` traits so that the trainers are generic over the backend;
    `EvalConfig` flags (hoisting, BSGS, lazy relinearization) for the ablation;
  - `CudaEvaluator`, bit-exact with the CPU evaluator on an RTX 4090, with multi-stream
    matrix-vector products;
  - `CudaBootstrapper` (GPU bootstrapping; hardware measurement pending);
  - the `cuda_ops` benchmark and the `ablation` binary.
- **M5, extensions and release**:
  - the `plain` binary (plaintext twins), WDBC and Fashion-MNIST configurations;
  - `BlockMlp` with local classifiers (ReBoot's eMLP-2/3 structure) and its trainers;
  - result provenance in every result JSON;
  - release preparation: crate metadata, READMEs, `docs/architecture.md`,
    `docs/release.md` and `scripts/rename.sh`.

### Security

- Estimated security (BDGL16 / MATZOV): `n16_boot` main key 131.5 / 129.3 bits, SSE key
  133.8 / 134.2 bits, `n16_train` 143.9 / 141.2 bits. Core-SVP is lower and is reported in
  `docs/security.md`. Research code, not audited.

## [1.0.0-preview.1] - 2026-11-08

- Published a research preview from fixed source `6fba46c`, including the source,
  10-page paper PDF and arXiv preparation ZIP. The preview assets remain fixed.
- The preview's six Linux CI jobs and distribution checks passed for that source;
  this does not validate later source changes or close the research gates.
- Stable v1.0 and arXiv submission remain pending. See the
  [publication evidence](results/validation/research-preview-public-release-6fba46c/README.md).

[Unreleased]: https://github.com/kenmaro3/ckks-train/compare/v1.0.0-preview.1...HEAD
[1.0.0]: docs/release.md
[1.0.0-preview.1]: https://github.com/kenmaro3/ckks-train/releases/tag/v1.0.0-preview.1
