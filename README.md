# ckks-train

ckks-train is an MIT-licensed Rust library for training neural networks on data encrypted
with RNS-CKKS. It extends the 2020 work of Mihara et al. (arXiv:2012.13552), which trained an
MLP under CKKS with a client refreshing the ciphertexts, to training without the client, using
bootstrapping, and adds a CUDA backend.

The workspace crates use the `ckks-train-*` names. Historical measurements retain the
`hekit-*` names they were produced with; see [docs/release.md](docs/release.md).

## Overview

- **RNS-CKKS** (`ckks-train-core`, `ckks-train-ckks`): RNS/NTT arithmetic, encoding bit-exact with
  Microsoft SEAL 4.1.2, hybrid key switching, hoisted rotations, baby-step giant-step (BSGS)
  matrix-vector products, exact-scale arithmetic and an operation counter that counts every
  key switch.
- **Bootstrapping with sparse-secret encapsulation (SSE)** (`ckks-train-ckks::bootstrap`): the
  ciphertext is switched to a sparse ephemeral key (h' = 32) before ModRaise, then
  CoeffToSlot, a Chebyshev EvalMod and SlotToCoeff. On the `n16_boot` preset (N = 2^16) one
  bootstrap gives 17.46 bits of precision with 111 key switches and leaves 16 levels.
- **CUDA backend** (`ckks-train-cuda`, feature `cuda`): batched NTT, base conversion, key switching
  and the full evaluator on the GPU, bit-exact with the CPU code. The `ckks-train-nn` trainers run
  on it unchanged through the `HeOps` trait.
- **Encrypted MLP training** (`ckks-train-nn`), in two modes:
  - *interactive* (`Trainer`): the key owner refreshes the ciphertexts after each update, as
    in the 2020 work;
  - *non-interactive* (`NonInteractiveTrainer`): the server trains alone, restoring levels by
    bootstrapping. Weights are repacked compactly so that one bootstrap refreshes a whole
    layer, and the bootstrap placement is configurable (`AfterUpdate`, `WhenNeeded`,
    `AfterActivation`).
  - Block MLPs with local classifiers (`BlockMlp`) reproduce the structure of ReBoot's
    eMLP-2/3.
- **Comparison with ReBoot** (AAAI-26): the benchmarks compare key switches per training
  sample and wall-clock time with ReBoot's published implementation on the same machine, with
  its configuration, initial weights and sample order. The figures and their caveats are in
  [docs/m3-report.md](docs/m3-report.md). The first full-length run (FULL-1, B=64,
  one initial-weight seed) is archived. FULL-2 is still pending; equal-security comparison
  and full-setting official ReBoot CPU timing remain unresolved.

Architecture: [docs/architecture.md](docs/architecture.md). Changes:
[CHANGELOG.md](CHANGELOG.md).

## Build

Requirements: Rust **1.85** (tested with 1.85.1; `rust-toolchain.toml` selects the
1.85 toolchain through rustup), edition 2024. No C/C++ toolchain is needed.

```sh
cargo build --release --workspace
cargo test --workspace            # about 400 tests; tests above ~10 GB of RAM are #[ignore]d
cargo test --workspace -- --ignored   # the heavy N = 2^16 bootstrapping tests (~13 GB RAM)
```

Optional features:

- `parallel` (`ckks-train-ckks`, `ckks-train-bench`): rayon data parallelism. Results are bit-identical
  with and without it. The thread count is rayon's (`RAYON_NUM_THREADS`).
- `cuda` (`ckks-train-cuda`, `ckks-train-bench`): the GPU backend.

```sh
cargo build --release -p ckks-train-cuda --features cuda
cargo test -p ckks-train-cuda --features cuda --release -- --ignored   # needs a GPU
```

`cudarc` loads `libcuda` and `libnvrtc` at run time, so the `cuda` feature builds on machines
without CUDA. Running it needs an NVIDIA driver supporting CUDA >= 13.1 and NVRTC 13.1 on the
library path (for example `/usr/local/cuda-13.1/lib64` in `LD_LIBRARY_PATH`). Kernels are
compiled at run time by NVRTC with `--fmad=false`. Each GPU test declares the GPU memory it
needs and prints a `SKIP` line when it is not available, so a passing exit code alone is not
proof that a GPU test ran.

## Reproducing the benchmarks

`scripts/run_all.sh` currently runs a **partial experiment matrix**. The complete registered
matrix and its CPU/GPU training connections are still being implemented; the full `--quick`
validation has not completed. See [results/README.md](results/README.md) for supported rows,
missing measurements, and runtime estimates. To regenerate the draft's tables from archived
outputs without building or launching experiments, run:

```sh
scripts/run_all.sh --tables-only
```

The [paper draft](paper/README.md) describes the available results and unresolved comparisons;
it is not submission-ready. The datasets that are not
in the repository are fetched by `scripts/fetch_mnist.sh` and `scripts/fetch_fashion_mnist.sh`.
Every binary records its own provenance (commit, command line, threads, load, memory) in its
result JSON. Comparisons of wall-clock time are only meaningful on the same machine with the
same thread count.

## Security

The parameter presets are estimated with the lattice-estimator (secret and error
distributions as used; details and the full output in [docs/security.md](docs/security.md)):

| preset / key | BDGL16 (HE Standard sieve) | MATZOV (estimator default) |
|---|---|---|
| `n16_boot`, main key (dense ternary) | 131.5 bits | 129.3 bits |
| `n16_boot`, SSE ephemeral key (h' = 32) | 133.8 bits | 134.2 bits |
| `n16_train` | 143.9 bits | 141.2 bits |

The Core-SVP (ADPS16) estimates are lower, below 128 bits for these presets; they are
reported in `docs/security.md` for information. The HE Standard uses the sieve model. The
presets `test_tiny` and `n16_boot_sparse` (about 122 bits) are not secure and exist for tests
and comparison only.

**This is research code. It has not been audited** and was not designed or reviewed for
side-channel resistance (constant-time execution) or for attacks that use decryption results
(IND-CPA-D). Do not use it to protect real data.

## Citation

The paper describing this library is in preparation. Until it appears, please cite:

```bibtex
@misc{ckks-train,
  title  = {ckks-train: {RNS-CKKS} with bootstrapping and {CUDA} for encrypted neural-network training},
  author = {TODO},
  year   = {2026},
  note   = {Software, https://github.com/kenmaro3/ckks-train},
}
```

The 2020 work it extends is arXiv:2012.13552 (Mihara et al.).

## License

MIT; see [LICENSE](LICENSE). Third-party notices are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES):

- **Microsoft SEAL 4.1.2 (MIT).** Parts of the CKKS encoder (`ckks-train-ckks`: the DWT, the root
  table, the slot index map) are ported from SEAL; other parts reproduce SEAL's formulas and
  serialization format for interoperability. SEAL's copyright and permission notice is
  reproduced in `THIRD_PARTY_NOTICES`. No SEAL source is vendored.
- **Lattigo (Apache-2.0), acknowledgement only.** The baby-step split rule of the Chebyshev
  evaluation follows a cost formula known from Lattigo's implementation; it was reimplemented
  from the formula and no Lattigo code was used.
- **libloading (ISC)**, a dependency of `cudarc` (feature `cuda` only).
- **Unicode character-property tables:** `unicode-ident 1.0.27` declares
  `(MIT OR Apache-2.0) AND Unicode-3.0`. The Unicode data license is an approved
  exception for that package and version only; its notice is included in
  `THIRD_PARTY_NOTICES`. Other dependencies must offer MIT / Apache-2.0 / BSD / ISC.
  `cargo deny check licenses bans` checks this scoped policy.

**GPL note.** ReBoot (GPL-3.0) was only *run*, as a baseline, in a separate environment. None
of its code is included in or linked by this repository, and no part of this library is
derived from it. Values taken from
ReBoot's published configuration (initial weights, sample order, learning-rate schedule,
encryption parameters) are used as data.

### Data

- **Iris** (R. A. Fisher, 1936): UCI Machine Learning Repository, doi:10.24432/C56C76,
  CC BY 4.0. `crates/ckks-train-bench/data/iris.csv` is the copy bundled with scikit-learn.
- **Breast Cancer Wisconsin (Diagnostic), WDBC** (W. H. Wolberg, W. N. Street,
  O. L. Mangasarian): UCI Machine Learning Repository, doi:10.24432/C5DW2B, CC BY 4.0.
  Distributed unmodified as `crates/ckks-train-bench/data/breast_cancer.csv`.
- **MNIST** (Y. LeCun, C. Cortes, C. J. C. Burges): not distributed;
  `scripts/fetch_mnist.sh` downloads it.
- **Fashion-MNIST** (H. Xiao, K. Rasul, R. Vollgraf; Zalando SE, MIT): not distributed;
  `scripts/fetch_fashion_mnist.sh` downloads it.

Sources and checksums are in `crates/ckks-train-bench/data/README.md`.

## Status

- **M0** (spikes: NTT, cost model, the 2020 SEAL reproduction): feasibility work recorded.
  Cost estimates and CPU-emulated CUDA checks are not hardware measurements. The spikes
  under `spikes/` are not part of the workspace; `docs/m0-report.md`.
- **M1** (CKKS core): implemented, with known-answer tests bit-exact with SEAL 4.1.2.
  Some key-switch performance targets remain unmet; `docs/m1-report.md`.
- **M2** (interactive encrypted training): done. Iris reproduction (test accuracy 0.967) and
  MNIST eMLP-1; `docs/m2-report.md`.
- **M3** (bootstrapping, non-interactive training, ReBoot comparison): implemented; final RQ1
  assessment awaits the full-length runs (4800 samples x 10 epochs) and retains the
  security-comparability limitation. `docs/m3-report.md`.
- **M4** (CUDA backend): the GPU kernels and evaluator are verified bit-exact with the CPU on an
  RTX 4090; GPU bootstrapping is implemented but its measurement on hardware is deferred.
  GPU training and matched CPU/GPU ablations remain unmeasured; `docs/m4-report.md`.
- **M5** (block MLPs, additional datasets, release, paper): in progress.
