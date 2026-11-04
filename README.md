# hekit

hekit (temporary name) is an MIT-licensed Rust RNS-CKKS library for encrypted neural-network training (CPU first, CUDA later).

## Crates

- `hekit-core`: RNS arithmetic, NTT, sampling
- `hekit-ckks`: RNS-CKKS scheme
- `hekit-nn`: encrypted NN training
- `hekit-bench`: benchmarks and cost model

## License

MIT (see `LICENSE`). Dependencies must be MIT / Apache-2.0 / BSD-style only; `tfhe` and `concrete*` are banned.

Parts of `hekit-ckks` (the CKKS encoder's DWT, root table and slot map) are ported from Microsoft SEAL 4.1.2 (MIT), and other parts reproduce SEAL's formulas and formats for interoperability; see `THIRD_PARTY_NOTICES` for the list and SEAL's copyright and permission notice.

## Data

`crates/hekit-bench/data/iris.csv` is Fisher's Iris data set (R. A. Fisher, 1936), from the UCI Machine Learning Repository, "Iris", doi:10.24432/C56C76, licensed CC BY 4.0. The copy is the one bundled with scikit-learn (identical to R's `iris`; it differs from the UCI file in two known-erroneous entries), taken from `spikes/seal-2020/data/`. `crates/hekit-bench/data/iris_split_seed2020.txt` is that spike's 80:20 train/test split (seed 2020).

MNIST (Y. LeCun, C. Cortes, C. J. C. Burges) is not in the repository: `scripts/fetch_mnist.sh` downloads it (torchvision's mirror, MD5-checked) into the gitignored `data/mnist/` for `crates/hekit-bench/src/bin/mnist.rs`.

## Status

M1 complete: the CKKS core (`hekit-core`, `hekit-ckks`): RNS / NTT arithmetic, the `seal_2020` and `n16_train` presets, encoding, encryption, hybrid key switching, rotations with hoisting, BSGS matrix-vector products, operation counters, a precision suite and known-answer interop with SEAL 4.1.2. See `docs/m1-report.md`. Next: M2, encrypted NN training (`hekit-nn`).

M0 spikes (throwaway experiments under `spikes/`) are not part of the workspace.
