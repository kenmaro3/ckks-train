# Precision of the basic operations

Measured by `cargo run -p ckks-train-ckks --release --example precision` (`ckks_train_ckks::precision::run_precision_suite`).

- Date: 2026-11-05
- Machine: Apple M3 Pro (12 cores, 36 GB), macOS; rustc 1.85.1; ran in about 21 s for both presets (including key generation)
- Commit: 13d2f59 plus the Task 20 changes (the table is produced by the same code that is committed with it)
- Rerun 2026-11-05 (Task 24) after the CDT Gaussian sampler (e2326a3) and the NTT / key-switching performance commits
  (2152d02, 69225da): the error samples changed, so the numbers moved by up to 0.4 bit; the tables below are the rerun,
  the Task 20 values are in parentheses where they differ.
- Rerun 2026-11-05 after the M1 fix wave (commit 37f9502: strided BSGS, `Rng` without `Clone`, CDT refactor):
  every value identical to the Task 24 rerun below (no randomness consumption changed).

`bits` = `-log2(max abs error)` over all N/2 slots, inputs uniform in [-1, 1) at the top level with the preset scale, fixed seeds.
`level` is the level of the decrypted ciphertext. `matvec` is a 16 x 16 matrix-vector product (BSGS, plaintext diagonals, then one rescale);
`matvec_enc` the same with encrypted diagonals. `rotate` is a rotation by 1. Test thresholds: seal_2020 >= 20 bits, n16_train >= 24 bits
(`crates/ckks-train-ckks/tests/precision.rs`, about 19 s in the normal test suite).

### seal_2020

| op | level | bits |
|----|-------|------|
| encode/decode | 9 | 32.14 |
| encrypt | 9 | 24.81 |
| add | 9 | 24.32 (24.36) |
| mul+rescale | 8 | 24.13 (24.41) |
| rotate | 9 | 22.34 (22.73) |
| matvec | 8 | 22.26 (21.93) |
| matvec_enc | 8 | 22.33 (22.10) |

### n16_train

| op | level | bits |
|----|-------|------|
| encode/decode | 31 | 33.70 |
| encrypt | 31 | 25.72 |
| add | 31 | 25.19 (24.83) |
| mul+rescale | 30 | 25.38 (25.44) |
| rotate | 31 | 25.23 (25.44) |
| matvec | 30 | 24.71 (24.88) |
| matvec_enc | 30 | 24.47 (24.53) |

Notes: the encode/decode floor is the rounding of coefficients at scale 2^40 / 2^42 over N coefficients. Encryption noise
(sigma 3.2, ternary secret) dominates the other rows; the max over 2^14 / 2^15 slots of a roughly Gaussian error costs
about 2 bits against the typical error (the maximum of 2^14 Gaussian samples is about 4 sigma). `matvec` loses another 2-3 bits to the 16 summed products and the key switches of the giant steps.
