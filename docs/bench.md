# Benchmarks

Machine: Apple M3 Pro (12 cores; benches use 1 core). Criterion, `cargo bench -p hekit-core --bench ntt`
(profile.bench: opt-level 3, LTO, codegen-units 1). Values are criterion's point estimate (mean) of
a single in-place transform, input clone excluded from timing.

## NTT (hekit-core, Task 10)

| date | rustc | N | q | forward (µs) | inverse (µs) | `uptime` load (1-min, before -> after) |
|------|-------|---|---|--------------|--------------|----------------------------------------|
| 2026-11-04 | 1.85.1 | 2^16 | 1152921504606584833 (60-bit) | 443.1 (CI 435.0-453.0) | 398.9 (CI 397.3-401.1) | 4.38 -> 7.38 |
| 2026-11-04 | 1.85.1 | 2^16 | 1152921504606584833 (60-bit) | 435.5 (CI 434.4-436.9) | 392.0 (CI 391.3-392.8) | 3.92 -> 5.73 |

Machine was not fully idle (other sessions running); load shown is the 1-min average at start and end.

Reference points from M0 (`spikes/RESULTS.md`, idle machine): Rust spike forward 417.8 µs, SEAL 4.1
forward 373.7 µs. hekit-core forward is about 4% above the spike and about 17% above SEAL
(under load; the spike itself was measured at 3.0-3.2 load).

## Rotations: `rotate` x 32 vs `hoisted_rotate` (hekit-ckks, Task 18)

`cargo bench -p hekit-ckks --bench rotate` (`crates/hekit-ckks/benches/rotate.rs`): one fresh
seal_2020 ciphertext at the top level (N = 2^15, level 9, dnum 10, alpha 1, one 60-bit special
prime), steps 1..=32, one Galois key per step. `rotate_x32` calls `rotate` 32 times;
`hoisted_rotate_32` is one `hoisted_rotate(ct, &[1..=32])`. Single thread, criterion with
10 samples; values are the point estimate (CI in parentheses).

| date | rustc | `rotate` x 32 | `hoisted_rotate` (32 steps) | speed-up | `uptime` load (1-min, before -> after) |
|------|-------|---------------|-----------------------------|----------|----------------------------------------|
| 2026-11-05 | 1.85.1 | 1.424 s (1.420-1.428) | 666.2 ms (660.4-671.6) | 2.14x | 7.42 -> 6.98 (before lazy reduction) |
| 2026-11-05 | 1.85.1 | 1.179 s (1.172-1.186) | 443.9 ms (432.4-458.3) | 2.66x | 6.72 -> 5.83 |
| 2026-11-05 | 1.85.1 | 1.185 s (1.176-1.197) | 432.9 ms (429.0-436.3) | 2.74x | 5.17 -> 4.82 |

The first row is from an earlier revision whose key inner product reduced every product
(`Modulus::mul` + `add`); the committed `apply_key` accumulates exact `u128` products and
reduces once per coefficient (lazy reduction), which made a single rotation about 17% faster
(44.5 -> 37.0 ms) and a hoisted step about 35% faster (20.8 -> 13.5 ms).

Per rotation (last row): `rotate` costs 37.0 ms, of which the digit decomposition (10 INTT +
100 NTT of 2^15 points for the 10 one-prime digits) is about 23.5 ms. A hoisted step costs
13.5 ms: the automorphism of the 10 lifted digits (110 limb permutations), the inner product
with the key, and the two `mod_down` (22 NTT-sized transforms). That gives the 2.7x speed-up
within the 2-3x target. Further gains would come from fusing the permutation into the inner
product (index the digit through the Galois permutation instead of materialising permuted
copies); not done here. The machine was not idle (other sessions running).
