# SEAL 4.1.2 encoder known-answer vectors

`kat.cpp` generates the vectors that are hard-coded in the `matches_seal_4_1_encoder` test in
`crates/ckks-train-ckks/src/encoder.rs`. That test pins hekit's CKKS encoder (slot order with
generator 3, the DWT, and rounding) to Microsoft SEAL 4.1.2's `CKKSEncoder`.

Parameters:
- CKKS with N ∈ {16, 32}
- `CoeffModulus::Create(N, {60, 40, 60})`, so the data primes are {60, 40} and the last 60-bit prime is the special prime
- `sec_level_type::none`
- scale 2^30
- encode at `first_parms_id()`

Inputs:
- a real vector `v[i] = 0.37 i - 1.1 + (i % 3 == 0 ? 0.25 : -0.125)`
- a complex vector `(0.5 - 0.21 i, 0.13 i - 0.4)`
- the short real input `[0.75, -0.5, 0.125]`, zero-padded

The program prints the inputs with `%.17g`, which the test copies verbatim. For each encoding it prints the centered
coefficients of **limb 0** after `inverse_ntt_negacyclic_harvey`. The program checks that these equal limb 1's centered
coefficients, so the integers do not depend on the prime.

## Build and run

From the repository root:

```sh
cmake -S tools/seal-kat/encoder_kat -B tools/seal-kat/encoder_kat/build -DCMAKE_BUILD_TYPE=Release
cmake --build tools/seal-kat/encoder_kat/build -j
tools/seal-kat/encoder_kat/build/encoder_kat
```

To build offline, reuse the SEAL checkout of the spike (if it has been configured):
`-DFETCHCONTENT_SOURCE_DIR_SEAL=$PWD/spikes/seal-2020/build/_deps/seal-src`.

The `build/` directory is gitignored (`tools/**/build/`).
