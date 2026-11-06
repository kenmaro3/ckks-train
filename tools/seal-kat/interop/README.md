# SEAL 4.1.2 interop known-answer test

`kat.cpp` (Microsoft SEAL 4.1.2, MIT, via CMake FetchContent; SEAL is not vendored) and
`crates/ckks-train-ckks/tests/seal_kat.rs` check that ckks-train-ckks and SEAL read each other's
objects and agree on the scheme, for the `seal_2020` preset:
CKKS, N = 2^15, `CoeffModulus::Create(32768, {60, 40 x 9, 60})` (the last prime is the
special prime), `sec_level_type::tc128`, scale 2^40, `compr_mode_type::none`.

Run everything from the repository root:

```sh
tools/seal-kat/run_kat.sh
```

It builds this directory (Release), runs `seal_interop_kat generate kat/seal_2020`, then
`cargo test -p ckks-train-ckks --test seal_kat -- --ignored`, then
`seal_interop_kat verify kat/seal_2020` (exit 0 iff SEAL accepts and decrypts every hekit
file within its bound). `kat/` is gitignored (~80 MB). Each run uses fresh SEAL randomness
(SEAL's default PRNG); the inputs are fixed (`mt19937_64`, seed 20201225, uniform in
[-1, 1)). The offline build reuses `spikes/seal-2020/build/_deps/seal-src` when it exists.

## Files written by `generate`

| file | content |
| --- | --- |
| `secret_key.bin`, `public_key.bin` | SEAL's keys (key level: all 11 primes) |
| `ct_x.bin`, `ct_y.bin` | `encrypt_symmetric` of x, y (level 9, scale 2^40) |
| `ct_x_pk.bin` | public-key `encrypt` of x |
| `ct_x_plus_y.bin` | `add(ct_x, ct_y)` |
| `ct_x_times_y_3.bin` | `multiply(ct_x, ct_y)`: 3 components, scale 2^80, level 9 |
| `ct_x_times_y.bin` | `relinearize` of the above: 2 components, scale 2^80, level 9 (not rescaled) |
| `ct_x_times_y_rescaled.bin` | `rescale_to_next` of the above: level 8, scale 2^80 / q_9 |
| `ct_rot1.bin` | `rotate_vector(ct_x, 1)` with SEAL Galois keys for step 1 |
| `pt_*.bin` | SEAL's `Decryptor::decrypt` of each `ct_*` (NTT form) |
| `inputs.f64` | x then y, 2^14 raw little-endian doubles each |
| `values.json` | SEAL version, N, scale, `coeff_modulus`, `key_parms_id`, `data_parms_ids[l]`, per object metadata + SEAL's decoded values + SEAL's own max error, and x, y, x_plus_y, x_times_y, x_rot1 (`%.17g`) |

`verify` decrypts `hekit_ct_x_sym.bin` (level 9), `hekit_ct_x_sym_l3.bin` (level 3),
`hekit_ct_x_pk.bin` (hekit public-key encryption with SEAL's public key),
`hekit_ct_x_plus_y.bin`, `hekit_ct_x_times_y_3.bin` (3 components) and
`hekit_ct_x_times_y_rescaled.bin` (hekit's `rescale` of SEAL's `ct_x_times_y`).

## Bounds

Max absolute slot error against the exact values: 2^-25 for symmetric encryption, add, the
tensor product and relinearization at scale 2^80; 2^-23 for public-key encryption and
rescale, whose noise is the rounding `r_0 + r_1 s` (SEAL's uniform ternary key: median max
error 2^-24.8, so 2^-25 fails SEAL's own decryption about half the time); 2^-15 for SEAL's
rotation (SEAL's non-centered per-prime decomposition puts ~2^-17.5 of key-switching error
on slot 0). The derivations are in `bound_log2` in `crates/ckks-train-ckks/tests/seal_kat.rs`.
Independently of these bounds, hekit's decryption of every SEAL ciphertext must be
bit-identical to SEAL's, and hekit's `add` / `mul_no_relin` / `rescale` of SEAL
ciphertexts bit-identical to SEAL's `add` / `multiply` / `rescale_to_next`.
