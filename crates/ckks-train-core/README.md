# ckks-train-core

Part of **ckks-train**, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network
training (repository: <https://github.com/kenmaro3/ckks-train>). The workspace uses the
`ckks-train-*` crate names.

This crate holds the arithmetic under the scheme:

- `Modulus` / `Shoup`: word-size modular arithmetic with Barrett and Shoup reduction;
- `primes`: NTT-friendly prime generation;
- `NttTable`: the negacyclic NTT (SEAL 4.1.2's root and storage conventions);
- `RnsBasis`, `RnsPoly`: RNS polynomials, rescaling, `mod_down`;
- `BaseConverter`: fast base conversion for hybrid key switching;
- `sampler::Rng`: ChaCha20-based sampling (the only randomness source of the library).

Most users want `ckks-train-ckks` instead. Research code, not audited: see the security note in the
[repository README](https://github.com/kenmaro3/ckks-train#security).

License: MIT. Some formulas and conventions follow Microsoft SEAL 4.1.2 (MIT); see
`THIRD_PARTY_NOTICES`.
