# ckks-train-ckks

Part of **ckks-train**, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network
training (repository: <https://github.com/kenmaro3/ckks-train>). The workspace uses the
`ckks-train-*` crate names.

The RNS-CKKS scheme on `ckks-train-core`:

- `CkksParams` and its presets (`seal_2020`, `n16_train`, `n16_boot`; `test_tiny` for tests
  only), `Encoder` (bit-exact with SEAL 4.1.2), `Encryptor` / `Decryptor`, key generation;
- `Evaluator`: add, multiply, relinearize, rescale, rotations (hoisted), conjugation, with
  hybrid key switching and exact-scale arithmetic;
- `linalg`: diagonal-encoded matrices and baby-step giant-step matrix-vector products;
- `bootstrap::Bootstrapper`: bootstrapping with sparse-secret encapsulation (SSE),
  CoeffToSlot, a Chebyshev EvalMod and SlotToCoeff (17.46 bits of precision, 111 key switches
  per bootstrap on `n16_boot`);
- `HeOps` / `BootstrapOps`: the backend traits that `ckks-train-nn` and `ckks-train-cuda` build on;
- `OpCounter`: counts of every key switch, multiplication and rescale;
- `seal_compat`: SEAL's serialization format, for known-answer tests.

Feature `parallel`: rayon data parallelism (results are bit-identical with and without it).

Research code, not audited: see the security note in the
[repository README](https://github.com/kenmaro3/ckks-train#security).

License: MIT. Parts of the encoder are ported from Microsoft SEAL 4.1.2 (MIT); see
`THIRD_PARTY_NOTICES` for the list and SEAL's notice.
