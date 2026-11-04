# M0 spike results

Each task appends one `## Task N: ...` section below (edit only your own).

## Task 2: CPU NTT (Rust)

Crate: `spikes/ntt-cpu` (scalar Harvey NTT, Shoup precomputation, lazy reduction, no SIMD, single thread).

| item | value |
|---|---|
| machine | Apple M3 Pro (12 cores; bench uses 1 core) |
| rustc | 1.85.1 (4eb161250 2025-03-15) |
| build profile | `cargo bench` = release (`opt-level=3`, `lto=true`, `codegen-units=1`), default target-cpu |
| N | 65536 (2^16), negacyclic mod X^N+1 |
| q (60-bit, q ≡ 1 mod 2N) | 1152921504606584833 |
| forward NTT, median | **416.4 µs** (95% CI of median 416.2 – 416.5 µs; criterion mean CI 417.7 – 418.9 µs) |
| inverse NTT (incl. N^-1 scaling), median | 385.5 µs (95% CI 385.4 – 385.6 µs) |
| plan target band (1 Apple M-core) | 200 – 400 µs |
| inside band? | **No, slightly above (~4% over the upper edge)**; same order of magnitude, scalar code only |

Notes:
- Timing excludes the input copy (criterion `iter_batched_ref`, setup not timed). Two bench runs gave forward 423 µs and 417 µs (means of criterion's estimate), so run-to-run spread is ~1–2%.
- Twiddle/ordering convention is documented in the crate-level docs of `src/lib.rs` (forward: CT, natural in, bit-reversed out; inverse: GS, bit-reversed in, natural out).
- The SEAL 4.1 NTT on the same machine will be measured in Task 4 (SEAL is built there) to compute the SEAL ratio.
