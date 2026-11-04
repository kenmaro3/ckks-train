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
