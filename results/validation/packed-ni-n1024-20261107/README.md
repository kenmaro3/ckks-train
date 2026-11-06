# Packed NI/AA: finite N1024 validation, first attempt

The registered small fixture passed once on a local Mac on 2026-11-07. Independent outcome audit: **coverage, numerical result and evidence integrity PASS**. This is empirical correctness evidence for the stated fixture, not a secure-size training result, GPU measurement, full experiment matrix, production admission or performance benchmark.

Source: [`f53f34f0396e05ecae3d670fee21140454123349`](https://github.com/kenmaro3/ckks-train/commit/f53f34f0396e05ecae3d670fee21140454123349). The [fixed validation plan](../../../docs/packed-ni-validation-plan.md) and [implementation contract](../../../docs/packed-refresh-design.md) retain the original protocol and review-fix chronology. Source review closed the foreign-batch parameter binding and Host target key-structure findings before this attempt.

## Actual coverage

N=1024, full n16_boot Q/P chain, scale2^42, dnum8, log_slots9 and output level16; held-out key seed153. Square/SGD with bias and Poly/NAG without bias, each in two orientations and AU/WN/AA, give12 deterministic cases ×4 updates. Numerical coverage is Lazy only. These correlated cases use one deterministic key context, not12 independent seeds.

| Observed item | Result |
|---|---:|
| Validated / missing updates | 48 / 0 |
| Attempted / completed bootstrap calls | 212 / 212 |
| Registered call cap | 228 |
| Weight / bias / velocity / hidden events | 88 / 48 / 40 / 36 |
| Support projections | 176 |
| Saved parity diagnostics | 1,916 |

For each orientation, AU and WN perform4 calls per update for either mode. Square AA performs7,5,6,7; Poly AA performs3,5,2,7. Every observed update's seven operation-counter fields matches its frozen prediction. All nested bootstrap deltas are `(37,40,46,207,56,2,1)` in rotations/relins/ct_ct_tensor/ct_pt_mults/rescales/sse_switches/bootstraps order. They are already included in the update totals and must not be added twice. The N1024 inclusive key-switch cost is79 per call; do not substitute the N2^16 metadata figure.

## Recorded numerical gates

| Diagnostic | Largest printed value | Gate / interpretation | First raw line |
|---|---:|---:|---:|
| Full complex error | 2.53557821117368795e-8 | E=2^-12 | stdout577 |
| Replica spread | 3.06720765902612449e-8 | separate diagnostic; also below E | stdout807 |
| Incremental bootstrap error | 1.26730185407340126e-8 | epsilon_boot=2^-20 | stdout592 |
| Bootstrap input norm, overall (hidden) | 3.36142541618032420e-1 | hidden≤0.5 and universal≤1 | stdout2348 |

Every full-error/incremental gate and every role-specific input range passed, not only these global maxima. Separate replica/imaginary diagnostics are also below E; acceptance uses the full complex maximum. Weight/bias/U bounds are0.4/0.4/0.1. [The audit](audit-report.md) gives all maxima, role bounds, raw locations and count crosswalks. Replica spread is a separate metric and can exceed full error. Projection drift has no separate registered epsilon gate; pre/post projection parity is checked against E.

The program checks all512 complex slots, including padding, replicas and imaginary components. Saved logs contain aggregate maxima and selected worst-slot indices, not all decoded arrays/ciphertexts; the external audit validates all saved aggregates, but cannot independently redecrypt all slots. Some graph metadata is checked in the program without being serialized. Neither aggregate equality nor this audit alone proves ciphertext object identity; the source and separate operand-use control address AA reuse.

Diagnostics occur after synchronous prepare_update and before candidate commit/next update. Later calls in the same update may precede the first detected failure. This PASS is not evidence of immediate per-call range enforcement or universal approximation bounds. No retry, seed search, threshold change or calibration rerun occurred in this retained first attempt.

## Reproduction and artifact identities

Exact selected command:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_training_ni packed_ni_primary_12_cases_max_228_calls -- --ignored --exact --nocapture --test-threads=1
```

The primary is ignored by ordinary test discovery. Re-execution creates new evidence; it does not replace this attempt. Local runner time was00:19:19.919358–00:20:38.735245 JST, exit0. The harness reports78.24s. This interval is a test-run identity, not a server timing comparison.

- Executable SHA256: `9ee26bbbd7ea239b67d99e8de7d7c67aeb1ecf6764d65cf98a013f90c6c7eb5b` (frozen binary retained in the implementation ledger, not committed here).
- Original scoped code review SHA256: `36198aab83efa7b7ba46d661fe4c1426ba8e17b0a7181c9614699156a215fdc2`.
- Independent outcome report SHA256: `1eb345d689c811bc18c6aa55af794683ae31f36c2c2847a640a5a788aac06831`.

`stdout.txt`, `stderr.txt`, `runner.txt`, `source-SHA256SUMS` and `runner-program.txt` are byte-exact copies of the frozen attempt. Runner metadata is separate from binary diagnostics. `audit-report.md`, `audit-evidence.txt`, `audit-maxima.tsv` and `audit-reader.txt` are byte-exact copies of the independent audit artifacts. Their historical ledger-relative paths describe the original audit context; the reader/runner text archives are not standalone commands for this directory. `source-SHA256SUMS` identifies the13 files at the source commit above, including the documentation as it existed before this outcome appendix; it is not a manifest of later edited documentation. `SHA256SUMS` verifies all public payload files in this directory.

Remaining work includes c3 local blocks, full profile/backend binding, reference sessions and targets, the full registered matrix, secure-size/GPU execution and memory admission. This result does not resolve Task42 RQ1, Task47/48 GPU measurements or M4/M5 final gates.
