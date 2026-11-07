# P3 fix2 — isolated encoding omissions only

2026-11-07. Root accepts the remaining R3 scoped-review gaps: changing source/mean/kernel together cannot discriminate omission of any one field; delta remained constant, and None-to-Some velocity alone cannot discriminate omission of the present label's numeric payload. R1/R2/R4/Q1 and heterogeneous-target fixes are independently reported closed, with final report still forthcoming. This release is a narrow test correction; all full P3/P4/D7 requirements remain unchanged.

**Dispatch/ownership is deferred until P4 returns source/Cargo.** On explicit dispatch, original P3 author owns only `crates/ckks-train-bench/src/study_admission.rs`, and only the existing `p3_digest_golden_and_mutations` unit test (plus a private test helper if needed). No production function, encoder, parameter constructor API, schema, arithmetic, CKKS file, integration test or other author's source change. Preserve all fix1 reports/snapshots/logs/delta and archive copies unchanged.

Add independent one-field source/mean/delta/kernel mutations through actual `encode_rate_schedule_fields`, with every other field, reference and consumed prefix fixed. Include an independently specified fixed rate-vector byte expectation so each field's location/content has an oracle separate from merely hashing current output. Keep existing valid schedule/suffix/reference cases intact. Explicit borrowed test field values test encoding, not canonical numeric acceptance.

Reuse the existing accounting test's single `CkksParams::test_tiny()` and valid CtMeta. Through actual `encode_state_labels`, compare Some(label) with Some(changed_level) and with Some(changed_scale_bits) while option shape, lengths and all other state fields remain unchanged. Construct those labels through CtMeta::new using the same tiny params; no fabricated CtMeta/unsafe field mutation. The baseline and both variants remain Some, so dropping either numerical field must fail its assertion.

Reason for one tiny fixture: a fresh process cannot borrow a label from a prior process without reconstructing its parameter object. Factoring another production encoding view solely to advertise zero constructors adds unnecessary production changes. The already authorized accounting test provides a real validated label with one bounded small constructor. Actual fix2 allowance is exactly one existing tiny parameter construction, zero proof/descriptor/materializer/geometry/initial-state/update/handoff/wrapper/HE/key/Encoder/evaluator work. There is no rate/parameter/fixture search.

This is tests-only correction of currently correct encoders. Write the new assertions first; do not deliberately alter production code to manufacture a RED. Freeze source after static checks, then execute exactly one filtered GREEN below. An unexpected failure stops with original evidence; no automatic retry. Current target, CARGO_INCREMENTAL=0,CARGO_BUILD_JOBS=8,RAYON_NUM_THREADS=8,diskfree>=805306368B:

```
cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings
rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_admission.rs
cargo test --locked --offline -p ckks-train-bench --lib study_admission::tests::p3_digest_golden_and_mutations -- --test-threads=1 --nocapture
```

Compilation/lint does not execute other tests. No whole11+5 runtime repeat is needed because production and other test code remain byte-identical. Record the exact delta against fix1 frozen source and verify that claim. Pin all four P3 source files before/after, plus the current P4 three source files as unchanged dependency inputs if present; no extra source ownership follows. Preserve new artifacts in `task-54-p3-fix2-validation/` and report `task-54-p3-fix2-report.md`. Include exact command/exit/count/field discrimination and unchanged-production proof. Return source/Cargo for original reviewer's narrow rereview before integration.
