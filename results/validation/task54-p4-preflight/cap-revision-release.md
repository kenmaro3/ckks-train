# P4 pure fixture: stricter cap fixed before HE

2026-11-07. Root decision prepared from the actual first pure run and independently recomputed margins. Dispatch waits for root to read the completed independent review and verify its artifact. No HE validation is released here.

## Measured basis and decision

The original v1 run remains FAIL for Original and Signal under I1e-5/NI5e-4. It executed136 actual Engine::train calls and emitted every prescribed row. The independent reviewer recomputed all228 margin records from snapshots and reported no scoped source defect, exact counts/field values and all baseline range/finite prerequisites. Signal's smallest required first-affected absolute difference is0.00351562499999999445; dividing by the required separation8 gives an upper epsilon of approximately0.000439453125. The per-layer no-op minima are larger.

Ruling: retain the fixed Signal fixture, all controls, first-affected selection and8x separation, and freeze **NI epsilon4e-4**, with I1e-5 unchanged, before any HE operation. At this proposed cap the existing Signal observations imply minimum normalized distance8.789062499999986. This is a mathematical inference from the first raw run, not a new runtime PASS or evidence that HE achieves that accuracy. The sensitivity disposition explicitly allowed root to accept a smaller PRE-HE cap from measured margins.

Reason: a stricter HE error requirement makes the existing meaningful fixture discriminate all required controls without choosing new inputs/rates or weakening the8x criterion. Alternatives rejected: lower the separation requirement to accommodate5e-4, alter several fixture values without need, or choose a tolerance after seeing HE errors. Cost if the tighter cap is infeasible: later HE parity is unmet and must be reported/debugged; it cannot silently be loosened to convert a failure into success. Scientific presets/accuracy thresholds are unchanged.

The original log/source/report/manifest and both FAIL verdicts remain byte-identical. The new run is a separate versioned synthetic test criterion. Future HE remains subject to its own explicit implementation/review/release; if that work is released, every logical W/b entry must satisfy the frozen cap and the separately specified wrong-control separation.

## Exact limited source and validation

After explicit dispatch, original P4 author owns only `crates/ckks-train-bench/src/study_plain/p4_fixtures.rs`: change NI CAPS5e-4 to4e-4 and test evidence VERSION to `task54-p4-pure-v2`. No fixture/rate/control/Engine call/shape/range/threshold8/production source change. Preserve the three previous frozen source snapshots, all originals and derived tables. Prove the exact two-constant delta before execution. The study_plain declaration and reference helper relocation stay unchanged.

Use a new `task-54-p4-pure-cap-revision-validation/` directory and `task-54-p4-pure-cap-revision-report.md`. Record the prior RED numerical source/evidence as the existing v1 run, not an invented new RED. Freeze source after targeted strict bench lib/tests Clippy and rustfmt; then execute the same single `study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate` invocation once, test-threads1/nocapture. Capture exact stdout/stderr/terminal, current seven P3/P4 source hashes before/after, and source snapshots. No broad runtime tests/no-run loop is needed.

Additional actual allowance: at most68 Engine::train calls per fixed candidate,136 total, zero CKKS parameters/proofs/descriptors/Encoder/keys/evaluator/HE/materializer/planner work. Same2layers/input2/output<=16/rows<=12, unchanged candidate/control set. Prior136 plus this136 gives cumulative pure preflight272 calls. Re-extraction/read-only comparison of retained logs adds no train calls. No additional candidate or search; unexpected failure stops with original evidence.

Use current target only, CARGO_INCREMENTAL=0,CARGO_BUILD_JOBS=8,RAYON_NUM_THREADS=8,--locked --offline,diskfree>=805306368B. Commands:

```
cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings
rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate -- --test-threads=1 --nocapture
```

Compare all actual input bits, snapshot state bits, range values and absolute witness distances with the prior v1 run. They must remain identical; only VERSION/cap/normalized distance/gate status should change. Report Original and Signal separately, with exact minima and all failed rows. A new Signal pure gate PASS still proves no encrypted precision, production session/factory/D7 capability or full scientific acceptance. Return source/Cargo and exact two-constant delta/evidence for independent scoped rereview; no Git/remote/dependency or other source changes.
