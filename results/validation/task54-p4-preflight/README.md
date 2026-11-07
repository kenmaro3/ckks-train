# Task54 P4 — pure numerical test sensitivity

This archive records **Mac synthetic test-fixture validation**, not encrypted training, scientific accuracy/performance or production admission. The cfg(test) fixture uses the existing D6b F64 Engine::train and forward. No CKKS parameters, keys, Encoder, evaluator, ciphertext, bootstrap or HE operations are constructed. A small source prerequisite moves an existing private reference-codec forwarding function before its tests without changing its body.

The original tiny initialization lacks sufficient sensitivity to incorrect training. The independently reviewed harness compares a fixed stronger Signal fixture with wrong loss/rate/denominator/tail/target/velocity controls at each control's first affected update, plus a separate first-update no-op weight witness in each trainable layer. Equivalent/inapplicable controls remain labeled. The matrix has eight baseline families,60 control rows and57 required witnesses per candidate/backend. It does not substitute for the P4 client/session implementation.

| Version and fixed cap | Original | Signal | Actual F64 update calls |
|---|---|---|---:|
| v1: I1e-5, NI5e-4 | FAIL | FAIL: five NI witnesses below8x |136|
| v2: I1e-5, NI4e-4 | FAIL | PASS: all57 witnesses/backend >=8x |136|

Both v1 failures remain immutable. Before any HE, root tightened the NI cap to4e-4 while retaining the same fixture, controls, scientific rates/presets and8x separation. The v1 minimum absolute distance0.00351562499999999445 supports this stricter cap. The separate v2 run measures minimum NI separation8.789062499999986. Its138 input records,1,760 state snapshots,276 range records,120 controls,136 actual call ordinals and every absolute witness distance are identical to v1. Only evidence version, NI cap/normalized values and resulting five Signal statuses change. Total actual pure training calls are272.

Both runs retain finite W/b/velocity and exact frozen F64 zero checks. Signal's baseline hidden activation maximum is0.3288487366949463, below the fixture prerequisite0.75. This plaintext range check does not prove encrypted/bootstrap precision or range. Later HE must independently meet the fixed all-entry cap and required control separation; a later failure is not permission to loosen the cap.

Strict Clippy and formatting pass on the three affected source paths; seven source/dependency hashes remain unchanged around the v2 run. The selected runtime command is:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate -- --test-threads=1 --nocapture
```

The outer v2 test passes because Signal is the chosen usable fixture; Original's failure is deliberately retained. This is not a claim that both candidates or any HE case passed. Source and numerical-scope independent reviews are in `v1-review.md` and `v2-review.md`.

`v1/` and `v2/` are exact copies of the corresponding implementation-ledger validation directories. Raw command/status logs are original operational test records; TSVs are explicitly derived NONSCIENTIFIC_TEST_EVIDENCE. The v2 extractor initially copied an `invocation08` comment from v1. Its corrected comment names actual `03-single-numerical.stdout.log`; every TSV non-comment byte and all raw/Rust/state evidence are unchanged. `v2/header-before/` preserves the initial manifest, extractor, four TSVs and original report. The artifact-only correction adds zero numerical calls; its independent custody closure is recorded in `header-closure-review.md`.

Original manifests/reports retain their historical ledger paths and hashes. Map `task-54-p4-pure-preflight-validation/` to `v1/` and `task-54-p4-pure-cap-revision-validation/` to `v2/`. The current v2 manifest's `../task-54-p4-pure-cap-header-correction-report.md` resolves to the exact top-level report copy here. Earlier direct current-file pins refer to retained before/frozen snapshots, not later worktree state. Raw evidence whitespace and failed statuses were not normalized.

P4 client/session/packet/poison/count/metadata behavior, the original P3 real pre-key factory acceptance, P5 blocks, D7 genuine gates/durable start, secure ranges/resources/security and full scientific campaigns remain required and unproved here. Full registered scope and encrypted-block holds remain unchanged. This archive grants no HE or campaign authority.
