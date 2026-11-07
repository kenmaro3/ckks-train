# D7 Train capture source release

RELEASED for one fresh sole source/Cargo implementer. This supersedes only the inactive-ownership paragraph in task-54-d7-train-capture-source-draft.md; the draft requirements below remain binding. P5 source/runtime/publication is completed at BASE. No active writer remains. This is a dependency of full Task54, not Task54 completion.

BASE: 9aba62f0a2a2b565466216511238dec3fd7a244f

Read task-54-brief.md first; then task-54-d7-train-capture-source-draft.md, task-54-d7-owned-path-api-map.md and named public design documents. Existing source is authoritative. Preserve the full matrix, distinct Reference-only authority and all unimplemented downstream gates. Implement actual production capture plumbing, not only synthetic fixtures.

## Concrete API integration

Add crate-private TrainCapture<'a> and RecheckedTrainCapture<'a> (names may follow local conventions) with private fields borrowing the exact VerifiedStudyTrainContext, RegisteredCase and MaterializedProfile. Expose capture/finish only through VerifiedStudyTrainContext. Expose a private fact view for producer use that derives workers/host from its retained CaseRuntimeLease, retaining runtime identity checks. Do not copy Host::execution_threads's Reference-only constant. Retain existing current-build and source verification through context checks. Capture the invocation and executable at start; require matching source/input/receipt/context/case/executor at finish.

Add document-level crate-private Case component producer from the rechecked type, returning the existing canonical bounded bytes and hash (and JSON inspection value only if useful internally). No conversion from parsed claims to captured authority. In run_owned, use the selected RegisteredCase's actual profile/seed to resolve and materialize once, capture/recheck/encode in the same owner frame, then return MissingExecutionDependency because descriptors/live gates/admission/start/science are still missing. Do not add an unused-only capture implementation.

## Source ownership

Only the following existing paths, plus focused new train_capture.rs / train_capture_tests.rs nested under study_provenance, or case_provenance_tests.rs nested under study_reference/document, are writable. Narrow module wiring is allowed; no new public authority constructor, generic callback, dependency, or P4/P5 changes. If genuinely necessary source is outside this boundary, report the precise dependency before editing it.

- crates/ckks-train-bench/src/study_runtime.rs
- crates/ckks-train-bench/src/study_runtime/tests.rs
- crates/ckks-train-bench/src/study_provenance/context.rs
- crates/ckks-train-bench/src/study_provenance/context_tests.rs
- crates/ckks-train-bench/src/study_provenance/mod.rs
- crates/ckks-train-bench/src/study_reference/document.rs
- crates/ckks-train-bench/src/study_reference/document/case_provenance.rs
- crates/ckks-train-bench/src/study_reference/document/producer.rs
- crates/ckks-train-bench/src/study_reference/document/tests.rs

## Tests and execution inventory

Tests first: add focused tests that exercise the missing actual APIs and capture/producer invariants. Record genuine compile RED via cargo test -p ckks-train-bench --lib --features parallel --no-run; then implement and record final no-run. Strict clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings; cargo check -p ckks-train-bench --lib --no-default-features; cargo check -p ckks-train-bench --lib --features parallel,cuda; targeted rustfmt are allowed. Capture each command and exit in the ledger validation directory, preserve failed commands too. No test execution yet; no keys, HE, training, remote SSH or production invocation. Existing runtime/Reference/Case tests must compile unchanged.

Design focused pure acceptance test(s) so root can execute once after independent source review: canonical Iris seeds 0 and 7 at most one materialization each, no learning; reuse immutable inputs for mismatch negatives. Codec/synthetic facts are explicitly labeled synthetic, not actual Linux verified context evidence. Inventory exact per-test materializations, executor creation and subprocess use in the report, plus exact selected names and relevant existing regression tests. Prefer no materialization for codec-only tests. Preserve exact existing Case/Reference golden bytes and compile-fail authority coverage. Actual positive Linux production integration remains unmeasured; Mac ProcessStamp must still fail.

## Report contract

Write task-54-d7-train-capture-source-report.md and task-54-d7-train-capture-source-validation/ evidence inside this plan ledger. Include changed paths/SHA256, genuine RED, final compile/static evidence, API flow, ownership/lifetime/role boundaries, exact proposed pure-runtime test inventory and residual production gaps. No commit/push: root packages independent review before commit. Do not spawn agents. Return a short status, report path and SHA256.

BASE e6011d502884777cadce46d7aac3117c18b47cdc20d7a97872994b5958222bd0 crates/ckks-train-bench/src/study_runtime.rs
BASE 0667c8efc79546af95c9124ddbe5bce1511348b3e4019890764100db870dfa68 crates/ckks-train-bench/src/study_runtime/tests.rs
BASE 1b1e9b8a7fce70df2d3eb380d08a30186034f4d1c116582053f93d025a71a381 crates/ckks-train-bench/src/study_provenance/context.rs
BASE 9452c6150cab27e3f3506d0cf0e40ae90b8a0a8c67afd9ce75a4b8b5507e2558 crates/ckks-train-bench/src/study_provenance/context_tests.rs
BASE ABSENT crates/ckks-train-bench/src/study_provenance/mod.rs
BASE eaa09efc927de5b67417ff91a8b5c5f8f71d6decd7d8e92859b57cde50d5477f crates/ckks-train-bench/src/study_reference/document.rs
BASE f5092f6c45e43848b57414e874983d1ec4107cbf1b616cbf151ec12c0a46e399 crates/ckks-train-bench/src/study_reference/document/case_provenance.rs
BASE ecbee634b7b9cfcc5e932d3d4a9baf7b6fefee23b20bfa4ca21ab48871f3e362 crates/ckks-train-bench/src/study_reference/document/producer.rs
BASE ABSENT crates/ckks-train-bench/src/study_reference/document/tests.rs

## Root ownership-path correction

2026-11-07T09:22:14.906932+00:00 The module is existing crates/ckks-train-bench/src/study_provenance.rs, not study_provenance/mod.rs. Narrow crate-private capture reexports in that .rs file are authorized; no public constructor or other scope change. Original released text above retained.
