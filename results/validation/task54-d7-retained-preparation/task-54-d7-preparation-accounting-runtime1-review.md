# Phase2 runtime1 partial outcome and runtime2 correction review

2026-11-07. **Partial evidence ACCEPTED; runtime1 batch NOT COMPLETE. Runtime2 correction SPEC/QUALITY PASS as an unreleased draft.** Runtime1 contains seven exact passing tests and one zero-selection call, not eight test passes or a completed fifteen-test batch. No source re-review, test rerun or execution authorization is supplied by this document.

`L/` denotes `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.

## Evidence identities and independent custody check

| Artifact under L/ | Independently verified SHA256 |
|---|---|
| `task-54-d7-preparation-accounting-runtime1-report.md` | `8bbf93e948a0847b7d95f873072010e3db0d4a1cb60f792931cbd088532750bd` |
| `task-54-d7-preparation-accounting-runtime1-release.json` | `bf182f3cb21a382772568e1f2ef9bc3308f5813b4c82d29c1d198724d76605d9` |
| `task-54-d7-preparation-accounting-runtime1-controller.py` | `7b9381f2041079370b43e3d8664cbc89f65116a5a5daa5449fcc6aea62f6dc88` |
| `task-54-d7-preparation-accounting-runtime2-draft.json` | `c5cd5c3dfe807cff609a104fc660dc53da45fa4bdfb9c7374744caaec1aa2ad1` |
| `task-54-d7-preparation-accounting-runtime2-controller.py` | `786b9fc70b2feec412236b8a6903efad791661649542c072edfa6ee5502b67ba` |

Independently joined all eight runtime1 per-call launch records, metadata and summary entries to their released names/argv, actual stdout selections/counts, and all sixteen raw stdout/stderr lengths and hashes. All joins match; raw streams total **2777 bytes**. Every Cargo call exited zero. Calls 1–7 each have the single expected named success and result `[1 passed, 0 failed, 0 ignored]`. Call 8 has no named success, result `[0,0,0]`, and 224 filtered tests. Its `expected_selected` is false. The saved summary contains exactly eight results and `complete:false`.

All 490 current source/input pins match both runtime1 release and runtime2 draft; their pin maps are identical. All release/draft input-document pins match their current files. Root's live observation of controller handle 89058 reaching terminal exit 1 is consistent with the saved failed-selection entry and the controller's stop branch. This reviewer did not independently resume or observe the live handle. Launch custody records PID 24094, the runtime1 release hash, no automatic retry and `CARGO_NET_OFFLINE=true`.

## Runtime1 outcome and orchestration finding

The seven accepted exact successes are:

1. `study_execution::tests::d7_accounting_pure_boundaries`
2. `data::tests::d7_sha_fallible_known_vectors`
3. `data::tests::d7_sha_output_reservation_boundary`
4. `study_admission::tests::d7_transcript_failed_prefix_and_counting`
5. `study_reference::tests::independent_minimal_plan_golden_bytes_and_sha256`
6. `study_reference::tests::every_plan_field_and_numeric_coefficient_has_its_declared_scope`
7. `data::tests::sha256_known_vectors`

**Important O1, historical orchestration defect:** runtime1 selected `data::tests::streaming_sha_independent_vectors_and_empty_updates`, but the actual test belongs to `data::streaming_hash_tests` (`crates/ckks-train-bench/src/data.rs:1403,1407`). Cargo exit zero for an empty selection cannot satisfy the released exact-one-test contract. The author candidate path and root release were wrong; the controller's exact-selection guard correctly stopped instead of treating this as a pass. This is not a failed SHA assertion or a reason to modify the reviewed source. O1 is corrected in the reviewed runtime2 draft; the corrected test remains unexecuted.

The selected seven bodies and the zero-selection call reach no CkksParams, P1/proof, canonical materialization, P2 planner, key, HE Encoder/evaluator/session or native/GPU path. This is a source-path interpretation of the exact selected tests, not instrumentation of every process allocation. Cargo/test-harness processes exist. None of the descriptor/P3/canonical tests later in the intended list was invoked by runtime1; no actual canonical preparation success follows from this partial batch.

## Runtime2 correction readiness

The reviewed draft has `released:false`, eight unique test names and `expected_test_count:8`. Its list consists exactly of the corrected streaming test followed by the seven remaining runtime1 names. It has no intersection with the seven accepted passes. Source/module inspection confirms all eight paths:

| Exact test | Source declaration |
|---|---|
| `data::streaming_hash_tests::streaming_sha_independent_vectors_and_empty_updates` | `data.rs:1403,1407` |
| `study_reference::document::execution_descriptor_tests::execution_parameter_literal_bytes` | `study_reference/document.rs:1985`; `execution_descriptor_tests.rs:362` |
| `study_reference::document::execution_descriptor_tests::execution_layout_literal_bytes` | same module; `execution_descriptor_tests.rs:366` |
| `study_reference::document::execution_descriptor_tests::execution_key_literal_bytes` | same module; `execution_descriptor_tests.rs:370` |
| `study_reference::document::execution_descriptor_tests::execution_security_literal_bytes` | same module; `execution_descriptor_tests.rs:374` |
| `study_reference::document::execution_descriptor_tests::execution_shared_p3_geometry_matches_frozen_layout_slices` | same module; `execution_descriptor_tests.rs:584` |
| `study_admission::tests::p3_fix1_structured_golden` | `study_admission.rs:2790,3922` |
| `study_execution::tests::d7_canonical_six_retained_sources` | `study_execution.rs:281,357` |

The separate runtime2 controller requires a released runtime2 document and the correct cwd; checks eight unique names/count; verifies input/source pins before execution and around each call; uses a new output directory and exclusive raw-stream creation; and executes one exact test per serial Cargo call with `--test-threads=1` and Cargo offline. It requires the expected named success plus exactly one passed/zero failed/zero ignored and stops on failure or wrong selection without retry (controller lines 7–47). It does not read the unreleased draft as authorization or overwrite runtime1 records. Future completion must be assessed from all per-call exit/selection results and terminal controller status, not from Cargo exit codes or the summary's loop-completion flag alone.

The correction is ready for a separate root release decision against these identities. It does not authorize rerunning the seven passes. No open correction-blocking finding remains in this bounded review.

## Pending constructor and work scope

The eight proposed tests still contain all of the originally pending construction work: two secure N=2^16 parameter owners, two tiny N=2^10 parameter objects, two real scalar DFT proof verifications, two P1 BootstrapMetadata owners, two registry loads, and three canonical resolve/materialize operations. The structured golden reaches its OnceLock fixture (`study_admission.rs:2804–2817`) in its own test process; it is not constructor-free and shares no fixture with the later canonical test process.

The corrected BootParams inventory is **14 scalar default constructions**: one retained canonical fixture, twelve temporary NI validation defaults, and one adjusted tiny fixture. The extra twelve create no heap/parameter/P1/HE object. The six canonical cases conditionally retain the source-derived 24 geometry / 24 initial / 24 update / 12 handoff / six plan-wrapper inventory, plus fifteen binding output and eight counting traversals including the named refusals. These remain conditional until those tests execute and complete; they are not newly measured counters. Full materialization remains separate from two-update Quick consumption. Zero domain key/HE Encoder/evaluator/session/training/GPU/network work is intended; real scalar proof arithmetic and metadata/planning work are explicitly included.

Source acceptance `ceddbe36` is unchanged. This review checked outcome custody, exact correction names/controller and pending inventory only. No compiler, test, constructor, planner, materialization, network, native/HE, Git operation, source edit or subagent was used. Read-only evidence/hash checks and this review file were the only work; this file is the sole write.

Runtime1 partial acceptance and runtime2 draft readiness do not complete phase2 runtime validation, native integration, phase3 ownership/continuation, gates/start or Task54. Existing holds, unresolved scientific decisions and full 21 / full 2680 + 20 + 7 / quick 268 remain unchanged.
