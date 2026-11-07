# Phase2 runtime2 — independent bounded outcome review

2026-11-07. **SPEC PASS / QUALITY PASS / EVIDENCE PASS for the released eight-test runtime2 batch.** Combined with the accepted seven runtime1 successes, all fifteen distinct intended selected tests have now passed. No new Critical, Important or Minor finding in this outcome scope. Runtime1 remains a partial batch stopped on wrong selection; its controller failure is not reclassified as success.

This review accepts local unit/canonical-metadata validation evidence for the already-reviewed phase2 source. It does not establish native/HE integration, a production route, gates/start, resource fitness or scientific completion. Source commit/publication remains a separate root action.

## Custody and exact selection

`L/` denotes `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`. Independently verified full identities:

| Artifact under L/ | SHA256 |
|---|---|
| `task-54-d7-preparation-accounting-runtime2-report.md` | `cc1a5fa222a7ff11a03988d68186f5dbd72ee6d260203d7d9c4d326f27edb521` |
| `task-54-d7-preparation-accounting-runtime2-release.json` | `8aad58193c8d13d93d4120af66b739dd7b4335012b5e72194b2c1ae70216412c` |
| `task-54-d7-preparation-accounting-runtime2-controller.py` | `786b9fc70b2feec412236b8a6903efad791661649542c072edfa6ee5502b67ba` |

Joined every runtime2 release name and exact argv to its launch record, metadata, summary entry and raw stdout result. All eight Cargo exits are zero; all eight entries have the expected single named success, exactly one passed/zero failed/zero ignored, and `expected_selected:true`. All sixteen raw stdout/stderr byte lengths and SHA256 values match metadata, totaling **2927 bytes**. Summary is complete with exactly eight entries. All 490 current source/input pins and all release-input document pins match; the source pin map equals runtime1's reviewed map. No source identity changed between the accepted source review, partial batch and correction batch.

The launch records controller PID 29640, the exact runtime2 release hash, Cargo offline and no automatic retry. Every command uses one exact filter, `--test-threads=1`, the bench library and parallel feature. Root's live observation of original handle 20020 reaching terminal exit 0 and printing `ALL_SELECTED_CHECKS_PASS` agrees with the saved complete successful results. This reviewer did not independently observe or resume that live handle.

The eight verified runtime2 names are:

1. `data::streaming_hash_tests::streaming_sha_independent_vectors_and_empty_updates`
2. `study_reference::document::execution_descriptor_tests::execution_parameter_literal_bytes`
3. `study_reference::document::execution_descriptor_tests::execution_layout_literal_bytes`
4. `study_reference::document::execution_descriptor_tests::execution_key_literal_bytes`
5. `study_reference::document::execution_descriptor_tests::execution_security_literal_bytes`
6. `study_reference::document::execution_descriptor_tests::execution_shared_p3_geometry_matches_frozen_layout_slices`
7. `study_admission::tests::p3_fix1_structured_golden`
8. `study_execution::tests::d7_canonical_six_retained_sources`

Compared the successful-name sets with accepted runtime1 evidence/review `7350fffe`: they are disjoint and total fifteen unique names. Across both batches there are sixteen Cargo invocations with exit 0, including the one runtime1 empty selection. Runtime1 controller exit 1 and runtime2 controller exit 0 remain distinct. The wrong `data::tests` filter and original raw records remain preserved; only the corrected `data::streaming_hash_tests` test is credited as a success.

## What the successful assertions support

The canonical test's fixed three-stage-count by two-mode loop completed, including all six retained preparation/collector calls and the named refusal branches (`crates/ckks-train-bench/src/study_execution.rs:357–570`). It checks the same retained input/case/request/plan/parameter associations, zero remaining live temporary charge, descriptor size/output/source/digest comparisons, partition refusal, exhausted candidate-budget diagnostics, the legacy refusal callback count and exact original work-budget usage. The reviewed source retains original receipt limits and moved ownership; this is supported by source review plus the same-request/canonical-binding tests, not separate runtime telemetry of every byte reservation or a dedicated assertion of each limit field.

The four descriptor literal tests exercised fourteen complete independent examples with byte equality, parser/byte matching and frozen digest assertions. The geometry regression exercised the six stage bodies from the 1/2/3-stage fixtures. The structured P3 golden checked its exact 321-byte literal. The corrected streaming SHA test now ran its literal/boundary/large-input and chunk/empty-update assertions. Together with runtime1, the four new pure boundary tests and selected legacy plan/SHA regressions passed. This is meaningful coverage of the changed representation/accounting layer; it does not claim every decoder negative/prefix suite or every preparation branch was rerun.

The completed selected bodies entail three canonical materializations, two secure N=2^16 plus two tiny N=2^10 parameter objects, two real scalar DFT proof verifications, two P1 BootstrapMetadata owners and two registry loads. The corrected BootParams count is **14**: one retained canonical fixture + twelve temporary scalar defaults during NI validation + one tiny fixture default. The temporary defaults contain no heap/parameter/P1/HE ownership. The tiny fixture is separate from the canonical test's fixture because each exact Cargo invocation runs a separate test process.

The canonical test asserts work-budget usage for each completed case; summing those assertions gives 24 geometry, 24 initial, 24 update, 12 handoff and six wrapper calls. Fifteen output binding traversals and eight count traversals, including refusals, follow from the reviewed completed paths. Constructor and traversal totals are **source-derived inventories of successfully completed test bodies**, not independent per-call event logs or benchmark counters. Full materialization really precedes two-update Quick consumption, and scalar proof/metadata planning really occurred; neither should be relabelled pure byte-only work.

No domain keys, HE Encoder/evaluator/session, training, GPU, native observer, production context/start or scientific publisher occurs on these selected paths. That statement combines exact test selection with reviewed source paths; it is not an instrumented claim that Cargo/test-harness processes, ordinary allocations or compiled native support are absent. Raw test elapsed times are custody data, not training-performance evidence.

## Retained limits and review actions

Independent source review `ceddbe36` remains the source assessment. This outcome review did not reopen it. Existing negative/prefix decoder tests were not selected in these two batches; their older evidence is not a new rerun. Reuse/full-schedule coverage, physical allocation/RSS, HE numerics, security/precision/range/resource gates and native Linux integration are not established. Test-only comparison buffers remain outside the retained production receipt's requested-payload accounting.

Only report/release/raw evidence, relevant assertion scope and current pinned identities were read/compared. No test/Cargo command, parameter/proof/metadata constructor, planner, materialization, native/HE/network operation, Git command, source mutation or subagent was used by this reviewer. This review file is the only write.

Production remains MissingExecutionDependency. Explicit immutable request/consumption wiring, fixed phase3 CPU continuation and same-owner accounting through crypto teardown, actual outer/runtime bindings, live gates/durable admission/start and the scientific publisher remain unfinished. Existing holds, unresolved B/same-N/authors decisions and full 21 / full 2680 + 20 + 7 / quick 268 obligations are unchanged.
