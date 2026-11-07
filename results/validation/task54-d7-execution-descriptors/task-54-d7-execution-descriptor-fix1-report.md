# Task 54 D7 execution descriptor FIX1 — Q1 test oracle

Status: DONE, tests-only fix ready for scoped independent re-review. Compilation/static checks pass; the ten pure tests remain UNEXECUTED and unreleased. No encoder/source-view change, actual-source constructor, HE/native/network call, commit, push or subagent occurred. Task54, the retained route/producer and all ownership/accounting/gate gaps remain incomplete as recorded in the original report.

## Finding and precise fix

Read the complete independent source review `task-54-d7-execution-descriptor-source-review.md`, SHA256 `a48b6f32df942dc28cd1c0ab3e72cf16fd18396bb60da1c20a1b73fcbfec3816`. Q1 correctly identifies that the prior `else if let Ok(wire)` silently accepts parser errors for every semantic specimen. The review found no production encoder defect. This fix changes only `crates/ckks-train-bench/src/study_reference/document/execution_descriptor_tests.rs`, specifically `execution_complete_negative_images_distinguish_wire_and_binding`.

Before any edit, the complete test file was copied byte-for-byte to `task-54-d7-execution-descriptor-tests-before-fix1.rs`, SHA256 `e46f5ad1cabb7e0db464ca8dcb1a40c43c40e27ac1b698e667832eb8c1c63c53`. The final test file SHA256 is `5212fced7ce53cf3a5f52c5da1ff400f1c99c403c5bdccecf810f4fc5f7523fe`. There is no new production implementation to write after this assertion fix, and no deliberate new compile RED or behavioral test run was performed; the original tests-first source RED evidence remains preserved. This phase authorizes only compile/static/no-run checks.

The test now uses three explicit frozen-name arrays, with fixed lengths30/9/17. Every fixture must match one named class. A stack-only56-element seen array rejects duplicate names and requires every named specimen to appear; three class counters must equal `[30,9,17]`. Unclassified images panic. Archive expectation labels are also checked against their corresponding grammar/semantic class. There is no silent parser-error path:

- All30 explicitly named grammar images require `parse` to return `Err`.
- All nine parameter-invariant specimens require `parse` to return `Err`: `parameter-empty-chain`, `parameter-bit-vector-count`, `parameter-dnum-zero`, `parameter-dnum-not-divisor`, `parameter-logN-overflow`, `parameter-zero-scale`, `parameter-main-weight-zero`, `parameter-main-weight-overN`, `parameter-swapped-bit-vectors`.
- The17 remaining explicitly named layout/key/security images use `unwrap_or_else` with the specimen name and diagnostic. Every one MUST parse successfully, then MUST fail complete-byte comparison with its unchanged original synthetic retained expectation. Accidental over-rejection now fails this test. No `PreparationDescriptorFacts`, actual owner, source-validity assertion or capability is supplied by these synthetic expectations.

The17 parse-success cases are exactly: `layout-valid-zero`, `layout-valid-over-lanes`, `layout-start-discontinuous`, `layout-ordered-chain-swap`, `keys-zero-rotation`, `keys-duplicate-rotation`, `keys-superset-not-equal`, `keys-generated-executed-swap`, `keys-bootstrap-absent-with-bundle`, `keys-reuse-pool-reorder`, `security-weight-zero`, `security-weight-overN`, `security-weight-change`, `security-limitations-order`, `security-limitations-count`, `security-NI-no-bootstrap`, `security-absent-not-dense`.

The separate pure declaration predicate is called on the three existing weight images, without any new vectors. `security-weight-zero` (ephemeral0) and `security-weight-overN` (ephemeral17) must return `Err` from `validate_security_declarations(16)`. `security-weight-change` (ephemeral4) must return `Ok` from that range check while still failing equality to the original `security_ni_fixed` image (ephemeral3). Thus range validity and original-byte identity are independently asserted.

## Preserved evidence and scope

The original source report remains byte-identical at SHA256 `df2f399e9b4dba847012bed92e33d0072483f928c7342db3b244dd4b69323b5c`. The original70-path final manifest remains byte-identical at SHA256 `39c19a3b96064f4647ec8235bd85df52e64a8e822fda1dcbf3cb11750f4ee661`; it is historical evidence and was not rewritten to hide this fix. Of those70 original recorded paths,69 still match exactly and the sole expected current mismatch is the owned test file. All original raw logs and pins are among the unchanged paths. The other six owned Rust paths, every descriptor/geometry implementation byte, the20 copied fixture files and original report/logs remain unchanged. Copied fixtures still equal the independently frozen archive, whose SUMS SHA256 remains `899b45991113581638a5758bd9bdaec441df768e771fa706a16549faf93b287e`.

Evidence was appended only under `source-validation/fix1/`; no prior report, manifest or raw log was overwritten. `d7-execution-fix1-static-pins.json` records all69 unchanged original paths, the sole changed path before/after hashes, review/snapshot/original report/original manifest identities, fixture equality and30/9/17 counts. `d7-execution-fix1-tests.diff` is the complete snapshot-to-current test diff. The separate `d7-execution-fix1-final-paths.sha256` pins the updated current path inventory and appended report/evidence; its own hash is returned separately to root to avoid a self-hash cycle.

No original unowned file was edited. Root's operational request and runtime draft belong root; this fix does not release or execute that draft. No production-code defect was discovered that required wider source ownership.

## Fresh compile/static evidence

All raw outputs and exact command/env/exit/elapsed JSON metadata are in `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/source-validation/fix1/`. Cargo used `CARGO_NET_OFFLINE=true`. All four commands returned exit0:

| Evidence prefix | Exact command |
|---|---|
|d7-execution-fix1-01-no-run|`cargo test -p ckks-train-bench --lib --features parallel --no-run`|
|d7-execution-fix1-02-clippy|`cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings`|
|d7-execution-fix1-03-format|`rustfmt --edition 2024 --check --config skip_children=true crates/ckks-train-bench/src/study_reference/document/execution_descriptor_tests.rs`|
|d7-execution-fix1-04-diff-check|`git diff --check`|

The no-run build emitted `target/debug/deps/ckks_train_bench-190377a24cc433d7`, which was not launched. Serial/CUDA library sources and features were not changed; their prior compile checks remain original evidence, not freshly rerun claims. No test result or security/ownership/scientific success is inferred from compilation. A read-only fixture-inspection command initially requested an absent optional `mutation` key and produced a Python KeyError; inspection was corrected to the actual `replacement_input` field. It did not alter fixtures or execute any Rust test or constructor.

## Frozen proposed pure workload after FIX1

The exact ten test names remain unchanged, all under `study_reference::document::execution_descriptor_tests::`:

1. `execution_parameter_literal_bytes`
2. `execution_layout_literal_bytes`
3. `execution_key_literal_bytes`
4. `execution_security_literal_bytes`
5. `execution_complete_negative_images_distinguish_wire_and_binding`
6. `execution_every_proper_prefix_rejected`
7. `execution_checked_size_and_arithmetic_guards`
8. `execution_synthetic_wire_has_no_registered_policy_authority`
9. `execution_parameter_constructor_invariants_without_construction`
10. `execution_shared_p3_geometry_matches_frozen_layout_slices`

Proposed successful-path call inventory, not observed execution:

| Work | Exact bounded inventory |
|---|---|
|Positive encoding|14 descriptors, unchanged frozen full bytes/digests|
|Parameter-invalid encoding|13 pre-output refusals, unchanged|
|Complete negative parse|56 calls:30 grammar Err,9 named parameter-invariant Err,17 required semantic Ok|
|Synthetic exact-byte matcher in negative test|17 REQUIRED calls, each Err; all17 retained original images are parsed successfully inside the matcher|
|Added declaration checks|3 direct calls:2 expected Err and1 expected Ok, as specified above|
|Declaration checks across all ten tests|7 direct calls (4 existing unsupported-policy specimens+3 added weight specimens);4 additional internal declaration-range calls from the existing registered-policy checks,11 total range-predicate invocations|
|All WireDescriptor::parse calls across the ten tests|6,419 expected entries:14 positive image parses+14 positive matcher expected-image parses+56 negative parses+17 negative matcher expected-image parses+6,314 proper prefixes+4 unsupported-policy specimen parses|
|All matches_bytes calls|31:14 positive equality successes+17 synthetic retained-image mismatches|
|Actual-source matches_source calls|0; no canonical preparation/source constructors are introduced|
|Shared geometry body|Unchanged12 traversals in two-pass layout encoders+6 original-Transcript regression traversals|
|Pure DTO construction|Unchanged12 literal stage DTOs across positive layout and original-Transcript tests;0 actual StageGeometry construction|
|Outputs|Unchanged14 descriptor-output Vecs+6 small original-Transcript Vecs;13 invalid parameter calls refuse before output allocation|

The class arrays contain only static string slices; the new seen array/counters stay on the stack. The additional three predicates borrow existing complete byte images/cursors and allocate only the existing bounded error-path diagnostic on the two expected refusals. No new fixture, multi-megabyte padding, vector generator, dataset, file/process/environment/native/network read or output was added to the test paths. All fixture inputs still use the existing `include_str!` files. Existing bounded JSON tree/clones, hex/literal vectors, output buffers, SHA256 strings and diagnostics remain fixture work and are not HE/planning evidence.

The exact six original-Transcript comparison slices stay unchanged: layout_i_single stage0[196,603); layout_ni_two stage0[196,845),stage1[877,1526); layout_i_three stage0[196,603),stage1[635,1042),stage2[1074,1481). Crypto/production constructor count remains0 across the whole proposed batch: no CkksParams/preset/BootParams/BootstrapMetadata, candidate/geometry/initial/update/handoff/full-plan planner, checked preparation/context, dataset materialization, KeyGenerator, CKKS Encoder/Evaluator/Bootstrapper or session. Document Encoder and Transcript are byte writers only.

Q1 is addressed in source assertions and compiled, pending scoped independent re-review and any later separately frozen pure runtime release. Actual retained CPU route/ExecutionBinding production, same-owner/context/Arc checks, additive original-request accounting, post-start plan-binding workspace, live gates/durable start/drop-order, factory integration, physical/native evidence and scientific metrics remain outstanding. This tests-only fix clears none of those boundaries and does not mark Task54 complete.
