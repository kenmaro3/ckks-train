# D7 execution descriptors — runtime1 independent outcome review

2026-11-07. **SPEC PASS / QUALITY PASS / EVIDENCE PASS within the released ten pure unit tests.** No Critical, Important, or Minor findings in this outcome-review scope. The accepted source candidate is initial review `a48b6f32` plus FIX1 review `a66c3aeb`; this verdict does not accept a production execution route, native integration, or scientific result.

## Reviewed identities and custody

The following complete SHA256 values were independently checked against the saved files:

| Artifact | SHA256 |
|---|---|
| `task-54-d7-execution-descriptor-runtime1-release.json` | `40807f6639a7de919368db705e21069c059ce213d78a11e72902091549c315e8` |
| `task-54-d7-execution-descriptor-runtime1-controller.py` | `7092b759ba1991038ce6b695d3c0170d672d640eeb20e193b3f55fc3f0017743` |
| `task-54-d7-execution-descriptor-runtime1-report.md` | `0262962b8df2362c00ac0d9f9251f7ea3a9c174be22c33acf9c5ff7bbd0a38c9` |

Read-only artifact inspection joined all ten release names, per-call launch records, metadata, summary entries, and all twenty raw stdout/stderr streams. Each raw length and SHA256 agrees with its metadata; the streams total 3856 bytes. All ten Cargo exit codes are zero, each stdout names exactly its released test once, and each reports one passed, zero failed, zero ignored. The summary contains exactly ten matching results and is complete. All 135 selected source pins and all seven release-input hashes match their current files. These are selected custody pins, not an exhaustive transitive attestation.

Every saved argv is `cargo test -p ckks-train-bench --lib --features parallel <full-name> -- --exact --test-threads=1`; the recorded environment override is `CARGO_NET_OFFLINE=true`. The controller checks the release and pins before launching, checks pins around each call, and uses a new output directory and exclusive raw-stream creation (controller lines 7–26). It waits for each child, verifies the exact named selection and counts, records stream hashes, and stops on failure without retry (lines 27–47). This supports one recorded ten-call batch against the accepted candidate. Root's live observation of handle 4861 reaching terminal exit 0 and printing `ALL_SELECTED_CHECKS_PASS` is consistent with the saved evidence; this reviewer did not independently observe or resume that live handle.

## Released results

All names below have the exact prefix `study_reference::document::execution_descriptor_tests::`. Each has one selected PASS with Cargo exit 0:

- `execution_parameter_literal_bytes`
- `execution_layout_literal_bytes`
- `execution_key_literal_bytes`
- `execution_security_literal_bytes`
- `execution_complete_negative_images_distinguish_wire_and_binding`
- `execution_every_proper_prefix_rejected`
- `execution_checked_size_and_arithmetic_guards`
- `execution_synthetic_wire_has_no_registered_policy_authority`
- `execution_parameter_constructor_invariants_without_construction`
- `execution_shared_p3_geometry_matches_frozen_layout_slices`

The executed test-source pin is `5212fced7ce53cf3a5f52c5da1ff400f1c99c403c5bdccecf810f4fc5f7523fe`, the reviewed FIX1 candidate. Thus Q1's corrected assertions were selected: 30 explicit grammar refusals, nine parameter-invariant refusals, 17 required parse-success/byte-mismatch cases, complete classification accounting, and the three added security-weight predicate checks. The initial permissive test is preserved as historical evidence and is not the executed source identity.

## Evidence interpretation and retained limits

The report's assertion inventory (lines 11–13) agrees with the reviewed selected test bodies: 14 complete positive byte/digest vectors, 56 classified negative images, 6314 proper prefixes, 13 parameter pre-output refusals, arithmetic/size guards, four unsupported synthetic security specimens, and six original-Transcript P3 geometry slices. Ten passing selected tests support successful completion of these assertions.

The 6419 parser entries, 31 `matches_bytes` calls, and 11 declaration-range predicate calls are **source-derived successful-path workload counts**, not instrumented runtime counters. The parser total is 14 positive parses + 14 parses through positive byte matching + 56 negative parses + 17 parses through negative byte matching + 6314 prefixes + four synthetic specimens. Byte matching is 14 + 17; declaration-range checking is four existing direct checks + three FIX1 direct checks + four indirect checks. The report labels this distinction correctly.

Actual-source `matches_source` invocation count is zero in these selected paths. Their literal DTOs, embedded fixture inputs, document encoders, wire inspection, hashing, and original Transcript geometry checks do not construct real parameters, P1/planner/checked preparation, contexts, datasets, or HE/native/GPU work. This is a source-path assessment coupled with the selected-test results, not separate runtime instrumentation of those absences. Cargo and Rust harness processes/threads exist and are not claimed absent. Fixture allocations do not measure cryptographic memory, RSS, or additive request-budget compliance.

Real canonical-source positive integration, an actual ExecutionBinding producer, retained CPU ownership/routes, live source rechecks and gates, durable start, post-start validation and drop order remain outside this release. Approved native Linux and full/scientific experiments remain unexecuted by this batch. Existing P1/P3/gate limitations and full encrypted-block holds remain; the full 21 / full 2680 + 20 + 7 / quick 268 obligations are not reduced. No whole Task54, M4, M5, or security-policy acceptance follows from these unit results. Observer 1952 is outside this review.

This reviewer performed only document/source-evidence reads and read-only artifact/hash joins, then wrote this review. No Cargo/test rerun, constructor, native/HE execution, network access, Git operation, or source change was performed.
