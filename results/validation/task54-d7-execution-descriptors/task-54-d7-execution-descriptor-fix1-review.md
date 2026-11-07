# D7 execution descriptor SOURCE FIX1 — scoped independent re-review

2026-11-07. **SPEC PASS / QUALITY PASS for the scoped FIX1. Q1 is addressed.** No new Critical, Important or Minor finding was found in the changed material. This closes the prior negative-test-oracle finding at source-review level; it is not an observed test pass or runtime/scientific execution release.

## Scope and evidence

Read Task 54 brief, source release `cb753ddf`, original review `a48b6f32`, FIX1 report and the complete `source-validation/fix1/d7-execution-fix1-tests.diff`. Review is limited to Q1 and new breakage caused by this tests-only change. The previously reviewed encoder/source-view implementation and unrelated operational observer work were not reopened.

Read-only hashes checked here:

| Artifact | SHA256 |
|---|---|
| FIX1 report | `1ec91e4f36bd840d1b2ebade1716fbfb754c0f1bd6380661e065e068f7417210` |
| Preserved test snapshot before FIX1 | `e46f5ad1cabb7e0db464ca8dcb1a40c43c40e27ac1b698e667832eb8c1c63c53` |
| Current test source | `5212fced7ce53cf3a5f52c5da1ff400f1c99c403c5bdccecf810f4fc5f7523fe` |
| FIX1 final-path manifest | `34550bf7dc3854f167b7c67eb6a3a90f775844d8d42bdb9f6990ef139eaa7575` |

The test file's content outside `execution_complete_negative_images_distinguish_wire_and_binding` is unchanged, apart from the displayed blank-line separation. Root independently verified the 84 current path pins; that complete manifest check was not repeated. Original report/manifest/raw evidence remain historical rather than overwritten. Saved fresh no-run/clippy/format/diff-check metadata and raw logs show exit 0; they show compilation/static validation, not test execution.

## Q1 resolution

All references below are to `crates/ckks-train-bench/src/study_reference/document/execution_descriptor_tests.rs` at the current pinned hash.

- Lines 379-440 declare exact fixed-name arrays of **30 grammar refusals, nine parameter-invariant refusals and 17 semantic mismatches**. Independent text/fixture inspection confirmed 56 unique names and exact equality with the frozen fixture-name set. No fixture was reclassified or rewritten.
- Lines 449-469 require parser errors in the first two classes, with the nine parameter exceptions named explicitly and constrained to parameter kind. Archive expectation labels are checked. The old silent `else if let Ok` fallback is gone.
- Lines 470-477 require successful parsing for every semantic specimen and then require complete-byte mismatch against the unchanged synthetic retained expectation. Parser over-rejection now fails with the fixture name. These expectations remain explicitly synthetic bytes, not checked preparation/source ownership.
- Lines 478-484 independently exercise the three requested declaration predicates: ephemeral 0 and 17 fail at synthetic N=16, while changed weight 4 passes the range check and still fails identity comparison with original weight 3. This separates range validity from original-source equality without a parameter/P1 constructor.
- Lines 487-499 map each class to disjoint seen indices 0..29, 30..38 and 39..55, reject unknown/duplicate names, require every name present and assert exact `[30,9,17]` totals. Thus a missing, duplicated or unclassified case cannot silently pass the batch.

These changes satisfy all four required parts of Q1. They introduce no production encoding change, actual-source constructor, readiness capability or test execution. Current parser/source expectations remain the same; the assertions now enforce them.

## Work inventory and new-breakage assessment

The proposed successful-path counts are internally consistent with the changed test and previously reviewed helpers:

| Operation | Static inventory |
|---|---:|
| Complete negative parse | 56: 30 grammar Err + 9 parameter-invariant Err + 17 required Ok |
| Total `WireDescriptor::parse` entries | **6,419** = 14 positive + 14 positive matcher + 56 negative + 17 negative matcher + 6,314 proper prefixes + 4 unsupported-policy specimens |
| Total `matches_bytes` calls | **31** = 14 positive equalities + 17 required synthetic mismatches |
| Total declaration-range predicate calls | **11** = 4 existing direct + 3 added direct + 4 internal calls from registered-policy validation |
| Actual-source `matches_source` calls | **0** |

The three arrays contain static string slices; seen flags/counters are stack values. The three added range checks borrow the already decoded byte images and use the existing bounded error diagnostics. The change adds no descriptor output, geometry DTO, fixture, I/O, environment/process access, native/network work or cryptographic/planning constructor. The original ten exact test names, 14 descriptor outputs, 13 parameter pre-output refusals, 12 literal stage DTOs and six original-Transcript outputs remain unchanged. Crypto/production constructor count remains zero.

These are source-derived planned counts, not measured execution counters. The ten tests are still **compiled but unexecuted**. No new breakage was found in the tests-only delta; no additional fix is required for Q1.

## Preserved boundaries

This re-review used reads, text comparison, fixture-name inspection and hashes only. No Cargo/test/constructor/HE/native/network/Git execution, source mutation or subagent was used. Only this review file was written. The separately completed operational observer does not supply evidence for these tests or the D7 producer.

Actual retained CPU route/ExecutionBinding production, source-owner/context/Arc rechecks, additive original-request accounting, post-start binding workspace, live gates/durable admission/start, crypto-before-lease teardown, factory integration, native physical evidence and scientific metrics remain outside this first layer. Existing P1/P3 limitations, five outer gates, full encrypted-block user hold and the complete 21-profile/full 2680 + 20 + 7/quick 268 inventory remain unchanged. Root source acceptance and any later bounded pure runtime release remain separate actions.
