# Task54 D7 Full admission/start codec runtime1 independent outcome review

This is a focused review of the already completed bounded run. It does not reopen source design or execute tests. Read the full runtime report, release, controller, launch records, all thirteen per-test metadata records, summary and raw result streams; independently checked their selections, arguments, counts, exits, sequencing, byte lengths and hashes. No Cargo, test rerun, source edit, HE/native work, SSH or subagent was used. Only this review was written.

## Verdicts

**SPEC: PASS for the released thirteen-test unit/metadata validation scope.** The recorded run matches exactly the nine accepted new codec filters plus four touched shared regressions from source review Q1. Default features, twelve library tests and one `registered_case` integration test were used. C1 remains explicitly required for the future live reader.

**QUALITY: PASS, with one low-severity wording clarification below.** All thirteen exact selections passed once in the recorded sequential run; no failure, ignored selection or retry is recorded. This is not a broad-suite, hardware, HE or live admission/start result.

**EVIDENCE: PASS.** The outcome report agrees with the independently inspected raw results, metadata, release and pins. Controller terminal exit0 is a root-observed tool-session fact, distinguished below from the directly inspectable child exits and files.

## Verified evidence

| Artifact | SHA256 |
|---|---|
| Runtime report | `5f4bcafb8b7b7660c4dbede4a92a03a56385ea13c8390ba42e2457c14d047f33` |
| Runtime release | `ccb922fb9a27e2c49f67a82519c37f78b197da1500a0136050bf70c5376fc347` |
| Runtime controller | `d6a2cff4d6dec1b74cbbe531187f40714f61cfce029aa528abf161df08bdc2f7` |
| Runtime summary | `2b0433dd15d4376d219b3b5c9724db675a9c096e9de421d1bc5c755d490ad38e` |
| Prior source review, unchanged | `042d0bcf2a6bb50dfb9380a55abc837561151ab43939640afa28dece66749659` |
| Complete source diff, unchanged | `76c088bb90f8e04910c33b1c4083dfa353fd4b06ff9f6c0c48e6b204aa5f30e1` |

Independently rehashed all 496 release source/input pins and all nine release evidence pins against current files: all match. The controller launch record pins the same release SHA256 and records PID83864, automatic_retry=false and `CARGO_NET_OFFLINE=true`. No release/evidence/raw file was rewritten by this review.

The summary is complete and contains thirteen unique ordered results. Every result equals its corresponding `.meta.json`; every `.launch.json` field matches the finalized metadata. Each exact test name appears once as a named success, with exactly one `running 1 test` result and counts `[1 passed, 0 failed, 0 ignored]`. Every Cargo child exit is0. All 26 stdout/stderr byte counts and SHA256 values match their metadata, totaling 5,247 bytes. Child start/end intervals are ordered sequentially without overlap.

Arguments match the release exactly: indices1–12 use `cargo test -p ckks-train-bench --lib <exact-name> -- --exact --test-threads=1`; index13 uses `--test registered_case` in place of `--lib`. There are no feature overrides, substring selections, suite substitutions or retry commands in the controller. Stderr identifies the library test executable for the first twelve and `tests/registered_case.rs` for the last. The recorded child PIDs are83865,84025,84028,84030,84070,84072,84074,84076,84078,84080,84082,84122,84124; these match the report.

Root reports observing the original tool session76127 terminal exit0, with launch chunk947c10 and final chunk61e759 containing `ALL_SELECTED_CHECKS_PASS`. This reviewer did not independently replay or access that tool session. The attribution is preserved as a root observation; directly inspected files establish every child exit and selected result, and the controller emits that terminal marker only after its final pin check. No parent-exit inference from PID disappearance or summary completeness is needed or asserted.

## Findings and limits

No blocking outcome/evidence finding.

**Q1 fulfilled:** the run adds exactly the required Reference canonical, Reference structural-path, Lock canonical and actual registered-case canonical regressions. It does not rerun the prior unrelated seventeen-filter package. The integration inventory correctly distinguishes its work: one StudyRegistry load reads the two pinned registry/lock metadata files, projects2,680 family plus20 exact-Iris identities and seven supplement descriptors, compares nineteen frozen canonical vectors and checks2,700 digest uniqueness. The first twelve test bodies have no runtime filesystem/registry work. This statement concerns the tests, not Cargo's build/file activity.

**Low-severity wording clarification:** `release.inventory.expected_zero` uses the bare term `Encoder` among zero-construction claims. It must mean the CKKS/cryptographic Encoder. The canonical **document Encoder is exercised** by these codec/regression tests, with ordinary text/Value/vector/hash allocation and serialization work. The release's detailed inventories and outcome report already describe that work correctly. Preserve the historical release hash; carry this explicit distinction in downstream summaries rather than claiming zero encoding or zero allocation. This wording does not undermine selection or results.

**C1 retained:** the future Retry reader must validate both current and predecessor admissions against the same retained target/reference and approved policy. Current-target verification followed only by `check_predecessor_claims` is insufficient. Changed-lock rejection and complete Start target/reference/live checks remain mandatory in that later package. Passing synthetic claims tests does not fill this gap or create a live token.

The zero parameter/P1/NTT/RNG/HE/native-constructor statements and input/allocation inventory remain source-derived reachability findings from the accepted source review; this runtime did not instrument those counters or measure allocator bounds/RSS. No actual approved context, target/capsule, resource/user lease, live gate, nofollow store/publication, durable start, Quick, HE/GPU execution, performance or scientific result was tested. Original Full holds/gates, C/R/A, all21 profiles, full2,680+20+seven supplement accounting and Quick268 remain unchanged and unfinished. Acceptance closes only this bounded runtime outcome review.
