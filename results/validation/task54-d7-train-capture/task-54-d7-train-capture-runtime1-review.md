# D7 Train capture runtime1 — independent outcome review

2026-11-07. **SPEC: PASS for the bounded local unit/type/codec outcome. QUALITY: PASS for the reviewed implementation within that scope; one controller accounting defect remains recorded. EVIDENCE: PASS. Controller orchestration outcome remains FAIL, distinct from test PASS.** No production Linux integration, HE acceptance or new invocation is released.

Read the runtime report, frozen release/controller, source pin inventory, proposed exact-test manifest, batch launch, all32 per-command launch/terminal metadata and all64 raw stdout/stderr streams, and summary. Reused the completed source review and its test/authority limitations. Independent checks used only file reads, hashes, text parsing and arithmetic, plus this review-file write. No Cargo/test, source edit, network, Git operation, constructor, learning/HE or rerun occurred.

## Custody and exact selection

| Artifact | Verified SHA256 |
|---|---|
| Runtime report | `539c3d77baeec3f44c2990e1dbcb6d850c5a8ddd90e8fcb9212a3c0fb6e8a7f9` |
| Runtime release | `3442d7199012d9eabfb8207e0275b53c6917ea62af513ae96738c1a49fbb5f2f` |
| Original controller | `3c6d03f7eccc745bb526d32c637c467d0e647c76824635cee2973a27c4475482` |
| Source review | `85ffa25bea6cab54ad28b36d7093691cde8b54302bebaf9b8bcc4457cddd934f` |
| Source report | `54c2183d01d39bb0b0daf663f3763be40387d19f91e812d911bf1b332cc98388` |
| Ten-source inventory | `be178dd20f29b7b37b84ad07d8e09c6f7f5236648d06f50b65b286f163f4bed3` |
| Exact proposed test manifest | `64f700519d56dbbf15f6fac780b76c67c05baade9725b9e88c252bcf2a2e48fc` |
| Batch launch | `f60aca8bb0ec70690eb5293d3a1e3a0c69566c23c4f6b158c7f1f5e2acb3eabb` |
| Original summary | `e0470ca6f6007cb254623863b7b4cf3dd01087311113eddb3bbb4b0ad4049310` |
| Invocation32 doctest stdout | `6871b170ca53fd08168676356a8d84a972f518711bfece4f9329f2d91f3b2f7f` |

All five release input hashes match their retained files. The ten current source files still match the released candidate. The controller checks these pins before and after every invocation; the final invocation's in-loop postcheck occurs before writing its metadata and triggering the accounting failure. The later final `ALL_SELECTED_CHECKS_PASS` path is not reached and is not claimed.

Batch launch binds the exact release, controller PID35405, start1794044445.41754 and automatic_retry=false. All32 numbered launch records are exact subsets of their terminal metadata; every summary entry equals its corresponding terminal metadata. All64 stream byte lengths/hashes independently match, totaling13,685raw bytes. Command timestamps are sequential, and all32 Cargo exits are0. Stderr contains ordinary Cargo build/runner output, with no warning/error diagnostics. Root reports original tool handle30296 terminated1; this is root-observed controller custody, not a global terminal field in the per-Cargo metadata.

Invocations01–31 exactly match the manifest's names, argv and order, each with `--lib --features parallel`, exact filtering and one test thread. They comprise5new/26regression parent tests. Each parent has one successful named outcome and a1passed/0failed/0ignored summary. Invocation09 additionally contains exactly one same-name isolated child run and its successful1/0/0 summary. The nested child is a direct test-binary invocation, not a33rd Cargo command. The two same-name occurrences and two summaries are consistent with the reviewed environment99 child branch and the parent's successful status assertion.

Invocation32 exactly matches the separate manifest doctest command, filtering `study_provenance::context::VerifiedStudyTrainContext` with parallel features and one test thread. No ignored test or unrelated numerical test is selected.

## Actual nine doctest outcomes

The unmodified stdout has nine distinct named successful outcomes on the expected context documentation:

| Source line | Example | Actual outcome |
|---:|---|---|
|170 |Borrowed artifact-context view |compile-pass example PASS |
|129 |Train context Send refusal |compile-fail example PASS |
|134 |Train context Sync refusal |compile-fail example PASS |
|140 |Reference context Send refusal |compile-fail example PASS |
|146 |Train public default construction refusal |compile-fail example PASS |
|152 |Private TrainCapture import/construction refusal |compile-fail example PASS |
|156 |Private RecheckedTrainCapture import/construction refusal |compile-fail example PASS |
|160 |Reference context Sync refusal |compile-fail example PASS |
|166 |Private dispatcher import refusal |compile-fail example PASS |

Rustdoc emitted two successful suites: **1passed/0failed/0ignored**, then **8passed/0failed/0ignored**. The actual selected total is9, with no duplicate named example. A passing compile-fail example means the documented program was rejected as required; it is not a failed Cargo test. Raw success establishes these examples' expected compilation dispositions, not a new approved runtime context.

## Preserved controller failure

**Minor operational tooling defect — `task-54-d7-train-capture-runtime1-controller.py:40–48`.** The controller records all suite summaries but evaluates selection using only `matches[-1]`. For invocation32 it compares the final `[8,0,0]` with expected `[9,0,0]`, ignoring the preceding successful `[1,0,0]` suite. It therefore writes the truthful raw summaries with `expected_selected=false`, updates summary with `complete=true`, and stops through SystemExit before the final success marker.

This exactly explains the retained controller FAIL without turning it into a successful controller run. `summary.complete=true` means all scheduled command records were written; it does not override `expected_selected=false` or the root-observed exit1. The actual9named results and both zero-failure suites independently resolve test selection, so the local test outcome is acceptable without rerunning or changing any test criterion. Original controller, metadata, summary and raw remain untouched.

Before this tooling is reused, selection accounting should handle multiple rustdoc suites and verify the expected named set, while treating the intentional isolated child separately from parent counts. That is a future tooling correction, not authorization to modify or rerun this retained batch.

## Scope of the passing evidence

The five new behavioral parents passed: canonical Iris0/7 case/input/owner predicates; distinct Train/Reference role predicates; actual borrowed serial projection; actual retained/foreign pool and owner projection checks; and explicitly synthetic Case producer-codec preservation. Passing invocation01 reaches its two reviewed materialization calls, once each for canonical iris__emlp1 seeds0/7. No additional materializing regression is selected.

The selected unchanged regressions pass the executor bounds/ownership/environment99/count-drift/refusal checks, Mac production ProcessStamp refusal, Case exact-golden/raw-negative/order/cap/Reference-shape tests, current/artifact/reader/source/receipt/process predicates, and the existing Reference canonical-domain/timing-vector test. The six positive and67raw-negative Case vectors are assertion-backed fixture coverage, not individually emitted live provenance observations.

The11dedicated pools,10HostExecutor objects,2explicit std threads and1subprocess are source-derived workload counts for the selected paths, consistent with the source review/manifest. They are not independently instrumented thread or process telemetry. The observed child and named successful executor tests support the stated finite behavior; they do not show an approved Linux retained runtime. Compile-only actual-API wiring and ten negative trait assertions are compiled source coverage, not additional executed parent tests.

Canonical Iris inputs are actual local materializations, but owner-pointer/source/receipt tests use shared predicates, snapshots and plain sentinels. The codec tests use synthetic facts. Neither those tests nor the private import doctests invoke real `capture_train` → `finish_train_capture` → captured producer with an approved current Linux Train context. Positive installed `run_owned` capture remains **UNMEASURED**, and Mac refusal remains intact. Selected source pins and local command custody are not an exhaustive transitive dependency/binary attestation.

The runtime report accurately distinguishes this local behavior/type/codec PASS from the controller aggregation FAIL and from missing production evidence. No rerun, source repair, altered JSON flag or tolerance/criteria change was needed for this review.

Full21profiles/full2680+20+7/quick268, distinct Reference-only publication, unsupported routes, descriptor/checked-route/live-gate/admission/start/scientific-output dependencies, secure-profile/GPU/full-campaign gates and existing holds remain unchanged. This review accepts only the bounded outcome above and grants no new execution or full Task54 completion.
