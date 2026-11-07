# Task54 D7 fixed CPU continuation — author report

Status: SOURCE IMPLEMENTED; COMPILE/STATIC GREEN; ZERO TEST RUNTIME EXECUTED. Not production readiness, runtime release, genuine start/admission, full Task54 completion, or scientific evidence. No commit/push by author.

## Requirements and source scope

Read task-54-brief.md FIRST, then task-54-d7-fixed-cpu-continuation-source-release.md and the read-only next-map. Binding architecture was docs/study-execution-design.md §§1,3,4,5 with study-runtime-design.md and study-target-design.md. Accepted source BASE024a14023a1b735fe6310ad1252dff712a459317. Root advanced docs-only HEAD to a84dab96e439f13e4a6a02cbf89102fba3729eac during this work; this author did not commit or push. No architecture conflict or unapproved redesign was necessary.

Only these source files changed:

- crates/ckks-train-bench/src/study_execution.rs
- crates/ckks-train-bench/src/study_execution/cpu_continuation.rs (NEW, currently untracked; explicitly included in source.diff and SHA256 pins)
- crates/ckks-train-bench/src/study_client.rs
- crates/ckks-train-bench/src/study_client/blocks.rs

No Cargo/dependency, math/P4/P5, CLI, gate, dispatcher, registration, fixture, threshold, public report or public ledger change. Existing unowned .vscode and the two results/mnist_blocks_*_plain_b48.json files were not edited. All author evidence/report stays in this private ledger directory.

## Implementation and authority boundary

CpuContinuation is a private fixed NON-authorizing intermediate owner. Single/block constructors consume an already retained PreparedCanonicalBundle. They never call legacy prepare_owned_canonical/prepare_block_owned_canonical, prepare_then_make, check_preparation or candidate preparation. They have no caller-hash constructor, public callback, plugin/permit abstraction, fake live gate/recheck, begin_checked_he, or StartedHeCase. Original checked geometry moves exactly once into the existing encrypt_initial/encrypt_block_initial body. Existing P4/P5 mathematics are unchanged.

The retained tail stores the SAME descriptor buffers and accounting value (moved, not reconstructed/reseeded), original materialized/registered/request borrows, exact canonical Arc<DiagonalPlan>, original actual parameter/P1/BootParams borrows, and a non-Send/non-Sync marker. Descriptors remain immutable and precede accounting in field drop order. These fields can move together into the eventual genuine LiveCaseFrame; there is no accounting lease, detached descriptor owner or permit.

The fixed client setup check is shared with the new pre-key refusal tests. It checks the actual parameter pointer, EXACT training rotation union/order rather than legacy subset acceptance, actual P1 descriptor/BootParams equality and retained BootstrapMetadata object pointer. The fixed Single rotation input borrows plan.keys().training_rotations(); Block also checks its retained candidate union equals that exact slice. No new rotation copy is made. Evaluation config comes directly from the retained checked geometry to the unchanged existing factory/Evaluator::with_config path. NI uses the actual existing owned Bootstrapper branch; I creates its refresher from the SAME StudyClient.

## Ownership/drop design

CpuContinuation declares setup before retained. An unused driver destroys client/evaluation/initial/block binding before retained descriptors/accounting. Setup functions first transfer the retained tail, then create crypto; setup errors destroy checked/input outputs and any local arrays/client/evaluation/initial/binding before returning through the still-live tail.

execute_cpu moves the tail into a local declared before setup. Each match arm owns its client/evaluation and initial/binding. An inner lexical scope creates same-client refresher, optional block facade, then session. The session dies first, then facade/refresher, then crypto locals, and finally the retained tail. No session/facade/refresher is returned or self-borrowed in the owner. Constructor failures and update failures use the same lexical cleanup; the original descriptor/accounting tail remains live through all of them. No unsafe, leak, self-reference or hidden reopened context is introduced.

The loop encrypts exactly the next canonical update through existing encrypt_update or encrypt_block_update and calls step sequentially. Successful update/sample progress is copied only from returned successful completion. Any failure returns immediately, preserves successful progress, poisoned state, actual inclusive counter delta and available failed-update prefix phase/update/counters. No retry, rollback, per-update/endpoint ciphertext snapshot or scientific JSON exists. Session-constructor errors also retain the actual counter delta from their entry.

## Requested-payload accounting

Original plan charges are seeded only by accepted phase2 preparation. No new seed or limit mutation appears in the continuation.

- Canonical initial array sizing borrows materialized arrays. It charges owned Array2/bias tuple-vector payload, every copied f64 weight/bias buffer and, for Block, the outer per-stage Vec payload. Checked arithmetic and per-auxiliary A_MAX=2,097,152-byte checks happen before copies. copy_canonical_layers uses checked allocation helpers and preserves logical shape/order. No allocate-to-measure path.
- The SAME original accounting reserves these arrays before key setup or array allocation. The array reservation stays live through initial encryption and releases only after arrays die.
- Block additionally sizes the actual binding description Vec, class Vec, every nested StageGeometry payload, activation Vec and transpose reuse Vec, by borrowed traversal. That auxiliary obeys A_MAX. Its reservation is added to the canonical array reservation BEFORE existing encrypt_block_initial clones descriptions/classes. After success, only binding bytes become retained; canonical arrays drop and release the temporary remainder. If encryption/retention fails, crypto/binding/arrays die before reservation cleanup and tail teardown.
- Rotation union, parameter/P1 bytes and plan are borrowed/shared; no duplicate rotation or full-plan/P1 transcript is allocated for fixed setup.
- Each loop's existing encrypt-update binding revalidation has its independently sized workspace reserved under that SAME accounting. Workspace sizing is allocation-free; actual revalidation runs inside the reservation, which releases before step. This does not replace the distinct required post-durable-start validation.

This remains application requested-payload accounting, not allocator capacity, total crypto/native allocations, RSS, resource admission, memory fitness or timing evidence. Existing encryption/trainer allocation behavior is unchanged. Two narrow clippy allows keep fixed setup owners and partial progress inline (large_enum_variant and result_large_err); boxing merely to satisfy size heuristics would introduce unrelated heap allocations and error-path allocation. Strict -D warnings passes with those documented local decisions.

## Tests-first and compile/static evidence

Raw logs, separate Cargo exit files, final source snapshots, complete source.diff (including the untracked child), hashes, HEAD/status, and machine-readable constructor inventory are under continuation-source-validation/.

| Evidence prefix | Command / purpose | Cargo/tool exit |
| --- | --- | --- |
|01-red|cargo test -p ckks-train-bench --lib --no-run; initial pre-key tests before helper implementation|101|
|02-red-corrected-tests|same command; corrected mistaken test-only BootstrapMetadata Clone/BootParams field assumptions; remaining failures are missing validate_fixed_setup|101|
|03-owner-tests-red|same command; retained-owner/reservation tests before child ownership/reservation bodies|101|
|04-compile|same command after implementation|0|
|05-format|cargo fmt --all|0|
|06-parallel-no-run|cargo test -p ckks-train-bench --lib --features parallel --no-run|0|
|07-cuda-check|cargo check -p ckks-train-bench --lib --features cuda|0|
|08-clippy|cargo clippy -p ckks-train-bench --all-targets --all-features -- -D warnings; inline size heuristic findings only|101|
|09-format|cargo fmt --all|0|
|10-default-no-run|cargo test -p ckks-train-bench --lib --no-run|0|
|11-clippy-all-features|cargo clippy -p ckks-train-bench --all-targets --all-features -- -D warnings|0|
|12-final-format|cargo fmt --all after final counter-error preservation/full work assertions|0|
|13-final-no-run|cargo test -p ckks-train-bench --lib --no-run on pinned final source|0|
|14-final-clippy-lib|cargo clippy -p ckks-train-bench --lib --all-features -- -D warnings on pinned final source|0|
|15-final-fmt-check|cargo fmt --all -- --check|0|
|16-final-diff-check|git diff --check|0|

Cargo invocations were compile/no-run/static ONLY. No unit test binary, HE, KeyGenerator, Encoder, session, native observer, server or benchmark was invoked. Feature compilation includes CUDA code compilation, with no CUDA runtime/context/native construction. No broad legacy runtime matrix was rerun.

The first RED included two avoidable test-only API mistakes; raw evidence is preserved rather than overwritten. Corrected RED02 isolates the intended missing-helper dependency. Owner RED03 also had missing module imports alongside missing bodies; final tests compile but remain unexecuted. These are compile-time TDD observations only, never runtime assertions of refusal correctness.

## Frozen proposed runtime filters and work/constructor inventory (NOT RELEASED / NOT EXECUTED)

Each exact filter below is selectable as:

`cargo test -p ckks-train-bench --lib <FULL_FILTER> -- --exact --nocapture`

Do not run a broad substring/full test suite. This report requests review of inventory, not execution authority. All counts below are source-derived expected work, conditional on successful execution; constructor instrumentation was not executed.

1. `study_client::fixed_continuation_tests::d7_fixed_prekey_exact_parameters_and_rotations`
   - CkksParams::n16_train twice, two distinct Arc owners with equal values; EvalConfig::default borrowed field values; no canonical materialization/planning/RNG.
   - Exactly6 calls to actual production validate_fixed_setup:1 positive,1 equal-value/different-parameter-owner refusal,4 rotation refusals (subset, superset, order, duplicate).
   - Zero keys/Encoder/Evaluator/Bootstrapper/encryption/session/refresher/training/snapshot/native observer.
2. `study_client::fixed_continuation_tests::d7_fixed_prekey_bootstrap_owner_and_parameters`
   - CkksParams::n16_boot once; BootParams::default_n16 once; DftSupportProof::verify_current once; BootstrapMetadata::new_with_verified_support twice, producing distinct structural objects with equal bytes; two BootParams clones and Arc clones only (no secret material).
   - Exactly4 calls to production validate_fixed_setup:1 positive,1 equal-byte/different-P1-owner refusal,1 missing-boot refusal,1 actual BootParams mutation refusal (ctos_levels decreased).
   - No canonical planner/materializer or RNG. Zero keys/Encoder/Evaluator/Bootstrapper/encryption/session/refresher/training/snapshot/native observer. P1 metadata construction is structural, not a Bootstrapper.
3. `study_execution::cpu_continuation::tests::d7_fixed_retained_association_and_copy_refusal`
   - Registry load1, registered-case inventory1, n16_train2 distinct owners, materialize exactly2 canonical Iris seed0 profiles (iris__emlp1 and iris__emlp2), CPU Main ClientRefresh, Quick2-update consumption, dedicated transpose, original optimized EvalConfig. No NI or secure-profile HE execution.
   - Exactly1 plan_diagonal and1 retained candidate preparation/descriptor collection per profile. Per S=1/2 original request PlanningUsage is fixed and asserted: geometries2S, initial_states2S, updates2S, handoffs2(S-1), wrapper_calls1. Combined: geometries6, initial_states6, metadata-updates6, metadata-handoffs2, wrappers2. There is no extra candidate preparation after retention.
   - Accounted validate_plan_binding during retained preparation:2 invocations total, each its existing borrowed sizing + actual validation. Original planner also performs its normal canonical binding traversal. Descriptor collection derives each retained bundle once. No post-start validation is claimed.
   - Asserts original input/case/request/parameter/Arc-plan associations, unchanged descriptor buffer address, unchanged retained accounting across moves and setup refusals.
   - Calls actual fixed make_retained_owned once and make_retained_blocks_owned once with equal-value/different parameter owner. Both must refuse before copy reservation and key construction. Four Rng::from_seed instances [41;32]…[44;32] are created, never consumed by crypto on these refusal branches. No entropy calls.
   - Two independent pure accounting probes per profile: original-needed copy amount minus1 refuses, exact amount succeeds; total4 reservation attempts /2 successes. Pure probes are separate accounting test values and do not mutate the canonical original request/plan. Their binding bytes transition to retained, array remainder releases on drop. Actual refused canonical owner remains unchanged and drops without crypto.
   - No canonical W/b copy allocation, binding clone, encrypted packet, session, snapshot or training. It sizes copies by borrow only. Zero KeyGenerator/Encoder/Evaluator/Bootstrapper/encryptor/decryptor/session/refresher/native observer expected. These zero-crypto conclusions need independent branch/source review before runtime release; successful setup/drop still has no runtime evidence.

No successful I/NI setup, unused encrypted driver destruction, numerical loop, poison injection, endpoint or performance test is proposed for immediate execution. Such tests require a new reviewed bounded crypto constructor/parameter/plan/work/memory inventory. Current tests cover pre-key refusal/association and pure drop-accounting release; static ownership/drop review covers successful crypto teardown paths until such release exists.

## Remaining exact dependencies

Production run_owned stays MissingExecutionDependency. LiveCaseFrame, actual five gate/resource-user capabilities/rechecks, ProcessStamp/context/input rechecks at setup/update boundaries, StartedHeCase and durable admission/start/store-handle transfer do not exist and are not synthesized here. Genuine begin_checked_he must move this SAME retained owner into the real frame, perform the separately accounted validate_plan_binding AFTER durable start and BEFORE keys, and recheck the live frame before crypto/update entry. The fixed compile-visible body is not that authority boundary.

Actual explicit immutable request/consumption/parameter/P1 wiring into the installed run_owned frame, runtime-policy producer from retained lease facts plus this closed I/NI body, outer ExecutionBinding producer with real implementation/source/physical/runtime owners, target/admission/start integration and scientific publisher remain separate source dependencies. Runtime-policy and outer binding were not expanded here. GPU/native/external/matched/packed/baseline adapters and full matrix remain unavailable as applicable. Full21/full2680+20+7/quick268 and encrypted block hold remain unchanged.

## Self-review and source pins

Checked no legacy reprepare call on either new constructor path; no math modifications; original parameter/P1/Arc-plan associations; exact union borrowed without copies; requested copy reservation before allocation; actual binding retention; setup/session/crypto before accounting drop; error prefix/counter preservation; no snapshots/retry/rollback/fake live checks; dispatcher unavailable; zero runtime commands. Final source pins:

- study_execution.rs a0fe49da5bff97cb617ec771b5c870edf399bb0fb78820cb6fc29e404f12b96a
- study_execution/cpu_continuation.rs f6695959663fd1fcff206d6e04f4092154d7ed68344bb9aa58339fcdfce82e67
- study_client.rs f043916ef3801c2c4c3bafa122f0d4dd5980a748f4f83926f792cd137ce4ec0d
- study_client/blocks.rs ce8dc1ef0d42c8c4c0832dd418059dadb8eba15470b6e9176ef0391169465f40

Independent SPEC/QUALITY review and separately bounded runtime release remain root decisions. If fixes follow, append fix-round evidence rather than replacing this record.
