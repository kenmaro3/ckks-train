# Task54 P5: actual CPU block bridge source release

Read this first. This is the bounded source implementation requirement for Task54's real multi-stage CPU path, following accepted P5 pure v2. Parent task text is `task-54-brief.md`. Detailed interfaces, closed recipes and prospective accounting are in `task-54-p5-api-map.md`; the explicit overrides below supersede its obsolete preflight proposals. Public design sections13–14 in docs/study-diagonal-admission-design.md bind the implementation. Full registered 21 profiles/full2680+20+7/quick268 and secure/full encrypted-block holds remain unchanged. This package is a needed implementation dependency, never a substitute for full Task54 acceptance.

## Source ownership and deliverable

Implement the actual multi-stage client / CPU evaluation / session route using existing BlockMlp, BlockTrainer and BlockNiTrainer, genuine P3 typed transitions and the existing StudyClient ownership. Cover all four homogeneous modes I/WN/AU/AA, two and three stages, per-stage labels and activations, rate/reference schedules and handoff semantics. Do not reduce the package to one smoke case, a stub, fake Ready/start or a duplicate optimizer. The actual canonical preparation adapter and checked private factory must exist, while genuine D7 owned admission/start remains disconnected and honestly unimplemented.

Sole source/Cargo owner, no subagents, no commit/stage/push. Root reviews and commits after evidence. Allowed files:
- NEW crates/ckks-train-bench/src/study_blocks.rs and study_blocks/p5_tests.rs
- NEW crates/ckks-train-bench/src/study_client/blocks.rs; parent study_client.rs narrow shared setup extraction and child declaration
- NEW crates/ckks-train-bench/src/study_admission/p5_fixtures.rs; parent study_admission.rs cfg(test) child declaration only
- study_diagonal.rs narrow reusable read-only stage helpers, actual bootstrap/refresh handle seams; preserve accepted P4 behavior
- lib.rs module declaration.
No NN numerical/source changes, no Cargo/dependency/preset/registry/runtime/D7 edits, no P4 fixture edits and no P5 pure source edits. If a required seam cannot fit these files, report exact issue before expanding. Archive/source snapshots and owned task evidence under this ledger may be written. Preserve .vscode and results/mnist_blocks_*_plain_b48.json and every prior archive.

## Existing seams and exact decisions

Use map sections1–3 for call graph and lifetime design. Separate closed private BlockPlanOwner and block packet, canonical Arc versus cfg(test) synthetic Arc. Do not generalize PlanOwner or allow synthetic promotion. Canonical adapter validates P3 binding, partitions, candidate and union key manifest, derives stage descriptions from actual StageGraph/layer ranges and uses plan initial metadata (not candidate placeholders). Reserve S extra geometry+initial calls in the original request before its hash; preserve typed budget errors. Keep original single-stage check rejecting block inputs in P4.

Shared setup: extract existing study_client::make_owned parameter-pointer/bootstrap identity/rotation validation and first KeyGenerator+make_material sequence into a private borrowed setup-fields helper. P4 adapter passes one description's existing fields; block adapter passes actual union fields. Preserve existing order and error meaning, no public callback or digest capability and no second secret-bearing client. Existing make_material P4 test hooks must continue to work: P5 may add a narrow private guard path for its actual constructor sites but must not silently bypass actual accounting or change P4 counts. No new generic authorization framework. Public already-created keys cannot establish pre-key rejection.

Shared state comparison: extract existing read-only single-stage I/NI checks from study_diagonal::check_state_without_progress into crate-private helpers accepting actual stage trainer/state/description, plus minimally expose check_mlp/check_batch/check_optimizer. Preserve exact comparison expressions/cardinality/Option/errors for P4; use helpers for every P5 stage. No copied derivative/optimizer/activation formula. Configured NI reference-factor helper is allowed existing implementation, not literal internal kappa/mask observation.

Caller owns base CpuEvaluationContext and sized block forwarding facade, so borrowed trainers never self-reference. Forward real bootstrap through the same actual Bootstrapper Arc. I client handles come from the same secret-bearing ClientRefreshService via a narrow crate-private seam. Add bench boundary recorder for I, no NN getter change. Bound retention and failure clearing explicitly; output clones are actual additional allocations. NI uses existing last_handoff_bootstraps; AA compares exact source activation ciphertext to handoff (all limbs/lengths/metadata) and counts zero extra conversion/bootstrap. NI destination handoff metadata comes from transition.micros.x (level16), not fresh BatchSpec.x (level15).

Packet construction gathers canonical D6a rows once in order and builds padded targets separately for each stage classes/padding. Client encrypts x0 once per micro and a target per stage; later inputs only encrypted old-hidden handoffs. All local heads update locally; never pass local-head or post-update-hidden representation. No plaintext in server session. Client-only diagnostics compare all stage W/b and all old-hidden handoffs to the true F64 oracle. fork diagnostics must leave training counters unchanged.

Before work, validate owner/plan/index/progress/partition/full ordered rows, micro and stage-target cardinality, all stage descriptions and actual initial/pre-state, parameters/bootstrap, CT contracts, current/reference/read-only coefficient validity and lookahead. Poison before actual setters/trainer call; call existing atomic wrapper setters and exactly one wrapper train_step_valid. Publish successful progress only after all typed stage/handoff outputs and inclusive seven counter sums match. Sum stage+handoff inclusive counts once; no invented per-stage measured windows. On partial failure retain actual counters and available traces, reject retry, preserve source committed state/downstream untouched state; no rollback claim.

Checked BlockRetentionReceipt must report roles/liveness/actual retained CT sizes separately from analytical estimates and unobserved temporary/bootstrap workspace. Include model/velocity/compact state, packet, current inputs/hidden/gradients/reference work, boundary outputs/AA clones/snapshots, key/bootstrap bundles. Do not label a count as RSS/security/resource admission or hide duplicate payloads. No secure memory gate follows from mini tests.

## Frozen pure prerequisite and future numerical test

P5 pure v2 accepted under I1e-5 and **NI3e-5** absolute caps with required sensitivity>=8x. This replaces map's obsolete4e-4/>2x proposal. Pure v1 FAIL and stricter PRE-HE decision preserved; no tolerance loosening or new recipe. Reuse accepted study_plain::p5_fixtures::{BlockFamily,BlockRecipe,PairedOraclePhase}; old free block_oracle helper does not exist. One shared PairedOraclePhase guards14oracles/14Engine/40train/128externalforward across the entire HE test. Do not rerun pure mutant sweeps. All-entry finite errors within caps and frozen F64 biases positive zero. Fixed families/arrays/rates/parts are obtained from accepted recipe getters; do not retype them into independent numerical initializer.

Use map section5 complete matrix unchanged:16sessions,42block attempts =40success+2intentional failures. A/B each four modes/four updates; C four modes/one update; D I/AU two updates; poison I/AU. Heterogeneous C/P and C finalSquare remain synthetic only. Private synthetic plan uses existing encode_rate_schedule_fields with explicitreference and compose, never canonical full-maximum policy or invented suffix. Exact source-derived prospective limits:
- Params2, keys2, proof1, descriptor1, Bootstrapper1; Encoders40 (2setup+36stage+2config-onlyI).
- Geometry36, initial36, update plans100, handoffs58,16plans/158transitions; no canonical mini-suite materialization.
- Actual real bootstrap helper <=361 (284state+34AAactivation+43new boundary); actual I client refresh <=357 (334model/velocity+23boundary). Before-call guards at actual sites, no warmup or hidden constructors.
- Paired F64 shared14/14/40/128; no failed-prefix neworacle.
- Logicalpackets42; first-input CT encryptions64; stage-target CT encryptions152; successful handoff outputs84 (I22/NI62); failure prefix separately recorded.
- Positive checked private factory spy plus invalid actual-binding tests before key factory; reuse already planned cases, no extra plan/constructor. Source report must count these exactly. No canonical full owned factory/start claim follows from synthetic spy.
- Block negatives24 (12each existing C I/AU): association,plan,index,total,partitiongap,zerovalid,missingmicro,missingtarget,extratarget,heterotargetswap,xlevel,yscale. Add2later-rate negatives A I/AU with initialized velocity,40replay/exhaustion,2poisonretry=68unchanged comparisons/46snapshots. Private association mutation only, not third key. Arbitrary metadata-equal target substitution is not claimed detectable.
- Atomic late-stage setter checks: config-onlyI2stage references .08/.04,current.04 rejects .06 before change; same .04 unchanged. NI reuse uninitialized existingA/AU laststage lowerreference then reject .06/restore. Three config snapshots/four comparisons; include2extraIEncoders. Test reference locking after real velocity without adding actual updates.
- 40combined client reporting endpoints,96stage model inspections and84handoff inspections. Capture all metadata/values/counters/progress/config needed for independent outcome review. No plaintext server shortcut.
PoisonI: source stage14realrefreshes+firstboundary1 then secondboundary refused=>15 actual; NI-AU source4stateboots+firstboundary1 then second refused=>5 actual. Count conversion work before refusal, failedglobalprogress0, earlier stage can commit; NI last_handoff outputs empty after failure.

The map's counts are prospective, not evidence or runtime authority. Confirm exact actual constructor/helper/diagnostic sites in source report. If impossible, surface conflict with exact derivation; no dropping cases, automatic cap increases or runtime attempts. Production limits/gates do not inherit synthetic numbers.

## Tests first; compile/static only

Write focused ignored test skeleton and actual missing bridge assertions first. Capture one genuine compile RED for missing API. Then implement. One selected future runtime entrypoint: study_blocks::p5_tests::p5_cpu_block_bridge_contract. No test/Engine/planner/HE numerical runtime in this source phase. Allowed build commands (existing target only):
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_blocks::p5_tests::p5_cpu_block_bridge_contract --no-run
same env cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
rustfmt --edition 2024 on owned files only, skip_children=true for roots when needed.
Before each Cargo invocation check free disk>=805306368bytes. Preserve command/start/end/exit/process handle/raw output/source-before-after under task-54-p5-cpu-source-validation; poll same handle to terminal, no redundant already-passing rerun. Build scripts ordinary compile work. No workspace/release/allfeatures, old runtime rerun, SSH/network/GPU/dataset, cache deletion, other-process inspection or outside-worktree writes.

## Report

Write task-54-p5-cpu-source-report.md. Include status DONE/DONE_WITH_CONCERNS/NEEDS_CONTEXT, actual interfaces and ownership, moved P4 hunks and preservation analysis, compile RED/no-run GREEN/static raw evidence and source pins, full future runtime inventory tied to actual guard sites, diagnostic/retention limitations and unresolved requirements. Do not claim runtime parity, HE3e-5, full P3/P5/D7/Task54/M4/M5 completion. No execution release from implementer. Root will independently review source and later decide exact single-test runtime release.
