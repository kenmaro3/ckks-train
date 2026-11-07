# Task54 P5 pure source review

2026-11-07. **SPEC: NEEDS_FIXES for S1, the exposed future oracle's missing shared phase budget. Pure sensitivity entrypoint: SPEC PASS at source level. QUALITY: PASS apart from that scoped guard boundary.** No other important/minor finding. Numerical sensitivity, hidden range and invariance remain unmeasured; this review grants no execution release.

Read the full source/review briefs, fixed P5 API map/design decisions, complete two-file diff/report, all new fixture source, unchanged real Engine/forward/gradient/parameter-update semantics needed for the controls, compile RED skeleton, final compile/static/custody records and source-derived inventory. Only reads/hash/text analysis and this review-file write occurred. No tests, Cargo, Engine/forward/HE/constructors, source edits, network, Git mutations or subagents.

## Frozen scope and static evidence

Independently verified:

- Source brief SHA256 `5552967d90cf3d1d4a50ffa66f5ae880024e22f4c03cc217fb6ad1141746f4e2`.
- Source report SHA256 `1504c197dd58fb183009ab8fc1ddb813255468bed1ada134ea3fed1a5dccedd7`.
- Complete owned diff SHA256 `d7b8d1cf8b0b22ba2c5f882513934fe4efa7be0b94a85a10a3851efb32c02ffb`.
- Current `study_plain.rs` SHA256 `fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a`; current `study_plain/p5_fixtures.rs` SHA256 `b9cdf2a685ddea16d054ff56b916f738e7110e9857e2c54f3075d08fb58efd00`.
- Parent file is byte-for-byte saved original (`d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b`) plus only the cfg(test) child declaration. Accepted P4 fixture still matches its preserved pin; no production arithmetic/API changed.
- `01-compile-red` exit101 genuinely reports missing `run_pure_sensitivity`; the retained skeleton matches its recorded before hash. This is compile RED, not numerical RED.
- Final `07-clippy` and `08-compile-final` exit0, clean raw stderr, exact scoped strict Clippy/focused no-run commands and required environment. Their before/after two-source inventories match each other and current reviewed pins. Recorded owned formatting uses skip_children=true. Cargo custody records free disk well above805306368 bytes and terminal process/tool handles.
- Compiled test binary and retained P4 fixture match `build-custody.json` hashes. The binary was read/hashed, not invoked. Compile/Clippy success establishes no numerical outcome.

## Required finding S1: future paired-oracle calls reset the only invocation budget

**Locations:** `crates/ckks-train-bench/src/study_plain/p5_fixtures.rs:218-233` and `534-545`.

The pure entrypoint correctly shares one Report across its entire156train/61Engine/179external-forward matrix. However, the exposed closed `block_oracle(family)` seam creates a fresh private `Report::default()` on every call. Its only before-call guard uses the pure LIMITS156/61/179, not a shared paired-oracle phase. It returns neither a shared budget nor aggregate counts to the future caller.

Thus repeated oracle calls are independently allowed up to the pure limits; the proposed next consumer cannot enforce the14Engine/40actual train/128external-forward aggregate at the actual hidden helper invocation sites. Counting whole oracle calls or reserving their planned train totals outside the function would not satisfy actual-call guard coverage. A wrong family/extra oracle invocation could exceed the paired phase budget without any of these local guards stopping it. No such invocation has occurred: this is a missing bounded consumer seam, not a current pure-matrix count failure or numerical defect.

**Required scoped correction, confirmed by root during review:** provide one closed cfg(test) paired-oracle phase with shared before-call counters for14Engine constructions,40actual Engine::train calls and128external forwards across14oracle calls. Keep family/recipe construction closed, use the same baseline/arrays/arithmetic and actual guarded sites, expose no caller-set caps/reset/Clone that defeats aggregate custody, and perform no extra numerical execution. Preserve the existing single pure Report and its156/61/179 matrix, all controls/thresholds and production/P4 sources unchanged. Root will separately release/review this narrow source fix before accepting the package/runtime. No HE implementation is authorized by this correction.

## Pure fixture and actual Engine semantics

The four immutable recipes exactly match the ratified A/B/C/D proposals: A two stages `[2,4,4]` C3/P4 and `[4,2,2]` C2/P2; B adds `[2,3,16]` C10/P16; C changes only the second head to Square; D uses the A graph at first-reference schedule. Hidden activations are existing SquarePlusLinear, other heads Identity, all biases frozen zero, actual NAG momentum.9, RSS delta1 and explicit reference.08. A/B rates/parts are .04/.08/.02/.08 with8+4/8/5/8+4; C .04 with8+4; D .08/.08 with8/8. Initial hidden/head weights are exactly.45/fan_in and.75/hidden_width. No search, initializer alternative, future suffix or rate/reference change exists.

Rows are constructed from global logical r across parts, restarting only at the next update. The specified two feature coordinates and stage-local label r%C_s are shared by baseline, read-only recipe and future oracle. The only row modifications are the explicitly named LostTail/MicroRowRestart mutants. Stage-local C_s/output-width/row-count checks precede every actual train path. Engine's private global class maximum is only its synthetic label-validation ceiling; targets still use the real stage output width and RSS delta1. No canonical heterogeneous-class axis is introduced.

True baseline uses one actual full-block Engine with StageSpec indices mapping each next input to the previous hidden representation. Inspection of unchanged Engine::train confirms it caches every stage's old input/forward before gradients or parameter updates. The fixture does not copy forward, derivative, gradient or NAG recurrence. Baseline diagnostic captures use existing `forward` on the same actual hidden layers, retaining all stage old-hidden arrays at every update entrance and final endpoint. They correspond to the inputs the full-block Engine itself caches.

The new closed cfg(test) recipe/oracle types expose immutable initial/state/label/rate/part/hidden views only. There is no caller-mutated Engine, general materializer constructor, canonical plan/readiness capability, secret/ciphertext or HE setup path. The same initial-array factory and baseline are reused by the future oracle, subject to S1's required phase-guard correction.

## Controls and discrimination checks

Local controls are actual scheduling mistakes with shape-compatible H=P at each selected boundary, not malformed-shape rejection:

- LocalHeadAsHidden captures the selected old head with actual forward and feeds that array to the next actual one-stage Engine.
- PostUpdateHidden computes selected hidden again after its own real Engine update, then feeds that updated array downstream.
- OmittedLocalUpdate skips only the selected boundary-source Engine train call and retains the old hidden handoff; it does not manufacture a zero gradient. Downstream stages still train.

Each local mutant owns persistent one-stage Engines outside its update loop, so real velocities survive across the schedule. Unselected boundaries use old hidden computed before that stage update. This preserves the intended correct local scheduling everywhere except the selected mutation. One-stage Engine snapshots are flattened back into canonical stage/layer order. The selected affected stage for witnesses is downstream of handoff substitution, or the omitted source stage itself.

ResetVelocity clears existing buffers while retaining NAG/current schedule; update0 is unchanged because initial buffers are already zero. ConstantReference changes actual current mean to.08 with velocities preserved. LostTail removes the later parts and their actual rows/labels and lets the unchanged real Engine use the retained valid-row denominator; it is not a fake denominator formula. MicroRowRestart retains all rows but restarts the feature/label index within each part. These controls are semantically distinct.

Status/first-affected logic is correct: ResetVelocity first affects1 for A/B/D, inapplicable C; ConstantReference first affects0 for A/B/C, equivalent D; tail/restart first affect0 for A/B/C, inapplicable D. Earlier snapshots before a first affected update are compared bit-identically across every stage W/b/velocity entry. Local and target controls first affect0. All40 rows are accounted for:15local +16schedule/status +9target, with36applicable/1equivalent/3inapplicable and72I/NI margin records intended.

Each of9 target controls changes only the selected stage's labels to legal class0, retaining other stage label bytes and the same initial/x arrays. It performs one actual full-block first update, requires every unselected stage's full W/b/velocity state to remain bit-identical, and requires a selected-stage W witness. This is the stated target-isolation evidence, not a claimed generic cross-stage-backprop mutant or copied derivative implementation.

Witness search uses all logical weight coordinates in the intended affected stage (all stages for schedule controls); dimensions are checked, selected raw baseline/control values/bits/location/difference are emitted, and each normalized margin must be finite and>=8 for I1e-5 and NI4e-4. The map's lesser2x proposal is not selected. Bias/velocity are inspected/emitted separately; frozen bias cannot provide a fake discriminating witness. No numeric margin or range pass is assumed by source inspection.

## Manifest, state and failure evidence

All four recipes' initial state, dimensions/classes/activations, row arrays/order, labels, parts, rates/reference/momentum and exact F64 bits are emitted before the first Engine. Baseline entrance/final hidden arrays are fully retained/emitted and checked finite/abs<=.75. All initial/post-mutant/post-baseline W/b/velocity dimensions and entries are inspected for finiteness and exact positive-zero frozen bias/bias-velocity.

Every paired comparison entry is emitted, including zero differences and unselected stages. Final structured control/margin/count/failure summaries retain failures; no constants/recipes are adapted on failure, no control is dropped automatically and no HE continuation exists. Arithmetic failures are recorded and produce missing-snapshot/count/final failures; before-call cap overflow stops before the excess operation. The ignored test's final assertions reject any reported failure. There is no claim that these prospective records have been produced yet.

## Independently derived actual-call inventory

There is one new Engine literal, one Engine::train call site and one external forward call site. Each reserves/logs immediately before its actual invocation. Existing Engine-internal forwards/gradients remain internal and are not double-counted. With S stages/U updates/B=S-1/K applicable schedule controls, the inspected loops imply:

- Baseline train U, Engines1, external forwards(U+1)S.
- Handoff mutants train2BUS, Engines2BS, external forwards2BU(S+1): each old hidden plus one selected old-head/post-hidden forward.
- Omission trainBU(S-1), EnginesBS, external forwardsBUS.
- Schedules trainKU, EnginesK, no external forwards.
- Targets trainS, EnginesS, no external forwards.

| Family | S/U/K | Train categories baseline/handoff/omit/schedule/target | Train | Engines | External forwards |
|---|---|---|---:|---:|---:|
| A |2/4/4 |4/16/4/16/2 |42 |13 |42 |
| B |3/4/4 |4/48/16/16/3 |87 |26 |103 |
| C |2/1/3 |1/4/1/3/2 |11 |12 |12 |
| D |2/2/1 |2/8/2/2/2 |16 |10 |22 |
| Total | |11/76/23/37/9 |156 |61 |179 |

The direct loops, guard placements and independent formulas match the source report/inventory; no count discrepancy or hidden extra pure Engine/forward was found. Per-family counts and final totals are asserted from actual reservations, not assigned after the fact. Counts are still source-derived, not observed executions. S1 concerns the separate future oracle's per-call counter reset, not this shared pure entrypoint.

## Disposition

The pure-entrypoint source, controls, frozen recipes,>=8x rule and156/61/179 guarded accounting are suitable for a later separately authorized pure sensitivity run. The full exposed package requires S1's narrow shared paired-oracle phase correction first, as root explicitly ruled; then freeze/review its delta and final compile/static pins. No other source change or extra experiment is requested.

Do not infer numerical sensitivity/range/invariance, paired HE parity, canonical materialization/D7/start, security/resource/performance, block campaign or Task54/P4/P5 completion from this review. No runtime is released here.
