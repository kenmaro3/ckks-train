# D7 execution descriptor source — independent SPEC / QUALITY review

2026-11-07. **Overall: CHANGES REQUIRED. SPEC: implementation boundary is consistent with the released first layer; QUALITY: one Important test-oracle gap must be resolved before source acceptance/runtime-batch release.** Critical: 0. Important: 1. Minor: 0. No encoder, shared P3 byte-body or authority-construction defect was found by this static review. No tests were executed.

## Scope and inspected evidence

Read `task-54-brief.md` first, then source release `cb753ddf7527ffe578701e02d9344d6caf278beca1f443e35d2213d296241963`, the ratified execution design/accounting scope, full author report and all seven changed Rust paths/full diff. Fixtures were previously independently reviewed; no vector generator was rerun. This review covers four private encoders, allocation-free successful wire inspection, canonical preparation fact borrows, shared geometry writer, current Train manifest view and the ten compiled pure tests. The subsequent retained route/producer, shared additive accounting, actual-owner rechecks and gates/start are explicitly outside this release, not completed by it.

Read-only pin checks performed here:

| Evidence | Result |
|---|---|
| Author report, 23,374 bytes | SHA256 `df2f399e9b4dba847012bed92e33d0072483f928c7342db3b244dd4b69323b5c` matches |
| Full source-review diff, 73,001 bytes | SHA256 `7af6a19d0ddc9ae2edf71733ea0d94adf67414779aa3243277a6e40b8cdf9f6f` matches |
| Final path manifest | SHA256 `39c19a3b96064f4647ec8235bd85df52e64a8e822fda1dcbf3cb11750f4ee661` matches |
| Current seven changed Rust source files | All seven match author-report hashes |
| `d7-execution-static-pins.json` | All 108 current unchanged-binding paths match their recorded current and BASE hashes; zero mismatches |

Root separately checked the complete 70-path final manifest and exact 20 fixture copies. Those repetitive checks were not rerun. Current BASE/HEAD distinction is preserved: `7da6d0a` is the source release base; `56d0fb3` adds the root-owned operational request only.

The saved metadata/logs for final no-run, strict clippy, serial check and CUDA-feature check show exit 0; final formatting/diff-check metadata also show exit 0 (evidence 15–20). They establish compilation/static checks only. The source report retains tests-first RED01, integration02, helper RED03 and formatting13 failures; no runtime success is inferred. This reviewer did not invoke Cargo, the compiled test executable, constructors, native APIs, network, Git operations or subagents, and changed only this report.

## Important Q1 — semantic negatives silently pass on any parser error

**File:** `crates/ckks-train-bench/src/study_reference/document/execution_descriptor_tests.rs:378-394`, especially 390-393. **Relevant implementation:** `execution_descriptors.rs:620-653,654-699,776-799`.

The test branches explicitly on the 30 malformed-wire expectations. For every remaining specimen, however, it uses `else if let Ok(wire) = parsed` with no assertion when parsing returns `Err`. Consequently an accidental over-rejection of any source-binding specimen passes the test without reaching `matches_bytes`. For example, rejecting a reordered reuse pool or a changed-but-in-range ephemeral declaration during parsing would still make this test green. It therefore does not enforce the released distinction between wire grammar and retained-source mismatch, despite its name and the reported 30/26 inventory. Merely observing a changed complete byte array is also not a test of the security declaration validator.

This is a test-oracle defect, not a finding that current parsing/encoding is incorrect. The current source does appear to accept the intended nonparameter semantic images structurally and compare exact bytes, and constructor-derived parameter refusals are a deliberate documented distinction. The problem is that the test would not detect crossing that boundary.

**Required focused fix, without changing frozen vectors or adding constructors:** replace the permissive fallback with an explicit exhaustive expectation partition for this implementation:

1. The **30 frozen malformed-wire images** must return `Err` from parse.
2. The **nine constructor-invalid parameter images** must return `Err` from parse under the released parameter-invariant checks: `parameter-empty-chain`, `parameter-bit-vector-count`, `parameter-dnum-zero`, `parameter-dnum-not-divisor`, `parameter-logN-overflow`, `parameter-zero-scale`, `parameter-main-weight-zero`, `parameter-main-weight-overN`, and `parameter-swapped-bit-vectors`.
3. The remaining **17 nonparameter semantic/source images** must parse successfully under the current grammar and then fail the exact comparison to their unchanged synthetic retained byte expectation. There must be no silent parser-error success path and no unclassified fixture. Keep this explicitly synthetic; do not call it a `PreparationDescriptorFacts` or actual-owner test.
4. Exercise the separate declaration predicate for the existing security weight specimens: zero and over-N ephemeral weights must fail `validate_security_declarations(16)`; the changed in-range weight must pass that range check but still fail equality to the original retained expectation. These calls are pure borrowed inspections and need no new parameter/P1 constructor, live capability or evidence fixture.

Freeze/report the counts and any changed pure call inventory before the later runtime batch. A narrow tests-only fix and no-run/static validation under the existing source authorization is sufficient for this finding; no actual canonical preparation/HE run is requested. The pure runtime draft remains uninvoked pending review resolution.

## SPEC assessment of implementation

**Encoding and bounds.** The four domains and exact field orders match the accepted design. `execution_descriptors.rs:14-17,72-148` performs a borrowed sizing pass, rejects checked overflow/A_MAX before `try_reserve_exact`, writes through the existing document Encoder, and requires the second-pass length to match. No geometry/partition clone or hash is performed in sizing. U64/Text/Hash/Bool/Option encodings and signed four-byte rotation steps are preserved; P1's separate rotation representation is untouched. Private DTOs remain representation inputs, not checked preparation or readiness constructors.

Parameter validation at 194-246 checks the declared constructor ranges, chain lengths, dnum divisibility, exact scale value bits and weights without constructing parameters. Actual I projection uses a borrowed CkksParams; NI uses unchanged borrowed P1 bytes (491-499). Wire parsing at 620-636 and 713-900 bounds the entire input by A_MAX, uses checked count multiplication/conversion against remaining bytes before loops, checks tags/hash spellings and rejects trailing bytes. Nested minimum byte sizes are consistent with the grammar, including mask49, shape64, compact40 and minimum container49. Successful inspection holds slices/cursors only; error construction may allocate the existing bounded diagnostic string, as the report discloses.

**Shared P3 body.** The diff in `study_admission.rs:2025-2151` changes ownership/writer factoring only. Each original field and iteration order remains: layout triple, eight-value shapes, per-mask periods/steps/Bool, ordered nested transpose chains, rotations, compact plans/containers/biases. The generic writer's number/raw/Option/steps implementations preserve the old Transcript operations, byte widths and limits; no planner code changes. The test-only original-Transcript bridge checks six complete frozen geometry slices and creates no StageGeometry or parameter object (`execution_descriptor_tests.rs:476-503`). Runtime equality remains a pending assertion rather than an already observed test pass.

**Actual borrowed source.** `PreparationDescriptorFacts` has private fields and immutable getters (`study_client.rs:105-149`); canonical checked preparations provide the view, while cfg(test) synthetic owner variants return None. The added P1 reference is borrowed, and the block equivalent borrows the retained descriptions/plan (`study_client/blocks.rs:21-44`). No source-object constructor is exposed through the document DTO/parser. Existing canonical preparation still performs plan binding, secure candidate and manifest checks before constructing these owners; this layer has not executed that path.

`physical_layout_bytes` projects actual geometry and plan partitions and checks slots/config; `cpu_key_policy_bytes` checks normalized sorted training rotations and both directions of the geometry union, then exact P1 generated/execution roles (`execution_descriptors.rs:501-591`). Security projection compares actual main declarations and retains declarations-only assurance plus current registered declaration restrictions (593-608,654-699). `matches_source` requires this real borrowed view and compares complete recomputed bytes (645-652). The fixture-only `matches_bytes` comparison cannot supply such a view or create runtime authority. Context/Arc/source-owner freshness and factory exact-union enforcement still require the subsequent adapter and are not proved by these functions.

**Source manifest.** `study_provenance/context.rs:1293-1302` returns a borrow from the current receipt only after checking current StudyTrain role; it does not search historical Reference receipts, copy a new subset manifest or replace source approval. Existing `source_commit` remains reused. The getter is an immutable view, not a fresh live recheck; the later retained producer must keep the existing context/source checks as explicitly required by this release.

No `StartedHeCase`, public callback/readiness trait, runtime policy from selectors, new JSON envelope, scientific output or successful production dispatch was introduced. No dependency, scientific cap, parameter preset, experiment denominator or public runtime allocation boundary changed. The complete outer ExecutionBinding producer and additive preparation budget are honestly absent; local A_MAX checks do not substitute for those obligations.

## Pure batch inventory review

All ten nominated exact tests were traced through their helpers. Their crypto/production constructor count is **zero**: no CkksParams/preset/BootParams/BootstrapMetadata, candidate/geometry/state/update/handoff/full-plan planner, checked preparation/context, dataset, KeyGenerator, CKKS Encoder/Evaluator/Bootstrapper, session, native/GPU or network construction appears on these paths. `LiteralStage::new` only builds DTO structs from frozen JSON; document Encoder and Transcript are byte writers.

The current proposed workload matches source: 14 successful encoded descriptor outputs, 13 parameter pre-output refusals, 56 complete negative inputs, 6,314 proper prefixes, the exact numeric size/arithmetic cases, four unsupported declaration-policy specimens, and six original-Transcript geometry outputs. Positive layout encoding creates six literal stage DTOs; the P3 comparison creates six more; two-pass layout writing traverses 12 geometry bodies plus six Transcript comparisons. The fixtures are embedded with `include_str!`; no runtime file/dataset/environment/process access is present. JSON trees/clones, decoded hex Vecs, literal nested vectors, descriptor/output buffers, digests and bounded diagnostics are real fixture allocations and remain separate from HE/planning evidence.

Q1 modifies only the assertions/borrowed-inspection inventory within that pure scope. The ten tests are **compiled, not passed**. There is no need to execute an unrelated broad suite to review this source layer, and this report grants no execution authority.

## Preserved outstanding work

After Q1 is resolved, source acceptance can cover this representation/source-view layer only. Still required: actual retained pre-key CPU route and ExecutionBinding producer, same-owner/context/parameter/Arc rechecks, shared additive requested-payload accounting, post-start `validate_plan_binding` workspace, runtime-policy producer from the actual lease/route, exact factory rotations, live gates/resource/user ownership, durable admission/start, unused-driver/setup-failure drop order, native integration and scientific measurements/publication. P1 limitations, eight P3 unproved gates, five outer Gates and full encrypted-block user hold remain.

All 21 profiles, full 2680 family rows + 20 exact-Iris prospective rows + seven supplement entries and 268 quick projections remain the denominator. Unsupported CUDA/native/external/matched/packed routes stay visible. No scientific or full Task54 completion is asserted.
