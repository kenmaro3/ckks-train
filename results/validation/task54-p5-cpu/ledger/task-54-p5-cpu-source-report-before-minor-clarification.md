# Task54 P5 actual CPU block bridge — source-only report

Status: **DONE_WITH_CONCERNS**. The actual client/encrypted-packet/evaluation/session route and the complete fixed future CPU matrix are implemented and compile. This is source/build evidence only. No test binary, Engine, numerical planner, HE update, pure sweep, GPU/dataset/remote/network work or runtime release was performed. No stage/commit/push was performed.

Source BASE: `2ea00601c5d6f4c10a98f4e0ddb2405935c34022`. Root separately advanced HEAD with its requests-only commit; this implementer did not own that change. Source release SHA256: `a33a075f6c7e68b1f918917d60b403b4cb300f0531b1a8e504114d30b188be6e`. Poison-capture addendum SHA256: `448d8cd00c6477b3332060dc51727b7f673b4c1fd0fb768cf12ab8d669bee264`. Public design sections 13–15 and the extracted Task54 brief/API map were read. The release's I1e-5/NI3e-5/required pure sensitivity>=8x supersedes the map's obsolete numbers. Accepted P5 pure source is unchanged. Pure v1 FAIL and its pre-HE tightening decision remain unchanged.

## Actual route and ownership

- `study_blocks.rs`: distinct closed `BlockPlanOwner::{Canonical,Synthetic}` domains; immutable block initial/update packets; caller-owned sized `CpuBlockEvaluation`; actual `CpuBlockSession`; completion/failure evidence; checked retained-payload receipt. Canonical and synthetic Arcs cannot be promoted or converted into each other. P4's `PlanOwner` is unchanged.
- `study_client/blocks.rs`: child of the existing `StudyClient`, with no second secret-bearing client. `prepare_blocks_then_make` checks P3 binding, actual parameters/partition, candidate geometries and the full candidate/plan union manifest before its private terminal factory. Descriptions and canonical array ranges come from actual StageGraph layer ranges and representation roles; initial labels come from the plan, never candidate placeholder states. A separate immutable `BlockEncryptionBinding` retains descriptions/layout/class information without adding a new field or cache to StudyClient.
- Actual public client endpoints are `prepare_block_initial`, `encrypt_block_update`, and `inspect_block_snapshot`; the initial endpoint returns the packet and immutable encryption binding. The private `prepare_block_owned_canonical` continuation really validates, calls the checked shared owned factory, and encrypts exact canonical arrays. It is compile-visible and disconnected from D7/dispatch/start. It grants no Ready/admission authority.
- The canonical adapter uses the original request throughout, without changing or replacing its budget after binding. The caller must reserve S additional geometry and S additional initial calls in that original request **before** creating its hashed plan. Existing typed binding/candidate/budget errors are retained. No canonical materializer/planner is called in the mini matrix.
- Canonical updates gather the actual D6a ordered rows once; the existing padded_targets helper runs separately per stage C/P. Initial x0 is encrypted once per micro, each stage gets one independent y, and subsequent stage inputs come only from actual encrypted old-hidden handoffs. No plaintext row/model/Decryptor is present in CpuBlockSession.
- The caller owns the base `CpuEvaluationContext`, then the facade, then the borrowing session. The facade forwards NI boots through the base context's same actual Bootstrapper Arc. All I stages and boundaries borrow the same actual client-owned refresh service through its narrow handle. A facade admits one live session and clears I retained outputs at the next accepted attempt or session drop; permanent failure prevents retry. There are no self-referential trainers and no public arbitrary-bootstrap injection.
- Every stage uses the existing NN Trainer/NonInteractiveTrainer; exactly one existing BlockTrainer/BlockNiTrainer `train_step_valid` is called per accepted logical attempt. Every local head updates locally. No optimizer, gradient, activation derivative, loss fold, or recurrence was reimplemented.

## Validation and failure semantics

Before setters/work, the route checks association, exact Arc plan, logical index/exhaustion/progress, full ordered MicroParts and denominator, micro/target cardinalities, stage C/P/shapes/activations, all actual initial/pre-state ciphertext metadata and optimizer state, parameter/key/bootstrap binding, common lookahead, every stage's read-only rate validation, fixed reference and finite actual/receipt coefficients. Configured folded-factor and branch equality is then checked through the existing optimizer view/helper immediately after atomic wrapper setters and before its one trainer call.

Poison is installed before any setter or trainer call. Successful global progress is published only after every typed post-state, hidden trace, canonical activation, handoff/destination-input contract and inclusive seven-counter sum passes. Counts are summed once over all stage and boundary receipts and compared with one measured whole-block delta. No separately measured stage windows are invented.

I boundary outputs are recorded by the bench Refresher, retaining one explicit additional ciphertext clone per successful boundary output. NI uses existing `last_handoff_bootstraps`. Destination incoming metadata is checked against transition `micros.x` (NI level16), not fresh batch x (level15). AA compares every source activation and returned handoff for exact components, full lengths, level, scale bits, basis/form and all limbs, and requires zero additional handoff conversion/bootstrap work.

Midwork failure retains actual partial counters, available actual trainer traces, a separate per-stage trace-changed flag, and I boundary-prefix retention count. Equal traces do not prove whether a stage ran. Earlier stage commits and downstream untouched state are explicitly inspected in the two fixed poison cases; no rollback is claimed. NI's failed wrapper publishes no handoff outputs. A postcheck failure also remains poisoned even if all underlying stages committed.

## P4 preservation and moved hunks

The shared factory change is limited to the existing `study_client::make_owned` checks and first KeyGenerator/make_material sequence. `SetupFields` borrows actual parameter pointer, bootstrap descriptor identity/BootParams, expected rotation slice and eval config. P4 passes its one existing description's fields; P5 passes actual union fields. Pointer check -> bootstrap identity check -> rotation containment -> existing P4 budget hook -> KeyGenerator -> make_material order and error strings are preserved. Existing make_material P4 Encoder/Bootstrapper hooks remain; additional P5-local thread-local hooks precede those same actual constructor sites and are inactive when the P5 matrix is not active. P4 counts were not rewritten or bypassed.

`study_diagonal::check_state_without_progress` delegates to extracted `check_interactive_stage` / generic-sized-bootstrap `check_ni_stage`. Their read-only comparison expressions, cardinalities, Option matching and errors are identical to the original P4 arms. Existing check_mlp/check_batch/check_optimizer bodies are unchanged and minimally crate-visible. Description gains Clone for immutable retained descriptions. The only refresh seam is a crate-private borrowed handle accessor. Actual CpuBootstrap forwarding and P4 failure hooks are unchanged.

`static-source-audit.json` verifies normalized expression equality for both moved arms; body equality for check_mlp/check_batch/check_optimizer/check_ct/validate_initial_packet; unchanged P4 check_shape/from_owner/validate_packet/step/check_coefficients/check_hidden_trace; byte-identical P4 fixtures, accepted P5 pure fixtures, study_plain root, NN block source and Cargo files against BASE. It is a static Python/text audit, not a Rust test or runtime parity claim. P4's original single-stage rejection remains unchanged.

## Exact future matrix, unexecuted

One ignored entrypoint: `study_blocks::p5_tests::p5_cpu_block_bridge_contract`.

| Cases | Sessions | Successful logical steps | Planned stage updates | Planned handoffs |
|---|---:|---:|---:|---:|
| A, I/WN/AU/AA, four updates | 4 | 16 | 32 | 16 |
| B, I/WN/AU/AA, four updates | 4 | 16 | 48 | 32 |
| C, I/WN/AU/AA, one update | 4 | 4 | 8 | 4 |
| D, I/AU, two updates | 2 | 4 | 8 | 4 |
| Poison first A prefix, I/AU | 2 | 0; two intentional failures | 4 | 2 |
| Total | 16 | 40 success + 2 failed attempts | 100 | 58 |

The 36 stage geometries and initial-state calls, 16 compose calls, 100 update plans and 58 handoff plans produce 158 typed transition records. The driver uses accepted BlockFamily/BlockRecipe getters for all arrays, specs, classes, rates, references, rows and partitions; poison takes the first A prefix without another numerical recipe or oracle. Synthetic rates use the existing `encode_rate_schedule_fields` with explicit .08 reference and the existing P3 compose. Canonical full-maximum policy is unchanged. C's heterogeneous C/P and final Square remain synthetic only.

A single PairedOraclePhase owns all14 actual F64 oracles/14 Engine objects/40 full-block train calls/128 explicit external forward calls. No failed-prefix new oracle and no pure mutant sweep occur. Every logical padded W/b and all old-hidden handoff entries must be finite and within I1e-5/NI3e-5; every F64 frozen bias must have positive-zero bits. All40 successful endpoints are combined client reporting endpoints: 96 delegated stage model inspections and84 handoff ciphertext inspections, explicitly asserted as I22/NI62. Diagnostics use existing client inspection plus unpacking of explicitly exported ciphertexts and require unchanged training counters. The test emits actual values, errors/caps, CT metadata, complete typed transition metadata, configs, progress, traces, whole-step counts and retention roles.

24 prework block negatives are the 12 named mutations on each existing C I/AU packet: association, plan, index, total, partition gap, zero-valid, missing micro, missing target, extra target, heterogeneous target swap, xlevel, yscale. Two A I/AU later-rate negatives occur with actual velocity initialized, before the next rate setter can change state.40 success replay/exhaustion rejects and2 poison retries complete68 unchanged comparisons. Wrong-owner is private association mutation, not a third key experiment. Metadata-equal arbitrary target substitution is not claimed detectable.

Atomic late-stage setter tests use a two-stage config-only I block (.08/.04 references; .04 current): .06 fails without changing the earlier stage, then same .04 is unchanged. NI temporarily lowers the last uninitialized stage reference in the existing A/AU session, rejects .06 atomically, restores .08 and compares the original config. This is exactly3 config snapshots/4 config comparisons and2 extra I Encoders, with no extra model/plan/oracle/update. After real NI velocity, reference reset is rejected under the existing later-rate snapshot; I's existing read-only stage API exposes no mutable reference setter, and its actual reference is checked before/after real updates and in raw snapshots.

Poison I allows exactly14 source-stage real refreshes plus first boundary1 before the second boundary is refused:15 actual. Poison AU allows4 state boots plus first boundary1 before the second boundary is refused:5 actual. Work converting the refused second boundary is included in actual partial counters. Both assert global progress0, real source post-state/changed payload, downstream exact initial expanded/NI compact payload, retained prefix evidence, and retry rejection. No destination trainer call or fabricated failed-success receipt is used.

## Actual guard sites and prospective limits

All caps below are source-derived future limits, **not observed numerical counts and not execution authority**. The thread-local P5 budget is activated before the first actual parameter constructor, guards before calls, and final exact counts are asserted except the explicitly conservative boot/refresh ceilings. No warmup constructor or hidden numerical probe is added.

| Limit | Source site and derivation |
|---|---|
| Params2, proof1, descriptor1 | Future entrypoint immediately before actual CkksParams::new, DftSupportProof::verify_current, BootstrapMetadata::new_with_verified_support. |
| Keys2 | Shared make_owned_fields immediately before actual KeyGenerator::new, after all actual pointer/bootstrap/rotation checks. Positive private terminal spy1; invalid identity/actual-vs-bound parameter +7 parameter variants +7 partition variants all stop before the private factory (16 negatives). |
| Bootstrapper1 | Existing make_material actual Bootstrapper::new site, with added P5-local before-call hook; same real Arc used by facade and all NI stages. |
| Encoders40 | Existing make_material site2 + actual session stage Trainer/NI constructors36 + config-only I Trainer constructors2. Block wrappers/facades/clients clone the existing Encoder. NN has another Encoder site in ActCost::probe, but this bridge/driver does not invoke that helper or predictor path. |
| Geometry36/initial36/update100/handoff58/wrapper0 | New synthetic child charges existing Work::Geometry and Initial immediately before real helpers. Existing parent compose supplies actual update/handoff charges once per typed transition. No canonical mini materialization. |
| Block attempts42 | Guard immediately before the one existing block wrapper call, after preflight/configuration.40 successes +2 intentional failed prefixes;98 intended underlying stage calls =96 successes+2 source prefixes, not measured stage windows. |
| Actual boot helper <=361 | CpuBlockEvaluation::bootstrap immediately before the actual same-Arc real.bootstrap. Conservative284 state +34 AA activation +43 newly performed boundary =361. AA boundaries add0. Injected refused call is rejected before actual-call reservation. |
| Actual client refresh <=357 | BlockRefresh::refresh immediately before the narrow actual service refresh:334 model/velocity +23 boundaries =357; refused call is rejected first. |
| Packets42 / x064 / stage y152 | encrypt_block_rows one logical guard and separate immediate guards at actual x0 and each y Encryptor::encrypt sites. No later-stage plaintext x encryption or dataset ciphertext cache. |
| Successful handoff outputs84 | Actual combined client inspections84, explicit I22/NI62 assertion; NI62 includes42 new-boundary outputs and20 AA clones. Failed prefix outputs are reported separately. |
| F64 14/14/40/128 | One reused accepted PairedOraclePhase; its own before-call actual guards and final completion/count checks. |
| Full session snapshots46 / unchanged68 | RawSnapshot::capture and assert_unchanged:40 success replay/exhaustion +2 C shared negative snapshots +2 later-rate +2 poison retry. Raw views include model/velocity/compact/AA/boundary payloads, shapes, cardinalities, Option presence, actual config/flags/counters/traces/progress/failure. |
| Additional poison initial-model captures2 | Approved addendum: before-copy guard at capture_prefix_initial. I expanded and NI expanded+compact initial model payloads are copied once per existing poison session, charged as live roles, fully compared, then dropped before retry snapshot. Total state/model capture events48; config-only3 separate. |
| Combined endpoints40 / stage96 / handoff84 | Before-call guards in existing client's block inspection child around the actual endpoint, delegated inspect_snapshot calls, and handoff decrypt/unpack sites. No extra failure decryption. |

The static audit independently derives16 plans,36 geometry/initial,100 updates,58 handoffs/158 transitions,42 packets,64 first inputs,152 labels,96 stage inspections and84 successful handoff inspections from the frozen matrix arithmetic only. There has been no planner/Engine/HE execution to establish these counts dynamically.

## Retention and limits of observation

BlockRetentionReceipt checks arithmetic over actual retained ciphertext coefficient words and actual training/bootstrap key bundle words. It separates model W/b, I velocity, NI compact W/velocity, current caller packet, I recorded boundary clones, NI boundary clones, AA source activation outputs, client exported model/handoff snapshots, full-session diagnostic copies, and the two approved initial-model copies. Every ciphertext clone is a distinct payload role even if full limbs match. The exact currently live packet is included in endpoint/raw-snapshot receipts; successful endpoint copies are dropped before the replay snapshot. Negative snapshots drop before valid work; poison initial copies drop after corresponding failure comparison and before retry snapshot; packet and session snapshots remain bounded by the fixed loop scope.

Analytical role/count estimates for wrapper input/next clones, current old-hidden/forward caches, active gradients and scheduled reference/transpose work are explicitly separate. NN temporary overlap, bootstrap CT/PT workspace and retained transforms, evaluator/encoder caches, parameter/client public+secret/RNG storage, object/Vec capacity and allocator overhead remain unobserved. The receipt is quiescent payload accounting, not a measured peak, RSS, resource/security admission or secure-memory gate. Descriptions/geometry clones also occupy metadata storage; no P3 receipt-byte claim was changed to hide them.

## Concerns and unclaimed completion

1. Runtime parity, finite HE values, actual conservative boot counts and NI3e-5 attainment are unmeasured. A compile or clippy success does not establish any of them. Future root review must pin exact final sources, inspect actual guards, and separately authorize only the one ignored test if appropriate; this report grants no execution release.
2. Full P3 typed trace/literal internal mask/kappa evidence remains outside the accepted read-only seams. Configured reference helper evidence is not observation of internal literals. Equal actual traces on repeated schedules have ambiguous temporal provenance, explicitly retained with changed flags.
3. Canonical preparation exists, but canonical full owned D7 gates/start remain unimplemented/disconnected. The synthetic spy cannot establish canonical owned start or secure-profile admission. The original request's extra candidate allowance is a pre-binding caller contract, not a post-hash workaround.
4. Retention is an honest role/liveness inventory with unobserved workspace and overhead, not secure memory/security/range/precision acceptance. External caller-retained copies must be provided/recorded by that caller; no total-RSS claim follows.
5. Full registered21 profiles, full2680+20+7 /quick268 rows and secure/full encrypted-block campaigns remain unchanged and held. No full P3/P5/D7/Task54/M4/M5 completion, scientific accuracy/performance, or GPU/device evidence is claimed.

## Build/static evidence

Evidence directory: `task-54-p5-cpu-source-validation/`. Each Cargo invocation records exact command/env, UTC start/end, free disk bytes, PID/process handle, terminal exit, raw output, source-before/source-after hashes and a source snapshot. Every Cargo call checked free disk >=805306368 bytes. Only approved selected `cargo test ... --no-run` and strict `cargo clippy ... --lib --tests -- -D warnings` ran, with CARGO_INCREMENTAL=0/CARGO_BUILD_JOBS=8/RAYON_NUM_THREADS=8. All owned roots were rustfmt-ed with edition2024 and skip_children=true. No whole-workspace/release/all-features build or runtime command ran.

`red.raw.log` is the genuine first compile RED: E0432 missing actual BlockPlanOwner/CpuBlockSession APIs, before implementation, with the initial ignored skeleton preserved under red-source. `compile1` records the initial implementation's actual lifetime/borrow compiler failures; repair fixed view borrowing without adding hidden ciphertext clones. `clippy1` and `clippy2` record real warning failures (dead test-only identity helper, range loops, boolean simplification, then missing handoff-accumulator use); they are retained, not hidden. Subsequent changed-source builds/static checks are separately recorded; passing commands were not redundantly rerun on identical sources. Final source pins, final selected no-run and final strict clippy are in the machine-readable records appended below.

| Invocation | UTC start | UTC end | PID | Exit | Free disk bytes |
|---|---|---|---:|---:|---:|
| red | 2026-11-07T08:22:40.561537+00:00 | 2026-11-07T08:22:43.658528+00:00 | 31102 | 101 | 60389982208 |
| compile1 | 2026-11-07T08:35:39.403273+00:00 | 2026-11-07T08:35:43.018679+00:00 | 47658 | 101 | 60359016448 |
| compile2 | 2026-11-07T08:36:27.616541+00:00 | 2026-11-07T08:36:42.265424+00:00 | 48734 | 0 | 60357472256 |
| clippy1 | 2026-11-07T08:37:26.443638+00:00 | 2026-11-07T08:37:29.996307+00:00 | 50029 | 101 | 60325601280 |
| clippy2 | 2026-11-07T08:38:41.883907+00:00 | 2026-11-07T08:38:46.639430+00:00 | 51567 | 101 | 60333154304 |
| clippy3 | 2026-11-07T08:39:12.353188+00:00 | 2026-11-07T08:39:16.182543+00:00 | 52321 | 0 | 60322271232 |
| compile-final | 2026-11-07T08:40:39.847690+00:00 | 2026-11-07T08:40:57.235067+00:00 | 54268 | 0 | 60255739904 |
| clippy-final | 2026-11-07T08:41:18.020179+00:00 | 2026-11-07T08:41:22.407835+00:00 | 55204 | 0 | 60253528064 |
| compile-trace-final | 2026-11-07T08:42:07.915640+00:00 | 2026-11-07T08:42:24.860751+00:00 | 56239 | 0 | 60264341504 |
| compile-accepted-source | 2026-11-07T08:43:22.561880+00:00 | 2026-11-07T08:43:39.890283+00:00 | 57854 | 0 | 60258340864 |
| clippy-accepted-source | 2026-11-07T08:43:43.370064+00:00 | 2026-11-07T08:43:47.123026+00:00 | 58282 | 0 | 60249649152 |

Final selected no-run: **GREEN**, `compile-accepted-source.raw.log`. Final strict clippy: **GREEN**, `clippy-accepted-source.raw.log`. Both command records match every final source pin below, before and after. `git diff --check`: exit0; raw record retained. Static source audit: all expression/body/unchanged-source checks true. These are compilation/static results only.

```text
afa043377a9b2403f6f79e94ab2b89d7077484d2c6bb9f00a7e6da6885edd4e1  crates/ckks-train-bench/src/lib.rs
0c9fca750faeac190626744bb8d16c81c1215557a209a7aa28c2ad315ce99adc  crates/ckks-train-bench/src/study_blocks.rs
8421b9b33d843de7833c3f1e280fcd94e521dec17618fb96e65d25dcc331f19d  crates/ckks-train-bench/src/study_blocks/p5_tests.rs
0143659b8d9cd5f19e9726288199088951f7ed18dbb5853d6b7a8cca79f2076d  crates/ckks-train-bench/src/study_client.rs
7c23eddca07d8fd2e96bf52b7e97f62c34c821886c9fc2c56cb9e5ebb595efb5  crates/ckks-train-bench/src/study_client/blocks.rs
72cd597e46eca197e349032c89f1f59c0ed7e109b4671b2c3d300eb7922a784a  crates/ckks-train-bench/src/study_admission.rs
5e72a14b3adf70946e20f087d4ab0c1371c3b723f0c1fae0593129c8f7b5e435  crates/ckks-train-bench/src/study_admission/p5_fixtures.rs
6360dacc998bda23ee9494dcf21ee7ae035fcb8f7ad74467c5fd2e1df91278e0  crates/ckks-train-bench/src/study_diagonal.rs
```

No execution release. Independent root source review remains required.
