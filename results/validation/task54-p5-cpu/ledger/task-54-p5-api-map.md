# Task54 P5 CPU block bridge — read-only API and bounded acceptance map

**Mapping only; no implementation or execution release.** Actual P4 bounded acceptance is32successful updates+2injected failures/71real boots, accepted by `task-54-p4-cpu-runtime-review.md`; its full typed trace/literal-mask/canonical-D7 limits remain. Root reports published P4 source at `77fd08de743b9a310cf9ae42f8036e3931ff0c5c`. P5 has no new numerical, metadata-planner, runtime or scientific result here. Only this report is written; no source/Cargo/test/Engine/HE/constructor/network/Git/subagent operation was run.

The original Task54 goal is unchanged: all21profiles/seeds and full2,680+20+7 /quick268 rows, CPU/GPU/I/NI/baselines, provenance/targets/reporting and held block campaigns. No rows, missing exports or holds are removed. Mini tests below are separate noncanonical evidence, not replacement registry cases, canonical owned D7 gates/start, secure memory/range/precision/security, dataset accuracy or performance acceptance. Seed export/remote/target capture are outside this assignment.

## 1. Existing call graph is sufficient for training

The actual source supersedes the older §5 design text describing missing setters/microsteps: these NN amendments already exist.

| Existing source API | P5 use and exact boundary |
|---|---|
| `nn/block.rs:134,171,198–227` `BlockMlp::{new,encrypt_at,num_stages,stage,stage_mut,stage_dims}` | Assemble actual stage MLPs, not a new network/recurrence. Canonical StageGraph layer ranges/representation roles select each stage's arrays/activations. Local heads are retained/trained but never feed the next stage. |
| `block.rs:388,423,435,475` `BlockTrainer::{new,stages,set_lr,train_step_valid}` | Stage trainers and separate boundary refresher share a common Layout/refresh level. `set_lr` validates all stages before committing any. One block invocation consumes all micros; each stage's actual `train_step_valid_with_hidden` runs once; boundary takes `hidden[micro][0]` computed before that stage update. `handoffs()` counts successful boundary refreshes. |
| `block.rs:545,584,609–646,693` `BlockNiTrainer::{encrypt_network,new,stages,stages_mut,set_lr,set_lr_reference,set_next_batch_full,train_step_valid}` | Encrypt expanded+compact state client-side; configure frozen biases/reference before assembly/velocity. Current and reference setters validate all stages first. One block call; NI later aligns incoming ciphertext levels internally. Stages may support mixed placements in NN, but registered P3 uses one case mode; this first bridge proposes the four homogeneous modes only, adding no registry axis. |
| `block.rs:653,666` `handoff_bootstraps`, `last_handoff_bootstraps` | WN/AU: convert old hidden to Delta, bootstrap once per boundary/micro. AA: exact source `last_activation_bootstraps()[micro][0]` clone, no conversion/new boundary boot. Getter is `[boundary][micro]`; empty before/after failed update, published after success. |
| `trainer.rs:861` optimizer view; stage W/b/trace; NI optimizer/reference/state/placement/boot accessors | Read every stage/layer/container/Option entry and all raw configuration. Unsupported I diagnostics stay errors. Existing NI reference-factor helper is configured-state evidence, not literal internal mask/kappa observation. I trace can precede refresh failure; NI stage trace is published at commit. Earlier completed stages can expose valid traces when a later boundary fails. |
| `study_admission` `DiagonalPlan` + `TransitionIndex::{update,stage,is_boundary,next_full,update_metadata,handoff_metadata}` | Validate exact order `update(s0),handoff(s0),update(s1),…` for every consumed update: U(2S−1) entries, Sinitial states, Uordered partitions. Do not apply P4's `transitions.len()==partitions.len()` or stage0-only rule to blocks. |
| `UpdateMetadata` full pre/post/micros/hidden/canonical/config/count getters and `HandoffMetadata::{outputs,counts_nonbootstrap,counts_including_bootstrap,client_refreshes,bootstrap_count}` | Sum every stage and boundary's inclusive seven counts once; compare aggregate measured counter delta around the one block call. Preserve separate nonbootstrap model, refreshes, state/activation/boundary boot decisions and per-stage diagnostics. Whole-step seven-field comparison does not create separately measured per-stage counter windows. |

`plan_stage_handoff` (`nn/diagonal_plan.rs:1932`) already verifies source/destination geometry, takes old hidden, makes I/WN/AU conversion/refresh/boot or AA reuse, and yields typed per-micro outputs. P3 `compose` already feeds these exact outputs into the next stage's `MicroMetadata.x`. In NI this is boot level16 while fresh `state.batch().x` is level15: **compare incoming handoff to that transition's micros, not the fresh-input BatchSpec**. Compare each stage's client y to its own receipt/BatchSpec. No one global y-scale assumption.

```text
client-owned trusted MaterializedProfile + sealed registration + Arc<DiagonalPlan>
  -> P3 validate_plan_binding -> describe_candidate(S geometries, full manifest)
  -> private block checked preparation -> shared private owned key-material factory
  -> existing StudyClient + caller-owned CpuEvaluationContext
  -> client encrypts S initial models, x0 and one y per stage/micro
caller-owned block evaluation facade (borrows base context; real bootstrap forwarder)
  -> CpuBlockSession borrows facade/client refresh handles
  -> BlockMlp + BlockTrainer OR BlockNiTrainer
  -> exactly one train_step_valid per logical update
  -> postchecks/progress OR permanent poison + actual partial delta
  -> explicit encrypted block snapshot -> client-only diagnostics
```

## 2. Smallest seams and future ownership (not assigned here)

Keep accepted P4's public packet/session behavior and single-stage rejection. Add a closed private `BlockPlanOwner` (canonical Arc; cfg(test) distinct synthetic Arc), not a generic Plan trait or conversion into DiagonalPlan. Keep source shape explicit at2/3stages. This is preparation data, never Ready/start authority.

| Future file/hunk | Minimal proposed work / alternative |
|---|---|
| **NEW** `study_blocks.rs`, `study_blocks/p5_tests.rs`; `lib.rs` declaration | Own immutable block packets, caller-context facade, session, stage/handoff diagnostics, checked retention receipt and the two separately selectable ignored pure/HE test entrypoints in one lib binary. Reuse `EncryptedMicro`'s existing x/targets/part shape internally; a new outer block packet binds its own plan, Swidth/class descriptors and association. P4 packets retain their one-target contract. |
| `study_client.rs`: one child declaration; **NEW** `study_client/blocks.rs` | Implement block client methods as a child, so existing private owner/encryptor/decryptor/encoder/RNG remain private and are not copied into a second client. Gather D6a rows once in exact order; call existing padded_targets separately for each stage C/P; encrypt x0 once and Sdistinct y ciphertexts at bound per-stage labels. Later inputs are encrypted server handoffs. No plaintext server handoff, new initializer or alternate loss fold. Snapshot inspection delegates existing StudyClient per-stage inspection plus client-only unpacking of explicitly exported handoffs. |
| `study_client.rs`: narrow common setup helper | `prepare_then_make`/`CheckedPreparation` deliberately reject blocks today. Add a separate closed block preparation path using the same P3 binding/candidate/manifest rules; do not weaken that P4 gate or use candidate placeholder initial states. Prefer extracting only the existing private parameter/bootstrap/union-key checks + owned KeyGenerator-first factory into a small shared **borrowed setup-fields helper**, called by unchanged-semantics P4 adapter and block adapter, then existing `StudyClient::make_material`. No duplicate client/key ownership implementation, public callback/digest override or fake permit. A future source brief must name the exact moved hunk and independently preserve P4's consuming-order assertions. Public caller-created keys still cannot prove pre-key rejection. |
| `study_diagonal.rs`: narrow helper visibility/extraction | Reuse context owner/evaluator/encoder/boot metadata, `check_geometry`, `check_ct`, `Description` and actual activation construction. `check_mlp`, `check_batch`, `check_optimizer` are private; expose them crate-private or extract the existing read-only stage comparator into functions taking actual MLP/Trainer or NI trainer + description + StateMetadata + successful index. Preserve comparison expressions/order/errors and P4 entrypoints; do not copy optimizer math or relax any absent/unsupported/cardinality checks. Block initial packet checks all Sstates before any trainer construction. |
| `ClientRefreshService` narrow handle seam | `RefreshHandle` is private. Add a crate-private handle method returning the existing borrowed Refresher (or equivalently narrow type visibility); create stage and boundary handles from the same client-owned service. Do not expose mutable client crypto. |
| Block facade + I boundary recorder, new block module only | P4's cfg(test) CpuBootstrap budget is concretely tied to P4 RuntimeBudget. Prefer a new caller-owned sized forwarding facade over the **same actual** bootstrap Arc and base context, with P5-local guards/failure hooks; no P4 budget rewrite, fake HeOps or arbitrary public BootstrapOps injection. A boundary-only Refresher recorder forwards the narrow client handle and retains ordered I output ciphertexts/metadata for this update; all stage/boundary handles share a P5 actual-call refresh guard, while only boundary handles retain outputs. This fills I's missing handoff output diagnostic without NN math changes. |
| **NEW** `study_admission/p5_fixtures.rs`, cfg(test) declaration only in parent | Distinct synthetic block geometry/state/identity, explicit Layout512/d16/lanes8, exact closed recipe, existing initial_stage_metadata and parent compose once. Reuse parent codecs, including **encode_rate_schedule_fields with explicit synthetic reference**; canonical rates()/full-max policy remains unchanged. No P4 SyntheticCpuPlan promotion/reuse or invented future suffix. |
| **NEW** `study_plain/p5_fixtures.rs`, cfg(test) declaration only in parent | Closed block families and true full-block Engine<f64> oracle/sensitivity owners, read-only actual initial/post/old-hidden arrays. No public Engine/materializer constructor or copied gradient/NAG/activation recurrence. Leave accepted P4 fixture untouched. |

**Actual NN diagnostic gap:** I `BlockTrainer` exposes handoff count but not ciphertext outputs. Preferred bench boundary recorder above suffices. If root instead wants NN parity with NI's API, the smallest alternative is a read-only `last_handoffs() -> &[Vec<E::Ct>]` plus bounded storage of existing outputs in `block.rs`; it must explicitly define success/failure clearing and charge retained clones. That is a separately owned NN storage/accessor amendment, not needed for training. No new NN setter, microstep or numerical recurrence is otherwise necessary. I stages are read-only through `stages()`; do not add mutable optimizer access just to manufacture diagnostics.

Canonical preparation uses genuine MaterializedProfile/RegisteredCase and existing binding validation, then clones only S candidate geometries; stage states/rates come exclusively from plan. Reserve the extra **Sgeometry+Sinitial calls in the original request before hashing**; preserve original typed budget issues. Mini synthetic tests bypass neither secure canonical validation nor D7: they use their distinct domain. Production run_owned remains disconnected until real owned gates/durable start exist; full blocks remain held.

## 3. Ownership, checks and retention

| Owner/lifetime | Allowed contents / obligation |
|---|---|
| Driver/client | Materializer, exact arrays/order, StudyClient secrets/RNG and refresh service. Initial/label encryption only here. Context/setup keygen material is created after checked preparation; key owner association is fresh and private. |
| Caller-owned base context, then block facade | Public evaluation material, base counter fork, same owner token and actual Bootstrapper Arc. Facade outlives stage trainers; its guard records actual forwarding operations, not a predicted boot total. No self-referential trainer borrowing its own struct. |
| Session borrowing facade/services | Immutable Arc plan, BlockMlp, existing block trainer, encrypted packets/diagnostics, successful logical index/sample totals, poison/prefix evidence. No dataset, initial plaintext arrays, Decryptor or reset/repair callback. |
| Diagnostic endpoint | Client inspects one explicit encrypted block snapshot per successful update, including all stage W/b and old-hidden handoffs; training evaluator is fork-isolated and counters unchanged. Test disclosure is not a terminal-only NI deployment claim. |

Before work: validate association/plan/index/exhaustion, full ordered parts/total, one x0 and exactly Sstage targets, each stage shape/C/P/activation/bias policy/config, actual parameter/key/bootstrap identities and all CT contracts; compare every pre-state before setting the new rate. Validate every stage's rate read-only, progress arithmetic and common lookahead before poison. Then poison **before** wrapper setters, apply actual atomic wrapper current/reference policy, and invoke the wrapper once. Clear poison/publish successful totals only after all stage/handoff/post/count checks. On error preserve actual partial delta and available stage traces, reject retry, and do not imply rollback: earlier stage states can be committed while global successful index stays unchanged.

For AA, compare every NI handoff CT exactly with its recorded source AA CT (scale/level/basis/components/limbs and full lengths), and require0extra handoff boot/conversion count. I recorder and NI last_handoff outputs compare every typed HandoffMetadata output and corresponding destination MicroMetadata.x. Numerically compare old hidden against true F64 cached old-forward arrays in the same client endpoint; compare all logical padded-channel W/b as well. Full typed P3 trace/literal internal kappa or masks remain unclaimed. Equal-scale ciphertext contents are not authenticated by metadata: arbitrary same-scale target substitution is outside the private immutable packet API; target semantics are established by trusted client construction and actual all-entry oracle comparison, not a fabricated ciphertext-content validator.

Proposed checked `BlockRetentionReceipt` is a phase/liveness payload bound, not RSS/admission: all Sstage model/velocity/compact state; one current packet; active stage gradients plus scheduled reference branch/transpose work; current inputs/old hidden; recorded boundary outputs; AA retained activation clones; snapshot copies; actual key/bootstrap bundles/workspace lifetimes. For the largest proposed3stage case: expanded W/b22CT, Ivelocity16CT, NIcompact W/U6+6CT, packet at most8CT, successful handoff outputs at most4CT, and NI per-stage AA outputs up to6CT. These are **roles**, not a total peak: wrapper next/inputs/hidden clones, gradients, all RNS limb levels, duplicated bundles and bootstrap workspace must be included. Ciphertext::clone copies polynomial payload; equal limbs do not mean shared allocation. Drop negative-test snapshots before valid training and current packets after the endpoint. Do not alter P3 receipt-byte claims to hide cloned geometry/retention. Full secure-profile resource/range/security/precision gates remain separate.

## 4. Closed proposed fixtures and sensitivity prerequisite

These are proposed immutable recipes, **unmeasured**. Freeze source/arrays/tolerances and publish pure sensitivity witnesses before any HE release. Carry P4's I1e-5/NI4e-4 absolute caps as proposed P5 caps, with mandatory finite/all-entry/frozen-F64-zero checks. Require each relevant mutant's affected-entry L-infinity witness **>2x each cap** (NI threshold8e-4), recording family/boundary/update/layer/entry and equivalence/inapplicability. A failure stops before crypto and needs a new reviewed recipe decision; no parameter/rate/tolerance search or automatic loosening.

All stages use NAG.9/frozen zero biases/RSS delta1, reference.08, stage-specific classes/padded widths. One closed initial-array factory uses hidden W=.45/fan_in and head W=.75/hidden_width, zero biases; identical arrays feed real Engine and encryption. First input x[r,0]=.875+.125*(r%2), x[r,1]=.875+.0625*(r%3); y_s[r]=r%C_s, using global logical r across micros and restarting only at each update. These candidate signal levels do **not** prove sensitivity/range; pure checks must confirm every old hidden stays within proposed abs .75 at all baseline update entrances/final endpoint and that all required mutants separate. No clipping/normalization repair.

| Family | Stage dimensions, C/P and activations | Updates / parts |
|---|---|---|
| A, e2 schedule | `[2,4,4]` C3/P4; `[4,2,2]` C2/P2; both Poly/Identity | .04,.08,.02,.08; `[8,4];[8];[5];[8,4]` |
| B, e3 schedule | A's two local widths, then `[2,3,16]` C10/P16; all Poly/Identity | Same4rates/parts |
| C, heterogeneous label probe | A dimensions/C/P; stage0 Poly/Identity, stage1 Poly/Square | .04; `[8,4]`, explicit ref.08 |
| D, first-reference | A dimensions/acts/C/P | .08,.08; `[8];[8]` |
| Poison prefix | A actual initial arrays/first rows, a distinct one-update synthetic .04/ref.08 plan | `[8,4]`; no new F64 trajectory |

C is explicitly a synthetic heterogeneous-y-label probe, not a new registered eMLP activation profile. A/B already cover heterogeneous incoming NI handoff level16 versus fresh15. The deliberate H=P at each local boundary makes head-as-hidden errors shape-compatible, so a shape rejection cannot masquerade as numerical discrimination. Canonical profiles currently have one real C throughout; heterogeneous synthetic C_s validates each stage locally before padded_targets. Existing Engine has one global classes field: the private fixture can set max(C_s), validate every supplied label against its closed stage C_s before Engine::train, and use each layer's real output width; RSS delta1 introduces no stage-C arithmetic. No public/canonical per-stage-class axis is added.

Baseline oracle uses the actual multi-stage Engine::train (`study_plain.rs:586`): it caches **all** stage old inputs/forwards before computing/applying any gradient. Closed sibling child access is sufficient. Hand-off mutants alone use one-stage instances of this same Engine plus existing forward: substitute actual local-head output or actual post-update hidden at one selected boundary, with persistent real Engine velocities; skip-local-update control likewise skips that selected Engine call. These are intentionally wrong scheduling controls, not new gradients/recurrences. Schedule controls reset existing velocity or change existing rates/row assembly, then invoke real Engine. Do not turn the old one-dimensional test fixture's tiny signal into assumed HE evidence.

Proposed pure phase:4baseline trajectories; for each local boundary, head-as-hidden and updated-hidden trajectories; omit-selected-local-update trajectories; applicable ResetVelocity, ConstantReference, LostTail and micro-r-restart trajectories;9first-step stage-target perturbations (2+3+2+2) verifying other stages bit-identical and selected stage responsive under genuine old-forward semantics. No literal generic cross-stage-backprop mutant is implemented by copying derivative math: use target-isolation invariance, actual wrapper graph and all-entry HE checks; retain that distinction.

| Pure actual Engine::train calls | A | B | C | D | Total |
|---|---:|---:|---:|---:|---:|
| Full-block baseline |4|4|1|2|11|
| Two handoff mutants x each boundary, one-stage Engine calls |16|48|4|8|76|
| Omit-local-update mutants, one-stage Engine calls |4|16|1|2|23|
| Applicable four schedule/row controls |16|16|3|2|37|
| First-step target-isolation controls |2|3|2|2|9|
| **Pure cap** |42|87|11|16|**156**|

This entails61private Engine objects, not156new engines. Closed diagnostic/mutant forward calls outside train are bounded179:35baseline entrance/final-stage forward captures +106handoff-control forward calls +38skip-control forwards. The paired HE oracle phase below adds14Engines/40train calls and128such diagnostic forward calls, for total75Engine objects/196train calls/307explicit external forwards across one pure phase plus one HE phase. Internal Engine forward/gradient calls remain its own implementation. Guard actual helper invocations before calls; do not count a one-stage mutant call as one whole-block oracle update. Separate ignored pure and HE entrypoints allow root to release phases independently without rerunning the mutant sweep in HE. Source/evidence pinning is root's execution custody, not a new fake readiness token.

## 5. Proposed finite HE acceptance and derived caps

One separately released ignored HE lib test; two local contexts only, no OnceLock/global crypto fixture, shared union keys planned before the two real key factories. Use P4's same explicit synthetic I/NI parameters, proof/descriptor/bootstrap setup and Layout512/d16/lanes8; they remain noncanonical. All models borrow context forks. All successes pair with one actual full-block F64 oracle per session, no mutant rerun.

| Cases | Sessions | Logical successes | Planned stage updates | Planned handoff records |
|---|---:|---:|---:|---:|
| A, four-rate I/WN/AU/AA |4|16|32|16|
| B, four-rate I/WN/AU/AA |4|16|48|32|
| C, heterogeneous labels I/WN/AU/AA |4|4|8|4|
| D, first-reference I/AU |2|4|8|4|
| Two poison-prefix I/AU plans |2|0 (2failed attempts)|4|2|
| **Total** |**16**|**40**|**100**|**58**|

Geometry/initial helpers:36each (20A/B +8C +4D +4poison). P3 compose once per16plan,100update +58handoff calls,158typed transition records; no extra binding plans. Exactly42actual block-wrapper attempts; on the intended completed paths98NN stage calls (96successful-stage calls +one source-stage call per poison) are source-derived, not separately measured per-stage windows. Cases cover full8, partial5, unequal8+4, first below/equal reference, persistence/decrease/restart and terminal unknown lookahead. Plan full reference and all future suffix coefficients remain authoritative for canonical quick use.

Poison I: stage0 completes14real model/velocity refreshes, first boundary refresh succeeds, second refuses;15real refreshes total, no destination update. Poison NI-AU: stage0 completes4real state boots, first boundary boot succeeds, second refuses;5real boots total. Both conversions before the second refusal still contribute actual counter work. Assert global successful progress0, source committed-state evidence, downstream untouched state, real prefix/counter delta and permanent retry rejection; no rollback or hidden NI boundary trace claimed. Existing NI last_handoffs remains empty after failed block step. Guards forward real implementations only.

Atomicity: add one config-only I BlockTrainer containing2stage trainers with references .08/.04 and current .04; .06 must fail at the later stage without changing earlier rates/counters, and same-rate .04 is unchanged. This requires **2additional Encoder constructions**, no model/Engine/plan/HE step. NI can temporarily lower the last uninitialized stage reference in an already-planned A/AU session, test the same late-stage .06 rejection, then restore .08 and prove the bound snapshot restored before training, using existing stages_mut/setters. Reference locking after velocity is checked on the real scheduled session; no mutable I optimizer API or bogus state is introduced.

Block negatives:24prework rejects (12each on existing C I/AU packets): association, plan, index, logical total, partition gap, zero valid, missing micro, missing stage target, extra stage target, hetero-label target swap, xlevel and yscale. Metadata-equal arbitrary target substitutions are not claimed detectable. Add2later changing-rate rejects in A I/AU at update1 with all velocities present; reuse40success replay/exhaustion and2poison retries. Full raw/encrypted snapshots yield46captures/68unchanged comparisons, plus separate3config-only snapshots/4configuration comparisons for atomicity. No new valid attempt/boot/encryption/oracle/plan for these checks. Wrong-owner remains an association mutation, not a third actual key experiment. Return all per-stage W/b/old-hidden comparisons in40combined client reporting endpoints (96delegated per-stage model inspections +84successful handoff CT inspections); other negatives need no client decryption.

| Hard expensive bound | Proposed cap and derivation |
|---|---|
| Params / keys / proof / descriptor / Bootstrapper |2 /2 /1 /1 /1; setup once per mode, actual union keys not guessed numeric cardinalities |
| Encoders |**40** =2setup+36session-stage constructors+2config-only I trainers; block wrappers, facades and clients clone existing Encoder |
| Main planner |36geometry /36initial /100updates /58handoffs /0canonical wrappers;16plans. No candidate/canonical materialization in mini suite |
| Paired F64 |14oracles/14Engine objects,40actual full-block updates; pure156calls remain a separate prior phase |
| Block attempts |42, with40intended successes+2failures; stage-call inventory98as above |
| NI state boots, conservative |284: A96+B144+C24+D16+poison source4; each stage has exactly2weight containers and2NAG velocities, hence max4state boots/stage/update |
| NI AA activation boots |34: A12+B18+C4 |
| Newly performed NI boundary boots |43: successful A12+B24+C4+D2, plus poison first1; AA adds0 |
| **Actual real bootstrap helper cap** |**361 =284+34+43**. WN/AA may use fewer state boots; no bootstrap warmup/probe beyond this bound |
| I model/velocity refreshes |334: A96+B152+C24+D48+poison source14 |
| I boundary refreshes |23: A6+B12+C2+D2+poison first1 |
| **Actual real client-refresh cap** |**357 =334+23** |
| Handoff payload budget |58planned boundary records/88planned boundary-micro outputs including failure plans;84successful returned outputs (I22/NI62). NI62=42new-boundary-boot outputs+20AA reuses. Failure prefix outputs are separate; never fill a successful receipt from them |
| Client packets |42logical packets,64first-input CT encryptions and152stage-target CT encryptions; no encrypted dataset cache or later plaintext input encryption |

These are proposed source-derived budgets under the outlined driver, not measured counts or execution authority. Helper sites and exact diagnostic allocation/call counts must be independently verified and frozen during implementation review; missing invocation visibility must remain explicit rather than filled with invented per-stage measurements.

Every stage's compact L<=16, cap=512/L>=32, used weight positions<=4 and at most2bias slots, so the2containers/stage and no-bias-only-container bound follows from existing CompactLayout placement rules. I expanded diagonal sums are6/4/6 for the three stage shapes; NAG refreshes per-stage are14/10/14 including frozen biases. For A/B each4step schedule has total6micros; these facts derive the boot/refresh bounds, not a model of observed performance. The heterogeneous final activation does not change container/diagonal counts.

Before any future runtime, root must review/freeze exact source, all phase guards/diagnostic counts and pure sensitivity witnesses. Unexpected geometry, container count, scale failure, cost/discrimination failure or extra constructor stops at its cap and is reported; it does not authorize a rerun, silent cap change, dropped case or tolerance change. First implement tests/no-run/static under a new explicit ownership release; then pure sensitivity release; only after acceptance consider the separately scoped HE release. Full canonical block/campaign/D7/resource/target/GPU work remains held.

## Inspected primary files / drift pins

P4 runtime report/review and the cited design sections were read; no tests were rerun. Source pins at mapping time:

```text
4227bff4f06db0dbe61a2d97e4a20986209347b8f15e74d8db852b16cdad0950  crates/ckks-train-bench/src/study_client.rs
44107c8617582883558e79dbc9cb37abefd2502d3c3ceaf6f361953c29f9102f  crates/ckks-train-bench/src/study_diagonal.rs
60f1dbf83e30ea59eee89e2a38889a9161fc04570493a13fff00e0a21ac4e9a2  crates/ckks-train-bench/src/study_admission.rs
d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b  crates/ckks-train-bench/src/study_plain.rs
0a7f53e1e61ff6b1660322aa9c23f5967a14f1a5aa92a7d69bcb745203ce7cef  crates/ckks-train-nn/src/block.rs
```
