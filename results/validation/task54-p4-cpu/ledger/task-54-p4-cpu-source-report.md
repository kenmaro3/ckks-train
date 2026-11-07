# Task54 P4 CPU bridge source implementation report

Status: **DONE_WITH_CONCERNS — source/compile/static scope only.** The assigned library implementation and finite ignored contract test are written. No HE, F64 Engine, planner, parameter, key, bootstrap descriptor/proof or bootstrapper runtime was invoked. No runtime RED/GREEN, numerical parity, independent source review, canonical pre-key acceptance, full P4/P3/D7/Task54/M4/M5 completion or production admission is claimed. A separate root source freeze/review/runtime release is still required.

Source base was `07f7a989ea46d3398835cfb0535f1aa96fc0b78c`. Root separately reported its docs-only `1dce7a3` during this implementation; the implementer did not stage, commit or push. No subagents, network/SSH/GPU/dataset commands, Cargo dependency changes, target/cache deletion or unowned edits were performed. Existing `.vscode/`, both `results/mnist_blocks_*_plain_b48.json`, and root's design/docs edits were not edited by this implementer.

## Implemented interface and ownership

- New `study_client.rs`: secret-bearing `StudyClient`, explicit client plaintext snapshot endpoint, `CpuBootstrapDescription`, and immutable packet encryption over trusted canonical materialized arrays/row order and existing `padded_targets`. Client retains only prepared plan/Layout association for update encryption; it never reconstructs geometry per update. Public packet fields have no public constructor/mutator/Deserialize path or arbitrary digest/callback authority.
- New `study_diagonal.rs`: caller-owned `CpuEvaluationContext` with independent counter forks, distinct private fresh key-owner association, client-owned `ClientRefreshService` with a narrow borrowed server handle, immutable initial/update/encrypted snapshot packets, and real single-stage `CpuDiagonalSession`. The session owns the MLP and existing I/NI trainer and invokes its actual `train_step_valid` once per validated logical update. It receives no materializer, plaintext arrays, KeyGenerator or Decryptor. NI receives no refresh handle. This is in-process role separation, not process isolation or cryptographic packet authentication.
- New `study_diagonal/p4_tests.rs`: exactly one orchestrating ignored lib unit test, `study_diagonal::p4_tests::p4_cpu_bridge_contract`. No global/OnceLock crypto fixture. A thread-local scope carries only the test's guard-counter handle, not contexts/keys/oracles; both real setup owners are local values in that one test.
- New `study_admission/p4_fixtures.rs` plus only cfg(test) child declaration in `study_admission.rs`: distinct private `SyntheticCpuPlan` using existing P2 geometry/initial APIs, existing P3 compose once, existing transcript/rate/parameter/geometry/partition codecs and typed summaries. It cannot construct MaterializedProfile/DiagonalPlan or production permits. Explicit synthetic Layout512/d16/lanes8 does not change canonical layout policy. Reference .08 is bound explicitly even for Poly smoke .04.
- `study_plain/p4_fixtures.rs`: closed eight-value `SignalFamily`, `SignalRecipe`, `SignalOracle`. Its baseline row/label/rate assembly is shared with the existing trajectory; Family::engine(Signal) remains the sole Signal initial-array constructor and real Engine::train remains the arithmetic owner. Existing v2 mutant/sensitivity tests/values are preserved, not run. The new oracle calls only the baseline trajectory once per successful session. Guards sit immediately before the actual Engine calls.
- `trainer.rs`: borrowed read-only `OptimizerView`, default-unsupported `LayerOptimizer::diagnostic_view`, actual Sgd/Nag implementations, and checked `Trainer::optimizer_view`. No optimizer formula/recurrence or NI source changes. Unsupported is an error, not absent SGD state.
- `lib.rs`: module declarations only. No other source/Cargo changes.

## Canonical preparation, real owned factory and evidence boundary

`study_client::prepare_then_make` has one canonical branch and a cfg(test) distinct synthetic branch, then **one shared `factory(checked)` invocation at line211**. Canonical branch calls existing `validate_plan_binding` first, then existing `describe_candidate`, rejects its issue and any multi-stage shape, compares the complete logical training/bootstrap manifest, and clones only the one geometry into private prepared evaluation data. It never uses candidate placeholder initial states/rates. All initial/pre/post labels and coefficients come from the bound plan. The original request must already reserve the extra geometry/initial call; no limits are mutated after hashing. `CpuPreparationError` retains the original owned typed AdmissionIssue or CandidateDescription, including exact required/configured budget information, rather than flattening it into a display string.

The compile-visible **private** `prepare_owned_canonical` continuation at line704 calls that canonical branch and its same factory invocation, then `make_owned` at line653. `make_owned` verifies exact borrowed actual parameter/bootstrap association and supplied union-key coverage before its first secret/key action, **`KeyGenerator::new` at line689**. It then builds real evaluation material and encrypts the initial arrays through the client. The test's real synthetic setups call this same owned factory through the same consuming seam. Neither function is a test-only fake, and neither is connected to successful production `run_owned`; `run_owned` remains untouched/MissingExecutionDependency pending genuine D7 gate/durable-start ownership.

Public `StudyClient::from_key_generator` accepts a caller-created key owner, validates canonical binding, and returns client/context/initial together so that preparation is reused. This public path **does not establish that rejection preceded the caller's external KeyGenerator allocation**. Public `prepare_initial` and `encrypt_update` likewise grant no D7 capability. Canonical/owned paths compile but were not invoked. The reachable synthetic terminal spy is proposed local consuming-library order evidence only, not a positive canonical validator or production pre-key/start test.

## Runtime checks implemented (not executed)

Session construction and prework compare all exposed expanded W/b entries (not one representative), full diagonal/shape/BSGS/stride cardinality, every compact weight and Option velocity entry, exact BatchSpec bits, raw optimizer config, context/config/key/boot identities, exact ordered partitions/valid totals, input/target width and one stage0 target per micro. Wrong-owner association, plan/index/replay/exhaustion, malformed partition/target/ct basis/level/scale/components/config and nested initial-state cases are encoded without extra updates. The wrong-owner test mutates the private association token while retaining packet parameters; it is **not** a real third-KeyGenerator/same-parameters different-key ciphertext experiment.

Entering state is checked against pre_state before rate setters. The immutable plan selects current rate/lookahead; actual public current/bias/reference-helper factors, raw fixed reference/momentum, reference branch and first/later velocity are checked. Root's explicit clarification retained: NI reference-factor comparison uses existing `NiOptimizer::with_lr(reference).grad_scale()` as configured-state helper computation, not an observed internal mask. Literal internal kappa/mask/scalar observation remains unclaimed; no kappa formula was copied and no NI ownership expansion was made.

The session poisons before the first fallible setter/train operation. Successful update/sample totals publish only after every postcheck; failures keep those totals unchanged, retain actual partial seven-field operation delta and reject retries. Public trace diagnostics are retained only when they correspond to the attempted operation; an unchanged prior trace is omitted instead of presented as a failed trace. Thin real refresh/bootstrap forwarding guards permit one real operation and fail before the second; no fake HeOps or arithmetic substitute.

Every intended success compares all seven measured fields directly with the plan's already-inclusive counts, client refreshes, separate NI state/activation bootstrap decisions, all post-state entries/cardinalities, actual public trace levels and available a1/AA canonical subset. Bootstrap internals are never added to measured counts again. Full typed P3 trace equality is unclaimed (no public typed complete trace/decoder is introduced). All W/b entries are compared with actual Signal F64 snapshots with mandatory finiteness and frozen pure bias exactly zero, at fixed absolute I1e-5/NI4e-4. No tolerance loosening code exists. Diagnostics use evaluator forks and assert training counter isolation; no scientific precision/range/security/memory-admission claim follows.

## Tests-first and permitted static evidence

Initial skeleton/assertion was written before implementation. Focused no-run RED failed because the real consuming bridge types/functions were absent (`CpuDiagonalSession`, `CpuCapability`, `run_contract_matrix`), not because of a fabricated numerical mismatch. This is **compile RED**, not a runtime failed test. The final ignored contract test was compiled, never invoked.

Every Cargo invocation checked free disk >=805306368 bytes before launching, used the existing target with CARGO_INCREMENTAL=0/CARGO_BUILD_JOBS=8/RAYON_NUM_THREADS=8, and polled its same session handle to terminal. Per-command JSON records include exact command, UTC start/end, exit, process PID, tool session handle, free bytes, and before/after source SHA256. Raw output is retained next to each JSON. Cargo hashes are equal before/after; final Clippy and final no-run source hashes match.

| Evidence prefix | Command/outcome | UTC start / end | Handle |
|---|---|---|---|
| `01-compile-red` | Focused no-run RED; exit101, genuinely missing bridge API | 2026-11-07T06:11:21Z / 2026-11-07T06:11:24Z | `72084` |
| `02-compile` | Focused no-run; exit101, type inference and borrow errors corrected | 2026-11-07T06:21:20Z / 2026-11-07T06:21:25Z | `87381` |
| `03-compile` | Focused no-run exit0; unused API warnings later corrected | 2026-11-07T06:24:35Z / 2026-11-07T06:24:47Z | `47497` |
| `04-rustfmt` | Owned-file rustfmt exit0; skip_children=true | 2026-11-07T06:26:36Z / 2026-11-07T06:26:36Z | `62115` |
| `05-clippy` | Clippy exit101; encrypt_rows argument-count lint corrected | 2026-11-07T06:26:36Z / 2026-11-07T06:26:42Z | `62115` |
| `06-rustfmt` | Owned-file rustfmt exit0; skip_children=true | 2026-11-07T06:28:12Z / 2026-11-07T06:28:12Z | `53811` |
| `07-clippy` | Clippy exit101; diagnostic enum large-error lint corrected by boxing payloads | 2026-11-07T06:28:12Z / 2026-11-07T06:28:15Z | `53811` |
| `08-rustfmt` | Final owned-file rustfmt exit0; skip_children=true | 2026-11-07T06:29:54Z / 2026-11-07T06:29:54Z | `37625` |
| `09-clippy` | Final Clippy --lib --tests -- -D warnings exit0, no warnings | 2026-11-07T06:29:54Z / 2026-11-07T06:29:58Z | `37625` |
| `10-compile-final` | Final focused no-run exit0, no warnings | 2026-11-07T06:30:41Z / 2026-11-07T06:30:56Z | `31537` |
| `11-diff-check` | Owned-file git diff --check exit0 | 2026-11-07T06:31:42Z / 2026-11-07T06:31:42Z | `completed direct command` |

Exact Cargo command strings (executed only with `--no-run`, or Clippy):

```text
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

Evidence directory: `task-54-p4-cpu-source-validation/`. Final binary named by Cargo: `target/debug/deps/ckks_train_bench-fe037ad79be4cfe0`; this is only a compiled test binary and was not executed. Formatting exact file list/command is in `08-rustfmt.json`; no child traversal/unowned formatting. `source-final.sha256` contains the same final source inventory below.

## Proposed exact future runtime inventory (no execution release)

One test invocation only, after root's separate freeze/review/explicit release, could select the exact ignored test in the single lib binary with `-- --ignored --exact --nocapture --test-threads=1`. No additional test group, warmup, old pure mutant sweep, canonical materialization, integration binary, full workspace/release/allfeatures or campaign is included.

| Future work | Exact intended count / hard cap |
|---|---|
| Smoke sessions | 12: six existing Signal families in I and NI; C3/C2/C10 NI placements WN/AU/AA, repeated for PolyC3/P4, PolyC2/P2, PolyC10/P16 |
| Four-rate sessions | 4: I/WN/AU/AA, each .04,.08,.02,.08 with parts8+4;8;5;8+4 and fixed reference .08 =16 successes |
| First-reference sessions | 2: I/AU, each .08,.08, parts8;8 =4 successes |
| Poison sessions | 2: I/AU first PolyC3 update, one real refresh/boot succeeds then second rejected; no F64 oracle |
| All sessions | Exactly20 (I9/NI11); exactly34 trainer attempts, intended32 successes+2 failures; guard before train invocation |
| Paired oracles / F64 Engine calls | Exactly18 closed baseline oracles; exactly32 actual Engine::train calls guarded at actual invocation, no mutant/caching/extra oracle calls |
| Pure planner | Exactly20 geometry+20 initial+34 update calls and0 handoffs; all through existing helpers. Shared PlanningBudget hard caps28/28/42/0, wrapper_calls0. No added binding fixture plans: spy reuses first NI plan |
| Parameters/key owner/proof/descriptor/bootstrapper | Exactly2 CkksParams,2 KeyGenerator,1 DftSupportProof verification,1 BootstrapMetadata,1 Bootstrapper, guarded before constructors |
| Encoders | Exactly22 constructions (2 setup+20 trainer); client/service/context copies use Encoder::clone |
| Actual bootstrap helper invocations | Hard cap85 across successes+failed prefix, no extra bootstrap diagnostics/warmup; actual count remains unknown until released run |
| Shared factory spy | Exactly1 valid terminal marker/factory call;16 isolated negatives each0 calls:2 identity/binding,7 existing-codec q/Delta/dnum/secret/p/N/actual NI descriptor mutations,7 empty/overlap/zero/oversize/nonconserving/offset/reordered partitions. No real constructor in spy |
| Initial encryption | Exactly20 client initial model encryptions; expanded and compact NI states use existing Mlp/NI encrypt_network |
| Successful update packet encryption |32 logical packets;2 poison packets. 36 invalid client label/cardinality requests stop before encryption. Ordered full logical Signal x/y assembled once per update and sliced at global offsets; r never restarts at a micro boundary |
| Post-step explicit client snapshot endpoints |32 paired success comparisons plus36 prework-before/after inspection endpoints =68; no extra F64 update. Initial/pre/post state metadata probes remain fork-isolated |
| BatchSpec checks |20 initial+32 successful post-step =52 checks through existing fork-isolated probe paths; no Encoder construction/boots added |
| Other negatives |Reuse existing initial packets/session packets/plans/context forks, including Replay/exhaustion, plan/index/association, partition, labels/shape/target count, basis/level/scale/components, config and nested initial cardinality/velocity.0 extra trainer/planner/key calls |
| Canonical dataset/production dispatch |0 materializations;0 canonical HE runs;0 durable starts;0 successful production dispatches |

The exact constructor/key/Encoder counts and guard final assertions are part of the ignored test; thresholds are not raised on failure. The two key owners have their respective assigned parameter sets. The suite does not add a third owner to experimentally test different keys with identical parameter bytes. Private association-token rejection is separately labelled.

## Final source SHA256 (frozen by implementer; root must independently review/freeze)

```text
4227bff4f06db0dbe61a2d97e4a20986209347b8f15e74d8db852b16cdad0950  crates/ckks-train-bench/src/study_client.rs
44107c8617582883558e79dbc9cb37abefd2502d3c3ceaf6f361953c29f9102f  crates/ckks-train-bench/src/study_diagonal.rs
2427105c82a39cc5fc4bb60b4763ecd9629a739ce75cf9cf6ed50a25597b79c8  crates/ckks-train-bench/src/study_diagonal/p4_tests.rs
9b001769d71954a225df80dcbfcc180301bd347959b00f620abc6d40d384863a  crates/ckks-train-bench/src/lib.rs
4c8dc4f2d30d6a52ad9b5c9f6218eb34ed748234b465a72adb52b4f0dadf6930  crates/ckks-train-nn/src/trainer.rs
60f1dbf83e30ea59eee89e2a38889a9161fc04570493a13fff00e0a21ac4e9a2  crates/ckks-train-bench/src/study_admission.rs
f3379d136f4b0d59e32440752d89cdd6ad8762c4cd58342d1ba5f6d576a09154  crates/ckks-train-bench/src/study_admission/p4_fixtures.rs
e062bc2678a205431ea2fcd522cfb4351795bd375a7aa073c3204d12915813f8  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
```


## Source fix round1 — S1/S2 and Q1/Q2 (2026-11-07)

Status: **DONE_WITH_CONCERNS, compile/static only; independent re-review and runtime release pending.** This section supersedes the original report's prospective parameter-negative and diagnostic-coverage descriptions where they were too broad. Earlier report text is preserved as the historical source proposal; the original complete source/report/review pins also remain untouched under `task-54-p4-cpu-fix1-before/`. No runtime numerical, rejection, poison or counter result is claimed.

Root's exact fix release was read and hash-verified: `task-54-p4-cpu-fix1-release.md` SHA256 `1ed29b1e79154106f1252d0e5607eafbc9f93f9c9948b563fa46660b15f30868`. The full goal and original32-success+2-failure matrix remain unchanged. Only the released two cfg(test) files changed: `study_diagonal/p4_tests.rs` and `study_admission/p4_fixtures.rs`. All six other original source pins, including production client/session, NN trainer, lib declarations, P3 parent, and Signal oracle, remain byte-identical. No production/NN API/math, Cargo dependency, parameter/oracle/plan/trainer/boot fixture, tolerance, dispatcher, doc outside this appended report, preserved-before file or unowned file was changed. No Git mutation/network/subagent/runtime operation occurred.

**S1 correction.** The synthetic child now has the narrow `claim_for_actual_parameters` helper, which uses existing `encode_params` to encode the actual supplied borrowed PlanningParameters while retaining the original NI plan's identity/binding. The spy group passes the already-existing I parameters (with no bootstrap descriptor) against the already-existing NI-bound plan and supplies that self-consistent actual-I claim. Thus `claim.parameters == actual.bytes`, while `actual.bytes != self.parameters`; removing the actual-versus-bound discriminator would let this negative reach the same shared terminal factory and fail the exact-error/zero-call assertions. No extra parameter/key/proof/descriptor/geometry/initial/update constructor is added, no production override/digest authority is exposed, and canonical/D7 evidence is unclaimed.

The redundant second identity/binding-only negative was replaced, keeping exactly16 isolated negatives: **1 identity +1 actual supplied-parameter mismatch +7 malformed claim-codec mutations +7 partition negatives**. The seven q/Delta/dnum/secret/p/N/descriptor mutations change only claim bytes and are now explicitly described as malformed-claim rejection evidence, not seven experiments changing actual supplied cryptographic parameter objects. Positive factory count remains1; every negative expects the exact named error and0 factory calls. These remain proposed unexecuted assertions.

**S2 correction.** New test-owned `ReadOnlySessionSnapshot`/exact comparison helpers capture actual raw current/reference/momentum/current-weight/bias/reference-helper factors, encrypted W/b, every I velocity diagonal and every NI compact weight/Option velocity entry and complete container layout/cardinalities. Ciphertext equality checks complete component count, level, exact scale bits, basis Arc identity, NTT flag and every RNS limb, with full outer/inner lengths before zip. They also compare successful update/sample totals, measured counters, poison/failure/trace state and BatchSpec. They do not decrypt, evaluate, encode, call a planner, reset an optimizer, or reproduce a recurrence. NI reference factor remains the already-approved existing `with_lr(reference).grad_scale()` helper computation; literal internal masks/kappa observation is still unclaimed.

The existing first-update malformed-packet group is preserved and strengthened: one complete encrypted/config snapshot is compared after every rejection. The same helper strengthens every existing32 success-replay/exhaustion rejection and both poisoned retries. In exactly two already-planned scheduled sessions (I and NI-AU), update1 now first rejects a missing-stage-target packet with the exact `stage target cardinality` error. The helper explicitly proves successful index1, nonempty present velocity, and every current raw rate differing from the next .08 rate (the retained prior rate is .04). It compares all encrypted/config/progress/counter state before/after, asserts the session remains usable, restores the original target, and performs the same existing valid update. Moving the setter or a velocity reset ahead of validation would now be detected. No valid trainer attempt, plan, oracle, setup or bootstrap is added.

**Q1 correction and inspected limitation.** Both original poison attempts assert `FailedUpdate.update == 0`, trainer phase, exact nonzero partial seven-counter delta and the one real successful refresh/bootstrap prefix. For I, they additionally require a nonempty failed trace exactly equal to the current exposed attempted trainer trace: the existing I trainer publishes `self.trace = trace` before `refresh_all` (`trainer.rs:1366`). For NI, the failed trace and current exposed trace are both required to be empty, because the existing NI trainer publishes trace only at final commit (`trainer_ni.rs:2164`) and the injected second-bootstrap refusal occurs before that commit. This is the precise source-supported reason for absence; no hidden NI partial trace/observer or full typed P3 trace equality is claimed. Poisoned retries compare the entire retained failure payload and encrypted state unchanged. The same2failed attempts are reused.

**Q2 correction.** The foreign-basis packet negative executes only for I: level9 fits the other NI context, whereas NI level15 exceeds the other I max9, so that branch is skipped for NI. The source now says this explicitly beside the conditional. Eight intended successful I sessions exercise that shared-guard negative; no separate NI foreign-basis negative execution, new basis fixture or scientific per-mode evidence is claimed. Both modes still call the same production `check_ct` basis guard for their checked entries.

### Fix1 validation evidence

No test binary was invoked. Every Cargo command retained the exact original no-run/Clippy command forms and environment, checked free disk >=805306368 bytes first, used the existing target, recorded all8 before/after source pins plus process PID/tool handle/start/end/exit/raw output, and polled the same handle to terminal. Final Clippy and no-run hashes agree. Formatting touched only the two assigned cfg(test) files with skip_children=true. Raw/JSON records are in `task-54-p4-cpu-fix1-validation/`.

| Evidence prefix | Result | UTC start / end | Handle |
|---|---|---|---|
| `01-rustfmt` | Exit0, assigned two files only | 2026-11-07T06:44:41Z / 2026-11-07T06:44:42Z | `96330` |
| `02-compile` | Exit101, missing explicit Diagonals<Ciphertext> test-helper type arguments; corrected | 2026-11-07T06:44:42Z / 2026-11-07T06:44:44Z | `96330` |
| `03-rustfmt` | Final assigned-file formatting exit0 | 2026-11-07T06:45:35Z / 2026-11-07T06:45:35Z | `67933` |
| `04-clippy` | Strict scoped Clippy exit0, no warnings | 2026-11-07T06:45:36Z / 2026-11-07T06:45:39Z | `67933` |
| `05-compile-final` | Focused ignored-test no-run exit0, no warnings | 2026-11-07T06:46:02Z / 2026-11-07T06:46:16Z | `3955` |
| `07-diff-check` | Assigned-file static diff check exit0 | 2026-11-07T06:46:51Z / 2026-11-07T06:46:51Z | `completed direct command` |

`06-static-pins.json` verifies that only the released two cfg(test) files differ from the preserved pre-fix source and that the original report remains a byte-identical prefix. `source-final.sha256` records all8final source pins below. Original compile RED evidence stays preserved; fix1's intermediate compile error is an ordinary corrected generic-type error, not a new runtime RED or rejection result.

Exact Cargo commands executed:

```text
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

### Updated exact future inventory (prospective, not runtime authority)

All original expensive-operation counts stay unchanged:20sessions (I9/NI11);20geometry+20initial+34update planner calls/0handoffs under28/28/42/0 caps;18Signal oracles/32actual Engine updates;34trainer attempts with32intended successes+2failures;2params/2KeyGenerators/1proof/1descriptor/1Bootstrapper/22Encoder constructions; actual bootstrap invocation cap85;0canonical materializations/production starts. The same single ignored lib test covers the full matrix, with no extra test binary or warmup.

| Diagnostic/rejection work | Exact intended total after fix1 |
|---|---|
| Shared terminal factory spy |1positive call and16negative calls each expecting0factory invocations;1actual-I-vs-bound-NI discriminator replaces the redundant binding-only negative |
| First-update malformed packet rejects |228:18groups x12unconditional checks +4two-micro reorder checks +8I-only foreign-basis checks; all original checks retained |
| Later changing-rate rejects |2: existing scheduled I/AU sessions at update1 (.04->.08), velocity present; no extra valid trainer attempt |
| Success replay/exhaustion rejects |32, now checked against full raw/encrypted state snapshots |
| Poisoned retries |2, now checked against full raw/encrypted/failure state snapshots |
| Test-owned encrypted-state snapshot captures |54 =18first-update groups+2later-rate checks+32replays+2poisoned retries; guard checked immediately before capture |
| Exact unchanged assertions |264 =228first-update rejects+2later-rate rejects+32replays+2poisoned retries; guard checked immediately before assertion |
| Read-only optimizer-configuration collection |318 prospective helper calls =54captures+264comparisons; no HE work/decryption/encoding |
| Encrypted model snapshot copies |122 =68existing explicit comparison snapshots+54new test-owned encrypted-state snapshots; these are encrypted copies, not new decrypt endpoints |
| Client decrypt/inspect endpoints |68unchanged =32paired success endpoints+36first-update before/after endpoints |
| Existing BatchSpec checks |52unchanged =20initial+32successful post-step checks, using existing fork-isolated probes |
| Invalid client label/cardinality requests |36unchanged, before encryption |

The new54/264diagnostic limits are checked and asserted exactly in the ignored test; all expensive-operation guards/caps and frozen I1e-5/NI4e-4 tolerances are retained. Snapshot copies add only test-owned transient encrypted comparison storage; this is not a complete memory/RSS admission claim. Every number above is an inspected source inventory to be reviewed/frozen by root, not a measured result. The original report's68client endpoints remain68; the new internal encrypted copies are counted separately to avoid conflating copying with client decryption.

### Fix1 final all8 source SHA256

```text
4227bff4f06db0dbe61a2d97e4a20986209347b8f15e74d8db852b16cdad0950  crates/ckks-train-bench/src/study_client.rs
44107c8617582883558e79dbc9cb37abefd2502d3c3ceaf6f361953c29f9102f  crates/ckks-train-bench/src/study_diagonal.rs
3e965739db8df3fa1d014d2ce0218638f933533e0c61d17ddb41e27223adc9aa  crates/ckks-train-bench/src/study_diagonal/p4_tests.rs
9b001769d71954a225df80dcbfcc180301bd347959b00f620abc6d40d384863a  crates/ckks-train-bench/src/lib.rs
4c8dc4f2d30d6a52ad9b5c9f6218eb34ed748234b465a72adb52b4f0dadf6930  crates/ckks-train-nn/src/trainer.rs
60f1dbf83e30ea59eee89e2a38889a9161fc04570493a13fff00e0a21ac4e9a2  crates/ckks-train-bench/src/study_admission.rs
87f45887cc5d400765e0295165d2cf8479724168f0002e1f1c66301a159e554c  crates/ckks-train-bench/src/study_admission/p4_fixtures.rs
e062bc2678a205431ea2fcd522cfb4351795bd375a7aa073c3204d12915813f8  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
```


## Source fix round2 — F1 diagnostic inventory correction (2026-11-07)

Status: **DONE_WITH_CONCERNS, source/static only.** Root's exact release was read and SHA256-verified: `task-54-p4-cpu-fix2-release.md` = `bc51763bf3d7c48156d42f85eac6ce855573dc9be23f2d47ee056ae8a13dddf5`. No runtime was invoked; this is a source-derived correction of an analytical count, not a measured failure/result or a standalone runtime release.

The previous fix1 inventory missed the four scheduled PolyFourRate sessions' first-update two-micro reorder checks. There are eight such reorder checks: four SquareC10/PolyC10 smoke sessions in I/NI plus four PolyFourRate sessions in I/WN/AU/AA. All eight remain. Correct intended counts are:

| Read-only diagnostic | Correct prospective count |
|---|---:|
| First-update unchanged assertions |18 x12 +8 reorder +8 I-only foreign-basis =232 |
| Later changing-rate assertions |2 |
| Success replay/exhaustion assertions |32 |
| Poison-retry assertions |2 |
| Total unchanged assertions |268 |
| Encrypted-state snapshot captures |54, unchanged |
| Optimizer-configuration collections |54+268 =322 |
| Client decrypt/inspect endpoints |68, unchanged |
| Encrypted model snapshot copies |122, unchanged |

Only the assertion guard and final expected assertion in `study_diagonal/p4_tests.rs` changed from264 to268. Inverting exactly those two literal replacements reproduces the previous fix1 pinned source hash, confirming no other source logic/comment/format change. The other seven source pins are unchanged. Formatting was unnecessary for two same-width literal replacements. No assertion, reorder case, matrix case, numerical tolerance, expensive-operation guard/cap or other production/source file was removed/changed. The existing20sessions,20geometry+20initial+34update planner calls/0handoffs,18oracles/32Engine calls,34trainer attempts (32intended successes+2failures),2params/2keys/1proof/1descriptor/1Bootstrapper/22Encoders and actual bootstrap cap85 are unchanged. S1/S2 and the Q1 trace-availability/Q2 I-only basis boundaries from fix1 remain in effect.

The original report and fix1 section/evidence remain historical prefixes. This section supersedes their mistaken228/264/318diagnostic counts with232/268/322; no historical evidence was rewritten. All read-only diagnostics and expensive counts remain prospective/unexecuted, subject to root's independent freeze/re-review and separate runtime authorization.

### Fix2 terminal static evidence

New evidence directory: `task-54-p4-cpu-fix2-validation/`. Both Cargo invocations checked free disk >=805306368 bytes, used the exact original CARGO_INCREMENTAL=0/CARGO_BUILD_JOBS=8/RAYON_NUM_THREADS=8 environment and current target, captured all8before/after pins, PID/tool handle/start/end/exit/raw output and polled the same handle to terminal. No test binary was executed. Final no-run/Clippy pins agree. `03-static-pins.json` and `source-final.sha256` retain the complete final inventory; `source-before.sha256` preserves the fix1 pins and `report-prefix.sha256` preserves the pre-append report hash.

| Evidence | Result | UTC start / end | Handle |
|---|---|---|---|
| `01-compile-final` | Focused no-run exit0, no warnings | 2026-11-07T06:52:00Z / 2026-11-07T06:52:14Z | `22127` |
| `02-clippy` | Strict scoped Clippy exit0, no warnings | 2026-11-07T06:52:27Z / 2026-11-07T06:52:30Z | `43514` |
| `04-diff-check` | Static assigned-file diff check exit0 | 2026-11-07T06:52:52Z / 2026-11-07T06:52:52Z | `completed direct command` |

Exact Cargo commands (once each on amended source):

```text
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

No runtime/HE/F64/planner/parameter/key/proof/bootstrap constructor invocation, network, subagent, Git staging/commit/push or unrelated edit occurred. Source/Cargo ownership returns to root after this terminal verification; independent source review/runtime parity/full P4/P3/D7/Task54 completion remain unclaimed.

### Fix2 final all8 source SHA256

```text
4227bff4f06db0dbe61a2d97e4a20986209347b8f15e74d8db852b16cdad0950  crates/ckks-train-bench/src/study_client.rs
44107c8617582883558e79dbc9cb37abefd2502d3c3ceaf6f361953c29f9102f  crates/ckks-train-bench/src/study_diagonal.rs
4f858727aa89053518ba94cd1120a855c4c80c64a866ccb76ebc8e79de63e40d  crates/ckks-train-bench/src/study_diagonal/p4_tests.rs
9b001769d71954a225df80dcbfcc180301bd347959b00f620abc6d40d384863a  crates/ckks-train-bench/src/lib.rs
4c8dc4f2d30d6a52ad9b5c9f6218eb34ed748234b465a72adb52b4f0dadf6930  crates/ckks-train-nn/src/trainer.rs
60f1dbf83e30ea59eee89e2a38889a9161fc04570493a13fff00e0a21ac4e9a2  crates/ckks-train-bench/src/study_admission.rs
87f45887cc5d400765e0295165d2cf8479724168f0002e1f1c66301a159e554c  crates/ckks-train-bench/src/study_admission/p4_fixtures.rs
e062bc2678a205431ea2fcd522cfb4351795bd375a7aa073c3204d12915813f8  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
```


## Runtime1 actual failure and source fix round3 — explicit synthetic reference policy (2026-11-07)

Status: **DONE_WITH_CONCERNS, fix3 source/static only.** Runtime1 was actually executed by root and failed before keys; no runtime2 or automatic retry was invoked by this implementer. Earlier source-only statements remain preserved historical phase records, not a claim that runtime1 never occurred. Root's fix3 release was read and SHA256-verified: `task-54-p4-cpu-fix3-release.md` = `bd82713b9aaa477e7939a28816e02b967d81b48c5eeb7317e3d901fa34941b02`.

### Actual runtime1 RED, retained unchanged

The immutable `task-54-p4-cpu-runtime1-validation/output.raw.txt` and `terminal.json` were read. Raw output is exactly1965bytes with SHA256 `f1d14696c5cc7e4daeee089db83a34017b8b306310bd8535b759d32d8d4ef242`. Root's runtime controller handle44985 and terminal custody record show test/controller exit101; local controller PID92480, Cargo PID92481. The terminal's before/after all8source pins agree and match the fix2 source pins. These runtime1 files were not edited.

Exactly one test ran:0passed,1failed,190filtered out. Six Square metadata plans completed; the seventh case, PolyC3 I, panicked while unwrapping `AdmissionIssue { kind: RateConflict, operation: "full maximum reference", ... }`. Its closed recipe has current/kernel .04 and explicit synthetic reference .08. The private synthetic builder incorrectly called canonical `super::rates`, whose full-plan policy requires reference equal to max(full scheduled kernels), so it rejected .04 != .08. This was a fixture-policy error, not evidence of failed CKKS numerical precision.

Actual partial work is **2CkksParams +1DftSupportProof verification +1BootstrapMetadata;7geometry +7initial-state calls +6update plans**. The raw log's last printed cumulative plan row is6/6/6; seventh geometry/initial completion is established by the exact later rate-policy error and inspected build ordering, not by inventing a seventh printed success row. Actual KeyGenerator, Encoder, Bootstrapper, Engine, trainer, HE, factory spy and session work are all0: the failure occurred while assembling plans, before the setup/spy/session paths. No smoke numerical/cryptographic success or canonical pre-key/D7 acceptance is claimed.

### Assigned correction and pre-parameter regression

Only the released cfg(test) `study_admission/p4_fixtures.rs` and `study_diagonal/p4_tests.rs` changed. The actual synthetic builder now calls a thin private `encode_synthetic_rates` wrapper, whose sole operation is delegating byte encoding to existing parent `encode_rate_schedule_fields`. It retains the original explicit reference .08 and single consumed .04 rate. It does not call/loosen canonical full-maximum validation, copy a schema/optimizer formula, raise/lower a rate/reference, append a fabricated future suffix, or construct a canonical synthetic plan. Existing closed SignalRecipe and actual P2 initial/update validators continue to own numerical/config validity. All six other original source pins, including parent P3 canonical rates(), production bridge/client, NN trainer and Signal oracle, are byte-identical.

A single small `explicit_reference_policy_regression` is called at the beginning of the existing orchestrated ignored test, before all parameter/setup constructors. It uses the existing closed PolyC3 recipe and explicitly checks its single .04 kernel bits and .08 reference bits. It invokes unchanged canonical rates() once and requires exact typed `RateConflict`/`full maximum reference` rejection with no rate bytes written. It invokes the exact private synthetic wrapper used by build once and requires the original current/reference bit patterns in the shared codec's bytes, without field-offset/schema duplication or an independent decoder. A local guard logs/asserts exactly2policy/encoding invocations in this regression. No additional parameter/geometry/initial/update/Engine/trainer/HE call or test group/binary is added. The regression is compiled but **not executed** during fix3; runtime1 is the recorded RED, and runtime GREEN remains unclaimed.

### Fix3 terminal source/static evidence

New evidence directory: `task-54-p4-cpu-fix3-validation/`. Both Cargo invocations checked free disk >=805306368 bytes, used the exact original environment/current target, retained all8before/after source pins, PID/tool handle/start/end/exit/raw output, and polled the same handle to terminal. Final no-run and strict Clippy source pins agree. Only the two assigned files were formatted with skip_children=true. `04-static-pins.json` also re-verifies runtime1 immutable bytes/custody/source pins and the six unchanged source files.

| Evidence | Result | UTC start / end | Handle |
|---|---|---|---|
| `01-rustfmt` | Assigned two-file formatting exit0 | 2026-11-07T07:00:38Z / 2026-11-07T07:00:38Z | `2583` |
| `02-compile-final` | Focused ignored-test no-run exit0, no warnings | 2026-11-07T07:00:38Z / 2026-11-07T07:00:52Z | `2583` |
| `03-clippy` | Strict scoped Clippy exit0, no warnings | 2026-11-07T07:01:16Z / 2026-11-07T07:01:19Z | `36174` |
| `05-diff-check` | Static assigned-file diff check exit0 | 2026-11-07T07:02:16Z / 2026-11-07T07:02:16Z | `completed direct command` |

Exact Cargo commands (compile/static only):

```text
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

### Revised future inventory and unchanged holds

The only added prospective work is the small pre-parameter policy regression:1canonical policy invocation that intentionally rejects before rate encoding and1actual synthetic-wrapper shared-codec encoding. Its local cap is2, with0new expensive calls. The intended20synthetic plan builds each continue to encode their own bound schedule once through that same shared field codec; no future rate/update is fabricated.

All original matrix/expensive-operation limits stay unchanged:20sessions (I9/NI11),20geometry+20initial+34update planner calls/0handoffs under28/28/42/0caps,18Signal oracles/32Engine updates,34trainer attempts with32intended successes+2failures,2params/2keys/1proof/1descriptor/1Bootstrapper/22Encoders, actual bootstrap invocation cap85,1positive terminal spy/16isolated negatives, fixed I1e-5/NI4e-4 tolerances and0canonical dataset/production starts. Fix2's corrected diagnostic inventory stays232first-update comparisons,268total unchanged comparisons,54captures,322configuration collections,68client decrypt endpoints and122encrypted snapshot copies. No expensive cap or tolerance was increased, and no rejection coverage was removed.

These remain prospective full-matrix counts; runtime1 only reached the partial setup/planner counts above. Independent re-review, separate runtime2 release, numerical/branch/counter/poison parity, genuine canonical/D7 start ownership and full Task54/P4 completion remain pending. No source outside the two assigned test files, Git mutation, network/subagent/cache operation or runtime retry occurred. The prior report prefix and runtime1 failure evidence are preserved. Source/Cargo ownership returns to root after this terminal evidence.

### Fix3 final all8 source SHA256

```text
4227bff4f06db0dbe61a2d97e4a20986209347b8f15e74d8db852b16cdad0950  crates/ckks-train-bench/src/study_client.rs
44107c8617582883558e79dbc9cb37abefd2502d3c3ceaf6f361953c29f9102f  crates/ckks-train-bench/src/study_diagonal.rs
ceebab9bba7b2e4705ed9867f2b0ed411c377a2ee6fab1b0894a78317a1610a4  crates/ckks-train-bench/src/study_diagonal/p4_tests.rs
9b001769d71954a225df80dcbfcc180301bd347959b00f620abc6d40d384863a  crates/ckks-train-bench/src/lib.rs
4c8dc4f2d30d6a52ad9b5c9f6218eb34ed748234b465a72adb52b4f0dadf6930  crates/ckks-train-nn/src/trainer.rs
60f1dbf83e30ea59eee89e2a38889a9161fc04570493a13fff00e0a21ac4e9a2  crates/ckks-train-bench/src/study_admission.rs
c2426266444120b084ed3351696f0a7b42376f09b24b1589f5642e98ac71c1f7  crates/ckks-train-bench/src/study_admission/p4_fixtures.rs
e062bc2678a205431ea2fcd522cfb4351795bd375a7aa073c3204d12915813f8  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
```
