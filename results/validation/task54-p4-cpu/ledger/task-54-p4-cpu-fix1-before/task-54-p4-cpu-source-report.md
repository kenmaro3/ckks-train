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
