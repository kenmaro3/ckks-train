# Packed refresh, NI and old-forward local blocks (Task54 D4c)

Status: **root-ratified and independently reviewed design; implementation separately gated**. This refines the root-ratified [packed training design](packed-training-design.md), following reviewed D4b2 `0388282` as supplied by the assignment. Source was read on 2026-11-06; no code, test, HE computation, key generation, dataset, remote host or performance measurement was run for this design. Names below are proposed APIs. Approval of this document alone does not authorize implementation or experiments.

## 1. Recommendation and source gaps

Implement three separately assigned/reviewed packages: c1 typed refresh/projection and interactive stage orchestration; c2 a metadata-only DAG, fixed AA seam and single-stage NI; c3 old-forward local blocks and forward-only inference. Keep the existing two learning functions, fixed-reference NAG, source n16 presets, common geometry, logical sample counts and losses. No diagonal containers, arbitrary callbacks inside the learning graph, velocity reset, range adjustment or preset substitution are needed.

| Current source | Reusable behavior and precise missing boundary |
|---|---|
| `nn/src/packed_training/optimizer.rs`, `PackedLayerState::{new,request,prepare_traced}` | Private `Arc<()>` generation and immutable policy protect bound gradients. `new` always sets U absent/index0; `prepare_traced` always advances the update. Neither reconstructs refreshed W/U while preserving policy/index. Public replacement via `new` would silently erase momentum history. |
| `packed_training/stage.rs`, `PackedStage::{new,plan_step,prepare_step}` | Old state is borrowed until a complete candidate exists. `plan_step` reads real ciphertext metadata and performs host encodings; it is not reusable on hypothetical refreshed/future states without ciphertexts. `minimum_state_level` reports the current candidate only and does not enforce projection reserve. |
| `stage.rs` hidden activation followed immediately by second affine | AA must intervene here, and the post-bootstrap value must also feed the second matrix outer product. Returning a bootstrapped receipt after the update is too late. Current receipt has only one hidden vector list; keep pre/post values distinct. |
| `stage.rs` private `History`, `Planner::{product,mask,affine,activation,derivative}` | Actual q/F64 order and exact-target encodings already modeled. Extract metadata inputs/outputs and share arithmetic, without replacing the independent runner or adding an HE probe. Current stage has no forward-only method. |
| `packed_training/types.rs` | Context, two components, finite scales, vector specs and exact-Delta weights are checked. Vector construction permits natural scales, so refresh requires an additional exact-Delta check. These checks assert representation; they cannot prove encrypted support/ranges. |
| `nn/src/refresh.rs`, `Refresher` | Legacy periodic diagonal-layout binding and scale-preserving/relabel path are unsuitable for row/column matrix roles. Reuse the client transport idea, not its `Layout` binding or `BootstrapRefresher` scale relabeling. |
| `ckks/src/ops.rs`, `BootstrapOps` / `HostBootstrapper` | Trait exposes bootstrap and output level only. Host adapter already merges host delta into device counts exactly once after upload. Generic trait has no context/count/key descriptor. |
| `ckks/src/bootstrap/bootstrapper.rs` | Constructed `Bootstrapper` exposes params, BootParams, keys, output level and **all fields** of `cost_per_boot()`. This is a pure accessor after construction, not pre-key admission: construction encodes DFT tables and generates keys. |

Paths abbreviated above have prefix `crates/ckks-train-`. No edits to existing diagonal trainer/optimizer/block files are proposed. [D2c1 §2/P1](study-diagonal-admission-design.md) proposes CKKS-owned pure scale and `BootstrapMetadata` primitives; consume that reviewed shared implementation when available rather than independently introducing a second pre-key bootstrap cost algorithm.

## 2. Typed refresh and immutable reconstruction

Use private validated packets, with accessors rather than public mutable fields:

```text
RefreshRole = Weight(layer) | Velocity(layer) | Bias(layer) | Hidden(stage, record)
RefreshShape = Matrix(LayerGeometry) | Vector(VectorSpec)
PackedRefreshRequest<C> = { private transaction/role/shape, ct, output_contract }
PackedRefreshResponse<C> = { same identity, ct }
PackedClientRefresh<E>::refresh(request) -> Result<(response, ClientRefreshTiming)>
PackedClientRefresh<E>::output_contract() -> { context, level, exact_delta }
PackedRefreshCandidate<C>::into_stage() -> PackedStage<C>
```

The server packet contains encrypted canonical state only, never gradients, plaintext model arrays, a requested new optimizer policy or a rescaling coefficient. A refresh batch fixes order W0,U0 if present,b0 if present,W1,U1 if present,b1 if present. Validate every input and output role exactly once; reject missing, extra, duplicated or cross-layer/transaction results. A hidden packet additionally binds valid-record identity. The response cannot set an update index, NAG reference or generation token.

Add narrow **module-private** constructors in optimizer/stage:

```text
PackedLayerState::with_refreshed_ciphertexts(ev, W, optional_U) -> Result<Self>
PackedStage::with_refreshed_parts(ev, layers, optional_biases) -> Result<Self>
```

They borrow the original candidate/state, preserve its exact policy, reference, momentum, geometry and completed-update index, require the same U presence and bias domain, validate all replacement ciphertexts, and allocate fresh `Arc<()>` generation tokens. Refresh is not an optimizer update. A request/gradient bound before refresh must fail against reconstructed state even if layer/index are unchanged. Both matrix generations rotate; do not copy an old Arc or use numerical index as freshness. Stage reconstruction cannot call `PackedStage::new` (which resets both matrices). No setter, public restore-with-arbitrary-U, or serialized-token API is required.

Client service implementation belongs on the client side. It may decrypt updated W/U/bias or old hidden values, extract logical coordinates, restore structural zeros and repack at exact Delta/declared level. Matrix coordinates follow `LayerGeometry::pack_weights_out_by_in`; vector canonicalization uses arithmetic mean across the repeated copies of each logical coordinate, then `VectorSpec::pack`. Averaging redundant copies estimates the same coordinate; it is not a model update. It must not clip, normalize active values, change U's fixed-reference units, inspect gradients or compute optimizer updates. Record download, decrypt/decode, canonical repack/encode/encrypt and upload separately, including bytes. In-process fixtures may own client secrets outside the server object and explicitly demonstrate role separation, not process isolation.

**Transaction:** prepare the encrypted stage update, prepare every refreshed part while retaining the old stage, validate the entire result, then return a single replacement candidate. Any failure discards the candidate and preserves the caller's old stage/index/generations. Already completed HE counts and client calls/time are not rolled back. Return a failure receipt with phase/role, completed event list and counter delta; it is not a successful checkpoint. Consume candidates by value for commit. Do not perform an optimizer update twice to retry a failed refresh.

Interactive mode refreshes all updated state after every stage update and hidden values at each local boundary. Its client can refresh level0 state directly; the NI support-projection reserve below is not an artificial requirement on client decrypt/repack. The I output contract is checked separately from NI bootstrap output. Secure I remains source `n16_train`; no deeper custom preset is introduced to admit it.

## 3. NI projection, range and bootstrap capability

For **state** bootstrap, use a role-specific support projection before level reduction:

```text
plan_project(role, (l,S)) -> projected(l-1,Delta), boot_input(0,Delta)
run_project(ev, encoder, typed_state) -> private PackedBootInput<C>
```

Require matching context/slots/shape, two components (degree1), S exactly Delta, l>=1. Weight and U use the layer active-entry matrix mask; bias uses the full model-output vector mask. Plain scale is `exact_product_scale(l,Delta)/S`, using the actual q[l] search/rounding. Validate encoding and exact output before work. Execute one ct-pt multiplication and one rescale, then free mod-switch to0. Never relabel a natural scale or use free mod-switch as the missing projection. Projection failure at state level0 is unrecoverable in NI and must be prevented by admission.

Canonical hidden activation already ends in its support-mask/rescale at exact Delta. Both the AA seam and WN/AU local handoff use that completed normalization, then mod-switch directly to0 and bootstrap. **No additional hidden projection is required**: it would charge the same support normalization twice. AA handoff reuses the exact post-bootstrap vector and adds neither normalization nor bootstrap. Keep this distinction explicit in types (`CanonicalHidden` vs state requiring projection) and independent count tests. Bootstrap error may reintroduce small padding error; neither state nor hidden outputs are claimed to have mathematically exact zeros after bootstrap. Subsequent canonical masks and diagnostic padding tolerances handle that documented approximation; adding a post-bootstrap mask would change output levels and must not be hidden in a wrapper.

Each input has a public role-specific range contract `|slot|<=1` covering **all slots**, including padding/error. Metadata validation cannot enforce it. Record range obligations separately from structural admission; test diagnostics may decrypt outside the NI object to measure actual ranges and precision. A violated bound is failure/unverified evidence, never permission to clip, reset U, scale by secret maxima or tune production inputs/rates. Mathematical synthetic small weights/inputs may be chosen independently of dataset observations for correctness fixtures.

Propose a packed-local `PlannedBootstrap<E>: BootstrapOps<E>` capability returning immutable public `PackedBootstrapContract { parameter identity, output(level,Delta), per_boot: OpCounter, provenance }`. Implement it for CPU Bootstrapper from its existing accessors. A small `PackedHostBootstrap` wrapper owns references to CPU Bootstrapper and host evaluator, delegates arithmetic to existing `HostBootstrapper`, and exposes the same CPU contract. No need to add mandatory methods to every existing BootstrapOps implementor. External generic implementations must provide a checked contract and are trusted to honor the trait just as in existing NN; generic Rust trait bounds cannot prove absence of a hidden decryptor. A supplied decrypt/re-encrypt mock is **not NI validation**, regardless of trait implementation.

Runtime constructed metadata is enough for c2 small CPU tests. Full pre-key counts/keys require D2c1 P1's shared `BootstrapMetadata`; verify equality with constructed CPU `cost_per_boot()` for the same params/BootParams. Do not claim that constructing Bootstrapper, dense DFT stages, encoded tables, or probing a zero ciphertext is lightweight pre-key admission. Until that dependency is implemented/reviewed, mark production pre-key admission `MissingCode(BootstrapMetadata)`. The packed worker owns no duplicate CKKS descriptor implementation.

Successful total count prediction is `stage_nonboot + state_projection_counts + event_count * configured_per_boot`, summed field-by-field with checked integer arithmetic. Canonical hidden mask counts are already inside the stage. Never add `event_count*111`, or add a bootstrap estimate to measured inclusive counters. The runner samples one outer evaluator delta; per-event snapshots are observations within that interval, not independently added again. HostBootstrapper already merges once; the wrapper must not merge it again. Failed host bootstrap/upload may leave host work unmerged per current contract; preserve server/host failure diagnostics rather than claiming globally rolled-back work.

## 4. Minimal metadata DAG and AA execution seam

Keep `PackedStage::plan_step` and `prepare_step` as no-AA wrappers with unchanged semantics. Extract a private `dag.rs` and add the following checked metadata forms:

```text
PackedStageMeta { geometry, config, real_classes,
  W[2], U[2] presence+history, bias[2] presence+history, update_indices }
PackedBatchMeta { ordered valid record identities, input histories, target histories }
PackedUpdateMeta { rate, valid_count, relin_policy, hidden_refresh }
HiddenRefresh = None | Bootstrap(immutable_contract)
plan_stage_dag(params, public_encoder, geometry, stage_meta, batch_meta, update_meta)
  -> PackedDagPlan { trace, counts, candidate_state_meta,
       hidden_pre_meta, hidden_post_meta, projection_plan }
```

The pure kernel takes no ciphertext, evaluator, secret key, RNG or execution callback. Its public Encoder is parameter-only: preserve existing host mask/activation encodings as a pure feasibility check, with no HE counter movement. This is not allocation-free planning. `ops.rs::mask_plain` currently calls `Encoder::encode`; `encoder.rs::encode_complex` checks rounded inverse-transform coefficients against Q/2. A scalar-only rounding bound cannot replace that mask check. Validate each actual factor/mask/history with the same encoder, release temporary plaintexts, and retain only metadata; an optimized analytic mask bound would require later equivalence proof. Reuse shared pure exact-target/scalar primitives from D2c1 P1 when available; meanwhile keep the existing evaluator-bound runtime planner honestly labeled as partial. Key availability/context checks remain in the runtime binding wrapper. Do not pretend extracted metadata alone proves that the supplied evaluator owns keys. No new arbitrary activation/loss combination is supported.

At the fixed seam after `activation("hidden",z1)`, record `hidden.pre_boot`, lower its canonical ciphertext to0, call the planned bootstrap exactly once per valid record, validate returned context/spec/components/output level/Delta and record `hidden.post_boot`. The second affine and its outer-product gradient consume **post** hidden. First-layer derivative still uses **old z1**; bootstrap is treated as approximate identity for the learning graph, not a new differentiable function. Both pre/post snapshots refer to pre-update weights. Preserve `receipt.hidden` as the actual second-layer input for no-AA compatibility; expose explicit `StageForwardSnapshot { hidden_pre, hidden_post: Option<_>, output, valid_ids }` for new callers, with a canonical accessor choosing post when present. Store/clone the post ciphertext, not a second bootstrap or re-evaluated hidden.

Plan outputs contain full candidate W/U/bias histories, not only their minimum, so future state can be simulated without constructing fake ciphertexts. Plans bind immutable config, generation identities for executable current-state plans, record/order metadata, parameter identity and exact F64 values. Rebind/check at execution; a plan for another generation cannot silently execute. Hypothetical next-state metadata contains no usable optimizer gradient token. The runner records actual output metadata after each primitive/refresh and compares complete trace and OpCounter before candidate release. Never substitute planned values for observed bootstrap histories.

### Exact arithmetic and structural witnesses

For aligned product level j: `S'=(S_a*S_b)/(q[j] as f64)`, output j-1. Preserve Rust F64 operation order. A normalization computes the exact-product-label search, then plain scale divided by source scale and performs its original encode checks; no ideal `q=Delta` substitution. `n16_boot` has Delta=2^42, compute levels1..16 with actual distinct42-bit primes, and bootstrap output16. Top31 has several different bit bands; all initial and resumed traces use actual q values.

For fresh W/U/bias/input/target at level16 (U absent on the first update), these are **structural examples**, not executed exact-scale admission:

| Boundary | eMLP WN/AU | Square/Square WN/AU | eMLP AA | Square/Square AA |
|---|---:|---:|---:|---:|
| z1, hidden before boot | 14,12 | 14,12 | 14,12 | 14,12 |
| hidden used by layer2 | 12 | 12 | 16 | 16 |
| z2, prediction | 10,10 | 10,8 | 14,14 | 14,12 |
| loss, dz2 | 9,9 | 7,6 | 13,13 | 11,10 |
| hidden upstream, dz1 | 7,6 | 4,3 | 11,10 | 8,7 |
| gradients h1,h2 | 4,7 | 1,4 | 8,11 | 5,8 |
| lowest state / projected state | 4/3 | 1/0 | 8/7 | 5/4 |

Thus no-AA stage costs are12/15 weight levels plus one projection level. AA changes the DAG; it is not merely an extra event appended to the no-AA receipt. Example exact eMLP no-AA product histories include `affine1.product=(15,Delta*Delta/q[16])`, its mask `(14,Delta)`, hidden square `(13,Delta*Delta/q[14])`, canonical hidden `(12,Delta)`; second affine product uses q[12]. AA replaces that second product by `(15,Delta*Delta/q[16])`. These expressions identify actual prime indices; numerical bit patterns and success are not claimed without the future metadata implementation.

First NAG at `lr<r`: two branches per matrix, same level, `h=lr*(1+mu)*sum/n`, `h_ref=r*(1+mu)*sum/n`; U'=h_ref, W'=W-h. No old-U scalar work. Later updates: `mu*U_old` and `kappa*U_old` at each gradient level require U.level>=gradient.level+1, add four ct-pt/rescales across two matrices, and use `kappa=mu^2/(1+mu)*(lr/r)`. Equal lr/r reuses one gradient branch. Every future branch must check nonzero scalar rounding, finite/plain scale and the runtime integer bound; adding a branch changes count, not critical depth.

Mixed levels matter. Following the fresh eMLP AA step, W/U can be at(8,11). With fresh input/target16, a second AA step has z1=6, hidden.pre=4, hidden.post=16, z2=9, loss=8, upstream=6, derivative1=5, dz1=4, gradients=(2,6). Old U levels(8,11) suffice, projection minima1. A third step must be simulated and will not inherit a generic "AA costs8" budget. Low target/bias levels can lower the loss/forward path independently; they are not repaired by state-only bootstrap.

## 5. WN/AA admission and secret-free sessions

```text
PackedNiTrainer<'a,E,B: PlannedBootstrap<E>>
  { &E, public Encoder/metadata, &B, immutable placement }
prepare_update(&old_stage, encrypted_records, update_meta, next_meta)
  -> Result<(PackedStageCandidate, PackedNiReceipt), PackedFailure>
NextStep = Known(checked_batch_and_schedule_metadata) | Unknown | Terminal
```

NI stores neither SecretKey, Decryptor, plaintext model/dataset, materializer nor client refresh callback. Evaluation/bootstrap keys (including SSE evaluation material) are allowed. Interactive session is a different type receiving a typed client capability, not an enum inside NI that can switch to decrypt/re-encrypt. New local CPU test harnesses keep decryption and plaintext references outside either NI session.

Admission proceeds before current HE work:

1. Simulate the complete current DAG and require every state selected for eventual NI refresh to retain at least one projection level. AA includes all current activation bootstrap contracts/counts.
2. AU always schedules one state event after this update. WN/AA simulate next **actual** batch/target/input metadata and schedule rate using the full candidate state, selected placement and a projection reserve at the end of that next step. Skip state refresh only if this next DAG succeeds with all reserve/scale/scalar checks.
3. If insufficient future state depth is repairable, plan all state projections and bootstrap outputs now, then re-plan that known next step from refreshed state. If it still fails (e.g. target/input level or scalar representability), return the specific gap **before current work**, not an unconditional "refresh fixes it". Invalid future rates/schedule identity are configuration failures, never a reason to mutate learning state.
4. If future metadata is Unknown, eagerly refresh after the current step and mark future admission unknown. This is a conservative placement choice, **not proof that every positive future lr is representable**. At the next call validate actual metadata before any HE work. Terminal is explicitly recorded; recommend conservative refresh there too for a restartable canonical checkpoint and consistent event accounting.

A complete bound schedule can be iterated purely over reachable metadata states before execution, including valid counts, known/unknown terminal policy, first-below-reference/equal/decreasing/restart rates, mixed inputs/targets/bias/U and orientation/loss branches. Include all actual registered finite rate/sample combinations; do not approximate an arbitrary continuum of positive rates by its endpoints (round-to-zero defeats that). Tests must demonstrate rejected unrepresentable future factors even when depth fits. Pure branch proofs are not encrypted tests; require literal independent trace/count cases in addition to planner-runner agreement. No production full-schedule proof is claimed here.

A state event after update1 or later is exactly `2 + 2*[NAG] + 2*[bias]` bootstraps: two W, two initialized U for eMLP, or two biases for Square/SGD. Both registered modes therefore have four per event, with different roles. Before first update U is absent and must not be fabricated. For n valid records and q stages, AA has n*q hidden events; WN/AU add n*(q-1) boundary events; AA has zero extra boundary events. State-event frequency derives from the actual carried plan. Invalid records contribute no event or denominator; partitions32+16 retain n48 and one update/rate/index per stage.

Distinguish `GraphLevelScaleAdmitted`, `RangeContractUnverified`, `PrecisionUnmeasured`, `MemoryUnadmitted`, `BackendUnavailable/MissingCode` and parameter/security binding. Structural success is not Ready, a secure-precision guarantee, or a memory-fit result. Source presets are not modified to make a case fit.

## 6. Old-forward local blocks and inference

`PackedBlockGraph<C>` owns ordered independent PackedStages on one PackedGeometryPlan; last stage is the output stack. A graph record binds one encrypted initial input, ordered encrypted per-stage targets, valid identity and public partition metadata. Validate all target representations, geometry adjacency, sample/partition totals and schedule metadata before work. A graph candidate privately owns every prepared stage and commits once; partial graph success is not a checkpoint.

For each stage s, compute its full old-state forward/backward and prepare a candidate. Pass its canonical **old** hidden to stage s+1 after typed handoff restoration (I client; WN/AU bootstrap; AA existing post-bootstrap). Later stage uses its own old W/head/U/target. Never feed a local classifier prediction or recompute hidden from candidate W. Preserve graph old state until all stage candidates/handoffs succeed, so sequential preparation is equivalent to first capturing old forward values then updating independent losses in reverse order. This transaction retains old/candidate state and hidden buffers; memory remains an explicit later admission problem.

Each stage accumulates all valid sample contributions and applies lr/n once. No group mean averaging; no backward derivative crosses the handoff. Independent local heads train on their own targets. Changing a later target cannot change an earlier update; changing a local-head target changes that stage's gradients, but not the old hidden supplied to the next stage during that same update. Changing a backbone weight changes downstream old inputs, as it should. Freeze those snapshots in the independent plaintext comparison.

Add `plan_forward`/`prepare_forward` for a backbone edge and a two-edge output stack, sharing affine/activation helpers but invoking no loss, outer sum, request or optimizer. Graph prediction traverses only each local backbone and the final stack; local classifiers are not evaluated. Forward-only refresh policy is an explicit inference plan (AA hidden bootstrap or required boundary restoration), separately counted/timed; it cannot silently invoke a training step, require targets or mutate generations. No inference accuracy/performance claim follows from this API design.

## 7. Packages, RED obligations and dependencies

All paths are proposed ownership for subsequent assignments; **none is authorized by this design work**. Tests below describe future RED intent, not executed results. Avoid broad legacy regression rewrites or dependency changes.

| Package | Exact proposed owned paths / narrow changes | RED and review gate |
|---|---|---|
| c1 projection, typed refresh, I | New `crates/ckks-train-nn/src/packed_training/refresh.rs`; narrow `optimizer.rs` private reconstruction, `stage.rs` reconstruction/candidate access, `mod.rs` exports; new `tests/packed_refresh.rs` | Bound gradients must be stale after refresh at same update index; U/reference/index preserved; altered/missing response role/context/scale/components rejected; state level0 projection rejected before counts, level1 reaches0 with exactly1 ct-pt/rescale; every row/column/vector slot mask; failures late in W/U/bias/client sequence preserve old stage. I multi-update numeric parity includes two biases and separate timing. Independent c1 review before c2. |
| c2 NI and AA metadata | New `packed_training/dag.rs`, `trainer.rs`; narrow `stage.rs` fixed seam/snapshots/extraction, `refresh.rs` planned bootstrap wrapper, `mod.rs`; new `tests/packed_training_ni.rs`, `tests/packed_plan.rs` | Absence-of-API RED separately from behavioral RED; future step consuming projection reserve must reject before current counts; first/later NAG and scalar-underflow future branch; actual q mixed levels; fresh12/15+projection literals, AA literals above; all OpCounter fields, lazy/per-sample, observed bootstrap output mismatch and Host single merge; genuine CPU bootstrap repeated W/U and biases. Needs reviewed c1 and shared P1 for complete evaluator-free/pre-key admission; runtime-only metadata is explicitly partial until then. |
| c3 independent local blocks | New `packed_training/block.rs`; narrow `stage.rs` forward-only helpers, `mod.rs`; new `tests/packed_blocks.rs` | Wrong target repr/partition/adjacency rejection; eMLP2/3 local-loss isolation and old-forward sequential/reverse equivalence; unequal2+1 synthetic and pure32+16 denominators; WN/AU per-record handoff, AA exact post-object reuse/zero extras; last-stage failure leaves entire graph old; inference never touches local heads/loss/optimizer. Requires reviewed c2. |

Small real-bootstrap fixtures should reuse the **construction pattern**, not claim legacy numerical results as packed evidence: `nn/tests/trainer_ni.rs::mini/fx` uses N=2^10 with `n16_boot_bits()`, scale42, dnum8, BootParams log_slots9 and real Bootstrapper, giving the needed16 compute output levels. Build the packed lattice rotation-key union for the new small geometries. The separate CKKS `bootstrapper.rs::mini` has only output level2 and is suitable for isolated projection/bootstrap tests, not repeated packed training. Existing D4b2 N=256 deep fixtures contain no real bootstrap and do not discharge NI validation. No secure-size key generation or ignored large tests in these packages.

Use original small analytic input/weight/target arrays, non-power-of-two widths, both orientations and active padded-class outputs. Include at least four scheduled updates (first below reference, decrease, intermediate/restart, reference), real U decoded separately against independent unscaled V, Square/SGD bias, old/post hidden, full-slot padding errors and bootstrap input maxima per role. Bootstrap numerical tolerances must be preregistered in the implementation brief based on established small bootstrap error plus operation propagation, not copied from no-bootstrap1e-7 tests or loosened after failures. Report actual range failures and precision gaps. A fake capability may test count/metadata failure paths only.

## 8. Root rulings and remaining proof boundaries

Recommended rulings: approve fresh-generation private reconstruction; canonical hidden consumes its existing mask before bootstrap and incurs no extra projection; unknown/terminal next metadata causes conservative state refresh while future admission remains explicit; approve packed-local planned capability with shared CKKS P1 dependency; preserve one graph-wide immutable candidate commit.

Unresolved proof/API work is concrete: P1's pure exact scalar/transform support descriptor is unimplemented; c2 must prove all reachable registered schedule histories and compare full costs to runtime; typed bootstrap capability cannot attest arbitrary implementation internals; bootstrap input ranges/production precision cannot be inferred from metadata; retained graph buffers have no memory proof. None warrants a secret fallback, modified preset, guessed count or reduced experiment coverage. D4d profile/backend/host timing and memory work, GPU validation, runner/campaign work and the explicit full-block user hold remain separate gates.

## 9. c1 concrete API (2026-11-06, recorded before implementation)

The authorized c1 surface is in `packed_training::refresh` (re-exported by
`packed_training`). The client owns decryption keys and RNG; the generic server
object stores only evaluator and public encoder references.

```rust,ignore
PackedRefreshContract::new(ev, output_level) -> Result<Self>;
PackedRefreshBatch::new(ev, &old_or_updated_stage, contract) -> Result<Self>;
batch.requests() -> &[PackedRefreshRequest<C>];
PackedRefreshResponse::new(ev, &request, ct) -> Result<Self>;
batch.complete(ev, Vec<PackedRefreshResponse<C>>) -> Result<PackedRefreshCandidate<C>>;
candidate.stage() -> &PackedStage<C>;
candidate.into_stage() -> PackedStage<C>; // infallible whole-state transfer

trait PackedClientRefresh<E: HeOps> {
    fn output_contract(&self) -> &PackedRefreshContract;
    fn refresh(&mut self, request: &PackedRefreshRequest<E::Ct>)
        -> Result<(PackedRefreshResponse<E::Ct>, ClientRefreshTiming)>;
}
PackedInteractiveStage::new(&ev, &encoder) -> Result<Self>;
server.prepare_step(&old_stage, &encrypted_records, rate, relin, &mut client)
    -> Result<(PackedRefreshCandidate<C>, PackedInteractiveReceipt<C>), PackedRefreshFailure>;
project_state(ev, encoder, &state_request) -> Result<PackedBootInput<C>>;
CanonicalHidden::new(ev, &backbone_layer, record_id, &canonical_hidden) -> Result<Self>;
hidden.request(ev, contract) -> Result<PackedRefreshRequest<C>>;
hidden.boot_input(ev) -> Result<PackedBootInput<C>>; // mod-switch only
```

`PackedRefreshRole` is `Weight(layer_ordinal)`, `Velocity(layer_ordinal)`,
`Bias(layer_ordinal)`, or `Hidden { stage, record }`. `PackedRefreshShape` is
`Matrix(LayerGeometry)` or `Vector(VectorSpec)`. Packet identity uses private
process-local Arc tokens, including a distinct token for each batch role. Public
read-only accessors expose role, exact shape, ciphertext and output contract.
Responses carry the request token privately, so another role/transaction cannot
be substituted even when level/shape match. This is an in-process protocol
binding, not a cryptographic authenticated transport or serializable token.
Batch completion requires exactly the ordered requests, rejecting permutations,
duplicates, missing and extra entries before any reconstruction. The contract
binds parameter Arc identity, output level and the parameter's exact Delta.

Private reconstruction borrows the original layer and accepts only checked
replacement W/U with identical geometry and U presence. Stage reconstruction
receives ciphertext parts (not arbitrary replacement optimizer states), preserves
bias presence/config/classes and invokes that narrow layer reconstruction.
Every refreshed layer receives a new generation, even for identity refresh at
the same index. `PackedStage::new` is not used for refresh.

The interactive receipt retains the complete updated stage read-only, the old
forward `PackedStepReceipt`, ordered successful refresh events and the single
outer evaluator delta. It does not re-run an update. Its updated snapshot and the
returned refreshed candidate allow subsequent code to compare boundaries without
opening mutation APIs. Failure records carry a phase, optional role, completed
events and measured counter delta when available; prior work is never rolled
back. Client timing fields are optional monotonic durations/byte counts; absent
transport/device timing remains `None`, not zero or fabricated GPU data.

Projection accepts only state-role requests and checks context, shape, two
components, exact Delta and level >= 1, then pre-encodes/uploads the support mask
before counted work. It performs exactly one ct-pt product/rescale and mod-switch
to zero. `PackedBootInput` exposes the ciphertext, role/shape and the explicit
unverified all-slot absolute bound 1. Hidden adapters validate backbone ordinal,
output VectorSpec and Delta, bind record identity, and mod-switch without another
mask. Constructors validate representation metadata; callers still assert that
canonical hidden values came from the completed stage activation mask. There is
no bootstrap or NI trainer in c1 and no numeric range attestation.

Validation uses an original insecure N=256 CPU fixture; no secure preset, GPU,
real-bootstrap precision, production timing or memory claim is made. Existing
stage/optimizer regressions are retained. Shared P1 exports are not required.

`PackedRefreshEvent::counter_delta()` exposes an optional evaluator delta sampled
around that client call and output validation. It is an observation nested inside
the outer interval, never summed into it. In-process decrypt/re-encrypt fixtures
must report zero HE work per successful client event; unavailable/reset counters
remain absent. This completes the per-role count evidence without inventing a
bootstrap or transfer measurement.

### c1 implementation evidence (2026-11-06, independently reviewed; c1 approved)

The six-path c1 patch implements the API above. The client reference exists only
in `crates/ckks-train-nn/tests/packed_refresh.rs`; production refresh code has no
SecretKey, Decryptor, RNG, plaintext optimizer or loss. The old stage is borrowed,
the updated stage is retained read-only in the receipt, and the refreshed stage
is a separate consumable candidate. No c2/c3 implementation is included.

Focused CPU validation passes: 7 refresh tests, 10 existing packed stage tests and
6 existing packed optimizer tests. Strict focused Clippy and owned-file rustfmt
checks pass. Tests cover both RE/CE and CE/RE stages, non-power-of-two widths,
active padded NAG outputs, encrypted labels and ignored invalid records. Four
rates `[0.04, 0.00008, 0.02, 0.08]` with valid counts `[3,1,2,3]` preserve independent
unscaled-V NAG, all W/U/bias values and old-forward outputs at absolute error below
`1e-7`. No tolerance, rate, production preset or learning rule was changed.

Per-role client events have zero HE counter delta; the inclusive I delta equals
the unmodified stage plan. Projection of W, initialized U and biases costs exactly
one ct-pt/rescale per role, and level1 reaches0; level0 fails before counters move.
Canonical hidden lowers without a second mask. Decoded full-slot checks include
structural zeros, vector-copy averaging, preservation of active values above1
(no clipping) and trained padded-class channels. Those above1 client fixtures are
not NI input-range evidence. Generation tests reject both old bound layer gradients
after same-index refresh. Wrong transaction/role, replay, missing/duplicate/extra
responses, context, scale, components and output level fail. Late backend W/bias,
client W/U/bias and response-replay failures preserve the old stage and report
completed events/observed work; counters and client side effects are not undone.

These are insecure N=256, deep-chain CPU learning/protocol tests. Client refresh
removes accumulated numerical error differently from a no-refresh continuation;
its only arithmetic on logical values is averaging redundant vector copies.
Persistent U is never reset or converted to another reference unit. There is no
real bootstrap, NI/AA admission, full schedule/profile proof, GPU/memory/transport
measurement, dataset run or scientific-performance claim. Complete run commands,
RED logs and frozen full patch hashes are in the Task54 c1 implementation report.


## 10. c2 concrete implementation contract (2026-11-06, before source edits)

Root assigned c2 after the reviewed c1/P1/parser/Linux-candidate dependencies.
The following freezes the bounded API choices; it does not release main HE execution.

* `PackedStageMeta` is immutable public metadata: parameter Arc, two exact layer
  geometries, config/classes, per-role `CtMeta`, and both completed-update indices.
  `new` constructs hypothetical metadata with no generation authority. Extraction
  from a checked `PackedStage` retains private existing layer generation Arcs.
  Every derived candidate/refreshed metadata value drops executable authority.
* `PackedRecordMeta` holds original record index and exact input/target histories;
  `PackedBatchMeta` binds the ordered unique valid indices, vector specs and update
  index. The runtime adapter uses original positions, including holes. Invalid
  records contribute nothing. `PackedUpdateMeta` fixes rate, GradRelin and the
  evaluator lazy flag. The pure kernel reads parameters and public Encoder only.
* `PackedDagPlan` privately retains the full primitive trace, all seven counts and
  candidate W/U/bias histories. The enclosing `PackedNiPlan` retains complete stage
  input and batch/update metadata.
  `plan_stage_dag` handles no refresh or one fixed hidden bootstrap contract.
  It shares the old stage recurrence, actual q casts and F64 expression order;
  it performs actual public mask encodings. It is neither allocation-free nor
  a proof that evaluation keys exist. Legacy no-AA calls remain wrappers.
* `PackedNiPlan` binds placement (AU/WN/AA), current DAG, next request and immutable
  bootstrap contract; `PackedNextStep` is Known(batch,update), Unknown or Terminal.
  Known must have index current+1. All current state keeps projection reserve.
  WN/AA first try carried candidate; only `NoLevelsLeft` may request repair, and
  the complete refreshed next DAG must pass before any current HE. AU also checks
  known next metadata against its refreshed state. Unknown/Terminal conservatively
  refresh and explicitly retain unknown/terminal future status.
* `PackedBootstrapContract::from_metadata(params, receipt)` consumes reviewed P1b
  conditional structural identity, level/Delta and complete cost; no invented
  per-boot constant. `PlannedBootstrap<E>: BootstrapOps<E>` supplies a checked
  contract and runtime binding. CPU Bootstrapper validates constructed cost,
  output and key support against P1b. `PackedHostBootstrap` delegates exclusively
  to existing HostBootstrapper; it NEVER adds or merges bootstrap counts itself.
  Runtime outer evaluator delta is inclusive; event deltas are nested observations.
  Generic implementors remain trusted and cannot be attested secret-free by a trait.
* `PackedNiTrainer` borrows evaluator, public Encoder and planned bootstrap only.
  `plan_update` extracts and binds current generations; `prepare_planned` rechecks
  them, actual encrypted records, evaluator configuration and capability before
  work. Hypothetical plans and plans from earlier refreshed/updated generations
  cannot execute. `prepare_update` composes those operations. No SecretKey,
  Decryptor, plaintext arrays, materializer or client-refresh callback is stored.
* The sole seam is after canonical first hidden, before second affine. It retains
  pre, lowers without another mask, calls bootstrap once, validates actual context,
  two components, exact level/scale, and retains post. That exact post vector is
  borrowed by affine2, later outer1 and receipt.hidden; derivative1 keeps old z1.
  `StageForwardSnapshot` exposes valid IDs, pre/post and output. Structural reuse
  is explicit in this data flow, beyond approximate numerical diagnostics.
* Receipt retains pre-refresh updated state, old-forward snapshots, per-event
  encrypted before/projection/input/output snapshots and inclusive counts. Candidate
  owns the complete final stage; failure exposes phase/role/completed events and
  optional observed count delta, never commits any stage subset. Completed HE and
  bootstrap side effects are not rolled back. Main diagnostics decrypt only outside
  NI, after returned observations, and stop on the first failed gate.

Root approved a narrow optimizer.rs ownership extension before editing: expose
existing `PackedOptimizer::coefficients` module-privately without changing its math,
guards or expression order, and a module-private read-only accessor to the existing
private generation Arc. This avoids duplicated policy validation or a second freshness
system. No public token, setter, serialization or arbitrary optimizer-state constructor
is introduced. Stage uses it for metadata binding; hypothetical metadata never carries it.

Returned graph-level/scale admission is conditional on P1b successful bootstrap
execution. EncodingQOverTwoFeasibility/internal EvalMod, encrypted RangeContractUnverified,
PrecisionUnmeasured, actual secret/key provenance, security, MemoryUnadmitted and backend
admission remain separate. This package has no Ready result, production graph/session
admission, c3 block trainer, dataset/profile binding, GPU or benchmark claim.

Concrete signature refinement before capability use: `PackedBootstrapContract::new(
Arc<CkksParams>, &BootParams, &DftSupportProof)` constructs P1b itself. The sketched
`from_metadata(params,receipt)` cannot independently bind every parameter/BootParams
field because P1b intentionally exposes no unchecked identity parser; recomputing the
checked receipt avoids duplicating canonical serialization. CPU binding compares the
constructed cost/output/key support; arbitrary third-party capabilities remain trusted.
No new CKKS-owned API is needed. The no-state-bootstrap branch retains its updated
snapshot and constructs the consumable candidate from identical encrypted parts via
c1's checked reconstruction (fresh generation, same policy/index), without HE or refresh.

### c2 bounded implementation disposition

The extracted graph kernel is evaluator-free but **uses Encoder** to check actual
mask/activation encodings. Its stage inputs include exact per-role CtMeta and original
valid record positions. Executable NI plans bind the original private layer generation
Arcs; all derived future states drop that authority. CPU capability binding checks
actual relin/Galois/conjugation/SSE key basis identity, level, digits and NTT form using
public evaluation-key accessors. This checks structure, not matching unknown secret
coefficients or provenance; the generic capability still has an explicit trust limit.

The fixed AA runtime consumes the same selected post ciphertext in affine2, outer1
and receipt.hidden. A separate test backend observes matching **all RNS coefficient
prefixes** at the actual two multiplication boundaries, beyond approximate plaintext
parity. Original valid IDs0/2 remain0/2. State projections retain their actual pre-lowering
ciphertexts for diagnostics; canonical hidden has no redundant projection. Failure
retains a complete pre-refresh candidate when one exists, completed events, phase/role
and optional inclusive target-evaluator count. A Host failure may leave additional
unmerged host work; the caller owns the host evaluator and must inspect it separately.
No target-only receipt claims that host work was rolled back.

Focused construction checks cover13 metadata/scalar tests,9 non-main control/diagnostic tests,
one separately run miniature Host test, and23 existing packed learning/refresh/optimizer
regressions. These are program/control/metadata evidence, not main encrypted training
parity. The public NI plan retains named unvalidated requirements and no Ready result:
bootstrap encoding/internal EvalMod, range, precision, actual secret/key provenance,
security, memory, backend and full profile schedule remain separate gates. Retaining old,
candidate, post-bootstrap and diagnostic snapshots is not a production memory-fit claim.

### D4c2 review fix round 1: pre-code binding contract (2026-11-07)

The independent c2 review found two structural binding defects (F1/F2); no main,
calibration or additional real Host bootstrap run is released by this fix. The
original 13-file freeze, report, complete patch, raw evidence and both executables
are preserved in the separate `task-54-d4c2-ni-original-freeze-archive` ledger
folder before these edits.

F1 changes the new, uncommitted metadata constructor to
`PackedBatchMeta::new(params: Arc<CkksParams>, index: u64, input: VectorSpec,
target: VectorSpec, records: Vec<PackedRecordMeta>) -> Result<Self>`.
The immutable batch retains that Arc. Its equality uses Arc pointer identity plus
all existing fields, rather than numerical parameter equivalence. At shared DAG
entry, compare that identity to the stage and revalidate every actual input/target
CtMeta against the consuming parameters (level bounds and exact Delta bits).
Scalar-only revalidation would leave same-shape foreign-context ambiguity. Known
next batches use this same gate before any current arithmetic. No declared scale
is silently normalized. Encoder remains required by the subsequent factor/mask
checks; this is not complete pure pre-key admission.

F2 proposed NN-local API, before production edit:
`pub trait PackedTargetKeyBinding: HeOps {
fn validate_packed_target_keys(&self, params: &Arc<CkksParams>) -> Result<()>; }`.
The CPU implementation checks parameter identity and actual relin plus every
available Galois key's Q/P basis Arc, levels, digits and NTT form. The stage's gate
still requires the complete stage rotation set. Host implementations of
`BootstrapOps<E>` and `PlannedBootstrap<E>` require `E: PackedTargetKeyBinding`;
they check the target on every binding preflight and direct bootstrap entry.
Existing HostBootstrapper remains the single count-merge owner. No shared
CKKS/HeOps/CUDA edits, secret callbacks or secret-provenance claim are proposed.
Other backends are unsupported for this adapter until they provide this explicit
checked binding; this preserves the future GPU implementation requirement rather
than waiving target structural checks. Third-party trait implementations remain a
stated trust boundary.

Root's pre-edit F2 API refinement replaces the proposed target supertrait above
with a separate binder object to avoid orphan-rule pressure in the later
bench/CUDA bridge (both an NN trait and CudaEvaluator would be foreign there):

```rust,ignore
pub trait PackedTargetKeyBinding<E: HeOps> {
    fn validate_packed_target_keys(&self, ev: &E, params: &Arc<CkksParams>) -> Result<()>;
}
pub struct PackedCpuTargetKeyBinding;
pub struct PackedHostBootstrap<'a, E = Evaluator, V = PackedCpuTargetKeyBinding> { /* private */ }
// CPU default:
PackedHostBootstrap::new(bs: &'a Bootstrapper, host: &'a Evaluator) -> Result<Self>
// Explicit backend binder, V: PackedTargetKeyBinding<E>:
PackedHostBootstrap::with_target_binding(bs: &'a Bootstrapper, host: &'a Evaluator,
    binding: V) -> Result<Self>
```

The adapter retains the binder and an evaluator type marker, with no target borrow
until preflight or execution. CPU default delegates to the CPU binder; the generic
constructor validates host material and stores the explicit binder. A future local
backend binder can implement the NN trait without a CUDA-to-NN production
dependency or full HeOps newtype. Such external binders are explicitly trusted to
perform read-only public key validation; their existence cannot mint Ready or
prove secret-key identity. No backend other than CPU receives an implementation
in this fix. Preflight/direct dispatch both validate the actual target each time.
