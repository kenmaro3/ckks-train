# Exact static diagonal admission and CPU bridge

Status: **independently reviewed and root-ratified design; implementation separately gated**. This refines Task54 D2c1 in `study-training-design.md` §§4–6/8. It authorizes no source change or execution. Source inspected on 2026-11-06; the assignment brief, rather than older future-tense passages in that document, supplies the current integration gates.

## Recommendation and decisions to ratify

Extract small parameter/NN metadata primitives and make the encrypted CPU bridge consume their immutable receipts. Keep model/order/rate ownership in the reviewed D6a materializer. Do not implement a second training algorithm in bench. The required new source packages are P1–P5 below; each requires its own assignment and review. P1/P2 are necessary before secure admission can be claimed, even though geometry already fits all21 profiles.

| Choice | Recommendation / reason / cost |
|---|---|
| Exact scale arithmetic | Reuse a public pure form of CKKS's existing exact-product-label search; add NN metadata transitions using actual `q()[level]`. Cost: narrow CKKS exports and NN planning module. Rejected: treating all scales as Delta or every compute prime as `2^42`; those miss real rejection branches. |
| Existing executable probes | Retain `ScaleProbe` as an independent post-construction consistency oracle on its evaluator fork. It cannot be the pre-key admission mechanism. Cost: compare public metadata and fail closed before the first update. |
| Planner ownership | NN owns operation/level/scale transitions; CKKS owns exact target-scale and bootstrap metadata; bench binds profiles, plans, inputs and physical choices. Rejected: copying private `pass_levels`/optimizer/compact formulas into bench, or a general fake-HeOps interpreter requiring ciphertext-shaped allocation. |
| Activation scope | An explicit metadata enum for Square, PolyRelu2, Identity only; unsupported/custom activations return a capability error. These cover every registered main stage. No need to generalize arbitrary trait objects or polynomial training now. |
| First CPU bridge | Single-stage I/NI first, then block composition using the already reviewed wrappers. Dedicated transpose keys as explicit default; reuse is admitted only with a named, reachable pool and matching plan. No automatic key-memory fallback. |
| Secret boundary | Client owns all plaintext inputs and secret-bearing objects. Server owns evaluation material and encrypted state. Interactive server receives a refresh capability that communicates with the client; it must not construct/store `ClientRefresher` itself. In-process local driver may own both sides, explicitly labeled a role separation, not process isolation. |
| Bootstrap metadata cost | Add a small public pre-key bootstrap descriptor sharing the existing DFT/encoded-transform support/count rules. Do not force dense DFT vectors/encoded tables into a purported lightweight admission API. Existing `dft_stages` is pure but materializes values; `transforms_cost` requires encoded transforms. A shape-only descriptor needs focused equality tests. |

Already completed and **not re-proposed**: D2a scheduled/frozen interactive NAG and `validate_lr` (`8c8c7bc`); D2c2 unequal interactive micro accumulation (`9a04fcc`); D3b weighted interactive blocks and atomic I/NI block setters (`cf9241e`, NI step/AA unchanged); D3a canonical AA reuse (`1a2d23e`); D6a immutable materializer (`da24811`). F32 engine work remains D6b's owner. No additional matrix/target/runner design is needed here.

## 1. What exists, and what does not

Evidence refers to current source, not an inference from tests or an executed build.

| Existing API / source | Reuse and limitation |
|---|---|
| `Layout::new`, `LayerShape::new`, `CompactPlan`, `CompactLayout::new` (`nn/src/layout.rs`, `compact.rs`, `trainer_ni.rs`) | Layout is checked. Shape/compact constructors can panic on invalid dimensions; prevalidate nonzero dimensions with checked next-power-of-two and arithmetic. Compact layout includes actual bias slots and bias-only containers. |
| `mlp_key_pool`, `mlp_rotation_steps`, `ni_rotation_steps`, `block_rotation_steps`, `ni_block_rotation_steps`, `transpose_rotation_plan` | Existing pure key/count geometry. NI normalizes rotations modulo slots. The bridge must normalize its final union and separately represent conjugation and SSE/relinearization material. |
| `levels_per_iter_of` | Pure but generic over activation traits; assumes full batch, uniform starting levels and ideal bootstrap reset. It provides a necessary depth check, not exact scale or schedule admission. |
| Private `ActMeta`, `pass_levels`, `LevelState`, `update_levels`, `unpacked`, `must_boot`; public `boot_schedule` | Useful shared structural machinery. `boot_schedule` already models first/later velocity levels and next-fullness; it omits scales, exact extraction retries, individual micro states and scheduled reference branches/counts. Reference gradients add no depth, so their absence does not by itself invalidate this structural schedule. |
| `ActCost::{QUADRATIC,IDENTITY}`, `predict_update_with`, `ref_shift_rotations`, `predict_block_update_with` | Pure NI rotations/relinearizations/bootstrap-event estimates, including EvalConfig. They do not predict all OpCounter fields, rate/reference selection, target-alignment work, bias extraction retries or exact scales. `QUADRATIC` does not distinguish Square's and PolyRelu2's ct-pt counts. Existing block predictor is homogeneous-placement and adds handoff bootstrap events only. |
| `Mlp::{encrypt_at,batch_spec_at}`, `NonInteractiveTrainer::{encrypt_network,batch_spec,predicted_trace}` | Available runtime client/verification paths. Batch scale calculation currently invokes private `ScaleProbe` on an evaluator fork. NI `predicted_trace` assumes full batches; its structural bias levels need not equal the actual lower extraction level after a retry. Not a pre-key metadata API. |
| `Trainer::{train_step_valid,train_step_valid_with_hidden,set_lr,with_frozen_biases}`, NI equivalents; block wrappers | Actual singleton/micro/block paths exist. Reuse their behavior, including mutation limits on HE/refresh failures. Atomic configuration validation is not transactional training rollback. |
| `CkksParams::{n16_train,n16_boot,q,p,scale,max_level}`, `BootParams::{default_n16,validate}`, `levels_after_boot` | Actual parameter identity and output level available before keys. `CkksParams` itself constructs RNS metadata, not evaluation keys/ciphertexts. |
| CKKS private `exact_product_scale`; public evaluator `exact_product_scale`, `target_plain_scale`, `MIN_PLAIN_SCALE` | Pure underlying label search exists but is crate-private. Public evaluator APIs require an evaluator. Extract/export without duplicating search, rounding or minimum-scale rules. |
| `dft_stages`, `galois_steps_for`, `evalmod_cost`, `transforms_cost`; `Bootstrapper::cost_per_boot` | Bootstrap output count arithmetic partly pure; complete shape/count receipt without encoded tables is missing. Never hardcode historical111/502 as the new count. |
| D6a `MaterializedProfile::{metadata,graph,plan,identity,numeric_f64,reference_kernel_rate_f64,update_rows}` | Immutable authoritative graph, named arrays, full plan/rates and old-weight stage inputs already available. No reinitialization, transposition, rate conversion, reshuffling or public mutable prepared-input path is necessary. |

`sim_train_step_sgd` is the legacy Square/SGD/default-configuration simulator with slot arrays; it is neither the new generic scheduled planner nor a justification for copying formulas into bench.

## 2. Proposed public metadata contract

Names in this section are **proposed**, not implemented. Avoid breaking existing APIs: legacy helpers can delegate to the same internal transitions and retain their existing output types.

**P1, CKKS-owned primitives.** Public pure functions take `&CkksParams`, level and finite scales, never an evaluator, Encoder, RNG or key:

```text
exact_product_scale_at(params, level, target) -> Result<f64>
target_plain_scale_at(params, level, source_scale, target) -> Result<f64>
scalar_metadata_at(params, CtMeta, value, target) -> Result<ScalarProductMeta>
bootstrap_metadata(params, BootParams) -> Result<BootstrapMetadata>
```

`CtMeta { level, scale_bits }` stores the exact F64 bits. Scalar metadata includes pre-rescale label/plain scale and the integer coefficient validity check from `mul_scalar` (`abs(round(value*plain_scale)) < 2^62`), not just positivity. The target/plain helper enforces `MIN_PLAIN_SCALE=2^20` wherever runtime does. Ordinary `mul_plain`/activation encoding paths retain their own existing checks; do not impose the exact-target path's policy on every activation multiplication silently. `BootstrapMetadata` binds complete q/p values, Delta, N/dnum/main-secret descriptor, BootParams including ephemeral key setting, exact output level/scale, DFT support/split descriptors, normalized rotation keys, conjugation/SSE requirements and a predicted per-bootstrap OpCounter. Construct it through the same support/rounding structure used by real transforms; no guessed zero-diagonal pruning. If encoded support cannot yet be derived exactly without tables, return `MissingCode(BootstrapMetadata)` and retain a separately labeled conservative key bound, not an exact count claim.

**P2, NN-owned pure stage planner.** Add `nn/src/diagonal_plan.rs`, with bounded changes in trainers/activation/dense/compact/bootstrap call sites only as separately assigned:

```text
plan_stage_geometry(dims, layout, activations, transpose, mode) -> StageGeometry
initial_stage_metadata(geometry, params, bootstrap?, optimizer, bias_policy) -> StageState
plan_stage_update(geometry, params, bootstrap?, state, UpdateSpec)
    -> Result<PlannedStageUpdate, PlanError>
```

`StageGeometry` contains logical dimensions, padded dimensions, BSGS splits/folds, target width, per-layer canonical periods, exact transpose source/rotation sequences, mask/replication recipes, optional CompactLayout and sorted key identities. Inputs are checked before calling current constructors.

`StageState` contains actual level/scale per expanded weight layer, bias, compact weight/velocity container and presence/absence of velocity; fixed full-plan reference and current rate; bootstrap output contract; stage identity. `UpdateSpec` binds **ordered** per-micro `n_valid`, x/y metadata, the one logical total, current kernel rate, and `next_full: Option<bool>`. `full` is `all(n_valid == lanes)`, not `total_valid == lanes` or a micro count. Stages share the logical schedule, not mutable state. Partial/full deltas must be traced individually before the NI reduce path aligns them to their minimum level. Interactive contributions instead form each gradient before their same-scale minimum-level accumulation.

`PlannedStageUpdate` includes per-micro forward/backward/gradient metadata; exact label alignment; bias retry decisions; compact occupancy/shift/unpack traces; current/reference coefficient identities; first/later NAG branch; pre-update retained representation metadata; pre-refresh updated state; bootstrap/refresh decisions and post-state; structured nonbootstrap OpCounter; bootstrap events and bootstrap-cost descriptor identity; client refresh count. Explicitly name all returned ciphertext roles. An operation trace can carry checked summed counts; additions/mod-switches have zero OpCounter cost but still affect level equality/liveness. Unknown data/range is never synthesized into the metadata.

**P3, bench binding.** `plan_diagonal(&MaterializedProfile,&CaseSpec) -> Result<DiagonalPlan,AdmissionIssue>` uses immutable D6a views. A separate `describe_candidate(ProfileMetadata,CaseSpec)` can produce dataset-independent geometry receipts, but its type cannot construct an admitted session. Production requires `identity.canonical_prepared()==true`; private synthetic fixtures use a distinct constructor/type. `DiagonalPlan` binds full materialized identity/plan digest, all full-plan coefficients and their maximum reference, consumed-update subset, concrete preset/BootParams, stages, logical/micro partitions, placement, transpose strategy, EvalConfig and all state-transition receipts. Hash typed/versioned fields including F64 bits and actual q/p integers; do not hash pretty Debug output. Quick retains the full reference and full-plan identity while explicitly recording only its consumed prefix.

Plan iteration must carry state through the entire intended consumed schedule, while the reference is derived from the full plan. Classify equal-current/reference exactly as the runtime does (`lr/reference == 1.0` in NI; the interactive optimizer's current condition), not with a tolerance. Cache repeated transition classes only after proving the cache key includes all incoming scale/level/container states, fullness tuples, rates/references and next-fullness. Do not collapse a cosine schedule to one constant representative.

Typed errors distinguish invalid registration/geometry (`InvalidAxis`, `InvalidDimensions`, `ArithmeticOverflow`, `InvalidMicroPartition`), finite/rate/reference conflicts, parameter/secret-profile mismatch, `MissingCode{capability}`, `LevelInadmissible{stage,update,micro,operation,trace}`, `ScaleInadmissible{...source,target,prime,reason}`, `RotationUnreachable`, and runtime `PlanMismatch{field,expected,actual}`. Missing datasets/exports remain D6a input states. Memory capacity and precision/range/security decisions stay independent fields; missing code is not evidence of insufficient levels, and a plausible geometry is not `Ready`.

## 3. Exact recurrence and branch obligations

Let `(l,S)` be ciphertext metadata and `q_l = params.q()[l] as f64`, using the actual ordered data primes. Mod-switch preserves S and lowers l; rotate/relin preserve both. For ordinary multiplication, align to `j=min(l_a,l_b)`, then runtime F64 multiplication followed by division gives `(j-1,(S_a*S_b)/q_j)`. Dense forward adds a bias at product scale **before** that rescale. Its bias must be at exactly `S_x*S_w`, at a level at least the chosen dense input level. Square maps `(l,S)` to `(l-1,(S*S)/q_l)`; PolyRelu2 has that same output metadata, plus its linear term's ct-pt/rescale path. Identity preserves metadata. Derivative factors use the activation's actual operation order and constant/mask coefficients; padded class channels and inactive sample lanes are different masks.

For an exact target T, use the existing search of `L` around `T*q_l` (nearest first, ±8 ULPs) with the exact predicate `L/q_l == T`. A mathematically equivalent reordered expression is not the executable contract. Then check `plain_scale=L/S`, runtime rounding/finite bounds and minimum plaintext scale. Include failed-search control flow but no counters for failed operations. Do not replace a miss by tolerance relabeling.

For two-layer WN/AU (or I before refresh), structural levels from common start s are:

| Value | Square/Square | PolyRelu2/Identity, full | PolyRelu2/Identity, partial |
|---|---:|---:|---:|
| z1, a1 | s−1, s−2 | s−1, s−2 | s−1, s−2 |
| z2, a2=delta2 | s−3, s−4 | s−3, s−3 | s−3, s−3 |
| dz2 | s−5 | s−3 | s−4 |
| delta1, dz1 | s−6, s−7 | s−4, s−5 | s−5, s−6 |
| weight gradients h1, h2 | s−9, s−7 | s−7, s−5 | s−8, s−6 |
| bias gradients before I retry | s−8, s−6 | s−6, s−4 | s−7, s−5 |

This table is **structural arithmetic**, not a complete scale proof. Interactive `bias_grad` retries exactly once from one level lower on `ScaleMismatch`, provided dz has at least2 levels. Its second miss fails. NI compact `extract_bias` retries downward while `level > need+1`; the resulting bias's actual level must appear in the receipt. `pass_levels` alone does not expose either branch. NI update predicts WN using its current structural `must_boot` rule, then also checks the actual extracted metadata for the next step. Do not silently change the bootstrap policy to repair a metadata failure.

The exact fresh-scale path for registered `mnist_released__emlp1` WN/AU, n16_boot, is:

```text
Delta = 2^42, boot output = (16, Delta), expanded start s = 15
W1,W2: (15,Delta); x: (15,Delta)
b1 scale = Delta*Delta
z1: (14, (Delta*Delta)/q_15) = (14,Z)
a1: (13, (Z*Z)/q_14)         = (13,A)
b2 scale = A*Delta
z2 = a2 = y: (12,(A*Delta)/q_13)
```

The registered B48-in64 batch is partial: dz2 level11, delta1 level10, s1 level13, dz1 level9, h1 level7/h2 level9, hb1 level8/hb2 level10. Both compact gradient branches land at Delta. Initial velocity is absent; when current<reference it is initialized from the **reference** branch without the two velocity multiplications. At current==reference the extra branch is absent. Later updates have two ct-pt/rescale operations per velocity container, applied at the chosen update level+1. Compact subtraction aligns with all gradients/biases and previous velocity−1; it never resets velocity when the rate changes. This example specifies actual q indices and exact expression order; no numerical q/scale bits or successful evaluation of these formulas is claimed by this design-only reading.

For AA at fresh s=15, a1 first converts to Delta at level12 and bootstraps to16; second dense aligns with W2/b2 at15. Its z2 is `(14,Delta*Delta/q_15)`. Identity/full gives minimum h1=10; partial gives9. Square/Square AA minimum is8. WN/AU fresh minima are6 for Square/Square and8(full)/7(partial) for eMLP. I starts at31 with minima22 and24(full)/23(partial); Square bias fallback can lower a bias by one but remains above or at the earliest weight-gradient level in this fresh two-layer example. These positive fresh minima prove no *fresh structural depth* obstruction; they do not prove later lazy-scale schedules.

For each update, simulate per-container levels from `min(old compact W, h_W, all its h_b, U−1 when present)`. Unpack weight containers costs one level; bias extraction targets the next actual input-scale product. AU bootstraps every container/velocity after every update. WN and AA use `must_boot`'s next-full decision, with `None` conservatively partial even after the final consumed update. In a quick run, use the actual chosen lookahead policy consistently: default to unknown at its terminal update as current `train_epoch_valid` does; record it. Do not report its last bootstrap decision as the full-run prefix decision if the full run knows a different next fullness.

NI lazy label alignment is a distinct branch: equal scale returns y unchanged (loss aligns levels); otherwise y must have `y.level > out.level`, is switched to `out.level+1`, multiplied by1 to the exact output scale, then rescaled. Count one ct-pt/rescale for each such micro/stage. This is off the main depth path but can itself fail. Declared per-stage fresh BatchSpec labels are not replaceable by one global y encryption.

Block source-stage representation is the first hidden activation from **old weights**. I/WN/AU spend one exact conversion-to-Delta rescale per micro/boundary followed by client refresh / bootstrap. AA takes `last_activation_bootstraps()[micro][0]` directly; zero extra conversion/bootstrap, only necessary downstream mod-switch. Count stage AA bootstraps once. Every stage has its own state/labels/scale path; local heads never become next-stage inputs.

## 4. Counts, keys and metadata verification

Complete predicted counts have seven fields matching OpCounter and a separate client-refresh/handoff count. `predicted_nonbootstrap + events * parameter_specific_predicted_bootstrap_cost` is a model; measured `ev.counter()` already includes boot work and must never receive this addition. Precision/range/host time do not follow from either vector.

Use existing NI `predict_update_with` as the rotation/relin component and check it against the richer trace. For I, derive its small generalized operation trace beside the trainer (no generic predictor currently exists). Crucial differences are:

- I computes each micro's gradient/reduction/replication separately, then adds gradients. Its transposes occur per micro. NI shares its transposes and reduces accumulated products, with relinearization affected by `lazy_relin`.
- Each I extra reference branch adds `m` ct-pt + `m` rescales + `m*log2(lanes)` replication rotations **per layer per micro**. NI adds `m` ct-pt + `m` rescales plus `CompactPlan::pack_rotations()` **per layer per logical update**, since products/reduction are shared across micros. No reference branch adds depth or ct-ct products.
- Later I NAG adds two ct-pt/rescales per expanded weight diagonal; later NI NAG adds two per compact weight container. First NAG has neither, including first-below-reference. Frozen biases still run zero-coefficient gradient, extraction and refresh operations.
- Partial Identity derivatives add one ct-pt/rescale per partial micro. PolyRelu2 forward adds its linear ct-pt/rescale, while Square does not. Target alignment, exact-scale retries, AA conversion and boundary conversion remain explicit.
- NI compact-shift reference costs are additional to existing predictor output. I refreshes `sum(m_l)+layers` ciphertexts for SGD and `2*sum(m_l)+layers` after each NAG update; first NAG already has velocity to refresh. Add boundary refreshes separately.

Keys are a union of exact stage geometry, transpose strategy, gradient reduction/replication, compact and bootstrap requirements, with modulo-slot normalization, zero removal and deterministic order. Record each role's level/basis/dnum and exact parameter identity; an equal rotation integer on different parameters is not the same key. The current CPU bootstrap constructor generates its own key bundle, so logical union cardinality is **not** the physical resident-key allocation count: separate training and bootstrap bundles, possible duplicates, conjugation and SSE bytes in the allocation/liveness receipt. Sharing bundles would require a separately justified source change and must not be assumed in memory estimates.

Runtime verification after setup and before update1 compares parameter object contents, layer dims/periods/diagonals/BSGS, CompactLayout and bias slots, key manifest, initial state metadata and all BatchSpec fields (levels and scale bits) to the pure plan. Inspect/probe with an independent evaluator fork. After each focused test step compare trace labels/levels, exposed weight/bias/compact/velocity metadata and actual refresh/bootstrap decisions to the corresponding plan; include optimizer-velocity accessors only where a separately reviewed diagnostic is needed. No server decryption to prove metadata. Test decryption stays in the client fixture. Unknown metadata means `MissingCode`, not a skipped comparison. Measured-counter tests take training-counter snapshots around only the update and show probe/reporting calls do not alter them.

## 5. All21 candidate metadata outcomes

The registry's16 primary plus4 B64 plus1 B128 profiles are retained below. Stage dimensions are authoritative D6a graph dimensions: 2020 `[input,10,C]`; e1 `[input,32,P]`; e2 `[input,64,P] ; [64,32,P]`; e3 `[input,128,P] ; [128,64,P] ; [64,32,P]`. C=3/2/10/10 and eMLP P=4/2/16/16. Layout padding of2020 C is not an extra trainable output class.

Columns: `m total` is sum of expanded diagonals across all stages; `Cw` is compact weight-container count, with biases packed into those containers and **zero bias-only containers** for these candidates. All layers have `m <= slots/L`; each two-layer stage therefore needs two weight containers, and its biases fit spare positions. For the MNIST/Fashion e3 first layer, m=128 exactly fills capacity128 (L256); the head's L128 container has room for both biases. This is why B128 must still inspect bias occupancy, even though its final count remains2. NAG adds `Cw` velocity containers; SGD adds none. `I refresh` excludes boundary handoffs. `replicas` is slots/(d*lanes).

| Profile ID | stages | d | lanes / valid | replicas | m total | Cw | I refresh |
|---|---:|---:|---:|---:|---:|---:|---:|
| iris__mlp2020 |1|16|32 /20|64|8|2|10|
| iris__emlp1 |1|32|32 /20|32|8|2|18|
| iris__emlp2 |2|64|32 /20|16|44|4|92|
| iris__emlp3 |3|128|32 /20|8|112|6|230|
| breast_cancer__mlp2020 |1|32|16 /16|64|18|2|20|
| breast_cancer__emlp1 |1|32|16 /16|64|34|2|70|
| breast_cancer__emlp2 |2|64|16 /16|32|68|4|140|
| breast_cancer__emlp3 |3|128|16 /16|16|134|6|274|
| mnist_released__mlp2020 |1|256|64 /48|2|32|2|34|
| mnist_released__emlp1 |1|256|64 /48|2|48|2|98|
| mnist_released__emlp2 |2|256|64 /48|2|128|4|260|
| mnist_released__emlp3 |3|256|64 /48|2|272|6|550|
| fashion_mnist__mlp2020 |1|256|64 /48|2|32|2|34|
| fashion_mnist__emlp1 |1|256|64 /48|2|48|2|98|
| fashion_mnist__emlp2 |2|256|64 /48|2|128|4|260|
| fashion_mnist__emlp3 |3|256|64 /48|2|272|6|550|
| mnist_released__mlp2020__b64 |1|256|64 /64|2|32|2|34|
| mnist_released__emlp1__b64 |1|256|64 /64|2|48|2|98|
| mnist_released__emlp2__b64 |2|256|64 /64|2|128|4|260|
| mnist_released__emlp3__b64 |3|256|64 /64|2|272|6|550|
| mnist_released__emlp1__b128_candidate |1|256|128 /128|1|48|2|98|

**Outcome for every row:** analytically fits checked slot geometry; compact container counts above follow source formulas; fresh structural levels for I and all three NI placements are nonnegative by §3. **Exact admission unanswered** pending P1/P2/P3 actual-prime scale/key/full-count receipts and runtime equality review. This is not21 passing executable planner tests. No numerical q/scale-bit output, key cardinalities, maximum-live-byte values, precision/range or resource measurements were computed. None is presently promoted to `Ready` by this document.

For a one-micro default row with s stages: each NI state refresh has `Cw` bootstraps for SGD, `2*Cw` for NAG; AA has s activation bootstraps per update; WN/AU has s−1 extra boundary bootstraps; AA has0 extra boundaries. WN/AA state-refresh frequency depends on the exact carried plan. Defaults20-in32 and48-in64 are always partial. B64/B128 and BC16 are full. Optional capped32 B48 uses32+16, one optimizer update/one schedule coefficient and total48; its geometry, counts, retained handoff liveness and full-plan identity form a **different physical plan**, not two updates or permission to change the registered learning profile.

Input readiness remains independent: missing captured MNIST eMLP2/3 seed3–9 exports stay missing; missing dataset bytes stay missing. No local or captured data availability was verified in this design. All full encrypted block user holds remain. Secure I uses only n16_train and NI only n16_boot/default_n16; test-only mini or sparse presets cannot discharge production admission.

## 6. CPU client/server interfaces and ownership

P4 should expose narrow proposed types, not a self-referential struct holding trainers that borrow its own evaluator. A caller-owned immutable `CpuEvaluationContext` (Evaluator and NI Bootstrapper) lives longer than `CpuDiagonalSession<'ctx>`. The session receives encrypted initial state and immutable `Arc<DiagonalPlan>`; trainer instances borrow that context. Client-side bootstrap/key generation returns evaluation material and encrypted initial state to the server before training. KeyGenerator and Decryptor remain client-owned or are dropped after setup as appropriate; no full server secret key or materializer in the session.

```text
StudyClient::prepare_initial(materialized, plan, client_crypto)
    -> EncryptedInitialState
StudyClient::encrypt_update(materialized, plan, update_index, client_crypto)
    -> EncryptedLogicalUpdate { plan_hash, index, total_valid,
         micros: [ { x, stage_targets: [y_s], n_valid } ] }
CpuDiagonalSession::new(context, plan, encrypted_initial, refresh_capability?)
    -> Result<Session>
Session::step(encrypted_update) -> Result<UpdateCompletion>
Session::snapshot(reporting_endpoint) -> EncryptedModelSnapshot
```

All public packet fields should be private with validated constructors/accessors. The session checks update index/plan hash/partition/stage-target cardinality plus metadata before touching rates or counters. It selects the rate from the bound public plan, not packet-supplied plaintext arrays or arbitrary caller lr. Client encryption gathers D6a's exact row slice, constructs C/P-aware targets and uses each stage's own BatchSpec. Upstream hidden activations are computed and handed off encrypted inside the block session; the client never computes replacement stage features.

For I, a `Refresher` implementation on the server side carries only a client transport/callback handle and public layout/output-level contract. A local test/local CPU driver can implement this through a client-owned service containing `ClientRefresher`; do not call that in-process wiring network isolation. Avoid capturing the client materializer/Decryptor directly inside a server-owned closure. NI has no refresh capability; only Bootstrapper evaluation keys, including SSE switching keys rather than either full secret. Reporting uses an explicit encrypted snapshot handed to the client at an allowed endpoint; prediction/diagnostic evaluator is an independent fork and never used to select rates, clip values, choose bootstrap decisions or repair state.

Single-stage P4 constructs existing activation objects, Mlp and Trainer/NI Trainer from matching metadata, creates SGD or NAG with D6a's first kernel rate, sets NAG full-plan reference before velocity initialization, freezes eMLP biases before wrapping and uses `train_step_valid` once per logical update. Loss factor is already folded into D6a's kernel coefficient; do not multiply it again. Client refresh canonicalizes replicated blocks under the chosen common Layout. P5 creates BlockMlp and stage trainers, configures every stage before assembly, and calls reviewed BlockTrainer/BlockNiTrainer once with per-stage target slices. Use existing atomic setters; no new optimizer/control math.

An HE/refresh/bootstrap failure can leave committed optimizer or earlier-stage state. Mark session failed/poisoned and reject further steps; do not retry the same logical update against partially changed state. A failed step never increments the successful logical-update counter. This wrapper policy requires no whole-model rollback implementation. Errors before work leave counters/state unchanged where existing APIs already guarantee it.

This bridge returns update completion/diagnostic metadata only. D6 owns measurement scopes, reporting endpoints, F32 target locks and cumulative artifacts. Precision and bootstrap range checks remain separate gates: exact scale/level validity says nothing about plaintext magnitude, CKKS approximation error, overflow in encoded data or acceptable final accuracy. Memory planning must bind parameter/key shapes and simultaneous role liveness (including duplicate key bundles, compact+expanded state, both gradient branches, retained block/micro handoffs and bootstrap workspaces); CPU/GPU resource admission and measured high water are not supplied by a guessed count of live ciphertexts.

## 7. Bounded future packages and acceptance tests

Paths are proposed future ownership, not permission to edit now. Root reserves shared `lib.rs` exports and assigns narrow existing-file hunks to avoid D4/D6 ownership overlap. Each package is independently reviewable; no package imports data, launches a campaign or changes thresholds/profiles.

| Package | Owned future scope | Acceptance evidence required before dependent package |
|---|---|---|
| P1 CKKS public metadata | New `ckks/src/metadata.rs`; narrow evaluator shared helpers and bootstrap shape/support helpers/exports; focused metadata tests | Actual secure q vectors/identities and levels_after_boot16; exact-label reachable/unreachable fixtures, F64 expression order, finite/overflow/min-scale/integer-bound failures; bootstrap pure descriptor equality to constructed small fixture's actual key support and `cost_per_boot`; no keys/Encoder/evaluator in pure API. Secure metadata tests do not encrypt or claim performance. |
| P2 NN exact stage planning | New `nn/src/diagonal_plan.rs`; narrow shared recurrence extraction in trainer/trainer_ni/dense/activation/compact as approved; no change to reviewed update semantics | Square/Square and Poly/Identity, C3/2/10 and logical P4/2/16, full/partial/unequal micros, first<ref/first=ref/later decreased/restarted NAG; I bias retry, NI extraction retry/label alignment; all placements/next-full unknown; dedicated/reuse keys and EvalConfig; exact metadata and all seven OpCounter fields against existing small CPU runtime fixtures; singleton legacy behavior preserved. No secure HE profile run. |
| P3 checked study admission | New `bench/src/study_admission.rs`, `tests/study_admission.rs`; exports separately coordinated | Enumerate all21 metadata rows and all I/NI placements; checked geometry/compact/keys/levels/scales; preserve invalid/missing/unproved statuses; seed-independent geometry versus input readiness; full-plan reference survives quick prefix; corrupt identity/params/partition rejected before key builder spy called. Pure fixtures include secure presets; no dataset download or key allocation. |
| P4 CPU single-stage bridge | New `bench/src/study_client.rs`, `study_diagonal.rs`, focused tests | Stage-specific labels and encrypted initial metadata match plan; actual small CPU I and NI parity versus reviewed D6b F64 oracle across loss/optimizer/activation/class variants and unequal micros; scheduled branch/count parity; client-only secret paths, independent probe/report counters; poisoned-session failure behavior. Synthetic mini fixtures cannot be labeled admitted production profiles. NI tests can be bounded/tiny; full secure bootstrap cost/range evidence remains separate. |
| P5 CPU block composition | New `bench/src/study_blocks.rs`, focused tests (future D3 bridge scope); reuse completed NN wrappers | e2/e3 old-weight representations/local-head losses, per-stage P/classes/bias freeze; rate/reference atomics already reviewed stay unchanged; all placements, stage labels and heterogeneous incoming metadata; WN/AU boundary work and AA exact canonical reuse/no double boot; unequal micros, retention/liveness receipt and predicted-versus-actual counts. Full encrypted blocks remain held. |

Keep bootstrap cost model validation separately labeled from numerical bootstrap precision. A test double can verify boundary wiring, but cannot establish real NI numerical parity or secure runtime feasibility. D6b F64 is the diagnostic oracle; F32 arithmetic/targets remain D6b/D6c, not substituted by HE metadata tests. A future P1/P2 implementation finding a scale failure retains that row with an operation trace and proposed correction review, rather than lowering presets or altering rates.

## 8. Evidence boundary

This document was prepared by reading the assignment, registry/design/input contracts, current parameter/activation/dense/optimizer/compact/trainer/block/counter/bootstrap code and D6a immutable APIs. Table arithmetic and recurrence examples are analytical deductions from those sources, not executed planner outputs. No source/test/dependency/git change, build/test/training/probe/key allocation, SSH, download or campaign occurred. Only this new document and its design-report ledger are owned outputs. Independent review and root ratification are complete; P1–P5 still require separate implementation assignments.

## 9. P3 binding decisions before implementation (2026-11-07)

Section8 describes the original design publication. P1/P2 have since been implemented; [P2's bounded validation](../results/validation/task54-p2/README.md) is separate from production admission. P3 is now assigned for tests-first implementation. The following decisions resolve actual API gaps without changing scientific profiles or the original final acceptance requirements.

**Registration and consumption.** The existing sealed `RegisteredCase` supplies registration; `CaseSpec` above was a proposed name. Preserve Full-only registry selection. A private checked consumption request can plan a separately tagged quick prefix of that Full case, retaining its original canonical case bytes, complete full-plan coefficients and full maximum reference. This does not enable quick execution or alter CaseProvenance. A second registry or reinterpretation of Full identity would create competing sources of truth and is rejected.

**Complete receipts with explicit capacity.** Encode every consumed stage transition and handoff into an immutable typed transcript with a checked offset index, retaining only current stage states and temporary P2 receipts during composition. Keeping all nested Rust receipts would duplicate state and trace allocation; retaining hashes alone would discard inspectable transitions. Use caller-supplied byte/index limits, with64MiB only an initial convenience limit, not a claim that a full Fashion schedule fits. Exceeding a limit returns a distinct operational budget failure and the completed-prefix diagnostic, with no complete `DiagonalPlan`. It must not shorten the intended schedule, omit failed/unplanned rows, silently select quick or report a level failure. Current transient allocation/RSS and HE liveness remain separate unproved conditions. A future larger limit requires measured receipt sizes; the full registered experiment matrix remains unchanged.

**Canonical positive fixture.** Synthetic planning inputs stay private and distinct from canonical `MaterializedProfile`. In addition to bounded synthetic checks, the P3 assignment includes one actual bundled Iris materialization through the existing production constructor, binding its entire2400-update full plan while planning a separately tagged two-update prefix. It reuses secure parameter metadata and performs no key allocation, encryption, training or download. This is a planned test, not a reported pass. Synthetic success alone cannot establish the production wrapper's positive path.

**Pre-key rejection remains required.** Section7's key-builder-spy acceptance cannot be established by a spy that no valid path can reach. The current dispatcher ends at `MissingExecutionDependency`; real D7 admission/start and factory connection are absent. P3 implements pure binding revalidation now. The first actual P4/D7 consuming boundary must revalidate before its factory call and test both an otherwise-valid nonallocating spy control (one call) and corrupted identity, parameters and partition (zero calls), under private nonproduction fixtures for other prerequisites. Public fake readiness/start capabilities are rejected. P3's standalone source and pure checks may be reviewed and committed as an implementation stage, but the original pre-key integration acceptance stays **pending**, and full P3 acceptance must not be claimed until it passes. Production execution still requires genuine D7 gates/start; a test fixture cannot supply that authority.

All21 candidate profiles and I/WN/AU/AA records remain represented, including missing, invalid and unproved outcomes. Successful metadata checks do not prove resource capacity, range, precision, security, target validity or campaign authorization. These decisions do not release any full encrypted-block campaign or change unresolved B, same-N or author decisions.

## 10. P3 review corrections and P4 numerical-test preparation (2026-11-07)

P3's independent source review found that a capacity failure during final retention could discard an already completed two-update prefix, and operational failures inside composition could omit the actual stage and work-limit values. Focused tests reproduced both failures before correction. The correction must retain complete records through finalization, distinguish update/handoff/finalization phases, and preserve exact required/limit values. An early capacity check alone would not cover later allocation failure, so all fallible finalization operations need the same prefix-preserving policy. Successful metadata planning remains distinct from scientific admission.

The original requirement to bind construction limits is retained: capture the four actual `MetadataLimits` ceilings in successful BootstrapMetadata construction, expose them immutably, and encode them separately from actual resource charges in P3. Larger sufficient ceilings may leave the cryptographic descriptor unchanged while changing this operational identity. Treating actual resource use as its configured ceiling, or accepting an unverified caller assertion, would lose that distinction. Existing P1 cryptographic encoding, arithmetic and enforcement stay unchanged. Tests must also discriminate isolated parameter/rate/partition/state fields using the production encoders and distinguish a destination stage's actual target label from its preceding stage's label. These are assigned corrections; this section does not assert their final validation or full P3 acceptance.

For P4, first verify that its numerical test fixture can distinguish incorrect training. The original tiny weights and provisional NI absolute tolerance5e-3 have no measured discrimination margin. Use a cfg(test) child of the existing D6b F64 engine for bounded comparisons; copying a second trainer or exposing a public synthetic materializer is unnecessary. Test the original fixture and one predetermined signal fixture, with at most200 actual Engine::train calls and no CKKS/key/HE construction. Keep scientific models, rates and presets unchanged.

Before any HE validation, require each applicable wrong-loss/rate/denominator/tail/target/velocity control to differ at its first affected update by at least eight times the fixed candidate caps (I1e-5, NI5e-4). Require a separate no-update weight witness in each trainable layer, and actual baseline hidden activations within3/4 for the proposed trajectory. Equivalent controls are labeled rather than counted as detected errors. A failed sensitivity or range gate is evidence to review, not permission to loosen tolerance or claim numerical parity. The test-fixture measurements, encrypted precision/range validation, actual pre-key boundary, D7 start/admission and scientific campaign remain separate requirements.

## 11. Measured sensitivity and stricter PRE-HE cap (2026-11-07)

The first pure run used the section10 caps unchanged and executed136 actual D6b Engine updates,68 per fixed candidate. Both original and signal candidates failed that criterion. Signal passed all57 I witnesses but missed5 of57 NI witnesses; its smallest required first-affected absolute difference was0.00351562499999999445. Independent review reconciled all228 margin records against retained state snapshots and confirmed the bounded test implementation, finite states and baseline hidden activation maximum0.3288487366949463<=0.75. These are pure fixture observations, not encrypted precision or scientific training results.

Before any HE, retain the signal fixture and8x separation requirement while tightening NI epsilon to4e-4; I stays1e-5. The measured minimum divided by8 is approximately4.39453125e-4, so the selected smaller cap implies at least8.789x separation for all prescribed signal witnesses. This is the disposition's permitted choice of a stricter cap from measured pure margins. Lowering the required separation, silently changing fixture/rates, or selecting tolerance after observing HE errors would weaken the test and is rejected. The original source/logs/caps and FAIL statuses remain immutable; a versioned pure rerun must validate the two-constant change before adoption.

This decision does not prove that HE can meet4e-4. Later HE validation must independently satisfy the fixed all-entry bound and the required mutant separation; a failure remains unmet and cannot be silently converted to success by loosening the cap. Scientific presets, final-accuracy criteria, P3/P4/D7 authority and all campaign holds are unchanged.

## 12. P4 CPU接続の実装裁定（2026-11-07、実測前）

単一stageのCPU接続は、既存の計画結合検証後にcandidateから配置を取得する。candidateの仮の初期学習率は利用せず、実際の初期状態・更新前後の状態は不変の計画から取得する。配置の再構築に必要なgeometry／initial各1回の余裕を最初のrequestに含め、hash確定後に制限を増やさない。P3に全配置を保持する代替案は、保持量と寿命の変更を伴うため採用しない。

全ciphertextの状態と個数、7種類の演算カウンタ、設定された係数・reference・refresh／bootstrap判断を照合する。公開されているtraceとhidden状態も記録するが、型付きの完全なP3 traceとの一致は未検証として残す。完全なtraceを新規に保持・復号する方式はこの接続には追加しない。

鍵と平文はclient側が所有し、server sessionは暗号化packetと評価資材、対話方式の場合のみ狭いrefreshハンドルを受け取る。同じ暗号パラメータでも異なる鍵所有者のpacketを受理しないため、setupごとに内部のcontext識別子を関連付ける。これは同一プロセス内の役割分離であり、プロセス隔離や暗号学的認証の証明ではない。処理途中の失敗はsessionを再利用不可にし、成功した更新数を増やさない。

検証は既存のSignal平文Engineと既存の学習器を使う。合成planはprivateな別型とし、正式なmaterializationや実行許可へ変換しない。局所的なfactory呼出順序テストを、正式なD7のgate／start検証の代用にしない。数値許容誤差は暗号実行前に固定済みのI=1e-5、NI=4e-4を維持する。まずテストと実装を作成し、compile／静的検査と独立レビュー後に実行対象を固定する。元の32成功＋2失敗、最大85 bootstrapの検証範囲は削減しない。正式なrunnerと全実験行列の完了は引き続き未達である。
