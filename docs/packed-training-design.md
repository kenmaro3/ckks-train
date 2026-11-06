# Genuine encrypted ReBoot-packing training (Task54 D4)

Status: root-ratified design; D4a geometry/primitives are implemented, locally validated with focused CPU tests and independently reviewed. D4b1 matrix optimizer preparation is implemented, locally validated and independently reviewed. Later stages remain gated; full training level/precision admission is unimplemented. This document creates no experiment result or dispatch authorization. The approved [experiment matrix](experiment-matrix.md), original research-plan §4.4 baseline (c), and [interactive profile design](interactive-profile-design.md) govern learning semantics. Full encrypted block runs remain held for the user's separate decision.

## Decision and scope

Add an original, generic `packed_training` module over `HeOps`, reusing `RebootLayout`'s row/column kernels. Keep one sample per ciphertext and alternating RE/CE along the model's backbone. Add distinct typed packed weights, bias vectors, gradients and persistent optimizer state. Do not wrap a packed matrix ciphertext in `Diagonals`, `EncWeights`, or the existing diagonal optimizer merely to reuse an API.

The NI trainer holds an evaluator and `BootstrapOps<E>`, public encoders/metadata, encrypted inputs/targets/state, and no decryptor, secret key or client callback. The interactive trainer computes the same encrypted gradients and updates, then delegates only state refresh or block-activation refresh to a typed client boundary. Neither mode sends gradients to a client for plaintext optimizer updates. CPU generic execution is a correctness oracle; baseline (c) measurements require the GPU implementation. CUDA level operations with `HostBootstrapper` are explicitly allowed and reported as GPU level operations / CPU bootstrap, including transfers.

Preserve `mnist_reboot_layout.rs` and its historical results as the legacy client-updated, plaintext-label count reference. A new entrypoint consumes the shared registered profile binder; do not silently change the meaning of the old binary/results. Packing correspondence does not establish cryptographic, backend, security or implementation identity with official ReBoot. No GPL/OpenFHE/ReBoot implementation source or snippets are used. Algorithmic inputs here are this repository's original Rust kernels, documented mathematics and already captured configuration values.

Eventual coverage includes every valid packed-c row: four primary datasets, all four models, registered batches/alternate studies, I and NI (WN/AU/AA), seeds 0–9. Missing exports, code, device capacity or authorization remain explicit case states, not scope exclusions. `mlp2020` is Square/Square with biases and SGD; eMLP is PolyRelu2/identity, no biases, NAG, and local heads for eMLP-2/3. Initial arrays, preprocessing, sample order, loss, schedule and tail policy come from the same bound profile as main/FP32. No locally invented rate, initialization or external default is permitted.

## Canonical geometry: one lattice for the complete graph

A `RebootLayout(r,c)` has `r*c = slots`, with both powers of two. Expanded vector `E_d(v)` stores `v[i]` at every `(i,j)` for `i<d`; repeated vector `R_d(v)` stores `v[j]` for `j<d`. Remaining logical coordinates are zero, within CKKS error. A row matrix stores `W[i,j]`, a column matrix stores `W[j,i]`. Shape, representation and lattice are part of the ciphertext's checked metadata; equal ciphertext lengths alone never imply compatibility.

An RE edge takes `E_d -> R_k` with a row matrix (`d<=r, k<=c`); a CE edge takes `R_d -> E_k` with a column matrix (`d<=c, k<=r`). Backward reverses the vector representations using the same matrix: RE's input gradient uses `vecmat_col`, CE's uses `vecmat`. Row gradient is `outer(E_input,R_dz)`; column gradient is `outer(E_dz,R_input)`. No transposed encrypted copy is needed.

Here `W` always denotes the mathematical `d_in x d_out` matrix and row-vector forward is `x W`; CE stores its transpose in slots only. `dz` is in the output vector representation; the input gradient is `dz W^T`. The matrix gradient and persistent `U` use exactly their weight's row/column orientation and active-entry mask throughout updates and refresh. A bias and its gradient use the output vector representation. Square and PolyRelu2 act coordinate-wise and preserve representation; they do not turn expanded vectors into repeated vectors.

| Graph path or branch | Vector direction and output loss representation |
|---|---|
| 2020 | `E_input --RE--> R_10 --Square--> R_10 --CE--> E_classes --Square--> E_classes`; encrypted target, loss delta and final `dz` are Expanded. Output Square's derivative is evaluated before propagating the gradient through the CE weight. |
| eMLP-1 | `E_input --RE/PolyRelu2--> R_32 --CE/Identity--> E_output`; final target/delta Expanded. |
| eMLP-2 local block | `E_input --RE/PolyRelu2--> R_64`; local classifier `R_64 --CE/Identity--> E_output`. Handoff is `R_64`. |
| eMLP-2 output stack | `R_64 --CE/PolyRelu2--> E_32 --RE/Identity--> R_output`; its own target/delta Repeated. |
| eMLP-3 first block | `E_input --RE/PolyRelu2--> R_128`; local head CE yields `E_output`. Handoff is `R_128`. |
| eMLP-3 second block | `R_128 --CE/PolyRelu2--> E_64`; independent local head RE yields `R_output`. Handoff is `E_64`. |
| eMLP-3 output stack | `E_64 --RE/PolyRelu2--> R_32 --CE/Identity--> E_output`; its own target/delta Expanded. |

`E_d` support is rows `i<d` across all columns; `R_d` support is columns `j<d` across all rows. A row matrix mask is `i<d_in && j<d_out`; a column matrix mask is `i<d_out && j<d_in`. These masks define zero structural padding for weights, gradients and velocity. Different local-head output representations require different encrypted target packings even when the underlying one-hot labels are equal.

Choose a **single shared lattice per complete graph**, including local classifier edges. Start the first backbone edge RE and alternate along the backbone; each local classifier uses the opposite orientation to that local backbone edge. Do not restart every stage at RE. Then a local hidden activation is already in the next backbone edge's required representation, even when stage dimensions differ. A typed identity handoff (plus level restoration) replaces repacking. Independent targets are packed into each stage's output representation by the client at ingestion.

Planner rule: collect all RE output widths and CE input widths, including heads; set `c` to the smallest power of two at least their maximum, then `r=slots/c`. Check **every** RE input/CE output against `r`, and all matrix/vector/mask shapes. If that rule fails, search the other power-of-two factorizations satisfying the same orientation constraints; if none fit, return a named capacity gap. A future tiled layout needs a separately reviewed extension, not a changed graph or automatic profile exclusion.

For the registered `n16_*` slot count 32768, the following are derived feasibility witnesses, not externally sourced settings or memory/performance measurements:

| Graph | Backbone / local head orientation | Common `(r,c)` witness |
|---|---|---|
| 2020, input at most196, hidden10, output at most10 | RE, CE | (2048,16) |
| eMLP-1, input196 or existing784 supplement, hidden32 | RE, CE | (1024,32) |
| eMLP-2, widths64 then32 | RE(head CE), CE then RE | (512,64) |
| eMLP-3, widths128 then64 then32 | RE(head CE), CE(head RE), RE then CE | (256,128) |

All four primary dataset dimensions fit these witnesses; the planner must prove this from each bound profile, not hard-code the table. A single `(1024,32)` lattice cannot hold the eMLP-3 RE128 output. The new trainer therefore cannot simply reuse the legacy eMLP-1 constructor. Common-lattice identity handoffs cover the presently registered shapes without a new permutation kernel. Future unequal-lattice handoffs must fail before work until a counted homomorphic conversion is designed; decrypt/repack is never an NI fallback.

Batch size is a count of sample ciphertexts, not a lane dimension. B20, B48, B64 and B128 require no power-of-two sample padding in this layout. B128's diagonal input-slot-fill rationale does not make packed-c use128 lanes or increase its per-ciphertext model capacity. Admit model slots, depth, live sample ciphertexts and memory separately at every stage. A `weighted_partial_v1` 32+16 partition maps to48 valid sample records; it must give exactly one48-sample update. Dropped tails, flat48000 order, B64 sample-aligned rates and B128 crossing-update checkpoints remain profile rules, unchanged.

## Types and interfaces

Names below are proposed new APIs, not existing implementation claims.

| Type / API | Contract |
|---|---|
| `PackedPlan::new(profile_shape, slots, mode, placement, schedule_meta)` | Pure checked graph/lattice/orientation/mask/level/count/key/memory plan before key generation; no data-dependent tuning. |
| `PackedVector<C> { ct, layout_id, repr, logical_len }` | Checked Expanded/Repeated vector; no implicit casts or secret-side conversion. |
| `PackedWeights<C> { ct, layout_id, orientation, d_in, d_out }` | One row/column matrix ciphertext; active-entry mask belongs to the plan. |
| `PackedBias<C>` | Optional encrypted vector in the layer output representation; present only for the 2020 graph. |
| `PackedGradient<C>` | Matrix shape/orientation, exact Delta scale, sum/mean/folded-rate metadata and generation ID; cannot be passed to another layer. |
| `PackedLayerState<C>` | Weights, optional bias, optional fixed-reference `U`, immutable NAG basis and update index. No plaintext weights/velocity. |
| `PackedStage<E>` / `PackedBlockGraph<E>` | Two-layer stage or independent local stages plus separate output stack; stores activation kinds and graph IDs. |
| `PackedBatch<C>` | Ordered per-sample input and stage-target vectors; public validity/group metadata; validates total valid count and target width/representation. |
| `PackedStepPlan` / `PackedStepReceipt` | Structural level+scale trace, exact nonbootstrap counters, bootstrap events, rate branches, checked shape IDs and actual counter delta. |
| `PackedNiTrainer<'a,E,B: BootstrapOps<E>>` | Read-only validate/plan followed by encrypted update; no secret or refresher field. |
| `PackedInteractiveTrainer<'a,E,R: PackedRefresh<E>>` | Same update engine; explicit timed refresh protocol carrying matrix/vector role and layout metadata. |
| `last_activation_bootstraps()` / `StageForwardSnapshot` | AA post-bootstrap hidden values distinct from pre-bootstrap forward hidden values; snapshots refer to pre-update weights. |

All constructors check parameter identity, slot count, component count where needed, supported shapes, finite public factors, exact CKKS scale and level. Rate updates use validate-all then commit-all metadata. State assignment occurs only after all new weights, biases and velocities for that stage have been prepared. Reject structural errors before evaluator work/counter movement; do not claim rollback of already completed generic HE operations or a failed device transfer. Failure receipts identify the last completed phase and no successful training checkpoint is emitted.

## Learning function, masks and persistent state

For each stage, evaluate with the pre-update state. Loss delta is `pred-target` for eMLP, and `2*(pred-target)/real_classes` for2020. Mean gradients divide by the total valid logical samples `n`, never padded model width, physical allocation length or number of micro-groups. eMLP target widths are Iris4/BC2/MNIST16/Fashion16: every one of these model outputs participates in delta and gradient, including padded-class channels whose encrypted target is zero. Accuracy alone restricts argmax to real classes. 2020 uses exactly3/2/10/10 output channels. The *structural* mask removes lattice entries beyond model width; it must not remove the eMLP trained padded-class channels.

For partitioned work, accumulate per-sample unscaled outer products over all valid groups, then apply the folded coefficient once using `n=sum(n_valid)`. Equivalently, independently computed group means require weights `n_group/n`; an unweighted average of group means is forbidden. Public invalid sample records are skipped before HE computation (or explicitly zero-masked in a separately counted fixed-allocation path); zero ciphertext records never change the denominator. All products entering a lazy sum must have the same level and actual scale. Mod-switch levels and normalize branches explicitly when histories differ; metadata relabeling of scale is forbidden.

Forward RE output normalization includes its repeated output mask; CE's first-column mask includes its valid output-row support before replication. Activation normalization includes the correct vector mask. Loss normalization includes the model-output mask and loss factor. Derivative factors include their vector support masks, especially PolyRelu2's `+1` term. Final gradient normalization includes the matrix active-entry mask, so gradient padding cannot accumulate as optimizer state. Bias gradients sum valid `dz` vectors, then apply their output mask and `lr/n` at exact Delta. Biases use SGD. No-bias eMLP layers store no bias ciphertext; this difference from main's frozen-zero bias representation is visible in counts, while the plaintext function remains equal.

Use the fixed-reference NAG convention already chosen in [interactive profile design](interactive-profile-design.md):

```
g_t = sum_valid(dW_sample) / n
U_t = r * (1 + mu) * V_t,       V_0 = 0
h_t = lr_t * (1 + mu) * g_t
h_ref_t = r * (1 + mu) * g_t
kappa_t = mu^2 / (1 + mu) * (lr_t / r)
U_(t+1) = mu * U_t + h_ref_t
W_(t+1) = W_t - h_t - kappa_t * U_t
```

The last line uses **old** `U_t`. First update: `U_1=h_ref_0`, `W_1=W_0-h_0`; do not substitute `h_0` for `h_ref_0` when `lr_0 != r`. At `lr==r`, reuse one gradient branch. Otherwise share the outer-product sum/relinearization/rescale, then emit two masked scalar branches. `r` is positive and fixed before state initialization, at least the maximum bound mean-gradient rate, including varying valid batch sizes. `lr>r`, nonpositive/nonfinite rates, invalid derived factors, or a changed momentum/reference after initialization fail before work. SGD uses `W'=W-lr*g` and needs no reference ceiling. Refresh preserves the numerical meaning of `U`, even when its ciphertext is temporarily removed; no velocity reset/rescale on schedule changes.

## Concrete level and exact-scale schedule

Use a conservative canonical-Delta implementation first, not an undocumented optimized scale trick. Every value at a layer API boundary has exact `Delta=params.scale()`. Natural product scales remain real intermediate scales; use `target_plain_scale`, `mul_plain_to_scale` and rescale for normalization. A public scalar/mask multiplication is counted even when its value is1. Encoded masks/factors are uploaded through `HeOps`, and encode/upload time is included in its declared scope.

For a ct-ct product at common level `l`, rescale yields level `l-1`, scale `s_a*s_b/q[l]`. A normalization of ciphertext `(l,s)` to Delta uses plaintext scale `exact_product_scale(l,Delta)/s`, then one ct-pt product/rescale, ending at `(l-1,Delta)`. Preflight checks representability and enough levels. Mod-switch alone changes level, never scale. Encrypted targets originate at Delta; they mod-switch to the normalized prediction level, permitting encrypted subtraction with no label re-encryption per step.

| Operation | Output level and scale; exact expensive operations |
|---|---|
| RE affine | `min(input,W)-2`, Delta: existing `vecmat` (tensor/relin/rescale, `log2 r` rotations) then one output-mask normalization. |
| CE affine | `min(input,W)-2`, Delta: existing ct product plus expanded output-mask/normalization; `2 log2 c` rotations, one ct-pt, two rescales total. |
| Encrypted bias add | Bias mod-switched to affine level, both Delta; no multiply/rescale. |
| Square + canonical mask | `z-2`, Delta: one tensor/relin and two rescales, one ct-pt. |
| PolyRelu2 + canonical mask | `z-2`, Delta: existing forward (one tensor/relin, one ct-pt, two rescales) plus mask normalization (one ct-pt/rescale). Parallel branches consume two depth levels total. |
| Identity | Input unchanged; support was checked/masked by affine. |
| Loss delta | `prediction-1`, Delta: encrypted subtract then one output-mask/loss-factor normalization. |
| Nonconstant derivative | `z-1`, Delta: existing masked derivative factor, one ct-pt/rescale, optional public add. |
| `dz` for Square/PolyRelu2 | `min(derivative,upstream)-1`, natural product scale: one tensor/relin/rescale. Identity uses upstream directly. |
| Backward input | Reverse-oriented affine, depth2 with canonical input-vector support; natural `dz` is allowed at the kernel input, output Delta. |
| Matrix outer sum | Common pair level `p`, natural scale after one rescale, level `p-1`; `n` tensors, one lazy relin or `n` relins. |
| Folded matrix gradient(s) | Each branch at `p-2`, Delta: one active-matrix mask/scalar ct-pt/rescale per branch; shared outer sum. |
| Bias folded gradient | Sum same-history `dz`, then one ct-pt/rescale to Delta with `lr/n` and support mask. |
| Later NAG state | `mu*U` and `kappa*U` each normalize to Delta at gradient level, by starting at `gradient_level+1`; two ct-pt/rescales per matrix, no rotations. First update has neither. |
| State projection before bootstrap | One role-specific support mask/rescale at exact Delta, then free mod-switch to0. Reserve this level in admission. |

Every layer's weight/bias/velocity may finish at different levels. Store those levels; align only the required operands. A layer's optimizer requires old state sufficiently high for its scalar paths, and aligns old weights with gradients without relabeling scale. Reference and current gradients share levels and Delta. Bias and matrix state stay separate; there is no diagonal compact/unpack stage.

A concrete no-AA two-layer witness with all starting state/inputs at level `L` makes the conservative budget auditable:

| Trace | eMLP PolyRelu2/identity | 2020 Square/Square |
|---|---:|---:|
| `z1, a1, z2, prediction` | `L-2,L-4,L-6,L-6` | `L-2,L-4,L-6,L-8` |
| loss delta, `dz2` | `L-7,L-7` | `L-9,L-10` |
| hidden upstream, `dz1` | `L-9,L-10` | `L-12,L-13` |
| folded `h2,h1` | `L-9,L-12` | `L-12,L-15` |
| lowest state after boot-input projection | `L-13` | `L-16` |

Thus the source `n16_boot` bootstrap output level16 can admit this conservative two-layer schedule including projection;2020 has no spare lowest-layer level. This is an algebraic witness, not measured precision or a substitute for implementing the actual level/scale planner. `n16_train` starts from its source preset maximum level. Derive budgets from actual params and `BootstrapOps::output_level()`, never change the preset to make a case pass. AA, mixed state levels and resumed WN steps require DAG simulation with the recurrence above, not subtraction of one universal depth number. Require predicted trace to equal executed trace. If exact-scale feasibility exposes an additional level, revise the plan and report the gap; do not launch under a formula-only admission.

## Refresh placements, ranges and local blocks

I uses explicit client state refresh after every stage update, and explicit hidden handoff refresh for local stages. It is a distinct mode; WN/AU/AA are NI placements, not aliases for client refresh. Refresh carries typed state metadata and returns Delta at the declared input level. The client may recover canonical matrix/vector coordinates and repack for precision, but must not perform optimizer arithmetic or decrypt gradients. Record download, client decrypt/repack/encrypt and upload times/bytes separately; server updates include only their declared work.

NI placements use the same learning function and fixed-reference state:

- **AU:** after each stage update, project and bootstrap every matrix weight, initialized velocity and present bias. No activation bootstrap except the required local-stage handoff.
- **WN:** no ordinary hidden activation bootstrap. Simulate the next step using its public batch/schedule metadata and state levels; bootstrap all stage state if it cannot fit including a subsequent mandatory projection reserve. If future metadata is unknown, use the conservative supported branch. Never consume the last projection level and then attempt recovery. Local-stage handoff remains explicit.
- **AA:** bootstrap each stage's hidden activation after its canonical activation mask. State refresh uses the same next-step fit rule as WN, with the AA DAG. The bootstrap output is retained as the canonical post-bootstrap activation accessor. Local handoff reuses that exact value (clone or mod-switch if required), with **no second projection, normalization, repack or bootstrap**. Pre-bootstrap hidden snapshot remains a separate API value.

Every bootstrap receives a degree1/two-component ciphertext, exact Delta, level0 (explicitly mod-switched), matching context, and all slots within its declared input range. The current bootstrap's mathematical precondition is `|slot|<=1`; it applies to weights, reference-scaled velocity, biases and activations, including padding/error. Masking removes structural garbage; it does not prove active-value bounds. Store per-role range contracts and report diagnostic range/precision evidence separately. NI cannot decrypt to enforce the range. An out-of-range diagnostic must be reported as a range/precision failure or unverified condition; no clipping, secret-derived scaling, optimizer reset or silent profile tuning. A new public normalization convention would need separate design/ruling and matching count/depth/reference tests.

For each local block, use its own classifier target, loss, optimizer and state. Cache its first-layer hidden activation computed from old weights, then train the next stage on that snapshot. No gradient crosses a stage boundary; do not add local losses or backward through the handed-off ciphertext. Sequential stage updates are mathematically equivalent to the captured reverse update order because later stages read only old forward activations and not preceding updated weights. Keep independent plaintext tests of this equivalence and graph stop-gradient boundaries. Inference follows backbone hidden activations and the final output stack, ignoring local classifier outputs.

With `n` valid sample ciphertexts and `q` two-layer stages, AA has `n*q` hidden activation bootstraps per logical update. There are `q-1` handoff boundaries: WN/AU add `n*(q-1)` handoff bootstraps; AA adds zero beyond its existing activations. A stage-state refresh event has `2 + 2*[NAG] + 2*[bias]` ciphertext bootstraps for these registered two-layer stages (velocity exists after update1). WN/AA event counts derive from each stage's level simulation, not a fixed per-update guess. All counts include heads/output stack. Do not use the diagonal compact-container bootstrap formula here.

## Counters, keys and memory

Build one execution DAG and a pure predictor with the same declared primitives but independent counter assertions. Predict every `OpCounter` field, not only KS: rotations, relins, tensors, ct-pt, rescales, SSE switches, bootstraps. Counts are summed over valid samples, all stages, current/reference branches, bias paths, state projections and refresh events. Report public additions/encoding/upload bytes and timings separately when absent from `OpCounter`.

For an ordinary eMLP two-layer RE/CE stage, put `A=log2 r`, `C=2 log2 c`. The per-sample forward/backward rotations are `2*A+C`; for a CE/RE stage they are `A+2*C`. Outside outer sums, both use5 tensors/relins per sample (two affines, hidden polynomial square, backward affine and hidden `dz`). Two matrix outer sums add `2*n` tensors and either2 lazy relins or`2*n` per-sample relins. Therefore a no-bootstrap RE/CE eMLP stage has `rot=n*(2*A+C)`, `tensor=7*n`, `relin=5*n+2` for lazy, or`7*n` for per-sample. The conservative canonical masks add ct-pt/rescales, not rotations/relins. Heads, reversed stage orientations, Square outputs, state scalar paths and bootstraps must still be added according to the table; no universal37-KS claim follows.

Scheduled NAG `lr!=r` adds one ct-pt/rescale per weight matrix to create `h_ref` from the shared outer sum, including update1, with no extra tensor/relin/rotation or critical depth. Later updates add two velocity ct-pt/rescales per matrix regardless of the rate branch. Unlike diagonal replicated gradients, packed matrices need no lane replication rotations for the second gradient. Constant-rate operation order stays stable within the new trainer; this is not ciphertext identity with the legacy client-updated path.

The effective lazy policy is `GradRelin::Lazy && EvalConfig.lazy_relin`; record requested and effective policies. Existing BSGS/hoist flags do not magically accelerate these sequential dependent rotations. A predictor must record their actual applicability. Bootstrap predicted cost comes from the configured bootstrap's cost, never a global constant. Actual `HostBootstrapper` merges host bootstrap counts into the device evaluator once; do not add `boot_count*KS` again to an already inclusive measured count.

Required evaluation keys are the deduplicated union of each admitted lattice's `row_sum_steps` and `expand_steps`, relin keys, and configured bootstrap keys. No transpose or diagonal BSGS keys are implied. Verify keys before work. State-only masks and scalar paths require no rotation keys. Allow a shared key context only with explicit reset of all learning state, rate basis, receipts and per-case timer/counter origin.

Memory admission occurs before key generation/device allocations and has host/device columns. Size actual RNS representations from parameter degree, level, component count and backend word size; a degree1 ciphertext payload is `2*N*(level+1)*word_bytes`, with separately accounted object/alignment/backend overhead. Include public key/evaluation-key host copies, device expanded keys and level caches, bootstrap keys/CPU scratch, all weight/U/bias ciphertexts, old and candidate new state during commit, sample inputs/stage targets, cached `z/a/delta`, degree2 lazy accumulators (three components), both gradient branches, masks/plaintext caches, transfer buffers and allocator workspace. Label estimates as estimates, expose source assumptions and live-object lifetimes, and do not guess a VRAM number or fit from input slots alone.

Use sequential GPU operations. A bounded sample-group implementation may stream forward/backward and accumulate unrescaled gradients, retaining only required local handoffs; it must preserve one logical update and normalization. Group size is a resource choice recorded separately from logical B. Choosing a group size after preflight must not change order/rates/tail policy or replace a row with a different model. Sum accumulation order and numerical tolerance are explicit. Report need/free with exit3 only for a capacity preflight rejection. Unsupported geometry/code or missing input is a different state; runtime OOM is failure.

At epochs/registered update-boundary intervals, release transient own buffers and invoke `trim_pool(0)` using the established backend hook. Record pool-used and pool-reserved high-water reset boundaries and sampled process VRAM separately; none is an estimate or checkpoint maximum disguised as process peak. Setup, synchronized server update, CPU bootstrap, transfers/client refresh, diagnostics and end-to-end time have explicit disjoint/inclusive definitions. Preserve D6 cumulative-counter origin and interval-conservation rules, especially B128 crossing boundaries. Store source hash, binary hash/build flags, config/profile/input/order hashes, backend/host/thread/security/placement/rate/policy identity inside the binary output. Hardware not tested means hardware unmeasured.

## Minimal primitive extensions and staged ownership

Existing `layout_reboot.rs` already provides generic product/row sum/expand/outer-sum with correct rotation keys and basic preflight. Three concrete packed adapters/extensions are needed; none changes the legacy public behavior:

1. `expand_valid_to_scale(..., valid_rows, factor, target)`: restrict the existing first-column mask to actual output rows before replication; retain its rotations/depth and add no extra ct-pt. Existing `expand*` remains an all-rows wrapper. This handles CE structural padding with the normalization already present.
2. `outer_sum_scaled_masks(..., constants, matrix_mask, target)`: reuse the existing outer-sum product/relin/rescale once, then return one/two exact-scale masked branches. Base implementation may call `outer_sum` once and apply new module-local public-mask helpers; no duplicate outer products. Validate every branch before work. Bias sum uses the same exact-scale helper with vector masks.
3. Checked metadata/probe support for natural intermediate scales: model `vecmat`/`vecmat_col`, activation derivatives, target subtraction and normalization using actual `q[level]` and reachable scales. Keep a separate declarative trace, rather than assuming every CKKS multiply lands at Delta. No change to D2a's Dense/optimizer files is needed.

A per-layer lattice conversion, matrix tiling, native GPU bootstrap and automatic range rescaling are **not needed for the registered witness geometry** and are not part of the initial implementation. If a newly registered shape cannot use the common lattice, it remains an explicit extension request rather than being silently removed.

| Package after root authorization | Owned new files / narrow shared changes | Acceptance before advancing |
|---|---|---|
| D4a geometry/primitives | `nn/src/packed_training/{mod,plan,types,ops}.rs`, `nn/tests/packed_layout.rs`; narrow opt-in method in `layout_reboot.rs`, `lib.rs` export | All registered geometry witnesses and masks; negative capacity/shape/key/scale; generic CPU primitive values/counts. |
| D4b encrypted stage/update | `packed_training/{optimizer,stage}.rs`, `nn/tests/packed_training.rs` | Independent plaintext updates incl2020 bias and scheduled persistent NAG; encrypted labels; partial groups; exact trace/counters. |
| D4c refresh/NI/blocks | `packed_training/{refresh,trainer,block}.rs`, `nn/tests/packed_training_ni.rs` | Real CPU bootstrap small fixtures, WN/AU/AA, no-secret NI type, nonduplicating AA handoff, local-loss isolation, state ranges recorded. |
| D4d backend/profile bridge | new `bench/src/bin/packed_training.rs`, `bench/src/packed_profile.rs`, focused tests/CUDA generic fixtures | Consume D1/D2b binder; pre-keygen memory and provenance; CPU/CUDA build and authorized hardware parity later. Reserve shared binder/preflight edits with their owner first. |
| D4e campaign readiness | Runner/schema integration by D6/D7 owners | Independent code review, binary identity, required inputs/FP32 locks and root scheduling; no full blocks until user decision. |

Paths above abbreviate `crates/ckks-train-*`. D2a owns existing `optim.rs`, `trainer.rs`, `dense.rs` and their tests; D2b owns shared profile binding/preflight. D4a owns the geometry/primitives paths listed above and this document; it must not edit those concurrent sources. Each later package freezes its owned diff for independent review before integration. Ordinary subset code preparation is not an actual research run or launch authorization.

## D4a implementation evidence and remaining boundary

`packed_training::PackedGeometryPlan` is a **geometry-only** checked constructor over generic `StageShape` metadata. It constructs the shared lattice and every stage's two edges, including the final output stack, with alternating backbone directions and opposite local classifiers. `LayerGeometry` carries the graph definition and layer ordinal; `VectorSpec` checks each actual axis. Batch size remains a count of sample ciphertexts. The smallest feasible column power is sufficient: increasing it only reduces row capacity, so failure at that column bound proves no factorization satisfies these fixed orientations. This is not the proposed full `PackedPlan`: optimizer, activation, mode/placement, levels, keys for bootstrap, memory and schedule admission remain future work.

`PackedWeights` and `PackedVector` validate ciphertext context, slot/component/level metadata, with exact Delta for weights and positive finite natural scales permitted for derivative vectors. Their constructors assert the caller's encrypted representation; they cannot inspect ciphertext contents. Packing uses the original logical output-by-input arrays and explicit row/column mapping. `PackedGradient` records layer, valid pair count and the exact public factor applied to the sum; no update generation or optimizer policy is implied.

Generic `affine` executes RE/CE multiplication with structural output masks, ending at exact Delta after two levels. The new opt-in `expand_valid_to_scale` restricts CE's first-column mask to valid output rows without extra depth; all legacy method bodies remain unchanged. `outer_sum_scaled_masks` performs one encrypted sum of valid sample products and returns one or more active-matrix masked constant branches at exact Delta, sharing tensor/relin/rescale work. It does not average implicitly or accept physical invalid records: the caller supplies precisely the valid pairs. All shapes, context, keys, scale history, factors and branch encoding are checked before counted work. Runtime backend failures still do not roll back completed operations. CE preflight currently performs a redundant host encoding before the helper's runtime encoding/upload; both belong in measured primitive wall time.

Focused executable evidence: `cargo test -p ckks-train-nn --test packed_layout --test layout_reboot -- --test-threads=8` passed **9 new tests and 12 legacy tests**, with the 2 pre-existing large-keygen tests ignored. Build/Rayon threads were capped at8. New encrypted fixtures use small N=1024 CKKS contexts solely as correctness oracles; the 32768-slot witnesses for every primary dataset/model graph and the 784-input eMLP-1 supplement are pure metadata tests, not large-keygen or training runs. Independent full-slot references check nonsquare RE/CE values, trained padded-class channels, removal of deliberate structural garbage, exact levels/scales, all `OpCounter` fields, single/unequal valid-pair counts, mixed operand levels and natural input scales, one/multiple constant branches, both requested relin policies and disabled-lazy fallback. Negative cases cover dimensions, capacity, context, components, missing keys, levels, mixed product histories and invalid/unrepresentable constants/scales with zero counted work. These tests do not establish encrypted training, actual NI, GPU behavior, production precision, memory fit, baseline-c completion or Task54 completion.

## Alternatives and acceptance gates

A direct GPU port of the old bench was rejected because it decrypts gradients/client-updates weights and uses plaintext labels. Reusing diagonal `Optimizer` was rejected because its containers/replication semantics do not describe packed matrices. Stage-local RE resets with arbitrary lattice changes were rejected because they require new encrypted permutations and invalidate a no-repack claim; the shared-lattice backbone plan fits current shapes. An unscaled velocity is mathematically possible but diverges from the already selected fixed-reference convention. Secret-side state rescaling, clipping, momentum resets and omitting profiles change the experiment. Aggressive fused natural-scale arithmetic may reduce ct-pt/depth later, but must be a separately reviewed equivalent path with its own predictor; correctness and inspectable counts come first.

After authorization, use small original synthetic encrypted CPU fixtures and an independently coded plaintext reference, never production datasets for development tuning. Required gates:

- Multi-update SGD and NAG parity for each layer orientation; at least four scheduled updates including first rate below reference, decrease, restart to reference and persistent velocity decoded separately for the test. Distinguish stale-rate/reset-velocity wrong references. Include2020 both Square outputs and bias updates.
- Encrypted target subtraction with real/padded classes, support masks, non-power-of-two dimensions, n=1 and48 as logical examples, retain-tail and unequal partitions (32+16 plus a smaller partial example). Show equality to one logical mean/sum update and no gradient from invalid sample records or structural padding. Trained eMLP padded-class outputs must still contribute.
- All four graph plans and all registered dimension/batch/mode cases admitted or explicit gap reports; malformed lattice, cross-stage target representation, missing keys, insufficient level/projection reserve, wrong exact scale, invalid rates and insufficient preflight capacity fail before work where promised.
- Actual counter deltas equal full predicted fields, not only relins. Assert lazy/per-sample, current/reference branch, first/later update, both stage orientations, bias/no-bias, all placements and negative paths. Verify predicted trace equals every actual level/scale boundary.
- NI type/static API has no secret/decryptor/client refresh route. Real small CPU bootstraps verify state and velocity, slot ranges, padding error and repeated updates; a fake bootstrap/level probe alone cannot establish NI numerical correctness.
- eMLP-2/3 stop-gradient graphs: independently vary a local classifier target or a later-stage target and assert only the appropriate stage gradients change, holding that update's old forward snapshots fixed. Verify handoffs use old weights and prediction uses only final stack. AA handoff aliases/accesses the post-bootstrap ciphertext, with zero extra boot; WN/AU have counted explicit handoffs. Compare handoff layout/Delta/levels and numerical values.
- Compile generic CPU and CUDA consumers without training. After independent review and explicit hardware scheduling, run small original CPU/CUDA numerical/count parity on the actual GPU with declared host-bootstrap path, memory measurements and synchronization. No available/sufficient GPU means unmeasured, not passed or infeasible by inference.
- Profile bridge parity verifies bound manifests/arrays/order/rates/tails and exact selected security preset. Record code/build/hash provenance and all measurement boundaries. Full encrypted block work remains held even if small tests/subset preparation pass.

No timing, speedup, precision, security-equivalence, memory-fit or campaign-coverage claim is made by this design.

## D4b1 API refinement before implementation

D4b is split into D4b1 matrix optimizer correctness and D4b2 the full encrypted
stage (including biases). D4b1 uses an immutable `PackedOptimizer` policy and
`PackedLayerState`, with a fixed momentum/reference basis from construction.
There are no policy setters or state-removal APIs in this slice. Each state
issues a `PackedUpdateRequest` for one rate, actual valid-sample count and update
index. Its branch factors are checked before gradient work; `bind` checks the
D4a gradient layer, sample count, factors and required branch count. Bound
`PackedUpdateGradients` retain a private state-generation token. Preparation
rejects stale or unrelated tokens before evaluator work, prepares new encrypted
weights and fixed-reference U without modifying the old state, and returns a
candidate whose `into_state` is infallible. A future stage prepares all matrix
and bias candidates before assigning any of them. Device failures can leave
operation counters advanced but cannot partially mutate the old layer.

A private allocation-identity token is used in addition to the numerical update
index: an index alone would admit gradients from a separately constructed state
at the same layer/index. Requiring a borrowed state until commit would prevent
useful owned stage preparation; publicly mutable IDs would allow accidental
rebinding. Raw D4a gradients remain caller-asserted ciphertext contents: binding
checks metadata but cannot prove when the encrypted derivative was computed,
that it is numerically a derivative, or stop a caller deliberately binding old
raw gradient ciphertexts anew. Already-bound gradients cannot silently pass to
a different generation. Tokens are local runtime metadata, not cryptographic
freshness proofs or serialized checkpoint IDs.

Tests use a small explicitly insecure, sufficiently deep CKKS chain with
decreasing gradient levels across four updates. This avoids introducing a
production refresh setter merely for fixtures. Refresh APIs, checkpoint restore,
bias updates, stage execution, NI and GPU remain D4b2/D4c/D4d gates. Immutable
policy is intentionally stricter than the legacy mutable optimizer API; it
preserves the registered learning function without adding reconfiguration
semantics that the packed stage does not yet need.

The new packed policy accepts conventional `0 <= mu < 1`, including the
registered `mu=0.9`. Keeping the legacy broader finite algebraic domain was an
alternative, but would add unregistered negative/expansive momentum semantics
to a new API. D2a's legacy optimizer API and domain are unchanged. `mu=0` is a
valid zero coefficient, not an underflow error: later updates still execute the
two scalar products/rescales, so their declared counts are unchanged. Positive
folded f64 factors must be normal (not zero/subnormal/overflow); at preparation,
nonzero state scalars that would round to zero at the actual plaintext scale
are rejected before either scalar operation. This is representability checking,
not an accuracy/range admission claim for an eventual training stage.


## D4b1 local implementation evidence

`PackedLayerState::request` selects one folded branch for SGD/equal-rate NAG,
or current/reference branches for scheduled NAG. `PackedUpdateRequest::bind`
checks exact layer identity, actual sample count and public factors.
`PackedLayerState::prepare` validates its private generation, ciphertext context,
components, history/levels and exact Delta before preparing the complete new
matrix state. First NAG update stores the reference branch even below reference;
later updates use old U for both scalar terms. Weight subtraction aligns to the
minimum participating level. No matrix is cast into diagonal optimizer storage.

Six small N=1024 CPU tests pass alongside all nine D4a packed-layout tests.
They check both nonsquare orientations and every structural slot, four rates
`[0.04, 0.00008, 0.02, 0.08]` against an independent unscaled-V f64 recurrence,
SGD rate increases, actual sample means, decoded W/U/V with absolute tolerance
1e-7, exact Delta/levels and all operation-counter fields. First NAG update has
zero optimizer scalar work; later updates have exactly two ct-pt products and
two rescales, no tensor/relin/rotation/key-switch work, including mu=0.
Equal rates use one D4a outer-sum branch. Invalid rates/basis configuration,
branch factors/count/sample metadata, layer identity (including equal-shape
other graphs), stale/unrelated generation, contexts, scale/components,
inadequate U levels and unrepresentable nonzero scalar terms are rejected
before optimizer counted work. Old state remains unchanged during preparation.

The fixture's ten-prime chain and N=1024 are explicitly insecure development
parameters. No bootstrap or client refresh is used. This establishes encrypted
matrix optimizer correctness only: biases, activation/loss/backward composition,
atomic stage assignment, full level/range/precision/memory admission, refresh,
NI, GPU and campaign execution remain later gates. Runtime device failures
are outside this CPU evidence and may advance counters without yielding a
candidate; no cross-backend transaction is claimed.
