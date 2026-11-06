# Matched interactive training profiles (Task54 D2a design)

Status: root-ratified design, implemented, locally validated and independently reviewed (including the scoped Clippy fix). The design was written and ratified before code changes. This library work implements part of the independently reviewed D0 experiment matrix at `8eabcdd`. It does not implement its runner, remove benchmark CLI restrictions, or authorize matrix training.

## Required learning semantics

Interactive and noninteractive instances of a concrete profile must use the same arrays, sample order, loss, valid-batch normalization, optimizer and per-step learning rates. The existing interactive loss delta is `prediction - target`; weight gradients are means over valid samples. Consequently a registered batch-sum rate becomes `lr_mean = lr_sum * n_valid`, supplied before each update. D2b owns schedule loading, sample alignment and that conversion.

The eMLP profiles require NAG with persistent velocity and zero initialized, frozen biases. For a mean weight gradient `g_t`, the independent f64 reference is

```
V_(t+1) = mu * V_t + g_t
W_(t+1) = W_t - lr_t * (g_t + mu * V_(t+1))
V_0 = 0
```

The existing interactive `Nag` stores `U = lr * (1 + mu) * V`. Changing only its public `lr` after an update changes the meaning of its state and gives the wrong momentum term. Resetting velocity at a rate change also gives a different learning problem.

## Chosen representation

Use the existing NI mathematical convention, with a fixed positive reference rate `r` chosen before state initialization, normally the maximum mean-gradient rate of the schedule:

```
U_t = r * (1 + mu) * V_t
h_t = lr_t * (1 + mu) * g_t
h_ref_t = r * (1 + mu) * g_t
s_t = lr_t / r
kappa_t = mu^2 / (1 + mu) * s_t
U_(t+1) = mu * U_t + h_ref_t
W_(t+1) = W_t - h_t - kappa_t * U_t
```

At the first update, `U_1 = h_ref_0` and `W_1 = W_0 - h_0`. Both gradients are required even at that first update if `lr_0 != r`. At `lr_t == r`, reuse `h_t` as `h_ref_t`, use an exactly `1.0` factor, and retain the legacy arithmetic operation order. Later decreases and increases/restarts up to `r` need no velocity rescaling, re-encryption or reset. Rates above `r` are rejected; callers must register the maximum before the first update. SGD has no such upper bound.

`mu` is fixed once velocity has been initialized or supplied. The stored reference rate and momentum basis are metadata of `U`, not values to derive anew from public mutable fields. The implementation must retain that metadata even when `take_velocity()` or `set_velocity(None)` temporarily removes ciphertexts.

## Public API and compatibility

Keep the existing constructors, `Sgd { lr }`, `Nag.lr`, `Nag.momentum`, `Optimizer::step` and `LayerOptimizer` requirements source-compatible. New trait methods have default implementations.

| API | Contract |
|---|---|
| `Sgd::set_lr(lr) -> Result<()>` | Require finite positive rate, then change the next gradient coefficient. No state and no reference ceiling. |
| `Nag::set_lr(lr) -> Result<()>` | Validate finite positive rate, `lr <= r`, finite usable folded factors and the stored momentum basis; change only current rate. |
| `Nag::set_lr_reference(r) -> Result<()>`, `lr_reference()` | Default reference is constructor rate. Reference must be finite, positive, at least current rate, and set only before any velocity initialization/supply. No homomorphic operations. |
| `Optimizer::reference_grad_scale() -> Result<Option<f64>>` | Default `Ok(None)`. NAG validates its configuration/basis, then returns `Some(r * (1 + mu))` only when its current rate differs from `r`. Trainer preflights all these queries before forward work. |
| `Optimizer::step_with_reference_gradient(ev, w, dw, reference: Option<&Diagonals<_>>) -> Result<()>` | Default delegates to existing `step` for `None`, rejects `Some`. NAG validates and consumes both folded gradients. Existing NAG `step` delegates to the same implementation with `None`, rejecting a missing reference gradient when one is required. |
| `LayerOptimizer::validate_lr(lr) -> Result<()>` | Read-only preflight; default rejects schedule control as unsupported. Built-ins validate their full rate/basis constraints. |
| `LayerOptimizer::apply_lr(lr)` | Infallible commit after successful validation; default does nothing because the default validation rejects. Built-ins assign only current rate. Custom implementations opting into rate control must make this the non-failing commit of the validated value. |
| `Trainer::set_lr(lr) -> Result<()>` | Reject malformed rates before calling optimizers; validate every layer, then call every infallible `apply_lr`. Validation failure cannot partially alter a multi-layer rate set. Applies to subsequent updates, with no velocity reset. |
| `Trainer::with_frozen_biases()` | Builder, default off. Sets the bias-gradient folded coefficient to exactly zero; all existing bias operations/refreshes remain. |

The two-phase layer interface deliberately separates fallible validation from assignment. It avoids pretending a sequence of arbitrary custom fallible setters is atomic, and avoids requiring custom optimizers to be cloneable. The guarantee covers validation failures; a custom implementation that panics or violates its infallible commit contract is outside it. Existing custom optimizers remain usable with unchanged constant-rate training and get an explicit error from `Trainer::set_lr` until they opt in.

Public NAG field writes remain possible for source compatibility, but cannot silently change the basis of existing `U`. A direct valid `lr` write is interpreted as the current schedule rate by the reference-aware trainer; direct `step` requires the extra gradient when appropriate and fails if it is absent. An out-of-range/nonfinite rate or a changed momentum after state initialization is rejected before optimizer mutation; trainer preflight also rejects it before forward work. Momentum changes before state initialization are permitted only when all derived factors remain valid. `set_lr` is the recommended checked interface. The constructor/reference setter must not allow nonfinite or zero derived reference factors into scheduled updates.

`velocity()`, `take_velocity()` and `set_velocity()` retain their ciphertext types and signatures. Their documented scale becomes `r * (1 + mu) * V`, identical to the legacy meaning at constant rate. Refreshing the existing state at the same exact CKKS scale is unchanged. Supplying velocity for the first time locks the current momentum/reference basis; supplying it subsequently must use the already locked basis. Values of externally supplied encrypted state cannot be inspected by the server, so that semantic obligation is documented while shape/level/CKKS-scale checks remain enforced. Removing state does not unlock the basis. Changing momentum after initialization fails explicitly even if state was temporarily taken out.

The reference-gradient entry point checks both gradients' shape, uniform level, exact weight scale, mutual level equality, and required/extra-reference status before work. Existing velocity must share shape/scale and have a level strictly above the gradients. The stored basis is committed only after a successful optimizer update; both gradient branches and the complete new weight/velocity are prepared before assignment. This does not claim general rollback of arbitrary custom optimizer failures or refresh failures across a whole trainer step; the existing trainer has no such transaction mechanism.

## One narrow Dense extension

Add `Dense::weight_grad_scaled_consts(ev, delta, a_prev, layout, cs) -> Result<Vec<Diagonals<_>>>` in `dense.rs`. This is the only additional source ownership requested beyond `optim.rs` and `trainer.rs`.

It produces the same pre-rotated, fully replicated layout as `weight_grad_scaled`. Share input rotations, each ciphertext product/relinearization/rescale and batch reduction. For each supplied constant, apply its own lane-zero mask/rescale and lane replication. Validate nonempty finite constants and all original shape/key/level/scale constraints before evaluator work. Keep `weight_grad_scaled` as the one-constant wrapper with the same operation order and arithmetic result. Factor only the batch reduction/mask/replication needed here; do not alter NI's compact-gradient methods or their first-period masks.

Trainer requests one gradient for legacy/SGD layers, or `[current_scale / n_valid, reference_scale / n_valid]` for a scheduled NAG layer. It passes both results to the reference-aware optimizer step. All computation uses generic public `HeOps`; no decryptor, secret key, client-side velocity conversion or backend-specific implementation is added.

Frozen biases preserve their initialized values in the plaintext model. Zero-initialized biases therefore implement the registered no-bias profiles; freezing a nonzero bias does not zero it. Keeping the zero-folded gradient and ordinary refresh follows NI's convention and preserves operation counts and levels. CKKS comparisons use declared tolerances; ciphertext equality or exact decoded zero after refresh is not promised.

## Operation counts and levels

Let `m` be the layer's stored diagonal count, `B` its physical power-of-two lane count, and `l = min(dz.level, a_prev.level)`. Both gradient branches finish at level `l - 2` and exactly the weight scale. Sharing products is essential: a second complete gradient computation is unnecessary.

| Cost relative to the unchanged single-gradient path | `lr == r` or SGD | NAG `lr != r` |
|---|---:|---:|
| Ciphertext-plaintext products | 0 | `+m` |
| Rescales | 0 | `+m` |
| Rotations / key switches | 0 | `+m * log2(B)` |
| Ciphertext tensor products / relinearizations | 0 | 0 |
| Additional critical-path levels | 0 | 0 |
| Additional refreshed ciphertexts | 0 | 0 |

The extra rotations replicate the reference gradient; NI's compact lane-shift formula is not applicable to this layout. For later NAG updates, existing `mu * U` and `kappa * U` still cost `2m` ct-pt products and `2m` rescales on velocity at gradient level plus one. Only the second scalar's value changes. The initial update has neither velocity product. The two-gradient overhead also applies to an initial update below the reference rate. Frozen biases retain all bias counts, depth and refreshes. For multiple layers, add the overhead only for layers whose current rate differs from reference.

`Trainer::trace`, evaluator counters and `EpochLog` remain measurements of actual operations. The old constant-rate simulator/benchmark formula is not a scheduled-profile prediction. D2b must incorporate the extra replicated-gradient cost in its plan/prediction and record the reference/current rate condition. No simulator or benchmark claim is changed by D2a alone.

## Alternatives considered

1. **Rescale stored velocity when the rate changes.** Algebraically possible, but a restart can magnify accumulated CKKS error; a public ciphertext multiplication/rescale also consumes a level, adds counts, and complicates refresh ordering. Rejected in favor of NI's established bounded `lr/r` convention.
2. **Rescale velocity inside client refresh.** Would add secret-side optimizer semantics and a special refresher contract, obscure server/client counts, and be unsuitable for a generic `HeOps` trainer. Rejected.
3. **Store unscaled `V`.** Sound algebra, but requires an additional gradient scaling path or different encoded factors and loses the simple legacy correspondence. The fixed reference is already an unchanging linear encoding of `V` and matches NI.
4. **Compute two complete gradients.** Correct, but duplicates tensor products, relinearizations and reductions. The small multi-constant Dense helper preserves the existing critical depth at lower extra cost.
5. **Only mutate `lr`, reset momentum, or skip unsupported matrix profiles.** Each changes the registered learning problem; rejected.

## Acceptance and local validation

Write failing focused tests first. Use existing small synthetic CKKS fixtures, fixed seeds and the existing numerical tolerances; no downloaded data or full training.

- At least four NAG updates against an independent f64 reference with persistent unscaled `V`, including a first rate below reference, decrease and restart/increase to reference; also a large drop/recovery where numerical tolerance allows. Check every update's weights and (optimizer test) decoded reference-scaled velocity. A reset-velocity or stale-rate reference must be visibly distinguishable.
- SGD changing rates match independent updates and permit a finite increase above the initial rate.
- Freeze zero-initialized biases while weights change, including multiple updates and a partial valid batch. Default bias updates remain active. Verify frozen/nonfrozen operation counts and level traces are the same for otherwise matched runs.
- Zero, negative, NaN and infinite rates; above-reference NAG rates; late reference changes; changed public momentum after state creation/take/refresh; malformed reference gradients. Validate errors before state changes and, where specified, before evaluator counters move.
- Multi-layer rate validation: a valid first layer followed by a rejecting built-in/custom layer must leave the entire rate set unchanged. An existing custom optimizer implementing only the old trait requirements must compile and train; rate control must fail explicitly.
- At constant rate, repeated `set_lr(current)` versus never calling it produces identical seeded decoded results, counts, levels and refreshes. Retain existing `grad` and `trainer` tests, including the level-zero seal_2020 path.
- Two-constant Dense results match independent one-constant gradients in value/scale/layout/level. Assert the exact additional `m` ct-pt, `m` rescale and `m log2(B)` rotations with no tensor/relinearization/extra-depth cost, including `B=1` when practical. Assert full-step scheduled counters versus the constant-rate path and the analytical delta.

Run scoped `cargo test -p ckks-train-nn --test grad --test trainer -- --test-threads=8` with `CARGO_BUILD_JOBS=8` and `RAYON_NUM_THREADS=8`, plus focused new test target(s) if separated. Run an appropriate compile check for generic consumers without executing production training. Format only owned paths. Record RED/GREEN commands, failures, results and owned-path hashes/diff in the task ledger, then freeze for independent review.

The implementation's scoped run passes all **24 tests** (`grad`:12, `trainer`:12). New cases establish four-step rate changes `[0.04, 0.00008, 0.02, 0.08]` with reference `0.08`, decoded fixed-basis velocity, checked SGD schedules, partial valid batches, frozen zero biases, atomic multi-layer validation including a legacy custom optimizer, malformed reference-gradient rejection, and public-field/basis guards. The helper is checked at `B=1` and `B=8`; its two branches exactly match independently computed one-constant gradients in decoded values, level, scale and layout. Full-step tests measure exactly `+6` ct-pt, `+6` rescales and `+18` rotations on the two-layer B8 fixture whenever rate differs from reference, with unchanged traces and 14 refreshed ciphertexts per update. Calling the constant-rate setter versus leaving it unused gives identical seeded decoded results, counters, traces and refresh counts. Existing legacy simulator/count tests also pass.

Before implementation, the old rate-mutation path was observed to fail at the second update (`0.08 → 0.02`): weight error `1.2476188803830779e-2`, exceeding the unchanged gradient-test tolerance `2^-14`. During validation, two new trainer tests initially failed an arbitrary weight-change guard (`>10 * 2^-10`) although parity errors were approximately `2^-25`; that guard was corrected to require the actual decrypted weight change to exceed measured CKKS error by 100 times and exceed `1e-7`. The parity tolerance remains `2^-10`. All failures and final commands are retained in the D2a ledger report.

The MNIST consumer also passes `cargo check -p ckks-train-bench --bin mnist`. This is generic API integration/compilation evidence, not scheduled benchmark support or CPU/GPU performance evidence; no dataset training, CUDA execution, server work or matrix campaign was run.

## Boundaries and handoff

D2a owns `optim.rs`, `trainer.rs`, the named Dense helper, focused optimizer/trainer/gradient tests and this document. It does not change `trainer_ni.rs`, block sources/tests, CUDA, benchmark loaders/CLI, dependencies, runner/tables or raw results. D1 data work and D3a block handoff are independent.

D2b must configure each NAG reference maximum before boxing its optimizer, apply the actual per-step mean-gradient rate using `Trainer::set_lr`, initialize frozen biases as zero, and update schedule-specific count predictions and benchmark provenance. It must separately remove only supported CLI restrictions and preserve the exact sample order/tail rules. Library correctness is not evidence that the benchmark integration or any registered campaign has run.
