# Interactive weighted microbatch update

Status: implemented, locally validated and independently reviewed.

Task54 D2c2 implements the separately authorized accumulation in [study training design](study-training-design.md), sections 3–5. This document precedes its source changes. It extends only the interactive trainer and does not enable benchmark plans, block execution, or hardware campaigns.

## Mathematics and API

`Trainer::train_step_valid(mlp, batches: &[(&E::Ct, &E::Ct, usize)])` consumes a nonempty ordered list of encrypted inputs, encrypted targets, and valid-lane counts. Every count is in `1..=layout.batch`; checked addition gives `N = sum(n_valid)`. Inputs/targets follow the same network `BatchSpec` as the singleton API. The hidden-returning variant returns owned ciphertexts indexed `[micro][hidden layer]`, excluding input and final output. They retain the original forward levels/scales and come from the weights before this logical update. The input slice is borrowed only during the call. Returned ciphertexts can outlive the call and input slice.

For each layer, with unnormalized per-sample derivatives `dz_i` and previous activations `a_i`, form

```
h = sum_i (current_gradient_scale / N) * dz_i^T a_i
h_ref = sum_i (reference_gradient_scale / N) * dz_i^T a_i
q = sum_i (bias_gradient_scale / N) * sum_valid(dz_i)
```

Each derivative uses its own valid mask, including constant derivatives on partial batches. Frozen bias uses exactly zero in place of the bias coefficient. The fixed-reference NAG representation remains [D2a's](interactive-profile-design.md), including a first current rate below the reference and later decreases/restarts. The reference branch exists only when current and reference rates differ; Dense's existing multi-constant gradient helper shares its products/reduction.

All microbatch forward/backward passes read the same old network. Only after all contributions exist, apply one optimizer step per layer, subtract one accumulated bias gradient, and refresh one set of weights, biases and optimizer state. Two sequential singleton updates would change the second microbatch's forward weights and advance momentum twice. Uniformly averaging micro-means would overweight the four samples in an 8+4 split; the common denominator gives each sample weight `1/12`.

## Levels, scales, lifetime and compatibility

A full Identity derivative takes no level, while a partial Identity derivative needs a mask/rescale. Consequently corresponding gradient contributions may have different levels. Before adding each pair, require exactly equal scales (weight gradients already target the old weight scale; bias gradients target the old bias scale), then mod-switch both to their minimum level and add. No scale retagging or rescaling is used for accumulation. Apply this independently to current, reference and bias contributions. Diagonal shape/stride remain those of the layer's existing gradient helper. This uses public generic `HeOps` only.

Compute and accumulate one microbatch at a time; release its intermediate preactivations, derivatives and final output before the next. Keep only gradient accumulators and requested hidden activations. The no-hidden API does not retain hidden ciphertexts across microbatches. Accumulation memory includes both gradient branches when required. Returned hidden activations are not refreshed by this method.

Legacy batch/valid/hidden/epoch methods remain, using a singleton slice with unchanged coefficient, homomorphic arithmetic order, trace labels and refresh ordering. Multi-microbatch trace labels prefix each forward/backward value with `micro{index}/` (zero-based); final updated weights/biases retain `W{layer}'` and `b{layer}'`. Thus the trace exposes mixed full/partial levels and the common update level without altering singleton receipts.

## Validation and failure boundary

Reject empty lists, invalid counts, count overflow, network/optimizer mismatch and invalid read-only optimizer configuration before forward work. Validate folded coefficients for every layer before evaluation. This does not promise all malformed ciphertexts/keys/levels are discovered before work: existing primitive checks can fail during collection. No network or optimizer state is written until all gradients are collected. Counters record actual completed primitives even on a collection failure. Later arbitrary custom-optimizer or refresh failure retains the existing partial-failure contract; there is no general rollback or safe retry claim. Rate setting remains D2a's separate atomic preflight/commit interface.

Counts are the sum of the actual microbatch passes plus one optimizer/refresh phase. An extra reference branch costs `m` plaintext products, `m` rescales and `m*log2(B)` rotations per microbatch per layer. Later NAG velocity work costs `2m` plaintext products/rescales once per logical update. Addition/mod-switch calls are real evaluator operations; the current `OpCounter` has no fields for them, so they add zero to its listed fields. No simulator or benchmark prediction is enabled by this extension.

Validation is synthetic CPU CKKS only: unequal 8+4 with dirty inactive lanes, independent f64 updates, scheduled NAG and SGD, mixed Identity levels, Square/Square, retained one-sample tail, hidden old-weight receipts, invalid input rejection, exact counters and seeded singleton identity. No secure-profile, GPU, dataset or campaign evidence follows from these tests.

## Local validation receipt

The scoped CPU run passes 33 tests: `study_microbatch` 9, existing `grad` 12 and existing `trainer` 12. Clippy with `-D warnings` passes for the library and the new test target; the legacy MNIST consumer compiles. The numerical bound remains `2^-10`, including all four scheduled NAG updates. The 8+4 fixture differs from both sequential updates and equally weighted micro-means by more than ten times that bound. Full/partial and partial/full/one-valid-tail orders are exercised. Returned hidden values cover two hidden layers and two micros as well as the scheduled two-layer network.

For the synthetic two-layer fixture (`m_total=4`, physical lanes 8), every nonreference-rate micro adds 4 plaintext products, 4 rescales and 12 rotations; two micros add exactly 8/8/24. Later NAG adds exactly 8 plaintext products/rescales once, compared with SGD. SGD's complete `OpCounter` equals the sum of singleton pass counters because its updates, bias subtraction, addition and mod-switching have no fields in the current counter. Refreshed ciphertexts are 10 per NAG update and 6 per SGD update, regardless of micro count. Failed second-micro collection retains completed primitive counts, leaves model/velocity untouched and performs no refresh.

The initial test compilation failed only for the two missing APIs. The additional three-layer hidden fixture initially exhausted its ten-prime synthetic chain; it now uses twelve primes and matching key-switch decomposition. All two-layer tests retain their original ten-prime fixture. An intermediate twelve-prime fixture rejected a mismatched decomposition count; two test-only Clippy warnings were also corrected. The task ledger retains these failures and the final commands. These results do not validate scientific presets, GPU execution, block microsteps, benchmark scheduling or any campaign.
