# Experiment matrix registration (Task54 D0)

This document and [experiment_matrix.json](../crates/ckks-train-bench/configs/experiment_matrix.json) register the remaining experiment design. They do not implement a runner, authorize execution, certify coverage, or complete Task54. The design below is **independently reviewed and root-ratified for D0 registration**. Code preparation proceeds under separate delegated scopes; full encrypted eMLP-2/3 execution still awaits the user's separate decision.

The design uses the renamed-crate source snapshot `09202cf5650aa76b9b30dd79a68dd615c75a91ae`. It was prepared after historical reports, subsets and registered FULL jobs already existed. Its prospective time-to-target rules apply only to a **new matrix HE campaign** after design approval and target-artifact locking. They cannot turn prior results, subsets, FULL-1 or FULL-2 into preregistered time-to-target evidence.

## Authority and expansion

The original research plan (2026-11-04, §4.3.3–4.4) and implementation plan (2026-11-04, Task54) require four datasets, the 2020 model and eMLP-1/2/3, CPU/GPU, interactive/noninteractive, applicable baselines, and seeds **0–9**. The JSON is a declarative registration with source paths/pointers; no current program consumes it yet. Existing source configs and captured exports supply their stated values. Explicit new adaptations are marked as such; this file does not rewrite those sources or establish undocumented upstream defaults.

Each family expands its profile IDs, backends, modes, placements, packing variants and campaign seeds. Primary profiles cover all 16 dataset/model pairs. Noninteractive main and packed-c families retain `when_needed`, `after_update` and `after_activation`; interactive refresh is a different mode. Missing implementation, inputs, memory or authorization never deletes a valid main combination.

| Role | Applicable experiment axes | Meaning and present boundary |
|---|---|---|
| Main | Every dataset/model, CPU and CUDA, I and NI | Batched diagonal method. Interactive rows document development/continuity; the threat-model claim requires NI. |
| (a) CPU2020 | Every dataset with `mlp2020`, CPU I, row **and** diagonal | Matched `n16_train` comparison. Only Iris has an exact 2020 setting; other datasets are new adaptations. NI/GPU 2020-model cases remain in the main families. |
| (b) official ReBoot | eMLP-1 per dataset, external CPU NI | Same physical CPU/thread count. Released MNIST is captured; Iris/BC/Fashion upstream profiles, dataset keys and matched exports are missing inputs, not guessed settings. |
| (c) library ReBoot packing | Every dataset/model, CUDA I and NI | Required alternating row/column implementation with encrypted labels, weights, gradients and optimizer state. Present CPU/client-updated code is insufficient. |
| Plain references | Each concrete profile/seed | Actual FP32 reference and separate f64 parity diagnostic; neither is an encrypted GPU row. |

Baseline (b) encrypted eMLP-2/3 is excluded by the original hardware scope (reported 181/214 GB versus approximately 128 GB RAM). Those figures describe the reported official setting, not new measured lower bounds for every adapted dataset. An official `mlp2020` graph and a GPU official baseline are outside that role's definition. These role exclusions do not excuse missing main or baseline-c cases.

## Concrete data and training profiles

| Dataset ID | Inputs / real classes | Fixed split and preprocessing | Primary B / logical epochs / tail |
|---|---|---|---|
| `iris` | 4 / 3 | Existing fixed 120/30 split; training mean and population-standard-deviation z-score | 20 / 400 / retain tail |
| `breast_cancer` | 30 / 2 | WDBC fixed stratified seed0 split, 455/114; training per-column min-max | 16 / 100 / drop last |
| `mnist_released` | 196 / 10 | First4800 training IDX rows / all10000 test rows unchanged; bicubic 14×14 then resized-training-only min-max; exact captured training orders applied separately | 48 / 10 / drop last |
| `fashion_mnist` | 196 / 10 | Published 60000/10000 IDX split; Task51 bicubic 14×14 and training min-max | 48 / 50 / drop last |

Sources: [Iris defaults and preprocessing](../crates/ckks-train-bench/src/bin/iris.rs), [BC config](../crates/ckks-train-bench/configs/bc_mlp.json), [Fashion config](../crates/ckks-train-bench/configs/fmnist_emlp1.json), [released eMLP-1 values](../results/quadruped/raw/reboot_emlp1_full/config_values.json), and [released block values](../results/quadruped/raw/reboot_emlp23_full/config_values.json). BC's 100 epochs and Fashion's 50 epochs are prior Task51 choices, not stated upstream paper epochs. Fashion preprocessing is the existing MNIST-pipeline assumption. BC's earlier z-score divergence stays historical evidence; it does not authorize fresh test-directed preprocessing changes.

The [released data manifest](../results/quadruped/data/reboot_emlp1_full/data_manifest.json) fixes the first4800 training IDX images with unchanged row identity/order and all10000 test IDX images in order. Resize float32 images first, then derive min/max from those resized4800 training rows alone and apply the same values to test. There is no preprocessing permutation: the captured per-model/per-seed training-order indices are a separate operation. Verify captured array hashes from the source manifest and lock/verify exact order-file hashes before a run. The primary MNIST population is the released **4800 samples × 10 passes**. Its flat 48000-index file already contains those ten passes: legacy `--epochs 1` must not become ten repetitions of that file. The separate [MNIST paper-setting reference](../crates/ckks-train-bench/configs/mnist_emlp1_paper.json) uses 60000 training samples, 50 epochs, `lr_sum=0.001` and weight decay `0.001`; it remains a distinct existing plaintext study, not a replacement for the released encrypted population.

| Model | Graph, output and initialization | Loss / optimizer / rate |
|---|---|---|
| `mlp2020` | Input→10→real classes, Square at both layers, biases, Normal σ=0.2 | Mean over real output classes and samples MSE, delta `2(pred-target)/classes`, SGD; exact Iris mean-MSE lr=0.1 |
| `emlp1` | Input→32→padded classes, `z²+z` then identity, no biases, Xavier bound `sqrt(1/(fan_in+fan_out))` | RSS-style delta `pred-target`, no factor2/output-width divisor; NAG μ=0.9, weight decay0 |
| `emlp2` | Local block64 with its own classifier/loss; separate 64→32→output stack | Same eMLP loss/optimizer; no gradient below a local block; test prediction from the output stack |
| `emlp3` | Local blocks128 and64 with their own classifiers/losses; separate 64→32→output stack | Same independent stage losses and output-stack evaluation; losses are not summed into one backpropagation |

Graph/rate sources are [faithful eMLP-1](../crates/ckks-train-bench/configs/reboot_emlp1_faithful.json), [eMLP-2](../crates/ckks-train-bench/configs/reboot_emlp2.json), [eMLP-3](../crates/ckks-train-bench/configs/reboot_emlp3.json), and the dataset configs above. eMLP output widths are **Iris4, BC2, MNIST16, Fashion16**. All padded channels have zero targets and participate in loss/delta and initialization fan-out; accuracy argmax uses real classes only. The 2020 model has exactly 3/2/10/10 outputs and no added output padding in its loss.

The eMLP batch-sum step coefficient is Iris `0.0005` (explicit new borrowing), BC `0.005`, and MNIST/Fashion `0.0005`. A trainer that averages sample gradients uses `lr_mean = lr_sum × valid logical batch size`, with no extra factor2. BC and Fashion schedules are constant; MNIST uses the exact model/seed `lr_per_step` arrays, including their warmup/constant and cosine-restart regions, rather than approximating the schedule from prose. All primary profiles have weight decay0.

For `mlp2020` outside Iris, the fixed new study adaptation uses the dataset's batch-sum step magnitude times the nominal logical batch as its mean-MSE learning rate, retains `2/classes`, and uses SGD. This is neither official ReBoot SGD nor an exact 2020 reproduction. Non-MNIST deeper eMLPs retain the fixed dataset data/rate/epoch policy and source graph widths. Root ratified these independently reviewed adaptations to preserve valid combinations without pretending undocumented upstream settings are known. Divergence or poor accuracy must be reported; changing them requires a new registered profile, not hidden tuning.

## Batches and slot-oriented comparison

B48 released MNIST profiles and separately named `mnist_released__<model>__b64` profiles each have full seeds0–9. All four model graphs retain main CPU/CUDA I/NI cases; baseline-a adds its matched 2020 CPU variants, packed-c adds GPU I/NI variants, and external eMLP-1 B64 is an explicit override study pending verified upstream support. B64 blocks retain the full-execution user hold. None of these new studies is an alias for FULL-1.

B64 consumes the same per-seed initialized arrays and sample order as its B48 parent, with 75 updates per 4800-sample pass (750 total). Its lr coefficient is sample-aligned to the B48 source: at the first sample of an update, choose the source step `floor(start/48)` within that logical pass. This follows the existing [sample-aligned schedule helper](../crates/ckks-train-bench/src/bin/mnist.rs) rather than taking the first 750 entries of a 1000-step schedule. B64 gets its own batch-matched FP32 run and target.

The research plan also calls for slot-oriented B beside ReBoot B. **B64 is the registered alternate, not proof of maximum slot occupancy.** A separate `mnist_released__emlp1__b128_candidate` registers seeds0–9 (quick0), main CPU/CUDA I/NI, packed-c GPU I/NI, and an external-b verified-override study pending inputs/support. The faithful eMLP-1 config's `/input_note` records that 196 inputs pad to256 and up to128 samples fit its input ciphertext. B128 is therefore an **input-slot-fill candidate**, pending per-stage/layout/level admission; it is neither a measurement nor a new FULL control. B48 and B64 remain unchanged.

For this B128 candidate the captured flat48000 order gives375 updates. The arithmetic4800/128=37.5 means some logical pass boundaries lie inside a complete update; fractional updates are not observations. Preserve flat order and sample-aligned schedule lookup `source[floor(first_flat_sample/48)]`. Checkpoint only at actual update boundaries, recording cumulative completed updates, `processed_sample_count`, KS/bootstrap counters, every declared time component and which logical pass boundary was crossed. The first checkpoint time after a boundary is an upper bound on that boundary time; its measured accuracy is not a bound on accuracy at the unobserved boundary. Do not invent an exact-pass accuracy/time, interpolate, split that update, or drop64 samples per pass. Those alternatives would be separately declared profiles. The full flat order has no128-sample tail, and existing FULL1/2 exact checkpoint boundaries stay unchanged.

Other model/dataset slot-oriented capacities remain explicit planner/profile-readiness work. Derive candidates from actual padded input/hidden/output stage layouts, masks, parameter slots and level admission; retain source sample/order/drop-last/epoch semantics and register a distinct profile before new observations. Do not guess numerical capacity, equate B64 with maximum, or treat a bare `d×B≤N/2` check as a full training proof. Iris20, BC16 and Fashion48 remain source-defined logical batches; unsupported non-power-of-two lanes are an implementation gap, not permission to change them.

For 784-input B48, register the distinct `weighted_partial_v1` extension with physical32+16 samples. Existing NI math already normalizes by `sum(n_valid)`, masks partial lanes and aligns mixed delta levels ([trainer](../crates/ckks-train-nn/src/trainer_ni.rs)). Required work is chiefly admission/planner/validation: prove the weighted mean matches one48-sample update, prove padding contributes zero, and verify exact operation/level counts. Historical R-M3-4 exclusions and formula-only rows remain unchanged. Reduced-level L7/L8 and RQ3 supplements have separate IDs; they cannot replace secure main profiles.

## Seeds, initialization and inputs

Full statistical seeds are 0–9; quick uses0. One encryption-seed repetition does not make a new model/order seed. Within each concrete profile and seed, main, packed-c, matched row/diagonal and FP32 share **exact arrays, order, preprocessing, graph, loss, optimizer and schedules**. Algorithms may use different encrypted layouts without changing the learning problem. Captured weight/order files take precedence over a bare `--seed` flag.

Released eMLP-1 has local weights/orders/schedules for seeds0–9. Released eMLP-2/3 currently have exports for0–2;3–9 are missing inputs, not substitutes from eMLP-1 (whose shuffle differs after different initialization draws). The 784 subset supplement currently has one captured initialization/order;1–9 require matched exports. Official Iris/BC/Fashion external profiles, dataset keys and exports remain unresolved inputs; null fields are intentional.

Exact Iris controls preserve their existing master/fork RNG behavior. New locally initialized profiles use the existing ChaCha20 bench convention with explicit stream domains: keys1, initialization2, shuffle3, encryption4, refresh5. No legacy control is reseeded to fit that new adapter. Fixed data splits stay fixed across study seeds; new noncaptured training orders are seed-specific per epoch. Future results must include exact manifests/hashes, source/build identity and binary-owned provenance. Runner-created annotations cannot supply missing measurement provenance or repair historical JSON.

## Security and encrypted semantics

Main and packed-c interactive cases use `n16_train`; NI cases use `n16_boot`. Baseline-a matched cases use `n16_train`. The exact Iris `seal_2020` row/diagonal supplement preserves historical continuity and reports its different cryptography separately. Reduced-level supplements state their own top level, chain and secret distribution; they do not stand in for primary NI. CUDA with `HostBootstrapper` is the declared M4a path; native CUDA bootstrap is a separate hardware profile/gate. [Security limitations](security.md), especially ReBoot's unequal security configuration, remain visible; registration does not establish the RQ1 comparability condition.

The intended NI server has encrypted data, labels, weights, gradients and optimizer state and no client secret. Interactive refresh transfers and client work are explicit timed boundaries. A client that decrypts gradients and updates plaintext weights does not establish NI; the present legacy packed CPU path remains a bounded historical reference until a genuine packed-c trainer exists. Use public algorithms with generic `HeOps`; no GPL source is copied into this library.

For block after-activation (AA) handoff, reuse **`stage.last_activation_bootstraps()[micro][0]`**, the canonical-Delta post-bootstrap activation. `train_step_valid_with_hidden` returns pre-bootstrap hidden values and keeps that API meaning. Reusing its return and bootstrapping again would duplicate AA work. Mod-switch the accessor value only as required by the registered downstream input level; do not decrypt/repack or add another bootstrap. Non-AA handoff keeps its explicit counted bootstrap. Tests must establish handoff parity/ranges, Delta/level/layout compatibility and closed-form counters. This is a reviewed design direction, not an implemented extension claim.

## Measurements and targets

An **actual f32 computation** with dtype provenance supplies FP32 references; the existing deterministic f64 twin remains a numerical-parity diagnostic. Both use the registered arrays/order/loss/optimizer. No f64 field may be relabeled FP32.

For each concrete full profile, lock the actual seed0 FP32 final artifact, its hash and the numeric target `max(0, FP32_final_accuracy − 0.005)` before examining **any new paired matrix HE campaign results**. Use that common target for seeds0–9 of the profile and a separate target for B48/B64. Record the lock time and design revision. Independent design review and root ratification accepted the half-percentage-point convention as a new checkpoint-based time metric, not an accuracy acceptance gate. Existing historical/subset/FULL results informed this design; they are excluded from its prospective crossing evidence. Never attach this target retroactively to FULL-1/FULL-2 or change their gates.

Evaluate after each complete logical epoch boundary and at the final tail, observing only actual update boundaries; if an update crosses a logical boundary, record the first following observation with actual processed sample count. Only its observation time bounds the earlier boundary time; its accuracy is not a bound on unobserved boundary accuracy. For released MNIST, logical boundaries are each4800 samples inside the flat48000 order; B128 follows the crossing and conservation rules above/below. Time to target is the first declared checkpoint at/above target, an upper bound without interpolation. An unreached target is right-censored at final training time; report reached/expected counts and do not turn it into zero or average it as a successful crossing. Final accuracy is the fixed final checkpoint; best-test-epoch historical figures remain separately labeled optimistic evidence.

D6 must emit binary-owned cumulative counters/times at update0/sample0 and every actual checkpoint: completed updates, processed samples, KS, bootstrap count, each declared time component, crossed logical boundary IDs and accuracy at that processed sample count. Setup has its own explicit pre-training baseline; it is not silently charged twice. D8 must validate **disjoint consecutive interval deltas** by cumulative subtraction. Every complete update and its operations/time belongs to exactly one interval; each additive metric conserves `sum(interval deltas) = final cumulative − origin cumulative`. Do not prorate a crossing update, double-count it, or sum overlapping timer categories as if disjoint.

B128 checkpoint updates are38,75,113,150,188,225,263,300,338,375 at samples4864,9600,14464,19200,24064,28800,33664,38400,43264,48000. Consecutive observed intervals alternate **38/37 updates and4864/4736 samples**, totaling375 updates/48000 samples. Normalize interval KS/bootstrap/time by its actual sample count (or actual update count for a per-update metric), never blindly by4800. Label these as **observed update-boundary intervals**, not ten observed4800-sample epochs. Exact epoch measurements are unavailable where an update crosses the boundary. A whole-campaign total/10 is permitted only as an explicitly **amortized logical-pass average**, not ten epoch observations or a measured per-epoch SD. Accuracy remains attached to actual sample endpoints; only the checkpoint detection/crossing-time interpretation is an upper bound. FULL1/2 exact-boundary semantics stay unchanged.

Record setup, synchronized server updates, host bootstrap, client refresh/transfer, evaluation diagnostics and end-to-end time separately. CPU timing comparisons require the same physical host and thread count; no Mac/server pooling. GPU memory must distinguish estimates, pool-used high-water, pool-reserved high-water and sampled process-VRAM peak, with phase/reset/source boundaries. Checkpoint maxima cannot masquerade as process peak. KS counts, bootstrap counts, levels, precision and scope retain binary provenance; unavailable external counters are missing evidence.

Aggregate only exact matching profile/security/mode/placement/batch/optimizer/loss/order/backend/host/thread/target groups. Show expected10, measured n, missing-input, missing-code, resource-skipped, failed and awaiting-user-decision counts. Use sample SD (`ddof=1`); at n1 SD is unmeasured, not0. Benchmark repetitions are not statistical seeds. Same-seed and same-optimizer matching applies to CPU/GPU and fallback comparisons.

## Quick, resource states and controls

Quick projects **every applicable active family case** to seed0 and the first two logical batches of its registered order. It retains full-profile train-side preprocessing statistics, graph/classes, batch, security preset, mode, placement and baseline role. Two-update checkpoints establish pipeline coverage only. A missing implementation/export remains an explicit missing row; it is not a completed quick pass. Deferred Task53 stretch is separately tracked rather than implied covered.

The ≤7200-second target is unmeasured. Optional same-preset key/context reuse must reset all learning state, show shared versus cold setup accounting, and prevent cross-case leakage; release own caches before external phases. If the target is missed, report it and review startup/reuse optimizations. Do not drop valid cases or choose insecure presets to claim a pass.

The four RQ3 caveats from [M4 §2.3](m4-report.md) remain required in resulting tables:

1. `bsgs=false` keeps pre-rotated weights and the same Galois keys; it does not implement the strongest textbook diagonal layout. Only baby rotations can be hoisted and only the zero-shift block summed lazily.
2. Consequently its measured BSGS gain is an upper bound against the strongest textbook baseline, while marginal hoist/lazy gains inside `bsgs=false` are lower bounds.
3. Hoist changes no `OpCounter` value; its contribution appears in measured time, not KS counts.
4. BSGS does not change gradient outer-product rotations, and lazy relinearization does not change the interactive gradient's individual diagonal products; hoist may change their execution.

Capacity preflight records need/free and exit3 as an **unmeasured resource skip**; device OOM is failure. A code/input gap is not a memory skip. Preserve sequential GPU operations and epoch `trim_pool(0)`. RQ3 keeps all eight flags and all four caveats; any exit3-triggered NAG→SGD row needs its matching CPU SGD profile, level, batch and layout. Unknown GPU availability is not evidence that an experiment is memory-infeasible.

| Existing control | Fixed identity | Treatment |
|---|---|---|
| FULL-1 | MNIST196 eMLP1 CPU NI WN B64, seed0 exported initialization, `n16_boot`, 48000 samples/750 updates, B48-reference sample-aligned schedule | Preserve its original raw path, values and gates. Not ten-seed coverage. |
| FULL-2 | Same released population, CPU NI WN B48 in64 lanes, seed0, 48000 samples/1000 updates | Preserve registered `abs(acc − captured ReBoot seed0 twin) ≤0.005`; do not substitute a ten-seed mean or new FP32 target. |

The original research plan and adopted [M3 report rulings](m3-report.md) govern these controls. [RQ1 finalization](rq1-finalization.md) is a pending Task42 verification procedure/proposal and does not supersede that authority. They are archival/current registered jobs: **quick never launches or restarts them**. Any equivalent quick profile gets a distinct quick case/output ID and two-update subset, never a FULL control ID, raw path or FULL validator. Current FULL runs are not restarted by adding new study profiles.

Fixed gates remain RQ1 KS ratio≤0.70 with its existing precision/security/wall-clock conditions; bootstrap≥16 bits, latency≤220ms, and the HEonGPU 2× operation comparison goal. Historical failures, misses and unmeasured hardware evidence remain so. This design changes none of those thresholds.

## Rulings, alternatives and next gates

| Root direction, independently reviewed and ratified | Reason | Alternative and cost if wrong |
|---|---|---|
| Released4800×10 primary;60000 paper-setting reference separate | Exact exports and FULL continuity | Replacing it mixes distinct lr/decay/epoch populations and breaks pairing. |
| Exact Iris crypto supplement plus matched-N baseline-a | Preserve RQ4 and isolate packing at main N | A mixed-crypto ratio confounds cryptography and algorithm. |
| Fixed new cross-dataset adaptations, including Iris output4/lr_sum0.0005 | Keep all valid combinations without guessing official settings | Dropping cases shrinks scope; a poor adaptation may fail, which must be reported rather than tuned secretly. |
| AA accessor reuse; preserve existing partial-batch math | Avoid duplicate boot and retain established normalization | Wrong Delta/layout or weighting invalidates parity/costs; require focused proofs before use. |
| Locked FP32 final−0.005 target for new campaigns only | Fixed target before new HE observations; true FP32 reference | The target may be insensitive or unreachable; preserve censored rows instead of tuning it or reinterpreting old FULL jobs. |
| Separate ten-seed B64 matched studies | One FULL seed0 control cannot supply batch-study statistics | Adds real input/FP32/admission/execution cost; maximum-slot status still needs proof. |
| Separate ten-seed B128 input-slot-fill candidate | Spec §4.3.3; the existing config supports input fit, while B48/B64 alone leave a slot-oriented gap | More storage/time/runs and possible resource skips; stage admission and other graph/dataset candidates remain unproved. |

The dependency order is self-contained here; no private ledger is required to interpret it:

1. **D1 — Shared profiles/data:** class-aware loaders, identical train-only preprocessing/splits, output padding, source manifests, initialization/order and fail-before-allocation validation.
2. **D2 — Matched training and baseline-a:** faithful I/NI loss/optimizer/rate/schedule semantics, non-power-of-two/partial lanes, planner admission and a genuine CPU2020 row variant beside diagonal.
3. **D3 — Block bridges:** interactive/CUDA block entrypoints, independent local losses and separate output stack, AA accessor reuse, counts, memory preflight and authorized subsets.
4. **D4 — GPU packed-c:** generic packed trainer with encrypted targets/weights/velocity, explicit I versus NI refresh, correct graph/padding/levels, CPU parity and CUDA hardware validation.
5. **D5 — External adapter/inputs:** verified upstream dataset/profile/seed/override support, same-CPU/thread provenance, value-only missing seed exports and isolated output archives; preserve GPL separation and explicit block exclusion.
6. **D6 — Binary metrics:** actual FP32 references, target locks, actual-update cumulative counters/times and endpoints, B128 crossing intervals with exact conservation, synchronized timing and defined GPU high-water scope.
7. **D7 — Runner:** expand all applicable profiles/seeds with unique IDs, preserve source/build/argv/thread receipts, states and authorization gates, serial quick/full/external orchestration and safe memory fallback matching.
8. **D8 — Aggregation:** exact-group ten-seed mean/sample-SD and coverage counts, matched comparisons, actual-interval denominators/conservation and amortized-versus-observed labels, censoring, malformed-input failure without partial tables; preserve historical table/F1/F2 API and FULL gates.
9. **D9 — Physical campaign:** root first recovers current FULL evidence, then dispatches reviewed CUDA checks and full quick coverage with measured runtime/disk usage, then authorized ten-seed experiments; full encrypted blocks remain held.

Each package requires focused correctness/provenance tests and independent review before dispatch. This registration creates none of that code and provides no physical validation.

D0 independent scientific review and root ratification are complete. Remaining decisions/evidence are per-stage B128 admission and general slot-oriented profile readiness, verified external inputs/overrides, actual FP32 target locks, and root scheduling after current FULL. Full encrypted blocks still await user decision even when subset/code checks pass. Task53 degree4 ReLU approximation on [-4,4] is a **deferred stretch (scope-cut #3)** with coefficients/profile/range validation and an independent future gate; it is neither implemented nor required to complete this D0 registration. No build, training, measurement or dispatch is authorized by this document.
