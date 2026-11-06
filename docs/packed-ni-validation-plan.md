# Task54 D4c2: small real-bootstrap fixture proposal, revision3

Status: **ROOT-RATIFIED AND INDEPENDENTLY REVIEWED DESIGN; implementation and HE execution separately gated.** Written 2026-11-06 before any new HE execution. This is a prospective measured small-fixture correctness test, not a universal precision guarantee. No source edits, builds, keys, HE, downloads, SSH, git operations or subagents were used. Work comprised source reading and ordinary scalar/rational plaintext calculations. The original proposal is preserved byte-for-byte in `task-54-d4c2-fixture-proposal-v1.md`; revision2 is preserved byte-for-byte in `task-54-d4c2-fixture-proposal-v2.md`, both in the implementation ledger `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.

## 1. Revision decisions and evidence boundary

Revision3 corrects a source-domain error found in independent review, before any HE: `stage.rs::PackedStage::new` requires Square output width to equal real-class count. Square therefore uses output3/real_classes3 and loss factor2/3; Poly retains output3/real_classes2. No production-domain extension is proposed. All corrected Square values below are recomputed from that loss. The original root visibility requirement is >4E; revision2's optional >8E condition is removed because corrected Square W1 is6.3945E. Arrays, rates and precision gates are unchanged.

Root rejected v1's proposed outward-rounded all-slot interval certificate and arbitrary64 amplification ceilings: they would introduce a new proof project beyond the existing c2 design. Root also rejected treating an unsupported2^-25 incremental-bootstrap target as the practical prerequisite. Revision2 removes those prerequisites and the separate rho gate. It strengthens the analytic learning signal before any HE observations and prospectively registers `epsilon_boot=2^-20`, `E=2^-12`, independent same-context bootstrap calibration, actual all-slot per-event diagnostics, and direct end-to-end parity. None of these proposed precision gates has been demonstrated for this fixture.

Relevant source evidence:

- `docs/packed-refresh-design.md` §§3–7 requires real CPU bootstrap, all-slot range contracts, actual-prime histories, old/pre/post hidden values, W/U or W/bias refresh, and four-rate NAG. Its structural examples are not precision measurements.
- `nn/tests/trainer_ni.rs::{mini,fx}` provides the full-chain construction pattern: N=2^10, `n16_boot_bits()`, scale42, dnum8, `BootParams { log_slots:9, ..default_n16() }`, output16, actual CPU Bootstrapper. Its `TOL_MINI=2^-20` is an asserted same-context legacy diagonal-trainer criterion, **not an established per-bootstrap bound or packed measurement**. The new incremental2^-20 criterion is explicitly unvalidated. Legacy real-only error assertions do not establish the new all-complex-slot criterion either.
- `docs/bootstrap.md` §11.4 reports26.77/26.18 bits for the separate CKKS miniature. `ckks-train-ckks/src/bootstrap/bootstrapper.rs::mini` has only two compute levels, P `[60,60]`, dnum9 and output2. Those measurements cannot certify this full-chain output16 context.
- `nn/tests/{packed_training,packed_optimizer}.rs` exercise packing and the unscaled-V reference without real bootstrap. Their custom-chain numerical assertions are not inherited NI evidence.
- `packed_training/{plan,ops,stage,optimizer}.rs` defines the actual slot maps, reduction trees, masks, old-state losses and fixed-reference U recurrence. Use those structural paths without copying the production arithmetic into the independent scalar oracle.

The original tiny fixture had Square W1 final movement1.5833e-5, below2^-12. A loose test could conceal all of that learning. Revision2 instead doubles W and Square biases, triples valid inputs, and quadruples valid targets, keeping rates, counts, dimensions, masks, modes and reference fixed. Its Square real-class annotation was erroneous and is corrected in revision3. The original exploration used the mistaken Square loss factor1 and reported W1 movement0.0023608 for tripled X; that number is superseded. The corrected loss factor2/3 gives W1 movement0.00156116594 (6.3945E), while hidden magnitudes across both modes remain below0.337. Larger weights/biases and targets were also inspected with scalar arithmetic, but were unnecessary. These choices precede every HE/calibration observation; no failed encrypted result informed them.

## 2. Fixed parameters, layout and original arrays

Use N=2^10, the unchanged `n16_boot_bits()` Q/P arrays, scale42, dnum8, log_slots9, default n16 BootParams otherwise, real CPU Bootstrapper with observed output16. Delta=2^42, slots512. Initial W and Square bias ciphertexts, and fresh inputs and targets at every update, are encoded at **level16**, exact Delta. U is absent initially. Do not borrow top31 transform primes, change the chain, or relabel a natural scale. Restored values must have observed level16 and exact Delta.

Use two geometry plans: Square is `PackedGeometryPlan::new(3, &[StageShape::new(3,3,3), StageShape::new(3,3,3)], 512)`; Poly is `PackedGeometryPlan::new(3, &[StageShape::new(3,3,2), StageShape::new(3,3,2)], 512)`. Both plans have rows128, columns4. Exercise ordinal0 and ordinal1 as independent single-stage cases initialized from the same arrays. Ordinal0 is RE backbone/CE head; ordinal1 is CE backbone/RE head. This gives both directions without implementing c3 handoff. The output width is3. Square has three real classes and MSE coefficient `2/3`; its channel2 is a real class whose targets in this fixture are all0. Poly has two real classes and a third trained model channel with target0. Channel2 is structurally active in both modes and must never be masked away.

All matrices are logical `[output,input]`:

```text
W1 = [[ 1/4,  -1/8,   1/16],
      [-1/16,  3/16,  1/8 ],
      [ 1/8,   1/16, -3/16]]
W2 = [[ 3/16, -1/8,   1/16],
      [ 1/16,  1/4,  -1/8 ],
      [-1/8,   1/16,  3/16]]
Square b1 = [1/4, 3/16, -1/8]
Square b2 = [1/4, -3/16, 1/8]
Poly/Identity: no bias objects or dummy bias arithmetic.

record0 X = [ 3/4, -3/8,  3/8 ]; Y = [1/2, 0, 0]
record1 X = [-3/8,  3/4,  3/16]; Y = [0, 1/2, 0]
record2 X = [ 3/8,  3/16,-3/4 ]; Y = [1/4, 1/4, 0]
record3 X = [ 1/2, -1/2,  1/2 ]; Y = [7/8, -7/8, 7/8]
```

All four records stay in this order, including finite nontrivial invalid sentinel3. Valid masks by update are `[T,T,T,F]`, `[T,F,F,F]`, `[T,F,T,F]`, `[T,T,T,F]`, giving n=`[3,1,2,3]` and snapshot valid IDs `[0,1,2]`, `[0]`, `[0,2]`, `[0,1,2]`. Invalid records produce no arithmetic, denominator contribution, snapshot or bootstrap. The valid record2 at update3 must retain ID2; filtered enumeration must not relabel it as1.

Rates are fixed Rust f64 literals `[0.04,0.01,0.02,0.08]`: below-reference first, decrease, restart/increase below reference, equality. NAG has mu0.9 and immutable r0.08; plaintext V starts0, `U=r*(1+mu)*V`, following actual f64 expression order when checking Rust values. Square uses the same rates with active two-bias SGD. No seed/rate/range selection after encrypted observations.

For slot `s=4*i+j`:

- RE stores W[o,i] at `4*i+o`; CE stores W[o,j] at `4*o+j`. Matrix support is `i<3 && j<3` in either orientation; the values differ.
- Expanded vectors store a[i] for i<3, replicated across all four columns. Repeated vectors store a[j] for j<3, replicated across all128 rows. Other slots are0.
- W/U use matrix support; bias/hidden/output use full vector support. Channel2 is active; padding outside those supports is checked separately.

The independent diagnostic packer implements these literal formulas, not production pack/mask helpers. Expected sorted training key steps are `[-2,-1,1,2,4,8,16,32,64,128,256]`; compare against the generated lattice union before HE. Bootstrap builds its own required keys.

Primary held-out key seed: byte153 repeated32 times. Record encryption seed byte=`160+8*update+2*record+role`, zero-based update/record, role0=input/1=target. Initial-role RNG seeds are `[200;32]` for W1, `[201;32]` for W2, `[202;32]` for b1, `[203;32]` for b2; Poly performs no bias encryption. Reset each case's deterministic initial RNGs. Fixed reuse across cases supplies reproducibility, not independent-sample evidence. Freeze the key-generation call order described in §6 and source identity with the implementation snapshot.

## 3. Coverage, graph semantics and independent oracle

Run two modes × two ordinals × AU/WN/AA =12 cases; each commits four updates. Primary numeric cases use `GradRelin::Lazy`. PerSample belongs to separate explicit counter/trace coverage and is not claimed as encrypted numerical coverage from these Lazy cases. Each case has at most16 state bootstraps; an AA case adds9 hidden bootstraps. The full12-case suite has at most192 state plus36 AA hidden calls, total228, a bound rather than a predicted exact count. No secure-size case is part of this proposal.

AU refreshes four roles every update in the existing ordered contract: W1,b1,W2,b2 for Square; W1,U1,W2,U2 for NAG. Role ordinals are the actual layer ordinals of the chosen stage, not always0/1. WN/AA use the real carried future DAG with known next rate, valid count, input16 and target16; terminal requests a canonical refreshed checkpoint. Freeze the structurally admitted schedule before numerical execution, without copying legacy diagonal schedules or forcing an extra refresh to hide drift. AA has hidden-event counts `[3,1,2,3]`. A single-stage WN/AU case has no boundary-hidden bootstrap. Unexpected events fail counts.

Use independent scalar loops:

```text
z1 = W1_old*x + b1_old
h = z1*z1                     [Square]
h = z1*z1+z1                  [Poly]
z2 = W2_old*h + b2_old
y = z2*z2                     [Square]
y = z2                        [Identity]
dz2 = (y-target)*(2/3)*(2*z2)  [Square; 2/real_classes=2/3]
dz2 = y-target                [Identity]
upstream = transpose(W2_old)*dz2
dz1 = upstream*2*z1_old        [Square]
dz1 = upstream*(2*z1_old+1)    [Poly]
gk = mean_valid(dzk outer input_k); gbk = mean_valid(dzk)
Wk -= lr*gk; bk -= lr*gbk      [Square/SGD]
Vk = mu*Vk+gk; Wk -= lr*(gk+mu*Vk); Uk=r*(1+mu)*Vk [NAG]
```

Compare every update's old state, candidate before refresh, refreshed committed state, hidden.pre, AA hidden.post, and old-state output against the untouched scalar twin, using all512 expected slots including replicas and padding. Both hidden snapshots correspond to old weights; compare post additionally against the actual decrypted pre to isolate bootstrap error. `receipt.hidden` must select post for AA. Establish structurally that second affine and its outer product consume that exact post object and first derivative uses old z1. Approximate-identity numerical closeness alone cannot prove object reuse.

All decryptors and plaintext references stay outside the NI session. No decrypt/re-encrypt mock, callback, plaintext state repair or secret scaling is allowed. No dummy U for Square or dummy biases for Poly.

## 4. Plaintext learning visibility and range evidence

Ordinary scalar evaluation of the fixed schedule gives:

| mode | final W1 movement | final W2 movement | final b1 movement | final b2 movement | final max U1 | final max U2 |
|---|---:|---:|---:|---:|---:|---:|
| Square | 0.00156116594257 | 0.00279636965903 | 0.00212190561644 | 0.0123295127444 | absent | absent |
| Poly | 0.0244288490083 | 0.0345990416005 | absent | absent | 0.0275826478171 | 0.0394056143031 |

All W/b role movements exceed4E with E=2^-12; smallest Square W1 is6.3945E, about1.60 times the4E requirement. Thus the proposed gate remains informative for each role's final learning signal. The corrected W1 does not meet revision2's optional8E condition; that stronger condition is explicitly superseded before HE, as directed by root. It does not assert that each coordinate or the tiny decrease-rate update is individually observable above noise.

| mode/update | max W1 | max W2 | max b1 | max b2 | max U1 | max U2 | max old hidden |
|---|---:|---:|---:|---:|---:|---:|---:|
| Square1 | .250348 | .249809 | .250470 | .252792 | — | — | .257874 |
| Square2 | .250588 | .249810 | .250790 | .254434 | — | — | .258717 |
| Square3 | .250848 | .249805 | .251161 | .256719 | — | — | .259318 |
| Square4 | .251562 | .249419 | .252122 | .262330 | — | — | .259970 |
| Poly1 | .252545 | .251974 | — | — | .006613 | .006932 | .324280 |
| Poly2 | .254465 | .252187 | — | — | .017775 | .027573 | .327269 |
| Poly3 | .258523 | .252149 | — | — | .024650 | .035958 | .330412 |
| Poly4 | .274429 | .256896 | — | — | .027583 | .039406 | .336143 |

Entries are rounded upward to six decimals. Max output across the schedule is below0.094 for Square and0.077 for Poly. A separate Python standard-library `Fraction` evaluation with exact rational decimal rates `[1/25,1/100,1/50,2/25]`, mu9/10 and r2/25 checked every update's ideal W/b<2/5, U<1/10, hidden<1/2, and W/b role movement>4/4096. The corrected Square evaluation includes the exact rational2/3 loss factor. This verifies a finite ideal plaintext schedule, not f64 bit patterns or CKKS ranges. The scalar f64 results agree to the displayed digits. These calculations require no keys or encrypted feedback and do not introduce a formal interval proof prerequisite.

Registered **actual pre-bootstrap** maxima, after projection and mod-switch immediately before each CPU bootstrap:

| role | fixture bound | universal contract |
|---|---:|---:|
| W1, W2 | <=0.4 | <=1 |
| U1, U2 | <=0.1 | <=1 |
| Square b1, b2 | <=0.4 | <=1 |
| Square or Poly AA hidden | <=0.5 | <=1 |

The bounds have substantial margin over the finite plaintext trajectory and E. They remain empirical obligations: externally measure every actual bootstrap input across **all512 complex slots**, including padding/error and each replica. Record per-event `(mode,ordinal,placement,update,record/role,max magnitude,worst slot)`, pre/post projection error for state, and before/after bootstrap error. Violation fails even if final parity passes. Inputs can have magnitude0.75 because they are not themselves bootstrapped in these single-stage fixtures; hidden/state inputs to bootstrap have the tighter table bounds. No clipping or parameter/rate adjustment is permitted.

## 5. Fixed prospective numerical gates and amplification risk

1. `epsilon_boot=2^-20=9.5367431640625e-7`: maximum incremental complex all-slot error of each bootstrap output against the actual level0 input. Observed output metadata must independently match level16/Delta/two components/context/spec. This is an **unvalidated prospective criterion** motivated by the existing same-context asserted threshold, not a transferred measurement.
2. `E=2^-12=0.000244140625`: maximum complex all-slot error for every W/U/bias/hidden/output snapshot against the untouched plaintext twin, on **every update**, before and after refresh where applicable. Check active error, padding error, replica spread and imaginary leakage separately as diagnostics, and use the full maximum for acceptance. No percentile, mean, padding omission or final-only averaging.
3. Analytic final W/b movement and final NAG U magnitude must exceed4E for each respective role. The table already supports this prospectively. Actual states must agree with the nontrivial reference; a “moved” boolean alone is insufficient. If reporting unscaled V, its corresponding error bound is `E/[r*(1+mu)]`, not E in different units.
4. Exact counters/all OpCounter fields, actual-prime scales/levels, valid IDs, projection reserves and per-role range bounds remain independent hard gates.

E is256 times epsilon_boot, but that ratio is **not an amplification theorem**. RE can sum128 rows; CE sums4 columns then broadcasts; bootstrap introduces padding/replica errors, which can enter later products. NAG carries perturbed U across updates and changes its coefficient with the fixed rate schedule. Square and Poly derivatives can amplify input discrepancies. Multiple bootstrap events and CKKS arithmetic can therefore fail E even if each bootstrap meets epsilon_boot. Event counts alone do not justify adding independent errors in quadrature or assuming a factor25 bound.

The practical validation is deliberately direct: isolated same-context calibration, per-event actual-input/output checks, and independent full-run all-slot parity at every snapshot. E was selected before HE with clear learning visibility; its feasibility remains to be tested. There is no requirement for a new formal interval certificate, a synthetic perturbation proof, guessed64 constants or a separate rho primitive project. A failure remains evidence that this fixed fixture/implementation does not meet the registered test, with no universal correctness or production-precision claim from a pass.

## 6. Bounded independent calibration, not yet authorized or executed

Under a separately authorized implementation task, implement a separate small CPU test, provisionally `nn/tests/packed_ni_calibration.rs`, which constructs the unchanged full-chain16 Bootstrapper without instantiating a packed NI trainer. Use key seeds `[151;32]` and `[152;32]`; retain seed153 for the main integration cases. Number key cases j=0,1 and listed vectors k=0..20. Fresh encryption uses `Rng::from_seed([100+21*j+k;32])`, giving bytes100..141, instantiated once per vector/key. The second bootstrap has no new encryption/RNG. Register construction order: create params and KeyGenerator, create Encoder, construct Bootstrapper, generate the sorted lattice Galois-key set for main integration only (none for standalone calibration), obtain evaluator relin key, obtain public key, then obtain the diagnostic secret-key clone. Freeze this call order and source identity in the implementation snapshot; a necessary API deviation must be recorded before execution rather than tried after a failure.

The following21 fixed512-slot vectors cover the final fixture's magnitudes and supports; all unmentioned slots are0. For envelope matrices, logical value `[o,i]` is `amplitude*(-1)^(o+2*i)` on3×3, deliberately not symmetric under transpose. Vector envelopes use `[amplitude,-amplitude,amplitude]`:

1. all0;
2–3. constant +0.5 and -0.5;
4. `0.5*(-1)^s`;
5. `0.5*(2*s/511 - 1)`;
6–7. only slot0=+0.5, only slot511=-0.5;
8–9. W-envelope matrix amplitude0.4, RE then CE;
10–11. U-envelope matrix amplitude0.1, RE then CE;
12–13. bias-envelope vector amplitude0.4, expanded then repeated;
14–15. hidden-envelope vector amplitude0.5, expanded then repeated;
16–17. actual Square old hidden for record0 at initialization, expanded then repeated;
18–19. actual Poly old hidden for record0 at initialization, expanded then repeated;
20. actual initial W1 in RE;
21. actual initial W2 in CE.

For each vector/key: freshly encode/encrypt at Delta/level0, bootstrap once, lower that exact output to0 without projection or re-encryption, bootstrap again. Define these externally decoded complex512-slot vectors: m is the intended real vector; x0 is the fresh level0 input to bootstrap1; y1 is its level16 output; x1 is that exact output lowered to0 immediately before bootstrap2; y2 is its level16 output. Let `norm(v)=max_s |v[s]|`, `e0=norm(x0-m)`, `e1=norm(y1-x0)`, `e2=norm(y2-x1)`, `a1=norm(x1-m)`. Require:

- all values finite;
- `e0<=epsilon_boot`, `e1<=epsilon_boot`, `e2<=epsilon_boot`;
- `a1<=e0+epsilon_boot` and `norm(y2-m)<=e0+2*epsilon_boot`;
- `norm(x0)<=0.5+e0` with hard ceiling `0.5+epsilon_boot`;
- `norm(x1)<=0.5+a1` with hard ceiling `0.5+2*epsilon_boot`;
- both actual input norms<=1, exact output metadata and complete counters.

Record `norm(x1-y1)` separately to expose lowering/decode drift. The a1 gate charges the first bootstrap and its lowering transition to the cumulative allowance, without silently adding a new error budget. Thus second-input range includes fresh, first-bootstrap and lowering effects, not only fresh error; a failed a1/final/range criterion is failure even if isolated e1/e2 pass. Print vector/key/pass/worst-slot/real-error/imaginary-error/complex-error/range and all e0/e1/e2/a1/drift values. These calibration allowances do not replace training's tighter fixed role-specific bounds.

A fully successful calibration executes exactly21×2×2=84 bootstrap calls. At most84 are permitted. Stop at the first failed range, error, metadata or count gate, or any runtime/construction error; preserve completed observations and report actual attempted/completed bootstrap counts and missing coverage. Do not start the main integration, retry, substitute a seed or describe an interrupted run as an84-call pass.

One bounded execution after implementation and authorization:

```text
cargo test -p ckks-train-nn --test packed_ni_calibration -- --nocapture --test-threads=1
```

No `--ignored`, release benchmark, secure-size/GPU/remote work or repeated seed search. Preserve exact source/parameter identities, seeds, stdout/stderr and failures. Finite calibration is empirical evidence for these vectors only, not proof for all subsequent ciphertexts. The main trainer still checks every actual event against epsilon_boot and each snapshot against E.

Before integration HE, perform the already-required structural/Encoder admission for actual registered level16 inputs/targets and reachable histories, using actual q and source f64 scale arithmetic. Freeze the predicted event schedule. Report any unrepresentable mask/factor, insufficient reserve, observed output mismatch, or invalid future history with its exact role/level/prime/scale/error. Do not silently change input/target levels or use the evaluator-bound planner as proof of pre-key admission. Missing shared P1 support remains the existing design boundary, not a new fixture theorem requirement.

## 7. Failure handling and root decision

Calibration failure: preserve the actual maximum error, slot and metadata; report **calibration criterion not met** and stop before the main encrypted integration. Main integration failure: preserve the failing case/update/role, actual ranges/errors/counts/trace and prior successful observations; report the exact gate. A success elsewhere cannot average away a failure. Do not retune seeds, rates, arrays, biases, reference, bounds, E, epsilon_boot or Bootstrapper; no hidden U reset or fresh client encryption. Any new hypothesis requires a separately versioned, root-ratified proposal retaining this negative result.

Root ratified this exact fixture/gates/84-bootstrap calibration cap after independent review. Source changes and HE execution remain separately assigned bounded tasks; design ratification alone authorizes neither. This revision does not broaden c2 into c3/D4d.

Current disposition: **literal fixture fixed; finite ideal plaintext ranges and learning visibility checked; all-slot CKKS precision unmeasured; same-context calibration unrun; exact-scale schedule not yet validated; implementation untouched by this task.**

Publication provenance: reviewed revision3 SHA256 `0fbd3238092a174a60aa8f645ce258de46ce83c61499ce8517653ba0fddbcc62`; independent scoped review `task-54-d4c2-fixture-rereview.md` SHA256 `87cf3d30661c49a16152286a23db0bf2737fa729ecc3f141e10e58fbfae5de79`, retained in the implementation ledger named above. Only status and destination-path wording changed for this public copy.

## 8. D4c2 calibration program implementation freeze (2026-11-06)

`crates/ckks-train-nn/tests/packed_ni_calibration.rs` implements only the standalone
calibration in §6. **Primary calibration UNRUN; no encrypted precision result.**
It uses the unchanged full n16_boot Q/P chain at N1024, scale42, dnum8,
log_slots9/default n16 BootParams and output16. The independent literal packer covers
all21 registered vectors, including corrected Square3 and Poly2 old-initialization
hidden values; it never invokes production pack/mask helpers. Five individually
selected non-HE tests check the corpus/maps, rational old-hidden values, geometry key
lattice, seed schedule, complex metric/range negatives, metadata and all counter fields.

Construction order is params, KeyGenerator (seed151 then152), Encoder, Bootstrapper
(internal bootstrap Galois/conjugation/SSE keys), **no training Galois keys**, evaluator
relin key, public key, diagnostic secret clone. Each vector uses one fresh encryption
RNG byte100+21*j+k and exactly one level0 encryption. The second bootstrap consumes the
first returned ciphertext lowered directly to0. Seed153 is held out. No training,
projection, new encryption between boots, scale relabel, repair or retry is present.

The program externally decrypts all512 complex slots and checks §6's exact e0/e1/e2,
a1, final and input-range gates, and separately prints lowering drift. Each output's
two components, shared Q-basis context, NTT form, exact scale bits and level16 are
checked; configured Bootstrapper context/spec/output are checked independently.
Every call's complete observed OpCounter delta is compared with cost_per_boot() and
must contain exactly one bootstrap. Inclusive measured key switches are not charged
again. Direct ciphertext checks are not a P1 verified metadata token or a new Linux
support claim. The P1b fix1 report's historical accidental n16 EvalMod execution and
unsupported Linux metadata-proof boundary remain unchanged.

The real test is serial and stops on the first error or panic, printing the current
key/vector/pass/stage, attempted calls, returned calls, fully validated calls and
missing coverage. At most84 calls are allowed; success requires all84 returned and
validated. Program output prints actual parameter values and Q/P identity, configured
cost, per-event complex diagnostics and available program identity. Source/executable
SHA256, compiler/platform/build metadata and stdout/stderr are retained separately by
the runner, not synthesized as program observations. No research-result JSON is made.

Construction validation used CARGO_INCREMENTAL=0, CARGO_BUILD_JOBS=8 and
RAYON_NUM_THREADS=8: target `cargo test --no-run`, five `non_he_*` tests each selected
with `--exact`, target-only strict Clippy and owned-file rustfmt. All passed. The exact
primary command below is **UNRUN**, reserved for root's independent-review release:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --test packed_ni_calibration packed_ni_calibration_primary_84_calls -- --exact --nocapture --test-threads=1
```

Do not execute the whole integration target during program construction. This exact
command supersedes §6's provisional whole-target command for selecting the primary
attempt. The implementation ledger records the pre-HE API/consumption registration,
complete patch, hashes, test executable identity and construction logs in
`task-54-d4c2-calibration-{api,report}.md` and adjacent freeze artifacts under
`.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.
Independent review and root release precede the single primary local attempt. A pass
would describe this finite corpus/context only; main integration event precision,
exact-scale schedule, runtime/memory, GPU and production security remain separate.
