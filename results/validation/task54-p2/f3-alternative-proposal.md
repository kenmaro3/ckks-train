# F3 pure alternative proposal — source-derived, not executed

The672/2688 authorized scan is exhausted and remains UNMET. This is a read-only proposal after the six-file source/Cargo freeze. No CkksParams, BootstrapMetadata, proof constructor, prime search, tests or HE were executed in preparing it. Existing profiles/main HE fixtures and prior evidence remain unchanged.

## Why another similar ULP-miss scan is poorly motivated

The source has a useful exact-label closure mechanism, not just high precision. CkksParams::scale is a power of two. Compact W and U updates target that Delta; weight unpack targets its source's same Delta. For finite normal values, multiplying the F64 q by a power-of-two target is an exact exponent shift, and dividing that product by q returns the representable power-of-two target. Those target-label searches have an explicit preimage; coarse q alone does not destroy it.

AA next_input_scales returns Delta for every layer, so bias targets are Delta², again powers of two. WN's head-bias target is the recomputed first hidden scale times Delta. For Square/PolyRelu2 that first hidden scale was itself produced as squared_z/q_j; after multiplication by Delta, the shifted squared_z is a concrete label preimage at that same q_j (assuming finite normal shifts). If that producing q_j lies in the extraction retry interval, the downward retry has a reachable endpoint even if other levels miss. Identity's corresponding preimage comes from its preceding dense rescale. Thus a new coarse search intended only to find an exhausted ULP miss can still be structurally defeated by an available preimage. This is a source-derived mechanism, not a proved global closure theorem for every legal schedule; no assertion that all registered profiles are admitted follows. It explains why simply lengthening the previous domain is weak.

A more directly justified counterexample can exercise the exact helper's OTHER existing executable gate: MIN_PLAIN_SCALE. Equal source/target labels require a plaintext scale equal to the actual q. A compute prime below2^20 cannot pass that gate even though its level is structurally usable. This is a valid actual-prime parameter recipe and reachable immutable state, not an invalid raw x/y or mutated W/U substitution.

## Recommended tiny candidate set and fixed ceiling

Propose exactly ONE PURE-ONLY candidate, no adaptive alternatives:

- CkksParams::new(log_n8, q_bits=[31,20×16,34×4,58×7,56×4], p_bits=[60×4], scale_bits19, dnum8), dense main key. Exact q length32/max_level31. This is not a replacement for any released preset or main HE fixture.
- BootParams { log_slots7, ..BootParams::default_n16() }, hence CTOS4/EvalMod7/STOC4, output level16/start15, K13, degree127, double_angle0, ephemeral_hw32, sparse_hwNone. Use the unchanged current DftSupportProof::verify_current and BootstrapMetadata::new_with_verified_support; no fake descriptor.
- Layout(slots128,d16,B8), dimensions2→4→4, Identity/Identity (both are already supported enum/runtime activations), Dedicated transpose, OPTIMIZED EvalConfig; SGD lr0.08, active biases.
- Fresh state only from initial_stage_metadata. Exactly one full-valid8 micro, source x/y exactly that state's BatchMetadata, logical total8, rate0.08, next_fullSome(true). No manual metadata change and no prior trajectory.
- Before planning, require every actual compute q[1..=16] to be at least3*Delta/2 and strictly below2^20. The upper bound follows from20-bit construction; the lower bound is a declared finite admissibility gate for this candidate, not a parameter mutation or prime-selection search. Record actual ordered primes. If prime generation, lower-bound gate, descriptor/proof or initialization fails, report that failure and STOP with F3 unmet; no replacement candidate, ring, bits, scale or extra update.
- Ceiling: one CkksParams constructor, one proof check, one BootstrapMetadata constructor, one geometry/initial-state construction, and ONE plan_stage_update call. No prime skip/sweep, no ciphertext/Encoder/HeOps/bootstrap execution. Constructor internal deterministic prime generation is part of that one candidate, not a second discovery family.

Using N2^8 leaves more20-bit NTT candidates for the required16 compute primes than the old N2^12 two-prime helper domain, and fits physical128slots exactly at d16/B8. Existence and the actual lower-bound gate have NOT been computed here; this proposal makes no false constructor-PASS claim.

## Constructor constraints derived from current source

CkksParams::new validates log_n in1..=MAX_LOG_N, nonempty q/P, all bit sizes2..60, scale bits1..60, positive dnum dividing32, then deterministic distinct NTT primes1mod2N and RNS bases. Prime availability remains a genuine checked construction outcome. Delta is2^19 and every q is positive finite in F64.

BootstrapMetadata preflight requires full packing log_slots+1=log_n, log_slots1..16, positive CTOS/STOC bands no larger than7, positive EvalMod/K/degree, depth<=band, positive output31-(4+7+4)=16, declared weights consistent, ephemeral32 below dense expected main128, and K>=its existing minimum (K13 with ephemeral32 as in the default). DftSupportProof requires the existing supported target/compiler/ordinary FP premises and streamed scalar checks, not an asserted token. Default resource limits stay unchanged.

The DFT scale path is powers of two followed by its exact final target. CTOS starts at q0≈2^31 and reaches nominal EvalMod scale2^58 across4 stages; its upper-band q56 supplies ample plaintext scale. STOC starts near2^(58+19-31)=2^46 and reaches2^19 over4 stages; its q34 stages supply plaintext scales around2^27..2^28, above MIN2^20. All intended exponents are finite and far from subnormal/overflow. Folded factors/default EvalMod/bands are unchanged. These bounds motivate expecting descriptor construction; the exact constructor/proof result must still be recorded, and encoding/range/precision/security/resource admission remains explicitly unvalidated by P1 metadata.

## Predicted reachable policy→exact failure, with explicit qualification

Identity/Identity on a full micro avoids nonlinear ordinary scalar work and partial derivative masks. With q_compute/Delta in[1.5,2), forward scales shrink. Using real source operation order, rough bounds place each weight-gradient product no higher than(2/3)^4*Delta and no lower thanDelta/16; the exact-gradient plaintext scale is then at least(1.5/(2/3)^4)*Delta >2^20 and at most32Delta. Bias-gradient plaintext scales similarly exceed2^20. Folded0.08/8 coefficients stay far below the2^62 integer bound. These are reasons that a real initial forward/backward update can reach the compact policy; no success is asserted before execution.

The expected structural first-update compact levels are10 and11; unpacked W levels9 and10, and a known-full next step fits. Reconstruct the complete LevelState from emitted container_updated/velocity events and call the unchanged must_boot helper, asserting false. No state-bootstrap event may appear.

The first weight_unpacked attempt then has source Delta at a20-bit compute q. Its exact target is Delta, whose preimage Delta*q exists. The derived plaintext scale is q<2^20, so the existing target_plain_scale_at/check_plain_scale contract rejects it as InvalidParams. Preserve exact actual operation, prime, source/target bits, source level, full ordered attempts and predecision container trace. No new bootstrap or relabeling repairs it.

**Disposition required before execution:** the previous discovery release demanded HekitError::ScaleMismatch (an exhausted exact-ULP miss). This alternative deliberately predicts the same exact-target helper's existing minimum-plaintext-scale rejection, InvalidParams, after a valid structural no-boot decision. It is not an exhausted-ULP-miss witness and must not be renamed as one. Root must explicitly decide whether this source-valid actual-prime policy→exact-label-admissibility counterexample satisfies the original API-map requirement, or retain the ULP-miss obligation separately. If root requires ScaleMismatch specifically, this proposal is not authority to weaken that requirement or run another search. Either way no P2-complete claim is made.

If authorized and witnessed, freeze the exact literal single-case regression and discard discovery machinery. If it fails earlier or the policy differs, preserve that result and leave F3 unmet; no automatic tuning. Strengthened main runtime and independent rereview remain separate gates regardless of this pure-only outcome.
