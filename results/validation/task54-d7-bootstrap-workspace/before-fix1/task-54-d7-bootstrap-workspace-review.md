# Task54 D7 bootstrap workspace independent review

Fixed source BASE: `5b50e841e65a0bad698b9a61621ba2e1fde7571d`.

SPEC: **CHANGES REQUIRED**. The Cheb recurrence substantially answers the requested recursive composition problem, but the proposed general BSGS resource expression omits a simultaneously retained owner in one branch. Do not adopt the current expression as a conservative general resource bound.

QUALITY: **CHANGES REQUIRED**, for the same single finding F1 below. No separate defect was established in the inspected Cheb algebra or pipeline lifetime composition. This review is conditional source/math review, not physical-model acceptance, memory fit, execution authorization, numerical correctness, or scientific evidence.

## F1 — P1: retain row backing in the BSGS collection/reduction branch

**Source:** fixed `linalg.rs:812–825` creates `xs: Vec<Vec<Option<Ciphertext>>>`; `828–870` borrows its rows for block evaluation; `872–885` collects and reduces giant results before the function returns. There is no consumption or drop of `xs` before collection/reduction. Its outer row slots and every inner Option row remain allocated alongside baby ciphertext backing, all collected giant outputs, and the current block/reduction work.

**Affected formula:** report section “Serial BSGS collection and primitive boundary” defines `BabyBuild = ... + Rows(t,b_s)` but then gives

```
W_BSGS <= max(BabyBuild,
              X + max(Collect, Reduce, U_mul_plain(2,l)+C2(l)))
          + V_ResultOptionCt(g_s) + ChecksAndCollectionMetadata
```

`X` is explicitly only baby ciphertext payload. `ChecksAndCollectionMetadata` explicitly lists rotation-step validation, terms/iterator metadata and reduction IntoIter backing, not the retained `xs` rows. The table entries are neither primitive scratch nor result-vector slots. It is unsafe to silently enlarge that unknown term to repair a known omitted owner.

**Algebra:** write `BabyBuild = rho+B`, where `rho=Rows(t,b_s)`, and let `C` denote the post-build collection/reduction term including X. The formula uses `max(rho+B,C)`, but the later phase requires `rho+C`. Whenever a shared additive resource coordinate has `C>rho+B` and `rho>0`, the stated expression returns C and omits rho. These inequalities are a symbolic compositional witness, not numeric physical values or assumptions that actual primitive costs are zero. Independent unknown primitive envelopes provide no inequality proving the build phase dominates the later phase.

**Resource-vector nuance:** a specially chosen disjoint coordinate for row-only slots would preserve that coordinate through componentwise max; hence this finding does not assert that every possible typed-coordinate implementation loses row counts. The report permits general resource coordinates, including shared vector/object counts, and does not require such a partition or prohibit scalar additive lowering before max. Even with a disjoint basis, lowering first generally does not commute with componentwise max. The current expression therefore lacks the stated general conservative guarantee. No claim is made that a measured n16 peak is underestimated by a particular byte count.

**Bounded remedy:** leave the existing saturated `BabyBuild` intact and explicitly add retained rows to the second branch:

```
W_BSGS_fixed <= max(BabyBuild,
                    Rows(t,b_s) + X
                      + max(Collect, Reduce, U_mul_plain(2,l)+C2(l)))
                + V_ResultOptionCt(g_s) + ChecksAndCollectionMetadata
```

Equivalently define a build-local expression without its saturated Rows term and factor Rows outside the outer max. Preserve row metadata on both successful and failed collection/reduction prefixes. Keep the existing distinction between Option slots and separately counted ciphertext backing, and do not count inline ciphertext storage twice during later layout lowering. This is a bounded report/formula correction; no production-source change or execution is needed. Renew report/diff pins and review the corrected formula before reuse. The P1 severity blocks use of the expression as a guaranteed general upper bound; it does not indicate a production-code defect.

## SPEC checks and supported conclusions

- **Cache closure:** `A={1..min(d,b-1)} union powers-of-two<=d` is closed under ensure prerequisites. For non-power i below b, a, i-a and positive 2a-i are smaller and below b; power-of-two dependencies are i/2 and zero. Leaves request indices below b and splits request a power of two. `h(i)=T-ceil(log2 i)` and `l_i=h(i)+1` follow by induction from the smaller dependency levels. The saturated cache is charged across recursion, so earlier completed child powers are retained when later children execute.
- **Components and overlap:** CPU multiplication returns three components, relinearization two, and rescaling lowers the level by one. Deep clone and even equal-level mod-switch allocate independent polynomial backing. The ensure seven-C2/C3/final-C2 envelope conservatively covers its lexical temporaries. Leaf's four-C2 envelope covers old accumulator, aligned power, term and replacement. Internal primitive scratch remains additional, including the relinearization popped polynomial and rescale's cloned input representation.
- **Cancellation and recursive shape:** the max over every actual trimmed degree `h<=m` addresses `rposition` and split cancellation. `split_coeffs` owns q/r of combined length h+1 throughout both calls; the report charges both vectors outside the phase max. The q child has degree at most h-k; the r child at most k-1; both decrease for every split. `C3(t+1)+R(k-1,t)` retains qres while r runs. The J envelope retains qres/relinearized/rescaled/aligned-power and r/product/sum output overlap. No assumption about actual floating coefficient signs or zero patterns is needed for these bounds.
- **Dispatch/EvalMod:** cache retention across final relinearize/rescale is represented; normalization is conditional on affine dispatch. Interpolation owns theta, fx and output simultaneously. The returned coefficient vector and target vector survive Cheb/double-angle calls. Old c survives the fallible double-angle assignment. The r=0 empty maximum must remain omitted.
- **BSGS:** block-local four-C2 bound covers the nonshared rotation route's shadowed pre-rotation product as well as old/new sum. `par::map_range` returns a full Vec before reduction, even in the non-parallel build. A returned Err does not short-circuit later block computation; the report correctly requires result payload replacement by `max(e_k C2,E_err)` and retains all result/IntoIter slots. F1 concerns additional persistent row backing, not this ciphertext/result accumulation argument.
- **Hoist/primitive boundary:** inspected source confirms original digit retention, a rotated digit vector per active serial rotation, output Result collection followed by Result<Vec> conversion, and retained kb/ka into finish_galois. The displayed H expression is conditional on its named primitive/collection/error envelopes. This review does not independently close key-switch/base-conversion/NTT scratch or prove a general hoist error payload bound. If fallible collected outputs are admitted, their dynamically owned errors must be included per retained result (or proved unreachable under exact selected preconditions), not hidden by the inline Result slot.
- **Caller composition:** apply_stage retains term references and its BSGS result until rescale completes. CtS retains old x and its optional conjugate during each stage, then x/xc/re/subtraction/im in its tail. StC retains its old intermediate across the next assignment. Bootstrap retains raised and conservatively retains the shadowed CtS pair; serial join evaluates both closures and retains the first Result while the second EvalMod executes. The switched StC arguments survive the entire final call. Test trace is None in the inspected public bootstrap call.
- **Scope:** the formulas expressly target the compiled non-parallel branch, not Rayon with one worker. Unknown primitive, collection capacity/layout, allocator, unwind, stack, runtime, external-owner integration and first-use cache costs remain unresolved; none is assigned zero by this review. Those limits are appropriate for the derivation brief, not additional review defects.

## Exact reviewed artifacts and integrity work

All local paths below are in `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| task-54-d7-bootstrap-workspace-derivation-report.md | 30309 | `75e847b3086266d0ccf2a2c1af3ac5bd567096e530c32248de2beb9f7e9f7853` |
| task-54-d7-bootstrap-workspace-source-pins.json | 10199 | `9bd058ebff48465845e4000cb9c758b648bbd42a1b6521eb4970ecf2b277e86e` |
| task-54-d7-bootstrap-workspace-review-package.diff | 41189 | `82fd4599bc1f2a4383e9b5d5951ce71c45dc45a9e14901a25c193f41b5f3c71c` |
| task-54-d7-bootstrap-workspace-review-brief.md | 1899 | `d3638417b335fd45e7bba4703899a21585eff3b6cd202bd23d0d6d23b8029995` |
| task-54-d7-bootstrap-workspace-derivation-brief.md | 3110 | `900db37fd4a931f6e6fbddbf11fd170f32b4456f378391d40594a99bb487c591` |
| task-54-brief.md | 1032 | `f935214f80bbedf0033dae1decf0b5a789ec4d589336342493b35342a1af31fb` |

The entire new-file diff was reconstructed in memory and matched byte-for-byte to exactly the report and source-pin JSON. This avoids treating a truncated tool display as complete diff evidence. The six local inputs and thirteen fixed blobs below were preinventoried before the stdlib hash/integrity calculation. Their full SHA256, Git blob identity and lengths matched the report's pin ledger. No author's finite recurrence worksheet was rerun; its reported 888*N and state counts were not independently certified or used as a bound.

## Actually consulted fixed sources

All identities below are Git blob SHA1 at the fixed BASE; the exact SHA256/bytes are bound by the verified source-pin JSON hash above. Ranges describe this review, not the author's inspection. Whole-blob hashing does not mean whole-file semantic review.

| Fixed path | Git blob | Semantic range |
|---|---|---|
| docs/study-resource-allocation.md | `b2349d777b31c17749e9a9c188b2a858dc480d87` | opening/context, kinds12/13; 109–155 explicitly reread |
| docs/study-resource-design.md | `cbe624d0814a0dd641c0f80f6ae52d1afbd43134` | full |
| crates/ckks-train-ckks/src/bootstrap/poly.rs | `efaea9b79f59d8790a50e1c2fbfda27ac4815c69` | 75–100,315–733 |
| crates/ckks-train-ckks/src/bootstrap/evalmod.rs | `377ec0f8e9d628204a469548f707dbfd686c0776` | 35–195 |
| crates/ckks-train-ckks/src/bootstrap/bootstrapper.rs | `f9b59155e64cd52e2c0bca42ee639311f6b7b195` | 350–432 |
| crates/ckks-train-ckks/src/bootstrap/linear.rs | `4d491f3e8638f34061e72bda32af694034f07e0c` | 700–935 |
| crates/ckks-train-ckks/src/linalg.rs | `51ee94b98a0395822f9cba92f118a3beca10faaf` | 741–887 |
| crates/ckks-train-ckks/src/par.rs | `38b28ea8e44e9f954c3a8988af682477cdc16863` | full 1–58 |
| crates/ckks-train-ckks/src/evaluator.rs | `76743a4f57d1b46c5327f2463b410854cb2cb947` | 300–610,790–823 |
| crates/ckks-train-ckks/src/ciphertext.rs | `93d8c5183f198b1f02c8860c4ce39b1b50fd2412` | 1–90 |
| crates/ckks-train-ckks/src/keys.rs | `23670b565d533cf6ea9b752be0f3a12bc4b60010` | 107–170 |
| crates/ckks-train-ckks/src/ops.rs | `3104375dd660fd15eb4cf76da50ff3ba0aa5fdeb` | 420–475 |
| crates/ckks-train-core/src/poly.rs | `9d3abda8c26b01a6447922d3df1adcaa23ce069e` | 15–105 |

Not reopened or certified in this review: the author's `bootstrap/params.rs`, `params.rs`, `keyswitch.rs`, `plaintext.rs`, and `core/baseconv.rs`; candidate `bootstrap/modraise.rs` and core `ntt.rs`, `rns.rs`, `modulus.rs`. Their inclusion in the author's inventory is not independent semantic inspection here. The recipe's default numerical constants were not rederived because F1 and the recurrence checks are symbolic.

Performed only fixed git source/document reads, named local instruction/report/pin/diff reads, preinventoried stdlib byte/hash and exact diff reconstruction, and this canonical review-file write. No current parser/source inspection or mutation, Cargo/tests/build/--list, project imports/helpers, HE/native/SSH/CUDA, datasets/process/credentials, network/install, subagents, commit or push. Some requested slices displayed adjacent test headings; no tests were executed. This review does not complete Task54 or release any existing gate.
