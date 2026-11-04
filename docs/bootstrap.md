# Bootstrap level design for n16_boot (M3, Task 35)

This document fixes four things:
- the modulus chain of `CkksParams::n16_boot()`;
- the nominal per-level scale `scale_at`;
- the level budget of mod-raise, CoeffToSlot, EvalMod and SlotToCoeff;
- the EvalMod constants of `BootParams::default_n16()`.

It also checks that the design reaches the 16-bit precision target. Tasks 36–39 build on it.

Every number below comes from the reproducibility script `tools/bootstrap/evalmod_design.py`. It uses only the standard library and numpy and runs in under a second:
`python3 tools/bootstrap/evalmod_design.py`.

> **Decision (controller rulings R-M3-5, R-M3-7 and R-M3-8).**
> - **Key.** The main key is dense. Sparse-secret encapsulation (SSE; Bossuat, Troncoso-Pastoriza and Hubaux, ACNS 2022) switches to an ephemeral key with h' = 32 for the mod-raise.
> - **q0/Δ.** q_0 = 54 bits, so q0/Δ = 2^12. The bootstrap input stays at scale 2^42.
> - **P.** 4×60 bits.
> - **Bands (R-M3-7, R-M3-8).** The large primes go to CoeffToSlot, because CtS works on values of size about K·q_0. The chain is [54, 42×16, 34×4 (StC), 58×7 (EvalMod), 56×4 (CtS)], dnum 8. That gives log PQ = 1732 and levels_after_boot = 16.
> - **EvalMod (R-M3-8).** K = 13, Chebyshev degree 127, r = 0 (no double angle), in 7 levels. This needs the level-optimal Chebyshev evaluation (§4), which is a requirement on Task 37.
> - **Predicted precision: 17.4 bits (§6).**
> - **Fallback if Task 38 measures below 16 bits:** q_0 = 52. That gives 19.4 bits predicted and log PQ 1730.
> - **Fallback if Task 37 cannot fit EvalMod in 7 levels:** the e8dbe3e chain (EvalMod 8, compute 15, log PQ 1748) with the same r = 0 and degree 127. A textbook evaluator fits degree 127 in 7 + 1 = 8 levels, and the predicted precision stays at 17.4 bits (§5).

The plan's first numbers did not survive the calculation; §5–§7 keep the analysis.
- **q0/Δ.** At the plan's q0/Δ = 2^16, no design with a dense key reaches 16 bits.
- **K.** The plan's K values were wrong. A dense key needs K ≈ 400–420, not 25. h = 192 needs K = 31–33, not 12. K = 12–13 is the value for h = 32.
- **Bands.** R-M3-5's chain still had 42-bit CtS primes. Rounding the CtS diagonals then limits the bootstrap to about 9 bits (review C1, confirmed in §6).
- **Double angle.** The first choice of r = 1 wasted a level. At K = 13, degree 127 with r = 0 is already at the float64 floor (R-M3-8, §4).

## 1. Modulus chain and scales

Level i means a ciphertext over primes q_0..q_i. Bands from the bottom:

| levels | primes | bits | band | nominal `scale_at(level)` |
|---|---|---|---|---|
| 0 | q_0 | 54 | base | 2^42 (preset scale) |
| 1–16 | q_1..q_16 | 42 | compute | 2^42 |
| 17–20 | q_17..q_20 | 34 | SlotToCoeff (StC) | 2^34 |
| 21–27 | q_21..q_27 | 58 | EvalMod | 2^58 |
| 28–31 | q_28..q_31 | 56 | CoeffToSlot (CtS) | 2^56 |
| P | p_0..p_3 | 60 | special | — |

**Prime sizes and log PQ.** `q_bits = [54, 42×16, 34×4, 58×7, 56×4]` and `p_bits = [60×4]`. That is 32 q primes with dnum 8, so 4 primes per digit and max_level 31.

log PQ = 54 + 16·42 + 4·34 + 7·58 + 4·56 + 4·60 = 1732. The plan ceiling is 1750. The exact value is 1732.0 minus about 1e-9.

**Digits.** In bits, the digits are:
- 180 (q_0..q_3)
- 168, 168, 168
- 144 (q_16 + 3 StC)
- 208 (1 StC + 3 EvalMod)
- 232 (q_24..q_27, EvalMod)
- 224 (CtS)

The largest digit, 232, is 8 bits below log P = 240 (`n16_boot_digits_below_p_exactly`). At level 0, compute values must satisfy |m|·2^42 < q_0/2, that is |m| < 2^11.

**What `scale_at` means.** `scale_at(level)` is the nominal 2^(bits of q_level) for level ≥ 1, and the preset scale (2^42) at level 0. It is the scale that a ciphertext-by-ciphertext product keeps when rescaled by q_level. So it is the working scale of the compute band (2^42) and of the EvalMod band (2^58).

**`scale_at` is not an invariant of CtS or StC.** There the ciphertext is multiplied only by plaintext diagonals and carries its own scale:
- CtS: 2^54 → 2^58, against a nominal 2^56;
- StC: 2^46 → 2^42, against a nominal 2^34.

A diagonal encoded at q_l keeps the ciphertext scale, and one encoded at q_l·σ multiplies it by σ. Exact scales also differ from these powers of two by the ratio prime/2^bits. For the single-size presets (seal_2020, n16_train), `scale_at` equals `scale()` at every level.

`levels_after_boot(params, bp) = max_level − (ctos + evalmod + stoc) = 31 − 15 = 16`. After a bootstrap the ciphertext is at level 16, with 16 compute levels plus the base.

## 2. Pipeline: levels and scales

The input is a ciphertext at level 0 with scale Δ = 2^42; Task 38 checks both conditions. Its plaintext coefficients are Δ·c_j, where c = encode(m). For m uniform in [-1, 1] in all 2^15 slots, the coefficient rms is 1/sqrt(3N) ≈ 2^-8.8.

| step | levels | primes consumed | scale in / out | content after the step |
|---|---|---|---|---|
| SSE s → s' | 0 | — | Δ / Δ | the same message under the ephemeral key s' (h' = 32) |
| ModRaise | 0 → 31 | — | Δ | t_j = Δ c_j + q_0 I_j (I_j integer, small because h' = 32, §3); read as x = t/q_0 at scale q_0 ≈ 2^54 |
| SSE s' → s | 31 | — | 2^54 / 2^54 | back under the dense main key (standard key modulo Q·P) |
| CoeffToSlot | 31 → 27 | q_31..q_28 (56-bit) | 2^54 / 2^58 | two ciphertexts (the real halves of the N coefficients), slots y_j = x_j / K in [-1, 1]* |
| EvalMod | 27 → 20 | q_27..q_21 (58-bit) | 2^58 / 2^58 | sin(2π x_j) = cos(2π(x_j − 1/4))** |
| SlotToCoeff | 20 → 16 | q_20..q_17 (34-bit) | 2^46 / 2^42 | coefficients q_0 sin(2π x_j)/(2π) ≈ Δ c_j: m at scale Δ, level 16 |

\* The 1/K factor and the conjugate-split 1/2 are folded into the CtS diagonals, so they cost no level. The diagonal scales multiply to q_28···q_31·2^58/q_0 (§6).

\** A Chebyshev polynomial of degree d (⌈log2(d+1)⌉ levels with the level-optimal evaluation), followed by r double angles (r levels). The default is degree 127 and r = 0: 7 levels.

Notes:
- **EvalMod scale.** At level 27 the ciphertext enters EvalMod at scale 2^58, which matches `scale_at(27)`. The products are rescaled by 58-bit primes.
- **StC scale.** The EvalMod output is v ≈ 2πΔc/q_0 at scale 2^58. That is the same plaintext as 2πc at scale Δ·2^58/q_0 ≈ 2^46, so the relabelling is free. The 1/(2π) factor and the step from 2^46 to Δ are folded into the StC diagonals. The last diagonal is encoded at q_l·Δ/(current scale), so StC ends exactly on the preset compute scale (Task 38's `ScaleMismatch` contract). Each StC diagonal scale is about 2^32 to 2^34, above the 2^20 plaintext floor.
- **No arcsine correction.** The uncorrected sine-versus-mod error at q0/Δ = 2^12 allows 36.1 bits for |m| ≤ 1 and 27.1 bits for |m| ≤ 8 (§6).

## 3. The overflow I and the interval K

**The overflow I.** After the mod-raise, I_j = round(b_j/q_0 + Σ_{s_k ≠ 0} ±a_k/q_0). Here b and a are uniform mod q_0 for any ciphertext that went through computation. So I_j is the rounded sum of h+1 iid U(−1/2, 1/2) terms. Its standard deviation is σ_I = sqrt((h+1)/12), where h is the Hamming weight of the key the mod-raise runs under (`BootParams::mod_raise_hw`).

**The interval K.** The EvalMod polynomial is accurate on x ∈ [−K, K] (`k_intervals = K`). This covers |I_j| ≤ K − 1 with room for the tiny message part. A bootstrap fails if any coefficient has |I_j| ≥ K, so the failure probability is a union bound over all N = 2^16 coefficients.

**Target: at most 2^-30 failures per bootstrap.** A training run does at most about 2^14 bootstraps.

| mod-raise key | h | σ_I | log2 P_fail per bootstrap |
|---|---|---|---|
| dense (hekit default: −1/0/1 with probability 1/4, 1/2, 1/4) | ≈ N/2 = 32768 | 52.26 | K=25: +15.4 (certain failure); K=400: −29.4; K=420: −33.8 |
| sparse | 192 | 4.01 | K=12: +8.1; K=25: −14.4; K=30: −27.4; K=32: −33.3; K=33: −36.4 |
| **SSE ephemeral (chosen)** | 32 | 1.66 | K=12: −29.1; **K=13: −39.7**; K=14: −53.4 |

Notes on the table:
- The dense rows use a Gaussian tail. The h = 192 and h = 32 rows use the exact Irwin–Hall tail, against which the Gaussian is slightly pessimistic.
- A positive value means the expected number of failing coefficients is above 1.
- At K = 25, a dense key overflows in 64% of the coefficients.

**The guard in `validate`.** `BootParams::validate` enforces K ≥ 7σ_I + 1/2 (`min_k_intervals`). That bound is 13 for h' = 32, 29 for h = 192 and 367 for the dense key. A 7σ tail is about 2^-22.5 per bootstrap, so the guard is necessary but not sufficient; the table above sets K.

σ_I comes from the key's real weight. That weight is `ephemeral_hw` if set; otherwise it is `CkksParams::hamming_weight()`, which `validate` requires to equal `sparse_hw`. Without that check, the guard would accept the SSE constants with no ephemeral key, or an h = 192 interval on a dense key.

## 4. EvalMod: Chebyshev approximation plus double angles

We approximate f_r(x) = cos(2π(x − 1/4)/2^r) on x ∈ [−K, K] by Chebyshev interpolation of degree d. The variable y = x/K ∈ [−1, 1] is folded into CtS. Then r double-angle steps c ← 2c² − 1 give cos(2π(x − 1/4)) = sin(2πx).

**Depth, and a requirement on Task 37.** The level count is

  evalmod_depth = ⌈log2(cheb_degree + 1)⌉ + double_angle ≤ evalmod_levels.

`evalmod_depth()` computes this and `BootParams::validate` enforces it. The ⌈log2(d+1)⌉ part holds only for the level-optimal, scale-invariant baby-step giant-step evaluation of Bossuat et al., Eurocrypt 2021 (Alg. 2, as in Lattigo). That evaluation has two properties:
- the giant steps are powers of two (T_8 … T_64 for degree 127);
- the baby-step scalar products are not rescaled on their own. Their scales are chosen so that one rescale after the giant-step product lands on the target scale.

**This evaluation is a requirement on Task 37.** Textbook Paterson–Stockmeyer, or a BSGS that rescales each c_i·T_i, costs ⌈log2(d+1)⌉ + 1. For the default (degree 127, r = 0) that is 8 levels, one more than the 7-level band.

The other parts of EvalMod cost no level beyond these:
- The 1/K, the 1/(2π) and the −1/4 shift are folded into the matrices or into the interpolated function.
- There is no arcsine step.
- Each double angle, when used (not in the default), is one level: a square, then ×2 (an integer multiply) and −1 (an addition).

**Fallback for a textbook evaluator.** Go back to the e8dbe3e chain, which has 8 EvalMod levels (§5), and keep r = 0 and degree 127: 7 + 1 = 8 levels, sine error 2^-45.4, predicted 17.4 bits (the same). Moving to more double angles instead does not work: degree 63 with r = 1 gives 5.7 bits, and degree 31 with r = 2 gives below 0.

**Error.** The Chebyshev coefficients of cos(ω y + φ) are Bessel values J_n(ω) with ω = 2πK/2^r. They fall off only once n > ω. Each double angle halves ω but costs a level. It also amplifies an error δ by 4|cos θ|, since 2(c+δ)² − 1 = cos 2θ + 4δ cos θ + 2δ². For the angles that occur, the amplification is at most `amp` = max_I Π_i 4|cos(2^i θ_I)|.

The table gives the measured maximum sine error at the points x = I + δ that occur (|I| ≤ K − 1, |δ| ≤ 2^-20). Each column uses all 8 EvalMod levels: d = 2^(8−r) − 1. Values near 2^-45 are the float64 floor of the measurement.

| K | r=0, d=255 | r=1, d=127 | r=2, d=63 | r=3, d=31 |
|---|---|---|---|---|
| 12 | 2^-45.5 | 2^-44.2 (amp 2.8) | 2^-43.5 (amp 10.5) | 2^-40.7 (amp 41) |
| **13** | 2^-45.4 | **2^-43.8** | 2^-43.7 | 2^-37.2 |
| 16 | 2^-46.0 | 2^-43.6 | 2^-43.8 | 2^-28.1 |
| 25 | 2^-44.6 | 2^-43.0 | 2^-25.5 | 2^-10.0 |
| 30 | 2^-44.5 | 2^-28.8 | 2^-12.8 | 2^-3.6 |
| 32 | 2^-42.7 | 2^-20.9 | 2^-8.7 | 2^-1.4 |
| 34 | 2^-29.5 | 2^-14.1 | 2^-5.3 | 2^0.6 |

**What 8 levels can carry:**
- K ≤ 32 at about 2^-42 or better, but only with r = 0 (degree 255).
- K = 13 needs only degree 127 even at r = 0 (2^-45.4).
- A mod-raise under the dense key itself (K = 400 or 420, r = 0, degree 2^L − 1) still has an error of about 1 at L = 9, 10 and 11. It reaches 2^-39.8 only at L = 12.

Raising r does not lower the depth. ⌈log2(d+1)⌉ + r stays near log2(2πK·1.2) for any r, and each double angle adds amplification.

**Chosen constants (ruling R-M3-8)** (`N16_EPHEMERAL_HW` = 32, `N16_K_INTERVALS`, `N16_CHEB_DEGREE`, `N16_DOUBLE_ANGLE`): **K = 13, degree 127, r = 0.** The depth is ⌈log2 128⌉ + 0 = 7 = `evalmod_levels`. These constants assume SSE with h' = 32 and the level-optimal evaluation. The sine error is 2^-45.4, which is the float64 floor, and there is no amplification. The failure rate is 2^-39.7.

The earlier choice was r = 1 (7 + 1 = 8 levels, sine error 2^-43.8). That spent a level for nothing, because ω = 2π·13 ≈ 82 is already below degree 127. The freed EvalMod prime became a 16th compute prime (§5).

**Alternative: sparse h = 192 main key without SSE.** K = 32 (failure 2^-33.3), degree 255, r = 0, depth 8, sine error 2^-42.7. This needs the 8-level fallback chain. K = 33 does not fit 8 levels at that accuracy.

## 5. Chain choice

**Why not more EvalMod levels.** A dense key without SSE needs K ≈ 420, and so 12 EvalMod levels. Even then the input-noise term alone limits precision to 8.4 bits at q0/Δ = 2^16, and to 16.4 at 2^8. Precision binds, not depth, so the rulings use SSE and a smaller q0/Δ instead.

**Bit sizes per band (R-M3-7).** Rounding the CtS diagonals limits the bootstrap, as §6 shows, so the CtS primes need about 52 bits or more. StC sees only m·Δ, so 34–36-bit primes suffice there.

**The chain must meet all of these:**
- q_0 = 54;
- CtS primes of at least 52 bits, 56 preferred;
- StC primes of about 36 bits;
- P at least 8 bits above every digit;
- dnum divides the prime count;
- log PQ ≤ 1750;
- levels_after_boot ≥ 12, preferring 15–16;
- predicted precision ≥ 16 bits.

**Candidates** (design: dense + SSE h' = 32, K = 13, r = 0, d = 127; key limbs = dnum·(#q + #p), which sets key size and key-switch cost). Chain G has 7 EvalMod levels; all the others have 8.

| chain | primes / dnum (α) | max digit / log P | log PQ | levels after | key limbs | predicted bits |
|---|---|---|---|---|---|---|
| R-M3-5: 16 compute, StC 3×42, CtS 4×42 (superseded) | 32 / 8 (4) | 232 / 240 | 1724 | 16 | 288 | **8.8** (CtS term 8.8) |
| A: 16, StC 3×36, CtS 4×52 | 32 / 8 (4) | 232 / 240 | 1746 | 16 | 288 | 17.3 (CtS term 18.8) |
| B (e8dbe3e; fallback): 15, StC 4×34, CtS 4×56 | 32 / 8 (4) | 232 / 240 | 1748 | 15 | 288 | 17.4 |
| **G (chosen, R-M3-8): 16, StC 4×34, EvalMod 7×58, CtS 4×56** | 32 / 8 (4) | 232 / 240 | **1732** | **16** | 288 | **17.4** (CtS 22.8, StC 23.9) |
| C: 14, StC 3×36, CtS 4×56, P 4×46 | 30 / 10 (3) | 174 / 184 | 1622 | 14 | 340 | 17.4 |
| D: 14, StC 3×36, CtS 4×56, P 5×60 | 30 / 6 (5) | 290 / 300 | 1738 | 14 | 210 | 17.4 |
| E: 16, StC 3×36, CtS 4×56, P 3×42 | 32 / 16 (2) | 116 / 126 | 1648 | 16 | 560 | 17.4 |
| F: 16, StC 3×36, CtS 4×56, P 4×60 | 32 / 8 (4) | 232 / 240 | 1762 ✗ | 16 | 288 | 17.4 |

**Why G (ruling R-M3-8).**
- With r = 0, degree 127 needs only 7 EvalMod levels (§4). The freed 58-bit prime becomes a 16th 42-bit compute prime.
- Compared with B, that saves 16 bits of log PQ (1748 → 1732) and gains one level after bootstrap (15 → 16).
- G keeps B's 56-bit CtS and 4×34 StC primes, and R-M3-5's digit structure (dnum 8, α = 4, P = 4×60), so key size and key-switch cost are unchanged.
- Only one digit is all 58-bit now (q_24..q_27, 232 bits). It stays 8 bits below log P.

**How B came about** (ruling R-M3-7, while EvalMod still had 8 levels). B was the only chain with the preferred CtS size (56 bits), at least 15 levels after bootstrap, dnum 8 and log PQ ≤ 1750. The 4×34 StC primes keep log PQ under 1750, and the fourth StC level makes each StC factor sparser.

**Why not the others.**
- The reviewer's example (15 compute, StC 3×36) has 31 primes, which no useful dnum divides.
- A keeps 16 levels, but its CtS primes are at the 52-bit minimum, with a CtS term of only 18.8 bits (±2).
- F, the obvious 56-bit chain with 16 levels and 8 EvalMod levels, exceeds 1750.
- E keeps 16 levels at log PQ 1648, but doubles the key size (dnum 16). Galois keys are then about 590 MB each, against about 300 MB.

**Fallback if Task 37 cannot fit EvalMod in 7 levels:** revert to B (the e8dbe3e chain: EvalMod 8, compute 15, log PQ 1748) and keep r = 0, degree 127. A textbook evaluator fits that in 7 + 1 = 8 levels, and the prediction stays at 17.4 bits.

**Security trade-off.** G at log PQ 1732 is 18 bits below the ceiling (§8). If Task 39 finds G below 128 bits, the security fallback is C: log PQ 1622, 14 levels after bootstrap, dnum 10 and keys 18% larger.

## 6. Precision argument (target: ≥ 16 bits, max over all 2^15 slots, inputs in [−1, 1])

**From slot errors to bits.** Let ε_j be the EvalMod-side error of slot j, in x units. It becomes the error in coefficient j of the output, which is q_0·x_j. Decoding maps iid coefficient errors to slot errors with real-part rms sqrt(N/2)·q_0·σ_ε, and the maximum over 2^15 slots is about 4.2 times the rms. Hence

  precision ≈ −log2( 4.2 · sqrt(N/2) · (q_0/Δ) · σ_ε )   = −21.6 − log2 σ_ε   at q_0/Δ = 2^12.

The factor 4.2 counts real-part errors. Counting the imaginary part as well would cost at most 0.5 bit (review M7).

The model has five error terms. The ones in x units go through the formula above; the StC term is directly in output slots.

1. **Input noise (dominant).** CtS hands EvalMod y = x/K at scale 2^58 with the absolute slot-domain noise of its levels. For the dense key, one rescale is sqrt(N/2)·sqrt((h+1)/12) = 2^13.2 and one ModDown is 2^14.2. The ModDown value is measured: the n16_train `encrypt` row of `docs/precision.md` has max error 25.72 bits, so rms 2^16.28/4.2. The model counts one rescale plus one key switch per CtS level (×sqrt(4), review M1), plus the digit term below. In x units this is multiplied by K, so σ_in ≈ K·2·2^14.5/2^58. SSE helps precision as well as depth because it keeps K small.
2. **Internal noise.** Each of the 7 EvalMod levels adds about one rescale plus one key switch, in sine units. With r = 0 there is no double-angle amplification. Relinearization noise is added at scale 2^116 and divided by q_l, so it does not count. Divide by 2π for x units.
3. **Approximation error.** The §4 table divided by 2π. It is counted as if it were the rms, which is pessimistic.
4. **CtS plaintext rounding (review C1).** Each CtS level multiplies by diagonals encoded at a scale D_l. Rounding their N coefficients leaves a slot error of rms sqrt(N/12)/D_l = 2^6.2/D_l. For entries of size a_l, that is a relative error of 2^6.2/(a_l·D_l). Relabelling the scale cannot change Π a_l·D_l: the integer magnitudes fix it at

     Π a_l D_l = Π q_l · (abs_out/abs_in) / sqrt(N/2).

   Here abs_in = sqrt(N)·σ_I·q_0 (the slots of the raised plaintext) and abs_out = sqrt(2)·(σ_I/K)·2^58 (packed y). The division by sqrt(N/2) is there because every output sums N/2 inputs. Split evenly over the levels, the error relative to x is about σ_I·sqrt(L)·2^6.2/(a·D). This term limits precision as follows:

   | CtS primes (×4) | q0/Δ = 2^12 | q0/Δ = 2^10 |
   |---|---|---|
   | 42 | 8.8 bits | 11.3 bits |
   | 52 | 18.8 bits | 21.3 bits |
   | 56 | **22.8 bits** | 25.3 bits |

   I derived this independently and it reproduces the reviewer's table. It is uncertain by about ±2 bits.
5. **StC plaintext rounding.** The same bound with abs_in ≈ 2^58·sqrt(2)·2π·2^-12/sqrt(3N) and abs_out = 2^42/sqrt(3). The StC error appears directly in the output slots and is not multiplied by q0/Δ. With 4×34 it allows 23.9 bits; with 3×36 it allows 25.5.

**Terms that do not bind.**
- **Sine versus modular reduction, uncorrected.** On the message part x = Δc/q_0, x − sin(2πx)/(2π) ≈ (2π)²x³/6. This term alone allows 44.1 bits at q0/Δ = 2^16, 36.1 at 2^12, 32.1 at 2^10 and 28.1 at 2^8. At 2^12 with |m| ≤ 8 it allows 27.1 bits.
- **Key-error digit term.** A key switch adds Σ_d [c]_{Q_d}·e_d/P, with coefficient std (Q_d/P)·3.2·sqrt(N/12) ≈ 236·(Q_d/P) per digit. The two all-58-bit digits have Q_d = 2^232. With P = 4×58 the term would be about 2^15.9 slot rms, and would cost 1.6 bits at the worst case (it would predict 15.8). With P = 4×60, Q_d/P = 2^-8 and the term is negligible.

**Chosen design** (chain G; dense + SSE h' = 32; K = 13, r = 0, d = 127; 7 EvalMod levels at scale 2^58; q0/Δ = 2^12; P = 4×60):

| variant | predicted bits | bits each term alone allows (input / internal / approx / CtS pt / StC pt) |
|---|---|---|
| **chosen (chain G)** | **17.4** | 17.4 / 23.3 / 26.5 / 22.8 / 23.9 |
| P = 4×58 (digit term, worst case) | 15.8 | — |
| fallback q_0 = 52 (q0/Δ = 2^10, log PQ 1730) | 19.4 | 19.4 / 25.3 / 28.5 / 25.3 / 23.4 |
| fallback chain B (8 EvalMod levels, textbook evaluator, log PQ 1748) | 17.4 | 17.4 / 23.2 / 26.5 / 22.8 / 23.9 |

**Alternatives, predicted bits.** These are computed on the 8-level chain B, because some rows need 8 EvalMod levels; chain G has the same rounding terms.

| design | q0/Δ = 2^16 | 2^12 | 2^10 | 2^8 |
|---|---|---|---|---|
| dense + SSE h'=32, K=13, r=0, d=255 | 13.4 | 17.4 | 19.4 | 21.3 |
| **dense + SSE h'=32, K=13, r=0, d=127 (chosen; 7 levels)** | 13.4 | **17.4** | 19.4 | 21.3 |
| dense + SSE h'=32, K=13, r=1, d=127 (R-M3-5/7) | 13.4 | 17.4 | 19.4 | 21.3 |
| dense + SSE h'=32, K=13, r=2, d=63 | 13.3 | 17.3 | 19.3 | 21.3 |
| sparse h=192, K=32, r=0, d=255 | 15.4 | 19.7 | 21.6 | 22.7 |
| dense, no SSE, K=420 (12 levels) | ≤ 8.4 | — | — | ≤ 16.4 |

**Margin and uncertainty.** The model is good to about ±1–2 bits. Its noise constants are n16_train measurements, and the CtS noise and rounding terms depend on how Task 37 factors and orders the CtS matrices. The 17.4-bit prediction leaves 1.4 bits above the gate, and the input-noise term binds. If Task 38 measures below 16 bits, q_0 = 52 adds 2 bits.

## 7. Options considered

1. **Chosen:** dense main key + SSE (h' = 32) + q0/Δ = 2^12 (q_0 = 54) + P = 4×60 + chain G (EvalMod degree 127, r = 0, 7 levels). SSE works in three steps:
   - at level 0, key-switch s → s';
   - mod-raise under s';
   - at the top level, switch s' → s with a standard key modulo Q·P.

   CtS, EvalMod and StC then run under the dense key, and `ephemeral_hw` records h'. For the s → s' key, a single 60-bit special prime is enough (log modulus 54 + 60 = 114, against 294 with the full P). The only digit, q_0, is then 6 bits below p, so the key error is about 236·2^-6 ≈ 4 per coefficient, which is negligible at level 0. The sparse key also gets much more security margin (review M6; Task 37 decides).
2. SSE with 60-bit EvalMod primes, q0/Δ kept at 2^16: too thin, before the CtS term is counted.
3. Sparse main key h = 192 (K = 32, degree 255, r = 0, 8 levels, so it needs chain B): 19.7 bits at 2^12. Its security at log PQ ≈ 1748 is doubtful (§8).
4. Dense key without SSE: infeasible (12 EvalMod levels and ≤ 8.4 bits at 2^16).

## 8. Security expectation and what Task 39 needs

n16_train (log PQ 1592) estimated 143.9 bits under BDGL16 and 141.2 under MATZOV (`docs/security.md`). Scaling by n/log q at fixed n gives these expectations for the dense n16_boot at log PQ 1732:
- about **132 bits** under BDGL16;
- about **130** under MATZOV;
- about 99 under Core-SVP.

That is above 128 with a margin of about 2 bits under MATZOV. For comparison, the 8-level fallback chain B at 1748 would give about 131 / 128.6. The lattice-estimator run of Task 39 decides. If n16_boot comes in below 128, the security fallback is chain C (log PQ 1622; about 141 / 139; 14 levels after bootstrap).

**Task 39 needs:**
- (a) the dense n16_boot at log PQ 1732, and at 1730 if the q_0 = 52 fallback is used. Also chain B at 1748 if Task 37 cannot fit 7 EvalMod levels, and chain C at 1622 as the security fallback;
- (b) the SSE switching key s → s': N = 2^16, ternary secret with h = 32. The modulus is q_0·P with log(q_0·P) = 54 + 240 = 294 as ruled, or 114 with a single 60-bit special prime (§7, option 1). This is a sparse-secret instance, so the hybrid and MITM attacks must be enabled. The s' → s key uses the dense main key at modulus Q·P and is covered by (a);
- (c) h = 192 as an alternative only, at log PQ 1748 (it needs the 8-level chain B). The rough model in docs/security.md gave 107 bits at 1592, so I expect this to come in below 128.
