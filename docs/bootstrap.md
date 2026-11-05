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
> - **Predicted precision: 17.4 bits (§6). Measured in Task 38: 17.46 bits (§11.4), with 111 key switches per bootstrap (§11.2).**
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
- **StC scale.** The EvalMod output is v ≈ 2πΔc/q_0 at scale 2^58. That is the same plaintext as 2πc at scale Δ·2^58/q_0 ≈ 2^46, so the relabelling is free. The 1/(2π) factor and the step from 2^46 to Δ are folded into the StC diagonals. Every StC diagonal is encoded so that the rescale lands exactly on the next planned scale (2^45, 2^44, 2^43), and the last one exactly on Δ, the preset compute scale (Task 38's `ScaleMismatch` contract). Each StC diagonal scale is about 2^33, above the 2^20 plaintext floor (§9).
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

**This evaluation is a requirement on Task 37**, met by `bootstrap::poly` (§10.1: degree 127 in exactly 7 levels). Textbook Paterson–Stockmeyer, or a BSGS that rescales each c_i·T_i, costs ⌈log2(d+1)⌉ + 1. For the default (degree 127, r = 0) that is 8 levels, one more than the 7-level band.

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
- **Key-error digit term.** A key switch adds Σ_d [c]_{Q_d}·e_d/P, with coefficient std (Q_d/P)·3.2·sqrt(N/12) ≈ 236·(Q_d/P) per digit. Chain G has one all-58-bit digit (q_24..q_27), with Q_d = 2^232; the fallback chain B has two. With P = 4×58 the term would be about 2^15.4 slot rms on chain G, and would cost 1.2 bits at the worst case (it would predict 16.2). With P = 4×60, Q_d/P = 2^-8 and the term is negligible.

**Chosen design** (chain G; dense + SSE h' = 32; K = 13, r = 0, d = 127; 7 EvalMod levels at scale 2^58; q0/Δ = 2^12; P = 4×60):

| variant | predicted bits | bits each term alone allows (input / internal / approx / CtS pt / StC pt) |
|---|---|---|
| **chosen (chain G)** | **17.4** | 17.4 / 23.3 / 26.5 / 22.8 / 23.9 |
| P = 4×58 (digit term, worst case) | 16.2 | — |
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

That is above 128 with a margin of about 2 bits under MATZOV. For comparison, the 8-level fallback chain B at 1748 would give about 131 / 128.6. The lattice-estimator run of Task 39 decides. *(Superseded.)* This paragraph used to name chain C (log PQ 1622; about 141 / 139; 14 levels after bootstrap) as the fallback if n16_boot came in below 128. The parent's rule replaced that: trim the EvalMod prime width by 1 bit at a time (58 → 57 → 56) and keep every level. It did not trigger (Result below).

**Task 39 needs:**
- (a) the dense n16_boot at log PQ 1732, and at 1730 if the q_0 = 52 fallback is used. Also chain B at 1748 if Task 37 cannot fit 7 EvalMod levels, and chain C at 1622 as the security fallback;
- (b) the SSE switching key s → s': N = 2^16, ternary secret with h = 32. The modulus is q_0·P with log(q_0·P) = 54 + 240 = 294 as ruled, or 114 with a single 60-bit special prime (§7, option 1). This is a sparse-secret instance, so the hybrid and MITM attacks must be enabled. The s' → s key uses the dense main key at modulus Q·P and is covered by (a);
- (c) h = 192 as an alternative only. *(Superseded: this item used to say log PQ 1748 on the 8-level chain B. Ruling R-M3-10 keeps SSE with h' = 32 for h = 192, so it uses the n16_boot chain at 1732.)* The rough model in docs/security.md gave 107 bits at 1592, so I expect this to come in below 128.

**Result (Task 39, `docs/security.md`).** Under ruling R-M3-10, h = 192 uses the same chain with SSE, so only 1732 was needed; the fallback chains were not estimated.
- (a) dense n16_boot at 1732: 131.5 bits (BDGL16) and 129.3 (MATZOV); Core-SVP 95.2. No hybrid attack lowers it.
- (b) SSE key at log 294: 133.8 / 134.2, from the dual MITM hybrid (grid minimum with the numerical MITM solver; the analytical solver gives 134.4 / 135.0).
- (c) `n16_boot_sparse` (h = 192) at 1732: 122.2 / 121.7, below 128, so it is **not used in the paper**. Its bootstrap measures 21.12 bits (model 21.0).

n16_boot is kept unchanged, with no EvalMod prime trimmed.

## 9. CoeffToSlot and SlotToCoeff (Task 36)

`crates/hekit-ckks/src/bootstrap/linear.rs` implements both transforms. Its tests check them on `CkksParams::test_tiny()` (N = 2^5, log_slots 4, insecure, tests only). They count the n16_boot layouts and keys without generating any key.

### 9.1 The special DFT in SEAL's slot order

Let n = N/2 (full packing) and u_k = m_k + i·m_{k+n} for k < n. Slot j of the plaintext m decodes to z_j = m(ζ^(3^j)), because hekit uses SEAL's generator 3. The textbook radix-2 special FFT needs a generator g ≡ 1 mod 4, so that ζ_j^n = i for every slot. With generator 3 the value alternates: i at even j, −i at odd j.

The fix uses −3, which is ≡ 1 mod 4 and generates the same subgroup as 5. Since 3^j = (−3)^j for even j and −(−3)^j for odd j,

  z = C_o(U'' u),

where U'' is the special FFT for the generator −3 and C_o conjugates the odd slots.

**The stages.** U'' factors into log_slots butterfly stages S_len, for len = 2, 4, …, n. Each stage works on the pairs (j, j + len/2) of a block, with twiddle exp(2πi((−3)^j mod 4len)/(4len)). The twiddles depend on len only, not on N. The stages work in the bit-reversed layout. `dft_stages` merges them into `levels` groups.

**Removing C_o.** C_o is not complex-linear. Every stage except the distance-1 butterflies maps even slots to even slots, and for such a matrix S, C_o S C_o = ρ(S): the matrix with its odd rows conjugated. So C_o moves through every group except the one that holds the distance-1 stage:

- **SlotToCoeff.** z = ρ(H_L)···ρ(H_2)·[ρ(H_1)·a + ρ(i·H_1)·b], with a = Re u and b = Im u. The first stage takes the `re` and `im` ciphertexts as two inputs. It needs no conjugation, and its two BSGS sums share the giant rotations (`Evaluator::matvec_plain_sum`).
- **CoeffToSlot.** u = H_1^−1·C_o·ρ(R)·z, where R is the product of the other groups' inverses. The last stage is real-linear: w = A·h + B·conj(h), with
  - A = (D_e·H·D_e + D_o·conj(H)·D_o)/2 and B = (D_e·H·D_o + D_o·conj(H)·D_e)/2;
  - H = H_1^−1, and D_e, D_o the even and odd slot masks.

  This gives w = C_o(u)/2. Then re = w + conj(w) and im = X^(N/2)·(conj(w) − w). The monomial X^(N/2) multiplies slot j by i^(3^j) = (−1)^j·i. That is an exact signed permutation of the coefficients: no key, no level, no noise. It undoes the odd-slot conjugation. CoeffToSlot costs two conjugation key switches in all: one at the input of its last stage and one at the output. A and B have only even and only odd diagonals respectively, so half of their entries are zero. `matvec_plain_sum` skips those entries, and the baby rotations that only they would use.

**Output order.** The outputs are in the bit-reversed order of every bootstrapping DFT. Slot j of `re` holds m_{bitrev(j)}, and slot j of `im` holds m_{n+bitrev(j)}. SlotToCoeff takes that order back, and EvalMod is slot-wise. The brief's "decrypts to the coefficient sequence" is tested in this order.

**Folded factor.** `StageSpec::factor` is a complex c. CoeffToSlot then returns re + i·im = c·u, and SlotToCoeff returns c·z. The modulus |c| is spread evenly as |c|^(1/L) per stage. For CoeffToSlot the phase goes inside the conjugation sandwich of the last stage. For SlotToCoeff it goes onto the first stage. The 1/K of §2 is a factor of the CoeffToSlot list, and the 1/(2π) of §2 a factor of the SlotToCoeff list. Tested: c = 0.6 − 0.45i gives c times the unfolded result, both in the clear and encrypted.

### 9.2 Level collapsing

The `levels` groups split the radix evenly (`stage_radices`). When log_slots is not a multiple of `levels`, the groups farthest from the distance-1 stage get the extra stage. For n16_boot, 15 stages over 4 levels give **[3, 4, 4, 4]** from distance 1 upward. The radix-3 group holds distances 1, 2 and 4; the other groups hold 8–64, 128–1024 and 2048–16384. CoeffToSlot applies the groups in the reverse order.

The noise model of §6 splits the CtS and StC rounding terms evenly and does not depend on where the extra stages go. So the placement is chosen by cost:
- the distance-1 group is the two-input or real-linear stage, where every diagonal is multiplied twice, so it gets the smallest radix;
- the group with distance n/2 wraps around modulo n, so it has 2^r diagonals instead of 2^(r+1) − 1.

BSGS layouts (`d` diagonals at slot stride `s`, window starting at `offset`). The offset is a new field of `Diagonals` (`Diagonals::encode_complex`, `bsgs_rotation_steps_with_offset`). It lets a window cover negative diagonals, and its zero giant step costs no rotation.

| group (distances) | diagonals | d, stride, offset | BSGS | rotation keys |
|---|---|---|---|---|
| 1, 2, 4 | 15 | 16, 1, −8 | 4 × 4 | 1, 2, 3; −8, −4, 4 |
| 8 … 64 | 31 | 32, 8, −16 | 8 × 4 | 8 … 56; −128, −64, 64 |
| 128 … 1024 | 31 | 32, 128, −16 | 8 × 4 | 128 … 896; −2048, −1024, 1024 |
| 2048 … 16384 | 16 (wraps) | 16, 2048, 0 | 4 × 4 | 2048, 4096, 6144; 8192, 16384, 24576 |

Key switches per transform at n16_boot, from these layouts and the zero diagonals (test `n16_boot_rotation_steps`). The planner that gives them matches the op counter exactly on test_tiny (`encoded_steps_and_op_counts`); nothing was run at N = 2^16:
- **CoeffToSlot:** 6 + 10 + 10 + (3 baby + 3 giant) + 2 conjugations = **34**, and 93 plaintext products.
- **SlotToCoeff:** (6 baby + 3 giant) + 10 + 10 + 6 = **35**, and 108 plaintext products.

### 9.3 Rotation keys

The CoeffToSlot and SlotToCoeff groups have the same layouts, so they share their keys. `galois_steps_for` gives **32 rotation steps** for n16_boot (CtS 4 + StC 4, log_slots 15). That is below the brief's bound of 2·log_slots + stages = 38. The conjugation key comes on top.

One n16_boot Galois key is 8 digits × 2 polynomials × 36 limbs × 2^16 words = **0.302 GB**. The 33 keys take **9.97 GB**, before the relinearization key of EvalMod.

This is the largest memory item of the bootstrap, so Task 38 and the GPU port must plan for it. Ways to reduce it, none implemented here:
- smaller radices with more levels;
- truncating the keys that only SlotToCoeff uses to Q_20·P. Here every key is shared with CoeffToSlot, so this helps only if the layouts differ.

The encoded diagonals of n16_boot take about 1.8 GB for CoeffToSlot (levels 31–28) and 1.2 GB for SlotToCoeff (levels 20–17). Since Task 37 (review M6) the all-zero entries of the BSGS windows are not stored (`Diagonals::<Option<Plaintext>>::encode_complex_sparse`): 19 of the 112 CoeffToSlot windows and 4 of the 112 SlotToCoeff windows, about 0.29 GB and 0.04 GB less (test `n16_boot_stages_match_a_direct_dft`).

### 9.4 Scale history (exact)

Every stage is one BSGS product of plaintext diagonals followed by one rescale. `encode_stages` sets each stage's diagonal scale to L_k/s_k:
- s_k is the planned input scale;
- L_k is the label whose rescale by q_level gives exactly the next planned scale (`Evaluator::exact_product_scale`, the exact-scale path of the evaluator).

The planned scales are powers of two on the geometric path, and exactly `output_scale` after the last stage. At run time the product is relabelled to L_k. That differs from the f64 product of the scales by at most one rounding, and is checked to 1e−15. `coeff_to_slot` and `slot_to_coeff` reject an input whose scale is not exactly the planned one, with `ScaleMismatch`, before any work.

n16_boot (from the `n16_boot_rotation_steps` test, log2 of the scales):

| transform | level in → out | ciphertext scale | diagonal scales |
|---|---|---|---|
| CoeffToSlot | 31 → 27 | q_0 (2^54.00) → 2^55 → 2^56 → 2^57 → **2^58** | 2^57.00 at every stage |
| SlotToCoeff | 20 → 16 | 2^58·Δ/q_0 (2^46.00) → 2^45 → 2^44 → 2^43 → **2^42 = Δ** | 2^33.00, 2^33.00, 2^33.00, 2^33.00 |

The CtS diagonal scales multiply to q_28···q_31·2^58/q_0, as in §2. The StC output is exactly the preset scale at level 16 (`levels_after_boot`).

**Measured on test_tiny** (N = 2^5, the band layout in miniature: CtS 4 × 56 bits, StC 4 × 34 bits). These are maximum absolute errors over all slots:

| case | brief threshold | measured |
|---|---|---|
| `coeff_to_slot`, coefficients in [−1, 1], 1–4 stages | 2^−20 | 2^−50.4 … 2^−51.0 |
| the same, coefficients I + δ with \|I\| ≤ 5 and 1/K folded in (K = 6) | 2^−20 | 2^−50.6 … 2^−51.2 |
| `slot_to_coeff(coeff_to_slot(ct))`, (CtS, StC) stages (4, 4), (2, 4), (4, 3), (1, 2) | 2^−18 | 2^−25.8, 2^−27.0, 2^−26.2, 2^−22.7 |

In the identity test, StC starts from the CtS output scale 2^58 rather than from the planned 2^46, so its diagonal scales are only 2^26 to 2^30. That is the source of the 2^−22.7 with 2 StC stages. With the planned 2^46 input, the folded-factor test measures 2^−28.8. These N = 2^5 figures do not predict the n16_boot precision; Task 38 measures that.

**The input scale q_0 (review M8).** The 54-bit q_0 is not exactly representable in f64 (53-bit mantissa), so `q_0 as f64` is q_0 rounded to an even integer. Labelling the raised ciphertext with it decodes x = t/q_0 with a relative error of at most 2^-54, which is negligible. The stage checks compare f64 scales exactly, so Task 38 must use the same value, `params.q()[0] as f64`, for the relabel and for `StageSpec::input_scale`.

**Value test at log_slots 15 (review M1).** `n16_boot_stages_match_a_direct_dft` applies the n16_boot stage lists (4 + 4 levels) in the clear and compares them with a direct evaluation of z_j = m(ζ^(3^j)) on 48 sampled slots. It needs no keys and no encoding. Measured: forward stages 2^-45.5 and adapted SlotToCoeff (factor 0.6 − 0.45i) 2^-45.9, both relative to sqrt(N); adapted CoeffToSlot (factor 1/K) 2^-52.8 absolute. `coeff_to_slot` also rejects a stage list whose `second` diagonals are misplaced, as `slot_to_coeff` does (review M2).

## 10. ModRaise, SSE, Chebyshev evaluation and EvalMod (Task 37)

The code is in `crates/hekit-ckks/src/bootstrap/`: `modraise.rs`, `poly.rs` and `evalmod.rs`, plus `RnsPoly::raise_from_level0` in hekit-core and `KeyGenerator::sse_keys` / `SseKeys` in `keys.rs`.

### 10.1 Level-optimal, scale-invariant Chebyshev evaluation

`Evaluator::eval_chebyshev(ct, coeffs, a, b)` evaluates p(y) = Σ c_i T_i(y). It follows Bossuat, Mouchet, Troncoso-Pastoriza and Hubaux (Eurocrypt 2021, §4–5, Alg. 2). The recursion and the scale bookkeeping follow the algorithm of the paper, and the baby-step rule (`optimal_split`) follows Lattigo's split rule, which is a cost formula. The code was written from the algorithm: no Lattigo source was read or copied for it, so no third-party notice applies (Task 37 review I1). The result has exactly the input scale, or `target_scale` with `eval_chebyshev_to_scale`. It sits ⌈log2(d+1)⌉ levels lower; an interval other than [−1, 1] costs one more level.

**Power basis.** T_1 is the input at level L. Each T_i is 2·T_a·T_b − T_(a−b), with a = 2^(⌈log2 i⌉−1) and b = i − a. The steps are:
1. multiply T_a and T_b at the lower of their two levels, and relinearize;
2. double the product by adding it to itself (relabelling it to half its scale instead would halve the scale of every later power);
3. subtract T_(a−b) as an integer multiple round(S_aS_b / S_(a−b)) labelled S_aS_b, or the integer round(S_aS_b) for T_0 = 1;
4. rescale once.

T_i lies at level L − ⌈log2 i⌉ with scale S_aS_b/q_l, about 2^58 in the EvalMod band. The powers are built on demand. An odd series therefore builds only the powers of two and the odd T_i.

**Recursion.** rec(p, l, ts) returns p at level l with the exact scale ts, before its rescale. Let m = deg p.
- **Leaf.** When m < 2^s (s = `optimal_split(D)`, so 4 and leaves of degree up to 15 for D = 7) and T_m lies at level ≥ l: Σ k_i T_i + round(c_0·ts). Each k_i = round(c_i·ts/S_i) is an integer constant of about log2 q_l bits (ts ≈ S·q_l). The product is labelled exactly ts, with no rescale and so no level. A constant scale below 2^20 or an integer of 2^126 or more is refused.
- **Split.** Otherwise split at k = 2^j, the largest power of two ≤ m, as p = q·T_k + r, using T_i = 2T_(i−k)T_k − T_(2k−i). Then:
  - q is asked for at (l + 1, ts·q_(l+1)/S_k), relinearized and rescaled, which leaves scale ts/S_k at level l;
  - it is multiplied by T_k and relabelled ts (a few f64 roundings, checked to 1e-12);
  - r is asked for at (l, ts) and added.
- **Top.** rec(p, L − D + 1, label) with D = ⌈log2(d+1)⌉. The label comes from `exact_product_scale(L − D + 1, target)`, and one final relinearization and rescale put the result at level L − D with exactly the target scale.

**Why D levels suffice.** rec(p, l) is feasible when l ≤ L − ⌈log2(m+1)⌉ + 1. A split at 2^j has ⌈log2(m+1)⌉ = j + 1, so l ≤ L − j and three things hold:
- T_k (level L − j) is available at l;
- q, of degree below 2^j, is asked for at l + 1 ≤ L − j + 1, within its bound;
- r, of degree below 2^j, is asked for at l, also within its bound.

The leaf condition holds for m = 1 at any l ≤ L. The leading branch (the top coefficients) is tight, so it splits all the way down to degree 1. The other branches have one level of slack, and their leaves have degree up to 15. The scalar products are never rescaled on their own, and that is the level a textbook evaluator spends.

**Cost of the n16 EvalMod series** (degree 127, odd; counted in `odd_degree_127_costs` and an independent Python simulation of the recursion):
- 13 powers: T_2, T_4, …, T_64 and the odd T_3 … T_15;
- 10 splits;
- so 23 ciphertext products, 20 relinearizations (products that are only summed are relinearized once) and 24 rescales;
- 71 integer-constant products (64 leaf coefficients and 7 power corrections).

A dense degree-127 series needs 17 powers and 128 constant products.

**Modulus room.** The last unrescaled product, at level L − D + 1, has scale about S_out·q_(L−D+1). Q at that level must hold it. For n16_boot that is about 2^116 at level 21, against Q_21 ≈ 2^920.

**Measured** (N = 2^10, insecure test chains, errors against the plaintext series):

| case | threshold | measured |
|---|---|---|
| `chebyshev_coeffs`, degree 31: cos x and cos 2πx on [−1, 1], cos(2π(x − 1/4)) on [−2, 2] | 2^-20 | 2^-47.4, 2^-47.2, 2^-35.0 |
| `chebyshev_coeffs`, degree 127: sin(2π·13y) on [−1, 1] (10001-point grid) | — | 2^-45.1 |
| x³ − x, degree 3: [−1, 1] (2 levels) and [−2, 2] (3 levels), 42-bit primes at scale 2^42 | 2^-20 | 2^-32.1, 2^-28.4 |
| the same in the 58-bit band at scale 2^58 | 2^-20 | 2^-47.3, 2^-45.1 |
| random series of degree 1 … 255, each in exactly ⌈log2(d+1)⌉ levels at scale 2^58 | 2^-25 | 2^-48.3 … 2^-41.9 |
| sin(2π·13y), degree 127, level 8 → 1 | 2^-25 | 2^-39.6 |

### 10.2 EvalMod

`Evaluator::eval_mod(ct, bp)` takes the CoeffToSlot output. Its slots must be normalized as

  y_j = x_j / K = (I_j + Δc_j/q_0) / K ∈ [−1, 1],

where t_j = Δc_j + q_0 I_j is coefficient j of the raised plaintext. The 1/K is folded into CoeffToSlot through `StageSpec::factor`, so eval_mod no longer takes q_0 (a deviation from the brief's `eval_mod(ct, bp, q0)`). It returns sin(2πK y_j) = sin(2π x_j) ≈ 2πΔc_j/q_0 at the same scale, `bp.evalmod_depth()` levels lower. For n16_boot that is level 27 → 20, both at scale 2^58. The factor q_0/(2π) back to Δc_j belongs to SlotToCoeff (§2).

**The series.** For r = 0 it is the degree-127 interpolant of sin(2π·13·y), with its even coefficients set exactly to zero because the sine is odd. With r > 0 it is cos(2π(Ky − 1/4)/2^r) followed by r double angles. Each double angle is one level. Its square is relabelled to the exact label of the next planned scale, and the Chebyshev output scale is chosen within a few ulps of sqrt(2σ) for that. There is no arcsine correction (§6).

**Measured.** N = 2^10, the n16 constants (K = 13, degree 127, r = 0), inputs m + q_0 I with |m| ≤ 1 and |I| ≤ 13: the edge slots carry I = ±13 with m pointing inward, so x stays in [−K, K]. The result m' = q_0/(2πΔ)·output matches m to **2^-21.3** (threshold 2^-16) in exactly 7 levels. This is the sine-versus-modulus term at |m| = 1, (2πΔ/q_0)²/6 = 2^-21.3, so it is not noise. In a bootstrap the sine acts on the coefficients, whose rms is 2^-8.8 for slots in [−1, 1], and §6 puts this term at 36.1 bits.

**Preconditions** (stated in the `eval_mod` rustdoc, review m2): |y| ≤ 1 and message slots |m| ≤ 1.
- **Just past the edge.** The outward edge, I = ±K with m pointing outward (|y| − 1 ≈ 2^-15.8), still measures 2^-21.5 at N = 2^10.
- **Further out.** For |I| > K the series diverges.
- **Larger messages.** Precision falls quadratically with |m|.
- **r > 0.** This path is experimental: it is checked for levels and exact scale only.

**n16_boot** (`n16_boot_keys_eval_mod`, `BootParams::default_n16()`, all 2^15 slots): level 27 → 20 (7 = `evalmod_levels`), scale exactly 2^58 in and out, max |m − m'| = **2^-21.27** (threshold 2^-16), against a sine term alone of 2^-21.28. The counts are 23 products, 20 relinearizations, 24 rescales and 71 constant products, as in §10.1. It took 5.0 s on one thread of the Mac while another job ran.

### 10.3 ModRaise with SSE

`KeyGenerator::sse_keys(h')` samples an ephemeral ternary s' with exactly h' nonzero coefficients and keeps two keys in `SseKeys`:
- `to_sparse`: s → s' modulo q_0·P, with one part and level-0 Q parts. Its RLWE samples are under s' with log(q_0 P) = 294 (§8 (b)).
- `to_dense`: s' → s modulo Q·P, with dnum parts.

s' itself is dropped. `Evaluator::mod_raise(ct, &sse)` works in three steps:
1. at level 0, key-switch c_1 to s';
2. raise both components to the top level with `RnsPoly::raise_from_level0`, the centered lift into [−(q_0−1)/2, (q_0−1)/2];
3. key-switch c_1 back to s at the top level.

The scale stays Δ, and Task 38 relabels it to q_0. `mod_raise_main_key` skips SSE and is for tests only.

**Key switches.** Each `mod_raise` performs exactly 2 key switches: one at level 0 with `to_sparse`, and one at the top level with `to_dense`. Since Task 38 they are counted as `OpCounter::sse_switches` (+2 per successful `mod_raise`, included in `key_switches()`), so that RQ1 counts every key switch (§11.2).

**Noise of the two switches.**
- The level-0 switch adds Σ d·e/P + r_b + r_a·s'. The digit term is about q_0/P ≈ 2^-186 of the key error. The P-rounding term is sqrt((h'+1)/12) = 1.66 per coefficient.
- The top switch adds the dense rounding term sqrt((N/2+1)/12), which is 52 at N = 2^16, plus the digit term of §6 (negligible with P = 4×60).

**What each term alone allows**, in the §6 model −log2(4.2·sqrt(N/2)·σ/Δ) with σ the coefficient rms:
- the level-0 switch, about 31.7 bits at n16_boot;
- the top switch, about 26.7 bits.

Both are far above the binding 17.4-bit input-noise term, and the top switch moves the total by under 0.1 bit.

**Measured** (N = 2^12, insecure test chain with α = 2, 4 ciphertexts, 16384 coefficients):

| quantity | predicted | measured |
|---|---|---|
| σ_I under SSE, h' = 32 | sqrt(33/12) = 1.658 | 1.663 (mean −0.005, max \|I\| = 7 ≤ K = 13) |
| σ_I without SSE (dense key, h = 1994) | 12.89 | 12.92 |
| level-0 switch s → s', rms per coefficient | 1.658 | 1.682 (33.7 bits alone at N = 2^12) |
| top switch s' → s, rms per coefficient | sqrt(2049/12) = 13.1 | 12.90 (30.7 bits alone) |

**n16_boot** (`n16_boot_keys_sse_mod_raise`, dense main key, h' = 32, two ciphertexts, 2^17 coefficients):

| quantity | predicted | measured |
|---|---|---|
| σ_I | 1.658 | **1.664** (mean −0.003, max \|I\| = 8 ≤ K = 13) |
| level-0 switch s → s', rms per coefficient | 1.658 | **1.654**: 31.7 bits alone |
| top switch s' → s, rms per coefficient | sqrt(32769/12) = 52.3 | **52.35**: 26.7 bits alone |
| residual t − Δc − q_0 I (fresh encryption + both switches) | — | 74.0: 26.2 bits alone |

**Consistency with the input-noise term.** The binding term of §6 allows 17.4 bits. The level-0 switch allows 31.7 bits alone, 14 bits of margin, so it is negligible. The top switch allows 26.7 bits. In the §6 units (coefficient error in x units) it is 2^-48.3 against the binding term's 2^-39.0, so adding it in quadrature costs under 0.01 bit and the 17.4-bit prediction stands.

## 11. The assembled bootstrap (Task 38)

`crates/hekit-ckks/src/bootstrap/bootstrapper.rs` has `BootstrapKeys` and `Bootstrapper`; `crates/hekit-nn/src/refresh.rs` has `BootstrapRefresher`.

### 11.1 Pipeline and interfaces

`Bootstrapper::new(params, bp, &encoder, &mut kg)` encodes the stages and generates the keys. `bootstrap(&ev, &ct)` runs the §2 pipeline:
1. `mod_raise` with SSE (2 key switches), relabelled to the scale `q_0 as f64`;
2. CoeffToSlot from level 31, input scale q_0, output exactly `scale_at(27)` = 2^58, factor 1/K;
3. `eval_mod` on `re` and on `im` (27 → 20, scale 2^58);
4. relabel both to 2^58·Δ/q_0 (2^46.00); SlotToCoeff from level 20 to exactly Δ at level 16, factor 1/(2π).

The input must be at level 0 (else `InvalidParams`) with exactly Δ (else `ScaleMismatch`). Both checks, and the evaluator checks (same parameters, a relinearization key), run before any work. The output is at `levels_after_boot` = 16 with exactly Δ.

Deviations from the brief's interface:
- `BootstrapKeys` keeps the conjugation key inside its `GaloisKeys` (behind an `Arc`), not in a separate `conj` field. `Evaluator::conjugate` looks it up there, and sharing the `Arc` avoids copying 10 GB per bootstrap. It also holds the `SseKeys`.
- The bootstrap runs on an internal evaluator that shares the caller's relinearization key and the bootstrapper's Galois keys. On success its counts are added to the caller's counter.
- `Bootstrapper::cost_per_boot()` plans every count from the encoded stages (`transforms_cost`, which skips the stored-as-zero diagonals as the BSGS does) and from the EvalMod series (`evalmod_cost` = `chebyshev_cost`, a ciphertext-free run of the same recursion). The tests assert that a real bootstrap adds exactly these counts, at N = 2^10 and at n16_boot.
- Optional rayon parallelism (cargo feature `parallel`, off by default, `src/par.rs`) covers the digits and limbs of key switching, the hoisted rotation steps, the BSGS giant-step blocks, and the two EvalMods. Every parallel section is collected in its sequential order, so the output is bit-identical for any thread count (tested on 1 and 4 threads).

### 11.2 Counting (RQ1)

`OpCounter::key_switches()` is `rotations + relins + sse_switches`. There is no per-bootstrap constant any more: `BOOT_KS_EQUIV` is removed. A bootstrap adds its own rotations, conjugations, relinearizations and its 2 SSE switches (the `sse_switches` field) to the caller's counter, and adds 1 to `bootstraps`, which is a plain event count. A measured counter therefore never double-counts. A simulator or cost model that plans `b` bootstraps instead of running them adds `b · Bootstrapper::key_switches_per_boot()`.

Per n16_boot bootstrap (measured by the counter, equal to the plan):

| part | key switches | other operations |
|---|---|---|
| mod-raise (SSE) | 2 | — |
| CoeffToSlot | 34 (32 rotations + 2 conjugations) | 93 plaintext products, 4 rescales |
| EvalMod × 2 | 40 relinearizations | 46 products, 142 constant products, 48 rescales |
| SlotToCoeff | 35 rotations | 108 plaintext products, 4 rescales |
| **total** | **111** | 46 ct-ct products, 343 plaintext/constant products, 56 rescales |

**Against the M0 cost model.** The cost model's BOOT_KS = 502 is ReBoot's OpenFHE EvalBootstrap: Meta-BTS with 2 iterations, each with 2 × 109 linear-transform rotations, 1 conjugation and an estimated 2 × 16 EvalMod key switches. One hekit bootstrap measures **111**, against 251 for one OpenFHE iteration and 502 for ReBoot's two. Two things make the difference:
- the 4 + 4-level CtS/StC collapse with BSGS (69 rotations against 218);
- the level-optimal degree-127 EvalMod (20 relinearizations per part).

`spikes/costmodel/costmodel.py` now has `HEKIT_BOOT_KS = {2^15: 111}`. The default of `network_cost` stays the OpenFHE estimate, because it is the bootstrap of the ReBoot side and the M0 tables use it for both layouts. `table.py` prints two new rows:
- the verdict at ReBoot's setting (N = 2^16, B = 48) with diag_batched at hekit's measured 111 and ReBoot at its 502: KS_spec 26.1 against 78.8, a ratio of **0.331** (0.744 with 502 on both sides);
- a sensitivity row at BOOT_KS = 111 on both sides: 1.12 / 0.74 / 0.56 / 0.44 / 0.26 for B = 16 / 32 / 48 / 64 / 128.

### 11.3 Keys and memory (n16_boot)

| item | count | size |
|---|---|---|
| rotation keys (`galois_steps_for`, CtS and StC share them) | 32 | 0.302 GB each |
| conjugation key | 1 | 0.302 GB |
| SSE s' → s (mod Q·P, 8 parts) | 1 | 0.302 GB |
| SSE s → s' (mod q_0·P, 1 part) | 1 | 5.2 MB |
| **`BootstrapKeys::bytes()`** | | **10.27 GB** |
| relinearization key (the caller's evaluator) | 1 | 0.302 GB |
| encoded CtS + StC diagonals | 93 + 108 plaintexts | about 3 GB (§9.3) |

`Bootstrapper::new` takes 27 s on one thread of the Mac: key generation plus encoding.

### 11.4 Precision (gate: ≥ 16 bits, all 2^15 slots, m uniform in [−1, 1])

| case | measured | model (§6) |
|---|---|---|
| **n16_boot, m uniform in [−1, 1]** (`n16_boot_keys_bootstrap`, the gate) | **17.46 bits** | 17.4 |
| n16_boot, m uniform in [−1, 1], second seed (observation) | 17.41 bits | 17.4 |
| n16_boot, \|m\| = 1, random signs (observation) | 17.53 bits | — |
| n16_boot, \|m\| = 4, random signs (observation, outside the \|m\| ≤ 1 precondition) | 17.38 bits absolute (19.4 relative to \|m\|) | — |
| N = 2^10 miniature (same bands, insecure), m uniform in [−1, 1] | 26.77 bits; 26.18 after a second bootstrap | — |

The measurement matches the model to 0.1 bit, so the q_0 = 52 fallback is not needed.

**|m| = 4 costs nothing in absolute terms.** The precondition |m| ≤ 1 of `eval_mod` is conservative for full-slot bootstrapping. The sine acts on the coefficients, and at |m| = 4 these have an rms of only about 4/sqrt(N) = 2^-6, so the sine-versus-modulus term still allows about 28 bits (§6 gives 27.1 bits at |m| ≤ 8). The absolute error stays at the input-noise floor. Slot magnitudes are bounded by |m|·Δ < q_0/2 at level 0 (|m| < 2^11) and by the EvalMod interval. The Review Focus 2 observation is therefore that |m| = 4 is usable at 17.4 absolute bits; it is not gated.

**Per-stage errors** (`n16_boot_keys_observe_m1`, m uniform in [−1, 1]). The "bits alone" column applies the §6 conversion −log2(4.2·sqrt(N/2)·(q_0/Δ)·σ) to an x-unit rms σ:

| stage (output level, scale) | max error | rms error | bits alone | model term (§6) |
|---|---|---|---|---|
| mod-raise with SSE: coefficient noise of t − Δc − q_0 I (max \|I\| = 7 ≤ K − 1) | 336 | 73.8 (2^-47.8 in x units) | 26.2 | 26.7 top switch (§10.3) |
| CoeffToSlot (27, 2^58.00), x error, raise included | 2^-37.0 | 2^-39.9 | 18.3 | 17.4 input noise |
| EvalMod (20, 2^58.00), against sin(2πx)/(2π) | 2^-36.0 | 2^-39.4 | 17.8 | + internal 23.3, approximation 26.5 |
| SlotToCoeff (16, 2^42.00 = Δ), slot error | 2^-17.53 (2^-17.41 uniform m) | 2^-19.55 | **17.5 / 17.4** | 17.4 total |

The rows show uniform m. The \|m\| = 1 run gives the same numbers to within 0.1 bit. The binding term is the CoeffToSlot output noise, as §6 predicts. The model is 0.9 bits pessimistic for that stage alone, and EvalMod and SlotToCoeff together take another 0.9 bit.

### 11.5 Time

| run | threads | time per bootstrap |
|---|---|---|
| `n16_boot_keys_bootstrap` (test profile, no `parallel`), Mac | 1 | 37.8 s, 38.8 s, 39.1 s (three runs, with Task 33's timed baseline running on the same machine) |
| criterion `benches/bootstrap.rs` (`--features parallel`, bench profile with LTO), Mac | 1 | **23.8 s** (95% interval 22.4–25.5 s, 10 samples) |
| the same | 12 | **7.70 s** (7.41–8.05 s, 10 samples): a 3.1× speedup |
| criterion, quadruped-gpu | 1 and 32 | **pending (user-driven session)** |

Run: `cargo bench -p hekit-ckks --features parallel --bench bootstrap` on 2026-11-05, setup 22.5 s. The load average was 6.30 / 5.52 / 5.89 before and 10.96 / 8.10 / 6.89 after (1 / 5 / 15 min). The before value includes the bench build that had just finished. A reviewer's single-threaded Sage estimator held one core for the whole run, so the 12-thread run shared that core.

**Target (≤ 10 s with 32 threads): met on the Mac with 12 threads (7.70 s).** The Mac has 12 cores, so the brief's "32 threads" runs as 12 threads here. The 32-thread figure on quadruped-gpu is pending.

The test-profile runs are slower than the criterion 1-thread run (38 s against 23.8 s). They use no LTO and ran next to Task 33's baseline. In the single-thread run, EvalMod takes about 10 s (2 × 5.0 s, §10.2). The rest is the 69 rotations of CoeffToSlot and SlotToCoeff at levels 31–17 and the two SSE switches. The parallel speedup is limited by what stays sequential:
- the Chebyshev recursion (only the two EvalMods run side by side);
- `mod_down` and the NTTs inside each key switch, which are in hekit-core and not parallelized;
- rescales and plaintext products.

### 11.6 Heavy tests

The tests that generate the full set of n16_boot bootstrapping keys (about 13 GB) are `#[ignore]`d, because CI runners cannot hold them. They are all named `n16_boot_keys_*`:

| test | file | peak memory | default |
|---|---|---|---|
| `n16_boot_keys_bootstrap`, the **precision gate**, dense main key (17.46 bits) | bootstrapper.rs | ~13 GB (keys 10.27 GB + diagonals) | ignored |
| `n16_boot_keys_bootstrap_sparse`, the precision gate on `n16_boot_sparse` (Task 39) | bootstrapper.rs | ~13 GB | ignored |
| `n16_boot_keys_observe_m1` and `_m4`, observations with the per-stage breakdown | bootstrapper.rs | ~13 GB | ignored |
| `n16_boot_keys_eval_mod` (§10.2) | evalmod.rs | 1.19 GB measured (one relinearization key), 9 s | runs |
| `n16_boot_keys_sse_mod_raise` (§10.3) | modraise.rs | 0.74 GB measured (SSE keys only), 2 s | runs |

The last two stay in the default run, because they need far less than the ~10 GB threshold.

The n16_boot tests that need no keys (layouts, scale history, stage values: `n16_boot_rotation_steps`, `n16_boot_stages_match_a_direct_dft` and the params tests) still run by default. On a machine with at least 16 GB free, run the ignored ones one at a time:

```
RUST_TEST_THREADS=1 cargo test -p hekit-ckks --lib -- --ignored n16_boot_keys --skip observe
RUST_TEST_THREADS=1 cargo test -p hekit-ckks --lib -- --ignored n16_boot_keys_observe --nocapture
```

hekit-nn's n16_boot tests (`tests/compact.rs`, `tests/trainer_ni.rs`) were already ignored on the same grounds. `trainer_ni`'s gate (Task 41, 3 non-interactive updates on n16_boot, ~13 GB of bootstrap keys plus the training keys) is `n16_boot_keys_three_updates`, with the same reason string:

```
cargo test --release -p hekit-nn --test trainer_ni -- --ignored n16_boot_keys --nocapture
```
