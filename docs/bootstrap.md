# Bootstrap level design for n16_boot (M3, Task 35)

This document fixes the modulus chain of `CkksParams::n16_boot()`, the per-level default
scale `scale_at`, the level budget of mod-raise, CoeffToSlot, EvalMod and SlotToCoeff, and the
EvalMod constants of `BootParams::default_n16()`. It also checks whether the plan's 16-bit
precision target can be met. Tasks 36–39 build on it.

Every number below comes from `tools/bootstrap/evalmod_design.py` (numpy only, under a second:
`python3 tools/bootstrap/evalmod_design.py`).

> **Decision needed before Task 37/38.** With the plan's constants (q0/Δ = 2^58/2^42 = 2^16,
> 58-bit EvalMod primes, dense ternary secret), no EvalMod design reaches 16 bits of precision.
> Moving levels between bands does not help either (§5, §6). Two plan claims also do not
> survive the calculation. For a dense key the plan says K ≈ 25, but a 2^-30 failure rate per
> bootstrap needs K ≈ 400–420. For h = 192 the plan says K ≈ 12, but h = 192 needs K = 31–33.
> K = 12–13 is the value for h = 32. The constants committed here (K = 13, degree 127, r = 1)
> are for the recommended fix in §7: sparse-secret encapsulation (SSE) to an ephemeral h' = 32
> key for the mod-raise, plus q0/Δ ≤ 2^12. Without SSE they are wrong for the default dense key.

## 1. Modulus chain and scales

Level i means a ciphertext over primes q_0..q_i. Bands from the bottom:

| levels | primes | bits | band | `scale_at(level)` |
|---|---|---|---|---|
| 0 | q_0 | 58 | base | 2^42 (preset scale) |
| 1–16 | q_1..q_16 | 42 | compute | 2^42 |
| 17–19 | q_17..q_19 | 42 | SlotToCoeff (StC) | 2^42 |
| 20–27 | q_20..q_27 | 58 | EvalMod | 2^58 |
| 28–31 | q_28..q_31 | 42 | CoeffToSlot (CtS) | 2^42 |
| P | p_0..p_3 | 58 | special | — |

`q_bits = [58, 42×16, 42×3, 58×8, 42×4]`, `p_bits = [58×4]`, dnum 8 (4 primes per digit),
max_level 31. Nominal log PQ = 13·58 + 23·42 = 1720. The exact value is 1720.0 minus about
1e-9, because every prime sits just below its power of two. This is within the plan ceiling
of 1750. Primes come from the SEAL-style joint generation of `CkksParams::new`. P gets the four
largest 58-bit primes, so the two all-58-bit digits (q_20..q_23 and q_24..q_27) stay strictly
below P (`n16_boot_digits_below_p_exactly`).

`scale_at(level)` is 2^(bits of q_level) for level ≥ 1, and the preset scale (2^42) at level 0.
It is the scale a ciphertext at that level should carry when it is multiplied by another
ciphertext and rescaled by q_level. For CtS and StC the ciphertext is multiplied only by
plaintext diagonals, which are encoded at the exact scale q_l. The ciphertext scale is
therefore carried through those bands unchanged (§2). For the single-size presets (seal_2020,
n16_train), `scale_at` equals `scale()` at every level.

`levels_after_boot(params, bp) = max_level − (ctos + evalmod + stoc) = 31 − 15 = 16`: after a
bootstrap the ciphertext is at level 16, with 16 compute levels plus the base.

## 2. Pipeline: levels and scales

The input is a ciphertext at level 0 with scale Δ = 2^42 (Task 38 checks both conditions). Its
plaintext coefficients are Δ·c_j. Here c = encode(m), and for m uniform in [-1, 1] in all 2^15
slots the coefficient rms is 1/sqrt(3N) ≈ 2^-8.8.

| step | levels | primes consumed | scale in / out | content after the step |
|---|---|---|---|---|
| ModRaise | 0 → 31 | — | Δ | t_j = Δ c_j + q_0 I_j (I_j integer, §3); reinterpreted for free as x = t/q_0 at scale q_0 ≈ 2^58 |
| CoeffToSlot | 31 → 27 | q_31..q_28 (42-bit) | 2^58 / 2^58 | two ciphertexts (real halves of the N coefficients) with slots y_j = x_j / K in [-1, 1]; the 1/K and the conjugate-split 1/2 are folded into the diagonals (no level) |
| EvalMod | 27 → 19 | q_27..q_20 (58-bit) | 2^58 / 2^58 | sin(2π x_j) = cos(2π(x_j − 1/4)): Chebyshev degree d (⌈log2(d+1)⌉ levels) then r double angles (r levels) |
| SlotToCoeff | 19 → 16 | q_19..q_17 (42-bit) | 2^42 / 2^42 | coefficients q_0 sin(2π x_j)/(2π) ≈ Δ c_j: m at scale Δ, level 16 |

Notes:
- At the start of EvalMod the scale is q_0 (carried through CtS), and q_0 sits next to the
  58-bit EvalMod primes. The ciphertext-by-ciphertext products in EvalMod are then rescaled by
  primes of the same size, so `scale_at(27) = 2^58` holds.
- At EvalMod output the ciphertext holds v ≈ 2πΔc/q_0 at scale ≈ 2^58. As a plaintext this is
  the same as 2πc at scale Δ·2^58/q_0. Relabelling the scale is free. The 1/(2π) is folded into
  the StC diagonals, and StC rescales by 42-bit primes with the diagonals at q_l, so the output
  is at scale ≈ 2^42 = `scale_at(16)`. The exact output scale differs from 2^42 by the ratio of
  the actual primes to powers of two. Task 38 encodes the last StC diagonal at the scale that
  lands exactly on the preset compute scale (the `ScaleMismatch` contract).
- No arcsine correction is used. The sine-versus-modular-reduction error is far below the
  target (§6).

## 3. The overflow I and the interval K

After mod-raise, I_j = round(b_j/q_0 + Σ_{s_k ≠ 0} ±a_k/q_0), where b and a are uniform mod
q_0 for any ciphertext that went through computation. I_j is therefore the rounded sum of h+1
iid U(−1/2, 1/2) variables, where h is the Hamming weight of the key under which the mod-raise
happens. Its standard deviation is σ_I = sqrt((h+1)/12).

The EvalMod polynomial is accurate on x ∈ [−K, K] (`k_intervals = K`). This covers
|I_j| ≤ K − 1 with room for the tiny message part. A bootstrap fails if some coefficient has
|I_j| ≥ K; the union bound over all N = 2^16 coefficients gives the failure rate. Target:
**≤ 2^-30 per bootstrap** (a training run does at most about 2^14 bootstraps).

| key | h | σ_I | log2 P_fail per bootstrap |
|---|---|---|---|
| dense (hekit default: −1/0/1 with probability 1/4, 1/2, 1/4) | ≈ N/2 = 32768 | 52.26 | K=25: +15.4 (certain failure); K=400: −29.4; K=420: −33.8 |
| sparse | 192 | 4.01 | K=12: +8.1; K=25: −14.4; K=30: −27.4; K=32: −33.3; K=33: −36.4 |
| sparse (SSE ephemeral) | 32 | 1.66 | K=12: −29.1; K=13: −39.7; K=14: −53.4 |

The dense rows are Gaussian. The h = 192 and h = 32 rows are the exact Irwin–Hall tail; the
Gaussian tail is slightly pessimistic for those. A positive value means the expected number of
failing coefficients is above 1.

The plan's "dense K ≈ 25" fails with certainty. At K = 25, 64% of the coefficients overflow.
The plan's "h = 192 K ≈ 12" fails about 270 coefficients per bootstrap. The values that meet
the target are K ≈ 400–420 (dense), K = 31–33 (h = 192) and K = 13 (h = 32).

## 4. EvalMod: Chebyshev approximation plus double angles

We approximate f_r(x) = cos(2π(x − 1/4)/2^r) on x ∈ [−K, K], with y = x/K ∈ [−1, 1] folded
into CtS, by Chebyshev interpolation of degree d. Then r double-angle steps c ← 2c² − 1 give
cos(2π(x − 1/4)) = sin(2πx).

**Depth.** The baby-step giant-step evaluation in the Chebyshev basis (Bossuat et al.,
Eurocrypt 2021) uses ⌈log2(d + 1)⌉ levels, and each double angle uses one more:

  evalmod_depth = ⌈log2(cheb_degree + 1)⌉ + double_angle ≤ evalmod_levels.

`BootParams::validate` enforces this, and `evalmod_depth()` computes it.

**Error.** For a Chebyshev interpolant of cos(ω y + φ), the coefficients are Bessel values
J_n(ω) with ω = 2πK/2^r. These die off only once n > ω, so the degree needed is a bit above
2πK/2^r. Each double angle halves ω but costs a level, and it amplifies an error δ by
4|cos θ| (2(c+δ)² − 1 = cos 2θ + 4δ cos θ + 2δ²). For the angles that occur, the amplification
is at most `amp` = max_I Π_i 4|cos(2^i θ_I)|. Measured maximum sine error at the points
x = I + δ that occur (|I| ≤ K − 1, |δ| ≤ 2^-20), with all 8 EvalMod levels used
(d = 2^(8−r) − 1). Values around 2^-45 are the float64 floor of the measurement:

| K | r=0, d=255 | r=1, d=127 | r=2, d=63 | r=3, d=31 |
|---|---|---|---|---|
| 12 | 2^-45.5 | 2^-44.2 (amp 2.8) | 2^-43.5 (amp 10.5) | 2^-40.7 (amp 41) |
| 13 | 2^-45.4 | 2^-43.8 | 2^-43.7 | 2^-37.2 |
| 16 | 2^-46.0 | 2^-43.6 | 2^-43.8 | 2^-28.1 |
| 25 | 2^-44.6 | 2^-43.0 | 2^-25.5 | 2^-10.0 |
| 30 | 2^-44.5 | 2^-28.8 | 2^-12.8 | 2^-3.6 |
| 32 | 2^-42.7 | 2^-20.9 | 2^-8.7 | 2^-1.4 |
| 34 | 2^-29.5 | 2^-14.1 | 2^-5.3 | 2^0.6 |

8 levels carry K ≤ 32 at about 2^-42 or better, but only with r = 0 (degree 255). The dense
key needs about 12 levels: for K = 400 or 420, degree 2^L − 1 with r = 0 still has error
around 1 at L = 9, 10 and 11, and reaches 2^-39.8 only at L = 12. Raising r does not lower the
depth. ⌈log2(d+1)⌉ + r stays near log2(2πK·1.2) whatever r is, and r only adds amplification.

**Chosen constants** (`N16_K_INTERVALS`, `N16_CHEB_DEGREE`, `N16_DOUBLE_ANGLE`):
**K = 13, degree 127, r = 1** → depth ⌈log2 128⌉ + 1 = 7 + 1 = 8 = `evalmod_levels`.
Sine error is 2^-43.8 (the amplification of 2.8 is included). The failure rate is 2^-39.7
with SSE to h' = 32. Why r = 1: r = 0 (degree 255) and r = 2 (degree 63) give the same
predicted precision within 0.1 bit (§6). r = 1 needs about 2·sqrt(128) ≈ 23 ciphertext
products instead of about 32 for degree 255. Its internal-noise amplification (2.8) stays far
from binding, whereas for r = 2 (10.5) the internal-noise term comes within 1.4 bits of the
input term.

**For Task 39: the sparse h = 192 numbers.** K = 32 (failure 2^-33.3), degree 255, r = 0:
depth 8, sine error 2^-42.7. K = 33 does not fit 8 levels at that accuracy. The plan's K ≈ 12
is not valid for h = 192.

## 5. Why the dense default cannot be fixed by rebalancing the bands

A dense key without SSE needs K ≈ 420 and therefore 12 EvalMod levels. Taking 4 levels from
the compute band (16 → 12, the floor) and turning them into 58-bit EvalMod primes gives
log PQ = 1720 + 4·16 = 1784. That is above the 1750 ceiling and above the extrapolated
128-bit bound for N = 2^16. Keeping log PQ ≤ 1750 with 12 compute levels forces 31 or 30 q
primes. With 31 primes, dnum must divide a prime number. With 30 primes, dnum 10 and k = 3,
CtS and StC together get only 5 levels. Even a chain that did fit would fail the precision
gate. With K = 420, the CtS noise term alone limits precision to **9.4 bits** at q0/Δ = 2^16
(§6). The binding constraint is precision, not depth, so the preset keeps the plan's chain and
the Step 1 numbers are unchanged.

## 6. Precision argument (target: ≥ 16 bits, max over all 2^15 slots, inputs in [−1, 1])

Let ε_j be the error of EvalMod in slot j in x units. ε_j is an error in coefficient j of the
output, and the output coefficient is q_0·x_j. Decoding maps iid coefficient errors to slot
errors with real-part rms sqrt(N/2)·q_0·σ_ε, and the maximum over 2^15 slots is about 4.2
times the rms. Hence

  precision ≈ −log2( 4.2 · sqrt(N/2) · (q_0/Δ) · σ_ε )   = −25.6 − log2 σ_ε   at q_0/Δ = 2^16.

The ratio q_0/Δ multiplies every EvalMod error, and each factor of 2 costs one bit. σ_ε has
four parts:

1. **Input noise (dominant).** CtS hands EvalMod y = x/K at scale 2^58 with absolute
   slot-domain noise of one rescale plus one key switch. For a dense key that is
   sqrt(N/2)·sqrt((h+1)/12) = 2^13.2 for the rescale and 2^14.2 for the ModDown. The second
   number is measured: it is the n16_train `encrypt` row of `docs/precision.md`, 25.72 bits
   max, so rms 2^16.28/4.2. Together that is 2^14.5. In x units this is multiplied by K:
   σ_in ≈ K · 2^14.5 / 2^58. This term limits every design.
2. **Internal noise.** Each EvalMod level adds about one rescale-plus-key-switch of noise in
   sine units, amplified by the double angles. Relinearization noise is added at scale 2^116
   and divided by q_l, so it does not count. Divided by 2π, this gives x units.
3. **Approximation error.** The §4 table divided by 2π. It is counted as if it were the rms,
   which is pessimistic.
4. **Sine versus modular reduction.** x − sin(2πx)/(2π) ≈ (2π)²x³/6 on the message part
   x = Δc/q_0. With coefficient rms 2^-8.8 this term alone allows 44.1 bits at
   q0/Δ = 2^16, 36.1 at 2^12, 32.1 at 2^10 and 28.1 at 2^8. It is never binding here. A large
   q0/Δ buys sine accuracy (and input-range headroom) that the noise budget cannot use.

Predicted precision (bits) with EvalMod scale 2^58. The input, internal and approximation
columns are what each term alone would allow:

| design | q0/Δ = 2^16 | 2^12 | 2^10 | 2^8 | binding term at 2^16 (input / internal / approx) |
|---|---|---|---|---|---|
| dense, plan K=25 (fails anyway, §3) | 13.4 | 17.4 | 19.4 | 21.4 | 13.4 / 19.2 / 21.7 |
| dense, no SSE, K=420 (12 levels) | ≤ 9.4 | — | — | ≤ 17.4 | input |
| dense + SSE h'=32, K=13, r=0, d=255 | 14.4 | 18.4 | 20.4 | 22.4 | 14.4 / 19.2 / 22.4 |
| **dense + SSE h'=32, K=13, r=1, d=127 (chosen)** | **14.4** | **18.4** | **20.4** | **22.4** | 14.4 / 17.7 / 20.9 |
| dense + SSE h'=32, K=13, r=2, d=63 | 14.3 | 18.3 | 20.3 | 22.3 | 14.4 / 15.8 / 20.7 |
| sparse h=192, K=32, r=0, d=255 | 16.8 | 20.8 | 22.8 | 24.8 | 16.8 / 22.9 / 19.8 |
| sparse h=192 + SSE h'=32, K=13, r=1 | 18.0 | 22.0 | 24.0 | 26.0 | 18.1 / 21.4 / 20.9 |

With 60-bit EvalMod primes (log PQ 1736), the chosen design at q0/Δ = 2^16 gives 16.4 bits.
That is within the model's error of the gate.

Model uncertainty is about ±1–2 bits. The noise constants are n16_train measurements. The CtS
output noise depends on how Task 37 orders the rotations: noise from baby-step rotations is
scaled by the diagonals, and giant-step noise is divided by q_l at the rescale. One n16_boot
specific risk: the two all-58-bit digits have Q_d/P ≈ 1. A key switch at levels ≥ 20 then
carries a key-error term of about (Q_d/P)·3.2·sqrt(N/12) ≈ 236 per such digit, against a ModDown
rounding of about 52. That is about 2^15.9 in slot rms if it is added at the working scale. P
= 4×60 bits (log PQ +8) would remove it.

**Conclusion.** At q0/Δ = 2^16, every design with a dense main key falls short of 16 bits,
even with SSE (14.4). The h = 192 main key reaches 16.8, but its security at log PQ ≈ 1720 is
doubtful (§8).

## 7. Options for the controller (recommendation first)

1. **Recommended: dense main key + SSE (h' = 32) + q0/Δ ≤ 2^12.** Bossuat, Troncoso-Pastoriza
   and Hubaux, ACNS 2022, describe SSE. At level 0, key-switch s → s' with a key modulo q_0·P.
   Mod-raise under s'. At the top level, switch s' → s. CtS, EvalMod and StC then run under
   the dense key. K = 13 fits 8 levels with the committed constants, and the bands stay as
   planned (levels_after_boot 16). Predicted precision is 18.4 bits at 2^12 and 20.4 at 2^10.
   There are two ways to lower q0/Δ:
   (a) q_0 = 54 or 52 bits. log PQ becomes 1716 or 1714, `scale_at(0)` stays 2^42, and the
   Step 1 log PQ test moves to 1716 or 1714 ± 2.
   (b) Bootstrap input at scale 2^46–2^48 at level 0. The last compute rescale targets that
   scale through `mul_scalar(…, target_scale)`. This changes Task 38's `ScaleMismatch`
   contract.
   Either way, the bootstrappable input range narrows: |m| ≲ 2^3 keeps the sine term above 26
   bits at 2^10. SSE needs an extra key pair and probably a `BootParams` field such as
   `ephemeral_hw: Option<usize>`. Neither is part of this task's interface.
2. SSE + 60-bit EvalMod primes, q0/Δ kept at 2^16 (log PQ 1736): 16.4 predicted. Too thin.
3. Sparse main key h = 192 (K = 32, degree 255, r = 0): 16.8 predicted at 2^16. This is
   marginal, and the h = 192 secret at log PQ 1720 is likely below 128 bits (§8).
4. Dense key without SSE: infeasible (12 EvalMod levels and ≤ 9.4 bits).

## 8. Security expectation (Task 39 decides)

n16_boot has log PQ ≈ 1720 with the dense default secret. n16_train (log PQ 1592) estimated
143.9 bits under BDGL16 and 141.2 under MATZOV (`docs/security.md`). Scaling with n/log q at
fixed n, I expect about **133 bits (BDGL16) and 131 (MATZOV)** for n16_boot: above 128 with a
thin margin. Core-SVP would be about 100. **Task 39's lattice-estimator run decides.** Task 39
also needs:
- the h = 192 secret at log PQ 1720 (if option 3). The rough model gave 107 bits at 1592, so I
  expect this to fail;
- the SSE switching key with the h' = 32 secret modulo q_0·P (about 290 bits, if option 1).
  This is far below the main modulus, but a sparse-secret hybrid estimate is still required;
- any q_0, EvalMod-prime or P change from §7: log PQ 1714–1736.
