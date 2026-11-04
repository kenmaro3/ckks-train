# Bootstrap level design for n16_boot (M3, Task 35)

This document fixes the modulus chain of `CkksParams::n16_boot()`, the per-level default
scale `scale_at`, the level budget of mod-raise, CoeffToSlot, EvalMod and SlotToCoeff, and the
EvalMod constants of `BootParams::default_n16()`. It also checks that the design reaches the
16-bit precision target. Tasks 36–39 build on it.

Every number below comes from the reproducibility script `tools/bootstrap/evalmod_design.py`
(standard library and numpy only, under a second: `python3 tools/bootstrap/evalmod_design.py`).

> **Decision (controller ruling R-M3-5).** The main key is dense. Sparse-secret encapsulation
> (SSE: Bossuat, Troncoso-Pastoriza and Hubaux, ACNS 2022) switches the ciphertext to an
> ephemeral key with h' = 32 for the mod-raise. q_0 = 54 bits, so q0/Δ = 2^12, and the
> bootstrap input stays at scale 2^42. P = 4×60 bits. The bands stay at CtS 4 / EvalMod 8 /
> StC 3, so levels_after_boot = 16, and log PQ = 1724. EvalMod uses K = 13, Chebyshev degree
> 127 and r = 1. The predicted precision is **18.4 bits** (§6). If Task 38 measures below 16
> bits, the fallback is q_0 = 52 (q0/Δ = 2^10, predicted 20.4 bits, log PQ 1722).

The plan's first numbers did not survive the calculation; the full analysis is kept in §5–§7.
At the plan's q0/Δ = 2^16, no design with a dense key reaches 16 bits. The plan's K values
were also wrong: a dense key needs K ≈ 400–420, not 25, and h = 192 needs K = 31–33, not 12.
K = 12–13 is the value for h = 32.

## 1. Modulus chain and scales

Level i means a ciphertext over primes q_0..q_i. Bands from the bottom:

| levels | primes | bits | band | `scale_at(level)` |
|---|---|---|---|---|
| 0 | q_0 | 54 | base | 2^42 (preset scale) |
| 1–16 | q_1..q_16 | 42 | compute | 2^42 |
| 17–19 | q_17..q_19 | 42 | SlotToCoeff (StC) | 2^42 |
| 20–27 | q_20..q_27 | 58 | EvalMod | 2^58 |
| 28–31 | q_28..q_31 | 42 | CoeffToSlot (CtS) | 2^42 |
| P | p_0..p_3 | 60 | special | — |

`q_bits = [54, 42×16, 42×3, 58×8, 42×4]`, `p_bits = [60×4]`, dnum 8 (4 primes per digit),
max_level 31. log PQ = 54 + 23·42 + 8·58 + 4·60 = 1724, within the plan ceiling of 1750 (the
exact value is 1724.0 minus about 1e-9). The largest digits are the two all-58-bit EvalMod
digits, at 232 bits, so they sit 8 bits below log P = 240 (`n16_boot_digits_below_p_exactly`).
That margin is why P is 60-bit (§6). At level 0, compute values must satisfy
|m|·2^42 < q_0/2, that is |m| < 2^11.

`scale_at(level)` is 2^(bits of q_level) for level ≥ 1, and the preset scale (2^42) at level 0.
It is the scale a ciphertext at that level should carry when it is multiplied by another
ciphertext and rescaled by q_level. In CtS and StC the ciphertext is multiplied only by
plaintext diagonals. A diagonal encoded at exactly q_l keeps the ciphertext scale, and one
encoded at q_l·σ multiplies the scale by σ. For the single-size presets (seal_2020,
n16_train), `scale_at` equals `scale()` at every level.

`levels_after_boot(params, bp) = max_level − (ctos + evalmod + stoc) = 31 − 15 = 16`: after a
bootstrap the ciphertext is at level 16, with 16 compute levels plus the base.

## 2. Pipeline: levels and scales

The input is a ciphertext at level 0 with scale Δ = 2^42; Task 38 checks both conditions. Its
plaintext coefficients are Δ·c_j, where c = encode(m). For m uniform in [-1, 1] in all 2^15
slots, the coefficient rms is 1/sqrt(3N) ≈ 2^-8.8.

| step | levels | primes consumed | scale in / out | content after the step |
|---|---|---|---|---|
| SSE s → s' | 0 | — | Δ / Δ | the same message under the ephemeral key s' (h' = 32); key modulo q_0·P |
| ModRaise | 0 → 31 | — | Δ | t_j = Δ c_j + q_0 I_j (I_j integer, small because h' = 32, §3); read as x = t/q_0 at scale q_0 ≈ 2^54 |
| SSE s' → s | 31 | — | 2^54 / 2^54 | back under the dense main key (standard key, modulus Q·P) |
| CoeffToSlot | 31 → 27 | q_31..q_28 (42-bit) | 2^54 / 2^58 | two ciphertexts (real halves of the N coefficients) with slots y_j = x_j / K in [-1, 1]. The 1/K and the conjugate-split 1/2 are folded into the diagonals (no level). One diagonal is encoded at q_l·2^58/q_0 to raise the scale to 2^58 |
| EvalMod | 27 → 19 | q_27..q_20 (58-bit) | 2^58 / 2^58 | sin(2π x_j) = cos(2π(x_j − 1/4)): Chebyshev of degree d (⌈log2(d+1)⌉ levels), then r double angles (r levels) |
| SlotToCoeff | 19 → 16 | q_19..q_17 (42-bit) | 2^42 / 2^42 | coefficients q_0 sin(2π x_j)/(2π) ≈ Δ c_j: m at scale Δ, level 16 |

Notes:
- At level 27 the ciphertext enters EvalMod at scale 2^58, which matches `scale_at(27)`.
  Ciphertext-by-ciphertext products are rescaled by 58-bit primes of the same size.
- EvalMod outputs v ≈ 2πΔc/q_0 at scale 2^58. As a plaintext this equals 2πc at scale
  Δ·2^58/q_0 ≈ 2^46, and relabelling the scale costs nothing. The 1/(2π) and the scale step
  from 2^46 down to Δ are folded into the StC diagonals; the last one is encoded at
  q_l·Δ/(current scale). StC therefore ends at exactly the preset compute scale, which is
  Task 38's `ScaleMismatch` contract.
- No arcsine correction is used. The uncorrected sine-versus-modular-reduction error at
  q0/Δ = 2^12 is 36.1 bits for |m| ≤ 1 and 27.1 bits for |m| ≤ 8, so it never binds (§6).

## 3. The overflow I and the interval K

After mod-raise, I_j = round(b_j/q_0 + Σ_{s_k ≠ 0} ±a_k/q_0), where b and a are uniform mod
q_0 for any ciphertext that went through computation. I_j is therefore the rounded sum of h+1
iid U(−1/2, 1/2) variables, where h is the Hamming weight of the key the mod-raise runs under
(`BootParams::mod_raise_hw`). Its standard deviation is σ_I = sqrt((h+1)/12).

The EvalMod polynomial is accurate on x ∈ [−K, K] (`k_intervals = K`). This covers
|I_j| ≤ K − 1 with room for the tiny message part. A bootstrap fails if some coefficient has
|I_j| ≥ K, and the union bound runs over all N = 2^16 coefficients. Target:
**≤ 2^-30 per bootstrap**. A training run does at most about 2^14 bootstraps.

| mod-raise key | h | σ_I | log2 P_fail per bootstrap |
|---|---|---|---|
| dense (hekit default: −1/0/1 with probability 1/4, 1/2, 1/4) | ≈ N/2 = 32768 | 52.26 | K=25: +15.4 (certain failure); K=400: −29.4; K=420: −33.8 |
| sparse | 192 | 4.01 | K=12: +8.1; K=25: −14.4; K=30: −27.4; K=32: −33.3; K=33: −36.4 |
| **SSE ephemeral (chosen)** | 32 | 1.66 | K=12: −29.1; **K=13: −39.7**; K=14: −53.4 |

The dense rows use the Gaussian tail. The h = 192 and h = 32 rows use the exact Irwin–Hall
tail, for which the Gaussian is slightly pessimistic. A positive value means the expected
number of failing coefficients is above 1. At K = 25 a dense key overflows in 64% of the
coefficients.

`BootParams::validate` applies a coarse guard, K ≥ 7σ_I + 1/2 (`min_k_intervals`): 13 for
h' = 32, 29 for h = 192 and 367 for the dense key. A 7σ tail is about 2^-22.5 per bootstrap,
so the guard is necessary but not sufficient. The table above sets the actual K. The guard
does reject the SSE constants when no ephemeral key is set.

## 4. EvalMod: Chebyshev approximation plus double angles

We approximate f_r(x) = cos(2π(x − 1/4)/2^r) on x ∈ [−K, K], with y = x/K ∈ [−1, 1] folded
into CtS, by Chebyshev interpolation of degree d. Then r double-angle steps c ← 2c² − 1 give
cos(2π(x − 1/4)) = sin(2πx).

**Depth.** The baby-step giant-step evaluation in the Chebyshev basis (Bossuat et al.,
Eurocrypt 2021) uses ⌈log2(d + 1)⌉ levels, and each double angle uses one more:

  evalmod_depth = ⌈log2(cheb_degree + 1)⌉ + double_angle ≤ evalmod_levels.

`BootParams::validate` enforces this, and `evalmod_depth()` computes it.

**Error.** The Chebyshev coefficients of cos(ω y + φ) are Bessel values J_n(ω) with
ω = 2πK/2^r. They die off only once n > ω, so the degree needed is a bit above 2πK/2^r. Each
double angle halves ω but costs a level. It also amplifies an error δ by 4|cos θ|, since
2(c+δ)² − 1 = cos 2θ + 4δ cos θ + 2δ². For the angles that occur, the total amplification is
at most `amp` = max_I Π_i 4|cos(2^i θ_I)|.

The table gives the measured maximum sine error at the points x = I + δ that occur
(|I| ≤ K − 1, |δ| ≤ 2^-20), using all 8 EvalMod levels (d = 2^(8−r) − 1). Values near 2^-45
are the float64 floor of the measurement.

| K | r=0, d=255 | r=1, d=127 | r=2, d=63 | r=3, d=31 |
|---|---|---|---|---|
| 12 | 2^-45.5 | 2^-44.2 (amp 2.8) | 2^-43.5 (amp 10.5) | 2^-40.7 (amp 41) |
| **13** | 2^-45.4 | **2^-43.8** | 2^-43.7 | 2^-37.2 |
| 16 | 2^-46.0 | 2^-43.6 | 2^-43.8 | 2^-28.1 |
| 25 | 2^-44.6 | 2^-43.0 | 2^-25.5 | 2^-10.0 |
| 30 | 2^-44.5 | 2^-28.8 | 2^-12.8 | 2^-3.6 |
| 32 | 2^-42.7 | 2^-20.9 | 2^-8.7 | 2^-1.4 |
| 34 | 2^-29.5 | 2^-14.1 | 2^-5.3 | 2^0.6 |

8 levels carry K ≤ 32 at an error of about 2^-42 or better, but only with r = 0 (degree 255).
A mod-raise under the dense key itself (K = 400 or 420, r = 0, degree 2^L − 1) still has
error around 1 at L = 9, 10 and 11, and reaches 2^-39.8 only at L = 12. Raising r does not
reduce the depth: ⌈log2(d+1)⌉ + r stays near log2(2πK·1.2) for any r, and each extra r only
adds amplification.

**Chosen constants** (`N16_EPHEMERAL_HW` = 32, `N16_K_INTERVALS`, `N16_CHEB_DEGREE`,
`N16_DOUBLE_ANGLE`): **K = 13, degree 127, r = 1, so the depth is ⌈log2 128⌉ + 1 = 7 + 1 =
8 = `evalmod_levels`.** These constants assume SSE with h' = 32. The sine error is 2^-43.8,
including the amplification of 2.8, and the failure rate is 2^-39.7. Why r = 1 rather than
the alternatives:
- r = 0 (degree 255) and r = 2 (degree 63) predict the same precision to within 0.1 bit (§6).
- r = 1 needs about 2·sqrt(128) ≈ 23 ciphertext products, against about 32 for degree 255.
- Its noise amplification (2.8) stays far from binding. With r = 2 (amplification 10.5) the
  internal-noise term comes within 1.4 bits of the input term at q0/Δ = 2^16.

**Alternative, sparse h = 192 main key without SSE:** K = 32 (failure 2^-33.3), degree 255,
r = 0, depth 8, sine error 2^-42.7. K = 33 does not fit 8 levels at that accuracy.

## 5. Why the plan's dense default could not be fixed by rebalancing the bands

Without SSE the dense key needs K ≈ 420, and so 12 EvalMod levels. Taking 4 levels from the
compute band (16 → 12, the floor) and making them 58-bit EvalMod primes adds 4·16 bits to log
PQ, which then exceeds the 1750 ceiling. Keeping log PQ ≤ 1750 with 12 compute levels forces
31 or 30 q primes. With 31 primes, dnum must divide a prime. With 30 primes, dnum 10 and k = 3
leave only 5 levels for CtS and StC combined. Even a chain that fit would fail precision:
with K = 420, the input-noise term alone gives at most 9.4 bits at q0/Δ = 2^16 and 17.4 bits
at 2^8. Precision binds, not depth. That is why the ruling uses SSE and a smaller q0/Δ
instead of rebalancing.

## 6. Precision argument (target: ≥ 16 bits, max over all 2^15 slots, inputs in [−1, 1])

Let ε_j be the EvalMod error in slot j, in x units. It becomes the error in coefficient j of
the output, which is q_0·x_j. Decoding maps iid coefficient errors to slot errors with
real-part rms sqrt(N/2)·q_0·σ_ε, and the maximum over 2^15 slots is about 4.2 times the rms.
Hence

  precision ≈ −log2( 4.2 · sqrt(N/2) · (q_0/Δ) · σ_ε )   = −21.6 − log2 σ_ε   at q_0/Δ = 2^12.

The ratio q_0/Δ multiplies every EvalMod error, and each factor of 2 costs one bit. σ_ε has
four terms:

1. **Input noise, the dominant term.** CtS hands EvalMod y = x/K at scale 2^58. Its absolute
   slot-domain noise is one rescale plus one key switch. Under the dense key that is
   sqrt(N/2)·sqrt((h+1)/12) = 2^13.2 for the rescale and 2^14.2 for the ModDown, 2^14.5
   combined. The ModDown value is measured: the n16_train `encrypt` row of
   `docs/precision.md` reaches 25.72 bits max, so rms 2^16.28/4.2. In x units the noise is
   multiplied by K, so σ_in ≈ K · 2^14.5 / 2^58. SSE matters for precision as well as depth
   because it makes K small.
2. **Internal noise.** Each EvalMod level adds about one rescale plus one key switch of noise
   in sine units, and the double angles amplify it. Relinearization noise is added at scale
   2^116 and divided by q_l, so it contributes nothing. Dividing by 2π converts to x units.
3. **Approximation error.** The §4 table divided by 2π. It is counted as an rms value, which
   is pessimistic.
4. **Sine versus modular reduction**, uncorrected. On the message part x = Δc/q_0,
   x − sin(2πx)/(2π) ≈ (2π)²x³/6. This term alone gives 44.1 bits at q0/Δ = 2^16, 36.1 at
   2^12, 32.1 at 2^10 and 28.1 at 2^8. At 2^12 it is 27.1 bits for |m| ≤ 8.

A fifth term is specific to n16_boot. The key error of a key switch,
Σ_d [c]_{Q_d}·e_d/P, has coefficient std (Q_d/P)·3.2·sqrt(N/12) ≈ 236·(Q_d/P) per digit. The
two all-58-bit digits have Q_d = 2^232. With P = 4×58, Q_d/P ≈ 1 and the term is about 2^15.9
slot rms. Counted at the CtS output scale, which is the worst case, that costs 1.6 bits. With
the chosen P = 4×60, Q_d/P = 2^-8 and the term is 2^7.9, which is negligible.

**Chosen design** (dense + SSE h' = 32, K = 13, r = 1, d = 127, q0/Δ = 2^12):

| variant | predicted bits |
|---|---|
| **P = 4×60 (chosen)** | **18.4** |
| P = 4×58, digit term at its worst case | 16.8 |
| fallback q_0 = 52 (q0/Δ = 2^10), P = 4×60 | 20.4 |

**Alternatives** (EvalMod scale 2^58, digit term not included). The last column gives what
each term alone would allow at q0/Δ = 2^16:

| design | 2^16 | 2^12 | 2^10 | 2^8 | input / internal / approx at 2^16 |
|---|---|---|---|---|---|
| dense, plan K=25 (fails anyway, §3) | 13.4 | 17.4 | 19.4 | 21.4 | 13.4 / 19.2 / 21.7 |
| dense, no SSE, K=420 (12 levels) | ≤ 9.4 | — | — | ≤ 17.4 | input |
| dense + SSE h'=32, K=13, r=0, d=255 | 14.4 | 18.4 | 20.4 | 22.4 | 14.4 / 19.2 / 22.4 |
| **dense + SSE h'=32, K=13, r=1, d=127 (chosen)** | 14.4 | **18.4** | 20.4 | 22.4 | 14.4 / 17.7 / 20.9 |
| dense + SSE h'=32, K=13, r=2, d=63 | 14.3 | 18.3 | 20.3 | 22.3 | 14.4 / 15.8 / 20.7 |
| sparse h=192, K=32, r=0, d=255 | 16.8 | 20.8 | 22.8 | 24.8 | 16.8 / 22.9 / 19.8 |
| sparse h=192 + SSE h'=32, K=13, r=1 | 18.0 | 22.0 | 24.0 | 26.0 | 18.1 / 21.4 / 20.9 |

One more variant: SSE with 60-bit EvalMod primes at q0/Δ = 2^16 gives 16.4 bits.

The model is good to about ±1–2 bits. Its noise constants are n16_train measurements, and the
CtS output noise depends on how Task 37 orders the rotations: baby-step noise is scaled by the
diagonals, and giant-step noise is divided by q_l at the rescale. The 18.4-bit prediction
leaves 2.4 bits of margin. If Task 38 measures below 16 bits, q_0 = 52 adds 2 bits.

## 7. Options considered (the ruling took option 1)

1. **Chosen:** dense main key + SSE (h' = 32) + q0/Δ = 2^12 (q_0 = 54) + P = 4×60. SSE works
   in three steps. At level 0, key-switch s → s' with a key modulo q_0·P. Mod-raise under s'.
   At the top level, switch s' → s with a standard key modulo Q·P. CtS, EvalMod and StC then
   run under the dense key. The `ephemeral_hw` field records h'.
2. SSE with 60-bit EvalMod primes and q0/Δ kept at 2^16: 16.4 bits predicted, too thin.
3. Sparse main key h = 192 (K = 32, degree 255, r = 0): 16.8 bits at 2^16. Its security at
   log PQ ≈ 1724 is doubtful.
4. Dense key without SSE: infeasible (12 EvalMod levels and at most 9.4 bits).

## 8. Security expectation and what Task 39 needs

n16_train (log PQ 1592) estimated at 143.9 bits under BDGL16 and 141.2 under MATZOV
(`docs/security.md`). Scaling by n/log q at fixed n, I expect n16_boot (dense secret, log PQ
1724) to land at about **133 bits (BDGL16) and 130 (MATZOV)**. That is above 128 with a thin
margin; Core-SVP would give about 99. The lattice-estimator run of Task 39 decides.

**Task 39 needs:**
- (a) the dense n16_boot at log PQ 1724, and at 1722 if the q_0 = 52 fallback is used;
- (b) the SSE switching key s → s': N = 2^16, ternary secret with h = 32, modulus q_0·P with
  log(q_0·P) = 54 + 240 = 294. This is a sparse-secret instance, so the hybrid and MITM
  attacks have to be enabled. The s' → s key uses the dense main key at modulus Q·P and is
  covered by (a);
- (c) h = 192 at log PQ 1724, as an alternative only. The rough model in docs/security.md gave
  107 bits at 1592, so I expect it to fall below 128.
