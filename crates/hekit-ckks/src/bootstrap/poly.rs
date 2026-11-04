//! Chebyshev interpolation and the **level-optimal, scale-invariant** evaluation of a
//! Chebyshev series on a ciphertext (Task 37), the evaluation of Bossuat, Mouchet,
//! Troncoso-Pastoriza and Hubaux, "Efficient Bootstrapping for Approximate Homomorphic
//! Encryption with Non-Sparse Keys" (Eurocrypt 2021, §4–5, Alg. 2). The recursion and the
//! scale bookkeeping follow the algorithm of that paper, and [`optimal_split`] follows
//! Lattigo's baby-step split rule (a cost formula). This code was written from the
//! algorithm: no Lattigo source was read or copied for it (Task 37 fix round 1).
//! `docs/bootstrap.md` §10 has the level and scale analysis.
//!
//! # What "level-optimal" means here
//!
//! A polynomial `p(y) = sum_i c_i T_i(y)` of degree `d` on a ciphertext at level `L` costs
//! exactly `ceil(log2(d + 1))` levels ([`chebyshev_depth`]): 7 for degree 127. Textbook
//! Paterson–Stockmeyer, or a baby-step giant-step that rescales every `c_i T_i`, costs one
//! level more. Two things remove that level:
//!
//! * **The scalar products are never rescaled on their own.** `c_i T_i` is an integer
//!   multiple `k_i = round(c_i * ts / S_i)` of `T_i` (`S_i` its scale), labelled exactly `ts`;
//!   with `ts ~ S * q` the constant keeps about `log2 q` bits and costs no level. The one
//!   rescale comes after the product with the giant step.
//! * **Every branch gets its target level and scale from its parent** (scale-invariant
//!   evaluation): `rec(p, l, ts)` returns `p(y)` at level `l` with the exact scale `ts`,
//!   **before** its rescale. For `p = q T_k + r` it asks for `q` at `(l + 1, ts q_{l+1} / S_k)`,
//!   rescales it (to `l`, scale `ts / S_k`), multiplies by `T_k` (scale `ts`) and adds `r`,
//!   asked for at `(l, ts)` directly. The final rescale lands exactly on the requested
//!   output scale. No branch rounds its scale to "the nominal one", so the power basis can
//!   have any scales.
//!
//! # Levels
//!
//! `T_i` is computed as `2 T_a T_b - T_{a-b}` with `a = 2^(ceil(log2 i) - 1)`, `b = i - a`,
//! at level `L - ceil(log2 i)` (the product is taken at the lower operand level, the
//! subtraction before the rescale with `T_{a-b}` times an integer constant, so it is exact
//! in scale). Powers are computed on demand, so an odd polynomial (EvalMod's sine) only
//! builds the odd `T_i` and the powers of two.
//!
//! `rec(p, l, ts)` with `m = deg p` is
//!
//! * a **leaf** `sum_i k_i T_i + round(c_0 ts)` when `m < 2^s` (the baby steps,
//!   `s =` [`optimal_split`]`(ceil(log2(d + 1)))`) and `T_m` lies at level `>= l`, that is
//!   `l + ceil(log2 m) <= L`;
//! * otherwise a **split** at `k = 2^j`, the largest power of two `<= m`:
//!   `p = q T_k + r` ([`T_i = 2 T_{i-k} T_k - T_{2k-i}`] for `k < i < 2k`).
//!
//! The top call is `rec(p, L - D + 1, label)` with `D = ceil(log2(d + 1))`, and the result is
//! rescaled once more, to level `L - D`. Feasibility by induction: `rec(p, l)` is feasible when
//! `l <= L - ceil(log2(m + 1)) + 1`. A split at `k = 2^j` has `ceil(log2(m + 1)) = j + 1`, so
//! `l <= L - j`: `T_k` (level `L - j`) is available at `l`, `q` (degree `< 2^j`) is asked
//! for at `l + 1 <= L - j + 1` and `r` (degree `< 2^j`) at `l`, both within their bounds.
//! The leaf condition holds at `m = 1` for every `l <= L`, so the recursion ends. Along the
//! leading branch (the top coefficients) the bound is tight and the recursion splits down to
//! degree 1; elsewhere there is a level of slack and the leaves have degree up to `2^s - 1`.

use std::collections::BTreeMap;
use std::f64::consts::PI;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;

use crate::ciphertext::Ciphertext;
use crate::evaluator::{Evaluator, MIN_PLAIN_SCALE};

/// Largest magnitude of an integer constant ([`mul_const_to`], [`add_const`]): `2^126`,
/// so that it fits an `i128`.
const MAX_CONST: f64 = 85_070_591_730_234_615_865_843_651_857_942_052_864.0;

/// Largest relative difference between a computed product scale and the planned one that
/// is relabelled (a few f64 roundings give about `2^-50`).
const RELABEL_TOL: f64 = 1e-12;

/// Chebyshev interpolation coefficients `c_0..=c_degree` of `f` on `[a, b]`:
/// `p(x) = sum_j c_j T_j(y)` with `y = (2x - a - b) / (b - a)`, interpolating `f` at the
/// `degree + 1` Chebyshev nodes of the first kind `y_k = cos(pi (k + 1/2) / (degree + 1))`.
///
/// `c_j = (2 - [j = 0]) / (d + 1) * sum_k f(x_k) cos(j pi (k + 1/2) / (d + 1))` (the discrete
/// orthogonality of the `T_j` on the nodes), in f64 with the cosines evaluated directly. For
/// smooth `f` the interpolation error is within a factor of about 2 of the best uniform
/// approximation of that degree.
///
/// # Panics
/// If `a < b` does not hold for finite `a`, `b`.
pub fn chebyshev_coeffs(f: impl Fn(f64) -> f64, a: f64, b: f64, degree: usize) -> Vec<f64> {
    assert!(
        a.is_finite() && b.is_finite() && a < b,
        "interval [{a}, {b}] must be finite with a < b"
    );
    let n = degree + 1;
    let theta: Vec<f64> = (0..n).map(|k| PI * (k as f64 + 0.5) / n as f64).collect();
    let fx: Vec<f64> = theta
        .iter()
        .map(|&t| f(0.5 * (b - a) * t.cos() + 0.5 * (a + b)))
        .collect();
    (0..n)
        .map(|j| {
            let s: f64 = theta
                .iter()
                .zip(&fx)
                .map(|(&t, &v)| v * (j as f64 * t).cos())
                .sum();
            let c = 2.0 * s / n as f64;
            if j == 0 { c / 2.0 } else { c }
        })
        .collect()
}

/// `sum_j coeffs[j] T_j(y)` at `y = (2x - a - b) / (b - a)` (Clenshaw's recurrence): the
/// plaintext reference of [`Evaluator::eval_chebyshev`].
///
/// # Panics
/// If `coeffs` is empty.
pub fn chebyshev_eval(coeffs: &[f64], a: f64, b: f64, x: f64) -> f64 {
    assert!(!coeffs.is_empty(), "no coefficients");
    let y = (2.0 * x - a - b) / (b - a);
    let (mut b1, mut b2) = (0.0, 0.0);
    for &c in coeffs[1..].iter().rev() {
        let t = 2.0 * y * b1 - b2 + c;
        b2 = b1;
        b1 = t;
    }
    coeffs[0] + y * b1 - b2
}

/// Levels [`Evaluator::eval_chebyshev`] consumes for a degree-`degree` series on `[-1, 1]`:
/// `ceil(log2(degree + 1))` (0 for a constant), the count of
/// [`BootParams::evalmod_depth`](crate::bootstrap::BootParams::evalmod_depth).
pub fn chebyshev_depth(degree: usize) -> usize {
    (degree + 1).next_power_of_two().trailing_zeros() as usize
}

/// The baby-step exponent `s` for a series of depth `log_degree`: the leaves have degree `< 2^s`.
/// This follows Lattigo's split rule (`OptimalSplit`), a cost formula: start at `log_degree / 2`
/// and take one more when `2^s + 2^(D-s) + D - s - 3 > 2^(s+1) + 2^(D-s-1) + D - s - 4` (the
/// estimated product counts). 4 for degree 127 (`D = 7`): leaves up to degree 15.
pub fn optimal_split(log_degree: usize) -> usize {
    let d = log_degree as i64;
    let mut s = d >> 1;
    let a = (1i64 << s) + (1i64 << (d - s)) + d - s - 3;
    let b = (1i64 << (s + 1)) + (1i64 << (d - s - 1).max(0)) + d - s - 4;
    if a > b {
        s += 1;
    }
    s as usize
}

/// `p = q T_k + r` for a Chebyshev series `p` of degree `m` with `k <= m < 2k`:
/// `q_0 = c_k`, `q_{i-k} = 2 c_i` and `r_{2k-i} -= c_i` for `k < i <= m` (from
/// `T_i = 2 T_{i-k} T_k - T_{2k-i}`), `r_i = c_i` for `i < k`.
fn split_coeffs(p: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let m = p.len() - 1;
    debug_assert!(k <= m && m < 2 * k, "split of degree {m} at {k}");
    let mut r = p[..k].to_vec();
    let mut q = vec![0.0; m - k + 1];
    q[0] = p[k];
    for i in k + 1..=m {
        q[i - k] = 2.0 * p[i];
        r[2 * k - i] -= p[i];
    }
    (q, r)
}

/// `ceil(log2(m))` for `m >= 1`.
fn ceil_log2(m: usize) -> usize {
    m.next_power_of_two().trailing_zeros() as usize
}

/// `k mod q` as a residue.
fn residue(k: i128, q: u64) -> u64 {
    k.rem_euclid(q as i128) as u64
}

/// The integer `round(v)` as `i128`, `InvalidParams` beyond `2^126` or for non-finite `v`.
fn to_int(v: f64) -> Result<i128> {
    let r = v.round();
    if !(r.is_finite() && r.abs() < MAX_CONST) {
        return Err(HekitError::InvalidParams(format!(
            "integer constant {r:e} out of range (|k| < 2^126)"
        )));
    }
    Ok(r as i128)
}

/// `poly * k` for an integer `k` (either form).
fn mul_int(poly: &RnsPoly, k: i128) -> RnsPoly {
    let basis = poly.basis();
    let limbs = poly
        .limbs()
        .iter()
        .zip(basis.moduli())
        .map(|(l, m)| {
            let s = m.shoup(residue(k, m.value()));
            l.iter().map(|&x| m.mul_shoup(x, &s)).collect()
        })
        .collect();
    RnsPoly::from_limbs(basis, limbs, poly.is_ntt_form()).expect("reduced limbs")
}

/// `ct * value` as the integer multiple `k = round(value * product_scale / ct.scale())`,
/// labelled exactly `product_scale` (no rescale, no level; the label differs from
/// `ct.scale() * k / value` only by the rounding of `k`, relative `<= 1 / (2 |k|)`). Counts
/// one `ct_pt_mults`.
///
/// Errors (`InvalidParams`): `product_scale / ct.scale()` below `2^20` (the constant would
/// keep fewer than 20 bits) or `|k| >= 2^126`.
fn mul_const_to(
    eval: &Evaluator,
    ct: &Ciphertext,
    value: f64,
    product_scale: f64,
) -> Result<Ciphertext> {
    let ratio = product_scale / ct.scale();
    if ratio.is_nan() || ratio < MIN_PLAIN_SCALE {
        return Err(HekitError::InvalidParams(format!(
            "constant scale {ratio:e} below 2^20 (scale {:e} -> {product_scale:e})",
            ct.scale()
        )));
    }
    let k = to_int(value * ratio)?;
    let c = ct.polys().iter().map(|p| mul_int(p, k)).collect();
    let r = Ciphertext::new(c, product_scale)?;
    eval.count(|c| c.ct_pt_mults += 1);
    Ok(r)
}

/// `ct + value`: adds the integer `round(value * ct.scale())` to `c_0` (in NTT form a
/// constant polynomial is the same constant at every evaluation point). No noise, no level.
fn add_const(ct: &Ciphertext, value: f64) -> Result<Ciphertext> {
    let k = to_int(value * ct.scale())?;
    let mut c = ct.polys().to_vec();
    let p0 = &c[0];
    let basis = p0.basis();
    let limbs = p0
        .limbs()
        .iter()
        .zip(basis.moduli())
        .map(|(l, m)| {
            let r = residue(k, m.value());
            l.iter().map(|&x| m.add(x, r)).collect()
        })
        .collect();
    c[0] = RnsPoly::from_limbs(basis, limbs, true)?;
    Ciphertext::new(c, ct.scale())
}

/// `a + b` (or `a - b`) for operands of **bit-identical** scales, else `ScaleMismatch`:
/// every addition of the evaluation is exact in scale by construction, and this keeps
/// [`SCALE_TOLERANCE`](crate::evaluator::SCALE_TOLERANCE) from silently relabelling after a
/// future change (review m3).
fn add_exact(eval: &Evaluator, a: &Ciphertext, b: &Ciphertext, sub: bool) -> Result<Ciphertext> {
    if a.scale() != b.scale() {
        return Err(HekitError::ScaleMismatch);
    }
    if sub { eval.sub(a, b) } else { eval.add(a, b) }
}

/// The trivial encryption of zero at `level` with `scale` (2 components).
fn zero_ct(eval: &Evaluator, level: usize, scale: f64) -> Result<Ciphertext> {
    let mut z = RnsPoly::zero(eval.params().basis_q(), level);
    z.to_ntt();
    Ciphertext::new(vec![z.clone(), z], scale)
}

/// `ct` relabelled to `scale`, which must be within [`RELABEL_TOL`] of its scale.
fn relabel(ct: Ciphertext, scale: f64) -> Result<Ciphertext> {
    if ((ct.scale() - scale) / scale).abs() > RELABEL_TOL {
        return Err(HekitError::ScaleMismatch);
    }
    Ciphertext::new(ct.into_polys(), scale)
}

/// The state of one evaluation: the evaluator, the level `L` of `T_1`, the baby-step bound
/// `2^s` and the power basis computed so far.
struct Cheb<'e> {
    eval: &'e Evaluator,
    top: usize,
    baby: usize,
    powers: BTreeMap<usize, Ciphertext>,
}

impl Cheb<'_> {
    /// Computes `T_i` (and what it needs) if absent: `T_i = 2 T_a T_b - T_{a-b}` at level
    /// `L - ceil(log2 i)` (module docs). The product is relinearized and doubled (added to
    /// itself: relabelling it with half its scale would double the message for free but halve
    /// the scale of every later power); `T_{a-b}` is subtracted as an integer multiple
    /// labelled with the product scale (`T_0 = 1` as an integer constant); then one rescale,
    /// so `T_i` has scale `S_a S_b / q_l`.
    fn ensure(&mut self, i: usize) -> Result<()> {
        if self.powers.contains_key(&i) {
            return Ok(());
        }
        debug_assert!(i >= 2, "T_1 is the input");
        let a = 1usize << (ceil_log2(i) - 1);
        let (b, c) = (i - a, 2 * a - i);
        self.ensure(a)?;
        self.ensure(b)?;
        if c > 0 {
            self.ensure(c)?;
        }
        let ev = self.eval;
        let (ta, tb) = (&self.powers[&a], &self.powers[&b]);
        let l = ta.level().min(tb.level());
        let ta = ev.mod_switch_to_level(ta, l)?;
        let tb = ev.mod_switch_to_level(tb, l)?;
        let prod = ev.relinearize(&ev.mul_no_relin(&ta, &tb)?)?;
        let prod = add_exact(ev, &prod, &prod, false)?;
        let prod = if c == 0 {
            add_const(&prod, -1.0)?
        } else {
            let tc = ev.mod_switch_to_level(&self.powers[&c], l)?;
            add_exact(ev, &prod, &mul_const_to(ev, &tc, 1.0, prod.scale())?, true)?
        };
        let t = ev.rescale(&prod)?;
        debug_assert_eq!(t.level() + ceil_log2(i), self.top);
        self.powers.insert(i, t);
        Ok(())
    }

    /// `sum_i p_i T_i` at level `tl` with the exact scale `ts`, unrescaled (module docs);
    /// `None` for the zero polynomial.
    fn rec(&mut self, p: &[f64], tl: usize, ts: f64) -> Result<Option<Ciphertext>> {
        let Some(m) = p.iter().rposition(|&c| c != 0.0) else {
            return Ok(None);
        };
        let p = &p[..=m];
        if m == 0 || (m < self.baby && tl + ceil_log2(m) <= self.top) {
            return self.leaf(p, tl, ts).map(Some);
        }
        let k = 1usize << (usize::BITS - 1 - m.leading_zeros());
        let (q, r) = split_coeffs(p, k);
        self.ensure(k)?;
        let s_pow = self.powers[&k].scale();
        debug_assert!(self.powers[&k].level() >= tl, "giant step below its level");
        let q_next = self.eval.params().q()[tl + 1] as f64;
        let qres = self.rec(&q, tl + 1, ts * q_next / s_pow)?;
        let rres = self.rec(&r, tl, ts)?;
        let ev = self.eval;
        let prod = match qres {
            None => None,
            Some(qc) => {
                let qc = if qc.len() == 3 {
                    ev.relinearize(&qc)?
                } else {
                    qc
                };
                let qc = ev.rescale(&qc)?;
                let tp = ev.mod_switch_to_level(&self.powers[&k], tl)?;
                Some(relabel(ev.mul_no_relin(&qc, &tp)?, ts)?)
            }
        };
        Ok(match (prod, rres) {
            (None, r) => r,
            (Some(a), None) => Some(a),
            (Some(a), Some(b)) => Some(add_exact(ev, &a, &b, false)?),
        })
    }

    /// The leaf `sum_{i >= 1} k_i T_i + round(p_0 ts)` at level `tl`, scale `ts`.
    fn leaf(&mut self, p: &[f64], tl: usize, ts: f64) -> Result<Ciphertext> {
        let mut acc: Option<Ciphertext> = None;
        for (i, &c) in p.iter().enumerate().skip(1) {
            if c == 0.0 {
                continue;
            }
            self.ensure(i)?;
            let t = self.eval.mod_switch_to_level(&self.powers[&i], tl)?;
            let term = mul_const_to(self.eval, &t, c, ts)?;
            acc = Some(match acc {
                None => term,
                Some(s) => add_exact(self.eval, &s, &term, false)?,
            });
        }
        let acc = match acc {
            Some(a) => a,
            None => zero_ct(self.eval, tl, ts)?,
        };
        if p[0] != 0.0 {
            add_const(&acc, p[0])
        } else {
            Ok(acc)
        }
    }
}

impl Evaluator {
    /// `sum_j coeffs[j] T_j(y)`, `y = (2x - a - b) / (b - a)`, for the slots `x` of `ct`
    /// ([`chebyshev_coeffs`] gives such coefficients), with the **level-optimal,
    /// scale-invariant** evaluation of the [module docs](self): the result has exactly the
    /// scale of `ct` and level `ct.level() - ceil(log2(degree + 1))`
    /// ([`chebyshev_depth`]; `degree = coeffs.len() - 1`), plus one level when
    /// `[a, b] != [-1, 1]` (the change of variable is a scalar product and a rescale; fold it
    /// into an earlier linear step to save that level). Same as
    /// [`Evaluator::eval_chebyshev_to_scale`] with `target_scale = ct.scale()`.
    pub fn eval_chebyshev(
        &self,
        ct: &Ciphertext,
        coeffs: &[f64],
        a: f64,
        b: f64,
    ) -> Result<Ciphertext> {
        self.eval_chebyshev_to_scale(ct, coeffs, a, b, ct.scale())
    }

    /// [`Evaluator::eval_chebyshev`] with the output at exactly `target_scale`.
    ///
    /// Cost (degree `d`, `D = ceil(log2(d + 1))`, `s =` [`optimal_split`]`(D)`): one
    /// multiplication and relinearization per power `T_i` used (the powers of two up to
    /// `2^(D-1)` and the baby steps below `2^s` with a nonzero coefficient in some leaf;
    /// for an odd series only the odd ones) and one per split; one integer constant
    /// product (counted as `ct_pt_mults`) per nonzero coefficient of every leaf and per
    /// `T_i` with `i` not a power of two. Products that are only added are relinearized once,
    /// after the sum. Degree 127, odd (EvalMod's sine): 13 powers (`T_2, T_4, ..., T_64` and
    /// the odd `T_3 .. T_15`) and 10 splits, so 23 products, 20 relinearizations, 24 rescales
    /// and 71 constant products (`docs/bootstrap.md` §10).
    ///
    /// The unrescaled result at level `out + 1` has scale `~ target_scale * q_{out+1}`, so the
    /// modulus `Q_{out+1}` must hold it (as for any product before its rescale).
    ///
    /// Errors, before any work: `InvalidParams` if `ct` is not a 2-component ciphertext over
    /// this evaluator's basis, there is no valid relinearization key, `coeffs` has fewer than
    /// two entries (degree 0) or a non-finite one, `[a, b]` is not finite with `a < b`, or
    /// `target_scale` is not positive and finite; `NoLevelsLeft` if `ct` is below the depth;
    /// `ScaleMismatch` if no rescale from the last level reaches `target_scale` exactly
    /// ([`Evaluator::exact_product_scale`]). During the evaluation, `InvalidParams` if an
    /// integer constant leaves `2^126` or a constant's scale falls below `2^20` (scales far
    /// from the primes, or huge coefficients).
    pub fn eval_chebyshev_to_scale(
        &self,
        ct: &Ciphertext,
        coeffs: &[f64],
        a: f64,
        b: f64,
        target_scale: f64,
    ) -> Result<Ciphertext> {
        let bad = |m: String| Err(HekitError::InvalidParams(m));
        self.check_galois_input(ct)?;
        self.relin_key_checked()?;
        if coeffs.len() < 2 {
            return bad("eval_chebyshev needs degree >= 1 (two coefficients or more)".into());
        }
        if let Some(c) = coeffs.iter().find(|c| !c.is_finite()) {
            return bad(format!("non-finite Chebyshev coefficient {c}"));
        }
        if !(a.is_finite() && b.is_finite() && a < b) {
            return bad(format!("interval [{a}, {b}] must be finite with a < b"));
        }
        if !(target_scale.is_finite() && target_scale > 0.0) {
            return bad(format!(
                "target scale {target_scale} must be positive and finite"
            ));
        }
        let affine = a != -1.0 || b != 1.0;
        let depth = chebyshev_depth(coeffs.len() - 1);
        if ct.level() < depth + affine as usize {
            return Err(HekitError::NoLevelsLeft);
        }
        let x = if affine {
            // y = alpha x + beta at exactly ct.scale(), one level.
            let (alpha, beta) = (2.0 / (b - a), -(a + b) / (b - a));
            let y = self.rescale(&self.mul_scalar(ct, alpha, ct.scale())?)?;
            add_const(&y, beta)?
        } else {
            ct.clone()
        };
        let top = x.level();
        let tl = top + 1 - depth;
        let label = self.exact_product_scale(tl, target_scale)?;
        let mut cheb = Cheb {
            eval: self,
            top,
            baby: 1 << optimal_split(depth),
            powers: BTreeMap::from([(1, x)]),
        };
        let res = match cheb.rec(coeffs, tl, label)? {
            Some(c) => c,
            None => zero_ct(self, tl, label)?,
        };
        let res = if res.len() == 3 {
            self.relinearize(&res)?
        } else {
            res
        };
        let out = self.rescale(&res)?;
        debug_assert_eq!(out.scale(), target_scale);
        Ok(out)
    }
}

/// The double-angle step `c <- 2 c^2 - 1` at the level of `c`, landing exactly on
/// `out_scale` after its rescale: the square is labelled `sigma = exact label of out_scale`,
/// which must be `c.scale()^2 / 2` up to the few-ulp choice of `c.scale()` made by the
/// caller (a relative relabelling error of a few `2^-52`; more than `2^-40` is
/// `ScaleMismatch`). The relabel doubles the message; then `1` is subtracted at that scale
/// and the result rescaled. One level.
pub(crate) fn double_angle(eval: &Evaluator, c: &Ciphertext, out_scale: f64) -> Result<Ciphertext> {
    let sigma = eval.exact_product_scale(c.level(), out_scale)?;
    let sq = eval.relinearize(&eval.mul_no_relin(c, c)?)?;
    if ((sq.scale() / 2.0 - sigma) / sigma).abs() > 2f64.powi(-40) {
        return Err(HekitError::ScaleMismatch);
    }
    let sq = Ciphertext::new(sq.into_polys(), sigma)?;
    eval.rescale(&add_const(&sq, -1.0)?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use crate::params::CkksParams;
    use hekit_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// Uniform values in `[lo, hi)`.
    fn rand_in(r: &mut Rng, len: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..len)
            .map(|_| lo + (hi - lo) * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)))
            .collect()
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// Max error of `coeffs` against `f` on a grid of 10001 points of `[a, b]`.
    fn grid_err(coeffs: &[f64], f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        (0..=10_000)
            .map(|i| {
                let x = a + (b - a) * i as f64 / 10_000.0;
                (chebyshev_eval(coeffs, a, b, x) - f(x)).abs()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn interpolation_coefficients() {
        // x^3 - x = (T_3 - T_1) / 4 exactly.
        let c = chebyshev_coeffs(|x| x * x * x - x, -1.0, 1.0, 3);
        let want = [0.0, -0.25, 0.0, 0.25];
        assert!(max_err(&c, &want) < 1e-15, "{c:?}");
        // Brief: degree-31 cosine approximations within 2^-20 on the interval.
        type Case<'a> = (&'a dyn Fn(f64) -> f64, f64, f64);
        let cases: [Case; 3] = [
            (&|x: f64| x.cos(), -1.0, 1.0),
            (&|x: f64| (2.0 * PI * x).cos(), -1.0, 1.0),
            (&|x: f64| (2.0 * PI * (x - 0.25)).cos(), -2.0, 2.0),
        ];
        for (f, a, b) in cases {
            let c = chebyshev_coeffs(f, a, b, 31);
            let e = grid_err(&c, f, a, b);
            eprintln!(
                "degree 31 cosine on [{a}, {b}]: max error 2^{:.1}",
                e.log2()
            );
            assert!(e < 2f64.powi(-20), "{e:e}");
        }
        // EvalMod's sine (K = 13, degree 127): docs/bootstrap.md §4 measures 2^-45.4 at the
        // points that occur; on the whole interval it stays near the float64 floor.
        let k = 13.0;
        let f = |y: f64| (2.0 * PI * k * y).sin();
        let c = chebyshev_coeffs(f, -1.0, 1.0, 127);
        let e = grid_err(&c, f, -1.0, 1.0);
        eprintln!("degree 127 sin(2 pi 13 y): max error 2^{:.1}", e.log2());
        assert!(e < 2f64.powi(-40), "{e:e}");
        // An odd function has (numerically) zero even coefficients.
        assert!(c.iter().step_by(2).all(|v| v.abs() < 1e-14));
    }

    #[test]
    fn depth_and_split() {
        let depths = [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 2),
            (4, 3),
            (7, 3),
            (8, 4),
            (127, 7),
        ];
        for (d, want) in depths {
            assert_eq!(chebyshev_depth(d), want, "degree {d}");
        }
        assert_eq!(chebyshev_depth(255), 8);
        assert_eq!(chebyshev_depth(256), 9);
        // The split rule: 4 for degree 127 (leaves up to degree 15).
        assert_eq!(optimal_split(7), 4);
        assert_eq!(optimal_split(2), 1);
        assert_eq!(optimal_split(1), 1);
        // The split identity p = q T_k + r in the clear.
        let mut r = rng(1);
        let p = rand_in(&mut r, 100, -1.0, 1.0);
        let (q, rem) = split_coeffs(&p, 64);
        let t64 = |y: f64| (64.0 * y.acos()).cos();
        for &y in &rand_in(&mut r, 50, -1.0, 1.0) {
            let lhs = chebyshev_eval(&p, -1.0, 1.0, y);
            let rhs =
                chebyshev_eval(&q, -1.0, 1.0, y) * t64(y) + chebyshev_eval(&rem, -1.0, 1.0, y);
            assert!((lhs - rhs).abs() < 1e-12, "{lhs} {rhs}");
        }
    }

    struct Ctx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        dec: Decryptor,
        eval: Evaluator,
    }

    fn ctx(params: CkksParams, seed: u8) -> Ctx {
        let params = Arc::new(params);
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        Ctx {
            enc: Encoder::new(Arc::clone(&params)),
            encryptor: Encryptor::new(Arc::clone(&params), pk),
            dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
            eval: Evaluator::new(Arc::clone(&params), Some(rk), None),
            params,
        }
    }

    /// N = 2^10 with a 58-bit band of 9 levels above two base primes (insecure, tests only):
    /// the EvalMod regime (input scale 2^58 on 58-bit primes).
    fn band58() -> CkksParams {
        let mut q = vec![54, 42];
        q.extend([58; 9]);
        CkksParams::new(10, &q, &[60, 60], 42, 11).unwrap()
    }

    impl Ctx {
        fn encrypt(&self, v: &[f64], scale: f64, level: usize, r: &mut Rng) -> Ciphertext {
            self.encryptor
                .encrypt(&self.enc.encode(v, scale, level).unwrap(), r)
        }

        fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
            self.enc.decode(&self.dec.decrypt(ct))
        }

        /// Evaluates `coeffs` on random inputs of `[a, b]`; checks the level, the exact scale
        /// and returns the max error against the plaintext series.
        fn run(&self, coeffs: &[f64], a: f64, b: f64, scale: f64, level: usize, seed: u8) -> f64 {
            let mut r = rng(seed);
            let x = rand_in(&mut r, self.params.slots(), a, b);
            let ct = self.encrypt(&x, scale, level, &mut r);
            let out = self.eval.eval_chebyshev(&ct, coeffs, a, b).unwrap();
            let affine = (a, b) != (-1.0, 1.0);
            let depth = chebyshev_depth(coeffs.len() - 1) + affine as usize;
            assert_eq!(out.level(), level - depth, "degree {}", coeffs.len() - 1);
            assert_eq!(out.scale(), scale);
            let want: Vec<f64> = x.iter().map(|&v| chebyshev_eval(coeffs, a, b, v)).collect();
            max_err(&self.decrypt(&out), &want)
        }
    }

    /// Brief, Step 1: x^3 - x at degree 3 within 2^-20, in two regimes (42-bit primes at
    /// scale 2^42 and the 58-bit band), on [-1, 1] (2 levels) and on [-2, 2] (3 levels).
    #[test]
    fn cubic_matches_the_plaintext() {
        let c42 = ctx(
            CkksParams::new(10, &[60, 42, 42, 42, 42], &[60], 42, 5).unwrap(),
            2,
        );
        let c58 = ctx(band58(), 3);
        let coeffs = chebyshev_coeffs(|x| x * x * x - x, -1.0, 1.0, 3);
        for (c, scale) in [(&c42, 2f64.powi(42)), (&c58, 2f64.powi(58))] {
            let top = c.params.max_level();
            let e = c.run(&coeffs, -1.0, 1.0, scale, top, 4);
            eprintln!("x^3 - x at 2^{}: 2^{:.1}", scale.log2(), e.log2());
            assert!(e < 2f64.powi(-20), "{e:e}");
            let wide = chebyshev_coeffs(|x| x * x * x - x, -2.0, 2.0, 3);
            let e = c.run(&wide, -2.0, 2.0, scale, top, 5);
            eprintln!(
                "x^3 - x on [-2, 2] at 2^{}: 2^{:.1}",
                scale.log2(),
                e.log2()
            );
            assert!(e < 2f64.powi(-20), "{e:e}");
        }
    }

    /// Every degree consumes exactly ceil(log2(d + 1)) levels (the level-optimal count of
    /// `BootParams::evalmod_depth`) and matches the plaintext series, for dense random series
    /// with decaying coefficients and for the EvalMod sine.
    #[test]
    fn level_optimal_for_every_degree() {
        let c = ctx(band58(), 6);
        let top = c.params.max_level();
        let s = 2f64.powi(58);
        let mut r = rng(7);
        for d in [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 100, 127, 255,
        ] {
            let coeffs: Vec<f64> = rand_in(&mut r, d + 1, -1.0, 1.0)
                .iter()
                .enumerate()
                .map(|(i, v)| v / (1.0 + i as f64))
                .collect();
            let e = c.run(&coeffs, -1.0, 1.0, s, top, d as u8);
            eprintln!(
                "degree {d}: depth {}, error 2^{:.1}",
                chebyshev_depth(d),
                e.log2()
            );
            assert!(e < 2f64.powi(-25), "degree {d}: {e:e}");
        }
        let k = 13.0;
        let sine = chebyshev_coeffs(|y| (2.0 * PI * k * y).sin(), -1.0, 1.0, 127);
        // From level 8 to level 1: the last unrescaled product (scale 2^116 at level 2) fits
        // Q_2 ~ 2^154; from level 7 it would not fit Q_1 ~ 2^96.
        let e = c.run(&sine, -1.0, 1.0, s, 8, 8);
        eprintln!(
            "sin(2 pi 13 y) degree 127 from level 8: error 2^{:.1}",
            e.log2()
        );
        assert!(e < 2f64.powi(-25), "{e:e}");
    }

    /// The operation counts of the odd degree-127 series match the module docs: 13 powers
    /// (6 powers of two, 7 odd baby steps) and 10 splits; 64 leaf coefficients and 7 power
    /// corrections are constant products.
    #[test]
    fn odd_degree_127_costs() {
        let c = ctx(band58(), 9);
        let mut r = rng(10);
        let mut sine = chebyshev_coeffs(|y| (2.0 * PI * 13.0 * y).sin(), -1.0, 1.0, 127);
        sine.iter_mut().step_by(2).for_each(|v| *v = 0.0);
        let x = rand_in(&mut r, c.params.slots(), -1.0, 1.0);
        let ct = c.encrypt(&x, 2f64.powi(58), 8, &mut r);
        c.eval.reset_counter();
        let out = c.eval.eval_chebyshev(&ct, &sine, -1.0, 1.0).unwrap();
        assert_eq!(out.level(), 1);
        let n = c.eval.counter();
        eprintln!("degree 127 odd: {n:?}");
        assert_eq!(
            (n.ct_ct_tensor, n.relins, n.rescales, n.ct_pt_mults),
            (23, 20, 24, 71)
        );
    }

    #[test]
    fn errors_before_any_work() {
        let c = ctx(band58(), 11);
        let mut r = rng(12);
        let top = c.params.max_level();
        let ct = c.encrypt(&[0.5], 2f64.powi(58), top, &mut r);
        let coeffs = [0.0, 1.0, 0.5];
        let inv = |e: Result<Ciphertext>| matches!(e, Err(HekitError::InvalidParams(_)));
        assert!(inv(c.eval.eval_chebyshev(&ct, &[1.0], -1.0, 1.0)));
        assert!(inv(c.eval.eval_chebyshev(&ct, &[0.0, f64::NAN], -1.0, 1.0)));
        assert!(inv(c.eval.eval_chebyshev(&ct, &coeffs, 1.0, 1.0)));
        assert!(inv(c.eval.eval_chebyshev(
            &ct,
            &coeffs,
            -1.0,
            f64::INFINITY
        )));
        assert!(inv(c
            .eval
            .eval_chebyshev_to_scale(&ct, &coeffs, -1.0, 1.0, 0.0)));
        let no_rk = Evaluator::new(Arc::clone(&c.params), None, None);
        assert!(inv(no_rk.eval_chebyshev(&ct, &coeffs, -1.0, 1.0)));
        let low = c.eval.mod_switch_to_level(&ct, 1).unwrap();
        assert_eq!(
            c.eval.eval_chebyshev(&low, &coeffs, -1.0, 1.0).unwrap_err(),
            HekitError::NoLevelsLeft
        );
        let low2 = c.eval.mod_switch_to_level(&ct, 2).unwrap();
        assert_eq!(
            c.eval
                .eval_chebyshev(&low2, &coeffs, -2.0, 2.0)
                .unwrap_err(),
            HekitError::NoLevelsLeft
        );
        let three = c.eval.mul_no_relin(&ct, &ct).unwrap();
        c.eval.reset_counter();
        assert!(inv(c.eval.eval_chebyshev(&three, &coeffs, -1.0, 1.0)));
        assert_eq!(c.eval.counter(), Default::default());
        // A level-2 input reaches level 0 for depth 2.
        let out = c.eval.eval_chebyshev(&low2, &coeffs, -1.0, 1.0).unwrap();
        assert_eq!((out.level(), out.scale()), (0, 2f64.powi(58)));
    }

    /// The double-angle step: cos(2 theta) from cos(theta) in one level, exact output scale.
    #[test]
    fn double_angle_step() {
        let c = ctx(band58(), 13);
        let mut r = rng(14);
        let x = rand_in(&mut r, c.params.slots(), -1.0, 1.0);
        let level = 4;
        let out_scale = 2f64.powi(58);
        let sigma = c.eval.exact_product_scale(level, out_scale).unwrap();
        let ct = c.encrypt(&x, (2.0 * sigma).sqrt(), level, &mut r);
        let out = double_angle(&c.eval, &ct, out_scale).unwrap();
        assert_eq!((out.level(), out.scale()), (level - 1, out_scale));
        let want: Vec<f64> = x.iter().map(|v| 2.0 * v * v - 1.0).collect();
        let e = max_err(&c.decrypt(&out), &want);
        assert!(e < 2f64.powi(-30), "{e:e}");
        // A scale far from sqrt(2 sigma) is refused.
        let bad = c.encrypt(&x, 2f64.powi(58), level, &mut r);
        assert_eq!(
            double_angle(&c.eval, &bad, out_scale).unwrap_err(),
            HekitError::ScaleMismatch
        );
    }
}
