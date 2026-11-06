//! EvalMod (Task 37): the homomorphic reduction modulo `q_0` of bootstrapping, approximated
//! by `sin(2 pi x) / (2 pi)` (Cheon, Han, Kim, Kim, Song, Eurocrypt 2018) evaluated as a
//! Chebyshev series with the level-optimal evaluation of [`crate::bootstrap::poly`]
//! (`docs/bootstrap.md` §4, §10).
//!
//! # Input convention (scaling folded into CoeffToSlot)
//!
//! After the mod-raise, coefficient `j` of the plaintext is `t_j = Delta c_j + q_0 I_j`
//! (`c = encode(m)`, `I_j` the integer overflow, `|I_j| <= K - 1` with overwhelming
//! probability). Write `x_j = t_j / q_0 = I_j + Delta c_j / q_0`. CoeffToSlot folds the factor
//! `1 / K` into its diagonals (`StageSpec::factor`), so [`Evaluator::eval_mod`] receives the
//! **normalized** slots
//!
//! `y_j = x_j / K = (I_j + Delta c_j / q_0) / K`, in `[-1, 1]`,
//!
//! at the CoeffToSlot output scale (2^58 at level 27 for n16_boot). It returns, at **the same
//! scale**, the slots `sin(2 pi K y_j) = sin(2 pi x_j) ~ 2 pi Delta c_j / q_0` (the integer
//! `I_j` drops out), `evalmod_depth` levels lower (7 for [`BootParams::default_n16`]: level 20).
//! The factor `q_0 / (2 pi)` that turns this back into `Delta c_j` is folded into
//! SlotToCoeff (`docs/bootstrap.md` §2). There is no arcsine correction: at
//! `q_0 / Delta = 2^12` the sine-versus-modulus error alone allows 36 bits (§6).
//!
//! With `r = double_angle > 0` the series approximates `cos(2 pi (K y - 1/4) / 2^r)` and `r`
//! steps `c <- 2 c^2 - 1` give `cos(2 pi (x - 1/4)) = sin(2 pi x)`, one level each. The default
//! is `r = 0`: the sine itself, an odd function, so its even Chebyshev coefficients are set
//! to exactly zero and only odd powers are built.

use std::f64::consts::PI;

use ckks_train_core::error::{HekitError, Result};

use crate::bootstrap::params::BootParams;
use crate::bootstrap::poly::{BootstrapArithmetic, eval_chebyshev_to_scale};
use crate::bootstrap::poly::{chebyshev_coeffs, chebyshev_cost, chebyshev_depth, double_angle};
use crate::ciphertext::Ciphertext;
use crate::counter::OpCounter;
use crate::evaluator::Evaluator;
use crate::linalg::DiagonalEntry;

/// The Chebyshev coefficients EvalMod evaluates for `bp` on `y` in `[-1, 1]`: for `r = 0`
/// the interpolant of degree `bp.cheb_degree` of `sin(2 pi K y)` with the even coefficients
/// set to zero (the function is odd); for `r = bp.double_angle > 0` that of
/// `cos(2 pi (K y - 1/4) / 2^r)`.
///
/// # Panics
/// If `bp.k_intervals` or `bp.cheb_degree` is zero.
pub fn evalmod_coeffs(bp: &BootParams) -> Vec<f64> {
    assert!(
        bp.k_intervals > 0 && bp.cheb_degree > 0,
        "k_intervals and cheb_degree must be positive"
    );
    let mut c = chebyshev_coeffs(|y| evalmod_function(bp, y), -1.0, 1.0, bp.cheb_degree);
    evalmod_parity(bp, &mut c);
    c
}
fn evalmod_function(bp: &BootParams, y: f64) -> f64 {
    let k = bp.k_intervals as f64;
    if bp.double_angle == 0 {
        (2.0 * PI * k * y).sin()
    } else {
        let div = (1u64 << bp.double_angle) as f64;
        (2.0 * PI * (k * y - 0.25) / div).cos()
    }
}
fn evalmod_parity(bp: &BootParams, c: &mut [f64]) {
    if bp.double_angle == 0 {
        c.iter_mut().step_by(2).for_each(|v| *v = 0.0);
    }
}
pub(crate) fn evalmod_coeffs_checked(
    bp: &BootParams,
    budget: &mut super::metadata::Budget,
) -> super::metadata::MetaResult<Vec<f64>> {
    let mut c = super::poly::chebyshev_coeffs_checked(
        |y| evalmod_function(bp, y),
        -1.0,
        1.0,
        bp.cheb_degree,
        budget,
    )?;
    evalmod_parity(bp, &mut c);
    Ok(c)
}

/// The operations [`Evaluator::eval_mod`] performs for `bp` on one ciphertext, as counter
/// fields: [`chebyshev_cost`] of [`evalmod_coeffs`] plus one product, relinearization and
/// rescale per double angle. For [`BootParams::default_n16`]: 23 products, 20
/// relinearizations, 24 rescales and 71 constant products (`docs/bootstrap.md` §10).
///
/// # Panics
/// As [`evalmod_coeffs`].
pub fn evalmod_cost(bp: &BootParams) -> OpCounter {
    let mut c = chebyshev_cost(&evalmod_coeffs(bp), -1.0, 1.0);
    let r = bp.double_angle as u64;
    c.ct_ct_tensor += r;
    c.relins += r;
    c.rescales += r;
    c
}

impl Evaluator {
    /// EvalMod: from the normalized CoeffToSlot output `y = (I + Delta c / q_0) / K` (see the
    /// [module docs](self) for the exact convention) to `sin(2 pi K y) ~ 2 pi Delta c / q_0`,
    /// at exactly `ct.scale()` and level `ct.level() - bp.evalmod_depth()`
    /// (`ceil(log2(cheb_degree + 1)) + double_angle`: 7 for [`BootParams::default_n16`],
    /// 27 -> 20 on n16_boot). Uses [`evalmod_coeffs`], [`Evaluator::eval_chebyshev_to_scale`]
    /// and `r` double-angle steps.
    ///
    /// # Preconditions (not checked: the slots are encrypted)
    ///
    /// * **`|y| <= 1`**, that is `|I + Delta c / q_0| <= K`. The series interpolates on
    ///   `[-1, 1]` only. Just past the edge (`|I| = K` with the outward sign of `m`,
    ///   `|y| - 1 ~ 2^-15.8`) it still holds (measured 2^-21.5 at N = 2^10), but further out
    ///   a degree-127 Chebyshev series grows fast, so a slot with `|I| > K` (`|I| >= K` has
    ///   probability 2^-39.7 per n16_boot bootstrap, `docs/bootstrap.md` §3) is garbage, not
    ///   an approximation.
    /// * **`|Delta c / q_0| << 1`**, that is message slots `|m| <= 1` (Review Focus 2). The
    ///   output is `sin(2 pi x)`, not `2 pi x`: the relative error of reading it as the
    ///   modulus is `(2 pi Delta c / q_0)^2 / 6`, 2^-21.3 at `|c| = 1` with
    ///   `q_0 / Delta = 2^12` and far less for real coefficients (rms 2^-8.8 for slots in
    ///   `[-1, 1]`; `docs/bootstrap.md` §6 gives 36.1 bits at `|m| <= 1`, 27.1 at `|m| <= 8`).
    ///   Larger messages lose precision quadratically and wrap past `|Delta c / q_0| = 1/4`.
    ///
    /// The `r > 0` (double-angle) path is **experimental**: it is checked for levels and
    /// exact scale, and one double-angle step for its value, but no `r > 0` design is
    /// accurate at K = 13 within 7 levels (`docs/bootstrap.md` §4), so the composed path has
    /// no value test; n16_boot uses `r = 0`.
    ///
    /// With double angles, the Chebyshev output scale `t` is chosen so that each square is
    /// relabelled from `t^2` to the exact label of the next planned scale (`t ~ sqrt(2 sigma)`,
    /// a relabelling error of a few `2^-52` relative); the final scale is `ct.scale()` exactly.
    ///
    /// Only the EvalMod fields of `bp` are read (`k_intervals`, `cheb_degree`,
    /// `double_angle`, `evalmod_levels`); [`BootParams::validate`] checks the rest against the
    /// parameters.
    ///
    /// Errors, before any work: `InvalidParams` if `k_intervals` or `cheb_degree` is zero, the
    /// depth exceeds `bp.evalmod_levels`, or as [`Evaluator::eval_chebyshev_to_scale`];
    /// `NoLevelsLeft` if `ct.level()` is below the depth; `ScaleMismatch` if a planned
    /// scale is unreachable.
    pub fn eval_mod(&self, ct: &Ciphertext, bp: &BootParams) -> Result<Ciphertext> {
        eval_mod(self, ct, bp)
    }
}

/// Backend-neutral EvalMod; the same algorithm and scale checks as [`Evaluator::eval_mod`].
pub fn eval_mod<E: BootstrapArithmetic>(ev: &E, ct: &E::Ct, bp: &BootParams) -> Result<E::Ct> {
    if bp.k_intervals == 0 || bp.cheb_degree == 0 {
        return Err(HekitError::InvalidParams(
            "k_intervals and cheb_degree must be positive".into(),
        ));
    }
    let depth = bp.evalmod_depth();
    if depth > bp.evalmod_levels {
        return Err(HekitError::InvalidParams(format!(
            "EvalMod depth {depth} exceeds evalmod_levels {}",
            bp.evalmod_levels
        )));
    }
    ev.check_boot_input(ct)?;
    if ct.level() < depth {
        return Err(HekitError::NoLevelsLeft);
    }
    // Planned scales, backwards from the output: the double angle at level l takes
    // t_{i-1} to t_i; its square is labelled sigma_i = exact label of t_i at l.
    let r = bp.double_angle;
    let mut targets = vec![ct.scale(); r + 1];
    plan_double_angle_targets(
        ct.level(),
        chebyshev_depth(bp.cheb_degree),
        ct.scale(),
        &mut targets,
        |level, target| ev.exact_product_scale(level, target),
        false,
    )
    .map_err(|error| match error {
        TargetPlanError::Source(source) => source,
        TargetPlanError::Unreachable { .. } | TargetPlanError::NonFiniteSeed { .. } => {
            HekitError::ScaleMismatch
        }
    })?;
    let coeffs = evalmod_coeffs(bp);
    let mut c = eval_chebyshev_to_scale(ev, ct, &coeffs, -1.0, 1.0, targets[0])?;
    for &t in &targets[1..] {
        c = double_angle(ev, &c, t)?;
    }
    debug_assert_eq!(c.level(), ct.level() - depth);
    Ok(c)
}

#[derive(Debug)]
pub(crate) enum TargetPlanError<E> {
    Source(E),
    Unreachable { level: usize },
    NonFiniteSeed { level: usize },
}

/// Shared backward boundary planning; target buffer is supplied by the caller.
/// The callback owns the finite-domain guards; legacy runtime keeps its prior domain.
pub(crate) fn plan_double_angle_targets<E>(
    input_level: usize,
    cheb_depth: usize,
    scale: f64,
    targets: &mut [f64],
    mut exact: impl FnMut(usize, f64) -> std::result::Result<f64, E>,
    strict_finite: bool,
) -> std::result::Result<(), TargetPlanError<E>> {
    let cheb_out = input_level - cheb_depth;
    targets.fill(scale);
    for i in (1..targets.len()).rev() {
        let level = cheb_out - (i - 1);
        let sigma = exact(level, targets[i]).map_err(TargetPlanError::Source)?;
        let seed = (2.0 * sigma).sqrt();
        if strict_finite && !seed.is_finite() {
            return Err(TargetPlanError::NonFiniteSeed { level });
        }
        let bits = seed.to_bits();
        // Same nearest-first 0,+1,-1,...,+16,-16 order (including duplicate zero).
        targets[i - 1] = (0..=16u64)
            .flat_map(|k| [bits + k, bits - k])
            .map(f64::from_bits)
            .find(|&s| exact(level + 1, s).is_ok())
            .ok_or(TargetPlanError::Unreachable { level: level + 1 })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::encoder::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use crate::params::CkksParams;
    use ckks_train_core::sampler::Rng;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    fn unit(r: &mut Rng) -> f64 {
        2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
    }

    struct Case {
        /// The integers `I_j`.
        i: Vec<i64>,
        /// The messages `m_j` in `[-1, 1]`.
        m: Vec<f64>,
        /// The normalized inputs `y_j = (I_j + m_j Delta / q_0) / K`.
        y: Vec<f64>,
    }

    /// Brief's EvalMod input `m + q_0 I` (`|m| <= 1`, `|I| <= K`) in the normalized form of
    /// the module docs: `I` uniform in `-(K-1)..=K-1`, the first slots at the edges
    /// `I = +-K` with `m` of the inward sign (so `x = I + m Delta / q_0` stays in `[-K, K]`),
    /// `m` uniform in `[-1, 1]`, and `delta_over_q0 = Delta / q_0`.
    ///
    /// The inward sign at `I = +-K` is a deliberate narrowing of the brief (review m6): the
    /// design covers `|I| <= K - 1` (`BootParams::validate`, `docs/bootstrap.md` §3) and the
    /// series is accurate only for `|y| <= 1`. With `outward`, those two slots get the outward
    /// sign instead (`|y| = 1 + |m| Delta / (q_0 K)`), to observe that case.
    fn case(slots: usize, k: usize, delta_over_q0: f64, outward: bool, r: &mut Rng) -> Case {
        let k = k as i64;
        let mut i: Vec<i64> = (0..slots)
            .map(|_| (r.next_u64() % (2 * k as u64 - 1)) as i64 - (k - 1))
            .collect();
        let mut m: Vec<f64> = (0..slots).map(|_| unit(r)).collect();
        for (j, s) in [(0, 1), (1, -1)] {
            i[j] = s * k;
            m[j] = if outward { s as f64 } else { -(s as f64) } * m[j].abs();
        }
        let y = i
            .iter()
            .zip(&m)
            .map(|(&ij, &mj)| (ij as f64 + mj * delta_over_q0) / k as f64)
            .collect();
        Case { i, m, y }
    }

    /// N = 2^10 with seven 58-bit EvalMod primes above `[54, 42, 34, 34]` (the n16_boot
    /// EvalMod band and q_0 in miniature; insecure, tests only).
    fn band() -> CkksParams {
        let mut q = vec![54, 42, 34, 34];
        q.extend([58; 8]);
        CkksParams::new(10, &q, &[60, 60], 42, 12).unwrap()
    }

    fn bp(degree: usize, r: usize, levels: usize) -> BootParams {
        BootParams {
            k_intervals: 13,
            cheb_degree: degree,
            double_angle: r,
            evalmod_levels: levels,
            ..BootParams::default_n16()
        }
    }

    /// EvalMod on the n16_boot constants (K = 13, degree 127, r = 0) at N = 2^10: the
    /// normalized `m + q_0 I` gives back `m` (after `q_0 / (2 pi Delta)`) within 2^-16 in
    /// exactly 7 levels, at the input scale. Also r = 1 (degree 63 + one double angle, the
    /// docs' 5.7-bit design, checked only for its level count and scale) and the errors.
    #[test]
    fn eval_mod_removes_the_integer_part() {
        let params = Arc::new(band());
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
        let pk = kg.public_key();
        let eval = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
        let enc = Encoder::new(Arc::clone(&params));
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let mut r = rng(2);
        let (q0, delta) = (params.q()[0] as f64, params.scale());
        let c = case(params.slots(), 13, delta / q0, false, &mut r);
        let top = params.max_level();
        let s = 2f64.powi(58);
        let ct = encryptor.encrypt(&enc.encode(&c.y, s, top).unwrap(), &mut r);
        let b = bp(127, 0, 7);
        let k0 = eval.counter();
        let out = eval.eval_mod(&ct, &b).unwrap();
        assert_eq!((out.level(), out.scale()), (top - 7, s));
        // Task 38: the planned counts are exactly what the evaluator counted.
        assert_eq!(k0, Default::default());
        assert_eq!(eval.counter(), evalmod_cost(&b));
        assert_eq!(evalmod_cost(&b).relins, 20);
        let back = q0 / (2.0 * PI * delta);
        let got: Vec<f64> = enc
            .decode(&dec.decrypt(&out))
            .iter()
            .map(|v| v * back)
            .collect();
        let e = got
            .iter()
            .zip(&c.m)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "eval_mod K 13 degree 127 at N = 2^10: max |m - m'| = 2^{:.1} (|I| <= 13)",
            e.log2()
        );
        assert!(e < 2f64.powi(-16), "{e:e}");
        // Regression gate (review m4): the error is the sine term (2 pi Delta / q_0)^2 / 6 at
        // |m| = 1, with little room for noise or approximation error.
        let sine = (2.0 * PI * delta / q0).powi(2) / 6.0;
        assert!(e < 1.05 * sine, "{e:e} vs sine term {sine:e}");
        assert!(c.i.iter().any(|&v| v.abs() == 13));
        // Review m6: the outward edge (|I| = K, |y| slightly above 1).
        let co = case(params.slots(), 13, delta / q0, true, &mut r);
        let cto = encryptor.encrypt(&enc.encode(&co.y, s, top).unwrap(), &mut r);
        let got = enc.decode(&dec.decrypt(&eval.eval_mod(&cto, &b).unwrap()));
        let edge = (0..2)
            .map(|j| (got[j] * back - co.m[j]).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "outward edge |y| = 1 + {:.1e}: max |m - m'| at the two edge slots = 2^{:.1}",
            co.y[0].abs() - 1.0,
            edge.log2()
        );
        assert!(edge < 2f64.powi(-16), "{edge:e}");

        // r = 1: degree 63 then one double angle, 7 levels, exact scale.
        eval.reset_counter();
        let out = eval.eval_mod(&ct, &bp(63, 1, 7)).unwrap();
        assert_eq!((out.level(), out.scale()), (top - 7, s));
        assert_eq!(eval.counter(), evalmod_cost(&bp(63, 1, 7)));

        // Errors before any work.
        eval.reset_counter();
        let inv = |e: Result<Ciphertext>| matches!(e, Err(HekitError::InvalidParams(_)));
        assert!(inv(eval.eval_mod(&ct, &bp(127, 0, 6))));
        assert!(inv(eval.eval_mod(&ct, &bp(0, 0, 7))));
        assert!(inv(eval.eval_mod(
            &ct,
            &BootParams {
                k_intervals: 0,
                ..b.clone()
            }
        )));
        let no_rk = Evaluator::new(Arc::clone(&params), None, None);
        assert!(inv(no_rk.eval_mod(&ct, &b)));
        let low = eval.mod_switch_to_level(&ct, 6).unwrap();
        assert_eq!(
            eval.eval_mod(&low, &b).unwrap_err(),
            HekitError::NoLevelsLeft
        );
        assert_eq!(eval.counter(), Default::default());
    }

    /// **Heavy (N = 2^16, n16_boot relinearization key 0.3 GB, degree 127 at 28 limbs).**
    /// The brief's EvalMod test on n16_boot with `BootParams::default_n16()`: the normalized
    /// `m + q_0 I` (`|m| <= 1`, `|I| <= K = 13`) at level 27 and scale 2^58 gives back `m`
    /// within 2^-16 at level 20 (7 = `evalmod_levels` levels) and scale 2^58.
    #[test]
    fn n16_boot_keys_eval_mod() {
        let params = Arc::new(CkksParams::n16_boot());
        let b = BootParams::default_n16();
        b.validate(&params).unwrap();
        let mut kg = KeyGenerator::new(Arc::clone(&params), rng(31)).unwrap();
        let pk = kg.public_key();
        let eval = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
        let enc = Encoder::new(Arc::clone(&params));
        let encryptor = Encryptor::new(Arc::clone(&params), pk);
        let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
        let mut r = rng(32);
        let (q0, delta) = (params.q()[0] as f64, params.scale());
        let c = case(params.slots(), b.k_intervals, delta / q0, false, &mut r);
        let level =
            crate::bootstrap::levels_after_boot(&params, &b) + b.stoc_levels + b.evalmod_levels;
        assert_eq!(level, 27);
        let s = params.scale_at(level);
        assert_eq!(s, 2f64.powi(58));
        let ct = encryptor.encrypt(&enc.encode(&c.y, s, level).unwrap(), &mut r);
        let t = std::time::Instant::now();
        let out = eval.eval_mod(&ct, &b).unwrap();
        let secs = t.elapsed().as_secs_f64();
        assert_eq!((out.level(), out.scale()), (level - b.evalmod_levels, s));
        let back = q0 / (2.0 * PI * delta);
        let got: Vec<f64> = enc
            .decode(&dec.decrypt(&out))
            .iter()
            .map(|v| v * back)
            .collect();
        let e = got
            .iter()
            .zip(&c.m)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        // The sine-versus-modulus error alone at |m| = 1: (2 pi Delta / q_0)^2 / 6.
        let sine = (2.0 * PI * delta / q0).powi(2) / 6.0;
        eprintln!(
            "n16_boot eval_mod (K 13, degree 127, r 0): level 27 -> {}, max |m - m'| = 2^{:.2} \
             (sine term alone 2^{:.2}), {secs:.1} s, {:?}",
            out.level(),
            e.log2(),
            sine.log2(),
            eval.counter()
        );
        assert!(e < 2f64.powi(-16), "{e:e}");
        // Regression gate (review m4): within 5% of the analytic sine term.
        assert!(e < 1.05 * sine, "{e:e} vs sine term {sine:e}");
    }

    #[test]
    fn coefficients() {
        let c = evalmod_coeffs(&BootParams::default_n16());
        assert_eq!(c.len(), 128);
        assert!(c.iter().step_by(2).all(|&v| v == 0.0));
        // The series is the sine of 2 pi K y, so at y = (I + d) / K it is sin(2 pi d).
        for (i, d) in [
            (0, 0.01),
            (5, -0.003),
            (12, 0.0001),
            (-12, 0.2),
            (-13, 0.001),
        ] {
            let y = (i as f64 + d) / 13.0;
            let v = crate::bootstrap::poly::chebyshev_eval(&c, -1.0, 1.0, y);
            assert!((v - (2.0 * PI * d).sin()).abs() < 2f64.powi(-40), "{i} {d}");
        }
        let c1 = evalmod_coeffs(&bp(63, 1, 7));
        assert_eq!(c1.len(), 64);
        let y: f64 = 0.3;
        let v = crate::bootstrap::poly::chebyshev_eval(&c1, -1.0, 1.0, y);
        let want = (2.0 * PI * (13.0 * y - 0.25) / 2.0).cos();
        assert!((v - want).abs() < 1e-3);
    }
}

#[cfg(test)]
mod metadata_target_tests {
    use super::*;
    #[test]
    fn boundary_search_preserves_nearest_order_and_context() {
        let mut targets = [0.0; 2];
        let mut seen = Vec::new();
        let seed = 2f64.sqrt().to_bits();
        plan_double_angle_targets(
            4,
            2,
            1.0,
            &mut targets,
            |level, target| {
                if level == 2 {
                    return Ok::<_, HekitError>(1.0);
                }
                seen.push(target.to_bits());
                if target.to_bits() == seed - 16 {
                    Ok(1.0)
                } else {
                    Err(HekitError::ScaleMismatch)
                }
            },
            true,
        )
        .unwrap();
        assert_eq!(
            seen,
            (0..=16u64)
                .flat_map(|k| [seed + k, seed - k])
                .collect::<Vec<_>>()
        );
        assert_eq!(targets[0].to_bits(), seed - 16);
        let error = plan_double_angle_targets(
            4,
            2,
            1.0,
            &mut targets,
            |level, _| {
                if level == 2 {
                    Ok::<_, HekitError>(1.0)
                } else {
                    Err(HekitError::ScaleMismatch)
                }
            },
            true,
        )
        .unwrap_err();
        assert!(matches!(error, TargetPlanError::Unreachable { level: 3 }));
        let error = plan_double_angle_targets(
            4,
            2,
            1.0,
            &mut targets,
            |_, _| Ok::<_, HekitError>(f64::MAX),
            true,
        )
        .unwrap_err();
        assert!(matches!(error, TargetPlanError::NonFiniteSeed { level: 2 }));
    }
}
