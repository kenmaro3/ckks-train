//! Polynomial activations on ciphertexts: [`Square`], [`PolyRelu2`] (ReBoot's `z^2 + z`),
//! [`Poly`] (degree 2..=4) and [`Identity`] (a linear layer, Task 32).
//!
//! # Exact-scale discipline
//!
//! Every plaintext constant is encoded so that the result of the plaintext multiply, after the
//! rescale, has exactly the scale the other addend has: to multiply `ct` (scale `S`, level `l`)
//! by `v` and land on scale `T`, `v` is encoded at scale `T q_l / S` (`q_l` the prime that the
//! rescale divides by), so `S * (T q_l / S) / q_l = T`. Additive constants are encoded at the
//! exact scale and level of the ciphertext they are added to. Only ciphertexts of identical
//! history are ever added (no `2^-10` relabelling). Level switches (free) align levels.
//!
//! # Levels
//!
//! A ciphertext-by-plaintext multiply costs a level (rescale), as does a ct-ct multiply.
//! [`Activation::depth`] is the number of levels the forward pass consumes:
//! `Square` 1, `PolyRelu2` 1 (`z` is carried to `z^2`'s level and scale by an exact-scale
//! multiply by 1 encoded at scale `S`, which consumes the same level as the square: 0 extra
//! levels, +1 ct-pt and +1 rescale),
//! `Poly` of degree `d` is `ceil(log2(d + 1))`: 2 for `d = 2, 3`, 3 for `d = 4`. A degree-4 poly
//! is evaluated as `c0 + c1 z + z^2 (c2 + c3 z + c4 z^2)`, which is minimal-depth when a
//! scalar multiply costs a level; plain Horner would take 4. No Paterson-Stockmeyer.
//!
//! # Derivative factor and constant folding (how Task 30 uses it)
//!
//! `docs/layout.md` §3-§4 fold every constant into the derivative factor
//! `s_l = act'(z_l) * (scale_const * mask)`, e.g. `s_L = z_L * (2 c valid)` for `Square`
//! (`c = lr / n`). [`Activation::derivative_factor`] returns exactly that: the constant
//! `scale_const` and the slot mask are multiplied into the *coefficients* of the derivative
//! polynomial (plaintext vectors), so they cost no level beyond the derivative itself
//! ([`Activation::derivative_depth`]: 1 for `Square`/`PolyRelu2`, `ceil(log2 d)` for `Poly`).
//! `z` is available early, so `s_l` is off the critical path; the critical-path cost is the one
//! ct-ct product `dz = s_l * upstream`. [`Activation::derivative_factor_at`] additionally lets
//! the caller pick the exact scale of `s_l` (the "free scale knob" of layout.md §4) to match
//! `scale(W_l)` at the update. [`Activation::backward`] is `derivative_factor(1, None)` times
//! `upstream` (one level more than the derivative factor, one relin); the training loop should
//! call `derivative_factor` with its constants and multiply itself.
//!
//! The `Encoder` is a parameter of every method: building one costs `O(N)` tables.

use ckks_train_ckks::{DiagonalEntry, Encoder, Evaluator, HeOps};
use ckks_train_core::error::{HekitError, Result};

/// A polynomial activation applied to a ciphertext of pre-activations `z`.
///
/// Generic over the evaluator `E` ([`Evaluator`] by default, so `dyn Activation` is the CPU
/// one); the library's activations implement it for every `E: HeOps`, and also have the
/// `E`-independent methods ([`Activation::eval_plain`], [`Activation::depth`], ...) as inherent
/// methods, so `Square.depth()` needs no evaluator type. Constants are encoded on the host
/// with `enc` and uploaded ([`HeOps::upload_pt_owned`]).
pub trait Activation<E: HeOps = Evaluator> {
    /// `f(z)`. Consumes [`Activation::depth`] levels; the result's scale is the natural scale
    /// of the computation's history (e.g. `S^2 / q_l` for `Square`).
    ///
    /// Errors: `NoLevelsLeft` if `z.level() < depth()`; evaluator errors (missing relin key).
    fn forward(&self, ev: &E, enc: &Encoder, z: &E::Ct) -> Result<E::Ct>;

    /// `scale_const * mask * f'(z)` with the constant and slot mask folded into the plaintext
    /// multiplies (no extra level), at the scale `target_scale` (default: `z.scale()`).
    /// `mask` (zero-padded to all slots, `None` = all ones) must not be longer than the slot
    /// count. The result is at level `z.level() - derivative_depth()`.
    fn derivative_factor_at(
        &self,
        ev: &E,
        enc: &Encoder,
        z: &E::Ct,
        scale_const: f64,
        mask: Option<&[f64]>,
        target_scale: Option<f64>,
    ) -> Result<E::Ct>;

    /// [`Activation::derivative_factor_at`] at `z`'s own scale.
    fn derivative_factor(
        &self,
        ev: &E,
        enc: &Encoder,
        z: &E::Ct,
        scale_const: f64,
        mask: Option<&[f64]>,
    ) -> Result<E::Ct> {
        self.derivative_factor_at(ev, enc, z, scale_const, mask, None)
    }

    /// `f'(z) * upstream`, rescaled: levels are aligned by switching the higher operand down,
    /// the result is at `min(z.level() - derivative_depth(), upstream.level()) - 1`.
    fn backward(&self, ev: &E, enc: &Encoder, z: &E::Ct, upstream: &E::Ct) -> Result<E::Ct> {
        let s = self.derivative_factor(ev, enc, z, 1.0, None)?;
        let level = s.level().min(upstream.level());
        let s = ev.mod_switch_to_level(&s, level)?;
        let u = ev.mod_switch_to_level(upstream, level)?;
        ev.rescale(&ev.mul(&s, &u)?)
    }

    /// `f(z)` on a plaintext value: evaluation on decrypted weights
    /// ([`crate::trainer::Mlp::predict_plain`]).
    fn eval_plain(&self, z: f64) -> f64;

    /// Levels consumed by [`Activation::forward`].
    fn depth(&self) -> usize;

    /// Levels consumed by [`Activation::derivative_factor`].
    fn derivative_depth(&self) -> usize;

    /// `Some(k)` if the derivative is the constant `k` (an affine activation, e.g.
    /// [`Identity`]). A constant is not a ciphertext, so [`Activation::derivative_factor_at`]
    /// does not apply; the trainer then forms `dz = delta * (k * valid)` with one plaintext
    /// multiply (no relinearization), or takes `dz = delta` when `k = 1` and every lane is
    /// valid (no op, no level). Default `None`.
    fn constant_derivative(&self) -> Option<f64> {
        None
    }
}

/// The identity `f(z) = z`: a linear layer (ReBoot's `"none"` output activation). Depth 0,
/// derivative the constant 1 ([`Activation::constant_derivative`]).
#[derive(Debug, Clone, Copy, Default)]
pub struct Identity;

/// `z^2`; derivative `2 z`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Square;

/// ReBoot's `z^2 + z`; derivative `2 z + 1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PolyRelu2;

/// `c0 + c1 z + ... + c4 z^4` (`coeffs[i] = c_i`), degree 2..=4 after trimming trailing zeros
/// (degree 1 has a constant derivative, which is not a ciphertext).
#[derive(Debug, Clone, PartialEq)]
pub struct Poly {
    coeffs: Vec<f64>,
}

impl Poly {
    /// `InvalidParams` unless the degree (after trimming trailing zeros) is in `2..=4` and all
    /// coefficients are finite.
    pub fn new(mut coeffs: Vec<f64>) -> Result<Poly> {
        if coeffs.iter().any(|c| !c.is_finite()) {
            return Err(HekitError::InvalidParams("non-finite coefficient".into()));
        }
        while coeffs.last() == Some(&0.0) {
            coeffs.pop();
        }
        if !(3..=5).contains(&coeffs.len()) {
            return Err(HekitError::InvalidParams(format!(
                "Poly degree must be 2..=4 (got {})",
                coeffs.len().saturating_sub(1)
            )));
        }
        Ok(Poly { coeffs })
    }

    /// The coefficients, `c_0` first (trimmed).
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// The degree.
    pub fn degree(&self) -> usize {
        self.coeffs.len() - 1
    }
}

fn ceil_log2(x: usize) -> usize {
    (usize::BITS - (x - 1).leading_zeros()) as usize
}

/// A coefficient vector (slot-wise); `None` is the zero vector (the term is skipped).
type Coef = Option<Vec<f64>>;

fn constant(slots: usize, c: f64) -> Coef {
    (c != 0.0).then(|| vec![c; slots])
}

fn q_at<E: HeOps>(ev: &E, level: usize) -> f64 {
    ev.params().q()[level] as f64
}

/// Shared ordinary multiply label expression; deliberately not exact-target search.
pub(crate) fn ordinary_plain_scale(q: f64, source: f64, target: f64) -> f64 {
    target * q / source
}

/// `rescale(ct * v)` with `v` encoded so that the result has scale `target` (see the module
/// docs). One level.
fn mul_to<E: HeOps>(ev: &E, enc: &Encoder, ct: &E::Ct, v: &[f64], target: f64) -> Result<E::Ct> {
    let l = ct.level();
    if l == 0 {
        return Err(HekitError::NoLevelsLeft);
    }
    let pt = ev.upload_pt_owned(enc.encode(
        v,
        ordinary_plain_scale(q_at(ev, l), ct.scale(), target),
        l,
    )?)?;
    ev.rescale(&ev.mul_plain(ct, &pt)?)
}

/// `ct + v` with `v` encoded at `ct`'s exact scale and level.
fn add_vec<E: HeOps>(ev: &E, enc: &Encoder, ct: &E::Ct, v: &Coef) -> Result<E::Ct> {
    match v {
        None => Ok(ct.clone()),
        Some(v) => ev.add_plain(
            ct,
            &ev.upload_pt_owned(enc.encode(v, ct.scale(), ct.level())?)?,
        ),
    }
}

fn add_aligned<E: HeOps>(ev: &E, a: &E::Ct, b: &E::Ct) -> Result<E::Ct> {
    let l = a.level().min(b.level());
    ev.add(
        &ev.mod_switch_to_level(a, l)?,
        &ev.mod_switch_to_level(b, l)?,
    )
}

fn square<E: HeOps>(ev: &E, z: &E::Ct) -> Result<E::Ct> {
    ev.rescale(&ev.mul(z, z)?)
}

fn check_levels<C: DiagonalEntry>(z: &C, need: usize) -> Result<()> {
    if z.level() < need {
        return Err(HekitError::NoLevelsLeft);
    }
    Ok(())
}

/// Evaluates `sum_i coeffs[i] z^i` for slot-vector coefficients (`coeffs.len() - 1 = d` in
/// `1..=4`, `coeffs[d]` nonzero), in `ceil(log2(d + 1))` levels, landing on `target` (default:
/// the natural scale). See the module docs for the scheme.
fn eval_poly<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    z: &E::Ct,
    coeffs: &[Coef],
    target: Option<f64>,
) -> Result<E::Ct> {
    let d = coeffs.len() - 1;
    assert!((1..=4).contains(&d) && coeffs[d].is_some());
    check_levels(z, ceil_log2(d + 1))?;
    let lin = |t: f64| -> Result<Option<E::Ct>> {
        coeffs[1]
            .as_ref()
            .map(|c| mul_to(ev, enc, z, c, t))
            .transpose()
    };
    let finish = |acc: E::Ct, t: f64| -> Result<E::Ct> {
        // `acc` has scale `t`; add `c1 z` (exact scale `t`) and `c0`.
        let acc = match lin(t)? {
            Some(l1) => add_aligned(ev, &acc, &l1)?,
            None => acc,
        };
        add_vec(ev, enc, &acc, &coeffs[0])
    };
    if d == 1 {
        let t = target.unwrap_or(z.scale());
        let out = lin(t)?.expect("c1 nonzero");
        return add_vec(ev, enc, &out, &coeffs[0]);
    }
    let z2 = square(ev, z)?;
    if d == 2 {
        let t = target.unwrap_or(z2.scale());
        let p2 = mul_to(ev, enc, &z2, coeffs[2].as_ref().expect("c2"), t)?;
        let t = p2.scale();
        return finish(p2, t);
    }
    // q = c2 + c3 z + c4 z^2, at level ql; result = z^2 q + c1 z + c0.
    let ql = z.level() - if d == 4 { 2 } else { 1 };
    let t_in = match target {
        Some(t) => t * q_at(ev, ql) / z2.scale(),
        None => z2.scale(),
    };
    let mut q: Option<E::Ct> = None;
    if let Some(c3) = &coeffs[3] {
        q = Some(mul_to(ev, enc, z, c3, t_in)?);
    }
    if let Some(c4) = coeffs.get(4).and_then(Option::as_ref) {
        let t4 = mul_to(ev, enc, &z2, c4, t_in)?;
        q = Some(match q {
            Some(q3) => add_aligned(ev, &q3, &t4)?,
            None => t4,
        });
    }
    let q = q.expect("c3 or c4 nonzero");
    let q = ev.mod_switch_to_level(&q, ql)?;
    let q = add_vec(ev, enc, &q, &coeffs[2])?;
    let z2 = ev.mod_switch_to_level(&z2, ql)?;
    let prod = ev.rescale(&ev.mul(&q, &z2)?)?;
    let t = prod.scale();
    finish(prod, t)
}

/// `scale_const * mask` as a slot vector (`None` mask: all ones).
fn folded<E: HeOps>(ev: &E, scale_const: f64, mask: Option<&[f64]>) -> Result<Vec<f64>> {
    let slots = ev.params().slots();
    if !scale_const.is_finite() {
        return Err(HekitError::InvalidParams("non-finite scale_const".into()));
    }
    match mask {
        None => Ok(vec![scale_const; slots]),
        Some(m) if m.len() > slots => Err(HekitError::InvalidParams(format!(
            "mask has {} entries, {slots} slots",
            m.len()
        ))),
        Some(m) => Ok(m.iter().map(|&x| scale_const * x).collect()),
    }
}

fn scaled(v: &[f64], f: f64) -> Coef {
    (f != 0.0 && v.iter().any(|&x| x != 0.0)).then(|| v.iter().map(|&x| f * x).collect())
}

/// The E-independent methods of [`Activation`], delegated to the inherent ones of the same
/// name (so that `Square.depth()` needs no evaluator type).
macro_rules! shape_methods {
    () => {
        fn eval_plain(&self, z: f64) -> f64 {
            Self::eval_plain(self, z)
        }
        fn depth(&self) -> usize {
            Self::depth(self)
        }
        fn derivative_depth(&self) -> usize {
            Self::derivative_depth(self)
        }
        fn constant_derivative(&self) -> Option<f64> {
            Self::constant_derivative(self)
        }
    };
}

impl Identity {
    /// `f(z)` on a plaintext value ([`Activation::eval_plain`]).
    pub fn eval_plain(&self, z: f64) -> f64 {
        z
    }

    /// Levels consumed by the forward ([`Activation::depth`]).
    pub fn depth(&self) -> usize {
        0
    }

    /// Levels consumed by the derivative factor ([`Activation::derivative_depth`]).
    pub fn derivative_depth(&self) -> usize {
        0
    }

    /// The constant derivative, if any ([`Activation::constant_derivative`]).
    pub fn constant_derivative(&self) -> Option<f64> {
        Some(1.0)
    }
}

impl<E: HeOps> Activation<E> for Identity {
    /// `z` itself (no op, no level).
    fn forward(&self, _ev: &E, _enc: &Encoder, z: &E::Ct) -> Result<E::Ct> {
        Ok(z.clone())
    }

    /// Always `InvalidParams`: the derivative is the constant 1, which is not a ciphertext
    /// (use [`Activation::constant_derivative`]).
    fn derivative_factor_at(
        &self,
        _ev: &E,
        _enc: &Encoder,
        _z: &E::Ct,
        _scale_const: f64,
        _mask: Option<&[f64]>,
        _target_scale: Option<f64>,
    ) -> Result<E::Ct> {
        Err(HekitError::InvalidParams(
            "the identity has a constant derivative (no ciphertext); use constant_derivative"
                .into(),
        ))
    }

    /// `upstream` itself (`f' = 1`: no op, no level).
    fn backward(&self, _ev: &E, _enc: &Encoder, _z: &E::Ct, upstream: &E::Ct) -> Result<E::Ct> {
        Ok(upstream.clone())
    }

    shape_methods!();
}

impl Square {
    /// `f(z)` on a plaintext value ([`Activation::eval_plain`]).
    pub fn eval_plain(&self, z: f64) -> f64 {
        z * z
    }

    /// Levels consumed by the forward ([`Activation::depth`]).
    pub fn depth(&self) -> usize {
        1
    }

    /// Levels consumed by the derivative factor ([`Activation::derivative_depth`]).
    pub fn derivative_depth(&self) -> usize {
        1
    }

    /// The constant derivative, if any ([`Activation::constant_derivative`]).
    pub fn constant_derivative(&self) -> Option<f64> {
        None
    }
}

impl<E: HeOps> Activation<E> for Square {
    fn forward(&self, ev: &E, _enc: &Encoder, z: &E::Ct) -> Result<E::Ct> {
        check_levels(z, 1)?;
        square(ev, z)
    }

    fn derivative_factor_at(
        &self,
        ev: &E,
        enc: &Encoder,
        z: &E::Ct,
        scale_const: f64,
        mask: Option<&[f64]>,
        target_scale: Option<f64>,
    ) -> Result<E::Ct> {
        let km = folded(ev, scale_const, mask)?;
        let c1 = scaled(&km, 2.0).ok_or_else(|| {
            HekitError::InvalidParams("derivative factor is identically zero".into())
        })?;
        eval_poly(ev, enc, z, &[None, Some(c1)], target_scale)
    }

    shape_methods!();
}

impl PolyRelu2 {
    /// `f(z)` on a plaintext value ([`Activation::eval_plain`]).
    pub fn eval_plain(&self, z: f64) -> f64 {
        z * z + z
    }

    /// Levels consumed by the forward ([`Activation::depth`]).
    pub fn depth(&self) -> usize {
        1
    }

    /// Levels consumed by the derivative factor ([`Activation::derivative_depth`]).
    pub fn derivative_depth(&self) -> usize {
        1
    }

    /// The constant derivative, if any ([`Activation::constant_derivative`]).
    pub fn constant_derivative(&self) -> Option<f64> {
        None
    }
}

impl<E: HeOps> Activation<E> for PolyRelu2 {
    /// `z^2 + z`: `z` is brought to `z^2`'s level and scale by `rescale(z * 1)` with `1`
    /// encoded at scale `S` (exact), so the sum adds identical histories.
    fn forward(&self, ev: &E, enc: &Encoder, z: &E::Ct) -> Result<E::Ct> {
        check_levels(z, 1)?;
        let z2 = square(ev, z)?;
        let ones = vec![1.0; ev.params().slots()];
        let z1 = mul_to(ev, enc, z, &ones, z2.scale())?;
        ev.add(&z2, &z1)
    }

    fn derivative_factor_at(
        &self,
        ev: &E,
        enc: &Encoder,
        z: &E::Ct,
        scale_const: f64,
        mask: Option<&[f64]>,
        target_scale: Option<f64>,
    ) -> Result<E::Ct> {
        let km = folded(ev, scale_const, mask)?;
        let c1 = scaled(&km, 2.0).ok_or_else(|| {
            HekitError::InvalidParams("derivative factor is identically zero".into())
        })?;
        eval_poly(ev, enc, z, &[scaled(&km, 1.0), Some(c1)], target_scale)
    }

    shape_methods!();
}

impl Poly {
    /// `f(z)` on a plaintext value ([`Activation::eval_plain`]).
    pub fn eval_plain(&self, z: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * z + c)
    }

    /// Levels consumed by the forward ([`Activation::depth`]).
    pub fn depth(&self) -> usize {
        ceil_log2(self.degree() + 1)
    }

    /// Levels consumed by the derivative factor ([`Activation::derivative_depth`]).
    pub fn derivative_depth(&self) -> usize {
        ceil_log2(self.degree())
    }

    /// The constant derivative, if any ([`Activation::constant_derivative`]).
    pub fn constant_derivative(&self) -> Option<f64> {
        None
    }
}

impl<E: HeOps> Activation<E> for Poly {
    fn forward(&self, ev: &E, enc: &Encoder, z: &E::Ct) -> Result<E::Ct> {
        let slots = ev.params().slots();
        let coeffs: Vec<Coef> = self.coeffs.iter().map(|&c| constant(slots, c)).collect();
        eval_poly(ev, enc, z, &coeffs, None)
    }

    fn derivative_factor_at(
        &self,
        ev: &E,
        enc: &Encoder,
        z: &E::Ct,
        scale_const: f64,
        mask: Option<&[f64]>,
        target_scale: Option<f64>,
    ) -> Result<E::Ct> {
        let km = folded(ev, scale_const, mask)?;
        let coeffs: Vec<Coef> = (1..self.coeffs.len())
            .map(|i| scaled(&km, i as f64 * self.coeffs[i]))
            .collect();
        if coeffs.last().is_none_or(Option::is_none) {
            return Err(HekitError::InvalidParams(
                "derivative factor is identically zero".into(),
            ));
        }
        eval_poly(ev, enc, z, &coeffs, target_scale.or(Some(z.scale())))
    }

    shape_methods!();
}

#[cfg(test)]
mod p2_helpers {
    use super::*;
    use crate::diagonal_plan::tests::context;
    #[test]
    fn p2_ordinary_label_order() {
        crate::diagonal_plan::tests::reserve_helper_calls(6);
        let c = context(false);
        let ev = c.ev.fork();
        let z = c.encrypt_at(7, c.p.scale() * 1.125, 0.125);
        let q = c.p.q()[7] as f64;
        let before = ev.counter();
        let square = Square.forward(&ev, &c.enc, &z).unwrap();
        assert_eq!(
            square.scale().to_bits(),
            ((z.scale() * z.scale()) / q).to_bits()
        );
        let poly = PolyRelu2.forward(&ev, &c.enc, &z).unwrap();
        assert_eq!(poly.scale().to_bits(), square.scale().to_bits());
        for act in [&Square as &dyn Activation, &PolyRelu2 as &dyn Activation] {
            let s = act
                .derivative_factor_at(&ev, &c.enc, &z, 1., None, None)
                .unwrap();
            let sp = ordinary_plain_scale(q, z.scale(), z.scale());
            assert_eq!(
                (s.level(), s.scale().to_bits()),
                (6, ((z.scale() * sp) / q).to_bits())
            );
        }
        let mask = vec![1.; c.p.slots()];
        let d = crate::trainer::constant_dz(&ev, &c.enc, &z, 1., &mask, true).unwrap();
        assert_eq!(d.scale().to_bits(), z.scale().to_bits());
        let p = crate::trainer::constant_dz(&ev, &c.enc, &z, 1., &mask, false).unwrap();
        assert_eq!((p.level(), p.scale().to_bits()), (6, z.scale().to_bits()));
        let got = &ev.counter() - &before;
        assert_eq!(
            (got.ct_ct_tensor, got.relins, got.ct_pt_mults, got.rescales),
            (2, 2, 4, 6)
        );
    }
}
