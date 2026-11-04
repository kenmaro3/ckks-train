//! Homomorphic arithmetic on CKKS ciphertexts.
//!
//! # Rules (checked, never silently fixed)
//!
//! * Every operand must be over the evaluator's data basis (`params.basis_q()`, same `Arc`);
//!   otherwise `InvalidParams`.
//! * Binary operations (`add`, `sub`, `add_plain`, `mul_plain`, `mul_no_relin`, `mul`)
//!   require both operands at the same level, else `LevelMismatch`; use
//!   [`Evaluator::mod_switch_to_level`] to bring the higher operand down explicitly.
//! * Additions (`add`, `sub`, `add_plain`) require scales whose relative difference
//!   `|s_a - s_b| / min(s_a, s_b)` is below [`SCALE_TOLERANCE`] (`2^-10`), else
//!   `ScaleMismatch`; scales are never forced equal. The result carries the first operand's
//!   scale (differences below the tolerance arise from rescaling by different primes, e.g.
//!   `Delta` vs `Delta^2 / q`, relative `~2^-17` for seal_2020).
//! * Multiplications multiply the scales; [`Evaluator::rescale`] divides the scale by the
//!   exact value of the dropped prime `q_level`.
//! * `mul_no_relin`, `mul` and `rescale` at level 0 return `NoLevelsLeft` (a product at
//!   level 0 has scale `~Delta^2` and cannot be rescaled). The level check comes after the
//!   level-mismatch check. `mul_plain` and `relinearize` are allowed at level 0.
//! * Ciphertexts may have 2 or 3 components: `add`/`sub`/`negate`/`add_plain`/`mul_plain`/
//!   `rescale`/`mod_switch_to_level` accept both (adding 3-component ciphertexts and
//!   relinearizing once is the lazy-relinearization pattern); `mul_no_relin`/`mul` need
//!   2-component operands and `relinearize` exactly 3 (`InvalidParams` otherwise).
//! * A missing relinearization key is `InvalidParams`.
//!
//! Operation counting (`OpCounter`) is wired in by Task 19; see the marker in
//! [`Evaluator`].

use std::sync::Arc;

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;

use crate::ciphertext::Ciphertext;
use crate::keys::{GaloisKeys, RelinKey, truncated};
use crate::keyswitch::key_switch;
use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// Largest accepted relative scale difference for additions, `2^-10`.
pub const SCALE_TOLERANCE: f64 = 1.0 / 1024.0;

/// Evaluates arithmetic on ciphertexts with optional relinearization and Galois keys.
#[derive(Debug, Clone)]
pub struct Evaluator {
    params: Arc<CkksParams>,
    relin: Option<RelinKey>,
    galois: Option<GaloisKeys>,
    // Task 19: `counter: OpCounter` goes here; every public method below is one counted op.
}

impl Evaluator {
    /// An evaluator for `params`. `relin` is needed by [`Evaluator::relinearize`] and
    /// [`Evaluator::mul`]; `galois` by the rotations (Task 18). The keys must have been
    /// generated for `params` (not checked).
    pub fn new(
        params: Arc<CkksParams>,
        relin: Option<RelinKey>,
        galois: Option<GaloisKeys>,
    ) -> Evaluator {
        Evaluator {
            params,
            relin,
            galois,
        }
    }

    /// The parameters.
    pub fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }

    /// The relinearization key, if any.
    pub fn relin_key(&self) -> Option<&RelinKey> {
        self.relin.as_ref()
    }

    /// The Galois keys, if any.
    pub fn galois_keys(&self) -> Option<&GaloisKeys> {
        self.galois.as_ref()
    }

    /// `a + b`.
    pub fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.add_sub(a, b, false)
    }

    /// `a - b`.
    pub fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.add_sub(a, b, true)
    }

    /// `-a`.
    pub fn negate(&self, a: &Ciphertext) -> Result<Ciphertext> {
        self.check_basis(a.polys())?;
        let mut c = a.polys().to_vec();
        c.iter_mut().for_each(RnsPoly::neg);
        Ciphertext::new(c, a.scale())
    }

    /// `a + p` (same level, scales within [`SCALE_TOLERANCE`]).
    pub fn add_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        self.check_basis(a.polys())?;
        self.check_basis(std::slice::from_ref(p.poly()))?;
        check_level(a.level(), p.level())?;
        check_scale(a.scale(), p.scale())?;
        let mut c = a.polys().to_vec();
        c[0].add_assign(p.poly());
        Ciphertext::new(c, a.scale())
    }

    /// `a * p` (same level); the scale is the product of the scales.
    pub fn mul_plain(&self, a: &Ciphertext, p: &Plaintext) -> Result<Ciphertext> {
        self.check_basis(a.polys())?;
        self.check_basis(std::slice::from_ref(p.poly()))?;
        check_level(a.level(), p.level())?;
        let mut c = a.polys().to_vec();
        c.iter_mut().for_each(|x| x.mul_assign(p.poly()));
        Ciphertext::new(c, a.scale() * p.scale())
    }

    /// The tensor product `(a0 b0, a0 b1 + a1 b0, a1 b1)` of two 2-component ciphertexts at
    /// the same level `>= 1`; 3 components, scale `s_a * s_b`, decrypting with `s^2`.
    pub fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.check_basis(a.polys())?;
        self.check_basis(b.polys())?;
        if a.len() != 2 || b.len() != 2 {
            return Err(HekitError::InvalidParams(format!(
                "multiplication needs 2-component ciphertexts (got {} and {}); relinearize first",
                a.len(),
                b.len()
            )));
        }
        check_level(a.level(), b.level())?;
        if a.level() == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let (x, y) = (a.polys(), b.polys());
        let prod = |p: &RnsPoly, q: &RnsPoly| {
            let mut r = p.clone();
            r.mul_assign(q);
            r
        };
        let d0 = prod(&x[0], &y[0]);
        let mut d1 = prod(&x[0], &y[1]);
        d1.add_assign(&prod(&x[1], &y[0]));
        let d2 = prod(&x[1], &y[1]);
        Ciphertext::new(vec![d0, d1, d2], a.scale() * b.scale())
    }

    /// Relinearizes a 3-component ciphertext `(c0, c1, c2)` to `(c0 + kb, c1 + ka)` with
    /// `(kb, ka)` the key switch of `c2` by the relinearization key.
    pub fn relinearize(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.check_basis(ct.polys())?;
        if ct.len() != 3 {
            return Err(HekitError::InvalidParams(format!(
                "relinearize needs a 3-component ciphertext (got {})",
                ct.len()
            )));
        }
        let rk = self.relin.as_ref().ok_or_else(|| {
            HekitError::InvalidParams("evaluator has no relinearization key".into())
        })?;
        let mut c = ct.polys().to_vec();
        let c2 = c.pop().expect("3 components");
        let (kb, ka) = key_switch(&self.params, &c2, rk.key());
        c[0].add_assign(&kb);
        c[1].add_assign(&ka);
        Ciphertext::new(c, ct.scale())
    }

    /// `relinearize(mul_no_relin(a, b))`.
    pub fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        if self.relin.is_none() {
            return Err(HekitError::InvalidParams(
                "evaluator has no relinearization key".into(),
            ));
        }
        self.relinearize(&self.mul_no_relin(a, b)?)
    }

    /// Divides by the last prime `q_l` with rounding: level `l -> l - 1`, scale `/ q_l`.
    pub fn rescale(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.check_basis(ct.polys())?;
        let l = ct.level();
        if l == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let mut c = ct.polys().to_vec();
        c.iter_mut().for_each(RnsPoly::rescale_by_last);
        Ciphertext::new(c, ct.scale() / self.params.q()[l] as f64)
    }

    /// Drops primes down to `level` (no division; scale unchanged). `level == ct.level()`
    /// returns a copy; `level > ct.level()` is `InvalidParams`.
    pub fn mod_switch_to_level(&self, ct: &Ciphertext, level: usize) -> Result<Ciphertext> {
        self.check_basis(ct.polys())?;
        if level > ct.level() {
            return Err(HekitError::InvalidParams(format!(
                "cannot switch a level-{} ciphertext up to level {level}",
                ct.level()
            )));
        }
        let c = ct.polys().iter().map(|p| truncated(p, level)).collect();
        Ciphertext::new(c, ct.scale())
    }

    /// `a + b` or `a - b`; the shorter operand is padded with zero components.
    fn add_sub(&self, a: &Ciphertext, b: &Ciphertext, sub: bool) -> Result<Ciphertext> {
        self.check_basis(a.polys())?;
        self.check_basis(b.polys())?;
        check_level(a.level(), b.level())?;
        check_scale(a.scale(), b.scale())?;
        let (x, y) = (a.polys(), b.polys());
        let mut c = x.to_vec();
        for (i, q) in y.iter().enumerate() {
            match (c.get_mut(i), sub) {
                (Some(p), false) => p.add_assign(q),
                (Some(p), true) => p.sub_assign(q),
                (None, false) => c.push(q.clone()),
                (None, true) => {
                    let mut q = q.clone();
                    q.neg();
                    c.push(q);
                }
            }
        }
        Ciphertext::new(c, a.scale())
    }

    /// `InvalidParams` unless every polynomial is over this parameter set's data basis.
    fn check_basis(&self, polys: &[RnsPoly]) -> Result<()> {
        if polys
            .iter()
            .all(|p| Arc::ptr_eq(p.basis(), self.params.basis_q()))
        {
            Ok(())
        } else {
            Err(HekitError::InvalidParams(
                "operand is not over this evaluator's data basis".into(),
            ))
        }
    }
}

/// `LevelMismatch` unless the levels are equal.
fn check_level(a: usize, b: usize) -> Result<()> {
    if a == b {
        Ok(())
    } else {
        Err(HekitError::LevelMismatch)
    }
}

/// `ScaleMismatch` when `|a - b| / min(a, b) >= SCALE_TOLERANCE`.
fn check_scale(a: f64, b: f64) -> Result<()> {
    if (a - b).abs() < SCALE_TOLERANCE * a.min(b) {
        Ok(())
    } else {
        Err(HekitError::ScaleMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Encoder;
    use crate::encrypt::{Decryptor, Encryptor};
    use crate::keys::KeyGenerator;
    use hekit_core::sampler::Rng;
    use std::sync::OnceLock;

    fn rng(seed: u8) -> Rng {
        Rng::from_seed([seed; 32])
    }

    /// Uniform values in `[lo, hi)`.
    fn rand_range(seed: u8, len: usize, lo: f64, hi: f64) -> Vec<f64> {
        let mut r = rng(seed);
        (0..len)
            .map(|_| lo + (hi - lo) * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)))
            .collect()
    }

    fn rand_vals(seed: u8, len: usize) -> Vec<f64> {
        rand_range(seed, len, -1.0, 1.0)
    }

    fn max_err(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    struct Ctx {
        params: Arc<CkksParams>,
        enc: Encoder,
        encryptor: Encryptor,
        dec: Decryptor,
        eval: Evaluator,
    }

    impl Ctx {
        fn new(params: CkksParams, seed: u8) -> Ctx {
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

        fn encrypt(&self, v: &[f64], level: usize, r: &mut Rng) -> Ciphertext {
            let pt = self.enc.encode(v, self.params.scale(), level).unwrap();
            self.encryptor.encrypt(&pt, r)
        }

        fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
            self.enc.decode(&self.dec.decrypt(ct))
        }
    }

    fn seal() -> &'static Ctx {
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| Ctx::new(CkksParams::seal_2020(), 41))
    }

    /// N = 2^12, 6 data primes in 3 digits of 2, 2 special primes (fast; alpha > 1, k > 1).
    fn small() -> &'static Ctx {
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| {
            Ctx::new(
                CkksParams::new(12, &[60, 40, 40, 40, 40, 40], &[60, 60], 40, 3).unwrap(),
                42,
            )
        })
    }

    #[test]
    fn add_sub_negate_match_plaintext() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let (va, vb) = (rand_vals(1, slots), rand_vals(2, slots));
            let mut r = rng(3);
            let top = ctx.params.max_level();
            let (a, b) = (ctx.encrypt(&va, top, &mut r), ctx.encrypt(&vb, top, &mut r));
            let e = &ctx.eval;
            let cases: [(Ciphertext, Vec<f64>); 3] = [
                (
                    e.add(&a, &b).unwrap(),
                    va.iter().zip(&vb).map(|(x, y)| x + y).collect(),
                ),
                (
                    e.sub(&a, &b).unwrap(),
                    va.iter().zip(&vb).map(|(x, y)| x - y).collect(),
                ),
                (e.negate(&a).unwrap(), va.iter().map(|x| -x).collect()),
            ];
            for (i, (ct, want)) in cases.iter().enumerate() {
                assert_eq!((ct.level(), ct.scale(), ct.len()), (top, a.scale(), 2));
                let err = max_err(&ctx.decrypt(ct), want);
                eprintln!(
                    "N=2^{} op {i}: error 2^{:.2}",
                    ctx.params.log_n(),
                    err.log2()
                );
                assert!(err < 2f64.powi(-20), "op {i}: {err:e}");
            }
        }
    }

    #[test]
    fn mul_then_rescale_matches_plaintext() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let (va, vb) = (rand_vals(4, slots), rand_vals(5, slots));
            let want: Vec<f64> = va.iter().zip(&vb).map(|(x, y)| x * y).collect();
            let mut r = rng(6);
            for level in [ctx.params.max_level(), 1] {
                let (a, b) = (
                    ctx.encrypt(&va, level, &mut r),
                    ctx.encrypt(&vb, level, &mut r),
                );
                let prod = ctx.eval.mul(&a, &b).unwrap();
                assert_eq!((prod.len(), prod.level()), (2, level));
                assert_eq!(prod.scale(), a.scale() * b.scale());
                let res = ctx.eval.rescale(&prod).unwrap();
                let err = max_err(&ctx.decrypt(&res), &want);
                eprintln!(
                    "N=2^{} level {level}: mul+rescale error 2^{:.2}",
                    ctx.params.log_n(),
                    err.log2()
                );
                assert!(err < 2f64.powi(-20), "{err:e}");
            }
        }
    }

    #[test]
    fn rescale_scale_bookkeeping() {
        let ctx = seal();
        let mut r = rng(7);
        let top = ctx.params.max_level();
        let a = ctx.encrypt(&rand_vals(8, 16), top, &mut r);
        let sq = ctx.eval.mul(&a, &a).unwrap();
        let res = ctx.eval.rescale(&sq).unwrap();
        assert_eq!(res.level(), top - 1);
        assert_eq!(res.scale(), sq.scale() / ctx.params.q()[top] as f64);
        // Rescaling a fresh ciphertext divides its scale by q_top as well (3 components too).
        let r1 = ctx.eval.rescale(&a).unwrap();
        assert_eq!(
            (r1.level(), r1.scale()),
            (top - 1, a.scale() / ctx.params.q()[top] as f64)
        );
        let t = ctx.eval.mul_no_relin(&a, &a).unwrap();
        let r3 = ctx.eval.rescale(&t).unwrap();
        assert_eq!((r3.len(), r3.level()), (3, top - 1));
    }

    #[test]
    fn plain_ops_match_plaintext() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let (va, vp) = (rand_vals(9, slots), rand_vals(10, slots));
            let mut r = rng(11);
            let top = ctx.params.max_level();
            let a = ctx.encrypt(&va, top, &mut r);
            let p = ctx.enc.encode(&vp, ctx.params.scale(), top).unwrap();
            let sum = ctx.eval.add_plain(&a, &p).unwrap();
            let want: Vec<f64> = va.iter().zip(&vp).map(|(x, y)| x + y).collect();
            assert!(max_err(&ctx.decrypt(&sum), &want) < 2f64.powi(-20));
            let prod = ctx.eval.mul_plain(&a, &p).unwrap();
            assert_eq!(prod.scale(), a.scale() * p.scale());
            let prod = ctx.eval.rescale(&prod).unwrap();
            let want: Vec<f64> = va.iter().zip(&vp).map(|(x, y)| x * y).collect();
            let err = max_err(&ctx.decrypt(&prod), &want);
            assert!(err < 2f64.powi(-20), "{err:e}");
            // Level mismatch for plaintext operands too.
            let p_low = ctx.enc.encode(&vp, ctx.params.scale(), top - 1).unwrap();
            assert_eq!(
                ctx.eval.add_plain(&a, &p_low).unwrap_err(),
                HekitError::LevelMismatch
            );
            assert_eq!(
                ctx.eval.mul_plain(&a, &p_low).unwrap_err(),
                HekitError::LevelMismatch
            );
            let p2 = ctx.enc.encode(&vp, 2.0 * ctx.params.scale(), top).unwrap();
            assert_eq!(
                ctx.eval.add_plain(&a, &p2).unwrap_err(),
                HekitError::ScaleMismatch
            );
        }
    }

    /// Review Focus #2: mismatched scales in add/sub are `Err(ScaleMismatch)` (no panic, no
    /// forcing); a relative difference below 2^-10 is accepted.
    #[test]
    fn scale_mismatch_is_an_error() {
        let ctx = small();
        let mut r = rng(12);
        let top = ctx.params.max_level();
        let v = rand_vals(13, 8);
        let a = ctx.encrypt(&v, top, &mut r);
        let delta = ctx.params.scale();
        for factor in [2.0, 0.5, 1.0 + 2f64.powi(-10), 1.0 - 2f64.powi(-9)] {
            let pt = ctx.enc.encode(&v, delta * factor, top).unwrap();
            let b = ctx.encryptor.encrypt(&pt, &mut r);
            assert_eq!(ctx.eval.add(&a, &b).unwrap_err(), HekitError::ScaleMismatch);
            assert_eq!(ctx.eval.sub(&a, &b).unwrap_err(), HekitError::ScaleMismatch);
            assert_eq!(ctx.eval.add(&b, &a).unwrap_err(), HekitError::ScaleMismatch);
        }
        let pt = ctx
            .enc
            .encode(&v, delta * (1.0 + 2f64.powi(-12)), top)
            .unwrap();
        let b = ctx.encryptor.encrypt(&pt, &mut r);
        let s = ctx.eval.add(&a, &b).unwrap();
        assert_eq!(s.scale(), a.scale());
        // Realistic case: Delta vs Delta^2 / q_top after a rescale (relative ~2^-17).
        let one = ctx.encrypt(&[1.0; 8], top, &mut r);
        let x1 = ctx.eval.rescale(&ctx.eval.mul(&a, &one).unwrap()).unwrap();
        let x0 = ctx.eval.mod_switch_to_level(&a, top - 1).unwrap();
        assert_ne!(x1.scale(), x0.scale());
        let d = ctx.eval.sub(&x1, &x0).unwrap();
        assert!(
            ctx.decrypt(&d)[..8]
                .iter()
                .all(|x| x.abs() < 2f64.powi(-12))
        );
    }

    /// Review Focus #2: `mul` (and `add`) at different levels is `Err(LevelMismatch)`;
    /// `mod_switch_to_level` is the explicit fix.
    #[test]
    fn level_mismatch_is_an_error() {
        let ctx = small();
        let mut r = rng(14);
        let top = ctx.params.max_level();
        let (va, vb) = (rand_vals(15, 8), rand_vals(16, 8));
        let a = ctx.encrypt(&va, top, &mut r);
        let b = ctx.encrypt(&vb, top - 2, &mut r);
        let e = &ctx.eval;
        assert_eq!(e.mul(&a, &b).unwrap_err(), HekitError::LevelMismatch);
        assert_eq!(e.mul(&b, &a).unwrap_err(), HekitError::LevelMismatch);
        assert_eq!(
            e.mul_no_relin(&a, &b).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert_eq!(e.add(&a, &b).unwrap_err(), HekitError::LevelMismatch);
        assert_eq!(e.sub(&a, &b).unwrap_err(), HekitError::LevelMismatch);
        let a0 = ctx.encrypt(&va, 0, &mut r);
        assert_eq!(e.mul(&a0, &b).unwrap_err(), HekitError::LevelMismatch);

        let a_low = e.mod_switch_to_level(&a, top - 2).unwrap();
        assert_eq!((a_low.level(), a_low.scale()), (top - 2, a.scale()));
        let s = e.add(&a_low, &b).unwrap();
        let want: Vec<f64> = va.iter().zip(&vb).map(|(x, y)| x + y).collect();
        assert!(max_err(&ctx.decrypt(&s)[..8], &want) < 2f64.powi(-20));
        let p = e.mul(&a_low, &b).unwrap();
        assert_eq!(p.level(), top - 2);
        assert_eq!(e.mod_switch_to_level(&a, top).unwrap().level(), top);
        assert!(matches!(
            e.mod_switch_to_level(&b, top),
            Err(HekitError::InvalidParams(_))
        ));
    }

    /// Review Focus #4: `rescale` / `mul` at level 0 is `Err(NoLevelsLeft)`, no panic.
    #[test]
    fn level_zero_has_no_levels_left() {
        let ctx = small();
        let mut r = rng(17);
        let v = rand_vals(18, 8);
        let a = ctx.encrypt(&v, 0, &mut r);
        let b = ctx.encrypt(&v, 0, &mut r);
        let e = &ctx.eval;
        assert_eq!(e.rescale(&a).unwrap_err(), HekitError::NoLevelsLeft);
        assert_eq!(e.mul(&a, &b).unwrap_err(), HekitError::NoLevelsLeft);
        assert_eq!(
            e.mul_no_relin(&a, &b).unwrap_err(),
            HekitError::NoLevelsLeft
        );
        // Reaching level 0 by rescaling, then trying again.
        let c = ctx.encrypt(&v, 1, &mut r);
        let c0 = e.rescale(&e.mul(&c, &c).unwrap()).unwrap();
        assert_eq!(c0.level(), 0);
        assert_eq!(e.rescale(&c0).unwrap_err(), HekitError::NoLevelsLeft);
        assert_eq!(e.mul(&c0, &c0).unwrap_err(), HekitError::NoLevelsLeft);
        // Additions and plaintext products still work at level 0.
        assert!(e.add(&a, &b).is_ok());
        let p = ctx.enc.encode(&v, 2.0, 0).unwrap();
        assert!(e.mul_plain(&a, &p).is_ok());
    }

    #[test]
    fn component_count_and_key_checks() {
        let ctx = small();
        let mut r = rng(19);
        let top = ctx.params.max_level();
        let v = rand_vals(20, 8);
        let a = ctx.encrypt(&v, top, &mut r);
        let e = &ctx.eval;
        let t = e.mul_no_relin(&a, &a).unwrap();
        assert_eq!((t.len(), t.scale()), (3, a.scale() * a.scale()));
        assert!(matches!(
            e.relinearize(&a),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(e.mul(&t, &a), Err(HekitError::InvalidParams(_))));
        assert!(matches!(
            e.mul_no_relin(&a, &t),
            Err(HekitError::InvalidParams(_))
        ));
        // 3 + 2 components: the result keeps the third component.
        let s = e.add(&t, &e.mul(&a, &a).unwrap()).unwrap();
        assert_eq!(s.len(), 3);
        let s = e.sub(&e.mul(&a, &a).unwrap(), &t).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(e.negate(&t).unwrap().len(), 3);
        // No relinearization key.
        let bare = Evaluator::new(Arc::clone(&ctx.params), None, None);
        assert!(matches!(
            bare.relinearize(&t),
            Err(HekitError::InvalidParams(_))
        ));
        assert!(matches!(
            bare.mul(&a, &a),
            Err(HekitError::InvalidParams(_))
        ));
        // Foreign basis.
        let other = Ctx::new(
            CkksParams::new(12, &[60, 40, 40, 40, 40, 40], &[60, 60], 40, 3).unwrap(),
            43,
        );
        let f = other.encrypt(&v, top, &mut r);
        assert!(matches!(e.add(&a, &f), Err(HekitError::InvalidParams(_))));
        assert!(matches!(e.rescale(&f), Err(HekitError::InvalidParams(_))));
    }

    /// Lazy relinearization: `relinearize(sum of 4 mul_no_relin)` equals the sum of 4 `mul`
    /// within 2^-20 (both rescaled), and both match the plaintext.
    #[test]
    fn lazy_relinearization_matches_eager() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let top = ctx.params.max_level();
            let mut r = rng(21);
            let vs: Vec<(Vec<f64>, Vec<f64>)> = (0..4)
                .map(|i| (rand_vals(22 + 2 * i, slots), rand_vals(23 + 2 * i, slots)))
                .collect();
            let cts: Vec<(Ciphertext, Ciphertext)> = vs
                .iter()
                .map(|(a, b)| (ctx.encrypt(a, top, &mut r), ctx.encrypt(b, top, &mut r)))
                .collect();
            let e = &ctx.eval;
            let mut lazy = e.mul_no_relin(&cts[0].0, &cts[0].1).unwrap();
            let mut eager = e.mul(&cts[0].0, &cts[0].1).unwrap();
            for (a, b) in &cts[1..] {
                lazy = e.add(&lazy, &e.mul_no_relin(a, b).unwrap()).unwrap();
                eager = e.add(&eager, &e.mul(a, b).unwrap()).unwrap();
            }
            assert_eq!(lazy.len(), 3);
            let lazy = e.rescale(&e.relinearize(&lazy).unwrap()).unwrap();
            let eager = e.rescale(&eager).unwrap();
            assert_eq!((lazy.len(), lazy.scale()), (2, eager.scale()));
            let (dl, de) = (ctx.decrypt(&lazy), ctx.decrypt(&eager));
            let want: Vec<f64> = (0..slots)
                .map(|i| vs.iter().map(|(a, b)| a[i] * b[i]).sum())
                .collect();
            let (diff, err) = (max_err(&dl, &de), max_err(&dl, &want));
            eprintln!(
                "N=2^{}: lazy vs eager 2^{:.2}, lazy vs plaintext 2^{:.2}",
                ctx.params.log_n(),
                diff.log2(),
                err.log2()
            );
            assert!(diff < 2f64.powi(-20), "{diff:e}");
            assert!(err < 2f64.powi(-20), "{err:e}");
        }
    }

    /// Depth-9 chain `x^(2^9)` on seal_2020, using all 9 levels (level 9 -> 0).
    ///
    /// Deviation from the plan's `x in [0.9, 1.1]`: `1.1^512 ~ 2^70` cannot be represented
    /// (at level 0, `Q = q_0 ~ 2^60` and scale `~2^40` leave `|m| < 2^19`; the last product
    /// at level 1 needs `|m| Delta^2 < Q_1 / 2 ~ 2^99`), so `x in [0.9, 1.025]`
    /// (`1.025^512 ~ 2^18.2`). Values range over `[2^-78, 2^18.2]`, so the bound `2^-10` is
    /// applied as `|got - want| <= 2^-10 * max(1, |want|)` (absolute below 1, relative above;
    /// the relative error doubles with every squaring).
    #[test]
    fn depth_nine_squaring_chain() {
        let ctx = seal();
        let slots = ctx.params.slots();
        let mut x = rand_range(30, slots, 0.9, 1.025);
        x[..4].copy_from_slice(&[0.9, 1.0, 1.025, 0.95]);
        let mut r = rng(31);
        let mut ct = ctx.encrypt(&x, ctx.params.max_level(), &mut r);
        for k in 1..=9 {
            ct = ctx.eval.rescale(&ctx.eval.mul(&ct, &ct).unwrap()).unwrap();
            assert_eq!(ct.level(), 9 - k);
        }
        assert_eq!(ct.level(), 0);
        assert_eq!(
            ctx.eval.mul(&ct, &ct).unwrap_err(),
            HekitError::NoLevelsLeft
        );
        let got = ctx.decrypt(&ct);
        let (mut worst, mut worst_abs, mut worst_rel) = (0f64, 0f64, 0f64);
        for (g, xi) in got.iter().zip(&x) {
            let want = xi.powi(512);
            let err = (g - want).abs();
            worst = worst.max(err / want.abs().max(1.0));
            if want < 1.0 {
                worst_abs = worst_abs.max(err);
            } else {
                worst_rel = worst_rel.max(err / want);
            }
        }
        eprintln!(
            "depth 9: max normalized error 2^{:.2} (absolute, want < 1: 2^{:.2}; relative, \
             want >= 1: 2^{:.2}); final scale 2^{:.6}",
            worst.log2(),
            worst_abs.log2(),
            worst_rel.log2(),
            ct.scale().log2()
        );
        assert!(worst <= 2f64.powi(-10), "{worst:e}");
    }

    /// n16_train sanity: one mul + rescale at the top level.
    #[test]
    fn n16_train_mul_rescale() {
        let ctx = Ctx::new(CkksParams::n16_train(), 44);
        let slots = ctx.params.slots();
        let (va, vb) = (rand_vals(32, slots), rand_vals(33, slots));
        let mut r = rng(34);
        let top = ctx.params.max_level();
        let (a, b) = (ctx.encrypt(&va, top, &mut r), ctx.encrypt(&vb, top, &mut r));
        let res = ctx.eval.rescale(&ctx.eval.mul(&a, &b).unwrap()).unwrap();
        assert_eq!(res.level(), top - 1);
        let want: Vec<f64> = va.iter().zip(&vb).map(|(x, y)| x * y).collect();
        let err = max_err(&ctx.decrypt(&res), &want);
        eprintln!("n16_train: mul+rescale error 2^{:.2}", err.log2());
        assert!(err < 2f64.powi(-20), "{err:e}");
    }
}
