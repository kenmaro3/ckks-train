//! Homomorphic arithmetic on CKKS ciphertexts.
//!
//! # Rules (checked; one documented relabelling)
//!
//! * Every operand must be over the evaluator's data basis (`params.basis_q()`, same `Arc`);
//!   otherwise `InvalidParams`.
//! * Binary operations (`add`, `sub`, `add_plain`, `mul_plain`, `mul_no_relin`, `mul`)
//!   require both operands at the same level, else `LevelMismatch`; use
//!   [`Evaluator::mod_switch_to_level`] to bring the higher operand down explicitly.
//! * Additions (`add`, `sub`, `add_plain`) require scales whose relative difference
//!   `|s_a - s_b| / min(s_a, s_b)` is below [`SCALE_TOLERANCE`] (`2^-10`), else
//!   `ScaleMismatch`. Within the tolerance the scales are **not** equalised (no
//!   multiplication by a correction constant): the polynomials are added as they are and the
//!   result is **labelled with the first operand's scale** `s_a`. This relabelling is not
//!   free: `b`'s message then decodes as `m_b * s_b / s_a`, a relative error of
//!   `(s_b - s_a) / s_a` on `m_b` (absolute `|m_b| * |s_b - s_a| / s_a`). See "Scale
//!   tolerance" below for the sizes and the exact-scale path.
//! * Multiplications multiply the scales; [`Evaluator::rescale`] divides the scale by the
//!   exact value of the dropped prime `q_level`.
//! * `mul_no_relin`, `mul` and `rescale` at level 0 return `NoLevelsLeft` (a product at
//!   level 0 has scale `~Delta^2` and cannot be rescaled). The level check comes after the
//!   level-mismatch check. `mul_plain` and `relinearize` are allowed at level 0.
//! * Ciphertexts may have 2 or 3 components: `add`/`sub`/`negate`/`add_plain`/`mul_plain`/
//!   `rescale`/`mod_switch_to_level` accept both (adding 3-component ciphertexts and
//!   relinearizing once is the lazy-relinearization pattern); `mul_no_relin`/`mul` need
//!   2-component operands and `relinearize` exactly 3 (`InvalidParams` otherwise).
//! * A missing relinearization key is `InvalidParams`; `mul` checks its operands first.
//! * Keys must be over the evaluator's `basis_q` / `basis_p` (same `Arc`s, `dnum` parts),
//!   else `InvalidParams`.
//!
//! # Scale tolerance (what it costs and how to avoid it)
//!
//! Scales of different histories differ because `rescale` divides by the exact prime
//! `q_level`, not by `Delta`: a fresh `Delta = 2^40` encoding and a product rescaled to
//! `Delta^2 / q_l` differ relatively by `|1 - Delta / q_l|`, which for the seal_2020 40-bit
//! primes is `2^-19.4 ... 2^-16.8`, and for the n16_train 42-bit primes (`Delta = 2^42`)
//! `2^-21.4 ... 2^-16.0`. Two products rescaled by different primes differ by about the sum.
//! Every `add` / `sub` / `add_plain` of such mismatched-history operands therefore injects
//! up to about `2^-16` relative error into the second operand (for `|m| <= 1`, up to
//! `2^-16` absolute), silently: far above the encryption noise (`~2^-25`) of seal_2020.
//!
//! To avoid it, keep scales exactly equal:
//!
//! * encode plaintext constants, biases and masks at `ct.scale()` and `ct.level()` of the
//!   ciphertext they are added to (`encoder.encode(v, ct.scale(), ct.level())`), not at
//!   `params.scale()`;
//! * add ciphertexts only when they have the same history (same multiplications and the
//!   same rescaled primes), e.g. the terms of one BSGS sum ([`crate::linalg`]);
//! * a scalar multiplication that feeds an addition must take the target scale (M2's
//!   `mul_scalar` in Task 28): choose the plaintext scale `s_t * q_l / s_ct` so that after
//!   the rescale the result has exactly the scale `s_t` of the other operand.
//!
//! [`SCALE_TOLERANCE`] stays at the plan's `2^-10`; check `a.scale() == b.scale()` yourself
//! where exactness matters.
//!
//! # Rotations
//!
//! [`Evaluator::rotate`] rotates left (slot `i` of the result is slot `i + steps` of the
//! input, indices mod `slots = N/2`); `steps` is reduced mod `slots` and a multiple of
//! `slots` returns a copy. Galois elements follow SEAL 4.1 (generator 3, see
//! [`crate::keys`]). [`Evaluator::conjugate`] uses element `2N - 1`, whose key is stored in
//! the same [`GaloisKeys`]. [`Evaluator::hoisted_rotate`] shares one digit decomposition
//! among several rotations of the same ciphertext. A missing key is
//! `InvalidParams("missing galois key")`; rotations need 2-component ciphertexts.
//!
//! # Operation counts
//!
//! Every evaluator counts its successful expensive operations in an [`OpCounter`]
//! (see [`crate::counter`] for what each field counts); read a snapshot with
//! [`Evaluator::counter`] and zero it with [`Evaluator::reset_counter`]. The counter sits behind a `Mutex`, so `Evaluator` is `Sync` and
//! `&Evaluator` can be shared across threads.

use std::sync::{Arc, Mutex, PoisonError};

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;

use crate::ciphertext::Ciphertext;
use crate::counter::OpCounter;
use crate::keys::{
    GaloisKey, GaloisKeys, KskKey, QpPoly, RelinKey, conjugation_elt, galois_elt, truncated,
};
use crate::keyswitch::{apply_key, decompose, key_switch};
use crate::params::CkksParams;
use crate::plaintext::Plaintext;

/// Largest accepted relative scale difference for additions, `2^-10`. Within it the result
/// is relabelled with the first operand's scale, which costs up to the relative scale
/// difference in accuracy (see the [module docs](self)).
pub const SCALE_TOLERANCE: f64 = 1.0 / 1024.0;

/// Evaluates arithmetic on ciphertexts with optional relinearization and Galois keys, and
/// counts the expensive operations it performs ([`Evaluator::counter`]).
///
/// `Clone` copies the keys and the current counts into an independent counter. `Debug`
/// prints a summary (ring degree, top level, which keys are present, the counts), not the
/// keys.
pub struct Evaluator {
    params: Arc<CkksParams>,
    relin: Option<RelinKey>,
    galois: Option<GaloisKeys>,
    counter: Mutex<OpCounter>,
}

impl std::fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator")
            .field("n", &self.params.n())
            .field("max_level", &self.params.max_level())
            .field("relin_key", &self.relin.is_some())
            .field(
                "galois_keys",
                &self.galois.as_ref().map_or(0, GaloisKeys::len),
            )
            .field("counter", &self.counter())
            .finish()
    }
}

impl Clone for Evaluator {
    fn clone(&self) -> Evaluator {
        Evaluator {
            params: Arc::clone(&self.params),
            relin: self.relin.clone(),
            galois: self.galois.clone(),
            counter: Mutex::new(self.counter()),
        }
    }
}

// `Evaluator` must stay shareable across threads (the counter uses a `Mutex`, not a `Cell`).
const _: fn() = || {
    fn assert_sync_send<T: Sync + Send>() {}
    assert_sync_send::<Evaluator>();
};

impl Evaluator {
    /// An evaluator for `params`. `relin` is needed by [`Evaluator::relinearize`] and
    /// [`Evaluator::mul`]; `galois` by the rotations and conjugation. The keys must have been
    /// generated for `params` (the same `Arc`): checked when a key is used.
    pub fn new(
        params: Arc<CkksParams>,
        relin: Option<RelinKey>,
        galois: Option<GaloisKeys>,
    ) -> Evaluator {
        Evaluator {
            params,
            relin,
            galois,
            counter: Mutex::new(OpCounter::default()),
        }
    }

    /// A snapshot of the operation counts so far (see [`crate::counter`]).
    pub fn counter(&self) -> OpCounter {
        self.counter
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Resets the operation counts to zero.
    pub fn reset_counter(&self) {
        self.count(|c| *c = OpCounter::default());
    }

    /// Applies `f` to the counter (called after an operation succeeded).
    fn count(&self, f: impl FnOnce(&mut OpCounter)) {
        f(&mut self.counter.lock().unwrap_or_else(PoisonError::into_inner));
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

    /// `a + b` (same level, scales within [`SCALE_TOLERANCE`]; the result has `a`'s scale,
    /// see the [module docs](self)).
    pub fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.add_sub(a, b, false)
    }

    /// `a - b` (as [`Evaluator::add`]).
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

    /// `a + p` (same level, scales within [`SCALE_TOLERANCE`]; the result has `a`'s scale,
    /// see the [module docs](self) for the error of a mismatched `p`).
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
        let r = Ciphertext::new(c, a.scale() * p.scale())?;
        self.count(|c| c.ct_pt_mults += 1);
        Ok(r)
    }

    /// The tensor product `(a0 b0, a0 b1 + a1 b0, a1 b1)` of two 2-component ciphertexts at
    /// the same level `>= 1`; 3 components, scale `s_a * s_b`, decrypting with `s^2`.
    pub fn mul_no_relin(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.check_mul_operands(a, b)?;
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
        let r = Ciphertext::new(vec![d0, d1, d2], a.scale() * b.scale())?;
        self.count(|c| c.ct_ct_tensor += 1);
        Ok(r)
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
        let rk = self.relin_key_checked()?;
        let mut c = ct.polys().to_vec();
        let c2 = c.pop().expect("3 components");
        let (kb, ka) = key_switch(&self.params, &c2, rk.key());
        c[0].add_assign(&kb);
        c[1].add_assign(&ka);
        let r = Ciphertext::new(c, ct.scale())?;
        self.count(|c| c.relins += 1);
        Ok(r)
    }

    /// `relinearize(mul_no_relin(a, b))`. Operand errors (basis, component count,
    /// `LevelMismatch`, `NoLevelsLeft`) are reported before a missing or foreign
    /// relinearization key (`InvalidParams`), and both before any work is done.
    pub fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        self.check_mul_operands(a, b)?;
        self.relin_key_checked()?;
        self.relinearize(&self.mul_no_relin(a, b)?)
    }

    /// The relinearization key, validated against this evaluator's bases.
    pub(crate) fn relin_key_checked(&self) -> Result<&RelinKey> {
        let rk = self.relin.as_ref().ok_or_else(|| {
            HekitError::InvalidParams("evaluator has no relinearization key".into())
        })?;
        self.check_key_basis(rk.key())?;
        Ok(rk)
    }

    /// The operand checks of `mul_no_relin` / `mul`, in order: basis, component count,
    /// level mismatch, level 0.
    fn check_mul_operands(&self, a: &Ciphertext, b: &Ciphertext) -> Result<()> {
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
        Ok(())
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
        let r = Ciphertext::new(c, ct.scale() / self.params.q()[l] as f64)?;
        self.count(|c| c.rescales += 1);
        Ok(r)
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

    /// Rotates the slots left by `steps`: slot `i` of the result holds slot
    /// `(i + steps) mod slots` of `ct` (negative `steps` rotate right). `steps` is reduced
    /// modulo `slots = N/2`; a multiple of `slots` returns a copy of `ct` (no key needed).
    ///
    /// `(c0, c1) -> (sigma_g(c0) + kb, ka)` with `g = galois_elt(N, steps)` and `(kb, ka)` the
    /// key switch of `sigma_g(c1)` by the Galois key of `g`. Level and scale are unchanged.
    ///
    /// Errors (`InvalidParams`): `ct` not over this evaluator's basis or not 2 components;
    /// no Galois key for `g` (`"missing galois key"`); a key not over this evaluator's bases.
    pub fn rotate(&self, ct: &Ciphertext, steps: i32) -> Result<Ciphertext> {
        self.check_galois_input(ct)?;
        let Some(elt) = self.rotation_elt(steps) else {
            return Ok(ct.clone());
        };
        self.apply_galois(ct, elt)
    }

    /// Complex conjugation of every slot (Galois element `2N - 1`), with the conjugation key
    /// stored in the evaluator's [`GaloisKeys`] (see [`GaloisKeys::insert`] and
    /// [`KeyGenerator::conjugation_key`](crate::KeyGenerator::conjugation_key)).
    ///
    /// Errors as [`Evaluator::rotate`].
    pub fn conjugate(&self, ct: &Ciphertext) -> Result<Ciphertext> {
        self.check_galois_input(ct)?;
        self.apply_galois(ct, conjugation_elt(self.params.n()))
    }

    /// `steps.iter().map(|s| rotate(ct, s))`, with the expensive part of the key switch
    /// (the digit decomposition and lift of `c1`, [`decompose`]) done once.
    ///
    /// For every nonzero step (mod slots), the automorphism `sigma_g` is applied to both
    /// parts of every lifted digit and the result is passed to [`apply_key`] with the
    /// Galois key of `g`; then `sigma_g(c0)` is added. This is valid because `sigma_g` is a
    /// signed permutation of the coefficients (an index permutation in NTT form, the same
    /// for every prime), so it commutes with reduction modulo every prime: the permuted
    /// lift is a lift of `sigma_g(c1)`'s digit. The lift used by [`decompose`] is the exact
    /// centered one, which is symmetric (`lift(-x) = -lift(x)`), so permuting it gives
    /// exactly the lift of the permuted digit: the result is bit-identical to
    /// [`Evaluator::rotate`] for every `alpha` (see [`crate::keyswitch`], "Hoisting").
    ///
    /// Results are in the order of `steps`; steps that are multiples of `slots` give copies
    /// of `ct`, and repeated steps are allowed. Every key is looked up before any work, so a
    /// missing key fails fast. Errors as [`Evaluator::rotate`].
    pub fn hoisted_rotate(&self, ct: &Ciphertext, steps: &[i32]) -> Result<Vec<Ciphertext>> {
        self.check_galois_input(ct)?;
        let keys = steps
            .iter()
            .map(|&s| {
                self.rotation_elt(s)
                    .map(|elt| self.galois_key(elt).map(|k| (elt, k)))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let c = ct.polys();
        let digits = if keys.iter().any(Option::is_some) {
            decompose(&self.params, &c[1])
        } else {
            Vec::new()
        };
        let nontrivial = keys.iter().filter(|k| k.is_some()).count() as u64;
        let out = keys
            .into_iter()
            .map(|k| match k {
                None => Ok(ct.clone()),
                Some((elt, gk)) => {
                    let rotated: Vec<QpPoly> = digits
                        .iter()
                        .map(|d| QpPoly::new(d.q().automorphism(elt), d.p().automorphism(elt)))
                        .collect();
                    let (kb, ka) = apply_key(&self.params, &rotated, gk.key());
                    finish_galois(ct, elt, kb, ka)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.count(|c| c.rotations += nontrivial);
        Ok(out)
    }

    /// `Ok` when a valid Galois key exists for every step of `steps` that is not a multiple
    /// of `slots`, else the error [`Evaluator::rotate`] would return
    /// (`InvalidParams("missing galois key")` or a foreign key).
    pub(crate) fn check_rotation_keys(&self, steps: &[i32]) -> Result<()> {
        for &s in steps {
            if let Some(elt) = self.rotation_elt(s) {
                self.galois_key(elt)?;
            }
        }
        Ok(())
    }

    /// The Galois element of a rotation by `steps`, or `None` for a multiple of `slots`.
    fn rotation_elt(&self, steps: i32) -> Option<u64> {
        let slots = self.params.slots() as i64;
        let k = (steps as i64).rem_euclid(slots);
        (k != 0).then(|| galois_elt(self.params.n(), k as i32))
    }

    /// `sigma_elt` with a key switch of `c1` (no hoisting).
    fn apply_galois(&self, ct: &Ciphertext, elt: u64) -> Result<Ciphertext> {
        let gk = self.galois_key(elt)?;
        let d = ct.polys()[1].automorphism(elt);
        let (kb, ka) = key_switch(&self.params, &d, gk.key());
        let r = finish_galois(ct, elt, kb, ka)?;
        self.count(|c| c.rotations += 1);
        Ok(r)
    }

    /// The (validated) Galois key for `elt`.
    fn galois_key(&self, elt: u64) -> Result<&GaloisKey> {
        let two_n = 2 * self.params.n() as u64;
        // The core automorphism only debug_asserts this.
        if elt % 2 == 0 || elt >= two_n {
            return Err(HekitError::InvalidParams(format!(
                "Galois element {elt} must be odd and < 2N = {two_n}"
            )));
        }
        let gk = self
            .galois
            .as_ref()
            .and_then(|g| g.get_elt(elt))
            .ok_or_else(|| HekitError::InvalidParams("missing galois key".into()))?;
        self.check_key_basis(gk.key())?;
        Ok(gk)
    }

    /// `InvalidParams` unless `ct` is over this evaluator's basis with 2 components.
    pub(crate) fn check_galois_input(&self, ct: &Ciphertext) -> Result<()> {
        self.check_basis(ct.polys())?;
        if ct.len() != 2 {
            return Err(HekitError::InvalidParams(format!(
                "rotation needs a 2-component ciphertext (got {}); relinearize first",
                ct.len()
            )));
        }
        Ok(())
    }

    /// `InvalidParams` unless `key` has `dnum` parts over this evaluator's `basis_q` and
    /// `basis_p` (same `Arc`s): a key generated for an equal but separately constructed
    /// parameter set is rejected instead of silently producing garbage.
    fn check_key_basis(&self, key: &KskKey) -> Result<()> {
        let (bq, bp) = (self.params.basis_q(), self.params.basis_p());
        let ok = key.dnum() == self.params.dnum()
            && key.parts().iter().all(|(b, a)| {
                [b, a]
                    .iter()
                    .all(|x| Arc::ptr_eq(x.q().basis(), bq) && Arc::ptr_eq(x.p().basis(), bp))
            });
        if ok {
            Ok(())
        } else {
            Err(HekitError::InvalidParams(
                "key is not over this evaluator's bases".into(),
            ))
        }
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
    pub(crate) fn check_basis(&self, polys: &[RnsPoly]) -> Result<()> {
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

/// `(sigma_elt(c0) + kb, ka)` with the scale of `ct`.
fn finish_galois(ct: &Ciphertext, elt: u64, kb: RnsPoly, ka: RnsPoly) -> Result<Ciphertext> {
    let mut r0 = ct.polys()[0].automorphism(elt);
    r0.add_assign(&kb);
    Ciphertext::new(vec![r0, ka], ct.scale())
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
    use num_complex::Complex64;
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
            Ctx::with_rotations(params, seed, None)
        }

        /// With `Some(steps)`: Galois keys for `steps` plus the conjugation key (generated
        /// after the public and relinearization keys, so those do not depend on `steps`).
        fn with_rotations(params: CkksParams, seed: u8, steps: Option<&[i32]>) -> Ctx {
            let params = Arc::new(params);
            let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
            let pk = kg.public_key();
            let rk = kg.relin_key();
            let gks = steps.map(|steps| {
                let mut gks = kg.galois_keys(steps);
                gks.insert(kg.conjugation_key());
                gks
            });
            Ctx {
                enc: Encoder::new(Arc::clone(&params)),
                encryptor: Encryptor::new(Arc::clone(&params), pk),
                dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
                eval: Evaluator::new(Arc::clone(&params), Some(rk), gks),
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

    /// Rotation steps with Galois keys in the shared contexts: `1, -1 (= slots - 1), 2, 4, 8`
    /// and `slots / 2` (only the keys the tests need; key generation at N = 2^15 is slowish).
    fn rotation_steps(slots: usize) -> Vec<i32> {
        vec![1, -1, 2, 4, 8, (slots / 2) as i32]
    }

    fn seal() -> &'static Ctx {
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| {
            let params = CkksParams::seal_2020();
            let steps = rotation_steps(params.slots());
            Ctx::with_rotations(params, 41, Some(&steps))
        })
    }

    fn small_params() -> CkksParams {
        CkksParams::new(12, &[60, 40, 40, 40, 40, 40], &[60, 60], 40, 3).unwrap()
    }

    /// N = 2^12, 6 data primes in 3 digits of 2, 2 special primes (fast; alpha > 1, k > 1).
    fn small() -> &'static Ctx {
        static CTX: OnceLock<Ctx> = OnceLock::new();
        CTX.get_or_init(|| {
            let params = small_params();
            let steps = rotation_steps(params.slots());
            Ctx::with_rotations(params, 42, Some(&steps))
        })
    }

    /// `v` rotated left by `steps` (slot `i` gets `v[i + steps mod len]`).
    fn shifted<T: Copy>(v: &[T], steps: i64) -> Vec<T> {
        let n = v.len() as i64;
        (0..n)
            .map(|i| v[(i + steps).rem_euclid(n) as usize])
            .collect()
    }

    fn max_err_complex(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    /// m3: `Debug` of the evaluator, ciphertexts and plaintexts prints shapes, not limbs.
    #[test]
    fn debug_is_summarized() {
        let c = small();
        let s = format!("{:?}", c.eval);
        assert!(
            s.starts_with("Evaluator { n: 4096, max_level: 5, relin_key: true, galois_keys: 7, counter: OpCounter {"),
            "{s}"
        );
        assert!(s.len() < 300, "{s}");
        let pt = c.enc.encode(&[0.5], 4.0, 2).unwrap();
        assert_eq!(
            format!("{pt:?}"),
            "Plaintext { n: 4096, level: 2, scale: 4.0 }"
        );
        let ct = c.encryptor.encrypt(&pt, &mut rng(3));
        assert_eq!(
            format!("{ct:?}"),
            "Ciphertext { n: 4096, len: 2, level: 2, scale: 4.0 }"
        );
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
        let other = Ctx::new(small_params(), 43);
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

    /// n16_train sanity: one mul + rescale and one rotation at the top level (one context,
    /// since key generation at N = 2^16 is slow).
    #[test]
    fn n16_train_mul_rescale_rotate() {
        let ctx = Ctx::with_rotations(CkksParams::n16_train(), 44, Some(&[1]));
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
        let rot = ctx.eval.rotate(&a, 1).unwrap();
        assert_eq!((rot.len(), rot.level(), rot.scale()), (2, top, a.scale()));
        let err = max_err(&ctx.decrypt(&rot), &shifted(&va, 1));
        eprintln!("n16_train: rotate(1) error 2^{:.2}", err.log2());
        assert!(err < 2f64.powi(-20), "{err:e}");
        let hoisted = ctx.eval.hoisted_rotate(&a, &[1]).unwrap();
        let diff = max_err(&ctx.decrypt(&hoisted[0]), &ctx.decrypt(&rot));
        eprintln!("n16_train: hoisted vs rotate(1) 2^{:.2}", diff.log2());
        assert!(diff < 2f64.powi(-25), "{diff:e}");
        // alpha = 4: bit-identical with the exact centered lift (the fast lift gave 2^-25.5).
        assert_eq!(hoisted[0].polys()[1].limbs(), rot.polys()[1].limbs());
    }

    /// Review Focus #1: `decode(rotate(encrypt(v), s))` is the cyclic left shift of `v` by `s`
    /// for `s` negative, 0, `slots/2`, `slots - 1`, `-slots` (= 0) and `>= slots`
    /// (`slots + 1`, `2 slots - 1`), within 2^-20.
    #[test]
    fn rotate_matches_cyclic_shift() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let s = slots as i32;
            let v = rand_vals(40, slots);
            let mut r = rng(41);
            let top = ctx.params.max_level();
            let ct = ctx.encrypt(&v, top, &mut r);
            for steps in [1, -1, 0, s / 2, s - 1, -s, s + 1, 2 * s - 1] {
                let rot = ctx.eval.rotate(&ct, steps).unwrap();
                assert_eq!((rot.len(), rot.level(), rot.scale()), (2, top, ct.scale()));
                let err = max_err(&ctx.decrypt(&rot), &shifted(&v, steps as i64));
                eprintln!(
                    "N=2^{} rotate({steps}): error 2^{:.2}",
                    ctx.params.log_n(),
                    err.log2()
                );
                assert!(err < 2f64.powi(-20), "steps {steps}: {err:e}");
            }
            // Steps congruent to 0 return the input unchanged (no key switch).
            for steps in [0, -s, 3 * s] {
                let rot = ctx.eval.rotate(&ct, steps).unwrap();
                assert_eq!(rot.polys()[0].limbs(), ct.polys()[0].limbs());
                assert_eq!(rot.polys()[1].limbs(), ct.polys()[1].limbs());
            }
        }
    }

    /// Rotations below the top level (including level 0) and composition `rotate(-1) o
    /// rotate(1) = id`.
    #[test]
    fn rotate_at_lower_levels() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let v = rand_vals(42, slots);
            let mut r = rng(43);
            for level in [2, 0] {
                let ct = ctx.encrypt(&v, level, &mut r);
                for steps in [1, -1, 8] {
                    let rot = ctx.eval.rotate(&ct, steps).unwrap();
                    assert_eq!(rot.level(), level);
                    let err = max_err(&ctx.decrypt(&rot), &shifted(&v, steps as i64));
                    eprintln!(
                        "N=2^{} level {level} rotate({steps}): error 2^{:.2}",
                        ctx.params.log_n(),
                        err.log2()
                    );
                    assert!(err < 2f64.powi(-20), "level {level} steps {steps}: {err:e}");
                }
                let back = ctx
                    .eval
                    .rotate(&ctx.eval.rotate(&ct, 1).unwrap(), -1)
                    .unwrap();
                assert!(max_err(&ctx.decrypt(&back), &v) < 2f64.powi(-20));
            }
            // A rescaled ciphertext (scale Delta^2 / q_top) rotates with its scale kept.
            let top = ctx.params.max_level();
            let ct = ctx.encrypt(&v, top, &mut r);
            let sq = ctx.eval.rescale(&ctx.eval.mul(&ct, &ct).unwrap()).unwrap();
            let rot = ctx.eval.rotate(&sq, 2).unwrap();
            assert_eq!((rot.level(), rot.scale()), (top - 1, sq.scale()));
            let want: Vec<f64> = shifted(&v, 2).iter().map(|x| x * x).collect();
            assert!(max_err(&ctx.decrypt(&rot), &want) < 2f64.powi(-20));
        }
    }

    /// `hoisted_rotate(ct, &[1, 2, 4, 8])` equals the individual rotations within 2^-25
    /// (decoded; in fact bit-identical, for alpha = 1 (seal_2020) and alpha = 2, thanks to
    /// the exact centered lift), in the order of `steps`; step 0 and repeated steps are
    /// allowed.
    #[test]
    fn hoisted_rotate_matches_individual() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let v = rand_vals(44, slots);
            let mut r = rng(45);
            for level in [ctx.params.max_level(), 1] {
                let ct = ctx.encrypt(&v, level, &mut r);
                let steps = [1, 2, 4, 8, 0, -1, 2];
                let hoisted = ctx.eval.hoisted_rotate(&ct, &steps).unwrap();
                assert_eq!(hoisted.len(), steps.len());
                for (h, &s) in hoisted.iter().zip(&steps) {
                    let single = ctx.eval.rotate(&ct, s).unwrap();
                    assert_eq!((h.len(), h.level(), h.scale()), (2, level, ct.scale()));
                    let dh = ctx.decrypt(h);
                    let diff = max_err(&dh, &ctx.decrypt(&single));
                    let err = max_err(&dh, &shifted(&v, s as i64));
                    eprintln!(
                        "N=2^{} level {level} step {s}: hoisted vs rotate 2^{:.2}, vs shift \
                         2^{:.2}",
                        ctx.params.log_n(),
                        diff.log2(),
                        err.log2()
                    );
                    assert!(diff < 2f64.powi(-25), "step {s}: {diff:e}");
                    assert!(err < 2f64.powi(-20), "step {s}: {err:e}");
                    for (x, y) in h.polys().iter().zip(single.polys()) {
                        assert_eq!(x.limbs(), y.limbs(), "step {s}: not bit-identical");
                    }
                }
            }
            assert!(
                ctx.eval
                    .hoisted_rotate(&ctx.encrypt(&v, 0, &mut r), &[])
                    .unwrap()
                    .is_empty()
            );
        }
    }

    /// `conjugate` decrypts to the complex conjugate (end to end, complex slots), at the top
    /// level and level 0; conjugating twice is the identity.
    #[test]
    fn conjugate_flips_imaginary_parts() {
        for ctx in [small(), seal()] {
            let slots = ctx.params.slots();
            let (re, im) = (rand_vals(46, slots), rand_vals(47, slots));
            let z: Vec<Complex64> = re
                .iter()
                .zip(&im)
                .map(|(&a, &b)| Complex64::new(a, b))
                .collect();
            let want: Vec<Complex64> = z.iter().map(Complex64::conj).collect();
            let mut r = rng(48);
            for level in [ctx.params.max_level(), 0] {
                let pt = ctx
                    .enc
                    .encode_complex(&z, ctx.params.scale(), level)
                    .unwrap();
                let ct = ctx.encryptor.encrypt(&pt, &mut r);
                let cj = ctx.eval.conjugate(&ct).unwrap();
                assert_eq!((cj.len(), cj.level(), cj.scale()), (2, level, ct.scale()));
                let got = ctx.enc.decode_complex(&ctx.dec.decrypt(&cj));
                let err = max_err_complex(&got, &want);
                eprintln!(
                    "N=2^{} level {level}: conjugate error 2^{:.2}",
                    ctx.params.log_n(),
                    err.log2()
                );
                assert!(err < 2f64.powi(-20), "{err:e}");
                // The input really had imaginary parts, and they flipped sign.
                assert!(
                    got.iter()
                        .zip(&z)
                        .any(|(g, x)| x.im.abs() > 0.5 && g.im * x.im < 0.0)
                );
                let twice = ctx.eval.conjugate(&cj).unwrap();
                let back = ctx.enc.decode_complex(&ctx.dec.decrypt(&twice));
                assert!(max_err_complex(&back, &z) < 2f64.powi(-20));
            }
        }
    }

    /// Missing keys, foreign-basis keys and 3-component inputs are `InvalidParams`.
    #[test]
    fn rotation_key_errors() {
        let ctx = small();
        let mut r = rng(49);
        let top = ctx.params.max_level();
        let v = rand_vals(50, 8);
        let ct = ctx.encrypt(&v, top, &mut r);
        let e = &ctx.eval;
        let missing = HekitError::InvalidParams("missing galois key".into());
        assert_eq!(e.rotate(&ct, 3).unwrap_err(), missing);
        assert_eq!(e.hoisted_rotate(&ct, &[1, 3]).unwrap_err(), missing);
        // No Galois keys at all: rotations by a multiple of slots still work.
        let bare = Evaluator::new(Arc::clone(&ctx.params), None, None);
        assert_eq!(bare.rotate(&ct, 1).unwrap_err(), missing);
        assert_eq!(bare.hoisted_rotate(&ct, &[0, 1]).unwrap_err(), missing);
        assert_eq!(bare.conjugate(&ct).unwrap_err(), missing);
        assert!(bare.rotate(&ct, 0).is_ok());
        assert_eq!(bare.hoisted_rotate(&ct, &[0, 0]).unwrap().len(), 2);
        // Galois keys without the conjugation key.
        let mut kg = KeyGenerator::new(Arc::clone(&ctx.params), rng(51)).unwrap();
        let no_conj = Evaluator::new(Arc::clone(&ctx.params), None, Some(kg.galois_keys(&[1])));
        assert!(no_conj.rotate(&ct, 1).is_ok());
        assert_eq!(no_conj.conjugate(&ct).unwrap_err(), missing);
        // 3-component input.
        let t = e.mul_no_relin(&ct, &ct).unwrap();
        for res in [e.rotate(&t, 1).map(|_| ()), e.conjugate(&t).map(|_| ())] {
            assert!(matches!(res, Err(HekitError::InvalidParams(_))));
        }
        assert!(matches!(
            e.hoisted_rotate(&t, &[1]),
            Err(HekitError::InvalidParams(_))
        ));
        // Keys generated for an equal but separately constructed parameter set (different
        // basis `Arc`s): InvalidParams, not a panic or a wrong result.
        let other = Ctx::with_rotations(small_params(), 52, Some(&[1]));
        let foreign = Evaluator::new(
            Arc::clone(&ctx.params),
            other.eval.relin_key().cloned(),
            other.eval.galois_keys().cloned(),
        );
        for res in [
            foreign.rotate(&ct, 1).map(|_| ()),
            foreign.conjugate(&ct).map(|_| ()),
            foreign.hoisted_rotate(&ct, &[1]).map(|_| ()),
            foreign.relinearize(&t).map(|_| ()),
            foreign.mul(&ct, &ct).map(|_| ()),
        ] {
            assert!(matches!(res, Err(HekitError::InvalidParams(_))), "{res:?}");
        }
        // Foreign ciphertexts are rejected as before.
        let f = other.encrypt(&v, top, &mut r);
        assert!(matches!(e.rotate(&f, 1), Err(HekitError::InvalidParams(_))));
        assert!(matches!(
            e.hoisted_rotate(&f, &[1]),
            Err(HekitError::InvalidParams(_))
        ));
    }

    /// `mul` reports operand errors (level mismatch, level 0, component count) before a
    /// missing relinearization key.
    #[test]
    fn mul_error_order() {
        let ctx = small();
        let mut r = rng(53);
        let top = ctx.params.max_level();
        let v = rand_vals(54, 8);
        let (a, b) = (
            ctx.encrypt(&v, top, &mut r),
            ctx.encrypt(&v, top - 1, &mut r),
        );
        let z = ctx.encrypt(&v, 0, &mut r);
        let bare = Evaluator::new(Arc::clone(&ctx.params), None, None);
        assert_eq!(bare.mul(&a, &b).unwrap_err(), HekitError::LevelMismatch);
        assert_eq!(bare.mul(&z, &z).unwrap_err(), HekitError::NoLevelsLeft);
        assert!(matches!(
            bare.mul(&a, &a),
            Err(HekitError::InvalidParams(_))
        ));
    }
}
