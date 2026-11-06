//! CKKS ciphertexts: a list of NTT-form RNS polynomials with a scale and a level.

use std::sync::Arc;

use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::poly::RnsPoly;

/// A CKKS ciphertext `(c_0, c_1, ..., c_{k-1})` over `R_{Q_level}`, decrypting to
/// `c_0 + c_1 s + ... + c_{k-1} s^{k-1}`.
///
/// `k = 2` for fresh / relinearized ciphertexts and `k = 3` after a multiplication before
/// relinearization. As in SEAL 4.1 every component is in **NTT form**; all components share
/// the data basis `q_0..=q_level` (the `basis_q` of the parameters, same `Arc`), and `level`
/// always equals the level of every component. `Debug` prints the shape (`n`, `len`,
/// `level`, `scale`), not the residues.
#[derive(Clone)]
pub struct Ciphertext {
    c: Vec<RnsPoly>,
    scale: f64,
    level: usize,
}

impl std::fmt::Debug for Ciphertext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ciphertext")
            .field("n", &self.c[0].basis().n())
            .field("len", &self.c.len())
            .field("level", &self.level)
            .field("scale", &self.scale)
            .finish()
    }
}

impl Ciphertext {
    /// Wraps components with their scale; the level is the common level of the components.
    ///
    /// Errors: `InvalidParams` for fewer than 2 components, a component in coefficient form,
    /// components over different bases, or a scale that is not positive and finite;
    /// `LevelMismatch` for components at different levels.
    pub fn new(c: Vec<RnsPoly>, scale: f64) -> Result<Ciphertext> {
        let bad = |m: &str| Err(HekitError::InvalidParams(m.into()));
        if c.len() < 2 {
            return bad("a ciphertext needs at least 2 components");
        }
        if c.iter().any(|p| !p.is_ntt_form()) {
            return bad("ciphertext components must be in NTT form");
        }
        if c.iter().any(|p| !Arc::ptr_eq(p.basis(), c[0].basis())) {
            return bad("ciphertext components over different bases");
        }
        let level = c[0].level();
        if c.iter().any(|p| p.level() != level) {
            return Err(HekitError::LevelMismatch);
        }
        if !(scale.is_finite() && scale > 0.0) {
            return bad("ciphertext scale must be positive and finite");
        }
        Ok(Ciphertext { c, scale, level })
    }

    /// The components `c_0, c_1, ...` (NTT form).
    pub fn polys(&self) -> &[RnsPoly] {
        &self.c
    }

    /// Consumes the ciphertext and returns its components (NTT form).
    pub fn into_polys(self) -> Vec<RnsPoly> {
        self.c
    }

    /// Number of components (2, or 3 before relinearization).
    pub fn len(&self) -> usize {
        self.c.len()
    }

    /// Always `false`: a ciphertext has at least 2 components.
    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    /// The scale Delta of the encrypted message.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The level: every component has `level + 1` limbs (primes `q_0..=q_level`).
    pub fn level(&self) -> usize {
        self.level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ckks_train_core::rns::RnsBasis;

    fn basis() -> Arc<RnsBasis> {
        let qs = ckks_train_core::primes::ntt_primes(30, 8, 3).unwrap();
        RnsBasis::new(8, &qs).unwrap()
    }

    fn ntt_zero(b: &Arc<RnsBasis>, level: usize) -> RnsPoly {
        let mut p = RnsPoly::zero(b, level);
        p.to_ntt();
        p
    }

    #[test]
    fn new_validates() {
        let b = basis();
        let ct = Ciphertext::new(vec![ntt_zero(&b, 2), ntt_zero(&b, 2)], 4.0).unwrap();
        assert_eq!((ct.len(), ct.level(), ct.scale()), (2, 2, 4.0));
        assert!(!ct.is_empty());
        assert!(Ciphertext::new(vec![ntt_zero(&b, 1)], 4.0).is_err());
        assert_eq!(
            Ciphertext::new(vec![ntt_zero(&b, 2), ntt_zero(&b, 1)], 4.0).unwrap_err(),
            HekitError::LevelMismatch
        );
        assert!(Ciphertext::new(vec![ntt_zero(&b, 1), RnsPoly::zero(&b, 1)], 4.0).is_err());
        assert!(Ciphertext::new(vec![ntt_zero(&b, 1), ntt_zero(&basis(), 1)], 4.0).is_err());
        assert!(Ciphertext::new(vec![ntt_zero(&b, 1), ntt_zero(&b, 1)], 0.0).is_err());
        assert!(Ciphertext::new(vec![ntt_zero(&b, 1), ntt_zero(&b, 1)], f64::NAN).is_err());
    }
}
