//! CKKS plaintexts: an RNS polynomial in NTT form together with its scale and level.

use hekit_core::error::{HekitError, Result};
use hekit_core::poly::RnsPoly;

/// A CKKS plaintext `m(X)` over `Z_{Q_level}[X]/(X^N + 1)`, `Q_level = q_0 * ... * q_level`.
///
/// As in SEAL 4.1, the polynomial is always stored in **NTT form** (bit-reversed evaluation
/// order of [`hekit_core::ntt::NttTable::forward`]), so it can be multiplied with NTT-form
/// ciphertexts without conversion. `level` always equals `poly.level()`.
#[derive(Debug, Clone)]
pub struct Plaintext {
    poly: RnsPoly,
    scale: f64,
    level: usize,
}

impl Plaintext {
    /// Wraps an NTT-form polynomial with its scale; the level is `poly.level()`.
    ///
    /// Fails with `InvalidParams` if `poly` is in coefficient form or `scale` is not a
    /// positive finite number.
    pub fn new(poly: RnsPoly, scale: f64) -> Result<Plaintext> {
        if !poly.is_ntt_form() {
            return Err(HekitError::InvalidParams(
                "plaintext polynomial must be in NTT form".into(),
            ));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(HekitError::InvalidParams(format!(
                "plaintext scale {scale} must be positive and finite"
            )));
        }
        let level = poly.level();
        Ok(Plaintext { poly, scale, level })
    }

    /// The underlying polynomial (NTT form).
    pub fn poly(&self) -> &RnsPoly {
        &self.poly
    }

    /// Consumes the plaintext and returns its polynomial (NTT form).
    pub fn into_poly(self) -> RnsPoly {
        self.poly
    }

    /// The scale Delta the values were multiplied by.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The level: the polynomial has `level + 1` limbs (primes `q_0..=q_level`).
    pub fn level(&self) -> usize {
        self.level
    }
}
