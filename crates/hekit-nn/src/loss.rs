//! Loss gradients.
//!
//! [`Mse`]: for the loss `1/2 ||a_L - y||^2` (per sample, up to the constant folded into the
//! derivative factor `s_L` by Task 30) the output gradient is `delta_L = a_L - y`.
//!
//! # Scale and level contract
//!
//! `y` is a ciphertext (the client encrypts the labels). `Evaluator::sub` accepts scales within
//! `2^-10` and silently relabels, which compounds (see the `hekit-ckks` evaluator docs), so
//! [`Mse::grad`] requires the two scales to be **identical up to floating-point rounding**
//! (relative difference `<= 1e-12`, else `ScaleMismatch`). The client therefore encrypts `y` at
//! exactly the scale `pred` will have, which it can compute from the public parameters (the
//! activation's output scale is a deterministic function of the input scale and the primes).
//! Levels need not match: the higher operand is switched down (free) to the lower one's level.
//! The result has the lower level and `pred`'s scale. Cost: one subtraction, no relinearization,
//! no rescale, no level.

use hekit_ckks::{DiagonalEntry, HeOps};
use hekit_core::error::{HekitError, Result};

/// Relative scale difference accepted as "the same scale" (floating-point rounding only).
const SAME_SCALE: f64 = 1e-12;

/// Mean-squared-error loss; the gradient of `1/2 ||pred - target||^2` w.r.t. `pred`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mse;

impl Mse {
    /// `pred - target` (see the [module docs](self) for the scale/level contract).
    ///
    /// Errors: `ScaleMismatch` unless the scales agree to `1e-12` (relative); the operand
    /// errors of `Evaluator::sub` (foreign basis, ...).
    pub fn grad<E: HeOps>(&self, ev: &E, pred: &E::Ct, target: &E::Ct) -> Result<E::Ct> {
        let (a, b) = (pred.scale(), target.scale());
        if (a - b).abs() > SAME_SCALE * a.min(b) {
            return Err(HekitError::ScaleMismatch);
        }
        let level = pred.level().min(target.level());
        let p = ev.mod_switch_to_level(pred, level)?;
        let t = ev.mod_switch_to_level(target, level)?;
        ev.sub(&p, &t)
    }

    /// Multiplicative levels consumed by [`Mse::grad`]: 0.
    pub fn depth(&self) -> usize {
        0
    }
}
