//! Optimizers on encrypted weights (Task 28): [`Sgd`] and [`Nag`] (`docs/layout.md` §3–§4).
//!
//! # Constant folding (why `step` multiplies nothing on the critical path)
//!
//! On seal_2020 the Task 30 step uses all 9 levels (`4L + 1`); a separate multiplication of
//! the gradient by `lr`, `1 / n` or a NAG factor would cost a 10th. So every optimizer constant
//! that scales the gradient is folded into a plaintext the gradient already passes through:
//! [`Optimizer::grad_scale`] is the factor the gradient must carry when it reaches
//! [`Optimizer::step`]. The caller passes `c = opt.grad_scale() / n` (`n` valid samples) to
//! [`crate::dense::Dense::weight_grad_scaled`], which multiplies it into the lane-0 mask
//! plaintext of the batch sum (0 extra levels), or folds it into the derivative factor
//! `s_L` and passes `c = 1`. The gradient then arrives at exactly the weights' scale
//! (`Dense::weight_grad_scaled` targets `W.scale()`), and:
//!
//! - [`Sgd`]: `grad_scale = lr`, `W' = W - dw` (subtractions only).
//! - [`Nag`]: `grad_scale = lr (1 + mu)`, and with the scaled velocity `U = lr (1 + mu) V`
//!   (`h = dw`): `U' = mu U + h`, `W' = W - h - (mu^2 / (1 + mu)) U`. The two scalar
//!   multiplications act on `U` only ([`ckks_train_ckks::Evaluator::mul_scalar`] with the exact
//!   target scale), which sits at least one level above `h`, so `W'` and `U'` land at `h`'s
//!   level: no level on the critical path.
//!
//! The updated weights keep their packed, pre-rotated, replicated layout and their exact
//! scale, so the next [`crate::dense::Dense::forward`] needs no repacking (only levels: the
//! client refresh of Task 30 restores them).
//!
//! The plaintext NAG this implements (Nesterov in the form of `docs/layout.md` §4, `V_0 = 0`):
//! `V' = mu V + g`, `W' = W - lr (g + mu V')`; expanding gives
//! `W' = W - lr (1 + mu) g - lr mu^2 V`, which is the `U` form above. Weight decay and bias
//! momentum are not part of this API (biases are updated by `b - db` with
//! [`crate::dense::Dense::bias_grad_scaled`]).

use ckks_train_ckks::{Ciphertext, DiagonalEntry, Diagonals, Evaluator, HeCiphertext, HeOps};
use ckks_train_core::error::{HekitError, Result};

use crate::dense::{EncWeights, check_uniform};

/// An update rule for one layer's encrypted weights.
pub trait Optimizer<E: HeOps = Evaluator> {
    /// The factor the gradient passed to [`Optimizer::step`] must already carry (folded into a
    /// plaintext of the gradient computation, see the [module docs](self)): pass
    /// `grad_scale() / n` as `c` to [`crate::dense::Dense::weight_grad_scaled`].
    fn grad_scale(&self) -> f64;

    /// Updates `w` in place with the folded gradient `dw = grad_scale() * dW` (as returned by
    /// [`crate::dense::Dense::weight_grad_scaled`]): the result is at `dw`'s level with
    /// `w`'s scale and layout.
    ///
    /// Errors, all before any homomorphic work (`w` and the optimizer state are unchanged):
    /// `InvalidParams` if `dw` does not have `w`'s shape (`d`, stride); `LevelMismatch` if
    /// the entries of `dw` do not share one level or `w` is below `dw`'s level;
    /// `ScaleMismatch` unless every scale equals `w.scale()` exactly; and the optimizer's own
    /// checks.
    fn step(&mut self, ev: &E, w: &mut EncWeights<E::Ct>, dw: &Diagonals<E::Ct>) -> Result<()>;
}

/// The checks shared by the optimizers: shape, uniform level and scale of `dw`, `w` at or
/// above `dw`'s level with exactly `dw`'s scale. Returns `dw`'s level.
fn check_update<C: HeCiphertext + Clone>(w: &EncWeights<C>, dw: &Diagonals<C>) -> Result<usize> {
    w.check_same_shape(dw)?;
    check_uniform(dw)?;
    let level = dw.entries()[0].level();
    if w.level() < level {
        return Err(HekitError::LevelMismatch);
    }
    if w.scale() != dw.entries()[0].scale() {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(level)
}

/// `w` switched down to `level` (free), entry by entry.
fn weights_at<E: HeOps>(ev: &E, w: &EncWeights<E::Ct>, level: usize) -> Result<Vec<E::Ct>> {
    w.diagonals()
        .entries()
        .iter()
        .map(|c| ev.mod_switch_to_level(c, level))
        .collect()
}

/// Rebuilds diagonals of `like`'s shape from `entries`.
fn diags_like<C: DiagonalEntry>(like: &Diagonals<C>, entries: Vec<C>) -> Result<Diagonals<C>> {
    Diagonals::from_prerotated(like.d(), like.stride(), entries)
}

/// Plain SGD, `W' = W - lr dW`, with `lr` folded into the gradient
/// ([`Optimizer::grad_scale`] `= lr`): the step itself is `m` subtractions (no product, no
/// level, no counted op).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sgd {
    /// The learning rate.
    pub lr: f64,
}

impl Sgd {
    /// The gradient constant `lr` ([`Optimizer::grad_scale`]; inherent so that it needs no
    /// evaluator type).
    pub fn grad_scale(&self) -> f64 {
        self.lr
    }

    /// The bias gradient constant `lr` (`LayerOptimizer::bias_grad_scale`).
    pub fn bias_grad_scale(&self) -> f64 {
        self.lr
    }
}

impl<E: HeOps> Optimizer<E> for Sgd {
    fn grad_scale(&self) -> f64 {
        Sgd::grad_scale(self)
    }

    fn step(&mut self, ev: &E, w: &mut EncWeights<E::Ct>, dw: &Diagonals<E::Ct>) -> Result<()> {
        let level = check_update(w, dw)?;
        let entries = weights_at(ev, w, level)?
            .iter()
            .zip(dw.entries())
            .map(|(we, g)| ev.sub(we, g))
            .collect::<Result<Vec<_>>>()?;
        w.set_diagonals(diags_like(dw, entries)?)
    }
}

/// Nesterov accelerated gradient for one layer (one instance per layer: it holds that layer's
/// velocity), in the scaled-velocity form of the [module docs](self):
/// [`Optimizer::grad_scale`] `= lr (1 + mu)`, `U' = mu U + h`,
/// `W' = W - h - (mu^2 / (1 + mu)) U` with `h` the folded gradient.
///
/// The first step (no velocity) is `U' = h`, `W' = W - h` (subtractions only). Later steps
/// cost `2 m` [`Evaluator::mul_scalar`] (counted as ct-pt) and `2 m` rescales on `U`, which
/// must be at least one level above `h` (`LevelMismatch` otherwise): `U'` lands at `h`'s
/// level, like `W'`, so the client must refresh the velocity together with the weights
/// ([`Nag::set_velocity`]).
///
/// `C` is the ciphertext type of the velocity ([`Ciphertext`] by default; [`Nag::new`] is
/// the CPU constructor, [`Nag::new_for`] the one for any backend).
#[derive(Debug, Clone)]
pub struct Nag<C = Ciphertext> {
    /// The learning rate.
    pub lr: f64,
    /// The momentum `mu`.
    pub momentum: f64,
    velocity: Option<Diagonals<C>>,
}

impl Nag {
    /// A NAG optimizer with zero velocity (CPU ciphertexts).
    pub fn new(lr: f64, momentum: f64) -> Nag {
        Nag::new_for(lr, momentum)
    }
}

impl<C> Nag<C> {
    /// A NAG optimizer with zero velocity for the ciphertexts `C` of any backend
    /// (`Nag::<E::Ct>::new_for(lr, mu)`).
    pub fn new_for(lr: f64, momentum: f64) -> Nag<C> {
        Nag {
            lr,
            momentum,
            velocity: None,
        }
    }

    /// The scaled velocity `U = lr (1 + mu) V` (pre-rotated diagonals in the weights' layout
    /// and scale), `None` before the first step.
    pub fn velocity(&self) -> Option<&Diagonals<C>> {
        self.velocity.as_ref()
    }

    /// Replaces the velocity, e.g. by a client-refreshed copy at the top level. It must have
    /// the weights' shape and scale; checked at the next [`Optimizer::step`].
    pub fn set_velocity(&mut self, velocity: Option<Diagonals<C>>) {
        self.velocity = velocity;
    }

    /// Takes the velocity out (e.g. to refresh it), leaving `None`.
    pub fn take_velocity(&mut self) -> Option<Diagonals<C>> {
        self.velocity.take()
    }
}

impl<C> Nag<C> {
    /// The gradient constant `lr (1 + mu)` ([`Optimizer::grad_scale`]; inherent so that it
    /// needs no evaluator type).
    pub fn grad_scale(&self) -> f64 {
        self.lr * (1.0 + self.momentum)
    }

    /// The bias gradient constant `lr` (`LayerOptimizer::bias_grad_scale`).
    pub fn bias_grad_scale(&self) -> f64 {
        self.lr
    }
}

impl<E: HeOps> Optimizer<E> for Nag<E::Ct> {
    fn grad_scale(&self) -> f64 {
        Nag::grad_scale(self)
    }

    fn step(&mut self, ev: &E, w: &mut EncWeights<E::Ct>, dw: &Diagonals<E::Ct>) -> Result<()> {
        let level = check_update(w, dw)?;
        let mu = self.momentum;
        let kappa = mu * mu / (1.0 + mu);
        let target = w.scale();
        let ws = weights_at(ev, w, level)?;
        let (w_new, u_new) = match &self.velocity {
            None => {
                let w_new = ws
                    .iter()
                    .zip(dw.entries())
                    .map(|(we, h)| ev.sub(we, h))
                    .collect::<Result<Vec<_>>>()?;
                (w_new, dw.entries().to_vec())
            }
            Some(u) => {
                w.check_same_shape(u)?;
                check_uniform(u)?;
                if u.entries()[0].level() <= level {
                    return Err(HekitError::LevelMismatch);
                }
                if u.entries()[0].scale() != target {
                    return Err(HekitError::ScaleMismatch);
                }
                if !(mu.is_finite() && kappa.is_finite()) {
                    return Err(HekitError::InvalidParams(format!(
                        "momentum {mu} gives a non-finite factor"
                    )));
                }
                // The target is reachable from level + 1 (checked before any work).
                ev.exact_product_scale(level + 1, target)?;
                let mut w_new = Vec::with_capacity(ws.len());
                let mut u_new = Vec::with_capacity(ws.len());
                for ((we, h), ue) in ws.iter().zip(dw.entries()).zip(u.entries()) {
                    let ue = ev.mod_switch_to_level(ue, level + 1)?;
                    let mu_u = ev.rescale(&ev.mul_scalar(&ue, mu, target)?)?;
                    let k_u = ev.rescale(&ev.mul_scalar(&ue, kappa, target)?)?;
                    w_new.push(ev.sub(&ev.sub(we, h)?, &k_u)?);
                    u_new.push(ev.add(&mu_u, h)?);
                }
                (w_new, u_new)
            }
        };
        w.set_diagonals(diags_like(dw, w_new)?)?;
        self.velocity = Some(diags_like(dw, u_new)?);
        Ok(())
    }
}
