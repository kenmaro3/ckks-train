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
//! - [`Nag`] at constant rate: `grad_scale = lr (1 + mu)`, with velocity `U = lr (1 + mu) V`
//!   (`h = dw`): `U' = mu U + h`, `W' = W - h - (mu^2 / (1 + mu)) U`. The two scalar
//!   multiplications act on `U` only ([`ckks_train_ckks::Evaluator::mul_scalar`] with the exact
//!   target scale), which sits at least one level above `h`, so `W'` and `U'` land at `h`'s
//!   level: no level on the critical path.
//!
//! Scheduled NAG fixes `U = r (1 + mu) V` at reference rate `r`. When `lr != r`, a second
//! gradient `h_ref = r (1 + mu) g` feeds `U' = mu U + h_ref`, and the weight's scalar is
//! `(mu^2 / (1 + mu)) * (lr / r)`. [`Optimizer::reference_grad_scale`] and
//! [`Optimizer::step_with_reference_gradient`] expose this branch. The trainer shares
//! products/reductions with [`Dense::weight_grad_scaled_consts`](crate::dense::Dense::weight_grad_scaled_consts):
//! each extra replicated gradient costs `m` ct-pt, `m` rescales and `m log2(B)` rotations,
//! but no extra depth. Configure `r` as the schedule maximum using [`Nag::set_lr_reference`]
//! before initialization; later rates must be positive and at most `r`.
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

#[cfg(test)]
use crate::diagonal_plan::observation::{self, Handle as ObservationHandle};

use ckks_train_ckks::{Ciphertext, DiagonalEntry, Diagonals, Evaluator, HeCiphertext, HeOps};
use ckks_train_core::error::{HekitError, Result};

use crate::dense::{EncWeights, check_uniform};

/// An update rule for one layer's encrypted weights.
pub trait Optimizer<E: HeOps = Evaluator> {
    /// The factor the gradient passed to [`Optimizer::step`] must already carry (folded into a
    /// plaintext of the gradient computation, see the [module docs](self)): pass
    /// `grad_scale() / n` as `c` to [`crate::dense::Dense::weight_grad_scaled`].
    fn grad_scale(&self) -> f64;

    /// An additional folded gradient constant for a fixed-basis optimizer state.
    /// The default needs only [`Optimizer::grad_scale`]. The trainer queries every layer
    /// before forward work, so configuration errors leave the entire step untouched.
    fn reference_grad_scale(&self) -> Result<Option<f64>> {
        Ok(None)
    }

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

    /// Update with an optional second gradient at [`Optimizer::reference_grad_scale`].
    /// Both gradients have the same shape, level and exact scale. The default preserves
    /// the legacy update for `None` and rejects an unsupported extra gradient before work.
    fn step_with_reference_gradient(
        &mut self,
        ev: &E,
        w: &mut EncWeights<E::Ct>,
        dw: &Diagonals<E::Ct>,
        reference: Option<&Diagonals<E::Ct>>,
    ) -> Result<()> {
        if reference.is_some() {
            return Err(HekitError::InvalidParams(
                "optimizer does not accept a reference gradient".into(),
            ));
        }
        self.step(ev, w, dw)
    }
}

pub(crate) fn check_lr(lr: f64) -> Result<()> {
    if !(lr.is_finite() && lr > 0.0) {
        return Err(HekitError::InvalidParams(format!(
            "learning rate {lr} must be finite and positive"
        )));
    }
    Ok(())
}

/// Shared exact evaluation order for the scaled-velocity NAG coefficient.
pub(crate) fn nag_kappa(momentum: f64, rate_factor: f64) -> f64 {
    momentum * momentum / (1.0 + momentum) * rate_factor
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
    /// Changes the rate of following updates. Rejects nonfinite/nonpositive rates without
    /// changing this optimizer; SGD has no velocity or reference-rate ceiling.
    pub fn set_lr(&mut self, lr: f64) -> Result<()> {
        check_lr(lr)?;
        self.lr = lr;
        Ok(())
    }

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
/// `W' = W - h - (mu^2 / (1 + mu)) U` at the reference rate, with `h` the folded gradient.
/// For changing rates use the additional reference gradient and scaled momentum contribution
/// described in the module docs. `step` rejects a required missing reference gradient.
///
/// The first step (no velocity) is `U' = h_ref`, `W' = W - h` (subtractions only). Later steps
/// cost `2 m` [`Evaluator::mul_scalar`] (counted as ct-pt) and `2 m` rescales on `U`, which
/// must be at least one level above `h` (`LevelMismatch` otherwise): `U'` lands at `h`'s
/// level, like `W'`, so the client must refresh the velocity together with the weights
/// ([`Nag::set_velocity`]).
///
/// `C` is the ciphertext type of the velocity ([`Ciphertext`] by default; [`Nag::new`] is
/// the CPU constructor, [`Nag::new_for`] the one for any backend).
#[derive(Debug, Clone)]
pub struct Nag<C = Ciphertext> {
    /// Current learning rate. Prefer [`Nag::set_lr`], which validates it before assignment.
    /// Direct field writes are checked by the reference-gradient query and update.
    pub lr: f64,
    /// The momentum `mu`. Changing it after velocity initialization is unsupported and is
    /// rejected by the next query/update, even when the ciphertext state was taken out.
    pub momentum: f64,
    velocity: Option<Diagonals<C>>,
    lr_reference: f64,
    basis_momentum: Option<f64>,
    #[cfg(test)]
    pub(crate) observer: Option<ObservationHandle>,
    #[cfg(test)]
    pub(crate) observer_layer: Option<usize>,
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
            lr_reference: lr,
            basis_momentum: None,
            #[cfg(test)]
            observer: None,
            #[cfg(test)]
            observer_layer: None,
        }
    }

    /// The scaled velocity `U = lr_reference (1 + mu) V` (pre-rotated diagonals in the weights' layout
    /// and scale), `None` before the first step.
    pub fn velocity(&self) -> Option<&Diagonals<C>> {
        self.velocity.as_ref()
    }

    /// Replaces the velocity, e.g. by a client-refreshed copy at the top level. It must have
    /// the weights' shape and scale; checked at the next update. Supplied values must use
    /// the existing reference-rate/momentum basis (or the current basis for the first supply).
    /// Removing state does not unlock its basis; encrypted values cannot be inspected here.
    pub fn set_velocity(&mut self, velocity: Option<Diagonals<C>>) {
        if velocity.is_some() && self.basis_momentum.is_none() {
            self.basis_momentum = Some(self.momentum);
        }
        self.velocity = velocity;
    }

    /// Takes the velocity out (e.g. to refresh it), leaving `None`.
    pub fn take_velocity(&mut self) -> Option<Diagonals<C>> {
        self.velocity.take()
    }

    /// Changes the current rate, retaining the fixed-basis velocity. The rate must be
    /// finite, positive and at most [`Nag::lr_reference`]. Invalid factors or a changed
    /// momentum basis also fail before mutation. Configure the schedule maximum first.
    pub fn set_lr(&mut self, lr: f64) -> Result<()> {
        self.validate_rate(lr)?;
        self.lr = lr;
        Ok(())
    }

    /// Sets the velocity's reference rate before its first initialization/supply. It must
    /// be finite, positive and at least the current rate, with finite positive folded
    /// gradient factors. Taking or clearing velocity does not permit a later basis change.
    pub fn set_lr_reference(&mut self, lr_reference: f64) -> Result<()> {
        if self.basis_momentum.is_some() {
            return Err(HekitError::InvalidParams(
                "reference learning rate is locked after velocity initialization".into(),
            ));
        }
        self.check_config(self.lr, lr_reference)?;
        self.lr_reference = lr_reference;
        Ok(())
    }

    /// The immutable reference rate of initialized velocity; initially the constructor rate.
    pub fn lr_reference(&self) -> f64 {
        self.lr_reference
    }

    pub(crate) fn validate_rate(&self, lr: f64) -> Result<()> {
        self.check_config(lr, self.lr_reference)
    }

    fn check_config(&self, lr: f64, reference: f64) -> Result<()> {
        check_lr(lr)?;
        check_lr(reference)?;
        if lr > reference {
            return Err(HekitError::InvalidParams(format!(
                "learning rate {lr} above velocity reference {reference}"
            )));
        }
        let mu = self.momentum;
        if self.basis_momentum.is_some_and(|basis| basis != mu) {
            return Err(HekitError::InvalidParams(
                "momentum differs from initialized velocity basis".into(),
            ));
        }
        let current_factor = lr * (1.0 + mu);
        let reference_factor = reference * (1.0 + mu);
        let kappa = mu * mu / (1.0 + mu);
        if !(mu.is_finite()
            && kappa.is_finite()
            && current_factor.is_finite()
            && current_factor > 0.0
            && reference_factor.is_finite()
            && reference_factor > 0.0
            && lr / reference != 0.0)
        {
            return Err(HekitError::InvalidParams(
                "NAG rate and momentum must give finite usable folded factors".into(),
            ));
        }
        Ok(())
    }

    /// The extra reference-rate gradient constant, when current and reference rates differ.
    /// Validates rate/factor/basis metadata even when no second gradient is needed.
    pub fn reference_grad_scale(&self) -> Result<Option<f64>> {
        self.validate_rate(self.lr)?;
        Ok((self.lr != self.lr_reference).then_some(self.lr_reference * (1.0 + self.momentum)))
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

    fn reference_grad_scale(&self) -> Result<Option<f64>> {
        Nag::reference_grad_scale(self)
    }

    fn step(&mut self, ev: &E, w: &mut EncWeights<E::Ct>, dw: &Diagonals<E::Ct>) -> Result<()> {
        self.step_with_reference_gradient(ev, w, dw, None)
    }

    fn step_with_reference_gradient(
        &mut self,
        ev: &E,
        w: &mut EncWeights<E::Ct>,
        dw: &Diagonals<E::Ct>,
        reference: Option<&Diagonals<E::Ct>>,
    ) -> Result<()> {
        let needs_reference = self.reference_grad_scale()?.is_some();
        if needs_reference != reference.is_some() {
            return Err(HekitError::InvalidParams(
                "NAG reference gradient presence must match current/reference rate difference"
                    .into(),
            ));
        }
        let level = check_update(w, dw)?;
        let h_ref = reference.unwrap_or(dw);
        if check_update(w, h_ref)? != level {
            return Err(HekitError::LevelMismatch);
        }
        let mu = self.momentum;
        let factor = if needs_reference {
            self.lr / self.lr_reference
        } else {
            1.0
        };
        let kappa = nag_kappa(mu, factor);
        #[cfg(test)]
        {
            observation::coefficient(self.observer.as_ref(), "mu", self.observer_layer, 0, mu);
            observation::coefficient(
                self.observer.as_ref(),
                "kappa",
                self.observer_layer,
                0,
                kappa,
            );
        }
        let target = w.scale();
        if let Some(u) = &self.velocity {
            w.check_same_shape(u)?;
            check_uniform(u)?;
            if u.entries()[0].level() <= level {
                return Err(HekitError::LevelMismatch);
            }
            if u.entries()[0].scale() != target {
                return Err(HekitError::ScaleMismatch);
            }
            ev.exact_product_scale(level + 1, target)?;
        }
        let ws = weights_at(ev, w, level)?;
        let (w_new, u_new) = match &self.velocity {
            None => {
                let w_new = ws
                    .iter()
                    .zip(dw.entries())
                    .map(|(we, h)| ev.sub(we, h))
                    .collect::<Result<Vec<_>>>()?;
                (w_new, h_ref.entries().to_vec())
            }
            Some(u) => {
                let mut w_new = Vec::with_capacity(ws.len());
                let mut u_new = Vec::with_capacity(ws.len());
                #[cfg_attr(not(test), allow(clippy::unused_enumerate_index))]
                for (_position, (((we, h), ue), hr)) in ws
                    .iter()
                    .zip(dw.entries())
                    .zip(u.entries())
                    .zip(h_ref.entries())
                    .enumerate()
                {
                    let ue = ev.mod_switch_to_level(ue, level + 1)?;
                    let mu_u = ev.rescale(&ev.mul_scalar(&ue, mu, target)?)?;
                    let k_u = ev.rescale(&ev.mul_scalar(&ue, kappa, target)?)?;
                    #[cfg(test)]
                    {
                        observation::value(
                            self.observer.as_ref(),
                            "mu_velocity",
                            None,
                            self.observer_layer,
                            _position,
                            &mu_u,
                        );
                        observation::value(
                            self.observer.as_ref(),
                            "kappa_velocity",
                            None,
                            self.observer_layer,
                            _position,
                            &k_u,
                        );
                    }
                    w_new.push(ev.sub(&ev.sub(we, h)?, &k_u)?);
                    u_new.push(ev.add(&mu_u, hr)?);
                }
                (w_new, u_new)
            }
        };
        let new_velocity = diags_like(dw, u_new)?;
        w.set_diagonals(diags_like(dw, w_new)?)?;
        #[cfg(test)]
        observation::values(
            self.observer.as_ref(),
            "velocity_updated",
            None,
            self.observer_layer,
            new_velocity.entries(),
        );
        self.velocity = Some(new_velocity);
        self.basis_momentum = Some(mu);
        Ok(())
    }
}
