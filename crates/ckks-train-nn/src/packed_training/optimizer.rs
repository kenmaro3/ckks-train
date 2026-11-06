//! Immutable, generation-bound encrypted matrix update preparation.
use super::types::check_ct;
use super::{LayerGeometry, PackedGradient, PackedWeights, invalid};
use ckks_train_ckks::{DiagonalEntry, HeCiphertext, HeOps};
use ckks_train_core::error::{HekitError, Result};
use std::sync::Arc;

type TracedUpdate<C> = (PackedUpdateCandidate<C>, Vec<(usize, f64)>);

#[derive(Clone, Copy, Debug)]
enum Policy {
    Sgd,
    Nag { momentum: f64, reference: f64 },
}

/// Immutable optimizer policy. NAG's fixed reference and momentum cannot be rebound.
#[derive(Clone, Copy, Debug)]
pub struct PackedOptimizer(Policy);
impl PackedOptimizer {
    /// SGD has no reference ceiling. Rates are checked when a request is made.
    pub fn sgd() -> Self {
        Self(Policy::Sgd)
    }

    /// NAG with `0 <= momentum < 1` and a finite positive fixed reference rate.
    pub fn nag(momentum: f64, reference: f64) -> Result<Self> {
        if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
            return invalid("packed NAG momentum must be in [0,1)");
        }
        let policy = Self(Policy::Nag {
            momentum,
            reference,
        });
        policy.coefficients(reference, 1)?;
        Ok(policy)
    }

    fn coefficients(self, rate: f64, samples: usize) -> Result<(Vec<f64>, Option<f64>)> {
        usable(rate)?;
        if samples == 0 {
            return invalid("packed update needs a positive actual sample count");
        }
        let n = samples as f64;
        match self.0 {
            Policy::Sgd => {
                let factor = rate / n;
                usable(factor)?;
                Ok((vec![factor], None))
            }
            Policy::Nag {
                momentum: mu,
                reference: r,
            } => {
                usable(r)?;
                if rate > r {
                    return invalid("packed NAG rate exceeds fixed reference");
                }
                let current = rate * (1.0 + mu);
                let reference = r * (1.0 + mu);
                usable(current)?;
                usable(reference)?;
                let ratio = rate / r;
                usable(ratio)?;
                let kappa = mu * mu / (1.0 + mu) * ratio;
                if mu != 0.0 {
                    usable(kappa)?;
                }
                let current = current / n;
                let reference = reference / n;
                usable(current)?;
                usable(reference)?;
                Ok((
                    if rate == r {
                        vec![current]
                    } else {
                        vec![current, reference]
                    },
                    Some(kappa),
                ))
            }
        }
    }
}

fn usable(value: f64) -> Result<()> {
    if value.is_normal() && value > 0.0 {
        Ok(())
    } else {
        invalid("packed optimizer coefficient must be finite positive and non-subnormal")
    }
}

/// Matrix-only encrypted state. No bias, decryptor, refresh callback or plaintext velocity.
/// Geometry, optimizer basis and generation are immutable; candidates replace whole states.
#[derive(Debug)]
pub struct PackedLayerState<C> {
    weights: PackedWeights<C>,
    velocity: Option<PackedWeights<C>>,
    policy: PackedOptimizer,
    update_index: u64,
    generation: Arc<()>,
}
impl<C: HeCiphertext + Clone> PackedLayerState<C> {
    /// Checks the packed weight context and exact Delta. NAG starts from zero velocity.
    /// The caller asserts the encrypted active matrix and its zero structural padding.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        weights: PackedWeights<C>,
        policy: PackedOptimizer,
    ) -> Result<Self> {
        validate_weights(ev, &weights)?;
        Ok(Self {
            weights,
            velocity: None,
            policy,
            update_index: 0,
            generation: Arc::new(()),
        })
    }
    /// Current encrypted weights.
    pub fn weights(&self) -> &PackedWeights<C> {
        &self.weights
    }
    /// Fixed-reference `U = reference * (1 + momentum) * V`, absent before the first step.
    pub fn velocity(&self) -> Option<&PackedWeights<C>> {
        self.velocity.as_ref()
    }
    /// Number of completed updates in this state.
    pub fn update_index(&self) -> u64 {
        self.update_index
    }

    /// Binds checked factors and actual valid sample count to this state's generation.
    /// Use these factors in `outer_sum_scaled_masks`; equal rates need only one branch.
    pub fn request(&self, rate: f64, valid_samples: usize) -> Result<PackedUpdateRequest> {
        let (factors, kappa) = self.policy.coefficients(rate, valid_samples)?;
        self.update_index
            .checked_add(1)
            .ok_or_else(|| HekitError::InvalidParams("packed update index overflow".into()))?;
        Ok(PackedUpdateRequest {
            generation: self.generation.clone(),
            update_index: self.update_index,
            layer: self.weights.layer().clone(),
            valid_samples,
            factors,
            kappa,
        })
    }

    /// Prepares complete new W/U without changing this state. All structural/scale/level
    /// validation precedes counted work. Unexpected backend errors may advance counters.
    /// A future stage must prepare every matrix and bias before assigning any candidate.
    pub fn prepare<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        gradients: &PackedUpdateGradients<C>,
    ) -> Result<PackedUpdateCandidate<C>> {
        self.prepare_traced(ev, gradients)
            .map(|(candidate, _)| candidate)
    }

    // Metadata from actual scalar outputs; no additional HE work or retained ciphertexts.
    pub(super) fn prepare_traced<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        gradients: &PackedUpdateGradients<C>,
    ) -> Result<TracedUpdate<C>> {
        let mut scalar_history = Vec::new();
        let req = &gradients.request;
        if !Arc::ptr_eq(&self.generation, &req.generation)
            || self.update_index != req.update_index
            || self.weights.layer() != &req.layer
        {
            return invalid("packed update belongs to a stale or different state generation");
        }
        validate_weights(ev, &self.weights)?;
        req.check_branches(&gradients.branches)?;
        let h = gradients.branches[0].ciphertext();
        let level = h.level();
        for g in &gradients.branches {
            check_ct(ev, g.ciphertext(), req.layer.layout().slots())?;
            if g.ciphertext().scale() != ev.params().scale() {
                return Err(HekitError::ScaleMismatch);
            }
            if g.ciphertext().level() != level {
                return Err(HekitError::LevelMismatch);
            }
        }
        if let Some(u) = &self.velocity {
            validate_weights(ev, u)?;
            if u.layer() != self.weights.layer() {
                return invalid("packed velocity layer differs from weights");
            }
            if u.ciphertext().level() <= level {
                return Err(HekitError::NoLevelsLeft);
            }
            let Policy::Nag { momentum, .. } = self.policy.0 else {
                return invalid("SGD cannot have velocity");
            };
            // Match generic mul_scalar's exact-Delta path before either counted branch.
            scalar_preflight(ev, level + 1, momentum)?;
            scalar_preflight(ev, level + 1, req.kappa.expect("NAG request"))?;
        }
        let subtract = |a: &C, b: &C| -> Result<C> {
            let l = a.level().min(b.level());
            ev.sub(
                &ev.mod_switch_to_level(a, l)?,
                &ev.mod_switch_to_level(b, l)?,
            )
        };
        let href = gradients
            .branches
            .last()
            .expect("checked branches")
            .ciphertext();
        let (new_w, new_u) = match self.policy.0 {
            Policy::Sgd => (subtract(self.weights.ciphertext(), h)?, None),
            Policy::Nag { momentum, .. } => {
                if let Some(u) = &self.velocity {
                    let u = ev.mod_switch_to_level(u.ciphertext(), level + 1)?;
                    let mu_u = ev.rescale(&ev.mul_scalar(&u, momentum, ev.params().scale())?)?;
                    let k_u = ev.rescale(&ev.mul_scalar(
                        &u,
                        req.kappa.expect("NAG request"),
                        ev.params().scale(),
                    )?)?;
                    scalar_history.push((mu_u.level(), mu_u.scale()));
                    scalar_history.push((k_u.level(), k_u.scale()));
                    // Both scalar branches use OLD U; the f64 reference stores unscaled V.
                    (
                        subtract(&subtract(self.weights.ciphertext(), h)?, &k_u)?,
                        Some(ev.add(&mu_u, href)?),
                    )
                } else {
                    (subtract(self.weights.ciphertext(), h)?, Some(href.clone()))
                }
            }
        };
        let weights = PackedWeights::new(ev, req.layer.clone(), new_w)?;
        let velocity = new_u
            .map(|ct| PackedWeights::new(ev, req.layer.clone(), ct))
            .transpose()?;
        Ok((
            PackedUpdateCandidate {
                state: Self {
                    weights,
                    velocity,
                    policy: self.policy,
                    update_index: self.update_index + 1,
                    generation: Arc::new(()),
                },
            },
            scalar_history,
        ))
    }
}

fn validate_weights<E: HeOps>(ev: &E, w: &PackedWeights<E::Ct>) -> Result<()> {
    check_ct(ev, w.ciphertext(), w.layer().layout().slots())?;
    if w.ciphertext().scale() != ev.params().scale() {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(())
}
fn scalar_preflight<E: HeOps>(ev: &E, level: usize, value: f64) -> Result<()> {
    let ps = ev.exact_product_scale(level, ev.params().scale())? / ev.params().scale();
    let k = (value * ps).round();
    if !ps.is_finite()
        || ps < ckks_train_ckks::evaluator::MIN_PLAIN_SCALE
        || !k.is_finite()
        || k.abs() >= (1u64 << 62) as f64
        || (value != 0.0 && k == 0.0)
    {
        return invalid("packed state scalar cannot be represented at this level/scale");
    }
    Ok(())
}

/// One checked request. This runtime token is not a cryptographic freshness proof.
#[derive(Debug)]
pub struct PackedUpdateRequest {
    generation: Arc<()>,
    update_index: u64,
    layer: LayerGeometry,
    valid_samples: usize,
    factors: Vec<f64>,
    kappa: Option<f64>,
}
impl PackedUpdateRequest {
    /// Branch constants for current then (only when needed) reference mean gradient.
    pub fn factors(&self) -> &[f64] {
        &self.factors
    }
    /// Index of the update that this request prepares.
    pub fn update_index(&self) -> u64 {
        self.update_index
    }
    fn check_branches<C>(&self, branches: &[PackedGradient<C>]) -> Result<()> {
        if branches.len() != self.factors.len() {
            return invalid("packed update has missing or extra gradient branch");
        }
        for (g, &factor) in branches.iter().zip(&self.factors) {
            if g.layer() != &self.layer
                || g.valid_samples() != self.valid_samples
                || g.factor() != factor
            {
                return invalid("packed update gradient layer, sample count or factor mismatch");
            }
        }
        Ok(())
    }
    /// Assigns generation metadata to raw D4a gradients after checking their metadata.
    /// Caller asserts their encrypted contents were computed for this update. This cannot
    /// detect deliberate rebinding of old raw ciphertexts; already-bound tokens cannot replay.
    pub fn bind<C>(&self, branches: Vec<PackedGradient<C>>) -> Result<PackedUpdateGradients<C>> {
        self.check_branches(&branches)?;
        Ok(PackedUpdateGradients {
            request: Self {
                generation: self.generation.clone(),
                update_index: self.update_index,
                layer: self.layer.clone(),
                valid_samples: self.valid_samples,
                factors: self.factors.clone(),
                kappa: self.kappa,
            },
            branches,
        })
    }
}

/// Gradient branches bound to one checked state generation; fields are private.
#[derive(Debug)]
pub struct PackedUpdateGradients<C> {
    request: PackedUpdateRequest,
    branches: Vec<PackedGradient<C>>,
}

/// Complete candidate. Dropping it leaves the old state untouched.
#[derive(Debug)]
pub struct PackedUpdateCandidate<C> {
    state: PackedLayerState<C>,
}
impl<C> PackedUpdateCandidate<C> {
    /// Infallibly transfers the prepared state, permitting later stage-wide assignment.
    pub fn into_state(self) -> PackedLayerState<C> {
        self.state
    }
}
