//! Current-stage encrypted learning. No refresh, bootstrap, secret callback or NI admission.
use super::ops::{directed_affine, mask_plain, outer_traced};
use super::types::check_ct;
use super::{
    LayerGeometry, Orientation, PackedGeometryPlan, PackedLayerState, PackedOptimizer,
    PackedVector, PackedWeights, VectorSpec, invalid,
};
use crate::{
    activation::{Activation, PolyRelu2, Square},
    layout_reboot::GradRelin,
};
use ckks_train_ckks::{DiagonalEntry, Encoder, HeCiphertext, HeOps, OpCounter};
use ckks_train_core::error::{HekitError, Result};
use std::sync::Arc;

/// Registered learning semantics; fields are immutable and arbitrary combinations are excluded.
#[derive(Clone, Copy, Debug)]
pub struct PackedStageConfig {
    nag: Option<(f64, f64)>,
}
impl PackedStageConfig {
    /// Square/Square, mean-output MSE, SGD and two required encrypted biases.
    pub fn square_sgd() -> Self {
        Self { nag: None }
    }
    /// PolyRelu2/Identity, residual-sum delta, NAG and no bias ciphertexts.
    pub fn poly_relu2_nag(momentum: f64, reference: f64) -> Result<Self> {
        PackedOptimizer::nag(momentum, reference)?;
        Ok(Self {
            nag: Some((momentum, reference)),
        })
    }
    fn square(self) -> bool {
        self.nag.is_none()
    }
    fn policy(self) -> Result<PackedOptimizer> {
        self.nag.map_or(Ok(PackedOptimizer::sgd()), |(m, r)| {
            PackedOptimizer::nag(m, r)
        })
    }
}

/// Ordered encrypted record. Invalid records are explicitly skipped, never denominators.
#[derive(Clone, Debug)]
pub struct PackedRecord<C> {
    input: PackedVector<C>,
    target: PackedVector<C>,
    valid: bool,
}
impl<C> PackedRecord<C> {
    /// The caller asserts plaintext packing; ciphertext metadata is checked during planning.
    pub fn new(input: PackedVector<C>, target: PackedVector<C>, valid: bool) -> Self {
        Self {
            input,
            target,
            valid,
        }
    }
}

/// One actual or predicted primitive boundary; natural product scales remain explicit.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTraceEntry {
    /// Valid-record ordinal, or None for shared gradient/state work.
    pub sample: Option<usize>,
    /// Stable operation name within this stage.
    pub name: String,
    /// Actual remaining modulus level.
    pub level: usize,
    /// Actual scale, never a relabelled approximation.
    pub scale: f64,
}
#[derive(Clone, Copy, Debug, PartialEq)]
struct History {
    level: usize,
    scale: f64,
}
impl History {
    fn of<C: DiagonalEntry>(ct: &C) -> Self {
        Self {
            level: ct.level(),
            scale: ct.scale(),
        }
    }
}
fn entry(trace: &mut Vec<PackedTraceEntry>, sample: Option<usize>, name: &str, h: History) {
    trace.push(PackedTraceEntry {
        sample,
        name: name.into(),
        level: h.level,
        scale: h.scale,
    });
}

/// Pure metadata plan for this step only. No future refresh/memory/range/precision admission.
#[derive(Clone, Debug)]
pub struct PackedStepPlan {
    /// Number of actual valid ordered records.
    pub valid_samples: usize,
    /// Complete expensive-operation prediction, independent from evaluator snapshots.
    pub predicted_counts: OpCounter,
    /// Boundary histories including natural affine/activation/dz/outer products.
    pub trace: Vec<PackedTraceEntry>,
    /// Minimum remaining candidate state level; future projection needs at least one.
    pub minimum_state_level: usize,
    /// Requested outer-sum policy, with evaluator lazy flag applied.
    pub effective_lazy: bool,
    input_level: usize,
    target_level: usize,
}
/// Old-forward snapshots and independently measured execution evidence.
#[derive(Debug)]
pub struct PackedStepReceipt<C> {
    /// Hidden activations from old state, in valid-record order.
    pub hidden: Vec<PackedVector<C>>,
    /// Predictions from old state, in valid-record order.
    pub output: Vec<PackedVector<C>>,
    /// The pre-execution metadata plan.
    pub plan: PackedStepPlan,
    /// Actual expensive-operation counter difference; no reset is performed.
    pub actual_counts: OpCounter,
    /// Observed histories, checked against the plan before returning a candidate.
    pub trace: Vec<PackedTraceEntry>,
}
/// Two matrix states and optional encrypted biases, with immutable learning semantics.
#[derive(Debug)]
pub struct PackedStage<C> {
    layers: [PackedLayerState<C>; 2],
    biases: Option<[PackedVector<C>; 2]>,
    config: PackedStageConfig,
    real_classes: usize,
}
/// Prepared whole-stage state and old-forward execution receipt.
pub type PackedStepResult<C> = (PackedStageCandidate<C>, PackedStepReceipt<C>);

/// Complete owned stage candidate. Dropping it leaves every old layer/bias unchanged.
#[derive(Debug)]
pub struct PackedStageCandidate<C> {
    stage: PackedStage<C>,
}
impl<C> PackedStageCandidate<C> {
    /// Infallible whole-stage state transfer; no partial-layer commit operation is exposed.
    pub fn into_stage(self) -> PackedStage<C> {
        self.stage
    }
}

impl<C: HeCiphertext + Clone> PackedStage<C> {
    /// Checks a pair against one existing geometry stage and the registered bias domain.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        geometry: &PackedGeometryPlan,
        stage: usize,
        weights: [PackedWeights<C>; 2],
        biases: Option<[PackedVector<C>; 2]>,
        config: PackedStageConfig,
    ) -> Result<Self> {
        let Some(shape) = geometry.stages().get(stage) else {
            return invalid("packed stage index out of range");
        };
        let expected = &geometry.layers()[2 * stage..2 * stage + 2];
        for k in 0..2 {
            if weights[k].layer() != &expected[k] {
                return invalid("packed stage weights do not belong to this geometry stage");
            }
        }
        if config.square() != biases.is_some() {
            return invalid("packed stage bias domain differs from learning semantics");
        }
        if config.square() && shape.output() != shape.real_classes() {
            return invalid("mean-output MSE stage requires exactly real-class outputs");
        }
        if let Some(b) = &biases {
            for k in 0..2 {
                validate_vector(ev, &b[k], expected[k].output(), true)?;
            }
        }
        let [a, b] = weights;
        Ok(Self {
            layers: [
                PackedLayerState::new(ev, a, config.policy()?)?,
                PackedLayerState::new(ev, b, config.policy()?)?,
            ],
            biases,
            config,
            real_classes: shape.real_classes(),
        })
    }
    /// Immutable matrix states; no plaintext weights or velocities are retained.
    pub fn layers(&self) -> &[PackedLayerState<C>; 2] {
        &self.layers
    }
    /// Present only for Square/Square SGD; absent means no dummy encrypted bias work.
    pub fn biases(&self) -> Option<&[PackedVector<C>; 2]> {
        self.biases.as_ref()
    }

    // Complete reconstruction only. Callers cannot supply policy/index/generation tokens.
    pub(super) fn with_refreshed_parts<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        weights: [PackedWeights<C>; 2],
        velocities: [Option<PackedWeights<C>>; 2],
        biases: Option<[PackedVector<C>; 2]>,
    ) -> Result<Self> {
        if biases.is_some() != self.biases.is_some() {
            return invalid("packed refresh changes bias domain");
        }
        if let Some(b) = &biases {
            for (k, bias) in b.iter().enumerate() {
                validate_vector(ev, bias, self.layers[k].weights().layer().output(), true)?;
            }
        }
        let [w0, w1] = weights;
        let [u0, u1] = velocities;
        let layers = [
            self.layers[0].with_refreshed_ciphertexts(ev, w0, u0)?,
            self.layers[1].with_refreshed_ciphertexts(ev, w1, u1)?,
        ];
        Ok(Self {
            layers,
            biases,
            config: self.config,
            real_classes: self.real_classes,
        })
    }

    /// Validates the entire current-stage DAG using metadata/host encodings only.
    /// All requested keys, contexts, factors and levels are checked before counted work.
    pub fn plan_step<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        records: &[PackedRecord<C>],
        rate: f64,
        relin: GradRelin,
    ) -> Result<PackedStepPlan> {
        if !Arc::ptr_eq(ev.params(), enc.params()) || !ev.has_relin_key() {
            return invalid("packed stage encoder context or relin key mismatch");
        }
        let ls = [
            self.layers[0].weights().layer(),
            self.layers[1].weights().layer(),
        ];
        for s in ls[0].layout().galois_steps() {
            if !ev.has_galois_key(s) {
                return invalid("packed stage missing galois key");
            }
        }
        let valid: Vec<_> = records.iter().filter(|r| r.valid).collect();
        if valid.is_empty() {
            return invalid("packed stage needs valid records");
        }
        let requests = [
            self.layers[0].request(rate, valid.len())?,
            self.layers[1].request(rate, valid.len())?,
        ];
        for k in 0..2 {
            for w in std::iter::once(self.layers[k].weights()).chain(self.layers[k].velocity()) {
                check_ct(ev, w.ciphertext(), ls[k].layout().slots())?;
                if w.ciphertext().scale() != ev.params().scale() {
                    return Err(HekitError::ScaleMismatch);
                }
            }
            if let Some(b) = &self.biases {
                validate_vector(ev, &b[k], ls[k].output(), true)?;
            }
        }
        for r in &valid {
            validate_vector(ev, &r.input, ls[0].input(), true)?;
            validate_vector(ev, &r.target, ls[1].output(), true)?;
        }
        let input_level = valid
            .iter()
            .map(|r| r.input.ciphertext().level())
            .min()
            .unwrap();
        let target_level = valid
            .iter()
            .map(|r| r.target.ciphertext().level())
            .min()
            .unwrap();
        let delta = ev.params().scale();
        let input = History {
            level: input_level,
            scale: delta,
        };
        let mut p = Planner {
            ev,
            enc,
            trace: vec![],
            counts: OpCounter::default(),
            sample: None,
        };
        let mut last = None;
        for j in 0..valid.len() {
            p.sample = Some(j);
            let z1 = p.affine(
                "affine1",
                input,
                History::of(self.layers[0].weights().ciphertext()),
                ls[0],
                false,
            )?;
            let z1 = p.bias(
                "z1",
                z1,
                self.biases.as_ref().map(|b| History::of(b[0].ciphertext())),
            )?;
            let h = p.activation("hidden", z1, ls[0].output(), !self.config.square())?;
            let z2 = p.affine(
                "affine2",
                h,
                History::of(self.layers[1].weights().ciphertext()),
                ls[1],
                false,
            )?;
            let z2 = p.bias(
                "z2",
                z2,
                self.biases.as_ref().map(|b| History::of(b[1].ciphertext())),
            )?;
            let y = if self.config.square() {
                p.activation("output", z2, ls[1].output(), false)?
            } else {
                p.record("output", z2)
            };
            let loss_input = History {
                level: y.level.min(target_level),
                scale: delta,
            };
            p.record("residual", loss_input);
            let loss = p.mask(
                "loss",
                loss_input,
                &ls[1].output().mask(),
                if self.config.square() {
                    2.0 / self.real_classes as f64
                } else {
                    1.0
                },
            )?;
            let dz2 = if self.config.square() {
                p.derivative("derivative2", z2, loss, ls[1].output(), false)?
            } else {
                p.record("dz2", loss)
            };
            let up = p.affine(
                "upstream",
                dz2,
                History::of(self.layers[1].weights().ciphertext()),
                ls[1],
                true,
            )?;
            let dz1 = p.derivative("derivative1", z1, up, ls[0].output(), !self.config.square())?;
            last = Some((h, dz1, dz2));
        }
        let (h, dz1, dz2) = last.unwrap();
        p.sample = None;
        let effective_lazy = relin == GradRelin::Lazy && ev.config().lazy_relin;
        let mut minimum_state_level = usize::MAX;
        for k in 0..2 {
            let (a, d) = if k == 0 { (input, dz1) } else { (h, dz2) };
            let sum = p.product(&format!("outer{k}"), a, d)?;
            p.counts.ct_ct_tensor += valid.len() as u64 - 1;
            if !effective_lazy {
                p.counts.relins += valid.len() as u64 - 1;
            }
            let mut grad = None;
            for (branch, &f) in requests[k].factors().iter().enumerate() {
                grad = Some(p.mask(
                    &format!("gradient{k}.{branch}"),
                    sum,
                    &ls[k].matrix_mask(),
                    f,
                )?);
            }
            let g = grad.unwrap();
            if let Some(u) = self.layers[k].velocity() {
                if u.ciphertext().level() <= g.level {
                    return Err(HekitError::NoLevelsLeft);
                }
                let (mu, r) = self
                    .config
                    .nag
                    .ok_or_else(|| HekitError::InvalidParams("unexpected SGD velocity".into()))?;
                for (name, f) in [("mu", mu), ("kappa", mu * mu / (1.0 + mu) * (rate / r))] {
                    let a = History {
                        level: g.level + 1,
                        scale: delta,
                    };
                    let ps = ev.exact_product_scale(a.level, delta)? / delta;
                    let rounded = (f * ps).round();
                    if !rounded.is_finite()
                        || rounded.abs() >= (1u64 << 62) as f64
                        || (f != 0.0 && rounded == 0.0)
                    {
                        return invalid("packed stage U scalar is unrepresentable");
                    }
                    p.mask(&format!("{name}{k}"), a, &vec![1.0; ev.params().slots()], f)?;
                }
            }
            let w = p.record(
                &format!("weight{k}"),
                History {
                    level: g.level.min(self.layers[k].weights().ciphertext().level()),
                    scale: delta,
                },
            );
            minimum_state_level = minimum_state_level.min(w.level);
            if self.config.nag.is_some() {
                p.record(&format!("velocity{k}"), g);
                minimum_state_level = minimum_state_level.min(g.level);
            }
            if let Some(b) = &self.biases {
                let bg = p.mask(
                    &format!("bias_gradient{k}"),
                    d,
                    &ls[k].output().mask(),
                    rate / valid.len() as f64,
                )?;
                let bh = p.record(
                    &format!("bias{k}"),
                    History {
                        level: bg.level.min(b[k].ciphertext().level()),
                        scale: delta,
                    },
                );
                minimum_state_level = minimum_state_level.min(bh.level);
            }
        }
        Ok(PackedStepPlan {
            valid_samples: valid.len(),
            predicted_counts: p.counts,
            trace: p.trace,
            minimum_state_level,
            effective_lazy,
            input_level,
            target_level,
        })
    }

    /// Prepares all W/U/bias candidates, returning old-forward snapshots. No old state mutates.
    /// Unexpected backend failures can advance counters, but never partially commit a stage.
    pub fn prepare_step<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        records: &[PackedRecord<C>],
        rate: f64,
        relin: GradRelin,
    ) -> Result<PackedStepResult<C>> {
        let plan = self.plan_step(ev, enc, records, rate, relin)?;
        let before = ev.counter();
        let mut trace = vec![];
        let mut inputs = vec![];
        let mut hidden = vec![];
        let mut output = vec![];
        let mut d1 = vec![];
        let mut d2 = vec![];
        let ls = [
            self.layers[0].weights().layer(),
            self.layers[1].weights().layer(),
        ];
        for (j, r) in records.iter().filter(|r| r.valid).enumerate() {
            let mut run = Runner {
                ev,
                enc,
                trace: &mut trace,
                sample: Some(j),
            };
            let x = PackedVector::new(
                ev,
                r.input.spec(),
                ev.mod_switch_to_level(r.input.ciphertext(), plan.input_level)?,
            )?;
            let z1 = run.affine("affine1", self.layers[0].weights(), &x, false)?;
            let z1 = run.bias("z1", z1, self.biases.as_ref().map(|b| &b[0]))?;
            let h = run.activation("hidden", &z1, !self.config.square())?;
            let z2 = run.affine("affine2", self.layers[1].weights(), &h, false)?;
            let z2 = run.bias("z2", z2, self.biases.as_ref().map(|b| &b[1]))?;
            let y = if self.config.square() {
                run.activation("output", &z2, false)?
            } else {
                run.record("output", &z2);
                z2.clone()
            };
            let l = y.ciphertext().level().min(plan.target_level);
            let residual = ev.sub(
                &ev.mod_switch_to_level(y.ciphertext(), l)?,
                &ev.mod_switch_to_level(r.target.ciphertext(), l)?,
            )?;
            run.record_ct("residual", &residual);
            let loss = run.mask(
                "loss",
                y.spec(),
                &residual,
                if self.config.square() {
                    2.0 / self.real_classes as f64
                } else {
                    1.0
                },
            )?;
            let dz2 = if self.config.square() {
                run.derivative("derivative2", &z2, &loss, false)?
            } else {
                run.record("dz2", &loss);
                loss
            };
            let up = run.affine("upstream", self.layers[1].weights(), &dz2, true)?;
            let dz1 = run.derivative("derivative1", &z1, &up, !self.config.square())?;
            inputs.push(x);
            hidden.push(h);
            output.push(y);
            d1.push(dz1);
            d2.push(dz2);
        }
        let mut states = vec![];
        let mut biases = vec![];
        for k in 0..2 {
            let (a, d) = if k == 0 {
                (&inputs, &d1)
            } else {
                (&hidden, &d2)
            };
            let pairs: Vec<_> = a.iter().zip(d).collect();
            let req = self.layers[k].request(rate, plan.valid_samples)?;
            let (gradients, history) = outer_traced(ev, enc, ls[k], &pairs, req.factors(), relin)?;
            entry(
                &mut trace,
                None,
                &format!("outer{k}"),
                History {
                    level: history.0,
                    scale: history.1,
                },
            );
            for (i, g) in gradients.iter().enumerate() {
                entry(
                    &mut trace,
                    None,
                    &format!("gradient{k}.{i}"),
                    History::of(g.ciphertext()),
                );
            }
            let (candidate, scalars) = self.layers[k].prepare_traced(ev, &req.bind(gradients)?)?;
            let state = candidate.into_state();
            for (name, (level, scale)) in ["mu", "kappa"].into_iter().zip(scalars) {
                entry(
                    &mut trace,
                    None,
                    &format!("{name}{k}"),
                    History { level, scale },
                );
            }
            entry(
                &mut trace,
                None,
                &format!("weight{k}"),
                History::of(state.weights().ciphertext()),
            );
            if let Some(u) = state.velocity() {
                entry(
                    &mut trace,
                    None,
                    &format!("velocity{k}"),
                    History::of(u.ciphertext()),
                );
            }
            states.push(state);
            if let Some(b) = &self.biases {
                let mut sum = d[0].ciphertext().clone();
                for x in &d[1..] {
                    sum = ev.add(&sum, x.ciphertext())?;
                }
                let mut run = Runner {
                    ev,
                    enc,
                    trace: &mut trace,
                    sample: None,
                };
                let gradient = run.mask(
                    &format!("bias_gradient{k}"),
                    ls[k].output(),
                    &sum,
                    rate / plan.valid_samples as f64,
                )?;
                let level = gradient.ciphertext().level().min(b[k].ciphertext().level());
                let ct = ev.sub(
                    &ev.mod_switch_to_level(b[k].ciphertext(), level)?,
                    &ev.mod_switch_to_level(gradient.ciphertext(), level)?,
                )?;
                run.record_ct(&format!("bias{k}"), &ct);
                biases.push(PackedVector::new(ev, ls[k].output(), ct)?);
            }
        }
        let actual_counts = &ev.counter() - &before;
        if trace != plan.trace || actual_counts != plan.predicted_counts {
            return invalid("packed stage executed history/count differs from preflight plan");
        }
        let mut it = states.into_iter();
        let stage = Self {
            layers: [it.next().unwrap(), it.next().unwrap()],
            biases: if self.biases.is_some() {
                let mut it = biases.into_iter();
                Some([it.next().unwrap(), it.next().unwrap()])
            } else {
                None
            },
            config: self.config,
            real_classes: self.real_classes,
        };
        Ok((
            PackedStageCandidate { stage },
            PackedStepReceipt {
                hidden,
                output,
                plan,
                actual_counts,
                trace,
            },
        ))
    }
}
fn validate_vector<E: HeOps>(
    ev: &E,
    v: &PackedVector<E::Ct>,
    spec: VectorSpec,
    canonical: bool,
) -> Result<()> {
    check_ct(ev, v.ciphertext(), spec.layout().slots())?;
    if v.spec() != spec {
        return invalid("packed stage vector representation or width mismatch");
    }
    if canonical && v.ciphertext().scale() != ev.params().scale() {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(())
}

struct Planner<'a, E: HeOps> {
    ev: &'a E,
    enc: &'a Encoder,
    trace: Vec<PackedTraceEntry>,
    counts: OpCounter,
    sample: Option<usize>,
}
impl<E: HeOps> Planner<'_, E> {
    fn record(&mut self, name: &str, h: History) -> History {
        entry(&mut self.trace, self.sample, name, h);
        h
    }
    fn product(&mut self, name: &str, a: History, b: History) -> Result<History> {
        let level = a.level.min(b.level);
        if level == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let scale = a.scale * b.scale / self.ev.params().q()[level] as f64;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(HekitError::ScaleMismatch);
        }
        self.counts.ct_ct_tensor += 1;
        self.counts.relins += 1;
        self.counts.rescales += 1;
        Ok(self.record(
            name,
            History {
                level: level - 1,
                scale,
            },
        ))
    }
    fn mask(&mut self, name: &str, a: History, mask: &[f64], factor: f64) -> Result<History> {
        if a.level == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let ps = self
            .ev
            .exact_product_scale(a.level, self.ev.params().scale())?
            / a.scale;
        if factor != 0.0 && (factor * ps).round() == 0.0 {
            return invalid("packed stage folded factor rounds to zero");
        }
        mask_plain(
            self.ev,
            self.enc,
            &mask.iter().map(|x| x * factor).collect::<Vec<_>>(),
            a.level,
            a.scale,
        )?;
        self.counts.ct_pt_mults += 1;
        self.counts.rescales += 1;
        Ok(self.record(
            name,
            History {
                level: a.level - 1,
                scale: self.ev.params().scale(),
            },
        ))
    }
    fn affine(
        &mut self,
        name: &str,
        x: History,
        w: History,
        l: &LayerGeometry,
        reverse: bool,
    ) -> Result<History> {
        let product = self.product(&format!("{name}.product"), x, w)?;
        let row = (l.orientation() == Orientation::Row) != reverse;
        let spec = if reverse { l.input() } else { l.output() };
        self.counts.rotations += if row {
            l.layout().r().ilog2() as u64
        } else {
            2 * l.layout().c().ilog2() as u64
        };
        self.mask(name, product, &spec.mask(), 1.0)
    }
    fn bias(&mut self, name: &str, z: History, b: Option<History>) -> Result<History> {
        Ok(self.record(
            name,
            History {
                level: b.map_or(z.level, |b| z.level.min(b.level)),
                scale: z.scale,
            },
        ))
    }
    fn activation(
        &mut self,
        name: &str,
        z: History,
        spec: VectorSpec,
        poly: bool,
    ) -> Result<History> {
        let nat = self.product(&format!("{name}.natural"), z, z)?;
        if poly {
            let ps = nat.scale * self.ev.params().q()[z.level] as f64 / z.scale;
            self.enc
                .encode(&vec![1.0; self.ev.params().slots()], ps, z.level)?;
            if z.scale * ps / self.ev.params().q()[z.level] as f64 != nat.scale {
                return Err(HekitError::ScaleMismatch);
            }
            self.counts.ct_pt_mults += 1;
            self.counts.rescales += 1;
        }
        self.mask(name, nat, &spec.mask(), 1.0)
    }
    fn derivative(
        &mut self,
        name: &str,
        z: History,
        up: History,
        spec: VectorSpec,
        poly: bool,
    ) -> Result<History> {
        if z.level == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let delta = self.ev.params().scale();
        let ps = delta * self.ev.params().q()[z.level] as f64 / z.scale;
        self.enc.encode(
            &spec.mask().iter().map(|x| 2.0 * x).collect::<Vec<_>>(),
            ps,
            z.level,
        )?;
        let scale = z.scale * ps / self.ev.params().q()[z.level] as f64;
        if scale != delta {
            return Err(HekitError::ScaleMismatch);
        }
        if poly {
            self.enc.encode(&spec.mask(), scale, z.level - 1)?;
        }
        self.counts.ct_pt_mults += 1;
        self.counts.rescales += 1;
        let s = self.record(
            name,
            History {
                level: z.level - 1,
                scale,
            },
        );
        self.product(if name == "derivative1" { "dz1" } else { "dz2" }, s, up)
    }
}
struct Runner<'a, E: HeOps> {
    ev: &'a E,
    enc: &'a Encoder,
    trace: &'a mut Vec<PackedTraceEntry>,
    sample: Option<usize>,
}
impl<E: HeOps> Runner<'_, E> {
    fn record_ct(&mut self, name: &str, ct: &E::Ct) {
        entry(self.trace, self.sample, name, History::of(ct));
    }
    fn record(&mut self, name: &str, v: &PackedVector<E::Ct>) {
        self.record_ct(name, v.ciphertext());
    }
    fn mask(
        &mut self,
        name: &str,
        spec: VectorSpec,
        ct: &E::Ct,
        factor: f64,
    ) -> Result<PackedVector<E::Ct>> {
        let pt = mask_plain(
            self.ev,
            self.enc,
            &spec.mask().iter().map(|x| x * factor).collect::<Vec<_>>(),
            ct.level(),
            ct.scale(),
        )?;
        let ct = self.ev.rescale(&self.ev.mul_plain_to_scale(
            ct,
            &self.ev.upload_pt_owned(pt)?,
            self.ev.params().scale(),
        )?)?;
        self.record_ct(name, &ct);
        PackedVector::new(self.ev, spec, ct)
    }
    fn affine(
        &mut self,
        name: &str,
        w: &PackedWeights<E::Ct>,
        x: &PackedVector<E::Ct>,
        reverse: bool,
    ) -> Result<PackedVector<E::Ct>> {
        let (out, (level, scale)) = directed_affine(self.ev, self.enc, w, x, reverse)?;
        entry(
            self.trace,
            self.sample,
            &format!("{name}.product"),
            History { level, scale },
        );
        self.record(name, &out);
        Ok(out)
    }
    fn bias(
        &mut self,
        name: &str,
        z: PackedVector<E::Ct>,
        b: Option<&PackedVector<E::Ct>>,
    ) -> Result<PackedVector<E::Ct>> {
        let out = if let Some(b) = b {
            let l = z.ciphertext().level().min(b.ciphertext().level());
            PackedVector::new(
                self.ev,
                z.spec(),
                self.ev.add(
                    &self.ev.mod_switch_to_level(z.ciphertext(), l)?,
                    &self.ev.mod_switch_to_level(b.ciphertext(), l)?,
                )?,
            )?
        } else {
            z
        };
        self.record(name, &out);
        Ok(out)
    }
    fn activation(
        &mut self,
        name: &str,
        z: &PackedVector<E::Ct>,
        poly: bool,
    ) -> Result<PackedVector<E::Ct>> {
        let nat = if poly {
            PolyRelu2.forward(self.ev, self.enc, z.ciphertext())?
        } else {
            Square.forward(self.ev, self.enc, z.ciphertext())?
        };
        self.record_ct(&format!("{name}.natural"), &nat);
        self.mask(name, z.spec(), &nat, 1.0)
    }
    fn derivative(
        &mut self,
        name: &str,
        z: &PackedVector<E::Ct>,
        up: &PackedVector<E::Ct>,
        poly: bool,
    ) -> Result<PackedVector<E::Ct>> {
        let s = if poly {
            PolyRelu2.derivative_factor_at(
                self.ev,
                self.enc,
                z.ciphertext(),
                1.0,
                Some(&z.spec().mask()),
                Some(self.ev.params().scale()),
            )?
        } else {
            Square.derivative_factor_at(
                self.ev,
                self.enc,
                z.ciphertext(),
                1.0,
                Some(&z.spec().mask()),
                Some(self.ev.params().scale()),
            )?
        };
        self.record_ct(name, &s);
        let l = s.level().min(up.ciphertext().level());
        let ct = self.ev.rescale(&self.ev.mul(
            &self.ev.mod_switch_to_level(&s, l)?,
            &self.ev.mod_switch_to_level(up.ciphertext(), l)?,
        )?)?;
        self.record_ct(if name == "derivative1" { "dz1" } else { "dz2" }, &ct);
        PackedVector::new(self.ev, z.spec(), ct)
    }
}
