//! Immutable current-stage learning with a fixed typed AA seam; no secret callback.
use super::ops::{directed_affine, mask_plain, outer_traced};
use super::types::check_ct;
use super::{
    PackedGeometryPlan, PackedLayerState, PackedOptimizer, PackedVector, PackedWeights, VectorSpec,
    invalid,
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
    pub(super) nag: Option<(f64, f64)>,
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
    pub(super) fn square(self) -> bool {
        self.nag.is_none()
    }
    pub(super) fn policy(self) -> Result<PackedOptimizer> {
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
    /// Original record position (including holes), or None for shared gradient/state work.
    pub sample: Option<usize>,
    /// Stable operation name within this stage.
    pub name: String,
    /// Actual remaining modulus level.
    pub level: usize,
    /// Actual scale, never a relabelled approximation.
    pub scale: f64,
}
use super::dag::{History, entry};

/// Encoder-checked metadata plan for this step only; no memory/range/precision admission.
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
    pub(super) input_level: usize,
    pub(super) target_level: usize,
}
/// Old-forward snapshots and independently measured execution evidence.
#[derive(Debug)]
pub struct PackedStepReceipt<C> {
    /// Hidden activations from old state, in valid-record order.
    pub hidden: Vec<PackedVector<C>>,
    forward: StageForwardSnapshot<C>,
    /// Predictions from old state, in valid-record order.
    pub output: Vec<PackedVector<C>>,
    /// The pre-execution metadata plan.
    pub plan: PackedStepPlan,
    /// Actual expensive-operation counter difference; no reset is performed.
    pub actual_counts: OpCounter,
    /// Observed histories, checked against the plan before returning a candidate.
    pub trace: Vec<PackedTraceEntry>,
}
/// Retained original valid identities and old-state forward boundaries.
#[derive(Debug)]
pub struct StageForwardSnapshot<C> {
    valid_ids: Vec<usize>,
    hidden_pre: Vec<PackedVector<C>>,
    hidden_post: Option<Vec<PackedVector<C>>>,
    output: Vec<PackedVector<C>>,
}
impl<C> StageForwardSnapshot<C> {
    /// Original input positions, preserving holes.
    pub fn valid_ids(&self) -> &[usize] {
        &self.valid_ids
    }
    /// Canonical old hidden before optional bootstrap.
    pub fn hidden_pre(&self) -> &[PackedVector<C>] {
        &self.hidden_pre
    }
    /// Exact post objects selected by the second layer and its gradient.
    pub fn hidden_post(&self) -> Option<&[PackedVector<C>]> {
        self.hidden_post.as_deref()
    }
    /// Actual selected second-layer inputs.
    pub fn hidden(&self) -> &[PackedVector<C>] {
        self.hidden_post.as_deref().unwrap_or(&self.hidden_pre)
    }
    /// Old-state model outputs.
    pub fn output(&self) -> &[PackedVector<C>] {
        &self.output
    }
}
impl<C> PackedStepReceipt<C> {
    /// Explicit pre/post and original valid-record snapshots.
    pub fn forward(&self) -> &StageForwardSnapshot<C> {
        &self.forward
    }
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

    /// Extracts checked current histories with private existing layer-generation authority.
    pub fn metadata<E: HeOps<Ct = C>>(&self, ev: &E) -> Result<super::PackedStageMeta> {
        use ckks_train_ckks::metadata::CtMeta;
        let meta = |ct: &C| -> Result<CtMeta> {
            check_ct(ev, ct, ev.params().slots())?;
            if ct.scale() != ev.params().scale() {
                return Err(HekitError::ScaleMismatch);
            }
            CtMeta::new(ev.params(), ct.level(), ct.scale())
        };
        Ok(super::PackedStageMeta {
            params: ev.params().clone(),
            layers: std::array::from_fn(|k| self.layers[k].weights().layer().clone()),
            config: self.config,
            classes: self.real_classes,
            weights: [
                meta(self.layers[0].weights().ciphertext())?,
                meta(self.layers[1].weights().ciphertext())?,
            ],
            velocities: [
                self.layers[0]
                    .velocity()
                    .map(|u| meta(u.ciphertext()))
                    .transpose()?,
                self.layers[1]
                    .velocity()
                    .map(|u| meta(u.ciphertext()))
                    .transpose()?,
            ],
            biases: self
                .biases
                .as_ref()
                .map(|b| Ok([meta(b[0].ciphertext())?, meta(b[1].ciphertext())?]))
                .transpose()?,
            indices: std::array::from_fn(|k| self.layers[k].update_index()),
            generations: Some(std::array::from_fn(|k| self.layers[k].generation())),
        })
    }
    /// Validates encrypted valid records and keeps their original positions and full metadata.
    pub fn batch_metadata<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        records: &[PackedRecord<C>],
    ) -> Result<super::PackedBatchMeta> {
        use ckks_train_ckks::metadata::CtMeta;
        let input = self.layers[0].weights().layer().input();
        let target = self.layers[1].weights().layer().output();
        let mut out = Vec::new();
        for (id, r) in records.iter().enumerate().filter(|(_, r)| r.valid) {
            validate_vector(ev, &r.input, input, true)?;
            validate_vector(ev, &r.target, target, true)?;
            out.push(super::PackedRecordMeta::new(
                id,
                CtMeta::new(
                    ev.params(),
                    r.input.ciphertext().level(),
                    r.input.ciphertext().scale(),
                )?,
                CtMeta::new(
                    ev.params(),
                    r.target.ciphertext().level(),
                    r.target.ciphertext().scale(),
                )?,
            ));
        }
        super::PackedBatchMeta::new(
            ev.params().clone(),
            self.layers[0].update_index(),
            input,
            target,
            out,
        )
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
        self.layers[0].request(rate, valid.len())?;
        self.layers[1].request(rate, valid.len())?;
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
        let state = self.metadata(ev)?;
        let batch = self.batch_metadata(ev, records)?;
        let update = super::PackedUpdateMeta::new(rate, relin, ev.config().lazy_relin)?;
        super::plan_stage_dag(enc, &state, &batch, update, None).map(|p| p.step)
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
        self.prepare_with_plan(
            ev,
            enc,
            records,
            rate,
            relin,
            plan,
            None,
            &mut Vec::new(),
            &mut None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare_with_plan<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        records: &[PackedRecord<C>],
        rate: f64,
        relin: GradRelin,
        plan: PackedStepPlan,
        bootstrap: Option<(
            &dyn super::PlannedBootstrap<E>,
            &super::PackedBootstrapContract,
        )>,
        events: &mut Vec<super::PackedBootstrapEvent<C>>,
        role: &mut Option<super::PackedRefreshRole>,
    ) -> Result<PackedStepResult<C>> {
        let before = ev.counter();
        let mut trace = vec![];
        let mut inputs = vec![];
        let mut hidden = vec![];
        let mut hidden_pre = vec![];
        let mut output = vec![];
        let mut d1 = vec![];
        let mut d2 = vec![];
        let ls = [
            self.layers[0].weights().layer(),
            self.layers[1].weights().layer(),
        ];
        for (j, r) in records.iter().enumerate().filter(|(_, r)| r.valid) {
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
            hidden_pre.push(h.clone());
            let h = if let Some((boot, contract)) = bootstrap {
                run.record("hidden.pre_boot", &h);
                let post = super::trainer::bootstrap_hidden(
                    ev, ls[0], j, &h, boot, contract, events, role,
                )?;
                run.record("hidden.post_boot", &post);
                post
            } else {
                h
            };
            // This same selected object feeds affine2, is retained below, and is borrowed
            // by outer1 from `hidden`. derivative1 continues to use the old z1 above.
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
                forward: StageForwardSnapshot {
                    valid_ids: records
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.valid)
                        .map(|(i, _)| i)
                        .collect(),
                    hidden_pre,
                    hidden_post: bootstrap.map(|_| hidden.clone()),
                    output: output.clone(),
                },
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

/// Input-only encrypted record; no target or learning rate can enter inference.
#[derive(Clone, Debug)]
pub struct PackedForwardRecord<C> {
    pub(super) id: usize,
    pub(super) input: PackedVector<C>,
    pub(super) valid: bool,
}
impl<C> PackedForwardRecord<C> {
    /// Declares a physical record; checked at planning/preparation.
    pub fn new(id: usize, input: PackedVector<C>, valid: bool) -> Self {
        Self { id, input, valid }
    }
    /// Original physical identity.
    pub fn id(&self) -> usize {
        self.id
    }
    /// Input ciphertext in its declared representation.
    pub fn input(&self) -> &PackedVector<C> {
        &self.input
    }
    /// Public validity flag.
    pub fn valid(&self) -> bool {
        self.valid
    }
}
/// Precisely which used layers belong to an inference stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedForwardPart {
    /// Backbone only; never inspects local head or any velocity.
    Backbone,
    /// Final backbone plus final classifier and configured output activation.
    OutputStack,
}
/// Immutable generation-bound forward plan, with no training DAG.
#[derive(Clone, Debug)]
pub struct PackedForwardPlan {
    part: PackedForwardPart,
    generations: Vec<Arc<()>>,
    inputs: Vec<(usize, ckks_train_ckks::metadata::CtMeta)>,
    outputs: Vec<ckks_train_ckks::metadata::CtMeta>,
    trace: Vec<PackedTraceEntry>,
    counts: OpCounter,
    hidden: Option<super::PackedBootstrapContract>,
}
impl PackedForwardPlan {
    /// Selected path, excluding unused local heads.
    pub fn part(&self) -> PackedForwardPart {
        self.part
    }
    /// Complete operation prediction, including explicit hidden refresh.
    pub fn predicted_counts(&self) -> &OpCounter {
        &self.counts
    }
    /// Exact planned primitive boundaries.
    pub fn trace(&self) -> &[PackedTraceEntry] {
        &self.trace
    }
    /// Selected path outputs in valid-record order.
    pub fn output_histories(&self) -> &[ckks_train_ckks::metadata::CtMeta] {
        &self.outputs
    }
}
/// Encrypted path results, without any state candidate.
#[derive(Debug)]
pub struct PackedForwardOutput<C> {
    pub(super) ids: Vec<usize>,
    pub(super) values: Vec<PackedVector<C>>,
}
impl<C> PackedForwardOutput<C> {
    /// Original valid physical IDs.
    pub fn valid_ids(&self) -> &[usize] {
        &self.ids
    }
    /// Final selected path values.
    pub fn values(&self) -> &[PackedVector<C>] {
        &self.values
    }
}
/// Available encrypted results for one started physical inference record, including partial failure.
#[derive(Debug)]
pub struct PackedForwardObservation<C> {
    id: usize,
    input: PackedVector<C>,
    affine: Option<PackedVector<C>>,
    preactivation: Option<PackedVector<C>>,
    hidden_pre: Option<PackedVector<C>>,
    hidden_post: Option<PackedVector<C>>,
    head_affine: Option<PackedVector<C>>,
    head_preactivation: Option<PackedVector<C>>,
    output: Option<PackedVector<C>>,
    bootstrap: Option<super::PackedBootstrapAttempt<C>>,
    counts: Option<OpCounter>,
}
impl<C> PackedForwardObservation<C> {
    /// Original physical ID, preserving invalid holes.
    pub fn id(&self) -> usize {
        self.id
    }
    /// Actual selected input, retained before the first primitive.
    pub fn input(&self) -> &PackedVector<C> {
        &self.input
    }
    /// Completed backbone affine, before optional bias.
    pub fn affine(&self) -> Option<&PackedVector<C>> {
        self.affine.as_ref()
    }
    /// Completed backbone affine plus optional bias.
    pub fn preactivation(&self) -> Option<&PackedVector<C>> {
        self.preactivation.as_ref()
    }
    /// Completed hidden before optional bootstrap.
    pub fn hidden_pre(&self) -> Option<&PackedVector<C>> {
        self.hidden_pre.as_ref()
    }
    /// Checked post-bootstrap hidden, absent when its raw output failed validation.
    pub fn hidden_post(&self) -> Option<&PackedVector<C>> {
        self.hidden_post.as_ref()
    }
    /// Completed final-head affine, absent on backbone-only paths.
    pub fn head_affine(&self) -> Option<&PackedVector<C>> {
        self.head_affine.as_ref()
    }
    /// Completed final-head affine plus bias.
    pub fn head_preactivation(&self) -> Option<&PackedVector<C>> {
        self.head_preactivation.as_ref()
    }
    /// Completed selected path output; no placeholder is manufactured on failure.
    pub fn output(&self) -> Option<&PackedVector<C>> {
        self.output.as_ref()
    }
    /// Raw encrypted attempted hidden refresh, including rejected returned ciphertexts.
    pub fn bootstrap_attempt(&self) -> Option<&super::PackedBootstrapAttempt<C>> {
        self.bootstrap.as_ref()
    }
    /// Inclusive interval for this started record, including partial failed work.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_ref()
    }
}
/// Forward-only old-state observations and separately counted explicit refresh.
#[derive(Debug)]
pub struct PackedForwardReceipt<C> {
    observations: Vec<PackedForwardObservation<C>>,
    plan: PackedForwardPlan,
    trace: Vec<PackedTraceEntry>,
    inputs: Vec<PackedVector<C>>,
    preactivations: Vec<PackedVector<C>>,
    outputs: Vec<PackedVector<C>>,
    hidden_pre: Vec<PackedVector<C>>,
    hidden_post: Option<Vec<PackedVector<C>>>,
    events: Vec<super::PackedBootstrapEvent<C>>,
    counts: OpCounter,
}
impl<C> PackedForwardReceipt<C> {
    /// Every started physical record and its available encrypted primitive results.
    pub fn observations(&self) -> &[PackedForwardObservation<C>] {
        &self.observations
    }
    /// Actual selected inputs, including restored old-hidden boundaries.
    pub fn inputs(&self) -> &[PackedVector<C>] {
        &self.inputs
    }
    /// Backbone affine plus bias, before its activation.
    pub fn preactivations(&self) -> &[PackedVector<C>] {
        &self.preactivations
    }
    /// Actual selected path outputs, including the final head only in OutputStack.
    pub fn outputs(&self) -> &[PackedVector<C>] {
        &self.outputs
    }

    /// Immutable pre-execution plan.
    pub fn plan(&self) -> &PackedForwardPlan {
        &self.plan
    }
    /// Observed primitive boundaries.
    pub fn trace(&self) -> &[PackedTraceEntry] {
        &self.trace
    }
    /// Old hidden before the optional fixed refresh.
    pub fn hidden_pre(&self) -> &[PackedVector<C>] {
        &self.hidden_pre
    }
    /// Exact refreshed hidden selected by the final head or returned backbone.
    pub fn hidden_post(&self) -> Option<&[PackedVector<C>]> {
        self.hidden_post.as_deref()
    }
    /// Only explicitly planned hidden bootstrap events.
    pub fn events(&self) -> &[super::PackedBootstrapEvent<C>] {
        &self.events
    }
    /// Single inclusive observed stage interval.
    pub fn actual_counts(&self) -> &OpCounter {
        &self.counts
    }
}
/// Partial inference evidence; no state commit exists.
#[derive(Debug)]
pub struct PackedForwardFailure<C> {
    observations: Box<[PackedForwardObservation<C>]>,
    error: Box<HekitError>,
    role: Option<super::PackedRefreshRole>,
    trace: Vec<PackedTraceEntry>,
    events: Vec<super::PackedBootstrapEvent<C>>,
    counts: Option<Box<OpCounter>>,
}
impl<C> PackedForwardFailure<C> {
    /// Completed and partial started records; invalid holes have no observations.
    pub fn observations(&self) -> &[PackedForwardObservation<C>] {
        &self.observations
    }
    /// Original failure.
    pub fn error(&self) -> &HekitError {
        &self.error
    }
    /// Current refresh role, if any.
    pub fn role(&self) -> Option<super::PackedRefreshRole> {
        self.role
    }
    /// Partial actual primitive trace.
    pub fn trace(&self) -> &[PackedTraceEntry] {
        &self.trace
    }
    /// Completed validated refreshes.
    pub fn events(&self) -> &[super::PackedBootstrapEvent<C>] {
        &self.events
    }
    /// Inclusive attempted interval.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_deref()
    }
}
/// Forward-only result, with no optimizer or graph candidate.
pub type PackedForwardResult<C> =
    std::result::Result<(PackedForwardOutput<C>, PackedForwardReceipt<C>), PackedForwardFailure<C>>;
impl<C: HeCiphertext + Clone> PackedStage<C> {
    fn forward_inputs<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        records: &[PackedForwardRecord<C>],
    ) -> Result<Vec<(usize, ckks_train_ckks::metadata::CtMeta)>> {
        let mut out = vec![];
        for (id, r) in records.iter().enumerate() {
            if id != r.id {
                return invalid("packed forward physical ID mismatch");
            }
            validate_vector(ev, &r.input, self.layers[0].weights().layer().input(), true)?;
            if r.valid {
                out.push((
                    id,
                    ckks_train_ckks::metadata::CtMeta::new(
                        ev.params(),
                        r.input.ciphertext().level(),
                        r.input.ciphertext().scale(),
                    )?,
                ));
            }
        }
        if out.is_empty() {
            return invalid("packed forward needs valid records");
        }
        Ok(out)
    }
    /// Plans exactly the selected path using real q/Encoder recurrences; no training metadata.
    pub fn plan_forward<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        records: &[PackedForwardRecord<C>],
        part: PackedForwardPart,
        hidden: Option<&super::PackedBootstrapContract>,
    ) -> Result<PackedForwardPlan> {
        self.plan_forward_histories(ev, enc, self.forward_inputs(ev, records)?, part, hidden)
    }
    pub(super) fn plan_forward_histories<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        inputs: Vec<(usize, ckks_train_ckks::metadata::CtMeta)>,
        part: PackedForwardPart,
        hidden: Option<&super::PackedBootstrapContract>,
    ) -> Result<PackedForwardPlan> {
        if !Arc::ptr_eq(ev.params(), enc.params())
            || !ev.has_relin_key()
            || inputs.is_empty()
            || inputs.windows(2).any(|w| w[0].0 >= w[1].0)
        {
            return invalid("packed forward context, key or IDs mismatch");
        }
        if let Some(b) = hidden {
            b.validate_params(ev.params())?;
        }
        let used = if part == PackedForwardPart::Backbone {
            1
        } else {
            2
        };
        let mut weights = vec![];
        let mut biases = vec![];
        let mut generations = vec![];
        for k in 0..used {
            let w = self.layers[k].weights();
            check_ct(ev, w.ciphertext(), w.layer().layout().slots())?;
            if w.ciphertext().scale() != ev.params().scale() {
                return Err(HekitError::ScaleMismatch);
            }
            if w.layer()
                .layout()
                .galois_steps()
                .iter()
                .any(|&s| !ev.has_galois_key(s))
            {
                return invalid("packed forward missing used layer keys");
            }
            weights.push(History::of(w.ciphertext()));
            generations.push(self.layers[k].generation());
            let b = if let Some(b) = &self.biases {
                validate_vector(ev, &b[k], w.layer().output(), true)?;
                Some(History::of(b[k].ciphertext()))
            } else {
                None
            };
            biases.push(b);
        }
        let mut planner = super::dag::ForwardPlanner::new(ev.params(), enc);
        let mut outputs = vec![];
        let mut extra = OpCounter::default();
        for &(id, input) in &inputs {
            ckks_train_ckks::metadata::CtMeta::new(ev.params(), input.level(), input.scale())?;
            if input.scale_bits() != ev.params().scale().to_bits() {
                return Err(HekitError::ScaleMismatch);
            }
            planner.set_sample(Some(id));
            let z = planner.affine(
                "affine1",
                input.into(),
                weights[0],
                self.layers[0].weights().layer(),
                false,
            )?;
            let z = planner.bias("z1", z, biases[0])?;
            let mut h = planner.activation(
                "hidden",
                z,
                self.layers[0].weights().layer().output(),
                !self.config.square(),
            )?;
            if let Some(b) = hidden {
                planner.record("hidden.pre_boot", h);
                h = planner.record("hidden.post_boot", b.output().into());
                extra = super::dag::checked_counts(&extra, b.cost_per_boot())?;
            }
            let y = if part == PackedForwardPart::OutputStack {
                let z = planner.affine(
                    "affine2",
                    h,
                    weights[1],
                    self.layers[1].weights().layer(),
                    false,
                )?;
                let z = planner.bias("z2", z, biases[1])?;
                if self.config.square() {
                    planner.activation(
                        "output",
                        z,
                        self.layers[1].weights().layer().output(),
                        false,
                    )?
                } else {
                    planner.record("output", z)
                }
            } else {
                h
            };
            outputs.push(ckks_train_ckks::metadata::CtMeta::new(
                ev.params(),
                y.level,
                y.scale,
            )?);
        }
        let (trace, counts) = planner.finish();
        let counts = super::dag::checked_counts(&counts, &extra)?;
        Ok(PackedForwardPlan {
            part,
            generations,
            inputs,
            outputs,
            trace,
            counts,
            hidden: hidden.cloned(),
        })
    }
    /// Executes only affine/bias/activation and the explicit hidden seam. Never updates state.
    pub fn prepare_forward<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        enc: &Encoder,
        records: &[PackedForwardRecord<C>],
        plan: &PackedForwardPlan,
        bootstrap: Option<&dyn super::PlannedBootstrap<E>>,
    ) -> PackedForwardResult<C> {
        let before = ev.counter();
        let mut trace = vec![];
        let mut events = vec![];
        let mut role = None;
        let mut observations: Vec<PackedForwardObservation<C>> = vec![];
        let mut record_before = None;
        let result = (|| -> Result<PackedForwardOutput<C>> {
            let actual = self.plan_forward(ev, enc, records, plan.part, plan.hidden.as_ref())?;
            if actual.inputs != plan.inputs
                || actual.trace != plan.trace
                || actual.counts != plan.counts
                || actual.generations.len() != plan.generations.len()
                || actual
                    .generations
                    .iter()
                    .zip(&plan.generations)
                    .any(|(a, b)| !Arc::ptr_eq(a, b))
            {
                return invalid("packed forward stale plan or input histories");
            }
            match (bootstrap, plan.hidden.as_ref()) {
                (Some(b), Some(c)) => b.validate_binding(ev, c)?,
                (None, None) => {}
                _ => return invalid("packed forward hidden capability mismatch"),
            }
            let mut values = vec![];
            for r in records.iter().filter(|r| r.valid) {
                record_before = Some(ev.counter());
                observations.push(PackedForwardObservation {
                    id: r.id,
                    input: r.input.clone(),
                    affine: None,
                    preactivation: None,
                    hidden_pre: None,
                    hidden_post: None,
                    head_affine: None,
                    head_preactivation: None,
                    output: None,
                    bootstrap: None,
                    counts: None,
                });
                let observed = observations.last_mut().unwrap();
                let mut run = Runner {
                    ev,
                    enc,
                    trace: &mut trace,
                    sample: Some(r.id),
                };
                let z = run.affine("affine1", self.layers[0].weights(), &r.input, false)?;
                observed.affine = Some(z.clone());
                let z = run.bias("z1", z, self.biases.as_ref().map(|b| &b[0]))?;
                observed.preactivation = Some(z.clone());
                let h = run.activation("hidden", &z, !self.config.square())?;
                observed.hidden_pre = Some(h.clone());
                let h = if let (Some(b), Some(c)) = (bootstrap, plan.hidden.as_ref()) {
                    run.record("hidden.pre_boot", &h);
                    let capture = super::block::BootstrapCapture::new(
                        b,
                        super::PackedRefreshRole::Hidden {
                            stage: self.layers[0].weights().layer().ordinal() / 2,
                            record: r.id as u64,
                        },
                    );
                    let result = super::trainer::bootstrap_hidden(
                        ev,
                        self.layers[0].weights().layer(),
                        r.id,
                        &h,
                        &capture,
                        c,
                        &mut events,
                        &mut role,
                    );
                    observed.bootstrap = capture.finish(result.is_ok());
                    let h = result?;
                    run.record("hidden.post_boot", &h);
                    observed.hidden_post = Some(h.clone());
                    h
                } else {
                    h
                };
                let y = if plan.part == PackedForwardPart::OutputStack {
                    let z = run.affine("affine2", self.layers[1].weights(), &h, false)?;
                    observed.head_affine = Some(z.clone());
                    let z = run.bias("z2", z, self.biases.as_ref().map(|b| &b[1]))?;
                    observed.head_preactivation = Some(z.clone());
                    if self.config.square() {
                        run.activation("output", &z, false)?
                    } else {
                        run.record("output", &z);
                        z
                    }
                } else {
                    h
                };
                observed.output = Some(y.clone());
                observed.counts =
                    super::refresh::counter_delta(&ev.counter(), record_before.as_ref().unwrap());
                values.push(y);
            }
            if trace != plan.trace
                || super::refresh::counter_delta(&ev.counter(), &before).as_ref()
                    != Some(&plan.counts)
            {
                return invalid("packed forward observed trace/count mismatch");
            }
            Ok(PackedForwardOutput {
                ids: plan.inputs.iter().map(|x| x.0).collect(),
                values,
            })
        })();
        if let (Some(observed), Some(start)) = (observations.last_mut(), record_before.as_ref()) {
            if observed.counts.is_none() {
                observed.counts = super::refresh::counter_delta(&ev.counter(), start);
            }
        }
        let counts = super::refresh::counter_delta(&ev.counter(), &before);
        match result {
            Ok(out) => {
                let outputs = out.values.clone();
                let receipt = PackedForwardReceipt {
                    plan: plan.clone(),
                    trace,
                    inputs: observations.iter().map(|o| o.input.clone()).collect(),
                    preactivations: observations
                        .iter()
                        .map(|o| o.preactivation.as_ref().unwrap().clone())
                        .collect(),
                    outputs,
                    hidden_pre: observations
                        .iter()
                        .map(|o| o.hidden_pre.as_ref().unwrap().clone())
                        .collect(),
                    hidden_post: plan.hidden.is_some().then(|| {
                        observations
                            .iter()
                            .map(|o| o.hidden_post.as_ref().unwrap().clone())
                            .collect()
                    }),
                    observations,
                    events,
                    counts: counts.unwrap(),
                };
                Ok((out, receipt))
            }
            Err(error) => Err(PackedForwardFailure {
                error: Box::new(error),
                role,
                trace,
                events,
                observations: observations.into_boxed_slice(),
                counts: counts.map(Box::new),
            }),
        }
    }
}
