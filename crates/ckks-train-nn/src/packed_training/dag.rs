//! Public, parameter-only stage histories. Conditional structural admission is not Ready.
use super::*;
use crate::layout_reboot::GradRelin;
use ckks_train_ckks::{
    CkksParams, DiagonalEntry, Encoder, OpCounter,
    metadata::{CtMeta, scalar_metadata_at, target_plain_scale_at},
};
use ckks_train_core::error::{HekitError, Result};
use std::sync::Arc;

/// Immutable per-valid-record histories; identity is the original record position.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedRecordMeta {
    id: usize,
    input: CtMeta,
    target: CtMeta,
}
impl PackedRecordMeta {
    /// Declares one valid record; validated against parameters by the batch constructor.
    pub fn new(id: usize, input: CtMeta, target: CtMeta) -> Self {
        Self { id, input, target }
    }
    /// Original record position, never a filtered enumeration.
    pub fn id(&self) -> usize {
        self.id
    }
}
/// Ordered valid metadata and the schedule position, with no encrypted data.
#[derive(Clone, Debug)]
pub struct PackedBatchMeta {
    params: Arc<CkksParams>,
    index: u64,
    input: VectorSpec,
    target: VectorSpec,
    records: Vec<PackedRecordMeta>,
}
impl PartialEq for PackedBatchMeta {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.params, &other.params)
            && self.index == other.index
            && self.input == other.input
            && self.target == other.target
            && self.records == other.records
    }
}
impl PackedBatchMeta {
    /// Binds parameter identity and checks uniqueness, exact Delta, levels and slots.
    pub fn new(
        params: Arc<CkksParams>,
        index: u64,
        input: VectorSpec,
        target: VectorSpec,
        records: Vec<PackedRecordMeta>,
    ) -> Result<Self> {
        if records.is_empty()
            || input.layout().slots() != params.slots()
            || target.layout().slots() != params.slots()
        {
            return invalid("packed batch empty or slot mismatch");
        }
        let mut ids = std::collections::BTreeSet::new();
        for r in &records {
            if !ids.insert(r.id) {
                return invalid("packed batch duplicate valid identity");
            }
            for h in [r.input, r.target] {
                canonical(&params, h)?;
            }
        }
        Ok(Self {
            params,
            index,
            input,
            target,
            records,
        })
    }
    /// Required input vector support.
    pub fn input_spec(&self) -> VectorSpec {
        self.input
    }
    /// Required full model-output target support.
    pub fn target_spec(&self) -> VectorSpec {
        self.target
    }
    /// Original valid identities and full incoming histories.
    pub fn records(&self) -> &[PackedRecordMeta] {
        &self.records
    }
    /// Update position.
    pub fn update_index(&self) -> u64 {
        self.index
    }
}
/// Public rate and explicit effective relinearization policy inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedUpdateMeta {
    pub(super) rate: f64,
    pub(super) relin: GradRelin,
    pub(super) lazy: bool,
}
impl PackedUpdateMeta {
    /// Checks the public rate; fixed-reference and scalar feasibility checks follow in DAG.
    pub fn new(rate: f64, relin: GradRelin, lazy: bool) -> Result<Self> {
        if !rate.is_normal() || rate <= 0.0 {
            return invalid("packed update invalid rate");
        }
        Ok(Self { rate, relin, lazy })
    }
    /// Requested actual mean-gradient rate.
    pub fn rate(&self) -> f64 {
        self.rate
    }
}
/// Exact W/U/bias histories. Hypothetical values have no executable generation authority.
#[derive(Clone, Debug)]
pub struct PackedStageMeta {
    pub(super) params: Arc<CkksParams>,
    pub(super) layers: [LayerGeometry; 2],
    pub(super) config: PackedStageConfig,
    pub(super) classes: usize,
    pub(super) weights: [CtMeta; 2],
    pub(super) velocities: [Option<CtMeta>; 2],
    pub(super) biases: Option<[CtMeta; 2]>,
    pub(super) indices: [u64; 2],
    pub(super) generations: Option<[Arc<()>; 2]>,
}
impl PackedStageMeta {
    /// Declares hypothetical state. This constructor can never produce executable authority.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Arc<CkksParams>,
        geometry: &PackedGeometryPlan,
        ordinal: usize,
        config: PackedStageConfig,
        weights: [CtMeta; 2],
        velocities: [Option<CtMeta>; 2],
        biases: Option<[CtMeta; 2]>,
        indices: [u64; 2],
    ) -> Result<Self> {
        let Some(shape) = geometry.stages().get(ordinal) else {
            return invalid("packed metadata stage out of range");
        };
        if geometry.layout().slots() != params.slots()
            || config.square() != biases.is_some()
            || (config.square() && shape.output() != shape.real_classes())
            || indices[0] != indices[1]
        {
            return invalid("packed metadata geometry or learning domain mismatch");
        }
        for k in 0..2 {
            canonical(&params, weights[k])?;
            if let Some(u) = velocities[k] {
                canonical(&params, u)?;
            }
            if velocities[k].is_some() != (!config.square() && indices[k] > 0) {
                return invalid("packed metadata velocity presence differs from update history");
            }
            if let Some(b) = biases {
                canonical(&params, b[k])?;
            }
        }
        Ok(Self {
            params,
            layers: [
                geometry.layers()[2 * ordinal].clone(),
                geometry.layers()[2 * ordinal + 1].clone(),
            ],
            config,
            classes: shape.real_classes(),
            weights,
            velocities,
            biases,
            indices,
            generations: None,
        })
    }
    /// False for constructor/derived future metadata; true only for checked current extraction.
    pub fn is_executable(&self) -> bool {
        self.generations.is_some()
    }
    /// All matrix histories.
    pub fn weights(&self) -> [CtMeta; 2] {
        self.weights
    }
    /// Actual U presence/history.
    pub fn velocities(&self) -> [Option<CtMeta>; 2] {
        self.velocities
    }
    /// Actual bias presence/history.
    pub fn biases(&self) -> Option<[CtMeta; 2]> {
        self.biases
    }
    /// Completed update indices.
    pub fn update_indices(&self) -> [u64; 2] {
        self.indices
    }
}
fn canonical(params: &CkksParams, h: CtMeta) -> Result<()> {
    CtMeta::new(params, h.level(), h.scale())?;
    if h.scale_bits() != params.scale().to_bits() {
        return invalid("packed metadata requires exact Delta");
    }
    Ok(())
}
/// Immutable complete trace and complete next state, without future generation authority.
#[derive(Clone, Debug)]
pub struct PackedDagPlan {
    pub(super) step: PackedStepPlan,
    pub(super) candidate: PackedStageMeta,
}
impl PackedDagPlan {
    /// Exact primitive trace/counts of this DAG.
    pub fn step(&self) -> &PackedStepPlan {
        &self.step
    }
    /// Full hypothetical candidate, not just a minimum level.
    pub fn candidate(&self) -> &PackedStageMeta {
        &self.candidate
    }
}
/// Evaluator-free DAG; Encoder performs public factor/mask feasibility checks.
pub fn plan_stage_dag(
    enc: &Encoder,
    state: &PackedStageMeta,
    batch: &PackedBatchMeta,
    update: PackedUpdateMeta,
    hidden: Option<&PackedBootstrapContract>,
) -> Result<PackedDagPlan> {
    let params = &state.params;
    if !Arc::ptr_eq(params, enc.params()) {
        return invalid("packed DAG Encoder context mismatch");
    }
    if !Arc::ptr_eq(params, &batch.params) {
        return invalid("packed DAG batch parameter identity mismatch");
    }
    for record in &batch.records {
        canonical(params, record.input)?;
        canonical(params, record.target)?;
    }
    if batch.index != state.indices[0]
        || state.indices[0] != state.indices[1]
        || batch.input != state.layers[0].input()
        || batch.target != state.layers[1].output()
    {
        return invalid("packed DAG schedule or representation mismatch");
    }
    let next_index = batch
        .index
        .checked_add(1)
        .ok_or_else(|| HekitError::InvalidParams("packed update index overflow".into()))?;
    let rate = update.rate;
    let (factors, _) = state
        .config
        .policy()?
        .coefficients(rate, batch.records.len())?;
    if let Some(c) = hidden {
        c.validate_params(params)?;
    }
    let input_level = batch.records.iter().map(|r| r.input.level()).min().unwrap();
    let target_level = batch
        .records
        .iter()
        .map(|r| r.target.level())
        .min()
        .unwrap();
    let ls = [&state.layers[0], &state.layers[1]];
    let delta = params.scale();
    let input = History {
        level: input_level,
        scale: delta,
    };
    let mut p = Planner {
        params,
        enc,
        trace: vec![],
        counts: OpCounter::default(),
        sample: None,
    };
    let mut last = None;
    for j in 0..batch.records.len() {
        p.sample = Some(batch.records[j].id);
        let z1 = p.affine(
            "affine1",
            input,
            History::from(state.weights[0]),
            ls[0],
            false,
        )?;
        let z1 = p.bias("z1", z1, state.biases.map(|b| History::from(b[0])))?;
        let h = p.activation("hidden", z1, ls[0].output(), !state.config.square())?;
        let h = if let Some(contract) = hidden {
            p.record("hidden.pre_boot", h);
            p.counts = checked_counts(&p.counts, contract.cost_per_boot())?;
            p.record("hidden.post_boot", History::from(contract.output()))
        } else {
            h
        };
        let z2 = p.affine("affine2", h, History::from(state.weights[1]), ls[1], false)?;
        let z2 = p.bias("z2", z2, state.biases.map(|b| History::from(b[1])))?;
        let y = if state.config.square() {
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
            if state.config.square() {
                2.0 / state.classes as f64
            } else {
                1.0
            },
        )?;
        let dz2 = if state.config.square() {
            p.derivative("derivative2", z2, loss, ls[1].output(), false)?
        } else {
            p.record("dz2", loss)
        };
        let up = p.affine(
            "upstream",
            dz2,
            History::from(state.weights[1]),
            ls[1],
            true,
        )?;
        let dz1 = p.derivative(
            "derivative1",
            z1,
            up,
            ls[0].output(),
            !state.config.square(),
        )?;
        last = Some((h, dz1, dz2));
    }
    let (h, dz1, dz2) = last.unwrap();
    p.sample = None;
    let effective_lazy = update.relin == GradRelin::Lazy && update.lazy;
    let mut minimum_state_level = usize::MAX;
    let mut candidate = state.clone();
    candidate.generations = None;
    candidate.indices = [next_index; 2];
    for k in 0..2 {
        let (a, d) = if k == 0 { (input, dz1) } else { (h, dz2) };
        let sum = p.product(&format!("outer{k}"), a, d)?;
        p.counts.ct_ct_tensor += batch.records.len() as u64 - 1;
        if !effective_lazy {
            p.counts.relins += batch.records.len() as u64 - 1;
        }
        let mut grad = None;
        for (branch, &f) in factors.as_slice().iter().enumerate() {
            grad = Some(p.mask(
                &format!("gradient{k}.{branch}"),
                sum,
                &ls[k].matrix_mask(),
                f,
            )?);
        }
        let g = grad.unwrap();
        if let Some(u) = state.velocities[k] {
            if u.level() <= g.level {
                return Err(HekitError::NoLevelsLeft);
            }
            let (mu, r) = state
                .config
                .nag
                .ok_or_else(|| HekitError::InvalidParams("unexpected SGD velocity".into()))?;
            for (name, f) in [("mu", mu), ("kappa", mu * mu / (1.0 + mu) * (rate / r))] {
                let a = History {
                    level: g.level + 1,
                    scale: delta,
                };
                let scalar =
                    scalar_metadata_at(params, CtMeta::new(params, a.level, a.scale)?, f, delta)?;
                let ps = scalar.plain_scale();
                let rounded = (f * ps).round();
                if !rounded.is_finite()
                    || rounded.abs() >= (1u64 << 62) as f64
                    || (f != 0.0 && rounded == 0.0)
                {
                    return invalid("packed stage U scalar is unrepresentable");
                }
                p.mask(&format!("{name}{k}"), a, &vec![1.0; params.slots()], f)?;
            }
        }
        let w = p.record(
            &format!("weight{k}"),
            History {
                level: g.level.min(state.weights[k].level()),
                scale: delta,
            },
        );
        minimum_state_level = minimum_state_level.min(w.level);
        candidate.weights[k] = w.meta(params)?;
        if state.config.nag.is_some() {
            p.record(&format!("velocity{k}"), g);
            candidate.velocities[k] = Some(g.meta(params)?);
            minimum_state_level = minimum_state_level.min(g.level);
        }
        if let Some(b) = &state.biases {
            let bg = p.mask(
                &format!("bias_gradient{k}"),
                d,
                &ls[k].output().mask(),
                rate / batch.records.len() as f64,
            )?;
            let bh = p.record(
                &format!("bias{k}"),
                History {
                    level: bg.level.min(b[k].level()),
                    scale: delta,
                },
            );
            minimum_state_level = minimum_state_level.min(bh.level);
            candidate.biases.as_mut().unwrap()[k] = bh.meta(params)?;
        }
    }
    Ok(PackedDagPlan {
        candidate,
        step: PackedStepPlan {
            valid_samples: batch.records.len(),
            predicted_counts: p.counts,
            trace: p.trace,
            minimum_state_level,
            effective_lazy,
            input_level,
            target_level,
        },
    })
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct History {
    pub(super) level: usize,
    pub(super) scale: f64,
}
impl History {
    pub(super) fn of<C: DiagonalEntry>(ct: &C) -> Self {
        Self {
            level: ct.level(),
            scale: ct.scale(),
        }
    }
}
pub(super) fn entry(
    trace: &mut Vec<PackedTraceEntry>,
    sample: Option<usize>,
    name: &str,
    h: History,
) {
    trace.push(PackedTraceEntry {
        sample,
        name: name.into(),
        level: h.level,
        scale: h.scale,
    });
}

impl From<CtMeta> for History {
    fn from(h: CtMeta) -> Self {
        Self {
            level: h.level(),
            scale: h.scale(),
        }
    }
}
impl History {
    fn meta(self, p: &CkksParams) -> Result<CtMeta> {
        CtMeta::new(p, self.level, self.scale)
    }
}
struct Planner<'a> {
    params: &'a CkksParams,
    enc: &'a Encoder,
    trace: Vec<PackedTraceEntry>,
    counts: OpCounter,
    sample: Option<usize>,
}
impl Planner<'_> {
    fn record(&mut self, name: &str, h: History) -> History {
        entry(&mut self.trace, self.sample, name, h);
        h
    }
    fn product(&mut self, name: &str, a: History, b: History) -> Result<History> {
        let level = a.level.min(b.level);
        if level == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        let scale = a.scale * b.scale / self.params.q()[level] as f64;
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
        let ps = target_plain_scale_at(self.params, a.level, a.scale, self.params.scale())?;
        if factor != 0.0 && (factor * ps).round() == 0.0 {
            return invalid("packed stage folded factor rounds to zero");
        }
        self.enc.encode(
            &mask.iter().map(|x| x * factor).collect::<Vec<_>>(),
            ps,
            a.level,
        )?;
        self.counts.ct_pt_mults += 1;
        self.counts.rescales += 1;
        Ok(self.record(
            name,
            History {
                level: a.level - 1,
                scale: self.params.scale(),
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
            let ps = nat.scale * self.params.q()[z.level] as f64 / z.scale;
            self.enc
                .encode(&vec![1.0; self.params.slots()], ps, z.level)?;
            if z.scale * ps / self.params.q()[z.level] as f64 != nat.scale {
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
        let delta = self.params.scale();
        let ps = delta * self.params.q()[z.level] as f64 / z.scale;
        self.enc.encode(
            &spec.mask().iter().map(|x| 2.0 * x).collect::<Vec<_>>(),
            ps,
            z.level,
        )?;
        let scale = z.scale * ps / self.params.q()[z.level] as f64;
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

pub(super) fn checked_counts(a: &OpCounter, b: &OpCounter) -> Result<OpCounter> {
    Ok(OpCounter {
        rotations: a
            .rotations
            .checked_add(b.rotations)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        relins: a
            .relins
            .checked_add(b.relins)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        ct_ct_tensor: a
            .ct_ct_tensor
            .checked_add(b.ct_ct_tensor)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        ct_pt_mults: a
            .ct_pt_mults
            .checked_add(b.ct_pt_mults)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        rescales: a
            .rescales
            .checked_add(b.rescales)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        sse_switches: a
            .sse_switches
            .checked_add(b.sse_switches)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
        bootstraps: a
            .bootstraps
            .checked_add(b.bootstraps)
            .ok_or_else(|| HekitError::InvalidParams("packed count overflow".into()))?,
    })
}

// Forward-only adapter: all recurrence bodies remain in the existing Planner.
pub(super) struct ForwardPlanner<'p>(Planner<'p>);
impl<'p> ForwardPlanner<'p> {
    pub(super) fn new(params: &'p CkksParams, enc: &'p Encoder) -> Self {
        Self(Planner {
            params,
            enc,
            trace: vec![],
            counts: OpCounter::default(),
            sample: None,
        })
    }
    pub(super) fn set_sample(&mut self, sample: Option<usize>) {
        self.0.sample = sample;
    }
    pub(super) fn record(&mut self, name: &str, h: History) -> History {
        self.0.record(name, h)
    }
    pub(super) fn affine(
        &mut self,
        name: &str,
        x: History,
        w: History,
        l: &LayerGeometry,
        reverse: bool,
    ) -> Result<History> {
        self.0.affine(name, x, w, l, reverse)
    }
    pub(super) fn bias(&mut self, name: &str, z: History, b: Option<History>) -> Result<History> {
        self.0.bias(name, z, b)
    }
    pub(super) fn activation(
        &mut self,
        name: &str,
        z: History,
        spec: VectorSpec,
        poly: bool,
    ) -> Result<History> {
        self.0.activation(name, z, spec, poly)
    }
    pub(super) fn finish(self) -> (Vec<PackedTraceEntry>, OpCounter) {
        (self.0.trace, self.0.counts)
    }
}
