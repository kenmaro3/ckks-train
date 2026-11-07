//! Single-stage CPU consuming bridge. This library path grants no production admission.
//! Context association is private in-process evidence, not cryptographic authentication.
use crate::study_admission::{DiagonalPlan, MicroPart, StateMetadata, TransitionIndex};
use crate::study_inputs::RateCoefficients;
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, CkksParams, Encoder, Evaluator, HeOps, OpCounter,
    bootstrap::{BootParams, Bootstrapper, metadata::BootstrapMetadata},
    metadata::CtMeta,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_nn::{
    activation::{Activation, Identity, PolyRelu2, Square},
    dense::{Dense, TransposeKeys},
    diagonal_plan::{DiagonalActivation, DiagonalMode, DiagonalOptimizer, StageGeometry},
    layout::Layout,
    optim::{Nag, Sgd},
    refresh::Refresher,
    trainer::{BatchSpec, LayerOptimizer, Mlp, Trainer},
    trainer_ni::{CompactState, NiOptimizer, NonInteractiveTrainer},
};
use std::{cell::RefCell, sync::Arc};

pub(crate) fn mismatch(field: &str) -> HekitError {
    HekitError::InvalidParams(format!("CPU bridge plan mismatch: {field}"))
}
pub(crate) fn ensure(ok: bool, field: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(mismatch(field)) }
}
pub(crate) fn require_single_stage(stages: usize) -> Result<()> {
    ensure(stages == 1, "missing block-stage execution capability")
}
/// Supported encrypted execution shape; blocks require the separate P5 bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuCapability {
    /// Exactly one existing NN stage.
    SingleStage,
}
#[derive(Clone)]
pub(crate) enum PlanOwner {
    Canonical(Arc<DiagonalPlan>),
    #[cfg(test)]
    Synthetic(Arc<crate::study_admission::p4_fixtures::SyntheticCpuPlan>),
}
impl PlanOwner {
    pub(crate) fn identity(&self) -> &str {
        match self {
            Self::Canonical(p) => p.digest(),
            #[cfg(test)]
            Self::Synthetic(p) => p.identity(),
        }
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Canonical(a), Self::Canonical(b)) => Arc::ptr_eq(a, b),
            #[cfg(test)]
            (Self::Synthetic(a), Self::Synthetic(b)) => Arc::ptr_eq(a, b),
            #[cfg(test)]
            _ => false,
        }
    }
    pub(crate) fn initial(&self) -> &[StateMetadata] {
        match self {
            Self::Canonical(p) => p.initial_states(),
            #[cfg(test)]
            Self::Synthetic(p) => std::slice::from_ref(&p.initial),
        }
    }
    pub(crate) fn transitions(&self) -> &[TransitionIndex] {
        match self {
            Self::Canonical(p) => p.transitions(),
            #[cfg(test)]
            Self::Synthetic(p) => &p.index,
        }
    }
    pub(crate) fn parts(&self) -> &[Vec<MicroPart>] {
        match self {
            Self::Canonical(p) => p.partitions(),
            #[cfg(test)]
            Self::Synthetic(p) => &p.parts,
        }
    }
    pub(crate) fn rates(&self) -> &[RateCoefficients<f64>] {
        match self {
            Self::Canonical(p) => p.full_rates(),
            #[cfg(test)]
            Self::Synthetic(p) => &p.rates,
        }
    }
    pub(crate) fn reference(&self) -> f64 {
        match self {
            Self::Canonical(p) => f64::from_bits(p.reference_bits()),
            #[cfg(test)]
            Self::Synthetic(p) => p.reference,
        }
    }
    pub(crate) fn check_shape(&self) -> Result<()> {
        require_single_stage(self.initial().len())?;
        ensure(
            !self.identity().is_empty() && self.reference().is_finite() && self.reference() > 0.,
            "plan identity/reference",
        )?;
        if let DiagonalOptimizer::Nag { reference, .. } = self.initial()[0].optimizer() {
            ensure(
                reference.to_bits() == self.reference().to_bits(),
                "full reference binding",
            )?;
        }
        ensure(
            self.parts().len() == self.transitions().len(),
            "transition cardinality",
        )?;
        ensure(
            !self.parts().is_empty() && self.rates().len() >= self.parts().len(),
            "consumption",
        )?;
        for (i, t) in self.transitions().iter().enumerate() {
            ensure(
                t.update() == i
                    && t.stage() == 0
                    && !t.is_boundary()
                    && t.update_metadata().is_some(),
                "single-stage transition ordering",
            )?;
        }
        Ok(())
    }
}
pub(crate) struct Description {
    pub(crate) geometry: StageGeometry,
    pub(crate) activations: Vec<DiagonalActivation>,
    pub(crate) transpose: TransposeKeys,
    pub(crate) frozen: bool,
}
pub(crate) fn activations(kinds: &[DiagonalActivation]) -> Vec<Box<dyn Activation>> {
    kinds
        .iter()
        .map(|a| match a {
            DiagonalActivation::Square => Box::new(Square) as Box<dyn Activation>,
            DiagonalActivation::PolyRelu2 => Box::new(PolyRelu2),
            DiagonalActivation::Identity => Box::new(Identity),
        })
        .collect()
}
/// Immutable caller-owned public evaluation material, with no secret or materializer.
pub struct CpuEvaluationContext {
    pub(crate) owner: Arc<()>,
    pub(crate) evaluator: Evaluator,
    pub(crate) encoder: Encoder,
    pub(crate) bootstrap: Option<CpuBootstrap>,
    pub(crate) metadata: Option<Arc<BootstrapMetadata>>,
    pub(crate) boot_params: Option<BootParams>,
}
impl CpuEvaluationContext {
    /// Forks only training evaluation counters; the private key-owner association is kept.
    pub fn fork(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            evaluator: self.evaluator.fork(),
            encoder: self.encoder.clone(),
            bootstrap: self.bootstrap.as_ref().map(CpuBootstrap::fork),
            metadata: self.metadata.clone(),
            boot_params: self.boot_params.clone(),
        }
    }
    /// Current measured counts, including bootstrap internals exactly once.
    pub fn counter(&self) -> OpCounter {
        self.evaluator.counter()
    }
    /// Immutable parameters used by all evaluation material in this context.
    pub fn parameters(&self) -> &Arc<CkksParams> {
        self.evaluator.params()
    }
    pub(crate) fn check_geometry(&self, g: &StageGeometry) -> Result<()> {
        ensure(
            self.evaluator.config() == &g.eval_config(),
            "evaluation configuration",
        )?;
        ensure(
            g.layout().slots == self.parameters().slots(),
            "layout parameters",
        )?;
        ensure(self.evaluator.relin_key().is_some(), "relinearization key")?;
        ensure(
            g.rotation_steps().iter().all(|&s| {
                self.evaluator
                    .galois_keys()
                    .is_some_and(|k| k.get(s).is_some())
            }),
            "training key manifest",
        )?;
        match g.mode() {
            DiagonalMode::Interactive => ensure(
                self.bootstrap.is_none() && self.metadata.is_none() && self.boot_params.is_none(),
                "interactive bootstrap absence",
            ),
            DiagonalMode::NonInteractive(_) => {
                let b = self
                    .bootstrap
                    .as_ref()
                    .ok_or_else(|| mismatch("bootstrap dependency"))?;
                let m = self
                    .metadata
                    .as_ref()
                    .ok_or_else(|| mismatch("bootstrap metadata"))?;
                ensure(
                    Arc::ptr_eq(b.real.params(), self.parameters())
                        && b.output_level() == m.output().level()
                        && b.real.cost_per_boot() == *m.cost_per_boot()
                        && self.boot_params.as_ref() == Some(b.real.boot_params())
                        && b.real.keys().rotation_steps() == m.generated_keys().rotation_steps(),
                    "bootstrap binding",
                )
            }
        }
    }
}
pub(crate) struct CpuBootstrap {
    pub(crate) real: Arc<Bootstrapper>,
    #[cfg(test)]
    pub(crate) budget: Option<std::rc::Rc<p4_tests::RuntimeBudget>>,
    #[cfg(test)]
    pub(crate) fail_after: Option<usize>,
    #[cfg(test)]
    pub(crate) successful: std::cell::Cell<usize>,
}
impl CpuBootstrap {
    fn fork(&self) -> Self {
        Self {
            real: self.real.clone(),
            #[cfg(test)]
            budget: self.budget.clone(),
            #[cfg(test)]
            fail_after: None,
            #[cfg(test)]
            successful: std::cell::Cell::new(0),
        }
    }
}
impl BootstrapOps<Evaluator> for CpuBootstrap {
    fn output_level(&self) -> usize {
        self.real.output_level()
    }
    fn bootstrap(&self, e: &Evaluator, c: &Ciphertext) -> Result<Ciphertext> {
        #[cfg(test)]
        {
            if self.fail_after.is_some_and(|n| self.successful.get() >= n) {
                return Err(mismatch("injected second bootstrap"));
            }
            if let Some(budget) = &self.budget {
                budget.reserve_boot();
            }
        }
        let result = self.real.bootstrap(e, c);
        #[cfg(test)]
        if result.is_ok() {
            self.successful.set(self.successful.get() + 1);
        }
        result
    }
}
/// Client-owned interactive service. The session borrows only its narrow handle.
pub struct ClientRefreshService {
    pub(crate) owner: Arc<()>,
    pub(crate) inner: RefCell<ckks_train_nn::refresh::ClientRefresher>,
    #[cfg(test)]
    pub(crate) fail_after: Option<usize>,
    #[cfg(test)]
    pub(crate) successful: std::cell::Cell<usize>,
}
struct RefreshHandle<'a>(&'a ClientRefreshService);
impl Refresher for RefreshHandle<'_> {
    fn output_level(&self) -> usize {
        self.0.inner.borrow().output_level()
    }
    fn bind_layout(&mut self, l: &Layout) -> Result<()> {
        self.0.inner.borrow_mut().bind_layout(l)
    }
    fn refresh(&mut self, c: &Ciphertext) -> Result<Ciphertext> {
        #[cfg(test)]
        if self
            .0
            .fail_after
            .is_some_and(|n| self.0.successful.get() >= n)
        {
            return Err(mismatch("injected second refresh"));
        }
        let result = self.0.inner.borrow_mut().refresh(c);
        #[cfg(test)]
        if result.is_ok() {
            self.0.successful.set(self.0.successful.get() + 1);
        }
        result
    }
}
/// Immutable client-produced model packet. It exposes no unchecked construction path.
pub struct EncryptedInitialState {
    pub(crate) owner: Arc<()>,
    pub(crate) plan: PlanOwner,
    pub(crate) description: Description,
    pub(crate) mlp: Mlp,
    pub(crate) compact: Option<CompactState>,
    pub(crate) batch: BatchSpec,
}
pub(crate) struct EncryptedMicro {
    pub(crate) part: MicroPart,
    pub(crate) x: Ciphertext,
    pub(crate) targets: Vec<Ciphertext>,
}
/// Immutable encrypted logical update, with private ordered micros and no supplied rate.
pub struct EncryptedLogicalUpdate {
    pub(crate) owner: Arc<()>,
    pub(crate) plan: PlanOwner,
    pub(crate) index: usize,
    pub(crate) total: usize,
    pub(crate) input_width: usize,
    pub(crate) target_width: usize,
    pub(crate) micros: Vec<EncryptedMicro>,
}
impl EncryptedLogicalUpdate {
    /// Consumed update index bound by the client.
    pub fn index(&self) -> usize {
        self.index
    }
    /// Number of logical valid rows, conserved across the ordered micros.
    pub fn total_valid(&self) -> usize {
        self.total
    }
}
/// Explicit client diagnostic endpoint; holds only encrypted dense layers.
pub struct EncryptedModelSnapshot {
    pub(crate) owner: Arc<()>,
    pub(crate) layers: Vec<Dense>,
    pub(crate) layout: Layout,
}
/// Successful update completion, distinct from a scientific measurement artifact.
#[derive(Debug)]
pub struct UpdateCompletion {
    /// Successful consumed updates, including this one.
    pub completed_updates: usize,
    /// Successful logical samples, including this update.
    pub processed_samples: usize,
    /// Actual seven-field measured delta, already inclusive of bootstrap internals.
    pub measured: OpCounter,
    /// Interactive refresh calls for this update.
    pub client_refreshes: usize,
    /// Actual publicly exposed runtime labelled-level diagnostics.
    pub trace: Vec<(String, usize)>,
}
/// Poison evidence from a failure after entering fallible configuration/training work.
#[derive(Debug)]
pub struct FailedUpdate {
    /// Update that remains uncommitted in successful progress.
    pub update: usize,
    /// Last entered bridge phase.
    pub phase: &'static str,
    /// Actual partial measured delta, with no rollback claim.
    pub measured: OpCounter,
    /// Available runtime diagnostics from the attempted step, absent for setter failure.
    pub trace: Vec<(String, usize)>,
}
enum Training<'a> {
    Interactive(Trainer<'a>),
    NonInteractive(NonInteractiveTrainer<'a, Evaluator, CpuBootstrap>),
}
/// Borrowed caller-context session over existing trainers, with poison-on-midwork failure.
pub struct CpuDiagonalSession<'a> {
    context: &'a CpuEvaluationContext,
    plan: PlanOwner,
    description: Description,
    mlp: Mlp,
    trainer: Training<'a>,
    batch: BatchSpec,
    completed: usize,
    samples: usize,
    poisoned: bool,
    failed: Option<FailedUpdate>,
}
impl<'a> CpuDiagonalSession<'a> {
    /// The sole supported execution shape.
    pub fn capability() -> CpuCapability {
        CpuCapability::SingleStage
    }
    /// Consumes a canonical client packet without granting production admission/start.
    pub fn new(
        context: &'a CpuEvaluationContext,
        plan: Arc<DiagonalPlan>,
        initial: EncryptedInitialState,
        refresh: Option<&'a ClientRefreshService>,
    ) -> Result<Self> {
        Self::from_owner(context, PlanOwner::Canonical(plan), initial, refresh)
    }
    pub(crate) fn from_owner(
        context: &'a CpuEvaluationContext,
        plan: PlanOwner,
        initial: EncryptedInitialState,
        refresh: Option<&'a ClientRefreshService>,
    ) -> Result<Self> {
        validate_initial_packet(context, &plan, &initial)?;
        let EncryptedInitialState {
            description,
            mlp,
            compact,
            batch,
            ..
        } = initial;
        let opt = plan.initial()[0].optimizer();
        let trainer = match description.geometry.mode() {
            DiagonalMode::Interactive => {
                ensure(compact.is_none(), "unexpected initial compact state")?;
                let service = refresh.ok_or_else(|| mismatch("interactive refresh dependency"))?;
                ensure(
                    Arc::ptr_eq(&context.owner, &service.owner),
                    "refresh context identity",
                )?;
                let optims = (0..mlp.layers.len())
                    .map(|_| -> Result<Box<dyn LayerOptimizer>> {
                        Ok(match opt {
                            DiagonalOptimizer::Sgd { lr } => Box::new(Sgd { lr }),
                            DiagonalOptimizer::Nag {
                                lr,
                                reference,
                                momentum,
                            } => {
                                let mut n = Nag::new(lr, momentum);
                                n.set_lr_reference(reference)?;
                                Box::new(n)
                            }
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                #[cfg(test)]
                p4_tests::reserve_encoder();
                let mut t = Trainer::new(
                    &context.evaluator,
                    Box::new(RefreshHandle(service)),
                    optims,
                    description.geometry.layout(),
                )?
                .with_transpose_keys(description.transpose.clone());
                if description.frozen {
                    t = t.with_frozen_biases();
                }
                Training::Interactive(t)
            }
            DiagonalMode::NonInteractive(placement) => {
                ensure(refresh.is_none(), "NI refresh capability absence")?;
                let state = compact.ok_or_else(|| mismatch("initial compact state"))?;
                let optim = match opt {
                    DiagonalOptimizer::Sgd { lr } => NiOptimizer::Sgd { lr },
                    DiagonalOptimizer::Nag { lr, momentum, .. } => {
                        NiOptimizer::Nag { lr, momentum }
                    }
                };
                #[cfg(test)]
                p4_tests::reserve_encoder();
                let mut t = NonInteractiveTrainer::new(
                    &context.evaluator,
                    context
                        .bootstrap
                        .as_ref()
                        .ok_or_else(|| mismatch("NI bootstrap dependency"))?,
                    optim,
                    description.geometry.layout(),
                    placement,
                    &mlp,
                    state,
                )?
                .with_transpose_keys(description.transpose.clone());
                if let DiagonalOptimizer::Nag { reference, .. } = opt {
                    t.set_lr_reference(reference)?;
                }
                if description.frozen {
                    t = t.with_frozen_biases();
                }
                Training::NonInteractive(t)
            }
        };
        let s = Self {
            context,
            plan,
            description,
            mlp,
            trainer,
            batch,
            completed: 0,
            samples: 0,
            poisoned: false,
            failed: None,
        };
        s.check_state(&s.plan.initial()[0])?;
        s.check_actual_batch()?;
        Ok(s)
    }
    /// Successful progress only; prework and poisoned failures never advance it.
    pub fn completed_updates(&self) -> usize {
        self.completed
    }
    /// Successful logical sample progress only.
    pub fn processed_samples(&self) -> usize {
        self.samples
    }
    /// Whether a prior midwork failure permanently prevents reuse.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Preserved actual partial-work evidence, if any.
    pub fn failed_update(&self) -> Option<&FailedUpdate> {
        self.failed.as_ref()
    }
    /// Copies encrypted dense state to an explicit diagnostic endpoint, without evaluation.
    pub fn snapshot(&self) -> EncryptedModelSnapshot {
        EncryptedModelSnapshot {
            owner: self.context.owner.clone(),
            layers: self.mlp.layers.iter().map(|(d, _)| d.clone()).collect(),
            layout: self.description.geometry.layout(),
        }
    }
    fn check_actual_batch(&self) -> Result<()> {
        let before = self.context.counter();
        let probe = self.context.evaluator.fork();
        let actual = match &self.trainer {
            Training::Interactive(t) => {
                self.mlp
                    .batch_spec_at(&probe, &self.context.encoder, t.refresh_level())?
            }
            Training::NonInteractive(t) => t.batch_spec(&self.mlp)?,
        };
        ensure(self.context.counter() == before, "probe counter isolation")?;
        check_batch(actual, self.plan.initial()[0].batch())?;
        ensure(actual == self.batch, "retained batch specification")
    }
    fn check_state(&self, state: &StateMetadata) -> Result<()> {
        ensure(
            state.completed_updates() == self.completed,
            "state successful index",
        )?;
        self.check_state_without_progress(state)
    }
    fn check_state_without_progress(&self, state: &StateMetadata) -> Result<()> {
        check_mlp(self.context, &self.description, &self.mlp, state)?;
        check_batch(self.batch, state.batch())?;
        match &self.trainer {
            Training::Interactive(t) => {
                ensure(
                    state.compact_weights().is_empty()
                        && state.velocity().len() == self.mlp.layers.len(),
                    "interactive state cardinalities",
                )?;
                for (i, (dense, _)) in self.mlp.layers.iter().enumerate() {
                    let v = t.optimizer_view(i)?;
                    check_optimizer(state.optimizer(), v.lr(), v.reference(), v.momentum())?;
                    match (v.velocity(), state.velocity()[i]) {
                        (None, None) => {}
                        (Some(actual), Some(expected)) => {
                            ensure(
                                actual.d() == dense.w.diagonals().d()
                                    && actual.stride() == dense.w.diagonals().stride()
                                    && actual.entries().len()
                                        == dense.w.diagonals().entries().len(),
                                "velocity geometry",
                            )?;
                            for c in actual.entries() {
                                check_ct(self.context, c, expected)?;
                            }
                        }
                        _ => return Err(mismatch("interactive velocity presence")),
                    }
                }
                ensure(
                    t.optimizer_view(self.mlp.layers.len()).is_err(),
                    "optimizer bounds",
                )?;
            }
            Training::NonInteractive(t) => {
                let o = t.optimizer();
                let (reference, momentum) = match o {
                    NiOptimizer::Sgd { .. } => (None, None),
                    NiOptimizer::Nag { momentum, .. } => (Some(t.lr_reference()), Some(momentum)),
                };
                check_optimizer(state.optimizer(), o.lr(), reference, momentum)?;
                ensure(
                    t.state().weights.len() == state.compact_weights().len()
                        && t.state().velocity.len() == state.velocity().len(),
                    "compact state cardinality",
                )?;
                for (c, &m) in t.state().weights.iter().zip(state.compact_weights()) {
                    check_ct(self.context, c, m)?;
                }
                for (actual, expected) in t.state().velocity.iter().zip(state.velocity()) {
                    match (actual, expected) {
                        (None, None) => {}
                        (Some(c), Some(m)) => check_ct(self.context, c, *m)?,
                        _ => return Err(mismatch("NI velocity presence")),
                    }
                }
                ensure(
                    Some(t.compact_layout()) == self.description.geometry.compact(),
                    "compact geometry and bias slots",
                )?;
            }
        }
        Ok(())
    }
    fn validate_packet(&self, p: &EncryptedLogicalUpdate) -> Result<(usize, usize)> {
        ensure(!self.poisoned, "poisoned session")?;
        ensure(
            self.completed < self.plan.parts().len(),
            "exhausted session",
        )?;
        ensure(p.index == self.completed, "update index or replay")?;
        ensure(self.plan.same(&p.plan), "packet plan identity")?;
        ensure(
            Arc::ptr_eq(&p.owner, &self.context.owner),
            "packet context identity",
        )?;
        self.context.check_geometry(&self.description.geometry)?;
        let t = &self.plan.transitions()[self.completed];
        let u = t
            .update_metadata()
            .ok_or_else(|| mismatch("update metadata"))?;
        self.check_state(u.pre_state())?;
        let parts = &self.plan.parts()[self.completed];
        ensure(
            p.micros.len() == parts.len() && p.micros.len() == u.micros().len(),
            "micro cardinality",
        )?;
        let dims = self.description.geometry.dims();
        ensure(
            p.input_width == dims[0] && p.target_width == *dims.last().unwrap(),
            "target/input shape",
        )?;
        let mut total = 0usize;
        for ((m, part), expected) in p.micros.iter().zip(parts).zip(u.micros()) {
            ensure(
                m.part == *part
                    && m.part.start == total
                    && m.part.valid == expected.n_valid
                    && m.part.valid > 0
                    && m.part.valid <= self.description.geometry.layout().batch,
                "ordered partition",
            )?;
            total = total
                .checked_add(m.part.valid)
                .ok_or_else(|| mismatch("partition overflow"))?;
            ensure(m.targets.len() == 1, "stage target cardinality")?;
            check_ct(self.context, &m.x, expected.x)?;
            check_ct(self.context, &m.targets[0], expected.y)?;
        }
        ensure(total == p.total, "logical valid denominator")?;
        let completed = self
            .completed
            .checked_add(1)
            .ok_or_else(|| mismatch("update progress overflow"))?;
        let samples = self
            .samples
            .checked_add(total)
            .ok_or_else(|| mismatch("sample progress overflow"))?;
        let rate = self.plan.rates()[self.completed].kernel;
        match &self.trainer {
            Training::Interactive(t) => t.validate_lr(rate)?,
            Training::NonInteractive(t) => t.validate_lr(rate)?,
        }
        Ok((completed, samples))
    }
    /// Validates immutable binding before setters/work; invokes one existing logical step.
    /// Any error after preflight poisons the session and preserves actual partial counts.
    pub fn step(&mut self, p: &EncryptedLogicalUpdate) -> Result<UpdateCompletion> {
        let (next, samples) = self.validate_packet(p)?;
        let origin = self.context.counter();
        let trace_origin = self.trace().to_vec();
        let boots_before = match &self.trainer {
            Training::NonInteractive(t) => (t.weight_bootstraps(), t.activation_bootstraps()),
            _ => (0, 0),
        };
        let refresh_before = match &self.trainer {
            Training::Interactive(t) => t.refreshes(),
            _ => 0,
        };
        self.poisoned = true;
        self.failed = Some(FailedUpdate {
            update: self.completed,
            phase: "configuration",
            measured: OpCounter::default(),
            trace: vec![],
        });
        let result = (|| {
            let rate = self.plan.rates()[self.completed].kernel;
            match &mut self.trainer {
                Training::Interactive(t) => t.set_lr(rate)?,
                Training::NonInteractive(t) => {
                    t.set_lr(rate)?;
                    t.set_next_batch_full(self.plan.transitions()[self.completed].next_full());
                }
            }
            self.check_coefficients(p.total)?;
            let batches: Vec<_> = p
                .micros
                .iter()
                .map(|m| (&m.x, &m.targets[0], m.part.valid))
                .collect();
            self.failed.as_mut().unwrap().phase = "trainer";
            #[cfg(test)]
            p4_tests::reserve_attempt();
            match &mut self.trainer {
                Training::Interactive(t) => t.train_step_valid(&mut self.mlp, &batches)?,
                Training::NonInteractive(t) => t.train_step_valid(&mut self.mlp, &batches)?,
            }
            self.failed.as_mut().unwrap().phase = "postcheck";
            let actual = &self.context.counter() - &origin;
            let u = self.plan.transitions()[self.completed]
                .update_metadata()
                .unwrap();
            ensure(
                actual == *u.counts_including_bootstrap(),
                "seven inclusive measured counters",
            )?;
            let refreshes = match &self.trainer {
                Training::Interactive(t) => t.refreshes() - refresh_before,
                _ => 0,
            };
            ensure(
                refreshes == u.client_refreshes(),
                "client refresh decisions",
            )?;
            ensure(
                actual.bootstraps as usize == u.bootstrap_count(),
                "bootstrap decisions",
            )?;
            if let Training::NonInteractive(t) = &self.trainer {
                ensure(
                    t.last_update_rescaled() == u.reference_branch()
                        && t.last_update_booted() == u.state_bootstrapped(),
                    "NI reference/state bootstrap decisions",
                )?;
                let activation_boots = u
                    .canonical_activations()
                    .iter()
                    .map(Vec::len)
                    .sum::<usize>();
                ensure(
                    t.activation_bootstraps() - boots_before.1 == activation_boots as u64
                        && t.weight_bootstraps() - boots_before.0
                            == (u.bootstrap_count() - activation_boots) as u64,
                    "NI weight/activation bootstrap decisions",
                )?;
            }
            self.check_hidden_trace()?;
            // Compare post state's successful index without publishing progress until all checks pass.
            self.check_post_state(u.post_state(), next)?;
            self.check_actual_batch()?;
            Ok(UpdateCompletion {
                completed_updates: next,
                processed_samples: samples,
                measured: actual,
                client_refreshes: refreshes,
                trace: self.trace().to_vec(),
            })
        })();
        match result {
            Ok(c) => {
                self.completed = next;
                self.samples = samples;
                self.poisoned = false;
                self.failed = None;
                Ok(c)
            }
            Err(e) => {
                let trace = if self.failed.as_ref().unwrap().phase == "configuration"
                    || self.trace() == trace_origin
                {
                    vec![]
                } else {
                    self.trace().to_vec()
                };
                let failure = self.failed.as_mut().unwrap();
                failure.measured = &self.context.counter() - &origin;
                failure.trace = trace;
                Err(e)
            }
        }
    }
    fn check_post_state(&self, state: &StateMetadata, next: usize) -> Result<()> {
        // check_state reads self.completed; use the identical checker with an explicit index.
        ensure(state.completed_updates() == next, "post successful index")?;
        self.check_state_without_progress(state)
    }
    fn trace(&self) -> &[(String, usize)] {
        match &self.trainer {
            Training::Interactive(t) => t.trace(),
            Training::NonInteractive(t) => t.trace(),
        }
    }
    fn check_coefficients(&self, total: usize) -> Result<()> {
        let u = self.plan.transitions()[self.completed]
            .update_metadata()
            .unwrap();
        let c = u.coefficients();
        let n = total as f64;
        let rate = self.plan.rates()[self.completed].kernel;
        match &self.trainer {
            Training::Interactive(t) => {
                for i in 0..self.mlp.layers.len() {
                    let v = t.optimizer_view(i)?;
                    ensure(
                        v.lr().to_bits() == rate.to_bits(),
                        "configured current rate",
                    )?;
                    ensure(
                        (v.weight_factor() / n).to_bits() == c.current_bits
                            && v.reference_factor().map(|x| (x / n).to_bits()) == c.reference_bits
                            && (if self.description.frozen {
                                0.
                            } else {
                                v.bias_factor() / n
                            })
                            .to_bits()
                                == c.bias_bits,
                        "configured optimizer folded factors",
                    )?;
                    ensure(
                        v.momentum().map(f64::to_bits) == c.velocity_mu_bits,
                        "configured momentum coefficient",
                    )?;
                    ensure(
                        v.reference_factor().is_some() == u.reference_branch()
                            && (v.momentum().is_some() && v.velocity().is_none())
                                == u.first_nag_update(),
                        "interactive reference and first velocity branches",
                    )?;
                }
            }
            Training::NonInteractive(t) => {
                let o = t.optimizer();
                let reference = if o.has_velocity() && rate / t.lr_reference() != 1. {
                    Some(o.with_lr(t.lr_reference()).grad_scale() / n)
                } else {
                    None
                };
                ensure(
                    o.lr().to_bits() == rate.to_bits()
                        && (o.grad_scale() / n).to_bits() == c.current_bits
                        && reference.map(f64::to_bits) == c.reference_bits
                        && (if self.description.frozen {
                            0.
                        } else {
                            o.bias_grad_scale() / n
                        })
                        .to_bits()
                            == c.bias_bits,
                    "NI configured optimizer folded factors",
                )?;
                let momentum = match o {
                    NiOptimizer::Nag { momentum, .. } => Some(momentum),
                    _ => None,
                };
                ensure(
                    momentum.map(f64::to_bits) == c.velocity_mu_bits,
                    "NI configured momentum coefficient",
                )?;
                ensure(
                    reference.is_some() == u.reference_branch()
                        && (o.has_velocity() && t.state().velocity.iter().all(Option::is_none))
                            == u.first_nag_update(),
                    "NI configured branches",
                )?;
            }
        }
        Ok(())
    }
    fn check_hidden_trace(&self) -> Result<()> {
        let u = self.plan.transitions()[self.completed]
            .update_metadata()
            .unwrap();
        ensure(
            u.retained_hidden().len() == u.micros().len(),
            "hidden micro cardinality",
        )?;
        let count = match &self.trainer {
            Training::Interactive(_) => u.micros().len(),
            _ => 1,
        };
        for i in 0..count {
            let labels = &u.retained_hidden()[i];
            ensure(
                labels.len() + 1 == self.mlp.layers.len(),
                "hidden layer cardinality",
            )?;
            for (l, m) in labels.iter().enumerate() {
                let label = if matches!(&self.trainer, Training::Interactive(_)) && count > 1 {
                    format!("micro{i}/a{}", l + 1)
                } else {
                    format!("a{}", l + 1)
                };
                let actual: Vec<_> = self.trace().iter().filter(|(s, _)| s == &label).collect();
                ensure(
                    actual.len() == 1 && actual[0].1 == m.level(),
                    "hidden trace level",
                )?;
            }
        }
        if let Training::NonInteractive(t) = &self.trainer {
            let actual = t.last_activation_bootstraps();
            let canonical = u.canonical_activations();
            if t.placement() == ckks_train_nn::trainer_ni::BootPlacement::AfterActivation {
                ensure(
                    actual.len() == canonical.len(),
                    "AA canonical micro cardinality",
                )?;
                for (a, e) in actual.iter().zip(canonical) {
                    ensure(a.len() == e.len(), "AA canonical layer cardinality")?;
                    for (c, &m) in a.iter().zip(e) {
                        check_ct(self.context, c, m)?;
                    }
                }
                for (l, m) in canonical[0].iter().enumerate() {
                    let label = format!("a{}_boot", l + 1);
                    let levels = self
                        .trace()
                        .iter()
                        .filter(|(s, _)| s == &label)
                        .collect::<Vec<_>>();
                    ensure(
                        levels.len() == 1 && levels[0].1 == m.level(),
                        "AA canonical trace level",
                    )?;
                }
            } else {
                ensure(
                    actual.is_empty() && canonical.iter().all(Vec::is_empty),
                    "non-AA canonical absence",
                )?;
            }
        }
        Ok(())
    }
}
pub(crate) fn validate_initial_packet(
    context: &CpuEvaluationContext,
    plan: &PlanOwner,
    initial: &EncryptedInitialState,
) -> Result<()> {
    plan.check_shape()?;
    ensure(
        Arc::ptr_eq(&context.owner, &initial.owner),
        "initial context identity",
    )?;
    ensure(plan.same(&initial.plan), "initial plan identity")?;
    context.check_geometry(&initial.description.geometry)?;
    let state = &plan.initial()[0];
    check_mlp(context, &initial.description, &initial.mlp, state)?;
    check_batch(initial.batch, state.batch())?;
    match &initial.compact {
        None => ensure(
            state.compact_weights().is_empty() && state.velocity().iter().all(Option::is_none),
            "initial compact absence",
        )?,
        Some(c) => {
            ensure(
                c.weights.len() == state.compact_weights().len()
                    && c.velocity.len() == state.velocity().len(),
                "initial compact cardinality",
            )?;
            for (ct, &meta) in c.weights.iter().zip(state.compact_weights()) {
                check_ct(context, ct, meta)?;
            }
            for (ct, meta) in c.velocity.iter().zip(state.velocity()) {
                match (ct, meta) {
                    (None, None) => {}
                    (Some(ct), Some(meta)) => check_ct(context, ct, *meta)?,
                    _ => return Err(mismatch("initial velocity presence")),
                }
            }
        }
    }
    Ok(())
}
pub(crate) fn check_ct(context: &CpuEvaluationContext, c: &Ciphertext, m: CtMeta) -> Result<()> {
    ensure(
        c.len() == 2 && c.level() == m.level() && c.scale().to_bits() == m.scale_bits(),
        "ciphertext level/scale/components",
    )?;
    ensure(
        c.polys().iter().all(|p| {
            p.is_ntt_form()
                && p.level() == c.level()
                && Arc::ptr_eq(p.basis(), context.parameters().basis_q())
        }),
        "ciphertext basis",
    )
}
fn check_batch(b: BatchSpec, m: ckks_train_nn::diagonal_plan::BatchMetadata) -> Result<()> {
    ensure(
        b.x_level == m.x.level()
            && b.x_scale.to_bits() == m.x.scale_bits()
            && b.y_level == m.y.level()
            && b.y_scale.to_bits() == m.y.scale_bits(),
        "BatchSpec all fields",
    )
}
fn check_optimizer(
    o: DiagonalOptimizer,
    lr: f64,
    reference: Option<f64>,
    momentum: Option<f64>,
) -> Result<()> {
    let (expected, r, m) = match o {
        DiagonalOptimizer::Sgd { lr } => (lr, None, None),
        DiagonalOptimizer::Nag {
            lr,
            reference,
            momentum,
        } => (lr, Some(reference), Some(momentum)),
    };
    ensure(
        lr.to_bits() == expected.to_bits()
            && reference.map(f64::to_bits) == r.map(f64::to_bits)
            && momentum.map(f64::to_bits) == m.map(f64::to_bits),
        "optimizer raw config",
    )
}
fn check_mlp(
    context: &CpuEvaluationContext,
    description: &Description,
    mlp: &Mlp,
    s: &StateMetadata,
) -> Result<()> {
    let g = &description.geometry;
    ensure(
        mlp.dims() == g.dims()
            && mlp.layers.len() == s.weights().len()
            && mlp.layers.len() == s.biases().len()
            && mlp.layers.len() == g.shapes().len()
            && mlp.layers.len() == description.activations.len(),
        "expanded state cardinalities",
    )?;
    for (i, (dense, act)) in mlp.layers.iter().enumerate() {
        let expected = &g.shapes()[i];
        let shape = dense.w.shape(&g.layout())?;
        ensure(
            &shape == expected
                && dense.w.diagonals().d() == expected.m
                && dense.w.diagonals().stride() == g.layout().batch
                && dense.w.diagonals().entries().len() == expected.m,
            "expanded diagonal geometry",
        )?;
        let kinds = activations(&[description.activations[i]]);
        let expected_act = &kinds[0];
        ensure(
            act.depth() == expected_act.depth()
                && act.derivative_depth() == expected_act.derivative_depth()
                && act.constant_derivative() == expected_act.constant_derivative(),
            "activation binding",
        )?;
        for c in dense.w.diagonals().entries() {
            check_ct(context, c, s.weights()[i])?;
        }
        check_ct(context, &dense.b, s.biases()[i])?;
    }
    Ok(())
}
#[cfg(test)]
pub(crate) mod p4_tests;
