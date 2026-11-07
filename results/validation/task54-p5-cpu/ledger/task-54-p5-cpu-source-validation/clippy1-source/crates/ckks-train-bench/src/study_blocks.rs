//! Actual multi-stage CPU consuming route, with no owned D7 admission/start authority.
//! Plaintext data and secret material stay with `StudyClient`. Counters are whole-step
//! inclusive deltas, never separately measured per-stage windows.
pub use crate::study_client::blocks::{BlockEncryptionBinding, ClientBlockSnapshot};
use crate::{
    study_admission::{DiagonalPlan, MicroPart, StateMetadata, TransitionIndex},
    study_diagonal::{
        ClientRefreshService, CpuEvaluationContext, Description, EncryptedMicro,
        EncryptedModelSnapshot, check_batch, check_ct, check_interactive_stage, check_mlp,
        check_ni_stage, ensure, mismatch,
    },
    study_inputs::RateCoefficients,
};
use ckks_train_ckks::{BootstrapOps, Ciphertext, Evaluator, HeOps, OpCounter};
use ckks_train_core::error::Result;
use ckks_train_nn::{
    block::{BlockMlp, BlockNiTrainer, BlockTrainer},
    diagonal_plan::{DiagonalMode, DiagonalOptimizer},
    layout::Layout,
    optim::{Nag, Sgd},
    refresh::Refresher,
    trainer::{BatchSpec, LayerOptimizer, Trainer},
    trainer_ni::{BootPlacement, CompactState, NiOptimizer, NonInteractiveTrainer},
};
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

/// Closed ownership domains; synthetic plans cannot be promoted to canonical values.
#[derive(Clone)]
pub(crate) enum BlockPlanOwner {
    Canonical(Arc<DiagonalPlan>),
    #[cfg(test)]
    Synthetic(Arc<crate::study_admission::p5_fixtures::SyntheticBlockPlan>),
}
impl BlockPlanOwner {
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Canonical(a), Self::Canonical(b)) => Arc::ptr_eq(a, b),
            #[cfg(test)]
            (Self::Synthetic(a), Self::Synthetic(b)) => Arc::ptr_eq(a, b),
            #[cfg(test)]
            _ => false,
        }
    }
    pub(crate) fn identity(&self) -> &str {
        match self {
            Self::Canonical(p) => p.digest(),
            #[cfg(test)]
            Self::Synthetic(p) => p.identity(),
        }
    }
    pub(crate) fn initial(&self) -> &[StateMetadata] {
        match self {
            Self::Canonical(p) => p.initial_states(),
            #[cfg(test)]
            Self::Synthetic(p) => &p.initial,
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
    fn stride(&self) -> usize {
        2 * self.initial().len() - 1
    }
    pub(crate) fn transition(&self, update: usize, stage: usize) -> &TransitionIndex {
        &self.transitions()[update * self.stride() + 2 * stage]
    }
    pub(crate) fn boundary(&self, update: usize, stage: usize) -> &TransitionIndex {
        &self.transitions()[update * self.stride() + 2 * stage + 1]
    }
    pub(crate) fn check_shape(&self) -> Result<()> {
        let stages = self.initial().len();
        ensure((2..=3).contains(&stages), "block stage cardinality")?;
        ensure(
            !self.parts().is_empty() && self.rates().len() >= self.parts().len(),
            "block schedule cardinality",
        )?;
        let expected = self
            .parts()
            .len()
            .checked_mul(self.stride())
            .ok_or_else(|| mismatch("block transition overflow"))?;
        ensure(
            self.transitions().len() == expected,
            "block transition cardinality",
        )?;
        for update in 0..self.parts().len() {
            let next = self.transition(update, 0).next_full();
            for stage in 0..stages {
                let t = self.transition(update, stage);
                ensure(
                    t.update() == update
                        && t.stage() == stage
                        && !t.is_boundary()
                        && t.update_metadata().is_some()
                        && t.handoff_metadata().is_none()
                        && t.next_full() == next,
                    "block typed update order/lookahead",
                )?;
                if stage + 1 < stages {
                    let t = self.boundary(update, stage);
                    ensure(
                        t.update() == update
                            && t.stage() == stage
                            && t.is_boundary()
                            && t.handoff_metadata().is_some()
                            && t.update_metadata().is_none(),
                        "block typed handoff order",
                    )?;
                }
            }
        }
        ensure(
            self.reference().is_finite() && self.reference() > 0.,
            "block reference",
        )
    }
}
/// Immutable client-produced block model, with no unchecked public constructor.
pub struct EncryptedBlockInitial {
    pub(crate) owner: Arc<()>,
    pub(crate) plan: BlockPlanOwner,
    pub(crate) descriptions: Vec<Description>,
    pub(crate) classes: Vec<usize>,
    pub(crate) model: BlockMlp,
    pub(crate) compact: Option<Vec<CompactState>>,
    pub(crate) batches: Vec<BatchSpec>,
}
/// Immutable first input plus one independently encrypted target per stage and micro.
pub struct EncryptedBlockUpdate {
    pub(crate) owner: Arc<()>,
    pub(crate) plan: BlockPlanOwner,
    pub(crate) index: usize,
    pub(crate) total: usize,
    pub(crate) input_width: usize,
    pub(crate) target_widths: Vec<usize>,
    pub(crate) micros: Vec<EncryptedMicro>,
}
/// Explicit encrypted endpoint copies; each ciphertext clone allocates its payload.
pub struct EncryptedBlockSnapshot {
    pub(crate) stages: Vec<EncryptedModelSnapshot>,
    pub(crate) handoffs: Vec<Vec<Ciphertext>>,
    pub(crate) valid: Vec<usize>,
    pub(crate) widths: Vec<usize>,
}
/// One successful logical block update, after every typed postcheck.
#[derive(Debug)]
pub struct BlockCompletion {
    /// Successful block update count.
    pub completed_updates: usize,
    /// Successful logical sample count.
    pub processed_samples: usize,
    /// Whole-step inclusive seven-counter delta.
    pub measured: OpCounter,
    /// Actual client refresh delta including boundaries.
    pub client_refreshes: usize,
    /// Available per-stage existing trainer traces, without per-stage measured windows.
    pub traces: Vec<Vec<(String, usize)>>,
}
/// Actual prefix evidence after a permanently poisoned attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockFailure {
    /// Zero-based attempted update.
    pub update: usize,
    /// Last reached configuration, trainer or postcheck phase.
    pub phase: &'static str,
    /// Actual inclusive partial counter delta; no rollback is implied.
    pub measured: OpCounter,
    /// Existing traces changed by this attempt, in stage order.
    pub traces: Vec<Vec<(String, usize)>>,
    /// Successful I boundary outputs retained before failure; NI publishes none on failure.
    pub retained_boundary_outputs: usize,
}
/// Caller-owned sized facade over the same real public evaluation/bootstrap material.
/// It must outlive sessions, preventing self-referential trainer storage.
pub struct CpuBlockEvaluation<'a> {
    pub(crate) base: &'a CpuEvaluationContext,
    boundary: RefCell<Vec<Ciphertext>>,
    refreshes: Cell<usize>,
    boots: Cell<usize>,
    boundary_cap: Cell<usize>,
    active: Cell<bool>,
    #[cfg(test)]
    pub(crate) fail_refresh_after: Cell<Option<usize>>,
    #[cfg(test)]
    pub(crate) fail_boot_after: Cell<Option<usize>>,
}
impl<'a> CpuBlockEvaluation<'a> {
    /// Borrows the caller-owned base context; no new key, encoder or Bootstrapper.
    pub fn new(base: &'a CpuEvaluationContext) -> Self {
        Self {
            base,
            boundary: RefCell::new(Vec::new()),
            refreshes: Cell::new(0),
            boots: Cell::new(0),
            boundary_cap: Cell::new(0),
            active: Cell::new(false),
            #[cfg(test)]
            fail_refresh_after: Cell::new(None),
            #[cfg(test)]
            fail_boot_after: Cell::new(None),
        }
    }
    fn begin(&self, cap: usize) {
        self.boundary.borrow_mut().clear();
        self.boundary_cap.set(cap);
    }
}
impl BootstrapOps<Evaluator> for CpuBlockEvaluation<'_> {
    fn output_level(&self) -> usize {
        self.base
            .bootstrap
            .as_ref()
            .map_or(0, |b| b.real.output_level())
    }
    fn bootstrap(&self, e: &Evaluator, c: &Ciphertext) -> Result<Ciphertext> {
        #[cfg(test)]
        {
            ensure(
                !self
                    .fail_boot_after
                    .get()
                    .is_some_and(|n| self.boots.get() >= n),
                "injected second block boundary bootstrap",
            )?;
            p5_tests::reserve_boot();
        }
        let out = self
            .base
            .bootstrap
            .as_ref()
            .ok_or_else(|| mismatch("block bootstrap dependency"))?
            .real
            .bootstrap(e, c)?;
        self.boots.set(
            self.boots
                .get()
                .checked_add(1)
                .ok_or_else(|| mismatch("block bootstrap overflow"))?,
        );
        Ok(out)
    }
}
struct BlockRefresh<'a> {
    facade: &'a CpuBlockEvaluation<'a>,
    service: &'a ClientRefreshService,
    boundary: bool,
}
impl Refresher for BlockRefresh<'_> {
    fn output_level(&self) -> usize {
        self.service.handle().output_level()
    }
    fn bind_layout(&mut self, l: &Layout) -> Result<()> {
        self.service.handle().bind_layout(l)
    }
    fn refresh(&mut self, c: &Ciphertext) -> Result<Ciphertext> {
        #[cfg(test)]
        {
            ensure(
                !self
                    .facade
                    .fail_refresh_after
                    .get()
                    .is_some_and(|n| self.facade.refreshes.get() >= n),
                "injected second block boundary refresh",
            )?;
            p5_tests::reserve_refresh();
        }
        if self.boundary {
            ensure(
                self.facade.boundary.borrow().len() < self.facade.boundary_cap.get(),
                "block boundary retention bound",
            )?;
        }
        let out = self.service.handle().refresh(c)?;
        self.facade.refreshes.set(
            self.facade
                .refreshes
                .get()
                .checked_add(1)
                .ok_or_else(|| mismatch("block refresh overflow"))?,
        );
        if self.boundary {
            self.facade.boundary.borrow_mut().push(out.clone());
        }
        Ok(out)
    }
}
enum BlockTraining<'a> {
    Interactive(BlockTrainer<'a>),
    NonInteractive(BlockNiTrainer<'a, Evaluator, CpuBlockEvaluation<'a>>),
}
/// Actual multi-stage session. It contains encrypted state and public evaluation handles only.
pub struct CpuBlockSession<'a> {
    facade: &'a CpuBlockEvaluation<'a>,
    plan: BlockPlanOwner,
    descriptions: Vec<Description>,
    classes: Vec<usize>,
    model: BlockMlp,
    trainer: BlockTraining<'a>,
    batches: Vec<BatchSpec>,
    completed: usize,
    samples: usize,
    poisoned: bool,
    failed: Option<BlockFailure>,
}
impl<'a> CpuBlockSession<'a> {
    /// Consumes an actual canonical block packet; this grants no D7 start capability.
    pub fn new(
        facade: &'a CpuBlockEvaluation<'a>,
        plan: Arc<DiagonalPlan>,
        initial: EncryptedBlockInitial,
        refresh: Option<&'a ClientRefreshService>,
    ) -> Result<Self> {
        Self::from_owner(facade, BlockPlanOwner::Canonical(plan), initial, refresh)
    }
    pub(crate) fn from_owner(
        facade: &'a CpuBlockEvaluation<'a>,
        plan: BlockPlanOwner,
        initial: EncryptedBlockInitial,
        refresh: Option<&'a ClientRefreshService>,
    ) -> Result<Self> {
        plan.check_shape()?;
        ensure(
            !facade.active.get(),
            "block facade already borrowed by session",
        )?;
        let context = facade.base;
        let n = plan.initial().len();
        ensure(
            Arc::ptr_eq(&context.owner, &initial.owner) && plan.same(&initial.plan),
            "block initial association",
        )?;
        ensure(
            initial.descriptions.len() == n
                && initial.classes.len() == n
                && initial.batches.len() == n
                && initial.model.num_stages() == n,
            "block initial cardinality",
        )?;
        let mode = initial.descriptions[0].geometry.mode();
        let layout = initial.descriptions[0].geometry.layout();
        // Every stage and every initial CT is checked before any Trainer construction.
        for stage in 0..n {
            let description = &initial.descriptions[stage];
            let state = &plan.initial()[stage];
            context.check_geometry(&description.geometry)?;
            ensure(
                description.geometry.mode() == mode && description.geometry.layout() == layout,
                "block homogeneous mode/layout",
            )?;
            ensure(
                initial.classes[stage] > 0
                    && initial.classes[stage] <= *description.geometry.dims().last().unwrap(),
                "block stage classes",
            )?;
            check_mlp(context, description, initial.model.stage(stage), state)?;
            check_batch(initial.batches[stage], state.batch())?;
            ensure(
                state.completed_updates() == 0 && state.velocity().iter().all(Option::is_none),
                "block initial progress/velocity",
            )?;
            match &initial.compact {
                None => ensure(
                    state.compact_weights().is_empty() && mode == DiagonalMode::Interactive,
                    "block initial compact absence",
                )?,
                Some(states) => {
                    ensure(states.len() == n, "block compact stage cardinality")?;
                    let actual = &states[stage];
                    ensure(
                        actual.weights.len() == state.compact_weights().len()
                            && actual.velocity.len() == state.velocity().len(),
                        "block initial compact cardinality",
                    )?;
                    for (ct, &m) in actual.weights.iter().zip(state.compact_weights()) {
                        check_ct(context, ct, m)?;
                    }
                    ensure(
                        actual.velocity.iter().all(Option::is_none),
                        "block initial compact velocity absence",
                    )?;
                }
            }
        }
        let EncryptedBlockInitial {
            descriptions,
            classes,
            model,
            compact,
            batches,
            ..
        } = initial;
        let trainer = match mode {
            DiagonalMode::Interactive => {
                let service = refresh.ok_or_else(|| mismatch("block refresh dependency"))?;
                ensure(
                    Arc::ptr_eq(&service.owner, &context.owner),
                    "block refresh identity",
                )?;
                let mut stages = Vec::with_capacity(n);
                for stage in 0..n {
                    let opt = plan.initial()[stage].optimizer();
                    let optims = (0..model.stage(stage).layers.len())
                        .map(|_| -> Result<Box<dyn LayerOptimizer>> {
                            Ok(match opt {
                                DiagonalOptimizer::Sgd { lr } => Box::new(Sgd { lr }),
                                DiagonalOptimizer::Nag {
                                    lr,
                                    reference,
                                    momentum,
                                } => {
                                    let mut nag = Nag::new(lr, momentum);
                                    nag.set_lr_reference(reference)?;
                                    Box::new(nag)
                                }
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    #[cfg(test)]
                    p5_tests::reserve_encoder();
                    let mut t = Trainer::new(
                        &context.evaluator,
                        Box::new(BlockRefresh {
                            facade,
                            service,
                            boundary: false,
                        }),
                        optims,
                        layout,
                    )?
                    .with_transpose_keys(descriptions[stage].transpose.clone());
                    if descriptions[stage].frozen {
                        t = t.with_frozen_biases();
                    }
                    stages.push(t);
                }
                BlockTraining::Interactive(BlockTrainer::new(
                    &context.evaluator,
                    stages,
                    Box::new(BlockRefresh {
                        facade,
                        service,
                        boundary: true,
                    }),
                )?)
            }
            DiagonalMode::NonInteractive(placement) => {
                ensure(refresh.is_none(), "block NI refresh absence")?;
                let states = compact.ok_or_else(|| mismatch("block compact dependency"))?;
                let mut stages = Vec::with_capacity(n);
                for (stage, state) in states.into_iter().enumerate() {
                    let opt = plan.initial()[stage].optimizer();
                    let optim = match opt {
                        DiagonalOptimizer::Sgd { lr } => NiOptimizer::Sgd { lr },
                        DiagonalOptimizer::Nag { lr, momentum, .. } => {
                            NiOptimizer::Nag { lr, momentum }
                        }
                    };
                    #[cfg(test)]
                    p5_tests::reserve_encoder();
                    let mut t = NonInteractiveTrainer::new(
                        &context.evaluator,
                        facade,
                        optim,
                        layout,
                        placement,
                        model.stage(stage),
                        state,
                    )?
                    .with_transpose_keys(descriptions[stage].transpose.clone());
                    if let DiagonalOptimizer::Nag { reference, .. } = opt {
                        t.set_lr_reference(reference)?;
                    }
                    if descriptions[stage].frozen {
                        t = t.with_frozen_biases();
                    }
                    stages.push(t);
                }
                BlockTraining::NonInteractive(BlockNiTrainer::new(
                    &context.evaluator,
                    facade,
                    stages,
                )?)
            }
        };
        let s = Self {
            facade,
            plan,
            descriptions,
            classes,
            model,
            trainer,
            batches,
            completed: 0,
            samples: 0,
            poisoned: false,
            failed: None,
        };
        for stage in 0..n {
            s.check_state(stage, &s.plan.initial()[stage], 0)?;
            s.check_actual_batch(stage)?;
        }
        facade.active.set(true);
        Ok(s)
    }
    /// Successful logical block updates.
    pub fn completed_updates(&self) -> usize {
        self.completed
    }
    /// Successful logical samples; stages do not multiply this denominator.
    pub fn processed_samples(&self) -> usize {
        self.samples
    }
    /// Permanent midwork failure state.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Actual prefix evidence, when poisoned.
    pub fn failed_update(&self) -> Option<&BlockFailure> {
        self.failed.as_ref()
    }
    fn traces(&self) -> Vec<Vec<(String, usize)>> {
        match &self.trainer {
            BlockTraining::Interactive(t) => {
                t.stages().iter().map(|t| t.trace().to_vec()).collect()
            }
            BlockTraining::NonInteractive(t) => {
                t.stages().iter().map(|t| t.trace().to_vec()).collect()
            }
        }
    }
    fn check_state(&self, stage: usize, state: &StateMetadata, progress: usize) -> Result<()> {
        ensure(
            state.completed_updates() == progress,
            "block state successful index",
        )?;
        match &self.trainer {
            BlockTraining::Interactive(t) => check_interactive_stage(
                self.facade.base,
                &self.descriptions[stage],
                self.model.stage(stage),
                self.batches[stage],
                &t.stages()[stage],
                state,
            ),
            BlockTraining::NonInteractive(t) => check_ni_stage(
                self.facade.base,
                &self.descriptions[stage],
                self.model.stage(stage),
                self.batches[stage],
                &t.stages()[stage],
                state,
            ),
        }
    }
    fn check_actual_batch(&self, stage: usize) -> Result<()> {
        let c = self.facade.base;
        let before = c.counter();
        let ev = c.evaluator.fork();
        let actual = match &self.trainer {
            BlockTraining::Interactive(t) => self.model.stage(stage).batch_spec_at(
                &ev,
                &c.encoder,
                t.stages()[stage].refresh_level(),
            )?,
            BlockTraining::NonInteractive(t) => {
                t.stages()[stage].batch_spec(self.model.stage(stage))?
            }
        };
        ensure(before == c.counter(), "block probe counter isolation")?;
        ensure(
            actual == self.batches[stage],
            "block retained batch specification",
        )
    }
    fn validate_packet(&self, p: &EncryptedBlockUpdate) -> Result<(usize, usize)> {
        ensure(!self.poisoned, "poisoned block session")?;
        ensure(
            self.completed < self.plan.parts().len(),
            "exhausted block session",
        )?;
        ensure(p.index == self.completed, "block update index/replay")?;
        ensure(self.plan.same(&p.plan), "block packet plan identity")?;
        ensure(
            Arc::ptr_eq(&p.owner, &self.facade.base.owner),
            "block packet context identity",
        )?;
        let n = self.descriptions.len();
        let parts = &self.plan.parts()[self.completed];
        ensure(p.micros.len() == parts.len(), "block micro cardinality")?;
        ensure(
            p.target_widths.len() == n && p.input_width == self.descriptions[0].geometry.dims()[0],
            "block packet widths",
        )?;
        let mut total = 0usize;
        for (micro, part) in p.micros.iter().zip(parts) {
            ensure(
                micro.part == *part
                    && part.start == total
                    && part.valid > 0
                    && part.valid <= self.descriptions[0].geometry.layout().batch,
                "block ordered partition",
            )?;
            total = total
                .checked_add(part.valid)
                .ok_or_else(|| mismatch("block partition overflow"))?;
            ensure(micro.targets.len() == n, "block stage target cardinality")?;
        }
        ensure(p.total == total, "block logical denominator")?;
        let next = self
            .completed
            .checked_add(1)
            .ok_or_else(|| mismatch("block progress overflow"))?;
        let samples = self
            .samples
            .checked_add(total)
            .ok_or_else(|| mismatch("block sample overflow"))?;
        let rate = self.plan.rates()[self.completed].kernel;
        for stage in 0..n {
            let d = &self.descriptions[stage];
            self.facade.base.check_geometry(&d.geometry)?;
            ensure(
                self.classes[stage] > 0
                    && self.classes[stage] <= p.target_widths[stage]
                    && p.target_widths[stage] == *d.geometry.dims().last().unwrap(),
                "block stage target shape",
            )?;
            let t = self.plan.transition(self.completed, stage);
            let u = t.update_metadata().unwrap();
            self.check_state(stage, u.pre_state(), self.completed)?;
            self.check_actual_batch(stage)?;
            ensure(
                u.micros().len() == parts.len(),
                "block receipt micro cardinality",
            )?;
            for (mi, ((micro, part), m)) in p.micros.iter().zip(parts).zip(u.micros()).enumerate() {
                ensure(m.n_valid == part.valid, "block receipt ordered rows")?;
                check_ct(self.facade.base, &micro.targets[stage], m.y)?;
                if stage == 0 {
                    check_ct(self.facade.base, &micro.x, m.x)?;
                } else {
                    let h = self
                        .plan
                        .boundary(self.completed, stage - 1)
                        .handoff_metadata()
                        .unwrap();
                    ensure(
                        h.outputs().len() == parts.len() && h.outputs()[mi] == m.x,
                        "block incoming transition metadata",
                    )?;
                }
            }
            let c = u.coefficients();
            ensure(
                f64::from_bits(c.current_bits).is_finite()
                    && f64::from_bits(c.bias_bits).is_finite()
                    && c.reference_bits
                        .is_none_or(|v| f64::from_bits(v).is_finite()),
                "block finite read-only coefficients",
            )?;
            match &self.trainer {
                BlockTraining::Interactive(t) => t.stages()[stage].validate_lr(rate)?,
                BlockTraining::NonInteractive(t) => {
                    t.stages()[stage].validate_lr(rate)?;
                    ensure(
                        t.stages()[stage].lr_reference().to_bits()
                            == self.plan.reference().to_bits(),
                        "block fixed reference",
                    )?;
                }
            }
        }
        Ok((next, samples))
    }
    /// Validates the full packet and every stage before setters; invokes one existing wrapper.
    /// Midwork errors retain prefix counters and permanently reject retries.
    pub fn step(&mut self, p: &EncryptedBlockUpdate) -> Result<BlockCompletion> {
        let (next, samples) = self.validate_packet(p)?;
        let before = self.facade.base.counter();
        let trace_before = self.traces();
        let refresh_before = self.facade.refreshes.get();
        let stage_before = match &self.trainer {
            BlockTraining::NonInteractive(t) => t
                .stages()
                .iter()
                .map(|t| (t.weight_bootstraps(), t.activation_bootstraps()))
                .collect::<Vec<_>>(),
            _ => vec![],
        };
        let handoff_before = match &self.trainer {
            BlockTraining::Interactive(t) => t.handoffs() as u64,
            BlockTraining::NonInteractive(t) => t.handoff_bootstraps(),
        };
        self.poisoned = true;
        self.failed = Some(BlockFailure {
            update: self.completed,
            phase: "configuration",
            measured: OpCounter::default(),
            traces: vec![],
            retained_boundary_outputs: 0,
        });
        self.facade
            .begin((self.descriptions.len() - 1) * p.micros.len());
        let result = (|| {
            let rate = self.plan.rates()[self.completed].kernel;
            let next_full = self.plan.transition(self.completed, 0).next_full();
            match &mut self.trainer {
                BlockTraining::Interactive(t) => t.set_lr(rate)?,
                BlockTraining::NonInteractive(t) => {
                    t.set_lr(rate)?;
                    t.set_next_batch_full(next_full);
                }
            }
            for stage in 0..self.descriptions.len() {
                self.check_coefficients(stage, p.total)?;
            }
            let targets = p
                .micros
                .iter()
                .map(|m| m.targets.iter().collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let batches = p
                .micros
                .iter()
                .zip(&targets)
                .map(|(m, y)| (&m.x, y.as_slice(), m.part.valid))
                .collect::<Vec<_>>();
            self.failed.as_mut().unwrap().phase = "trainer";
            #[cfg(test)]
            p5_tests::reserve_attempt();
            match &mut self.trainer {
                BlockTraining::Interactive(t) => t.train_step_valid(&mut self.model, &batches)?,
                BlockTraining::NonInteractive(t) => {
                    t.train_step_valid(&mut self.model, &batches)?
                }
            }
            self.failed.as_mut().unwrap().phase = "postcheck";
            let measured = &self.facade.base.counter() - &before;
            let mut expected = OpCounter::default();
            let mut refreshes = 0;
            let mut boots = 0;
            let mut boundary_events = 0;
            for stage in 0..self.descriptions.len() {
                let u = self
                    .plan
                    .transition(self.completed, stage)
                    .update_metadata()
                    .unwrap();
                expected += u.counts_including_bootstrap();
                refreshes += u.client_refreshes();
                boots += u.bootstrap_count();
                self.check_state(stage, u.post_state(), next)?;
                self.check_actual_batch(stage)?;
                self.check_trace(stage)?;
                if let BlockTraining::NonInteractive(t) = &self.trainer {
                    let actual = &t.stages()[stage];
                    let activation = u
                        .canonical_activations()
                        .iter()
                        .map(Vec::len)
                        .sum::<usize>();
                    ensure(
                        actual.last_update_rescaled() == u.reference_branch()
                            && actual.last_update_booted() == u.state_bootstrapped()
                            && actual.activation_bootstraps() - stage_before[stage].1
                                == activation as u64
                            && actual.weight_bootstraps() - stage_before[stage].0
                                == (u.bootstrap_count() - activation) as u64,
                        "block NI stage bootstrap decisions",
                    )?;
                }
                if stage + 1 < self.descriptions.len() {
                    let h = self
                        .plan
                        .boundary(self.completed, stage)
                        .handoff_metadata()
                        .unwrap();
                    expected += h.counts_including_bootstrap();
                    refreshes += h.client_refreshes();
                    boots += h.bootstrap_count();
                    boundary_events += match &self.trainer {
                        BlockTraining::Interactive(_) => h.client_refreshes(),
                        BlockTraining::NonInteractive(_) => h.bootstrap_count(),
                    };
                }
            }
            self.check_handoffs(p.micros.len())?;
            let handoff_after = match &self.trainer {
                BlockTraining::Interactive(t) => t.handoffs() as u64,
                BlockTraining::NonInteractive(t) => t.handoff_bootstraps(),
            };
            ensure(
                handoff_after - handoff_before == boundary_events as u64,
                "block boundary refresh/bootstrap decisions",
            )?;
            ensure(
                measured == expected && measured.bootstraps == boots as u64,
                "block seven inclusive stage+handoff counters",
            )?;
            ensure(
                self.facade.refreshes.get() - refresh_before == refreshes,
                "block inclusive client refreshes",
            )?;
            Ok(BlockCompletion {
                completed_updates: next,
                processed_samples: samples,
                measured,
                client_refreshes: refreshes,
                traces: self.traces(),
            })
        })();
        match result {
            Ok(done) => {
                self.completed = next;
                self.samples = samples;
                self.poisoned = false;
                self.failed = None;
                Ok(done)
            }
            Err(e) => {
                let traces = self
                    .traces()
                    .into_iter()
                    .zip(trace_before)
                    .map(|(a, b)| if a == b { vec![] } else { a })
                    .collect();
                let failure = self.failed.as_mut().unwrap();
                failure.measured = &self.facade.base.counter() - &before;
                failure.traces = traces;
                failure.retained_boundary_outputs = self.facade.boundary.borrow().len();
                Err(e)
            }
        }
    }
    fn check_coefficients(&self, stage: usize, total: usize) -> Result<()> {
        let u = self
            .plan
            .transition(self.completed, stage)
            .update_metadata()
            .unwrap();
        let c = u.coefficients();
        let n = total as f64;
        let rate = self.plan.rates()[self.completed].kernel;
        match &self.trainer {
            BlockTraining::Interactive(t) => {
                for layer in 0..self.model.stage(stage).layers.len() {
                    let v = t.stages()[stage].optimizer_view(layer)?;
                    ensure(
                        v.lr().to_bits() == rate.to_bits()
                            && (v.weight_factor() / n).to_bits() == c.current_bits
                            && v.reference_factor().map(|x| (x / n).to_bits()) == c.reference_bits
                            && (if self.descriptions[stage].frozen {
                                0.
                            } else {
                                v.bias_factor() / n
                            })
                            .to_bits()
                                == c.bias_bits
                            && v.momentum().map(f64::to_bits) == c.velocity_mu_bits
                            && v.reference_factor().is_some() == u.reference_branch()
                            && (v.momentum().is_some() && v.velocity().is_none())
                                == u.first_nag_update(),
                        "block interactive configured coefficients/branches",
                    )?;
                }
            }
            BlockTraining::NonInteractive(t) => {
                let t = &t.stages()[stage];
                let o = t.optimizer();
                let reference = (o.has_velocity() && rate / t.lr_reference() != 1.)
                    .then(|| o.with_lr(t.lr_reference()).grad_scale() / n);
                let mu = match o {
                    NiOptimizer::Nag { momentum, .. } => Some(momentum),
                    _ => None,
                };
                ensure(
                    o.lr().to_bits() == rate.to_bits()
                        && (o.grad_scale() / n).to_bits() == c.current_bits
                        && reference.map(f64::to_bits) == c.reference_bits
                        && (if self.descriptions[stage].frozen {
                            0.
                        } else {
                            o.bias_grad_scale() / n
                        })
                        .to_bits()
                            == c.bias_bits
                        && mu.map(f64::to_bits) == c.velocity_mu_bits
                        && reference.is_some() == u.reference_branch()
                        && (o.has_velocity() && t.state().velocity.iter().all(Option::is_none))
                            == u.first_nag_update(),
                    "block NI configured coefficients/branches",
                )?;
            }
        }
        Ok(())
    }
    fn check_trace(&self, stage: usize) -> Result<()> {
        let u = self
            .plan
            .transition(self.completed, stage)
            .update_metadata()
            .unwrap();
        let traces = self.traces();
        let trace = &traces[stage];
        let interactive = matches!(&self.trainer, BlockTraining::Interactive(_));
        let count = if interactive { u.micros().len() } else { 1 };
        ensure(
            u.retained_hidden().len() == u.micros().len(),
            "block hidden micro cardinality",
        )?;
        for micro in 0..count {
            let labels = &u.retained_hidden()[micro];
            ensure(
                labels.len() + 1 == self.model.stage(stage).layers.len(),
                "block hidden layer cardinality",
            )?;
            for (layer, m) in labels.iter().enumerate() {
                let name = if interactive && count > 1 {
                    format!("micro{micro}/a{}", layer + 1)
                } else {
                    format!("a{}", layer + 1)
                };
                let found = trace.iter().filter(|(n, _)| n == &name).collect::<Vec<_>>();
                ensure(
                    found.len() == 1 && found[0].1 == m.level(),
                    "block hidden trace level",
                )?;
            }
        }
        if let BlockTraining::NonInteractive(t) = &self.trainer {
            let t = &t.stages()[stage];
            let actual = t.last_activation_bootstraps();
            let canonical = u.canonical_activations();
            if t.placement() == BootPlacement::AfterActivation {
                ensure(
                    actual.len() == canonical.len(),
                    "block AA micro cardinality",
                )?;
                for (a, b) in actual.iter().zip(canonical) {
                    ensure(a.len() == b.len(), "block AA layer cardinality")?;
                    for (ct, &m) in a.iter().zip(b) {
                        check_ct(self.facade.base, ct, m)?;
                    }
                }
                for (layer, m) in canonical[0].iter().enumerate() {
                    let name = format!("a{}_boot", layer + 1);
                    let found = trace.iter().filter(|(n, _)| n == &name).collect::<Vec<_>>();
                    ensure(
                        found.len() == 1 && found[0].1 == m.level(),
                        "block AA trace level",
                    )?;
                }
            } else {
                ensure(
                    actual.is_empty() && canonical.iter().all(Vec::is_empty),
                    "block non-AA activation absence",
                )?;
            }
        }
        Ok(())
    }
    fn check_handoffs(&self, micros: usize) -> Result<()> {
        let boundaries = self.descriptions.len() - 1;
        match &self.trainer {
            BlockTraining::Interactive(_) => {
                let output = self.facade.boundary.borrow();
                ensure(
                    output.len() == boundaries * micros,
                    "block I recorded handoff cardinality",
                )?;
                for (stage, part) in output.chunks_exact(micros).enumerate() {
                    let expected = self
                        .plan
                        .boundary(self.completed, stage)
                        .handoff_metadata()
                        .unwrap();
                    ensure(
                        part.len() == expected.outputs().len(),
                        "block I typed handoff cardinality",
                    )?;
                    for (ct, &m) in part.iter().zip(expected.outputs()) {
                        check_ct(self.facade.base, ct, m)?;
                    }
                }
            }
            BlockTraining::NonInteractive(t) => {
                let output = t.last_handoff_bootstraps();
                ensure(
                    output.len() == boundaries,
                    "block NI handoff boundary cardinality",
                )?;
                for (stage, part) in output.iter().enumerate() {
                    let expected = self
                        .plan
                        .boundary(self.completed, stage)
                        .handoff_metadata()
                        .unwrap();
                    let dest = self
                        .plan
                        .transition(self.completed, stage + 1)
                        .update_metadata()
                        .unwrap();
                    ensure(
                        part.len() == micros && part.len() == expected.outputs().len(),
                        "block NI typed handoff cardinality",
                    )?;
                    for (micro, (ct, &m)) in part.iter().zip(expected.outputs()).enumerate() {
                        check_ct(self.facade.base, ct, m)?;
                        ensure(
                            m == dest.micros()[micro].x,
                            "block incoming handoff level16 transition",
                        )?;
                        if t.stages()[stage].placement() == BootPlacement::AfterActivation {
                            ensure(
                                expected.bootstrap_count() == 0
                                    && *expected.counts_including_bootstrap()
                                        == OpCounter::default()
                                    && ciphertext_exact(
                                        ct,
                                        &t.stages()[stage].last_activation_bootstraps()[micro][0],
                                    ),
                                "block exact AA reuse/zero conversion",
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Copies all stage models and available old-hidden handoffs to a client endpoint.
    pub fn snapshot(&self) -> EncryptedBlockSnapshot {
        let stages = (0..self.descriptions.len())
            .map(|stage| EncryptedModelSnapshot {
                owner: self.facade.base.owner.clone(),
                layers: self
                    .model
                    .stage(stage)
                    .layers
                    .iter()
                    .map(|(d, _)| d.clone())
                    .collect(),
                layout: self.descriptions[stage].geometry.layout(),
            })
            .collect();
        let update = self.completed.saturating_sub(1);
        let micros = self.plan.parts()[update].len();
        let handoffs = match &self.trainer {
            BlockTraining::Interactive(_) => self
                .facade
                .boundary
                .borrow()
                .chunks(micros)
                .map(|p| p.to_vec())
                .collect(),
            BlockTraining::NonInteractive(t) => t.last_handoff_bootstraps().to_vec(),
        };
        EncryptedBlockSnapshot {
            stages,
            handoffs,
            valid: self.plan.parts()[update].iter().map(|p| p.valid).collect(),
            widths: self
                .descriptions
                .iter()
                .take(self.descriptions.len() - 1)
                .map(|d| d.geometry.dims()[1])
                .collect(),
        }
    }
}
/// Exact ciphertext comparison includes lengths, all limbs, form and basis identity.
pub(crate) fn ciphertext_exact(a: &Ciphertext, b: &Ciphertext) -> bool {
    a.len() == b.len()
        && a.level() == b.level()
        && a.scale().to_bits() == b.scale().to_bits()
        && a.polys().len() == b.polys().len()
        && a.polys().iter().zip(b.polys()).all(|(a, b)| {
            Arc::ptr_eq(a.basis(), b.basis())
                && a.level() == b.level()
                && a.is_ntt_form() == b.is_ntt_form()
                && a.limbs() == b.limbs()
        })
}
#[cfg(test)]
pub(crate) mod p5_tests;

impl Drop for CpuBlockSession<'_> {
    fn drop(&mut self) {
        self.facade.boundary.borrow_mut().clear();
        self.facade.active.set(false);
    }
}
/// Observed retained payload for one explicitly named live role.
#[derive(Debug)]
pub struct BlockRetainedRole {
    /// Role and endpoint liveness; equal ciphertext contents are still separate payloads.
    pub role: &'static str,
    /// Number of actually retained ciphertext allocations in this role.
    pub ciphertexts: usize,
    /// Checked bytes of actual coefficient words; excludes object and allocator overhead.
    pub coefficient_bytes: usize,
}
/// Checked payload inventory at a quiescent endpoint, never RSS or resource admission.
#[derive(Debug)]
pub struct BlockRetentionReceipt {
    /// Actual retained ciphertext roles, counted separately for every clone owner.
    pub retained: Vec<BlockRetainedRole>,
    /// Checked sum of the observed roles only.
    pub retained_coefficient_bytes: usize,
    /// Actual retained training key bundle coefficient payload, separately owned.
    pub training_key_bytes: usize,
    /// Actual retained separate bootstrap key bundle payload, even when keys overlap.
    pub bootstrap_key_bytes: usize,
    /// Analytical role/count bounds for work inside the NN wrapper; not observations.
    pub analytical_work: Vec<(&'static str, usize)>,
    /// Explicit unobserved memory, with no fabricated total peak or secure-memory gate.
    pub unobserved: &'static str,
}
fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| mismatch("block retention overflow"))
}
fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| mismatch("block retention overflow"))
}
fn ciphertext_bytes(c: &Ciphertext) -> Result<usize> {
    c.polys().iter().try_fold(0, |sum, p| {
        p.limbs().iter().try_fold(sum, |sum, limb| {
            checked_add(sum, checked_mul(limb.len(), std::mem::size_of::<u64>())?)
        })
    })
}
fn role<'a>(
    name: &'static str,
    cts: impl IntoIterator<Item = &'a Ciphertext>,
) -> Result<BlockRetainedRole> {
    let mut count = 0;
    let mut bytes = 0;
    for c in cts {
        count = checked_add(count, 1)?;
        bytes = checked_add(bytes, ciphertext_bytes(c)?)?;
    }
    Ok(BlockRetainedRole {
        role: name,
        ciphertexts: count,
        coefficient_bytes: bytes,
    })
}
fn key_bytes(key: &ckks_train_ckks::KskKey) -> Result<usize> {
    let mut words = 0;
    for (b, a) in key.parts() {
        for part in [b.q(), b.p(), a.q(), a.p()] {
            for limb in part.limbs() {
                words = checked_add(words, limb.len())?;
            }
        }
    }
    checked_mul(words, 8)
}
impl CpuBlockSession<'_> {
    /// Accounts actual retained payloads at an endpoint, optionally including the caller's
    /// current packet and exported snapshot. Other diagnostic copies remain caller-owned.
    pub fn retention_receipt(
        &self,
        packet: Option<&EncryptedBlockUpdate>,
        snapshot: Option<&EncryptedBlockSnapshot>,
    ) -> Result<BlockRetentionReceipt> {
        let mut retained = vec![role(
            "session expanded W/b; live for session",
            (0..self.model.num_stages()).flat_map(|s| {
                self.model.stage(s).layers.iter().flat_map(|(d, _)| {
                    d.w.diagonals()
                        .entries()
                        .iter()
                        .chain(std::iter::once(&d.b))
                })
            }),
        )?];
        match &self.trainer {
            BlockTraining::Interactive(t) => {
                let mut velocity = BlockRetainedRole {
                    role: "I velocity; live for session",
                    ciphertexts: 0,
                    coefficient_bytes: 0,
                };
                for (s, stage) in t.stages().iter().enumerate() {
                    for layer in 0..self.model.stage(s).layers.len() {
                        let view = stage.optimizer_view(layer)?;
                        if let Some(v) = view.velocity() {
                            let part = role("I velocity layer", v.entries())?;
                            velocity.ciphertexts =
                                checked_add(velocity.ciphertexts, part.ciphertexts)?;
                            velocity.coefficient_bytes =
                                checked_add(velocity.coefficient_bytes, part.coefficient_bytes)?;
                        }
                    }
                }
                retained.push(velocity);
                retained.push(role(
                    "I boundary recorder; current attempt, cleared at next accepted attempt/drop",
                    self.facade.boundary.borrow().iter(),
                )?);
            }
            BlockTraining::NonInteractive(t) => {
                retained.push(role(
                    "NI compact W; live for session",
                    t.stages().iter().flat_map(|s| &s.state().weights),
                )?);
                retained.push(role(
                    "NI compact velocity; live for session",
                    t.stages()
                        .iter()
                        .flat_map(|s| s.state().velocity.iter().flatten()),
                )?);
                retained.push(role(
                    "NI boundary output clones; last success, cleared by next wrapper attempt",
                    t.last_handoff_bootstraps().iter().flatten(),
                )?);
                retained.push(role(
                    "AA activation outputs; per-stage last successful update",
                    t.stages()
                        .iter()
                        .flat_map(|s| s.last_activation_bootstraps().iter().flatten()),
                )?);
            }
        }
        if let Some(p) = packet {
            ensure(
                self.plan.same(&p.plan) && Arc::ptr_eq(&p.owner, &self.facade.base.owner),
                "block retention packet association",
            )?;
            retained.push(role(
                "caller packet x0 and stage targets; until endpoint/drop",
                p.micros
                    .iter()
                    .flat_map(|m| std::iter::once(&m.x).chain(&m.targets)),
            )?);
        }
        if let Some(s) = snapshot {
            ensure(
                s.stages.len() == self.descriptions.len()
                    && s.stages
                        .iter()
                        .all(|s| Arc::ptr_eq(&s.owner, &self.facade.base.owner)),
                "block retention snapshot association",
            )?;
            retained.push(role(
                "client endpoint model snapshot clones; caller owned",
                s.stages.iter().flat_map(|s| {
                    s.layers.iter().flat_map(|d| {
                        d.w.diagonals()
                            .entries()
                            .iter()
                            .chain(std::iter::once(&d.b))
                    })
                }),
            )?);
            retained.push(role(
                "client endpoint handoff snapshot clones; caller owned",
                s.handoffs.iter().flatten(),
            )?);
        }
        let retained_coefficient_bytes = retained
            .iter()
            .try_fold(0, |sum, r| checked_add(sum, r.coefficient_bytes))?;
        let ev = &self.facade.base.evaluator;
        let mut training_key_bytes = 0;
        if let Some(k) = ev.relin_key() {
            training_key_bytes = checked_add(training_key_bytes, key_bytes(k.key())?)?;
        }
        if let Some(keys) = ev.galois_keys() {
            for k in keys.iter() {
                training_key_bytes = checked_add(training_key_bytes, key_bytes(k.key())?)?;
            }
        }
        let mut bootstrap_key_bytes = 0;
        if let Some(b) = &self.facade.base.bootstrap {
            for k in b.real.keys().galois().iter() {
                bootstrap_key_bytes = checked_add(bootstrap_key_bytes, key_bytes(k.key())?)?;
            }
            if let Some(sse) = b.real.keys().sse() {
                bootstrap_key_bytes =
                    checked_add(bootstrap_key_bytes, key_bytes(sse.to_sparse())?)?;
                bootstrap_key_bytes = checked_add(bootstrap_key_bytes, key_bytes(sse.to_dense())?)?;
            }
        }
        let micros = self.plan.parts().iter().map(Vec::len).max().unwrap_or(0);
        let expanded = self.descriptions.iter().try_fold(0, |sum, d| {
            d.geometry.dims().windows(2).try_fold(sum, |sum, w| {
                checked_add(sum, checked_add(w[0].max(w[1]).next_power_of_two(), 1)?)
            })
        })?;
        let layers = self
            .descriptions
            .iter()
            .try_fold(0, |sum, d| checked_add(sum, d.activations.len()))?;
        let analytical_work = vec![
            (
                "wrapper current inputs clones plus next handoff clones; per active boundary",
                checked_mul(micros, 2)?,
            ),
            (
                "old hidden and forward caches; all stage layers upper role bound",
                checked_mul(micros, layers)?,
            ),
            (
                "active gradients plus reference gradients/transpose work; coefficient payload not observed",
                checked_mul(expanded, 3)?,
            ),
            (
                "AA wrapper retains separate source activation and handoff clones",
                checked_mul(micros, self.descriptions.len() - 1)?,
            ),
        ];
        Ok(BlockRetentionReceipt {
            retained,
            retained_coefficient_bytes,
            training_key_bytes,
            bootstrap_key_bytes,
            analytical_work,
            unobserved: "NN temporary lifetime overlap, bootstrap ciphertext/plaintext workspace and transforms, evaluator/encoder caches, parameter tables, client public/secret/encryption RNG storage, caller diagnostic comparison copies, object/Vec capacity and allocator overhead are not measured. This receipt is not RSS, security, resource admission or a secure-memory gate.",
        })
    }
}

#[cfg(test)]
impl BlockRetentionReceipt {
    fn include_diagnostic_copy<'a>(
        &mut self,
        name: &'static str,
        cts: impl IntoIterator<Item = &'a Ciphertext>,
    ) -> Result<()> {
        let observed = role(name, cts)?;
        self.retained_coefficient_bytes =
            checked_add(self.retained_coefficient_bytes, observed.coefficient_bytes)?;
        self.retained.push(observed);
        Ok(())
    }
}
