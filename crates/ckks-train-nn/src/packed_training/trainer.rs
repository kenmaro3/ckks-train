//! Secret-free single-stage NI orchestration. Structural success is conditional, not Ready.
use super::dag::checked_counts;
use super::*;
use ckks_train_ckks::metadata::target_plain_scale_at;
use ckks_train_ckks::{DiagonalEntry, Encoder, HeOps, OpCounter};
use ckks_train_core::error::{HekitError, Result};
/// Public NI placement; interactive refresh is deliberately a separate API.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedNiPlacement {
    /// State refresh after every update.
    Au,
    /// State refresh when the actual next carried DAG requires it.
    Wn,
    /// One fixed hidden bootstrap per valid record plus carried state admission.
    Aa,
}
/// Actual next metadata or an explicitly conservative checkpoint policy.
#[derive(Clone, Debug)]
pub enum PackedNextStep {
    /// Complete next rate, schedule index and valid record histories.
    Known(PackedBatchMeta, PackedUpdateMeta),
    /// Future remains unadmitted; refresh conservatively.
    Unknown,
    /// Terminal restartable checkpoint; refresh conservatively.
    Terminal,
}
/// Complete immutable NI plan. Derived histories do not carry executable generation tokens.
#[derive(Clone, Debug)]
pub struct PackedNiPlan {
    refresh: bool,
    counts: OpCounter,
    pub(super) source: PackedStageMeta,
    pub(super) batch: PackedBatchMeta,
    pub(super) update: PackedUpdateMeta,
    pub(super) dag: PackedDagPlan,
    pub(super) next: PackedNextStep,
    pub(super) placement: PackedNiPlacement,
    pub(super) contract: PackedBootstrapContract,
    committed: PackedStageMeta,
}
impl PackedNiPlan {
    /// Plans current graph, reserve, known next feasibility and complete inclusive counts.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enc: &Encoder,
        state: &PackedStageMeta,
        batch: &PackedBatchMeta,
        update: PackedUpdateMeta,
        next: PackedNextStep,
        placement: PackedNiPlacement,
        contract: &PackedBootstrapContract,
    ) -> Result<Self> {
        contract.validate_params(&state.params)?;
        let hidden = (placement == PackedNiPlacement::Aa).then_some(contract);
        let dag = plan_stage_dag(enc, state, batch, update, hidden)?;
        let projection = projection_preflight(enc, dag.candidate())?;
        let mut restored = dag.candidate().clone();
        restored.weights = [contract.output(); 2];
        restored.velocities = restored.velocities.map(|u| u.map(|_| contract.output()));
        restored.biases = restored.biases.map(|_| [contract.output(); 2]);
        let mut refresh =
            placement == PackedNiPlacement::Au || !matches!(next, PackedNextStep::Known(..));
        if let PackedNextStep::Known(ref next_batch, next_update) = next {
            // Validate schedule/rate independently before trying a depth repair.
            if next_batch.update_index() != dag.candidate().indices[0] {
                return invalid("packed NI future schedule mismatch");
            }
            state
                .config
                .policy()?
                .coefficients(next_update.rate(), next_batch.records().len())?;
            let attempt = |s: &PackedStageMeta| -> Result<()> {
                let p = plan_stage_dag(enc, s, next_batch, next_update, hidden)?;
                projection_preflight(enc, p.candidate())?;
                Ok(())
            };
            if !refresh {
                match attempt(dag.candidate()) {
                    Ok(()) => {}
                    Err(HekitError::NoLevelsLeft) => refresh = true,
                    Err(e) => return Err(e),
                }
            }
            if refresh {
                attempt(&restored)?;
            }
        }
        let mut counts = dag.step().predicted_counts.clone();
        if refresh {
            counts = checked_counts(&counts, &projection)?;
            for _ in 0..projection.ct_pt_mults {
                counts = checked_counts(&counts, contract.cost_per_boot())?;
            }
        }
        let committed = if refresh {
            restored
        } else {
            dag.candidate().clone()
        };
        Ok(Self {
            refresh,
            counts,
            source: state.clone(),
            batch: batch.clone(),
            update,
            dag,
            next,
            placement,
            contract: contract.clone(),
            committed,
        })
    }
    /// Conditional complete candidate histories after the selected refresh policy.
    pub fn committed(&self) -> &PackedStageMeta {
        &self.committed
    }
    /// Current exact DAG before optional state refresh.
    pub fn dag(&self) -> &PackedDagPlan {
        &self.dag
    }
    /// Preserved explicit next-request policy.
    pub fn next(&self) -> &PackedNextStep {
        &self.next
    }

    /// Structural graph success is conditional on successful bootstrap. Independent gaps.
    pub fn unvalidated_requirements(&self) -> &'static [&'static str] {
        &[
            "BootstrapEncodingQOverTwoFeasibility",
            "EvalModInternalScaleUnvalidated",
            "RangeContractUnverified",
            "PrecisionUnmeasured",
            "ActualSecretKeyProvenanceUnverified",
            "SecurityUnadmitted",
            "MemoryUnadmitted",
            "BackendUnadmitted",
            "FullProfileScheduleMissingCode",
        ]
    }
    /// Exact graph metadata/Encoder checks passed; never a production Ready receipt.
    pub fn graph_level_scale_admitted(&self) -> bool {
        true
    }
    /// Whether all candidate states must be projected and bootstrapped.
    pub fn refresh_state(&self) -> bool {
        self.refresh
    }
    /// Complete inclusive conditional successful-execution count.
    pub fn predicted_counts(&self) -> &OpCounter {
        &self.counts
    }
}

fn projection_preflight(enc: &Encoder, state: &PackedStageMeta) -> Result<OpCounter> {
    let mut counts = OpCounter::default();
    for k in 0..2 {
        for h in std::iter::once(state.weights[k]).chain(state.velocities[k]) {
            if h.level() == 0 {
                return Err(HekitError::NoLevelsLeft);
            }
            let ps =
                target_plain_scale_at(&state.params, h.level(), h.scale(), state.params.scale())?;
            enc.encode(&state.layers[k].matrix_mask(), ps, h.level())?;
            counts = checked_counts(
                &counts,
                &OpCounter {
                    ct_pt_mults: 1,
                    rescales: 1,
                    ..Default::default()
                },
            )?;
        }
        if let Some(b) = state.biases {
            if b[k].level() == 0 {
                return Err(HekitError::NoLevelsLeft);
            }
            let ps = target_plain_scale_at(
                &state.params,
                b[k].level(),
                b[k].scale(),
                state.params.scale(),
            )?;
            enc.encode(&state.layers[k].output().mask(), ps, b[k].level())?;
            counts = checked_counts(
                &counts,
                &OpCounter {
                    ct_pt_mults: 1,
                    rescales: 1,
                    ..Default::default()
                },
            )?;
        }
    }
    Ok(counts)
}

/// Actual retained encrypted boundaries of one successful bootstrap, for external diagnostics.
#[derive(Debug)]
pub struct PackedBootstrapEvent<C> {
    role: PackedRefreshRole,
    source: C,
    projected: Option<C>,
    input: C,
    output: C,
    delta: OpCounter,
}
impl<C> PackedBootstrapEvent<C> {
    /// Original state role or original valid-record identity.
    pub fn role(&self) -> PackedRefreshRole {
        self.role
    }
    /// Before support projection (or canonical hidden).
    pub fn source(&self) -> &C {
        &self.source
    }
    /// After the single state support projection, before lowering; absent for hidden.
    pub fn projected(&self) -> Option<&C> {
        self.projected.as_ref()
    }
    /// Actual level-zero bootstrap input.
    pub fn input(&self) -> &C {
        &self.input
    }
    /// Actual returned output.
    pub fn output(&self) -> &C {
        &self.output
    }
    /// Nested inclusive bootstrap delta, never re-added to outer counts.
    pub fn counter_delta(&self) -> &OpCounter {
        &self.delta
    }
}
/// Last failure boundary. Caller-owned old stage is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedNiPhase {
    /// No current HE started.
    Preflight,
    /// Current stage/AA seam work.
    Update,
    /// State projection/bootstrap or reconstruction.
    Refresh,
    /// Final observed receipt comparison.
    Verification,
}
/// No successful checkpoint; completed work and bootstrap effects may remain.
#[derive(Debug)]
pub struct PackedNiFailure<C> {
    phase: PackedNiPhase,
    role: Option<PackedRefreshRole>,
    events: Vec<PackedBootstrapEvent<C>>,
    counts: Option<Box<OpCounter>>,
    updated: Option<Box<PackedStage<C>>>,
    error: HekitError,
}
impl<C> PackedNiFailure<C> {
    /// Last boundary reached.
    pub fn phase(&self) -> PackedNiPhase {
        self.phase
    }
    /// Role in progress, if known.
    pub fn role(&self) -> Option<PackedRefreshRole> {
        self.role
    }
    /// Successfully validated earlier events only.
    pub fn events(&self) -> &[PackedBootstrapEvent<C>] {
        &self.events
    }
    /// Observed inclusive work; None if a counter reset prevents subtraction.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_deref()
    }
    /// Complete encrypted candidate if update finished before the failure.
    pub fn updated_stage(&self) -> Option<&PackedStage<C>> {
        self.updated.as_deref()
    }
    /// Specific original backend/metadata failure.
    pub fn error(&self) -> &HekitError {
        &self.error
    }
}
/// Entire state candidate, transferred in one operation.
#[derive(Debug)]
pub struct PackedNiCandidate<C> {
    stage: PackedStage<C>,
}
impl<C> PackedNiCandidate<C> {
    /// Immutable complete candidate.
    pub fn stage(&self) -> &PackedStage<C> {
        &self.stage
    }
    /// Atomic ownership transfer of every stage part.
    pub fn into_stage(self) -> PackedStage<C> {
        self.stage
    }
}
/// Independent observed boundaries and the complete pre-refresh candidate state.
#[derive(Debug)]
pub struct PackedNiReceipt<C> {
    step: PackedStepReceipt<C>,
    updated: PackedStage<C>,
    events: Vec<PackedBootstrapEvent<C>>,
    counts: OpCounter,
    plan: PackedNiPlan,
}
impl<C> PackedNiReceipt<C> {
    /// Old-forward and actual current-DAG trace.
    pub fn step(&self) -> &PackedStepReceipt<C> {
        &self.step
    }
    /// Encrypted candidate before state refresh.
    pub fn updated_stage(&self) -> &PackedStage<C> {
        &self.updated
    }
    /// Ordered actual successful bootstrap events.
    pub fn events(&self) -> &[PackedBootstrapEvent<C>] {
        &self.events
    }
    /// Single inclusive evaluator interval.
    pub fn actual_counts(&self) -> &OpCounter {
        &self.counts
    }
    /// Immutable actual metadata plan.
    pub fn plan(&self) -> &PackedNiPlan {
        &self.plan
    }
}
/// Complete stage result or failure with no old-state mutation.
pub type PackedNiResult<C> =
    std::result::Result<(PackedNiCandidate<C>, PackedNiReceipt<C>), PackedNiFailure<C>>;
/// Public evaluation material only. No client refresh, diagnostic secret or plaintext model.
pub struct PackedNiTrainer<'a, E: ckks_train_ckks::HeOps, B: PlannedBootstrap<E>> {
    ev: &'a E,
    enc: &'a Encoder,
    bootstrap: &'a B,
    placement: PackedNiPlacement,
    contract: PackedBootstrapContract,
}
impl<'a, E: ckks_train_ckks::HeOps, B: PlannedBootstrap<E>> PackedNiTrainer<'a, E, B> {
    /// Checks capability identity and runtime evaluator binding before any work.
    pub fn new(
        ev: &'a E,
        enc: &'a Encoder,
        bootstrap: &'a B,
        placement: PackedNiPlacement,
    ) -> Result<Self> {
        if !std::sync::Arc::ptr_eq(ev.params(), enc.params()) {
            return invalid("packed NI Encoder context mismatch");
        }
        let contract = bootstrap.contract()?;
        bootstrap.validate_binding(ev, &contract)?;
        Ok(Self {
            ev,
            enc,
            bootstrap,
            placement,
            contract,
        })
    }
    /// Validates and binds a current-generation executable plan.
    pub fn plan_update(
        &self,
        old: &PackedStage<E::Ct>,
        records: &[PackedRecord<E::Ct>],
        rate: f64,
        relin: crate::layout_reboot::GradRelin,
        next: PackedNextStep,
    ) -> Result<PackedNiPlan> {
        self.bootstrap.validate_binding(self.ev, &self.contract)?;
        let state = old.metadata(self.ev)?;
        let batch = old.batch_metadata(self.ev, records)?;
        PackedNiPlan::new(
            self.enc,
            &state,
            &batch,
            PackedUpdateMeta::new(rate, relin, self.ev.config().lazy_relin)?,
            next,
            self.placement,
            &self.contract,
        )
    }
    /// Prepare exactly one encrypted update and all selected state refreshes.
    pub fn prepare_update(
        &self,
        old: &PackedStage<E::Ct>,
        records: &[PackedRecord<E::Ct>],
        rate: f64,
        relin: crate::layout_reboot::GradRelin,
        next: PackedNextStep,
    ) -> PackedNiResult<E::Ct> {
        let before = self.ev.counter();
        let plan = self
            .plan_update(old, records, rate, relin, next)
            .map_err(|error| PackedNiFailure {
                phase: PackedNiPhase::Preflight,
                role: None,
                events: vec![],
                counts: super::refresh::counter_delta(&self.ev.counter(), &before).map(Box::new),
                updated: None,
                error,
            })?;
        self.prepare_planned(old, records, &plan)
    }
    /// Rebinds current generations, records and capability before current HE.
    pub fn prepare_planned(
        &self,
        old: &PackedStage<E::Ct>,
        records: &[PackedRecord<E::Ct>],
        plan: &PackedNiPlan,
    ) -> PackedNiResult<E::Ct> {
        let before = self.ev.counter();
        let mut events = Vec::new();
        let mut role = None;
        let mut phase = PackedNiPhase::Preflight;
        let mut retained = None;
        let result = (|| -> Result<PreparedExecution<E::Ct>> {
            self.bootstrap.validate_binding(self.ev, &plan.contract)?;
            if plan.placement != self.placement
                || plan.update.lazy != self.ev.config().lazy_relin
                || plan.contract.metadata().descriptor_identity_bytes()
                    != self.contract.metadata().descriptor_identity_bytes()
            {
                return invalid("packed NI plan capability/config mismatch");
            }
            let actual = old.metadata(self.ev)?;
            match (&actual.generations, &plan.source.generations) {
                (Some(a), Some(b))
                    if a.iter().zip(b).all(|(x, y)| std::sync::Arc::ptr_eq(x, y)) => {}
                _ => {
                    return invalid(
                        "packed NI plan is hypothetical, stale or from another generation",
                    );
                }
            }
            if old.batch_metadata(self.ev, records)? != plan.batch {
                return invalid("packed NI batch metadata changed");
            }
            if !self.ev.has_relin_key()
                || actual.layers[0]
                    .layout()
                    .galois_steps()
                    .iter()
                    .any(|&s| !self.ev.has_galois_key(s))
            {
                return invalid("packed NI evaluator missing stage keys");
            }
            phase = PackedNiPhase::Update;
            let hidden = (plan.placement == PackedNiPlacement::Aa)
                .then_some((self.bootstrap as &dyn PlannedBootstrap<E>, &self.contract));
            let (updated, step) = old.prepare_with_plan(
                self.ev,
                self.enc,
                records,
                plan.update.rate,
                plan.update.relin,
                plan.dag.step().clone(),
                hidden,
                &mut events,
                &mut role,
            )?;
            retained = Some(updated.into_stage());
            let updated = retained.as_ref().unwrap();
            phase = PackedNiPhase::Refresh;
            let stage = if plan.refresh {
                let batch = PackedRefreshBatch::new(
                    self.ev,
                    updated,
                    PackedRefreshContract::new(self.ev, self.contract.output().level())?,
                )?;
                let mut responses = Vec::new();
                for request in batch.requests() {
                    role = Some(request.role());
                    let input = project_state(self.ev, self.enc, request)?;
                    let output =
                        bootstrap_observed(self.ev, self.bootstrap, &self.contract, &input)?;
                    let response = PackedRefreshResponse::new(self.ev, request, output.0.clone())?;
                    events.push(PackedBootstrapEvent {
                        role: request.role(),
                        source: request.ciphertext().clone(),
                        projected: input.projected().cloned(),
                        input: input.ciphertext().clone(),
                        output: output.0,
                        delta: output.1,
                    });
                    responses.push(response);
                }
                batch.complete(self.ev, responses)?.into_stage()
            } else {
                // A read-only candidate snapshot and committed ownership both remain available.
                // Reconstruct only ciphertext ownership; no HE, refresh or optimizer update.
                updated.with_refreshed_parts(
                    self.ev,
                    std::array::from_fn(|k| updated.layers()[k].weights().clone()),
                    std::array::from_fn(|k| updated.layers()[k].velocity().cloned()),
                    updated.biases().cloned(),
                )?
            };
            phase = PackedNiPhase::Verification;
            role = None;
            let actual = stage.metadata(self.ev)?;
            if actual.weights != plan.committed.weights
                || actual.velocities != plan.committed.velocities
                || actual.biases != plan.committed.biases
                || actual.indices != plan.committed.indices
            {
                return invalid("packed NI committed output metadata differs from plan");
            }
            let counts =
                super::refresh::counter_delta(&self.ev.counter(), &before).ok_or_else(|| {
                    HekitError::InvalidParams("packed NI evaluator counter reset".into())
                })?;
            if counts != plan.counts {
                return invalid("packed NI inclusive observed counts differ from plan");
            }
            Ok((PackedNiCandidate { stage }, step, counts))
        })();
        match result {
            Ok((candidate, step, counts)) => Ok((
                candidate,
                PackedNiReceipt {
                    step,
                    updated: retained.unwrap(),
                    events,
                    counts,
                    plan: plan.clone(),
                },
            )),
            Err(error) => Err(PackedNiFailure {
                phase,
                role,
                events,
                counts: super::refresh::counter_delta(&self.ev.counter(), &before).map(Box::new),
                updated: retained.map(Box::new),
                error,
            }),
        }
    }
}

fn bootstrap_observed<E: HeOps>(
    ev: &E,
    boot: &dyn PlannedBootstrap<E>,
    contract: &PackedBootstrapContract,
    input: &PackedBootInput<E::Ct>,
) -> Result<(E::Ct, OpCounter)> {
    let before = ev.counter();
    let output = boot.bootstrap(ev, input.ciphertext())?;
    super::types::check_ct(ev, &output, ev.params().slots())?;
    if output.level() != contract.output().level() {
        return invalid(&format!(
            "packed bootstrap output level {} expected {}",
            output.level(),
            contract.output().level()
        ));
    }
    if output.scale().to_bits() != contract.output().scale_bits() {
        return invalid(&format!(
            "packed bootstrap output scale bits {} expected {}",
            output.scale().to_bits(),
            contract.output().scale_bits()
        ));
    }
    let delta = super::refresh::counter_delta(&ev.counter(), &before)
        .ok_or_else(|| HekitError::InvalidParams("packed bootstrap counter reset".into()))?;
    if &delta != contract.cost_per_boot() {
        return invalid("packed bootstrap actual cost differs from contract");
    }
    Ok((output, delta))
}
#[allow(clippy::too_many_arguments)]
pub(super) fn bootstrap_hidden<E: HeOps>(
    ev: &E,
    layer: &LayerGeometry,
    id: usize,
    hidden: &PackedVector<E::Ct>,
    boot: &dyn PlannedBootstrap<E>,
    contract: &PackedBootstrapContract,
    events: &mut Vec<PackedBootstrapEvent<E::Ct>>,
    role: &mut Option<PackedRefreshRole>,
) -> Result<PackedVector<E::Ct>> {
    *role = Some(PackedRefreshRole::Hidden {
        stage: layer.ordinal() / 2,
        record: id as u64,
    });
    let input = CanonicalHidden::new(ev, layer, id as u64, hidden)?.boot_input(ev)?;
    let (output, delta) = bootstrap_observed(ev, boot, contract, &input)?;
    let selected = PackedVector::new(ev, hidden.spec(), output)?;
    events.push(PackedBootstrapEvent {
        role: input.role(),
        source: hidden.ciphertext().clone(),
        projected: None,
        input: input.ciphertext().clone(),
        output: selected.ciphertext().clone(),
        delta,
    });
    *role = None;
    Ok(selected)
}

type PreparedExecution<C> = (PackedNiCandidate<C>, PackedStepReceipt<C>, OpCounter);
