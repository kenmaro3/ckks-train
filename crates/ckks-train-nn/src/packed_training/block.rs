//! Immutable local-block composition. Conditional metadata admission is not numerical proof.
use super::*;
use super::{dag::checked_counts, refresh::counter_delta, types::check_ct};
use crate::layout_reboot::GradRelin;
use ckks_train_ckks::{
    CkksParams, DiagonalEntry, Encoder, HeCiphertext, HeOps, OpCounter, metadata::CtMeta,
};
use ckks_train_core::error::{HekitError, Result};
use std::sync::Arc;

/// One contiguous physical partition; valid counts never form independent means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedPartition {
    start: usize,
    end: usize,
    valid: usize,
}
impl PackedPartition {
    /// Declares a range; the complete batch constructor checks coverage and count.
    pub fn new(start: usize, end: usize, declared_valid: usize) -> Self {
        Self {
            start,
            end,
            valid: declared_valid,
        }
    }
    /// First physical record.
    pub fn start(&self) -> usize {
        self.start
    }
    /// Exclusive physical end.
    pub fn end(&self) -> usize {
        self.end
    }
    /// Checked number of valid records in this range.
    pub fn valid_count(&self) -> usize {
        self.valid
    }
}
fn partitions(ids: &[usize], parts: &[PackedPartition], physical: usize) -> Result<()> {
    if ids.is_empty()
        || ids.windows(2).any(|w| w[0] >= w[1])
        || ids.last().is_some_and(|&i| i >= physical)
    {
        return invalid("packed graph valid IDs must be nonempty, ordered physical positions");
    }
    let mut end = 0;
    let mut total = 0usize;
    for p in parts {
        if p.start != end
            || p.start >= p.end
            || p.end > physical
            || p.valid != ids.iter().filter(|&&i| i >= p.start && i < p.end).count()
        {
            return invalid("packed graph partition coverage or valid count mismatch");
        }
        end = p.end;
        total = total
            .checked_add(p.valid)
            .ok_or_else(|| HekitError::InvalidParams("packed graph count overflow".into()))?;
    }
    if end != physical || total != ids.len() {
        return invalid("packed graph incomplete partition coverage");
    }
    Ok(())
}
/// Complete parameter-only next-batch schedule. No executable payload identity.
#[derive(Clone, Debug)]
pub struct PackedGraphBatchMeta {
    params: Arc<CkksParams>,
    geometry: PackedGeometryPlan,
    index: u64,
    ids: Vec<usize>,
    inputs: Vec<CtMeta>,
    targets: Vec<Vec<CtMeta>>,
    partitions: Vec<PackedPartition>,
}
impl PackedGraphBatchMeta {
    /// Checks all histories and ordered physical partition coverage before any HE.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        params: Arc<CkksParams>,
        geometry: &PackedGeometryPlan,
        index: u64,
        ordered_valid_ids: Vec<usize>,
        initial_input_histories: Vec<CtMeta>,
        per_stage_target_histories: Vec<Vec<CtMeta>>,
        partitions: Vec<PackedPartition>,
    ) -> Result<Self> {
        let n = ordered_valid_ids.len();
        let physical = partitions.last().map_or(0, |p| p.end);
        partitions_check(&ordered_valid_ids, &partitions, physical)?;
        if geometry.layout().slots() != params.slots()
            || initial_input_histories.len() != n
            || per_stage_target_histories.len() != geometry.stages().len()
            || per_stage_target_histories.iter().any(|h| h.len() != n)
        {
            return invalid("packed graph batch history dimensions mismatch");
        }
        for h in initial_input_histories
            .iter()
            .chain(per_stage_target_histories.iter().flatten())
        {
            CtMeta::new(&params, h.level(), h.scale())?;
            if h.scale_bits() != params.scale().to_bits() {
                return Err(HekitError::ScaleMismatch);
            }
        }
        Ok(Self {
            params,
            geometry: geometry.clone(),
            index,
            ids: ordered_valid_ids,
            inputs: initial_input_histories,
            targets: per_stage_target_histories,
            partitions,
        })
    }
    /// Original physical valid IDs, preserving holes.
    pub fn valid_ids(&self) -> &[usize] {
        &self.ids
    }
    /// One denominator across all partitions.
    pub fn valid_count(&self) -> usize {
        self.ids.len()
    }
    /// Schedule update position.
    pub fn update_index(&self) -> u64 {
        self.index
    }
    /// Physical partition coverage.
    pub fn partitions(&self) -> &[PackedPartition] {
        &self.partitions
    }
    fn stage(&self, s: usize, restored: CtMeta) -> Result<PackedBatchMeta> {
        PackedBatchMeta::new(
            self.params.clone(),
            self.index,
            self.geometry.layers()[2 * s].input(),
            self.geometry.layers()[2 * s + 1].output(),
            self.ids
                .iter()
                .enumerate()
                .map(|(i, &id)| {
                    PackedRecordMeta::new(
                        id,
                        if s == 0 { self.inputs[i] } else { restored },
                        self.targets[s][i],
                    )
                })
                .collect(),
        )
    }
}
// Keep constructor argument and coverage function names distinct.
fn partitions_check(ids: &[usize], p: &[PackedPartition], n: usize) -> Result<()> {
    partitions(ids, p, n)
}
/// Explicit known future, unknown conservative checkpoint, or terminal checkpoint.
#[derive(Clone, Debug)]
pub enum PackedGraphNext {
    /// Complete next metadata and public rate/relinearization policy.
    Known(PackedGraphBatchMeta, PackedUpdateMeta),
    /// Future admission remains unknown; child policies conservatively refresh.
    Unknown,
    /// Terminal restartable checkpoint; not future admission.
    Terminal,
}

/// All old stages in one immutable transaction domain.
#[derive(Debug)]
pub struct PackedBlockGraph<C> {
    geometry: PackedGeometryPlan,
    stages: Vec<PackedStage<C>>,
    token: Arc<()>,
}
impl<C: HeCiphertext + Clone> PackedBlockGraph<C> {
    /// Validates exact stage ownership, contexts, learning domains and adjacency.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        geometry: PackedGeometryPlan,
        stages: Vec<PackedStage<C>>,
    ) -> Result<Self> {
        let graph = Self {
            geometry,
            stages,
            token: Arc::new(()),
        };
        graph.metadata(ev)?;
        Ok(graph)
    }
    /// Immutable old stages in execution order.
    pub fn stages(&self) -> &[PackedStage<C>] {
        &self.stages
    }
    /// Whole graph geometry.
    pub fn geometry(&self) -> &PackedGeometryPlan {
        &self.geometry
    }
    fn metadata<E: HeOps<Ct = C>>(&self, ev: &E) -> Result<Vec<PackedStageMeta>> {
        let states = self
            .stages
            .iter()
            .map(|s| s.metadata(ev))
            .collect::<Result<Vec<_>>>()?;
        validate_states(ev.params(), &self.geometry, &states)?;
        Ok(states)
    }
}
fn validate_states(
    p: &Arc<CkksParams>,
    g: &PackedGeometryPlan,
    states: &[PackedStageMeta],
) -> Result<()> {
    if states.len() != g.stages().len() || states.is_empty() {
        return invalid("packed graph stage count mismatch");
    }
    let index = states[0].indices[0];
    for (s, state) in states.iter().enumerate() {
        if !Arc::ptr_eq(p, &state.params)
            || state.layers != [g.layers()[2 * s].clone(), g.layers()[2 * s + 1].clone()]
            || state.classes != g.stages()[s].real_classes()
            || state.indices != [index; 2]
            || (s > 0 && states[s - 1].layers[0].output() != state.layers[0].input())
        {
            return invalid("packed graph stage geometry, adjacency, context or index mismatch");
        }
        PackedStageMeta::new(
            p.clone(),
            g,
            s,
            state.config,
            state.weights,
            state.velocities,
            state.biases,
            state.indices,
        )?;
    }
    Ok(())
}
fn vector_meta<E: HeOps>(ev: &E, v: &PackedVector<E::Ct>, spec: VectorSpec) -> Result<CtMeta> {
    check_ct(ev, v.ciphertext(), spec.layout().slots())?;
    if v.spec() != spec {
        return invalid("packed graph vector representation mismatch");
    }
    if v.ciphertext().scale().to_bits() != ev.params().scale().to_bits() {
        return Err(HekitError::ScaleMismatch);
    }
    CtMeta::new(ev.params(), v.ciphertext().level(), v.ciphertext().scale())
}
/// One original physical record with independently encrypted local targets.
#[derive(Clone, Debug)]
pub struct PackedGraphRecord<C> {
    id: usize,
    input: PackedVector<C>,
    targets: Vec<PackedVector<C>>,
    valid: bool,
}
impl<C> PackedGraphRecord<C> {
    /// Declares encrypted payloads; the checked batch validates even invalid records.
    pub fn new(
        id: usize,
        input: PackedVector<C>,
        targets: Vec<PackedVector<C>>,
        valid: bool,
    ) -> Self {
        Self {
            id,
            input,
            targets,
            valid,
        }
    }
    /// Original physical record ID.
    pub fn id(&self) -> usize {
        self.id
    }
    /// Public validity flag.
    pub fn valid(&self) -> bool {
        self.valid
    }
    /// Encrypted initial input.
    pub fn input(&self) -> &PackedVector<C> {
        &self.input
    }
    /// Ordered stage-specific encrypted targets.
    pub fn targets(&self) -> &[PackedVector<C>] {
        &self.targets
    }
}
/// Checked immutable encrypted batch with private payload identity.
#[derive(Debug)]
pub struct PackedGraphBatch<C> {
    records: Vec<PackedGraphRecord<C>>,
    partitions: Vec<PackedPartition>,
    ids: Vec<usize>,
    token: Arc<()>,
}
impl<C: HeCiphertext + Clone> PackedGraphBatch<C> {
    /// Checks the full physical batch, all targets and partition coverage without HE.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        graph: &PackedBlockGraph<C>,
        records: Vec<PackedGraphRecord<C>>,
        partitions: Vec<PackedPartition>,
    ) -> Result<Self> {
        let ids = records.iter().filter(|r| r.valid).map(|r| r.id).collect();
        let batch = Self {
            records,
            partitions,
            ids,
            token: Arc::new(()),
        };
        batch.metadata(ev, graph)?;
        Ok(batch)
    }
    /// Original physical records, including invalid holes.
    pub fn records(&self) -> &[PackedGraphRecord<C>] {
        &self.records
    }
    /// Ordered valid physical IDs.
    pub fn valid_ids(&self) -> &[usize] {
        &self.ids
    }
    /// One logical denominator.
    pub fn valid_count(&self) -> usize {
        self.ids.len()
    }
    /// Original physical partition map.
    pub fn partitions(&self) -> &[PackedPartition] {
        &self.partitions
    }
    /// Exact histories for this graph's update position.
    pub fn metadata<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        graph: &PackedBlockGraph<C>,
    ) -> Result<PackedGraphBatchMeta> {
        partitions(&self.ids, &self.partitions, self.records.len())?;
        let stages = graph.stages.len();
        let mut inputs = vec![];
        let mut targets = vec![vec![]; stages];
        for (id, r) in self.records.iter().enumerate() {
            if r.id != id || r.targets.len() != stages {
                return invalid("packed graph original position or target count mismatch");
            }
            let h = vector_meta(ev, &r.input, graph.geometry.layers()[0].input())?;
            if r.valid {
                inputs.push(h);
            }
            for (s, target) in r.targets.iter().enumerate() {
                let h = vector_meta(ev, target, graph.geometry.layers()[2 * s + 1].output())?;
                if r.valid {
                    targets[s].push(h);
                }
            }
        }
        PackedGraphBatchMeta::new(
            ev.params().clone(),
            &graph.geometry,
            graph.stages[0].layers()[0].update_index(),
            self.ids.clone(),
            inputs,
            targets,
            self.partitions.clone(),
        )
    }
    fn child(&self, s: usize, hidden: Option<&[PackedVector<C>]>) -> Result<Vec<PackedRecord<C>>> {
        if hidden.is_some_and(|h| h.len() != self.ids.len()) {
            return invalid("packed graph handoff count mismatch");
        }
        let mut valid = 0;
        Ok(self
            .records
            .iter()
            .map(|r| {
                let input = match hidden {
                    None => r.input.clone(),
                    Some(h) => {
                        let i = if r.valid {
                            let i = valid;
                            valid += 1;
                            i
                        } else {
                            0
                        };
                        h[i].clone()
                    }
                };
                PackedRecord::new(input, r.targets[s].clone(), r.valid)
            })
            .collect())
    }
}
fn same_geometry(a: &PackedGeometryPlan, b: &PackedGeometryPlan) -> bool {
    a.layers() == b.layers() && a.stages() == b.stages()
}
fn same_batch(a: &PackedGraphBatchMeta, b: &PackedGraphBatchMeta) -> bool {
    Arc::ptr_eq(&a.params, &b.params)
        && same_geometry(&a.geometry, &b.geometry)
        && a.index == b.index
        && a.ids == b.ids
        && a.inputs == b.inputs
        && a.targets == b.targets
        && a.partitions == b.partitions
}
fn keys<E: HeOps>(ev: &E, g: &PackedGeometryPlan) -> Result<()> {
    if !ev.has_relin_key()
        || g.layout()
            .galois_steps()
            .iter()
            .any(|&s| !ev.has_galois_key(s))
    {
        return invalid("packed graph missing key union");
    }
    Ok(())
}
/// Complete child schedules and inclusive successful graph prediction.
#[derive(Clone, Debug)]
pub struct PackedGraphNiPlan {
    children: Vec<PackedNiPlan>,
    batch: PackedGraphBatchMeta,
    next: PackedGraphNext,
    counts: OpCounter,
    graph_token: Option<Arc<()>>,
    batch_token: Option<Arc<()>>,
}
impl PackedGraphNiPlan {
    /// Parameter-only planning. This constructor never grants execution authority.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        enc: &Encoder,
        geometry: &PackedGeometryPlan,
        states: &[PackedStageMeta],
        batch: &PackedGraphBatchMeta,
        update: PackedUpdateMeta,
        next: PackedGraphNext,
        placement: PackedNiPlacement,
        contract: &PackedBootstrapContract,
    ) -> Result<Self> {
        validate_states(enc.params(), geometry, states)?;
        contract.validate_params(enc.params())?;
        if !Arc::ptr_eq(enc.params(), &batch.params)
            || !same_geometry(geometry, &batch.geometry)
            || batch.index != states[0].indices[0]
        {
            return invalid("packed graph current metadata mismatch");
        }
        if let PackedGraphNext::Known(b, _) = &next {
            if !Arc::ptr_eq(enc.params(), &b.params)
                || !same_geometry(geometry, &b.geometry)
                || batch.index.checked_add(1) != Some(b.index)
            {
                return invalid("packed graph future metadata mismatch");
            }
        }
        let mut counts = OpCounter::default();
        let mut children = vec![];
        for (s, state) in states.iter().enumerate() {
            let future = match &next {
                PackedGraphNext::Known(b, u) => {
                    PackedNextStep::Known(b.stage(s, contract.output())?, *u)
                }
                PackedGraphNext::Unknown => PackedNextStep::Unknown,
                PackedGraphNext::Terminal => PackedNextStep::Terminal,
            };
            let child = PackedNiPlan::new(
                enc,
                state,
                &batch.stage(s, contract.output())?,
                update,
                future,
                placement,
                contract,
            )?;
            counts = checked_counts(&counts, child.predicted_counts())?;
            if s + 1 < states.len() && placement != PackedNiPlacement::Aa {
                for _ in &batch.ids {
                    counts = checked_counts(&counts, contract.cost_per_boot())?;
                }
            }
            children.push(child);
        }
        Ok(Self {
            children,
            batch: batch.clone(),
            next,
            counts,
            graph_token: None,
            batch_token: None,
        })
    }
    /// Child plans in stage order, including their conditional committed histories.
    pub fn children(&self) -> &[PackedNiPlan] {
        &self.children
    }
    /// Inclusive graph counts; nested events are not added again.
    pub fn predicted_counts(&self) -> &OpCounter {
        &self.counts
    }
    /// Explicit retained future policy.
    pub fn next(&self) -> &PackedGraphNext {
        &self.next
    }
    /// Hypothetical committed histories for subsequent parameter-only planning.
    pub fn committed(&self) -> Vec<PackedStageMeta> {
        self.children
            .iter()
            .map(|c| c.committed().clone())
            .collect()
    }
}
/// Entire graph candidate. There is no partial-stage commit API.
#[derive(Debug)]
pub struct PackedGraphCandidate<C> {
    graph: PackedBlockGraph<C>,
}
impl<C> PackedGraphCandidate<C> {
    /// Immutable complete candidate.
    pub fn graph(&self) -> &PackedBlockGraph<C> {
        &self.graph
    }
    /// Infallible whole graph ownership transfer after every boundary succeeds.
    pub fn into_graph(self) -> PackedBlockGraph<C> {
        self.graph
    }
}
/// Transaction boundary at failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedGraphPhase {
    /// Before any current HE.
    Preflight,
    /// Executing an old stage.
    Stage,
    /// Restoring an old hidden boundary.
    Handoff,
    /// Checking complete counts/state.
    Verification,
}
/// Exact origin of a selected hidden handoff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandoffOrigin {
    /// Independently restored old hidden.
    Restored,
    /// Alias of the exact selected child post-bootstrap object.
    AaPost {
        /// Source stage ordinal.
        stage: usize,
        /// Position within the ordered valid list.
        valid_position: usize,
        /// Index within that child's event vector.
        child_event_index: usize,
    },
}
/// Retained encrypted boundary, including an incomplete attempted boundary on failure.
#[derive(Debug)]
pub struct PackedGraphHandoff<C> {
    stage: usize,
    id: usize,
    origin: HandoffOrigin,
    source: PackedVector<C>,
    output: Option<PackedVector<C>>,
    boot: Option<PackedBootstrapEvent<C>>,
    raw_attempt: Option<PackedBootstrapAttempt<C>>,
    request: Option<PackedRefreshRequest<C>>,
    response: Option<PackedRefreshResponse<C>>,
    timing: Option<ClientRefreshTiming>,
    counts: Option<OpCounter>,
    attempted: bool,
    completed: bool,
    validated: bool,
}
impl<C> PackedGraphHandoff<C> {
    /// Source stage.
    pub fn stage(&self) -> usize {
        self.stage
    }
    /// Original physical record.
    pub fn record(&self) -> usize {
        self.id
    }
    /// Exact selected origin, never a numerical identity assertion.
    pub fn origin(&self) -> &HandoffOrigin {
        &self.origin
    }
    /// Old hidden encrypted source.
    pub fn source(&self) -> &PackedVector<C> {
        &self.source
    }
    /// Checked vector output. An invalid NI response remains available through raw_attempt.
    pub fn output(&self) -> Option<&PackedVector<C>> {
        self.output.as_ref()
    }
    /// Raw NI capability attempt, including an available output rejected by validation.
    pub fn raw_attempt(&self) -> Option<&PackedBootstrapAttempt<C>> {
        self.raw_attempt.as_ref()
    }
    /// Validated NI boundary bootstrap, absent for AA alias and I.
    pub fn bootstrap(&self) -> Option<&PackedBootstrapEvent<C>> {
        self.boot.as_ref()
    }
    /// I request with private replay token.
    pub fn request(&self) -> Option<&PackedRefreshRequest<C>> {
        self.request.as_ref()
    }
    /// I response, retained even if token validation failed.
    pub fn response(&self) -> Option<&PackedRefreshResponse<C>> {
        self.response.as_ref()
    }
    /// I timing, when available.
    pub fn timing(&self) -> Option<&ClientRefreshTiming> {
        self.timing.as_ref()
    }
    /// Boundary interval; nested in the inclusive graph interval.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_ref()
    }
    /// Whether this boundary operation was attempted.
    pub fn attempted(&self) -> bool {
        self.attempted
    }
    /// Whether it returned an output.
    pub fn completed(&self) -> bool {
        self.completed
    }
    /// Whether token, output and counter validation passed.
    pub fn validated(&self) -> bool {
        self.validated
    }
}
/// Ordered child evidence; trainer APIs retain separate I and NI capabilities.
#[derive(Debug)]
pub enum PackedGraphChildReceipt<C> {
    /// NI encrypted child evidence.
    Ni(Box<PackedNiReceipt<C>>),
    /// Interactive child evidence.
    Interactive(Box<PackedInteractiveReceipt<C>>),
}
impl<C> PackedGraphChildReceipt<C> {
    /// Old-forward child evidence shared by both modes.
    pub fn step(&self) -> &PackedStepReceipt<C> {
        match self {
            Self::Ni(r) => r.step(),
            Self::Interactive(r) => r.step(),
        }
    }
}
/// Typed partial child failure.
#[derive(Debug)]
pub enum PackedGraphChildFailure<C> {
    /// NI child failure retaining partial encrypted events.
    Ni(PackedNiFailure<C>),
    /// Interactive child failure retaining completed refresh evidence.
    Interactive(PackedRefreshFailure),
}
/// Failure without a graph commit method. Old state remains borrowed and unchanged.
#[derive(Debug)]
pub struct PackedGraphFailure<C> {
    phase: PackedGraphPhase,
    stage: usize,
    role: Option<PackedRefreshRole>,
    children: Vec<PackedGraphChildReceipt<C>>,
    handoffs: Vec<PackedGraphHandoff<C>>,
    child: Option<Box<PackedGraphChildFailure<C>>>,
    counts: Option<Box<OpCounter>>,
    error: Box<HekitError>,
}
impl<C> PackedGraphFailure<C> {
    /// Last boundary reached.
    pub fn phase(&self) -> PackedGraphPhase {
        self.phase
    }
    /// Stage at that boundary.
    pub fn stage(&self) -> usize {
        self.stage
    }
    /// Current attempted refresh role.
    pub fn role(&self) -> Option<PackedRefreshRole> {
        self.role
    }
    /// Fully completed previous child receipts.
    pub fn children(&self) -> &[PackedGraphChildReceipt<C>] {
        &self.children
    }
    /// Completed and current attempted handoffs.
    pub fn handoffs(&self) -> &[PackedGraphHandoff<C>] {
        &self.handoffs
    }
    /// Typed partial current-child evidence.
    pub fn child_failure(&self) -> Option<&PackedGraphChildFailure<C>> {
        self.child.as_deref()
    }
    /// Single inclusive interval; None if counters reset.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_deref()
    }
    /// Original error.
    pub fn error(&self) -> &HekitError {
        &self.error
    }
}
/// Completed graph transaction evidence with one inclusive counter interval.
#[derive(Debug)]
pub struct PackedGraphReceipt<C> {
    children: Vec<PackedGraphChildReceipt<C>>,
    handoffs: Vec<PackedGraphHandoff<C>>,
    ids: Vec<usize>,
    partitions: Vec<PackedPartition>,
    counts: OpCounter,
    predicted: OpCounter,
}
impl<C> PackedGraphReceipt<C> {
    /// Ordered child receipts.
    pub fn children(&self) -> &[PackedGraphChildReceipt<C>] {
        &self.children
    }
    /// Original valid IDs.
    pub fn valid_ids(&self) -> &[usize] {
        &self.ids
    }
    /// Physical partition map.
    pub fn partitions(&self) -> &[PackedPartition] {
        &self.partitions
    }
    /// Old hidden boundary evidence.
    pub fn handoffs(&self) -> &[PackedGraphHandoff<C>] {
        &self.handoffs
    }
    /// Inclusive evaluator interval, counted once.
    pub fn actual_counts(&self) -> &OpCounter {
        &self.counts
    }
    /// Preflight predicted server counts, excluding opaque interactive callback work.
    pub fn predicted_counts(&self) -> &OpCounter {
        &self.predicted
    }
}
/// NI graph receipt.
pub type PackedGraphNiReceipt<C> = PackedGraphReceipt<C>;
/// NI graph failure.
pub type PackedGraphNiFailure<C> = PackedGraphFailure<C>;
/// NI graph transaction result.
pub type PackedGraphNiResult<C> = std::result::Result<
    (PackedGraphCandidate<C>, PackedGraphNiReceipt<C>),
    PackedGraphNiFailure<C>,
>;
/// Interactive graph transaction result.
pub type PackedGraphInteractiveResult<C> =
    std::result::Result<(PackedGraphCandidate<C>, PackedGraphReceipt<C>), PackedGraphFailure<C>>;
fn failed<C>(
    phase: PackedGraphPhase,
    stage: usize,
    children: Vec<PackedGraphChildReceipt<C>>,
    handoffs: Vec<PackedGraphHandoff<C>>,
    counts: Option<OpCounter>,
    error: HekitError,
) -> PackedGraphFailure<C> {
    PackedGraphFailure {
        phase,
        stage,
        role: None,
        children,
        handoffs,
        child: None,
        counts: counts.map(Box::new),
        error: Box::new(error),
    }
}
fn handoff<C>(
    stage: usize,
    id: usize,
    source: PackedVector<C>,
    origin: HandoffOrigin,
) -> PackedGraphHandoff<C> {
    PackedGraphHandoff {
        stage,
        id,
        origin,
        source,
        output: None,
        boot: None,
        raw_attempt: None,
        request: None,
        response: None,
        timing: None,
        counts: None,
        attempted: false,
        completed: false,
        validated: false,
    }
}

/// Public-key-only graph trainer. No secret, client, materializer or diagnostic route.
pub struct PackedBlockNiTrainer<'a, E: HeOps, B: PlannedBootstrap<E>> {
    ev: &'a E,
    enc: &'a Encoder,
    boot: &'a B,
    placement: PackedNiPlacement,
    contract: PackedBootstrapContract,
    child: PackedNiTrainer<'a, E, B>,
}
impl<'a, E: HeOps, B: PlannedBootstrap<E>> PackedBlockNiTrainer<'a, E, B> {
    /// Binds external immutable evaluator/encoder/bootstrap borrows; not self-referential.
    pub fn new(
        ev: &'a E,
        enc: &'a Encoder,
        boot: &'a B,
        placement: PackedNiPlacement,
    ) -> Result<Self> {
        let child = PackedNiTrainer::new(ev, enc, boot, placement)?;
        Ok(Self {
            ev,
            enc,
            boot,
            placement,
            contract: boot.contract()?,
            child,
        })
    }
    /// Validates the whole current/future graph before the first current operation.
    pub fn plan_update(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedGraphBatch<E::Ct>,
        rate: f64,
        relin: GradRelin,
        next: PackedGraphNext,
    ) -> Result<PackedGraphNiPlan> {
        self.boot.validate_binding(self.ev, &self.contract)?;
        keys(self.ev, &graph.geometry)?;
        let mut plan = PackedGraphNiPlan::new(
            self.enc,
            &graph.geometry,
            &graph.metadata(self.ev)?,
            &batch.metadata(self.ev, graph)?,
            PackedUpdateMeta::new(rate, relin, self.ev.config().lazy_relin)?,
            next,
            self.placement,
            &self.contract,
        )?;
        plan.graph_token = Some(graph.token.clone());
        plan.batch_token = Some(batch.token.clone());
        Ok(plan)
    }
    /// Preflights and prepares one complete graph update.
    pub fn prepare_update(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedGraphBatch<E::Ct>,
        rate: f64,
        relin: GradRelin,
        next: PackedGraphNext,
    ) -> PackedGraphNiResult<E::Ct> {
        let before = self.ev.counter();
        let plan = self
            .plan_update(graph, batch, rate, relin, next)
            .map_err(|e| {
                failed(
                    PackedGraphPhase::Preflight,
                    0,
                    vec![],
                    vec![],
                    counter_delta(&self.ev.counter(), &before),
                    e,
                )
            })?;
        self.prepare_planned(graph, batch, &plan)
    }
    /// Requires exact graph, batch payload and current-generation identities.
    pub fn prepare_planned(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedGraphBatch<E::Ct>,
        plan: &PackedGraphNiPlan,
    ) -> PackedGraphNiResult<E::Ct> {
        let before = self.ev.counter();
        let mut children = vec![];
        let mut handoffs = vec![];
        let mut stage = 0;
        let mut phase = PackedGraphPhase::Preflight;
        let mut child_failure = None;
        let mut role = None;
        let result = (|| -> Result<PackedBlockGraph<E::Ct>> {
            if !plan
                .graph_token
                .as_ref()
                .is_some_and(|t| Arc::ptr_eq(t, &graph.token))
                || !plan
                    .batch_token
                    .as_ref()
                    .is_some_and(|t| Arc::ptr_eq(t, &batch.token))
            {
                return invalid(
                    "packed graph plan is hypothetical, stale or bound to another payload",
                );
            }
            let fresh = self.plan_update(
                graph,
                batch,
                plan.children[0].update.rate,
                plan.children[0].update.relin,
                plan.next.clone(),
            )?;
            if !same_batch(&fresh.batch, &plan.batch)
                || fresh.counts != plan.counts
                || fresh.children.iter().zip(&plan.children).any(|(a, b)| {
                    a.placement != b.placement
                        || a.update != b.update
                        || a.contract.metadata().descriptor_identity_bytes()
                            != b.contract.metadata().descriptor_identity_bytes()
                })
            {
                return invalid("packed graph plan capability or metadata changed");
            }
            let mut candidates = vec![];
            let mut selected: Option<Vec<PackedVector<E::Ct>>> = None;
            for s in 0..graph.stages.len() {
                stage = s;
                phase = PackedGraphPhase::Stage;
                let records = batch.child(s, selected.as_deref())?;
                let (candidate, receipt) =
                    match self
                        .child
                        .prepare_planned(&graph.stages[s], &records, &plan.children[s])
                    {
                        Ok(v) => v,
                        Err(f) => {
                            role = f.role();
                            let e = HekitError::InvalidParams(format!(
                                "packed graph NI child failed: {}",
                                f.error()
                            ));
                            child_failure = Some(Box::new(PackedGraphChildFailure::Ni(f)));
                            return Err(e);
                        }
                    };
                candidates.push(candidate.into_stage());
                children.push(PackedGraphChildReceipt::Ni(Box::new(receipt)));
                if s + 1 == graph.stages.len() {
                    continue;
                }
                phase = PackedGraphPhase::Handoff;
                let PackedGraphChildReceipt::Ni(receipt) = children.last().unwrap() else {
                    unreachable!()
                };
                let mut next = vec![];
                for (j, &id) in batch.ids.iter().enumerate() {
                    let aa = self.placement == PackedNiPlacement::Aa;
                    let source = if aa {
                        receipt.step().forward().hidden_post().ok_or_else(|| {
                            HekitError::InvalidParams(
                                "AA child has no selected post payload".into(),
                            )
                        })?[j]
                            .clone()
                    } else {
                        receipt.step().forward().hidden_pre()[j].clone()
                    };
                    let origin = if aa {
                        let child_event_index = receipt
                            .events()
                            .iter()
                            .position(|e| {
                                e.role()
                                    == PackedRefreshRole::Hidden {
                                        stage: s,
                                        record: id as u64,
                                    }
                            })
                            .ok_or_else(|| {
                                HekitError::InvalidParams(
                                    "AA selected payload lacks child event".into(),
                                )
                            })?;
                        HandoffOrigin::AaPost {
                            stage: s,
                            valid_position: j,
                            child_event_index,
                        }
                    } else {
                        HandoffOrigin::Restored
                    };
                    handoffs.push(handoff(s, id, source.clone(), origin));
                    let event = handoffs.last_mut().unwrap();
                    let start = self.ev.counter();
                    event.attempted = true;
                    let output = if aa {
                        Ok(source)
                    } else {
                        let mut events = vec![];
                        let capture = BootstrapCapture::new(
                            self.boot,
                            PackedRefreshRole::Hidden {
                                stage: s,
                                record: id as u64,
                            },
                        );
                        let out = super::trainer::bootstrap_hidden(
                            self.ev,
                            &graph.geometry.layers()[2 * s],
                            id,
                            &source,
                            &capture,
                            &self.contract,
                            &mut events,
                            &mut role,
                        );
                        event.raw_attempt = capture.finish(out.is_ok());
                        event.completed = event.raw_attempt.as_ref().is_some_and(|a| a.completed());
                        event.boot = events.pop();
                        out
                    };
                    event.counts = counter_delta(&self.ev.counter(), &start);
                    let output = output?;
                    event.completed = true;
                    event.output = Some(output.clone());
                    if vector_meta(self.ev, &output, graph.geometry.layers()[2 * s + 2].input())?
                        != self.contract.output()
                    {
                        return invalid("packed graph restored boundary metadata mismatch");
                    }
                    event.validated = true;
                    next.push(output);
                }
                selected = Some(next);
            }
            phase = PackedGraphPhase::Verification;
            if counter_delta(&self.ev.counter(), &before).as_ref() != Some(&plan.counts) {
                return invalid("packed graph inclusive counts mismatch");
            }
            PackedBlockGraph::new(self.ev, graph.geometry.clone(), candidates)
        })();
        let counts = counter_delta(&self.ev.counter(), &before);
        match result {
            Ok(graph) => Ok((
                PackedGraphCandidate { graph },
                PackedGraphReceipt {
                    children,
                    handoffs,
                    ids: batch.ids.clone(),
                    partitions: batch.partitions.clone(),
                    counts: counts.unwrap(),
                    predicted: plan.counts.clone(),
                },
            )),
            Err(error) => {
                let mut f = failed(phase, stage, children, handoffs, counts, error);
                f.child = child_failure;
                f.role = role;
                Err(f)
            }
        }
    }
}

/// Explicit inference policy, independent of training placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedInferencePlan {
    /// No refresh; accepted only when the complete exact depth trace is feasible.
    NoRefresh,
    /// NI restore every nonfinal backbone output; no final-stack hidden refresh.
    BoundaryBootstrap,
    /// NI restore every selected hidden, including the final-stack hidden.
    AllHiddenBootstrap,
    /// Interactive restore only nonfinal backbone boundaries.
    ClientBoundary,
}
/// Input-only physical graph batch; no target or update fields.
#[derive(Debug)]
pub struct PackedForwardBatch<C> {
    records: Vec<PackedForwardRecord<C>>,
    ids: Vec<usize>,
    partitions: Vec<PackedPartition>,
    token: Arc<()>,
}
impl<C: HeCiphertext + Clone> PackedForwardBatch<C> {
    /// Checks every physical input and exact partition coverage without inspecting state.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        graph: &PackedBlockGraph<C>,
        records: Vec<PackedForwardRecord<C>>,
        partitions: Vec<PackedPartition>,
    ) -> Result<Self> {
        let ids = records.iter().filter(|r| r.valid).map(|r| r.id).collect();
        let out = Self {
            records,
            ids,
            partitions,
            token: Arc::new(()),
        };
        out.histories(ev, graph)?;
        Ok(out)
    }
    /// Original input records, including invalid holes.
    pub fn records(&self) -> &[PackedForwardRecord<C>] {
        &self.records
    }
    /// Ordered valid physical identities.
    pub fn valid_ids(&self) -> &[usize] {
        &self.ids
    }
    /// Original partition coverage.
    pub fn partitions(&self) -> &[PackedPartition] {
        &self.partitions
    }
    fn histories<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        graph: &PackedBlockGraph<C>,
    ) -> Result<Vec<(usize, CtMeta)>> {
        partitions(&self.ids, &self.partitions, self.records.len())?;
        let mut out = vec![];
        for (id, r) in self.records.iter().enumerate() {
            if id != r.id {
                return invalid("packed forward batch physical identity mismatch");
            }
            let h = vector_meta(ev, &r.input, graph.geometry.layers()[0].input())?;
            if r.valid {
                out.push((id, h));
            }
        }
        Ok(out)
    }
    fn child(&self, values: Option<&[PackedVector<C>]>) -> Result<Vec<PackedForwardRecord<C>>> {
        if values.is_some_and(|v| v.len() != self.ids.len()) {
            return invalid("packed graph forward handoff length mismatch");
        }
        let mut j = 0;
        Ok(self
            .records
            .iter()
            .map(|r| {
                let input = match values {
                    None => r.input.clone(),
                    Some(v) => {
                        let k = if r.valid {
                            let k = j;
                            j += 1;
                            k
                        } else {
                            0
                        };
                        v[k].clone()
                    }
                };
                PackedForwardRecord::new(r.id, input, r.valid)
            })
            .collect())
    }
}
/// Full graph forward-only path and its inclusive exact prediction.
#[derive(Clone, Debug)]
pub struct PackedGraphForwardPlan {
    children: Vec<PackedForwardPlan>,
    policy: PackedInferencePlan,
    counts: OpCounter,
    graph: Arc<()>,
    batch: Arc<()>,
}
impl PackedGraphForwardPlan {
    /// Selected independent inference policy.
    pub fn policy(&self) -> PackedInferencePlan {
        self.policy
    }
    /// Ordered backbone paths followed by the final output stack.
    pub fn children(&self) -> &[PackedForwardPlan] {
        &self.children
    }
    /// Inclusive predicted operations, with each selected refresh counted once.
    pub fn predicted_counts(&self) -> &OpCounter {
        &self.counts
    }
}
/// Full inference evidence; no graph candidate or commit method.
#[derive(Debug)]
pub struct PackedGraphForwardReceipt<C> {
    children: Vec<PackedForwardReceipt<C>>,
    handoffs: Vec<PackedGraphHandoff<C>>,
    counts: OpCounter,
    plan: PackedGraphForwardPlan,
}
impl<C> PackedGraphForwardReceipt<C> {
    /// Per-stage actual selected path evidence.
    pub fn children(&self) -> &[PackedForwardReceipt<C>] {
        &self.children
    }
    /// Interactive handoffs; NI events are nested in the selected stage receipts.
    pub fn handoffs(&self) -> &[PackedGraphHandoff<C>] {
        &self.handoffs
    }
    /// Inclusive evaluator interval.
    pub fn actual_counts(&self) -> &OpCounter {
        &self.counts
    }
    /// Exact prior plan.
    pub fn plan(&self) -> &PackedGraphForwardPlan {
        &self.plan
    }
}
/// Partial forward-only execution, retaining encrypted event evidence.
#[derive(Debug)]
pub struct PackedGraphForwardFailure<C> {
    stage: usize,
    children: Vec<PackedForwardReceipt<C>>,
    handoffs: Vec<PackedGraphHandoff<C>>,
    child: Option<Box<PackedForwardFailure<C>>>,
    counts: Option<Box<OpCounter>>,
    error: HekitError,
}
impl<C> PackedGraphForwardFailure<C> {
    /// Stage where preparation stopped.
    pub fn stage(&self) -> usize {
        self.stage
    }
    /// Completed old-state selected paths.
    pub fn children(&self) -> &[PackedForwardReceipt<C>] {
        &self.children
    }
    /// Partial interactive handoffs.
    pub fn handoffs(&self) -> &[PackedGraphHandoff<C>] {
        &self.handoffs
    }
    /// Partial current selected path.
    pub fn child_failure(&self) -> Option<&PackedForwardFailure<C>> {
        self.child.as_deref()
    }
    /// Inclusive observed interval, or None after reset.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_deref()
    }
    /// Original error.
    pub fn error(&self) -> &HekitError {
        &self.error
    }
}
/// Input-only graph inference result.
pub type PackedGraphForwardResult<C> = std::result::Result<
    (PackedForwardOutput<C>, PackedGraphForwardReceipt<C>),
    PackedGraphForwardFailure<C>,
>;
fn plan_graph_forward<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    graph: &PackedBlockGraph<E::Ct>,
    batch: &PackedForwardBatch<E::Ct>,
    policy: PackedInferencePlan,
    boot: Option<&PackedBootstrapContract>,
    client: Option<CtMeta>,
) -> Result<PackedGraphForwardPlan> {
    let mut inputs = batch.histories(ev, graph)?;
    let mut children = vec![];
    let mut counts = OpCounter::default();
    for (s, stage) in graph.stages.iter().enumerate() {
        let last = s + 1 == graph.stages.len();
        let hidden = if policy == PackedInferencePlan::AllHiddenBootstrap
            || (!last && policy == PackedInferencePlan::BoundaryBootstrap)
        {
            Some(boot.ok_or_else(|| {
                HekitError::InvalidParams("missing forward bootstrap contract".into())
            })?)
        } else {
            None
        };
        let child = stage.plan_forward_histories(
            ev,
            enc,
            inputs,
            if last {
                PackedForwardPart::OutputStack
            } else {
                PackedForwardPart::Backbone
            },
            hidden,
        )?;
        inputs = batch
            .ids
            .iter()
            .copied()
            .zip(child.output_histories().iter().copied())
            .collect();
        if !last && policy == PackedInferencePlan::ClientBoundary {
            let h = client.ok_or_else(|| {
                HekitError::InvalidParams("missing inference client contract".into())
            })?;
            inputs = batch.ids.iter().map(|&id| (id, h)).collect();
        }
        counts = checked_counts(&counts, child.predicted_counts())?;
        children.push(child);
    }
    Ok(PackedGraphForwardPlan {
        children,
        policy,
        counts,
        graph: graph.token.clone(),
        batch: batch.token.clone(),
    })
}
impl<E: HeOps, B: PlannedBootstrap<E>> PackedBlockNiTrainer<'_, E, B> {
    /// Plans all used inference paths without inspecting unused heads or velocities.
    pub fn plan_forward(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedForwardBatch<E::Ct>,
        policy: PackedInferencePlan,
    ) -> Result<PackedGraphForwardPlan> {
        if policy == PackedInferencePlan::ClientBoundary {
            return invalid("NI inference cannot accept a client policy");
        }
        self.boot.validate_binding(self.ev, &self.contract)?;
        plan_graph_forward(
            self.ev,
            self.enc,
            graph,
            batch,
            policy,
            Some(&self.contract),
            None,
        )
    }
    /// Executes a graph-bound input-only plan, returning encrypted predictions and evidence.
    pub fn prepare_forward(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedForwardBatch<E::Ct>,
        plan: &PackedGraphForwardPlan,
    ) -> PackedGraphForwardResult<E::Ct> {
        let before = self.ev.counter();
        let mut children = vec![];
        let mut child_failure = None;
        let mut stage = 0;
        let result = (|| -> Result<PackedForwardOutput<E::Ct>> {
            if !Arc::ptr_eq(&graph.token, &plan.graph) || !Arc::ptr_eq(&batch.token, &plan.batch) {
                return invalid("packed graph inference stale graph or payload");
            }
            let fresh = self.plan_forward(graph, batch, plan.policy)?;
            if fresh.counts != plan.counts || fresh.children.len() != plan.children.len() {
                return invalid("packed graph inference capability changed");
            }
            let mut output: Option<PackedForwardOutput<E::Ct>> = None;
            for (s, old) in graph.stages.iter().enumerate() {
                stage = s;
                let records = batch.child(output.as_ref().map(|o| o.values()))?;
                let boot = (plan.policy == PackedInferencePlan::AllHiddenBootstrap
                    || (s + 1 < graph.stages.len()
                        && plan.policy == PackedInferencePlan::BoundaryBootstrap))
                    .then_some(self.boot as &dyn PlannedBootstrap<E>);
                match old.prepare_forward(self.ev, self.enc, &records, &plan.children[s], boot) {
                    Ok((out, r)) => {
                        output = Some(out);
                        children.push(r);
                    }
                    Err(f) => {
                        let e = HekitError::InvalidParams(format!(
                            "packed graph forward child failed: {}",
                            f.error()
                        ));
                        child_failure = Some(Box::new(f));
                        return Err(e);
                    }
                }
            }
            if counter_delta(&self.ev.counter(), &before).as_ref() != Some(&plan.counts) {
                return invalid("packed graph forward inclusive count mismatch");
            }
            Ok(output.unwrap())
        })();
        let counts = counter_delta(&self.ev.counter(), &before);
        match result {
            Ok(out) => Ok((
                out,
                PackedGraphForwardReceipt {
                    children,
                    handoffs: vec![],
                    counts: counts.unwrap(),
                    plan: plan.clone(),
                },
            )),
            Err(error) => Err(PackedGraphForwardFailure {
                stage,
                children,
                handoffs: vec![],
                child: child_failure,
                counts: counts.map(Box::new),
                error,
            }),
        }
    }
}

fn client_contract<E: HeOps>(ev: &E, c: &PackedRefreshContract) -> Result<CtMeta> {
    if !Arc::ptr_eq(ev.params(), c.params()) || c.delta().to_bits() != ev.params().scale().to_bits()
    {
        return invalid("packed graph client output context mismatch");
    }
    CtMeta::new(ev.params(), c.level(), c.delta())
}
fn client_handoff<E: HeOps, R: PackedClientRefresh<E>>(
    ev: &E,
    layer: &LayerGeometry,
    id: usize,
    source: &PackedVector<E::Ct>,
    contract: &PackedRefreshContract,
    client: &mut R,
    event: &mut PackedGraphHandoff<E::Ct>,
) -> Result<PackedVector<E::Ct>> {
    let request =
        CanonicalHidden::new(ev, layer, id as u64, source)?.request(ev, contract.clone())?;
    event.request = Some(request);
    event.attempted = true;
    let before = ev.counter();
    let response = client.refresh(event.request.as_ref().unwrap());
    event.counts = counter_delta(&ev.counter(), &before);
    let (response, timing) = response?;
    event.completed = true;
    event.timing = Some(timing);
    event.response = Some(response);
    event
        .request
        .as_ref()
        .unwrap()
        .validate_response(ev, event.response.as_ref().unwrap())?;
    let out = PackedVector::new(
        ev,
        source.spec(),
        event.response.as_ref().unwrap().ciphertext().clone(),
    )?;
    event.output = Some(out.clone());
    event.validated = true;
    Ok(out)
}
/// Interactive graph orchestrator. Client capability is accepted only on I methods.
pub struct PackedBlockInteractiveTrainer<'a, E: HeOps> {
    ev: &'a E,
    enc: &'a Encoder,
    child: PackedInteractiveStage<'a, E>,
}
impl<'a, E: HeOps> PackedBlockInteractiveTrainer<'a, E> {
    /// Borrows public evaluator and Encoder, without owning a client callback.
    pub fn new(ev: &'a E, enc: &'a Encoder) -> Result<Self> {
        Ok(Self {
            ev,
            enc,
            child: PackedInteractiveStage::new(ev, enc)?,
        })
    }
    /// Preflights every old stage/target/updated role, then prepares one whole graph.
    pub fn prepare_update<R: PackedClientRefresh<E>>(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedGraphBatch<E::Ct>,
        rate: f64,
        relin: GradRelin,
        client: &mut R,
    ) -> PackedGraphInteractiveResult<E::Ct> {
        let before = self.ev.counter();
        let mut children = vec![];
        let mut handoffs = vec![];
        let mut phase = PackedGraphPhase::Preflight;
        let mut stage = 0;
        let mut role = None;
        let mut child_failure = None;
        let mut predicted = OpCounter::default();
        let result = (|| -> Result<PackedBlockGraph<E::Ct>> {
            let contract = client.output_contract().clone();
            let restored = client_contract(self.ev, &contract)?;
            let states = graph.metadata(self.ev)?;
            let bm = batch.metadata(self.ev, graph)?;
            keys(self.ev, &graph.geometry)?;
            let update = PackedUpdateMeta::new(rate, relin, self.ev.config().lazy_relin)?;
            let mut plans = vec![];
            for (s, state) in states.iter().enumerate() {
                let dag = plan_stage_dag(self.enc, state, &bm.stage(s, restored)?, update, None)?;
                let candidate = dag.candidate();
                PackedStageMeta::new(
                    self.ev.params().clone(),
                    &graph.geometry,
                    s,
                    candidate.config,
                    [restored; 2],
                    candidate.velocities.map(|u| u.map(|_| restored)),
                    candidate.biases.map(|_| [restored; 2]),
                    candidate.indices,
                )?;
                predicted = checked_counts(&predicted, &dag.step().predicted_counts)?;
                plans.push(dag);
            }
            let mut candidates = vec![];
            let mut selected: Option<Vec<PackedVector<E::Ct>>> = None;
            for (s, old) in graph.stages.iter().enumerate() {
                stage = s;
                phase = PackedGraphPhase::Stage;
                let records = batch.child(s, selected.as_deref())?;
                let (candidate, receipt) =
                    match self.child.prepare_step(old, &records, rate, relin, client) {
                        Ok(v) => v,
                        Err(f) => {
                            role = f.role();
                            let e = HekitError::InvalidParams(format!(
                                "packed graph I child failed: {}",
                                f.error()
                            ));
                            child_failure = Some(Box::new(PackedGraphChildFailure::Interactive(f)));
                            return Err(e);
                        }
                    };
                children.push(PackedGraphChildReceipt::Interactive(Box::new(receipt)));
                let receipt = children.last().unwrap();
                if receipt.step().plan.trace != plans[s].step().trace
                    || receipt.step().actual_counts != plans[s].step().predicted_counts
                {
                    return invalid("packed graph I child differs from whole-graph preflight");
                }
                // A client cannot silently change its contract between child calls.
                let m = candidate.stage().metadata(self.ev)?;
                if m.weights != [restored; 2]
                    || m.velocities.iter().flatten().any(|h| *h != restored)
                    || m.biases.is_some_and(|b| b != [restored; 2])
                {
                    return invalid("packed graph I restored role differs from preflight");
                }
                candidates.push(candidate.into_stage());
                if s + 1 == graph.stages.len() {
                    continue;
                }
                phase = PackedGraphPhase::Handoff;
                let source = children.last().unwrap().step().forward().hidden();
                let mut next = vec![];
                for (j, &id) in batch.ids.iter().enumerate() {
                    role = Some(PackedRefreshRole::Hidden {
                        stage: s,
                        record: id as u64,
                    });
                    handoffs.push(handoff(s, id, source[j].clone(), HandoffOrigin::Restored));
                    next.push(client_handoff(
                        self.ev,
                        &graph.geometry.layers()[2 * s],
                        id,
                        &source[j],
                        &contract,
                        client,
                        handoffs.last_mut().unwrap(),
                    )?);
                    role = None;
                }
                selected = Some(next);
            }
            phase = PackedGraphPhase::Verification;
            let mut inclusive = predicted.clone();
            for child in &children {
                if let PackedGraphChildReceipt::Interactive(r) = child {
                    for e in r.events() {
                        inclusive = checked_counts(
                            &inclusive,
                            e.counter_delta().ok_or_else(|| {
                                HekitError::InvalidParams(
                                    "packed graph client counter reset".into(),
                                )
                            })?,
                        )?;
                    }
                }
            }
            for h in &handoffs {
                inclusive = checked_counts(
                    &inclusive,
                    h.counts.as_ref().ok_or_else(|| {
                        HekitError::InvalidParams("packed graph handoff counter reset".into())
                    })?,
                )?;
            }
            if counter_delta(&self.ev.counter(), &before).as_ref() != Some(&inclusive) {
                return invalid("packed graph I inclusive count mismatch");
            }
            PackedBlockGraph::new(self.ev, graph.geometry.clone(), candidates)
        })();
        let counts = counter_delta(&self.ev.counter(), &before);
        match result {
            Ok(graph) => Ok((
                PackedGraphCandidate { graph },
                PackedGraphReceipt {
                    children,
                    handoffs,
                    ids: batch.ids.clone(),
                    partitions: batch.partitions.clone(),
                    counts: counts.unwrap(),
                    predicted,
                },
            )),
            Err(error) => {
                let mut f = failed(phase, stage, children, handoffs, counts, error);
                f.child = child_failure;
                f.role = role;
                Err(f)
            }
        }
    }
    /// Preflights and executes input-only backbone/final-stack inference with a separate I policy.
    pub fn prepare_forward<R: PackedClientRefresh<E>>(
        &self,
        graph: &PackedBlockGraph<E::Ct>,
        batch: &PackedForwardBatch<E::Ct>,
        policy: PackedInferencePlan,
        client: &mut R,
    ) -> PackedGraphForwardResult<E::Ct> {
        let before = self.ev.counter();
        let mut stage = 0;
        let mut children = vec![];
        let mut handoffs = vec![];
        let mut child_failure = None;
        let result = (|| -> Result<(PackedForwardOutput<E::Ct>, PackedGraphForwardPlan)> {
            if !matches!(
                policy,
                PackedInferencePlan::NoRefresh | PackedInferencePlan::ClientBoundary
            ) {
                return invalid("interactive inference needs an I inference policy");
            }
            let contract = client.output_contract().clone();
            let restored = client_contract(self.ev, &contract)?;
            let plan = plan_graph_forward(
                self.ev,
                self.enc,
                graph,
                batch,
                policy,
                None,
                Some(restored),
            )?;
            let mut output: Option<PackedForwardOutput<E::Ct>> = None;
            for (s, old) in graph.stages.iter().enumerate() {
                stage = s;
                let records = batch.child(output.as_ref().map(|o| o.values()))?;
                let (mut out, receipt) =
                    match old.prepare_forward(self.ev, self.enc, &records, &plan.children[s], None)
                    {
                        Ok(v) => v,
                        Err(f) => {
                            let e = HekitError::InvalidParams(format!(
                                "packed graph I inference child failed: {}",
                                f.error()
                            ));
                            child_failure = Some(Box::new(f));
                            return Err(e);
                        }
                    };
                children.push(receipt);
                if s + 1 < graph.stages.len() && policy == PackedInferencePlan::ClientBoundary {
                    for (j, &id) in batch.ids.iter().enumerate() {
                        let source = out.values[j].clone();
                        handoffs.push(handoff(s, id, source.clone(), HandoffOrigin::Restored));
                        out.values[j] = client_handoff(
                            self.ev,
                            &graph.geometry.layers()[2 * s],
                            id,
                            &source,
                            &contract,
                            client,
                            handoffs.last_mut().unwrap(),
                        )?;
                    }
                }
                output = Some(out);
            }
            let mut inclusive = plan.counts.clone();
            for e in &handoffs {
                inclusive = checked_counts(
                    &inclusive,
                    e.counts.as_ref().ok_or_else(|| {
                        HekitError::InvalidParams("packed inference client counter reset".into())
                    })?,
                )?;
            }
            if counter_delta(&self.ev.counter(), &before).as_ref() != Some(&inclusive) {
                return invalid("packed I inference inclusive count mismatch");
            }
            Ok((output.unwrap(), plan))
        })();
        let counts = counter_delta(&self.ev.counter(), &before);
        match result {
            Ok((out, plan)) => Ok((
                out,
                PackedGraphForwardReceipt {
                    children,
                    handoffs,
                    counts: counts.unwrap(),
                    plan,
                },
            )),
            Err(error) => Err(PackedGraphForwardFailure {
                stage,
                children,
                handoffs,
                child: child_failure,
                counts: counts.map(Box::new),
                error,
            }),
        }
    }
}

/// Raw encrypted capability attempt, retained independently of checked vector/event validation.
#[derive(Debug)]
pub struct PackedBootstrapAttempt<C> {
    role: PackedRefreshRole,
    input: C,
    output: Option<C>,
    counts: Option<OpCounter>,
    validated: bool,
}
impl<C> PackedBootstrapAttempt<C> {
    /// Exact role assigned before invoking the capability.
    pub fn role(&self) -> PackedRefreshRole {
        self.role
    }
    /// Actual encrypted argument supplied to the bootstrap capability.
    pub fn input(&self) -> &C {
        &self.input
    }
    /// Raw returned ciphertext, possibly invalid; never presented as a checked packed vector.
    pub fn output(&self) -> Option<&C> {
        self.output.as_ref()
    }
    /// The capability returned an encrypted output, regardless of subsequent validation.
    pub fn completed(&self) -> bool {
        self.output.is_some()
    }
    /// Existing hidden helper accepted output metadata and the seven-field interval.
    pub fn validated(&self) -> bool {
        self.validated
    }
    /// Inclusive attempted capability interval; None after counter reset.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.counts.as_ref()
    }
}
// A per-call encrypted-only recorder. It delegates to the existing helper/capability,
// retaining a raw return before validation. No callback, secret or materializer exists.
pub(super) struct BootstrapCapture<'a, E: HeOps> {
    inner: &'a dyn PlannedBootstrap<E>,
    role: PackedRefreshRole,
    attempt: std::cell::RefCell<Option<PackedBootstrapAttempt<E::Ct>>>,
}
impl<'a, E: HeOps> BootstrapCapture<'a, E> {
    pub(super) fn new(inner: &'a dyn PlannedBootstrap<E>, role: PackedRefreshRole) -> Self {
        Self {
            inner,
            role,
            attempt: std::cell::RefCell::new(None),
        }
    }
    pub(super) fn finish(self, validated: bool) -> Option<PackedBootstrapAttempt<E::Ct>> {
        self.attempt.into_inner().map(|mut a| {
            a.validated = validated;
            a
        })
    }
}
impl<E: HeOps> ckks_train_ckks::BootstrapOps<E> for BootstrapCapture<'_, E> {
    fn bootstrap(&self, ev: &E, input: &E::Ct) -> Result<E::Ct> {
        if self.attempt.borrow().is_some() {
            return invalid("packed encrypted capture is single-use");
        }
        let before = ev.counter();
        self.attempt.replace(Some(PackedBootstrapAttempt {
            role: self.role,
            input: input.clone(),
            output: None,
            counts: None,
            validated: false,
        }));
        let result = self.inner.bootstrap(ev, input);
        let mut attempt = self.attempt.borrow_mut();
        let attempt = attempt.as_mut().unwrap();
        attempt.counts = counter_delta(&ev.counter(), &before);
        if let Ok(output) = &result {
            attempt.output = Some(output.clone());
        }
        result
    }
    fn output_level(&self) -> usize {
        self.inner.output_level()
    }
}
impl<E: HeOps> PlannedBootstrap<E> for BootstrapCapture<'_, E> {
    fn contract(&self) -> Result<PackedBootstrapContract> {
        self.inner.contract()
    }
    fn validate_binding(&self, ev: &E, contract: &PackedBootstrapContract) -> Result<()> {
        self.inner.validate_binding(ev, contract)
    }
}
