//! Checked metadata binding for diagonal study plans, with no HE construction authority.
//!
//! Wire v1 is an ordered LE transcript: domain/revision, trusted input and full-case
//! binding, full coefficients/plan, consumption/partitions/parameters/physical request,
//! stage geometry/initial state, then indexed update and boundary records. Integers and
//! lengths are u64, F64s use their exact bits, options are byte0/1, strings/bytes are
//! length-prefixed. Every trace/attempt/bootstrap/state is retained; no Debug is hashed.
//! Candidate and private synthetic domains cannot construct a canonical plan. Success
//! proves only the conditional metadata contract; the independent gates remain unproved.

use crate::{
    data::sha256_hex,
    study_inputs::{MaterializedProfile, RateCoefficients, StageInput},
    study_profile::{
        Activation, BatchPlan, CaseBackend, CaseMode, CasePacking, CasePlacement, CaseRole,
        CaseScope, Initialization, LayerRole, Model, Optimizer, ProfileMetadata, RegisteredCase,
        SecurityPolicy,
    },
};
use ckks_train_ckks::{
    CkksParams, EvalConfig, OpCounter,
    bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, KeyRoleMetadata, UnvalidatedRequirement},
    },
    metadata::CtMeta,
};
use ckks_train_core::error::HekitError;
use ckks_train_nn::{
    compact::CompactPlan,
    dense::TransposeKeys,
    diagonal_plan::{
        BatchMetadata, BiasPolicy, BootstrapEvent, DiagonalActivation, DiagonalMode,
        DiagonalOptimizer, GradientCoefficients, MicroMetadata, PlanError, PlannedHandoff,
        PlannedStageUpdate, StageGeometry, StageState, UpdateSpec, initial_stage_metadata,
        plan_stage_geometry, plan_stage_handoff, plan_stage_update,
    },
    layout::Layout,
    trainer_ni::BootPlacement,
};
use std::{
    collections::BTreeSet,
    fmt,
    sync::{Arc, Mutex},
};

/// Intended consumption; Quick is exactly two updates of a sealed Full registration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Consumption {
    /// Every actual update in the full materialized plan.
    Full,
    /// Two-update metadata prefix; never a registered quick execution capability.
    Quick,
}
/// Explicit ordered slice within one logical update, measured in valid rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MicroPart {
    /// Offset within this logical update, never the full dataset.
    pub start: usize,
    /// Positive number of valid rows, at most the physical lanes.
    pub valid: usize,
}
/// Operational transcript limits, independent from CKKS/runtime memory admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiptLimits {
    /// Maximum combined transcript and retained metadata/index requested payload bytes.
    /// Does not bound allocator overhead, transient P2 allocations or RSS.
    pub max_bytes: usize,
    /// Maximum indexed stage-update and boundary records.
    pub max_records: usize,
}
impl Default for ReceiptLimits {
    fn default() -> Self {
        Self {
            max_bytes: 64 << 20,
            max_records: 1_000_000,
        }
    }
}
/// Cumulative pure-planner work counters for one explicitly shared budget.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanningUsage {
    /// Geometry API calls.
    pub geometries: usize,
    /// Initial-state API calls.
    pub initial_states: usize,
    /// Stage-update API calls, including rejected calls.
    pub updates: usize,
    /// Boundary API calls, including rejected calls.
    pub handoffs: usize,
    /// Production wrapper invocations, including validation failures.
    pub wrapper_calls: usize,
}
/// Scoped operational work cap; counters grant no scientific or allocation capability.
#[derive(Debug)]
pub struct PlanningBudget {
    limits: PlanningUsage,
    usage: Mutex<PlanningUsage>,
}
impl PlanningBudget {
    /// Creates an explicit shared cap, checked before every corresponding P2 call.
    pub fn new(limits: PlanningUsage) -> Self {
        Self {
            limits,
            usage: Mutex::new(PlanningUsage::default()),
        }
    }
    /// Returns counters even if a previous caller panicked while observing the budget.
    pub fn usage(&self) -> PlanningUsage {
        *self.usage.lock().unwrap_or_else(|e| e.into_inner())
    }
    fn charge(&self, field: Work) -> Result<()> {
        let mut usage = self
            .usage
            .lock()
            .map_err(|_| issue(IssueKind::PlanningBudgetExceeded, "poisoned work budget"))?;
        let (used, limit) = match field {
            Work::Geometry => (&mut usage.geometries, self.limits.geometries),
            Work::Initial => (&mut usage.initial_states, self.limits.initial_states),
            Work::Update => (&mut usage.updates, self.limits.updates),
            Work::Handoff => (&mut usage.handoffs, self.limits.handoffs),
            Work::Wrapper => (&mut usage.wrapper_calls, self.limits.wrapper_calls),
        };
        let next = used
            .checked_add(1)
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "work counter"))?;
        if next > limit {
            return Err(issue(
                IssueKind::PlanningBudgetExceeded,
                "pure planning work limit",
            ));
        }
        *used = next;
        Ok(())
    }
}
#[derive(Clone, Copy)]
enum Work {
    Geometry,
    Initial,
    Update,
    Handoff,
    Wrapper,
}
/// Explicit physical and consumption choices; these are requests, never trusted facts.
#[derive(Clone, Debug)]
pub struct DiagonalRequest {
    /// Full or two-update consumption of the immutable full profile.
    pub consumption: Consumption,
    /// None uses the next power of two of the registered logical batch.
    pub physical_lanes: Option<usize>,
    /// Explicit dedicated keys or an exact reachable reuse pool.
    pub transpose: TransposeKeys,
    /// Exact evaluated flags; must match the sealed registration's policy.
    pub eval_config: EvalConfig,
    /// Optional exact partitions, one entry per consumed update.
    pub partitions: Option<Vec<Vec<MicroPart>>>,
    /// Explicit payload/index cap, not an estimated memory fit.
    pub limits: ReceiptLimits,
    /// Shared scoped pure-work cap. No key/HE factory is held here.
    pub work_budget: Arc<PlanningBudget>,
}
impl DiagonalRequest {
    /// A two-update request with dedicated transpose and all optimizations enabled.
    /// Callers should replace the permissive work cap with their operational limits.
    pub fn quick() -> Self {
        Self {
            consumption: Consumption::Quick,
            physical_lanes: None,
            transpose: TransposeKeys::Dedicated,
            eval_config: EvalConfig::OPTIMIZED,
            partitions: None,
            limits: ReceiptLimits::default(),
            work_budget: Arc::new(PlanningBudget::new(PlanningUsage {
                geometries: usize::MAX,
                initial_states: usize::MAX,
                updates: usize::MAX,
                handoffs: usize::MAX,
                wrapper_calls: usize::MAX,
            })),
        }
    }
}
/// Caller-owned immutable CKKS values; this does not construct parameters or keys.
#[derive(Clone, Copy)]
pub struct PlanningParameters<'a> {
    /// Actual complete parameter object.
    pub params: &'a CkksParams,
    /// Existing P1 conditional bootstrap descriptor, NI only.
    pub bootstrap: Option<&'a BootstrapMetadata>,
    /// Concrete bootstrap parameters used by that descriptor, NI only.
    pub boot_params: Option<&'a BootParams>,
}
/// Independent D6a input state of a metadata-only candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputStatus {
    /// Metadata does not verify any dataset bytes or prepare arrays.
    Unverified,
    /// The selected captured initialization has no registered export.
    MissingCaptured,
}
/// Gates deliberately not established by a metadata plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnprovedGate {
    /// Actual key allocation, workspace liveness and host/device capacity.
    Resource,
    /// Encrypted input/encoding and internal bootstrap magnitude ranges.
    Range,
    /// Approximation/decryption precision.
    Precision,
    /// Security estimate and sampled-secret contract.
    Security,
    /// Native implementation/backend and retained runtime capture.
    Runtime,
    /// Actual F32 reference and target lock.
    Target,
    /// Applicable user campaign decisions, including encrypted block holds.
    UserDecision,
    /// Reachable P4/D7 factory spy test and actual durable start/gates.
    PreKeyFactoryIntegration,
}
const UNPROVED: [UnprovedGate; 8] = [
    UnprovedGate::Resource,
    UnprovedGate::Range,
    UnprovedGate::Precision,
    UnprovedGate::Security,
    UnprovedGate::Runtime,
    UnprovedGate::Target,
    UnprovedGate::UserDecision,
    UnprovedGate::PreKeyFactoryIntegration,
];
/// Typed binding failure; no failure discards its expected inventory row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueKind {
    /// Request disagrees with a sealed registration or materialized selection.
    InvalidAxis,
    /// Malformed graph, dimensions or representation source.
    InvalidDimensions,
    /// Checked count/index/length arithmetic overflow.
    ArithmeticOverflow,
    /// Empty, overlapping, out-of-order or nonconserving logical partition.
    InvalidMicroPartition,
    /// Nonfinite, nonpositive or inconsistent authoritative coefficient/reference.
    RateConflict,
    /// Parameters, descriptor, canonical input or bound receipt identity conflict.
    IdentityMismatch,
    /// A registered capability has no reviewed implementation here.
    MissingCode,
    /// Operational byte/index/work cap exceeded; not insufficient CKKS depth.
    PlanningBudgetExceeded,
    /// A fallible metadata reservation failed, independently from the configured cap.
    AllocationFailed,
    /// Existing P2 returned its preserved typed operation failure.
    StagePlanning,
}
/// A failure with exact P2 cause, completed prefix and inspectable partial transcript.
#[derive(Debug)]
pub struct AdmissionIssue {
    kind: IssueKind,
    operation: &'static str,
    stage: Option<usize>,
    update: Option<usize>,
    completed: usize,
    source: Option<Box<PlanError>>,
    prefix: Vec<u8>,
    prefix_index: Box<[TransitionIndex]>,
    budget: Option<Box<(usize, usize)>>,
}
impl AdmissionIssue {
    /// Broad binding category; inspect the P2 source for exact level/scale failures.
    pub fn kind(&self) -> IssueKind {
        self.kind
    }
    /// Named checked boundary.
    pub fn operation(&self) -> &'static str {
        self.operation
    }
    /// Stage associated with this failure, if any.
    pub fn stage(&self) -> Option<usize> {
        self.stage
    }
    /// Consumed update associated with this failure, if any.
    pub fn update(&self) -> Option<usize> {
        self.update
    }
    /// Completely composed logical updates, never a successful substitute plan.
    pub fn completed_updates(&self) -> usize {
        self.completed
    }
    /// Full typed P2 diagnostic, including attempts and actual-prime context.
    pub fn stage_error(&self) -> Option<&PlanError> {
        self.source.as_deref()
    }
    /// Completed bounded transcript before failure, not a complete plan.
    pub fn completed_prefix_bytes(&self) -> &[u8] {
        &self.prefix
    }
    /// Complete retained prefix records with their exact typed summaries.
    pub fn completed_prefix_transitions(&self) -> &[TransitionIndex] {
        &self.prefix_index
    }
    /// Actual required and configured limit for a byte/index/work cap failure.
    pub fn budget_requirement(&self) -> Option<(usize, usize)> {
        self.budget.as_deref().copied()
    }
}
impl fmt::Display for AdmissionIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.operation)
    }
}
impl std::error::Error for AdmissionIssue {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}
type Result<T> = std::result::Result<T, AdmissionIssue>;
fn issue(kind: IssueKind, operation: &'static str) -> AdmissionIssue {
    AdmissionIssue {
        kind,
        operation,
        stage: None,
        update: None,
        completed: 0,
        source: None,
        prefix: vec![],
        prefix_index: Box::default(),
        budget: None,
    }
}
fn require(ok: bool, kind: IssueKind, operation: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(issue(kind, operation))
    }
}
fn budget_error(operation: &'static str, required: usize, limit: usize) -> AdmissionIssue {
    let mut error = issue(IssueKind::PlanningBudgetExceeded, operation);
    error.budget = Some(Box::new((required, limit)));
    error
}
fn p2(error: Box<PlanError>, stage: usize) -> AdmissionIssue {
    let update = error.update;
    AdmissionIssue {
        kind: IssueKind::StagePlanning,
        operation: "P2 transition",
        stage: Some(stage),
        update,
        completed: 0,
        source: Some(error),
        prefix: vec![],
        prefix_index: Box::default(),
        budget: None,
    }
}
/// Logical training and bootstrap key identities; physical bundles stay separate.
#[derive(Debug)]
pub struct KeyManifest {
    training: Vec<i32>,
    bootstrap_generated: Vec<i32>,
    bootstrap_execution: Vec<i32>,
    conjugation: bool,
    sse: bool,
}
impl KeyManifest {
    /// Union of normalized nonzero stage training rotations, not physical allocation count.
    pub fn training_rotations(&self) -> &[i32] {
        &self.training
    }
    /// Full separate bootstrap bundle's generated rotations.
    pub fn bootstrap_generated_rotations(&self) -> &[i32] {
        &self.bootstrap_generated
    }
    /// Bootstrap execution support, distinct from its generated key support.
    pub fn bootstrap_execution_rotations(&self) -> &[i32] {
        &self.bootstrap_execution
    }
    /// Whether the separate bootstrap bundle requires conjugation.
    pub fn bootstrap_conjugation(&self) -> bool {
        self.conjugation
    }
    /// Whether it requires both SSE switching roles.
    pub fn bootstrap_sse(&self) -> bool {
        self.sse
    }
    /// Sharing/merging physical key bundles is never assumed.
    pub fn separate_bootstrap_bundle(&self) -> bool {
        !self.bootstrap_generated.is_empty()
    }
}
/// Dataset-independent description; cannot be passed to a runtime/session constructor.
#[derive(Debug)]
pub struct CandidateDescription {
    geometry_digest: String,
    geometries: Vec<StageGeometry>,
    states: Vec<StageState>,
    keys: Option<KeyManifest>,
    input: InputStatus,
    issue: Option<AdmissionIssue>,
}
impl CandidateDescription {
    /// Seed-independent metadata geometry identity; empty on malformed geometry.
    pub fn geometry_digest(&self) -> &str {
        &self.geometry_digest
    }
    /// Complete checked stage geometries when available.
    pub fn geometries(&self) -> &[StageGeometry] {
        &self.geometries
    }
    /// Descriptive fresh labels; placeholder rates are not schedule admission.
    pub fn initial_states(&self) -> &[StageState] {
        &self.states
    }
    /// Separate exact logical key role manifest, if geometry/parameters could be checked.
    pub fn keys(&self) -> Option<&KeyManifest> {
        self.keys.as_ref()
    }
    /// Independent input status; success never means dataset prepared.
    pub fn input_status(&self) -> InputStatus {
        self.input
    }
    /// Typed failure retained in this expected row.
    pub fn issue(&self) -> Option<&AdmissionIssue> {
        self.issue.as_ref()
    }
    /// All remaining independent admission gates.
    pub fn unproved(&self) -> &[UnprovedGate] {
        &UNPROVED
    }
}
/// Metadata-only input status, without preflight, dataset import or fallback.
pub fn metadata_input_status(metadata: &ProfileMetadata) -> InputStatus {
    if matches!(metadata.initialization, Initialization::MissingCaptured) {
        InputStatus::MissingCaptured
    } else {
        InputStatus::Unverified
    }
}

// One checked append stream. A failed scalar append leaves no silently accepted plan.
struct Transcript {
    bytes: Vec<u8>,
    limits: ReceiptLimits,
    metadata_bytes: usize,
}
impl Transcript {
    fn new(limits: ReceiptLimits) -> Self {
        Self {
            bytes: vec![],
            limits,
            metadata_bytes: 0,
        }
    }
    fn metadata(&mut self, amount: usize) -> Result<()> {
        let next = self
            .metadata_bytes
            .checked_add(amount)
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "metadata payload"))?;
        let combined = self
            .bytes
            .len()
            .checked_add(next)
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "combined payload"))?;
        if combined > self.limits.max_bytes {
            return Err(budget_error(
                "combined transcript/metadata bytes",
                combined,
                self.limits.max_bytes,
            ));
        }
        self.metadata_bytes = next;
        Ok(())
    }
    fn raw(&mut self, value: &[u8]) -> Result<()> {
        let next = self
            .bytes
            .len()
            .checked_add(value.len())
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "transcript length"))?;
        let combined = next
            .checked_add(self.metadata_bytes)
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "combined payload"))?;
        if combined > self.limits.max_bytes {
            return Err(budget_error(
                "combined transcript/metadata bytes",
                combined,
                self.limits.max_bytes,
            ));
        }
        self.bytes
            .try_reserve_exact(value.len())
            .map_err(|_| issue(IssueKind::AllocationFailed, "transcript reservation"))?;
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    fn tag(&mut self, value: u8) -> Result<()> {
        self.raw(&[value])
    }
    fn number(&mut self, value: usize) -> Result<()> {
        self.u64(
            u64::try_from(value).map_err(|_| issue(IssueKind::ArithmeticOverflow, "u64 index"))?,
        )
    }
    fn u64(&mut self, value: u64) -> Result<()> {
        self.raw(&value.to_le_bytes())
    }
    fn float(&mut self, value: f64) -> Result<()> {
        self.u64(value.to_bits())
    }
    fn blob(&mut self, value: &[u8]) -> Result<()> {
        self.number(value.len())?;
        self.raw(value)
    }
    fn text(&mut self, value: &str) -> Result<()> {
        self.blob(value.as_bytes())
    }
    fn option_number(&mut self, value: Option<usize>) -> Result<()> {
        self.tag(u8::from(value.is_some()))?;
        if let Some(n) = value {
            self.number(n)?;
        }
        Ok(())
    }
    fn option_bits(&mut self, value: Option<u64>) -> Result<()> {
        self.tag(u8::from(value.is_some()))?;
        if let Some(n) = value {
            self.u64(n)?;
        }
        Ok(())
    }
    fn bool_option(&mut self, value: Option<bool>) -> Result<()> {
        self.tag(u8::from(value.is_some()))?;
        if let Some(n) = value {
            self.tag(u8::from(n))?;
        }
        Ok(())
    }
}
fn payload<T>(count: usize) -> Result<usize> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "metadata element payload"))
}
fn sum_payload(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "metadata payload sum"))
    })
}
fn copy_vec<T: Clone>(values: &[T]) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len())
        .map_err(|_| issue(IssueKind::AllocationFailed, "metadata vector reservation"))?;
    out.extend_from_slice(values);
    Ok(out)
}
fn matrix_payload(values: &[Vec<CtMeta>]) -> Result<usize> {
    let mut bytes = payload::<Vec<CtMeta>>(values.len())?;
    for row in values {
        bytes = sum_payload([bytes, payload::<CtMeta>(row.len())?])?;
    }
    Ok(bytes)
}
fn copy_matrix(values: &[Vec<CtMeta>]) -> Result<Vec<Vec<CtMeta>>> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len())
        .map_err(|_| issue(IssueKind::AllocationFailed, "metadata matrix reservation"))?;
    for row in values {
        out.push(copy_vec(row)?);
    }
    Ok(out)
}
fn state_payload(s: &StageState) -> Result<usize> {
    StateLabels::from_state(s).payload()
}
#[derive(Clone, Copy)]
struct StateLabels<'a> {
    weights: &'a [CtMeta],
    biases: &'a [CtMeta],
    compact: &'a [CtMeta],
    velocity: &'a [Option<CtMeta>],
    batch: BatchMetadata,
    optimizer: DiagonalOptimizer,
    updates: usize,
}
impl<'a> StateLabels<'a> {
    fn from_state(s: &'a StageState) -> Self {
        Self {
            weights: s.weights(),
            biases: s.biases(),
            compact: s.compact_weights(),
            velocity: s.velocity(),
            batch: s.batch(),
            optimizer: s.optimizer(),
            updates: s.completed_updates(),
        }
    }
    fn payload(self) -> Result<usize> {
        sum_payload([
            payload::<CtMeta>(self.weights.len())?,
            payload::<CtMeta>(self.biases.len())?,
            payload::<CtMeta>(self.compact.len())?,
            payload::<Option<CtMeta>>(self.velocity.len())?,
        ])
    }
    fn retain(self) -> Result<StateMetadata> {
        Ok(StateMetadata {
            weights: copy_vec(self.weights)?,
            biases: copy_vec(self.biases)?,
            compact: copy_vec(self.compact)?,
            velocity: copy_vec(self.velocity)?,
            batch: self.batch,
            optimizer: self.optimizer,
            updates: self.updates,
        })
    }
}
struct UpdateSummarySource<'a> {
    before: StateLabels<'a>,
    after: StateLabels<'a>,
    inputs: &'a [MicroMetadata],
    hidden: &'a [Vec<CtMeta>],
    canonical: &'a [Vec<CtMeta>],
    coefficients: &'a GradientCoefficients,
    reference: bool,
    first: bool,
    booted: bool,
    nonbootstrap: &'a OpCounter,
    inclusive: &'a OpCounter,
    refreshes: usize,
    bootstrap_count: usize,
}
fn update_summary_payload(s: &UpdateSummarySource<'_>) -> Result<usize> {
    sum_payload([
        s.before.payload()?,
        s.after.payload()?,
        payload::<MicroMetadata>(s.inputs.len())?,
        matrix_payload(s.hidden)?,
        matrix_payload(s.canonical)?,
    ])
}
fn retain_update_summary(w: &mut Transcript, s: UpdateSummarySource<'_>) -> Result<UpdateMetadata> {
    w.metadata(update_summary_payload(&s)?)?;
    Ok(UpdateMetadata {
        before: s.before.retain()?,
        after: s.after.retain()?,
        inputs: copy_vec(s.inputs)?,
        hidden: copy_matrix(s.hidden)?,
        canonical: copy_matrix(s.canonical)?,
        coefficients: s.coefficients.clone(),
        reference: s.reference,
        first: s.first,
        booted: s.booted,
        nonbootstrap: s.nonbootstrap.clone(),
        inclusive: s.inclusive.clone(),
        refreshes: s.refreshes,
        bootstrap_count: s.bootstrap_count,
    })
}
fn retain_handoff_summary(w: &mut Transcript, h: &PlannedHandoff) -> Result<HandoffMetadata> {
    w.metadata(payload::<CtMeta>(h.outputs.len())?)?;
    Ok(HandoffMetadata {
        outputs: copy_vec(&h.outputs)?,
        nonbootstrap: h.counts_nonbootstrap.clone(),
        inclusive: h.counts_including_bootstrap.clone(),
        refreshes: h.client_refreshes,
        bootstrap_count: h.bootstraps.len(),
    })
}
fn role(w: &mut Transcript, r: LayerRole) -> Result<()> {
    match r {
        LayerRole::Hidden => w.tag(0),
        LayerRole::Output => w.tag(1),
        LayerRole::LocalForward { block } => {
            w.tag(2)?;
            w.number(block)
        }
        LayerRole::LocalHead { block } => {
            w.tag(3)?;
            w.number(block)
        }
        LayerRole::OutputHidden => w.tag(4),
    }
}
fn activation(a: Activation) -> DiagonalActivation {
    match a {
        Activation::Square => DiagonalActivation::Square,
        Activation::SquarePlusLinear => DiagonalActivation::PolyRelu2,
        Activation::Identity => DiagonalActivation::Identity,
    }
}
fn model_stages(m: Model) -> usize {
    match m {
        Model::Mlp2020 | Model::Emlp1 => 1,
        Model::Emlp2 => 2,
        Model::Emlp3 => 3,
    }
}
fn metadata_geometry(w: &mut Transcript, m: &ProfileMetadata) -> Result<()> {
    require(
        m.layers.len() == 2 * model_stages(m.model),
        IssueKind::InvalidDimensions,
        "stage count",
    )?;
    let expected_width = if m.model == Model::Mlp2020 {
        m.classes
    } else {
        m.classes
            .checked_next_power_of_two()
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "class padding"))?
    };
    require(
        m.classes >= 2 && m.output_width == expected_width,
        IssueKind::InvalidDimensions,
        "target width",
    )?;
    w.text(&m.id)?;
    w.text(&m.dataset)?;
    w.number(m.classes)?;
    w.number(m.output_width)?;
    w.tag(match m.model {
        Model::Mlp2020 => 0,
        Model::Emlp1 => 1,
        Model::Emlp2 => 2,
        Model::Emlp3 => 3,
    })?;
    w.number(m.layers.len())?;
    let mut names = BTreeSet::new();
    let blocks = model_stages(m.model) - 1;
    for (i, pair) in m.layers.chunks_exact(2).enumerate() {
        let (a, b) = (&pair[0], &pair[1]);
        require(
            a.input > 0
                && a.output > 0
                && b.input == a.output
                && b.output == m.output_width
                && (i == 0 || a.input == m.layers[2 * (i - 1)].output),
            IssueKind::InvalidDimensions,
            "stage connection",
        )?;
        let ra = if i < blocks {
            LayerRole::LocalForward { block: i }
        } else if blocks > 0 {
            LayerRole::OutputHidden
        } else {
            LayerRole::Hidden
        };
        let rb = if i < blocks {
            LayerRole::LocalHead { block: i }
        } else {
            LayerRole::Output
        };
        require(
            a.role == ra && b.role == rb,
            IssueKind::InvalidDimensions,
            "old-hidden stage roles",
        )?;
        require(
            a.activation
                == if m.model == Model::Mlp2020 {
                    Activation::Square
                } else {
                    Activation::SquarePlusLinear
                }
                && b.activation
                    == if m.model == Model::Mlp2020 {
                        Activation::Square
                    } else {
                        Activation::Identity
                    }
                && a.bias == (m.model == Model::Mlp2020)
                && b.bias == a.bias,
            IssueKind::InvalidDimensions,
            "activation/bias contract",
        )?;
        for layer in pair {
            require(
                !layer.name.is_empty() && names.insert(&layer.name),
                IssueKind::InvalidDimensions,
                "layer names",
            )?;
            layer
                .input
                .checked_next_power_of_two()
                .and_then(|_| layer.output.checked_next_power_of_two())
                .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "dimension padding"))?;
            w.text(&layer.name)?;
            w.number(layer.input)?;
            w.number(layer.output)?;
            w.tag(match layer.activation {
                Activation::Square => 0,
                Activation::SquarePlusLinear => 1,
                Activation::Identity => 2,
            })?;
            w.tag(u8::from(layer.bias))?;
            role(w, layer.role)?;
        }
    }
    Ok(())
}
/// Seed-independent checked geometry identity; input readiness is deliberately excluded.
pub fn geometry_identity(metadata: &ProfileMetadata) -> Result<String> {
    let mut w = Transcript::new(ReceiptLimits::default());
    w.text("hekit-study-diagonal-geometry-v1")?;
    metadata_geometry(&mut w, metadata)?;
    Ok(sha256_hex(&w.bytes))
}
fn checked_mode(
    m: &ProfileMetadata,
    c: &RegisteredCase,
    r: &DiagonalRequest,
) -> Result<DiagonalMode> {
    let s = c.selector();
    require(
        s.scope == CaseScope::Full
            && s.seed == u64::from(m.seed)
            && s.profile_id == m.id
            && c.logical_batch() == m.batch as u64,
        IssueKind::InvalidAxis,
        "registered profile/seed/batch",
    )?;
    require(
        c.role() == CaseRole::Main
            && s.packing == CasePacking::BatchedDiagonal
            && matches!(s.backend, CaseBackend::Cpu | CaseBackend::Cuda),
        IssueKind::MissingCode,
        "non-main diagonal capability",
    )?;
    let mode = match (s.mode, s.placement, c.security_policy()) {
        (CaseMode::Interactive, CasePlacement::ClientRefresh, SecurityPolicy::N16Train) => {
            DiagonalMode::Interactive
        }
        (CaseMode::Noninteractive, CasePlacement::WhenNeeded, SecurityPolicy::N16Boot) => {
            DiagonalMode::NonInteractive(BootPlacement::WhenNeeded)
        }
        (CaseMode::Noninteractive, CasePlacement::AfterUpdate, SecurityPolicy::N16Boot) => {
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate)
        }
        (CaseMode::Noninteractive, CasePlacement::AfterActivation, SecurityPolicy::N16Boot) => {
            DiagonalMode::NonInteractive(BootPlacement::AfterActivation)
        }
        _ => return Err(issue(IssueKind::InvalidAxis, "mode/placement/security")),
    };
    let cfg = c
        .eval_config()
        .map_or(EvalConfig::OPTIMIZED, |v| EvalConfig {
            hoist: v.hoist,
            bsgs: v.bsgs,
            lazy_relin: v.lazy_relin,
        });
    require(
        cfg == r.eval_config,
        IssueKind::InvalidAxis,
        "unregistered EvalConfig",
    )?;
    Ok(mode)
}
fn secure_parameters(p: PlanningParameters<'_>, mode: DiagonalMode) -> Result<()> {
    let q = p.params;
    // CkksParams is sealed: its safe constructor deterministically generates actual
    // q/p from these exact bit vectors; no arbitrary-prime constructor is exposed.
    let (qb, pb, dnum) = if mode == DiagonalMode::Interactive {
        let mut a = vec![58];
        a.extend(vec![42; ckks_train_ckks::params::N16_TRAIN_L]);
        (
            a,
            vec![58; ckks_train_ckks::params::N16_TRAIN_K],
            ckks_train_ckks::params::N16_TRAIN_DNUM,
        )
    } else {
        let (a, b) = ckks_train_ckks::params::n16_boot_bits();
        (a, b, ckks_train_ckks::params::N16_BOOT_DNUM)
    };
    require(
        q.n() == 1 << 16
            && q.q_bits() == qb
            && q.p_bits() == pb
            && q.scale().to_bits() == (2f64.powi(42)).to_bits()
            && q.dnum() == dnum
            && q.hamming_weight().is_none(),
        IssueKind::IdentityMismatch,
        "registered secure preset",
    )?;
    if mode == DiagonalMode::Interactive {
        require(
            p.bootstrap.is_none() && p.boot_params.is_none(),
            IssueKind::IdentityMismatch,
            "I bootstrap absence",
        )
    } else {
        require(
            p.boot_params == Some(&BootParams::default_n16()),
            IssueKind::IdentityMismatch,
            "default_n16 BootParams",
        )?;
        let descriptor = p
            .bootstrap
            .ok_or_else(|| issue(IssueKind::MissingCode, "P1 bootstrap descriptor"))?;
        require(
            p.boot_params == Some(descriptor.boot_params()),
            IssueKind::IdentityMismatch,
            "actual descriptor BootParams",
        )
    }
}
fn encode_params(w: &mut Transcript, p: PlanningParameters<'_>) -> Result<()> {
    let q = p.params;
    w.number(q.n())?;
    w.number(q.q().len())?;
    for &n in q.q() {
        w.u64(n)?;
    }
    w.number(q.p().len())?;
    for &n in q.p() {
        w.u64(n)?;
    }
    w.float(q.scale())?;
    w.number(q.dnum())?;
    w.option_number(q.hamming_weight())?;
    w.tag(u8::from(p.boot_params.is_some()))?;
    if let Some(b) = p.boot_params {
        w.u64(u64::from(b.log_slots))?;
        for n in [
            b.ctos_levels,
            b.evalmod_levels,
            b.stoc_levels,
            b.k_intervals,
            b.cheb_degree,
            b.double_angle,
        ] {
            w.number(n)?;
        }
        w.option_number(b.sparse_hw)?;
        w.option_number(b.ephemeral_hw)?;
    }
    w.tag(u8::from(p.bootstrap.is_some()))?;
    if let Some(b) = p.bootstrap {
        w.blob(b.parameter_identity_bytes())?;
        w.blob(b.descriptor_identity_bytes())?;
        w.u64(u64::from(b.schema_version()))?;
        w.u64(u64::from(b.algorithm_revision()))?;
        w.tag(0)?; // ProofScope::StructuralCostAndBoundaryScales, only current variant.
        w.number(b.unvalidated_requirements().len())?;
        for u in b.unvalidated_requirements() {
            w.tag(match u {
                UnvalidatedRequirement::EncodingQOverTwoFeasibility => 0,
                UnvalidatedRequirement::EvalModInternalScaleUnvalidated => 1,
                UnvalidatedRequirement::EncryptedInputRange => 2,
                UnvalidatedRequirement::ApproximationPrecision => 3,
                UnvalidatedRequirement::ActualSecretContract => 4,
                UnvalidatedRequirement::KeyBinding => 5,
                UnvalidatedRequirement::SecurityEstimate => 6,
                UnvalidatedRequirement::RuntimeMemoryFit => 7,
                UnvalidatedRequirement::BackendSupport => 8,
            })?;
        }
        let proof = b.dft_support_proof();
        w.u64(proof.root_count())?;
        for v in [proof.configured_compiler(), proof.target(), proof.host()] {
            w.tag(u8::from(v.is_some()))?;
            if let Some(v) = v {
                w.text(v)?;
            }
        }
        let (a, c) = proof.root_squared_norm_bounds();
        let (d, e) = proof.inverse_squared_norm_bounds();
        for n in [a, c, d, e] {
            w.float(n)?;
        }
        let resource = b.resource_bounds();
        for n in [
            resource.coefficients(),
            resource.interpolation_terms(),
            resource.polynomial_work(),
            resource.requested_bytes(),
        ] {
            w.u64(n)?;
        }
        w.number(resource.recursion_depth())?;
        let keys = b.generated_keys();
        w.blob(keys.parameter_identity_bytes())?;
        for n in [keys.galois(), keys.relin()] {
            key_role(w, n)?;
        }
        for k in [keys.conjugation(), keys.sse_to_sparse(), keys.sse_to_main()] {
            w.tag(u8::from(k.is_some()))?;
            if let Some(k) = k {
                key_role(w, k)?;
            }
        }
        int_steps(w, keys.rotation_steps())?;
        int_steps(w, b.execution_rotation_steps())?;
    }
    Ok(())
}
fn key_role(w: &mut Transcript, k: &KeyRoleMetadata) -> Result<()> {
    for n in [
        k.n(),
        k.q_prefix_len(),
        k.p_len(),
        k.level(),
        k.digits(),
        k.alpha(),
    ] {
        w.number(n)?;
    }
    for s in [k.source(), k.destination()] {
        match s {
            ckks_train_ckks::bootstrap::metadata::SecretDeclaration::DenseTernary => w.tag(0)?,
            ckks_train_ckks::bootstrap::metadata::SecretDeclaration::FixedWeight(n) => {
                w.tag(1)?;
                w.number(n)?;
            }
        }
    }
    Ok(())
}
fn int_steps(w: &mut Transcript, steps: &[i32]) -> Result<()> {
    w.number(steps.len())?;
    for &s in steps {
        w.raw(&s.to_le_bytes())?;
    }
    Ok(())
}
fn encode_request(w: &mut Transcript, r: &DiagonalRequest) -> Result<()> {
    w.tag(match r.consumption {
        Consumption::Full => 0,
        Consumption::Quick => 1,
    })?;
    w.option_number(r.physical_lanes)?;
    match &r.transpose {
        TransposeKeys::Dedicated => w.tag(0)?,
        TransposeKeys::Reuse(pool) => {
            w.tag(1)?;
            int_steps(w, pool)?;
        }
    }
    for n in [
        r.eval_config.hoist,
        r.eval_config.bsgs,
        r.eval_config.lazy_relin,
    ] {
        w.tag(u8::from(n))?;
    }
    w.number(r.limits.max_bytes)?;
    w.number(r.limits.max_records)?;
    for n in [
        r.work_budget.limits.geometries,
        r.work_budget.limits.initial_states,
        r.work_budget.limits.updates,
        r.work_budget.limits.handoffs,
        r.work_budget.limits.wrapper_calls,
    ] {
        w.number(n)?;
    }
    Ok(())
}
fn make_stages(
    m: &ProfileMetadata,
    p: PlanningParameters<'_>,
    r: &DiagonalRequest,
    mode: DiagonalMode,
    lr: f64,
    reference: f64,
) -> Result<(Vec<StageGeometry>, Vec<StageState>)> {
    geometry_identity(m)?;
    let d = m.layers.iter().try_fold(1usize, |d, l| {
        let a = l
            .input
            .checked_next_power_of_two()
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "input padding"))?;
        let b = l
            .output
            .checked_next_power_of_two()
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "output padding"))?;
        Ok::<_, AdmissionIssue>(d.max(a).max(b))
    })?;
    let lanes = r.physical_lanes.unwrap_or(
        m.batch
            .checked_next_power_of_two()
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "lane padding"))?,
    );
    let layout = Layout::new(p.params.slots(), d, lanes)
        .map_err(|_| issue(IssueKind::InvalidDimensions, "common layout"))?;
    let mut geometries = vec![];
    let mut states = vec![];
    for (s, l) in m.layers.chunks_exact(2).enumerate() {
        r.work_budget.charge(Work::Geometry)?;
        let g = plan_stage_geometry(
            &[l[0].input, l[0].output, l[1].output],
            &layout,
            &[activation(l[0].activation), activation(l[1].activation)],
            &r.transpose,
            mode,
            r.eval_config,
        )
        .map_err(|e| p2(e, s))?;
        let optim = match m.optimizer {
            Optimizer::Sgd => DiagonalOptimizer::Sgd { lr },
            Optimizer::Nesterov { momentum } => DiagonalOptimizer::Nag {
                lr,
                reference,
                momentum,
            },
        };
        r.work_budget.charge(Work::Initial)?;
        let state = initial_stage_metadata(
            &g,
            p.params,
            p.bootstrap,
            optim,
            if l[0].bias {
                BiasPolicy::Active
            } else {
                BiasPolicy::Frozen
            },
        )
        .map_err(|e| p2(e, s))?;
        geometries.push(g);
        states.push(state);
    }
    Ok((geometries, states))
}
fn manifest(geometries: &[StageGeometry], p: PlanningParameters<'_>) -> KeyManifest {
    let training = geometries
        .iter()
        .flat_map(|g| g.rotation_steps().iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    KeyManifest {
        training,
        bootstrap_generated: p
            .bootstrap
            .map_or(vec![], |b| b.generated_keys().rotation_steps().to_vec()),
        bootstrap_execution: p
            .bootstrap
            .map_or(vec![], |b| b.execution_rotation_steps().to_vec()),
        conjugation: p
            .bootstrap
            .is_some_and(|b| b.generated_keys().conjugation().is_some()),
        sse: p.bootstrap.is_some_and(|b| {
            b.generated_keys().sse_to_sparse().is_some()
                && b.generated_keys().sse_to_main().is_some()
        }),
    }
}
fn bounded_manifest(
    g: &[StageGeometry],
    p: PlanningParameters<'_>,
    w: &mut Transcript,
) -> Result<KeyManifest> {
    let union = g
        .iter()
        .flat_map(|g| g.rotation_steps().iter().copied())
        .collect::<BTreeSet<_>>();
    let generated = p
        .bootstrap
        .map_or(&[][..], |b| b.generated_keys().rotation_steps());
    let execution = p
        .bootstrap
        .map_or(&[][..], |b| b.execution_rotation_steps());
    w.metadata(sum_payload([
        payload::<i32>(union.len())?,
        payload::<i32>(generated.len())?,
        payload::<i32>(execution.len())?,
    ])?)?;
    let mut training = Vec::new();
    training
        .try_reserve_exact(union.len())
        .map_err(|_| issue(IssueKind::AllocationFailed, "key manifest reservation"))?;
    training.extend(union);
    Ok(KeyManifest {
        training,
        bootstrap_generated: copy_vec(generated)?,
        bootstrap_execution: copy_vec(execution)?,
        conjugation: p
            .bootstrap
            .is_some_and(|b| b.generated_keys().conjugation().is_some()),
        sse: p.bootstrap.is_some_and(|b| {
            b.generated_keys().sse_to_sparse().is_some()
                && b.generated_keys().sse_to_main().is_some()
        }),
    })
}
/// Describes one expected main diagonal candidate without verifying datasets or running updates.
/// Every caller retains this returned row even if its typed issue is present.
pub fn describe_candidate(
    m: &ProfileMetadata,
    c: &RegisteredCase,
    p: PlanningParameters<'_>,
    r: &DiagonalRequest,
) -> CandidateDescription {
    let mut row = CandidateDescription {
        geometry_digest: String::new(),
        geometries: vec![],
        states: vec![],
        keys: None,
        input: metadata_input_status(m),
        issue: None,
    };
    let result: Result<()> = (|| {
        let mode = checked_mode(m, c, r)?;
        secure_parameters(p, mode)?;
        row.geometry_digest = geometry_identity(m)?;
        let (g, s) = make_stages(m, p, r, mode, 0.08, 0.08)?;
        row.keys = Some(manifest(&g, p));
        row.geometries = g;
        row.states = s;
        Ok(())
    })();
    row.issue = result.err();
    row
}

/// One indexed immutable stage-update or boundary receipt.
#[derive(Clone, Debug)]
pub struct TransitionIndex {
    update: usize,
    stage: usize,
    boundary: bool,
    start: usize,
    end: usize,
    next_full: Option<bool>,
    update_metadata: Option<UpdateMetadata>,
    handoff_metadata: Option<HandoffMetadata>,
}
impl TransitionIndex {
    /// Zero-based consumed update.
    pub fn update(&self) -> usize {
        self.update
    }
    /// Zero-based source stage.
    pub fn stage(&self) -> usize {
        self.stage
    }
    /// True for a boundary receipt, false for an optimizer update.
    pub fn is_boundary(&self) -> bool {
        self.boundary
    }
    /// Lookahead passed to P2; terminal consumption always uses None.
    pub fn next_full(&self) -> Option<bool> {
        self.next_full
    }
    /// Exact typed update summary, without decoding or duplicating complete operation traces.
    pub fn update_metadata(&self) -> Option<&UpdateMetadata> {
        self.update_metadata.as_ref()
    }
    /// Exact typed boundary summary for a handoff entry.
    pub fn handoff_metadata(&self) -> Option<&HandoffMetadata> {
        self.handoff_metadata.as_ref()
    }
}
/// Immutable labels needed to compare one runtime state, without a P2/parameter clone.
#[derive(Clone, Debug)]
pub struct StateMetadata {
    weights: Vec<CtMeta>,
    biases: Vec<CtMeta>,
    compact: Vec<CtMeta>,
    velocity: Vec<Option<CtMeta>>,
    batch: BatchMetadata,
    optimizer: DiagonalOptimizer,
    updates: usize,
}
impl StateMetadata {
    fn from_state(s: &StageState) -> Result<Self> {
        StateLabels::from_state(s).retain()
    }
    /// One uniform expanded diagonal label per layer.
    pub fn weights(&self) -> &[CtMeta] {
        &self.weights
    }
    /// Extracted/refreshed bias labels.
    pub fn biases(&self) -> &[CtMeta] {
        &self.biases
    }
    /// NI compact-container labels.
    pub fn compact_weights(&self) -> &[CtMeta] {
        &self.compact
    }
    /// Persistent velocity presence and exact labels.
    pub fn velocity(&self) -> &[Option<CtMeta>] {
        &self.velocity
    }
    /// Fresh client labels belonging to this stage.
    pub fn batch(&self) -> BatchMetadata {
        self.batch
    }
    /// Current rate and fixed reference/momentum basis.
    pub fn optimizer(&self) -> DiagonalOptimizer {
        self.optimizer
    }
    /// Successful updates in this stage state.
    pub fn completed_updates(&self) -> usize {
        self.updates
    }
}
/// Immutable update comparison surface for a future separately gated CPU bridge.
#[derive(Clone, Debug)]
pub struct UpdateMetadata {
    before: StateMetadata,
    after: StateMetadata,
    inputs: Vec<MicroMetadata>,
    hidden: Vec<Vec<CtMeta>>,
    canonical: Vec<Vec<CtMeta>>,
    coefficients: GradientCoefficients,
    reference: bool,
    first: bool,
    booted: bool,
    nonbootstrap: OpCounter,
    inclusive: OpCounter,
    refreshes: usize,
    bootstrap_count: usize,
}
impl UpdateMetadata {
    /// State entering the actual update.
    pub fn pre_state(&self) -> &StateMetadata {
        &self.before
    }
    /// State after update/refresh/unpack/extraction.
    pub fn post_state(&self) -> &StateMetadata {
        &self.after
    }
    /// Exact ordered valid counts and stage-specific x/y labels passed to P2.
    pub fn micros(&self) -> &[MicroMetadata] {
        &self.inputs
    }
    /// Hidden labels computed from the old weights, indexed micro then hidden layer.
    pub fn retained_hidden(&self) -> &[Vec<CtMeta>] {
        &self.hidden
    }
    /// AA canonical labels suitable for exact boundary reuse.
    pub fn canonical_activations(&self) -> &[Vec<CtMeta>] {
        &self.canonical
    }
    /// Exact current/reference/bias/mu/kappa coefficient bits.
    pub fn coefficients(&self) -> &GradientCoefficients {
        &self.coefficients
    }
    /// Actual runtime-compatible current/reference branch.
    pub fn reference_branch(&self) -> bool {
        self.reference
    }
    /// Whether this step initialized absent NAG velocity.
    pub fn first_nag_update(&self) -> bool {
        self.first
    }
    /// Structural compact-state bootstrap decision.
    pub fn state_bootstrapped(&self) -> bool {
        self.booted
    }
    /// Predicted training work excluding bootstrap internals.
    pub fn counts_nonbootstrap(&self) -> &OpCounter {
        &self.nonbootstrap
    }
    /// Predicted work including parameter-specific bootstrap work once.
    pub fn counts_including_bootstrap(&self) -> &OpCounter {
        &self.inclusive
    }
    /// Client refresh calls, excluding separate handoff work.
    pub fn client_refreshes(&self) -> usize {
        self.refreshes
    }
    /// Number of state/activation bootstrap events; complete events remain in transcript.
    pub fn bootstrap_count(&self) -> usize {
        self.bootstrap_count
    }
}
/// Boundary outputs and counts, distinct from a stage's own bootstrap work.
#[derive(Clone, Debug)]
pub struct HandoffMetadata {
    outputs: Vec<CtMeta>,
    nonbootstrap: OpCounter,
    inclusive: OpCounter,
    refreshes: usize,
    bootstrap_count: usize,
}
impl HandoffMetadata {
    /// Canonical old-hidden inputs supplied unchanged to the next stage's UpdateSpec.
    pub fn outputs(&self) -> &[CtMeta] {
        &self.outputs
    }
    /// Conversion-only predicted work.
    pub fn counts_nonbootstrap(&self) -> &OpCounter {
        &self.nonbootstrap
    }
    /// Boundary predicted work including bootstrap internals once.
    pub fn counts_including_bootstrap(&self) -> &OpCounter {
        &self.inclusive
    }
    /// Interactive boundary refresh calls.
    pub fn client_refreshes(&self) -> usize {
        self.refreshes
    }
    /// Extra boundary bootstrap events; AA reuse has zero.
    pub fn bootstrap_count(&self) -> usize {
        self.bootstrap_count
    }
}
/// An immutable, complete canonical metadata plan. This type grants no Ready/start/session capability.
#[derive(Debug)]
pub struct DiagonalPlan {
    bytes: Vec<u8>,
    digest: String,
    binding: Vec<u8>,
    full: String,
    consumed: String,
    full_count: usize,
    consumed_count: usize,
    reference: u64,
    rates: Vec<RateCoefficients<f64>>,
    scope: Consumption,
    index: Vec<TransitionIndex>,
    keys: KeyManifest,
    partitions: Vec<Vec<MicroPart>>,
    initial: Vec<StateMetadata>,
    charges: ReceiptCharges,
}
/// Requested retained payload charged under the combined operational cap, not RSS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReceiptCharges {
    /// Complete canonical transcript bytes.
    pub transcript_bytes: usize,
    /// Typed index/nested labels, initial summaries, bindings, partitions, rates and keys.
    pub metadata_bytes: usize,
}
impl ReceiptCharges {
    /// Combined charge; checked during construction before every retained growth.
    pub fn combined(&self) -> usize {
        self.transcript_bytes + self.metadata_bytes
    }
}
impl DiagonalPlan {
    /// Complete typed operational transcript including all consumed state transitions.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// SHA256 of the exact complete transcript.
    pub fn digest(&self) -> &str {
        &self.digest
    }
    /// D6a full plan identity, unchanged by quick consumption.
    pub fn full_plan_sha256(&self) -> &str {
        &self.full
    }
    /// Domain-separated identity of the actual consumed prefix/partitions/lookahead.
    pub fn consumed_sha256(&self) -> &str {
        &self.consumed
    }
    /// Full authoritative update count, including unconsumed quick suffix.
    pub fn full_update_count(&self) -> usize {
        self.full_count
    }
    /// Number of intended updates, all of which have been planned on success.
    pub fn consumed_updates(&self) -> usize {
        self.consumed_count
    }
    /// Exact full-plan maximum kernel rate bits.
    pub fn reference_bits(&self) -> u64 {
        self.reference
    }
    /// Every source/mean/delta/kernel coefficient from the full immutable profile.
    pub fn full_rates(&self) -> &[RateCoefficients<f64>] {
        &self.rates
    }
    /// Consumption tag, separate from the sealed Full case identity.
    pub fn consumption(&self) -> Consumption {
        self.scope
    }
    /// Checked offsets of complete stage-update and boundary receipts.
    pub fn transitions(&self) -> &[TransitionIndex] {
        &self.index
    }
    /// Exact typed bytes for an indexed receipt; None outside the checked index.
    pub fn receipt_bytes(&self, index: usize) -> Option<&[u8]> {
        let r = self.index.get(index)?;
        self.bytes.get(r.start..r.end)
    }
    /// Logical roles only; not physical allocation count or bytes.
    pub fn keys(&self) -> &KeyManifest {
        &self.keys
    }
    /// Exact ordered partitions for every consumed logical update.
    pub fn partitions(&self) -> &[Vec<MicroPart>] {
        &self.partitions
    }
    /// Initial stage labels and optimizer basis before update zero.
    pub fn initial_states(&self) -> &[StateMetadata] {
        &self.initial
    }
    /// Actual requested payload charged under ReceiptLimits, separately from allocator/RSS.
    pub fn charges(&self) -> ReceiptCharges {
        self.charges
    }
    /// Independent required gates that remain unproved after metadata success.
    pub fn unproved(&self) -> &[UnprovedGate] {
        &UNPROVED
    }
}
fn partitions(counts: &[usize], lanes: usize, r: &DiagonalRequest) -> Result<Vec<Vec<MicroPart>>> {
    require(
        lanes.is_power_of_two(),
        IssueKind::InvalidDimensions,
        "physical lanes",
    )?;
    if let Some(parts) = &r.partitions {
        require(
            parts.len() == counts.len(),
            IssueKind::InvalidMicroPartition,
            "partition update count",
        )?;
    }
    let mut out = Vec::new();
    out.try_reserve_exact(counts.len())
        .map_err(|_| issue(IssueKind::AllocationFailed, "partition vector"))?;
    for (i, &total) in counts.iter().enumerate() {
        let parts = if let Some(p) = &r.partitions {
            copy_vec(&p[i])?
        } else {
            require(
                total > 0,
                IssueKind::InvalidMicroPartition,
                "empty logical update",
            )?;
            let mut parts = Vec::new();
            parts
                .try_reserve_exact(total.div_ceil(lanes))
                .map_err(|_| issue(IssueKind::AllocationFailed, "micro slices"))?;
            for start in (0..total).step_by(lanes) {
                parts.push(MicroPart {
                    start,
                    valid: lanes.min(total - start),
                });
            }
            parts
        };
        let mut end = 0usize;
        for p in &parts {
            require(
                p.start == end && p.valid > 0 && p.valid <= lanes,
                IssueKind::InvalidMicroPartition,
                "ordered contiguous micro slices",
            )?;
            end = end
                .checked_add(p.valid)
                .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "micro partition total"))?;
        }
        require(
            !parts.is_empty() && end == total,
            IssueKind::InvalidMicroPartition,
            "micro conservation",
        )?;
        out.push(parts);
    }
    Ok(out)
}
fn rates(w: &mut Transcript, rates: &[RateCoefficients<f64>], reference: f64) -> Result<()> {
    require(
        !rates.is_empty() && reference.is_finite() && reference > 0.,
        IssueKind::RateConflict,
        "reference",
    )?;
    let mut maximum = 0f64;
    w.number(rates.len())?;
    for r in rates {
        require(
            [r.source, r.mean, r.delta, r.kernel]
                .iter()
                .all(|x| x.is_finite() && *x > 0.)
                && (r.mean * r.delta).to_bits() == r.kernel.to_bits(),
            IssueKind::RateConflict,
            "authoritative kernel coefficient",
        )?;
        for v in [r.source, r.mean, r.delta, r.kernel] {
            w.float(v)?;
        }
        maximum = maximum.max(r.kernel);
    }
    require(
        maximum.to_bits() == reference.to_bits(),
        IssueKind::RateConflict,
        "full maximum reference",
    )?;
    w.float(reference)
}
fn encode_plan(w: &mut Transcript, p: &BatchPlan) -> Result<()> {
    w.number(p.updates.len())?;
    for u in &p.updates {
        for n in [u.index, u.processed_start, u.valid_samples] {
            w.number(n)?;
        }
        w.option_number(u.source_schedule_index)?;
        for v in [
            u.source_coefficient,
            u.mean_gradient_rate,
            u.delta_multiplier,
        ] {
            w.float(v)?;
        }
    }
    w.number(p.checkpoints.len())?;
    for c in &p.checkpoints {
        for n in [c.completed_updates, c.processed_samples] {
            w.number(n)?;
        }
        w.number(c.crossed_logical_boundaries.len())?;
        for &b in &c.crossed_logical_boundaries {
            w.number(b)?;
        }
        w.tag(u8::from(c.exact_logical_boundary))?;
        w.number(c.interval_updates)?;
        w.number(c.interval_samples)?;
    }
    w.number(p.processed_samples)?;
    w.number(p.dropped_samples)
}
fn bound_partitions(w: &mut Transcript, parts: &[Vec<MicroPart>], lanes: usize) -> Result<()> {
    w.number(parts.len())?;
    for (i, p) in parts.iter().enumerate() {
        w.number(i)?;
        w.number(p.len())?;
        for v in p {
            w.number(v.start)?;
            w.number(v.valid)?;
        }
        w.bool_option(
            parts
                .get(i + 1)
                .map(|next| next.iter().all(|x| x.valid == lanes)),
        )?;
    }
    Ok(())
}
fn canonical_binding(
    p: &MaterializedProfile,
    c: &RegisteredCase,
    params: PlanningParameters<'_>,
    r: &DiagonalRequest,
) -> Result<(Transcript, Vec<Vec<MicroPart>>, DiagonalMode, String)> {
    require(
        p.identity().canonical_prepared(),
        IssueKind::IdentityMismatch,
        "canonical preparation required",
    )?;
    let m = p.metadata();
    let mode = checked_mode(m, c, r)?;
    secure_parameters(params, mode)?;
    require(
        p.shared_selector().is_some(),
        IssueKind::IdentityMismatch,
        "canonical selector",
    )?;
    geometry_identity(m)?;
    for (i, s) in p.graph().stages().iter().enumerate() {
        require(
            s.layers() == (2 * i..2 * i + 2),
            IssueKind::IdentityMismatch,
            "canonical stage range",
        )?;
        require(
            s.input()
                == if i == 0 {
                    StageInput::Dataset
                } else {
                    StageInput::PreUpdateActivation { layer: 2 * (i - 1) }
                }
                && s.representation_layer() == (i + 1 < model_stages(m.model)).then_some(2 * i),
            IssueKind::IdentityMismatch,
            "canonical old-hidden source",
        )?;
    }
    require(
        p.graph().stages().len() == model_stages(m.model),
        IssueKind::IdentityMismatch,
        "canonical stage count",
    )?;
    let count = match r.consumption {
        Consumption::Full => p.plan().updates.len(),
        Consumption::Quick => 2,
    };
    require(
        count > 0 && count <= p.plan().updates.len(),
        IssueKind::InvalidAxis,
        "consumed count",
    )?;
    let lanes = r.physical_lanes.unwrap_or(
        m.batch
            .checked_next_power_of_two()
            .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "lane padding"))?,
    );
    let counts = p.plan().updates[..count]
        .iter()
        .map(|u| u.valid_samples)
        .collect::<Vec<_>>();
    require(
        lanes.is_power_of_two(),
        IssueKind::InvalidDimensions,
        "physical lanes",
    )?;
    let mut w = Transcript::new(r.limits);
    let mut partition_bytes = payload::<Vec<MicroPart>>(count)?;
    if let Some(v) = &r.partitions {
        require(
            v.len() == count,
            IssueKind::InvalidMicroPartition,
            "partition update count",
        )?;
    }
    for (i, &n) in counts.iter().enumerate() {
        let len = r
            .partitions
            .as_ref()
            .map_or(n.div_ceil(lanes), |v| v[i].len());
        partition_bytes = sum_payload([partition_bytes, payload::<MicroPart>(len)?])?;
    }
    w.metadata(partition_bytes)?;
    let parts = partitions(&counts, lanes, r)?;
    w.text("hekit-study-diagonal-plan-v1")?;
    w.u64(1)?;
    w.blob(c.canonical_bytes())?;
    w.text(c.digest())?;
    w.text(c.registry_sha256())?;
    let identity = p.identity();
    for s in [
        identity.digest(),
        identity.weights_sha256(),
        identity.biases_sha256(),
        identity.order_sha256(),
        identity.plan_sha256(),
        identity.f64_inputs_sha256(),
        identity.f32_inputs_sha256(),
        identity.dataset_sha256(),
        identity.split_sha256(),
        identity.preprocessing_sha256(),
        identity.code_build_sha256(),
    ] {
        w.text(s)?;
    }
    w.number(identity.sources().len())?;
    for s in identity.sources() {
        require(
            s.expected_sha256 == s.actual_sha256,
            IssueKind::IdentityMismatch,
            "verified source digest",
        )?;
        w.text(&s.path)?;
        w.text(&s.expected_sha256)?;
        w.text(&s.actual_sha256)?;
    }
    // Bind BOTH COMPLETE trusted identities, not transported hashes alone.
    let (shared, selected) = crate::study_reference::derive_identities(p)
        .map_err(|_| issue(IssueKind::IdentityMismatch, "canonical dual identity"))?;
    w.blob(shared.canonical_bytes())?;
    w.text(shared.digest())?;
    w.blob(selected.canonical_bytes())?;
    w.text(selected.digest())?;
    metadata_geometry(&mut w, m)?;
    w.tag(match m.loss {
        crate::study_profile::Loss::MeanOutputMse => 0,
        crate::study_profile::Loss::RssDelta => 1,
    })?;
    match m.optimizer {
        Optimizer::Sgd => w.tag(0)?,
        Optimizer::Nesterov { momentum } => {
            w.tag(1)?;
            w.float(momentum)?;
        }
    }
    w.float(m.weight_decay)?;
    encode_plan(&mut w, p.plan())?;
    let numeric = p.numeric_f64();
    require(
        numeric.rates().len() == p.plan().updates.len(),
        IssueKind::IdentityMismatch,
        "full rates cardinality",
    )?;
    rates(&mut w, numeric.rates(), p.reference_kernel_rate_f64())?;
    encode_request(&mut w, r)?;
    encode_params(&mut w, params)?;
    let mut consumed = Transcript::new(r.limits);
    consumed.text("hekit-study-diagonal-consumed-v1")?;
    consumed.text(identity.plan_sha256())?;
    bound_partitions(&mut consumed, &parts, lanes)?;
    for (i, &valid_count) in counts.iter().enumerate() {
        let rows = p
            .update_rows(i)
            .map_err(|_| issue(IssueKind::IdentityMismatch, "canonical update rows"))?;
        require(
            rows.planned().index == i && rows.rows().len() == valid_count,
            IssueKind::IdentityMismatch,
            "update row identity",
        )?;
        consumed.number(rows.rows().len())?;
        for &row in rows.rows() {
            consumed.u64(u64::from(row))?;
        }
        consumed.option_number(rows.pass())?;
        consumed.number(rows.crossed_boundaries().len())?;
        for &b in rows.crossed_boundaries() {
            consumed.number(b)?;
        }
    }
    w.metadata(64)?; // Retained consumed-digest String, charged before hashing allocates it.
    let digest = sha256_hex(&consumed.bytes);
    w.blob(&consumed.bytes)?;
    Ok((w, parts, mode, digest))
}

/// Plans the entire explicitly consumed schedule against one sealed Full registration.
/// Production requires D6a canonical preparation; success is never full admission/Ready.
pub fn plan_diagonal(
    p: &MaterializedProfile,
    c: &RegisteredCase,
    params: PlanningParameters<'_>,
    r: &DiagonalRequest,
) -> Result<DiagonalPlan> {
    r.work_budget.charge(Work::Wrapper)?;
    let (mut transcript, parts, mode, consumed) = canonical_binding(p, c, params, r)?;
    transcript.metadata(transcript.bytes.len())?;
    let binding = copy_vec(&transcript.bytes)?;
    let numeric = p.numeric_f64();
    let (geometry, mut states) = make_stages(
        p.metadata(),
        params,
        r,
        mode,
        numeric.rates()[0].kernel,
        p.reference_kernel_rate_f64(),
    )?;
    let mut initial_bytes = payload::<StateMetadata>(states.len())?;
    for state in &states {
        initial_bytes = sum_payload([initial_bytes, state_payload(state)?])?;
    }
    transcript.metadata(initial_bytes)?;
    let mut initial = Vec::new();
    initial
        .try_reserve_exact(states.len())
        .map_err(|_| issue(IssueKind::AllocationFailed, "initial metadata vector"))?;
    for state in &states {
        initial.push(StateMetadata::from_state(state)?);
    }
    for (g, s) in geometry.iter().zip(&states) {
        encode_geometry(&mut transcript, g)?;
        encode_state(&mut transcript, s)?;
    }
    let index = compose(
        &geometry,
        &mut states,
        params,
        &parts,
        numeric.rates(),
        r,
        &mut transcript,
    )?;
    // All remaining retained arrays/strings are charged before their allocation.
    transcript.metadata(sum_payload([
        payload::<RateCoefficients<f64>>(numeric.rates().len())?,
        p.identity().plan_sha256().len(),
        64usize,
    ])?)?;
    let full_rates = copy_vec(numeric.rates())?;
    let keys = bounded_manifest(&geometry, params, &mut transcript)?;
    let digest = sha256_hex(&transcript.bytes);
    let charges = ReceiptCharges {
        transcript_bytes: transcript.bytes.len(),
        metadata_bytes: transcript.metadata_bytes,
    };
    Ok(DiagonalPlan {
        bytes: transcript.bytes,
        digest,
        binding,
        full: p.identity().plan_sha256().into(),
        consumed,
        full_count: p.plan().updates.len(),
        consumed_count: parts.len(),
        reference: p.reference_kernel_rate_f64().to_bits(),
        rates: full_rates,
        scope: r.consumption,
        index,
        keys,
        partitions: parts,
        initial,
        charges,
    })
}
/// Revalidates a plan against actual trusted materialization/case/parameters/partition.
/// It performs no updates or allocation and returns no execution capability. A real
/// consuming P4/D7 boundary must call this after its genuine gates/start, before keys.
pub fn validate_plan_binding(
    p: &MaterializedProfile,
    c: &RegisteredCase,
    plan: &DiagonalPlan,
    params: PlanningParameters<'_>,
    r: &DiagonalRequest,
) -> Result<()> {
    let (binding, _, _, _) = canonical_binding(p, c, params, r)?;
    require(
        binding.bytes == plan.binding && sha256_hex(&plan.bytes) == plan.digest,
        IssueKind::IdentityMismatch,
        "actual plan/input/parameters/partition binding",
    )
}

fn ct(w: &mut Transcript, m: CtMeta) -> Result<()> {
    w.number(m.level())?;
    w.u64(m.scale_bits())
}
fn cts(w: &mut Transcript, values: &[CtMeta]) -> Result<()> {
    w.number(values.len())?;
    for &m in values {
        ct(w, m)?;
    }
    Ok(())
}
fn counts(w: &mut Transcript, c: &OpCounter) -> Result<()> {
    for n in [
        c.rotations,
        c.relins,
        c.ct_ct_tensor,
        c.ct_pt_mults,
        c.rescales,
        c.sse_switches,
        c.bootstraps,
    ] {
        w.u64(n)?;
    }
    Ok(())
}
fn compact(w: &mut Transcript, c: CompactPlan) -> Result<()> {
    for n in [c.d_out, c.d_in, c.batch, c.diags_per_ct, c.lane_shift] {
        w.number(n)?;
    }
    Ok(())
}
fn encode_geometry(w: &mut Transcript, g: &StageGeometry) -> Result<()> {
    let l = g.layout();
    for n in [l.slots, l.d, l.batch] {
        w.number(n)?;
    }
    w.number(g.shapes().len())?;
    for s in g.shapes() {
        for n in [s.d_out, s.d_in, s.n_out, s.n_in, s.l, s.m, s.baby, s.giant] {
            w.number(n)?;
        }
    }
    w.number(g.mask_recipes().len())?;
    for m in g.mask_recipes() {
        for n in [
            m.canonical_period,
            m.gradient_period,
            m.bias_period,
            m.lanes,
        ] {
            w.number(n)?;
        }
        int_steps(w, &m.reduce_steps)?;
        int_steps(w, &m.replication_steps)?;
        w.tag(u8::from(m.compact_first_period))?;
    }
    w.number(g.transpose_recipes().len())?;
    for r in g.transpose_recipes() {
        w.number(r.len())?;
        for (source, steps) in r {
            w.number(*source)?;
            int_steps(w, steps)?;
        }
    }
    int_steps(w, g.rotation_steps())?;
    w.tag(u8::from(g.compact().is_some()))?;
    if let Some(c) = g.compact() {
        w.number(c.plans.len())?;
        for &p in &c.plans {
            compact(w, p)?;
        }
        w.number(c.containers.len())?;
        for v in &c.containers {
            w.option_number(v.layer)?;
            w.number(v.index)?;
            compact(w, v.plan)?;
        }
        w.number(c.biases.len())?;
        for b in &c.biases {
            w.number(b.container)?;
            w.number(b.position)?;
        }
    }
    Ok(())
}
fn encode_state(w: &mut Transcript, s: &StageState) -> Result<()> {
    for v in [s.weights(), s.biases(), s.compact_weights()] {
        cts(w, v)?;
    }
    w.number(s.velocity().len())?;
    for &v in s.velocity() {
        w.tag(u8::from(v.is_some()))?;
        if let Some(v) = v {
            ct(w, v)?;
        }
    }
    let b = s.batch();
    ct(w, b.x)?;
    ct(w, b.y)?;
    w.number(s.completed_updates())?;
    match s.optimizer() {
        DiagonalOptimizer::Sgd { lr } => {
            w.tag(0)?;
            w.float(lr)?;
        }
        DiagonalOptimizer::Nag {
            lr,
            reference,
            momentum,
        } => {
            w.tag(1)?;
            w.float(lr)?;
            w.float(reference)?;
            w.float(momentum)?;
        }
    }
    let p = s.parameter_identity();
    w.number(p.n())?;
    for q in [p.q(), p.p()] {
        w.number(q.len())?;
        for &n in q {
            w.u64(n)?;
        }
    }
    w.u64(p.scale_bits())?;
    w.number(p.dnum())?;
    w.option_number(p.secret_hamming_weight())?;
    w.tag(u8::from(s.bootstrap_descriptor_identity().is_some()))?;
    if let Some(b) = s.bootstrap_descriptor_identity() {
        w.blob(b)?;
    }
    let k = s.training_keys();
    w.number(k.level)?;
    int_steps(w, &k.rotation_steps)?;
    for n in [k.relinearization, k.conjugation, k.sse_switching] {
        w.tag(u8::from(n))?;
    }
    Ok(())
}
fn encode_he_error(w: &mut Transcript, e: &HekitError) -> Result<()> {
    match e {
        HekitError::InvalidParams(s) => {
            w.tag(0)?;
            w.text(s)
        }
        HekitError::ScaleMismatch => w.tag(1),
        HekitError::LevelMismatch => w.tag(2),
        HekitError::NoLevelsLeft => w.tag(3),
        HekitError::Decode(s) => {
            w.tag(4)?;
            w.text(s)
        }
        HekitError::Device(s) => {
            w.tag(5)?;
            w.text(s)
        }
    }
}
fn encode_boot(w: &mut Transcript, b: &BootstrapEvent) -> Result<()> {
    w.text(&b.role)?;
    w.option_number(b.micro)?;
    w.option_number(b.layer)?;
    ct(w, b.source)?;
    ct(w, b.output)?;
    w.blob(&b.descriptor_identity)
}
fn encode_update(w: &mut Transcript, u: &PlannedStageUpdate, input: &UpdateSpec) -> Result<()> {
    w.number(input.total_valid)?;
    w.float(input.rate)?;
    w.bool_option(input.next_full)?;
    w.number(input.micros.len())?;
    for m in &input.micros {
        w.number(m.n_valid)?;
        ct(w, m.x)?;
        ct(w, m.y)?;
    }
    w.tag(u8::from(u.reference_branch()))?;
    w.tag(u8::from(u.first_nag_update()))?;
    w.tag(u8::from(u.state_bootstrapped()))?;
    let c = u.coefficients();
    w.u64(c.current_bits)?;
    w.option_bits(c.reference_bits)?;
    w.u64(c.bias_bits)?;
    w.option_bits(c.velocity_mu_bits)?;
    w.option_bits(c.velocity_kappa_bits)?;
    counts(w, u.counts_nonbootstrap())?;
    counts(w, u.counts_including_bootstrap())?;
    w.number(u.client_refreshes())?;
    w.number(u.trace().len())?;
    for t in u.trace() {
        w.text(&t.role)?;
        w.option_number(t.micro)?;
        w.option_number(t.layer)?;
        ct(w, t.metadata)?;
        w.number(t.multiplicity)?;
        counts(w, &t.counts)?;
    }
    w.number(u.scale_attempts().len())?;
    for t in u.scale_attempts() {
        w.text(&t.operation)?;
        w.option_number(t.micro)?;
        w.option_number(t.layer)?;
        ct(w, t.source)?;
        w.u64(t.target_bits)?;
        w.tag(u8::from(t.error.is_some()))?;
        if let Some(e) = &t.error {
            encode_he_error(w, e)?;
        }
    }
    w.number(u.bootstrap_events().len())?;
    for t in u.bootstrap_events() {
        encode_boot(w, t)?;
    }
    for t in [u.retained_hidden(), u.canonical_activations()] {
        w.number(t.len())?;
        for micro in t {
            cts(w, micro)?;
        }
    }
    encode_state(w, u.post_state())
}
fn encode_handoff(w: &mut Transcript, h: &PlannedHandoff) -> Result<()> {
    cts(w, &h.outputs)?;
    counts(w, &h.counts_nonbootstrap)?;
    counts(w, &h.counts_including_bootstrap)?;
    w.number(h.client_refreshes)?;
    w.number(h.bootstraps.len())?;
    for b in &h.bootstraps {
        encode_boot(w, b)?;
    }
    Ok(())
}
fn index_room(index: &mut Vec<TransitionIndex>, w: &mut Transcript) -> Result<()> {
    let count = index
        .len()
        .checked_add(1)
        .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "receipt index"))?;
    if count > w.limits.max_records {
        return Err(budget_error(
            "receipt index entries",
            count,
            w.limits.max_records,
        ));
    }
    w.metadata(payload::<TransitionIndex>(1)?)?;
    index
        .try_reserve_exact(1)
        .map_err(|_| issue(IssueKind::AllocationFailed, "receipt index reservation"))
}
#[derive(Clone, Copy)]
struct RecordCheckpoint {
    bytes: usize,
    metadata: usize,
    index: usize,
}
impl RecordCheckpoint {
    fn capture(w: &Transcript, index: &[TransitionIndex]) -> Self {
        Self {
            bytes: w.bytes.len(),
            metadata: w.metadata_bytes,
            index: index.len(),
        }
    }
}
fn commit_record(
    index: &mut Vec<TransitionIndex>,
    w: &Transcript,
    last: &mut RecordCheckpoint,
    record: TransitionIndex,
) {
    index.push(record);
    *last = RecordCheckpoint::capture(w, index);
}
fn fail_record(
    w: &mut Transcript,
    index: &mut Vec<TransitionIndex>,
    last: RecordCheckpoint,
    mut error: AdmissionIssue,
) -> AdmissionIssue {
    w.bytes.truncate(last.bytes);
    w.metadata_bytes = last.metadata;
    index.truncate(last.index);
    error.prefix = std::mem::take(&mut w.bytes);
    error.prefix_index = std::mem::take(index).into_boxed_slice();
    error
}
fn compose(
    g: &[StageGeometry],
    states: &mut [StageState],
    p: PlanningParameters<'_>,
    parts: &[Vec<MicroPart>],
    rates: &[RateCoefficients<f64>],
    r: &DiagonalRequest,
    w: &mut Transcript,
) -> Result<Vec<TransitionIndex>> {
    let mut index = vec![];
    let mut last = RecordCheckpoint::capture(w, &index);
    let mut completed = 0;
    let result: Result<()> = (|| {
        require(
            !g.is_empty() && g.len() == states.len() && rates.len() >= parts.len(),
            IssueKind::IdentityMismatch,
            "composition cardinality",
        )?;
        for (i, part) in parts.iter().enumerate() {
            let next_full = parts
                .get(i + 1)
                .map(|v| v.iter().all(|p| p.valid == g[0].layout().batch));
            let total = part.iter().try_fold(0usize, |n, p| {
                n.checked_add(p.valid)
                    .ok_or_else(|| issue(IssueKind::ArithmeticOverflow, "logical denominator"))
            })?;
            let mut inputs = vec![states[0].batch().x; part.len()];
            for s in 0..g.len() {
                require(
                    inputs.len() == part.len(),
                    IssueKind::IdentityMismatch,
                    "handoff micro count",
                )?;
                let spec = UpdateSpec::new(
                    part.iter()
                        .zip(&inputs)
                        .map(|(v, &x)| MicroMetadata {
                            n_valid: v.valid,
                            x,
                            y: states[s].batch().y,
                        })
                        .collect(),
                    total,
                    rates[i].kernel,
                    next_full,
                );
                index_room(&mut index, w)?;
                r.work_budget.charge(Work::Update)?;
                let update = plan_stage_update(&g[s], p.params, p.bootstrap, &states[s], &spec)
                    .map_err(|e| p2(e, s))?;
                let start = w.bytes.len();
                w.tag(0)?;
                w.number(i)?;
                w.number(s)?;
                // Explicit pre-state and exact incoming entries survive every update.
                encode_state(w, &states[s])?;
                encode_update(w, &update, &spec)?;
                let summary = retain_update_summary(
                    w,
                    UpdateSummarySource {
                        before: StateLabels::from_state(&states[s]),
                        after: StateLabels::from_state(update.post_state()),
                        inputs: &spec.micros,
                        hidden: update.retained_hidden(),
                        canonical: update.canonical_activations(),
                        coefficients: update.coefficients(),
                        reference: update.reference_branch(),
                        first: update.first_nag_update(),
                        booted: update.state_bootstrapped(),
                        nonbootstrap: update.counts_nonbootstrap(),
                        inclusive: update.counts_including_bootstrap(),
                        refreshes: update.client_refreshes(),
                        bootstrap_count: update.bootstrap_events().len(),
                    },
                )?;
                commit_record(
                    &mut index,
                    w,
                    &mut last,
                    TransitionIndex {
                        update: i,
                        stage: s,
                        boundary: false,
                        start,
                        end: w.bytes.len(),
                        next_full,
                        update_metadata: Some(summary),
                        handoff_metadata: None,
                    },
                );
                if s + 1 < g.len() {
                    index_room(&mut index, w)?;
                    r.work_budget.charge(Work::Handoff)?;
                    let h = plan_stage_handoff(&g[s], p.params, p.bootstrap, &update, &g[s + 1])
                        .map_err(|e| p2(e, s))?;
                    let start = w.bytes.len();
                    w.tag(1)?;
                    w.number(i)?;
                    w.number(s)?;
                    encode_handoff(w, &h)?;
                    let summary = retain_handoff_summary(w, &h)?;
                    commit_record(
                        &mut index,
                        w,
                        &mut last,
                        TransitionIndex {
                            update: i,
                            stage: s,
                            boundary: true,
                            start,
                            end: w.bytes.len(),
                            next_full,
                            update_metadata: None,
                            handoff_metadata: Some(summary),
                        },
                    );
                    inputs = h.outputs;
                }
                states[s] = update.post_state().clone();
            }
            completed = i + 1;
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(index),
        Err(mut e) => {
            e.completed = completed;
            e.update = e.update.or(Some(completed));
            Err(fail_record(w, &mut index, last, e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study_profile::StudyRegistry;
    use ckks_train_ckks::bootstrap::metadata::DftSupportProof;
    use std::{path::Path, sync::OnceLock};
    struct SyntheticFixtures {
        registry: StudyRegistry,
        interactive: CkksParams,
        noninteractive: CkksParams,
        boot: BootstrapMetadata,
        bp: BootParams,
        budget: Arc<PlanningBudget>,
    }
    fn fixtures() -> &'static SyntheticFixtures {
        static FIXTURES: OnceLock<SyntheticFixtures> = OnceLock::new();
        FIXTURES.get_or_init(|| {
            let registry=StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
            let interactive=CkksParams::new(10,&[60,40,40,40,40,40,40,40,40,40],&[60],40,10).unwrap();
            let (q,p)=ckks_train_ckks::params::n16_boot_bits();
            let noninteractive=CkksParams::new(10,&q,&p,42,8).unwrap();
            let proof=DftSupportProof::verify_current().unwrap();
            let bp=BootParams{log_slots:9,..BootParams::default_n16()};
            let boot=BootstrapMetadata::new_with_verified_support(&noninteractive,&bp,&proof).unwrap();
            eprintln!("P3 synthetic fixture: params=2 proof=1 bootstrap_metadata=1; data_materializations=0 HE=0");
            SyntheticFixtures {registry,interactive,noninteractive,boot,bp,budget:Arc::new(PlanningBudget::new(
                PlanningUsage{geometries:96,initial_states:96,updates:512,handoffs:256,wrapper_calls:0}))}
        })
    }
    fn parameters(mode: DiagonalMode) -> PlanningParameters<'static> {
        let f = fixtures();
        if mode == DiagonalMode::Interactive {
            PlanningParameters {
                params: &f.interactive,
                bootstrap: None,
                boot_params: None,
            }
        } else {
            PlanningParameters {
                params: &f.noninteractive,
                bootstrap: Some(&f.boot),
                boot_params: Some(&f.bp),
            }
        }
    }
    fn metadata(model: &str) -> ProfileMetadata {
        let mut m = fixtures()
            .registry
            .resolve(&format!("iris__{model}"), 0)
            .unwrap()
            .metadata()
            .clone();
        m.classes = 2;
        m.output_width = 2;
        m.batch = 12;
        for l in &mut m.layers {
            l.input = 2;
            l.output = 2;
        }
        m
    }
    fn request() -> DiagonalRequest {
        let mut r = DiagonalRequest::quick();
        r.consumption = Consumption::Full;
        r.physical_lanes = Some(8);
        r.work_budget = fixtures().budget.clone();
        r
    }
    struct SyntheticPlan {
        index: Vec<TransitionIndex>,
        transcript: Transcript,
        states: Vec<StageState>,
    }
    fn synthetic(
        model: &str,
        mode: DiagonalMode,
        kernel: &[f64],
        parts: Vec<Vec<MicroPart>>,
        r: &DiagonalRequest,
    ) -> Result<SyntheticPlan> {
        // This test-only type/constructor cannot construct MaterializedProfile or DiagonalPlan.
        let m = metadata(model);
        let p = parameters(mode);
        let reference = kernel.iter().copied().fold(0., f64::max);
        let rates: Vec<_> = kernel
            .iter()
            .map(|&kernel| RateCoefficients {
                source: kernel,
                mean: kernel,
                delta: 1.,
                kernel,
            })
            .collect();
        let mut transcript = Transcript::new(r.limits);
        transcript.text("hekit-study-diagonal-synthetic-v1")?;
        rates_encode(&mut transcript, &rates, reference)?;
        let (g, mut states) = make_stages(&m, p, r, mode, kernel[0], reference)?;
        for (a, b) in g.iter().zip(&states) {
            encode_geometry(&mut transcript, a)?;
            encode_state(&mut transcript, b)?;
        }
        let index = compose(&g, &mut states, p, &parts, &rates, r, &mut transcript)?;
        Ok(SyntheticPlan {
            index,
            transcript,
            states,
        })
    }
    fn rates_encode(w: &mut Transcript, r: &[RateCoefficients<f64>], reference: f64) -> Result<()> {
        rates(w, r, reference)
    }
    fn all_modes() -> [DiagonalMode; 4] {
        [
            DiagonalMode::Interactive,
            DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
            DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
        ]
    }
    fn schedule_parts() -> Vec<Vec<MicroPart>> {
        vec![
            vec![MicroPart { start: 0, valid: 8 }],
            vec![
                MicroPart { start: 0, valid: 8 },
                MicroPart { start: 8, valid: 4 },
            ],
            vec![MicroPart { start: 0, valid: 4 }],
            vec![MicroPart { start: 0, valid: 8 }],
        ]
    }
    fn synthetic_schedule_acceptance() {
        for mode in all_modes() {
            for model in ["mlp2020", "emlp1"] {
                let r = request();
                let result =
                    synthetic(model, mode, &[0.02, 0.01, 0.01, 0.02], schedule_parts(), &r);
                let plan = result.unwrap_or_else(|e| panic!("{model} {mode:?}: {e:?}"));
                assert_eq!(plan.index.len(), 4);
                let mut old = None;
                for (i, index) in plan.index.iter().enumerate() {
                    let u = index.update_metadata().unwrap();
                    assert_eq!(u.pre_state().completed_updates(), i);
                    assert_eq!(u.post_state().completed_updates(), i + 1);
                    assert_eq!(
                        index.next_full(),
                        [Some(false), Some(false), Some(true), None][i]
                    );
                    if let Some(before) = old {
                        assert_eq!(u.pre_state().velocity(), before);
                    }
                    old = Some(u.post_state().velocity());
                    if model == "emlp1" {
                        assert_eq!(u.reference_branch(), i == 1 || i == 2);
                        assert_eq!(u.first_nag_update(), i == 0);
                        assert!(u.post_state().velocity().iter().any(Option::is_some));
                        assert_eq!(u.coefficients().reference_bits.is_some(), i == 1 || i == 2);
                    }
                }
                assert_eq!(plan.states[0].completed_updates(), 4);
                assert!(!plan.transcript.bytes.is_empty());
            }
        }
        // Fail only on the third logical update: previous complete prefix stays inspectable.
        let r = request();
        let p = parameters(DiagonalMode::Interactive);
        let m = metadata("emlp1");
        let (g, mut states) =
            make_stages(&m, p, &r, DiagonalMode::Interactive, 0.02, 0.02).unwrap();
        let rs = [0.02, 0.01, f64::NAN].map(|kernel| RateCoefficients {
            source: kernel,
            mean: kernel,
            delta: 1.,
            kernel,
        });
        let mut w = Transcript::new(r.limits);
        let error =
            compose(&g, &mut states, p, &schedule_parts()[..3], &rs, &r, &mut w).unwrap_err();
        assert_eq!(error.completed_updates(), 2);
        assert_eq!(error.update(), Some(2));
        assert!(error.stage_error().is_some());
        assert!(!error.completed_prefix_bytes().is_empty());
        eprintln!(
            "P3 synthetic schedule guarded usage: {:?}; retained late failure={error:?}",
            fixtures().budget.usage()
        );
    }
    fn synthetic_block_acceptance() {
        for mode in all_modes() {
            let plan = synthetic(
                "emlp3",
                mode,
                &[0.01, 0.005, 0.02],
                schedule_parts()[..3].to_vec(),
                &request(),
            )
            .unwrap_or_else(|e| panic!("block {mode:?}: {e:?}"));
            assert_eq!(plan.index.len(), 15); // 3 stage updates and 2 boundaries per logical update.
            for chunk in plan.index.chunks_exact(5) {
                for s in 0..2 {
                    let source = chunk[2 * s].update_metadata().unwrap();
                    let handoff = chunk[2 * s + 1].handoff_metadata().unwrap();
                    let destination = chunk[2 * s + 2].update_metadata().unwrap();
                    assert_eq!(handoff.outputs().len(), source.micros().len());
                    for (i, &x) in handoff.outputs().iter().enumerate() {
                        assert_eq!(destination.micros()[i].x, x);
                        assert_eq!(destination.micros()[i].y, destination.pre_state().batch().y);
                        assert_eq!(destination.micros()[i].n_valid, source.micros()[i].n_valid);
                        if mode == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
                            assert_eq!(x, source.canonical_activations()[i][0]);
                        } else {
                            assert_eq!(x.scale_bits(), parameters(mode).params.scale().to_bits());
                        }
                        assert!(!source.retained_hidden()[i].is_empty());
                    }
                    let n = source.micros().len();
                    if mode == DiagonalMode::Interactive {
                        assert_eq!(
                            (handoff.client_refreshes(), handoff.bootstrap_count()),
                            (n, 0)
                        );
                    } else if mode == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
                        assert_eq!(
                            (
                                handoff.bootstrap_count(),
                                handoff.counts_nonbootstrap().ct_pt_mults
                            ),
                            (0, 0)
                        );
                    } else {
                        assert_eq!(handoff.bootstrap_count(), n);
                        assert_eq!(handoff.counts_nonbootstrap().ct_pt_mults, n as u64);
                    }
                }
            }
        }
        eprintln!(
            "P3 synthetic blocks guarded usage: {:?}",
            fixtures().budget.usage()
        );
    }
    fn synthetic_binding_negatives() {
        let mut r = request();
        for parts in [
            vec![],
            vec![MicroPart { start: 0, valid: 0 }],
            vec![MicroPart { start: 0, valid: 9 }],
            vec![
                MicroPart { start: 0, valid: 8 },
                MicroPart { start: 7, valid: 4 },
            ],
            vec![MicroPart { start: 0, valid: 8 }],
        ] {
            r.partitions = Some(vec![parts]);
            assert!(partitions(&[12], 8, &r).is_err());
        }
        let mut t = Transcript::new(ReceiptLimits::default());
        let r1 = [
            RateCoefficients {
                source: 0.01,
                mean: 0.01,
                delta: 1.,
                kernel: 0.01,
            },
            RateCoefficients {
                source: 0.02,
                mean: 0.02,
                delta: 1.,
                kernel: 0.02,
            },
        ];
        assert!(rates(&mut t, &r1, 0.01).is_err()); // Prefix maximum is not the full maximum.
        let mut a = Transcript::new(ReceiptLimits::default());
        rates(&mut a, &r1, 0.02).unwrap();
        let mut b = Transcript::new(ReceiptLimits::default());
        let mut changed = r1;
        changed[1].kernel = 0.03;
        changed[1].mean = 0.03;
        changed[1].source = 0.03;
        rates(&mut b, &changed, 0.03).unwrap();
        assert_ne!(sha256_hex(&a.bytes), sha256_hex(&b.bytes));
        let p = parameters(DiagonalMode::Interactive);
        assert!(secure_parameters(p, DiagonalMode::Interactive).is_err()); // Mini never admits secure production.
        let mut req = request();
        req.limits.max_records = 1;
        let error = synthetic(
            "emlp1",
            DiagonalMode::Interactive,
            &[0.02, 0.02],
            schedule_parts()[..2].to_vec(),
            &req,
        )
        .err()
        .unwrap();
        assert_eq!(error.kind(), IssueKind::PlanningBudgetExceeded);
        assert_eq!(error.completed_updates(), 1);
        assert!(!error.completed_prefix_bytes().is_empty());
        let mut m = metadata("emlp3");
        m.layers[2].role = LayerRole::Hidden;
        assert!(geometry_identity(&m).is_err());
        // No fake factory exists here. The reachable P4/D7 spy is explicitly pending.
    }
    // Independent wire primitives: explicit bytes, not an encoder-produced golden.
    #[test]
    fn p3_digest_golden_and_mutations() {
        let mut w = Transcript::new(ReceiptLimits {
            max_bytes: 64,
            max_records: 2,
        });
        w.tag(3).unwrap();
        w.number(2).unwrap();
        w.text("x").unwrap();
        w.float(-0.0).unwrap();
        let expected = [
            3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 128,
        ];
        assert_eq!(w.bytes, expected);
        assert_eq!(
            crate::data::sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let mut limit = Transcript::new(ReceiptLimits {
            max_bytes: 1,
            max_records: 1,
        });
        limit.tag(0).unwrap();
        assert!(matches!(
            limit.tag(1).unwrap_err().kind(),
            IssueKind::PlanningBudgetExceeded
        ));
        assert_eq!(limit.bytes, vec![0]);
        let mut combined = Transcript::new(ReceiptLimits {
            max_bytes: 64,
            max_records: 2,
        });
        combined.metadata(24).unwrap();
        combined.raw(&[0; 40]).unwrap();
        assert_eq!(combined.bytes.len() + combined.metadata_bytes, 64);
        assert!(combined.tag(1).is_err());
        assert!(combined.metadata(1).is_err());
        assert_eq!((combined.bytes.len(), combined.metadata_bytes), (40, 24));
        // Actual record commit/failure helpers, no parameters or CtMeta construction.
        let actual_parts = [
            MicroPart { start: 0, valid: 8 },
            MicroPart { start: 8, valid: 4 },
        ];
        let nested_bytes = payload::<MicroPart>(actual_parts.len()).unwrap();
        let structural_bytes = payload::<TransitionIndex>(1).unwrap();
        let mut records = Vec::new();
        let mut wire = Transcript::new(ReceiptLimits {
            max_bytes: 4096,
            max_records: 2,
        });
        wire.raw(b"header").unwrap();
        let mut last = RecordCheckpoint::capture(&wire, &records);
        index_room(&mut records, &mut wire).unwrap();
        wire.metadata(nested_bytes).unwrap();
        let retained_parts = copy_vec(&actual_parts).unwrap();
        wire.raw(b"complete").unwrap();
        let record = TransitionIndex {
            update: 0,
            stage: 0,
            boundary: true,
            start: 6,
            end: 14,
            next_full: None,
            update_metadata: None,
            handoff_metadata: Some(HandoffMetadata {
                outputs: vec![],
                nonbootstrap: OpCounter::default(),
                inclusive: OpCounter::default(),
                refreshes: 2,
                bootstrap_count: 0,
            }),
        };
        commit_record(&mut records, &wire, &mut last, record);
        assert_eq!(wire.metadata_bytes, structural_bytes + nested_bytes);
        // One separately authorized tiny parameter object; no shared fixture or planner.
        let tiny = CkksParams::test_tiny();
        let label = CtMeta::new(&tiny, 3, tiny.scale()).unwrap();
        let labels = [label, label];
        let velocity = [Some(label), None];
        let before = StateLabels {
            weights: &labels,
            biases: &labels,
            compact: &labels,
            velocity: &velocity,
            batch: BatchMetadata { x: label, y: label },
            optimizer: DiagonalOptimizer::Nag {
                lr: 0.01,
                reference: 0.02,
                momentum: 0.9,
            },
            updates: 0,
        };
        let micros = [
            MicroMetadata {
                n_valid: 8,
                x: label,
                y: label,
            },
            MicroMetadata {
                n_valid: 4,
                x: label,
                y: label,
            },
        ];
        let hidden = vec![labels.to_vec(), labels.to_vec()];
        let canonical = hidden.clone();
        let coefficients = GradientCoefficients {
            current_bits: 0.01f64.to_bits(),
            reference_bits: Some(0.02f64.to_bits()),
            bias_bits: 0,
            velocity_mu_bits: Some(0.9f64.to_bits()),
            velocity_kappa_bits: Some(1f64.to_bits()),
        };
        let counter = OpCounter::default();
        let source = || UpdateSummarySource {
            before,
            after: StateLabels {
                updates: 1,
                ..before
            },
            inputs: &micros,
            hidden: &hidden,
            canonical: &canonical,
            coefficients: &coefficients,
            reference: true,
            first: true,
            booted: false,
            nonbootstrap: &counter,
            inclusive: &counter,
            refreshes: 2,
            bootstrap_count: 0,
        };
        let expected_update = 2
            * (6 * std::mem::size_of::<CtMeta>() + 2 * std::mem::size_of::<Option<CtMeta>>())
            + 2 * std::mem::size_of::<MicroMetadata>()
            + 2 * (2 * std::mem::size_of::<Vec<CtMeta>>() + 4 * std::mem::size_of::<CtMeta>());
        assert_eq!(update_summary_payload(&source()).unwrap(), expected_update);
        let handoff = PlannedHandoff {
            outputs: labels.to_vec(),
            counts_nonbootstrap: counter.clone(),
            counts_including_bootstrap: counter.clone(),
            client_refreshes: 2,
            bootstraps: vec![],
        };
        let handoff_payload = 2 * std::mem::size_of::<CtMeta>();
        let cap = 3 + 2 * structural_bytes + expected_update + handoff_payload + nested_bytes;
        let mut charged = Transcript::new(ReceiptLimits {
            max_bytes: cap,
            max_records: 2,
        });
        let mut index = vec![];
        charged.raw(b"H").unwrap();
        let mut checkpoint = RecordCheckpoint::capture(&charged, &index);
        index_room(&mut index, &mut charged).unwrap();
        let update = retain_update_summary(&mut charged, source()).unwrap();
        assert_eq!(
            (
                update.pre_state().weights().len(),
                update.post_state().velocity().len(),
                update.micros().len()
            ),
            (2, 2, 2)
        );
        assert_eq!(
            (
                update.retained_hidden().len(),
                update.canonical_activations().len()
            ),
            (2, 2)
        );
        charged.raw(b"U").unwrap();
        commit_record(
            &mut index,
            &charged,
            &mut checkpoint,
            TransitionIndex {
                update: 0,
                stage: 0,
                boundary: false,
                start: 1,
                end: 2,
                next_full: None,
                update_metadata: Some(update),
                handoff_metadata: None,
            },
        );
        index_room(&mut index, &mut charged).unwrap();
        let h = retain_handoff_summary(&mut charged, &handoff).unwrap();
        assert_eq!(h.outputs(), labels);
        charged.raw(b"B").unwrap();
        charged.metadata(nested_bytes).unwrap();
        let exact_parts = copy_vec(&actual_parts).unwrap();
        commit_record(
            &mut index,
            &charged,
            &mut checkpoint,
            TransitionIndex {
                update: 0,
                stage: 0,
                boundary: true,
                start: 2,
                end: 3,
                next_full: None,
                update_metadata: None,
                handoff_metadata: Some(h),
            },
        );
        assert_eq!(exact_parts, actual_parts);
        assert_eq!(charged.bytes.len() + charged.metadata_bytes, cap);
        let error = charged.tag(0).unwrap_err();
        assert_eq!(error.budget_requirement(), Some((cap + 1, cap)));
        let error = retain_handoff_summary(&mut charged, &handoff).unwrap_err();
        assert_eq!(
            error.budget_requirement(),
            Some((cap + handoff_payload, cap))
        );
        assert_eq!(charged.bytes.len() + charged.metadata_bytes, cap);
        let source_error = Box::new(PlanError {
            kind: ckks_train_nn::diagonal_plan::PlanFailure::ScaleInadmissible,
            update: Some(1),
            operation: "retained failure".into(),
            micro: Some(0),
            layer: Some(0),
            source: Some(HekitError::ScaleMismatch),
            source_scale_bits: Some(label.scale_bits()),
            target_scale_bits: Some(label.scale_bits()),
            prime: Some(tiny.q()[3]),
            trace: vec![ckks_train_nn::diagonal_plan::MetadataEvent {
                role: "failed role".into(),
                micro: Some(0),
                layer: Some(0),
                metadata: label,
                multiplicity: 1,
                counts: counter,
            }],
            attempts: vec![ckks_train_nn::diagonal_plan::ScaleAttempt {
                operation: "exact attempt".into(),
                micro: Some(0),
                layer: Some(0),
                source: label,
                target_bits: label.scale_bits(),
                error: Some(HekitError::ScaleMismatch),
            }],
        });
        let error = fail_record(&mut charged, &mut index, checkpoint, p2(source_error, 1));
        assert_eq!(error.completed_prefix_bytes(), b"HUB");
        assert_eq!(error.completed_prefix_transitions().len(), 2);
        assert_eq!(error.stage_error().unwrap().trace.len(), 1);
        assert_eq!(error.stage_error().unwrap().attempts.len(), 1);
        eprintln!(
            "P3 prefix accounting: tiny_params=1; transcript=3 metadata={} combined={cap}; proof/descriptor/stage/handoff/materialization/HE=0",
            cap - 3
        );
        assert_eq!(retained_parts, actual_parts);
        wire.raw(b"incomplete").unwrap();
        let mut failure = issue(IssueKind::StagePlanning, "failed next record");
        failure.completed = 0; // earlier stage complete, whole logical update incomplete.
        let error = fail_record(&mut wire, &mut records, last, failure);
        assert_eq!(error.completed_prefix_bytes(), b"headercomplete");
        assert_eq!(error.completed_prefix_transitions().len(), 1);
        assert_eq!(
            error.completed_prefix_transitions()[0]
                .handoff_metadata()
                .unwrap()
                .client_refreshes(),
            2
        );
        assert!(
            error
                .completed_prefix_transitions()
                .iter()
                .all(|r| r.end <= error.completed_prefix_bytes().len())
        );
        assert_eq!(wire.metadata_bytes, structural_bytes + nested_bytes);
    }

    #[test]
    fn p3_schedule_state_and_failures() {
        synthetic_schedule_acceptance();
    }

    #[test]
    fn p3_block_old_hidden_and_targets() {
        synthetic_block_acceptance();
    }

    #[test]
    fn p3_pre_key_fail_closed() {
        synthetic_binding_negatives();
    }

    #[test]
    fn p3_unproved_and_key_roles() {
        assert!(UNPROVED.contains(&UnprovedGate::Resource));
        assert!(UNPROVED.contains(&UnprovedGate::Range));
        assert!(UNPROVED.contains(&UnprovedGate::Precision));
        assert!(UNPROVED.contains(&UnprovedGate::Security));
        assert!(UNPROVED.contains(&UnprovedGate::PreKeyFactoryIntegration));
    }
    #[test]
    fn p3_fix1_operational_context() {
        let mut r=request();r.limits.max_records=2;
        let error=synthetic("emlp3",DiagonalMode::Interactive,&[0.02],vec![vec![MicroPart{start:0,valid:8}]],&r).err().unwrap();
        eprintln!("P3 fix1 index RED/GREEN usage {:?}; completed={} stage={:?} update={:?}",r.work_budget.usage(),error.completed_updates(),error.stage(),error.update());
        assert_eq!(error.stage(),Some(1));assert_eq!(error.update(),Some(0));assert_eq!(error.completed_updates(),0);
        assert_eq!(error.budget_requirement(),Some((3,2)));assert_eq!(error.completed_prefix_transitions().len(),2);
        assert!(error.completed_prefix_transitions().iter().all(|x|x.end<=error.completed_prefix_bytes().len()));
        let mut r=request();r.work_budget=Arc::new(PlanningBudget::new(PlanningUsage{geometries:3,initial_states:3,updates:1,handoffs:1,wrapper_calls:0}));
        let error=synthetic("emlp3",DiagonalMode::Interactive,&[0.02],vec![vec![MicroPart{start:0,valid:8}]],&r).err().unwrap();
        assert_eq!(error.stage(),Some(1));assert_eq!(error.update(),Some(0));assert_eq!(error.completed_updates(),0);
        assert_eq!(error.operation(),"stage-update calls");assert_eq!(error.budget_requirement(),Some((2,1)));
        assert_eq!(error.completed_prefix_transitions().len(),2);
        assert_eq!(r.work_budget.usage().updates,1);assert_eq!(r.work_budget.usage().handoffs,1);
        eprintln!("P3 fix1 update-work denial actual usage {:?}",r.work_budget.usage());
    }
}
