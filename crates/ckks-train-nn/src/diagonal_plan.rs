//! Pure exact metadata planning for diagonal training stages.
//!
//! This module allocates metadata only: no Encoder, evaluator, keys or ciphertexts.
//! Receipts describe the existing runtime, including its rejection branches. They are not
//! precision, range, resource, security, dataset or scientific-execution admission.
use crate::{
    dense::{TransposeKeys, transpose_rotation_plan},
    layout::{LayerShape, Layout},
    trainer::mlp_rotation_steps,
    trainer_ni::{
        ActMeta, BootPlacement, CompactLayout, LevelState, must_boot, ni_rotation_steps,
        pass_levels, unpacked,
    },
};
use ckks_train_ckks::bootstrap::metadata::{BootstrapMetadata, SecretDeclaration};
use ckks_train_ckks::evaluator::SCALE_TOLERANCE;
use ckks_train_ckks::metadata::{CtMeta, scalar_metadata_at, target_plain_scale_at};
use ckks_train_ckks::{CkksParams, EvalConfig, OpCounter};
use ckks_train_core::error::HekitError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The three activation implementations covered by exact diagonal planning.
pub enum DiagonalActivation {
    /// The runtime square activation.
    Square,
    /// The runtime quadratic-plus-linear activation.
    PolyRelu2,
    /// The runtime identity activation.
    Identity,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Interactive refresh or one of the existing NI bootstrap placements.
pub enum DiagonalMode {
    /// Top-level client refresh preserving exact input scales.
    Interactive,
    /// Compact NI state with the supplied placement.
    NonInteractive(BootPlacement),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Whether bias gradients use the rate or a zero coefficient; work is retained.
pub enum BiasPolicy {
    /// Biases take SGD gradients at the current kernel rate.
    Active,
    /// Bias coefficient is zero; masking/extraction/refresh still execute.
    Frozen,
}
#[derive(Clone, Copy, Debug, PartialEq)]
/// Explicit SGD or fixed-reference NAG configuration.
pub enum DiagonalOptimizer {
    /// Plain SGD without velocity.
    Sgd {
        /// Current finite positive kernel rate.
        lr: f64,
    },
    /// NAG with persistent reference-scaled velocity.
    Nag {
        /// Current finite positive kernel rate.
        lr: f64,
        /// Fixed full-plan kernel-rate maximum.
        reference: f64,
        /// Momentum in the existing scaled-velocity NAG basis.
        momentum: f64,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// Typed cause of a rejected metadata transition, without a readiness claim.
pub enum PlanFailure {
    /// Dimensions/layout cannot be represented by the existing constructors.
    InvalidDimensions,
    /// A checked count, index or dimension overflowed.
    ArithmeticOverflow,
    /// Micro counts are empty/invalid or do not conserve the logical total.
    InvalidMicroPartition,
    /// Geometry, parameters or bootstrap identity differ from the carried state.
    IdentityMismatch,
    /// Current rate/reference/momentum is invalid.
    RateConflict,
    /// The named transpose pool cannot implement a required rotation.
    RotationUnreachable,
    /// A required supported capability or descriptor is absent.
    MissingCode,
    /// The runtime operation lacks levels or cannot align levels.
    LevelInadmissible,
    /// An exact label or existing runtime scale constraint fails.
    ScaleInadmissible,
    /// Existing reduce-only micro labels differ; automatic repair is unsupported.
    MicroMetadataMismatch,
}
#[derive(Clone, Debug)]
/// Operation context and completed metadata trace for a rejected transition.
pub struct PlanError {
    /// Typed failure category.
    pub kind: PlanFailure,
    /// Zero-based consumed update position when planning an update.
    pub update: Option<usize>,
    /// Named runtime operation or validation boundary.
    pub operation: String,
    /// Zero-based micro index; None for shared logical-update work.
    pub micro: Option<usize>,
    /// Zero-based producing layer; None for container/shared work.
    pub layer: Option<usize>,
    /// Original runtime-compatible arithmetic/scale error, where applicable.
    pub source: Option<HekitError>,
    /// Exact source label when a scale error is reported.
    pub source_scale_bits: Option<u64>,
    /// Exact requested target label when a scale error is reported.
    pub target_scale_bits: Option<u64>,
    /// Actual q prime for a rescale boundary; absent for a plain equality check.
    pub prime: Option<u64>,
    /// Completed role trace before rejection.
    pub trace: Vec<MetadataEvent>,
    /// Completed exact-target attempts including failed retries.
    pub attempts: Vec<ScaleAttempt>,
}
impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} at {}", self.kind, self.operation)
    }
}
impl std::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}
fn error(kind: PlanFailure, op: &str) -> Box<PlanError> {
    Box::new(PlanError {
        kind,
        update: None,
        operation: op.into(),
        micro: None,
        layer: None,
        source: None,
        source_scale_bits: None,
        target_scale_bits: None,
        prime: None,
        trace: vec![],
        attempts: vec![],
    })
}
type Result<T> = std::result::Result<T, Box<PlanError>>;
#[derive(Clone, Debug, PartialEq, Eq)]
enum TransposeIdentity {
    Dedicated,
    Reuse(Vec<i32>),
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct GeometryIdentity {
    dims: Vec<usize>,
    layout: (usize, usize, usize),
    acts: Vec<DiagonalActivation>,
    transpose: TransposeIdentity,
    mode: DiagonalMode,
    cfg: EvalConfig,
}
/// Exact typed identity; no Debug hashing or new serialized scientific descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
/// Exact typed parameter equality; no serialization or Debug hashing.
pub struct ParameterIdentity {
    n: usize,
    q: Vec<u64>,
    p: Vec<u64>,
    scale_bits: u64,
    dnum: usize,
    secret: Option<usize>,
}
impl ParameterIdentity {
    fn new(p: &CkksParams) -> Self {
        Self {
            n: p.n(),
            q: p.q().to_vec(),
            p: p.p().to_vec(),
            scale_bits: p.scale().to_bits(),
            dnum: p.dnum(),
            secret: p.hamming_weight(),
        }
    }
}
impl ParameterIdentity {
    /// Ring degree.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Complete ordered data-prime vector.
    pub fn q(&self) -> &[u64] {
        &self.q
    }
    /// Complete ordered special-prime vector.
    pub fn p(&self) -> &[u64] {
        &self.p
    }
    /// Exact Delta label bits.
    pub fn scale_bits(&self) -> u64 {
        self.scale_bits
    }
    /// Key-switch digit count.
    pub fn dnum(&self) -> usize {
        self.dnum
    }
    /// Declared fixed main-key weight, or None for dense ternary.
    pub fn secret_hamming_weight(&self) -> Option<usize> {
        self.secret
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// Training evaluation-key bundle, separate from bootstrap-owned keys.
pub struct TrainingKeyRequirements {
    /// Exact parameter identity for this key role.
    pub parameters: ParameterIdentity,
    /// Key generation level, over the complete q prefix.
    pub level: usize,
    /// Normalized sorted nonzero required rotation identities.
    pub rotation_steps: Vec<i32>,
    /// Whether the training bundle requires a relinearization key.
    pub relinearization: bool,
    /// Whether this training bundle requires conjugation.
    pub conjugation: bool,
    /// Whether this training bundle requires SSE switching; bootstrap is separate.
    pub sse_switching: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
/// Period and lane recipes; this contains no allocated slot masks.
pub struct MaskRecipe {
    /// Expanded diagonal feature period L.
    pub canonical_period: usize,
    /// First-period feature width for a compact gradient mask.
    pub gradient_period: usize,
    /// Bias mask period selected by actual compact occupancy.
    pub bias_period: usize,
    /// Physical sample lanes per layout block.
    pub lanes: usize,
    /// Ordered positive lane-reduction rotations.
    pub reduce_steps: Vec<i32>,
    /// Ordered negative lane-replication rotations.
    pub replication_steps: Vec<i32>,
    /// Whether gradient masks keep only the first feature period.
    pub compact_first_period: bool,
}
/// Source diagonal and ordered rotation chain per transposed output diagonal.
pub type TransposeRecipe = Vec<(usize, Vec<i32>)>;
/// Checked stage dimensions, physical layout, transpose recipes and config.
#[derive(Clone, Debug)]
pub struct StageGeometry {
    identity: GeometryIdentity,
    shapes: Vec<LayerShape>,
    compact: Option<CompactLayout>,
    transpose: Vec<TransposeRecipe>,
    keys: Vec<i32>,
    masks: Vec<MaskRecipe>,
}
impl StageGeometry {
    /// Logical dimensions, including the actual trainable output width.
    pub fn dims(&self) -> &[usize] {
        &self.identity.dims
    }
    /// Padded layer geometry in canonical layer order.
    pub fn shapes(&self) -> &[LayerShape] {
        &self.shapes
    }
    /// The explicitly bound evaluation flags.
    pub fn eval_config(&self) -> EvalConfig {
        self.identity.cfg
    }
    /// NI compact occupancy including bias-only containers, if applicable.
    pub fn compact(&self) -> Option<&CompactLayout> {
        self.compact.as_ref()
    }
    /// Required training rotations, normalized modulo slots, sorted and nonzero.
    pub fn rotation_steps(&self) -> &[i32] {
        &self.keys
    }
    /// Canonical period, reduce and replication recipes per layer.
    pub fn mask_recipes(&self) -> &[MaskRecipe] {
        &self.masks
    }
    /// Source diagonal and ordered rotation chain per transposed output diagonal.
    pub fn transpose_recipes(&self) -> &[TransposeRecipe] {
        &self.transpose
    }
    /// The checked common stage layout.
    pub fn layout(&self) -> Layout {
        let (s, d, b) = self.identity.layout;
        Layout {
            slots: s,
            d,
            batch: b,
        }
    }
    /// The bound refresh/bootstrap mode.
    pub fn mode(&self) -> DiagonalMode {
        self.identity.mode
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Fresh client encryption labels for this particular stage.
pub struct BatchMetadata {
    /// Input ciphertext metadata.
    pub x: CtMeta,
    /// Stage-specific target ciphertext metadata.
    pub y: CtMeta,
}
#[derive(Clone, Debug)]
/// Immutable carried labels and optimizer basis, bound to exact geometry/parameters.
pub struct StageState {
    geometry: GeometryIdentity,
    params: ParameterIdentity,
    bootstrap: Option<Vec<u8>>,
    optim: DiagonalOptimizer,
    bias_policy: BiasPolicy,
    w: Vec<CtMeta>,
    b: Vec<CtMeta>,
    c: Vec<CtMeta>,
    u: Vec<Option<CtMeta>>,
    batch: BatchMetadata,
    updates: usize,
    training_keys: TrainingKeyRequirements,
}
impl StageState {
    /// Actual expanded-weight labels, one per uniform layer.
    pub fn weights(&self) -> &[CtMeta] {
        &self.w
    }
    /// Actual extracted/refreshed bias labels per layer.
    pub fn biases(&self) -> &[CtMeta] {
        &self.b
    }
    /// I: one uniform diagonal receipt per layer; NI: one entry per compact container.
    pub fn velocity(&self) -> &[Option<CtMeta>] {
        &self.u
    }
    /// NI compact-weight labels per physical container.
    pub fn compact_weights(&self) -> &[CtMeta] {
        &self.c
    }
    /// Fresh stage-specific client BatchSpec labels.
    pub fn batch(&self) -> BatchMetadata {
        self.batch
    }
    /// Number of successfully planned logical updates.
    pub fn completed_updates(&self) -> usize {
        self.updates
    }
    /// Current optimizer rate and persistent full-plan reference/momentum.
    pub fn optimizer(&self) -> DiagonalOptimizer {
        self.optim
    }
    /// Exact parameter values bound to this state.
    pub fn parameter_identity(&self) -> &ParameterIdentity {
        &self.params
    }
    /// Separate training-key requirements at their generation level.
    pub fn training_keys(&self) -> &TrainingKeyRequirements {
        &self.training_keys
    }
    /// Existing P1 descriptor identity bytes, when NI is selected.
    pub fn bootstrap_descriptor_identity(&self) -> Option<&[u8]> {
        self.bootstrap.as_deref()
    }
}
#[derive(Clone, Copy, Debug)]
/// One ordered micro contribution; a valid count is checked against physical lanes.
pub struct MicroMetadata {
    /// Actual active sample lanes in this micro.
    pub n_valid: usize,
    /// Input ciphertext metadata.
    pub x: CtMeta,
    /// Stage-specific target ciphertext metadata.
    pub y: CtMeta,
}
#[derive(Clone, Debug)]
/// One logical update with its full denominator and explicit next-full lookahead.
pub struct UpdateSpec {
    /// Ordered nonempty micro contributions from one logical update.
    pub micros: Vec<MicroMetadata>,
    /// One checked sum of all valid sample counts.
    pub total_valid: usize,
    /// Current kernel rate for this update, without loss-factor conversion.
    pub rate: f64,
    /// True only when every next micro is full; None is conservative partial.
    pub next_full: Option<bool>,
}
impl UpdateSpec {
    /// Builds an update request; validation occurs before planning any operation.
    pub fn new(
        micros: Vec<MicroMetadata>,
        total_valid: usize,
        rate: f64,
        next_full: Option<bool>,
    ) -> Self {
        Self {
            micros,
            total_valid,
            rate,
            next_full,
        }
    }
}
/// A named uniform ciphertext role; multiplicity names actual diagonal copies, not samples.
#[derive(Clone, Debug)]
/// A named runtime ciphertext role, exact label and operation count delta.
pub struct MetadataEvent {
    /// Explicit ciphertext role name.
    pub role: String,
    /// Zero-based micro index; None for shared logical-update work.
    pub micro: Option<usize>,
    /// Zero-based producing layer; None for container/shared work.
    pub layer: Option<usize>,
    /// Actual level and exact F64 scale bits.
    pub metadata: CtMeta,
    /// Number of uniform ciphertexts represented by this metadata row.
    pub multiplicity: usize,
    /// Count delta of successful expensive operations for this role.
    pub counts: OpCounter,
}
#[derive(Clone, Debug)]
/// An exact-target attempt, including failures before homomorphic work.
pub struct ScaleAttempt {
    /// Named runtime operation or validation boundary.
    pub operation: String,
    /// Zero-based micro index; None for shared logical-update work.
    pub micro: Option<usize>,
    /// Zero-based producing layer; None for container/shared work.
    pub layer: Option<usize>,
    /// Ciphertext metadata before the attempted exact-target operation.
    pub source: CtMeta,
    /// Exact requested target label.
    pub target_bits: u64,
    /// Actual scale-planning error, or None on success.
    pub error: Option<HekitError>,
}
#[derive(Clone, Debug)]
/// A bootstrap invocation at level zero with a bound P1 descriptor identity.
pub struct BootstrapEvent {
    /// Explicit ciphertext role name.
    pub role: String,
    /// Zero-based micro index; None for shared logical-update work.
    pub micro: Option<usize>,
    /// Zero-based producing layer; None for container/shared work.
    pub layer: Option<usize>,
    /// Actual bootstrap input after switching to level zero.
    pub source: CtMeta,
    /// Descriptor-declared bootstrap output metadata.
    pub output: CtMeta,
    /// Existing P1 identity bytes, including full bootstrap configuration.
    pub descriptor_identity: Vec<u8>,
}
#[derive(Clone, Debug)]
/// Complete metadata/count receipt and immutable next stage state.
pub struct PlannedStageUpdate {
    state: StageState,
    counts: OpCounter,
    including_boot: OpCounter,
    reference: bool,
    first: bool,
    refreshes: usize,
    coefficients: GradientCoefficients,
    events: Vec<MetadataEvent>,
    attempts: Vec<ScaleAttempt>,
    boots: Vec<BootstrapEvent>,
    hidden: Vec<Vec<CtMeta>>,
    canonical: Vec<Vec<CtMeta>>,
    booted: bool,
}
impl PlannedStageUpdate {
    /// The immutable state after update/refresh/unpack/extraction.
    pub fn post_state(&self) -> &StageState {
        &self.state
    }
    /// Seven-field training counts excluding bootstrap internal work.
    pub fn counts_nonbootstrap(&self) -> &OpCounter {
        &self.counts
    }
    /// Seven-field model including descriptor-specific bootstrap costs once.
    pub fn counts_including_bootstrap(&self) -> &OpCounter {
        &self.including_boot
    }
    /// Whether this update emits the extra reference-rate gradient.
    pub fn reference_branch(&self) -> bool {
        self.reference
    }
    /// Whether NAG initializes previously absent velocity on this update.
    pub fn first_nag_update(&self) -> bool {
        self.first
    }
    /// Client refresh calls, including newly initialized I velocity.
    pub fn client_refreshes(&self) -> usize {
        self.refreshes
    }
    /// Named role labels and counted operation deltas in execution order.
    pub fn trace(&self) -> &[MetadataEvent] {
        &self.events
    }
    /// Actual-prime exact-search attempts and retry decisions.
    pub fn scale_attempts(&self) -> &[ScaleAttempt] {
        &self.attempts
    }
    /// State/activation bootstrap events, separately identified.
    pub fn bootstrap_events(&self) -> &[BootstrapEvent] {
        &self.boots
    }
    /// Raw pre-update hidden activation labels, indexed micro then hidden layer.
    pub fn retained_hidden(&self) -> &[Vec<CtMeta>] {
        &self.hidden
    }
    /// AA outputs indexed micro then hidden layer, suitable for boundary reuse.
    pub fn canonical_activations(&self) -> &[Vec<CtMeta>] {
        &self.canonical
    }
    /// The unchanged structural policy decision for NI compact-state refresh.
    pub fn state_bootstrapped(&self) -> bool {
        self.booted
    }
}
fn count(r: u64, l: u64, t: u64, p: u64, s: u64) -> OpCounter {
    OpCounter {
        rotations: r,
        relins: l,
        ct_ct_tensor: t,
        ct_pt_mults: p,
        rescales: s,
        ..Default::default()
    }
}
fn merge(a: &mut OpCounter, b: &OpCounter) -> Result<()> {
    macro_rules! add {
        ($f:ident) => {
            a.$f =
                a.$f.checked_add(b.$f)
                    .ok_or_else(|| error(PlanFailure::ArithmeticOverflow, "operation counts"))?;
        };
    }
    add!(rotations);
    add!(relins);
    add!(ct_ct_tensor);
    add!(ct_pt_mults);
    add!(rescales);
    add!(sse_switches);
    add!(bootstraps);
    Ok(())
}
fn times(a: usize, b: usize) -> Result<u64> {
    u64::try_from(
        a.checked_mul(b)
            .ok_or_else(|| error(PlanFailure::ArithmeticOverflow, "count product"))?,
    )
    .map_err(|_| error(PlanFailure::ArithmeticOverflow, "count u64"))
}
fn meta(p: &CkksParams, l: usize, s: f64) -> Result<CtMeta> {
    CtMeta::new(p, l, s).map_err(|e| {
        let mut x = error(PlanFailure::ScaleInadmissible, "finite metadata");
        x.source = Some(e);
        x
    })
}
fn act_meta(a: DiagonalActivation) -> ActMeta {
    match a {
        DiagonalActivation::Identity => ActMeta {
            depth: 0,
            dderiv: 0,
            constant: Some(1.),
        },
        _ => ActMeta {
            depth: 1,
            dderiv: 1,
            constant: None,
        },
    }
}
fn act_output(p: &CkksParams, a: DiagonalActivation, z: CtMeta) -> Result<CtMeta> {
    if a == DiagonalActivation::Identity {
        return Ok(z);
    }
    if z.level() == 0 {
        return Err(error(PlanFailure::LevelInadmissible, "activation"));
    }
    let q = p.q()[z.level()] as f64;
    let squared = meta(p, z.level() - 1, (z.scale() * z.scale()) / q)?;
    if a == DiagonalActivation::PolyRelu2 {
        let sp = crate::activation::ordinary_plain_scale(q, z.scale(), squared.scale());
        let lin = meta(p, z.level() - 1, (z.scale() * sp) / q)?;
        if (lin.scale() - squared.scale()).abs()
            >= SCALE_TOLERANCE * lin.scale().min(squared.scale())
        {
            return Err(error(
                PlanFailure::ScaleInadmissible,
                "PolyRelu2 linear add",
            ));
        }
    }
    Ok(squared)
}
fn validate_optimizer(o: DiagonalOptimizer, rate: f64) -> Result<()> {
    if !(rate.is_finite() && rate > 0.) {
        return Err(error(PlanFailure::RateConflict, "positive finite rate"));
    }
    if let DiagonalOptimizer::Nag {
        reference,
        momentum,
        ..
    } = o
    {
        // Exactly the existing concrete Nag config validation, with no velocity mutation.
        let mut n = crate::optim::Nag::new(rate, momentum);
        n.set_lr_reference(reference).map_err(|e| {
            let mut x = error(PlanFailure::RateConflict, "NAG reference/config");
            x.source = Some(e);
            x
        })?;
    }
    Ok(())
}
fn lr(o: DiagonalOptimizer) -> f64 {
    match o {
        DiagonalOptimizer::Sgd { lr } | DiagonalOptimizer::Nag { lr, .. } => lr,
    }
}
fn is_nag(o: DiagonalOptimizer) -> bool {
    matches!(o, DiagonalOptimizer::Nag { .. })
}
fn with_rate(o: DiagonalOptimizer, rate: f64) -> DiagonalOptimizer {
    match o {
        DiagonalOptimizer::Sgd { .. } => DiagonalOptimizer::Sgd { lr: rate },
        DiagonalOptimizer::Nag {
            reference,
            momentum,
            ..
        } => DiagonalOptimizer::Nag {
            lr: rate,
            reference,
            momentum,
        },
    }
}
fn check_boot(p: &CkksParams, g: &StageGeometry, b: Option<&BootstrapMetadata>) -> Result<()> {
    if g.layout().slots != p.slots() {
        return Err(error(
            PlanFailure::IdentityMismatch,
            "layout parameter slots",
        ));
    }
    if let DiagonalMode::NonInteractive(_) = g.mode() {
        let b =
            b.ok_or_else(|| error(PlanFailure::MissingCode, "bootstrap descriptor required"))?;
        let k = b.generated_keys();
        let secret = p.hamming_weight().map_or(
            SecretDeclaration::DenseTernary,
            SecretDeclaration::FixedWeight,
        );
        if k.q() != p.q()
            || k.p() != p.p()
            || k.galois().n() != p.n()
            || k.galois().digits() != p.dnum()
            || b.declared_main_secret() != secret
            || b.output().scale_bits() != p.scale().to_bits()
            || b.output().level() == 0
            || b.output().level() > p.max_level()
        {
            return Err(error(
                PlanFailure::IdentityMismatch,
                "bootstrap parameter identity",
            ));
        }
    } else if b.is_some() {
        return Err(error(
            PlanFailure::IdentityMismatch,
            "interactive bootstrap descriptor",
        ));
    }
    Ok(())
}
/// Validates geometry before any panicking shape constructor; EvalConfig is explicit.
pub fn plan_stage_geometry(
    dims: &[usize],
    layout: &Layout,
    acts: &[DiagonalActivation],
    transpose: &TransposeKeys,
    mode: DiagonalMode,
    cfg: EvalConfig,
) -> Result<StageGeometry> {
    if !layout.slots.is_power_of_two()
        || layout.slots > (1usize << (ckks_train_ckks::params::MAX_LOG_N - 1))
        || layout.slots < (1usize << (ckks_train_ckks::params::MIN_LOG_N - 1))
    {
        return Err(error(PlanFailure::InvalidDimensions, "CKKS slot domain"));
    }
    Layout::new(layout.slots, layout.d, layout.batch)
        .map_err(|_| error(PlanFailure::InvalidDimensions, "Layout"))?;
    if dims
        .iter()
        .any(|&d| d > 0 && d.checked_next_power_of_two().is_none())
    {
        return Err(error(PlanFailure::ArithmeticOverflow, "padded dimension"));
    }
    if dims.len() < 2
        || acts.len() + 1 != dims.len()
        || layout.slots > i32::MAX as usize
        || dims
            .iter()
            .any(|&d| d == 0 || d.checked_next_power_of_two().is_none_or(|n| n > layout.d))
    {
        return Err(error(PlanFailure::InvalidDimensions, "stage dimensions"));
    }
    let shapes = dims
        .windows(2)
        .map(|d| LayerShape::new(d[1], d[0], layout))
        .collect::<Vec<_>>();
    let recipes = dims
        .windows(2)
        .enumerate()
        .map(|(i, d)| {
            if i == 0 {
                Ok(vec![])
            } else {
                transpose_rotation_plan(d[1], d[0], layout, transpose)
            }
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| error(PlanFailure::RotationUnreachable, "transpose pool"))?;
    let raw = match mode {
        DiagonalMode::Interactive => mlp_rotation_steps(dims, layout, transpose),
        _ => ni_rotation_steps(dims, layout, transpose),
    }
    .map_err(|_| error(PlanFailure::RotationUnreachable, "rotation union"))?;
    let mut keys = raw
        .into_iter()
        .map(|s| (s as i64).rem_euclid(layout.slots as i64) as i32)
        .filter(|&s| s != 0)
        .collect::<Vec<_>>();
    keys.sort_unstable();
    keys.dedup();
    let compact = if matches!(mode, DiagonalMode::NonInteractive(_)) {
        Some(
            CompactLayout::new(dims, layout)
                .map_err(|_| error(PlanFailure::InvalidDimensions, "compact layout"))?,
        )
    } else {
        None
    };
    let masks = shapes
        .iter()
        .enumerate()
        .map(|(i, sh)| MaskRecipe {
            canonical_period: sh.l,
            gradient_period: sh.l,
            bias_period: compact
                .as_ref()
                .map_or(sh.n_out, |cl| cl.bias_plan(i).diag_len()),
            lanes: layout.batch,
            reduce_steps: (0..layout.log_batch()).map(|j| 1 << j).collect(),
            replication_steps: (0..layout.log_batch()).map(|j| -(1 << j)).collect(),
            compact_first_period: compact.is_some(),
        })
        .collect();
    Ok(StageGeometry {
        identity: GeometryIdentity {
            dims: dims.to_vec(),
            layout: (layout.slots, layout.d, layout.batch),
            acts: acts.to_vec(),
            transpose: match transpose {
                TransposeKeys::Dedicated => TransposeIdentity::Dedicated,
                TransposeKeys::Reuse(p) => TransposeIdentity::Reuse(p.clone()),
            },
            mode,
            cfg,
        },
        shapes,
        compact,
        transpose: recipes,
        keys,
        masks,
    })
}
fn fresh_labels(
    g: &StageGeometry,
    p: &CkksParams,
    start: usize,
    boot: Option<CtMeta>,
    w: &[CtMeta],
    inputs: Option<&[i64]>,
) -> Result<(Vec<f64>, CtMeta)> {
    let mut a = meta(p, start, p.scale())?;
    let mut scales = vec![];
    for (i, (&act, w)) in g.identity.acts.iter().zip(w).enumerate() {
        let l = if let Some(ins) = inputs {
            usize::try_from(ins[i])
                .map_err(|_| error(PlanFailure::LevelInadmissible, "input scales"))?
        } else {
            a.level().min(w.level()).min(start)
        };
        scales.push(a.scale());
        if l == 0 {
            return Err(error(PlanFailure::LevelInadmissible, "initial dense"));
        }
        a = act_output(
            p,
            act,
            meta(p, l - 1, (a.scale() * w.scale()) / p.q()[l] as f64)?,
        )?;
        if g.mode() == DiagonalMode::NonInteractive(BootPlacement::AfterActivation)
            && i + 1 < w_len(g)
        {
            a = boot.ok_or_else(|| error(PlanFailure::MissingCode, "activation bootstrap"))?;
        }
    }
    Ok((scales, a))
}
fn next_input_scales(
    g: &StageGeometry,
    p: &CkksParams,
    w: &[CtMeta],
    inputs: &[i64],
) -> Result<Vec<f64>> {
    let mut scales = vec![p.scale()];
    for i in 0..w_len(g) - 1 {
        if g.mode() == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
            scales.push(p.scale());
            continue;
        }
        let l = usize::try_from(inputs[i])
            .map_err(|_| error(PlanFailure::LevelInadmissible, "next input scale"))?;
        if l == 0 {
            return Err(error(PlanFailure::LevelInadmissible, "next dense scale"));
        }
        let z = meta(p, l - 1, (scales[i] * w[i].scale()) / p.q()[l] as f64)?;
        scales.push(act_output(p, g.identity.acts[i], z)?.scale());
    }
    Ok(scales)
}
fn w_len(g: &StageGeometry) -> usize {
    g.shapes.len()
}
/// Plans fresh labels without evaluation material; binds the full-plan optimizer reference.
pub fn initial_stage_metadata(
    g: &StageGeometry,
    p: &CkksParams,
    boot: Option<&BootstrapMetadata>,
    optim: DiagonalOptimizer,
    bias_policy: BiasPolicy,
) -> Result<StageState> {
    check_boot(p, g, boot)?;
    validate_optimizer(optim, lr(optim))?;
    let start = boot.map_or(p.max_level(), |b| b.output().level() - 1);
    let n = w_len(g);
    let w = vec![meta(p, start, p.scale())?; n];
    let (scales, y) = fresh_labels(g, p, start, boot.map(|b| b.output()), &w, None)?;
    let b = scales
        .iter()
        .map(|&s| meta(p, start, s * p.scale()))
        .collect::<Result<Vec<_>>>()?;
    let c = if let Some(cl) = &g.compact {
        vec![boot.unwrap().output(); cl.containers.len()]
    } else {
        vec![]
    };
    let u = vec![None; if g.compact.is_some() { c.len() } else { n }];
    let state = StageState {
        geometry: g.identity.clone(),
        params: ParameterIdentity::new(p),
        bootstrap: boot.map(|b| b.descriptor_identity_bytes().to_vec()),
        optim,
        bias_policy,
        w,
        b,
        c,
        u,
        batch: BatchMetadata {
            x: meta(p, start, p.scale())?,
            y,
        },
        updates: 0,
        training_keys: TrainingKeyRequirements {
            parameters: ParameterIdentity::new(p),
            level: p.max_level(),
            rotation_steps: g.keys.clone(),
            relinearization: true,
            conjugation: false,
            sse_switching: false,
        },
    };
    if let Some(b) = boot {
        let metas = g
            .identity
            .acts
            .iter()
            .copied()
            .map(act_meta)
            .collect::<Vec<_>>();
        let pass = pass_levels(
            &metas,
            g.mode() == DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
            &vec![start as i64; n],
            &vec![start as i64; n],
            start as i64,
            b.output().level() as i64,
            true,
        );
        if pass.min < 0 {
            return Err(error(
                PlanFailure::LevelInadmissible,
                "initial structural depth",
            ));
        }
    }
    Ok(state)
}
struct ForwardPass {
    acts: Vec<CtMeta>,
    zs: Vec<CtMeta>,
    hidden: Vec<CtMeta>,
    canonical: Vec<CtMeta>,
}
struct Planning<'a> {
    g: &'a StageGeometry,
    p: &'a CkksParams,
    boot: Option<&'a BootstrapMetadata>,
    trace: Vec<MetadataEvent>,
    attempts: Vec<ScaleAttempt>,
    boots: Vec<BootstrapEvent>,
    counts: OpCounter,
    micro: Option<usize>,
    layer: Option<usize>,
}
impl<'a> Planning<'a> {
    fn new(g: &'a StageGeometry, p: &'a CkksParams, boot: Option<&'a BootstrapMetadata>) -> Self {
        Self {
            g,
            p,
            boot,
            trace: vec![],
            attempts: vec![],
            boots: vec![],
            counts: OpCounter::default(),
            micro: None,
            layer: None,
        }
    }
    fn fail(&self, kind: PlanFailure, op: &str) -> Box<PlanError> {
        let mut e = error(kind, op);
        e.micro = self.micro;
        e.layer = self.layer;
        e.trace = self.trace.clone();
        e.attempts = self.attempts.clone();
        e
    }
    fn scale_error(&self, op: &str, source: CtMeta, target: f64) -> Box<PlanError> {
        let mut e = self.fail(PlanFailure::ScaleInadmissible, op);
        e.source = Some(HekitError::ScaleMismatch);
        e.source_scale_bits = Some(source.scale_bits());
        e.target_scale_bits = Some(target.to_bits());
        e
    }
    fn checked<T>(&self, r: Result<T>) -> Result<T> {
        r.map_err(|mut e| {
            e.micro = self.micro;
            e.layer = self.layer;
            e.trace = self.trace.clone();
            e.attempts = self.attempts.clone();
            e
        })
    }
    fn emit(&mut self, role: &str, m: CtMeta, multiplicity: usize, c: OpCounter) -> Result<CtMeta> {
        let mut summed = self.counts.clone();
        self.checked(merge(&mut summed, &c))?;
        self.counts = summed;
        self.trace.push(MetadataEvent {
            role: role.into(),
            micro: self.micro,
            layer: self.layer,
            metadata: m,
            multiplicity,
            counts: c,
        });
        Ok(m)
    }
    fn note(&mut self, role: &str, m: CtMeta) -> Result<CtMeta> {
        self.emit(role, m, 1, OpCounter::default())
    }
    fn at(&self, m: CtMeta, l: usize) -> Result<CtMeta> {
        if l > m.level() {
            return Err(self.fail(PlanFailure::LevelInadmissible, "mod switch upward"));
        }
        self.checked(meta(self.p, l, m.scale()))
    }
    fn sub(&self, a: CtMeta, b: CtMeta, exact: bool, op: &str) -> Result<CtMeta> {
        if a.level() != b.level() {
            return Err(self.fail(PlanFailure::LevelInadmissible, op));
        }
        if (exact && a.scale() != b.scale())
            || ((a.scale() - b.scale()).abs() >= SCALE_TOLERANCE * a.scale().min(b.scale()))
        {
            return Err(self.scale_error(op, b, a.scale()));
        }
        Ok(a)
    }
    fn product(&self, a: CtMeta, b: CtMeta) -> Result<CtMeta> {
        let l = a.level().min(b.level());
        if l == 0 {
            return Err(self.fail(PlanFailure::LevelInadmissible, "ct product"));
        }
        self.checked(meta(
            self.p,
            l - 1,
            (a.scale() * b.scale()) / self.p.q()[l] as f64,
        ))
    }
    fn exact(
        &mut self,
        role: &str,
        src: CtMeta,
        target: f64,
        scalar: Option<f64>,
        mult: usize,
        rot: u64,
    ) -> Result<CtMeta> {
        let r = if let Some(v) = scalar {
            scalar_metadata_at(self.p, src, v, target).map(|m| m.post_rescale())
        } else {
            target_plain_scale_at(self.p, src.level(), src.scale(), target)
                .and_then(|_| CtMeta::new(self.p, src.level() - 1, target))
        };
        self.attempts.push(ScaleAttempt {
            operation: role.into(),
            micro: self.micro,
            layer: self.layer,
            source: src,
            target_bits: target.to_bits(),
            error: r.as_ref().err().cloned(),
        });
        match r {
            Ok(m) => self.emit(role, m, mult, count(rot, 0, 0, mult as u64, mult as u64)),
            Err(source) => {
                let mut e = self.fail(
                    if source == HekitError::NoLevelsLeft {
                        PlanFailure::LevelInadmissible
                    } else {
                        PlanFailure::ScaleInadmissible
                    },
                    role,
                );
                e.source = Some(source);
                e.source_scale_bits = Some(src.scale_bits());
                e.target_scale_bits = Some(target.to_bits());
                e.prime = self.p.q().get(src.level()).copied();
                Err(e)
            }
        }
    }
    fn activation(&mut self, a: DiagonalActivation, z: CtMeta) -> Result<CtMeta> {
        let m = self.checked(act_output(self.p, a, z))?;
        let c = match a {
            DiagonalActivation::Identity => OpCounter::default(),
            DiagonalActivation::Square => count(0, 1, 1, 0, 1),
            DiagonalActivation::PolyRelu2 => count(0, 1, 1, 1, 2),
        };
        self.emit("a", m, 1, c)
    }
    fn dz(
        &mut self,
        a: DiagonalActivation,
        z: CtMeta,
        delta: CtMeta,
        full: bool,
    ) -> Result<CtMeta> {
        if a == DiagonalActivation::Identity {
            return if full {
                self.note("dz", delta)
            } else {
                self.exact("dz", delta, delta.scale(), None, 1, 0)
            };
        }
        if z.level() == 0 {
            return Err(self.fail(PlanFailure::LevelInadmissible, "derivative factor"));
        }
        let sp = crate::activation::ordinary_plain_scale(
            self.p.q()[z.level()] as f64,
            z.scale(),
            z.scale(),
        );
        let s = self.checked(meta(
            self.p,
            z.level() - 1,
            (z.scale() * sp) / self.p.q()[z.level()] as f64,
        ))?;
        self.emit("s", s, 1, count(0, 0, 0, 1, 1))?;
        let d = self.product(s, delta)?;
        self.emit("dz", d, 1, count(0, 1, 1, 0, 1))
    }
    fn dense(
        &mut self,
        role: &str,
        x: CtMeta,
        w: CtMeta,
        b: Option<CtMeta>,
        sh: LayerShape,
    ) -> Result<CtMeta> {
        let l = x.level();
        if w.level() < l || b.is_some_and(|b| b.level() < l) {
            return Err(self.fail(PlanFailure::LevelInadmissible, role));
        }
        if b.is_some_and(|b| b.scale() != x.scale() * w.scale()) {
            let mut e = self.scale_error(
                "dense bias exact product",
                b.unwrap(),
                x.scale() * w.scale(),
            );
            e.prime = self.p.q().get(l).copied();
            return Err(e);
        }
        let m = self.product(x, w)?;
        let cfg = self.g.eval_config();
        self.emit(
            role,
            m,
            1,
            count(
                cfg.matvec_rotations(sh.m) + sh.fold_steps() as u64,
                cfg.matvec_enc_relins(sh.m),
                sh.m as u64,
                0,
                1,
            ),
        )
    }
    fn bootstrap(&mut self, role: &str, src: CtMeta) -> Result<CtMeta> {
        let b = self
            .boot
            .ok_or_else(|| self.fail(PlanFailure::MissingCode, "bootstrap descriptor"))?;
        if src.scale_bits() != b.input().scale_bits() {
            return Err(self.scale_error("bootstrap input scale", src, b.input().scale()));
        }
        let output = b.output();
        let source = self.at(src, 0)?;
        self.boots.push(BootstrapEvent {
            role: role.into(),
            micro: self.micro,
            layer: self.layer,
            source,
            output,
            descriptor_identity: b.descriptor_identity_bytes().to_vec(),
        });
        self.note(role, output)
    }
    fn forward(&mut self, s: &StageState, x: CtMeta) -> Result<ForwardPass> {
        let mut acts = vec![];
        let mut zs = vec![];
        let mut hidden = vec![];
        let mut canon = vec![];
        let mut prev = x;
        for i in 0..w_len(self.g) {
            self.layer = Some(i);
            let input = if self.g.mode() == DiagonalMode::Interactive {
                prev
            } else {
                self.at(prev, prev.level().min(s.w[i].level()).min(s.b[i].level()))?
            };
            self.note("input", input)?;
            if i == 0 {
                acts.push(input)
            }
            let z = self.dense("z", input, s.w[i], Some(s.b[i]), self.g.shapes[i])?;
            let a = self.activation(self.g.identity.acts[i], z)?;
            zs.push(z);
            if i + 1 < w_len(self.g) {
                hidden.push(a)
            }
            prev = if self.g.mode() == DiagonalMode::NonInteractive(BootPlacement::AfterActivation)
                && i + 1 < w_len(self.g)
            {
                let d = self.exact("a_to_delta", a, self.p.scale(), Some(1.), 1, 0)?;
                let b = self.bootstrap("a_boot", d)?;
                canon.push(b);
                b
            } else {
                a
            };
            acts.push(prev);
        }
        Ok(ForwardPass {
            acts,
            zs,
            hidden,
            canonical: canon,
        })
    }
    fn loss(&mut self, out: CtMeta, y: CtMeta, ni: bool) -> Result<CtMeta> {
        let y = if ni && y.scale() != out.scale() {
            if y.level() <= out.level() {
                return Err(self.scale_error("target alignment level", y, out.scale()));
            }
            let y = self.at(y, out.level() + 1)?;
            self.exact("aligned_target", y, out.scale(), Some(1.), 1, 0)?
        } else {
            self.note("aligned_target", y)?
        };
        // Source: loss.rs::Mse::grad; preserve the existing 1e-12 relative bound exactly.
        if (out.scale() - y.scale()).abs() > 1e-12 * out.scale().min(y.scale()) {
            return Err(self.scale_error("Mse target scale", y, out.scale()));
        }
        let d = self.at(out, out.level().min(y.level()))?;
        self.note("loss_delta", d)
    }
    fn backward(&mut self, i: usize, w: CtMeta, dz: CtMeta, transpose: bool) -> Result<CtMeta> {
        let wt = self.at(w, dz.level())?;
        if transpose {
            let r = self.g.transpose[i]
                .iter()
                .map(|(_, r)| r.len() as u64)
                .sum();
            self.emit("transpose", wt, self.g.shapes[i].m, count(r, 0, 0, 0, 0))?;
        }
        // The layer index denotes the producing/source weight layer, not the receiver.
        self.dense(
            "backward_delta",
            dz,
            wt,
            None,
            self.g.shapes[i].transposed(),
        )
    }
    fn gradient_label_precheck(&mut self, prod: CtMeta, target: f64) -> Result<()> {
        let result =
            ckks_train_ckks::metadata::exact_product_scale_at(self.p, prod.level(), target);
        match result {
            Ok(label) if (label / prod.scale()).is_finite() && label / prod.scale() > 0. => Ok(()),
            outcome => {
                let source = outcome.err().unwrap_or_else(|| {
                    HekitError::InvalidParams("nonfinite gradient plaintext scale".into())
                });
                self.attempts.push(ScaleAttempt {
                    operation: "gradient_mask_precheck".into(),
                    micro: self.micro,
                    layer: self.layer,
                    source: prod,
                    target_bits: target.to_bits(),
                    error: Some(source.clone()),
                });
                let mut e = self.fail(PlanFailure::ScaleInadmissible, "gradient_mask_precheck");
                e.source = Some(source);
                e.prime = Some(self.p.q()[prod.level()]);
                e.source_scale_bits = Some(prod.scale_bits());
                e.target_scale_bits = Some(target.to_bits());
                Err(e)
            }
        }
    }
    fn bias_extract(
        &mut self,
        src: CtMeta,
        plan: crate::compact::CompactPlan,
        position: usize,
        target: f64,
        need: usize,
    ) -> Result<CtMeta> {
        let mut level = src.level();
        let rot = u64::from(plan.offset(position) % self.g.layout().slots != 0)
            + plan.diags_per_ct.trailing_zeros() as u64;
        loop {
            let from = self.at(src, level)?;
            match self.exact("bias_extracted", from, target, None, 1, rot) {
                Err(e) if e.source == Some(HekitError::ScaleMismatch) && level > need + 1 => {
                    level -= 1
                }
                r => return r,
            }
        }
    }
    fn grad_i(
        &mut self,
        i: usize,
        dz: CtMeta,
        a: CtMeta,
        target: f64,
        reference: bool,
    ) -> Result<(CtMeta, Option<CtMeta>)> {
        let sh = self.g.shapes[i];
        let prod = self.product(dz, a)?;
        if prod.level() == 0 {
            return Err(self.fail(PlanFailure::LevelInadmissible, "weight gradient mask"));
        }
        let lb = self.g.layout().log_batch();
        let m = sh.m;
        let r = (sh.baby + sh.giant - 2) as u64 + times(m, lb)?;
        self.gradient_label_precheck(prod, target)?;
        self.emit(
            "gradient_product",
            prod,
            m,
            count(r, m as u64, m as u64, 0, m as u64),
        )?;
        let cur = self.exact("gradient", prod, target, None, m, times(m, lb)?)?;
        let rf = if reference {
            Some(self.exact("reference_gradient", prod, target, None, m, times(m, lb)?)?)
        } else {
            None
        };
        Ok((cur, rf))
    }
    fn bias_i(&mut self, dz: CtMeta, target: f64) -> Result<CtMeta> {
        let r = 2 * self.g.layout().log_batch() as u64;
        match self.exact("bias_gradient", dz, target, None, 1, r) {
            Err(e) if e.source == Some(HekitError::ScaleMismatch) && dz.level() >= 2 => self.exact(
                "bias_gradient",
                self.at(dz, dz.level() - 1)?,
                target,
                None,
                1,
                r,
            ),
            r => r,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Exact F64 coefficient bits with one logical-update denominator.
pub struct GradientCoefficients {
    /// Current folded weight coefficient bits.
    pub current_bits: u64,
    /// Additional reference folded coefficient bits, when that branch is taken.
    pub reference_bits: Option<u64>,
    /// Bias coefficient bits, zero for frozen biases.
    pub bias_bits: u64,
    /// Momentum scalar bits for later velocity updates.
    pub velocity_mu_bits: Option<u64>,
    /// Runtime-ordered rate-adjusted kappa scalar bits.
    pub velocity_kappa_bits: Option<u64>,
}
fn coefficients(
    o: DiagonalOptimizer,
    rate: f64,
    total: usize,
    policy: BiasPolicy,
    mode: DiagonalMode,
) -> Result<(GradientCoefficients, bool)> {
    validate_optimizer(o, rate)?;
    let n = total as f64;
    let (c, reference, mu, kappa) = match o {
        DiagonalOptimizer::Sgd { .. } => (rate / n, None, None, None),
        DiagonalOptimizer::Nag {
            reference,
            momentum,
            ..
        } => {
            let branch = match mode {
                DiagonalMode::Interactive => rate != reference,
                _ => rate / reference != 1.,
            };
            (
                (rate * (1. + momentum)) / n,
                branch.then_some((reference * (1. + momentum)) / n),
                Some(momentum),
                Some(crate::optim::nag_kappa(
                    momentum,
                    if branch { rate / reference } else { 1. },
                )),
            )
        }
    };
    let b = if policy == BiasPolicy::Frozen {
        0.
    } else {
        rate / n
    };
    if !c.is_finite() || !b.is_finite() || reference.is_some_and(|x| !x.is_finite()) {
        return Err(error(PlanFailure::RateConflict, "folded coefficients"));
    }
    Ok((
        GradientCoefficients {
            current_bits: c.to_bits(),
            reference_bits: reference.map(f64::to_bits),
            bias_bits: b.to_bits(),
            velocity_mu_bits: mu.map(f64::to_bits),
            velocity_kappa_bits: kappa.map(f64::to_bits),
        },
        reference.is_some(),
    ))
}
impl PlannedStageUpdate {
    /// Folded current/reference/bias/velocity coefficients as exact bits.
    pub fn coefficients(&self) -> &GradientCoefficients {
        &self.coefficients
    }
}

/// Plans one immutable-state transition, preserving existing runtime errors and decisions.
pub fn plan_stage_update(
    g: &StageGeometry,
    p: &CkksParams,
    boot: Option<&BootstrapMetadata>,
    s: &StageState,
    u: &UpdateSpec,
) -> Result<PlannedStageUpdate> {
    plan_stage_update_inner(g, p, boot, s, u).map_err(|mut e| {
        e.update = Some(s.updates);
        e
    })
}
fn plan_stage_update_inner(
    g: &StageGeometry,
    p: &CkksParams,
    boot: Option<&BootstrapMetadata>,
    s: &StageState,
    u: &UpdateSpec,
) -> Result<PlannedStageUpdate> {
    check_boot(p, g, boot)?;
    if s.geometry != g.identity
        || s.params != ParameterIdentity::new(p)
        || s.bootstrap.as_deref() != boot.map(|b| b.descriptor_identity_bytes())
    {
        return Err(error(PlanFailure::IdentityMismatch, "stage state identity"));
    }
    let mut total = 0usize;
    for m in &u.micros {
        if m.n_valid == 0 || m.n_valid > g.layout().batch {
            return Err(error(PlanFailure::InvalidMicroPartition, "valid lanes"));
        }
        total = total
            .checked_add(m.n_valid)
            .ok_or_else(|| error(PlanFailure::ArithmeticOverflow, "valid total"))?;
        CtMeta::new(p, m.x.level(), m.x.scale())
            .map_err(|_| error(PlanFailure::IdentityMismatch, "micro input metadata"))?;
        CtMeta::new(p, m.y.level(), m.y.scale())
            .map_err(|_| error(PlanFailure::IdentityMismatch, "micro target metadata"))?;
    }
    if total == 0 || total != u.total_valid {
        return Err(error(PlanFailure::InvalidMicroPartition, "logical total"));
    }
    let (coefs, reference) = coefficients(s.optim, u.rate, total, s.bias_policy, g.mode())?;
    let mut c = Planning::new(g, p, boot);
    let n = w_len(g);
    let mut hidden = vec![];
    let mut canonical = vec![];
    let mut next = s.clone();
    next.optim = with_rate(s.optim, u.rate);
    let first = is_nag(s.optim) && s.u.iter().all(Option::is_none);
    let mut refreshes = 0usize;
    let mut state_booted = false;
    match g.mode() {
        DiagonalMode::Interactive => {
            let mut gradients = vec![None; n];
            let mut references = vec![None; n];
            let mut biases = vec![None; n];
            for (mb, m) in u.micros.iter().enumerate() {
                c.micro = Some(mb);
                c.layer = None;
                c.note("entry_input", m.x)?;
                c.note("entry_target", m.y)?;
                let ForwardPass {
                    acts,
                    zs,
                    hidden: h,
                    canonical: ca,
                } = c.forward(s, m.x)?;
                hidden.push(h);
                canonical.push(ca);
                let mut delta = c.loss(acts[n], m.y, false)?;
                for i in (0..n).rev() {
                    c.layer = Some(i);
                    let dz = c.dz(
                        g.identity.acts[i],
                        zs[i],
                        delta,
                        m.n_valid == g.layout().batch,
                    )?;
                    let back = if i > 0 {
                        Some(c.backward(i, s.w[i], dz, true)?)
                    } else {
                        None
                    };
                    let (dw, rf) = c.grad_i(i, dz, acts[i], s.w[i].scale(), reference)?;
                    let db = c.bias_i(dz, s.b[i].scale())?;
                    let multiplicity = g.shapes[i].m;
                    let mut sum = |old: Option<CtMeta>,
                                   v: CtMeta,
                                   role: &str,
                                   copies: usize|
                     -> Result<CtMeta> {
                        let v = if let Some(a) = old {
                            let l = a.level().min(v.level());
                            c.sub(c.at(a, l)?, c.at(v, l)?, true, role)?
                        } else {
                            v
                        };
                        c.emit(role, v, copies, OpCounter::default())
                    };
                    gradients[i] =
                        Some(sum(gradients[i], dw, "accumulated_gradient", multiplicity)?);
                    if let Some(rf) = rf {
                        references[i] = Some(sum(
                            references[i],
                            rf,
                            "accumulated_reference",
                            multiplicity,
                        )?)
                    }
                    biases[i] = Some(sum(biases[i], db, "accumulated_bias", 1)?);
                    if let Some(back) = back {
                        delta = back
                    }
                }
            }
            c.micro = None;
            for i in (0..n).rev() {
                c.layer = Some(i);
                let b = biases[i].unwrap();
                next.b[i] = c.sub(c.at(s.b[i], b.level())?, b, false, "bias subtraction")?;
                c.note("bias_updated", next.b[i])?;
            }
            for i in (0..n).rev() {
                c.layer = Some(i);
                let dw = gradients[i].unwrap();
                let lv = dw.level();
                let mut w = c.sub(c.at(s.w[i], lv)?, dw, false, "weight subtraction")?;
                if let DiagonalOptimizer::Nag { momentum, .. } = s.optim {
                    let h = references[i].unwrap_or(dw);
                    let new_u = if let Some(old) = s.u[i] {
                        if old.level() <= lv || old.scale() != s.w[i].scale() {
                            return Err(c.fail(PlanFailure::LevelInadmissible, "NAG old velocity"));
                        }
                        let old = c.at(old, lv + 1)?;
                        let mu = c.exact(
                            "mu_velocity",
                            old,
                            s.w[i].scale(),
                            Some(momentum),
                            g.shapes[i].m,
                            0,
                        )?;
                        let k = c.exact(
                            "kappa_velocity",
                            old,
                            s.w[i].scale(),
                            Some(f64::from_bits(coefs.velocity_kappa_bits.unwrap())),
                            g.shapes[i].m,
                            0,
                        )?;
                        w = c.sub(w, k, false, "NAG weight subtraction")?;
                        c.sub(mu, h, false, "NAG velocity addition")?
                    } else {
                        h
                    };
                    next.u[i] = Some(new_u);
                    c.emit(
                        "velocity_updated",
                        new_u,
                        g.shapes[i].m,
                        OpCounter::default(),
                    )?;
                }
                next.w[i] = w;
                c.emit("weight_updated", w, g.shapes[i].m, OpCounter::default())?;
            }
            for i in 0..n {
                c.layer = Some(i);
                let m = g.shapes[i].m;
                refreshes = refreshes.checked_add(m + 1).ok_or_else(|| {
                    c.fail(PlanFailure::ArithmeticOverflow, "client refresh count")
                })?;
                next.w[i] = meta(p, p.max_level(), next.w[i].scale())?;
                next.b[i] = meta(p, p.max_level(), next.b[i].scale())?;
                c.emit("weight_refreshed", next.w[i], m, OpCounter::default())?;
                c.note("bias_refreshed", next.b[i])?;
                if let Some(v) = next.u[i] {
                    refreshes = refreshes.checked_add(m).ok_or_else(|| {
                        c.fail(PlanFailure::ArithmeticOverflow, "velocity refresh count")
                    })?;
                    let v = meta(p, p.max_level(), v.scale())?;
                    next.u[i] = Some(v);
                    c.emit("velocity_refreshed", v, m, OpCounter::default())?;
                }
            }
        }
        DiagonalMode::NonInteractive(placement) => {
            let b = boot.unwrap();
            let cl = g.compact.as_ref().unwrap();
            let metas = g
                .identity
                .acts
                .iter()
                .copied()
                .map(act_meta)
                .collect::<Vec<_>>();
            let start = b.output().level() - 1;
            let st = level_state(s);
            let full = u.micros.iter().all(|m| m.n_valid == g.layout().batch);
            let minx = u.micros.iter().map(|m| m.x.level()).min().unwrap();
            let pass = pass_levels(
                &metas,
                placement == BootPlacement::AfterActivation,
                &st.w,
                &st.b,
                minx as i64,
                b.output().level() as i64,
                full,
            );
            if pass.min < 0 {
                return Err(c.fail(PlanFailure::LevelInadmissible, "structural step precheck"));
            }
            let mut all_acts = vec![];
            let mut all_zs = vec![];
            for (mb, m) in u.micros.iter().enumerate() {
                c.micro = Some(mb);
                c.layer = None;
                c.note("entry_input", m.x)?;
                c.note("entry_target", m.y)?;
                let ForwardPass {
                    acts: a,
                    zs: z,
                    hidden: h,
                    canonical: ca,
                } = c.forward(s, m.x)?;
                all_acts.push(a);
                all_zs.push(z);
                hidden.push(h);
                canonical.push(ca);
            }
            let mut deltas = vec![];
            for (mb, m) in u.micros.iter().enumerate() {
                c.micro = Some(mb);
                deltas.push(c.loss(all_acts[mb][n], m.y, true)?);
            }
            let mut hw = vec![vec![]; n];
            let mut hr = vec![vec![]; n];
            let mut hb = vec![None; n];
            for i in (0..n).rev() {
                c.layer = Some(i);
                let mut dzs = vec![];
                for (mb, m) in u.micros.iter().enumerate() {
                    c.micro = Some(mb);
                    dzs.push(c.dz(
                        g.identity.acts[i],
                        all_zs[mb][i],
                        deltas[mb],
                        m.n_valid == g.layout().batch,
                    )?);
                }
                let low = dzs.iter().map(|d| d.level()).min().unwrap();
                for (mb, d) in dzs.iter_mut().enumerate() {
                    c.micro = Some(mb);
                    *d = c.at(*d, low)?;
                    c.note("dz_aligned", *d)?;
                }
                c.micro = None;
                if i > 0 {
                    let wt = c.at(s.w[i], low)?;
                    let rot = g.transpose[i].iter().map(|(_, v)| v.len() as u64).sum();
                    c.emit("transpose", wt, g.shapes[i].m, count(rot, 0, 0, 0, 0))?;
                    for (mb, d) in dzs.iter().enumerate() {
                        c.micro = Some(mb);
                        deltas[mb] = c.backward(i, s.w[i], *d, false)?;
                    }
                }
                c.micro = None;
                let a = all_acts.iter().map(|a| a[i]).collect::<Vec<_>>();
                for (name, values) in [("delta", &dzs), ("layer input", &a)] {
                    if let Some(other) = values
                        .iter()
                        .find(|m| m.level() != values[0].level() || m.scale() != values[0].scale())
                    {
                        let mut e = c.fail(
                            PlanFailure::MicroMetadataMismatch,
                            "weight_grad_reduced_consts/check_micro",
                        );
                        e.source = Some(HekitError::InvalidParams(format!(
                            "the {name} micro-batches must share one level and one scale"
                        )));
                        e.source_scale_bits = Some(other.scale_bits());
                        e.target_scale_bits = Some(values[0].scale_bits());
                        e.prime = p.q().get(other.level()).copied();
                        return Err(e);
                    }
                }
                let sh = g.shapes[i];
                let k = u.micros.len();
                let m = sh.m;
                let lb = g.layout().log_batch();
                let prod = c.product(dzs[0], a[0])?;
                if prod.level() == 0 {
                    return Err(c.fail(PlanFailure::LevelInadmissible, "reduced gradient mask"));
                }
                let r = times(k, sh.baby + sh.giant - 2)?
                    .checked_add(times(m, lb)?)
                    .ok_or_else(|| c.fail(PlanFailure::ArithmeticOverflow, "reduced rotations"))?;
                c.gradient_label_precheck(prod, s.w[i].scale())?;
                c.emit(
                    "gradient_product",
                    prod,
                    m,
                    count(
                        r,
                        if g.eval_config().lazy_relin {
                            m as u64
                        } else {
                            times(k, m)?
                        },
                        times(k, m)?,
                        0,
                        m as u64,
                    ),
                )?;
                let cur = c.exact("gradient", prod, s.w[i].scale(), None, m, 0)?;
                let plan = cl.plans[i];
                c.emit(
                    "compact_gradient",
                    cur,
                    plan.ciphertexts(),
                    count(plan.pack_rotations() as u64, 0, 0, 0, 0),
                )?;
                hw[i] = vec![cur; plan.ciphertexts()];
                if reference {
                    let rf = c.exact("reference_gradient", prod, s.w[i].scale(), None, m, 0)?;
                    c.emit(
                        "compact_reference",
                        rf,
                        plan.ciphertexts(),
                        count(plan.pack_rotations() as u64, 0, 0, 0, 0),
                    )?;
                    hr[i] = vec![rf; plan.ciphertexts()];
                }
                let slot = cl.biases[i];
                let bp = cl.bias_plan(i);
                let gb = c.exact("bias_gradient", dzs[0], p.scale(), None, 1, lb as u64)?;
                let r = u64::from(bp.offset(slot.position) % g.layout().slots != 0);
                c.emit("compact_bias_gradient", gb, 1, count(r, 0, 0, 0, 0))?;
                hb[i] = Some(gb);
            }
            c.micro = None;
            let mut cs = vec![];
            let mut us = vec![];
            for (ci, spec) in cl.containers.iter().enumerate() {
                c.layer = spec.layer;
                let old = s.c[ci];
                let h = spec.layer.map(|l| hw[l][spec.index]);
                let hs = cl
                    .biases
                    .iter()
                    .enumerate()
                    .filter(|(_, slot)| slot.container == ci)
                    .map(|(l, _)| hb[l].unwrap())
                    .collect::<Vec<_>>();
                let ov = if spec.layer.is_some() { s.u[ci] } else { None };
                let mut lv = old.level();
                if let Some(h) = h {
                    lv = lv.min(h.level())
                }
                for h in &hs {
                    lv = lv.min(h.level())
                }
                if is_nag(s.optim) && spec.layer.is_some() {
                    if let Some(v) = ov {
                        if v.level() == 0 {
                            return Err(c.fail(PlanFailure::LevelInadmissible, "compact velocity"));
                        }
                        lv = lv.min(v.level() - 1)
                    }
                }
                let mut w = c.at(old, lv)?;
                if let Some(h) = h {
                    w = c.sub(
                        w,
                        c.at(h, lv)?,
                        false,
                        "compact weight gradient subtraction",
                    )?
                }
                for h in hs {
                    w = c.sub(w, c.at(h, lv)?, false, "compact bias subtraction")?
                }
                let new_u = if let (DiagonalOptimizer::Nag { momentum, .. }, Some(h)) = (s.optim, h)
                {
                    let l = spec.layer.unwrap();
                    let h = if reference { hr[l][spec.index] } else { h };
                    let h = c.at(h, lv)?;
                    Some(if let Some(ov) = ov {
                        let v = c.at(ov, lv + 1)?;
                        let mu = c.exact(
                            &format!("container{ci}_mu_velocity"),
                            v,
                            p.scale(),
                            Some(momentum),
                            1,
                            0,
                        )?;
                        let k = c.exact(
                            &format!("container{ci}_kappa_velocity"),
                            v,
                            p.scale(),
                            Some(f64::from_bits(coefs.velocity_kappa_bits.unwrap())),
                            1,
                            0,
                        )?;
                        w = c.sub(w, k, false, "compact NAG subtraction")?;
                        c.sub(mu, h, false, "compact NAG velocity addition")?
                    } else {
                        h
                    })
                } else {
                    None
                };
                c.note(&format!("container{ci}_updated"), w)?;
                if let Some(v) = new_u {
                    c.note(&format!("container{ci}_velocity"), v)?;
                }
                cs.push(w);
                us.push(new_u);
            }
            let levels = cs.iter().map(|m| m.level() as i64).collect::<Vec<_>>();
            let (ws, bs) = unpacked(cl, &levels);
            let structural = LevelState {
                w: ws,
                b: bs,
                c: levels,
                u: us.iter().map(|u| u.map(|m| m.level() as i64)).collect(),
            };
            state_booted = must_boot(
                &metas,
                placement,
                &structural,
                start as i64,
                b.output().level() as i64,
                u.next_full,
            );
            if state_booted {
                for (i, w) in cs.iter_mut().enumerate() {
                    c.layer = None;
                    *w = c.bootstrap(&format!("container{i}_boot"), *w)?;
                }
                for (i, v) in us.iter_mut().enumerate() {
                    if let Some(vv) = v {
                        *vv = c.bootstrap(&format!("velocity{i}_boot"), *vv)?;
                    }
                }
            }
            let levels = cs.iter().map(|m| m.level() as i64).collect::<Vec<_>>();
            let (ws, bs) = unpacked(cl, &levels);
            let np = pass_levels(
                &metas,
                placement == BootPlacement::AfterActivation,
                &ws,
                &bs,
                start as i64,
                b.output().level() as i64,
                true,
            );
            for i in 0..n {
                c.layer = Some(i);
                let range = cl.layer_containers(i);
                let low = range.clone().map(|j| cs[j].level()).min().unwrap();
                let first = cs[range.start];
                if range.clone().any(|j| cs[j].scale() != first.scale()) {
                    return Err(c.fail(PlanFailure::ScaleInadmissible, "unpack uniform scale"));
                }
                let src = c.at(first, low)?;
                let plan = cl.plans[i];
                next.w[i] = c.exact(
                    "weight_unpacked",
                    src,
                    src.scale(),
                    None,
                    plan.diagonals(),
                    plan.unpack_rotations() as u64,
                )?;
            }
            let scales = next_input_scales(g, p, &next.w, &np.inputs)?;
            for (i, &scale) in scales.iter().enumerate() {
                c.layer = Some(i);
                let slot = cl.biases[i];
                let plan = cl.containers[slot.container].plan;
                let src = cs[slot.container];
                let target = scale * next.w[i].scale();
                let need = np.inputs[i].max(0) as usize;
                next.b[i] = c.bias_extract(src, plan, slot.position, target, need)?;
            }
            next.c = cs;
            next.u = us;
        }
    }
    if g.mode() != DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
        canonical.clear();
    }
    next.updates = next
        .updates
        .checked_add(1)
        .ok_or_else(|| c.fail(PlanFailure::ArithmeticOverflow, "update count"))?;
    let mut including_boot = c.counts.clone();
    if let Some(b) = boot {
        for _ in &c.boots {
            merge(&mut including_boot, b.cost_per_boot())?;
        }
    }
    Ok(PlannedStageUpdate {
        state: next,
        counts: c.counts,
        including_boot,
        reference,
        first,
        refreshes,
        events: c.trace,
        attempts: c.attempts,
        boots: c.boots,
        hidden,
        canonical,
        booted: state_booted,
        coefficients: coefs,
    })
}
fn level_state(s: &StageState) -> LevelState {
    LevelState {
        w: s.w.iter().map(|m| m.level() as i64).collect(),
        b: s.b.iter().map(|m| m.level() as i64).collect(),
        c: s.c.iter().map(|m| m.level() as i64).collect(),
        u: s.u.iter().map(|u| u.map(|m| m.level() as i64)).collect(),
    }
}

#[derive(Clone, Debug)]
/// Old first-hidden representations and separate boundary work.
pub struct PlannedHandoff {
    /// One canonical boundary input label per micro.
    pub outputs: Vec<CtMeta>,
    /// Boundary conversion counts excluding bootstrap internal work.
    pub counts_nonbootstrap: OpCounter,
    /// Boundary counts with parameter-specific bootstrap work included once.
    pub counts_including_bootstrap: OpCounter,
    /// Boundary client refresh calls.
    pub client_refreshes: usize,
    /// Boundary bootstrap events; AA reuse has none.
    pub bootstraps: Vec<BootstrapEvent>,
}
/// Plans old first-hidden handoffs, reusing AA canonical outputs exactly once.
pub fn plan_stage_handoff(
    g: &StageGeometry,
    p: &CkksParams,
    boot: Option<&BootstrapMetadata>,
    update: &PlannedStageUpdate,
    destination: &StageGeometry,
) -> Result<PlannedHandoff> {
    if (g.mode() == DiagonalMode::Interactive) != (destination.mode() == DiagonalMode::Interactive)
    {
        return Err(error(
            PlanFailure::IdentityMismatch,
            "mixed interactive/noninteractive boundary",
        ));
    }
    if update.state.geometry != g.identity
        || update.state.params != ParameterIdentity::new(p)
        || g.identity.layout != destination.identity.layout
        || g.dims().len() < 3
        || destination.dims()[0] != g.dims()[1]
    {
        return Err(error(
            PlanFailure::IdentityMismatch,
            "stage handoff geometry",
        ));
    }
    check_boot(p, g, boot)?;
    let mut c = Planning::new(g, p, boot);
    let mut outputs = vec![];
    let mut refreshes = 0;
    for (mb, h) in update.hidden.iter().enumerate() {
        c.micro = Some(mb);
        let out = if g.mode() == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
            *update.canonical[mb]
                .first()
                .ok_or_else(|| c.fail(PlanFailure::MissingCode, "canonical first hidden"))?
        } else {
            let h = *h
                .first()
                .ok_or_else(|| c.fail(PlanFailure::MissingCode, "first hidden"))?;
            let d = c.exact("handoff_to_delta", h, p.scale(), Some(1.), 1, 0)?;
            if g.mode() == DiagonalMode::Interactive {
                refreshes += 1;
                meta(p, p.max_level(), d.scale())?
            } else {
                c.bootstrap("handoff_boot", d)?
            }
        };
        outputs.push(out);
    }
    let mut including = c.counts.clone();
    if let Some(b) = boot {
        for _ in &c.boots {
            merge(&mut including, b.cost_per_boot())?
        }
    }
    Ok(PlannedHandoff {
        outputs,
        counts_nonbootstrap: c.counts,
        counts_including_bootstrap: including,
        client_refreshes: refreshes,
        bootstraps: c.boots,
    })
}

#[cfg(test)]
mod pure_tests {
    use super::*;
    use ckks_train_ckks::metadata::exact_product_scale_at;
    #[test]
    fn p2_pure_mse_rounding_boundary_keeps_existing_runtime_rule() {
        let p = CkksParams::new(10, &[60, 40, 40, 40], &[60], 40, 4).unwrap();
        let g = plan_stage_geometry(
            &[2, 2],
            &Layout::new(512, 2, 8).unwrap(),
            &[DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
        )
        .unwrap();
        let out = meta(&p, 2, p.scale()).unwrap();
        for (factor, ok) in [(1.0, true), (1.0 + 0.5e-12, true), (1.0 + 2.0e-12, false)] {
            let y = meta(&p, 3, p.scale() * factor).unwrap();
            let mut c = Planning::new(&g, &p, None);
            let result = c.loss(out, y, false);
            assert_eq!(result.is_ok(), ok);
            if let Ok(m) = result {
                assert_eq!(m, out)
            }
        }
    }
    #[test]
    fn p2_pure_exact_retry_controls_and_prework_failure_counts() {
        let p = CkksParams::new(12, &[60, 20, 20], &[60], 20, 3).unwrap();
        let g = plan_stage_geometry(
            &[2, 2],
            &Layout::new(p.slots(), 4, 8).unwrap(),
            &[DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
        )
        .unwrap();
        let src = meta(&p, 2, p.scale()).unwrap();
        for (bits, ok) in [
            (0x4135_eb85_1ebf_b0df, true),
            (0x4135_eb85_1ec1_0524, false),
        ] {
            let mut c = Planning::new(&g, &p, None);
            let r = c.bias_i(src, f64::from_bits(bits));
            assert_eq!(r.is_ok(), ok);
            assert_eq!(c.attempts.len(), 2);
            assert_eq!(
                c.counts,
                if ok {
                    count(6, 0, 0, 1, 1)
                } else {
                    OpCounter::default()
                }
            );
            let mut c = Planning::new(&g, &p, None);
            let r = c.bias_extract(
                src,
                crate::compact::CompactPlan::for_layer(&g.layout(), 2, 2),
                0,
                f64::from_bits(bits),
                0,
            );
            assert_eq!(r.is_ok(), ok);
            assert_eq!(c.attempts.len(), 2);
            assert_eq!(
                c.counts,
                if ok {
                    count(10, 0, 0, 1, 1)
                } else {
                    OpCounter::default()
                }
            );
        }
        let mut c = Planning::new(&g, &p, None);
        let e = c
            .grad_i(0, src, src, f64::from_bits(0x4135_eb85_1eb8_aeb9), false)
            .unwrap_err();
        assert_eq!(e.source, Some(HekitError::ScaleMismatch));
        assert_eq!(e.operation, "gradient_mask_precheck");
        assert_eq!(c.counts, OpCounter::default());
    }
    #[test]
    fn p2_pure_scalar_errors_have_prime_and_scale_context() {
        let p = CkksParams::new(10, &[60, 40, 40, 40], &[60], 40, 4).unwrap();
        let g = plan_stage_geometry(
            &[2, 2],
            &Layout::new(p.slots(), 2, 8).unwrap(),
            &[DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
        )
        .unwrap();
        for (value, target) in [(1e100, p.scale()), (1., p.scale() * 2f64.powi(-30))] {
            let mut c = Planning::new(&g, &p, None);
            let src = meta(&p, 2, p.scale()).unwrap();
            let e = c
                .exact("scalar", src, target, Some(value), 1, 0)
                .unwrap_err();
            assert_eq!(e.kind, PlanFailure::ScaleInadmissible);
            assert_eq!(e.prime, Some(p.q()[2]));
            assert_eq!(e.source_scale_bits, Some(src.scale_bits()));
            assert_eq!(e.target_scale_bits, Some(target.to_bits()));
            assert_eq!(c.counts, OpCounter::default());
        }
    }
    #[test]
    fn p2_pure_retry_witnesses_frozen_before_he() {
        let p = CkksParams::new(12, &[60, 20, 20], &[60], 20, 3).unwrap();
        assert_eq!(&p.q()[1..], &[974849, 1032193]);
        const RETRY: u64 = 0x4135_eb85_1ebf_b0df;
        const EXHAUSTED: u64 = 0x4135_eb85_1ec1_0524;
        let target = f64::from_bits(RETRY);
        assert_eq!(
            exact_product_scale_at(&p, 2, target),
            Err(HekitError::ScaleMismatch)
        );
        let plain = target_plain_scale_at(&p, 1, p.scale(), target).unwrap();
        assert_eq!(plain.to_bits(), 0x4134_60f7_214e_885b);
        assert_eq!(
            exact_product_scale_at(&p, 1, target).unwrap().to_bits(),
            0x4274_60f7_214e_885b
        );
        for level in [1, 2] {
            assert_eq!(
                exact_product_scale_at(&p, level, f64::from_bits(EXHAUSTED)),
                Err(HekitError::ScaleMismatch)
            );
        }
        assert!(plain.is_finite() && plain >= 1048576.);
        let q1_bound = 2f64.powf(p.q()[..2].iter().map(|q| (*q as f64).log2()).sum::<f64>() - 1.);
        assert!(2. * plain < q1_bound); // conservative inverse-transform coefficient bound
        assert!((0.125 * plain).round().abs() < 2f64.powi(62));
        assert!(2. * p.scale() < p.q()[0] as f64 / 2.);
    }
}

/// Private test observation of real runtime values. No planner or evaluator calls.
#[cfg(test)]
pub(crate) mod observation {
    use ckks_train_ckks::DiagonalEntry;
    use std::{cell::RefCell, rc::Rc};
    #[derive(Clone, Debug)]
    pub(crate) struct Observed {
        /// Explicit ciphertext role name.
        pub role: &'static str,
        /// Zero-based micro index; None for shared logical-update work.
        pub micro: Option<usize>,
        /// Zero-based producing layer; None for container/shared work.
        pub layer: Option<usize>,
        pub position: usize,
        /// Key generation level, over the complete q prefix.
        pub level: usize,
        pub scale_bits: u64,
    }
    #[derive(Default, Debug)]
    pub(crate) struct Recorder {
        pub values: Vec<Observed>,
        /// Completed exact-target attempts including failed retries.
        pub attempts: Vec<Attempt>,
        pub coefficients: Vec<Coefficient>,
        /// Zero-based micro index; None for shared logical-update work.
        pub micro: Option<usize>,
        /// Zero-based producing layer; None for container/shared work.
        pub layer: Option<usize>,
    }
    #[derive(Clone, Debug)]
    pub(crate) struct Attempt {
        /// Explicit ciphertext role name.
        pub role: &'static str,
        /// Zero-based micro index; None for shared logical-update work.
        pub micro: Option<usize>,
        /// Zero-based producing layer; None for container/shared work.
        pub layer: Option<usize>,
        pub source_level: usize,
        /// Exact source label when a scale error is reported.
        pub source_scale_bits: u64,
        /// Exact requested target label.
        pub target_bits: u64,
        /// Actual scale-planning error, or None on success.
        pub error: Option<ckks_train_core::error::HekitError>,
    }
    #[derive(Clone, Debug)]
    pub(crate) struct Coefficient {
        pub role: &'static str,
        pub layer: Option<usize>,
        pub position: usize,
        pub bits: u64,
    }
    pub(crate) type Handle = Rc<RefCell<Recorder>>;
    pub(crate) fn value<C: DiagonalEntry>(
        sink: Option<&Handle>,
        role: &'static str,
        micro: Option<usize>,
        layer: Option<usize>,
        position: usize,
        ct: &C,
    ) {
        if let Some(sink) = sink {
            sink.borrow_mut().values.push(Observed {
                role,
                micro,
                layer,
                position,
                level: ct.level(),
                scale_bits: ct.scale().to_bits(),
            });
        }
    }
    pub(crate) fn coefficient(
        sink: Option<&Handle>,
        role: &'static str,
        layer: Option<usize>,
        position: usize,
        value: f64,
    ) {
        if let Some(sink) = sink {
            sink.borrow_mut().coefficients.push(Coefficient {
                role,
                layer,
                position,
                bits: value.to_bits(),
            });
        }
    }
    pub(crate) fn scope(sink: Option<&Handle>, micro: Option<usize>, layer: Option<usize>) {
        if let Some(sink) = sink {
            let mut r = sink.borrow_mut();
            r.micro = micro;
            r.layer = layer;
        }
    }
    pub(crate) fn attempt<C: DiagonalEntry, T>(
        sink: Option<&Handle>,
        role: &'static str,
        source: &C,
        target: f64,
        result: &ckks_train_core::error::Result<T>,
    ) {
        if let Some(sink) = sink {
            let mut recorder = sink.borrow_mut();
            let micro = recorder.micro;
            let layer = recorder.layer;
            recorder.attempts.push(Attempt {
                role,
                micro,
                layer,
                source_level: source.level(),
                source_scale_bits: source.scale().to_bits(),
                target_bits: target.to_bits(),
                error: result.as_ref().err().cloned(),
            });
        }
    }
    pub(crate) fn values<C: DiagonalEntry>(
        sink: Option<&Handle>,
        role: &'static str,
        micro: Option<usize>,
        layer: Option<usize>,
        cts: &[C],
    ) {
        for (i, ct) in cts.iter().enumerate() {
            value(sink, role, micro, layer, i, ct);
        }
    }
}

#[cfg(test)]
#[path = "diagonal_plan_tests.rs"]
pub(crate) mod tests;
