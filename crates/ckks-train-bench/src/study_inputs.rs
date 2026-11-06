//! Canonical host-only study arrays, row orders and versioned content identities.
//!
//! [`materialize`] is the only production constructor. It consumes exactly one canonical
//! preparation and freezes all values behind immutable views. F32 values are eagerly cast
//! once; neither numeric view trains, allocates HE state, nor establishes an FP32 target.
use crate::{
    data::sha256_hex,
    profile_data::{DataRequest, DataSource, Dataset, OrderCoverage, VerifiedSeed},
    study_profile::{
        Activation, BatchPlan, Distribution, Initialization, LayerRole, Loss, Model, Optimizer,
        OrderPolicy, PlannedUpdate, PreparedDataset, PreparedInputs, ProfileError, ProfileMetadata,
        RateUnits, ResolvedProfile, RngConvention,
    },
};
use ckks_train_core::sampler::Rng;
use ndarray::Array2;
use std::{collections::BTreeSet, fmt, ops::Range, path::Path};

/// Encoding and local recipe implementation version; changes require a new identity.
pub const IDENTITY_VERSION: &str = "hekit-study-inputs-v1";
/// Host-side input failure, never a successful scientific/reference observation.
#[derive(Debug)]
pub enum StudyInputError {
    /// Canonical source/data preparation failed, retaining its typed missing/conflict error.
    Preparation(ProfileError),
    /// Prepared contents, graph, order or numerical cast violate the contract.
    Invalid(String),
}
impl fmt::Display for StudyInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for StudyInputError {}
impl From<ProfileError> for StudyInputError {
    fn from(e: ProfileError) -> Self {
        Self::Preparation(e)
    }
}
type Result<T> = std::result::Result<T, StudyInputError>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(StudyInputError::Invalid(message.into()))
    }
}
/// Numerical storage type, independent from any future arithmetic implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Canonical binary64 arrays.
    F64,
    /// Once-rounded binary32 arrays.
    F32,
}
/// Transformation from canonical source values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastPolicy {
    /// Canonical F64 preprocessing and initialization, without a numeric recast.
    CanonicalF64,
    /// Rust `as f32` exactly once from the canonical F64 values.
    RoundToF32Once,
}
/// One named logical dense layer, always output-by-input row-major.
#[derive(Debug)]
pub struct InitialLayer<T> {
    name: String,
    weights: Array2<T>,
    biases: Vec<T>,
}
impl<T> InitialLayer<T> {
    /// Registered layer name in canonical parameter order.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Logical output-by-input matrix; captured matrices were already transposed by D2b1.
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }
    /// Active or explicitly materialized zero frozen biases.
    pub fn biases(&self) -> &[T] {
        &self.biases
    }
}
/// Ordered initial parameters shared by all future adapters.
#[derive(Debug)]
pub struct InitialModel<T> {
    layers: Vec<InitialLayer<T>>,
}
impl<T> InitialModel<T> {
    /// Forward, local-head, then output-stack order, never regrouped by width.
    pub fn layers(&self) -> &[InitialLayer<T>] {
        &self.layers
    }
}
/// Input source for one independent loss stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageInput {
    /// Canonical data rows.
    Dataset,
    /// Representation from this layer's weights before the logical update.
    PreUpdateActivation {
        /// Canonical layer index; never a local classification head.
        layer: usize,
    },
}
/// Independent two-layer loss stage; gradients stop at its input.
#[derive(Debug)]
pub struct Stage {
    layers: Range<usize>,
    input: StageInput,
    representation_layer: Option<usize>,
}
impl Stage {
    /// Contiguous canonical parameter range.
    pub fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }
    /// Dataset or pre-update upstream representation.
    pub fn input(&self) -> StageInput {
        self.input
    }
    /// Forward layer exported downstream; local heads are ignored by inference.
    pub fn representation_layer(&self) -> Option<usize> {
        self.representation_layer
    }
}
/// Validated independent-stage topology, including old-weight handoffs and loss barriers.
#[derive(Debug)]
pub struct StageGraph {
    stages: Vec<Stage>,
}
impl StageGraph {
    /// Stages in registered order, one independent target/loss per stage.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }
}
/// Full consumed canonical-row order with offsets, without expanded feature batches.
#[derive(Debug)]
pub struct MaterializedOrder {
    rows: Vec<u32>,
    pass_offsets: Vec<usize>,
}
impl MaterializedOrder {
    /// Flat consumed row IDs into `Dataset::train_x`; these are not source-row IDs.
    pub fn rows(&self) -> &[u32] {
        &self.rows
    }
    /// Start/end offsets for each retained logical pass; captured stream remains flat.
    pub fn pass_offsets(&self) -> &[usize] {
        &self.pass_offsets
    }
    /// Retained rows of a declared pass (before any cross-pass update grouping).
    pub fn pass_rows(&self, pass: usize) -> Option<&[u32]> {
        let start = *self.pass_offsets.get(pass)?;
        let end = *self.pass_offsets.get(pass.checked_add(1)?)?;
        self.rows.get(start..end)
    }
}
/// Typed coefficients derived from the authoritative full D2b1 plan.
#[derive(Debug, Clone, Copy)]
pub struct RateCoefficients<T> {
    /// Original captured/constant coefficient.
    pub source: T,
    /// Rate multiplying the logical batch-mean gradient.
    pub mean: T,
    /// Explicit registered loss derivative multiplier.
    pub delta: T,
    /// Mean rate times delta, for future encrypted MSE kernels.
    pub kernel: T,
}
/// Borrowed immutable numerical inputs. This is storage provenance, not F32/F64 training.
#[derive(Debug)]
pub struct NumericInputs<'a, T> {
    train_x: &'a Array2<T>,
    test_x: &'a Array2<T>,
    train_y: &'a [u8],
    test_y: &'a [u8],
    initial: &'a InitialModel<T>,
    rates: &'a [RateCoefficients<T>],
    reference: T,
    dtype: Dtype,
    cast_policy: CastPolicy,
}
impl<T: Copy> NumericInputs<'_, T> {
    /// Canonical training features (or their single F32 cast).
    pub fn train_x(&self) -> &Array2<T> {
        self.train_x
    }
    /// Canonical held-out features (or their single F32 cast).
    pub fn test_x(&self) -> &Array2<T> {
        self.test_x
    }
    /// Real class labels, with padded targets constructed later by D1's target helper.
    pub fn train_y(&self) -> &[u8] {
        self.train_y
    }
    /// Real held-out class labels.
    pub fn test_y(&self) -> &[u8] {
        self.test_y
    }
    /// Shared canonical initialization in this storage dtype.
    pub fn initial(&self) -> &InitialModel<T> {
        self.initial
    }
    /// Full plan coefficients, even when a future quick run consumes a prefix.
    pub fn rates(&self) -> &[RateCoefficients<T>] {
        self.rates
    }
    /// Maximum kernel coefficient over the full registered plan.
    pub fn reference_kernel_rate(&self) -> T {
        self.reference
    }
    /// Explicit dtype.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Explicit F64 origin/cast policy.
    pub fn cast_policy(&self) -> CastPolicy {
        self.cast_policy
    }
}
#[derive(Debug)]
struct F32Storage {
    train_x: Array2<f32>,
    test_x: Array2<f32>,
    initial: InitialModel<f32>,
    rates: Vec<RateCoefficients<f32>>,
    reference: f32,
}
/// Exact rows and already planned metadata of one logical optimizer update.
#[derive(Debug)]
pub struct UpdateRows<'a> {
    planned: &'a PlannedUpdate,
    rows: &'a [u32],
    pass: Option<usize>,
    crossed: &'a [usize],
}
impl UpdateRows<'_> {
    /// Authoritative D2b1 update; the accessor never calculates rates.
    pub fn planned(&self) -> &PlannedUpdate {
        self.planned
    }
    /// Exact row IDs in consumed order.
    pub fn rows(&self) -> &[u32] {
        self.rows
    }
    /// Zero-based pass if all rows belong to one pass; None for a crossing update.
    pub fn pass(&self) -> Option<usize> {
        self.pass
    }
    /// One-based declared logical boundaries crossed by this actual update.
    pub fn crossed_boundaries(&self) -> &[usize] {
        self.crossed
    }
}
/// Expected and verified source content. Equality is integrity, not upstream attestation.
#[derive(Debug)]
pub struct SourceDigest {
    /// Repository-relative source path.
    pub path: String,
    /// Expected digest supplied by the prepared binding.
    pub expected_sha256: String,
    /// Digest verified during canonical preparation.
    pub actual_sha256: String,
}
/// Immutable content identity, with independent array/order/plan/dtype digests.
#[derive(Debug)]
pub struct MaterializedIdentity {
    digest: String,
    weights: String,
    biases: String,
    order: String,
    plan: String,
    f64_inputs: String,
    f32_inputs: String,
    dataset: String,
    split: String,
    preprocessing: String,
    sources: Vec<SourceDigest>,
    code: String,
    canonical_prepared: bool,
}
impl MaterializedIdentity {
    /// Combined versioned input/content/source identity; not a measurement or attestation.
    pub fn digest(&self) -> &str {
        &self.digest
    }
    /// Named canonical F64 initialized weight matrices.
    pub fn weights_sha256(&self) -> &str {
        &self.weights
    }
    /// Named canonical F64 initialized bias vectors.
    pub fn biases_sha256(&self) -> &str {
        &self.biases
    }
    /// Flat consumed u32 order and u64 pass offsets.
    pub fn order_sha256(&self) -> &str {
        &self.order
    }
    /// Full authoritative plan and checkpoints.
    pub fn plan_sha256(&self) -> &str {
        &self.plan
    }
    /// Canonical F64 features, labels, parameters and coefficients.
    pub fn f64_inputs_sha256(&self) -> &str {
        &self.f64_inputs
    }
    /// Once-cast F32 features, labels, parameters and coefficients.
    pub fn f32_inputs_sha256(&self) -> &str {
        &self.f32_inputs
    }
    /// Canonical dataset numeric arrays and labels.
    pub fn dataset_sha256(&self) -> &str {
        &self.dataset
    }
    /// Actual source split indices and split description.
    pub fn split_sha256(&self) -> &str {
        &self.split
    }
    /// Registered and actual preprocessing descriptions.
    pub fn preprocessing_sha256(&self) -> &str {
        &self.preprocessing
    }
    /// Expected and verified file digests including registry, portable lock and captured files.
    pub fn sources(&self) -> &[SourceDigest] {
        &self.sources
    }
    /// Embedded materializer/dependency source and Cargo.lock digest plus package/target tags.
    /// A binary digest and Git revision are not available at this pure library boundary.
    pub fn code_build_sha256(&self) -> &str {
        &self.code
    }
    /// True only for the public one-preparation production path; fixtures remain non-scientific.
    pub fn canonical_prepared(&self) -> bool {
        self.canonical_prepared
    }
}
/// Owned immutable canonical host inputs; only [`materialize`] constructs production values.
#[derive(Debug)]
pub struct MaterializedProfile {
    metadata: ProfileMetadata,
    graph: StageGraph,
    dataset: Dataset,
    initial: InitialModel<f64>,
    order: MaterializedOrder,
    plan: BatchPlan,
    identity: MaterializedIdentity,
    f32: F32Storage,
    rates: Vec<RateCoefficients<f64>>,
    reference: f64,
}
impl MaterializedProfile {
    /// Registered immutable metadata, including layer names, activation and bias roles.
    pub fn metadata(&self) -> &ProfileMetadata {
        &self.metadata
    }
    /// Validated stage topology with pre-update representation handoffs.
    pub fn graph(&self) -> &StageGraph {
        &self.graph
    }
    /// Canonical D1 dataset including original source split indices.
    pub fn dataset(&self) -> &Dataset {
        &self.dataset
    }
    /// Canonical initialized F64 arrays.
    pub fn initial(&self) -> &InitialModel<f64> {
        &self.initial
    }
    /// Complete retained order.
    pub fn order(&self) -> &MaterializedOrder {
        &self.order
    }
    /// Full prepared plan, never shortened for quick consumers.
    pub fn plan(&self) -> &BatchPlan {
        &self.plan
    }
    /// Frozen content/source identity.
    pub fn identity(&self) -> &MaterializedIdentity {
        &self.identity
    }
    /// Maximum full-plan kernel coefficient, before any consumer chooses a prefix.
    pub fn reference_kernel_rate_f64(&self) -> f64 {
        self.reference
    }
    /// Borrow the already once-cast F32 inputs; repeated calls do no casts or RNG work.
    pub fn numeric_f32(&self) -> NumericInputs<'_, f32> {
        NumericInputs {
            train_x: &self.f32.train_x,
            test_x: &self.f32.test_x,
            train_y: &self.dataset.train_y,
            test_y: &self.dataset.test_y,
            initial: &self.f32.initial,
            rates: &self.f32.rates,
            reference: self.f32.reference,
            dtype: Dtype::F32,
            cast_policy: CastPolicy::RoundToF32Once,
        }
    }
    /// Borrow canonical F64 inputs, with no regeneration or preprocessing.
    pub fn numeric_f64(&self) -> NumericInputs<'_, f64> {
        NumericInputs {
            train_x: &self.dataset.train_x,
            test_x: &self.dataset.test_x,
            train_y: &self.dataset.train_y,
            test_y: &self.dataset.test_y,
            initial: &self.initial,
            rates: &self.rates,
            reference: self.reference,
            dtype: Dtype::F64,
            cast_policy: CastPolicy::CanonicalF64,
        }
    }
    /// Return exact already-planned rows, pass membership and crossed boundaries.
    pub fn update_rows(&self, index: usize) -> Result<UpdateRows<'_>> {
        let u = self
            .plan
            .updates
            .get(index)
            .ok_or_else(|| StudyInputError::Invalid("update index out of range".into()))?;
        let end = u.processed_start + u.valid_samples;
        let a = self
            .order
            .pass_offsets
            .partition_point(|&v| v <= u.processed_start)
            - 1;
        let b = self.order.pass_offsets.partition_point(|&v| v < end) - 1;
        let crossed = self
            .plan
            .checkpoints
            .iter()
            .find(|c| c.completed_updates == index + 1)
            .map_or(&[][..], |c| c.crossed_logical_boundaries.as_slice());
        Ok(UpdateRows {
            planned: u,
            rows: &self.order.rows[u.processed_start..end],
            pass: (a == b).then_some(a),
            crossed,
        })
    }
}
/// Perform canonical combined preparation exactly once, then validate, own and freeze all
/// arrays/order/identity. No caller-provided mutable prepared dataset is accepted.
pub fn materialize(resolved: &ResolvedProfile) -> Result<MaterializedProfile> {
    let prepared = resolved.prepare_dataset()?;
    assemble(
        resolved.metadata().clone(),
        &resolved.data_request(),
        prepared,
        true,
    )
}

fn stage_graph(m: &ProfileMetadata) -> Result<StageGraph> {
    let blocks = match m.model {
        Model::Mlp2020 | Model::Emlp1 => 0,
        Model::Emlp2 => 1,
        Model::Emlp3 => 2,
    };
    require(m.layers.len() == 2 * (blocks + 1), "stage/layer count")?;
    require(
        m.classes >= 2
            && m.output_width
                == if m.model == Model::Mlp2020 {
                    m.classes
                } else {
                    m.classes.next_power_of_two()
                },
        "real/padded classes",
    )?;
    let mut names = BTreeSet::new();
    let mut stages = Vec::new();
    for i in 0..=blocks {
        let a = &m.layers[2 * i];
        let b = &m.layers[2 * i + 1];
        require(
            a.input > 0 && a.output > 0 && b.input == a.output && b.output == m.output_width,
            "stage layer dimensions",
        )?;
        require(
            names.insert(&a.name)
                && names.insert(&b.name)
                && !a.name.is_empty()
                && !b.name.is_empty(),
            "unique nonempty layer names",
        )?;
        let expected_a = if i < blocks {
            LayerRole::LocalForward { block: i }
        } else if blocks > 0 {
            LayerRole::OutputHidden
        } else {
            LayerRole::Hidden
        };
        let expected_b = if i < blocks {
            LayerRole::LocalHead { block: i }
        } else {
            LayerRole::Output
        };
        require(
            a.role == expected_a && b.role == expected_b,
            "stage roles/barriers",
        )?;
        require(
            i == 0 || a.input == m.layers[2 * (i - 1)].output,
            "stage representation connection",
        )?;
        let activation = if m.model == Model::Mlp2020 {
            Activation::Square
        } else {
            Activation::SquarePlusLinear
        };
        require(
            a.activation == activation
                && b.activation
                    == if m.model == Model::Mlp2020 {
                        Activation::Square
                    } else {
                        Activation::Identity
                    },
            "stage activations",
        )?;
        require(
            a.bias == (m.model == Model::Mlp2020) && b.bias == a.bias,
            "registered bias policy",
        )?;
        stages.push(Stage {
            layers: 2 * i..2 * i + 2,
            input: if i == 0 {
                StageInput::Dataset
            } else {
                StageInput::PreUpdateActivation { layer: 2 * (i - 1) }
            },
            representation_layer: (i < blocks).then_some(2 * i),
        });
    }
    Ok(StageGraph { stages })
}
fn streams(seed: u8, convention: RngConvention) -> (Rng, Rng) {
    match convention {
        RngConvention::DomainSeparated => (crate::twin::rng(seed, 2), crate::twin::rng(seed, 3)),
        RngConvention::LegacyIrisMasterFork => {
            let mut key = [0; 32];
            key[..8].copy_from_slice(&(seed as u64).to_le_bytes());
            let mut master = Rng::from_seed(key);
            (master.fork(), master.fork())
        }
    }
}
fn initialize(
    m: &ProfileMetadata,
    captured: Option<&mut VerifiedSeed>,
) -> Result<InitialModel<f64>> {
    let weights = match &m.initialization {
        Initialization::Captured(binding) => {
            let c = captured
                .ok_or_else(|| StudyInputError::Invalid("missing captured arrays".into()))?;
            require(
                binding.len() == m.layers.len() && c.initialization.len() == binding.len(),
                "captured array count",
            )?;
            for (l, a) in m.layers.iter().zip(binding) {
                require(
                    a.name == l.name
                        && a.logical_shape == (l.output, l.input)
                        && a.source_shape == (l.input, l.output),
                    "captured name/shape binding",
                )?;
            }
            std::mem::take(&mut c.initialization)
        }
        Initialization::Recipe(recipe) => {
            require(
                captured
                    .as_ref()
                    .is_none_or(|c| c.initialization.is_empty()),
                "local recipe cannot consume donor initialization",
            )?;
            require(
                recipe.seed == m.seed && recipe.layers == m.layers,
                "recipe seed/layers",
            )?;
            let (mut rng, _) = streams(m.seed, recipe.rng);
            let mut layers = Vec::new();
            for l in &m.layers {
                let mut draw = || match recipe.distribution {
                    Distribution::Normal { sigma } => {
                        let u1 = ((rng.next_u64() >> 11) as f64 * 2f64.powi(-53)).max(1e-300);
                        let u2 = (rng.next_u64() >> 11) as f64 * 2f64.powi(-53);
                        // Preserve iris.rs::normal followed by sigma * normal(r):
                        // reassociating sigma into this product changes canonical F64 bits.
                        let normal =
                            (-2. * u1.ln()).sqrt() * (2. * std::f64::consts::PI * u2).cos();
                        sigma * normal
                    }
                    Distribution::XavierUniform => {
                        (1. / (l.input + l.output) as f64).sqrt() * crate::twin::uni(&mut rng)
                    }
                };
                let weights = Array2::from_shape_fn((l.output, l.input), |_| draw());
                let biases = (0..l.output)
                    .map(|_| {
                        if l.bias {
                            draw()
                        } else {
                            if recipe.distribution == Distribution::XavierUniform {
                                let _ = draw();
                            }
                            0.
                        }
                    })
                    .collect();
                layers.push(InitialLayer {
                    name: l.name.clone(),
                    weights,
                    biases,
                });
            }
            return validate_initial(m, InitialModel { layers });
        }
        Initialization::MissingCaptured => {
            return Err(StudyInputError::Invalid(
                "missing captured seed, no fallback".into(),
            ));
        }
    };
    validate_initial(
        m,
        InitialModel {
            layers: m
                .layers
                .iter()
                .zip(weights)
                .map(|(l, w)| InitialLayer {
                    name: l.name.clone(),
                    weights: w,
                    biases: vec![0.; l.output],
                })
                .collect(),
        },
    )
}
fn validate_initial(m: &ProfileMetadata, initial: InitialModel<f64>) -> Result<InitialModel<f64>> {
    require(
        initial.layers.len() == m.layers.len(),
        "initial layer count",
    )?;
    for (l, a) in m.layers.iter().zip(&initial.layers) {
        require(
            a.name == l.name
                && a.weights.dim() == (l.output, l.input)
                && a.biases.len() == l.output,
            "initial names/dimensions",
        )?;
        require(
            a.weights.iter().chain(&a.biases).all(|v| v.is_finite()),
            "nonfinite initialization",
        )?;
        require(
            l.bias || a.biases.iter().all(|&v| v == 0.),
            "frozen bias must be zero",
        )?;
    }
    Ok(initial)
}
fn make_order(
    m: &ProfileMetadata,
    population: usize,
    captured: Option<&VerifiedSeed>,
) -> Result<MaterializedOrder> {
    require(
        population > 0 && u32::try_from(population - 1).is_ok(),
        "row ID exceeds u32",
    )?;
    let retained = if m.drop_last && !matches!(m.order, OrderPolicy::Captured) {
        population / m.batch * m.batch
    } else {
        population
    };
    let total = retained
        .checked_mul(m.logical_epochs)
        .ok_or_else(|| StudyInputError::Invalid("order length overflow".into()))?;
    require(retained > 0, "empty retained pass")?;
    let rows = match &m.order {
        OrderPolicy::Captured => {
            let order = captured
                .and_then(|c| c.order.as_ref())
                .ok_or_else(|| StudyInputError::Invalid("missing captured order".into()))?;
            require(order.len() == total, "captured order cardinality")?;
            order
                .iter()
                .map(|&i| {
                    u32::try_from(i)
                        .map_err(|_| StudyInputError::Invalid("captured row width".into()))
                })
                .collect::<Result<Vec<_>>>()?
        }
        OrderPolicy::PerEpochRecipe { rng, coverage } => {
            require(
                match coverage {
                    OrderCoverage::FullPermutation => retained == population,
                    OrderCoverage::UniqueSubset { samples_per_pass } => {
                        *samples_per_pass == retained
                    }
                },
                "local coverage declaration",
            )?;
            let (_, mut stream) = streams(m.seed, *rng);
            let mut permutation: Vec<u32> = (0..population).map(|i| i as u32).collect();
            let mut rows = Vec::with_capacity(total);
            for _ in 0..m.logical_epochs {
                if *rng == RngConvention::DomainSeparated {
                    for (i, v) in permutation.iter_mut().enumerate() {
                        *v = i as u32;
                    }
                }
                for i in (1..population).rev() {
                    permutation.swap(i, stream.below(i as u64 + 1) as usize);
                }
                rows.extend_from_slice(&permutation[..retained]);
            }
            rows
        }
    };
    let offsets = (0..=m.logical_epochs)
        .map(|i| i * retained)
        .collect::<Vec<_>>();
    for pass in rows.chunks_exact(retained) {
        let mut seen = vec![false; population];
        for &i in pass {
            let i = i as usize;
            require(i < population && !seen[i], "order range/unique coverage")?;
            seen[i] = true;
        }
    }
    require(rows.len() == total, "order conservation")?;
    Ok(MaterializedOrder {
        rows,
        pass_offsets: offsets,
    })
}
fn validate_dataset(m: &ProfileMetadata, q: &DataRequest, d: &Dataset) -> Result<()> {
    require(
        q.profile_id == m.id
            && q.dataset_id == m.dataset
            && d.profile_id == m.id
            && d.dataset_id == m.dataset
            && d.classes == m.classes
            && q.classes == m.classes,
        "prepared dataset identity",
    )?;
    require(
        q.features == m.layers[0].input
            && d.train_x.ncols() == q.features
            && d.test_x.ncols() == q.features
            && q.train_samples == Some(d.train_x.nrows())
            && q.test_samples == Some(d.test_x.nrows()),
        "prepared dataset dimensions",
    )?;
    for (x, y, indices) in [
        (&d.train_x, &d.train_y, &d.train_indices),
        (&d.test_x, &d.test_y, &d.test_indices),
    ] {
        require(
            x.nrows() > 0 && y.len() == x.nrows() && indices.len() == x.nrows(),
            "dataset/split cardinality",
        )?;
        require(
            x.iter().all(|v| v.is_finite()) && y.iter().all(|&y| (y as usize) < m.classes),
            "finite data/real labels",
        )?;
        require(
            indices.iter().collect::<BTreeSet<_>>().len() == indices.len(),
            "duplicate source split row",
        )?;
    }
    if matches!(
        q.source,
        DataSource::IrisCsv { .. } | DataSource::Wdbc { .. }
    ) {
        let total = d.train_indices.len() + d.test_indices.len();
        let all = d
            .train_indices
            .iter()
            .chain(&d.test_indices)
            .copied()
            .collect::<BTreeSet<_>>();
        require(
            all.len() == total && all.last().is_some_and(|&n| n < total),
            "fixed source split partition",
        )?;
    } else {
        require(
            d.train_indices.iter().copied().eq(0..d.train_y.len())
                && d.test_indices.iter().copied().eq(0..d.test_y.len()),
            "published source row order",
        )?;
    }
    for f in &m.identity.dataset_files {
        require(
            d.files
                .iter()
                .any(|(p, h)| p == &f.path.display().to_string() && h == &f.sha256),
            "prepared dataset file digest",
        )?;
    }
    Ok(())
}
fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.
}
fn validate_plan(
    m: &ProfileMetadata,
    plan: &BatchPlan,
    order: &MaterializedOrder,
    captured: Option<&VerifiedSeed>,
) -> Result<Vec<RateCoefficients<f64>>> {
    require(
        m.batch > 0 && m.logical_epochs > 0 && !plan.updates.is_empty(),
        "nonempty batch/epochs/plan",
    )?;
    let total = order.rows.len();
    let per_pass = order.pass_offsets[1];
    require(plan.processed_samples == total, "plan/order conservation")?;
    let mut processed = 0;
    let mut checkpoint = 0;
    let mut boundary = 1;
    let mut prev_updates = 0;
    let mut prev_samples = 0;
    let mut rates = Vec::new();
    for (index, u) in plan.updates.iter().enumerate() {
        let expected_valid = if matches!(m.order, OrderPolicy::Captured) {
            m.batch.min(total - processed)
        } else {
            m.batch.min(per_pass - processed % per_pass)
        };
        require(
            u.index == index
                && u.processed_start == processed
                && u.valid_samples == expected_valid
                && expected_valid > 0,
            "plan row offsets/valid count",
        )?;
        let expected_index = m.schedule.as_ref().map(|s| processed / s.reference_batch);
        require(
            u.source_schedule_index == expected_index,
            "plan schedule index",
        )?;
        let coefficient = match expected_index {
            Some(i) => captured
                .and_then(|c| c.schedule.as_ref())
                .and_then(|s| s.get(i))
                .copied()
                .ok_or_else(|| StudyInputError::Invalid("selected schedule missing".into()))?,
            None => m.rate,
        };
        let scale = match m.rate_units {
            RateUnits::Mean => 1.,
            RateUnits::SumTimesValid => expected_valid as f64,
            RateUnits::SumTimesNominal => m.batch as f64,
        };
        let delta = if m.loss == Loss::MeanOutputMse {
            2. / m.classes as f64
        } else {
            1.
        };
        require(
            u.source_coefficient == coefficient
                && u.mean_gradient_rate == coefficient * scale
                && u.delta_multiplier == delta,
            "plan coefficient/loss binding",
        )?;
        let kernel = u.mean_gradient_rate * u.delta_multiplier;
        require(
            [
                u.source_coefficient,
                u.mean_gradient_rate,
                u.delta_multiplier,
                kernel,
            ]
            .into_iter()
            .all(positive),
            "finite positive full-plan coefficients",
        )?;
        rates.push(RateCoefficients {
            source: u.source_coefficient,
            mean: u.mean_gradient_rate,
            delta: u.delta_multiplier,
            kernel,
        });
        processed = processed
            .checked_add(u.valid_samples)
            .ok_or_else(|| StudyInputError::Invalid("plan sample overflow".into()))?;
        require(processed <= total, "plan exceeds order")?;
        let mut crossed = Vec::new();
        while boundary <= m.logical_epochs && processed >= boundary * per_pass {
            crossed.push(boundary);
            boundary += 1;
        }
        if !crossed.is_empty() {
            let c = plan
                .checkpoints
                .get(checkpoint)
                .ok_or_else(|| StudyInputError::Invalid("missing checkpoint".into()))?;
            require(
                c.completed_updates == index + 1
                    && c.processed_samples == processed
                    && c.crossed_logical_boundaries == crossed
                    && c.exact_logical_boundary == (processed % per_pass == 0)
                    && c.interval_updates == index + 1 - prev_updates
                    && c.interval_samples == processed - prev_samples,
                "checkpoint endpoints/interval conservation",
            )?;
            checkpoint += 1;
            prev_updates = index + 1;
            prev_samples = processed;
        }
    }
    require(
        processed == total
            && checkpoint == plan.checkpoints.len()
            && boundary == m.logical_epochs + 1,
        "full plan/checkpoint coverage",
    )?;
    Ok(rates)
}
fn cast32(v: f64) -> Result<f32> {
    let v = v as f32;
    require(v.is_finite(), "nonfinite F32 cast")?;
    Ok(v)
}
fn rate32(r: &RateCoefficients<f64>) -> Result<RateCoefficients<f32>> {
    let c = RateCoefficients {
        source: cast32(r.source)?,
        mean: cast32(r.mean)?,
        delta: cast32(r.delta)?,
        kernel: cast32(r.kernel)?,
    };
    require(
        [c.source, c.mean, c.delta, c.kernel]
            .into_iter()
            .all(|v| v > 0.),
        "F32 rate underflow",
    )?;
    Ok(c)
}
fn array32(a: &Array2<f64>) -> Result<Array2<f32>> {
    let values = a.iter().map(|&v| cast32(v)).collect::<Result<Vec<_>>>()?;
    Array2::from_shape_vec(a.dim(), values)
        .map_err(|_| StudyInputError::Invalid("F32 shape".into()))
}
fn assemble(
    m: ProfileMetadata,
    q: &DataRequest,
    mut prepared: PreparedDataset,
    canonical_prepared: bool,
) -> Result<MaterializedProfile> {
    let graph = stage_graph(&m)?;
    validate_dataset(&m, q, &prepared.dataset)?;
    require(m.batch > 0 && m.logical_epochs > 0, "positive batch/epochs")?;
    if let Some(c) = &prepared.inputs.captured {
        require(
            c.profile_id == m.id && c.seed == m.seed,
            "prepared captured profile/seed",
        )?;
        let expected = m
            .captured_order
            .as_ref()
            .ok_or_else(|| StudyInputError::Invalid("unexpected captured input".into()))?;
        require(
            expected.population == prepared.dataset.train_x.nrows()
                && expected.passes == m.logical_epochs,
            "captured population/passes",
        )?;
        require(
            c.files.iter().any(|(p, h)| {
                p == &expected.path.display().to_string() && Some(h) == expected.sha256.as_ref()
            }),
            "captured order digest binding",
        )?;
        if let Initialization::Captured(arrays) = &m.initialization {
            for a in arrays {
                require(
                    c.files.iter().any(|(p, h)| {
                        p == &a.file.path.display().to_string() && h == &a.file.sha256
                    }),
                    "captured array digest binding",
                )?;
            }
        }
    }
    let order = make_order(
        &m,
        prepared.dataset.train_x.nrows(),
        prepared.inputs.captured.as_ref(),
    )?;
    let rates = validate_plan(
        &m,
        &prepared.inputs.plan,
        &order,
        prepared.inputs.captured.as_ref(),
    )?;
    let per_pass = order.pass_offsets[1];
    require(
        prepared.inputs.plan.dropped_samples
            == (prepared.dataset.train_x.nrows() - per_pass) * m.logical_epochs,
        "dropped row accounting",
    )?;
    let initial = initialize(&m, prepared.inputs.captured.as_mut())?;
    let reference = rates.iter().map(|r| r.kernel).fold(0., f64::max);
    let rates32 = rates.iter().map(rate32).collect::<Result<Vec<_>>>()?;
    let reference32 = cast32(reference)?;
    if let Optimizer::Nesterov { momentum } = m.optimizer {
        require(
            positive(momentum) && momentum < 1. && positive(reference * momentum),
            "finite reference/momentum product",
        )?;
        let mu = cast32(momentum)?;
        for coefficient in rates
            .iter()
            .map(|r| r.kernel)
            .chain(std::iter::once(reference))
        {
            require(
                positive(coefficient * momentum) && positive(coefficient * (1. + momentum)),
                "finite folded NAG coefficients",
            )?;
        }
        for coefficient in rates32
            .iter()
            .map(|r| r.kernel)
            .chain(std::iter::once(reference32))
        {
            require(
                [coefficient * mu, coefficient * (1. + mu)]
                    .into_iter()
                    .all(|v| v.is_finite() && v > 0.),
                "finite F32 folded NAG coefficients",
            )?;
        }
    }
    let initial32 = InitialModel {
        layers: initial
            .layers
            .iter()
            .map(|a| {
                Ok(InitialLayer {
                    name: a.name.clone(),
                    weights: array32(&a.weights)?,
                    biases: a
                        .biases
                        .iter()
                        .map(|&v| cast32(v))
                        .collect::<Result<Vec<_>>>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let f32 = F32Storage {
        train_x: array32(&prepared.dataset.train_x)?,
        test_x: array32(&prepared.dataset.test_x)?,
        initial: initial32,
        rates: rates32,
        reference: reference32,
    };
    let identity = identity(
        &m,
        &prepared.dataset,
        &order,
        &prepared.inputs,
        NumericInputs {
            train_x: &prepared.dataset.train_x,
            test_x: &prepared.dataset.test_x,
            train_y: &prepared.dataset.train_y,
            test_y: &prepared.dataset.test_y,
            initial: &initial,
            rates: &rates,
            reference,
            dtype: Dtype::F64,
            cast_policy: CastPolicy::CanonicalF64,
        },
        &f32,
        canonical_prepared,
    )?;
    Ok(MaterializedProfile {
        metadata: m,
        graph,
        dataset: prepared.dataset,
        initial,
        order,
        plan: prepared.inputs.plan,
        identity,
        f32,
        rates,
        reference,
    })
}

// Every digest begins with the version and domain as length-prefixed UTF-8. All integer
// lengths/dimensions/options are u64 LE (option marker is one u8); rows are u32 LE.
// Enum tags are explicit below. Arrays use logical row-major IEEE bits, never native layout.
struct Encoding(Vec<u8>);
impl Encoding {
    fn new(domain: &str) -> Self {
        let mut e = Self(Vec::new());
        e.text(IDENTITY_VERSION);
        e.text(domain);
        e
    }
    fn u64(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_le_bytes());
    }
    fn byte(&mut self, n: u8) {
        self.0.push(n);
    }
    fn text(&mut self, s: &str) {
        self.u64(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn float(&mut self, n: f64) {
        self.0.extend_from_slice(&n.to_bits().to_le_bytes());
    }
    fn optional(&mut self, n: Option<usize>) {
        self.byte(u8::from(n.is_some()));
        if let Some(n) = n {
            self.u64(n);
        }
    }
    fn indices(&mut self, tag: &str, values: &[usize]) {
        self.text(tag);
        self.u64(values.len());
        for &v in values {
            self.u64(v);
        }
    }
    fn finish(self) -> String {
        sha256_hex(&self.0)
    }
}
trait Bits: Copy {
    const DTYPE: u8;
    fn put(self, e: &mut Encoding);
}
impl Bits for f64 {
    const DTYPE: u8 = 64;
    fn put(self, e: &mut Encoding) {
        e.float(self);
    }
}
impl Bits for f32 {
    const DTYPE: u8 = 32;
    fn put(self, e: &mut Encoding) {
        e.0.extend_from_slice(&self.to_bits().to_le_bytes());
    }
}
fn matrix<T: Bits>(e: &mut Encoding, name: &str, a: &Array2<T>) {
    e.text(name);
    e.byte(T::DTYPE);
    e.u64(2);
    e.u64(a.nrows());
    e.u64(a.ncols());
    for &v in a {
        v.put(e);
    }
}
fn parameters<T: Bits>(model: &InitialModel<T>, bias: bool) -> String {
    let mut e = Encoding::new(if bias { "biases" } else { "weights" });
    e.u64(model.layers.len());
    for a in &model.layers {
        if bias {
            e.text(&a.name);
            e.byte(T::DTYPE);
            e.u64(1);
            e.u64(a.biases.len());
            for &v in &a.biases {
                v.put(&mut e);
            }
        } else {
            matrix(&mut e, &a.name, &a.weights);
        }
    }
    e.finish()
}
fn data_encoding<T: Bits>(train: &Array2<T>, test: &Array2<T>, d: &Dataset) -> Encoding {
    let mut e = Encoding::new("dataset");
    matrix(&mut e, "train_x", train);
    matrix(&mut e, "test_x", test);
    for (name, labels) in [("train_y", &d.train_y), ("test_y", &d.test_y)] {
        e.text(name);
        e.byte(8);
        e.u64(1);
        e.u64(labels.len());
        e.0.extend_from_slice(labels);
    }
    e
}
fn numeric_digest<T: Bits>(
    train: &Array2<T>,
    test: &Array2<T>,
    d: &Dataset,
    initial: &InitialModel<T>,
    rates: &[RateCoefficients<T>],
    reference: T,
    cast: &str,
) -> String {
    let mut e = Encoding::new("numeric-inputs");
    e.byte(T::DTYPE);
    e.text(cast);
    e.text(&data_encoding(train, test, d).finish());
    e.text(&parameters(initial, false));
    e.text(&parameters(initial, true));
    e.u64(rates.len());
    for r in rates {
        for v in [r.source, r.mean, r.delta, r.kernel] {
            v.put(&mut e);
        }
    }
    reference.put(&mut e);
    e.finish()
}
fn plan_digest(p: &BatchPlan) -> String {
    let mut e = Encoding::new("plan");
    e.u64(p.processed_samples);
    e.u64(p.dropped_samples);
    e.u64(p.updates.len());
    for u in &p.updates {
        e.u64(u.index);
        e.u64(u.processed_start);
        e.u64(u.valid_samples);
        e.optional(u.source_schedule_index);
        e.float(u.source_coefficient);
        e.float(u.mean_gradient_rate);
        e.float(u.delta_multiplier);
    }
    e.u64(p.checkpoints.len());
    for c in &p.checkpoints {
        e.u64(c.completed_updates);
        e.u64(c.processed_samples);
        e.indices("crossed", &c.crossed_logical_boundaries);
        e.byte(u8::from(c.exact_logical_boundary));
        e.u64(c.interval_updates);
        e.u64(c.interval_samples);
    }
    e.finish()
}
fn metadata_encoding(e: &mut Encoding, m: &ProfileMetadata) {
    e.text(&m.id);
    e.byte(m.seed);
    e.text(&m.dataset);
    e.byte(match m.model {
        Model::Mlp2020 => 0,
        Model::Emlp1 => 1,
        Model::Emlp2 => 2,
        Model::Emlp3 => 3,
    });
    e.u64(m.classes);
    e.u64(m.output_width);
    e.u64(m.layers.len());
    for l in &m.layers {
        e.text(&l.name);
        e.u64(l.input);
        e.u64(l.output);
        e.byte(u8::from(l.bias));
        e.byte(match l.activation {
            Activation::Square => 0,
            Activation::SquarePlusLinear => 1,
            Activation::Identity => 2,
        });
        match l.role {
            LayerRole::Hidden => e.byte(0),
            LayerRole::Output => e.byte(1),
            LayerRole::LocalForward { block } => {
                e.byte(2);
                e.u64(block);
            }
            LayerRole::LocalHead { block } => {
                e.byte(3);
                e.u64(block);
            }
            LayerRole::OutputHidden => e.byte(4),
        }
    }
    e.byte(match m.loss {
        Loss::MeanOutputMse => 0,
        Loss::RssDelta => 1,
    });
    match m.optimizer {
        Optimizer::Sgd => e.byte(0),
        Optimizer::Nesterov { momentum } => {
            e.byte(1);
            e.float(momentum);
        }
    }
    e.float(m.weight_decay);
    e.float(m.rate);
    e.text(&m.rate_description);
    e.byte(match m.rate_units {
        RateUnits::Mean => 0,
        RateUnits::SumTimesValid => 1,
        RateUnits::SumTimesNominal => 2,
    });
    e.u64(m.batch);
    e.u64(m.logical_epochs);
    e.byte(u8::from(m.drop_last));
    match &m.initialization {
        Initialization::Captured(a) => {
            e.byte(0);
            e.text("source-in-by-out; prepared-logical-out-by-in");
            e.u64(a.len());
            for a in a {
                e.text(&a.name);
                e.u64(a.logical_shape.0);
                e.u64(a.logical_shape.1);
                e.u64(a.source_shape.0);
                e.u64(a.source_shape.1);
            }
        }
        Initialization::Recipe(r) => {
            e.byte(1);
            e.byte(r.seed);
            rng_encoding(e, r.rng);
            match r.distribution {
                Distribution::Normal { sigma } => {
                    e.byte(0);
                    e.float(sigma);
                }
                Distribution::XavierUniform => e.byte(1),
            }
        }
        Initialization::MissingCaptured => e.byte(2),
    }
    match m.order {
        OrderPolicy::Captured => e.byte(0),
        OrderPolicy::PerEpochRecipe { rng, coverage } => {
            e.byte(1);
            rng_encoding(e, rng);
            match coverage {
                OrderCoverage::FullPermutation => e.byte(0),
                OrderCoverage::UniqueSubset { samples_per_pass } => {
                    e.byte(1);
                    e.u64(samples_per_pass);
                }
            }
        }
    }
    if let Some(s) = &m.schedule {
        e.byte(1);
        e.text(&s.json_pointer);
        e.u64(s.steps);
        e.u64(s.reference_batch);
    } else {
        e.byte(0);
    }
}
fn rng_encoding(e: &mut Encoding, r: RngConvention) {
    e.byte(match r {
        RngConvention::LegacyIrisMasterFork => 0,
        RngConvention::DomainSeparated => 1,
    });
}
fn identity(
    m: &ProfileMetadata,
    d: &Dataset,
    order: &MaterializedOrder,
    inputs: &PreparedInputs,
    wide: NumericInputs<'_, f64>,
    f32: &F32Storage,
    canonical: bool,
) -> Result<MaterializedIdentity> {
    let weights = parameters(wide.initial, false);
    let biases = parameters(wide.initial, true);
    let plan = plan_digest(&inputs.plan);
    let mut e = Encoding::new("order");
    e.u64(order.rows.len());
    for i in &order.rows {
        e.0.extend_from_slice(&i.to_le_bytes());
    }
    e.indices("pass_offsets", &order.pass_offsets);
    let order = e.finish();
    let dataset = data_encoding(&d.train_x, &d.test_x, d).finish();
    let mut e = Encoding::new("split");
    e.text(&m.identity.split_policy);
    e.text(&d.split_identity);
    e.indices("train_source_indices", &d.train_indices);
    e.indices("test_source_indices", &d.test_indices);
    let split = e.finish();
    let mut e = Encoding::new("preprocessing");
    e.text(&m.identity.preprocessing);
    e.text(&d.prep);
    let preprocessing = e.finish();
    let f64_inputs = numeric_digest(
        wide.train_x,
        wide.test_x,
        d,
        wide.initial,
        wide.rates,
        wide.reference,
        "canonical-f64",
    );
    let f32_inputs = numeric_digest(
        &f32.train_x,
        &f32.test_x,
        d,
        &f32.initial,
        &f32.rates,
        f32.reference,
        "round-to-f32-once",
    );
    let root = m
        .identity
        .registry
        .path
        .ancestors()
        .nth(4)
        .ok_or_else(|| StudyInputError::Invalid("registry root path".into()))?;
    let mut sources = Vec::new();
    let mut add = |path: &Path, expected: &str, actual: &str| -> Result<()> {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| StudyInputError::Invalid("source outside registry root".into()))?
            .to_str()
            .ok_or_else(|| StudyInputError::Invalid("source path encoding".into()))?
            .replace('\\', "/");
        if let Some(old) = sources.iter().find(|s: &&SourceDigest| s.path == relative) {
            require(
                old.expected_sha256 == expected && old.actual_sha256 == actual,
                "conflicting source digests",
            )?;
        } else {
            sources.push(SourceDigest {
                path: relative,
                expected_sha256: expected.into(),
                actual_sha256: actual.into(),
            });
        }
        Ok(())
    };
    for f in std::iter::once(&m.identity.registry)
        .chain(std::iter::once(&m.identity.integrity_lock))
        .chain(&m.identity.sources)
    {
        add(&f.path, &f.sha256, &f.sha256)?;
    }
    for (p, h) in &d.files {
        add(Path::new(p), h, h)?;
    }
    if let Some(c) = &inputs.captured {
        for (p, h) in &c.files {
            add(Path::new(p), h, h)?;
        }
    }
    for f in &m.identity.dataset_files {
        add(&f.path, &f.sha256, &f.sha256)?;
    }
    sources.sort_by(|a, b| a.path.cmp(&b.path));
    let mut e = Encoding::new("embedded-code-build");
    for (name, bytes) in [
        (
            "study_inputs.rs",
            include_bytes!("study_inputs.rs").as_slice(),
        ),
        (
            "study_profile.rs",
            include_bytes!("study_profile.rs").as_slice(),
        ),
        (
            "profile_data.rs",
            include_bytes!("profile_data.rs").as_slice(),
        ),
        ("data.rs", include_bytes!("data.rs").as_slice()),
        ("twin.rs", include_bytes!("twin.rs").as_slice()),
        (
            "sampler.rs",
            include_bytes!("../../ckks-train-core/src/sampler.rs").as_slice(),
        ),
        (
            "Cargo.lock",
            include_bytes!("../../../Cargo.lock").as_slice(),
        ),
    ] {
        e.text(name);
        e.text(&sha256_hex(bytes));
    }
    e.text(env!("CARGO_PKG_VERSION"));
    e.text(std::env::consts::ARCH);
    e.text(std::env::consts::OS);
    let code = e.finish();
    let mut e = Encoding::new("materialized-identity");
    e.byte(u8::from(canonical));
    metadata_encoding(&mut e, m);
    for hash in [
        &weights,
        &biases,
        &order,
        &plan,
        &dataset,
        &split,
        &preprocessing,
        &f64_inputs,
        &f32_inputs,
        &code,
    ] {
        e.text(hash);
    }
    e.u64(sources.len());
    for s in &sources {
        e.text(&s.path);
        e.text(&s.expected_sha256);
        e.text(&s.actual_sha256);
    }
    Ok(MaterializedIdentity {
        digest: e.finish(),
        weights,
        biases,
        order,
        plan,
        f64_inputs,
        f32_inputs,
        dataset,
        split,
        preprocessing,
        sources,
        code,
        canonical_prepared: canonical,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study_profile::StudyRegistry;
    use std::path::Path;
    fn registry() -> StudyRegistry {
        StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
    }
    fn synthetic(m: &ProfileMetadata, q: &DataRequest, inputs: PreparedInputs) -> PreparedDataset {
        let train = q.train_samples.unwrap();
        let test = q.test_samples.unwrap();
        PreparedDataset {
            dataset: Dataset {
                profile_id: m.id.clone(),
                dataset_id: m.dataset.clone(),
                classes: m.classes,
                train_x: Array2::zeros((train, q.features)),
                test_x: Array2::zeros((test, q.features)),
                train_y: vec![0; train],
                test_y: vec![0; test],
                files: m
                    .identity
                    .dataset_files
                    .iter()
                    .map(|f| (f.path.display().to_string(), f.sha256.clone()))
                    .collect(),
                prep: "SYNTHETIC NON SCIENTIFIC".into(),
                split_identity: "SYNTHETIC NON SCIENTIFIC".into(),
                train_indices: (0..train).collect(),
                test_indices: (0..test).collect(),
            },
            inputs,
        }
    }
    #[test]
    fn captured_block_layers_use_preflight_logical_matrices_without_another_transpose() {
        let r = registry();
        for (id, count) in [("mnist_released__emlp2", 4), ("mnist_released__emlp3", 6)] {
            let p = r.resolve(id, 2).unwrap();
            let mut inputs = p.preflight(None).unwrap();
            let actual = initialize(p.metadata(), inputs.captured.as_mut()).unwrap();
            assert_eq!(actual.layers().len(), count);
            let Initialization::Captured(bindings) = &p.metadata().initialization else {
                panic!("captured")
            };
            for (layer, binding) in actual.layers().iter().zip(bindings) {
                let raw = crate::data::parse_npy_f64(&std::fs::read(&binding.file.path).unwrap())
                    .unwrap();
                assert_eq!(layer.weights().dim(), binding.logical_shape);
                for i in [0, 1] {
                    for j in [0, 1] {
                        assert_eq!(layer.weights()[[i, j]], raw[[j, i]]);
                    }
                }
                assert!(layer.biases().iter().all(|&v| v == 0.));
            }
        }
    }

    #[test]
    fn local_mnist2020_borrows_only_order_schedule_and_generates_active_normal_biases() {
        let p = registry()
            .resolve("mnist_released__mlp2020__b64", 9)
            .unwrap();
        let m = p.metadata();
        let q = p.data_request();
        let inputs = p.preflight(None).unwrap();
        assert!(inputs.captured.as_ref().unwrap().initialization.is_empty());
        let got = assemble(m.clone(), &q, synthetic(m, &q, inputs), false).unwrap();
        assert_eq!(got.initial().layers()[0].weights().dim(), (10, 196));
        assert!(
            got.initial()
                .layers()
                .iter()
                .flat_map(|a| a.biases())
                .all(|&v| v != 0.)
        );
        assert_eq!(
            got.update_rows(749).unwrap().planned().processed_start,
            47936
        );
        assert_eq!(got.update_rows(749).unwrap().rows().len(), 64);
        assert_eq!(
            got.update_rows(749).unwrap().planned().delta_multiplier,
            0.2
        );
        assert_eq!(
            got.reference_kernel_rate_f64(),
            got.numeric_f64()
                .rates()
                .iter()
                .map(|r| r.kernel)
                .fold(0., f64::max)
        );
    }

    #[test]
    fn stable_named_ieee_encoding_matches_independent_python_sha256_fixture() {
        let model = InitialModel {
            layers: vec![InitialLayer {
                name: "h".into(),
                weights: Array2::from_shape_vec((1, 2), vec![1., -2.]).unwrap(),
                biases: vec![0.25],
            }],
        };
        assert_eq!(
            parameters(&model, false),
            "c6c2e937c424cc432d1308d3feab160aeff1f0728d119e3a8f076acdcb9485b4"
        );
        assert_eq!(
            parameters(&model, true),
            "97551481ec0ecfb5122940684c136fe493fd9522f268f38f78783171102be731"
        );
    }

    #[test]
    fn local_initializer_rejects_unexpected_borrowed_captured_weights() {
        let p = registry().resolve("mnist_released__mlp2020", 0).unwrap();
        let m = p.metadata();
        let q = p.data_request();
        let mut data = synthetic(m, &q, p.preflight(None).unwrap());
        data.inputs
            .captured
            .as_mut()
            .unwrap()
            .initialization
            .push(Array2::zeros((1, 1)));
        assert!(
            assemble(m.clone(), &q, data, false).is_err(),
            "ignored unexpected donor weights"
        );
    }

    #[test]
    fn finite_f32_rate_with_overflowing_nag_fold_is_rejected() {
        let p = registry().resolve("iris__emlp1", 0).unwrap();
        let mut m = p.metadata().clone();
        let q = p.data_request();
        let mut prepared = p.prepare_dataset().unwrap();
        m.rate = (f32::MAX as f64) * 0.75 / m.batch as f64;
        for u in &mut prepared.inputs.plan.updates {
            u.source_coefficient = m.rate;
            u.mean_gradient_rate = m.rate * u.valid_samples as f64;
        }
        assert!(
            assemble(m, &q, prepared, false).is_err(),
            "accepted overflowing (1+momentum)*rate in F32"
        );
    }

    #[test]
    fn retained_partial_tail_uses_exact_plan_rows_and_no_epoch_crossing() {
        use crate::study_profile::Checkpoint;
        let p = registry().resolve("iris__emlp1", 0).unwrap();
        let q = p.data_request();
        let mut m = p.metadata().clone();
        m.batch = 19;
        m.logical_epochs = 2;
        let mut prepared = p.prepare_dataset().unwrap();
        let mut updates = Vec::new();
        let mut checkpoints = Vec::new();
        let mut processed = 0;
        for epoch in 0..2 {
            for batch in 0..7 {
                let valid = if batch == 6 { 6 } else { 19 };
                updates.push(PlannedUpdate {
                    index: updates.len(),
                    processed_start: processed,
                    valid_samples: valid,
                    source_schedule_index: None,
                    source_coefficient: m.rate,
                    mean_gradient_rate: m.rate * valid as f64,
                    delta_multiplier: 1.,
                });
                processed += valid;
            }
            checkpoints.push(Checkpoint {
                completed_updates: 7 * (epoch + 1),
                processed_samples: 120 * (epoch + 1),
                crossed_logical_boundaries: vec![epoch + 1],
                exact_logical_boundary: true,
                interval_updates: 7,
                interval_samples: 120,
            });
        }
        prepared.inputs.plan = BatchPlan {
            updates,
            checkpoints,
            processed_samples: 240,
            dropped_samples: 0,
        };
        let got = assemble(m, &q, prepared, false).unwrap();
        assert_eq!(
            got.update_rows(6).unwrap().rows(),
            &got.order().pass_rows(0).unwrap()[114..]
        );
        assert_eq!(got.update_rows(6).unwrap().pass(), Some(0));
        assert_eq!(
            got.update_rows(7).unwrap().rows(),
            &got.order().pass_rows(1).unwrap()[..19]
        );
        assert_eq!(got.update_rows(13).unwrap().planned().valid_samples, 6);
        assert_eq!(got.update_rows(13).unwrap().crossed_boundaries(), &[2]);
    }

    #[test]
    fn invalid_stage_connections_roles_names_and_activations_are_rejected() {
        let p = registry().resolve("iris__emlp3", 0).unwrap();
        for change in 0..4 {
            let mut m = p.metadata().clone();
            match change {
                0 => m.layers[2].input = 129,
                1 => m.layers[1].role = LayerRole::Output,
                2 => m.layers[2].name = m.layers[0].name.clone(),
                _ => m.layers[0].activation = Activation::Identity,
            }
            assert!(stage_graph(&m).is_err(), "accepted stage mutation {change}");
        }
    }

    #[test]
    fn xavier_discarded_bias_draws_match_existing_twin_stream() {
        let p = registry().resolve("iris__emlp1", 4).unwrap();
        let m = p.metadata();
        let got = initialize(m, None).unwrap();
        let expected =
            crate::twin::init_net(&[4, 32, 4], &mut crate::twin::rng(4, 2), "xavier", true);
        for (a, b) in got.layers().iter().zip(expected) {
            assert_eq!(a.weights(), &b.w);
            assert_eq!(a.biases(), b.b);
        }
        assert!(got.layers()[1].weights().iter().any(|v| *v != 0.));
    }
    #[test]
    fn actual_captured_selected_seed_orientation_and_flat_batch_rows_are_preserved() {
        let r = registry();
        for (id, count, b) in [
            ("mnist_released__emlp1", 1000, 48),
            ("mnist_released__emlp1__b64", 750, 64),
            ("mnist_released__emlp1__b128_candidate", 375, 128),
        ] {
            let p = r.resolve(id, 2).unwrap();
            let q = p.data_request();
            let m = p.metadata().clone();
            let inputs = p.preflight(None).unwrap();
            let expected = inputs.captured.as_ref().unwrap().order.clone().unwrap();
            let source = match &m.initialization {
                Initialization::Captured(a) => a[0].file.path.clone(),
                _ => panic!("captured"),
            };
            let raw = crate::data::parse_npy_f64(&std::fs::read(source).unwrap()).unwrap();
            let got = assemble(m.clone(), &q, synthetic(&m, &q, inputs), false).unwrap();
            assert!(!got.identity().canonical_prepared());
            assert_eq!(got.initial().layers()[0].weights()[[0, 1]], raw[[1, 0]]);
            assert_eq!(got.plan().updates.len(), count);
            assert_eq!(got.order().rows().len(), 48000);
            assert_eq!(
                got.update_rows(0).unwrap().rows(),
                &expected[..b].iter().map(|&i| i as u32).collect::<Vec<_>>()
            );
            assert_eq!(
                got.update_rows(count - 1).unwrap().rows(),
                &expected[48000 - b..]
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                got.plan()
                    .checkpoints
                    .iter()
                    .map(|c| c.interval_samples)
                    .sum::<usize>(),
                48000
            );
            if b == 128 {
                assert_eq!(got.update_rows(37).unwrap().crossed_boundaries(), &[1]);
                assert_eq!(got.update_rows(37).unwrap().pass(), None);
            }
        }
    }
    #[test]
    fn local_drop_last_shuffles_full_population_and_restarts_canonical_rows_with_persistent_stream()
    {
        let p = registry().resolve("breast_cancer__emlp1", 3).unwrap();
        let m = p.metadata();
        let inputs = p.preflight(None).unwrap();
        let order = make_order(m, 455, inputs.captured.as_ref()).unwrap();
        let mut stream = crate::twin::rng(3, 3);
        for epoch in 0..100 {
            let expected = crate::twin::permutation(455, &mut stream);
            assert_eq!(
                order.pass_rows(epoch).unwrap(),
                &expected[..448]
                    .iter()
                    .map(|&i| i as u32)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(order.rows().len(), 44800);
        assert_eq!(inputs.plan.dropped_samples, 700);
    }
    #[test]
    fn private_boundary_rejects_mutated_shapes_nonfinite_identity_order_and_plan() {
        let p = registry().resolve("mnist_released__emlp1", 1).unwrap();
        let m = p.metadata();
        let q = p.data_request();
        for change in 0..9 {
            let mut data = synthetic(m, &q, p.preflight(None).unwrap());
            match change {
                0 => data.dataset.profile_id = "other".into(),
                1 => data.dataset.train_x = Array2::zeros((4800, 195)),
                2 => data.dataset.train_x[[0, 0]] = f64::NAN,
                3 => data.inputs.captured.as_mut().unwrap().seed = 0,
                4 => {
                    data.inputs.captured.as_mut().unwrap().initialization[0] =
                        Array2::zeros((196, 32))
                }
                5 => {
                    data.inputs.captured.as_mut().unwrap().initialization[0][[0, 0]] = f64::INFINITY
                }
                6 => {
                    data.inputs
                        .captured
                        .as_mut()
                        .unwrap()
                        .order
                        .as_mut()
                        .unwrap()[0] = 4800
                }
                7 => data.inputs.plan.updates[1].processed_start = 0,
                _ => data.inputs.plan.checkpoints[0].interval_samples = 1,
            }
            assert!(
                assemble(m.clone(), &q, data, false).is_err(),
                "accepted mutation {change}"
            );
        }
    }
    #[test]
    fn f32_overflow_and_underflow_rates_fail_instead_of_claiming_reference_readiness() {
        let p = registry().resolve("mnist_released__emlp1", 0).unwrap();
        let q = p.data_request();
        let m = p.metadata();
        let mut data = synthetic(m, &q, p.preflight(None).unwrap());
        data.dataset.train_x[[0, 0]] = f64::MAX;
        assert!(assemble(m.clone(), &q, data, false).is_err());
        assert!(
            rate32(&RateCoefficients {
                source: 1e-100,
                mean: 1e-100,
                delta: 1.,
                kernel: 1e-100
            })
            .is_err()
        );
    }
    #[test]
    fn array_and_order_mutations_change_content_identity_and_synthetic_stays_untrusted() {
        let p = registry().resolve("mnist_released__emlp1", 0).unwrap();
        let q = p.data_request();
        let m = p.metadata();
        let build = |kind| {
            let mut data = synthetic(m, &q, p.preflight(None).unwrap());
            if kind == 1 {
                data.inputs.captured.as_mut().unwrap().initialization[0][[0, 0]] += 1.;
            }
            if kind == 2 {
                data.inputs
                    .captured
                    .as_mut()
                    .unwrap()
                    .order
                    .as_mut()
                    .unwrap()
                    .swap(0, 1);
            }
            assemble(m.clone(), &q, data, false).unwrap()
        };
        let a = build(0);
        let b = build(1);
        let c = build(2);
        assert_ne!(a.identity().weights_sha256(), b.identity().weights_sha256());
        assert_ne!(a.identity().order_sha256(), c.identity().order_sha256());
        assert_ne!(a.identity().digest(), b.identity().digest());
        assert_ne!(a.identity().digest(), c.identity().digest());
        assert!(!a.identity().canonical_prepared());
    }
}
