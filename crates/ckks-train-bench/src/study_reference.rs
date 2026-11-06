//! Bounded canonical shared contracts and selected-seed input receipts.
//!
//! These identities contain no training observations, reference completion, target,
//! build attestation or serialization/publication capability. Only canonical immutable
//! materializations can construct the private-field public values.
use crate::{
    data::sha256_hex,
    profile_data::OrderCoverage,
    study_inputs::{IDENTITY_VERSION, InitialModel, MaterializedProfile, StageInput},
    study_profile::{
        Activation, BatchPlan, Checkpoint, Distribution, GenerationRecipe, Initialization,
        InputIdentity, Layer, LayerRole, Loss, Model, Optimizer, OrderPolicy, Orientation,
        PlannedUpdate, ProfileMetadata, RateUnits, RngConvention, SELECTOR_VERSION, SharedSelector,
        StudySource, StudySourceRole,
    },
};
use std::fmt;

/// Canonical typed encoding version, independent of any future JSON file schema.
pub const ENCODING_VERSION: &str = "hekit-study-reference-encoding-v1";
/// Shared seed-independent scientific contract version.
pub const CONTRACT_VERSION: &str = "hekit-study-profile-contract-v1";
/// Selected-seed input receipt version.
pub const RECEIPT_VERSION: &str = "hekit-study-seed-input-receipt-v1";
/// Immutable ratified baseline revision, not a current working-tree claim.
pub const POLICY_REVISION: &str = "e018bd7401a6a606e306e58decad7af68cc56b84";
/// Ratified pre-implementation design digest; later document appendices are excluded.
pub const POLICY_SHA256: &str = "6cdca16097d4acbf28c0f9bf0a36cccc1fccfae2fb604b0410926c066b85a035";
const ARITHMETIC: &str = "study-plain-v1;actual-f32;round-f64-to-f32-once;forward-row-output-input;gradient-row-output-input;separate-product-add-no-fma;mean-gradient-divide-after-sum;zero-velocity;nag-v=mu*v+g,step=g+mu*v;logical-output-loss;stage-gradient-barriers;pre-update-handoff";
const CLASSIFICATION: &str =
    "real-classes-only;last-total_cmp-maximum;signed-zero-total-order;all-logical-logits-finite";
const DRAW_POLICY: &str = "chacha20;layer-order;out-by-in-row-major;then-bias;normal-cosine-box-muller-then-sigma;normal-disabled-bias-no-draw;xavier-disabled-bias-draw-discard";
const ORDER_POLICY: &str = "persistent-shuffle-stream;fisher-yates-descending-unbiased-below;legacy-in-place-permutation;domain-fresh-rows-per-pass;captured-flat-no-reshuffle";
const LOOKUP_POLICY: &str = "floor(processed_start/reference_batch);no-interpolation;source-length-bounded;typed-rate-units-v1";

/// Identity construction failure, never evidence of completed training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceIdentityError {
    /// Private fixtures and materializations lacking a canonical resolver descriptor fail closed.
    NonCanonical,
    /// An unsupported or inconsistent selected input cannot be projected.
    Unsupported(&'static str),
}
impl fmt::Display for ReferenceIdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ReferenceIdentityError {}
type Result<T> = std::result::Result<T, ReferenceIdentityError>;

/// Immutable canonical shared profile contract; not a full-reference artifact.
#[derive(Debug)]
pub struct StudyProfileContractV1 {
    bytes: Vec<u8>,
    digest: String,
}
impl StudyProfileContractV1 {
    /// Complete canonical typed payload, suitable for a future reviewed writer.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// SHA256 of exactly the canonical bytes.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}
/// Immutable selected-seed input identity, independent of any training result.
#[derive(Debug)]
pub struct SeedInputReceiptV1 {
    bytes: Vec<u8>,
    digest: String,
    seed: u8,
    contract: String,
    materialized: String,
}
impl SeedInputReceiptV1 {
    /// Complete selected-seed canonical payload.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// SHA256 of exactly the canonical bytes.
    pub fn digest(&self) -> &str {
        &self.digest
    }
    /// Selected statistical seed, never replaced by the seed0 reference seed.
    pub fn seed(&self) -> u8 {
        self.seed
    }
    /// Shared scientific contract to which this selected input belongs.
    pub fn contract_digest(&self) -> &str {
        &self.contract
    }
    /// Complete existing D6a identity of this seed's materialization.
    pub fn materialized_digest(&self) -> &str {
        &self.materialized
    }
}

struct Encoding(Vec<u8>);
impl Encoding {
    fn new(domain: &str) -> Self {
        let mut e = Self(Vec::new());
        e.text(ENCODING_VERSION);
        e.text(domain);
        e
    }
    fn byte(&mut self, n: u8) {
        self.0.push(n);
    }
    fn count(&mut self, n: usize) {
        self.0.extend_from_slice(&(n as u64).to_le_bytes());
    }
    fn text(&mut self, s: &str) {
        self.count(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }
    fn float(&mut self, n: f64) {
        self.0.extend_from_slice(&n.to_bits().to_le_bytes());
    }
    fn option(&mut self, n: Option<usize>) {
        self.byte(u8::from(n.is_some()));
        if let Some(n) = n {
            self.count(n);
        }
    }
    fn optional_text(&mut self, s: Option<&str>) {
        self.byte(u8::from(s.is_some()));
        if let Some(s) = s {
            self.text(s);
        }
    }
    fn indices(&mut self, v: &[usize]) {
        self.count(v.len());
        for &n in v {
            self.count(n);
        }
    }
}
fn rng(e: &mut Encoding, r: RngConvention) {
    e.byte(match r {
        RngConvention::LegacyIrisMasterFork => 0,
        RngConvention::DomainSeparated => 1,
    });
}
fn coverage(e: &mut Encoding, c: OrderCoverage) {
    match c {
        OrderCoverage::FullPermutation => e.byte(0),
        OrderCoverage::UniqueSubset { samples_per_pass } => {
            e.byte(1);
            e.count(samples_per_pass);
        }
    }
}
fn layers(e: &mut Encoding, values: &[Layer]) {
    e.count(values.len());
    for Layer {
        name,
        input,
        output,
        activation,
        bias,
        role,
    } in values
    {
        e.text(name);
        e.count(*input);
        e.count(*output);
        e.byte(u8::from(*bias));
        e.byte(match activation {
            Activation::Square => 0,
            Activation::SquarePlusLinear => 1,
            Activation::Identity => 2,
        });
        match role {
            LayerRole::Hidden => e.byte(0),
            LayerRole::Output => e.byte(1),
            LayerRole::LocalForward { block } => {
                e.byte(2);
                e.count(*block);
            }
            LayerRole::LocalHead { block } => {
                e.byte(3);
                e.count(*block);
            }
            LayerRole::OutputHidden => e.byte(4),
        }
    }
}
fn sources(e: &mut Encoding, files: &[StudySource]) {
    let mut sorted = files.iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| (a.path(), a.role()).cmp(&(b.path(), b.role())));
    e.count(sorted.len());
    for f in sorted {
        e.byte(match f.role() {
            StudySourceRole::Registry => 0,
            StudySourceRole::PortableLock => 1,
            StudySourceRole::SharedConfiguration => 2,
            StudySourceRole::LocalRecipeCode => 3,
            StudySourceRole::DatasetInput => 4,
            StudySourceRole::ScheduleContainer => 5,
            StudySourceRole::SelectedInitialization => 6,
            StudySourceRole::SelectedOrder => 7,
        });
        e.text(f.path());
        e.text(f.sha256());
    }
}
fn selector(e: &mut Encoding, s: &SharedSelector) {
    e.text(SELECTOR_VERSION);
    sources(e, s.sources());
    e.byte(u8::from(s.captured().is_some()));
    if let Some(c) = s.captured() {
        e.text(c.family());
        e.text(c.model());
        e.optional_text(c.initialization_template());
        e.text(c.order_template());
        e.text(c.schedule_template());
    }
}
fn plan(e: &mut Encoding, p: &BatchPlan, numeric: bool) {
    let BatchPlan {
        updates,
        checkpoints,
        processed_samples,
        dropped_samples,
    } = p;
    e.count(*processed_samples);
    e.count(*dropped_samples);
    e.count(updates.len());
    for PlannedUpdate {
        index,
        processed_start,
        valid_samples,
        source_schedule_index,
        source_coefficient,
        mean_gradient_rate,
        delta_multiplier,
    } in updates
    {
        e.count(*index);
        e.count(*processed_start);
        e.count(*valid_samples);
        e.option(*source_schedule_index);
        if numeric {
            e.float(*source_coefficient);
            e.float(*mean_gradient_rate);
            e.float(*delta_multiplier);
        }
    }
    e.count(checkpoints.len());
    for Checkpoint {
        completed_updates,
        processed_samples,
        crossed_logical_boundaries,
        exact_logical_boundary,
        interval_updates,
        interval_samples,
    } in checkpoints
    {
        e.count(*completed_updates);
        e.count(*processed_samples);
        e.indices(crossed_logical_boundaries);
        e.byte(u8::from(*exact_logical_boundary));
        e.count(*interval_updates);
        e.count(*interval_samples);
    }
}
// Private raw projection inputs may be varied by unit fixtures; only derive_identities
// can construct production contract types, after checking the canonical capability.
#[derive(Clone, Copy)]
struct SharedView<'a> {
    profile: &'a MaterializedProfile,
    metadata: &'a ProfileMetadata,
    plan: &'a BatchPlan,
    selector: &'a SharedSelector,
    dataset: &'a str,
    split: &'a str,
    preprocessing: &'a str,
    code: &'a str,
    policy: &'a str,
    classification: &'a str,
}
impl<'a> SharedView<'a> {
    fn canonical(p: &'a MaterializedProfile) -> Result<Self> {
        if !p.identity().canonical_prepared() {
            return Err(ReferenceIdentityError::NonCanonical);
        }
        Ok(Self {
            profile: p,
            metadata: p.metadata(),
            plan: p.plan(),
            selector: p
                .shared_selector()
                .ok_or(ReferenceIdentityError::NonCanonical)?,
            dataset: p.identity().dataset_sha256(),
            split: p.identity().split_sha256(),
            preprocessing: p.identity().preprocessing_sha256(),
            code: p.identity().code_build_sha256(),
            policy: POLICY_SHA256,
            classification: CLASSIFICATION,
        })
    }
}
fn shared_bytes(v: &SharedView<'_>) -> Result<Vec<u8>> {
    let ProfileMetadata {
        id,
        seed: _,
        dataset,
        model,
        classes,
        output_width,
        layers: graph_layers,
        loss,
        optimizer,
        weight_decay,
        rate,
        rate_description,
        rate_units,
        batch,
        logical_epochs,
        drop_last,
        initialization,
        order,
        captured_order,
        schedule,
        identity,
    } = v.metadata;
    let InputIdentity {
        registry: _,
        integrity_lock: _,
        sources: _,
        preprocessing,
        split_policy,
        dataset_files: _,
    } = identity; // all file fields in immutable selector
    let mut e = Encoding::new(CONTRACT_VERSION);
    e.text(CONTRACT_VERSION);
    e.text(SELECTOR_VERSION);
    e.text(POLICY_REVISION);
    e.text(v.policy);
    e.text(IDENTITY_VERSION);
    e.text("study-inputs-v1/embedded-code-build");
    e.text(v.code);
    e.byte(0);
    e.byte(9);
    e.text(id);
    e.text(dataset);
    e.byte(match model {
        Model::Mlp2020 => 0,
        Model::Emlp1 => 1,
        Model::Emlp2 => 2,
        Model::Emlp3 => 3,
    });
    e.count(*classes);
    e.count(*output_width);
    let d = v.profile.dataset();
    e.text(&d.dataset_id);
    e.text(&d.profile_id);
    e.count(d.classes);
    for n in [
        d.train_x.nrows(),
        d.train_x.ncols(),
        d.test_x.nrows(),
        d.test_x.ncols(),
        d.train_y.len(),
        d.test_y.len(),
    ] {
        e.count(n);
    }
    for s in [
        v.dataset,
        v.split,
        v.preprocessing,
        preprocessing,
        split_policy,
        &d.prep,
        &d.split_identity,
    ] {
        e.text(s);
    }
    layers(&mut e, graph_layers);
    e.count(v.profile.graph().stages().len());
    for stage in v.profile.graph().stages() {
        e.count(stage.layers().start);
        e.count(stage.layers().end);
        match stage.input() {
            StageInput::Dataset => e.byte(0),
            StageInput::PreUpdateActivation { layer } => {
                e.byte(1);
                e.count(layer);
            }
        }
        e.option(stage.representation_layer());
    }
    e.byte(match loss {
        Loss::MeanOutputMse => 0,
        Loss::RssDelta => 1,
    });
    match optimizer {
        Optimizer::Sgd => e.byte(0),
        Optimizer::Nesterov { momentum } => {
            e.byte(1);
            e.float(*momentum);
        }
    }
    e.float(*weight_decay);
    e.text(rate_description);
    e.byte(match rate_units {
        RateUnits::Mean => 0,
        RateUnits::SumTimesValid => 1,
        RateUnits::SumTimesNominal => 2,
    });
    e.byte(u8::from(schedule.is_none()));
    if schedule.is_none() {
        e.float(*rate);
    }
    e.text(ARITHMETIC);
    e.text(v.classification);
    e.count(*batch);
    e.count(*logical_epochs);
    e.byte(u8::from(*drop_last));
    match order {
        OrderPolicy::Captured => e.byte(0),
        OrderPolicy::PerEpochRecipe {
            rng: r,
            coverage: c,
        } => {
            e.byte(1);
            rng(&mut e, *r);
            coverage(&mut e, *c);
        }
    }
    e.text(ORDER_POLICY);
    e.byte(u8::from(captured_order.is_some()));
    if let Some(c) = captured_order {
        e.count(c.population);
        e.count(c.passes);
        coverage(&mut e, c.coverage);
    }
    match initialization {
        Initialization::Captured(arrays) => {
            e.byte(0);
            e.count(arrays.len());
            for a in arrays {
                e.text(&a.name);
                e.count(a.logical_shape.0);
                e.count(a.logical_shape.1);
                e.count(a.source_shape.0);
                e.count(a.source_shape.1);
                e.byte(match a.orientation {
                    Orientation::InputByOutput => 0,
                });
            }
        }
        Initialization::Recipe(GenerationRecipe {
            seed: _,
            rng: r,
            distribution,
            layers: recipe_layers,
        }) => {
            e.byte(1);
            rng(&mut e, *r);
            match distribution {
                Distribution::Normal { sigma } => {
                    e.byte(0);
                    e.float(*sigma);
                }
                Distribution::XavierUniform => e.byte(1),
            }
            layers(&mut e, recipe_layers);
            e.text(DRAW_POLICY);
        }
        Initialization::MissingCaptured => {
            return Err(ReferenceIdentityError::Unsupported(
                "missing captured initialization",
            ));
        }
    }
    e.text("selected-statistical-seed-0-through-9;no-fallback;no-cross-seed-substitution");
    selector(&mut e, v.selector);
    e.byte(u8::from(schedule.is_some()));
    if let Some(s) = schedule {
        let file = v
            .selector
            .sources()
            .iter()
            .find(|f| f.role() == StudySourceRole::ScheduleContainer)
            .ok_or(ReferenceIdentityError::Unsupported("missing schedule role"))?;
        if file.sha256() != s.source.sha256 {
            return Err(ReferenceIdentityError::Unsupported(
                "schedule source binding",
            ));
        }
        e.text(file.path());
        e.text(file.sha256());
        e.count(s.steps);
        e.count(s.reference_batch);
        e.text(LOOKUP_POLICY);
    }
    plan(&mut e, v.plan, false);
    Ok(e.0)
}
fn parameters64(e: &mut Encoding, m: &InitialModel<f64>) {
    e.byte(64);
    e.byte(0);
    e.count(m.layers().len());
    for l in m.layers() {
        e.text(l.name());
        e.count(l.weights().nrows());
        e.count(l.weights().ncols());
        for &v in l.weights() {
            e.float(v);
        }
        e.count(l.biases().len());
        for &v in l.biases() {
            e.float(v);
        }
    }
}
fn parameters32(e: &mut Encoding, m: &InitialModel<f32>) {
    e.byte(32);
    e.byte(1);
    e.count(m.layers().len());
    for l in m.layers() {
        e.text(l.name());
        e.count(l.weights().nrows());
        e.count(l.weights().ncols());
        for &v in l.weights() {
            e.0.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        e.count(l.biases().len());
        for &v in l.biases() {
            e.0.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
}
fn receipt_bytes(p: &MaterializedProfile, s: &SharedSelector, contract: &str) -> Vec<u8> {
    let mut e = Encoding::new(RECEIPT_VERSION);
    e.text(RECEIPT_VERSION);
    e.text(contract);
    e.byte(p.metadata().seed);
    e.byte(u8::from(p.identity().canonical_prepared()));
    let i = p.identity();
    for value in [
        i.digest(),
        i.weights_sha256(),
        i.biases_sha256(),
        i.order_sha256(),
        i.plan_sha256(),
        i.f64_inputs_sha256(),
        i.f32_inputs_sha256(),
        i.dataset_sha256(),
        i.split_sha256(),
        i.preprocessing_sha256(),
        i.code_build_sha256(),
    ] {
        e.text(value);
    }
    let mut sorted = i.sources().iter().collect::<Vec<_>>();
    sorted.sort_by(|a, b| a.path.cmp(&b.path));
    e.count(sorted.len());
    for f in sorted {
        e.text(&f.path);
        e.text(&f.expected_sha256);
        e.text(&f.actual_sha256);
    }
    selector(&mut e, s);
    sources(&mut e, p.selected_sources());
    // Shapes/orientation are fully shared above; selected role paths/digests are here.
    e.optional_text(
        p.metadata()
            .schedule
            .as_ref()
            .map(|s| s.json_pointer.as_str()),
    );
    plan(&mut e, p.plan(), true);
    e.float(p.reference_kernel_rate_f64());
    let f = p.numeric_f32();
    e.0.extend_from_slice(&f.reference_kernel_rate().to_bits().to_le_bytes());
    let d = p.numeric_f64();
    e.count(d.rates().len());
    for r in d.rates() {
        for v in [r.source, r.mean, r.delta, r.kernel] {
            e.float(v);
        }
    }
    e.count(f.rates().len());
    for r in f.rates() {
        for v in [r.source, r.mean, r.delta, r.kernel] {
            e.0.extend_from_slice(&v.to_bits().to_le_bytes());
        }
    }
    e.count(p.order().rows().len());
    for &n in p.order().rows() {
        e.count(n as usize);
    }
    e.indices(p.order().pass_offsets());
    parameters64(&mut e, p.initial());
    parameters32(&mut e, f.initial());
    e.0
}
/// Derive immutable dual identities from one already prepared canonical profile.
/// This does not prepare again, train, generate a target, or attest external provenance.
pub fn derive_identities(
    p: &MaterializedProfile,
) -> Result<(StudyProfileContractV1, SeedInputReceiptV1)> {
    let v = SharedView::canonical(p)?;
    let bytes = shared_bytes(&v)?;
    let digest = sha256_hex(&bytes);
    let receipt = receipt_bytes(p, v.selector, &digest);
    let receipt_digest = sha256_hex(&receipt);
    Ok((
        StudyProfileContractV1 {
            bytes,
            digest: digest.clone(),
        },
        SeedInputReceiptV1 {
            bytes: receipt,
            digest: receipt_digest,
            seed: p.metadata().seed,
            contract: digest,
            materialized: p.identity().digest().into(),
        },
    ))
}

// Test-only bridge returns raw bytes, never a trusted contract. The resolver module's
// private-field mutation tests cannot access or manufacture MaterializedProfile state.
#[cfg(test)]
pub(crate) fn synthetic_source_projection(
    p: &MaterializedProfile,
    selector: &SharedSelector,
) -> Result<Vec<u8>> {
    let mut view = SharedView::canonical(p)?;
    view.selector = selector;
    shared_bytes(&view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{study_inputs::materialize, study_profile::StudyRegistry};
    use std::path::Path;
    fn registry() -> StudyRegistry {
        StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
    }
    fn fixture_plan() -> BatchPlan {
        BatchPlan {
            updates: vec![PlannedUpdate {
                index: 0,
                processed_start: 0,
                valid_samples: 2,
                source_schedule_index: Some(3),
                source_coefficient: 0.25,
                mean_gradient_rate: 0.5,
                delta_multiplier: 1.,
            }],
            checkpoints: vec![Checkpoint {
                completed_updates: 1,
                processed_samples: 2,
                crossed_logical_boundaries: vec![1],
                exact_logical_boundary: true,
                interval_updates: 1,
                interval_samples: 2,
            }],
            processed_samples: 2,
            dropped_samples: 0,
        }
    }
    // Independent Python struct.pack('<Q'/'<d') oracle, not generated by this encoder.
    // Detects integer/float endianness, option/boolean width, lengths and field order.
    #[test]
    fn independent_minimal_plan_golden_bytes_and_sha256() {
        let mut e = Encoding::new("golden-plan-v1");
        plan(&mut e, &fixture_plan(), true);
        let expected = "210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76310e00000000000000676f6c64656e2d706c616e2d7631020000000000000000000000000000000100000000000000000000000000000000000000000000000200000000000000010300000000000000000000000000d03f000000000000e03f000000000000f03f010000000000000001000000000000000200000000000000010000000000000001000000000000000101000000000000000200000000000000";
        let expected_bytes: Vec<u8> = (0..expected.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&expected[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(e.0, expected_bytes);
        assert_eq!(
            sha256_hex(&e.0),
            "4e99587a77915a320fc1332ad79c1bd53bd3b74a61a6e845278044b684810b34"
        );
    }
    // Omitting any full-topology field would allow a different full endpoint or
    // sampling schedule to share a contract. Coefficients belong only to numeric receipt.
    #[test]
    fn every_plan_field_and_numeric_coefficient_has_its_declared_scope() {
        let p = fixture_plan();
        let bytes = |p: &BatchPlan, numeric| {
            let mut e = Encoding::new("test-plan");
            plan(&mut e, p, numeric);
            e.0
        };
        type Change = fn(&mut BatchPlan);
        let changes: &[Change] = &[
            |p| p.processed_samples += 1,
            |p| p.dropped_samples += 1,
            |p| p.updates.push(p.updates[0].clone()),
            |p| p.updates[0].index += 1,
            |p| p.updates[0].processed_start += 1,
            |p| p.updates[0].valid_samples += 1,
            |p| p.updates[0].source_schedule_index = None,
            |p| p.checkpoints.push(p.checkpoints[0].clone()),
            |p| p.checkpoints[0].completed_updates += 1,
            |p| p.checkpoints[0].processed_samples += 1,
            |p| p.checkpoints[0].crossed_logical_boundaries.push(2),
            |p| p.checkpoints[0].exact_logical_boundary = false,
            |p| p.checkpoints[0].interval_updates += 1,
            |p| p.checkpoints[0].interval_samples += 1,
        ];
        for (index, change) in changes.iter().enumerate() {
            let mut q = p.clone();
            change(&mut q);
            assert_ne!(bytes(&p, false), bytes(&q, false), "topology field {index}");
            assert_ne!(bytes(&p, true), bytes(&q, true), "receipt field {index}");
        }
        for change in [
            (|p: &mut BatchPlan| p.updates[0].source_coefficient = 0.125) as Change,
            |p: &mut BatchPlan| p.updates[0].mean_gradient_rate = 0.25,
            |p: &mut BatchPlan| p.updates[0].delta_multiplier = 2.,
        ] {
            let mut q = p.clone();
            change(&mut q);
            assert_eq!(bytes(&p, false), bytes(&q, false));
            assert_ne!(bytes(&p, true), bytes(&q, true));
        }
    }
    // Only private raw bytes are produced after mutations: no trusted contract/receipt
    // constructor accepts public mutable metadata or these synthetic mixed projections.
    #[test]
    fn scientific_metadata_fields_cannot_alias_in_shared_projection() {
        let r = registry();
        let p = materialize(&r.resolve("iris__emlp2", 0).unwrap()).unwrap();
        let original = shared_bytes(&SharedView::canonical(&p).unwrap()).unwrap();
        type Change = fn(&mut ProfileMetadata);
        let changes: &[(&str, Change)] = &[
            ("id", |m| m.id.push('x')),
            ("dataset", |m| m.dataset.push('x')),
            ("model", |m| m.model = Model::Emlp3),
            ("classes", |m| m.classes += 1),
            ("output width", |m| m.output_width += 1),
            ("layer count", |m| m.layers.pop().map(|_| ()).unwrap()),
            ("layer name", |m| m.layers[0].name.push('x')),
            ("layer input", |m| m.layers[0].input += 1),
            ("layer output", |m| m.layers[0].output += 1),
            ("layer bias", |m| m.layers[0].bias = true),
            ("layer activation", |m| {
                m.layers[0].activation = Activation::Identity
            }),
            ("layer role", |m| {
                m.layers[0].role = LayerRole::LocalHead { block: 1 }
            }),
            ("loss", |m| m.loss = Loss::MeanOutputMse),
            ("optimizer", |m| m.optimizer = Optimizer::Sgd),
            ("momentum", |m| {
                m.optimizer = Optimizer::Nesterov { momentum: 0.8 }
            }),
            ("decay", |m| m.weight_decay = 0.1),
            ("constant rate", |m| m.rate *= 2.),
            ("rate units text", |m| m.rate_description.push('x')),
            ("rate units", |m| m.rate_units = RateUnits::Mean),
            ("batch", |m| m.batch += 1),
            ("epochs", |m| m.logical_epochs += 1),
            ("tail", |m| m.drop_last = !m.drop_last),
            ("order rng", |m| {
                m.order = OrderPolicy::PerEpochRecipe {
                    rng: RngConvention::LegacyIrisMasterFork,
                    coverage: OrderCoverage::FullPermutation,
                }
            }),
            ("order coverage", |m| {
                m.order = OrderPolicy::PerEpochRecipe {
                    rng: RngConvention::DomainSeparated,
                    coverage: OrderCoverage::UniqueSubset {
                        samples_per_pass: 100,
                    },
                }
            }),
            ("init distribution", |m| {
                if let Initialization::Recipe(r) = &mut m.initialization {
                    r.distribution = Distribution::Normal { sigma: 0.2 };
                }
            }),
            ("init rng", |m| {
                if let Initialization::Recipe(r) = &mut m.initialization {
                    r.rng = RngConvention::LegacyIrisMasterFork;
                }
            }),
            ("init layers", |m| {
                if let Initialization::Recipe(r) = &mut m.initialization {
                    r.layers[0].name.push('x');
                }
            }),
            ("preprocessing", |m| m.identity.preprocessing.push('x')),
            ("split policy", |m| m.identity.split_policy.push('x')),
        ];
        for (name, change) in changes {
            let mut m = p.metadata().clone();
            change(&mut m);
            let mut v = SharedView::canonical(&p).unwrap();
            v.metadata = &m;
            assert_ne!(original, shared_bytes(&v).unwrap(), "omitted {name}");
        }
        let mut m = p.metadata().clone();
        m.seed = 7;
        if let Initialization::Recipe(r) = &mut m.initialization {
            r.seed = 7;
        }
        let mut v = SharedView::canonical(&p).unwrap();
        v.metadata = &m;
        assert_eq!(original, shared_bytes(&v).unwrap());
        let mut missing = m.clone();
        missing.initialization = Initialization::MissingCaptured;
        v.metadata = &missing;
        assert!(shared_bytes(&v).is_err());
    }
    #[test]
    fn verified_content_code_policy_and_tie_semantics_are_shared_bindings() {
        let r = registry();
        let p = materialize(&r.resolve("iris__mlp2020", 0).unwrap()).unwrap();
        let baseline = shared_bytes(&SharedView::canonical(&p).unwrap()).unwrap();
        for field in 0..6 {
            let mut v = SharedView::canonical(&p).unwrap();
            match field {
                0 => v.dataset = "changed-data",
                1 => v.split = "changed-split",
                2 => v.preprocessing = "changed-preprocessing",
                3 => v.code = "changed-input-code",
                4 => v.policy = "changed-policy",
                5 => v.classification = "first-partial-order-maximum",
                _ => unreachable!(),
            }
            assert_ne!(baseline, shared_bytes(&v).unwrap(), "binding {field}");
        }
    }
    // Metadata/preflight only: mixed Iris numerical backing intentionally makes these
    // private raw projections non-scientific; never materialize MNIST or construct contracts.
    #[test]
    fn captured_seed_projection_and_b48_b64_b128_full_topology_are_distinct() {
        let r = registry();
        let iris = materialize(&r.resolve("iris__emlp1", 0).unwrap()).unwrap();
        let mut projections = Vec::new();
        for (id, expected_updates) in [
            ("mnist_released__emlp1", 1000),
            ("mnist_released__emlp1__b64", 750),
            ("mnist_released__emlp1__b128_candidate", 375),
        ] {
            let a = r.resolve(id, 0).unwrap();
            let b = r.resolve(id, 7).unwrap();
            let ap = a.preflight(None).unwrap();
            let bp = b.preflight(None).unwrap();
            assert_eq!(ap.plan.updates.len(), expected_updates);
            assert_eq!(ap.plan.processed_samples, 48000);
            let mut av = SharedView::canonical(&iris).unwrap();
            av.metadata = a.metadata();
            av.plan = &ap.plan;
            av.selector = a.shared_selector();
            let ab = shared_bytes(&av).unwrap();
            let mut bv = SharedView::canonical(&iris).unwrap();
            bv.metadata = b.metadata();
            bv.plan = &bp.plan;
            bv.selector = b.shared_selector();
            assert_eq!(ab, shared_bytes(&bv).unwrap(), "{id}");
            let mut numeric_a = Encoding::new("numeric");
            plan(&mut numeric_a, &ap.plan, true);
            let mut numeric_b = Encoding::new("numeric");
            plan(&mut numeric_b, &bp.plan, true);
            // Current released schedules happen to be equal for seeds 0/7;
            // synthetic coefficient mutations above test the permitted difference.
            assert_eq!(sha256_hex(&numeric_a.0), sha256_hex(&numeric_b.0));
            for mutate in [
                (|m: &mut ProfileMetadata| m.schedule.as_mut().unwrap().steps += 1)
                    as fn(&mut ProfileMetadata),
                |m| m.schedule.as_mut().unwrap().reference_batch += 1,
                |m| m.captured_order.as_mut().unwrap().population += 1,
                |m| m.captured_order.as_mut().unwrap().passes += 1,
            ] {
                let mut m = a.metadata().clone();
                mutate(&mut m);
                let mut changed = av;
                changed.metadata = &m;
                assert_ne!(ab, shared_bytes(&changed).unwrap());
            }
            if expected_updates == 375 {
                assert_eq!(ap.plan.checkpoints[0].interval_updates, 38);
                assert_eq!(ap.plan.checkpoints[0].interval_samples, 4864);
                assert_eq!(ap.plan.checkpoints[1].interval_updates, 37);
                assert_eq!(ap.plan.checkpoints[1].interval_samples, 4736);
            }
            projections.push(ab);
        }
        assert_ne!(projections[0], projections[1]);
        assert_ne!(projections[1], projections[2]);
        assert_ne!(projections[0], projections[2]);
    }
}
