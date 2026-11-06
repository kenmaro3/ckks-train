//! Approved local-process compatibility/integrity, never an execution replay proof.
use super::producer::{ALGORITHM, API, CLASSIFICATION, REDUCTION};
use super::*;
use crate::{
    study_inputs::{MaterializedProfile, materialize},
    study_profile::{BatchPlan, StudyRegistry},
    study_provenance::VerifiedStudyContext,
    study_reference::{SeedInputReceiptV1, StudyProfileContractV1, derive_identities},
};
use std::os::unix::ffi::OsStrExt;
/// A verified file claim under explicit approved local execution trust.
/// Distinct from in-process completion; its state digest has not been replayed.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{VerifiedReference,CompletedFp32Reference};
/// fn forge(v: VerifiedReference) -> CompletedFp32Reference<'static> { v.into() }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::{ParsedReferenceDocument,VerifiedReference};
/// fn forge(p: ParsedReferenceDocument) -> VerifiedReference { p.into() }
/// ```
#[derive(Debug)]
pub struct VerifiedReference {
    parsed: ParsedReferenceDocument,
    contract: StudyProfileContractV1,
    receipt: SeedInputReceiptV1,
    approval: String,
    capsule: String,
}
impl VerifiedReference {
    pub(super) fn parsed(&self) -> &ParsedReferenceDocument {
        &self.parsed
    }
    /// Validated reference run selector.
    pub fn run_id(&self) -> &str {
        &self.parsed.common.run_id.0
    }
    /// Verified canonical profile identifier.
    pub fn profile_id(&self) -> &str {
        &self.parsed.body.identity.profile_id.0
    }
    /// Immutable verified timing claims, not replay evidence.
    pub fn verified_time(&self) -> ReferenceTimeView<'_> {
        ReferenceTimeView(&self.parsed.body.timing)
    }
    /// Complete immutable final completion claims.
    pub fn verified_completion(&self) -> ReferenceCompletionView<'_> {
        ReferenceCompletionView {
            completion: &self.parsed.body.completion,
            endpoint: self
                .parsed
                .body
                .endpoints
                .0
                .last()
                .expect("verified final endpoint"),
        }
    }
    #[doc = "SHA256 of the exact immutable file bytes."]
    pub fn file_sha256(&self) -> &str {
        self.parsed.file_sha256()
    }
    #[doc = "SHA256 of the canonical typed document payload."]
    pub fn payload_sha256(&self) -> &str {
        self.parsed.payload_sha256()
    }
    #[doc = "Independently reconstructed complete shared canonical identity."]
    pub fn contract(&self) -> &StudyProfileContractV1 {
        &self.contract
    }
    #[doc = "Independently reconstructed selected-seed input identity."]
    pub fn receipt(&self) -> &SeedInputReceiptV1 {
        &self.receipt
    }
    #[doc = "Pinned local approval digest used for this operation."]
    pub fn approval_sha256(&self) -> &str {
        &self.approval
    }
    #[doc = "Approved writer compile capsule digest."]
    pub fn writer_capsule_sha256(&self) -> &str {
        &self.capsule
    }
    #[doc = "Final correct count claimed by the approved local session."]
    pub fn final_correct(&self) -> u64 {
        self.parsed.body.final_observation.correct
    }
    #[doc = "Actual held-out denominator, never a caller-supplied count."]
    pub fn test_count(&self) -> u64 {
        self.parsed.body.final_observation.test_count
    }
    #[doc = "Session-origin state digest claim; not independently replayed model content."]
    pub fn final_state_sha256_claim(&self) -> &str {
        &self.parsed.body.completion.final_state_sha256.0
    }
}
/// Immutable timing already checked by the approved reference reader.
pub struct ReferenceTimeView<'a>(&'a Timing);
impl ReferenceTimeView<'_> {
    /// Start Unix nanoseconds.
    pub fn start_unix_ns(&self) -> u64 {
        self.0.start_unix_ns
    }
    /// Completion Unix nanoseconds.
    pub fn completion_unix_ns(&self) -> u64 {
        self.0.end_unix_ns
    }
    /// Fixed verified wall clock source.
    pub fn wall_source(&self) -> &str {
        "unix_system_time"
    }
    /// Fixed verified monotonic clock source.
    pub fn monotonic_source(&self) -> &str {
        "std_instant"
    }
    /// Complete measured operation duration.
    pub fn end_to_end_ns(&self) -> u64 {
        self.0.end_to_end_ns
    }
    /// Verified available setup_ns timing.
    pub fn setup_ns(&self) -> u64 {
        *self
            .0
            .setup_ns
            .available()
            .expect("verified available timing")
    }
    /// Verified available update_ns timing.
    pub fn update_ns(&self) -> u64 {
        *self
            .0
            .update_ns
            .available()
            .expect("verified available timing")
    }
    /// Verified available evaluation_ns timing.
    pub fn evaluation_ns(&self) -> u64 {
        *self
            .0
            .evaluation_ns
            .available()
            .expect("verified available timing")
    }
    /// Verified available finalization_ns timing.
    pub fn finalization_ns(&self) -> u64 {
        *self
            .0
            .finalization_ns
            .available()
            .expect("verified available timing")
    }
}
/// Immutable complete endpoint, including actual boundary and interval observations.
pub struct ReferenceEndpointView<'a>(&'a Endpoint);
impl ReferenceEndpointView<'_> {
    /// Verified completed_updates.
    pub fn completed_updates(&self) -> u64 {
        self.0.completed_updates
    }
    /// Verified processed_samples.
    pub fn processed_samples(&self) -> u64 {
        self.0.processed_samples
    }
    /// Verified interval_updates.
    pub fn interval_updates(&self) -> u64 {
        self.0.interval_updates
    }
    /// Verified interval_samples.
    pub fn interval_samples(&self) -> u64 {
        self.0.interval_samples
    }
    /// Exact crossed logical boundary IDs.
    pub fn crossed_logical_boundaries(&self) -> &[u64] {
        &self.0.crossed_logical_boundaries.0
    }
    /// Whether this actual endpoint is an exact logical boundary.
    pub fn exact_logical_boundary(&self) -> bool {
        self.0.exact_logical_boundary
    }
    /// Approved final correct count claim.
    pub fn correct(&self) -> u64 {
        self.0.observation.correct
    }
    /// Verified actual held-out count.
    pub fn test_count(&self) -> u64 {
        self.0.observation.test_count
    }
    /// Exact binary64 reporting observation bits.
    pub fn accuracy_bits(&self) -> u64 {
        self.0.observation.bits
    }
    /// Available cumulative update duration.
    pub fn update_elapsed_ns(&self) -> u64 {
        *self
            .0
            .update_elapsed_ns
            .available()
            .expect("verified timing")
    }
    /// Available cumulative evaluation duration.
    pub fn evaluation_elapsed_ns(&self) -> u64 {
        *self
            .0
            .evaluation_elapsed_ns
            .available()
            .expect("verified timing")
    }
}
/// Complete reader-verified completion projection, with no replacement or constructor.
pub struct ReferenceCompletionView<'a> {
    completion: &'a Completion,
    endpoint: &'a Endpoint,
}
impl ReferenceCompletionView<'_> {
    /// Verified origin_completed_updates.
    pub fn origin_completed_updates(&self) -> u64 {
        self.completion.origin_completed_updates
    }
    /// Verified origin_processed_samples.
    pub fn origin_processed_samples(&self) -> u64 {
        self.completion.origin_processed_samples
    }
    /// Verified planned_updates.
    pub fn planned_updates(&self) -> u64 {
        self.completion.planned_updates
    }
    /// Verified actual_updates.
    pub fn actual_updates(&self) -> u64 {
        self.completion.actual_updates
    }
    /// Verified planned_samples.
    pub fn planned_samples(&self) -> u64 {
        self.completion.planned_samples
    }
    /// Verified actual_samples.
    pub fn actual_samples(&self) -> u64 {
        self.completion.actual_samples
    }
    /// Verified planned_dropped_samples.
    pub fn planned_dropped_samples(&self) -> u64 {
        self.completion.planned_dropped_samples
    }
    /// Verified actual_dropped_samples.
    pub fn actual_dropped_samples(&self) -> u64 {
        self.completion.actual_dropped_samples
    }
    /// Verified final_endpoint_index.
    pub fn final_endpoint_index(&self) -> u64 {
        self.completion.final_endpoint_index
    }
    /// Verified full_plan_sha256 claim; final model is not replayed.
    pub fn full_plan_sha256(&self) -> &str {
        &self.completion.full_plan_sha256.0
    }
    /// Verified consumed_plan_sha256 claim; final model is not replayed.
    pub fn consumed_plan_sha256(&self) -> &str {
        &self.completion.consumed_plan_sha256.0
    }
    /// Verified final_state_sha256 claim; final model is not replayed.
    pub fn final_state_sha256(&self) -> &str {
        &self.completion.final_state_sha256.0
    }
    /// Complete actual final checkpoint.
    pub fn final_endpoint(&self) -> ReferenceEndpointView<'_> {
        ReferenceEndpointView(self.endpoint)
    }
}
fn check(ok: bool, field: &str) -> Result<()> {
    ensure(ok, DocumentErrorCode::CrossLinkMismatch, field)
}
// This pure component is not a capability constructor. Both FULL byte arrays are mandatory.
fn identity_matches(
    claim: &ClaimedIdentity,
    profile: &str,
    seed: u64,
    materialized: &str,
    contract: &[u8],
    receipt: &[u8],
) -> Result<()> {
    check(
        claim.profile_id.0 == profile
            && claim.seed == seed
            && claim.materialized_sha256.0 == materialized,
        "identity.selector",
    )?;
    for (claimed, expected, version) in [
        (&claim.contract, contract, CONTRACT_VERSION),
        (&claim.seed_receipt, receipt, RECEIPT_VERSION),
    ] {
        check(
            claimed.schema.0 == version
                && claimed.encoding.0 == ENCODING_VERSION
                && claimed.bytes.len() == expected.len()
                && claimed.bytes.as_slice() == expected
                && claimed.sha256.0 == sha256_hex(expected),
            "identity.canonical_bytes",
        )?;
    }
    Ok(())
}
fn science(body: &ReferenceBody, plan: &BatchPlan, hash: &str, denominator: usize) -> Result<()> {
    let c = &body.completion;
    check(
        c.origin_completed_updates == 0
            && c.origin_processed_samples == 0
            && c.planned_updates == plan.updates.len() as u64
            && c.actual_updates == plan.updates.len() as u64
            && c.planned_samples == plan.processed_samples as u64
            && c.actual_samples == plan.processed_samples as u64
            && c.planned_dropped_samples == plan.dropped_samples as u64
            && c.actual_dropped_samples == plan.dropped_samples as u64,
        "completion.counts",
    )?;
    check(
        c.full_plan_sha256.0 == hash
            && c.consumed_plan_sha256.0
                == crate::study_reference::digests::consumed(plan, hash, plan.updates.len())
                    .map_err(|_| {
                        err(DocumentErrorCode::CrossLinkMismatch, "completion.consumed")
                    })?,
        "completion.plan",
    )?;
    check(
        !plan.checkpoints.is_empty()
            && body.endpoints.0.len() == plan.checkpoints.len()
            && c.final_endpoint_index == plan.checkpoints.len() as u64 - 1,
        "endpoints.count",
    )?;
    for (e, t) in body.endpoints.0.iter().zip(&plan.checkpoints) {
        check(
            e.completed_updates == t.completed_updates as u64
                && e.processed_samples == t.processed_samples as u64
                && e.crossed_logical_boundaries
                    .0
                    .iter()
                    .copied()
                    .eq(t.crossed_logical_boundaries.iter().map(|&x| x as u64))
                && e.exact_logical_boundary == t.exact_logical_boundary
                && e.interval_updates == t.interval_updates as u64
                && e.interval_samples == t.interval_samples as u64,
            "endpoints.topology",
        )?;
        check(
            e.observation.test_count == denominator as u64 && denominator > 0,
            "endpoints.denominator",
        )?;
        check(
            e.update_elapsed_ns.available().is_some()
                && e.evaluation_elapsed_ns.available().is_some(),
            "endpoints.timing",
        )?;
    }
    let last = body
        .endpoints
        .0
        .last()
        .ok_or_else(|| err(DocumentErrorCode::OuterProgressMismatch, "endpoints"))?;
    check(
        last.completed_updates == c.actual_updates
            && last.processed_samples == c.actual_samples
            && last.observation == body.final_observation
            && body.final_observation.test_count == denominator as u64,
        "final_observation",
    )?;
    Ok(())
}
fn required_policy(c: &Common, t: &Timing) -> Result<()> {
    check(
        c.schema.0 == REFERENCE
            && c.kind == Kind::Reference
            && c.status == Status::Complete
            && !c.synthetic
            && c.writer.api_version.0 == API
            && c.writer.algorithm_revision.0 == ALGORITHM,
        "writer",
    )?;
    check(
        c.arithmetic.dtype == Dtype::F32
            && c.arithmetic.accumulation == Accumulation::F32
            && c.arithmetic.cast == Cast::F32
            && c.arithmetic.reduction.0 == REDUCTION
            && c.arithmetic.classification.0 == CLASSIFICATION,
        "arithmetic",
    )?;
    check(
        t.scope == TimingScope::After
            && t.setup_ns.available().is_some()
            && t.update_ns.available().is_some()
            && t.evaluation_ns.available().is_some()
            && t.finalization_ns.available().is_some(),
        "timing.required",
    )?;
    check(
        matches!(c.provenance, ProvenanceClaim::Captured(_)),
        "provenance.required",
    )
}
fn canonical_encoded(v: &impl Encode) -> Result<Vec<u8>> {
    let mut e = Encoder {
        layout: DocumentLayout::Reference,
        bytes: vec![],
        outer: 0,
    };
    v.encode(&mut e)?;
    Ok(e.bytes)
}
fn approved_build_matches(actual: &Build, expected: &Build) -> Result<()> {
    check(
        canonical_encoded(actual)? == canonical_encoded(expected)?,
        "provenance.build",
    )
}
fn approved_provenance(c: &Common, context: &VerifiedStudyContext) -> Result<String> {
    approved_provenance_claim(&c.provenance, context)
}
pub(super) fn approved_provenance_claim(
    claim: &ProvenanceClaim,
    context: &VerifiedStudyContext,
) -> Result<String> {
    let ProvenanceClaim::Captured(p) = claim else {
        return Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "provenance.required",
        ));
    };
    let f = context
        .build_facts(&p.build.receipt_file_sha256.0)
        .map_err(|_| {
            err(
                DocumentErrorCode::CrossLinkMismatch,
                "provenance.approved_receipt",
            )
        })?;
    let expected = serde_json::value::to_raw_value(&producer::build(&f))
        .map_err(|e| json_err(e, "provenance.build"))?;
    let expected = Build::decode(&expected, "provenance.build")?;
    approved_build_matches(&p.build, &expected)?;
    check(
        p.source.root.bytes == context.root().as_os_str().as_bytes()
            && p.invocation.cwd.bytes == context.root().as_os_str().as_bytes(),
        "provenance.root",
    )?;
    check(
        matches!(&p.source.commit,Observation::Available(v) if v.0.0==context.source_commit())
            && p.source.tracked == GitState::Clean
            && p.source.untracked == GitState::Clean,
        "provenance.clean_source",
    )?;
    let files = context
        .source_files()
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "provenance.source"))?;
    check(
        p.source.manifest.0.len() == files.len()
            && p.source
                .manifest
                .0
                .iter()
                .zip(&files)
                .all(|(a, (path, hash))| a.path.0 == *path && a.sha256.0 == *hash),
        "provenance.source_manifest",
    )?;
    check(
        p.executable.path.bytes == f.executable_path()
            && p.executable.start_sha256.0 == f.executable_sha256()
            && p.executable.end_sha256.0 == f.executable_sha256(),
        "provenance.executable",
    )?;
    check(
        p.recheck.source_manifest_sha256.0 == context.source_manifest_sha256()
            && p.recheck.build_receipt_file_sha256.0 == f.receipt_file_sha256(),
        "provenance.recheck",
    )?;
    runtime_inventory(p)?;
    Ok(f.capsule_sha256())
}
fn runtime_inventory(p: &Provenance) -> Result<()> {
    const KEYS: [&str; 7] = [
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ];
    check(
        p.thread_environment.0.len() == KEYS.len()
            && p.thread_environment.0.iter().zip(KEYS).all(|(v, k)| {
                v.key.0 == k
                    && match &v.value {
                        Observation::Available(t) => !t.0.is_empty(),
                        Observation::Unavailable(r) => r.0 == "unset",
                    }
            }),
        "provenance.thread_environment",
    )?;
    check(
        !p.invocation.argv.0.is_empty() && p.host.execution_threads == 1,
        "provenance.runtime",
    )
}
fn profile_caps(
    p: &MaterializedProfile,
    bytes: usize,
    contract_len: usize,
    receipt_len: usize,
) -> Result<()> {
    let u = p.plan().updates.len() as u128;
    let e = p.plan().checkpoints.len() as u128;
    let r = p.order().rows().len() as u128;
    let params = p
        .metadata()
        .layers
        .iter()
        .map(|l| (l.input as u128) * (l.output as u128) + l.output as u128)
        .sum::<u128>();
    let s = u128::from(
        p.plan()
            .updates
            .iter()
            .any(|u| u.source_schedule_index.is_some()),
    );
    let cb = (25 + 8 * s) * u + 49 * e + 1_048_576;
    let rb = (97 + 8 * s) * u + 8 * r + 57 * e + 8 + 12 * params + 1_048_576;
    let jb = 2 * (cb + rb) + A_MAX as u128;
    check(
        contract_len as u128 <= cb && receipt_len as u128 <= rb && bytes as u128 <= jb,
        "identity.profile_caps",
    )
}
pub(in crate::study_reference) fn verify_reference(
    bytes: &[u8],
    context: &VerifiedStudyContext,
    id: &str,
) -> Result<VerifiedReference> {
    context
        .recheck()
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "context.recheck"))?;
    let parsed = parse_reference_document(bytes)?;
    check(parsed.claimed_run_id() == id, "run_id")?;
    required_policy(&parsed.common, &parsed.body.timing)?;
    let capsule = approved_provenance(&parsed.common, context)?;
    let claim = &parsed.body.identity;
    let registry = StudyRegistry::load(context.root())
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "registry"))?;
    let resolved = registry
        .resolve(
            &claim.profile_id.0,
            u8::try_from(claim.seed).map_err(|_| err(DocumentErrorCode::OutOfRange, "seed"))?,
        )
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "profile"))?;
    let p = materialize(&resolved).map_err(|_| {
        err(
            DocumentErrorCode::CrossLinkMismatch,
            "canonical_inputs_missing_or_invalid",
        )
    })?;
    let (contract, receipt) =
        derive_identities(&p).map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "identity"))?;
    check(
        receipt.contract_digest() == contract.digest()
            && receipt.materialized_digest() == p.identity().digest(),
        "identity.trusted_linkage",
    )?;
    identity_matches(
        claim,
        &p.metadata().id,
        u64::from(p.metadata().seed),
        p.identity().digest(),
        contract.canonical_bytes(),
        receipt.canonical_bytes(),
    )?;
    let numeric = p.numeric_f32();
    check(
        numeric.test_y().len() == numeric.test_x().nrows(),
        "held_out",
    )?;
    science(
        &parsed.body,
        p.plan(),
        p.identity().plan_sha256(),
        numeric.test_y().len(),
    )?;
    profile_caps(
        &p,
        bytes.len(),
        contract.canonical_bytes().len(),
        receipt.canonical_bytes().len(),
    )?;
    Ok(VerifiedReference {
        parsed,
        contract,
        receipt,
        approval: context.approval_sha256().into(),
        capsule,
    })
}
#[cfg(test)]
#[path = "reader_tests.rs"]
mod tests;
