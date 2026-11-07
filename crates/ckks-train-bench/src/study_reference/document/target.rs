//! Closed lock wire layout. Parsed claims never grant publication or execution authority.
use super::*;
use crate::study_provenance::VerifiedStudyContext;
use crate::study_reference::{POLICY_REVISION, POLICY_SHA256, session::CaptureFact};
use serde_json::{Value, json};
const SCHEMA: &str = "hekit-study-target-lock-v1";
const ALGORITHM: &str = "seed0-final-minus-half-percentage-point-f64-v1";
const API: &str = "hekit-study-target-api-v1";
const PROTOCOL: &str = "hekit-study-target-protocol-v1";
const DESIGNS: [&str; 5] = [
    "docs/experiment-matrix.md",
    "docs/study-reference-design.md",
    "docs/study-reference-document-design.md",
    "docs/study-reference-session-design.md",
    "docs/study-training-design.md",
];

tags!(LockKind { Lock=0=>"target_lock" });
record!(CompletionProjection {
    origin_completed_updates: u64,
    origin_processed_samples: u64,
    planned_updates: u64,
    actual_updates: u64,
    planned_samples: u64,
    actual_samples: u64,
    planned_dropped_samples: u64,
    actual_dropped_samples: u64,
    full_plan_sha256: Hash,
    consumed_plan_sha256: Hash,
    final_endpoint_index: u64,
    final_endpoint: Endpoint,
    final_state_sha256: Hash,
});
record!(ReferenceLink {
    run_id: Text<128>, schema: Text<128>, kind: Kind, link: Link, seed: u64,
    seed_receipt_sha256: Hash, materialized_sha256: Hash, writer_capsule_sha256: Hash,
    start_unix_ns: u64, completion_unix_ns: u64, completion: CompletionProjection,
    final_observation: ClaimedAccuracy,
});
#[derive(Debug, Clone)]
pub(super) struct Target(u64);
impl Decode for Target {
    fn decode(raw: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(raw, &["value", "bits"], p)?;
        let raw = f.get("value", p)?.get();
        ensure(raw.len() <= 64, DocumentErrorCode::TokenLimit, p)?;
        ensure(
            raw.as_bytes()
                .first()
                .is_some_and(|b| b.is_ascii_digit() || *b == b'-'),
            DocumentErrorCode::BadJson,
            p,
        )?;
        let value: f64 = raw
            .parse()
            .map_err(|_| err(DocumentErrorCode::BadJson, p))?;
        ensure(value.is_finite(), DocumentErrorCode::NonFinite, p)?;
        let bits: Text<16> = f.field("bits", p)?;
        ensure(
            bits.0.len() == 16 && lower_hex(&bits.0),
            DocumentErrorCode::BadHex,
            p,
        )?;
        let bits =
            u64::from_str_radix(&bits.0, 16).map_err(|_| err(DocumentErrorCode::BadHex, p))?;
        ensure(
            value.to_bits() == bits && !value.is_sign_negative() && value <= 1.0,
            DocumentErrorCode::ObservationMismatch,
            p,
        )?;
        Ok(Self(bits))
    }
}
impl Encode for Target {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        e.u64(self.0)
    }
}

pub(in crate::study_reference) fn target_bits(correct: u64, count: u64) -> Result<(u64, u64)> {
    ensure(
        count > 0 && count <= 1u64 << 53 && correct <= count,
        DocumentErrorCode::OutOfRange,
        "target.counts",
    )?;
    let accuracy = (correct as f64) / (count as f64);
    let difference = accuracy - f64::from_bits(0x3f747ae147ae147b);
    let target = if difference > 0.0 { difference } else { 0.0 };
    Ok((accuracy.to_bits(), target.to_bits()))
}
pub(super) fn auxiliary(domain: &str, fields: &[&str]) -> Result<Vec<u8>> {
    let mut e = Encoder::new(DocumentLayout::TargetLock);
    e.text(ENCODING_VERSION)?;
    e.text(domain)?;
    for field in fields {
        e.text(field)?;
    }
    Ok(e.bytes)
}
pub(in crate::study_reference) fn profile_slot(profile: &str) -> Result<String> {
    Ok(sha256_hex(&auxiliary(
        "hekit-study-profile-slot-v1",
        &[profile],
    )?))
}
pub(super) fn lock_id(campaign: &str, profile: &str) -> Result<String> {
    Ok(sha256_hex(&auxiliary(
        "hekit-study-target-lock-id-v1",
        &[campaign, profile],
    )?))
}
pub(super) fn valid_id(value: &str, p: &str) -> Result<()> {
    ensure(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-')),
        DocumentErrorCode::OutOfRange,
        p,
    )
}

#[derive(Debug)]
pub(in crate::study_reference) struct Lock {
    schema: Text<128>,
    kind: LockKind,
    status: Status,
    writer: Writer,
    synthetic: bool,
    campaign_id: Text<128>,
    profile_id: Text<128>,
    lock_id: Hash,
    policy: Policy,
    contract: ClaimedIdentityBytes,
    reference: ReferenceLink,
    algorithm: Text<128>,
    target: Target,
    created: Clock,
    provenance: ProvenanceClaim,
    integrity: Integrity,
    file_sha256: String,
}
impl Lock {
    fn canonical(&self) -> Result<Vec<u8>> {
        let mut e = Encoder::new(DocumentLayout::TargetLock);
        e.text(ENCODING_VERSION)?;
        self.schema.encode(&mut e)?;
        self.schema.encode(&mut e)?;
        self.kind.encode(&mut e)?;
        self.status.encode(&mut e)?;
        self.writer.encode(&mut e)?;
        self.synthetic.encode(&mut e)?;
        self.campaign_id.encode(&mut e)?;
        self.profile_id.encode(&mut e)?;
        self.lock_id.encode(&mut e)?;
        self.policy.encode(&mut e)?;
        self.contract.encode(&mut e)?;
        self.reference.encode(&mut e)?;
        self.algorithm.encode(&mut e)?;
        self.target.encode(&mut e)?;
        self.created.encode(&mut e)?;
        self.provenance.encode(&mut e)?;
        self.integrity.encoding.encode(&mut e)?;
        self.integrity.hash.encode(&mut e)?;
        Ok(e.bytes)
    }
    pub(in crate::study_reference) fn payload_sha256(&self) -> &str {
        &self.integrity.payload_sha256.0
    }
    pub(in crate::study_reference) fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub(in crate::study_reference) fn run_id(&self) -> &str {
        &self.reference.run_id.0
    }
    pub(in crate::study_reference) fn threshold_bits(&self) -> u64 {
        self.target.0
    }
}
fn decode_lock(bytes: &[u8]) -> Result<Lock> {
    preflight_layout(bytes, DocumentLayout::TargetLock)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|e| json_err(e, ""))?;
    let mut f = Fields::new(
        raw,
        &[
            "schema",
            "kind",
            "status",
            "writer",
            "synthetic",
            "campaign_id",
            "profile_id",
            "lock_id",
            "policy",
            "contract",
            "reference",
            "algorithm",
            "target",
            "created",
            "provenance",
            "integrity",
        ],
        "",
    )?;
    let mut lock = Lock {
        schema: f.field("schema", "")?,
        kind: f.field("kind", "")?,
        status: f.field("status", "")?,
        writer: f.field("writer", "")?,
        synthetic: f.field("synthetic", "")?,
        campaign_id: f.field("campaign_id", "")?,
        profile_id: f.field("profile_id", "")?,
        lock_id: f.field("lock_id", "")?,
        policy: f.field("policy", "")?,
        contract: ClaimedIdentityBytes::parse(
            f.get("contract", "")?,
            "contract",
            CONTRACT_VERSION,
            C_MAX,
        )?,
        reference: f.field("reference", "")?,
        algorithm: f.field("algorithm", "")?,
        target: f.field("target", "")?,
        created: f.field("created", "")?,
        provenance: f.field("provenance", "")?,
        integrity: f.field("integrity", "")?,
        file_sha256: sha256_hex(bytes),
    };
    validate_provenance(&mut lock.provenance, true)?;
    Ok(lock)
}
fn validate(lock: &Lock, schema: &str, reference_schema: &str, synthetic: bool) -> Result<()> {
    ensure(
        lock.schema.0 == schema && lock.reference.schema.0 == reference_schema,
        DocumentErrorCode::UnsupportedSchema,
        "schema",
    )?;
    ensure(
        lock.synthetic == synthetic,
        DocumentErrorCode::UnsupportedSchema,
        "synthetic",
    )?;
    ensure(
        lock.status == Status::Complete
            && lock.reference.kind == Kind::Reference
            && lock.reference.seed == 0,
        DocumentErrorCode::UnsupportedTag,
        "reference",
    )?;
    ensure(
        lock.writer.api_version.0 == API
            && lock.writer.algorithm_revision.0 == PROTOCOL
            && lock.algorithm.0 == ALGORITHM,
        DocumentErrorCode::UnsupportedSchema,
        "writer",
    )?;
    for (id, p) in [
        (&lock.campaign_id.0, "campaign_id"),
        (&lock.profile_id.0, "profile_id"),
        (&lock.reference.run_id.0, "reference.run_id"),
    ] {
        valid_id(id, p)?;
    }
    ensure(
        lock.lock_id.0 == lock_id(&lock.campaign_id.0, &lock.profile_id.0)?,
        DocumentErrorCode::CrossLinkMismatch,
        "lock_id",
    )?;
    validate_policy(&lock.policy)?;
    ensure(
        matches!(lock.provenance, ProvenanceClaim::Captured(_)),
        DocumentErrorCode::CrossLinkMismatch,
        "provenance",
    )?;
    ensure(
        lock.reference.start_unix_ns <= lock.reference.completion_unix_ns
            && lock.reference.completion_unix_ns <= lock.created.unix_ns,
        DocumentErrorCode::TimingMismatch,
        "created",
    )?;
    let o = &lock.reference.final_observation;
    ensure(
        target_bits(o.correct, o.test_count)? == (o.bits, lock.target.0),
        DocumentErrorCode::ObservationMismatch,
        "target",
    )?;
    let c = &lock.reference.completion;
    let e = &c.final_endpoint;
    ensure(
        c.origin_completed_updates == 0
            && c.origin_processed_samples == 0
            && c.planned_updates == c.actual_updates
            && c.planned_samples == c.actual_samples
            && c.planned_dropped_samples == c.actual_dropped_samples
            && c.full_plan_sha256 == c.consumed_plan_sha256
            && e.completed_updates == c.actual_updates
            && e.processed_samples == c.actual_samples
            && e.observation == *o,
        DocumentErrorCode::CrossLinkMismatch,
        "reference.completion",
    )?;
    ensure(
        e.interval_updates > 0
            && e.interval_updates <= e.completed_updates
            && e.interval_samples > 0
            && e.interval_samples <= e.processed_samples
            && e.update_elapsed_ns.available().is_some()
            && e.evaluation_elapsed_ns.available().is_some(),
        DocumentErrorCode::CrossLinkMismatch,
        "reference.endpoint",
    )?;
    ensure(
        lock.integrity.encoding.0 == ENCODING_VERSION && lock.integrity.hash.0 == "sha256",
        DocumentErrorCode::UnsupportedSchema,
        "integrity",
    )?;
    Ok(())
}
pub(in crate::study_reference) fn parse_lock(bytes: &[u8]) -> Result<Lock> {
    let lock = decode_lock(bytes)?;
    validate(&lock, SCHEMA, REFERENCE, false)?;
    ensure(
        sha256_hex(&lock.canonical()?) == lock.integrity.payload_sha256.0,
        DocumentErrorCode::DigestMismatch,
        "integrity.payload_sha256",
    )?;
    Ok(lock)
}
#[cfg(test)]
pub(super) struct SyntheticTargetLock(Lock);
#[cfg(test)]
impl SyntheticTargetLock {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let lock = decode_lock(bytes)?;
        validate(
            &lock,
            "hekit-test-target-lock-v1",
            "hekit-test-reference-document-v1",
            true,
        )?;
        ensure(
            sha256_hex(&lock.canonical()?) == lock.integrity.payload_sha256.0,
            DocumentErrorCode::DigestMismatch,
            "integrity.payload_sha256",
        )?;
        Ok(Self(lock))
    }
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        self.0.canonical()
    }
    pub(super) fn payload_sha256(&self) -> &str {
        self.0.payload_sha256()
    }
}
fn policy_value(context: &VerifiedStudyContext) -> Result<Value> {
    let files = context
        .source_files()
        .map_err(|_| err(DocumentErrorCode::CrossLinkMismatch, "policy.source"))?;
    let pinned = |path: &str| -> Result<&str> {
        let mut matches = files.iter().filter(|(p, _)| p == path);
        let (_, hash) = matches
            .next()
            .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "policy.path_missing"))?;
        ensure(
            matches.next().is_none(),
            DocumentErrorCode::CrossLinkMismatch,
            "policy.path_duplicate",
        )?;
        Ok(hash)
    };
    Ok(
        json!({"scientific_revision":POLICY_REVISION,"scientific_sha256":POLICY_SHA256,
        "registry_sha256":pinned("crates/ckks-train-bench/configs/experiment_matrix.json")?,
        "portable_lock_sha256":pinned("crates/ckks-train-bench/configs/experiment_input_locks.json")?,
        "design_documents":context.approved_design_documents().iter().map(|(path,sha256)|json!({"path":path,"sha256":sha256})).collect::<Vec<_>>(),
        "approval_sha256":context.approval_sha256()}),
    )
}
fn encoded(value: &impl Encode) -> Result<Vec<u8>> {
    let mut e = Encoder::new(DocumentLayout::TargetLock);
    value.encode(&mut e)?;
    Ok(e.bytes)
}
fn projection(p: &ParsedReferenceDocument) -> CompletionProjection {
    let c = &p.body.completion;
    CompletionProjection {
        origin_completed_updates: c.origin_completed_updates,
        origin_processed_samples: c.origin_processed_samples,
        planned_updates: c.planned_updates,
        actual_updates: c.actual_updates,
        planned_samples: c.planned_samples,
        actual_samples: c.actual_samples,
        planned_dropped_samples: c.planned_dropped_samples,
        actual_dropped_samples: c.actual_dropped_samples,
        full_plan_sha256: c.full_plan_sha256.clone(),
        consumed_plan_sha256: c.consumed_plan_sha256.clone(),
        final_endpoint_index: c.final_endpoint_index,
        final_endpoint: p
            .body
            .endpoints
            .0
            .last()
            .expect("parsed reference endpoint")
            .clone(),
        final_state_sha256: c.final_state_sha256.clone(),
    }
}
fn source_matches(lock: &Lock, expected: &ReferenceLink, contract: &[u8]) -> Result<()> {
    ensure(
        lock.contract.bytes == contract && lock.contract.sha256.0 == sha256_hex(contract),
        DocumentErrorCode::CrossLinkMismatch,
        "reference.identity",
    )?;
    ensure(
        encoded(&lock.reference)? == encoded(expected)?,
        DocumentErrorCode::CrossLinkMismatch,
        "reference.linkage",
    )
}
fn policy_matches(actual: &Policy, expected: &Policy) -> Result<()> {
    ensure(
        encoded(actual)? == encoded(expected)?,
        DocumentErrorCode::CrossLinkMismatch,
        "policy",
    )
}
pub(in crate::study_reference) fn verify_link(
    lock: &Lock,
    reference: &VerifiedReference,
    context: &VerifiedStudyContext,
    campaign: &str,
    profile: &str,
) -> Result<()> {
    let p = reference.parsed();
    ensure(
        lock.campaign_id.0 == campaign
            && lock.profile_id.0 == profile
            && reference.profile_id() == profile
            && reference.receipt().seed() == 0,
        DocumentErrorCode::CrossLinkMismatch,
        "selector",
    )?;
    let expected = ReferenceLink {
        run_id: Text(reference.run_id().into()),
        schema: Text(REFERENCE.into()),
        kind: Kind::Reference,
        link: Link {
            file_sha256: Hash(reference.file_sha256().into()),
            payload_sha256: Hash(reference.payload_sha256().into()),
        },
        seed: 0,
        seed_receipt_sha256: Hash(reference.receipt().digest().into()),
        materialized_sha256: Hash(reference.receipt().materialized_digest().into()),
        writer_capsule_sha256: Hash(reference.writer_capsule_sha256().into()),
        start_unix_ns: p.body.timing.start_unix_ns,
        completion_unix_ns: p.body.timing.end_unix_ns,
        completion: projection(reference.parsed()),
        final_observation: p.body.final_observation.clone(),
    };
    source_matches(lock, &expected, reference.contract().canonical_bytes())?;
    let policy = serde_json::value::to_raw_value(&policy_value(context)?)
        .map_err(|e| json_err(e, "policy"))?;
    policy_matches(&lock.policy, &Policy::decode(&policy, "policy")?)?;
    reader::approved_provenance_claim(&lock.provenance, context)?;
    Ok(())
}
/// Only called with the originating store's exact verified snapshot and captured writer facts.
pub(in crate::study_reference) fn serialize_lock(
    reference: &VerifiedReference,
    reference_bytes: &[u8],
    capture: &CaptureFact<'_>,
    campaign: &str,
    unix_ns: u64,
    elapsed_ns: u64,
) -> Result<Vec<u8>> {
    ensure(
        reference.receipt().seed() == 0,
        DocumentErrorCode::CrossLinkMismatch,
        "reference.seed",
    )?;
    ensure(
        sha256_hex(reference_bytes) == reference.file_sha256(),
        DocumentErrorCode::CrossLinkMismatch,
        "reference.bytes",
    )?;
    let parsed = reference.parsed();
    let c = &parsed.body.completion;
    let e = parsed.body.endpoints.0.last().expect("verified endpoint");
    let accuracy = |a: &ClaimedAccuracy| json!({"correct":a.correct,"test_count":a.test_count,"value":f64::from_bits(a.bits),"bits":format!("{:016x}",a.bits)});
    let endpoint = json!({"completed_updates":e.completed_updates,"processed_samples":e.processed_samples,"crossed_logical_boundaries":e.crossed_logical_boundaries.0,"exact_logical_boundary":e.exact_logical_boundary,
        "interval_updates":e.interval_updates,"interval_samples":e.interval_samples,"observation":accuracy(&e.observation),
        "update_elapsed_ns":{"status":"available","value":e.update_elapsed_ns.available().expect("verified timing")},"evaluation_elapsed_ns":{"status":"available","value":e.evaluation_elapsed_ns.available().expect("verified timing")}});
    let completion = json!({"origin_completed_updates":c.origin_completed_updates,"origin_processed_samples":c.origin_processed_samples,
        "planned_updates":c.planned_updates,"actual_updates":c.actual_updates,"planned_samples":c.planned_samples,"actual_samples":c.actual_samples,
        "planned_dropped_samples":c.planned_dropped_samples,"actual_dropped_samples":c.actual_dropped_samples,"full_plan_sha256":c.full_plan_sha256.0,"consumed_plan_sha256":c.consumed_plan_sha256.0,
        "final_endpoint_index":c.final_endpoint_index,"final_endpoint":endpoint,"final_state_sha256":c.final_state_sha256.0});
    let (_, bits) = target_bits(reference.final_correct(), reference.test_count())?;
    let mut value = json!({"schema":SCHEMA,"kind":"target_lock","status":"complete","writer":{"api_version":API,"algorithm_revision":PROTOCOL},"synthetic":false,
        "campaign_id":campaign,"profile_id":reference.profile_id(),"lock_id":lock_id(campaign,reference.profile_id())?,"policy":policy_value(capture.context)?,
        "contract":{"schema":CONTRACT_VERSION,"encoding":ENCODING_VERSION,"canonical_bytes":producer::hex(reference.contract().canonical_bytes()),"sha256":reference.contract().digest()},
        "reference":{"run_id":reference.run_id(),"schema":REFERENCE,"kind":"full_fp32_final","link":{"file_sha256":reference.file_sha256(),"payload_sha256":reference.payload_sha256()},"seed":0,
            "seed_receipt_sha256":reference.receipt().digest(),"materialized_sha256":reference.receipt().materialized_digest(),"writer_capsule_sha256":reference.writer_capsule_sha256(),
            "start_unix_ns":reference.verified_time().start_unix_ns(),"completion_unix_ns":reference.verified_time().completion_unix_ns(),"completion":completion,"final_observation":accuracy(&parsed.body.final_observation)},
        "algorithm":ALGORITHM,"target":{"value":f64::from_bits(bits),"bits":format!("{bits:016x}")},
        "created":{"wall_source":"unix_system_time","unix_ns":unix_ns,"monotonic_source":"std_instant","elapsed_since_operation_start_ns":elapsed_ns},
        "provenance":producer::captured(capture)?,"integrity":{"encoding":ENCODING_VERSION,"hash":"sha256","payload_sha256":"0".repeat(64)}});
    let provisional = serde_json::to_vec(&value).map_err(|e| json_err(e, ""))?;
    let lock = decode_lock(&provisional)?;
    validate(&lock, SCHEMA, REFERENCE, false)?;
    value["integrity"]["payload_sha256"] = json!(sha256_hex(&lock.canonical()?));
    let bytes = serde_json::to_vec(&value).map_err(|e| json_err(e, ""))?;
    let lock = parse_lock(&bytes)?;
    verify_link(
        &lock,
        reference,
        capture.context,
        campaign,
        reference.profile_id(),
    )?;
    Ok(bytes)
}

#[cfg(test)]
#[path = "target_link_tests.rs"]
mod linkage_tests;

// Narrow inert projections shared with admission/start consistency checks.
#[allow(dead_code)] // Used by the separately released inert admission codec.
pub(super) struct LockClaims<'a> {
    pub(super) synthetic: bool,
    pub(super) campaign: &'a Text<128>,
    pub(super) profile: &'a Text<128>,
    pub(super) id: &'a Hash,
    pub(super) policy: &'a Policy,
    pub(super) contract: &'a ClaimedIdentityBytes,
    pub(super) reference: &'a Link,
    pub(super) created_unix_ns: u64,
}
impl Lock {
    #[allow(dead_code)] // Future live reader supplies this inert projection.
    pub(super) fn claims(&self) -> LockClaims<'_> {
        LockClaims {
            synthetic: self.synthetic,
            campaign: &self.campaign_id,
            profile: &self.profile_id,
            id: &self.lock_id,
            policy: &self.policy,
            contract: &self.contract,
            reference: &self.reference.link,
            created_unix_ns: self.created.unix_ns,
        }
    }
}
#[cfg(test)]
impl SyntheticTargetLock {
    pub(super) fn claims_lock(&self) -> &Lock {
        &self.0
    }
}
pub(super) fn validate_policy(policy: &Policy) -> Result<()> {
    ensure(
        policy.scientific_revision.0.0 == POLICY_REVISION
            && policy.scientific_sha256.0 == POLICY_SHA256,
        DocumentErrorCode::CrossLinkMismatch,
        "policy.scientific",
    )?;
    ensure(
        policy
            .design_documents
            .0
            .iter()
            .map(|d| d.path.0.as_str())
            .eq(DESIGNS),
        DocumentErrorCode::CrossLinkMismatch,
        "policy.design_documents",
    )
}

impl Lock {
    // All available duplicate claims; writer_capsule remains a claim that needs
    // actual approved build-receipt verification by the future live reader.
    #[allow(dead_code)]
    pub(super) fn check_reference_claims(&self, p: &ParsedReferenceDocument) -> Result<()> {
        let r = &self.reference;
        let c = &p.common;
        let b = &p.body;
        let i = &b.identity;
        ensure(
            self.synthetic == c.synthetic
                && r.run_id == c.run_id
                && r.schema == c.schema
                && r.kind == c.kind
                && r.link.file_sha256.0 == c.file_sha256
                && r.link.payload_sha256 == c.integrity.payload_sha256
                && self.profile_id == i.profile_id
                && r.seed == i.seed
                && r.seed_receipt_sha256 == i.seed_receipt.sha256
                && r.materialized_sha256 == i.materialized_sha256
                && r.start_unix_ns == b.timing.start_unix_ns
                && r.completion_unix_ns == b.timing.end_unix_ns
                && r.final_observation == b.final_observation,
            DocumentErrorCode::CrossLinkMismatch,
            "reference.claims",
        )?;
        ensure(
            encoded(&r.completion)? == encoded(&projection(p))?
                && self.contract.schema == i.contract.schema
                && self.contract.encoding == i.contract.encoding
                && self.contract.bytes == i.contract.bytes
                && self.contract.sha256 == i.contract.sha256,
            DocumentErrorCode::CrossLinkMismatch,
            "reference.projection",
        )
    }
}
