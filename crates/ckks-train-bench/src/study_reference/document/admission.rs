//! Bounded Full admission/start CLAIM codecs. Parsing never grants a live gate,
//! Initial eligibility, verified target, disk snapshot or training authority.
use super::case_provenance::ParsedCaseProvenance;
use super::execution_binding::ExecutionFields;
use super::*;
use crate::study_profile::{
    CaseBackend, CaseEvalConfig, CaseMode, CaseOrigin, CasePacking, CasePlacement, CaseRole,
    CaseScope, SecurityPolicy,
};
const ADMISSION: &str = "hekit-study-he-admission-v1";
const START: &str = "hekit-study-he-start-v1";
const CASE_DOMAIN: &str = "hekit-study-registered-case-v1";
const GATES_DOMAIN: &str = "hekit-study-admission-gates-v1";
const API: &str = "hekit-study-target-api-v1";
const PROTOCOL: &str = "hekit-study-target-protocol-v1";

// Reuse the registry's exact closed vocabulary and numeric discriminants. These
// implementations cannot deserialize or construct a RegisteredCase.
macro_rules! case_tag {
    ($($ty:ty),+ $(,)?) => {$ (
        impl Decode for $ty {
            fn decode(raw: &RawValue, p: &str) -> Result<Self> {
                Text::<128>::decode(raw,p)?.0.parse().map_err(|_| err(DocumentErrorCode::UnsupportedTag,p))
            }
        }
        impl Encode for $ty {
            fn encode(&self,e:&mut Encoder)->Result<()> { e.byte(*self as u8) }
        }
    )+};
}
case_tag!(
    CaseOrigin,
    CaseScope,
    CaseRole,
    CaseBackend,
    CaseMode,
    CasePlacement,
    CasePacking,
    SecurityPolicy
);
impl Decode for CaseEvalConfig {
    fn decode(raw: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(raw, &["hoist", "bsgs", "lazy_relin"], p)?;
        Ok(Self {
            hoist: f.field("hoist", p)?,
            bsgs: f.field("bsgs", p)?,
            lazy_relin: f.field("lazy_relin", p)?,
        })
    }
}
impl Encode for CaseEvalConfig {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.hoist.encode(e)?;
        self.bsgs.encode(e)?;
        self.lazy_relin.encode(e)
    }
}
record!(pub(super) CaseIdentity {
    registry_sha256:Hash, origin:CaseOrigin, registration_id:Text<128>, profile_id:Text<128>, seed:u64,
    scope:CaseScope, role:CaseRole, backend:CaseBackend, mode:CaseMode, placement:CasePlacement,
    packing:CasePacking, security_policy:SecurityPolicy, logical_batch:u64, eval_config:Option<CaseEvalConfig>,
});
fn validate_case(case: &CaseIdentity) -> Result<()> {
    ensure(
        case.scope == CaseScope::Full && case.seed <= 9 && case.logical_batch > 0,
        DocumentErrorCode::OutOfRange,
        "case",
    )?;
    target::valid_id(&case.registration_id.0, "case.registration_id")?;
    target::valid_id(&case.profile_id.0, "case.profile_id")
}
// Shared traversal for the existing actual registry projection AND parsed claims.
// No JSON reconstruction, synthetic registration, or duplicate binary grammar.
pub(super) fn registered_case_bytes(
    case: &crate::study_profile::RegisteredCase,
) -> Result<Vec<u8>> {
    let s = case.selector();
    let fields = CaseIdentity {
        registry_sha256: Hash(case.registry_sha256().into()),
        origin: s.origin,
        registration_id: Text(s.registration_id.clone()),
        profile_id: Text(s.profile_id.clone()),
        seed: s.seed,
        scope: s.scope,
        role: case.role(),
        backend: s.backend,
        mode: s.mode,
        placement: s.placement,
        packing: s.packing,
        security_policy: case.security_policy(),
        logical_batch: case.logical_batch(),
        eval_config: case.eval_config(),
    };
    validate_case(&fields)?;
    ensure(
        fields.registry_sha256.0.len() == 64 && lower_hex(&fields.registry_sha256.0),
        DocumentErrorCode::BadHex,
        "registered_case.registry",
    )?;
    component(Some(CASE_DOMAIN), &fields)
}
tags!(Implementation {
    Main=0=>"main_diagonal", Row=1=>"matched_row2020", Diagonal=2=>"matched_diagonal2020", Packed=3=>"packed_c", External=4=>"external_b",
});
tags!(BootstrapEngine { None=0=>"none", Host=1=>"host", Native=2=>"native_cuda", Upstream=3=>"upstream" });
pub(super) struct Execution {
    implementation: Implementation,
    implementation_revision: Text<128>,
    implementation_manifest_sha256: Hash,
    parameter_identity_sha256: Hash,
    bootstrap_descriptor_sha256: Option<Hash>,
    layout_plan_sha256: Hash,
    trainer_plan_sha256: Hash,
    eval_config: CaseEvalConfig,
    bootstrap_engine: BootstrapEngine,
    key_policy_sha256: Hash,
    security_policy_sha256: Hash,
    physical_binding_sha256: Hash,
    runtime_policy_sha256: Hash,
}
impl Decode for Execution {
    fn decode(raw: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            raw,
            &[
                "implementation",
                "implementation_revision",
                "implementation_manifest_sha256",
                "parameter_identity_sha256",
                "bootstrap_descriptor_sha256",
                "layout_plan_sha256",
                "trainer_plan_sha256",
                "eval_config",
                "bootstrap_engine",
                "key_policy_sha256",
                "security_policy_sha256",
                "physical_binding_sha256",
                "runtime_policy_sha256",
            ],
            p,
        )?;
        Ok(Self {
            implementation: f.field("implementation", p)?,
            implementation_revision: f.field("implementation_revision", p)?,
            implementation_manifest_sha256: f.field("implementation_manifest_sha256", p)?,
            parameter_identity_sha256: f.field("parameter_identity_sha256", p)?,
            bootstrap_descriptor_sha256: f.field("bootstrap_descriptor_sha256", p)?,
            layout_plan_sha256: f.field("layout_plan_sha256", p)?,
            trainer_plan_sha256: f.field("trainer_plan_sha256", p)?,
            eval_config: f.field("eval_config", p)?,
            bootstrap_engine: f.field("bootstrap_engine", p)?,
            key_policy_sha256: f.field("key_policy_sha256", p)?,
            security_policy_sha256: f.field("security_policy_sha256", p)?,
            physical_binding_sha256: f.field("physical_binding_sha256", p)?,
            runtime_policy_sha256: f.field("runtime_policy_sha256", p)?,
        })
    }
}
impl Encode for Execution {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        let (_, bytes) = self.fields().traverse(true)?;
        e.append(&bytes, false)
    }
}

impl Execution {
    pub(super) fn fields(&self) -> ExecutionFields<'_> {
        ExecutionFields {
            implementation: self.implementation as u8,
            revision: &self.implementation_revision.0,
            manifest: &self.implementation_manifest_sha256.0,
            parameter: &self.parameter_identity_sha256.0,
            bootstrap: self
                .bootstrap_descriptor_sha256
                .as_ref()
                .map(|h| h.0.as_str()),
            layout: &self.layout_plan_sha256.0,
            trainer: &self.trainer_plan_sha256.0,
            eval: ckks_train_ckks::EvalConfig {
                hoist: self.eval_config.hoist,
                bsgs: self.eval_config.bsgs,
                lazy_relin: self.eval_config.lazy_relin,
            },
            engine: self.bootstrap_engine as u8,
            keys: &self.key_policy_sha256.0,
            security: &self.security_policy_sha256.0,
            physical: &self.physical_binding_sha256.0,
            runtime: &self.runtime_policy_sha256.0,
        }
    }
}
record!(pub(super) Gate { policy_revision:Text<128>,evidence_sha256:Hash });
record!(pub(super) Gates {
    inputs: Gate,
    implementation: Gate,
    layout: Gate,
    resource: Gate,
    user: Gate
});
record!(pub(super) Predecessor { attempt_id:Text<128>,admission:Link,start:Option<Link> });
// The receipt's inner canonical grammar is deliberately opaque and bounded.
#[derive(Debug)]
struct SelectedInput {
    seed: u64,
    materialized_sha256: Hash,
    seed_receipt: ClaimedIdentityBytes,
}
impl Decode for SelectedInput {
    fn decode(raw: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(raw, &["seed", "materialized_sha256", "seed_receipt"], p)?;
        Ok(Self {
            seed: f.field("seed", p)?,
            materialized_sha256: f.field("materialized_sha256", p)?,
            seed_receipt: ClaimedIdentityBytes::parse(
                f.get("seed_receipt", p)?,
                &path(p, "seed_receipt"),
                RECEIPT_VERSION,
                R_MAX,
            )?,
        })
    }
}
impl Encode for SelectedInput {
    fn encode(&self, e: &mut Encoder) -> Result<()> {
        self.seed.encode(e)?;
        self.materialized_sha256.encode(e)?;
        self.seed_receipt.encode(e)
    }
}
tags!(AdmissionKind { Admission=1=>"he_admission" });
tags!(StartKind { Start=2=>"he_start" });
// Private, inert DTOs. Not Deserialize/Into any live capability.
pub(super) struct Admission {
    pub(super) schema: Text<128>,
    kind: AdmissionKind,
    status: Status,
    writer: Writer,
    synthetic: bool,
    campaign_id: Text<128>,
    profile_id: Text<128>,
    lock_id: Hash,
    policy: Policy,
    lock: Link,
    reference: Link,
    contract: ClaimedIdentityBytes,
    selected_input: SelectedInput,
    case: CaseIdentity,
    pub(super) case_id: Hash,
    execution: Execution,
    pub(super) execution_binding_sha256: Hash,
    attempt_id: Text<128>,
    predecessor: Option<Predecessor>,
    gates: Gates,
    pub(super) gates_sha256: Hash,
    admitted: Clock,
    provenance: ParsedCaseProvenance,
    pub(super) integrity: Integrity,
    pub(super) file_sha256: String,
}
pub(super) struct Start {
    schema: Text<128>,
    kind: StartKind,
    status: Status,
    writer: Writer,
    synthetic: bool,
    campaign_id: Text<128>,
    profile_id: Text<128>,
    lock_id: Hash,
    policy: Policy,
    lock: Link,
    reference: Link,
    case_id: Hash,
    execution_binding_sha256: Hash,
    seed: u64,
    seed_receipt_sha256: Hash,
    materialized_sha256: Hash,
    attempt_id: Text<128>,
    admission: Link,
    started: Clock,
    provenance: ParsedCaseProvenance,
    pub(super) integrity: Integrity,
    pub(super) file_sha256: String,
}
pub(super) fn component(domain: Option<&str>, value: &impl Encode) -> Result<Vec<u8>> {
    let mut e = Encoder::new(DocumentLayout::Admission);
    if let Some(domain) = domain {
        e.text(ENCODING_VERSION)?;
        e.text(domain)?;
    }
    value.encode(&mut e)?;
    Ok(e.bytes)
}
impl Admission {
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        let mut e = Encoder::new(DocumentLayout::Admission);
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
        self.lock.encode(&mut e)?;
        self.reference.encode(&mut e)?;
        self.contract.encode(&mut e)?;
        self.selected_input.encode(&mut e)?;
        self.case.encode(&mut e)?;
        self.case_id.encode(&mut e)?;
        // The actual execution producer and this claim share one traversal.
        self.execution.encode(&mut e)?;
        self.execution_binding_sha256.encode(&mut e)?;
        self.attempt_id.encode(&mut e)?;
        self.predecessor.encode(&mut e)?;
        self.gates.encode(&mut e)?;
        self.gates_sha256.encode(&mut e)?;
        self.admitted.encode(&mut e)?;
        self.provenance.encode(&mut e)?;
        self.integrity.encoding.encode(&mut e)?;
        self.integrity.hash.encode(&mut e)?;
        Ok(e.bytes)
    }
}
impl Start {
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        let mut e = Encoder::new(DocumentLayout::Start);
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
        self.lock.encode(&mut e)?;
        self.reference.encode(&mut e)?;
        self.case_id.encode(&mut e)?;
        self.execution_binding_sha256.encode(&mut e)?;
        self.seed.encode(&mut e)?;
        self.seed_receipt_sha256.encode(&mut e)?;
        self.materialized_sha256.encode(&mut e)?;
        self.attempt_id.encode(&mut e)?;
        self.admission.encode(&mut e)?;
        self.started.encode(&mut e)?;
        self.provenance.encode(&mut e)?;
        self.integrity.encoding.encode(&mut e)?;
        self.integrity.hash.encode(&mut e)?;
        Ok(e.bytes)
    }
}
fn decode_admission(bytes: &[u8]) -> Result<Admission> {
    preflight_layout(bytes, DocumentLayout::Admission)?;
    let raw = serde_json::from_slice::<&RawValue>(bytes).map_err(|e| json_err(e, ""))?;
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
            "lock",
            "reference",
            "contract",
            "selected_input",
            "case",
            "case_id",
            "execution",
            "execution_binding_sha256",
            "attempt_id",
            "predecessor",
            "gates",
            "gates_sha256",
            "admitted",
            "provenance",
            "integrity",
        ],
        "",
    )?;
    Ok(Admission {
        schema: f.field("schema", "")?,
        kind: f.field("kind", "")?,
        status: f.field("status", "")?,
        writer: f.field("writer", "")?,
        synthetic: f.field("synthetic", "")?,
        campaign_id: f.field("campaign_id", "")?,
        profile_id: f.field("profile_id", "")?,
        lock_id: f.field("lock_id", "")?,
        policy: f.field("policy", "")?,
        lock: f.field("lock", "")?,
        reference: f.field("reference", "")?,
        contract: ClaimedIdentityBytes::parse(
            f.get("contract", "")?,
            "contract",
            CONTRACT_VERSION,
            C_MAX,
        )?,
        selected_input: f.field("selected_input", "")?,
        case: f.field("case", "")?,
        case_id: f.field("case_id", "")?,
        execution: f.field("execution", "")?,
        execution_binding_sha256: f.field("execution_binding_sha256", "")?,
        attempt_id: f.field("attempt_id", "")?,
        predecessor: f.field("predecessor", "")?,
        gates: f.field("gates", "")?,
        gates_sha256: f.field("gates_sha256", "")?,
        admitted: f.field("admitted", "")?,
        provenance: f.field("provenance", "")?,
        integrity: f.field("integrity", "")?,
        file_sha256: sha256_hex(bytes),
    })
}
fn decode_start(bytes: &[u8]) -> Result<Start> {
    preflight_layout(bytes, DocumentLayout::Start)?;
    let raw = serde_json::from_slice::<&RawValue>(bytes).map_err(|e| json_err(e, ""))?;
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
            "lock",
            "reference",
            "case_id",
            "execution_binding_sha256",
            "seed",
            "seed_receipt_sha256",
            "materialized_sha256",
            "attempt_id",
            "admission",
            "started",
            "provenance",
            "integrity",
        ],
        "",
    )?;
    Ok(Start {
        schema: f.field("schema", "")?,
        kind: f.field("kind", "")?,
        status: f.field("status", "")?,
        writer: f.field("writer", "")?,
        synthetic: f.field("synthetic", "")?,
        campaign_id: f.field("campaign_id", "")?,
        profile_id: f.field("profile_id", "")?,
        lock_id: f.field("lock_id", "")?,
        policy: f.field("policy", "")?,
        lock: f.field("lock", "")?,
        reference: f.field("reference", "")?,
        case_id: f.field("case_id", "")?,
        execution_binding_sha256: f.field("execution_binding_sha256", "")?,
        seed: f.field("seed", "")?,
        seed_receipt_sha256: f.field("seed_receipt_sha256", "")?,
        materialized_sha256: f.field("materialized_sha256", "")?,
        attempt_id: f.field("attempt_id", "")?,
        admission: f.field("admission", "")?,
        started: f.field("started", "")?,
        provenance: f.field("provenance", "")?,
        integrity: f.field("integrity", "")?,
        file_sha256: sha256_hex(bytes),
    })
}
fn header(
    schema: &Text<128>,
    expected: &str,
    synthetic: bool,
    expected_synthetic: bool,
    status: Status,
    writer: &Writer,
    integrity: &Integrity,
) -> Result<()> {
    ensure(
        schema.0 == expected && synthetic == expected_synthetic,
        DocumentErrorCode::UnsupportedSchema,
        "schema",
    )?;
    ensure(
        status == Status::Complete,
        DocumentErrorCode::UnsupportedTag,
        "status",
    )?;
    ensure(
        writer.api_version.0 == API && writer.algorithm_revision.0 == PROTOCOL,
        DocumentErrorCode::UnsupportedSchema,
        "writer",
    )?;
    ensure(
        integrity.encoding.0 == ENCODING_VERSION && integrity.hash.0 == "sha256",
        DocumentErrorCode::UnsupportedSchema,
        "integrity",
    )
}
fn selector(campaign: &str, profile: &str, attempt: &str, lock_id: &Hash) -> Result<()> {
    for (id, p) in [
        (campaign, "campaign_id"),
        (profile, "profile_id"),
        (attempt, "attempt_id"),
    ] {
        target::valid_id(id, p)?;
    }
    ensure(
        lock_id.0 == target::lock_id(campaign, profile)?,
        DocumentErrorCode::CrossLinkMismatch,
        "lock_id",
    )
}
fn validate_admission(a: &Admission, schema: &str, synthetic: bool) -> Result<()> {
    header(
        &a.schema,
        schema,
        a.synthetic,
        synthetic,
        a.status,
        &a.writer,
        &a.integrity,
    )?;
    selector(
        &a.campaign_id.0,
        &a.profile_id.0,
        &a.attempt_id.0,
        &a.lock_id,
    )?;
    target::validate_policy(&a.policy)?;
    validate_case(&a.case)?;
    ensure(
        a.case.profile_id == a.profile_id
            && a.case.registry_sha256 == a.policy.registry_sha256
            && a.case.seed == a.selected_input.seed,
        DocumentErrorCode::CrossLinkMismatch,
        "case",
    )?;
    ensure(
        sha256_hex(&component(Some(CASE_DOMAIN), &a.case)?) == a.case_id.0,
        DocumentErrorCode::CrossLinkMismatch,
        "case_id",
    )?;
    nonempty(
        &a.execution.implementation_revision,
        "execution.implementation_revision",
    )?;
    ensure(
        sha256_hex(&a.execution.fields().binding_bytes()?) == a.execution_binding_sha256.0,
        DocumentErrorCode::CrossLinkMismatch,
        "execution_binding_sha256",
    )?;
    for gate in [
        &a.gates.inputs,
        &a.gates.implementation,
        &a.gates.layout,
        &a.gates.resource,
        &a.gates.user,
    ] {
        nonempty(&gate.policy_revision, "gates.policy_revision")?;
    }
    ensure(
        sha256_hex(&component(Some(GATES_DOMAIN), &a.gates)?) == a.gates_sha256.0,
        DocumentErrorCode::CrossLinkMismatch,
        "gates_sha256",
    )?;
    if let Some(p) = &a.predecessor {
        target::valid_id(&p.attempt_id.0, "predecessor.attempt_id")?;
        ensure(
            p.attempt_id != a.attempt_id,
            DocumentErrorCode::CrossLinkMismatch,
            "predecessor.attempt_id",
        )?;
    }
    ensure(
        sha256_hex(&a.canonical()?) == a.integrity.payload_sha256.0,
        DocumentErrorCode::DigestMismatch,
        "integrity.payload_sha256",
    )
}
fn validate_start(s: &Start, schema: &str, synthetic: bool) -> Result<()> {
    header(
        &s.schema,
        schema,
        s.synthetic,
        synthetic,
        s.status,
        &s.writer,
        &s.integrity,
    )?;
    selector(
        &s.campaign_id.0,
        &s.profile_id.0,
        &s.attempt_id.0,
        &s.lock_id,
    )?;
    target::validate_policy(&s.policy)?;
    ensure(s.seed <= 9, DocumentErrorCode::OutOfRange, "seed")?;
    ensure(
        sha256_hex(&s.canonical()?) == s.integrity.payload_sha256.0,
        DocumentErrorCode::DigestMismatch,
        "integrity.payload_sha256",
    )
}
pub(super) fn parse_admission(bytes: &[u8]) -> Result<Admission> {
    let a = decode_admission(bytes)?;
    validate_admission(&a, ADMISSION, false)?;
    Ok(a)
}
// Standalone parse validates transport/integrity only. It can NEVER fully verify
// a start, even if no linked-claim inconsistency has yet been observed.
pub(super) fn parse_start(bytes: &[u8]) -> Result<Start> {
    let s = decode_start(bytes)?;
    validate_start(&s, START, false)?;
    Ok(s)
}
fn equal(a: &impl Encode, b: &impl Encode, p: &str) -> Result<()> {
    ensure(
        component(None, a)? == component(None, b)?,
        DocumentErrorCode::CrossLinkMismatch,
        p,
    )
}
fn identity_equal(a: &ClaimedIdentityBytes, b: &ClaimedIdentityBytes, p: &str) -> Result<()> {
    ensure(
        a.schema == b.schema
            && a.encoding == b.encoding
            && a.bytes == b.bytes
            && a.sha256 == b.sha256,
        DocumentErrorCode::CrossLinkMismatch,
        p,
    )
}
fn link_matches(link: &Link, file: &str, payload: &str, p: &str) -> Result<()> {
    ensure(
        link.file_sha256.0 == file && link.payload_sha256.0 == payload,
        DocumentErrorCode::CrossLinkMismatch,
        p,
    )
}
/// Pure consistency of supplied inert claims; does not inspect disk or approve evidence.
pub(super) fn check_start_claims(s: &Start, a: &Admission) -> Result<()> {
    ensure(
        s.synthetic == a.synthetic
            && s.campaign_id == a.campaign_id
            && s.profile_id == a.profile_id
            && s.lock_id == a.lock_id
            && s.case_id == a.case_id
            && s.execution_binding_sha256 == a.execution_binding_sha256
            && s.seed == a.selected_input.seed
            && s.seed_receipt_sha256 == a.selected_input.seed_receipt.sha256
            && s.materialized_sha256 == a.selected_input.materialized_sha256
            && s.attempt_id == a.attempt_id,
        DocumentErrorCode::CrossLinkMismatch,
        "start.admission",
    )?;
    equal(&s.policy, &a.policy, "start.policy")?;
    equal(&s.lock, &a.lock, "start.lock")?;
    equal(&s.reference, &a.reference, "start.reference")?;
    link_matches(
        &s.admission,
        &a.file_sha256,
        &a.integrity.payload_sha256.0,
        "start.admission.link",
    )?;
    ensure(
        a.admitted.unix_ns <= s.started.unix_ns,
        DocumentErrorCode::TimingMismatch,
        "started",
    )
}
/// Optional prior start is an explicit supplied state expectation. None is not
/// proof of current filesystem absence or process death. Initial has no prior link.
pub(super) fn check_predecessor_claims(
    a: &Admission,
    prior: &Admission,
    start: Option<&Start>,
) -> Result<()> {
    let p = a
        .predecessor
        .as_ref()
        .ok_or_else(|| err(DocumentErrorCode::CrossLinkMismatch, "predecessor"))?;
    ensure(
        a.synthetic == prior.synthetic
            && a.campaign_id == prior.campaign_id
            && a.profile_id == prior.profile_id
            && a.case_id == prior.case_id
            && a.selected_input.seed == prior.selected_input.seed
            && p.attempt_id == prior.attempt_id
            && a.attempt_id != prior.attempt_id,
        DocumentErrorCode::CrossLinkMismatch,
        "predecessor.case",
    )?;
    identity_equal(&a.contract, &prior.contract, "predecessor.contract")?;
    ensure(
        a.selected_input.materialized_sha256 == prior.selected_input.materialized_sha256,
        DocumentErrorCode::CrossLinkMismatch,
        "predecessor.materialized",
    )?;
    identity_equal(
        &a.selected_input.seed_receipt,
        &prior.selected_input.seed_receipt,
        "predecessor.seed_receipt",
    )?;
    link_matches(
        &p.admission,
        &prior.file_sha256,
        &prior.integrity.payload_sha256.0,
        "predecessor.admission",
    )?;
    ensure(
        prior.admitted.unix_ns <= a.admitted.unix_ns,
        DocumentErrorCode::TimingMismatch,
        "predecessor.admitted",
    )?;
    match (&p.start, start) {
        (None, None) => Ok(()),
        (Some(link), Some(start)) => {
            check_start_claims(start, prior)?;
            link_matches(
                link,
                &start.file_sha256,
                &start.integrity.payload_sha256.0,
                "predecessor.start",
            )?;
            ensure(
                start.started.unix_ns <= a.admitted.unix_ns,
                DocumentErrorCode::TimingMismatch,
                "predecessor.start.time",
            )
        }
        _ => Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "predecessor.start.presence",
        )),
    }
}
/// Exact supplied target/reference byte links and duplicate claims only. The
/// reference has already been parsed by its appropriate production/test codec;
/// this function cannot verify canonical inputs, approved context or namespace.
pub(super) fn check_target_claims(
    a: &Admission,
    lock: &target::Lock,
    reference_bytes: &[u8],
    reference: &ParsedReferenceDocument,
) -> Result<()> {
    check_file_size(reference_bytes.len(), DocumentLayout::Reference)?;
    lock.check_reference_claims(reference)?;
    let t = lock.claims();
    ensure(
        a.synthetic == t.synthetic
            && a.campaign_id == *t.campaign
            && a.profile_id == *t.profile
            && a.lock_id == *t.id,
        DocumentErrorCode::CrossLinkMismatch,
        "target.selector",
    )?;
    equal(&a.policy, t.policy, "target.policy")?;
    identity_equal(&a.contract, t.contract, "target.contract")?;
    link_matches(
        &a.lock,
        lock.file_sha256(),
        lock.payload_sha256(),
        "target.lock",
    )?;
    equal(&a.reference, t.reference, "target.reference")?;
    ensure(
        sha256_hex(reference_bytes) == reference.file_sha256()
            && reference.file_sha256() == a.reference.file_sha256.0
            && reference.payload_sha256() == a.reference.payload_sha256.0,
        DocumentErrorCode::CrossLinkMismatch,
        "target.reference.bytes",
    )?;
    ensure(
        t.created_unix_ns <= a.admitted.unix_ns,
        DocumentErrorCode::TimingMismatch,
        "admitted",
    )
}
/// Representation-level C1 composition against ONE verified target/current policy.
/// It cannot supply live eligibility, filesystem absence, gates or authority.
pub(super) fn check_target_history_claims(
    current: &Admission,
    prior: Option<(&Admission, Option<&Start>)>,
    lock: &target::Lock,
    reference_bytes: &[u8],
    reference: &ParsedReferenceDocument,
) -> Result<()> {
    check_target_claims(current, lock, reference_bytes, reference)?;
    match (current.predecessor.as_ref(), prior) {
        (None, None) => Ok(()),
        (Some(_), Some((prior, start))) => {
            check_target_claims(prior, lock, reference_bytes, reference)?;
            check_predecessor_claims(current, prior, start)
        }
        _ => Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "predecessor.presence",
        )),
    }
}
#[cfg(test)]
pub(super) struct SyntheticAdmission(pub(super) Admission);
#[cfg(test)]
impl SyntheticAdmission {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let a = decode_admission(bytes)?;
        validate_admission(&a, "hekit-test-he-admission-v1", true)?;
        Ok(Self(a))
    }
    pub(super) fn check_target(
        &self,
        lock: &target::SyntheticTargetLock,
        reference_bytes: &[u8],
        reference: &ParsedReferenceDocument,
    ) -> Result<()> {
        check_target_claims(&self.0, lock.claims_lock(), reference_bytes, reference)
    }
}
#[cfg(test)]
pub(super) struct SyntheticStart(pub(super) Start);
#[cfg(test)]
impl SyntheticStart {
    pub(super) fn parse(bytes: &[u8]) -> Result<Self> {
        let s = decode_start(bytes)?;
        validate_start(&s, "hekit-test-he-start-v1", true)?;
        Ok(Self(s))
    }
}

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;

// Decode the frozen binary components into the SAME typed records used by Full.
// Encoding still uses their original canonical traversal; no alternate wire.
pub(super) fn parse_case_binary(bytes: &[u8]) -> Result<CaseIdentity> {
    use super::evidence::wire::{Cursor, H, Id, Wire};
    ensure(
        bytes.len() <= A_MAX,
        DocumentErrorCode::TooLarge,
        "canonical.component",
    )?;
    let mut c = Cursor::new(bytes);
    ensure(
        c.bounded_text(128)? == ENCODING_VERSION && c.bounded_text(128)? == CASE_DOMAIN,
        DocumentErrorCode::UnsupportedSchema,
        "case.header",
    )?;
    macro_rules! tag {($ty:ident; $($v:ident),+) => {{const VALUES:&[$ty]=&[$($ty::$v),+];VALUES[c.tag((VALUES.len()-1) as u8)? as usize]}};}
    let case = CaseIdentity {
        registry_sha256: Hash(H::read(&mut c)?.0.into()),
        origin: tag!(CaseOrigin; Family,Supplement),
        registration_id: Text(Id::read(&mut c)?.0.into()),
        profile_id: Text(Id::read(&mut c)?.0.into()),
        seed: c.u64()?,
        scope: tag!(CaseScope; Full,Quick,ArchivalFull),
        role: tag!(CaseRole; Main,BaselineA,BaselineB,BaselineC,MainAblation),
        backend: tag!(CaseBackend; Cpu,Cuda,ExternalCpu),
        mode: tag!(CaseMode; Interactive,Noninteractive),
        placement: tag!(CasePlacement; ClientRefresh,WhenNeeded,AfterUpdate,AfterActivation,UpstreamVerified),
        packing: tag!(CasePacking; BatchedDiagonal,Row2020,Diagonal2020,AlternatingRebootRowColumn,OfficialUpstream),
        security_policy: tag!(SecurityPolicy; N16Train,N16Boot,Seal2020,UpstreamVerified,ReducedLevelSupplement),
        logical_batch: c.u64()?,
        eval_config: c.option(|c| {
            Ok(CaseEvalConfig {
                hoist: c.tag(1)? != 0,
                bsgs: c.tag(1)? != 0,
                lazy_relin: c.tag(1)? != 0,
            })
        })?,
    };
    c.finish()?;
    validate_case(&case)?;
    ensure(
        component(Some(CASE_DOMAIN), &case)? == bytes,
        DocumentErrorCode::CrossLinkMismatch,
        "case.canonical",
    )?;
    Ok(case)
}
pub(super) fn parse_execution_binary(bytes: &[u8]) -> Result<Execution> {
    use super::evidence::wire::{Cursor, H, Wire};
    ensure(
        bytes.len() <= A_MAX,
        DocumentErrorCode::TooLarge,
        "canonical.component",
    )?;
    let mut c = Cursor::new(bytes);
    ensure(
        c.bounded_text(128)? == ENCODING_VERSION
            && c.bounded_text(128)? == "hekit-study-case-binding-v1",
        DocumentErrorCode::UnsupportedSchema,
        "execution.header",
    )?;
    let implementation = match c.tag(4)? {
        0 => Implementation::Main,
        1 => Implementation::Row,
        2 => Implementation::Diagonal,
        3 => Implementation::Packed,
        _ => Implementation::External,
    };
    let implementation_revision = Text(super::evidence::wire::Git::read(&mut c)?.0.into());
    let hash = |c: &mut Cursor<'_>| H::read(c).map(|h| Hash(h.0.into()));
    let implementation_manifest_sha256 = hash(&mut c)?;
    let parameter_identity_sha256 = hash(&mut c)?;
    let bootstrap_descriptor_sha256 = c.option(hash)?;
    let layout_plan_sha256 = hash(&mut c)?;
    let trainer_plan_sha256 = hash(&mut c)?;
    let eval_config = CaseEvalConfig {
        hoist: c.tag(1)? != 0,
        bsgs: c.tag(1)? != 0,
        lazy_relin: c.tag(1)? != 0,
    };
    let bootstrap_engine = match c.tag(3)? {
        0 => BootstrapEngine::None,
        1 => BootstrapEngine::Host,
        2 => BootstrapEngine::Native,
        _ => BootstrapEngine::Upstream,
    };
    let value = Execution {
        implementation,
        implementation_revision,
        implementation_manifest_sha256,
        parameter_identity_sha256,
        bootstrap_descriptor_sha256,
        layout_plan_sha256,
        trainer_plan_sha256,
        eval_config,
        bootstrap_engine,
        key_policy_sha256: hash(&mut c)?,
        security_policy_sha256: hash(&mut c)?,
        physical_binding_sha256: hash(&mut c)?,
        runtime_policy_sha256: hash(&mut c)?,
    };
    c.finish()?;
    ensure(
        value.fields().binding_bytes()? == bytes,
        DocumentErrorCode::CrossLinkMismatch,
        "execution.canonical",
    )?;
    Ok(value)
}

// Supplied observed-policy composition only. The generic Full parser/traversal
// above is unchanged and no capture/owner/start capability is constructed.
impl Admission {
    pub(super) fn gates(&self) -> &Gates {
        &self.gates
    }
    pub(super) fn admitted(&self) -> &Clock {
        &self.admitted
    }
    pub(super) fn check_observed_claims(
        &self,
        common: &super::evidence::Common<'_>,
        branch: &super::evidence::FullBody<'_>,
        contract_length: u64,
        receipt_length: u64,
    ) -> Result<()> {
        ensure(
            self.campaign_id.0 == common.campaign_id.0
                && self.profile_id.0 == common.profile_id.0
                && self.attempt_id.0 == branch.attempt_id.0
                && self.lock_id.0 == branch.lock_id.0
                && component(Some(CASE_DOMAIN), &self.case)? == branch.case.0
                && self.case_id.0 == branch.case_id.0,
            DocumentErrorCode::CrossLinkMismatch,
            "full.bundle.branch",
        )?;
        equal(&self.lock, &branch.lock, "full.bundle.lock")?;
        equal(&self.reference, &branch.reference, "full.bundle.reference")?;
        equal(
            &self.predecessor,
            &branch.predecessor,
            "full.bundle.predecessor",
        )?;
        ensure(
            self.execution.fields().binding_bytes()? == common.execution.0
                && self.contract.sha256.0 == common.contract_sha256.0
                && self.contract.bytes.len() as u64 == contract_length
                && self.selected_input.seed == self.case.seed
                && self.selected_input.materialized_sha256.0 == common.materialized_sha256.0
                && self.selected_input.seed_receipt.sha256.0 == common.selected_receipt_sha256.0
                && self.selected_input.seed_receipt.bytes.len() as u64 == receipt_length
                && self.policy.approval_sha256.0 == common.approval_sha256.0,
            DocumentErrorCode::CrossLinkMismatch,
            "full.bundle.source_input",
        )?;
        self.provenance.check_observed_context(common)
    }
}
impl Start {
    pub(super) fn started(&self) -> &Clock {
        &self.started
    }
    pub(super) fn check_observed_context(
        &self,
        common: &super::evidence::Common<'_>,
    ) -> Result<()> {
        self.provenance.check_observed_context(common)
    }
}
