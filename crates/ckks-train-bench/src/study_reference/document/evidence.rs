//! Private inert D7 claims. Syntax, integrity and supplied-record consistency
//! are NOT retained-file custody, verified target, current authority, complete
//! liveness model, capacity reservation or a live gate. No capability conversion.
#![allow(dead_code)] // Live readers/constructors are deliberately absent.
use super::admission;
use super::*;
pub(super) mod wire;
use wire::{Bytes, Cursor, Git, H, Id, Str, Tag, Wire, wire_record};
mod auxiliary;
use auxiliary::*;
mod tails;
pub(super) use tails::FileRef;
use tails::*;
pub(super) mod observed;
mod quick;
use quick::*;
pub(super) const QUICK_K_MAX: usize = 1_048_000;
const GATE_NAMES: [&str; 5] = ["inputs", "implementation", "layout", "resource", "user"];
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Production,
    #[cfg(test)]
    Oracle,
}
impl Mode {
    pub(super) fn synthetic(self) -> bool {
        match self {
            Self::Production => false,
            #[cfg(test)]
            Self::Oracle => true,
        }
    }
    pub(super) fn domain(self, name: &str) -> String {
        format!(
            "hekit-{}-{name}-v1",
            if self.synthetic() { "test" } else { "study" }
        )
    }
    fn policy(self, index: usize) -> String {
        format!(
            "{}d7-{}-v1",
            if self.synthetic() { "test-" } else { "" },
            GATE_NAMES[index]
        )
    }
}
fn same(ok: bool, p: &str) -> Result<()> {
    ensure(ok, DocumentErrorCode::CrossLinkMismatch, p)
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| err(DocumentErrorCode::TooLarge, "evidence.arithmetic"))
}
pub(super) fn check_transport_length(n: usize, canonical_quick: bool) -> Result<()> {
    ensure(
        n <= if canonical_quick { QUICK_K_MAX } else { A_MAX },
        DocumentErrorCode::TooLarge,
        "evidence.length",
    )
}
fn record_cursor<'a>(bytes: &'a [u8], domain: &str, mode: Mode) -> Result<Cursor<'a>> {
    check_transport_length(bytes.len(), false)?;
    let mut c = Cursor::new(bytes);
    ensure(
        c.bounded_text(128)? == ENCODING_VERSION && c.bounded_text(128)? == mode.domain(domain),
        DocumentErrorCode::UnsupportedSchema,
        "evidence.header",
    )?;
    ensure(
        c.tag(1)? == u8::from(mode.synthetic()),
        DocumentErrorCode::UnsupportedSchema,
        "evidence.synthetic",
    )?;
    Ok(c)
}
fn record_encoder(domain: &str, mode: Mode) -> Result<Encoder> {
    let layout = if matches!(domain, "quick-preflight" | "quick-start" | "quick-end") {
        DocumentLayout::Quick
    } else {
        DocumentLayout::Start
    };
    let mut e = Encoder::new(layout);
    e.text(ENCODING_VERSION)?;
    e.text(&mode.domain(domain))?;
    e.byte(u8::from(mode.synthetic()))?;
    Ok(e)
}
fn digest(bytes: &[u8], h: H<'_>, p: &str) -> Result<()> {
    ensure(
        sha256_hex(bytes) == h.0,
        DocumentErrorCode::DigestMismatch,
        p,
    )
}
wire_record!(Common {
    policy_revision:Id<'a>,scope:Tag<1>,execution:Bytes<'a>,contract_sha256:H<'a>,selected_receipt_sha256:H<'a>,materialized_sha256:H<'a>,campaign_id:Id<'a>,profile_id:Id<'a>,approval_sha256:H<'a>,train_receipt_sha256:H<'a>,executable_sha256:H<'a>,source_manifest_sha256:H<'a>,source_revision:Git<'a>
});
wire_record!(FullBody {case:Bytes<'a>,case_id:H<'a>,lock:Link,reference:Link,lock_id:H<'a>,threshold_bits:u64,attempt_id:Id<'a>,predecessor:Option<admission::Predecessor>});
wire_record!(QuickBody {registration:Bytes<'a>,registration_id:H<'a>,operational_identity:Bytes<'a>,operational_id:H<'a>,seed:u64,consumption:Tag<1>,consumed_plan_sha256:H<'a>,request_sha256:H<'a>,operational_attempt:Id<'a>});
#[derive(Debug, Clone)]
pub(super) enum Branch<'a> {
    Full(FullBody<'a>),
    Quick(QuickBody<'a>),
}
impl Branch<'_> {
    fn registration(&self) -> &[u8] {
        match self {
            Self::Full(v) => v.case.0,
            Self::Quick(v) => v.registration.0,
        }
    }
    fn registration_id(&self) -> H<'_> {
        match self {
            Self::Full(v) => v.case_id,
            Self::Quick(v) => v.registration_id,
        }
    }
    fn attempt(&self) -> Id<'_> {
        match self {
            Self::Full(v) => v.attempt_id,
            Self::Quick(v) => v.operational_attempt,
        }
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        match self {
            Self::Full(v) => v.write(e),
            Self::Quick(v) => v.write(e),
        }
    }
}
#[derive(Debug, Clone)]
enum Tail<'a> {
    Inputs(Inputs<'a>),
    Implementation(Implementation<'a>),
    Layout(Layout<'a>),
    Resource(Resource<'a>),
    User(User<'a>),
}
#[derive(Debug, Clone)]
pub(super) struct Evidence<'a> {
    mode: Mode,
    gate: usize,
    common: Common<'a>,
    branch: Branch<'a>,
    tail: Tail<'a>,
    bytes: &'a [u8],
}
impl<'a> Evidence<'a> {
    pub(super) fn parse(bytes: &'a [u8]) -> Result<Self> {
        Self::decode(bytes, Mode::Production)
    }
    #[cfg(test)]
    pub(super) fn parse_oracle(bytes: &'a [u8]) -> Result<Self> {
        Self::decode(bytes, Mode::Oracle)
    }
    fn decode(bytes: &'a [u8], mode: Mode) -> Result<Self> {
        check_transport_length(bytes.len(), false)?;
        let mut header = Cursor::new(bytes);
        let _ = header.bounded_text(128)?;
        let domain = header.bounded_text(128)?;
        let gate = GATE_NAMES
            .iter()
            .position(|name| domain == mode.domain(&format!("gate-{name}-evidence")))
            .ok_or_else(|| err(DocumentErrorCode::UnsupportedSchema, "evidence.domain"))?;
        let mut c = record_cursor(bytes, &format!("gate-{}-evidence", GATE_NAMES[gate]), mode)?;
        let common = Common::read(&mut c)?;
        let branch = if common.scope.0 == 0 {
            Branch::Full(FullBody::read(&mut c)?)
        } else {
            Branch::Quick(QuickBody::read(&mut c)?)
        };
        let tail = match gate {
            0 => Tail::Inputs(Inputs::read_branch(&mut c, common.scope.0)?),
            1 => Tail::Implementation(Implementation::read(&mut c)?),
            2 => Tail::Layout(Layout::read(&mut c)?),
            3 => Tail::Resource(Resource::read(&mut c)?),
            _ => Tail::User(User::read(&mut c)?),
        };
        c.finish()?;
        let value = Self {
            mode,
            gate,
            common,
            branch,
            tail,
            bytes,
        };
        value.validate()?;
        Ok(value)
    }
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = record_encoder(
            &format!("gate-{}-evidence", GATE_NAMES[self.gate]),
            self.mode,
        )?;
        self.common.write(&mut e)?;
        self.branch.write(&mut e)?;
        match &self.tail {
            Tail::Inputs(v) => v.write(&mut e)?,
            Tail::Implementation(v) => v.write(&mut e)?,
            Tail::Layout(v) => v.write(&mut e)?,
            Tail::Resource(v) => v.write(&mut e)?,
            Tail::User(v) => v.write(&mut e)?,
        };
        Ok(e.bytes)
    }
    fn validate(&self) -> Result<()> {
        let c = &self.common;
        ensure(
            c.policy_revision.0 == self.mode.policy(self.gate),
            DocumentErrorCode::UnsupportedSchema,
            "evidence.policy",
        )?;
        let (execution, case) = validate_common_branch(c, &self.branch, self.mode)?;
        let x = execution.fields();
        match &self.tail {
            Tail::Inputs(v) => v.validate(self, &case)?,
            Tail::Implementation(v) => v.validate(self, &x, &case)?,
            Tail::Layout(v) => v.validate(self, &x)?,
            Tail::Resource(v) => v.validate(self, &x, &case)?,
            Tail::User(v) => v.validate(self, &case)?,
        };
        Ok(())
    }
    fn consumed(&self, bytes: &[u8], hash: H<'_>) -> Result<()> {
        digest(bytes, hash, "evidence.consumption.digest")?;
        let c = Auxiliary::decode(AuxKind::Consumption, bytes, self.mode)?;
        let p = &c.consumption()?.prefix;
        let case = admission::parse_case_binary(self.branch.registration())?;
        let execution = admission::parse_execution_binary(self.common.execution.0)?;
        same(
            p.consumption == self.common.scope
                && p.seed == case.seed
                && p.selected_receipt_sha256 == self.common.selected_receipt_sha256
                && p.trainer_sha256.0 == execution.fields().trainer,
            "evidence.consumption",
        )?;
        if let Branch::Quick(q) = &self.branch {
            same(
                hash == q.consumed_plan_sha256 && p.request_sha256 == q.request_sha256,
                "evidence.quick.consumption",
            )?;
        }
        Ok(())
    }
}
/// Exact consistency of supplied sidecar claims only, no real gate assembler.
pub(super) fn check_evidence_bundle(records: &[Evidence<'_>]) -> Result<()> {
    check_evidence_refs(&records.iter().collect::<Vec<_>>())
}
pub(super) fn check_evidence_refs(records: &[&Evidence<'_>]) -> Result<()> {
    same(records.len() == 5, "evidence.bundle.count")?;
    let first = records[0];
    let mut common = Encoder::new(DocumentLayout::Start);
    first.common.write(&mut common)?;
    let mut branch = Encoder::new(DocumentLayout::Start);
    first.branch.write(&mut branch)?;
    for (index, record) in records.iter().enumerate() {
        same(
            record.gate == index && record.mode == first.mode,
            "evidence.bundle.kind",
        )?;
        // policy_revision is intentionally different per gate; compare every
        // remaining common field and exact branch bytes.
        let mut c = Encoder::new(DocumentLayout::Start);
        record.common.write(&mut c)?;
        let prefix = 8 + record.common.policy_revision.0.len();
        let first_prefix = 8 + first.common.policy_revision.0.len();
        same(
            c.bytes[prefix..] == common.bytes[first_prefix..],
            "evidence.bundle.common",
        )?;
        let mut b = Encoder::new(DocumentLayout::Start);
        record.branch.write(&mut b)?;
        same(b.bytes == branch.bytes, "evidence.bundle.branch")?;
    }
    // Full/Quick consumed identities and original request must also agree across
    // inputs/layout and the resource accounting limit projection.
    let (inputs, layout, resource) = match (&records[0].tail, &records[2].tail, &records[3].tail) {
        (Tail::Inputs(i), Tail::Layout(l), Tail::Resource(r)) => (i, l, r),
        _ => {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "evidence.bundle.tail",
            ));
        }
    };
    same(
        inputs.consumed_plan.0 == layout.consumed_plan.0,
        "evidence.bundle.consumption",
    )?;
    let request = Auxiliary::decode(AuxKind::Request, layout.request.0, first.mode)?;
    same(
        resource.requested.max_bytes == request.request()?.limits[0]
            && resource.requested.max_records == request.request()?.limits[1],
        "evidence.bundle.resource.request",
    )?;
    let user = match &records[4].tail {
        Tail::User(u) => u,
        _ => {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "evidence.bundle.user",
            ));
        }
    };
    let scope = Auxiliary::decode(AuxKind::UserScope, user.authority_bytes.0, first.mode)?;
    let scope = match &scope.body {
        AuxiliaryBody::UserScope(s) => s,
        _ => unreachable!("closed scope"),
    };
    same(
        scope.scopes[user.selected_scope_index as usize].overlap == resource.claim.overlap,
        "evidence.bundle.overlap",
    )
}
/// Compare Full target claims to independently obtained expected claims. Calling
/// this does not verify a target; a real reader MUST retain VerifiedTargetForSeed,
/// its independent receipt, reference and approved policy (including Retry C1).
pub(super) fn check_full_target_claims(
    record: &Evidence<'_>,
    expected: &Evidence<'_>,
) -> Result<()> {
    let (a, b) = match (&record.branch, &expected.branch) {
        (Branch::Full(a), Branch::Full(b)) => (a, b),
        _ => {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "evidence.full.required",
            ));
        }
    };
    same(
        record.mode == expected.mode
            && a.case.0 == b.case.0
            && a.case_id == b.case_id
            && a.lock_id == b.lock_id
            && a.threshold_bits == b.threshold_bits,
        "evidence.full.target",
    )?;
    let mut x = Encoder::new(DocumentLayout::Start);
    a.lock.write(&mut x)?;
    a.reference.write(&mut x)?;
    let mut y = Encoder::new(DocumentLayout::Start);
    b.lock.write(&mut y)?;
    b.reference.write(&mut y)?;
    same(x.bytes == y.bytes, "evidence.full.links")
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod tests;

fn case_binding_matches(
    case: &admission::CaseIdentity,
    x: &super::execution_binding::ExecutionFields<'_>,
) -> Result<()> {
    use crate::study_profile::{CaseMode, CasePacking, CaseRole};
    let expected = match case.role {
        CaseRole::Main | CaseRole::MainAblation => 0,
        CaseRole::BaselineA => {
            if case.packing == CasePacking::Row2020 {
                1
            } else {
                2
            }
        }
        CaseRole::BaselineB => 4,
        CaseRole::BaselineC => 3,
    };
    same(x.implementation == expected, "evidence.implementation")?;
    same(
        x.bootstrap.is_some() == (case.mode == CaseMode::Noninteractive),
        "evidence.bootstrap",
    )?;
    if let Some(eval) = &case.eval_config {
        same(
            eval.hoist == x.eval.hoist
                && eval.bsgs == x.eval.bsgs
                && eval.lazy_relin == x.eval.lazy_relin,
            "evidence.case.eval",
        )?;
    }
    Ok(())
}

// Inert Request decoding reuse for the closed resource-source Train branch.
// This projection grants no admission, geometry or execution capability.
pub(super) fn resource_request_consumption(bytes: &[u8], mode: Mode) -> Result<u8> {
    Ok(Auxiliary::decode(AuxKind::Request, bytes, mode)?
        .request()?
        .consumption
        .0)
}

#[cfg(test)]
mod cpu_wire_tests;

fn validate_common_branch(
    c: &Common<'_>,
    branch: &Branch<'_>,
    mode: Mode,
) -> Result<(admission::Execution, admission::CaseIdentity)> {
    let execution = admission::parse_execution_binary(c.execution.0)?;
    let x = execution.fields();
    same(
        x.manifest == c.source_manifest_sha256.0 && x.revision == c.source_revision.0,
        "evidence.source",
    )?;
    let registration = branch.registration();
    let case = admission::parse_case_binary(registration)?;
    digest(
        registration,
        branch.registration_id(),
        "evidence.registration_id",
    )?;
    same(case.profile_id.0 == c.profile_id.0, "evidence.profile")?;
    case_binding_matches(&case, &x)?;
    match branch {
        Branch::Full(v) => {
            let threshold = f64::from_bits(v.threshold_bits);
            ensure(
                threshold.is_finite() && !threshold.is_sign_negative() && threshold <= 1.,
                DocumentErrorCode::OutOfRange,
                "evidence.threshold",
            )?;
            if let Some(p) = &v.predecessor {
                same(
                    p.attempt_id.0 != v.attempt_id.0,
                    "evidence.predecessor.self",
                )?;
            }
        }
        Branch::Quick(v) => {
            same(
                case.seed == 0 && v.seed == 0 && v.consumption.0 == 1,
                "evidence.quick.scope",
            )?;
            digest(
                v.operational_identity.0,
                v.operational_id,
                "evidence.quick.identity",
            )?;
            let q = QuickIdentity::decode(v.operational_identity.0, mode)?;
            q.check_common(c)?;
            same(
                q.body.registration.0 == v.registration.0
                    && q.body.registration_id == v.registration_id
                    && q.body.operational_attempt == v.operational_attempt
                    && q.body.consumed_plan_sha256 == v.consumed_plan_sha256
                    && q.body.request_sha256 == v.request_sha256,
                "evidence.quick.duplicates",
            )?;
        }
    }
    Ok((execution, case))
}

#[cfg(test)]
mod cpu_wire_fix1_tests;
