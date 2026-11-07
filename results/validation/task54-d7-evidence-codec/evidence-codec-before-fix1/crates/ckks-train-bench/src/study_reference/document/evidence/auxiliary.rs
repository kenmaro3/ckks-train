//! Request, consumption and captured administrative claims. No authority loader.
use super::*;
use crate::study_admission::{GeometryWriter, RequestBody, encode_request_body};
use ckks_train_ckks::EvalConfig;
#[derive(Debug, Clone)]
pub(super) struct Request {
    pub(super) consumption: Tag<1>,
    pub(super) physical_lanes: Option<u64>,
    pub(super) transpose: Option<Vec<i32>>,
    pub(super) eval: [Tag<1>; 3],
    pub(super) limits: [u64; 2],
    pub(super) work: [u64; 5],
}
impl<'a> Wire<'a> for Request {
    const MIN: usize = 62;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let consumption = Tag::read(c)?;
        let physical_lanes = Option::read(c)?;
        let transpose = c.option(Vec::<i32>::read)?;
        let eval = [Tag::read(c)?, Tag::read(c)?, Tag::read(c)?];
        let value = Self {
            consumption,
            physical_lanes,
            transpose,
            eval,
            limits: [c.u64()?, c.u64()?],
            work: [c.u64()?, c.u64()?, c.u64()?, c.u64()?, c.u64()?],
        };
        ensure(
            value.physical_lanes.is_none_or(|n| n > 0),
            DocumentErrorCode::CrossLinkMismatch,
            "request.physical_lanes",
        )?;
        value.projection()?;
        Ok(value)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        encode_request_body(e, self.projection()?)
    }
}
impl Request {
    fn projection(&self) -> Result<RequestBody<'_>> {
        let number =
            |n| usize::try_from(n).map_err(|_| err(DocumentErrorCode::TooLarge, "request.number"));
        let mut limits = [0; 2];
        let mut work = [0; 5];
        for (out, n) in limits.iter_mut().zip(self.limits) {
            *out = number(n)?;
        }
        for (out, n) in work.iter_mut().zip(self.work) {
            *out = number(n)?;
        }
        Ok(RequestBody {
            consumption: self.consumption.0,
            physical_lanes: self.physical_lanes.map(number).transpose()?,
            transpose: self.transpose.as_deref(),
            eval: EvalConfig {
                hoist: self.eval[0].0 != 0,
                bsgs: self.eval[1].0 != 0,
                lazy_relin: self.eval[2].0 != 0,
            },
            limits,
            work,
        })
    }
}
impl GeometryWriter for Encoder {
    type Error = DocumentError;
    fn raw(&mut self, b: &[u8]) -> Result<()> {
        self.append(b, false)
    }
    fn number(&mut self, n: usize) -> Result<()> {
        self.u64(u64::try_from(n).map_err(|_| err(DocumentErrorCode::TooLarge, "request.number"))?)
    }
}
#[derive(Debug, Clone)]
pub(super) struct QuickUpdate {
    pub(super) index: u64,
    pub(super) epoch: u64,
    pub(super) batch: u64,
    pub(super) sample_indices: Vec<u64>,
    pub(super) parts: Vec<Part>,
}
#[derive(Debug, Clone)]
pub(super) struct Part {
    pub(super) start: u64,
    pub(super) valid: u64,
}
impl<'a> Wire<'a> for Part {
    const MIN: usize = 16;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            start: c.u64()?,
            valid: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        e.u64(self.start)?;
        e.u64(self.valid)
    }
}
impl<'a> Wire<'a> for QuickUpdate {
    const MIN: usize = 40;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let v = Self {
            index: c.u64()?,
            epoch: c.u64()?,
            batch: c.u64()?,
            sample_indices: Vec::read(c)?,
            parts: Vec::read(c)?,
        };
        v.validate(None)?;
        Ok(v)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.validate(None)?;
        e.u64(self.index)?;
        e.u64(self.epoch)?;
        e.u64(self.batch)?;
        self.sample_indices.write(e)?;
        self.parts.write(e)
    }
}
impl QuickUpdate {
    fn validate(&self, lanes: Option<u64>) -> Result<()> {
        same(
            !self.sample_indices.is_empty() && !self.parts.is_empty(),
            "consumption.empty",
        )?;
        let mut next = 0;
        for p in &self.parts {
            same(
                p.start == next && p.valid > 0 && lanes.is_none_or(|n| p.valid <= n),
                "consumption.partition",
            )?;
            next = add(next, p.valid)?;
        }
        same(
            next == self.sample_indices.len() as u64,
            "consumption.partition.sum",
        )
    }
}
#[derive(Debug, Clone)]
pub(super) enum ConsumptionBody {
    Full(u64),
    Quick(Vec<QuickUpdate>),
}
wire_record!(ConsumptionPrefix {consumption:Tag<1>,seed:u64,request_sha256:H<'a>,trainer_sha256:H<'a>,selected_receipt_sha256:H<'a>});
#[derive(Debug, Clone)]
pub(super) struct Consumption<'a> {
    pub(super) prefix: ConsumptionPrefix<'a>,
    pub(super) body: ConsumptionBody,
}
impl<'a> Wire<'a> for Consumption<'a> {
    const MIN: usize = 233;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let prefix = ConsumptionPrefix::read(c)?;
        let body = if prefix.consumption.0 == 0 {
            ConsumptionBody::Full(c.u64()?)
        } else {
            same(prefix.seed == 0, "consumption.quick.seed")?;
            let n = c.u64()?;
            same(n == 2, "consumption.quick.count")?;
            ensure(
                c.remaining() >= 80,
                DocumentErrorCode::OutOfRange,
                "consumption.quick.truncated",
            )?;
            let updates = vec![QuickUpdate::read(c)?, QuickUpdate::read(c)?];
            same(
                updates[0].index == 0 && updates[1].index == 1,
                "consumption.quick.indices",
            )?;
            ConsumptionBody::Quick(updates)
        };
        same(prefix.seed <= 9, "consumption.seed")?;
        if let ConsumptionBody::Full(n) = body {
            same(n > 0, "consumption.full.total")?;
        }
        Ok(Self { prefix, body })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.prefix.write(e)?;
        match &self.body {
            ConsumptionBody::Full(n) => e.u64(*n),
            ConsumptionBody::Quick(v) => v.write(e),
        }
    }
}
#[derive(Debug, Clone)]
pub(super) struct AllowedScope<'a> {
    pub(super) branch: Tag<1>,
    pub(super) campaign: Id<'a>,
    pub(super) profile: Id<'a>,
    pub(super) registration_id: H<'a>,
    pub(super) attempt: Id<'a>,
    pub(super) predecessor: Option<admission::Predecessor>,
    pub(super) overlap: Tag<1>,
}
impl<'a> Wire<'a> for AllowedScope<'a> {
    const MIN: usize = 102;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let branch = Tag::read(c)?;
        let campaign = Id::read(c)?;
        let profile = Id::read(c)?;
        let registration_id = H::read(c)?;
        let attempt = Id::read(c)?;
        let predecessor = c.option(admission::Predecessor::read)?;
        let overlap = Tag::read(c)?;
        same(
            branch.0 == 0 || predecessor.is_none(),
            "authority.quick.retry",
        )?;
        if let Some(p) = &predecessor {
            same(p.attempt_id.0 != attempt.0, "authority.retry.self")?;
        }
        Ok(Self {
            branch,
            campaign,
            profile,
            registration_id,
            attempt,
            predecessor,
            overlap,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.branch.write(e)?;
        self.campaign.write(e)?;
        self.profile.write(e)?;
        self.registration_id.write(e)?;
        self.attempt.write(e)?;
        self.predecessor.write(e)?;
        self.overlap.write(e)
    }
}
wire_record!(Hold {branch:Tag<1>,registration_id:H<'a>,reason:Tag<1>});
wire_record!(UserScope {origin:Tag<0>,conversation_id:Id<'a>,message_id:Id<'a>,instruction_bytes:Bytes<'a>,scope_generation:u64,scopes:Vec<AllowedScope<'a>>,holds:Vec<Hold<'a>>});
wire_record!(UserCurrent {conversation_id:Id<'a>,source_head_message_id:Id<'a>,scope_generation:u64,authority_sha256:H<'a>,revoked_scopes:Vec<u64>,revoked_all:Tag<1>,observed:Clock});
#[derive(Debug, Clone, Copy)]
pub(super) enum AuxKind {
    Request,
    Consumption,
    UserScope,
    UserCurrent,
}
impl AuxKind {
    fn domain(self) -> &'static str {
        match self {
            Self::Request => "gate-request",
            Self::Consumption => "gate-consumption",
            Self::UserScope => "user-scope",
            Self::UserCurrent => "user-current",
        }
    }
    #[cfg(test)]
    pub(super) fn fixture_name(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Consumption => "consumption",
            Self::UserScope => "authority.scope",
            Self::UserCurrent => "authority.current",
        }
    }
}
#[derive(Debug, Clone)]
pub(super) enum AuxiliaryBody<'a> {
    Request(Request),
    Consumption(Consumption<'a>),
    UserScope(UserScope<'a>),
    UserCurrent(UserCurrent<'a>),
}
#[derive(Debug, Clone)]
pub(super) struct Auxiliary<'a> {
    pub(super) mode: Mode,
    pub(super) kind: AuxKind,
    pub(super) body: AuxiliaryBody<'a>,
    pub(super) bytes: &'a [u8],
}
impl<'a> Auxiliary<'a> {
    pub(super) fn parse(kind: AuxKind, bytes: &'a [u8]) -> Result<Self> {
        Self::decode(kind, bytes, Mode::Production)
    }
    #[cfg(test)]
    pub(super) fn parse_oracle(kind: AuxKind, bytes: &'a [u8]) -> Result<Self> {
        Self::decode(kind, bytes, Mode::Oracle)
    }
    pub(super) fn decode(kind: AuxKind, bytes: &'a [u8], mode: Mode) -> Result<Self> {
        let mut c = record_cursor(bytes, kind.domain(), mode)?;
        let body = match kind {
            AuxKind::Request => AuxiliaryBody::Request(Request::read(&mut c)?),
            AuxKind::Consumption => AuxiliaryBody::Consumption(Consumption::read(&mut c)?),
            AuxKind::UserScope => AuxiliaryBody::UserScope(UserScope::read(&mut c)?),
            AuxKind::UserCurrent => AuxiliaryBody::UserCurrent(UserCurrent::read(&mut c)?),
        };
        c.finish()?;
        if let AuxiliaryBody::UserCurrent(current) = &body {
            same(
                current.revoked_scopes.windows(2).all(|w| w[0] < w[1]),
                "authority.revocations.order",
            )?;
        }
        Ok(Self {
            mode,
            kind,
            body,
            bytes,
        })
    }
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        let mut e = record_encoder(self.kind.domain(), self.mode)?;
        match &self.body {
            AuxiliaryBody::Request(v) => v.write(&mut e)?,
            AuxiliaryBody::Consumption(v) => v.write(&mut e)?,
            AuxiliaryBody::UserScope(v) => v.write(&mut e)?,
            AuxiliaryBody::UserCurrent(v) => v.write(&mut e)?,
        };
        Auxiliary::decode(self.kind, &e.bytes, self.mode)?;
        Ok(e.bytes)
    }
    pub(super) fn request(&self) -> Result<&Request> {
        match &self.body {
            AuxiliaryBody::Request(v) => Ok(v),
            _ => Err(err(DocumentErrorCode::CrossLinkMismatch, "request.kind")),
        }
    }
    pub(super) fn consumption(&self) -> Result<&Consumption<'a>> {
        match &self.body {
            AuxiliaryBody::Consumption(v) => Ok(v),
            _ => Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "consumption.kind",
            )),
        }
    }
}
/// Exact retained original request equality (including all limits), not a planner.
pub(super) fn check_request_retained(
    a: &Auxiliary<'_>,
    original: &crate::study_admission::DiagonalRequest,
) -> Result<()> {
    let mut e = record_encoder("gate-request", a.mode)?;
    encode_request_body(&mut e, RequestBody::from(original))?;
    same(e.bytes == a.bytes, "request.retained")
}
pub(super) fn check_request_consumption(r: &Auxiliary<'_>, c: &Auxiliary<'_>) -> Result<()> {
    same(r.mode == c.mode, "request.domain")?;
    let rbody = r.request()?;
    let cbody = c.consumption()?;
    same(
        cbody.prefix.request_sha256.0 == sha256_hex(r.bytes)
            && rbody.consumption == cbody.prefix.consumption,
        "consumption.request",
    )?;
    let updates = match &cbody.body {
        ConsumptionBody::Full(n) => *n,
        ConsumptionBody::Quick(v) => {
            for u in v {
                u.validate(rbody.physical_lanes)?;
            }
            v.len() as u64
        }
    };
    // Lower bounds on described work only. Actual stage/geometry/handoff counts
    // still require the retained P3 trainer/owner, never guessed here.
    same(
        updates <= rbody.work[2] && updates <= rbody.limits[1],
        "consumption.work_limit",
    )
}
pub(super) fn check_consumption_exact(a: &Auxiliary<'_>, retained: &Auxiliary<'_>) -> Result<()> {
    same(
        a.bytes == retained.bytes && a.mode == retained.mode,
        "consumption.retained_order",
    )
}
/// Captured source consistency only. Live current entry origin/freshness remains missing.
pub(super) fn check_authority(
    scope: &Auxiliary<'_>,
    current: &Auxiliary<'_>,
    selected: u64,
) -> Result<()> {
    same(scope.mode == current.mode, "authority.domain")?;
    let (s, c) = match (&scope.body, &current.body) {
        (AuxiliaryBody::UserScope(s), AuxiliaryBody::UserCurrent(c)) => (s, c),
        _ => return Err(err(DocumentErrorCode::CrossLinkMismatch, "authority.kind")),
    };
    same(
        c.conversation_id == s.conversation_id
            && c.source_head_message_id == s.message_id
            && c.scope_generation == s.scope_generation
            && c.authority_sha256.0 == sha256_hex(scope.bytes),
        "authority.current",
    )?;
    same(
        !s.instruction_bytes.0.is_empty() && selected < s.scopes.len() as u64,
        "authority.scope",
    )?;
    same(c.revoked_all.0 == 0, "authority.revoked_all")?;
    let mut prior = None;
    for n in &c.revoked_scopes {
        same(
            *n < s.scopes.len() as u64 && prior.is_none_or(|p| p < *n),
            "authority.revocations",
        )?;
        prior = Some(*n);
        same(*n != selected, "authority.selected_revoked")?;
    }
    let chosen = &s.scopes[selected as usize];
    same(
        !s.holds
            .iter()
            .any(|h| h.branch == chosen.branch && h.registration_id == chosen.registration_id),
        "authority.hold",
    )
}
