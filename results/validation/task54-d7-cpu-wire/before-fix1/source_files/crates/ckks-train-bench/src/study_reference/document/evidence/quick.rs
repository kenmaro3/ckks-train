//! Quick identity and operational envelopes. No scientific result or live frame.
use super::*;
wire_record!(IdentityBody {registration:Bytes<'a>,registration_id:H<'a>,execution:Bytes<'a>,execution_sha256:H<'a>,contract_sha256:H<'a>,selected_receipt_sha256:H<'a>,materialized_sha256:H<'a>,campaign_id:Id<'a>,profile_id:Id<'a>,seed:u64,consumption:Tag<1>,consumed_plan:Bytes<'a>,consumed_plan_sha256:H<'a>,request:Bytes<'a>,request_sha256:H<'a>,trainer_sha256:H<'a>,operational_attempt:Id<'a>});
#[derive(Debug, Clone)]
pub(super) struct QuickIdentity<'a> {
    pub(super) body: IdentityBody<'a>,
    mode: Mode,
}
impl<'a> QuickIdentity<'a> {
    pub(super) fn parse(bytes: &'a [u8]) -> Result<Self> {
        Self::decode(bytes, Mode::Production)
    }
    #[cfg(test)]
    pub(super) fn parse_oracle(bytes: &'a [u8]) -> Result<Self> {
        Self::decode(bytes, Mode::Oracle)
    }
    pub(super) fn decode(bytes: &'a [u8], mode: Mode) -> Result<Self> {
        let mut c = record_cursor(bytes, "quick-identity", mode)?;
        let body = IdentityBody::read(&mut c)?;
        c.finish()?;
        let q = Self { body, mode };
        q.validate()?;
        Ok(q)
    }
    fn validate(&self) -> Result<()> {
        let q = &self.body;
        same(q.seed == 0 && q.consumption.0 == 1, "quick.identity.scope")?;
        let case = admission::parse_case_binary(q.registration.0)?;
        same(
            case.seed == 0 && case.profile_id.0 == q.profile_id.0,
            "quick.identity.registration",
        )?;
        digest(
            q.registration.0,
            q.registration_id,
            "quick.identity.registration.digest",
        )?;
        digest(
            q.execution.0,
            q.execution_sha256,
            "quick.identity.execution.digest",
        )?;
        let execution = admission::parse_execution_binary(q.execution.0)?;
        case_binding_matches(&case, &execution.fields())?;
        same(
            execution.fields().trainer == q.trainer_sha256.0,
            "quick.identity.trainer",
        )?;
        digest(
            q.consumed_plan.0,
            q.consumed_plan_sha256,
            "quick.identity.consumption.digest",
        )?;
        digest(
            q.request.0,
            q.request_sha256,
            "quick.identity.request.digest",
        )?;
        let consumption = Auxiliary::decode(AuxKind::Consumption, q.consumed_plan.0, self.mode)?;
        let request = Auxiliary::decode(AuxKind::Request, q.request.0, self.mode)?;
        check_request_consumption(&request, &consumption)?;
        let eval = request.request()?.eval;
        let x = execution.fields();
        same(
            eval[0].0 == u8::from(x.eval.hoist)
                && eval[1].0 == u8::from(x.eval.bsgs)
                && eval[2].0 == u8::from(x.eval.lazy_relin),
            "quick.identity.eval",
        )?;
        let c = &consumption.consumption()?.prefix;
        same(
            c.seed == 0
                && c.consumption.0 == 1
                && c.request_sha256 == q.request_sha256
                && c.trainer_sha256 == q.trainer_sha256
                && c.selected_receipt_sha256 == q.selected_receipt_sha256,
            "quick.identity.consumption",
        )
    }
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = record_encoder("quick-identity", self.mode)?;
        self.body.write(&mut e)?;
        Ok(e.bytes)
    }
    pub(super) fn check_common(&self, c: &Common<'_>) -> Result<()> {
        let q = &self.body;
        same(
            q.execution.0 == c.execution.0
                && q.contract_sha256 == c.contract_sha256
                && q.selected_receipt_sha256 == c.selected_receipt_sha256
                && q.materialized_sha256 == c.materialized_sha256
                && q.campaign_id == c.campaign_id
                && q.profile_id == c.profile_id
                && c.scope.0 == 1,
            "quick.identity.common",
        )
    }
}
#[derive(Debug, Clone)]
pub(super) struct Argv<'a>(Vec<Bytes<'a>>);
impl<'a> Wire<'a> for Argv<'a> {
    const MIN: usize = 8;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        let n = c.bounded_count(256, 8)?;
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "quick.argv.allocate"))?;
        for _ in 0..n {
            let b = c.bounded_bytes(4096)?;
            same(!b.contains(&0), "quick.argv.nul")?;
            v.push(Bytes(b));
        }
        Ok(Self(v))
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.0.len() <= 256,
            DocumentErrorCode::TooLarge,
            "quick.argv.count",
        )?;
        e.u64(self.0.len() as u64)?;
        for b in &self.0 {
            ensure(
                b.0.len() <= 4096,
                DocumentErrorCode::TooLarge,
                "quick.argv.bytes",
            )?;
            same(!b.0.contains(&0), "quick.argv.nul")?;
            b.write(e)?;
        }
        Ok(())
    }
}
wire_record!(Preflight {identity:Bytes<'a>,identity_sha256:H<'a>,gates:admission::Gates,argv:Argv<'a>,source_manifest_sha256:H<'a>,source_revision:Git<'a>,approval_sha256:H<'a>,train_receipt_sha256:H<'a>,executable_sha256:H<'a>,observed:Clock});
wire_record!(Start {identity_sha256:H<'a>,gates:admission::Gates,preflight:Link,started:Clock});
#[derive(Debug, Clone)]
pub(super) struct Failure {
    phase: Tag<3>,
    code: Tag<6>,
}
impl<'a> Wire<'a> for Failure {
    const MIN: usize = 2;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            phase: Tag::read(c)?,
            code: Tag::read(c)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.phase.write(e)?;
        self.code.write(e)
    }
}
wire_record!(End {identity_sha256:H<'a>,gates:admission::Gates,preflight:Link,start:Link,outcome:Tag<1>,completed_updates:u64,failure:Option<Failure>,ended:Clock});
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EventKind {
    Preflight,
    Start,
    End,
}
impl EventKind {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Preflight => "preflight",
            Self::Start => "start",
            Self::End => "end",
        }
    }
    fn domain(self) -> String {
        format!("quick-{}", self.name())
    }
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "quick_preflight" => Ok(Self::Preflight),
            "quick_start" => Ok(Self::Start),
            "quick_end" => Ok(Self::End),
            _ => Err(err(DocumentErrorCode::UnsupportedTag, "quick.kind")),
        }
    }
}
enum EventBody<'a> {
    Preflight(Preflight<'a>),
    Start(Start<'a>),
    End(End<'a>),
}
impl EventBody<'_> {
    fn gates(&self) -> &admission::Gates {
        match self {
            Self::Preflight(v) => &v.gates,
            Self::Start(v) => &v.gates,
            Self::End(v) => &v.gates,
        }
    }
    fn identity(&self) -> H<'_> {
        match self {
            Self::Preflight(v) => v.identity_sha256,
            Self::Start(v) => v.identity_sha256,
            Self::End(v) => v.identity_sha256,
        }
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        match self {
            Self::Preflight(v) => v.write(e),
            Self::Start(v) => v.write(e),
            Self::End(v) => v.write(e),
        }
    }
}
pub(super) fn gates(g: &admission::Gates) -> [&admission::Gate; 5] {
    [
        &g.inputs,
        &g.implementation,
        &g.layout,
        &g.resource,
        &g.user,
    ]
}
fn validate_gates(g: &admission::Gates, mode: Mode) -> Result<()> {
    for (index, gate) in gates(g).into_iter().enumerate() {
        ensure(
            gate.policy_revision.0 == mode.policy(index),
            DocumentErrorCode::UnsupportedSchema,
            "quick.gates.policy",
        )?;
    }
    Ok(())
}
fn decode_event(kind: EventKind, bytes: &[u8], mode: Mode) -> Result<EventBody<'_>> {
    check_transport_length(bytes.len(), true)?;
    let mut c = record_cursor(bytes, &kind.domain(), mode)?;
    let body = match kind {
        EventKind::Preflight => EventBody::Preflight(Preflight::read(&mut c)?),
        EventKind::Start => EventBody::Start(Start::read(&mut c)?),
        EventKind::End => EventBody::End(End::read(&mut c)?),
    };
    c.finish()?;
    validate_gates(body.gates(), mode)?;
    match &body {
        EventBody::Preflight(p) => {
            digest(p.identity.0, p.identity_sha256, "quick.preflight.identity")?;
            let q = QuickIdentity::decode(p.identity.0, mode)?;
            let x = admission::parse_execution_binary(q.body.execution.0)?;
            let f = x.fields();
            same(
                p.source_manifest_sha256.0 == f.manifest && p.source_revision.0 == f.revision,
                "quick.preflight.source",
            )?;
        }
        EventBody::End(e) => {
            same(e.completed_updates <= 2, "quick.end.updates")?;
            same(
                if e.outcome.0 == 0 {
                    e.completed_updates == 2 && e.failure.is_none()
                } else {
                    e.failure.is_some()
                },
                "quick.end.outcome",
            )?;
        }
        EventBody::Start(_) => {}
    };
    Ok(body)
}
/// Owns bounded transport only. No deserialized continuation or authority exists.
#[derive(Debug, Clone)]
pub(super) struct QuickEvent {
    kind: EventKind,
    mode: Mode,
    canonical: Vec<u8>,
    file_sha256: String,
}
impl QuickEvent {
    pub(super) fn parse_json(raw: &[u8]) -> Result<Self> {
        Self::decode_json(raw, Mode::Production)
    }
    #[cfg(test)]
    pub(super) fn parse_json_oracle(raw: &[u8]) -> Result<Self> {
        Self::decode_json(raw, Mode::Oracle)
    }
    fn decode_json(raw: &[u8], mode: Mode) -> Result<Self> {
        preflight_layout(raw, DocumentLayout::Quick)?;
        let value: &RawValue = serde_json::from_slice(raw).map_err(|e| json_err(e, "quick"))?;
        let mut f = Fields::new(
            value,
            &[
                "schema",
                "kind",
                "synthetic",
                "canonical_bytes",
                "integrity",
            ],
            "quick",
        )?;
        let schema: Text<128> = f.field("schema", "quick")?;
        let name: Text<128> = f.field("kind", "quick")?;
        let synthetic: bool = f.field("synthetic", "quick")?;
        let kind = EventKind::from_name(&name.0)?;
        ensure(
            schema.0 == mode.domain(&kind.domain()) && synthetic == mode.synthetic(),
            DocumentErrorCode::UnsupportedSchema,
            "quick.schema",
        )?;
        let canonical = bytes(
            f.get("canonical_bytes", "quick")?,
            QUICK_K_MAX,
            "quick.canonical_bytes",
            true,
        )?;
        let integrity: Integrity = f.field("integrity", "quick")?;
        ensure(
            integrity.encoding.0 == ENCODING_VERSION && integrity.hash.0 == "sha256",
            DocumentErrorCode::UnsupportedSchema,
            "quick.integrity",
        )?;
        ensure(
            sha256_hex(&canonical) == integrity.payload_sha256.0,
            DocumentErrorCode::DigestMismatch,
            "quick.integrity",
        )?;
        decode_event(kind, &canonical, mode)?;
        Ok(Self {
            kind,
            mode,
            canonical,
            file_sha256: sha256_hex(raw),
        })
    }
    pub(super) fn canonical(&self) -> Result<Vec<u8>> {
        let body = decode_event(self.kind, &self.canonical, self.mode)?;
        let mut e = record_encoder(&self.kind.domain(), self.mode)?;
        body.write(&mut e)?;
        check_transport_length(e.bytes.len(), true)?;
        Ok(e.bytes)
    }
    pub(super) fn json(&self) -> Result<Vec<u8>> {
        let canonical = self.canonical()?;
        let prefix = format!(
            "{{\"schema\":\"{}\",\"kind\":\"quick_{}\",\"synthetic\":{},\"canonical_bytes\":\"",
            self.mode.domain(&self.kind.domain()),
            self.kind.name(),
            self.mode.synthetic()
        );
        let suffix = format!(
            "\",\"integrity\":{{\"encoding\":\"{}\",\"hash\":\"sha256\",\"payload_sha256\":\"{}\"}}}}\n",
            ENCODING_VERSION,
            sha256_hex(&canonical)
        );
        let hex_length = canonical
            .len()
            .checked_mul(2)
            .ok_or_else(|| err(DocumentErrorCode::TooLarge, "quick.hex.length"))?;
        let length = bounded_add(
            bounded_add(prefix.len(), hex_length, A_MAX, "quick.json.length")?,
            suffix.len(),
            A_MAX,
            "quick.json.length",
        )?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "quick.json.allocate"))?;
        output.extend_from_slice(prefix.as_bytes());
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for b in canonical {
            output.push(HEX[(b >> 4) as usize]);
            output.push(HEX[(b & 15) as usize]);
        }
        output.extend_from_slice(suffix.as_bytes());
        Ok(output)
    }
    pub(super) fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub(super) fn argv(&self) -> Result<Vec<Vec<u8>>> {
        match decode_event(self.kind, &self.canonical, self.mode)? {
            EventBody::Preflight(p) => Ok(p.argv.0.iter().map(|b| b.0.to_vec()).collect()),
            _ => Err(err(DocumentErrorCode::CrossLinkMismatch, "quick.argv.kind")),
        }
    }
    #[cfg(test)]
    pub(super) fn parse_binary_oracle(kind: EventKind, canonical: &[u8]) -> Result<Self> {
        decode_event(kind, canonical, Mode::Oracle)?;
        let mut value = Self {
            kind,
            mode: Mode::Oracle,
            canonical: canonical.to_vec(),
            file_sha256: String::new(),
        };
        value.file_sha256 = sha256_hex(&value.json()?);
        Ok(value)
    }
    #[cfg(test)]
    pub(super) fn with_canonical_oracle(&self, canonical: &[u8]) -> Result<Self> {
        Self::parse_binary_oracle(self.kind, canonical)
    }
}
pub(super) fn clock_order(before: &Clock, after: &Clock) -> Result<()> {
    ensure(
        before.unix_ns <= after.unix_ns
            && before.elapsed_since_operation_start_ns <= after.elapsed_since_operation_start_ns,
        DocumentErrorCode::TimingMismatch,
        "quick.clock_order",
    )
}
fn link(link: &Link, event: &QuickEvent) -> Result<()> {
    same(
        link.file_sha256.0 == event.file_sha256
            && link.payload_sha256.0 == sha256_hex(&event.canonical),
        "quick.event.link",
    )
}
fn check_gate_hashes(g: &admission::Gates, mode: Mode, sidecars: &[Evidence<'_>]) -> Result<()> {
    same(sidecars.len() == 5, "quick.events.sidecar.count")?;
    validate_gates(g, mode)?;
    for (index, sidecar) in sidecars.iter().enumerate() {
        let gate = gates(g)[index];
        same(
            sidecar.gate == index
                && sidecar.mode == mode
                && sidecar.common.scope.0 == 1
                && gate.policy_revision.0 == sidecar.common.policy_revision.0
                && gate.evidence_sha256.0 == sha256_hex(sidecar.bytes),
            "quick.events.gates",
        )?;
    }
    Ok(())
}
/// Pure staged validation: no future start/end is invented.
pub(super) fn check_quick_preflight(
    preflight: &QuickEvent,
    sidecars: &[Evidence<'_>],
    captured_argv: &[Vec<u8>],
) -> Result<()> {
    let EventBody::Preflight(p) =
        decode_event(preflight.kind, &preflight.canonical, preflight.mode)?
    else {
        return Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "quick.events.order",
        ));
    };
    check_evidence_bundle(sidecars)?;
    let identity = QuickIdentity::decode(p.identity.0, preflight.mode)?;
    for sidecar in sidecars {
        let Branch::Quick(branch) = &sidecar.branch else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "quick.events.scope",
            ));
        };
        same(
            sidecar.mode == preflight.mode
                && branch.operational_identity.0 == p.identity.0
                && branch.operational_id == p.identity_sha256,
            "quick.events.sidecar.identity",
        )?;
        identity.check_common(&sidecar.common)?;
        let c = &sidecar.common;
        same(
            c.source_manifest_sha256 == p.source_manifest_sha256
                && c.source_revision == p.source_revision
                && c.approval_sha256 == p.approval_sha256
                && c.train_receipt_sha256 == p.train_receipt_sha256
                && c.executable_sha256 == p.executable_sha256,
            "quick.events.context",
        )?;
    }
    check_gate_hashes(&p.gates, preflight.mode, sidecars)?;
    same(
        p.argv.0.len() == captured_argv.len()
            && p.argv.0.iter().zip(captured_argv).all(|(a, b)| a.0 == b),
        "quick.argv.retained_bytes",
    )
}
/// Start checks only already-existing preflight/start claims, never a fake end.
pub(super) fn check_quick_start(
    preflight: &QuickEvent,
    start: &QuickEvent,
    sidecars: &[Evidence<'_>],
    captured_argv: &[Vec<u8>],
) -> Result<()> {
    check_quick_preflight(preflight, sidecars, captured_argv)?;
    same(preflight.mode == start.mode, "quick.events.domain")?;
    let EventBody::Preflight(p) =
        decode_event(preflight.kind, &preflight.canonical, preflight.mode)?
    else {
        unreachable!("checked preflight")
    };
    let EventBody::Start(s) = decode_event(start.kind, &start.canonical, start.mode)? else {
        return Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "quick.events.order",
        ));
    };
    same(
        s.identity_sha256 == p.identity_sha256,
        "quick.events.identity",
    )?;
    check_gate_hashes(&s.gates, preflight.mode, sidecars)?;
    link(&s.preflight, preflight)?;
    clock_order(&p.observed, &s.started)
}
/// Complete inert three-event comparison remains available. Live owners, actual
/// completion count, clocks and lease freshness are separate obligations.
pub(super) fn check_quick_events(
    preflight: &QuickEvent,
    start: &QuickEvent,
    end: &QuickEvent,
    sidecars: &[Evidence<'_>],
    captured_argv: &[Vec<u8>],
) -> Result<()> {
    check_quick_start(preflight, start, sidecars, captured_argv)?;
    same(preflight.mode == end.mode, "quick.events.domain")?;
    let EventBody::Start(s) = decode_event(start.kind, &start.canonical, start.mode)? else {
        unreachable!("checked start")
    };
    let EventBody::End(e) = decode_event(end.kind, &end.canonical, end.mode)? else {
        return Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "quick.events.order",
        ));
    };
    same(
        e.identity_sha256 == s.identity_sha256,
        "quick.events.identity",
    )?;
    check_gate_hashes(&e.gates, preflight.mode, sidecars)?;
    link(&e.preflight, preflight)?;
    link(&e.start, start)?;
    clock_order(&s.started, &e.ended)
}
#[cfg(test)]
mod tests {
    use super::super::tests::fixture;
    use super::*;
    #[test]
    fn retained_quick_staged_checks_require_no_future_end() {
        let preflight = QuickEvent::parse_json_oracle(&fixture("quick.preflight.json")).unwrap();
        let start = QuickEvent::parse_json_oracle(&fixture("quick.start.json")).unwrap();
        let bytes: Vec<_> = GATE_NAMES
            .iter()
            .map(|n| fixture(&format!("quick.{n}.evidence.bin")))
            .collect();
        let sidecars: Vec<_> = bytes
            .iter()
            .map(|b| Evidence::parse_oracle(b).unwrap())
            .collect();
        let argv = preflight.argv().unwrap();
        check_quick_preflight(&preflight, &sidecars, &argv).unwrap();
        check_quick_start(&preflight, &start, &sidecars, &argv).unwrap();
        assert!(check_quick_preflight(&start, &sidecars, &argv).is_err());
        assert!(check_quick_start(&preflight, &preflight, &sidecars, &argv).is_err());
        let mut wrong = argv;
        wrong[0].push(1);
        assert!(check_quick_preflight(&preflight, &sidecars, &wrong).is_err());
    }
    fn permutations(values: &mut [usize; 5], at: usize, out: &mut Vec<[usize; 5]>) {
        if at == 5 {
            out.push(*values);
            return;
        }
        for i in at..5 {
            values.swap(at, i);
            permutations(values, at + 1, out);
            values.swap(at, i);
        }
    }
    #[test]
    fn d7_evidence_all_gate_policy_and_hash_permutations() {
        let original = fixture("quick.preflight.bin");
        let EventBody::Preflight(original) =
            decode_event(EventKind::Preflight, &original, Mode::Oracle).unwrap()
        else {
            panic!("preflight")
        };
        let mut permutations_list = Vec::new();
        permutations(&mut [0, 1, 2, 3, 4], 0, &mut permutations_list);
        let source = gates(&original.gates);
        let sidecar_bytes: Vec<_> = GATE_NAMES
            .iter()
            .map(|name| fixture(&format!("quick.{name}.evidence.bin")))
            .collect();
        let sidecars: Vec<_> = sidecar_bytes
            .iter()
            .map(|b| Evidence::parse_oracle(b).unwrap())
            .collect();
        for indices in permutations_list
            .into_iter()
            .filter(|p| *p != [0, 1, 2, 3, 4])
        {
            for policy in [true, false] {
                let mut changed = original.clone();
                let target = [
                    &mut changed.gates.inputs,
                    &mut changed.gates.implementation,
                    &mut changed.gates.layout,
                    &mut changed.gates.resource,
                    &mut changed.gates.user,
                ];
                for (i, gate) in target.into_iter().enumerate() {
                    if policy {
                        gate.policy_revision = source[indices[i]].policy_revision.clone();
                    } else {
                        gate.evidence_sha256 = source[indices[i]].evidence_sha256.clone();
                    }
                }
                let mut encoder = record_encoder("quick-preflight", Mode::Oracle).unwrap();
                changed.write(&mut encoder).unwrap();
                if policy {
                    match decode_event(EventKind::Preflight, &encoder.bytes, Mode::Oracle) {
                        Err(e) => assert_eq!(e.code(), DocumentErrorCode::UnsupportedSchema),
                        Ok(_) => panic!("policy permutation accepted"),
                    };
                } else {
                    let parsed =
                        decode_event(EventKind::Preflight, &encoder.bytes, Mode::Oracle).unwrap();
                    let error =
                        check_gate_hashes(parsed.gates(), Mode::Oracle, &sidecars).unwrap_err();
                    assert_eq!(error.code(), DocumentErrorCode::CrossLinkMismatch);
                }
            }
        }
    }
}
