//! Explicit v2 Quick wire. v1 identity is reused; v1 event links never promoted.
use super::*;
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct EndV2<'a> {
    pub(in crate::study_reference::document::evidence) original: quick::End<'a>,
    pub(in crate::study_reference::document::evidence) result: Option<FileRef<'a>>,
    pub(in crate::study_reference::document::evidence) result_json: Option<FileRef<'a>>,
    pub(in crate::study_reference::document::evidence) result_export: Option<FileRef<'a>>,
    pub(in crate::study_reference::document::evidence) result_publication_ns: u64,
}
impl<'a> Wire<'a> for EndV2<'a> {
    const MIN: usize = quick::End::MIN + 11;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            original: quick::End::read(c)?,
            result: Option::read(c)?,
            result_json: Option::read(c)?,
            result_export: Option::read(c)?,
            result_publication_ns: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.original.write(e)?;
        self.result.write(e)?;
        self.result_json.write(e)?;
        self.result_export.write(e)?;
        self.result_publication_ns.write(e)
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) enum QuickV2Body<'a> {
    Preflight(quick::Preflight<'a>),
    Start(quick::Start<'a>),
    End(EndV2<'a>),
}
impl QuickV2Body<'_> {
    fn gates(&self) -> &admission::Gates {
        match self {
            Self::Preflight(v) => &v.gates,
            Self::Start(v) => &v.gates,
            Self::End(v) => &v.original.gates,
        }
    }
    fn identity(&self) -> H<'_> {
        match self {
            Self::Preflight(v) => v.identity_sha256,
            Self::Start(v) => v.identity_sha256,
            Self::End(v) => v.original.identity_sha256,
        }
    }
}
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct QuickEventV2<'a> {
    pub(in crate::study_reference::document::evidence) kind: EventKind,
    pub(in crate::study_reference::document::evidence) mode: Mode,
    pub(in crate::study_reference::document::evidence) body: QuickV2Body<'a>,
}
impl<'a> QuickEventV2<'a> {
    pub(in crate::study_reference::document::evidence) fn decode(
        kind: EventKind,
        b: &'a [u8],
        mode: Mode,
    ) -> Result<Self> {
        let mut c = cursor(b, &format!("quick-{}", kind.name()), 2, mode, QUICK_K_MAX)?;
        let body = match kind {
            EventKind::Preflight => QuickV2Body::Preflight(quick::Preflight::read(&mut c)?),
            EventKind::Start => QuickV2Body::Start(quick::Start::read(&mut c)?),
            EventKind::End => QuickV2Body::End(EndV2::read(&mut c)?),
        };
        c.finish()?;
        let v = Self { kind, mode, body };
        v.validate()?;
        Ok(v)
    }
    fn validate(&self) -> Result<()> {
        validate_current_gates(self.body.gates(), self.mode)?;
        match &self.body {
            QuickV2Body::Preflight(p) => {
                digest(p.identity.0, p.identity_sha256, "quickv2.identity")?;
                let q = QuickIdentity::decode(p.identity.0, self.mode)?;
                let x = admission::parse_execution_binary(q.body.execution.0)?;
                same(
                    p.source_manifest_sha256.0 == x.fields().manifest
                        && p.source_revision.0 == x.fields().revision,
                    "quickv2.source",
                )?;
            }
            QuickV2Body::Start(_) => {}
            QuickV2Body::End(v) => {
                let e = &v.original;
                same(e.completed_updates <= 2, "quickv2.updates")?;
                same(
                    if e.outcome.0 == 0 {
                        e.completed_updates == 2
                            && e.failure.is_none()
                            && v.result.is_some()
                            && v.result_json.is_some()
                            && v.result_export.is_some()
                    } else {
                        e.failure.is_some()
                    },
                    "quickv2.end.outcome",
                )?;
                for f in [&v.result, &v.result_json, &v.result_export]
                    .into_iter()
                    .flatten()
                {
                    f.validate()?;
                }
            }
        }
        Ok(())
    }
    pub(in crate::study_reference::document::evidence) fn canonical(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = encoder(&format!("quick-{}", self.kind.name()), 2, self.mode)?;
        match &self.body {
            QuickV2Body::Preflight(v) => v.write(&mut e)?,
            QuickV2Body::Start(v) => v.write(&mut e)?,
            QuickV2Body::End(v) => v.write(&mut e)?,
        }
        ensure(
            e.bytes.len() <= QUICK_K_MAX,
            DocumentErrorCode::TooLarge,
            "quickv2.length",
        )?;
        Ok(e.bytes)
    }
    pub(in crate::study_reference::document::evidence) fn json(&self) -> Result<Vec<u8>> {
        let canonical = self.canonical()?;
        let value = serde_json::json!({"schema":domain(self.mode,&format!("quick-{}",self.kind.name()),2),"kind":format!("quick_{}",self.kind.name()),"synthetic":self.mode.synthetic(),"canonical_bytes":result_hex(&canonical),"integrity":{"encoding":ENCODING_VERSION,"hash":"sha256","payload_sha256":sha256_hex(&canonical)}});
        let raw = serde_json::to_vec(&value)
            .map_err(|_| err(DocumentErrorCode::BadJson, "quickv2.json"))?;
        ensure(
            raw.len() <= A_MAX,
            DocumentErrorCode::TooLarge,
            "quickv2.json",
        )?;
        Ok(raw)
    }
}
fn result_hex(b: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for b in b {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 15) as usize] as char);
    }
    s
}
/// Owns only bounded transport. Explicit schema dispatch rejects every v1 event.
pub(in crate::study_reference::document::evidence) fn decode_quick_v2_json(
    raw: &[u8],
    mode: Mode,
) -> Result<Vec<u8>> {
    preflight_layout(raw, DocumentLayout::Quick)?;
    let r: &RawValue = serde_json::from_slice(raw).map_err(|e| json_err(e, "quickv2"))?;
    let mut f = Fields::new(
        r,
        &[
            "schema",
            "kind",
            "synthetic",
            "canonical_bytes",
            "integrity",
        ],
        "quickv2",
    )?;
    let name = f.field::<Text<128>>("kind", "quickv2")?;
    let kind = match name.0.as_str() {
        "quick_preflight" => EventKind::Preflight,
        "quick_start" => EventKind::Start,
        "quick_end" => EventKind::End,
        _ => return Err(err(DocumentErrorCode::UnsupportedTag, "quickv2.kind")),
    };
    same(
        f.field::<Text<128>>("schema", "quickv2")?.0
            == domain(mode, &format!("quick-{}", kind.name()), 2)
            && f.field::<bool>("synthetic", "quickv2")? == mode.synthetic(),
        "quickv2.schema",
    )?;
    let b = bytes(
        f.get("canonical_bytes", "quickv2")?,
        QUICK_K_MAX,
        "quickv2.canonical_bytes",
        true,
    )?;
    let integrity = f.field::<Integrity>("integrity", "quickv2")?;
    same(
        integrity.encoding.0 == ENCODING_VERSION
            && integrity.hash.0 == "sha256"
            && integrity.payload_sha256.0 == sha256_hex(&b),
        "quickv2.integrity",
    )?;
    QuickEventV2::decode(kind, &b, mode)?;
    Ok(b)
}
fn quick_link(link: &Link, raw: &[u8], canonical: &[u8]) -> Result<()> {
    same(
        link.file_sha256.0 == sha256_hex(raw) && link.payload_sha256.0 == sha256_hex(canonical),
        "quickv2.link",
    )
}
pub(in crate::study_reference::document::evidence) fn check_quick_v2_chain(
    preflight_raw: &[u8],
    start_raw: &[u8],
    end_raw: &[u8],
    bundle: &CurrentEvidenceBundle<'_, '_>,
) -> Result<()> {
    let mode = bundle.resource.mode;
    let pb = decode_quick_v2_json(preflight_raw, mode)?;
    let sb = decode_quick_v2_json(start_raw, mode)?;
    let eb = decode_quick_v2_json(end_raw, mode)?;
    let p = QuickEventV2::decode(EventKind::Preflight, &pb, mode)?;
    let s = QuickEventV2::decode(EventKind::Start, &sb, mode)?;
    let e = QuickEventV2::decode(EventKind::End, &eb, mode)?;
    let (QuickV2Body::Preflight(p), QuickV2Body::Start(s), QuickV2Body::End(e)) =
        (p.body, s.body, e.body)
    else {
        unreachable!()
    };
    for g in [&p.gates, &s.gates, &e.original.gates] {
        bundle.check_gate_hashes(g)?;
    }
    same(
        p.identity_sha256 == s.identity_sha256 && s.identity_sha256 == e.original.identity_sha256,
        "quickv2.chain.identity",
    )?;
    let Branch::Quick(selected) = &bundle.resource.branch else {
        return Err(err(
            DocumentErrorCode::CrossLinkMismatch,
            "quickv2.chain.scope",
        ));
    };
    same(
        p.identity.0 == selected.operational_identity.0
            && p.identity_sha256 == selected.operational_id,
        "quickv2.chain.selected_identity",
    )?;
    let common = &bundle.resource.common;
    same(
        p.source_manifest_sha256 == common.source_manifest_sha256
            && p.source_revision == common.source_revision
            && p.approval_sha256 == common.approval_sha256
            && p.train_receipt_sha256 == common.train_receipt_sha256
            && p.executable_sha256 == common.executable_sha256,
        "quickv2.chain.context",
    )?;
    let q = QuickIdentity::decode(p.identity.0, mode)?;
    q.check_common(common)?;
    same(
        q.body.registration.0 == bundle.resource.branch.registration(),
        "quickv2.chain.registration",
    )?;
    quick_link(&s.preflight, preflight_raw, &pb)?;
    quick_link(&e.original.preflight, preflight_raw, &pb)?;
    quick_link(&e.original.start, start_raw, &sb)?;
    clock_order(&p.observed, &s.started)?;
    clock_order(&s.started, &e.original.ended)
}
