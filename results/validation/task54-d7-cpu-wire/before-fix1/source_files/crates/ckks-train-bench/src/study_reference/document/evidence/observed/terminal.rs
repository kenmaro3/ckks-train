//! Supplied terminal-chain consistency. No custody, durable visibility or live proof.
use super::*;
pub(in crate::study_reference::document::evidence) struct QuickTerminal<'a> {
    pub(in crate::study_reference::document::evidence) result: &'a [u8],
    pub(in crate::study_reference::document::evidence) result_json: &'a [u8],
    pub(in crate::study_reference::document::evidence) result_export: &'a [u8],
    pub(in crate::study_reference::document::evidence) preflight: &'a [u8],
    pub(in crate::study_reference::document::evidence) start: &'a [u8],
    pub(in crate::study_reference::document::evidence) end: &'a [u8],
    pub(in crate::study_reference::document::evidence) supervisor_end: &'a [u8],
}
impl QuickTerminal<'_> {
    pub(in crate::study_reference::document::evidence) fn check(
        &self,
        bundle: &CurrentEvidenceBundle<'_, '_>,
    ) -> Result<()> {
        bundle.check()?;
        same(bundle.resource.common.scope.0 == 1, "terminal.quick.scope")?;
        let mode = bundle.resource.mode;
        check_three_outputs(self.result, self.result_json, self.result_export, mode)?;
        check_quick_v2_chain(self.preflight, self.start, self.end, bundle)?;
        let record = ObservedRecord::decode(RecordKind::Result, self.result, mode)?;
        let RecordBody::Result(r) = record.body else {
            unreachable!()
        };
        r.check_bundle(bundle)?;
        r.start.check_bytes(self.start)?;
        r.admission_or_preflight.check_bytes(self.preflight)?;
        let end = decode_quick_v2_json(self.end, mode)?;
        let end = QuickEventV2::decode(EventKind::End, &end, mode)?;
        let QuickV2Body::End(end) = end.body else {
            unreachable!()
        };
        same(
            end.original.outcome.0 == 0
                && r.status.0 == 0
                && end.original.completed_updates == r.completed_updates,
            "terminal.quick.complete",
        )?;
        check_result_files(
            [&end.result, &end.result_json, &end.result_export],
            [self.result, self.result_json, self.result_export],
            &r,
            mode,
        )?;
        clock_order(&r.ended, &end.original.ended)?;
        check_supervisor(
            self.supervisor_end,
            self.result,
            self.start,
            self.end,
            bundle,
            &end.original.ended,
        )
    }
}
impl CpuResult<'_> {
    pub(in crate::study_reference::document::evidence) fn check_bundle(
        &self,
        bundle: &CurrentEvidenceBundle<'_, '_>,
    ) -> Result<()> {
        bundle.check()?;
        let e = &bundle.resource;
        let c = &e.common;
        same(
            self.scope == c.scope
                && self.case.0 == e.branch.registration()
                && self.execution.0 == c.execution.0
                && self.attempt == e.branch.attempt()
                && self.source_manifest == c.source_manifest_sha256
                && self.source_revision == c.source_revision
                && self.executable == c.executable_sha256
                && self.contract == c.contract_sha256
                && self.receipt == c.selected_receipt_sha256
                && self.materialized == c.materialized_sha256,
            "result.bundle.provenance",
        )?;
        for (i, f) in self.evidence.iter().enumerate() {
            f.check_bytes(if i == 3 {
                e.bytes
            } else {
                bundle.non_resource[if i == 4 { 3 } else { i }].bytes
            })?;
        }
        let Tail::Inputs(inputs) = &bundle.non_resource[0].tail else {
            unreachable!()
        };
        same(
            self.consumed_plan == inputs.consumed_plan_sha256,
            "result.bundle.consumed",
        )
    }
}
fn check_result_files(
    links: [&Option<FileRef<'_>>; 3],
    outputs: [&[u8]; 3],
    r: &CpuResult<'_>,
    mode: Mode,
) -> Result<()> {
    let [bin, json, export] = links;
    let [b, j, x] = outputs;
    let missing = || {
        err(
            DocumentErrorCode::CrossLinkMismatch,
            "terminal.result.links",
        )
    };
    bin.as_ref().ok_or_else(missing)?.check_bytes(b)?;
    json.as_ref().ok_or_else(missing)?.check_bytes(j)?;
    let export = export.as_ref().ok_or_else(missing)?;
    export.check_bytes(x)?;
    let id = export_id(r.scope.0, r.start.sha256.0, mode)?;
    same(
        export.root.0 == 5 && export.path.0 == format!("study-cpu-{id}.json"),
        "terminal.export.identity",
    )
}
fn check_supervisor(
    raw: &[u8],
    result: &[u8],
    start: &[u8],
    end: &[u8],
    bundle: &CurrentEvidenceBundle<'_, '_>,
    ended: &Clock,
) -> Result<()> {
    let supervisor = ObservedRecord::decode(RecordKind::SupervisorEnd, raw, bundle.resource.mode)?;
    let RecordBody::SupervisorEnd(s) = supervisor.body else {
        unreachable!()
    };
    same(
        s.reason.0 == 0 && s.custody_resolved.0 == 1,
        "terminal.supervisor.closed",
    )?;
    let required = |f: Option<FileRef<'_>>, b: &[u8]| {
        f.ok_or_else(|| {
            err(
                DocumentErrorCode::CrossLinkMismatch,
                "terminal.supervisor.link",
            )
        })?
        .check_bytes(b)
    };
    required(s.result, result)?;
    required(s.start, start)?;
    required(s.worker_end, end)?;
    required(s.claim, &bundle.claim.canonical()?)?;
    let RecordBody::Claim(claim) = &bundle.claim.body else {
        unreachable!()
    };
    let mut a = Encoder::new(DocumentLayout::Start);
    s.pending.write(&mut a)?;
    let mut b = Encoder::new(DocumentLayout::Start);
    claim.pending.write(&mut b)?;
    same(a.bytes == b.bytes, "terminal.supervisor.pending")?;
    clock_order(ended, &s.ended)
}
