//! Inert parity with four unchanged tails; no proof of live owner/source custody.
use super::*;
#[derive(Debug, Clone)]
pub(in crate::study_reference::document::evidence) struct OperationalResourceEvidence<'a> {
    pub(in crate::study_reference::document::evidence) mode: Mode,
    pub(in crate::study_reference::document::evidence) common: Common<'a>,
    pub(in crate::study_reference::document::evidence) branch: Branch<'a>,
    pub(in crate::study_reference::document::evidence) resource: OperationalResource<'a>,
    pub(in crate::study_reference::document::evidence) bytes: &'a [u8],
}
impl<'a> OperationalResourceEvidence<'a> {
    pub(in crate::study_reference::document::evidence) fn decode(
        b: &'a [u8],
        mode: Mode,
    ) -> Result<Self> {
        let mut c = cursor(b, "resource-observed-evidence", 1, mode, A_MAX)?;
        let common = Common::read(&mut c)?;
        let branch = if common.scope.0 == 0 {
            Branch::Full(FullBody::read(&mut c)?)
        } else {
            Branch::Quick(QuickBody::read(&mut c)?)
        };
        let resource = OperationalResource::read(&mut c)?;
        c.finish()?;
        let v = Self {
            mode,
            common,
            branch,
            resource,
            bytes: b,
        };
        v.validate()?;
        Ok(v)
    }
    fn validate(&self) -> Result<()> {
        same(
            self.common.policy_revision.0 == current_policy(self.mode, 3),
            "observed.evidence.policy",
        )?;
        let (execution, case) = validate_common_branch(&self.common, &self.branch, self.mode)?;
        let x = execution.fields();
        let r = &self.resource;
        r.applicability.matches_common(&self.common, &x, &case)?;
        same(
            case.backend as u8 == 0
                && r.physical.0 == x.physical
                && r.observation.physical == r.physical
                && r.supervisor.physical_host == r.physical
                && r.sizing_revision.0 == SIZING,
            "observed.evidence.binding",
        )?;
        r.supervisor.validate()?;
        r.demand.validate()?;
        for f in [
            &r.policy,
            &r.native_validation,
            &r.claim,
            &r.observation.host_raw,
            &r.observation.disk_raw,
        ] {
            f.validate()?;
        }
        for c in &r.observation.cgroups {
            same(
                !c.path.0.is_empty() && !c.path.0.contains('\0'),
                "observed.cgroup.path",
            )?;
            c.limit_raw.validate()?;
            c.current_raw.validate()?;
        }
        same(
            r.requested.live_temporary_bytes <= r.requested.peak_temporary_bytes
                && add(
                    add(r.requested.plan_bytes, r.requested.retained_bytes)?,
                    r.requested.live_temporary_bytes,
                )? <= r.requested.max_bytes,
            "observed.request.accounting",
        )
    }
    pub(in crate::study_reference::document::evidence) fn canonical(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let mut e = encoder("resource-observed-evidence", 1, self.mode)?;
        self.common.write(&mut e)?;
        self.branch.write(&mut e)?;
        self.resource.write(&mut e)?;
        Ok(e.bytes)
    }
}
pub(in crate::study_reference::document::evidence) fn current_policy(
    mode: Mode,
    index: usize,
) -> String {
    if index == 3 {
        format!("{}{}", if mode.synthetic() { "test-" } else { "" }, POLICY)
    } else {
        mode.policy(index)
    }
}
pub(in crate::study_reference::document::evidence) fn validate_current_gates(
    g: &admission::Gates,
    mode: Mode,
) -> Result<()> {
    for (i, gate) in quick::gates(g).iter().enumerate() {
        same(
            gate.policy_revision.0 == current_policy(mode, i),
            "current.policy.tuple",
        )?;
    }
    Ok(())
}
pub(in crate::study_reference::document::evidence) struct CurrentEvidenceBundle<'a, 'b> {
    pub(in crate::study_reference::document::evidence) non_resource: [&'b Evidence<'a>; 4],
    pub(in crate::study_reference::document::evidence) resource:
        &'b OperationalResourceEvidence<'a>,
    pub(in crate::study_reference::document::evidence) claim: &'b ObservedRecord<'a>,
}
impl CurrentEvidenceBundle<'_, '_> {
    pub(in crate::study_reference::document::evidence) fn check(&self) -> Result<()> {
        let first = self.non_resource[0];
        let mut original = Encoder::new(DocumentLayout::Start);
        first.common.write(&mut original)?;
        let mut branch = Encoder::new(DocumentLayout::Start);
        first.branch.write(&mut branch)?;
        for (i, r) in [0, 1, 2, 4].into_iter().zip(self.non_resource) {
            same(r.gate == i && r.mode == first.mode, "current.bundle.kind")?;
            r.validate()?;
            check_common(
                &original.bytes,
                &branch.bytes,
                &first.common,
                &r.common,
                &r.branch,
            )?;
        }
        same(
            self.resource.mode == first.mode && self.claim.mode == first.mode,
            "current.bundle.mode",
        )?;
        self.resource.validate()?;
        self.claim.validate()?;
        check_common(
            &original.bytes,
            &branch.bytes,
            &first.common,
            &self.resource.common,
            &self.resource.branch,
        )?;
        let (Tail::Inputs(i), Tail::Layout(l), Tail::User(u)) = (
            &self.non_resource[0].tail,
            &self.non_resource[2].tail,
            &self.non_resource[3].tail,
        ) else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "current.bundle.tail",
            ));
        };
        same(
            i.consumed_plan.0 == l.consumed_plan.0,
            "current.bundle.consumed",
        )?;
        let request = Auxiliary::decode(AuxKind::Request, l.request.0, first.mode)?;
        let r = &self.resource.resource;
        same(
            r.requested.max_bytes == request.request()?.limits[0]
                && r.requested.max_records == request.request()?.limits[1],
            "current.bundle.requested",
        )?;
        let scope = Auxiliary::decode(AuxKind::UserScope, u.authority_bytes.0, first.mode)?;
        let AuxiliaryBody::UserScope(scope) = scope.body else {
            unreachable!()
        };
        let selected = scope
            .scopes
            .get(
                usize::try_from(u.selected_scope_index)
                    .map_err(|_| err(DocumentErrorCode::OutOfRange, "current.scope.index"))?,
            )
            .ok_or_else(|| err(DocumentErrorCode::OutOfRange, "current.scope.index"))?;
        same(selected.overlap.0 == 0, "current.bundle.serial_overlap")?;
        let RecordBody::Claim(claim) = &self.claim.body else {
            return Err(err(
                DocumentErrorCode::CrossLinkMismatch,
                "current.bundle.claim",
            ));
        };
        r.claim.check_bytes(&self.claim.canonical()?)?;
        same(
            claim.approval == first.common.approval_sha256
                && claim.source_manifest == first.common.source_manifest_sha256
                && claim.source_revision == first.common.source_revision,
            "current.claim.source",
        )?;
        let mut a = Encoder::new(DocumentLayout::Start);
        claim.supervisor.write(&mut a)?;
        let mut b = Encoder::new(DocumentLayout::Start);
        r.supervisor.write(&mut b)?;
        same(a.bytes == b.bytes, "current.claim.supervisor")?;
        let work = RequestedWork::decode(claim.work.0, first.mode)?;
        let mut raw = Encoder::new(DocumentLayout::Start);
        request.request()?.write(&mut raw)?;
        work.bind_train(first.branch.registration(), TRAIN_RECIPE, &raw.bytes, None)?;
        same(
            compiled_scope(&work)? == first.common.scope.0,
            "current.claim.consumption",
        )
    }
    pub(in crate::study_reference::document::evidence) fn check_gate_hashes(
        &self,
        g: &admission::Gates,
    ) -> Result<()> {
        self.check()?;
        validate_current_gates(g, self.resource.mode)?;
        for (i, gate) in quick::gates(g).iter().enumerate() {
            let bytes = if i == 3 {
                self.resource.bytes
            } else {
                self.non_resource[if i == 4 { 3 } else { i }].bytes
            };
            same(
                gate.evidence_sha256.0 == sha256_hex(bytes),
                "current.gate.hash",
            )?;
        }
        Ok(())
    }
}
fn compiled_scope(w: &RequestedWork<'_>) -> Result<u8> {
    Ok(w.train_request()?[0])
}
fn check_common(
    original: &[u8],
    branch: &[u8],
    first: &Common<'_>,
    common: &Common<'_>,
    body: &Branch<'_>,
) -> Result<()> {
    let mut c = Encoder::new(DocumentLayout::Start);
    common.write(&mut c)?;
    same(
        c.bytes[8 + common.policy_revision.0.len()..]
            == original[8 + first.policy_revision.0.len()..],
        "current.bundle.common",
    )?;
    let mut b = Encoder::new(DocumentLayout::Start);
    body.write(&mut b)?;
    same(b.bytes == branch, "current.bundle.branch")
}
