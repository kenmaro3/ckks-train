//! Closed five typed tails. Descriptive file identities never select open paths.
use super::super::execution_binding::ExecutionFields;
use super::*;
use crate::study_profile::{CaseBackend, CaseMode};
wire_record!(FileRef {root:Tag<5>,path:Str<'a,4096>,length:u64,sha256:H<'a>});
impl FileRef<'_> {
    pub(in crate::study_reference::document::evidence) fn validate(&self) -> Result<()> {
        let p = self.path.0;
        same(
            !p.is_empty()
                && !p.starts_with('/')
                && !p.contains(['\0', '\\'])
                && p.split('/').all(|s| !s.is_empty() && s != "." && s != ".."),
            "file.path",
        )
    }
    /// Same exact supplied file bytes only. Custody/origin/applicability remain missing.
    pub(super) fn check_bytes(&self, b: &[u8]) -> Result<()> {
        self.validate()?;
        same(
            self.length == b.len() as u64 && self.sha256.0 == sha256_hex(b),
            "file.retained_bytes",
        )
    }
}
fn file_list(files: &[FileRef<'_>], required: bool) -> Result<()> {
    same(!required || !files.is_empty(), "files.required")?;
    let mut prior = None;
    for f in files {
        f.validate()?;
        let key = (f.root.0, f.path.0.as_bytes());
        same(prior.is_none_or(|p| p < key), "files.order")?;
        prior = Some(key);
    }
    Ok(())
}
wire_record!(IdentityRef {schema:Str<'a,128>,encoding:Str<'a,128>,canonical_length:u64,sha256:H<'a>});
impl IdentityRef<'_> {
    fn validate(&self, receipt: bool) -> Result<()> {
        ensure(
            self.schema.0
                == if receipt {
                    RECEIPT_VERSION
                } else {
                    CONTRACT_VERSION
                }
                && self.encoding.0 == ENCODING_VERSION,
            DocumentErrorCode::UnsupportedSchema,
            "input.identity.schema",
        )?;
        ensure(
            self.canonical_length > 0
                && self.canonical_length <= if receipt { R_MAX } else { C_MAX } as u64,
            DocumentErrorCode::TooLarge,
            "input.identity.length",
        )
    }
    pub(super) fn equals(&self, other: &Self) -> Result<()> {
        same(
            self.schema == other.schema
                && self.encoding == other.encoding
                && self.canonical_length == other.canonical_length
                && self.sha256 == other.sha256,
            "input.independent_receipt",
        )
    }
    pub(super) fn check_bytes(&self, bytes: &[u8], receipt: bool) -> Result<()> {
        self.validate(receipt)?;
        same(
            self.canonical_length == bytes.len() as u64 && self.sha256.0 == sha256_hex(bytes),
            "input.retained_identity",
        )
    }
}
#[derive(Debug, Clone)]
pub(super) struct Inputs<'a> {
    pub(super) source_files: Vec<FileRef<'a>>,
    pub(super) contract: IdentityRef<'a>,
    pub(super) selected_receipt: IdentityRef<'a>,
    pub(super) materialized_sha256: H<'a>,
    pub(super) consumed_plan: Bytes<'a>,
    pub(super) consumed_plan_sha256: H<'a>,
    pub(super) independent: Option<(IdentityRef<'a>, H<'a>)>,
}
impl<'a> Inputs<'a> {
    pub(super) fn read_branch(c: &mut Cursor<'a>, scope: u8) -> Result<Self> {
        Ok(Self {
            source_files: Vec::read(c)?,
            contract: IdentityRef::read(c)?,
            selected_receipt: IdentityRef::read(c)?,
            materialized_sha256: H::read(c)?,
            consumed_plan: Bytes::read(c)?,
            consumed_plan_sha256: H::read(c)?,
            independent: if scope == 0 {
                Some((IdentityRef::read(c)?, H::read(c)?))
            } else {
                None
            },
        })
    }
    pub(super) fn write(&self, e: &mut Encoder) -> Result<()> {
        self.source_files.write(e)?;
        self.contract.write(e)?;
        self.selected_receipt.write(e)?;
        self.materialized_sha256.write(e)?;
        self.consumed_plan.write(e)?;
        self.consumed_plan_sha256.write(e)?;
        if let Some((receipt, materialized)) = &self.independent {
            receipt.write(e)?;
            materialized.write(e)?;
        }
        Ok(())
    }
    pub(super) fn validate(&self, e: &Evidence<'_>, _: &admission::CaseIdentity) -> Result<()> {
        file_list(&self.source_files, true)?;
        self.contract.validate(false)?;
        self.selected_receipt.validate(true)?;
        same(
            self.contract.sha256 == e.common.contract_sha256
                && self.selected_receipt.sha256 == e.common.selected_receipt_sha256
                && self.materialized_sha256 == e.common.materialized_sha256,
            "inputs.common",
        )?;
        e.consumed(self.consumed_plan.0, self.consumed_plan_sha256)?;
        match (&e.branch, &self.independent) {
            (Branch::Full(_), Some((receipt, materialized))) => {
                receipt.validate(true)?;
                receipt.equals(&self.selected_receipt)?;
                same(
                    *materialized == self.materialized_sha256,
                    "inputs.independent_materialized",
                )
            }
            (Branch::Quick(_), None) => Ok(()),
            _ => Err(err(DocumentErrorCode::CrossLinkMismatch, "inputs.branch")),
        }
    }
}
wire_record!(Applicability {implementation_manifest:H<'a>,implementation_revision:Git<'a>,parameters:H<'a>,bootstrap:Option<H<'a>>,keys:H<'a>,security:H<'a>,layout:H<'a>,trainer:H<'a>,runtime:H<'a>,implementation:Tag<4>,backend:Tag<2>,mode:Tag<1>,consumption:Tag<1>});
impl Applicability<'_> {
    fn matches(
        &self,
        e: &Evidence<'_>,
        x: &ExecutionFields<'_>,
        case: &admission::CaseIdentity,
    ) -> Result<()> {
        self.matches_common(&e.common, x, case)
    }
    pub(in crate::study_reference::document::evidence) fn matches_common(
        &self,
        common: &Common<'_>,
        x: &ExecutionFields<'_>,
        case: &admission::CaseIdentity,
    ) -> Result<()> {
        same(
            self.implementation_manifest.0 == x.manifest
                && self.implementation_revision.0 == x.revision
                && self.parameters.0 == x.parameter
                && self.bootstrap.map(|h| h.0) == x.bootstrap
                && self.keys.0 == x.keys
                && self.security.0 == x.security
                && self.layout.0 == x.layout
                && self.trainer.0 == x.trainer
                && self.runtime.0 == x.runtime
                && self.implementation.0 == x.implementation
                && self.backend.0 == case.backend as u8
                && self.mode.0 == case.mode as u8
                && self.consumption == common.scope,
            "applicability.exact",
        )
    }
    fn exact(&self, other: &Self) -> Result<()> {
        let mut a = Encoder::new(DocumentLayout::Start);
        self.write(&mut a)?;
        let mut b = Encoder::new(DocumentLayout::Start);
        other.write(&mut b)?;
        same(a.bytes == b.bytes, "applicability.correctness")
    }
}
wire_record!(Correctness {kind:Tag<1>,report:FileRef<'a>,command:FileRef<'a>,source_manifest:FileRef<'a>,source_revision:Git<'a>,criterion_policy:FileRef<'a>,applicability:Applicability<'a>});
wire_record!(SecurityInstance {role:Tag<1>,problem:FileRef<'a>,bdgl16_raw:FileRef<'a>,matzov_raw:FileRef<'a>});
#[derive(Debug, Clone)]
pub(super) struct SecurityEvidence<'a> {
    pub(super) policy: FileRef<'a>,
    pub(super) parameter_export: FileRef<'a>,
    pub(super) estimator_revision: Git<'a>,
    pub(super) estimator_sources: Vec<FileRef<'a>>,
    pub(super) environment: FileRef<'a>,
    pub(super) command: FileRef<'a>,
    pub(super) model_config: FileRef<'a>,
    pub(super) instances: Vec<SecurityInstance<'a>>,
}
impl<'a> Wire<'a> for SecurityEvidence<'a> {
    const MIN: usize = 5 * FileRef::MIN + Git::MIN + 16;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            policy: FileRef::read(c)?,
            parameter_export: FileRef::read(c)?,
            estimator_revision: Git::read(c)?,
            estimator_sources: wire::read_vec(c, 65_536)?,
            environment: FileRef::read(c)?,
            command: FileRef::read(c)?,
            model_config: FileRef::read(c)?,
            instances: wire::read_vec(c, 2)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.estimator_sources.len() <= 65_536 && self.instances.len() <= 2,
            DocumentErrorCode::TooLarge,
            "security.list",
        )?;
        self.policy.write(e)?;
        self.parameter_export.write(e)?;
        self.estimator_revision.write(e)?;
        self.estimator_sources.write(e)?;
        self.environment.write(e)?;
        self.command.write(e)?;
        self.model_config.write(e)?;
        self.instances.write(e)
    }
}
#[derive(Debug, Clone)]
pub(super) struct Implementation<'a> {
    pub(super) applicability: Applicability<'a>,
    pub(super) correctness: Vec<Correctness<'a>>,
    pub(super) security: SecurityEvidence<'a>,
    pub(super) construction_obligations: Vec<Tag<4>>,
}
impl<'a> Wire<'a> for Implementation<'a> {
    const MIN: usize = Applicability::MIN + 16 + SecurityEvidence::MIN;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            applicability: Applicability::read(c)?,
            correctness: wire::read_vec(c, 2)?,
            security: SecurityEvidence::read(c)?,
            construction_obligations: wire::read_vec(c, 5)?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        ensure(
            self.correctness.len() <= 2 && self.construction_obligations.len() <= 5,
            DocumentErrorCode::TooLarge,
            "implementation.list",
        )?;
        self.applicability.write(e)?;
        self.correctness.write(e)?;
        self.security.write(e)?;
        self.construction_obligations.write(e)
    }
}
impl Implementation<'_> {
    pub(super) fn validate(
        &self,
        e: &Evidence<'_>,
        x: &ExecutionFields<'_>,
        case: &admission::CaseIdentity,
    ) -> Result<()> {
        self.applicability.matches(e, x, case)?;
        let count = if case.mode == CaseMode::Interactive {
            1
        } else {
            2
        };
        same(
            self.correctness.len() == count && self.security.instances.len() == count,
            "implementation.coverage",
        )?;
        for (i, c) in self.correctness.iter().enumerate() {
            same(
                c.kind.0 == i as u8
                    && c.source_manifest.sha256.0 == x.manifest
                    && c.source_revision.0 == x.revision,
                "correctness.source_kind",
            )?;
            for f in [
                &c.report,
                &c.command,
                &c.source_manifest,
                &c.criterion_policy,
            ] {
                f.validate()?;
            }
            self.applicability.exact(&c.applicability)?;
        }
        let s = &self.security;
        file_list(&s.estimator_sources, true)?;
        for f in [
            &s.policy,
            &s.parameter_export,
            &s.environment,
            &s.command,
            &s.model_config,
        ] {
            f.validate()?;
        }
        for (i, instance) in s.instances.iter().enumerate() {
            same(instance.role.0 == i as u8, "security.instance.role")?;
            for f in [
                &instance.problem,
                &instance.bdgl16_raw,
                &instance.matzov_raw,
            ] {
                f.validate()?;
            }
        }
        same(
            self.construction_obligations.len() == count + 3
                && self
                    .construction_obligations
                    .iter()
                    .enumerate()
                    .all(|(i, t)| t.0 == i as u8),
            "implementation.obligations",
        )
    }
}
wire_record!(Layout {parameters:H<'a>,bootstrap:Option<H<'a>>,layout:H<'a>,trainer:H<'a>,keys:H<'a>,consumed_plan:Bytes<'a>,consumed_plan_sha256:H<'a>,request:Bytes<'a>,request_sha256:H<'a>});
impl Layout<'_> {
    pub(super) fn validate(&self, e: &Evidence<'_>, x: &ExecutionFields<'_>) -> Result<()> {
        same(
            self.parameters.0 == x.parameter
                && self.bootstrap.map(|h| h.0) == x.bootstrap
                && self.layout.0 == x.layout
                && self.trainer.0 == x.trainer
                && self.keys.0 == x.keys,
            "layout.descriptors",
        )?;
        e.consumed(self.consumed_plan.0, self.consumed_plan_sha256)?;
        digest(self.request.0, self.request_sha256, "layout.request.digest")?;
        let request = Auxiliary::decode(AuxKind::Request, self.request.0, e.mode)?;
        let consumption = Auxiliary::decode(AuxKind::Consumption, self.consumed_plan.0, e.mode)?;
        check_request_consumption(&request, &consumption)?;
        let eval = request.request()?.eval;
        same(
            eval[0].0 == u8::from(x.eval.hoist)
                && eval[1].0 == u8::from(x.eval.bsgs)
                && eval[2].0 == u8::from(x.eval.lazy_relin),
            "layout.eval",
        )
    }
}
#[derive(Debug, Clone)]
pub(super) struct Requested {
    pub(super) max_bytes: u64,
    pub(super) max_records: u64,
    pub(super) plan_bytes: u64,
    pub(super) retained_bytes: u64,
    pub(super) live_temporary_bytes: u64,
    pub(super) peak_temporary_bytes: u64,
}
impl<'a> Wire<'a> for Requested {
    const MIN: usize = 48;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            max_bytes: c.u64()?,
            max_records: c.u64()?,
            plan_bytes: c.u64()?,
            retained_bytes: c.u64()?,
            live_temporary_bytes: c.u64()?,
            peak_temporary_bytes: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        for n in [
            self.max_bytes,
            self.max_records,
            self.plan_bytes,
            self.retained_bytes,
            self.live_temporary_bytes,
            self.peak_temporary_bytes,
        ] {
            e.u64(n)?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone)]
pub(super) struct Component {
    pub(super) kind: Tag<21>,
    pub(super) phase_mask: Tag<15>,
    pub(super) resident_bytes: u64,
    pub(super) additional_bytes: u64,
}
impl<'a> Wire<'a> for Component {
    const MIN: usize = 18;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        Ok(Self {
            kind: Tag::read(c)?,
            phase_mask: Tag::read(c)?,
            resident_bytes: c.u64()?,
            additional_bytes: c.u64()?,
        })
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.kind.write(e)?;
        self.phase_mask.write(e)?;
        e.u64(self.resident_bytes)?;
        e.u64(self.additional_bytes)
    }
}
#[derive(Debug, Clone)]
pub(super) struct Requirements {
    pub(super) components: Vec<Component>,
    pub(super) peak_host_bytes: u64,
    pub(super) incremental_host_bytes: u64,
    pub(super) required_disk_bytes: u64,
}
impl<'a> Wire<'a> for Requirements {
    const MIN: usize = 32;
    fn read(c: &mut Cursor<'a>) -> Result<Self> {
        // Fixed complete inventory, before allocating even a small vector.
        let count = c.u64()?;
        same(count == 22, "resource.components.count")?;
        ensure(
            c.remaining() >= 22 * 18 + 24,
            DocumentErrorCode::OutOfRange,
            "resource.components.truncated",
        )?;
        let mut components = Vec::new();
        components
            .try_reserve_exact(22)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "resource.components.allocate"))?;
        for _ in 0..22 {
            components.push(Component::read(c)?);
        }
        let value = Self {
            components,
            peak_host_bytes: c.u64()?,
            incremental_host_bytes: c.u64()?,
            required_disk_bytes: c.u64()?,
        };
        value.validate()?;
        Ok(value)
    }
    fn write(&self, e: &mut Encoder) -> Result<()> {
        self.validate()?;
        self.components.write(e)?;
        e.u64(self.peak_host_bytes)?;
        e.u64(self.incremental_host_bytes)?;
        e.u64(self.required_disk_bytes)
    }
}
impl Requirements {
    pub(super) fn validate(&self) -> Result<()> {
        same(self.components.len() == 22, "resource.components.count")?;
        let mut total = [0; 4];
        let mut incremental = [0; 4];
        for (i, c) in self.components.iter().enumerate() {
            same(
                c.kind.0 == i as u8 && c.phase_mask.0 <= 15,
                "resource.component.order",
            )?;
            same(
                c.phase_mask.0 != 0 || (c.resident_bytes == 0 && c.additional_bytes == 0),
                "resource.inapplicable.bytes",
            )?;
            let bytes = add(c.resident_bytes, c.additional_bytes)?;
            for phase in 0..4 {
                if c.phase_mask.0 & (1 << phase) != 0 {
                    total[phase] = add(total[phase], bytes)?;
                    incremental[phase] = add(incremental[phase], c.additional_bytes)?;
                }
            }
        }
        same(
            self.peak_host_bytes == *total.iter().max().expect("four phases")
                && self.incremental_host_bytes == *incremental.iter().max().expect("four phases"),
            "resource.peak",
        )
    }
}
wire_record!(Cgroup {version:Tag<1>,path:Str<'a,4096>,limit:Option<u64>,current:u64,limit_raw:FileRef<'a>,current_raw:FileRef<'a>});
wire_record!(Observation {physical:H<'a>,clock:Clock,host_mem_available:u64,cgroups:Vec<Cgroup<'a>>,disk_available:u64,host_raw:FileRef<'a>,disk_raw:FileRef<'a>});
wire_record!(Claim {mechanism:Tag<0>,claim_id:Id<'a>,claim_record:FileRef<'a>,overlap:Tag<1>,other_claimed_host_bytes:u64,other_claimed_disk_bytes:u64});
wire_record!(Resource {applicability:Applicability<'a>,physical:H<'a>,model:FileRef<'a>,model_validation:FileRef<'a>,requested:Requested,requirements:Requirements,observation:Observation<'a>,claim:Claim<'a>});
impl Resource<'_> {
    pub(super) fn validate(
        &self,
        e: &Evidence<'_>,
        x: &ExecutionFields<'_>,
        case: &admission::CaseIdentity,
    ) -> Result<()> {
        self.applicability.matches(e, x, case)?;
        same(case.backend == CaseBackend::Cpu, "resource.cpu_only")?;
        same(
            self.physical.0 == x.physical && self.physical == self.observation.physical,
            "resource.physical",
        )?;
        for f in [
            &self.model,
            &self.model_validation,
            &self.observation.host_raw,
            &self.observation.disk_raw,
            &self.claim.claim_record,
        ] {
            f.validate()?;
        }
        for c in &self.observation.cgroups {
            same(
                !c.path.0.is_empty() && !c.path.0.contains('\0'),
                "resource.cgroup.path",
            )?;
            c.limit_raw.validate()?;
            c.current_raw.validate()?;
        }
        same(
            self.requested.live_temporary_bytes <= self.requested.peak_temporary_bytes,
            "resource.temporary_peak",
        )?;
        same(
            add(
                add(self.requested.plan_bytes, self.requested.retained_bytes)?,
                self.requested.live_temporary_bytes,
            )? <= self.requested.max_bytes,
            "resource.requested_limit",
        )?;
        self.requirements.validate()
    }
    /// Arithmetic over supplied observations, NEVER an OS observation/reservation.
    pub(super) fn check_supplied_capacity(&self) -> Result<()> {
        self.requirements.validate()?;
        let mut available = self.observation.host_mem_available;
        for c in &self.observation.cgroups {
            if let Some(limit) = c.limit {
                available = available.min(limit.saturating_sub(c.current));
            }
        }
        same(
            add(
                self.requirements.incremental_host_bytes,
                self.claim.other_claimed_host_bytes,
            )? <= available
                && add(
                    self.requirements.required_disk_bytes,
                    self.claim.other_claimed_disk_bytes,
                )? <= self.observation.disk_available,
            "resource.supplied_capacity",
        )
    }
}
wire_record!(User {authority_bytes:Bytes<'a>,authority_sha256:H<'a>,current_bytes:Bytes<'a>,current_sha256:H<'a>,authority_file:FileRef<'a>,current_file:FileRef<'a>,selected_scope_index:u64,observed:Clock});
impl User<'_> {
    pub(super) fn validate(&self, e: &Evidence<'_>, case: &admission::CaseIdentity) -> Result<()> {
        digest(
            self.authority_bytes.0,
            self.authority_sha256,
            "user.authority.digest",
        )?;
        digest(
            self.current_bytes.0,
            self.current_sha256,
            "user.current.digest",
        )?;
        self.authority_file.check_bytes(self.authority_bytes.0)?;
        self.current_file.check_bytes(self.current_bytes.0)?;
        same(
            self.authority_file.root.0 == 4 && self.current_file.root.0 == 4,
            "user.file.root",
        )?;
        let scope = Auxiliary::decode(AuxKind::UserScope, self.authority_bytes.0, e.mode)?;
        let current = Auxiliary::decode(AuxKind::UserCurrent, self.current_bytes.0, e.mode)?;
        check_authority(&scope, &current, self.selected_scope_index)?;
        let s = match &scope.body {
            AuxiliaryBody::UserScope(s) => s,
            _ => unreachable!("closed kind"),
        };
        let selected = &s.scopes[self.selected_scope_index as usize];
        same(
            selected.branch == e.common.scope
                && selected.campaign == e.common.campaign_id
                && selected.profile == e.common.profile_id
                && selected.registration_id == e.branch.registration_id()
                && selected.attempt == e.branch.attempt(),
            "user.selected_scope",
        )?;
        match &e.branch {
            Branch::Full(f) => {
                let mut a = Encoder::new(DocumentLayout::Start);
                f.predecessor.write(&mut a)?;
                let mut b = Encoder::new(DocumentLayout::Start);
                selected.predecessor.write(&mut b)?;
                same(a.bytes == b.bytes, "user.retry.predecessor")?;
                // The immutable case carries no layer/source graph. The live reader
                // must independently classify encrypted blocks from retained canonical
                // profile/source and enforce the unchanged global hold even if omitted
                // from this administrative capture; no pass claim exists here.
                let _ = case;
            }
            Branch::Quick(_) => same(selected.predecessor.is_none(), "user.quick.initial")?,
        }
        // UserCurrent's administrative clock can have a different operation
        // origin. Only the three Quick event clocks have a shared-origin rule.
        Ok(())
    }
}
