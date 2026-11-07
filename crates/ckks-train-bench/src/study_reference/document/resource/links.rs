//! Exact comparisons between supplied inert records. This is not a trusted
//! package loader or live applicability check and cannot make an owner/permit.
use super::*;
#[derive(Debug)]
pub(super) struct Baseline<'a> {
    model: Source<'a>,
    validation: Source<'a>,
    policy: Source<'a>,
    claim: Source<'a>,
    host: Source<'a>,
    disk: Source<'a>,
}
impl<'a> Baseline<'a> {
    pub(super) fn new(
        model: Source<'a>,
        validation: Source<'a>,
        policy: Source<'a>,
        claim: Source<'a>,
        host: Source<'a>,
        disk: Source<'a>,
    ) -> SResult<Self> {
        require(
            matches!(model.body, Body::Model(_))
                && matches!(validation.body, Body::Validation(_))
                && matches!(policy.body, Body::Policy(_))
                && matches!(claim.body, Body::Claim(_))
                && matches!(host.body, Body::Host(_))
                && matches!(disk.body, Body::Disk(_)),
            "record-kind",
            "source.baseline",
        )?;
        require(
            [&validation, &policy, &claim, &host, &disk]
                .iter()
                .all(|s| s.mode == model.mode),
            "domain",
            "source.baseline",
        )?;
        Ok(Self {
            model,
            validation,
            policy,
            claim,
            host,
            disk,
        })
    }
    fn model(&self) -> &Model {
        let Body::Model(v) = &self.model.body else {
            unreachable!()
        };
        v
    }
    fn validation(&self) -> &Validation {
        let Body::Validation(v) = &self.validation.body else {
            unreachable!()
        };
        v
    }
    fn policy(&self) -> &HostPolicy {
        let Body::Policy(v) = &self.policy.body else {
            unreachable!()
        };
        v
    }
    fn claim(&self) -> &ClaimRecord {
        let Body::Claim(v) = &self.claim.body else {
            unreachable!()
        };
        v
    }
    fn link(
        reference: &FileRef,
        source: &Source<'_>,
        file: &str,
        reason: &'static str,
        p: &str,
    ) -> SResult<()> {
        require(
            reference.root.0 == 2
                && reference.path.0 == file
                && reference.length == source.raw.len() as u64
                && reference.sha256.0 == source.sha256(),
            reason,
            p,
        )
    }
    pub(super) fn check(
        &self,
        s: &Source<'_>,
        supplied_materialized: &str,
        physical: &str,
        sequence: u64,
    ) -> SResult<()> {
        require(s.mode == self.model.mode, "domain", "source.mode")?;
        let m = self.model();
        let v = self.validation();
        let p = self.policy();
        let c = self.claim();
        match &s.body {
            Body::Model(candidate) => {
                require(
                    candidate.registry == m.registry,
                    "registry-identity",
                    "model.registry",
                )?;
            }
            Body::Validation(candidate) => {
                Self::link(
                    &candidate.model,
                    &self.model,
                    "resource/linux-cpu-v1/model.json",
                    "file-link",
                    "validation.model",
                )?;
                require(
                    candidate.source_manifest == m.source_manifest,
                    "source-identity",
                    "validation.source_manifest",
                )?;
                require(
                    candidate.source_revision == m.source_revision,
                    "source-identity",
                    "validation.source_revision",
                )?;
                // The selected build record is retained evidence: equality is
                // against a supplied expected build identity, never a new hash.
                require(
                    candidate.environment.build_receipt == v.environment.build_receipt,
                    "build-identity",
                    "validation.environment.build_receipt",
                )?;
                for r in &candidate.runs.0 {
                    require(
                        r.metadata.source_manifest == candidate.source_manifest,
                        "source-identity",
                        "validation.runs.metadata.source_manifest",
                    )?;
                    require(
                        r.metadata.source_revision == candidate.source_revision,
                        "source-identity",
                        "validation.runs.metadata.source_revision",
                    )?;
                    require(
                        r.metadata.environment == candidate.environment,
                        "environment-equality",
                        "validation.runs.metadata.environment",
                    )?;
                }
                for coverage in &candidate.coverage.0 {
                    require(
                        coverage.source_derivation
                            == v.coverage.0[coverage.kind.0 as usize].source_derivation,
                        "derivation-link",
                        "validation.coverage.source_derivation",
                    )?;
                }
            }
            Body::Policy(candidate) => {
                Self::link(
                    &candidate.model,
                    &self.model,
                    "resource/linux-cpu-v1/model.json",
                    "file-link",
                    "policy.model",
                )?;
                Self::link(
                    &candidate.validation,
                    &self.validation,
                    "resource/linux-cpu-v1/validation.json",
                    "file-link",
                    "policy.validation",
                )?;
                require(
                    candidate.allocator_environment == v.environment.allocator
                        && candidate.kernel_environment == v.environment.kernel,
                    "policy-environment",
                    "policy.allocator_environment",
                )?;
            }
            Body::Claim(candidate) => {
                Self::link(
                    &candidate.model,
                    &self.model,
                    "resource/linux-cpu-v1/model.json",
                    "file-link",
                    "claim.model",
                )?;
                Self::link(
                    &candidate.validation,
                    &self.validation,
                    "resource/linux-cpu-v1/validation.json",
                    "file-link",
                    "claim.validation",
                )?;
                require(
                    candidate.source_manifest == m.source_manifest
                        && candidate.source_revision == m.source_revision,
                    "source-identity",
                    "claim.source_manifest",
                )?;
                require(
                    candidate.physical_host == p.physical_host && candidate.boot_id == p.boot_id,
                    "claim-host",
                    "claim.physical_host",
                )?;
                require(
                    candidate.approval == c.approval && candidate.claim_id == c.claim_id,
                    "claim-link",
                    "claim.claim_id",
                )?;
                let matched=match &candidate.work {
                    ClaimWork::Train{registration,request}=>{
                        let case=admission::parse_case_binary(&registration.0).map_err(|_|refusal("registration","claim.work.registration"))?;
                        require(case.registry_sha256==m.registry,"registry-identity","claim.work.registration.registry")?;
                        let consumption=evidence::resource_request_consumption(&request.0,s.mode).map_err(|_|refusal("request-domain","claim.work.request"))?;
                        let digest=sha256_hex(&registration.0);
                        m.supported_work.0.iter().any(|w|matches!(w,SupportedWork::Train{registration_id,consumption:c} if *registration_id==digest&&*c==consumption))
                    },
                    ClaimWork::Reference{operation,profile_id,seed,..}=>m.supported_work.0.iter().any(|w|matches!(w,SupportedWork::Reference{operation:o,profile_id:p,seeds} if o==operation&&p==&profile_id.0&&seeds.binary_search(seed).is_ok())),
                    ClaimWork::Target{operation,profile_id,selected_seed,supplied_input,..}=>{
                        if let Some(h)=supplied_input {require(h.0==supplied_materialized,"supplied-materialized","claim.work.supplied_input")?;}
                        m.supported_work.0.iter().any(|w|matches!(w,SupportedWork::Target{operation:o,profile_id:p,seeds} if o==operation&&p==&profile_id.0&&seeds.binary_search(selected_seed).is_ok()))
                    }
                };
                require(matched, "work-membership", "claim.work")?;
            }
            Body::Host(candidate) => {
                require(
                    candidate.claim_id == c.claim_id,
                    "claim-link",
                    "host.claim_id",
                )?;
                require(
                    candidate.sequence == sequence,
                    "observation-sequence",
                    "host.sequence",
                )?;
                require(
                    candidate.physical.0 == physical,
                    "physical-link",
                    "host.physical",
                )?;
                require(
                    candidate.topology_before.namespaces == p.host_namespace_origin
                        && candidate.topology_after.namespaces == p.host_namespace_origin,
                    "topology-origin",
                    "host.topology_after.namespaces",
                )?;
                require(
                    candidate.topology_before == candidate.topology_after,
                    "topology-change",
                    "host.topology_after",
                )?;
            }
            Body::Disk(candidate) => {
                require(
                    candidate.claim_id == c.claim_id,
                    "claim-link",
                    "disk.claim_id",
                )?;
                require(
                    candidate.sequence == sequence,
                    "observation-sequence",
                    "disk.sequence",
                )?;
                require(
                    candidate.filesystems.0.len() == p.filesystem_environment.0.len(),
                    "filesystem-coverage",
                    "disk.filesystems",
                )?;
                for (f, policy) in candidate
                    .filesystems
                    .0
                    .iter()
                    .zip(&p.filesystem_environment.0)
                {
                    require(
                        f.identity == policy.identity
                            && f.roots == policy.roots
                            && f.quota_evidence == policy.quota_evidence,
                        "filesystem-coverage",
                        "disk.filesystems",
                    )?;
                }
                let total =
                    checked_disk_sum(candidate.filesystems.0.iter().map(|f| f.required_bytes))?;
                require(
                    total == c.modeled_envelope.required_disk_bytes,
                    "disk-envelope",
                    "disk.filesystems.required_bytes",
                )?;
            }
        }
        Ok(())
    }
    pub(super) fn check_bootstrap(
        &self,
        a: &Bootstrap<'_>,
        expected: &Bootstrap<'_>,
    ) -> SResult<()> {
        let p = self.policy();
        require(
            a.policy == self.policy.sha256(),
            "bootstrap-policy",
            "bootstrap.policy",
        )?;
        require(
            a.origin_evidence == p.host_namespace_origin.installation_record.sha256.0,
            "bootstrap-origin",
            "bootstrap.origin_evidence",
        )?;
        require(
            a.boot_uuid == p.boot_id.0 && a.boot_uuid == p.host_namespace_origin.boot_id.0,
            "bootstrap-boot",
            "bootstrap.boot_uuid",
        )?;
        let o = &p.host_namespace_origin;
        let ns = [&o.mount_ns, &o.cgroup_ns, &o.pid_ns, &o.user_ns];
        require(
            a.namespaces
                .iter()
                .zip(ns)
                .all(|(a, b)| *a == [b.device, b.inode]),
            "bootstrap-namespaces",
            "bootstrap.namespaces",
        )?;
        require(
            a.kernel_profile == expected.kernel_profile
                && a.allocator_profile == expected.allocator_profile
                && a.filesystem_profile == expected.filesystem_profile,
            "bootstrap-profile",
            "bootstrap.kernel_profile",
        )?;
        require(
            a.mode == expected.mode
                && a.generation == expected.generation
                && a.account_uid == expected.account_uid
                && a.host_key == expected.host_key
                && a.proc_root == expected.proc_root
                && a.cgroup_root == expected.cgroup_root,
            "bootstrap-anchor",
            "bootstrap.expected",
        )
    }
}
