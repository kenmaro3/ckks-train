//! Held positive package acquisition, after lock and initial monitored claim.
use super::super::{
    approval::OriginRef,
    retained::{HeldDirectory, HeldFile, MinimalOrigin, mount_identity},
};
use super::{package_path, receipt_join};
use crate::{
    study_provenance::{
        ProvenanceError, Result, context::runtime_broker_policy::RuntimeBrokerPolicy, ensure,
    },
    study_reference::operational as wire,
};
use std::{collections::BTreeMap, path::Path};
pub(crate) struct Package {
    files: Vec<HeldFile>,
    pins: BTreeMap<String, wire::FileReference>,
    bytes: u64,
    pub roots: Vec<(u8, HeldDirectory)>,
    pub account: HeldDirectory,
    pub filesystems: Vec<wire::FilesystemFacts>,
    pub mountinfo: Vec<u8>,
    pub membership: Vec<u8>,
    pub controllers: Vec<u8>,
}
fn reference(r: &OriginRef) -> Result<wire::FileReference> {
    Ok(wire::FileReference {
        root: r.root,
        path: r.path.text()?.into(),
        length: r.length,
        hash: r.hash.text()?.into(),
    })
}
impl Package {
    fn read(
        &mut self,
        policy: &HeldDirectory,
        r: &wire::FileReference,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Vec<u8>> {
        package_path(r)?;
        pulse()?;
        if let Some(prior) = self.pins.get(&r.path) {
            ensure(prior == r, "package path rebound to different contents")?;
        } else {
            ensure(self.files.len() < 4096, "operational package file count")?;
            self.bytes = self
                .bytes
                .checked_add(r.length)
                .ok_or(ProvenanceError("package storage count overflow"))?;
            ensure(
                self.bytes <= 64 * 1024 * 1024,
                "aggregate retained package payload",
            )?;
        }
        let mut held = policy.retained_file(&r.path, 8 * 1024 * 1024)?;
        let bytes = held.bytes((r.length, &r.hash))?;
        pulse()?;
        if !self.pins.contains_key(&r.path) {
            self.pins.insert(r.path.clone(), r.clone());
            self.files.push(held);
        }
        Ok(bytes)
    }
    fn fixed(
        &mut self,
        policy: &HeldDirectory,
        name: &str,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Vec<u8>> {
        let mut held = policy.retained_file(name, 8 * 1024 * 1024)?;
        let raw = held.fixed_name_bytes(8 * 1024 * 1024)?;
        // Wrapper parser must subsequently check this exact fixed-name review
        // against its pinned FileRef. This read alone conveys no acceptance.
        let r = wire::FileReference {
            root: 3,
            path: name.into(),
            length: raw.len() as u64,
            hash: crate::data::sha256_hex(&raw),
        };
        self.read(policy, &r, pulse)
    }
    pub fn load(
        origin: &MinimalOrigin,
        build: &RuntimeBrokerPolicy,
        root: &Path,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let mut p = Self {
            files: Vec::new(),
            pins: BTreeMap::new(),
            bytes: 0,
            roots: vec![
                (1, HeldDirectory::open(root)?),
                (3, origin.policy.duplicate()?),
                (4, build.authority().duplicate()?),
                (5, origin.observation.duplicate()?),
            ],
            account: origin.account.duplicate()?,
            filesystems: Vec::new(),
            mountinfo: Vec::new(),
            membership: Vec::new(),
            controllers: Vec::new(),
        };
        let bootstrap = wire::bootstrap(&origin.bootstrap[..origin.bootstrap_len])
            .map_err(|_| ProvenanceError("package bootstrap"))?;
        ensure(
            bootstrap.policy.0 == 3
                && bootstrap.policy.1 == "resource/linux-cpu-observed-v1/policy.json",
            "fixed operational policy name",
        )?;
        let policy_ref = wire::FileReference {
            root: 3,
            path: bootstrap.policy.1.into(),
            length: bootstrap.policy.2,
            hash: bootstrap.policy.3.into(),
        };
        let policy = p.read(&origin.policy, &policy_ref, pulse)?;
        let native = wire::policy_native(&policy, &origin.bootstrap[..origin.bootstrap_len])
            .map_err(|_| ProvenanceError("policy/bootstrap/source joins"))?;
        ensure(
            native == reference(&origin.approval.origin.native_validation)?,
            "policy/approval native selection",
        )?;
        let validation = p.read(&origin.policy, &native, pulse)?;
        let review = p.fixed(
            &origin.policy,
            "resource/linux-cpu-observed-v1/native-review.json",
            pulse,
        )?;
        use std::os::unix::ffi::OsStrExt;
        let facts = wire::validate_native_sources(
            &validation,
            &review,
            std::env::current_exe()?.as_os_str().as_bytes(),
            false,
        )
        .map_err(|_| ProvenanceError("native selected evidence/review invalid"))?;
        ensure(
            facts.identity.source == bootstrap.source
                && facts.identity.revision == bootstrap.revision
                && facts.identity.executable == build.executable_hash
                && facts.identity.recipe == bootstrap.recipe
                && facts.identity.host == reference(&origin.approval.origin.host_origin)?,
            "native actual source/build/host applicability",
        )?;
        let authority = build.authority().identity()?;
        receipt_join(
            &facts.receipt,
            &build.receipt_reference()?,
            authority,
            p.roots[2].1.identity()?,
        )?;
        for check in &facts.checks {
            let command = p.read(&origin.policy, &check.refs[0], pulse)?;
            let stdout = p.read(&origin.policy, &check.refs[1], pulse)?;
            let stderr = p.read(&origin.policy, &check.refs[2], pulse)?;
            let observations = p.read(&origin.policy, &check.refs[3], pulse)?;
            use wire::probe_output::{ProbeObservation, check_stdout};
            wire::probe_output::command(&command, &check.argv, root.as_os_str().as_bytes())
                .map_err(|_| ProvenanceError("native original command"))?;
            let observed = ProbeObservation::parse(&observations)
                .map_err(|_| ProvenanceError("native observations envelope"))?;
            observed
                .argv_matches(&check.argv)
                .map_err(|_| ProvenanceError("native original argv/hash"))?;
            check_stdout(&stdout, &observed, &observations)
                .map_err(|_| ProvenanceError("native original/copy stdout"))?;
            ensure(
                stderr.is_empty()
                    && observed.check == check.id
                    && observed.started == (check.started.wall, check.started.elapsed)
                    && observed.finished == (check.finished.wall, check.finished.elapsed),
                "native original status/clocks",
            )?;
            ensure(
                observed.source == facts.identity.source
                    && observed.revision == facts.identity.revision
                    && observed.executable == facts.identity.executable
                    && observed.recipe == facts.identity.recipe
                    && observed.host == facts.identity.host,
                "native original observed tuple",
            )?;
            let probe = observed
                .origin()
                .map_err(|_| ProvenanceError("native original probe origin"))?;
            ensure(
                probe.account_uid == origin.host.account_uid
                    && probe.account_identity == origin.account.identity()?
                    && probe.account_path.as_bytes()
                        == origin.approval.origin.account_anchor.as_bytes()
                    && probe.policy_path.as_bytes()
                        == origin.approval.origin.policy_origin.as_bytes()
                    && probe.policy_identity == origin.policy.identity()?
                    && probe.observation_path.as_bytes()
                        == origin.approval.origin.observation_origin.as_bytes()
                    && probe.observation_identity == origin.observation.identity()?,
                "native original root applicability",
            )?;
            let copy = p.read(&origin.policy, &observed.receipt, pulse)?;
            build.check_probe_copy(&copy)?;
        }
        p.load_host(
            &origin.policy,
            &origin.host,
            bootstrap.physical,
            bootstrap.boot,
            bootstrap.source,
            bootstrap.revision,
            &build.executable_hash,
            pulse,
        )?;
        p.check_build_mounts(build)?;
        p.recheck()?;
        build.recheck()?;
        Ok(p)
    }
    pub fn load_probe(
        inputs: &super::super::probe::ProbeInputs,
        build: &RuntimeBrokerPolicy,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let mut p = Self {
            files: Vec::new(),
            pins: BTreeMap::new(),
            bytes: 0,
            roots: vec![
                (1, HeldDirectory::open(build.root())?),
                (3, inputs.policy.duplicate()?),
                (5, inputs.observation.duplicate()?),
            ],
            account: inputs.account.duplicate()?,
            filesystems: Vec::new(),
            mountinfo: Vec::new(),
            membership: Vec::new(),
            controllers: Vec::new(),
        };
        let o = &inputs.origin;
        let h = &inputs.host;
        // The upstream review can declare future Authority4, but this probe
        // retains and measures only the three roots listed above plus account.
        p.load_host(
            &inputs.policy,
            h,
            h.physical.text()?,
            h.boot.text()?,
            o.source.text()?,
            o.revision.text()?,
            o.executable.text()?,
            pulse,
        )?;
        p.check_build_mounts(build)?;
        p.recheck()?;
        build.recheck()?;
        Ok(p)
    }
    fn check_build_mounts(&self, build: &RuntimeBrokerPolicy) -> Result<()> {
        for identity in build.retained_mounts()? {
            ensure(
                self.filesystems.iter().any(|r| {
                    (r.identity.device, r.identity.mount_id, r.identity.fs_type) == identity
                }),
                "executable/helper filesystem outside independently reviewed mounts",
            )?;
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Closed retained identity projection; no generic authority provider.
    fn load_host(
        &mut self,
        policy: &HeldDirectory,
        host: &super::super::host::HostOrigin,
        physical: &str,
        boot: &str,
        source: &str,
        revision: &str,
        executable: &str,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        for r in [&host.installation, &host.machine] {
            self.read(policy, &reference(r)?, pulse)?;
        }
        self.mountinfo = self.read(policy, &reference(&host.mountinfo)?, pulse)?;
        self.membership = self.read(policy, &reference(&host.membership)?, pulse)?;
        self.controllers = self.read(policy, &reference(&host.controllers)?, pulse)?;
        let fs = self.read(policy, &reference(&host.filesystems)?, pulse)?;
        let review = self.fixed(
            policy,
            "resource/linux-cpu-observed-v1/filesystem-review.json",
            pulse,
        )?;
        wire::filesystem_applicability(&fs, physical, boot, source, revision, executable)
            .map_err(|_| ProvenanceError("filesystem actual package applicability"))?;
        self.filesystems = wire::validate_filesystem_sources(&fs, &review, &[1, 3, 4, 5], false)
            .map_err(|_| ProvenanceError("filesystem package/review/used roots"))?;
        let account = wire::filesystem_account(&fs)
            .map_err(|_| ProvenanceError("filesystem account projection"))?;
        ensure(
            mount_identity(self.account.file())?
                == (account.device, account.mount_id, account.fs_type),
            "actual account filesystem differs",
        )?;
        let mut refs = Vec::new();
        for row in &self.filesystems {
            for role in &row.roots {
                if let Some((_, root)) = self.roots.iter().find(|(r, _)| r == role) {
                    ensure(
                        mount_identity(root.file())?
                            == (
                                row.identity.device,
                                row.identity.mount_id,
                                row.identity.fs_type,
                            ),
                        "actual used root filesystem differs",
                    )?;
                }
            }
            refs.extend([
                row.mount_evidence.clone(),
                row.quota_evidence.clone(),
                row.backing_evidence.clone(),
            ]);
        }
        for r in refs {
            self.read(policy, &r, pulse)?;
        }
        self.recheck()?;
        Ok(())
    }
    pub fn recheck(&self) -> Result<()> {
        for f in &self.files {
            f.recheck()?;
        }
        self.account.recheck()?;
        for (role, root) in &self.roots {
            root.recheck()?;
            let row = self
                .filesystems
                .iter()
                .find(|r| r.roots.contains(role))
                .ok_or(ProvenanceError("used filesystem root uncovered"))?;
            ensure(
                mount_identity(root.file())?
                    == (
                        row.identity.device,
                        row.identity.mount_id,
                        row.identity.fs_type,
                    ),
                "used filesystem mount changed",
            )?;
        }
        Ok(())
    }
    pub fn require_roots(&self, required: &[u8]) -> Result<()> {
        ensure(
            required
                .iter()
                .all(|r| self.roots.iter().any(|(tag, _)| r == tag)),
            "Input/Reviewed roots not acquired by this operational owner",
        )
    }
}
