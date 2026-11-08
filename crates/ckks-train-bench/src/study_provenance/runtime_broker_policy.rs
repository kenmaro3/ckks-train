//! Supervisor-selected runtime helpers, from the already pinned current build.
use super::*;
use crate::study_provenance::resource_operational::retained::{
    HeldDirectory, HeldFile, MinimalOrigin,
};
pub(crate) struct RuntimeBrokerPolicy {
    root: PathBuf,
    receipt_path: PathBuf,
    python: PathBuf,
    git: PathBuf,
    environment: Environment,
    receipt: HeldFile,
    receipt_hash: String,
    receipt_bytes: Vec<u8>,
    held: Vec<HeldFile>,
    source: HeldDirectory,
    pub executable_hash: String,
}
impl RuntimeBrokerPolicy {
    pub fn authority(&self) -> &HeldDirectory {
        self.receipt.parent()
    }
    pub fn receipt_reference(&self) -> Result<crate::study_reference::operational::FileReference> {
        Ok(crate::study_reference::operational::FileReference {
            root: 4,
            path: self
                .receipt_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or(ProvenanceError("receipt basename"))?
                .into(),
            length: self.receipt.file.metadata()?.len(),
            hash: self.receipt_hash.clone(),
        })
    }
    pub fn load(
        origin: &MinimalOrigin,
        root: &Path,
        paths: &[PathBuf],
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        ensure(
            !paths.is_empty() && paths.len() <= 64,
            "broker selected receipts",
        )?;
        let embedded = EMBEDDED_CAPSULE.ok_or(ProvenanceError("broker compile capsule missing"))?;
        let capsule = Capsule::decode(embedded)?;
        let role = Recipe::capsule_policy(&capsule)?;
        ensure(
            matches!(
                role,
                Recipe::TrainCpuSerialObserved | Recipe::ReferenceObserved
            ),
            "historical build cannot own supervisor",
        )?;
        let executable = std::env::current_exe()?;
        let boot = crate::study_reference::operational::bootstrap(
            &origin.bootstrap[..origin.bootstrap_len],
        )
        .map_err(|_| ProvenanceError("broker bootstrap"))?;
        ensure(
            boot.recipe == capsule.recipe
                && capsule.cwd == root.as_os_str().as_bytes()
                && capsule.manifests[0] == origin.approval.source.text()?
                && capsule.commit == origin.approval.revision.text()?,
            "broker source/build identity",
        )?;
        let mut selected = None;
        for path in paths {
            pulse()?;
            let mut held = HeldFile::absolute(path, LIMIT)?;
            let bytes = held.fixed_name_bytes(LIMIT)?;
            let hash = sha256_hex(&bytes);
            let matches = origin.approval.builds[..origin.approval.build_count]
                .iter()
                .filter(|p| {
                    p.receipt.as_bytes() == hash.as_bytes()
                        && p.recipe.as_bytes() == capsule.recipe.as_bytes()
                        && p.capsule.as_bytes() == sha256_hex(embedded).as_bytes()
                        && p.executable.as_bytes() == boot.executable.as_bytes()
                })
                .count();
            if matches == 0 {
                continue;
            }
            ensure(
                matches == 1 && selected.is_none(),
                "ambiguous broker current receipt",
            )?;
            let receipt = Receipt::decode(&bytes)?;
            let decoded = receipt.closed_policy()?;
            ensure(
                receipt.capsule == embedded
                    && receipt.executable == executable.as_os_str().as_bytes()
                    && receipt.executable_hash == boot.executable,
                "broker exact executable receipt",
            )?;
            let environment = Environment::decode(&receipt.environment)?;
            ensure(
                validate_linkage(&receipt, &decoded, &environment)?.0 == role,
                "broker recipe linkage",
            )?;
            selected = Some((held, receipt, environment, path.clone(), hash));
        }
        let (receipt_file, receipt, environment, receipt_path, receipt_hash) =
            selected.ok_or(ProvenanceError("broker approved receipt unavailable"))?;
        Self::selected(
            root,
            receipt_file,
            receipt,
            environment,
            receipt_path,
            receipt_hash,
            role,
            pulse,
        )
    }
    pub fn load_probe(
        inputs: &crate::study_provenance::resource_operational::probe::ProbeInputs,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        use crate::study_provenance::resource_operational::probe::receipt_tuple;
        let p = &inputs.origin;
        let embedded = EMBEDDED_CAPSULE.ok_or(ProvenanceError("probe embedded capsule missing"))?;
        let capsule = Capsule::decode(embedded)?;
        let role = Recipe::capsule_policy(&capsule)?;
        ensure(
            matches!(
                role,
                Recipe::TrainCpuSerialObserved | Recipe::ReferenceObserved
            ),
            "probe requires observed v2 build",
        )?;
        let root = path(&capsule.cwd)?;
        ensure(
            root.join("results").as_os_str().as_bytes() == p.observation_path.as_bytes(),
            "probe Observation root must be build repository/results",
        )?;
        let mut file = inputs.policy.retained_file(p.receipt.path.text()?, LIMIT)?;
        let raw = file.bytes((p.receipt.length, p.receipt.hash.text()?))?;
        pulse()?;
        let executable = std::env::current_exe()?;
        receipt_tuple(
            &raw,
            embedded,
            p.source.text()?,
            p.revision.text()?,
            p.executable.text()?,
            p.recipe.text()?,
            executable.as_os_str().as_bytes(),
        )?;
        let receipt = Receipt::decode(&raw)?;
        let decoded = receipt.closed_policy()?;
        let environment = Environment::decode(&receipt.environment)?;
        ensure(
            validate_linkage(&receipt, &decoded, &environment)?.0 == role,
            "probe receipt linkage",
        )?;
        let receipt_path =
            Path::new(OsStr::from_bytes(p.policy_path.as_bytes())).join(p.receipt.path.text()?);
        Self::selected(
            &root,
            file,
            receipt,
            environment,
            receipt_path,
            sha256_hex(&raw),
            role,
            pulse,
        )
    }
    #[allow(clippy::too_many_arguments)] // Closed retained identity projection; no generic authority provider.
    fn selected(
        root: &Path,
        receipt_file: HeldFile,
        receipt: Receipt,
        environment: Environment,
        receipt_path: PathBuf,
        receipt_hash: String,
        role: Recipe,
        pulse: &mut impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        let executable = std::env::current_exe()?;
        let python = path(environment.get("HEKIT_STUDY_PYTHON")?)?;
        let git = path(flag(&receipt.argv, b"--git")?)?;
        let source = HeldDirectory::open(root)?;
        let source_manifest = Manifest::decode(&receipt.files[0])?;
        let tool_manifest = Manifest::decode(&receipt.files[3])?;
        let mut held = Vec::new();
        for file in [&python, &git, &executable] {
            let expected = if file == &executable {
                receipt.executable_hash.as_str()
            } else {
                tool_manifest
                    .records
                    .iter()
                    .find_map(|(name, value)| match value {
                        FileValue::Regular(h) if name == file.as_os_str().as_bytes() => {
                            Some(h.as_str())
                        }
                        _ => None,
                    })
                    .ok_or(ProvenanceError("broker tool hash absent"))?
            };
            let mut f = HeldFile::absolute(file, usize::MAX)?;
            ensure(
                f.hash_incremental(pulse)? == expected,
                "broker executable/tool changed",
            )?;
            held.push(f);
        }
        for name in [
            "scripts/study_reference_build.py",
            "scripts/study_reference_build_support.py",
            "scripts/study_runtime_broker.py",
            role.helper(),
        ] {
            let expected = source_manifest
                .records
                .iter()
                .find_map(|(p, value)| match value {
                    FileValue::Regular(h) if p == name.as_bytes() => Some(h.as_str()),
                    _ => None,
                })
                .ok_or(ProvenanceError("broker helper source absent"))?;
            let mut f = source.retained_file(name, LIMIT)?;
            ensure(
                f.hash_incremental(pulse)? == expected,
                "broker helper changed",
            )?;
            held.push(f);
        }
        let receipt_bytes = receipt.encode()?;
        ensure(
            crate::data::sha256_hex(&receipt_bytes) == receipt_hash,
            "held canonical receipt identity",
        )?;
        Ok(Self {
            root: root.into(),
            receipt_path,
            python,
            git,
            environment,
            receipt: receipt_file,
            receipt_hash,
            receipt_bytes,
            held,
            source,
            executable_hash: receipt.executable_hash,
        })
    }
    pub fn probe_requests(&self) -> Result<Vec<Command>> {
        self.recheck()?;
        let helper = self.root.join("scripts/study_reference_build.py");
        let mut hash = Command::new(&self.python);
        hash.arg("-B")
            .arg(&helper)
            .arg("hash-file")
            .arg("--path")
            .arg(std::env::current_exe()?)
            .current_dir(&self.root);
        let mut snapshot = Command::new(&self.python);
        snapshot
            .arg("-B")
            .arg(&helper)
            .arg("snapshot")
            .arg("--root")
            .arg(&self.root)
            .arg("--receipt")
            .arg(&self.receipt_path)
            .arg("--git")
            .arg(&self.git)
            .current_dir(&self.root);
        Ok(vec![hash, snapshot])
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn retained_mounts(&self) -> Result<Vec<(u64, u64, u64)>> {
        self.held
            .iter()
            .map(|f| {
                crate::study_provenance::resource_operational::retained::mount_identity(&f.file)
            })
            .collect()
    }
    pub fn check_probe_copy(&self, copy: &[u8]) -> Result<()> {
        self.receipt.recheck()?;
        crate::study_provenance::resource_operational::probe::receipt_copy(
            &self.receipt_bytes,
            copy,
        )
    }
    pub fn recheck(&self) -> Result<()> {
        self.source.recheck()?;
        self.receipt.recheck()?;
        for f in &self.held {
            f.recheck()?;
        }
        Ok(())
    }
    pub fn command(
        &self,
        kind: u8,
        cwd: &[u8],
        argv: &[Vec<u8>],
        lock: (u64, u64),
        parent: u32,
    ) -> Result<Command> {
        self.recheck()?;
        ensure(
            cwd == self.root.as_os_str().as_bytes(),
            "broker fixed working directory",
        )?;
        let helper = self.root.join("scripts/study_reference_build.py");
        let prefix = vec![
            self.python.as_os_str().as_bytes().to_vec(),
            b"-B".to_vec(),
            helper.as_os_str().as_bytes().to_vec(),
        ];
        if kind == 0 {
            let mut hash = prefix.clone();
            hash.extend([
                b"hash-file".to_vec(),
                b"--path".to_vec(),
                std::env::current_exe()?.as_os_str().as_bytes().to_vec(),
            ]);
            let mut snapshot = prefix;
            snapshot.extend([
                b"snapshot".to_vec(),
                b"--root".to_vec(),
                self.root.as_os_str().as_bytes().to_vec(),
                b"--receipt".to_vec(),
                self.receipt_path.as_os_str().as_bytes().to_vec(),
                b"--git".to_vec(),
                self.git.as_os_str().as_bytes().to_vec(),
            ]);
            ensure(argv == hash || argv == snapshot, "Python broker exact argv")?;
        } else {
            ensure(
                kind == 1
                    && argv
                        .first()
                        .is_some_and(|v| v == self.git.as_os_str().as_bytes())
                    && argv.len() >= 8,
                "Git broker program",
            )?;
            let suffix = argv[8..]
                .iter()
                .map(|v| std::str::from_utf8(v).map_err(|_| ProvenanceError("Git suffix UTF8")))
                .collect::<Result<Vec<_>>>()?;
            ensure(
                crate::study_supervisor::git_arguments(self.root.as_os_str().as_bytes(), &suffix)?
                    == argv[1..],
                "Git broker exact argv",
            )?;
        }
        let mut command = Command::new(OsStr::from_bytes(&argv[0]));
        command
            .args(argv[1..].iter().map(|a| OsStr::from_bytes(a)))
            .current_dir(&self.root)
            .env_clear();
        for (k, v) in &self.environment.0 {
            if let Some(v) = v {
                command.env(k, OsStr::from_bytes(v));
            }
        }
        if kind == 0 {
            command
                .arg("--broker-runtime")
                .arg("--lock-device")
                .arg(lock.0.to_string())
                .arg("--lock-inode")
                .arg(lock.1.to_string())
                .arg("--supervisor-pid")
                .arg(parent.to_string());
        }
        Ok(command)
    }
}
