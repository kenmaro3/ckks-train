//! Pinned local trust input, immutable captured facts and exact rechecks.
use super::{EMBEDDED_CAPSULE, ProvenanceError, Result, codec::*, ensure, publication::read_path};
use crate::{
    data::sha256_hex, study_inputs::MaterializedProfile, study_reference::derive_identities,
};
use std::{
    ffi::{OsStr, OsString},
    io::Read,
    os::unix::{
        ffi::{OsStrExt, OsStringExt},
        fs::MetadataExt,
    },
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
const DESIGNS: [&str; 5] = [
    "docs/experiment-matrix.md",
    "docs/study-reference-design.md",
    "docs/study-reference-document-design.md",
    "docs/study-reference-session-design.md",
    "docs/study-training-design.md",
];
const THREAD_KEYS: [&str; 7] = [
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
];
const BUILD_KEYS: [&str; 18] = [
    "CARGO_BUILD_JOBS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME",
    "CARGO_INCREMENTAL",
    "CARGO_NET_OFFLINE",
    "CARGO_TARGET_DIR",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "GIT_OPTIONAL_LOCKS",
    "HEKIT_STUDY_CAPSULE_REQUEST",
    "HEKIT_STUDY_PYTHON",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "RUSTC",
    "TMPDIR",
    "TZ",
];
/// Explicit local authority locations. No artifact selects these files.
pub struct ContextRequest {
    root: PathBuf,
    approval: PathBuf,
    approval_sha256: String,
    receipts: Vec<PathBuf>,
}
impl ContextRequest {
    /// Supply an exact approval pin and locally selected receipt paths.
    pub fn new(
        root: PathBuf,
        approval: PathBuf,
        approval_sha256: String,
        receipts: Vec<PathBuf>,
    ) -> Self {
        Self {
            root,
            approval,
            approval_sha256,
            receipts,
        }
    }
}
struct FrozenReceipt {
    path: PathBuf,
    bytes: Vec<u8>,
    receipt: Receipt,
    capsule: Capsule,
    hash: String,
}
/// Approved local source/build context. This does not establish completed training.
pub struct VerifiedStudyContext {
    request: ContextRequest,
    approval_bytes: Vec<u8>,
    approval: Approval,
    receipts: Vec<FrozenReceipt>,
    current: usize,
    python: PathBuf,
    git: PathBuf,
    executable: PathBuf,
    executable_identity: (u64, u64),
    env: Environment,
}
fn canonical(p: &Path) -> Result<()> {
    ensure(
        p.is_absolute() && p.canonicalize()? == p,
        "canonical absolute path required",
    )
}
fn path(b: &[u8]) -> Result<PathBuf> {
    ensure(!b.contains(&0) && b.len() <= 4096, "Unix path bounds")?;
    Ok(PathBuf::from(OsString::from_vec(b.to_vec())))
}
fn flag<'a>(argv: &'a [Vec<u8>], key: &[u8]) -> Result<&'a [u8]> {
    let mut values = argv
        .windows(2)
        .filter(|v| v[0] == key)
        .map(|v| v[1].as_slice());
    let first = values
        .next()
        .ok_or(ProvenanceError("required helper option"))?;
    ensure(values.next().is_none(), "duplicate helper option")?;
    Ok(first)
}
fn env_policy(env: &Environment) -> Result<()> {
    ensure(
        env.0.iter().map(|x| x.0.as_str()).eq(BUILD_KEYS),
        "exact build environment keys",
    )?;
    for (k, v) in [
        ("LANG", "C"),
        ("LC_ALL", "C"),
        ("TZ", "UTC"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("GIT_OPTIONAL_LOCKS", "0"),
        ("CARGO_INCREMENTAL", "0"),
        ("CARGO_NET_OFFLINE", "true"),
    ] {
        ensure(env.get(k)? == v.as_bytes(), "build environment value")?;
    }
    let jobs = env.get("CARGO_BUILD_JOBS")?;
    ensure(matches!(jobs, b"1" | b"2" | b"3" | b"4"), "recipe jobs")
}
fn validate_linkage(r: &Receipt, c: &Capsule, env: &Environment) -> Result<()> {
    env_policy(env)?;
    let root = path(&c.cwd)?;
    let target = path(env.get("CARGO_TARGET_DIR")?)?;
    ensure(root.is_absolute() && target.is_absolute(), "recipe roots")?;
    let cargo = flag(&r.argv, b"--cargo")?;
    let git = flag(&r.argv, b"--git")?;
    let linker = flag(&r.argv, b"--linker")?;
    ensure(
        flag(&r.argv, b"--root")? == c.cwd
            && flag(&r.argv, b"--rustc")? == env.get("RUSTC")?
            && flag(&r.argv, b"--python")? == env.get("HEKIT_STUDY_PYTHON")?
            && flag(&r.argv, b"--target")? == env.get("CARGO_TARGET_DIR")?,
        "recipe helper linkage",
    )?;
    let mut expected_flags = b"-Clinker=".to_vec();
    expected_flags.extend_from_slice(linker);
    ensure(
        env.get("CARGO_ENCODED_RUSTFLAGS")? == expected_flags && !linker.contains(&31),
        "linker flags",
    )?;
    let tool = Manifest::decode(&r.files[3])?;
    for p in [
        cargo,
        git,
        linker,
        env.get("RUSTC")?,
        env.get("HEKIT_STUDY_PYTHON")?,
    ] {
        ensure(
            tool.records
                .iter()
                .any(|(k, v)| k == p && matches!(v, FileValue::Regular(_))),
            "tool absent from inventory",
        )?;
    }
    let meta = vec![
        cargo.to_vec(),
        b"metadata".to_vec(),
        b"--format-version".to_vec(),
        b"1".to_vec(),
        b"--locked".to_vec(),
        b"--offline".to_vec(),
        b"--no-default-features".to_vec(),
        b"--filter-platform".to_vec(),
        TARGET.as_bytes().to_vec(),
        b"--manifest-path".to_vec(),
        root.join("Cargo.toml").as_os_str().as_bytes().to_vec(),
    ];
    ensure(c.metadata_argv == meta, "metadata argv")?;
    let mut build = vec![cargo.to_vec()];
    for x in [
        "build",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "ckks-train-bench",
        "--no-default-features",
        "--bin",
        "study_reference",
        "--target",
        TARGET,
        "--target-dir",
    ] {
        build.push(x.as_bytes().to_vec());
    }
    build.push(env.get("CARGO_TARGET_DIR")?.to_vec());
    build.push(b"-j".to_vec());
    build.push(env.get("CARGO_BUILD_JOBS")?.to_vec());
    build.push(b"--message-format=json-render-diagnostics".to_vec());
    ensure(c.build_argv == build, "build argv")?;
    for (option, key) in [
        (b"--cargo-home".as_slice(), "CARGO_HOME"),
        (b"--home".as_slice(), "HOME"),
        (b"--tmp".as_slice(), "TMPDIR"),
        (b"--request".as_slice(), "HEKIT_STUDY_CAPSULE_REQUEST"),
        (b"--jobs".as_slice(), "CARGO_BUILD_JOBS"),
    ] {
        ensure(
            flag(&r.argv, option)? == env.get(key)?,
            "helper/environment linkage",
        )?;
    }
    let mut dirs = std::collections::BTreeSet::new();
    for p in [
        cargo,
        git,
        linker,
        env.get("RUSTC")?,
        env.get("HEKIT_STUDY_PYTHON")?,
    ] {
        let p = path(p)?;
        ensure(p.is_absolute(), "absolute tool path")?;
        dirs.insert(
            p.parent()
                .ok_or(ProvenanceError("tool parent"))?
                .as_os_str()
                .as_bytes()
                .to_vec(),
        );
    }
    let expected_path = dirs.into_iter().collect::<Vec<_>>().join(&b':');
    ensure(env.get("PATH")? == expected_path, "explicit tool-only PATH")?;
    let source = Manifest::decode(&r.files[0])?;
    let dependency = Manifest::decode(&r.files[2])?;
    let registry = path(env.get("CARGO_HOME")?)?.join("registry/src");
    for p in &c.packages {
        let manifest = path(&p.manifest)?;
        ensure(
            manifest.is_absolute() && manifest.file_name() == Some(OsStr::new("Cargo.toml")),
            "package manifest path",
        )?;
        if p.tag == 0 {
            ensure(p.source.is_empty(), "workspace source")?;
            let rel = manifest
                .strip_prefix(&root)
                .map_err(|_| ProvenanceError("outside workspace package"))?;
            ensure(
                source.records.iter().any(|(k, v)| {
                    k == rel.as_os_str().as_bytes() && matches!(v, FileValue::Regular(_))
                }),
                "workspace package source uncovered",
            )?;
        } else {
            ensure(p.source.starts_with("registry+"), "unknown registry source")?;
            let rel = manifest
                .strip_prefix(&registry)
                .map_err(|_| ProvenanceError("outside registry cache"))?;
            ensure(rel.components().count() == 3, "registry package root")?;
            for key in [&manifest, &manifest.with_file_name(".cargo-checksum.json")] {
                ensure(
                    dependency.records.iter().any(|(k, v)| {
                        k == key.as_os_str().as_bytes() && matches!(v, FileValue::Regular(_))
                    }),
                    "registry package source uncovered",
                )?;
            }
        }
    }
    let selected = r
        .emitted
        .iter()
        .filter(|a| {
            a.name == "study_reference"
                && a.kinds == ["bin"]
                && a.executable.as_ref() == Some(&r.executable)
        })
        .collect::<Vec<_>>();
    ensure(
        selected.len() == 1
            && selected[0].manifest
                == root
                    .join("crates/ckks-train-bench/Cargo.toml")
                    .as_os_str()
                    .as_bytes()
            && selected[0].src
                == root
                    .join("crates/ckks-train-bench/src/bin/study_reference.rs")
                    .as_os_str()
                    .as_bytes(),
        "unique writer binary linkage",
    )?;
    for a in &r.emitted {
        let p = c
            .packages
            .iter()
            .find(|p| p.id == a.id)
            .ok_or(ProvenanceError("unknown emitted package"))?;
        ensure(
            p.manifest == a.manifest
                && path(&a.src)?.starts_with(
                    path(&p.manifest)?
                        .parent()
                        .ok_or(ProvenanceError("manifest parent"))?,
                ),
            "emitted source linkage",
        )?;
        if let Some(e) = &a.executable {
            ensure(path(e)?.starts_with(&target), "emitted executable escape")?;
        }
    }
    ensure(
        path(&r.executable)? == target.join(TARGET).join("release/study_reference"),
        "receipt executable path",
    )
}
fn decode_pinned_approval(bytes: &[u8], expected: &str) -> Result<Approval> {
    ensure(
        hash_valid(expected) && sha256_hex(bytes) == expected,
        "approval pin mismatch",
    )?;
    Approval::decode(bytes)
}
fn check_approved_receipt(a: &Approval, bytes: &[u8], r: &Receipt, c: &Capsule) -> Result<()> {
    ensure(
        c.recipe == a.recipe
            && c.algorithm == a.algorithm
            && c.commit == a.commit
            && c.manifests[0] == a.source,
        "approved source/version mismatch",
    )?;
    ensure(
        a.builds.contains(&(
            sha256_hex(bytes),
            sha256_hex(&r.capsule),
            r.executable_hash.clone(),
        )),
        "receipt/capsule/executable not approved",
    )
}
fn check_current_binary(r: &Receipt, embedded: &[u8], path: &[u8], hash: &str) -> Result<()> {
    ensure(
        r.capsule == embedded && r.executable == path && r.executable_hash == hash,
        "current executable/capsule/receipt mismatch",
    )
}
impl VerifiedStudyContext {
    /// Load exact pinned local approval and captured build receipts.
    /// Ordinary builds and unsupported production platforms fail closed.
    pub fn load(request: ContextRequest) -> Result<Self> {
        let embedded =
            EMBEDDED_CAPSULE.ok_or(ProvenanceError("build has no production compile capsule"))?;
        ensure(
            cfg!(target_os = "linux"),
            "production context requires native Linux",
        )?;
        ensure(
            hash_valid(&request.approval_sha256)
                && !request.receipts.is_empty()
                && request.receipts.len() <= 64,
            "context request bounds",
        )?;
        canonical(&request.root)?;
        canonical(&request.approval)?;
        let approval_bytes = read_path(&request.approval, LIMIT)?;
        let approval = decode_pinned_approval(&approval_bytes, &request.approval_sha256)?;
        ensure(
            approval.recipe == RECIPE && approval.algorithm == ALGORITHM && !approval.id.is_empty(),
            "approval policy",
        )?;
        ensure(
            approval.designs.iter().map(|d| d.0.as_str()).eq(DESIGNS),
            "approved design set",
        )?;
        for (name, h) in &approval.designs {
            ensure(
                sha256_hex(&read_path(&request.root.join(name), LIMIT)?) == *h,
                "design bytes changed",
            )?;
        }
        let mut receipts = Vec::new();
        for p in &request.receipts {
            canonical(p)?;
            let bytes = read_path(p, LIMIT)?;
            let hash = sha256_hex(&bytes);
            ensure(
                !receipts.iter().any(|x: &FrozenReceipt| x.hash == hash),
                "duplicate receipt",
            )?;
            let receipt = Receipt::decode(&bytes)?;
            let capsule = receipt.policy()?;
            ensure(
                capsule.commit == approval.commit
                    && capsule.manifests[0] == approval.source
                    && capsule.cwd == request.root.as_os_str().as_bytes(),
                "approved source mismatch",
            )?;
            check_approved_receipt(&approval, &bytes, &receipt, &capsule)?;
            let source = Manifest::decode(&receipt.files[0])?;
            for (name, h) in &approval.designs {
                ensure(
                    source
                        .records
                        .contains(&(name.as_bytes().to_vec(), FileValue::Regular(h.clone()))),
                    "design missing from source inventory",
                )?;
            }
            ensure(
                !source
                    .records
                    .iter()
                    .any(|(p, _)| request.root.join(OsStr::from_bytes(p)) == request.approval),
                "recursive approval inventory",
            )?;
            validate_linkage(
                &receipt,
                &capsule,
                &Environment::decode(&receipt.environment)?,
            )?;
            receipts.push(FrozenReceipt {
                path: p.clone(),
                bytes,
                receipt,
                capsule,
                hash,
            });
        }
        let executable = std::env::current_exe()?;
        canonical(&executable)?;
        let candidates = receipts
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                r.receipt.capsule == embedded
                    && r.receipt.executable == executable.as_os_str().as_bytes()
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ensure(
            candidates.len() == 1,
            "current executable/capsule receipt mismatch",
        )?;
        let current = candidates[0];
        let env = Environment::decode(&receipts[current].receipt.environment)?;
        let python = path(env.get("HEKIT_STUDY_PYTHON")?)?;
        let git = path(flag(&receipts[current].receipt.argv, b"--git")?)?;
        canonical(&python)?;
        canonical(&git)?;
        let m = std::fs::metadata(&executable)?;
        let result = Self {
            request,
            approval_bytes,
            approval,
            receipts,
            current,
            python,
            git,
            executable,
            executable_identity: (m.dev(), m.ino()),
            env,
        };
        result.recheck()?;
        Ok(result)
    }
    /// Exact approval bytes SHA256 chosen by the local caller.
    pub fn approval_sha256(&self) -> &str {
        &self.request.approval_sha256
    }
    /// Exact approved compile-source manifest SHA256.
    pub fn source_manifest_sha256(&self) -> &str {
        &self.approval.source
    }
    pub(crate) fn approved_design_documents(&self) -> &[(String, String)] {
        &self.approval.designs
    }
    pub(crate) fn root(&self) -> &Path {
        &self.request.root
    }
    pub(crate) fn current_receipt_sha256(&self) -> &str {
        &self.receipts[self.current].hash
    }
    pub(crate) fn receipt_for_writer(&self, hash: &str) -> Result<&[u8]> {
        self.receipts
            .iter()
            .find(|r| r.hash == hash)
            .map(|r| r.bytes.as_slice())
            .ok_or(ProvenanceError("writer receipt unavailable in context"))
    }
    fn helper(&self) -> Result<PathBuf> {
        let helper = self.request.root.join("scripts/study_reference_build.py");
        let support = self
            .request
            .root
            .join("scripts/study_reference_build_support.py");
        let m = Manifest::decode(&self.receipts[self.current].receipt.files[0])?;
        for p in [&helper, &support] {
            let rel = p
                .strip_prefix(&self.request.root)
                .map_err(|_| ProvenanceError("helper root"))?;
            let bytes = read_path(p, LIMIT)?;
            ensure(
                m.records.contains(&(
                    rel.as_os_str().as_bytes().to_vec(),
                    FileValue::Regular(sha256_hex(&bytes)),
                )),
                "runtime helper source changed",
            )?;
        }
        Ok(helper)
    }
    fn command(&self) -> Result<Command> {
        let mut cmd = Command::new(&self.python);
        cmd.arg("-B")
            .arg(self.helper()?)
            .current_dir(&self.request.root)
            .env_clear()
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .stdout(Stdio::piped());
        for (k, v) in &self.env.0 {
            if let Some(v) = v {
                cmd.env(k, OsStr::from_bytes(v));
            }
        }
        Ok(cmd)
    }
    fn hash_file(&self, p: &Path) -> Result<String> {
        let mut cmd = self.command()?;
        cmd.arg("hash-file").arg("--path").arg(p);
        let bytes = bounded_command(cmd, 65)?;
        let s = std::str::from_utf8(&bytes)
            .map_err(|_| ProvenanceError("hash output UTF8"))?
            .trim_end_matches('\n');
        ensure(hash_valid(s), "hash output")?;
        Ok(s.into())
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        reject_runtime_keys(std::env::vars_os().map(|x| x.0))?;
        ensure(
            read_path(&self.request.approval, LIMIT)? == self.approval_bytes,
            "approval changed",
        )?;
        for r in &self.receipts {
            ensure(read_path(&r.path, LIMIT)? == r.bytes, "receipt changed")?;
        }
        let m = std::fs::symlink_metadata(&self.executable)?;
        ensure(
            m.is_file()
                && (m.dev(), m.ino()) == self.executable_identity
                && std::env::current_exe()? == self.executable,
            "current executable identity changed",
        )?;
        check_current_binary(
            &self.receipts[self.current].receipt,
            EMBEDDED_CAPSULE.ok_or(ProvenanceError("embedded capsule unavailable"))?,
            self.executable.as_os_str().as_bytes(),
            &self.hash_file(&self.executable)?,
        )?;
        let mut cmd = self.command()?;
        cmd.arg("snapshot")
            .arg("--root")
            .arg(&self.request.root)
            .arg("--receipt")
            .arg(&self.receipts[self.current].path)
            .arg("--git")
            .arg(&self.git);
        ensure(
            bounded_command(cmd, 65)? == format!("{}\n", self.approval.source).as_bytes(),
            "source/config/Git recheck",
        )?;
        Ok(())
    }
    pub(crate) fn capture(&self, p: &MaterializedProfile) -> Result<RuntimeCapture> {
        self.recheck()?;
        ensure(
            p.identity().canonical_prepared(),
            "canonical input required",
        )?;
        let (contract, receipt) =
            derive_identities(p).map_err(|_| ProvenanceError("canonical identity unavailable"))?;
        self.check_inputs(p)?;
        let cwd = std::env::current_dir()?;
        canonical(&cwd)?;
        ensure(cwd == self.request.root, "runtime cwd/source root mismatch")?;
        Ok(RuntimeCapture {
            contract_hash: contract.digest().into(),
            input_receipt_hash: receipt.digest().into(),
            materialized_hash: p.identity().digest().into(),
            argv: argv(std::env::args_os())?,
            cwd: cwd.as_os_str().as_bytes().to_vec(),
            thread_environment: thread_environment(|k| std::env::var_os(k))?,
            host: Host::observe(),
            executable_start: self.receipts[self.current].receipt.executable_hash.clone(),
            receipt_file: self.current_receipt_sha256().into(),
            source: self.approval.source.clone(),
        })
    }
    fn check_inputs(&self, p: &MaterializedProfile) -> Result<()> {
        for source in p.identity().sources() {
            ensure(
                relative(source.path.as_bytes()) && source.expected_sha256 == source.actual_sha256,
                "input source binding",
            )?;
            let path = self.request.root.join(&source.path);
            canonical(&path)?;
            ensure(
                self.hash_file(&path)? == source.actual_sha256,
                "selected input source changed",
            )?;
        }
        Ok(())
    }
    pub(crate) fn finish_capture(
        &self,
        p: &MaterializedProfile,
        start: &RuntimeCapture,
    ) -> Result<RuntimeRecheck> {
        self.recheck()?;
        self.check_inputs(p)?;
        ensure(
            start.materialized_hash == p.identity().digest()
                && start.receipt_file == self.current_receipt_sha256()
                && start.source == self.approval.source,
            "capture binding changed",
        )?;
        Ok(RuntimeRecheck {
            source: self.approval.source.clone(),
            receipt_file: self.current_receipt_sha256().into(),
            executable_end: self.receipts[self.current].receipt.executable_hash.clone(),
        })
    }
}
fn bounded_command(mut cmd: Command, cap: usize) -> Result<Vec<u8>> {
    let mut child = cmd.spawn()?;
    let mut bytes = Vec::new();
    let read = child
        .stdout
        .take()
        .ok_or(ProvenanceError("child stdout"))?
        .take(cap as u64 + 1)
        .read_to_end(&mut bytes);
    if read.is_err() || bytes.len() > cap {
        let _ = child.kill();
        let _ = child.wait();
        return Err(ProvenanceError("bounded child output"));
    }
    ensure(child.wait()?.success(), "provenance helper failed")?;
    Ok(bytes)
}
fn reject_runtime_keys(keys: impl Iterator<Item = OsString>) -> Result<()> {
    for key in keys {
        let k = key.as_bytes();
        ensure(
            !k.starts_with(b"GIT_")
                && !matches!(
                    k,
                    b"LD_PRELOAD"
                        | b"LD_LIBRARY_PATH"
                        | b"LD_AUDIT"
                        | b"DYLD_INSERT_LIBRARIES"
                        | b"DYLD_LIBRARY_PATH"
                        | b"DYLD_FRAMEWORK_PATH"
                ),
            "runtime loader/Git override",
        )?;
    }
    Ok(())
}
fn argv(v: impl IntoIterator<Item = OsString>) -> Result<Vec<Vec<u8>>> {
    let mut result = Vec::new();
    for x in v {
        ensure(result.len() < 256, "runtime argv count")?;
        let b = x.into_vec();
        ensure(b.len() <= 4096, "runtime argv byte cap")?;
        result.push(b);
    }
    Ok(result)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Observation<T> {
    Available(T),
    Unavailable(&'static str),
}
fn thread_environment(
    mut get: impl FnMut(&str) -> Option<OsString>,
) -> Result<Vec<(String, Observation<String>)>> {
    THREAD_KEYS
        .into_iter()
        .map(|k| {
            let value = match get(k) {
                None => Observation::Unavailable("unset"),
                Some(v) => {
                    let s = v
                        .into_string()
                        .map_err(|_| ProvenanceError("nonUTF8 thread environment"))?;
                    ensure(
                        !s.is_empty() && s.len() <= 4096,
                        "empty/oversize thread environment",
                    )?;
                    Observation::Available(s)
                }
            };
            Ok((k.into(), value))
        })
        .collect()
}
pub(crate) struct Host {
    cpu_model: Observation<String>,
    os: Observation<String>,
    hostname: Observation<String>,
    available_cpus: Observation<u64>,
    process_id: Observation<u64>,
    observed_threads: Observation<u64>,
}
impl Host {
    fn observe() -> Self {
        fn text(b: &[u8], reason: &'static str) -> Observation<String> {
            match std::str::from_utf8(b) {
                Ok(s) if !s.is_empty() && s.len() <= 4096 => Observation::Available(s.into()),
                _ => Observation::Unavailable(reason),
            }
        }
        let cpu = super::publication::read_proc(Path::new("/proc/cpuinfo"), 1024 * 1024)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|s| {
                s.lines().find_map(|l| {
                    l.strip_prefix("model name")
                        .and_then(|v| v.split_once(':'))
                        .map(|(_, v)| v.trim().to_owned())
                })
            })
            .map_or(Observation::Unavailable("cpu_model_unavailable"), |s| {
                text(s.as_bytes(), "cpu_model_unavailable")
            });
        let mut u: libc::utsname = unsafe { std::mem::zeroed() };
        let os = if unsafe { libc::uname(&mut u) } == 0 {
            let name = unsafe { std::ffi::CStr::from_ptr(u.sysname.as_ptr()) }.to_bytes();
            let release = unsafe { std::ffi::CStr::from_ptr(u.release.as_ptr()) }.to_bytes();
            let mut b = name.to_vec();
            b.push(b' ');
            b.extend_from_slice(release);
            text(&b, "uname_unavailable")
        } else {
            Observation::Unavailable("uname_unavailable")
        };
        let mut h = [0u8; 4097];
        let hostname = if unsafe { libc::gethostname(h.as_mut_ptr().cast(), h.len()) } == 0 {
            h.iter()
                .position(|&b| b == 0)
                .map_or(Observation::Unavailable("hostname_unavailable"), |n| {
                    text(&h[..n], "hostname_unavailable")
                })
        } else {
            Observation::Unavailable("hostname_unavailable")
        };
        let available_cpus = std::thread::available_parallelism()
            .ok()
            .and_then(|n| u64::try_from(n.get()).ok())
            .map_or(
                Observation::Unavailable("available_cpus_unavailable"),
                Observation::Available,
            );
        let observed_threads = std::fs::read_dir("/proc/self/task")
            .ok()
            .and_then(|mut entries| {
                entries.try_fold(0u64, |n, e| e.ok().and_then(|_| n.checked_add(1)))
            })
            .filter(|n| *n > 0)
            .map_or(
                Observation::Unavailable("observed_threads_unavailable"),
                Observation::Available,
            );
        Self {
            cpu_model: cpu,
            os,
            hostname,
            available_cpus,
            process_id: Observation::Available(u64::from(std::process::id())),
            observed_threads,
        }
    }
}
pub(crate) struct RuntimeCapture {
    contract_hash: String,
    input_receipt_hash: String,
    materialized_hash: String,
    argv: Vec<Vec<u8>>,
    cwd: Vec<u8>,
    thread_environment: Vec<(String, Observation<String>)>,
    host: Host,
    executable_start: String,
    receipt_file: String,
    source: String,
}
pub(crate) struct RuntimeRecheck {
    source: String,
    receipt_file: String,
    executable_end: String,
}
#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;

/// Crate-internal immutable views; no caller can construct provenance from claims.
pub(crate) struct BuildFacts<'a> {
    capsule: &'a Capsule,
    receipt: &'a FrozenReceipt,
}
impl VerifiedStudyContext {
    pub(crate) fn build_facts(&self, receipt_sha256: &str) -> Result<BuildFacts<'_>> {
        let receipt = self
            .receipts
            .iter()
            .find(|r| r.hash == receipt_sha256)
            .ok_or(ProvenanceError("writer receipt not available"))?;
        Ok(BuildFacts {
            capsule: &receipt.capsule,
            receipt,
        })
    }
    pub(crate) fn source_files(&self) -> Result<Vec<(String, String)>> {
        Manifest::decode(&self.receipts[self.current].receipt.files[0])?
            .records
            .into_iter()
            .map(|(p, v)| {
                let name = String::from_utf8(p).map_err(|_| ProvenanceError("source UTF8"))?;
                match v {
                    FileValue::Regular(h) => Ok((name, h)),
                    _ => Err(ProvenanceError("source regular required")),
                }
            })
            .collect()
    }
    pub(crate) fn source_commit(&self) -> &str {
        &self.approval.commit
    }
}
impl BuildFacts<'_> {
    pub(crate) fn compiler_vv(&self) -> &str {
        &self.capsule.compiler
    }
    pub(crate) fn target(&self) -> &str {
        &self.capsule.target
    }
    pub(crate) fn host(&self) -> &str {
        &self.capsule.host
    }
    pub(crate) fn profile(&self) -> &str {
        &self.capsule.profile
    }
    pub(crate) fn opt_level(&self) -> &str {
        &self.capsule.opt
    }
    pub(crate) fn debug(&self) -> bool {
        self.capsule.debug == 1
    }
    pub(crate) fn features(&self) -> &[String] {
        &self.capsule.features
    }
    pub(crate) fn package_version(&self) -> &str {
        &self.capsule.version
    }
    pub(crate) fn cargo_lock_sha256(&self) -> &str {
        &self.capsule.lock
    }
    pub(crate) fn source_manifest_sha256(&self) -> &str {
        &self.capsule.manifests[0]
    }
    pub(crate) fn config_manifest_sha256(&self) -> &str {
        &self.capsule.manifests[1]
    }
    pub(crate) fn build_environment_sha256(&self) -> &str {
        &self.capsule.environment
    }
    pub(crate) fn recipe_version(&self) -> &str {
        &self.capsule.recipe
    }
    pub(crate) fn receipt_file_sha256(&self) -> &str {
        &self.receipt.hash
    }
    pub(crate) fn capsule_sha256(&self) -> String {
        sha256_hex(&self.receipt.receipt.capsule)
    }
    pub(crate) fn executable_sha256(&self) -> &str {
        &self.receipt.receipt.executable_hash
    }
    pub(crate) fn executable_path(&self) -> &[u8] {
        &self.receipt.receipt.executable
    }
}
impl RuntimeCapture {
    pub(crate) fn argv(&self) -> &[Vec<u8>] {
        &self.argv
    }
    pub(crate) fn cwd(&self) -> &[u8] {
        &self.cwd
    }
    pub(crate) fn contract_sha256(&self) -> &str {
        &self.contract_hash
    }
    pub(crate) fn input_receipt_sha256(&self) -> &str {
        &self.input_receipt_hash
    }
    pub(crate) fn materialized_sha256(&self) -> &str {
        &self.materialized_hash
    }
    pub(crate) fn thread_environment(&self) -> &[(String, Observation<String>)] {
        &self.thread_environment
    }
    pub(crate) fn host(&self) -> &Host {
        &self.host
    }
    pub(crate) fn executable_start_sha256(&self) -> &str {
        &self.executable_start
    }
    pub(crate) fn receipt_file_sha256(&self) -> &str {
        &self.receipt_file
    }
}
impl RuntimeRecheck {
    pub(crate) fn source_manifest_sha256(&self) -> &str {
        &self.source
    }
    pub(crate) fn receipt_file_sha256(&self) -> &str {
        &self.receipt_file
    }
    pub(crate) fn executable_end_sha256(&self) -> &str {
        &self.executable_end
    }
}
impl Host {
    pub(crate) fn cpu_model(&self) -> &Observation<String> {
        &self.cpu_model
    }
    pub(crate) fn os(&self) -> &Observation<String> {
        &self.os
    }
    pub(crate) fn hostname(&self) -> &Observation<String> {
        &self.hostname
    }
    pub(crate) fn available_cpus(&self) -> &Observation<u64> {
        &self.available_cpus
    }
    pub(crate) fn process_id(&self) -> &Observation<u64> {
        &self.process_id
    }
    pub(crate) fn observed_threads(&self) -> &Observation<u64> {
        &self.observed_threads
    }
    pub(crate) fn execution_threads(&self) -> u64 {
        1
    }
}
