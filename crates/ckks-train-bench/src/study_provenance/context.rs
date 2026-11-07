//! Pinned local trust input, immutable captured facts and exact rechecks.
use super::role::{ExecutableRole, Recipe};
use super::{EMBEDDED_CAPSULE, ProvenanceError, Result, codec::*, ensure, publication::read_path};
use crate::study_runtime::CaseRuntimeLease;
use crate::{
    data::sha256_hex,
    study_inputs::{MaterializedProfile, SourceDigest},
    study_reference::derive_identities,
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
struct CheckedBuildRole(Recipe);
struct FrozenReceipt {
    role: CheckedBuildRole,
    path: PathBuf,
    bytes: Vec<u8>,
    receipt: Receipt,
    capsule: Capsule,
    hash: String,
}
/// Approved local source/build context. This does not establish completed training.
pub struct VerifiedStudyContext {
    core: VerifiedContextCore,
}
struct VerifiedContextCore {
    runtime: CurrentRuntime,
    request: ContextRequest,
    approval_bytes: Vec<u8>,
    approval: LocalApproval,
    receipts: Vec<FrozenReceipt>,
    current: usize,
    python: PathBuf,
    git: PathBuf,
    executable: PathBuf,
    executable_identity: (u64, u64),
    env: Environment,
}
// One core owns the sole lease. Neither context can cross an owner frame.
enum CurrentRuntime {
    Reference,
    Train(CaseRuntimeLease),
}
impl CurrentRuntime {
    fn recheck(&self) -> Result<()> {
        match self {
            Self::Reference => Ok(()),
            Self::Train(runtime) => runtime.recheck(),
        }
    }
    fn recipe(&self) -> Recipe {
        match self {
            Self::Reference => Recipe::Reference,
            Self::Train(runtime) => runtime.recipe(),
        }
    }
}
/// Train context with its exact artifact context and privately retained runtime.
/// No production context can be moved to another thread or shared with nested tasks.
/// ```compile_fail
/// use ckks_train_bench::study_provenance::VerifiedStudyTrainContext;
/// fn needs_send<T: Send>() {}
/// needs_send::<VerifiedStudyTrainContext>();
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_provenance::VerifiedStudyTrainContext;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<VerifiedStudyTrainContext>();
/// ```
/// Reference contexts deliberately share the same non-Send core:
/// ```compile_fail
/// use ckks_train_bench::study_provenance::VerifiedStudyContext;
/// fn needs_send<T: Send>() {}
/// needs_send::<VerifiedStudyContext>();
/// ```
/// A named type does not provide an authority constructor:
/// ```compile_fail
/// use ckks_train_bench::study_provenance::VerifiedStudyTrainContext;
/// let forged = VerifiedStudyTrainContext::default();
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_provenance::VerifiedStudyContext;
/// fn needs_sync<T: Sync>() {}
/// needs_sync::<VerifiedStudyContext>();
/// ```
/// The private dispatcher is not an external arbitrary-closure execution API:
/// ```compile_fail
/// use ckks_train_bench::study_runtime::dispatch;
/// ```
/// Only a borrowed artifact view is exposed:
/// ```
/// use ckks_train_bench::study_provenance::{VerifiedStudyContext, VerifiedStudyTrainContext};
/// fn artifact(ctx: &VerifiedStudyTrainContext) -> &VerifiedStudyContext {
///     ctx.artifact_context()
/// }
/// ```
pub struct VerifiedStudyTrainContext {
    artifact: VerifiedStudyContext,
}
impl VerifiedStudyTrainContext {
    pub(crate) fn load(request: crate::study_runtime::StudyTrainContextRequest) -> Result<Self> {
        let (locations, runtime) = request.into_parts();
        Ok(Self {
            artifact: VerifiedStudyContext::load_current(
                locations,
                CurrentRuntime::Train(runtime),
            )?,
        })
    }
    /// Borrow the exact owned artifact context; reads still recheck its retained runtime.
    pub fn artifact_context(&self) -> &VerifiedStudyContext {
        &self.artifact
    }
}
fn validate_train_selection(
    approval: &LocalApproval,
    selected: Recipe,
    runtime: Recipe,
) -> Result<()> {
    ensure(
        matches!(approval.version, ApprovalVersion::V2(_)),
        "Train context requires approval v2",
    )?;
    ensure(
        selected.role() == ExecutableRole::StudyTrain && selected == runtime,
        "Train recipe/runtime mismatch",
    )
}
pub(crate) fn check_runtime_environment() -> Result<()> {
    reject_runtime_keys(std::env::vars_os().map(|x| x.0))
}
pub(crate) fn check_train_preflight(recipe: Recipe) -> Result<()> {
    check_runtime_environment()?;
    let bytes =
        EMBEDDED_CAPSULE.ok_or(ProvenanceError("build has no production compile capsule"))?;
    let capsule = Capsule::decode(bytes)?;
    ensure(
        Recipe::capsule_policy(&capsule)? == recipe && recipe.role() == ExecutableRole::StudyTrain,
        "runtime compile recipe mismatch",
    )
}

fn canonical(p: &Path) -> Result<()> {
    ensure(
        p.is_absolute() && p.canonicalize()? == p,
        "canonical absolute path required",
    )
}
fn path(b: &[u8]) -> Result<PathBuf> {
    ensure(
        !b.contains(&0)
            && b.len() <= 4096
            && !b
                .split(|c| *c == b'/')
                .any(|part| part == b"." || part == b".."),
        "Unix path bounds",
    )?;
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
fn validate_linkage(r: &Receipt, c: &Capsule, env: &Environment) -> Result<CheckedBuildRole> {
    let recipe = Recipe::capsule_policy(c)?;
    r.check_post(c)?;
    env_policy(env)?;
    validate_recipe_vectors(r, c, env, recipe)?;
    let root = path(&c.cwd)?;
    let target = path(env.get("CARGO_TARGET_DIR")?)?;
    ensure(
        root.is_absolute() && target.is_absolute() && !target.starts_with(&root),
        "recipe roots",
    )?;
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
    for rel in [
        "scripts/study_reference_build.py",
        "scripts/study_reference_build_support.py",
        recipe.helper(),
        "crates/ckks-train-bench/build.rs",
    ] {
        ensure(
            source
                .records
                .iter()
                .any(|(p, v)| p == rel.as_bytes() && matches!(v, FileValue::Regular(_))),
            "build helper source uncovered",
        )?;
    }
    let binary_source = format!("crates/ckks-train-bench/src/bin/{}.rs", recipe.bin());
    ensure(
        source
            .records
            .iter()
            .any(|(p, v)| p == binary_source.as_bytes() && matches!(v, FileValue::Regular(_))),
        "binary source uncovered",
    )?;
    for (name, option) in [
        ("rust-toolchain", b"--rust-toolchain-sha256".as_slice()),
        (
            "rust-toolchain.toml",
            b"--rust-toolchain-toml-sha256".as_slice(),
        ),
    ] {
        let expected = source.records.iter().find(|(p, _)| p == name.as_bytes());
        let pin = r.argv[2..]
            .chunks_exact(2)
            .find(|pair| pair[0] == option)
            .map(|pair| pair[1].as_slice());
        ensure(
            match (expected, pin) {
                (None, None) => true,
                (Some((_, FileValue::Regular(h))), Some(pin)) => h.as_bytes() == pin,
                _ => false,
            },
            "toolchain source pin linkage",
        )?;
    }
    for option in [b"--request".as_slice(), b"--receipt".as_slice()] {
        ensure(
            !path(flag(&r.argv, option)?)?.starts_with(&root),
            "helper output inside source",
        )?;
    }
    ensure(
        source
            .records
            .contains(&(b"Cargo.lock".to_vec(), FileValue::Regular(c.lock.clone()))),
        "Cargo.lock source linkage",
    )?;
    Ok(CheckedBuildRole(recipe))
}
fn validate_recipe_vectors(
    r: &Receipt,
    c: &Capsule,
    env: &Environment,
    recipe: Recipe,
) -> Result<()> {
    let root = path(&c.cwd)?;
    let cargo = flag(&r.argv, b"--cargo")?;
    let target = path(env.get("CARGO_TARGET_DIR")?)?;
    validate_helper_argv(r, c, recipe)?;
    let mut meta = vec![
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
    if let Some(feature) = recipe.feature() {
        meta.splice(
            7..7,
            [
                b"--features".to_vec(),
                format!("ckks-train-bench/{feature}").into_bytes(),
            ],
        );
    }
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
        recipe.bin(),
        "--target",
        TARGET,
        "--target-dir",
    ] {
        build.push(x.as_bytes().to_vec());
    }
    if let Some(feature) = recipe.feature() {
        build.splice(8..8, [b"--features".to_vec(), feature.as_bytes().to_vec()]);
    }
    build.push(env.get("CARGO_TARGET_DIR")?.to_vec());
    build.push(b"-j".to_vec());
    build.push(env.get("CARGO_BUILD_JOBS")?.to_vec());
    build.push(b"--message-format=json-render-diagnostics".to_vec());
    ensure(c.build_argv == build, "build argv")?;
    let selected = r
        .emitted
        .iter()
        .filter(|a| {
            a.name == recipe.bin()
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
                    .join(format!(
                        "crates/ckks-train-bench/src/bin/{}.rs",
                        recipe.bin()
                    ))
                    .as_os_str()
                    .as_bytes(),
        "unique writer binary linkage",
    )?;
    ensure(
        path(&r.executable)? == target.join(TARGET).join("release").join(recipe.bin()),
        "receipt executable path",
    )
}
fn validate_helper_argv(r: &Receipt, c: &Capsule, recipe: Recipe) -> Result<()> {
    let root = path(&c.cwd)?;
    ensure(
        r.argv.len() >= 2
            && r.argv[0] == root.join(recipe.helper()).as_os_str().as_bytes()
            && r.argv[1] == recipe.subcommand().as_bytes(),
        "fixed helper dispatch",
    )?;
    let mut options = std::collections::BTreeMap::new();
    ensure((r.argv.len() - 2) % 2 == 0, "helper option pairs")?;
    for pair in r.argv[2..].chunks_exact(2) {
        ensure(
            options
                .insert(pair[0].as_slice(), pair[1].as_slice())
                .is_none(),
            "duplicate helper option",
        )?;
    }
    for key in [
        "root",
        "cargo",
        "rustc",
        "python",
        "git",
        "linker",
        "cargo-home",
        "target",
        "home",
        "tmp",
        "request",
        "receipt",
        "jobs",
    ] {
        ensure(
            options.remove(format!("--{key}").as_bytes()).is_some(),
            "required helper option",
        )?;
    }
    for key in ["--rust-toolchain-sha256", "--rust-toolchain-toml-sha256"] {
        if let Some(value) = options.remove(key.as_bytes()) {
            ensure(
                std::str::from_utf8(value).is_ok_and(hash_valid),
                "toolchain pin",
            )?;
        }
    }
    ensure(options.is_empty(), "unsupported helper option")?;
    for key in [
        "--root",
        "--cargo",
        "--rustc",
        "--python",
        "--git",
        "--linker",
        "--cargo-home",
        "--target",
        "--home",
        "--tmp",
        "--request",
        "--receipt",
    ] {
        ensure(
            path(flag(&r.argv, key.as_bytes())?)?.is_absolute(),
            "absolute helper path",
        )?;
    }
    Ok(())
}
fn check_design_digest(expected: &str, actual: &str) -> Result<()> {
    ensure(expected == actual, "design bytes changed")
}
fn decode_pinned_approval(bytes: &[u8], expected: &str) -> Result<LocalApproval> {
    ensure(
        hash_valid(expected) && sha256_hex(bytes) == expected,
        "approval pin mismatch",
    )?;
    LocalApproval::decode(bytes)
}
fn check_approved_receipt(a: &LocalApproval, bytes: &[u8], r: &Receipt, c: &Capsule) -> Result<()> {
    ensure(
        c.commit == a.commit && c.manifests[0] == a.source,
        "approved source/version mismatch",
    )?;
    let triple = (
        sha256_hex(bytes),
        sha256_hex(&r.capsule),
        r.executable_hash.clone(),
    );
    let found = match &a.version {
        ApprovalVersion::V1(a) => {
            c.recipe == a.recipe && c.algorithm == a.algorithm && a.builds.contains(&triple)
        }
        ApprovalVersion::V2(a) => {
            let recipe = Recipe::classify(&c.recipe, &c.algorithm)?;
            a.builds.iter().any(|b| {
                b.role == u8::from(recipe.role() == ExecutableRole::StudyTrain)
                    && b.recipe == c.recipe
                    && b.algorithm == c.algorithm
                    && (&b.receipt, &b.capsule, &b.executable) == (&triple.0, &triple.1, &triple.2)
            })
        }
    };
    ensure(found, "receipt/capsule/executable not approved")
}
fn reference_recipe(recipe: Recipe) -> bool {
    recipe.role() == ExecutableRole::Reference
}
fn reference_current_candidate(
    recipe: Recipe,
    receipt: &Receipt,
    embedded: &[u8],
    executable: &[u8],
) -> bool {
    current_candidate(Recipe::Reference, recipe, receipt, embedded, executable)
}
fn current_candidate(
    expected: Recipe,
    recipe: Recipe,
    receipt: &Receipt,
    embedded: &[u8],
    executable: &[u8],
) -> bool {
    recipe == expected && receipt.capsule == embedded && receipt.executable == executable
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
        Self::load_current(request, CurrentRuntime::Reference)
    }
    fn load_current(request: ContextRequest, runtime: CurrentRuntime) -> Result<Self> {
        runtime.recheck()?;
        let embedded =
            EMBEDDED_CAPSULE.ok_or(ProvenanceError("build has no production compile capsule"))?;
        ensure(
            cfg!(target_os = "linux"),
            "production context requires native Linux",
        )?;
        check_runtime_environment()?;
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
        approval.policy(&DESIGNS)?;
        if let CurrentRuntime::Train(lease) = &runtime {
            validate_train_selection(
                &approval,
                Recipe::capsule_policy(&Capsule::decode(embedded)?)?,
                lease.recipe(),
            )?;
        }
        for (name, h) in &approval.designs {
            check_design_digest(h, &sha256_hex(&read_path(&request.root.join(name), LIMIT)?))?;
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
            let capsule = receipt.closed_policy()?;
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
            let role = validate_linkage(
                &receipt,
                &capsule,
                &Environment::decode(&receipt.environment)?,
            )?;
            receipts.push(FrozenReceipt {
                role,
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
            .filter(|(_, r)| match runtime.recipe() {
                Recipe::Reference => reference_current_candidate(
                    r.role.0,
                    &r.receipt,
                    embedded,
                    executable.as_os_str().as_bytes(),
                ),
                expected => current_candidate(
                    expected,
                    r.role.0,
                    &r.receipt,
                    embedded,
                    executable.as_os_str().as_bytes(),
                ),
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
            core: VerifiedContextCore {
                runtime,
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
            },
        };
        result.recheck()?;
        Ok(result)
    }
    /// Exact approval bytes SHA256 chosen by the local caller.
    pub fn approval_sha256(&self) -> &str {
        &self.core.request.approval_sha256
    }
    /// Exact approved compile-source manifest SHA256.
    pub fn source_manifest_sha256(&self) -> &str {
        &self.core.approval.source
    }
    pub(crate) fn approved_design_documents(&self) -> &[(String, String)] {
        &self.core.approval.designs
    }
    pub(crate) fn root(&self) -> &Path {
        &self.core.request.root
    }
    pub(crate) fn current_receipt_sha256(&self) -> &str {
        &self.core.receipts[self.core.current].hash
    }
    pub(crate) fn receipt_for_writer(&self, hash: &str) -> Result<&[u8]> {
        self.core
            .receipts
            .iter()
            .find(|r| r.hash == hash && reference_recipe(r.role.0))
            .map(|r| r.bytes.as_slice())
            .ok_or(ProvenanceError("writer receipt unavailable in context"))
    }
    fn helper(&self) -> Result<PathBuf> {
        let helper = self
            .core
            .request
            .root
            .join("scripts/study_reference_build.py");
        let support = self
            .core
            .request
            .root
            .join("scripts/study_reference_build_support.py");
        let m = Manifest::decode(&self.core.receipts[self.core.current].receipt.files[0])?;
        let role_helper = self
            .core
            .request
            .root
            .join(self.core.receipts[self.core.current].role.0.helper());
        for p in [&helper, &support, &role_helper] {
            let rel = p
                .strip_prefix(&self.core.request.root)
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
        let mut cmd = Command::new(&self.core.python);
        cmd.arg("-B")
            .arg(self.helper()?)
            .current_dir(&self.core.request.root)
            .env_clear()
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .stdout(Stdio::piped());
        for (k, v) in &self.core.env.0 {
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
        self.core.runtime.recheck()?;
        check_runtime_environment()?;
        ensure(
            read_path(&self.core.request.approval, LIMIT)? == self.core.approval_bytes,
            "approval changed",
        )?;
        for r in &self.core.receipts {
            ensure(read_path(&r.path, LIMIT)? == r.bytes, "receipt changed")?;
        }
        let m = std::fs::symlink_metadata(&self.core.executable)?;
        ensure(
            m.is_file()
                && (m.dev(), m.ino()) == self.core.executable_identity
                && std::env::current_exe()? == self.core.executable,
            "current executable identity changed",
        )?;
        check_current_binary(
            &self.core.receipts[self.core.current].receipt,
            EMBEDDED_CAPSULE.ok_or(ProvenanceError("embedded capsule unavailable"))?,
            self.core.executable.as_os_str().as_bytes(),
            &self.hash_file(&self.core.executable)?,
        )?;
        let mut cmd = self.command()?;
        cmd.arg("snapshot")
            .arg("--root")
            .arg(&self.core.request.root)
            .arg("--receipt")
            .arg(&self.core.receipts[self.core.current].path)
            .arg("--git")
            .arg(&self.core.git);
        ensure(
            bounded_command(cmd, 65)? == format!("{}\n", self.core.approval.source).as_bytes(),
            "source/config/Git recheck",
        )?;
        self.core.runtime.recheck()
    }
    pub(crate) fn capture(&self, p: &MaterializedProfile) -> Result<RuntimeCapture> {
        self.require_current_reference()?;
        let checked = self.check_materialized_input_now(p)?;
        let cwd = std::env::current_dir()?;
        canonical(&cwd)?;
        ensure(
            cwd == self.core.request.root,
            "runtime cwd/source root mismatch",
        )?;
        Ok(RuntimeCapture {
            contract_hash: checked.facts.contract_hash,
            input_receipt_hash: checked.facts.input_receipt_hash,
            materialized_hash: checked.facts.materialized_hash,
            argv: argv(std::env::args_os())?,
            cwd: cwd.as_os_str().as_bytes().to_vec(),
            thread_environment: thread_environment(|k| std::env::var_os(k))?,
            host: Host::observe(),
            executable_start: self.current_build_facts().executable_sha256().into(),
            receipt_file: self.current_receipt_sha256().into(),
            source: self.core.approval.source.clone(),
        })
    }
    /// Read-only selected-input validation, independent of Reference capture authority.
    pub(crate) fn check_materialized_input_now(
        &self,
        p: &MaterializedProfile,
    ) -> Result<CheckedMaterializedInput<'_>> {
        self.recheck()?;
        let facts = checked_materialized_facts(
            p,
            self.current_receipt_sha256(),
            &self.core.approval.source,
            |p| self.check_inputs(p),
        )?;
        Ok(CheckedMaterializedInput {
            context: self,
            facts,
        })
    }
    pub(crate) fn recheck_materialized_input_now(
        &self,
        p: &MaterializedProfile,
        start: &CheckedMaterializedInput<'_>,
    ) -> Result<()> {
        require_same_context(self, start.context)?;
        let current = self.check_materialized_input_now(p)?;
        ensure(
            current.facts == start.facts,
            "selected input binding changed",
        )
    }
    fn check_inputs(&self, p: &MaterializedProfile) -> Result<()> {
        for source in p.identity().sources() {
            check_input_source(&self.core.request.root, source, |path| self.hash_file(path))?;
        }
        Ok(())
    }
    pub(crate) fn finish_capture(
        &self,
        p: &MaterializedProfile,
        start: &RuntimeCapture,
    ) -> Result<RuntimeRecheck> {
        self.require_current_reference()?;
        let checked = self.check_materialized_input_now(p)?;
        ensure(
            start.contract_hash == checked.facts.contract_hash
                && start.input_receipt_hash == checked.facts.input_receipt_hash
                && start.materialized_hash == checked.facts.materialized_hash
                && start.receipt_file == checked.facts.receipt_file
                && start.source == checked.facts.source,
            "capture binding changed",
        )?;
        Ok(RuntimeRecheck {
            source: self.core.approval.source.clone(),
            receipt_file: self.current_receipt_sha256().into(),
            executable_end: self.core.receipts[self.core.current]
                .receipt
                .executable_hash
                .clone(),
        })
    }
}
// Non-authoritative inspection values: no constructor is exposed outside this module.
#[derive(Debug, PartialEq, Eq)]
struct MaterializedInputFacts {
    contract_hash: String,
    input_receipt_hash: String,
    materialized_hash: String,
    receipt_file: String,
    source: String,
}
pub(crate) struct CheckedMaterializedInput<'a> {
    context: &'a VerifiedStudyContext,
    facts: MaterializedInputFacts,
}
fn require_same_context<T>(current: &T, previous: &T) -> Result<()> {
    ensure(
        std::ptr::eq(current, previous),
        "selected input context changed",
    )
}
fn checked_materialized_facts(
    p: &MaterializedProfile,
    receipt_file: &str,
    source: &str,
    check_inputs: impl FnOnce(&MaterializedProfile) -> Result<()>,
) -> Result<MaterializedInputFacts> {
    ensure(
        p.identity().canonical_prepared(),
        "canonical input required",
    )?;
    let (contract, receipt) =
        derive_identities(p).map_err(|_| ProvenanceError("canonical identity unavailable"))?;
    check_inputs(p)?;
    Ok(MaterializedInputFacts {
        contract_hash: contract.digest().into(),
        input_receipt_hash: receipt.digest().into(),
        materialized_hash: p.identity().digest().into(),
        receipt_file: receipt_file.into(),
        source: source.into(),
    })
}
fn check_input_source(
    root: &Path,
    source: &SourceDigest,
    hash_file: impl FnOnce(&Path) -> Result<String>,
) -> Result<()> {
    ensure(
        relative(source.path.as_bytes()) && source.expected_sha256 == source.actual_sha256,
        "input source binding",
    )?;
    let path = root.join(&source.path);
    canonical(&path)?;
    ensure(
        hash_file(&path)? == source.actual_sha256,
        "selected input source changed",
    )
}
fn require_reference_role(role: ExecutableRole) -> Result<()> {
    ensure(
        role == ExecutableRole::Reference,
        "current Reference required",
    )
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
    pub(crate) fn observe() -> Self {
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
    pub(super) fn current_role(&self) -> ExecutableRole {
        self.core.receipts[self.core.current].role.0.role()
    }
    pub(crate) fn require_current_reference(&self) -> Result<()> {
        require_reference_role(self.current_role())
    }
    fn current_build_facts(&self) -> BuildFacts<'_> {
        let receipt = &self.core.receipts[self.core.current];
        BuildFacts {
            capsule: &receipt.capsule,
            receipt,
        }
    }
    pub(crate) fn reference_build_facts(&self, receipt_sha256: &str) -> Result<BuildFacts<'_>> {
        let receipt = self
            .core
            .receipts
            .iter()
            .find(|r| r.hash == receipt_sha256 && reference_recipe(r.role.0))
            .ok_or(ProvenanceError("writer receipt not available"))?;
        Ok(BuildFacts {
            capsule: &receipt.capsule,
            receipt,
        })
    }
    pub(crate) fn source_files(&self) -> Result<Vec<(String, String)>> {
        Manifest::decode(&self.core.receipts[self.core.current].receipt.files[0])?
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
        &self.core.approval.commit
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
