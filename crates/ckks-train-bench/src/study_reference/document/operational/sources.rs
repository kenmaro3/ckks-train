//! Strict finite independently reviewed observed sources, using existing field primitives.
use super::*;
use crate::study_reference::document::resource;
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MountIdentity {
    pub device: u64,
    pub mount_id: u64,
    pub fs_type: u64,
}
impl Decode for MountIdentity {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(r, &["device", "mount_id", "type"], p)?;
        Ok(Self {
            device: f.field("device", p)?,
            mount_id: f.field("mount_id", p)?,
            fs_type: f.field("type", p)?,
        })
    }
}
impl MountIdentity {
    fn key(&self) -> (u64, u64, u64) {
        (self.device, self.mount_id, self.fs_type)
    }
}
pub(crate) fn filesystem_account(raw: &[u8]) -> Result<MountIdentity> {
    Ok(parse_filesystems(raw, false, false)?.account)
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FilesystemFacts {
    pub identity: MountIdentity,
    pub roots: Vec<u8>,
    pub mount_evidence: FileReference,
    pub quota_evidence: FileReference,
    pub backing_evidence: FileReference,
}
fn reference(r: resource::FileRef) -> FileReference {
    FileReference {
        root: r.root.0,
        path: r.path.0,
        length: r.length,
        hash: r.sha256.0,
    }
}
impl Decode for FilesystemFacts {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            r,
            &[
                "device",
                "mount_id",
                "type",
                "roots",
                "mount_evidence",
                "quota_evidence",
                "backing_evidence",
            ],
            p,
        )?;
        Ok(Self {
            identity: MountIdentity {
                device: f.field("device", p)?,
                mount_id: f.field("mount_id", p)?,
                fs_type: f.field("type", p)?,
            },
            roots: f
                .field::<List<resource::Tag<5>, 6>>("roots", p)?
                .0
                .into_iter()
                .map(|v| v.0)
                .collect(),
            mount_evidence: reference(f.field("mount_evidence", p)?),
            quota_evidence: reference(f.field("quota_evidence", p)?),
            backing_evidence: reference(f.field("backing_evidence", p)?),
        })
    }
}
#[derive(Debug, PartialEq, Eq)]
struct Filesystems {
    physical: String,
    boot: String,
    source: String,
    revision: String,
    executable: String,
    account: MountIdentity,
    rows: Vec<FilesystemFacts>,
    review: Option<FileReference>,
}
fn file_pin(file: &FileReference, bytes: &[u8], fixed: &str) -> Result<()> {
    ensure(
        file.root == 3
            && file.path == fixed
            && file.length == bytes.len() as u64
            && file.hash == sha256_hex(bytes),
        DocumentErrorCode::CrossLinkMismatch,
        "observed.source.pin",
    )
}
fn parse_filesystems(bytes: &[u8], review: bool, oracle: bool) -> Result<Filesystems> {
    ensure(
        !oracle || cfg!(test),
        DocumentErrorCode::UnsupportedSchema,
        "source.oracle",
    )?;
    preflight_layout(bytes, DocumentLayout::Resource)?;
    let raw: &RawValue =
        serde_json::from_slice(bytes).map_err(|e| json_err(e, "observed.filesystems"))?;
    const SOURCE: &[&str] = &[
        "schema",
        "synthetic",
        "physical_host",
        "boot_id",
        "source_manifest",
        "source_revision",
        "executable",
        "account_filesystem",
        "filesystems",
        "review",
    ];
    const REVIEW: &[&str] = &[
        "schema",
        "synthetic",
        "physical_host",
        "boot_id",
        "source_manifest",
        "source_revision",
        "executable",
        "account_filesystem",
        "filesystems",
    ];
    let mut f = Fields::new(
        raw,
        if review { REVIEW } else { SOURCE },
        "observed.filesystems",
    )?;
    let name = if review {
        "resource-observed-filesystem-review-v1"
    } else {
        "resource-observed-filesystems-v1"
    };
    ensure(
        f.field::<Text<128>>("schema", "source")?.0
            == format!("hekit-{}-{name}", if oracle { "test" } else { "study" })
            && f.field::<bool>("synthetic", "source")? == oracle,
        DocumentErrorCode::UnsupportedSchema,
        "observed.filesystem.schema",
    )?;
    let value = Filesystems {
        physical: f.field::<Hash>("physical_host", "source")?.0,
        boot: f.field::<Text<36>>("boot_id", "source")?.0,
        source: f.field::<Hash>("source_manifest", "source")?.0,
        revision: f.field::<Text<40>>("source_revision", "source")?.0,
        executable: f.field::<Hash>("executable", "source")?.0,
        account: f.field("account_filesystem", "source")?,
        rows: f
            .field::<List<FilesystemFacts, 16>>("filesystems", "source")?
            .0,
        review: if review {
            None
        } else {
            Some(reference(f.field("review", "source")?))
        },
    };
    ensure(
        value.revision.len() == 40
            && lower_hex(&value.revision)
            && value.boot.len() == 36
            && !value.rows.is_empty(),
        DocumentErrorCode::CrossLinkMismatch,
        "filesystem.identity",
    )?;
    let mut prior = None;
    let mut account = 0;
    for row in &value.rows {
        let key = row.identity.key();
        ensure(
            row.identity.device > 0
                && row.identity.mount_id > 0
                && [0xef53, 0x58465342].contains(&row.identity.fs_type)
                && prior.is_none_or(|p| p < key),
            DocumentErrorCode::CrossLinkMismatch,
            "filesystem.sorted.supported",
        )?;
        prior = Some(key);
        if row.identity == value.account {
            account += 1;
        }
        ensure(
            !row.roots.is_empty() || row.identity == value.account,
            DocumentErrorCode::CrossLinkMismatch,
            "filesystem.empty.nonaccount",
        )?;
        ensure(
            row.roots.windows(2).all(|v| v[0] < v[1]),
            DocumentErrorCode::CrossLinkMismatch,
            "filesystem.root.order",
        )?;
    }
    ensure(
        account == 1,
        DocumentErrorCode::CrossLinkMismatch,
        "filesystem.account.coverage",
    )?;
    Ok(value)
}
pub(crate) fn validate_filesystem_sources(
    source: &[u8],
    review: &[u8],
    required_roles: &[u8],
    oracle: bool,
) -> Result<Vec<FilesystemFacts>> {
    let mut s = parse_filesystems(source, false, oracle)?;
    let r = parse_filesystems(review, true, oracle)?;
    let pin = s
        .review
        .take()
        .ok_or_else(|| err(DocumentErrorCode::MissingField, "filesystem.review"))?;
    file_pin(
        &pin,
        review,
        "resource/linux-cpu-observed-v1/filesystem-review.json",
    )?;
    ensure(
        s == r,
        DocumentErrorCode::CrossLinkMismatch,
        "filesystem.review.join",
    )?;
    let mut roles = s
        .rows
        .iter()
        .flat_map(|r| r.roots.iter().copied())
        .collect::<Vec<_>>();
    roles.sort_unstable();
    ensure(
        roles == required_roles,
        DocumentErrorCode::CrossLinkMismatch,
        "filesystem.complete.roles",
    )?;
    Ok(s.rows)
}
pub(crate) const NATIVE_CHECKS: [&str; 16] = [
    "pidfd-owned-unreaped",
    "subreaper-known-unknown",
    "lock-ofd-inheritance",
    "worker-handshake-exec",
    "python-git-direct-broker",
    "git-no-descendants",
    "parent-eof-custody",
    "nice-ten",
    "monotonic-common-origin",
    "rss-ten-second-deadline",
    "rss-term-grace-kill",
    "proc-namespace-cgroup",
    "storage-mount-quota",
    "publisher-no-clobber-fsync",
    "crash-pending-block",
    "resolved-closure",
];
pub(crate) fn filesystem_applicability(
    raw: &[u8],
    physical: &str,
    boot: &str,
    source: &str,
    revision: &str,
    executable: &str,
) -> Result<()> {
    let f = parse_filesystems(raw, false, false)?;
    ensure(
        f.physical == physical
            && f.boot == boot
            && f.source == source
            && f.revision == revision
            && f.executable == executable,
        DocumentErrorCode::CrossLinkMismatch,
        "filesystem.actual.applicability",
    )
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NativeIdentity {
    pub source: String,
    pub revision: String,
    pub executable: String,
    pub recipe: String,
    pub host: FileReference,
}
pub(crate) struct NativeFacts {
    pub identity: NativeIdentity,
    pub receipt: FileReference,
    pub checks: Vec<NativeCheck>,
}
pub(crate) struct CheckClock {
    pub wall: u64,
    pub elapsed: u64,
}
impl Decode for CheckClock {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            r,
            &[
                "wall_source",
                "unix_ns",
                "monotonic_source",
                "elapsed_since_operation_start_ns",
            ],
            p,
        )?;
        ensure(
            f.field::<u64>("wall_source", p)? == 0 && f.field::<u64>("monotonic_source", p)? == 0,
            DocumentErrorCode::UnsupportedTag,
            p,
        )?;
        Ok(Self {
            wall: f.field("unix_ns", p)?,
            elapsed: f.field("elapsed_since_operation_start_ns", p)?,
        })
    }
}
pub(super) struct Arg(pub(super) Vec<u8>);
impl Decode for Arg {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let s = Text::<8192>::decode(r, p)?.0;
        ensure(
            s.len() % 2 == 0 && lower_hex(&s),
            DocumentErrorCode::BadHex,
            p,
        )?;
        let bytes = s
            .as_bytes()
            .chunks_exact(2)
            .map(|p| {
                u8::from_str_radix(std::str::from_utf8(p).expect("ASCII hex"), 16)
                    .expect("checked hex")
            })
            .collect::<Vec<_>>();
        ensure(!bytes.contains(&0), DocumentErrorCode::OutOfRange, p)?;
        Ok(Self(bytes))
    }
}
pub(crate) struct NativeCheck {
    pub id: String,
    pub argv: Vec<Vec<u8>>,
    pub started: CheckClock,
    pub finished: CheckClock,
    pub exit: u64,
    pub refs: [FileReference; 4],
}
impl Decode for NativeCheck {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            r,
            &[
                "id",
                "argv",
                "started",
                "finished",
                "exit_code",
                "command",
                "stdout",
                "stderr",
                "observations",
            ],
            p,
        )?;
        Ok(Self {
            id: f.field::<Text<128>>("id", p)?.0,
            argv: f
                .field::<List<Arg, 256>>("argv", p)?
                .0
                .into_iter()
                .map(|v| v.0)
                .collect(),
            started: f.field("started", p)?,
            finished: f.field("finished", p)?,
            exit: f.field("exit_code", p)?,
            refs: [
                reference(f.field("command", p)?),
                reference(f.field("stdout", p)?),
                reference(f.field("stderr", p)?),
                reference(f.field("observations", p)?),
            ],
        })
    }
}
struct ReviewedCheck {
    id: String,
    refs: [FileReference; 4],
    threshold: bool,
}
impl Decode for ReviewedCheck {
    fn decode(r: &RawValue, p: &str) -> Result<Self> {
        let mut f = Fields::new(
            r,
            &[
                "id",
                "command",
                "stdout",
                "stderr",
                "observations",
                "threshold_event_injected",
            ],
            p,
        )?;
        Ok(Self {
            id: f.field::<Text<128>>("id", p)?.0,
            refs: [
                reference(f.field("command", p)?),
                reference(f.field("stdout", p)?),
                reference(f.field("stderr", p)?),
                reference(f.field("observations", p)?),
            ],
            threshold: f.field("threshold_event_injected", p)?,
        })
    }
}
fn native_identity(f: &mut Fields<'_>, review: bool, oracle: bool) -> Result<NativeIdentity> {
    let name = if review {
        "native-review"
    } else {
        "native-validation"
    };
    ensure(
        f.field::<Text<128>>("schema", "native")?.0
            == format!(
                "hekit-{}-resource-observed-{name}-v1",
                if oracle { "test" } else { "study" }
            )
            && f.field::<bool>("synthetic", "native")? == oracle,
        DocumentErrorCode::UnsupportedSchema,
        "native.schema",
    )?;
    let identity = NativeIdentity {
        source: f.field::<Hash>("source_manifest", "native")?.0,
        revision: f.field::<Text<40>>("source_revision", "native")?.0,
        executable: f.field::<Hash>("executable", "native")?.0,
        recipe: f.field::<Text<128>>("recipe", "native")?.0,
        host: reference(f.field("host_origin", "native")?),
    };
    ensure(
        identity.revision.len() == 40
            && lower_hex(&identity.revision)
            && [
                "study-train-cpu-serial-linux-clean-v2",
                "study-reference-linux-clean-v2",
            ]
            .contains(&identity.recipe.as_str())
            && identity.host.root == 3
            && identity.host.path == "resource/linux-cpu-observed-v1/host-origin.bin",
        DocumentErrorCode::CrossLinkMismatch,
        "native.identity",
    )?;
    Ok(identity)
}
pub(crate) fn validate_native_sources(
    source: &[u8],
    review: &[u8],
    executable: &[u8],
    oracle: bool,
) -> Result<NativeFacts> {
    ensure(
        !oracle || cfg!(test),
        DocumentErrorCode::UnsupportedSchema,
        "native.oracle",
    )?;
    preflight_layout(source, DocumentLayout::Resource)?;
    preflight_layout(review, DocumentLayout::Resource)?;
    let raw: &RawValue = serde_json::from_slice(source).map_err(|e| json_err(e, "native"))?;
    let rraw: &RawValue =
        serde_json::from_slice(review).map_err(|e| json_err(e, "native.review"))?;
    let mut f = Fields::new(
        raw,
        &[
            "schema",
            "synthetic",
            "source_manifest",
            "source_revision",
            "executable",
            "recipe",
            "build_receipt",
            "host_origin",
            "checks",
            "review",
        ],
        "native",
    )?;
    let mut r = Fields::new(
        rraw,
        &[
            "schema",
            "synthetic",
            "source_manifest",
            "source_revision",
            "executable",
            "recipe",
            "host_origin",
            "checks",
        ],
        "native.review",
    )?;
    let identity = native_identity(&mut f, false, oracle)?;
    let reviewed = native_identity(&mut r, true, oracle)?;
    ensure(
        identity == reviewed,
        DocumentErrorCode::CrossLinkMismatch,
        "native.review.identity",
    )?;
    file_pin(
        &reference(f.field("review", "native")?),
        review,
        "resource/linux-cpu-observed-v1/native-review.json",
    )?;
    let receipt = reference(f.field("build_receipt", "native")?);
    let checks = f.field::<List<NativeCheck, 16>>("checks", "native")?.0;
    let accepted = r
        .field::<List<ReviewedCheck, 16>>("checks", "native.review")?
        .0;
    ensure(
        checks.len() == 16 && accepted.len() == 16,
        DocumentErrorCode::CrossLinkMismatch,
        "native.check.count",
    )?;
    for ((check, review), id) in checks.iter().zip(&accepted).zip(NATIVE_CHECKS) {
        ensure(
            check.id == id
                && review.id == id
                && check.refs == review.refs
                && review.threshold == (id == "rss-term-grace-kill")
                && check.exit == 0
                && check.started.elapsed <= check.finished.elapsed
                && check.started.wall <= check.finished.wall,
            DocumentErrorCode::CrossLinkMismatch,
            "native.review.check",
        )?;
        ensure(
            check.argv.len() == 7
                && check.argv[0] == executable
                && check.argv[1] == b"--native-supervisor-check"
                && check.argv[2] == id.as_bytes()
                && check.argv[3] == b"--probe-origin"
                && crate::study_provenance::resource_operational::approval::absolute(
                    &check.argv[4],
                )
                && check.argv[5] == b"--probe-origin-sha256"
                && check.argv[6].len() == 64
                && check.argv[6]
                    .iter()
                    .all(|v| v.is_ascii_digit() || (b'a'..=b'f').contains(v)),
            DocumentErrorCode::CrossLinkMismatch,
            "native.check.argv",
        )?;
    }
    Ok(NativeFacts {
        identity,
        receipt,
        checks,
    })
}
