//! Descriptor-relative foundation. No scientific publication type is exposed.
use super::{ProvenanceError, Result, ensure};
pub(super) mod retained;
use crate::data::sha256_hex;
pub(crate) use retained::HeldRead;
#[cfg(test)]
use retained::{ReadBound, dataset_size};
use std::{
    ffi::{CStr, CString, OsStr, OsString},
    fs::File,
    io::{Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, RawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    },
    path::{Component, Path, PathBuf},
};

#[derive(Debug)]
pub(crate) enum PublishError {
    BeforeLink,
    VisibleUnconfirmed,
}
trait Ops {
    fn create(&mut self) -> Result<()>;
    fn write(&mut self, bytes: &[u8]) -> Result<()>;
    fn file_sync(&mut self) -> Result<()>;
    fn link(&mut self) -> Result<()>;
    fn dir_sync(&mut self) -> Result<()>;
}
fn publish_ops(o: &mut impl Ops, b: &[u8]) -> std::result::Result<(), PublishError> {
    o.create().map_err(|_| PublishError::BeforeLink)?;
    o.write(b)
        .and_then(|()| o.file_sync())
        .and_then(|()| o.link())
        .map_err(|_| PublishError::BeforeLink)?;
    // L returns immediately on syscall success. Everything from here is visible.
    o.dir_sync().map_err(|_| PublishError::VisibleUnconfirmed)
}
fn recover_ops(o: &mut impl Ops) -> Result<()> {
    o.file_sync()?;
    o.dir_sync()
}
fn write_complete(writer: &mut impl Write, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes)?;
    Ok(())
}
fn cstring(b: &[u8]) -> Result<CString> {
    CString::new(b).map_err(|_| ProvenanceError("path NUL"))
}
fn fd_file(fd: RawFd) -> Result<File> {
    if fd < 0 {
        Err(std::io::Error::last_os_error().into())
    } else {
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}
fn openat(parent: RawFd, name: &OsStr, flags: i32, mode: libc::mode_t) -> Result<File> {
    let c = cstring(name.as_bytes())?;
    fd_file(unsafe { libc::openat(parent, c.as_ptr(), flags, mode as libc::c_uint) })
}
fn identity(m: &std::fs::Metadata) -> (u64, u64, u64, i64, i64, i64, i64) {
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn read_file(file: &mut File, cap: usize) -> Result<Vec<u8>> {
    let before = file.metadata()?;
    ensure(before.is_file(), "nonregular read")?;
    let len = usize::try_from(before.len()).map_err(|_| ProvenanceError("file size overflow"))?;
    ensure(len <= cap, "file byte cap")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len.min(cap))
        .map_err(|_| ProvenanceError("allocation unavailable"))?;
    let bound = u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1);
    file.take(bound).read_to_end(&mut bytes)?;
    ensure(
        bytes.len() == len
            && bytes.len() <= cap
            && identity(&before) == identity(&file.metadata()?),
        "file changed/read incomplete",
    )?;
    Ok(bytes)
}
pub(super) fn read_path(path: &Path, cap: usize) -> Result<Vec<u8>> {
    // Caller-selected path, never an artifact-provided location. Parent identity
    // is checked separately by context or held by the publication store.
    let c = cstring(path.as_os_str().as_bytes())?;
    let mut f = fd_file(unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    })?;
    read_file(&mut f, cap)
}
pub(super) fn read_proc(path: &Path, cap: usize) -> Result<Vec<u8>> {
    let f = File::open(path)?;
    let mut bytes = Vec::new();
    f.take(cap as u64 + 1).read_to_end(&mut bytes)?;
    ensure(bytes.len() <= cap, "proc observation byte cap")?;
    Ok(bytes)
}
#[derive(Clone, Copy)]
struct Mount {
    device: u64,
    id: u64,
}
fn same_publication_mount(a: Mount, b: Mount) -> Result<()> {
    ensure(
        a.device == b.device && a.id == b.id,
        "publication mount mismatch",
    )
}
fn mount_type(name: &str, magic: u64) -> Result<()> {
    ensure(
        matches!((name, magic), ("ext4", 0xef53) | ("xfs", 0x58465342)),
        "unsupported publication filesystem",
    )
}
#[cfg(target_os = "linux")]
fn mount(file: &File) -> Result<Mount> {
    let fd = file.as_raw_fd();
    let info_bytes = read_proc(Path::new(&format!("/proc/self/fdinfo/{fd}")), 16384)?;
    let info = std::str::from_utf8(&info_bytes).map_err(|_| ProvenanceError("fdinfo UTF8"))?;
    ensure(info.len() <= 16384, "fdinfo cap")?;
    let ids = info
        .lines()
        .filter_map(|l| l.strip_prefix("mnt_id:"))
        .map(str::trim)
        .collect::<Vec<_>>();
    ensure(ids.len() == 1, "fd mount ID unavailable")?;
    let id = ids[0]
        .parse::<u64>()
        .map_err(|_| ProvenanceError("mount id"))?;
    let table = read_proc(Path::new("/proc/self/mountinfo"), 8 * 1024 * 1024)?;
    let text = std::str::from_utf8(&table).map_err(|_| ProvenanceError("mountinfo UTF8"))?;
    let mut names = Vec::new();
    for line in text.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.first().and_then(|x| x.parse::<u64>().ok()) == Some(id) {
            let sep = fields
                .iter()
                .position(|x| *x == "-")
                .ok_or(ProvenanceError("mountinfo separator"))?;
            names.push(
                *fields
                    .get(sep + 1)
                    .ok_or(ProvenanceError("mountinfo type"))?,
            );
        }
    }
    ensure(names.len() == 1, "mount correlation")?;
    let mut fs: libc::statfs = unsafe { std::mem::zeroed() };
    ensure(
        unsafe { libc::fstatfs(fd, &mut fs) } == 0,
        "fstatfs unavailable",
    )?;
    mount_type(names[0], fs.f_type as u64)?;
    Ok(Mount {
        device: file.metadata()?.dev(),
        id,
    })
}
#[cfg(not(target_os = "linux"))]
fn mount(_: &File) -> Result<Mount> {
    Err(ProvenanceError(
        "production publication requires Linux ext4/XFS",
    ))
}
fn effective_uid() -> u32 {
    unsafe { libc::geteuid() }
}
fn safe_directory(f: &File, private: bool) -> Result<()> {
    let m = f.metadata()?;
    ensure(m.is_dir(), "directory required")?;
    let uid = effective_uid();
    if private {
        ensure(
            m.uid() == uid && m.mode() & 0o7777 == 0o700,
            "private directory ownership/mode",
        )?;
    } else {
        ensure(
            (m.uid() == 0 || m.uid() == uid) && m.mode() & 0o022 == 0,
            "unsafe ancestor ownership/mode",
        )?;
    }
    Ok(())
}
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Reference,
    Diagnostic,
    Failure,
}
impl Kind {
    fn directory(self) -> &'static str {
        match self {
            Self::Reference => "references",
            Self::Diagnostic => "diagnostics",
            Self::Failure => "failures",
        }
    }
    fn extension(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Diagnostic => "diagnostic",
            Self::Failure => "failure",
        }
    }
}
pub(crate) struct Store {
    anchors: Anchors,
    root: File,
    parent: File,
    path: PathBuf,
    kind: Kind,
}
pub(crate) struct DurableBytes {
    file_sha256: String,
    path: PathBuf,
    recovered: bool,
}
impl DurableBytes {
    pub(crate) fn file_sha256(&self) -> &str {
        &self.file_sha256
    }
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
    pub(crate) fn temporary_cleanup_not_performed(&self) -> bool {
        self.recovered
    }
}
fn run_id(s: &str) -> Result<()> {
    ensure(
        !s.is_empty()
            && s.len() <= 128
            && s.bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "run ID",
    )
}
impl Store {
    pub(crate) fn open(
        root: &Path,
        context: &super::VerifiedStudyContext,
        kind: Kind,
    ) -> Result<Self> {
        ensure(
            !root.starts_with(context.root().join("results")),
            "historical results namespace",
        )?;
        ensure(
            cfg!(target_os = "linux"),
            "production publication requires Linux ext4/XFS",
        )?;
        let anchors = Anchors::open(root)?;
        let f = anchors.last().try_clone()?;
        let parent = openat(
            f.as_raw_fd(),
            OsStr::new(kind.directory()),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )?;
        safe_directory(&parent, true)?;
        same_publication_mount(mount(&f)?, mount(&parent)?)?;
        Ok(Self {
            anchors,
            root: f,
            parent,
            path: root.join(kind.directory()),
            kind,
        })
    }
    fn basename(&self, id: &str) -> Result<String> {
        run_id(id)?;
        Ok(format!("{id}.{}.json", self.kind.extension()))
    }
    fn recheck(&self) -> Result<()> {
        self.anchors.recheck()?;
        let current = openat(
            self.root.as_raw_fd(),
            OsStr::new(self.kind.directory()),
            DIR_FLAGS,
            0,
        )?;
        ensure(
            inode(&current)? == inode(&self.parent)?,
            "reference parent namespace changed",
        )?;
        safe_directory(&self.root, true)?;
        safe_directory(&self.parent, true)?;
        same_publication_mount(mount(&self.root)?, mount(&self.parent)?)
    }
    pub(crate) fn publish(
        &self,
        id: &str,
        bytes: &[u8],
        cap: usize,
    ) -> std::result::Result<DurableBytes, PublishError> {
        self.recheck().map_err(|_| PublishError::BeforeLink)?;
        ensure(bytes.len() <= cap, "publication byte cap").map_err(|_| PublishError::BeforeLink)?;
        let name = self.basename(id).map_err(|_| PublishError::BeforeLink)?;
        let mut o = Native::new(&self.parent, name.clone());
        publish_ops(&mut o, bytes)?;
        o.confirm_final()
            .and_then(|()| {
                #[cfg(test)]
                test_hooks::checkpoint("anchor-check")?;
                self.recheck()
            })
            .map_err(|_| PublishError::VisibleUnconfirmed)?;
        #[cfg(test)]
        test_hooks::checkpoint("release").map_err(|_| PublishError::VisibleUnconfirmed)?;
        Ok(DurableBytes {
            file_sha256: sha256_hex(bytes),
            path: self.path.join(name),
            recovered: false,
        })
    }
    pub(crate) fn read(&self, id: &str, cap: usize) -> Result<Vec<u8>> {
        self.recheck()?;
        let name = self.basename(id)?;
        let mut f = openat(
            self.parent.as_raw_fd(),
            OsStr::new(&name),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )?;
        read_file(&mut f, cap)
    }
    pub(crate) fn recover(
        &self,
        id: &str,
        expected: &str,
        cap: usize,
        verify: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<DurableBytes> {
        self.recheck()?;
        let name = self.basename(id)?;
        let mut file = openat(
            self.parent.as_raw_fd(),
            OsStr::new(&name),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )?;
        let bytes = read_file(&mut file, cap)?;
        ensure(sha256_hex(&bytes) == expected, "recovery digest mismatch")?;
        verify(&bytes)?;
        let mut o = Native::recovery(&self.parent, name.clone(), file);
        recover_ops(&mut o)?;
        Ok(DurableBytes {
            file_sha256: expected.into(),
            path: self.path.join(name),
            recovered: true,
        })
    }
}
#[path = "publication_anonymous.rs"]
mod anonymous;
#[cfg(test)]
#[path = "publication_test_hooks.rs"]
mod test_hooks;

struct Native<'a> {
    parent: &'a File,
    final_name: String,
    file: Option<File>,
    proc_fds: Option<anonymous::ProcFdDirectory>,
}
impl<'a> Native<'a> {
    fn new(parent: &'a File, final_name: String) -> Self {
        Self {
            parent,
            final_name,
            file: None,
            proc_fds: None,
        }
    }
    // Recovery uses only file_sync/dir_sync: no proc capture or O_TMPFILE gate.
    fn recovery(parent: &'a File, final_name: String, file: File) -> Self {
        Self {
            parent,
            final_name,
            file: Some(file),
            proc_fds: None,
        }
    }
    fn confirm_final(&self) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("final-check")?;
        let held = self
            .file
            .as_ref()
            .ok_or(ProvenanceError("owned file absent"))?;
        let current = openat(
            self.parent.as_raw_fd(),
            OsStr::new(&self.final_name),
            FILE_FLAGS,
            0,
        )?;
        safe_file(&current)?;
        ensure(inode(&current)? == inode(held)?, "published inode replaced")?;
        same_publication_mount(mount(self.parent)?, mount(&current)?)
    }
}
impl Ops for Native<'_> {
    fn create(&mut self) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("C")?;
        ensure(
            self.file.is_none() && self.proc_fds.is_none(),
            "publication already owns file",
        )?;
        let (file, proc_fds) = anonymous::create(self.parent)?;
        self.file = Some(file);
        self.proc_fds = Some(proc_fds);
        Ok(())
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("W")?;
        write_complete(
            self.file
                .as_mut()
                .ok_or(ProvenanceError("owned file absent"))?,
            bytes,
        )
    }
    fn file_sync(&mut self) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("F")?;
        self.file
            .as_ref()
            .ok_or(ProvenanceError("owned file absent"))?
            .sync_all()?;
        Ok(())
    }
    fn link(&mut self) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("L")?;
        let file = self
            .file
            .as_ref()
            .ok_or(ProvenanceError("owned file absent"))?;
        let proc_fds = self
            .proc_fds
            .as_ref()
            .ok_or(ProvenanceError("proc capture absent"))?;
        proc_fds.link(file, self.parent, &self.final_name)
    }
    fn dir_sync(&mut self) -> Result<()> {
        #[cfg(test)]
        test_hooks::checkpoint("D")?;
        self.parent.sync_all()?;
        Ok(())
    }
}
#[cfg(test)]
#[path = "publication_tests.rs"]
mod tests;

const DIR_FLAGS: i32 = libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC;
const FILE_FLAGS: i32 = libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC;
fn inode(f: &File) -> Result<(u64, u64)> {
    let m = f.metadata()?;
    Ok((m.dev(), m.ino()))
}
fn safe_file(f: &File) -> Result<()> {
    let m = f.metadata()?;
    ensure(
        m.is_file() && m.uid() == effective_uid() && m.mode() & 0o7777 == 0o600,
        "private regular file owner/mode",
    )
}
/// Every path component is retained; checks reopen children only relative to their held parents.
struct Anchors {
    files: Vec<File>,
    names: Vec<OsString>,
    private_from: usize,
    root_mount: Mount,
    mount_from: usize,
}
impl Anchors {
    fn open(path: &Path) -> Result<Self> {
        Self::open_mode(path, true)
    }
    fn open_mode(path: &Path, private: bool) -> Result<Self> {
        ensure(
            path.is_absolute() && path.as_os_str().as_bytes().len() <= 4096,
            "absolute bounded root",
        )?;
        let slash = cstring(b"/")?;
        let root = fd_file(unsafe { libc::open(slash.as_ptr(), DIR_FLAGS) })?;
        safe_directory(&root, false)?;
        let mut files = vec![root];
        let mut names = Vec::new();
        for component in path.components() {
            match component {
                Component::RootDir => {}
                Component::Normal(name) => {
                    let child =
                        openat(files.last().expect("root").as_raw_fd(), name, DIR_FLAGS, 0)?;
                    safe_directory(&child, false)?;
                    files.push(child);
                    names.push(name.to_os_string());
                }
                _ => return Err(ProvenanceError("noncanonical root")),
            }
        }
        let private_from = files.len() - 1;
        safe_directory(&files[private_from], private)?;
        let root_mount = mount(&files[private_from])?;
        let result = Self {
            files,
            names,
            private_from: if private { private_from } else { usize::MAX },
            root_mount,
            mount_from: private_from,
        };
        result.recheck()?;
        Ok(result)
    }
    fn duplicate(&self) -> Result<Self> {
        Ok(Self {
            files: self
                .files
                .iter()
                .map(File::try_clone)
                .collect::<std::io::Result<_>>()?,
            names: self.names.clone(),
            private_from: self.private_from,
            root_mount: self.root_mount,
            mount_from: self.mount_from,
        })
    }
    fn last(&self) -> &File {
        self.files.last().expect("held directory")
    }
    fn recheck(&self) -> Result<()> {
        for (i, file) in self.files.iter().enumerate() {
            safe_directory(file, i >= self.private_from)?;
            if i > 0 {
                check_child(&self.files[i - 1], file, &self.names[i - 1])?;
            }
            if i >= self.mount_from {
                same_publication_mount(self.root_mount, mount(file)?)?;
            }
        }
        Ok(())
    }
    fn child_existing(&mut self, name: &str) -> Result<()> {
        self.recheck()?;
        let child = existing_private_child(self.last(), name)?;
        same_publication_mount(self.root_mount, mount(&child)?)?;
        self.files.push(child);
        self.names.push(name.into());
        self.recheck()
    }
    fn child(&mut self, name: &str) -> Result<()> {
        run_id(name)?;
        self.recheck()?;
        let child = private_child(self.last(), name)?;
        same_publication_mount(self.root_mount, mount(&child)?)?;
        child.sync_all()?;
        self.last().sync_all()?;
        self.files.push(child);
        self.names.push(name.into());
        self.recheck()
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ReadStamp {
    file: (u64, u64, u64, i64, i64, i64, i64),
    parent: (u64, u64),
}
#[cfg(test)]
fn read_stamped_at(parent: &File, name: &str, cap: usize) -> Result<(Vec<u8>, ReadStamp)> {
    read_stamped_on(parent, name, cap, |_| Ok(()))
}
fn read_stamped_on(
    parent: &File,
    name: &str,
    cap: usize,
    check_mount: impl Fn(&File) -> Result<()>,
) -> Result<(Vec<u8>, ReadStamp)> {
    let mut f = openat(parent.as_raw_fd(), OsStr::new(name), FILE_FLAGS, 0)?;
    safe_file(&f)?;
    check_mount(&f)?;
    let stamp = ReadStamp {
        file: identity(&f.metadata()?),
        parent: inode(parent)?,
    };
    let bytes = read_file(&mut f, cap)?;
    let current = openat(parent.as_raw_fd(), OsStr::new(name), FILE_FLAGS, 0)?;
    safe_file(&current)?;
    check_mount(&current)?;
    ensure(
        identity(&current.metadata()?) == stamp.file,
        "file namespace changed",
    )?;
    Ok((bytes, stamp))
}
impl Store {
    pub(crate) fn read_held(&self, id: &str, cap: usize) -> Result<HeldRead> {
        self.recheck()?;
        let mut anchors = self.anchors.duplicate()?;
        anchors.child_existing(self.kind.directory())?;
        let held = HeldRead::acquire(
            anchors,
            self.basename(id)?,
            retained::ReadBound::AtMost(cap),
            retained::ReadRole::Published(self.kind),
        )?;
        self.recheck()?;
        Ok(held)
    }
    pub(crate) fn read_stamped(&self, id: &str, cap: usize) -> Result<(Vec<u8>, ReadStamp)> {
        self.recheck()?;
        let expected = mount(&self.root)?;
        let result = read_stamped_on(&self.parent, &self.basename(id)?, cap, |file| {
            same_publication_mount(expected, mount(file)?)
        })?;
        self.recheck()?;
        Ok(result)
    }
}
/// Closed foundation: no caller-supplied filename or arbitrary path publisher.
pub(crate) struct TargetStore {
    anchors: Anchors,
    path: PathBuf,
    guard: (u64, u64),
}
pub(crate) struct ProfileGuard {
    file: File,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetOpenPolicy {
    CreateReference,
    ExistingTrain,
}
fn target_open_policy(role: super::role::ExecutableRole) -> TargetOpenPolicy {
    match role {
        super::role::ExecutableRole::Reference => TargetOpenPolicy::CreateReference,
        super::role::ExecutableRole::StudyTrain => TargetOpenPolicy::ExistingTrain,
    }
}
impl TargetStore {
    pub(crate) fn open(
        root: &Path,
        campaign: &str,
        slot: &str,
        context: &super::VerifiedStudyContext,
    ) -> Result<Self> {
        match target_open_policy(context.current_role()) {
            TargetOpenPolicy::CreateReference => {
                context.require_current_reference()?;
                Self::open_with_policy(
                    root,
                    campaign,
                    slot,
                    context,
                    TargetOpenPolicy::CreateReference,
                )
            }
            TargetOpenPolicy::ExistingTrain => Self::open_existing(root, campaign, slot, context),
        }
    }
    fn open_existing(
        root: &Path,
        campaign: &str,
        slot: &str,
        context: &super::VerifiedStudyContext,
    ) -> Result<Self> {
        Self::open_with_policy(
            root,
            campaign,
            slot,
            context,
            TargetOpenPolicy::ExistingTrain,
        )
    }
    fn open_with_policy(
        root: &Path,
        campaign: &str,
        slot: &str,
        context: &super::VerifiedStudyContext,
        policy: TargetOpenPolicy,
    ) -> Result<Self> {
        ensure(
            !root.starts_with(context.root().join("results")),
            "historical results namespace",
        )?;
        run_id(campaign)?;
        ensure(
            slot.len() == 64
                && slot
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "profile slot",
        )?;
        let mut anchors = Anchors::open(root)?;
        for name in ["campaigns", campaign, "profiles", &format!("p-{slot}")] {
            match policy {
                TargetOpenPolicy::CreateReference => anchors.child(name)?,
                TargetOpenPolicy::ExistingTrain => anchors.child_existing(name)?,
            }
        }
        let path = root
            .join("campaigns")
            .join(campaign)
            .join("profiles")
            .join(format!("p-{slot}"));
        let parent = anchors.last();
        let f = match policy {
            TargetOpenPolicy::CreateReference => {
                let f = create_or_open_profile_guard(parent)?;
                same_publication_mount(anchors.root_mount, mount(&f)?)?;
                f.sync_all()?;
                parent.sync_all()?;
                f
            }
            TargetOpenPolicy::ExistingTrain => {
                let f = existing_profile_guard(parent)?;
                same_publication_mount(anchors.root_mount, mount(&f)?)?;
                f
            }
        };
        let result = Self {
            anchors,
            path,
            guard: inode(&f)?,
        };
        result.recheck()?;
        Ok(result)
    }
    pub(crate) fn path(&self) -> PathBuf {
        self.path.join("target.lock.json")
    }
    fn recheck(&self) -> Result<()> {
        self.anchors.recheck()?;
        let f = openat(
            self.anchors.last().as_raw_fd(),
            OsStr::new(".profile-guard"),
            FILE_FLAGS,
            0,
        )?;
        validate_guard(&f)?;
        ensure(inode(&f)? == self.guard, "profile guard replaced")?;
        same_publication_mount(self.anchors.root_mount, mount(&f)?)
    }
    pub(crate) fn acquire(&self) -> Result<ProfileGuard> {
        self.recheck()?;
        let guard = fresh_guard(self.anchors.last(), self.guard)?;
        self.check_guard(&guard)?;
        Ok(guard)
    }
    fn check_guard(&self, guard: &ProfileGuard) -> Result<()> {
        self.recheck()?;
        validate_guard(&guard.file)?;
        ensure(
            inode(&guard.file)? == self.guard,
            "held profile guard changed",
        )
    }
    pub(crate) fn claim_new(&self, guard: &ProfileGuard) -> Result<()> {
        self.check_guard(guard)?;
        profile_empty(self.anchors.last())
    }
    fn read_snapshot(&self) -> Result<(Vec<u8>, ReadStamp)> {
        read_stamped_on(self.anchors.last(), "target.lock.json", 7_324_204, |file| {
            same_publication_mount(self.anchors.root_mount, mount(file)?)
        })
    }
    pub(crate) fn recheck_read(&self, read: &HeldRead) -> Result<()> {
        self.recheck()?;
        read.recheck_target(self.anchors.last())?;
        self.recheck()
    }
    pub(crate) fn read_held(&self) -> Result<HeldRead> {
        self.recheck()?;
        let held = HeldRead::acquire(
            self.anchors.duplicate()?,
            "target.lock.json".into(),
            retained::ReadBound::AtMost(7_324_204),
            retained::ReadRole::Target,
        )?;
        self.recheck()?;
        Ok(held)
    }
    pub(crate) fn read(&self) -> Result<Vec<u8>> {
        self.recheck()?;
        let (bytes, _) = self.read_snapshot()?;
        self.recheck()?;
        Ok(bytes)
    }
    pub(crate) fn publish(
        &self,
        guard: &ProfileGuard,
        bytes: &[u8],
    ) -> std::result::Result<DurableBytes, PublishError> {
        self.claim_new(guard)
            .map_err(|_| PublishError::BeforeLink)?;
        ensure(bytes.len() <= 7_324_204, "lock byte cap").map_err(|_| PublishError::BeforeLink)?;
        let mut o = Native::new(self.anchors.last(), "target.lock.json".into());
        publish_ops(&mut o, bytes)?;
        o.confirm_final()
            .map_err(|_| PublishError::VisibleUnconfirmed)?;
        #[cfg(test)]
        test_hooks::checkpoint("anchor-check").map_err(|_| PublishError::VisibleUnconfirmed)?;
        self.check_guard(guard)
            .map_err(|_| PublishError::VisibleUnconfirmed)?;
        #[cfg(test)]
        test_hooks::checkpoint("release").map_err(|_| PublishError::VisibleUnconfirmed)?;
        Ok(DurableBytes {
            file_sha256: sha256_hex(bytes),
            path: self.path(),
            recovered: false,
        })
    }
    pub(crate) fn recover(
        &self,
        guard: &ProfileGuard,
        expected: &str,
        verify: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<DurableBytes> {
        self.check_guard(guard)?;
        let mut file = openat(
            self.anchors.last().as_raw_fd(),
            OsStr::new("target.lock.json"),
            FILE_FLAGS,
            0,
        )?;
        safe_file(&file)?;
        same_publication_mount(self.anchors.root_mount, mount(&file)?)?;
        let stamp = identity(&file.metadata()?);
        let bytes = read_file(&mut file, 7_324_204)?;
        ensure(sha256_hex(&bytes) == expected, "target recovery digest")?;
        verify(&bytes)?;
        let (current, current_stamp) = self.read_snapshot()?;
        ensure(
            current == bytes && current_stamp.file == stamp,
            "target changed during recovery",
        )?;
        self.check_guard(guard)?;
        let mut o = Native::recovery(self.anchors.last(), "target.lock.json".into(), file);
        recover_ops(&mut o)?;
        self.check_guard(guard)?;
        let (_, after) = self.read_snapshot()?;
        ensure(after.file == stamp, "target recovery namespace changed")?;
        Ok(DurableBytes {
            file_sha256: expected.into(),
            path: self.path(),
            recovered: true,
        })
    }
}
fn validate_guard(file: &File) -> Result<()> {
    safe_file(file)?;
    ensure(
        file.metadata()?.len() == 0 && file.metadata()?.nlink() == 1,
        "profile guard size/links",
    )
}
fn profile_empty(parent: &File) -> Result<()> {
    let file = openat(parent.as_raw_fd(), OsStr::new("."), DIR_FLAGS, 0)?;
    use std::os::fd::IntoRawFd;
    let fd = file.into_raw_fd();
    let stream = unsafe { libc::fdopendir(fd) };
    if stream.is_null() {
        unsafe { libc::close(fd) };
        return Err(ProvenanceError("profile listing unavailable"));
    }
    struct Directory(*mut libc::DIR);
    impl Drop for Directory {
        fn drop(&mut self) {
            unsafe { libc::closedir(self.0) };
        }
    }
    let stream = Directory(stream);
    loop {
        // errno must be cleared to distinguish successful end-of-directory from I/O failure.
        #[cfg(target_os = "linux")]
        unsafe {
            *libc::__errno_location() = 0;
        }
        #[cfg(target_os = "macos")]
        unsafe {
            *libc::__error() = 0;
        }
        let entry = unsafe { libc::readdir(stream.0) };
        if entry.is_null() {
            ensure(
                std::io::Error::last_os_error().raw_os_error() == Some(0),
                "profile listing failed",
            )?;
            return Ok(());
        }
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
        ensure(
            matches!(name, b"." | b".." | b".profile-guard"),
            "existing target/cases/unsupported profile state",
        )?;
    }
}

fn check_child(parent: &File, held: &File, name: &OsStr) -> Result<()> {
    let current = openat(parent.as_raw_fd(), name, DIR_FLAGS, 0)?;
    ensure(
        inode(&current)? == inode(held)?,
        "directory namespace changed",
    )
}
fn fresh_guard(parent: &File, expected: (u64, u64)) -> Result<ProfileGuard> {
    // Fresh open file description; never dup a flock FD.
    let file = openat(
        parent.as_raw_fd(),
        OsStr::new(".profile-guard"),
        FILE_FLAGS,
        0,
    )?;
    validate_guard(&file)?;
    ensure(inode(&file)? == expected, "profile guard replaced")?;
    ensure(
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0,
        "profile flock",
    )?;
    let current = openat(
        parent.as_raw_fd(),
        OsStr::new(".profile-guard"),
        FILE_FLAGS,
        0,
    )?;
    validate_guard(&current)?;
    ensure(
        inode(&current)? == expected,
        "profile guard replaced while waiting",
    )?;
    Ok(ProfileGuard { file })
}

fn private_child(parent: &File, name: &str) -> Result<File> {
    run_id(name)?;
    let name_c = cstring(name.as_bytes())?;
    if unsafe { libc::mkdirat(parent.as_raw_fd(), name_c.as_ptr(), 0o700) } != 0 {
        ensure(
            std::io::Error::last_os_error().raw_os_error() == Some(libc::EEXIST),
            "private directory create",
        )?;
    }
    let child = openat(parent.as_raw_fd(), OsStr::new(name), DIR_FLAGS, 0)?;
    // mkdirat returns no owned inode FD. Never chmod an object reached later by name:
    // a swapped entry could belong to someone else. Restrictive umasks fail closed.
    safe_directory(&child, true)?;
    Ok(child)
}

fn existing_private_child(parent: &File, name: &str) -> Result<File> {
    run_id(name)?;
    let child = openat(parent.as_raw_fd(), OsStr::new(name), DIR_FLAGS, 0)?;
    safe_directory(&child, true)?;
    Ok(child)
}
fn existing_profile_guard(parent: &File) -> Result<File> {
    let guard = openat(
        parent.as_raw_fd(),
        OsStr::new(".profile-guard"),
        FILE_FLAGS,
        0,
    )?;
    validate_guard(&guard)?;
    Ok(guard)
}
fn create_or_open_profile_guard(parent: &File) -> Result<File> {
    let name = cstring(b".profile-guard")?;
    let fd = unsafe {
        libc::openat(
            parent.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0o600,
        )
    };
    let f = if fd >= 0 {
        let f = fd_file(fd)?;
        ensure(unsafe { libc::fchmod(fd, 0o600) } == 0, "owned guard mode")?;
        f
    } else {
        ensure(
            std::io::Error::last_os_error().raw_os_error() == Some(libc::EEXIST),
            "guard create",
        )?;
        openat(
            parent.as_raw_fd(),
            OsStr::new(".profile-guard"),
            FILE_FLAGS,
            0,
        )?
    };
    validate_guard(&f)?;
    Ok(f)
}

#[cfg(test)]
mod retained_tests {
    use super::*;
    #[test]
    fn retained_exact_size_is_checked_before_allocation() {
        assert!(ReadBound::Exact(4).check(3).is_err());
        assert!(ReadBound::Exact(4).check(5).is_err());
        assert_eq!(ReadBound::Exact(4).check(4).unwrap(), 4);
        assert!(ReadBound::AtMost(4).check(5).is_err());
    }
    #[test]
    fn retained_dataset_policy_is_closed_and_exact() {
        assert_eq!(
            dataset_size("data/fashion_mnist/train-images-idx3-ubyte").unwrap(),
            47_040_016
        );
        assert!(dataset_size("../data/fashion_mnist/train-images-idx3-ubyte").is_err());
        assert!(dataset_size("data/other.npy").is_err());
    }
}
