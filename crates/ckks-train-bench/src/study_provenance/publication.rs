//! Descriptor-relative foundation. No scientific publication type is exposed.
use super::{ProvenanceError, Result, ensure};
use crate::data::sha256_hex;
use std::{
    ffi::{CString, OsStr},
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
    fn unlink_owned(&mut self) -> Result<()>;
}
fn publish_ops(o: &mut impl Ops, b: &[u8]) -> std::result::Result<(), PublishError> {
    o.create().map_err(|_| PublishError::BeforeLink)?;
    if o.write(b)
        .and_then(|()| o.file_sync())
        .and_then(|()| o.link())
        .is_err()
    {
        let _ = o.unlink_owned();
        return Err(PublishError::BeforeLink);
    }
    o.dir_sync()
        .and_then(|()| o.unlink_owned())
        .and_then(|()| o.dir_sync())
        .map_err(|_| PublishError::VisibleUnconfirmed)
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
fn safe_directory(f: &File, private: bool) -> Result<()> {
    let m = f.metadata()?;
    ensure(m.is_dir(), "directory required")?;
    let uid = unsafe { libc::geteuid() };
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
fn walk(path: &Path) -> Result<File> {
    ensure(path.is_absolute(), "absolute publication root")?;
    let c = cstring(b"/")?;
    let mut fd = fd_file(unsafe {
        libc::open(
            c.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    })?;
    safe_directory(&fd, false)?;
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(name) => {
                fd = openat(
                    fd.as_raw_fd(),
                    name,
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                    0,
                )?;
                safe_directory(&fd, false)?;
            }
            _ => return Err(ProvenanceError("noncanonical publication path")),
        }
    }
    safe_directory(&fd, true)?;
    Ok(fd)
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
        let f = walk(root)?;
        let parent = openat(
            f.as_raw_fd(),
            OsStr::new(kind.directory()),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            0,
        )?;
        safe_directory(&parent, true)?;
        same_publication_mount(mount(&f)?, mount(&parent)?)?;
        Ok(Self {
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
        let mut o = Native {
            parent: &self.parent,
            final_name: name.clone(),
            temp: None,
            file: None,
            id: id.into(),
        };
        publish_ops(&mut o, bytes)?;
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
        let mut o = Native {
            parent: &self.parent,
            final_name: name.clone(),
            temp: None,
            file: Some(file),
            id: id.into(),
        };
        recover_ops(&mut o)?;
        Ok(DurableBytes {
            file_sha256: expected.into(),
            path: self.path.join(name),
            recovered: true,
        })
    }
}
struct Native<'a> {
    parent: &'a File,
    final_name: String,
    temp: Option<String>,
    file: Option<File>,
    id: String,
}
impl Ops for Native<'_> {
    fn create(&mut self) -> Result<()> {
        for counter in 0..64 {
            let name = format!(".tmp-{}-{counter}", self.id);
            let c = cstring(name.as_bytes())?;
            let fd = unsafe {
                libc::openat(
                    self.parent.as_raw_fd(),
                    c.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o600,
                )
            };
            if fd >= 0 {
                self.file = Some(unsafe { File::from_raw_fd(fd) });
                self.temp = Some(name);
                // Set the exact mode on OUR newly created inode, independent of
                // caller umask. Existing directories/files are never chmodded.
                if unsafe { libc::fchmod(fd, 0o600) } != 0 {
                    let _ = self.unlink_owned();
                    return Err(ProvenanceError("temporary mode unavailable"));
                }
                return Ok(());
            }
            if std::io::Error::last_os_error().raw_os_error() != Some(libc::EEXIST) {
                return Err(ProvenanceError("temporary create"));
            }
        }
        Err(ProvenanceError("temporary collisions exhausted"))
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        write_complete(
            self.file
                .as_mut()
                .ok_or(ProvenanceError("owned file absent"))?,
            bytes,
        )
    }
    fn file_sync(&mut self) -> Result<()> {
        self.file
            .as_ref()
            .ok_or(ProvenanceError("owned file absent"))?
            .sync_all()?;
        Ok(())
    }
    fn link(&mut self) -> Result<()> {
        let from = cstring(
            self.temp
                .as_ref()
                .ok_or(ProvenanceError("owned temp absent"))?
                .as_bytes(),
        )?;
        let to = cstring(self.final_name.as_bytes())?;
        ensure(
            unsafe {
                libc::linkat(
                    self.parent.as_raw_fd(),
                    from.as_ptr(),
                    self.parent.as_raw_fd(),
                    to.as_ptr(),
                    0,
                )
            } == 0,
            "no-replace link failed",
        )
    }
    fn dir_sync(&mut self) -> Result<()> {
        self.parent.sync_all()?;
        Ok(())
    }
    fn unlink_owned(&mut self) -> Result<()> {
        let name = self
            .temp
            .as_ref()
            .ok_or(ProvenanceError("no temp ownership"))?;
        let c = cstring(name.as_bytes())?;
        ensure(
            unsafe { libc::unlinkat(self.parent.as_raw_fd(), c.as_ptr(), 0) } == 0,
            "owned temp cleanup failed",
        )?;
        self.temp = None;
        Ok(())
    }
}
#[cfg(test)]
#[path = "publication_tests.rs"]
mod tests;
