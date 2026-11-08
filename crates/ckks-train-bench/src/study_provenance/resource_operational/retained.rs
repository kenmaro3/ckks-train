//! Real descriptor-relative origins and stable account lock. No record grants a live gate.
use super::{
    approval::{ScannedApproval, scan_approval},
    host::{HostOrigin, parse_host},
};
use crate::{
    data::Sha256State,
    study_provenance::{ProvenanceError, Result, ensure},
    study_reference::operational,
};
use rustix::fs::{self, Mode, OFlags};
use std::{
    ffi::OsString,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
    path::{Component, Path},
};
fn io<T>(r: rustix::io::Result<T>) -> Result<T> {
    r.map_err(|e| std::io::Error::from(e).into())
}
const DIR: OFlags = OFlags::RDONLY
    .union(OFlags::DIRECTORY)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
const FILE: OFlags = OFlags::RDONLY
    .union(OFlags::NONBLOCK)
    .union(OFlags::NOFOLLOW)
    .union(OFlags::CLOEXEC);
fn identity(f: &File) -> Result<(u64, u64)> {
    let m = f.metadata()?;
    Ok((m.dev(), m.ino()))
}
fn safe_directory(f: &File, private: bool) -> Result<()> {
    let m = f.metadata()?;
    let uid = rustix::process::geteuid().as_raw();
    ensure(
        m.is_dir() && (m.uid() == 0 || m.uid() == uid) && m.mode() & 0o7022 == 0,
        "unsafe origin directory",
    )?;
    if private {
        ensure(
            m.uid() == uid && m.mode() & 0o7777 == 0o700,
            "private account directory",
        )?;
    }
    Ok(())
}
pub(crate) struct HeldDirectory {
    ancestors: Vec<File>,
    names: Vec<OsString>,
    ids: Vec<(u64, u64)>,
}
impl HeldDirectory {
    pub fn open(path: &Path) -> Result<Self> {
        ensure(
            super::approval::absolute(path.as_os_str().as_bytes()),
            "origin canonical absolute path",
        )?;
        let root = File::from(io(fs::open("/", DIR, Mode::empty()))?);
        safe_directory(&root, false)?;
        let mut value = Self {
            ids: vec![identity(&root)?],
            ancestors: vec![root],
            names: Vec::new(),
        };
        for part in path.components() {
            if let Component::Normal(name) = part {
                value.child_os(name, false, false)?;
            }
        }
        value.recheck()?;
        Ok(value)
    }
    pub fn file(&self) -> &File {
        self.ancestors.last().expect("retained origin root")
    }
    pub fn identity(&self) -> Result<(u64, u64)> {
        identity(self.file())
    }
    fn child_os(&mut self, name: &std::ffi::OsStr, create: bool, private: bool) -> Result<()> {
        ensure(self.names.len() < 2048, "origin ancestor cap")?;
        if create {
            match fs::mkdirat(self.file(), name, Mode::from_raw_mode(0o700)) {
                Ok(()) => self.file().sync_all()?,
                Err(rustix::io::Errno::EXIST) => {}
                Err(e) => return Err(std::io::Error::from(e).into()),
            }
        }
        let child = File::from(io(fs::openat(self.file(), name, DIR, Mode::empty()))?);
        safe_directory(&child, private)?;
        if create {
            child.sync_all()?;
        }
        self.ids.push(identity(&child)?);
        self.ancestors.push(child);
        self.names.push(name.into());
        Ok(())
    }
    pub fn child(&mut self, name: &str, create: bool, private: bool) -> Result<()> {
        ensure(
            !name.is_empty() && !name.contains(['/', '\\', '\0']) && !matches!(name, "." | ".."),
            "origin child name",
        )?;
        self.child_os(std::ffi::OsStr::new(name), create, private)
    }
    pub fn exclusive_child(&mut self, name: &str) -> Result<()> {
        ensure(
            !name.is_empty() && !name.contains(['/', '\\', '\0']) && !matches!(name, "." | ".."),
            "exclusive child name",
        )?;
        io(fs::mkdirat(self.file(), name, Mode::from_raw_mode(0o700)))?;
        self.file().sync_all()?;
        let mount = mount_identity(self.file())?;
        self.child(name, false, true)?;
        ensure(
            mount_identity(self.file())? == mount,
            "operational destination mount changed",
        )?;
        self.file().sync_all()?;
        Ok(())
    }
    pub fn duplicate(&self) -> Result<Self> {
        Ok(Self {
            ancestors: self
                .ancestors
                .iter()
                .map(File::try_clone)
                .collect::<std::io::Result<_>>()?,
            names: self.names.clone(),
            ids: self.ids.clone(),
        })
    }
    pub fn recheck(&self) -> Result<()> {
        for (i, file) in self.ancestors.iter().enumerate() {
            safe_directory(file, false)?;
            ensure(
                identity(file)? == self.ids[i],
                "origin held identity changed",
            )?;
            if i > 0 {
                let current = File::from(io(fs::openat(
                    &self.ancestors[i - 1],
                    &self.names[i - 1],
                    DIR,
                    Mode::empty(),
                ))?);
                ensure(
                    identity(&current)? == self.ids[i],
                    "origin namespace changed",
                )?;
            }
        }
        Ok(())
    }
    pub fn retained_file(&self, path: &str, cap: usize) -> Result<HeldFile> {
        ensure(
            super::super::codec::relative(path.as_bytes()),
            "origin relative file",
        )?;
        let mut parent = self.duplicate()?;
        let mut names = path.split('/').peekable();
        while let Some(name) = names.next() {
            if names.peek().is_some() {
                parent.child(name, false, false)?;
            } else {
                return HeldFile::open(parent, name, cap);
            }
        }
        Err(ProvenanceError("missing origin file"))
    }
}
pub(crate) struct HeldFile {
    pub file: File,
    parent: HeldDirectory,
    name: String,
    stamp: (u64, u64, u64, i64, i64, i64, i64),
}
fn stamp(f: &File) -> Result<(u64, u64, u64, i64, i64, i64, i64)> {
    let m = f.metadata()?;
    Ok((
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    ))
}
impl HeldFile {
    pub fn parent(&self) -> &HeldDirectory {
        &self.parent
    }
    pub fn absolute(path: &Path, cap: usize) -> Result<Self> {
        let parent = HeldDirectory::open(
            path.parent()
                .ok_or(ProvenanceError("retained file parent"))?,
        )?;
        Self::open(
            parent,
            path.file_name()
                .and_then(|n| n.to_str())
                .ok_or(ProvenanceError("retained file name"))?,
            cap,
        )
    }
    pub fn hash_incremental(&mut self, pulse: &mut impl FnMut() -> Result<()>) -> Result<String> {
        self.recheck()?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut sha = Sha256State::new();
        let mut scratch = [0; 65536];
        let mut count = 0u64;
        loop {
            pulse()?;
            let n = self.file.read(&mut scratch)?;
            if n == 0 {
                break;
            }
            count = count
                .checked_add(n as u64)
                .ok_or(ProvenanceError("retained file byte overflow"))?;
            ensure(count <= self.stamp.2, "retained file grew")?;
            sha.update(&scratch[..n]);
        }
        ensure(count == self.stamp.2, "retained file truncated")?;
        self.recheck()?;
        String::from_utf8(sha.finish_fixed().to_vec()).map_err(|_| ProvenanceError("hash ASCII"))
    }
    fn open(parent: HeldDirectory, name: &str, cap: usize) -> Result<Self> {
        parent.recheck()?;
        let file = File::from(io(fs::openat(parent.file(), name, FILE, Mode::empty()))?);
        let m = file.metadata()?;
        ensure(
            m.is_file()
                && m.nlink() == 1
                && (m.uid() == 0 || m.uid() == rustix::process::geteuid().as_raw())
                && m.mode() & 0o7022 == 0
                && m.len() <= cap as u64,
            "retained source file metadata",
        )?;
        let stamp = stamp(&file)?;
        let value = Self {
            file,
            parent,
            name: name.into(),
            stamp,
        };
        value.recheck()?;
        Ok(value)
    }
    pub fn recheck(&self) -> Result<()> {
        self.parent.recheck()?;
        ensure(stamp(&self.file)? == self.stamp, "retained source changed")?;
        let current = File::from(io(fs::openat(
            self.parent.file(),
            self.name.as_str(),
            FILE,
            Mode::empty(),
        ))?);
        ensure(
            stamp(&current)? == self.stamp,
            "retained source namespace changed",
        )
    }
    pub fn fixed<const N: usize>(&mut self) -> Result<([u8; N], usize)> {
        self.recheck()?;
        let length =
            usize::try_from(self.stamp.2).map_err(|_| ProvenanceError("source length overflow"))?;
        ensure(length <= N, "source fixed cap")?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = [0; N];
        self.file.read_exact(&mut bytes[..length])?;
        let mut extra = [0];
        ensure(self.file.read(&mut extra)? == 0, "source grew")?;
        self.recheck()?;
        Ok((bytes, length))
    }
    pub fn bytes(&mut self, expected: (u64, &str)) -> Result<Vec<u8>> {
        ensure(
            self.stamp.2 == expected.0 && expected.0 <= 8 * 1024 * 1024,
            "source length pin",
        )?;
        self.recheck()?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = vec![0; expected.0 as usize];
        self.file.read_exact(&mut bytes)?;
        ensure(
            crate::data::sha256_hex(&bytes) == expected.1,
            "source hash pin",
        )?;
        self.recheck()?;
        Ok(bytes)
    }
    pub fn fixed_name_bytes(&mut self, cap: usize) -> Result<Vec<u8>> {
        ensure(self.stamp.2 <= cap as u64, "fixed source read cap")?;
        self.recheck()?;
        self.file.seek(SeekFrom::Start(0))?;
        let mut bytes = vec![0; self.stamp.2 as usize];
        self.file.read_exact(&mut bytes)?;
        self.recheck()?;
        Ok(bytes)
    }
}
pub(crate) struct MinimalOrigin {
    pub approval: ScannedApproval,
    pub approval_file: HeldFile,
    pub policy: HeldDirectory,
    pub observation: HeldDirectory,
    pub account: HeldDirectory,
    pub bootstrap_file: HeldFile,
    pub host_file: HeldFile,
    pub host: HostOrigin,
    pub bootstrap: [u8; 4096],
    pub bootstrap_len: usize,
}
impl MinimalOrigin {
    pub fn acquire(approval_path: &Path, approval_hash: &str, repository: &Path) -> Result<Self> {
        let parent = HeldDirectory::open(
            approval_path
                .parent()
                .ok_or(ProvenanceError("approval parent"))?,
        )?;
        let name = approval_path
            .file_name()
            .and_then(|v| v.to_str())
            .ok_or(ProvenanceError("approval basename"))?;
        let mut approval_file = HeldFile::open(parent, name, 8 * 1024 * 1024)?;
        let approval = scan_approval(&mut approval_file.file, approval_hash)?;
        approval_file.recheck()?;
        let o = &approval.origin;
        ensure(
            o.account_uid == u64::from(rustix::process::geteuid().as_raw())
                && rustix::process::getuid() == rustix::process::geteuid(),
            "launcher account UID",
        )?;
        let account = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            o.account_anchor.as_bytes(),
        )))?;
        let policy = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            o.policy_origin.as_bytes(),
        )))?;
        let observation = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
            o.observation_origin.as_bytes(),
        )))?;
        ensure(
            account.identity()? == o.account_identity
                && policy.identity()? == o.policy_identity
                && observation.identity()? == o.observation_identity,
            "launcher origin identities",
        )?;
        ensure(
            repository.join("results").as_os_str().as_bytes() == o.observation_origin.as_bytes(),
            "fixed Observation root",
        )?;
        ensure(
            o.host_origin.root == 3
                && o.host_origin.path.as_bytes()
                    == b"resource/linux-cpu-observed-v1/host-origin.bin"
                && o.native_validation.root == 3
                && o.native_validation.path.as_bytes()
                    == b"resource/linux-cpu-observed-v1/native-validation.json",
            "fixed Policy origin names",
        )?;
        let mut bootstrap_file =
            policy.retained_file("resource/linux-cpu-observed-v1/bootstrap.bin", 4096)?;
        let (bootstrap, bootstrap_len) = bootstrap_file.fixed::<4096>()?;
        let mut digest = Sha256State::new();
        digest.update(&bootstrap[..bootstrap_len]);
        ensure(
            digest.finish_fixed() == o.bootstrap.as_bytes(),
            "expected launcher bootstrap hash",
        )?;
        let b = operational::bootstrap(&bootstrap[..bootstrap_len])
            .map_err(|_| ProvenanceError("bootstrap record"))?;
        ensure(
            b.physical.as_bytes() == o.physical_host.as_bytes()
                && b.source.as_bytes() == approval.source.as_bytes()
                && b.revision.as_bytes() == approval.revision.as_bytes(),
            "bootstrap/approval identity",
        )?;
        ensure(
            b.host
                == (
                    o.host_origin.root,
                    o.host_origin.path.text()?,
                    o.host_origin.length,
                    o.host_origin.hash.text()?,
                ),
            "bootstrap namespace origin pin",
        )?;
        let mut host_file = policy.retained_file(o.host_origin.path.text()?, 4096)?;
        let (raw, length) = host_file.fixed::<4096>()?;
        let mut digest = Sha256State::new();
        digest.update(&raw[..length]);
        ensure(
            length as u64 == o.host_origin.length
                && digest.finish_fixed() == o.host_origin.hash.as_bytes(),
            "host-origin pin",
        )?;
        let host = parse_host(&raw[..length], false)?;
        ensure(
            host.account_uid == o.account_uid
                && host.account_identity == o.account_identity
                && host.physical == o.physical_host
                && host.boot.text()? == b.boot,
            "host/launcher projection",
        )?;
        check_kernel_projection(&host)?;
        Ok(Self {
            approval,
            approval_file,
            policy,
            observation,
            account,
            bootstrap_file,
            host_file,
            host,
            bootstrap,
            bootstrap_len,
        })
    }
    pub fn acquire_lock(&self) -> Result<(HeldDirectory, File)> {
        acquire_account_lock(&self.account, self.approval.origin.physical_host.text()?)
    }
}
pub(crate) fn acquire_account_lock(
    account: &HeldDirectory,
    physical: &str,
) -> Result<(HeldDirectory, File)> {
    ensure(
        super::super::codec::hash_valid(physical),
        "stable physical host key grammar",
    )?;
    let mut state = account.duplicate()?;
    for name in [
        ".local",
        "state",
        "hekit",
        "task54-resource-v1",
        "hosts",
        physical,
    ] {
        state.child(name, true, true)?;
    }
    ensure(
        mount_identity(state.file())? == mount_identity(account.file())?,
        "account state mount differs from covered account filesystem",
    )?;
    let fd = match fs::openat(
        state.file(),
        "execution.lock",
        OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::CREATE | OFlags::EXCL,
        Mode::from_raw_mode(0o600),
    ) {
        Ok(fd) => {
            let file = File::from(fd);
            file.sync_all()?;
            state.file().sync_all()?;
            file
        }
        Err(rustix::io::Errno::EXIST) => File::from(io(fs::openat(
            state.file(),
            "execution.lock",
            OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        ))?),
        Err(e) => return Err(std::io::Error::from(e).into()),
    };
    let m = fd.metadata()?;
    ensure(
        m.is_file()
            && m.nlink() == 1
            && m.uid() == rustix::process::geteuid().as_raw()
            && m.mode() & 0o7777 == 0o600,
        "stable resource lock identity/mode",
    )?;
    io(fs::flock(&fd, fs::FlockOperation::NonBlockingLockExclusive))?;
    state.recheck()?;
    Ok((state, fd))
}

fn kernel_file(parent: &File, name: &str) -> Result<File> {
    Ok(File::from(io(fs::openat(
        parent,
        name,
        FILE,
        Mode::empty(),
    ))?))
}
fn fixed_kernel<const N: usize>(parent: &File, name: &str) -> Result<([u8; N], usize)> {
    let mut file = kernel_file(parent, name)?;
    let mut out = [0; N];
    let mut count = 0;
    while count < N {
        let n = file.read(&mut out[count..])?;
        if n == 0 {
            break;
        }
        count += n;
    }
    let mut extra = [0];
    ensure(file.read(&mut extra)? == 0, "fixed kernel projection cap")?;
    Ok((out, count))
}
pub(crate) fn mount_identity(file: &File) -> Result<(u64, u64, u64)> {
    let metadata = file.metadata()?;
    let stat = io(fs::statx(
        file,
        "",
        fs::AtFlags::EMPTY_PATH,
        fs::StatxFlags::MNT_ID,
    ))?;
    ensure(
        stat.stx_mask & fs::StatxFlags::MNT_ID.bits() != 0 && stat.stx_mnt_id > 0,
        "actual mount ID unavailable",
    )?;
    let fs_type = io(fs::fstatfs(file))?.f_type as u64;
    Ok((metadata.dev(), stat.stx_mnt_id, fs_type))
}
pub(crate) fn check_kernel_projection(host: &HostOrigin) -> Result<()> {
    let proc = HeldDirectory::open(Path::new("/proc"))?;
    let cgroup = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
        host.cgroup_root.path.as_bytes(),
    )))?;
    for (file, expected) in [
        (proc.file(), &host.proc_root),
        (cgroup.file(), &host.cgroup_root),
    ] {
        ensure(
            identity(file)? == (expected.device, expected.inode)
                && mount_identity(file)? == (expected.device, expected.mount_id, expected.fs_type),
            "actual retained kernel root differs",
        )?;
    }
    let (boot, length) = fixed_kernel::<37>(proc.file(), "sys/kernel/random/boot_id")?;
    ensure(
        length == 37 && boot[36] == b'\n' && &boot[..36] == host.boot.as_bytes(),
        "actual boot differs",
    )?;
    let pid = rustix::process::getpid();
    let own = File::from(io(fs::openat(
        proc.file(),
        pid.as_raw_nonzero().get().to_string(),
        DIR,
        Mode::empty(),
    ))?);
    let (before, before_len) = fixed_kernel::<4096>(&own, "stat")?;
    for (name, expected) in ["mnt", "cgroup", "pid", "user"].iter().zip(host.namespaces) {
        // Only the four literal kernel namespace magic links are intentionally followed.
        let namespace = File::from(io(fs::openat(
            &own,
            format!("ns/{name}"),
            OFlags::RDONLY | OFlags::CLOEXEC,
            Mode::empty(),
        ))?);
        ensure(
            identity(&namespace)? == expected
                && io(fs::fstatfs(&namespace))?.f_type as u64 == 0x6e736673,
            "actual host namespace differs",
        )?;
    }
    let (after, after_len) = fixed_kernel::<4096>(&own, "stat")?;
    // Compare immutable field22, not mutable stat counters. The held own directory
    // and current syscall PID additionally preserve process-origin continuity.
    fn start(bytes: &[u8]) -> Result<&[u8]> {
        let end = bytes
            .iter()
            .rposition(|v| *v == b')')
            .ok_or(ProvenanceError("kernel stat comm"))?;
        bytes[end + 1..]
            .split(u8::is_ascii_whitespace)
            .filter(|v| !v.is_empty())
            .nth(19)
            .ok_or(ProvenanceError("kernel stat start"))
    }
    ensure(
        start(&before[..before_len])? == start(&after[..after_len])?
            && rustix::process::getpid() == pid,
        "kernel process origin changed",
    )?;
    let (current, current_len) = fixed_kernel::<37>(proc.file(), "sys/kernel/random/boot_id")?;
    ensure(
        current_len == length && current == boot,
        "boot changed during origin check",
    )?;
    proc.recheck()?;
    cgroup.recheck()
}
