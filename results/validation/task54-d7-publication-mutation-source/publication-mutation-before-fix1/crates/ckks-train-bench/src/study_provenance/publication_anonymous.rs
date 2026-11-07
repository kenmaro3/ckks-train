//! Private anonymous inode custody. No project raw-descriptor ownership or path cleanup.
#![forbid(unsafe_code)]
#[cfg(not(target_os = "linux"))]
use super::ProvenanceError;
use super::{File, Result};
#[cfg(target_os = "linux")]
use super::{ensure, inode, mount, safe_file, same_publication_mount};
#[cfg(target_os = "linux")]
use rustix::{
    fs::{self, AtFlags, Mode, OFlags, ResolveFlags},
    path::DecInt,
};
#[cfg(target_os = "linux")]
use std::os::unix::fs::MetadataExt;

pub(super) struct ProcFdDirectory {
    #[cfg(target_os = "linux")]
    directory: File,
    #[cfg(target_os = "linux")]
    process: u32,
}

#[cfg(target_os = "linux")]
fn io<T>(value: rustix::io::Result<T>) -> Result<T> {
    value.map_err(|e| std::io::Error::from(e).into())
}
#[cfg(target_os = "linux")]
pub(super) fn proc_directory(file: &File) -> Result<()> {
    ensure(file.metadata()?.is_dir(), "proc directory required")?;
    ensure(
        io(fs::fstatfs(file))?.f_type == fs::PROC_SUPER_MAGIC,
        "proc filesystem required",
    )
}
#[cfg(target_os = "linux")]
pub(super) fn same_process(origin: u32, current: u32) -> Result<()> {
    ensure(origin == current, "publication process changed")
}

#[cfg(target_os = "linux")]
pub(super) fn create(parent: &File) -> Result<(File, ProcFdDirectory)> {
    let process = std::process::id();
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    #[cfg(test)]
    super::test_hooks::checkpoint("c1")?;
    // Fixed internal root. No configurable alternative or weaker retry.
    let root = File::from(io(fs::open("/proc", flags, Mode::empty()))?);
    #[cfg(test)]
    super::test_hooks::checkpoint("c2")?;
    proc_directory(&root)?;
    #[cfg(test)]
    super::test_hooks::checkpoint("c3")?;
    let directory = File::from(io(fs::openat2(
        &root,
        "self/fd",
        flags,
        Mode::empty(),
        ResolveFlags::BENEATH | ResolveFlags::NO_XDEV | ResolveFlags::NO_MAGICLINKS,
    ))?);
    #[cfg(test)]
    super::test_hooks::checkpoint("c4")?;
    proc_directory(&directory)?;
    let proc_fds = ProcFdDirectory { directory, process };
    #[cfg(test)]
    super::test_hooks::checkpoint("c5")?;
    let mode = Mode::from_raw_mode(0o600);
    let file = File::from(io(fs::openat(
        parent,
        ".",
        OFlags::TMPFILE | OFlags::RDWR | OFlags::CLOEXEC | OFlags::NOFOLLOW,
        mode,
    ))?);
    #[cfg(test)]
    super::test_hooks::owned(&file)?;
    #[cfg(test)]
    super::test_hooks::checkpoint("c5-owned")?;
    #[cfg(test)]
    super::test_hooks::checkpoint("c6")?;
    io(fs::fchmod(&file, mode))?;
    #[cfg(test)]
    super::test_hooks::checkpoint("c7")?;
    safe_file(&file)?;
    ensure(
        file.metadata()?.nlink() == 0,
        "anonymous inode already linked",
    )?;
    let expected_mount = mount(parent)?;
    same_publication_mount(expected_mount, mount(&file)?)?;
    #[cfg(test)]
    super::test_hooks::checkpoint("c8")?;
    same_process(proc_fds.process, std::process::id())?;
    let checked = File::from(io(fs::openat(
        &proc_fds.directory,
        DecInt::from_fd(&file),
        OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    ))?);
    safe_file(&checked)?;
    same_publication_mount(expected_mount, mount(&checked)?)?;
    ensure(
        inode(&checked)? == inode(&file)?,
        "proc entry inode mismatch",
    )?;
    drop(checked);
    drop(root);
    #[cfg(test)]
    super::test_hooks::checkpoint("c8-done")?;
    Ok((file, proc_fds))
}
#[cfg(not(target_os = "linux"))]
pub(super) fn create(_: &File) -> Result<(File, ProcFdDirectory)> {
    Err(ProvenanceError("anonymous publication requires Linux"))
}
impl ProcFdDirectory {
    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn invalidate_process_for_test(&mut self) {
        self.process = std::process::id().wrapping_add(1);
    }
    #[cfg(target_os = "linux")]
    pub(super) fn link(&self, file: &File, parent: &File, final_name: &str) -> Result<()> {
        same_process(self.process, std::process::id())?;
        // Both borrows survive the entire syscall. DecInt alone is NOT an FD pin.
        // Return immediately: no fallible post-link check can become BeforeLink.
        io(fs::linkat(
            &self.directory,
            DecInt::from_fd(file),
            parent,
            final_name,
            AtFlags::SYMLINK_FOLLOW,
        ))
    }
    #[cfg(not(target_os = "linux"))]
    pub(super) fn link(&self, _: &File, _: &File, _: &str) -> Result<()> {
        Err(ProvenanceError("anonymous publication requires Linux"))
    }
}
