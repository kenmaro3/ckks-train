//! Necessary byte payload versus live observed availability, never physical fit.
use crate::study_provenance::{ProvenanceError, Result, ensure};
fn decimal(raw: &[u8]) -> Result<u64> {
    ensure(
        !raw.is_empty() && raw.iter().all(u8::is_ascii_digit) && (raw == b"0" || raw[0] != b'0'),
        "kernel canonical decimal",
    )?;
    raw.iter().try_fold(0u64, |n, b| {
        n.checked_mul(10)
            .and_then(|v| v.checked_add(u64::from(b - b'0')))
            .ok_or(ProvenanceError("kernel decimal overflow"))
    })
}
pub(super) fn memory_limit(raw: &[u8]) -> Result<Option<u64>> {
    ensure(raw.ends_with(b"\n"), "cgroup scalar newline")?;
    let value = &raw[..raw.len() - 1];
    if value == b"max" {
        Ok(None)
    } else {
        Ok(Some(decimal(value)?))
    }
}
pub(super) fn mem_available(raw: &[u8]) -> Result<u64> {
    let mut value = None;
    for line in raw.split(|b| *b == b'\n') {
        if let Some(tail) = line.strip_prefix(b"MemAvailable:") {
            ensure(value.is_none(), "duplicate MemAvailable")?;
            let parts = tail
                .split(u8::is_ascii_whitespace)
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            ensure(parts.len() == 2 && parts[1] == b"kB", "MemAvailable units")?;
            value = Some(
                decimal(parts[0])?
                    .checked_mul(1024)
                    .ok_or(ProvenanceError("MemAvailable byte overflow"))?,
            );
        }
    }
    value.ok_or(ProvenanceError("MemAvailable unavailable"))
}
pub(super) fn membership(raw: &[u8]) -> Result<&str> {
    ensure(
        raw.starts_with(b"0::/")
            && raw.ends_with(b"\n")
            && raw.iter().filter(|b| **b == b'\n').count() == 1,
        "unified cgroup membership required",
    )?;
    let path = &raw[3..raw.len() - 1];
    ensure(
        path == b"/" || crate::study_provenance::resource_operational::approval::absolute(path),
        "cgroup membership path",
    )?;
    std::str::from_utf8(path).map_err(|_| ProvenanceError("cgroup membership UTF8"))
}
pub(super) fn global_mount(raw: &[u8], expected: u64, path: &[u8]) -> Result<()> {
    let mut found = 0;
    for line in raw.split(|b| *b == b'\n').filter(|s| !s.is_empty()) {
        let parts = line.split(|b| *b == b' ').collect::<Vec<_>>();
        let dash = parts
            .iter()
            .position(|p| *p == b"-")
            .ok_or(ProvenanceError("mountinfo separator"))?;
        ensure(dash >= 6 && parts.len() >= dash + 4, "mountinfo fields")?;
        ensure(
            parts[dash + 1] != b"cgroup",
            "hybrid cgroup hierarchy unsupported",
        )?;
        if parts[dash + 1] == b"cgroup2" {
            found += 1;
            ensure(
                decimal(parts[0])? == expected && parts[3] == b"/" && parts[4] == path,
                "cgroup mount hides structural ancestors",
            )?;
        }
    }
    ensure(found == 1, "unique global cgroup2 mapping required")
}
pub(super) fn fits(host: u64, groups: &[(Option<u64>, u64)], additional: u64) -> Result<()> {
    ensure(additional <= host, "necessary payload exceeds MemAvailable")?;
    for (limit, current) in groups {
        if let Some(limit) = limit {
            let headroom = limit
                .checked_sub(*current)
                .ok_or(ProvenanceError("cgroup already over finite limit"))?;
            ensure(
                additional <= headroom,
                "necessary payload exceeds cgroup headroom",
            )?;
        }
    }
    Ok(())
}
pub(super) fn disk_available(
    frsize: u64,
    bavail: u64,
    favail: u64,
    bytes: u64,
    files: u64,
) -> Result<u64> {
    ensure(frsize > 0, "filesystem fragment size unknown")?;
    let available = frsize
        .checked_mul(bavail)
        .ok_or(ProvenanceError("filesystem byte availability overflow"))?;
    ensure(
        bytes <= available && files <= favail,
        "necessary observation storage exceeds availability",
    )?;
    Ok(available)
}
#[cfg(target_os = "linux")]
pub(super) enum DiskRoot {
    Artifact(u8),
    Account,
}
/// root, device, mount, type, frsize, bavail, favail, derived available bytes.
#[cfg(target_os = "linux")]
type DiskObservation = (DiskRoot, u64, u64, u64, u64, u64, u64, u64);
#[cfg(target_os = "linux")]
pub(super) struct CapacityObservation {
    pub raw: Vec<(String, Vec<u8>)>,
    pub host_available: u64,
    pub groups: Vec<(Option<u64>, u64)>,
    pub disk: Vec<DiskObservation>,
}
#[cfg(target_os = "linux")]
pub(super) fn observe(
    host: &crate::study_provenance::resource_operational::host::HostOrigin,
    package: &crate::study_provenance::resource_operational::package::Package,
    additional: u64,
    disk_bytes: u64,
    disk_files: u64,
    pulse: &mut impl FnMut() -> Result<()>,
) -> Result<CapacityObservation> {
    use crate::study_provenance::resource_operational::retained::{
        HeldDirectory, check_kernel_projection, mount_identity,
    };
    use std::{os::unix::ffi::OsStrExt, path::Path};
    pulse()?;
    check_kernel_projection(host)?;
    package.recheck()?;
    let proc = HeldDirectory::open(Path::new("/proc"))?;
    let own = super::native::open_dir(proc.file(), &std::process::id().to_string())?;
    let mount = super::native::read_at(&own, "mountinfo", 8 * 1024 * 1024)?;
    let member = super::native::read_at(&own, "cgroup", 8 * 1024 * 1024)?;
    global_mount(
        &mount,
        host.cgroup_root.mount_id,
        host.cgroup_root.path.as_bytes(),
    )?;
    ensure(
        mount == package.mountinfo && member == package.membership,
        "current topology differs from direct-host capture",
    )?;
    let root = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
        host.cgroup_root.path.as_bytes(),
    )))?;
    let controllers = super::native::read_at(root.file(), "cgroup.controllers", 8 * 1024 * 1024)?;
    ensure(
        controllers == package.controllers
            && controllers
                .split(u8::is_ascii_whitespace)
                .any(|s| s == b"memory"),
        "memory controller visibility",
    )?;
    let mem = super::native::read_at(proc.file(), "meminfo", 8 * 1024 * 1024)?;
    let host_available = mem_available(&mem)?;
    let mut result = CapacityObservation {
        raw: vec![
            ("mountinfo-before".into(), mount.clone()),
            ("membership-before".into(), member.clone()),
            ("controllers".into(), controllers),
            ("meminfo".into(), mem),
        ],
        host_available,
        groups: Vec::new(),
        disk: Vec::new(),
    };
    let mut current = root.duplicate()?;
    for (index, name) in membership(&member)?
        .split('/')
        .filter(|n| !n.is_empty())
        .enumerate()
    {
        pulse()?;
        current.child(name, false, false)?;
        ensure(
            mount_identity(current.file())?.1 == host.cgroup_root.mount_id,
            "cgroup ancestor mount changed",
        )?;
        let kind = super::native::read_at(current.file(), "cgroup.type", 64)?;
        ensure(kind == b"domain\n", "threaded cgroup unsupported")?;
        let limit = super::native::read_at(current.file(), "memory.max", 64)?;
        let usage = super::native::read_at(current.file(), "memory.current", 64)?;
        result.groups.push((
            memory_limit(&limit)?,
            memory_limit(&usage)?.ok_or(ProvenanceError("memory.current cannot be max"))?,
        ));
        result.raw.extend([
            (format!("ancestor-{index}-type"), kind),
            (format!("ancestor-{index}-limit"), limit),
            (format!("ancestor-{index}-current"), usage),
        ]);
        ensure(result.raw.len() <= 4096, "capacity raw file cap")?;
    }
    fits(host_available, &result.groups, additional)?;
    for (role, directory) in package
        .roots
        .iter()
        .map(|(r, d)| (DiskRoot::Artifact(*r), d))
        .chain(std::iter::once((DiskRoot::Account, &package.account)))
    {
        pulse()?;
        let fs = super::native::io(rustix::fs::fstatvfs(directory.file()))?;
        let destination = matches!(role, DiskRoot::Artifact(5) | DiskRoot::Account);
        let available = disk_available(
            fs.f_frsize,
            fs.f_bavail,
            fs.f_favail,
            if destination { disk_bytes } else { 0 },
            if destination { disk_files } else { 0 },
        )?;
        let (device, mount, kind) = mount_identity(directory.file())?;
        result.disk.push((
            role,
            device,
            mount,
            kind,
            fs.f_frsize,
            fs.f_bavail,
            fs.f_favail,
            available,
        ));
    }
    pulse()?;
    let after_mount = super::native::read_at(&own, "mountinfo", 8 * 1024 * 1024)?;
    let after_member = super::native::read_at(&own, "cgroup", 8 * 1024 * 1024)?;
    ensure(
        after_mount == mount && after_member == member,
        "capacity topology changed during observation",
    )?;
    result.raw.extend([
        ("mountinfo-after".into(), after_mount),
        ("membership-after".into(), after_member),
    ]);
    pulse()?;
    check_kernel_projection(host)?;
    package.recheck()?;
    Ok(result)
}
