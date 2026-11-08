//! Fixed combined direct-host projection; parsing is not independent host evidence.
use super::approval::{Fixed, OriginRef, Scan};
use crate::study_provenance::{Result, ensure};
#[derive(Debug)]
pub(crate) struct KernelRoot {
    pub path: Fixed<4096>,
    pub device: u64,
    pub inode: u64,
    pub mount_id: u64,
    pub fs_type: u64,
}
#[derive(Debug)]
pub(crate) struct HostOrigin {
    pub physical: Fixed<64>,
    pub boot: Fixed<36>,
    pub namespaces: [(u64, u64); 4],
    pub installation: OriginRef,
    pub account_uid: u64,
    pub account_identity: (u64, u64),
    pub proc_root: KernelRoot,
    pub cgroup_root: KernelRoot,
    pub mountinfo: OriginRef,
    pub membership: OriginRef,
    pub controllers: OriginRef,
    pub machine: OriginRef,
    pub filesystems: OriginRef,
}
fn root(s: &mut Scan<'_, impl std::io::Read>) -> Result<KernelRoot> {
    let path = s.path()?;
    let (device, inode) = s.identity()?;
    let mount_id = s.number()?;
    let fs_type = s.number()?;
    ensure(mount_id > 0, "host root mount ID")?;
    Ok(KernelRoot {
        path,
        device,
        inode,
        mount_id,
        fs_type,
    })
}
pub(crate) fn parse_host(bytes: &[u8], oracle: bool) -> Result<HostOrigin> {
    ensure(
        bytes.len() <= 4096 && (!oracle || cfg!(test)),
        "host projection cap/domain",
    )?;
    let mut input = std::io::Cursor::new(bytes);
    let mut s = Scan::new(&mut input);
    ensure(
        s.text::<128>()?.as_bytes() == b"hekit-study-reference-encoding-v1",
        "host encoding",
    )?;
    ensure(
        s.text::<128>()?.as_bytes()
            == if oracle {
                b"hekit-test-resource-observed-host-origin-v1".as_slice()
            } else {
                b"hekit-study-resource-observed-host-origin-v1".as_slice()
            }
            && s.tag(1)? == u8::from(oracle),
        "host domain/synthetic",
    )?;
    let physical = s.hash()?;
    let boot = s.text::<36>()?;
    ensure(
        boot.as_bytes().len() == 36
            && boot.as_bytes().iter().enumerate().all(|(i, v)| {
                if [8, 13, 18, 23].contains(&i) {
                    *v == b'-'
                } else {
                    v.is_ascii_digit() || (b'a'..=b'f').contains(v)
                }
            }),
        "host boot UUID",
    )?;
    let namespaces = [s.identity()?, s.identity()?, s.identity()?, s.identity()?];
    let installation = s.reference()?;
    let account_uid = s.number()?;
    let account_identity = s.identity()?;
    let proc_root = root(&mut s)?;
    let cgroup_root = root(&mut s)?;
    ensure(
        proc_root.path.as_bytes() == b"/proc"
            && proc_root.fs_type == 0x9fa0
            && cgroup_root.fs_type == 0x63677270,
        "supported proc/cgroup root",
    )?;
    ensure(
        s.text::<4096>()?.as_bytes() == b"/",
        "global structural root projection",
    )?;
    let mountinfo = s.reference()?;
    let membership = s.reference()?;
    let controllers = s.reference()?;
    let machine = s.reference()?;
    let filesystems = s.reference()?;
    ensure(
        filesystems.root == 3
            && filesystems.path.as_bytes() == b"resource/linux-cpu-observed-v1/filesystems.json",
        "fixed filesystem source",
    )?;
    s.done()?;
    Ok(HostOrigin {
        physical,
        boot,
        namespaces,
        installation,
        account_uid,
        account_identity,
        proc_root,
        cgroup_root,
        mountinfo,
        membership,
        controllers,
        machine,
        filesystems,
    })
}
