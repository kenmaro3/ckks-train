//! Private transport projection; only live descriptor checks authenticate it.
use super::schedule::Token;
use crate::study_provenance::{ProvenanceError, Result, ensure};
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Projection {
    pub parent: Token,
    pub child: Token,
    pub lock: (u64, u64),
    pub origin: u64,
    pub boot: [u8; 36],
    pub executable: [u8; 64],
    pub invocation: [u8; 64],
}
impl Projection {
    pub const LENGTH: usize = 226;
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        ensure(
            bytes.len() == Self::LENGTH && &bytes[..6] == b"HKCPU1",
            "private handshake length/domain",
        )?;
        let mut n = [0; 7];
        for (slot, b) in n.iter_mut().zip(bytes[6..62].chunks_exact(8)) {
            *slot = u64::from_le_bytes(
                b.try_into()
                    .map_err(|_| ProvenanceError("handshake number"))?,
            );
        }
        ensure(
            n[..4].iter().all(|n| *n > 0)
                && n[0] <= i32::MAX as u64
                && n[2] <= i32::MAX as u64
                && n[0] != n[2],
            "handshake process fields",
        )?;
        let boot = bytes[62..98]
            .try_into()
            .map_err(|_| ProvenanceError("handshake boot"))?;
        for (i, b) in bytes[62..98].iter().enumerate() {
            ensure(
                if [8, 13, 18, 23].contains(&i) {
                    *b == b'-'
                } else {
                    b.is_ascii_digit() || (b'a'..=b'f').contains(b)
                },
                "handshake boot grammar",
            )?;
        }
        ensure(
            bytes[98..]
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)),
            "handshake hash grammar",
        )?;
        Ok(Self {
            parent: Token {
                pid: n[0] as u32,
                start: n[1],
            },
            child: Token {
                pid: n[2] as u32,
                start: n[3],
            },
            lock: (n[4], n[5]),
            origin: n[6],
            boot,
            executable: bytes[98..162]
                .try_into()
                .map_err(|_| ProvenanceError("handshake executable"))?,
            invocation: bytes[162..]
                .try_into()
                .map_err(|_| ProvenanceError("handshake invocation"))?,
        })
    }
    pub fn encode(&self) -> [u8; Self::LENGTH] {
        let mut out = [0; Self::LENGTH];
        out[..6].copy_from_slice(b"HKCPU1");
        for (slot, value) in out[6..62].chunks_exact_mut(8).zip([
            u64::from(self.parent.pid),
            self.parent.start,
            u64::from(self.child.pid),
            self.child.start,
            self.lock.0,
            self.lock.1,
            self.origin,
        ]) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        out[62..98].copy_from_slice(&self.boot);
        out[98..162].copy_from_slice(&self.executable);
        out[162..].copy_from_slice(&self.invocation);
        out
    }
}
pub(super) fn invocation(args: &[std::ffi::OsString]) -> Result<[u8; 64]> {
    use std::os::unix::ffi::OsStrExt;
    ensure(args.len() <= 256, "handshake argv count")?;
    let mut sha = crate::data::Sha256State::new();
    sha.update(b"hekit-private-child-argv-v1");
    sha.update(&(args.len() as u64).to_le_bytes());
    for arg in args {
        let raw = arg.as_bytes();
        ensure(
            raw.len() <= 4096 && !raw.contains(&0),
            "handshake argv item",
        )?;
        sha.update(&(raw.len() as u64).to_le_bytes());
        sha.update(raw);
    }
    Ok(sha.finish_fixed())
}
pub(super) fn peer_matches(
    unix_stream: bool,
    pid: u32,
    uid: u32,
    parent: u32,
    real: u32,
    effective: u32,
) -> Result<()> {
    ensure(
        unix_stream && parent > 0 && pid == parent && uid == effective && real == effective,
        "inherited socket creator/account mismatch",
    )
}
#[cfg(target_os = "linux")]
pub(super) struct ParentGuard {
    own: super::native::ProcOrigin,
    parent: std::fs::File,
    parent_token: Token,
    executable: (u64, u64),
    lock: std::fs::File,
    lock_identity: (u64, u64),
}
#[cfg(target_os = "linux")]
impl ParentGuard {
    pub fn lock_clone(&self) -> Result<std::fs::File> {
        self.recheck()?;
        Ok(std::fs::File::from(private_dup(&self.lock)?))
    }
    pub fn recheck(&self) -> Result<()> {
        use std::os::unix::fs::MetadataExt;
        self.own.recheck()?;
        ensure(
            rustix::process::getppid()
                .is_some_and(|p| p.as_raw_nonzero().get() as u32 == self.parent_token.pid),
            "private child reparented",
        )?;
        ensure(
            super::observation::start_ticks(
                &super::native::read_at(&self.parent, "stat", 4096)?,
                self.parent_token.pid,
            )? == self.parent_token.start,
            "supervisor start changed",
        )?;
        let m = self.lock.metadata()?;
        ensure(
            (m.dev(), m.ino()) == self.lock_identity
                && m.is_file()
                && m.nlink() == 1
                && m.mode() & 0o7777 == 0o600
                && m.uid() == rustix::process::geteuid().as_raw(),
            "inherited lock changed",
        )?;
        ensure(
            super::native::exe_identity(&self.parent)? == self.executable
                && self.own.executable_identity()? == self.executable,
            "private supervisor executable changed",
        )
    }
}
/// Private aliases share the original OFD but never leak through a later exec.
/// Only explicit Stdio or authenticated ancillary transfer inherits a handle.
#[cfg(target_os = "linux")]
pub(super) fn private_dup(fd: impl std::os::fd::AsFd) -> Result<std::os::fd::OwnedFd> {
    super::native::io(rustix::io::fcntl_dupfd_cloexec(fd, 3))
}
#[cfg(target_os = "linux")]
pub(super) fn child_channel(
    args: &[std::ffi::OsString],
) -> Result<(super::protocol::Channel, ParentGuard, Projection)> {
    use super::{
        native::{self, io},
        protocol::{Channel, Tag},
    };
    use rustix::net::{AddressFamily, SocketType, sockopt};
    use std::{
        fs::File,
        os::{
            fd::{AsFd, OwnedFd},
            unix::{fs::MetadataExt, net::UnixStream},
        },
    };
    // Original fd0/fd1 remain intentionally inherited; private aliases do not.
    let lock = File::from(private_dup(std::io::stdin().as_fd())?);
    let fd: OwnedFd = private_dup(std::io::stdout().as_fd())?;
    let unix_stream = io(sockopt::socket_type(&fd))? == SocketType::STREAM
        && io(sockopt::socket_domain(&fd))? == AddressFamily::UNIX;
    let peer = io(sockopt::socket_peercred(&fd))?;
    let parent =
        rustix::process::getppid().ok_or(ProvenanceError("private child has no supervisor"))?;
    peer_matches(
        unix_stream,
        peer.pid.as_raw_nonzero().get() as u32,
        peer.uid.as_raw(),
        parent.as_raw_nonzero().get() as u32,
        rustix::process::getuid().as_raw(),
        rustix::process::geteuid().as_raw(),
    )?;
    let own = native::ProcOrigin::capture()?;
    let parent_dir = own.process_directory(parent.as_raw_nonzero().get() as u32)?;
    let parent_token = Token {
        pid: parent.as_raw_nonzero().get() as u32,
        start: super::observation::start_ticks(
            &native::read_at(&parent_dir, "stat", 4096)?,
            parent.as_raw_nonzero().get() as u32,
        )?,
    };
    let executable = own.executable_identity()?;
    let m = lock.metadata()?;
    let guard = ParentGuard {
        own,
        parent: parent_dir,
        parent_token,
        executable,
        lock,
        lock_identity: (m.dev(), m.ino()),
    };
    guard.recheck()?;
    let mut channel = Channel::new(UnixStream::from(fd))?;
    let hello = loop {
        let frames = channel.turn()?;
        ensure(!channel.eof, "supervisor EOF before Hello")?;
        if let Some(frame) = frames.into_iter().next() {
            break frame;
        }
        super::broker::wait_channel(&channel)?;
    };
    ensure(hello.tag == Tag::Hello, "private Hello order")?;
    let projection = Projection::decode(&hello.payload)?;
    ensure(
        projection.parent == parent_token
            && projection.child == guard.own.own_token
            && projection.lock == guard.lock_identity
            && projection.boot.as_slice() == guard.own.boot()
            && projection.invocation == invocation(args)?,
        "private Hello identity mismatch",
    )?;
    super::protocol::elapsed(projection.origin, native::monotonic()?)?;
    guard.recheck()?;
    channel.send(Tag::Ready, hello.payload)?;
    Ok((channel, guard, projection))
}

#[cfg(all(test, target_os = "linux"))]
#[test]
fn operational_fix2_private_aliases_are_cloexec() {
    use std::io::Read;
    use std::os::{fd::AsRawFd, unix::net::UnixStream};
    let (left, mut right) = UnixStream::pair().unwrap();
    let originals = [
        rustix::io::fcntl_getfd(&left).unwrap(),
        rustix::io::fcntl_getfd(&right).unwrap(),
    ];
    assert!(
        originals
            .iter()
            .all(|flags| flags.contains(rustix::io::FdFlags::CLOEXEC))
    );
    let a = private_dup(&left).unwrap();
    let b = private_dup(&right).unwrap();
    assert_ne!(a.as_raw_fd(), left.as_raw_fd());
    assert_ne!(b.as_raw_fd(), right.as_raw_fd());
    for fd in [&a, &b] {
        assert!(
            rustix::io::fcntl_getfd(fd)
                .unwrap()
                .contains(rustix::io::FdFlags::CLOEXEC)
        );
    }
    assert_eq!(rustix::io::fcntl_getfd(&left).unwrap(), originals[0]);
    assert_eq!(rustix::io::fcntl_getfd(&right).unwrap(), originals[1]);
    let status = rustix::fs::fcntl_getfl(&left).unwrap();
    rustix::fs::fcntl_setfl(&a, status | rustix::fs::OFlags::NONBLOCK).unwrap();
    assert!(
        rustix::fs::fcntl_getfl(&left)
            .unwrap()
            .contains(rustix::fs::OFlags::NONBLOCK)
    );
    rustix::fs::fcntl_setfl(&a, status).unwrap();
    assert_eq!(rustix::io::write(&a, b"x").unwrap(), 1);
    let mut byte = [0];
    right.read_exact(&mut byte).unwrap();
    assert_eq!(byte, *b"x");
}
