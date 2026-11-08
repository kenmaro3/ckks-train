//! Safe Linux owned-pidfd primitives. A decoded PID is never a signal authority.
use super::{
    observation,
    protocol::{self, Channel},
    schedule::Token,
};
use crate::study_provenance::{ProvenanceError, Result, ensure};
use rustix::{
    fs::{self, Mode, OFlags},
    process::{self, Pid, PidfdFlags, Signal, WaitId, WaitIdOptions},
};
use std::{
    fs::File,
    io::Read,
    os::{
        fd::{AsFd, OwnedFd},
        unix::fs::MetadataExt,
    },
    path::Path,
    process::{Child, ChildStderr, Command, Stdio},
};
pub(super) fn io<T>(value: rustix::io::Result<T>) -> Result<T> {
    value.map_err(|e| std::io::Error::from(e).into())
}
pub(super) fn monotonic() -> Result<u64> {
    let t = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    protocol::checked_time(t.tv_sec, t.tv_nsec)
}
pub(super) fn prepare_supervisor() -> Result<()> {
    io(process::set_child_subreaper(Some(process::getpid())))?;
    io(process::setpriority_process(None, 10))
}
pub(super) fn nonblocking(fd: &impl AsFd) -> Result<()> {
    let flags = io(fs::fcntl_getfl(fd))?;
    io(fs::fcntl_setfl(fd, flags | OFlags::NONBLOCK))
}
pub(super) fn open_dir(parent: &File, name: &str) -> Result<File> {
    let fd = io(fs::openat(
        parent,
        name,
        OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    ))?;
    Ok(File::from(fd))
}
pub(super) fn read_at(parent: &File, name: &str, cap: usize) -> Result<Vec<u8>> {
    let mut f = File::from(io(fs::openat(
        parent,
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC | OFlags::NONBLOCK,
        Mode::empty(),
    ))?);
    let mut out = Vec::new();
    f.by_ref().take(cap as u64 + 1).read_to_end(&mut out)?;
    ensure(out.len() <= cap, "bounded proc read")?;
    Ok(out)
}
pub(super) struct ProcOrigin {
    root: File,
    boot: Vec<u8>,
    own: File,
    pub own_token: Token,
}
impl ProcOrigin {
    pub fn boot(&self) -> &[u8] {
        &self.boot[..36]
    }
    pub fn process_directory(&self, pid: u32) -> Result<File> {
        self.recheck()?;
        open_dir(&self.root, &pid.to_string())
    }
    pub fn executable_identity(&self) -> Result<(u64, u64)> {
        exe_identity(&self.own)
    }
    pub fn capture() -> Result<Self> {
        let root = File::from(io(fs::open(
            "/proc",
            OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ))?);
        ensure(
            io(fs::fstatfs(&root))?.f_type == fs::PROC_SUPER_MAGIC,
            "actual procfs required",
        )?;
        let boot = read_at(&root, "sys/kernel/random/boot_id", 37)?;
        ensure(boot.len() == 37 && boot[36] == b'\n', "boot UUID bytes")?;
        let pid = std::process::id();
        let own = open_dir(&root, &pid.to_string())?;
        let start = observation::start_ticks(&read_at(&own, "stat", 8 * 1024 * 1024)?, pid)?;
        Ok(Self {
            root,
            boot,
            own,
            own_token: Token { pid, start },
        })
    }
    pub fn recheck(&self) -> Result<()> {
        ensure(
            std::process::id() == self.own_token.pid,
            "supervisor process changed",
        )?;
        ensure(
            read_at(&self.root, "sys/kernel/random/boot_id", 37)? == self.boot,
            "boot changed",
        )?;
        ensure(
            observation::start_ticks(
                &read_at(&self.own, "stat", 8 * 1024 * 1024)?,
                self.own_token.pid,
            )? == self.own_token.start,
            "supervisor start changed",
        )
    }
    pub fn own_rss(&self) -> Result<u64> {
        self.recheck()?;
        let rss = observation::rss_kib(&read_at(&self.own, "status", 8 * 1024 * 1024)?)?;
        self.recheck()?;
        Ok(rss)
    }
    pub fn children(&self, pid: u32) -> Result<Vec<u32>> {
        let directory = open_dir(&self.root, &pid.to_string())?;
        let data = read_at(&directory, &format!("task/{pid}/children"), 8 * 1024 * 1024)?;
        let text = std::str::from_utf8(&data).map_err(|_| ProvenanceError("child list UTF8"))?;
        let mut result = Vec::new();
        for value in text.split_whitespace() {
            ensure(result.len() < 65536, "child list cap")?;
            result.push(
                value
                    .parse()
                    .map_err(|_| ProvenanceError("child list PID"))?,
            );
        }
        Ok(result)
    }
}
pub(super) fn exe_identity(proc: &File) -> Result<(u64, u64)> {
    // The literal kernel exe link is intentionally followed, never caller input.
    let exe = File::from(io(fs::openat(
        proc,
        "exe",
        OFlags::RDONLY | OFlags::CLOEXEC,
        Mode::empty(),
    ))?);
    let m = exe.metadata()?;
    ensure(m.is_file(), "process executable regular file")?;
    Ok((m.dev(), m.ino()))
}
pub(super) struct KernelChild {
    pidfd: OwnedFd,
    proc: File,
    proc_identity: (u64, u64),
    pub token: Token,
    pub reaped: bool,
    waitable: bool,
}
pub(super) struct OwnedChild {
    _child: Child,
    kernel: KernelChild,
    pub channel: Channel,
    pub stderr: ChildStderr,
}
impl std::ops::Deref for OwnedChild {
    type Target = KernelChild;
    fn deref(&self) -> &KernelChild {
        &self.kernel
    }
}
impl std::ops::DerefMut for OwnedChild {
    fn deref_mut(&mut self) -> &mut KernelChild {
        &mut self.kernel
    }
}
impl OwnedChild {
    #[allow(clippy::result_large_err)] // Keep spawned Child and acquired handles on stack: boxing here adds a post-spawn allocation failure.
    pub fn spawn(
        executable: &Path,
        mode: &str,
        arguments: &[std::ffi::OsString],
        lock: &File,
        origin: &ProcOrigin,
    ) -> std::result::Result<Self, SpawnFailure> {
        // Everything that can be prepared before spawn is prepared first.
        let prepared = (|| -> Result<_> {
            ensure(
                arguments.len() + 2 <= 256
                    && arguments.iter().all(|a| {
                        use std::os::unix::ffi::OsStrExt;
                        a.as_bytes().len() <= 4096 && !a.as_bytes().contains(&0)
                    }),
                "child argv cap",
            )?;
            origin.recheck()?;
            let (parent, socket) = std::os::unix::net::UnixStream::pair()?;
            Ok((
                Channel::new(parent)?,
                OwnedFd::from(socket),
                File::from(super::handshake::private_dup(lock)?),
            ))
        })()
        .map_err(|error| SpawnFailure {
            error,
            partial: None,
        })?;
        let (channel, socket, held) = prepared;
        let child = Command::new(executable)
            .arg(mode)
            .args(arguments)
            .stdin(Stdio::from(held))
            .stdout(Stdio::from(socket))
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| SpawnFailure {
                error: error.into(),
                partial: None,
            })?;
        // From this exact point every error owns the spawned Child and every
        // acquired handle. Dropping a std Child is never counted as termination.
        let mut partial = PartialChild {
            identity: (0, 0),
            child: Some(child),
            pidfd: None,
            proc: None,
            channel: Some(channel),
            stderr: None,
            token: None,
            progress: super::schedule::SpawnProgress::spawned(),
            reaped: false,
        };
        if let Err(error) = partial.bind(origin) {
            partial.progress.fail();
            return Err(SpawnFailure {
                error,
                partial: Some(partial),
            });
        }
        let proc = partial.proc.take().expect("bound proc");
        // No fallible operation is allowed while moving complete custody.
        let proc_identity = partial.identity;
        Ok(Self {
            _child: partial.child.take().expect("spawned child"),
            kernel: KernelChild {
                pidfd: partial.pidfd.take().expect("bound pidfd"),
                proc,
                proc_identity,
                token: partial.token.expect("bound token"),
                reaped: false,
                waitable: true,
            },
            channel: partial.channel.take().expect("bound channel"),
            stderr: partial.stderr.take().expect("bound diagnostics"),
        })
    }
}
impl KernelChild {
    pub fn rights(&self) -> [std::os::fd::BorrowedFd<'_>; 2] {
        [self.pidfd.as_fd(), self.proc.as_fd()]
    }
    pub fn parent_pid(&self) -> Result<u32> {
        observation::pid_field(&read_at(&self.proc, "status", 4096)?, b"PPid:")
    }
    pub fn nice(&self) -> Result<i32> {
        io(process::getpriority_process(Pid::from_raw(
            self.token.pid as i32,
        )))
    }
    fn identity(&self, origin: &ProcOrigin) -> Result<()> {
        use std::os::fd::AsRawFd;
        origin.recheck()?;
        let metadata = self.proc.metadata()?;
        ensure(
            metadata.is_dir()
                && (metadata.dev(), metadata.ino()) == self.proc_identity
                && io(fs::fstatfs(&self.proc))?.f_type == fs::PROC_SUPER_MAGIC,
            "held child proc identity/type",
        )?;
        ensure(
            observation::start_ticks(&read_at(&self.proc, "stat", 4096)?, self.token.pid)?
                == self.token.start,
            "child start identity",
        )?;
        let info = read_at(
            &origin.own,
            &format!("fdinfo/{}", self.pidfd.as_raw_fd()),
            4096,
        )?;
        ensure(
            observation::pid_field(&info, b"Pid:")? == self.token.pid,
            "acquired pidfd target identity",
        )?;
        origin.recheck()
    }
    /// Validation errors return every received handle to the private probe
    /// custodian. No malformed transfer becomes an empty child ledger.
    pub fn received(
        pidfd: OwnedFd,
        proc: File,
        token: Token,
        parent: u32,
        origin: &ProcOrigin,
    ) -> std::result::Result<Self, ReceivedKernelFailure> {
        let checked = (|| -> Result<(u64, u64)> {
            let m = proc.metadata()?;
            let identity = (m.dev(), m.ino());
            ensure(
                m.is_dir() && io(fs::fstatfs(&proc))?.f_type == fs::PROC_SUPER_MAGIC,
                "transferred proc type",
            )?;
            let fresh = origin.process_directory(token.pid)?;
            let f = fresh.metadata()?;
            ensure(
                identity == (f.dev(), f.ino()),
                "transferred proc wrong process",
            )?;
            ensure(
                observation::start_ticks(&read_at(&proc, "stat", 4096)?, token.pid)? == token.start
                    && observation::pid_field(&read_at(&proc, "status", 4096)?, b"PPid:")?
                        == parent,
                "transferred leaf start/parent",
            )?;
            ensure(
                exe_identity(&proc)? == origin.executable_identity()?,
                "transferred leaf executable",
            )?;
            use std::os::fd::AsRawFd;
            let info = read_at(&origin.own, &format!("fdinfo/{}", pidfd.as_raw_fd()), 4096)?;
            ensure(
                observation::pid_field(&info, b"Pid:")? == token.pid,
                "transferred pidfd wrong target/type",
            )?;
            ensure(
                matches!(
                    process::waitid(
                        WaitId::PidFd(pidfd.as_fd()),
                        WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT
                    ),
                    Err(rustix::io::Errno::CHILD)
                ),
                "pre-adoption ECHILD required; never an exit",
            )?;
            origin.recheck()?;
            Ok(identity)
        })();
        match checked {
            Ok(proc_identity) => Ok(Self {
                pidfd,
                proc,
                proc_identity,
                token,
                reaped: false,
                waitable: false,
            }),
            Err(error) => Err(ReceivedKernelFailure { error, pidfd, proc }),
        }
    }
    pub fn confirm_adoption(&mut self, origin: &ProcOrigin) -> Result<()> {
        ensure(!self.waitable && !self.reaped, "duplicate native adoption")?;
        self.identity(origin)?;
        ensure(
            self.parent_pid()? == origin.own_token.pid,
            "actual orphan adoption not observed",
        )?;
        self.waitable = true;
        Ok(())
    }
    pub fn status(&self) -> Result<Option<u64>> {
        ensure(
            self.waitable && !self.reaped,
            "wait before adoption or after reap",
        )?;
        let status = io(process::waitid(
            WaitId::PidFd(self.pidfd.as_fd()),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
        ))?;
        status
            .map(|s| {
                if let Some(code) = s.exit_status() {
                    Ok((code as u64) * 2)
                } else if let Some(signal) = s.terminating_signal() {
                    Ok((signal as u64) * 2 + 1)
                } else {
                    Err(ProvenanceError("nonterminal pidfd wait status"))
                }
            })
            .transpose()
    }
    pub fn recheck_live(&self, origin: &ProcOrigin) -> Result<bool> {
        origin.recheck()?;
        if self.status()?.is_some() {
            return Ok(false);
        }
        self.identity(origin)?;
        Ok(self.status()?.is_none())
    }
    pub fn rss(&self, origin: &ProcOrigin) -> Result<Option<u64>> {
        if !self.recheck_live(origin)? {
            return Ok(None);
        }
        let rss = observation::rss_kib(&read_at(&self.proc, "status", 8 * 1024 * 1024)?)?;
        if !self.recheck_live(origin)? {
            return Ok(None);
        }
        Ok(Some(rss))
    }
    pub fn signal(&self, origin: &ProcOrigin, kill: bool) -> Result<bool> {
        if !self.recheck_live(origin)? {
            return Ok(false);
        }
        io(process::pidfd_send_signal(
            &self.pidfd,
            if kill { Signal::KILL } else { Signal::TERM },
        ))?;
        Ok(true)
    }
    pub fn signal_for_probe(&self, origin: &ProcOrigin, middle: u32, kill: bool) -> Result<bool> {
        // Authenticated transferred custody only; non-child wait is forbidden,
        // but the acquired pidfd can still stop L while M cleanup is pending.
        self.identity(origin)?;
        ensure(
            [middle, origin.own_token.pid].contains(&self.parent_pid()?),
            "native cleanup transferred parent",
        )?;
        let mut fd = [rustix::event::PollFd::new(
            &self.pidfd,
            rustix::event::PollFlags::IN,
        )];
        io(rustix::event::poll(
            &mut fd,
            Some(&rustix::time::Timespec {
                tv_sec: 0,
                tv_nsec: 0,
            }),
        ))?;
        ensure(
            !fd[0]
                .revents()
                .intersects(rustix::event::PollFlags::ERR | rustix::event::PollFlags::NVAL),
            "native cleanup pidfd poll",
        )?;
        if fd[0].revents().intersects(rustix::event::PollFlags::IN) {
            return Ok(false);
        }
        io(process::pidfd_send_signal(
            &self.pidfd,
            if kill { Signal::KILL } else { Signal::TERM },
        ))?;
        Ok(true)
    }
    pub fn rss_for_probe(&self, origin: &ProcOrigin, parents: &[u32]) -> Result<Option<u64>> {
        // A received pidfd can be polled before adoption even though waitid must
        // return ECHILD. This never grants wait/reap or normal-worker authority.
        self.identity(origin)?;
        ensure(
            parents.contains(&self.parent_pid()?),
            "native transferred RSS parent",
        )?;
        let exited = || -> Result<bool> {
            let mut fd = [rustix::event::PollFd::new(
                &self.pidfd,
                rustix::event::PollFlags::IN,
            )];
            io(rustix::event::poll(
                &mut fd,
                Some(&rustix::time::Timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                }),
            ))?;
            ensure(
                !fd[0]
                    .revents()
                    .intersects(rustix::event::PollFlags::ERR | rustix::event::PollFlags::NVAL),
                "native pidfd poll identity error",
            )?;
            Ok(fd[0].revents().intersects(rustix::event::PollFlags::IN))
        };
        if exited()? {
            return Ok(None);
        }
        let rss = observation::rss_kib(&read_at(&self.proc, "status", 4096)?)?;
        self.identity(origin)?;
        if exited()? { Ok(None) } else { Ok(Some(rss)) }
    }
    pub fn stop_for_probe(&self, origin: &ProcOrigin) -> Result<()> {
        ensure(
            self.recheck_live(origin)?,
            "native STOP target already exited",
        )?;
        io(process::pidfd_send_signal(&self.pidfd, Signal::STOP))
    }
    pub fn stopped_for_probe(&self, origin: &ProcOrigin) -> Result<bool> {
        ensure(self.recheck_live(origin)?, "native STOP target exited")?;
        observation::stopped(&read_at(&self.proc, "status", 4096)?)
    }
    pub fn reap(&mut self) -> Result<u64> {
        let status = self
            .status()?
            .ok_or(ProvenanceError("reap before confirmed exit"))?;
        ensure(
            io(process::waitid(
                WaitId::PidFd(self.pidfd.as_fd()),
                WaitIdOptions::EXITED | WaitIdOptions::NOHANG,
            ))?
            .is_some(),
            "pidfd reap unavailable",
        )?;
        self.reaped = true;
        Ok(status)
    }
}

/// A post-spawn failure is not an ordinary empty spawn error.
pub(super) struct SpawnFailure {
    pub error: ProvenanceError,
    pub partial: Option<PartialChild>,
}
pub(super) struct PartialChild {
    child: Option<Child>,
    pidfd: Option<OwnedFd>,
    proc: Option<File>,
    channel: Option<Channel>,
    stderr: Option<ChildStderr>,
    token: Option<Token>,
    identity: (u64, u64),
    progress: super::schedule::SpawnProgress,
    reaped: bool,
}
impl PartialChild {
    fn bind(&mut self, origin: &ProcOrigin) -> Result<()> {
        let child = self
            .child
            .as_mut()
            .ok_or(ProvenanceError("partial child missing"))?;
        let pid =
            Pid::from_raw(child.id() as i32).ok_or(ProvenanceError("child PID representation"))?;
        self.pidfd = Some(io(process::pidfd_open(pid, PidfdFlags::empty()))?);
        self.progress.advance()?;
        let proc = open_dir(&origin.root, &child.id().to_string())?;
        let m = proc.metadata()?;
        self.identity = (m.dev(), m.ino());
        self.proc = Some(proc);
        self.progress.advance()?;
        let start = observation::start_ticks(
            &read_at(
                self.proc.as_ref().expect("held proc"),
                "stat",
                8 * 1024 * 1024,
            )?,
            child.id(),
        )?;
        self.token = Some(Token {
            pid: child.id(),
            start,
        });
        origin.recheck()?;
        self.progress.advance()?;
        self.stderr = Some(
            child
                .stderr
                .take()
                .ok_or(ProvenanceError("child diagnostic pipe"))?,
        );
        nonblocking(self.stderr.as_ref().expect("held diagnostics"))?;
        self.progress.advance()?;
        ensure(self.progress.ready(), "partial spawn cannot register")
    }
    /// EOF permits the waiting trampoline to stop; no PID-derived signal is used.
    pub fn request_boundary_stop(&mut self) {
        self.channel.take();
    }
    /// Only an actually acquired pidfd can establish exit/reap. Missing pidfd
    /// remains unresolved even if a discovered numeric PID later disappears.
    pub fn reap_confirmed(&mut self) -> Result<bool> {
        if self.reaped {
            return Ok(true);
        }
        let Some(fd) = self.pidfd.as_ref() else {
            return Ok(false);
        };
        let status = io(process::waitid(
            WaitId::PidFd(fd.as_fd()),
            WaitIdOptions::EXITED | WaitIdOptions::NOHANG | WaitIdOptions::NOWAIT,
        ))?;
        if status.is_none() {
            return Ok(false);
        }
        ensure(
            io(process::waitid(
                WaitId::PidFd(fd.as_fd()),
                WaitIdOptions::EXITED | WaitIdOptions::NOHANG,
            ))?
            .is_some(),
            "partial child confirmed reap",
        )?;
        self.reaped = true;
        Ok(true)
    }
    pub fn unresolved(&self) -> bool {
        self.progress.unresolved()
    }
}

/// Kept by the native probe on malformed/partial handoff; never signal from token.
pub(super) struct ReceivedKernelFailure {
    pub error: ProvenanceError,
    pub pidfd: OwnedFd,
    pub proc: File,
}
