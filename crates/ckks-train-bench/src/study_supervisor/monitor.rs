//! The real bounded poll/sampler and owned-child ledger shared by launch paths.
use super::{
    handshake::Projection,
    native::{self, OwnedChild, PartialChild, ProcOrigin},
    protocol::{self, Frame, Tag},
    schedule::{Action, Event, Reason, Scheduler, Token},
};
use crate::{
    study_provenance::{ProvenanceError, Result, ensure, publication::OperationalStream},
    study_reference::operational as wire,
};
use std::{ffi::OsString, fs::File, io::Read, os::unix::fs::MetadataExt, path::Path};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Worker,
    Python,
    Git,
}
pub(super) struct Tracked {
    pub child: OwnedChild,
    pub kind: Kind,
    pub hello: Projection,
    pub phase: u8,
    pub raw: Vec<u8>,
    pub result: Option<Vec<u8>>,
    pub notified_exit: bool,
    pub drain: protocol::DrainState,
}
pub(super) enum Message {
    Frame(Token, Frame),
    Exited(Token, u64),
    ControlLost {
        token: Token,
        observed: u64,
        stderr_eof: bool,
        write_pending: bool,
    },
}
pub(super) struct CleanupTiming {
    fallback: super::schedule::FallbackClock,
    started: Option<std::time::Instant>,
    pub failed: bool,
}
impl CleanupTiming {
    pub fn new() -> Self {
        Self {
            fallback: super::schedule::FallbackClock::new(),
            started: None,
            failed: false,
        }
    }
    pub fn armed(&self) -> bool {
        self.started.is_some()
    }
}
pub(super) struct Monitor {
    pub process: ProcOrigin,
    pub origin: u64,
    pub scheduler: Scheduler,
    pub children: Vec<Tracked>,
    pub partial: Vec<PartialChild>,
    pub lock: File,
    pub samples: Option<OperationalStream>,
    pub diagnostics: Option<OperationalStream>,
    pub probe_samples: Option<Vec<wire::SampleValues>>,
    pub probe_diagnostics: Option<protocol::DiagnosticCapture>,
    pub probe_signals: Vec<(Token, bool, bool, Option<u64>)>,
    pub sequence: u64,
    pub initialized: bool,
    pub publication_failed: bool,
    pub cleanup_io_failed: bool,
    pub unresolved: bool,
    pub clock_failure: super::schedule::ClockFailureLatch,
    probe_fault: Option<super::clock_fault::Fault>,
}
pub(super) fn unix_ns() -> Result<u64> {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_| ProvenanceError("wall clock before epoch"))?
        .as_nanos();
    u64::try_from(n).map_err(|_| ProvenanceError("wall clock overflow"))
}
impl Monitor {
    pub fn new(lock: File, output: &File, origin: u64, process: ProcOrigin) -> Result<Self> {
        Self::base(
            lock,
            origin,
            process,
            Some(OperationalStream::new(
                output,
                "diagnostic.bin",
                8 * 1024 * 1024,
            )?),
            None,
        )
    }
    pub fn new_probe(lock: File, origin: u64, process: ProcOrigin) -> Result<Self> {
        Self::base(
            lock,
            origin,
            process,
            None,
            Some(protocol::DiagnosticCapture::new()),
        )
    }
    fn base(
        lock: File,
        origin: u64,
        process: ProcOrigin,
        diagnostics: Option<OperationalStream>,
        probe_diagnostics: Option<protocol::DiagnosticCapture>,
    ) -> Result<Self> {
        process.recheck()?;
        protocol::elapsed(origin, native::monotonic()?)?;
        let mut scheduler = if probe_diagnostics.is_some() {
            Scheduler::new_probe()
        } else {
            Scheduler::new()
        };
        scheduler.event(Event::Pending, 0)?;
        Ok(Self {
            process,
            origin,
            scheduler,
            children: Vec::new(),
            partial: Vec::new(),
            lock,
            samples: None,
            diagnostics,
            probe_samples: probe_diagnostics.as_ref().map(|_| Vec::new()),
            probe_diagnostics,
            probe_signals: Vec::new(),
            sequence: 0,
            initialized: false,
            publication_failed: false,
            cleanup_io_failed: false,
            unresolved: false,
            clock_failure: super::schedule::ClockFailureLatch::new(),
            probe_fault: None,
        })
    }
    pub fn probe_clock_fault(&mut self, after_terms: bool) -> Result<()> {
        ensure(
            self.probe_diagnostics.is_some() && self.probe_fault.is_none(),
            "native clock fault boundary",
        )?;
        self.probe_fault = Some(super::clock_fault::Fault::new(after_terms));
        Ok(())
    }
    pub fn probe_clock_fault_consumed(&self) -> bool {
        self.probe_fault.as_ref().is_some_and(|f| f.consumed() == 1)
    }
    pub fn elapsed(&self) -> Result<u64> {
        super::clock_fault::checked(
            self.origin,
            native::monotonic(),
            self.probe_fault.as_ref(),
            &self.clock_failure,
        )
    }
    pub fn clock(&self) -> Result<(u64, u64)> {
        Ok((unix_ns()?, self.elapsed()?))
    }
    pub fn event(&mut self, event: Event) -> Result<()> {
        let actions = self.scheduler.event(event, self.elapsed()?)?;
        self.actions(actions)
    }
    pub(super) fn actions(&mut self, actions: Vec<Action>) -> Result<()> {
        let has_terms = actions
            .iter()
            .any(|action| matches!(action, Action::Term(_)));
        let failed = super::schedule::apply_all(actions, |action| {
            match action {
                Action::Term(t) | Action::Kill(t) => {
                    let kill = matches!(action, Action::Kill(_));
                    let child = self
                        .children
                        .iter_mut()
                        .find(|c| c.child.token == t)
                        .ok_or(ProvenanceError("stop target absent from owned ledger"))?;
                    let issued = child.child.signal(&self.process, kill)?;
                    if self.probe_diagnostics.is_some() {
                        // Timestamp follows the actual syscall. Failure stays absent
                        // and latches invalidity; no pre-attempt time is substituted.
                        let at = self.elapsed().ok();
                        ensure(self.probe_signals.len() < 32, "native signal event cap")?;
                        self.probe_signals
                            .try_reserve(1)
                            .map_err(|_| ProvenanceError("native signal event allocation"))?;
                        self.probe_signals.push((t, kill, issued, at));
                    }
                    Ok(())
                }
                Action::Reap(t) => {
                    let index = self
                        .children
                        .iter()
                        .position(|c| c.child.token == t)
                        .ok_or(ProvenanceError("reap ledger missing"))?;
                    self.children[index].child.reap()?;
                    self.children.remove(index);
                    Ok(())
                }
                Action::Run(_) | Action::StopBoundary | Action::PublishClosed => Ok(()),
            }
        });
        let mut first = None;
        if has_terms && self.scheduler.term_waiting() {
            if let Some(fault) = &self.probe_fault {
                fault.after_terms();
            }
            let completion = self
                .elapsed()
                .and_then(|now| self.scheduler.term_batch_completed(now));
            if let Err(error) = completion {
                self.clock_failure.failed();
                self.unresolved = true;
                first = Some(error);
            }
        }
        for (action, error) in failed {
            self.unresolved = true;
            if let Action::Term(t) | Action::Kill(t) = action {
                let now = self.elapsed()?;
                let _ = self.scheduler.event(Event::IdentityLost(t), now);
            }
            if first.is_none() {
                first = Some(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
    pub fn spawn(
        &mut self,
        kind: Kind,
        exe: &Path,
        mode: &str,
        args: &[OsString],
        executable: &str,
    ) -> Result<Token> {
        ensure(
            self.children.len() + self.partial.len() < 3 && !self.unresolved,
            "four process custody cap",
        )?;
        self.due()?;
        let metadata = self.lock.metadata()?;
        let mut invocation = vec![OsString::from(mode)];
        invocation.extend_from_slice(args);
        let invocation = super::handshake::invocation(&invocation)?;
        let boot = self
            .process
            .boot()
            .try_into()
            .map_err(|_| ProvenanceError("monitor boot UUID"))?;
        let executable = executable
            .as_bytes()
            .try_into()
            .map_err(|_| ProvenanceError("monitor executable hash"))?;
        self.children
            .try_reserve_exact(1)
            .map_err(|_| ProvenanceError("child ledger allocation before spawn"))?;
        self.due()?;
        ensure(
            self.scheduler.reason().is_none()
                && !self.unresolved
                && self.children.len() + self.partial.len() < 3,
            "spawn stopped by due sample or unresolved custody",
        )?;
        let reservation = self.scheduler.reserve()?;
        let child = match OwnedChild::spawn(exe, mode, args, &self.lock, &self.process) {
            Ok(child) => child,
            Err(mut failure) => {
                reservation.cancel();
                if let Some(mut partial) = failure.partial.take() {
                    partial.request_boundary_stop();
                    self.partial.push(partial);
                    self.unresolved = true;
                    self.event(Event::UnknownChild)?;
                }
                return Err(failure.error);
            }
        };
        let token = child.token;
        let hello = Projection {
            parent: self.process.own_token,
            child: token,
            lock: (metadata.dev(), metadata.ino()),
            origin: self.origin,
            boot,
            executable,
            invocation,
        };
        // Capacity and exclusive scheduler slot were reserved before spawn.
        // Retention and commit contain no fallible operation or clock read.
        self.children.push(Tracked {
            child,
            kind,
            hello,
            phase: 0,
            raw: Vec::new(),
            result: None,
            notified_exit: false,
            drain: protocol::DrainState::new(),
        });
        reservation.commit(token);
        self.check_descendants()?;
        Ok(token)
    }
    pub fn hello(&mut self, token: Token) -> Result<()> {
        let c = self.child_mut(token)?;
        ensure(c.phase == 0, "duplicate Hello")?;
        c.child
            .channel
            .send(Tag::Hello, c.hello.encode().to_vec())?;
        c.phase = 1;
        Ok(())
    }
    pub fn child_mut(&mut self, token: Token) -> Result<&mut Tracked> {
        self.children
            .iter_mut()
            .find(|c| c.child.token == token)
            .ok_or(ProvenanceError("unowned control target"))
    }
    pub fn check_descendants(&mut self) -> Result<()> {
        self.check_descendants_with(&[])
    }
    fn check_descendants_with(&mut self, borrowed: &[Tracked]) -> Result<()> {
        let result = (|| -> Result<()> {
            let known = self
                .children
                .iter()
                .chain(borrowed)
                .map(|c| c.child.token.pid)
                .collect::<Vec<_>>();
            if self
                .process
                .children(self.process.own_token.pid)?
                .iter()
                .any(|p| !known.contains(p))
            {
                return Err(ProvenanceError("unregistered supervisor child"));
            }
            for c in self.children.iter().chain(borrowed) {
                if c.child.status()?.is_some() {
                    continue;
                }
                let descendants = self.process.children(c.child.token.pid)?;
                if descendants.iter().any(|p| !known.contains(p)) {
                    return Err(ProvenanceError("unregistered descendant"));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.unresolved = true;
            let _ = self.event(Event::UnknownChild);
        }
        result
    }
    pub fn final_sample(&mut self) -> Result<wire::SampleValues> {
        let sample = self.sample(4, false);
        super::completion::final_sample_result(sample, &mut self.unresolved)
    }
    pub fn sample(&mut self, phase: u8, scheduled: bool) -> Result<wire::SampleValues> {
        self.sample_with(phase, scheduled, &[])
    }
    pub fn probe_due_with(&mut self, inner: &Monitor) -> Result<()> {
        ensure(
            self.probe_diagnostics.is_some()
                && inner.probe_diagnostics.is_some()
                && self.origin == inner.origin
                && self.process.own_token == inner.process.own_token
                && self.children.is_empty()
                && inner.children.len() <= 1
                && inner
                    .children
                    .iter()
                    .all(|c| self.scheduler.live().contains(&c.child.token)),
            "native borrowed custody boundary",
        )?;
        let a = self.lock.metadata()?;
        let b = inner.lock.metadata()?;
        ensure(
            a.dev() == b.dev() && a.ino() == b.ino(),
            "native borrowed lock identity",
        )?;
        if self.elapsed()? >= self.scheduler.due() {
            let sample = self.sample_with(0, true, &inner.children)?;
            if !sample.valid || sample.status != 0 {
                self.unresolved = true;
                return Err(ProvenanceError("native outer borrowed sample invalid"));
            }
        }
        Ok(())
    }
    fn sample_with(
        &mut self,
        phase: u8,
        scheduled: bool,
        borrowed: &[Tracked],
    ) -> Result<wire::SampleValues> {
        let started = self.clock()?;
        let mut entries = Vec::new();
        let capture = (|| -> Result<u64> {
            self.check_descendants_with(borrowed)?;
            entries.push((
                u64::from(self.process.own_token.pid),
                self.process.own_token.start,
                self.process.own_rss()?,
            ));
            for c in self.children.iter().chain(borrowed) {
                if let Some(rss) = c.child.rss(&self.process)? {
                    entries.push((u64::from(c.child.token.pid), c.child.token.start, rss));
                }
            }
            entries.sort_unstable_by_key(|p| p.0);
            entries.iter().try_fold(0u64, |n, p| {
                n.checked_add(p.2)
                    .ok_or(ProvenanceError("owned RSS sum overflow"))
            })
        })();
        let finished = self.clock()?;
        let rss = capture.ok();
        let deadline = if self.initialized {
            self.scheduler.due().checked_add(super::schedule::INTERVAL)
        } else {
            Some(super::schedule::INTERVAL)
        }
        .ok_or(ProvenanceError("sample deadline overflow"))?;
        let (valid, status) =
            super::completion::assessment(rss, started.1, finished.1, deadline, phase);
        let event = if scheduled {
            Event::Sample {
                started: started.1,
                finished: finished.1,
                rss,
            }
        } else {
            Event::PhaseSample {
                started: started.1,
                finished: finished.1,
                rss,
            }
        };
        self.event(event)?;
        let values = wire::SampleValues {
            sequence: self.sequence,
            started,
            finished,
            phase,
            processes: entries,
            valid,
            status,
        };
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(ProvenanceError("sample sequence exhausted"))?;
        self.initialized = true;
        if let Some(stream) = &mut self.samples {
            let raw =
                wire::sample_frame(&values).map_err(|_| ProvenanceError("sample wire encoding"))?;
            if let Err(e) = stream.append(&raw) {
                self.publication_failed = true;
                self.event(Event::PublicationFailed)?;
                return Err(e);
            }
        }
        if let Some(samples) = &mut self.probe_samples {
            ensure(samples.len() < 4096, "probe sample count cap")?;
            samples
                .try_reserve(1)
                .map_err(|_| ProvenanceError("probe sample allocation"))?;
            samples.push(values.clone());
        }
        Ok(values)
    }
    pub fn due(&mut self) -> Result<()> {
        if self.initialized && self.elapsed()? >= self.scheduler.due() {
            self.sample(0, true)?;
        }
        self.event(Event::Tick)
    }
    pub fn pump(&mut self) -> Result<Vec<Message>> {
        // First runnable turn services a due sample before any IPC/diagnostics.
        self.due()?;
        if self.scheduler.reason().is_some() || self.unresolved {
            return Ok(Vec::new());
        }
        let mut messages = Vec::new();
        for index in 0..self.children.len() {
            let c = &mut self.children[index];
            let mut scratch = [0; 65536];
            match c.child.stderr.read(&mut scratch) {
                Ok(n) if n > 0 => {
                    if let Err(error) = diagnostic(
                        &mut self.diagnostics,
                        &mut self.probe_diagnostics,
                        c.child.token,
                        &scratch[..n],
                    ) {
                        self.publication_failed = true;
                        return Err(error);
                    }
                }
                Ok(_) => c.drain.stderr_eof = true,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.into()),
            }
            if c.phase == 4 && c.kind == Kind::Git {
                match c.child.channel.stream.read(&mut scratch) {
                    Ok(0) => c.drain.stdout_eof = true,
                    Ok(n) => {
                        c.drain.stdout_bytes(n)?;
                        c.raw.extend_from_slice(&scratch[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e.into()),
                }
            } else {
                for frame in c.child.channel.turn()? {
                    messages.push(Message::Frame(c.child.token, frame));
                }
                c.drain.stdout_eof = c.child.channel.eof;
            }
            if c.drain.status.is_none() {
                c.drain.status = c.child.status()?;
            }
            let control = protocol::ControlObservation {
                kind: match c.kind {
                    Kind::Worker => 0,
                    Kind::Python => 1,
                    Kind::Git => 2,
                },
                phase: c.phase,
                result_present: c.result.is_some(),
                write_pending: c.child.channel.pending(),
                eof: c.drain.stdout_eof,
                status: c.drain.status,
            };
            let token = c.child.token;
            let stderr_eof = c.drain.stderr_eof;
            if control.lost() {
                // Capture the actual live EOF observation before any TERM. A
                // failed action retains custody; no more collected IPC is run.
                let observed = self.elapsed()?;
                let actions = self.scheduler.observe_control(token, control, observed)?;
                self.actions(actions)?;
                return Ok(vec![Message::ControlLost {
                    token,
                    observed,
                    stderr_eof,
                    write_pending: control.write_pending,
                }]);
            }
            if !c.notified_exit && c.drain.ready() {
                c.notified_exit = true;
                messages.push(Message::Exited(
                    c.child.token,
                    c.drain.status.expect("confirmed exit"),
                ));
            }
        }
        for p in &mut self.partial {
            let _ = p.reap_confirmed()?;
            ensure(p.unresolved(), "partial custody unexpectedly cleared")?;
        }
        Ok(messages)
    }
    pub fn pulse_waiting(&mut self) -> Result<()> {
        let events = self.pump()?;
        ensure(
            events.is_empty() && self.scheduler.reason().is_none(),
            "pre-Run child loss/control or monitor stop",
        )
    }
    pub fn drain_stopping(&mut self, common_clock_valid: bool) -> Vec<(Token, u64)> {
        // A malformed/truncated control stream or exhausted output cap must not
        // prevent grace/KILL or drain/reap of other actually owned children.
        if common_clock_valid {
            if self.due().is_err() {
                self.publication_failed = true;
                self.cleanup_io_failed = true;
            }
            if self.event(Event::Tick).is_err() {
                self.unresolved = true;
                self.cleanup_io_failed = true;
            }
        }
        let mut exits = Vec::new();
        for c in &mut self.children {
            let mut scratch = [0; 65536];
            if !c.drain.stderr_eof {
                match c.child.stderr.read(&mut scratch) {
                    Ok(0) => c.drain.stderr_eof = true,
                    Ok(n) => {
                        if diagnostic(
                            &mut self.diagnostics,
                            &mut self.probe_diagnostics,
                            c.child.token,
                            &scratch[..n],
                        )
                        .is_err()
                        {
                            self.publication_failed = true;
                            self.cleanup_io_failed = true;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(_) => {
                        self.unresolved = true;
                        self.cleanup_io_failed = true;
                    }
                }
            }
            if !c.drain.stdout_eof {
                match c.child.channel.stream.read(&mut scratch) {
                    Ok(0) => c.drain.stdout_eof = true,
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(_) => {
                        self.unresolved = true;
                        self.cleanup_io_failed = true;
                    }
                }
            }
            if c.drain.status.is_none() {
                match c.child.status() {
                    Ok(status) => c.drain.status = status,
                    Err(_) => {
                        self.unresolved = true;
                        self.cleanup_io_failed = true;
                    }
                }
            }
            if c.drain.ready() {
                exits.push((c.child.token, c.drain.status.expect("confirmed exit")));
            }
        }
        exits
    }
    fn stop_owned_result(&mut self, kill: bool) -> Result<()> {
        let actions = self
            .children
            .iter()
            .map(|c| {
                if kill {
                    Action::Kill(c.child.token)
                } else {
                    Action::Term(c.child.token)
                }
            })
            .collect();
        self.actions(actions)
    }
    pub fn cleanup_turn(&mut self, timing: &mut CleanupTiming) -> Vec<(Token, u64)> {
        if timing
            .fallback
            .should_start(&self.clock_failure, self.elapsed().is_ok())
        {
            self.unresolved = true;
            if self.stop_owned_result(false).is_err() {
                timing.failed = true;
            }
            if timing.fallback.arm().is_err() {
                timing.failed = true;
            }
            // Always after all legal TERM attempts, once, without guessed remainder.
            timing.started = Some(std::time::Instant::now());
        }
        if timing.fallback.kill_due(
            timing
                .started
                .and_then(|s| std::time::Instant::now().checked_duration_since(s)),
        ) && self.stop_owned_result(true).is_err()
        {
            timing.failed = true;
        }
        let exits = self.drain_stopping(!timing.armed());
        let mut reaped = Vec::new();
        for (token, status) in exits {
            if self.event(Event::Exited(token)).is_err() {
                timing.failed = true;
                self.reap_unresolved(token);
            }
            if !self.children.iter().any(|c| c.child.token == token) {
                reaped.push((token, status));
            }
        }
        if self.wait_stopping().is_err() {
            timing.failed = true;
            self.unresolved = true;
            std::thread::park_timeout(std::time::Duration::from_millis(100));
        }
        timing.failed |= self.cleanup_io_failed;
        if timing.failed {
            self.unresolved = true;
        }
        reaped
    }
    pub fn reap_unresolved(&mut self, token: Token) {
        self.unresolved = true;
        if let Some(index) = self.children.iter().position(|c| c.child.token == token) {
            if self.children[index].child.reap().is_ok() {
                self.children.remove(index);
            }
        }
    }
    pub fn wait(&self) -> Result<()> {
        self.wait_mode(false)
    }
    pub fn wait_stopping(&self) -> Result<()> {
        self.wait_mode(true)
    }
    fn wait_mode(&self, stopping: bool) -> Result<()> {
        use rustix::event::{PollFd, PollFlags};
        let mode = super::schedule::WaitInterest::new(stopping, false, false, false);
        if mode.park {
            std::thread::park_timeout(std::time::Duration::from_millis(100));
            return Ok(());
        }
        let mut fds = Vec::new();
        for c in &self.children {
            let interest = super::schedule::WaitInterest::new(
                false,
                c.drain.stdout_eof,
                c.drain.stderr_eof,
                c.child.channel.pending(),
            );
            let mut flags = PollFlags::empty();
            if interest.stdout {
                flags |= PollFlags::IN;
            }
            if interest.writable {
                flags |= PollFlags::OUT;
            }
            if !flags.is_empty() {
                fds.push(PollFd::new(&c.child.channel.stream, flags));
            }
            if interest.stderr {
                fds.push(PollFd::new(&c.child.stderr, PollFlags::IN));
            }
        }
        native::io(rustix::event::poll(
            &mut fds,
            Some(&rustix::time::Timespec {
                tv_sec: 0,
                tv_nsec: 100_000_000,
            }),
        ))?;
        Ok(())
    }
    pub fn fail(&mut self, reason: Reason) -> Result<()> {
        match reason {
            Reason::PublicationFailure => {
                self.publication_failed = true;
                self.event(Event::PublicationFailed)
            }
            Reason::UnresolvedChildren => {
                self.unresolved = true;
                self.event(Event::UnknownChild)
            }
            _ => {
                if let Some(t) = self.children.first().map(|c| c.child.token) {
                    self.event(Event::HelperFailed(t))
                } else {
                    Ok(())
                }
            }
        }
    }
}

fn diagnostic(
    stream: &mut Option<OperationalStream>,
    capture: &mut Option<protocol::DiagnosticCapture>,
    token: Token,
    bytes: &[u8],
) -> Result<()> {
    if let Some(capture) = capture {
        capture.push(token, bytes)
    } else {
        let stream = stream
            .as_mut()
            .ok_or(ProvenanceError("diagnostic stream already closed"))?;
        stream.append(&u64::from(token.pid).to_le_bytes())?;
        stream.append(&(bytes.len() as u64).to_le_bytes())?;
        stream.append(bytes)
    }
}
