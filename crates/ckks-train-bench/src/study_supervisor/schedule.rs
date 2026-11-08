//! Shared finite control transitions. Only the native owned-pidfd layer executes actions.
use crate::study_provenance::{ProvenanceError, Result, ensure};
pub(super) const INTERVAL: u64 = 10_000_000_000;
pub(super) const GRACE: u64 = 5_000_000_000;
pub(super) const RSS_LIMIT: u64 = 94_371_840;
pub(super) struct ClockFailureLatch(std::cell::Cell<bool>);
impl ClockFailureLatch {
    pub fn new() -> Self {
        Self(std::cell::Cell::new(false))
    }
    pub fn failed(&self) {
        self.0.set(true);
    }
    pub fn requires_fallback(&self, current_read_available: bool) -> bool {
        if !current_read_available {
            self.failed();
        }
        self.0.get()
    }
}
pub(super) struct FallbackClock {
    armed: bool,
    killed: bool,
}
impl FallbackClock {
    pub fn new() -> Self {
        Self {
            armed: false,
            killed: false,
        }
    }
    pub fn should_start(&self, latch: &ClockFailureLatch, available: bool) -> bool {
        !self.armed && latch.requires_fallback(available)
    }
    pub fn arm(&mut self) -> Result<()> {
        ensure(!self.armed, "cleanup fallback cannot rebase")?;
        self.armed = true;
        Ok(())
    }
    pub fn kill_due(&mut self, elapsed: Option<std::time::Duration>) -> bool {
        if self.armed
            && !self.killed
            && elapsed.is_some_and(|d| d >= std::time::Duration::from_nanos(GRACE))
        {
            self.killed = true;
            true
        } else {
            false
        }
    }
}
pub(super) struct SpawnProgress {
    stage: u8,
    failed: bool,
}
impl SpawnProgress {
    pub fn spawned() -> Self {
        Self {
            stage: 0,
            failed: false,
        }
    }
    pub fn advance(&mut self) -> Result<()> {
        ensure(
            !self.failed && self.stage < 4,
            "partial spawn progress invalid",
        )?;
        self.stage += 1;
        Ok(())
    }
    pub fn fail(&mut self) {
        self.failed = true;
    }
    pub fn unresolved(&self) -> bool {
        self.failed || self.stage != 4
    }
    pub fn ready(&self) -> bool {
        self.stage == 4 && !self.failed
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Token {
    pub pid: u32,
    pub start: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reason {
    PrestartRefusal,
    WorkerFailure,
    RssStop,
    InvalidSample,
    ProtocolFailure,
    PublicationFailure,
    UnresolvedChildren,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Run(Token),
    Term(Token),
    Kill(Token),
    Reap(Token),
    StopBoundary,
    PublishClosed,
}
pub(super) fn apply_all(
    actions: Vec<Action>,
    mut execute: impl FnMut(Action) -> Result<()>,
) -> Vec<(Action, ProvenanceError)> {
    let mut failed = Vec::new();
    for action in actions {
        if let Err(error) = execute(action) {
            failed.push((action, error));
        }
    }
    failed
}
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // Pure ordering cases plus explicitly pending Step3/4 producer events.
pub(super) enum Event {
    Pending,
    Register(Token),
    Sample {
        started: u64,
        finished: u64,
        rss: Option<u64>,
    },
    PhaseSample {
        started: u64,
        finished: u64,
        rss: Option<u64>,
    },
    ClaimVisible,
    ClaimDurable,
    RequestBound,
    StartVisible,
    WorkerComplete,
    Exited(Token),
    HelperComplete(Token),
    HelperFailed(Token),
    ParentEof,
    KnownReparent(Token),
    UnknownChild,
    IdentityLost(Token),
    PublicationFailed,
    TeardownSample,
    EndDurable,
    Tick,
}
#[derive(Clone)]
pub(super) struct Scheduler {
    processes: [Option<Token>; 3],
    worker: Option<Token>,
    pending: bool,
    initial: bool,
    probe_parent_sampling: bool,
    run: bool,
    bound: bool,
    start: bool,
    complete: bool,
    teardown: bool,
    unresolved: bool,
    publication_failed: bool,
    reason: Option<Reason>,
    due: u64,
    peak: Option<u64>,
    grace: Option<u64>,
    killed: bool,
}
/// Linear exclusive reservation across the only fallible native spawn. Dropping
/// it changes no process record; a completed native owner commits infallibly.
pub(super) struct Registration<'a> {
    scheduler: &'a mut Scheduler,
    slot: usize,
}
impl Registration<'_> {
    pub fn cancel(self) {} // Releases only the exclusive pre-spawn reservation; never custody.
    pub fn commit(self, token: Token) {
        self.scheduler.processes[self.slot] = Some(token);
        if self.scheduler.worker.is_none() {
            self.scheduler.worker = Some(token);
        }
    }
}
pub(super) fn closure_must_hold(
    unresolved: bool,
    partial: usize,
    clock_lost: bool,
    publication_failed: bool,
) -> bool {
    unresolved || partial != 0 || clock_lost || publication_failed
}
impl Scheduler {
    pub fn new() -> Self {
        Self {
            processes: [None; 3],
            worker: None,
            pending: false,
            initial: false,
            probe_parent_sampling: false,
            run: false,
            bound: false,
            start: false,
            complete: false,
            teardown: false,
            unresolved: false,
            publication_failed: false,
            reason: None,
            due: INTERVAL,
            peak: None,
            grace: None,
            killed: false,
        }
    }
    pub fn new_probe() -> Self {
        let mut value = Self::new();
        value.probe_parent_sampling = true;
        value
    }
    pub fn fork_probe(&self) -> Result<Self> {
        ensure(
            self.probe_parent_sampling
                && self.initial
                && !self.run
                && self.reason.is_none()
                && !self.unresolved
                && self.live().len() == 1,
            "native inner scheduler fork boundary",
        )?;
        Ok(self.clone())
    }
    pub fn reserve(&mut self) -> Result<Registration<'_>> {
        ensure(
            self.pending && self.reason.is_none() && !self.unresolved,
            "spawn reservation after stop/unresolved or before pending",
        )?;
        let slot = self
            .processes
            .iter()
            .position(Option::is_none)
            .ok_or(ProvenanceError("four process reservation cap"))?;
        Ok(Registration {
            scheduler: self,
            slot,
        })
    }
    pub fn probe_retire(&mut self, token: Token) -> Result<Vec<Action>> {
        ensure(
            self.probe_parent_sampling && !self.run && self.reason.is_none(),
            "native retirement outside private pre-policy probe",
        )?;
        let slot = self
            .processes
            .iter_mut()
            .find(|p| **p == Some(token))
            .ok_or(ProvenanceError("native retirement unowned/duplicate token"))?;
        *slot = None;
        if self.worker == Some(token) {
            self.worker = None;
        }
        Ok(vec![Action::Reap(token)])
    }
    pub fn live(&self) -> Vec<Token> {
        self.processes.iter().flatten().copied().collect()
    }
    pub fn due(&self) -> u64 {
        self.due
    }
    pub fn peak(&self) -> Option<u64> {
        self.peak
    }
    pub fn reason(&self) -> Option<Reason> {
        self.reason
    }
    pub fn can_close(&self) -> bool {
        self.live().is_empty() && !self.unresolved && !self.publication_failed && self.teardown
    }
    pub fn expand(&self, data: bool) -> Result<()> {
        ensure(
            self.run && self.reason.is_none() && (!data || self.bound),
            "expansion before durable claim/resolved binding",
        )
    }
    fn stop(&mut self, reason: Reason, _now: u64) -> Result<Vec<Action>> {
        if self.reason.is_some() {
            return Ok(Vec::new());
        }
        self.reason = Some(reason);
        // No Tick can escalate until the actual complete TERM batch is acknowledged.
        self.grace = None;
        Ok(self.live().into_iter().map(Action::Term).collect())
    }
    pub fn observe_control(
        &mut self,
        token: Token,
        observed: super::protocol::ControlObservation,
        now: u64,
    ) -> Result<Vec<Action>> {
        ensure(
            self.live().contains(&token),
            "control observation outside owned ledger",
        )?;
        if observed.lost() {
            self.stop(Reason::ProtocolFailure, now)
        } else {
            Ok(Vec::new())
        }
    }
    pub fn term_waiting(&self) -> bool {
        self.reason.is_some() && self.grace.is_none() && !self.killed
    }
    pub fn term_batch_completed(&mut self, now: u64) -> Result<()> {
        ensure(
            self.term_waiting(),
            "duplicate/unexpected TERM batch completion",
        )?;
        self.grace = Some(
            now.checked_add(GRACE)
                .ok_or(ProvenanceError("post-TERM grace clock overflow"))?,
        );
        Ok(())
    }
    pub fn event(&mut self, event: Event, now: u64) -> Result<Vec<Action>> {
        match event {
            Event::Pending => {
                ensure(!self.pending && self.worker.is_none(), "pending order")?;
                self.pending = true;
            }
            Event::Register(t) => {
                ensure(
                    self.pending && self.reason.is_none() && t.pid > 0 && t.start > 0,
                    "register order/identity",
                )?;
                ensure(
                    !self.live().iter().any(|p| p.pid == t.pid),
                    "duplicate/reused PID",
                )?;
                let slot = self
                    .processes
                    .iter_mut()
                    .find(|p| p.is_none())
                    .ok_or(ProvenanceError("four process cap"))?;
                *slot = Some(t);
                if self.worker.is_none() {
                    self.worker = Some(t);
                }
            }
            Event::Sample {
                started,
                finished,
                rss,
            } => {
                ensure(
                    self.pending && (self.worker.is_some() || self.probe_parent_sampling),
                    "sample before pending/worker",
                )?;
                let end = if self.initial {
                    self.due
                        .checked_add(INTERVAL)
                        .ok_or(ProvenanceError("sample clock overflow"))?
                } else {
                    INTERVAL
                };
                let valid = finished >= started
                    && finished <= end
                    && (!self.initial || started >= self.due)
                    && rss.is_some();
                if !valid {
                    if self.initial {
                        self.due = end;
                    }
                    return self.stop(Reason::InvalidSample, now);
                }
                let value = rss.ok_or(ProvenanceError("missing valid RSS"))?;
                self.peak = Some(self.peak.map_or(value, |p| p.max(value)));
                if self.initial {
                    self.due = end;
                } else {
                    self.initial = true;
                }
                if value > RSS_LIMIT {
                    return self.stop(Reason::RssStop, now);
                }
            }
            Event::PhaseSample {
                started,
                finished,
                rss,
            } => {
                ensure(self.initial, "phase sample before initial")?;
                if finished < started
                    || finished
                        > self
                            .due
                            .checked_add(INTERVAL)
                            .ok_or(ProvenanceError("phase clock overflow"))?
                    || rss.is_none()
                {
                    return self.stop(Reason::InvalidSample, now);
                }
                let value = rss.ok_or(ProvenanceError("missing phase RSS"))?;
                self.peak = Some(self.peak.map_or(value, |p| p.max(value)));
                if value > RSS_LIMIT {
                    return self.stop(Reason::RssStop, now);
                }
            }
            Event::ClaimVisible => {
                ensure(self.initial && !self.run, "claim visibility order")?;
            }
            Event::ClaimDurable => {
                ensure(
                    self.pending && self.initial && !self.run && self.reason.is_none(),
                    "Run before durable claim",
                )?;
                let worker = self.worker.ok_or(ProvenanceError("missing worker"))?;
                self.run = true;
                return Ok(vec![Action::Run(worker)]);
            }
            Event::RequestBound => {
                self.expand(false)?;
                self.bound = true;
            }
            Event::StartVisible => {
                self.expand(false)?;
                self.start = true;
            }
            Event::WorkerComplete => {
                ensure(
                    self.start && self.reason.is_none(),
                    "completion before start",
                )?;
                self.complete = true;
            }
            Event::HelperComplete(t) => {
                ensure(
                    Some(t) != self.worker && self.reason.is_none(),
                    "unexpected successful helper completion",
                )?;
                let slot = self
                    .processes
                    .iter_mut()
                    .find(|p| **p == Some(t))
                    .ok_or(ProvenanceError("unowned successful helper"))?;
                *slot = None;
                return Ok(vec![Action::Reap(t)]);
            }
            Event::Exited(t) => {
                let slot = self
                    .processes
                    .iter_mut()
                    .find(|p| **p == Some(t))
                    .ok_or(ProvenanceError("unowned wait identity"))?;
                *slot = None;
                let mut actions = vec![Action::Reap(t)];
                if !self.complete && self.reason.is_none() {
                    let reason = if Some(t) == self.worker && !self.start {
                        Reason::PrestartRefusal
                    } else {
                        Reason::WorkerFailure
                    };
                    actions.extend(self.stop(reason, now)?);
                }
                return Ok(actions);
            }
            Event::HelperFailed(t) => {
                ensure(self.live().contains(&t), "unowned helper failure")?;
                return self.stop(Reason::ProtocolFailure, now);
            }
            Event::ParentEof => {
                self.unresolved = true;
                return Ok(vec![Action::StopBoundary]);
            }
            Event::KnownReparent(t) => {
                ensure(self.live().contains(&t), "unknown reparent identity")?;
            }
            Event::UnknownChild => {
                self.unresolved = true;
                return self.stop(Reason::UnresolvedChildren, now);
            }
            Event::IdentityLost(t) => {
                let slot = self
                    .processes
                    .iter_mut()
                    .find(|p| **p == Some(t))
                    .ok_or(ProvenanceError("identity loss unowned"))?;
                *slot = None;
                self.unresolved = true;
                return self.stop(Reason::UnresolvedChildren, now);
            }
            Event::PublicationFailed => {
                self.publication_failed = true;
                return self.stop(Reason::PublicationFailure, now);
            }
            Event::TeardownSample => {
                ensure(self.live().is_empty(), "teardown with live children")?;
                self.teardown = true;
            }
            Event::EndDurable => {
                ensure(self.can_close(), "unresolved closure")?;
                return Ok(vec![Action::PublishClosed]);
            }
            Event::Tick => {
                if let Some(grace) = self.grace {
                    if now >= grace && !self.killed {
                        self.killed = true;
                        return Ok(self.live().into_iter().map(Action::Kill).collect());
                    }
                } else if self.reason.is_none()
                    && self.initial
                    && now
                        > self
                            .due
                            .checked_add(INTERVAL)
                            .ok_or(ProvenanceError("sample clock overflow"))?
                {
                    return self.stop(Reason::InvalidSample, now);
                }
            }
        }
        Ok(Vec::new())
    }
}

/// Stop waits deliberately park once per turn: closed/HUP/error descriptors and
/// abandoned outgoing frames cannot create an unbounded ready loop.
pub(super) struct WaitInterest {
    pub stdout: bool,
    pub stderr: bool,
    pub writable: bool,
    pub park: bool,
}
impl WaitInterest {
    pub fn new(stopping: bool, stdout_eof: bool, stderr_eof: bool, pending: bool) -> Self {
        if stopping {
            Self {
                stdout: false,
                stderr: false,
                writable: false,
                park: true,
            }
        } else {
            Self {
                stdout: !stdout_eof,
                stderr: !stderr_eof,
                writable: pending,
                park: false,
            }
        }
    }
}
