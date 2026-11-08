//! Concrete retained claim -> monitored worker/broker -> resolved refusal closure.
//! Scientific completion is separately gated; this controller never fabricates it.
use super::{
    capacity,
    monitor::{Kind, Message, Monitor, unix_ns},
    native,
    protocol::{self, Tag},
    schedule::{Event, Reason, Token},
};
use crate::{
    study_provenance::{
        ProvenanceError, Result, RuntimeBrokerPolicy, ensure,
        publication::{OperationalStream, publish_operational},
        resource_operational::{
            package::Package,
            retained::{HeldDirectory, MinimalOrigin},
        },
    },
    study_reference::operational as wire,
};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};
struct Controller {
    origin: MinimalOrigin,
    claims: HeldDirectory,
    output: HeldDirectory,
    id: String,
    marker_hash: String,
    monitor: Monitor,
    work: wire::Requested,
    pending: Option<wire::FileReference>,
    claim: Option<wire::FileReference>,
    build: Option<RuntimeBrokerPolicy>,
    package: Option<Package>,
    worker: Option<Token>,
    worker_status: Option<u64>,
    worker_run: Vec<u8>,
    helpers: super::broker_dispatch::BrokerDispatch,
    storage: super::observation::Storage,
    started_visible: bool,
}
pub(super) fn launch(
    root: &Path,
    approval: &Path,
    approval_hash: &str,
    receipts: &[PathBuf],
    args: &[OsString],
    work: wire::Requested,
) -> Result<()> {
    let common_origin = native::monotonic()?;
    let origin = MinimalOrigin::acquire(approval, approval_hash, root)?;
    let (state, lock) = origin.acquire_lock()?;
    let claims = super::markers::resolve_prior(&state)?;
    super::probe_index::resolve(&state, &origin.host)?;
    native::prepare_supervisor()?;
    let process = native::ProcOrigin::capture()?;
    let id = super::child::claim_id(
        process.boot(),
        process.own_token.pid,
        process.own_token.start,
    );
    let boot =
        std::str::from_utf8(process.boot()).map_err(|_| ProvenanceError("supervisor boot text"))?;
    let stamp = wire::Stamp {
        id: &id,
        boot,
        pid: u64::from(process.own_token.pid),
        start: process.own_token.start,
        unix_ns: unix_ns()?,
        elapsed_ns: protocol::elapsed(common_origin, native::monotonic()?)?,
    };
    let marker = wire::host_pending(
        &stamp,
        origin.approval.origin.observation_origin.as_bytes(),
        origin.observation.identity()?,
    )
    .map_err(|_| ProvenanceError("host pending encoding"))?;
    ensure(marker.len() <= 4096, "host marker cap")?;
    publish_operational(claims.file(), &format!("{id}.pending.bin"), &marker)
        .map_err(|_| ProvenanceError("host pending not confirmed durable"))?;
    // After the host marker is visible, any initialization error leaves it
    // unresolved. Nothing infers closure from a missing child or free flock.
    let mut output = origin.observation.duplicate()?;
    output.child("resource", true, true)?;
    output.exclusive_child(&id)?;
    let monitor = Monitor::new(lock, output.file(), common_origin, process)?;
    let mut c = Controller {
        origin,
        claims,
        output,
        id,
        marker_hash: crate::data::sha256_hex(&marker),
        monitor,
        work,
        pending: None,
        claim: None,
        build: None,
        package: None,
        worker: None,
        worker_status: None,
        worker_run: Vec::new(),
        helpers: super::broker_dispatch::BrokerDispatch::new(),
        storage: super::observation::Storage::new(),
        started_visible: false,
    };
    let result = c
        .start(root, approval_hash, receipts, args)
        .and_then(|()| c.run());
    if result.is_err() {
        let _ = c.monitor.fail(Reason::ProtocolFailure);
    }
    let closure = c.finish();
    result.and(closure)?;
    Err(ProvenanceError(
        "observed worker ended before scientific completion; inspect durable supervisor end",
    ))
}
impl Controller {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> Result<wire::FileReference> {
        self.output.recheck()?;
        if publish_operational(self.output.file(), name, bytes).is_err() {
            self.monitor.publication_failed = true;
            let _ = self.monitor.event(Event::PublicationFailed);
            return Err(ProvenanceError("operational publication unconfirmed"));
        }
        self.output.recheck()?;
        Ok(wire::FileReference {
            root: 5,
            path: format!("resource/{}/{name}", self.id),
            length: bytes.len() as u64,
            hash: crate::data::sha256_hex(bytes),
        })
    }
    fn stamp(&self) -> Result<wire::Stamp<'_>> {
        let (wall, elapsed) = self.monitor.clock()?;
        Ok(wire::Stamp {
            id: &self.id,
            boot: self.origin.host.boot.text()?,
            pid: u64::from(self.monitor.process.own_token.pid),
            start: self.monitor.process.own_token.start,
            unix_ns: wall,
            elapsed_ns: elapsed,
        })
    }
    fn start(
        &mut self,
        root: &Path,
        approval: &str,
        receipts: &[PathBuf],
        args: &[OsString],
    ) -> Result<()> {
        let b = wire::bootstrap(&self.origin.bootstrap[..self.origin.bootstrap_len])
            .map_err(|_| ProvenanceError("supervisor bootstrap"))?;
        ensure(
            b.policy.0 == 3 && b.policy.1 == "resource/linux-cpu-observed-v1/policy.json",
            "fixed policy source",
        )?;
        let executable_hash = b.executable.to_owned();
        let policy = wire::FileReference {
            root: b.policy.0,
            path: b.policy.1.into(),
            length: b.policy.2,
            hash: b.policy.3.into(),
        };
        let raw = wire::pending(&self.stamp()?, &policy, self.origin.host.physical.text()?)
            .map_err(|_| ProvenanceError("pending wire"))?;
        self.pending = Some(self.publish("pending.bin", &raw)?);
        let worker = self.monitor.spawn(
            Kind::Worker,
            &std::env::current_exe()?,
            "--private-worker",
            args,
            &executable_hash,
        )?;
        self.worker = Some(worker);
        let initial = self.monitor.sample(0, true)?;
        ensure(
            initial.valid && initial.status == 0 && self.monitor.scheduler.reason().is_none(),
            "initial sample did not authorize Run",
        )?;
        let raw = wire::initial_sample(&initial, self.pending.as_ref().expect("durable pending"))
            .map_err(|_| ProvenanceError("initial sample wire"))?;
        let initial_ref = self.publish("initial-sample.bin", &raw)?;
        let claim = wire::claim(&wire::ClaimInputs {
            facts: wire::SupervisorFacts {
                stamp: self.stamp()?,
                physical: self.origin.host.physical.text()?,
                worker_pid: u64::from(worker.pid),
                worker_start: worker.start,
            },
            pending: self.pending.as_ref().expect("durable pending"),
            initial: &initial_ref,
            work: &self.work,
            approval,
            source: self.origin.approval.source.text()?,
            revision: self.origin.approval.revision.text()?,
        })
        .map_err(|_| ProvenanceError("claim wire"))?;
        self.claim = Some(self.publish("claim.bin", &claim)?);
        self.monitor.event(Event::ClaimDurable)?;
        let mut samples =
            OperationalStream::new(self.output.file(), "samples.bin", 64 * 1024 * 1024)?;
        samples.append(
            &wire::sample_header(self.claim.as_ref().expect("durable claim"))
                .map_err(|_| ProvenanceError("samples header"))?,
        )?;
        self.monitor.samples = Some(samples);
        self.build = Some(RuntimeBrokerPolicy::load(
            &self.origin,
            root,
            receipts,
            &mut || self.monitor.pulse_waiting(),
        )?);
        self.package = Some(Package::load(
            &self.origin,
            self.build.as_ref().expect("selected broker policy"),
            root,
            &mut || self.monitor.pulse_waiting(),
        )?);
        // Retained metadata is monitored before full-context acquisition. These
        // are known byte copies; allocator/runtime/kernel overhead is unknown.
        let mut payload = self.origin.approval_file.file.metadata()?.len();
        for path in receipts {
            let m = std::fs::symlink_metadata(path)?;
            ensure(
                m.is_file() && m.len() <= 8 * 1024 * 1024,
                "context receipt payload bounds",
            )?;
            payload = payload
                .checked_add(m.len())
                .ok_or(ProvenanceError("known serialized receipt payload overflow"))?;
        }
        payload = payload
            .checked_add(self.origin.bootstrap_len as u64)
            .ok_or(ProvenanceError("known bootstrap payload overflow"))?;
        let observation = capacity::observe(
            &self.origin.host,
            self.package.as_ref().expect("validated operational roots"),
            payload,
            64 * 1024 * 1024 + 8 * 1024 * 1024 + 64 * 1024 * 1024 + 5 * 2_097_152,
            4104,
            &mut || self.monitor.pulse_waiting(),
        )?;
        for (index, (name, bytes)) in observation.raw.iter().enumerate() {
            self.monitor.pulse_waiting()?;
            self.storage.add_raw(bytes.len() as u64)?;
            self.publish(&format!("observe-0-{index}-{name}.raw"), bytes)?;
        }
        // Preserve actual capacity values and explicit unknown-overhead boundary.
        let mut summary = format!(
            "known_serialized_context_payload_bytes={payload}\nhost_mem_available_bytes={}\nfinite_ancestor_count={}\nunknown_decoder_copy_allocator_runtime_kernel_overhead=true\nphysical_fit_claim=false\n",
            observation.host_available,
            observation.groups.len()
        );
        for (role, device, mount, kind, frsize, bavail, favail, available) in &observation.disk {
            let name = match role {
                capacity::DiskRoot::Artifact(tag) => format!("artifact-{tag}"),
                capacity::DiskRoot::Account => "account".into(),
            };
            use std::fmt::Write;
            writeln!(&mut summary,"disk={name} device={device} mount_id={mount} type={kind} frsize={frsize} bavail={bavail} favail={favail} available_bytes={available}").map_err(|_|ProvenanceError("capacity summary formatting"))?;
        }
        self.storage.add_raw(summary.len() as u64)?;
        self.publish("observe-0-capacity.raw", summary.as_bytes())?;
        self.monitor.pulse_waiting()?;
        self.package
            .as_ref()
            .expect("package")
            .require_roots(&[1, 3, 4, 5])?;
        let mut run = Vec::new();
        protocol::item(&mut run, approval.as_bytes())?;
        protocol::item(&mut run, self.origin.approval.source.as_bytes())?;
        run.extend_from_slice(&(self.work.bytes().len() as u64).to_le_bytes());
        run.extend_from_slice(self.work.bytes());
        let c = self.claim.as_ref().expect("claim");
        run.extend_from_slice(&c.length.to_le_bytes());
        protocol::item(&mut run, c.hash.as_bytes())?;
        let roots = self
            .package
            .as_ref()
            .expect("package")
            .roots
            .iter()
            .map(|(tag, _)| *tag)
            .collect::<Vec<_>>();
        run.extend_from_slice(&(roots.len() as u64).to_le_bytes());
        run.extend(roots);
        self.worker_run = run;
        self.monitor.hello(worker)
    }
    fn payload(&self, token: Token) -> Result<&[u8]> {
        if Some(token) == self.worker {
            Ok(&self.worker_run)
        } else {
            self.helpers.payload(token)
        }
    }
    fn frame(&mut self, token: Token, frame: protocol::Frame) -> Result<()> {
        let phase = self.monitor.child_mut(token)?.phase;
        if frame.tag == Tag::Ready {
            if phase == 1 {
                ensure(
                    frame.payload == self.monitor.child_mut(token)?.hello.encode(),
                    "private Ready projection",
                )?;
                let payload = self.payload(token)?.to_vec();
                let c = self.monitor.child_mut(token)?;
                c.child.channel.send(Tag::Run, payload)?;
                c.phase = 3;
                return Ok(());
            }
            ensure(
                phase == 3
                    && frame.payload == crate::data::sha256_hex(self.payload(token)?).as_bytes(),
                "private Run acknowledgment",
            )?;
            self.monitor.child_mut(token)?.phase = 4;
            return Ok(());
        }
        ensure(phase == 4, "control before Run acknowledgment")?;
        match frame.tag {
            Tag::HelperRequest => self.helpers.request(
                &mut self.monitor,
                self.build
                    .as_ref()
                    .ok_or(ProvenanceError("broker policy absent"))?,
                token,
                &frame.payload,
            ),
            Tag::HelperReply => self.helpers.result(&mut self.monitor, token, frame.payload),
            Tag::StartLink | Tag::ResultLink => {
                // Step3/4 must add genuine typed producer/link consumers. A
                // received or uncertain start is never reclassified prestart.
                self.started_visible = true;
                self.monitor.event(Event::StartVisible)?;
                Err(ProvenanceError(
                    "scientific link producer/consumer not connected",
                ))
            }
            _ => Err(ProvenanceError("unexpected operational control frame")),
        }
    }
    fn exited(&mut self, token: Token, status: u64) -> Result<()> {
        let c = self.monitor.child_mut(token)?;
        ensure(c.drain.ready(), "exit before output drain")?;
        if c.kind == Kind::Worker {
            self.worker_status = Some(status);
            return self.monitor.event(Event::Exited(token));
        }
        self.helpers.exited(&mut self.monitor, token, status)
    }
    fn run(&mut self) -> Result<()> {
        while !self.monitor.children.is_empty() {
            for message in self.monitor.pump()? {
                match message {
                    Message::Frame(token, frame) => self.frame(token, frame)?,
                    Message::Exited(token, status) => self.exited(token, status)?,
                    Message::ControlLost { .. } => {} // pump already stopped all owned children
                }
            }
            if self.monitor.scheduler.reason().is_some() {
                break;
            }
            self.monitor.wait()?;
        }
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if self.monitor.scheduler.reason().is_none() && !self.monitor.children.is_empty() {
            let _ = self.monitor.fail(Reason::ProtocolFailure);
        }
        let mut timing = super::monitor::CleanupTiming::new();
        while !self.monitor.children.is_empty() {
            for (token, status) in self.monitor.cleanup_turn(&mut timing) {
                if Some(token) == self.worker {
                    self.worker_status = Some(status);
                }
            }
        }
        self.hold_unresolved();
        let result = self.finish_records();
        if result.is_err() {
            self.monitor.publication_failed = true;
        }
        self.hold_unresolved();
        result
    }
    fn hold_unresolved(&mut self) {
        if super::schedule::closure_must_hold(
            self.monitor.unresolved,
            self.monitor.partial.len(),
            self.monitor.clock_failure.requires_fallback(true),
            self.monitor.publication_failed,
        ) {
            self.monitor.unresolved = true;
            // Includes facts first discovered by the final teardown sample or
            // any teardown/publication error; no Err drops an unresolved lock.
            loop {
                for partial in &mut self.monitor.partial {
                    partial.request_boundary_stop();
                    let _ = partial.reap_confirmed();
                }
                std::thread::park_timeout(std::time::Duration::from_millis(100));
            }
        }
    }
    fn finish_records(&mut self) -> Result<()> {
        if self.monitor.initialized {
            self.monitor.final_sample()?;
            self.monitor.event(Event::TeardownSample)?;
        }
        if let Some(diagnostics) = self.monitor.diagnostics.take() {
            diagnostics
                .finish()
                .map_err(|_| ProvenanceError("diagnostic stream publication unconfirmed"))?;
        }
        let samples = if let Some(stream) = self.monitor.samples.take() {
            let (length, hash) = stream
                .finish()
                .map_err(|_| ProvenanceError("sample stream publication unconfirmed"))?;
            Some(wire::FileReference {
                root: 5,
                path: format!("resource/{}/samples.bin", self.id),
                length,
                hash,
            })
        } else {
            None
        };
        let Some(pending) = self.pending.as_ref() else {
            return Err(ProvenanceError(
                "missing durable pending leaves host marker unresolved",
            ));
        };
        let reason = match self.monitor.scheduler.reason() {
            Some(Reason::RssStop) => 3,
            Some(Reason::InvalidSample) => 4,
            Some(Reason::ProtocolFailure) => 5,
            Some(Reason::PublicationFailure) => 6,
            Some(Reason::UnresolvedChildren) => 7,
            Some(Reason::WorkerFailure) => 2,
            _ => {
                if self.started_visible {
                    2
                } else {
                    1
                }
            }
        };
        let resolved = !self.monitor.unresolved && !self.monitor.publication_failed;
        let raw = wire::supervisor_end(&wire::EndInputs {
            pending,
            claim: self.claim.as_ref(),
            start: None,
            result: None,
            worker_end: None,
            samples: samples.as_ref(),
            last_sequence: self.monitor.sequence.saturating_sub(1),
            peak: self.monitor.scheduler.peak(),
            reason,
            wait_status: self.worker_status,
            resolved,
            ended: self.monitor.clock()?,
        })
        .map_err(|_| ProvenanceError("supervisor end wire"))?;
        let end = self.publish("supervisor.end.bin", &raw)?;
        ensure(resolved, "unresolved supervisor closure")?;
        let closed = wire::closed(&self.marker_hash, &end)
            .map_err(|_| ProvenanceError("host closure wire"))?;
        ensure(closed.len() <= 4096, "host closure cap")?;
        self.claims.recheck()?;
        publish_operational(
            self.claims.file(),
            &format!("{}.closed.bin", self.id),
            &closed,
        )
        .map_err(|_| ProvenanceError("host closure publication unconfirmed"))?;
        self.claims.recheck()?;
        Ok(())
    }
}
