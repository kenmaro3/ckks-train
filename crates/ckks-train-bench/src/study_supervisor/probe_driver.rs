//! Isolated pre-policy native harness. It never obtains a production owner.
use super::{
    broker_dispatch::BrokerDispatch,
    capacity,
    monitor::{Kind, Message, Monitor, unix_ns},
    native,
    probe_plan::Selection,
    protocol::{self, Tag},
    schedule::{Event, Reason, Token},
};
use crate::{
    study_provenance::{
        ProvenanceError, Result, RuntimeBrokerPolicy, ensure,
        publication::publish_operational,
        resource_operational::{package::Package, probe::ProbeInputs, retained::HeldDirectory},
    },
    study_reference::operational::{
        self as wire,
        probe_output::{ProbeEvent, ProbeObservation},
    },
};
use std::{
    ffi::OsString,
    io::Write,
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
};
pub(super) struct MeasuredEvent {
    pub name: &'static str,
    pub elapsed: Option<u64>,
    pub values: Vec<u64>,
    pub detail: Vec<u8>,
}
struct ClockExperiment {
    monitor: Monitor,
    timing: super::monitor::CleanupTiming,
    confirmed: bool,
}
pub(super) struct Session {
    pub selection: Selection,
    pub inputs: ProbeInputs,
    pub state: HeldDirectory,
    pub output: HeldDirectory,
    pub index: HeldDirectory,
    pub command: Vec<u8>,
    pub monitor: Monitor,
    pub build: Option<RuntimeBrokerPolicy>,
    pub package: Option<Package>,
    pub worker: Option<Token>,
    pub events: Vec<MeasuredEvent>,
    broker: BrokerDispatch,
    run_payload: Vec<u8>,
    report: bool,
    worker_status: Option<u64>,
    git_offsets: Vec<(Token, usize)>,
    pub faults: Vec<&'static str>,
    pub threshold: bool,
    event_bytes: usize,
    started: (u64, u64),
    pub reacquire: bool,
    python_count: u64,
    git_count: u64,
    clock_inner: Option<ClockExperiment>,
}
fn pulse(monitor: &mut Monitor) -> Result<()> {
    ensure(
        monitor.elapsed()? <= 60_000_000_000,
        "native probe validity deadline",
    )?;
    monitor.pulse_waiting()
}
fn pin(
    r: &crate::study_provenance::resource_operational::approval::OriginRef,
) -> Result<wire::FileReference> {
    Ok(wire::FileReference {
        root: r.root,
        path: r.path.text()?.into(),
        length: r.length,
        hash: r.hash.text()?.into(),
    })
}
fn hex(raw: &[u8]) -> String {
    raw.iter()
        .flat_map(|b| {
            let h = b"0123456789abcdef";
            [
                char::from(h[(b >> 4) as usize]),
                char::from(h[(b & 15) as usize]),
            ]
        })
        .collect()
}
impl Session {
    pub fn open(args: &[OsString]) -> Result<Self> {
        let selection = Selection::parse(args)?;
        let origin = native::monotonic()?;
        let started = (unix_ns()?, protocol::elapsed(origin, native::monotonic()?)?);
        let inputs = ProbeInputs::acquire(&selection.origin, &selection.hash)?;
        let (state, lock) = inputs.acquire_lock()?;
        super::markers::resolve_prior(&state)?;
        super::probe_index::resolve(&state, &inputs.host)?;
        native::prepare_supervisor()?;
        let process = native::ProcOrigin::capture()?;
        let mut output = inputs.observation.duplicate()?;
        output.child("native-probes", true, true)?;
        output.child(inputs.origin.probe_id.text()?, true, true)?;
        output.exclusive_child(selection.check)?;
        let actual = std::env::args_os().collect::<Vec<_>>();
        ensure(
            actual.len() == 7
                && actual[1..] == *args
                && actual[0].as_bytes() == std::env::current_exe()?.as_os_str().as_bytes(),
            "native exact external argv",
        )?;
        let command=serde_json::to_vec(&serde_json::json!({"schema":"hekit-study-supervisor-probe-command-v1","argv":actual.iter().map(|v|hex(v.as_bytes())).collect::<Vec<_>>(),"cwd":hex(std::env::current_dir()?.as_os_str().as_bytes())})).map_err(|_|ProvenanceError("native command encoding"))?;
        // A durable command in an exclusive test directory is the test-only pending
        // state. It cannot be reused or interpreted as a production pending/claim.
        let index = super::probe_index::create(
            &state,
            inputs.origin.probe_id.text()?,
            selection.check,
            &command,
        )?;
        publish_operational(output.file(), "command.json", &command)
            .map_err(|_| ProvenanceError("native command pending unconfirmed"))?;
        let monitor = Monitor::new_probe(lock, origin, process)?;
        Ok(Self {
            selection,
            inputs,
            state,
            output,
            index,
            command,
            monitor,
            build: None,
            package: None,
            worker: None,
            events: Vec::new(),
            broker: BrokerDispatch::new(),
            run_payload: Vec::new(),
            report: false,
            worker_status: None,
            git_offsets: Vec::new(),
            faults: Vec::new(),
            threshold: false,
            event_bytes: 0,
            started,
            reacquire: false,
            python_count: 0,
            git_count: 0,
            clock_inner: None,
        })
    }
    pub fn event(
        &mut self,
        name: &'static str,
        elapsed: Option<u64>,
        values: Vec<u64>,
        detail: &[u8],
    ) -> Result<()> {
        ensure(
            values.len() <= 16 && detail.len() <= 4096 && self.events.len() < 4096,
            "native event bounds",
        )?;
        let length = 8usize
            .checked_add(name.len())
            .and_then(|n| n.checked_add(1 + if elapsed.is_some() { 8 } else { 0 }))
            .and_then(|n| n.checked_add(16 + values.len() * 8 + detail.len()))
            .ok_or(ProvenanceError("native event size overflow"))?;
        let total = self
            .event_bytes
            .checked_add(length)
            .ok_or(ProvenanceError("native events size overflow"))?;
        ensure(total <= 8 * 1024 * 1024, "native aggregate event cap")?;
        self.events
            .try_reserve(1)
            .map_err(|_| ProvenanceError("native event allocation"))?;
        let mut copy = Vec::new();
        copy.try_reserve_exact(detail.len())
            .map_err(|_| ProvenanceError("native detail allocation"))?;
        copy.extend_from_slice(detail);
        self.events.push(MeasuredEvent {
            name,
            elapsed,
            values,
            detail: copy,
        });
        self.event_bytes = total;
        Ok(())
    }
    pub fn chunks(&mut self, name: &'static str, token: Token, bytes: &[u8]) -> Result<()> {
        let now = self.monitor.elapsed()?;
        let count = bytes.len().div_ceil(4096);
        for (index, chunk) in bytes.chunks(4096).enumerate() {
            self.event(
                name,
                Some(now),
                vec![
                    u64::from(token.pid),
                    token.start,
                    index as u64,
                    count as u64,
                ],
                chunk,
            )?;
        }
        Ok(())
    }
    pub fn prepare(&mut self) -> Result<()> {
        self.monitor.sample(0, true)?;
        if self.selection.index == 14 {
            ensure(
                super::probe_index::resolve(&self.state, &self.inputs.host).is_err(),
                "durable pending failed to block another probe/production entry",
            )?;
            self.faults.push("completion-copy-absent-injected");
            self.event("durable-incomplete-index-refused",Some(self.monitor.elapsed()?),vec![],b"actual durable command pair with missing observations; shared new-probe/production resolver refuses regardless of next probe ID; no actual process crash claimed")?;
        }
        self.build = Some(RuntimeBrokerPolicy::load_probe(&self.inputs, &mut || {
            pulse(&mut self.monitor)
        })?);
        let build = self.build.as_ref().expect("probe selected build");
        ensure(
            std::env::current_dir()? == build.root(),
            "native actual cwd/capsule root",
        )?;
        self.package = Some(Package::load_probe(&self.inputs, build, &mut || {
            pulse(&mut self.monitor)
        })?);
        let observed = capacity::observe(
            &self.inputs.host,
            self.package.as_ref().expect("native retained package"),
            8 * 1024 * 1024,
            64 * 1024 * 1024,
            4096,
            &mut || pulse(&mut self.monitor),
        )?;
        for (name, bytes) in observed.raw {
            self.event(
                "capacity-source-name",
                Some(self.monitor.elapsed()?),
                vec![bytes.len() as u64],
                name.as_bytes(),
            )?;
            self.chunks(
                "capacity-source-bytes",
                self.monitor.process.own_token,
                &bytes,
            )?;
        }
        for (role, dev, mount, kind, frsize, bavail, favail, available) in observed.disk {
            let (name, mut values) = match role {
                capacity::DiskRoot::Account => ("account-filesystem", vec![]),
                capacity::DiskRoot::Artifact(tag) => ("root-filesystem", vec![u64::from(tag)]),
            };
            values.extend([dev, mount, kind, frsize, bavail, favail, available]);
            self.event(
                name,
                Some(self.monitor.elapsed()?),
                values,
                b"actual retained fstatvfs; quota/backing separately pinned reviewed evidence",
            )?;
        }
        self.event(
            "host-memory-headroom",
            Some(self.monitor.elapsed()?),
            vec![observed.host_available, observed.groups.len() as u64],
            b"unknown allocator/runtime/kernel overhead; no physical-fit claim",
        )?;
        for (index, (limit, current)) in observed.groups.iter().enumerate() {
            self.event(
                "cgroup-headroom",
                Some(self.monitor.elapsed()?),
                vec![
                    index as u64,
                    u64::from(limit.is_some()),
                    limit.unwrap_or(0),
                    *current,
                ],
                b"max remains unbounded, not zero capacity",
            )?;
        }
        self.inputs.origin_file.recheck()?;
        self.inputs.host_file.recheck()?;
        crate::study_provenance::resource_operational::retained::check_kernel_projection(
            &self.inputs.host,
        )?;
        self.event(
            "kernel-projection-matched",
            Some(self.monitor.elapsed()?),
            vec![self.inputs.origin.account_uid],
            self.inputs.origin.host.hash.as_bytes().to_vec().as_slice(),
        )?;
        let subreaper = native::io(rustix::process::child_subreaper())?;
        ensure(subreaper.is_some(), "native actual subreaper flag")?;
        let nice = native::io(rustix::process::getpriority_process(None))?;
        ensure(nice == 10, "native actual supervisor nice")?;
        self.event(
            "supervisor-kernel-state",
            Some(self.monitor.elapsed()?),
            vec![
                u64::from(self.monitor.process.own_token.pid),
                self.monitor.process.own_token.start,
                nice as u64,
                1,
            ],
            self.monitor.process.boot().to_vec().as_slice(),
        )?;
        let build = self.build.as_ref().expect("probe build");
        let requests = build.probe_requests()?;
        protocol::item(&mut self.run_payload, b"hekit-native-broker-probe-v1")?;
        self.run_payload.extend(2u64.to_le_bytes());
        for command in requests {
            let raw = protocol::encode_exec(&command)?;
            self.run_payload.extend((raw.len() as u64).to_le_bytes());
            self.run_payload.extend(raw);
        }
        let role = if [1, 6].contains(&self.selection.index) {
            "middle"
        } else {
            "idle"
        };
        let token = self.monitor.spawn(
            Kind::Worker,
            &std::env::current_exe()?,
            "--private-native-probe",
            &[self.selection.check.into(), role.into()],
            &build.executable_hash,
        )?;
        self.worker = Some(token);
        self.monitor.sample(1, false)?;
        self.monitor.hello(token)?;
        while !self.report {
            self.turn()?;
            self.monitor.wait()?;
        }
        ensure(
            self.broker.active() == 0 && self.python_count == 2 && self.git_count == 5,
            "native actual full broker source validation",
        )?;
        Ok(())
    }
    fn payload(&self, token: Token) -> Result<&[u8]> {
        if Some(token) == self.worker {
            Ok(&self.run_payload)
        } else {
            self.broker.payload(token)
        }
    }
    pub fn turn(&mut self) -> Result<()> {
        ensure(
            self.monitor.elapsed()? <= 60_000_000_000,
            "native probe validity deadline",
        )?;
        self.monitor.check_descendants()?;
        if self.selection.index == 5 {
            let actual = self
                .monitor
                .process
                .children(self.monitor.process.own_token.pid)?;
            let mut values = vec![
                u64::from(self.monitor.process.own_token.pid),
                actual.len() as u64,
            ];
            values.extend(actual.iter().map(|p| u64::from(*p)));
            self.event("actual-direct-child-census",Some(self.monitor.elapsed()?),values,b"actual proc children list; all entries compared with retained direct pidfd ledger")?;
            let mut descendants = Vec::new();
            for c in &self.monitor.children {
                if c.child.status()?.is_none() {
                    let actual = self.monitor.process.children(c.child.token.pid)?;
                    ensure(actual.is_empty(), "native runtime helper descendant")?;
                    descendants.push(c.child.token);
                }
            }
            for t in descendants {
                self.event(
                    "actual-helper-descendants-empty",
                    Some(self.monitor.elapsed()?),
                    vec![u64::from(t.pid), t.start, 0],
                    b"actual proc child list, not assumed from direct spawn",
                )?;
            }
        }
        let messages = self.monitor.pump()?;
        for message in messages {
            match message {
                Message::Frame(token, frame) => {
                    self.chunks("child-stdout-frame", token, &protocol::encode(&frame)?)?;
                    let phase = self.monitor.child_mut(token)?.phase;
                    if frame.tag == Tag::Ready {
                        if phase == 1 {
                            ensure(
                                frame.payload == self.monitor.child_mut(token)?.hello.encode(),
                                "native actual Hello/Ready",
                            )?;
                            let payload = self.payload(token)?.to_vec();
                            let c = self.monitor.child_mut(token)?;
                            c.child.channel.send(Tag::Run, payload)?;
                            c.phase = 3;
                            continue;
                        }
                        ensure(
                            phase == 3
                                && frame.payload
                                    == crate::data::sha256_hex(self.payload(token)?).as_bytes(),
                            "native actual Run acknowledgment",
                        )?;
                        self.monitor.child_mut(token)?.phase = 4;
                        continue;
                    }
                    ensure(phase == 4, "native child data before Run acknowledgment")?;
                    match frame.tag {
                        Tag::HelperRequest => {
                            let request = protocol::HelperRequest::decode(&frame.payload)?;
                            if request.kind == 0 {
                                self.python_count += 1;
                            } else {
                                self.git_count += 1;
                            }
                            self.broker.request(
                                &mut self.monitor,
                                self.build.as_ref().expect("probe build"),
                                token,
                                &frame.payload,
                            )?;
                            let child = self
                                .monitor
                                .children
                                .last()
                                .ok_or(ProvenanceError("native helper registration absent"))?;
                            let helper = child.child.token;
                            let nice = child.child.nice()?;
                            ensure(
                                nice == 10
                                    && child.child.parent_pid()?
                                        == self.monitor.process.own_token.pid,
                                "native direct helper/nice",
                            )?;
                            self.event(
                                "direct-helper-spawn",
                                Some(self.monitor.elapsed()?),
                                vec![
                                    u64::from(helper.pid),
                                    helper.start,
                                    u64::from(request.kind),
                                    nice as u64,
                                ],
                                b"actual retained owned pidfd before helper Run",
                            )?;
                        }
                        Tag::HelperReply => {
                            self.broker
                                .result(&mut self.monitor, token, frame.payload)?
                        }
                        Tag::ExitAck => {
                            ensure(
                                Some(token) == self.worker
                                    && !self.report
                                    && frame.payload.starts_with(b"HKPRB1"),
                                "native worker report boundary",
                            )?;
                            let mut r = protocol::Payload::new(&frame.payload[6..])?;
                            let mut values = Vec::new();
                            for _ in 0..6 {
                                values.push(r.number()?);
                            }
                            ensure(
                                values[0] == u64::from(token.pid)
                                    && values[1] == token.start
                                    && values[2] == u64::from(self.monitor.process.own_token.pid)
                                    && values[3] == 10
                                    && values[4] <= self.monitor.elapsed()?,
                                "native actual worker identity/nice/common origin",
                            )?;
                            ensure(r.number()? == 2, "native broker result count")?;
                            let executable = r.blob(65)?;
                            let source = r.blob(65)?;
                            r.done()?;
                            ensure(
                                executable
                                    == format!("{}\n", self.inputs.origin.executable.text()?)
                                        .as_bytes()
                                    && source
                                        == format!("{}\n", self.inputs.origin.source.text()?)
                                            .as_bytes(),
                                "native actual executable/source helper results",
                            )?;
                            self.event(
                                "worker-kernel-state",
                                Some(values[4]),
                                values,
                                b"actual same-executable private worker and full broker replies",
                            )?;
                            self.report = true;
                        }
                        _ => return Err(ProvenanceError("unexpected native probe frame")),
                    }
                }
                Message::ControlLost { .. } => {
                    return Err(ProvenanceError(
                        "native unexpected live framed EOF; owned stopping initiated",
                    ));
                }
                Message::Exited(token, status) => {
                    self.capture_git(token)?;
                    let kind = self.monitor.child_mut(token)?.kind;
                    self.event(
                        "owned-exit-unreaped",
                        Some(self.monitor.elapsed()?),
                        vec![u64::from(token.pid), token.start, status],
                        b"actual pidfd WNOWAIT with both output EOFs",
                    )?;
                    if kind == Kind::Worker {
                        self.worker_status = Some(status);
                        self.monitor.event(Event::Exited(token))?;
                    } else {
                        self.broker.exited(&mut self.monitor, token, status)?;
                    }
                    self.event(
                        "owned-reaped",
                        Some(self.monitor.elapsed()?),
                        vec![u64::from(token.pid), token.start, status],
                        b"actual owned pidfd reap completed",
                    )?;
                }
            }
        }
        let git = self
            .monitor
            .children
            .iter()
            .filter(|c| c.kind == Kind::Git)
            .map(|c| c.child.token)
            .collect::<Vec<_>>();
        for token in git {
            self.capture_git(token)?;
        }
        self.capture_diagnostics()?;
        ensure(
            self.monitor.scheduler.reason().is_none()
                || (self.monitor.children.is_empty() && self.worker_status == Some(0)),
            "native monitor stopped unexpectedly",
        )
    }
    fn capture_git(&mut self, token: Token) -> Result<()> {
        let child = self.monitor.child_mut(token)?;
        if child.kind != Kind::Git {
            return Ok(());
        }
        let old = self
            .git_offsets
            .iter()
            .find(|(t, _)| *t == token)
            .map_or(0, |(_, n)| *n);
        let end = child.raw.len();
        let bytes = child.raw[old..].to_vec();
        self.chunks("git-stdout-raw", token, &bytes)?;
        if let Some((_, offset)) = self.git_offsets.iter_mut().find(|(t, _)| *t == token) {
            *offset = end;
        } else {
            ensure(
                self.git_offsets.len() < 4096,
                "native helper output tracking cap",
            )?;
            self.git_offsets.push((token, end));
        }
        Ok(())
    }
    fn capture_diagnostics(&mut self) -> Result<()> {
        let chunks = if let Some(c) = &mut self.monitor.probe_diagnostics {
            std::mem::take(&mut c.chunks)
        } else {
            Vec::new()
        };
        for (token, bytes) in chunks {
            self.chunks("child-stderr", token, &bytes)?;
        }
        Ok(())
    }
    pub fn stop_worker(&mut self) -> Result<()> {
        ensure(
            self.report && self.broker.active() == 0,
            "native stop before broker completion",
        )?;
        let token = self
            .worker
            .ok_or(ProvenanceError("native worker missing"))?;
        let c = self.monitor.child_mut(token)?;
        ensure(
            c.phase == 4 && c.kind == Kind::Worker,
            "native terminal Stop phase",
        )?;
        c.child.channel.send(Tag::Stop, Vec::new())?;
        c.phase = 5; // EOF allowed only once Channel has actually flushed Stop.
        while !self.monitor.children.is_empty() {
            self.turn()?;
            self.monitor.wait()?;
        }
        ensure(
            self.worker_status == Some(0),
            "native worker unsuccessful exit",
        )
    }
    pub fn collect_samples(&mut self) -> Result<()> {
        let samples = self
            .monitor
            .probe_samples
            .take()
            .ok_or(ProvenanceError("native samples already collected"))?;
        for sample in samples {
            self.event(
                "rss-sample-window",
                Some(sample.finished.1),
                vec![
                    sample.sequence,
                    u64::from(sample.phase),
                    u64::from(sample.valid),
                    u64::from(sample.status),
                    sample.started.0,
                    sample.started.1,
                    sample.finished.0,
                    sample.finished.1,
                    sample.processes.len() as u64,
                ],
                b"actual observed RSS; fixed due windows; no physical-fit claim",
            )?;
            for (pid, start, rss) in sample.processes {
                self.event(
                    "rss-process",
                    Some(sample.finished.1),
                    vec![sample.sequence, pid, start, rss],
                    b"VmRSS kB from actual verified process",
                )?;
            }
        }
        Ok(())
    }
    pub fn complete(&mut self) -> Result<()> {
        ensure(
            self.clock_inner.as_ref().is_none_or(|i| {
                i.confirmed
                    && i.monitor.children.is_empty()
                    && i.monitor.partial.is_empty()
                    && !i.timing.failed
                    && !i.monitor.publication_failed
                    && i.monitor.unresolved
                    && i.monitor.clock_failure.requires_fallback(true)
            }),
            "native inner fault custody incomplete",
        )?;
        ensure(
            self.monitor.children.is_empty()
                && self.monitor.partial.is_empty()
                && !self.monitor.unresolved
                && !self.monitor.publication_failed
                && !self.monitor.clock_failure.requires_fallback(true)
                && !self.reacquire,
            "native unresolved custody/publication/clock",
        )?;
        self.monitor.final_sample()?;
        self.capture_diagnostics()?;
        self.collect_samples()?;
        ensure(
            !self.monitor.unresolved
                && !self.monitor.publication_failed
                && !self.monitor.clock_failure.requires_fallback(true),
            "native final sample newly invalid/unresolved",
        )?;
        super::probe_index::resolve_active(
            &self.state,
            &self.inputs.host,
            Some((
                self.inputs.origin.probe_id.text()?,
                self.selection.check,
                &self.command,
                &self.output,
            )),
        )?;
        self.package.as_ref().expect("probe package").recheck()?;
        self.build.as_ref().expect("probe build").recheck()?;
        self.inputs.origin_file.recheck()?;
        self.inputs.host_file.recheck()?;
        let finished = self.monitor.clock()?;
        ensure(finished.1 <= 60_000_000_000, "native completion deadline")?;
        let p = &self.inputs.origin;
        let output = format!(
            "native-probes/{}/{}/observations.bin",
            p.probe_id.text()?,
            self.selection.check
        );
        let receipt = pin(&p.receipt)?;
        let host = pin(&p.host)?;
        let events = self
            .events
            .iter()
            .map(|e| ProbeEvent {
                name: e.name,
                elapsed: e.elapsed,
                values: e.values.clone(),
                detail: &e.detail,
            })
            .collect();
        let record = ProbeObservation {
            check: self.selection.check,
            origin_path: self.selection.origin.as_os_str().as_bytes(),
            origin_hash: &self.selection.hash,
            origin_bytes: &self.inputs.bytes,
            source: p.source.text()?,
            revision: p.revision.text()?,
            executable: p.executable.text()?,
            recipe: p.recipe.text()?,
            receipt,
            host,
            observation_root: p.observation_path.as_bytes(),
            root_identity: p.observation_identity,
            output: &output,
            started: self.started,
            finished,
            threshold: self.threshold,
            faults: self.faults.clone(),
            events,
        };
        let raw = record
            .encode()
            .map_err(|_| ProvenanceError("native bounded observation encoding"))?;
        self.output.recheck()?;
        if publish_operational(self.output.file(), "observations.bin", &raw).is_err() {
            self.monitor.publication_failed = true;
            return Err(ProvenanceError(
                "native observation publication unconfirmed",
            ));
        }
        self.monitor.check_descendants()?;
        self.output.recheck()?;
        self.index.recheck()?;
        ensure(
            self.output
                .retained_file("observations.bin", 8 * 1024 * 1024)?
                .fixed_name_bytes(8 * 1024 * 1024)?
                == raw,
            "native original observation changed before index closure",
        )?;
        if publish_operational(self.index.file(), "observations.bin", &raw).is_err() {
            self.monitor.publication_failed = true;
            return Err(ProvenanceError(
                "native account closure publication unconfirmed",
            ));
        }
        self.index.recheck()?;
        let stdout=serde_json::to_vec(&serde_json::json!({"schema":"hekit-study-supervisor-probe-output-v1","check_id":self.selection.check,"observations":{"root":5,"path":output,"length":raw.len(),"sha256":crate::data::sha256_hex(&raw)}})).map_err(|_|ProvenanceError("native stdout encoding"))?;
        let mut out = std::io::stdout().lock();
        out.write_all(&stdout)?;
        out.write_all(b"\n")?;
        out.flush()?;
        Ok(())
    }
}

impl Session {
    pub fn cleanup(&mut self) {
        if let Some(inner) = &mut self.clock_inner {
            // Also covers failure after custody moved but before fault dispatch.
            if !inner.monitor.children.is_empty() && inner.monitor.scheduler.reason().is_none() {
                let _ = inner.monitor.fail(Reason::ProtocolFailure);
            }
            while !inner.monitor.children.is_empty() {
                if self.monitor.probe_due_with(&inner.monitor).is_err() {
                    self.monitor.unresolved = true;
                }
                inner.monitor.cleanup_turn(&mut inner.timing);
            }
            if !inner.confirmed {
                self.monitor.unresolved = true;
            }
        }
        if !self.monitor.children.is_empty() {
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
    }
    pub fn hold_unresolved(&mut self) {
        if self.clock_inner.as_ref().is_some_and(|i| {
            !i.confirmed
                || !i.monitor.children.is_empty()
                || !i.monitor.partial.is_empty()
                || i.timing.failed
                || i.monitor.publication_failed
        }) || self.monitor.unresolved
            || !self.monitor.partial.is_empty()
            || self.monitor.clock_failure.requires_fallback(true)
            || self.monitor.publication_failed
        {
            loop {
                for p in &mut self.monitor.partial {
                    p.request_boundary_stop();
                    let _ = p.reap_confirmed();
                }
                std::thread::park_timeout(std::time::Duration::from_millis(100));
            }
        }
    }
    pub fn release_original_lock(&mut self) -> Result<()> {
        use rustix::fs::{self, FlockOperation, Mode, OFlags};
        ensure(!self.reacquire, "native lock already released")?;
        self.state.recheck()?;
        let fresh = std::fs::File::from(native::io(fs::openat(
            self.state.file(),
            "execution.lock",
            OFlags::RDWR | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        ))?);
        let old = self.monitor.lock.metadata()?;
        let new = fresh.metadata()?;
        ensure(
            (old.dev(), old.ino()) == (new.dev(), new.ino()),
            "native fresh OFD wrong lock",
        )?;
        self.reacquire = true;
        drop(std::mem::replace(&mut self.monitor.lock, fresh));
        ensure(
            matches!(
                fs::flock(&self.monitor.lock, FlockOperation::NonBlockingLockExclusive),
                Err(rustix::io::Errno::WOULDBLOCK)
            ),
            "inherited original OFD did not block fresh OFD",
        )?;
        self.event(
            "original-ofd-child-only",
            Some(self.monitor.elapsed()?),
            vec![old.dev(), old.ino()],
            b"P original file and spawn aliases dropped; fresh OFD blocked",
        )
    }
    pub fn reacquire_lock(&mut self) -> Result<()> {
        ensure(
            self.reacquire && self.monitor.children.is_empty(),
            "native lock reacquisition before actual reap",
        )?;
        self.state.recheck()?;
        native::io(rustix::fs::flock(
            &self.monitor.lock,
            rustix::fs::FlockOperation::NonBlockingLockExclusive,
        ))?;
        super::markers::resolve_prior(&self.state)?;
        super::probe_index::resolve_active(
            &self.state,
            &self.inputs.host,
            Some((
                self.inputs.origin.probe_id.text()?,
                self.selection.check,
                &self.command,
                &self.output,
            )),
        )?;
        self.reacquire = false;
        self.event(
            "account-lock-reacquired",
            Some(self.monitor.elapsed()?),
            vec![],
            b"fresh OFD exclusive after actual reap; production markers rechecked",
        )
    }
    fn closed_streams(&mut self) -> Result<()> {
        let token = self
            .worker
            .ok_or(ProvenanceError("native closed-stream worker missing"))?;
        self.monitor
            .child_mut(token)?
            .child
            .channel
            .send(Tag::HelperRequest, b"native-close-streams-live-v1".to_vec())?;
        let mut armed = false;
        let (observed, stderr_eof) = loop {
            ensure(
                self.monitor.elapsed()? <= 60_000_000_000,
                "native closed-stream deadline",
            )?;
            let mut loss = None;
            for message in self.monitor.pump()? {
                match message {
                    Message::Frame(t, f) => {
                        ensure(
                            t == token
                                && !armed
                                && f.tag == Tag::ExitAck
                                && f.payload == b"native-close-streams-armed",
                            "native closed-stream acknowledgment",
                        )?;
                        armed = true;
                    }
                    Message::ControlLost {
                        token: t,
                        observed,
                        stderr_eof,
                        write_pending,
                    } => {
                        ensure(
                            armed && t == token && !write_pending,
                            "native clean EOF without queued write boundary",
                        )?;
                        loss = Some((observed, stderr_eof));
                    }
                    Message::Exited(_, _) => {
                        return Err(ProvenanceError(
                            "native child exited before live control-loss observation",
                        ));
                    }
                }
            }
            if let Some(loss) = loss {
                break loss;
            }
            self.monitor.wait()?;
        };
        ensure(
            self.monitor.scheduler.reason() == Some(Reason::ProtocolFailure),
            "live EOF did not initiate automatic protocol stop",
        )?;
        self.faults.push("both-streams-closed-live-injected");
        self.event("framed-eof-live-before-auto-term", Some(observed), vec![u64::from(token.pid), token.start, u64::from(stderr_eof)], b"actual frame-boundary EOF and live acquired pidfd before automatic TERM; value records whether stderr EOF was also observed then")?;
        self.monitor
            .child_mut(token)?
            .child
            .channel
            .send(Tag::Stop, Vec::new())?;
        ensure(
            self.monitor.child_mut(token)?.child.channel.pending(),
            "native unsent Stop queue absent",
        )?;
        let mut waited_both = false;
        while !self.monitor.children.is_empty() {
            ensure(
                self.monitor.elapsed()? <= 60_000_000_000,
                "native closed-stream cleanup deadline",
            )?;
            let exits = self.monitor.drain_stopping(true);
            let c = self.monitor.child_mut(token)?;
            let both = c.drain.stdout_eof && c.drain.stderr_eof;
            let live_at_wait = c.child.status()?.is_none();
            ensure(
                c.child.channel.pending(),
                "stopping drain flushed queued work",
            )?;
            let before = self.monitor.elapsed()?;
            self.monitor.wait_stopping()?;
            let after = self.monitor.elapsed()?;
            ensure(
                after.checked_sub(before).is_some_and(|n| n >= 100_000_000),
                "native stop wait returned before bounded park",
            )?;
            if both {
                waited_both = true;
                self.event("stopping-wait-park", Some(after), vec![before, after, u64::from(live_at_wait)], b"actual both EOFs and queued outgoing Stop; live flag is a separate pre-wait pidfd observation; TERM may already have ended child")?;
            }
            for (t, status) in exits {
                ensure(
                    t == token && status == 31,
                    "native automatic EOF stop must produce actual TERM exit",
                )?;
                self.worker_status = Some(status);
                self.monitor.event(Event::Exited(t))?;
                self.event(
                    "closed-stream-owned-reaped",
                    Some(self.monitor.elapsed()?),
                    vec![u64::from(t.pid), t.start, status],
                    b"actual owned pidfd wait/reap after automatic live-control-loss stopping",
                )?;
            }
        }
        ensure(
            waited_both,
            "native both-EOF queued-output wait was not observed",
        )
    }
    fn rss_windows(&mut self) -> Result<()> {
        // Keep the actual inherited worker alive for two scheduled completions.
        let end = self
            .monitor
            .scheduler
            .due()
            .checked_add(super::schedule::INTERVAL)
            .ok_or(ProvenanceError("native RSS window overflow"))?;
        while self.monitor.scheduler.due() <= end {
            self.turn()?;
            self.monitor.wait()?;
        }
        self.event(
            "two-scheduled-windows-completed",
            Some(self.monitor.elapsed()?),
            vec![end],
            b"actual first-runnable polls; OS scheduling gaps can exceed ten seconds",
        )
    }
    fn rss_kill(&mut self) -> Result<()> {
        let token = self
            .worker
            .ok_or(ProvenanceError("native STOP worker absent"))?;
        let child = self
            .monitor
            .children
            .iter()
            .find(|c| c.child.token == token)
            .ok_or(ProvenanceError("native STOP custody absent"))?;
        child.child.stop_for_probe(&self.monitor.process)?;
        self.faults.push("owned-sigstop-injected");
        loop {
            ensure(
                self.monitor.elapsed()? <= 60_000_000_000,
                "native stopped-state deadline",
            )?;
            let c = self
                .monitor
                .children
                .iter()
                .find(|c| c.child.token == token)
                .ok_or(ProvenanceError("native STOP custody lost"))?;
            if c.child.stopped_for_probe(&self.monitor.process)? {
                break;
            }
            self.monitor.due()?;
            std::thread::park_timeout(std::time::Duration::from_millis(10));
        }
        self.event(
            "owned-stopped-state-observed",
            Some(self.monitor.elapsed()?),
            vec![u64::from(token.pid), token.start],
            b"actual proc State T; pending signal fields not read",
        )?;
        self.threshold = true;
        let now = self.monitor.elapsed()?;
        self.event(
            "threshold-event-injected",
            Some(now),
            vec![super::schedule::RSS_LIMIT + 1],
            b"injected controller event; not measured allocation or RSS",
        )?;
        self.clock_fault_run(true)
    }
    fn clock_fault_run(&mut self, after_terms: bool) -> Result<()> {
        ensure(
            (self.selection.index == 8 && !after_terms)
                || (self.selection.index == 10 && after_terms),
            "native clock fault exact selector",
        )?;
        ensure(
            self.clock_inner.is_none()
                && self.monitor.children.len() == 1
                && self.broker.active() == 0,
            "native clock inner ownership boundary",
        )?;
        let token = self
            .worker
            .ok_or(ProvenanceError("native clock worker absent"))?;
        ensure(
            self.monitor.children[0].child.token == token,
            "native clock child identity",
        )?;
        let scheduler = self.monitor.scheduler.fork_probe()?;
        let mut inner = Monitor::new_probe(
            std::fs::File::from(super::handshake::private_dup(&self.monitor.lock)?),
            self.monitor.origin,
            native::ProcOrigin::capture()?,
        )?;
        inner.scheduler = scheduler;
        inner.initialized = self.monitor.initialized;
        inner.probe_clock_fault(after_terms)?;
        // All fallible setup precedes the move. The retained inner becomes sole
        // Tracked owner; the outer scheduler retains only its original token.
        inner.children = std::mem::take(&mut self.monitor.children);
        self.clock_inner = Some(ClockExperiment {
            monitor: inner,
            timing: super::monitor::CleanupTiming::new(),
            confirmed: false,
        });
        let label = if after_terms {
            "post-term-clock-read-failure-injected"
        } else {
            "pre-term-clock-read-failure-injected"
        };
        self.faults.push(label);
        let now = self.monitor.elapsed()?;
        let injected = {
            let inner = &mut self
                .clock_inner
                .as_mut()
                .expect("retained clock inner")
                .monitor;
            if after_terms {
                inner.event(Event::PhaseSample {
                    started: now,
                    finished: now,
                    rss: Some(super::schedule::RSS_LIMIT + 1),
                })
            } else {
                inner.fail(Reason::ProtocolFailure)
            }
        };
        ensure(
            injected.is_err(),
            "native injected clock did not fail actual path",
        )?;
        {
            let inner = &self
                .clock_inner
                .as_ref()
                .expect("retained clock inner")
                .monitor;
            ensure(
                if after_terms {
                    inner.scheduler.reason() == Some(Reason::RssStop)
                        && inner.scheduler.term_waiting()
                        && inner
                            .probe_signals
                            .iter()
                            .any(|(t, k, issued, at)| *t == token && !k && *issued && at.is_some())
                } else {
                    inner.scheduler.reason().is_none() && inner.probe_signals.is_empty()
                },
                "native clock failure wrong side of actual TERM",
            )?;
        }
        self.event(
            "injected-clock-read-invalid",
            None,
            vec![u64::from(after_terms)],
            b"intentional source fault, not kernel clock outage; invalid read has no elapsed value",
        )?;
        let recovered = {
            let inner = &self
                .clock_inner
                .as_ref()
                .expect("retained clock inner")
                .monitor;
            let recovered = inner.elapsed()?;
            ensure(
                inner.probe_clock_fault_consumed() && inner.clock_failure.requires_fallback(true),
                "native clock recovery cleared failure",
            )?;
            recovered
        };
        self.event("real-clock-recovered-latch-retained", Some(recovered), vec![], b"real checked common-domain read after one injected failure; inner invalidity unchanged")?;
        let mut armed_event = false;
        let mut reaped_status = None;
        loop {
            let inner = self.clock_inner.as_mut().expect("retained clock inner");
            self.monitor.probe_due_with(&inner.monitor)?;
            ensure(
                self.monitor.elapsed()? <= 60_000_000_000,
                "native clock cleanup validity deadline",
            )?;
            if inner.monitor.children.is_empty() {
                break;
            }
            let exits = inner.monitor.cleanup_turn(&mut inner.timing);
            let armed = inner.timing.armed();
            for (t, status) in exits {
                ensure(
                    t == token && reaped_status.is_none(),
                    "native inner actual reap identity/order",
                )?;
                reaped_status = Some(status);
                self.event("clock-fault-owned-exited-reaped", Some(self.monitor.elapsed()?), vec![u64::from(t.pid), t.start, status], b"actual pidfd WNOWAIT, both required EOFs and completed reap in shared cleanup turn")?;
            }
            if armed && !armed_event {
                armed_event = true;
                self.event("cleanup-fallback-armed-observed", Some(self.monitor.elapsed()?), vec![], b"observation after legal TERM batch and fresh Instant start; this timestamp is not the Instant origin")?;
            }
        }
        let expected_status = if after_terms { 19 } else { 31 };
        ensure(
            armed_event && reaped_status == Some(expected_status),
            "native clock fault actual termination missing",
        )?;
        let inner = self.clock_inner.as_mut().expect("retained clock inner");
        ensure(
            !inner.timing.failed
                && inner.monitor.partial.is_empty()
                && !inner.monitor.publication_failed
                && inner.monitor.unresolved
                && inner.monitor.clock_failure.requires_fallback(true),
            "native inner unexpected cleanup failure",
        )?;
        let signals = std::mem::take(&mut inner.monitor.probe_signals);
        let diagnostics = inner
            .monitor
            .probe_diagnostics
            .take()
            .ok_or(ProvenanceError("native inner diagnostics absent"))?;
        let mut last_term = None;
        let mut kill = None;
        for (t, killing, issued, at) in signals {
            ensure(
                t == token && issued && at.is_some(),
                "native inner signal issuance/time missing",
            )?;
            if killing {
                kill = at;
            } else {
                last_term = at;
            }
            self.event(if killing { "clock-fault-owned-kill-issued" } else { "clock-fault-owned-term-issued" }, at, vec![u64::from(t.pid), t.start], b"actual pidfd_send_signal completed before checked timestamp; no pending-bit observation")?;
        }
        if after_terms {
            ensure(
                kill.zip(last_term)
                    .and_then(|(k, t)| k.checked_sub(t))
                    .is_some_and(|n| n >= super::schedule::GRACE),
                "native clock-fallback actual TERM grace missing",
            )?;
        } else {
            ensure(
                kill.is_none() && last_term.is_some(),
                "native pre-TERM fault unexpected signal path",
            )?;
        }
        for (t, bytes) in diagnostics.chunks {
            self.chunks("clock-inner-stderr-bytes", t, &bytes)?;
        }
        let actions = self.monitor.scheduler.probe_retire(token)?;
        ensure(
            actions == vec![super::schedule::Action::Reap(token)],
            "native clock outer retirement",
        )?;
        self.worker_status = reaped_status;
        self.event("clock-inner-unresolved-custody-drained", Some(self.monitor.elapsed()?), vec![u64::from(token.pid), token.start], b"inner failure/latch and original lock retained; outer verified real exit/drain/reap only; not inner resolution or native PASS")?;
        self.clock_inner
            .as_mut()
            .expect("retained clock inner")
            .confirmed = true;
        Ok(())
    }
    fn publisher(&mut self) -> Result<()> {
        let bytes = b"hekit-native-publisher-fixture-v1\n";
        publish_operational(self.output.file(), "publisher.fixture", bytes)
            .map_err(|_| ProvenanceError("native publisher first publication"))?;
        ensure(
            publish_operational(self.output.file(), "publisher.fixture", b"must-not-clobber")
                .is_err(),
            "native publisher clobbered existing name",
        )?;
        let mut held = self.output.retained_file("publisher.fixture", 4096)?;
        ensure(
            held.fixed_name_bytes(4096)? == bytes,
            "native publisher original bytes changed",
        )?;
        self.output.recheck()?;
        self.event(
            "publisher-durable-no-clobber",
            Some(self.monitor.elapsed()?),
            vec![bytes.len() as u64],
            bytes,
        )
    }
    pub fn direct(&mut self) -> Result<()> {
        match self.selection.index {
            1 | 6 => return super::probe_adoption::run(self),
            2 => self.release_original_lock()?,
            3 => return self.closed_streams(),
            8 => return self.clock_fault_run(false),
            9 => self.rss_windows()?,
            10 => {
                self.rss_kill()?;
                return Ok(());
            }
            13 => self.publisher()?,
            _ => {}
        }
        self.stop_worker()?;
        if self.reacquire {
            self.reacquire_lock()?;
        }
        Ok(())
    }
}
pub(super) fn launch(args: &[OsString]) -> Result<()> {
    let mut session = Session::open(args)?;
    let result = session
        .prepare()
        .and_then(|()| session.direct())
        .and_then(|()| session.complete());
    if result.is_err() {
        session.cleanup();
        if session.reacquire {
            session.reacquire_lock()?;
        }
    }
    result
}
