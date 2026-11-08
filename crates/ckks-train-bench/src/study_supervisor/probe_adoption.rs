//! The two closed native-only P -> M -> L experiments. Production stays direct.
use super::{
    handshake::Projection,
    native::{self, KernelChild, OwnedChild, PartialChild},
    probe_child::{self, Prepared},
    probe_driver::Session,
    probe_transfer::{self, Boundary, TransferProjection},
    protocol::{self, Channel, Frame, Tag},
    schedule::{self, Event, Token},
};
use crate::study_provenance::{ProvenanceError, Result, ensure};
use std::{
    ffi::OsString,
    fs::File,
    io::Read,
    os::{
        fd::{AsFd, OwnedFd},
        unix::net::UnixStream,
    },
};
fn park() {
    std::thread::park_timeout(std::time::Duration::from_millis(10));
}
fn deadline(origin: u64) -> Result<u64> {
    let now = protocol::elapsed(origin, native::monotonic()?)?;
    ensure(now <= 60_000_000_000, "native adoption validity deadline")?;
    Ok(now)
}
fn flush(channel: &mut Channel, origin: u64) -> Result<()> {
    while channel.pending() {
        deadline(origin)?;
        channel.flush()?;
        if channel.pending() {
            park();
        }
    }
    Ok(())
}
fn frame(channel: &mut Channel, origin: u64) -> Result<Frame> {
    loop {
        deadline(origin)?;
        if let Some(f) = channel.turn()?.into_iter().next() {
            return Ok(f);
        }
        ensure(!channel.eof, "native adoption control EOF")?;
        park();
    }
}
fn send(channel: &mut Channel, tag: Tag, raw: Vec<u8>, origin: u64) -> Result<()> {
    channel.send(tag, raw)?;
    flush(channel, origin)
}
fn rights_send(
    channel: &Channel,
    b: Boundary,
    scenario: u8,
    rights: &[std::os::fd::BorrowedFd<'_>],
    origin: u64,
) -> Result<()> {
    while !probe_transfer::send_rights(&channel.stream, b, scenario, rights)? {
        deadline(origin)?;
        park();
    }
    Ok(())
}
fn rights_recv(
    channel: &Channel,
    b: Boundary,
    scenario: u8,
    origin: u64,
    loose: &mut Vec<OwnedFd>,
) -> Result<Vec<OwnedFd>> {
    loop {
        deadline(origin)?;
        match probe_transfer::receive_rights(&channel.stream, b, scenario) {
            Ok(Some(fds)) => return Ok(fds),
            Ok(None) => park(),
            Err(error) => {
                loose.extend(error.rights);
                return Err(error.error);
            }
        }
    }
}
fn cleanup_child(
    child: &mut OwnedChild,
    process: &native::ProcOrigin,
    extra: Option<&KernelChild>,
) -> bool {
    if child.reaped {
        return false;
    }
    let mut invalid = child.signal(process, false).is_err();
    if let Some(leaf) = extra {
        invalid |= leaf
            .signal_for_probe(process, child.token.pid, false)
            .is_err();
    }
    let mut common = native::monotonic().ok();
    let mut fallback = common.is_none().then(std::time::Instant::now);
    invalid |= common.is_none();
    let mut killed = false;
    let mut eof = [false; 2];
    loop {
        let due = if killed {
            false
        } else if let Some(start) = common {
            match native::monotonic()
                .ok()
                .and_then(|now| now.checked_sub(start))
            {
                Some(duration) => duration >= schedule::GRACE,
                None => {
                    invalid = true;
                    let _ = child.signal(process, false);
                    if let Some(leaf) = extra {
                        let _ = leaf.signal_for_probe(process, child.token.pid, false);
                    }
                    common = None;
                    fallback = Some(std::time::Instant::now());
                    false
                }
            }
        } else {
            fallback
                .and_then(|start| std::time::Instant::now().checked_duration_since(start))
                .is_some_and(|d| d >= std::time::Duration::from_secs(5))
        };
        if due {
            killed = true;
            invalid |= child.signal(process, true).is_err();
            if let Some(leaf) = extra {
                invalid |= leaf
                    .signal_for_probe(process, child.token.pid, true)
                    .is_err();
            }
        }
        let mut raw = [0; 65536];
        for (i, reader) in [
            &mut child.stderr as &mut dyn Read,
            &mut child.channel.stream as &mut dyn Read,
        ]
        .into_iter()
        .enumerate()
        {
            if !eof[i] {
                match reader.read(&mut raw) {
                    Ok(0) => eof[i] = true,
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(_) => invalid = true,
                }
            }
        }
        match child.status() {
            Ok(Some(_)) if eof == [true, true] => {
                if child.reap().is_ok() {
                    break;
                }
                invalid = true;
            }
            Ok(_) => {}
            Err(_) => invalid = true,
        }
        std::thread::park_timeout(std::time::Duration::from_millis(100));
    }
    invalid
}
/// M retains L and every partial handle until the exact P acknowledgment.
pub(super) fn middle(mut parent: Prepared) -> Result<()> {
    ensure(
        parent.role == "middle" && [1, 6].contains(&parent.check),
        "native middle boundary",
    )?;
    let process = native::ProcOrigin::capture()?;
    let mut child: Option<OwnedChild> = None;
    let mut partial: Option<PartialChild> = None;
    let mut loose = Vec::new();
    let mut handed = false;
    let mut transfer_started = false;
    let result = (|| -> Result<()> {
        let trigger = probe_child::frame(&mut parent.channel, &parent.guard, parent.hello.origin)?;
        ensure(trigger.tag == Tag::HelperRequest, "native adoption trigger")?;
        let rescue = if parent.check == 6 {
            probe_transfer::validate_rescue(
                &trigger.payload,
                Boundary::RescueToMiddle,
                &parent.hello,
                None,
            )?;
            send(
                &mut parent.channel,
                Tag::Ready,
                crate::data::sha256_hex(&trigger.payload).into_bytes(),
                parent.hello.origin,
            )?;
            transfer_started = true;
            let mut fds = rights_recv(
                &parent.channel,
                Boundary::RescueToMiddle,
                6,
                parent.hello.origin,
                &mut loose,
            )?;
            let rescue = UnixStream::from(fds.remove(0));
            send(
                &mut parent.channel,
                Tag::ExitAck,
                b"rescue-received".to_vec(),
                parent.hello.origin,
            )?;
            Some(rescue)
        } else {
            ensure(
                trigger.payload == b"hekit-native-adoption-v1",
                "native adoption trigger payload",
            )?;
            None
        };
        parent.guard.recheck()?;
        let lock = parent.guard.lock_clone()?;
        let args: Vec<OsString> = vec![
            crate::study_reference::operational::NATIVE_CHECKS[parent.check].into(),
            "leaf".into(),
        ];
        let spawned = OwnedChild::spawn(
            &std::env::current_exe()?,
            "--private-native-probe",
            &args,
            &lock,
            &process,
        );
        drop(lock);
        child = Some(match spawned {
            Ok(c) => c,
            Err(f) => {
                partial = f.partial;
                return Err(f.error);
            }
        });
        let c = child.as_mut().expect("owned native leaf");
        let mut invocation = vec![OsString::from("--private-native-probe")];
        invocation.extend(args);
        let hello = Projection {
            parent: process.own_token,
            child: c.token,
            lock: parent.hello.lock,
            origin: parent.hello.origin,
            boot: parent.hello.boot,
            executable: parent.hello.executable,
            invocation: super::handshake::invocation(&invocation)?,
        };
        send(
            &mut c.channel,
            Tag::Hello,
            hello.encode().to_vec(),
            hello.origin,
        )?;
        let ready = frame(&mut c.channel, hello.origin)?;
        ensure(
            ready.tag == Tag::Ready && ready.payload == hello.encode(),
            "native leaf actual Hello",
        )?;
        send(
            &mut c.channel,
            Tag::Run,
            b"hekit-native-leaf-v1".to_vec(),
            hello.origin,
        )?;
        let ready = frame(&mut c.channel, hello.origin)?;
        ensure(
            ready.tag == Tag::Ready
                && ready.payload == crate::data::sha256_hex(b"hekit-native-leaf-v1").as_bytes(),
            "native leaf actual Run",
        )?;
        if let Some(rescue) = rescue {
            let raw =
                probe_transfer::rescue_record(Boundary::RescueToLeaf, &parent.hello, Some(&hello))?;
            send(
                &mut c.channel,
                Tag::HelperRequest,
                raw.clone(),
                hello.origin,
            )?;
            let ready = frame(&mut c.channel, hello.origin)?;
            ensure(
                ready.tag == Tag::Ready
                    && ready.payload == crate::data::sha256_hex(&raw).as_bytes(),
                "native leaf rescue acknowledgment",
            )?;
            rights_send(
                &c.channel,
                Boundary::RescueToLeaf,
                6,
                &[rescue.as_fd()],
                hello.origin,
            )?;
            let ack = frame(&mut c.channel, hello.origin)?;
            ensure(
                ack.tag == Tag::ExitAck && ack.payload == b"rescue-received",
                "native leaf rescue receipt",
            )?;
        }
        parent.guard.recheck()?;
        ensure(
            c.parent_pid()? == process.own_token.pid && c.recheck_live(&process)?,
            "native leaf direct live identity",
        )?;
        let transfer = TransferProjection {
            scenario: parent.check as u8,
            rights: 2,
            outer: parent.hello.parent,
            inner: hello,
            session: parent.hello.invocation,
        }
        .encode();
        transfer_started = true;
        send(
            &mut parent.channel,
            Tag::HelperReply,
            transfer.clone(),
            parent.hello.origin,
        )?;
        let ready = probe_child::frame(&mut parent.channel, &parent.guard, parent.hello.origin)?;
        ensure(
            ready.tag == Tag::Ready
                && ready.payload == crate::data::sha256_hex(&transfer).as_bytes(),
            "native outer custody boundary acknowledgment",
        )?;
        rights_send(
            &parent.channel,
            Boundary::CustodyToOuter,
            parent.check as u8,
            &c.rights(),
            parent.hello.origin,
        )?;
        let ack = probe_child::frame(&mut parent.channel, &parent.guard, parent.hello.origin)?;
        ensure(
            ack.tag == Tag::ExitAck && ack.payload == crate::data::sha256_hex(&transfer).as_bytes(),
            "native exact acquired custody acknowledgment",
        )?;
        handed = true;
        Ok(())
    })();
    if !handed {
        if let Some(c) = &mut child {
            if cleanup_child(c, &process, None) {
                loop {
                    park();
                }
            }
        }
        if partial.is_some() || !loose.is_empty() || transfer_started {
            loop {
                if let Some(p) = &mut partial {
                    p.request_boundary_stop();
                    let _ = p.reap_confirmed();
                }
                park();
            }
        }
    }
    // Only acknowledged acquired custody permits M to exit with L still live.
    result
}
pub(super) fn leaf(mut parent: Prepared) -> Result<()> {
    ensure(
        parent.role == "leaf" && [1, 6].contains(&parent.check),
        "native leaf boundary",
    )?;
    let mut loose = Vec::new();
    let mut rescue = None;
    if parent.check == 6 {
        let record = probe_child::frame(&mut parent.channel, &parent.guard, parent.hello.origin)?;
        ensure(
            record.tag == Tag::HelperRequest && record.payload.len() == 7 + 2 * Projection::LENGTH,
            "native leaf rescue record",
        )?;
        let outer = Projection::decode(&record.payload[7..7 + Projection::LENGTH])?;
        probe_transfer::validate_rescue(
            &record.payload,
            Boundary::RescueToLeaf,
            &outer,
            Some(&parent.hello),
        )?;
        send(
            &mut parent.channel,
            Tag::Ready,
            crate::data::sha256_hex(&record.payload).into_bytes(),
            parent.hello.origin,
        )?;
        let fds = rights_recv(
            &parent.channel,
            Boundary::RescueToLeaf,
            6,
            parent.hello.origin,
            &mut loose,
        );
        let mut fds = match fds {
            Ok(f) => f,
            Err(e) => {
                if !loose.is_empty() {
                    loop {
                        park();
                    }
                }
                return Err(e);
            }
        };
        let socket = UnixStream::from(fds.remove(0));
        let peer = native::io(rustix::net::sockopt::socket_peercred(&socket))?;
        super::handshake::peer_matches(
            native::io(rustix::net::sockopt::socket_type(&socket))?
                == rustix::net::SocketType::STREAM,
            peer.pid.as_raw_nonzero().get() as u32,
            peer.uid.as_raw(),
            outer.parent.pid,
            rustix::process::getuid().as_raw(),
            rustix::process::geteuid().as_raw(),
        )?;
        rescue = Some(Channel::new(socket)?);
        send(
            &mut parent.channel,
            Tag::ExitAck,
            b"rescue-received".to_vec(),
            parent.hello.origin,
        )?;
    }
    loop {
        deadline(parent.hello.origin)?;
        let frames = parent.channel.turn()?;
        ensure(frames.is_empty(), "native leaf cannot receive normal work")?;
        if parent.channel.eof {
            break;
        }
        park();
    }
    ensure(
        parent.guard.recheck().is_err(),
        "actual ParentGuard must reject lost original parent",
    )?;
    if let Some(mut rescue) = rescue {
        let process = native::ProcOrigin::capture()?;
        let ppid = rustix::process::getppid()
            .ok_or(ProvenanceError("native leaf adopted parent missing"))?;
        let mut raw = b"HKPEOF1".to_vec();
        for n in [
            u64::from(process.own_token.pid),
            process.own_token.start,
            ppid.as_raw_nonzero().get() as u64,
            deadline(parent.hello.origin)?,
        ] {
            raw.extend(n.to_le_bytes());
        }
        send(&mut rescue, Tag::ExitAck, raw, parent.hello.origin)?;
        let stop = frame(&mut rescue, parent.hello.origin)?;
        ensure(
            stop.tag == Tag::Stop && stop.payload == b"native-rescue-stop",
            "rescue only bounded shutdown",
        )?;
        Ok(())
    } else {
        // No new work after EOF. P already holds the acquired pidfd and exercises TERM.
        loop {
            deadline(parent.hello.origin)?;
            park();
        }
    }
}
struct Custody {
    leaf: Option<KernelChild>,
    loose: Vec<OwnedFd>,
    partial_identity: bool,
    adopted: bool,
    invalid_inner: Option<super::monitor::Monitor>,
    transfer_started: bool,
}
impl Custody {
    fn new() -> Self {
        Self {
            leaf: None,
            loose: Vec::new(),
            partial_identity: false,
            adopted: false,
            invalid_inner: None,
            transfer_started: false,
        }
    }
}
/// Outer experiment retains every received handle across all failures.
pub(super) fn run(s: &mut Session) -> Result<()> {
    let mut custody = Custody::new();
    let result = experiment(s, &mut custody);
    if result.is_err() {
        // Resolve M first so a validated transferred L can become waitable. No raw PID
        // acquisition occurs if the transfer never established its identity.
        if let Some(t) = s.worker {
            if let Some(i) = s.monitor.children.iter().position(|c| c.child.token == t) {
                s.monitor.unresolved |= cleanup_child(
                    &mut s.monitor.children[i].child,
                    &s.monitor.process,
                    custody.leaf.as_ref(),
                );
                s.monitor.children.remove(i);
            }
        }
        if let Some(leaf) = custody.leaf.as_mut().filter(|l| !l.reaped) {
            if !custody.adopted {
                custody.adopted = leaf.confirm_adoption(&s.monitor.process).is_ok();
            }
            if custody.adopted {
                let mut invalid = leaf.signal(&s.monitor.process, false).is_err();
                let mut common = native::monotonic().ok();
                let mut fallback = common.is_none().then(std::time::Instant::now);
                invalid |= common.is_none();
                let mut killed = false;
                loop {
                    let due = if killed {
                        false
                    } else if let Some(start) = common {
                        match native::monotonic()
                            .ok()
                            .and_then(|now| now.checked_sub(start))
                        {
                            Some(duration) => duration >= schedule::GRACE,
                            None => {
                                invalid = true;
                                let _ = leaf.signal(&s.monitor.process, false);
                                common = None;
                                fallback = Some(std::time::Instant::now());
                                false
                            }
                        }
                    } else {
                        fallback
                            .and_then(|start| {
                                std::time::Instant::now().checked_duration_since(start)
                            })
                            .is_some_and(|d| d >= std::time::Duration::from_secs(5))
                    };
                    if due {
                        killed = true;
                        invalid |= leaf.signal(&s.monitor.process, true).is_err();
                    }
                    match leaf.status() {
                        Ok(Some(_)) => {
                            if leaf.reap().is_ok() {
                                break;
                            }
                            invalid = true;
                        }
                        Ok(None) => {}
                        Err(_) => invalid = true,
                    }
                    park();
                }
                s.monitor.unresolved |= invalid;
            } else {
                s.monitor.unresolved = true;
            }
        }
        s.monitor.unresolved |=
            custody.partial_identity || !custody.loose.is_empty() || custody.transfer_started;
        // A real census, not a missing transfer, determines whether any unknown remains.
        if s.monitor
            .process
            .children(s.monitor.process.own_token.pid)
            .map_or(true, |p| !p.is_empty())
        {
            s.monitor.unresolved = true;
        }
        if s.reacquire && !s.monitor.unresolved && s.reacquire_lock().is_err() {
            s.monitor.unresolved = true;
        }
        s.hold_unresolved();
    }
    result
}
fn outer_pulse(s: &mut Session, c: &Custody) -> Result<()> {
    let now = deadline(s.monitor.origin)?;
    if now < s.monitor.scheduler.due() {
        return Ok(());
    }
    let started = s.monitor.clock()?;
    let capture = (|| -> Result<Vec<(u64, u64, u64)>> {
        let mut known = s
            .monitor
            .children
            .iter()
            .map(|v| v.child.token.pid)
            .collect::<Vec<_>>();
        if let Some(l) = &c.leaf {
            known.push(l.token.pid);
        }
        ensure(
            s.monitor
                .process
                .children(s.monitor.process.own_token.pid)?
                .iter()
                .all(|p| known.contains(p)),
            "native outer unknown adopted child",
        )?;
        let mut entries = vec![(
            u64::from(s.monitor.process.own_token.pid),
            s.monitor.process.own_token.start,
            s.monitor.process.own_rss()?,
        )];
        for child in &s.monitor.children {
            if let Some(rss) = child.child.rss(&s.monitor.process)? {
                ensure(
                    s.monitor
                        .process
                        .children(child.child.token.pid)?
                        .iter()
                        .all(|p| c.leaf.as_ref().is_some_and(|l| l.token.pid == *p)),
                    "native unacquired nested custody at sample",
                )?;
                entries.push((
                    u64::from(child.child.token.pid),
                    child.child.token.start,
                    rss,
                ));
            }
        }
        if let Some(leaf) = c.leaf.as_ref().filter(|l| !l.reaped) {
            let mut parents = vec![s.monitor.process.own_token.pid];
            parents.extend(s.monitor.children.iter().map(|v| v.child.token.pid));
            if let Some(rss) = leaf.rss_for_probe(&s.monitor.process, &parents)? {
                ensure(
                    s.monitor.process.children(leaf.token.pid)?.is_empty(),
                    "native leaf spawned descendant",
                )?;
                entries.push((u64::from(leaf.token.pid), leaf.token.start, rss));
            }
        }
        entries.sort_unstable_by_key(|v| v.0);
        Ok(entries)
    })();
    let finished = s.monitor.clock()?;
    let valid = capture.is_ok()
        && finished.1
            <= s.monitor
                .scheduler
                .due()
                .checked_add(schedule::INTERVAL)
                .ok_or(ProvenanceError("native sample window overflow"))?;
    let entries = match capture {
        Ok(v) => v,
        Err(_) => {
            s.monitor.unresolved = true;
            Vec::new()
        }
    };
    let sum = if valid {
        Some(entries.iter().try_fold(0u64, |n, p| {
            n.checked_add(p.2)
                .ok_or(ProvenanceError("native RSS overflow"))
        })?)
    } else {
        None
    };
    s.monitor.event(Event::Sample {
        started: started.1,
        finished: finished.1,
        rss: sum,
    })?;
    let values = crate::study_reference::operational::SampleValues {
        sequence: s.monitor.sequence,
        started,
        finished,
        phase: 0,
        processes: entries,
        valid,
        status: if !valid {
            2
        } else if sum.is_some_and(|n| n > schedule::RSS_LIMIT) {
            1
        } else {
            0
        },
    };
    s.monitor.sequence = s
        .monitor
        .sequence
        .checked_add(1)
        .ok_or(ProvenanceError("native sample sequence overflow"))?;
    let samples = s
        .monitor
        .probe_samples
        .as_mut()
        .ok_or(ProvenanceError("native samples unavailable"))?;
    ensure(samples.len() < 4096, "native sample count cap")?;
    samples.push(values);
    ensure(
        valid && s.monitor.scheduler.reason().is_none(),
        "native adoption due sample stopped",
    )
}
fn outer_frame(s: &mut Session, c: &Custody, token: Token) -> Result<Frame> {
    loop {
        outer_pulse(s, c)?;
        let child = s.monitor.child_mut(token)?;
        if let Some(f) = child.child.channel.turn()?.into_iter().next() {
            return Ok(f);
        }
        ensure(!child.child.channel.eof, "native outer control EOF")?;
        park();
    }
}
fn outer_send(s: &mut Session, c: &Custody, token: Token, tag: Tag, raw: Vec<u8>) -> Result<()> {
    outer_pulse(s, c)?;
    s.monitor.child_mut(token)?.child.channel.send(tag, raw)?;
    while s.monitor.child_mut(token)?.child.channel.pending() {
        outer_pulse(s, c)?;
        s.monitor.child_mut(token)?.child.channel.flush()?;
        if s.monitor.child_mut(token)?.child.channel.pending() {
            park();
        }
    }
    Ok(())
}
fn experiment(s: &mut Session, custody: &mut Custody) -> Result<()> {
    let token = s.worker.ok_or(ProvenanceError("native middle absent"))?;
    ensure(
        s.monitor.children.len() == 1,
        "native nested topology before helpers reaped",
    )?;
    let outer = s.monitor.child_mut(token)?.hello.clone();
    let origin = outer.origin;
    let scenario = s.selection.index as u8;
    let mut rescue = None;
    custody.transfer_started = true;
    if scenario == 6 {
        let (local, remote) = UnixStream::pair()?;
        let raw = probe_transfer::rescue_record(Boundary::RescueToMiddle, &outer, None)?;
        outer_send(s, custody, token, Tag::HelperRequest, raw.clone())?;
        let ready = outer_frame(s, custody, token)?;
        ensure(
            ready.tag == Tag::Ready && ready.payload == crate::data::sha256_hex(&raw).as_bytes(),
            "native middle rescue boundary",
        )?;
        loop {
            outer_pulse(s, custody)?;
            if probe_transfer::send_rights(
                &s.monitor.child_mut(token)?.child.channel.stream,
                Boundary::RescueToMiddle,
                6,
                &[remote.as_fd()],
            )? {
                break;
            }
            park();
        }
        drop(remote);
        let ack = outer_frame(s, custody, token)?;
        ensure(
            ack.tag == Tag::ExitAck && ack.payload == b"rescue-received",
            "native middle rescue received",
        )?;
        rescue = Some(Channel::new(local)?);
    } else {
        outer_send(
            s,
            custody,
            token,
            Tag::HelperRequest,
            b"hekit-native-adoption-v1".to_vec(),
        )?;
    }
    let record = outer_frame(s, custody, token)?;
    ensure(
        record.tag == Tag::HelperReply,
        "native acquired transfer projection",
    )?;
    let projection = TransferProjection::decode(&record.payload, &outer, scenario)?;
    let expected: Vec<OsString> = vec![
        "--private-native-probe".into(),
        s.selection.check.into(),
        "leaf".into(),
    ];
    ensure(
        projection.inner.invocation == super::handshake::invocation(&expected)?,
        "native transferred leaf exact invocation",
    )?;
    outer_send(
        s,
        custody,
        token,
        Tag::Ready,
        crate::data::sha256_hex(&record.payload).into_bytes(),
    )?;
    let mut fds = loop {
        outer_pulse(s, custody)?;
        match probe_transfer::receive_rights(
            &s.monitor.child_mut(token)?.child.channel.stream,
            Boundary::CustodyToOuter,
            scenario,
        ) {
            Ok(Some(fds)) => break fds,
            Ok(None) => park(),
            Err(e) => {
                custody.loose.extend(e.rights);
                return Err(e.error);
            }
        }
    };
    let pidfd = fds.remove(0);
    let proc = File::from(fds.remove(0));
    custody.leaf = Some(
        match KernelChild::received(
            pidfd,
            proc,
            projection.inner.child,
            token.pid,
            &s.monitor.process,
        ) {
            Ok(c) => c,
            Err(error) => {
                custody.partial_identity = true;
                custody.loose.push(error.pidfd);
                custody.loose.push(error.proc.into());
                return Err(error.error);
            }
        },
    );
    s.event(
        "leaf-custody-received-echild",
        Some(deadline(origin)?),
        vec![
            u64::from(projection.inner.child.pid),
            projection.inner.child.start,
            u64::from(token.pid),
        ],
        &record.payload,
    )?;
    // Only L retains the original OFD after M exits. P's rescue and transferred
    // descriptors contain no lock alias; M still owns one until actual exit.
    if scenario == 6 {
        s.release_original_lock()?;
    }
    outer_send(
        s,
        custody,
        token,
        Tag::ExitAck,
        crate::data::sha256_hex(&record.payload).into_bytes(),
    )?;
    let mut stderr_eof = false;
    let mut status = None;
    loop {
        outer_pulse(s, custody)?;
        let c = s.monitor.child_mut(token)?;
        ensure(
            c.child.channel.turn()?.is_empty(),
            "native middle trailing frame",
        )?;
        let mut bytes = [0; 4096];
        match c.child.stderr.read(&mut bytes) {
            Ok(0) => stderr_eof = true,
            Ok(n) => s.chunks("middle-stderr", token, &bytes[..n])?,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }
        let c = s.monitor.child_mut(token)?;
        if status.is_none() {
            status = c.child.status()?;
        }
        if status.is_some() && stderr_eof && c.child.channel.eof {
            break;
        }
        park();
    }
    ensure(status == Some(0), "native middle successful exit")?;
    let i = s
        .monitor
        .children
        .iter()
        .position(|c| c.child.token == token)
        .ok_or(ProvenanceError("native middle custody missing"))?;
    s.monitor.children[i].child.reap()?;
    s.monitor.children.remove(i);
    // Scheduler token is retired only after real reap; its Reap action has already
    // been fulfilled here because native topology intentionally bypasses pump.
    let actions = s.monitor.scheduler.probe_retire(token)?;
    ensure(
        actions == vec![schedule::Action::Reap(token)],
        "native middle retirement actions",
    )?;
    custody
        .leaf
        .as_mut()
        .expect("retained leaf")
        .confirm_adoption(&s.monitor.process)?;
    custody.adopted = true;
    ensure(
        custody
            .leaf
            .as_ref()
            .expect("retained leaf")
            .status()?
            .is_none(),
        "native M EOF must precede live L teardown",
    )?;
    let leaf_token = custody.leaf.as_ref().expect("retained leaf").token;
    s.event("middle-eof-while-leaf-live", Some(deadline(origin)?), vec![u64::from(token.pid), u64::from(leaf_token.pid), leaf_token.start], b"actual M terminal status and both EOFs followed by reap; acquired L pidfd still live before any L Stop/TERM")?;
    s.event(
        "actual-orphan-adopted",
        Some(deadline(origin)?),
        vec![
            u64::from(leaf_token.pid),
            leaf_token.start,
            u64::from(token.pid),
            u64::from(s.monitor.process.own_token.pid),
        ],
        b"same acquired pidfd/proc/start/boot/executable; actual PPid M to P",
    )?;
    if scenario == 1 {
        // This intentionally invalid inner detection is never promoted to resolved.
        let mut inner = super::monitor::Monitor::new_probe(
            std::fs::File::from(super::handshake::private_dup(&s.monitor.lock)?),
            origin,
            native::ProcOrigin::capture()?,
        )?;
        // This is the actual production census/UnknownChild consumer, with L
        // intentionally absent only from its ledger. Outer custody retains L's
        // already authenticated handle until actual cleanup; no scan mints one.
        ensure(
            inner.check_descendants().is_err()
                && inner.unresolved
                && inner.scheduler.reason() == Some(schedule::Reason::UnresolvedChildren),
            "actual omitted-ledger monitor failed to invalidate",
        )?;
        custody.invalid_inner = Some(inner);
        s.faults.push("adopted-ledger-omission-injected");
        s.event("unknown-detection-inner-invalid",Some(deadline(origin)?),vec![u64::from(leaf_token.pid)],b"actual Monitor::check_descendants rejects adopted L absent from its ledger; inner stays invalid; outer acquired handle retains cleanup custody")?;
        ensure(
            custody
                .leaf
                .as_ref()
                .expect("retained leaf")
                .signal(&s.monitor.process, false)?,
            "native adopted TERM target exited early",
        )?;
        s.event(
            "adopted-term-issued",
            Some(deadline(origin)?),
            vec![u64::from(leaf_token.pid), leaf_token.start],
            b"actual retained pidfd TERM",
        )?;
    } else {
        ensure(
            matches!(
                rustix::fs::flock(
                    &s.monitor.lock,
                    rustix::fs::FlockOperation::NonBlockingLockExclusive
                ),
                Err(rustix::io::Errno::WOULDBLOCK)
            ),
            "leaf-only OFD not held after actual M reap",
        )?;
        s.event(
            "leaf-only-ofd-blocked",
            Some(deadline(origin)?),
            vec![u64::from(leaf_token.pid), leaf_token.start],
            b"M reaped and all P original aliases gone; only L inherited original OFD",
        )?;
        let rescue = rescue.as_mut().expect("parent EOF rescue");
        let report = loop {
            outer_pulse(s, custody)?;
            if let Some(frame) = rescue.turn()?.into_iter().next() {
                break frame;
            }
            ensure(
                !rescue.eof,
                "native rescue EOF before original-parent report",
            )?;
            park();
        };
        ensure(
            report.tag == Tag::ExitAck
                && report.payload.len() == 39
                && report.payload.starts_with(b"HKPEOF1"),
            "native true parent EOF report",
        )?;
        let nums = report.payload[7..]
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        ensure(
            nums[0] == u64::from(leaf_token.pid)
                && nums[1] == leaf_token.start
                && nums[2] == u64::from(s.monitor.process.own_token.pid)
                && nums[3] <= deadline(origin)?,
            "native actual parent EOF identity/time",
        )?;
        s.event(
            "original-control-eof-and-guard-refusal",
            Some(nums[3]),
            nums,
            &report.payload,
        )?;
        outer_pulse(s, custody)?;
        rescue.send(Tag::Stop, b"native-rescue-stop".to_vec())?;
        while rescue.pending() {
            outer_pulse(s, custody)?;
            rescue.flush()?;
            if rescue.pending() {
                park();
            }
        }
    }
    let term_completed = deadline(origin)?;
    let mut killed = false;
    loop {
        outer_pulse(s, custody)?;
        let now = deadline(origin)?;
        if let Some(status) = custody.leaf.as_ref().expect("retained leaf").status()? {
            if scenario == 1 {
                ensure(status == 31, "native adopted TERM wait status")?;
            } else {
                ensure(status == 0, "native rescued leaf exit status")?;
            }
            s.event(
                "adopted-exit-unreaped",
                Some(now),
                vec![u64::from(leaf_token.pid), leaf_token.start, status],
                b"actual acquired pidfd WNOWAIT",
            )?;
            custody.leaf.as_mut().expect("retained leaf").reap()?;
            s.event(
                "adopted-reaped",
                Some(deadline(origin)?),
                vec![u64::from(leaf_token.pid), leaf_token.start, status],
                b"actual acquired pidfd reap",
            )?;
            break;
        }
        if !killed
            && now
                .checked_sub(term_completed)
                .is_some_and(|n| n >= schedule::GRACE)
        {
            killed = true;
            if scenario == 6 {
                ensure(
                    custody
                        .leaf
                        .as_ref()
                        .expect("retained leaf")
                        .signal(&s.monitor.process, false)?,
                    "rescue did not exit; TERM",
                )?;
                return Err(ProvenanceError("rescue normal shutdown failed"));
            }
            let _ = custody
                .leaf
                .as_ref()
                .expect("retained leaf")
                .signal(&s.monitor.process, true)?;
        }
        park();
    }
    if let Some(rescue) = &mut rescue {
        loop {
            outer_pulse(s, custody)?;
            ensure(rescue.turn()?.is_empty(), "native rescue trailing frame")?;
            if rescue.eof {
                break;
            }
            park();
        }
    }
    ensure(
        s.monitor
            .process
            .children(s.monitor.process.own_token.pid)?
            .is_empty(),
        "native post-adoption unknown child",
    )?;
    if s.reacquire {
        s.reacquire_lock()?;
    }
    custody.transfer_started = false;
    Ok(())
}
