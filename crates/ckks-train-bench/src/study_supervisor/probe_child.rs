//! Same-executable native-only leaf. No production WorkerOwner is constructed.
use super::{
    handshake::{ParentGuard, Projection},
    native,
    protocol::{self, Channel, Frame, Tag},
};
use crate::study_provenance::{ProvenanceError, Result, ensure};
use std::ffi::OsString;
pub(super) struct Prepared {
    pub channel: Channel,
    pub guard: ParentGuard,
    pub hello: Projection,
    pub check: usize,
    pub role: &'static str,
}
pub(super) fn check_deadline(guard: &ParentGuard, origin: u64) -> Result<()> {
    guard.recheck()?;
    ensure(
        protocol::elapsed(origin, native::monotonic()?)? <= 60_000_000_000,
        "native child validity deadline",
    )
}
pub(super) fn flush(channel: &mut Channel, guard: &ParentGuard, origin: u64) -> Result<()> {
    while channel.pending() {
        check_deadline(guard, origin)?;
        channel.flush()?;
        if channel.pending() {
            super::broker::wait_channel(channel)?;
        }
    }
    Ok(())
}
pub(super) fn frame(channel: &mut Channel, guard: &ParentGuard, origin: u64) -> Result<Frame> {
    loop {
        check_deadline(guard, origin)?;
        let frames = channel.turn()?;
        ensure(!channel.eof, "native child parent EOF")?;
        if let Some(frame) = frames.into_iter().next() {
            return Ok(frame);
        }
        super::broker::wait_channel(channel)?;
    }
}
pub(super) fn prepare(args: &[OsString]) -> Result<Prepared> {
    ensure(
        args.len() == 3 && args[0] == "--private-native-probe",
        "private native exact argv",
    )?;
    let check = crate::study_reference::operational::NATIVE_CHECKS
        .iter()
        .position(|id| args[1] == *id)
        .ok_or(ProvenanceError("private native scenario"))?;
    let role = match args[2].to_str() {
        Some("idle") => "idle",
        Some("middle") if [1, 6].contains(&check) => "middle",
        Some("leaf") if [1, 6].contains(&check) => "leaf",
        _ => return Err(ProvenanceError("private native role")),
    };
    let (mut channel, guard, hello) = super::handshake::child_channel(args)?;
    let run = frame(&mut channel, &guard, hello.origin)?;
    ensure(run.tag == Tag::Run, "private native Run required")?;
    channel.send(
        Tag::Ready,
        crate::data::sha256_hex(&run.payload).into_bytes(),
    )?;
    flush(&mut channel, &guard, hello.origin)?;
    if role == "leaf" {
        ensure(
            run.payload == b"hekit-native-leaf-v1",
            "private native leaf payload",
        )?;
        return Ok(Prepared {
            channel,
            guard,
            hello,
            check,
            role,
        });
    }
    let mut payload = protocol::Payload::new(&run.payload)?;
    ensure(
        payload.item()? == b"hekit-native-broker-probe-v1" && payload.number()? == 2,
        "native initial broker payload",
    )?;
    let mut commands = Vec::new();
    for _ in 0..2 {
        commands.push(protocol::decode_exec(payload.blob(protocol::PAYLOAD_CAP)?)?);
    }
    payload.done()?;
    let broker = super::broker::Broker::authenticated_probe(channel, guard, hello.origin);
    let mut results = Vec::new();
    for command in commands {
        results.push(broker.command(command, 65)?);
    }
    let (mut channel, guard) = broker.into_probe_transport()?;
    let process = native::ProcOrigin::capture()?;
    ensure(
        process.own_token == hello.child,
        "native child token changed",
    )?;
    let nice = native::io(rustix::process::getpriority_process(None))?;
    ensure(nice >= 0, "native child priority unsupported")?;
    let parent =
        rustix::process::getppid().ok_or(ProvenanceError("native child parent missing"))?;
    let elapsed = protocol::elapsed(hello.origin, native::monotonic()?)?;
    let mut report = b"HKPRB1".to_vec();
    for n in [
        u64::from(process.own_token.pid),
        process.own_token.start,
        parent.as_raw_nonzero().get() as u64,
        nice as u64,
        elapsed,
        process.own_rss()?,
    ] {
        report.extend(n.to_le_bytes());
    }
    report.extend((results.len() as u64).to_le_bytes());
    for result in results {
        report.extend((result.len() as u64).to_le_bytes());
        report.extend(result);
    }
    channel.send(Tag::ExitAck, report)?;
    flush(&mut channel, &guard, hello.origin)?;
    Ok(Prepared {
        channel,
        guard,
        hello,
        check,
        role,
    })
}
pub(super) fn idle(mut child: Prepared) -> Result<()> {
    ensure(child.role == "idle", "native idle role")?;
    let stop = frame(&mut child.channel, &child.guard, child.hello.origin)?;
    if child.check == 3
        && stop.tag == Tag::HelperRequest
        && stop.payload == b"native-close-streams-live-v1"
    {
        child
            .channel
            .send(Tag::ExitAck, b"native-close-streams-armed".to_vec())?;
        flush(&mut child.channel, &child.guard, child.hello.origin)?;
        let null = std::fs::File::from(native::io(rustix::fs::open(
            "/dev/null",
            rustix::fs::OFlags::WRONLY | rustix::fs::OFlags::NOFOLLOW | rustix::fs::OFlags::CLOEXEC,
            rustix::fs::Mode::empty(),
        ))?);
        use std::os::unix::fs::FileTypeExt;
        ensure(
            null.metadata()?.file_type().is_char_device(),
            "native stderr null character device",
        )?;
        native::io(rustix::stdio::dup2_stderr(&null))?;
        child.channel.stream.shutdown(std::net::Shutdown::Both)?;
        loop {
            check_deadline(&child.guard, child.hello.origin)?;
            std::thread::park_timeout(std::time::Duration::from_millis(100));
        }
    }
    ensure(
        stop.tag == Tag::Stop && stop.payload.is_empty(),
        "native idle closed stop",
    )
}
