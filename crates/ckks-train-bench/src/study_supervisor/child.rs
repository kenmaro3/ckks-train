//! Same-executable private worker/helper boundaries. Flags alone create no owner.
use super::{
    broker::{self, WorkerOwner},
    handshake,
    protocol::{self, Payload, Tag},
};
use crate::{
    study_provenance::{
        ProvenanceError, Result, ensure, resource_operational::retained::HeldDirectory,
    },
    study_reference::operational as wire,
};
use std::{
    ffi::OsString,
    os::unix::{ffi::OsStrExt, process::CommandExt},
    path::Path,
};
pub(super) fn claim_id(boot: &[u8], pid: u32, start: u64) -> String {
    let mut hash = crate::data::Sha256State::new();
    hash.update(boot);
    hash.update(&u64::from(pid).to_le_bytes());
    hash.update(&start.to_le_bytes());
    String::from_utf8(hash.finish_fixed().to_vec()).expect("SHA ASCII")
}
fn wait_run(channel: &mut protocol::Channel, parent: &handshake::ParentGuard) -> Result<Vec<u8>> {
    loop {
        parent.recheck()?;
        let frames = channel.turn()?;
        ensure(!channel.eof, "supervisor EOF before Run")?;
        if let Some(frame) = frames.into_iter().next() {
            ensure(frame.tag == Tag::Run, "Run or stop boundary")?;
            return Ok(frame.payload);
        }
        broker::wait_channel(channel)?;
    }
}
fn acknowledge(
    channel: &mut protocol::Channel,
    parent: &handshake::ParentGuard,
    payload: &[u8],
) -> Result<()> {
    ensure(!channel.pending(), "Ready not drained before Run")?;
    channel.send(Tag::Ready, crate::data::sha256_hex(payload).into_bytes())?;
    while channel.pending() {
        parent.recheck()?;
        channel.flush()?;
        broker::wait_channel(channel)?;
    }
    parent.recheck()
}
pub(super) fn worker(
    args: &[OsString],
    root: &Path,
    approval: &str,
    work: wire::Requested,
) -> Result<WorkerOwner> {
    let (mut channel, parent, hello) = handshake::child_channel(args)?;
    let raw = wait_run(&mut channel, &parent)?;
    let mut r = Payload::new(&raw)?;
    let expected_approval =
        std::str::from_utf8(r.item()?).map_err(|_| ProvenanceError("Run approval UTF8"))?;
    let source = std::str::from_utf8(r.item()?).map_err(|_| ProvenanceError("Run source UTF8"))?;
    ensure(
        expected_approval == approval && r.blob(4096)? == work.bytes(),
        "original worker request/approval changed",
    )?;
    let length = r.number()?;
    ensure(length <= 2_097_152, "Run claim cap")?;
    let hash =
        std::str::from_utf8(r.item()?).map_err(|_| ProvenanceError("Run claim hash UTF8"))?;
    let roots = r.blob(6)?.to_vec();
    ensure(
        roots.windows(2).all(|p| p[0] < p[1]) && roots.iter().all(|r| *r <= 5),
        "Run root coverage",
    )?;
    r.done()?;
    let id = claim_id(&hello.boot, hello.parent.pid, hello.parent.start);
    let observation = HeldDirectory::open(&root.join("results"))?;
    let mut claim = observation.retained_file(&format!("resource/{id}/claim.bin"), 2_097_152)?;
    let bytes = claim.bytes((length, hash))?;
    wire::run_claim_matches(
        &bytes,
        &work,
        approval,
        source,
        &id,
        std::str::from_utf8(&hello.boot).map_err(|_| ProvenanceError("Run boot UTF8"))?,
        (u64::from(hello.parent.pid), hello.parent.start),
        (u64::from(hello.child.pid), hello.child.start),
    )
    .map_err(|_| ProvenanceError("live Run/durable claim join"))?;
    parent.recheck()?;
    claim.recheck()?;
    acknowledge(&mut channel, &parent, &raw)?;
    let lock = parent.lock_clone()?;
    Ok(WorkerOwner::authenticated(
        channel,
        parent,
        lock,
        work,
        approval.into(),
        source.into(),
        roots,
    ))
}
pub(super) fn helper(args: &[OsString]) -> Result<()> {
    ensure(
        args.len() == 2
            && args[0] == "--private-helper"
            && matches!(args[1].as_bytes(), b"python" | b"git"),
        "private helper mode",
    )?;
    let (mut channel, parent, _) = handshake::child_channel(args)?;
    let raw = wait_run(&mut channel, &parent)?;
    let mut command = protocol::decode_exec(&raw)?;
    acknowledge(&mut channel, &parent, &raw)?;
    parent.recheck()?;
    // Parent reads exactly one framed Ready, then switches Git to raw. Even if
    // exec writes immediately, Channel never reads beyond that frame boundary.
    // Original fd0/fd1 remain inherited. No replacement/null input is installed.
    channel.stream.set_nonblocking(false)?;
    Err(command.exec().into())
}
