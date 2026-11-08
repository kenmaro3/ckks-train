//! Bounded framed transport and closed broker command vocabulary.
use crate::study_provenance::{ProvenanceError, Result, ensure};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
pub(super) const PAYLOAD_CAP: usize = 8 * 1024 * 1024;
pub(super) struct DrainState {
    pub status: Option<u64>,
    pub stdout_eof: bool,
    pub stderr_eof: bool,
    bytes: usize,
}
impl DrainState {
    pub fn new() -> Self {
        Self {
            status: None,
            stdout_eof: false,
            stderr_eof: false,
            bytes: 0,
        }
    }
    pub fn stdout_bytes(&mut self, n: usize) -> Result<()> {
        self.bytes = self
            .bytes
            .checked_add(n)
            .ok_or(ProvenanceError("helper output overflow"))?;
        ensure(self.bytes <= PAYLOAD_CAP - 8, "helper output cap")
    }
    pub fn ready(&self) -> bool {
        self.status.is_some() && self.stdout_eof && self.stderr_eof
    }
}
pub(super) struct Payload<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Payload<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        ensure(bytes.len() <= PAYLOAD_CAP, "control payload bounds")?;
        Ok(Self { bytes, offset: 0 })
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(n)
            .ok_or(ProvenanceError("control field overflow"))?;
        ensure(end <= self.bytes.len(), "control field truncated")?;
        let v = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(v)
    }
    pub fn number(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| ProvenanceError("control number"))?,
        ))
    }
    pub fn blob(&mut self, cap: usize) -> Result<&'a [u8]> {
        let n = usize::try_from(self.number()?)
            .map_err(|_| ProvenanceError("control item overflow"))?;
        ensure(n <= cap, "control item cap")?;
        self.take(n)
    }
    pub fn item(&mut self) -> Result<&'a [u8]> {
        let v = self.blob(4096)?;
        ensure(!v.contains(&0), "control item NUL")?;
        Ok(v)
    }
    pub fn done(self) -> Result<()> {
        ensure(
            self.offset == self.bytes.len(),
            "control payload trailing bytes",
        )
    }
}
pub(super) fn item(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    ensure(
        bytes.len() <= 4096 && !bytes.contains(&0),
        "control item bounds",
    )?;
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
    ensure(out.len() <= PAYLOAD_CAP, "control aggregate payload")
}
pub(super) struct HelperRequest {
    pub kind: u8,
    pub cwd: Vec<u8>,
    pub argv: Vec<Vec<u8>>,
}
impl HelperRequest {
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let mut r = Payload::new(bytes)?;
        let kind = r.take(1)?[0];
        ensure(kind <= 1, "helper request kind")?;
        let cwd = r.item()?.to_vec();
        ensure(
            crate::study_provenance::resource_operational::approval::absolute(&cwd),
            "helper canonical cwd",
        )?;
        let count = r.number()?;
        ensure((1..=256).contains(&count), "helper argv count")?;
        let mut argv = Vec::new();
        for _ in 0..count {
            argv.push(r.item()?.to_vec());
        }
        r.done()?;
        Ok(Self { kind, cwd, argv })
    }
}
pub(super) fn encode_exec(command: &std::process::Command) -> Result<Vec<u8>> {
    use std::os::unix::ffi::OsStrExt;
    let mut out = Vec::new();
    item(
        &mut out,
        command
            .get_current_dir()
            .ok_or(ProvenanceError("exec cwd"))?
            .as_os_str()
            .as_bytes(),
    )?;
    let args = std::iter::once(command.get_program())
        .chain(command.get_args())
        .collect::<Vec<_>>();
    ensure(args.len() <= 256, "exec argv count")?;
    out.extend_from_slice(&(args.len() as u64).to_le_bytes());
    for a in args {
        item(&mut out, a.as_bytes())?;
    }
    let env = command.get_envs().collect::<Vec<_>>();
    ensure(env.len() <= 18, "exec environment count")?;
    out.extend_from_slice(&(env.len() as u64).to_le_bytes());
    for (k, v) in env {
        item(&mut out, k.as_bytes())?;
        item(
            &mut out,
            v.ok_or(ProvenanceError("exec explicit env value"))?
                .as_bytes(),
        )?;
    }
    Ok(out)
}
pub(super) fn decode_exec(bytes: &[u8]) -> Result<std::process::Command> {
    use std::os::unix::ffi::OsStrExt;
    let mut r = Payload::new(bytes)?;
    let cwd = r.item()?;
    ensure(
        crate::study_provenance::resource_operational::approval::absolute(cwd),
        "exec absolute cwd",
    )?;
    let n = r.number()?;
    ensure((1..=256).contains(&n), "exec argv count")?;
    let program = r.item()?;
    ensure(
        crate::study_provenance::resource_operational::approval::absolute(program),
        "exec absolute program",
    )?;
    let mut command = std::process::Command::new(std::ffi::OsStr::from_bytes(program));
    for _ in 1..n {
        command.arg(std::ffi::OsStr::from_bytes(r.item()?));
    }
    command
        .current_dir(std::ffi::OsStr::from_bytes(cwd))
        .env_clear();
    let n = r.number()?;
    ensure(n <= 18, "exec environment count")?;
    let mut prior = Vec::new();
    for _ in 0..n {
        let key = r.item()?;
        ensure(
            !key.is_empty() && !key.contains(&b'=') && prior.as_slice() < key,
            "exec environment key order",
        )?;
        let value = r.item()?;
        command.env(
            std::ffi::OsStr::from_bytes(key),
            std::ffi::OsStr::from_bytes(value),
        );
        prior = key.to_vec();
    }
    r.done()?;
    Ok(command)
}
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tag {
    Hello = 0,
    Ready = 1,
    Run = 2,
    HelperRequest = 3,
    HelperReply = 4,
    StartLink = 5,
    ResultLink = 6,
    Stop = 7,
    ExitAck = 8,
}
impl Tag {
    fn decode(v: u8) -> Result<Self> {
        Ok(match v {
            0 => Self::Hello,
            1 => Self::Ready,
            2 => Self::Run,
            3 => Self::HelperRequest,
            4 => Self::HelperReply,
            5 => Self::StartLink,
            6 => Self::ResultLink,
            7 => Self::Stop,
            8 => Self::ExitAck,
            _ => return Err(ProvenanceError("unknown control tag")),
        })
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Frame {
    pub tag: Tag,
    pub sequence: u64,
    pub payload: Vec<u8>,
}
pub(super) fn encode(frame: &Frame) -> Result<Vec<u8>> {
    ensure(frame.payload.len() <= PAYLOAD_CAP, "control payload cap")?;
    let mut out = Vec::with_capacity(frame.payload.len() + 13);
    out.extend_from_slice(&((frame.payload.len() + 9) as u32).to_le_bytes());
    out.push(frame.tag as u8);
    out.extend_from_slice(&frame.sequence.to_le_bytes());
    out.extend_from_slice(&frame.payload);
    Ok(out)
}
pub(super) struct Decoder {
    bytes: Vec<u8>,
    sequence: u64,
}
impl Decoder {
    pub fn new() -> Self {
        Self {
            bytes: Vec::new(),
            sequence: 0,
        }
    }
    pub fn feed(&mut self, b: &[u8]) -> Result<()> {
        ensure(
            self.bytes
                .len()
                .checked_add(b.len())
                .is_some_and(|n| n <= PAYLOAD_CAP + 13),
            "control receive cap",
        )?;
        self.bytes.extend_from_slice(b);
        if self.bytes.len() >= 4 {
            self.length()?;
        }
        Ok(())
    }
    fn length(&self) -> Result<usize> {
        let n = u32::from_le_bytes(
            self.bytes[..4]
                .try_into()
                .map_err(|_| ProvenanceError("control length"))?,
        ) as usize;
        ensure((9..=PAYLOAD_CAP + 9).contains(&n), "control frame cap")?;
        Ok(n + 4)
    }
    pub fn next(&mut self) -> Result<Option<Frame>> {
        if self.bytes.len() < 4 {
            return Ok(None);
        }
        let n = self.length()?;
        if self.bytes.len() < n {
            return Ok(None);
        }
        let tag = Tag::decode(self.bytes[4])?;
        let sequence = u64::from_le_bytes(
            self.bytes[5..13]
                .try_into()
                .map_err(|_| ProvenanceError("control sequence"))?,
        );
        ensure(sequence == self.sequence, "control sequence order")?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(ProvenanceError("control sequence exhausted"))?;
        let payload = self.bytes[13..n].to_vec();
        self.bytes.drain(..n);
        Ok(Some(Frame {
            tag,
            sequence,
            payload,
        }))
    }
    pub fn eof(&self) -> Result<()> {
        ensure(self.bytes.is_empty(), "truncated control EOF")
    }
    fn needed(&self) -> Result<usize> {
        if self.bytes.len() < 4 {
            Ok(4 - self.bytes.len())
        } else {
            Ok(self.length()?.saturating_sub(self.bytes.len()))
        }
    }
}
/// Parent uses only nonblocking bounded turns; child uses the same framing at boundaries.
pub(super) struct Channel {
    pub stream: UnixStream,
    decoder: Decoder,
    out: Vec<u8>,
    offset: usize,
    sequence: u64,
    pub eof: bool,
}
impl Channel {
    pub fn new(stream: UnixStream) -> Result<Self> {
        stream.set_nonblocking(true)?;
        Ok(Self {
            stream,
            decoder: Decoder::new(),
            out: Vec::new(),
            offset: 0,
            sequence: 0,
            eof: false,
        })
    }
    pub fn send(&mut self, tag: Tag, payload: Vec<u8>) -> Result<()> {
        ensure(self.out.is_empty(), "control backpressure")?;
        self.out = encode(&Frame {
            tag,
            sequence: self.sequence,
            payload,
        })?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or(ProvenanceError("control send sequence exhausted"))?;
        self.offset = 0;
        Ok(())
    }
    pub fn pending(&self) -> bool {
        !self.out.is_empty()
    }
    pub fn flush(&mut self) -> Result<()> {
        if !self.out.is_empty() {
            let end = (self.offset + 65_536).min(self.out.len());
            match self.stream.write(&self.out[self.offset..end]) {
                Ok(0) => return Err(ProvenanceError("control write EOF")),
                Ok(n) => self.offset += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => return Err(e.into()),
            }
            if self.offset == self.out.len() {
                self.out.clear();
                self.offset = 0;
            }
        }
        Ok(())
    }
    pub fn turn(&mut self) -> Result<Vec<Frame>> {
        self.flush()?;
        if let Some(frame) = self.decoder.next()? {
            return Ok(vec![frame]);
        }
        let mut scratch = [0; 65_536];
        let need = self.decoder.needed()?.min(scratch.len());
        match self.stream.read(&mut scratch[..need]) {
            Ok(0) => {
                self.decoder.eof()?;
                self.eof = true;
            }
            Ok(n) => self.decoder.feed(&scratch[..n])?,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e.into()),
        }
        let mut frames = Vec::new();
        while let Some(frame) = self.decoder.next()? {
            frames.push(frame);
        }
        Ok(frames)
    }
}
pub(super) fn checked_time(seconds: i64, nanos: i64) -> Result<u64> {
    ensure(
        seconds >= 0 && (0..1_000_000_000).contains(&nanos),
        "invalid monotonic Timespec",
    )?;
    (seconds as u64)
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(nanos as u64))
        .ok_or(ProvenanceError("monotonic conversion overflow"))
}
pub(super) fn elapsed(origin: u64, current: u64) -> Result<u64> {
    current
        .checked_sub(origin)
        .ok_or(ProvenanceError("monotonic before origin"))
}
pub(super) fn git_argv(root: &[u8], suffix: &[&str]) -> Result<Vec<Vec<u8>>> {
    ensure(
        root.len() <= 4096
            && crate::study_provenance::resource_operational::approval::absolute(root),
        "Git fixed root",
    )?;
    ensure(
        matches!(
            suffix,
            ["rev-parse", "--show-toplevel"]
                | ["rev-parse", "HEAD"]
                | [
                    "diff",
                    "--no-ext-diff",
                    "--no-textconv",
                    "--quiet",
                    "HEAD",
                    "--"
                ]
                | ["ls-files", "--others", "--exclude-standard", "-z"]
                | ["ls-files", "--stage", "-z"]
        ),
        "Git command whitelist",
    )?;
    let mut argv = vec![
        b"--no-pager".to_vec(),
        b"-c".to_vec(),
        b"core.fsmonitor=false".to_vec(),
        b"-c".to_vec(),
        b"core.untrackedCache=false".to_vec(),
        b"-C".to_vec(),
        root.to_vec(),
    ];
    argv.extend(suffix.iter().map(|s| s.as_bytes().to_vec()));
    Ok(argv)
}

pub(super) fn broker_edge(kind: u8, request: u8, active: usize) -> Result<()> {
    ensure(
        (kind == 0 && request == 0 && active == 0) || (kind == 1 && request == 1 && active == 1),
        "broker request/process nesting",
    )
}

/// Native-only capture into the adopted bounded observation-event payloads.
pub(super) struct DiagnosticCapture {
    pub bytes: usize,
    pub chunks: Vec<(super::schedule::Token, Vec<u8>)>,
}
impl DiagnosticCapture {
    pub fn new() -> Self {
        Self {
            bytes: 0,
            chunks: Vec::new(),
        }
    }
    pub fn push(&mut self, token: super::schedule::Token, bytes: &[u8]) -> Result<()> {
        let total = self
            .bytes
            .checked_add(bytes.len())
            .ok_or(ProvenanceError("probe diagnostic length overflow"))?;
        let chunks = bytes
            .len()
            .checked_add(4095)
            .ok_or(ProvenanceError("probe diagnostic chunk overflow"))?
            / 4096;
        ensure(
            total <= 8 * 1024 * 1024
                && self
                    .chunks
                    .len()
                    .checked_add(chunks)
                    .is_some_and(|n| n <= 4096),
            "probe diagnostic event cap",
        )?;
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(chunks)
            .map_err(|_| ProvenanceError("probe diagnostic allocation"))?;
        for chunk in bytes.chunks(4096) {
            let mut copy = Vec::new();
            copy.try_reserve_exact(chunk.len())
                .map_err(|_| ProvenanceError("probe diagnostic allocation"))?;
            copy.extend_from_slice(chunk);
            pending.push((token, copy));
        }
        self.chunks
            .try_reserve_exact(chunks)
            .map_err(|_| ProvenanceError("probe diagnostic allocation"))?;
        self.chunks.extend(pending);
        self.bytes = total;
        Ok(())
    }
}

/// Observed private channel lifecycle; EOF never establishes exit or reap.
#[derive(Clone, Copy)]
pub(super) struct ControlObservation {
    pub kind: u8, // Worker0/Python1/Git2, supplied by the actual owned ledger.
    pub phase: u8,
    pub result_present: bool,
    pub write_pending: bool,
    pub eof: bool,
    pub status: Option<u64>,
}
impl ControlObservation {
    pub fn lost(self) -> bool {
        if !self.eof || self.status.is_some() {
            return false;
        }
        let raw_git = self.kind == 2 && self.phase == 4;
        let terminal = self.phase == 5
            && !self.write_pending
            && ((self.kind == 1 && self.result_present) || self.kind == 0);
        !raw_git && !terminal
    }
}
