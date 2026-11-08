//! Bounded native-probe observations. Parsing these bytes grants no authority.
use super::*;
use crate::study_provenance::resource_operational::probe::{ProbeOrigin, parse_probe};
use evidence::wire::{Cursor, Git, Wire};
const CAP: usize = 8 * 1024 * 1024;
const DOMAIN: &str = "hekit-study-supervisor-probe-observations-v1";
pub(crate) struct ProbeEvent<'a> {
    pub name: &'a str,
    pub elapsed: Option<u64>,
    pub values: Vec<u64>,
    pub detail: &'a [u8],
}
pub(crate) struct ProbeObservation<'a> {
    pub check: &'a str,
    pub origin_path: &'a [u8],
    pub origin_hash: &'a str,
    pub origin_bytes: &'a [u8],
    pub source: &'a str,
    pub revision: &'a str,
    pub executable: &'a str,
    pub recipe: &'a str,
    pub receipt: FileReference,
    pub host: FileReference,
    pub observation_root: &'a [u8],
    pub root_identity: (u64, u64),
    pub output: &'a str,
    pub started: (u64, u64),
    pub finished: (u64, u64),
    pub threshold: bool,
    pub faults: Vec<&'a str>,
    pub events: Vec<ProbeEvent<'a>>,
}
fn checked(value: bool, field: &str) -> Result<()> {
    ensure(value, DocumentErrorCode::CrossLinkMismatch, field)
}
impl<'a> ProbeObservation<'a> {
    pub fn origin(&self) -> Result<ProbeOrigin> {
        parse_probe(self.origin_bytes, false).map_err(|_| {
            err(
                DocumentErrorCode::CrossLinkMismatch,
                "probe.original.origin",
            )
        })
    }
    fn validate(&self) -> Result<()> {
        let p = self.origin()?;
        checked(
            sources::NATIVE_CHECKS.contains(&self.check)
                && crate::study_provenance::resource_operational::approval::absolute(
                    self.origin_path,
                ),
            "probe.check/path",
        )?;
        checked(
            self.origin_hash == sha256_hex(self.origin_bytes)
                && self.source.as_bytes() == p.source.as_bytes()
                && self.revision.as_bytes() == p.revision.as_bytes()
                && self.executable.as_bytes() == p.executable.as_bytes()
                && self.recipe.as_bytes() == p.recipe.as_bytes(),
            "probe.origin.tuple",
        )?;
        let same =
            |a: &FileReference,
             b: &crate::study_provenance::resource_operational::approval::OriginRef| {
                a.root == b.root
                    && a.path.as_bytes() == b.path.as_bytes()
                    && a.length == b.length
                    && a.hash.as_bytes() == b.hash.as_bytes()
            };
        checked(
            same(&self.receipt, &p.receipt)
                && same(&self.host, &p.host)
                && self.observation_root == p.observation_path.as_bytes()
                && self.root_identity == p.observation_identity,
            "probe.original.refs",
        )?;
        checked(
            self.output
                == format!(
                    "native-probes/{}/{}/observations.bin",
                    p.probe_id
                        .text()
                        .map_err(|_| err(DocumentErrorCode::OutOfRange, "probe.id"))?,
                    self.check
                ),
            "probe.relative.output",
        )?;
        checked(
            self.started.0 <= self.finished.0
                && self.started.1 <= self.finished.1
                && self.finished.1 <= 60_000_000_000,
            "probe.common.clock/deadline",
        )?;
        checked(
            self.threshold == (self.check == "rss-term-grace-kill")
                && self.faults.len() <= 16
                && !self.events.is_empty()
                && self.events.len() <= 4096,
            "probe.flags/events",
        )?;
        for (i, fault) in self.faults.iter().enumerate() {
            target::valid_id(fault, "probe.fault")?;
            checked(!self.faults[..i].contains(fault), "probe.duplicate.fault")?;
        }
        for event in &self.events {
            target::valid_id(event.name, "probe.event")?;
            checked(
                event.values.len() <= 16
                    && event.detail.len() <= 4096
                    && (!event.values.is_empty() || !event.detail.is_empty()),
                "probe.empty/event.bounds",
            )?;
            if event.name == "common-clock-conversion-loss-injected" {
                checked(
                    event.elapsed.is_none() && self.faults.contains(&event.name),
                    "probe.clock.loss.absent",
                )?;
            }
            if let Some(elapsed) = event.elapsed {
                checked(
                    elapsed >= self.started.1 && elapsed <= self.finished.1,
                    "probe.event.clock",
                )?;
            }
        }
        Ok(())
    }
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        ensure(
            bytes.len() <= CAP,
            DocumentErrorCode::TooLarge,
            "probe.whole.cap",
        )?;
        let mut c = Cursor::new(bytes);
        checked(
            c.bounded_text(128)? == ENCODING_VERSION
                && c.bounded_text(128)? == DOMAIN
                && c.tag(0)? == 0,
            "probe.domain",
        )?;
        let check = Id::read(&mut c)?.0;
        let origin_path = c.bounded_bytes(4096)?;
        let origin_hash = H::read(&mut c)?.0;
        let origin_bytes = c.bounded_bytes(4096)?;
        let source = H::read(&mut c)?.0;
        let revision = Git::read(&mut c)?.0;
        let executable = H::read(&mut c)?.0;
        let recipe = Id::read(&mut c)?.0;
        let receipt = FileReference::from_wire(&evidence::FileRef::read(&mut c)?);
        let host = FileReference::from_wire(&evidence::FileRef::read(&mut c)?);
        let observation_root = c.bounded_bytes(4096)?;
        let root_identity = (c.u64()?, c.u64()?);
        let output = c.bounded_text(4096)?;
        let a = Clock::read(&mut c)?;
        let b = Clock::read(&mut c)?;
        let started = (a.unix_ns, a.elapsed_since_operation_start_ns);
        let finished = (b.unix_ns, b.elapsed_since_operation_start_ns);
        let threshold = c.tag(1)? == 1;
        let count = c.bounded_count(16, 9)?;
        let mut faults = Vec::with_capacity(count);
        for _ in 0..count {
            faults.push(Id::read(&mut c)?.0);
        }
        let count = c.bounded_count(4096, 26)?;
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let name = Id::read(&mut c)?.0;
            let elapsed = Option::<u64>::read(&mut c)?;
            let count = c.bounded_count(16, 8)?;
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(c.u64()?);
            }
            let detail = c.bounded_bytes(4096)?;
            events.push(ProbeEvent {
                name,
                elapsed,
                values,
                detail,
            });
        }
        c.finish()?;
        let value = Self {
            check,
            origin_path,
            origin_hash,
            origin_bytes,
            source,
            revision,
            executable,
            recipe,
            receipt,
            host,
            observation_root,
            root_identity,
            output,
            started,
            finished,
            threshold,
            faults,
            events,
        };
        value.validate()?;
        Ok(value)
    }
    fn encoded_len(&self) -> Result<usize> {
        let mut n = 0usize;
        let mut add = |v: usize| -> Result<()> {
            n = bounded_add(n, v, CAP, "probe.aggregate.preflight")?;
            Ok(())
        };
        for value in [
            ENCODING_VERSION.len(),
            DOMAIN.len(),
            self.check.len(),
            self.origin_path.len(),
            self.origin_hash.len(),
            self.origin_bytes.len(),
            self.source.len(),
            self.revision.len(),
            self.executable.len(),
            self.recipe.len(),
            self.observation_root.len(),
            self.output.len(),
        ] {
            add(8)?;
            add(value)?;
        }
        add(1 + 16 + 36 + 1 + 8 + 8)?;
        for r in [&self.receipt, &self.host] {
            add(1 + 8 + 8 + 8)?;
            add(r.path.len())?;
            add(r.hash.len())?;
        }
        for fault in &self.faults {
            add(8)?;
            add(fault.len())?;
        }
        for event in &self.events {
            add(8)?;
            add(event.name.len())?;
            add(1 + if event.elapsed.is_some() { 8 } else { 0 })?;
            add(8)?;
            add(event
                .values
                .len()
                .checked_mul(8)
                .ok_or_else(|| err(DocumentErrorCode::TooLarge, "probe.values"))?)?;
            add(8)?;
            add(event.detail.len())?;
        }
        Ok(n)
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let length = self.encoded_len()?;
        // Each small piece uses the unchanged shared wire encoder. Only this
        // exact probe envelope receives an independently preflighted 8MiB sink.
        let mut output = Vec::new();
        output
            .try_reserve_exact(length)
            .map_err(|_| err(DocumentErrorCode::TooLarge, "probe.allocation"))?;
        let mut e = Encoder::new(DocumentLayout::Start);
        e.text(ENCODING_VERSION)?;
        e.text(DOMAIN)?;
        e.byte(0)?;
        Id(self.check).write(&mut e)?;
        Bytes(self.origin_path).write(&mut e)?;
        H(self.origin_hash).write(&mut e)?;
        Bytes(self.origin_bytes).write(&mut e)?;
        H(self.source).write(&mut e)?;
        Git(self.revision).write(&mut e)?;
        H(self.executable).write(&mut e)?;
        Id(self.recipe).write(&mut e)?;
        self.receipt.wire().write(&mut e)?;
        self.host.wire().write(&mut e)?;
        Bytes(self.observation_root).write(&mut e)?;
        e.u64(self.root_identity.0)?;
        e.u64(self.root_identity.1)?;
        Str::<4096>(self.output).write(&mut e)?;
        clock(self.started.0, self.started.1).write(&mut e)?;
        clock(self.finished.0, self.finished.1).write(&mut e)?;
        e.byte(u8::from(self.threshold))?;
        e.u64(self.faults.len() as u64)?;
        for fault in &self.faults {
            Id(fault).write(&mut e)?;
        }
        e.u64(self.events.len() as u64)?;
        output.extend_from_slice(&e.bytes);
        for event in &self.events {
            let mut e = Encoder::new(DocumentLayout::Start);
            Id(event.name).write(&mut e)?;
            event.elapsed.write(&mut e)?;
            e.u64(event.values.len() as u64)?;
            for value in &event.values {
                e.u64(*value)?;
            }
            Bytes(event.detail).write(&mut e)?;
            bounded_add(
                output.len(),
                e.bytes.len(),
                length,
                "probe.preflighted.append",
            )?;
            output.extend_from_slice(&e.bytes);
        }
        checked(output.len() == length, "probe.exact.encoded.length")?;
        Ok(output)
    }
    pub fn argv_matches(&self, argv: &[Vec<u8>]) -> Result<()> {
        checked(
            argv.len() == 7
                && argv[2] == self.check.as_bytes()
                && argv[4] == self.origin_path
                && argv[6] == self.origin_hash.as_bytes(),
            "probe.raw.argv",
        )
    }
}
pub(crate) fn command_projection(bytes: &[u8]) -> Result<(Vec<Vec<u8>>, Vec<u8>)> {
    preflight_layout(bytes, DocumentLayout::Resource)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|e| json_err(e, "probe.command"))?;
    let mut f = Fields::new(raw, &["schema", "argv", "cwd"], "probe.command")?;
    checked(
        f.field::<Text<128>>("schema", "probe")?.0 == "hekit-study-supervisor-probe-command-v1",
        "probe.command.schema",
    )?;
    let actual = f
        .field::<List<sources::Arg, 256>>("argv", "probe")?
        .0
        .into_iter()
        .map(|v| v.0)
        .collect::<Vec<_>>();
    let dir = f.field::<sources::Arg>("cwd", "probe")?.0;
    checked(
        crate::study_provenance::resource_operational::approval::absolute(&dir),
        "probe.command.cwd",
    )?;
    Ok((actual, dir))
}
pub(crate) fn command(bytes: &[u8], argv: &[Vec<u8>], cwd: &[u8]) -> Result<()> {
    let (actual, dir) = command_projection(bytes)?;
    checked(actual == argv && dir == cwd, "probe.command.original")
}
pub(crate) fn index_join(command: &[u8], raw: &[u8], probe: &str, check: &str) -> Result<()> {
    let observation = ProbeObservation::parse(raw)?;
    let origin = observation.origin()?;
    checked(
        origin.probe_id.as_bytes() == probe.as_bytes() && observation.check == check,
        "probe.index.identity",
    )?;
    let (argv, cwd) = command_projection(command)?;
    checked(
        argv.len() == 7
            && crate::study_provenance::resource_operational::approval::absolute(&argv[0])
            && argv[1] == b"--native-supervisor-check"
            && argv[3] == b"--probe-origin"
            && argv[5] == b"--probe-origin-sha256",
        "probe.index.command.options",
    )?;
    observation.argv_matches(&argv)?;
    checked(
        observation.observation_root == [cwd.as_slice(), b"/results"].concat(),
        "probe.index.original.cwd",
    )
}
pub(crate) fn check_stdout(
    bytes: &[u8],
    observation: &ProbeObservation<'_>,
    raw_observation: &[u8],
) -> Result<()> {
    checked(
        bytes.last() == Some(&b'\n') && !bytes[..bytes.len() - 1].contains(&b'\n'),
        "probe.stdout.single.line",
    )?;
    preflight_layout(bytes, DocumentLayout::Resource)?;
    let raw: &RawValue = serde_json::from_slice(bytes).map_err(|e| json_err(e, "probe.stdout"))?;
    let mut f = Fields::new(raw, &["schema", "check_id", "observations"], "probe.stdout")?;
    let reference = f.field::<resource::FileRef>("observations", "probe.stdout")?;
    checked(
        f.field::<Text<128>>("schema", "probe.stdout")?.0
            == "hekit-study-supervisor-probe-output-v1"
            && f.field::<Text<128>>("check_id", "probe.stdout")?.0 == observation.check
            && reference.root.0 == 5
            && reference.path.0 == observation.output
            && reference.length == raw_observation.len() as u64
            && reference.sha256.0 == sha256_hex(raw_observation),
        "probe.stdout.original.observation",
    )
}
