//! Bounded kernel text and aggregate storage accounting, not physical-fit proof.
use crate::study_provenance::{ProvenanceError, Result, ensure};
fn decimal(b: &[u8]) -> Result<u64> {
    ensure(
        !b.is_empty() && b.iter().all(u8::is_ascii_digit),
        "kernel decimal",
    )?;
    b.iter().try_fold(0u64, |n, v| {
        n.checked_mul(10)
            .and_then(|n| n.checked_add(u64::from(v - b'0')))
            .ok_or(ProvenanceError("kernel decimal overflow"))
    })
}
pub(super) fn rss_kib(bytes: &[u8]) -> Result<u64> {
    ensure(bytes.len() <= 8 * 1024 * 1024, "proc byte cap")?;
    let mut found = None;
    for line in bytes.split(|b| *b == b'\n') {
        if let Some(tail) = line.strip_prefix(b"VmRSS:") {
            ensure(found.is_none(), "duplicate VmRSS")?;
            let mut fields = tail
                .split(u8::is_ascii_whitespace)
                .filter(|b| !b.is_empty());
            let value = decimal(fields.next().ok_or(ProvenanceError("missing VmRSS"))?)?;
            ensure(
                fields.next() == Some(b"kB".as_slice()) && fields.next().is_none(),
                "VmRSS unit/fields",
            )?;
            found = Some(value);
        }
    }
    found.ok_or(ProvenanceError("missing VmRSS"))
}
pub(super) fn start_ticks(bytes: &[u8], pid: u32) -> Result<u64> {
    ensure(bytes.len() <= 8 * 1024 * 1024, "proc stat cap")?;
    let first = bytes
        .iter()
        .position(|b| *b == b' ')
        .ok_or(ProvenanceError("proc stat PID"))?;
    ensure(
        decimal(&bytes[..first])? == u64::from(pid),
        "proc stat PID mismatch",
    )?;
    ensure(bytes.get(first + 1) == Some(&b'('), "proc stat comm")?;
    let end = bytes
        .iter()
        .rposition(|b| *b == b')')
        .ok_or(ProvenanceError("proc stat comm end"))?;
    let value = bytes[end + 1..]
        .split(u8::is_ascii_whitespace)
        .filter(|b| !b.is_empty())
        .nth(19)
        .ok_or(ProvenanceError("proc stat start field"))?;
    let value = decimal(value)?;
    ensure(value > 0, "zero process start")?;
    Ok(value)
}
pub(super) fn pid_field(raw: &[u8], key: &[u8]) -> Result<u32> {
    ensure(raw.len() <= 4096, "kernel identity field cap")?;
    let mut found = None;
    for line in raw.split(|b| *b == b'\n') {
        if let Some(tail) = line.strip_prefix(key) {
            ensure(found.is_none(), "duplicate kernel identity field")?;
            let mut fields = tail
                .split(u8::is_ascii_whitespace)
                .filter(|f| !f.is_empty());
            let value = decimal(fields.next().ok_or(ProvenanceError("missing kernel PID"))?)?;
            ensure(
                fields.next().is_none() && value > 0 && value <= i32::MAX as u64,
                "kernel PID fields/range",
            )?;
            found = Some(value as u32);
        }
    }
    found.ok_or(ProvenanceError("kernel PID field unavailable"))
}
pub(super) struct Storage {
    raw_bytes: u64,
    raw_files: u64,
}
impl Storage {
    pub fn new() -> Self {
        Self {
            raw_bytes: 0,
            raw_files: 0,
        }
    }
    pub fn add_raw(&mut self, length: u64) -> Result<()> {
        let bytes = self
            .raw_bytes
            .checked_add(length)
            .ok_or(ProvenanceError("raw storage overflow"))?;
        let files = self
            .raw_files
            .checked_add(1)
            .ok_or(ProvenanceError("raw file count overflow"))?;
        ensure(
            length <= 8 * 1024 * 1024 && bytes <= 64 * 1024 * 1024 && files <= 4096,
            "raw aggregate storage cap",
        )?;
        self.raw_bytes = bytes;
        self.raw_files = files;
        Ok(())
    }
}

/// A sent STOP is not an observed stopped state.
pub(super) fn stopped(raw: &[u8]) -> Result<bool> {
    ensure(raw.len() <= 4096, "process state cap")?;
    let mut found = None;
    for line in raw.split(|b| *b == b'\n') {
        if let Some(tail) = line.strip_prefix(b"State:") {
            ensure(found.is_none(), "duplicate process state")?;
            let mut fields = tail
                .split(u8::is_ascii_whitespace)
                .filter(|f| !f.is_empty());
            let state = fields
                .next()
                .ok_or(ProvenanceError("missing process state"))?;
            ensure(
                matches!(state, b"R" | b"S" | b"D" | b"T" | b"t" | b"Z" | b"I")
                    && fields.next().is_some(),
                "unknown/malformed process state",
            )?;
            found = Some(matches!(state, b"T" | b"t"));
        }
    }
    found.ok_or(ProvenanceError("missing process state"))
}
