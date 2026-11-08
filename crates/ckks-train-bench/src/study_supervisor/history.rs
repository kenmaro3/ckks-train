//! Streaming retained-history grammar; no lifetime storage/read budget.
use crate::study_provenance::{ProvenanceError, Result, ensure};
#[derive(Clone, Copy)]
pub(super) enum Names {
    Probes,
    Checks,
    Pair(bool),
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct DirectoryStamp {
    pub identity: (u64, u64),
    pub size: u64,
    pub links: u64,
    pub modified: (i64, i64),
    pub changed: (i64, i64),
}
impl DirectoryStamp {
    pub fn unchanged(&self, now: &Self) -> Result<()> {
        ensure(self == now, "native history directory mutated")
    }
}
pub(super) fn record_length(length: u64, cap: usize) -> Result<()> {
    ensure(length <= cap as u64, "native history record cap")
}
pub(super) fn visit(
    entries: impl IntoIterator<Item = Result<Vec<u8>>>,
    scope: Names,
    mut consume: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut mask = 0u16;
    for entry in entries {
        let bytes = entry?;
        if matches!(bytes.as_slice(), b"." | b"..") {
            continue;
        }
        ensure(
            !bytes.is_empty() && bytes.len() <= 128,
            "native history name bound",
        )?;
        let name =
            std::str::from_utf8(&bytes).map_err(|_| ProvenanceError("native history name UTF8"))?;
        let bit = match scope {
            Names::Probes => {
                ensure(
                    bytes
                        .iter()
                        .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')),
                    "native probe ID directory",
                )?;
                None
            }
            Names::Checks => Some(
                crate::study_reference::operational::NATIVE_CHECKS
                    .iter()
                    .position(|s| *s == name)
                    .ok_or(ProvenanceError("unknown native index check"))?,
            ),
            Names::Pair(_) => Some(match name {
                "command.json" => 0,
                "observations.bin" => 1,
                _ => return Err(ProvenanceError("unknown native index pair file")),
            }),
        };
        if let Some(bit) = bit {
            let bit = 1u16 << bit;
            ensure(mask & bit == 0, "native history duplicate local name")?;
            mask |= bit;
        }
        consume(name)?;
    }
    match scope {
        Names::Probes => Ok(()),
        Names::Checks => ensure(mask != 0, "incomplete native index group"),
        Names::Pair(active) => ensure(
            mask == if active { 1 } else { 3 },
            "unresolved native index pair",
        ),
    }
}
