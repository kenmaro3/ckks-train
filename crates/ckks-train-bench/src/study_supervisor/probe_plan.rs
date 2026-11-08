//! Fixed native-only selector. It is never a RequestedWork/owner constructor.
use crate::study_provenance::{ProvenanceError, Result, ensure};
use std::{ffi::OsString, os::unix::ffi::OsStrExt, path::PathBuf};
pub(super) struct Selection {
    pub index: usize,
    pub check: &'static str,
    pub origin: PathBuf,
    pub hash: String,
}
impl Selection {
    pub fn parse(args: &[OsString]) -> Result<Self> {
        ensure(
            args.len() == 6
                && args
                    .iter()
                    .all(|a| a.as_bytes().len() <= 4096 && !a.as_bytes().contains(&0)),
            "native probe exact argv bounds",
        )?;
        ensure(
            args[0] == "--native-supervisor-check"
                && args[2] == "--probe-origin"
                && args[4] == "--probe-origin-sha256",
            "native probe fixed options",
        )?;
        let index = crate::study_reference::operational::NATIVE_CHECKS
            .iter()
            .position(|id| args[1].as_bytes() == id.as_bytes())
            .ok_or(ProvenanceError("unknown native probe"))?;
        ensure(
            crate::study_provenance::resource_operational::approval::absolute(args[3].as_bytes()),
            "native probe absolute selector",
        )?;
        let hash = args[5].as_bytes();
        ensure(
            hash.len() == 64
                && hash
                    .iter()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)),
            "native probe selected hash",
        )?;
        Ok(Self {
            index,
            check: crate::study_reference::operational::NATIVE_CHECKS[index],
            origin: PathBuf::from(&args[3]),
            hash: std::str::from_utf8(hash)
                .map_err(|_| ProvenanceError("native hash UTF8"))?
                .into(),
        })
    }
}
