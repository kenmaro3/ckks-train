//! Cross-checkout unresolved-custody records. A free flock never clears a marker.
use crate::study_provenance::{ProvenanceError, Result, ensure};
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MarkerKind {
    Pending,
    Closed,
}
pub(super) fn marker_name(name: &[u8]) -> Result<(&str, MarkerKind)> {
    ensure(
        name.len() == 76 || name.len() == 75,
        "unknown host marker name",
    )?;
    let (id, suffix) = name.split_at(64);
    ensure(
        id.iter()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c)),
        "host marker claim ID",
    )?;
    let kind = match suffix {
        b".pending.bin" => MarkerKind::Pending,
        b".closed.bin" => MarkerKind::Closed,
        _ => return Err(ProvenanceError("unknown host marker suffix")),
    };
    Ok((
        std::str::from_utf8(id).map_err(|_| ProvenanceError("host marker UTF8"))?,
        kind,
    ))
}
#[cfg(target_os = "linux")]
pub(super) fn resolve_prior(
    state: &crate::study_provenance::resource_operational::retained::HeldDirectory,
) -> Result<crate::study_provenance::resource_operational::retained::HeldDirectory> {
    use crate::{
        data::sha256_hex, study_provenance::resource_operational::retained::HeldDirectory,
        study_reference::operational as wire,
    };
    use std::{os::unix::ffi::OsStrExt, path::Path};
    let mut claims = state.duplicate()?;
    claims.child("claims", true, true)?;
    let mut directory = super::native::io(rustix::fs::Dir::read_from(claims.file()))?;
    let mut count = 0usize;
    for entry in &mut directory {
        let entry = super::native::io(entry)?;
        let name = entry.file_name().to_bytes();
        if matches!(name, b"." | b"..") {
            continue;
        }
        count = count
            .checked_add(1)
            .ok_or(ProvenanceError("host marker count overflow"))?;
        ensure(count <= 65536, "host marker directory cap")?;
        let (id, kind) = marker_name(name)?;
        // Both directions are checked: orphan closed records are not ignored.
        let pending = claims
            .retained_file(&format!("{id}.pending.bin"), 4096)?
            .fixed_name_bytes(4096)?;
        let closed = claims
            .retained_file(&format!("{id}.closed.bin"), 4096)?
            .fixed_name_bytes(4096)?;
        if kind == MarkerKind::Closed {
            continue;
        }
        let (marker_id, root_path, root_identity) = wire::prior_marker(&pending)
            .map_err(|_| ProvenanceError("invalid host pending record"))?;
        ensure(marker_id == id, "host marker ID/name mismatch")?;
        let root = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(&root_path)))?;
        ensure(
            root.identity()? == root_identity,
            "prior Observation origin changed",
        )?;
        let end = wire::closed_reference(&closed, &sha256_hex(&pending))
            .map_err(|_| ProvenanceError("invalid host closed record"))?;
        ensure(
            end.root == 5 && end.path == format!("resource/{id}/supervisor.end.bin"),
            "prior closure fixed end path",
        )?;
        let raw = root
            .retained_file(&end.path, 2_097_152)?
            .bytes((end.length, &end.hash))?;
        let reference = wire::prior_end_pending(&raw)
            .map_err(|_| ProvenanceError("prior custody unresolved"))?;
        ensure(
            reference.root == 5 && reference.path == format!("resource/{id}/pending.bin"),
            "prior pending fixed path",
        )?;
        let observed = root
            .retained_file(&reference.path, 2_097_152)?
            .bytes((reference.length, &reference.hash))?;
        wire::prior_pending_matches(&pending, &observed)
            .map_err(|_| ProvenanceError("prior pending identity mismatch"))?;
        root.recheck()?;
    }
    claims.recheck()?;
    Ok(claims)
}
