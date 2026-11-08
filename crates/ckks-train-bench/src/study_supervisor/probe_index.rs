//! Fixed account index of original native command/observation bytes; no authority.
use crate::{
    study_provenance::{
        ProvenanceError, Result, ensure,
        resource_operational::{host::HostOrigin, retained::HeldDirectory},
    },
    study_reference::operational::probe_output,
};
use std::{os::unix::ffi::OsStrExt, path::Path};
fn stamp(directory: &HeldDirectory) -> Result<super::history::DirectoryStamp> {
    use std::os::unix::fs::MetadataExt;
    let m = directory.file().metadata()?;
    Ok(super::history::DirectoryStamp {
        identity: (m.dev(), m.ino()),
        size: m.size(),
        links: m.nlink(),
        modified: (m.mtime(), m.mtime_nsec()),
        changed: (m.ctime(), m.ctime_nsec()),
    })
}
fn walk(
    directory: &HeldDirectory,
    scope: super::history::Names,
    consume: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    directory.recheck()?;
    let before = stamp(directory)?;
    let mut read = super::native::io(rustix::fs::Dir::read_from(directory.file()))?;
    // Local supported FS names are unique in an unchanged directory. Preserve
    // descriptor/name custody and reject metadata mutation during the single
    // traversal; advisory account locking is not protection from same-UID edits.
    let result = super::history::visit(
        (&mut read).map(|entry| {
            directory.recheck()?;
            before.unchanged(&stamp(directory)?)?;
            let entry = super::native::io(entry)?;
            let bytes = entry.file_name().to_bytes();
            ensure(bytes.len() <= 128, "native index name cap")?;
            Ok(bytes.to_vec())
        }),
        scope,
        consume,
    );
    directory.recheck()?;
    before.unchanged(&stamp(directory)?)?;
    result
}
fn bytes(dir: &HeldDirectory, name: &str, cap: usize) -> Result<Vec<u8>> {
    let mut file = dir.retained_file(name, cap)?;
    super::history::record_length(file.file.metadata()?.len(), cap)?;
    file.fixed_name_bytes(cap)
}
pub(super) fn resolve(state: &HeldDirectory, host: &HostOrigin) -> Result<()> {
    resolve_active(state, host, None)
}
pub(super) fn resolve_active(
    state: &HeldDirectory,
    host: &HostOrigin,
    active: Option<(&str, &str, &[u8], &HeldDirectory)>,
) -> Result<()> {
    let mut index = state.duplicate()?;
    index.child("native-probes", true, true)?;
    walk(&index, super::history::Names::Probes, |probe| {
        let mut group = index.duplicate()?;
        group.child(probe, false, true)?;
        walk(&group, super::history::Names::Checks, |check| {
            let mut pair = group.duplicate()?;
            pair.child(check, false, true)?;
            let pair_stamp = stamp(&pair)?;
            let is_active = active.is_some_and(|(p, c, _, _)| probe == p && check == c);
            walk(&pair, super::history::Names::Pair(is_active), |_| Ok(()))?;
            if let Some((active_probe, active_check, active_command, original)) = active {
                if probe == active_probe && check == active_check {
                    ensure(
                        bytes(&pair, "command.json", 65536)? == active_command
                            && bytes(original, "command.json", 65536)? == active_command,
                        "active native original/indexed command mismatch",
                    )?;
                    pair.recheck()?;
                    pair_stamp.unchanged(&stamp(&pair)?)?;
                    return Ok(());
                }
            }
            let command = bytes(&pair, "command.json", 65536)?;
            let raw = bytes(&pair, "observations.bin", 8 * 1024 * 1024)?;
            probe_output::index_join(&command, &raw, probe, check)
                .map_err(|_| ProvenanceError("native index pair mismatch"))?;
            let observed = probe_output::ProbeObservation::parse(&raw)
                .map_err(|_| ProvenanceError("native index observation"))?;
            let origin = observed
                .origin()
                .map_err(|_| ProvenanceError("native index original origin"))?;
            ensure(
                origin.account_uid == host.account_uid
                    && origin.account_identity == host.account_identity,
                "native index foreign account",
            )?;
            let root = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
                observed.observation_root,
            )))?;
            ensure(
                root.identity()? == observed.root_identity,
                "native original Observation identity changed",
            )?;
            let original = bytes(&root, observed.output, 8 * 1024 * 1024)?;
            ensure(
                original == raw,
                "native indexed/original observation mismatch",
            )?;
            let command_path = format!("native-probes/{probe}/{check}/command.json");
            ensure(
                bytes(&root, &command_path, 65536)? == command,
                "native indexed/original command mismatch",
            )?;
            let policy = HeldDirectory::open(Path::new(std::ffi::OsStr::from_bytes(
                origin.policy_path.as_bytes(),
            )))?;
            ensure(
                policy.identity()? == origin.policy_identity,
                "native indexed Policy identity changed",
            )?;
            let raw_host = policy
                .retained_file(origin.host.path.text()?, 4096)?
                .bytes((origin.host.length, origin.host.hash.text()?))?;
            let original_host =
                crate::study_provenance::resource_operational::host::parse_host(&raw_host, false)?;
            ensure(
                original_host.physical == host.physical
                    && original_host.account_identity == host.account_identity,
                "native index foreign physical host",
            )?;
            walk(&pair, super::history::Names::Pair(false), |_| Ok(()))?;
            pair.recheck()?;
            pair_stamp.unchanged(&stamp(&pair)?)?;
            root.recheck()?;
            policy.recheck()?;
            Ok(())
        })
    })?;
    index.recheck()
}
pub(super) fn create(
    state: &HeldDirectory,
    probe: &str,
    check: &str,
    command: &[u8],
) -> Result<HeldDirectory> {
    let mut index = state.duplicate()?;
    index.child("native-probes", true, true)?;
    index.child(probe, true, true)?;
    index.exclusive_child(check)?;
    crate::study_provenance::publication::publish_operational(
        index.file(),
        "command.json",
        command,
    )
    .map_err(|_| ProvenanceError("native account pending publication unconfirmed"))?;
    index.recheck()?;
    Ok(index)
}
