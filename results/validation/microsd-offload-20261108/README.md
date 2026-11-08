# Local ckks-train storage migration to MicroSD

User authorized moving cold project data to the inserted MicroSD and explicitly
restricted this operation to ckks-train. Other projects and Downloads were not
inspected or moved. Destination: `/Volumes/Untitled/ckks-train-storage`, writable
exFAT, volume UUID `CAE1A977-C00C-3033-8605-069E26F8E6D2`.

Seven cold roots were archived directly to the card: four completed preview
bundles, current fixed-source package qualification workspace, retired Linux
checkout stage, and the dedicated older package-check workspace in the Mac temp
directory. No active target/Linux cross cache, working source, current evidence
ledger, experiment input, or scientific result was removed. Vendor source stays
in the user's private external archive; none is added to this public evidence.

bsdtar preserved mode/link and Apple metadata representation. Every normal entry,
regular-file SHA256/content, mode and symlink target was checked against the local
inventory, and the unchanged local trees were rechecked. Each whole destination
tar was then rehashed before deletion. Exact archive hashes/entry counts/original
paths are in the copy record; per-entry compressed manifests are retained on the
card and in the local ignored ledger. The operation did not dereference outside
symlinks; associated vendor targets are in the old-package-workspace archive.

Copy original session67216 terminal ec0ae9 exit0; removal original session71861
terminal95c244 exit0. All local originals were still present when copy verification
completed. Immediately before removal, no own open files were found for each cold
root. Removal completed2026-11-08T05:36:20.430811Z. Observed internal free space
rose from1,782,063,104 to2,807,615,488 bytes, about1.03GB. This is an observation
across the operation, not an exact attribution immune to other application writes.
Source allocated-byte sums can count shared/hardlinked blocks more than once.

Future large cold exports/snapshots are directed to the card by the local helper
`tools/release/_ext/storage_root.py`; it requires the exact mounted volume UUID,
external writable device and same-device directories, with no SSD fallback.
The helper's actual exports lookup returned the card path. It is local operator
configuration, not a library requirement or clean-checkout dependency. Active
build and provenance paths retain their original filesystem semantics.

The reviewed intermediate PDF and TeX ZIP were additionally copied unchanged to
`exports/2026-11-08-working-draft` and hash-checked. They remain drafts, with FULL-2
pending; no publication tag or arXiv submission changed. Old reports preserve
execution-time paths; the storage mapping identifies where cold originals now live.
