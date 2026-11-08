# cf58897 local preview verification

Source cf588971063874d4e8e400987a4c80eb9d0f30ed. Exact clean-checkout checks:
fmt, cargo-deny licenses/bans, run_all --tables-only, table/figure freshness, and
strict default/CUDA rustdoc all exit0. The checkout remained clean afterward.
Whole-workspace tests and strict Clippy are recorded on00f4409 in the previous
verification directory. git diff00f4409..cf58897 across crates/scripts/tools/
Cargo.toml/Cargo.lock/.github contains only the reviewed plain.rs doc-link change.
No full-test rerun is claimed forcf58897 and no current Linux/GPU pass is claimed.

Source archive: 5,211 tracked files, 56,313,450 bytes, SHA256
`d1d64ec33f3fdc1c9798383c2c5a67cbe37d2b0fb19967cc4cd1bcf8897105ec`.
Each archive file's content was checked against its Git blob ID, and permissions
were checked against tracked modes. The first packaging attempt used Git's0664
archive default and failed the desired0644 check; the archive was regenerated with
`git -c tar.umask=0022 archive`. This is a packaging correction, not result editing.

The 19-file arXiv ZIP (37,482 bytes) was rebuilt from its extracted source directory.
Its10-page PDF text matches the direct preprint build. Rendered pages were reviewed,
and final logs have no overflow, undefined citations/references or missing characters.
The paper source.tex/.bib bytes were checked againstcf58897; main.tex in the ZIP
has the documented ArxivDraft prefix, and main.bbl is generated. No arXiv server
build/upload/submission occurred. The author is checking the account independently.

Limited known-credential-pattern scans found0 matches in5,432 reachable blobs and
in3 compressed blobs (114,648,570 decompressed bytes). Four blobs exceed5MB. This is
not an exhaustive confidentiality or source-license audit. Full public-content
review remains required; no visibility change, tag, release or registry publish
occurred. The scientific FULL-2/GPU/security/full-matrix gates remain separate.

The local bundle is tools/release/_ext/research-preview-bundle-cf58897/. It contains
source tar.gz, named PDF, arXiv source ZIP, inventory, manifest and PREVIEW.md.
Binary deliverables stay outside Git; their hashes are retained here.
