# Task 55: fixed-source package verification, 2026-11-08

Five offline, locked Cargo package verification builds passed on macOS for
source `2dc049a796949b685a5c23f8b05440ee2ce474aa`: core, ckks, nn,
cuda, and cuda with the cuda feature. Independent SPEC and QUALITY review passed.
The four unique archives contain 17, 39, 60 and 41 members respectively.
The feature-enabled CUDA archive is byte-identical to the default archive.

[Author report](report.md) and [independent review](review.md) retain the exact
commands, dependencies, licenses, package contents and qualification limits.
`freeze.json` is the original pre-review snapshot; its pending-review field is
historical. The subsequent review is retained here without rewriting that snapshot.

## Reporting clarification adopted after review

The author report's phrase "own peak was about519MB allocated" is superseded by:
**The observed working area immediately before deletion was 519,069,696 allocated
bytes, below 1 GiB; transient peak allocation was not continuously measured.**
The task's own build cache was removed after preserving evidence (269,853,342
logical bytes). The frozen report is preserved byte-for-byte for auditability.

## Evidence layout and limits

The original evidence manifest covers 71 files. This directory retains the 65
metadata/log files unchanged; five crate archives and the source tarball remain
local at `tools/release/_ext/package-current-2dc049a/`. Their identities are listed
in `retained-local-artifacts.json`. The complete original source inventory remains
in `manifests/source.sha256.jsonl`. `SHA256SUMS` covers files actually retained here.

These checks compile the extracted packages using Rust/Cargo 1.85.1 on
`aarch64-apple-darwin`. They do not run packaged tests, native CUDA kernels,
training or scientific validation. They do not qualify active Task 54 changes,
publish crates, or establish stable-v1.0 completion. No registry upload occurred.
