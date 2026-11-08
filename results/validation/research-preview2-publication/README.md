# Research preview 2 publication — 2026-11-08

Published [v1.0.0-preview.2](https://github.com/kenmaro3/ckks-train/releases/tag/v1.0.0-preview.2),
release `438276716`, from fixed source `e93cec434652a8c9f8c48becc6eb3045a3f78a7f`.
This completes the resumed GitHub artifact-publication task. It does not complete
the original research plan, stable v1.0, or arXiv submission; the author retains
the latter responsibility. Preview 1 remains unchanged.

## Scope and checks

- `ci-completed.json` is a fresh API observation of exact-source run `42484163158`:
  all six jobs completed successfully. `ci-test.log.gz` preserves the test-job output.
  Its 80 passing summaries include focused test/release repetitions and workspace
  tests; they must not be reported as 80 independent targets or unique test counts.
  Hardware GPU execution is outside this CI evidence.
- `qualify.py` read back all 6,113 source archive files against Git blobs, modes,
  and the source inventory. Eighteen manuscript input hashes match the retained
  reviewed build. The 19 ZIP members match the retained manifest; PDF hash and
  ten-page metadata match. Separate software and paper notices match Git.
- The unchanged paper was previously compiled and visually reviewed in
  [the FULL-2/R47 record](../paper-full2-r47-20261108/README.md). This session
  verified identity and first-page extracted author/title/abstract, not a new
  compilation or an independent all-page review.
- Archived-table regeneration already succeeded on a clean checkout of this
  exact source in [the candidate record](../research-preview2-e93cec4/README.md).
  Inputs and outputs were unchanged, so that operation was not repeated. No
  experiment, server observation, new training or queue mutation was performed.
- Nine assets were first uploaded as a draft. `upload-verification.json` and
  `draft-release.json` record matching API sizes/digests, source and release notes
  before publication. The source archive is 68,962,151 bytes.
- `verify_public.py` then checked the anonymous public release page/API, the tag's
  exact source commit, and streamed **all nine public assets** without credentials.
  Each downloaded length/SHA256 matches the local bundle and API digest.
  `public-verification.json`, `public-release.json`, and `public-tag.json` retain
  those results and the actual publication/verification timestamps.

The fixed bundle remains at
`/Volumes/Untitled/ckks-train-storage/archives/research-preview2-e93cec4/`.
`manifest.json` describes seven payload assets; `SHA256SUMS` covers those seven
plus the manifest. The checksum file itself is the ninth release asset and was
also verified against its local bytes during public download.

PDF SHA256: `9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c`.
ZIP SHA256: `2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2`.

## Decisions and limitations

The handoff explicitly prioritizes release of the reviewed candidate under the
resumed publication request. Evidence is frozen at FULL-2/R47 for this version;
later queued results do not delay it. FULL-2 accuracy passes, criterion (b) fails,
the common-security condition is unmet, and full-setting baseline CPU time is
unmeasured. The release notes keep GPU training and physical bootstrap limitations.
The author can use the public ZIP/PDF and `paper/arxiv-metadata.md` for submission;
arXiv login, endorsement, license acceptance and server preview remain theirs.

All 50 pre-existing modified Rust source files were hash-recorded in
`wip-source-pins.json`. This release contains only the fixed committed snapshot.
Documentation changes correct the stale FULL-2 status and link to the new public
artifacts, without editing scientific results or generated tables.

The first qualification attempt stopped before manifest creation because macOS
AppleDouble `._*` sidecars were included in a directory enumeration. The corrected
script uses an explicit seven-payload allowlist and excludes only those filesystem
sidecars. Source/PDF/ZIP bytes were unchanged and the complete qualification was
rerun successfully. An initial draft lookup via the tag endpoint returned 404;
the authenticated release list located the unique draft, whose identity and all
assets were verified before publication. Neither was a scientific-test failure.
