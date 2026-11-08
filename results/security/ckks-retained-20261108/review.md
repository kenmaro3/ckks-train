# Raw security-output preservation — independent bounded review

SPEC: PASS. QUALITY: PASS. No findings within scope.

Reviewed the complete frozen documentation diff and the new `results/security/ckks-retained-20261108/` archive, against the pinned early inventory and its original local paths. This is preservation/claim review, not mathematical security validation, schema adoption, historical execution attestation or current-source applicability approval. No estimator, HE, tests, build, regeneration, network or source-WIP operations were run. Only this report was written.

## Verified preservation

- The early inventory is exactly 15,520 bytes, SHA256 `42b4574ac7a0d7f902a01d86e36bb2f6efd19dc82a613a75bbca10d6fe8c8eff`. All 33 row identities, original paths, sizes, hashes and classifications agree with the new manifest. Independently read/hash-checked all 33 originals; all match their pins.
- All 29 selected public numeric logs plus the separately classified partial scan are byte-identical to their original files. The three local-only traceback files have no public archive path and remain present with unchanged bytes. The historical progress copy also exactly matches its 6,797-byte original and recorded SHA256.
- Independently recomputed all 40 SHA256SUMS entries and compared its path set to every other archive file: exact coverage, no omitted or unexpected file. The archive contains 41 files including SHA256SUMS.
- Read representative raw output, the complete partial scan and historical progress, and all 149 normalized nonblank numeric-output line forms across the 30 copied numeric logs. Headers, numerical fields/summaries, grid rows and platform timing/RSS trailers were observed; no external implementation snippets or traceback frames were found in those copied files. A bounded source/credential-pattern scan also found none. This is content inspection, not a legal guarantee or exhaustive secret audit; runner/estimator/Sage implementations were not copied.
- Privately checked the three local-only files for tracebacks and source frames, without reproducing their excerpts here. Each corresponding END marker in historical progress says rc=0 despite its failure trace. The documentation's warning against interpreting these markers as independent success attestation is supported.

## Claims and regeneration records

`docs/security.md:333–341` is exactly the supplied frozen insertion; no existing numerical criteria or summaries are changed. Archive README paragraphs 1–6 consistently identify read-time preservation rather than new estimates, complete historical coverage, live-gate authority or reconstructed provenance. They keep unrecovered reviewer scratch output unavailable and distinguish local Darwin estimator timing from Linux HE-training measurements and ReBoot's different security level.

The partial wide scan stops after the early grid rows and has no completed minimum; its limitations directory, manifest classification and README all prohibit positive completion inference. Historical interruption/restart events remain intact.

Read all six retained table-check stdout/stderr files and verification.json. The recorded commands are run_all.sh --tables-only, tables.py --check and figs.py --check, each exit 0. Regeneration stdout records seven tables; figure stdout records three checked artifacts. All 18 recorded generated-file hashes independently match current bytes, and all 18 files are byte-identical to their committed HEAD versions. These are existing command records and static comparisons; no command was rerun and no scientific measurement was inferred.

## Input pins

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| Ledger review diff | 901 | `ee79d0791a95a0e80c548ad2ffad363a96080667ef890657c8f6b67d2feba16e` |
| Archive README.md | 3078 | `ac2ac79e157dd3065b2632de3f91ecd64482bdc4c13a4e2e74f8c8ff0c339568` |
| preservation-manifest.json | 22128 | `7c2269a8ebca43bceee08e53739d6870501af90de471203bbf7a882a1d12956e` |
| SHA256SUMS | 3846 | `3b46bec11d839cd33f4ddce9866f52f97e81c773862d79b205dc7a247c964f5c` |
| table-check/verification.json | 2567 | `f839a5e0a4292d283c043c9c45f812a27e5ef33b669bb451f1664cee94970a8f` |
