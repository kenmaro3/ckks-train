# CI cargo-deny installation failure — 2026-11-08

The `deny` job of source6910b14 (run42477151156, job128720918474)
failed while building the third-party action's Docker image. Its release-download
stream could not be decompressed (`gzip: invalid magic`); the runner automatically
attempted the Docker build three times. The audit command itself never started.
This job gives no license or forbidden-dependency verdict.

The following source40211e5 run42477797335, job128722911533, used the same resolved
cargo-deny-action SHA3c6349835b2b7b196a839186cb8b78e02f7b5f25 and cargo-deny0.20.2.
Its first Docker build also encountered the download problem; the runner's next
attempt succeeded, and the actual audit printed `bans ok, licenses ok`.
The workflow, deny.toml, Cargo.lock and workspace Cargo.toml are Git-object-identical
across b7c51dd,6910b14 and40211e5. Their exact IDs are in the diagnosis record.

This supports an intermittent installation/download failure. The response body and
HTTP status were not retained by the upstream curl pipeline, so the exact upstream
service/transport cause is unknown. No manual workflow rerun, configuration edit,
policy relaxation or ignored failure was used. Retain this failure as such; it does
not become a successful6910b14 audit because another source's job passed.

At the12:02 UTC observation, source40211e5 had five successful jobs and its test
job was still in progress; its overall CI was not yet certified. Exact earlier
sources2f7b8bd andb7c51dd had completed successfully in all six jobs. These records
do not validate active uncommitted Task54 source, GPU training, or a stable release.

The raw job logs came from the completed-job API because `gh run view --log-failed`
refused access while the overall run remained in progress. API commands and observation
times are retained. Original bytes were copied unchanged; hashes are inSHA256SUMS.
The diagnosis is operational metadata, not a scientific measurement. A bounded
known-credential-pattern scan of the archived bytes found no match; this is not a
complete secret audit. No credentials or environment values were requested.

The later terminal observation of run42477151156 is also retained: overall failure,
with only deny failed and the other five jobs successful. The original watcher
ended with exit1 at12:08:08 UTC. This does not certify40211e5's still-separate run.

Independent [operational review](task-ops-2105-ci-install-review.md): SPEC / QUALITY PASS for the original records and bounded interpretation. It does not certify the unfinished current CI or scientific results.

## Later completion of source40211e5 CI

At2026-11-08T12:27:09 UTC the direct GitHub run observation confirms run42477797335 completed successfully: all six jobs passed on exact40211e564c2ef6e2a94f19527710ef99ec54196d. The existing watcher ended with exit0 at12:22:28 UTC; its result is retained separately. The earlier pending snapshots and operational review are unchanged and precede this terminal observation. This does not retroactively pass source6910b14 or validate uncommitted Task54 source, GPU training, or a stable release. No manual rerun was performed.

The later terminal record has its own [independent supplement review](task-ops-40211-ci-terminal-review.md), SPEC / QUALITY PASS for this exact-source CI observation.
