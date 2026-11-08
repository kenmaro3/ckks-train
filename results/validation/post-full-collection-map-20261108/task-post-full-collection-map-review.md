# Post-full collection map: bounded independent review

SPEC: CHANGES REQUIRED. QUALITY: CHANGES REQUIRED. One P2 documentation finding; no other blocking finding in the requested section.

Reviewed only `docs/post-full-queue.md` lines 126–156, “Result collection map prepared 2026-11-08” through EOF, against the frozen review-package diff. Reviewed document SHA256: `fdf414059e1e307974f8b3bbac23da37a91b037c9b51015d1dd7277fc5d0b21a`. Frozen diff SHA256: `12ed02a586c1247ba02335e468e9848577cc465a32b4a2d606971952955e16f0`. Both matched the files read. Earlier sections are context, not a new general audit.

## Finding

P2 — line 153 invents a separate one-stream batch measurement. “The sequential baseline and 1/2/4-stream batch variants” does not match the pinned R47 producer. At `bd5cc493db166a282d26b05b1e069f2a0ee629ef`, `crates/hekit-cuda/examples/batch_streams.rs` lines 94–105 select default/max pool thresholds and streams 1/2/4; lines 122–135 use the one-stream branch exclusively for four sequential `matvec_enc` operations and a reference result. Only the `else` branch, lines 136–154, times `matvec_enc_batch` and checks its result against that reference. Consequently each threshold has one sequential x4 baseline and two batch x4 variants, not an additional one-stream batch variant. The acceptance wording could mislabel baseline timing or require a nonexistent coverage row.

Finite amendment: replace the first sentence of acceptance item 3 with:

> For R47, separate default/max pool thresholds, the one-stream sequential x4 baseline, and the two-/four-stream batch x4 variants.

The rest of that item can remain: the queue specifies 30 repetitions for the example and five for optional nsys (`post_full_queue.py` lines 363–369). No experiment or extra measurement is needed to resolve this documentation finding. Parent independently identified the same possible wording issue after this reviewer had inspected the source; the conclusion above is source-derived.

## Other scoped checks

- Producer roots, runner artifacts, queue job names, optimizer selection record, R48 test/ablation filenames, R52 subset filenames and optional nsys branch agree with the queue and pinned writers. The distinction between producer paths and shorter M4 archive names is useful and accurate. No current remote existence claim is made.
- R48 test coverage is correctly separated from harness success: the three MNIST tests and two Iris tests can return early on SKIP, before the paired CPU output. The level7 requirement, complete `final_weights` flattening in MNIST, and error-metric comparison scope in Iris agree with `0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3`. The prose's “5 passed” should be read as aggregate harness coverage, not proof of five executed measurements.
- The eight ablation configurations, matching-comparison restrictions, warmup/timed-iteration counts, summed step/refresh medians, and preservation of the four M4 caveats agree with the pinned ablation writer and queue arguments. Pool checkpoints are not promoted to peak memory. R52 subset completion is not promoted to full ten-pass evidence.
- Execution status, child exit status, SKIP, incomplete coverage, failure and scientific acceptance remain separate. The new section does not promote the preceding synthetic queue fixtures to GPU measurements.
- The retained 1408 observation supports the actual 14:22:14 JST observation time, 865/1000 completed updates, absent final JSON and owner-only queue state. The heading's preparation date does not become a new server observation.
- Retrieval remains a plan requiring an exact registered bounded command, owned evidence paths, fresh destinations, identity/hash checks, retained failures and no automatic retry. It neither performs collection nor authorizes experiments. Large optional profiler storage is constrained to the verified MicroSD destination without silent SSD fallback. Read-time identity is expressly not historical execution attestation.

## Evidence pins and limits

- `results/quadruped/scripts/post_full_queue.py`: SHA256 `1a5f780070e004ec826f21b7dcc187d9e9a65d44e694f2cd5e2f1831589a78f9`.
- `results/quadruped/scripts/run_job.sh`: SHA256 `46d2e83cc3b7ec175813eab9072b49f3eebc7b592125b71c6512315a2146c736`.
- R47 source: commit `bd5cc493db166a282d26b05b1e069f2a0ee629ef`, `crates/hekit-cuda/examples/batch_streams.rs`.
- R48 sources: commit `0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3`, `crates/hekit-bench/tests/mnist_cuda.rs`, `crates/hekit-bench/tests/iris_cuda.rs`, and its ablation binary writer. Read via local Git objects; no checkout or execution.
- R52 producer paths and run arguments: unchanged queue source, whose registered R52 pin is `9305d2e1f4f9999296cd4b4aaf55ea631fbca78e`.
- `docs/m4-report.md`: SHA256 `b270d88647819ad841d6082e3e5df00761e9509221ae7f90ae80dac5ced37ece`.
- Retained observer: `results/quadruped/raw/R-41-FULL2-OBS-20261108-1408/`, README, raw stdout and root outcome; stdout SHA256 `22abd151f5e3292f68eb42b13625afcac598f217dbb6ea45dba668e2521c1a01`.
- Existing retrieval registration context: `results/quadruped/requests.md`. This review does not establish execution of a future collector.

Only local documentation/source/retained-evidence reads and this report write were performed. No SSH, native execution, experiments, tests, source/document edits, commit or push. No assertion about current server state, remote file availability, completed post-full jobs, physical fit, GPU performance or new scientific acceptance follows from this review. Resolution requires only the finite prose correction and review of the resulting document diff/hash.
