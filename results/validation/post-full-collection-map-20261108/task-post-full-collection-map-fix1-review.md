# Post-full collection map fix1: scoped re-review

SPEC: PASS. QUALITY: PASS. Prior P2: ADDRESSED. No fix-induced finding in this bounded re-review.

Reviewed `task-post-full-collection-map-fix1.diff`, `task-post-full-collection-map-fix1-report.json`, and the corrected acceptance section of `docs/post-full-queue.md`. Independently computed final document SHA256: `73e0f902c2d7d45d342fdc8641f9508d32f4fec6196995a8f14abdb6310c5bd7`.

Acceptance item 3 now names the one-stream sequential x4 baseline and the two-/four-stream batch x4 variants separately for default/max pool thresholds. This exactly implements the finite amendment from `task-post-full-collection-map-review.md` and agrees with the previously inspected `bd5cc493db166a282d26b05b1e069f2a0ee629ef:crates/hekit-cuda/examples/batch_streams.rs` branches at lines 94–154. It no longer implies a separately measured one-stream batch variant.

The corrected sentence occurs once. Reversing only that replacement yields SHA256 `fdf414059e1e307974f8b3bbac23da37a91b037c9b51015d1dd7277fc5d0b21a`, the exact prior reviewed document. Thus no other document bytes changed in this fix. Repetition counts, profiling distinction, historical comparison, adjacent coverage qualifications and retrieval constraints are preserved. Parent reports `git diff --check` PASS; no checks or tests were rerun here beyond reading and hashing the prose artifacts.

Scope remains the prior P2 and breakage caused by its correction. This is documentation acceptance only, carrying forward the original review's evidence limits. No general audit, SSH, experiment, test rerun, source edit, document edit, commit or push was performed. Only this review report was written; no new measurement or current remote-state claim is established.
