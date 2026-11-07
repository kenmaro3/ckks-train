# R-REB-29: corrected context sidecar deployment and readback

Operational evidence only; context/capture/scientific execution has not occurred.

The reviewed correction replaces the erroneous0644 expectation with exact0664 retained by both R27 and R28, and binds the existing link exemption to the exact exporter path. It changes eight source lines and fixed local custody selection. No remote chmod or existing input repair was performed. Original failed R28 receipts and consumed sentinel remain intact; their historical conversion error is explicitly corrected in R28's diagnostic review. The new recovery1 custody was used once per phase after separate concrete action releases.

Deployment returned0 at2026-11-07 12:54:43 JST with `deployed_verified_no_target_startup`. Exactly one new309421-byte sidecar was created0600 in the existing v2 directory; SHA2566137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e. The three existing input identities were preserved.

Independent readback returned0 at12:55:55 JST with `readback_complete`. Its four full file records exactly match deployment's preserved-input plus sidecar records, including uid/mode/device/inode/bytes/hash/path. Directories retain uid1000/mode0700/device66311/inodes36309255 and36309263, matching the existing deployment directory pins. All three future context evidence paths were absent at that observation. Both operations retain complete streams, empty stderr and no cutoff or retry.

These facts establish deployment/readback only. No target context startup, package import/hash campaign, seed capture, training, GPU action or job/queue change occurred. Invoke/collect remain unreleased. The full timed-job gate remains: FULL-2 was observed live at12:49:42 JST238/1000, and that earlier observation is not refreshed by this placement. Fresh checks and separate execution/content reviews are still required before later actions. Current evidence-absence is time-specific, not a permanent promise.

All source, manifest, candidate, report, delta, test log, release and raw custody files here are exact ledger copies. `tests-green.log` is retained local evidence, not a reviewer rerun or Linux/GPU benchmark. `SHA256SUMS` covers all other files, including this README. Original operation JSON is not edited; no scientific result JSON is synthesized.
