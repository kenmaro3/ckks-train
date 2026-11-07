# Task54 D7 inert resource source codec

Source BASE: `17c18c3ecc4b1c87053e0d6397bfebca9a8316bf`. Final source review: SPEC PASS; QUALITY PASS after FIX1. This milestone implements representation parsing and supplied-record consistency, not a live resource gate.

Six files implement strict closed JSON sources, exact original-byte identity, bounded structural preflight, explicit nullable/work variants and cross-record comparisons, plus the borrowed fixed bootstrap transport. Production rejects synthetic domains. Legacy Resource Full/Quick bytes remain unchanged. The bootstrap parser/encoder uses borrowed or fixed storage; its 704-byte wire size is not a physical B0 bound.

Final verification: 13 focused pure tests passed, 0 failed/ignored, 269 filtered; 109 fixed inputs total 473,973 bytes. They cover 26 JSON and two bootstrap positives, 67 JSON and nine binary negatives, and small path/lexeme/count/identity/legacy controls. Separate private-context witness illustrations are not counted as parser or native tests. All-feature/all-target strict Clippy and formatting passed after FIX1; CUDA/native/HE operations were not executed.

Initial review passed scoped semantics but required a clean strict lint check. Root's added Clippy run failed on the large inline enum and two needless test borrows. FIX1 kept the enum inline with a local documented lint annotation and removed those two borrows; its strict Clippy, focused tests and formatting each ran once and passed. [Scoped re-review](final/task-54-d7-resource-source-fix1-review.md) closes Q2. [Initial review](final/task-54-d7-resource-source-review.md) records low Q1: a cfg(test)-only bootstrap comparison lacks explicit baseline-mode equality. Q1 is deferred to final integration review; production has no Oracle mode, and this is not a production synthetic acceptance claim.

## Evidence layout

- `initial/`: original author manifest and exact pinned artifacts, including original RED/failure/GREEN logs and pre-FIX1 report/diff. One administrative draft review brief was reconstructed by inversing root's release edit; it matches its original pin exactly but is not an originally captured snapshot.
- `before-fix1/`: six actual source snapshots and original report/diff/pins/review captured before FIX1. Earlier per-run source snapshots were not captured and are not reconstructed here.
- `final/`: append-only final report, correction brief/diffs/pins/logs, original semantic review and scoped re-review. Final complete source diff is against the original BASE; actual final sources are tracked in the repository.
- `root/`: released review instructions, exact administrative draft/release transition and original strict-Clippy terminal metadata.

The original frozen corpus stays in the adjacent `task54-d7-resource-oracle` archive; it is not regenerated or duplicated here. Artifact manifests retain their original worktree/ledger paths and are historical identities, not portable execution scripts. SHA256SUMS covers every archived file except itself.

A complete physical model, real host/cgroup/allocator support, origin/currentness, cooperative lock/helper lifetime, same retained bytes into actual typed loaders, actual model/receipt applicability, live gates and the full experiment matrix remain required. No memory fit, RQ result, Task54/M4/M5 completion or release follows from this codec PASS.
