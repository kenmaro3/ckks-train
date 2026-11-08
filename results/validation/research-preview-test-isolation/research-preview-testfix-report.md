# Bounded preview integration-test fix report

Status: source frozen for independent review; no commits, merges or publication. Detached base `90851d261b36b034d3b660ec4c95dd330eadc616`. Worktree `/Users/marconi/Documents/research/hekit/tools/release/_ext/research-preview-testfix`.

## Root cause and correction

The OnceLock fixture contains one Arc<PlanningBudget>. request() clones it for four tests. Canonical asserted <=4 wrapper calls against cumulative global counts even though late-finalization adds two; inventory's before/after interval could include other tests' geometry work. Production budget synchronization is correct and unchanged.

The only product change is `crates/ckks-train-bench/tests/study_admission.rs`: import Mutex; add and initialize fixture-local budget_test_lock; hold it through inventory, canonical, case projection and late finalization; snapshot canonical usage before work and subtract those baseline counters in its existing <=4 wrappers / <=8 updates / ==0 handoffs assertion. Inventory's existing deltas are now isolated. Exactly 14 additions and 2 deletions. All original tests and assertions remain, with numeric thresholds unchanged. Immutable parameters, proof, bootstrap metadata and canonical Iris OnceLock remain shared. Aggregate cap remains 148 geometries / 148 initial states / 8 updates / 0 handoffs / 8 wrappers. Independent bootstrap-binding fixture and other targets retain harness parallelism.

Pre-edit reason and rejected independent-budget alternative: `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/decision.md`. Original bytes: `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/study_admission.before.rs`. Existing RED is preserved in the three paths specified by the brief, including the original workspace exit101; no new fabricated failure history or redundant failing test was added.

## Verification on frozen source

Each run directory records exact argv, cwd, selected environment, source hash, launch PIDs, separate stdout/stderr, exit and elapsed time. All calls use nice10; Cargo calls are offline/locked with CARGO_BUILD_JOBS=2, CARGO_INCREMENTAL=0, RAYON_NUM_THREADS=2 and only the approved isolated preview target cache. RUST_TEST_THREADS was explicitly removed for genuine default harness parallelism; the second target run passes --test-threads=2 explicitly.

- `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000729.476146Z-format`: `nice -n 10 rustfmt --edition 2024 --check crates/ckks-train-bench/tests/study_admission.rs`, exit0.
- `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000729.756050Z-default`: `nice -n 10 cargo test --offline --locked -p ckks-train-bench --test study_admission -- --nocapture`, exit0; 5 passed, 0 failed/ignored/filtered; test duration0.32s (53.67s compile).
- `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000830.109217Z-parallel2`: `nice -n 10 cargo test --offline --locked -p ckks-train-bench --test study_admission -- --test-threads=2 --nocapture`, exit0; 5 passed, 0 failed/ignored/filtered; test duration0.31s.
- Both successful runs reach cumulative usage 147 geometries / 147 initial states / 6 updates / 0 handoffs / 5 wrappers, within unchanged aggregate cap. Canonical usage in these schedules is 1/1/2/0/3.
- `git diff --check` exit0 and changed-file scope/hash rechecked after tests; `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/final-scope-check.json`.

## Review artifacts and freeze

- Exact minimal patch: `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/patch.diff`.
- Review package with U10 context: `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/review-package.diff`.
- Frozen source: `/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-testfix-evidence/20261108T000644.837683Z-source/study_admission.frozen.rs`.
- Source SHA256: `94191ae23ca984dcc801f2efdf333da1b8ce8ac93eccdd6f4b1a8927dff6d8c0`.
- Patch SHA256: `241033f6a611505e8b76d317941e98d614dd9386d1e598702f7f960459751be7`.

## Limitations

This is local macOS test-isolation evidence, not native Linux or scientific/HE validation. No heavy HE smokes, server/network activity, production runtime changes, main-source edits, commits, pushes or cache deletion. The root performs independent review and final candidate qualification. The full original research goal remains open; this only addresses a candidate qualification defect.
