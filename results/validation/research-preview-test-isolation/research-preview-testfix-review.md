# Independent bounded test-isolation review

**SPEC: PASS. QUALITY: PASS. No blocking or nonblocking defects found in this frozen one-file change.** This verdict covers the integration-test isolation fix only; it is not whole-preview qualification or the main supervisor review.

Reviewed detached base `90851d261b36b034d3b660ec4c95dd330eadc616` in `/Users/marconi/Documents/research/hekit/tools/release/_ext/research-preview-testfix`. Current Git status shows exactly one modified file, `crates/ckks-train-bench/tests/study_admission.rs`; actual diff is 14 additions and 2 deletions. No production change, test deletion/ignore, raised threshold, or global single-thread setting is present.

## Source findings and specification checks

- **Root cause confirmed.** Test lines 32–66 retain one static fixture and clone its `Arc<PlanningBudget>` into requests. Production `src/study_admission.rs:80–138` defines cumulative, mutex-protected counters; charging is monotonic and checks the unchanged cap before incrementing. `plan_diagonal` charges each wrapper entry at production line 2179. Canonical makes three wrapper calls (test lines 200, 266, 277), while late finalization makes two (422, 428). The old canonical assertion wrongly compared its bound of four with their shared cumulative total of five. Inventory's old before/after interval could also include concurrent geometry work.
- **All shared-budget users covered.** Inventory, canonical, case projection, and late finalization acquire the same fixture mutex at test lines 105, 190, 296, and 414, holding named guards to function exit. Each acquisition precedes its request/counter work. Projection is protected despite currently rejecting before charging. Helpers `main_case` and `parameters` only read fixture state. Bootstrap binding uses its own budget (328–366), so it does not require this lock. There is no recursive acquisition or escaping background work in these test paths.
- **Original bounds applied to their intended scope.** Canonical snapshots usage under the guard (191) and checks deltas against the original `wrapper_calls <= 4`, `updates <= 8`, and `handoffs == 0` (284–289). Inventory preserves exact `144/144` deltas (176–183), now isolated. Monotonic counters make subtraction safe. The fixture still enforces the aggregate `148 geometries / 148 initial states / 8 updates / 0 handoffs / 8 wrappers` (53–59); no reset or separate per-test replacement budget removes that total cap. Existing negative-path and receipt assertions remain unchanged.
- **Minimal implementation.** Expensive parameters, proof, metadata, and Iris `OnceLock` remain shared. The test-local mutex serializes only the four fixture consumers. The independent bootstrap test and Rust harness remain parallel-capable. Lock poisoning propagates a genuine test failure rather than concealing it. The pre-edit `decision.md` explains the choice and rejects independent budgets because they would lose the aggregate constraint.

## Evidence independently read

Paths below are relative to this report's directory. For all six runs I read command metadata, exit results, and relevant stdout/stderr; full target logs were read. Original workspace logs were inspected at the failing target and termination. No tests were rerun.

| Evidence directory | Observed result |
| --- | --- |
| `research-preview-evidence/20261107T235505.137343Z-tests` | Original workspace exit 101; this target 4 passed / 1 failed at old line 279, cumulative wrapper assertion. Original failure remains preserved. |
| `research-preview-evidence/20261108T000402.372853Z-study-parallel` | Original explicit 2-thread target exit 101; same canonical failure; final cumulative counters `147/147/6/0/5`. |
| `research-preview-evidence/20261108T000417.211472Z-study-isolated` | Original canonical-only exit 0, 1 passed / 4 filtered; counters `1/1/2/0/3`. |
| `research-preview-testfix-evidence/20261108T000729.476146Z-format` | Frozen-source `rustfmt --edition 2024 --check` exit 0, empty stdout/stderr. |
| `research-preview-testfix-evidence/20261108T000729.756050Z-default` | Fixed default harness exit 0; 5 passed / 0 failed / 0 ignored / 0 filtered, 0.32 s test time. |
| `research-preview-testfix-evidence/20261108T000830.109217Z-parallel2` | Fixed explicit 2-thread harness exit 0; 5 passed / 0 failed / 0 ignored / 0 filtered, 0.31 s test time. |

The fixed Cargo commands use `nice -n 10`, `--offline --locked`, `CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=0`, `RAYON_NUM_THREADS=2`, and the approved isolated target directory. Recorded `RUST_TEST_THREADS` is absent; `run_checks.py` explicitly removes it. Default command has no harness thread override; the second supplies `--test-threads=2`. Both logs end at aggregate `147/147/6/0/5`, within the unchanged cap, and canonical logs `1/1/2/0/3` in those observed schedules. Canonical's diagnostic remains cumulative; the assertion correctly uses deltas for other orders as well.

## Frozen identity and limits

Independently recomputed current file SHA256: `94191ae23ca984dcc801f2efdf333da1b8ce8ac93eccdd6f4b1a8927dff6d8c0`.

Byte comparisons confirm `study_admission.before.rs` equals Git HEAD, `study_admission.frozen.rs` equals current source, and recorded `patch.diff` equals the actual Git diff. Patch SHA256: `241033f6a611505e8b76d317941e98d614dd9386d1e598702f7f960459751be7`. These artifacts are under `research-preview-testfix-evidence/20261108T000644.837683Z-source`; its final scope check also records `git diff --check` exit 0.

Evidence is local macOS integration-test isolation and formatting only. This target reports `HE=0`; neither the patch nor this review establishes native Linux behavior, HE/campaign/scientific performance, or full candidate qualification. No commits, publication, network/server actions, source edits, or additional test runs were performed by this reviewer. Only this review report was created.
