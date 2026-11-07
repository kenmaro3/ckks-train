# Task54 P4 CPU bridge source release

Root decision 2026-11-07; base 07f7a989ea46d3398835cfb0535f1aa96fc0b78c. Source implementation and compile/static validation only. Actual HE/F64/planning runtime remains unexecuted until a subsequent frozen-source runtime release. This is progress toward the full Task54 runner, not a substituted endpoint.

Read task-54-p3-parent-brief.md first for the original Task54 requirements; then task-54-p4-final-consumer-map.md (a6fb68014e4b6eb099136ed2c189551e812fc494d282b03fec2e301a9ee5a110) in full. It supersedes conflicting provisional choices in task-54-p4-api-map.md. Read that earlier map for actual APIs and its complete coverage table, but use ONLY final Signal recipes/NI tolerance from the final map and existing study_plain/p4_fixtures.rs. Relevant binding design: docs/study-diagonal-admission-design.md sections 2/4/6/7/12 and docs/study-training-design.md sections 2/3/5/6/7/8. Do not read whole research plans or unrelated ledgers.

## Rulings before implementation

1. Adopt final map's canonical validate_plan_binding then describe_candidate geometry path. Reserve one geometry/initial call in the original bound request before construction; do not mutate limits after hashing. Use plan states, NEVER candidate descriptive initial states. Alternative retaining all geometries in P3 would extend receipt allocations and is unnecessary. Cost if wrong: revise bridge preparation, not cryptographic math.
2. Compare every exposed initial/pre/post state entry, full cardinalities, all seven inclusive measured counters, actual configured factors/reference/branches, refresh/bootstrap decisions and progress. Keep actual runtime trace diagnostics and exact available hidden a1/AA a1_boot subset; full typed transcript equality remains unclaimed. Alternative full trace projection adds storage/accounting without being required for the first bridge.
3. Adopt private distinct synthetic plan child over existing P2 and P3 compose/codecs, and closed Signal oracle API over the existing actual baseline Engine/arrays. No public canonical constructor, no arbitrary digest capability, no copied recipe/math. Canonical D7/real pre-key acceptance remains pending; positive synthetic shared-factory spy proves only the local library consuming order.
4. Use one orchestrated ignored lib unit test `study_diagonal::p4_tests::p4_cpu_bridge_contract` for eventual runtime matrix. It owns exactly two parameter/context/key-generator setups once; all cases borrow those contexts and fork training evaluators. No OnceLock/global fixture or duplicate integration binary. Source-phase may compile this test but must never invoke it. Future runtime guard/log counters must be embedded before expensive constructors/operations.
5. Use caller-owned CPU evaluation context, client-owned secret/materialization state and immutable private encrypted packets. Assign a fresh private context identity at real setup, carried in client/session/packets, so same parameter bytes with a different key owner cannot be accepted. A private identity is association evidence, not cryptographic authentication. Server gets only narrow refresh service handle for I; NI gets none. In-process separation is explicitly not process isolation.
6. Poison before the first fallible setter/train operation, clear only after all postchecks. Prework mismatch leaves session usable and all counters/config/state unchanged. Midwork failure keeps successful update/sample counters unchanged, retains actual partial operation delta and rejects retries. Thin test-only real refresh/BootstrapOps forwarding guards fail on second operation after one real operation; no fake HeOps or arithmetic replacement.
7. NN change only trainer.rs: borrowed read-only OptimizerView, default unsupported trait diagnostic_view, existing Sgd/Nag implementations and checked Trainer forwarder. No optimizer recurrence/formula changes. Unsupported does not mean SGD. NI accessors already suffice.

## Source ownership

Single implementer owns exactly:
- NEW crates/ckks-train-bench/src/study_client.rs and study_diagonal.rs;
- NEW crates/ckks-train-bench/src/study_diagonal/p4_tests.rs;
- NEW crates/ckks-train-bench/src/study_admission/p4_fixtures.rs plus only cfg(test) child declaration in study_admission.rs;
- crates/ckks-train-bench/src/study_plain/p4_fixtures.rs: narrow closed SignalFamily/SignalOracle API and shared baseline assembly, preserving existing v2 sensitivity checks and values;
- crates/ckks-train-nn/src/trainer.rs: only specified borrowed diagnostic API, focused source tests if needed;
- crates/ckks-train-bench/src/lib.rs: module declarations only;
- ledger task-54-p4-cpu-source-report.md and task-54-p4-cpu-source-validation/ evidence.

No other source edit, Cargo dependency, math/planner schema change, mutable public packet, generic Plan framework or successful production dispatcher. run_owned remains MissingExecutionDependency. If a necessary interface is absent, report the exact need to root before expanding ownership. Root owns public docs/progress/commit/push. Do not commit or stage yourself.

## Concrete source acceptance

Implement real library client/session path over existing Trainer/NonInteractiveTrainer train_step_valid; not a test-only simulation. Canonical input path reuses materialized arrays/order/targets and bound plan coefficients, one stage only; explicit capability error for block stages. Canonical binding checks precede private real key factory; same prepare_then_make invocation site serves reachable nonallocating synthetic spy. No fake ready/start lease. Canonical metadata/gates remain separate production dependency.

Use exact final-map recipe and coverage: 12 smoke successes (6 families I/NI), four 4-step scheduled sessions (I/WN/AU/AA), two 2-step reference-first sessions (I/AU), two poison sessions (I/AU). 20 sessions, 34 trainer attempts, 32 successes, 2 failures, at most85 actual boots. Paired oracle once per successful session =32 actual F64 Engine updates, no mutant sweep. I all-entry W/b absolute tolerance1e-5; NI4e-4, fixed before HE; finite checks mandatory, frozen F64 bias exactlyzero. No tolerance loosening. Poly smoke .04 with reference.08; do not use older provisional recipes/rates.

Future runtime shape/planner limits: at most28 geometry and28 initial-state calls (20 session plans plus up to8 one-update binding fixtures), <=42 update plans,0 handoffs; prefer reusing one already-created session plan for spy/negative mutations. Exactly2 params/2 keys/1 proof/1 bootstrap descriptor/1 bootstrapper; <=22 Encoder constructions. Positive factory spy1 call and<=16 isolated negatives0calls, no actual constructors for spy. All counters must count actual helper invocations; stop at cap, never raise silently. No canonical dataset construction is included in mini suite. Include proposed exact inventory in report so root can inspect before runtime.

For each success compare all logical W/b to actual SignalOracle snapshot, all metadata entries/cardinalities and seven counts to existing plan, actual configured current/reference coefficients/first versus later velocity, expected refresh/boots/lookahead and successful index/sample totals. Diagnostics/probes use fork; assert training counters unchanged. Packet negatives exercise identity, index/replay/exhaustion, partition, labels/target shape, ct basis/level/scale, config and nested initial-state mismatches before any trainer work; reuse fixtures without extra updates. Wrong owner with same params must reject before HE. Preserve ordered global rows across micros. No lost or duplicated padded-channel learning.

## Tests first, source-phase commands

Write assertions/test skeleton first; capture a focused compile RED for the genuinely missing bridge API. This is compile evidence only; runtime RED/GREEN is not claimed. Then implement and self-review. May iteratively run ONLY:
- CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract --no-run
- same environment cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
- rustfmt --edition 2024 on owned files only (set skip_children=true for large module roots, avoid unowned child formatting).

One Cargo owner, existing target only, no release/workspace/allfeatures/old test runtime. Before each Cargo invocation require free disk>=805306368 bytes. Log exact command, start/end/exit/handle, source before/after SHA256, raw output under assigned evidence directory. Poll the same handle to terminal. No invoking any HE/parameter/Engine/planner constructors at source phase, no datasets, GPU, SSH, network, external writes, subagents, cache deletion or changes to unowned .vscode/ and results/mnist_blocks_*_plain_b48.json. Cargo compile-time build scripts are ordinary build operations, not scientific runtime.

Report DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT with actual implemented interface/ownership, tests and compile RED/no-run GREEN, commands/exit/raw paths/source hashes, full future runtime inventory and remaining limitations. Do not claim source review, runtime parity, full P4/P3/D7/Task54/M4/M5 completion. No standalone runtime release from implementer.
