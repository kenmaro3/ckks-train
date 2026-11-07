# Task54 P4 pure sensitivity preflight

2026-11-07. Prepared by root while P3 fix1 owns source/Cargo. **This file does not transfer ownership:** dispatch may occur only after P3 returns source/Cargo. It releases no HE, keys, server work, production admission, scientific result or campaign. The full P4 bridge and original P3 pre-key acceptance remain required afterward.

Read first: `task-54-p4-sensitivity-disposition.md` SHA256 `4c221f802d0cf62b973d74f97fbcaadde4bcc362a7f7c3e7cc30ad3a283e33d8`. It specifies the complete finite control matrix, two fixed fixture candidates, numerical margins and range gate. Read `task-54-p4-api-map.md` SHA256 `47e84a312bc2a7f7eaf2eb3d5af75f8535daeea5be66a43284fdf7426ee8a321` sections5–6 for the six exact shapes and rates. The sensitivity disposition supersedes the map's provisional NI5e-3 threshold; neither document supplies measured success.

## Root decision and reason

The existing tiny initial values could permit unchanged or incorrectly scaled weights to pass a loose absolute tolerance. Use the actual D6b F64 engine to establish test sensitivity before allocating HE keys. This is a bounded test-fixture preparation step, not another numerical trainer. Reject a copied F64 recurrence or hand-derived claimed distances. Reject widening tolerance after seeing HE errors. Preserve original scientific rates, datasets, presets and all registered rows.

Add a cfg(test) child module of study_plain. This child can use private Engine, PlainLayer::from_arrays, StageSpec and the existing forward function without exposing a public synthetic materializer. Alternative public PlainState construction requires canonical materialization and is unsuitable for these deliberately tiny fixtures. Production D6b APIs, arithmetic and visibility remain unchanged.

## Exact ownership

After explicit dispatch, sole implementation/Cargo writer owns:

- `crates/ckks-train-bench/src/study_plain.rs`: only a cfg(test) crate-private child-module declaration.
- New `crates/ckks-train-bench/src/study_plain/p4_fixtures.rs`: fixed synthetic fixture owner, actual-engine calls, read-only snapshots and bounded sensitivity tests.
- New ledger `task-54-p4-pure-preflight-validation/` and `task-54-p4-pure-preflight-report.md`.

No P3/NN/client/session/lib exports, production source changes beyond the declaration, dependencies, Git, network, scientific JSON or subagents. Preserve all prior artifacts and unowned files. This test module is intended for later crate-private P4 tests; it cannot construct DiagonalPlan/MaterializedProfile or execution permits.

## Tests and observations

Write the new discriminators first. One expected compile-only RED can establish the missing fixture interface. Then implement the actual-engine helper and run exactly one selected pure characterization/test invocation. Use one test named `p4_pure_sensitivity_gate` to share both predetermined candidates and all snapshots. Evaluate every applicable control for both candidates and emit its evidence before the aggregate final assertion, so a failed candidate does not hide other already authorized rows. A failure of the original candidate is a recorded finding, not an implementation retry trigger. The fixed signal candidate must pass the aggregate required gates to permit later HE acceptance. If it fails, stop and return all measurements; do not search, tighten epsilon, alter rates or rerun.

Use baseline and mutants through the same Engine::train. Preserve the exact input/label ordering across unequal micros; the F64 oracle concatenates logical rows once. Apply delta and mean separately. A no-op control snapshots initial weights, not a second implementation. Reset-velocity and SGD controls mutate only existing fixture state/configuration. Algebraically equivalent or inapplicable controls are explicitly labeled, not counted as discriminated failures.

Take snapshots after each successful actual train call. Read current hidden values using existing `forward`, before and after each step on the actual inputs. This does not replace gradients. Record both activations and preactivations so the meaning of the hidden-range prerequisite is inspectable; the disposition's hidden magnitude gate applies to hidden activations <=3/4. Assert finite W/b/velocity and exact frozen F64 biases.

Freeze candidate caps I1e-5 and NI5e-4 for this preflight. Compute both backend-normalized witness distances over trainable W and active biases. For each applicable control require distance>=8 at its first affected update; also require a separate first-update no-op W witness in each trainable layer. Use actual coordinate/value evidence. Do not use frozen biases as no-op witnesses. In scheduled controls, specify the first affected update explicitly from the control change rather than selecting the most favorable later endpoint. Full reference identity is tested by P3/runtime getters, not by this unscaled F64 oracle.

Emit labeled text lines (or an ordinary test-log encoding clearly marked NONSCIENTIFIC_TEST_EVIDENCE) carrying fixture/version, shape, classes, exact coefficient bits, control/equivalence status, update index, tensor/layer/coordinate, baseline/control values, maximum distance, hidden ranges and actual call count. Preserve complete stdout/stderr and terminal status. No manually entered scientific result JSON or reported training accuracy/timing is permitted.

## Finite work and terminal handling

At most the original and one fixed signal candidate described in the disposition. At most100 Engine::train calls per candidate,200 total, <=2 layers, input2/output<=16, <=12 rows/update. No CKKS parameters, proof, metadata, Encoder, keys, evaluator, ciphertext or bootstrap constructors; no HE or data materialization. Count actual calls before invoking the engine and preserve count on failure. Equivalent/no-op controls reduce work; unused allowance is not for new cases.

Use current target only, CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8, diskfree>=805306368B. Compile/lint fixes may rerun only the affected compile/lint command. Freeze source after lint/format before the single numerical invocation; pin both files before and after. Commands:

```
cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate --no-run
cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings
rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_plain.rs crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate -- --test-threads=1 --nocapture
```

The Clippy `--tests` compiles bench test targets but executes none; do not run them. No workspace/release/old P1/P2/P3 runtime tests. If another writer changes source/target ownership, wait for root; do not overlap Cargo. On unexpected runtime failure, retain original evidence and report, without automatic rerun.

Return exact two-file diff/snapshots, command logs/pins/terminal statuses, actual counts, per-candidate gate results and every failed/equivalent/inapplicable control. Separate source/test implementation correctness from whether the chosen numeric fixture passed sensitivity/range. No HE release is implied even if the pure gate passes: P3 final accessors, actual session/factory boundaries and bounded HE validation still need implementation/review.
