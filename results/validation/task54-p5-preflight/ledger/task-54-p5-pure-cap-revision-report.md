# Task54 P5 PRE-HE cap revision v2 — source/static report

**DONE for the authorized two-literal source/static revision.** No v2 numerical/fixture/test/Engine/forward or HE execution occurred. The original v1 FAIL and its seven NI failures retain their original records and disposition; no output was relabeled.

Read `task-54-p5-pure-cap-revision-release.md` (release SHA256 prefix42f40f4b) and ratified design section14 before editing. The decision keeps the same candidate trajectories, controls and required>=8x margin while tightening the prospective small P5 NI HE error cap to3e-5. It does not establish HE accuracy at that cap or alter P4/scientific/canonical gates.

## Exact allowed change and identity

Only `crates/ckks-train-bench/src/study_plain/p5_fixtures.rs` changed, at two literal occurrences:

```diff
-const VERSION: &str = "task54-p5-pure-v1";
-const CAPS: [(&str, f64); 2] = [("I", 1e-5), ("NI", 4e-4)];
+const VERSION: &str = "task54-p5-pure-v2";
+const CAPS: [(&str, f64); 2] = [("I", 1e-5), ("NI", 3e-5)];
```

Both original replacement sites occurred exactly once. Reversing exactly those two replacements reproduces every original fixture byte and SHA256 `fbf604ddc4a3dfae6f3c37bd6909d35adee8f83f639453fb97239bbd46bb2943`. Thus arrays, row order/labels, initial state, rates/reference, parts, every control/branch, all counts and8x rule are unchanged. Parent file and all other source files were not edited.

| File | Before SHA256 | After SHA256 |
|---|---|---|
| study_plain/p5_fixtures.rs | fbf604ddc4a3dfae6f3c37bd6909d35adee8f83f639453fb97239bbd46bb2943 | cd580c14c8f6aa8099c113701890ec2231bb5b162f2375b63103c6562dcdb99e |
| study_plain.rs | fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a | fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a |

New custody directory: `task-54-p5-pure-cap-revision-validation/`. It contains source-before/final copies, `two-literal.diff`, exact replacement/inverse identity JSON, final source pins, all command/raw outputs, read-only compiled test-binary identity, and SHA256 inventories proving the protected original source report, runtime1 report/review/validation bytes and all files preserved in `task-54-p5-pure-before-cap-v2/` remain unchanged. No original report or runtime record was appended/overwritten. Only this new report and new validation directory were written besides the allowed fixture literals.

## Actual validation

Each command used the existing target and required CARGO_INCREMENTAL=0/CARGO_BUILD_JOBS=8/RAYON_NUM_THREADS=8 environment. The wrapper verified free disk>=805306368bytes before launch; retained exact argv/cwd/environment, child PID, UTC start/end/exit, source-before/after hashes, stdout and stderr; and each command was polled to terminal on its same tool session. No binary was invoked.

| Record | Exact scope | Result |
|---|---|---|
|01-rustfmt-check|owned fixture only; --check --edition2024 --config skip_children=true|exit0, no formatting changes|
|02-compile-final|existing ignored pure test target, --no-run|exit0, no warnings|
|03-clippy-final|bench --lib --tests -- -D warnings|exit0, no warnings|

Exact commands:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 rustfmt --check --edition 2024 --config skip_children=true crates/ckks-train-bench/src/study_plain/p5_fixtures.rs
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_plain::p5_fixtures::p5_pure_block_sensitivity_gate --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

All three before/after source inventories match each other and the final pins above. `target/debug/deps/ckks_train_bench-fe037ad79be4cfe0` was compiled and hashed, never executed. This is compile/static validation, not v2 sensitivity/range/invariance or runtime-guard evidence.

## Retained limits and future holds

The pure source still contains exactly156actual train/61Engine/179external-forward per separately authorized complete run,40control rows,36applicable/1equivalent/3inapplicable,72I/NI margin records. I1e-5 and required>=8 remain unchanged. No case or control was removed. The shared paired phase remains fixed14oracles/14Engine/40train/128forward and has not been executed.

This assignment consumed zero additional Engine/forward/fixture/test runtime calls. Root retains the completed v1 consumed156/61/179 inventory and its failed NI witnesses. A later separately reviewed/released full v2 run would add156/61/179, giving cumulative312train/122Engine/358forward across v1+v2. No such additional run was performed or inferred here. No HE/crypto/planner/key, dataset/network/SSH, Git mutation, cleanup or subagent operation occurred.

Root must independently review this exact two-literal delta/source identity before any v2 runtime release. A future failure of the tighter3e-5 NI HE cap remains a failure; this revision supplies no automatic loosening or HE continuation. Full Task54, canonical/D7/campaign, P4 and scientific/security/performance evidence boundaries remain unchanged.
