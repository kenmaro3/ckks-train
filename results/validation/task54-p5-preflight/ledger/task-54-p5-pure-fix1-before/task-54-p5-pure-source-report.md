# Task54 P5 pure block source/static report

**DONE — authorized tests-first source/compile/static assignment only.** No numerical sensitivity, hidden-range, target-isolation or HE PASS is claimed. Source-defined arrays/order/rates are frozen; emitted runtime value/bit manifests are prospective until a separately authorized run.

## Ownership and pins

Source base supplied by root: `77fd08de743b9a310cf9ae42f8036e3931ff0c5c`. Root subsequently reported docs-only commit `6323e0a1e3e17526241c8b6483cdd6077823b350`; no Git commands/mutations were performed by this implementer.

Exactly two source files:

- NEW `crates/ckks-train-bench/src/study_plain/p5_fixtures.rs` (1,051 lines).
- `crates/ckks-train-bench/src/study_plain.rs`: append only a `#[cfg(test)] pub(crate) mod p5_fixtures;` child declaration (three lines including separating blank).

The saved original `study_plain.rs` SHA256 is `d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b`, matching the API map source pin. A byte comparison proves the final file is that exact original plus the declaration. Existing P4 is untouched. Final pins:

```text
fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a  crates/ckks-train-bench/src/study_plain.rs
b9cdf2a685ddea16d054ff56b916f738e7110e9857e2c54f3075d08fb58efd00  crates/ckks-train-bench/src/study_plain/p5_fixtures.rs
```

`task-54-p5-pure-source-validation/owned-two-file.diff` contains the complete source diff, generated against the saved original without Git. Final file copies, SHA256 inventory, the actual compile-RED skeleton (verified against its recorded pre-command hash), and build/test-binary SHA256 custody are retained in that same directory. No production arithmetic/API/materializer/profile/readiness file is modified.

## Fixed implementation

The closed private cfg(test) recipe implements exactly A/B/C/D:

| Family | Stages: input/hidden/padded; classes | Activations | Rates; ordered parts |
|---|---|---|---|
| A | 2/4/4 C3; 4/2/2 C2 | Poly/Identity each | .04,.08,.02,.08; [8,4];[8];[5];[8,4] |
| B | A plus 2/3/16 C10 | Poly/Identity each | Same four rates/parts |
| C | A widths/classes | stage0 Poly/Identity; stage1 Poly/Square | .04; [8,4] |
| D | A widths/classes | Poly/Identity each | .08,.08; [8];[8] |

Poly is the existing SquarePlusLinear activation. Every stage uses NAG .9, RSS delta1, reference .08, frozen positive-zero bias and velocity, hidden W=.45/fan_in, head W=.75/hidden_width. Global logical row r spans each update across micros, restarts only at the next update; x[r,0]=.875+.125*(r%2), x[r,1]=.875+.0625*(r%3), y_s[r]=r%C_s. All deterministic target perturbations are **all labels of the selected stage replaced by legal class0**, not chosen from a numerical search. Other stages' label bytes are retained exactly.

True baseline updates invoke actual multi-stage `Engine<f64>::train`. Every baseline entrance and final endpoint captures actual existing `forward` hidden arrays and checks finite/abs<=.75. Initial and post-update W/b/velocity dimensions, all finite values, and exact positive-zero b/velocity_b are checked and emitted. No arithmetic recurrence is copied.

At each boundary, separate persistent one-stage real Engines implement actual local-head-as-hidden and post-update-hidden handoffs; their H=P dimensions remain shape-compatible. Omitted-local-update skips the selected source stage's real train call and retains the actual old hidden handoff. Four schedule/row controls invoke the real full-block Engine after changing only the intended existing velocity/rate/rows. Target isolation uses one real full-block first update per stage, checks every other stage's W/b/velocity bit-identical and requires a selected-stage weight witness.

Before every real Engine literal/train/external forward, one guarded invocation counter is reserved and a `NONSCIENTIFIC_TEST_EVIDENCE BEFORE_CALL` record is printed. The source holds train/Engine/forward caps156/61/179; no planned total is presented as an observed count. Internal Engine forward calls are not separately counted.

The future ignored run first emits all four source-defined initial/input/rate numeric values and exact f64 bits before its first Engine. It emits every baseline hidden array, all state entries and all paired comparison entries including zero differences/unselected stages, effective rates, actual row controls, dimensions, first-affected update, and selected weight witness with original values/bits/difference/cap/normalized margin. Each applicable margin must be finite and **>=8x I1e-5 and NI4e-4**; map's >2x proposal is not used. Failure details and final PASS/FAIL/inapplicable/equivalent totals are retained; no constants are tuned on failure.

Only read-only, closed-family recipe/oracle seams are exposed within cfg(test): initial arrays/layer specs/classes, rows and labels, rates/parts/reference, baseline post states and old/final hidden snapshots. The future sibling's `block_oracle` path uses the identical array factory and actual baseline, with local guarded diagnostics and no mutant sweep. It has not been invoked. There is no caller-mutable Engine/general materializer API or HE source in this assignment.

## Source-derived inventory; not observed numerical execution

| Actual invocation category when later run | A | B | C | D | Total |
|---|---:|---:|---:|---:|---:|
| Full-block baseline train |4|4|1|2|11|
| Two handoff mutants per boundary, one-stage train |16|48|4|8|76|
| Omitted boundary-source-stage train |4|16|1|2|23|
| Applicable schedule/row train |16|16|3|2|37|
| First-update stage target train |2|3|2|2|9|
| **train total** |**42**|**87**|**11**|**16**|**156**|
| **Engine objects** |**13**|**26**|**12**|**10**|**61**|
| Baseline entrance/final external forward |10|15|4|6|35|
| Handoff mutant external forward |24|64|6|12|106|
| Omission mutant external forward |8|24|2|4|38|
| **external forward total** |**42**|**103**|**12**|**22**|**179**|

For S stages/U updates/B=S-1 boundaries/K applicable schedules: baseline Engines1; handoff2BS; omissionBS; schedulesK; targetsS. External forwards: baseline(U+1)S; handoff2BU(S+1), retaining every old hidden plus one selected head/post forward; omissionBUS. This is exactly the mapped cap, with no discrepancy and no runtime used to determine counts. Static arithmetic inventory is retained in `source-derived-inventory.json`.

| Schedule control | A | B | C | D | First affected update |
|---|---|---|---|---|---|
| ResetVelocity | applicable | applicable | inapplicable (one update) | applicable |1|
| ConstantReference | applicable | applicable | applicable | equivalent (all rates=.08) |0 where applicable|
| LostTail | applicable | applicable | applicable | inapplicable (no multi-micro) |0 where applicable|
| MicroRowRestart | applicable | applicable | applicable | inapplicable (no multi-micro) |0 where applicable|

LostTail removes later micros and their corresponding global rows/targets, retaining only the first part; it does not use a fake denominator. MicroRowRestart retains all parts but resets x row patterns and each stage label's row index within each part. ResetVelocity resets the existing persistent buffers, retains NAG and actual scheduled rates. ConstantReference replaces actual current mean with .08 while retaining velocities. They are distinct controls. Handoff/omission/target controls first affect update0. Prior updates before a declared first affected update are checked bit-identical.

There are40 declared control rows:15 local-boundary rows +16 schedule/row rows +9 target rows. Of these36 are applicable,1 equivalent,3 inapplicable, giving72 separately reported I/NI margins. All local omission rows refer to boundary source stages, as mapped; no additional terminal omission fixture is introduced. Stage-target isolation is the actual evidence; no fabricated cross-stage-backprop mutant is claimed. Stage-local C_s/padded label validation precedes every actual train path, while Engine classes privately uses max(C_s).

## Actual compile/static validation

TDD was applied as compile-only RED, honoring the explicit no-runtime release. RED is a missing intended helper, not an invented numerical failed test. All Cargo commands used the existing target and CARGO_INCREMENTAL=0/CARGO_BUILD_JOBS=8/RAYON_NUM_THREADS=8. The command wrapper checked free disk>=805306368bytes before launching, retained exact argv/cwd/environment/PID/UTC start/end/status/before-and-after source SHA256, and saved stdout/stderr independently. Tool session handles are retained separately. Each started command was polled to terminal.

| Record | Result |
|---|---|
|01-compile-red|exit101; E0425 missing intended `run_pure_sensitivity`|
|02-rustfmt|exit0; exactly owned files, skip_children=true|
|03-compile|exit0; targeted ignored-test --no-run, no warnings|
|04-rustfmt|exit0; owned files only|
|05-clippy|exit101; one nonminimal_bool lint in omission branch|
|06-rustfmt|exit0; owned files only|
|07-clippy|exit0; strict bench --lib --tests, no warnings|
|08-compile-final|exit0; targeted ignored-test --no-run, no warnings|

The lint was resolved with an explicit conditional: omission returns success without invoking train; all other paths call real train. Its short-circuit/count semantics remain intact. The final Clippy and no-run hashes agree exactly with the final pins above.

Exact Cargo command forms:

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-bench --lib study_plain::p5_fixtures::p5_pure_block_sensitivity_gate --no-run
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo clippy -p ckks-train-bench --lib --tests -- -D warnings
```

Formatting was `rustfmt --edition 2024 --config skip_children=true crates/ckks-train-bench/src/study_plain.rs crates/ckks-train-bench/src/study_plain/p5_fixtures.rs`. The compiled test binary `target/debug/deps/ckks_train_bench-fe037ad79be4cfe0` was only hashed/read; it was never invoked.

## Unmeasured gates and custody

No fixture/Engine/forward construction or execution, test binary execution, unfiltered test run, HE/CKKS setup/planner/proof/metadata/Encoder/keygen/evaluator/bootstrapper/ciphertext work, dataset/network/SSH, cleanup, subagents or Git mutation occurred. The sole pure entrypoint is `#[ignore]`; the runtime manifest, actual numerical margins, all-entry invariance, hidden ranges, counters and PASS/FAIL outcomes remain **unmeasured**. Compilation/Clippy do not establish any numerical sensitivity, canonical admission, scientific/performance/security result, paired-HE acceptance or full Task54 completion.

No source-derived count/control gap was found. Root must independently review this exact source/static inventory and issue any one-shot pure execution release. A candidate's eventual failed margin/range/isolation must be retained without tuning, omitted controls, changed caps or automatic HE work. Canonical materialization/readiness, all registered campaigns, D7 and all HE/remote work remain held.
