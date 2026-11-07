# Task54 P4 pure preflight report

2026-11-07. **Measured pure gate FAIL for both fixed candidates.** The single authorized numerical invocation exited101 after all rows were emitted. Signal satisfies all I margins but misses five NI8x prerequisites at the frozen NI5e-4 cap. No source/cap/rate/input changes or numerical rerun followed. This result does not release HE, keys, a P4 bridge, production D7, or scientific admission.

## Authority and source boundary

Read/hash-verified release3ea8b9c272f500f487b7298c75e478b38d990a78d2ecd2628f828ce3e8e4170e, disposition4c221f802d0cf62b973d74f97fbcaadde4bcc362a7f7c3e7cc30ad3a283e33d8, map47e84a312bc2a7f7eaf2eb3d5af75f8535daeea5be66a43284fdf7426ee8a321 and source-only release review01d5c66410cc139c3688d7978c7fc98b5d76c39c1a5e9c2e5a27020a4e917ec5. Root transferred source/Cargo after P3 returned. Main base was supplied by root as deb725baea105b553b1c83bc9f229344c6537f85; no Git command was used.

Source changes: the only study_plain.rs delta is a cfg(test) crate-private child declaration. New study_plain/p4_fixtures.rs constructs fixed synthetic private Engine<f64>/PlainLayer fixtures, executes the existing Engine::train, snapshots state and calls the existing forward for range inspection. It contains no copied forward/backward/gradient/optimizer recurrence and creates no synthetic MaterializedProfile or trusted plan. No dependency, export, production arithmetic or P3 source was changed.

Strict Clippy also exposed an existing items_after_test_module in study_reference.rs. Root amendment8fd2b76875438b71ccaee3d52f543840b4ca692f8ca3264699d54c3eb7515ba4 granted the narrow third-file hunk: move the exact registered_case_canonical comment/function bytes before the test module. The block was moved byte-for-byte, with no signature/body/visibility/import/test change or lint suppression. Before/final snapshots, exact deltas and the byte-move verification are retained.

Source/Cargo was returned to root immediately after final frozen evidence capture. Subsequent work in this report is read-only extraction from the retained raw log and ledger writing. Unowned P3/.vscode/result files were preserved; no subagent, network, remote host, dataset, CKKS parameter/Encoder/key/evaluator/ciphertext/bootstrap constructor or operation was used.

## Fixed experiment and counts

Root resolved the map's unstated NAG smoke index before execution: smoke mean .04/delta1/kernel .04/reference .08, matching the first prescribed schedule rate. SGD smoke mean .06/delta2/C/kernel mean*delta remains unchanged. This decision is saved in pre-numerical-decisions.md. Caps stayed I1e-5 and NI5e-4; required normalized distance8; baseline hidden activation max magnitude<=.75; mutant ranges diagnostic. No smaller cap was selected or tested.

| Family | Shape | C | Parts/update | Mean rates | Active b / optimizer |
|---|---|---:|---|---|---|
| square-c3 | [2,4,3] | 3 | 8 | .06 | active / SGD |
| square-c2 | [2,4,2] | 2 | 5 | .06 | active / SGD |
| square-c10 | [2,4,10] | 10 | 8+4 | .06 | active / SGD |
| poly-c3-p4 | [2,4,4] | 3 | 8 | .04 | frozen / NAG .9 |
| poly-c2-p2 | [2,4,2] | 2 | 5 | .04 | frozen / NAG .9 |
| poly-c10-p16 | [2,4,16] | 10 | 8+4 | .04 | frozen / NAG .9 |
| poly-four-rate | [2,4,4] | 3 | 8+4;8;5;8+4 | .04,.08,.02,.08 | frozen / NAG .9 |
| poly-first-reference | [2,4,4] | 3 | 8;8 | .08,.08 | frozen / NAG .9 |

Original initial W/b and logical x use the exact map formulas. Signal uses x=[1,((r%3)-1)/4], Square hidden rows[.5,0]/all output W=.875/active b0, Poly hidden rows[.25,0]/all output W=.25/frozen b0. All y=r%C spans a whole logical update across unequal micros. Each oracle train call concatenates logical rows once. Actual per-call source/mean/delta/kernel/reference/momentum bits, x/y/partitions, full W/b/velocities and baseline/control coordinate/value bits are in08-single-numerical.stdout.log; this is NONSCIENTIFIC_TEST_EVIDENCE, not a result JSON or accuracy/timing measurement.

| Candidate | Actual train calls | Baseline families | Control rows | Applicable / equivalent / inapplicable | I failed margins | NI failed margins | Gate |
|---|---:|---:|---:|---|---:|---:|---|
| Original | 68 | 8 | 60 | 41 / 11 / 8 | 51/57 | 57/57 | FAIL |
| Signal | 68 | 8 | 60 | 41 / 11 / 8 | 0/57 | 5/57 | FAIL |

**136 total actual Engine::train calls**, counted immediately before invocation; all returned successfully and were snapshotted. Per candidate12 baseline calls +56 mutant calls. No-op controls reuse unchanged initial state (0 calls); equivalent/inapplicable controls make0 calls. There are41 engine fixture trajectories/candidate, no replay for individual witnesses, <=100 calls/candidate and <=200 overall. All8 baselines and60 control rows/candidate were emitted before the aggregate signal assertion. The original candidate failure did not hide signal/control rows. Count manifest and each call ordinal are retained in raw stdout.

First affected zero-based updates are frozen from control semantics: smoke/no-update0; four-rate constant-first/lag/reset1, constant-reference/SGD0, skipped decrease2, skipped restart3; first-reference drop-second/reset1, half-means/no-update0. Separate per-layer no-op W witnesses are always measured at update0. Later prefixes are retained as state snapshots but do not replace first-affected witnesses.

## Signal misses and absolute distances

All five misses below use NI epsilon5e-4, required8. They also pass I epsilon1e-5. Every coordinate is from actual engine state; no frozen bias supplies a no-op witness.

| Family/control | Update | W layer/coordinate | Baseline | Control | Max abs distance | NI normalized distance |
|---|---:|---|---:|---:|---:|---:|
| square-c10/NoUpdatePerLayer | 0 | 1/[2,0] | 8.71417968749999994e-1 | 8.75000000000000000e-1 | 3.58203125000000622e-3 | 7.16406250000001243e0 |
| square-c10/LostTail | 0 | 0/[0,1] | 0.00000000000000000e0 | 3.82214355468749913e-3 | 3.82214355468749913e-3 | 7.64428710937499822e0 |
| poly-c3-p4/WrongWidth | 0 | 1/[3,0] | 2.42578125000000006e-1 | 2.46289062499999989e-1 | 3.71093749999998335e-3 | 7.42187499999996625e0 |
| poly-c10-p16/LostTail | 0 | 0/[0,1] | 0.00000000000000000e0 | 3.56250000000000013e-3 | 3.56250000000000013e-3 | 7.12500000000000000e0 |
| poly-four-rate/Sgd | 0 | 1/[3,0] | 2.42578125000000006e-1 | 2.46093750000000000e-1 | 3.51562499999999445e-3 | 7.03124999999998845e0 |

The smallest applicable first-affected control distance is **3.51562499999999445e-3**, poly-four-rate/Sgd update0, output W[3,0]. This is also the minimum including per-layer no-op distances. These are measured distances under the existing caps, not permission to change epsilon or relabel the result.

| Signal family | Smallest applicable control | Max abs distance at first affected update |
|---|---|---:|
| poly-c10-p16 | LostTail update0 | 3.56250000000000013e-3 |
| poly-c2-p2 | TenClasses update0 | 8.55000000000000204e-3 |
| poly-c3-p4 | WrongWidth update0 | 3.71093749999998335e-3 |
| poly-first-reference | ResetVelocity update1 | 6.32812500000001776e-3 |
| poly-four-rate | Sgd update0 | 3.51562499999999445e-3 |
| square-c10 | LostTail update0 | 3.82214355468749913e-3 |
| square-c2 | TenClasses update0 | 3.90468750000000364e-2 |
| square-c3 | WrongWidth update0 | 1.98583984374999800e-2 |

| Signal no-op W requirement | Minimum over8 families | Family/coordinate | NI distance |
|---|---:|---|---:|
| Layer 0 | 7.12499999999999245e-3 | poly-c3-p4/[0,0] | 1.42499999999999840e1 |
| Layer 1 | 3.58203125000000622e-3 | square-c10/[2,0] | 7.16406250000001243e0 |

All114 normalized margin rows/candidate are in all-margins.tsv, preserving exact caps/bits, update, layer/tensor/coordinate, correct/control values/bits, absolute/normalized distances and status. failed-margins.tsv lists every108 Original and5 Signal failed backend/witness row. Original fails every applicable NI control and every NI per-layer W no-op prerequisite; six of its57 I rows pass and51 fail. No Original failure was dropped or counted as a Signal failure.

## Range and finite-state evidence

| Candidate | Baseline max hidden activation magnitude | Baseline max hidden preactivation magnitude | Activation prerequisite |
|---|---:|---:|---|
| Original | 1.71278436543640181e-03 | 5.37112124049642960e-03 | PASS <=.75 |
| Signal | 3.28848736694946298e-01 | 5.00000000000000000e-01 | PASS <=.75 |

Signal maximum activation .3288487366949463 occurs in poly-c2-p2 after update0. Square baseline max activation .25/preactivation .5; every Poly baseline except C2 has max activation .3125/preactivation .25. C2 Poly max preactivation .26082109375. The meaning of range is actual first-layer forward activation, with signed activation/preactivation minima/maxima also retained before/after every successful baseline or mutant step on that step's actual inputs. all-hidden-ranges.tsv retains every prefix and marks baseline_gate=true/false; mutant ranges are diagnostic and never used to satisfy the baseline prerequisite.

All1,760 logged tensor snapshots /17,920 scalar entries (W, b, W velocity, b velocity, including initial states and mutants) were independently parsed as finite. All3,024 logged frozen b/bias-velocity scalar entries retain exact positive-zero F64 bits. Runtime harness_failures=[] for both candidates; no train/forward failure, missing first-affected snapshot or range failure was emitted. These checks support harness/state correctness separately from the measured sensitivity FAIL. They do not measure encrypted precision or AA/bootstrap range.

## Equivalent and inapplicable controls

The following classification applies identically to both candidates, with0 actual calls for each row; all-controls.tsv retains all120 candidate-specific control rows and the exact first-affected NONE labels. Algebraic cases are explicitly excluded from discrimination totals, not called rejected bugs.

| Family | EQUIVALENT controls | INAPPLICABLE controls |
|---|---|---|
| square-c3 | none | PhysicalDenominator, LostTail |
| square-c2 | OmittedFold, DoubleFold, WrongWidth | PhysicalDenominator, LostTail |
| square-c10 | TenClasses | none |
| poly-c3-p4 | OmittedFold, DoubleFold | PhysicalDenominator, LostTail |
| poly-c2-p2 | OmittedFold, DoubleFold, WrongWidth | PhysicalDenominator, LostTail |
| poly-c10-p16 | OmittedFold, DoubleFold | none |
| poly-four-rate | none | none |
| poly-first-reference | none | none |

Omitted/double fold is equivalent where delta1 (Square C2 and all Poly); wrong width is equivalent for Square C2 and Poly P2 because2/width=1; hardcoded ten classes is equivalent for Square C10 because delta.2. Singleton partition families have neither lost tail nor a distinct physical/logical denominator. Repeated same-rate setter/reference-basis identity is not a distinct numerical bug control: the unscaled F64 engine has no reference parameter. P3/runtime getter identity remains a separate requirement.

## Commands, raw evidence and terminal statuses

All Cargo commands used current target with CARGO_INCREMENTAL=0, CARGO_BUILD_JOBS=8, RAYON_NUM_THREADS=8, --locked --offline. Each runner status records exact argv, environment and before/after disk bytes; minimum observed disk free was far above805306368B. No workspace/release/full-profile/P1/P2/P3 runtime suite ran. Clippy --tests compiled target tests but executed none.

| Log stem | Invocation purpose | Exit |
|---|---|---:|
| 01-red | Expected compile-only RED: missing run_preflight E0425,0 engine calls | 101 |
| 02-compile | Initial targeted --no-run compile | 0 |
| 03-clippy | Owned lazy Option warning + pre-existing reference item order | 101 |
| 04-clippy-owned-fix | Owned warning fixed; only existing reference item order remains | 101 |
| 05-clippy-amendment | Strict lib/tests Clippy after authorized exact helper move | 0 |
| 06-format | rustfmt --check on all3 owned paths | 0 |
| 07-final-compile | Final targeted --no-run compile before freeze | 0 |
| 08-single-numerical | Sole selected numerical test, --test-threads=1 --nocapture; aggregate signal assertion fails | 101 |

Each stem has complete .stdout.log/.stderr.log/.status.json. In particular08 retains990,919 stdout bytes and13,458 stderr bytes, including every row, both candidate summaries, total count, final assertion and terminal test status. extract_evidence.py only reads that log and writes ordinary NONSCIENTIFIC_TEST_EVIDENCE TSV/summary; it never invokes training. It disambiguates the textual control-name and numeric control-value fields in MARGIN records. Re-extraction is not a numerical rerun.

## Freeze, deltas and remaining boundary

Before/after numerical pins are identical (freeze-before.sha256 / freeze-after.sha256 / freeze-equality.txt):

```text
d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b  crates/ckks-train-bench/src/study_plain.rs
9a09235dc09d08acb62f40359d8bf6c75cf4eec3400117b621b7d3099dd815b3  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
fcb7f1d34fd8c87527af4adf59c0ebd919eeeb2a07f5f635a965a436c233c9af  crates/ckks-train-bench/src/study_reference.rs
```

Exact source snapshots are study_plain.before.rs/frozen.rs/final.rs, p4_fixtures.frozen.rs/final.rs (new file, no before state), and study_reference.before.rs/frozen.rs/final.rs. Three unified patches are study_plain.patch, p4_fixtures.patch and study_reference.patch. static-amendment-move-check.txt verifies the exact third-file block bytes. These source snapshots and all retained evidence are inventoried in MANIFEST.sha256.

The test implementation compiles and passes required strict lint/format; its frozen runtime structural and finite-state observations are complete. The chosen numerical fixture fails the explicitly frozen sensitivity criterion. There is no permission here for further tests, fixture search, epsilon selection, P4/HE execution or adoption. Root may review the actual absolute distances under the existing disposition before any separate PRE-HE tolerance decision; all current statuses and raw evidence remain FAIL. P3 final accessors, original pre-key/factory/session acceptance, bounded HE parity and production D7 remain separate outstanding work.
