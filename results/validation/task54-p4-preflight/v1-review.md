# Task54 P4 pure-preflight independent review

2026-11-07. Reviewed the released requirements, three complete retained deltas, their frozen/before/final source snapshots, runner/extractor code, raw terminal evidence and derived records. This review used read-only parsing and array comparisons; it did not run source, builds, tests, Engine/forward, HE, network, Git or subagents. The only written file is this report. Concurrent P3 fix2 source was not inspected.

## Independent verdicts

| Dimension | Verdict | Meaning |
|---|---|---|
| SPEC | **PASS for the released pure-preflight implementation/evidence scope** | Fixed candidates, actual engine reuse, finite matrix, counts, source boundaries, first-affected witnesses and terminal handling comply. |
| QUALITY | **PASS for this bounded test harness; no concrete correction required** | No scoped correctness, security, performance or maintainability defect found. Static checks and runtime structural/state evidence support the implementation, separately from sensitivity. |
| Original numerical gate | **FAIL, unchanged** | I: 51/57 failed margins; NI: 57/57 failed margins at I1e-5/NI5e-4. |
| Signal numerical gate | **FAIL, unchanged** | I: 0/57 failed; NI: 5/57 failed at the same frozen caps. |
| Proposed NI4e-4 | **Read-only arithmetic supports >=8 separation for all retained Signal witnesses** | Inference from existing distances, not a new test result, accepted cap, HE precision result or execution release. |

The source is a correct way to measure a fixture which fails the released sensitivity criterion. Compile/static PASS does not turn the numerical result into PASS, and this review does not complete P3, full P4, P5 or D7.

## Requirements, identity and source scope

Read the complete release `task-54-p4-pure-preflight-release.md`, complete referenced sensitivity disposition and API map, the static ownership amendment and pre-numerical decisions. Independently rehashed:

- Release: `3ea8b9c272f500f487b7298c75e478b38d990a78d2ecd2628f828ce3e8e4170e`.
- Sensitivity disposition: `4c221f802d0cf62b973d74f97fbcaadde4bcc362a7f7c3e7cc30ad3a283e33d8`.
- API map: `47e84a312bc2a7f7eaf2eb3d5af75f8535daeea5be66a43284fdf7426ee8a321`.
- Static amendment: `8fd2b76875438b71ccaee3d52f543840b4ca692f8ca3264699d54c3eb7515ba4`.
- Reviewed report: `a77838545b256c7cbab4b7fdf332ffe599d85c95be83b43fc867538655e59937`.
- Validation manifest: `939bd697db1238e515593e339edc5ea4496bf0e59f04a989f30ce06ca33640b2`; **all 47 listed artifact hashes independently match**.

Every complete unified patch was independently reconstructed from retained before/frozen bytes (new fixture file from empty input) and matched the retained patch exactly. Frozen/final copies match byte-for-byte; freeze-before/freeze-after lists are identical. Source pins are study_plain `d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b`, p4_fixtures `9a09235dc09d08acb62f40359d8bf6c75cf4eec3400117b621b7d3099dd815b3`, and study_reference `fcb7f1d34fd8c87527af4adf59c0ebd919eeeb2a07f5f635a965a436c233c9af`.

`study_plain.patch` adds only the cfg(test), crate-private child declaration. Removing that declaration reproduces the entire before file exactly. The reference before/frozen comparison independently confirms one exact comment/function-block move before tests, with only its separating blank line relocated; body, signature, visibility and all other bytes are unchanged. The amendment therefore adds no codec/runtime behavior. This assessment is of retained frozen source, not concurrent files or an assumed recent commit.

## Actual engine, fixtures and controls

Line references below are to validation-directory frozen copies. `p4_fixtures.frozen.rs:34-100` constructs private Engine/PlainLayer/StageSpec. `trajectory:322-440` calls the existing `Engine::train` at line426; the parent's actual engine uses rate.delta for existing stage gradients and rate.mean for existing updates (`study_plain.frozen.rs:623-635`). Range inspection invokes existing checked `forward` at fixture line299; it does not calculate alternative gradients. The delta contains no copied recurrence, public synthetic materializer, production interface widening, dependency or HE constructor. The module is test-only and does not construct a plan or execution capability.

All six required [2,4,w] shapes and eight baseline families are present: Square C3/C2/C10 with w3/2/10, active b and SGD; Poly C3/C2/C10 with w4/2/16, frozen b and NAG .9; plus the four-rate and first-reference Poly C3/P4 schedules. Original and the single prescribed Signal replacement match the fixed formulas. Independently checked all logged initial tensor values/bits against those formulas and all **138 CALL_INPUT records** against exact dimensions, classes, partitions, kept/total row counts, x, labels and coefficient/momentum bits. Logical row r spans an entire update; 8+4 does not restart labels or features. Padded outputs participate with zero targets in the actual engine's existing target rule.

SGD uses mean.06 and delta2/C separately. NAG smoke .04/delta1/kernel.04/reference.08 follows the retained pre-numerical ruling. Scheduled means/parts are .04,.08,.02,.08 with [12,8,5,12], and .08,.08 with [8,8]. Logged reference bits are fixture identity evidence; the unscaled engine has no reference argument, so this is not actual encrypted reference-basis/configuration proof.

Mutants use the same train path, changing only the released inputs/configuration. DoubleFold changes mean while retaining delta; PhysicalDenominator changes mean by valid/8; LostTail keeps first8 with original mean/delta. ResetVelocity fills the actual existing W/b velocity arrays with zero before later updates (lines383-387); Sgd sets actual Engine.momentum=None (329-331). Logged Sgd velocity buffers remain zero. NoUpdate directly uses the baseline's initial snapshot and invokes no alternate training implementation. DropSecond retains identical step0/step1 state bits, proving no second commit in retained evidence.

All **120 control classifications and first-affected labels** independently agree with the fixed matrix: per candidate41 applicable,11 equivalent,8 inapplicable. Delta1 omission/doubling, C2/P2 width equivalence, Square C10 ten-class equivalence and singleton lost-tail/denominator inapplicability are valid algebraic exclusions. They invoke no actual train call and are not counted as discriminated bugs.

First affected zero-based indices follow semantics: constant-first/lag/reset/drop-second1; skipped decrease2; skipped restart3; all other applicable controls0. Every retained state prefix before that index is bit-identical to baseline; no later endpoint is substituted for a weak first witness. Separate NoUpdatePerLayer witnesses select W only at update0 in each of two trainable layers. Frozen biases never supply these witnesses. Applicable general comparisons include W and only active biases; velocities remain diagnostic state rather than incorrectly supplying W/b sensitivity.

The single nested traversal runs both candidates and all prescribed rows before the final Signal assertion (`run_preflight:517-611`, test:619-635). There is no threshold/candidate search, adaptive witness-update choice, per-witness replay or hidden alternate train path. Largest-coordinate selection within the specified first affected snapshot is the defined max norm, not adaptive update selection.

## Execution ledger and independently reconciled numerical evidence

Each candidate has 12 baseline and56 mutant actual calls, **68 each /136 total**. The call counter increments immediately before the engine call; candidate_call ordinals are exactly1..68. There are41 engine trajectories per candidate including8 baseline trajectories; NoUpdate and excluded rows need no trajectory. The fixed two-candidate design and <100 per-call guard respect100/candidate and200 total, with <=2 layers, input2, output<=16 and <=12 rows/update. Unused budget was not spent.

The raw log emits both candidate summaries and total136 before final failure. Stderr identifies the final Signal aggregate assertion at fixture line634. Both harness_failures lists are empty. No train/forward failure or missing snapshot is reported. The observed terminal result is exit101, one failed selected test,189 filtered tests; the final assertion was reached after complete collection rather than an earlier abort.

Independent read-only parsing checked all **228 raw MARGIN records** against retained W/b snapshots: coordinate/value bits, absolute difference, true maximum over allowed tensors/layers, exact epsilon bits, normalized distance and >=8 status all match. Also compared all TSV fields/order and counts against raw records: all-margins228, failed-margins113 (108 Original +5 Signal), all-controls120 and all-hidden-ranges276. No record is missing or relabeled.

`extract_evidence.py` intentionally separates the first textual control name from the later numeric control value; the latter initially wins the dictionary collision, is saved as control_value, then the former is restored by explicit first-match extraction. This is correct for these logs. Optional required_layer is retained for no-op W rows, and control bits/values and epsilon bits survive extraction. The TSVs are faithful derived NONSCIENTIFIC_TEST_EVIDENCE, not scientific binary output, training accuracy or encrypted measurement.

| Signal witness failing original NI5e-4 | Index / layer / coordinate | Measured max absolute difference | Original NI D | Inferred D at NI4e-4 |
|---|---|---:|---:|---:|
| square-c10 / NoUpdatePerLayer | 0 /1 /[2,0] | 3.58203125000000622e-3 | 7.16406250000001243 | 8.955078125000016 |
| square-c10 / LostTail | 0 /0 /[0,1] | 3.82214355468749913e-3 | 7.64428710937499822 | 9.555358886718748 |
| poly-c3-p4 / WrongWidth | 0 /1 /[3,0] | 3.71093749999998335e-3 | 7.42187499999996625 | 9.277343749999957 |
| poly-c10-p16 / LostTail | 0 /0 /[0,1] | 3.56250000000000013e-3 | 7.12500000000000000 | 8.90625 |
| poly-four-rate / Sgd | 0 /1 /[3,0] | 3.51562499999999445e-3 | 7.03124999999998845 | 8.789062499999986 |

All five pass I1e-5. Signal has57 required witnesses/backend:41 applicable controls plus16 separate per-layer W no-op witnesses. Original's failures and Signal's failures remain separate; neither is discarded.

## Range, finite state and static quality

All1,760 logged tensor snapshots /17,920 scalar values independently decode as finite, and their decimal values round-trip to their logged F64 bits. Every3,024 frozen b/Vb entries has exact positive-zero bits. Every logged range scalar is finite. Baseline before/after rows are separately marked baseline_gate=true; mutant diagnostics are never used to meet the prerequisite. The existing forward checks all intermediate arithmetic for finiteness before returning a cache, so its successful ranges do not rely on unchecked NaN extrema behavior.

| Candidate | Baseline max hidden activation | Baseline max hidden preactivation | Activation <=.75 |
|---|---:|---:|---|
| Original | 1.71278436543640181e-3 | 5.37112124049642960e-3 | PASS |
| Signal | 3.28848736694946298e-1 | 5.00000000000000000e-1 | PASS |

The48 baseline range records cover before/after all24 baseline train steps across both candidates; all276 total range records cover baseline and mutant inputs. Preactivation is separately observable and is not substituted for the activation prerequisite. This is a bounded plaintext fixture result, not encrypted AA/bootstrap or secure-range adoption.

Retained command statuses, exact argv/environment and logs support [101,0,101,101,0,0,0,101] for stems01..08. RED is the missing run_preflight compile interface. Initial Clippy correctly caught owned unnecessary_lazy_evaluations and pre-existing items_after_test_module. The owned fix and exact authorized reference relocation retain strict -D warnings; final Clippy05, format06 and no-run compile07 exit0. They execute no numerical tests. Exactly one retained Cargo test invocation lacks --no-run:08. All runner statuses retain incremental0/jobs8/rayon8 and disk bytes; minimum67,176,857,600 exceeds805,306,368. No check was rerun during review.

Security/source isolation is appropriate: no public test capability, secret/HE/network entry or production arithmetic delta. Performance is bounded by the finite loops and actual counter guard. Correctness is supported by actual engine/forward reuse, exact inputs and independent evidence reconciliation. Maintainability is appropriate for this fixed test matrix: explicit candidate/version/control labels and retained complete state make negative outcomes inspectable. No broad new framework or unrelated correction is warranted.

## Read-only NI4e-4 mathematical assessment

The actual minimum over all57 required Signal absolute distances, including both layers' no-op W witnesses, is d=3.51562499999999445e-3 at poly-four-rate/Sgd update0, output W[3,0]. Therefore:

- d /4e-4 = **8.789062499999986 >=8**.
- The largest uniform NI cap compatible with that measured minimum and an8x requirement is d/8 = **4.394531249999993e-4**; proposed4e-4 is smaller.
- All57 retained Signal NI witnesses have distance/candidate-cap>=8. Their limiting five values are shown above. I stays1e-5, with minimum Signal normalized distance351.56249999999943.
- If a later actual HE result independently meets the all-entry1x bound at that frozen cap, triangle inequality gives at least8.7890625-1=7.7890625 separation at the limiting witness, exceeding the separate6x comparison requirement. This conditional implication does not establish that HE achieves1x.

No observed control-coverage, first-index, finite-state or baseline-range limitation invalidates this arithmetic for the prescribed Signal matrix. Equivalent/inapplicable rows remain excluded for the same algebraic reasons, and Original is not rescued or relabeled. This inference covers the retained matrix only; it does not establish actual HE precision, failure-path behavior, bias decoded-zero bounds, velocity scaling parity, resources, security, or scientific performance.

A root decision to freeze a smaller cap **before any HE** is permitted by the disposition, subject to feasibility and later separate release. This review neither selects a cap for root nor authorizes source changes, characterization reruns, keys or HE. No loosen-after-HE-failure policy is implied. Preserve the original frozen source, caps, raw logs, report and FAIL statuses even if a separately recorded PRE-HE decision is made.

## Remaining acceptance boundary

P3's actual shared consuming pre-key/factory positive-and-negative boundary and real D7 connection remain broader acceptance work. P4 client/session binding, packet/state/poison semantics and bounded HE parity/count/metadata tests are not performed by this pure harness. P5 blocks, production D7 gates/durable start, secure range/precision/resource validation and every scientific campaign remain separate. A correct pure test implementation and a numerically supported candidate tighter cap do not complete or release any of them.
