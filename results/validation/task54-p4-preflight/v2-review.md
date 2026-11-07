# Task54 P4 pure cap revision independent review

2026-11-07. Scoped rereview of root's two-constant PRE-HE criterion revision, the new frozen evidence and design §11. The v1 complete source/evidence review remains the baseline. No tests, Engine/forward calls, HE, Git, network or source changes were performed. Read-only parsing/array-field comparisons add no numerical calls. Only this report was written; unrelated root documents/publication and source ownership remain with root.

## Separate verdicts

| Dimension | Verdict | Scope/qualification |
|---|---|---|
| SPEC | **PASS for the permitted source change, execution and numerical record** | Exact VERSION/NI-cap-only delta, unchanged fixture/matrix, strict static checks, prescribed single invocation and bounds verified. Derived-header correction below remains required before publication. |
| QUALITY: source/harness | **PASS** | The v1 implementation review remains applicable; no new numerical/source defect. |
| QUALITY: derived evidence custody | **Minor defect; publication pending correction** | Four v2 TSV comment headers incorrectly say invocation08. The records actually come from v2 invocation03 and reconcile exactly. Preserve initial artifacts and correct/re-extract/repin as below; no numerical rerun. |
| v2 Original numerical gate | **FAIL, retained** | I51/57 failures, NI57/57 failures. The outer test's success does not mean Original passed. |
| v2 Signal numerical gate | **PASS at I1e-5/NI4e-4** | All57 required witnesses/backend pass; minimum NI separation8.789062499999986. This is bounded pure sensitivity, not encrypted precision or execution permission. |
| v1 numerical gates | **Original FAIL and Signal FAIL, immutable** | Original NI5e-4 criterion and all old evidence remain unchanged. |

## Reviewed identities and exact source scope

Fully read the cap release `d0d4d728dddb7ccde60df23e7a0e737f3c886975225936cc3072fb316becede5`, the complete v2 report `3290a76c935d38f67699b48a25824605df2f9f9425b9cbd633bcd830422fd20a`, limited patch, runner/extractor/comparison code, static/terminal evidence and design §11. Rehashed v1 independent review `c4f8e63ec30fea4854556f037ce6413d071de80d60402ebc29a60feaea52a777` as the prior accepted review. Independently verified v2 manifest `c8d0eb705d023f617478c60d704f75aba74c99ac5d2e7d44fb539cba49490927` and **all53 listed hashes**. These identify the initially frozen v2 record reviewed here, before the header correction.

Reconstructed `source-delta.patch` from complete before/frozen fixture bytes and matched it exactly. Reversing only VERSION task54-p4-pure-v1 -> task54-p4-pure-v2 and NI CAPS5e-4 ->4e-4 reproduces the full v1 fixture byte-for-byte. I1e-5, separation8, hidden.75, all dimensions/inputs/rates/classes/labels/controls/first affected indices and actual engine calls are unchanged. Fixture source hash is `d4edd2262ff0b3e9c0792047fb5e4016d2b5532849c89a47ad3491d4c6757a68`.

All seven before/frozen/final source snapshots agree with their recorded pins; only that fixture differs between before and frozen. The other six dependency files are byte-identical. All seven pins are identical before/after each of Clippy, format and the numerical command and in freeze-before/freeze-after. This re-review hashes the P3 dependency snapshots for stability without reopening their implementation review.

Independently verified all47 old v1 manifest pins, the old manifest itself and all entries in v1-preservation-before/after, which are identical. The retained v1 source/report/raw/derived artifacts were not overwritten. A stricter separately versioned cap does not revise the old FAIL outcomes or consume new fixture candidates.

## Static commands, runtime and budget

The three status files contain the exact released commands, in this order, all exit0:

1. `cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings`
2. `rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_plain/p4_fixtures.rs`
3. `cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate -- --test-threads=1 --nocapture`

Clippy's --tests compiles tests and executes none. There is one retained v2 numerical invocation, no no-run loop or retry. All statuses retain CARGO_INCREMENTAL0/CARGO_BUILD_JOBS8/RAYON_NUM_THREADS8. Minimum observed disk67,034,116,096B exceeds805,306,368B. The numerical stdout terminal is one selected test passed, zero failed,189 filtered; stderr retains the corresponding binary execution. Source is frozen around every command.

The complete candidate summaries precede total136 and successful terminal status. Each candidate has68 actual Engine calls (12 baseline +56 mutants),8 baseline families,60 control rows, empty harness_failures and no train/forward/missing-snapshot failure. NoUpdate/equivalent/inapplicable rows add no calls; call ordinals are exactly the prior1..68/candidate. The authorized additional budget is68/candidate,136 total; cumulative pure-preflight calls are272 including v1. No capacity, scientific or encrypted operation is released by this budget.

## Independent v1-v2 evidence comparison

Compared records directly from retained logs, independently of the supplied comparison script. CALL_INPUT comparison includes each complete multiline x array and its trailing labels/coefficient/momentum bits, rather than only the first line. The following records are text-identical across runs:

| Kind | Count | Retained identity |
|---|---:|---|
| CALL_INPUT | 138 | Complete x/y, shape/C/P/partitions/rows and source/mean/delta/kernel/reference/momentum bits |
| SNAPSHOT | 1,760 | Complete W/b/velocity values and bits |
| RANGE | 276 | Before/after phases, signed extrema, activation/preactivation magnitudes and baseline markers |
| CONTROL | 120 | Classifications and first affected indices |
| ACTUAL_TRAIN | 136 | Candidate/family/control/update/call ordinals |
| TOTAL | 1 |136 actual calls, same candidates and no-HE declaration |

The manifest changes only VERSION. Both candidate identity/count/harness prefixes are identical; numerical failures/status differ as required by the revised cap. All114 I margin records are text-identical. All114 NI records change only epsilon, epsilon_bits, normalized distance and consequent status; each baseline/control coordinate/value/bit and maximum absolute distance remains identical. All228 raw margins were checked for the exact revised cap bits, difference/cap arithmetic and >=8 decision. The NI4e-4 F64 bits are4556014321273684781; I1e-5 remains4532020583610935537.

Exactly five status flips occur, all Signal NI FAIL->PASS. Original retains108 failed backend/witness rows; Signal has zero. Both all-margins228 and failed-margins108 TSV record fields/order independently match raw records, including disambiguated textual control and numeric control_value. Control/range TSV contents are identical to v1. There are no field omissions, swapped controls or extra candidate/control rows. The inaccurate comment header described below affects provenance labeling only.

| Former Signal NI miss | Unchanged absolute distance | Measured v2 NI D |
|---|---:|---:|
| square-c10 / NoUpdatePerLayer, layer1 | 3.58203125000000622e-3 | 8.95507812500001599 |
| square-c10 / LostTail | 3.82214355468749913e-3 | 9.55535888671874822 |
| poly-c3-p4 / WrongWidth | 3.71093749999998335e-3 | 9.27734374999995737 |
| poly-c10-p16 / LostTail | 3.56250000000000013e-3 | 8.90625 |
| poly-four-rate / Sgd | 3.51562499999999445e-3 | **8.78906249999998579** |

The minimum over all57 required Signal NI witnesses, including both layers' separate first-update W no-op requirements, is the last row: update0, output W[3,0]. Signal I minimum remains351.56249999999943. No later endpoint or changed witness was selected to repair a first-affected failure.

All17,920 logged scalar values independently round-trip to their recorded bits and are finite; all3,024 frozen b/Vb entries retain exact positive-zero bits. Snapshot/range identity carries the previously reviewed prerequisites forward. Baseline max hidden activation remains0.0017127843654364018 for Original and0.3288487366949463 for Signal, both <=.75; Signal max preactivation remains.5. Mutant ranges remain diagnostics. No encrypted range/precision result is inferred.

## Minor provenance finding and smallest correction

**Low severity, publication correction required:** `task-54-p4-pure-cap-revision-validation/extract_evidence.py:36` writes a copied comment, `# NONSCIENTIFIC_TEST_EVIDENCE extracted verbatim fields from sole invocation08`, into all-margins.tsv, failed-margins.tsv, all-controls.tsv and all-hidden-ranges.tsv. In v2 the actual input at extractor line5 is03-single-numerical.stdout.log. Invocation08 belongs to v1. The data are correct and independently matched, but the new artifact's own source label is inaccurate and can confuse provenance when shared apart from its directory/report.

Smallest correction: preserve the initial v2 manifest/report/extractor and four headers/TSVs with their current identities; change only the extractor's output provenance comment to name v2 invocation03 (prefer the exact03-single-numerical.stdout.log filename), re-extract the already retained raw log, verify every TSV non-comment byte remains identical, and repin affected derived artifacts/manifest/report references. No Rust source edit, build, Engine/forward call or numerical rerun is needed or justified. Root has accepted this limited ledger correction before publication; it remains pending in the record reviewed here. A follow-up ledger-only check can close it without reopening the source or numerical gate.

## Design §11 and authority boundary

`docs/study-diagonal-admission-design.md` §11 accurately preserves both v1 FAIL results,57 I/NI witnesses, five Signal NI misses, measured minimum and baseline activation result from the v1 review. Its wording treats d/8≈4.39453125e-4 and the8.789x consequence as the basis for a **root-selected stricter PRE-HE cap**, retaining I1e-5 and separation8. This agrees with the v1 review's mathematical assessment and the sensitivity disposition. It does not misrepresent the original result as PASS or claim the earlier inference already proved HE precision. The retained section10 describes the original criterion; §11 explicitly records its separately versioned replacement.

The design requires a versioned rerun before adoption; that separately released v2 pure run now supplies the measured Signal PASS, subject to the provenance-header correction before publication. The design expressly rejects weakening separation, silently changing fixture/rates, selecting tolerance after HE observations or converting later failures through cap loosening. Its final paragraph states no HE4e-4 feasibility result and leaves scientific presets, accuracy criteria, P3/P4/D7 authority and campaign holds unchanged. No wording correction to §11 is required by this review.

Source/static correctness and revised pure sensitivity pass in their limited scope. Actual HE parity/error and wrong-control separation at the frozen cap, P4 client/session/factory/poison/count/metadata behavior, P3 real consuming pre-key boundary, P5 blocks, D7 production admission/durable start, secure range/precision/resources and all scientific campaigns remain outstanding. Neither this review nor the revised pure PASS authorizes those operations or future cap loosening.
