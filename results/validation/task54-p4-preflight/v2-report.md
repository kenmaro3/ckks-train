# Task54 P4 pure cap revision report

2026-11-07. **v2 Signal pure sensitivity gate PASS at I1e-5/NI4e-4; Original remains FAIL.** The separately authorized sole v2 numerical invocation exited0, with68 actual Engine::train calls/candidate,136 additional and272 cumulative pure-preflight calls. The original v1 source/raw/report and both v1 FAIL verdicts remain unchanged. No HE accuracy, capability, scientific result or future execution permission follows.

## Root decision and exact scope

Read/hash-verified root cap revision release d0d4d728dddb7ccde60df23e7a0e737f3c886975225936cc3072fb316becede5 and complete independent v1 review c4f8e63ec30fea4854556f037ce6413d071de80d60402ebc29a60feaea52a777. Root selected NI4e-4 before any HE, from independently reconciled v1 absolute distances. This is a stricter future error requirement; I1e-5,8x separation and every fixture/control/range criterion remain fixed. No cap was loosened.

Source/Cargo ownership was transferred only after P3 fix2 returned. All seven supplied P3/P4 dependency pins were verified before work, including study_admission697f76cd289b626b7167d569561befe9b36375c03baa8971c0f0d7de34c6e561. The only source edit is exactly two constants in study_plain/p4_fixtures.rs:

```diff
--- crates/ckks-train-bench/src/study_plain/p4_fixtures.rs.v1
+++ crates/ckks-train-bench/src/study_plain/p4_fixtures.rs.v2
@@ -4,8 +4,8 @@
 use crate::study_profile::{Activation, Layer, LayerRole};
 use ndarray::Array2;
 
-const VERSION: &str = "task54-p4-pure-v1";
-const CAPS: [(&str, f64); 2] = [("I", 1e-5), ("NI", 5e-4)];
+const VERSION: &str = "task54-p4-pure-v2";
+const CAPS: [(&str, f64); 2] = [("I", 1e-5), ("NI", 4e-4)];
 
 #[derive(Clone, Copy, Debug)]
 enum Candidate {
```

scope-proof.txt verifies that reversing exactly these two replacements reproduces the full prior source. No fixture, control, first-affected index, rate, loss folding, call, shape, count/range criterion or production source changed. The other six source/dependency files match their P3 fix2 pins. Prior RED is the preserved v1 numerical failure; no new RED or no-run loop was invented. Only this new ledger directory/report and the one authorized source file were written.

## Runtime verdicts and counts

| Version/candidate | I failed /57 | NI failed /57 | Actual train calls | Pure gate |
|---|---:|---:|---:|---|
| Preserved v1 Original, NI5e-4 | 51 | 57 | 68 | FAIL |
| Preserved v1 Signal, NI5e-4 | 0 | 5 | 68 | FAIL |
| New v2 Original, NI4e-4 | 51 | 57 | 68 | FAIL |
| New v2 Signal, NI4e-4 | 0 | 0 | 68 | PASS |

Each v2 candidate retains8 baseline families,60 control rows (41 applicable,11 algebraically equivalent,8 inapplicable), and57 required witnesses/backend (41 control max-norm distances +16 separate first-update per-layer W no-op distances). All records were emitted before the final Signal assertion. Each candidate has12 baseline +56 mutant calls; all completed successfully, no early train/forward failure or missing snapshot occurred, and harness_failures=[] for both. No-op/equivalent/inapplicable controls add0 calls. No extra candidates, replays, warmups or numerical retries were used.

The fixed scope is unchanged: six [2,4,w] single-update smoke variants, Square C3/C2/C10 w3/2/10 with mean.06/delta2/C/active b/SGD; Poly C3/C2/C10 w4/2/16 with mean.04/delta1/frozen b/NAG.9; four-step Poly C3/P4 rates.04,.08,.02,.08 with rows12,8,5,12; and first-reference.08,.08 with rows8,8. Logical x/y ordering spans each whole update, including8+4 partitions. Full NAG reference remains.08. Engine/forward are the existing D6b implementation, unchanged. No alternative NN math, public materializer or execution capability exists in this test.

Additional bound consumed:68/candidate,136 total; prior136 +new136 =272 cumulative actual pure calls. All unrelated parameter/proof/descriptor/Encoder/key/evaluator/ciphertext/bootstrap/materializer/planner/HE constructors/operations are0 in this release. No P3 tests or broad runtime suite executed.

## Measured Signal margins at the revised cap

The minimum required first-affected absolute distance remains **3.51562499999999445e-3**, poly-four-rate/Sgd update0, output W[3,0]. Its measured NI normalized distance is **8.78906249999998579e0 >=8**; I distance is351.56249999999943. No coordinate/value/absolute-distance change was needed.

| Former v1 NI miss | Update/layer/coordinate | Identical max absolute distance | v2 NI D | Status |
|---|---|---:|---:|---|
| square-c10/NoUpdatePerLayer | 0/1/[2,0] | 3.58203125000000622e-3 | 8.95507812500001599e0 | PASS |
| square-c10/LostTail | 0/0/[0,1] | 3.82214355468749913e-3 | 9.55535888671874822e0 | PASS |
| poly-c3-p4/WrongWidth | 0/1/[3,0] | 3.71093749999998335e-3 | 9.27734374999995737e0 | PASS |
| poly-c10-p16/LostTail | 0/0/[0,1] | 3.56250000000000013e-3 | 8.90625000000000000e0 | PASS |
| poly-four-rate/Sgd | 0/1/[3,0] | 3.51562499999999445e-3 | 8.78906249999998579e0 | PASS |

| Signal per-layer first-update W no-op minimum | Family/coordinate | Absolute distance | NI D |
|---|---|---:|---:|
| Layer0 | poly-c3-p4/[0,0] | 7.12499999999999245e-3 | 1.78124999999999787e1 |
| Layer1 | square-c10/[2,0] | 3.58203125000000622e-3 | 8.95507812500001599e0 |

All228 v2 margin records, including exact current epsilon bits4556014321273684781 for NI4e-4, are retained in all-margins.tsv and raw stdout. failed-margins.tsv lists all108 Original failed backend/witness rows; Signal has none. No failed Original row was removed or relabeled. All120 control classifications/first-affected labels/call counts are in all-controls.tsv, identical to v1: singleton lost-tail/denominator cases remain inapplicable; delta1 omitted/double folding, C2/P2 width equivalence and Square C10 hardcoded-ten equivalence remain algebraic exclusions with0 calls. They are not claimed as discriminated bugs.

## Exact read-only comparison with v1

compare_v1_v2.py reads only retained logs/TSVs and never calls training/forward. Its complete result is v1-v2-comparison.txt. It verifies:

- All138 CALL_INPUT records are text-identical, including x/y ordering, partitions/shapes/classes, source/mean/delta/kernel/reference/momentum bits.
- All1,760 SNAPSHOT records are text-identical, including every W/b/velocity value and bit.
- All276 RANGE records and all120 CONTROL records are text-identical, including before/after phases, signed extrema, activation/preactivation magnitude and first-affected labels.
- All136 ACTUAL_TRAIN ordinals and the total-call manifest are identical.
- Every I margin is text-identical. Every NI margin differs only in epsilon/bits, normalized distance and resulting status; every baseline/control coordinate/value/bit and absolute witness distance is identical. Exactly five Signal NI statuses change FAIL->PASS.
- The evidence VERSION is the sole manifest difference; candidate identities/counts/harness fields are unchanged. Candidate numerical failures/status reflect the revised cap.
- Every file in the original v1 evidence directory and the original report retains its recorded hash. v1-preservation-before.sha256 and after.sha256 are identical.

No engine/source rerun was performed to make this comparison. Exact input/state/range/distance identity supports attributing the changed Signal verdict solely to root's separately versioned PRE-HE criterion. The outer test terminal changes from v1 failure to v2 success because the final assertion gates Signal; it does not imply Original passed.

## Range, finite state and static checks

| Candidate | Baseline max hidden activation magnitude | Baseline max hidden preactivation magnitude | Activation <=.75 |
|---|---:|---:|---|
| Original | 1.71278436543640181e-03 | 5.37112124049642960e-03 | PASS |
| Signal | 3.28848736694946298e-01 | 5.00000000000000000e-01 | PASS |

All17,920 logged scalar W/b/velocity entries are finite; all3,024 frozen b/bias-velocity entries retain exact positive-zero bits. Range values, hidden activation maximum.3288487366949463 and all other state are identical to v1. Mutant ranges remain diagnostics and do not satisfy baseline prerequisites. No encrypted/AA/bootstrap range or precision result is claimed.

All three released commands used current target, CARGO_INCREMENTAL=0,CARGO_BUILD_JOBS=8,RAYON_NUM_THREADS=8 and locked/offline Cargo. Complete stdout/stderr/status JSON and seven source pins before/after each command are retained:

| Stem | Exact command | Exit |
|---|---|---:|
| 01-clippy | `cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings` | 0 |
| 02-format | `rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_plain/p4_fixtures.rs` | 0 |
| 03-single-numerical | `cargo test --locked --offline -p ckks-train-bench --lib study_plain::p4_fixtures::tests::p4_pure_sensitivity_gate -- --test-threads=1 --nocapture` | 0 |

Minimum observed disk free 67034116096B exceeds805306368B. Clippy compiled but executed no tests. The only v2 numerical test reports1 passed/0 failed/189 filtered, exit0. The stdout/stderr logs retain 990391/263 bytes. No runtime test was repeated.

## Seven-source freeze and return

All seven pins below match before/after every static command and the sole numerical invocation. The source snapshots in before-source/frozen-source/final-source preserve exact paths; only p4_fixtures differs between before and frozen, by the two constants above.

```text
697f76cd289b626b7167d569561befe9b36375c03baa8971c0f0d7de34c6e561  crates/ckks-train-bench/src/study_admission.rs
571f4145239230ab124364a76d0fa16d3b940d5fa8b066a7194f0e1f1d846a0c  crates/ckks-train-bench/tests/study_admission.rs
e1bf9b3e7f856d0eb4a167874d05d6884b38d0054842f583b1add81c6e4eb2b8  crates/ckks-train-ckks/src/bootstrap/metadata.rs
867b2c6d8f4391861956c10483fe18c91636e9272e41cceee8953b90976b3e44  crates/ckks-train-bench/src/lib.rs
d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b  crates/ckks-train-bench/src/study_plain.rs
d4edd2262ff0b3e9c0792047fb5e4016d2b5532849c89a47ad3491d4c6757a68  crates/ckks-train-bench/src/study_plain/p4_fixtures.rs
fcb7f1d34fd8c87527af4adf59c0ebd919eeeb2a07f5f635a965a436c233c9af  crates/ckks-train-bench/src/study_reference.rs
```

freeze-before.sha256 equals freeze-after.sha256; freeze-equality.txt records exact identity and v1 preservation. source-delta.patch/scope-proof.txt show the full authorized delta. MANIFEST.sha256 inventories every evidence artifact except itself; the report hash is returned separately. Source/Cargo ownership was returned to root immediately after post-run frozen evidence capture, with no Cargo process running; subsequent work was ledger-only read-only extraction/comparison and this report.

Implementation/static checks and the revised Signal pure gate pass in this bounded scope; Original remains insensitive and v1 remains a measured failure. Independent scoped re-review of this two-constant delta/new evidence remains root's next review step. P4 client/session/factory boundaries, P3 broader pre-key acceptance, actual HE parity at the stricter cap, counts/metadata/poison behavior, production D7 and scientific acceptance are unresolved. This release authorizes none of those operations and provides no future cap-loosening permission.
