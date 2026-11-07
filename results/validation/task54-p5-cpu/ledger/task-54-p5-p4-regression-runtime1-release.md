# P5 shared-code regression: one P4 CPU contract execution

Root has read independent P5 source review0c4cb590 in full: SPEC PASS and QUALITY PASS with two deferred minor endpoint/accounting notes that do not affect this unchanged P4 test. Current source report has an artifact-only clarification, with original9ec63bc2 prefix preserved and all8source pins unchanged. Exact current source closure is bound by the adjacent pins JSON.

Justification: P5 extracted P4 stage-state comparators and the owned key-material factory and added inactive P5-local hooks. The earlier P4 runtime result tested the pre-extraction source, so one scoped regression is necessary. Run the existing unchanged P4 contract once BEFORE considering P5 HE. This is not another pure-sensitivity sweep or an automatic retry. No source change or new matrix/threshold is proposed.

Release exactly ONE invocation:
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 CARGO_TARGET_DIR=<existing target> cargo test -p ckks-train-bench --lib study_diagonal::p4_tests::p4_cpu_bridge_contract -- --ignored --exact --nocapture --test-threads=1

Root sole Cargo owner; freeze all bound sources for the entire run, free disk>=805306368bytes, existing target, no features/release/workspace changes, no warmup/other tests/retry. Preserve exclusive launch/process/raw/terminal files, actual command/env/source hashes before/after, same tool/process handle to terminal. Observation timeout is not restart permission. Failure stops the sequence for diagnosis; P5 is NOT released by this file.

Unchanged per-run P4 matrix:20synthetic plans/sessions,20geometry/20initial/34update/0handoff within28/28/42/0planner caps,18Signal oracles/32actual Engine updates,34trainer attempts (32success+2intentional failures),2params/2keys/1proof/1descriptor/1Bootstrapper/22Encoders, <=85realboots. Fixed I1e-5/NI4e-4 applies only to this P4 regression; P5 prospective NI3e-5 remains separate and unchanged. One positive checked factory spy/16negative preparations,54state captures/268unchanged assertions,322configuration collections/68clientinspection endpoints/122model-snapshot copies/52BatchSpecprobes/36badlabel orcardinality requests as the unchanged P4 source specifies. The existing2-call cheap explicit-reference policy regression remains in that source.

Accept only actual selected1test/1pass/0ignored/exit0, unchanged pins and final32success/34attempts/32F64updates plus mandatory guard assertions; retain raw actual error maxima/counts. Record this run's counts separately from earlier P4 failed pre-key and successful71boot runs, and from P5 pure312train122Engine358forward. New per-run work is not zero because tests were once run. Cumulative valid P4 runtime2+thisrun, if completed, would be64success/68attempts/64F64updates, plus earlier failed pre-key preparation; measured boots must be summed from actual raw outputs, not assumed71 again. No scientific accuracy/performance or secure-profile/GPU/D7/fullTask54 acceptance.

No SSH/GPU/dataset/fullcampaign/publicrelease. Root may proceed to a separate P5 release only after reviewing actual regression outcome and its evidence; this file grants no such subsequent invocation.
