# Task54 CPU wire FIX1 scoped independent re-review

SPEC: **COMPLIANT for the corrected Step1 source under the explicit CW-06 root disposition.**

QUALITY: **APPROVED.**

Open corrective finding IDs: **none**. CW-06 remains a **historically unmet process requirement**, explicitly disclosed and root-disposed; it is not retroactively satisfied or repaired. This approval covers only the frozen FIX1 supplied-record implementation and its scoped correction evidence, not runtime/native/scientific acceptance or Task54 completion.

## Scope and inputs

Reviewed CW-01–CW-07 and new breakage introduced by FIX1 only, against the preserved previously reviewed source. Read the FIX1 release and brief, complete 8905-byte report, pins, complete 25907-byte code/test companion, independent 163-line fixture writer, fixture inventory/manifest, root byte-custody result, Iris schedule derivation, both retained pre-correction test snapshots, and all named correction/regression/lint/format logs. No fixture writer, test, Cargo, native/helper/HE/dataset process, network operation, subagent, source edit, index edit or HEAD change was executed. This review report is the only output file written.

Independently checked the frozen artifact hashes:

- Report: `8b8596d3b21d5b1503d85cab0046dbf20134c73f6c3dc4e928455f9c40db9e00`.
- Pins: `43bee70ab13071e85da9bbded3f6ff773a2240fdfac57acba1074298dd6ff0d0`.
- Code/test companion: `1c7801487de7831959b3773b60ac34430a935366e88c030f71401d01082cd320`.
- Full FIX1 diff: `018f2a97cc834e3747972c968abcc25729d379929abd1d8418a6faed4df7944c`.

The companion omits only the identified large new literal JSON hunk. Its fixture semantics were assessed from the independent writer and tests. Root's all56 file-pin verification and forward immutable-base reconstruction of all137 expected members are retained custody evidence, not an independent semantic test performed by this reviewer. `root-cpu-wire-fix1-byte-custody.json:3-11` records 5809880 expanded bytes, maximum168536, compact2626323, every name/length/hash comparison, cycle/bounds/duplicate rejection and original78 preservation.

One named unchanged-code check was necessary for CW-01: whether the newly composed generic Full parsers actually validate their headers/integrity and whether `check_start_claims` binds the start to the supplied admission. Read `crates/ckks-train-bench/src/study_reference/document/admission.rs:522-682` and `case_provenance.rs:17-25,73-152`. These unchanged functions provide the existing parser/link/provenance validation the new helpers rely on. No broad audit or reread of unrelated runtime code was performed. All changed functions were reviewed from the complete companion hunks.

## Individual findings

### CW-01 — ADDRESSED

`crates/ckks-train-bench/src/study_reference/document/evidence/observed/terminal.rs:173-261` now contains an inert `FullTerminal::check`. It rejects a Quick bundle, uses the existing Production or cfg(test) generic admission/start parsers, validates the new gate tuple and evidence hashes, and calls unchanged `admission::check_start_claims` for the actual supplied admission link. The new `Admission::check_observed_claims` at `document/admission.rs:962-999` compares campaign/profile/attempt/case, lock/reference/predecessor, execution, contract/receipt identities and lengths, materialized input and approval to the selected bundle. `case_provenance.rs:157-173` compares captured source revision/manifest, build receipt and executable claims; both admission and start use this check.

The Full path then requires complete Full result scope/status, matching result provenance/evidence/consumed plan, exact start/admission bytes, complete end attempt/counts, all three result outputs and export identity, and the existing supplied complete/resolved supervisor closure. `terminal.rs:91-98` adds result target-lock whole-file SHA and threshold-bit equality to the selected Full branch; complete Full result validation already requires a target. `:218,250-260` closes the supplied clock order as well.

This is the missing pure composition, not a replacement live acceptance algorithm. Target FileRef matching is to the branch's supplied whole-file hash; actual retained target bytes, original owner, current/prior Retry target checks and durable origins remain later-stage duties. The generic Full parser/encoder byte formats are unchanged.

`cpu_wire_fix1_tests.rs:200-216` independently compares both I/NI generic admission/start canonical bytes and invokes their original link checker. `:243-275` checks two complete 400-endpoint Full positive chains, twelve independently rebuilt negatives with exact error paths, plus Full-consumer/Quick-bundle refusal. The negative frontiers cover tuple, attempt, admission-start link, target lock, threshold, source, case, output, evidence, closure, early clock and scope. `check_three_outputs` succeeds first for every negative, so a stale JSON/binary hash cannot account for those refusals. The target-threshold fixture uses a different still-reaching threshold, preserving the internally consistent target outcome until the intended branch comparison.

Evidence: `d7-cpu-wire-fix1-logs/cw01-api-red.log:2-16` is a genuine missing-FullTerminal RED (exit101). `cw01_frozen_full_generic_components-initial.log:4-9` passes the unchanged generic readers/canonical expectations. `cw01_full_supplied_terminal_chain-green-attempt.log:5-10` is 1/1 GREEN, exit0. Together with source inspection, this addresses the original Full supplied-link omission without crediting any actual Full run.

### CW-02 — ADDRESSED

`observed/quick_v2.rs:229-248` now requires the selected branch to be Quick, exact preflight identity bytes and hash equality with that branch, and equality of source manifest/revision, approval, train receipt and executable with Common. Existing event-to-event links, gate checks, registration/common checks and clock checks remain in place. Exact identity equality also binds its attempt/request/consumed-plan fields rather than relying on the narrower Common projection.

`cpu_wire_fix1_tests.rs:148-164` tests four consistently rehashed negative chains, independently authored at `task-54-d7-cpu-wire-fix1-fixtures.py:63-73`. Their original bundle/gate hashes remain fixed. The retained initial log explicitly reports all four accepted variants (`attempt`, `approval`, `receipt`, `executable`) before correction, at `cw02_quick_exact_preflight_bundle_binding-initial.log:7-8`, exit101. The corresponding `-green-attempt.log:4-9` is 1/1, exit0. These counterexamples reach the omitted parity relations rather than failing an unrelated digest check. Captured argv and genuine current authority acquisition are correctly not claimed here.

### CW-03 — ADDRESSED

`observed/requested.rs:247-250,266-269` documents the selected reconstruction seed and constrains roots4,5,6 to selected0. Root3's legal `{0,S}` relation remains unchanged. `cpu_wire_fix1_tests.rs:16-37` uses unchanged W09/W10, checks both prohibited selected1 children, then constructs and accepts root3 seeds `[0,1]` with selected1.

The original first-failure RED is retained. The later pre-fix aggregate RED records accepted roots `[4,5]`, rather than stopping at only root4 (`cw03_preparation_child_selected_seed-all-red.log:7-8`, exit101). `cw03_preparation_child_selected_seed-green.log:5-10` is 1/1 GREEN, exit0. This closes the pure seed relation; it does not discharge actual same-owner association/N06.

### CW-04 — ADDRESSED

`observed/result.rs:438-459` now validates Full endpoint index k as exactly updates6*k, samples120*k and crossed `[k]`. k is bounded by the existing 400-entry check, so the new multiplications cannot overflow. The check applies to incomplete/failed prefixes as well as complete Full records. Quick behavior is unchanged. The existing completion requirement still requires all400 and final2400. Missing/failed endpoint production is not synthesized.

This matches the adopted fixed Iris slice and the source-derived schedule in `root-cpu-wire-iris-schedule-derivation.md:5-9`; it does not generalize the constants to another profile or replace the actual plan binding later. `cpu_wire_fix1_tests.rs:40-82` accepts `(6,120,[1])`, rejects unscheduled1, skipped first12, foreign crossed `[2]` and missing crossed `[]`, with target coordinates changed consistently. `cw04_registered_full_checkpoint_prefix-all-red.log:6-7` records all four negatives accepted before correction (exit101); the corresponding `-green.log:4-9` passes. The old typed R-WIRE-3 positive now supplies crossed `[1]` (`cpu_wire_tests.rs:412`); no original literal is rewritten.

### CW-05 — ADDRESSED

`observed/terminal.rs:29-34` parses the linked Quick start and checks start.started <= result.ended. The new Full path checks the same relation at `:250`. Both retain result <= worker end <= supervisor end, using the existing supplied wall/monotonic comparison.

The independently rebuilt Quick clock fixture changes result-ended to0/0 while preserving its binary/JSON/export and downstream links (`fix1-fixtures.py:75-78`). The test at `cpu_wire_fix1_tests.rs:166-198` first verifies three-output reconstruction, Quick event-chain checks and result/bundle consistency. The initial log records terminal acceptance as the failed assertion (`cw05_quick_start_before_result_clock-initial.log:6-7`, exit101); the corresponding `-green-attempt.log:4-9` is 1/1, exit0. Full's twelve-negative test additionally requires the early-clock variant to fail at `quick.clock_order`. This proves the supplied inequality only, not actual clock provenance or measured elapsed time.

### CW-06 — ADDRESSED as disclosure and explicit disposition; historical ordering remains NOT SATISFIED

The original25 literals preceded production code; the extension53 followed the corresponding first encoder drafts. The missing complete intermediate production-source and initial-six-test snapshots remain missing. FIX1 neither repairs this history nor establishes that the53 expectations were uninfluenced conceptually by the drafts.

Root made an explicit disposition before correction, at `task-54-d7-cpu-wire-fix1-brief.md:7-9`: preserve the historical shortfall and original evidence, permit conditional repair/review using independent grammar review plus genuine pre-fix counterexamples, require all concrete omissions fixed and scoped independent review, and retain the limitation at publication. This is not an implied waiver from passing tests. The original finding and inaccurate historical inventory remain immutable evidence.

For the correction itself, the two retained test snapshots make the counterexamples inspectable: `d7-cpu-wire-fix1-logs/cw03-cw04-tests-before-production.rs:6-16` contains both seed cases, the legal nonzero-selected control, and all four schedule negatives; `all-fix1-tests-before-cw01-cw02-cw05-production.rs:32-54` contains the coherent Quick/context/clock tests and Full generic/terminal assertions. Their assertion semantics survive in the final formatted test file. The genuine assertion REDs and missing-API RED precede the corresponding correction per the retained handoff. The independent writer reconstructs the137 values directly from explicit grammar and known inputs, not production encoder outputs. The new generic Full canonical-byte comparison passed using the unchanged generic reader before FullTerminal existed.

This evidence supports adoption of the **corrected source slice under the recorded root disposition**, with the historical limitation carried forward. It does not justify “all78 were pre-encoder”, “universal original TDD compliance”, or a retrospective repair of CW-06. The disposition remains the reason that CW-06 is not an open code-fix blocker in this re-review.

### CW-07 — ADDRESSED

`task-54-d7-cpu-wire-fix1-test-inventory.md:20-22` expressly withdraws the old nonexistent Full-consumer claim and the overstated individual capacity/threshold/role/security/batch/link test coverage. `:38-44` accurately describes the six new filters, exact cases, the old typed-control change and the fifteen affected existing tests. Original inaccurate statements are preserved as historical evidence rather than silently rewritten. No newly claimed isolated role/security/batch or live sampler/capacity tests appear in the corrected accounting.

## New-breakage and fixture-quality assessment

No new blocking defect or new Minor finding was found in the FIX1 scope. The changes reuse the generic admission/start traversal and existing FileRef/output/supervisor/clock helpers; the new helpers expose supplied claims privately without constructing a capability. `CpuResult::check_bundle` now revalidates the typed result at `terminal.rs:68`, so its added target comparison does not become a standalone bypass of result invariants. No dependency, feature, production cap, unsafe block, native runtime or helper change appears in the FIX1 source diff.

The test-only patch loader (`cpu_wire_fix1_tests.rs:84-107`) reconstructs pinned bytes; it contains no alternative admission/terminal/schedule algorithm. Root independently checked the static DAG and patches using forward immutable-base slicing, a different operation from the loader's reverse splices. The writer's `store` function constructs expected values first and only then compacts differences. Its Full grammar and negative variants are explicit at `fix1-fixtures.py:80-159`, including independently rebuilt admission/start/result/end hashes. The larger fixture allowance is explicitly authorized and inventoried before generation (`fix1-brief.md:17`, `fix1-test-inventory.md:14,26-36`), while production limits remain fixed. Complete400-element byte fixtures are justified here; they are not400 evaluations or dataset/HE constructions.

## Verification evidence and limits

No checks were rerun. Retained logs show six distinct new pure filters passed, plus13 existing CPU-wire tests and two historical Full/evidence tests:21 distinct pure test IDs. The generic-component initial pass is one of those six, not an additional independently counted repetition. Every covering correction/regression log records exit0; every expected RED records exit101. The four initial concrete acceptance defects and missing Full API are real failures, not deliberately removed code or recreated historical REDs.

`regression-d7_cpu_wire_.log:5-22` records13/13. `regression-d7_admission_frozen_documents_complete_bytes_and_hashes.log:4-9` and `regression-d7_evidence_full_quick_exact_claims_and_nonpromotion.log:4-9` each record1/1. `clippy.log:1-3` is clean successful strict lint output with exit0; `fmt-check.log:1` records exit0. The frozen inventory/report specify the exact scoped commands and offline/jobs2 settings. These are saved author-run results; this reviewer did not execute them.

## Out-of-scope observations and retained boundaries

No new out-of-scope defect was pursued. Previously recorded source-manifest capacity/compaction and actual finalized build receipt work remain separate Task54 prerequisites, not validated by these tests. Actual supervisor/broker/flock/pidfd/sampling/OS closure, retained original-owner Reference/Target acquisition, current and prior target authority, genuine gates/start, actual model/FP32/timing, durable publication, runner/table integration, native controls and required full coverage remain Steps2–4 and their later release gates. Supplied complete Full/Quick success proves none of them.

The scoped code corrections and evidence are acceptable for root source-adoption consideration. Publication must retain CW-06's historical limitation and the distinction between this pure source approval and all remaining runtime/native/scientific obligations.
