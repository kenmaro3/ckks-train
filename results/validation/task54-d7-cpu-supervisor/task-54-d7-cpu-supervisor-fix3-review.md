# Task54 D7 CPU Step2 — scoped FIX3 SPEC / QUALITY rereview

Verdict: **SPEC PASS; QUALITY PASS. Regression finding ADDRESSED.** No new Important or Critical breakage was found in this one-file correction. Root owns adoption/commit. This verdict does not adopt Step2, complete Task54, establish scientific success, or replace the retained FIX2 whole-source quality review.

## Scope and identity

Read the FIX3 brief first, then ruling, complete report/diff, actual frontend entry and parser/closed selector branch, changed integration assertions, relevant existing request-construction test, raw command/result/stdout/stderr evidence and pins. Applied engineering:code-review dimensions to this bounded diff. No whole-source review was restarted. Reviewer actions were reads, hashing and in-memory archive/diff comparisons, followed only by writing this report. No builds/tests, network, process experiments, Git mutation or subagents were run.

Source BASE is a29dcf6613ed05d5e1ec7b5300458e74214b5ee9; execution HEAD recorded in the frozen pins is 7578673e070cebc9dc1a1b6309aaa99012c43f98 plus authored WIP. Public preview 6fba46c is separate. Independently verified all 529 source/input pin occurrences, 17 review pins and 28 command-log pins against current bytes/SHA256: zero mismatches. The 7036-byte report matches SHA256 59d1f33f5416688f756765688c92e1992069e2ddebd2fae100f136bcf21b7bc4; the 961-byte diff matches c49de2275fc4a09a7325f8b0dd928f4dea7553bda4f248d83df37dfd8981cbe3.

Read the 194-member original archive without extracting it. The unified diff reconstructed in memory from its test bytes and current test is exactly the frozen FIX3 diff. All 71 other authored sources are byte-identical to that FIX2 snapshot. The authored set is 72; selected 453/512 is selection metadata, not physical inventory or clean-build proof.

## Regression disposition — ADDRESSED

The only code change is crates/ckks-train-bench/tests/study_train.rs:3 (explanatory comment) and :64 (precise expected stderr). This is a contract correction, not masking a production contradiction.

Actual bin main first offers arguments to native_probe_entry; study_supervisor::probe_entry at :100–105 ignores these request vectors because they do not begin with a native probe flag. It then calls study_run::run. The ordinary request is parsed, then observed_work is called at study_run.rs:375, before supervisor::launch or worker/context dispatch. The first require at :331–340 demands SerialCpu, Family/Full/CPU, iris__emlp1, seed0 and batched_diagonal. Every fixture requests iris__mlp2020 and therefore fails with exactly "outside observed CPU slice". Default selects cpu-serial; parallel selects cpu-rayon/workers2; cuda selects cuda-host/workers2/device0. Cargo.toml:31 makes cuda imply parallel, so the CUDA fixture has the required workers option and reaches the same selector refusal. Parser runtime branches accept these shapes before the closed observed check. No target-OS conditional precedes that refusal on this ordinary path.

At tests/study_train.rs:71–74 the assertions remain nonzero exit, empty stdout, exact diagnostic equality after trim, and no entries in the owned temporary working directory after each child invocation. The unchanged empty/unknown-option cases still require their exact respective diagnostics. The two direct facade negatives at :79–83 remain. No any-error replacement, skip, ignored test, accepted-profile expansion or production behavior change was introduced. Security/performance/correctness/maintainability review of this diff found no actionable issue; the explanatory comment usefully identifies the fixture as a parseable unsupported request.

## Evidence and limits

Independently read the preserved root command and terminal result and inspected the failing assertion in stdout: exit101, 270.09552187498775 seconds, actual "outside observed CPU slice" versus obsolete generic expected message. Recounted all 33 summaries: 509 passed, 1 failed, 6 ignored. This original full-regression RED remains intact; it is not a whole-suite FIX3 pass.

Read all seven FIX3 commands and their raw logs: default, parallel and cuda focused study_train runs each show 2 passed / 0 failed; focused Mac and Linux all-feature Clippy, fmt, and isolated patch verification each terminated exit0. The parallel build-lock wait is preserved and ends normally. Patch-check output reports 71 prior sources plus existing test verified and all 72 resulting pins matching; the independent archive/diff comparison above confirms the one-file boundary without replaying the patch command.

The existing study_run::tests::operational_train_entry_compiles_only_exact_slice_work checks supported fixed request construction and rejects legacy/override shapes; its pass is present in root stdout. It is not supported-origin launch evidence. The corrected integration test proves unsupported-request refusal and its asserted local side-effect boundary only; it does not prove supported-context refusal, successful supervisor launch, Linux descriptor behavior, GPU execution or scientific execution. Linux Clippy is compile-only. Native16, Step3 retained roots/readers/gates and the full registered matrix remain pending. No additional fake live-origin fixture is needed to resolve this bounded legacy-request mismatch.
