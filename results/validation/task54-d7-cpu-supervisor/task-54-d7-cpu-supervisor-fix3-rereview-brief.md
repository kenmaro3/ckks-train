# Scoped Step2 FIX3 test-only rereview

Repo /Users/marconi/Documents/research/hekit; L is this ledger directory.
Read fix3-ruling.md, fix3-report.md, fix3.diff (all with prefix
 task-54-d7-cpu-supervisor-), then actual tests/study_train.rs and its
bin/study_run::observed_work entry path as necessary. This is one-file regression
correction after whole Step2 SPEC/QUALITY source reviews; no whole-code review.

Actual root regression at frozen FIX2:33 summaries509passed/1failed/6ignored,
exit101. Only failure expected generic context message while the now-closed CPU
slice rejects the legacy iris__mlp2020 request earlier with precise
"outside observed CPU slice". Logs at L/task-54-step2-root-integration-check/.
Root accepted the precise early refusal in the ruling; do not blindly accept
changing an expectation if actual entry behavior/spec contradicts it.

FIX3 changes only tests/study_train.rs: expected exact stderr and explanatory
comment. Nonzero, empty stdout and no artifacts must remain; no production
profile/runtime/gate expansion. Existing request vectors cover default, parallel,
cuda feature branches. Check the evidence and actual branch behavior, preserving
that unsupported-request coverage does not prove successful supported launch.
No need to add fake live origin tests or rerun native/scientific work.

Fullreport7036B SHA59d1f33f5416688f756765688c92e1992069e2ddebd2fae100f136bcf21b7bc4;
diff961B SHAc49de2275fc4a09a7325f8b0dd928f4dea7553bda4f248d83df37dfd8981cbe3.
SourceBASEa29dcf... unchanged; executionHEAD7578673 plusfrozen authoredWIP.
Root verified529source/inputpins,17reviewpins,28commandlogpins. All71FIX2production
files unchanged; authoredset72 includes this existing test. selected453/512.
Default/parallel/cuda focused integration each2passed; focused Mac/Linux Clippy,
fmt and isolatedpatch/all72pins passed. Linux compile is not Linux execution;
fullbench originalfailure is preserved, not falsely relabeled a whole-suite pass.

Review SPEC and QUALITY of this fix only; mark the regression finding ADDRESSED
or NOT ADDRESSED and flag new Important/Critical breakage in this diff. Verify
pins/relevant raw logs. Original FIX2 whole-source quality PASS is retained:
L/task-54-d7-cpu-supervisor-fix2-quality-review.md; no need to restart it.
No edits, builds/tests, network, process experiments, Git mutation, subagents.
Write only L/task-54-d7-cpu-supervisor-fix3-review.md, return path/bytes/SHA256
and concise verdict. Root owns adoption/commit. Native16, Step3 and full matrix
remain pending; fixed publicpreview6fba46c is separate from this WIP.
