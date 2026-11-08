# Step2 FIX round3/5 — existing frontend integration mismatch

Root full bench crate regression (same frozen FIX2 source) terminated exit101,
270.0955seconds:509 passed/1failed/6ignored across33 summaries. Actual command/logs:
L/task-54-step2-root-integration-check/{command.json,stdout,stderr,result.json}.
Only failed target: tests/study_train.rs,
study_train_invalid_and_ordinary_builds_refuse_without_artifacts:75.
Actual stderr "outside observed CPU slice", expected legacy
"study runtime/context/registration unavailable or inconsistent".
This is new runtime integration evidence, not a failure of the public6f release.

Root read the actual test and current CLI. Its valid() fixture requests
iris__mlp2020 with the legacy cpu-serial/parallel/CUDA variants. The adopted
registered observed CPU slice is iris__emlp1 with its closed runtime/request.
Ruling: preserve precise early unsupported-slice refusal and update the affected
frontend regression contract to match that intentional behavior. Do not weaken
checks to merely any error or remove the no-stdout/no-artifact/nonzero assertions.
Do not expand accepted profiles/backends, bypass pre-context checks, fabricate a
live origin or revert to a generic error merely to satisfy the old string.
Reason: the new source's earlier closed selector check is intended; the old test
still must verify failure without side effects. Alternative changing production
message masks a useful boundary and weakens diagnosis. Cost: platform/feature
expectations must be explicit and kept aligned with the actual accepted route.

Original author owns round3. Snapshot frozen FIX2 source/review metadata and the
existing test before editing; preserve original/FIX1/FIX2 evidence and root full
regression RED. Inspect exact entry path and feature branches before choosing the
minimum test correction; if actual production behavior contradicts the contract,
report the concrete contradiction instead of masking it. Add a bounded supported-
shape negative only if needed to preserve formerly intended ordinary-build refusal
coverage, and inventory its constructors/caps before running it. No native16,
server/process experiment, new environment or dependency is authorized.

Run focused --test study_train default and feature-relevant checks; no broad
whole-crate rerun is needed for a test-only expectation correction after the other
509tests passed, unless production changes or new evidence justify it. Keep new
Linux runtime checks authored/compile-only if not actually run. Freeze FIX3-only
diff, updated full authored paths/pins, command logs and report, then root dispatches
scoped review. Existing source-quality findings Q01/Q02 remain addressed; this is
an integration correction, not a whole Step2 review restart. No commit/push.
