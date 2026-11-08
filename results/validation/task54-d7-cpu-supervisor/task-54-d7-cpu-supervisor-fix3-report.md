# Task54 D7 CPU Step2 — FIX round 3 of 5

Status: **DONE — test-only integration correction frozen for scoped rereview.** This does not adopt Step2, prove native16/Linux FD behavior, complete Step3 or establish scientific success. No production source, policy, selector, runtime, dependency, cap or gate changed. No commit/push, server/network/native16/GPU/HE execution, installation, publication, billing action or subagent was used.

Read [FIX3 ruling](task-54-d7-cpu-supervisor-fix3-ruling.md) and the actual [root regression command](task-54-step2-root-integration-check/command.json), [stdout](task-54-step2-root-integration-check/stdout), [stderr](task-54-step2-root-integration-check/stderr), [terminal result](task-54-step2-root-integration-check/result.json). The root full-crate regression exited101 in270.0955seconds:33 summaries total509passed/1failed/6ignored. The sole failure at tests/study_train.rs:75 was a genuine executed assertion: actual `outside observed CPU slice`, expected old generic `study runtime/context/registration unavailable or inconsistent`. This original behavioral RED is preserved; it was not recreated or relabeled a passing full regression. Public preview/CI is separate from these uncommitted sources.

## Minimal correction and retained checks

Changed only `crates/ckks-train-bench/tests/study_train.rs`: replace that one obsolete expected stderr with exact `outside observed CPU slice`, and clarify that valid() constructs parseable legacy requests outside the observed CPU slice. Nonzero exit, empty stdout, exact stderr equality and zero files in the owned temporary directory after each frontend invocation are unchanged. The empty/unknown-option cases and facade argv0/no-context negatives are unchanged. No broad any-error match or accepted profile/runtime expansion was introduced.

Actual entry path is bin main → study_run::run → bounded parse → observed_work's first closed-slice require, before supervisor launch/context/registry/input/parameters/pool. Default fixture has cpu-serial but profile iris__mlp2020, which fails the required iris__emlp1. Parallel fixture has cpu-rayon/workers2; cuda implies parallel and selects cuda-host/workers2/device0. Those also fail SerialCpu/backend requirements. All therefore produce the same precise early diagnostic independently of target OS. This was verified by executing all three feature branches locally.

No additional supported-shape origin fixture was needed for this historical unsupported-request regression. Existing `operational_train_entry_compiles_only_exact_slice_work` covers construction of the supported fixed slice and rejects the old profile/runtime/override shapes; root's existing full regression lists it as passed. That is pure request construction, not live-origin/native execution. The corrected integration test still does not prove supported-context runtime refusal or a genuine operational launch, and no such claim is added. The original closed slice and later evidence gates remain intact.

## Frozen artifacts

Before edits, all71 FIX2 authored sources plus the existing integration test, review metadata and root regression logs were saved in [fix3-original.tar.gz](task-54-d7-cpu-supervisor-fix3-original.tar.gz):194files,4901645bytes,SHA256 `74475095e5e5979b0a8110904092ba0151045272a9f2cbc7ddb267b2d1b56188`. [Archive pin](task-54-d7-cpu-supervisor-fix3-original-pin.json). Original/FIX1/FIX2 reports, patches, pins, archives and211 earlier author command directories remain unchanged. Root's regression is separately pinned and archived.

- [FIX3-only diff](task-54-d7-cpu-supervisor-fix3.diff):1file,961bytes,SHA256 `c49de2275fc4a09a7325f8b0dd928f4dea7553bda4f248d83df37dfd8981cbe3`.
- [Current complete authored source/input pins](task-54-d7-cpu-supervisor-fix3-pins.json), [changed path](task-54-d7-cpu-supervisor-fix3-source-paths.txt), [rereview pins](task-54-d7-cpu-supervisor-fix3-review-pins.json).
- [Complete focused command/log index](task-54-d7-cpu-supervisor-fix3-commands.md), [machine-readable exact argv/exit/hash records](task-54-d7-cpu-supervisor-fix3-commands.json).
- [Pre-edit reasoning and finite work inventory](task-54-d7-cpu-supervisor-fix3-notes.md).

Source BASE remains a29dcf6613ed05d5e1ec7b5300458e74214b5ee9; execution HEAD7578673e070cebc9dc1a1b6309aaa99012c43f98 is root's documentation commit. Authored path set now72files because the already tracked integration test becomes part of this task's authored set. Selected-if-tracked remains453/512; that count is not physical inventory/clean-build proof. All71 previously frozen authored source files are byte-identical to FIX2.

## Focused results

All Cargo commands used offline/locked mode and CARGO_BUILD_JOBS2. Each integration command executes only the two tests in study_train: three bounded refusal-only frontend child invocations, plus two direct facade error calls. Actual legacy argument counts are28/30/32 for default/parallel/cuda; a prose inventory arithmetic overcount was corrected explicitly in notes without changing constructors/work. Parallel command briefly waited for the default command's Cargo build lock; both original handles reached terminal exit, with no cancellation or restart.

| Command directory | Exit | Scope/result |
|---|---:|---|
| 20261108T015504.538437Z-fix3-study-train-default | 0 | default,2passed |
| 20261108T015530.064465Z-fix3-study-train-parallel | 0 | parallel,2passed |
| 20261108T015648.272471Z-fix3-study-train-cuda | 0 | cuda (includes parallel),2passed; no GPU path entered |
| 20261108T015740.738930Z-fix3-clippy-mac-focused | 0 | all-features --test study_train, -D warnings |
| 20261108T015803.775022Z-fix3-clippy-linux-focused | 0 | x86_64-unknown-linux-gnu all-features --test study_train, -D warnings; compile-only |
| 20261108T015741.890617Z-fix3-format-check | 0 | fmt --all --check |

There was no new failing correction command and no redundant full-crate rerun. The original root failure remains the RED evidence; the three focused feature runs establish the corrected integration expectation. Linux integration/descriptor/native16 tests were not executed here. Earlier FIX1/FIX2 source passes and kernel-evidence gaps remain unchanged. Step3 retained roots/readers/gates and all registered scientific matrix follow-on work remain required. Final isolated patch/pin check is recorded in the addendum below and full command index.


### Final verification addendum

`20261108T015911.035845Z-fix3-patch-check-final` exited0. All71 archived FIX2 authored source hashes plus the existing integration test's prior selected-input hash matched. The one-file correction passed whitespace/applicability checks, was applied only in an isolated temporary directory, and all72 resulting authored source pins matched. No main index/source mutation was used by this check.

FIX3 has7 unique terminal commands and0 nonzero exits. The source is frozen for scoped review; the preserved root full-regression exit101 remains the behavioral RED, not an omitted failure.
