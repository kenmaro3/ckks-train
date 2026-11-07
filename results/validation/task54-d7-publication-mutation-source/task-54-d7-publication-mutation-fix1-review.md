# Publication mutation FIX1 — scoped independent review

2026-11-08. **SPEC: PASS. QUALITY: PASS. Important I1: ADDRESSED in source.** No new blocking finding in this scoped diff. Default Linux selection no longer reaches the scratch-dependent anonymous-publication tests. This is source/static re-review, not an executed CI/native result or a new runtime release.

The original source review `42e2b711297904512d345dbf00e7492826461ae855edc7411c32490f5b74e2cc` is preserved. Root I1 correctly identified a default-CI integration defect that review missed: `.github/workflows/ci.yml` runs `cargo test --workspace` without the required scratch variable, while the original Linux native tests were unignored. The historical Mac25 outcome could not detect Linux cfg selection. The original PASS must not be interpreted as proof that the pre-FIX1 source worked in default Linux CI.

## Scope and pinned inputs

Read root I1, the original source release, complete11,240-byte scoped diff, author FIX1 append and complete successor inventory. Inspected the actual default CI command, every scratch-dependent entrypoint and transitive child/helper route, unchanged guard-only pair, final source attributes and affected child bodies. Independently checked final pins, current filter locations/attributes, Mac function-body continuity, eight static log hashes and historical preservation with the administrative exception below. No source/Cargo/test/--list/native/runtime/SSH/subagent action or supplied audit-script rerun occurred. This review is the sole write.

All local ledger names below are relative to `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.

| Input | SHA256 |
|---|---|
| `task-54-d7-publication-mutation-root-finding-I1.md` | `239a7d223e05d2270e9afaa3751d51d1c1cb15cdd1604afbdb4df37db894421a` |
| Original source release | `a2608546b37d1faa3ac4c47fd74630fdf982198a07cb1c3eda73788b5651d10f` |
| Amended source report | `068e04ea6c25486e75b4f332b5f724990c1ec2db2db18b37777ba4810c3179d0` |
| `publication-mutation-fix1/scoped-source.diff` | `3a465b75a4f465ac756c7eb6c57e759e7d210dba114b9e7fbb895072f13e390e` |
| `publication-mutation-fix1/fix1-inventory.md` | `30642241afa6fbd4d09c11af826bd0400044a8676de573d01166a799790e8af5` |
| `publication-mutation-fix1/exact-filter-inventory.json` | `932775ed9a15351f240db22ad5c0cf4ae16c0bddb5df30bfa55b8deaa1b99b70` |
| `publication-mutation-fix1/mac25-applicability.json` | `583ed0d586e2388af44d53d9ea94990d9efb66e31c52927686071a8c352f820c` |
| `publication-mutation-fix1/source-change-inventory.json` | `6ccc47d18d358f81959316e328cfe862efcaa8c626bb785ee0ea3828179fb814` |
| `publication-mutation-fix1/final-source-pins.json` | `fc3586cc9b7c9c1b5b420fefe851579463a42442e36e84e096155cea5e0dc94e` |
| `publication-mutation-fix1/validation-log-index.json` | `6d19b3e5f9f406588555c33a45a3e10dba6c773f6beec76fd80a3978bd7b4780` |
| `publication-mutation-fix1/historical-input-pins.json` | `8b070c8ffaade62d3b4edaa9bf5ab1644034dd325a7e7a179e4672452c6f3c49` |
| `.github/workflows/ci.yml` (repository-relative) | `3223532b9f0ddac52bd1b92fe1cf077548c05bcd482e96987b361e2d54e19507` |

Exactly two final test files change:

- `crates/ckks-train-bench/src/study_provenance/publication_tests.rs`: `94b2b128993d7276a670df2c67fe896e2e6a5195f262f359e3db4d61b5e59ff1`.
- `crates/ckks-train-bench/src/study_provenance/publication_anonymous_tests.rs`: `089af33312b710057676e239d9b27cf98c9a1c416a3225c1cd9fc9f8892d3b5f`.

The other five pinned files—Cargo.lock, bench Cargo.toml, publication.rs, publication_anonymous.rs and publication_test_hooks.rs—match their pre-FIX1 pins exactly. No production protocol, dependency edge/features, limits, payload fixtures or workflow change is hidden in this remedy.

## I1 selection and transitive reachability

All48 test functions remain. Exactly23 have reasoned ignore attributes:18 new anonymous-module functions and5 migrated publication-module functions. They comprise21 parent entrypoints and2 child-only entrypoints. Every direct `SyntheticDirectory::native` user is ignored. The entire17-test Linux child module and the separate Linux anonymous-success test are opted in; in the parent file, engine/actor/thread/process publication tests and their publication child are opted in.

The private `synthetic_guarded_publish` helper is reachable only from the opted-in thread publisher and publication child. The private `run_child` helper is reachable only from the opted-in drop/visible-uncertain parents. No ordinary default-selected test reaches anonymous Native creation through those helpers. The non-Linux refusal test remains default-selected and uses portable `SyntheticDirectory::new`; its production create path still refuses Linux-only behavior.

The25 unignored source functions are not a single-platform runtime total: by source cfg,24 are default-selected on Linux and23 on non-Linux within these two files. Three unchanged outer Target tests are outside those counts. This agrees with the successor inventory and is not a --list result. Ordinary pure/portable tests were not broadly ignored or deleted. Default `cargo test --workspace` does not select the23 ignored entries even if a scratch variable is incidentally inherited. This addresses the concrete missing-environment panic and accidental unauthorized native selection without installing CI scratch, weakening mount refusal or silently returning success.

The existing `target_native_linux_guard_child` / `target_native_linux_separate_process_uses_fresh_flock_description` pair remains unchanged and unignored. It uses `SyntheticDirectory::new`, fresh_guard and its acquired marker; it never calls Native::create or the mount-required scratch constructor. Its parent also checks the marker after child success, so a zero-tests child cannot satisfy the full assertion. This retained guard-only pair is outside I1's scratch-dependent anonymous-publication route. Its existing wait behavior is not newly validated or broadened here.

## Child invocation and outcomes

`publication_anonymous_tests.rs:374–426` changes only the Linux child launch/entry behavior. Parent `run_child` validates the closed before-link/after-link cut before spawning, supplies the exact child filter with `--exact --ignored --test-threads=1`, and requires status70 for before-link or71 for after-link. Child requires both environment values, validates the same closed cut before opening a File, executes real C/W/F and optionally L, then intentionally exits with that cut's code. Panic/unsupported cut/missing environment/signal/exit0 cannot satisfy the parent. A wrong filter yielding zero tests and ordinary exit0 therefore fails explicitly. The existing5-second child timeout/owned kill-and-reap path and child count are unchanged.

`publication_tests.rs:556–608` applies the matching ignored-child flag to the old guarded process publisher. The child now requires its parent fixture environment and reports70 for a publication winner or71 for rejection. Parent requires exactly one of each across its existing two children, then checks exact final payload. A zero-tests exit0, panic or abnormal termination fails those assertions. Code71 alone is only the existing rejection outcome, not a proof of its cause; winner and final checks remain required. No extra process, file, marker, publisher operation or scientific constructor is added. The old two-child test retains its previously inventoried blocking wait and remains a separately released obligation, not part of the21-case minimum's bounded single-child cases.

These nonzero codes are private parent-interpreted control outcomes. They are not top-level libtest passes or durability/live-gate receipts. Future execution must select an exact **parent** with `--exact --ignored`, not run child-only filters directly or broadly select all ignored tests. The actual named parent must still report1 passed/0 failed/0 ignored and retain source/binary/scratch/mount/capability/outcome evidence under a separate release. No invocation was performed here.

## Historical Mac25 applicability and preservation

Independent before/current source comparison confirms all22 selected publication test bodies are byte-identical and retain ordinary selection; the three Target tests' containing file retains SHA256 `7a3c34222b70a9ec8cf02a46033d995c120d7fe18bbdceb0ea96aba6895806e3`. The only changed bodies are the two Linux child functions and their two Linux parent/helper functions. All other source edits are Linux-only ignore attributes. Fake/FiveOps/OwnedModel, portable directory creation/drop, read/recovery/guard/clock helpers and production code used by the historical Mac25 are unchanged.

Thus accepted outcome `8d08d149ccf1dd2f81661474f998c3ced62d830b2cf6953df537936960d7c2b2` remains evidence for its exact pre-FIX1 host/binary/source scope and unchanged selected paths. It is **not** a run of the new whole-file hashes or rebuilt binary, and not Linux default/native evidence. No25-test rerun is claimed.

The amended report preserves its original14,445-byte prefix exactly at SHA256 `07919aaeb3827893646ef70fbabc68337d86f1662cc93e7cd2bafeaa14350fa5`. Original source review42e2b711 and runtime outcome8d08d149 are unchanged. Of176 entries in the author's historical-input pins,174 currently match whole-file bytes, the report matches its original prefix, and one root-owned archive manifest has a separately confirmed later administrative update. Do not repeat the author's handoff-time “175 byte-identical plus report prefix” as a current all-file equality assertion.

Specifically, after FIX1 handoff root extended `task-54-d7-publication-mutation-archive-copies.json` from157 to194 records and redirected the old source-report source pointer to the preserved before-FIX1 copy. Its pinned old raw hash was `f06957f60c1e58b007b55207276e05cd23abcf62706b2fa06a193bff1129194e`; current observed hash is `062b897fc6a130438ec038a77ed043b28fc01a19e32a4244db4486d5b82ba211`. Root confirmed this bookkeeping action and preservation of all original157 archive bytes/hashes; the old raw manifest itself was not retained separately. I independently checked all194 current archive payloads against their recorded hashes. The exact old raw manifest transition cannot be independently reconstructed from its hash alone. This disclosed root administrative delta is not an author preservation error or a source fix gap.

## Static evidence and verdict limits

All eight FIX1 indexed logs match their hashes. `selection-red01.log` records expected exit1 listing the23 absent ignore attributes before editing; it is a source-audit RED, not a failing compiler/native test. Selection GREEN and inventory report exit0. Linux all-targets/all-features strict Clippy and host lib `--no-run` logs show successful compilation/typechecking; formatting/diff logs show0. No supplied script, Cargo command or test was rerun by this reviewer. These records do not establish Linux harness execution, actual supported-mount behavior or full-workspace CI success.

I1 is closed at the source-review boundary. No new Critical/Important/Minor issue was found in this diff. Actual native21 execution, retained guard/fork/mount validation, future15-selector/D7 protocol/custody, real five gates and consuming start/begin remain separately outstanding. C/W/F/L/D, shared recovery, Full/Quick wire schemas/caps, all21 profiles/Full2680+20+7/Quick268, holds, original science/security policies and permissive licenses with Unicode-table-only exception remain unchanged. This scoped PASS grants no new native execution or scientific authority.

Root subsequently froze the reviewed administrative manifest as `task-54-d7-publication-mutation-archive-copies-at-fix1-review.json`, exact SHA256 `062b897fc6a130438ec038a77ed043b28fc01a19e32a4244db4486d5b82ba211`, independently matched here. The explicit root transition record is `task-54-d7-publication-mutation-root-archive-update.json`, SHA256 `f74fb437de7c2940002c8a15ede521da98f2be120aef21329fc9b5ebd5a737f9`. Later archive assembly must not be confused with this review-time snapshot.
