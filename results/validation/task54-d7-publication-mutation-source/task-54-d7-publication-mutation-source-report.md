# Task54 shared publisher mutation source report

Status: **DONE_WITH_CONCERNS — source/static author phase complete; native remediation acceptance pending.** Sole source/Cargo ownership is released to root with this report. No tests, `--list`, native scratch publication, SSH/server command, HE workload, runtime codec suite, commit or push was executed by this author.

The prerequisite migrates the real shared Reference/Diagnostic/Failure/Target publisher. It does not implement the full Task54 runner/tables/matrix or grant any live gate. Parent task remains the exact four-dataset/model/CPU-GPU/interactive-noninteractive/baseline a-b-c campaign, ten seeds, quick subset/one seed/two-hour goal, all21 profiles/full2680+20+7/quick268 and registered holds. No source/runtime/science claim is substituted for those goals.

## Inputs, source and ownership

Read parent `task-54-brief.md` first, full source release, complete ratified proposal/review and root design section9. Source BASE is `2da02ab7c861c7d85ff25c4ee5bf67f4d86a6c72`. During this source assignment root separately committed cleanup archive/docs at `2ce43c2d9d49fc4da564b8ea816fba6bb6dc3273`; no source/Cargo baseline change resulted. Author did not commit/push.

Binding source release SHA256 `a2608546b37d1faa3ac4c47fd74630fdf982198a07cb1c3eda73788b5651d10f`; proposal `714880b778511c3be75d1ac0e6712e77e24d2564ba52f61cdb36f9da288509f3`; independent review `6e70cd0c7e6b253b002be9bb013cd3827d6d6a9c3b1ea23882fef6f8d0d02a13`.

[Baseline pins](publication-mutation-source-author1/baseline-pins.json) freeze271 input/source/dependency/design/golden files before editing. [Comparison](publication-mutation-source-author1/baseline-comparison.json) finds267 byte-identical and only4 authorized baseline changes: publication.rs, publication_tests.rs, bench Cargo.toml and Cargo.lock. Three new source files are added. Canonical/legacy fixtures and all shared outer role/source/context/clock code remain unchanged. `.vscode/` and the two unowned MNIST result JSONs were not written; initial/final status retain them as untracked. No scientific JSON, public document, policy, baseline implementation, dependency license allowance, credential or process outside owned compiler jobs was changed.

Full review diff: [source-review.diff](publication-mutation-source-author1/source-review.diff), including all3 new files (not merely tracked git diff). [Final exact source pins](publication-mutation-source-author1/final-source-pins.json):

| File | SHA256 |
|---|---|
| `Cargo.lock` | `e5059c771e93261c6fecf855f7a2d8ab052021f62cadafddbb23c43fa2f216c3` |
| `crates/ckks-train-bench/Cargo.toml` | `fc698bca60c78aa259180e6cd238514cb9e83bdb780a70e3c0287831b6a438b4` |
| `crates/ckks-train-bench/src/study_provenance/publication.rs` | `69a4731e5695d986702e079d7d900bb171cd602d07db3a76951a0bb378d61f47` |
| `crates/ckks-train-bench/src/study_provenance/publication_tests.rs` | `7f604bebbca0fa5ae47f63bac1fa97231d78c6a5affbd3371f866d5e95bfc39f` |
| `crates/ckks-train-bench/src/study_provenance/publication_anonymous.rs` | `f6e8b4b4938ba3c30741b7d0f9fcc745395250d65abc8e3f4a91898c84c38038` |
| `crates/ckks-train-bench/src/study_provenance/publication_anonymous_tests.rs` | `417b62401fa01ea1d32fc6211b476c259dfcc3f1e90a555d310632e932ccb7c3` |
| `crates/ckks-train-bench/src/study_provenance/publication_test_hooks.rs` | `301801b16c8961483fb418d8f48e7f583a33fedf49f8f980ba7b82c6c3f73f6e` |

Historical publication source/tests are retained verbatim as `publication.rs.baseline` and `publication_tests.rs.baseline` in the author directory. They are historical vulnerable behavior, not enabled production fallbacks or current passing tests.

## Shared migration and safety boundary

The actual call graph remains `study_reference/publication.rs` Reference/F64 Diagnostic/Diagnostic/Failure -> `Store::publish` -> `Native::new` -> C/W/F/L/D; target.rs `lock_target` -> `TargetStore::publish` -> the same Native. Existing prepublication role, cap, selector, context/source, profile-empty, Train-existing-only and guard rules remain. Future D7 can use this same private primitive; no invented D7 caller or context/gate factory is added.

The new private `publication_anonymous.rs` forbids unsafe code. C captures fixed `/proc` with nofollow directory flags; verifies procfs; captures `self/fd` using safe rustix openat2 with BENEATH|NO_XDEV|NO_MAGICLINKS; verifies the returned directory; opens O_TMPFILE relative to the held parent with RDWR/CLOEXEC/NOFOLLOW and no O_CREAT/O_EXCL; safely converts OwnedFd to the sole File, fchmods0600 and validates type/effective-owner/mode/nlink0 and publication mount; follows the owned descriptor's decimal entry through the held proc directory, compares dev/ino and revalidates owner/mode/mount. Root+proc-directory+O+checked-FD remain owned during capture; temporary check/root handles drop after success. No caller-selected proc root, textual readlink reopening, weaker retry or named fallback exists.

L uses retained File/proc-directory borrows for the safe rustix linkat with SYMLINK_FOLLOW and validates original process identity. It returns immediately after syscall success. No post-link fallible operation is hidden in L. D syncs the same parent once; both Store and Target then confirm final nofollow inode identity, owner/mode/mount and anchors/guard before a DurableBytes result. Target outer source/context/clock confirmation remains unchanged. Every D or late failure maps to VisibleUnconfirmed and leaves visible finals untouched. No new nlink1 restriction is imposed on finals.

`publish_ops` has exactly C,W,F,L,D, with errors at1/2/3/4 BeforeLink and at5 VisibleUnconfirmed; success5. There is no unlink operation, suffix loop, second parent sync, raw-FD ownership transfer, AT_EMPTY_PATH or cleanup receipt. File/proc ownership release is Rust drop, not instantaneous storage reclamation or a fallible close report. Existing unsafe anchoring/guard/mount helpers remain historical and are **not** globally audited or relabeled safe.

Both recover paths construct `Native::recovery` with an already opened File and no proc authority, then perform F,D only. They never call anonymous create/capture. Existing temp grammar, nlink2 old final/temp state, unknown-entry rejection, recovery flags and no-adopt/no-reset/no-cleanup policy are preserved. The historical linked-temp recovery regression now explicitly builds a safe old hardlink fixture instead of making new Native generate an obsolete state.

## Dependency and actual compile evidence

Only dependency change is the Linux normal direct edge `rustix = =1.1.6`, default features disabled, std/fs enabled, MIT alternative selected. Cargo.lock adds one bench dependency edge; no package version/checksum changes. rustix was already locked at checksum `e5e76742e5294153d4dde1207dbec5bdd81eef6fabdc0d2118babadabfd858aa`. Its declared MSRV1.65 is below actual fixed-toolchain rustc1.85.1 (host aarch64-apple-darwin). Installed x86_64-unknown-linux-gnu std was used, not installed or linked by this author. This imposes the ratified actual openat2/Linux5.6 API availability floor plus working procfs/O_TMPFILE filesystem semantics. Non-Linux and unsupported/restricted Linux refuse new publication.

[Dependency source pins](publication-mutation-source-author1/dependency-source-pins.json) cover inspected installed manifest/MIT license/API files; no external implementation or source excerpt was copied into project source. Existing THIRD_PARTY_NOTICES/deny.toml were inspected and unchanged. `cargo tree` confirms only std/fs/alloc on the relevant rustix edge. Offline cargo-deny license check passes with the existing unmatched BSD-3-Clause allowance warning; no widened license policy or claim of a completed redistribution package follows.

TDD ordering: meaningful small five-Ops and real native success tests were added before implementation. `red01-host.log` genuinely fails with E0046 because old Ops requires unlink_owned. `red02-linux.log` genuinely fails with E0046 plus E0599 because Native::new is absent under real Linux cfg. Neither run executes tests. Implementation then compiles GREEN. The two borrow witnesses separately fail as expected with E0505. They check the source-borrow prerequisite, not actual private helper execution.

All28 immutable completed logs, exact argv, status and SHA256 are in [validation-log-index.json](publication-mutation-source-author1/validation-log-index.json); logs were created with exclusive names and never overwritten. The only nonzero compile statuses are the2 expected REDs and2 expected borrow negatives. All intermediate GREEN/fmt passes and final passes remain recorded. Read-only discovery glob/missing-file errors are separately disclosed in `read-only-command-failures.txt`; there was no unreported compiler failure or automatic fallback.

| Final verification | Exact command / log | Result |
|---|---|---|
| Default host test compilation | `cargo test -p ckks-train-bench --no-run`, no-run-default-final01.log | exit0 |
| Parallel host test compilation | same + `--features parallel`, no-run-parallel-final01.log | exit0 |
| CUDA host test compilation | same + `--features cuda`, no-run-cuda-final01.log | exit0; no CUDA device/runtime invoked |
| Actual Linux cfg initial GREEN | `cargo check -p ckks-train-bench --all-targets --target x86_64-unknown-linux-gnu`, green02-linux.log | exit0 |
| Final Linux all-feature strict check | `cargo clippy -p ckks-train-bench --all-targets --all-features --target x86_64-unknown-linux-gnu -- -D warnings`, clippy-linux-final02.log | exit0; library + Linux tests typechecked, no Linux linker/execution |
| Final host strict check | same Clippy without target, clippy-host-final01.log | exit0 |
| Format/diff | `cargo fmt -p ckks-train-bench --check`; `git diff --check` | exit0 each |
| Dependency features/license | `cargo tree -p ckks-train-bench --target x86_64-unknown-linux-gnu -e features -i rustix`; `cargo deny --offline check licenses` | exit0 each |

Final no-run repetition was specifically justified by final test-fixture corrections: actor M now resides outside the selected attempt, fixture cleanup owns the enclosing scratch directory, and the release checkpoint is after late checks. Final source pins remained fixed throughout these final checks. No speculative rerun of unrelated32 codec tests occurred.

## Test code and bounded future execution

Read the complete [work inventory](publication-mutation-source-author1/work-inventory.md) together with [48 exact source-selected filters](publication-mutation-source-author1/exact-source-test-filters.json) before any execution release. This is source enumeration, not `--list`. It separates deterministic model tests, Mac filesystem/refusal tests, real Linux primitive/caller tests, child entrypoints and retained guard/process tests. It inventories payloads, fixtures, logical calls, threads/children, cleanup ownership, mount-reader caps and absence of scientific constructors. No maximum-size payload fixture is allocated.

Implemented shared-native test code includes actual4 caller x6 faults=24, actual4x6 actor schedules=24,4 no-clobber pairs,4x2 late confirmation failures=8, and4x65 low-level legacy prefixes=260. These are Linux filesystem tests, not fake-only integration and not executed passes. Target's real claim_new is preserved: actors add their sentinel only after claim at C; a preexisting remnant still refuses. Legacy-prefix tests are explicitly low-level, never policy bypass evidence. Actor rename1/create1 is coordinated; publisher issues no pathname cleanup.

Minimum Linux native package has21 bounded cases per actually available ext4/XFS mount:1 success,3 no-clobber,6 postcheck windows,2 legacy-prefix,4 drop,2 visible-uncertain/recovery,2 proc-refusal,1 final-swap. It uses real procFD link and inode identity where applicable, labels injected safe-boundary errors separately from OS EEXIST, and has bounded channel/child waits. Its private scratch environment is required and ordinary uid enforced; exact scratch path/mount/capabilities/barriers/child/cleanup approval and outcome recording remain root's separate release. Only a tested ext4 mount would support ext4 evidence; XFS remains unverified until separately run. Test cleanup removes its owned encompassing fixture (including actor M outside attempt), never publisher cleanup or user records.

The pure five-Ops/lifetime models do not assert away the named-entry actor or establish kernel mutation safety. Mode/fchmod/process/proc fault controls are explicitly injected where host changes would otherwise be required. Borrow/compiler evidence and Mac success cannot establish native defect repair, drop reclamation, kernel tracing or power-loss durability.

## Remaining gates and concerns

1. Independent full-diff SPEC/QUALITY review and separately released deterministic/native execution remain required. No current runtime/native acceptance is claimed; no genuine missing context blocked implementation.
2. The fifteen-selector future D7 integration does not yet exist. Its90 primitive/90 actor/975 prefix/120 C-cut+1 ownership-transfer/150 crash/60 ownership-cut and remaining C01-C10 read/lifetime/guard/directory/Initial-Retry/Quick obligations remain pending, with exact ratified five-Ops deltas. This shared source package does not counterfeit those counts with fabricated gates.
3. New tests cover low-level shared Store/Target production methods using private fixtures. They do not replace existing outer Reference/Target role/source/clock, fresh-flock/fork/mount, canonical codec or actual gate-to-begin lineage validation. Retained process tests with historical wait behavior are separately inventoried, not silently included in the21-case native release.
4. Working openat2/procfs/O_TMPFILE support, ordinary account/capability observations, Linux runtime syscall results and storage durability have not been measured here. Unsupported deployment requires reporting/refusal and a new design decision, never a weaker fallback.
5. All original live-gate/resource/currentness/security/science holds, full study runner/table generation, scientific seeds/matrix, Task5 baseline environment and actual GPU/HE measurements remain unchanged and unfinished by this prerequisite.

Source/Cargo ownership is now released. Root may use the frozen full diff, pins and logs for independent review; runtime release is still separate.
