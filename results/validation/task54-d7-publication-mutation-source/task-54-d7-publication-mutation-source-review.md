# Shared Native publisher migration — independent source review

2026-11-08. **SPEC: PASS. QUALITY: PASS**, for the reviewed source/static package only. No Critical, Important or Minor source finding requiring a fix was found. This accepts neither native remediation outcomes nor runtime authority. The separate 25-filter host draft is source-accurate and bounded as described below; it remains NOT RELEASED and was not executed here.

## Review boundary and pins

Read the Task54 parent brief, source release, complete ratified mutation proposal and independent design review, full author report, complete72,842-byte root diff including all three new files, complete test work inventory and exact-filter inventory. Inspected the relevant unchanged outer Reference/Diagnostic/Failure/Target callers and recovery, actual installed rustix safe APIs/manifest/license, the25-filter runtime draft and recorded static evidence. Independently verified all seven final source pins and all28 log hashes. No source/Cargo/test/`--list`/runtime/native/SSH/generator/HE/subagent/commit/public-doc action occurred. This review is the sole output write.

BASE: `2da02ab7c861c7d85ff25c4ee5bf67f4d86a6c72`. The root's cleanup/docs HEAD2ce43c2 and subsequent unrelated DatasetInput design addendum do not change these seven reviewed source pins or mutation requirements. Source ownership had been released before review.

| Reviewed artifact | SHA256 |
|---|---|
| `task-54-d7-publication-mutation-source-release.md` | `a2608546b37d1faa3ac4c47fd74630fdf982198a07cb1c3eda73788b5651d10f` |
| Ratified `results/validation/task54-d7-publication-mutation-design/task-54-d7-publication-mutation-proposal.md` | `714880b778511c3be75d1ac0e6712e77e24d2564ba52f61cdb36f9da288509f3` |
| Independent mutation design review in that directory | `6e70cd0c7e6b253b002be9bb013cd3827d6d6a9c3b1ea23882fef6f8d0d02a13` |
| `task-54-d7-publication-mutation-source-report.md` | `07919aaeb3827893646ef70fbabc68337d86f1662cc93e7cd2bafeaa14350fa5` |
| `task-54-d7-publication-mutation-source-root-review.diff` | `d1f93e1d027d10fa88048c9697fb145d97c2510623c69f2156361600eb5c3c9c` |
| `publication-mutation-source-author1/work-inventory.md` | `ddb7cbd38baecc208829339ddccbc029da05d78fd42295ffb3c7793a7d3133c0` |
| `publication-mutation-source-author1/exact-source-test-filters.json` | `aa2dd8091135e56083ed81fa7deaee8bad6a0bcf7cb81f4dd754c686b1f58173` |
| `publication-mutation-source-author1/validation-log-index.json` | `3fa982668352fc14f61e5d3bef2dc65f61d30474d93e9695182b2c82d0ffbe9e` |
| `publication-mutation-source-author1/dependency-source-pins.json` | `43902dd6fd907c07a2d6fa7132d084619aa66c3d767fb8bb82f03af86c2e679b` |
| `task-54-d7-publication-mutation-runtime1-DRAFT.json` | `ce89c271991590f59891baca29efce2ae2a3e0cc2bddf81392663d93a8c193c0` |

Exact final source set:

| File | SHA256 |
|---|---|
| `Cargo.lock` | `e5059c771e93261c6fecf855f7a2d8ab052021f62cadafddbb23c43fa2f216c3` |
| `crates/ckks-train-bench/Cargo.toml` | `fc698bca60c78aa259180e6cd238514cb9e83bdb780a70e3c0287831b6a438b4` |
| `crates/ckks-train-bench/src/study_provenance/publication.rs` | `69a4731e5695d986702e079d7d900bb171cd602d07db3a76951a0bb378d61f47` |
| `crates/ckks-train-bench/src/study_provenance/publication_tests.rs` | `7f604bebbca0fa5ae47f63bac1fa97231d78c6a5affbd3371f866d5e95bfc39f` |
| `crates/ckks-train-bench/src/study_provenance/publication_anonymous.rs` | `f6e8b4b4938ba3c30741b7d0f9fcc745395250d65abc8e3f4a91898c84c38038` |
| `crates/ckks-train-bench/src/study_provenance/publication_anonymous_tests.rs` | `417b62401fa01ea1d32fc6211b476c259dfcc3f1e90a555d310632e932ccb7c3` |
| `crates/ckks-train-bench/src/study_provenance/publication_test_hooks.rs` | `301801b16c8961483fb418d8f48e7f583a33fedf49f8f980ba7b82c6c3f73f6e` |

The final-source-pins manifest itself is `8e84b9e34b31da35b369a2d0d53f2cbfb1ae65a362f5e6bb2b1760a438982fcd`. Unchanged inspected outer sources were also compared byte-for-byte with BASE: `study_reference/publication.rs`=`0085c0a9c23a5c4309ec0fb029dc6061860eaafbffb8f1b6502ecf94745e5d3f`; `study_reference/target.rs`=`1c54c5918d0b73dabcf213bf43dacdaa3682951f471dd48a0c6de7579ad49d1b`; `study_reference/target_tests.rs`=`7a3c34222b70a9ec8cf02a46033d995c120d7fe18bbdceb0ea96aba6895806e3`.

## Production implementation assessment

Source paths below are relative to `crates/ckks-train-bench/src/` at the reviewed pins.

`study_provenance/publication_anonymous.rs:1–136` is a private child with `#![forbid(unsafe_code)]`. Its Linux C path captures fixed nofollow `/proc`, validates directory/procfs, then uses held-root `openat2("self/fd")` with BENEATH|NO_XDEV|NO_MAGICLINKS. It intentionally permits ordinary `self` symlink traversal rather than adding NO_SYMLINKS. The returned FD directory is separately checked as procfs. No caller/evidence/environment-selected alternate root or weaker retry exists.

The anonymous File is obtained with TMPFILE|RDWR|CLOEXEC|NOFOLLOW relative to the held publication parent; O_CREAT/O_EXCL are absent. Safe OwnedFd→File transfer leaves one owner. fchmod acts on that File, followed by regular/effective-owner/exact0600/nlink0 and same mount checks. The held proc directory's decimal entry is followed into a temporary read FD and compared by dev/ino and owner/mode/mount. Original File remains owned throughout; root and temporary check FD drop only after successful capture. On every intermediate error, local File/proc ownership is released by RAII, without pathname cleanup.

`ProcFdDirectory::link` checks the captured process ID, retains the borrowed File and proc-directory owner for the actual safe rustix link call, derives DecInt at the call site and requests SYMLINK_FOLLOW. It does not use AT_EMPTY_PATH, parse/readlink a textual target, or transfer/close a raw descriptor. DecInt itself does not pin a descriptor; the actual `&File` argument does. The helper returns the syscall result immediately, with no fallible check after successful linking hidden inside L. PID comparison is the accepted shared fork-origin refusal, not a replacement for future stronger D7 ProcessStamp/context checks.

`publication.rs:20–40` now has exactly C/W/F/L/D. C/W/F/L errors map to BeforeLink; D maps to VisibleUnconfirmed. Removed code includes named-temp creation/suffix search, publisher unlink, the cleanup operation and second parent sync. Existing legacy unsafe anchoring/mount/guard/read helpers remain unchanged; this review does not reclassify the whole parent as safe Rust or erase their prior obligations.

`Native::confirm_final:391–410` opens the final nofollow relative to the same held parent, checks safe regular owner/mode, dev/ino equality with the still-owned File and same publication mount. Both `Store::publish:296–326` and `TargetStore::publish:752–779` keep Native alive through confirmation. Store then rechecks anchors/current kind parent; Target rechecks its held guard and anchors. All such failures are after-L VisibleUnconfirmed. No final is removed after D failure or a late identity/anchor error. The checks provide the accepted bounded revalidation, not global atomicity against later same-UID edits or privileged mount changes.

Non-Linux new anonymous creation explicitly refuses. Linux support still requires successful actual procfs/openat2/O_TMPFILE/ext4-or-XFS observations; errors propagate without fallback. No support is inferred from a kernel version string or a successful Mac build.

## All shared callers and recovery

The real call graph is preserved: `study_reference/publication.rs:122–230` Reference, Diagnostic, F64 Diagnostic and Failure ultimately invoke the same `Store::publish`; `study_reference/target.rs:425–510` `lock_target` invokes the same `TargetStore::publish`. Both instantiate the migrated Native. No parallel D7 publisher or old native named-temp implementation survives on a production writer path.

Outer Reference/Diagnostic/Failure role checks, captured-owner checks, serialization/limits and existing **prepublication** source/context rechecks remain unchanged. Do not overstate this as newly added post-link source/clock confirmation for those three outer roles. The shared Store now adds final-inode/anchor confirmation; Target additionally retains its existing outer `confirm_publication` source/context/clock rechecks, mapping any late failure to `PublishedButDurabilityUnconfirmed`. Target source rechecks and exact materialized input pairing are not replaced by low-level test fixtures.

Recovery uses `Native::recovery` with an already opened File and `proc_fds:None`; `recover_ops` is F,D only. Neither recovery path invokes create or proc capture. Existing Linux/mount constraints still apply to production stores; “recovery does not require the new proc capture/O_TMPFILE support” does not mean an arbitrary unsupported filesystem becomes approved. Target retains stamped before/after namespace, guard, source/context/clock validation around recovery; Reference recovery retains its existing expected digest/record verification. No new recovery-to-live conversion exists.

`safe_file` does not impose nlink1, preserving old final/temp hardlink states. `profile_empty` still rejects every non-guard remnant/case/unknown entry before Target claim. New publication leaves no named temp; that does not reset old orphan state. The migrated historical recovery test constructs a test-owned old hardlink pair directly and keeps its nlink2/temp assertion. The removed vulnerable before-check test is archived as historical source and replaced by an actual anonymous-owner/actor-sentinel test, not relabeled as a new pass. The existing recovery flag remains unchanged.

## Test-source quality and exact work bounds

The tests are meaningful at their stated layers. Pure FiveOps exercises the real sequencing/error driver, with six cases and20 logical Ops; recovery is two sync Ops; the eight-case lifetime model uses25 Ops and one release per modeled acquired owner. These are not native FD identity or kernel cleanup proof. The two compiler witnesses fail on moving/dropping a File while a later use requires its borrow; they demonstrate that language prerequisite, not execution of the private Linux helper.

The native minimum has the specified21 cases:1 success+3 no-clobber+6 actor windows+2 legacy prefixes+4 drop+2 visible-uncertain/recovery+2 proc refusal+1 final swap. Source arithmetic matches78 logical Ops including the two recoveries. The code exercises real proc-FD linking, mode/nlink/inode checks, real existing regular/symlink EEXIST, two before-L coordinated competing publishers, six causally coordinated named-entry actors, explicit partial write, child exit before/after link and final replacement before real Store late confirmation. Actor M is outside the selected attempt but inside the fixture-owned cleanup root; no publisher path cleanup is assumed away. Two child launches are separate before-L/after-L observations, not a22nd validation case. Child waits have a5-second failure bound; channel coordination has2-second bounds. Ordinary runtime still needs the external per-filter bound for potentially blocked OS calls.

Shared tests use the actual private `Store::publish` for three Kind values and `TargetStore::publish` for Target:24 fault scenarios/80 Ops;24 actor scenarios/80 Ops;8 late-check failures/40 Ops;4 no-clobber pairs/32 Ops. Target's second publication performs zero P Ops because real claim_new refuses it. Actor sentinels are introduced only at C, after genuine Target eligibility, rather than fabricating an allowed preexisting-remnant state. These fixtures intentionally bypass public outer context construction; they do not validate a real captured Reference/Train capability.

The260 legacy-prefix cases/1300 Ops are explicitly low-level Native tests; they are not Target policy bypass acceptance. The nine C substep/ownership-transfer cuts, six create errno refusals, twelve proc/origin refusals and exact success recorder are present. Scripted “wrong inode,” nonproc, BENEATH/NO_XDEV and errno labels are failure-boundary injections; they do not themselves demonstrate a live hostile kernel/proc mount or exercise a successful return containing a wrong inode. The real fake-directory rejection, actual final swap, source checks and later native outcome scope must remain distinguished. Similarly an L hook error is not an observed failing kernel link syscall; the real EEXIST controls supply actual syscall failures. The author inventory states these distinctions correctly.

Native C-failure tests assert no adopted Native File/proc state and no named output; the pure lifetime model counts release. They do not directly measure OS FD close count or physical inode reclamation. The reviewed safe local RAII source supports release ownership, and the report appropriately makes no immediate-reclamation or close-error receipt claim. Successful native tests must still not be presented as power-loss durability, independent syscall tracing or comprehensive fault-kernel behavior.

No maximum-size payload fixture is introduced. New O/X/R literals are8 bytes; legacy-prefix occupancy is at most64×8bytes at once. Existing mount readers can consume their16KiB fdinfo and8MiB mountinfo limits plus parsing overhead; the inventory names this rather than equating an8-byte payload with the whole memory cost. Real model/materializer/registry/CKKS/P1/key/ciphertext/CUDA/HE construction in these new test paths is zero.

The48 source-filter entries are unique and match their source tests/module paths. Their `line` values consistently point to the `#[test]` attribute; the `fn` declaration is the following line. This is a valid test location, not a missing or wrong filter. Three entries are child helpers and Linux/non-Linux cases are mutually exclusive, so48 must not become a broad test-count expectation on one platform.

## Root host-runtime draft assessment

The pinned draft contains25 unique exact filters, all defined in the inspected source with applicable Mac cfg:3 new pure+1 non-Linux refusal+8 retained pure+10 portable filesystem+3 unchanged outer target clock/context tests. None selects a Linux-only anonymous success/helper, real guarded process race, registry/materializer or scientific constructor. The three target functions were read directly in `study_reference/target_tests.rs:16–44,369–395`; they are small scripted ClockTracker/confirm_publication checks.

The draft accurately includes the two-thread fake race and actual portable owned-filesystem work: directory/file creation, mode changes, FIFO, symlink, hardlink, replacement checks and F/D recovery syncing. These are not “zero OS work.” Its explicit60-second per-filter bound must be enforced by the eventual runtime controller, especially because the retained fake Barrier/joins have no independent internal timeout. The selected fixture paths use fresh PID-scoped create_dir and fail on preexistence; later release must retain owned-temp cleanup and timeout/failure evidence rather than broad removal/retry. No host change or arbitrary process action is implied.

No draft correction is required for filter names/count/category. Before actual release, root still needs the intended exact invocation/binary and source pins, one-selected/one-passed expectation, stdout/stderr and exit/timeout custody, owned scratch policy and no automatic retries. The current JSON explicitly says never execute it and supplies no execution authority. A successful25-test Mac outcome could establish the listed pure/portable/refusal behavior only; it cannot accept the native mutation repair.

## Dependency and static evidence assessment

The only manifest/lock change is the ratified Linux normal direct dependency `rustix = =1.1.6`, default features disabled, std/fs enabled, and the bench dependency edge in Cargo.lock. No package version/checksum change occurs. Installed manifest declares Rust1.65 and MIT among its alternatives; the pinned safe openat2/linkat/fchmod/fstatfs APIs and DecInt implementation match the call sites. All six dependency-source pins match. No external implementation copy, new project unsafe helper, expanded license allowance or Unicode exception is introduced. Cargo-deny recorded the existing unmatched BSD-3-Clause allowance warning, not a license failure. Notices remain a dependency/license policy boundary, not a claim of a completed distribution bundle.

The28 recorded log hashes match the index. The four nonzero entries are the two expected compile REDs (host E0046; Linux E0046/E0599) and two expected E0505 borrow witnesses. The inspected final host/Linux strict checks and no-run records align with the author report; dependency tree shows std/fs/alloc, and license log reports success. These are recorded author executions, not reviewer reruns. Linux `cargo check`/Clippy typecheck does not link or execute the Linux binary; default/parallel/cuda host `--no-run` does not run tests or GPU kernels. Hash verification and source inspection cannot independently reconstruct every original compiler process/timing, so this review does not convert static logs into native observations.

## Outstanding acceptance boundaries

1. Separately released, pinned host runtime outcomes remain unobserved by this review. No25-test pass is asserted.
2. The actual unprivileged Linux21-case package still needs scratch mount/kernel/options/effective uid/capability/proc observations and outcomes. One ext4 result cannot claim XFS. No mount, capability acquisition, security weakening, remote run or fallback is authorized here.
3. Actual fork/guard/mount and unchanged outer role/source/clock integration tests remain their separately inventoried obligations. The same-process negative mutates a private origin field; it is not a real fork observation. The minimal21 cases do not replace those tests.
4. Future15-selector D7 tests/protocol/custody/live readers/gates/begin remain absent and separate:90 primitive,90 actor,975 prefix,120 C cuts+1 transfer,150 file cuts,60 ownership cuts and the remaining accepted C01–C10 obligations. This package supplies the shared P5 primitive, not those totals. Full35/Quick40 logical publication totals remain design counts.
5. Retained evidence applicability, real physical model/current observations/cooperative claims, root-current authority, target-before-start/consuming ownership, scientific result publication and the complete runner/tables remain unimplemented or separately gated. Synthetic coherent Full oracle acceptance is independent from this source review.

All21 profiles, Full2680+20+7, Quick268, registered seeds, full-encrypted-block hold, original science/security thresholds, baseline/Task5 requirements and permissive-license policy with Unicode-table-only exception are unchanged. No live capability or scientific success follows from this source PASS.
