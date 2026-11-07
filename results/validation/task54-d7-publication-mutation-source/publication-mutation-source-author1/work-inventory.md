# Source-selected work inventory, not an execution release

No test or test-binary `--list` executed. Exact fully qualified names and line locations for all 48 source test functions are in `exact-source-test-filters.json`; this is source parsing, not runtime discovery. Of these, 26 are retained/migrated existing functions and 22 are new. Three are child entrypoints, not standalone validation cases. Linux and non-Linux cfg functions are mutually exclusive. Do not run a broad `publication::tests` filter as if it were a bounded pure package.

All new inputs are fixed literal synthetic bytes: O=`owned-ok` (8), R=`foreign!` (8), X=`sentinel` (8). New native tests load no registry, campaign, scientific JSON, golden, model, training data or genuine context/gate. They construct private `Native` instances, and the explicitly named Caller fixture constructs the actual private Store/TargetStore around owned scratch FDs. Those are not public Reference/Train authority factories. Counts below are logical Ops, not kernel syscall totals. Test fixture construction, mount reads, actors, filesystem cleanup, hashing and lifetime observations are additional work and are never hidden as P operations.

## New pure/non-Linux cases

Prefix A=`study_provenance::publication::tests::anonymous_contract::`.

| Exact suffix under A | Cases / work | Inputs and allocations |
|---|---|---|
| anonymous_five_operations_preserve_visibility_boundary | 6, real `publish_ops` with FiveOps: C/W/F/L/D failure + success; total20 Ops | 8-byte literal borrowed; each event Vec at most5 bytes; no filesystem/thread/scientific constructor |
| anonymous_recovery_has_only_two_sync_operations | 1 recovery, F,D | event Vec2 bytes; no filesystem |
| anonymous_owned_lifetime_model | 8 owner scopes: success,W,F,L,D,unused,unwind,lost-after-L; 25 Ops total; one catch_unwind | Rc<Cell> release count, one owned model at a time; events max5. This is a scope/accounting model, not real FD identity, close error or reclamation evidence |
| anonymous_non_linux_refuses_without_creating_entries | non-Linux only: 1 actual Native C refusal | one portable SyntheticDirectory under std::env::temp_dir(), zero payload files, no proc capture; private directory removed by fixture. This is a refusal check, not Linux success |

Two compiler-only files `borrow-move.rs`/`borrow-drop.rs` are 2 negative witnesses for moving/dropping a File with its source borrow still required by a link-signature function. Both produced E0505. They witness the borrowing prerequisite, do not instantiate the private Linux helper, and execute no File open or publication.

## New Linux-native minimum: 21 cases per actually supported mount

The first function below is under A; all others are under A+`linux::`. Exact native scratch root must be separately released via `HEKIT_PUBLICATION_NATIVE_SCRATCH`. No fallback to /tmp, no mount creation, no capability adjustment. A root-owned/effective-root scratch fails. The release must additionally observe actual effective capabilities, kernel, mount ID/type/options and proc support; uid checking alone is not a capability audit.

| Exact suffix | Cases / logical Ops | Additional work |
|---|---|---|
| publication_native_anonymous_success | 1 / C,W,F,L,D=5 | verifies nlink0/mode0600/no named entry before L, final dev/ino=O and 8 exact bytes after drop |
| linux::publication_native_no_clobber | 3 / 4+4+9=17 | regular and symlink existing-final OS EEXIST controls; two competing publishers each reach before-L rendezvous, one wins, two worker threads; 2s channel failure bounds; dev/ino identifies winner |
| linux::publication_native_postcheck_windows | 6 / 5+5+2+3+4+1=20 | six coordinated 1-worker actor schedules: link, post-D release, W failure, F failure, L failure, c6 failure; actor rename1/create1 each. T lives in selected attempt; M is outside it within fixture-owned ancestor. Assert original X inode at M, R bytes at T, F=O where linked. One actor at a time; bounded 2s coordination |
| linux::publication_native_legacy_prefix | 2 / 10 | prefixes k1 and k64; 65 total sentinel writes,512 maximum sentinel payload bytes at once; no suffix search; final only additional entry |
| linux::publication_native_drop | 4 / 1+2+1+3=7 | pre-link unused drop; real 3-byte O progress then injected W error; injected c6 error; one child process exits before L. First three use one Native each; child executes C/W/F and process::exit (not unwind). No named final/temp remains |
| linux::publication_native_visible_uncertain | 2 / 5+4 +2x recovery(F,D)=13 | injected D failure after real link; child process exits after real L before D. Each final remains; exact final bytes inspected, recovery syncs final and parent without proc capture. One child at a time |
| linux::publication_native_proc_refusal | 2 / 0+1=1 | actual filesystem-type validation rejects private ordinary directory; c3 ENOSYS injected before safe openat2. Neither disables real /proc |
| linux::publication_native_final_swap | 1 / 5 | actual Reference Store::publish; after L/D hook renames final, writes foreign8-byte replacement; real final confirmation rejects VisibleUnconfirmed; replacement and moved O remain, O inode checked |

Total78 logical Ops including the two explicitly listed recovery calls. There are 2 child launches in this 21-case package, sequential in their respective tests; helper `A+linux::publication_native_child` is not a 22nd case. Its two exact modes are `before-link` and `after-link`; parent passes `HEKIT_PUBLICATION_CHILD_PATH` and `HEKIT_PUBLICATION_CHILD_CUT`, waits via try_wait with a 5s failure bound, and only kills/reaps its own child on timeout. No fork, arbitrary external PID, SSH or scientific binary is involved. The parent invokes its own test executable with that helper's exact fully-qualified filter and `--test-threads=1`.

The test process remains the publisher and the coordinated worker is the named-entry actor. Coordination is causal via channels; sleeping is not the successful-race oracle. Mode/write/fsync/link/dir-sync injected failures are explicitly test-only failures before the named safe-call boundary; a failing L hook is not an observed failing kernel link syscall. The real existing-final EEXIST controls exercise OS failures. No trace here claims independent strace/ptrace or power-loss evidence.

## Actual shared callers and additional primitive controls

All following suffixes are under A+`linux::`.

| Exact suffix | Cases / logical work | Construction / allocation / side effects |
|---|---|---|
| shared_anonymous_publication_faults | 4 callers x6=24;80 Ops | 24 fixture/caller constructions. Real Store::publish for Reference/Diagnostic/Failure and TargetStore::publish for Target. 8-byte inputs, event strings max5; final/error mapping asserted |
| shared_anonymous_postcheck_windows | 4x6=24;80 Ops | 24 fixtures/callers and24 sequential actor workers. T sentinel created at C after genuine Target claim_new, so preexisting-remnant policy is not bypassed. Actor windows L/release/W/F/L-failure/c6; T/R, outside M/X preserved; successful F inode equals recorded owned O. `release` is after late checks immediately before return/File release |
| shared_anonymous_late_confirmation_faults | 4x2=8;40 Ops | 8 callers; explicit final-check or anchor-check failure after D; VisibleUnconfirmed and final bytes retained |
| shared_anonymous_no_clobber | 4 caller pairs /32 Ops | 4 callers,8 publish attempts: first4x5; second Reference/Diagnostic/Failure4 each, Target0 because real claim_new refuses existing final. All preserve existing inode/bytes |
| shared_legacy_temp_preservation | 4x(k0..64)=260;1300 Ops | 260 low-level Native/fixture constructions; not Target claim_new. Sum8320 legacy sentinel writes=66,560 payload bytes across sequential cases; maximum64 sentinels/512 bytes at once, one O8-byte payload per case. Case directories removed between iterations |
| anonymous_create_substep_failures | c1..c8 + c5-owned transfer=9;9 outer C Ops | nine fixtures; failed j visits exactly j logical c-substeps; c5-owned is separately counted, not a ninth stage. No named entry; Native has no adopted File/proc handle on C failure. Real helper RAII is used, but test assertions do not measure physical reclamation or close errors |
| anonymous_create_refusal | 6 injected errnos;6 C Ops | EOPNOTSUPP,ENOSPC,EROFS,EACCES,EINVAL,ENOENT at c5; six fixtures, no named fallback |
| proc_capture_refusal | 12;first11 C Ops +lastC/W/F/L=15 Ops | missing root/symlink/nonproc/ENOSYS/EPERM/BENEATH/NO_XDEV/magic/nonproc-fd/missing-decimal/wrong-inode are scripted error positions; no host mutation. Last case changes only the private test origin field after real capture and calls actual Native::link, verifying its same-process refusal. This is not a real fork test |
| anonymous_success_records_five_operations_and_eight_create_substeps | 1;5 Ops | exact recorder events C,c1..c5,c5-owned,c6..c8,c8-done,W,F,L,D,final-check. c5-owned/c8-done are ownership/binding cuts, not extra c or P stages |

Shared contract totals are independent of the 21 minimal native cases; none has run. No case silently adds a fifteen-selector D7 claim. Native successful low-level publication plus final confirmation calls the existing mount observer5 times (C: parent/O/checked3; final: parent/current2). The fixture adds3 mount observations (root, root again, attempt). A shallow synthetic Store::publish success has11 mount observations excluding fixture/caller construction (before recheck3, C3, final2, after recheck3); Target success9 (before guard2,C3,final2,after guard2). These are logical mount() invocations, not filesystem syscall totals. Actual production Anchors may retain more components; their previous checks are unchanged.

Each mount() observes bounded `/proc/self/fdinfo/<fd>` <=16,384 bytes and `/proc/self/mountinfo` <=8,388,608 bytes, and fstatfs; no test constructs either maximum-size buffer. The existing reader adds a one-byte over-cap witness and Vec/UTF8/line-field bookkeeping. These retained observers dominate possible transient input memory, not the 8-byte payloads. Publication itself owns at most proc root + proc FD directory + O + checked read FD during C; after C it retains O + proc directory through late checks. Caller-held parent/anchor/guard FDs and the fixture's own root/attempt FDs are separate. Competing-publisher test doubles the per-publisher bound. All new evidence/canonical parsing, registry loads, MaterializedProfile/model/CKKS/key/ciphertext construction, HE/CUDA kernels and data tensors: zero runtime calls.

## Retained affected tests and explicit legacy fixtures

Exact B prefix=`study_provenance::publication::tests::`. Names below are in the48-function source inventory. All are unexecuted here.

Pure retained filters and work:
- `every_publication_fault_withholds_capability_and_preserves_final`:5 failures+success,20 Ops,8-byte `complete`, pending/final Vec each <=8.
- `competing_publishers_never_replace_first_final`:2 calls/9 Ops,5/6-byte first/second literals.
- `recovery_never_cleans_any_temporary_name`:2 recovery attempts/4 sync Ops, explicit Fake historical temp remains;8-byte final.
- `recovery_file_sync_failure_preserves_temporary_name`:1 F failure, explicit historical temp remains.
- `publication_mount_equality_is_not_an_ancestor_requirement`:3 equality and4 type checks; no proc/native mount call.
- `short_writes_and_eintr_are_retried_without_losing_bytes`:14-byte input;8 underlying Write calls (1 EINTR +7 two-byte successes), no syscall.
- `concurrent_synthetic_publishers_get_exactly_one_capability`:2 worker threads,1-byte inputs,9 Ops total; existing Barrier/Mutex fake, not native link proof.
- `reader_guard_open_policy_is_derived_from_the_closed_role`:2 role mappings; no live role/context construction.

Portable filesystem tests (Mac filesystem checks, not anonymous Linux passes):
- `fifo_and_nonregular_reads_fail_without_waiting`:one owned temp dir,one FIFO;2 read_path refusals,cap1024; existing unsafe mkfifo fixture retained.
- `target_stamped_read_rejects_byte_identical_inode_replacement`:2 stamped reads,23-byte literal,2 inodes,cap1024.
- `target_unknown_entries_block_new_claim_without_removal`:empty +4 file-name inspection controls,6-byte orphan; final cases-directory fixture; no publisher.
- `target_nofollow_regular_owner_mode_and_guard_link_count_checks`:8 stamped-read paths (3 unsafe modes,good,nlink2,symlink,FIFO,directory),9-byte literal,cap64; safe legacy nlink2 stays legal.
- `target_ancestor_and_profile_swaps_reject_through_held_parent`:3 check_child calls,owned renamed/replaced directory and symlink; no payload/model.
- `target_guard_rejects_nonempty_multiple_links_and_replaced_inode`:3 fresh-guard refusals,empty/9-byte guard literals; no publication.
- `target_recovery_preserves_linked_owned_orphan_after_postlink_failure`:explicit safe fixture creates `.tmp-target-lock-0` and final hardlink to the same30-byte file; one stamped read(cap1024),F/D recovery2 Ops; nlink2/temp preserved. It no longer asks new Native to generate old named-temp state; recovery creates no proc capture.
- `target_directory_creation_never_repairs_an_existing_unsafe_object`:4 private_child calls (new,unsafe existing,symlink,file);7-byte unknown file preserved.
- `reader_guard_existing_children_never_create_or_repair`:4 missing +safe+symlink+ordinary-file+unsafe-mode calls;9-byte literal; no create/repair by existing-child helper.
- `reader_guard_existing_guard_never_creates_repairs_or_follows_links`:6 existing_profile_guard calls (missing,good,nlink2,nonempty,mode,symlink);7-byte literal; no live context.

Guard/thread/process obligations are separate from the21 native cases:
- Portable `target_fresh_guard_fds_serialize_threads_and_other_profiles_do_not_wait`:2 fixture profiles,3 fresh locks,one worker thread; existing40ms negative wait and2s positive wait. It is not causal adversarial publication proof.
- Linux `target_native_linux_separate_process_uses_fresh_flock_description`:one child and2 guard acquisitions; existing100ms negative observation and child wait; child exact helper `target_native_linux_guard_child` has one lock/9-byte marker when invoked with its environment. Existing child waiting behavior is retained, not broadened under this package.
- Linux `target_independent_thread_handles_publish_exactly_once_under_guard`:2 threads,one winner,28-byte private synthetic payload; held guard/claim behavior. Uses released scratch root now, not arbitrary temporary mount.
- Linux `target_native_linux_processes_publish_exactly_once`:2 children (existing fixture contract),one winner; helper `target_native_linux_publication_child` invokes guarded publisher/28-byte payload. This retained two-child test is not included in the new minimum one-child-at-a-time native package; requires its own release/inventory.
- Linux `target_native_engine_preserves_unknown_temp_and_does_not_overwrite`:2 low-level publisher attempts,7-byte sentinel,22-byte synthetic target/11-byte replacement;9 Ops.
- Linux `target_actor_sentinel_replacement_cannot_redirect_owned_inode`:one publisher5 Ops,8-byte O/X/R; actor rename1/create1 outside attempt, sequential source-cut regression. Old `target_replaced_owned_temp_is_never_unlinked_or_published` is archived verbatim in `publication_tests.rs.baseline`; it is not relabeled a new pass.

Required unchanged outer integration filters to retain for separately bounded review include `study_reference::target::tests::postlink_clock_or_context_failure_never_returns_even_synthetic_success` (3 failures+success, two64-character hashes, no context/scientific constructor), `observed_clock_reversal_or_unavailability_is_sticky` (3 small scripted clock sequences), and `clock_equalities_allowed_without_interpolation_or_tolerance` (one3-sample sequence including u64::MAX). Other Reference/Target codec/role/source/guard tests remain required existing coverage; their broad modules contain registry/materialization and are NOT authorized by this inventory. In particular this package does not rerun the previously accepted32 codec tests or imply zero-constructor cost for those unrelated suites.

## Pending D7 protocol accounting

The fifteen real D7 branch-selector callers do not yet exist. Their90 primitive/90 postcheck/975 legacy-prefix,120 C cuts +1 shared transfer,150 file crash cuts,60 c5/c6 ownership cuts,60 short-write scripts,45 drop/name controls,45 late identity/context/clock, original C01-C10 retained reads/lifetimes/directory/guard/Initial/Retry/Quick tests remain a future protocol integration obligation. Shared tests above do not claim those numbers were implemented/executed. Full ENTER+START stays35 P Ops and Quick through END40 in the ratified design; this code supplies only the shared P5 primitive. It supplies no new actual gates, freshness/currentness authority, reset/adoption, retry policy, matrix run or scientific result.
