# Task54 D7 concrete custody test inventory

Status: proposed before-protocol-source test contract, FIX1 for review I1, 2026-11-07. **No tests executed and no source release or runtime authority.** Read `task-54-brief.md` and the complete custody inventory brief first. This completes the C01–C10 specification work at the representation/operation level; root independent review precedes ratification. The concurrent evidence/Quick codec writer owns source, and is not instructed to implement protocol by this document.

## 1. Source pins and real existing hooks

Read-only HEAD observed `1e65598b73a3ff2fd407daa362ae1beaa423e02c`. The working-file SHA256 values, rather than an assumption that concurrent HEAD never moves, identify this inspection:

| Path | SHA256 |
|---|---|
| `crates/ckks-train-bench/src/study_provenance/publication.rs` | `8cfabba4dcf7daae93e90ce40a0bc1bcf13c59c713bb5c9d65964b97a4a2c004` |
| `crates/ckks-train-bench/src/study_provenance/publication_tests.rs` | `10f640a0961da262ddcd15627a5a1c59455b1fddbee7cc5c9a4711b6eff33fd6` |
| `crates/ckks-train-bench/src/study_reference/target.rs` | `1c54c5918d0b73dabcf213bf43dacdaa3682951f471dd48a0c6de7579ad49d1b` |
| `crates/ckks-train-bench/src/study_reference/target_tests.rs` | `7a3c34222b70a9ec8cf02a46033d995c120d7fe18bbdceb0ea96aba6895806e3` |
| `crates/ckks-train-bench/src/study_reference/document/admission.rs` | `bc9e54a697cb8990dd85e2e6853f80fb9309a05960a1ba08969abe440b836d01` |
| `docs/study-gates-design.md` | `89ef7d32d0f7e02b97b2497e5a466537772461153231717271d9d184ada7b4cf` |
| `docs/study-target-design.md` | `c4c14a98802de7d5f74a1bbe97d7912751ad80787d892940969a6905e47a5dba` |
| `results/validation/task54-d7-gate-custody-design/task-54-d7-gate-custody-proposal.md` | `fc93856110cb2e8c7382fe792b21cf0550a85007d9267d09f75d6e9637e8c73f` |
| `L/task-54-d7-gate-schema-proposal.md` | `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b` |

`L` is this plan ledger. Existing operations are **in publication.rs**, not presumed `native.rs` or `protocol.rs`:

At the final read-only pin check, the four publication/target source/test files above were unchanged. The concurrent codec owner's `document/admission.rs` had become `357a5bae9c33e3604fc36eb58a96adac709ec6a7c629672d0db7f13cda2a1cf4`: inspected diff adds private component visibility/binary decoders; the three C1 helpers re-read below are unchanged. This observation is not review/approval of that concurrent source. The inventory uses the originally pinned helper bodies, and protocol source must reconcile the owner's accepted final pin.

* `Ops::{create,write,file_sync,link,dir_sync,unlink_owned}` and `publish_ops` are private. The successful trace has7 calls. `recover_ops` has2 calls, file_sync then dir_sync. Current `Fake` records string events and one named failure; it cannot distinguish multiple files, arbitrary ordinal failures or inode identities. It remains useful for the7-step primitive, not a complete attempt protocol.
* `Native::create` tries exactly `.tmp-<id>-0` through `-63`, with exclusive nofollow creation, then fchmod0600 on its own new FD. A non-EEXIST error stops. fchmod failure attempts its own cleanup **inside create**. `publish_ops` does not call cleanup after create itself fails. `Native::link` opens the temp through the held parent, compares dev/ino to its owned file and uses no-replace linkat. `unlink_owned` repeats that identity check and refuses a replacement visible at that check. Both helpers then operate on a pathname: a replacement after the check and before linkat/unlinkat is not atomically excluded by the retained FD. The pinned source therefore does not establish unconditional replacement preservation; exact post-check exposures are specified in §7 FIX1 below. `write_complete` is `Write::write_all` (short-write/Interrupted handling), not a single OS write guarantee.
* `Anchors` retains parents; `check_child` reopens relative to the held parent. `private_child` permits existing directories, validates them and never repairs mode; therefore it is **not sufficient for exclusive named-attempt creation**. `Anchors::child` adds child-sync then parent-sync. `child_existing` does not create/sync. `safe_file` checks regular/UID0600, not nlink1; `validate_guard` additionally requires len0/nlink1. `fresh_guard` opens a fresh description, validates before flock and after waiting; do not substitute dup of the locked FD.
* `read_stamped_on` opens+validates, reads bounded bytes, reopens the name and compares the full stamp. Its local File closes on return. A later retained-FD helper is needed. `TargetStore` currently selects only `target.lock.json`, cap7,324,204, Native id `target-lock`; `Kind` selects references/diagnostics/failures only. There is no existing AttemptFile, Quick selector, ActiveAttemptHandles, QuickProfileStore or admission/start transaction API.
* `TargetStore::open` chooses CreateReference versus ExistingTrain from actual role. `profile_empty` permits only dot/dotdot/guard; any target remnant/cases/unknown blocks new Reference claim. Do not widen it for attempt inventory. `TargetStore::recover` verifies before resync and rereads identity before/after; it preserves temps. `ProfileTargetStore::recover_lock` remains Reference-only.
* `ProfileTargetStore::read_target` actually loads registry/materializes the selected input, derives identities and rechecks source/context. It is **not** called in these pure custody tests. `confirm_publication` already maps late context/source/clock failure to uncertainty. `ClockTracker` is sticky after failure, permits equality, rejects reversals/unavailability.
* Actual Full codec helpers exist: `check_target_claims`, `check_predecessor_claims`, `check_start_claims`. The predecessor helper intentionally does not prove prior target/policy applicability. C1 composition below is mandatory. SyntheticAdmission/SyntheticStart test wrappers cannot construct live gates.

### Minimum proposed private test seams, not new public architecture

1. Extend the **test-only** `Fake` into a selector-keyed fixed-state recorder with `(file,stage,occurrence)` fault selection and inode/owned-temp snapshots. It still implements existing `Ops`. No configurable production publisher or exposed callback is added.
2. For the closed attempt operations already required by custody design, keep a private test recorder at named internal checkpoints: guard acquired; eligibility complete; before/after each directory action; after each file primitive; retained read; current-context/clock check; transfer/drop. In tests the recorder supplies scripted fake handles/owners and returns only a synthetic observation result, never StartedHeCase. Production uses real handles/gates and fixed concrete operations. No public test factory, bool readiness or deserialized live owner.
3. Add the already-required private `read_held_attempt_file(selector)` returning File+ReadStamp+bounded immutable bytes, and `reread_held_attempt_file` with held-parent/path identity checks. Tests count its **logical bounded reads**, not its underlying read syscalls. The existing helper cannot be claimed to retain an FD.
4. Add the necessary private exclusive named-directory helper: successful mkdirat establishes a new name, EEXIST rejects; open/validate without repairing; sync new directory and parent. For `cases`, case and `attempts`, existing validated containers may be reused. A small internal ordinal hook around existing native create's open/fchmod and link/unlink identity boundaries permits deterministic failure tests; do not add a general syscall framework. Required custody safety is that unknown replacement objects remain untouched, but the current before-check identity test does not establish that assertion across the post-check/pre-path-operation window; see the explicit source prerequisite in §7 FIX1.
5. Recovery of6–8 files plus5 ancestor directories cannot be represented by repeatedly calling `recover_ops` without wrong extra dir-sync ordering. Use an explicit private loop over the closed final selectors, then the fixed parent sequence; a test ordinal counter observes each fsync. Keep existing target-lock recover_ops unchanged.

These are proposed source seams. Their absence now is not a failed test or resource/security unavailability observation. There is no runtime test plan to call real gates merely to exercise filesystem custody.

## 2. Fixtures, allocations and count notation

All fixtures are **private synthetic controls**. No genuine constructor, registry, materializer, params/P1, planner, keys, Encoder, bootstrap, trainer, update or endpoint is called. Protocol tests invoke synthetic owner checks, not a public injection mechanism. The eventual genuine integration remains separately gated.

| Fixture name | Exact content/use | Bounds and ownership |
|---|---|---|
| `F8` / `Q8` | File bytes are ASCII `F0-bytes`…`F6-bytes` / `Q0-bytes`…`Q7-bytes`, each exactly8 bytes, in selector order below; fake verifier observes exact hashes/identity and scripted claims | Opaque **filesystem-only**, not valid canonical evidence. Two fixed8-byte slots (pending/final) per selector; Full112B/Quick128B, one8-byte read scratch, fixed trace capacity256 entries. No file-sized heap allocations in this recorder |
| `F_EMPTY`, `Q_EMPTY` | Selected profile/guard ready, Full retained target control supplied; `cases` absent | One retained synthetic parent identity per path; no real FS |
| `F_CONTAINERS`, `Q_CONTAINERS` | cases/case/attempts ready and empty; new A absent | Only named A must be created |
| `F_A_k`, `Q_A_k` | Named A plus first k durable finals in exact branch order; k0..7/0..8 | Same fixed buffers; later records absent; can overlay one selected recognized temp |
| `F_A_START_EXPECT_NONE/SOME` | Prior A has all5 evidence+admission; start absent/present as explicit request | No current/prior scientific owner synthesized; controls only |
| `F_A_B_ABSENT`, `F_A_B_EXISTS`, `F_A_ORPHAN_C` | Valid scripted A; respectively B absent, B empty existing, or unrelated orphan C retained | Inspect selected A/B only; never classify C as dead/latest |
| `Q_LINKED_FIX1` | Existing accepted `task-54-d7-gate-schema-vectors`:5 Quick evidence and exact preflight/start/end JSON; nonUTF8 argv intact | Evidence lengths5571,7138,5359,6442,5429; JSON7089,1863,2163. Use existing complete bytes, not generator in custody tests |
| `FULL_CODEC_OLD` | Existing target_goldens lock/reference/initial/retry-no-start/retry-with-start/start-initial/start-retry and existing parser test wrappers | Raw lengths9156,7857,10546,10817,11002,7690,7699. Frozen originals never regenerated/edited; each mutation clones exactly1 chosen file ≤11002B |
| `FULL_EVIDENCE_SCHEMA` | Existing accepted5 Full evidence vectors3486,4962,3040,4265,3252 | **Not linked to FULL_CODEC_OLD**: they have different synthetic identities. Do not claim a combined Full positive lineage from them |
| `CAP_LENGTH` | Advertised size cap/cap+1 and synthetic EOF/error reader | Length arithmetic only, constant≤64B scratch; no allocation of A/Full70MB; unchanged cap under test |
| `NATIVE8` | Same8-byte file controls in explicitly private scratch directory on an explicitly approved Linux ext4/XFS mount | No production context bypass and no scientific tokens; fixture cleanup may remove only its test-owned root after all handles close. This is not recovery cleanup |

The `F8/Q8` recorder drives the same fixed ordered I/O body through private test seams with a synthetic validation result; it proves publication/control ordering only. Real Full codec composition tests use FULL_CODEC_OLD independently. A fully cross-linked new Full gate+admission fixture is not present: before **combined codec+protocol** integration, freeze a new independent linked synthetic fixture or explicitly retain this separation. No test may quietly pretend old admission gate hashes identify new evidence bytes. This is a concrete integration-fixture dependency, not permission to edit frozen vectors or introduce a synthetic live constructor.

Per scenario run sequentially, reset only the in-memory recorder between cases; never load a Cartesian product into memory. Read-only literal files can be embedded once per test binary (sum bounded by their small manifest); a case clones at most one mutated raw document≤16384B and holds at most two parse payload buffers≤16384B each in addition to immutable fixture bytes. Those are **test fixture/requested buffers**, not claims of exact Rust allocator calls or RSS. The real parsers' allocations still obey their existing caps; use their allocation counters where available, not invented zero-allocation claims. Fixed-state F8/Q8 needs no dynamic payload Vec. Trace256 covers any one normal/fault transition; longer interleavings use two bounded traces, not unbounded logging.

`P(f)` is exactly one `publish_ops` call and stages C,W,F,L,D1,U,D2 below. `R(f)` is one complete bounded held-file byte read plus stamp/namespace validation; metadata/open counts are not counted as R. `H` is one typed snapshot/owner/context checkpoint; its actual native source/provenance reads are separate and platform-dependent. `G` is one fresh-guard acquisition, `g` release. `Vt(a,T)` is one current/prior admission check against the SAME retained target/reference/policy T plus actual policy comparison; `Vp`, `Vs` are predecessor/start-claim checks. `D(d)` has four ordered logical actions M=mkdir, O=open/validate, S=child sync, P=parent sync; these are not four literal OS syscalls. Every error stops the trace except specified owned-temp cleanup. The exact count is the listed trace prefix plus cleanup, not a guessed syscall total.

## 3. Closed selectors and baseline deterministic traces

| Branch index | Final | Native temp id | Cap |
|---|---|---|---:|
| F0/Q0 | inputs.evidence.bin | gate-inputs | A2097152 |
| F1/Q1 | implementation.evidence.bin | gate-implementation | A |
| F2/Q2 | layout.evidence.bin | gate-layout | A |
| F3/Q3 | resource.evidence.bin | gate-resource | A |
| F4/Q4 | user.evidence.bin | gate-user | A |
| F5 | admission.json | admission |70492488 |
| F6 | start.json | start |A |
| Q5 | quick.preflight.json | quick-preflight |A |
| Q6 | quick.start.json | quick-start |A |
| Q7 | quick.end.json | quick-end |A |

All15 branch-selector instances are tested even where5 filenames match. Full path: `campaigns/C/profiles/p-P/cases/c-I/attempts/A`; Quick: `quick-campaigns/C/profiles/p-P/cases/q-I/attempts/A`. P/I derive from approved encoders; A is a validated explicit identifier, not a hash/path argument. Distinct operational Quick attempts retain the same q-I case slot. Full ExistingTrain ancestors never created by this source path.

### Baseline test trace contract

To make injected counts reproducible, the small test body observes the following explicit checkpoints. These are logical unit-test operations; production may perform additional internal metadata checks without changing byte authority. Test hooks may not omit a mandatory validation merely to match a count.

1. **Admission/preflight (`ENTER`, each branch):** bounded branch/owner fixture validation H0; G; guarded H1+eligibility inventory; D for each absent container in order cases,case,attempts,A; for i0..4 `P(i),R(i),H_after_i`; `R(0)..R(4),H_before_event`; `P(5),R(5),R(0)..R(4),H_after_event`; g. Namespace creation only after H1/eligibility. Normal counts P6=42 Ops calls, R16, H9 (H0,H1,five after-sidecar,H_before_event,H_after_event), G1/g1. With all4 containers absent D16 logical actions (4 M,4 O,8 sync); with F_CONTAINERS D4. H1/H_before_event/H_after_event for Full check actual retained target/current context, whereas Quick calls target reader0. Five synthetic gate-check slots run at each H; no genuine gate constructor is invoked.
2. **Own start (`START`):** G; `R(0)..R(5),H_start_before`; require own start absent; `P(6),R(6),R(0)..R(5),H_start_after`; g. P1=7 Ops, R13, H2, G1/g1, D0. Already-existing start rejects before P; no rewrite. Full additionally composes Vs with already-verified same target/reference/current context. Quick checks exact preflight Link.
3. **Before-allocation (`BEGIN_CHECK_ONLY`):** `R(0)..R(6),H_begin`; accounted validation spy1 only after start; checked transfer T; Full closes attempt/evidence handles after T, while Quick outer end-publisher frame retains its required handles until end/no-end continuation completes; no publisher, no new guard acquisition or held flock at allocation. R7,H1,P0,D0. Success factory-spy1; each failed check factory-spy0. This is a nonallocating **spy**, not actual CpuOwnedSetup or a live capability.
4. **Quick end (`END`):** still-origin-bound caller check; G; `R(0)..R(6),H_end_before`; require end absent; `P(7),R(7),R(0)..R(6),H_end_after`; g. P1=7 Ops,R15,H2,G1/g1,D0; no postmortem/recovery caller accepted.

Thus Full ENTER+START has7 publishers/49 Ops/29 R, and Quick ENTER+START+END has8 publishers/56 Ops/44 R (BEGIN_CHECK_ONLY adds7 R separately if exercised). No flock held across the driver/spy scope. Input source/provenance H-internal reads and native open/stat/flock syscalls are not hidden inside these deterministic **attempt-file** R counts; they require their own source-specific test bounds.

Each R retains or reuses one immutable read buffer, checks identity before/after, and frees only transient scratch after comparing. F8/Q8 can retain final bytes in fixed slots and compare via the8B scratch. No reading all temp contents or additional generic filename opening occurs. In native mode actual `read_file` may perform multiple read syscalls and one initial reserve plus implementation-dependent growth up to cap+1; only logical reads and payload ceilings are claimed here.

## 4. C01 closed inventory

Future exact test filters (not run):

| Filter | Exact parameter cases | Calls and expected result |
|---|---|---|
| `d7_custody_c01_selector_names` |15 selectors × each64 canonical suffixes0..63=960 temp names, plus15 finals; Full profile target temp64 | Selector lookup1/name, no payload reads/P/D/G; exact branch/cap/id mapping. No CLI filename/cap/id accepted |
| `d7_custody_c01_invalid_names` |15×suffixes `00,01,064,64,-1,+1,1x,1.extra,empty`=135;3 Quick finals in Full and2 Full finals in Quick=5; unknown `result.json`, `diagnostic.json`, arbitrary `.tmp-unknown-0`=3 per branch |Exactly135+5+6=146; one scan refusal each, first unknown stops, payload reads0, namespace mutation0. No aliases/truncation |
| `d7_custody_c01_inventory_stream` | Full455 recognized attempt entries (7+448), Quick520 (8+512); add unknown first/middle/last for each branch=6; duplicate injected directory entry2 | Safe entries stat/validate once each; no temp payload reads, bounded cursor/fixed recognized bitset. Unknown at position j causes exactly j entry checks; duplicate same final rejects. Duplicate is injected enumeration anomaly, not a claim POSIX permits two same-name entries |
| `d7_custody_c01_file_caps` |15×{cap,cap+1}=30 length-only cases; Full canonical admission36294820 and+1; Quick canonicalK1048000 and+1; rawA and+1 | Cap preflight1; oversized payload read/allocation0,P0. Exact-cap **transport** eligibility does not promise syntactically valid evidence |
| `d7_custody_c01_file_objects` |15×{symlink,directory,FIFO,wrong UID,mode0644,mode0400,different device,different mount ID}=120 | One selected metadata/identity rejection before payload read, P/D0; ordinary final hardlink2 is allowed if all other checks pass (15 controls), not guard semantics |
| `d7_custody_c01_guard_and_directories` | Guard {nonempty,nlink2,replaced inode,symlink,wrong UID,0644}=6; each of cases/case/attempts/A ×{symlink,regular file,wrong UID,0755,wrong mount}=20 | G refusal or selected directory validation refusal, no chmod/removal. Guard nlink1 required; only new own temp/guard may receive their existing creation fchmod |
| `d7_custody_c01_full_existing_only` | ExistingTrain missing campaigns/campaign/profiles/profile/guard=5; CreateReference fresh-profile entries cases/target/temp/unknown=4 | Current `target_open_policy`/`existing_private_child`/`existing_profile_guard`/`profile_empty`; no create/repair for Train; Reference claim4 reject |
| `d7_custody_c01_identifier_surface` | Empty, `.`, `..`, slash, backslash, nonASCII,NUL,129B (8); valid1B/128B/`A_9-` (3); forged caller hash/profile-slot mismatch1 | Existing validated identifier routines, one parse each; only11 grammar cases and one derived-slot equality case. Compile-fail2 for arbitrary filename/cap selector construction |

Selected Full profile allows guard/target/cases plus64 target remnants, Quick guard/cases. Selected case only attempts; selected attempts list is not an inferred latest history. Initial stops at first named attempt; Retry examines only selected predecessor/new name and required parent inventory. ENOENT alone is absence; EACCES/EIO/changed identity each refuse (`d7_custody_c01_absence_errors`,3×selected case/attempt/start=9). Stream error must not be mistaken for end-of-directory; inject at entry0/1/final (3×2branches=6).

## 5. C02 no early mutation and Quick bootstrap

`d7_custody_c02_precheck_no_mutation`: branch2 × five failing gate slots=10; five mixed-scope slots in Full+five in Quick=10; Full missing verified target/target receipt/threshold origin/current target change=4; both branches wrong execution/input/request/attempt/context owner=10; each15 selector cap+1=15. Total49 independent fixture cases, one precheck dispatcher invocation/case, P0/D0/G0 on precheck failures. A failing gate slot i (zero-based0..4) causes i+1 scripted gate checks and stops; scope/cap failures before gate dispatch cause0, genuine constructors always0. Frozen eight branch-test names remain separate, unchanged requirements, not replaced by this arithmetic.

`d7_custody_c02_guarded_refusal`: each branch×{context replaced,process changed,resource stale,user revoked,clock reversed,existing named attempt}=12, G1/g1, P0/D0;5 gate rechecks at H1 only until first failed slot, no later slot assumed checked. Full target/control check1, Quick target0. No attempt sidecar despite all earlier pure validation passing.

`d7_custody_c02_quick_bootstrap_cuts`: Quick valid gates first; missing administrative ancestors are exactly quick-campaigns,C,profiles,p-P (4). Bootstrap D4×4=16 logical actions plus create-or-open guard and its file-sync/profile-sync3 logical actions=19; fault at each index1..19 independently. Retain every successfully created ancestor/guard, no case/attempt/sidecar, no Full tree touch. This19-action count treats guard create/open+validation as one named high-level action, not one syscall. At native open/fchmod failure the same owned creation rules apply. Bootstrap parent syncing is administrative only, never Initial eligibility or authority. Full equivalent bootstrap calls0 (`d7_custody_c02_full_never_bootstraps`,1).

## 6. C03 order, counts and read custody

`d7_custody_c03_full_order` / `d7_custody_c03_quick_order`: one F_EMPTY/Q_EMPTY control each, exact §3 trace. Assert no event publisher before5 durable exact sidecars, no start publisher before fresh rechecks and own-event verification, no setup spy before start. Expected ENTER creation M4,Open4,sync8; P6=42 Ops; START P1=7. Quick END P1=7. Every recorded gate digest equals SHA256(exact retained sidecar bytes), never normalized JSON. Fixed native temp IDs and names match each selector.

`d7_custody_c03_read_failure`: for ENTER R1..16, START R1..13, BEGIN R1..7, and Quick END R1..15 inject {read error,changed held stamp,changed path inode,byte mismatch}; Full(16+13+7)×4=144 and Quick(16+13+7+15)×4=204. P count exactly number completed before that R in §3; no later write or factory spy; any final already linked is preserved. Read error does not become ENOENT. The R helper owns the same opened FD across reads, no file reopening from artifact-provided paths.

`d7_custody_c03_source_and_event_clock`: H checkpoints ENTER9/START2/BEGIN1 and Quick END2, fail source/context/clock each (Full36,Quick42). The H-after-publication refusal yields uncertain/no live observation, never apparent success. Before event time invalidity no event P; sidecars may already be durable. Clock harness uses existing ClockTracker with equal-wall/equal-monotonic positive, backward wall, backward monotonic, unavailable wall, unavailable monotonic, overflow5 negatives; failed tracker second sample must not query clock again. This is source-style clock injection, no OS clock manipulation.

## 7. C04 exact single-file publication failure traces

For each15 branch-selector instances, call the existing `publish_ops` once. Use `d7_custody_c04_publish_faults` with seven faults plus one success=120 scenarios. Test record indices are1-based and count the failed call. Symbols: C create,W write,F file sync,L link,D1 first directory sync,U unlink_owned,D2 second directory sync.

| Injected operation | Exact calls | Count | Primitive result | New final / own temp after return |
|---|---|---:|---|---|
| C1 fails before creation | C! |1| BeforeLink | absent / absent |
| W2 fails | C,W!,U |3| BeforeLink | absent / removed if U succeeds |
| F3 fails | C,W,F!,U |4| BeforeLink | absent / removed if U succeeds |
| L4 fails atomically before link | C,W,F,L!,U |5| BeforeLink | absent / removed if U succeeds |
| D1 index5 fails | C,W,F,L,D1! |5| VisibleUnconfirmed | exact complete final / own hardlink temp remains |
| U index6 fails | C,W,F,L,D1,U! |6| VisibleUnconfirmed | exact complete final / own hardlink temp remains or a replacement already visible at the U check remains untouched |
| D2 index7 fails | C,W,F,L,D1,U,D2! |7| VisibleUnconfirmed | exact complete final / own temp removed |
| no fault | C,W,F,L,D1,U,D2 |7| durable primitive only | exact complete final / absent |

`d7_custody_c04_cleanup_faults`:3 pre-link failures W/F/L ×15=45, also fail cleanup U. Counts remain3/4/5; result BeforeLink and owned temp remains. A cleanup error never erases already-durable earlier sidecars. For final already existing, the L failure preserves old bytes and removes only the new owned temp;15 cases `d7_custody_c04_final_exists`.

`d7_custody_c04_after_return`:15×identity/context/clock failures=45; P completed7 calls, final remains/no own temp, no capability. At attempt level an already-visible admission/start/end yields exact-path/file/payload uncertainty; sidecar failure is phase-specific incomplete evidence, not a false claim that no earlier mutation occurred.

**Do not inject a successful linkat side effect followed by Err from Ops::link and expect correct classification.** Current publish_ops assumes its L error is pre-link. A crash after L is C05, and a late validation failure after a successful L must be represented at the subsequent post-link checkpoint as VisibleUnconfirmed. A minimal new post-link hook must preserve this ownership/error boundary; it may not launder a visible final into BeforeLink.

`d7_custody_c04_short_write`: for each15 selectors run exact14-byte helper payload ASCII `complete bytes` with four `Write` scripts: (a) Interrupted once then seven2-byte writes =>8 Write calls/success, (b) zero on first write =>1/error, (c)2-byte progress then error =>2/error, (d)seven2-byte writes=>7/success. For these short-write-only cases enlarge just the selected current fixed pending/final/scratch slots from8 to14 bytes (42B total for that selected file); no large fixture allocation. One `write_complete` call per scenario, underlying Write calls8/1/2/7; P's W remains one Ops call. Success executes7 Ops; W error executes3 Ops including cleanup. No swallowed byte, repeated Interrupted not success, no claim one syscall per write_all. Existing short_writes_and_eintr helper is the anchor.

`d7_custody_c04_temp_collisions`:15×k0..63 occupied prefix lengths plus exhausted64=975 cases. C called1; its create loop tries k+1 names for k≤63, then owns only suffix k; successful rest W,F,L,D1,U,D2 each1. Exhausted64 tries64 names, returns BeforeLink after1 Ops call; no owned temp, no cleanup, preexisting64 names untouched. Extra non-EEXIST error at attempt0 or after one EEXIST gives2×15=30 cases, respectively1/2 create attempts, no further suffix or unlink. No enumeration/sorting of other names to choose a suffix.

`d7_custody_c04_native_create_mode`:15×{fchmod failure with successful owned cleanup,fchmod failure with cleanup failure,successful fchmod,unknown replacement before internal cleanup}=60 scripted native-internal cases. Existing create stores owned File/temp before fchmod. On failure publish_ops makes only C1; internal unlink attempt1; a replacement already visible at its ownership comparison is refused. No assertion of preservation across the later pathname-unlink window follows; FIX1 schedules that case separately. No real chmod/chown of unrelated objects is authorized.

`d7_custody_c04_temp_swap`:15×{before L identity check,before U identity check}=30. First case L refuses/no final, cleanup also refuses foreign replacement; original inode at separately moved name and replacement both preserved. Second case final exists, U refuses, result VisibleUnconfirmed, replacement preserved; final still equals owned bytes. These30 cases substitute before the check and establish only that boundary. They do not cover the after-check window and must not be generalized to atomic pathname safety. Exact additional schedules follow; native execution remains a separate obligation.

### FIX1 I1: exact post-check / pre-path-syscall schedules

The six filters below each expand over all15 Full/Quick selector instances: **90 additional scenarios**. They supplement, rather than replace, the existing30 before-check temp-swap and60 create-mode scenarios. Their expectation column describes a deterministic model of the **pinned current code**, not an observed native exploit or a passing preservation assertion. Fake operations must model lookup at the path syscall; an inode-conditional fake would hide the issue.

For every row define `T=.tmp-<selected-id>-0`, original newly owned inode **O**, replacement inode **R** containing exact8 bytes `foreign!`, and final name **F** from the selector. The scripted actor performs exactly two modeled namespace actions at the specified cut: rename T→M (M is an actor-held name on the same filesystem **outside the selected attempt directory**), then create T→R. This choice avoids relying on an extra unknown-entry rejection inside the attempt. Native still retains its own File→O and the just-opened checked File→O. Renaming does not change those descriptors. No inherited same-byte assumption is needed. No source change or actual actor operation is performed in this inventory task.

`L-check` means Native::link's `openat(T)`1 plus inode(current)1 and inode(owned File)1 have succeeded. `U-check` has the same three helper calls inside Native::unlink_owned. Actor substitutions occur **after those successful comparisons and immediately before the later path-based linkat/unlinkat**. The counts below are helper/outer operations, not complete OS syscall totals. Each inode helper uses metadata internally; CString work and variable write_all calls are not silently counted as one syscall.

| Exact filter suffix (prefix `d7_custody_c04_postcheck_`) | Cut and outer Ops trace | Outer Ops calls; Native L/U invocations | Path-helper segment counts and current-code resulting state |
|---|---|---|---|
| `link` | After successful L-check, before linkat(T,F); C,W,F,L,D1,U! |6; L1,U1 | L:open1,inode2,linkat1 succeeds and links **R** to final F. U:open1,inode2,unlinkat0 because T now R differs from owned O. Return VisibleUnconfirmed, D2 calls0. F→R and T→R survive; M→O and held O survive until drop. Original O was not published. Native.temp remains Some(T), but current T is not owned |
| `unlink_postlink` | Normal link made F→O; after successful U-check, before unlinkat(T); C,W,F,L,D1,U,D2 |7; L1,U1 | L:open1,inode2,linkat1 succeeds on O. U:open1,inode2,unlinkat1 succeeds on **replacement R**. T is absent, its R entry has been deleted; M→O and F→O survive, owned O FD retained until drop. Native.temp becomes None. Primitive returns Ok; later exact-final checks may still see correct O bytes/inode, so they do **not necessarily detect or repair the foreign-name deletion** |
| `unlink_write_cleanup` | W returns error before writing any bytes; after cleanup U-check, before unlinkat(T); C,W!,U |3; L0,U1 | U:open1,inode2,unlinkat1 removes T→R. F absent; M→O contains empty temp; owned O FD retained until drop. Native.temp=None. Outer BeforeLink; no F/L/D1/D2 calls. No live capability, but replacement preservation has failed in this model |
| `unlink_file_sync_cleanup` | W completed, F returns error; after cleanup U-check, before unlinkat(T); C,W,F!,U |4; L0,U1 | U:open1,inode2,unlinkat1 removes T→R. Final absent; M→O contains complete8-byte payload with sync unconfirmed, held O until drop; temp=None. BeforeLink; no L/D1/D2 |
| `unlink_link_cleanup` | L-check succeeds, linkat(T,F) returns injected atomic ENOSPC with no link side effect; cleanup U-check succeeds, then substitute before unlinkat(T); C,W,F,L!,U |5; L1,U1 | L:open1,inode2,linkat1 **fails without a final**. U:open1,inode2,unlinkat1 deletes T→R. F absent, M→O contains complete synced payload, held O until drop; temp=None. BeforeLink; no D1/D2. This L failure is genuinely pre-link, unlike the first row |
| `unlink_fchmod_cleanup` | C opens owned O, fchmod returns error; its internal U-check succeeds, then substitute before unlinkat(T); C! |1 outer; L0,U1 **inside C**, outer cleanup U0 | C creation openat1/fchmod1(error); internal U:open1,inode2,unlinkat1 removes T→R. F absent; M→O contains empty created temp (do not assert exact mode after fchmod failure), held O until drop; temp=None. C returns error, outer BeforeLink; W/F/L/D1/D2 and outer unlink calls0 |

In the first five rows, normal C has one create-open attempt and one successful fchmod; the last row enumerates its failed fchmod separately. For every row actor rename1/create1 are separate actor operations, not publisher Ops. All rows keep the original inode O at M; deletion of T→R is deletion of that replacement directory entry, irrespective of whether the actor separately retains an R descriptor or hardlink. A test must record F/T/M and both held file identities, not only the returned error. After U success, the source clears Native.temp even though it deleted R's entry and O remains at M. Recovery cannot adopt or clean M, and no cleanup of actor-owned state is part of the protocol.

**Current exposure versus required assertion.** Existing before-check cases demonstrate only rejection of substitutions visible at the comparison. These after-check schedules derive from the inspected path lookup order; no native deletion/race has been executed here. The required custody contract still forbids publishing a foreign replacement as the owned output or removing its entry. The first row withholds authority by the later U failure, but already made a foreign final visible; the unlink rows can remove a foreign entry, and even a later failure cannot undo that. A cooperative profile flock does not exclude this noncooperating actor.

Consequently, before a production-positive/native claim that includes this adversarial preservation guarantee, the source owner must supply and independently review a concrete mechanism/proof that binds the mutation to the verified inode or an actually enforced exclusion covering the entire check-to-use window. Current path rechecks, retained FD and cooperative flock do not by themselves supply that prerequisite. This document neither picks a new publisher architecture nor changes the ratified format/caps/threat assertions. The90 cases are exposure/required-behavior tests, not assumed-green implementation tests; a fake may not grant inode-conditional behavior absent from the production helper.

Post-link uncertainty remains unchanged: in the `link` row linkat succeeded, so L returns success and the later U failure is VisibleUnconfirmed; do not turn that side effect into `Ops::link -> Err(BeforeLink)`. The postlink-unlink row documents that **the current primitive can return Ok despite lost replacement preservation**; it must not be reported as a successful custody-safety test or converted to a live fixture authority. Future source changes must maintain truthful visibility/error classification even if their solution changes exact internal helper details, which must then be reviewed and counted.

A future bounded native companion filter `d7_custody_native_temp_postcheck_windows` has exactly these six schedules for one8-byte selected-file control per schedule with two coordinated actors; it supplements `d7_custody_native_temp_inode_swap`. It needs explicit after-comparison/before-path-call barriers, not a pre-helper barrier or timing-only sleep. Native syscall counts are measured separately; no such native run is authorized or performed here.

## 8. C05 every namespace and crash cut

`d7_custody_c05_directory_cuts`: each branch, four absent names cases/case/attempts/A, four D actions M/O/S/P, stop immediately before and after each =>2×4×4×2=64 crash cuts. Before M current name absent; after M the newly named directory remains even if O/S/P never finishes. A named A exists from after its M and blocks any future Initial. Earlier container-only states with no named attempt are not a consumed attempt. All-created-but-unsynced state is never called durable. No destructor removes created names. `d7_custody_c05_directory_failures` separately fails each action once per branch/name =>32; prefix count4j+(i−1) successful actions and one failing action, no subsequent action (j prior new directories, i1..4).

`d7_custody_c05_file_crash_cuts`:15 selector occurrences×7 P stages×before/after=210 crash cuts. Store exact prior completed selector prefix and current temp/final state:

| Last successful current P stage | Current visible simulated state | Recover/Initial outcome |
|---|---|---|
| none | no current temp/final | named attempt still blocks Initial |
| C | empty owned temp | preserve; no final promotion |
| W | complete temp bytes | preserve; never parse as final |
| F | synced complete temp | same, sync is not final publication |
| L | final+same-inode temp | uncertain final; validate/recover exact finals only |
| D1 | final+same-inode temp, first parent sync observed | still no live return after crash |
| U | final only | may lack final parent-sync confirmation; no live return |
| D2 | primitive durable final | lost caller still grants no live token/end continuation |

“Visible” here is deterministic process-state visibility. A native power-loss test cannot assume all unsynced entries persist; future real crash tests record observed state and apply the same conservative eligibility rules, never equate the fake disk model with filesystem power-loss proof.

For first5 evidence files, any k0..5 finalized prefix without admission/preflight is orphaned; cannot become predecessor or restarted Initial. F5 valid admission with all5 evidence may be inspection/retry predecessor after resync, never resumed. F6 start doesn't prove completion. Q5/Q6/Q7 never enable Quick Retry/recovered end. `d7_custody_c05_cut_state_classification` explicitly inspects F_A_k k0..7 and Q_A_k k0..8 (17 states), plus final admission/preflight with one missing preceding evidence (10 states), temp-only admission/start/end (5 cases across branches). No deleting/reusing A or silently rotating campaign.

## 9. C06 exact recovery composition, including C1

`d7_custody_c06_evidence_matrix`: both branches×each5 evidence×{absent,changed byte,wrong policy,wrong branch,wrong attempt/scope,oversized length}=60. Inspect final event only with complete required evidence, reject otherwise; no new B, no final write/unlink/rename, no live token. Unknown entry or unsafe recognized temp also rejects even if finals are valid. Temp final-name content may be valid canonical bytes but still is not a final.

`d7_custody_c06_links`: fixture rows and exact validation plan:

| Fixture variant | Required check count/expected |
|---|---|
| Full current and prior valid, expected-start absent | Vt(current,T)1,Vt(prior,T)1,Vp1; same T object+bytes/reference+approved policy identity; then recovery sync |
| Full current and prior valid, expected-start present | same2 Vt,Vp1 (Vp itself invokes Vs once when Some, so total actual check_start_claims calls1 and no redundant outer Vs); never call a target check on a different T |
| Current lock/reference/policy changed (3 variants) | current Vt1 fails,prior Vt0,Vp0,sync0,B-create0 |
| Prior lock/reference/policy changed with hashes recomputed (3 variants) | current Vt1 passes,prior Vt1 fails,Vp0,sync0,B-create0; this is C1, not merely self-hash rejection |
| Same target IDs but different retained target/reference bytes or approved policy object (3 variants) | actual scope/context equality fails before Vp; byte/hash equality alone does not replace owner/custody |
| Prior admission wrong contract/selected receipt/materialized/seed/case/attempt (6 variants) | target composition first, then predecessor/owner check rejects; no sync/new attempt |
| Prior threshold bits mismatch independent verified target (1) | scope/evidence check rejects even when supplied lock IDs look valid; no numeric placeholder |
| Expected-start None but actual final present, Some but absent, Some wrong file hash, Some wrong payload (4) | final absence observed under guard; refuse request unchanged; never infer dead process or auto-refresh expectation |
| Valid prior start with changed admission Link/target/reference/policy (4) | Vs/current-target composition rejects; parsed start never ready |
| Start-only Full, start-only Quick, end-only Quick (3) | missing required predecessor/evidence rejects; no reconstruction |
| Full admission or Quick preflight whitespace-only different raw JSON (2) | same canonical hash may hold; expected exact file hash must still reject |

Every listed row becomes a suffixed subcase in that exact filter. Valid-controls2 + mutations29 =31 cases (3+3+3+6+1+4+4+3+2=29). Malformed ordinary metadata is not accepted merely because test fixtures have synthetic headers. For C1 use existing `check_target_claims` **twice before** `check_predecessor_claims`, with same lock/reference snapshot and approved-policy comparison; wrapper tests spy exact call order. Current-target-only+Vp is expressly a failing implementation.

`d7_custody_c06_temps_preserved`: each15 selector×{recognized temp0 beside final,recognized temp63 beside final,temp hardlink to final}=45 safe controls; content reads for temps0, all temps preserved during recovery. `d7_custody_c06_temp_cannot_complete`: each15 selector absent final with its valid-content temp=15 refusals; no rename/link/delete. `d7_custody_c06_unrelated_attempts`: selected valid predecessor A plus orphan C, newer-looking C, live-looking C, unknown selected-profile entry (4); first3 preserve C and only explicit A evaluated, fourth rejects profile inventory. No “latest” call.

Full codec fixtures and evidence fixtures are deliberately separate as §2 states. Shared-file ops controls may script valid claims, but C1's actual pure codec composition tests must use FULL_CODEC_OLD synthetic parsing and exact mutations with recomputed dependent hash fields. No source-encoder roundtrip substitutes for the unchanged independent expected bytes. Any new linked Full fixture is a separately frozen prerequisite for combined codec/protocol coverage.

## 10. C07 recovery synchronization and failure indices

`d7_custody_c07_resync_order`: selected exact records verified/read first, each required final R once; then recheck namespace/guard/context, then ordered sync vector, then final namespace/guard/context recheck. Do not use recover_ops per file (would interleave parent syncs). No P,create,link,unlink or temp read. Five parent directories in exact order selected attempt → attempts → case → cases → profile.

| Selected state | File sync indices | Parent sync indices | Total syncs / initial R |
|---|---|---|---:|
| Full evidence+admission |1..5 evidence,6 admission |7 attempt,8 attempts,9 case,10 cases,11 profile |11 /6 |
| Full plus start |1..6 as above,7 start |8..12 same5 parents |12 /7 |
| Quick preflight |1..5 evidence,6 preflight |7..11 |11 /6 |
| Quick plus start |1..6,7 start |8..12 |12 /7 |
| Quick plus end |1..7,8 end |9..13 |13 /8 |

Five successful controls. `d7_custody_c07_sync_faults`: fail at every index in each row, total11+12+11+12+13=59 scenarios, exactly j sync calls including failed j, none afterward, no new Full B and no Quick retry. All finals/temps unchanged; successful prefix syncing never implies live authority.

`d7_custody_c07_recheck_cuts`: for each5 row at checkpoint {after all reads before verification,after verification before sync,after each sync,after final recheck} inject namespace or context change. Exact count per row2×(3+sync_count):28,30,28,30,32 =>148; no new B, no live receipt. The last checkpoint fault occurs after the actual final recheck success but before return; same late uncertainty discipline applies. Original read stamps/bytes retained; a same-byte replacement must fail identity recheck. R count remains6/7/6/7/8 (post-sync identity check is metadata, not an extra complete byte read in this minimal contract; if source chooses a complete reread, declare/charge the additional bounded R before acceptance).

`d7_custody_c07_retry_after_resync`: successful Full rows only (2), explicit B absent; after full verification and syncs, create only B (D4 logical actions) and publish its5 evidence+admission (P6); unchanged A, no temp cleanup/adoption. A single injected resync error must make B's M call count0. Quick corresponding controls3 never create a second attempt. `d7_custody_c07_no_recovery_conversion`: compile-fail4 for parsed admission→live admission, recovered start→StartedHeCase, recovered Quick preflight→Quick driver, recovered end→end continuation. Parsed inspection types remain independently useful.

## 11. C08 deterministic concurrency schedules and native obligations

`d7_custody_c08_initial_interleavings`: each branch×winner A/B×winner succeeds or first evidence W fails=8 schedules. Two G requests, only one owner at a time, winner creates named attempt exactly1; loser observes it and creates0 even if no final evidence survived. Successful winner ENTER P6 (42 Ops); W-failing winner P1 with3 Ops; loser P0. Common directory creation only once; no delete on winner drop. Release guard enables loser inspection, not takeover.

`d7_custody_c08_retry_interleavings`: (a) two explicit Retry requests same B, (b) different B/C with explicit overlap authority, (c) different B/C without such authority, (d) B already empty. Four schedules, two G acquisitions where both callers run. (a) first creates B once, second B-create0; (b) two distinct attempt creations, prior A unchanged; (c) second refused by user/resource scope, no inference A inactive; (d) no B creation/publication. Retry checks exact selected B absence under guard before doing work for a new attempt; successful prior verification/resync remains C1-composed. Same-name no-clobber also tested at primitive level15 times in C04.

`d7_custody_c08_expected_start_races`: four schedules: old caller final-start before Retry guard acquisition; after Retry takes guard (blocked until it releases); expected Some but start absent; start appears as temp-only before inspection. First requires matching explicit Some or refuses stale None; second may observe final absence and retain None while allowing later A start only under independent overlap authority; third refuses; fourth may match final absence, preserves temp and never infers inactivity. Retry never silently changes requested links. Count2 guard acquisitions in the first two schedules,1 inspection acquisition in other two; no native timing/sleep used by fake scheduler.

`d7_custody_c08_other_profile`: one held guard per profile, unrelated profile may progress; exactly2 acquisitions, neither reuses/dups a locked guard description. `d7_custody_c08_guard_wait_swap`: replace same guard entry while second caller waits; second must reject after wait rather than use stale inode. Fixed2 callers/2 acquire attempts, no second publication.

**Native filters, future explicitly bounded Linux run only:** `d7_custody_native_fresh_flock_threads`, `d7_custody_native_fresh_flock_processes`, `d7_custody_native_initial_race`, `d7_custody_native_retry_same_name`, `d7_custody_native_temp_inode_swap`, `d7_custody_native_guard_wait_swap`, `d7_custody_native_mount_rejection`, `d7_custody_native_fork_refusal`. Each is one named scenario with2 contenders/processes where applicable, one8B payload per final, no HE. Use explicit barriers/IPC rather than claiming timeout alone proves exclusion; timeout is a failure bound. Run on supported ext4/XFS and separately record actual mount facts. Process/fork/context identity assertions require actual production checks in the named native package; OS fake success does not stand in for them.

Existing anchors to preserve/include selectively: `target_fresh_guard_fds_serialize_threads_and_other_profiles_do_not_wait`, `target_native_linux_separate_process_uses_fresh_flock_description`, `target_independent_thread_handles_publish_exactly_once_under_guard`, `target_native_linux_processes_publish_exactly_once`, `target_replaced_owned_temp_is_never_unlinked_or_published`, `target_recovery_preserves_linked_owned_orphan_after_postlink_failure`. These tests currently bypass full approved production context for private filesystem probes; their success would not establish a real gate→begin lineage. Native syscall counts vary with mount observations, scheduling, write_all, fd checks and retries; no exact syscall count is promised by the logical Ops/D/R counts above. macOS mount refusal is expected unsupported production behavior, not native success.

## 12. C09 retained identities, lifetime and drop order

`d7_custody_c09_replacement_matrix`: object set = publication-root ancestor,profile,cases,case,attempts,attempt,guard,each5 evidence final,own admission/preflight,own start =14 objects. Inject same-byte/new-inode replacement at {ENTER H_before_event,ENTER H_after_event,START H_before,BEGIN H_begin}; omit own start before it exists (first3 checkpoints) and own admission/preflight at H_before_event (not yet published), yielding14×4−4=52 valid cases per branch. Every check fails at the selected checkpoint, no later P/spy, immutable old object retained; mutation is a simulated hostile actor, never sanctioned cleanup. Guards/directories validate against held parent; evidence checks compare both path and held file. Add Full target lock/reference snapshots×same4 checkpoints=8, with target-free Quick target calls0.

`d7_custody_c09_owner_changes`: each branch × {ProcessStamp mismatch,context-instance mismatch with metadata equal,actual store/context pointer mismatch,thread/runtime-owner mismatch,resource lease lost,user revoked,source approval changed} × {before ENTER,START,BEGIN}=42. Early no mutation, later completed files preserved; no HE/setup spy. Actual fork is native-only companion, not a fictional PID field or parsed event value.

`d7_custody_c09_transfer_order`: success and failures at {before plan-validation,plan-validation,transfer,after transfer before setup,setup partial,unused driver drop,update poison} plus success=8 controls per branch. Exact observed order on successful boundary: final checks→accounted validate_plan_binding spy1→transfer same context/input/plan/accounting+leases→close attempt/evidence handles/release store borrow→preallocation live recheck→setup spy1. No candidate geometry rebuild. Failure before transfer closes attempt handles through RAII, no lease relinquishment before any already-created crypto drops. In partial setup/poison/unused-driver cases, nested session/refresher/facade drops before keys/client/bootstrap material, then resource/user leases and accounting. Closed internal counters record each unique owner drop once, no duplication/clone. Quick retained end-publisher frame keeps its own required attempt handles until end/no-end continuation completes; do not accidentally apply Full handle release semantics to Quick.

`d7_custody_c09_lifetime_compile`: six compile cases, four fail (drop store before admission/start; replace borrowed context; send live owner to thread; deserialize/clone live owner), two compile (shorter store lifetime than independent longer input/context after checked transfer; consume driver without `'c:'p` constraint). These are future compile tests, no compiler invoked here. A compile failure alone is not a resource lease runtime observation.

## 13. C10 Quick minimal lifecycle

`d7_custody_c10_quick_success`: Q_LINKED_FIX1 inspection verifies all3 event links and sidecar identities; separate Q8 live-order control ENTER→START→BEGIN_CHECK_ONLY→two update spies→END. Actual crypto0, update spies2, target reader0, Full tree open/create/write0, P8/56 Ops, D16 with Q_EMPTY, R51 including BEGIN, G3/g3. Outcome complete is operational only, not full prospective success or Quick268 completeness.

`d7_custody_c10_quick_failure`: after durable start, setup failure, first-update failure, second-update failure, post-second-update live-recheck failure and endpoint failure, five cases; end failure outcome reflects exact completed count0/0/1/2/2, never fabricated metrics. Same live caller owns continuation; at most one Q7 publication. P total8 if end publication succeeds, even for setup failure; no driver retry. Process crash/lost origin before end=two extra cases with P7, end absent, incomplete. End publication postlink fault preserves exact uncertain final; no replacement/end retry within this attempt.

`d7_custody_c10_quick_end_authority`: caller lost,parsed preflight only,parsed start only,recovered all events,foreign context,foreign process,already-existing end,missing start,missing preflight,missing one evidence (five variants) =>14 cases. End P0 in each; successful past files unchanged. Existing canonical end may be inspected but never authorizes another end. No recovery-generated success/failure.

`d7_custody_c10_quick_initial_only`: repeated Initial with A empty/temp-only/k1..5 sidecars/preflight/start/end =10 states (empty1,temp1,sidecar prefixes5,events3), each refuses named-attempt reuse/P0. Explicit Quick Retry requests with absent/present start (2) reject before namespace mutation; new attempt name cannot bypass stable q-registration slot. A separately explicitly selected campaign within already-existing authority is outside this retry protocol; no automatic rotation/approval ceremony.

`d7_custody_c10_full_quick_separation`: all-Quick bundle into Full plus each five mixed gate positions=6 reject; all-Full bundle into Quick plus each five mixed positions=6 reject; equal registration/input hashes in both branches do not promote scope. Each assembler call1, production gate construction0, mutation0. Eight exact branch fixture filters from accepted FIX1 remain as originally frozen, including `full_missing_verified_target_rejected`, `quick_without_target_canonical_inputs`, `quick_bundle_cannot_enter_full_admission`, `full_target_threshold_placeholders_rejected`, `equal_registration_input_no_scope_promotion`, `quick_exact_consumption_bound`, `full_independent_target_receipt_mismatch_rejected`, `full_frozen_bytes_and_branch_evidence_distinct`.

## 14. Execution boundary, report and remaining concerns

Only read-only source/document inspection and this inventory were performed. No Cargo, tests, `--list`, runtime, generator, native observer, SSH, HE, real parameter/P1/materialization/registry calls, subagents, source/public-doc changes, commit or push. Existing tests were read, not run. Exact future filters and parameter counts above are test obligations, not observed pass counts. Logical I/O counts never claim exact native syscalls or memory capacity.

Three concrete source/fixture dependencies remain explicitly visible: the post-check pathname-mutation preservation prerequisite in §7 FIX1 is unresolved by the pinned helpers; retained-FD/exclusive-attempt/multi-file recovery seams do not yet exist; and the accepted new Full evidence fixtures and old Full admission goldens are not a single linked bundle. Pure ordering tests can use F8 and actual codec composition can use FULL_CODEC_OLD separately, but combined coverage must first pin an independent coherent Full fixture. These are implementation/fixture duties, not architecture changes or a new approval requirement. Root review should decide whether that separate combined-fixture package precedes or accompanies protocol source; no successful combined lineage is asserted here.

Preserve all21 profiles, full2680+20+7, Quick268, original holds/security/baselines/Task5, and MIT/Apache2/BSD/ISC plus Unicode-table exception. This inventory does not complete Task54 or release the genuine gates/native lifecycle.
