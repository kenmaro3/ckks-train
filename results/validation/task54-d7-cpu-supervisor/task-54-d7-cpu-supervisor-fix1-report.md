# Task54 D7 CPU Step2 — FIX round 1 of 5

Status: **DONE_WITH_CONCERNS — F01–F04 corrected in source/documentation and submitted for scoped independent rereview.** No adoption/native PASS/Task54/science completion is asserted. Native Linux execution, genuine filesystem history replay, clock fault syscalls and real child teardown remain unexecuted.

Original source BASE remains `a29dcf6613ed05d5e1ec7b5300458e74214b5ee9`. Original reviewed HEAD was `b8bead4e13e6846a9de05640e6a057fe42982b6c`; FIX1 actual HEAD is recorded in fix1-pins.json (root publication-only commits are excluded). Binding accepted findings are [SPEC FAIL](task-54-d7-cpu-supervisor-spec-review.md), SHA256 `c427b13e30bcca5a3e0d96bcc76eac476c5606d38af19dff54be2bdedd6ffa4e`, and [root FIX1 ruling](task-54-d7-cpu-supervisor-fix1-ruling.md). Root accepted the finite inner/outer F03 configuration before its source edits.

## Immutable baseline and correction artifacts

Before any correction edits, all 68 reviewed source files and review metadata were archived: [fix1-original.tar.gz](task-54-d7-cpu-supervisor-fix1-original.tar.gz),155 files,1107529bytes,SHA256 `a07d68cfd1903f2da30dd544ac3aba15808b0526942b86e3e386cffe6b9e1935`; [archive pin](task-54-d7-cpu-supervisor-fix1-original-pin.json). The original report/source.diff/pins/review-pins/178-command index were not rewritten; those hashes refer to their reviewed version retained in this archive. New source bytes necessarily differ from original source pins and are pinned separately below.

- [Fix-only complete diff](task-54-d7-cpu-supervisor-fix1.diff), against exact archived reviewed files, including three new private modules.
- [Current source and selected-input pins](task-54-d7-cpu-supervisor-fix1-pins.json), [changed source paths](task-54-d7-cpu-supervisor-fix1-source-paths.txt).
- [FIX1 exact command/log inventory](task-54-d7-cpu-supervisor-fix1-commands.json), [readable full-log links](task-54-d7-cpu-supervisor-fix1-commands.md). Original178 command directories/logs remain unchanged. Each added validation command has a unique directory and terminal result.
- [Pre-edit reasoning/test expectations](task-54-d7-cpu-supervisor-fix1-notes.md), [rereview metadata pins](task-54-d7-cpu-supervisor-fix1-review-pins.json).

No source commit/staging/push, server/network/native-supervisor/HE run, publication, billing action, target installation or broad workspace test was performed. Root's publication activity is separate. No new dependency, feature, unsafe, selector, wire schema, process hierarchy or scientific matrix change is introduced by FIX1.

## F01 — sequential history replay

`probe_index::resolve_active` no longer charges a cumulative byte count or directory-entry count against lifetime history. Per-observation8MiB and command64KiB bounds,128-byte names, all original/copy bytes and retained source joins remain. Newly added operation storage/capacity limits are unchanged. Increasing/deleting history or accepting only a prefix is not used.

The actual safe rustix Dir iterator streams one probe/check pair at a time through shared `history::visit`; no all-history Vec/set is constructed. Each check group has a fixed16-bit allowed-name mask; each pair has a two-bit mask with exactly command+observations, or command alone for the exact live active exception. Empty check group, duplicate/unknown local name, malformed name, iterator/visitor error or missing pair rejects. Original and indexed observation bytes are still compared directly, not only hashed. Simultaneous live buffers remain bounded per pair; parsed events/values and both8MiB copies can coexist, so this is not an8MiB total-RSS claim.

Before changing the iterator, notes recorded the duplicate/mutation reasoning: supported local filesystems have unique directory names; one unchanged directory traversal does not require a growing outer-name set. The retained descriptor/name chain and device/inode remain checked. Each directory's device/inode, size/link count and mtime/ctime with nanoseconds are compared before/during/after enumeration; the pair stamp also spans content validation. An extra exact pair rescan rejects unknown additions. Cached rustix1.1.6 Linux `Dir::read_from` opens `.` as a fresh descriptor, so repeated scans do not inherit an exhausted directory offset. The ordinary filesystem guarantees and retained/metadata checks are the basis for outer uniqueness; advisory account flock is not protection against arbitrary same-UID mutation, privileged rollback or races indistinguishable within filesystem timestamp resolution. No new malicious-same-UID guarantee is claimed. Full replay remains O(history), with no constant-latency or hard kernel wall-time promise.

Coverage calls the same visitor and per-record length validator for5000 outer names and80,000MiB cumulative supplied replay lengths, with bounded live name storage, plus incomplete/duplicate/unknown/error cases and independent stamp-change tuples. It does not read80,000MiB, create5000 directories, execute Linux getdents or establish native filesystem behavior. All actual retained filesystem readers are cross-compiled.

## F02 — final sample is an acceptance gate

Monitor::sample now uses the shared `completion::assessment` for valid/status. Both production Controller::finish_records and native Session::complete call Monitor::final_sample, which feeds the actual returned SampleValues or error to `completion::final_sample_result`. Only phase4,valid=true,status=teardown with no prior unresolved state succeeds. Missing RSS, parse failure, backward/late clock window, threshold+1, errors before a returned SampleValues or an existing unresolved state latch `unresolved=true`. A subsequent valid sample cannot clear it. The original scheduler stop reason is not overwritten to hide this failure.

This gate precedes production end/closed marker and native original/account observation completion. Existing hold/cleanup paths consume the sticky failure. The test obtains missing RSS from the actual malformed VmRSS parser, derives statuses through the same assessment as Monitor, and invokes the actual final gate. It covers late final completion after the specified fixed window, backward time, threshold+1, exact threshold/window equality and earlier error remaining sticky after a good sample. These are shared-decision tests, not native proc or clock fault experiments. Wire records can still represent invalid observations; wire parsing itself does not grant completion.

## F03 — native clock faults use actual cleanup source

The existing `monotonic-common-origin` check8 injects exactly one failed checked clock read before any TERM. The existing `rss-term-grace-kill` check10 retains real owned SIGSTOP/StateT and threshold-event injection, then injects exactly one read failure at the TERM batch-completion clock acquisition after actual TERM issuance timestamps. No17th selector exists. Production monitors have no fault object; only the closed native probe branch arms this private source fault. `clock_fault::checked` is the actual Monitor elapsed function and preserves checked common-origin conversion and sticky ClockFailureLatch. The next real source read can succeed, while invalidity stays latched.

At the boundary, one already owned/registered Tracked child moves into a retained inner Monitor after fallible setup. No new PID lookup, child spawn, duplicated Child ownership or process hierarchy is introduced. A private checked scheduler metadata fork preserves the original fixed-window state and registered token; it does not create signal authority. Outer Monitor retains the original token. The probe-only borrowed sampling path requires both monitors to be probes, same process token/common clock origin/lock inode, exactly bounded inner children and existing outer token membership. It performs the real census, acquired pidfd/proc identity/RSS and fixed due sample including inner-owned children; no child is hidden from the census. The outer retires its token only after inner actual required EOFs, status and reap.

Production Controller, ordinary Session cleanup and the native inner experiment share **Monitor::cleanup_turn**. It consumes the same real sticky loss/fallback decision, tries every legal owned TERM, arms fresh checked Instant grace once after those attempts, issues KILL when due, drains and reaps, and parks on the same stopping wait. Unexpected cleanup I/O/action/reap failures are sticky and cannot be accepted as the deliberate fault. Clock recovery does not switch the failed inner back to valid common-clock scheduling or reset grace. Instant remains another platform clock API, not an independent clock guarantee.

The source requires the expected side of TERM: pre-TERM injection has no reason/signal yet; post-TERM injection must have actual issued TERM, RssStop and a still-unacknowledged completion origin. Recovered clock plus sticky latch is explicitly checked. Check10 must observe actual SIGKILL status19 and at least5s between the last real TERM timestamp and KILL; check8 expects actual TERM status31 without KILL. Failed reads are recorded with elapsed=None (`injected-clock-read-invalid`); explicit fault labels are `pre-term-clock-read-failure-injected` or `post-term-clock-read-failure-injected`. They are source injections, never a claim of kernel clock outage. The shared fallback start is observed after the actual TERM/Instant boundary; its outer timestamp is not relabeled as the Instant origin.

The invalid inner, its clock latch and original lock remain retained through outer final record publication. Outer completion means actual known child custody is exited/drained/reaped plus the explicitly labeled injection observation is closed; it does not resolve the inner failure or mean native PASS. Any unconfirmed/unexpected experiment error enters ordinary outer failure cleanup/holding. The existing pending account record continues blocking unexplained interrupted runs.

Pure coverage exercises the actual checked fault/latch function and shared `FallbackClock::should_start`, including pre/post-TERM arming, one Err/no valid value, recovery with sticky latch, single fresh arm, absent/backward duration refusal and exact5s equality. Actual pidfd, inner/outer borrowing, signal syscall timing and cleanup source are compiled for Linux, not executed. Independent scoped review must assess that lifetime/control connection before native release.

## F04 — corrected account-origin description

The earlier report sentence “Passwd EUID mapping locates the account” is incorrect and superseded here. `MinimalOrigin::acquire` obtains the **selected account path and identity from the whole-file-pinned v3 approval**, checks the selected UID against actual real/effective IDs, and opens/rechecks the retained directory chain and identity. There is no passwd lookup in this acquisition path. Account anchor discovery is not physical-host/namespace proof; those independently selected sources/reviews remain necessary. No account acquisition code or trust boundary changed for this wording correction.

## Tests and exact failures

Expectations and iterator reasoning preceded changed implementations in fix1-notes.md. Original reviewed source/metadata snapshot preceded all edits. Added tests are `operational_fix1_history_stream_bounds`, `operational_fix1_directory_stamp`, `operational_fix1_final_sample_gate`, and `operational_fix1_clock_fault_cleanup_path`.

- `20261108T005808.840689Z-fix1-history-final-red`:101, missing history/completion imports (compile RED, no behavioral claim).
- `005928.831561Z-fix1-history-final-green1`:0,3 tests.
- `010030.372988Z-fix1-linux-check1`:0, lib/bins cross-target check.
- `010218.327672Z-fix1-clock-fault-red`:101, missing clock_fault/shared start API (compile RED).
- `010333.597983Z-fix1-clock-fault-green1`:101,E0382 moved Result in two closures; retained. Replaced with a match, no weakened test.
- `010457.057053Z-fix1-clock-fault-green2`:0,1 test.
- `010515.511823Z-fix1-linux-check2`:0, lib/bins cross-target check.
- `010617.147818Z-fix1-clippy-linux1`:101, obsolete unused stop wrapper after common cleanup extraction; wrapper removed, no dead-code allowance added.
- `010648.510908Z-fix1-clippy-linux2`:0; subsequent narrow expected-TERM-side assertions justified final recheck.
- `010750.836120Z-fix1-operational-covering`:0,42 passed, including the four additions and previously authorized real Mac socket-pair boundary test without children.
- `010820.612444Z-fix1-supervisor-covering`:0,30 passed.
- `010829.319141Z-fix1-clippy-linux-final`:0, offline locked bench all-features/all-targets with `-D warnings`, source before the final narrow pre-dispatch custody cleanup correction.

Final Mac strict Clippy, fmt and fix-patch validation terminal results are recorded in the linked FIX1 command index and appended below when collected. No original logs/results were overwritten or rerun to recreate RED. No new standalone post-test/pre-implementation RED tar is claimed; the pre-edit reviewed archive and exact RED command outputs are retained, and added test source is in the fix diff. The two compile REDs do not establish native behavioral failure. Unchanged Python/source recipe/SHA tests were not repeated without a change requiring them.

Step3 Input0/Reviewed2 acquisition and N01–N07 actual reader I/O, five gates/admission and later science remain exactly the original explicit gaps. Source has not entered the public research preview. All registered profiles/Full/Quick/seeds remain unchanged follow-on work.


### Final frozen verification addendum

Final source includes the narrow error interval after child custody moves into the inner but before intended fault dispatch: Session error cleanup initiates owned stopping when needed before the common cleanup-turn loop. Any such unexpected error remains unresolved. This Linux-only path is source-reviewed/compiled, not behaviorally run here.

- Mac strict Clippy `20261108T010840.032797Z-fix1-clippy-mac-final`: exit0, all-features/all-targets/-D warnings. The later correction is Linux-only; Mac was not rerun for that excluded branch.
- Final Linux strict Clippy `20261108T011131.653172Z-fix1-clippy-linux-final2`: exit0, all-features/all-targets/-D warnings after that correction.
- Final fmt `20261108T011205.880862Z-fix1-format-check-final2`: exit0.
- Final patch verification `20261108T011217.191430Z-fix1-patch-check-final`: exit0. Verified all68 archived original source hashes, checked/applied the fix-only patch in a temporary directory, and verified all71 resulting authored source hashes. No main-worktree index/source mutation from this check.

The frozen fix diff changes10source paths,52003bytes,SHA256 `b2ecff31ccf6aae8d84b029e4c24fac2a512a248a855c6636ec80d44051bdc7c`. Total authored source files71; selected-if-tracked453/512, not raw physical inventory/clean-build proof. Original public design and original review artifacts remain separate pinned inputs.

FIX1 command inventory: 20 terminal commands, 4 nonzero exits, no missing result. All four nonzero commands are described above.
