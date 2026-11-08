# Task54 D7 CPU Step2 — scoped FIX2 quality rereview

Verdict: **QUALITY PASS for the scoped FIX2 source correction. Q01 ADDRESSED; Q02 ADDRESSED.** No new Critical or Important breakage was found in the ten changed files. This closes the two findings from the original whole-Step2 quality review within the explicit source-review and local-evidence boundary. It does not claim native Linux execution, current-WIP Linux CI success, Step3 completion, full Task54 completion, or scientific success. Root owns adoption.

## Scope and source identity

Reviewed the exact scope in `task-54-d7-cpu-supervisor-fix2-rereview-brief.md`, the original Q01/Q02 findings, root's FIX2 ruling, pre-edit finite expectations, complete FIX2 report and diff, current source and relevant callers, archived reviewed FIX1 source, source/review pins, command index and all thirteen commands' stdout/stderr. This is a scoped rereview, not a restart of the whole-Step2 review. No new finding about untouched code is added to this loop.

- Repository: `/Users/marconi/Documents/research/hekit`.
- Source BASE: `a29dcf6613ed05d5e1ec7b5300458e74214b5ee9`.
- Reviewed FIX1 execution HEAD: `63a440a4d02c155eba446ce5fe01b747ecdce5cd`, plus its preserved authored source.
- FIX2 execution HEAD: `6fba46c38f1cdb5be88b9bf1eeecab0f4566b534`, plus the frozen authored worktree. That committed publication/evidence state is not the identity of a finalized clean build containing this WIP.
- Original quality report: 13292 bytes, SHA256 `a644d4d30d8bf4d043ae76f812c97a737bd474d4d2965e44cea51a46e962f33d`.
- FIX2 report: 11872 bytes, SHA256 `c08890e097dca5a725e72a95b01dedfcb3e2d7910072b6acb2ffd11636000d01`.
- FIX2 diff: 22085 bytes, SHA256 `f8a7f9452dea04003e8222b6abb9e1546c881740c4ced7f2522216bc0c8cd0d8`.
- FIX2 original archive: 2291961 bytes, SHA256 `2352f982ee7b6c70ddb269cf95ddd65e3a6015b3e75261d0e7ad6614dec995da`, 173 members.

Independently verified all 537 file-pin occurrences in FIX2 source/input pins, all 14 review pins, and all 52 command-log pins against current bytes and SHA256. Independently checked all 71 archived authored sources against FIX1 pins. Reconstructed a unified diff entirely in memory from those archived bytes and current files: it exactly equals the frozen 22085-byte FIX2 diff, and its ten changed paths exactly equal `changed_source_files`. The other 61 authored files are byte-identical to reviewed FIX1. Historical/current hash differences therefore reflect the documented correction. The 453 selected-input count remains a selection, not proof of raw physical inventory or a finalized clean build.

Reviewer actions were source/log reads, hashing and in-memory archive/diff comparisons, followed only by writing this report. No build, test, native probe, child-process experiment, signal, network, installation, Git mutation or subagent was run by this reviewer.

## Findings disposition

### 1. Q01 — Important — ADDRESSED

Original trigger: M's private alias of the P–M control socket survived L's exec. P required M's real status and both EOFs before adoption/leaf teardown, while L retained that endpoint and waited for P. The intended native1/6 success sequencing could not complete.

The correction is on the actual acquisition path. `handshake.rs:173` defines `private_dup` using safe atomic `fcntl_dupfd_cloexec(fd, 3)`; `child_channel` uses it for both private standard-descriptor aliases at lines 193–194. `ParentGuard::lock_clone` at line 137, `OwnedChild::spawn` preparation in `native.rs:197`, and the two retained inner-monitor aliases in `probe_adoption.rs:920` and `probe_driver.rs:1067` use the same helper. No ordinary inheritable `rustix::io::dup` or private `File::try_clone` remains in this supervisor path.

The same open file description is retained: this duplicates the existing handle and does not reopen the lock. The original fd0/fd1 are untouched. `OwnedChild::spawn` still explicitly maps the prepared lock and new socket through `Stdio`; L therefore receives its intended fd0/fd1, while the extra old P–M private alias is closed by exec. The existing authenticated ancillary receive still requests `CMSG_CLOEXEC`. There is no non-atomic duplicate-then-flag interval, unsafe block, new dependency or topology change.

The native adoption caller continues to require M's actual successful status, stderr EOF and control EOF, then actual reap, scheduler retirement and authenticated L adoption. Only after those steps does `probe_adoption.rs:894` require the retained L pidfd to remain live; line 905 records `middle-eof-while-leaf-live`, before either scenario's L TERM/Stop. Thus the added assertion checks the particular sequencing that the original leak prevented and does not replace EOF with status. An assertion failure retains the existing outer custody/error-cleanup route.

The new actual-helper test, `handshake.rs:255`, checks distinct alias descriptors, CLOEXEC on aliases and initial socket endpoints, unchanged original FD flags, shared OFD `O_NONBLOCK` status and byte transfer. This is relevant coverage of the real duplication helper. It is Linux-only and **compiled, not run**; neither it nor the source trace is a measured native inheritance/flock result. That limitation is explicitly permitted by the ruling and remains a later Linux validation gate. The implemented remedy and authored coverage address Q01 within this review's boundary.

### 2. Q02 — Important — ADDRESSED

Original trigger: a live control-dependent child closed its framed output cleanly with an empty parent output queue. `Channel::turn` recorded EOF, but no failure occurred until an unrelated trigger; the old native3 driver supplied a manual failure and bypassed the missing production transition.

The actual `Monitor::pump` now observes channel/raw EOF and acquired child status, then constructs `ControlObservation` from its own retained kind, phase, accepted result and actual pending-write state (`monitor.rs:529`). On unexpected live framed EOF it obtains the common-clock observation time, calls `Scheduler::observe_control`, executes the returned owned actions and only then returns `ControlLost`. `observe_control` (`schedule.rs:292`) requires ledger membership and enters the existing sticky ProtocolFailure stop, emitting TERM for the owned ledger without retiring any token. `Monitor::actions` still attempts the complete action batch and starts the five-second grace only after batch completion; Tick/KILL and real drain/reap remain the existing cleanup path. Signal/clock failures continue to retain unresolved custody and the FIX1 clock latch/fallback.

The production controller handles `ControlLost`, observes the scheduler reason and leaves its normal loop for `finish`/`cleanup_turn`. Normal native turns reject the notification and use their existing error cleanup. No caller must inject `fail()` to initiate this transition. Collected IPC is discarded once stopping begins, so a pending message cannot expand work after the detected loss; any already-observed exits remain available to stopping drain/reap independently of their normal notification flag.

The finite decision in `protocol.rs:496` matches the recorded expectation: phases 0/1/3 and Worker/Python phase4 require live framed control; Git phase4 is raw stdout; Python phase5 requires an accepted result and no pending write; Worker phase5 requires no pending write; other live EOF states fail. Already-observed terminal status continues required protocol/drain handling. EOF itself establishes neither terminal status nor reap.

These exceptions have constrained reachable producers. `BrokerDispatch::result` (`broker_dispatch.rs:85`) verifies Python kind, phase4, payload bounds, no previous result and no outstanding nested helper requested by that Python, then stores the result and enters phase5. Later frames fail the caller's phase check. Native `stop_worker` (`probe_driver.rs:579`) requires the completed report, no active broker helpers and an actual phase4 Worker, queues the empty Stop, then enters phase5. The pending-write condition prevents an unflushed Stop from granting the EOF exemption. No production scientific-success terminal state was introduced.

I specifically traced frame/EOF ordering: `Channel::turn` (`protocol.rs:353`) returns an already-decoded frame before another read, and otherwise reads only the bytes needed for the current frame. It does not consume a completed terminal-result frame and a subsequent zero-byte EOF in the same turn. The controller/native caller can therefore validate the Python result and enter phase5 before the next pump observes its EOF. The new check does not race its own delayed result dispatch in that path.

Native3 now consumes the real pre-TERM live EOF observation, requires an empty queued write and verifies that ProtocolFailure has already started. It records stderr EOF as an observed boolean rather than assuming simultaneity. Only afterward does it queue the intentionally unflushed Stop. It uses stopping drain, checks that the write remains pending, records a separate current live-status flag, and measures the bounded stopping wait with actual both-EOF state before eventual status31 and real reap. `waited_both` requires that the intended both-EOF/queued-output wait was actually seen. There is no manual `monitor.fail` in this case. The pre-TERM and later pre-wait observations are truthfully separate, including when TERM has already ended the child.

The new test at `tests.rs:772` obtains clean EOF and empty output state through a real local UnixStream and actual `Channel::turn`, then calls the same observation/reducer path as production. It asserts all-owned TERM, retained tokens, sticky reason/no duplicate TERM, and exact acknowledgment-plus-five-seconds KILL behavior; controls cover raw Git, accepted terminal phases, no EOF, terminal status, bad terminal states and an unknown token. This meaningfully covers the shared decision without manually supplying a failure. Its `status=None` is an inert input, not a live Linux pidfd observation. The implementation and covering test address Q02 without claiming native execution.

## New-breakage review and retained gates

Critical: none found in FIX2. Important: none found in FIX2. Minor: none raised in this scoped loop. No deferred untouched-code observation is added.

Reviewed all changed hunks in `broker_dispatch.rs`, `controller.rs`, `handshake.rs`, `monitor.rs`, `native.rs`, `probe_adoption.rs`, `probe_driver.rs`, `protocol.rs`, `schedule.rs` and `tests.rs`, with their actual descriptor, frame-dispatch, cleanup and adoption callers. The diff leaves F01 history streaming/caps and F02 final-sample gating intact. F03's actual clock-fault path retains its same-OFD inner lock with the corrected private duplicate; its shared latch, action completion and cleanup behavior are unchanged. F04's corrected pinned-v3 account description is unaffected.

All thirteen indexed commands have terminal results; three are preserved compile REDs for missing APIs, not behavioral failures. Read logs support the reported one focused Mac EOF test, 43 operational and 30 supervisor tests, Linux tests/bins compilation, strict Linux/Mac Clippy, final fmt, and author's isolated patch/all-71-source check. Q01's Linux-only descriptor test did not execute on Mac. Reviewer independently verified identities and source correspondence rather than rerunning those commands.

Native16, Linux inherited-descriptor/pidfd/subreaper/clock execution, supported-filesystem history replay and a finalized clean Linux receipt remain unexecuted or future gates. Step3 Input0/Reviewed2 reader I/O and N01–N07/gates/science remain intentionally unconnected. Existing public-preview/committed-source CI evidence is separate from this frozen authored WIP. The scope is **QUALITY PASS**, with these boundaries preserved.
