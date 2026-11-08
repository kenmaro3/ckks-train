# Task54 D7 CPU Step2 — FIX round 2 of 5

Status: **DONE_WITH_CONCERNS — Q01/Q02 corrected and frozen for scoped quality rereview.** No source adoption, native16 PASS, finalized Linux clean-build identity, Task54 completion or scientific success is asserted. No commit/push, server/network/native16/process experiment, installation, publication or billing action was performed by this implementer.

Binding findings: [whole-Step2 quality review](task-54-d7-cpu-supervisor-quality-review.md),13292bytes,SHA256 `a644d4d30d8bf4d043ae76f812c97a737bd474d4d2965e44cea51a46e962f33d`, and [FIX2 ruling](task-54-d7-cpu-supervisor-fix2-ruling.md). Source BASE stays `a29dcf6613ed05d5e1ec7b5300458e74214b5ee9`; actual execution HEAD is pinned in fix2-pins.json. Public preview/CI at the committed HEAD is separate from this uncommitted source.

## Artifacts and preserved history

Before edits, all71 FIX1 authored source files plus review metadata were archived in [fix2-original.tar.gz](task-54-d7-cpu-supervisor-fix2-original.tar.gz):173files,2291961bytes,SHA256 `2352f982ee7b6c70ddb269cf95ddd65e3a6015b3e75261d0e7ad6614dec995da`. [Archive pin](task-54-d7-cpu-supervisor-fix2-original-pin.json). Original and FIX1 reports, diffs, source/review pins, archives and all198 earlier command directories remain unchanged. Their source hashes refer to their preserved reviewed versions, not the later corrected worktree.

- [FIX2-only diff](task-54-d7-cpu-supervisor-fix2.diff), against the exact archived FIX1 source:10files,22085bytes,SHA256 `f8a7f9452dea04003e8222b6abb9e1546c881740c4ced7f2522216bc0c8cd0d8`.
- [Current source/input pins](task-54-d7-cpu-supervisor-fix2-pins.json), [changed paths](task-54-d7-cpu-supervisor-fix2-source-paths.txt), [review metadata pins](task-54-d7-cpu-supervisor-fix2-review-pins.json).
- [Exact argv, exits and full logs](task-54-d7-cpu-supervisor-fix2-commands.md), [machine-readable log pins](task-54-d7-cpu-supervisor-fix2-commands.json).
- [Pre-edit finite criteria and expectations](task-54-d7-cpu-supervisor-fix2-notes.md), [Q02 pre-implementation RED source snapshot](task-54-d7-cpu-supervisor-fix2-control-red-source.tar.gz). The latter includes the already implemented Q01 and both new test sources; it is not represented as Q01's missing-helper source snapshot.

Author source remains71files; selected-if-authored-files-tracked453/512, not raw physical inventory equality or a clean finalized build. No file-count limit, schema, selector, process hierarchy, dependency/feature, unsafe, origin trust or scientific matrix was changed. Root-owned operational design and unrelated work remain untouched.

## Q01 — atomic private descriptor aliases

`handshake::private_dup` uses safe `rustix::io::fcntl_dupfd_cloexec(fd,3)`. Both inherited fd0/fd1 private aliases use it, as do ParentGuard lock cloning, OwnedChild spawn preparation and both retained native inner-monitor lock aliases. The private alias is immediately CLOEXEC while sharing the original underlying open file description. Original standard descriptors remain intact; explicit Stdio maps the intended lock and channel to child fd0/fd1, preserving the same lock OFD.

Audit of the fixed P→M→L path: no ordinary inheritable `rustix::io::dup` or private File::try_clone remains in the supervisor. Proc/root/executable opens use CLOEXEC; authenticated ancillary receives still use CMSG_CLOEXEC; standard-library UnixStream pair endpoints are owned sockets and the new Linux test explicitly checks their CLOEXEC flags. Stdio mapping and the exact typed ancillary boundaries remain the intentional transfer mechanisms. The original fd1 is replaced by L's own control socket; the old private P–M duplicate is now closed at L exec. This removes the source leak without relaxing M EOF/drain/reap.

After M's real terminal status, stderr/control EOF and reap, adoption confirms L's actual PPid and acquired identity; the native driver additionally requires L's acquired pidfd to remain live before any L Stop/TERM and records `middle-eof-while-leaf-live`. The existing native1/6 sequencing and subsequent custody handling remain. No such native observation has been executed here.

`operational_fix2_private_aliases_are_cloexec` is **Linux-only, authored and cross-compiled, NOT RUN**. It calls the actual helper on owned UnixStream descriptors, checks distinct alias FDs and CLOEXEC, unchanged original descriptor flags, the shared OFD O_NONBLOCK status through one alias, and one-byte transfer. It creates no child and performs no lock provisioning/handshake/native16 run. This is a bounded descriptor-only unit test eligible for later separately authorized Linux CI after source review; compiling it is not FD flag or inheritance measurement. It does not independently prove real lock/flock inheritance.

## Q02 — live framed EOF triggers actual stopping

Monitor::pump constructs `protocol::ControlObservation` from the actual Tracked kind/phase, accepted result, queued-write state, channel/raw EOF and acquired child status. Unexpected live framed EOF goes through the shared `Scheduler::observe_control`, which checks token membership and enters ProtocolFailure with owned TERM actions. Monitor executes those actions before returning ControlLost; it discards collected IPC rather than dispatching new work after the stop. Controller leaves its normal loop and enters the existing shared TERM/grace/KILL/drain/reap/hold lifecycle. Native normal turn also fails if this unexpected notification appears. No caller must manually inject `fail()` to produce the transition.

EOF does not imply process exit or reap, and no owned token is removed by observe_control. A signal/action error retains custody and is handled by the established cleanup path. Truncated framing and queued-write errors retain their existing rejection paths. The finite EOF criteria, recorded before edits, are:

| Actual state | Live EOF decision |
|---|---|
| Handshake phases0/1/3, any kind | Required framed control; ProtocolFailure |
| Worker phase4, including informational native ExitAck | Required framed control; ProtocolFailure |
| Python phase4, even if an unaccepted result is suggested | Required framed control; ProtocolFailure |
| Git phase4 | Explicit raw stdout; continue required status/stderr drain |
| Python phase5 | Allowed only with its validated final result and no queued write |
| Worker phase5 | Native Stop path only; allowed only once queued Stop has actually flushed |
| Other/wrong-kind or pending-write terminal state | ProtocolFailure |
| Actual terminal status already observed | Continue protocol-specific result/status/drain checks; EOF still does not reap |

Python enters phase5 only in BrokerDispatch::result after existing kind/order/payload checks and after verifying no nested helper remains requested by that Python. Later requests/frames fail phase checks. Native Worker enters phase5 only in stop_worker after report/broker-completion/Worker phase4 checks and queuing its empty terminal Stop; pending bytes prevent the EOF exception until they are flushed. Step2 production has no scientific terminal-success phase, so none was added. Phase5 is an internal state, not a new wire tag or authority.

The new `operational_fix2_live_control_eof_stops_owned_children` test obtains a real clean EOF and empty output queue from local UnixStream::pair peer write-shutdown through actual Channel::turn. It passes the same ControlObservation into the same observe_control used by pump, without calling fail or supplying a failure event. Independent expected actions assert TERM for registered W/P/G, retained live tokens, sticky ProtocolFailure, no repeat TERM and KILL only after the actual-batch acknowledgment plus5s. Controls cover no EOF, observed terminal status, raw Git, validated Python terminal result, flushed native Stop, and premature/pending/wrong-kind states; an unknown token is rejected. Its live pidfd `status=None` is an inert test input: this is real socket transport plus pure shared decision/scheduler coverage, not a real child-process experiment.

Native check3 now observes ControlLost's actual pre-TERM live-pidfd EOF snapshot, including whether stderr EOF was already observed then and that no write was queued. The notification timestamp is captured after observation and before automatic TERM. It then queues an intentionally unflushed Stop and drains to both EOFs before measuring the same real stopping wait. The record includes a separate pre-wait live/status flag: TERM may already have exited the child, so it no longer incorrectly requires or claims simultaneous both-EOF+live observation after automatic signaling. Actual status31 and required drain/reap remain mandatory. The case calls no manual monitor.fail, adds no timeout/selector/fault type and does not count queued Stop as a sent protocol message. Native execution remains pending.

## Exact validation and limits

All Cargo commands are offline/locked, using the already installed targets. Test expectations preceded the implementations. No behavioral RED is inferred from a compile error:

- `20261108T013001.869615Z-fix2-cloexec-red`: exit101,E0425 missing private_dup in the Linux-only test. The test did not run.
- `013033.058252Z-fix2-cloexec-compile-green`: exit0, Linux `cargo check --tests`; compile-only, including the descriptor test.
- `013108.622485Z-fix2-control-eof-red`: exit101, missing ControlObservation/observe_control APIs.
- `013208.864687Z-fix2-control-eof-transport-red`: exit101, same missing APIs after adding actual socket EOF acquisition to the test before implementation. Original prior RED is retained, not replaced.
- `013312.427278Z-fix2-control-eof-green1`: exit0,1 executed Mac test of actual socket EOF/shared transition.
- `013335.655010Z-fix2-linux-check1`: exit0,Linux tests/bins compilation.
- `013506.347674Z-fix2-operational-covering`: exit0,43 tests passed, including all four FIX1 tests and the new Q02 case. Q01 is cfg-Linux and is not among these43.
- `013546.108767Z-fix2-supervisor-covering`: exit0,30 tests passed.
- `013555.085856Z-fix2-clippy-linux-final`: exit0,bench all-features/all-targets `-D warnings`, including compilation of the Linux-only descriptor test.
- `013620.223794Z-fix2-clippy-mac-final`: exit0,bench all-features/all-targets `-D warnings`.

Final fmt and isolated patch/source-pin verification appear in the command inventory and final addendum below. Full stdout/stderr, all failures and terminal exits are retained. Earlier Python/SHA/source-recipe checks remain their earlier scoped evidence; these unchanged components were not rerun to inflate coverage.

F01 history streaming/caps, F02 sticky final sampling and F03 inner clock-fault cleanup/latch remain covered and unchanged in their semantics. F04 selected pinned v3 account anchor description remains the corrected description. Actual native inherited FD/EOF/pidfd/subreaper/clock behavior, supported-filesystem history replay and finalized clean Linux receipt remain future gates. Input0/Reviewed2 and N01–N07 reader I/O/gates/science remain Step3+; all registered matrix follow-on work is preserved. Scoped independent review is required before adoption.


### Final frozen verification addendum

Final fmt `20261108T013653.408322Z-fix2-format-check-final` exited0. Final isolated patch verification `20261108T013832.141693Z-fix2-patch-check-final` exited0: all71 archived FIX1 source hashes matched; the FIX2 diff passed whitespace/applicability checks and was applied only to a temporary directory; all71 resulting source hashes match current pins. No index/commit/main-source mutation was used for patch verification.

FIX2 inventory has13 terminal commands,3 nonzero exits (the three preserved compile REDs above), and no missing terminal result. Source is frozen for scoped rereview; no later source edit is included implicitly.
