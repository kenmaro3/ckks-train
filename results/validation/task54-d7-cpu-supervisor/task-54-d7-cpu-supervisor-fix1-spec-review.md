# Task54 D7 CPU Step2 FIX1 scoped specification rereview

Reviewer: `/root/d7_cpu_supervisor_spec_review`. Scope is F01–F04 from the original specification review and new Critical/Important breakage in the FIX1 diff only. Read-only source/evidence review; only this report was created. No build, test rerun, server/native execution, installation, network action, commit or publication.

## Frozen package verified

- Original source BASE `a29dcf6613ed05d5e1ec7b5300458e74214b5ee9`; original reviewed HEAD `b8bead4e13e6846a9de05640e6a057fe42982b6c`; FIX1 execution HEAD `63a440a4d02c155eba446ce5fe01b747ecdce5cd`. HEAD does not contain this uncommitted source fix and is not its build identity.
- FIX1 diff: 52003 bytes, SHA256 `b2ecff31ccf6aae8d84b029e4c24fac2a512a248a855c6636ec80d44051bdc7c`. Read the entire diff, comprising ten changed/new source paths, then inspected the actual changed callers and cleanup paths.
- FIX1 report: 15142 bytes, SHA256 `da8be320f166d42ef1a341f9dedf4a591db4f96cd5c72b03c701f957e1ed1bc6`.
- FIX1 source/input pins: 102042 bytes, SHA256 `17cfbc621be5cfc1edf038e015bcce755ff3f5e31635749ce4e80d32d807b0fa`; all 537 referenced path/length/hash entries match current bytes.
- FIX1 review pins: 3199 bytes, SHA256 `a93da93bada0ff41b6b419c62aca06ace424780b62d0110dd7b858bd39b2e0fa`; all 13 entries match.
- Original archive: 1107529 bytes, SHA256 `a07d68cfd1903f2da30dd544ac3aba15808b0526942b86e3e386cffe6b9e1935`. Inspected its 155 regular members and independently verified all 68 original source files against the archived original pins. Original reviewed artifacts remain historical inputs, not current-source pins.
- FIX1 command index has 20 terminal commands and four nonzero exits. All 80 indexed command/result/stdout/stderr files match their pins; read all 20 commands and complete raw stdout/stderr. No missing terminal result.

The root FIX1 ruling and pre-edit notes were read in full. Root publication/handover edits are not added to this source review or repinned. The selected-if-tracked count 453/512 remains a selected-path calculation, not physical inventory equality or a clean finalized Linux build.

## Finding dispositions

### F01 — ADDRESSED

`study_supervisor/probe_index.rs:22–46`, `:55–142` now streams directories through `history::visit`; the cumulative lifetime byte and entry counters and all-history name vectors are gone. Each live pair still uses bounded command/observation reads and exact original/copy byte equality. The check and pair scopes use fixed local masks to reject duplicate/unknown names and incomplete pairs. The active exception requires the exact current probe/check and original/index command equality.

The directory stamp and held directory/name rechecks bracket enumeration and pair validation, including a final completed-pair rescan. The pre-edit notes explicitly state the unchanged supported-local-filesystem uniqueness assumption and metadata-granularity/same-UID limitations. This meets the root's finite disposition; it does not claim an advisory lock makes arbitrary same-user mutation impossible. Independently inspected cached rustix 1.1.6 `backend/linux_raw/fs/dir.rs:72–87`: `Dir::read_from` opens `.` into a fresh descriptor, so the rescan is not an exhausted shared directory offset.

The covering test invokes the actual streaming visitor with 5000 outer names and validates 80,000 MiB of supplied cumulative lengths without charging a lifetime budget. It also checks local malformed/incomplete/duplicate cases and iterator/visitor error propagation. This is pure traversal/decision evidence, not a real 5000-directory or 80,000-MiB filesystem replay. No new operation-storage relaxation was introduced.

### F02 — ADDRESSED

`study_supervisor/monitor.rs:354–356` now obtains the actual final sample and calls `completion::final_sample_result`. Both `controller.rs:408–411` and `probe_driver.rs:652` invoke this gate before their completion publications. The gate accepts only phase4, valid, non-threshold teardown with no prior unresolved state. Invalid RSS, late/backward sample, threshold excess, and an error before a returned sample all latch unresolved; later valid sampling cannot clear it.

Production's pre/post `finish_records` hold and native's error cleanup/hold consume this state. The original stop reason remains separate, so an intentionally stopped native child does not bypass or automatically fail the final observation. The test derives missing RSS from the malformed actual RSS parser and derives status through the same assessment used by Monitor, then exercises the shared final decision including equality boundaries and sticky failure. This addresses the original successful-closure trace without changing wire parsing into authority.

### F03 — ADDRESSED at the requested source boundary

`probe_driver.rs:1019–1209`, `:1239` connect existing checks8/10 to the exact private clock-fault route. `clock_fault.rs` feeds the same checked elapsed/latch path used by Monitor; ordinary monitors are constructed with no fault object. Check8 consumes its one failure before TERM; check10 arms it at the actual TERM batch-completion clock acquisition after successful issuance timestamps. The code verifies the intended side of TERM and verifies that a subsequent real read succeeds while the failure latch remains set.

Reviewed the ownership and cleanup connection, not just the presence of a fault helper: fallible setup precedes moving the one existing `Tracked` owner into retained `ClockExperiment`; no child spawn or discovered-PID authority is added. `fork_probe` copies only checked scheduler metadata. `probe_due_with` validates the two probe monitors' process/origin/lock/token association and borrows the actual inner child into the outer census and RSS sampling. The outer token is retired only after the inner's actual required EOFs, terminal status and reap.

`Monitor::cleanup_turn` is shared by production Controller cleanup, ordinary native cleanup and the inner fault experiment. It uses the sticky clock latch, attempts the legal TERM batch before starting one fresh Instant grace, does not restore ordinary sampling after clock recovery, and retains action/drain/reap/wait errors. Checked five-second KILL timing remains enforced. Check10 still requires actual stopped-child SIGKILL status and measured TERM-to-KILL interval; check8 requires actual TERM termination. The final narrow error interval after custody moves but before intended fault dispatch is covered by `Session::cleanup` initiating a stop when the inner has children and no reason.

The intentionally failed inner remains retained, unresolved, with its clock latch and lock through outer completion. Expected inner failure is distinguished from unexpected cleanup/partial/publication failure, which enters hold. Raw fault labels and the invalid clock event's absent elapsed value are explicit; recovery timestamps remain actual common-clock observations, and the fallback observation is not labeled as the Instant origin. The outer record closes the measured experiment's resolved child custody and does not claim that the inner became valid or that native validation passed.

The source is now reachable for the requested Linux fault experiment. Pure checked-clock/fallback tests and Linux compilation do not establish execution of the actual pidfd, timing, borrowed census, kernel EOF or failure paths. Those native facts remain unexecuted and require the later release/measurement gate.

### F04 — ADDRESSED

`task-54-d7-cpu-supervisor-fix1-report.md:48–50` explicitly supersedes the incorrect passwd wording with the selected whole-file-pinned v3 account path/identity and actual real/effective UID checks. The immutable original report is preserved. No trust-boundary source change was needed or introduced.

## New breakage and validation boundary

No new Critical or Important issue found in the FIX1 diff. No unrelated observation is used to extend this fix loop.

The raw evidence supports the stated source checks: three initial history/final-gate tests and one clock-fault test pass, followed by 42 operational and 30 supervisor covering tests. The two REDs are compile failures, not native behavioral reproductions; the first also contains an E0308 array-size inference diagnostic alongside the missing imports. The later E0382 and unused-wrapper Clippy failures remain recorded. Final Linux all-feature/all-target strict Clippy `011131.653172Z` and fmt `011205.880862Z` exit0 after the final Linux-only custody cleanup correction. Earlier Mac tests/strict Clippy do not execute that Linux-only path. Final patch-check raw output reports original68 verification and reconstructed71 equality in its temporary directory; this reviewer independently checked current pins/archive but did not rerun that mutation-capable script.

This is scoped source acceptance of F01–F04, not a native16 PASS, clean Linux build receipt, current WIP CI result, production adoption, Step3 Input0/Reviewed2/N01–N07 completion, or scientific result. Public preview HEAD and its Linux CI remain separate from this uncommitted frozen package.

SPEC PASS — scoped FIX1; F01–F04 ADDRESSED.
