# Task54 D7 CPU Step2 independent specification review

Reviewer: `/root/d7_cpu_supervisor_spec_review`. Reviewed the frozen source, not a running Linux supervisor. Read-only source review; the only file created is this report. No source edit, test rerun, commit, server/native execution, installation, network action, or publication was performed.

## Reviewed identity and evidence

- BASE `a29dcf6613ed05d5e1ec7b5300458e74214b5ee9`; submitted execution HEAD `b8bead4e13e6846a9de05640e6a057fe42982b6c`.
- Author diff: 552941 bytes, SHA256 `8c7dbbcdb8d1624b28032b63e5af90718323766c13acca3cdd575d41ff2abc46`; 68 authored source paths.
- Recursively verified all 520 path/length/hash entries in source/input pins and all 85 entries in review pins against current bytes: no mismatch. The 450 selected input count is not physical inventory equality or an accepted clean build.
- Independently checked the command index: 178 terminal commands, 63 nonzero exits, all 712 indexed command/result/stdout/stderr files match pinned lengths and hashes. Read the full raw outputs for original origin RED, v3 decoder behavioral RED, final 38 operational tests, final 30 supervisor tests, 7 Python broker tests, final Linux strict Clippy, and the final output-ignore integration test. The initial RED is E0433, not a behavioral failure. The v3 decoder RED is an executed failing assertion. Linux Clippy is cross-target compile evidence, not native execution.
- Binding review inputs: brief, report, review-focus draft, `docs/study-resource-operational-design.md`, adopted contract FIX2, and pinned origin/schema/window/account-filesystem/review-payload/source-roots/cleanup-clock/native-account-index/native-output rulings. Original proposals do not override later explicit rulings.

## Critical

None found in the reviewed scope.

## Important

### F01 — Unsupported lifetime history budgets block otherwise resolved accounts (known, independently confirmed)

Locations: `crates/ckks-train-bench/src/study_supervisor/probe_index.rs:19`, `:43`, `:66`, `:105`, `:126`, `:133`.

`resolve_active` passes one cumulative byte counter through every prior indexed command/observation and its original copy, and refuses after 64 MiB of reads. It also counts probe directories, check directories, and files together against one 4096-entry history cap. These are lifetime replay limits, not the adopted per-operation raw storage/file limits. Repeated legitimate completed probes can therefore make both production and future probes permanently refuse without any unresolved custody. For example, two retained 8 MiB observation copies per completed check already consume roughly 16 MiB of the read counter, independent of current operation storage.

Contract FIX2:54/95 defines per-operation raw storage; `task-54-d7-cpu-supervisor-native-account-index-ruling.md:5–7` requires sequential retained-history validation and preserved per-record caps. It does not adopt these cumulative history limits. Keep per-record bounds and finite, explicitly ruled directory traversal bounds; do not double-charge original/copy reads against an operation's storage cap or silently select a new lifetime budget. Root disposition and a newly frozen fix are required. The current report correctly exposes this finding rather than claiming acceptance.

### F02 — An invalid final sample can still close the native index and return success

Primary location: `crates/ckks-train-bench/src/study_supervisor/probe_driver.rs:633–641`. Shared cause: `study_supervisor/monitor.rs:314–357`; correlated production consumer: `study_supervisor/controller.rs:437–440`, `:480–498`.

Concrete source trace: after the known children have genuinely drained/exited/reaped, let the final `own_rss()` read/parse fail, or let that final sample finish after its fixed completion window while CLOCK_MONOTONIC itself remains valid. `Monitor::sample` converts the capture error to `rss=None`, computes `valid=false/status=2`, and returns `Ok(SampleValues)` after recording it. This branch does not latch `unresolved`, `publication_failed`, or clock loss. A final over-threshold sample likewise returns an otherwise valid threshold record. `Session::complete` discards the return value, then rechecks only those three flags. It can publish the original and account-copy observations and return exit 0 despite this unplanned final sampling failure.

The scheduler does not close this gap: `PhaseSample` invokes `stop`, but an existing reason (normally set by worker exit, or the deliberately tested stop) causes `stop` to return without recording the new failure. The wire encoder deliberately permits invalid samples; it is not a final-sample acceptance gate. `Controller::finish_records` discards the same value and can publish `custody_resolved=true` plus a closed marker; its post-teardown hold only sees the unchanged flags. The already fixed unknown-child and clock-loss cases do set flags, so they do not cover this distinct read/parse/window path.

This violates the native account-index ruling's explicit requirement that final sampling failures be checked after the final teardown, and the native-output ruling's prohibition on turning a failed probe invocation into success. Contract FIX2:36/97 also says invalid samples remain failures. The report's claim that a failed final sample retains lock/marker is broader than the implementation.

Fix direction: inspect final sample validity and threshold outcome explicitly, latch the adopted final-teardown failure state before any closure/index copy, and route it through retained custody/marker handling. Preserve the original stop cause separately if necessary; do not use `scheduler.reason().is_some()` alone because legitimate native scenarios deliberately stop. Add an independent negative for final RSS read/parse failure and a final late window after all children exit, covering the actual completion decision rather than only `closure_must_hold` with manually supplied booleans. This is a source-path finding, not a claim of native reproduction.

### F03 — Required native clock-loss fault path is absent from the submitted source

Locations: `crates/ckks-train-bench/src/study_supervisor/probe_driver.rs:1083–1096`, `:951–1058`; `study_supervisor/native.rs:25–27`; `study_supervisor/monitor.rs:111–118`.

`task-54-d7-cpu-supervisor-cleanup-clock-ruling.md:13` explicitly requires scoped, raw-labeled clock-failure injection in the existing common-clock / TERM-grace-KILL native scenarios, without adding a seventeenth ID. The `monotonic-common-origin` selection currently takes the ordinary default stop path. `rss-term-grace-kill` injects SIGSTOP and an RSS threshold event, but no checked-clock conversion/read failure. The actual elapsed accessor always reads/converts the native clock, and the driver has no closed fault seam that can induce the required before/after-TERM or recovery-after-latched-loss cases.

Consequently the real upper cleanup path, fallback grace, continued owned stop/reap work and unresolved outcome cannot be exercised by the authored native driver as required. Pure tests for `FallbackClock`, `ClockFailureLatch`, and an action loop establish useful local rules but do not supply this native source route. The report candidly calls common-clock-loss fault execution/review future work; candor does not satisfy the explicit source requirement.

Fix direction: add the narrowly scoped, explicitly labeled fault controls/branches under the existing two native checks, sharing the real cleanup/latching path and retaining legal outer custody where the inner monitor intentionally remains unresolved. Preserve absent invalid clock values and distinguish injection from a kernel clock outage. Authoring and compiling this route is Step2 work; executing it on Linux remains a separately authorized future gate. This finding does not request native execution now, or the unrelated Step3 readers/HE endpoints.

## Minor

### F04 — Account-origin description names an acquisition mechanism the source does not use

Location: `task-54-d7-cpu-supervisor-report.md:21`.

The report says “Passwd EUID mapping locates the account.” The reviewed `MinimalOrigin::acquire` instead takes the account path and identity from the whole-file-pinned v3 approval and checks its UID against real/effective IDs (`resource_operational/retained.rs:330–353`); the new acquisition modules contain no passwd lookup. This implementation matches the origin ruling's explicit selected-anchor design. Correct the report to describe that path and avoid suggesting an additional live account-mapping check occurred. No change to trust or account acquisition is requested by this wording finding.

## Coverage and boundaries

Traced normal/private train and reference entries into bounded RequestedWork, minimal retained approval/bootstrap acquisition, stable account lock and pending records, actual worker spawn/owned handles, initial sample/claim, build/package acquisition, authenticated Run, brokered full context, actual registry binding, and the explicit pre-data refusal. Verified the borrowed owner retains canonical Train bytes/digest and original request, and preparation borrows compare the original context/broker. Input0/Reviewed2 and N01–N07 actual reader I/O are Step3 prerequisites, not a missing Step2 scientific implementation finding.

Inspected production monitor/scheduler/action execution, partial-spawn custody, TERM-batch grace acknowledgement, sticky common-clock fallback, bounded stopped wait, exit versus both-stream EOF separation, direct Python/Git brokerage, retained source/root/receipt joins, capacity observation, account history resolution, and native driver/adoption ownership paths. Reviewed v2 build/helper/recipe changes and the precise runtime output Git exclusions. These positive source connections are substantial; none of them establishes host trust, a clean finalized Linux build, native16 PASS, native timing, physical fit, or science.

The 14 scheduler cases and W23/W24 are pure evidence. The authorized local socket transport case checks framing/raw separation without a child. The actual temporary-Git exclusion test checks only the two runtime exclusions and unrelated-untracked-source rejection. Native16 remains source-only/unexecuted. Existing publication preview `0b2d17a` is outside this review; this WIP is not released by the review.

No broad unrelated quality verdict is issued by this specification-review seat. F01, F02 and F03 require disposition/fix and a frozen scoped rereview; F04 is documentation cleanup.

SPEC FAIL
