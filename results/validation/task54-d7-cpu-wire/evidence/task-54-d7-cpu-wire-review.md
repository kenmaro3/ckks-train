# Task54 Phase2 Step1 CPU wire independent review

SPEC: **CHANGES REQUIRED**

QUALITY: **NEEDS FIXES**

This verdict covers the frozen observed CPU wire representation slice, against source BASE `abd6c3393b843d7d80fff4ebc890526866682c3b`. It does not approve Task54, a live owner, admission/start, native execution, a scientific result, or deployment. No Critical finding: the new code remains private and inert. Important supplied-consistency omissions and the unwaived process deviation prevent acceptance of this slice.

## Inputs and review method

Read the release instruction, implementation brief, adopted FIX2 contract, R-WIRE-1/2/3, complete author report, complete code companion, both finite reference writers, fixture inventories/manifests, retained RED/GREEN/compiler/lint logs, and the source-inventory finding. The code companion contains every production/test hunk; its explicitly omitted hunk is only the large literal bundle. I used the writers and manifest for that bundle rather than treating hexadecimal bulk as code. Root's all-57-file/all-78-literal pin verification and original-25 equality are release facts, not tests rerun by this reviewer.

Independently checked SHA256 of the full diff (`0e7fd07e72cc0275153dec7626eaef873d76ea1e57f605fd4747b75caa16ae6a`), companion (`3fea33c05af7542753ff1ebf7455f4f5901267ae7c1e6e2d15a39140274b7546`), original literal bundle (`a8bab670bc010b50053777f761abb8998ba8d62f0f86590512c215fc8d8c2a9d`), and both initial-writer copies (equal, `50c3ffee17e68f6854d980a66bf3eb2974831cc38b3e7b3d33efdcb98a23fb23`). No Cargo, tests, fixture writers, project helpers, native/HE/dataset work, network, SSH, process inspection, source edits, index changes, or HEAD changes were performed. The only written artifact is this review.

The following unchanged-code inspections addressed named risks, not a broader audit:

- Quick v2 losing v1 supplied identity/context parity: `crates/ckks-train-bench/src/study_reference/document/evidence/quick.rs:86` and `:431` (`QuickIdentity::check_common`, `check_quick_preflight`). They establish CW-02 below. The single changed visibility line in this file was reviewed in the diff; these are unchanged functions.
- Strict JSON/cap/number claims relying on shared primitives: `crates/ckks-train-bench/src/study_reference/document.rs:155`, `:430`, `:456`, `:572`; checked layout caps, exact-field visitor, trailing-content scan and unsigned-decimal decoder.
- Supplied FileRef/clock validation scope: `crates/ckks-train-bench/src/study_reference/document/evidence/tails.rs:18`, `evidence/wire.rs:193`, and `evidence/quick.rs:398`; FileRef byte comparison is length/hash plus valid relative path, not retained origin/custody; clocks check ordered supplied values, not actual clock provenance.

No changed function needed a separate source-file reread. A small companion-output section was retrieved again because the tool output had elided the `r.validate()` line; the original hunk itself is complete.

## Strengths and verified scope

- The new types are private, with no owner/gate/start capability constructor. `observed.rs:1-20` cleanly marks the inert branch and reuses existing modules; the source diff contains no runtime, trainer, dependency/feature, helper or native integration change.
- `observed/requested.rs:55-132` checks the 4096-byte envelope, exact 62-byte Train request, fixed selectors/options/caps, bounded preparation counts and sorted seed grammar. `:187-216` binds actual supplied case fields, runtime and request, including the Full registration used by Quick. The fixed writer redecodes its output (`:137-177`).
- `observed/bundle.rs:119-213` revalidates four original tails, exact common/branch identity, consumed-plan equality, selected UserScope and serial overlap, claim source/approval/supervisor, and RequestedWork binding. This preserves the principal R1/R2 relations. `evidence.rs:397-457` factors the pre-existing common/branch algorithm without redefining static resource semantics.
- `observed/records.rs:60-111` implements checked allocation products and conservation, closed canonical site suffixes and ordering. `:189-218` checks sample identities, checked RSS sum and status consistency. Initial-sample framing and stream framing are separate (`:337-344`, `:440-465`), matching R-WIRE-1. Actual sampler ownership/scheduling is not claimed.
- New policy tuple/domain checks are explicit in `observed/bundle.rs:92-110` and `observed/quick_v2.rs:60-119`; v1 bytes are not relabeled. The static Q1 change at `resource/links.rs:336` is a narrow baseline-mode comparison, with an actual focused assertion RED and GREEN.
- `observed/result.rs:191-195`, `:398-404`, `:597-654` reconstruct every JSON payload field into canonical binary and require exact binary equality. Ordered writing and named Counter/timing fields use the recorded R-WIRE-3 order. Finite diagnostics, unsigned JSON, exact schema members and export byte equality are real checks, not only round trips. The result latency uses server/update time rather than end-to-end (`:547-566`).
- Both writers are independent stdlib grammar implementations, with no production imports or encoder calls. The extension reads only the archived original literal bundle (`task-54-d7-cpu-wire-fixtures-extension.py:11`), then explicitly constructs request/evidence/event/result bytes and JSON. This is meaningful independent expected-value construction despite CW-06's ordering failure.

## Important findings

### CW-01 — Full supplied terminal linkage is missing from Step1

Evidence: `crates/ckks-train-bench/src/study_reference/document/evidence/observed/records.rs:378-405`; `observed/terminal.rs:3-56`, `:58-92`; `cpu_wire_tests.rs:526-566`. Requirements: `task-54-d7-cpu-wire-brief.md:7-11`, FIX2 `:119-123`.

`FullEnd` validates paths, update/sample counts and presence of three output references. It never consumes a supplied Full admission/start, result, evidence bundle or supervisor end. The sole terminal entry point is `QuickTerminal`, which requires scope Quick. `CpuResult::check_bundle` can compare common result provenance for either branch, but does not close the Full admission/start/result/end/supervisor chain or compare Full result target lock/threshold to the Full evidence branch. The unchanged generic Full wire format cannot supply this newly required composition by itself.

Consequently a complete-looking FullEnd can name arbitrary syntactically valid admission/start/result/JSON/export references without any Step1 API checking the complete supplied relationship. This is an omitted representation validator, not a request to implement live owner authority early. The report's deferral of the **real** Full runtime path (`task-54-d7-cpu-wire-report.md:15`) is correct for Steps3/4 but does not cover these already-required pure checks.

Add an inert Full chain consumer that composes the unchanged Full admission/start parser with the exact observed policy tuple, selected bundle/attempt/case/target provenance, supplied result/output identities and complete supervisor closure. Keep durable acquisition and actual owner checks later. Use independent bounded Full fixtures/negative mutations for mismatched scope, tuple, attempt, start/admission, target, result/output and closure. Do not introduce a test-only alternate acceptance rule to avoid a registered schedule fixture; explicitly inventory any necessary fixture-bound change first.

The inventory claims a “new Full terminal consumer rejects wrong scope and legacy resource tuple” (`task-54-d7-cpu-wire-test-inventory.md:45`). No such consumer or assertion is present. Correct this coverage statement alongside the implementation.

### CW-02 — Quick v2 chain fails to bind its preflight identity and context to the bundle

Evidence: `crates/ckks-train-bench/src/study_reference/document/evidence/observed/quick_v2.rs:204-240`, particularly `:226-234`; unchanged `evidence/quick.rs:86-98`, `:446-469`. Requirements: brief `:9-11`, FIX2 `:63-65`, `:132`.

The new checker compares the three event identity hashes to each other, then calls `q.check_common` and compares registration bytes. `check_common` deliberately does not compare operational attempt, consumed plan or request. The checker never compares the preflight identity bytes/hash with `Branch::Quick.operational_identity/operational_id`. It also omits preflight approval, train-receipt and executable equality with Common. The v1 preflight checker performs both exact identity and context comparisons.

A statically traceable accepted counterexample is a preflight identity changed from `attempt1` to `attempt2`, leaving its registration/common fields valid and the selected bundle at `attempt1`. Rehash the event identities and their inter-event links consistently while retaining the original bundle gate hashes. The new checks never compare the differing attempt-bearing identities. A complete result can still carry the bundle's original attempt because `CpuResult::check_bundle` compares only that result to the bundle. Independently changing a preflight approval/train-receipt/executable field also survives the new preflight validator and chain comparison after consistent link updates. These are supplied-value inconsistencies, requiring no owner or native facts to reject.

Restore exact preflight identity equality with the selected Quick branch and all existing supplied context parity. Captured argv/actual runtime identity can remain a separate later acquisition obligation. Add independently constructed, self-consistently rehashed negative chains for each omitted relation; merely corrupting a hash would exercise an earlier check and miss this defect. The current two positive chains and version mutations (`cpu_wire_tests.rs:272-301`) do not cover it.

### CW-03 — Preparation recovery/lock children accept an unexpected selected seed

Evidence: `crates/ckks-train-bench/src/study_reference/document/evidence/observed/requested.rs:248-286`, especially `:265` and `:276`; `cpu_wire_tests.rs:106-143`. Requirements: brief `:9`; FIX2 root/child table `:84-89` and unexpected-seed refusal `:92`.

For Preparation roots4 and5, `check_child` verifies `selected < 10` and immutable root seeds `[0]`, but never requires selected=0. Thus the existing W09/root4 or W10/root5 record can successfully bind its root with `[0]`, then pass `check_child(4, 2, "iris__emlp1", 1)` or `check_child(5, 2, "iris__emlp1", 1)`. The special selected=0 condition protects only Train root6. These preparation roots are expressly seed0-only, unlike root3's `{0,S}` coverage.

Require selected=0 on roots4/5's permitted child relation (and document the coordinate's meaning). Preserve root3's legal nonzero selected-seed coverage. Add two tiny negative calls using the unchanged W09/W10 literals and a positive root3 `{0,1}` representation case. This is a pure relation defect, distinct from N06's genuinely later foreign-owner/lifetime check.

### CW-04 — Full result accepts unscheduled checkpoints and therefore an impossible first target coordinate

Evidence: `crates/ckks-train-bench/src/study_reference/document/evidence/observed/result.rs:442-452`, `:508-534`, `:547-566`; `cpu_wire_tests.rs:388-430`. Requirements: FIX2 `:16` (400 checkpoints at updates6*k), `:111`, `:119`; R-WIRE-3 first-target-coordinate ruling.

The result validator checks increasing positive updates, upper bound, count and final equality; complete Full requires 400 entries. It does not check that Full endpoints occur at the registered six-update schedule, nor that `crossed` matches that schedule (only sortedness is checked). TargetOutcome then chooses the first qualifying **supplied** endpoint. An incomplete Full record with one endpoint at updates1, samples20, threshold0, first_checkpoint=Some(1) and matching cumulative time passes those branches. It violates R-WIRE-3's first possible registered coordinate6. A complete 400-entry record can similarly replace an early scheduled coordinate while keeping the count and final2400.

The existing R-WIRE-3 test rejects ordinal0/1 only while retaining endpoint.updates=6. It cannot detect a jointly changed endpoint and target coordinate. Validate supplied checkpoints against the exact registered schedule/consumed plan appropriate to this fixed slice; preserve the actual plan as the later runtime authority. Add a bounded incomplete Full negative at update1 plus skipped/foreign boundary negatives, and a valid update6 case. These tests need no 400-endpoint allocation or scientific execution.

### CW-05 — Successful Quick result can end before its linked start

Evidence: `crates/ckks-train-bench/src/study_reference/document/evidence/observed/terminal.rs:23-47`; `observed/quick_v2.rs:237-240`; `observed/result.rs:416-568`. Requirements: brief `:11`, FIX2 start/publication order `:105`, `:123`.

The chain orders preflight <= start <= worker-end, and terminal orders result-ended <= worker-end <= supervisor-end. No check orders start <= result-ended. `CpuResult::validate` does not compare its clock to the start. A consistent result/JSON/end/supervisor chain can therefore carry result.ended=(0,0), before a valid start at (104,4), and still satisfy all of these inequalities after its file links are rehashed.

Compare the supplied start clock to result.ended as part of terminal validation; reuse this for Full when CW-01 is implemented. Add a focused independently expected negative with only that ordering violated and all dependent links consistent. This does not require proving actual clock origin or elapsed runtime; those remain later obligations.

### CW-06 — Required universal literal-before-encoder ordering was not met

Evidence: `task-54-d7-cpu-wire-brief.md:17`; report `:36-40`; `d7-cpu-wire-logs/implementation-compile-1.log:688-766`, `implementation-compile-2.log:14-40`, and `red-policy-terminal-api.log:2-28`.

The initial25 literal set and genuine initial missing-module RED are retained. The coherent extension's expected observed bundles/Quickv2/result JSON were authored after the corresponding encoder drafts. The compiler logs already refer to draft JSON/Quick code before the later policy/terminal API RED, consistent with the disclosed chronology. A later independent writer and passing tests improve correctness evidence but do not meet the brief's explicit per-record ordering requirement. Root has not waived it.

There is no complete intermediate production-source snapshot or initial six-test snapshot (`report:40`). The current writers demonstrate an independent construction algorithm; they cannot prove absence of conceptual influence from the earlier drafts or reconstruct every intervening test edit. Do not claim that the initial missing-module RED covered all final78 literals or all13 tests. Do not rerun tests to manufacture historical ordering.

This is an Important process/spec finding, separately from CW-01–05's source defects. Preserve the accurate disclosure and evidence. Root must explicitly adjudicate this historical deviation and any compensating review before acceptance; future fixes should freeze their independent expected negatives before changing production checks. This reviewer grants no waiver.

## Minor finding

### CW-07 — Finite inventory overstates several asserted checks

Evidence: `task-54-d7-cpu-wire-test-inventory.md:12`, `:15`, `:19`, `:45`; `crates/ckks-train-bench/src/study_reference/document/evidence/cpu_wire_tests.rs:59-103`, `:163-186`, `:353-386`, `:526-566`.

Beyond CW-01's nonexistent Full consumer, resource_arithmetic does not call `check_supplied_capacity` or exercise threshold sample validity despite its inventory description. request_case_binding changes case mode, runtime, partitions and consumption, but not each listed role/security/batch field. terminal_links has two positive Quick chains and one unresolved-closure mutation; it does not independently mutate each missing result/start/evidence/end link. Treat these as planned or indirectly checked properties, not distinct passed assertions. Align the inventory with actual assertions and add only the bounded focused cases needed for the concrete defects above. The implementation contains several of these checks; lack of an individual negative is not by itself evidence that the check is incorrect.

## Retained test evidence and limitations

No tests were run during review. Retained logs show:

- Initial missing `observed` API RED (`red-missing-api.log:2-9`), then compiler failures and a 4/6 run. The original unused-import/ambiguous-name warnings and zsh read-only `status` wrapper failure are preserved; they are not pristine GREEN evidence. The current kind offset and opposite-consumption correction match the fixed independent grammar, but absent intermediate snapshots limit retrospective test-edit verification.
- A genuine missing policy/terminal API RED (`red-policy-terminal-api.log:2-28`), then 10/10 (`implementation-compile-3.log:17`). This establishes only those missing APIs' chronology.
- Q1's initial filter selected zero tests (`q1-filter-mismatch.log:5-7`), followed by a real assertion RED (`q1-red.log:7-17`) and 1/1 GREEN (`q1-green.log:7`). The zero-test invocation is correctly excluded.
- Writer-order RED is an exact byte-equality failure. The later stale negative expected authored JSON to differ (`green-expanded-1.log:15-16`); replacing it with a legal-whitespace export-byte inequality is legitimate given the fixed oracle. The current writer-order test still asserts exact independent bytes (`cpu_wire_tests.rs:398-402`).
- The later seed RED exposes Train selected1 (`seed-counter-red.log:9-10`); it does not cover Preparation roots4/5. Counter-overflow RED (`counter-overflow-red.log:7-8`) and direct binary NaN checking are meaningful, beyond JSON-vs-binary mismatch alone.
- `green-final-new.log:5-20` records 13/13; `terminal-after-lint-green.log:4-7` records the changed terminal helper's focused 1/1. Five named historical regressions each record 1/1, plus Q1: 19 distinct final pure tests, not native or full-matrix coverage.
- `clippy-final.log:1-2` is clean successful compiler/lint output; prior two lint failures remain preserved. `fmt-final.log` is empty as expected for successful formatting check, with exit0 reported in the frozen author handoff. The log text alone does not encode that exit status or the entire invocation; no replacement run was made.

The original25 values remain the stronger pre-encoder oracle subset. All53 extension literals are separately authored expected bytes, but have the sequencing limitation above. Full result success with all400 endpoints is not represented by the two-endpoint fixtures; the `full_terminal_records` test name must not be read as complete Full terminal coverage. R-WIRE-2 allocation owner attribution, actual measured f64 values, physical capacity and retained file origins cannot be proven by these supplied fields.

## Outstanding later-stage boundaries / cannot verify

- W23/W24, O07/O08 and N06 remain actual owner/order/producer/lifetime obligations. The inventory assigns these to Steps2/3 (`test-inventory.md:29-31`); no wire assertion discharges them. CW-03 concerns the independently testable seed relation, not foreign-owner authority.
- Actual supervisor pidfd/subreaper/flock inheritance, broker/helper whitelist, fixed10s deadlines, TERM/grace/KILL, durable markers, reaping/EOF/PID reuse, sampling/raw-file closure and all14 schedules remain Step2. `QuickTerminal` checks a supplied complete supervisor record, not the actual OS closure or sample-stream provenance. That limitation is acceptable only when maintained in its later consumers.
- Actual same-source/owner/context/plan acquisition, five genuine producers, Full current/predecessor target verification, held Full authority and durable admission/start remain Step3. The author names remaining `run_owned` refusals and Reference/Target entrypoints at `report:65-66`; this review did not broaden into those unchanged implementations. Nothing in this diff creates a replacement positive runtime path or a permanent refusal stub.
- Actual session snapshot/decryption, paired FP32, timing/counter measurement, executable-authored result publication, strict table reader and all-adapter/ten-seed execution remain Step4 (`report:67-68`, FIX2 `:127-139`). The supplied-link defects above must be fixed before these consumers rely on the new representation validators.
- The separately recorded finalized source-manifest issue remains material: `task-54-d7-source-inventory-finding.md:3-9` records553 pre-Step1 source records against cap512 and a not-yet-implemented compaction ruling. Ordinary Cargo/clippy success does not establish a finalized clean-recipe build receipt. I did not rerun inventory/helper/build work or treat forecast397 as an observed count. Root must close the separate compaction/custody review and recompute actual final records/bytes after source additions, retaining cap512 and original payload identity.
- Native prerequisites/security/correctness/input/reference facts, FULL2 and queue holds remain unchanged. No real CPU/GPU, performance, target-achievement, full-matrix or Task54 completion claim follows from this review.

The source separation and independent literal grammar are useful foundations. Acceptance requires the supplied consistency fixes, accurately scoped test evidence, and explicit root adjudication of the historical process deviation.
