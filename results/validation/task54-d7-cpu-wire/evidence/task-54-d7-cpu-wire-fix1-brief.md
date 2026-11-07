# CPU wire FIX1 ruling and release

Status: FIX1/5 RELEASED to original sole writer after root read the full independent review. Source BASE remainsabd6c339; reviewed source is exactly the16path author manifest a0d3f8f33f0f4abea67c58e8454952dfbe519f1d6db05f53b182cf02339c3a05 and preserved cpu-wire-review-snapshot/. Current HEAD4ff70c9 is documentation/observation only. Do not mutate the old report/diff/pins/logs/rulings; write separate FIX1 artifacts so original evidence remains independently reviewable.

## Root disposition of CW-06 before edits

Ruling: the universal per-record literal-before-encoder requirement was not met for the53 coherent extension literals. Retain CW-06 as a historical process shortfall, with the exact original25/extension53 distinction and missing snapshots. It must never be called repaired by later runs or full TDD compliance. The independent reviewer examined both stdlib reference writers and found independent grammar construction, no production import/call; this supports expected-value correctness but cannot prove absence of conceptual influence. Root accepts this evidence limitation for continued repair/review of this source slice, conditional on all CW-01–05 code omissions being corrected with new expectations frozen before the corresponding production changes, and independent scoped review of those fixes. No source adoption occurs merely from this disposition.

Reason: the historical order cannot be changed. Replaying RED after temporarily deleting/restoring production code would manufacture chronology. Discarding the draft and rewriting from the same already-known vectors would cost a new implementation/review without establishing historical independence. The chosen alternative preserves the actual failed prototype/source/logs, explicitly records process noncompliance, and uses independent review plus genuine first-failing counterexamples to close current correctness risks. Cost if wrong is shared conceptual error in the53extension expectations; source adoption must therefore rely on explicit contract-based cross-record/schedule/seed/clock checks, not all-green roundtrips. Future source work keeps the original test-before-implementation requirement. At publication report the process limitation separately from passing code tests, native evidence and scientific results. Reviewer should verify accurate preservation/disclosure and this recorded disposition, not relabel the history as compliant or grant an implied waiver.

## FIX1 implementation scope

Address CW-01–05 and the concrete coverage overclaims in CW-07 as one bounded source correction. The complete findings below are verbatim. Implement the missing Full supplied-chain consumer in this Step1 slice; actual OS/custody/gates/HE authority remains later. Reuse unchanged generic Full admission/start and existing common/link/clock checking instead of a second acceptance algorithm. Preserve current resource policy tuple and all historical bytes. For Full, selected target lock/threshold, branch/attempt/case/source, result/JSON/export/worker-end/supervisor linkage must all be checked. The new reader only validates supplied records; no live owner or result publication is enabled.

Quick v2 must preserve exact operational identity bytes/hash and all v1 supplied context parity. Negative chains must be consistently rehashed so they reach the omitted comparisons. Preparation4/5 children require selectedseed0, while root3 retains legal sorted{0,S} coverage. Validate Full checkpoints/target coordinate against the exact registered Iris schedule and its crossed boundaries; do not merely compare a claimed target coordinate to an equally wrong endpoint. Both terminal branches order start<=result-ended<=worker-end<=supervisor-end with existing clock rules.

Before any new fixture writer or executable filter, update a separate FIX1 finite inventory, listing exact named tests, input/output members/count/aggregate bytes, constructors and maximum shapes. Freeze independent expected negatives and any complete Full positive fixtures BEFORE changing corresponding production validators, then retain genuine assertion/API REDs. The current fixture ceiling16KiB per literal may not fit a full400-checkpoint Full positive. Explicitly inventory and justify any necessary finite fixture ceiling increase BEFORE execution; preserve production A/C/R/K/512 caps, keep fictional corpus compact, no cap-sized allocation shortcuts or test-only alternate schedule. A full400endpoint pure byte fixture is allowed when needed to verify complete Full representation; this is not an actual dataset/parameter/key/HE/native run. Keep original78 literals byte-identical and store new fix vectors separately or as a clearly preserved extension; no production encoder generates expected bytes.

Run the new/fixed focused tests and affected existing tests once after relevant code changes, then strict CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo clippy -p ckks-train-bench --all-features --all-targets -- -D warnings and cargo fmt --all -- --check. Retain every failure and exact exit status/invocation; no broad workspace/all-tests/--list, helper/native/HE/SSH/network/install/newcache work. No dependencies/features/project unsafe changes. No subagents/commit/push; root owns review/publication. Keep source ownership exclusive.

Write task-54-d7-cpu-wire-fix1-report.md with each CW finding disposition and exact covering commands/logs; task-54-d7-cpu-wire-fix1.diff against the frozen reviewed16sources plus new files, and task-54-d7-cpu-wire-fix1-pins.json. Also retain a full source-only diff against originalBASE for eventual publication. Report CW-06 as historically unmet/root-disposed, never repaired. Inventory CW-07 statements must match actual test assertions and remaining runtime obligations. Freeze source on handoff for original reviewer's scoped re-review of these findings/new breakage only.

## Verbatim independent findings

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

