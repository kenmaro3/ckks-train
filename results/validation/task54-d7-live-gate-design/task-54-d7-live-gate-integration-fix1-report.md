# Task54 D7 live-gate integration — FIX1/5 report

Status: design fix prepared; independent rereview/root ratification pending. No source or runtime acceptance claimed.

## Scope and custody

Read the complete independent review with SHA256 `2f629f5c28502703eefb66df695c97edd98db513a02e6fd6dba5f646c7182100`. Before editing, copied the original proposal byte-exact to `task-54-d7-live-gate-integration-proposal-before-fix1.md`; its SHA256 remains `f007784a7ccbb99574dfe0a1586e186534d4c5ba9904e716e1880d3af239cf5b`. Only the proposal, preserved copy, this report and the exact fix diff were written for this task. No Rust/source, Cargo, test execution, SSH/runtime, agents or commit.

## I1 Important — addressed

- Proposal §4 now defines a private closed `GateScope::Full` / `GateScope::Quick` boundary used by each of the same five constructors. Common owners/identities are explicit; no optional-target or caller bypass branch exists.
- Full requires the real verified target/reference/lock/threshold and the independently materialized selected receipt. Absence, unverified links and placeholders cannot satisfy it; a legitimate zero threshold must still come from the verified target.
- Quick binds exact full registration bytes solely as registration provenance, a separate operational Quick identity, seed0/two-update consumed plan/order, canonical input and operational attempt. It has no target/threshold/full admission assertion or target-receipt comparison.
- §5 separates common evidence prefix from mandatory Full and Quick branch bodies. New inner evidence scope tags distinguish the branches without modifying frozen full CaseIdentity, ExecutionBinding, Gates or admission/start contracts.
- Only all-Full capabilities assemble CheckedCaseForAdmission. The separate all-Quick bundle enters only the private Quick entry; mixed bundles and Quick-to-Full promotion reject. All five substantive checks and live rechecks remain.
- §9 freezes eight named source-acceptance branch tests, including all five requested cases: missing Full target, target-free Quick/full refusal, target/threshold placeholders, equal-hash cross-scope refusal, and exact Quick consumption. Their fixtures/call inventory must be fixed before source; no tests were executed here.

## Accepted clarification and carried M1

§4 makes accepted context-token promotion concrete: pre-admission gates retain inert instance identity and owned facts; full admission compares the original instance/ProcessStamp and store-context pointer, obtains a fresh checked-input token borrowing the actual context for `'c`, compares facts and retains it into the live frame. Quick performs the corresponding actual-context promotion without a store/target claim. Process-wide instance numbers never wrap/reuse; no `'c:'p`, unsafe lifetime extension or long-lived single-lifetime TrainCapture.

M1 remains a carried Minor pre-source requirement in §5/§10, not a new fix-loop item. Closed names/locations/entries/caps, relative durability, guard/held-FD/no-clobber/fsync ownership and every partial-sidecar recovery/Initial/Retry state must be specified before protocol/gate source. Admission hashes identify exact retained evidence bytes. Unknown entries and partial/orphan evidence never reset Initial eligibility; no sidecar publication precedes guarded eligibility.

## Artifacts and validation scope

| Artifact | SHA256 |
|---|---|
| `task-54-d7-live-gate-integration-proposal-before-fix1.md` | `f007784a7ccbb99574dfe0a1586e186534d4c5ba9904e716e1880d3af239cf5b` |
| `task-54-d7-live-gate-integration-proposal.md` | `c667176c41f1e5dda060e552c6fe9db1209446c76c3f2161ea949571cb539e5c` |
| `task-54-d7-live-gate-integration-fix1.diff` | `58c2e0d606d1107120c8752e62396108594b5610bb4c39276ae26e3c93305f2e` |

The diff is the exact unified text difference from the preserved original to FIX1, with stable relative filename labels. Read back affected sections and checked original/review hashes. This verifies document custody and the intended design edits only; no implementation, native durability, real gate, HE, scientific result or Task54 completion follows.
