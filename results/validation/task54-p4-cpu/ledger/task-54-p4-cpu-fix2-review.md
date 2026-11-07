# Task54 P4 CPU source fix2 re-review

2026-11-07. **F1 addressed. SPEC: PASS for the scoped source corrections. QUALITY: PASS for this tiny delta.** No new finding. Runtime and numerical acceptance remain pending; root retains separate finite execution authorization.

Reviewed the exact fix2 release, complete2091-byte delta, appended fix2 report, all8 pins and terminal no-run/Clippy records. Only read/hash operations and this review-file write were performed. No tests, Cargo, constructors, HE/Engine/planner, source edits, network, Git mutations or subagents.

## Frozen evidence

Independently verified:

- Source report SHA256 `95568607d115a63a7230a0bf3bc9e37ee4f4de352c6e2608226ddd24ecfcb12b`.
- Exact2091-byte delta SHA256 `33fa477213620af854294825fd84a34ca8457407d8404edbe1fcb614e230281f`.
- All8 pins JSON SHA256 `081db0d0a13c68ba7554e81338ccfeff6e19b11fea60912c2b0535d9e4d2609a`.
- All8 current sources match their pins. Only `study_diagonal/p4_tests.rs` differs from fix1. Reversing exactly the two264-to268 literal changes reproduces the fix1 pinned test-file hash, independently confirming no other test logic, assertion, recipe or formatting change. The other seven sources are byte-identical.
- Recorded `01-compile-final` and `02-clippy` exit0, with clean raw outputs. Both before/after8-source inventories equal the reviewed fix2 pins. These are focused no-run and strict scoped Clippy evidence; neither executes the test binary.

## F1 resolution

Both the `ReadOnlySessionSnapshot::assert_unchanged` guard and final expected count now use268. This agrees with the preserved source:

| Diagnostic | Correct prospective count |
|---|---:|
| First-update comparisons |18*12 +8 reorder +8 I-only foreign-basis =232 |
| Later rate-boundary comparisons |2 |
| Replay/exhaustion comparisons |32 |
| Poison-retry comparisons |2 |
| All unchanged comparisons |268 |
| Snapshot captures |54 |
| Configuration collections |54+268 =322 |

The appended report explicitly supersedes the historical228/264/318 counts with232/268/322. All8 reorder checks remain. The correction removes the deterministic264 guard/final-expectation mismatch without reducing coverage or adding work to the original matrix.

No production logic, expensive-operation guard, numerical tolerance or case changed. The54 captures,68 client inspect/decrypt endpoints and122 encrypted model snapshot copies remain unchanged, as do all expensive bounds:20plans/20geometry/20initial/34updates/0handoffs;20sessions I9/NI11;18oracles/32Engine calls;34trainer attempts32intended successes+2failures;2parameters/2keys/1proof/1descriptor/1bootstrapper/22Encoders;<=85actual bootstraps;1positive+16negative factory preparations. Prior S1/S2/Q1/Q2 source-level dispositions remain addressed.

The scoped source-review blockers are closed. This review does not assert any runtime rejection/parity/counter/cap result, canonical pre-key/D7/production acceptance, P4/Task54 completion, or scientific campaign evidence. It grants no execution release itself.
