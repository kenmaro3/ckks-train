# Scoped FIX2 independent review

Scope: FIX1 finding R1-N and new breakage introduced by FIX2 only. Source BASE remains `bb0ab8add24b4974a3f14bbeb9a63791b2b81e01`. Original/FIX1 artifacts and previous review findings are preserved. Unaffected architecture was not reopened.

Independently checked inputs:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| task-54-d7-cpu-integration-contract-fix2.md | 48850 | 7f1f4b3ed796735a09a44ccc033a26dd517bac21390c404229050869c12244f6 |
| task-54-d7-cpu-integration-contract-fix2.diff | 18437 | b70130e6cf054fa325f02114c7193ce3064a5aa58354a557b1ac8745590daaaf |
| task-54-d7-cpu-integration-contract-fix2-pins.json | 4881 | 12043bf27ccef54db2d714b1d5ddb307488326f46e885bb4c42073013efec4ce |

The supplied four-hunk diff reconstructs exact FIX2 text from FIX1. Hashing and hunk reconstruction used bounded Python standard-library reads and in-memory comparison; no patch was applied to source or contract files.

## Verdicts

**SPEC: PASS within this scoped contract rereview.** R1-N is addressed. The immutable root request is distinguished from the exact private child operations required by existing Target/Reference code. Same-owner lifetime, profile/seed coverage, Full current/Retry checks and target-free Quick are explicitly preserved.

**QUALITY: PASS within this scoped contract rereview.** The correction is finite and implementable without another admission, resource lock, invocation or public permit abstraction. N01-N07 and clarified W15 cover the previously missing positive and negative nesting boundaries. No new blocking defect was identified in the correction.

The prior R2 and 4096-byte/62-byte grammar conclusions remain unchanged. These are proposal-level verdicts for root's source-release decision, not source acceptance, test execution, native validation, scientific success or Task54 completion.

## R1-N: addressed

1. **Root equality and internal edges are distinct.** FIX2 section4 makes RequestedWork.operation the immutable root orchestration kind, checked at public/root entry. Private child bodies borrow the same ResourceOwner/context/store frame along the closed table. The rule no longer requires each legitimate internal method to equal the root's tag. It also does not authorize arbitrary transitive calls. W15 now rejects a wrong root or an out-of-subtree0->2 edge while allowing the explicitly listed3->2,4->2 and5->2 paths.

2. **The table matches the concrete reader graph.** `study_reference/target.rs:355` calls `read_reference_snapshot` inside `read_target`; line433 does so inside `recover_lock`. FIX2 allows exactly those edges. The target read independently reconstructs the selected seed at line364 and compares the supplied full receipt, so the specified root seed coverage sorted unique {0,S}, with child seed0, preserves the real work. Recovery remains seed0 reference verification rather than invented training. Profile equality covers the target store, reference document/snapshot and reconstructed materialized profile before affected expansion.

3. **Snapshot acquisition and consumption have one lifetime.** `VerifiedReferenceSnapshot` borrows its originating ReferenceStore/context (`publication.rs:289-315`); `lock_target` consumes that snapshot while checking seed0/profile/context and independently reconstructing seed0 (`target.rs:489-513`). FIX2's root5 wrapper starts the resource owner/claim before source opening or snapshot acquisition, creates stores inside that frame, acquires child2's snapshot, then obtains the short profile guard and invokes the existing consuming body. The owner survives consumption, publication and snapshot teardown. A snapshot from a separately acquired resource owner is expressly rejected even if its bytes or metadata match. This removes the previous impossible separate-invocation interpretation without weakening existing origin/lifetime checks.

4. **Train nesting does not forge preparation authority.** Full Train derives its read-only Target3 -> ReferenceSnapshot2 subtree from the already bound kind0 request. Current and explicitly authorized Retry predecessor checks use that same subtree; no kind1 claim, second flock or target creation/recovery is introduced. Quick has no Target/Reference operation0..5 edge and retains direct canonical input checking only. Numeric seed0 does not promote Quick into a target-bearing route. The generalized {0,S} notation does not grant additional selectors beyond the fixed slice.

5. **The dependency graph stays acyclic.** RequestedWork and immutable claim still precede context/registry expansion. Actual binding then selects the allowed private subtree and checks its profile/seed coverage. Child execution neither rewrites the claim nor produces a new root request needed to authorize its parent. Reference snapshot acquisition precedes its consumption under the existing frame; it is not a source of resource authority. No wire change, future hash dependency or new admission cycle is introduced.

## Finite verification coverage and limits

N01-N04 supply the four missing same-owner positives: Target read, Target recovery, root5 snapshot acquisition/consumption, and Full Train Target/Reference nesting. N05 prohibits a read-to-recovery edge, N06 rejects a foreign-owner snapshot, and N07 rejects Quick-to-target access before target/reference I/O or reconstruction. Clarified W15 remains a real wrong-root/out-of-subtree negative. The original32 FIX1 checks and other contract tests remain required future implementation checks; none were run here.

No additional design correction is required by this scoped review. Source review must still establish that the actual wrappers, private borrows and lifetime/order checks implement this table, rather than merely adding an unused representation or mocked success path. Native facts listed in the contract remain unobserved, with FULL2/queue ordering and separate native/scientific release boundaries unchanged.

## Review record

Read the entire FIX2 diff/pins and the affected final-contract section; reread the actual Target/Reference calls, snapshot representation and consuming function needed to assess R1-N. Independently checked the three listed artifact hashes and exact diff reconstruction. Root's verification of all8 pins is not represented as my own repeated full-manifest verification.

Only this canonical review was written. No tests, Cargo, project helpers, native/HE/server/SSH/network/install commands, process/dataset-directory scans, source/public-doc mutation, commits/pushes or subagents were used. The unchanged supervisor, resource-policy semantics, endpoint, JSON and timing architecture were not reopened. This report supplies no execution authorization or measured result.
