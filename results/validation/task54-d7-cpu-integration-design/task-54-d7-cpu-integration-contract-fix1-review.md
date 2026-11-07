# Scoped FIX1 independent review

Scope: original R1/R2 and breakage introduced by their correction only. Source BASE remains `bb0ab8add24b4974a3f14bbeb9a63791b2b81e01`. The original contract/review remain unchanged; this report does not reopen their otherwise accepted architecture.

Independently checked inputs:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| task-54-d7-cpu-integration-contract-fix1.md | 41451 | 26f4bf6a38ee39901b9abfc3725487499df2dcb518811897b4759449a894b5ca |
| task-54-d7-cpu-integration-contract-fix1.diff | 21881 | 63e8648b123ba365c0838fccc20648c2a2cbcd279b8fa2709a0f22daef5915da |
| task-54-d7-cpu-integration-contract-fix1-pins.json | 5205 | e6e1231dba1b8dcb49bfd4349cab712a45c9e8aafbe07f5b4adbeaf259dd3d0f |

The supplied four-hunk diff reconstructs the exact FIX1 text from the original contract. This was checked in memory using Python standard-library file/diff operations without applying a filesystem patch. Python difflib's default rendering chooses different blank-line context, but the supplied hunks and their counts are valid; that rendering difference is not a finding.

## Verdicts

**SPEC: CHANGES REQUIRED.** R2 is addressed. R1's original pre-expansion registration cycle is addressed and its new scalar grammar matches the source. R1 is not fully closed because its new exact operation-binding rule does not distinguish a requested top-level preparation operation from the required nested operations under the same retained owner.

**QUALITY: CHANGES REQUIRED.** One narrow FIX2 is needed to close that root/child-operation relation and add actual nested-path positives. Keep the 4096-byte RequestedWork grammar, 62-byte request representation, serial0 checks and all unaffected architecture. No new resource lock, separate invocation, generic operation-permit platform or physical proof is requested.

These verdicts concern contract readiness only. They authorize no source implementation, execution, native validation or scientific completion.

## R1: partially addressed; one fix-induced residual

### Addressed: pre-expansion wire and actual post-expansion binding

FIX1 lines70-78 replace canonical registered-case bytes in claim.work with a bounded non-authorizing RequestedWork. The immutable claim is durable before Run/context/registry expansion; only afterward does actual registry selection establish membership/canonical case bytes. The live owner and CurrentEvidenceBundle then compare the retained request to the actual selector, derived role/security/batch, runtime and Layout request. There is no claim rewrite or dependency on a future digest, so the original bootstrap cycle is removed.

The Train field discriminants agree with `study_profile/registered_case.rs:54-61`: family0, full0, cpu0, interactive0/noninteractive1, client_refresh0/when_needed1 and batched_diagonal0. Full registration remains distinct from Quick consumption. `study_admission.rs:1192-1233` maps Full/Quick to0/1, dedicated transpose to None, and writes consumption, two option tags, three flags, two limit u64s and five work u64s. With the compiled None/None request this is six tag bytes plus seven u64s, exactly62 bytes; it matches `document/evidence/auxiliary.rs:7-40` and its MIN62. The extra explicit partitions=None check is necessary because partitions are not part of that request-body encoder, and FIX1 provides it. No allocator-backed PlanningBudget/RegisteredCase is needed to encode these fixed bootstrap values.

Preparation's bounded operation/profile/sorted seed request is also distinguishable from registry authority. Its range/count/EOF checks are finite. W01-W24 and the later actual binding are appropriately future source checks, not claimed native facts.

### R1-N — New operation equality must preserve real nested calls and snapshot custody

FIX1 line76 says the preparation entrypoint fixes the operation and a later operation change requires a fresh ordinary invocation. Line78 requires the invoked operation to match the retained request before operation-specific expansion. Together with W15's rejection of operation0 used by operation2, this leaves all operation changes apparently forbidden, without specifying a different rule for internal borrowed calls. Saying "including nested reconstruction coverage" does not identify which child operation is permitted under which root request.

This is concrete incompatibility with the existing preparation graph:

- `ProfileTargetStore::read_target`, operation3, calls `ReferenceStore::read_reference_snapshot`, operation2, at `study_reference/target.rs:355`.
- `ProfileTargetStore::recover_lock`, operation4, calls that same operation2 at `target.rs:433`.
- `lock_target`, operation5, consumes an already retained `VerifiedReferenceSnapshot` (`target.rs:489-512`). That snapshot borrows its originating ReferenceStore/context (`study_reference/publication.rs:289-315`), and its resource custody must already cover its acquisition. A separate invocation cannot supply the same live snapshot/context owner.
- Full Train must borrow the same owner for its Target/Reference route, as the preserved section3 requires. Its immutable claim is kind0 Train, not a kind1 Preparation operation that can equal each nested wrapper's operation tag.

An implementation following the equality literally either rejects required paths, adds another invocation/lock that conflicts with retained custody, or invents an unreviewed nested-operation exception. Standalone positives W05-W10 exercise the six root operations but do not settle this distinction.

**Required bounded remedy:** define RequestedWork.operation as the immutable root invocation kind. Keep exact equality at public/root entry. Define the closed internal operation mapping under a borrowed same-owner frame: include read_target -> read_reference_snapshot, recover_lock -> read_reference_snapshot, and the lock_target root wrapper's snapshot acquisition before consuming that snapshot. Explicitly derive the allowed Full Train Target/Reference subtree from the bound Train request; Quick must retain its target-free restriction. Keep profile equality and define coverage of selected-seed plus required seed0 reconstructions against the retained root request without mutating its bytes. Child operations must not become new admissions or independently caller-selected permits. Unrelated operation changes and calls outside the closed subtree still refuse.

The existing wrapper names can remain; only the root-versus-borrowed entry distinction and finite call mapping need closing. For lock_target, ensure its root owner starts before snapshot acquisition and remains the owner when the existing consuming function runs. Do not transfer a snapshot from an independently acquired owner or relax current context/store checks.

**Required finite additions:** positive traces for standalone read_target -> snapshot, recover_lock -> snapshot, lock_target root -> snapshot acquisition -> lock_target consumption, and Full Train -> Target -> Reference under one owner/one claim/one lock. Add a prohibited child-operation/foreign-owner negative and a Quick-to-target negative. Preserve W15 as a wrong root invocation or out-of-subtree rejection, rather than applying it to every legitimate internal call. Source tests prove this ordering/binding only; they need no native execution to close the contract correction.

## R2: addressed

FIX1 lines60,66 explicitly fix the operational branch's semantic overlap to serial0 and require the retained selected UserScope tuple's overlap to equal that value in CurrentEvidenceBundle. The live resource/user producers enforce the same relation against current authority; explicit_overlap1 is rejected rather than reinterpreted. This preserves the additional equality in `document/evidence.rs:350-365` despite replacing the old structured Resource.claim with a FileRef. Existing selected-index validation, tuple binding, revocation, holds and Initial/Retry constraints remain.

O01-O04 cover the four I/NI Full/Quick serial positives. O05-O08 distinguish wrong-overlap rejection in the inert bundle from rejection by actual live producers. No historical Resource/User wire change or new caller-selectable overlap field is introduced. R2 needs no further correction in this scoped review.

## Work performed and retained boundaries

Read the entire FIX1 diff/pins and the affected final-contract sections; compared the new grammar to the actual request encoder, auxiliary decoder and enum declarations; inspected the concrete Target/Reference nesting and snapshot lifetime needed to assess the fix. Hashes and diff reconstruction were bounded Python standard-library operations. Root's report of checking all12 fix pins is not represented as my independent full-manifest verification.

Only this canonical review file was written. No tests, Cargo, project helpers, native/HE/server/SSH/network/install commands, process/dataset-directory scans, source/public-doc mutation, commits/pushes or subagents were used. The original supervision, wire transition, endpoint, JSON, timing, Full/Quick scope and FULL2/queue/native-release boundaries were not reopened. The remaining correction is the finite R1 root/child-operation rule above.
