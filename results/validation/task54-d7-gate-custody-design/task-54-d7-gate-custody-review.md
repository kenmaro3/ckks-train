# Task54 D7 gate evidence custody — independent design review

Date: 2026-11-07. Scope: carried M1's concrete publication/custody refinement, against ratified gates design and frozen target §§6–7. This is not a repeat of the Full/Quick scope review, a source review, or a runtime release.

## Verdict

- **SPEC: PASS.** The proposal concretely resolves M1's custody paths, closed names/caps, guarded ordering, retained evidence identity, partial-state eligibility and recovery requirements. Full's existing Initial/Retry and live-token rules remain intact. The explicit separate Quick adapter and Initial-only limitation are coherent proposed refinements for root ratification.
- **QUALITY: PASS.** No Critical, Important or Minor finding in the reviewed design. The remaining schema/vector and exact fault-inventory work is explicitly identified as required before its corresponding encoder/source package; it is not claimed complete here.

This accepts the design's adequacy for the carried M1 requirement. It does not establish actual durable publication, genuine gates, executed Quick/Full lineage, scientific results or Task54 completion.

## Reviewed identities and actual API check

The complete 171-line proposal was read, and the prepared new-file diff was checked to contain exactly that complete body.

| Artifact | SHA256 |
|---|---|
| `task-54-d7-gate-custody-proposal.md` | `fc93856110cb2e8c7382fe792b21cf0550a85007d9267d09f75d6e9637e8c73f` |
| `task-54-d7-gate-custody-review.diff` | `6618926f05f81b35fb406b38516e0f86e64f32875b884eec4ff26170d0dd2d62` |

Compared with `docs/study-gates-design.md:13–35`, `docs/study-target-design.md:553–718`, and the relevant unchanged publication/target APIs:

- `crates/ckks-train-bench/src/study_provenance/publication.rs:28–45` implements the stated per-file publication/recovery order and BeforeLink/VisibleUnconfirmed distinction. Native at `:378–488` uses exactly counters 0..63, O_EXCL creation, complete writes, file sync, no-replace linkat, parent sync, inode-checked owned-temp cleanup and parent sync. The proposal does not invent a rename/overwrite transaction or imply that BeforeLink rolls back earlier sidecars.
- `publication.rs:500–507`, `:849–854` distinguish ordinary private regular-file checks from the guard's additional zero-length/one-link requirement. The proposal correctly permits a final and a residual temp to share an inode after an uncertain publication; imposing final nlink=1 would wrongly reject that recoverable state.
- `publication.rs:516–584` provides retained ancestor handling/create-or-open/existing-only helpers; `:594–616` reads stamped bytes but does not return its local File. The proposed held-file helper is therefore real additional source work, correctly labelled as such. Exclusive named-attempt creation is also a proposed operation, not assumed to follow from create-or-open alone.
- `publication.rs:645–736`, `:766–840`, `:855–902` preserve the existing role-sensitive target opening, lock-only publication/recovery and strict fresh-profile claim. `crates/ckks-train-bench/src/study_reference/target.rs:108–113`, `:155–164` confirm that ProfileTargetStore has no current ActiveAttemptHandles and VerifiedTargetForSeed owns the claimed snapshots/receipt/threshold. New case/attempt/sidecar and Quick APIs are not misrepresented as already present.

## Substantive assessment

### Exact custody without a new authority source

Proposal lines 17–21 and 63–82 define closed typed selectors, fixed branch-local names, fixed caps and Native temp IDs. Five binary sidecars contain exactly the bytes whose whole-file SHA256 appears in the existing Gates fields; no transport canonicalization, wrapper or self-hash substitutes for those bytes. The branch/policy/scope must match the actual producer. External referenced artifacts still require their existing verified reader/custody, not a path taken from a sidecar. The publisher remains an I/O foundation and cannot manufacture a gate or live token.

The final-name/temp inventory is internally consistent: Full has seven final names and 7×64 recognized temps; Quick has eight and 8×64. Canonical decimal counters 0..63 match Native's generated grammar. Full profile target-lock remnants remain separately recognized for post-lock inspection, while the existing Reference claim_new/profile_empty still rejects every non-guard entry. Bounded-memory streaming scans, nofollow regular-file/owner/mode/mount/cap checks and unknown-entry rejection prevent a permissive leftover-file wildcard.

The stated ceilings were independently recalculated: five A-sized evidence files + 70,492,488-byte admission + A-sized start = **83,075,400** final bytes; Quick's eight A-sized finals = **16,777,216**. Existing admission/lock canonical and file caps are not enlarged. The new evidence A limit is explicitly administrative and may cause refusal; it is not presented as proof that every eventual schema fits, physical memory admission, disk reservation or a scientific threshold.

### Full ordering and immutable evidence transfer

Proposal lines 86–101 require genuine gates and bounded canonical records before attempt mutation, the real verified target before case creation, and guarded frozen eligibility before exclusive named-attempt creation. The same guard covers all five ordered sidecars and admission. Every sidecar must complete durable publication and exact held-file/byte/hash checks before admission begins. A capacity recheck may validate the current lease, but cannot silently rewrite the original resource snapshot under its old hash.

Admission/start preserve their existing bytes and same-store borrow. Evidence Files/stamps and anchored handles remain held and rechecked through admission, start and begin. Before closing those handles or releasing the store borrow, begin transfers immutable evidence/links and actual live owners/accounting/leases. The driver receives no hidden disk-reopen authority; runtime rechecks continue against genuine live obligations. This fits the ratified ownership boundary and does not conflate a durable snapshot with a continuously updated measurement.

### Partial states, uncertain finals and Retry

The table at lines 119–140 matches the frozen conservative rules. Any named attempt blocks Initial, including empty, temp-only, partial-evidence and complete-evidence/no-admission states. Those states cannot be promoted into an admission or used as a Retry predecessor. Missing/malformed/hash-mismatched evidence invalidates a claimed admission. A fully verified admission with its complete evidence may become an explicitly selected Retry predecessor after all required resyncs, even if the original publication returned uncertain; this creates an inspection/durability record, never a recovered live capability.

Expected-start=None tests final absence, not presumed process death; a safe recognized start temp neither satisfies a start nor proves inactivity. A present final start requires the exact expected links and valid evidence/admission. A different or appeared/disappeared final rejects immutable expectations. The absent new attempt path remains mandatory. Unrelated attempts are preserved without being treated as completed, and Retry may branch from an explicitly authorized valid predecessor rather than a computed latest/head. User/resource gates retain responsibility for possible overlap.

Lines 144–148 make verification precede resync, define final-inode and directory ordering, and reject new B creation on any validation/sync failure. Recovery preserves temps, cannot finish a missing publication, cannot repair evidence from current inputs and cannot mint runtime/context/resource/user authority. A complete uncertain final and an orphan are deliberately different states. Recognized hardlink remnants do not become cleanup permission. Blocking an orphaned first case in its existing campaign and prohibiting automatic campaign rotation preserve the frozen policy rather than invent a new reset path.

### Quick is separate and operational

The `quick-campaigns` tree, q-registration slot and narrow QuickProfileStore do not alter Full's existing-only Train opening or its target/profile claim rules. The stable Quick case slot excludes attempt identity, so changing only an operational attempt cannot bypass Initial eligibility. Empty administrative Quick ancestors may survive bootstrap, but case/attempt/evidence creation still requires checked gates and guarded eligibility.

Lines 105–111 define durable evidence → preflight → start dependencies and a same-process live entry. The installed outer caller retains origin-bound handles without holding flock during the driver and may publish one linked end after actual return, including observed setup failure after start. Lost callers/recovery cannot fabricate an end. No target, Full predecessor, StartedHeCase or prospective result follows.

Initial-only Quick is an explicit limited proposal, not an accidental reuse of Full Retry. A repeated case requires a recorded separately selected namespace within actual existing authority, with original evidence retained; no automatic campaign increment or extra user-confirmation workflow is added. Missing end stays incomplete. This is consistent with Quick's diagnostic role and does not reduce required matrix coverage.

## Source-acceptance notes retained from the proposal

These are required follow-through, not additional design findings:

1. **Before encoder source:** freeze the actual evidence common/branch/tail primitive schema and domains, all ten gate×branch literal byte/hash oracles, and Quick preflight/start/end fields/linkage/mutation vectors. Include size proofs at the fixed caps and actual external evidence-reader custody. In particular, demonstrate that the selected common ExecutionBinding and any retained referenced facts fit within A without changing frozen encodings or silently substituting hashes. An encoding that does not fit must refuse or return for an explicit refinement, as the proposal already requires.
2. **Before protocol test execution:** turn C01–C10 into the exact fixture, file-operation/constructor counts, injected cut positions and observable postconditions. Preserve all eight ratified Full/Quick branch tests. Include final-link/cleanup uncertainty with a real surviving hardlink, every 0..63 temp collision/exhaustion case, first-attempt sidecar failure blocking a competing Initial, missing/changed evidence, and every required recovery-sync failure preventing B creation. Synthetic controls remain incapable of producing production gates.
3. **At source review:** verify exclusive creation rather than create-or-open reuse for the new named attempt; immutable record/read-buffer accounting against original budgets; fresh bounded reads at each held-file check (with correct read offsets); exact path-to-inode rechecks; owned-temp-only cleanup; late failure semantics; no recovered-to-live conversions; origin-bound Quick end publication; and crypto-before-lease/accounting teardown. Current helpers do not by themselves implement this combined protocol.
4. **At a later bounded native release:** separately establish Linux ext4/XFS nofollow/link/flock/fsync and process/fork behavior plus real gate-to-begin lineage. Pure fault mocks, Mac refusal, this design and existing disconnected CPU bodies do not establish native durability or successful HE/scientific execution.

## Review limits

No source package or concurrent actual-binding diff was reviewed or edited. No Cargo, compiler, test, constructor, HE, SSH/network operation, Git mutation or subagent was used. Only this review file was written. Ratification is a root design decision within existing task authority; this review adds no approval/science gate and changes no hold, threshold, full inventory or lifetime contract.
