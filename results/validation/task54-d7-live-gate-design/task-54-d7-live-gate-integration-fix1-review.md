# Task54 D7 live-gate integration — scoped FIX1 review

Date: 2026-11-07. Scope: original Important I1 and new breakage in FIX1 only. Design review, not source/runtime release.

## Verdict

- **SPEC: PASS for FIX1. I1 addressed.** The closed Full/Quick scope, mandatory branch-specific checks/evidence and separate consuming boundaries resolve the original contract gap.
- **QUALITY: PASS for FIX1.** No new Critical, Important or Minor finding in the changed material. Original M1 remains a mandatory pre-source custody deliverable, not a completed implementation or another fix-loop finding.

The proposal is suitable for root design ratification with the carried source requirements. This review does not itself implement or execute genuine gates, durable admission/start or either driver.

## I1 resolution

Evidence below refers to the revised `task-54-d7-live-gate-integration-proposal.md`.

1. **Closed shared boundary (lines 66–72).** Each of the same five constructors accepts a private Full/Quick GateScope with checked common owners/identities and mandatory branch-specific fields. Full requires VerifiedTargetForSeed-derived reference/lock/threshold links and Full consumption; missing or placeholder target values cannot satisfy it. Legitimate numeric zero is correctly distinguished from unverified placeholder provenance. Quick carries exact full registration only as registration provenance, a separate operational identity, seed0/two-update consumption/order, actual consumed plan/request and operational attempt. It has no target/threshold/full predecessor fields. This resolves the original need to invent optional-target bypasses or counterfeit full links.

2. **Inputs and capability consumption (lines 86, 92, 113 and 147).** Common canonical source/materialization validation remains mandatory. Full additionally compares the independently materialized selected-seed target receipt; Quick performs its own registration/order/consumption validation without asserting that comparison. Every capability is sealed to its branch. Only five Full capabilities can assemble CheckedCaseForAdmission; only five Quick capabilities enter the private Quick bundle/entry. Mixed bundles reject. Full admission rejects Quick before namespace mutation, and no parsed evidence or conversion can promote Quick into NewlyPublishedAdmission or StartedHeCase. The implementation/layout/resource/user obligations and live rechecks remain substantive for both branches; Full's admission/start/begin and Quick's operational entry/pre-allocation checkpoints are explicitly distinguished. No sixth gate is introduced.

3. **Evidence identity (lines 94–101).** The new inner evidence records use an explicit closed scope tag and exactly one mandatory branch body. Full records its verified target/reference/threshold/attempt and independent target-receipt comparison. Quick records operational identity and consumed request/order with no target fields or null placeholders. Equal registration/input digests do not make the branch evidence interchangeable. Frozen CaseIdentity remains full-only; existing ExecutionBinding, full Gates and admission/start encodings are not extended or relabelled. Process/context-instance identity remains live-only. Exact domains, canonical tail details/caps and independent vectors are still required before encoding source; the design does not present them as already frozen artifacts.

4. **Meaningful proposed assertions (lines 151–162).** The eight named tests cover missing Full target, target-free canonical Quick control, Quick/mixed-bundle rejection by Full, placeholder provenance, equal-hash cross-scope refusal, exact Quick consumption/order/attempt, independent target-receipt mismatch and preservation of frozen full vectors with distinct new evidence. These are appropriately source acceptance requirements with fixtures/call inventory to freeze before execution. The nonallocating control is labelled branch-wiring evidence, not production admission or scientific success. No test result is claimed here.

## Changed-material checks and carried requirements

- **Context promotion remains sound at design level (lines 74–76).** The accepted inert process-wide instance number is checked for overflow, never reused/wrapped, and paired with ProcessStamp. Pre-admission gates retain owned facts, not a context token borrowing for `'p`. Full admission compares the original instance/process and store-context pointer, obtains a fresh token borrowing actual context for `'c`, compares facts and retains it into the live frame. Quick performs the corresponding actual-context promotion without claiming a target store. The change does not add `'c:'p`, a dereferenceable stale pointer, a deserialized authority or a long-lived single-lifetime TrainCapture. Existing runtime/pool/evidence checks remain required. The fact projection is explicitly proposed private API work, not misrepresented as an existing constructor.
- **M1 is carried concretely (lines 101–103 and 172).** FIX1 requires closed branch namespaces/names/entries/caps, relative durability, guard/held-FD/no-clobber/fsync ownership and recovery treatment of partial publications before protocol/gate source. It prohibits publication before guarded eligibility and preserves unknown-entry/orphan/named-attempt refusal. Admission hashes must identify exact retained evidence bytes; Quick has its own operational linkage. The detailed transaction/fault inventory still must be delivered before source. This satisfies the requested carry-forward, not runtime durability validation.
- The diff does not relax original correctness prerequisites, scientific thresholds, the full encrypted-block hold, matrix denominators, original budgets/caps or independent lifetimes. It does not add another user approval request. Unchanged material was not reopened, and the concurrently unfinished CPU continuation was not reviewed.

## Custody and evidence scope

Read the complete FIX1 report and complete prepared diff, and checked the affected proposal sections directly. Independently reconstructed the unified diff from preserved original and revised bytes; it matches the supplied diff exactly.

| Artifact | SHA256 |
|---|---|
| Original proposal preserved before FIX1 | `f007784a7ccbb99574dfe0a1586e186534d4c5ba9904e716e1880d3af239cf5b` |
| Original independent review | `2f629f5c28502703eefb66df695c97edd98db513a02e6fd6dba5f646c7182100` |
| Revised proposal | `c667176c41f1e5dda060e552c6fe9db1209446c76c3f2161ea949571cb539e5c` |
| FIX1 diff | `58c2e0d606d1107120c8752e62396108594b5610bb4c39276ae26e3c93305f2e` |
| FIX1 report | `f1492a674d549a8b98dd0b5aeb7010c7e507361989a20c99010f4ee4e388c8a6` |

No Cargo, tests, constructors, HE, SSH/network operations, source or Git changes, or subagents. Only this review file was written. PASS is limited to the scoped design correction; source compilation, native durability, real gate construction, runtime ordering and scientific outcomes remain unobserved by this review.
