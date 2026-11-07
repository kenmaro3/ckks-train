# D7 phase2 preparation/accounting — scoped FIX1 design re-review

2026-11-07. **SPEC PASS / QUALITY PASS for FIX1. Important I1 addressed at design level.** No new Critical, Important or Minor finding in the changed material. This closes the original design finding; it does not establish source implementation, compilation, canonical-case success, or runtime readiness.

Scope: original review `1fa646266d31593ef5ad0b8b0760617a88a5be2fb2ddd908d440724efb374b6f`, I1 ownership escape/lifetime correction and new breakage introduced by its fix. M1/M2 are carried source-acceptance requirements, not a renewed general architecture review. `L/` below denotes `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`.

## Artifact identity

Independently verified:

| Artifact under L/ | SHA256 |
|---|---|
| `task-54-d7-preparation-accounting-proposal-before-fix1.md` | `e51db6786477a1271183f3c54f85cd9b02e5d7ece179bb41573ca389fa0df55e` |
| `task-54-d7-preparation-accounting-proposal.md` | `6de4d485e49ef2dd27401cccc5d58fe2eebc3e1aa37a33c0e82600e5309836be` |
| `task-54-d7-preparation-accounting-fix1-report.md` | `0ecf7e6a9b90f37f0b25c2694bcae5b4d772d7967442235548c3baf80ac850a8` |
| `task-54-d7-preparation-accounting-fix1-review.diff` | `0fe344969b48db2100eff988e9476ebafa4cf0da9ac75b5fbd068712ed8dfbe8` |

Read the complete revised proposal, complete FIX1 report and complete supplied diff. A read-only text comparison confirmed that the diff exactly represents the preserved-original to revised-proposal change.

## I1 disposition and source evidence

**Independent lifetimes are now coherent.** Revised proposal lines 18–50 distinguish input/request lifetime `'i` from parameter/P1 lifetime `'p`. Shared checkers return `CheckedPreparation<'p>` or its block counterpart without retaining input borrows in that checked value. The enclosing retained result separately stores `'i` references. The existing private wrapper can therefore delegate without imposing either outlives relation or lengthening a shortened borrow. The proposed compile check remains necessary source evidence, not an already-passed test.

**The legacy/new accounting boundary is now explicit.** Proposal line 112 makes compatibility wrappers call the shared checker with `None`, while the new D7 preparation always passes its same accounting owner through `Some`. Legacy wrappers do not destructure a D7 retained owner and do not acquire a D7 accounting claim. A shared optional-accounting implementation is feasible provided its callers remain private and source review verifies the fixed call graph. The no-accounting branch is not a route for a retained D7 owner.

**The concrete checked-object escapes have been identified and assigned corrections.** Current `study_client.rs:419–436`, `study_diagonal/p4_tests.rs:197–208` and `study_blocks/p5_tests.rs:1005–1018` return checked values through `Ok` before encryption. FIX1 requires moving those existing encryption calls into their callbacks. The report correctly treats this as an audited private-call-site rule, not a type-system guarantee supplied by a generic callback. A search of current wrapper invocations agrees with the report's enumerated caller groups. These edits remain future source work; no existing tests were run or changed in this review.

**Accounting follows payload beyond encryption.** This is the essential extension beyond simply moving encryption into the callback. Current `study_client.rs:502–509` moves Description into EncryptedInitialState. `study_client/blocks.rs:328–345` clones descriptions/classes into BlockEncryptionBinding and moves the originals into EncryptedBlockInitial. Session construction takes those fields onward (`study_diagonal.rs:405–411`; `study_blocks.rs:419–426`). FIX1 explicitly recognizes all these owners rather than treating encryption as geometry destruction.

For phase2, the fixed collector consumes and returns the complete retained owner; `PreparedCanonicalBundle` declares descriptors before preparation, and RetainedPreparation declares checked state before accounting (proposal lines 25–30,91–99). There is no by-value checked extraction or generic factory on the new retained interface. Normal bundle drop therefore destroys descriptors/checked payload before accounting. Collection failure must likewise keep its owning preparation live until newly allocated outputs are destroyed; the stated whole-owner/error-cleanup contract remains a source-review check.

For future phase3, proposal line 114 requires moving the **same** accounting value and descriptors into the live frame before transferring checked geometry into encryption. The returned fixed driver keeps setup and live frame together, destroys setup/crypto before live, and destroys live descriptors before accounting. Failure scopes and execution scopes retain live outside checked/session/binding/crypto owners. The block copy is explicitly reserved before cloning; the single Description moves. This satisfies the intended lifetime of the retained payload without rebuilding geometry, resetting the receipt, changing public legacy encrypted return types, or adding a readiness capability. The fixed body and actual drop ordering remain separately reviewed phase3 implementation obligations.

## Changed clarification notes and retained scope

M1's revised inventory now names preset/extension vectors, the final plan-hash String and four identity digest strings. Stable-sort library scratch is explicitly excluded from application requested payload and is not claimed absent. Source acceptance must verify the actual live-interval maximum, successful allocation-free size traversal, original limits and this stated accounting convention; this review supplies no measured allocation or RSS result.

M2's digest cache and `_bytes_with` methods are now confined to the descriptor module's fixed same-owner collector (proposal lines 75–83,134). Existing source helpers still derive their own hashes; size-only placeholders cannot reach output. This addresses the intended source-projection confinement without a new identity framework.

No new breakage was found in these changes. Old P2 allocation exclusions remain explicit; retained D7 accounting does not retroactively bound old P2 work or legacy encrypted outputs. MissingExecutionDependency, explicit future request/consumption wiring, genuine gates/durable start, post-start checks and teardown remain mandatory. The six-case work inventory remains conditional and unexecuted. Full 21 / full 2680 + 20 + 7 / quick 268, existing holds and unresolved scientific decisions are unchanged.

This reviewer performed document/source reads, searches, hashes and a read-only diff comparison only. No source mutation, Cargo/test execution, constructor, planner, materialization, native/HE execution, network access, Git operation or subagent was used. This review file is the only write. Source release and any later execution decision remain separate.
