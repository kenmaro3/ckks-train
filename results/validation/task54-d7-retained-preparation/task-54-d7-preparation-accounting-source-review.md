# D7 phase2 preparation/accounting — independent source review

2026-11-07. **SPEC PASS / QUALITY PASS with one Minor runtime-inventory clarification.** No Critical or Important source finding. The implementation is consistent with the accepted phase2 design/FIX1 and the two authorized source addenda. This is source review only: no runtime test result, canonical-case success, phase3 route, genuine gate/start or scientific acceptance is asserted.

`L/` means `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`; source paths are repository-relative. Dispatch BASE is the supplied `68b9e96659ec6b2408e2d832b743d279571b5588`; concurrent root documentation work does not change the reviewed source identities.

## Evidence identity and scope

Read the source release and addenda, accepted proposal/FIX1 and review `3a129a3e`, complete author report, complete prepared source diff including new study_execution.rs, and relevant current source. Independently verified:

| Artifact | SHA256 |
|---|---|
| `L/task-54-d7-preparation-accounting-source-report.md` | `ff6c8a53e3f6b53e715e4c7a0d07d90f2e829a68005989d941a32db10896ed7a` |
| `L/task-54-d7-preparation-accounting-source-review.diff` (108923 bytes) | `f2d4164f8d57072d2c45a650fb013aac3ab3c77cdd839dda2f5168f1bea34299` |
| `L/preparation-accounting-source-validation/source-pins.json` | `49145a3dae3c4859ae10bc912c5a82c64e213c28aff42a27f31e2a48e15a6604` |

All twelve current source-file hashes match that manifest. The saved new-study-execution.rs equals the current new file. All twenty frozen descriptor fixture pins and both listed unchanged P1 source pins match current files. These are selected source/fixture identities, not exhaustive transitive attestation or a runtime byte-comparison result.

## Minor R1 — distinguish the retained BootParams fixture from scalar default-value calls

**Evidence:** author report line 88; `study_execution.rs:361–366`; `study_admission.rs:1013–1017,1363,1917`; `ckks-train-ckks/src/bootstrap/params.rs:35–82`.

The report's “one default BootParams” accurately describes the retained fixture owner, but is not the total `BootParams::default_n16()` invocation count for the proposed successful canonical test. Each of the three NI cases also invokes `secure_parameters` during plan binding output, preparation binding count, preparation binding output, and candidate description: twelve additional default-value calls, or thirteen including the retained fixture. The extra refusal cases are I and add none. These values contain scalar fields only; they introduce no extra CkksParams owner, P1 metadata/proof, heap buffer, key, native or HE constructor.

Before freezing an exact runtime inventory, label the existing count as **one retained BootParams fixture plus twelve temporary scalar default-value constructions**. No source change or compiler/test rerun is required for this clarification. Other reported primary construction counts remain consistent with the selected source paths.

## Ownership, associations and failure review

- `study_execution.rs:153–276` stores independent input/request and parameter/P1 lifetimes, seeds accounting from the supplied immutable request once, passes `Some(&mut accounting)` into both canonical checkers, and retains checked state, original input/case/request borrows and accounting together. Fields drop checked state before accounting; the complete descriptor bundle drops descriptors before preparation. The compile-use helper preserves independent lifetimes. No checked-by-value extraction or D7-to-legacy consuming adapter exists.
- `study_client.rs:161–320` and `study_client/blocks.rs:48–231` preserve explicit legacy `None` delegation, while the retained path uses the accounted binding validator. Exact actual partitions and canonical bindings are checked before candidate construction; complete key-manifest equality, graph/bias/activation/parameter/P1/BootParams/EvalConfig checks precede extraction. The supplied plan Arc is retained by clone, not replaced by an equal-hash plan. Actual parameter/P1 references remain the supplied owners. Future supplied crypto-Arc/owner rechecks remain phase3 work, not something these pre-key descriptors prove.
- Candidate extraction (`study_admission.rs:438–464`) returns the original candidate on its issue/missing-manifest boundary and moves successful geometry/key fields. Single preparation drops unused manifest copies and moves geometry; block preparation moves the training union. Existing placeholder states and geometry-digest storage are dropped rather than retained. Accounting/descriptor/allocation errors use their own variants, not manufactured candidate issues.
- Reservation creation precedes new activation/transpose/description/class output allocations. New locals are declared after the reservation, so failure destroys their buffers before releasing its remaining temporary charge. The single owning iterator is explicitly dropped before releasing its outer-buffer charge; the block owning iterator ends with the loop before release. Retained transfer preserves the combined charge. The enclosing preparation remains alive on collector failure until collector-local buffers and reservations have unwound.
- The three specified `Ok` escapes are removed: production `prepare_initial` and the two synthetic helpers encrypt within their callbacks. Legacy encrypted outputs may still own geometry and do not acquire D7 accounting. No claim that callback completion destroys that geometry remains. New retained outputs cannot enter those wrappers through an exposed extraction API. Actual live-frame transfer, block-copy accounting and crypto/session teardown are still absent phase3 obligations.

## Accounting and representation review

`PreparationAccounting` (`study_execution.rs:62–152`) checks the original plan charge/record count, checked aggregate reservation, and transfer/release without mutating state on rejected transfer. Private fields and exclusive reservation borrows preserve the subtraction invariants. The geometry helper (`ckks-train-nn/src/diagonal_plan.rs:253–287`) includes the private identity vectors, shapes, mask structs and step buffers, transpose headers/tuples/chains, rotations and compact vectors. CompactPlan/ContainerSpec/BiasSlot have no omitted nested heap fields. The source counts logical requested lengths and newly owned headers once; capacity and allocator/RSS overhead remain outside the agreed convention.

The binding count/output implementation (`study_admission.rs:1659–1739,1811–2147,2262–2285`) shares validation and serialization. Successful counting creates empty Vec/String roots only, traverses borrowed rows/partitions, and does not allocate output transcripts, identity bytes, sorted pointer buffers or digest strings. Preset validation now compares borrowed bands and scalars. Metadata-name uniqueness and stable source ordering use borrowed scans; original stable tie order is preserved by the original index. No duplicate SHA, alternative binding body or guessed multiplier was introduced.

The reported successful application-payload peak matches the implemented live scopes:

`max(Z + 64, Q + T + B + H + J + 256 + C + 64, B + 64)`.

Geometry transcript/hash precedes counts and partitions. Counts, partitions, both dual-identity buffers and their four digest strings remain live through the consumed transcript/hash and final binding append. After canonical_binding returns, discarded tuple fields and its other locals are gone before the validator's final plan hash. The accounted wrapper reserves the computed maximum before output traversal and keeps that reservation through validation. Counting still applies existing transcript limits; original request limits/record semantics and old P2 exclusions are preserved. This is not an all-preparation or physical-memory bound.

`study_reference.rs:136–211,269–291,392–758` uses shared count/output identity traversal, checked accumulation and fallible buffers/digest copies. `data.rs:369–386,429–448` reserves a 64-byte output String fallibly and shares existing SHA finalization/compression with the legacy path. Hex output fits the reservation; no new digest algorithm or byte format is used. Allocation errors remain typed on the new accounted paths.

Descriptor Fields/Writer bodies (`study_reference/document/execution_descriptors.rs:127–154,227–245,288–354,388–420,468–494`) retain their order/primitives and unchanged A_MAX. The fixed collectors (`679–809`) count/reserve complete outputs and the I/NI digest storage before allocation, encode I parameters once, borrow NI P1 bytes, and derive the cache inside the same-owner collector. Arbitrary-hash helpers/cache remain module-private. Size placeholders only enter count mode. Original no-cache source methods still derive their own hashes; new collector hash allocations use the fallible hook. Only the authorized fixed collectors are re-exported.

The new Transcript length counter is updated after successful appends, participates in metadata bounds, and is reset to zero after fail_record transfers the preserved prefix (`study_admission.rs:505–590,2635–2649`). The saved prefix/index/metadata rollback remains intact; the new pure regression checks truncation, transfer, subsequent append and count/output length agreement. This review found no unhandled mutation of the new counter in the changed production paths.

## Proposed tests and runtime limits

The five authored tests are meaningful for the released layer: pure reservation/overflow/cleanup boundaries; six real canonical retained-source preparations with independent owner checks and two retained refusals; fallible SHA known vectors and reservation failure; and the pure Transcript prefix regression. The canonical test has three materializations, two secure CkksParams owners, one real DFT proof verification and one NI BootstrapMetadata owner. Full materialization precedes Quick consumption; the proof performs real scalar arithmetic, and neither is constructor-free. Pure budget/SHA/prefix tests contain no hidden parameter/P1/planner/materialization or native/HE calls.

For all six intended successful cases, the reported 24 geometry, 24 initial, 24 update, 12 handoff and six plan-wrapper calls agree with source. The exhausted-budget refusal attempts an additional work charge but executes no extra P2 call. The three extra binding validations yield fifteen output plus eight count traversals overall, 46 borrowed update-row visits, and the stated 24 count/16 output traversals per identity encoder. The descriptor output/count/hash inventory and the partial 97 core plus 78 descriptor/comparison hash inventory also reconcile with the selected paths; registry/materialization hashing is correctly excluded from those partial totals. R1 supplies the scalar BootParams qualification.

These counts are conditional source inventories, not measured runtime counters. The canonical test's comparison buffers are explicit test-only allocations outside the production collector receipt. It covers Dedicated transpose and two-update I/NI cases; it does not prove every Reuse/full-schedule/failure-allocation branch. The zero callback Cell applies to the legacy refusal; the new D7 path has no crypto factory to invoke. No allocation/RSS instrumentation or production-route evidence is supplied.

The listed old pure SHA/encoding/descriptor/geometry regression candidates remain appropriate for separate exact selection. The P3 structured golden is explicitly not constructor-free: it reaches its shared registry/tiny-parameter/proof/metadata fixture. Do not silently include it or a broad P3/P4/P5 suite in a pure release. Frozen negative/prefix expectations remain unchanged; selecting any of them later requires its own runtime inventory, not a source-review rerun here.

## Validation evidence and remaining boundary

Read saved final command records/logs: no-run compilation 20 and clippy 21 report exit 0; no-default/cuda feature checks 18/19 report exit 0 before the subsequent cfg(test)-only prefix addition; final formatting/diff checks 22/23 report exit 0. Initial RED separately lacks an observed Cargo exit, as its metadata and report candidly state; enclosing shell exit 0 is not a Cargo success. Preserved later failures and fixes are not erased. No compiled test is yet claimed passing at runtime.

No source, Cargo/test, constructor, planner, materialization, native/HE, network or Git command was executed by this reviewer. Only evidence/source reads, searches and read-only hashes/comparisons were performed; this review file is the sole write. No subagents were used.

Source acceptance does not release execution. MissingExecutionDependency, explicit immutable request/consumption wiring, fixed CPU continuation, same-owner post-start/live accounting, outer binding/runtime policy, actual gates/durable admission/start and scientific publication remain open. Existing P1/P3 limitations, encrypted-block holds, unresolved B/same-N/authors decisions and the full 21 / full 2680 + 20 + 7 / quick 268 obligations remain unchanged.
