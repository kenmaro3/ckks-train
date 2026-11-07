# Task54 P3 independent SPEC + QUALITY source review

2026-11-07. **SPEC: changes required. QUALITY: changes required.** The frozen source has a concrete completed-prefix loss after successful composition, incomplete context for operational failures during composition, and materially incomplete required identity-mutation coverage. These findings do not invalidate the recorded focused PASS results; those tests do not discriminate the affected behavior. The separate P1 construction-limit contract discrepancy below also needs an explicit disposition.

This is a source/evidence review only. No source/test/build/constructor/HE/network/Git mutation or new execution occurred. The only written file is this report. The original full P3 pre-key spy remains **REQUIRED AND PENDING** at the actual P4/D7 consuming boundary. Neither this review nor the current implementation completes full P3, Task54, production admission, or a scientific campaign.

## Exact reviewed material

Read the complete review brief, parent Task54 brief, 152-line API map, design sections 2/4/6/7/9, source release, storage amendment, bootstrap-binding amendment, filtered getter release, prefix release and its nonempty-payload amendment, frozen regression release, implementation report, final manifest, complete new module/tests, and narrow existing-file diffs. Relevant P1/P2/materializer/reference implementations were inspected to check the actual reused contracts. No applicable AGENTS.md was found in the repository or ancestor paths checked.

| Current reviewed file | SHA256 |
|---|---|
| `crates/ckks-train-bench/src/study_admission.rs` | `4d6b1cc66f820b5782c6c9831f24eb94f3eef999e6dd2a572e38007f254fbb56` |
| `crates/ckks-train-bench/tests/study_admission.rs` | `85c13c334c3956cde76b2992130155d7225f214cc8087c1187520c1b69d88a0e` |
| `crates/ckks-train-ckks/src/bootstrap/metadata.rs` | `07ae0174669d5c494836cfb2cb9bddbbc1173cc70cf304a96e5d65e2d7013a12` |
| `crates/ckks-train-bench/src/lib.rs` | `867b2c6d8f4391861956c10483fe18c91636e9272e41cceee8953b90976b3e44` |
| Implementation report, 84 lines | `c339fde5f25c875a52101ff9a051480b29e00f1faf0dbd8fe954671449a47442` |
| `task-54-p3-validation/final-manifest.txt` | `768915b9aa9c88be34a69adc7855e435eff7388edd535e895886333ab7b52314` |

All 24 manifest pins were independently rehashed from disk with no mismatch, including raw logs and exact frozen source copies. This validates artifact identity, not their scientific or semantic correctness.

## Load-bearing findings

Severity labels below mean P1 high, P2 medium and P3 low priority; they are not the P1/P2/P3 implementation-package names.

### R1 — P1: successful composition is discarded if final retained payload cannot be stored

**Location:** `study_admission.rs::plan_diagonal`, lines 1847–1863, especially the bare `?` calls after `compose` returns.

**Requirement:** API map section 4, source release decision 2, design section 9 and the prefix release require every capacity/allocation failure to retain the complete indexed prefix and actual completed update count, with no shorter successful plan.

`compose` returns a fully populated index after all intended updates. The wrapper then charges/copies full rates and strings and constructs the bounded key manifest. Failure in the final combined-byte charge, `copy_vec(numeric.rates())`, or `bounded_manifest` returns the fresh helper error directly. That error has `completed=0`, empty prefix and empty prefix index. The local transcript/index are dropped even though every consumed update completed. This is a normal caller-selected cap failure, not merely a hypothetical allocator failure: a cap can allow the complete transcript/index but reject the additional full-rate payload charged only after composition. The real Iris path retains 2,400 full rates despite consuming two updates, so the late charge is substantial.

**Smallest correction:** give finalization the same complete-prefix error attachment as composition, or move the final retained-payload preflight/reservations before composition while still providing a common failure wrapper for any fallible post-composition operation. A genuine finalization failure should say all consumed updates completed and identify finalization as its phase; do not invent a failing stage or a level failure. Preserve exact required/limit for the operation that failed.

**Smallest discriminator:** reuse an already authorized canonical positive fixture and its deterministic payload accounting. Choose a cap that allows both consumed updates and their complete index but rejects the final retained-rate/key charge. Assert `PlanningBudgetExceeded`, the actual required/limit, `completed_updates()==2`, two complete indexed update receipts, valid ranges, and no complete plan. A focused shared finalization helper test can cover the error attachment without more parameters, but at least one real wrapper call must establish the production wiring under a separately released finite budget. Current integration's cap16 fails in the header and cannot detect this bug. No such test was run in this review.

### R2 — P2: operational failures lack the actual stage and work-limit diagnostics

**Location:** `study_admission.rs::compose`, lines 2218–2305 and 2317–2320; `PlanningBudget::charge`, lines 112–132.

**Requirement:** API map section 4 requires `PlanningBudgetExceeded{stage,update,resource,required,limit,completed_prefix}`; the review brief requires the actual failing stage/update and complete prefix, while preserving operational versus arithmetic failure distinctions.

Only the P2 `.map_err(|e| p2(e,s))` paths attach a stage. Errors from `index_room`, work charging, transcript encoding and typed-summary retention have `stage=None`; the final catch fills only the update from the number of completed logical updates. For example, an e3 request that exhausts `max_records` before stage1's optimizer record retains the prior stage0 update/handoff but cannot report that stage1 was the failing boundary. Additionally, `PlanningBudget::charge` discards the computed `next` and `limit`, leaving `budget_requirement()==None` even for an ordinary exhausted update/handoff cap. The resource text is also the same for every work category.

**Smallest correction:** attach current stage/update/phase around every fallible stage-update and handoff record operation, without replacing an existing P2 diagnostic. Use the actual computed next/limit and a category-specific resource name for normal work exhaustion. Keep the special poisoned-budget case distinct rather than inventing numerical limits for it.

**Smallest discriminator:** in the existing multistage synthetic composition seam, allow precisely the first stage update and boundary, then deny the next index entry; assert stage1/update0, completed logical updates0, exactly two complete preceding records and intact offsets. Independently exhaust an update or handoff work cap and assert exact next/limit and the correct stage/phase. These can share existing mini fixtures; any new runtime allocation/planning allowance still needs root release. Existing `max_records=1` test checks only kind/completed/nonempty bytes and would remain green with this defect.

### R3 — P2: named identity tests do not implement the required mutation discriminators

**Location:** `study_admission.rs::p3_digest_golden_and_mutations`, lines 2628–2902; `synthetic_binding_negatives`, lines 2565–2624; integration `p3_canonical_quick_full_identity_and_revalidation`, lines 179–280.

**Requirement:** API map section 6 explicitly lists one-field actual q/p, secret/BootParams, outside-quick coefficient, reference, consumed prefix, micro order/count, stage target/input, transpose pool, EvalConfig and retained-state-bit mutations, plus a typed independent golden and corrupt-claim trust discriminator.

The golden test is a useful independent 26-byte **primitive** oracle plus known SHA and real shared accounting/rollback helpers. It contains no bound receipt domain/version or parameter/graph/request/state mutation. The only rate digest mutation is in another test: it changes `source`, `mean`, `kernel` and the reference together in a two-entry rate-only stream. It does not isolate an unconsumed suffix coefficient while keeping an independently later full maximum. The canonical test checks its own hash and length, then rejects a malformed partition (19 rows versus the actual20) and NI parameters against an I case. Those are real negative paths, but they do not prove a valid alternative partition reaches complete binding comparison, nor the listed field-by-field identity coverage. There is no private corruption/rehashed-claim discriminator. The separately executed K14 getter test does provide a genuine BootParams negative with a reachable valid control; do not erase that valid evidence.

**Smallest correction:** add small table-driven encoding/binding mutation tests using the actual production encoding helpers and existing borrowed fixtures, with an independently fixed structured receipt vector that includes domain/version, options and at least one representative parameter/graph/request/state field. Isolate one binding dimension per case. Include a full-rate vector with a suffix change below an unchanged later maximum and a valid alternative micro partition of the same logical20 rows. Mark parser-only/unsupported unknown-tag checks honestly: there is currently no untrusted DiagonalPlan decoder to exercise. Any added private projection must remain incapable of making a canonical materializer or granting execution authority.

**Smallest runtime discriminator set:** the outside-prefix coefficient test must fail if encoding is shortened to the consumed prefix; actual q/p and retained-state-bit mutations must fail if their production encoder fields are removed; valid alternative partition must pass partition validation and then fail original-plan revalidation; transpose/EvalConfig and stage target/source changes must change the structured identity or return the exact intended axis/geometry failure. A self-hash assertion alone cannot provide these properties. No new tests were executed here.

### R4 — P2 contract omission requiring disposition: actual P1 construction limits are not bound

**Location:** API map section 2 BootstrapMetadata row and section 3 ordered field5; `study_admission.rs::encode_params`, lines 938–980; `BootstrapMetadata` fields/constructor in `ckks-train-ckks/src/bootstrap/metadata.rs`, lines 778–815.

The frozen contract asks for **explicit metadata limits** in addition to descriptor/proof information. P3 encodes P3 ReceiptLimits/work limits and P1 `resource_bounds()` (actual coefficient/interpolation/work/requested-byte charges), but not the four original P1 `MetadataLimits` ceilings. P1 accepts those ceilings in `with_limits_and_verified_support` and does not retain/expose them. Thus this is a real unavailable-input discrepancy in the specified operational identity, not a claim that the cryptographic descriptor or BootParams encoding is incomplete. Larger successful P1 ceilings do not change the mathematical descriptor.

**Smallest disposition:** root should explicitly decide whether the required identity is effective P1 result/bounds plus P3 limits, or truly includes original P1 construction ceilings. If the latter remains required, use a separately assigned minimal immutable original-limit capture/accessor and bind those exact values; caller assertions alone cannot establish actual constructor limits. If the former is the intended scope, record that interpretation against the map before stating all its fields are implemented. Do not silently equate a measured resource charge with its configured ceiling.

**Smallest discriminator if original ceilings remain required:** descriptors built under two known sufficient limit sets with identical crypto/proof inputs should have identical crypto descriptor bytes but distinguishable operational construction-limit identity. No such additional descriptor execution is authorized by this review.

## Quality correction and integrity boundary

**Q1 — P3 documentation:** `validate_plan_binding` lines 1887–1889 says it performs no allocation. It calls `canonical_binding`, which allocates transcripts, partitions, complete shared/selected identity bytes and digest Strings. Change the statement to no HE/key allocation or no HE construction; the implementation is neither allocation-free nor a total-RSS proof. No runtime rerun is needed just for this rustdoc correction.

`validate_plan_binding` compares freshly derived binding bytes with `plan.binding` and checks `hash(plan.bytes)==plan.digest`. It does not independently join the transcript prefix to `binding`, recompute all transitions, or check the typed index/summary fields against transcript bytes. **There is no current safe external forgery path found:** DiagonalPlan has private fields, no Deserialize/public constructor, immutable getters, and only the checked production wrapper constructs one. Consequently this is currently a validation of a trusted, locally constructed plan against actual current inputs/requests, not an adversarial parser for arbitrary transported receipts. Do not claim the existing tests prove rejection of arbitrary rehashed corrupt plans. A private mutation can demonstrate the narrower invariant, and any future parser/transport constructor must compare fresh complete canonical claims and maintain typed/transcript consistency before constructing this type. Do not introduce a public forging seam to satisfy a test name.

## Positive implementation observations

* The production wrapper requires canonical preparation and the existing sealed Full RegisteredCase; quick consumption is separately tagged. It does not alter frontend selection, provenance, scientific profile metadata or registration denominators. Candidate and cfg(test)-private synthetic plan types cannot construct DiagonalPlan or a session. Unsupported role/packing stays a typed issue.
* Full-plan rates/reference, canonical full identities and full update/checkpoint fields are encoded before composition; consumed row IDs, partitions and terminal lookahead have a separate domain. Actual ordered q/p integers and F64 bits are encoded. All nine BootParams fields, exact descriptor bytes and declared Galois/relin/conjugation/SSE roles are included. No Debug representation is hashed. The graph range/source/representation assertions and embedded complete shared identity bind canonical old-hidden roles.
* Secure preset checks rely on the currently sealed deterministic parameter constructor; inspection confirmed it deterministically derives actual q/p from ordered bit vectors. Receipts still encode actual integers. Mini parameters cannot admit secure production. No lowering or alternate scientific preset was introduced.
* Composition calls existing P2 geometry/init/update/handoff APIs, carries the returned state per stage, uses returned old-hidden handoffs, destination `states[s].batch().y`, all-next-micro fullness and terminalNone. No second recurrence, cache, fake evaluator or fresh state per update appears. I and WN/AU boundary work and AA canonical reuse come from P2 rather than copied formulas.
* The combined retained-payload ledger charges the transcript, binding copy, full rates, nested partitions, initial label arrays/structural summaries, key arrays, TransitionIndex structural storage and nested pre/post/incoming/raw/AA/handoff label arrays before retaining them. The nonempty exact2259-byte helper fixture exercises actual production retention/record helpers; it is stronger than a detached counter. `fail_record` rolls incomplete fragments/index entries back to the last complete checkpoint during compose. R1 concerns the unwrapped later phase, not that existing helper's positive behavior.
* P2 errors preserve their actual PlanError/trace/attempt/source/prime data without substituting a level error. The executed late failure is RateConflict with empty scale trace; the nonempty scale-trace preservation test uses a deliberately constructed PlanError passed through the real failure helper. These are different evidence and must stay labeled that way.
* The existing CKKS diff is exactly one private BootParams field, constructor clone and immutable getter (three hunks/seven added lines). Existing encoding/arithmetic/proof logic is untouched. The real K14/default K13 test checks caller mutation independence, unchanged original parameter bytes, equal q/output control and pre-geometry rejection at the actual binding check; its valid default control reaches geometry/init.

## What the named tests actually establish

| Test group | Meaningful present coverage | Remaining limit |
|---|---|---|
| Inventory/candidate | 21 seed0 profiles ×4 placements =84 candidates,144 stage geometry/init pairs; secure q equality, NI conjugation/SSE/separate bundle; broad registered denominators. | MissingCaptured is checked through metadata status, rather than a successful missing-seed candidate row. Normalized/full role identity assertions are mostly source inspection, not a full mutation oracle. |
| Canonical quick/revalidation | Actual production bundled Iris materialization,2400 full updates/rates,2 consumed updates, full case scope, terminalNone, actual validator positive and two negatives. | Self-hash/count checks do not establish the entire requested mutation matrix or late-finalization failure handling. |
| Scope/projection | Quick/archival selection remains rejected; malformed dimensions and mismatched seed retain a candidate issue. | Unsupported role/packing/security/backend combinations are not exhaustively exercised by this test. |
| Schedule/state/failures | Four updates across all4 modes for Square/SGD and Poly/Identity/NAG, exact terminal/fullness sequence, carried velocity, first-reference/decrease/restart branch flags; deliberate third-update RateConflict. | Most pre/post weight/bias/compact continuity and exact coefficient values are inspected in source, not asserted independently. Actual scale-failure trace reachability is not established by the manufactured helper error. |
| Block old-hidden/targets | e3 across all4 modes,3 updates, exact handoff-to-destination x/count wiring, I refresh, WN/AU conversion/bootstrap and AA canonical reuse/no added conversion. The block schedule begins below its later maximum. | All synthetic layer dimensions are2, making stage target labels equal; equality to the destination batch alone cannot detect a regression using source-stage y. A heterogeneous destination-label fixture is needed for that discriminator. Source currently uses the correct destination y. |
| Digest/accounting | Independent primitive bytes/SHA, actual combined variable-payload cap edge, real checkpoint rollback, retained supplied failure trace/attempts. | R3; no complete identity mutation matrix. |
| `p3_pre_key_fail_closed` | Partition/rate/reference/mini-secure failures and index cap through composition. | The unit function does not itself call validate_plan_binding or a factory. Actual validator coverage is in canonical integration; real factory ordering remains pending. The name must not be used as evidence of the original spy acceptance. |
| Unproved/key roles | Static required-gate membership, with actual key-role positives partly in inventory. | Static enum membership alone proves no runtime gate, allocation ordering, precision, security or resource feasibility. |

## Runtime evidence and scope verified without rerunning

Raw14 selects exactly five current-source unit tests and exits0; raw15 selects exactly four current-source integration tests and exits0. Both record all four exact source pins before and after, unchanged. This resolves the prior successful-callsite coverage gap for these selected tests. It does not establish R1/R2/R3 discriminators absent from their assertions. Targeted bench Clippy12 and format13 pin the frozen bench bytes and pass; CKKS Clippy11 passes the exact getter bytes (its printed bench hash is earlier and does not validate that later bench source). Getter07 and prefix09 remain specifically filtered evidence. Prefix08 is a real expected rollback RED before the corrected helper PASS.

Original03/05 remain evidence at their recorded hashes; original source bodies are unavailable. They are not treated as blanket current-source runtime evidence. No broad P1/P2 suite/HE rerun is visible in the supplied raw command records. The review did not execute any command in them.

The source/log arithmetic reconciles the reported actual cumulative totals: params12, proof6, metadata8, Iris2, geometry/init336 each, updates148, handoffs48, wrappers6, HE0. Per frozen14: unit mini2+tiny1 params, proof1/metadata1, geometry22/update72/handoff24. Per frozen15: shared secure2+dedicated getter1 params, proof2/metadata3, geometry146/update2/wrappers3/Iris1. Filtered09 contributes one tiny parameter and no planning. Counts are operational executed work, not profile admission, resident bytes or campaign results.

## P4/D7 and remaining acceptance

The typed initial/update/handoff labels, full rates/reference, partitions, branches and counters are useful P4 inputs. DiagonalPlan does not lend StageGeometry or typed full trace/attempt/event views; only its candidate sibling lends geometry. Complete geometry/role/trace information is present in canonical bytes, but writing another independent general decoder in P4 would enlarge the codec surface. Rebuilding geometry from the same trusted materialized input and revalidated request is possible; typed trace access still needs an explicitly assigned comparison/view strategy. This is a concrete future consumer limitation and a P4 integration acceptance gap, not evidence that the standalone source has already supplied a complete CPU bridge. No new accessor work is authorized by this review.

The first real P4/D7 consumer must establish genuine other gates/start, call binding revalidation before the actual key factory, and run a reachable nonallocating positive spy control plus corrupted identity/parameter/partition negatives. The original spy acceptance is not waived. No public fake Ready/start/callback capability should be added to make it pass.

Full-profile planning/storage feasibility, total/transient/RSS memory, physical key bundles/workspace/liveness, encrypted ranges, bootstrap internal feasibility/precision, security/sampled-secret compliance, runtime target locks, D7 descriptor/capture/start wiring and full campaigns remain unproved or separately gated. The64MiB operational convenience limit is not a full-Fashion fit result. Encrypted-block holds and broader registered denominators remain unchanged.

## Independent verdict

**SPEC: changes required** for R1/R2/R3, and explicit R4 disposition. **QUALITY: changes required** for the reachable failure-path defects, coverage gaps and Q1 documentation. Security/authority boundary: no new safe public forging or HE execution path found. Performance/resource claims: appropriately limited to metadata/payload accounting; full planning feasibility unmeasured. Maintainability: direct reuse of P1/P2 is sound, but all error phases need one consistent prefix/context policy.

Root can assign the smallest corrections and finite affected tests after reviewing these findings. No source change, extra runtime, Git integration or further campaign work is implied by this report. Even after corrections, standalone bounded source acceptance must remain distinct from full P3/P4/D7/Task54/scientific completion.
