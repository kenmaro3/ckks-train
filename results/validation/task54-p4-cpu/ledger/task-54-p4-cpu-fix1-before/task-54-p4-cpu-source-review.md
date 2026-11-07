# Task54 P4 CPU source review

Review date: 2026-11-07. Independent SPEC and QUALITY review of the frozen source-only package. **SPEC: NEEDS_FIXES (two important test-evidence gaps). QUALITY: PASS_WITH_MINOR_CONCERNS.** The implementation paths inspected are coherent; the important findings concern whether the proposed acceptance test independently establishes the required guarantees. No numerical acceptance or runtime release is given.

The reviewer read the CPU review brief, source release, original Task54 brief, final consumer map and earlier API/coverage map, the complete assigned diff/report, relevant binding design sections, and the necessary unchanged planner, trainer, optimizer/probe, bootstrap and metadata APIs. Only reads/hash calculations and this review-file write were performed. No Cargo, tests, HE/Engine/planner/parameter/key constructors, dataset work, network, Git mutation, source edit or subagent was invoked.

## Frozen evidence

Verified by reading and hashing, not by trusting the report:

- Report: SHA256 `ee518e6f04b597cd97036989d0e37931d2cd10d379885c642683cf6d09e8c447`.
- Full review diff: 123,972 bytes, SHA256 `5f48378c6f72d053b4b5cbf26be78b420c6a702e42570b3e207ebc9c2d0e1967`.
- Nine-path pins JSON: SHA256 `90a951724c0692affefe70fa76c4d5728980a88449b4ccd7d06bb40dd5e5cdaf`.
- All nine current files match those pins, including the eight owned sources and root's design clarification (`docs/study-diagonal-admission-design.md`, SHA256 `22d22f7ca4ba1f9d1375f4982291460555fd0f3fc29275c5cebdd1eeec6141f9`).
- The eight final source hashes equal both before/after inventories in recorded final Clippy and focused no-run evidence. The recorded raw Clippy output is clean, exit0; final focused no-run is clean, exit0, and names only a compiled lib-test executable. The owned-file rustfmt record is exit0 with `skip_children=true`; recorded owned diff-check is exit0.
- Compile RED raw output genuinely reports missing `CpuDiagonalSession`, `CpuCapability` and `run_contract_matrix`; it is compile RED only. It does not establish runtime RED or that the final acceptance assertions passed.

## Important findings requiring correction before acceptance/runtime release

### S1. Seven parameter negatives do not exercise the actual-parameter-to-bound-plan rejection

**Locations:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:119-150`; `crates/ckks-train-bench/src/study_admission/p4_fixtures.rs:142-145` and `159-203`.

Every call inside the `rejected` closure passes the unchanged `PlanningParameters p`. `bad_parameter_claim` encodes altered q/Delta/dnum/main-secret/p/N values, or flips a descriptor byte, into **only `SyntheticClaim.parameters`**. The real parameters and descriptor supplied to `prepare_then_make` never change. Thus all seven negatives exercise `claim.parameters != actual.bytes`, whereas `actual.bytes != self.parameters` remains untested. Removing that latter check would leave these seven tests passing. The ordinary source check is present and correctly uses the existing actual parameter codec; the deficiency is the promised independent rejection evidence, not a demonstrated production binding bypass.

**Required correction:** make the finite negative group distinguish actual parameter-description mismatch from malformed claim bytes and fail if actual-versus-bound validation is removed. Retain the shared `prepare_then_make` factory invocation, exact errors, zero negative factory calls, the two real context/parameter setups, and at most16 isolated negatives. Use a narrowly reviewed fixture/borrowed-codec route or already-owned contexts; do not allocate seven more cryptographic parameter objects, expose a production digest/parameter override, or claim canonical D7 coverage. Update the report's description so changed claim fields are not described as actual supplied parameter/descriptor mutation evidence.

### S2. Prework tests do not establish configuration/velocity preservation at a changing-rate boundary

**Locations:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:241-250`, `312-316`, `567`, and `661-664`.

The malformed-packet group runs only at update0. For every fixture, the optimizer is already configured at that first update's rate, and velocity is still absent. Its assertions compare counters, usable/progress flags and the aggregate before/after decrypted W/b snapshot; they do not compare raw optimizer current/reference/momentum/factors or encrypted velocity/compact state. Replays after successes check only error/counters/usable. Consequently, moving `set_lr(current_update_rate)` ahead of packet validation could mutate a scheduled session's rate on an invalid packet while these tests still pass: update0 is a same-rate operation, and subsequent replay checks do not inspect configuration. A reset of velocity before a prework reject would likewise not be detected by the update0 group.

The current implementation itself has the desired ordering: `validate_packet` checks pre_state and packet metadata before returning, and `step` calls setters only after validation and poison marking. Source inspection confirms that ordering; the test evidence is weaker than the explicit prework no-mutation requirement.

**Required correction:** reuse an already-planned scheduled I and NI session at a later update where old/current rates differ and velocity is present. Reject a malformed packet before work, compare actual raw configuration and all encrypted optimizer/compact state before/after, retain unchanged model/progress/counters and usable state, then perform the existing valid update. Do not add trainer attempts, plans, oracles, contexts or boots. Direct borrowed optimizer/state checks are available; no recurrence/formula or NI accessor change is required. The same assertion helper can strengthen replay checks.

## Minor quality findings

### Q1. Poison tests omit assertions for the diagnostic payload they promise to preserve

**Location:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:619-639`.

The tests check phase, nonzero exact partial counter delta, one successful real refresh/bootstrap, unchanged successful progress and rejected retry. They never inspect `FailedUpdate.update` or `FailedUpdate.trace`, so always-empty/stale trace retention would not be detected. The bridge does capture a changed runtime trace and intentionally omits unchanged prior diagnostics (`study_diagonal.rs:765-777`); that policy is reasonable and does not claim full P3 trace equality.

**Recommended correction:** in these same two failure attempts, assert update0 and a nonempty trace equal to the attempted trainer's exposed current trace (or a precise documented reason for absence). No additional runtime case is needed.

### Q2. Conditional foreign-basis mutation silently omits NI basis coverage

**Location:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:303-310`.

The NI update input level is the bootstrap start level, above the other I context's max9, so its foreign-basis branch is skipped. I sessions do exercise the shared `check_ct` basis rejection, and the source checks every I/NI x/y/W/b/velocity basis, so this is not an unimplemented basis guard. The report should distinguish shared guard coverage from per-mode negative execution. If per-mode basis rejection is intended, reuse foreign zero polynomials at a common supported level and ensure the assertion reaches the basis check rather than failing level first; keep the constructor/attempt caps unchanged.

## SPEC inspection results beyond the findings

### Canonical/synthetic construction and ownership

- `study_client::prepare_then_make` has a canonical immutable-materialization branch delegating to `validate_plan_binding`, followed by exactly one `describe_candidate` call. It uses trusted metadata, rejects candidate issues, compares the complete logical training/bootstrap manifest, and clones only one StageGeometry. Candidate placeholder initial states/rates are not used. Request budgets are not widened after hashing.
- `PlanOwner::check_shape` and candidate geometry cardinality reject block/multi-stage/boundary execution. Canonical and private cfg(test) synthetic owners remain distinct. There is no public canonical constructor, arbitrary digest constructor, callback authority, fake Ready/start capability or new successful dispatcher.
- The private real canonical continuation calls the same `prepare_then_make` invocation statement as synthetic setup/spy. Its `make_owned` checks actual parameter object, exact bootstrap association and supplied rotation coverage before `KeyGenerator::new`, the first key/secret action. Synthetic positive spy reaches that one shared invocation and returns only a terminal string. S1 limits the negative evidence.
- Public `from_key_generator` explicitly consumes caller-created keys and cannot prove rejection before their external allocation. This limitation is documented accurately. The private owned canonical continuation is compiled but unexecuted and remains disconnected pending genuine D7 owned gates/durable-start integration. Existing `run_owned` is outside the diff and remains a missing execution dependency.
- `StudyClient` owns Encryptor/Decryptor/RNG and explicit plaintext diagnostics. The caller owns immutable public evaluation context; sessions borrow it, and I borrows only a narrow refresh handle to the client-owned service. NI receives no refresh handle. Fresh `Arc<()>` setup identity is propagated through context/client/packets and checked by pointer association; it is local association evidence, not authentication or process isolation. The wrong-owner negative is correctly labelled a private token mutation, not a third-key experiment.
- Canonical update encryption uses real `update_rows`, numeric arrays, actual labels and existing `padded_targets`, then slices by bound global offsets. It does not restart rows/labels at micro boundaries, refold rates or reconstruct geometry per update. Packets are immutable externally, contain no caller-selected rate, and expose no untrusted deserialization path.

### State, coefficients, trace, counts and failure policy

- Initial packet validation checks association, plan, geometry/config/keys/boot binding, complete expanded W/b shapes/BSGS/diagonal lengths/stride and every ciphertext's exact level/scale bits/two-component NTT basis. Initial BatchSpec and compact/velocity lengths/presence/entries are checked. Session construction additionally verifies actual supported optimizer state and actual BatchSpec.
- Entering state is checked against pre_state before setters. The common checker covers every expanded W/b entry, every compact weight and Option velocity entry, complete cardinalities, actual raw current/reference/momentum and interactive velocity geometry. Post_state uses the same checker with an explicit successful index before progress publication. No zip-based length omission was found in those checks.
- The only NN source amendment is the borrowed read-only `OptimizerView`, default-unsupported `diagnostic_view`, existing Sgd/Nag implementations and checked forwarding method. No optimizer recurrence/math or NI source was changed. Borrowed velocity cannot outlive Trainer; unsupported is not treated as absent SGD state.
- Rates and lookahead come from the bound plan. Current/bias/reference helper factors and momentum are compared as exact bits; first/later velocity and reference branches are checked. The root NI clarification matches the code: reference-factor evidence is `NiOptimizer::with_lr(reference).grad_scale()` computed from configured state. Literal internal kappa/mask observation is not claimed; no duplicate kappa formula or new NI observer was introduced.
- `step` invokes exactly one existing I/NI `train_step_valid` per accepted attempt. All seven measured OpCounter fields are compared directly with inclusive prediction; bootstrap internals are not added again. Separate client refreshes, aggregate boots and NI state/activation bootstrap decisions are compared. Full actual trace is retained as diagnostics; hidden a1 level comparisons cover all I micros and NI micro0, and AA compares the complete available canonical encrypted activation subset/cardinalities/levels/scale bits. Full typed P3 trace equality remains unclaimed, consistently with the final map.
- Prework currently has no trainer setter/evaluation mutation path; progress overflow and rate validity are checked before work. Poison is set before the first fallible setter/train operation; progress is committed only after all postchecks. Midwork/postcheck failure preserves successful counters, actual partial delta and diagnostic phase, and rejects retries. Thin test guards permit one real refresh/bootstrap then fail before the second; neither HeOps nor training arithmetic is replaced. S2/Q1 concern independent assertions, not an observed violation of this implementation ordering.
- Initial/snapshot/probe paths fork evaluation counters, and checks assert training-counter isolation. NI BatchSpec helpers internally use existing ScaleProbe forks; no additional Encoder or bootstrap call was found in those probe paths.

### Matrix and oracle preservation

The single ignored lib test preserves all requested successful/failure sessions: six smoke families in both I and NI (12); Poly four-rate in I/WN/AU/AA (4); Poly first-reference in I/AU (2); PolyC3 poison in I/AU (2). That is20 sessions, I9/NI11, 32 successes plus2 failed attempts. NI smoke placement rotates WN/AU/AA for C3/C2/C10 and repeats for their Poly equivalents. Micros/rates/widths are the final-map values, including Poly smoke mean/kernel.04 with fixed reference.08, unequal8+4, partial5, full8, and .04/.08/.02/.08 scheduled transitions.

`SignalFamily` is closed over the existing eight recipes; `Family::engine(Signal)` remains the sole initial-array constructor, and actual existing Engine::train owns all F64 math. The baseline row/label/rate refactor preserves the original assembly and mutant behavior. One actual baseline trajectory is invoked per successful session:18 oracles/32 actual Engine updates, without the old sensitivity/mutant sweep. All logical W/b entries are compared with finite checks at fixed absolute I1e-5/NI4e-4. Pure frozen bias is asserted exactly zero; HE bias is compared at the ordinary fixed tolerance. Global logical-row assembly and zero padded targets are retained. Existing v1 failure/v2 pure sensitivity history remains separate, not fresh runtime evidence.

## Runtime guard and inventory audit

All values below are **source-proposed**, not observed execution totals. Guards occur immediately before the expensive allowed operation/wrapper invocation, and final assertions check actual helper reservation counts rather than summing a schedule formula. No cap mismatch or hidden additional constructor was found in the inspected reachable suite.

| Operation | Source inventory / guard placement |
|---|---|
| Plans / geometry / initial / updates / handoffs |20 plans; explicit charge before20 P2 geometry and20 initial calls; existing compose charges34 actual update planners,0 handoffs. Shared PlanningBudget caps28/28/42/0 and wrapper0. Spy reuses an existing NI plan. |
| Sessions |20 local cases, I9/NI11; session reservation cap20 before each case preparation. No global crypto fixture or second integration test binary. |
| F64 |18 guarded oracle creations and32 guards immediately before actual Engine::train. Failure sessions reuse initial/rows, without oracle calls. |
| Trainer calls |Attempt guard immediately before dispatch to train_step_valid; cap/final assertion34. Successful counter final assertion32; exactly2 intended failure branches. Rejected packets do not reserve attempts. |
| Parameters / KeyGenerator |Two explicit parameter guards before the only two constructors; two guarded private owned KeyGenerator calls. |
| Proof / descriptor / bootstrapper |One guard each before DftSupportProof verification, BootstrapMetadata construction, and real Bootstrapper constructor. No hidden Encoder in the bootstrap constructor. |
| Encoder |Two guarded setup Encoder calls; one guard before each of20 trainer constructors, whose unchanged implementations each construct one Encoder. Client/service/context use clones. Cap/final assertion22. No ActCost::probe/predict_update invocation adds encoders in this path. |
| Actual bootstrap |Thin forwarding wrapper reserves immediately before real.bootstrap, cap85. Injected second failure is rejected before reservation/call. No initial/probe/warmup bootstrap found. |
| Factory spy |One guarded terminal positive;16 counted isolated negative preparations each assert zero local factory calls. S1: actual-parameter mismatch is not established by its seven claim mutations. |
| Other negatives |Existing packets/context forks/initial packets are reused. No extra trainer/planner/key construction. Wrong-owner is token-only;36 invalid client label/cardinality requests stop before packet encryption. Q2 records omitted NI foreign-basis branch. |

The test currently uses68 explicit client snapshot endpoints (32 paired comparisons plus36 prework before/after endpoints),52 actual BatchSpec checks (20 initial+32 successful posts),20 initial model encryptions and34 logical update packets. These counts follow the inspected call paths; S2 can strengthen direct state/config assertions without adding expensive work. Source-only compilation does not establish any of these runtime assertions passed.

## Disposition and limits

Correct S1/S2, consider Q1/Q2, then re-freeze changed source/report/diff, repeat only the authorized compile/static checks, and independently review the corrections before root considers a finite runtime release. Do not loosen I1e-5/NI4e-4, raise85 boots, add parameter/key/planner/trainer calls, or drop matrix cases to address these findings.

This review accepts the ownership/architecture and the explicit NI helper-versus-internal-mask evidence boundary. It does not accept the two missing independent test guarantees as already proven. It cannot assert numerical parity, actual counter equality, actual cap completion, canonical pre-key acceptance, D7/production admission, P4/Task54 completion, scientific range/precision/security/resource evidence, GPU execution or full/quick campaign completion.
