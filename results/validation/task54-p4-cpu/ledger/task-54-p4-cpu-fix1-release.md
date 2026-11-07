# Task54 P4 source fix round1

Original implementer resumes. Root read full independent review; SPEC NEEDS_FIXES, quality minor concerns. Correct before any runtime. Original source/evidence frozen in task-54-p4-cpu-fix1-before; do not edit these originals. Full goal and original matrix remain unchanged.

## Open review findings (verbatim)

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


## Root decisions and exact scope

S1: Add one actual supplied-parameter mismatch using the already-owned interactive PlanningParameters against the existing NI plan. Its claim.parameters must be encoded from those actual supplied parameters using the existing codec, while the plan's original binding/identity remain unchanged; this isolates actual-vs-bound rejection and would reach the shared terminal spy if that check were removed. A narrow cfg(test) helper to produce this self-consistent actual-parameter claim is allowed inside the existing synthetic child, not a production override. Reuse both existing param objects, no constructor/planner additions. Replace the redundant second identity/binding negative with this case to retain exactly16 negatives:1identity+1actual-parameter mismatch+7claim-codec mutations+7partition. Keep every negative factory count0/exact error and positive1. Report seven claim-only mutations honestly; do not relabel them as actual field mutations or canonical evidence.

S2: In an already planned later step of the I and NI scheduled sessions, where old/current rates differ and velocity exists, reject a malformed packet before the same valid update. Compare actual raw optimizer config/factors/reference/momentum and every encrypted velocity/compact state before/after, in addition to model/progress/counters/usability. Prefer a read-only snapshot/assertion helper reused for prework/replay checks, with no decrypt endpoint additions if full encrypted comparisons suffice. No new math/NN accessor/plan/oracle/train/boot/context. Document exact resulting diagnostic counts; existing source-report inventory is prospective, not runtime authority. Do not weaken existing first-update negatives.

Q1: Add assertions to the same2poison attempts for failed update0 and nonempty attempted trace exactly equal to the current trainer's exposed trace (or report concrete reason if unavailable). No extra attempt.

Q2: Retain existing foreign-basis negative without adding a new parameter/basis fixture. Explicitly document that this negative executes only for I; NI uses the same inspected guard but no separate NI foreign-basis negative is claimed. This resolves report overbreadth; shared guard is implemented for both. No scientific mode coverage claim.

Owned source for fix only: crates/ckks-train-bench/src/study_diagonal/p4_tests.rs and crates/ckks-train-bench/src/study_admission/p4_fixtures.rs. Production source stays byte-identical. Report any actual need outside these two before editing. Append fix section to original source report, preserve original statements as historical source proposal, and save new raw/JSON evidence in task-54-p4-cpu-fix1-validation/. Source-only commands/owner/disk/environment remain exactly original release: focused no-run, strict scoped Clippy, owned rustfmt. No runtime, network, Git mutation, subagents, unrelated files or tolerance changes. Final no-run/static source pins must match. Produce frozen source-final.sha256 of all8 sources and before/after8pins on Cargo invocations. Covering test remains the single ignored p4_cpu_bridge_contract, not old suites. Root will independently re-review just this fix before runtime.
