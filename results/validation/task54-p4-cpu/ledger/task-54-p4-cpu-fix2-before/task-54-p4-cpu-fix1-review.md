# Task54 P4 CPU source fix1 re-review

2026-11-07. **SPEC: NEEDS_FIXES for one new deterministic diagnostic-cap mismatch. QUALITY: PASS for the S1/S2/Q1/Q2 corrections, subject to that mismatch.** S1, S2, Q1 and Q2 are addressed at source level. The frozen ignored runtime test cannot complete its intended matrix with the new264 assertion cap. Numerical/runtime acceptance remains pending.

Scope: read the exact fix1 release, appended fix1 report, complete two-file delta, frozen pins and final static evidence, plus the unchanged trainer trace-publication points and Signal recipe fields needed to assess the corrections. No source edits, Cargo/test/HE/Engine/planner/constructor execution, network, Git mutations or subagents. Only this review artifact was written.

## Exact frozen package

Read/hash checks independently matched:

- Source report: `14e4829151b368aa398153048b6e903557ebd1bb1a1785ae00a0c7ea4537cb5c`.
- Fix1 delta:23,761 bytes, `a6c2357692ed2a63c61fe27e806b8acf6a67cc9699834265adb7d1fbf8a8d061`.
- All8 pins JSON: `fd86565cb6b32cae15aefd5fa25f46a08205c84d4633e1b6471b41e20d5a56ff`.
- Current all8 source hashes match those pins. Comparing with the original review pins confirms only the two assigned cfg(test) files changed and the other six sources are byte-identical.
- Recorded `04-clippy` and `05-compile-final` both exit0; both before/after8-hash inventories equal the reviewed pins. Raw outputs are clean. This is strict scoped Clippy and focused no-run evidence only; the test executable was not invoked.

## New important finding F1:264 unchanged assertions cannot cover the retained matrix

**Locations:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:386-392` (hard cap264), `512-515` (one reorder assertion for every two-micro first update), `619-633` (smoke/scheduled case assembly), `857-863` (all first-update groups), `865-869`, `897-902`, `924-931`, `945` (final264 assertion). The appended source-report diagnostic inventory contains the same undercount.

The report counts only4 two-micro reorder checks. There are **8**:

- SquareC10 and PolyC10 smoke in I and NI:4 first-update groups with `[8,4]`.
- PolyFourRate in I/WN/AU/AA:4 more first-update groups with `[8,4]`.

All18 successful sessions call `prework_negatives` at update0. The retained helper has12 unconditional rejects, plus reorder when two-micro, plus foreign-basis for8 successful I sessions. Therefore:

| Read-only work | Correct source count |
|---|---:|
| First-update unchanged assertions |18*12 +8 reorders +8 I basis =232 |
| Later changing-rate unchanged assertions |2 |
| Success replay/exhaustion unchanged assertions |32 |
| Poison retry unchanged assertions |2 |
| Total unchanged assertions |268 |
| Snapshot captures |54, unchanged |
| Configuration collection calls |54+268 =322 |

The guard stops at assertion265, before the complete intended matrix can finish; even removing its cap without correcting the final assertion would end in a264-versus268 failure. This is a source-derived deterministic breakage, not an observed runtime failure. Source compilation cannot expose it.

**Required correction:** root must explicitly correct the cheap diagnostic inventory/cap/final assertion from264 to268, first-update count228 to232, and configuration collection318 to322; then freeze/recheck the tiny correction before runtime. Retain all8 reorder checks and the full original matrix. No parameter/key/Encoder/plan/oracle/trainer/bootstrap cap needs to increase, and none should be changed. Do not reduce rejection coverage merely to fit264. This reviewer has not changed any cap or source.

## Original finding dispositions

### S1 — addressed

`p4_fixtures.rs::claim_for_actual_parameters` uses the existing `encode_params` on actual borrowed supplied PlanningParameters while retaining the original plan identity/binding. The shared spy now supplies the already-owned I parameters against the already-owned NI-bound plan, with claim parameter bytes matching actual I. `actual.bytes != self.parameters` is the only relevant failing discriminator; removing it would reach the same shared terminal factory, causing the expected-error/zero-call assertions to fail. The synthetic preparation still uses the same factory invocation as the real consuming seam.

The redundant second identity/binding negative was replaced, preserving16:1identity+1actual-parameter mismatch+7claim-codec mutations+7partition negatives. The seven codec mutations are now explicitly labelled claim-only evidence. No extra constructor/planner or production override was added, and no canonical/D7 acceptance is asserted.

### S2 — addressed, with F1 preventing successful suite completion until corrected

The new read-only helper captures raw current/reference/momentum and gradient-factor bits, W/b, all I velocity diagonals, all NI compact weights/Option velocity and complete layout/cardinalities, counters/progress/poison/failure/trace/BatchSpec. Exact ciphertext comparisons check component counts, levels, scale bits, basis identity, NTT flags and every limb, with lengths checked before zip. It introduces no HE operation, decryption, encoding, recurrence or accessor change.

The two later-rate rejects run at update1 of already-planned I and NI-AU PolyFourRate sessions. They assert completed1, all velocity entries present, and every old current rate differs from the upcoming.08; then remove the stage target, assert the exact prework error, compare complete state/configuration unchanged, restore the target, and continue the same valid update. This detects an early setter or velocity reset. Original first-update negatives remain; all32 existing replays/exhaustions and both poisoned retries now use the same complete-state comparison. No expensive operation is added.

### Q1 — addressed

Both poison attempts now assert failure update0. I requires nonempty failed diagnostics equal to the current exposed attempted trainer trace. Inspection confirms the unchanged I trainer publishes `self.trace` immediately before `refresh_all` (`trainer.rs:1390-1391`), so the injected second-refresh failure occurs after trace publication.

NI requires both current and failed trace to remain empty, with an explicit source-supported reason: unchanged NI publishes `self.trace` only at final commit (`trainer_ni.rs:2164`), after the injected second-bootstrap failure. This is the requested precise absence boundary, not claimed hidden partial NI trace evidence. Poisoned retries compare the retained failure payload unchanged. No attempt or observation API was added.

### Q2 — addressed by explicit limitation

The conditional foreign-basis negative remains unchanged. Its new comment and appended report explicitly state I-only execution (level9 fits the NI context), while NI level15 exceeds the other I max9 and skips that mutation. Both modes retain the same inspected production basis guard, and separate NI foreign-basis negative execution is unclaimed. This follows root's fix1 ruling and adds no fixture or constructor.

## Remaining scope and disposition

Apart from F1, no new important breakage was found in this two-test-file delta. The54 snapshot captures and68 existing client decrypt/inspect endpoints are consistent with the source; the new encrypted comparison copies have their own test-owned transient storage and do not establish memory admission. Expensive counts remain20plans/20geometry/20initial/34update/0handoff,20sessions I9/NI11,18oracles/32Engine updates,34trainer attempts32intended successes+2failures,2parameters/2keys/1proof/1descriptor/1bootstrapper/22Encoders,<=85actual bootstraps and1positive+16negative factory preparations. Tolerances remain I1e-5/NI4e-4.

Fix/re-freeze F1 before root considers runtime. This review gives no runtime release and asserts no numerical parity, actual cap completion, canonical pre-key/D7/production acceptance, P4/Task54 completion or scientific campaign evidence.
