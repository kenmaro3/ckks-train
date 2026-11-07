# P2 independent SPEC / QUALITY review

Reviewed 2026-11-07 against baseline `6a9210d` and the complete frozen nine-source fix2 patch, not just the final hook insertion.

**SPEC: CHANGES REQUIRED. QUALITY: CHANGES REQUIRED.** The retained miniature runs pass, and no change to the existing training arithmetic or bootstrap policy was found. P2 acceptance is nevertheless blocked by the concrete receipt/observation defects and required coverage gaps below. A passing test harness does not establish assertions that the harness does not make.

## Review scope and method

Read the ratified `docs/study-diagonal-admission-design.md` P2 requirements, validation brief, root disposition and API map first. Inspected the complete planner, pure integration tests, runtime fixtures, all changed runtime/helper hunks, and the relevant unchanged optimizer/dense/activation/compact/block/encoder contracts. Read initial readiness, v2 diagnostics, fix1/fix2 readiness, approved helper witness disposition, retained attempt1/attempt2 reports, final implementation report and manifests. Independently hashed the current nine sources and final evidence manifest entries and counted raw stage/helper/bootstrap reservations and returns. Used the engineering code-review skill for correctness, coverage and maintainability review.

No Cargo invocation, test execution, HE, network/server action, Git mutation, source fix or subagent delegation was performed by this reviewer. This document is the only file written. Findings below are static source findings; hypothetical negative test inputs described below were not executed.

## Blocking findings

### F1 — Observation equality does not establish ciphertext cardinality or identity; an existing wrong position and wrong receipt multiplicity already pass (high)

Locations: `crates/ckks-train-nn/src/optim.rs:414–439`; `src/diagonal_plan_tests.rs:400–547`, `:832–840`, `:856–861`, `:1266–1270`; `src/diagonal_plan.rs:1470–1483` (all under `crates/ckks-train-nn`).

Requirement: validation brief §1 requires direct role/micro/layer/container metadata and runtime coefficient identity; §4 requires exact equality for every observed field and every exposed final W/bias/compact/U field. The independent-review brief explicitly requires two-way/cardinality/position/container/state checks. API map §2 permits one uniform-diagonal receipt only with its real multiplicity.

Actual defects:

- In later interactive NAG, the loop records `mu_u.level()` and `k_u.level()` as the **position**, not the diagonal index (`optim.rs:430,438`). Every diagonal at the same level therefore receives the same wrong position. The existing later-NAG runs passed because `compare_observed` ignores `position` for every non-container role.
- Reverse matching only requires `!actual.is_empty()` and checks level/scale bits. It never requires `actual.len() == expected.multiplicity`, a unique position set or valid position range. Dropping all but one of a multi-diagonal gradient/transpose/velocity vector, duplicating the surviving position, or recording an out-of-range position is accepted by this comparator as long as the remaining label matches.
- The interactive planner itself reports `accumulated_gradient` and `accumulated_reference` through `note`, whose multiplicity is 1, although these are `shapes[i].m`-entry diagonal vectors. For the frozen 2→4→4 fixture the real multiplicities are 2 and 4. The current matcher hides this public receipt error.
- `container*_updated`, `container*_velocity`, and `bias_updated` are absent from the reverse-required set. The forward mapping checks a container position **if a record exists**, but deleting all such pre-refresh records passes the comparison. Post-bootstrap state equality does not replace the required observation of the pre-refresh state.
- Coefficients have only a non-None layer assertion, exact bits, and aggregate current/bias/reference counts. Two identical layer-0 coefficient records replacing layer 0 plus layer 1 still pass; no layer range or uniqueness is checked. `mu`/`kappa` have no required-presence or cardinality check at all. Expected occurrence counts must follow actual branches: I computes these coefficients per layer even on first NAG; NI computes them per weight container only when old U exists. Do not impose a fabricated common branch rule.
- Retry comparison is only actual→planned `.find` by role/micro/layer/source-level. Removing or duplicating stage retry observations is accepted. Require ordered, two-way equality for the **observed retry roles** (`bias_gradient` for I, `bias_extracted` for NI), including attempt count, source/target bits and error. Do not require observations for unrelated planner scale attempts for which no runtime hook was authorized. The direct helper tests already check their own two attempts; that does not repair this stage-level gap.
- Final NI compact W/U and final expanded layers/diagonals are traversed actual→expected without checking lengths first. Extra actual entries often panic, but missing actual entries can go unchecked. The boundary handoff loop likewise lacks outer/inner cardinality checks. Existing hidden/AA cardinality assertions are good and should remain.

Expected correction: preserve real values and required roles, record actual diagonal/container identities, make uniform multiplicities accurate, and compare exact expected/actual key sets, lengths and positions in both directions. Add bounded negative comparator checks showing that a missing/duplicate/wrong-position observation, missing coefficient layer, missing retry and missing pre-refresh container record are rejected. These are observation/receipt corrections, not a reason to change arithmetic, tolerances, fixtures or rerun a broad HE campaign. Existing readiness/attempt reports' claims of full cardinality/position checks exceed the pinned source; preserve historical reports and qualify that claim in the current disposition.

### F2 — Required original micro input/target roles disappear before observation and from the receipt (medium)

Locations: `src/diagonal_plan.rs:1144–1157`, `:1186–1195`; `src/trainer_ni.rs:1738–1754`, `:1805–1816`; `src/trainer.rs:1012–1025`, `:1036–1045`.

Requirement: validation brief §1 explicitly calls for x/y entry metadata after validation and for NI original/aligned y. API map §2 requires target **before/after** alignment and exact input/output metadata. These are already authorized test-only hooks and receipt fields, not P3/P4 expansion.

Actual: planner `forward` records only the input after the NI minimum-level switch. Planner `loss` records only `aligned_target`; its exact-target attempt sees y only after it has been switched to `out.level + 1`. The original `UpdateSpec` is not retained by `PlannedStageUpdate`. NI runtime hooks similarly observe only aligned input/y. The I hook gives y an `aligned_target` role and there is no distinct entry role.

Concrete consequence: NI inputs at levels 15 and 16 with the same scale are both recorded as level 15 when the layer W/bias are at 15. In the target-alignment branch, y at two different levels above `out.level` is switched to the same `out.level + 1`; neither the trace nor the recorder identifies its original level. The comparator can therefore certify the post-alignment result without proving the requested incoming-micro metadata identity. The current helpers verify the three alignment outcomes but do not provide this missing stage observation.

Expected correction: retain explicit raw entry x/y metadata per ordered micro in the receipt and observe those actual runtime ciphertexts at the already authorized boundary, then retain and compare the existing aligned roles separately. Keep this numeric/test-only and do not copy payloads or call the planner from a hook. Include both raw and aligned records in the exact completeness checks from F1.

### F3 — Two explicitly required pure branch witnesses are absent from the frozen coverage (medium)

Locations: `tests/diagonal_plan.rs` (complete 11-test file, especially scheduled cases at `:223–304` and first-reference/heterogeneous cases at `:481–585`); `src/diagonal_plan.rs:1972–2111`; `src/diagonal_plan_tests.rs:330–398`. Unchanged `tests/trainer_ni.rs:631–707` was also checked.

Requirement: validation brief §4 requires a **one-valid-tail** pure case; API map §5 item 3 explicitly requires **structural no-boot followed by exact next-scale failure**, with its operation/scale trace retained. The validation brief says no P2 obligation may disappear, and a rejected/unattempted required row is not completion.

Actual: the new pure and runtime schedules use full 8, partial 5, and unequal 8+4; no one-valid micro is present. The unchanged lazy regression uses 31/32, not one valid sample. No new pure test asserts the second required structural-then-exact-failure path. All NI schedule calls are unwrapped as success; the heterogeneous fixture fails during forward `dense bias exact product` on update 0, not after an unbooted structural decision. The old lazy regression's expected refusal is a structural `InvalidParams` before any work and does not invoke or compare the planner; it is not the requested exact-scale failure witness.

Expected correction: add the bounded pure cases and assert the exact denominator/ordered partial role for one-valid input, and the actual no-boot decision followed by the exact failure operation/prime/source/target/attempt trace for the latter. If the latter witness cannot be established in the frozen domain, retain that unmet obligation and obtain a specific root disposition for a concrete pure alternative; do not silently reinterpret the structural regression as it. No new HE search, main-fixture change or larger runtime matrix is requested by this finding.

## Positive source findings and resolved non-findings

- Geometry checks run before shape/compact constructors; logical C3/2/10 remains distinct from P4/2/16 and padding. Geometry binds the required explicit `EvalConfig`; carried state checks complete q/P/N/dnum/Delta/main-secret and geometry/bootstrap identity.
- The planner uses actual ordered primes and existing exact-target metadata helpers. Ordinary activation scaling remains separate, with the original F64 expression order and existing MSE relative 1e-12 rule. No tolerance enlargement was found.
- Current/reference NAG branching retains I `lr != reference` versus NI ratio semantics; first/later velocity work, frozen-bias work, compact bias-only U absence, ordered unequal micros, reduced input rejection and structural `must_boot` policy are represented. Failed exact bias retries retain attempt information. Counts include all seven fields with checked accumulation and bootstrap cost added once.
- Production arithmetic changes are narrow shared-expression extractions (`nag_kappa`, ordinary plaintext scale) plus visibility needed for pure existing structure. Observation is case-owned `Rc<RefCell>` under cfg(test), numeric only, with no planner invocation/payload copying. The global atomics are runtime budget guards, not global metadata observers. Real bootstrap wrappers delegate actual work and guard before invocation.
- Raw old hidden, AA canonical reuse, per-stage labels, mixed placements and separate boundary work are represented. A distinct handoff bootstrap descriptor is **not** a defect merely because it differs from the source stage descriptor: API map §3.4 permits a boundary contract, and unchanged `BlockNiTrainer::new` owns a separate boundary bootstrapper. Imposing descriptor equality here would change the supported runtime contract. AA correctly uses the source canonical receipt. The examined deeper-AA cached-input behavior also mirrors the existing runtime; it is not a confirmed finding.
- Unsupported custom/Poly activations are deliberately excluded from this enum; API map §2 assigns their capability outcome to the binding layer. Encoded data range, secret isolation, live-memory/resource bounds and secure numerical behavior are not supplied by P2 metadata and are not demanded as additional P2 work here.

Correctness: blocked by F1–F2. Test completeness: blocked by F1–F3. Maintainability: narrow extraction and ownership boundaries are sound; observation contracts must be made explicit. No additional security or performance finding was established in this scoped review; that is not a secure/runtime-performance admission.

## Verified runtime and historical evidence

The final author report `task-54-diagonal-plan-final-report.md` (9287 bytes, SHA256 `f1bbfe6d79163b9ea21d2fe49060bf061d1fb7c6be8e26c3ac929ddd87b51c5c`) correctly says test PASS / independent review pending, not P2 complete. Its final manifest SHA256 is `6bf65a61427f34c0fa06bdc93e3b408927e13e1afde3d4be697274a70d53ef16`; all 45 listed source/regression/evidence byte/hash pins matched on this review. The final report's then-open handoff/deeper-AA questions are resolved as non-findings above. Its warning that PASS does not prove exhaustive observation coverage is material.

| Evidence group | Verified retained result and actual scope |
|---|---|
| Initial pure RED | Five behavioral MissingCode failures before implementation; retained log inspected. |
| External pure GREEN | 11 PASS; all21 registered geometries use secure **metadata only**. This is geometry/initial metadata coverage, not secure HE or a full schedule for all21. |
| Current pure unit/static | 7 pure tests PASS; retained no-run/strict Clippy/format logs and hashes verified. No reviewer rerun. |
| Attempt1 | 9 stage reservations, 15 helper calls, 10 actual boot attempts/returns all Ok; 5 helper+4 pure tests passed, 8 runtime tests failed at the retained role collision. |
| Attempt2 | 27 stage reservations, 20 actual boots/returns all Ok; 3 I-only functions PASS, 5 functions FAIL on missing NI raw bias observation. The passing I-only 6+8+4=18 updates remain source-valid after the cfg(test)-only NI fix2 insertion. |
| Final attempt3 matrix | 5 PASS, 81 stage reservations, zero helper reruns, 220 actual boots and 220 Ok returns; 46 singleton models plus 5 bounded block cases. All eight runtime functions have passing retained results across attempts2/3, comprising 99 stage updates. These do not discharge F1–F3. |
| Original regressions | Singleton PASS once, lazy partial PASS once. Their fixed source/returned assertions imply 8 and 15 stage calls respectively and 44 lazy boots; the latter log has no per-boot raw ordinal stream. |

Final controller terminal is `2026-11-07T03:08:33.151921Z`, exit0, commands1/2/3 each0, none unstarted, nine before/after/terminal pins matching. The final matrix SHA256 is `d1026350fc6c5f08fda6dfea60f096ec8d0715f6a2d2c77dd994d40383a04fb9` (159844 bytes); singleton `d7f99b7974203283e2d1087093deda1916b8773d5af936b527977283f6b5257f` (443 bytes); lazy `95e3afdf8631c176dfbd595d31f45f13e9aab6aeebdd6cf70502a19f4ef6c6c8` (415 bytes). The per-row/boot summary is pinned at `5121438f49e76c1ac517158aa561e70039657968b322fa00bb7ceb735292c03b`.

Across the retained attempts there are 250 directly logged real boot attempts and Ok returns; the unchanged lazy regression adds 44 source/assertion-derived boots, giving combined 294 below the released cumulative bound340. Failed comparison attempts remain consumed work, not passing coverage. The original high-prime retry searches failed; the pre-code approved N4096 coarse helper-only alternative yielded frozen literals and four passing real helper calls, with no main-matrix parameter change or bootstrap in that context. Historical failures and numerical bounds are retained. Nested boot counters are already included in encompassing stage counters and are not added twice. No orchestration/test elapsed time is treated as training performance.

## Reviewed source pins

Whole patch: `task-54-p2-source-full-fix2.patch`, 196809 bytes, SHA256 `ab4941d8ab083544e66ff4094d692c11e0c6849c8d05a8c867dcafc063c587ed`. Root and author fix2 manifests agree on all nine files, and every current file matched its byte/hash pin.

| File under crates/ckks-train-nn | SHA256 |
|---|---|
| src/lib.rs | `abe8c9f1775208f91fabeeb60a459eb2468d1426349dd0903e8290fba2044ff8` |
| src/diagonal_plan.rs | `9f4cc40cc8aafe77faac0ac3c4e43aa521593d84f56da47b81028560c1074169` |
| src/diagonal_plan_tests.rs | `387b89b3deff04b97cc92f8a39ab60435812ca415e14f450c447cf07566aaf17` |
| src/trainer.rs | `55ba0dced5fe2bd6eb03cfbea817fed9ce6579e67f629959898e15d08953138f` |
| src/trainer_ni.rs | `41b9c6c9da6212466c46fcb2f035d704582912c179b1fea283539300c5cfebab` |
| src/optim.rs | `b58d14c959cec777fdc0b98ddd6e53bb148cf660daaac202e8a84bb6dc63d9b2` |
| src/activation.rs | `10c76cd71ea109ae46262d01fdf3ae30ce3717e9f22c74a8c13e2f8cc092c1e5` |
| src/dense.rs | `b7e919813b63412de759d42779c69f481a8f133c61c6d999473e49a2fb3a2fa2` |
| tests/diagonal_plan.rs | `40c570388ee45c3b9dd3a40c79804cb5f21eca89d0b6dc4302a807f712fdde29` |

P3/P4/P5, exporter context/capture, public release, secure campaigns, data/range/precision/resource admission and research performance remain separate future work. This review neither authorizes those tasks nor requests additional execution. Original author/root should address the bounded findings under the registered ownership/release process before P2 acceptance.
