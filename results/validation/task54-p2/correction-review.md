# P2 review-fix1 scoped independent review

2026-11-07. Scope: the frozen six-file correction delta addressing F1/F2 and the one-valid part of F3; the proposed single pure alternative for the remaining F3 witness. This supplements, and does not replace, `task-54-p2-independent-review.md`.

**Scoped source SPEC: PASS for F1, F2 and F3 one-valid correction. Scoped source QUALITY: PASS; no new blocking defect found in this delta.** These are source/pure-test conclusions. **P2 acceptance remains HOLD:** strengthened real-runtime comparisons have not run, and F3's structural-no-boot→exact-scale-failure witness remains UNMET. Previous runtime passes do not establish the new observation checks.

## Findings disposition

| Original finding | Scoped result | Source/evidence assessed |
|---|---|---|
| F1 position, multiplicity and two-way metadata checks | Corrected in source | `src/optim.rs:421–446` records the enumerated diagonal position. `src/diagonal_plan.rs:1473–1497` gives accumulated gradients/references their real diagonal multiplicity. `src/diagonal_plan_tests.rs:920–1105` expands uniform roles, checks container ownership, rejects duplicate keys on both sides and compares complete role/micro/layer/position→level/scale maps. Updated bias and indexed compact W/U roles are included. |
| F1 coefficient identity and retry completeness | Corrected in source | Coefficient keys include role/layer/position. Expected I mu/kappa are per layer, including first NAG; expected NI mu/kappa are per weight container only on later U branches, never bias-only U. Exact coefficient maps reject missing/duplicate/extra identities. Hooked retry roles compare complete ordered vectors including context, source/target bits and errors; unrelated unhooked planner attempts remain outside that assertion. |
| F1 final state/handoff cardinality | Corrected in source | `compare_expanded` (`src/diagonal_plan_tests.rs:317`) compares complete nested diagonal labels and the complete bias vector. NI compact W/U compare complete vectors including Option absence. Block stage counts, compact state and handoff outer/inner labels compare complete collections. Existing hidden/AA count checks remain. |
| F2 original x/y entries | Corrected in source | Planner entry roles at `src/diagonal_plan.rs:1447` and `:1605`; actual raw x/y cfg(test) hooks before switching at `src/trainer.rs:1017–1038` and `src/trainer_ni.rs:1733–1746`. `input` and `aligned_target` remain separate. The exact comparator requires both raw roles. Pure external test `p2_raw_entries_remain_distinct_when_alignment_merges_labels` retains levels15/16 while aligned labels coincide. |
| F3 one-valid tail | Corrected with pure coverage | `tests/diagonal_plan.rs:706–816`: [8,1] and [1], both activation pairs, I/WN/AU/AA, SGD/NAG. It independently asserts denominator/coefficient bits, ordered micro entry identities, and Identity partial extra level/ct-pt/rescale. This is the required pure branch coverage; no one-valid HE claim is made. |
| F3 structural no-boot then exact-scale failure | Still UNMET | One frozen scan exhausted672 initializations/2688 calls with zero errors/witnesses. Passing no-boot transitions and the prior structural refusal are not relabeled as this witness. The alternative is prospective only. |

NI mu/kappa receipt roles now include the container index. That is a justified correction: otherwise two weight containers belonging to the same layer collide under the strengthened key contract. Runtime observations retain actual container positions. The pure [32,32,32]/B8 case checks four weight-container velocity roles and coefficient positions; the existing small runtime matrix is unchanged. The external first/later test was updated to recognize these indexed NI roles while retaining its branch assertions.

The numeric recorder fixtures are explicitly comparator unit fixtures. Their construction from planned metadata is acceptable for testing whether missing/duplicated/wrong-key records are rejected, and they include independent multiplicity/branch expectations. They are not independent HE observations and must not be promoted to runtime parity. The actual `run_case`/block paths still feed the comparator recorder values captured directly from real ciphertexts and computed runtime coefficients; no planner values manufacture those actual records.

No tolerance, arithmetic expression order, optimizer branch, bootstrap decision, parameter, rate, seed, existing runtime row, cap or dependency changed in this six-file delta. Production runtime edits are the harmless enumeration/binding needed by cfg(test) observations; the planner edits change trace roles/multiplicities and zero-count entry records. No global/TLS observer, fake HE, payload copying or public optimizer-trait expansion was added. The earlier handoff-contract/deeper-AA non-findings remain unchanged.

## Evidence and pins

Read the complete50813-byte delta and correction report, current matching six source files and relevant comparator/runtime context, baseline snapshots, pure RED/GREEN/static logs, F3 outcome/discovery log, the alternative proposal, original API-map §5 item3, P1 metadata gate and unchanged structural policy. No test/Cargo/constructor/prime search/HE/network/SSH/Git mutation/source edit/subagent action was performed by this reviewer. This scoped review is the only newly written file.

All51 current manifest entries checked successfully:9 current sources,9 baseline snapshots and33 evidence items. The delta/report/manifest and proposal match the dispatched hashes:

- Correction report: `d62d18d8e09151d64cb4ef246bb354bb11499044f4499ecd61780d2de173a115`.
- Six-file delta: `80ae01776ac2589f20acbc2a01e3d2152e675795eaa79189c7d3b4021d50e8cf`.
- Source/evidence manifest: `dd3a539e4aab5d9bb3f1454ab638993b4a8ba663f31ae7dbbf11ef1cdc01a072`.
- F3 alternative proposal: `1ea11afbb6a705e28c95a130c666da49706c2e0a5a0f7fae44395e00daee0690`.

Retained RED2 is7PASS/6FAIL before correction. Final current unit pure tests are15PASS, external pure tests13PASS; no-run compile, strict Clippy and changed-file format pass. Development failures remain retained and were not treated as final success. Raw F3 counts independently read from the retained log are672 INIT,2688 CALL,2688 OK,808 state_booted=false and1880 true. These are metadata transitions, not encrypted updates or real bootstraps. The terminal discovery failure explicitly says F3_UNMET with zero errors in the candidate trajectories. This reviewer checked raw counts/terminal and pinned outcome, not independently reconstructed every frozen-domain tuple.

Current changed source SHA256 pins:

| File under crates/ckks-train-nn | SHA256 |
|---|---|
| src/diagonal_plan.rs | `5388a7860d76947dad3f86f01b2590de9e026d57bfe7509f86a06c642167508d` |
| src/diagonal_plan_tests.rs | `067d96f3ac1082b084f5f87f17ef2222da0e35c4be7b16a210bb6e5e88d766af` |
| src/trainer.rs | `69dc05446594afe019dba987d42474c30cce7843bc559905518c77c42efef975` |
| src/trainer_ni.rs | `8b218cccd0106e8a925c8cc3be8a328864e04dfba91f059a63b86021250a3dbe` |
| src/optim.rs | `7f99d28585c0656f0faa00b87d9808b4c52cda1ce0d99de877b39bcffe74bcfe` |
| tests/diagonal_plan.rs | `765704f38d2499ffe7280f588125413102691070b58afb0131ef46193d33d9ae` |

The three other owned sources remain pinned unchanged in that manifest. Old source/runtime reports remain historical evidence for their older checks, not an assertion that the strengthened comparator has passed real runtime.

## Independent assessment of the F3 pure alternative

**The minimum-plaintext-scale alternative is relevant to the original requirement, conditional on an actual witness.** API-map §5 item3 asks for structural no-boot followed by exact next-scale failure; it does not require the underlying error specifically to be `HekitError::ScaleMismatch`. That narrower predicate belonged to the exhausted discovery release. P1 `target_plain_scale_at` performs exact label search, derives the plaintext scale, then applies the existing minimum gate. Failure at that gate after a valid no-boot decision is a direct counterexample to structural sufficiency. It is distinct from both an exhausted ULP search and an early structural `InvalidParams` refusal.

The proposed mechanism is concrete. For source=target=Delta and a positive integer compute prime q<2^20, the finite power-of-two label Delta*q is a direct exact preimage. Its derived plaintext scale is exactly q, so the existing minimum test rejects it. The proposed full Identity/Identity WN path avoids nonlinear/partial derivative work and aims to reach compact update levels[10,11], structurally unpacked W[9,10], followed by the first `weight_unpacked` minimum-scale failure. Under its q/Delta range[1.5,2), preceding gradient/bias target scales have enough multiplicative factors to exceed the minimum. This is a source-level rationale, not a constructor, reachability, count or failure-result measurement.

The root prospective disposition `task-54-p2-f3-alternative-root-disposition.md`, SHA256 `d32fe4ac789bb4af3fe5ee0b2f369830aaa5fecc70d778e87b576df81a4e8756`, preserves the correct distinction and explicitly releases no execution. I independently agree with that relevance ruling. The closure/preimage discussion is appropriately qualified; it is not a proof that every valid schedule has an exact reachable label or that registered profiles are admitted.

One concrete omission in the proposal must be resolved before a later exact release: its candidate bullet list does not explicitly name the mode/placement. Freeze **`DiagonalMode::NonInteractive(BootPlacement::WhenNeeded)` exclusively**. The predicted compact levels and no-state-boot path assume WN; AU forces a state bootstrap and AA changes the forward path. Root acknowledged this clarification during review and will put WN in any later release. Do not infer permission to choose placement or tune another candidate.

A qualifying result must retain all of the following within the stated one-candidate/one-update ceiling:

1. Actual constructor/proof/bootstrap descriptor success and ordered primes satisfying the predeclared lower/upper gate; initial state only from `initial_stage_metadata` and its unchanged BatchMetadata.
2. Actual ordered trace reaches compact update under WN without an earlier error. Reconstruct the exact complete structural LevelState from that trace and call unchanged `must_boot` to establish false; preserve its inputs and known-full lookahead. No state-bootstrap event may occur.
3. Exact failing operation, level, prime, source/target bits and full attempts demonstrate `target_plain_scale_at` minimum rejection at the post-policy unpack boundary. Preserve `InvalidParams` as that actual underlying error and `ScaleInadmissible` if that is the planner wrapper; do not relabel it `ScaleMismatch`.
4. Any construction, admissibility gate, earlier operation or policy mismatch leaves F3 unmet. No fallback, parameter adjustment, injected state, adaptive search, HE, production preset change or wider campaign follows.

If those facts are observed, the result can satisfy the original non-error-specific structural-versus-exact-admissibility counterexample under the explicit root disposition. The old exhausted-ULP scan remains a zero-witness negative result and is never reclassified. Until then F3 remains open. Strengthened main runtime validation is independently still required, and P3/P4/P5 plus secure/data/range/precision/resource/scientific gates remain outside this scoped review.
