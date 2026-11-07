# Task54 P5 pure runtime1 outcome review

2026-11-07. **SPEC outcome: FAIL for the current candidate at the frozen combined I/NI sensitivity gate. QUALITY/custody: PASS. EVIDENCE: PASS.** The actual failure is faithful and fully retained; it blocks P5 HE acceptance at I1e-5/NI4e-4 with>=8x separation. No source defect, rerun or alternative cap is selected by this outcome review.

Read runtime1 report/release/authorization/controller/source pins, immutable raw/terminal/margins TSV, and the previously reviewed source invocation/assertion boundaries. Independently parsed all events and numeric/value joins. Only read/hash/text analysis and arithmetic on retained values, plus this review-file write, were performed. No source changes, Engine/forward, Cargo/tests/HE, network, Git mutations or subagents. Runtime2 and HE remain unreleased.

## Custody and exact execution

Verified report SHA256 `98ba8a86ea01fab7dfca16e453ef31feaaf5dc0536a13a98629d765d9d3db30a`; raw5,677,785bytes SHA256 `4702340394d79d98135401b3538a833ba6d0a1b8f70d7b738a3c195b12461b32`; derived margins TSV18,309bytes SHA256 `7d9cd886b8e05f46bd10f67c273ace2740b1f5d4d280680bcfe0e6a8d5e6b182`; controller SHA256 `d949328fd4114dfdc9cddd7ca2b8a435798a0af3bf2b93ea3fbaa8bb7e1fa4cc`.

Authorization bindings to exact fix1 review, release, source report and two-source pins all match. Terminal binds that authorization, selected command/environment, disk check, source-before/after, PIDs/start/end, raw hash/length and101exit. Current source files still match both terminal inventories. The controller has one subprocess call, exclusive evidence-file creation and no automatic retry path. Root reports retained handle45658; local custody records controller94860/Cargo94861. This review did not query or rerun that tool handle.

The raw confirms exactly one actual selected ignored test: `study_plain::p5_fixtures::p5_pure_block_sensitivity_gate`,0passed/1failed/0ignored/192filtered. The separate ignored empty-phase test was not executed. The failed assertion is the final nonempty failure report after the full matrix/count checks, rather than an early aborted arithmetic/cap failure. Test-reported0.18s and controller timestamps are custody only, not performance measurements.

## Actual completed inventory

All396 BEFORE_CALL records were independently checked for contiguous ordinals, fixed caps and exact operation kinds:61Engine constructions,156actual train calls and179external forwards. No `oracle`/paired phase guard appears. Per-family counts agree with raw FAMILY_SUMMARY and reviewed actual call paths:

| Family | Train | Engine objects | External forwards |
|---|---:|---:|---:|
| A |42 |13 |42 |
| B |87 |26 |103 |
| C |11 |12 |12 |
| D |16 |10 |22 |
| Total |156 |61 |179 |

There are40control rows (36applicable/1equivalent/3inapplicable),72margin rows,698CHECK rows allPASS,35baseline RANGE rows,5354unique COMPARISON rows, and7FAILURE rows. Final SUMMARY reports those counts and FAIL. All four frozen initial/input/rate manifests precede the first Engine guard, including the leading MANIFEST sharing the harness line.

All22,235 raw VALUE tokens round-trip to their recorded F64 bits and are finite. All4368 emitted frozen bias/bias-velocity values have exact positive-zero bits. All5354comparison rows have consistent bit-identical flags and finite endpoints. Every unselected stage in target-isolation comparison rows is bit-identical. Each RANGE maximum was independently recomputed from that context's retained old_hidden VALUE records; the largest is6.52500000000000080e-1 (first occurrence A/baseline update0/stage0), below.75.

These are actual finite synthetic observations under the reviewed source. They are not secure-profile range/precision/security evidence. The completed pure matrix invokes no CKKS parameter/planner/proof/key/Encoder/evaluator/bootstrapper/ciphertext/HE or paired-oracle phase. No later phase is inferred from completing all pure work.

## Failure/value joins and outcome

Every one of72TSV rows matches all corresponding raw fields exactly, including context, backend, layer/tensor/entry, original values and bits, difference and bits, cap and bits, normalized margin, required8 and status. No rounding or relabelling was introduced by the derivative. Each raw witness joins the same COMPARISON entry and is the maximum W difference over the source-specified affected stage (or all stages for schedule controls). Raw difference bits equal absolute subtraction of retained endpoint values, normalized margin equals retained difference/cap, and status agrees with>=8. The7failure strings join exactly the7failed margin rows.

| Backend / frozen cap | Applicable rows | PASS | FAIL | Minimum difference | Minimum normalized margin |
|---|---:|---:|---:|---:|---:|
| I /1e-5 |36 |36 |0 |3.17824570814917440e-4 |31.782457081491742 |
| NI /4e-4 |36 |29 |7 |3.17824570814917440e-4 |0.7945614270372936 |

All seven failures occur at first affected update0:

| Context | Multiplicity | NI margin |
|---|---:|---:|
| A/B LocalHeadAsHidden boundary0 |2 |3.27412927631307715 |
| A/B PostUpdateHidden boundary0 |2 |0.794561427037293599 |
| C PostUpdateHidden boundary0 |1 |6.85726806028390889 |
| D LocalHeadAsHidden boundary0 |1 |6.69195162463276372 |
| D PostUpdateHidden boundary0 |1 |5.19349670258982510 |

Thus the weakest A/B updated-hidden mutants differ by less than the current NI4e-4 error allowance, as well as falling below the selected8x separation requirement. The target-isolation/finite/shape/frozen-zero/range/count checks passing does not rescue those seven discrimination failures. The combined candidate is not accepted for HE under the frozen criteria; the I component alone passing must not erase the retained NI cases or full scope.

## Arithmetic-only prospective NI3e-5 assessment

The immutable smallest observed absolute difference is approximately3.1782457081491744e-4. Keeping the same observations and required8x separation implies, for sensitivity arithmetic alone:

`prospective_cap <= minimum_difference / 8 ≈ 3.972807135186468e-5`.

At a prospective NI3e-5, the minimum normalized difference would be approximately10.59415236049725, and all36retained NI differences would exceed8times that prospective cap. This is division of already-recorded differences, not another trajectory, not new source execution and not a replacement PASS for runtime1. Minor last-digit decimal/binary rounding does not affect that inequality.

NI3e-5 would demand about13.33times smaller future HE error than4e-4. It therefore strengthens the HE accuracy requirement while reducing the sensitivity denominator; it is not a relaxed encrypted-error allowance. However, choosing it after seeing pure outcomes is still a new explicit criterion decision that must retain runtime1's original FAIL and be recorded/frozen before any HE result is examined. These F64 differences provide no evidence that P5 HE, its block handoffs, all placements or secure profiles can achieve3e-5. P4 tiny-fixture errors cannot establish that feasibility either.

The other option is a newly justified/frozen stronger signal or recipe, which changes baseline/control trajectories and requires new range/isolation/discrimination evidence. Neither that redesign nor the stricter-error option is selected by this review. No constants, caps, labels or current evidence were changed, and no retry is authorized.

## Disposition and limits

The runtime report faithfully states the failure, counts, immutable evidence and blocked HE boundary. The observed pure work is complete and must remain counted as156train/61Engine/179external forwards even if root later selects another phase. Any new source/criterion/run needs its own reasoned decision, freeze and cumulative accounting; the original7NI failures remain visible.

Root retains the next design/cap decision. Current P5 HE, runtime2, canonical/D7/resource/range/security/scientific performance, registered full/quick campaigns and full P5/Task54 completion remain unaccepted. No rerun or HE release is given here.
