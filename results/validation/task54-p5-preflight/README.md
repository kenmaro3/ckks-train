# Task54 P5: pure block-comparison preflight

This archive records a bounded synthetic F64 sensitivity test, not encrypted training or a scientific benchmark. All 128 files under `ledger/` are exact copies of the working ledger evidence. `SHA256SUMS` covers them and this README. JSON files here are operational/test custody or explicitly derived read-only analysis, not scientific result JSON.

## Outcome

| Run | Prospective HE error cap I / NI | Required separation | I witnesses | NI witnesses | Outcome |
|---|---|---|---|---|---|
| v1 | 1e-5 / 4e-4 | at least 8 times cap | 36/36 pass | 29/36 pass, 7 fail | FAIL, retained |
| v2 | 1e-5 / 3e-5 | at least 8 times cap | 36/36 pass | 36/36 pass | PASS for pure sensitivity only |

The v1 minimum absolute difference is 3.17824570814917440e-4. Before any P5 HE execution, the reason and alternatives for a stricter future NI error cap were recorded in [design section 14](../../../docs/study-diagonal-admission-design.md#14-p5感度検証v1の実測失敗とhe前の厳格化2026-11-07). V2 preserves the recipes, inputs, initial states, schedules, controls and required factor of eight. It requires future HE error to be about 13.33 times smaller. The original failure is not relabelled. V2 minimum normalized separation is 31.782457081491742 for I and 10.594152360497247 for NI. This does not show that HE achieves 3e-5.

Each actual run selected exactly one ignored test, `study_plain::p5_fixtures::p5_pure_block_sensitivity_gate`, and completed 156 Engine::train calls, 61 Engine constructions and 179 explicit external forwards. The v1 harness failed; v2 passed. Actual cumulative work, including the failure, is 312 train calls, 122 Engine constructions and 358 external forwards. Engine-internal work is not counted twice. The separate empty-phase test and future paired-oracle phase were not executed.

Four fixed families cover two/three stages, distinct stage classes/padding, local-head and post-update-hidden misuse, missing local updates, schedule/tail/row-restart controls and stage-target isolation. Each run emitted 40 controls (36 applicable, one equivalent, three inapplicable), 72 cap comparisons, 698 passing checks and 35 baseline range checks. The two cap comparisons of each witness share the same F64 difference; they are not independent backend measurements. Maximum baseline hidden magnitude is 6.52500000000000080e-1, below .75.

Independent review verified 30,836 emitted records byte-identical in order between v1 and v2, including candidate arrays, inputs, states, ranges and absolute differences. All 72 witness contexts/values/bits/differences are unchanged; all 36 I margin records are identical in full. Within test evidence, changes are limited to version, NI cap/normalized margin/status and failure/summary records. Full raw logs also contain naturally different compiler/harness/timing lines. No direct claim about unlogged gradients follows from this comparison.

## Evidence

- [Source and fix1 report](ledger/task-54-p5-pure-source-report.md), [initial review](ledger/task-54-p5-pure-source-review.md), [fix1 review](ledger/task-54-p5-pure-fix1-review.md).
- [V1 report](ledger/task-54-p5-pure-runtime1-report.md) and [independent outcome review](ledger/task-54-p5-pure-runtime1-review.md), with retained failed raw output.
- [Cap revision report](ledger/task-54-p5-pure-cap-revision-report.md) and [review](ledger/task-54-p5-pure-cap-revision-review.md). Only the test version and NI cap changed; static compilation, strict Clippy and format checks passed.
- [V2 report](ledger/task-54-p5-pure-runtime2-report.md) and [independent outcome review](ledger/task-54-p5-pure-runtime2-review.md), with actual launch, process, terminal and source-before/after records.

V1 complete raw output SHA256: `4702340394d79d98135401b3538a833ba6d0a1b8f70d7b738a3c195b12461b32` (5,677,785 bytes). V2 complete raw output SHA256: `ccd416c4fb342d6e2a0f75035d1dfd07aff9160ad1822a3ddb54a70e371a51be` (5,675,618 bytes). Final source pins are parent `fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a` and test fixture `cd580c14c8f6aa8099c113701890ec2231bb5b162f2375b63103c6562dcdb99e`.

## Not established

There was no P5 CKKS parameter/planner/proof/key/Encoder/evaluator/bootstrapper/ciphertext/HE execution. Actual block client/session/packet/retention behavior, HE error at 3e-5, canonical D7 admission and durable start, secure-profile resource/range/security/precision, GPU performance and full/quick registered scientific campaigns remain unverified. The future guarded paired-oracle phase is source only. This is not P5, Task54, M4 or M5 completion. Mac test duration is not server or research performance evidence.
