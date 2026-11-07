# Task54 P5: actual CPU block training contract

**PASS for the bounded synthetic CPU implementation contract.** This archive preserves143 exact ledger files, plus this README and SHA256SUMS. It is not a scientific benchmark, dataset-accuracy result, secure-profile admission or complete Task54 runner. The first P5 HE invocation passed; no numerical retry occurred.

## Implementation and actual results

The client owns secrets and encrypts initial models, first inputs and stage-specific targets. A separate block packet/session uses the existing BlockMlp/BlockTrainer/BlockNiTrainer, with old-hidden encrypted handoffs, per-stage checks, exact AA reuse, inclusive counter checks and permanent poison after partial failure. The source and its complete16-session matrix were independently reviewed after tests-first compile RED, final selected no-run, strict Clippy and format checks.

| Actual bounded evidence | Result |
|---|---|
| Two/three-stage I/WN/AU/AA matrix |16sessions,42attempts:40successful updates and2intentional failed prefixes |
| Planning |36geometry,36initial,100stage-update,58handoff,158typed transitions |
| Real helper calls |313bootstrap calls (cap361),357client refresh calls (cap357) |
| I numerical endpoints |11; maximum absolute error8.56300169660428878e-9, cap1e-5 |
| NI numerical endpoints |29; maximum absolute error9.06713890636012820e-8, cap3e-5 |
| All-entry comparisons |4,888:2,304weights,784biases,1,800old-hidden entries |
| Successful handoff outputs |84: I22,NI62, including AA reuses |
| State/configuration checks |68unchanged comparisons;46full-state snapshots+2initial-model captures;3separate configuration snapshots/4comparisons |

Independent outcome review matched every expected comparison value to the actual paired F64 oracle bits, checked complete tensor-coordinate coverage, all40 endpoint maxima and TSV joins, and verified all40 measured seven-counter vectors against stage+handoff inclusive sums. Source-asserted metadata/configuration/AA/negative/poison checks completed. Underlying98stage calls remain source-derived; no separately measured stage windows are claimed.

The two poison tests retain actual partial work, committed source state and exact unchanged downstream payload, with global successful progress0 and retry refusal. I performed15refreshes with1retained boundary output; AU performed5boots with no published NI boundary output. Partial counters are measured self-consistency evidence, not an independent predicted partial-operation oracle. Trace equality alone does not prove execution; rollback is not claimed.

One paired phase actually used14oracles/14Engine objects/40train calls/128external forwards. Including both earlier pure runs, P5 F64 cumulative work is352train/136Engine/486external forwards. The [pure v1 failure and v2 acceptance](../task54-p5-preflight/README.md) remain distinct. The stricter NI3e-5 requirement was fixed before any P5 HE; it was not adjusted to fit this result.

## Shared-code regression and evidence lineage

P5 extracts shared P4 checks/factory code, so the unchanged single-stage contract was run once on the changed source before P5. It passed32updates+2intentional failures,71realboots,54state captures and268unchanged comparisons. Its32parity records equal the historical successful P4 output, with maximum errors I7.64926433305390674e-9/NI1.89993002996935445e-8 under the unchanged P4 caps1e-5/4e-4. This P4 cap is separate from P5's3e-5.

- [Source report with minor clarifications](ledger/task-54-p5-cpu-source-report.md), [source review](ledger/task-54-p5-cpu-source-review.md).
- [New P4 regression report](ledger/task-54-p5-p4-regression-runtime1-report.md), [outcome review](ledger/task-54-p5-p4-regression-runtime1-review.md).
- [P5 runtime report](ledger/task-54-p5-cpu-runtime1-report.md), [independent outcome review](ledger/task-54-p5-cpu-runtime1-review.md), [original-token parity TSV](ledger/task-54-p5-cpu-runtime1-validation/parity.tsv).

The original P5 report missed the first guard attached to the harness line. The corrected total is1,377P5_BEFORE_CALL records; all22per-kind totals were already correct. The original report is retained alongside the correction. Source/raw/numerical values/criteria did not change and nothing was rerun. The selected15-file source inventory covers the8owned files and selected dependencies; it is not exhaustive transitive-source or binary attestation. Historical release/report statements are phase records, not new authority.

P5 raw3,957,053bytes SHA256 `f01b6274bf391b9f1125a62113910cb351614eae50c83d9697b2b6b64b4ddaec`; new P4 regression raw228,070bytes SHA256 `d8d3d6b12c958774276abbc8c3a8d30ca6e19ef263bb1ea70cbf5475b9966c1a`. Both selected tests actually ran once and passed, with15selected source pins unchanged before/after. JSON here is non-scientific test custody; TSV is a read-only extraction. Mac test duration is not server/research performance.

## Limits and deferred notes

Snapshot export is used only immediately after a successful update. Before first success or after poison it may be incomplete/incoherent; explicit wider-API preconditions or rejection remain deferred. Negative-mutation temporary ciphertext copies are excluded diagnostic workspace, not all individually observed payload roles. The128retention receipts inventory quiescent coefficient payload and key material with separate analytical roles and unobserved workspace/caches/client storage/allocator overhead; they prove no peak,RSS or secure resource fit.

Canonical preparation exists, but genuine owned D7 gates/durable start/production runner, secure-profile resource/range/security/precision, GPU and full/quick registered scientific campaigns remain unverified. Full typed P3 trace and literal internal kappa/masks are unobserved; wrong-owner negatives use an association mutation, not a third-key experiment. Metadata-equal target substitution is outside the claimed negative checks. FullP3/P5/Task54/M4/M5 completion is not claimed. All21profiles/full2680+20+7/quick268 and full encrypted-block holds remain unchanged.
