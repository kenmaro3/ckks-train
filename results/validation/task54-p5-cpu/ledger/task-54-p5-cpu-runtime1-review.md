# P5 actual CPU block runtime1 — independent outcome review

2026-11-07. **SPEC: PASS for the frozen bounded synthetic CPU contract. QUALITY: PASS within that scope, with the two previously deferred source notes unchanged. EVIDENCE: PASS against the corrected report. No blocking finding.** The original report had a one-record aggregate-count error; correction provenance is retained below. This review grants no new invocation or runtime release and does not establish full P5/Task54 acceptance.

Read the runtime report, release, authorization, controller, selected source pins, launch/process/terminal custody, complete raw record stream and parity TSV; the source review/diff, source-report minor clarification, relevant unconditional test assertions, accepted pure v1/v2 outcome reviews and P4 regression outcome review were also inspected. Independently checked custody joins, all operation ordinals, every P5 value and expected-oracle join, endpoint tensor coverage, typed inclusive counter sums, retention arithmetic and cumulative work.

Only file reads, hashes, text parsing and arithmetic on retained values, plus this review-file write, were performed. No test, Cargo, constructor, Engine/forward/planner/HE, network, Git mutation, source edit or subagent occurred.

## Custody and report correction

| Retained artifact | Verified SHA256 |
|---|---|
| Corrected runtime report | `ed98620c474f3cb1e7765bd561991b0bf62d8ba49e99357cb1aeb8275f66791d` |
| Original report, preserved as `task-54-p5-cpu-runtime1-report-before-count-clarification.md` | `df924719b383bb605b60e43beeae531df17d8a3a0924710d6ace818f2bbcc2e2` |
| Release | `1a810664b7754deaa893fc40e79d319a703072a5364acccf4870a8a0bb558ce1` |
| Authorization | `7c613c4f2ee5462a3f339c9cb8ab3733c1e1e321af5ff2c6fab5587d31546ac3` |
| Controller | `68f525f8a70800b76ef601663d1e65b64739e028dd67fb388924754eff989562` |
| Selected 15-source pin inventory | `e0a955eb6b123b0189907637a316a497b381b66079996615f060bce7693a0176` |
| Raw, 3,957,053 bytes | `f01b6274bf391b9f1125a62113910cb351614eae50c83d9697b2b6b64b4ddaec` |
| Parity TSV | `f0a9946bbae7e6600a669de675ad798b7f91d6b55655a965e85938aea3f4e5e8` |

Authorization binds the exact accepted source review `0c4cb590…`, clarified source report `0bb3fcc3…`, P4 regression review `bc9ff9af…`, release and pin inventory. Those hash joins match. All launch fields are an exact subset of terminal; process joins controller82739/Cargo82740. Terminal records test_exit0/controller_exit0, start1794042068.8310342/end1794042192.143295, unchanged selected source and the complete raw byte length/hash. All 15 current files match both terminal inventories. The eight owned implementation/test sources are the same sources independently reviewed earlier.

The controller uses a fresh evidence directory and exclusive files, one subprocess call, the existing target and no retry path. Its selected command is exactly `study_blocks::p5_tests::p5_cpu_block_bridge_contract` with `--ignored --exact --nocapture --test-threads=1`, incremental0/jobs8/Rayon8. Free disk60,205,805,568bytes exceeded805,306,368bytes. The raw confirms one selected test, **1passed/0failed/0ignored/193filtered**. Root reports original handle99633 retained to terminal0; that is root-observed tool custody, not a separate handle encoded in the raw. Harness122.94seconds is custody, not a performance measurement.

**Minor reporting discrepancy, corrected without execution:** the original report said1,376 P5_BEFORE_CALL records. The actual total is **1,377**. The first Params ordinal1 record shares the harness line and is omitted by counting only lines beginning with P5_. Independent matching anywhere in each raw line finds it. All 22 per-kind sequences/final totals were already consistent; no operation exceeded its cap. Root preserved the original report exactly and corrected only reporting text. The corrected report also properly calls the 15 pins a selected inventory, not an exhaustive transitive-source closure. Release, authorization, raw, TSV and source files remain unchanged.

## Complete actual matrix and call accounting

Session grouping matches the frozen source order: sessions1–4 are A I/WN/AU/AA,5–8 B I/WN/AU/AA,9–12 C I/WN/AU/AA,13–14 D I/AU,15–16 I/AU poison prefixes. A/B sessions each have four attempts/completions/endpoints; C each one; D each two; poison each one attempt and no successful endpoint. This establishes **16sessions /42attempts /40successes /2intentional failures**. The final unconditional assertions were reached before the harness PASS.

All16 actual plan records have correct cumulative increments for their stage/update counts. Final usage is36geometry/36initial/100update/58handoff/0canonical-wrapper, with158typed transitions. Raw contains100typed stage-update and58typed handoff records, including planned failure suffixes. The98 underlying stage calls remain source-derived; they are not separately measured stage windows.

Every P5 guard has a contiguous per-kind ordinal starting at1, its unchanged fixed cap, and no excess. All22 final records match those independently counted guards:

| Actual work | Observed total |
|---|---:|
| Parameters / keys / proof / descriptor / Bootstrapper |2 /2 /1 /1 /1 |
| Encoders / positive terminal factory spy / sessions |40 /1 /16 |
| Block wrapper attempts |42 |
| Real bootstrap helper calls |313, cap361 |
| Real client refresh calls |357, cap357 |
| Packets / first-input CT encryptions / stage-target CT encryptions |42 /64 /152 |
| Client endpoints / delegated stage model inspections / handoff inspections |40 /96 /84 |
| Full-session snapshots / unchanged comparisons |46 /68 |
| Configuration snapshots / configuration comparisons |3 /4 |
| Additional poison initial-model captures |2 |

The actual constructor/helper guard sites are those reviewed in the frozen source, including rejection of the injected refused boundary before a real-call reservation. The synthetic terminal spy has one positive and16pre-factory negatives; this is not canonical D7 owned-start evidence. The24block negatives,2later-rate negatives with real velocity,40replay/exhaustion checks and2poison retries complete the68unchanged comparisons. Configuration-only I late-stage rejection/same-rate checks and the existing uninitialized A/AU late-reference rejection/restore checks finish their3/4 budget. Reference-lock rejection after real NI velocity remains assertion-backed. These checks are supported by reached unconditional source assertions and actual diagnostic guards; they are not individually emitted new numerical model trajectories.

For **each of all40 successes**, the printed stage and boundary inclusive OpCounter vectors sum exactly once to the completion's measured seven-field vector, and their refresh counts sum to its client_refreshes. Every endpoint cumulative counter equals the sum of preceding completion deltas in that session. Successes total308actual boots/342refreshes; adding the poison prefixes gives313/357, exactly matching actual helper guards. The84successful handoffs are explicitly I22/NI62, including AA reused clones. Source postchecks for exact AA all-limb reuse/zero added boundary work, incoming NI transition level16 versus fresh15, configured branches, all-stage states, traces and metadata completed in this passing path. This does not observe literal internal masks/kappa or create per-stage measured counters.

## All-entry numerical evidence and TSV joins

All4,888 P5_VALUE records were independently parsed:2,304W,784bias and1,800old_hidden. Every actual/expected/error/cap is finite, absolute subtraction reproduces the logged error, every error is within the unchanged mode cap, and every expected frozen bias has positive-zero F64 bits.

Each P5 expected value was additionally joined **bit for bit** to the paired oracle's emitted F64 VALUE record for the same session/family/update/global layer or old-hidden boundary/global row/column. All4,888joins match. Numeric tokens in all12,660paired emitted VALUE records round-trip to their recorded F64 bits. This verifies that the encrypted comparisons use the retained real baseline outputs, including old-forward hidden arrays, rather than relying only on the endpoint summaries.

For every endpoint, the exact set of tensor coordinates equals the full frozen model dimensions and valid old-hidden micro rows/widths, with no missing or duplicate coordinate. This includes B's padded16-output head. Each endpoint joins its preceding completion family/mode/plan/update, has the expected cumulative sample count, and its maximum is exactly the largest of its own entry errors.

| Mode group | Endpoints | Largest absolute error | Frozen cap |
|---|---:|---|---|
| I |11 |`8.56300169660428878e-9` |1e-5 |
| NI, all three placements |29 |`9.06713890636012820e-8` |3e-5 |

The largest I error occurs at B/I/update3; largest NI at A/AU/update0. All40 TSV rows preserve the original maximum/cap tokens exactly and match their completion/endpoint joins, including mode and plan. The TSV is a read-only extraction of this synthetic test, not a scientific JSON/table, dataset-accuracy result, general-model precision bound or performance measurement. No tolerance, recipe or comparison criterion was changed.

## Poison, diagnostic copies and retention

Both poison sessions assert successfulindex0/samples0, permanent poison/retry rejection, source post-state metadata and changed source ciphertext payload, exact all-limb unchanged downstream initial expanded/NIcompact payload, available source trace and empty destination trace. Source-commit evidence is not an extra decrypted prefix-oracle comparison. Both trace_changed vectors are[true,false]; trace equality alone is not execution provenance.

| Poison | Actual work / published prefix | Retained measured partial vector: rotations,relins,ct_ct_tensor,ct_pt,rescales,SSE,boots |
|---|---|---|
| I |15refreshes;1I recorder output |154,26,36,35,57,0,0 |
| AU |5boots;0published NI boundary outputs |300,220,266,1064,325,10,5 |

The source wrapper converts the second boundary before the injected refusal; its work remains in these measured partial deltas. The deltas are retained self-consistency evidence, **not an independently predicted seven-field partial-operation oracle**. There is no rollback claim or invented successful failed-prefix receipt.

All128retention records were independently counted and their retained coefficient-byte sums recomputed:40packet,40endpoint,46full-snapshot and2poison-initial-live receipts. The two guarded initial captures contain I14CT/2,293,760bytes and NI18CT/4,784,128bytes. Their actual roles are added to the live receipts; source lifetime drops them after the corresponding failure comparison and before the retry snapshot. Thus46full-session+2partial model captureevents=48, with3configuration snapshots separate. No additional model encryption, oracle, update or failed-prefix decryption is present.

The earlier two minor notes remain deferred: public snapshots before first success/after poison may be incoherent, while this test exports them only immediately after success; transient negative-mutation payload copies are excluded diagnostic workspace, not all explicitly inventoried roles. The corrected source/runtime reports disclose both. Retention is observed quiescent coefficient payload plus separate analytical roles and unobserved workspace/caches/client storage/allocator overhead. It establishes no peak memory, RSS, resource/security admission or secure-memory gate.

## Paired and cumulative evidence; remaining limits

The shared phase has14oracle/14Engine/40actual full-block train/128external-forward guards, each contiguous at its correct cap, plus the reached final completion/count assertions. No poison oracle or pure mutant sweep ran in this invocation. The prior v1/v2 raw hashes still match their retained terminal records and the accepted reviews; independently recounted guards are156train/61Engine/179externalforward each. V1 remains exit101/FAIL, v2exit0/PASS. Accordingly P5 F64 cumulative work is **352train /136Engine /486externalforward**, plus14paired-oracle calls. Failed v1 consumption remains included; P4 regression and other task phases are excluded.

The15source pins cover the8owned files and selected fixture/NN dependencies, not the complete Rust/crypto dependency graph or a binary attestation. In particular, this inventory does not pin every trainer/activation/CKKS source; unchanged-code numerical dependencies and internal workspace remain outside exhaustive independent verification. The prior source review's preservation/selective-inspection limits continue to apply. Passing the test is actual evidence for its frozen synthetic path, not proof of all unchanged library arithmetic or canonical full-profile behavior.

Accepted pure v2's I1e-5/NI3e-5 and required>=8x separation remain distinct from retained v1's seven NI failures; this HE run neither reruns nor relabels those sweeps. Full21registered profiles, full2680+20+7 /quick268, canonical owned D7 gates/start/runner, secure-profile resource/range/security/precision, GPU/full encrypted-block campaigns, scientific accuracy/performance and fullP3/P5/D7/Task54/M4/M5 acceptance remain open. The bounded CPU outcome and corrected evidence are accepted within the scope above. No subsequent invocation or release follows from this review.
