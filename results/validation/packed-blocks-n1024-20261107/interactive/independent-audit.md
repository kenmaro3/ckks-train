# Task54 D4c3 I-phase independent actual-evidence audit

Date: 2026-11-07 JST. Scope: retained first and only released I attempt, not a source implementation review or an HE rerun.

**Integrity: PASS. Registered I coverage: PASS. Independent numerical parity of every retained snapshot: PASS at the original E=2^-12. No blocking finding in this bounded I attempt. NI remains unreleased.**

These conclusions cover the original N1024, 512-complex-slot fixture, G2/G3, four updates and one final n3 inference per graph, one held-out key seed167. They do not establish NI correctness, bootstrap precision, GPU results, performance, security, production readiness, or the full scientific matrix. The absence of separate raw counters/seeds/targets and ciphertexts limits what this audit can independently verify, as stated below.

## Evidence and independent method

Read the root I release, runner, complete runner metadata/stderr, FIX1 manifest and re-review, normative design fixture and §§5–7, and relevant current `primary.rs`, `fixture.rs`, parameter construction, and receipt semantics. The complete 16,550,125-byte stdout was parsed, not sampled. All 710 SLOTS records and their 710 immediately following PARITY summaries were decoded and evaluated; no slot payload was omitted.

The new stdlib-only `task-54-d4c3-i-runtime-audit.py` reconstructs the original matrices, X/Y, valid-ID lists, learning rates and scalar equations directly. It does not import the Rust scalar implementation, prior Python scalar script, stored scalar JSON, or production pack/mask helpers. It captures all old stage forwards, computes local-loss gradients, then applies NAG updates in reverse stage order. Each stage uses total valid n=[3,1,2,3], V'=0.9V+g, W'=W-lr*(g+0.9V'), and U=0.08*1.9V'. The independent literal slot map is s=4*i+j with alternating matrix orientation and Expanded/Repeated vector representations. This independently reconstructed scalar expectation, not an aggregate self-report, is used for all 710×512 complex comparisons.

The audit generates the complete expected ordered protocol from G2/G3, stage/layer counts, physical IDs and callback rules. It then requires equality with the observed BEGIN/SLOTS/trace/commit/inference sequence, including expected repeated labels. A missing/extra/out-of-order client, stage, physical ID, slot record or commit fails that comparison. Unknown stdout lines, malformed records, nonfinite components, unexpected harness results, and failure diagnostics are rejected.

Only Python read-only calculations and read-only `git show <commit>:<path>` were used to inspect source identity. No Cargo/build/test invocation, keys, HE/bootstrap, network, SSH, Git mutation, or subagent was used. Writes are these audit artifacts in the ledger. An initial audit-script assumption that two same-labelled candidate payloads should match was corrected after reading the documented receipt semantics: they intentionally represent pre-refresh and post-refresh states. The actual evidence and numerical gates were unchanged; no scientific attempt was repeated.

## Integrity and provenance

- Runner records source commit `ad607daf3e53c3763c4c5db25701f2ede5115154`, exit0, start `2026-11-07T05:27:05.950818+09:00`, finish `2026-11-07T05:27:09.557224+09:00`.
- All **66 manifest entries**, including the nine owned source files, retained prior artifacts, normative documents, metadata evidence and executable, independently match both SHA256 and recorded sizes where present. All nine owned source files also match the contents at the integrated commit via read-only `git show`.
- Frozen executable and current prospective executable match the recorded pre/post executable SHA256. Runner source and FIX1 re-review independently match their pinned SHA256 values.
- Stdout and stderr independently match runner SHA256/byte lengths. Stderr is 158 bytes and contains only Cargo's Finished/Running lines; no error output. The stderr executable path agrees with the frozen executable identity.
- Exact command is `cargo test -p ckks-train-nn --test packed_blocks block_interactive_original_fixture -- --ignored --exact --nocapture --test-threads=1`; recorded cwd is the requested hekit root. Environment is `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8`.
- This runner ran Cargo, not the copied executable directly. The retained copy, source assertions, stderr path and unchanged pre/post hash corroborate which executable Cargo ran. These are local runner attestations, not an external tamper-proof execution attestation or independent reproducible build.

| Artifact | SHA256 |
|---|---|
| stdout.txt | `7256d4572d8da7e8b87dad7f193106500a998d3d245f116eb7365a2ef8593afd` |
| stderr.txt | `b1e2c1bd1be90ce23c277501f5b1148e06950a21afc2bf7b977ec9f43bfb6201` |
| frozen/current test executable | `91dfd7ef7bd451cc8e423c1fc7cd4241c379c22e777dcf30c6e15c73b26ee5c8` |
| copied FIX1 source manifest | `4788e0af8128c514b36a11cb651fa018ec39db8b7e140425d08be16c4915796e` |
| root I runner | `d8770ace57c2cd6de0ca1a7f17892cdf3de14e1616d425b3aa9dd13f6cc9962c` |
| FIX1 independent re-review | `6bb24e383bfd8a82acb6e6335f41db2456b970f3daba6e8f3fb1e2dcbe682f3b` |

## Parameters, seeds and coverage

The single actual PARAMS line reports ni=false, N1024, dnum8, Delta bits4796333603149578240 (=2^42), held_out167. Its complete Q/P arrays exactly match the frozen metadata log, including all32 Q and4 P entries. Bit lengths are `[54,42×16,34×4,58×7,56×4]` and `[60×4]`; every modulus is1 mod2048. BootParams fields match the registered default_n16/log_slots9 profile, but no Bootstrapper is constructed for this I path. One context pointer is observed across all710 records; a pointer alone is a process-local label, not a parameter hash. Every record has2 components, exact Delta bits, and512 finite complex values.

The source binds held-out key seed `[167;32]`; initial/fresh encryption uses `[195,1,G,0,update,stage,record,role] || [0;24]`, including initial W update/record255, fresh-input stage255, and inference update4. Client refresh uses `[195,2,G,0] || LE64(global_index) || [0;20]`. Inspection confirms the attempts vector persists across both graphs and is not reset. Observed client indices are exactly0..115 in the correct G2/G3 order. There is **no per-encryption seed payload in stdout**: exact RNG-seed use is supported by the frozen source/executable and execution path, not independently recoverable from decoded values. This audit did not regenerate keys or ciphertexts to verify seeds.

Harness: exactly one registered primary executed, **1 passed,0 failed,0 ignored,0 measured,22 filtered out**. Body: two graphs,8 commits,2 final inferences,116 attempted/completed/validated calls. All8 BEGIN records,8 global commits with consecutive count1..8 and stage indices1..4,20 training flags and5 inference flags exactly match the expected order. There are no NI event/attempt records, failure diagnostics or nonfinite values.

| Graph/phase | Callback indices | W | U | hidden | Total |
|---|---|---:|---:|---:|---:|
| G2 training | 0–40 | 16 | 16 | 9 | 41 |
| G2 inference | 41–43 | 0 | 0 | 3 | 3 |
| G3 training | 44–109 | 24 | 24 | 18 | 66 |
| G3 inference | 110–115 | 0 | 0 | 6 | 6 |
| Total | 0–115 | 40 | 40 | 36 | 116 |

Training update call counts are G2 `[11,9,10,11]` and G3 `[18,14,16,18]`. At every stage, callback order is W0,U0,W1,U1, then each valid hidden physical ID if another stage follows. Training hidden IDs follow `[0,1,2]`, `[0]`, `[0,2]`, `[0,1,2]`; inference hidden IDs0,1,2. No invalid physical ID3 callback, final-stage boundary callback, state callback during inference or reset of the global index occurs.

Every client input has the expected level: both W0/U0 level4, both W1/U1 level7, hidden level12; all116 outputs are level16. Across the whole log the level histogram is level4=80, level7=80, level10=57, level12=141, level14=15, level16=337. No level/scale/components/slot-count anomaly is observed.

Training flags show one head affine and residual per valid record, exactly2 outer-sum and2 state-update trace entries per stage. Inference shows zero head affines on every nonfinal stage,3 on each final stage, and zero forbidden loss/outer/update/derivative entries. Logged `graph_commits=0` in inference is a literal diagnostic field, not an independently measured counter. The unchanged W/U payloads and frozen source path supply the supporting state evidence.

## Full numerical recomputation

All **363,520 complex slots /727,040 f64 components** are finite and parsed from the retained integer bit patterns. Every individual complex error against the independent scalar expectation is <=the unchanged **E=0.000244140625**. This is an all-slot/all-snapshot gate, with no averaging or selection of a favorable subset.

Maximum error is **3.775018180620039e-9**, G3 inference, client111 hidden(stage0,record1) input, slot393; it also appears in the matching repeated hidden/path/handoff observations. Maximum imaginary leakage is **2.7634222896069656e-9**; maximum replica spread **2.9242414831335105e-9**; maximum padding error **5.267728700083869e-10**. These maxima were computed from raw bits, not copied from summaries.

| Recorded callback input family | Count | Maximum complex magnitude | Original fixed bound | Maximum scalar parity error |
|---|---:|---:|---:|---:|
| Training W | 40 | .38118846766530157 | .4 | 5.186765468434188e-10 |
| Training U | 40 | .07651060983845534 | .1 | 5.04043993059589e-10 |
| Training hidden | 27 | .42565953479332425 | .5 | 2.8420234165034265e-9 |
| Inference hidden | 9 | .529203094933428 | .6 | 3.775018180620039e-9 |

These fixed bounds also pass on every corresponding logged W/U/training-hidden/inference-hidden snapshot where applicable. They are descriptive I range checks; I has no bootstrap range admission and this does not predict NI bootstrap input ranges. The inference maximum correctly exceeds the obsolete .5 training envelope and passes the prospectively ratified .6 inference bound. No bound, fixture, rate, seed or gate was changed after this observation. Input vectors are not assigned a hidden bound: inference input magnitude reaches .7500000000589007, reflecting encryption error around the original .75 input.

The full log contains275 G2 and435 G3 snapshots. Its non-client families cover130 W,120 U,45 training hidden,45 training outputs,54 training handoff records,15 each of inference input/z/path_output/hidden_pre,18 inference handoff records and6 final outputs, alongside232 client input/output records. Every family is compared with the correct independently evolved phase/stage/record state.

All710 reported worst indices equal the independent recomputation. All710 imaginary maxima match exactly. Maximum absolute summary-vs-recompute difference is1.1034966121207134e-16 for max/active error,1.0339757656912846e-25 for padding, and2.0679515313825692e-25 for replica. The first difference is compatible with independent floating-point scalar evaluation, and the latter differences with hypot rounding; exact bit-identical arithmetic between Rust and Python is not claimed. Both independently computed and reported maxima pass the original E, with no new tolerance replacing E.

The independent canonical-repack calculation zeros matrix padding, preserves each active real matrix value, and averages vector replicas by representation. Maximum output-vs-canonical-repack complex error across116 callbacks is2.7915625132540167e-10. This is a derived consistency statistic, not an added bootstrap incremental-error gate. No clipping/model-update repair is needed to explain the recorded outputs.

## Payload identity and interpretive limits

There are **312 exact decoded-bit identity checks**:80 pre-refresh state snapshots equal the corresponding client inputs,80 whole-graph candidate state snapshots equal client outputs,80 restored candidates equal the next old state or final inference unchanged state, and72 hidden input/output observations equal their handoff source/output observations. These are identities of decoded-slot bit payloads, not independently retained ciphertext-limb identities.

Two ambiguities must survive the handoff:

1. `I G… t=… candidate stage=… W/U` is emitted twice with the same label. The first is `PackedInteractiveReceipt::updated_stage()`, documented as **before client refresh**; the second is the whole-graph candidate **after refresh**. They differ numerically and in level by design. The audit preserves ordinal context and binds each to its correct callback side instead of deduplicating labels.
2. `I inference …` omits G and has no inference BEGIN line. Its graph identity is recovered from the complete enclosing ordered protocol and global callback indices, then checked against the expected G2 and G3 snapshots. A standalone copied inference line would be ambiguous.

Limits of independently recoverable runtime evidence:

- The I log does not print complete seven-field actual operation counts/trace payloads or per-callback response tokens. Frozen source asserts exact child traces/counts against frozen schedules before success, and response construction/server validation enforce the protocol. Exit0 and exact terminal coverage corroborate these assertions. This audit cannot independently recalculate encrypted operation counters or token identities from SLOTS alone.
- Zero bootstraps is enforced by the frozen source's final counter assertion and by the I-only path without Bootstrapper construction, with no bootstrap-event output. `I_END` itself does not print a numeric bootstrap field. This is source-plus-success evidence, not a separately dumped raw counter interval.
- Fresh training inputs/targets, invalid sentinel target3, training preactivations and gradients are not separate SLOTS families. Correct target representation and invalid-record exclusion are supported by frozen construction/preflight/source and the independently reconstructed resulting outputs/states. This report does not claim direct all-slot measurement of unlogged fresh target ciphertexts or every intermediate.
- No raw ciphertext/secret-key archive is present in this attempt directory; the independently auditable numeric object is the full decoded bit payload emitted by the reviewed test harness. The audit does not independently decrypt or authenticate ciphertext provenance.
- Final committed state is checked via exact prior candidate/next-old and final unchanged payloads. There is no extra standalone post-commit decrypt after each commit, and inference's zero-commit flag is not a measured generation counter. These distinctions prevent overclaiming the diagnostic fields.
- Source/runner identities demonstrate the retained local attempt; no independent host attestation, external CI run, real-failure run, generic backend/CUDA parity, memory-admission measurement, or performance benchmark was performed.

No limitation above produces a contrary observed I result or blocks accepting this registered bounded I attempt. They do prevent describing the report as independent verification of raw encryption seeds, complete operation counters, unlogged targets, bootstrap epsilon, or a larger experiment. Epsilon_boot=2^-20 remains untested because there were zero I bootstraps. NI requires its own explicit root release and actual attempt audit. The production/full-dataset matrix and user-held campaign gates remain open.

## Reproducible audit artifacts

Run from the repo root: `python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d4c3-i-runtime-audit.py`. This only reads retained evidence and writes its derived audit JSON; it does not execute the test binary.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `task-54-d4c3-i-runtime-audit.py` | 18,637 | `43099bbfaf353db0af9bb49e07f9ee878d5d29f7fa209bc42ae95b149273e9c7` |
| `task-54-d4c3-i-runtime-audit.json` | 341,294 | `991ebc7a7333d12902f8ff7bb33910bf1b3ec46e7b20a67113cdb199f4bed650` |
| `task-54-d4c3-i-runtime-audit-summary.txt` | 6,270 | `4f6c0edc6da96314b07943edf0facddd73ff9c9fd4c0a01c014a9fdf948c3d7e` |

JSON retains one modest numeric summary per recorded snapshot and the full116-entry role protocol. Full raw slot payloads remain solely in the unchanged original stdout. The audit script's digest is also embedded in JSON.
