# Task54 D4c3 NI primary independent actual-evidence audit

Date: 2026-11-07 JST. Scope: the retained first and only released NI primary attempt. This is an actual-log audit, not a new source implementation review or an HE rerun.

**Integrity: PASS. Registered six-case NI coverage and observable counter checks: PASS. Independent all-slot scalar parity: PASS at original E=2^-12. Independent per-bootstrap incremental error: PASS at original epsilon_boot=2^-20. All original role bounds: PASS. No blocking finding in this bounded NI attempt.**

This result covers N1024,512 complex slots, one held-out key seed167, the unchanged tiny original fixture, G2/G3 × AU/WN/AA, four updates and one final n3 inference per case. It does not release another experiment or establish secure production parameters, GPU/backend parity, performance, full-dataset or full-matrix scientific results. Raw counter, seed, generation and ciphertext evidence limits are explicitly retained below.

## Method and retained identity

Read the root NI release and runner, terminal runner metadata/stderr, FIX1 manifest and source review context, original normative fixture/§§5–7 and implementation brief, and the relevant `primary.rs` and `fixture.rs` execution/diagnostic paths. The new standalone stdlib script streams the complete95,120,811-byte stdout, decodes every bit payload, and verifies the full ordered protocol. It does not import Rust/model/packing code, the prior scalar JSON, or a producer's numerical summary as its oracle.

The independent original scalar reconstruction is copied as standalone audit code from the I audit: original W/X/Y, rates `[.04,.01,.02,.08]`, IDs `[0,1,2]`, `[0]`, `[0,2]`, `[0,1,2]`, local polynomial h=z²+z, old-head transpose gradient, total valid mean, V'=0.9V+g, W'=W-lr*(g+0.9V'), U=.08*1.9V'. All old forwards are captured before reverse-order application of local updates. Literal independent 512-slot packing preserves alternating matrix/vector orientation and all active/padding slots. Each case resets to the original fixture. Final inference uses final committed scalar weights and no intermediate local heads.

The audit independently constructs the expected332-attempt order from graph/stage/valid-ID rules and frozen refresh decisions, then cross-checks every role against the complete frozen metadata ROLE records. It constructs the expected ordered SLOTS/PARITY/attempt/event/trace/commit protocol, including all interleaving and inference phase boundaries. All4,050 expected records and their expected levels match. Parsed levels come from the already-frozen metadata traces/histories, not from fitting the actual NI result.

No test binary, Cargo, key generation, HE/bootstrap, network, SSH, Git operation, production-file edit, or subagent was used for this audit. Only these new audit artifacts were written. The audit script passed on its first execution. No raw evidence, fixture, reference, rate, tolerance, bound or seed was altered.

Runner metadata records source `ad607daf3e53c3763c4c5db25701f2ede5115154`, exit0, start `2026-11-07T05:36:23.603999+09:00`, finish `2026-11-07T05:38:26.816677+09:00`. All66 FIX1 manifest entries were rehashed, including nine owned source files, normative/evidence files and the prospective executable; their recorded sizes also match. The copied executable and current prospective executable match the pre/post hash. Unrelated bench working edits are recorded by the runner, but none changes the frozen NN source/executable used here.

| Artifact | SHA256 |
|---|---|
| stdout.txt,95,120,811 bytes | `d8a9f62fb828a2b9f27ce60adb83d626f5835739d4546317d1ddd0012c8a9678` |
| stderr.txt,158 bytes | `90d36c0705eb55e62ffb85cfc9f8c3ac06a9c689d0ecb3c41dda0e23c5ef3f73` |
| frozen/current executable | `91dfd7ef7bd451cc8e423c1fc7cd4241c379c22e777dcf30c6e15c73b26ee5c8` |
| copied FIX1 source manifest | `4788e0af8128c514b36a11cb651fa018ec39db8b7e140425d08be16c4915796e` |
| NI runner | `5c5b8307f9e3206dc926d5a81e98dba5d0088de33eb096526322d8d6948a15e8` |
| FIX1 re-review | `6bb24e383bfd8a82acb6e6335f41db2456b970f3daba6e8f3fb1e2dcbe682f3b` |

The command exactly matches `cargo test -p ckks-train-nn --test packed_blocks block_ni_original_fixture -- --ignored --exact --nocapture --test-threads=1`, in the requested hekit root, with `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8`. Stderr contains only Cargo Finished/Running messages for the expected executable. This is local source/runner/log evidence, not externally authenticated execution or an independently reproduced build.

## Parameters, seed scope and complete coverage

Actual PARAMS reports ni=true,N1024,dnum8,scale bits4796333603149578240 (=2^42),held_out167. Complete Q/P arrays exactly equal the frozen metadata arrays:32 Q primes with bit lengths `[54,42×16,34×4,58×7,56×4]`,4 P primes of60 bits, each1 mod2048. All BootParams fields match default_n16 with log_slots9, including4/7/4 transform levels,k13,degree127,double_angle0,sparse_hw=None,ephemeral_hw=Some(32). Every bootstrap input is level0 and output level16; every decoded record has2 components and exact Delta bits. A single context pointer occurs throughout; the pointer is a process-local label, not a cryptographic context fingerprint.

The frozen source uses key seed `[167;32]` once and domain1 encryption seeds `[195,1,G,mode,update,stage,record,role] || [0;24]`, modes1/2/3 for AU/WN/AA, initial W update/record255, fresh inputs stage255, final inference update4. No interactive client or encryption RNG is used by the NI bootstrap capability. Exact seed use remains source/executable-attested: neither per-encryption RNG seeds nor the secret-key/ciphertext archive are present in stdout, and this audit did not regenerate them.

Harness: exactly one `block_ni_original_fixture` execution; **1 passed,0 failed,0 ignored,0 measured,22 filtered out**. Terminal body: **6 cases,24 commits,6 final inferences,332 attempted/completed/validated,expected332,cap372**. No failure/panic/undecodable/nonfinite diagnostic occurs. Unknown or missing protocol lines would fail the audit; no such mismatch was found.

| Case, in exact order | Attempt indices | Training calls by t=0,1,2,3 | Training | Inference | Total | W / U / hidden, both phases |
|---|---|---|---:|---:|---:|---|
| G2 AU | 0–43 | 11,9,10,11 | 41 | 3 | 44 | 16 /16 /12 |
| G2 WN | 44–87 | 11,9,10,11 | 41 | 3 | 44 | 16 /16 /12 |
| G2 AA | 88–127 | 6,10,4,14 | 34 | 6 | 40 | 8 /8 /24 |
| G3 AU | 128–199 | 18,14,16,18 | 66 | 6 | 72 | 24 /24 /24 |
| G3 WN | 200–271 | 18,14,16,18 | 66 | 6 | 72 | 24 /24 /24 |
| G3 AA | 272–331 | 9,15,6,21 | 51 | 9 | 60 | 12 /12 /36 |
| Total | 0–331 | | **299** | **33** | **332** | **100 /100 /132** |

The132 hidden bootstraps split99 training+33 inference. All200 state bootstraps are training W/U; no biases occur. AU and WN refresh all states at every registered update. AA refreshes state only at t1 and t3, exactly as frozen, while bootstrapping every valid hidden within every stage. Non-AA hidden calls occur once per nonfinal boundary. AA has no extra boundary bootstrap. Inference has one bootstrap for each nonfinal hidden in AU/WN and every hidden in AA, with no state update.

The global ordinal never resets or skips. Every ATTEMPT identity includes correct graph,placement,phase,update,stage,physical record and role. Invalid record3 never appears in work; holes use original physical IDs0/2, not renumbering. All24 BEGIN and GRAPH_COMMIT records are ordered; global commit counts1..24 and every stage's first-layer update index1..4 are correct.

All60 training flags match n head-affine/residual entries and2 outer/update entries per stage. All15 inference flags have zero forbidden loss/outer/update/derivative entries,0 head affines on nonfinal stages and3 on final stages. `graph_commits=0` in the inference flag is printed literally; it is not an independent measured generation counter. Unchanged final W/U payloads provide additional direct state evidence.

Log inventory is exactly4,050 SLOTS,2,654 PARITY,332 each ATTEMPT_INTERVAL/ATTEMPT/INCREMENTAL/EVENT,200 PROJECTION,24 each BEGIN/UPDATE_COUNTS/GRAPH_COMMIT,60 TRACE_FLAGS and15 INFERENCE_TRACE_FLAGS. The1,396 raw records without an adjacent PARITY summary are664 attempt input/output records,400 projection source/raw records and332 range records. This audit independently compares **all4,050**, including those1,396 raw records, against scalar expectations.

## Independent numerical gates

All **2,073,600 complex slots /4,147,200 f64 components** are finite and decoded from retained integer bit patterns. Each individual complex scalar-parity error passes unchanged **E=2^-12=.000244140625**. This count includes intentional repeated observations of the same payload; it is not a count of independent statistical samples, seeds or experiments.

Maximum scalar-parity error is **3.441401240719656e-8**, G3 WN inference attempt267 output, hidden(stage0,record1), slot465. The matching event/post-hidden observations carry the same payload. Maximum imaginary leakage is **3.308081736605577e-8**, maximum replica spread **1.5443748512212872e-8**, maximum padding error **1.5454310611905637e-8**. The18 final n3 output records have maximum error **9.870789374419542e-9**.

Each of the332 raw level0 input/output pairs was compared directly across all512 complex slots. Every incremental error passes unchanged **epsilon_boot=2^-20=9.5367431640625e-7**. Maximum is **1.8763991003465872e-8**, attempt194,G3 AU inference,hidden(stage0,record0). This is against the actual decoded input, not against the ideal scalar vector and not a comparison of scalar maxima. No averaging across events or slots is used.

| Actual bootstrap-input role | Calls | Maximum complex magnitude | Fixed bound | Maximum input scalar error | Maximum incremental error |
|---|---:|---:|---:|---:|---:|
| W0, across stages/cases | 50 | .3811884708395674 | .4 | 1.5802302601546122e-8 | 1.1265365853915369e-8 |
| U0, across stages/cases | 50 | .07651061526290981 | .1 | 9.845490759721354e-9 | 1.0724585049337457e-8 |
| W1, across stages/cases | 50 | .3802074132140616 | .4 | 1.4662400025005946e-8 | 1.1244005309352888e-8 |
| U1, across stages/cases | 50 | .07632583926287725 | .1 | 1.0223618425024576e-8 | 1.1537530334984206e-8 |
| Training hidden | 99 | .42565953737395656 | .5 | 2.038202961046103e-8 | 1.634031256791772e-8 |
| Inference hidden | 33 | .5292030956908136 | .6 | 2.725207472428965e-8 | 1.8763991003465872e-8 |

Every input also passes universal magnitude<=1. All phase/role bounds are the original registered ones; inference uses the prospectively ratified .6, not training's .5. Related pre/post/projected/committed W/U/hidden observations also pass their applicable fixed role bound. Fresh inference inputs are compared to their original .75-scale values, not assigned a hidden bound. U remains fixed-reference-scaled velocity, not an unscaled V measurement.

All200 state projection drifts were independently recalculated from their full source/projected raw records. Maximum is **1.3013471442383563e-8**, attempt185. Projection drift is kept distinct from bootstrap incremental error; no new projection tolerance is invented. The projected/source/input/output snapshots individually pass E, and bootstrap epsilon is calculated only from actual bootstrap input/output.

| Case | SLOTS records | Maximum scalar error | Maximum incremental error |
|---|---:|---:|---:|
| G2 AU | 544 | 3.11295718309328e-8 | 1.7407898240161243e-8 |
| G2 WN | 544 | 3.014114936455641e-8 | 1.6593049259407835e-8 |
| G2 AA | 493 | 2.9172576570540833e-8 | 1.691219660339175e-8 |
| G3 AU | 861 | 2.7880783207019227e-8 | 1.8763991003465872e-8 |
| G3 WN | 861 | 3.441401240719656e-8 | 1.574316011487388e-8 |
| G3 AA | 747 | 2.3075916370395683e-8 | 1.795564948268938e-8 |

All2,654 reported worst indices match independent recomputation. All imaginary maxima match exactly. Maximum absolute report-vs-independent differences are1.1089202317039978e-16 for max/active parity,1.6543612251060553e-24 for padding and replica,1.1102230246251565e-16 for ranges,3.308722450212111e-24 for incremental errors and1.6543612251060553e-24 for projection drifts. These are compatible with independent scalar floating-point/hypot evaluation, not bit-identical Rust/Python arithmetic. Both reports and independent calculations pass their original gates; no replacement tolerance was selected to accept a discrepancy.

## Observable counters and state identity

All332 ATTEMPT before/after counters are present with all7 fields. Independent subtraction equals the separately logged ATTEMPT_INTERVAL and the matching EVENT interval exactly:

```text
(rotations,relins,ct_ct_tensor,ct_pt_mults,rescales,sse_switches,bootstraps)
=(37,40,46,207,56,2,1)
```

Every global counter is nondecreasing between attempts. Attempt i has before.bootstraps=i and after.bootstraps=i+1, with0..331 complete and validated; no gap, reset, hidden extra bootstrap between recorded calls, or count373 is observed. Matching event identity and decoded payload hashes bind each event to its correct attempt. Cap372 compliance follows, but reaching/refusing373 is not exercised by this successful run.

All24 logged UPDATE_COUNTS exactly equal the corresponding frozen metadata update counts for **all7 fields**, not merely bootstrap totals. Number of recorded attempts per update equals that update's bootstrap count. State ROLE metadata includes projection overhead, so its inclusive cost is B plus1 ct-pt multiplication and1 rescale; raw bootstrap ATTEMPT/EVENT intervals correctly remain B. The audit does not double-count child/event observations on top of inclusive graph update totals.

The log lacks a standalone actual INFERENCE_COUNTS or final full seven-field CASE/PHASE total. Exact inference bootstrap counts and every individual bootstrap interval are directly observable. Exact non-bootstrap inference counts/trace equality are source assertions checked by the successful reviewed execution path, not independently dumped totals. The last recorded attempt's after counters are `(14908,14182,16324,70411,21181,664,332)`; work can follow that final bootstrap, so those values are **not** labelled the full phase total. Frozen case/inference totals remain predictions corroborated by source assertions where actual aggregate fields are not emitted.

There are **2,502 checked decoded-payload identities**, requiring both the same canonical SHA256 of1024 f64 bit words and the same level. These bind attempt input/output to event input/range/output; projection source/raw aliases; pre-refresh candidate states to event sources and restored committed states to outputs; AA selected hidden outputs to post-hidden and boundary output/source; each committed state to the next old state or final post-inference unchanged state; and each inference selected hidden to path_output and the next child input. Non-refreshed AA state candidates also exactly match their committed candidates. No silently repaired scalar substitution is needed to explain any observed payload.

Every one of4,050 logged levels exactly matches the metadata-derived expectation. Bootstrap inputs0/output16, projected state source-minus-one, old/committed state histories, hidden/preactivation/output levels and alternating AA refresh/nonrefresh histories all match. State generations themselves are not logged. Update indices, phase-specific operation flags and payload continuity are direct evidence; complete generation-token identity remains source-asserted.

## Scope limits and remaining gates

- **Retrospective range evidence:** the recording capability retains encrypted arguments/results; decoding/range diagnostics execute externally after synchronous prepare returns. These measurements confirm actual ranges for this attempt. They do not prove admission before each bootstrap, universal interval bounds, or failure safety for an unseen input.
- **Decoded evidence, not independent decryption:** the attempt retains complete decoded bits, not a raw ciphertext/key archive. The audit independently recomputes all mathematics on those bits but cannot independently decrypt/authenticate their cryptographic provenance. Hash-equal decoded snapshots are not a standalone proof of ciphertext-limb identity; the reviewed source performs ciphertext identity binding separately.
- **Seeds and public-key-only boundary:** exact seeds, lack of a secret/diagnostic callback within the NI capability, and no output clipping/repair follow the frozen source/executable path. They cannot be reconstructed solely from numerical outputs. No seed search or alternate execution was conducted by this audit.
- **Unlogged intermediates:** fresh training input/target ciphertexts, invalid sentinel target3, training z/gradients, full primitive traces and raw generation tokens are not separate all-slot records. Their construction/structural validation and effects are source-supported. This report does not claim direct measurement of every unlogged tensor or exact AA affine2/outer1 operand ciphertext identity beyond the source assertions and recorded selected/output consistency.
- **Inference labels:** ordinary `NI inference …` and `post inference unchanged …` labels omit graph/placement. The complete enclosing ordered protocol and explicit ATTEMPT identities disambiguate all six cases. A detached inference line would be ambiguous. NI inference has no separate handoff-record family in this path; selected hidden/next-input identity is checked directly.
- **Generations and commits:** commit records contain first-layer indices and global counts; inference's zero-commit flag is literal. All four W/U roles preserve exact observed values after final inference, but an independent raw generation counter dump is absent.
- **Counts:** complete training UPDATE_COUNTS and per-bootstrap before/after intervals are independently checked. Full actual inference/case seven-field totals are not printed; do not relabel frozen predicted totals or the last event's after counter as an independently measured phase total.
- **No failure experiment:** all332 calls complete and validate. Retained partial-failure machinery, cap refusal, memory admission, host panic behavior and real numerical failure diagnostics are not newly exercised by this passing attempt. Earlier controlled/source reviews remain separate evidence.
- **Scientific limits:** one small insecure N1024 correctness context and one held-out key seed do not establish production CKKS security, scalable memory/time, full profile/batch/data coverage, CUDA/GPU correctness, training quality, efficiency rankings, or the21 profiles×10 seeds×CPU/GPU×I/NI/ablations matrix. Wall-clock runner timestamps are provenance, not a controlled performance benchmark.

No observed integrity, coverage, range, epsilon or scalar-parity defect blocks accepting this registered NI attempt. The actual bounded NI correctness gate is supported by the retained evidence. All larger production/backend/security/scientific campaign gates remain open and require their own scope/release decisions.

## Audit artifacts

Reproduce only the read-only calculation from repo root with `python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d4c3-ni-runtime-audit.py`. The standalone script reads retained evidence and writes derived JSON; it never launches Cargo or the test binary. Derived files do not duplicate the95 MB raw slot log. JSON keeps per-snapshot statistics and payload digests, all332 roles/counter intervals,200 projection drifts,24 update counts, and per-case/per-role summaries.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `task-54-d4c3-ni-runtime-audit.py` | 29,238 | `26a2e189ba79b380e65490a120ac10f8afa43a5d594627422d47afe2549424de` |
| `task-54-d4c3-ni-runtime-audit.json` | 3,094,270 | `0884fad47339f052be9a8ec05b82c73801b5c49e4ff34060674de893c8189800` |
| `task-54-d4c3-ni-runtime-audit-summary.txt` | 4,234 | `55b48f7b761bf6395a4ddf8eded22306d8d5020845f9cccc509062c42e594c41` |
