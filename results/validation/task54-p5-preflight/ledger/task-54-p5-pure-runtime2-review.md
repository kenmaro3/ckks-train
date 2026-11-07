# Task54 P5 pure runtime2 — independent outcome review

2026-11-07. **SPEC outcome: PASS for the frozen v2 synthetic F64 sensitivity gate at I1e-5/NI3e-5 and required>=8x. QUALITY: PASS. EVIDENCE: PASS. No blocking finding.** Original v1 remains FAIL with seven NI failures at4e-4. This review establishes no HE precision, canonical admission, secure-profile, performance or campaign acceptance and grants no new execution.

Read the runtime2 report/release/authorization/controller/pins, complete retained record streams, launch/process/terminal custody, margins TSV/comparison JSON and accepted runtime1/cap-revision reviews. Independently checked hashes, record identity, guard sequences, numeric/bit joins and arithmetic. No tests, Cargo, fixture/Engine/forward/HE, constructors, source edits, network, Git operation or subagents were used. Only read/hash/arithmetic work and this review-file write occurred.

## Frozen custody and actual execution

| Evidence | Verified SHA256 |
|---|---|
| Runtime2 report | 749313afc6fa300b1c6000a3852123ab53307cb005ab460980a6c7f7c50d5289 |
| Runtime2 release | b4ec5cd86ff6852697bc4788b32fecfcdf6bc8fb384efd862bbe71c9e420d5b1 |
| Authorization | 6e96e66f8265b78de9ec7df6dbdae2d285c27926111329f9c784a8a0d699fdec |
| Controller | d98ba4ce951ac367d093fc283b50de3ca5e8dd7704c6305dda02a5979fe2a838 |
| Two-source pins | 0272cd10c7d3d0b2ff424523cd6c72812f9f064202c95535161b0f97f4ecd3e8 |
| V2 raw,5,675,618bytes | ccd416c4fb342d6e2a0f75035d1dfd07aff9160ad1822a3ddb54a70e371a51be |
| V2 margins TSV | 35c085781ea3aef259e46e135bc1e4aaedcaee7dd3d1d336706b27746ab727c4 |
| V1/v2 comparison JSON | 486c57a5f142b6e1d5e04748ea19457d20450f1b080844facda74dc8aeaed28d |
| V1 raw,5,677,785bytes | 4702340394d79d98135401b3538a833ba6d0a1b8f70d7b738a3c195b12461b32 |

Authorization joins the exact accepted cap review5bc86890, corrected source reportbfc1e46c, release and two-source pins. Current parent SHA256 `fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a` and fixture SHA256 `cd580c14c8f6aa8099c113701890ec2231bb5b162f2375b63103c6562dcdb99e` match both terminal inventories and the pin file. Launch is an exact subset of terminal; process/controller PID joins match. Terminal binds the complete raw bytes/hash, unchanged source, test_exit0/controller_exit0, controller11854/Cargo11855 and Unix start1794038874.866776/end1794038889.385102.

The controller checks the frozen authorization/source inputs, uses exclusive fresh evidence files, starts one exact selected subprocess and has no automatic retry path. The command selects only `study_plain::p5_fixtures::p5_pure_block_sensitivity_gate` with --ignored/--exact/--nocapture/--test-threads=1; bound environment is CARGO_INCREMENTAL0, CARGO_BUILD_JOBS8, RAYON_NUM_THREADS8 and the existing target. Free disk60,425,277,440bytes exceeds the recorded805,306,368-byte floor.

The raw has exactly one selected test:1passed/0failed/0ignored/192filtered. The separate empty-phase test is not executed. Root reports original handle92692 terminal0; this is root-observed tool custody, not a handle independently encoded in the raw. Test0.17s and controller timing are custody only, not performance measurements.

## Independently verified actual work

Every BEFORE_CALL has its correct operation cap and contiguous per-operation actual ordinal through its limit, without excess or oracle guards. All396guards account for61Engine constructions,156actual Engine::train calls and179explicit external forwards. Engine-internal work is not counted again.

| Family | Train | Engine objects | External forwards |
|---|---:|---:|---:|
| A |42 |13 |42 |
| B |87 |26 |103 |
| C |11 |12 |12 |
| D |16 |10 |22 |
| Total |156 |61 |179 |

Per-family guards agree with retained FAMILY_SUMMARY actual/expected counts. Final SUMMARY reports40controls,72margins,1equivalent,3inapplicable,0failures,PASS. Control records independently count36applicable/1equivalent/3inapplicable. All698CHECK records pass. Frozen records for all four families finish before the first Engine guard; this includes the first MANIFEST attached to the harness line.

All22,235 VALUE tokens are finite and round-trip to their stored F64 bits; all4368 frozen b/velocity_b values have exact positive-zero bits. Each of35 RANGE maxima was independently recomputed from its matching old_hidden VALUE records and matches both its printed maximum and stored bits. Maximum is6.52500000000000080e-1<.75. These are actual synthetic F64 observations, not secure-profile range/precision evidence.

## V1/v2 identity and margin joins

Extracted evidence records from both complete immutable streams, retaining their original record bytes and order. The following30,836records are exactly identical: RECIPE4, CHECK698, ARRAY1237, VALUE22235, VECTOR520, STAGE9, UPDATE11, LABELS137, BEFORE_CALL396, RANGE35, TRAIN_INPUT156, CONTROL40, COMPARISON5354, FAMILY_SUMMARY4. This preserves the prior accepted target-isolation/state/range evidence rather than inferring a new trajectory from a changed summary.

The derived comparison hash `0b6b5d708b023ae143212dd86f42d97d54fa764de8cde42f4daf7dcf8afb26d3` independently matches this precise projection: strip the literal `NONSCIENTIFIC_TEST_EVIDENCE ` prefix from each unchanged record, join in original order with LF and include a final LF. This is a derived record-projection digest, distinct from either complete raw-file digest.

All72MARGIN records retain exactly the same context/backend/location/tensor/entry, original values/bits, absolute differences/bits and required8. All36I records are byte-identical in full. Each TSV row matches all raw MARGIN field tokens exactly. Every original numeric value and difference/cap bit field round-trips; absolute subtraction and normalized difference/cap reproduce the raw difference/margin, and each witness joins the same unique COMPARISON context/layer/tensor/entry and values.

Within test-evidence records, the only differences are the exact MANIFEST version v1→v2, NI cap/bits/normalized margins/statuses, removal of v1's seven FAILURE records, and final SUMMARY failures7/FAIL→0/PASS. No other emitted evidence record changed. Ordinary Cargo/harness timing and failed-versus-successful harness diagnostics also differ outside the evidence projection, so this is not a claim that the complete raw files differ only at those fields.

| Gate | Rows | PASS/FAIL | Minimum absolute difference | Minimum normalized margin |
|---|---:|---|---|---|
| V2 I,1e-5 |36 |36/0 |3.17824570814917440e-4 |31.782457081491742 |
| V2 NI,3e-5 |36 |36/0 |3.17824570814917440e-4 |10.594152360497247 |
| Retained V1 NI,4e-4 |36 |29/7 |same |0.7945614270372936 |

The original v1 raw still contains its seven failures and failed harness result. The v2 acceptance is under the separately documented/frozen stricter prospective HE error cap; it does not relabel v1 or show that encrypted execution can achieve3e-5. Byte identity concerns emitted inputs/state/witnesses and the accepted unchanged execution source; it is not an additional direct measurement of unlogged internal gradients.

## Disposition and limits

Actual cumulative pure work is312Engine::train calls,122Engine constructions and358external forwards across v1+v2. The consumed failed v1 remains counted. The retained records and exact selected source support no paired-oracle phase or CKKS parameter/planner/proof/key/Encoder/evaluator/bootstrapper/ciphertext/HE execution in these pure runs.

Paired-oracle14/14/40/128 and the future paired CPU HE matrix remain unexecuted. Actual block client/session/packet/retention behavior, HE3e-5 feasibility, canonical D7/gates/start, resource/range/security, GPU, registered full/quick scientific campaigns and full P5/Task54/M4/M5 acceptance remain open. This outcome supports bounded v2 pure-fixture acceptance and truthful publication only; subsequent source and execution releases remain separate root decisions.
