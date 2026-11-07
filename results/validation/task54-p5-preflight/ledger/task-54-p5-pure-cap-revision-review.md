# Task54 P5 pure cap revision v2 — scoped source/decision review

2026-11-07. **SPEC: PASS for the exact two-literal source/static revision and documented PRE-HE decision. QUALITY: PASS. No blocking findings.** This is not a v2 numerical PASS, HE acceptance or runtime release. The completed v1 result remains FAIL with seven NI sensitivity failures.

Reviewed the release, corrected report, exact diff/inverse identity, current two-source pins, retained compile/static/custody records, protected original evidence, accepted fix1 source review, runtime1 outcome review and design §14. This review did not repeat the full fixture/source/numerical audit. Only read/hash/text work, arithmetic on retained numbers and this review-file write occurred; no source edits, Cargo/tests, fixture/Engine/forward/HE/constructor, network, Git operation or subagents.

## Exact source and evidence pins

| Item | SHA256 |
|---|---|
| Cap revision release | 42f40f4bddfb026cdbc0189a8ba4c6979274a9336314795ca574ad384402c085 |
| Corrected cap revision report | bfc1e46c6ff0cf0e591fae9cab26abaa5f107d8d5933010c45205f7fd083ce44 |
| Review pins JSON | 0272cd10c7d3d0b2ff424523cd6c72812f9f064202c95535161b0f97f4ecd3e8 |
| Exact two-literal diff | c18d6e6fab709b2cc600625d837c1112c09424d2aecdddd7fcca6d755756f5e7 |
| Accepted prior fix1 source review | 8b6a17f8302ded7c60f809071c4f60bbd5cced29b07b2ee33c37ed940c84b41e |
| Original runtime1 outcome review | a78cbfa4aab354cfe8770b257d0c1ee171bc069b49bb23d2c9b05377853a5984 |
| Current design document | 62b344a262e392a51ec75e8e78918ea2c69021b64c8ece72c91317d9c223da9d |
| Design §14 bytes through EOF | 069002af6f6759513521c650b38e4f8190d14282cbe37e3c633a88f9203f6b55 |

The accepted prior package is fix1, which resolved the original review's S1 shared paired-phase budget finding. Its frozen fixture SHA is `fbf604ddc4a3dfae6f3c37bd6909d35adee8f83f639453fb97239bbd46bb2943`; the original initial source review's earlier fixture pin is historical, not the baseline for this revision.

Independently compared saved before/final and current source bytes. Exactly one VERSION literal changes `task54-p5-pure-v1`→`task54-p5-pure-v2`; exactly one NI cap changes `4e-4`→`3e-5`. The full reconstructed unified diff equals `two-literal.diff`. Reversing only those replacements yields exact before bytes and original fbf604dd hash, confirming inverse-identity.json independently.

Current fixture and saved final both hash to `cd580c14c8f6aa8099c113701890ec2231bb5b162f2375b63103c6562dcdb99e`. Parent `study_plain.rs` remains `fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a`. Both match review pins, source-final.sha256, final custody and every static command's before/after inventory.

Because the inverse restores every fixture byte, arrays, initial states, row order/labels, rates/reference, parts, controls/branches, counts, fixed I1e-5 and required>=8 rule are unchanged. The existing shared paired-phase limits and closed source boundary are also unchanged. No broader source acceptance is inferred from this narrow equality proof.

## Static checks and original custody

Retained records show owned rustfmt --check/edition2024/skip_children=true exit0; exact focused `cargo test -p ckks-train-bench --lib study_plain::p5_fixtures::p5_pure_block_sensitivity_gate --no-run` exit0; and `cargo clippy -p ckks-train-bench --lib --tests -- -D warnings` exit0. All use CARGO_INCREMENTAL0/CARGO_BUILD_JOBS8/RAYON_NUM_THREADS8, retained cwd/PID/start/end/free-disk records and unchanged before/after source inventories. Raw outputs agree: formatting streams empty, compile and Clippy stdout empty, stderr contains successful compilation/checking only, no warnings or test execution summary.

The retained validation wrapper requires --no-run for cargo test and starts only each supplied static command; these actual command records contain no binary invocation. The compiled binary currently matches retained8,342,544bytes/SHA256 `14a09f97113bf92b29dc6df2baaa8f478dac2613558a068beb9f6aead18c008b`. I read/hashed it and did not execute it. Tool-session records are retained custody, not a new query of those handles. Compile/static PASS establishes no runtime guard, margin/range/invariance or HE result.

All14 protected-before and protected-after entries are identical and every listed current file matches its pin. This covers original source/runtime1 reports, outcome review, launch/process/terminal/raw/margins records and all six preserved before-cap-v2 files. The original raw remains5,677,785bytes/SHA256 `4702340394d79d98135401b3538a833ba6d0a1b8f70d7b738a3c195b12461b32`; original margins TSV remains SHA256 `7d9cd886b8e05f46bd10f67c273ace2740b1f5d4d280680bcfe0e6a8d5e6b182`. No output was relabeled v2 or changed to PASS.

## Decision and truthful outcome boundary

Design §14 accurately records the independently reviewed v1 outcome:156actual Engine::train calls,61Engine constructions,179explicit external forwards, all completed; I36/36PASS, NI29/36PASS and7FAIL. Its references to698 passing checks,35 baseline range checks and maximum hidden absolute value0.652500000000000080 agree with the accepted runtime review. These retained checks are synthetic F64 observations, not secure-profile or scientific accuracy evidence.

A bounded read of the unchanged TSV independently confirms72rows,36per I/NI, seven NI failures, original caps I1e-5/NI4e-4, required8 and minimum difference3.17824570814917440e-4. Arithmetic gives minimum_difference/8≈3.972807135186468e-5, prospective ratio at3e-5≈10.594152360497247 and tightening factor4e-4/3e-5≈13.3333. These calculations do not execute the v2 trajectory or establish any encrypted error.

The choice explicitly follows the pure v1 failure but precedes any P5 HE result. It preserves the diagnostic trajectories while demanding a smaller future HE error. The alternative of strengthening inputs/initial values would change trajectories and require fresh range/locality evidence; reducing8x or removing failed controls is expressly rejected. Future failure at3e-5 remains a failure with no automatic return to4e-4. This is a new, documented synthetic PRE-HE criterion, not a retroactive scientific threshold adjustment, P4 reclassification or v1 PASS.

No v2 fixture/Engine/forward/test runtime was performed in this source/static assignment. Original consumed work remains156train/61Engine/179forward. A later complete authorized v2 run would add the same quantities, yielding cumulative312train/122Engine/358forward. The separate paired phase remains14oracles/14Engine/40train/128external-forward, unexecuted; neither static compilation nor this review consumes or releases it.

Canonical/D7/durable start, secure-profile resource/range/security/precision, HE pairing, full registered/full-quick campaigns, scientific/performance gates and full P5/Task54 acceptance remain open. This exact source/static revision is suitable for root's next separate freeze/release decision; no runtime or HE authority is granted here.
