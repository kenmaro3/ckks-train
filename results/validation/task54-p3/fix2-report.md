# Task54 P3 fix2 — isolated omission discriminators

2026-11-07. **Tests-only source/evidence handoff for narrow independent rereview.** Source/Cargo ownership returned to root. Production encoders, schemas, arithmetic and all other test code remain byte-identical to fix1. Full P3/P4/D7 admission and the real pre-key factory spy remain required/pending.

## Exact correction

Only `crates/ckks-train-bench/src/study_admission.rs::tests::p3_digest_golden_and_mutations` changed. Its SHA256 is now `697f76cd289b626b7167d569561befe9b36375c03baa8971c0f0d7de34c6e561`. `scope-proof.txt` compares the exact bytes before/after excluding only that function; the remaining bytes are equal, with excluded-function identity `d7dc0543d5eadeb6568def2ae9dff2dab5cd5c680fca9d2e7fe11bae67ef24fd`. The other six pinned P3/P4 files are unchanged from the before snapshot. No helper or production function was modified.

| Gap | Actual new discriminator |
|---|---|
| R3a: individual rate fields | Actual production `encode_rate_schedule_fields` receives three raw entries containing distinct values1..12 and fixed reference16. An independently specified **112-byte** LE count/F64 vector checks field content/order. Four variants independently change exactly one source/mean/delta/kernel field in entry1 by0.5. Every other entry/field/reference stays fixed; byte prefixes and suffixes outside that field's eight-byte slot must remain identical. Each variant differs from baseline. Removing any individual field breaks the golden and/or its isolated mutation. |
| R3b: present velocity numbers | Reuses the existing single tiny parameter and valid CtMeta. Actual production `encode_state_labels` compares the same `[Some(label),None]` shape with Some(level3→2,same scale), then Some(same level3,scale_bits+1). Both labels are built through CtMeta::new. Vector lengths, option tags and every other state field stay identical. Exact prefix/suffix assertions isolate level bytes129..136 and scale bytes137..144; both variants differ from baseline. Omitting either present-label numeric field breaks its assertion. |

These are raw encoder-field tests, not authoritative schedule validation or parameter admission. The raw rate fields deliberately need not satisfy kernel=mean×delta. The preexisting valid full-suffix-below-unchanged-maximum, reference, structured321-byte golden, geometry and canonical tests remain unchanged. No production encoder was deliberately altered to manufacture a RED; the release selected a tests-only correction of currently correct encoders.

## Validation

All commands used locked/offline, current target, CARGO_INCREMENTAL=0,CARGO_BUILD_JOBS=8,RAYON_NUM_THREADS=8 and disk free>=805306368 bytes. All seven source/dependency pins were recorded before and after each command and matched.

| Raw log | Exact scope/outcome |
|---|---|
| `01-clippy.log` | `cargo clippy --locked --offline -p ckks-train-bench --lib --tests --no-deps -- -D warnings`: **PASS**, compilation/lint only. |
| `02-format.log` | `rustfmt --edition 2024 --check crates/ckks-train-bench/src/study_admission.rs`: **PASS**. |
| `03-filtered-green.log` | `cargo test --locked --offline -p ckks-train-bench --lib study_admission::tests::p3_digest_golden_and_mutations -- --test-threads=1 --nocapture`: **PASS1/1**,189 other unit tests filtered. Exactly once; no retry. |

The existing accounting fixture also still reports3 transcript+2256 metadata=2259 bytes. Runtime work was **one existing CkksParams::test_tiny constructor**, reused for all labels; proofs, descriptors, datasets/materializers, geometry/initial-state/update/handoff/wrapper, keys/Encoder/evaluator/ciphertext/HE and plaintext training were all **zero**. No hidden shared fixture initialized. P3-owner cumulative totals become params24/proofs11/descriptors15/Iris5/geometry527/initial527/updates243/handoffs79/wrappers14/HE0; separately owned P4 activity is not added to these counts.

## Unchanged inputs and frozen evidence

| Source/dependency | SHA256 |
|---|---|
| P3 integration tests | `571f4145239230ab124364a76d0fa16d3b940d5fa8b066a7194f0e1f1d846a0c` |
| CKKS bootstrap metadata | `e1bf9b3e7f856d0eb4a167874d05d6884b38d0054842f583b1add81c6e4eb2b8` |
| Root bench lib export | `867b2c6d8f4391861956c10483fe18c91636e9272e41cceee8953b90976b3e44` |
| Read-only P4 `study_plain.rs` | `d7a20981ec0c908f78a579f73fe68df4b92e2939827bf511d41a2168b2d85c9b` |
| Read-only P4 `study_plain/p4_fixtures.rs` | `9a09235dc09d08acb62f40359d8bf6c75cf4eec3400117b621b7d3099dd815b3` |
| Read-only P4 `study_reference.rs` | `fcb7f1d34fd8c87527af4adf59c0ebd919eeeb2a07f5f635a965a436c233c9af` |

`task-54-p3-fix2-validation/` retains all seven before/frozen source copies, command logs, before/frozen manifests, exact unified delta against fix1, and the unchanged-production scope proof. `final-manifest.txt` pins them and this report. All original/fix1 evidence and snapshots remain unchanged. P4 files were only read/copied as dependency inputs; no P4 execution evidence is claimed.

No broad11+5 runtime rerun, new production functionality, Git, network/SSH, dependency, dataset download, scientific JSON, HE or campaign action occurred. Independent rereview is still required; the scientific matrix, encrypted-block holds, resource/range/precision/security/target/start gates and real factory-spy requirement are unchanged.
