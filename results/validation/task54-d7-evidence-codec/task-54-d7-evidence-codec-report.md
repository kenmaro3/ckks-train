# Task54 D7 evidence/Quick codec source report

Status: **DONE_WITH_CONCERNS**. Author source/static package complete; runtime execution and `--list` count **0**. Ownership of the ten source paths below is released to root for independent full-diff SPEC/QUALITY review. No commit, push, publicdocs edit, server operation, protocol implementation, actual gate assembly or scientific output was performed.

Parent requirement read first: `task-54-brief.md`. Source release: `task-54-d7-evidence-codec-source-release.md`. Source BASE remains `1e65598b73a3ff2fd407daa362ae1beaa423e02c`; root concurrently advanced documentation-only HEAD to `d1bfb8718e738ca1bd4922e42aead6c2a8bcb7c6`. Proposal FIX1 SHA256 is the required `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b`. All proposal sections were treated as binding; no ratified byte was changed and no schema conflict required a root ruling.

## Changes and sharing

- Added private `document/evidence.rs` plus focused `evidence/{wire,auxiliary,tails,quick}.rs` and pure `evidence_tests.rs`. Five exact evidence domains, fixed policies, mandatory Full/Quick bodies, typed inputs/implementation/layout/resource/user tails, Request/Consumption/UserScope/UserCurrent/QuickIdentity, and all three Quick canonical/JSON operational envelopes are represented. Parsed output is inert and cannot construct a registered case, retained target, lease, live gate, continuation or execution authority.
- Reused the existing `execution_descriptors::Cursor` primitive reads (narrow visibility plus bounded helpers) and the existing document Encoder/lexical Scan/Fields/hex transport. Quick adds only its closed DocumentLayout path with raw A=2097152 and canonical K=1048000; Full/Reference/Lock/Admission/Start caps and encodings remain unchanged. All variable list counts and count/minimum-byte products are checked before reservation; resource count22 and Quick count2/argv256 are closed. The referenced approved implementation source-manifest reader's existing512 cap is not increased; this codec does not pretend a FileRef contains that manifest.
- Extended private `record!` declarations with optional private visibility so the same existing CaseIdentity, Gates, Gate and Predecessor records can be shared. Added bounded binary readers into the same CaseIdentity/Execution types; their output still uses original `component`/ExecutionFields canonical traversal. Link, Clock, Gates and Predecessor output reuses the original Encode implementations. Binary case-binding source revision is checked as exact Git40 for these new records; existing Full JSON behavior is unchanged.
- Extracted one `study_admission::encode_request_body` generic traversal and a borrowed `RequestBody` projection. Both original P3 `encode_request` and the new Request wrapper use it, including original limits, five work ceilings, signed LE i32 steps, transpose choice and eval flags. Requested partitions remain represented by the actual Consumption records, exactly as the preexisting P3 request body requires; no Debug serialization or reset limit was introduced.
- Added pure supplied-record helpers for request/consumption identity and lower-bound work/partition checks, complete sidecar cross-consistency, Full expected target/link claims, captured administrative scope/current/revocation/hold consistency, resource phase arithmetic and supplied-capacity arithmetic, and exact Quick identity/gates/context/argv/event links. These helpers identify mismatches among supplied bytes; none proves actual provenance, registry membership, retained canonical order, target origin, model completeness/residency, OS observations, lease freshness or root authority currentness.
- Quick argv is ordered lossless Unix Bytes: nonUTF8 and captured empty arguments are legal; NUL is rejected. JSON is exact closed-key transport, rejects decoded aliases/unknowns/wrong types/trailing tokens and escaped/nonlowercase/odd hex, verifies payload digest, retains exact input-file SHA256, and writes compact UTF8 plus one LF after length preflight. Only the three Quick event clocks have the shared-origin elapsed comparison; administrative UserCurrent Clock is not assumed to share that operation origin.

No Cargo manifest, dependency, license policy, scientific threshold, registry/profile/row count, Full gate, security requirement, baseline, hold or approval bypass changed. Preserve21 profiles, Full2680+20+7 and Quick268, registered10 seeds. Unowned `.vscode/` and both named MNIST result JSON files were not edited.

## Frozen inputs and source pins

Before source editing, froze137 committed oracle files, all78 old Full goldens and3 source contracts:218 paths. `frozen-input-verification.json` confirms218/218 unchanged. `baseline-source-sha256.txt` froze all77 existing Rust files before editing; final check found73 unchanged and only the4 owned existing files changed. The6 new evidence/test source files are separately pinned below. No generator or expected-vector regeneration was run.

Old Full admission/target goldens and the new Full evidence oracle have different synthetic identities/gate hashes. They remain separate exact-byte families. Old standalone predecessor component literals are reused only to test their unchanged component traversal; this is not a linked Full admission+evidence positive bundle. A coherent independently authored Full codec/protocol positive bundle remains a future oracle prerequisite.

| Owned source | Bytes | SHA256 |
|---|---:|---|
| `crates/ckks-train-bench/src/study_admission.rs` | 145973 | `04bf7081481b868956406fbbd9392937c21d96a7411509ee657e9cf42f5b0f25` |
| `crates/ckks-train-bench/src/study_reference/document.rs` | 64309 | `1562c0754660d715c5d09addddc905b6547d899c03c7d39a9db7e2b5257be59a` |
| `crates/ckks-train-bench/src/study_reference/document/admission.rs` | 32107 | `92fcaaaa1dd2b53a8542ed16591e86128956ac7bbded3d3bd71909528313d8ad` |
| `crates/ckks-train-bench/src/study_reference/document/execution_descriptors.rs` | 39756 | `100a462812c3b80f2f3207b3028af653369bd5b7d57db42bc10d46ab33043923` |
| `crates/ckks-train-bench/src/study_reference/document/evidence.rs` | 15901 | `4f727106cdbccd765a0cb5b3712fc81ec658efc170dddb86192e8ecabb83ae82` |
| `crates/ckks-train-bench/src/study_reference/document/evidence/wire.rs` | 7972 | `c5b922128b845171a8639c08fc2d14d7cc87d587bac795ae7a509b618614c29d` |
| `crates/ckks-train-bench/src/study_reference/document/evidence/auxiliary.rs` | 14266 | `3d39848107d2db079b97a1ab2c725c98ff7cc16bdd1df827908f0b1ee9ce4387` |
| `crates/ckks-train-bench/src/study_reference/document/evidence/tails.rs` | 22789 | `a591f3618d3e3c736a86dcea95bb8e5b8948180615c6a6a043e7f09d3f002590` |
| `crates/ckks-train-bench/src/study_reference/document/evidence/quick.rs` | 21021 | `f1ff6d5154c3446a5ef131add86b2a834fbf1b83f498acabf3227c648744772b` |
| `crates/ckks-train-bench/src/study_reference/document/evidence_tests.rs` | 31623 | `79dce199230fc42522636ca66a0675c73b0becd7ebba74d333aedf5fed65da7d` |

Validation artifacts are under `evidence-codec-source-validation/`:

- `frozen-input-sha256.txt`: `e8a29c3bb36a05843b4d29a5b7439ec7b96a8c27eff3d4e93bed9dc6353d9f5a`.
- `baseline-source-sha256.txt`: `789c402026ac8e368ad9a670a13ba85392cb0df62a0b2099f4bf115516fa129b`.
- `final-source-sha256.json`: `a26ceb1e6a67f430e8510f3e30c68b7db105902b375bb800c3e047d2fbb8f7d5`.
- `final-command-results.json`: `511a5f28a96417513610d41fbf1b9579b55e93882258d99b1994b0c453cd3b4c`.
- `frozen-input-verification.json`: `cd1a92c27db97c27e05db1c2f514c6890c103d627de262b393a880b8ef29824a`.
- `baseline-source-verification.json`: `37b806398f9aabcbe6a82b5673b318974f92211dd3b0d6632bbb524332f43f2d`.
- `proposed-runtime-inventory.json`: `816825ac662a6f8a8638b2df8bc774a36be0f2f0cbf52aef385c5ba25b2366ac`.
- `proposed-shared-regressions.json`: `4abbfaa1125eddfee4d48c072143ece93bb320fd87f31df4673a7105d0533053`.

## Actual RED/GREEN/static history

All commands were local. Every `cargo test` invocation used `--no-run`; none invoked tests or `--list`. Final authoritative commands, all on the final pinned source:

| Exact command | Exit | Seconds | Log |
|---|---:|---:|---|
| `cargo clippy -p ckks-train-bench --all-targets --all-features -- -D warnings` | 0 | 2.924 | `clippy-complete-final.log` |
| `cargo test -p ckks-train-bench --lib d7_evidence --no-run` | 0 | 10.364 | `compile-default-complete-final.log` |
| `cargo test -p ckks-train-bench --lib --features parallel d7_evidence --no-run` | 0 | 10.382 | `compile-parallel-complete-final.log` |
| `cargo test -p ckks-train-bench --lib --features cuda d7_evidence --no-run` | 0 | 10.264 | `compile-cuda-complete-final.log` |
| `cargo fmt --all -- --check` | 0 | 1.041 | `fmt-complete-final.log` |
| `git diff --check` | 0 | 0.04 | `diff-complete-final.log` |

Earlier meaningful history (all retained unless explicitly identified):

1. Initial `cargo test -p ckks-train-bench --lib d7_evidence --no-run` produced missing-module E0432. Its shell wrapper then failed exit1 because zsh `status` is read-only. This was an author wrapper mistake; no source was written before the missing-API RED. The corrected identical Cargo command recorded actual compiler exit101/E0432 in `compile-red.log` before implementation.
2. Same default no-run command: `compile-green-attempt1.log`, exit101 (new private visibility/imports and missing test helper); `compile-green-attempt2.log`, exit101 (private IdentityRef equality helper). These original failed compiler outputs remain intact; wrapper exit101 was observed by the tool, although these first two files do not append an explicit exit line.
3. The next default no-run succeeded exit0 with two private-interface warnings. **Its saved filename was later reused and its original file content is lost.** The tool output retains that actual exit0 and warnings; no reconstructed file is presented as original evidence. The later run at that filename failed exit101 for an overconstrained re-decode lifetime/unused import. That actual failed file is preserved as `compile-default-intermediate-failed.log`. This audit loss is an explicit concern.
4. Strict all-target/all-feature Clippy: `clippy-all-features.log` exit101 (lifetime/unused import/index); `clippy-all-features-attempt2.log` exit101 (unused index, needless borrow/lifetime); `clippy-all-features-attempt3.log` exit0. Corrective implementation changes resolved these diagnostics.
5. With additional closed-branch/boundary tests, `clippy-all-features-final.log` exit0, and `compile-default-final.log`, `compile-parallel-final.log`, `compile-cuda-final.log` all exit0. These were not runtime passes.
6. Added the missing explicit source/context, argv0/order, coherently rehashed request-work/max-record/physical-lane and other-attempt negatives; `clippy-all-features-final-with-context.log` exit0 covers the context additions before the final other-attempt addition. The final6-command table above then rechecked the final source. Those justified repeats are the last checks; no further broad repeat is planned.

Actual Cargo build/check totals:12 no-run Cargo test invocations (including the first wrapper failure and the overwritten successful log),6 strict Clippy invocations. Formatting was applied during source work; final fmt/diff checks are separately recorded above. These totals are author command history, not test execution counts.

## Proposed exact runtime release and complete inventory

These normal meaningful tests are compiled but **unexecuted**. Root must first review the complete owned source diff and this inventory. A bounded subsequent release can use each exact filter individually with:

```sh
cargo test -p ckks-train-bench --lib <EXACT_FILTER> -- --exact --test-threads=1 --nocapture
```

This report does not execute or release that command. The16 new exact filters:

```text
study_reference::document::evidence::tests::d7_evidence_independent_all_ten_branches
study_reference::document::evidence::tests::d7_evidence_auxiliary_and_quick_exact_json
study_reference::document::evidence::tests::d7_evidence_concrete_representation_mutations
study_reference::document::evidence::tests::d7_evidence_full_quick_exact_claims_and_nonpromotion
study_reference::document::evidence::tests::d7_evidence_quick_event_links_policies_argv_and_context
study_reference::document::evidence::tests::d7_evidence_strict_json_and_caps
study_reference::document::evidence::tests::d7_evidence_request_consumption_retained_comparisons
study_reference::document::evidence::tests::d7_evidence_authority_resource_and_closed_branches
study_reference::document::evidence::tests::d7_evidence_all_literals_closed_tags_and_trailing
study_reference::document::evidence::tests::d7_evidence_target_expected_claims_and_file_identity
study_reference::document::evidence::tests::d7_evidence_implementation_ni_mandatory_and_missing
study_reference::document::evidence::tests::d7_evidence_quick_failed_outcomes_and_binary_argv
study_reference::document::evidence::tests::d7_evidence_request_shared_p3_traversal_and_all_original_limits
study_reference::document::evidence::tests::d7_evidence_production_mixed_nested_domains_tail_and_length_refusals
study_reference::document::evidence::tests::d7_evidence_shared_predecessors_and_clock_claims
study_reference::document::evidence::quick::tests::d7_evidence_all_gate_policy_and_hash_permutations
```

| New exact test suffix | Fixture reads estimate | Top-level parse calls upper estimate | Inputs and covered branches |
|---|---:|---:|---|
| `d7_evidence_independent_all_ten_branches` | 10 | 30 | ten Full/Quick evidence .bin files; no expected-value generation |
| `d7_evidence_auxiliary_and_quick_exact_json` | 15 | 24 | eight Request/Consumption/UserScope/UserCurrent .bin files; Quick identity; three Quick .json and three corresponding .bin files |
| `d7_evidence_concrete_representation_mutations` | 9 | 9 | nine concrete negative .bin files (unknown-tag/synthetic-bit/identity/input-receipt/partition/resource-peak/Quick seed/count) |
| `d7_evidence_full_quick_exact_claims_and_nonpromotion` | 22 | 22 | all ten evidence .bin files, each opposite-scope replacement; Quick identity and Quick inputs again for rehashed other-attempt negative |
| `d7_evidence_quick_event_links_policies_argv_and_context` | 20 | 18 | three event .json files, five Quick evidence files, captured-argv.json, four concrete event/argv negatives, end-incomplete.bin, preflight.bin, five literal-inputs.json offset lookups |
| `d7_evidence_strict_json_and_caps` | 2 | 10 | Quick start.json and Quick inputs.bin; small JSON aliases/types/trailing mutations and length-only A/K witnesses |
| `d7_evidence_request_consumption_retained_comparisons` | 5 | 20 | Quick request/consumption plus request/order/full-consumption concrete negatives; three locally rehashed request work/max_records/physical-lane mutations |
| `d7_evidence_authority_resource_and_closed_branches` | 5 | 7 | Quick authority scope/current; two concrete authority current negatives; Quick resource evidence; typed arithmetic/claim/hold mutations |
| `d7_evidence_all_literals_closed_tags_and_trailing` | 26 | 1487 | literal-inputs.json once and all25 .bin files;25 literal positives,302 closed-tag mutations,1135 field-boundary truncations,25 trailing mutations |
| `d7_evidence_target_expected_claims_and_file_identity` | 5 | 4 | Full inputs; target-link/threshold negatives; Quick inputs; first independently named retained source fixture bytes |
| `d7_evidence_implementation_ni_mandatory_and_missing` | 1 | 3 | Full implementation evidence only; typed I and NI mandatory-coverage/missing-coverage mutations; no CKKS objects |
| `d7_evidence_quick_failed_outcomes_and_binary_argv` | 6 | 88 | Quick end/preflight .bin; four literal-inputs.json offset reads;84 failed phase/code/update combinations, two invalid outcomes, NUL and empty argv mutations |
| `d7_evidence_request_shared_p3_traversal_and_all_original_limits` | 3 | 5 | Quick request.bin plus two literal-inputs.json offset reads; eight DiagonalRequest/PlanningBudget metadata objects; dedicated/reuse signed-i32 and physical-None branches |
| `d7_evidence_production_mixed_nested_domains_tail_and_length_refusals` | 3 | 6 | Quick inputs.bin, branch-tag negative and one literal-inputs.json offset read; production/test nested-domain and domain/tail mutations; u64 length and A/K arithmetic witnesses |
| `d7_evidence_shared_predecessors_and_clock_claims` | 0 | 3 | three compile-included old standalone predecessor .hex literals only; three small Clock supplied-claim comparisons; no linked Full family assertion |
| `d7_evidence_all_gate_policy_and_hash_permutations` | 6 | 244 | Quick preflight.bin and five Quick sidecars; all119 nonidentity policies and119 nonidentity hash permutations; policies=>UnsupportedSchema; supplied-sidecar hash mismatch=>CrossLinkMismatch |

The static inventory is `proposed-runtime-inventory.json`:138 fixture-read calls estimated,13 metadata reads of278638B (3622294B total), total fixture read bytes conservatively<8MiB. Largest binary7138B, largest event JSON7089B; mutated transport witnesses<16KiB. Ordinary component lists22, Quick updates2, fixed permutation arrays120. Field-boundary truncations1135 and closed-tag corruptions302 derive from the unchanged literal metadata; all119 nonidentity gate-policy permutations reject UnsupportedSchema and119 hash permutations reject CrossLinkMismatch at the same pure supplied-sidecar helper used by event checks.84 failure combinations cover4 phases×7 codes×completed0..2. NI coverage is typed pure representation mutation, not a newly independent NI byte oracle or accepted NI evidence.

Counts in the JSON/report are static estimates, not measured timing/allocation telemetry. Top-level counts exclude primitive field reads and nested validation re-parses; conservative complete record-parse estimate across the16 filters is<16384. Only small byte/string/vector copies and serde_json metadata trees are allocated. No boundary test allocates A/C/R/K-sized storage; limits/overflow use arithmetic/small witnesses. serde_json tree allocator capacity, allocator overhead and harness RSS are unmeasured. This inventory makes no physical memory-fit claim. Eight DiagonalRequest/PlanningBudget/Arc metadata objects exercise retained request projection; actual planner/geometry/initial/update/handoff/wrapper calls remain0.

Across every proposed new test: registry/materializer/parameters/P1/RNG/NTT/HE/native/Host/CUDA/SSH/observer/actual-gate/publication/attempt-mutation/external-process calls0. Inputs are frozen fixture files and small literal/typed mutations only. Expected semantic refusals assert stable DocumentErrorCode, and coherent work-limit mutations additionally assert the semantic field path. No fake gate assembler or claimed retained evidence source is constructed.

The16 affected shared-regression filters (also unexecuted) are inventoried separately in `proposed-shared-regressions.json`:

```text
study_reference::document::admission::tests::d7_admission_frozen_documents_complete_bytes_and_hashes
study_reference::document::admission::tests::d7_admission_six_frozen_branches_and_domain_components
study_reference::document::admission::tests::d7_admission_closed_layout_arithmetic_and_small_lexical_boundaries
study_reference::document::tests::independent_python_exact_canonical_bytes_cover_all_domains_and_timing_options
study_reference::document::tests::fix1_genuine_structural_and_escaped_keys_keep_every_string_exception
study_reference::document::tests::fix1_genuine_argv_array_elements_keep_os_byte_exception
study_reference::document::execution_descriptor_tests::execution_parameter_literal_bytes
study_reference::document::execution_descriptor_tests::execution_layout_literal_bytes
study_reference::document::execution_descriptor_tests::execution_key_literal_bytes
study_reference::document::execution_descriptor_tests::execution_security_literal_bytes
study_reference::document::execution_descriptor_tests::execution_complete_negative_images_distinguish_wire_and_binding
study_reference::document::execution_descriptor_tests::execution_every_proper_prefix_rejected
study_reference::document::execution_descriptor_tests::execution_checked_size_and_arithmetic_guards
study_reference::document::execution_binding_tests::d7_outer_six_frozen_independent_vectors
study_admission::tests::d7_transcript_failed_prefix_and_counting
study_admission::tests::p3_digest_golden_and_mutations
```

All selected shared tests are pure codec/metadata tests, not the broader study_admission fixture group that initializes CKKS/P1. They cover original Full bytes, shared lexical exceptions, original descriptor Cursor parsing, frozen outer binding and original Transcript accounting/primitives. Only the outer-six-vectors test performs runtime fixture reads (13); others use compile-included literals/small constructed witnesses. No test here constructs actual HE/native/context owners. Root can review/release these exact filters without an unbounded broad test filter.

## Self-review and outstanding dependencies

- Canonical typed encoding is compared against unchanged independent `.bin`/original exact `.json` inputs, not just a self-roundtrip. Fixture manifests/generator/expected leaves were not rewritten. Production has only Mode::Production; Oracle variant and decoding entry points exist under cfg(test) only. No runtime synthetic switch or caller pass flag exists.
- All binary records are closed and bounded. Frozen case/binding/Clock/Link/Gates/Predecessor traversal is shared. Requests retain all original limits; payload/file hashes remain distinct, including whitespace-sensitive exact JSON file links. Full required target fields cannot disappear into an optional Quick schema, and Quick cannot pass Full expected-claim helpers or mixed-scope bundle comparison.
- Resource22-component completeness/order, zero-mask byte consistency, resident+additional phase maxima, incremental-only maxima and checked arithmetic are enforced as representation identities. These cannot prove model completeness, allocation inventory, actual residency/accounting credit, host/cgroup observations, stale-capacity/competing-claim freshness or capacity reservation. Missing model/validation/applicability must remain missing in the genuine gate. CPU-only resource claims do not relabel CUDA/external observations.
- User selected tuple, exact scope/authority/current identity, generation/head, revoked-all/index ordering/range/selection, holds, Initial/Retry predecessor and cross-resource overlap are checked as supplied records. Actual root-origin source control, latest received task provenance and the trusted active `authority.current.bin` update/reload/race mechanism remain absent. Re-reading a stale or caller-authored capture is not currentness. **Frozen CaseIdentity carries no actual layer/source graph**: this codec does not invent a placement/profile-string classifier for the global Full encrypted-block hold. The genuine reader must classify the retained canonical profile/source independently and enforce that unchanged hold even when a malformed administrative capture omits it. No user confirmation is requested or fabricated.
- Actual expected source inventory, complete contract/selected receipt bytes and canonical order, descriptor/cryptographic declarations, complete approved Train manifest/current revision and512-source-role cap, reviewed KAT/NI/security/model report parsers and applicability implications, attack/export completeness and the existing8MiB role-specific evidence read bound remain reader obligations. A hash or report path in this codec proves none of them. Pure exact supplied-claim comparisons do not upgrade them to verified evidence.
- Filesystem safe roots/held descriptors/nofollow/owner/mount/regular-file/read stamps, guarded eligibility, all five no-clobber synchronized evidence finals, custody transfer and durable admission/preflight/start/end publication remain unimplemented. C01–C10 fault inventory belongs before that protocol source; this package implements no protocol or mutation. Partial/recovered records never yield live authority. Full new/old coherent positive linkage needs an independent future oracle as noted above.
- Prior Full C1 remains unchanged: a real Retry reader must verify BOTH current and prior admissions against the SAME retained target/reference/approved policy before predecessor/start helpers. This package neither waives nor claims that composition. Missing Full target origin, zero threshold without verified origin, unknown source inventory, actual stale OS/claim/authority sources, current leases/context/process identity, before-allocation construction obligations and actual runtime completion all remain genuine live tests/readers, not synthetic pass flags.
- Gate capabilities/live constructors, begin/dispatch lifetime integration, context/source approvals, native Linux publication/fault evidence, actual resource fit, full experiment runner/table publication and scientific outputs remain future dependencies. Local compilation is not any native/GPU/noninteractive performance, security adoption, scientific validity or full campaign authorization claim. Existing evidence maps and historical measurements are not erased by these missing prospective readers.

Author work ends at this pinned source/static report. Independent review, root-selected runtime outcomes and any later integration/commit are separate steps.


## FIX1 — Important I1/I2 correction and added semantic negatives

Status: **DONE_WITH_CONCERNS — author FIX1 complete, independent re-review/runtime pending**. Read the entire independent source review, including both full Important findings: `task-54-d7-evidence-codec-source-review.md`, SHA256 `131f04eb5366f8dfa8446b414ac644855d1a831e65b422df05795505d5f2476d`. Root resumed sole source/Cargo ownership for this bounded fix; no new production defect or feature was asserted. Main at assignment was documentation/linked-fixture-only `15cd48b`; this fix does not integrate that separate linked oracle.

The original 25484-byte report prefix is preserved exactly (SHA256 `9a200eb062a58629ccc4bdf62fc4c78faf8670298adb53dfd3c0501db796631c`). Its original statements and inventory files are historical and explicitly superseded where corrected below. All26 earlier validation files remain byte-identical against `evidence-codec-before-fix1/`; that root-owned snapshot and original review were not overwritten. All218 frozen inputs remain unchanged. No oracle generation or expected-value regeneration occurred.

### I1: corrected shared-regression constructor inventory

The original report's assertion that all16 selected shared regressions are parameter-free was **incorrect**. In particular, original `proposed-shared-regressions.json` zero aggregate flag and the `p3_digest_golden_and_mutations` row must not be used to release that test as constructor-free. The16 new evidence test filters remain constructor-free;15 of the16 selected shared filters remain parameter-free on the scoped inspection. The necessary sixteenth filter is retained, unchanged, and proposed for a **separate bounded exact release**, not dropped:

```text
study_admission::tests::p3_digest_golden_and_mutations
```

The authoritative corrected JSON versions are now `fix1/proposed-shared-regressions-fix1.json` (aggregate zero flag false;15 parameter-free shared plus this1 constructor-bearing filter), `fix1/proposed-runtime-inventory-fix1.json` (same16 new filter names and their updated cases), and `fix1/p3-tiny-transitive-inventory.json` (separate P3 cost/release inventory). Original JSON files remain untouched as audit evidence, not current release inputs. Root's uninvoked32-filter controller must consume this corrected classification before any release.

Independently inspected the actual unchanged test body through its end, `CkksParams::test_tiny/new`, `RnsBasis::new`, `NttTable::new`/power-table/root helpers, deterministic prime search, Modulus/Shoup, CtMeta and scalar OpCounter, plus relevant state/matrix/update/handoff retention and error helpers. The corrected static successful-path constructor/count inventory is:

| Work / values | Count or fixed shape |
|---|---|
| `CkksParams::test_tiny` / `CkksParams::new` |1 /1, insecure toy N=32, scale bits42, dnum12 |
| q and p primes |q12: `[54,42,34,34,34,34,58,58,56,56,56,56]`; p1: `[60]` |
| Distinct deterministic prime pools |6: `(34,4),(42,1),(54,1),(56,4),(58,2),(60,1)`;13 selected primes total |
| RnsBasis / Modulus / NttTable constructors |2 bases (q12/p1),13 Modulus,13 NttTable |
| Twiddle table allocation shape |26 vectors×32=832 Shoup entries; each Shoup has2 u64 fields,13312B value-field payload |
| Additional scalar Shoup fields |26 (n_inv/w_n_inv),416B value-field payload; not counted in the832 vector entries |
| Root/table preparation |13 minimal-root searches,26 power-table fills;31 deterministic odd-power steps per successful root search,403 total after candidate selection |
| CtMeta::new |3 valid stack/value metadata constructions (initial label, changed level, changed scale); no heap allocation on these successful constructor paths |
| Update metadata retention |1 successful update summary;2 StateLabels retains;2 matrix copies;15 retained vectors including6 CtMeta state vectors/2 velocity vectors/1 micro vector/2 matrix outer vectors/4 matrix rows |
| Copy counts |16 successful `copy_vec` calls total:2 two-element MicroPart copies,13 inside update retention,1 retained handoff-output copy |
| Handoff retention |2 helper calls;1 succeeds,1 rejects accounting before any output copy |
| Other fixed metadata |Local hidden+canonical:2 outer vectors of2 rows,4 rows×2 CtMeta. Local and retained handoff outputs length2 each;1 boxed PlanError with1 trace record and1 attempt record plus3 bounded String fields;3 index reservations |
| Encoded helper buffers |8: five fixed112B rate encodings and three fixed211B state-label encodings. The helper uses default ReceiptLimits; the original blanket “configured buffers<=4096B” must not be applied to that helper. Fixed witness lengths remain small. |
| Actual NTT forward/inverse / RNG / HE |0 /0 /0 /0 |
| P1/DFT/bootstrap metadata, keys/Encoder/Evaluator/ciphertexts, registry/materializer/planner/stage update/actual handoff/native/Host/CUDA/SSH |all0 |

Prime candidate primality checks and initial root candidate iterations are deterministic but data-dependent and **not measured or claimed exactly**. Each of the13 table constructors additionally rechecks primality. RnsBasis holds12/1 Modulus slots and12/1 table slots in two bases/two Arc allocations with initially empty Galois caches; no permutation is materialized. The corrected JSON pins every inspected transitive source. Vector lengths, stored field-payload bytes and static constructor calls do not establish Vec capacity, allocator overhead, heap-call counts, runtime or RSS. Actual parameter-constructor runtime execution is still0 in this author turn.

The retained accounting payload is recorded symbolically, using the actual test expression:

```text
expected_update = 2*(6*sizeof(CtMeta)+2*sizeof(Option<CtMeta>))
                + 2*sizeof(MicroMetadata)
                + 2*(2*sizeof(Vec<CtMeta>)+4*sizeof(CtMeta))
charged_cap = 3 + 2*sizeof(TransitionIndex) + expected_update
                + 2*sizeof(CtMeta) + 2*sizeof(MicroPart)
```

No exact platform sizeof/RSS measurement is invented. The separately proposed release is this exact filter, one invocation, `--exact --test-threads=1`; root must approve the corrected fixed-size constructor inventory and choose the process bound before running it. No automatic retry or broader test filter is proposed. This test is unexecuted and not described as passed.

### I2: existing pure checks now have the missing meaningful negatives

Only `crates/ckks-train-bench/src/study_reference/document/evidence_tests.rs` changed against root's before-fix1 source snapshot. No production module, Cargo file or existing P3 test was changed. Existing16 new exact filter names are retained; additions live inside two existing tests:

| Exact test suffix | Added case / expected intended refusal |
|---|---|
| `d7_evidence_authority_resource_and_closed_branches` |Both legal overlap mismatches: selected UserScope1/resource0 and selected UserScope0/resource1. Each changed sidecar parses successfully on its own; bundled supplied claims reject `CrossLinkMismatch` at `evidence.bundle.overlap`. |
| same |Full evidence predecessor None versus selected authority Retry1 with a syntactically valid fictional predecessor. Rebound UserScope/current/file identities are coherent; parse rejects `CrossLinkMismatch` at `user.retry.predecessor`. |
| same |Quick UserScope Retry1 carrying that same validly encoded predecessor. Scope header, Bool, registration, prior ID, Link and overlap are legal; rejects `CrossLinkMismatch` at `authority.quick.retry`. This is syntax/representation only, never a verified actual prior admission. |
| `d7_evidence_quick_failed_outcomes_and_binary_argv` |Failed outcome with failure present and legal completed count, then phase4/code0 or phase0/code7. Original link fields remain intact; both reject `UnsupportedTag` at existing shared tag path `execution.tag`. No JSON digest is bypassed: these test the canonical binary codec, which has no outer JSON envelope. |

The test-only rebind helper serializes changed authority bytes, computes their SHA256, updates UserCurrent.authority_sha256, serializes/hashes Current, then updates both embedded Bytes/digests and both FileRef lengths/hashes before encoding the User evidence. It uses existing canonical traversals to prepare inert negative inputs and then calls the actual parser/consistency helper under test. No stale enclosing digest can mask the intended overlap/predecessor checks. It creates no gate, origin-current flag, source loader, filesystem mutation, HE object or new oracle. Full and Quick fixture families are still not spliced into a live/linked Full positive.

New16-filter inventory delta: authority/resource filter fixture reads5→11 (five Quick sidecars29939B plus Full user3252B added), top-level parse-call upper estimate7→32; failed-outcome filter parse estimate88→90. Total fixture-read calls138→144; added literal bytes33191; metadata offset reads stay13×278638B. Ordinary generated transport remains<16KiB; all other fixed small bounds and the conservative<8MiB cumulative fixture-read envelope remain unchanged. Allocation/parse counts remain static estimates, not measured telemetry. All16 new filters still have parameters/P1/RNG/NTT-table/forward/inverse/HE/native/Host/CUDA/SSH/registry/materializer/actual-gate/protocol-publication calls0. The corrected P3 shared exception above is explicitly outside that zero-constructor claim.

### FIX1 compile/static outcome, pins and ownership release

No compile failure occurred in FIX1; no tests or `--list` were run. Each log was opened with exclusive creation under a new `fix1/` path; earlier failed/successful validation logs were preserved. Only test bodies changed, without feature-specific cfg or any production-source change, so default no-run plus strict all-target/all-feature Clippy were the relevant compile checks; unchanged broad/native/Linux checks were not added or repeated.

| Exact command | Exit | Seconds | Unique log |
|---|---:|---:|---|
| `cargo test -p ckks-train-bench --lib d7_evidence --no-run` | 0 | 13.678 | `fix1/01-default-no-run.log` |
| `cargo clippy -p ckks-train-bench --all-targets --all-features -- -D warnings` | 0 | 2.712 | `fix1/02-strict-clippy.log` |
| `cargo fmt --all -- --check` | 0 | 1.029 | `fix1/03-fmt-check.log` |
| `git diff --check` | 0 | 0.012 | `fix1/04-diff-check.log` |

Changed source final pin: `crates/ckks-train-bench/src/study_reference/document/evidence_tests.rs`, 36987B, SHA256 `430a84b108899f027feed6f4edee80762f3460b7c541954e251416960d678ac2`. All other9 owned source files exactly match before-fix1; all10 current final pins are in the new versioned manifest. Author FIX1 runtime/test-list counts0/0.

| New authoritative artifact (within evidence-codec-source-validation) | SHA256 |
|---|---|
| `fix1/final-source-sha256-fix1.json` | `87cb90f1fda25a52c83c7f95f628dab483d0debb128b9394c6c8c8d8758f89a9` |
| `fix1/proposed-runtime-inventory-fix1.json` | `b26f8fca72e5d72507de2195e16058158f70e37801cf00e87f1fb4e1a5f69fe5` |
| `fix1/proposed-shared-regressions-fix1.json` | `a847ebc467fe5b1915d3bf85529c170da0cd2d4333ed4b4af52ec46f9fadbb65` |
| `fix1/p3-tiny-transitive-inventory.json` | `274e7f9d88bd5393f3a6df77dc6d74da2a523be58292688323d774e5c3304471` |
| `fix1/preservation-fix1.json` | `2460119bb4f230912d3bcb881aec177a8afad336eb329d107f64cc5431be9501` |
| `fix1/command-results.json` | `d486227d366511f914b60dd080ed7047bead916fa99f3b5dc3beee39668dbd81` |
| `fix1/01-default-no-run.log` | `abd769c3cf57529e2d9e3b4d33a6a89b015df2d1e607f84549ef8168b297f20e` |
| `fix1/02-strict-clippy.log` | `f28dc274b41da2f0326907d7b2e54e5643638332792411986488e85baf301d86` |
| `fix1/03-fmt-check.log` | `3c696c094e2a1323fa00cf0974a861af0cd6824b1672248c1d12791a70a61e4e` |
| `fix1/04-diff-check.log` | `3c696c094e2a1323fa00cf0974a861af0cd6824b1672248c1d12791a70a61e4e` |

Remaining actual readers/target/currentness/global-hold/resource/protocol/gate/native/scientific obligations and earlier disclosed lost intermediate compile log remain as recorded in the preserved prefix. I1/I2 are addressed in this author package; independent scoped re-review must decide acceptance and any exact runtime release. Source ownership is released back to root at these pins.
