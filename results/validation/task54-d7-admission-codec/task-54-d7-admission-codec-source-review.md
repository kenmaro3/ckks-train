# Task54 D7 Full admission/start codec independent source review

Reviewer: independent SPEC/QUALITY review of the released private wire-codec package. `task-54-brief.md` was read first, followed by the source release, full author report, complete five-file diff, actual source and relevant unchanged helpers. Review is against `docs/study-target-design.md` §§3–5/7–8 and the explicitly limited source release. No Cargo, test execution, `--list`, HE/native work, SSH, source edits or subagents were used. Only this review artifact was written.

## Verdicts

**SPEC: PASS for the released inert Full wire-codec dependency.** Exact domains, canonical field order, fixed tags/options, caps and strict inherited parser are preserved. The package does not implement or authorize live admission/start. Future reader/protocol obligations below remain mandatory.

**QUALITY: PASS for source, with the bounded runtime release correction Q1 below.** No blocking source defect found in this scope. Static evidence supports compilation, not runtime correctness. The nine new exact filters are accepted as proposed; add four specifically touched shared regressions to the next separately released runtime inventory. No blanket suite or prior 17-filter rerun is justified.

**Runtime result: NOT EXECUTED.** This review accepts a scope to be released by root; it reports no test pass or measured allocation/performance result.

## Evidence identity

- Source BASE: `99b8e88c50af791b89975e3e781f710f5d59ce52`.
- Full author report SHA256: `52a5e8e409c19db8b2986ddc9694748f35d4323b4b91f5fbb9261ac3fe75a9a6` (16,921 bytes), independently rehashed.
- Complete source diff SHA256: `76c088bb90f8e04910c33b1c4083dfa353fd4b06ff9f6c0c48e6b204aa5f30e1` (77,263 bytes), independently rehashed; includes both new files.
- All five current source hashes match `admission-codec-source-validation/final-source-pins.json`.
- All 78 current frozen fixture hashes match the expected hashes in `fixture-verification.json`. All 50 unique compile-time includes in `admission_tests.rs` match the input inventory names and byte sizes. Expected positive bytes remain frozen independently of the new Rust encoder.
- Inspected supplied command records and logs: genuine missing-module/layout RED exit101; final default, parallel and CUDA library `--no-run` exits0; strict library/tests/all-features Clippy exit0; fmt and diff checks exit0. These are author-produced static observations, not commands rerun by this reviewer. Logs show compilation rather than test execution.

## Source findings and contract checks

No critical/high/medium source finding within the released inert-codec scope.

The Admission/Start implementations encode the duplicated schema header and complete body in frozen table order; writer, kind, status, synthetic marker, integrity encoding/hash, enums and mandatory options are checked. Admission recomputes registered-case, execution-binding and Gates hashes in their separate domains, then the full payload digest. Start standalone parsing validates syntax/integrity and selector constraints without claiming complete verification. Production rejects test domains and synthetic markers; test wrappers remain private and `cfg(test)` only. Case tags reuse the actual registry vocabulary, and both actual and parsed CaseIdentity now share one traversal. Execution uses the existing `ExecutionFields` traversal and binding domain. There is no source-derived or test-derived constructor for a live gate/token, RegisteredCase, approved context or verified target.

Admission remains file/canonical 70,492,488/36,294,820 bytes; Start remains 2,097,152/2,097,152. C=2,613,526, R=31,584,142 and A=2,097,152 are unchanged. Only the two exact Admission object paths receive C/R hex exemptions; Start receives none. Existing Reference and Lock paths/caps are preserved. File size, UTF8, lexical depth/token/path checks and outer-byte accounting precede typed body decoding. The preflight uses bounded small lexical scratch, not an allocation-free scan. Blob decode verifies length/cap/lowercase hex before fallible reserve; canonical append checks logical total/outer limits before fallible reserve. Fixed list visitors refuse excess items before decoding/pushing those items. Unknown/missing/duplicate decoded keys and escaped aliases use the inherited strict Fields machinery. Ordinary serde/string/list allocations remain inherited; recoverable global OOM and measured RSS are not established by this change.

Policy/Link/Clock extraction retains their exact field order. Target policy validation and verified-source verification preserve their prior conditions. The refactored completion projection receives a parsed reference whose nonempty endpoint invariant is established by the existing parser (and explicitly by the private synthetic helper); no new unchecked public reference constructor is introduced. New target-claim checking compares full contract bytes, reference projections and exact supplied reference file/payload links. Writer-capsule approval is explicitly unavailable from these outer claims and remains required later.

Start/admission consistency compares all duplicated selector/input/attempt/policy/lock/reference/admission links and ordered wall times. Retry checks explicit predecessor admission/start file and payload links, distinct attempt, same campaign/profile/case/seed, complete contract/receipt and materialized identity, and relevant time order. Equality timestamps remain permitted. Large identity comparisons borrow buffers; small policy/link/projection comparisons re-encode bounded components.

### Q1 — low severity: runtime inventory must include the touched actual shared encoder

The nine new fixture-only tests exercise the new parsed CaseIdentity path but do not invoke the changed `registered_case_canonical(RegisteredCase)` adapter. Existing Reference/Lock encoders also share the changed fallible append/hex helpers. Before claiming the runtime regression gate complete, add the four exact existing filters below and inventory their distinct inputs/work. This is a release/evidence correction, not an observed failed test or a demand for unrelated suites. In particular, do not describe all thirteen filters as registry-free or runtime-filesystem-free.

### C1 — required future integration boundary, not current live authority

`check_predecessor_claims` (`admission.rs:682`) does not itself compare current/prior lock, reference or policy links. A rehashed prior admission with different target-link claims can satisfy that helper's stated same-case/input/predecessor checks. Consequently, a future Retry reader must validate **both current and prior admissions against the same retained target/reference and approved policy**, as target design §6 requires. Calling only current-target checking plus predecessor checking is insufficient. The current author report accurately states the helper's subset, the source calls it pure claims consistency, and no live reader/token is present, so this is not treated as an authority escape or a blocker for the disconnected codec package. Preserve this obligation explicitly in the next live-reader release and its changed-lock rejection tests. Likewise, `check_start_claims` must be composed with target/reference and live verification; parsing or crosslinks alone never verify start authority.

## Accepted new-test call/input/allocation inventory

All nine filters are pure codec/claims work. Source inspection found zero registry loads, input materialization, RNG creation (including nested RNG), parameter constructors, NTT/RNS tables, P1 geometry proofs, HE/key/encryption/bootstrap calls, native/CUDA contexts, host inspection, environment reads, subprocesses, runtime filesystem reads or store operations. Enum parsing and EvalConfig values are transport records, not runtime constructors.

The 50 compile-time files are small frozen JSON/hex/manifest inputs; maximum included file is 11,231 bytes. Work includes bounded preflight/Fields/record parsing, ordinary serde Values/clones/serialization for mutations, canonical Vecs and unhex vectors, pointer/key strings, streaming SHA256 and hash strings. Target/reference helpers use a strict synthetic parsed reference and never produce VerifiedReference. The largest listed generated oversize-list JSON is the 513-entry manifest witness, approximately 60,125 compact bytes by static arithmetic; the other listed lengths are 8,885, 24,525 and 9,707. Seven malformed initial JSON strings coexist in one test loop. The lexical probes use small 8,194-character hex strings, depth17 and a 24,582-byte escaped ordinary string. Arithmetic and private counter probes test C/R/A/total limits without creating boundary-sized blobs. Rehashed negative helpers are not positive oracle generators. No allocator/RSS bound is measured.

Accepted default-feature library filters, each exact and once in the later release:

```text
study_reference::document::admission::tests::d7_admission_frozen_documents_complete_bytes_and_hashes
study_reference::document::admission::tests::d7_admission_six_frozen_branches_and_domain_components
study_reference::document::admission::tests::d7_admission_target_start_and_explicit_predecessor_claim_links
study_reference::document::admission::tests::d7_admission_required_nested_fields_unknowns_types_and_link_mutations
study_reference::document::admission::tests::d7_admission_closed_layout_arithmetic_and_small_lexical_boundaries
study_reference::document::admission::tests::d7_admission_rehashed_cross_record_fields_and_clock_ordering
study_reference::document::admission::tests::d7_admission_production_rejects_rehashed_synthetic_markers
study_reference::document::admission::tests::d7_admission_nested_aliases_hex_tokens_and_unsupported_schema
study_reference::document::admission::tests::d7_admission_inherited_caps_and_small_oversize_witnesses
```

## Minimal touched shared regressions to add

Three additional default-feature library filters, each using `--lib` and `-- --exact`:

```text
study_reference::document::tests::independent_python_exact_canonical_bytes_cover_all_domains_and_timing_options
study_reference::document::tests::fix1_genuine_structural_and_escaped_keys_keep_every_string_exception
study_reference::document::target_tests::independent_literal_lock_canonical_and_file_identity
```

The first parses/encodes ten small literal Reference/Diagnostic/Failure cases from compiled `document_goldens.rs`, verifying unchanged complete canonical bytes and digests across the shared append/hex changes. The second exercises six genuine structural string paths and escaped keys with small 4,098/8,192/16,384-character strings; it specifically protects the old Reference path exemptions touched by layout expansion. The third parses/encodes the 9,156-byte frozen Lock and compares its 8,227-byte canonical-hex oracle, file and payload hashes, then checks production rejection. These three use no runtime reads or registry/materializer/parameter/HE/native construction. The existing multi-megabyte Lock layout test is deliberately not added to this bounded release.

One additional default-feature integration filter, using `--test registered_case` and `-- --exact`:

```text
canonical_header_fields_and_digests_match_nineteen_independent_vectors
```

This is the narrow existing regression for the actual adapter changed by this package. Its `registry()` calls `StudyRegistry::load` once, reading/hash-verifying exactly `crates/ckks-train-bench/configs/experiment_matrix.json` (96,937 bytes; SHA256 `164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc`) and `experiment_input_locks.json` (3,696 bytes; SHA256 `cc04f6f7451b0ec99416e10d14c2bf6b2eaf5f703c8de43d500cac3e40da126b`). It parses their Values/lock map and projects 2,680 family plus 20 exact-Iris RegisteredCase identities, retaining their selectors, small canonical Vecs/digests and seven supplement descriptors. The shared actual adapter now makes three small owned text copies per identity (registry hash, registration ID, profile ID); zero HE does not mean zero allocations. The test compares nineteen frozen complete canonical vectors and builds a 2,700-digest uniqueness set. Its compile-time oracle `tests/fixtures/registered-case-v1/representative-vectors.json` is 53,593 bytes, SHA256 `4745cbda4995d01d916f0c6cc0c8ebfae6386046cab3af408e05e3bad09d551c`. It does not resolve dataset/input locks, materialize inputs, build parameters/P1/NTT, create RNGs, inspect host/CUDA or construct readiness. Root should pin these inputs and the exact test sources in the runtime release, separately from the nine-test inventory.

Total recommended released runtime: thirteen exact tests, sequential/default feature; no wildcard/module suite, no default full test suite, no hardware feature execution, no boundary-sized inputs, and no repeat solely to obtain additional passes.

## Preserved limitations

No successful current-source/current-context gate, resource/user lease, nofollow store reader, durable publication/start state transition, process stamp, recovery/live reconstruction, immediate-before-allocation HE handoff, Quick implementation, performance or scientific evidence is established. Full approval/hold, target-before-Full-HE ordering, original C/R/A and Full domains remain unchanged. All 21 profiles, 2,680 Full family rows plus 20 exact-Iris rows and seven supplement descriptors, Quick268, CPU/GPU I/NI and baseline/security/resource requirements remain required. No external baseline code/dependency/license change or new science threshold was introduced. This review closes only the released source-review step.
