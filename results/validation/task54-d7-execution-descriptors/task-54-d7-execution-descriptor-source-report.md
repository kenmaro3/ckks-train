# Task 54 D7 execution descriptor source report

Status: DONE for the released representation/source-view layer, with the concerns below retained. No test execution or scientific execution occurred. Task 54 and the full D7 producer/owned driver remain incomplete. No commit, push, cleanup or subagent was performed by this writer.

Release: `task-54-d7-execution-descriptor-source-release.md`, SHA256 `cb753ddf7527ffe578701e02d9344d6caf278beca1f443e35d2213d296241963`. Read Task54 brief first, then release, ratified design, frozen vector archive and accounting notes. Source BASE `7da6d0a6aab3ee38d7641762bcc82a1d2150b39a`; root separately advanced only its observation request to HEAD `56d0fb369682a4daf6d092ab2a381ccabccd5a52` while source work ran. This writer did not modify `results/quadruped/requests.md`.

## Implemented boundary

Four fixed private encodings implement the exact reviewed domains and field orders: interactive parameters, physical diagonal layout, CPU key policy and security declarations. Raw descriptor field types and the narrow allocation-free wire reader are private to `study_reference::document`. They introduce no JSON envelope, dependency, scientific cap, public API, route or authority. Source encodings are crate-private extension methods on the narrow `PreparationDescriptorFacts` borrow, which has private fields and no caller constructor.

Both existing checked preparation types retain one additional borrowed P1 metadata reference. Their `descriptor_facts()` returns a borrow only for their canonical `PlanOwner`/`BlockPlanOwner`; test-only synthetic owners return `None`, even if bytes could compare equal. The view borrows the actual same plan, params, P1 and already-checked descriptions/geometries. It clones no geometry, parts, parameter chains, plan, source manifest or bootstrap metadata. The complete trainer bytes remain the existing `facts.plan().canonical_bytes()`; NI parameter/bootstrap identity remains the existing P1 bytes. No readiness is derived from a selector, transported field set or digest.

The exact existing P3 geometry body is factored once behind `GeometryWriter`, with an immutable `GeometryBody` borrow. P3 continues through its original `Transcript` writer and original limits. The new layout writer uses the same body. The old four-byte I32 chain order, bool/option tags, compact plans, containers and biases remain unchanged; original P1 eight-byte signed rotations are untouched. No planner logic or canonical P3 field order was changed.

`VerifiedStudyContext::train_source_manifest_bytes()` borrows the complete `receipt.files[0]` from the current receipt, requiring current role `StudyTrain`. It does not search historical Reference receipts or make a second implementation manifest. The existing `source_commit()` is reused; source/live rechecks and receipt validation are unchanged. Nothing in this layer establishes a new approval or source identity by itself.

## Wire and actual-source semantics

`WireDescriptor::parse` is an allocation-free grammar/size inspection. It validates exact domains, little-endian widths, lower-case Hash Text64, bool/option/enum tags, list byte availability, count multiplication/conversion, stage count1..3 and trailing bytes. Every count is checked against the remaining complete input before loops. Parameter wire parsing additionally enforces constructor-derived chain/bit-count, logN1..17, bits2..60, dnum divisibility, exact IEEE754 value of scale2^exponent with exponent1..60, and main weight1..N. There is no arbitrary parameter or rotation count cap besides checked total A_MAX and actual source/cardinality constraints.

Security parsing accepts all reviewed structural branches, including synthetic fixed-main and absent/dense/non-default ephemeral declarations. `validate_security_declarations(N)` separately checks declared positive weights<=N. `validate_registered_security_declarations(N)` is only the current registered declaration restriction: N=2^16, dense main, I bootstrap/ephemeral absent; NI exact scope0, ordered requirements0..8 and default ephemeral weight from `N16_EPHEMERAL_HW`. It does not validate actual parameters, estimate security, grant a gate, or make a synthetic wire image a source object. Assurance remains declarations_only0.

Actual-source encoding receives only the canonical checked-preparation view, whose existing preparation already performed `validate_plan_binding`, `describe_candidate` and exact secure preset checks. It derives parameter/bootstrap hashes from the borrowed real P1 or actual I parameter object, takes physical geometry/config and exact plan partitions from the retained checked descriptions/plan, and validates the training key union in BOTH directions against every actual geometry requirement. NI generated/execution rotations, conjugation, both SSE roles and separate bundle are compared with P1/plan; I requires absence. Main/P1 secret declarations must agree; registered security declarations remain checked independently. `matches_source` recomputes the selected descriptor from this genuine view and compares COMPLETE bytes. `matches_bytes` is only a non-authorizing exact comparison against supplied canonical bytes and is used with synthetic fixture expectations in the pure tests.

No actual canonical checked preparation was constructed or executed in this source phase. Actual-source extension methods and source borrowing compile, but their real-object runtime validation is not claimed. Fake source/route constructors were not introduced to turn the synthetic vectors into production positives.

## Independent literal identity and tests-first evidence

The archive proposal is `0b4c92c2f3723589ec9006089e0999255b6b209b8286ef097e4aee74abe289fe`; independent FIX1 review is `28f3cc9a2cd73b463449025d77fe09a473ba9c01d498e943f4ac67408b6b6a48`. Frozen vector `SHA256SUMS` is `899b45991113581638a5758bd9bdaec441df768e771fa706a16549faf93b287e`. All19 listed artifacts plus that SUMS file (20 actual files) were copied exactly to `document/execution_descriptor_fixtures/` before encoder source. They remain byte-for-byte equal to the archive and BASE. `generate.py` was copied as frozen documentation but never executed. Expected bytes/digests were not obtained from production Rust.

All14 positive literal images (2 parameters,3 layouts,4 keys,5 security) contain6314 bytes in total. The archive supplies56 complete negative images:30 grammar failures and26 semantic/source mismatches. The pure test treats constructor-invariant parameter semantic failures as parser refusals, and otherwise checks the unmodified synthetic retained byte expectation even after the changed image has been parsed. These comparisons are explicitly not actual canonical source evidence. All6314 proper prefixes are nominated for rejection. Numeric A_MAX tests use2097151,2097152,2097153,u64::MAX without allocating filler. Checked arithmetic uses exact overflow and32-bit conversion families.

Tests were written first. `d7-execution-01-red.log/json` records exit101 from the mandated no-run command because the new module and GeometryBody did not exist. The later added pure P3 writer regression was also written before its test-only bridge: `03-p3-helper-red` records exit101 solely for missing `test_geometry_transcript`. Raw integration compile errors in `02-no-run` are preserved: wrong local ensure name, ambiguous Kind import, wrong Transcript error associated type, Rust1.85-incompatible is_multiple_of and a borrowed parse lifetime. Those were fixed without executing tests. `13-format` preserves one formatting-only exit1; final format succeeds. No failures are omitted or rewritten.

## Compile/static evidence

Exact log directory: `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/source-validation/`. Every command has raw `.log` plus `.json` command/env/exit/elapsed metadata, prefixed `d7-execution-`. Cargo ran with `CARGO_NET_OFFLINE=true`; no network execution was requested. Final commands:

| Evidence prefix | Exact command | Exit |
|---|---|---:|
|15-final-no-run|`cargo test -p ckks-train-bench --lib --features parallel --no-run`|0|
|16-final-clippy|`cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings`|0|
|17-final-check-serial|`cargo check -p ckks-train-bench --lib --no-default-features`|0|
|18-final-check-cuda|`cargo check -p ckks-train-bench --lib --features parallel,cuda`|0|
|19-final-format|`rustfmt --edition 2024 --check --config skip_children=true` followed by exactly the seven owned Rust paths listed below|0|
|20-final-diff-check|`git diff --check`|0|

The compiled test executable is `target/debug/deps/ckks_train_bench-190377a24cc433d7`; it was never launched by this writer. CUDA-feature compilation is not a GPU/native observer or runtime check. Earlier successful/intermediate compile evidence04..12 remains alongside failures01..03/13. Source-validation pin files record108 unchanged binding paths against BASE, including P1 params/metadata, original P/R codec, runtime/dispatch, package/dependency files, original target golden files and the entire independent vector package. Existing golden bytes remain unchanged on disk; their runtime assertions have not run here.

## Proposed bounded pure batch (not executed or released here)

Prefix every test below with `study_reference::document::execution_descriptor_tests::`, use an exact-name filter and one test thread. These10 tests are proposed for root's independent review/freeze; they are not permission for a broader suite.

| Exact test suffix | Bounded work |
|---|---|
|execution_parameter_literal_bytes|2 encoder images, complete literal bytes plus frozen digests|
|execution_layout_literal_bytes|3 encoder images,6 hand-written literal stage DTOs; all stage/compact/partial branches|
|execution_key_literal_bytes|4 encoder images; dedicated/reuse and separate generated/execution bootstrap lists|
|execution_security_literal_bytes|5 encoder images; None/fixed/dense secret branches, no passing security claim|
|execution_complete_negative_images_distinguish_wire_and_binding|56 complete images;30 explicit grammar errors,26 constructor/source-semantic cases|
|execution_every_proper_prefix_rejected|6314 proper prefixes across the14 complete images|
|execution_checked_size_and_arithmetic_guards|4 exact A_MAX numeric cases;2 overflow and2 conversion cases; no filler allocation|
|execution_synthetic_wire_has_no_registered_policy_authority|4 unsupported synthetic security examples; valid declaration weights at synthetic N16, refused current registered declaration policy at N65536|
|execution_parameter_constructor_invariants_without_construction|13 literal-field refusals:7 logN/dnum/scale/weight,3 bit-range,2 scale-exponent and1 non-power-of-two scale|
|execution_shared_p3_geometry_matches_frozen_layout_slices|6 literal geometry bodies through the ORIGINAL P3 Transcript writer; exact frozen complete slice comparisons, zero planner/params construction|

Exact frozen half-open geometry slice ranges are: `layout_i_single` stage0[196,603); `layout_ni_two` stage0[196,845),stage1[877,1526); `layout_i_three` stage0[196,603),stage1[635,1042),stage2[1074,1481). Both the encoder test and original-Transcript regression take expected ranges from frozen independent manifest offsets. This is directly relevant coverage of the changed shared geometry body; unrelated broad P3/HE tests are not nominated.

Constructor inventory for this batch:0 CkksParams/new/preset/reduced-level constructors;0 BootstrapMetadata/BootParams/DFT constructors;0 candidate/geometry/initial/update/handoff/full-plan planners;0 checked-preparation/source-context constructors;0 dataset materializations;0 KeyGenerator, CKKS Encoder, Evaluator, Bootstrapper, HeSession or session constructors;0 native/GPU/network calls. The document Encoder/Transcript are byte writers only. Plain literal Layout/LayerShape/MaskRecipe/CompactPlan/CompactLayout/ContainerSpec/BiasSlot/MicroPart structs and test LiteralStage DTOs are built from the frozen JSON; none is a checked StageGeometry or valid-source constructor. There are12 literal stage DTOs total across the positive layouts and P3 regression. The encoder batch has14 successful descriptor-output Vecs and13 pre-output parameter refusals; the P3 regression adds6 small Transcript-output Vecs. Geometry body writing is12 calls in the two-pass layout encoders plus6 original-Transcript calls. There is no retained source adapter or HE work hidden inside this inventory.

Fixture/file inventory: all runtime test inputs use `include_str!` on `inputs.json`, `manifest.json`, `negative-vectors.json` and the14 named `.hex` files. No test opens files, creates files, reads a dataset or accesses environment/process/network. Copied README/generator/SUMS are immutable evidence files and are not executed or read at runtime by the test batch. Bounded fixture allocations are serde_json literal Value trees/clones, hex Vecs, small literal geometry/key/parts vectors, descriptor Vecs, SHA256 strings and bounded error-path strings. No multi-megabyte padding artifact is constructed. Repeated JSON parsing/Value clones remain fixture work and are not cryptographic or planning evidence.

## Allocation/accounting call-site inventory

All new descriptor output has one checked borrowed sizing pass, followed by `try_reserve_exact(n)` only after all checked arithmetic and total `n<=A_MAX`; the write pass rechecks every append and final exact length. No geometry clone is used for sizing. The existing document Encoder provides LE/Text/Bool/Option representation, and the existing hash helper hashes borrowed bytes. Failure diagnostics allocate only an existing bounded DocumentError path string. Requested payload below does not mean capacity or RSS.

| Call site | Retained/transient owner and checked sizing/reservation/release |
|---|---|
|CheckedPreparation / CheckedBlockPreparation new P1 field; descriptor_facts|Borrowed references and stack-only view; no new heap payload or Arc clone; lifetime limited by checked owner/P1/plan|
|ParameterFields::encode|Borrows q/p/bit vectors; checked complete size pass, then one output Vec of exact requested bytes<=A_MAX; returned Vec owned by caller and released on caller drop|
|LayoutFields::encode|Borrows stages, complete recipes and plan parts; counts/cardinality/checked multiplication and full size before output reserve; returned Vec owned by caller|
|KeyFields::encode / SecurityFields::encode|Borrows ordered pools/rotations/P1 requirements; complete checked size before output reserve; returned Vec owned by caller|
|source parameter_hash, I branch|One separately bounded I parameter Vec, then64-byte digest String; parameter Vec released after hashing before layout/key/security output allocation|
|source parameter_hash, NI branch|Hashes borrowed unchanged P1 parameter bytes;64-byte String only; no P1 payload copy|
|physical_layout_bytes|Transient64-byte parameter digest remains alive with returned-layout construction; released on method return; returned layout Vec remains|
|cpu_key_policy_bytes / security_declaration_bytes|Transient64-byte parameter digest and optional64-byte bootstrap digest, plus output; P1/plan/geometry/transpose stay borrowed; transient digest Strings released on return|
|WireDescriptor::parse / matches_bytes / security declaration checks|Borrowed slices/cursors only on success; no parsed geometry/list/text allocations; bounded error strings on refusal|
|matches_source|One selected expected descriptor Vec and its transient source hashes as above; expected Vec released after full-byte comparison; this is validation only|
|existing prepare_then_make / prepare_blocks_then_make|Unchanged validation/candidate planners, retained checked geometry clones/activation/transpose/classes/rotations ownership and current original-request allowance rules; no new actual preparation call was made here|
|future post-start validate_plan_binding|NOT implemented/called here. Its canonical-binding transcript, partitions and identity/row temporary workspace still require a distinct reservation/release under original request and independent counting before keys|

One additive preparation accounting owner seeded exactly once with `plan.charges().combined()` is still required in the subsequent retained-route layer. This phase does not claim that the four local A_MAX encoders account for existing plan/preparation clones, materialization/params/P1 bounds, target verification, future factory arrays, cryptographic allocations, live resource admission or process RSS. Original request max_bytes/max_records, parameter/P1 caps and plan charges are unchanged. No allowance was increased/reset after hashing. Future callers must charge the listed retained/peak transient payload together with the existing plan charge before these helper calls; codec success cannot hide this missing integration.

## Remaining concerns and gaps

- The10 pure tests are compiled but unexecuted; current byte compatibility claims are source-preserving design/fixtures plus compiled assertions, not observed test passes. Root independently reviews and freezes runtime authority next.
- The actual retained pre-key CPU preparation/closed route and ExecutionBinding producer do not exist yet. No `study_execution.rs`, PreparedCpuRoute, StartedHeCase, live gate, durable admission/start or completed result was added. Existing run_owned refusal is unchanged.
- Exact plan/context/Arc/parameter ownership rechecks, runtime producer from the actual route/lease, post-start validate_plan_binding workspace, repeated boundary rechecks and additive requested-payload accounting remain open. Factory exact rotation-union enforcement must be wired at actual construction, not inferred from this descriptor check.
- The FIX1 unused-driver/setup-failure drop order (crypto before live leases), exclusive store-borrow transfer and fixed post-start owned driver remain subsequent work. This layer constructs no HeSession and clears none of those obligations.
- P1 limitations, eight P3 unproved gates, five real outer Gates, security estimates, actual secret/key compliance, physical resource admission, GPU/native observers, scientific publishers and metrics remain separate. Full encrypted block user hold remains. No experiment denominator/profile/budget/domain was changed and no science/benchmark/baseline was run.
- No source ownership expansion was needed. Original `.vscode/` and both pre-existing untracked MNIST result files were preserved; current hashes are recorded as present/unwritten, without claiming an unavailable pre-phase untracked-byte snapshot. Root's request file belongs root throughout.

## Actual changed source/fixture hashes

The following hashes are final source-phase bytes. The final path manifest in source-validation additionally pins this report and every raw compile/pin evidence file; the manifest's own hash is reported separately to root to avoid a self-hash cycle.

| Path | SHA256 |
|---|---|
|`crates/ckks-train-bench/src/study_admission.rs`|`c69e354eab5779765873be97d2ebdb64792619c7f6a26a2759e4ea318c74f603`|
|`crates/ckks-train-bench/src/study_client.rs`|`7a900f97426dbab4cfc6338b0c7a7a5407eaa8b970879f24f01d6d4402352672`|
|`crates/ckks-train-bench/src/study_client/blocks.rs`|`4d28caa621435d2e5b64ec163ec172f1ba7722cbbf6d91cd9dec2c0a0b416223`|
|`crates/ckks-train-bench/src/study_provenance/context.rs`|`f8d62d8561c6c3750d0c3538ec2d5959e706ebcd11f5fc143f12fba64b69b81a`|
|`crates/ckks-train-bench/src/study_reference/document.rs`|`688940f3e0849804e9a91e0bd9e0d9bbf33d4a3f2ee8ac12b8116a7299215900`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptors.rs`|`59510ec04c75337c747dd6f9e55a8823d75c872f1ef0125b36ad4353c5754974`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_tests.rs`|`e46f5ad1cabb7e0db464ca8dcb1a40c43c40e27ac1b698e667832eb8c1c63c53`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/README.md`|`d9d36e8f0ba53fc5659f3c9eebe4b673c34e45f05f83376e70651b78eb1ffc8b`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/SHA256SUMS`|`899b45991113581638a5758bd9bdaec441df768e771fa706a16549faf93b287e`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/generate.py`|`8e1bdbea566ebc051a3b3c36d07d16406bb2fbc75f4a62c1d3a5374aa979356c`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/inputs.json`|`1f30a3c7665fa184ed0b0e971343c0770f51eabcbb26c5984c934ab312c05b85`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/keys_i_dedicated.hex`|`d993093d2210e53279db5bc6a44380e7ebd787f7eb58f061801752e3c29010ef`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/keys_i_reuse.hex`|`8340d77d596216ba3d68b8894a9758c6fdea42da300418420dcd22533976241a`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/keys_ni_dedicated.hex`|`ec2543b1d824fed9ccd9e7eb0e22fe427a0c0ae04b1be71b4253124c4526b49c`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/keys_ni_reuse.hex`|`c5d1c62417f79a332df0690de274efad09bbe7cf7cf12e8be78d0bcd49ce0b31`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/layout_i_single.hex`|`d4ed1f0762e026a6483f1e71df4aad1bc95e74860ab6cd4b2f873e3a85a20e5c`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/layout_i_three.hex`|`7a91ad819ed0f7d4fe858cdc9d4b604cca079df3b3772601ffdf8bc65492b777`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/layout_ni_two.hex`|`ce94c7b1613a6ed010d0f4bd6ffe2c8a1c221aa679d76049e85733bfdcfe2e6b`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/manifest.json`|`6b983e7d31016756ed5906be1558e0eebcd923d37f5b3f8332dd2e28428e96d7`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/negative-vectors.json`|`d8c2fcf1e4abf13b271e4a0c1e48af4bbc45506aecc946c5a9e14ed5f7a16a70`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/parameter_dense.hex`|`bb2f659c77068ae1e2fa5ae4c2befe52d3e83897018f083fbaad23db494090d8`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/parameter_fixed.hex`|`f55464bafd17dc40404e13758cdfbad85a94ad34dc63cbb6363239a084af090b`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/security_i_dense.hex`|`9c5f2bf26e41de319f1b5c82f2fe559742ff626eabdd149476be5ef80f3fbb5e`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/security_i_fixed.hex`|`90b5431d112d0e46cbe66de8a454d07a4ccef53ecc39c8372a5be42d6ca335bf`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/security_ni_absent.hex`|`4a631472f676da1af8b09d9126075573974a8e94d513f8141eefb83c3cb99987`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/security_ni_dense.hex`|`e8c49ec91a42be52dd4829644f591c252aae32b377fbf07aaef15b67224c35f4`|
|`crates/ckks-train-bench/src/study_reference/document/execution_descriptor_fixtures/security_ni_fixed.hex`|`b554dfa3958790f6ffad3a33cb38c3325b4eb71e5dfe32c7905f2a921ca7b737`|
