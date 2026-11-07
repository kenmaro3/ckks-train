# Task54 D7 Train capture source report

Status: source dependency implemented; compile/static validation complete. No runtime test, production invocation, key/HE/learning operation, remote SSH, commit or push was executed by the implementer. This is not full Task54 completion. Root independent source review is next; the proposed pure-runtime inventory below is not an execution release.

Working directory: `/Users/marconi/Documents/research/hekit`. Source BASE: `9aba62f0a2a2b565466216511238dec3fd7a244f`. Root subsequently committed only prepared `requests.md` observation text at `bb4f65f`; no source base changed. The source writer did not stage or commit.

Read order: `task-54-brief.md`, Train source release/draft, owned-path API map, `docs/study-runtime-design.md`, `docs/study-target-design.md`, `docs/study-runtime-descriptors.md`, then authoritative existing source. Original dispatch release pin: `45631742d11ed8b4e7f2608b40b348b0906eb875752c98993a0bdf49032de070`. Root confirmed the append-only correction granting narrow wiring in actual `study_provenance.rs` (rather than nonexistent `study_provenance/mod.rs`); resulting release SHA256: `a28ad4c1d6bdbc76f624330bd7d48e87b5603ab37bd941a7a1db2cf539674d28`. No additional source ownership was inferred.

Implemented flow: private retained `run_owned` loads the approved Train context and pinned registry, selects the actual immutable `RegisteredCase`, resolves its own `profile_id` and checked seed, materializes exactly once, invokes `capture_train`, consumes the capture through `finish_train_capture`, and encodes the existing Case component in that same synchronous owner frame. Resolution/materialization/context/codec failure remains an explicit error. After component preparation it still returns `RunOutcome::MissingExecutionDependency`. It publishes nothing and reaches no key generator, cryptographic Encoder/evaluator/Engine/session, JIT, learning loop or scientific writer. The document codec Encoder is byte serialization only.

`TrainCapture` and `RecheckedTrainCapture` are crate-private, have private fields, no constructor/Clone/Default/Deserialize, and borrow the exact `VerifiedStudyTrainContext`, `RegisteredCase` and `MaterializedProfile`. Finish consumes the first capture, checks exact pointers for all three, repeats role/recipe/backend/profile/seed/batch/canonical-input checks, and reuses the existing checked-input recheck for contract/receipt/materialized/source/build-receipt equality. Context rechecks preserve the actual process/owner/pool/host/CUDA/executable/approval/source checks. Executable start/end digest and invocation cwd/argv are retained and compared. Current build access is Train-role checked and derives from that exact context, not a caller-selected receipt.

`CaseRuntimeLease::facts` brackets its private immutable projection with the unchanged full lease recheck. Worker count comes from `HostExecutor::workers`; host is borrowed from that same retained lease. It does not use the Reference-only `Host::execution_threads` constant, environment99, a requested count or global Rayon discovery. Stable host comparison remains cpu_model/os/hostname; telemetry snapshot drift (available CPUs/observed_threads/thread environment) does not become a stable identity failure. CUDA module/device/process liveness checks remain those of the retained runtime, without new CUDA facts/fields or observers.

`RecheckedTrainCapture::case_provenance`, implemented inside the private document module, formats only actual rechecked capture facts and returns bounded canonical bytes plus their SHA256. It reuses existing nested build/raw/observation formatting, existing `CaseValue` decoding/validation/canonical field order, captured tag0, StudyTrain role1 and schema `hekit-study-case-provenance-v1`. The component is explicitly inspection, with no inverse conversion into capture/context/admission/start authority. The private codec accepts a producer-assembled JSON value as a formatting boundary; tests may supply synthetic fixture values to that same codec, but no parsed claim or JSON can construct either captured type. Original caps, Reference schema/role guards and five design pins remain unchanged. ProcessStamp, pool addresses and native symbols are not added to wire facts.

Changed source files and SHA256 (also in `source-pins.json`):

| Path | SHA256 |
|---|---|
| `crates/ckks-train-bench/src/study_runtime.rs` | `cbd8be188554061f75b78b6924ac4d7be947a88ad482f5d64d6d3e3c0a73d5c0` |
| `crates/ckks-train-bench/src/study_runtime/tests.rs` | `d95a197623e9132f9189cad6cca744ea9c5564479088993f2f070767165398bf` |
| `crates/ckks-train-bench/src/study_provenance.rs` | `0bb4200cab631d940cbf28c18d09e939af5751d441f8897746aeffe8b5524e35` |
| `crates/ckks-train-bench/src/study_provenance/context.rs` | `1a704b951a4557788c6e48aba939e096d219a4e5f144f6f0858d297e9f972928` |
| `crates/ckks-train-bench/src/study_provenance/train_capture.rs` | `5825c51c15ac917b2fb8df7c9177ced08fb6e44c374108cf6f8a6a947087ae9f` |
| `crates/ckks-train-bench/src/study_provenance/train_capture_tests.rs` | `c1d7d412feb27e4149328097d5a93ab826f70c429a081e22509a0a77b3930857` |
| `crates/ckks-train-bench/src/study_reference/document.rs` | `d42e96b7c493ff501f4de7f794e3e04895f60da32b805c34a83dcdc76e697d15` |
| `crates/ckks-train-bench/src/study_reference/document/case_provenance.rs` | `97dffae2b064a4bfa730eec9c4ceff18c21cd8d7a7c4fa40c567843194095e2e` |
| `crates/ckks-train-bench/src/study_reference/document/case_provenance_tests.rs` | `427698f2a3c5232245fcd645866195d05536e449e6d5212371388b6fb7a7e068` |
| `crates/ckks-train-bench/src/study_reference/document/producer.rs` | `ecbb85c62430b47f7ea698b5e8b81a03a8fc6ab41d814231dccff326f202f06f` |

All changes are inside the released ownership boundary plus root-confirmed narrow `study_provenance.rs` wiring. New files are only `study_provenance/train_capture.rs` and `train_capture_tests.rs`. No Cargo/dependency/P4/P5/Reference publication source was changed. `source-static-check.py` verified 256 tracked bench-source/workspace-Cargo paths outside the owned list byte-for-byte against BASE, including all 82 Case golden files. It separately compares original Reference capture/finish/input-helper bodies, Reference captured producer, and preexisting Case tests/document source prefix byte-for-byte. `.vscode/` and the two unowned `results/mnist_blocks*` files were untouched and remain present; hashes are recorded in `static-preservation.json`.

Compile/static evidence: every invocation below has exact argv/cwd/start/end/exit in `<id>.meta.json`, unedited stdout/stderr in the matching files, all beneath `task-54-d7-train-capture-source-validation/`. Failed commands are retained. No test executable was launched.

| Evidence | Exact command | Exit / interpretation |
|---|---|
| `01-red` | `cargo test -p ckks-train-bench --lib --features parallel --no-run` | 101; genuine compile RED: 10 errors for absent Train module/capture/finish/runtime-facts/Case-codec APIs |
| `02-rustfmt` | `rustfmt --edition 2024 --config skip_children=true crates/ckks-train-bench/src/study_runtime.rs crates/ckks-train-bench/src/study_runtime/tests.rs crates/ckks-train-bench/src/study_provenance.rs crates/ckks-train-bench/src/study_provenance/context.rs crates/ckks-train-bench/src/study_provenance/train_capture.rs crates/ckks-train-bench/src/study_provenance/train_capture_tests.rs crates/ckks-train-bench/src/study_reference/document.rs crates/ckks-train-bench/src/study_reference/document/case_provenance.rs crates/ckks-train-bench/src/study_reference/document/case_provenance_tests.rs crates/ckks-train-bench/src/study_reference/document/producer.rs` | 0; passed |
| `03-no-run` | `cargo test -p ckks-train-bench --lib --features parallel --no-run` | 0; passed |
| `04-clippy` | `cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings` | 101; preexisting redundant closure in runtime nested-pool test |
| `05-check-serial` | `cargo check -p ckks-train-bench --lib --no-default-features` | 0; passed |
| `06-check-cuda` | `cargo check -p ckks-train-bench --lib --features parallel,cuda` | 0; passed |
| `07-rustfmt` | `rustfmt --edition 2024 --config skip_children=true crates/ckks-train-bench/src/study_runtime/tests.rs` | 0; passed |
| `08-clippy-final` | `cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings` | 101; Clippy-suggested &check triggered needless-borrow lint; preserved |
| `11-clippy-final` | `cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings` | 0; passed |
| `12-no-run-final` | `cargo test -p ckks-train-bench --lib --features parallel --no-run` | 0; passed |
| `13-diff-check` | `git diff --check` | 0; passed |
| `14-rustfmt-check` | `rustfmt --edition 2024 --check --config skip_children=true crates/ckks-train-bench/src/study_runtime.rs crates/ckks-train-bench/src/study_runtime/tests.rs crates/ckks-train-bench/src/study_provenance.rs crates/ckks-train-bench/src/study_provenance/context.rs crates/ckks-train-bench/src/study_provenance/train_capture.rs crates/ckks-train-bench/src/study_provenance/train_capture_tests.rs crates/ckks-train-bench/src/study_reference/document.rs crates/ckks-train-bench/src/study_reference/document/case_provenance.rs crates/ckks-train-bench/src/study_reference/document/case_provenance_tests.rs crates/ckks-train-bench/src/study_reference/document/producer.rs` | 0; passed |
| `15-static-preservation` | `python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-train-capture-source-validation/source-static-check.py` | 0; passed |

Tests were added before implementation. RED was a genuine `cargo test -p ckks-train-bench --lib --features parallel --no-run` failure for missing actual APIs, not an executed test failure or an intentional typo. Initial successful no-run is `03-no-run`; final successful no-run after all edits is `12-no-run-final`. Strict clippy passed at `11-clippy-final`. The only preexisting test-body cleanup is replacing a redundant second `rayon::join` closure with the equivalent `check` callable, after preserving both lint attempts. The existing environment99 isolated-child test now also asserts the actual new facts projection remains2 and borrows the same observed Host. Other existing runtime tests, Reference/Case tests and existing compile-fail examples remain present and compile.

Proposed bounded pure-runtime inventory (NOT EXECUTED): exact full names/argv/features/work counts are in `proposed-pure-test-inventory.json`. Use `--features parallel`, exact filtering, and one test thread. There are 31 selected parent unit tests on Mac: 5 new behavioral tests and 26 relevant regressions. Total materialization work is exactly two canonical `iris__emlp1` inputs (seed0 once, seed7 once); no additional canonical source is created for negatives. There are 11 dedicated pools with worker counts `[1,1,2,2,1,1,2,2,2,2,2]`, 10 HostExecutor objects, two explicitly spawned std threads and one subprocess. The environment99 parent must begin with `CKKS_RUNTIME_POOL_TEST_CHILD` unset; its one child runs only the same exact test and creates pool2. All other tests create zero subprocesses. These counts exclude Cargo/rustdoc infrastructure. No live approved Train context, keys, HE, learning or output publisher appears in this inventory.

| Exact test name | Kind | Materializations | Pools (workers) / executor objects | Explicit threads / subprocesses |
|---|---|---|---|---|
| `study_provenance::context::train_capture_tests::train_capture_canonical_iris_0_7_case_input_and_owner_binding` | new | 2 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::train_capture_tests::train_capture_role_is_distinct_from_reference_authority` | new | 0 | `[]` / 0 | 0 / 0 |
| `study_runtime::tests::runtime_capture_fact_projection_serial_is_actual_and_borrowed` | new | 0 | `[]` / 1 | 0 / 0 |
| `study_runtime::tests::runtime_capture_fact_projection_exact_pools_and_owner` | new | 0 | `[1, 1, 2, 2]` / 3 | 1 / 0 |
| `study_runtime::tests::runtime_worker_bounds_and_compile_role_match` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_runtime::tests::runtime_serial_owner_thread_is_not_transferable` | regression | 0 | `[]` / 1 | 1 / 0 |
| `study_runtime::tests::runtime_exact_pool_rejects_foreign_same_size_and_outside` | regression | 0 | `[1, 1, 2, 2]` / 2 | 0 / 0 |
| `study_runtime::tests::runtime_nested_join_iteration_and_synchronous_host_route_stay_owned` | regression | 0 | `[2]` / 1 | 0 / 0 |
| `study_runtime::tests::runtime_owned_pool_ignores_environment_in_isolated_child` | regression | 0 | `[2]` / 1 | 0 / 1 |
| `study_runtime::tests::runtime_retained_count_drift_fails_even_inside_exact_pool` | regression | 0 | `[2]` / 1 | 0 / 0 |
| `study_runtime::tests::runtime_owned_selector_refuses_backend_before_context_or_pool` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_runtime::tests::runtime_mac_never_fabricates_production_process_stamp` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_producer_codec_preserves_independent_components_synthetic` | new | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_all_six_independent_components_exact` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_all_67_actual_negative_literals_reject` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_member_order_does_not_choose_canonical_order` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_reference_stays_scalar_and_unavailable` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_preflight_exact_paths_and_aggregate_budget` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::case_provenance_tests::case_provenance_canonical_bytes_cannot_escape_outer_budget` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::runtime_environment_absent_empty_and_non_utf8_are_distinct` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::runtime_override_and_argument_bounds_fail_closed` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::role_current_and_artifact_reference_predicates_are_independent` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::reader_guard_reference_mutation_policy_rejects_all_train_recipes` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::reader_guard_selected_sources_reject_before_hash_or_accept_exact_file` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::train_context_requires_v2_train_recipe_and_exact_runtime` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::context::tests::train_current_candidate_requires_role_capsule_and_executable` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::process::tests::process_exact_match_checks_every_field_and_observed_pid` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::process::tests::process_stat_comm_delimiter_and_u64_boundaries` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::process::tests::process_stat_malformed_missing_overflow_and_trailing_reject` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_provenance::process::tests::process_boot_exact_grammar_not_uuid_version_policy` | regression | 0 | `[]` / 0 | 0 / 0 |
| `study_reference::document::tests::independent_python_exact_canonical_bytes_cover_all_domains_and_timing_options` | regression | 0 | `[]` / 0 | 0 / 0 |

Evidence boundaries for those proposed tests: canonical Iris inputs are actual local immutable materializations, but context-pointer tests use plain owner sentinels and snapshot predicates; this does not manufacture a verified Linux context. Runtime projection tests use actual private serial/pool executors with an observed Host; they do not capture a production ProcessStamp/approval. Role/receipt/process and Case/Reference vectors are synthetic structural/codec evidence. The new codec test checks all six independent positive byte/hash goldens and representable semantic negatives after Value roundtrip; the unchanged existing 67-negative test alone preserves all exact raw transport negatives. Source review and unchanged runtime checks provide the production connection; actual positive installed Linux Train capture remains unmeasured.

Compile-only coverage is deliberately excluded from unit-test pass counts. `train_capture_actual_api_compile_only` typechecks the actual borrowed capture→finish→component APIs but is never invoked. Ten compile-time negative trait assertions check the actual two private types against Send, Sync, Clone, Default and Deserialize; adding any such implementation makes inference ambiguous. Two added external constructor/import compile-fail examples supplement the existing context/executor/borrow examples. A proposed separate doctest filter is:

```sh
cargo test -p ckks-train-bench --doc --features parallel study_provenance::context::VerifiedStudyTrainContext -- --test-threads=1
```

That filter selects nine current-source examples on `VerifiedStudyTrainContext`: eight compile-fail cases (including the two new private capture type constructor/import failures) and one compile-pass artifact-borrow example. It is NOT RUN in this source phase. Import privacy demonstrates external inaccessibility; the internal negative trait assertions separately verify actual Clone/Send/Sync/Default/Deserialize absence. Lifetimes/fields retain all three exact borrows; no unsafe Send/Sync or public callback is added.

To keep the exact two-input budget, do not additionally select `reader_guard_materialized_checks_retain_selected_seed_and_context_binding` or `actual_iris_seed_zero_and_seven_all_graphs_roundtrip_only_input_identities` in this batch: the former would independently materialize Iris0/7 again; the latter materializes more graphs. Their source is unchanged and compiles under the final no-run. The finite proposed exact-name manifest avoids them; broad module filters would exceed this inventory.

Residual production dependencies remain: physical/runtime and remaining execution descriptor encoders/checked concrete routes; genuine five live gate adapters bound to the exact case/input/owner; target/reference store request plumbing; admission/start protocol and durable receipts; the concrete pre-key consuming handoff; CPU/CUDA/matched/packed/external adapters; checkpoint/measurement/scientific output and full campaign/quick orchestration. Ratified runtime/physical descriptor representation is design only in this package; no descriptor/gate/admission authority is inferred from it. A fresh approved current Linux Train build/context and installed positive integration evidence are still required. Mac `ProcessStamp::capture` remains a production failure, never fabricated. Native CUDA availability/route/resource checks and historical FULL encrypted-block user holds are not released.

Full study scope is preserved: all21 concrete profiles, 2680 family Full registrations,20 distinct prospective exact-Iris supplement rows, seven retained supplement entries,268 quick family projections, missing captured inputs, unresolved supplements, distinct Reference-only publication and unsupported backend/route states. P4/P5 remain their prior reviewed scope. No full Task54, full matrix execution, scientific accuracy/timing/GPU-memory/security result, or HE readiness claim follows from this source dependency.
