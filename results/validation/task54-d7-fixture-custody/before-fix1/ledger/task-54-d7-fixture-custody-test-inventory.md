# Finite fixture custody test inventory

Frozen before executable test filters and source changes. Source BASE ec2d2656f21e9139d3dde12abdce13d770ea2757.

Selected source: 565 regular files, 12812789 bytes. Original family inventory: 160 files including two retained adoption notes. See task-54-d7-fixture-custody-original-inventory.json for every path/length/SHA256.

case_provenance_tests: 7 existing tests.
- `study_reference::document::case_provenance_tests::case_provenance_all_six_independent_components_exact`
- `study_reference::document::case_provenance_tests::case_provenance_all_67_actual_negative_literals_reject`
- `study_reference::document::case_provenance_tests::case_provenance_member_order_does_not_choose_canonical_order`
- `study_reference::document::case_provenance_tests::case_provenance_reference_stays_scalar_and_unavailable`
- `study_reference::document::case_provenance_tests::case_provenance_preflight_exact_paths_and_aggregate_budget`
- `study_reference::document::case_provenance_tests::case_provenance_canonical_bytes_cannot_escape_outer_budget`
- `study_reference::document::case_provenance_tests::case_provenance_producer_codec_preserves_independent_components_synthetic`

execution_binding_tests: 3 existing tests.
- `study_reference::document::execution_binding_tests::d7_outer_six_frozen_independent_vectors`
- `study_reference::document::execution_binding_tests::d7_outer_payload_checked_live_intervals`
- `study_reference::document::execution_binding_tests::d7_outer_transport_bounds_refuse_without_route_authority`

target_link_tests: 4 existing tests.
- `study_reference::document::target::linkage_tests::literal_synthetic_reference_links_every_projection_field`
- `study_reference::document::target::linkage_tests::digest_confusion_whitespace_source_clock_and_endpoint_substitutions_reject`
- `study_reference::document::target::linkage_tests::every_policy_binding_rejects_rehashed_substitution`
- `study_reference::document::target::linkage_tests::independent_literal_reference_uses_unchanged_inherited_record_encoder_only`

target_tests: 9 existing tests.
- `study_reference::document::target_tests::independent_literal_lock_canonical_and_file_identity`
- `study_reference::document::target_tests::binary64_separate_operations_exact_vectors`
- `study_reference::document::target_tests::auxiliary_ids_are_domain_separated_exact_canonical_bytes`
- `study_reference::document::target_tests::every_required_lock_field_and_unknown_decoded_duplicate_keys_reject`
- `study_reference::document::target_tests::target_rejects_one_ulp_neighbors_negative_zero_nonfinite_and_f32`
- `study_reference::document::target_tests::target_layout_exempts_only_exact_contract_path_and_charges_all_outer_bytes`
- `study_reference::document::target_tests::maximal_individual_manifest_already_overflows_aggregate_outer_budget`
- `study_reference::document::target_tests::bounded_integer_list_depth_and_decoded_string_limits`
- `study_reference::document::target_tests::exact_rational_substitution_fails_observation_before_payload_integrity`

evidence_tests: 15 existing tests.
- `study_reference::document::evidence::tests::d7_evidence_independent_all_ten_branches`
- `study_reference::document::evidence::tests::d7_evidence_auxiliary_and_quick_exact_json`
- `study_reference::document::evidence::tests::d7_evidence_concrete_representation_mutations`
- `study_reference::document::evidence::tests::d7_evidence_full_quick_exact_claims_and_nonpromotion`
- `study_reference::document::evidence::tests::d7_evidence_quick_event_links_policies_argv_and_context`
- `study_reference::document::evidence::tests::d7_evidence_strict_json_and_caps`
- `study_reference::document::evidence::tests::d7_evidence_request_consumption_retained_comparisons`
- `study_reference::document::evidence::tests::d7_evidence_authority_resource_and_closed_branches`
- `study_reference::document::evidence::tests::d7_evidence_all_literals_closed_tags_and_trailing`
- `study_reference::document::evidence::tests::d7_evidence_target_expected_claims_and_file_identity`
- `study_reference::document::evidence::tests::d7_evidence_implementation_ni_mandatory_and_missing`
- `study_reference::document::evidence::tests::d7_evidence_quick_failed_outcomes_and_binary_argv`
- `study_reference::document::evidence::tests::d7_evidence_request_shared_p3_traversal_and_all_original_limits`
- `study_reference::document::evidence::tests::d7_evidence_production_mixed_nested_domains_tail_and_length_refusals`
- `study_reference::document::evidence::tests::d7_evidence_shared_predecessors_and_clock_claims`

admission_tests: 11 existing tests.
- `study_reference::document::admission::tests::d7_admission_frozen_documents_complete_bytes_and_hashes`
- `study_reference::document::admission::tests::d7_admission_six_frozen_branches_and_domain_components`
- `study_reference::document::admission::tests::d7_admission_target_start_and_explicit_predecessor_claim_links`
- `study_reference::document::admission::tests::d7_admission_required_nested_fields_unknowns_types_and_link_mutations`
- `study_reference::document::admission::tests::d7_admission_closed_layout_arithmetic_and_small_lexical_boundaries`
- `study_reference::document::admission::tests::d7_admission_rehashed_cross_record_fields_and_clock_ordering`
- `study_reference::document::admission::tests::d7_admission_production_rejects_rehashed_synthetic_markers`
- `study_reference::document::admission::tests::d7_admission_nested_aliases_hex_tokens_and_unsupported_schema`
- `study_reference::document::admission::tests::d7_admission_inherited_caps_and_small_oversize_witnesses`
- `study_reference::document::admission::tests::retained_c1_current_and_prior_share_one_verified_target_representation`
- `study_reference::document::admission::tests::retained_c1_exact_file_payload_and_stale_self_are_distinct`

crates/ckks-train-bench/src/study_reference/target_goldens/: 77 payloads, 553493 original bytes, maximum 79461 bytes.
crates/ckks-train-bench/src/study_reference/document/case_provenance_goldens/: 81 payloads, 489492 original bytes, maximum 77935 bytes.

All named tests use inert parser/supplied-data constructors. ExecutionFields/binding_payload performs pure integer/byte work; encode_captured_value receives synthetic Value only. PlanningBudget/DiagonalRequest in evidence tests are supplied record constructors; no planning traversal, registry, materializer, Context, native, HE, dataset or process call is invoked. External evidence/linked-oracle fixture readers read existing literal bytes only.

Maximum pre-inventoried synthetic constructions: target layout at_limit=7,324,204 bytes (+1 refusal), C_MAX=2,613,526, A_MAX=2,097,152; encoder limit=4,710,678. Target/Case aggregate witnesses construct512 paths of4096 bytes (about2.2MiB JSON), Case wrapper extends A_MAX spaces. Admission max manifest513, argv257, feature65, environment33; compiler16385 bytes, hex8194 bytes, escaped string24582 bytes; evidence malicious list lengths are supplied u64 values, not allocated vectors. No scientific JSON is written. Tests run serially (--test-threads=1).

Commands (existing cache, no dependency installation):
- `python3 -B ...task-54-d7-fixture-custody-preflight.py red|green` calls existing pure preflight_manifest(0, rows), no source_inventory/build/Git/process helper.
- `python3 -B ...task-54-d7-fixture-custody-check.py` finite independent custody test, first RED before bundles exist, then GREEN.
- Each exact ID above: `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo test -p ckks-train-bench --lib ID -- --exact --test-threads=1`.
- New narrow accessor custody tests: IDs frozen below when added before loader implementation; exact commands and outcomes retained in commands log.
- `CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=2 cargo clippy -p ckks-train-bench --all-features --all-targets -- -D warnings` (compile/check only).
- `cargo fmt --all -- --check`.

565 preflight RED rows come only from current tracked roots plus fixed files, read and hashed directly. GREEN rows are the same frozen selection minus158 deleted payload paths plus2 bundles and1 cfg(test) helper, and re-read all current bytes. Existing source cap512 and membership unchanged.

New accessor custody test (frozen before RED): `study_reference::document::fixture_literals::all_158_logical_members_retain_lengths_hashes_and_counts`. Iterates exactly158 strings, total1,042,985 bytes; no live work.

External existing literal fixture inventory, frozen before consumer filters:
- `results/validation/task54-d7-gate-schema-design/task-54-d7-gate-schema-vectors`: 137 files, 829473 bytes, max278638 bytes. Exact paths/hashes in task-54-d7-fixture-custody-external-literal-inventory.json.
- `results/validation/task54-d7-full-linked-oracle/task-54-d7-full-linked-oracle`: 318 files, 3704604 bytes, max1531337 bytes. Exact paths/hashes in task-54-d7-fixture-custody-external-literal-inventory.json.
