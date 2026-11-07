# Publication mutation runtime1 independent outcome review

SPEC: **PASS**. QUALITY: **PASS**. EVIDENCE: **PASS**, limited to the released default-feature macOS host checks below. No concrete blocking or nonblocking issue was found in this bounded outcome review. This does not confer Linux native, durability, HE, or scientific validation.

## Scope and pinned inputs

Reviewed the runtime report, official release, controller, author work inventory, and accepted source-review boundaries. Reviewed all 25 launch/metadata pairs and all 50 raw stdout/stderr streams, plus launch, summary and tool-session records. Independently checked current source and fixed input hashes against the official release. The source implementation was not re-reviewed or rerun. One narrow source inspection resolved the intentional caught-panic diagnostic, described below.

No Cargo, test discovery, tests, project runtime, native probe, SSH, or subagent was invoked by this reviewer. Only this review report was written. The accepted source review remains a separate source-only result; the historical draft release was not substituted for the official release.

Paths below are relative to this ledger directory; hashes are SHA-256.

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `task-54-d7-publication-mutation-runtime1-report.md` | 2926 | `51b146256be4229496c37f3796fcb9384d443e0fc0f0f52ae7c748b33fa51c14` |
| `task-54-d7-publication-mutation-source-review.md` | 18882 | `42e2b711297904512d345dbf00e7492826461ae855edc7411c32490f5b74e2cc` |
| `task-54-d7-publication-mutation-runtime1-release.json` | 111153 | `1fad069562b8fa71e1157f32f8c151abe02f851b802ec6e5fa721c80ca5f0e9c` |
| `task-54-d7-publication-mutation-runtime1-controller.py` | 3659 | `e2144e21bb65a4ab019afed786f0825f35962b0f9bdcdfd4c44f8993791ff9bc` |
| `publication-mutation-source-author1/work-inventory.md` | 16927 | `ddb7cbd38baecc208829339ddccbc029da05d78fd42295ffb3c7793a7d3133c0` |
| `task-54-d7-publication-mutation-runtime1/summary.json` | 28079 | `beccee4db62728ab0586560552c2dab8a801ad1d3276b265423250557b1c7684` |
| `task-54-d7-publication-mutation-runtime1/tool-session.json` | 454 | `5fc89760d860d9dbdbe5de6a542b8585d4fdc3ef577c1cd94762c57fc45f0966` |
| `task-54-d7-publication-mutation-runtime1/launch.json` | 267 | `0f658689b9e95d479262bcb9ef68a4eb0633623fdb128bc1c96608307b9970dc` |

## Execution and custody checks

The official release has `released=true`, 25 expected tests, and 25 unique fully qualified selectors. Each recorded argv exactly matches the released selector and order:

```text
cargo test -p ckks-train-bench --lib <fully-qualified-selector> -- --exact --test-threads=1 --nocapture
```

There is no `--ignored`, `--list`, feature override, or broad module selector. Every raw stdout independently reports exactly the intended named success and `1 passed; 0 failed; 0 ignored; 0 measured; 254 filtered out`. All 25 child exit codes are 0, none timed out, and each metadata record agrees with summary and corresponding launch fields. The 254 filtered-out tests per invocation were not executed and are not counted as validation.

There are 25 distinct child PIDs with sequential, nonoverlapping recorded intervals, each below its released 60-second bound. The controller has one traversal of the release, fails on a pre-existing output directory, creates streams exclusively, and stops on failed selection, timeout, or nonzero exit without retry. Its timeout termination path was installed but not exercised. `CARGO_NET_OFFLINE=true` was recorded. The controller checks the macOS/default-target conditions before launch.

The retained tool-session record identifies original session **71344**, launch chunk `5b2b38`, intermediate chunk `3b6692`, final chunk `6e8e17`, controller terminal exit **0**, one launch invocation, and no automatic retry. This review checks those retained custody records and child evidence; it does not claim to have independently observed the original live terminal session. No retry or missing selected invocation appears in the recorded package.

The directory has exactly 103 expected files: 25 launch JSON, 25 metadata JSON, 50 raw streams, and the three package records. All 50 raw byte counts and hashes match metadata, totaling **9,950 bytes**. The summary is complete. All **646 current source pins**, totaling **10,105,354 bytes**, and all **52 fixed release input pins**, totaling **365,240 bytes**, match the official release. Controller source specifies source-pin checks at both endpoints and before/after each test: 52 full passes. That bookkeeping I/O is not scientific timing evidence. Raw stderr consistently identifies `target/debug/deps/ckks_train_bench-c3000792d73ef8e0`; this is an observed executable path, not an independently verified executable digest.

## Exact outcome inventory

For row NN, the evidence is `task-54-d7-publication-mutation-runtime1/NN.launch.json`, `NN.meta.json`, `NN.stdout`, and `NN.stderr`.

| Row | Exact selector | Passed / failed / ignored | Exit |
| --- | --- | --- | ---: |
| 01 | `study_provenance::publication::tests::anonymous_contract::anonymous_five_operations_preserve_visibility_boundary` | 1 / 0 / 0 | 0 |
| 02 | `study_provenance::publication::tests::anonymous_contract::anonymous_recovery_has_only_two_sync_operations` | 1 / 0 / 0 | 0 |
| 03 | `study_provenance::publication::tests::anonymous_contract::anonymous_owned_lifetime_model` | 1 / 0 / 0 | 0 |
| 04 | `study_provenance::publication::tests::anonymous_contract::anonymous_non_linux_refuses_without_creating_entries` | 1 / 0 / 0 | 0 |
| 05 | `study_provenance::publication::tests::every_publication_fault_withholds_capability_and_preserves_final` | 1 / 0 / 0 | 0 |
| 06 | `study_provenance::publication::tests::competing_publishers_never_replace_first_final` | 1 / 0 / 0 | 0 |
| 07 | `study_provenance::publication::tests::recovery_never_cleans_any_temporary_name` | 1 / 0 / 0 | 0 |
| 08 | `study_provenance::publication::tests::recovery_file_sync_failure_preserves_temporary_name` | 1 / 0 / 0 | 0 |
| 09 | `study_provenance::publication::tests::publication_mount_equality_is_not_an_ancestor_requirement` | 1 / 0 / 0 | 0 |
| 10 | `study_provenance::publication::tests::short_writes_and_eintr_are_retried_without_losing_bytes` | 1 / 0 / 0 | 0 |
| 11 | `study_provenance::publication::tests::concurrent_synthetic_publishers_get_exactly_one_capability` | 1 / 0 / 0 | 0 |
| 12 | `study_provenance::publication::tests::reader_guard_open_policy_is_derived_from_the_closed_role` | 1 / 0 / 0 | 0 |
| 13 | `study_provenance::publication::tests::fifo_and_nonregular_reads_fail_without_waiting` | 1 / 0 / 0 | 0 |
| 14 | `study_provenance::publication::tests::target_stamped_read_rejects_byte_identical_inode_replacement` | 1 / 0 / 0 | 0 |
| 15 | `study_provenance::publication::tests::target_unknown_entries_block_new_claim_without_removal` | 1 / 0 / 0 | 0 |
| 16 | `study_provenance::publication::tests::target_nofollow_regular_owner_mode_and_guard_link_count_checks` | 1 / 0 / 0 | 0 |
| 17 | `study_provenance::publication::tests::target_ancestor_and_profile_swaps_reject_through_held_parent` | 1 / 0 / 0 | 0 |
| 18 | `study_provenance::publication::tests::target_guard_rejects_nonempty_multiple_links_and_replaced_inode` | 1 / 0 / 0 | 0 |
| 19 | `study_provenance::publication::tests::target_recovery_preserves_linked_owned_orphan_after_postlink_failure` | 1 / 0 / 0 | 0 |
| 20 | `study_provenance::publication::tests::target_directory_creation_never_repairs_an_existing_unsafe_object` | 1 / 0 / 0 | 0 |
| 21 | `study_provenance::publication::tests::reader_guard_existing_children_never_create_or_repair` | 1 / 0 / 0 | 0 |
| 22 | `study_provenance::publication::tests::reader_guard_existing_guard_never_creates_repairs_or_follows_links` | 1 / 0 / 0 | 0 |
| 23 | `study_reference::target::tests::postlink_clock_or_context_failure_never_returns_even_synthetic_success` | 1 / 0 / 0 | 0 |
| 24 | `study_reference::target::tests::observed_clock_reversal_or_unavailability_is_sticky` | 1 / 0 / 0 | 0 |
| 25 | `study_reference::target::tests::clock_equalities_allowed_without_interpolation_or_tolerance` | 1 / 0 / 0 | 0 |

Rows 01–03 are the three new pure anonymous-publication models. Row 04 is the actual macOS/non-Linux refusal check. Rows 05–12 are eight retained pure checks, including row 11's two-worker fake/Barrier case. Rows 13–22 are ten portable filesystem checks with real test-owned temporary files/directories, FIFO, symlink/hardlink, replacements, metadata and synchronization operations. Rows 23–25 are three unchanged scripted outer clock/context checks. This matches the released **3 + 1 + 8 + 10 + 3 = 25** inventory; portable filesystem work is not described as zero-OS work.

Row 03 stderr contains `scripted pre-link unwind` at `crates/ckks-train-bench/src/study_provenance/publication_anonymous_tests.rs:882`. The named-risk source inspection at lines 859–896 confirms that the `unwind` branch intentionally panics inside `catch_unwind(AssertUnwindSafe(...))` and asserts that an unwind occurred. The raw named success, 1/0/0 counts and process exit 0 agree. This diagnostic is expected evidence of that branch, not an ignored test or an unhandled process failure. The review does not assert that all stderr was empty.

## Findings and remaining boundaries

No selector, count, exit, timeout, retry, stream-hash, source-pin, input-pin, or category discrepancy was found. No inventory correction is required for this runtime package. The source-review PASS is preserved as an accepted input, not regenerated by these outcomes.

The three pure models do not establish physical anonymous-inode/FD reclamation, close-error handling, or power-loss durability. The macOS refusal does not establish Linux `O_TMPFILE`, `openat2`, procfs-FD linking, mount support, or native recovery. All minimum 21 Linux-native cases and additional shared-caller/fork/guard/mount obligations remain unexecuted; an eventual ext4 result would not imply XFS coverage. Scripted outer checks do not establish genuine live authority or current-context behavior.

Actual retained readers, gate evaluation, custody, attempt protocol, consuming boundaries and the Full/Quick driver remain separate unfinished obligations. No key generation, CKKS/HE, CUDA runtime, server command, registered scientific experiment, throughput, precision, memory-fit or kernel-durability result follows from this package. The original **21 profiles / Full 2680 + 20 + 7 / Quick 268**, C/R/A/K, scientific thresholds, baseline/Task 5 obligations, approvals and existing holds remain intact. This outcome review supplies evidence only for the 25 released host selectors and creates no broader release or bypass.
