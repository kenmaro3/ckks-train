# FIX1 explicit native selection inventory

This is an append-only successor inventory, not runtime authorization. Original `publication-mutation-source-author1/work-inventory.md`, its48-filter inventory, original28logs and runtime1 remain unchanged. All source test names/case bodies remain except the strictly described Linux child harness changes. Native21 cases, shared24+24+4+8+260 cases, primitive controls and legacy tests are retained, with original payload/allocation/actor/process budgets. No new scientific constructor, data input, fixture byte, filesystem operation or child launch is added.

## Source selection

48 functions remain in the two publication test files.23 are now explicit opt-in (21 parent entrypoints +2 child-only entrypoints);25 functions have no ignore attribute. Conditional compilation means those25 are not all present on one host:24 default-selected functions on Linux,23 on non-Linux, by source enumeration. Those local counts do not include the three unchanged outer Target tests selected in the historical25-Mac package and are not runtime --list results. None of the25 historical Mac-selected functions is ignored or changed. The ignored set is exact below; every other function keeps its prior attributes/body.

- `study_provenance::publication::tests::target_native_engine_preserves_unknown_temp_and_does_not_overwrite` — native parent requires separate release.
- `study_provenance::publication::tests::target_actor_sentinel_replacement_cannot_redirect_owned_inode` — native parent requires separate release.
- `study_provenance::publication::tests::target_independent_thread_handles_publish_exactly_once_under_guard` — native parent requires separate release.
- `study_provenance::publication::tests::target_native_linux_publication_child` — internal child only; not independently released.
- `study_provenance::publication::tests::target_native_linux_processes_publish_exactly_once` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::publication_native_anonymous_success` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_no_clobber` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_postcheck_windows` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_legacy_prefix` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_drop` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_child` — internal child only; not independently released.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_visible_uncertain` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_proc_refusal` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::publication_native_final_swap` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::shared_anonymous_publication_faults` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::shared_anonymous_late_confirmation_faults` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::shared_anonymous_no_clobber` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::shared_anonymous_postcheck_windows` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::shared_legacy_temp_preservation` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::anonymous_create_substep_failures` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::anonymous_create_refusal` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::proc_capture_refusal` — native parent requires separate release.
- `study_provenance::publication::tests::anonymous_contract::linux::anonymous_success_records_five_operations_and_eight_create_substeps` — native parent requires separate release.

## Reachability and child verification

Every direct `SyntheticDirectory::native` user is opted in. All17 tests within `anonymous_contract::linux` plus the Linux `publication_native_anonymous_success` are opted in. In the old parent test module, the4 direct scratch users (engine, actor-sentinel, independent-thread publisher, separate-process publishers) plus their publication child are opted in. The transitive `synthetic_guarded_publish` helper remains private and has only the opted-in thread publisher and child callers. The non-Linux actual refusal test still runs normally; it uses `SyntheticDirectory::new` and cannot publish.

| Parent | Exact child filter | Child argv flags | Required result |
|---|---|---|---|
| `anonymous_contract::linux::publication_native_drop` via run_child(before-link) | `study_provenance::publication::tests::anonymous_contract::linux::publication_native_child` | --exact --ignored --test-threads=1 | exit70 only, reached real C/W/F before process exit |
| `anonymous_contract::linux::publication_native_visible_uncertain` via run_child(after-link) | same | same | exit71 only, reached real C/W/F/L before process exit |
| `target_native_linux_processes_publish_exactly_once` | `study_provenance::publication::tests::target_native_linux_publication_child` | --exact --ignored --test-threads=1 | exactly one exit70(publication won) and one exit71(rejected); exit0 fails |

Previously the anonymous child used exit0; it now uses closed70/71 cut outcomes. Previously the guarded publication child used0/10; it now uses70/71. These are private harness control outcomes, not durability/live-gate receipts. Parent cut values are validated before spawn; child validates the same closed values before touching its File. Missing child path/cut variables fail instead of returning success. Both parent flows reject exit0, so a typo/wrong harness selection yielding zero tests cannot be mistaken for child execution. Existing child counts/timeouts/cleanup scope are unchanged: two sequential single-child cases in the21 package, and the separately inventoried old two-child guarded-publication test. No new output/marker file or IPC operation is needed.

The existing guard-only `target_native_linux_guard_child` / `target_native_linux_separate_process_uses_fresh_flock_description` pair is unchanged and not newly ignored: it uses portable `SyntheticDirectory::new`/fresh_guard, never Native::create or the supported-mount scratch helper. Its existing child-acquired marker assertion already rejects a zero-test parent success. It remains a separate existing guard obligation and is outside the migrated anonymous-publication reachability set. Likewise ordinary portable filesystem and pure tests are not broadly ignored.

## Future explicit invocation, after a separate native release

Use one exact **parent** filter from the inventory, with root-selected already-authorized scratch:

```text
HEKIT_PUBLICATION_NATIVE_SCRATCH=<exact-private-released-supported-mount-path> cargo test -p ckks-train-bench --lib <exact-parent-filter> -- --exact --ignored --test-threads=1 --nocapture
```

This is a required invocation form, not a command executed or authorization supplied by this report. Root/controller must still verify the exact named parent ran with1 passed/0 failed/0 ignored, source/test-binary/input pins, actual mount/capabilities, bounded child/process behavior and cleanup ownership. Child-only filters must be reached through their listed parent, with parent-owned fixture path/cut environment; their intentional nonzero outcomes are interpreted only by that parent. Do not use broad --ignored/--include-ignored selection or construct CI scratch/environment as a fallback. Even if scratch is inherited accidentally, default `cargo test --workspace` does not select these ignored tests.

## Historical host applicability

`mac25-applicability.json` lists all25 old exact selectors and unchanged function-body hashes. The22 publication functions are byte-identical in their bodies and retain ordinary selection; the3 Target source functions' containing source file matches the official runtime1 release pin. All five non-test production/dependency files from the7-file baseline are byte-identical; test helpers used by these25 (Fake, FiveOps, OwnedModel, portable SyntheticDirectory/new/Drop, read/recovery/guard/clock helpers and test_hooks) are unchanged. Full scoped diff confines body edits to Linux-only `run_child`, `publication_native_child`, `target_native_linux_publication_child` and its process parent; all other edits are23 Linux-only ignore attributes.

Prior source42e2b711 and25-host outcome8d08d149 remain historical bounded PASS. New whole-file source hashes and rebuilt host binary were not executed by runtime1; neither its release JSON nor records are rewritten. FIX1 executes no tests or --list. It supplies static selection and host-path applicability evidence, not a new25-pass claim or Linux native/CI runtime result.
