# Task 54 P1b GNU/Linux focused validation evidence

Date: 2026-11-06. **Independent evidence review passed; validated conditional structural metadata on the recorded GNU x86_64 Linux64 environment in test and release profiles.**
This bundle does not approve itself or close the public support gate. Historical
Apple-only and candidate-pending records remain valid for their original scope.

## Provenance and exact scope

Candidate commit: `23ac79d5332086e6237f9d6b548d084417a5c376`.
[CI run](https://github.com/kenmaro3/ckks-train/actions/runs/42230241832),
[exact test job](https://github.com/kenmaro3/ckks-train/actions/runs/42230241832/job/127933965036).
The five `.log` files are byte-exact copies of the same-run uploaded artifact
`bootstrap-linux-evidence-42230241832-1`, artifact ID `13507907310`.
The external API reports the artifact's head SHA as this candidate, archive size
7,429 bytes, archive digest `sha256:fb9756a605945d107338b382955500783e2228ab958e8802be2476f368757789`,
created `2026-11-06T14:04:58Z`; the reported expiry is `2026-11-20T14:04:57Z`.
That archive digest is external API metadata; this package's SHA256SUMS verifies the
individual extracted/copied files, not a retained ZIP archive.

The three `ci-*.txt` files are **byte-exact copies of existing CI collection/API
metadata JSON**, preserved under text extensions for provenance. They are external
CI metadata, not authored scientific result JSON. No API query, download or test
was performed while assembling this package. The original private ledger freeze
contains eleven hashed collection files; all eleven hashes were checked before copying.

The frozen API observation at 2026-11-06T14:05:17.013844+00:00 records the diagnostics, focused test
profile, focused release profile and artifact preservation steps as `success`.
The broader `cargo test --workspace` step, test job and aggregate run were still
`in_progress`, without a conclusion. This bundle claims only the focused steps;
it does not claim whole-job or whole-CI success. It is a historical observation,
not a live status query. See [raw observation](ci-job-status-observation.txt).

[diagnostics.log](diagnostics.log) reports the candidate HEAD and all five
candidate path hashes. It also reports `?? bootstrap-linux-evidence/`: tee had
created the evidence directory. A globally clean checkout is **not** claimed.
The three candidate code/workflow paths still match those diagnostic hashes;
public documentation subsequently gains dated evidence appendices.

```text
3bf68915b5f70e9e2ae3405bb011196f919631a7651b1829a636b5ca4b768ae8  crates/ckks-train-ckks/src/bootstrap/metadata.rs
b2864ecb906ffa73e07a63ff2f9c6b8b8f30428cc578fcacf4b99845c17d9cad  crates/ckks-train-ckks/tests/bootstrap_metadata.rs
3223532b9f0ddac52bd1b92fe1cf077548c05bcd482e96987b361e2d54e19507  .github/workflows/ci.yml
2d41635f6665f37d6376835d4c220e03b7a1ca84e858969c1b5a8ed1482a0cd9  docs/bootstrap-metadata-design.md
04a7f7df19ab034ad2d134bc801beb1e30f8aef13c5ccbfa9548db6d597ea190  docs/bootstrap-linux-validation-plan.md
```

## Actual environment, commands and profiles

Actual configured TARGET and HOST(build) are both `x86_64-unknown-linux-gnu`.
Runtime uname independently reports x86_64 GNU/Linux, kernel `7.0.0-1017-azure`
on runner `runnervm7aw5e`; Ubuntu `24.04.5 LTS`, glibc `2.39`.
Runner image: `ImageOS=ubuntu24`, `ImageVersion=20261101.360.1`,
`RUNNER_OS=Linux`, `RUNNER_ARCH=X64`.

Compiler: rustc `1.85.1 (4eb161250 2025-03-15)`, commit
`4eb161250e340c8f48f66e2b929ef4a5bed7c181`, LLVM `19.1.7`;
Cargo `1.85.1 (d73d2caf9 2024-12-31)`. Both focused library logs independently
print the actual verifier token's configured compiler, TARGET and HOST(build).
HOST is a build diagnostic, not runtime-host attestation. Rustup's diagnostic
notes the missing 1.85 toolchain was auto-installed in CI; this assembly installed
nothing.

Each profile used `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2`,
harness threads=1, and a 15-minute profile timeout. The workflow's exact commands:

```text
cargo test --locked -p ckks-train-ckks --lib metadata -- --test-threads=1 --nocapture
cargo test --locked -p ckks-train-ckks --test bootstrap_metadata -- --test-threads=1 --nocapture
cargo test --locked --release -p ckks-train-ckks --lib metadata -- --test-threads=1 --nocapture
cargo test --locked --release -p ckks-train-ckks --test bootstrap_metadata -- --test-threads=1 --nocapture
```

Test is `optimized + debuginfo`, repository opt-level=3 with default test debug
assertions/overflow checks enabled. Release is separately compiled `optimized`,
Cargo release defaults (opt-level=3, debug assertions/overflow checks disabled).
Named compiler/wrapper/rustflags/build-target/target-linker/test-profile/release-profile
override variables are all explicitly `<unset>` in diagnostics. Every listed possible
Cargo config/config.toml path is absent. This records ordinary build configuration;
it does not attest the executable, wrappers, runtime FP-state or loaded libm.

## Actual positive outcomes in each profile

| Profile | Library | Integration | Ignored / failed | Filtered library / integration |
| --- | --- | --- | --- | --- |
| test | 13 passed | 9 passed | 0 / 0 | 159 / 0 |
| release | 13 passed | 9 passed | 0 / 0 | 159 / 0 |

The exact same positive names below occur in both profiles. These are actual
executions, not zero-test cfg success or eligibility-only substitution. The
focused step success and fail-propagating workflow pair establish success of
both commands; raw Cargo per-command exit markers were not emitted in artifacts.

Library names (13):

- `bootstrap::evalmod::metadata_target_tests::boundary_search_preserves_nearest_order_and_context`
- `bootstrap::metadata::proof_failure_tests::checked_stage_scales_and_folded_values_reject_before_encoding`
- `bootstrap::metadata::proof_failure_tests::exact_targets_are_eligible_for_current_process_verification`
- `bootstrap::metadata::proof_failure_tests::ineligible_diagnostics_return_exact_missing_requirement_and_reason`
- `bootstrap::metadata::tests::checked_counter_overflow_covers_all_seven_fields`
- `bootstrap::metadata::tests::finite_scalar_proof_on_supported_target`
- `bootstrap::metadata::tests::folded_extremes_and_complex_support_operations`
- `bootstrap::metadata::tests::parameter_bytes_bind_every_field_even_if_changed_value_is_invalid`
- `bootstrap::metadata::tests::scale_error_guards_precede_legacy_allocating_errors`
- `bootstrap::poly::metadata_cost_tests::checked_split_rejects_actual_nonfinite_intermediates`
- `bootstrap::poly::metadata_cost_tests::zero_sensitive_split_and_declared_length_literals`
- `linalg::bootstrap_metadata_iterator_tests::iterator_preserves_full_shape_order_and_wrap_literals`
- `linalg::bootstrap_metadata_iterator_tests::iterators_validate_eagerly_in_legacy_panic_order`

Integration names (9):

- `default_n16_structural_receipt_without_he`
- `every_small_group_matches_connectivity_minkowski_and_actual_dft`
- `identities_bind_contents_not_arc_or_presets`
- `key_requirements_carry_parameter_binding_and_max_scalar_budget_is_operational`
- `malformed_sizes_weights_packing_and_budgets_fail_without_panics`
- `nonzero_unencoded_diagonal_remains_some_when_plaintext_rounds_to_zero`
- `small_observed_bootstrap_counter_equals_structural_receipt`
- `small_runtime_receipts_match_encoding_scales_keys_and_all_counters`
- `stage_cost_places_final_ctos_conjugation_only_in_direction_total`

The `iterators_validate_eagerly_in_legacy_panic_order` fixture intentionally
catches invalid-input panics. With `--nocapture`, the panic hook prints ten
messages before that test finishes `ok`; both library summaries are 13 passed,
zero failed. These hook messages are not failed tests.

In **each** library profile the actual current-process verifier streamed 65,535
roots/inverses and reported root squared-norm extrema
`(0.9999999999999998, 1.0000000000000002)` and inverse extrema
`(0.2499999999999999, 0.2500000000000001)`.
Each integration profile reports default metadata requested bytes `318806`, work
`32768`. The former native Mac record was `318794`: the 12-byte difference is
platform diagnostic text retained in descriptor identity (compiler/TARGET/HOST),
not RSS, key memory or a regression in numerical tolerances. Descriptor identity
is allowed to vary with platform/transcendental diagnostics; fixed structural
literals and local runtime comparisons passed unchanged.

## Conditional proof boundary and remaining gates

The reviewed source lemma and current-process checks remain jointly required:
existing `linear::root`/`pow_mod`/inverse operation order and pinned num-complex
arithmetic, ordinary binary64/FP-environment assumptions, unique path and 8u
normwise bound, root `[1/2,2]` / inverse `[1/32,2]` squared-norm gates, residual
`1e-28`, full 65,535-root domain, descriptor folded scalar checks `>=2^-66`, and
loose nonzero witness `2^-120`. Target names and finite test samples alone do not
replace that source argument or assert universal libm accuracy. No source,
threshold, private token or schema was changed in this evidence package.

After independent evidence review, the supported statement is limited to
**validated conditional structural metadata on this recorded GNU x86_64 Linux64
environment in the test and release profiles**. It does not cover all Linux
systems/toolchains or constitute full bootstrap/matrix/server admission.
Encoding Q/2 feasibility, internal EvalMod scale feasibility, encrypted range,
precision, actual keys/realized secret binding, security, runtime memory/backend
and the full registered Task 54 matrix remain separate gates. The eventual
server executable must call the same verifier in its own current process and
construct its actual registered descriptors under separate operational release;
CI cannot issue it a transferable proof token. Small N=32 runtime fixtures and
<=64-slot DFT comparisons are the existing bounded focused scope; default n16
checks construct metadata without secure-size key generation.

Package validation was byte/hash equality, exact positive-name/count comparison,
relative-link checks and append-only document diff checks. No Cargo/HE/SSH/server
or full-matrix run was needed or performed for this documentation-only package.
