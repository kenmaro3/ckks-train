# Fixed preview candidate verification, 2026-11-08

These are Mac observations, not Linux CI/GPU measurements or scientific result JSON.
Candidate base: 00f4409c585fe3395b93aabffdbe020053fb01e1, isolated checkout/cache.

- Whole workspace cargo test --no-fail-fast: exit0, 731.299 seconds; 74 target
  summaries totaling 1,025 passed / 0 failed / 19 ignored, including doctests.
  Ignored tests were not executed or treated as passed.
- Strict Clippy all-targets: exit0, both default and CUDA features.
- Strict rustdoc default: exit101, unresolved plain.rs reboot_preprocess link.
- One-line doc-link patch: resolve to ckks_train_bench::data::reboot_preprocess;
  independent review PASS. No executable code changed.
- Strict rustdoc after that patch: exit0, both default and CUDA features.

Every execution directory records the original argv/cwd/environment overrides,
stdout, stderr, controller/child IDs and terminal result. Base executions were
clean. Patched doc executions explicitly record the uncommitted source diff and
base commit; they are not falsely labeled exact clean-base checks. Whole workspace
and Clippy successes apply to the base executable source, not the ongoing Task54 WIP.
No current Linux CI success is claimed: the latest GitHub run was prevented from
starting all6 jobs by payment/spending-limit conditions. Final package/public-history
review and full research gates remain separate requirements.

Earlier initial-candidate failure and the shared-fixture test correction are under
../research-preview-test-isolation/. The initial90851 source archive remains a
failed qualification attempt; these checks do not rewrite that history.
