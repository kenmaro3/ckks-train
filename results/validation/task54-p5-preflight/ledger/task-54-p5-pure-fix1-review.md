# Task54 P5 pure source fix1 re-review

2026-11-07. **S1 addressed. SPEC: PASS for the scoped source package. QUALITY: PASS.** No new finding in this delta. Pure sensitivity and paired-oracle runtime behavior remain unexecuted; this review grants no runtime/HE release.

Reviewed original S1, exact fix1 release, complete source delta, appended report, static invariance/custody and final two-source pins. Independently compared preserved/current source segments rather than relying only on the invariance JSON. Only reads/hashes/text comparisons and this review-file write occurred; no tests, Cargo, fixture/Engine/forward/HE, source edits, network, Git mutations or subagents.

## Frozen evidence

Verified:

- Fix1 release SHA256 `2568ce68ad876bdc16773e03038ff874f5fe4766b1686ba3d241cc2435531910`.
- Appended source report SHA256 `be73160724d1af62ebe290bb219111e4eac8137f78976d12df61267f2e0e52dc`.
- Exact delta SHA256 `8406cb57676d26479feb75ab20b8292ad0dcd97f0fb193dcd8328ef8b03357e7`.
- Pins JSON SHA256 `a8e0a4c88e4409b0b49579a54ed65c4558c6761af24edeea770c37b0bc2c258d`.
- Current parent `study_plain.rs` remains `fc6c60bde0b8ec28d2fd8b69abe79809a3f36b0d6986dea3c14a3a1227fd160a`; current fixture is `fbf604ddc4a3dfae6f3c37bd6909d35adee8f83f639453fb97239bbd46bb2943`. Both match reviewed pins and final strict Clippy/no-run before/after hashes.
- `05-clippy-final` and `06-compile-final` exit0 with clean raw outputs and the exact authorized strict Clippy/focused no-run forms. The retained initial compile RED genuinely reports missing PairedOraclePhase; it is compile evidence only. Neither numerical nor zero-numerical ignored test was executed.

## S1 resolution

The old free `block_oracle(family)` helper is removed. The only future oracle seam is now mutable borrowing of one `PairedOraclePhase`, whose private persistent Report uses fixed paired limits and retains counters/failures across calls. It exposes no caller-set limits, reset, mutable Report, Clone or Default path. Construction is ordinary test ownership, not readiness/start authority; the later reviewed HE driver must own exactly one phase.

`oracle(family)` checks/logs its14-call cap before recipe/baseline work. Existing actual Engine literal, Engine::train and external forward helper sites are unchanged and reserve against that same persistent Report immediately before each actual call. Their limits are14/40/128, respectively. Consequently, an extra oracle or excess real helper operation is stopped at its own site; no planned trajectory total is substituted for actual calls.

`actual_counts()` exposes a read-only `[oracles, engines, train, forward]` snapshot. `check_complete()` requires exactly `[14,14,40,128]` and no accumulated failures. A failed range/state/arithmetic outcome survives into later calls/completeness rather than being reset. The closed recipe source, baseline arithmetic and all error-return semantics remain unchanged.

The stated paired counts remain correct source-derived future totals for4A+4B+4C+2D:14Engines,16+16+4+4=40actual baseline train calls, and40+60+16+12=128external baseline forwards. They are unmeasured bounds, not additional authorized numerical work. Exact future family/session assembly and single-phase ownership still belong to the later HE driver review.

## Pure-path preservation and new-test boundary

Independent source comparisons confirm byte-identical recipe/rows, actual Engine/train/forward/baseline bodies, local/schedule/target controls and comparisons. The entire pure matrix differs only in replacing `Report::default()` with the closed `Report::pure()` constructor; that constructor preserves zero counters/failures and the original156/61/179 limits. Reserve still increments/logs before calls, now reading the chosen private limit field. No arrays, rates, thresholds, controls, first-affected logic or margin semantics changed.

Thus the previously reviewed pure inventory remains156train/61Engine/179external forwards,40control rows/72I-NI margins, at fixed>=8x I1e-5/NI4e-4. No case or expensive operation was added to the numerical pure entrypoint.

The new ignored guard specification constructs only an empty phase and asserts zero counts/incomplete status. It invokes no fixture, Engine or forward and has only been compiled. Exact selection of the existing numerical pure test will not execute this separate guard test; later raw filtered-test totals may reflect its addition. No runtime guard PASS or numerical result is inferred from compiling it.

## Disposition

S1's shared actual-call phase boundary is now present and the complete scoped source package has no remaining review blocker. Root can separately freeze/authorize a one-shot pure sensitivity run. Paired HE execution remains a later reviewed phase, with exact driver/session ownership and budget custody still required. Numerical margins/ranges/target isolation, canonical/D7/security/resource/scientific/campaign acceptance and full P5/Task54 completion remain unclaimed.
