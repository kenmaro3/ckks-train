# Task 54 P1b prospective Linux support gate

Date: 2026-11-06. Status: **independently reviewed and root-ratified DESIGN ONLY.**
The current implementation remains Apple-only. This document is not a Linux source
candidate, execution result, source assignment or Linux support/admission claim.

## Decision

Recommend **one explicitly reviewed source-candidate commit, followed by actual Linux
positive validation before a support/admission claim**. Reuse the existing public
`DftSupportProof::verify_current()` and receipt constructors without a second verifier,
test-only trusted token, bypass feature, attestation registry or new dependency.

P1b `090f8fb` correctly limited production proof to `aarch64-apple-darwin`; fix1 fixed
Linux test selection without widening that domain. This proposal is a new, separately
ratified extension to **only `x86_64-unknown-linux-gnu` with 64-bit usize**. Its purpose
is eventually to unblock actual structural metadata for the original quadruped-gpu
Linux CPU matrix, not merely to make CI green. Mac N=1024 calibration at `9610822`
(84/84 primary cells, as reported in its separate reviewed record) is finite Mac
calibration evidence and establishes no Linux support premise.

The candidate may execute the normal process-local verifier on that exact Linux
target; it is nevertheless marked **Linux validation pending** until the evidence
gate below passes. This separates a reviewable implementation from a tested-support
claim. Native Mac tests are not represented as Linux execution. A proof token still
requires actual current-process checks, even after Linux validation is recorded.

## Exact proposed source/test delta

1. `crates/ckks-train-ckks/src/bootstrap/metadata.rs`: change only the exact target
   allowlist in `check_proof_environment` to Apple above OR Linux above. Preserve
   pointer-width-first and required nonempty TARGET/compiler/HOST checks and their
   named errors. Do not interpret arbitrary `linux`, `gnu`, compiler text or HOST text
   as sufficient numerical proof; HOST is build-host diagnostic, not runtime identity.
   Keep `verify_current` scalar loop, IEEE probes, norm/residual checks and token fields
   unchanged. Update its scope comment to distinguish candidate from tested evidence.
2. In that file's supported-verifier unit test and **every** supported import/helper/
   test in `tests/bootstrap_metadata.rs`, extend the cfg to the union of existing
   Apple64 and `all(target_arch="x86_64", target_vendor="unknown", target_os="linux",
   target_env="gnu", target_pointer_width="64")`. The public unsupported-target test
   uses the exact complementary predicate; it remains mandatory elsewhere. Do not
   make positives `ignored`, tolerate an error, or count a filtered-out test as PASS.
   The unconditional quantized-zero fixture and generic tests remain unconditional.
3. Add table-driven **eligibility-only** unit coverage: both exact names accepted by
   the pure diagnostic guard; `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-gnu`,
   `x86_64-pc-windows-gnu`, an unknown target rejected; 32-bit and absent/empty each
   required diagnostic rejected with exact requirement/reason. Synthetic strings
   exercise policy only and never construct a token. Rename the existing guard-test
   wording if needed to avoid implying that this helper verifies arithmetic.
4. Extend the existing supported public-verifier test to assert the actual configured
   target matches its cfg branch and print compiler `-vV`, TARGET, HOST, root count,
   root and inverse squared-norm extrema. It must still call the real verifier first.
   No expected transcendental bit patterns, reduced root domain or looser tolerance.
5. `build.rs` needs **no change**: it already captures Cargo-selected compiler, TARGET
   and HOST, explicit unavailable values, safe escaping and rerun dependencies. Its
   strings are diagnostics, not wrapper/libm/executable attestation. No schema or
   arithmetic-proof revision bump is needed for unchanged proof/identity encoding;
   the reviewed candidate commit identifies the allowlist policy change, and existing
   descriptor bytes already bind actual target/compiler/host and observed proof data.

Prospective ownership: these two Rust paths, a narrowly scoped focused-validation step
in `.github/workflows/ci.yml` if needed for visible evidence, and a status-only appendix
to `docs/bootstrap-metadata-design.md` after ratification. No linear/poly/EvalMod,
build.rs, dependency, NN, bench, preset, scale-search or server-script edits are needed.
Public documentation must retain the historical fix1 scope and add a dated candidate
status, not rewrite the previous rejection as a mistake.

## Proof retained, without stronger claims

Source review must reconfirm the existing `linear::root`, `pow_mod`, inverse expression
and pinned num-complex multiplication/division order; ordinary binary64 source reasoning
also applies to the proposed 64-bit target under the same FP-environment assumptions.
The actual public gate streams all **65,535** roots/inverses using that implementation.
Keep squared-norm root `[1/2,2]`, inverse `[1/32,2]`, residual `1e-28`, ordinary-FP probes,
and the existing unique-path/8u normwise bound unchanged. Each descriptor still checks
its actual folded scalars (including powf output), at least `2^-66`, preserving the
existing loose nonzero witness bound `2^-120`. Neither target names nor passing finite
samples alone replace the source lemma. No universal libm accuracy is asserted.

## Native RED/GREEN and actual Linux gate

**A. After explicit source release, on native Mac:** add the eligibility test first;
its Linux-success expectation must fail against the Apple-only guard (RED), then pass
after the exact guard change (GREEN). Record the selected test name, assertion, actual
native compiler/host, exit codes and patch identity. Run the focused metadata unit and
integration suites for Apple regression. This proves the new policy and retained Apple
behavior only; no fake target cfg, cross-compile or host-side test is Linux evidence.

**B. Review and freeze candidate:** independent review checks the full delta, guards,
source lemma, unchanged thresholds, call graph and cfg complement. Freeze exact commit
and path hashes while status is still Linux-validation-pending. Releasing that candidate
to an already authorized CI surface is distinct from claiming support. No receipt from
an unreviewed or different source revision closes this gate.

**C. Actual Linux execution:** use the repository's existing Ubuntu CI surface, currently
`ubuntu-latest` with Rust `1.85`, or an already available native Linux surface approved
for these focused commands. Do not assume the runner's actual architecture, rustc version
or target from the YAML label. Add focused, named steps before the existing workspace
step if necessary to retain positive evidence and diagnostics; leave unrelated workflow
behavior unchanged. The gate itself invokes only:

```text
cargo test --locked -p ckks-train-ckks --lib metadata -- --test-threads=1 --nocapture
cargo test --locked -p ckks-train-ckks --test bootstrap_metadata -- --test-threads=1 --nocapture
```

Use `CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2`. Current test profile
uses opt-level=3, with test debug assertions/overflow checks; record this profile.
Before claiming a separately compiled release executable is covered, run the same
focused pair with `--release` on the actual Linux surface and record that profile too.
Do not use an unqualified EvalMod filter or launch the whole workspace/secure-size HE
suite as this bounded gate. Existing broader CI policy is separate authorization and
evidence; a green aggregate workflow without the positive test names is insufficient.

Retain all **nine current integration tests** (eight newly Linux-positive plus the
unconditional fixture), not a new reduced smoke substitute:

- default n16 receipt: exact tuple `(69,40,46,343,56,2,1)`, 111 switches, level16/Delta,
  32 rotation keys and existing exact stage-shape/identity assertions; metadata only;
- every small grouping (log_slots1..6), both directions: independent connectivity and
  Minkowski supports plus actual DFT zero support;
- small runtime Some/None, scale bits, keys, SSE/no-SSE, r=0/r>0, odd/even degree and
  unequal groupings; observed bootstrap all-seven-counter/output equality;
- malformed sizes/weights/packing/budgets, content identities, quantized-zero Some,
  stage-conjugation accounting and maximum operational scalar budget.

The library selection must include the actual finite scalar verifier, folded extremes,
generic error/counter/identity checks and the new eligibility cases. Preserve exact
existing local-runtime comparisons: Linux-produced coefficients/scale bits are compared
with the same Linux runtime, not forced to equal Mac transcendental bytes. Fixed
structural literals must still pass unchanged; a mismatch stops admission and is
investigated, never hidden by accepting errors or changing tolerances.

Evidence bundle must contain exact candidate SHA/path hashes and checkout identity,
CI run/job URL and conclusion (or local process exit), actual test names/counts/failures,
commands/profile/env, `rustc -vV`/Cargo version, token TARGET/HOST/compiler diagnostics,
runtime `uname -a`/architecture, `/etc/os-release`, runner image identifier if supplied,
and available glibc version diagnostic (`getconf GNU_LIBC_VERSION`). Record normal
compiler flags/wrapper presence and any overrides without claiming complete attestation.
Store root count/extrema and metadata requested-byte/work output. Missing environment
information is marked unavailable; absent essential exact-target/compiler/run evidence
leaves the validation gate open. Logs and results must be from the same frozen candidate.

## Bounds, failure handling and next scope

This is one focused pass per required profile, no benchmark/matrix sweep. Verifier:
65,535 scalar iterations and O(1) storage per invocation. Existing tests keep actual
DFT matrices at <=64 slots and runtime HE fixtures at N=32; n16 cases construct only
parameters/metadata. Metadata caps remain 4096 coefficients, `2^24` interpolation and
polynomial work budgets and 64MiB requested storage per receipt (not process RSS).
Use a 15-minute focused-step timeout; timeout/OOM is incomplete evidence, never PASS.
No secure N=2^16 key generation, GPU, SSH, server job alteration or calibration run.

Any runtime proof failure returns the existing named MissingCode; folded/scale/budget
failures still produce no receipt. No retry fallback grants a token. Failed Linux CI
leaves support unvalidated; root may fix a specifically reviewed cause or revert only
the candidate extension to the Apple-only guard/cfg, preserving all failure logs and
the unrelated fix1 correction. Do not retain a broad claim merely because some Linux
tests or the pure guard passed.

After successful independent evidence review, public status may name **validated
conditional structural metadata on the recorded x86_64 GNU/Linux environment/profile**.
It still cannot say full bootstrap admitted, every Linux/toolchain supported or server
ready. The eventual quadruped-gpu executable must call the same verifier in its own
process, record its own environment, and construct the actual registered descriptors;
CI does not issue it a transferable token. That future server step requires its own
operational release and must retain the full registered matrix scope.

Encoding Q/2 feasibility, internal EvalMod scale feasibility, encrypted range, precision,
actual key/secret binding, security, runtime memory, backend and full Task54 matrix
admission remain separate. Linux structural support closes only this missing-proof
dependency. A two-stage scalar probe would repeat or refactor an already available
mandatory verifier and would still need the same actual metadata/runtime tests later;
it adds no useful gate here. Keeping Linux rejected forever avoids a candidate but
cannot satisfy the original Linux CPU matrix. A target-only allowlist or test-only
token shortcut would fail the existing proof contract and is rejected.

## Read-only basis and freeze

Read actual `docs/bootstrap-metadata-design.md`, metadata.rs/build.rs/integration tests,
relevant linear helpers, `.github/workflows/ci.yml`, Cargo profiles/toolchain file,
`task-54-p1b-api-proposal.md` and `task-54-p1b-fix1-rereview.md`. No current CI outcome
was queried or inferred. This design has received independent review and root ratification. Source assignment
and the specified actual Linux execution/evidence review remain separate future gates.

## Publication identity

Reviewed proposal SHA256: `43d3d9da2f520eb1c18a0747a7b11575d606d77e654dfc46608be8b4d5a30cbc`. Independent design review SHA256: `40407b8c5b8d6c3a08c7edf7db768ec8f85a6f04b8af4c98333c66ff2b86d05d`. The public copy changes only status/ownership/handoff language and adds this identity record. No normative guard, test, resource, evidence or failure rule is changed.
