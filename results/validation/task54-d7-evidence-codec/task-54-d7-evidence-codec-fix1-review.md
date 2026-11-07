# Task54 D7 evidence/Quick codec FIX1 review

## SPEC verdict: PASS — I1 and I2 ADDRESSED

## QUALITY verdict: APPROVED for the scoped source/static FIX1

Both prior Important findings are addressed. No new breakage was identified in the scoped changes. This verdict approves the correction package for root's subsequent exact runtime release; it does **not** report runtime success or authorize a broader filter. Reviewer Cargo/tests/`--list`/SSH/native execution counts are all **0**.

## Reviewed scope and evidence pins

Read the complete 58,354-byte scoped diff once against `evidence-codec-before-fix1`, including the appended report, versioned inventories and all four covering logs. This was an I1/I2 re-review, not a second full source review. Only `crates/ckks-train-bench/src/study_reference/document/evidence_tests.rs` changes source; production modules, Cargo and the existing P3 regression remain unchanged.

Independently verified:

- Scoped diff SHA256 `62c015088b6a609a89cc9cad10728fb86b6b4ff6b01a15785653abf0ed402835`.
- Full amended report SHA256 `c86f8990501d19d91055f59056909d27e4cabff838e46612209909d97486ec3d`; its original 25,484-byte prefix remains SHA256 `9a200eb062a58629ccc4bdf62fc4c78faf8670298adb53dfd3c0501db796631c`.
- Final source manifest SHA256 `87cb90f1fda25a52c83c7f95f628dab483d0debb128b9394c6c8c8d8758f89a9`; all ten current source byte lengths and hashes match it.
- All ten artifacts named by `fix1/artifact-sha256-fix1.json` match their pins; all nine transitive source pins in `fix1/p3-tiny-transitive-inventory.json` match the current files.
- Changed test source: 36,987 bytes, SHA256 `430a84b108899f027feed6f4edee80762f3460b7c541954e251416960d678ac2`.

Root's preservation report records all 26 earlier validation files and all 218 frozen inputs unchanged. This scoped re-review independently checks the original report prefix and current source/artifact pins above; it does not repeat the prior full 218-file audit.

## I1 — ADDRESSED: constructor-bearing shared test is accurately classified

The amended author report at `task-54-d7-evidence-codec-report.md:169–211` explicitly supersedes the historical all-zero claim, retains the necessary exact P3 test and separates its release class. The new `evidence-codec-source-validation/fix1/proposed-shared-regressions-fix1.json:109–144` removes that test's zero-cost classification, records `forbidden_runtime_calls_all_zero=false`, and identifies 15 parameter-free shared filters plus one constructor-bearing filter:

`study_admission::tests::p3_digest_golden_and_mutations`.

The separate `fix1/p3-tiny-transitive-inventory.json` correctly describes one `CkksParams::test_tiny/new`, N=32, q12/p1, six deterministic prime pools, two RnsBasis allocations, 13 Modulus/NTT-table constructors, 26 vectors of 32 Shoup entries, 26 additional scalar Shoup fields, and three CtMeta constructors. It distinguishes this preparation from forward/inverse transforms, RNG and HE, each of which remains zero for this test. It does not invent exact candidate-search counts, allocation capacities, heap-call counts, runtime or RSS.

Named outside-diff risk checked: whether the corrected transitive cost still hides constructors or confuses table construction with transform execution. Inspected only the relevant existing constructors/helpers: `crates/ckks-train-ckks/src/params.rs:87–159,203–209`; `crates/ckks-train-core/src/rns.rs:64–100`; `crates/ckks-train-core/src/ntt.rs:56–131`; `crates/ckks-train-core/src/primes.rs:23–117`; `crates/ckks-train-core/src/modulus.rs:10–21,50–67,127–149`; `crates/ckks-train-ckks/src/metadata.rs:25–41`; and scalar `counter.rs:42–58`. These support the corrected constructor counts, deterministic search distinction and fixed N=32 table shape.

Named allocation/accounting risk checked: the new detailed metadata counts must not repeat the old blanket 4096-byte assertion. The unchanged test was read in the original review; focused helper inspection here covered `study_admission.rs:643–665,699–709,735–762,3532–3536`. Two retained states perform eight copy_vec calls, micro input adds one, two two-row matrices add four, giving 13 inside the update; two MicroPart copies and one successful handoff copy give 16 total. The failed second handoff rejects before copying. The encoded helper really uses default ReceiptLimits, and the corrected report no longer presents its configuration as <=4096. Symbolic payload expressions and fixed output lengths remain separate from allocator telemetry.

**Release meaning:** preserve all 32 necessary exact filters, classifying them as 16 new parameter-free + 15 shared parameter-free + one separately bounded constructor-bearing P3 filter. The original unversioned inventory is historical and must not drive a runtime controller. Root still chooses the P3 process bound and explicit one-invocation exact release; no implicit retry or broad `study_admission` execution is justified. There is no remaining I1 inventory correction identified in this scope.

## I2 — ADDRESSED: negatives reach the intended semantic checks

`crates/ckks-train-bench/src/study_reference/document/evidence_tests.rs:451–477` adds a test-only rebind helper that updates changed UserScope bytes/digest, Current.authority_sha256, Current bytes/digest, both embedded values and both FileRef lengths/hashes. It uses the existing canonical traversal to prepare inert negative records; the actual decoder and consistency functions remain under test. This prevents an unrelated stale digest from masking the target semantic refusal.

`evidence_tests.rs:479–549` adds all requested scope checks:

- Both legal overlap mismatches, selected1/claim0 and selected0/claim1. Each sidecar is parsed successfully before bundle comparison. Both assert `CrossLinkMismatch` at `evidence.bundle.overlap` (`:491–512`).
- Full evidence Initial versus selected authority Retry with a syntactically valid fictional predecessor. Authority/current/FileRef hashes are rebound; the decoder must reject at `user.retry.predecessor` (`:514–537`). The borrowed lock Link is expressly only a validly encoded synthetic claim, not a real prior admission.
- Quick Retry1 containing that valid predecessor. This directly tests auxiliary decoding and asserts `CrossLinkMismatch` at `authority.quick.retry` (`:538–548`).

`evidence_tests.rs:731–739` adds failed-outcome records with present Failure and legal completed count, then phase4/code0 and phase0/code7. Both assert `UnsupportedTag` at the shared `execution.tag` path. These cases cover the two tags absent from the frozen successful-end literals; the existing 84 valid combinations are retained.

No production assertion is bypassed, no public synthetic switch is introduced, and no live authority/gate/HE object is created. Existing 16 exact test names are retained. Updated fixture reads 138→144 and the +6 reads/+33,191 bytes correspond to the five Quick sidecars and one Full user literal. The authority filter's parse upper estimate32 covers the added calls; failure parse estimate90 includes its two added negatives. Metadata reads stay13 and generated records remain small. No remaining I2 correction is identified.

## Covering compile/static evidence

Read the actual new logs from the complete diff and independently verified their artifact hashes. All are uniquely named under `evidence-codec-source-validation/fix1/`:

| Check | Evidence | Result |
|---|---|---|
| Default `cargo test -p ckks-train-bench --lib d7_evidence --no-run` | `01-default-no-run.log:1–5` | exit0, test executable compiled; no test execution |
| Strict all-target/all-feature Clippy with `-D warnings` | `02-strict-clippy.log:1–4` | exit0, no warnings |
| `cargo fmt --all -- --check` | `03-fmt-check.log:1–2` | exit0 |
| `git diff --check` | `04-diff-check.log:1–2` | exit0 |

Default no-run plus all-feature Clippy appropriately covers this test-only, feature-independent edit. No additional broad compile/runtime rerun was performed or requested by this reviewer.

## Remaining boundaries

The original source-review production assessment and all retained dependencies remain unchanged. Full and existing Reference/Lock/Admission/Start bytes/caps, cfg(test)-only oracle decoding, lossless Unix argv with NUL refusal, Full21/full2680+20+7/quick268, holds, license constraints and approval/scientific thresholds are untouched. Actual readers/custody, C1 current-and-prior same-target Retry checks, current authority, complete resource models/observations, live gates, protocol/native evidence and scientific results remain required. The separately accepted Full oracle does not create a combined source-positive test in this package. Final acceptance of runtime behavior awaits root's corrected exact-filter release and observed results.
