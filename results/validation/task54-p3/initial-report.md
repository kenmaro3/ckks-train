# Task54 P3 source-stage handoff — frozen local evidence

2026-11-07. **Operational implementation/validation report, not scientific results or admission.** Source and Cargo ownership are returned to root for independent review. Original full P3 acceptance remains incomplete because the reachable P4/D7 pre-key factory integration does not yet exist. No Ready, secure profile/campaign admission, HE run, deployment or performance result is claimed.

## Frozen source and scope

| File | SHA256 | Ownership/change |
|---|---|---|
| `crates/ckks-train-bench/src/study_admission.rs` | `4d6b1cc66f820b5782c6c9831f24eb94f3eef999e6dd2a572e38007f254fbb56` | New P3 metadata binder, complete bounded transcript/index, typed state/handoff views, private unit fixtures. |
| `crates/ckks-train-bench/tests/study_admission.rs` | `85c13c334c3956cde76b2992130155d7225f214cc8087c1187520c1b69d88a0e` | New external inventory/canonical/configuration tests. |
| `crates/ckks-train-ckks/src/bootstrap/metadata.rs` | `07ae0174669d5c494836cfb2cb9bddbbc1173cc70cf304a96e5d65e2d7013a12` | Exactly the root-amended private original `BootParams` field, constructor clone and immutable `boot_params()` getter. Existing codecs/arithmetic/proof/limits unchanged. |
| `crates/ckks-train-bench/src/lib.rs` | `867b2c6d8f4391861956c10483fe18c91636e9272e41cceee8953b90976b3e44` | Root-coordinated export only; worker did not edit. |

Exact current source copies are in `task-54-p3-validation/frozen-source/`. Both final runtime logs record all four hashes before and after; they match. Original03/05 source bodies were **not** saved as filesystem snapshots: their logs retain exact hashes, and tool history retains edits. The separately released current-source joint regression resolves the successful-composition coverage gap without treating old hashes as current validation.

## Implemented behavior and acceptance-by-acceptance evidence

| Requirement | Actual implementation/evidence | Limit/status |
|---|---|---|
| All21 metadata × I/WN/AU/AA | `p3_inventory_and_candidate_separation`:84 expected records,144 stage geometry/initial pairs using actual secure parameters; metadata geometry excludes seed, seed9 missing capture remains typed. Logical/padded classes, compact/key/initial labels come from P2. | Fresh candidate metadata only; descriptive rate0.08 never becomes a production schedule receipt. Dataset readiness stays Unverified/MissingCaptured. |
| Sealed case/private projection | `checked_mode` borrows the existing sealed `RegisteredCase`; no CaseSpec registry. Full-only selection and268 inspection-only quick selectors still reject runtime quick/archival selection. | P3 quick is a separate consumed-prefix request against Full case bytes. No frontend/provenance scope mutation. Other roles/packings remain MissingCode. |
| Canonical versus candidate versus synthetic | `plan_diagonal` requires canonical_prepared and canonical selector; CandidateDescription has a separate return type; cfg(test)-private SyntheticPlan never constructs MaterializedProfile/DiagonalPlan. | No public synthetic materializer, deserializer, Ready, start, callback or session capability. |
| Real canonical positive | Final integration runs one bundled `iris__mlp2020` seed0 materialization, binding all2400 updates/full rates/reference and consuming exactly2 quick updates. Full/consumed identities differ; original full registration remains Full. | Repeated exactly once in the explicitly released frozen regression, so cumulative materializations2. No plaintext training/download. Actual canonical retained-byte totals were not printed and are not inferred. |
| Full identity/typed digest | Versioned ordered LE encoding binds complete canonical dual identities, all materialized component/source digests, full plan/checkpoints, every full coefficient/reference, actual q/p, concrete BootParams, physical request, partitions/rows/lookahead, geometry, states and every consumed operation trace/attempt/event. Raw IEEE bits; no Debug hashing. | Primitive fixed-byte golden and known SHA vector; actual full canonical bytes/hash self-check. This is an operational metadata encoding, not a new scientific artifact schema or cryptographic attestation. |
| Concrete actual bootstrap configuration | A genuine missing accessor was amended narrowly. Dedicated getter test verifies exact cloned original fields and unchanged parameter identity after caller mutation; constructs real K14 descriptor on same q/p/output and rejects it against default K13 before any stage update. Valid default candidate control succeeds. | Root-ratified extra metadata-only negative. No alternative secure configuration adopted. Preset checking relies explicitly on sealed deterministic CkksParams constructors and exact ordered bit vectors/scale/n/dnum/secret; receipts bind actual prime integers. |
| Whole consumed schedule/state | Synthetic Square/SGD and Poly/Identity/NAG across all4 placements traverse all supplied4 updates, including equality/decrease/restart, unequal/full/partial micros and terminal unknown lookahead. Persistent state/velocity and coefficients are borrowed from P2 results. | No cache/constant/cosine shortcut. Production Full iterates the entire intended schedule or returns failure; no large full-profile feasibility run occurred. |
| Multistage old-hidden handoffs | Synthetic e3 across all4 placements uses P2 `plan_stage_handoff`, destination-specific y, identical valid counts, returned old-hidden x, independent stage state. I refresh, WN/AU conversion+boot, AA canonical reuse with zero extra boundary conversion/boot. | Source performs no NN recurrence/predictor copy or fake HeOps. Actual CPU HE/block parity belongs to P4/P5. |
| Invalid/rate/partition/identity failures retained | Invalid dimensions/axes, malformed/overlap/zero/oversize/nonconserving micros, full-reference mismatch, mini-versus-secure and actual production parameter/partition mismatch reject. Deliberate third-update RateConflict retains exact P2 error and two completed updates. | Missing datasets/exports are independent D6a states, not level errors; no fallback, threshold/profile/rate change. |
| Complete inspectable bounded receipts | Full canonical transcript + indexed typed compact summaries; no per-update cloned P2 geometry/parameter/descriptor/StageState/trace objects. Shared retention helpers charge combined bytes before fallible reservations. Prefix helper commits complete records atomically and discards only incomplete fragments on failure, preserving completed typed index and failing P2 trace/attempts. | AllocationFailed and configured cap are separate from level/scale failure. Current/transient allocations, allocator overhead/capacity rounding, RSS and HE memory remain unproved. |
| Exact byte-cap discriminator | Final prefix test uses one real test_tiny only for valid CtMeta, nonempty pre/post weight/bias/compact/velocity/incoming/raw/AA/handoff arrays and partitions. Actual charge:3 transcript+2256 metadata=2259 bytes. Exact cap succeeds; one-byte-over and extra handoff payload reject with exact required/limit; unchanged previous records/typed source trace survive failure. | The accounting fixture is not a registered parameter or HE test. No proof/bootstrap/stage/update/materialization in it. |
| Keys and independent unproved gates | Training union, separate generated/executed bootstrap rotations, conjugation/SSE and complete key-role parameter descriptors are bound. Resource/range/precision/security/runtime/target/user/pre-key integration remain explicitly Unproved. | Logical union cardinality never becomes physical resident-key count/bytes; no bundle sharing assumed. No Ready variant. |
| Required key-builder-spy ordering | Real `validate_plan_binding` exists and is exercised against canonical actual parameters/partition; no unreachable fake spy was introduced. | **REQUIRED AND PENDING:** first actual P4/D7 consuming boundary must satisfy genuine non-P3 start/gates, provide a reachable nonallocating positive spy control, then show corrupt identity/parameters/partitions reject before factory invocation. Standalone P3 does not discharge it. |

## Exact executed evidence

All Cargo commands used existing target, `--locked --offline`, CARGO_INCREMENTAL=0, CARGO_BUILD_JOBS=8, RAYON_NUM_THREADS=8, one Cargo owner and disk free >=805306368 bytes. No automatic retry occurred in the final frozen release.

| Raw log | Outcome / purpose |
|---|---|
| `01-red.log` | Expected compile RED: missing P3 types/helpers. No fixture executed. |
| `02-unit.log` | Compile-only type-inference error; fixed by explicit Result annotation. No fixture executed. |
| `03-unit.log` | Original5 unit tests PASS on its printed source hash. Retained verbatim, including verbose failure-prefix Debug output. |
| `04-integration.log` | Compile-only misplaced local `before` variable in test diagnostics; corrected scope. No fixture executed. |
| `05-integration.log` | Original3 integration tests PASS on their printed source hashes. |
| `06-getter-red.log` | Expected missing-method RED for `boot_params()`. No fixture executed. |
| `07-getter.log` | Filtered actual getter/configuration binding test PASS1/1. |
| `08-prefix-red.log` | Runtime RED: actual production failure helper incorrectly retained appended incomplete fragment. No parameter fixture reached. |
| `09-prefix.log` | Filtered corrected prefix/actual combined-payload test PASS1/1. One tiny parameter; no planning/HE. |
| `10-bench-clippy.log` | One owned needless_range_loop failure; changed to equivalent counts.iter().enumerate(). |
| `11-ckks-clippy.log` | Targeted CKKS lib Clippy PASS. |
| `12-bench-clippy.log` | Targeted bench lib + owned integration target, no-deps, -D warnings PASS on frozen bytes. |
| `13-format.log` | Owned3-file Rust2024 rustfmt check PASS on frozen bytes. |
| `14-frozen-unit.log` | Exact root-released current-source5 unit tests PASS. All four before/after pins match. |
| `15-frozen-integration.log` | Exact root-released current-source4 integration tests PASS. All four before/after pins match. |

Final runtime commands were precisely the source-release `cargo test ... --lib study_admission::tests::p3_ -- --test-threads=1 --nocapture` and `cargo test ... --test study_admission -- --test-threads=1 --nocapture`. Earlier filtered commands and complete command lines/exits are in their logs. No old P1/P2 tests, broad workspace/release, old84/212, helper matrices or CPU HE tests were run.

Post-original-PASS functions changed and now exercised by final regression: compose→shared commit/fail record helpers; shared StateLabels/update/handoff summary retention; canonical_binding equivalent enumerate iteration; secure_parameters actual descriptor BootParams equality; budget-error required/limit diagnostics. The current successful canonical/multistage callsites therefore have fresh complete focused runtime evidence rather than only helper tests.

## Actual cumulative work, including all authorized processes

| Resource | Actual total |
|---|---:|
| CkksParams constructors |12|
| DftSupportProof verifications |6|
| BootstrapMetadata constructors |8|
| Canonical bundled Iris materializations |2|
| P2 geometry / initial-state calls |336 /336|
| P2 stage-update calls, including expected rejection |148|
| P2 handoff calls |48|
| Production wrapper calls, including rejected calls |6|
| Key / Encoder / evaluator / ciphertext / HE / plaintext-training calls |0|

These totals include original03/05, filtered07/09 and the separately ratified frozen joint regression14/15. Compile-only failures and prefix RED08 constructed nothing. Per frozen joint run: unit3 params/1proof/1descriptor and22 geometry/72 updates/24 handoffs; integration3 params/2proofs/3descriptors and146 geometry/2 updates/0handoff/3 wrappers/Iris1. All additional allowances were explicit root amendments; unspent allowance was not used. Frozen regression selected5+4 tests only.

## P4 surface and remaining dependencies

DiagonalPlan lends immutable full rates/reference, actual consumed partitions, initial states, indexed update/handoff summaries, incoming stage-specific labels, pre/post weight/bias/compact/velocity metadata, coefficient bits/branch/count decisions and retained raw/AA/handoff labels. Complete traces/attempts/events and stage geometry/key-role details remain inspectable as canonical bytes with checked record ranges.

Specific accessor gap: it does **not** retain/lend StageGeometry objects or a typed full-trace/attempt decoder. P4 must not invent a second codec. It can reconstruct geometry from the same trusted MaterializedProfile and validated request using existing P2 APIs, or root can separately assign a bounded typed geometry/trace accessor hunk after review. A cfg(test) crate-private synthetic P4 fixture seam was not added; future ownership is explicit.

Full-profile transcript/typed-index feasibility, actual total RSS/current/transient planning memory, cryptographic resident keys/workspace, bootstrap encoding/range/precision/security, target locks, real D7 descriptors/capture/start/gates and all physical/campaign evidence remain pending. The64MiB convenience cap is an operational limit, not a claim full Fashion fits. A budget failure returns a failed expected row and complete prefix, never a shortened complete plan. Full encrypted-block holds and2,680+20+7 full /268 quick inventory remain unchanged.

No Git action, network/SSH, download/export adoption, dependency change, scientific JSON, server process, cloud GPU, HE or campaign occurred. Unowned `.vscode/` and `results/mnist_blocks_*_plain_b48.json` were not touched. Source is frozen for independent review; no further Cargo or source work is implied by this handoff.
