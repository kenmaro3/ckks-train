# D7 phase2 retained canonical CPU preparation/accounting source report

Source-only package on BASE `68b9e96659ec6b2408e2d832b743d279571b5588` (source baseline `02317621a89197210e1321bbc9f2b01b2533b708`). Read the full Task54 brief, exact released proposal/FIX1, source release and public design section5 before writing. The full experiment runner remains the objective; this package does not complete Task54 or admit/start a case.

## Scope and actual execution

Implemented the retained single/block preparation owner, shared compatible validation/consuming extraction, requested-payload accounting, shared descriptor count/output traversal and confined cache, binding/identity count/output hooks, and three specified callback escape fixes. Added five focused tests. No test binary, parameter constructor, P1/proof constructor, materialization, planner, crypto/native application, GPU, network process or science experiment was executed. Cargo compilation and static checks only. No dependencies, Cargo files, public envelopes, NN math, optimizer/bootstrap proof semantics, CLI, runtime/server wiring, commit or push were changed. Original `.vscode/` and both original untracked MNIST result JSONs remain untouched.

Directly necessary scope additions were reported before use and authorized by root: (1) the fixed collector re-export bridge in `study_reference/document.rs`; (2) the shared fallible SHA output hook in `data.rs` (root source-addendum2). The module and arbitrary-hash helpers remain private.

## Source files

- New private `study_execution.rs`, registered by `lib.rs`: independent input/parameter lifetime input, retained checked/input/case/request/accounting owner, complete immutable descriptor bundle, accounting/reservation/transfer/release and focused tests.
- `study_client.rs` and `study_client/blocks.rs`: legacy callbacks delegate shared checks with `None`; new retained preparation supplies its same owner with `Some`. Canonical geometries move instead of clone. Block training union moves. Full key roles/separate bundle, binding, partitions, graph, activation/bias, secure parameters/P1/BootParams and EvalConfig validate before extraction. Legacy synthetic allocation scope stays explicit.
- `study_admission.rs`: failure-preserving consuming candidate extraction, narrow payload projection, full manifest equality; real counting/output Transcript and shared partition serialization; exact binding workspace hook and accounted validator; allocation-free preset/name validation; prefix transfer keeps the new Transcript length counter consistent. New pure prefix regression is below.
- `study_reference.rs`: one shared count/output identity encoding body, checked/fallible output, fallible identity String copies/hashes. Stable source ordering is preserved with borrowed ordered-index selection, removing both source-sort pointer vectors and library sort scratch from these identity paths.
- `study_reference/document/execution_descriptors.rs`: existing Fields validation/Writer bodies serve count/output; size-only fixed-width hashes never reach output. Fixed single/block collectors take and return the complete retained owner. Only module-private collector code derives/uses parameter/bootstrap hashes; existing source helpers still derive actual hashes independently.
- `study_reference/document.rs`: only crate-private fixed collector re-exports; parent `study_reference.rs` forwards those names. Neither module nor arbitrary-hash methods become crate-visible.
- `ckks-train-nn/src/diagonal_plan.rs`: documented public cross-crate logical nested-payload helper includes private identity dimensions/activations/Reuse pool, shapes, masks and both step vectors, transpose headers/tuples/chains, keys, and compact plans/containers/biases. No planner/math changes.
- `data.rs`: crate-private `try_sha256_hex` reserves its 64-byte String fallibly before shared finalization. Legacy `sha256_hex` retains its signature/behavior and uses the same finalization/compression. No duplicate SHA algorithm.
- Only the specified consuming callback fixes in `study_diagonal/p4_tests.rs` and `study_blocks/p5_tests.rs`.

Exact paths, hashes, saved diff/new source and untouched frozen-vector pins are in `preparation-accounting-source-validation/`. Source pins are refreshed after final validation; report SHA is reported separately to root.

## Validation evidence and preserved failures

All command stdout/stderr logs remain unedited. Each numbered command after the initial RED has a JSON argv/exit/elapsed record.

| Evidence | Result |
|---|---|
| `01-red.log`: initial tests-first `cargo test -p ckks-train-bench --lib --features parallel --no-run` | Compile RED: missing new types/APIs. It also exposed a missing DftSupportProof test import, corrected later. The initial shell continued to print the log, so its exit0 is **not** an observed Cargo exit0/101. Cargo exit was not separately captured; see `01-red-metadata.json`. |
| `02-compile` | exit101: ordered-index type inference and two tuple-field replacement mistakes; preserved and corrected. |
| `03-compile`, `04-compile` | exit0 at those source revisions; no test execution. |
| `05-clippy` | exit101: tuple type complexity and needless impl lifetimes; factored/elided. |
| `06-sha-red` | Clean additional tests-first compile RED, exit101, only missing fallible SHA/reservation APIs. |
| `07-compile` | exit0 after SHA implementation. |
| `08-clippy` | exit101: parent sha256_hex import became test-only; import corrected. |
| `09-final-compile` | exit101: missing pointer assertion local; corrected. |
| `10`–`13` | Required four Cargo checks passed before the final Transcript transfer invariant audit. |
| `16`–`19` | Required four Cargo checks passed after the invariant correction. |
| `20`–`21` | Final test compilation/clippy after adding only the pure prefix test; no-runtime feature checks from18/19 are unaffected by that cfg(test)-only addition. |
| final changed-file rustfmt and `git diff --check` | Saved final static logs/JSON. |

Required commands: `cargo test -p ckks-train-bench --lib --features parallel --no-run`; `cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings`; `cargo check -p ckks-train-bench --lib --no-default-features`; `cargo check -p ckks-train-bench --lib --features parallel,cuda`. Cargo calls were sequential. None of the compiled tests is claimed passing at runtime. No broad science/HE regression run was made.

## Lifetime, failure and drop contract

`CanonicalPreparationInput<'i,'p>` and `RetainedPreparation<'i, Checked[Block]Preparation<'p>>` keep independent lifetimes with no outlives relationship. The compile-use function accepts arbitrary independent lifetimes. Actual materialized/case/request borrows survive inside the retained owner; the checked plan keeps the input Arc and parameters/P1 remain borrowed from the original owners. There is no checked-by-value getter or consuming compatibility adapter for a retained owner.

`RetainedPreparation` declares checked first and accounting last. `PreparedCanonicalBundle` declares descriptors before preparation. Its fixed collectors consume/return that whole owner. A crate-private borrowing descriptor/accounting projection is only used by the two fixed collectors; it cannot extract checked/accounting by value. Neither collector accepts caller hashes. Failure destroys collector locals before releasing their reservation, then destroys checked state before the containing accounting. Preparation locals are similarly declared after the reservation; retained transfer subtracts temporary bytes and adds retained bytes once without changing total charge.

Candidate validation failure boxes the original candidate with its original issue, geometry digest, available geometry/states/keys and unproved gates. No candidate cloning or manufactured Candidate issue represents accounting/descriptor/allocation failures. Those have distinct typed errors, including original required/configured accounting byte/record values. Shared identity allocation/overflow maps to the corresponding Admission kind.

Extraction drops placeholder states, candidate geometry digest and unused candidate fields. Single preparation drops all candidate manifest copies; block moves `keys.training`, dropping unused bootstrap copies. Single nested geometry and block training union include debug pointer identity assertions. Single keeps the owning geometry iterator alive until activation/transpose allocation, then drops it and releases exactly its outer requested buffer. Block allocates descriptions/classes before consuming its owning iterator; the iterator drops at loop end and its exact outer payload releases there. A later retained transfer does not re-charge moved payload.

Legacy `prepare_initial` now builds arrays/encrypts inside its existing callback. The two specified synthetic escapes now encrypt inside their callbacks. Encrypted initial state, block binding and sessions can retain descriptions. Callback completion is not geometry destruction and legacy outputs do not acquire new D7 accounting. Phase3 must move this same owner/accounting through the actual live crypto/session lifetime and reserve block copies. That continuation is not implemented.

## Exact requested-payload sizing and peak scopes

Let V(T,n)=checked(n*size_of(T)), G=sum of each moved geometry's narrow nested helper, S=stage count, A=sum activation counts, X=sum Description transpose Reuse-pool payload (zero for the proposed canonical Dedicated cases), U=training union length. Logical requested lengths are counted, not capacities, inline roots, Arc handles, allocator overhead or RSS.

| Payload | Exact reservation and lifetime |
|---|---|
| Original plan | `plan.charges().combined()` seeded once, plus original indexed-record count checked once against unchanged original limits. Reused Arc/borrows add no payload. |
| Moved geometry | G reserved before consuming candidate. Existing candidate creation is old P2, outside this retention boundary. |
| Geometry outer iterator | V(StageGeometry,S), reserved with conversion, released immediately after actual owning iterator drop. |
| Single description | Inline Description root; retained G+V(DiagonalActivation,A)+X. |
| Block descriptions/classes/union | Retained G+V(Description,S)+V(DiagonalActivation,A)+X+V(usize,S)+V(i32,U). Training vector moves. |
| Descriptor bytes | I retains P+L+K+D; NI retains L+K+D while borrowing old P1 bytes. Each output's real shared traversal enforces unchanged A_MAX=2,097,152. Sum checked and reserved before any output allocation. |
| Descriptor digests | I retains256 bytes (4 Strings); NI320 (5 Strings). Same reservation precedes fallible hash String allocation. Parameter/bootstrap hashes are computed once on this collector path. |
| Error/diagnostic paths | Existing Candidate/Admission diagnostics and old P2 failure traces are outside successful-output payload accounting, explicitly not zero-allocation/RSS claims. |

Binding workspace uses real live intervals after byte-preserving allocation removals. Define B=final binding transcript length, C=consumed transcript length, Q=V(usize,consumed updates), T=V(Vec<MicroPart>,updates)+sum V(MicroPart,row length), H=shared identity byte length, J=selected receipt byte length, Z=geometry identity transcript length. The exact successful application-buffer peak is:

`max(Z+64, Q+T+B+H+J+256+C+64, B+64)`.

The first interval is the standalone geometry identity transcript and digest, before counts/parts exist. In the second interval, counts, partitions, both identity byte buffers and their four64-byte Strings remain alive while the complete consumed transcript, its64-byte digest and final binding append overlap. The third is the validator's final plan-hash String after canonical_binding has dropped counts/parts/identities/consumed. These intervals are not a sum of mutually dead buffers. The accounted hook reserves that maximum before any output binding call, and releases after validation returns. Counting uses the same Transcript/identity/partition encoding bodies and allocates no successful-path transcript, partition/count vector, identity, source-sort vector, geometry scratch or digest.

`secure_parameters` now compares borrowed preset bit bands/values, so preset/extension Vecs no longer exist in this path. Geometry name uniqueness uses borrowed prior-layer comparison, eliminating its former set allocation. Identity source ordering selects indices in stable `(key,index)` order without pointer vectors/sort workspace. Stable-sort library scratch is not claimed zero: these specific sort calls were removed. Old P2 manifest/set/library scratch and all other excluded construction remain outside the boundary. `update_rows` continues borrowing rows/checkpoint slices; no row buffer is created.

The Transcript length counter also follows `fail_record`'s truncate/take: bytes transfer into the preserved prefix, metadata reverts to its saved charge, and the emptied Transcript has length0. The new pure regression checks exact prefix bytes, metadata, subsequent append and count/output equality without the existing broader test's tiny parameter constructor.

## Proposed bounded runtime inventory (NOT executed or released)

Authored test names:

1. `study_execution::tests::d7_accounting_pure_boundaries`: no parameter/P1/proof/materialization/planner/hash/file/subprocess. Seed byte/record overage, exact fit, one-byte excess, checked overflow, transfer/release, rejected transfer without mutation, and capacity-overflow allocation failure with reservation cleanup. `allocated::<u8>(usize::MAX)` fails reservation; it does not attempt a huge successful allocation.
2. `study_execution::tests::d7_canonical_six_retained_sources`: one registry load, one registered-case projection; two secure parameter owners (`n16_train`, `n16_boot`), one default BootParams, one real DftSupportProof verification, one NI BootstrapMetadata. Exactly three resolve/materialize calls for local Xavier `iris__emlp1/2/3`, seed0; each reused for I and NI WhenNeeded. Six explicit Quick requests/plan owners, six retained preparations/collectors; no targets, keys, HE Encoder/evaluator/session/factory or runtime start. Identity/Arc/input pointer associations, moved payload assertions, retained-vs-source descriptor bytes/lengths/digests and zero legacy callback calls are checked. These are canonical candidates, not synthetic success.
3. `data::tests::d7_sha_fallible_known_vectors`: two literal inputs (empty/abc), two fallible and two legacy hashes; no constructors/files.
4. `data::tests::d7_sha_output_reservation_boundary`: empty String reserved64; capacity-overflow reservation refusal at usize::MAX; no hashes/constructors/files.
5. `study_admission::tests::d7_transcript_failed_prefix_and_counting`: pure Transcript/prefix/counting test; no parameters/P1/proof/materialization/planner/hash/file/subprocess.

For the six **successful** U=2 plan+preparation pairs only, source formulas remain **24 geometry,24 initial,24 updates,12 handoffs,6 wrappers**. They include12 geometry/initial calls from planning and12 from candidate description. There are12 binding output traversals (six planning plus six preparation) and6 additional allocation-free binding count traversals. These are conditional source inventories, not observed totals.

Extra operations in the canonical test are explicit:

- First successful I/single fixture is reused for two retained refusals: wrong actual partition (after a complete binding validation), then exhausted original candidate budget (after complete binding validation). This adds2 binding outputs+2 count traversals. The latter calls describe_candidate once, performs2 geometry-identity hashes before its first failed work-budget reservation, and invokes **zero** additional P2 geometry/initial/update/handoff APIs. Candidate diagnostic is asserted intact; no crypto factory exists on the new path.
- One legacy wrong-partition callback refusal adds1 binding output, no accounting count pass and no candidate/P2 call; its Cell counter must stay0.
- Thus the whole canonical test proposes15 output binding traversals and8 count-only binding traversals. Each consumes2 borrowed update-row views:46 row visits, no row-copy buffers. Each binding traversal sizes both dual identities; an explicit final identity size/output comparison adds1 count pair and1 derive-identities output pair. Totals:24 count traversals of each shared/selected identity encoder and16 output traversals of each; no extra materialization.
- There are7 describe_candidate invocations (six success, one budget refusal), six plan wrappers, no hidden constructor in budget-only tests. Total geometry-identity hashes in this test's binding/planning/candidate bodies are35;23 additional geometry count-only traversals. These figures exclude registry/materialization's existing provenance/hash work.
- Descriptor collectors encode I parameter bytes once each (3 retained parameter outputs); NI hashes borrowed P1. Test comparison invokes the three existing no-cache source helpers per owner and an extra descriptor_sizes per owner. Including collector, sizing and comparison:12 I parameter byte outputs/18 parameter count traversals overall; for each of L/K/D,12 outputs/24 count traversals overall. There are27 collector hashes,24 no-cache source-helper hashes and27 assertion hashes=78 descriptor/comparison hashes. These verification-only comparison buffers do not constitute retained production collector outputs and are outside its accounting; they are explicitly listed, not silently attributed to the owned receipt.
- Binding/identity/core planner hashes, excluding registry/materialization and descriptors, are35 geometry+32 dual-identity+15 consumed+9 validator-plan+6 final P3-plan=97. Do not combine this partial inventory into a claim of total test SHA calls: registry/membership/materialization have their separate existing hashing paths.
- Full materialization is not Quick materialization: each Iris profile prepares the full2400-update input plan/rates/order and both F64/F32 initial models. Quick only consumes2 diagonal updates per case. This construction remains outside retained preparation accounting and does not establish Full scientific admission.

Planned file reads are local source/input verification only: matrix and input-lock JSON; `src/bin/iris.rs`; faithful emlp1 and emlp2/emlp3 JSON configs; `results/quadruped/raw/reboot_emlp23_full/config_values.json` as registered block metadata; `src/twin.rs`; core `src/sampler.rs`; Iris CSV and fixed split text. Resolver/preflight intentionally reread/hash some files; no syscall-count claim is made. Embedded code-build identity reads include compiled `study_inputs.rs`, `study_profile.rs`, `profile_data.rs`, `data.rs`, twin/sampler and Cargo.lock bytes, not runtime network/file repair. New data.rs bytes naturally change code-build provenance of a newly materialized input; unchanged wire format does not mean identical provenance hashes across changed source builds. All five tests propose0 subprocess/native/HE/GPU/network calls. DftSupportProof is real scalar runtime arithmetic (65,535 roots), not a native observer or fake token. Root must freeze its execution separately.

Directly relevant existing regression candidates, each requiring root's exact selection:

- `study_reference::tests::independent_minimal_plan_golden_bytes_and_sha256` and `every_plan_field_and_numeric_coefficient_has_its_declared_scope`: pure BatchPlan/Encoding fixtures, no parameters/P1/materialization/planner/file/subprocess.
- `data::tests::sha256_known_vectors` and `streaming_sha_independent_vectors_and_empty_updates`: pure literal/streaming SHA regression, no parameters/P1/materialization/planner/files. The first has3 known inputs; the latter's literal cases/empty-update/chunk loops remain unchanged. No broad data-loader tests proposed.
- Four `study_reference::document::execution_descriptor_tests::execution_{parameter,layout,key,security}_literal_bytes` tests: existing14 literal vectors, borrowed Fields/Writer encodings and frozen digest checks. No CkksParams/P1/proof/materialization/planner or runtime file reads (include_str fixtures).
- `execution_shared_p3_geometry_matches_frozen_layout_slices`: pure literal geometry projections for the three layout fixtures (1+2+3 stages), no geometry planner or parameters/P1/materialization/files. Frozen negative vectors and prefix tests remain unchanged and available for exact root selection.
- `study_admission::tests::p3_fix1_structured_golden`: **not constructor-free**. Its shared synthetic fixture lazily loads a registry, constructs2 tiny N=2^10 parameter objects,1 real proof and1 adjusted tiny NI metadata/BootParams, plus its single CtMeta. Geometry/initial/update/handoff APIs and materialization remain0 for this isolated test; no broad P3 suite is proposed. Existing `p3_digest_golden_and_mutations` also later constructs1 tiny parameter object despite its earlier pure-prefix section, so the new focused prefix regression avoids silently importing that constructor.

## Frozen bytes and remaining boundaries

No frozen descriptor literal/negative/manifest fixture was changed; exact current-vs-HEAD comparisons and SHA pins are saved. P1 constructors/proof encoding source is unchanged. Shared P3/identity/descriptor bodies retain their existing field order/primitives and successful wire format; golden/negative runtime assertions have only been compiled, not executed. No unchanged runtime digest claim is made across changed source-build provenance.

Existing old P2/candidate construction, materialization/input/context/parameter/P1/target construction, allocator overhead/RSS and diagnostic paths retain their separate boundaries. This package proves no all-preparation bound, resource fitness, security/precision/range/runtime gate, target completion or science result. `study_runtime::run_owned` still returns MissingExecutionDependency. Explicit immutable request/consumption wiring, phase3 same-owner live transfer, fixed CPU bodies, outer ExecutionBinding/runtime policy, genuine gates, durable admission/start and publication are still absent. Quick use of Full registration remains an operational prefix. Full21 profiles/full2680+20+7/quick268, unresolved B/sameN/authors and existing holds remain unchanged.
