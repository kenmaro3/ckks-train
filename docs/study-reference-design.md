# Prospective study reference artifacts and target locks

**Status: INDEPENDENTLY REVIEWED AND ROOT-RATIFIED DESIGN — Task54 D6c; implementation and execution separately gated.**

This document proposes a bounded extension of independently reviewed D6a materialization and D6b actual-F32 computation (`b897f696eb97f9fc093d3d8b95abe1266d9c7d37`, supplied reviewed dependency). It creates no implementation, trained reference, target lock, campaign authorization or measurement. The governing contracts are [study training sections 7–8 and 10–11](study-training-design.md) and [the experiment matrix](experiment-matrix.md). Existing FULL-1/FULL-2 inputs, acceptance gates and historical outputs remain unchanged.

## 1. Recommendation, reasons and limits

Add a thin `src/bin/study_reference.rs` over a small `src/study_reference.rs` library module, with a narrowly scoped `src/study_provenance.rs` and compile-time build receipt. The binary selects an approved profile/seed and an explicit full-F32 or two-update diagnostic mode; the library owns the state, execution, evaluation and publication. The writer derives facts from immutable `MaterializedProfile` and actual `PlainState<f32>`; it accepts no accuracy, cursor, checkpoint, provenance map or target override from the caller.

For each concrete profile, only its actual full seed0 F32 **final endpoint** can create a prospective target. The target is `max(0, final_accuracy - 0.005)` and applies unchanged to statistical seeds0–9 of that same profile. B48, B64 and B128 require separate references and locks. A full other-seed reference may later be useful for diagnostics, but never substitutes for seed0 in this target algorithm. F64, synthetic, quick, truncated, best-epoch and cross-profile artifacts are invalid target sources. Missing captured inputs stay `MissingInput`; no initialization/order/schedule fallback is added for block seeds3–9.

The lock binds both the exact seed0 reference bytes and a shared scientific profile contract. It does **not** require seed7's initialized arrays, consumed order, realized rates or combined input digest to equal seed0's. Same-seed HE/plain pairing still requires exact identity within that seed.

| Alternative | Reason for decision / cost |
|---|---|
| New thin binary and typed library session | Recommended: preserves legacy arithmetic/CLI behavior, makes final-state and writer ownership reviewable. Costs one CLI, versioned schemas and small provenance implementation. |
| Add fields to historical `plain` / `mnist` output | Rejected: historical execution/dtype/output paths do not establish the new materialized actual-F32 contract. |
| Runner constructs JSON from a public metrics map | Rejected: callers could supply fabricated counts, completion and runtime identity; schema conformance would not demonstrate their origin. |
| Require all seed-specific hashes to equal the seed0 lock | Rejected: silently reduces the registered ten-seed matrix to seed0 or admits substituted seed0 inputs. |
| Trust timestamps as proof of preregistration | Rejected: local clocks, files and code are mutable. Use a stated local ordering boundary and fail closed at admission; stronger attestation is a separate system. |
| Global cohort seal and hash-chained event journal | Rejected by root before independent review: coupling every profile's readiness and adding mutable-journal serialization/recovery does not provide trustworthy external attestation. Immutable per-profile locks and per-case receipts supply the required local lock-before-start boundary without a global total-order claim. |

D6c does not redesign the D6 HE counter/timer implementation, D7 matrix scheduler, physical admission or D8 aggregation. It defines their reference/lock interfaces only. Full encrypted blocks retain their user hold. Generation of a full scientific reference is a later explicit execution package, even after this implementation passes review.

## 2. Current source evidence versus proposed APIs

These are code-reading findings, not new tests:

- `study_inputs::materialize(&ResolvedProfile)` internally prepares once, freezes graph/data/initial arrays/full plan and exposes immutable identity views. `canonical_prepared()` excludes private synthetic assembly. Its `hekit-study-inputs-v1` digest already binds selected-seed arrays/order/rates, source locks, dataset/split/preprocessing and embedded input-code identity.
- `PlainState::<f32/f64>::new(&MaterializedProfile)` owns private parameters, velocity, cursor and poison state. `step_plain` has no caller-supplied row/rate/index and consumes one next update. Success alone advances counts. `evaluate_plain` computes integer `correct` and `test_count` from the bound held-out data. Public `PlainEvaluation` fields are mutable values, so the new finalizer must call evaluation itself, never accept one as evidence.
- Existing D6b unit tests can set a private cursor near exhaustion. Those tests are explicitly synthetic control-flow fixtures; their cursor values are not a trained full endpoint. D6c must not use that technique to create a production reference or a positive production target-lock fixture.
- `src/provenance.rs` provides useful `build_info`, CPU/host/load/peak-RSS observations. Its current `provenance()` joins argv into a string, observes `rustc` on PATH at runtime, and maps failed/empty Git status through `unknown` to `none`. It provides no executable hash or actual build-compiler receipt. Reuse useful observations, but do not copy these ambiguities into scientific readiness. No root `build.rs` or `crates/ckks-train-bench/build.rs` exists in the inspected tree; a compile-time receipt is proposed new work.
- `tests/output_provenance.rs` checks legacy binary-written fields and bounded synthetic outputs; these tests do not establish full study references or immutable publication. `scripts/run_all.sh` separately binds source/build/binary identities, checks build receipts, records tracked and untracked dirty state and compiler/config/environment hashes. Its metadata is honestly labeled runner metadata; its existing remove-and-replace output behavior is unsuitable for new immutable artifacts.

All type/function names below are proposals. Existing modules and legacy runners are not changed by this design task. Future edits to shared modules require explicit package ownership.

## 3. Two distinct identities and explicit seed selection

`StudyProfileContractV1` is a private typed value derived from one canonical materialization and the approved registry/input-lock binding, using a reviewed, explicit seed-selection projection. Its constructor must reject unknown policy variants rather than delete arbitrary fields containing “seed”. The canonical payload contains:

1. Schema/contract algorithm versions, concrete registry profile ID, approved registry digest, portable input-lock digest, ratified design document digests/revision and permitted statistical seed domain `0..=9`.
2. Dataset key and verified dataset content, split, source-data and preprocessing hashes; feature/train/test/class counts; ordered layer shapes, activation/bias roles, stages/local-head/representation semantics, loss, optimizer, momentum, decay and arithmetic/cast/classification policy.
3. Logical batch, full epochs/passes, tail/flat-stream semantics, processed/dropped totals, and full **plan topology**: every update index/start/valid count/source-schedule index plus every checkpoint/boundary/interval field. This topology excludes seed-selected numeric coefficients, not their selection semantics.
4. Initialization distribution/orientation/RNG/draw recipe and seed-selector policy; order recipe, coverage/population/pass policy and seed-selector policy; schedule source identity and source length/reference batch/lookup/conversion policy. Constant rates are included when there is no captured schedule. For captured schedules, bind the complete approved containing source JSON hash and the resolver algorithm that selects the model/seed pointer; per-seed pointer and realized coefficients belong to the next identity.
5. Shared config/code/source bindings and explicit source roles. For captured seed-selected weights/order, bind the approved portable lock **as a whole**, captured model/export family and resolver/selection version; do not include only seed0's selected filenames in the shared contract. Other shared sources remain bound individually. Paths are repository-relative. Selection is the canonical resolver's known profile/model/seed rule, never filename regex replacement or a caller-supplied template.

The initial implementation must inventory every input-source role and metadata field in this projection. Metadata added later requires an explicit version decision; it cannot be silently omitted. Where current public metadata does not expose a shared source role or selector directly, add a small reviewed immutable descriptor in `study_profile`/`study_inputs` under separately assigned ownership. Do not reconstruct selection heuristically in the writer. The pinned registry and portable lock plus that descriptor define which selected-seed inputs are approved without loading all seeds or inventing missing ones.

`SeedInputReceiptV1` separately contains selected statistical seed, the complete existing materialized digest, all component digests (initial F64 weights/biases, F32/F64 numeric inputs, order, exact numeric full plan, dataset, split, preprocessing, code), selected source records with expected/actual digests, selected schedule pointer and selector bindings, plus full-plan counts. It is derived directly from that seed's canonical materialization. No digest is copied from the seed0 receipt to fill a missing seed.

For a future seed7 HE case, resolve/materialize seed7 using approved bindings; compare the newly derived shared contract to the lock; retain and validate seed7's complete receipt against its own approved inputs; and embed it in the case admission and result. Seed0 reference bytes and their own receipt remain unchanged. Changed profile/batch, registry/lock, dataset/split/preprocessing, graph, loss, order recipe, selector, schedule source or full-plan topology fail shared comparison. Changed selected arrays/order/rates fail that seed's canonical preparation or expected receipt check. Missing captured seed7 fails before HE allocation. Two valid different seeds can share the contract while differing in every expected seed-specific hash.

## 4. Typed execution and finalization

Proposed public surface (all returned state/provenance/artifact fields private, immutable accessors only):

```text
ReferenceSession::start_fp32(profile: &MaterializedProfile,
                             context: &VerifiedStudyContext) -> Result<Fp32Session>
Fp32Session::advance(&mut self) -> Result<ReferenceProgress>
Fp32Session::finish_full(self) -> Result<CompletedFp32Reference>
Fp32Session::finish_quick(self) -> Result<DiagnosticReference>
write_fp32_artifact(&CompletedFp32Reference, &NewArtifactPath)
    -> Result<PublishedReference>
write_diagnostic(&DiagnosticReference, &NewDiagnosticPath)
    -> Result<PublishedDiagnostic>
read_reference(&Path, &ReferenceReadPolicy) -> Result<VerifiedReference>
lock_target(&VerifiedReference, &VerifiedStudyContext, &NewProfileTargetStore)
    -> Result<PublishedTargetLock>
verify_target_lock(&Path, &Path /* reference */, &MaterializedProfile,
                   &VerifiedStudyContext) -> Result<VerifiedTargetForSeed>
```

`VerifiedStudyContext` is constructed by reading pinned design/registry policies and capturing runtime/build facts internally; there is no public struct-literal or arbitrary JSON constructor. Paths/selectors are input requests, not measurement claims. `ReferenceReadPolicy` is an approved context selector, not an `allow_synthetic` switch. The F64 path has a separate diagnostic session/result type and can never coerce into `CompletedFp32Reference` or `VerifiedReference` eligible for locking.

The session borrows its immutable profile and privately owns `PlainState<f32>`, start provenance, monotonic timers, next expected update and checkpoint receipts. Start requires canonical preparation and supported policy; records origin0/0 after state construction. Each `advance` calls `step_plain` once and checks its returned update index/sample count against the same full plan. It records every crossed checkpoint using the bound plan and internally evaluated state. There is no public mutable state accessor, resume-from-cursor API, imported checkpoint setter or untyped constructor. Any failed advance/checkpoint evaluation terminally fails the session, including failures after a successful training update; finish cannot recover it as success.

`finish_full` requires all of: session never failed; state unpoisoned; actual dtype F32 and round-to-F32-once inputs; state input identity equals the bound immutable profile; start at0/0 with exactly sequential successful consumption; completed updates equals full plan length; processed samples equals full total; recorded checkpoint sequence equals the full plan without missing/repeated boundaries; last endpoint is the full final checkpoint. It computes final evaluation internally (or reuses only its own identical final evaluation receipt), checks `0 <= correct <= test_count`, `test_count > 0`, and finite model/velocity and reporting values, and derives a final state digest from canonical named F32 parameter/velocity bits. No best-checkpoint selection is available. Origin/final endpoint identity, full plan and consumed prefix hashes coincide only after true completion.

Sequential completion is guaranteed by the reviewed safe session/state construction path, not by comparing two integers in parsed JSON. A future resume feature would need a separate authenticated state/receipt design and review. Private tests can bypass Rust invariants; they are not evidence that production training happened. A modified executable or forged file can also invent these claims: the provenance/read trust boundary below remains necessary.

`finish_quick` permits exactly the registered two-update prefix, labels it `diagnostic_quick` and includes full-plan and consumed-prefix identities separately. It cannot call the full constructor. Bounded real Iris smoke tests exercise this path, never a shortened scientific “full” profile.

## 5. Minimal reference and diagnostic schemas

`hekit-study-reference-v1` is a successful full actual-F32 artifact with the following required logical fields; serialization exposes values but does not make arbitrary deserialization into the trusted construction type possible.

| Field group | Required contents and derivation |
|---|---|
| Header | Schema, artifact kind `full_fp32_final`, status `complete`, run ID, writer binary/API version and algorithm revision; synthetic=false is derived, never caller-selected. |
| Scientific identity | Complete shared contract plus its digest, complete selected-seed receipt, dtype/accumulation/cast/reduction/classification policy. Target eligibility additionally requires seed0. |
| Completion | Origin0/0; full/consumed plan hashes; planned/actual update/sample/drop totals; final checkpoint; cumulative endpoint receipts with all boundary and interval metadata; final parameter/velocity digest. |
| Observation | Integer `correct` and `test_count` and reporting accuracy at each recorded endpoint, with the final explicitly identified; no caller-authored raw research numbers. |
| Time | Integer Unix nanosecond start/completion readings and monotonic durations; declared clock source/scope. Cumulative update duration and evaluation duration are disjoint; end-to-end is enclosing and must not be added to them. Setup includes materialization only if the binary timer actually begins before it; otherwise label excluded. |
| Provenance | Internally captured source/build/compiler/executable/argv/host/thread/environment/input receipts described below; unavailable optional observations are explicit, not fabricated zeros. |
| Integrity | Encoding/hash algorithm versions and canonical payload SHA256; excludes only its own digest field. File-byte SHA256 is returned by publication and embedded in the target lock, not recursively in this file. |

The writer may initially omit optional timing metrics with explicit `unavailable` reasons rather than claim D6 timing completeness. Any included clock specifies actual boundary and scope. CPU scalar arithmetic remains one execution thread even if the crate enables `parallel`; record build features, available CPUs and environment separately. No GPU memory/timing fields are invented for this plaintext CPU artifact.

Diagnostics use `hekit-study-diagnostic-v1`, explicit dtype, quick/full-diagnostic scope and `target_eligible=false`, on a separate path/type. Failure/partial records use `hekit-study-reference-failure-v1` with phase/error/last successful endpoint and no successful final-accuracy/target field. Killing a process may leave only a temporary file or an earlier failure/progress record; absence of a finalized artifact is not completion.

### Canonical content and file-byte integrity

Use a new domain-separated deterministic typed encoding, `hekit-study-reference-encoding-v1`, following D6a's explicit encoding conventions, not arbitrary JSON serialization order: fixed schema field order; length-prefixed UTF-8; little-endian u64 integers; single-byte enum/option tags; IEEE754 little-endian float bits; array lengths and ordered elements; source/environment maps sorted by UTF-8 key. Include all identity, observation, provenance and timing fields. Optional absent fields encode an explicit tag. Define stable enum tags and exact field order in the implementation's encoding specification and golden byte fixtures before code is frozen. This version does not hash platform `usize`, physical ndarray strides or map iteration order.

Disk JSON is a readable representation of that typed payload with required keys, no duplicate/unknown fields, exact integer types/ranges and no nonfinite values. Float reporting fields must round-trip to specified f64 bits; hashes are lowercase fixed-length hex. Readers enforce bounded file/collection sizes, reject trailing documents/duplicate keys, reconstruct the typed outer envelope including full lossless identity byte strings, re-encode that envelope and verify its payload digest. A content digest identifies semantic typed content; the file digest hashes **all exact bytes read** including JSON whitespace. A lock binds both, so reformatting an already locked file is a file-identity mismatch even if content digest is unchanged. No serde default may fill a missing scientific/provenance field.

Approved D6c1b refinement ([complete schema and parser design](study-reference-document-design.md)): Readers reconstruct and canonically encode the typed outer artifact envelope, including the full losslessly decoded identity byte strings, and verify envelope/identity digests. An initial parser returns untrusted claims only and need not duplicate the D6c1a identity binary grammar. Before verified-reference eligibility, an approved-context reader must compare the complete embedded shared contract and selected-seed receipt bytes to identities freshly derived from the same canonically materialized profile and seed; it must validate all completion/observation/checkpoint claims against that trusted plan/data, and enforce approved source/build and completion policy. Neither digest equality alone nor successful document parsing establishes inner semantic validity or completed training.

## 6. Binary-owned provenance and its honest trust boundary

Add a narrow study build receipt rather than reinterpret `rustc_on_path` as a build fact. Proposed `crates/ckks-train-bench/build.rs` records the actual Cargo-selected `RUSTC -vV`, target/host/profile/optimization/debug settings, enabled features, package/lock identity, approved arithmetic algorithm/source manifest, and relevant compiler/config/build-environment digests. Build tracking must include all owned reference/input/plain/provenance/graph/data/RNG code, manifests, Cargo.lock and selected workspace/ancestor/Cargo-home configuration that can affect compilation. Use explicit rerun-if-changed/env-changed coverage; an external build recipe must reject wrappers/unsupported overrides or record their identity. Unknown compiler/build evidence fails production readiness; the runtime `rustc` on PATH remains an optional separately named diagnostic.

Reuse `run_all.sh`'s source/config/environment hashing **policy** where accurate, without modifying that historical runner or copying its measurement output path. The new build recipe records source identity before and after compilation and executable hash afterward. The binary embeds compile facts and verifies a supplied build-receipt file against its embedded identity and its own executable bytes. A build receipt is an integrity link within a trusted local build process, not proof against a malicious compiler. Changes to ignored build-affecting source/config must be covered or explicitly rejected; an ordinary Git commit alone is insufficient.

At session start, internal capture records:

- Absolute canonical executable path and SHA256 from `current_exe()` bytes, then rechecks at finalization; detect replacement or fail if unreadable. This does not cryptographically attest the already loaded process image.
- Exact argv as an array (on supported Unix targets, lossless encoded raw bytes where not UTF-8), working directory, run ID and writer schema. Do not join argv and later guess quoting.
- Embedded build/source receipt and verified build-receipt byte digest; source checkout identity with root-explicit Git invocation, commit, tracked and untracked dirty states, and relevant file manifest. `clean`, `dirty` and `unavailable` are distinct. Dirty source is not silently equated to HEAD: production locks require the approved source-manifest digest; an explicitly reviewed dirty snapshot must have that exact approved digest. Unavailable required source/build evidence rejects production readiness.
- CPU model, OS, host, process/thread observations, execution thread count1 for the scalar engine, feature flags and allowlisted threading environment (`RAYON_NUM_THREADS`, OMP/BLAS equivalents as relevant). Hash build-environment values using the audited policy; do not dump arbitrary environment secrets into results. Load/peak-RSS helpers may be reused with source/unit labels, but are not full scoped resource measurements.
- Start/completion wall times, monotonic timer source and scopes; input-source expected/actual digests from canonical materialization. Recheck mutable source/build receipt identity at finalization as required by the approved recipe; a mismatch creates failure, never rewrites the start facts.

No public `RuntimeProvenance { ... }` constructor is accepted by the production writer. Test clocks/filesystems are private test seams returning explicitly untrusted fixtures. Library types prevent accidental caller substitution in the reviewed safe process. They cannot establish external runtime truth merely because a deserializer or a caller supplies matching-looking fields. Offline readers establish schema/integrity/compatibility against approved digests and the declared trusted local process, not cryptographic execution attestation. A malicious operator can forge a whole file and recompute its hashes. Source/build/executable receipts and local ordering are inspectable accountability, not remote or tamper-proof proof.

## 7. Atomic no-overwrite publication

`NewArtifactPath` validates the dedicated study output root, kind-specific namespace and unique final name; reject historical/FULL destinations, symlinks and traversal. Callers cannot opt into overwrite. Reference, diagnostic, failure, lock and admission files have disjoint suffixes/namespaces. Output directories are locally trusted and access-controlled; support only verified local filesystems with the required atomic operations in the first implementation.

Serialize the complete validated artifact in memory before publication. Create an exclusive unique temporary file in the final directory using `create_new`; write all bytes, flush and `sync_all`. Publish the completed inode with a same-filesystem **no-replace** operation (e.g. hard-link temporary path to final path, which fails if final exists), sync the containing directory, unlink the temporary name and sync directory cleanup. Ordinary rename-over-existing is forbidden; `exists()` followed by rename is not atomic no-overwrite. Validate final-path nonexistence atomically, including competing writers. File and directory APIs must use no-follow/verified directory handling or explicitly reject untrusted concurrent directory mutation; a path precheck alone is not a security boundary.

Return `PublishedReference` only after successful publication/directory sync, carrying exact file digest, payload digest and final path. Failures before publication leave no final success file. If linking succeeded but a subsequent sync fails, return `PublishedButDurabilityUnconfirmed { path, digests }`; do not delete or overwrite the visible final file and do not automatically create a target lock from it. Recovery requires a separate explicit verification/sync operation under the same publication policy. Unsupported atomic/durability operations fail closed for scientific publication; diagnostics may use a separately labeled weaker guarantee, never a production lock. A complete file visible after a crash still requires normal verification before reuse.

Temporary/partial files are never read as success. A failure record is separate, best-effort and also no-overwrite. Crashes can prevent a failure record, so successful completion is established only by a verified finalized artifact and admission policy, not by process absence or a log line. No artifact claims that directory fsync occurred before its own bytes were written; publication protocol and the successful publisher return establish that local durability boundary.

## 8. Target lock schema, threshold and verification

The concrete API, wire and publication contract is [the target and admission design](study-target-design.md). Its source and execution gates remain separate.

`hekit-study-target-lock-v1` contains: unique lock/campaign IDs; approved design revision/document digests; full shared contract and digest; source reference payload digest **and exact file SHA256**, reference kind/schema, selected seed0 full input receipt digest and full plan/final endpoint identity; final integer counts/reporting accuracy; target algorithm version; threshold bits/reporting value; lock-created wall time; lock-writer build/runtime provenance; and canonical lock payload digest. Final lock bytes are no-overwrite and their file SHA256 is bound by admission. A target lock is not an editable configuration file.

Proposed numerical algorithm `seed0-final-minus-half-percentage-point-f64-v1` makes the existing reporting formula precise: require positive count and both integer counts <=2^53; compute `accuracy = (correct as f64)/(test_count as f64)`; compute `target = max(+0.0_f64, accuracy - 0.005_f64)` with ordinary binary64 operations in that order, normalize zero to +0, and store the resulting IEEE bits plus round-tripping JSON number. This is reporting arithmetic only, never model arithmetic. Readers recompute and require exact bits, not a tolerance. No independently supplied threshold is accepted. D8 uses the same count-to-f64 reporting algorithm when testing checkpoint accuracy against target. An exact-rational threshold/comparison would be a different version and requires ratification; do not silently mix the two at a rounding boundary.

`lock_target` first reads and verifies the exact reference bytes, the canonical source/build policy and scientific completion fields. It requires canonical, actual F32, full final endpoint, seed0, declared approved design and compatible shared contract; rejects diagnostics, failures, partials, F64, synthetic markers, nonfinite values, missing provenance and wrong digest. It derives threshold/identity fields itself, captures lock creation time internally and publishes atomically at the unique no-overwrite lock path for this campaign/profile, before any paired HE case of that profile starts. `NewProfileTargetStore` rejects an existing lock or pre-existing admission/start receipts for that profile. Locks and readiness of other profiles are independent. No manual numeric reference or target field can enter through CLI options.

`verify_target_lock` reads each file once into immutable bounded buffers and checks file digests and canonical content digests, exact schema/policy versions, threshold derivation and all field cross-links. It does not trust a reference filename as identity. Re-materialization of the requested seed checks approved input sources and derives its shared contract; compare that contract, design and profile/batch to the lock. For seed0 also require exact selected-seed receipt equality to the reference. For other seeds require their own canonical receipt; equality to seed0 is specifically forbidden as an admission requirement. The returned `VerifiedTargetForSeed` contains the lock digest, shared contract, selected-seed receipt and exact threshold in private fields; it is not itself permission to allocate keys/run HE.

Wall-clock sanity checks require reference start <= completion <= lock creation <= recorded HE admission/start, with explicit clock validity. Backward/discontinuous local clocks are an admission failure requiring diagnosis; do not repair times. These comparisons detect inconsistent records but cannot establish actual historical ordering without the execution protocol below.

## 9. Prospective local admission and D7 integration

Use immutable no-overwrite files within an explicitly identified prospective campaign: one lock per concrete profile and separate admission/start receipts per case attempt. The canonical case identity includes profile, seed and registered family/backend/mode/placement/security axes; an explicit attempt ID distinguishes a separately requested retry. Receipt paths derive from validated identities under the dedicated root, not arbitrary caller paths. There is no global cohort seal, mutable event journal, shared sequence number or required ordering between different profiles.

The concrete future interface is:

```text
study_train --case <registered-case> --seed <seed> --attempt <id>
            --campaign <id> --target-lock <path> ...
    -> materializes its own seed inputs and verifies case/reference/lock/build
    -> admit_he_case(verified_target_for_seed, internally_checked_case)
       -> NewlyPublishedAdmission  // private in-process capability
    -> start_he_case(NewlyPublishedAdmission)
       -> durably published binary-owned StartReceipt, then key allocation/training
```

Both receipts are binary-owned and use section7's durable atomic no-overwrite publication. `AdmissionReceipt` binds campaign/profile/case/attempt IDs, selected seed and its canonical input receipt, shared contract, exact lock file/payload digests, reference digest, actual binary/source/build provenance, admission time and completed independent admission gates. `StartReceipt` additionally binds the exact admission file digest and internally captured start time/provenance. The same live process receives the private publication capability only after admission file and directory sync succeed; it verifies the lock bytes again before publishing start. Deserializing an old admission file cannot create that capability or start execution. No key allocation or training begins until both receipts are durably published. A start receipt records admission to execution, not successful training; subsequent allocation/training can fail and must leave failure or incomplete evidence.

Concurrent attempts to publish the same case/attempt admission collide atomically: only the successful publisher may continue. Any existing admission or start for that identity rejects replay, including a crash after admission but before start. A retry requires a new explicit attempt ID and a link to the prior attempt; preserve prior bytes and status, with no automatic replacement or silently inferred completion. Readers reject missing/truncated/mismatched files, unsupported schemas, invalid time order, changed lock bytes and inconsistent receipt/input/build links. A failure after publication with uncertain durability blocks progress under section7's policy. The binary embeds admission/start/lock digests in every cumulative checkpoint and final output. The runner schedules a request; it cannot author these receipts or supply missing accuracy/counter/runtime facts. Verified lock bytes stay immutable in memory for the process, preventing a later path swap from changing the target actually used.

A ready profile may begin after its own lock and per-case admission gates pass, while other registered profiles remain unresolved. D7 retains the complete expected matrix and every missing-input/target/code/resource/user-decision row; executing ready profiles never removes unready ones. A changed shared contract requires an explicitly new prospective campaign identity and new lock, with prior campaign receipts left intact. Later lock generation never retroactively repairs an earlier unlocked HE start or reclassifies historical results.

This establishes **local per-profile lock-before-start ordering for cooperating reviewed processes**: that profile's lock publication precedes durable binary-owned admission and start for its cases. It makes no global total-order claim across profiles. It cannot prove that no HE experiment occurred elsewhere, that no operator previously examined results, or that local wall times/files were never altered. Without an independent trusted timestamp/append-only service, describe results as prospectively locked under the recorded local process, not externally attested preregistration. A future independent service could witness a profile lock digest before admission, but no service/account/send is introduced here.

Root chose this boundary before the first independent design review. The rejected global-cohort seal would have blocked a ready profile on unrelated missing targets; its hash-chained journal would have added serialization, torn-write recovery and mutable-history handling without removing the trusted-local-operator assumption. That alternative is not an implementation finding or a required readiness gate.

D6 cumulative checkpoint integration preserves origin0/0 and every actual D2b1 endpoint. `VerifiedTargetForSeed` supplies only the immutable threshold and identity; the HE binary supplies actual cumulative times/counts/accuracy. D8 validates matching lock/case/seed inputs and conservation, reports first declared checkpoint crossing as an upper-bound detection time without interpolation, and right-censors unreached targets at final training time. B128 keeps the existing38/37-update and4864/4736-sample intervals. These APIs never alter FULL controls or infer missing counters/times.

## 10. Bounded implementation packages and evidence gates

Each package requires explicit assignment, failing tests before implementation, focused validation, a complete owned diff/hash ledger and independent review before a separate commit. This proposal assigns no code ownership yet.

| Package | Proposed bounded ownership | Required evidence before integration |
|---|---|---|
| D6c1 identity and artifact types | New `study_reference.rs`; narrow approved `lib.rs` export; only necessary immutable selector descriptors in `study_profile.rs`/`study_inputs.rs` under explicit shared ownership | Projection/encoding golden fixtures, real bounded Iris input identities across seeds, all registered profile metadata projection coverage, source-role inventory, strict parser negatives. No training to full endpoint. |
| D6c2 session, provenance and publication | Reference session/finalizer, new `study_provenance.rs`, new bench `build.rs`, narrow build-recipe helper and tests | Full-prefix/poison/checkpoint invariants, internally evaluated counts, compiler/runtime separation, argv/dirty/source/build identity errors, atomic writer fault injection and collision tests. Legacy provenance APIs unchanged. |
| D6c3 lock and prospective admission library | Lock/read/verify methods, small per-profile-store/per-case receipt module and tests; no entire matrix runner | Derivation, seed0-only source plus all-seed compatibility, forged target/hash/time/order negatives, immutable admission/start linkage, collision/retry behavior and independent profile readiness; explicit trust labels. |
| D6c4 thin reference binary | New `src/bin/study_reference.rs` and focused integration test; no `plain`/`mnist` changes | CLI rejects target overrides, bad profile/mode/dtype/paths before work; real two-update Iris diagnostic receipt/reload; full mode wired but not executed in this assignment or routine CI. |
| Later D7/HE hook and reference execution | Separately owned scheduler/start/checkpoint integration; separately authorized full reference generation | Reviewed code/source/build; complete actual seed0 full reference for each executed profile; verified per-profile lock and binary-owned admission/start receipts before its paired HE begins; full expected matrix/missing rows retained; remaining input/layout/resource/user gates independently satisfied. |

No dependency addition is presumed: use current SHA helper, `serde_json`, standard filesystem/process/time APIs and existing libc only after reviewing portability. Strict duplicate-key parsing or serialization support may need a small explicitly approved dependency/direct dependency declaration; resolve this in the implementation brief, not silently during design. Audit any Cargo changes under the existing permissive-license constraints.

## 11. Meaningful negative tests and synthetic trust separation

Use a private generic test engine/clock/filesystem and a distinct `SyntheticReference`/test policy under `cfg(test)`. A short test plan that calls itself “full” is full only within that fixture: it carries an untrusted marker that is immutable and has no conversion into production `CompletedFp32Reference`/production lock readiness. Serialization round-trip tests retain that marker. Test-only lock derivation may return a `SyntheticTargetLock` in temporary directories, never a production lock type/file namespace. Avoid a production `--allow-synthetic` flag or Cargo feature that can publish trusted artifacts. Positive production full generation remains an explicit later execution gate, not something simulated by placing the cursor at the last update.

Required focused cases:

- Fresh/one-update/last-minus-one prefixes cannot finish full; exhaustion cannot replay; poison or session checkpoint failure cannot finalize; fabricated public `PlainEvaluation` cannot be supplied; F64 cannot use the F32 final constructor; private cursor-position fixtures are never exported as scientific evidence.
- Actual bounded canonical Iris seeds0/7 share the scientific contract while initial/order/materialized receipts differ; verified selected-seed receipts round-trip. A seed7 case may consume a synthetic test-policy seed0 target without equality to seed0 input digest, but production read rejects that synthetic target. Missing captured block seeds remain missing.
- Cross profile, B48/B64/B128, registry/portable-lock, dataset/split, order recipe/selector, schedule source/topology, full versus prefix, design or arithmetic revision mismatch fail. Distinguish legitimate seed-selected rates/order changes from an altered shared rule. Do not bypass registry pins to turn a toy dataset into a production input.
- Target source seed1/7, quick/F64/best-epoch/partial/synthetic input, forged target bits/counts, wrong denominator, unknown fields/schema, duplicate JSON keys, payload mutation and file-byte mutation fail. Test zero-floor and finite binary64 rounding boundaries with synthetic integer counts only.
- Final endpoint/checkpoint metadata must match plan, including crossed boundaries and disjoint cumulative conservation; zero/extra/repeated endpoints fail. No B128 re-interpretation as exact4800-sample observations.
- Atomic writer rejects an existing sentinel unchanged and competing publishers; inject short writes, file-sync/link/directory-sync failures and interruption before/after publish; never accept temporary/failure files as success and report ambiguous durability distinctly.
- Build/runtime compiler mismatch, unknown Git state, source/executable change, misleading joined argv, missing required receipt and unsupported filesystem fail closed; optional resource data remains honestly unavailable.
- Reference-complete-after-lock, lock-after-start, replaced lock, missing/truncated admission/start file, altered receipt linkage, wrong seed/case/build and replay of an existing attempt fail admission. Competing publishers cannot both start the same attempt. A crash after admission requires an explicit new retry ID and retains prior bytes; reading an old admission cannot recreate the live capability. A ready profile can start while another profile is target-unlocked, preserving all expected matrix rows. Changed contracts require new prospective campaign identity and cannot repair old evidence. A valid timestamp sequence with forged local files cannot be proven authentic; tests and documentation must not claim otherwise.

Routine validation executes no full scientific profile and creates no scientific target. The report must distinguish schema/unit fixtures, real bounded diagnostic generation, future full-reference evidence and future HE campaign evidence.

## 12. Ratification points and residual risks

Recommend ratifying: the separate thin binary and privately owned session; dual shared/per-seed identity; exact binary64 reporting-threshold algorithm; strict separate synthetic/F64/quick types; atomic no-overwrite plus explicit durability failure; source/build/compiler provenance with an honest local trust boundary; and independent per-profile lock-before-start admission using binary-owned immutable per-case receipts. Root selected this ordering boundary before independent review and rejected the global cohort seal/journal alternative. The overall design has passed independent review and root ratification; the design revision/digest set and each bounded implementation assignment remain separate gates.

Implementation must settle the explicit immutable selector/source-role descriptor and canonical field/tag table before coding its hashes, verify local platform publication primitives, and assign shared build/module changes. A complete source/build receipt is more work than reusing the legacy JSON helper, but omitting it would leave the requested evidence gap. Final model arrays are represented by a digest, not an exported/resumable checkpoint; adding independently replayable model export is separate scope. No hash or timestamp can replace trust in the local build/operator or prove absence of prior outside observations. None of these unresolved execution facts is repaired by this design or a synthetic fixture.

## 13. D6c1a exact identity projection inventory (before implementation)

This bounded package implements identity bytes only. Policy binding is the immutable
ratified baseline `e018bd7401a6a606e306e58decad7af68cc56b84`, document SHA256
`6cdca16097d4acbf28c0f9bf0a36cccc1fccfae2fb604b0410926c066b85a035`.
That digest does **not** describe this subsequently appended document. The explicit
selector version is `hekit-study-selector-v1`; contract and receipt versions are
`hekit-study-profile-contract-v1` and `hekit-study-seed-input-receipt-v1`.
No scientific policy, registry, portable lock, preparation recipe or arithmetic changes.

Encoding is `hekit-study-reference-encoding-v1`: header strings (encoding version,
domain), followed by the following fixed-order fields. UTF-8 has u64LE byte length;
counts/indices have u64LE; flags/options/enum tags have one byte; floats use f64
bits LE (f32 receipt arrays/rates use f32 bits LE). Collections have u64LE length.
File maps sort by repository-relative UTF-8 path then explicit role; no absolute
paths, native usize bytes, JSON/Debug serialization or heuristic seed stripping.

| Shared contract field order | Exact coverage and exclusions |
|---|---|
| versions/policy | Contract version, selector version, immutable baseline revision/digest, input identity version, input-code algorithm name and existing code digest; permitted seeds 0 and 9. |
| profile/data | `ProfileMetadata.id,dataset,model,classes,output_width`; actual dataset ID/profile ID/classes/train rows/train columns/test rows/test columns/label counts; existing dataset, split and preprocessing digests; registered preprocessing/split strings and actual prep/split strings. Source row indices and numeric arrays are covered by the existing independent digests. |
| layers/stages | Every ordered `Layer.name,input,output,bias,activation,role` (including block index); every `Stage.layers` start/end, input kind/upstream layer and optional representation layer. |
| optimizer/arithmetic | `loss,optimizer` plus momentum, `weight_decay,rate_description,rate_units`; optional constant `rate` only without schedule. F32 once-cast from canonical F64, scalar row/output/input loop accumulation, loss over logical outputs, real-class-only total_cmp/last-maximum argmax (including signed-zero ordering), zero initial velocity, NAG v'=mu*v+g then g+mu*v', no cross-stage gradient and pre-update handoff are fixed versioned policy strings. |
| batching | `batch,logical_epochs,drop_last`; order kind, RNG, coverage and optional subset count; captured population/passes/coverage. |
| initialization | Local recipe RNG/distribution(+sigma), full ordered recipe layers, registered draw-order string, seed-selector policy; captured family/model/export template and expected logical/source shape/orientation per named layer. `GenerationRecipe.seed`, captured filenames/digests and missing-export availability are excluded from shared identity explicitly. |
| selector/sources | Immutable resolver-produced descriptor: selector version, relative registry/whole portable lock, shared sources, expected dataset sources; captured source family ID/model, unexpanded registered initialization/order templates and schedule pointer template. Explicit roles: registry=0, portable lock=1, shared config=2, local recipe code=3, dataset input=4, schedule container=5, selected initialization=6, selected order=7. Shared config includes registered graph/data/config/script bindings; schedule container is identified by exact bound source equality; recipe code by the two resolver-owned exact paths. Selected roles 6/7 only occur in the receipt. Every source retains relative path and digest. |
| schedule | Absent/present; if present source relative path/digest, steps, reference batch; lookup policy floor(processed_start/reference_batch), bounded by source length, rate conversion from typed rate units. Selected pointer and realized coefficients excluded only from shared identity. |
| full topology | `BatchPlan.processed_samples,dropped_samples`; each update `index,processed_start,valid_samples,source_schedule_index`; each checkpoint `completed_updates,processed_samples,crossed_logical_boundaries,exact_logical_boundary,interval_updates,interval_samples`. Numeric update source/mean/delta coefficients excluded here and retained in receipt. |

| Seed receipt field order | Exact coverage |
|---|---|
| versions/linkage | Receipt version, shared contract digest, selected `ProfileMetadata.seed`, canonical flag; existing materialized digest and each component in order weights,biases,order,plan,f64_inputs,f32_inputs,dataset,split,preprocessing,code. |
| sources/selector | All existing selected `SourceDigest.path,expected_sha256,actual_sha256` sorted by path; immutable selector bytes; selected captured initializer paths/hashes and captured order path/hash as explicit role records (shapes/orientation are retained by the shared contract link); selected schedule pointer. |
| full plan | Full topology as above plus each numeric source/mean/delta coefficient; own reference kernel rates F64/F32 and rates source/mean/delta/kernel in both dtypes. |
| selected values | Full materialized order rows (u64LE) and pass offsets; all named initialized layer shapes/weights/biases F64 then F32, cast tags 0/1 and dtype tags 64/32. No regeneration/preparation occurs. |

Enum tags are exhaustive: Model Mlp2020/Emlp1/Emlp2/Emlp3=0/1/2/3;
Activation Square/SquarePlusLinear/Identity=0/1/2; LayerRole
Hidden/Output/LocalForward(+block)/LocalHead(+block)/OutputHidden=0/1/2/3/4;
Loss MeanOutputMse/RssDelta=0/1; Optimizer Sgd/Nesterov(+momentum)=0/1;
RateUnits Mean/SumTimesValid/SumTimesNominal=0/1/2; RNG
LegacyIrisMasterFork/DomainSeparated=0/1; Distribution Normal(+sigma)/XavierUniform=0/1;
Initialization Captured/Recipe=0/1 (MissingCaptured rejects materialized production
projection; its resolver descriptor keeps the same captured family); Order
Captured/PerEpochRecipe=0/1; Coverage FullPermutation/UniqueSubset(+count)=0/1;
Orientation InputByOutput=0; StageInput Dataset/PreUpdateActivation(+layer)=0/1.
Unsupported future enum variants require an explicit version decision rather than a
wildcard fallback. Presence tags absent/present=0/1; booleans false/true=0/1.

`ProfileMetadata.identity` is fully represented through registry/lock, source-role
records, dataset bindings and split/preprocessing strings. `MaterializedIdentity`
fields are all retained by the receipt; shared identity intentionally retains only
seed-independent dataset/split/preprocessing/code. The canonical boolean gates
construction, not merely serialization. Public mutable metadata is never a trusted
constructor input. Descriptor fields are private, created only by the canonical
resolver; materialization copies the descriptor internally after its one preparation.
Private unit projections never construct or publish a trusted contract/receipt.
Future artifacts, strict JSON parser, session, build/runtime provenance, writer,
lock, target admission, CLI and actual full references remain unimplemented gates.


### D6c1a implementation details and bounded evidence

The implementation exposes only `derive_identities(&MaterializedProfile)` and immutable
contract/receipt accessors; neither value has a public constructor/deserializer.
`SharedSelector`, `CapturedSelector` and `StudySource` fields are private and originate
in `StudyRegistry::resolve`. Shared file records are sorted by `(relative path, role)`;
the encoded record is role byte, path string, expected SHA256 string. The selected
record list uses the same format. `selector` is encoded inline: selector version,
source list, captured option tag, then family/model/optional init template/order
template/schedule template. The receipt retains the shared contract digest as the
shape/orientation and policy linkage instead of duplicating those shared fields.

The exact arithmetic strings bind forward and gradient traversal as row/output/input,
separate product/add operations, division after gradient sums, zero velocity, native
F32 operations and D1's last `total_cmp` maximum including signed zeros. This corrects
an initial inventory wording error before encoding; the D6b arithmetic is unchanged.
The selected constant rate option is present only when no schedule exists. Every
plan topology field is emitted in both identities; the receipt additionally includes
the three stored update coefficients and both dtype views' four rate coefficients.
Domain headers contain the contract/receipt version; that version is repeated as the
first payload string intentionally. These bytes are new identity data, not a rewrite
of any historical input/result JSON. Existing input-code hashes change because the
owned source files changed; existing array/order/plan recipes and version remain intact.

Focused tests cover actual canonical Iris seed0/7 for all four graphs; all21 profiles
and all10 seeds' metadata-only shared source roles; independent Python-packed minimal
plan bytes/SHA256; every plan topology/coefficient field's scope; private metadata,
content/code/policy/selector/source/registry/lock mutations; B48/B64/B128 preflight
plans and 38/37-update B128 intervals; and rejection of private noncanonical assemblies.
The current released eMLP1 seed0/7 numeric schedules are equal; a first test assumed
they differed, failed, and was corrected to preserve the observed source equality.
Private synthetic coefficient changes independently demonstrate that numeric receipt
encoding changes while shared topology encoding does not. No seed differences are
invented and no synthetic projection constructs a trusted contract.

This package creates no artifact schema/parser/writer, execution session, provenance
receipt, target, lock, admission, CLI, full reference or campaign result. Source/code
bindings are local identity, not external attestation. Final validation commands and
frozen owned patch/hash manifest are in the D6c1a implementation report. Independent
review passed for this bounded identity package; integration is recorded separately.
