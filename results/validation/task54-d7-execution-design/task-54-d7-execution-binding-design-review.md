# D7 execution-binding design — independent SPEC / QUALITY review

2026-11-07. **CHANGES REQUIRED before the whole proposed source package is implementation-ready.** No Critical finding; two Important findings and two Minor clarifications. This is a bounded architecture review, not design ratification, source authorization, gate clearance or execution release.

Reviewed proposal: `task-54-d7-execution-binding-design-proposal.md`, SHA256 `25d88851549d4ad85989572b7a541745dd2f240545f434b490bae7fb0cedebe0` (verified by read-only hashing). Read the Task 54 brief first, then the design brief, proposal, binding target/runtime contracts and named source edges needed to assess the proposed ownership and encodings. No source/Cargo mutation, test/constructor/native execution, network/Git operation or subagent was used. Only this review file was written.

Current evidence qualification: the controller reports the unchanged six-file P/R candidate committed as `d0cf11456a592695f1b0a5fd5ab477644dad372d`, after source review `e5d06127`, 9/9 pure tests and outcome review `00f1d530`. Those results were not rerun here. The inspected codec still explicitly calls its selections inspection claims, not checked routes (`study_reference/document/runtime_descriptors.rs:350-357`). No native Linux integration or runtime-checked route follows from that source/test status. Proposal line 3's older candidate status is historical.

All source paths below are relative to `crates/ckks-train-bench/src/` unless a full repository-relative path is shown. Proposal line references refer to the frozen file above.

## SPEC assessment

The proposed descriptor split is coherent and appropriately narrow:

| Field / encoding | Assessment against actual source |
|---|---|
| Implementation revision / manifest | Reusing current approved complete manifest bytes is appropriate. Context checks capsule commit and source against approval before retaining the receipt (`study_provenance/context.rs:720-729`); current-receipt selection exists at 1258-1263 and exact approval source accessor at 815-816. Select that current Train receipt, never an arbitrary retained Reference receipt. No second source-manifest schema is necessary. |
| NI parameter / bootstrap identities | Reuse P1 bytes unchanged. P1 parameter bytes really include BootParams, so they cannot be truthfully manufactured for I (`crates/ckks-train-ckks/src/bootstrap/metadata.rs:1527-1557`). |
| New I parameter encoding | Its ordered q/p, bit vectors, log N, exact scale bits, dnum and optional main weight cover the required actual parameter identity. Production equality remains `secure_parameters`, including dense main key and I bootstrap absence (`study_admission.rs:891-936`). New domain rather than invented NI metadata is justified. |
| New physical-layout encoding | Stage dimensions plus the existing exact `encode_geometry` body and consumed partitions cover the physical mapping omitted by architecture metadata. The proposed P3 body matches `study_admission.rs:2031-2082`, including per-mask steps, ordered transpose chains and compact occupancy. Preserve that body byte-for-byte when factoring it. |
| Trainer identity | Complete `DiagonalPlan::canonical_bytes()` is an appropriate distinct transcript. Construction retains binding, consumed/full identities, rates, partitions, initial state and transition index (`study_admission.rs:1889-1975`); neither the full-schedule digest nor the consumed-schedule digest alone suffices. |
| New key policy | Distinguishes exact training union, request reuse pool, generated/executed bootstrap sets and separate bundles. These distinctions exist in `KeyManifest` (`study_admission.rs:382-415`). The stricter equality proposed at line 63 is necessary: the present factory accepts a superset (`study_client.rs:713-715`). Training relin/Galois and separate Bootstrapper construction are real (`study_client.rs:304-340`). |
| New security declarations | Dense/fixed-weight tags and optional ephemeral declaration correspond to P1, with explicit declarations-only assurance. P1 ephemeral absence remains `None`, not dense by default (`crates/ckks-train-ckks/src/bootstrap/metadata.rs:923-957`). Preserve the nine ordered limitation tags from `study_admission.rs:999-1011`. No security result is established by this record. |
| P/R and EvalConfig / engine | Retained physical bytes and a closed prepared CPU route are the correct producer inputs. Route facts must still be checked against actual post-start construction. Runtime codec selector checks alone (`runtime_descriptors.rs:371-407`) cannot produce that ownership. |

No hash cycle was found: new inner descriptors do not hash ExecutionBinding/Gates; P has no R link; R links only to P. Whole trainer/source bytes can remain borrowed or Arc-owned. Proposal lines 25, 79 and 83 correctly distinguish retained owners from independently recreated equal digests.

The pre-key / post-start separation is directionally correct. Fact capture, checked immutable plans and descriptor projection precede authority. `make_owned_fields` checks actual parameter pointer and bootstrap identity before the first `KeyGenerator::new` (`study_client.rs:701-725`). The proposed fixed single/block bodies use genuine current factories and sessions. P5's snapshot restriction is correctly preserved: current snapshot indexes `completed.saturating_sub(1)` and reads handoffs without a successful-state guard (`study_blocks.rs:1094-1130`); it must remain success-immediately only, with no origin/poison publication inferred from it.

## Findings

### Important I1 — the return ownership of the ratified consuming boundary remains unspecified

**Proposal lines 77-87; binding `docs/study-target-design.md:137-178,214-220`; actual `study_diagonal.rs:372-395`, `study_blocks.rs:205-221,323-343`.**

The proposal names `begin_checked_he(StartedHeCase)` and then describes a fixed body that creates local client/context/refresher, a local block facade, and a session borrowing those locals. Nested scopes and drop order correctly allow a run-to-completion body, but they do not define how this body satisfies the ratified `begin_checked_he -> HeSession<'c,'p>` return. Actual P4 borrows the evaluation context/refresher; P5 additionally borrows `CpuBlockEvaluation`, which itself borrows that context. Returning those sessions together with their local owners would be self-referential; returning a plain completed outcome instead silently changes the ratified boundary. The proposal excludes a self-borrowing struct but does not supply the remaining type/lifetime handoff.

**Required resolution before continuation source:** give a small explicit signature/ownership sketch for the returned driver and its fixed consuming execution method. Identify what `HeSession` owns, what it borrows, when the checked route is consumed, and which nested frame owns the evaluation context, refresher and facade while P4/P5 run. A returned owned driver with a private fixed method that creates/runs/drops the borrowed session is one possible resolution; it must not expose an allocator permit or public callback. The sketch must retain ProcessStamp, Train context/input and gate/resource leases through each allocation/execution entry, and release the store borrow only after the ratified checked transfer. No unsafe lifetime extension, leaked owner or fake StartedHeCase may close this gap.

Also name the exact post-start plan revalidation call in that sketch. `validate_plan_binding` explicitly requires the real P4/D7 consumer to call it after gates/start and before keys (`study_admission.rs:1978-1997`). Retaining pre-start `CheckedPreparation` avoids repeating candidate geometry construction; it does not make that source requirement disappear. Proposal line 83's general “recheck ... descriptors” should identify this call and its temporary allocations.

### Important I2 — four new descriptor vectors are not frozen before encoder work

**Proposal lines 29-73 and 95; `docs/study-runtime-design.md:199,247`; design brief lines 9-13.**

The four domains and most field/tag order are specified, but the proposal contains no independent complete bytes/length/digest vectors. Line 95 puts “pure encoders/accessors and independent literal vectors” in one suggested work item and does not require independent vectors to be frozen first. Thus the package cannot yet establish an independent byte oracle before hash-producing implementation. A round-trip or expected bytes generated by the production encoder would not close this requirement.

**Required resolution before hash-producing source:** freeze and independently review complete literal vectors with input values, exact canonical hex, byte count and SHA256. Cover each new domain and every structural branch needed by its format: I parameter main-weight option; layout compact absent/present, multi-stage and partial partitions, negative ordered transpose steps; key Dedicated/Reuse and I/NI generated-versus-executed/bootstrap-option branches; security I/NI with ephemeral option and fixed-weight tag. Synthetic branch examples may be explicitly unsupported by current production policy; never label them canonical secure-profile evidence. Include exact expected failure cases for changed tags/order/lengths, invalid counts/weights, overflow and A_MAX. Existing P1/P3/P/R vectors stay unchanged. This is a representation prerequisite, not a new scientific/user approval gate.

### Minor M1 — make the additive budget owner and charge convention explicit

**Proposal lines 29,81,93-95; `study_admission.rs:63-76,467-519,1258-1283`; `study_client.rs:173-182`; `study_client/blocks.rs:84-120`.**

The proposed accounting principle is sound, and the successful pure-planner count `2S geometry / 2S initial / S*U update / (S-1)*U handoff / one wrapper` matches the named plan-plus-candidate path. Current `ReceiptLimits` is a per-transcript requested-payload cap, however, and `describe_candidate` plus the preparation clones have no shared retained-byte charge. Saying “against the original request” alone does not identify where additive usage lives. Line 93 also switches from existing requested payload to nested capacities without defining whether old plan charges remain length-based.

The source brief should identify one private additive preparation-accounting owner, seeded with existing plan charges exactly once, and enumerate the extra retained and peak-transient charges before each fallible reservation/copy. Specify whether accounting measures requested lengths or observed capacities, how capacities exceeding reservations are handled, and which separate budget covers materialization/parameter/P1/target-verification work. Preserve existing `ReceiptLimits` semantics and bytes; do not retroactively call them an RSS bound. Two-pass descriptor sizing must not first clone full geometry/partitions just to calculate its bound. This can be resolved in the bounded implementation brief without an unrelated framework or new scientific cap.

### Minor M2 — reuse the existing approved commit accessor

**Proposal lines 12,77; `study_provenance/context.rs:1290-1292`.**

`VerifiedStudyContext::source_commit()` already returns the borrowed approved commit. The Train context already exposes its exact artifact context (189-191). A new commit accessor is unnecessary. A narrow borrowed current source-manifest-byte view may still be needed; qualify it to the current Train receipt and keep the existing rechecks. Update the source inventory accordingly rather than expanding provenance APIs redundantly.

## Required before-code constants and acceptance record

The proposal already specifies the four domain strings, encoding header, LE integer widths, Option/Bool tags, A_MAX=2,097,152, Full/Quick, I/NI, transpose, secret and declarations-only tags. No additional scientific threshold is missing. Freeze these verbatim together with I2's vectors; in particular `scale_bits` means IEEE754 `scale().to_bits()`, not the constructor's exponent argument, and I32 is four-byte signed LE while existing P1 rotation bytes remain unchanged.

For an implementation brief, expand “checked constructor invariants” into the precise existing source constraints or immutable references: parameter log-N range and bit/exponent ranges, nonempty Q/P and dnum divisibility (`crates/ckks-train-ckks/src/params.rs:84-115`); production equality to the current n16 presets; `main_weight=None`; exact default NI BootParams and P1 declaration equality. State that all list lengths are bounded by checked total-size computation before allocation plus their source shape/cardinality invariants. No arbitrary new parameter, rotation or worker cap should be invented.

Freeze bounded test counts/constructor inventory before any test execution. Required acceptance remains independent bytes/mutations, unchanged old bytes, same-Arc/context/parameter/route refusals, exact union rejection and zero early key/Encoder/Bootstrapper/session construction on refusals. Compile-check the resolved I1 ownership sketch without native setup; synthetic tests cannot issue production start authority. Post-start binding revalidation needs its own bounded allocation/work entry even though it does not consume another geometry/update planner call.

## QUALITY verdict and preserved scope

The proposal is focused, reuses actual source, rejects digest-only authority and preserves unsupported adapters. It is suitable for a narrow revision addressing I1/I2, with M1/M2 carried into the source brief. It is not yet approval to execute the complete proposed package. No finding requires redesigning the registered study or resolving B/same-N/authors first.

All eight existing P3 unproved gates, P1 limitations, five outer live gates, durable admission/start, native/runtime integration, measurements and scientific publisher remain outstanding as applicable. Full encrypted-block user hold remains. Preserve all 21 profiles, full 2680 family rows + 20 exact-Iris prospective rows + seven supplement entries and 268 quick projections. CUDA/native/external/matched/packed absence must remain visible rather than narrowing the denominator. No new dependency or external baseline code is justified by this review.
