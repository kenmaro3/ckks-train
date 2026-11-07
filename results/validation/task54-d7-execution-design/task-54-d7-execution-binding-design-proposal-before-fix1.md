# D7 remaining descriptors and owned CPU continuation — proposal

2026-11-07. Read-only architecture proposal, not ratification, gate evidence or execution release. Basis: task-54-brief.md; current source at the stated BASE9f04ecd plus frozen descriptor candidate (source report a15905ad10d0c7dbedff9a3193c395a0cb8483d7e0c9f81b017eeac0f6953636), and current target/runtime contracts. No constructors, planners, keys, tests, native queries, network or Git operations were run. Only this report was written.

## 1. Concrete package and existing bytes

Build one private CPU preparation/continuation adapter, factoring the existing P4/P5 validation blocks so the SAME checked preparation supplies descriptors and later actual construction. Retain P1/P3 bytes unchanged. Add four narrow inner encodings: interactive parameters, physical layout, key policy and security declarations. Reuse the approved source manifest for implementation identity; a second implementation manifest is unnecessary.

| ExecutionBinding field | Exact source and ownership |
|---|---|
| implementation | main_diagonal=0 only for this adapter; CPU Main/BatchedDiagonal exact registration checked by checked_mode; unsupported roles fail |
| implementation_revision | approved complete source commit from VerifiedStudyTrainContext's private core approval/capsule linkage, Text128; add borrowed accessor, no caller revision or Git invocation |
| implementation_manifest_sha256 | SHA256 of existing canonical source Manifest bytes in current FrozenReceipt receipt.files[0], equal approval.source/source_manifest_sha256(); retain context/receipt, not a new subset manifest; includes the reviewed adapter source |
| parameter_identity_sha256 | NI: hash BootstrapMetadata::parameter_identity_bytes() unchanged. I: new I-parameter bytes below; P1's encoding requires BootParams, so cannot truthfully synthesize an I BootstrapMetadata/helper using default bootstrap parameters |
| bootstrap_descriptor_sha256 | NI: hash exact retained BootstrapMetadata::descriptor_identity_bytes(); I: None; compare actual BootParams to metadata.boot_params() and P3 secure_parameters |
| layout_plan_sha256 | new physical-layout descriptor from actual checked preparation geometries plus exact plan.partitions(); geometry_identity(metadata) alone omits lanes/slots/parts |
| trainer_plan_sha256 | hash complete DiagonalPlan::canonical_bytes() unchanged, including canonical bindings, rates, full/consumed schedule, parameters, initial/update/handoff typed metadata; retain same Arc, not full_plan_sha256 or consumed_sha256 alone |
| eval_config | actual checked geometry.eval_config(), all stages equal; match request/registration and actual Evaluator::with_config input |
| bootstrap_engine | I none=0; NI CPU host=1, actual make_material Bootstrapper branch; never infer successful construction pre-start |
| key_policy_sha256 | new exact key descriptor below, derived from checked geometry/plan/P1; factory takes its exact training rotations |
| security_policy_sha256 | new declaration/policy descriptor below; never a security estimate or passing gate |
| physical_binding_sha256 | existing candidate RecheckedTrainCapture::physical_descriptor(), same retained live context; actual production physical capture exists |
| runtime_policy_sha256 | existing ratified bytes, new private producer combines same lease facts with closed PreparedCpuRoute branch; no CLI/PolicySelection-to-readiness conversion |

The source manifest is intentionally broader than the implementation; exact commit plus complete approved source avoids accidental omission of transitive Rust/math/build sources. Existing build/receipt/executable checks still apply. A docs-only HEAD label is not the approved current implementation revision. Parsed or independently recreated equal hashes do not substitute for retained owners.

## 2. Exact new encodings

Each new descriptor starts `Text("hekit-study-reference-encoding-v1"), Text(domain)` and has no self-hash. Existing document primitives: U64 little-endian, Bool one byte0/1, Hash Text64 lowercase SHA256, Bytes U64 length/raw, List U64 count/ordered values, Option00/01. I32 below means four little-endian two's-complement bytes, matching P3 int_steps; never reinterpret P1's eight-byte signed rotation encoding. All new canonical descriptors <=A_MAX=2,097,152 bytes. Two-pass checked length/count accounting before allocation; reject overflow, wrong tags, trailing bytes and mismatches. No new JSON envelope/parser or generic registry.

### Interactive parameters

Domain `hekit-study-interactive-parameters-v1`.

| Exact body order | Source/constraints |
|---|---|
| q:List<U64>; p:List<U64>; q_bits:List<U64>; p_bits:List<U64>; log_n:U64; scale_bits:U64; dnum:U64; main_weight:Option<U64> | CkksParams q/p/q_bits/p_bits/log_n/scale().to_bits()/dnum/hamming_weight; each bit vector has corresponding prime count; nonempty chains, finite positive scale, checked constructor invariants; production additionally exact P3 n16_train and main_weight=None |

Lengths bind maximum level (q.len()-1), N=2^log_n and slots=N/2; no inconsistent redundant level field. NI retains all original P1 bootstrap parameters/proof bytes and original domains. Rejected: preset-name hash loses actual primes; copying P1 with invented BootParams lies about I; changing P1 breaks existing identity compatibility. Cost of wrong choice is accepting different cryptographic objects under the same binding.

### Physical layout

Domain `hekit-study-diagonal-layout-v1`.

| Exact body order | Source/constraints |
|---|---|
| parameter_identity:Hash; consumption:byte Full0/Quick1; mode:byte I0/NI1; eval:{Bool hoist,bsgs,lazy_relin}; stages:List<Stage>; parts:List<List<Part>> | all from same checked plan/preparation; stages1..3; each Stage in graph order; parts exactly consumed_updates, nonempty |
| Stage | dims:List<U64>, followed directly by existing P3 encode_geometry body, byte-for-byte unchanged |
| Part | start:U64, valid:U64; ordered contiguous from zero per update; 1<=valid<=actual layout.batch; sum equals canonical update sample count |

Existing `encode_geometry` body is precisely: slots,d,batch U64; shapes List of eight U64(d_out,d_in,n_out,n_in,l,m,baby,giant); masks List of four U64(canonical_period,gradient_period,bias_period,lanes), reduce_steps List<I32>, replication_steps List<I32>, compact_first_period Bool; transpose List<List<(source U64,steps List<I32>)>>; rotations List<I32>; compact Option containing plans List<Compact>, containers List<(layer Option<U64>,index U64,plan Compact)>, biases List<(container U64,position U64)>. Compact is five U64(d_out,d_in,batch,diags_per_ct,lane_shift). Borrow/factor this encoder without changing old transcript bytes. Stage dims must equal adjacent logical shape dimensions; layouts common, slots=params.slots(), positive powers of two, checked d*batch divides slots; all masks/transpose/compact recipes equal the actual StageGeometry. Preserve ordered rotation chains; never sort them. This is the actual diagonal route's StageGeometry, not unrelated PackedGeometryPlan fabrication. No packed adapter is created.

Architecture, activation, bias, stage graph and state levels already belong to the complete trainer descriptor; layout identity covers physical mapping and consumed partial lanes. Rejected: geometry_identity alone, or the whole trainer digest in both fields, loses the independent physical projection. Derivation of geometries from immutable inputs remains validated, not a digest-only equality check.

### Key policy

Domain `hekit-study-cpu-diagonal-keys-v1`.

| Exact body order | Source/constraints |
|---|---|
| parameter_identity:Hash; bootstrap_descriptor:Option<Hash>; transpose:Transpose; training_level:U64; training_rotations:List<I32>; training_relin:Bool; training_conjugation:Bool; training_sse:Bool; bootstrap_generated:List<I32>; bootstrap_execution:List<I32>; bootstrap_conjugation:Bool; bootstrap_sse:Bool; separate_bootstrap_bundle:Bool; owner_policy:byte | Transpose Dedicated00 or Reuse01+List<I32> in exact request pool order; training level params.max_level(); sorted normalized nonzero training union equals plan.keys and every checked geometry requirement; true,false,false training flags; bootstrap lists/bools equal P1/plan KeyManifest; separate bundle equals KeyManifest::separate_bootstrap_bundle(); owner_policy=0 only, meaning one main key generator per case, one training evaluator bundle and separately constructed bootstrap bundle |

I requires bootstrap absent, empty bootstrap lists and false bootstrap flags/separate bundle. NI requires exact P1 descriptor and generated/execution sets separately; neither list substitutes for the other. make_material currently generates public/secret/relin/Galois material from that one main KeyGenerator and separately calls Bootstrapper::new. Factory rotation input must EQUAL the manifest union, strengthening the current subset-only make_owned_fields check on the D7 path. Reuse pool identity and actual transpose chains remain distinct. No secret key bytes, RNG state or public-key hashes are serialized. Rejected: union alone loses reuse and separate bootstrap roles; claiming one shared evaluation bundle misdescribes allocation and security assumptions.

### Security declarations

Domain `hekit-study-main-security-policy-v1`.

| Exact body order | Source/constraints |
|---|---|
| registered_policy:byte n16_train0/n16_boot1; parameter_identity:Hash; bootstrap_descriptor:Option<Hash>; main:Secret; ephemeral:Option<Secret>; assurance:byte; bootstrap_scope:Option<byte>; bootstrap_unvalidated:List<byte> | Secret DenseTernary00 or FixedWeight01+U64; actual parameter declaration, and NI P1 declared_main_secret/declared_ephemeral_secret; assurance=0 means declarations_only; NI scope Some(0)=StructuralCostAndBoundaryScales and requirements exactly P1 ordered tags0..8; I scope None/list empty/ephemeral None |

Requirement tags0..8 are exactly P3 encode_params' P1 mapping: encoding-Q/2, EvalMod-internal-scale, encrypted-input-range, approximation-precision, actual-secret, key-binding, security-estimate, runtime-memory, backend-support. Empty I bootstrap list means bootstrap inapplicable, never general security validated. assurance0 explicitly leaves actual-secret/key/security-estimate checks unproved for BOTH modes. Fixed weights must be positive and <=N and equal the actual declarations. Reject sparse-main substitution under current registered policy, changed ephemeral weight or reduced-level substitution even if hashes are recomputed. No numerical security strength or evidence hash is invented. Rejected: policy name alone omits actual secrets; a pass bit creates phantom authority. This descriptor records limitations, while live gate adapters must independently satisfy applicable existing policies.

## 3. Ownership and actual CPU handoff

Add private `study_execution.rs` orchestration and private `study_reference/document/execution_descriptors.rs`; tiny existing-module changes only. Keep P/R candidate encoders frozen except reviewed fixes or a narrowly reviewed producer hook. Add borrowed approved-source view in context.rs. Factor validation into `prepare_canonical` and `prepare_blocks_canonical` returning existing CheckedPreparation/CheckedBlockPreparation; existing prepare_then_make/prepare_blocks_then_make delegate to those functions unchanged semantically. Existing private callback stays internal; no new public callback or plugin framework. Keep array encryption/owned continuation helpers in study_client.rs and its blocks.rs child so private fields and canonical_arrays remain owned there; expose only crate-private borrowed descriptor views and fixed continuation calls to study_execution.rs.

Inside installed `run_owned` frame retain context, canonical materialization, RegisteredCase, immutable request, Arc<CkksParams>, optional CpuBootstrapDescription, same Arc<DiagonalPlan>. Then create private `PreparedCpuRoute<'a> = Single(CheckedPreparation<'a>) | Block(CheckedBlockPreparation<'a>)`, borrowing parameters from those outer locals. Descriptors borrow this route/plan/context or retain bounded immutable bytes. It is explicitly non-authorizing. Keep locals in nested scopes; do not put a self-borrowing context/session/facade in one struct. Seal its fields and keep context lifetime/non-Send marker; codecs accept borrowed fact projections, cannot construct PreparedCpuRoute.

Pre-key work: canonical input capture/recheck, bounded parameters/P1 structural metadata, P3 plan, exact candidate checks, descriptor bytes, target inspection. No KeyGenerator, Encoder, Bootstrapper, encryptor, session or native runtime allocation. Reserve original request allowances before plan hashing. Factor out rather than repeat candidate preparation after start; retain actual checked geometry, never candidate placeholder initial states. PlanOwner/BlockPlanOwner keep the exact canonical Arc; all initial/update labels come from DiagonalPlan.

The future genuine `begin_checked_he(StartedHeCase)` consuming boundary must recheck same process/context/capture, case/attempt/target links, exact route/parameter/plan owners, descriptors and live gates/resource lease, then transfer the retained prepared route into the following fixed body. It cannot accept a parsed start file, selector, descriptor or test permit. This package does NOT implement missing gates or durable admission/start: until those exist the production dispatcher must still return MissingExecutionDependency. Implement the real private continuation body now, disconnected like current prepare_owned_canonical; do not fabricate StartedHeCase or a successful adapter callsite to compile around this dependency.

Single body: use retained CheckedPreparation with make_owned; clone canonical initial arrays only after start; client.encrypt_initial; I client.refresh_service or NI context-owned Bootstrapper; CpuDiagonalSession::new with same Arc; each logical index client.encrypt_update then session.step exactly once. Replace prepare_owned_canonical's duplicate validation with this retained-preparation continuation, retaining the existing private validation wrapper for callers/tests.

Block body: make_blocks_owned from retained CheckedBlockPreparation and exact union; canonical_arrays then encrypt_block_initial yields initial+BlockEncryptionBinding; local CpuBlockEvaluation::new(&context); optional same-client refresher; CpuBlockSession::new; client.encrypt_block_update(&binding,...) then step sequentially. Drop session before facade/context/client. Recheck owner/context/resources before allocation and each execution entry. Poisoned failures stop; preserve failure counters, no retry/rollback. Block snapshot only immediately after successful update; never before first success or after poison without a separately reviewed API rejection/fix. Failed-endpoint publication cannot assume coherent old-hidden/model state.

CPU runtime producer derives refresh_route0/1 from this closed I/NI continuation and lease actual workers/form/recipe; binds retained P bytes. Pre-start runtime bytes describe this reviewed construction requirement, not an existing Bootstrapper. Post-start actual config/branch/owner checks remain mandatory. CUDA/native/external/matched/packed remain unsupported adapters, never CPU fallbacks.

## 4. Accounting, work inventory and remaining gates

Reuse P1/plan/source bytes by borrow/Arc, charging once; new descriptor Vec payload, digest strings, retained checked geometry nested vectors and duplicated factory arrays must be counted explicitly. Charge checked arithmetic before growth against the ORIGINAL request's retained budget, including plan.charges().combined(); do not silently reset max_bytes or increase it after hashing. New auxiliaries also individually obey A_MAX. Existing plan ReceiptLimits and context/P1 caps remain unchanged; a large existing transcript is not copied into a capped auxiliary field. Count temporary preparation, target verification's independent materialization and later array/key copies separately; requested-payload accounting is not RSS/resource admission. New ownership structs must include nested capacities in retained accounting; no unobserved diagnostic CT copies become measured full-RSS evidence.

Suggested bounded work: (1) pure encoders/accessors and independent literal vectors; (2) preparation factoring/retained accounting, no HE; (3) private concrete CPU continuation and refusal tests; (4) separate real gate/admission/start integration. Freeze test constructor/work inventory before execution, with no broad campaign or automatic HE reruns implied by this design. For one successful preparation with S stages and U consumed updates, existing P3 plus one retained candidate costs 2S geometries, 2S initial-state calls, S*U update calls, (S-1)*U handoffs and one plan wrapper; every extra rejected/repeated planning call needs its own budget. Binding revalidation/encoding work and descriptor allocations remain separately bounded even where existing counters do not charge them. Meaningful tests: every new field/order/tag/length mutation; unchanged existing P1/P3/P/R vectors; distinct physical lanes/parts and transpose choices; equal-hash but different Arc/context/params refusal; different runtime pool/route or source manifest refusal; wrong BootParams/EvalConfig/bootstrap branch; exact rotation union rejection; no early key/Encoder/evaluator/Bootstrapper/session calls on every refusal. Independent vectors must not be generated by production encoders. Synthetic values stay synthetic, never canonical positive evidence.

The genuine dependency is capability construction and durable-start ownership, not a hash cycle: parameter/layout/key/security/runtime descriptors never hash ExecutionBinding or Gates. Existing eight P3 unproved gates, P1 limitations, five live outer Gates, target/admission/start publisher, resource lifetime/recheck and scientific HE metrics/output remain unimplemented by this package. Full encrypted block user hold remains. Source approval for changed implementation is separate from source design; B/same-N/authors unresolved decisions do not prevent ordinary descriptor/CPU adapter development. Preserve all21 profiles, full2680 family rows +20 exact-Iris prospective rows +7 supplement entries and quick268 projections. No new dependencies, external baseline code or license exceptions beyond already approved MIT/Apache2/BSD/ISC plus Unicode tables.
