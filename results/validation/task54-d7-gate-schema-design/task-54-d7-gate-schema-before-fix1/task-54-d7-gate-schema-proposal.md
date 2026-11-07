# Task54 D7 gate evidence and Quick operational bytes

Status: unratified pre-encoder proposal and independent synthetic byte oracle, 2026-11-07. Read `task-54-brief.md` first. Implements the representation duties of ratified `docs/study-gates-design.md` §§2–6, the FIX1 live-gate proposal §§3–5/8–10 and custody proposal §9. This is not a gate, live capability, source release, resource model, scientific result or experiment authorization. The five-capability decision, Full/Quick split, custody namespace and Full wire are unchanged. Root independent SPEC/QUALITY review is required before ratification/source.

## 1. Boundaries, primitives and exact domains

Use `docs/study-target-design.md` §3 primitives unchanged: U64 little endian; tag/Bool one byte; Text U64 UTF8-length plus bytes; Bytes U64 length plus raw bytes; Hash Text of 64 lowercase SHA256 hex; List U64 count plus values; Option 00 or 01+value. Text128 identifiers retain `[A-Za-z0-9_-]{1,128}`; revision identifiers are Text128, Git40 is exactly 40 lowercase hex; other text is Text4096. All counts use checked U64→usize conversion. No unspecified fields, extensions, pass booleans, default values, sentinel hashes, fallback revisions or trailing bytes. Hashes may numerically be zero only if derived from actual checked bytes; zero threshold is legal only from the independently verified Full target. Every new record begins `Text("hekit-study-reference-encoding-v1"),Text(domain)`.

Production evidence domains are exactly `hekit-study-gate-inputs-evidence-v1`, `hekit-study-gate-implementation-evidence-v1`, `hekit-study-gate-layout-evidence-v1`, `hekit-study-gate-resource-evidence-v1`, `hekit-study-gate-user-evidence-v1`. Immediately after header encode `synthetic:Bool=false`, then the common prefix below. Private oracle substitutes `hekit-test-` for the leading `hekit-study-` and encodes true; production rejects either test domain or true, including mixed-domain nested records. No mode flag enables synthetic production parsing.

Additional exact production domains: `hekit-study-quick-identity-v1`, `hekit-study-gate-request-v1`, `hekit-study-gate-consumption-v1`, `hekit-study-user-scope-v1`, `hekit-study-user-current-v1`, `hekit-study-quick-preflight-v1`, `hekit-study-quick-start-v1`, `hekit-study-quick-end-v1`. These also encode synthetic immediately after their header. New request/consumption wrappers are projections of retained existing objects, not alternative planning or authority. Existing CaseIdentity/case-binding/parameter/layout/key/security/trainer/runtime domains and their bytes remain unchanged, even when nested inside a synthetic outer fixture.

Each binary evidence whole file is canonical bytes, ≤A=2,097,152 including all headers; its whole-file SHA256 equals the outer Gates evidence hash. No embedded self-hash or newline. Each new auxiliary binary record is also ≤A and subject to its enclosing record budget. Text maximum4096 (identifiers/revisions128), list count≤65,536, argv count≤256, canonical input source-file counts≤65,536; approved implementation source-manifest records≤512 (existing source-role cap), feature lists≤64 and argv≤256; aggregate checked byte accounting wins over individual caps. Reuse existing stricter type-specific limits. Count/length products are preflighted before allocating, nesting is fixed by the tables, and slices are borrowed during bounded decoding where possible. Existing C/R/Full limits do not increase. A huge receipt is represented by verified identity/length, not copied into a sidecar.

Quick JSON files have raw file cap A and decoded canonical cap K=1,048,000. The deliberately smaller K accommodates lowercase hex transport without raising A; whitespace can still exceed A and must reject. Exact JSON envelope keys are `schema,kind,synthetic,canonical_bytes,integrity`; integrity keys exactly `encoding,hash,payload_sha256`. schema equals the canonical header domain; kind is `quick_preflight`, `quick_start`, or `quick_end`; synthetic matches the canonical Bool; encoding is the fixed primitive version; hash is `sha256`; canonical_bytes is unescaped, even-length lowercase hex. No other keys or decoded duplicate keys. Payload digest is SHA256(canonical_bytes), including header and synthetic flag. Entire exact JSON file has independent file SHA256; do not hash normalized JSON or include the file digest recursively. Writer uses compact UTF8 JSON plus one LF; decoder accepts bounded legal whitespace. Envelope strings and integrity digest do not appear again inside canonical bytes. This is an operational envelope, not a new scientific result schema or a modification of Full Integrity.

## 2. Reused and new closed subrecords

Field order in every following row is normative. Records concatenate fields; only List/Option/Bytes introduce their declared framing.

| Record | Exact ordered fields |
|---|---|
| Link | `file_sha256:Hash,payload_sha256:Hash` unchanged target §3 |
| Clock | `wall_source:tag0,unix_ns:U64,monotonic_source:tag0,elapsed_since_operation_start_ns:U64` unchanged target §3; same live operation origin across Quick events |
| Predecessor | `attempt_id:Text128,admission:Link,start:Option<Link>` unchanged target §3 |
| IdentityRef | `schema:Text128,encoding:Text128,canonical_length:U64,sha256:Hash`; actual retained verified contract/receipt identity, existing C or R cap respectively, schemas exactly `hekit-study-profile-contract-v1` / `hekit-study-seed-input-receipt-v1`, encoding exactly primitive version |
| FileRef | `root:tag,path:Text4096,length:U64,sha256:Hash`; root Input0/Source1/Reviewed2/Policy3/Authority4/Observation5; path nonempty UTF8 relative path, no empty/`.`/`..` components, NUL, backslash or absolute prefix; exact bytes, no normalization |
| Applicability | `implementation_manifest:Hash,implementation_revision:Git40,parameters:Hash,bootstrap:Option<Hash>,keys:Hash,security:Hash,layout:Hash,trainer:Hash,runtime:Hash,implementation:tag,backend:tag,mode:tag,consumption:tag` |
| Part | `start:U64,valid:U64`, existing P3 semantics |
| QuickUpdate | `index:U64,epoch:U64,batch:U64,sample_indices:List<U64>,parts:List<Part>`; exactly the canonical logical update, not sorted; two updates indices0,1; nonempty samples, partitions contiguous from0, positive valid≤physical lanes, exact sample-count sum |
| Gates | fixed `inputs,implementation,layout,resource,user`; each `policy_revision:Text128,evidence_sha256:Hash`, unchanged target §5 body |

Applicability enums exactly reuse CaseIdentity/ExecutionBinding tags in target §5. Consumption is Full0/Quick1. Bootstrap Option follows actual I absence/NI descriptor; absence is not a placeholder for missing NI. Comparisons use the actual retained source/params/plan/key/security/runtime descriptors, including all cryptographic declarations, not preset strings. `implementation_revision` must equal actual source commit; the ExecutionBinding text is compared to that approved revision, not interpreted as an arbitrary token.

`Request` uses new request header/synthetic then **the exact existing P3 `study_admission.rs::encode_request` body**: consumption tag; physical_lanes Option<U64>; transpose Dedicated0 or Reuse1+List<I32>; eval three Bools hoist,bsgs,lazy_relin; limits max_bytes,max_records U64; work limits geometries,initial_states,updates,handoffs,wrapper_calls U64. I32 is four-byte LE two's complement as in P3. Do not reserialize Debug, reset limits or confuse requested work with actual counters. Request SHA256 hashes the complete new wrapper, derived from the retained object.

`Consumption` uses new consumption header/synthetic then `consumption:tag,seed:U64,request_sha256:Hash,trainer_sha256:Hash,selected_receipt_sha256:Hash`, followed by exactly one closed branch. Full0: `total_updates:U64` (the whole retained full registered plan; checked against the canonical selected receipt and P3 trainer). Quick1: `updates:List<QuickUpdate>` count exactly2; seed exactly0. The full branch avoids copying the entire potentially large receipt/order; its verified receipt+trainer/request and checked total bind every full update. Quick stores the actual first two ordered sample/partition records. No digest caller can choose these facts. The complete wrapper digest is the consumption identity.

## 3. Common prefix and mandatory branch bodies

After evidence header/synthetic, encode in this exact order:

```text
policy_revision:Text128, scope:tag Full0/Quick1,
execution:Bytes [complete frozen case-binding canonical bytes],
contract_sha256:Hash, selected_receipt_sha256:Hash, materialized_sha256:Hash,
campaign_id:Text128, profile_id:Text128,
approval_sha256:Hash, train_receipt_sha256:Hash, executable_sha256:Hash,
source_manifest_sha256:Hash, source_revision:Git40
```

This preserves the FIX1 ordered prefix. The ExecutionBinding digest is derived from the Bytes and must equal the actual retained owner projection. Source identity is exactly the existing complete approved Train manifest/revision; no second implementation manifest. Approval/receipt/executable are current checked context identities. No PID, ProcessStamp, context-instance number, pointer, fd, runtime-owner token or lease is serialized. Applicability to that live context is a private constructor/recheck obligation.

Full0 body, all mandatory: `case:Bytes [complete frozen registered-case canonical],case_id:Hash,lock:Link,reference:Link,lock_id:Hash,threshold_bits:U64,attempt_id:Text128,predecessor:Option<Predecessor>`. `case_id=SHA256(case)`, source canonical full scope0, exact profile/registry/seed and execution axes. Lock/reference/threshold/ID are extracted only from mandatory VerifiedTargetForSeed, with the independent selected receipt retained. Full consumption required. Case bytes/header are decoded/verified with frozen existing routines, not accepted opaque on hash equality. Optional predecessor encodes only already-ratified Initial/Retry; absence cannot erase a named orphan attempt.

Quick1 body, all mandatory: `registration:Bytes [frozen registered-case canonical],registration_id:Hash,operational_identity:Bytes,operational_id:Hash,seed:U64=0,consumption:tag1,consumed_plan_sha256:Hash,request_sha256:Hash,operational_attempt:Text128`. Registration uses full scope0, seed0 **as provenance only**, exact registered tuple; no admitted-full assertion. operational_id hashes complete identity bytes; every duplicated field must equal decoded identity/current common facts. No target/reference/lock/threshold/predecessor key, option or null is permitted. Unrecognized trailing Full bytes reject.

Quick identity body after its header/synthetic, in order: `registration:Bytes,registration_id:Hash,execution:Bytes,execution_sha256:Hash,contract_sha256:Hash,selected_receipt_sha256:Hash,materialized_sha256:Hash,campaign_id:Text128,profile_id:Text128,seed:U64=0,consumption:tag1,consumed_plan:Bytes [Consumption],consumed_plan_sha256:Hash,request:Bytes [Request],request_sha256:Hash,trainer_sha256:Hash,operational_attempt:Text128`. The consumed plan links the same request/receipt/trainer; same canonical input first two updates, never a hand-selected subset. Identity contains no evidence/event hashes. Dependency order is source/input/descriptors→request/consumption→identity→five evidence→preflight→start→end; no cycle.

## 4. Five typed tails

Tail follows the mandatory branch body directly. No optional generic metadata or wildcard JSON is present. Lists with required members may not be empty, incomplete or silently narrowed to fit A.

### Inputs

`source_files:List<FileRef>,contract:IdentityRef,selected_receipt:IdentityRef,materialized_sha256:Hash,consumed_plan:Bytes [Consumption],consumed_plan_sha256:Hash`, then Full only `independent_target_receipt:IdentityRef,independent_materialized_sha256:Hash`. Quick ends without these two fields. Source files exactly equal the canonical source manifest's selected required records, ordered by `(root tag,path UTF8 bytes)`, unique. Sources include every actual train/test/order/initial-state file required by the canonical profile/seed, with whole-file hashes/lengths. Contract/selected identities equal common prefix; independent Full identities are derived separately by the target reader and checked equal in schema/encoding/length/hash/materialized fields. No encoded `equal=true`. Quick performs canonical-source/order checks directly.

### Implementation

`applicability:Applicability,correctness:List<Correctness>,security:SecurityEvidence,construction_obligations:List<tag>`.

`Correctness = kind:tag KAT0/NI1,report:FileRef,command:FileRef,source_manifest:FileRef,source_revision:Git40,criterion_policy:FileRef,applicability:Applicability`. Exactly KAT0 for I; KAT0 then NI1 for NI. These records identify genuine accepted source-specific prerequisites, not this synthetic package or this run's future outcome. Same exact applicability required unless the existing approved criterion explicitly establishes a checked implication; that implication must be parsed from the retained policy/report, not a caller-provided acceptance flag. Missing implication means missing applicability.

`SecurityEvidence = policy:FileRef,parameter_export:FileRef,estimator_revision:Git40,estimator_sources:List<FileRef>,environment:FileRef,command:FileRef,model_config:FileRef,instances:List<SecurityInstance>`.

`SecurityInstance = role:tag main0/ephemeral1,problem:FileRef,bdgl16_raw:FileRef,matzov_raw:FileRef`. Exact list main0 for I, main0+ephemeral1 for NI; retain all raw artifacts/config needed by the accepted original analysis (an exact multi-file report can be named by a retained manifest FileRef, whose parser validates complete listed files under the same custody). Estimator source list sorted unique. `problem` retains the exact LWE instance derivation, modulus bound, distribution, error, samples and key-role correspondence; the reader checks these against full retained prime/parameter/secret/key descriptors. The existing policy chooses BDGL16/MATZOV requirements and records Core-SVP/optimizer limitations. No invented security threshold, rounded log export sufficiency or document-number gate. Additional needed attack/optimizer files belong in the checked report manifest, never omitted. Unknown report/config schema or coverage returns missing evidence.

Construction obligations are exactly tags0 factory parameter equality,1 actual secret contract,2 actual key ownership/roles,3 encoding domain; add4 EvalMod domain for NI. They describe mandatory later checks, never observations that random keys/encrypted values already exist. Actual construction checks occur after durable start and before the affected operation; approximation/NI/accuracy/performance outcomes remain future measurements.

### Layout

`parameters:Hash,bootstrap:Option<Hash>,layout:Hash,trainer:Hash,keys:Hash,consumed_plan:Bytes [Consumption],consumed_plan_sha256:Hash,request:Bytes [Request],request_sha256:Hash`.

All digests are the same retained descriptor bytes, exact actual checked geometry/stages/levels/scale/optimizer/partition/key policy and P3 request; verify every owner/plan correspondence, source-defined invariants and original work budgets. No layout digest proves encrypted precision. Full total order and Quick two-update order obey §2, and the runtime uses these same objects.

### Resource

`applicability:Applicability,physical:Hash,model:FileRef,model_validation:FileRef,requested:Requested,requirements:Requirements,observation:Observation,claim:Claim`.

`Requested = max_bytes,max_records,plan_bytes,retained_bytes,live_temporary_bytes,peak_temporary_bytes` all U64, actual PreparationAccounting projection with its original limits; requested payload is not RSS. `Requirements = components:List<Component>,peak_host_bytes:U64,incremental_host_bytes:U64,required_disk_bytes:U64`; exactly22 components ordered tag0..21. `Component = kind:tag,phase_mask:tag,resident_bytes:U64,additional_bytes:U64`; phases bits1 prepare,2 key/setup,4 update,8 endpoint, masks1..15. resident_bytes is the portion already charged in the same capacity observations and retained by actual current owners; additional_bytes is the conservative further peak charge during that phase from the retained source-specific model. Inapplicable component has mask0/both byte fields0, permitted only when model/source proves inapplicable; unknown is not zero. Component tags:0 main generator,1 main secret/public,2 training relin,3 training rotations,4 separate bootstrap bundle,5 Encoder tables,6 client/decryptor,7 compact state,8 expanded state,9 NAG branches/old state,10 micro packets,11 interstage handoffs,12 bootstrap workspace,13 worker workspace,14 source/materialized input,15 target/reference/input receipt buffers,16 plan/preparation metadata,17 diagnostics,18 evidence/event/publication read/write buffers,19 allocator/alignment overhead,20 runtime/thread stacks,21 other source-owned allocations identified exhaustively by the reviewed model. Component21 is a modeled residual with exact allocation inventory in `model`, never guessed headroom or an unknown catchall. Every live allocation belongs exactly once to the model; lifetime masks conservatively cover overlaps. peak_host_bytes equals max over4 phase sums of resident+additional; incremental_host_bytes equals max over4 additional-only phase sums (checked additions). Compare incremental_host_bytes against current MemAvailable/cgroup headroom, whose counters already reflect resident allocations; do not subtract total peak again. The model must justify the resident portion against the same actual capacity charge snapshot, separately for applicable host/cgroup accounting, or conservatively retain uncertain future charge in additional_bytes without inventing a credit. Resident ownership/accounting is not by itself observed RSS. No negative future release credit is assumed. These arithmetic identities cannot validate completeness or residency of the model.

`Observation = physical:Hash,clock:Clock,host_mem_available:U64,cgroups:List<Cgroup>,disk_available:U64,host_raw:FileRef,disk_raw:FileRef`; CPU only in this version; CUDA/external resource evidence is missing until separately reviewed concrete support, never a CPU estimate relabeled GPU. `Cgroup = version:tag v1=0/v2=1,path:Text4096,limit:Option<U64>,current:U64,limit_raw:FileRef,current_raw:FileRef`. List is exact applicable finite/unlimited hierarchy from leaf to root; None means observed kernel unlimited, not missing; empty only when actual observer proves no applicable memory controller. Available host capacity is min(host MemAvailable, each finite saturating(limit-current)); policy additionally accounts for existing cooperative claims/other reserved work without counting its own requirement twice. Observation contains no claimed RSS/VRAM measurement. Disk needed includes actual retained/source/output/log requirements separately from transport caps. Unsupported observer or missing cgroup/host evidence is missing, never pass.

`Claim = mechanism:tag0,claim_id:Text128,claim_record:FileRef,overlap:tag serial0/explicit_overlap1,other_claimed_host_bytes:U64,other_claimed_disk_bytes:U64`. Exact current cooperative claim record and observed competing reservations; no PID or lock handle in wire, no claim of physical reservation. Live origin-bound lease and fresh OS/claim checks remain required at admission/start/begin/pre-allocation and runtime boundaries. No numerical safety multiplier added. Need/free comparison reports both validated total physical peak and incremental requirement, comparing the incremental requirement with currently available capacity under existing policy; missing model terms or custody reject before allocation.

### User

`authority_bytes:Bytes [UserScope],authority_sha256:Hash,current_bytes:Bytes [UserCurrent],current_sha256:Hash,authority_file:FileRef,current_file:FileRef,selected_scope_index:U64,observed:Clock`.

UserScope header/synthetic body: `origin:tag conversation0,conversation_id:Text128,message_id:Text128,instruction_bytes:Bytes,scope_generation:U64,scopes:List<AllowedScope>,holds:List<Hold>`; exact already-granted instruction bytes, not a newly requested signature. `AllowedScope = branch:tag, campaign:Text128,profile:Text128,registration_id:Hash,attempt:Text128,retry:tag Initial0/Retry1`, followed only for Retry1 by `predecessor:Predecessor`, then `overlap:tag serial0/explicit_overlap1`. Quick disallows Retry1. Each allowed tuple is concrete; no wildcard/broader-axis interpretation. `Hold = branch:tag,registration_id:Hash,reason:tag full_encrypted_block0/explicit_user_hold1`. Root records all applicable current holds from the actual instruction; selected tuple cannot match a hold, and full encrypted block classification independently enforces the unchanged global hold even if a malformed record omits it.

UserCurrent header/synthetic body: `conversation_id:Text128,source_head_message_id:Text128,scope_generation:U64,authority_sha256:Hash,revoked_scopes:List<U64>,revoked_all:Bool,observed:Clock`. Indices unique ascending and within scope list; selected scope not revoked, revoked_all false. This is a captured administrative record of actual source currentness, not a revocation service. Hashes identify exact retained bytes, not a caller's assertion of authority. The minimal concrete source-currentness mechanism is the root-controlled active `authority.current.bin` entry: root is its sole trusted updater, records the latest actually received task instruction/holds/revocations, and advances scope_generation on a scope change. At each live checkpoint the private loader opens that exact current entry afresh through the retained trusted authority directory, verifies its current bytes/identity and generation/source-head against the selected authority, and rejects revocation or any changed generation; it never keeps using only the immutable captured sidecar. A root update may replace the current entry atomically, so recheck directory-to-entry identity around each read and refuse a race; do not insist that the live current-entry inode stay equal to the initial snapshot. Root-origin control/current task provenance must already be established by the trusted launcher, never an untrusted CLI path or self-authored record. No conversational API, signing service or new user-confirmation flow is needed. If that concrete root update/check channel is not installed, return missing-user-evidence; rereading a caller-authored or knowingly stale file is insufficient. Do not request a new confirmation for work already authorized. No automatic external send, spending, host mutation or full-block release follows from this format.

## 5. Evidence-reader custody and fixed policy selection

Five exact production revision strings are `d7-inputs-v1`, `d7-implementation-v1`, `d7-layout-v1`, `d7-resource-v1`, `d7-user-v1`. Private fixtures use `test-d7-<name>-v1`. Closed per-gate compiled dispatch selects exactly this revision; unknown revision is unsupported/missing, never fallback or caller-selected implementation. These administrative schema revisions do not revise scientific policy. Each reader uses the actual approved current policy/context and existing criterion policy files (including original security criteria), not independently configured thresholds. There is no sixth pinned design document or alternate approval/source manifest.

The private producer receives already-held concrete inputs from the actual context/owner: canonical input files; approved source manifest/revision; approved current policy files; reviewed KAT/NI/security/model evidence; root-recorded existing task authority; actual observer/claim evidence. FileRef root/path is a descriptive identity checked against the producer's expected inventory, never used to open arbitrary paths supplied by an artifact. Readers select fixed expected roles from that inventory, open beneath retained safe root/ancestor FDs using nofollow, verify regular-file/owner/mode/mount/size and exact whole-file bytes/hash, retain File/read stamp/immutable bytes, recheck identity after bounded read, and derive decoded applicability only from known reviewed report formats. External evidence is bounded by existing 8MiB provenance read limit per raw file unless an existing stricter typed limit applies; existing contract/receipt readers keep C/R. Reports exceeding a current limit remain missing pending design refinement. Streaming raw hashing does not license unbounded decoding. No generic externally extensible evidence schema/service.

The root-owned authority inputs use closed loader names `authority.scope.bin` and `authority.current.bin` at the already selected trusted authority root; claim inputs use the concrete resource mechanism's selected record. These are retained source inputs, not new files inside the attempt tree. Their current source origin remains a live check. The constructor takes no caller-supplied FileRef digest, pass bit, source-current flag or evidence-root path. Exact held reader custody transfers into gate owners; immutable evidence snapshots transfer to the live frame while the actual resource/user/context owners remain live. Replacing a file with equal bytes does not satisfy required held identity. A report's source manifest may differ from current source only when a reviewed retained applicability criterion proves the exact relevant source equivalence; no silent commit wildcard.

Five evidence sidecars follow the already-ratified custody proposal verbatim: guarded eligibility before any attempt mutation; fixed order inputs→implementation→layout→resource→user; no-clobber sync; held final-file descriptors, exact byte/hash equality and identity rechecks; durable admission/preflight only after all five finals. Unknown entries/partial attempts/recovery cannot yield live capabilities. Publication caps/path/temporary IDs/Initial/Retry, Full start bytes and `begin_checked_he` lifetime signature are unchanged. Source implementation of this reader, accepted evidence applicability and physical model completeness remain required; this specification establishes none of them. Existing security and correctness evidence maps locate retained raw files and historical measurements with different coverage; absent prospective readers do not mean those measurements never existed. Neither map passes this reader by itself. Historical `hekit-*` to `ckks-train-*` source renames must be mapped before substantive applicability comparison; path-only diffs cannot establish changed algorithms.

## 6. Quick event canonical bodies

Header/synthetic per §1, followed by exact table order:

| Event | Ordered body |
|---|---|
| Preflight | `identity:Bytes,identity_sha256:Hash,gates:Gates,argv:List<Text4096>,source_manifest_sha256:Hash,source_revision:Git40,approval_sha256:Hash,train_receipt_sha256:Hash,executable_sha256:Hash,observed:Clock` |
| Start | `identity_sha256:Hash,gates:Gates,preflight:Link,started:Clock` |
| End | `identity_sha256:Hash,gates:Gates,preflight:Link,start:Link,outcome:tag complete0/failed1,completed_updates:U64,failure:Option<Failure>,ended:Clock` |

`Failure = phase:tag setup0/update1/endpoint2/live_recheck3,code:tag input0/implementation1/layout2/resource3/authority4/allocation5/execution6`. These are closed operational categories with no arbitrary error log inside the event. Extra diagnostics use separately authorized namespace. Complete requires completed_updates2/failure None; failed requires failure Some and completed_updates0..2 exactly actual completed updates (endpoint failure may follow2). A setup failure after durable start may report0. No end without valid preflight/start; only actual same-origin retained continuation may write it. A crash or lost continuation leaves no end; recovery never invents one. Neither complete nor start proves a scientific result, full prospective threshold or all268-row Quick pass.

All five gate revisions/hashes in all three events match the same exact five retained Quick sidecars. Identity common facts and request/consumption must match those sidecars; argv is the actual checked capture argv, not a second requested command. Preflight has no event links; start links exact preflight file/payload; end links exact preflight and start. Clocks captured from same operation origin, nondecreasing elapsed and existing lifecycle wall requirements; static canonical parsing is not proof of clock capture or current leases. End written by retained Quick frame without flock during HE. K/A checked before any namespace mutation; oversized planned records remain unavailable, not truncated. Quick is Initial only in its separate tree.

## 7. Why these remaining routine choices

Compact IdentityRef avoids embedding a31MB receipt into A; duplicating full receipt is rejected. Full consumption binds complete existing receipt/trainer instead of duplicating every index; Quick explicitly retains exactly two actual updates so its projection is inspectable. A request-domain wrapper gives the existing P3 request body an unambiguous independently hashable identity; hashing debug/argv or the whole trainer as a substitute loses original limits. Closed five revision strings avoid arbitrary policy selection. FileRef is a retained identity, not a file-opening instruction or new evidence registry. A fixed typed resource breakdown preserves missing-term detection and separates payload/physical/observation; guessed multipliers, union-key cardinalities and arbitrary pass flags were rejected. Source-current authority snapshots record existing scope without inventing signatures/approvals; static text hashes cannot establish ongoing authority. Compact Quick canonical hex JSON permits exact whole-file and payload links with a calculable K budget and no new parser family; directly copying Full JSON would introduce target placeholders and change meaning. Fixed event outcome tags avoid arbitrary claimed scientific metrics. All are representation choices; unresolved genuine applicability/resource/currentness mechanisms remain explicit missing dependencies. No architectural conflict with the ratified five-gate/custody design is asserted.

## 8. Eight named branch fixtures and exact call inventory

These are frozen future test plans, not executed runtime tests. `S` means private scope fixture harness; `V` pure syntax/link validator; `G` the five-gate assembler spy; `W` attempt namespace mutation spy. The synthetic harness cannot call genuine capability constructors or HE. Every row has actual parameter/P1/planner/materializer/observer/registry/native/Cargo/HE calls0. Production rejection test invokes production **parser** only, not production capability construction. `positive` means synthetic representation/control acceptance only.

| Exact test name | Fixtures and calls | Expected |
|---|---|---|
| `full_missing_verified_target_rejected` | Full inputs base with mandatory verified-target owner omitted; S1,G0,W0 | missing before scope/namespace |
| `quick_without_target_canonical_inputs` | Quick identity + five Quick evidence + all3 events; V9,S1,G1,W0 | synthetic target-free control accepts; five real constructors0 |
| `quick_bundle_cannot_enter_full_admission` | all-Quick bundle and five mixtures (one Quick at each Full slot); S6,G6,W0 | all6 refuse Full; compile-fail live type conversion separately1 compile case, not run here |
| `full_target_threshold_placeholders_rejected` | Full inputs: missing owner, zero fabricated lock, zero fabricated reference, absent threshold, zero threshold without origin; S5,G0,W0 | all5 reject; a truly verified zero numeric target is separately required positive owner test, not forged here |
| `equal_registration_input_no_scope_promotion` | Full/Quick pair with equal seed0 registration/contract/receipt/materialized; S2,G2,W0 | correct branch accepted once each; cross-scope consume V2 rejects |
| `quick_exact_consumption_bound` | seed1, count1, count3, swap first two indices, partition gap, Full tag, other attempt, altered request limit; V8,S8,G0,W0 | all8 reject |
| `full_independent_target_receipt_mismatch_rejected` | independent receipt hash changed; independent materialized hash changed; lock link changed; threshold one-bit changed; S4,V4,G0,W0 | all4 reject despite common inputs |
| `full_frozen_bytes_and_branch_evidence_distinct` | read-only original Full golden manifest; all10 new evidence; V10; original file hash comparisons once/file,W0 | exact old files unchanged; new mandatory branch bytes differ |

Additional mandatory source tests: every closed tag wrong; header test↔production or synthetic false/true mismatch; domain↔tail mismatch; field truncation/trailing bytes; inconsistent IdentityRef length; digest versus retained file substitution; every gate policy/hash permutation; start/preflight/end link mismatch; argv/source/context mismatch; request limit/consumption/order/partition mutation; absent correctness/security/model/currentness; partial resource term/overflow/stale capacity/claim; revoked hold/overlap/retry scope; whole-file A+1 and Quick K+1; raw JSON whitespace A+1 and escaped hex. Cap boundary checks use length arithmetic/simulated bounded readers, no huge allocations by default. The vector directory includes concrete offset/value mutations and exact expected semantic refusal reasons; these are specifications until a separate reviewed production decoder exists.

## 9. Oracle generation and remaining work

`task-54-d7-gate-schema-vectors/` contains standalone stdlib `generate.py`, literal leaf inputs (including nested-record inputs), complete `.bin`/`.hex` and Quick `.json` files, offsets/lengths/hashes manifest, mutation specifications and README. No project encoder imported and no Rust/HE/source test run. All synthetic oracle data has explicit test domains/labels. Original Full golden files are read solely for hash preservation, never regenerated or edited. `generation-report.json` records exact command/counts/hashes and arithmetic. The generator's production-encoding representation is not a gate implementation.

Remaining before production: independent SPEC/QUALITY design review/ratification; closed source readers/decoders and actual applicability criteria; genuine accepted KAT/NI/security evidence; complete validated physical liveness model and real observations/cooperative leases; root-origin authority currentness loader; custody/publication and consuming lifetime source/tests; actual runtime lineage and scientific results. Retain21 profiles, full2680+20+7 and quick268, all existing holds/baselines/security/Task5 and license boundaries. No source/native/runtime availability, resource-fit or scientific success claim follows from these bytes.

## 10. Frozen generation/verification report

Executed only standalone Python stdlib: generator twice (initial draft exit0, revised final exit0); independent literal verifier once, exit0. Final verifier checked25 records /1,134 primitive spans,129 generation-manifest files,133 checksum entries, and three exact Quick event link sets. All78 old Full golden files matched the preexisting frozen hashes. There are135 files in this new vector directory, including the post-generation verifier report. Final output has14 core vectors,21 concrete mutation files and16 additional mutation specifications. Semantic negative rejection remains unexecuted; no production parser, live constructor, parameter/P1/planner/materializer/registry, Cargo/Rust/native observer/SSH/HE or scientific output was invoked. The script report counts one generator invocation per run; the actual total for this task is two.

Commands (cwd `/Users/marconi/Documents/research/hekit`), both exit0:

```sh
python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-gate-schema-vectors/generate.py
python3 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-gate-schema-vectors/verify_literals.py
```

| Core canonical file | Bytes | SHA256 (binary whole file = canonical payload) |
|---|---:|---|
| `quick.identity.bin` | 2311 | `b872e944869801d2d72aaaee5b2e84c46d4dd13f3aa4186503bbc8a7b6eb89ca` |
| `full.inputs.evidence.bin` | 3486 | `5286b1a7dd10f97f38780797a6518c8f2646968a7514234ffd476b5af448b414` |
| `full.implementation.evidence.bin` | 4962 | `c4845f24991ef111d57336a8e643f99e92c95422a2c73e3d360bc992883d6e60` |
| `full.layout.evidence.bin` | 3040 | `fbb08ae16b1a253a37eaae5f3a8755ea6dc40d6e3b1a10e2c9a1639d4dab17eb` |
| `full.resource.evidence.bin` | 4265 | `988156ca6ec4dfbe2519f6ef3913299687b82df1efa49b8c04582129cfd4e3dd` |
| `full.user.evidence.bin` | 3252 | `80937a3e8f88b574de45e8efc29f29731aa6fc003fc042ea753788567e886692` |
| `quick.inputs.evidence.bin` | 5571 | `d0b8058dc8053c7881374bf23c4bc37bf5c0ec2f1a875c57d241a54c785efff1` |
| `quick.implementation.evidence.bin` | 7138 | `c1308dd1d5dc4b880f4e9692de7d1593e30b7a5dc65df9600a6e46aabadf36d2` |
| `quick.layout.evidence.bin` | 5359 | `fe350f6a0a48be25985abf5f581fbe389a625ae595234217a51ff897bcf896ad` |
| `quick.resource.evidence.bin` | 6442 | `704cab6bf5f3776931e7a3763c330ecb53cb50789022016a5c7403da200a56c6` |
| `quick.user.evidence.bin` | 5429 | `1ccc0d7eed4935e0df3088392dbda1fbfe93c484b9769b767510a4708e13a492` |
| `quick.preflight.bin` | 3369 | `880392197bad78908a27fbe5e76bf6adbbef5f992f5e5c022886c6e26cd8e4d7` |
| `quick.start.bin` | 802 | `b76fe8982c5596cfed6308ebd0344123126ec31df6b7af0a874138deac1be3ed` |
| `quick.end.bin` | 954 | `6169fa0135a27b778dd21500c5d1058f471ff9356d6ad58321741ff377e8eb14` |

| Quick JSON | Raw bytes | Exact raw-file SHA256 |
|---|---:|---|
| `quick.preflight.json` | 7005 | `6c81186a0bd46ee75da3d7d6818965e05888ac47d97848fa6524ccd5625c54a9` |
| `quick.start.json` | 1863 | `5f6ac7dad2c858cdfbeb9d7215ad83a74855b91b65c30ef4fadd0ef55b151663` |
| `quick.end.json` | 2163 | `f45b89c689f01af22703f4afb911cbe2ab46a383c6a91c00322d101f08536473` |

| Audit file | SHA256 |
|---|---|
| `generate.py` | `7a59795c065c953f286ae2e61844eb304a8adb92521585fc05cb6a426d0bc198` |
| `manifest.json` | `8dd99cc4f068b2f3209bd497a06cc31a772227a5f49d6ae1111bfb7d30ff88bb` |
| `literal-inputs.json` | `a68a295df7c948923d98064a114e514baa1ef9ed9ac8aff02d6b01496c198ff5` |
| `generation-report.json` | `ab2ca38c511b5343507ef3d78bf0752bfecf2627ebcb314f7fa5cc7bd71e25a6` |
| `verification-report.json` | `8c635ed389ad734cd6d475b628a149375b4bbe866105816c4524f9eb34608336` |
| `SHA256SUMS` | `13f0017cfa425b079d8994a64296da632440994a62bf3367055bf5fe7c4fe00a` |

Root feedback incorporated before final bytes: resource terms explicitly separate already-charged resident owners from incremental allocations and compare only incremental need with capacity headroom; approved source-manifest cap512 remains; authority currentness uses the minimal root-controlled fresh current-record check, with source-update dependency explicit and no new confirmation/service. Historical correctness/security evidence remains available with distinct applicability; crate rename mapping is required before source comparison.
