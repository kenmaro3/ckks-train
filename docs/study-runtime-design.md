# StudyTrain role, build and runtime contract

2026-11-07. **RATIFIED DESIGN; implementation and execution gates remain pending.** This administrative provenance contract complements [study-target-design.md](study-target-design.md). It does not change scientific policy, artifact schemas, parser caps, the existing five context-pinned design paths, or registered experiment coverage.

Root selected the three closed Train recipes, unchanged capsule/receipt grammar with approval-v2, distinct current/artifact role checks, explicit read/mutation matrix and retained-executor lifecycle after independent design review. Sections 1–10 below preserve the reviewed proposal text exactly; proposed API names describe required future source, not already available APIs. Reasons, rejected alternatives and costs are recorded in sections 1 and 10. No source package, native runtime, production reference/admission, HE allocation or full block campaign is released by this publication.

## 1. Recommended minimum coherent extension

1. Keep compile-capsule-v1 and build-receipt-v1 **wire grammar unchanged**. Classify their role through an exact closed recipe/algorithm policy, checked together with fixed Cargo target/source/executable/features. Existing Reference policy remains byte-for-byte compatible and Reference-only. Add three fixed StudyTrain recipes: CPU serial, CPU owned Rayon pool, CUDA with owned host Rayon pool and HostBootstrapper/HostRefresher. No arbitrary Cargo/target/feature escape.
2. Add **local-approval-v2** with per-build explicit role/recipe/algorithm/triple. Preserve local-approval-v1 as Reference-only. Both roles in a prospective campaign must use the same exact v2 approval pin and approved source; artifact writer and current executable remain independently approved identities.
3. Share a private verified core. The Reference loader selects Reference current role; the Train loader selects StudyTrain. Artifact-context borrowing permits verified reads but does not permit Reference capture, new lock publication, reference publication or their recovery mutations from Train.
4. Retain a real runtime owner. Serial execution reports1 because the checked serial code path and its owning OS thread provide that fact. Parallel execution reports the queried worker count of the exact retained pool that executes the case. CUDA reports that same host-pool count, with device/stream/compiler facts bound separately; it never reports GPU lanes as CPU threads.
5. Start the entire parallel case lifecycle inside one private concrete `ThreadPool::install` dispatcher. Construct/load contexts, materialize inputs, read target, admit, start, construct/use/drop HeSession inside that closure; return only a non-authoritative outcome. This respects non-Send live tokens and avoids passing them through a Send closure.

The resulting build supports only the explicitly checked backend/runtime form for its recipe. Missing algorithms/adapters/input/gates remain states attached to the complete registration inventory; a successful build is not implementation coverage, admission or a run.

## 2. Findings from actual local source

* `study_provenance/codec.rs` already separates structural decode/canonical re-encode from `Capsule::policy` and `Receipt::policy`. The grammar stores recipe, algorithm and feature strings; policy currently fixes Reference recipe/session, x86_64 Linux release opt3 debug0, empty features. Thus new role semantics can use the existing grammar without changing Reference bytes or adding an optional field.
* Approval-v1 has a single recipe/algorithm followed by a list of hash triples. It cannot honestly enumerate multiple recipe roles while preserving its current meaning. A distinct v2 approval domain is necessary; silent reinterpretation of v1 is rejected.
* `context.rs::validate_linkage` hard-codes Reference build argv, emitted target/source and executable basename; `build.rs` rejects every CARGO_FEATURE_* on a production request; both Python helper policy and runtime snapshot are Reference-only. All of these are owned dependencies, not just the Rust enum.
* `ckks-train-ckks/src/par.rs` uses current-pool Rayon parallel iterators/join when `parallel` is compiled and direct sequential iterators/calls otherwise. Bootstrapper `evaluate_raised` uses that same `par::join` for the two EvalMods. Neither helper creates a pool or switches pools.
* A search of Rust sources under crates found no production `build_global` or RAYON_NUM_THREADS setter. Dedicated ThreadPoolBuilder calls are in benchmarks and a bootstrap test. Historical mnist/ablation/provenance output merely prints the environment value and logical CPUs; these strings are not measured pool counts and must not be reused as case evidence. This is a static source finding, not proof of a process's runtime state.
* `HostBootstrapper::bootstrap` synchronously downloads, calls the borrowed CPU Bootstrapper with the distinct host evaluator, uploads, and merges counts only after successful upload. `HostRefresher::refresh` likewise calls its CPU refresher synchronously. Neither escapes onto a newly created host thread. Therefore the CPU bootstrap's nested par calls use the installed host pool when the outer case runs there. Preserve the separate host evaluator/fork and no concurrent use requirement; pool parallelism within an operation is not permission to run multiple case bootstraps against that counter concurrently.
* Cargo.lock pins Rayon1.12.0 / rayon-core1.13.0 and cudarc0.19.0. Locally read rayon-core `ThreadPool::install` requires Send closure and Send return; `current_thread_index` on a particular pool returns None for a different pool. Its documented nested join/scope/parallel iteration runs in that installed pool. No `use_current_thread`, global-pool initialization or public install closure is needed.
* `ckks-train-cuda` has no build.rs and uses cudarc dynamic-loading with driver/NVRTC/cuda-13010. Cargo builds Rust bindings without running NVCC or NVRTC. `CudaDevice::new` later creates a CUDA context and compiles four owned embedded `.cu` sources with NVRTC. Actual options from the pinned cudarc CompileOptions order are `--fmad=false`, then `--gpu-architecture=compute_{major}{minor}`. Kernel compilation belongs after durable start, not in a claimed Cargo compile receipt.
* Existing `ReferenceStore::open` opens existing directories. Existing TargetStore::open creates campaign/profile directories and `.profile-guard`, then syncs them. Train's read-only path therefore needs an existing-only target-store open branch; simply exposing today's open via artifact_context would grant those mutations.

## 3. Closed role and recipe table

The role enum is a classification, never authority: `Reference=0`, `StudyTrain=1`. Do not expose a setter or a load(request, arbitrary_role) API.

| Recipe tag for new private dispatch | Exact recipe string | Exact algorithm string | Target/source basename | Exact bench features | Runtime/backend allowed |
|---|---|---|---|---|---|
| Reference (existing) | `study-reference-linux-clean-v1` | `study-reference-session-v1` | `study_reference` / `src/bin/study_reference.rs` | empty | Existing scalar plaintext reference/diagnostic only |
| train_cpu_serial=0 | `study-train-cpu-serial-linux-clean-v1` | `study-train-session-v1` | `study_train` / `src/bin/study_train.rs` | empty | CPU serial only |
| train_cpu_rayon=1 | `study-train-cpu-rayon-linux-clean-v1` | `study-train-session-v1` | same StudyTrain target | `[parallel]` | CPU explicit retained Rayon pool only |
| train_cuda_host=2 | `study-train-cuda-host-linux-clean-v1` | `study-train-session-v1` | same StudyTrain target | `[cuda,parallel]` sorted | CUDA sequential stream with explicit retained host pool; I HostRefresher or NI HostBootstrapper |

All recipes retain native `x86_64-unknown-linux-gnu` host/target, release, opt3, debug0, locked/offline, no default features, explicitly selected direct Rust/Cargo/Python/Git/linker paths, controlled HOME/TMP/CARGO_HOME and fresh external target directory. Compile features are actual checked Cargo facts, not copied from the request. Algorithm string versions the reviewed dispatcher contract; it is not evidence that a missing NN/baseline implementation exists.

CPU serial rejects a parallel/cuda compiled capability. CPU Rayon rejects CUDA backend and requires an explicitly requested positive worker count; CUDA recipe rejects CPU dispatch, native-CUDA bootstrap and external-b. A Rayon count of1 is valid if explicitly requested and actually constructed; it is not a scalar capture copied to parallel code. No numerical default/count cap/performance gate is introduced. Requests must fit existing integer/argv bounds and the platform usize; OS/pool construction failure is an availability failure, not permission to choose another count.

For minimal Cargo ownership, add `dep:rayon` to the existing bench `parallel` feature, which already enables CKKS parallel. This enables bench to own the same locked Rayon pool without a new scientific feature. CUDA already enables parallel and dep:rayon. Verify exact resolved feature sets in metadata and actual build-script feature env; reject unknown bench features. No NN production source change is required. A distinct benchmark's environment defaults are not inherited.

## 4. Exact transport and compatibility

### 4.1 Unchanged capsule and receipt grammar, new closed semantic policy

Encoding primitives/header remain `hekit-study-reference-encoding-v1`, existing LE integers, fixed byte tags, bounded UTF8/raw bytes and lowercase SHA256 strings. Existing bounds remain LIMIT8MiB, CAPSULE_LIMIT2MiB, Text4096 (compiler16384), argv256, feature-list64, package/artifact COUNT65536 and existing manifest limits. No cap is raised.

Capsule header domain remains `hekit-study-compile-capsule-v1`. Body is exactly existing order:

```text
recipe Text, algorithm Text,
source_manifest Hash, config_manifest Hash, dependency_manifest Hash, toolchain_manifest Hash,
environment Hash, cargo_lock Hash, commit Text(valid Git40), compiler_vv Text16384, cargo_version Text,
target Text, host Text, profile Text, opt Text, debug byte0/1,
features List<Text> sorted unique,
package_version Text, cargo_metadata Hash,
packages List<{id Text,manifest Bytes,source_tag workspace0|registry1,source Text}> sorted by id,
metadata_argv List<Bytes>, build_argv List<Bytes>, cwd Bytes
```

Receipt domain remains `hekit-study-build-receipt-v1`; body remains:

```text
capsule Bytes,
four files-manifest Bytes in source0/config1/dependency2/toolchain3 order,
environment Bytes, executable Bytes, executable_hash Hash,
five post Hash values (four manifests then environment), metadata Hash,
emitted List<{id Text,manifest Bytes,name Text,src Bytes,kinds List<Text>,executable Option<Bytes>}>,
helper_argv List<Bytes>
```

Keep the exact current ordering/uniqueness/key definition and canonical re-encode equality. Receipt embeds capsule: there is no independent free role field that can disagree. A private `CheckedBuildRole` is derived only after **all** recipe + algorithm + feature + fixed target/source/argv/output + source/pre-post checks succeed. Unknown combinations reject. Reference's existing `policy()` entrypoint continues rejecting every train recipe; new private train policy functions are distinct. A generic structural decoder returning untrusted claims never establishes a role.

### 4.2 New approval-v2 exact body

Header: Text(encoding), Text(`hekit-study-local-approval-v2`). Body, in this exact order:

```text
id Text4096(nonempty),
designs List<{path Text4096,sha256 Hash}> (exact five existing paths, sorted),
commit Text4096(valid Git40), source_manifest Hash,
builds List<{
  role byte Reference0|StudyTrain1,
  recipe Text4096,
  algorithm Text4096,
  receipt_file_sha256 Hash,
  capsule_sha256 Hash,
  executable_sha256 Hash
}>
```

Reuse existing approval LIMIT8MiB, designs-list32 transport bound with exact-five semantic check, builds-list64 bound. Sort builds strictly by receipt_file_sha256, as v1 triples are sorted; duplicate receipt hash is invalid even across roles. Each entry's role/recipe/algorithm must be one exact table combination. Hashes are exact file/capsule/executable digests, not substitutes for validating the supplied receipt's contents. V2 drops v1's global recipe/algorithm fields because they now belong to each build entry. No guessed role, unknown extension field, empty policy ID, duplicate/trailing bytes or repair path.

All supplied receipts must be approved by exact role/recipe/algorithm/triple, share approval commit/source/root, and cover exact current five document bytes. Multiple approved builds remain allowed; one unique current executable+embedded-capsule candidate must match the loader's fixed role. Additional approved builds need not all be supplied; a missing artifact writer receipt rejects that artifact when needed. No current executable is selected using an artifact path.

### 4.3 Compatibility rules

* Approval-v1 parse/bytes/policy are unchanged: only existing Reference recipe/session and Reference-classified receipts. Train load rejects v1 even if someone appends a train-looking triple and rehashes it.
* `VerifiedStudyContext::load` accepts approved v1 Reference or v2 with current Reference receipt. It can retain v2 train receipts as checked noncurrent entries; that does not authorize train capture. It cannot select StudyTrain as current.
* Train load accepts v2 only and requires current StudyTrain. It can retain approved Reference receipts for reading. Reference artifact receipt lookup requires Reference classification and strict scalar provenance regardless of current role.
* Legacy Reference capsule/receipt bytes can be explicitly listed in v2 **only if** the existing source/commit/root/design checks also match exactly. Format compatibility does not allow an old-source receipt into a new-source approval. No old receipt, artifact or lock is rewritten/relabelled.
* Reference artifact grammar and scalar checks remain unchanged. Old binaries that only know approval-v1 reject v2; that is intentional. New approved binaries may read legacy-format reference artifacts through the same exact science/identity verifier when all current approval requirements hold.
* Lock policy binds exact approval SHA256. A v1 lock cannot be reused under v2 by declaring the policies equivalent. Produce a separately approved new campaign/lock through Reference under the common v2 pin; preserve old bytes. Reference source artifacts may be reused only when the ordinary exact checks genuinely pass. Changed shared input contracts still require the already mandated new campaign/lock.
* Existing scientific constants and five document list do not change. This proposal and study-target-design.md are not silently added as new design pins. Current approved source manifests already bind changed source/build helpers; a role implementation will need fresh build/source approval rather than inheriting historical hashes.

## 5. Build helper and compile facts

Keep existing `study_reference_build.py build` exact command/recipe. Add a narrowly owned `study_train_build.py` with three fixed build subcommands `cpu-serial`, `cpu-rayon`, `cuda-host`. Options are only the existing explicit local tool/root/output paths/jobs/toolchain-file pins; no --bin, --features, --rustflags, --target-triple, compiler wrapper, arbitrary recipe or extra Cargo arguments. Subcommand maps to one table entry internally.

For StudyTrain build argv, use the existing exact vector, replacing only `--bin study_reference` with `--bin study_train`, and inserting `--features parallel` or `--features cuda` immediately after `--no-default-features` for the two nonserial recipes. Serial has no --features. Metadata argv uses the same feature selection qualified as `ckks-train-bench/parallel` or `ckks-train-bench/cuda` for the workspace metadata query, inserted at that same point; use one fixed reviewed ordering. Exact selected bench features and resolved dependency feature facts must agree with the emitted build. Runtime never accepts a recipe based only on the requested features string.

Keep four full pre/post manifests, package identity/checksum checks, fresh target, actual selected `RUSTC -vV`, Cargo version, current source/config/dependency/toolchain/environment hashes and unique emitted target/source/executable checks. Runtime snapshot must dispatch via the already checked closed recipe and use that receipt's exact helper policy. Keep helper sources in the source manifest. Runtime hash-file remains bounded and source checked.

Reuse private common primitive functions. Add a fixed capture dispatcher to the existing build helper: after bounded request/capsule decoding and closed recipe classification, validate the exact actual CARGO_FEATURE_* set against the table, before writing the embedded capsule. For Reference, the allowed set remains empty. `build.rs` must no longer blanket-reject known train features before capture, but it must retain wrapper/override rejection and pass actual compile facts; the fixed capture helper enforces the role-specific feature policy. No request chooses an arbitrary capture script. Add rerun tracking for the new owned train helper and any shared role code.

The controlled build environment remains the existing exact18 keys. CUDA toolkit/version override variables (`CUDARC_CUDA_VERSION`, CUDA_HOME/PATH/ROOT/TOOLKIT_ROOT_DIR/CUDNN_LIB) are not propagated, so cudarc selects the checked cuda-13010 feature, not ambient discovery. Reject attempts to smuggle them through extra arguments/configuration. Snapshot/receipt identifies actual Rust/dependency/build sources; it must not invent an nvcc path, nvcc version, CUDA compilation duration or GPU runtime success.

CUDA builds legitimately succeed on a machine without CUDA runtime libraries because the pinned dependency uses dynamic loading. Such success produces only a Rust build receipt. At run admission the real CUDA runtime/device observation below must exist; missing runtime is explicit unavailable/MissingCode as appropriate, not a fabricated receipt or fallback to CPU. Runtime JIT/compiler facts are not compiler_vv: that remains the actual Cargo-selected Rust compiler.

## 6. Private core, API and complete operation matrix

Proposed ownership shape (names are concrete proposals, not available APIs):

```rust
enum ExecutableRole { Reference, StudyTrain } // closed inspection value, no authority
struct VerifiedContextCore { /* private approval bytes, frozen classified receipts,
 current index/role, root, executable identity, helpers/environment */ }
pub struct VerifiedStudyContext { core: VerifiedContextCore }
pub struct VerifiedStudyTrainContext {
    artifact: VerifiedStudyContext, // SAME owned core, not a reloaded or aliased context
    runtime: CaseRuntimeLease,      // private retained runtime, bound owner thread/process
}
pub struct StudyTrainContextRequest { /* private authority locations + private runtime lease */ }
impl VerifiedStudyContext { pub fn load(request: ContextRequest) -> Result<Self>; }
impl VerifiedStudyTrainContext {
    pub fn load(request: StudyTrainContextRequest) -> Result<Self>;
    pub fn artifact_context(&self) -> &VerifiedStudyContext;
}
```

Reference load selects Reference privately. Train's request is constructed only inside the private concrete dispatcher after runtime construction; paths/count/backend requests are still untrusted requests. Do not provide a public constructor accepting runtime evidence, role, core, pool pointer, observed thread count or caller callback. A public thin frontend may accept a typed `StudyRunRequest`; it does not expose the internal runtime lease or arbitrary execution closure. Train load checks the selected StudyTrain recipe and private runtime owner's backend/form before returning.

Private core methods: `recheck_current()`, `require_current_reference()`, `reference_build_facts(receipt_hash)`, `current_build_facts()`, and checked canonical selected-input validation. The first lookup always demands an approved Reference artifact receipt; the second always returns the current receipt. Do not keep an unqualified build_facts lookup in a caller that can confuse these purposes.

| Operation | Current Reference | Current StudyTrain | Notes |
|---|---|---|---|
| ReferenceStore open existing directories; read_reference/read_reference_snapshot | allow | allow via artifact_context | Shared current recheck plus artifact Reference receipt/scalar verification. No directory creation. |
| Reference captured session, captured reference/diagnostic/failure publication | allow as existing typed rules permit | reject before capture/write | Include F64 diagnostic wrapper and failure publisher; no accidental cross-role writer through common store. Local DiagnosticContext remains unapproved. |
| recover_reference (verify + fsync + PublishedReference) | allow | reject before sync | Recovery is a mutation/publication-state action, not the granted train read. |
| ProfileTargetStore open | existing create/open semantics | existing-only | Train requires every campaign/profile/guard path already present, checked and on approved filesystem; no mkdir/create/chmod/fsync side effect. |
| claim_new / lock_target | allow | reject before claim/mutation | Both source and target writer contexts Reference, same policy; source snapshot ownership unchanged. |
| read_target | allow | allow | Share input/source revalidation privately; do not call Reference capture from Train. |
| recover_lock (verify + resync) | allow | reject before sync | Operator uses approved Reference recovery path; Train cannot repair missing/uncertain lock automatically. |
| admission/start publication and attempt-directory creation | reject | later allow only through real checked gates and consuming protocol | Opening a train store alone grants no attempt creation. E synthetic protocol is not this real constructor. |
| future admission/start inspection/recovery | not added by this package | later separately typed inspection, no live-token recovery | E owns exact implementation; any resync uses the explicit verified-record path and never recreates execution authority. |

Implement a private closed `TargetStore::open_existing` branch sharing all held-handle/mount/guard identity checks. It must not fall back to create if a path is absent. Train later obtains a fresh exclusive guard FD for admission/start without replacing the guard. Creating case/attempt directories belongs inside the guarded checked admission transition, after the prescribed target/case/context/user checks. Do not expose caller-selected open modes publicly.

Factor `read_target` revalidation into an internal `check_materialized_input_now` function: canonical_prepared, derive own selected seed identities, verify every bound source's expected/actual hash and canonical path, match shared contract/selected receipt, recheck current source/executable/approval/receipts. Return a private checked snapshot/identity result, not Reference RuntimeCapture. Before/after read checks retain existing failure behavior. Keep Reference `capture` and `finish_capture` explicitly Reference-only. `approved_provenance_claim` for reference/lock selects reference_build_facts, while its current context independently rechecks whichever approved role is calling. Context pointer equality in later admission uses the exact borrowed artifact object.

## 7. Actual runtime ownership and nested Rayon behavior

### 7.1 Retained host executor

Private runtime variants: `SerialCpu { owner_thread, process_stamp }`, `RayonCpu { pool: Arc<ThreadPool>, owner_thread, process_stamp, observed_worker_count }`, `RayonCudaHost { same pool facts, cuda_observation }`. No public constructors/clone/serde. Internal Arc cloning retains ownership but cannot expose capability duplication. Pool lifetime extends through context and HeSession use; fresh pool per active case initially, no global pool or cross-case learning state reuse. This is a local execution-policy choice, not a new scientific axis or claimed optimal thread count.

Create a dedicated pool with `ThreadPoolBuilder::new().num_threads(requested_positive_count).build()`. Read its actual `current_num_threads`, require it positive and equal to the explicit request, and retain that same pool. Do not call global `rayon::current_num_threads` outside the pool as discovery: it can initialize the global pool. The dispatcher itself must begin outside a Rayon worker and must not invoke a second pool while a case is live.

`pool.install(move || private_run_case(requests, retained_pool))` captures only owned untrusted requests/paths and internal Arc to the constructed pool. **All non-Send context/store/gate/token/session values are created inside and dropped inside.** It returns only a plain owned outcome/diagnostic record, never a context, permit, session or borrowed capability. Do not capture `&VerifiedStudyTrainContext`, CheckedCaseForAdmission or StartedHeCase from outside, even if they appear otherwise immutable. Do not resolve this compile constraint by unsafe Send/Sync or a public callback API.

Inside the installed dispatcher, retain the owner `ThreadId`, exact process stamp, and the pool Arc. At context load, capture_case, recheck_case, admit, start, begin_checked_he and session entrypoints require the same process and owner thread, `pool.current_thread_index().is_some()`, and unchanged queried worker count. The pool-specific index check rejects a different pool even with the same size. Context checks remain on the case owner; internal pure/numerical Rayon tasks need no context token and stay under that installed registry. Serial checks bind the same owner OS thread and a serial-only compiled recipe, yielding execution_threads=1.

The existing CKKS nested parallel maps and EvalMod join stay in this pool. HostBootstrapper is synchronous on the case owner and calls the same CKKS helpers, so its host work stays in the retained pool. HostRefresher's synchronous client path stays on that owner and any approved nested CKKS parallel work uses that pool as well. Preserve sequential GPU dispatch/one stream and the independent host counter evaluator; no cross-case bootstrap concurrency. Prohibit future unbound std::thread/spawn/global-pool work in reviewed case execution until its runtime ownership is separately represented. This restriction does not claim that driver/Rayon/helper process infrastructure has exactly N OS threads.

Meaning of host.execution_threads: **the actual number of workers in the retained executor available to host arithmetic for this case** (serial1 or the checked pool count). It is not the number simultaneously busy at each instruction, the whole process's task count, an NVRTC implementation thread count, or CUDA kernel threads. Optional observed_threads remains a contemporaneous `/proc/self/task` observation with its existing availability tag; available_cpus and seven threading environment values remain separate. Environment values can differ from the explicit pool size and are recorded literally; they do not override it. No RAYON_NUM_THREADS parsing/default/fallback supplies the actual field.

### 7.2 CUDA build facts versus live runtime facts

Add a narrow CUDA-library-owned observation adapter, with bench owning the checked lifetime. Before admission it may load the existing cudarc driver/NVRTC singleton handles, read library versions and physical device identity via driver device-query APIs, and inspect the resolved modules. It must **not** create CudaContext/CudaDevice, allocate keys/CKKS parameters/encoding tables, compile kernels, or probe capacity by allocating. Missing API/library/device/identity is an explicit runtime availability failure. A memory query that requires a CUDA context belongs in the existing reviewed later resource mechanism; this proposal does not invent one or relax the allocation boundary.

Concrete observation sources: cudarc's public `sys::culib()` gives stable library owner and exported function pointers (`cuDriverGetVersion`, `nvrtcVersion`); use the actual pointers with `dladdr` and bounded regular descriptor reads to capture canonical module path/file hash and before/after identity. Do not call dladdr on the Rust wrapper function. Keep the existing rejection of LD_PRELOAD/LD_LIBRARY_PATH/LD_AUDIT and DYLD/Git overrides for both roles. Approved host setup must make libraries discoverable through its ordinary loader configuration; this proposal changes no host loader setup and gives no install authority.

Capture returned driver version, NVRTC major/minor, device ordinal and stable device UUID/PCI identity, compute capability and host identity. Require available successful observations with checked integer conversion, not placeholder0. Retain selected device binding and exact module descriptors/function-owner identity for recheck. Kernel source hashes come from the four owned embedded source byte strings; intended options derive from observed compute capability and the existing explicit fmad=false policy. These are actual loaded runtime/compiler facts and a bound future compilation request; **they are not a claim that NVRTC compilation or GPU execution has occurred before start**.

After durable start, concrete HE construction uses that same device selection and checked modules, verifies the created device's identity, and compiles through those retained cudarc singletons with the exact owned kernel sources/options. Recheck module files/identities and device binding at the consuming boundary and before updates. Creation/JIT failures leave start evidence incomplete and return no HeSession. Any later emitted JIT result may record actual compilation/PTX observations under D7's output ownership; it must not rewrite admission as successful compilation. This role package does not add a PTX evidence schema or call runtime NVRTC the Cargo compiler.

This is the same honest trusted-local-process boundary as existing executable/source receipts: module-path/file hashes plus stable loaded-owner observations are inspectable provenance, not a full dynamic-linker attestation or proof of every transitive NVIDIA/internal compiler module's bytes. Do not claim complete native dependency closure from two hashes. If a future claim requires those stronger facts, that is separately scoped evidence, not a fabricated field in this schema.

### 7.3 Binding and CaseProvenance

CaseProvenance's exact fixed shape/encoding remains untouched: captured-only, schema `hekit-study-case-provenance-v1`, role StudyTrain1, and unchanged seven nested record layouts. `build` comes from current_build_facts; source/executable/argv are current train facts; host.execution_threads comes only from the retained runtime; recheck ties exact current source/build/executable and selected materialization. `capture_case` / `recheck_case` are private and accept the privately checked registered case/runtime binding, never a thread count or provenance map.

D7's existing `ExecutionBinding.runtime_policy_sha256` and `physical_binding_sha256` must cover retained runtime facts rather than hash Debug or CLI strings. For this role proposal, fix the required canonical projection inventory: runtime form/recipe, backend, actual worker count, sequential matvec/stream count, refresh/HostBootstrapper route, actual loaded driver/NVRTC path+hash+versions when CUDA, exact four kernel source identities and intended options, and physical-binding link. Physical binding contains actual host identity and, for CUDA, observed device UUID/PCI/ordinal/compute capability. PID/start ticks/boot stamp stay private authority checks as already ratified; they are not added to CaseProvenance or receipt fields.

Do not silently create an ad hoc encoder here. The already required execution-descriptor owner must freeze exact canonical field/tag order for its full runtime/physical descriptor together with the remaining ExecutionBinding descriptors before hash-producing source; no new scientific artifact schema/cap is introduced by this proposal. Until that encoder and actual D7 gate binding exist, role context/case-provenance codec can be implemented and tested but **no production checked case or admission can be produced**, including CUDA. This is a concrete dependency, not permission to omit the runtime facts or narrow registration. Runtime pool/capture ownership itself is fully specified above and can be implemented without choosing scientific policy.

External-b remains explicit unavailable for admission until a reviewed adapter owns the actual upstream executable/process/input/export provenance and matching physical CPU/thread facts. A StudyTrain wrapper's pool receipt cannot speak for upstream threads or ordering. No GPL implementation is copied. Native CUDA bootstrap remains its separately registered/gated future path.

## 8. Independent vectors and meaningful no-HE acceptance plan

No vector/test was generated or run in this task. Before source, freeze independently assembled canonical bytes and expected hashes from the exact tables above, not the tested encoder. Preserve existing Reference vectors byte-for-byte.

1. Existing Reference capsule/receipt/approval-v1 still decode/policy-check identically; each train recipe in an otherwise coherent v1 approval rejects. New approval-v2 vectors: Reference only; Reference+CPU serial; Reference+CPU Rayon; Reference+CUDA host; multiple legitimate builds. Sort order and all exact nested hashes independently recorded.
2. Train capsule/receipt vectors use unchanged grammar but exact table policies/argv/features. Mutate recipe, algorithm, role, feature set, target/source/output path, metadata feature selection, compiler fact, capsule/receipt/executable triple and pre/post inventory; recomputed hashes do not bypass policy. Unknown recipe/role, duplicate receipt across roles, trailing bytes, field-bound overflow reject. Verify old caps remain unchanged and preflight remains before bounded allocation/hash work.
3. Pure current/artifact matrix: current Train+Reference artifact passes the role checks; Train artifact posing as Reference rejects; Ref current cannot invoke train capture; Train current cannot Reference capture/publish/lock/recover; ref scalar2 and case unavailable/zero threads reject. Ordinary nonrecipe builds never obtain either verified context.
4. Target existing-only open: missing any campaign/profile/guard rejects without filesystem creation/sync; Reference create semantics unchanged. Train target read succeeds in private synthetic policy harness through shared selected-input revalidation, not Reference capture; input/source substitution and distinct context pointer fail. No live approved Mac context is fabricated.
5. Private executor seam: serial reports1; dedicated pools with explicit1 and2 report their queried counts; environment may say99 while owned pool2 remains2. A different same-size pool and execution outside the retained pool reject. Nested join/parallel iterators record membership in that exact pool without HE. Retain only inspection counters, not fake production context constructors.
6. Non-Send compile tests: external creation/move of live tokens into install is rejected; positive compile wiring creates and consumes private synthetic tokens entirely inside the one concrete dispatcher and returns only plain outcome. No public closure can obtain an allocator permit. Fork/process-stamp tests remain separately scoped native synthetic protocol tests from the ratified target design.
7. Host-bootstrap ownership seam: a private synchronous stand-in calls nested current-pool work and verifies owner/pool identity; independent source review ties that seam to HostBootstrapper's actual synchronous path. This is no-HE ownership evidence, not numerical bootstrap validation or CUDA execution. Later assigned small HE checks remain separately labelled.
8. CUDA runtime observation uses private injected sys/loader observations for missing/wrong module/function owner/version/device/substitution negatives; assert no CudaContext/CudaDevice/key/JIT constructor was reached. Real CUDA observation/JIT checks require later authorized Linux hardware execution and do not follow from synthetic tests or a cuda-feature compile.

Keep the complete registration test denominator and all unresolved supplement states beside these tests. Do not reduce CUDA or external expected coverage because their positive physical evidence is not in this local package.

## 9. Source ownership and dependency sequence after ratification

1. Independent design review of this proposal, especially approval-v2 compatibility, existing-only read store mode, pool/non-Send lifecycle, and runtime/compiler evidence boundary. Freeze transport/role vectors separately. Root may continue the already ratified exact-case projection in its bounded scope meanwhile.
2. Provenance/build owner: `study_provenance/codec.rs`, context.rs and module exports; `build.rs`; existing build/support helpers; new fixed train build helper; narrow bench Cargo parallel feature. Same owner or explicitly sequenced shared edits keeps Rust/Python semantics consistent. No current nn source changes.
3. Reader/publication owner coordinated with core owner: Reference artifact role lookup, Reference-only producers/sessions/recovery, target open_existing and read_target factoring; keep lock/source snapshot identities and lifetimes. No admission shortcut.
4. Runtime owner: new bench `study_runtime.rs` private pool/owner/process lease and concrete dispatcher boundary; separately assigned narrow CUDA observation adapter in `ckks-train-cuda` plus immutable device identity accessor. No changes to NN algorithms or HostBootstrapper math required; review its synchronous ownership contract as a consumer.
5. Case provenance owner: distinct codec/private captured-fact adapter using ratified shape and current train facts. Coordinate immutable Reference nested-codec reuse without broadening existing parser/capture. Thin study_train binary only exposes explicit request parsing/dispatch and fails closed for absent D7 work; it is not a complete runner. Thin study_reference remains separately owned D6c4 work.
6. Later D6c3(E)/D7 owners: complete canonical runtime/physical/implementation descriptor encoders, real checked gates/leases, admission/start protocol, concrete HE boundary and checkpoints; integrate available reviewed NN/CUDA/baseline/external executors with full registration state retention. Actual source/build approvals, native production reference/lock/admission/start/HE tests, campaign launches and full encrypted-block release remain later authority/evidence gates.

## 10. Alternatives and costs

| Alternative | Assessment |
|---|---|
| Add role byte/new domains to every capsule and receipt | More duplicate encoders/fixtures and needless Reference migration. Closed recipe classification plus explicit role in approval-v2 already binds role to complete checked build. Reconsider only if future formats actually need new fields. |
| Reinterpret approval-v1 as multi-role | Rejected: changes historical meaning and lets existing global recipe fields lie. V2 is small and old readers fail closed. |
| Separate independent Reference/Train approval pins with policy equivalence | Rejected here: target compatibility already requires exact common approval pin; introduces equivalence/migration machinery. One v2 enumeration preserves that boundary. |
| Global Rayon pool or RAYON_NUM_THREADS-derived count | Rejected: pool may already exist or differ from environment; actual retained executor is unknown. Dedicated pool costs startup/RSS and requires same-size matching, but provides a direct fact and no global mutation. |
| Per-operation install around an already created non-Send session | Rejected: needs moving/capturing non-Send capability across Send closure or unsafe relaxation. Entire lifecycle inside one concrete install is simpler and maintains owner-thread checks. |
| Rayon use_current_thread / custom global TLS authority | Rejected for first package: lifecycle and reentrancy complexity is unnecessary. Private retained owner with pool-specific membership works with current APIs. |
| Call every CUDA execution_threads=1 because dispatch is sequential | Rejected: host bootstrap uses parallel CKKS when compiled; sequential GPU stream does not make host pool scalar. Record real host worker count plus separate stream/device facts. |
| Require NVCC/toolkit execution at Cargo build | Rejected: it does not happen with this backend. Rust receipt and actual later NVRTC facts are separate. Cost: a CUDA build alone remains insufficient for admission/physical evidence. |
| Implement complete dynamic-linker attestation now | Rejected as disproportionate and beyond local provenance claims. Capture exact observed top-level runtime modules/versions and owned source/options, retain honest transitive/native limitations. |
| Expose train recovery/create to make reads convenient | Rejected: read grant should not create a lock namespace or confer Reference publication state. Requires a small existing-only target open path and explicit role guards. |


## Publication record

Reviewed proposal SHA256: `88ea7e46815f2aed72c0a3c039ef142cff50708a3e37d8d99487becb769cb92e`. Independent design review: `b738f973943d9a6bd7ee22e008b9db74f9740e797921c95e5fa4eb5e7675d33c`; no blocking findings. Root read the complete proposal/review and ratifies the bounded contract, preserving all registered rows and unresolved supplemental roles.

Carry review observations into source briefs: qualify every current/artifact role lookup; verify existing-only opens perform no namespace mutation; distinguish valid CUDA ordinal/minor-version zero from unavailable API facts; handle loader missing-symbol/panic paths as errors; record actual retained pool costs/counts; freeze independent transport vectors before encoders. CUDA ordinary-library discovery on quadruped remains unverified; no LD_LIBRARY_PATH workaround, host loader change or installation is authorized. D7 runtime/physical descriptor encoders and real checked gates remain mandatory before any production admission. This document is not silently added as a sixth pinned design.

## 11. Concrete ownership and bounded module reads (2026-11-07)

This records implementation choices before source work; it does not report an implemented runtime or successful native observation. The current source map is retained as `task-54-d6c3-runtime-api-map.md` in the Task54 ledger. Sections6–7 remain the behavioral authority, with the proposed ownership sketch refined as follows.

The single private verified core owns `CurrentRuntime::Reference` or `CurrentRuntime::Train(CaseRuntimeLease)`. `VerifiedStudyTrainContext` owns its one artifact context/core and returns that exact artifact object by reference. The lease has no public constructor, Clone or serialization; a non-Send/non-Sync marker prevents context transfer. The only internally shared owner is the already specified Rayon pool Arc. This avoids both a self-reference from the artifact to an outer lease and a second shared runtime-authority object. It deliberately tightens Reference context Send/Sync behavior too; existing reference loading, publication and artifact bytes remain unchanged. An alternative shared Rc lease would avoid self-reference but introduce unnecessary ownership duplication, so it is not selected.

Implement the narrow CUDA-library observation adapter before the bench runtime consumes its frozen API. The adapter observes actual exported driver/NVRTC function owners, versions and physical device identity without constructing a CUDA context, compiling kernels or allocating HE state. Bench owns module descriptor/hash retention and the checked runtime lifetime. This dependency sequence does not omit the registered CUDA form or allow a CPU fallback to stand in for CUDA evidence. All environment rejection and pre-admission allocation limits in section7 still apply.

For native module files only, select an administrative cap of **1GiB per module** and **1MiB fixed read chunks**. This is a bounded-I/O choice, not an observed library size, scientific gate, or change to existing artifact parser caps. Reject an oversized file before hashing, detect growth/extra bytes, hash the entire accepted file through its retained descriptor, and preserve the existing before/after identity and function-owner checks. A larger actual module is explicitly unavailable until a reasoned revision; never hash a prefix or silently increase the cap.

Use a private streaming-state extraction of the existing original SHA-256 implementation, preserving the public one-shot API and its outputs. Independent known vectors and chunk-boundary expectations must precede the implementation. The reason is to avoid cloning an entire permitted native module into memory. Alternatives are a newly audited permissive streaming-hash dependency or a redesigned external helper protocol; both add broader dependency or execution ownership and are not selected here. D7 still owns exact canonical descriptor encodings; this extraction creates no alternate wire schema, Debug/CLI hash or admission authority.

Actual Train read/write-refusal integration, native Linux process/fork checks, CUDA hardware observations, D7 gates and full runner execution remain required. These choices alone satisfy none of those gates.

## 12. D7 runtime/physical byte contract (2026-11-07)

The inner representation delegated by §7.3 is now fixed in [runtime/physical descriptors](study-runtime-descriptors.md), with four independently checked literal vectors frozen before encoder source. This closes the representation decision only. Actual retained-fact projections, checked route/created-stream binding, canonical encoders, live gates and durable start remain unimplemented dependencies; the five context-pinned design paths and scientific policies are unchanged.
