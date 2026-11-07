# D7 phase2 retained preparation/accounting proposal

BASE `02317621a89197210e1321bbc9f2b01b2533b708`. Read-only source inspection; no constructor, planner, materialization, test, native or network execution. This is a source-package proposal, not implementation/readiness evidence.

## Decision and boundary

Factor canonical validation from `prepare_then_make` / `prepare_blocks_then_make`; retain its actual checked objects, then collect source descriptors. Preserve existing internal compatibility callbacks, synthetic refusal by `descriptor_facts()`, and all P1/P3/P/R bytes. No `PreparedCpuRoute`, runtime-policy producer, ExecutionBinding or started capability is introduced in phase2.

`describe_candidate` currently calls `geometry_identity`, `make_stages`, and `manifest`; these allocate geometry, placeholder states and a `BTreeSet` before returning. `PlanningBudget` limits calls, **not allocation bytes**. `geometry_identity` uses its own default ReceiptLimits. P3's `bounded_manifest` also constructs its set before charging retained vectors. These preexisting P2/identity workspaces remain outside the new retained receipt; they are not retrospectively bounded by M1. Materialization, context, parameter/P1 construction and target verification likewise retain their separate mechanisms, including existing coverage gaps.

The new boundary is explicit: after successful candidate validation, measure its actual borrowable payload; reserve before retaining/moving it, and before allocating any additional descriptions, activation/transpose/class vectors or descriptors. Discard placeholder states and unused candidate fields. This satisfies new retained accounting, **not an all-preparation allocation bound**. If M1 is interpreted to include every preexisting P2 internal allocation before creation, this package cannot honestly satisfy it: a separately reviewed count/check path inside `plan_stage_geometry` / `initial_stage_metadata`, compact-layout and manifest-set construction is required. A guessed maximum or allocating the candidate to discover that pre-allocation size is not an alternative.

## Ownership and signatures

The following are proposed crate-private signatures unless explicitly marked cross-crate. Names are concrete, not capabilities. `CanonicalPreparation` keeps the actual owner borrows and the original request; the plan remains the same Arc inside CheckedPreparation/CheckedBlockPreparation. No self-borrowing object is required.

```rust
struct CanonicalPreparationInput<'a> {
    materialized: &'a MaterializedProfile,
    registered: &'a RegisteredCase,
    plan: Arc<DiagonalPlan>,
    request: &'a DiagonalRequest,
    actual: PlanningParameters<'a>,
}
struct RetainedPreparation<'a, T> {
    checked: T,
    materialized: &'a MaterializedProfile,
    registered: &'a RegisteredCase,
    request: &'a DiagonalRequest,
    accounting: PreparationAccounting,
}
fn prepare_canonical<'a>(input: CanonicalPreparationInput<'a>,
    actual_partition: &[Vec<MicroPart>])
    -> CpuPreparationResult<RetainedPreparation<'a, CheckedPreparation<'a>>>;
fn prepare_blocks_canonical<'a>(input: CanonicalPreparationInput<'a>,
    actual_partition: &[Vec<MicroPart>])
    -> CpuPreparationResult<RetainedPreparation<'a, CheckedBlockPreparation<'a>>>;

// Private fields; consuming extraction only after every borrowed check/reservation.
struct CandidateComponents { geometries: Vec<StageGeometry>, keys: KeyManifest }
impl CandidateDescription {
    fn retained_components_payload(&self) -> Result<CandidatePayload>;
    fn into_checked_components(self) -> std::result::Result<CandidateComponents, Self>;
}
// Public only because nn is a separate crate; narrowly documented logical payload,
// no authority and no serialization. Includes private GeometryIdentity fields.
impl StageGeometry { pub fn requested_heap_payload_bytes(&self) -> Option<usize>; }

struct PreparationAccounting {
    limits: ReceiptLimits, plan_bytes: usize, retained: usize,
    live_temporary: usize, peak_temporary: usize,
}
impl PreparationAccounting {
    fn seed(plan: &DiagonalPlan, original: ReceiptLimits) -> Result<Self>;
    fn reserve(&mut self, bytes: usize) -> Result<Reservation<'_>>;
}
struct Reservation<'b> { accounting: &'b mut PreparationAccounting, remaining: usize }
impl Reservation<'_> {
    fn retain(&mut self, bytes: usize) -> Result<()>; // temporary -> retained, no double charge
} // Drop releases remaining temporary; failure consumes the containing preparation.

struct DescriptorSizes { interactive: Option<usize>, layout: usize, keys: usize, security: usize }
struct DescriptorDigests<'a> { parameter: &'a str, bootstrap: Option<&'a str> }
impl PreparationDescriptorFacts<'_> {
    fn descriptor_sizes(&self) -> DocumentResult<DescriptorSizes>; // allocation-free
    fn physical_layout_bytes_with(&self, hashes: DescriptorDigests<'_>) -> DocumentResult<Vec<u8>>;
    fn cpu_key_policy_bytes_with(&self, hashes: DescriptorDigests<'_>) -> DocumentResult<Vec<u8>>;
    fn security_declaration_bytes_with(&self, hashes: DescriptorDigests<'_>) -> DocumentResult<Vec<u8>>;
}
struct CanonicalDescriptors {
    interactive_parameters: Option<Vec<u8>>, layout: Vec<u8>, keys: Vec<u8>, security: Vec<u8>,
    parameter_digest: String, bootstrap_digest: Option<String>,
    layout_digest: String, keys_digest: String, security_digest: String,
}
// private fixed methods on single/block retained owners call the same collector;
// return owner plus descriptors, never an independently forgeable source view.

fn binding_workspace_payload(p: &MaterializedProfile, c: &RegisteredCase,
    params: PlanningParameters<'_>, r: &DiagonalRequest) -> Result<usize>;
fn validate_plan_binding_accounted(p: &MaterializedProfile, c: &RegisteredCase,
    plan: &DiagonalPlan, params: PlanningParameters<'_>, r: &DiagonalRequest,
    accounting: &mut PreparationAccounting) -> Result<()>;
```

`Result` aliases in candidate/accounting/binding signatures mean admission errors; descriptor methods use document errors. Add a distinct preparation accounting error preserving required/configured bytes; never manufacture a Candidate issue. `seed` checks plan charges and existing indexed-record count against unchanged limits. New objects add no P3 indexed records. Every arithmetic operation is checked.

Validate partitions, stage graph, bias/activation policy, complete key manifest including separate-bundle semantics, actual BootParams/P1 and EvalConfig before consuming candidate. Candidate failures return the original boxed row without cloning. Success moves geometries; block moves `keys.training` into `rotations`; single uses its geometry's exact union. Drop unused bootstrap manifest vectors, digest and placeholder states. Reserve transient outer geometry-vector storage until its iterator drops; reserve Description storage before filling it. No clone of geometry or states.

The retained input/request/case references make associations inspectable. Store/compare the same plan Arc (`Arc::ptr_eq`) and parameter allocation (`ptr::eq(actual.params, supplied_arc.as_ref())`); compare actual BootParams and retained metadata identity. Byte equality alone is insufficient. No second input resolution/materialization or equal-hash replacement plan is accepted. Compatibility wrappers keep accounting alive through their factory invocation; canonical production callers retain the whole owner.

## Exact requested-payload accounting

Let `V(T,n)=checked(n*size_of::<T>())`; nested heap payload is added to containing vector elements once. Inline root structs/Arc handles add no heap charge. Sizes below are logical requested lengths, not capacities or RSS.

| Allocation / owner | Exact size and reservation point | Transfer / release |
|---|---|---|
| Existing plan | `plan.charges().combined()` once; record count `plan.transitions().len()` | Borrow/Arc reuse costs zero; owner retained |
| Existing candidate P2 | `make_stages` geometry/state internals, `manifest` set/vectors, geometry digest | Explicit old-P2 exclusion; states/digest/unused keys dropped before retention; errors retain existing diagnostics |
| Geometry nested payload `G` | nn helper counts identity dims, acts, Reuse pool; shapes; mask structs plus both step vectors; transpose outer headers, tuple vectors and chains; keys; compact plans/containers/biases | Reserve actual `G` before consuming candidate; transfer without cloning |
| Moved geometry outer buffer | `V(StageGeometry,S)` | Count live transient during conversion; release only when owning iterator drops |
| Descriptions | block `V(Description,S)`; single inline. Add each stage's `V(DiagonalActivation,A_s)` and explicit Description transpose `V(i32,pool_len)` | Reserve before allocation; retained, including the distinct transpose copy already inside geometry |
| Block extra arrays | `V(usize,S)+V(i32,training_union_len)` | Classes allocated after reserve; training vector moved, not recopied |
| Descriptors | I: `P+L+K+D`; NI: `L+K+D`; each individual byte vector <=2,097,152 | Reserve each exact output before `try_reserve_exact`; retain returned vector |
| Descriptor digests | I `4*64`; NI `5*64` | Check before each hash String allocation; retain parameter/bootstrap and L/K/D hashes |
| Plan/P1/source bytes | Borrow existing canonical bytes, original identities and approved receipt | No auxiliary copy, no new A_MAX on old transcript |
| Binding check workspace | `binding_workspace_payload` exact maximum of live requested buffers, described below | Reserve before canonical_binding; release after result and temporaries drop; same hook for future post-start recheck |
| Error paths | Existing boxed Candidate/Admission issues and P2 traces may allocate/retain outside new successful-output charge | Preserve diagnostics; explicitly not a zero-allocation/error-RSS claim |

`descriptor_sizes()` factors each existing Fields encoder into validation plus one shared Writer body; run it with `output=None`. Hash fields have known Text64 width; private size-only placeholders never escape or produce bytes. No duplicate encoding body, geometry clone or preliminary encoded parameter buffer. The four existing source helpers become wrappers over those same projections. The collector encodes I parameters once, hashes that retained buffer once (NI hashes borrowed P1), computes bootstrap hash once, then produces L/K/D using borrowed retained digests. Thus the current repeated temporary I parameter Vec and temporary hash Strings disappear on this path. Source validations, registered security validation and allocation-free wire parsing remain.

Binding reservation needs a real narrow hook, not `max_bytes` multiplied by a constant. Factor existing transcript/identity encoding traversals into counting and output sinks, preserving bytes. Count simultaneous binding/consumed transcript lengths, partition outer/inner vectors, consumed-count Vec, dual identity bytes and digest copies, source-sort pointer vectors, and geometry-identity scratch. Track their actual scope ends for the maximum; do not sum mutually dead buffers. `update_rows()` borrows rows/checkpoint slices and allocates no successful-path row buffer. Existing identity construction has no equivalent borrowed-size hook today; add it before claiming this reservation is implemented. Use this same accounted validator pre-key now and after genuine start in phase3; it performs no P2 update calls.

## Minimal package, callsite and bounded verification

Change `study_client.rs`, `study_client/blocks.rs`, `study_admission.rs`, `study_reference/document/execution_descriptors.rs`; add narrow sizing support in `study_reference.rs` and `ckks-train-nn/src/diagonal_plan.rs`; add private `study_execution.rs` plus module declaration for accounting/collection. Freeze existing vectors; no new dependency. Include only focused tests with this package.

Production location is `study_runtime::run_owned`, after retained context/case/input capture. It currently has no planning parameters/request. Phase2's collector stays disconnected there: adding implicit defaults would hide original limits and consumption. Future wiring must carry an explicit immutable `DiagonalRequest` (including Consumption, original limits and work budget) through StudyRunRequest and `study_run::run` parsing, then select owned registered parameter/P1 objects inside the installed frame. Full-only RegisteredCase membership may support a Quick operational plan, never Full scientific admission. Keep `MissingExecutionDependency` until the reviewed continuation and genuine gates/start exist.

Proposed positives: six canonical Quick-consumption preparations, I/NI × stages1/2/3, using local Xavier Iris emlp profiles confirmed by root. Share three materializations, two secure parameter owners and one NI metadata owner where signatures permit. For each successful plan+retained preparation: `plan_diagonal` contributes S geometry/initial, SU updates, (S−1)U handoffs and one wrapper; `describe_candidate` contributes S geometry/initial. With U=2 the six total **24 geometry,24 initial,24 updates,12 handoffs,6 wrappers**, and 12 binding traversals (planning plus preparation). Extra explicit rechecks/refusals are additional inventory, not hidden in those totals.

Check retained pointer/Arc identity, moved geometry payload, descriptor lengths/digests, zero crypto-factory counters, and preservation of candidate diagnostics. Pure tests cover seed overage, exact-fit/one-byte-over reservation, overflow, transfer/release and failure cleanup without constructors. Source-only negative association tests must identify their actual earliest refusal boundary; no promise of zero planners after candidate-stage rejection. Compile first; freeze runtime inventory after independent source review before execution. Synthetic block fixtures cannot count as canonical positives.

Full Task54 remains 21 profiles, full2680+20+7 and quick268, all required baselines/tables/seeds; unresolved B/same-N/authors and existing holds are unchanged.
