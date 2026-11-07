# D6c3 concrete target and prospective admission design

Status: DESIGN RATIFIED after independent FIX1 review. No implementation assignment or execution authority.
2026-11-07. Governing requirements: `docs/study-reference-design.md` §§8–12 and
`docs/experiment-matrix.md`. FIX1: root reports D6c2b reviewed/integrated at
`d76ef05034d078b459500f83dc9271aabe88e157`; current public reader/store APIs
were re-inspected and remain unchanged. Original candidate/report/review bytes
are archived separately. Every proposed extension still requires separate
ownership and review. This design changes no
scientific policy, historical FULL gate, input lock, registered row or user hold.

## 1. Construction boundary and exact target

Only the reviewed `ReferenceStore::read_reference` path can supply a
`VerifiedReference`. A successful parser result is not that capability. Require
actual F32, complete origin-to-final canonical full plan, seed0, approved
reference writer/session/arithmetic, exact canonical shared and seed0 receipt
bytes, valid clock, full endpoint and approved context. Never accept a threshold,
accuracy, endpoint, source path or provenance map supplied as measurement facts.

Algorithm string is exactly
`seed0-final-minus-half-percentage-point-f64-v1`. Require
`0 <= correct <= test_count <= 9007199254740992` and `test_count > 0`.
Compute binary64 `a = (correct as f64) / (test_count as f64)`, then binary64
`d = a - 0.005_f64`, then `t = max(+0.0_f64, d)`, normalizing any zero to +0.
The offset bits are `3f747ae147ae147b`. Use ordinary separate operations, no
decimal/rational replacement, reassociation, tolerance or F32 conversion.
Store and recompute exact accuracy/target bits; JSON values must round-trip to
those bits. These are reporting operations, not training arithmetic.

Lock the full exact seed0 reference file SHA256 and payload SHA256 separately.
For seed0 compare the complete selected receipt to the reference. For seed1–9
derive that seed's own canonical receipt through the normal resolver/materializer;
compare complete shared contract bytes, not seed-specific bytes to seed0.
Require receipt.seed == selected seed and all normal selected-source checks;
never manufacture another seed's receipt by changing its seed field. Distinct
seed receipts include the seed and therefore must differ from seed0. Missing
captured seeds remain MissingInput. Changed shared contracts require a new
campaign and lock; no override or input fallback is introduced.

Root's concurrent input ruling selects a future complete baseline-produced
eMLP2/3 seeds0–9 bundle while retaining historical raw files and old registry/
lock. This design does not migrate that data or alias old digests. Resolve only
the explicitly reviewed canonical registry/input version active for the chosen
campaign; its changed shared identities require new campaigns and locks. The
case projection binds the exact registry digest, never a caller configuration.

## 2. Small API and ownership surface

Names below are proposed signatures, not available APIs. All capability fields
are private; none implements Deserialize, public From/FromParts, Default, Clone,
or a public trait allowing arbitrary readiness implementations. Immutable
inspection views are not construction authority.

```rust
// Existing reviewed reader, with narrowly extended immutable accessors:
impl VerifiedReference {
    pub fn run_id(&self) -> &str;
    pub fn profile_id(&self) -> &str;
    pub fn verified_time(&self) -> ReferenceTimeView<'_>;
    pub fn verified_completion(&self) -> ReferenceCompletionView<'_>;
    // Existing contract(), receipt(), final_correct(), test_count(), digests.
}
// Views have private fields, immutable getters only. They borrow the verified
// object, never expose ParsedReferenceDocument or permit replacement of claims.

pub struct CampaignId(/* private validated Text128 */);
pub struct AttemptId(/* private validated Text128 */);
impl CampaignId { pub fn parse(request: &str) -> Result<Self, TargetError>; }
impl AttemptId { pub fn parse(request: &str) -> Result<Self, TargetError>; }

pub struct ProfileTargetStore<'c> { /* held directory handles, &'c context */ }
pub struct NewProfileTargetStore<'s, 'c: 's> {
    /* &'s mut ProfileTargetStore<'c>, owned exclusive profile-guard FD */
}
pub struct VerifiedReferenceSnapshot<'r, 'c: 'r> {
    /* &'r ReferenceStore<'c>, owned VerifiedReference and exact bytes,
       validated run ID and source-directory/file identity stamps */
}
impl<'c> ReferenceStore<'c> {
    pub fn read_reference_snapshot<'r>(&'r self, id: &str)
        -> Result<VerifiedReferenceSnapshot<'r, 'c>, PublicationError>;
}
impl<'c> ProfileTargetStore<'c> {
    pub fn open(root: &Path, campaign: &CampaignId, profile: &ResolvedProfile,
                context: &'c VerifiedStudyContext) -> Result<Self, TargetError>;
    pub fn claim_new<'s>(&'s mut self)
        -> Result<NewProfileTargetStore<'s, 'c>, TargetError>;
    pub fn read_target(&self, references: &ReferenceStore<'c>,
                       seed_profile: &MaterializedProfile)
        -> Result<VerifiedTargetForSeed, TargetError>;
    pub fn recover_lock(&mut self, references: &ReferenceStore<'c>,
                        expected_file_sha256: &str)
        -> Result<PublishedTargetLock, TargetError>;
}
pub fn lock_target<'r, 's, 'c: 'r + 's>(
    source: VerifiedReferenceSnapshot<'r, 'c>,
    new_store: NewProfileTargetStore<'s, 'c>)
    -> Result<PublishedTargetLock, TargetError>;

// read_target owns reference/lock byte snapshots and derived identities. It
// reads the reference run ID from the lock only as a validated store selector;
// identity still comes from the exact bytes and canonical verified reader.
pub struct VerifiedTargetForSeed { /* immutable snapshots, identities, bits */ }
impl VerifiedTargetForSeed {
    pub fn threshold_bits(&self) -> u64;
    pub fn seed_receipt(&self) -> &SeedInputReceiptV1;
    pub fn lock_file_sha256(&self) -> &str;
    pub fn lock_payload_sha256(&self) -> &str;
}

pub struct RegisteredCase { /* registry-derived axes only; no readiness */ }
pub fn select_registered_case(registry: &StudyRegistry, selector: CaseSelector)
    -> Result<RegisteredCase, CaseSelectionError>;

// Later D7-owned adapter. Not a public bool or trait extension point.
pub struct CheckedCaseForAdmission<'p> { /* private, owns gate leases */ }
// Its sole future constructor lives in the reviewed bench gate module and
// consumes real Input/Layout/Implementation/Resource/User gate capabilities.
// Until that module exists, production cannot construct this value.
pub enum ExplicitAttemptRequest {
    Initial { attempt: AttemptId },
    Retry { attempt: AttemptId, predecessor: PredecessorRequest },
}
pub struct PredecessorRequest { /* validated prior AttemptId and expected Links */ }
pub struct ExpectedLink { /* private validated file/payload SHA256 pair */ }
impl ExpectedLink {
    pub fn parse(file_sha256: &str, payload_sha256: &str)
        -> Result<Self, AdmissionError>;
}
impl PredecessorRequest {
    pub fn new(prior: AttemptId, expected_admission: ExpectedLink,
               expected_start: Option<ExpectedLink>) -> Result<Self, AdmissionError>;
}
// ExpectedLink parses two fixed SHA256 strings; it is only a request claim.
// Neither request variant is a gate, durable receipt or readiness capability.
pub fn admit_he_case<'s, 'c: 's, 'p: 's>(target: VerifiedTargetForSeed,
    case: CheckedCaseForAdmission<'p>, attempt: ExplicitAttemptRequest,
    store: &'s mut ProfileTargetStore<'c>, context: &'c VerifiedStudyTrainContext)
    -> Result<NewlyPublishedAdmission<'s, 'c, 'p>, AdmissionError>;
pub fn start_he_case<'s, 'c: 's, 'p: 's>(
    admission: NewlyPublishedAdmission<'s, 'c, 'p>)
    -> Result<StartedHeCase<'s, 'c, 'p>, AdmissionError>;
// Later D7-owned concrete HE constructor, not a caller-supplied closure/trait:
pub fn begin_checked_he<'s, 'c: 's, 'p: 's>(
    started: StartedHeCase<'s, 'c, 'p>) -> Result<HeSession<'c, 'p>, HeStartError>;
```

`VerifiedStudyTrainContext` does NOT currently exist. It cannot be a type alias
that makes today's `VerifiedStudyContext` accept arbitrary executable names.
Section 9 defines its separate review dependency. `CheckedCaseForAdmission`
contains the same immutable MaterializedProfile borrow that is handed to the
later trainer, exact case/target/context binding, and non-forgeable gate leases.
The ownership choice is explicit borrowing, not path reopening or hidden FD
transfer. `NewlyPublishedAdmission<'s,'c,'p>` privately retains
`&'s mut ProfileTargetStore<'c>` and `&'c VerifiedStudyTrainContext`; that store
owns root/ancestor/profile FDs and owns an optional private
`ActiveAttemptHandles` containing the validated case/attempt directory FDs and
identity stamps. Admission populates those handles under the profile guard;
the exclusive store borrow prevents replacement while either live token exists.
The token also owns checked case/gate leases, target snapshots, exact admission
bytes/links, explicit attempt and `ProcessStamp`. No flock is held after admission
returns. Start reacquires a fresh guard FD through the borrowed store; it never
opens a root path taken from a token, receipt or caller. The context pointer must
equal `store.context` via `context.artifact_context()`, and all target/case/context
identities must match. Equal lifetimes alone do not substitute for that check.

`StartedHeCase<'s,'c,'p>` receives those SAME borrows, handles through the store,
gate ownership, process stamp and snapshots, plus exact durable start bytes and
links. `begin_checked_he` consumes it, rechecks all held-handle identities and
process/context/gates before any key/context/encoding/training allocation, then
transfers materialized input borrow, gate leases, immutable linkage/threshold
and `&'c VerifiedStudyTrainContext` to `HeSession<'c,'p>`. Only after this checked
handoff may it release the store borrow and close ActiveAttemptHandles. HE
session creation errors consume authority and leave start evidence incomplete;
they never return the token. The context and inputs therefore outlive HE use;
the store need only outlive the two live tokens and consuming boundary. No
public conversion extracts an unchecked allocator permit or materialized trainer.

All three live types are non-Clone, non-Send/non-Sync and non-deserializable;
their private process stamp is captured internally as getpid(), Linux
`/proc/self/stat` start-time ticks, and `/proc/sys/kernel/random/boot_id`, using
bounded checked parsing. Capture/read failure rejects. Check the exact stamp at
admission, start, and AGAIN in begin_checked_he immediately before allocation;
fork children necessarily fail the PID check even if Rust values/FDs were copied
by fork. Exec cannot reconstruct them. The later HeSession retains the stamp
and rejects fork transfer at its execution/allocator entrypoints; reviewed code
does not fork with live tokens/session state. No retry method reuses a consumed
token. D6c3 itself allocates no HE resources.

For this live-only stamp, cap `/proc/self/stat` at16,384 bytes (existing bounded
proc-reader size), parse the kernel comm field through its final closing `)`
before counting the numeric tail fields, and extract exact positive PID and
u64 start-time field22. Cap boot_id at64 bytes, accepting only one36-byte
canonical lowercase UUID plus the kernel trailing newline. Reject trailing or
missing data/overflow; do not hash lossy strings or invent a process start value.
These caps bound kernel-text transport, not scientific timing policy. Retain
the stamp privately; no receipt schema field or deserialization route is added.

The source ownership is equally explicit: read_reference_snapshot is the ONLY
constructor for the private-field source snapshot and invokes the unchanged
approved reference verifier on its single descriptor read. It borrows its
originating ReferenceStore and retains that source directory/file identity;
lock_target consumes this snapshot and rereads its run ID through THAT held
store. It requires byte equality, unchanged source identities and compatible
approved Reference-role context with new_store. There is no API that attaches
an unrelated store B to a VerifiedReference read from A, no conversion from a
bare VerifiedReference into a snapshot, and no lock_target overload accepting
only a parsed/verified claim. Both stores remain alive until lock publication
finishes. Reference and target roots need not be co-located.
Reading/recovering receipts returns only `VerifiedAdmissionRecord` or
`VerifiedStartRecord` inspection types, never either live capability.

| State | Owns | Borrows retained | Authority released next |
|---|---|---|---|
| VerifiedReferenceSnapshot | exact bytes, verified data, validated run ID and source identity stamps | originating ReferenceStore for `'r`, which borrows context for `'c` | consumed only for verified lock publication or internal read-target verification |
| NewProfileTargetStore | exclusive profile-guard FD | target store mutably for `'s`, context transitively for `'c` | lock publication consumes guard; no HE authority |
| NewlyPublishedAdmission | checked case/gate leases, ProcessStamp, attempt, target/admission bytes and links | target store mutably for `'s` (owns ActiveAttemptHandles); train context for `'c`; inputs transitively for `'p` | one start call consumes it |
| StartedHeCase | previous owned state plus durable start bytes/links | SAME `'s/'c/'p` borrows | one concrete begin_checked_he call consumes it |
| HeSession | gates, immutable execution links/threshold and ProcessStamp, then actual HE state | train context for `'c`, canonical inputs for `'p`; no target-store borrow after checked handoff | process-checked execution only, no transferable public permit |

Dropping a token closes only owned live state/borrows; it deletes no evidence,
rolls back no publication, and grants no replay. On a consumed-call error the
exclusive store borrow ends but any leftover ActiveAttemptHandles are closed
under ordinary RAII/reset before the store is reused; no files are removed.
Independent callers may open a second ProfileTargetStore and cooperate through
flock; a second Rust borrow of the SAME store is prohibited while a token lives.

ReferenceTimeView must expose verified start/completion u64 Unix nanoseconds,
wall and monotonic source tags, end-to-end duration and valid/available timing
already enforced by 2b. ReferenceCompletionView must expose full/consumed-plan
digests, all origin/planned/actual/drop counts, final endpoint index, complete
final checkpoint (including boundary IDs, exact-boundary flag and interval
counts), and final-state digest claim. The endpoint's immutable observation and
timing getters preserve the reviewed 2b values. These additions do not change
the reader's construction path or imply final-model replay.

## 3. Fixed canonical/wire contract

Reuse `hekit-study-reference-encoding-v1` primitive conventions. Header is
Text(encoding), Text(schema domain); body follows tables below in exact order.
Every displayed key is mandatory, including option keys with null. JSON key
order is irrelevant; typed field order is not. Hash all fields except the
integrity object's own payload digest. JSON includes integrity encoding/hash
strings, and those strings are encoded. File SHA256 hashes all exact file bytes,
including whitespace, and is never embedded recursively in its own document.

Primitives are the unchanged document grammar: Text = u64LE UTF8 length+bytes;
U64 = u64LE; Bool = false00/true01; enums = explicit byte tags; lists = u64LE
count+ordered items; options = null00/present01+value; Hash = Text of exactly64
lowercase hex; Bytes = u64LE decoded length+raw bytes, transported as unescaped
lowercase even-length hex; F64 = u64LE IEEE bits. Hex16 JSON is the conventional
16-digit lowercase spelling of the integer bit pattern. Accuracy object is
`{correct,test_count,value,bits}`, canonically correct,count,value-bits (bits is
redundant validation, not encoded twice). Target is `{value,bits}`, encoded once.
Reject negative/exponent/fraction/leading-zero integer tokens, overflow,
nonfinite float, duplicate decoded keys (including escaped aliases), unknown or
missing keys at every level, unsupported enum/schema, trailing documents,
invalid UTF8/escapes and noncanonical hex. No serde defaults or repair.

Domains and header enum values:

| Kind | schema/domain | kind byte | status |
|---|---|---:|---|
| Lock | `hekit-study-target-lock-v1` | target_lock=0 | complete=0 |
| Admission | `hekit-study-he-admission-v1` | he_admission=1 | complete=0 |
| Start | `hekit-study-he-start-v1` | he_start=2 | complete=0 |

Auxiliary hash domains: `hekit-study-profile-slot-v1`,
`hekit-study-target-lock-id-v1`, `hekit-study-registered-case-v1`,
`hekit-study-case-binding-v1`, `hekit-study-admission-gates-v1`.
They use the same header convention. Profile slot encodes profile_id only;
lock ID encodes campaign_id,profile_id; registered-case encodes §5 CaseIdentity;
case-binding encodes §5 ExecutionBinding; admission-gates encodes §5 Gates.
This prevents SHA-of-JSON and cross-domain identity confusion.

Shared subrecords, in canonical order:

* `Policy = {scientific_revision:Git40,scientific_sha256:Hash,
  registry_sha256:Hash,portable_lock_sha256:Hash,
  design_documents:List<{path:Text4096,sha256:Hash}>,approval_sha256:Hash}`.
  Scientific revision/hash remain existing POLICY_REVISION/POLICY_SHA256.
  Design documents are exactly the context's five sorted paths (experiment
  matrix, reference design, reference document design, reference session
  design, training design); capture approved current file digests independently
  of that historical scientific digest. No sixth document is silently pinned.
* `IdentityDocument = {schema,encoding,canonical_bytes,sha256}` is unchanged
  existing lossless contract/receipt transport; do not parse inner identities.
* `Link = {file_sha256:Hash,payload_sha256:Hash}`.
* `Clock = {wall_source:unix_system_time=0,unix_ns:U64,
  monotonic_source:std_instant=0,elapsed_since_operation_start_ns:U64}`.
  All fields are captured, never supplied by a caller. Validity is a constructor
  obligation, not a caller-controlled valid=true field.
* `Writer = {api_version:Text128,algorithm_revision:Text128}` with exact strings
  `hekit-study-target-api-v1` and `hekit-study-target-protocol-v1`.
* `Integrity = {encoding:ENCODING_VERSION,hash:"sha256",payload_sha256:Hash}`.

Lock fields after header, in this exact order:

```text
schema, kind, status, writer, synthetic:false,
campaign_id:Text128, profile_id:Text128, lock_id:Hash,
policy:Policy, contract:IdentityDocument,
reference:{run_id:Text128,schema:"hekit-study-reference-v1",
           kind:full_fp32_final=0,link:Link,seed:U64=0,
           seed_receipt_sha256:Hash,materialized_sha256:Hash,
           writer_capsule_sha256:Hash,
           start_unix_ns:U64,completion_unix_ns:U64,
           completion:ReferenceCompletionProjection,
           final_observation:Accuracy},
algorithm:Text128 exact algorithm in §1, target:Target,
created:Clock, provenance:ReferenceProvenanceV1, integrity:Integrity
```

`ReferenceCompletionProjection` order:
origin_completed_updates,origin_processed_samples,planned_updates,actual_updates,
planned_samples,actual_samples,planned_dropped_samples,actual_dropped_samples,
full_plan_sha256,consumed_plan_sha256,final_endpoint_index,final_endpoint,
final_state_sha256. First eight and index are U64; hashes are Hash;
final_endpoint is the unchanged document `Endpoint` record, including its
observation and available update/evaluation elapsed observations. Exact equality
to the verified source is mandatory, not just matching actual/planned numbers.
Lock has complete contract bytes but only seed0 receipt digest; the complete
receipt is retained in the exact linked verified reference, avoiding needless
large duplicate receipt transport. Reverification requires that reference.

`ReferenceProvenanceV1` is the existing captured-only `ProvenanceClaim` branch
with its tag0 and unchanged nested fields, schema and canonical encoding from
document.rs; unavailable is rejected for locks. It belongs to the lock writer,
not a copy of reference provenance. Capture/recheck around lock operation uses
the seed0 canonical input that the reference reader verified. The lock must
carry the same approved policy binding as the source, while permitting another
explicitly approved build receipt in that policy.

Admission fields after header, in order:

```text
schema, kind, status, writer, synthetic:false,
campaign_id:Text128, profile_id:Text128, lock_id:Hash,
policy:Policy, lock:Link, reference:Link,
contract:IdentityDocument, selected_input:{seed:U64,materialized_sha256:Hash,
                                        seed_receipt:IdentityDocument},
case:CaseIdentity, case_id:Hash,
execution:ExecutionBinding, execution_binding_sha256:Hash,
attempt_id:Text128, predecessor:Option<Predecessor>,
gates:Gates, gates_sha256:Hash,
admitted:Clock, provenance:CaseProvenanceV1, integrity:Integrity
```

`Predecessor = {attempt_id:Text128,admission:Link,start:Option<Link>}`. On disk,
null is the Initial branch of the closed request in §2 and a present object is
Retry; there is no third branch or independent caller-controlled optional field.
Same campaign/profile/case/seed required; predecessor attempt differs. Both
presence and absence of prior start are observed under the profile guard and
must match the requested exact links; a changed optional-start state rejects
the request for explicit resubmission, never silently refreshes it. The record
means observed prior receipt state, never proves failed/finished training.
Only the case-level eligibility rule in §6 permits null. A fresh filename does
not imply a first admission. Every subsequent admission must be Retry with an
explicit verified predecessor; no latest-attempt search, auto increment or
inferred chain head exists. Prior immutable files remain unchanged. An empty
or orphan attempt directory is not proof of no history and cannot be relabeled
as a fresh first attempt or reused by this protocol.

Start fields after header, in order:

```text
schema, kind, status, writer, synthetic:false,
campaign_id:Text128, profile_id:Text128, lock_id:Hash,
policy:Policy, lock:Link, reference:Link,
case_id:Hash, execution_binding_sha256:Hash, seed:U64,
seed_receipt_sha256:Hash, materialized_sha256:Hash,
attempt_id:Text128, admission:Link,
started:Clock, provenance:CaseProvenanceV1, integrity:Integrity
```

Start records duplicate identity digests for easy rejection, and link the full
case/input/gates through the exact admission. They do not embed another 31MB
receipt. All duplicate fields must equal verified admission/target fields.
No start verification succeeds without its admission, lock and reference.
No receipt asserts training success or emits invented result counters.

## 4. Bounded transport and parser reuse

Keep new wire parsers as `document/target.rs` and `document/admission.rs` children
of existing private document module, so they reuse Decode/Encode, Field/Map
visitors, float parsing, Hash/IdentityDocument/Endpoint and provenance records.
Do not fork the JSON parser or publish its private DTOs. Narrow proposed changes
to document.rs: bounded Encoder constructor accepting the three fixed new
document layouts; a closed `DocumentLayout` enum for lexical preflight's exact
large-hex paths; existing entrypoints continue to select Reference layout with
identical caps. No public configurable cap/path list and no path-prefix wildcard.

Reuse global `C=2,613,526`, `R=31,584,142`, `A=2,097,152` from the approved
reference schema. New parser caps derive strictly from included blobs:

| Document | large blob paths | max file bytes | max canonical bytes |
|---|---|---:|---:|
| lock | contract.canonical_bytes | 2C+A = 7,324,204 | C+A = 4,710,678 |
| admission | contract.canonical_bytes; selected_input.seed_receipt.canonical_bytes | 2C+2R+A = 70,492,488 | C+R+A = 36,294,820 |
| start | none | A = 2,097,152 | A = 2,097,152 |

Charge every other byte, whitespace and escape to A; no escaped large hex.
Independently cap each blob, then enforce total and outer budget before serde
allocation. Later enforce trusted profile's existing C/R formulas after exact
canonical identity matching, never based on claimed profile allocation hints.
Reference read cap remains 70,492,488. Finite integers/floats and typed list caps
are checked before conversion to usize. Allocation failures return error.

Other caps reuse existing limits: JSON container depth16, key64 decoded bytes;
ordinary decoded string limit4096, Unix-byte-hex string limit8192, compiler
decoded string limit16384; their raw token bound is decoded_limit*6+2 for
JSON escaping, while the unescaped large identity bound is blob_limit*2+2
(the existing lexical path exceptions remain exact);
Text128 IDs/versions, Text4096 paths and normal text, compiler Text16384;
exact five design docs; exact seven thread environment keys; source manifest
at most512 records, features64, argv256 and each decoded Unix byte vector4096.
Thread-environment transport cap32 remains, narrowed to the exact seven unique
keys by production policy. These are the existing document.rs limits unchanged.
Completion endpoint inherits crossed-boundary cap400 although today's final
endpoint has one. No endpoint array is duplicated in locks. Gates has exactly
five named fields, not an unbounded list; CaseIdentity and ExecutionBinding have
fixed fields. Their strings are <=128, descriptor hashes64, options fixed and
not recursive. These additions fit A by independent maximal-field accounting
before code freeze; if inherited provenance already consumes A, reject at A,
do not silently enlarge it. A is a transport headroom budget, no scientific
tolerance/cap. Store presence checks stop on the first entry and never load an
unbounded attempts listing. No cap reduces expected profiles/seeds/attempts.

Read once into an immutable owned buffer through held descriptor-relative
nofollow regular-file handles. Extend ReferenceStore with the private-field
`VerifiedReferenceSnapshot<'r,'c>` API in §2, retaining its originating store
borrow, validated run ID, source-directory/file identity stamps, verified value
and `Box<[u8]>`; construction invokes the same reviewed verifier on that single
buffer, not a second path read. The foundation read extension returns checked
identity stamps captured around that read; callers cannot supply them.
The public VerifiedReference need not expose parser internals or bytes.
Target keeps both original buffers throughout admission/start. At start the
ratified protocol additionally requires a fresh bounded read of the current
lock through the same held store, exact byte/hash comparison to the snapshot,
and revalidation; that fresh buffer is discarded after comparison. This is a
separate recheck, not parse-then-reopen within a verification operation. Also
recheck exact admission bytes before publishing start. Never dereference an
artifact-supplied absolute path. A source filename is only a store selector.
Require snapshot source identities and bytes to match a descriptor read through
its own borrowed source store at lock creation, so a stale standalone capability
cannot lock substituted bytes. read_target may consume the snapshot into owned
verified data/bytes internally after its complete read; the later live token
owns target snapshots and borrows the target store as §2 specifies. It does not
need to reopen the reference path at start, and retains no hidden source path
lookup authority. All source/build/input rechecks still use the approved context.

Memory remains bounded but can be substantial: simultaneous reference,
admission and canonical identity buffers plus SHA helper copies can reach
hundreds of MiB for Fashion. Sequence buffer creation/drop, use borrowed slices
where possible, retain the two required snapshots, and publish measured
allocator bounds only after later implementation tests. The existing SHA helper
is not a streaming zero-copy promise. Do not reduce Fashion to fit memory.

## 5. Exact registered case and later checked gates

`CaseSelector` is a request tuple, never an authority: origin/family, profile,
seed, backend, mode, placement, packing variant. Resolve exact membership from
the pinned bytes of `configs/experiment_matrix.json` through StudyRegistry's
approved root; propose one immutable registry bytes/digest accessor if needed.
No free-form case ID controls a path or grants applicability. Unknown key/value,
duplicate axes, unsupported registry schema/version or unmatched tuple fails.

`CaseIdentity` order/types:
registry_sha256:Hash, origin:family=0|supplement=1,
registration_id:Text128, profile_id:Text128, seed:U64(0..9),
scope:full=0, role:main=0|baseline_a=1|baseline_b=2|baseline_c=3|main_ablation=4,
backend:cpu=0|cuda=1|external_cpu=2,
mode:interactive=0|noninteractive=1,
placement:client_refresh=0|when_needed=1|after_update=2|after_activation=3|upstream_verified=4,
packing:batched_diagonal=0|row_2020=1|diagonal_2020=2|alternating_reboot_row_column=3|official_upstream=4,
security_policy:n16_train=0|n16_boot=1|seal_2020=2|upstream_verified=3|reduced_level_supplement=4,
logical_batch:U64, eval_config:Option<{hoist:Bool,bsgs:Bool,lazy_relin:Bool}>.
Full target admission rejects quick and historical FULL selectors. Quick remains
every applicable seed0/two-update row, separately handled outside this protocol.
Do not assign invented optimization flags to families: optional flags are null
when registration has no ablation axis; actual implementation flags are still
bound in ExecutionBinding. A supplement with incomplete concrete registration
cannot yield RegisteredCase merely by supplying caller flags.

Exact family expansion: iterate all17 families' listed profile_ids × backends ×
modes × placements × packing_variants (or singleton packing) × full_seeds.
Derive role and security from `/crypto` and exact override, never CLI defaults.
This is 2,040 base rows + 510 B64 rows + 130 B128 rows = 2,680 full family rows;
their seed0 quick projections number268. These are expected registrations,
not availability or execution counts. Main/packed I use n16_train, main/packed
NI n16_boot, matched-a n16_train; external retains upstream_verified and exact
captured cryptography later, never fabricated equal-security n16_boot.

Retain all seven supplement entries separately: exact-Iris supplement expands
the two listed packings × seeds0–9 under seal_2020 (20 new prospective rows,
distinct historical seed2020 anchor). FULL-1 preservation entry is archival,
never dispatchable. MNIST784 weighted partial, reduced-level, and rq3_all8 retain
their complete declared axes/expected seeds and unresolved concrete profile
bindings; rq3 retains all eight booleans and its CPU:L31/CPU:L7/CUDA:L7 groups.
Do not pretend these prose profile selectors are among the21 concrete profiles
or quietly discard them. Their unresolved row descriptors cannot obtain a
target until their canonical profile/implementation binding is separately
reviewed. Task51 plaintext and deferred Task53 remain explicit different roles.
Do not interpret 2,680 as the whole study denominator including supplements.

`ExecutionBinding` exact order:
implementation:main_diagonal=0|matched_row2020=1|matched_diagonal2020=2|packed_c=3|external_b=4,
implementation_revision:Text128, implementation_manifest_sha256:Hash,
parameter_identity_sha256:Hash, bootstrap_descriptor_sha256:Option<Hash>,
layout_plan_sha256:Hash, trainer_plan_sha256:Hash,
eval_config:{hoist:Bool,bsgs:Bool,lazy_relin:Bool},
bootstrap_engine:none=0|host=1|native_cuda=2|upstream=3,
key_policy_sha256:Hash, security_policy_sha256:Hash,
physical_binding_sha256:Hash, runtime_policy_sha256:Hash.

These are derived from reviewed implementation objects, never arbitrary caller
strings. Available source anchors include BackendOpts/EvalConfig,
PackedGeometryPlan/VectorSpec/LayerGeometry, and BootstrapMetadata's
parameter_identity_bytes/descriptor_identity_bytes. They do not currently form
a complete checked-case registry. Parameter hash binds full q/p/N/dnum/Delta,
levels and secret declarations, not preset name alone. Key policy binds exact
key/transpose/reuse choices; security policy links declared secret and applicable
security limitations. Runtime binding includes host/device/thread and refresh
role choices. HostBootstrapper CUDA is the registered default; native CUDA must
have its separate reviewed registration/gate. Parameter/descriptor digests refer
to exact canonical owned descriptors the later checker derives and retains;
absence of any needed canonical encoder is MissingCode, never hash(Debug).
Require crypto/mode/placement/packing/profile/optimizer/rates/full plan to match
the checked trainer; mismatches remain errors even with all hashes recomputed.

`Gates` has exactly ordered fields `inputs,implementation,layout,resource,user`.
Each is `{policy_revision:Text128,evidence_sha256:Hash}`. The hash names existing
reviewed evidence/capability identity; it is NOT proof of a passed bool. Later
gate adapters privately construct the records after consuming actual gate
capabilities, bind CaseIdentity+ExecutionBinding+selected receipt+campaign+
attempt+context, and transfer ownership of live resource/user leases. Inputs
include exact canonical source completeness; implementation includes truthful
I/NI role separation; layout covers every stage/partial lane and exact levels;
resource includes physical host/device and actual required/available memory;
user includes the actual authorized scope. Resource skip != missing code/input,
OOM != resource skip. Full encrypted block user hold cannot produce user gate.
Resource observations may become stale: the D7 owner must hold a resource lease
or perform its actual pre-allocation recheck immediately after durable start;
failure produces incomplete/failure evidence, never starts training with a
fabricated resource pass. D6c3 adds no numerical memory tolerance or gate policy.

## 6. Store paths, uniqueness and separate-process concurrency

Use a prepared private root outside historical results. All components traversed
from held FDs using openat/mkdirat with O_DIRECTORY/O_NOFOLLOW/O_CLOEXEC, same
mount-ID/device checks and existing ownership/mode rules. Newly created
directories use0700, files0600, sync new parent directory entries before use.
No arbitrary path argument reaches a publisher. Campaign/attempt identifiers
are nonempty <=128 ASCII `[A-Za-z0-9_-]+`, byte/case sensitive, no normalization.
Profile IDs must be the exact registry strings; profile slot is `p-` followed
by profile-slot domain SHA256. Lock ID is the domain hash defined above. Case
slot is `c-` followed by case ID. Hash slot collisions with unequal full identity
fail; they never merge or overwrite. Stable paths are:

```text
<root>/campaigns/<campaign_id>/profiles/p-<profile-slot-hash>/
    .profile-guard                 # zero-byte stable advisory-lock inode
    target.lock.json               # exactly one per campaign/profile
    cases/c-<case_id>/attempts/<attempt_id>/
        admission.json
        start.json
```

No target file is named by random lock ID; the fixed slot enforces uniqueness.
No case/attempt directory is created before a durable lock is verified. Use
closed `ProfileFile::{Target,Admission,Start}` in the existing publication
foundation, not a generic caller-controlled nested relative path API. Proposed
foundation extensions: open profile directory/guard, open validated case-attempt
directory, bounded read/publish/recover fixed filename, absent-final and
directory-empty checks, and guarded creation. Reuse the same short-write loop,
file sync, no-replace linkat, parent sync, owned-temp cleanup and final sync.
Do not create a second publisher or use rename-with-overwrite.

All cooperating publishers acquire an exclusive advisory `flock(LOCK_EX)` on
the SAME stable profile guard inode before claim_new, lock recovery, admission
or start. Open a fresh guard FD per acquisition, including same-process threads;
independent open-file descriptions then serialize. Never unlink/replace the
guard; validate regular file, owner/mode, zero length, link count1, path-to-inode
identity and same mount before and after acquiring. Creator uses O_EXCL then
fsync guard+parent; racing openers validate/sync the existing guard rather than
truncate it. A crash releases kernel flock; there is no persistent locked=true
bit, counter, mutable event journal, global seal or cross-profile mutex.

`claim_new` holds the guard through checks and durable lock publication. Reject
any existing target, any entry in cases (including malformed or temporary state),
or unsupported directory state. Thus two NewProfileTargetStore callers across
processes cannot both observe emptiness and publish: the second acquires after
the first, sees the final, and fails; linkat uniqueness is an independent backstop.
Admissions/start acquire the same guard, verify the final lock and exact identity,
and apply the following CLOSED case-level eligibility rule BEFORE creating any
case/attempt directory or temporary admission file:

| Request | Required existing case state under the guard | Action |
|---|---|---|
| Initial A | Case directory absent, OR a valid case directory containing only a valid empty attempts directory (or no attempts directory); no other entry is allowed | A may be created and published; serialized predecessor is null |
| Initial A with ANY existing attempt entry | Reject, whether that entry contains a durable admission, start, uncertain final, orphan temp, malformed object or nothing | `InitialNotEligible`; never look for a new ID that passes |
| Retry B from A | A differs from B and has an exact verified admission; optional exact start state matches request; B's whole attempt path is absent | Verify/re-sync A's named final receipts and parents, then create B and publish with explicit predecessor |
| Retry from absent/empty/orphan/malformed A, or start-only A | Reject: no valid predecessor admission exists | `InvalidRetryLink` or `UnresolvedAttemptState` |
| Any request whose selected attempt directory already exists | Reject even if both final names are absent | `Replay`/`UnresolvedAttemptState`; no directory reuse |

Initial's empty-attempts check is descriptor-relative, bounded-memory and
short-circuits on the FIRST entry other than `.`/`..`; it never assumes an entry
without a committed final is harmless. Validate absence by ENOENT only; I/O,
permissions, unreadable directories, symlinks and unknown case-level entries
are errors, not empty state. An empty case/attempts container with NO attempt
entry contains no attempt identity and may be reused after validating/syncing
its identities. In contrast, even an EMPTY named attempt directory blocks
Initial. This conservative distinction handles a crash during namespace setup
without inferring that a named attempted publication never occurred.

For Retry, open only the explicitly named predecessor, requested-new path and
required parent handles; never search for a latest attempt or require the
predecessor to be a mutable chain head. Check complete prior admission/start
against this same case, seed, campaign, lock and canonical inputs, exact expected
file/payload links and supported policy. Under the guard, resync the verified
predecessor final inode(s) and parent(s) before new publication; any sync failure
remains `PublishedButDurabilityUnconfirmed` with no new admission capability.
This is inspection/recovery of immutable records, never recovery of a live token.
Expected-start null rejects a start that is now present; expected-start Some
rejects absence or any differing bytes. A late start of A after B's admission
is still possible and does not falsify B's recorded snapshot of A; overlap
requires independent authorization, as below. No completed/dead status is inferred.

Uncertain or orphan evidence cannot be used as an Initial reset. A complete
uncertain admission can become a named Retry predecessor only through the full
verification/re-sync above (and with a valid linked start if one is present).
An orphan/temp-only/empty named predecessor cannot. Other attempt entries are
preserved and are not silently classified as completed/failed; Retry does not
assert there are no other attempts. A retry may branch from ANY explicitly
authorized valid predecessor, so the private user/resource gates must authorize
the requested retry/possible overlap without relying on presumed inactivity of
unnamed or incomplete attempts. No implicit 'only active attempt' gate exists.
If a first attempt leaves only an empty/orphan directory and no valid predecessor
exists, this case is deliberately blocked in this campaign. There is no built-in
clear/reset/rename-to-hide recovery; retain evidence for a separately reviewed
explicit recovery decision. Do not automatically create another campaign or
delete an entry to make Initial pass. This is a conservative failure policy,
not a scientific-result or time policy change.

The guard stays held from eligibility through directory creation and durable
admission publication. Thus competing Initial A/B cannot both succeed: after
A creates any named attempt entry, B rejects, even if A subsequently fails.
Atomic final link remains a second uniqueness check. At start, exact own
admission must be present and start absent; use the ActiveAttemptHandles owned
by the exclusively borrowed store, never a fresh caller path. Existing final
bytes are never removed. Lock publication precedes all case-dir creation by
cooperating code. Directory fsync completes before any success capability
escapes; profile guard release alone grants nothing.

Do not hold profile guard while HE trains or waits for a user; use it only for
bounded validation/publication. Different profiles never wait for one another's
targets. Same profile different attempts serialize receipt transitions but may
run concurrently only if later resource/user gates permit. An explicit retry
links the prior attempt's observed receipt state, not an unproved assertion
that the prior process died. Later scheduler must authorize any overlap; D6c3
does not infer failure from lack of a start/final result.

Retain directory FDs and recheck anchored path/inode identities before each
operation. Reject moved/replaced ancestor/profile/case/attempt directories.
Stable held descriptors avoid following swapped paths; rechecks keep displayed
namespace consistent. An adversarial same-user/root writer can race or change
files and is outside cooperating-process proof. These checks and timestamps
do not provide external attestation. No unsupported claim of a globally ordered
campaign or prevention of outside HE observations is made.

## 7. Lifecycle, clocks, crash and recovery

1. ReferenceStore::read_reference_snapshot reads exactly once and verifies
   canonical actual seed0 full endpoint, returning the source-store-borrowing
   snapshot. Lock creator consumes that snapshot and new-store guard, rereads
   through the retained originating source FD, verifies source identities and
   exact current bytes equal the snapshot, rechecks context
   and inputs, captures lock time, requires reference-start <= completion <=
   lock time, derives all fields and durably publishes target.lock.json.
2. Selected-seed reader snapshots and verifies lock and exact linked reference,
   derives its own canonical identities and cross-links every field. This can
   happen before independent HE gates, but grants no key allocation authority.
3. Admission consumes checked gates and target and takes the explicit store/
   train-context borrows; under profile guard rechecks current lock bytes/context
   and the CLOSED Initial/Retry eligibility table (including absent new directory
   and exact recovered predecessor state), validates retry/overlap authorization,
   captures admission time and requires lock time <= admission. Publishes
   admission atomically/durably. Only this successful call grants its private
   live token, bound to the current process, context, inputs and byte snapshots,
   retaining the store borrow and its ActiveAttemptHandles. Failure never returns
   a token or unverified public directory-handle constructor.
4. Start consumes that token; under profile guard rechecks source/build/inputs,
   exact current lock and admission, immutable identities and start absence,
   plus live independent gates and exact ProcessStamp. Captures start time and requires admission <=
   start; durably publishes start. Only then releases StartedHeCase to HE hooks.
   StartedHeCase retains the same store/context borrows. begin_checked_he checks
   ProcessStamp AGAIN plus context/gates/held identities immediately before
   allocation, transfers context/input/gate ownership to HeSession, and only then
   releases the store borrow. Allocation may fail; start is not training success.
5. Later binary-owned cumulative/checkpoint/final output must embed exact
   reference/lock/admission/start file and payload links, case/attempt, selected
   input and threshold. D8 preserves full origin and actual-update checkpoints.

Clock captures reject SystemTime errors/negative Unix era/u64 overflow,
Instant.checked_duration_since failure, any observed backwards wall reading
within an operation or across live admission/start, and any unavailable/invalid
required source timing. Capture wall before/after validation/publication and
monotonic operation durations; late clock failure consumes capability even if
file is visible. Never clamp, reorder, replace, interpolate or repair times.
There is no new wall-vs-monotonic discrepancy tolerance: without an approved
tolerance an arbitrary forward discontinuity cannot be distinguished from
elapsed time across processes. Detectable clock errors fail; no claim to detect
every unseen clock adjustment. Stored event time is the captured pre-publication
boundary, not a fabricated fsync completion timestamp. Durable publication
ordering is established by the protocol, not by clock granularity. Equality
is valid in start<=completion<=lock<=admission<=start.

Error categories: UnsupportedPolicy/Platform, MissingInput, MissingCode,
CaseMismatch, ContextChanged, ClockInvalid, ByteOrPayloadMismatch,
AlreadyExists/Replay, InitialNotEligible, UnresolvedAttemptState,
InvalidRetryLink, ProcessChanged, BeforePublication,
PublishedButDurabilityUnconfirmed. Errors carry bounded stable field labels,
expected attempted file/payload hashes and path for uncertainty, no arbitrary
input dumps or success capability. After final link, ANY sync/cleanup/clock
uncertainty blocks continuation. Caller cannot turn an uncertain return into
success by merely seeing the filename.

Before link: no admission capability; preserve unrelated/orphan files and apply
the named-directory rules above on any later request. After
admission durable but before start (including lost response/crash): same attempt
is permanently replay-ineligible. Explicit new attempt+predecessor required.
After start or uncertain start: no resume/live reconstruction; record inspection
may classify durable bytes but cannot restart. Recovery only reads, fully
verifies exact expected bytes under guard and resyncs existing inode/parent.
Recovered lock may be used for a NEW admission after full verification; recovered
admission/start never recreate live capability. Temps are preserved; cleanup is
separately authorized and never auto-unlinks an unknown inode. Partial namespace
initialization fails safely; empty case/attempts containers and empty named
attempt directories have the DIFFERENT treatment fixed in §6. No inspection
alone grants Initial after a named attempt entry. It cannot be mistaken for a
successful lock. No auto replay after
lock/admission/start uncertainty, and no later lock repairs unlocked history.

## 8. Synthetic vectors and meaningful RED contract

The following literal vectors were calculated independently with standalone
Python stdlib binary64 arithmetic, without project imports/training. They are
synthetic integer examples only, not scientific accuracy, reference or target
artifacts. Hex columns are u64 bit spelling (wire order is little-endian).

| correct | test_count | accuracy bits | target bits |
|---:|---:|---|---|
|0|1|0000000000000000|0000000000000000|
|1|201|3f7460cbc7f5cf9a|0000000000000000|
|1|200|3f747ae147ae147b|0000000000000000|
|1|199|3f749539e3b2d067|3efa589c04bbec00|
|1|3|3fd5555555555555|3fd50369d0369d03|
|199|200|3fefd70a3d70a3d7|3fefae147ae147ae|
|45035996273704|9007199254740992|3f747ae147ae1400|0000000000000000|
|45035996273705|9007199254740992|3f747ae147ae1480|3c54000000000000|
|9007199254740991|9007199254740992|3fefffffffffffff|3fefd70a3d70a3d6|
|9007199254740992|9007199254740992|3ff0000000000000|3fefd70a3d70a3d7|

Literal component schema goldens to freeze before production code: Accuracy
1/200 is `0100000000000000c8000000000000007b14ae47e17a743f`;
Target zero is `0000000000000000`; present retry begins `01` followed by
Text(attempt), Link, Option<Link>; null retry is exactly `00`; complete Bool
synthetic=false is `00`; case backend cuda/mode NI/placement AA/packing diagonal
is `01 01 03 00` at those exact table offsets. These independently specified
components do not replace complete envelope goldens. BEFORE any implementation,
freeze literal full synthetic lock/admission/start JSON plus canonical byte hex
and expected payload/file hashes under cfg(test)-only domains
`hekit-test-target-lock-v1`, `hekit-test-he-admission-v1`,
`hekit-test-he-start-v1`; include both predecessor branches and all option/tag
branches. Author those fixtures from the tables, independently of Rust encoder,
and independently review byte layout/cap accounting. This is a required first
implementation-package deliverable, not permission to generate scientific JSON.

SyntheticReference/SyntheticTargetLock/SyntheticAdmission/SyntheticStart are
separate private test types. Test stores use an explicit test namespace beneath
temporary directories; no synthetic constructor converts to production target,
CheckedCase, NewlyPublishedAdmission or StartedHeCase. Production parser rejects
test domains even when rehashed; no `allow_synthetic` CLI/feature/policy. Mac
portable tests may exercise private pure codec/clock/fs operation models, never
fabricate approved Linux contexts or publish production locks.

Before source, meaningful RED tests must fail for absent APIs/behavior, not a
deliberate typo or fixture regenerated by the tested encoder. Required groups:

* Count0 denominator, correct>denominator, >2^53, forged correct/denominator,
  F32 rounding, rational substitution, target ±1 ULP, negative zero/nonfinite,
  wrong algorithm/field/key/tag/digest; unchanged JSON value but changed bits.
* Full reference source seed1/7; quick, failure, F64, partial, best checkpoint,
  synthetic; source/payload/file digest confusion; whitespace-only replacement;
  full contract and seed0 receipt bytes altered then all hashes recomputed;
  coherent shortened/extra/repeated endpoint and wrong denominator rejection
  against the canonical oracle; time/provenance/approval/capsule/build mutation.
* Canonical Iris0/7 identity comparison without full training: shared bytes same,
  selected receipts distinct; synthetic target positive seed7 compatibility;
  production rejects synthetic target. Missing block seeds fail without repair.
  B48/B64/B128, dataset/split/selector/source/schedule/plan mismatch all fail.
* Every case axis, registry digest, seed, preset, q/p/Delta/secret/level,
  implementation/packing/EvalConfig, host/device/bootstrap/refresh/security and
  gate linkage substitution fails; arbitrary Registration/CheckedCase/live
  token constructors and deserialize/clone/fork use fail at compile/runtime.
* Fixed caps at limit/limit+1, escaped large blobs and duplicate aliases, outer
  depth/token/collection/overflow; bounded failure without allocating claimed
  counts. All21-profile metadata cap inventory and2680-row family expansion
  checked independently, plus20 Iris supplement rows and all unresolved
  supplement descriptors retained. Missing rows are not excluded from coverage.
* Two independently opened same-profile stores and separate OS processes race
  lock publication: exactly one succeeds. Independent threads use separate guard
  FDs. Same-attempt competing admission/start: at most one live success. Different
  profiles do not wait for each other's missing locks. Directory/symlink/FIFO/
  socket/mount/path-swap, malicious existing final and unknown temp states fail.
* I1 RED: A admission durable/start absent, then Initial B with a different ID,
  must reject; Initial A/Initial B racing through separate store handles and
  processes yield at most one successful initial publication. If A leaves only
  a named empty directory or temporary/uncertain final, B still rejects. A
  retry from absent/empty/start-only/torn predecessor rejects; a valid explicit
  predecessor with exact optional-start state succeeds ONLY with all ordinary
  gates and completed verification/re-sync. Wrong/late-start expectation and
  failed predecessor sync reject without new admission. Test zero-entry case/
  attempts containers separately from a named empty attempt; unknown entries
  and I/O errors never count as empty. No auto latest/reset/removal/new-campaign
  path exists. Unrelated existing attempts are preserved; overlapping retry
  does not infer predecessor inactivity and lacks authorization until the real
  private user/resource gates cover the requested overlap policy.
* I2 RED: a bare VerifiedReference cannot lock, cannot be attached to another
  store, and cannot construct a source snapshot. Snapshot from A cannot be
  relabeled as store B; changed origin file/directory inode or bytes reject
  before lock publication, including byte-identical replacement with a new
  inode. Destroying the source store while its snapshot is live is compile-fail.
  Destroying/mutably reusing the target store, train context or materialized
  inputs while a live token still requires them is compile-fail; destroying
  context/inputs while HeSession requires them is compile-fail. After successful
  consuming handoff the target-store borrow may end, but context/input borrows
  remain. Fault tests replace/symlink/move source/target/case/attempt paths and
  require failure through held identity checks, with no path-based fallback.
  Fork after admission, after durable start and at the later session boundary
  fails exact ProcessStamp checks before HE allocation/execution; missing/malformed
  process-start data fails closed. Deserialization/clone/context substitution
  cannot construct either token or HeSession. Private synthetic allocation
  counters must stay zero on every such rejection; no real HE test is required
  to prove these lifetime/protocol negatives.
* Inject short write, create/link/fsync/dir-fsync/cleanup and post-publication
  clock errors; no capability on uncertainty. Kill/fault at every transition,
  specifically admission-durable/start-absent; replay refuses, explicit linked
  retry preserves bytes, record recovery cannot start. Wrong predecessor digest,
  self-link, other-case/seed/campaign predecessor and changed lock reject.
* Timestamp equalities valid; completion>lock>admission or admission>start invalid;
  backward/unavailable clock cannot be repaired by a later good reading. Any
  source/build/executable change before start invalidates the token.
* B128 exact checkpoints38,75,113,150,188,225,263,300,338,375 and samples4864,
  9600,14464,19200,24064,28800,33664,38400,43264,48000 survive linkage unchanged;
  38/37 and4864/4736 intervals conserve375/48000. No exact4800 accuracy,
  interpolation, prorating, invented metrics or modified FULL controls.

## 9. Production executable provenance and bounded dependency order

Today's context validates only fixed `study_reference` cargo target, source
path, executable path, build receipt recipe and scalar execution_threads=1.
It is insufficient to admit HE from another binary. D6c3 must not broaden this
by accepting arbitrary target names, using reference receipts for study_train,
or claiming this makes any current legacy binary prospective.

Propose a separate reviewed role extension AFTER lock-only integration:
closed `ExecutableRole::{Reference,StudyTrain}`; fixed study_train source/target
path; exact approved role-specific recipe/capsule/receipt; same pinned source
and scientific policy; approval enumerates each role's receipt/capsule/executable
triple. Reference verification accepts reference-role receipts only. Admission/
start requires current StudyTrain role and embeds that role's actual executable,
argv/source/compiler/runtime facts. Artifact-provided executable paths never
select an executable. External adapters remain unavailable until a separately
reviewed adapter can bind actual external binary/context/input/export provenance;
a wrapper receipt alone cannot prove upstream admission or allocation ordering.

Concretely, keep one private verified context core with a closed checked role.
`VerifiedStudyContext::load` continues to load Reference role only. A separate
`VerifiedStudyTrainContext::load(StudyTrainContextRequest)` performs the reviewed
StudyTrain-role checks and wraps that same private core; its immutable
`artifact_context(&self) -> &VerifiedStudyContext` may be used by the store and
reference reader. This requires explicitly reviewing reader/current-process
separation: a train process may READ an approved reference-role artifact, but
ReferenceStore::publish_reference, reference session capture and lock creation
must require current Reference role. Admission/start require current StudyTrain
role. Reader checks the artifact's reference-role receipt and the calling
context's separately approved current-role receipt, never equating them. Existing
Reference capture remains scalar-only; new private capture_case/recheck_case
methods provide truthful case provenance. No public context-core constructor,
target-name string switch or bare role enum grants approval. This makes the
store/read signatures in §2 usable by both approved roles without pretending
today's hard-coded context already supports them.

`CaseProvenanceV1` is a DISTINCT schema, exactly
`{status:"captured",value:{schema:"hekit-study-case-provenance-v1",
role:"study_train",build,source,executable,invocation,host,thread_environment,recheck}}`.
Canonical order is captured-tag00, schema Text, role study_train=1 byte, then
the unchanged ordered nested fields of those seven existing record types.
Unavailable is not a legal branch in this schema. Use the same byte/field/list
grammar. Its host.execution_threads
is actual positive U64 bound to reviewed backend/runtime policy, not copied
scalar1; other host observations retain availability tags. Source/executable/
compiler/build and exact seven environment fields retain existing rules. The
new context must capture process identity and correct CPU/CUDA/external thread
semantics before this can be produced. Current reference provenance and parser
remain unchanged. This explicit schema/role extension is a dependency package,
not implemented by a silent parser exception. No new thread-count numerical
policy is invented here; unchecked or unavailable required runtime facts fail.

Order/ownership: (A) use root-reviewed2b `d76ef05034d078b459500f83dc9271aabe88e157`
with refreshed source hashes; (B) independently freeze full
synthetic goldens and caps, assign narrow verified-view/snapshot accessors and
closed publication extensions; (C) implement/review lock codec/store/read/verify
with pure/synthetic tests; (D) separately review exact registered-case projection
and executable-role/context support; (E) implement receipt codec/protocol with
private synthetic gate harness, while actual checked constructor stays absent;
(F) later D7 owns real checked-gate adapter, HE start/checkpoint hook and full
matrix state retention. A lock-only package is complete only for its named
scope, not a claim that full D6c3 production admission is available.

No production full reference, target, admission, HE run, benchmark, network,
download, account or user-hold release is authorized. Positive approved Linux
ext4/XFS full-reference→lock→admission→start→HE evidence remains a later explicit
execution gate. Local source/codec tests cannot establish it. The proof is only
local per-profile lock-before-start for cooperating reviewed processes; neither
hashes nor clocks prove that no outside results were inspected.

## Publication record

Root ratified this exact FIX1 contract after independent design review closed
I1 (initial/retry eligibility) and I2 (source/store/context ownership and fork
checks). Original and revised ledger artifacts remain preserved. Reviewed
proposal SHA256: `2df9354b3d7dc7a9f9bdabe3043945daaf19e2edc60d8b9c73fd65569177b284`.
FIX1 report: `f56de863488131ae950989f17328b6dd70f23ab859bb9d93f096c66a2cb5c404`.
Scoped re-review: `c685cfe1a541e1efdee1902446fec5989e0ba6ac2d534f2c05848d01e845498f`.
This publication changes only the candidate/status wording before appending
this record; the normative field tables, bounds and behavior are unchanged.
Independent complete synthetic goldens and maximal outer-field accounting,
separately reviewed source packages, genuine D7 capabilities and actual Linux
execution remain the stated future gates. This publication performs no input
migration and grants no scientific execution or full encrypted-block release.

## 10. Full wire-codec implementation status

The [bounded source and runtime package](../results/validation/task54-d7-admission-codec/README.md) implements strict Full admission/start inert codecs, shared CaseIdentity/ExecutionBinding traversal and pure supplied-record consistency helpers. Existing78 oracle files and C/R/A/Full bytes remain unchanged. Independent source and outcome reviews passed; nine new exact tests plus four touched Reference/Lock/actual-case regressions passed once. This is not a live reader, store/publication protocol or admission/start capability.

Required integration boundary C1: validate BOTH current and predecessor admissions against the same retained target/reference and approved policy before combining `check_predecessor_claims`. Checking only current-target plus that helper is insufficient; its stated contract does not compare prior target/policy links itself. Add changed prior lock/reference/policy rejection coverage in the live Retry reader. `check_start_claims` likewise needs composed target/reference/current-context verification. Parsed consistency and recovered records never grant live execution authority. Actual gates/leases/custody, durable start, before-allocation checks and full scientific runs remain required.

## 11. Coherent independent Full fixture for C1 integration

The [linked Full oracle](../results/validation/task54-d7-full-linked-oracle/README.md) is accepted as synthetic test data after independent SPEC/QUALITYPASS01329d80. It joins the previously separate evidence and Full admission/start families under one coherent identity, with Initial and both Retry predecessor shapes. Three fully rehashed prior-lock/reference/policy mutations retain valid current-target and predecessor links while failing prior-same-target checks; whitespace, wrong-payload link and stale-integrity negatives distinguish other stages. Original schemas and215 existing fixtures are unchanged.

This supplies the independent input prerequisite for composed C1 tests, not executed production acceptance. Selected snapshot boundaries must be respected. Genuine canonical inputs, applicability reports, current source/authority, physical model, retained owners and protocol lifecycle remain absent from these fictional fixtures. Later source tests must exercise actual current+prior same-target/policy composition, rather than only synthetic-domain rejection. No live gate or scientific authorization follows.
