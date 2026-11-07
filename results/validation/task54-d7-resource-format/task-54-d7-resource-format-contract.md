# Task54 D7 resource source-format contract addendum

Status: **PROPOSED FORMAT CLOSURE FIX1; scoped re-review and root ratification pending.** Supplements immutable resource proposal e72b99a7; resolves normalization R1–R7 without changing its architecture, lifecycle or model obligations. This is an inner-source representation contract, not source/encoder/corpus/runtime authority. Existing outer Resource/FileRef/Full/Quick bytes and caps remain unchanged. Six JSON source records plus the already proposed binary bootstrap; no seventh JSON record.

## 1. Common notation and JSON lift (R1)

Every object below has exactly the listed keys, including keys whose value is null. Reject missing/unknown keys, duplicate decoded keys (including escaped aliases), invalid UTF8/Unicode scalars, malformed JSON and trailing non-whitespace. JSON member order is immaterial; table order is a writer/document convention only. Exact original whole-file bytes are retained and hashed; no canonicalized-JSON digest. Legal JSON string escaping is accepted except for Bytes, whose transport spelling is deliberately unescaped hexadecimal. No BOM or implicit defaults. Synthetic/private forms never enter production acceptance.

| Type | Exact representation and range |
|---|---|
| U64 | JSON number token matching `0|[1-9][0-9]*`, value 0..18446744073709551615. Reject quoted numbers, booleans, fractions, exponent notation, plus sign and negative zero. |
| I32 | JSON integer token `0|-?[1-9][0-9]*`, value −2147483648..2147483647; also reject −0, fractions and exponents. |
| tag(S) | U64 lexical rule, restricted to the finite set S in the field table; binary tag remains one byte. |
| Bool | JSON true/false only; binary 00/01. |
| Text128 | JSON string whose decoded bytes match `[A-Za-z0-9_-]{1,128}`. |
| Text4096 | JSON string of at most4096 decoded UTF8 bytes. Field semantics may require nonempty; no normalization. |
| Hash / Git40 | JSON string of exactly64 /40 lowercase hex characters after JSON decoding. Hash denotes the specifically identified bytes/opaque profile identity below, never Debug or a guessed substitute. |
| UUID | JSON string matching lowercase `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}`; no extra version/variant restriction. |
| Bytes | JSON string token containing only literal lowercase hex digits, even count, no backslash escapes. Decode pairs to raw bytes; empty allowed unless the underlying record forbids it. Embedded registration/request keeps its existing schema/cap. |
| Option<T> | JSON null for None; the direct T value for Some. There is no `{some:...}`, tag wrapper or omitted-key alternative; no nested Option type occurs here. |
| List<T> | JSON array. Default count0..65536; source-file lists≤512, argv≤256 and each decoded argv item≤4096 bytes/no NUL. Below lists may impose exact/nonempty counts. Preflight decoded lengths/counts/aggregate checked U64 arithmetic and conversions before allocation. |

Each raw JSON source≤8MiB and any stricter existing typed cap. These transport ceilings grant no allocation budget. Existing C/R/A/K and DatasetInput bounds are untouched. Text/Hash/Bytes/List binary primitives are unchanged: U64 length/count followed by payload, Hash encoded as Text64, tag/Bool one byte; I32 is LE two's complement. JSON has no binary header fields in addition to schema/synthetic.

`schema` is the exact production domain below with `synthetic=false`. Private oracle changes only its leading `hekit-study-` to `hekit-test-` and uses true. Production rejects either test domain or true and any mixed-domain embedded record. Existing legacy nested registered-case bytes retain their frozen grammar; no invented synthetic variant.

## 2. Complete source-record fields (R2/R3)

Repeated names below are typed independently; this table is exhaustive, not an extensible metadata schema. `FR`, `Clock`, `Environment`, `Requirements` and other named records are defined in §3. Domain strings are Text128 literals; fixed strings are not tag numbers.

| JSON record/domain | Exact fields in documentary order |
|---|---|
| Model / `hekit-study-resource-model-v1` | `schema:domain, synthetic:Bool, revision:Text128, source_manifest:Hash, source_revision:Git40, source_files:List<FR>, registry:Hash, build_environment:FR, rule_revision:Text128="d7-cpu-allocation-v1", supported_work:List<SupportedWork>, components:List<ModelComponent>[22], bootstrap_derivation:FR` |
| Validation / `hekit-study-resource-validation-v1` | `schema:domain, synthetic:Bool, model:FR, source_manifest:Hash, source_revision:Git40, environment:Environment, coverage:List<Coverage>[22], runs:List<Run>` |
| HostPolicy / `hekit-study-resource-host-policy-v1` | `schema:domain, synthetic:Bool, boot_id:UUID, physical_host:Hash, host_namespace_origin:NamespaceOrigin, allocator_environment:Allocator, kernel_environment:Kernel, filesystem_environment:List<FilesystemProfile>, claim_domain:Text128="task54-host-serial-v1", model:FR, validation:FR` |
| ClaimRecord / `hekit-study-resource-claim-v1` | `schema:domain, synthetic:Bool, claim_id:Text128, claim_domain:Text128="task54-host-serial-v1", physical_host:Hash, boot_id:UUID, model:FR, validation:FR, work:ClaimWork, approval:Hash, source_manifest:Hash, source_revision:Git40, created:Clock, modeled_envelope:Requirements` |
| HostObservation / `hekit-study-resource-host-observation-v1` | `schema:domain, synthetic:Bool, claim_id:Text128, sequence:U64, physical:Hash, started:Clock, finished:Clock, topology_before:Topology, topology_after:Topology, meminfo_before:FR, meminfo_after:FR, controllers:List<ControllerRaw>, structural_root:StructuralRoot` |
| DiskObservation / `hekit-study-resource-disk-observation-v1` | `schema:domain, synthetic:Bool, claim_id:Text128, sequence:U64, started:Clock, finished:Clock, filesystems:List<DiskFilesystem>` |

Identity meanings and links:

- Model.revision is a retained model-package revision identifier, not a digest, source commit, gate-policy revision or executable rule. `rule_revision` plus the exact numeric rule selects the compiled closed rule. Unknown revision/rule is unsupported, never fallback.
- `source_manifest` is the whole-file SHA256 of the SAME existing complete approved compile-source manifest (`VerifiedStudyContext::source_manifest_sha256`); `source_revision` is its actual approved Git40 implementation revision. Model.source_files is the relevant exact subset of that manifest, not a second authority. Validation source identity must equal its Model's; Claim source identity is the actual current context and must meet the existing model applicability/equivalence policy. No new commit wildcard or unconditional equality bypass.
- **Model.registry is SHA256 of the exact approved `crates/ckks-train-bench/configs/experiment_matrix.json` bytes.** At fixed BASE it is `164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc`, exposed as APPROVED_REGISTRY_SHA256 and RegisteredCase.registry_sha256(). It is neither a registered-case digest nor the separate input-lock hash. RegisteredCase.digest() hashes its complete frozen registered-case canonical bytes; SupportedWork.registration_id uses that digest. Existing inner CaseIdentity.registration_id remains its Text128 registration label, a different field.
- Validation.model, HostPolicy.model and ClaimRecord.model are the SAME exact FR, Reviewed root2, fixed path `resource/linux-cpu-v1/model.json`; policy/claim validation is the SAME Reviewed FR at `resource/linux-cpu-v1/validation.json`. These are whole-file length/hash links. Package selection follows existing trusted policy, not self-authorship.
- Model.build_environment is a FR to existing retained reviewed build/environment evidence, not an inline Environment or an implicitly new JSON-domain file. Model/bootstrap and component derivations likewise name retained reviewed sources. Their actual accepted source formats/applicability remain evidence obligations; this contract does not invent their content.
- Claim approval is actual current approval hash. Claim IDs match the existing outer Claim and observation directory; observations match that claim, host/physical descriptor and sequence. `physical_host` and observation `physical` have their existing distinct roles: host identity versus actual physical-execution descriptor. Do not equate merely because both are Hash.

## 3. Complete nested records

| Record | Exact fields / constraints |
|---|---|
| FR (existing FileRef) | `root:tag(0..5), path:Text4096, length:U64, sha256:Hash`; roots Input0/Source1/Reviewed2/Policy3/Authority4/Observation5. Path nonempty relative UTF8; reject absolute prefix, empty/dot/dot-dot components, NUL or backslash. No normalization. FR is an identity checked against known expected inventory, never permission to open an artifact-selected path. |
| Clock (existing) | `wall_source:tag{0}, unix_ns:U64, monotonic_source:tag{0}, elapsed_since_operation_start_ns:U64`. Same operation origin and nondecreasing elapsed for paired started/finished; existing live clock obligations unchanged. |
| ModelComponent | `kind:tag(0..21), rule:tag(0..21), derivation:FR`; rule equals kind under the fixed rule_revision. |
| Coverage | `kind:tag(0..21), source_derivation:FR, validation_runs:List<Text128>`; IDs resolve as §6. Derivation must identify the same relevant rule source proof as ModelComponent.derivation; this may be the same FR or a separately retained independent review of it, never an unverified assertion of coverage. |
| Run | `id:Text128, command:FR, stdout:FR, stderr:FR, metadata:RunMetadata, covered_components:List<tag(0..21)>`. **metadata is inline**, not a FR/string or extra JSON record. Command and both streams name retained exact artifacts, including empty streams with their real length/hash. |
| Environment | `target:Text128, toolchain:FR, cargo_lock:FR, build_receipt:FR, allocator:Allocator, kernel:Kernel, workers:U64, bootstrap:BootstrapCoverage`. Workers is the actual recipe; no format-wide worker count or default. |
| Allocator | `implementation:tag{0}, library_files:List<FR>, configuration:FR, derivation:FR`; tag0 system-glibc only. Actual loaded identities/configuration and their reviewed proof required. |
| Kernel | `release:Text4096, version:Text4096, page_size:U64, source:FR, configuration:FR`; release/version nonempty exact observed uname values; actual page_size>0. Representation does not prove supported kernel/profile. |
| BootstrapCoverage | `source_derivation:FR, validation_runs:List<Text128>`; finite IDs as §6, no recursive Run objects. |
| RunMetadata | `source_manifest:Hash, source_revision:Git40, executable:Hash, environment:Environment, argv:List<Bytes>, exit_code:Option<I32>, timed_out:Bool, started:Clock, finished:Clock, measurements:List<Measurement>`. executable hashes actual executable bytes. Source identity equals containing Validation. argv retains argv0/order/empty and nonUTF8 arguments; nonempty list required, no NUL. No inferred success: exit_code null means no recorded exit code, not zero; timed_out records actual outcome independently. Failed/timeout runs may be retained; acceptance depends on actual required validation, not parsing. |
| Measurement | `kind:tag{0,1,2}, scope:tag{0,1,2}, bytes:U64, raw:FR`; kinds allocator-owned peak0/process-RSS peak1/cgroup-current peak2; scopes self0/self-plus-controlled-children1/current-cgroup2. Missing metric is omitted from list, never emitted as guessed zero. No source formula or RSS equivalence follows from a number. |
| NamespaceIdentity | `device:U64, inode:U64`. |
| NamespaceOrigin | `boot_id:UUID, mount_ns:NamespaceIdentity, cgroup_ns:NamespaceIdentity, pid_ns:NamespaceIdentity, user_ns:NamespaceIdentity, installation_record:FR`. Last field is independently trusted direct-host provenance, not mandatory UID0 installation. |
| Topology | `membership:FR, mountinfo:FR, controllers:FR, namespaces:NamespaceOrigin`. namespaces is the **full NamespaceOrigin**, not an unspecified projection. Controllers source is actual global-root cgroup.controllers for v2, proc/cgroups only in applicable v1/hybrid mapping. |
| FilesystemIdentity | `device:U64, mount_id:U64, type:U64`; type is the statfs filesystem magic represented as U64, not filesystem-name text. Device/mount identity follows the same actual held descriptors; no extra inode field. |
| StructuralRoot | `version:tag{0,1}, domain_path:Text4096, mount_identity:FilesystemIdentity, disposition:tag{0,1}, controller_raw:List<FR>`. Version v1=0/v2=1; disposition no-controller0/global-root-no-local-interface1. domain_path is the actual nonempty cgroup domain path, may be absolute including `/`; it is not FR.path and is never directly opened from an artifact. Disposition1 requires proved v2 global structural root. Unsupported v1/hybrid remains missing actual observer support. |
| ControllerRaw | `version:tag{0,1}, path:Text4096, limit_raw:FR, current_raw:FR`; nonempty actual cgroup path; version/path/raw identities match the corresponding existing outer Cgroup row. Unlimited is interpreted only from real raw data; no omitted current or invented root row. |
| FilesystemProfile | `identity:FilesystemIdentity, roots:List<tag(0..5)>, mount_evidence:FR, quota_evidence:FR, derivation:FR`; roots are exactly existing FR root roles, not a new filesystem-root enum. |
| DiskFilesystem | `identity:FilesystemIdentity, roots:List<tag(0..5)>, frsize:U64, bavail:U64, favail:U64, required_bytes:U64, required_files:U64, quota_evidence:FR`; frsize>0 and products/sums checked. Scope/identity/root coverage matches actual retained sources/destinations and applicable policy profile. Actual unavailable syscall/quota facts remain missing, not fabricated max values. |
| Requirements (existing) | `components:List<Component>[22], peak_host_bytes:U64, incremental_host_bytes:U64, required_disk_bytes:U64`. |
| Component (existing) | `kind:tag(0..21), phase_mask:tag(0..15), resident_bytes:U64, additional_bytes:U64`; mask0 permits only both byte fields0 plus independently source-proved inapplicability; phases1 prepare/2 key-setup/4 update/8 endpoint. Exactly existing checked four phase maxima and residency rules; no unknown-as-zero. |

The exact rule identity is the pair `(d7-cpu-allocation-v1, rule number)` below. Names explain existing component semantics; they are not extra encoded strings or formulas.

| kind=rule | Existing component meaning | kind=rule | Existing component meaning |
|---:|---|---:|---|
|0|main generator|11|interstage handoffs|
|1|main secret/public|12|bootstrap workspace|
|2|training relin|13|worker workspace|
|3|training rotations|14|source/materialized input|
|4|separate bootstrap bundle|15|target/reference/input receipt buffers|
|5|Encoder tables|16|plan/preparation metadata|
|6|client/decryptor|17|diagnostics|
|7|compact state|18|evidence/event/publication buffers|
|8|expanded state|19|allocator/alignment overhead|
|9|NAG branches/old state|20|runtime/thread stacks|
|10|micro packets|21|exhaustively identified other source-owned allocations|

The table does not supply physical rule implementations, numeric totals, model completeness or support for any route.

## 4. Work branches, options and exact matching (R7)

| Record/branch | Exact keys and types |
|---|---|
| SupportedWork Train0 | `kind:tag{0}, registration_id:Hash, consumption:tag{0,1}`; Full0/Quick1. |
| SupportedWork Reference1 | `kind:tag{1}, operation:tag{0,1,2}, profile_id:Text128, seeds:List<U64>`; read0/recover1/snapshot2; seeds exactly `[0]`. |
| SupportedWork Target2 | `kind:tag{2}, operation:tag{0,1}, profile_id:Text128, seeds:List<U64>`; read0/lock1; read has nonempty explicit seeds, lock exactly `[0]`. |
| ClaimWork Train0 | `kind:tag{0}, registration:Bytes, request:Bytes`; complete frozen registered-case canonical bytes and existing Request wrapper, not their digests. |
| ClaimWork Reference1 | `kind:tag{1}, operation:tag{0,1,2}, reference_id:Text128, expected_file:Option<Hash>, profile_id:Text128, seed:U64, document:FR`; seed0. expected_file Some iff recovery1, equal actual expected whole-file hash and acquired document.sha256; read/snapshot null. |
| ClaimWork Target2 | `kind:tag{2}, operation:tag{0,1}, campaign_id:Text128, profile_id:Text128, selected_seed:U64, target:Option<FR>, reference:FR, supplied_input:Option<Hash>`; target Some iff read0. Lock1 requires selected_seed0, target null and supplied_input null. Read supplied_input Some iff the input already exists under SAME live owner at selection; its value is exactly that retained supplied `MaterializedProfile.identity().digest()` (existing D6a materialized identity). Otherwise null and its future construction is included in the envelope. Never hash an unbuilt input to authorize it. |

**FIX1 supplied-input identity/equality:** derive Some only from the actual retained supplied materialized owner. When its selected `SeedInputReceiptV1` is derived, require its Some value to equal both `retained_supplied.identity().digest()` and `supplied_receipt.materialized_digest()`. The value is **not** `SeedInputReceiptV1.digest()` (hash of the complete selected receipt canonical bytes), nor `StudyProfileContractV1.digest()` (hash of the shared contract canonical bytes). Keep the existing independent actual/selected reconstruction and complete `actual_receipt.canonical_bytes() == supplied_receipt.canonical_bytes()` check; matching materialized hashes alone never replaces it. None stays None when construction is future, with its work charged before allocation; the immutable claim is not rewritten afterward. Target lock retains null and seed0. Future oracle expectations must distinguish the selected D6a identity from receipt digest, contract digest and a one-byte wrong digest, while preserving all current option/operation controls. No new hash, field, preimage or outer wire is introduced.

For Train, decode/check the unchanged registration/request bodies, derive registration SHA256 and actual consumption, then match exactly one `(0,digest,consumption)` entry. Registry hash and actual selected membership must agree with Model.registry; Quick uses its registered seed0 provenance without Full promotion. For Reference/Target match kind+operation+profile and actual seed membership in its single seeds list. Profiles come from the existing registry's21 profile IDs; no new registration is created. Seed scalar is U64 in this contract; applicable existing registry/operation decides membership, not a new global seed ceiling. At fixed BASE current concrete cases and resolver use0..9; Reference/lock specifically remain0. Historical seed2020 and unresolved supplements are not silently promoted by permitting a U64 value.

Nested reconstruction must match the enclosing selected source call graph/envelope; it retains one outer ClaimRecord and owner, without making a second ClaimWork or lock. Preparation-only branches never emit outer Resource or finalize HE. Structural bytes cannot prove live ownership, input-currentness or reconstruction.

## 5. All list ordering and uniqueness (R7)

Comparisons are numeric for tags/U64 and lexicographic on decoded UTF8 bytes for strings, with shorter prefix first. No locale or normalized spelling. FR key means `(root,path decoded UTF8)`; identical key with conflicting length/hash is never a second identity.

| List | Required order / uniqueness / size beyond common caps |
|---|---|
| Model.source_files | Nonempty, ≤512, ascending unique FR key; exact relevant approved source subset. |
| Model.supported_work | Nonempty; ascending lexicographic key Train `(kind,registration_id,consumption)`, Reference/Target `(kind,operation,profile_id,seeds)` with numeric lexicographic seed vectors. Unique Train key; at most one Reference/Target entry per `(kind,operation,profile_id)` even if seeds differ. Merge explicit supported seeds into one sorted list; this avoids ambiguous overlapping selectors. |
| Every work seeds list | Nonempty, strictly ascending numeric U64; operation constraints §4. |
| Model.components / Validation.coverage / Requirements.components | Exactly22 entries ordered kind0..21, no duplicates/gaps. |
| Validation.runs | Ascending unique id Text128. May retain unsuccessful runs; an empty list is representable but cannot by itself satisfy required native validation. |
| Coverage.validation_runs / Environment.bootstrap.validation_runs | Ascending unique run IDs, all resolve within containing Validation (§6). Empty is representable; missing required proof remains missing. Source-proved inapplicability does not waive applicable obligations elsewhere. |
| Run.covered_components | Ascending unique kind tags0..21; may be empty for bootstrap-only work. |
| Allocator.library_files | Actual declared loaded-library evidence order, unique FR keys,≤512. Do not sort away load/evidence order. Actual applicable inventory must be complete/nonempty as required by supported allocator proof. |
| StructuralRoot.controller_raw | Declared evidence/acquisition order, unique FR keys. Exact needed controller/root evidence required; empty cannot assert absent controllers. |
| HostObservation.controllers | Exact applicable accounting leaf→root order, unique `(version,path)`. Preserve finite and unlimited ancestors; structural root is separately represented. Empty only with the ratified actual root/no-controller proof. |
| HostPolicy.filesystem_environment / DiskObservation.filesystems | Ascending unique `(device,mount_id,type)` identity tuple; actual applicable root/destination coverage required. No duplicate identity with a second roots list. |
| Filesystem roots lists | Nonempty strictly ascending unique tags0..5. A filesystem combines its covered roles; no claim that a role split across real mounts can be omitted. Exact inventory decides needed profiles/records. |
| RunMetadata.argv | Actual argument sequence, not sorted/deduplicated; nonempty≤256 and each decoded arg≤4096/no NUL. |
| RunMetadata.measurements | Actual reported measurement sequence, not sorted. Repeated kinds/scopes permitted; raw evidence identifies scope/phase. Repetition does not create additional validation coverage. Empty is permitted to represent missing metrics, not success. |

All unspecified list occurrences are the named lists above. Reordering JSON object keys is not a list-order violation. No new whole-matrix execution, count truncation or headroom threshold follows from these representation rules.

## 6. Finite reference graph and environment projections (R4/R6)

HostPolicy.allocator_environment is exactly Environment.allocator (full Allocator object); kernel_environment is exactly Environment.kernel (full Kernel object). Topology.namespaces is exactly full NamespaceOrigin and must agree with the actual trusted origin/policy. No undocumented partial keys. Compare decoded field values, with ordered lists; this is not JSON canonicalization or a new hash preimage.

Each Run.metadata is an inline RunMetadata. Its environment equals the containing Validation.environment in all fields, including bootstrap IDs; its source identity equals containing Validation. Every validation_runs string in top coverage, top Environment.bootstrap or repeated RunMetadata.environment.bootstrap resolves **only in this same Validation.runs array**, once by exact unique id. No cross-file/global registry, implicit latest run or transitive recursive expansion. Run IDs are labels, not hashes of a run or of the Validation file. Each Coverage kind's referenced run must list that kind in covered_components; conversely a run's listed component must reference its id in that component's Coverage. A run with empty covered_components is linked through bootstrap coverage; unrelated unreferenced runs reject. Bootstrap references are additional coverage associations, not a fabricated component22. An ID may appear in several coverage lists when the actual run covers those obligations.

This creates a finite graph: Validation contains Environment and Run objects; repeated Environment contains strings only, never runs. A bootstrap run can name its own id in its repeated environment without recursively containing itself. Decode/index once within existing caps, then compare IDs/fields; do not recursively chase coverage IDs. Applicable command/outcome/measurement proof still decides whether any association is meaningful.

Whole-file hash direction is acyclic: pre-existing reviewed source/build/derivation/origin/raw-run artifacts → Model → Validation → HostPolicy → bootstrap anchor; selected Model/Validation and actual context/work → ClaimRecord → observations/outer Resource. Arrow means the record at right may refer to exact artifacts at left. Model build_environment/derivation sources cannot acquire a hash dependency back to the containing Model/Validation/policy/anchor. Run command/streams/raw files do not embed a required hash of their future containing Validation. Their IDs/actual environment facts may precede report assembly. No self-hash; no FR to a containing record in place of an ID. Existing independently trusted expected anchor identity is established after its assembly, outside its own bytes. R6 was an unresolved reference type, not a proven preexisting hash cycle.

`kernel_profile`, `allocator_profile`, `filesystem_profile` in bootstrap are opaque Hash-valued exact compiled-supported-profile selectors. They are **not** SHA256(JSON projection), concatenated fields or profile enum tags. This addendum defines no new preimage. The concrete supported implementation must recognize an exact selector and validate its actual corresponding Kernel/Allocator/filesystem policy/held facts; unknown selector remains unsupported. Synthetic literal identities may exercise exact matching/mismatch only, never establish a real supported profile or physical bound.

## 7. Bootstrap bytes and duplicated facts (R5)

The already proposed binary domain is `hekit-study-resource-bootstrap-v1`, test `hekit-test-resource-bootstrap-v1`. No JSON wrapper. Exact order and production size:

| Ordered field | Existing primitive | Bytes |
|---|---|---:|
| encoding header literal `hekit-study-reference-encoding-v1` | Text:8+33 |41|
| domain production / test | Text:8+33 /8+32 |41 /40|
| synthetic | Bool, false production /true test |1|
| generation | U64 |8|
| origin_evidence, policy | two Hash, each Text64 |144|
| boot_uuid | Bytes with count36, exact ASCII UUID spelling from §1 |44|
| account_uid | U64 |8|
| host_key | Hash |72|
| namespaces | exactly mount,cgroup,pid,user; each device U64 then inode U64; **no List prefix** |64|
| proc_root | device,inode,mount_id,fs_type, each U64; no object framing |32|
| cgroup_root | same four U64 fields |32|
| kernel_profile, allocator_profile, filesystem_profile | three Hash |216|
| claim_domain | tag0 |1|

Sum `41+41+1+8+144+44+8+72+64+32+32+216+1 = 704` production; test domain subtracts1→703. Existing Header/Hash/Bytes length prefixes are included; no extra array lengths or LF. Fixed maximum704 plus one overflow byte is a parser scratch sizing fact, not a validated physical budget. Require exact length, domains/synthetic pairing and EOF.

| Bootstrap fact | Exact later check; no new HostPolicy keys |
|---|---|
| policy | SHA256 of exact whole HostPolicy JSON bytes. Policy has no anchor-hash backlink. |
| boot_uuid | ASCII bytes equal HostPolicy.boot_id and host_namespace_origin.boot_id. |
| namespaces | Four device/inode pairs equal corresponding HostPolicy.host_namespace_origin fields; actual descriptor checks remain mandatory. |
| origin_evidence | Equal HostPolicy.host_namespace_origin.installation_record.sha256. That FR's exact origin record is already independently expected by the trusted launcher; it is not self-authored host proof. This selects the SAME provenance artifact, not an additional origin service. |
| claim_domain tag0 | Maps exactly to policy/claim string `task54-host-serial-v1`. |
| generation, account_uid, host_key, proc_root, cgroup_root | Anchor-only facts verified against trusted expected anchor/current launcher/held descriptors under the ratified bootstrap. No corresponding HostPolicy keys are added. generation has no invented minimum/latest registry; actual expected anchor decides currentness. host_key is the stable physical-host lock selector; it is not required equal physical_host. |
| three profile selectors | Exact supported dispatch and corresponding policy validation described in §6; no invented JSON hash equality. |

The independently trusted origin capture precedes anchor/policy assembly and cannot depend on their future hashes. The separate trusted expected anchor identity can then select the assembled anchor. Anchor-only facts were not missing HostPolicy fields (R5 clarification). All host-origin/bootstrap/model/native evidence remains genuinely required; fictional matching bytes discharge none of it.

## 8. Source basis and scope

Fixed source BASE `3a65f42dac83a17112c84dc3f23d3eb0aae7d0a8` (retained source d1aa3af): `study_profile.rs:25–33,600–617,670–675` defines exact registry/input-lock identities, loads checked registry and resolves current seeds; `study_profile/registered_case.rs:130–158,212–246` separates registry hash/canonical-case digest and preserves full/Quick registration sets; `study_reference/document/admission.rs:58–99` supplies the existing canonical CaseIdentity body; `study_provenance/context.rs:816–819` returns approved compile-source manifest identity. Full paths/hashes are in the closure report. Source was inspected read-only; no registry/project constructor was executed.

FIX1 accessor evidence at the SAME fixed BASE: `study_inputs.rs:285–289,388–390` exposes the combined D6a identity through the materialized owner; `study_reference.rs:95–103,114–133` distinguishes contract/receipt/materialized getters; `derive_identities:737–758` copies `p.identity().digest()` into the receipt materialized field; `study_reference/target.rs:364–369` retains the independent actual/supplied complete canonical-receipt comparison. Full source pins and the original contract/report snapshots are in the appended FIX1 closure report and separate fix pins.

R1/R2/R3/R4/R7 are concrete representation choices; R5/R6 clarify anchor-only fields and finite IDs, not established missing-field/cycle defects. Reasons/alternatives/costs are in the paired closure report. Proposal, reviews, maps and normalization report remain byte-identical. No outer wire/schema/cap/scientific threshold/model authority change; all21/full2680+20+7/quick268 and original holds remain. Independent literals and encoders wait for scoped review/ratification. Real typed wiring, actual physical model/host observations/helper lifetime and GPU/external obligations remain separate unfinished work.
