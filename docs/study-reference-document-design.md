# Task54 D6c1b — reference artifact schema and parser design

Revision: **v3 — scoped F1/F2 timing-wire and outer-variant consistency fixes; opaque identity/context verification unchanged.**

Status: **INDEPENDENTLY REVIEWED AND ROOT-RATIFIED DESIGN, 2026-11-06; implementation and execution separately gated.** Prepared against reviewed D6c1a identity implementation `6aba8f4`. This publication is a design artifact, not implementation validation, training evidence, source ownership or execution authorization. The frozen proposal, earlier versions and independent review records are retained in the implementation ledger.

## 1. Recommended next package and decisions before code

Implement one small `study_reference` child module for **untrusted artifact document types, canonical envelope encoding and strict bounded parsing**. Reuse complete D6c1a contract/receipt canonical bytes unchanged. Do not add a session, writer, build script, provenance capture, target/lock/admission API, CLI, journal, campaign scheduler or reference execution. Do not refactor historical JSON readers or D6a/D6b arithmetic.

Root accepts lowercase hex transport, direct locked serde plus serde_json raw_value, the provenance **transport** inventory and the semantic-validation placement below after independent review. These decisions are ratified before source work:

1. **Lossless identity transport with deferred semantic verification:** store each complete D6c1a canonical byte payload as lowercase hex plus its existing version/SHA256. This preserves every order row, coefficient, initialized parameter, selector/source and topology byte; it is not digest-only storage. The D6c1b parser validates hex/length/hash and the typed outer envelope but does **not** decode or certify the embedded binary grammar. Later, before any `VerifiedReference` or target source, the approved-context reader must canonically materialize the reference's exact profile/seed, call `derive_identities`, compare **both complete byte arrays exactly** and compare all identity/digest links. It must then check outer completion/endpoints/counts against that trusted materialization's plan and data. No imported blob creates a trusted identity type.
2. **Strict JSON implementation:** explicitly declare already-locked `serde = "1"` as a direct bench dependency (no derive needed), and enable **serde_json's** existing `raw_value` feature for bounded lexical number access. Cargo.lock contains serde1.0.229 and serde_json1.0.154; no new package family, codec or crypto dependency is proposed. Custom bounded visitors reject duplicate decoded keys before map insertion and enforce per-field limits. Reporting F64 fields use document-specific bounded RawValue tokens, JSON-number grammar checks and std `str::parse::<f64>()`, requiring finite and exact stored-bit/count agreement. **Do not enable `float_roundtrip`: Cargo feature unification could change bits in existing captured initialization/rate readers.** Existing input-reader number conversion must remain unchanged. Direct dependency/feature edits still require assigned source authorization and lock/license review. Ordinary serde_json::Value alone loses duplicate-key evidence.
3. **Provenance transport now; capture/read policy later:** implement the exact typed field inventory in §5 in this package. Field parsing uses only explicit syntax, bounds and enum rules defined here; recipe/source/config/environment digest values are bounded untrusted claims, not purportedly verified policies. The parser has no dependency on an unspecified manifest/capture algorithm. D6c2 later specifies supported recipe versions, capture, manifest hashing/coverage and readiness policy. Missing/unavailable required evidence never becomes zeros or a trusted completion capability. An arbitrary metadata map is not an alternative.

**Approved refinement to [reference design §5](study-reference-design.md#5-minimal-reference-and-diagnostic-schemas):** “Readers reconstruct and canonically encode the typed outer artifact envelope, including the full losslessly decoded identity byte strings, and verify envelope/identity digests. An initial parser returns untrusted claims only and need not duplicate the D6c1a identity binary grammar. Before verified-reference eligibility, an approved-context reader must compare the complete embedded shared contract and selected-seed receipt bytes to identities freshly derived from the same canonically materialized profile and seed; it must validate all completion/observation/checkpoint claims against that trusted plan/data, and enforce approved source/build and completion policy. Neither digest equality alone nor successful document parsing establishes inner semantic validity or completed training.”

This refinement changes verification placement, not the full-final requirement. The original §5 wording can be read as requiring immediate inner reconstruction; that is the actual textual incompatibility requiring explicit refinement. Exact-byte reconstruction is strictly stronger than checking a duplicate decoder's self-consistency once the trusted input context exists. Cost: standalone parsing cannot inspect or semantically validate the blob, and scientific verification now requires available canonical source inputs/materialization for the reference seed. Missing inputs fail closed; no offline digest-only shortcut is introduced. Those operational costs must be accepted with the refinement.

These are pre-code refinements, not retroactive changes to the ratified identity baseline. The baseline remains revision `e018bd7401a6a606e306e58decad7af68cc56b84`, SHA256 `6cdca16097d4acbf28c0f9bf0a36cccc1fccfae2fb604b0410926c066b85a035`. No new target policy is proposed: only the same concrete profile's actual, complete, origin-to-final seed0 F32 run may later supply its target. B48/B64/B128 remain separate. Each profile's lock and case receipts remain independent; no global journal/cohort seal is introduced.

## 2. Ownership and trust surface

Proposed files after separate authorization: `src/study_reference/document.rs`, `src/study_reference/document_tests.rs` (or a private test submodule), `tests/study_reference_document.rs`; one narrow `mod document`/re-export edit in `src/study_reference.rs`; Cargo.toml/Cargo.lock only if item2 is explicitly approved. Avoid edits to study_profile/study_inputs/study_plain, public design docs or build files in this package. D6c1a's existing identity constructors and byte layouts remain unchanged. The child module handles lossless hex decoding and opaque byte storage only; no second identity grammar decoder or imported trusted constructor is added.

Proposed public surface:

```text
parse_reference_document(bytes: &[u8]) -> Result<ParsedReferenceDocument, DocumentError>
parse_diagnostic_document(bytes: &[u8]) -> Result<ParsedDiagnosticDocument, DocumentError>
parse_failure_document(bytes: &[u8]) -> Result<ParsedFailureDocument, DocumentError>
```

All fields of these returned document types are private; accessors return immutable borrowed data and explicitly named *claimed* observations. Their documentation states: typed outer syntax, safe global bounds, canonical outer encoding, outer self-consistency and byte-digest checks only; no inner grammar or identity-semantic validity; **untrusted declarations, not proof that training/capture/publication happened**. No caller-supplied parser policy can increase limits or admit synthetic/F64 as a trusted full reference. `DocumentError` is a bounded code/path/offset diagnostic, not a transport failure artifact.

There is no public `CompletedFp32Reference`, `VerifiedReference`, serializer of arbitrary metrics, `from_parts`, `Deserialize` implementation for D6c1a trusted identities, `TryFrom<Parsed...>` into trusted identities, or completion/target constructor in D6c1b. Private encoding routines accept private parsed DTOs to compute integrity; they expose no publication capability. Future session-owned completion remains in a different private construction path. A parsed full document can contain a claimed correct count; only the later approved reader/context verifier can assess its source/build policy, and only the real session can create a new in-process completed result. Recomputed matching hashes are not attestation.

The future writer takes the session-created completed/diagnostic/failure result, copies immutable identity bytes, derives observations itself and uses the same envelope encoder. It will not take a parsed document or a public `PlainEvaluation` as completion evidence. Future readers hash all exact input bytes once; path validation, reading once, no-overwrite and durability remain writer/reader packages. No file I/O API is needed here.

## 3. Representation primitives and identity transport

The following tables give **encoding order**, regardless of JSON member order. Every listed key is required; option-valued keys are present with `null` or a value. Unknown keys, duplicate decoded keys, unsupported tags and missing keys reject. No `serde(default)`. Integer tokens are unsigned decimal JSON integers, with no sign/fraction/exponent; bounded u64 conversion is checked before platform indexing. Schema constants, rather than Rust discriminants, assign tags.

| Primitive | Disk JSON / validation | Canonical representation |
|---|---|---|
| `Text` | UTF8 string; field-specific byte bound | u64LE UTF8 byte length, then bytes |
| `U64` | exact `0..=18446744073709551615` integer token | u64LE |
| `Bool` | true/false only | false0 / true1 byte |
| `Sha256` | exactly64 lowercase ASCII hex chars | length-prefixed64-byte UTF8 string, consistent with D6c1a |
| `GitRevision` | exactly40 lowercase ASCII hex chars (current SHA1 repository policy) | length-prefixed UTF8 |
| `F64` | bounded raw JSON-number token, document-local std f64 parse, finite and exact required IEEE754 bits | u64LE `to_bits()`; never JSON text hashing |
| `Accuracy` | object `correct:U64,test_count:U64,value:F64,bits:Hex16` | correct,test_count,u64 bits; the redundant disk bits validate the same value, not a second independent datum |
| `Option<T>` | required key; null or T | absent0 / present1 byte, then T |
| `Vec<T>` | bounded JSON array | u64LE count, then each T |
| `Bytes` | lowercase even-length hex string | u64LE decoded length, then raw bytes |
| fixed enum | exact specified string | explicit one-byte tag |
| named map | bounded array of exact `{key,value}` objects | sort by UTF8 key, reject duplicates, count + records |

For every Accuracy require `0 <= correct <= test_count`, `0 < test_count <= 2^53`, `correct <= 2^53`; calculate `(correct as f64)/(test_count as f64)` and require exact value/bits equality, including +0. No tolerance and no independently accepted score. Hex16 means the conventional sixteen-digit lowercase hexadecimal spelling of the u64 bit pattern, then encoded as u64LE. All parsed outer floats reject nonfinite values. Inner bytes are opaque: this parser does not inspect their floats, tags or lengths. Later exact comparison to canonically materialized trusted identities rejects any different/noncanonical payload, including nonfinite or malformed imported contents.

`IdentityDocument = {schema:Text,encoding:Text,canonical_bytes:Bytes,sha256:Sha256}`. Outer schema must be the existing contract/receipt version as appropriate; encoding is exactly `hekit-study-reference-encoding-v1`. Hex-decode the **complete** supplied bytes into one bounded buffer and verify their SHA256. Nonempty byte payloads are required, but their internal headers/lengths/flags/UTF8/arrays are intentionally not parsed. A bounded arbitrary byte string with a matching digest can therefore appear in a successfully parsed document: it remains an explicitly untrusted identity claim. Do not use an “identity verified” name or boolean for this result.

The parser returns immutable untrusted byte slices/digests, never `StudyProfileContractV1` or `SeedInputReceiptV1`. It never iterates embedded length prefixes, allocates per-row vectors, regenerates arrays/order from a seed, or interprets arbitrary internal bytes as allocation sizes. The typed envelope encoder includes the exact decoded blob bytes and its declared-and-checked digest; no re-encoding of internal identity fields occurs. All internal binary f32/f64 bits are preserved without decimal conversion.

**Mandatory later approved-context reader obligations, all before VerifiedReference/target eligibility:**

1. Select the reference's requested profile/seed through the approved registry/context, treating outer identifiers as requests only. Resolve and canonically materialize those exact approved inputs with normal input checks; MissingInput remains an error. Do not derive a reference receipt from a different HE seed. No materialization occurs in this D6c1b parser package.
2. Call existing `derive_identities(&trusted_materialized)`; compare the embedded contract's **full byte slice** and receipt's **full byte slice** with the respective trusted canonical bytes. Require equal lengths and exact byte equality, independently of SHA256 comparison. Recompute/check both digests and compare outer `profile_id`, `seed`, `materialized_sha256`, identity versions, shared/receipt links and arithmetic policy to trusted accessors/current supported algorithms. Derive trusted identities exclusively through this constructor; never fill private fields from the file.
3. Use the trusted materialization's entire plan, checkpoint topology, held-out data size and approved arithmetic policy to check every outer full/prefix count, origin, drop total, hash link, endpoint, boundary/interval, final endpoint index and denominator described in §4. No shortened or attacker-described plan may serve as the comparison oracle. For full completion require all planned updates and samples, the exact final checkpoint, final observation equality and actual F32 policy. The source reader must reject any remaining unsupported digest/session algorithm instead of marking these obligations completed.
4. Enforce approved source/build/executable/read policy and the safe session's origin-to-final completion contract under the explicitly declared local-process trust boundary. Byte equality proves input identity, **not execution**. Final-state bytes are not exported, so an offline reader cannot independently replay/inspect the final model merely from its digest. Matching state/prefix claims become interpretable only with frozen session algorithms and approved provenance; successful parsing or matching initialization never proves training occurred.
5. Only a separately reviewed verifier may produce `VerifiedReference`; no parser conversion or metrics injection exists. Only a verified actual complete same-profile seed0 F32 final endpoint may later supply that profile's target. For a subsequent seed7 HE case, verify seed0 reference against seed0 materialization first, and separately derive seed7's own receipt/contract: require shared-contract equality, not selected-receipt equality to seed0. No global readiness journal is introduced.

The preserved inner content remains exactly D6c1a's existing complete encoding: shared policy/profile/data/graph/optimizer/arithmetic/selector/source/schedule/full topology and selected receipt linkage/component hashes/full numeric plan/both rate views/full order/pass offsets/both initialized dtype arrays. Existing enum/byte tags and version constants are **transported unchanged**, not duplicated as a second parser-owned grammar. A future identity-version change requires explicit support/context review; it cannot be silently accepted as an old version merely because the outer digest matches.

**Alternative rejected for this package:** implement a full read-only binary grammar decoder and compare every internal field. It adds a second large schema/semantic implementation parallel to `shared_bytes`/`receipt_bytes`, including evolving source-role/selector/graph/rate/array rules and test obligations; it still cannot establish approved inputs or executed training without the same later context checks. It could provide standalone inspection, but that is a separately justified tool, not necessary for this lossless artifact parser. Exact trusted reconstruction eliminates the duplicate grammar while preserving mandatory final scientific verification.

## 4. Outer field inventory and tags

Three distinct schema domains; no generic arbitrary `Artifact` map. Canonical header is `Text(ENCODING_VERSION),Text(schema_domain)`, followed by fields below in table order. Include all JSON fields semantically, except `integrity.payload_sha256` itself. Integrity encoding/hash version strings are included. File SHA256 is returned alongside the parsed object, never embedded recursively in its own payload. Original bytes need not be retained after exact file digest calculation; the caller still controls its input buffer lifetime.

| Common fields, in order | Exact type / constraint |
|---|---|
| `schema` | constant: `hekit-study-reference-v1`, `hekit-study-diagnostic-v1`, or `hekit-study-reference-failure-v1` |
| `kind` | reference `full_fp32_final`0; diagnostic `diagnostic_quick`1 or `diagnostic_full`2; failure `reference_failure`3 |
| `status` | `complete`0 for full/diagnostic; `failed`1 for failure; complete diagnostic never means scientific full |
| `run_id` | Text<=128 bytes, ASCII `[A-Za-z0-9_-]+`; identity only, no filesystem authority |
| `writer` | `{api_version:Text<=128,algorithm_revision:Text<=128}`; bounded nonempty version claims; later context reader checks exact approved API/session revision |
| `synthetic` | must false in these production schema namespaces; cfg(test) fixture domains are distinct |
| `identity` | `{profile_id:Text<=128,seed:U64 0..9,materialized_sha256:Sha256,contract:IdentityDocument,seed_receipt:IdentityDocument}`; failure uses Option of this object, as failure may precede materialization |
| `arithmetic` | `{dtype:Enum,accumulation:Enum,cast:Enum,reduction:Text<=1024,classification:Text<=1024}`; dtype f32=32/f64=64; accumulation scalar_f32=0/scalar_f64=1; cast f64_to_f32_once=0/canonical_f64=1; nonempty bounded reduction/classification claims; later context reader requires exact supported policy strings for the declared dtype |
| variant body | reference/diagnostic/failure body below |
| `timing` | reference/diagnostic: `Timing` with no option byte; every failure phase: `Option<Timing>` with absent0/present1 byte, null allowed in any phase; rules below |
| `provenance` | Provenance transport below (failed-before-capture may carry unavailable branch) |
| `integrity` | `{encoding:Text exact ENCODING_VERSION,hash:Text exact "sha256",payload_sha256:Sha256}` |

Reference body, in exact order:

```text
completion: {
  origin_completed_updates: U64 = 0,
  origin_processed_samples: U64 = 0,
  planned_updates: U64,
  actual_updates: U64,
  planned_samples: U64,
  actual_samples: U64,
  planned_dropped_samples: U64,
  actual_dropped_samples: U64,
  full_plan_sha256: Sha256,
  consumed_plan_sha256: Sha256,
  final_endpoint_index: U64,
  final_state_sha256: Sha256
},
endpoints: Vec<Endpoint>,
final_observation: Accuracy
```

Parser-level full self-consistency requires f32/scalar_f32/f64_to_f32_once, equal supplied full/consumed hash claims, actual/planned count equality, nonempty ordered endpoints, final index `len-1`, last endpoint counts equal outer actual totals, and final observation bit-identical to that endpoint. These are comparisons among untrusted claims only. **Later context verification additionally requires** all totals/hashes and the entire endpoint sequence to equal the canonically materialized full plan, and every denominator to equal trusted held-out rows; parser self-consistency is never completion. The selected seed may be0..9; **the parser never labels seed0 target-ready**, and a full seed7 document never supplies a prospective target. No public numeric target field exists.

`Endpoint` field order: `completed_updates:U64,processed_samples:U64,crossed_logical_boundaries:Vec<U64>,exact_logical_boundary:Bool,interval_updates:U64,interval_samples:U64,observation:Accuracy,update_elapsed_ns:Observation<U64>,evaluation_elapsed_ns:Observation<U64>`. For **reference/diagnostic endpoint sequences**, start the previous cumulative counts at origin0/0: each endpoint has strictly larger update/sample counts, positive intervals equal the corresponding cumulative differences, and nondecreasing available cumulative durations. Do not apply that sequence-from-origin rule to a failure's standalone last endpoint; its different rules are frozen below. Durations are cumulative to that checkpoint and have explicit unavailable reasons when absent. The later context reader requires exact equality to actual trusted checkpoint topology, including crossed boundaries, flags and intervals. No interpolation; origin has no invented accuracy/checkpoint. B128 retains actual38/37-update and4864/4736-sample intervals.

Diagnostic body, in exact order:

```text
target_eligible: false,
scope: quick_two_updates=0 | full_diagnostic=1,
progress: {
  origin_completed_updates: 0, origin_processed_samples: 0,
  planned_updates: U64, consumed_updates: U64,
  planned_samples: U64, consumed_samples: U64,
  planned_dropped_samples: U64,
  full_plan_sha256: Sha256, consumed_plan_sha256: Sha256
},
endpoints: Vec<Endpoint>,
last_observation: Accuracy,
last_state_sha256: Sha256
```

`diagnostic_quick` must have scope0, consumed_updates exactly2 and consumed counts no greater than planned counts. Its endpoint sequence may be empty. For a nonempty sequence apply the sequence-from-origin update/sample/interval rules above, and require **each** endpoint's updates and samples <= the respective consumed totals. If the last endpoint equals **both** consumed totals, its entire Accuracy (`correct`, `test_count`, value and bits) must equal `last_observation`; otherwise `last_observation` is a separate terminal evaluation and need not equal an earlier endpoint. No full/consumed plan-hash equality or inequality is required for quick parsing; both are opaque supplied SHA256 claims here, to be checked against frozen full/prefix algorithms and the trusted plan later.

`diagnostic_full` must have scope1 and consumed counts equal planned counts; its endpoint sequence must be nonempty and satisfy the sequence-from-origin rules. The last endpoint's update/sample counts must equal both consumed/planned totals, its entire Accuracy must equal `last_observation`, and the supplied full/consumed plan hashes must be equal. These are outer self-consistency checks, never evidence that the full plan was executed. The kind/scope pairs are exact: `diagnostic_quick`/scope0 and `diagnostic_full`/scope1 only.

Both diagnostic kinds accept only the arithmetic triples in the matrix below and remain permanently target-ineligible. The intended producer includes only crossed full-plan checkpoints in endpoints (later context-checked); a quick terminal evaluation before the next checkpoint belongs in `last_observation`, not a fabricated epoch endpoint. Later context verification still requires exact counts/checkpoints/denominators/full or prefix identity against the canonically materialized trusted plan. Prefix/state hashes remain opaque claims until their exact session algorithms are frozen and separately tested before production output/verification; do not reserve a conflicting prefix domain while requiring equality to the existing full-plan digest.

Failure body, in exact order:

```text
target_eligible: false,
requested_scope: full_fp32=0 | quick=1 | full_f64_diagnostic=2,
phase: resolve=0 | materialize=1 | start=2 | update=3 |
       checkpoint=4 | finalize=5 | publish=6,
error: {code:Text<=128, message:Text<=4096},
last_successful_endpoint: Option<Endpoint>,
progress: Option<{completed_updates:U64,processed_samples:U64}>
```

No `final_observation`, final-success accuracy, target or success completion object is legal in failure schema. Last endpoint's historical observation is permitted because it is explicitly not the final successful reference. Progress may exceed the last evaluated checkpoint if an update succeeded before evaluation failed. Missing failure record remains possible after a kill. Failure message sanitization and filesystem handling belong to the later producer, not a parser inference.

A failure's `last_successful_endpoint`, when present, **requires both identity and progress to be present**. Its completed updates and processed samples must each be <= the corresponding supplied progress count. Validate its interval_updates and interval_samples as positive and <= its own respective cumulative counts; **do not require interval counts to equal cumulative counts**, because prior endpoints are not stored. For example cumulative12 updates with interval6 is valid outer structure when progress.completed_updates>=12. Membership in the trusted checkpoint sequence and the actual interval delta remain later context checks. With no endpoint, identity and progress retain their declared independent Option types; their absence is never filled with zero and phase alone does not imply their presence. Exact phase/capture/progress truth remains a later producer/context obligation.

### Parser-only arithmetic and scope matrix

Exactly two triples are legal: **F32** = `(f32, scalar_f32, f64_to_f32_once)` and **F64** = `(f64, scalar_f64, canonical_f64)`. Reject every mixed tuple even when each individual enum value is known. These aliases add no wire fields/tags; they constrain the existing three fields.

| Document kind / scope | Allowed triple | Outer scope relation |
|---|---|---|
| `full_fp32_final` | F32 only | full reference rules above |
| `diagnostic_quick` | F32 or F64 | scope `quick_two_updates`0 only |
| `diagnostic_full` | F32 or F64 | scope `full_diagnostic`1 only |
| `reference_failure`, requested_scope `full_fp32`0 | F32 only | all failure phases use the same Option field types |
| `reference_failure`, requested_scope `quick`1 | F32 or F64 | all failure phases use the same Option field types |
| `reference_failure`, requested_scope `full_f64_diagnostic`2 | F64 only | all failure phases use the same Option field types |

Reduction/classification text remains a bounded transport claim with exact supported-policy checking later; this matrix does not claim that any arithmetic was executed. Scientific completion, actual plan coverage, boundary flags, held-out denominator truth and provenance checks remain in the mandatory trusted-context verifier unchanged.


## 5. Timing/provenance transport, not capture authorization

**Fixed timing wire types:** reference and diagnostic `timing` is always `Timing`, encoded directly with **no option byte**. Failure `timing` is always `Option<Timing>` in **every phase** (resolve0 through publish6), encoded with absent0 or present1 followed by Timing. The key is mandatory in every schema. A failure's null timing is a syntactically valid unavailable-timing claim in **any** phase, including start, finalize and publish; presence/null is never inferred from phase. This parser rule does not relax later producer/capture obligations. A late failure with present timing still emits the present1 byte; it never switches to the direct reference/diagnostic layout.

With null failure timing, validate each stored endpoint duration's own Observation/U64 shape but perform **no enclosing-total duration comparison** and do not substitute zero. With present failure timing, apply the normal endpoint-to-total/end_to_end duration bounds below. A standalone failure endpoint supplies no prior endpoint against which to check duration monotonicity. Available timing captured by the later producer and actual capture/phase truth remain its responsibility.

`Observation<T>` is exactly either `{status:"available",value:T}` (tag0) or `{status:"unavailable",reason:Text<=256}` (tag1). No extra keys in either branch. It avoids ambiguous null/zero optional observations. `Timing` order:

```text
wall_source: unix_system_time=0,
start_unix_ns: U64,
end_unix_ns: U64,
monotonic_source: std_instant=0,
scope: after_materialization=0 | includes_materialization=1,
end_to_end_ns: U64,
setup_ns: Observation<U64>,
update_ns: Observation<U64>,
evaluation_ns: Observation<U64>,
finalization_ns: Observation<U64>
```

Require wall start<=end (validity does not prove historical ordering). Outer cumulative duration claims must be monotonic; available disjoint components must not sum beyond enclosing end-to-end. Evaluate/update are disjoint, not added again to enclosing totals. The parser recognizes the includes-materialization tag as a claim; a later producer/verifier may use it only if the binary actually establishes that start boundary. When Timing is present (always for reference/diagnostic), endpoint cumulative duration claims must be <= the corresponding available total and <= end_to_end_ns; the context/session verifier checks actual timing scope. Integer wall timestamps use u64 nanoseconds for the current Unix-era policy; unsupported clock/range rejects instead of wrapping. No GPU/resource metric is synthesized.

Provenance is exactly `{status:"captured",value:ProvenanceV1}` tag0 or `{status:"unavailable",reason:Text<=256}` tag1. The unavailable branch is parseable accountability, always **non-ready** for later production reference verification/locking. No `verified:true` field is accepted. `ProvenanceV1` is the root-accepted transport inventory for this proposal, in order; its typed syntax is implementable now without capture/manifest policy:

| Field | Type and fields in order |
|---|---|
| `schema` | exact `hekit-study-provenance-v1` |
| `build` | `{compiler_vv:Text,target:Text,host:Text,profile:Text,opt_level:Text,debug:Bool,features:Vec<Text>,package_version:Text,cargo_lock_sha256:Sha256,source_manifest_sha256:Sha256,config_manifest_sha256:Sha256,build_environment_sha256:Sha256,recipe_version:Text,receipt_file_sha256:Sha256}` |
| `source` | `{root:RawOsString,commit:Observation<GitRevision>,tracked:GitState,untracked:GitState,manifest:Vec<SourceFile>}` |
| `executable` | `{path:RawOsString,start_sha256:Sha256,end_sha256:Sha256}` |
| `invocation` | `{argv:Vec<RawOsString>,cwd:RawOsString}` |
| `host` | `{cpu_model:Observation<Text>,os:Observation<Text>,hostname:Observation<Text>,available_cpus:Observation<U64>,process_id:Observation<U64>,observed_threads:Observation<U64>,execution_threads:U64 = 1}` |
| `thread_environment` | Vec of `{key:Text,value:Observation<Text>}`, fixed transport key set listed below, sorted unique UTF8 key |
| `recheck` | `{source_manifest_sha256:Sha256,build_receipt_file_sha256:Sha256}` |

`thread_environment` transport keys are exactly `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `BLIS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`; no arbitrary environment key is accepted. This is a syntax allowlist, not a claim that any variable was captured, applied, or sufficient for build/runtime reproducibility. D6c2 decides which of these keys must be present and how they are captured; a new key requires explicit schema review. The aggregate array cap below remains conservative; effective unique-key maximum is seven.

`GitState`: clean0/dirty1/unavailable2, encoded as enum strings; unavailable must block required-provenance readiness. `SourceFile`: `{path:Text,sha256:Sha256}`, sorted unique repository-relative path. Features sorted unique UTF8. `RawOsString`: `{encoding:"unix_bytes" (tag0),bytes:Bytes}` gives one lossless representation for UTF8 and non-UTF8 Unix argv/path bytes; no joined argv, lossy replacement or two competing representations. Non-Unix capture is unsupported until explicitly designed. Source root/executable/cwd are runtime absolute byte strings and never enter the portable D6c1a scientific identities.

This inventory deliberately excludes arbitrary environment values, generic metadata maps, GPU timing and a new provenance framework. Full build receipt source/config coverage, exact supported invocation/environment policy, ignored-file handling, recipe identity and checksum relation are **D6c2's unresolved responsibilities**, not established by these fields. Recipe/version texts are bounded nonempty claims; manifest/config/environment digests are syntactically valid opaque SHA256 values until D6c2 freezes their algorithms. A manifest list is parsed as sorted unique bounded relative-path/hash records, with path syntax defined as nonempty slash-separated UTF8 components excluding empty, `.` and `..`, leading `/`, backslash and NUL; it is not hashed against an invented manifest algorithm here. For parsed full/complete diagnostic captured provenance, require supplied executable start/end SHA256 equality, build.source_manifest_sha256 == recheck.source_manifest_sha256, and build.receipt_file_sha256 == recheck.build_receipt_file_sha256 as outer self-consistency only. Failure records may retain differing hashes as evidence of the failure. These comparisons do not inspect filesystem truth; even matching start/end hashes do not attest the loaded process. Optional load/RSS is omitted initially, as allowed by the ratified design. Adding it later requires explicit version/schema review, never unknown-field acceptance. No source/build verification or trusted `RuntimeProvenance` constructor is implemented here.

## 6. Source-derived size accounting for every registered profile

These are arithmetic bounds from the pinned registry, `study_profile::graph/build_plan`, `study_inputs::order/initial`, and D6c1a encoder; **not measured artifact sizes or runtime/RSS claims**. Only small source/config files were read. No Fashion/MNIST feature arrays, full order realization or trained run was loaded/generated. Missing block seeds3..9 remain MissingInput; their registered shapes/counts are retained for sizing and are not fabricated receipts.

Definitions: `R` retained order rows; `U` updates; `E` epochs/checkpoints (all current profiles cross one boundary per checkpoint); `P = sum(input*output + output)` over **all** layers, including local heads and explicitly stored zero disabled biases. Each initialized scalar appears as8-byte F64 plus4-byte F32. Fashion eMLP3 has39,184 such scalars, not only inference-stack parameters. Per-pass offsets number E+1.

A shared plan update is25 bytes without a source schedule index and33 with one. A numeric receipt plan update adds24 bytes. Its two rate vectors add32+16=48 bytes per update. One checkpoint is49 bytes with its single boundary. Thus, with `S=1` for captured MNIST schedule and0 otherwise:

```text
C_bound = (25 + 8*S)*U + 49*E + M
R_bound = (97 + 8*S)*U + 8*R + 57*E + 8 + 12*P + M
J_bound = 2*(C_bound + R_bound) + A
M = 1,048,576 bytes per identity of non-array metadata/framing budget
A = 2,097,152 bytes of outer JSON, whitespace, observations and provenance budget
```

Here C_bound/R_bound include complete identities; receipt57E includes49E checkpoint bytes plus8E pass offsets, and the extra8 is the last pass offset. All remaining vector headers, version strings, layer names/shapes and plan's fixed32 bytes are charged to M. M is explicit metadata/framing **headroom in the enforced total byte-cap formula**, not a measured size. An opaque parser cannot separately inspect or enforce internal metadata consumption of M; it enforces only C/R total lengths. Trusted-context exact-byte comparison and metadata-only sizing inventory later establish the actual complete receipt behind that cap. Current code has at most6 layers/3 stages,20 global registry source definitions (per profile fewer), selected initialization<=6 plus order1, expected dataset files<=4 and a small fixed set of added manifests/configs/code paths. For the source-derived bound audit, allowance of source lists<=64, internal strings<=4096 and cumulative identity metadata<=M comfortably covers those known source shapes; these are sizing assumptions checked against trusted resolver metadata, **not internal parser checks**; registry strings inspected here max493 UTF8 bytes. Before implementation freeze, the all21-profile metadata-only test must inventory actual source counts/string lengths against these bounds, without materializing features. Unknown future profile/metadata expansion fails for review, not silent truncation.

For JSON only hex identity strings use the large-string exception; hex ASCII has exactly2x expansion with no writer escaping. The parser enforces A as **all raw file bytes other than the two raw identity string contents**, including excess whitespace/key escaping. Large hex strings containing JSON escape sequences are rejected (the scientific bytes have an unambiguous lowercase hex transport), so a malicious escaped string cannot evade J_bound. Reformatting within A changes exact file SHA256, not canonical payload SHA256. Other bounded strings may be validly escaped and charge their actual byte cost to A.

| Profile | R | U | E | P | C_bound bytes | R_bound bytes | J_bound bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| `iris__mlp2020` | 48,000 | 2,400 | 400 | 83 | 1,128,176 | 1,689,180 | 7,731,864 |
| `iris__emlp1` | 48,000 | 2,400 | 400 | 292 | 1,128,176 | 1,691,688 | 7,736,880 |
| `iris__emlp2` | 48,000 | 2,400 | 400 | 2,792 | 1,128,176 | 1,721,688 | 7,796,880 |
| `iris__emlp3` | 48,000 | 2,400 | 400 | 11,884 | 1,128,176 | 1,830,792 | 8,015,088 |
| `breast_cancer__mlp2020` | 44,800 | 2,800 | 100 | 332 | 1,123,476 | 1,688,268 | 7,720,640 |
| `breast_cancer__emlp1` | 44,800 | 2,800 | 100 | 1,058 | 1,123,476 | 1,696,980 | 7,738,064 |
| `breast_cancer__emlp2` | 44,800 | 2,800 | 100 | 4,260 | 1,123,476 | 1,735,404 | 7,814,912 |
| `breast_cancer__emlp3` | 44,800 | 2,800 | 100 | 14,758 | 1,123,476 | 1,861,380 | 8,066,864 |
| `mnist_released__mlp2020` | 48,000 | 1,000 | 10 | 2,080 | 1,082,066 | 1,563,114 | 7,387,512 |
| `mnist_released__emlp1` | 48,000 | 1,000 | 10 | 6,832 | 1,082,066 | 1,620,138 | 7,501,560 |
| `mnist_released__emlp2` | 48,000 | 1,000 | 10 | 16,256 | 1,082,066 | 1,733,226 | 7,727,736 |
| `mnist_released__emlp3` | 48,000 | 1,000 | 10 | 39,184 | 1,082,066 | 2,008,362 | 8,278,008 |
| `fashion_mnist__mlp2020` | 3,000,000 | 62,500 | 50 | 2,080 | 2,613,526 | 31,138,894 | 69,601,992 |
| `fashion_mnist__emlp1` | 3,000,000 | 62,500 | 50 | 6,832 | 2,613,526 | 31,195,918 | 69,716,040 |
| `fashion_mnist__emlp2` | 3,000,000 | 62,500 | 50 | 16,256 | 2,613,526 | 31,309,006 | 69,942,216 |
| `fashion_mnist__emlp3` | 3,000,000 | 62,500 | 50 | 39,184 | 2,613,526 | 31,584,142 | 70,492,488 |
| `mnist_released__mlp2020__b64` | 48,000 | 750 | 10 | 2,080 | 1,073,816 | 1,536,864 | 7,318,512 |
| `mnist_released__emlp1__b64` | 48,000 | 750 | 10 | 6,832 | 1,073,816 | 1,593,888 | 7,432,560 |
| `mnist_released__emlp2__b64` | 48,000 | 750 | 10 | 16,256 | 1,073,816 | 1,706,976 | 7,658,736 |
| `mnist_released__emlp3__b64` | 48,000 | 750 | 10 | 39,184 | 1,073,816 | 1,982,112 | 8,209,008 |
| `mnist_released__emlp1__b128_candidate` | 48,000 | 375 | 10 | 6,832 | 1,061,441 | 1,554,513 | 7,329,060 |

The maximal admitted envelope is70,492,488 bytes (about67.23MiB); 64MiB is67,108,864 bytes and therefore excludes valid documents within these proposed budgets. This is **not** a measurement or claim that today's normally serialized Fashion artifact necessarily exceeds64MiB: its actual metadata/provenance do not yet exist. No arbitrary128MiB allowance is needed either. Use global pre-parse caps `C_max=2,613,526`, `R_max=31,584,142`, `J_max=70,492,488` bytes for every document with identity, irrespective of its claimed profile. Both blob caps are checked independently plus total J/A. Only **after approved-context canonical matching** may the reader enforce the matched trusted profile's C/R/J limits and all semantic counts. Quick diagnostics retain complete receipts and use the same global pre-parse caps; failures with no identity use A. A claimed small/unknown profile must not alter allocation authority or grant trust; an unknown profile cannot pass later context matching. No successful profile is dropped to meet a parser limit.

Fashion order alone is24,000,000 binary bytes (u64LE rows) or48,000,000 hex characters. Receipt numeric updates plus both rate views are6,062,500 bytes (97*62,500), shared topology1,562,500 bytes, and eMLP3 F64/F32 initialized scalars470,208 bytes. These are fixed-layout arithmetic, not data-dependent float text estimates. BC uses44,800 retained rows and700 dropped samples, not45,500 stored rows. Full Iris has2,400 updates/400 checkpoints, so its checkpoint count exceeds Fashion's50 even though Fashion is the largest receipt. All ten seeds use the same size ceilings, including missing registered block seeds.

### Alternatives and working-storage limits

- Full named JSON is human-readable but repeats all identity schema decisions and numeric conversion obligations. Even rows alone could be up to18,000,001 compact JSON bytes for Fashion (five digits plus comma each); its per-update objects and dual floats add substantial data. A Vec<Value> per3,000,000 rows is an avoidable allocation. Bounds would need a separate exact JSON field/digit formula before approving this alternative.
- Byte arrays `[0,...,255]` preserve exact bytes but worst-case grow to4x bytes plus framing: only the complete Fashion order portion can approach96MB. Reject as the default representation.
- Base64 would reduce lossless blob expansion to4*ceil(n/3), but no directly declared base64 dependency exists here. Adding one or writing a bespoke base64 codec is unnecessary for this focused package. Compression introduces codec/version/decompression-bound issues; external sidecars add atomic multi-file publication/linkage. Neither is needed.
- Digest-only, order seeds instead of full rows, omitted F64/F32 views, generated-on-read inputs and shortening a registered plan are not storage alternatives: they violate the complete approved receipt.

The preferred parser keeps the caller's input buffer and hex-decodes each opaque identity once into a globally bounded byte buffer. It does not simultaneously retain a serde_json::Value tree, copied giant hex strings, per-row typed vectors and duplicate re-encoded arrays. No identity grammar walk or re-encoding occurs; later trusted reconstruction compares exact complete byte slices. The canonical envelope may be one bounded Vec for the current SHA helper, with raw identity bytes appended once. Enforce an independent encoded-envelope ceiling `C_max + R_max + A = 36,294,820` bytes using checked bounded appends; do not assume canonical encoding is smaller than arbitrary JSON spelling. The same 2MiB A is separately proposed for canonical outer framing/fields. Independent size tests must show the registered outer field shapes fit this budget; failure is an error, not truncation.

The existing `data::sha256_hex` clones its input and appends SHA padding; Vec growth can temporarily approximately double that clone's capacity. Do not call it “streaming” or assume constant memory. Calculate file SHA first, release its scratch, then decode; hash identity buffers sequentially; retain no identity re-encoding scratch; proceed to envelope hash. With maximal input J≈70.5MB, identity bytes C+R≈34.2MB, canonical-envelope body bounded by(C+R)+A≈36.3MB and SHA working capacity approximately twice that body, the planned major simultaneous buffers are roughly214MB plus small parser/allocator overhead. This is a source-derived allocation estimate, **not measured peak RSS or a hard allocator guarantee**. The future session/writer additionally retains materialized data/state/identities and serialization output; this proposal does not claim that total fits any execution gate. Avoid changing the shared SHA helper in this package merely to optimize that estimate.

The writer's ratified serialize-before-publish requirement remains feasible at these explicit bounded sizes but still needs later resource checks. No full-size synthetic artifact need be allocated in ordinary unit tests: check size formula/counter overflow and per-field refusal paths with bounded fixtures. A later explicitly scoped full-size serialization-only stress test can use test-only invalid/synthetic domains and must not publish a scientific success file or claim trained evidence.

## 7. Strict parser behavior and failure bounds

Processing order:

1. Refuse input longer than global J before parsing. Compute exact file SHA256 here and release its scratch before identity decoding. Validate UTF8 and a single JSON document with trailing whitespace only; reject BOM, comments, trailing commas and trailing second document. Bound depth to16, key UTF8 length to64, ordinary text to4096, ordinary raw string token length to24,578 (six-byte escaping plus quotes), and numeric token length to64. Special raw identity strings may be up to2*C_max or2*R_max with no escapes, independently of claimed profile. Build `compiler_vv` may be up to16,384 decoded bytes with matching raw escaped-token budget. Lossless OS bytes use a separate<=4096 decoded-byte hex bound.
2. Use a small structural/token preflight to enforce token/depth/raw-string budgets before an escaped string can grow serde's scratch buffer. It recognizes JSON token boundaries with correct backslash/quote handling; serde_json remains the syntax/unicode authority. On the ordinary-string rule's only exceptions, require the exact observed object path (`identity.contract.canonical_bytes` / `identity.seed_receipt.canonical_bytes` or the explicitly enumerated compiler/OS fields). Decode bounded object keys using serde_json string rules to track the exact path, including escaped spellings; do not recognize a giant string merely by finding a matching key substring. Duplicate decoded keys reject in the visitor, including `"seed"` versus `"\u0073eed"`. Preflight is not a second full semantic parser.
3. Exact-field custom visitors enforce required keys and reject duplicates/unknowns at every object, including Observation branches. Explicit seen-bitsets avoid last-key-wins. Reject duplicates even when values agree. Reject over-limit arrays before collecting next element. Accept JSON member reordering; sort only the named semantic map fields specified above, rejecting duplicate keys. Never sort ordered scientific vectors to hide inconsistency.
4. Enforce only outer collection bounds: endpoints<=400, crossed boundaries<=400 and total boundary entries<=400; provenance manifest<=512, argv<=256, features<=64, threading environment<=32 plus the fixed seven-key set. Available thread counts must be positive. Text limits and aggregate A apply regardless of individual list limits. No Vec allocation trusts unchecked attacker lengths. Do not inspect embedded identity counts/shapes/rates/source arrays or allocate from them. Claimed profile/seed stays bounded text/integer metadata; use global blob/file caps in all field orders. Per-profile counts/length caps belong to the later approved-context verifier.
5. Validate outer kind/dtype tags, full/quick/failure self-consistency, count/interval conservation, integer domain, finite exact reporting bits, timing and supplied hash equality where explicitly required. Hex-decode the entire bounded blobs and check SHA256; no inner-header, grammar, source-role or semantic validity claim. Failure progress/endpoint must be outer-self-consistent; later context verification checks any identity-bound limits. Use checked arithmetic for buffer sizing/outer counters and no native usize hashing or unchecked u64-to-usize cast.
6. Deterministically encode the typed envelope in the stated field order. Verify payload SHA256. Return the already-computed exact-file SHA256 over all original bytes including whitespace; never accept file digest in place of payload digest. Return only untrusted parsed documents and both digests. No filesystem/resource/training operation occurs.

`DocumentError` proposed stable codes (Rust variants, not a new persisted schema): `TooLarge`, `BadJson`, `DepthLimit`, `TokenLimit`, `DuplicateField`, `UnknownField`, `MissingField`, `UnsupportedSchema`, `UnsupportedTag`, `OutOfRange`, `NonFinite`, `BadHex`, `DigestMismatch`, `CrossLinkMismatch`, `OuterProgressMismatch`, `ObservationMismatch`, `TimingMismatch`. Include a bounded logical field path<=256 and optional byte offset; do not echo entire input/argv/source/environment values. The failure artifact's producer-owned error code remains a separate namespace and cannot be used to switch the parser into success.

## 8. Independent validation package, no completion shortcuts

The later implementation brief should freeze tests before implementation. Required evidence:

1. Independent small byte goldens constructed from the table, using a separate Python struct/hashlib oracle during test authoring: exact header/domain, field order, UTF8 lengths, enum/option flags, integer limits, float bits (including signed zero), array lengths and payload SHA. Commit literal bytes/digests in tests; do not use production encoder output as the expected oracle. Check each schema domain and its rejection by the other parser entry points.
2. Strict JSON adversarial cases: duplicate every nested object kind; same-value duplicates; escaped aliases; unknown/missing fields; exponent/fraction/negative integers; u64 overflow; 2^53 count boundary; oversized depth/key/text/array; malformed escapes/surrogates/hex; leading-zero forms; trailing second document. A matching recomputed hash cannot rescue unsupported schema/enum, wrong dtype/cast/accumulation tags or invalid outer fields. Arbitrary bounded reduction/classification strings remain policy claims for the later context checker; the parser does not silently endorse them. Document-local float parsing must include binary64 values known to differ at decimal rounding boundaries and require exact expected bits, while captured initialization/rate golden bits remain unchanged.
3. Identity **transport** cases: malformed/odd/uppercase/escaped hex, empty blob, over-global-cap blob, changed bytes with stale digest, and identity/envelope digest confusion reject. Conversely, bounded arbitrary blob bytes with recomputed identity/envelope digests can parse as untrusted claims; explicitly test this result cannot create a trusted identity or VerifiedReference. Do not accidentally promise embedded grammar validation. Required **later reader** negative tests mutate every identity region (including internal headers/tags/nonfinite scalars/trailing bytes/selector/source/topology), recompute all file hashes and require exact trusted-byte comparison to reject. Those semantic tests are deferred-reader obligations, not D6c1b coverage claims.
4. Metadata-only all21-profile bound table oracle: exact R/U/E/P and all stated C/R/J values, BC dropping700, all four Fashion3,000,000/62,500 profiles, all B64 counts and B128375/alternating38/37. Derive expected constants independently from registry arithmetic and graph widths, not from `estimate_size()` itself. Inventory current source record/text limits from resolver metadata without feature loads. Exercise outer counter/hex-length checked-overflow and boundary-limit helpers without large allocations. Never inspect attacker-supplied binary headers to allocate. Test that a small-profile claim cannot reduce/raise the parser's fixed global cap or bypass later context matching; valid complete Fashion blobs remain admitted by size. No unavailable captured seed is materialized to satisfy this test.
5. Bounded actual canonical Iris seed0 and7 identity derivation/transport round-trip (all four graphs): preserve exact D6c1a bytes/digests and different selected receipts under equal shared contracts. This is **input identity evidence only**, no full-session claim. No cursor injection, shortened full plan, or mutable metadata-to-trusted constructor.
6. Private fixture domains, e.g. `hekit-test-reference-document-v1`, may carry tiny synthetic observations/identities to exercise positive encoders under `cfg(test)`. They remain untrusted and are rejected by production schema entry points. Production schema bytes claiming completion may be adversarial parse fixtures only: no success artifact path, writer or completed capability, and no statement that a successful parse proves completion. No checked-in/generated fixture should masquerade as an actual full reference.
7. Separate checks by layer. **Parser tests:** missing required outer keys, repeated/out-of-order endpoints that violate outer ordering/conservation, inconsistent actual/planned/last counts, final observation mismatch, complete/quick/F64 tag swaps, forbidden failure final-observation/target fields and diagnostic true target eligibility reject. Reordered JSON/whitespace preserves payload but changes file SHA; payload/identity mutation without recomputed matching hashes rejects. **Required later context-reader tests:** a self-consistent shortened plan/final-minus-one/extra checkpoint, coherent wrong denominator, wrong B128 exact-boundary or38/37/4864/4736 sequence, cross-profile/seed receipt swap and coherent forged plan hash all fail against the canonical full materialization, even after recomputing every transport hash. Missing canonical reference inputs blocks verification. A real seed7 receipt is matched to seed7 while a target reference is independently matched to seed0; no selected-receipt equality requirement across seeds. Prefix/state digests remain unverified claims until their algorithms are frozen; no parser test can mark them semantically verified.
8. **F1 literal wire goldens:** freeze independent canonical byte fixtures for failure timing null (exact option byte00) and present (exact option byte01 then the specified Timing fields), including a late `publish` phase6 null and present case. Verify the same phase does not change the timing wire type; reference/diagnostic Timing has no option byte. Missing timing key rejects, null failure timing is accepted in every phase, null reference/diagnostic timing rejects, and a late failure cannot omit the present byte. With an available endpoint duration and null failure timing, do not fabricate a zero total or reject solely because an enclosing total is absent; with present timing, an endpoint duration above the available/enclosing total rejects. These are untrusted schema fixtures, not captured timing evidence.
9. **F2 arithmetic/variant cases:** exercise both allowed triples, every mixed triple, and each allowed/forbidden kind/requested-scope pairing in the matrix. Quick diagnostics accept empty endpoints and a prior endpoint whose Accuracy differs from a separate terminal evaluation; reject any endpoint exceeding consumed updates or samples, incorrect origin-based interval deltas, or a terminal endpoint matching both consumed totals whose entire Accuracy differs from last_observation. Full diagnostics reject empty endpoints, final counts not equal to consumed/planned totals, differing last Accuracy and unequal supplied full/consumed hashes; reject quick/full kind-scope swaps. Quick parser tests must not require unequal hashes or terminal/earlier observation equality when no terminal checkpoint exists.
10. **F2 standalone failure cases:** endpoint-present with missing identity or missing progress rejects; endpoint updates/samples above supplied progress rejects; zero interval or interval greater than corresponding cumulative count rejects. A singleton cumulative12/interval6 update endpoint (with valid positive sample inequalities and sufficient progress) is accepted as an untrusted historical claim; do not require origin-to-singleton conservation. An absent Option remains absent, never zero-filled. Later checkpoint membership/actual interval/phase truth tests remain deferred-context obligations. Independent golden fixtures cover endpoint present/absent option tags and demonstrate failure singleton layout without changing the required Timing Option tag.
11. API/privacy checks: external callers cannot construct trusted contracts/receipts/completed results from parsed data or public metrics. Use existing project-compatible compile-fail doctests if suitable; no new trybuild dependency solely for a negative-constructor test. Review module visibility explicitly. No public function accepts a metric map for publication.

Run only focused unit/integration tests and appropriate Clippy/format checks after source authorization; retain bounded-resource environment used by D6c1a. Do not run historical CLI/HE integration binaries, keygen, full scientific training, downloads or large datasets. Report parser/golden tests separately from real bounded Iris identity evidence. This design has run none of those implementation tests.

## 9. Unresolved boundaries and handoff gate

Root has ratified this full schema/parser design and the explicit §5 verification-placement refinement after independent review and scoped F1/F2 rereview. Hex transport, direct locked serde plus serde_json raw_value, provenance transport and stated size limits are approved design choices; source assignment remains separate. Capture/readiness policies remain later and do not block implementing typed untrusted transport. Future profiles/fields require size-bound rederivation, not a parser allow-bigger flag. The ledger preserves v1, reviewed v2, v3, exact diffs, fix reports and manifests. This publication changes only status/context/handoff language; normative schema and validation obligations remain the reviewed v3.

The consumed-prefix plan and final F32 parameter/velocity state algorithms are not supplied by D6c1a. Their exact domain/field order and session provenance binding must be frozen and independently golden-tested before any production session output or verified-reference eligibility. The full-plan digest is existing D6a `plan_sha256`; that encoding is not simply a receipt plan byte span (its own domain and `crossed` string are included). Parser full/consumed equality is only equality of supplied claims. The later context reader compares full hash against `trusted_materialized.identity().plan_sha256()` and uses approved full/prefix algorithms; never equate different-domain hashes. Until those algorithms are defined, the parser treats prefix/state SHA256 values as opaque claims and no verifier may mark the obligations satisfied. A final state digest is integrity metadata supplied by the future safe session; without exported final arrays it does not let an offline reader independently reconstruct the trained state. Writer API/algorithm revision claims are bounded text in the parser and exact approved revision requirements in that later verifier.

The later reader is obligated to compare both complete shared and selected byte payloads to trusted `derive_identities`, then check all completion/checkpoint/count claims against that same trusted full plan, before any VerifiedReference/target. Failure to load/reconstruct any required input is a blocking verification error, never a reason to accept blob hashes alone. Reference-seed canonical materialization may be resource-intensive and needs its own later execution/resource scope; this design performs none. Exact-byte comparison avoids a second grammar but does not avoid actual complete scientific verification or the local-process trust limitation.

Timing boundary/cumulative observation production, terminal failure ownership, actual F32 sequential finish checks, source/build receipt semantics, source/executable recheck policy, publication path/fsync/no-overwrite behavior, exact read policy, target derivation, per-profile locks and admission/start capabilities are still unimplemented gates. A schema-valid local forgery is possible and is not repaired by canonical encoding. Only actual full seed0 F32 final endpoints, generated later under separate execution authorization and approved source/build policy, can supply new prospective targets; no synthetic fixture, best epoch, quick prefix, F64 diagnostic or historical FULL output substitutes.

This proposal is intentionally one artifact/schema package. No provenance redesign, global event journal, new target algorithm, supplemental registry, shortened Fashion/Iris run, or expansion of authorized execution is requested.

## Publication record

This design publishes reviewed proposal v3 SHA256 `7ec4766f1fc6eed6bf2b480011522875e995de73229d8492045507d11e7dc09a` after independent review and F1/F2 closure. Scoped rereview SHA256: `66dc4236c709bc7a5d4d567132c7a835a0ccbc19787402d8d1cf130190fc71e2`. Source implementation and execution were not performed by this design package. The pinned D6c1a scientific policy baseline remains the explicit earlier revision/digest recorded in §1; this later document is not substituted for that historical hash.
