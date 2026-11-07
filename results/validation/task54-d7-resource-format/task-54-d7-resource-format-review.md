# Task54 D7 resource format closure — independent scoped review

SPEC: **REQUEST CHANGES**. QUALITY: **REQUEST CHANGES**.

One medium, bounded hash-role clarification remains before the independent oracle can freeze all meanings. R1/R2/R4/R5/R6/R7 are closed within this representation review; R3 is otherwise complete but remains open for `ClaimWork.Target.supplied_input`. No broad architecture issue or change to the ratified resource design is requested. No vectors, encoders, implementation, model validation or runtime are authorized by this review.

## Reviewed inputs and boundary

Read `task-54-d7-resource-format-closure-brief.md` and the independent normalization report first, then the complete contract and closure report. Used engineering:code-review's correctness/maintainability criteria, limited to the new inner formats and introduced inconsistencies. Independently checked these exact inputs:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `task-54-d7-resource-format-contract.md` | 29012 | `7f9130f56d629247ec08fe0ed0739e984168caae0eaecdfe77e51f2898318044` |
| `task-54-d7-resource-format-closure-report.md` | 9328 | `5298ef3a3203f1b140683ca3bc481c75179aeb4cf90bb3fd3d7ae665f23775d2` |
| `task-54-d7-resource-oracle-report.md` | — | `6c8bb195139b379d54fe6d0ad999d962fdf5e2e5c00a938fd440bebb7e25a402` |
| Immutable resource proposal | — | `e72b99a7a599460ea3f2c206a5ecaa002bfa3ad9b62925f02da9d5d07f63604a` |

Source reads were committed blobs at BASE `3a65f42dac83a17112c84dc3f23d3eb0aae7d0a8` only. Inspected the exact registry constants/load/seed rule, RegisteredCase identity getters and full/Quick membership, canonical CaseIdentity construction, approved source-manifest getter, and the target/materialized/receipt identity accessors needed for the finding below. No moving writer source, later documentation commit, project constructor or helper was executed. Existing gate-schema primitive and Request definitions were read as authority.

## 1. Medium — Target supplied_input lacks an exact hash identity/equality

**Location:** format contract line 102, read together with lines 17, 40–47 and 104–106. Immutable producer proposal line 76 likewise gives the Some/None lifetime predicate but does not choose this digest's meaning.

`supplied_input:Option<Hash>` is Some only when an input already exists under the SAME live owner, and None when construction is future. These predicates are precise. The remaining missing choice is **which existing identity its Some value contains** and the corresponding equality check. It is not enough to call it an input hash: this source has several different input-related digests.

At fixed BASE:

- `study_inputs.rs:285–289,388–390` exposes `seed_profile.identity().digest()`, the existing combined versioned input/content/source identity.
- `study_reference.rs:95–103` exposes a shared profile-contract digest.
- `study_reference.rs:114–133` exposes the selected-seed receipt digest and its distinct `materialized_digest()`.
- `study_reference/target.rs:364–369` independently derives actual and supplied selected receipts and compares their complete canonical bytes.

An oracle author cannot tell from the current field definition whether to put the materialized identity, receipt digest or contract digest in this field, or which equality distinguishes a positive fixture from a wrong-identity fixture. These are different frozen meanings even though each fits Hash. Same-owner/existence checks and the separately mandatory full receipt comparison do not choose the serialized meaning. This is a representation gap, not evidence of a failed live gate or a request to change the architecture.

**Bounded remedy:** state that Some is exactly the already-retained supplied `MaterializedProfile.identity().digest()` (the existing D6a materialized identity), derived from that actual owner; when its selected receipt is derived, require equality with `SeedInputReceiptV1.materialized_digest()`. Explicitly distinguish it from `SeedInputReceiptV1.digest()` and `StudyProfileContractV1.digest()`. Retain the current Some/None and lock predicates, no-future-hash rule, same-owner requirement and independent complete actual/supplied receipt comparison. This reuses an existing digest; do not add a new field, hash algorithm, JSON preimage or outer wire change. If another existing identity is deliberately selected instead, name that exact accessor/bytes and equality with the same precision before oracle generation.

Update only the inner-format contract and its closure report, preserving proposal e72b99a7. The later oracle should distinguish the selected identity from the other existing hashes and a one-byte wrong value, in addition to the existing option/operation controls. That is a required future expectation, not permission to generate or run fixtures now.

## R1–R7 dispositions

| Ruling | Disposition | Review result |
|---|---|---|
| R1 JSON lift | CLOSED | Exact integer token/range rules reject bool/quoted/fraction/exponent/negative-zero alternatives; tags and Bool are explicit; Bytes is literal lowercase even hex; Option null/direct value retains mandatory keys. Duplicate decoded keys, unknown/missing fields, invalid Unicode/BOM/trailing data and byte identity are specified. Escaped ordinary strings and unescaped Bytes are intentionally distinct. |
| R2 closed rules | CLOSED | Every component kind0..21 has numeric rule equal to kind under `d7-cpu-allocation-v1`. The table selects identities only; it invents no physical formulas, byte totals or model PASS. |
| R3 scalar/nested/hash meanings | OPEN only for finding 1 | All six JSON record key lists and nested object types are explicit; Run.metadata is inline; source/registry/case identities, FileRefs, operation tags and syscall fields are assigned. The remaining supplied_input Hash meaning requires the one-field clarification above. |
| R4 projections/profile selectors | CLOSED | Allocator/Kernel projections are the full named objects; Topology.namespaces is full NamespaceOrigin. Decoded field/list equality is explicit. Compiled-profile Hash selectors remain opaque exact identities with actual support validation separate; no invented canonical-JSON digest/preimage. |
| R5 anchor-only facts | CLOSED | Policy exact-byte hash, duplicated boot/namespaces/domain and origin_evidence→installation_record.sha256 are defined. Generation/UID/host key/root tuples remain anchor-only, checked against actual trusted anchor/held facts. No HostPolicy keys are added; host_key is not equated to physical_host. This was a clarification, not an established missing-field defect. |
| R6 finite ID graph | CLOSED | Unique run IDs are scoped only to the containing Validation.runs. Every coverage/bootstrap reference resolves there; component associations are reciprocal; unrelated unreferenced runs reject; bootstrap-only runs have empty component list and a bootstrap reference. Repeated Environment contains finite strings, not nested runs. No required self-containing whole-file hash. |
| R7 ordering and work options | CLOSED | Numeric/decoded-UTF8 ordering and every list's uniqueness/size policy are specified. One operation/profile selector contains explicit sorted seeds; meaningful controller, argv and evidence sequence order is preserved. Option omission rejects, operation predicates are explicit, preparation cannot promote into Train, and no new global seed ceiling narrows registration. The hash value clarification is isolated to R3. |

## Exact checks and positive findings

The registry meaning is correct. `study_profile.rs:25–32,600–617` pins the raw approved experiment_matrix.json SHA256 to `164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc` and keeps the portable input-lock identity separate. `registered_case.rs:130–156` distinguishes that registry identity from SHA256 of complete registered-case canonical bytes. `document/admission.rs:58–100` separately encodes CaseIdentity.registration_id as the Text128 label. The contract preserves all three meanings and uses the canonical-case digest in SupportedWork.registration_id. `context.rs:816–819` confirms the existing approved compile-source manifest identity. No new registration identity or manifest authority is introduced.

The U64 seed representation does not promote unsupported inputs. Fixed `study_profile.rs:669–677` resolves current seeds0..9, and `registered_case.rs:213–246` retains the existing family/exact-Iris/supplement distinction and Full-only membership. The contract leaves actual membership authoritative, with Reference/target-lock specifically seed0; historical seed2020 is not admitted simply because it is representable as U64.

Run.metadata being inline removes the unlisted-seventh-JSON-format ambiguity. Full Environment equality, finite ID resolution and reciprocal component coverage are implementable with a bounded nonrecursive index. A repeated bootstrap ID may name the enclosing run without recursively containing that run. Source/build/raw artifacts precede their containing reports; model, validation, policy and anchor links remain acyclic. None of these structural checks determines whether a failed/timeout run or missing measurement establishes real validation. Empty/missing evidence stays representable as missing, not guessed zero or success.

All list occurrences were checked against the exhaustive ordering table: source files, supported work/seeds, three22-entry lists, runs and coverage IDs, covered components, allocator files, controller evidence/leaf-to-root rows, filesystem records/roles, argv and measurements. Some evidence and measurement lists deliberately preserve declared sequence instead of imposing a new sort; repeated measurements do not multiply coverage. Exact raw source caps and existing stricter caps remain controlling.

Independently recomputed the bootstrap using only literal string lengths and integer arithmetic, without generating binary bytes or calling a codec: primitive header is33 ASCII bytes, production domain33, test domain32. The framed field sum is:

`41 + 41 + 1 + 8 + 144 + 44 + 8 + 72 + 64 + 32 + 32 + 216 + 1 = 704`.

The test domain removes one byte, yielding703. Hash is framed Text64=72; UUID is Bytes36=44; namespace pairs and root tuples have no List prefix or extra object framing. Maximum704 plus one overflow byte is correct parser scratch arithmetic and is not B0/physical-model validation.

## Limits and next step

The contract is otherwise a suitable narrow closure for an independent literal oracle. Correctness and maintainability require finishing the one ambiguous Hash role before freezing positives/negatives; this does not justify reopening the selected resource architecture or adding a generic schema framework. No change to existing outer Resource/FileRef/Full/Quick bytes, caps, thresholds, authority, all21/full2680+20+7/quick268 scope or original holds was found or requested. R5/R6 were correctly treated as clarifications rather than asserted missing-field/hash-cycle bugs.

Actual model/bootstrap proof, native host/cgroup/filesystem observations, helper lifetime, same-owner typed wiring, later retained-source allocation costs and GPU/external/baseline work remain unchanged prerequisites. This review supplies no model numbers or fit claim.

Only `task-54-d7-resource-format-review.md` was written. No source/Cargo/publicdocs edit, project test/helper/generator/--list, native/runtime/HE execution, SSH, dataset/process scan, dependency install, spending, subagent, commit or push occurred. No literal vectors were created. The reviewed contract/report and ratified proposal were not modified. Return the bounded field clarification for scoped rereview before root ratification/resumption of the independent oracle author.
