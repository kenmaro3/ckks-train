# Task54 D7 resource format closure report

Status: **PROPOSED ADDENDUM, frozen for scoped independent review**. All seven normalization questions have explicit answers in `task-54-d7-resource-format-contract.md`; no unresolved representational contradiction was found. This is not a SPEC/QUALITY acceptance or source/encoder/corpus/runtime release. R1/R2/R3/R4/R7 were true representational gaps; R5/R6 are clarifications, not established missing-field or unavoidable-cycle defects. Ratified proposal e72b99a7, its reviews/maps and oracle report6c8bb195 remain unchanged.

Read the exact closure brief first, then complete oracle normalization report. Used engineering:system-design for the bounded field/tradeoff closure. Source inspection was confined to committed BASE `3a65f42dac83a17112c84dc3f23d3eb0aae7d0a8`; no constructor, helper or registry execution. The actual registry identity agrees with the root's source observation.

## Decisions, reasons, alternatives and cost

| Ruling | Selected exact closure | Reason, alternative and cost |
|---|---|---|
| R1 JSON lift | Bytes literal unescaped lowercase even hex; Option null/direct value with mandatory keys; numeric finite tags; lexical integer U64/I32 ranges, no exponents/fractions/negative zero; decoded-key duplicate rejection. | Reuses Quick-style byte transport and ordinary JSON values with minimal objects. Tagged option objects, byte arrays or string tags would create another spelling and larger fixtures. Cost: token-level validation and separate decoded-value/whole-file identity checks; no changed binary primitive. |
| R2 rules | Rule is numeric tag exactly equal to component kind0..21 under fixed `d7-cpu-allocation-v1`; all22 meanings explicitly tabulated. | Closed dispatch identity without inventing formulas/names/bounds. Alternative22 string names requires another vocabulary with no functionality gain. Cost: exact revision/tag completeness negatives before any implementation; numeric identity does not implement or validate the rule. |
| R3 scalar/record mapping | Exhaustive top-level/nested tables; model revision Text128, source/registry identities Hash, actual Git revision Git40; model/validation links FR; run.metadata inline RunMetadata; command/streams FR; syscall type U64; root tags existing0..5; absolute cgroup Text kept distinct from FR relative path. | Fits ratified six-record design and existing identities. A metadata FR would require another explicitly specified external format/inventory; it is not silently introduced. Model.registry is exact raw approved experiment_matrix.json SHA256, not case digest or input-lock. Cost: inline metadata duplicates bounded Environment bytes; existing raw8MiB cap wins, no truncation. |
| R4 projections/profile selectors | Host allocator_environment full Allocator, kernel_environment full Kernel; Topology.namespaces full NamespaceOrigin. Three bootstrap profiles remain opaque exact compiled selectors; no JSON hash preimage. | Removes partial-key ambiguity without adding architecture. Alternative smaller projections need a new explicit subset and more matching logic; canonical-JSON hashing invents an unratified hash contract. Cost: repeated bounded fields, exact comparison and later actual supported-profile validation still required. |
| R5 duplicated versus anchor-only | Match policy exact-file hash, boot/namespaces/domain; origin_evidence equals installation_record.sha256, selecting the same independently trusted provenance. Generation/UID/host key/root tuples remain anchor-only; no HostPolicy keys added. | The proposal already only required duplicated facts to agree. A separate independently pinned origin artifact could be acyclic but adds inventory/link work; using the same provenance is narrower. Host key does not equal physical_host by fiat. Cost: one explicit equality/negative and retention of actual independently expected origin; no new origin service or sudo dependency. |
| R6 finite IDs | validation_runs sorted run-ID strings scoped to the containing Validation.runs only; unique IDs, reciprocal component coverage, bootstrap-only run support. Metadata Environment equals enclosing Environment, including finite ID strings. | Repeating strings does not recursively contain runs. Alternative external report refs require extra source-role contracts; whole containing-file self-hashes are prohibited. Cost: bounded index and consistency checks. Model→Validation→policy→anchor whole-file links remain acyclic; raw artifacts precede their containing reports. |
| R7 ordering/matching | Explicit order/uniqueness for every list; numeric/UTF8 lexicographic supported-work keys; one Reference/Target selector per operation/profile with sorted seed union; preserve controller leaf→root, argv and declared evidence orders. All options/operation predicates are explicit. | Avoids ambiguous selectors without removing any supported seed. Alternative disjoint repeated selectors requires overlap-resolution checks; sorting observed/argv sequences would erase meaning. Cost: deterministic membership/duplicate tests. U64 seed representation has no new global ceiling; actual existing registered applicability governs, Reference/lock remain0. |

Six JSON domains and one previously proposed binary anchor remain. There is no new encoded metadata record, generic schema framework, approval layer, byte-total formula, model PASS or scientific registration. The table distinguishes SupportedWork.registration_id (case canonical digest) from inner CaseIdentity.registration_id (existing registration label).

## Bootstrap arithmetic and verification boundary

Verified existing header/domain lengths as33/33 production and32 test bytes. Hash is Text64 (72 bytes), boot UUID is Bytes36 (44), namespace/root tuples have no list framing. Sum `41+41+1+8+144+44+8+72+64+32+32+216+1 = 704`; test703. This was read-only string-length/integer arithmetic, not fixture generation or an encoder test. No literal anchor bytes, vectors, generators or parsers were created/run. 705-byte scratch sizing is not B0 physical-model validation.

Every record/branch field, null predicate, list and projection in the normalization report was compared with the contract tables. Source identity was checked against actual constants/getters and existing canonical-case construction. Exact input pins below were rehashed; the frozen identities match earlier task pins. This report's own hash is returned separately to avoid self-reference.

Remaining actual evidence is unchanged: complete source/allocator/helper/kernel/physical model, B0 bound, host origin/current observers, same-owner typed wiring and helper crash lifetime, all21/full2680+20+7/quick268, GPU/external/baselines and original holds. Representation fixtures cannot discharge these. Independent oracle work resumes only after root's scoped review/ratification; no runtime authority follows.

Only the canonical ledger contract and this report were written. No literal L directory; no source/Cargo/publicdocs changes, project generator/helper/test/--list/native execution, SSH, data scan, installation, subagent, commit or push. Read-only shell and stdlib hashing/document arithmetic only.

## Exact pins

| Ledger input/output | Bytes | SHA256 |
|---|---:|---|
| `task-54-d7-resource-format-closure-brief.md` | 3219 | `f6aef810c8966b923f2bac007edc698e24eabba26f0434cb789efe562c357f06` |
| `task-54-d7-resource-oracle-report.md` | 19235 | `6c8bb195139b379d54fe6d0ad999d962fdf5e2e5c00a938fd440bebb7e25a402` |
| `task-54-d7-resource-producer-proposal.md` | 65108 | `e72b99a7a599460ea3f2c206a5ecaa002bfa3ad9b62925f02da9d5d07f63604a` |
| `task-54-d7-resource-producer-review.md` | 18208 | `1081ab3e1297582247995da5ac801638be896cf02b59a4bb4094f7f5a9a97575` |
| `task-54-d7-resource-producer-fix1-review.md` | 12570 | `2474c2f7815fb05d301dfae7d6cc3916d674f40908d9ee2585ce88110a990351` |
| `task-54-d7-typed-materializer-preallocation-map.md` | 44480 | `31504e7c3cda346de741448103ce1d9629003bd3fc4359d3d6916db9ac583f9c` |
| `task-54-d7-resource-producer-proposal-before-fix1.md` | 45323 | `80d6bdb917e33f4db7fde5233dac15339ea91c76903a403a588bcda757ecaf74` |
| `task-54-d7-gate-schema-proposal.md` | 39493 | `1e54707536ee52c481cfa789c308d9f4605f096a1a720649c2eebe7a1dc6f50b` |
| `task-54-d7-resource-format-contract.md` | 29012 | `7f9130f56d629247ec08fe0ed0739e984168caae0eaecdfe77e51f2898318044` |

| Fixed BASE source blob | SHA256 | Relevant facts |
|---|---|---|
| `crates/ckks-train-bench/src/study_profile.rs` | `eaaf3ad085f1a347b664099f720a1ad2fb068ada66939baf60c3c6bbefe3670f` | 25–33 registry/input-lock constants;600–617 checked registry load;669–675 actual resolver seed rule |
| `crates/ckks-train-bench/src/study_profile/registered_case.rs` | `4ce1738959718d23e6265c2ae17f4d4507a30ed0260fb13bc965affa26067404` | 131–158 registry versus canonical-case digest;212–246 actual full/Quick sets and membership |
| `crates/ckks-train-bench/src/study_reference/document/admission.rs` | `d140e6879af76174ab6d1f226143d9454052b915c6872625f56a7510a2387cb0` | 58–99 existing CaseIdentity fields and canonical construction |
| `crates/ckks-train-bench/src/study_provenance/context.rs` | `31454576d90d557ef1950d3b45e4d404a4630fc29b862d5fa6fd3e3360263aa3` | 816–819 existing approved compile-source manifest identity |

## FORMAT FIX1 — selected supplied-input hash role

Current status: **single Medium R3 finding addressed in proposed prose; scoped independent re-review pending**. The preceding original closure report is preserved byte-for-byte as historical submission5298ef3a. Review `task-54-d7-resource-format-review.md` (dac82a73) requested changes only for the identity carried by Target supplied_input; its R1/R2/R4/R5/R6/R7 findings were CLOSED. This append does not claim reviewer acceptance or authorize vectors/encoder/source/runtime.

**Resolution:** Some is exactly the actual already-retained supplied `MaterializedProfile.identity().digest()` — the existing D6a combined materialized input/content/source identity. Derive it from that SAME live owner. Its later selected receipt must satisfy `Some value == SeedInputReceiptV1.materialized_digest()`. It is expressly not the receipt's complete canonical-byte digest or the shared contract digest. Independent actual/selected reconstruction and the complete actual/supplied receipt canonical-byte equality remain required, even when the materialized hashes match. Same-owner/existence predicates, Target lock null/seed0, None for future construction, preallocation accounting and immutable no-future-hash claim are unchanged.

**Reason / alternative / cost:** D6a already names the supplied object and is copied into the selected receipt; choosing it reuses existing identity semantics without another field/hash/preimage. Selecting receipt.digest or contract.digest instead would represent a different artifact and require a distinct meaning/link; neither is substituted. Cost is one explicit source-derived equality and future oracle distinctions for D6a versus receipt hash, contract hash and a one-byte wrong digest, alongside existing Some/None/operation controls. These are future expectations only; no vectors or tests ran. Hash equality is not a replacement for the complete receipt byte comparison.

**Source evidence, fixed BASE3a65f42:** `study_inputs.rs:285–289` returns MaterializedIdentity.digest; `388–390` exposes it from MaterializedProfile. `study_reference.rs:95–103` hashes shared contract canonical bytes; `114–133` separately exposes receipt digest and materialized_digest. `derive_identities:737–758` copies `p.identity().digest()` into the receipt's materialized field. `study_reference/target.rs:364–369` independently derives selected receipts from actual reconstruction and supplied profile, then compares their complete canonical bytes. These accessors were read, never executed.

Only the contract received the one-field definition/equality clarification plus FIX1 status/source references. This report received this append only. Before-FIX1 contract7f9130f5/report5298ef3a snapshots are exact. Scoped diff covers those two text deltas; `task-54-d7-resource-format-fix1-pins.json` records final artifact/source hashes, including the diff, without a self-hash. Ratified proposal e72b99a7, normalization6c8bb195, reviewdac82a73 and prior maps/reviews are immutable. All outer formats/caps, scientific criteria, all21/full2680+20+7/quick268 and original holds remain unchanged.

No source/Cargo/publicdocs, generator/vector/project-helper/test/--list/native/runtime/SSH/data/process scan, subagent, installation, commit or push. Read-only source/doc inspection and stdlib document snapshot/diff/hash bookkeeping only. No physical/model/host/lifetime evidence is established.

| FIX1 input/output | Bytes | SHA256 |
|---|---:|---|
| `task-54-d7-resource-format-contract-before-fix1.md` | 29012 | `7f9130f56d629247ec08fe0ed0739e984168caae0eaecdfe77e51f2898318044` |
| `task-54-d7-resource-format-closure-report-before-fix1.md` | 9328 | `5298ef3a3203f1b140683ca3bc481c75179aeb4cf90bb3fd3d7ae665f23775d2` |
| `task-54-d7-resource-format-review.md` | 11817 | `dac82a73edf6dfa800ac01dce63ec70b0e5b75ccd020cfba4be43ff08ee5f43e` |
| `task-54-d7-resource-format-contract.md` | 30832 | `a7c947fceca6cf7bb7415114838de0de87c7bbe21657efef4da792b4bd392429` |

| Fixed BASE source | SHA256 |
|---|---|
| `crates/ckks-train-bench/src/study_inputs.rs` | `fd641f27fc454ad85eb3c1ef67db759459a1ad22dc2a42dff15373c9650063a9` |
| `crates/ckks-train-bench/src/study_reference.rs` | `c04c340c754f17a674215f56e9d92bbdf643ab3cd3e33a1f01d482d8dd91e6c2` |
| `crates/ckks-train-bench/src/study_reference/target.rs` | `ccb13e42286c80bf609ae6d2e05b7e9f059e664a22e3ad4e1acb20bd8620b69b` |
