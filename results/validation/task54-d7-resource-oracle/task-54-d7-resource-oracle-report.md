# Task54 D7 resource source-format independent oracle

Status: NORMALIZATION / RULINGS PENDING. Artifact-only; no fixture generation or verification has run. This is not a model/source/native/host validation or execution release. Full Task54, all21 profiles, Full2680+20+7, Quick268 and CPU/GPU/baseline/external obligations remain.

## Authority and read boundary

The oracle brief was read first, then `task-54-brief.md`, `docs/study-resource-design.md`, the ratified resource-design README, proposal and FIX1 review. Controlling proposal is `results/validation/task54-d7-resource-design/task-54-d7-resource-producer-proposal.md` (e72b99a7a599460ea3f2c206a5ecaa002bfa3ad9b62925f02da9d5d07f63604a), with scoped review2474c2f7 and ratified BASE3a65f42dac83a17112c84dc3f23d3eb0aae7d0a8. Existing primitive descriptions were inspected in `task-54-d7-gate-schema-proposal.md` sections1/2/6 and `docs/study-target-design.md`. No project codec/helper/test was called. Source d1aa3af retained-store costs are an actual-model obligation, not covered by this representation work.

## Exact normalization determined by the accepted prose

P denotes the controlling resource proposal; G denotes the gate-schema proposal. JSON member order is immaterial; the listed order is a documentary/writer convention, not a parser restriction. Binary field order is normative. All source JSON is strict UTF8, duplicate/unknown keys and trailing non-whitespace reject, and hashes/lengths refer to the original whole-file bytes without normalization (P section1). No self-hash, pass bit, extra metadata or default/omitted branch fields.

| Primitive / cap | Exact existing rule |
|---|---|
| U64 / I32 / tag / Bool | U64 little-endian in binary; I32 little-endian two's-complement; tag and Bool one byte (G sections1/2). JSON representations require ruling R1 below. |
| Text / Bytes / Hash / Git40 | Binary Text is U64 UTF8-byte-length plus bytes; Bytes is U64 length plus bytes; Hash is Text of exactly64 lowercase SHA256 hex; Git40 exactly40 lowercase hex. Text128 identifiers match `[A-Za-z0-9_-]{1,128}`; other Text maximum4096 UTF8 bytes (G section1). Bytes JSON lifting remains R1. |
| List / Option | Binary List U64 count then values; Option00 or01+value; checked counts/conversions/aggregate accounting. JSON lifting remains R1. |
| Binary header | `Text("hekit-study-reference-encoding-v1"),Text(domain),synthetic:Bool`; production domain prefix `hekit-study-` with false; private test domain prefix `hekit-test-` with true. Primitive header remains unchanged (G section1; P minimum anchor). |
| FileRef | `{root,path,length,sha256}`; root Input0/Source1/Reviewed2/Policy3/Authority4/Observation5, relative nonempty UTF8 path, no empty/dot/dot-dot components, NUL, backslash or absolute prefix; length U64, sha256 Hash. Identity only, never artifact-directed opening (G section2). |
| Clock | `{wall_source,unix_ns,monotonic_source,elapsed_since_operation_start_ns}`; source tags0, U64 counters, same operation origin (G section2). |
| Transport | Source raw file at most8MiB and any stricter type cap; list at most65536, source-list512; exact cap/cap+1 by arithmetic witness without allocating cap-sized files (P section1; G section1). |
| Requirements | `{components,peak_host_bytes,incremental_host_bytes,required_disk_bytes}`; exactly22 ordered components0..21, each `{kind,phase_mask,resident_bytes,additional_bytes}`. Masks1..15, or0 with both bytes0 only for source-proved inapplicability. Four phase sums of resident+additional and additional alone produce the respective maxima using checked U64 arithmetic (G section6). |

| Source / exact keys | Fields and exact known types / unresolved mappings |
|---|---|
| Model | `schema,synthetic,revision,source_manifest,source_revision,source_files,registry,build_environment,rule_revision,supported_work,components,bootstrap_derivation`; schema exact `hekit-study-resource-model-v1` (or test prefix), synthetic Bool; source_revision Git40; source_files List<FileRef> of the relevant SAME approved manifest subset; build_environment/bootstrap_derivation FileRef; rule_revision exact `d7-cpu-allocation-v1`; components22 ordered `{kind,rule,derivation}`, kind tags0..21, derivation FileRef. Other scalar assignments and rule tags R2/R3. P section1. |
| Validation | `schema,synthetic,model,source_manifest,source_revision,environment,coverage,runs`; schema exact `hekit-study-resource-validation-v1`; Environment below; coverage22 ordered `{kind,source_derivation,validation_runs}`; source_derivation FileRef; runs List of `{id,command,stdout,stderr,metadata,covered_components}`. Exact remaining mappings/references R3/R6. P section1. |
| Host policy | `schema,synthetic,boot_id,physical_host,host_namespace_origin,allocator_environment,kernel_environment,filesystem_environment,claim_domain,model,validation`; schema exact `hekit-study-resource-host-policy-v1`; boot_id canonical boot UUID; physical_host Hash; host_namespace_origin NamespaceOrigin; filesystem_environment List<FilesystemProfile>; claim_domain exact `task54-host-serial-v1`; model/validation reviewed FileRefs fixed `resource/linux-cpu-v1/model.json` and `resource/linux-cpu-v1/validation.json`. Projection definitions R4/R5. P section1. |
| Claim | `schema,synthetic,claim_id,claim_domain,physical_host,boot_id,model,validation,work,approval,source_manifest,source_revision,created,modeled_envelope`; schema exact `hekit-study-resource-claim-v1`; work closed branch below; approval Hash; created Clock; modeled_envelope Requirements; physical_host Hash; source_revision Git40; fixed serial domain. Remaining mappings R3. P sections1/2A. |
| Host observation | `schema,synthetic,claim_id,sequence,physical,started,finished,topology_before,topology_after,meminfo_before,meminfo_after,controllers,structural_root`; schema exact `hekit-study-resource-host-observation-v1`; sequence U64, physical Hash, clocks, Topology objects, meminfo FileRefs; controllers List of `{version,path,limit_raw,current_raw}` using v1 tag0/v2 tag1 and FileRefs matching existing Cgroup rows; structural_root StructuralRoot. P section1; G section6. |
| Disk observation | `schema,synthetic,claim_id,sequence,started,finished,filesystems`; schema exact `hekit-study-resource-disk-observation-v1`; sequence U64, clocks; filesystems List of `{identity,roots,frsize,bavail,favail,required_bytes,required_files,quota_evidence}`; numeric counts U64, quota_evidence FileRef; identity/roots shared with FilesystemProfile. P section1. |

| Closed nested record | Exact keys / known types |
|---|---|
| Environment | `{target,toolchain,cargo_lock,build_receipt,allocator,kernel,workers,bootstrap}`; target Text128; next three FileRefs; workers U64; exact Allocator/Kernel/BootstrapCoverage below. P section1. |
| Allocator | `{implementation,library_files,configuration,derivation}`; implementation system-glibc tag0 only; library_files ordered List<FileRef>; configuration and derivation FileRefs (configuration inferred by explicit "identities/configuration and reviewed source derivation"; confirm R3). |
| Kernel | `{release,version,page_size,source,configuration}`; version Text4096; page_size U64; source/configuration FileRefs; release Text mapping R3. |
| BootstrapCoverage | `{source_derivation,validation_runs}`; source_derivation FileRef; validation_runs mapping/graph R6. |
| RunMetadata | `{source_manifest,source_revision,executable,environment,argv,exit_code,timed_out,started,finished,measurements}`; source_revision Git40; environment full Environment; argv ordered Bytes list; exit_code Option<I32>; timed_out Bool; clocks; measurements List of Measurement; source_manifest/executable hashes per P's "Hashes/revision". |
| Measurement | `{kind,scope,bytes,raw}`; kind allocator-owned peak0/process-RSS peak1/cgroup-current peak2; scope self0/self-plus-controlled-children1/current-cgroup2; bytes U64; raw FileRef. No missing-as-zero. |
| NamespaceOrigin | `{boot_id,mount_ns,cgroup_ns,pid_ns,user_ns,installation_record}`; four identities each `{device,inode}` U64; installation_record retained independently trusted FileRef; boot_id canonical UUID. Its presence in fictional bytes proves no host origin. |
| Topology | `{membership,mountinfo,controllers,namespaces}`; first three FileRefs; namespaces NamespaceOrigin projection, exact projection unresolved R4. |
| StructuralRoot | `{version,domain_path,mount_identity,disposition,controller_raw}`; version v1=0/v2=1; disposition no-controller0/global-root-no-local-interface1; controller_raw ordered List<FileRef>; domain_path/mount_identity exact types R3. No fabricated root memory.current row, hidden ancestry remains missing evidence. |
| FilesystemProfile | `{identity,roots,mount_evidence,quota_evidence,derivation}`; identity `{device,mount_id,type}` from held descriptors/mount data; roots sorted existing trusted root-role tags; last three FileRefs. Identity scalar/root tag mapping R3. |

Work predicates (P section2A; no omitted option fields):

| Record / kind | Exact branch fields and predicates |
|---|---|
| Supported Train0 | `{kind,registration_id,consumption}`; exact registered case Hash and Full0/Quick1. |
| Supported Reference1 | `{kind,operation,profile_id,seeds}`; read0/recover1/snapshot2, exact registered Text128 profile, sorted explicit seeds; acceptance seed0. |
| Supported Target2 | `{kind,operation,profile_id,seeds}`; read0/lock1; sorted explicit selected seeds, lock only0. |
| Claim Train0 | `{kind,registration,request}`; canonical registration/request Bytes only. Registration/request domains/payloads remain existing contracts; no new scientific case. |
| Claim Reference1 | `{kind,operation,reference_id,expected_file,profile_id,seed,document}`; expected_file Option<Hash> Some iff recovery and equals actual expected file; seed0 acceptance; document FileRef. |
| Claim Target2 | `{kind,operation,campaign_id,profile_id,selected_seed,target,reference,supplied_input}`; target Option<FileRef> Some iff read; reference FileRef; supplied_input Option<Hash> Some only for an input already under SAME live owner, None when construction is future. Lock seed0, target None and supplied_input None. |

Supported-work sort key is not explicitly defined across branches (R7). Train is the only branch allowed to finalize Resource/HE; preparation may not promote, emit outer Resource or invent registration/request. Nested work borrows one owner/outer envelope, creates no second claim or lock. Fictional structural fixtures cannot prove owner lifetime, namespace origin, canonical materialization or source-model applicability.

### Exact bootstrap binary layout

P minimum-anchor paragraph fixes domain `hekit-study-resource-bootstrap-v1`, followed by ordered fields:
`generation:U64,origin_evidence:Hash,policy:Hash,boot_uuid:Bytes[exact36],account_uid:U64,host_key:Hash,namespaces:[mount,cgroup,pid,user]x(device:U64,inode:U64),proc_root:(device,inode,mount_id,fs_type):U64x4,cgroup_root:(device,inode,mount_id,fs_type):U64x4,kernel_profile:Hash,allocator_profile:Hash,filesystem_profile:Hash,claim_domain:tag0`.

Manually derived length: primitive header Text33=41, production domain Text33=41, Bool1, generation8, two Hash72=144, UUID Bytes44, uid8, host Hash72, namespace pairs64, proc/cgroup root tuples64, three profile Hash216, claim tag1: **704 bytes production / 703 test** (test domain is one byte shorter). Fixed scratch needs maximum704 plus one overflow byte. This is an arithmetic normalization expectation, not yet executed/hashed. Exact profile-hash preimages and policy projection links remain R4/R5; no numeric model allowance exists here.

## True design questions reported before choosing/generating

No unresolved field is silently assigned a wire representation. Root rulings must be recorded before affected bytes are generated.

| ID | Precise gap | Safe alternatives / proposed narrow choice and cost |
|---|---|---|
| R1 JSON lift | Existing primitives are binary contracts. P says JSON reuses primitive rules but does not specify Bytes JSON representation, Option JSON representation, tag numeric versus string spelling, or whether integer lexemes such as1.0/1e0/-0 are valid U64. | Recommend Bytes unescaped lowercase even hex (as existing Quick transport); Option null/value (all keys mandatory); tags integer JSON numbers; U64/I32 lexical integer only, range checked, reject Bool/float/exponent and negative zero for unsigned. Alternative tagged `{some:...}`/numeric arrays/string tags would be a new incompatible wire choice and add objects. Must rule, not infer encoding. |
| R2 closed rules | P fixes rule_revision `d7-cpu-allocation-v1` and says each rule equals its matching compiled rule but gives no 22 exact rule values/type and no existing compiled rule table. | Recommend `rule` numeric tag equal to kind0..21 under that fixed revision, not new strings. Alternative explicit closed22 Text128 names requires a root-approved literal table. No formulas or byte totals either way. |
| R3 scalar/nested mapping | P's "all remaining names use existing ... primitives" does not map model registry, revision/source_manifest; validation.model; coverage.validation_runs; run id/command/stdout/stderr/covered_components; StructuralRoot.mount_identity/domain_path; filesystem identity.type; root-role namespace. | Recommend model revision Text128, source_manifest/registry Hash, validation.model and claim model/validation FileRef; runs id Text128, command/stdout/stderr FileRef, covered_components sorted List<tag>; validation_runs sorted List<Text128> referring run ids; kernel.release Text4096, filesystem identity.type U64 statfs magic, StructuralRoot.mount_identity same `{device,mount_id,type}` as FilesystemProfile, domain_path Text4096 allowing absolute cgroup paths; roots FileRef root tags0..5. Alternative mount-identity tuple including inode is materially different; type as filesystem-name Text also differs. These must be explicit. |
| R4 projections/profile hashes | Host allocator_environment/kernel_environment are called Environment identity projections "below" but no exact projection key list is given. Topology.namespaces is also called a NamespaceOrigin projection without exact keys. Bootstrap profile hashes select compiled supported profiles; this does not itself require hashing JSON projections. | Recommend policy allocator_environment exact Allocator object and kernel_environment exact Kernel object, topology.namespaces full NamespaceOrigin. Treat profile hashes as opaque exact compiled-supported-profile selectors with independent matching fixtures, unless a specific byte preimage is already frozen. Do not invent canonicalized-JSON hashing or new hash contracts. Confirm whether the oracle must exercise any additional specified projection relationship. |
| R5 anchor-only facts (not missing fields) | Bootstrap account_uid, host_key, proc_root/cgroup_root and generation are absent as explicit host.json keys. P requires only duplicated projection facts to agree. Their absence does NOT establish a design gap or authorize adding policy keys. The exact origin_evidence↔installation_record relationship is not stated. | Compare duplicated boot_uuid↔boot_id and namespaces; policy Hash equals exact host.json original bytes. Remaining anchor-only facts match independent launcher expected anchor/held descriptors, without invented JSON fields. host_key is a stable selector, physical_host a physical descriptor hash: no equality is assumed. If origin_evidence must equal installation_record.sha256 rather than select independent provenance, root should specify that exact link; either choice is acyclic. Profile field shapes remain R4. |
| R6 reference graph (conditional concern only) | P defines coverage `{kind,source_derivation,validation_runs}`, Environment.bootstrap `{source_derivation,validation_runs}`, runs `{id,command,stdout,stderr,metadata,covered_components}` and RunMetadata.environment as full Environment. `validation_runs` reference type is absent. These clauses do NOT force a hash cycle: ID links are finite. Only choosing references to containing validation whole-file bytes would create one. | Recommend IDs, not containing-file refs/self-hashes; unique run IDs, all coverage/bootstrap IDs resolve within top-level runs; run metadata Environment matches validation.environment, including same ID lists. Alternative external evidence FileRefs also need not cycle but require an explicit role/link definition and inventory. Do not describe a forced graph defect before the reference type is settled. |
| R7 ordering/scope | supported_work must be sorted unique but sort relation is unspecified; roots/filesystem entries/controller_raw/library_files ordering beyond some "ordered" wording is not fully fixed. | Recommend work lexicographic `(kind, registration_id,consumption)` / `(kind,operation,profile_id,seeds)` with numeric tags and UTF8 byte string order; unique roots ascending tags, source files existing `(root,path UTF8)` order. Preserve meaningful observed controller leaf→root and argv order. For filesystem list recommend identity tuple order. For library/controller_raw preserve explicit evidence order, not invent sorting. Root should confirm exact lists where uniqueness applies. |

## Inventory BEFORE any standalone generator/verifier execution

No generator/verifier exists or has executed yet. Until R1–R7 are resolved, permitted output is this report only. No descendant process, project/runtime constructor, project encoder/parser/generator/materializer, tests/--list, native/HE/SSH or dataset/host observation is permitted. Read-only commands so far inspect named project documentation, one source root-role search, and the memory registry. No source/Cargo/publicdocs change, commit/push or subagent.

The next exact bounded inventory will be added here before any standalone code is run; it will enumerate literal input and output paths/count/aggregate bounds, script commands and the negative-case check ordering. This paragraph is not an execution authorization or a vague substitute for that inventory.

## Performed commands / failures

Read-only shell commands returned exit0 except one combined read/search call: final `rg --files .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5 | rg 'golden|primitive|schema|oracle'` returned1 (no filename matched). Earlier commands in that call succeeded; complete captured output remains tool chunk f23bc7, warning `truncated output (original token count:10733)`, actual command exit1. No retry changed any input, no generation was attempted and no scientific result JSON was written. Other large read output was truncated by the tool budget; the relevant exact sections were subsequently read in smaller slices.

## Remaining gaps / review freeze

Corpus generation, independent literal pins, negative execution and review freeze remain pending root's format rulings. Physical/source model proof, B0 allocation bound, trusted host origin, actual loaded allocator/kernel/filesystem/helper coverage, same-owner byte wiring, claim lifetime/native observer tests, full scientific matrix and GPU/external work remain absent. Representation fixtures will never discharge these obligations.

## Resumption after exact format ratification — pre-execution inventory

Root resumed artifact-only work against committed format52f10dad592cf0a6def8b3ddf9b8061203f91a50. Read original oracle brief first, then complete format ratification, final contract30832 bytes SHA256 `a7c947fceca6cf7bb7415114838de0de87c7bbe21657efef4da792b4bd392429`, closure report77e2fc1d and scoped SPEC/QUALITY PASS0f1edebe. R1–R7 are closed exactly by that contract. Initial report above remains an exact byte prefix; its exact19235-byte snapshot6c8bb195 is retained as `task-54-d7-resource-oracle/normalization-before-corpus.md` before any generation.

Exact permitted code execution: `python3 <canonical-ledger>/task-54-d7-resource-oracle/generate.py` once to write the fixed corpus, and `python3 <canonical-ledger>/task-54-d7-resource-oracle/verify.py` once for independent structural verification. Any later run, correction, exception or failed output will be appended explicitly. Both are NEW standalone Python standard-library code, no subprocess/descendants, network, project imports/encoders/parsers/generators/materializers/runtime constructors, source-tree enumeration, native/HE/SSH or real datasets/host observations. Exact input whitelist:

- Local authored literal/code inputs: `generate.py`, `verify.py`, `literal-expectations.json`, `negative-cases.json` in this oracle directory. The scripts contain only finite fictional literal record construction/checks, no generic schema framework.
- Read-only reference documents: original oracle brief, final archived `results/validation/task54-d7-resource-format/task-54-d7-resource-format-contract.md`, its closure report, FIX1 review and ratification. Generator does not parse documentation into a schema or invoke code from it; verifier may hash the contract for its pin.
- Existing frozen byte inputs (read/copy only): canonical-ledger `task-54-d7-gate-schema-vectors/registration.bin`, `full.request.bin`, `quick.request.bin`. Their own project generator/verifier are NOT executed. Existing outer Resource/Full/Quick fixtures are not written. Registration remains its frozen legacy grammar; Request domain/synthetic projection is explicitly re-created for each fictional package while retaining its unchanged body. No actual registry or case constructor is called.

Exact finite output classes under this oracle directory: one unchanged normalization snapshot; eleven named tiny raw artifacts `raw/facts.txt,source.txt,manifest.txt,registry.txt,reference.json,target.json,materialized.txt,receipt.txt,contract.txt,empty.bin,executable.txt`; two packages `synthetic/` and `inert-production/`, each exactly14 files: `model.json,validation.json,host-policy.json,claim-train-full.json,claim-train-quick.json,claim-reference-read.json,claim-reference-recover.json,claim-reference-snapshot.json,claim-target-read-future.json,claim-target-read-supplied.json,claim-target-lock.json,host-observation.json,disk-observation.json,bootstrap.bin`. Eight claim branches exercise two Train, three Reference and three Target cases. Each package is representation-only, with no real source/model/host/capacity PASS. Only synthetic uses test domains/true; production-shaped inert examples use study domains/false without trusted origin, matching actual build/model or native evidence.

Negatives are one file per exact entry in the authored `negative-cases.json`, whose complete fixed name list and count will be written here before execution; no adaptive/fuzz/random input. Additional outputs are `manifest.json` (all corpus lengths/SHA256), `generation-result.json` (bounded write inventory), `verification-result.json` (exact independent outcomes) and `SHA256SUMS` (final independent pin ledger). Generator outputs at most120 files/2MiB aggregate, individual≤64KiB; raw artifacts each≤256 bytes, package JSON each≤32KiB, bootstrap exactly703/704. Generator checks both bounds before writes. Verifier reads only its exact manifest/input lists, never descendants or project runtime. The final SHA256SUMS excludes itself and the append-only report to avoid self-reference.

Hand-authored `literal-expectations.json` will precede generator execution and pin literal primitive offsets, bootstrap lengths/facts, independently calculated Requirements phase sums, disk arithmetic, option predicates, distinct supplied materialized/receipt/contract roles, finite run-ID links and fixture-context scope. Negative expectations are authored separately from structural checks so each mutation must first reach its intended refusal. Lexical/closed-object validation precedes semantic/link checks; syntactically valid link negatives must preserve necessary raw-file/policy dependency identities in their private check contexts instead of failing an earlier stale hash. No verifier result proves native semantics, owner lifetime or accepted physical evidence.

Read-only resumption search: one filename search returned1 because ignored ledger files were not included; following explicit `rg --files --hidden --no-ignore` located the exact frozen input names. No generation had run at this point. The earlier accidental literal `L/` report location was corrected by exact-byte move to this canonical ledger and empty-directory cleanup; snapshot hash remained6c8bb195.

### Finite negative inventory frozen BEFORE execution

`task-54-d7-resource-oracle/negative-cases.json` now contains exactly84 numbered cases, each with its complete literal mutation/witness and independently specified first-refusal `expect`. This is the coverage matrix. Output is `negative/<id>.bin` for n69–n77 (nine bootstrap mutations), and `negative/<id>.json` for all other75 cases; exact ids follow. No random/adaptive cases. Expected generator output count is125: eleven raw +28 package files +84 negatives +manifest/generation-result. Bound is revised BEFORE execution from120/2MiB to **150 files/3MiB**, individual≤64KiB; all other limits unchanged. Final directory count is132 including four authored inputs, normalization snapshot, verifier-result and SHA256SUMS. This still excludes the sibling append-only report.

```
n01-missing-component
n02-duplicate-component
n03-component-order
n04-unknown-rule
n05-rule-revision
n06-malformed-json
n07-duplicate-decoded-key
n08-unknown-key
n09-missing-key
n10-fraction
n11-exponent
n12-negative-zero
n13-bool-as-integer
n14-u64-overflow
n15-quoted-number
n16-invalid-utf8
n17-trailing-junk
n18-bom
n19-invalid-scalar
n20-source-manifest
n21-source-revision
n22-build-receipt
n23-run-environment
n24-missing-validation
n25-validation-hash
n26-model-hash
n27-mixed-domain
n28-synthetic-pair
n29-absolute-fileref
n30-duplicate-run
n31-unresolved-run
n32-reciprocal-coverage
n33-unreferenced-run
n34-duplicate-source
n35-source-order
n36-work-order
n37-overlap-work
n38-read-expected-some
n39-recover-expected-none
n40-recover-wrong-hash
n41-snapshot-expected-some
n42-reference-seed
n43-read-target-none
n44-lock-target-some
n45-lock-input-some
n46-lock-seed
n47-supplied-receipt-role
n48-supplied-contract-role
n49-supplied-one-byte
n50-future-input-some
n51-other-owner
n52-preparation-train-promotion
n53-request-mixed-domain
n54-seed-membership
n55-claim-observation-link
n56-observation-sequence
n57-observation-physical
n58-topology-namespace
n59-v1-global-root
n60-fabricated-root-row
n61-hidden-ancestor
n62-duplicate-root-role
n63-disk-product-overflow
n64-disk-sum-overflow
n65-phase-overflow
n66-peak-mismatch
n67-incremental-mismatch
n68-raw-cap-plus-one
n69-bootstrap-short
n70-bootstrap-policy
n71-bootstrap-namespace
n72-bootstrap-origin
n73-bootstrap-profile
n74-bootstrap-domain
n75-bootstrap-synthetic
n76-bootstrap-trailing
n77-bootstrap-hash-prefix
n78-escaped-bytes
n79-i32-overflow
n80-argv-nul
n81-insufficient-inodes
n82-huge-finite-not-unlimited
n83-zero-other-without-serial-domain
n84-omitted-accounting-parent
```

Literal expectations file is authored separately before execution: fixed source bytes and nested input pins; bootstrap field offsets/703–704 lengths; phase totals `[6,3,0,0]`, additional `[5,3,0,0]`; disk32 available/12 required/two files; D6a/receipt/contract distinct preimages; finite cap/headroom/huge-finite witnesses. Static fixture context is labeled fictional and cannot certify real live owners, trusted origin or a model. First-refusal outcomes will be recorded by the new verifier and independently reviewed, not inferred from mutation names.

## First execution and direct-inspection FIX1 inventory

Generator1 command `python3 <canonical-ledger>/task-54-d7-resource-oracle/generate.py`, exit0, no stderr; exact stdout (tool chunk d619c4):
`{"aggregate_bytes": 441582, "largest_bytes": 15637, "notice": "FICTIONAL REPRESENTATION ONLY; no source/model/host/native authority", "output_count": 125}`.

Verifier1 command `python3 <canonical-ledger>/task-54-d7-resource-oracle/verify.py`, exit0, no stderr; exact stdout (f3ed54):
`{"mismatches": [], "negatives": 84, "positive": 28}`.

It matched66 source-JSON,9 bootstrap and9 private-context witness negatives;28 source/bootstrap positives and5 private arithmetic/selector witnesses. This first green structural result was **insufficient for the Train registry relation**, as direct `cat`/`xxd` inspection then found. The unchanged nested legacy registration embeds registry Hash `a56c31a8778df6304d467a3d7f2fdf1844c8d8d75117ebb017c6e6e2c4ffa26f`; initial Model.registry was `f73b463d59b8b5adcf8a6f17a2fbc413271c1d67bbaf9e9728a51c2bf0f11a53`, SHA256 of a different fictional literal. Verifier1 pinned the complete registration bytes and preserved Request bodies but did not separately decode/check that registry relationship. This is a fixture/check defect, not a design gap or permission to change frozen legacy registration.

BEFORE correction/execution: preserve all131 existing oracle files under `before-fix1/` (exact original snapshot, four authored code/input files,125 generator outputs and verifier-result). Named added read-only fictional input is `<canonical-ledger>/task-54-d7-gate-schema-vectors/retained/synthetic/registry.txt`, SHA256 `a56c31a8778df6304d467a3d7f2fdf1844c8d8d75117ebb017c6e6e2c4ffa26f`; it is explicitly synthetic text, not experiment_matrix.json or actual scientific data. Its exact byte count is recorded with the subsequent pre-run pin read below. The correction uses those exact original fictional registry bytes as `raw/registry.txt`; legacy registration.bin remains byte-identical. Add n85-registry-identity.json: a well-formed Model with only registry Hash changed, expected first refusal `registry-identity`; correct Model.registry/nested registry exact equality is the valid prerequisite. Existing Request body equality checks remain.

Revised combined bound BEFORE snapshot/corrective execution: **300 files/6MiB**, individual≤64KiB, with snapshot separate from current manifest. Corrected generator outputs126 (eleven raw+28 package+85 negatives+two bookkeeping); corrected current submission133 including snapshot-normalization/four inputs/verifier/SHA256SUMS. Combined with before-fix1's131 files gives264 files. Both scripts remain standalone stdlib/no descendants. Generator2 will read the four fixed authored inputs, three original nested binaries plus the exact frozen synthetic registry text; preflight all output names/aggregate bounds then replace only its own125 prior outputs and add n85, after their complete snapshot is preserved. No physical/source/evidence JSON outside this private fictional oracle is written. Verifier2 will inspect only corrected current artifacts, not run the archived scripts. No source/native/HE/SSH/project helpers/tests/materializers or real data are introduced.

Direct inspection already read complete synthetic Model/Validation/HostPolicy, full bootstrap hex, Train Full/Reference recovery/Target supplied/Target lock/HostObservation/DiskObservation raw JSON plus inert-production bootstrap header and independent shell hashes. It confirmed inline run metadata, run-a all22 reciprocal links and run-b bootstrap-only timeout/null-exit, nonUTF8/empty argv, mandatory null options, separate host/physical hashes, structural-root row exclusion and claimed Requirements arithmetic. These checks did not supply actual source/native evidence.

Exact added frozen registry input length:89 bytes (read-only wc before corrective execution, tool chunk b51fb6). All131 old files were copied before edits to `before-fix1/`; no prior output is discarded. The added85th negative is a source-JSON case, so corrected negative category counts will be67 source-JSON/9 bootstrap/9 witness. Initial category counts/results remain historical above.

## Final corrected corpus — frozen for independent review

Status: **ARTIFACT IMPLEMENTED / AUTHOR STRUCTURAL CHECKS PASS; independent review pending**. No production source/model/native acceptance. The original19235-byte report prefix and its snapshot6c8bb195 are unchanged; subsequent work is this append. Root ratified format a7c947fc governs every unresolved entry in the historical normalization. No new format question remains open.

| Normalization closure | Adopted exact rule |
|---|---|
| R1 | Lexical bounded integers (not Bool/fraction/exponent/−0), direct/null mandatory Options, numeric finite tags, literal unescaped lowercase even hex Bytes; original UTF8 JSON bytes/hash, duplicate decoded-key rejection. |
| R2/R3 | All22 numeric rule identities equal kind under fixed d7-cpu-allocation-v1; source/registry Hashes, Git40 revision, FileRef model/validation links, inline run.metadata, run-ID strings, numeric statfs magic and existing root tags0..5. Model.registry is raw-registry-file SHA, distinct from registered-case SHA/registration label/input-lock. |
| R4/R5 | Full Allocator/Kernel policy projections; full NamespaceOrigin topology projection. Opaque supported-profile selectors, no invented JSON hash preimage. Anchor policy selects exact host-policy bytes; boot/namespaces/domain and origin_evidence=installation_record.sha256 cross-check; generation/UID/hostkey/root tuples remain anchor-only. No hostkey=physical_host or host=execution-physical equality. |
| R6 | Same-Validation finite IDs, inline metadata Environment equality, reciprocal component associations, bootstrap-only timeout/null-exit run; no containing-file self-hash or recursive expansion. |
| R7 | Exact contract list ordering/uniqueness; eight explicit claim examples, no option omission, Reference/lock seed0 and read membership only. Seed7 is a fixture example, not a new global limit. Supplied Some denotes fictional D6a materialized identity, separately distinguished from receipt/contract digest; actual full receipt comparison remains mandatory future source behavior. |

Generator2 command `python3 <canonical-ledger>/task-54-d7-resource-oracle/generate.py`, exit0, no stderr; exact stdout (10155d):
`{"aggregate_bytes": 446975, "largest_bytes": 15767, "notice": "FICTIONAL REPRESENTATION ONLY; no source/model/host/native authority", "output_count": 126}`.

Verifier2 command `python3 <canonical-ledger>/task-54-d7-resource-oracle/verify.py`, exit0, no stderr; exact stdout (641c13):
`{"mismatches": [], "negatives": 85, "positive": 28}`.

Author invocations total: **generator2 / verifier2**. No standalone invocation failed. Initial green did not include the registry-link check and is retained, not overwritten as history. No further generator/verifier run occurred. Direct-inspection/final shell checksum bookkeeping is separate from these invocation counts.

Corrected corpus contains26 source JSON examples (all six source families, with eight ClaimWork variants in each domain) and two fixed bootstrap binaries. The examples represent every named new nested record, all22 model/coverage/Requirements positions, every supported-work branch, Full/Quick Train request consumption, Reference read/recovery/snapshot, Target read with future input/retained supplied input and lock. Library/root/raw evidence order, nonUTF8 and empty argv, exit Some0/None, timeout, all three measurement kinds/scopes and empty measurements are concrete. Fictional registry source is exactly89 bytes, never a real approved experiment_matrix.json. Production-shaped inert examples are **shape examples only**, with fictional source revision/registry/allocator/kernel/evidence and no trusted launch origin or admissible model. Their false synthetic flag does not create authority.

`negative-cases.json` is the expected coverage matrix; `verification-result.json` records every expected and observed first refusal:

| Class | Count | Actual result / interpretation |
|---|---:|---|
| Six-source JSON structural/parser-stage cases |67|67 intended first refusals matched, including malformed/duplicate/Unicode/lexical JSON, unknown/missing fields, all22 ordering/rule/coverage, source/build/environment/file links, work predicates and D6a role, claim/observation links, Requirements/disk arithmetic, registry identity. |
| Bootstrap binary structural cases |9|9 intended first refusals matched for length/EOF/domain/synthetic/hash prefix/policy/origin/namespaces/profile selectors. |
| Private context / arithmetic witnesses |9|9 illustrative expectations matched: pre-existing/same-owner, preparation nonpromotion, trusted ancestry/omitted parent, U64 sum/raw cap+1, huge finite distinct from unlimited and serial-domain prerequisite. **These are NOT source-record fields, actual native controls or proof of an owner/host/lock implementation.** |

Five additional positive private witnesses cover exact8MiB cap without allocation, finite/saturated headroom, huge finite and opaque-profile semantics. No cap-sized file was allocated. Raw kernel-like interpretation is deliberately absent: generic evidence bytes explicitly say fictional. Native v1 unlimited normalization requires a supported kernel profile; the huge-finite witness only prevents a number from silently becoming None, and does not implement that profile.

Each negative is pinned by its own exact file bytes/length/hash. JSON semantic mutations enter the specific reader stage after the fixture transport identity in manifest is available; they are not advertised as complete admission-chain tests. Other package artifacts remain fixed valid prerequisites. Link negatives deliberately modify the intended FileRef; mutations to a Model/Validation itself are inspected as that record, rather than accidentally failing an unrelated containing-policy hash first. The new n85 Model has all other same literal fields; correct source registry and independently decoded legacy CaseIdentity registry form its valid prerequisite. Request checking separately verifies each exact unchanged old body after explicit domain/synthetic projection; registration bytes remain completely pinned/unmodified.

### Final direct inspection and pins

After FIX1, direct reads inspected raw registry, current Model and n85's only changed registry field, synthetic/inert bootstrap headers, plus prior complete source/branch examples. The independent shell SHA256 ledger has263 entries (all current and before-FIX1 files except itself). Direct checksum verification and byte accounting cover264 total files/**1,108,805 bytes**, within the prewritten300files/6MiB combined bound. `manifest.json` has124 exact corpus data entries (eleven raw,28 package,85 negatives); generation-result accounts for its own bookkeeping separately. The largest generator file is15,767 bytes. Initial snapshot consists of131 files and is immutable. SHA256SUMS excludes itself and this sibling report; report's final hash is returned separately.

| Current artifact | Bytes | SHA256 |
|---|---:|---|
| generate.py |13815|6ddd057046daac7a19029e9e2bc9226e5dd58a8ee2c80f0ce7cbdd3884a503a0|
| verify.py |31079|81923b11a78f22263d6cf57297110fc9f15c126d22657353d41b3f28bde05a28|
| literal-expectations.json |3233|83f70dcd1d532302e27aebe1cbb6fbd36fd4aa1b193a3a365d679e2cf6f4b56e|
| negative-cases.json |11387|1cadd3ac6e7ddf0089b3653dac86c3d2e3f0edc4d4010d36dda4bfe1eb0c0f66|
| manifest.json |15767|3cdd988685c9ee3564d8af2a58072c29d5e23a9ee833a28167dfe76180164bd4|
| generation-result.json |646|49c6537f02a39e7e8d045084cbbebd92df1707806820dcce8f525a9c13a0dfa3|
| verification-result.json |18569|ca30af6635346b63cf779696ee9a160c2787b93e4727496b3e90e76ee8c5b94a|
| SHA256SUMS |27530|16373eb4ec861e5d61d6cb0ba1c4786ff0c3160ae4ceda780a974828a4c90fcf|
| synthetic/bootstrap.bin |703|6e51201e90c76642e4f30a9e7ff4724d40c6703656c63b6fcdbc9758b00811ae|
| inert-production/bootstrap.bin |704|3d8bbb071f5804ca194f9a37f5938587a5dd7e7672f4a5e76fd11b5480502980|
| synthetic/model.json |5099|616cdb9287e0b1f1fb549b03b5560b8139c245244687af47513f2cfe46973bdb|
| synthetic/validation.json |11649|b6cbeaab5582b9e7f2eb7a3e8e47bab06fe10392805321de888f52ef23fe60dc|
| synthetic/host-policy.json |2320|166d88a45ff66a2056f8dd42cd63029627749ee7ee7ae3be8110562305504f17|
| inert-production/model.json |5101|d8e285db572a64a36c05f7f4a02059b0fa8716e6d5ab046b90ad21131f14e50f|
| inert-production/validation.json |11651|5bbdb7d845836d2395192c10ae622f2f0bf1c8ef52be4933d6206e998d66ec85|
| inert-production/host-policy.json |2322|904e995c081866785bb6b63f8c761d2e48cbf686aa94d9611b8482bfb1405191|

Raw/claim/observation/negative files all have exact lengths/SHA256 in manifest and independent SHA256SUMS. Literal expectations were authored before execution; they include independently calculated bootstrap offsets and phase/disk values. Verifier imports no generator code, and checksum collection uses separate system shasum. Independent reviewer reconstruction remains necessary; author-produced matching expectations do not grant independent acceptance.

All standalone commands and final direct-read/checksum commands exited0. Read-only search exit1/no-match and truncated listing/read outputs are preserved in this report/tool records above; no failed scientific result was rewritten. Final shell tools did not run any project helper. Outer Resource/Requested/Requirements/Observation/Claim/FileRef and Full/Quick files were not edited/regenerated. Only these private oracle artifacts and this append-only report were written; no source/Cargo/publicdocs/commit/push/subagent/project-test/--list/native/HE/SSH/host/data/process scan.

### Remaining evidence and freeze

The oracle is deliberately a finite fictional representation/check corpus, not a production parser or a generic schema framework. It does not validate actual source/build/model applicability, complete native coverage/acceptance, trusted host origin, live namespaces/cgroup ancestry, real quotas/capacity, lock/helper crash lifetime, B0 physical bound, all22 physical allocation rules or the retained D6a/actual full receipt constructions. The complete actual/supplied receipt-byte comparison remains mandatory in production and is not replaced by these distinct fictional hash-role controls. Current source d1aa3af Rc/guard FD/mount checks still require actual-model inventory. Same-owner typed materialization, five live gates and full scientific/native/GPU/external/baseline obligations remain unfinished. All21/full2680+20+7/quick268 and original holds remain unchanged.

Current files are frozen for independent corpus review; no subsequent production source begins from this author's result. Any review-requested change must preserve this submission and be recorded as a new explicit correction round.
