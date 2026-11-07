# Task54 D7 gate schema independent SPEC + QUALITY review

Reviewed 2026-11-07. Design/oracle review only; no implementation or runtime authority.

**SPEC: CHANGES REQUESTED. QUALITY: CHANGES REQUESTED.** One Important finding (I1), no Critical or additional Important findings. This is a bounded representation correction, not NEEDS_CONTEXT or a new scientific/authorization decision. Preserve the accepted five-capability architecture and current task authority.

## Reviewed scope and exact identities

Read `task-54-brief.md` first, then `task-54-d7-gate-schema-design-brief.md`. Reviewed the complete proposal, generator, independent verifier and README; inspected the complete literal input ledger through independent primitive reconstruction, and checked every byte of all five complete added-file diff hunks against the retained files. Normative comparisons used `docs/study-gates-design.md` §§2–6, frozen `docs/study-target-design.md` §§3–5, exact archived FIX1 §§3–5/8–10 and custody proposal §§3–9. Read current `study_admission.rs::encode_request` and relevant actual Train argv capture/frontend source; no source changes.

- Proposal SHA256: `3c0f489a1129cffbd37894daa0e2cb35d38328fb1fe3f43eeba90f68624767c3`.
- Complete review diff SHA256: `359dbb42b74a0ef234a45853dfc73cfbf422714bcc0e0f914f0cb94f2e7f0659`.
- Vector manifest SHA256: `8dd99cc4f068b2f3209bd497a06cc31a772227a5f49d6ae1111bfb7d30ff88bb`.
- Generator SHA256: `7a59795c065c953f286ae2e61844eb304a8adb92521585fc05cb6a426d0bc198`.

## I1 — Important: Quick preflight must preserve actual Unix argv bytes

Pointers: proposal line120 (`argv:List<Text4096>`) and line126 (actual checked capture argv); vector `generate.py` lines507–509 (`r.text`); `docs/study-target-design.md` lines411–417 (existing Unix-byte vectors, 4096 bytes each); `crates/ckks-train-bench/src/study_provenance/train_capture.rs` lines24/33/250–251 (`Vec<Vec<u8>>`, `args_os`, retained byte-slice accessor); `crates/ckks-train-bench/src/study_run.rs` lines72–80 and123–126 (byte transport bounds and lossless receipt `PathBuf`).

The new table cannot encode every invocation accepted and retained by the existing implementation. Unix argument/path bytes need not be valid UTF8; in particular an existing receipt path or argv0 may contain an invalid UTF8 byte. `Text` either rejects such an already-supported invocation, or a lossy conversion changes the captured provenance. Neither preserves the actual checked argv promised by the proposal. All current literal arguments are ASCII, so primitive reconstruction and existing link checks cannot expose this mismatch.

Required fix before schema ratification/source: encode argv as `List<Bytes>` with the unchanged count≤256, each decoded argument≤4096 bytes, existing NUL rejection and exact captured ordering. Do not add UTF8-only admission as a workaround. Specify lossless byte equality to the retained capture, keep the ordinary canonical/raw caps, and update literal input types/generator/docs accordingly. Add at least one explicit nonUTF8 argument positive representation vector plus a byte-tampering mismatch negative specification/vector. Refresh every dependent Quick event whole-file/payload link and corresponding manifest/report/checksum identity as applicable; unchanged ASCII canonical bytes may legitimately retain their hash. Do not modify old Full vectors. This fix needs no source/runtime expansion or user confirmation.

## Other SPEC conclusions

1. Header/domain/synthetic separation, ordered common prefix, mandatory closed Full and Quick bodies, and all five typed tails are concrete. Full preserves mandatory verified target/reference/threshold and independent selected receipt; Quick has separate operational identity, seed0/two ordered updates, exact request/attempt, and registration-only Full provenance. Zero numeric threshold is not presented as origin authority. Frozen CaseIdentity/ExecutionBinding/Gates remain unchanged.
2. P3 request body order agrees with current `encode_request`: consumption, optional lanes, transpose/reuse I32 list, three evaluation booleans, receipt limits, five work limits. Consumption and Quick identity bind the retained request/receipt/trainer without embedding the giant Full receipt. The directed dependency from request/consumption to identity, evidence and sequential events has no hash cycle.
3. Implementation tails retain actual source/revision/parameters/keys/security/layout/trainer/runtime applicability, accepted prerequisite reports and construction obligations separately. Unknown report formats, missing source equivalence and absent applicability remain missing evidence. Historical evidence and source rename mapping are not treated as prospective acceptance. No new numerical scientific threshold or policy-document pin appears.
4. Resource representation separates requested payload, validated physical model, current capacity and cooperative claim. Exactly22 components, resident/additional accounting and four-phase checked sums are explicit. Unknown terms cannot be zero or guessed component21 headroom. Incremental requirement is compared with capacity already reflecting resident charges, with conservative accounting if residency cannot be justified. CPU-only observer support is explicit; no false GPU/RSS/resource-fit claim follows. Physical-model completeness and actual observation remain future genuine evidence obligations.
5. User evidence records existing exact scope, holds, retry and overlap with root-controlled currentness updates. Fresh current-entry reads, atomic replacement/race rejection and source-generation checks are concrete; stale immutable sidecars cannot stand in for current authority. The root-origin launcher/update channel must actually exist, otherwise missing-user-evidence. No new approval service or confirmation is introduced.
6. External FileRefs are identities checked against expected inventories, not artifact-selected opening instructions. Held FD/ancestor/current-context custody and immutable snapshots remain distinct from live owner/lease rechecks. Scope parsing, hashes, synthetic controls and recovered records cannot construct capabilities. Ratified sidecar order, attempt eligibility, Full wire and Quick Initial-only lifecycle are preserved.
7. A=2,097,152 and C/R/Full limits remain unchanged. Quick K=1,048,000 plus the compact envelope fits A, while raw whitespace and all whole-file bounds still independently reject excess. Lists/counts/fixed nested shape and aggregate cap checks precede allocation. Oversized evidence remains unavailable; no truncation or cap increase is proposed.
8. All eight FIX1 exact branch test names are retained, with explicit fixture/call counts. The target-free case, six all/mixed Quick refusals, matching-registration cross-scope refusal, seed/order/partition/request/attempt variations, independent receipt/target mismatch, and old Full preservation are addressed. Pure fixture calls and future compile-fail checks are correctly distinguished from genuine constructors. The extra closed-tag, identity, evidence, link, cap, placeholder, resource and currentness mutations are explicit future source obligations.

## Independent QUALITY checks actually performed

Executed one read-only inline Python stdlib inspection command, exit0, with no import or execution of either supplied script. It independently reconstructed each primitive from retained literal values; checked exact offsets and lengths, every binary/hex pair, raw SHA256 identities, all referenced fictional retained file bytes, resource arithmetic, all Quick event links and concrete mutation changed ranges. This also read/hash-verified old Full files against the preexisting frozen golden manifest. Generator/verifier scripts and old tests were not rerun.

| Check | Observed |
|---|---:|
| Complete diff added-file contents equal retained bytes | 5 files |
| Literal records / primitive spans | 25 / 1,134 |
| Manifest file hashes and lengths | 129 |
| SHA256SUMS entries | 133 |
| Exact retained FileRef length/hash checks | 50 |
| Concrete mutation files changed only specified byte range | 21 |
| Unchanged preexisting Full golden files | 78 |
| Quick preflight/start/end exact link sets | 3 |

Both fictional resource vectors independently recompute peak1995 and incremental1935 bytes, demonstrating the60-byte resident distinction without making a real capacity claim. Quick canonical/raw byte counts are preflight3369/7005, start802/1863 and end954/2163. Production compact raw lengths at K, computed independently from envelopes, are respectively2,096,269;2,096,261;2,096,257, all within A. No cap-sized allocation was needed.

The fourteen core vectors are present and explicitly synthetic. Retained correctness/security/model/contract/receipt text stubs are intentionally fictional byte sources, not accepted evidence-reader schemas or positive genuine gates. The package accurately reports that semantic rejection, exact live applicability, actual capability assembly and native durability have not run. The21 concrete negative files and16 additional specifications are useful oracle/test-plan inputs, but independent range/hash checks do not turn them into executed negative parser tests. The supplied verifier independently checks primitive consistency but derives its inputs from the generator-produced ledger; this review additionally checked the normative tables, actual P3 source and live argv type, which exposed I1.

Remaining production source/evidence/native obligations are already explicit in the proposal and ratified contracts; they are not additional findings or permission stops. No Cargo/Rust/HE/native/SSH/subagents, source/public-doc edits, commits, external research, real parameters/P1/materialization/registry/observer calls, or scientific outputs were performed. Only this review file was written. Keep all21 profiles, full2680+20+7, quick268, current holds, original security/baseline/Task5/license boundaries and accepted authority unchanged.
