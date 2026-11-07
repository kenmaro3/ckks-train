# D7 execution-binding design fix1 report

2026-11-07. Original-author design fix round1/5. Addresses Important I1/I2 from task-54-d7-execution-binding-design-review.md, verified SHA256 `6fd9e31175b729012660c9aa22dcc31e22ceac713c54b7625eea6876af960f0d`. This report claims design changes and independent literal artifact checks only; no design ratification, implementation readiness, gate clearance or execution authority.

## Frozen inputs and outputs

All paths are relative to this ledger directory.

| Artifact | SHA256 |
|---|---|
| task-54-d7-execution-binding-design-proposal-before-fix1.md | `25d88851549d4ad85989572b7a541745dd2f240545f434b490bae7fb0cedebe0` |
| task-54-d7-execution-binding-design-proposal.md | `0b4c92c2f3723589ec9006089e0999255b6b209b8286ef097e4aee74abe289fe` |
| task-54-d7-execution-binding-design-vectors/SHA256SUMS | `899b45991113581638a5758bd9bdaec441df768e771fa706a16549faf93b287e` |
| task-54-d7-execution-binding-design-vectors/generate.py | `8e1bdbea566ebc051a3b3c36d07d16406bb2fbc75f4a62c1d3a5374aa979356c` |
| task-54-d7-execution-binding-design-vectors/inputs.json | `1f30a3c7665fa184ed0b0e971343c0770f51eabcbb26c5984c934ab312c05b85` |
| task-54-d7-execution-binding-design-vectors/manifest.json | `6b983e7d31016756ed5906be1558e0eebcd923d37f5b3f8332dd2e28428e96d7` |
| task-54-d7-execution-binding-design-vectors/negative-vectors.json | `d8c2fcf1e4abf13b271e4a0c1e48af4bbc45506aecc946c5a9e14ed5f7a16a70` |

The original proposal was copied byte-for-byte and its25d hash checked BEFORE editing. Vector SHA256SUMS inventories19 files (14 full .hex files plus generator, input/manifest/negative JSON and README); the checksum file itself is pinned above. This is an artifact inventory, not an evidence/capability hash or scientific schema.

## Important I1 — exact requested resolution and changes

Review requirement, verbatim:

> **Required resolution before continuation source:** give a small explicit signature/ownership sketch for the returned driver and its fixed consuming execution method. Identify what `HeSession` owns, what it borrows, when the checked route is consumed, and which nested frame owns the evaluation context, refresher and facade while P4/P5 run. A returned owned driver with a private fixed method that creates/runs/drops the borrowed session is one possible resolution; it must not expose an allocator permit or public callback. The sketch must retain ProcessStamp, Train context/input and gate/resource leases through each allocation/execution entry, and release the store borrow only after the ratified checked transfer. No unsafe lifetime extension, leaked owner or fake StartedHeCase may close this gap.

Proposal §3 now preserves the exact `begin_checked_he<'s,'c:'s,'p:'s>(StartedHeCase<'s,'c,'p>) -> HeSession<'c,'p>` boundary and adds private consuming `execute_cpu(self)`. Returned HeSession owns a live frame and non-self-referential owned crypto/initial packets. The live frame owns stamps, genuine gates/resource-user leases, links/threshold, descriptors and plan Arc; it separately borrows Train context and canonical input/parameter owners. Context retains runtime/pool. Existing CheckedMaterializedInput can borrow context for `'c` independently of input `'p`; the single-lifetime TrainCapture remains short-lived. No invented `'c:'p` relation or unsafe extension is needed.

The store borrow and handles survive exact start/context/owner/gate checks and transfer into the live frame. Only then may handles/store borrowing end. Keys and initial ciphertext setup occur behind another immediate live check; construction failure consumes authority. The returned driver owns client/evaluation/initial state without internal borrows. Its fixed execution method creates refresher/facade/P4 or P5 session in a nested local frame, executes once, drops borrowed objects before their owners, and retains live leases through all allocations/updates. run_owned consumes the driver inside its installed executor frame; no live object leaves through a Send closure return. Existing poison/partial-counter and P5 success-only snapshot restrictions remain.

The exact `study_admission::validate_plan_binding(materialized, registered, &plan, actual_planning_parameters, request)` call is now required AFTER durable start and BEFORE keys. Its temporary canonical_binding transcript/partitions/identity/row allocations receive explicit peak-workspace accounting under the existing request limits, released at actual scope end. It does not consume additional P2 geometry/update calls, and it is not replaced by retaining CheckedPreparation. The actual start/gates still do not exist; disconnected fixed helpers are permitted, fake token/driver construction and production readiness are not.

## Important I2 — exact requested resolution and frozen coverage

Review requirement, verbatim:

> **Required resolution before hash-producing source:** freeze and independently review complete literal vectors with input values, exact canonical hex, byte count and SHA256. Cover each new domain and every structural branch needed by its format: I parameter main-weight option; layout compact absent/present, multi-stage and partial partitions, negative ordered transpose steps; key Dedicated/Reuse and I/NI generated-versus-executed/bootstrap-option branches; security I/NI with ephemeral option and fixed-weight tag. Synthetic branch examples may be explicitly unsupported by current production policy; never label them canonical secure-profile evidence. Include exact expected failure cases for changed tags/order/lengths, invalid counts/weights, overflow and A_MAX. Existing P1/P3/P/R vectors stay unchanged. This is a representation prerequisite, not a new scientific/user approval gate.

Proposal §4 now explicitly requires this independent vector package to be frozen/reviewed BEFORE hash-producing source. All14 complete positive transport examples have full input fields, hex, length and SHA256:

| Domain | Vectors | Branch coverage |
|---|---:|---|
| I parameters | 2 | main weight None/Some(3); exact scale bits and ordered prime/bit lists |
| Physical layout | 3 | Full/Quick; I/NI;1/2/3 stages; compact None/Some; container layer None/Some; full and partial micros; empty/nonempty ordered chains including negative I32; Bool0/1 |
| Key policy | 4 | I/NI × Dedicated/Reuse; bootstrap None/Some; generated versus execution sets; bootstrap conjugation/SSE/separate-bundle false/true; exact reuse order |
| Security | 5 | I/NI; Dense/Fixed main; bootstrap/scope None/Some; ephemeral None/Some(Fixed)/Some(Dense); empty versus all nine ordered limitations |

All examples are transport-only: the a/b hashes are literal placeholders and geometry is handwritten, never a derived StageGeometry. Fixed main and non-default ephemeral specimens are not registered production settings. Some(DenseTernary) ephemeral is a grammar branch that current P1's ephemeral_hw mapping does not produce. A byte oracle is not a live owned producer.

56 complete negative vectors include actual input hex, length and digest for changed domains/tags/text lengths; impossible list counts; zero/over-N weights; invalid dnum/logN/scale; reordered bit vectors/transpose chains/reuse pools; wrong generated/executed sets; exact-union superset/duplicate/zero rotations; missing bootstrap links; unsupported assurance/scope; reordered/omitted limitations; uppercase hashes; trailing bytes. Expected outcomes distinguish malformed-wire refusal from well-formed semantic/owner mismatch, including recomputed outer digests. Additional precise families specify every proper prefix of every positive vector, total-size guard inputs A_MAX−1/A_MAX/A_MAX+1/u64::MAX, and overflowing add/multiply/usize conversion. Size-guard acceptance alone does not establish a valid document. No oversized filler files or new scientific cap were introduced.

The14 positive wire images total6,314 bytes; largest1,582 bytes. The19 inventoried vector artifacts total175,352 bytes. Python stdlib generation was rerun and all artifact files reproduced byte-for-byte. All14 positive and56 negative lengths/digests were checked independently against saved hex. Two full parameter/security bodies were additionally joined using literal bytes and int.to_bytes without importing the generator, and matched the saved images. These are artifact checks, not Rust codec tests or semantic acceptance results. No old P1/P3/P/R file was edited.

## Minor carry-forward and exact text delta

M1 remains a bounded source-brief accounting clarification, now explicitly linked to root's task-54-d7-execution-binding-accounting-source-notes.md: one additive owner seeded once with old plan requested-payload charge, explicit retained/transient requested sizes before copies, capacities merely diagnostic, separate materialization/P1/target/resource coverage. No existing ReceiptLimits bytes/meaning are changed. M2 uses existing artifact_context().source_commit(); only a current-Train-receipt source-manifest-byte view may be added. These are carried clarifications, not separate fix-loop requirements.

Relative to frozen25d baseline, changes are: fix1/historical source-status qualification; replace redundant commit accessor; replace ambiguous continuation paragraph with I1 returned-driver/lifetime/store-transfer/revalidation order; add I2 frozen vector package and before-source sequencing; carry M1 requested-payload convention/root-note link. Four domain strings, exact field tables/tag mappings and original P1/P3/P/R bytes are unchanged.

For the earlier root question, interim e3854481 →25d88851 contained EXACTLY three textual additions: (1) keep canonical_arrays, encryption and fixed continuation helpers in study_client.rs/blocks.rs, exposing only private borrowed projections/fixed calls; (2) fixed secret weights positive and<=N and equal actual declarations; (3)2S geometry/2S initial/S*U update/(S−1)*U handoff/one-wrapper successful preparation inventory, with repeated/rejected calls and encoding/allocation work separately bounded. Those additions remain in the fixed design.

Only report/vector artifacts were written. No source/Cargo/tests/constructors/native/HE/network/Git/subagent operations were performed. Independent design re-review remains required. All existing gates, durable admission/start integration, scientific output and full encrypted-block user hold remain; all21 profiles/full2680+20+7/quick268 and unsupported adapters remain visible.
