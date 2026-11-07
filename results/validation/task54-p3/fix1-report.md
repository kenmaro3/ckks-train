# Task54 P3 fix1 — frozen correction and evidence handoff

2026-11-07. **Operational source/validation report; not scientific results or production admission.** Root has received source/Cargo ownership back. No further source edit or Cargo execution is pending. Original reports, snapshots and raw evidence are unchanged. Independent rereview is required before integration; the original real P4/D7 factory-spy acceptance remains required and pending.

## Frozen source

| File | SHA256 |
|---|---|
| `crates/ckks-train-bench/src/study_admission.rs` | `326fd68a967ba025f3502b9d9797a20900a928562c417e94fcb735c816117356` |
| `crates/ckks-train-bench/tests/study_admission.rs` | `571f4145239230ab124364a76d0fa16d3b940d5fa8b066a7194f0e1f1d846a0c` |
| `crates/ckks-train-ckks/src/bootstrap/metadata.rs` | `e1bf9b3e7f856d0eb4a167874d05d6884b38d0054842f583b1add81c6e4eb2b8` |
| Root-owned unchanged `crates/ckks-train-bench/src/lib.rs` | `867b2c6d8f4391861956c10483fe18c91636e9272e41cceee8953b90976b3e44` |

Exact copies: `task-54-p3-fix1-validation/frozen-source/`. Exact stdlib-generated unified delta against the original four frozen copies: `task-54-p3-fix1-validation/source-delta.patch`, SHA256 `3603f8f85eb0f3ea14e7ef68df1860fcbffff9969c86da293bf0ce4360e362e4`. Only the three assigned worker files differ; lib.rs is identical. Runtime RED and final frozen source copies are preserved separately. `final-manifest.txt` pins source, copies, raw logs, delta and this report.

## Review corrections and actual closure evidence

| Review item | Correction | Discriminator and observed outcome |
|---|---|---|
| **R1: post-compose loss** | All fallible final retained-rate charging/copying and bounded key-manifest work now shares one finalization error attachment. It retains the complete transcript/index/typed summaries and all consumed successful updates, sets Finalization phase, and leaves failing stage/update absent. Work remains after compose; no artificial early preflight removes the late-failure path. | Integration RED02 reproduced completed0/prefix0 after two composed updates. Final integration reuses one real Iris positive and selects cap **1,904,016 bytes**, allowing both updates but rejecting final retention. It now reports completed2, Finalization, stage/updateNone, two complete records, exact required/limit, matching typed post-states and identical indexed record bytes versus the valid control. Bound request-limit header bytes intentionally differ. No complete short plan is returned. |
| **R2: missing operational context** | compose tracks actual current stage/update and Update/Handoff phase around index/work/byte/summary operations; existing richer P2 error fields survive. Work denial uses category-specific operation text and exact computed next/limit. A poisoned lock has distinct PlanningBudgetPoisoned and no invented numeric requirement. | Unit RED01 reproduced stageNone at stage1/update0. Final test allows stage0 update+handoff, then denies index3 with limit2: stage1/update0/Update, logical completed0 and two intact earlier records. Separate update-work exhaustion reports `stage-update calls`, required2/limit1, same context/prefix. Actual P2 calls remain1 update+1handoff; the denied reservation is not counted as a reached P2 call. |
| **R3: missing identity discriminators** | Private borrowed encoding views/helpers are used by the actual production parameter, layer, preamble, request, rate, state-label and update-input encoders. They cannot construct CkksParams, canonical inputs or admitted plans. Fresh revalidation also checks transcript-prefix/binding consistency. | Compile RED03 captured missing new views/getter. Final unit tests pass the independently assembled **321-byte** fixed structured golden (production domain/version/options plus representative parameter/graph/request/state fields); actual ordered q/p integer, N/Delta/dnum/secret, all nine BootParams, all four MetadataLimits, consumption/transpose/EvalConfig, valid micro ordering/count, stage x/y and retained weight/bias/compact/velocity encoding mutations. An index2 suffix rate changes .03→.04 outside the two-update prefix, below the unchanged separate index3 maximum .10; prefix bytes and reference remain unchanged while full identity changes. |
| **R3: trusted binding and valid partition** | Private corruption tests keep production types sealed/non-Deserialize. | One real canonical Iris plan accepts original binding; alternate two-micro10+10 partitions conserve each logical20 and successfully pass canonical partition construction, then fail original-plan identity comparison. A privately altered header with rehashed digest rejects; making privately altered binding/header mutually consistent still rejects against freshly derived trusted bytes. This is not arbitrary transported-plan authentication. |
| **R4: actual original P1 ceilings** | Narrow CKKS change stores the original fixed-size MetadataLimits on successful construction and exposes immutable Copy accessor. P3 encodes all four actual ceilings separately from resource_bounds/crypto/proof bytes. Existing P1 identity encoders, arithmetic, enforcement, counters and algorithm revision are unchanged. | Shared params/proof/default descriptor plus one descriptor with only max_peak_bytes64MiB→64MiB+1: crypto descriptor bytes are equal; P3 operational bytes differ. Getter observes actual original ceilings. Isolated shared encoder tests distinguish each of coefficients4096, interpolation2^24, polynomial work2^24 and requested bytes64MiB. No caller-supplied unverifiable limit substitutes for the getter. |
| **Q1** | validate_plan_binding rustdoc now states no updates/HE/key allocation, explicitly acknowledges metadata transcript/partition/identity allocations and trusted-local-plan scope. | Source/doc correction; no allocation-free/RSS claim. |
| **Heterogeneous destination-y** | Added private supported P2 two-stage composition: PolyRelu2/Identity followed by Square/Square under one layout, with two unequal micros. Canonical registered graphs remain unchanged. Existing schedule continuity now asserts weights, biases, compact containers, velocity, optimizer basis and fresh batch labels between every pre/post state. | All I/WN/AU/AA actual source/destination y labels differ in both level and bits; destination micros use exact destination y and returned old-hidden handoff x. No fake labels, recurrence or HeOps. Existing three-stage fixture remains as separate coverage. |

Actual heterogeneous metadata (toy P2 fixture, **not HE numerical evidence**):

| Mode | Source y `(level, scale_bits)` | Destination y `(level, scale_bits)` |
|---|---|---|
| I | `(6,4787326404607852618)` | `(5,4787326405614465397)` |
| WN | `(12,4796333603373969416)` | `(11,4796333603772423219)` |
| AU | `(12,4796333603373969416)` | `(11,4796333603772423219)` |
| AA | `(14,4796333603174743040)` | `(13,4796333603229266945)` |

The q/p/secret mutations are **encoder-field discrimination**, not constructor acceptance or altered secure parameters. Their baselines use actual existing parameter/metadata objects; changed private views never construct a CkksParams or production plan. The golden's tiny representative field projection likewise is not a parameter-admission fixture. All runtime parameter tuples remain the explicitly enumerated existing presets/fixtures: mini I `(logN10,q60+40×9,p60,scale40,dnum10)`, mini NI `(logN10,n16_boot_bits,scale42,dnum8)`, existing test_tiny, secure n16_train/n16_boot. No mutation parameter constructor, prime search sweep or synthetic canonical materializer was added.

There is no public untrusted transcript/plan decoder; unknown wire tags are not public inputs tested here. No parser was added for artificial unknown-tag coverage. Revalidation of a trusted local plan does not authenticate every possible arbitrary rehashed transition or independently reconstruct every typed summary from bytes. A future transport constructor needs separately reviewed full consistency/trust checks. Full P4 geometry/trace access and the real start/gate/factory boundary remain separate ownership.

## Executed validation and preserved failures

All Cargo used the existing target, locked/offline, CARGO_INCREMENTAL=0, jobs8, Rayon8, one owner, test-threads1/nocapture, and checked disk free >=805306368 bytes. No automatic runtime retry occurred.

| Log in `task-54-p3-fix1-validation/` | Outcome |
|---|---|
| `01-r2-red.log` | Expected runtime RED: stageNone versusSome(1), after actual first update+handoff. Source snapshots retained. |
| `02-r1-red.log` | Expected runtime RED: completed0 versus2 at real canonical post-compose retention cap. Source snapshots retained. |
| `03-api-red.log` | Compile-only RED for missing new private field encoders/original-limits accessor; no fixture executed. Source snapshots retained. |
| `04-bench-clippy.log` | Preliminary targeted bench Clippy PASS before final test additions. |
| `05-ckks-clippy.log` | Targeted CKKS lib Clippy PASS on final CKKS bytes. |
| `06-bench-clippy.log` | Targeted bench lib + owned integration target, no-deps, -D warnings PASS on final bytes. |
| `07-format.log` | Owned three-file Rust2024 formatting check PASS on final bytes. |
| `08-frozen-unit-green.log` | **11 unit functions PASS**, including original5 and six fix discriminators; all four before/after pins match. |
| `09-frozen-integration-green.log` | **5 integration functions PASS**, original4 plus R1 discriminator; all four before/after pins match. |

Final GREEN commands are exactly the release's `cargo test ... --lib study_admission::tests::p3_ -- --test-threads=1 --nocapture` and `cargo test ... --test study_admission -- --test-threads=1 --nocapture`, once each. No old P1/P2 suite, bootstrap/runtime helper matrix, broad workspace/release or HE test ran. The existing nonempty retained-payload helper test still observes3 transcript+2256 metadata=2259 charged bytes and exact cap/one-byte-over behavior. The canonical late cap is newly exercised; no total-RSS or full-Fashion planning fit is inferred.

## Actual work accounting

| Resource | RED unit / integration | GREEN unit / integration | Fix1 total | Cumulative including prior12/6/8… |
|---|---:|---:|---:|---:|
| CkksParams |2 /2|4 /3|11|23|
| DftSupportProof |1 /1|1 /2|5|11|
| BootstrapMetadata |1 /1|2 /3|7|15|
| Bundled Iris materializations |0 /1|1 /1|3|5|
| Geometry and initial-state calls, each |3 /2|38 /148|191|527|
| Actual stage-update calls |1 /4|84 /6|95|243|
| Actual handoff calls |1 /0|30 /0|31|79|
| Production wrapper calls |0 /2|1 /5|8|14|
| HE/key/Encoder/evaluator/plaintext training |0 /0|0 /0|0|0|

GREEN unit accounting is shared fixture34 geometry/81 updates/29 handoffs, plus isolated denied-work fixture3/1/1 and private canonical fixture1/2/0. GREEN integration is shared147 geometry/6 updates/5 wrappers plus dedicated getter1 geometry. Source logs retain category-specific denial versus actual reached-call counters. All totals are within fix1's released maxima; no use of unused allowance for additional cases occurred. Compiler-only RED and static checks constructed nothing. Existing process-local params/proofs were reused; only the explicitly required original-limit descriptor was added to its shared fixture.

## Remaining limits

Independent rereview must evaluate these corrections; this report does not self-approve them. All21×4 candidate records and broader2,680+20+7 full /268 quick inventories remain intact. Full intended consumed schedules still either complete or return failure with a retained complete prefix; they are never silently shortened. Unknown planning/RSS/HE key/workspace capacity, range/precision/security, actual runtime/provenance/target/start gates and full-profile feasibility remain unproved. Full encrypted-block holds remain.

The actual reachable P4/D7 pre-key spy test is still required: genuine other gates/start, reachable nonallocating positive control, corrupt identity/parameters/partition negatives before the real factory. No fake Ready/start/public callback/decoder was introduced. P4 typed StageGeometry/full-trace comparison access remains explicitly separate; no P4 source was changed.

No Git, network/SSH, remote/cloud GPU, dependency, dataset download/export adoption, scientific JSON, HE or campaign action occurred. Only assigned source and operational evidence files were written; unowned `.vscode/` and `results/mnist_blocks_*_plain_b48.json` were untouched.
