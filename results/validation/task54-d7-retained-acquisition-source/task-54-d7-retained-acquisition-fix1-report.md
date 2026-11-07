# Retained acquisition FIX1 — source/static report

**DONE_WITH_CONCERNS: I1 source fix implemented; independent scoped re-review required. Runtime remains unreleased.** No test function/native/--list/helper/registry/materializer/SSH/HE execution, subagent, commit/push, fixture/generator change, public document or scientific JSON write occurred. Original source report/review/full diff/pins and all original32 logs remain byte-identical; they are historical pre-fix evidence, not results for this source.

## Finding and narrow correction

Read the FIX1 release, original source release and ROOT FULL READ review `c3e8bf0205e4c6f4bc11451129ae143333abf368e5dc99620f434715860e6279`. The Important/P2 I1 is valid: preserving only the short store object's address allows a different store value in that same place to capture a replacement guard and satisfy its own guard check.

The fix changes only4 source paths:

- `study_provenance/publication.rs`: each actual TargetStore owns one private Rc<TargetStoreIdentity> allocation containing its actual original guard inode association, plus the actual original open guard File. `from_opened` is private and is used by the existing actual store open path; it takes the existing actual opened File, creates one fresh identity allocation and performs the real store recheck. It supplies no public constructor or authority layer.
- The private non-Clone/non-serializable TargetReadOrigin retains that same allocation and a try_clone of the original guard FD. It holds no publication flock. `retained_origin` rechecks the actual store before this transfer. The actual `require_same_store_owner` comparator uses Rc::ptr_eq and the immutable original guard association. A retained snapshot keeps the allocation alive, so another store cannot reuse that identity even after the short original store is dropped. Keeping the actual guard File alive also prevents its inode from being reclaimed/reused behind a tuple-only guard check.
- `TargetStore::recheck` now validates its held original guard File/mode/owner/zero length/nlink1, original inode association and mount, then validates the current fixed .profile-guard entry against that original identity. `recheck_read` takes the original TargetReadOrigin and checks original owner/guard before AND after retained lock-byte/ancestor checks. A new store's agreement with its new guard does not satisfy this comparison.
- `study_reference/target.rs`: the actual read_target captures this origin before retained lock acquisition and transfers it into VerifiedTargetForSeed. Its C1 adapter retains the existing original ProfileTargetStore address check and actual context-instance/address/ProcessStamp checks, and passes the retained original store/guard origin into every target recheck. Both current and selected prior still compare against one verified target/current policy. Existing shorter-store and independent context/input lifetime compile obligations remain.
- `study_provenance/publication_anonymous_tests.rs`: the one existing synthetic TargetStore fixture now calls the same private actual from_opened helper with its actual opened guard File, instead of manually initializing incomplete store fields. Existing ignored native entrypoints, --ignored child selection,70/71 checks and anonymous C/W/F/L/D publisher behavior are unchanged.
- `study_provenance/publication/retained.rs`: adds one separately ignored Linux native control of actual guard/store replacement, detailed below. No production retained read/size/typed parser/Quick schema code changes occur in that file.

Original store address/context checks, independent lifetimes, no public Clone/serde/Send/Sync capability construction, original same-FD immutable-byte custody and original error mappings remain. All new types/functions are private or crate-private closed ownership projections. Rc uses only safe existing std APIs; no dependency, unsafe operation or license change. The original guard File is unflocked: fresh ProfileGuard acquisition/release remains unchanged, and this fix does not hold flock through HE.

## Tests first and static evidence

Logs are exclusively under `task-54-d7-retained-acquisition-fix1-artifacts/logs/`, each with exact command/output/EXIT. No earlier log was appended or overwritten.

| Log | Command/result |
|---|---|
| red01.log | Four new pure regression controls added first. cargo test -p ckks-train-bench --lib --no-run exited101 with7 focused errors: missing private TargetStoreIdentity type/constructor and actual require_same_store_owner comparator. No unrelated pre-existing error was used as RED. |
| fmt01.log | cargo fmt --all, exit0 |
| green01.log | cargo test -p ckks-train-bench --lib --all-features --no-run, exit0; compiled only |
| host-clippy01.log | cargo clippy -p ckks-train-bench --all-targets --all-features -- -D warnings, exit0 |
| linux-clippy01.log | same with --target x86_64-unknown-linux-gnu, exit0; actual Linux cfg/all-target test typing, no Linux linking/native execution |
| fmt-check01.log | cargo fmt --all -- --check, exit0 |
| diff-check01.log | git diff --check, exit0 |

Relevant all-feature no-run and strict host/Linux checks cover this mode-independent owner/guard change. Earlier default/parallel/cuda no-run checks remain historical; this mode-independent fix uses final all-feature lib no-run plus host/Linux all-target strict checking. No unrelated broad checks, test executable or --list ran. Static compilation is not behavioral PASS or native guard-continuity evidence.

## Exact new test filters and work inventory

Future commands require a separate root bounded runtime release and exact1-pass/nonzero-test verification. None is currently released/executed.

Four pure lib tests use the actual production private comparator. Exact prefix is `study_provenance::publication::retained_tests::`:

1. `retained_store_identity_original_owner_accepts`: one private identity allocation, one retained Rc clone; same original owner accepts.
2. `retained_store_identity_equal_metadata_distinct_owner_refuses`: two distinct identity allocations with equal guard metadata; replacement refuses.
3. `retained_store_identity_same_address_replacement_refuses`: retains old identity, replaces the Rc in the SAME variable/place, asserts its variable address remains the same, then requires owner refusal. Both live allocations have equal metadata, so neither address comparison nor metadata equality explains the refusal.
4. `retained_store_identity_survives_original_store_drop`: retained snapshot plus Weak prove the original identity remains live after original owner variable drop; actual comparator accepts retained original and refuses a distinct equal-metadata owner; dropping the retained strong references makes Weak upgrade fail. Weak bookkeeping lives until test end; there is no leaked allocation or global counter mutation.

These4 controls perform7 small Rc allocation events in total (1+2+2+2; peak2 per test), metadata is two u64 values, Rc clone/downgrade/upgrade do not allocate new objects, and stack pointer comparisons do not dereference stale pointers. All four have filesystem/process/helper/context/store/guard reader/registry/materializer/parameter/P1/planner/preparation/crypto/capacity/authority/HE call counts0. They prove the private comparator/retention behavior only, not native source custody or live capability construction. The two earlier compile-only lifetime obligations still compile and remain uninvoked.

New native exact filter:

`study_provenance::publication::retained::tests::retained_native_original_guard_and_same_address_store_replacement_refuse`

This is cfg Linux and ignored by default, requiring the unchanged separately authorized safe-ancestor/private0700/unprivileged/ext4-or-XFS scratch contract and an explicit future --ignored selection. It adds1 scratch namespace (aggregate9 for the5 retained-native entrypoints), one4-byte target.lock.json raw acquisition and zero-byte original/replacement guards. It constructs2 actual private TargetStores from actual held anchors/guard Files; captures1 retained origin, performs an initial successful actual recheck, renames the original guard and creates a replacement, and requires the original store to refuse. It then replaces the store in the same variable/address, shows the replacement can pass its OWN new-guard recheck, but must fail when checked against the retained ORIGINAL owner/guard. After store drop it verifies original guard FD identity survives and the unchanged held target itself still rechecks. This distinguishes I1 from bad lock bytes/ancestor failure. It has1 in-process actor, no child/fork/thread, no actual context/registry/materializer/parameters/HE. Actual filesystem/proc/mount reads and owned-scratch mutations are real work, not pure or currently executed. It is not a native durability/power-loss claim.

Final source now contains19 retained-package test entrypoints:14 portable (10 original+4 fix) and5 ignored Linux native (4 original+1 fix). Executed counts remain0 for every one. The previous runtime1 candidate remains released:false and pins PRE-FIX source only; this report does not modify/release it.

## Real owner/guard resource delta

Each actual TargetStore adds one Rc identity allocation (two-u64 identity payload plus Rc bookkeeping/alignment), and retains the guard File that previously closed after open. Each retained target origin increments that same allocation's strong count and duplicates1 original guard FD; it does not allocate another identity or copy a raw byte buffer. Dropping the original short store closes its own guard FD/decrements Rc, while an owned target retains its independent guard FD and allocation. RAII drops them on all later failure/drop paths. These FDs/allocator costs must be included in the mandatory physical/resource model; no memory/FD capacity PASS follows.

Each ordinary TargetStore::recheck additionally validates/mount-checks its held original guard. Each recheck_origin mount-checks its captured original guard and performs the actual store recheck; recheck_read does this before/after bytes. Actual read_target adds1 retained_origin store recheck and1 guard-FD duplication before its existing retained lock acquisition. These add real metadata/open/proc fdinfo≤16KiB/mountinfo≤8MiB work, without changing the original2 additional materializers,5 derive_identities,5 context rechecks/10 direct helpers+2N input helpers already inventoried in the original report. Shared legacy target caller operations receive the stronger guard check/retention too. Their new original-FD/Rc costs are not silently called zero. File publication operations/order/no-clobber/recovery and shared child counts are unchanged; native shared-caller observations remain pending.

## Exact final pins and preservation

BASE remains `71d3f0937741a4ff8d156b36540332ea82203d3c`. `fix1-artifacts/before/` contains exact pre-fix snapshots of all4 changed source files; before-pins.json records them. The other9 original source paths remain byte-identical. Complete final13-source rows retain base hashes and are recorded in new pins.json, with source-set SHA256:

`272494592fc44952584eb608d9836f94b8d2ea82c60a750539534402c1281c94`.

- `scoped.diff`:11,190 bytes, SHA256 `51082f09fc3a5bb71c200db032069e44a99642a94b23f1f86950af667483e35a`; all4 FIX1 source changes against saved pre-fix files.
- `full.diff`:65,800 bytes, SHA256 `a155ebefa1c20b7455f1da0ad2d0af51a3849c86cc641384d751f76f5132158b`; complete final source against BASE including the entire untracked new retained.rs.
- `pins.json`:28,689 bytes, SHA256 `9e3ab8601a591d847831e9af6a52a2102def5462cb11527a90db6569cb2f4f89`.

New pins record final13 source files,4 pre-fix snapshots,7 new logs, new release and artifact hashes, and84 preserved prior input/log/report/review/pin/diff entries. Original report44e4b16e, source-reviewc3e8bf02, pins3aba995d, full.diff63f03010,48 old inputs and32 original logs were independently rehashed and match; no historical evidence was relabeled or rewritten. This package updates only source and NEW fix artifacts/report. Unowned .vscode and the2 MNIST result files are untouched.

## Remaining scope and next action

Return this candidate to independent I1 scoped SPEC/QUALITY re-review before root releases any runtime. All original source-map boundaries remain: mandatory SAME retained bytes through actual typed materialization without relevant path reopens, genuine validated physical resource producer/current capacity before acquisition, exact correctness/security/layout applicability, trusted root-current authority channel, five sealed live gates, guarded Full/Quick attempts/sidecars/durable lifecycle, consuming begin and scientific publisher remain incomplete. Actual Linux native guard/fork/mount/shared publisher proof and all scientific execution remain separately pending.

Task54 remains unfinished across all21 profiles, Full2680+20+7 and Quick268, original baseline/GPU/external/Task5 requirements and encrypted-block hold. This fix proves no resource fit, live gate, native custody, HE outcome, matrix coverage or scientific acceptance.
