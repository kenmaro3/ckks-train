# Retained acquisition FIX1 — independent scoped re-review

2026-11-08 JST. Scope: original Important/P2 I1 closure and breakage introduced by its four-file fix. The original complete review remains immutable and supplies the unchanged baseline analysis. This reviewer wrote only this report; no source/Cargo/doc/fixture modification, compilation, test/--list/helper/registry/materializer/parameter/native/SSH execution, generator, subagent, commit or push occurred.

## Verdict

**SPEC: PASS. QUALITY: PASS. I1: ADDRESSED at source/static scope. No new findings.** Original store-address reuse no longer suffices: the actual retained target must match a retained per-store identity allocation and original guard FD association. The original source review's REQUEST CHANGES is resolved by this scoped fix plus the unchanged baseline, not erased or relabeled.

The final source is suitable for root's separately bounded **14 portable test** release after final pin verification. This review does not execute or automatically release them. Runtime count remains0; runtime1 is still released:false with pre-fix pins. Native custody/guard/fork/mount support and all live gate/scientific completion remain unproved. SAME-byte typed materialization and a genuine pre-acquisition resource producer remain mandatory next implementation.

## Exact boundary and preservation

Read the complete FIX1 release/report/scoped diff, original I1 and its required remedy, changed production/test functions and surrounding actual callers. Independently verified:

| Artifact | SHA256 |
|---|---|
| FIX1 release | `c84d039bb9b705997f0268a468732d8969e610a24ac8f5fa56a69280fc08f361` |
| FIX1 report (13,150 bytes) | `7c10d3ac7d910ef23cd789c229efeec99f2a74de558e408ac698d5e3130fa8af` |
| Scoped diff (11,190 bytes) | `51082f09fc3a5bb71c200db032069e44a99642a94b23f1f86950af667483e35a` |
| Complete final diff (65,800 bytes) | `a155ebefa1c20b7455f1da0ad2d0af51a3849c86cc641384d751f76f5132158b` |
| Final pins.json (28,689 bytes) | `9e3ab8601a591d847831e9af6a52a2102def5462cb11527a90db6569cb2f4f89` |
| Final13-source set, independently recomputed compact sorted-key JSON hash | `272494592fc44952584eb608d9836f94b8d2ea82c60a750539534402c1281c94` |

All final13 source rows,84 preserved rows,1 fix input,3 artifact rows and7 new log rows match recorded length/SHA256. All4 pre-fix snapshots match the corresponding original source hashes; the other9 source paths are unchanged. Every non-null BASE source SHA matches `git show 71d3f0937741a4ff8d156b36540332ea82203d3c:path`. Original report `44e4b16ead46170a0431106abf164bc470ff7e1219811dcaa61459b245de58b1`, source review `c3e8bf0205e4c6f4bc11451129ae143333abf368e5dc99620f434715860e6279`, pins `3aba995da8b88d44acf71d2af9e5db06c8535c4c092a20f37a1c4953806ca258`, diff `63f030103c806e11763d78ab54ac3a6910fd7c70fb0039d509fb583cb5e2526a`,48 prior inputs and32 old logs are preserved. The final full diff/pins identify the new retained.rs too; no omitted untracked-source acceptance.

## Why I1 is closed

1. **Non-reusable owner identity:** publication.rs:638–664 adds the actual TargetStore's private Rc<TargetStoreIdentity> and held original guard File, and the closed TargetReadOrigin carrying that same allocation plus cloned original guard FD. The comparator at :650 uses Rc::ptr_eq; equal guard metadata and equal outer store address are insufficient. A live snapshot keeps the original allocation allocated even after the original short store drops, so a subsequent store cannot acquire its identity by address/allocation reuse.
2. **Actual origin construction:** publication.rs:763–780 creates the token only from the existing opened store/guard path, validates the actual store, and transfers Rc plus try_clone of the held guard. The three call sites are the actual open path and two private native test fixture paths. No public reconstruction, arbitrary FileRef/cap/pass flag or deserializable capability was added.
3. **Original guard continuity:** publication.rs:782–812 compares the captured guard inode to the captured owner and actual store, validates its regular0600/current-owner/zero-length/nlink1 policy, checks its mount, then checks the actual store's held guard and current fixed `.profile-guard` entry. Retention keeps the original inode alive. A replacement store agreeing with its newly captured guard cannot satisfy the retained original owner comparison.
4. **Before and after bytes:** publication.rs:837–840 checks that origin before and after the held lock/ancestor/namespace/byte checks. target.rs:345–348 captures origin before lock acquisition; :381 performs final origin-aware verification before returning it at :389. check_attempt_history retains the original outer address and context-instance/process checks (:177–188) and passes original origin for both target rechecks (:191/:213). Current and selected prior still use the same verified target/current policy; the fix does not weaken C1.
5. **Lifetime and locks:** the retained origin owns its guard clone and Rc independently of shorter store borrows. Both earlier compile-only lifetime obligations remain. The original guard File comes from create_or_open_profile_guard/existing_profile_guard, which do not flock. fresh_guard at publication.rs:976 independently opens the guard before flock; retained_origin does not duplicate that flock description. Thus retaining the identity/FD does not accidentally retain a publication flock through HE. Drop/failure uses ordinary RAII; no unsafe, leak or stale pointer dereference is added.

The original reproducer now fails even with identical metadata and reused ProfileTargetStore address, because its new inner store has a distinct still-live Rc allocation. Guard replacement while retaining the original store fails its original-file/current-entry validation. This is a source proof of the refusal path, not a reported native execution result.

## Regression/source-quality assessment

Only four source files changed. Target publication/acquire/recovery now use the stronger original-guard check as well as readers; their logical C/W/F/L/D ordering, no-clobber/late-error/recovery behavior and fresh-flock protocol are unchanged. The existing native Target fixture now passes its actual opened guard File through the same private from_opened constructor, preserving the actual guard rather than hand-building incomplete fields. This adds real validation to fixture setup; it does not add a portable test path or change existing ignore/child70/71 selection.

The retained.rs production reader is unchanged; its addition is one ignored Linux test. Quick/C1 codecs, caps, source sizes/hashes, original context/process semantics, preparation/accounting/continuation and public wire formats are unchanged. TargetReadOrigin is not Clone/serde/Send/Sync authority; sharing the private Rc internally does not expose duplication of a live gate. No new dependency/license exception or unsafe operation. No reason to rerun unrelated broad historical suites was found.

Resource delta is accurately disclosed: each actual store retains one original guard FD and one small Rc allocation; each origin adds one guard FD and Rc strong reference. Actual rechecks add held-guard metadata and mount observations (including existing bounded /proc reads). No extra raw file buffer or additional materializer is introduced. Original actual read_target counts remain2 extra materializations,5 identity derivations,5 context rechecks/10 direct helper subprocesses+2N selected input helpers, with the newly added guard reads/FDs separately charged. These are future resource-model obligations, not a physical memory/FD capacity PASS.

## Static evidence and test-first boundary

Read all7 pinned new command/output/EXIT logs. red01.log records genuine compile RED, exit101, for seven missing new identity/comparator references in the four regression tests. green01.log records all-feature lib --no-run exit0. host-clippy01.log and linux-clippy01.log record all-target/all-feature strict Clippy exit0, the latter with actual x86_64-unknown-linux-gnu cfg. fmt01/fmt-check01/diff-check01 all exit0. No test executable/--list invocation appears. Linux cfg typechecking supplies no Linux linker/runtime evidence. Earlier default/parallel/cuda no-run results remain historical; final all-feature lib no-run plus host/Linux all-target strict checks reasonably cover this mode-independent fix.

## Exact14 portable filters and transitive work

All are `ckks-train-bench --lib` exact selections for a later root release. Require exactly1 passed/0 failed/0 ignored for each; zero-tests must not be success. The four new tests call the actual private production comparator. No fixture regeneration or native selection is needed. Full exact names:

- `study_provenance::publication::retained_tests::retained_exact_size_is_checked_before_allocation`
- `study_provenance::publication::retained_tests::retained_dataset_policy_is_closed_and_exact`
- `study_provenance::context::context_instance_tests::retained_context_instance_never_reuses_and_overflow_refuses`
- `study_provenance::context::context_instance_tests::retained_equal_values_are_not_original_owner`
- `study_provenance::publication::retained::tests::retained_bounded_reads_short_truncated_growing_and_complete`
- `study_provenance::publication::retained::tests::retained_source_type_owner_and_mode_policy`
- `study_provenance::publication::retained::tests::retained_same_size_wrong_digest_refuses`
- `study_reference::document::evidence::quick::tests::retained_quick_staged_checks_require_no_future_end`
- `study_reference::document::admission::tests::retained_c1_current_and_prior_share_one_verified_target_representation`
- `study_reference::document::admission::tests::retained_c1_exact_file_payload_and_stale_self_are_distinct`
- `study_provenance::publication::retained_tests::retained_store_identity_original_owner_accepts`
- `study_provenance::publication::retained_tests::retained_store_identity_equal_metadata_distinct_owner_refuses`
- `study_provenance::publication::retained_tests::retained_store_identity_same_address_replacement_refuses`
- `study_provenance::publication::retained_tests::retained_store_identity_survives_original_store_drop`

The original10 bodies/transitive selection paths are unaffected:7 small pure controls plus3 inert fixture comparison tests;24 fixture reads across22 files,209,797 raw bytes total (Quick38,891; C1history120,872; exact-file/payload/self50,034), largest file10,970 bytes. Parsing/canonical/hash buffers are real bounded work, not zero allocation. No actual context/store/source acquisition/registry/materializer/parameters/planner/P1/preparation/crypto/runtime-capacity/root-authority/native/helper call is selected.

The additional4 controls perform7 small Rc allocation events in total, peak2 allocations per test. They cover (1) original owner acceptance with clone, (2) distinct equal-metadata owner refusal, (3) replacement in the same Rc variable/address with retained old allocation and equal metadata, and (4) identity survival after original owner drop, distinct-owner refusal and Weak inability to upgrade after all strong references drop. Rc clone/downgrade/upgrade do not allocate another object; the private payload is two u64 guard values plus allocator/Rc bookkeeping. They perform no filesystem, store construction, guard acquisition, ProcessStamp capture, subprocess/thread/fork, context/registry/materializer/parameter/HE calls. They model owner identity, not the whole native filesystem. The existing two compile-only lifetime functions remain uninvoked.

Thus the14 portable candidate is accurately bounded as **11 pure +3 fixture controls**, not a release of real retained acquisition or materialization. Old runtime1 was inspected as released:false and remains pre-fix only. Root's future runtime2 must use these final pins; this report does not modify either controller/release.

## New native control and still-missing proof

`study_provenance::publication::retained::tests::retained_native_original_guard_and_same_address_store_replacement_refuse` is Linux cfg and explicitly ignored. It adds1 scratch namespace to the earlier8, one4-byte target.lock.json acquisition, two zero-byte guards, two actual private TargetStores and one retained origin. The single in-process actor renames/replaces the guard; original store must refuse; replacement store must pass its own check yet fail original origin; original guard FD survives store drop while unchanged target HeldRead recheck succeeds. This isolates I1 from corrupt lock bytes/ancestor failure. It adds no children/fork/threads or context/materializer/HE construction.

Aggregate retained package:19 tests, **14 portable +5 ignored native**; all executed counts remain0 here. Native scratch remains separately authorized safe-ancestor/private0700/unprivileged supported ext4/XFS with explicit --ignored; no fallback/auto scratch/missing-env success. Actual owner/guard/mount/fork/races, shared publisher native21 cases, reclamation/durability/power-loss behavior and eventual live custody remain unexecuted. This scoped PASS closes a source defect; it does not declare those proof obligations satisfied.

## Remaining mandatory work

All unchanged original review limits remain. The retained API does not bound earlier whole-file materializer reads. SAME retained bytes must reach actual typed materialization, and a genuine validated resource producer/current capacity/claim must precede acquisition. Implementation/layout/applicability and trusted authority producers, all five sealed gates, guarded attempts/sidecars, Full/Quick durable lifecycle, consuming begin with same owners/accounting and post-start binding, and scientific runner/tables remain incomplete. All21 profiles, Full2680+20+7, Quick268, baseline/GPU/external/secure-N/NI/scientific requirements and existing encrypted-block hold remain unchanged. No positive fake authority or Task54 completion is claimed.

## Changed final source pins

| Path under crates/ckks-train-bench/src/ | Bytes | SHA256 |
|---|---:|---|
| study_provenance/publication.rs | 37228 | `4cb08e06dc57459e87784fc22cf1aa32904382fe716ec6bd46dc1a2f05666557` |
| study_provenance/publication_anonymous_tests.rs | 37265 | `8805757ca2a0c35b3a0fa351ff5b986bf6db3a6fa6f435c43d0e6b2e374c6fa4` |
| study_reference/target.rs | 23993 | `ccb13e42286c80bf609ae6d2e05b7e9f059e664a22e3ad4e1acb20bd8620b69b` |
| study_provenance/publication/retained.rs | 20306 | `311d1806eaf39642fc1b2904ac9a9e02569659fc079003d90baa2734d27b01af` |
