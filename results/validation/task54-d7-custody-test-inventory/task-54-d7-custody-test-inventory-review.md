# Task54 D7 custody C01–C10 independent SPEC + QUALITY review

Reviewed 2026-11-07. **SPEC: CHANGES REQUESTED. QUALITY: CHANGES REQUESTED.** One Important finding I1; no Critical or additional Important findings. This is an inventory/current-source guarantee correction. No protocol implementation or architecture change was performed or authorized by this review.

## Scope and exact pins

Read the custody inventory brief, complete inventory and report. Compared C01–C10 against ratified custody proposal `fc938561`, accepted schema `1e547075`, gates design §§6–7, actual publication/target operation bodies and relevant existing tests. Used committed `eb9aaef` admission helper source, SHA256 `bc9e54a697cb8990dd85e2e6853f80fb9309a05960a1ba08969abe440b836d01`, for C1 composition; concurrently changing codec source was not the review subject.

- Inventory,48760 bytes: `7993132a97175f2e508d1e696a1069948032750e4dd804e96c1ee3ac22f1523c`.
- Report,3470 bytes: `592e7ce931e70c0901473c4f452064dbf37b64e80880587384541e9ac3833cc0`.
- Complete diff,53209 bytes: `8a4a5046f7d159fc97ca54b66d8eefc6e2baceac54addd6c30c5475337ad0c95`.
- Publication source: `8cfabba4dcf7daae93e90ce40a0bc1bcf13c59c713bb5c9d65964b97a4a2c004`.
- Publication tests: `10f640a0961da262ddcd15627a5a1c59455b1fddbee7cc5c9a4711b6eff33fd6`.
- Target source/tests: `1c54c5918d0b73dabcf213bf43dacdaa3682951f471dd48a0c6de7579ad49d1b` / `7a3c34222b70a9ec8cf02a46033d995c120d7fe18bbdceb0ea96aba6895806e3`.

All listed stable source/design pins matched read bytes. Both complete added-file diff bodies matched their retained files exactly. Source/test inspection and read-only byte/hash/arithmetic inspection only; none of the named tests was run.

## I1 — Important: native replacement guarantees omit the check-to-use window

Exact pointers: inventory lines26 (`unlink_owned ... never removes a replacement`),38 (`Unknown replacement objects remain untouched`),155 (internal cleanup preservation),157 (`d7_custody_c04_temp_swap`, only before-check cases); `publication.rs` lines432–458 (temp inode check followed by pathname linkat),471–486 (temp inode check followed by pathname unlinkat); `publication_tests.rs` lines386–404 (replacement occurs before calling either helper).

The recorded source does not atomically bind the checked inode to either later pathname operation. A scripted actor can replace the temporary entry **after** its successful inode check and **before** linkat: the later call may link the replacement inode. Likewise, replacing the temporary entry after unlink_owned's successful check and before unlinkat can cause the replacement entry to be removed. The retained FD does not make these pathname operations conditional on its inode. A cooperative profile flock does not exclude a scripted/noncooperating actor that bypasses that flock; replacement actors are already part of this inventory's tests.

The existing30 before-check tests are valid, but they establish only rejection when substitution is visible at that check. The native filter name and a general hook "around ... identity boundaries" do not supply exact post-check cut cases or truthful expected states. Later final-byte/identity rechecks may withhold live authority after wrong publication; they cannot retroactively preserve a replacement entry already unlinked. Therefore the unconditional guarantee and current case coverage are stronger than the pinned helper actually supplies.

Necessary correction before ratifying this inventory:

1. Qualify the existing-source description and before-check tests to the actual observed/checkable boundary. Do not claim atomic replacement immunity from the current open/check/path-operation sequence.
2. Add exact named after-check/before-path-syscall schedules for link and unlink, including unlink invoked by pre-link cleanup and create/fchmod-error cleanup. Specify helper/outer Ops counts separately, held original inode, moved original name, substituted name, visible final and surviving/deleted temp entries. A fixed parameter table is sufficient; do not merely add "all races" or reuse a pre-operation hook.
3. Record current-source exposure separately from the required custody safety assertion. If these adversarial windows are in the promised preservation contract, identify the concrete unresolved source prerequisite before a production-positive/native custody claim. A passing fake must not assume inode-conditional link/unlink that the current helper lacks. This review does not prescribe a new publisher architecture or ask for a source change in this document task.
4. Keep the existing post-link uncertainty/no-live rule: a successful link side effect cannot be injected as `Ops::link -> Err` and mislabeled BeforeLink. Update inventory/report case counts for the added schedules, without altering any file format, cap or old golden.

No execution demonstrated a real deletion or native race here. The finding follows directly from the pinned call sequence and the inventory's stronger asserted guarantee.

## Other SPEC conclusions

- Existing helper inventory is correctly located in publication.rs. Successful publish has7 logical Ops calls; pre-link failure cleanup and post-link VisibleUnconfirmed traces match the implementation. Create's fchmod-error cleanup is internal; there is no second outer cleanup after create returns an error. Native write_all call counts are separated from outer Ops counts. Repeated collision bounds and exhaustion preserve preexisting names.
- Retained-FD reads, exclusive named-attempt creation and multi-file recovery are properly marked proposed seams rather than existing APIs. Reusing `private_child` for exclusive A or `recover_ops` per recovered file would be wrong; the inventory explicitly prevents both substitutions. Full Train existing-only opening and Reference profile-empty semantics are retained.
- C01–C03 cover closed names, bounded scans, caps, object/guard constraints, no early attempt mutation, limited Quick administrative ancestors and exact evidence-before-event ordering. Attempt-file R counts are separated from context/source/native reads. Guard acquisition remains fresh; no flock is held across the driver.
- C05 crash cuts distinguish visible process state from actual power-loss durability. Named empty/orphan attempts block Initial. Complete evidence without admission cannot become a predecessor, and no temp is promoted or removed by recovery.
- C06 correctly carries C1: `check_target_claims(current,T)` and `check_target_claims(prior,T)` use the same retained lock/reference and approved policy before `check_predecessor_claims`. The committed helper intentionally does not prove prior target applicability itself. Changed-prior-lock/reference/policy cases recompute hashes and expect rejection before Vp/sync/B creation. Some-start Vp calls Vs once; no duplicate outer Vs is counted. Policy/custody object checks remain separate from pure byte helpers.
- C07's ordered five evidence files→admission/preflight→optional start/end→attempt/attempts/case/cases/profile sync sequence matches the ratified protocol. Verification precedes sync; all failures prevent new B and no recovered live owner is reconstructed. Quick recovery is inspection only.
- C08 defines bounded Initial/Retry/expected-start interleavings, same-name refusal, allowed overlap and guard replacement after waiting. Native thread/process/mount/fork obligations remain distinct from fake schedules and cannot establish genuine gate→begin lineage by themselves.
- C09 carries retained identities, process/context/owner substitution, transfer/drop order and independent context/input lifetime obligations. Full releases handles only after checked transfer; Quick retains its end-publisher frame. Failed setup/poison/unused-driver paths retain leases through crypto teardown.
- C10 is target-free and Initial-only; same live origin is required for an end, crash/lost continuation leaves no end, and parsed/recovered events cannot authorize execution or completion. Full/Quick cross-scope assertions and all eight accepted branch names remain.

## Fixture coherence and QUALITY checks

The inventory truthfully separates opaque8-byte F8/Q8 filesystem controls from actual codec parsing. FULL_EVIDENCE_SCHEMA and FULL_CODEC_OLD are **not** a linked positive Full bundle; this explicit dependency is appropriate. Before claiming combined Full codec/protocol acceptance, a separately frozen coherent independent fixture is required. This need not block source work that retains the declared separation, and it is not a new user-approval gate. Q_LINKED_FIX1 supplies the accepted coherent Quick byte/link fixture, not genuine evidence.

Read-only inspection confirmed the seven old Full raw lengths9156,7857,10546,10817,11002,7690,7699 and the eight selected Quick files totaling41054 bytes. Test buffer ceilings and trace limits are stated as fixture/requested bounds rather than RSS/native-allocation proofs. Existing C/R/A, Full wire, Quick K and original goldens remain unchanged.

Independent arithmetic checks agree with the published baseline and case totals: ENTER42 Ops/16 R, START7 Ops/13 R, END7 Ops/15 R; Full ENTER+START49 Ops/29 R; Quick with END56 Ops/44 R and51 R when BEGIN is included. Checked960 canonical temp names,146 invalid names,49 early-failure cases,144/204 read failures,120 primitive publish controls/faults,975 collision scenarios,64 directory crash cuts,210 file cuts,31 C1/link cases,59 sync-failure positions,148 recheck cuts and52 replacement cases per branch. These are correct planned counts, not observed passes. I1 requires an additional explicit native-internal schedule table rather than invalidating these existing arithmetic results.

No Cargo, compiler, test, --list, generator, runtime, native operation, SSH, HE, real gate/parameter/P1/materializer/registry call, subagent, source/public-doc edit, commit or push occurred. Read-only inline Python checked source/file hashes, complete diff contents, literal lengths and count arithmetic; it did not invoke project code. Only this review file was written.

Preserve current accepted authority, all21 profiles/full2680+20+7/quick268, holds, original scientific/security thresholds, baselines/Task5, and MIT/Apache2/BSD/ISC plus Unicode-tables-only exception. This review makes no native custody, genuine gate, resource-fit or scientific success claim.
