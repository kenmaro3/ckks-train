# Task54 P3 fix2 independent narrow rereview

2026-11-07. **SPEC: PASS for residual R3a/R3b correction. QUALITY: PASS for this test-only insertion.** No further load-bearing finding was identified in the assigned delta. Together with the accepted fix1 corrections, the reviewed standalone P3 source stage can proceed to root integration. This does not complete full P3 acceptance, the real pre-key spy, P4/D7 gates, Task54 or the scientific campaign.

Read release `task-54-p3-fix2-release.md`, the complete41-line report, complete delta, scope proof, final manifest and all three raw logs. Reviewed only the two residual omission discriminators and regressions introduced by the insertion. No Cargo/test execution, constructors, planner calls, HE, source edits, Git/network or subagents occurred. Only this report was written. P4 copies were checked as dependency provenance; their implementation and scientific claims were not reviewed here.

## Frozen scope verified

* Current module SHA256: `697f76cd289b626b7167d569561befe9b36375c03baa8971c0f0d7de34c6e561`.
* Fix2 report SHA256: `50c7aa4b2c92dbd9cb45ab058920da791dd658086ad7df9386a197386714174e`.
* Exact delta SHA256: `be48818c5955dbafa475289dccf0f5d18ada8d0f912266b712703a0253f3d426`.
* Final manifest SHA256: `f023ea968c1410ddc76a134ee3ab64bc1c81a46d19776c77d3a4429267a19357`.

Independently rehashed all22 manifest artifacts with no mismatch. All seven current source/dependency files match the retained frozen copies. The before module matches the exact fix1 frozen module. Recreating the unified diff yields the exact supplied delta and81 inserted lines. An independent comparison excluding only the named test function confirms all remaining module bytes are identical before/after. No production helper or other test changed. The other three P3 files and three P4 dependency files are unchanged across the fix2 before/frozen/current views.

## R3a accepted — each raw rate field has an independent omission discriminator

The insertion in `p3_digest_golden_and_mutations` calls the actual production `encode_rate_schedule_fields`. Its literal112-byte expected vector specifies count3, twelve distinct F64 values1 through12 in source/mean/delta/kernel order, and reference16. The length is8+12×8+8=112; the literal is independent of encoder output. It covers field content/order as well as presence.

Each variant changes exactly one field of entry1 by0.5. The four eight-byte slots begin at40/48/56/64, consistent with the count and preceding entry. Each whole encoding must differ, while prefix and suffix outside that slot must match exactly. Removing source, mean, delta or kernel from the production helper would break the fixed-vector expectation and/or the corresponding isolated variant. Entry0, entry2, other fields and the reference remain fixed. The original valid full-suffix-below-maximum/reference tests remain byte-identical elsewhere.

The raw values intentionally need not satisfy authoritative kernel=mean×delta. This test discriminates serialization fields; it neither admits a schedule nor bypasses `rates()` validation. No parameter or planning work is needed for these assertions.

## R3b accepted — present velocity level and scale bits are separately discriminated

The same test reuses its existing tiny parameter object and actual valid level3 label. The baseline velocity is `[Some(label),None]`. `CtMeta::new` constructs one variant at level2 with identical scale and another at level3 with scale bits increased by1. Both preserve Some/None tags, list lengths and every other StateLabels field.

Both variants use the actual production `encode_state_labels`. The independent layout calculation places the present label at offset129: three lists of count8+two labels×16, followed by velocity count8 and the Some tag. Assertions preserve length and all bytes outside the level slot129..136 or scale slot137..144 while requiring a different whole encoding. Omitting either numerical field while retaining the option tag would now fail its corresponding variant. No fabricated CtMeta, unsafe mutation or production encoding refactor was introduced.

## Execution evidence and regression scope

Raw `03-filtered-green.log`, SHA256 `59e82aa8847f849bcc8aa2540678e44340c12ef9a79d2255156ca05e5cec38d3`, records exactly the released filter, one selected test PASS,189 filtered tests and exit0. Its diagnostics show both new groups were reached, followed by the existing2259-byte combined accounting checks. There is no runtime RED requirement in this tests-only release, and production was not deliberately broken to manufacture one.

Clippy01 (`--lib --tests --no-deps -- -D warnings`) and the single-file Rust2024 formatting check02 both exit0. Their compilation does not execute the other tests. All three logs record the same seven before/after pins, matching current frozen copies. No11+5 runtime rerun is claimed or required for this insertion; fix1's broader focused evidence remains attributed to its own source/dependency snapshot and selected assertions.

The selected test has only its one existing `CkksParams::test_tiny()` constructor. The added calls use that object and pure encoding helpers; they do not initialize shared fixtures. Fix2 actual work is params1, proofs/descriptors/materializers/geometry/initial-state/updates/handoffs/wrappers/HE all0, within the explicit release. P3-owner cumulative counts become params24/proofs11/descriptors15/Iris5/geometry527/initial527/updates243/handoffs79/wrappers14/HE0. Separately owned P4 work is excluded from this ledger.

## Acceptance boundary

R1/R2/R4/Q1 and heterogeneous-y/state continuity retain their accepted fix1 dispositions; R3a/R3b are now closed for the standalone source correction. The public plan remains a trusted local sealed value rather than an arbitrary transported receipt decoder. No production functionality, schema, arithmetic, scope or execution authority changed in fix2.

The actual P4/D7 consumer still must establish genuine gates/start and demonstrate a reachable nonallocating positive factory spy plus corrupt identity/parameter/partition rejection before the real factory. That original requirement is REQUIRED/PENDING. P4 typed geometry/full-trace comparison access, total/transient/RSS and HE memory, range/precision/security/target/start, full-profile planning feasibility, encrypted-block holds and scientific campaigns remain separate. This scoped SPEC/QUALITY PASS supports source integration only.
