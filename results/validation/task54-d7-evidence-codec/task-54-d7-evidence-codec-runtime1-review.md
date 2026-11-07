# Task54 D7 evidence/Quick codec runtime1 — independent outcome review

2026-11-08 JST. **SPEC: PASS. QUALITY: PASS. EVIDENCE: PASS for the recorded bounded local unit run.** No blocking finding or correction to the current runtime1 outcome report is required. This is an outcome review after the original source review and scoped FIX1 approval, not a new general code audit or a runtime rerun.

## Binding scope and verified pins

Read the binding source release, amended author report, complete original source review, scoped FIX1 review, runtime1 outcome report, release and controller. Read all64 raw stdout/stderr streams and inspect every32 child launch/outcome metadata pair plus the complete summary. Source contents were not reopened for a new audit; only read-only hashing was used to verify the released source/input identity. No concrete outcome discrepancy required a source hunk review.

| Artifact | SHA256 |
|---|---|
| Amended author report | `c86f8990501d19d91055f59056909d27e4cabff838e46612209909d97486ec3d` |
| Original source review | `131f04eb5366f8dfa8446b414ac644855d1a831e65b422df05795505d5f2476d` |
| Scoped FIX1 review | `34f43991aca69f71bda205ba6ddfb865d5d0e206269887e85003b0cb715ec023` |
| Runtime1 report | `a2cab79ac2b86caaff2c21dd99d587f1bf06dcc455919b80fec80446a6de7063` |
| Runtime1 release | `3e487b5a260c3cb8e2ce3e32d93c31c20cdadf420e2ca64aec1471c730f75673` |
| Runtime1 controller | `0648171ad73b88e25bf76b09576bdd02830193e7a6e652fe737fd103bdfa2164` |
| Runtime1 summary | `b579ae85f5bab29cc729b74efaed340ef629c6d2bf01e0f6d3c0180e81ebae6a` |
| Recorded tool-session receipt | `9e9d4d8d065e38b4d7478bb0510c3cc310ccdf9bfb213a09c4c17e4775941e9c` |

All these pins match. Independently recomputed all642 released source/input hashes and18 release-input hashes against current bytes; all match. This establishes current identity with the pinned reviewed release. The controller's before/after-child checks and successful recorded completion support the historical run identity; current hashing alone is not presented as historical observation.

## SPEC outcome

The exact ordered test list equals the corrected versioned inventories:16 new evidence/Quick tests,15 parameter-free shared tests, and one separately bounded `study_admission::tests::p3_digest_golden_and_mutations`. The release embeds the corrected `fix1/proposed-runtime-inventory-fix1.json`, `fix1/proposed-shared-regressions-fix1.json` and `fix1/p3-tiny-transitive-inventory.json` exactly. It does not use the original all-zero shared classification as its release authority.

The author report's original25,484-byte prefix remains byte-exact, SHA256 `9a200eb062a58629ccc4bdf62fc4c78faf8670298adb53dfd3c0501db796631c`; its original erroneous all-zero inventory remains historical. The FIX1 section explicitly supersedes it. Preserving that earlier claim does not make it the current governing claim.

The P3 exception is explicitly released for one invocation, one child at a time,60seconds. It is correctly described as one insecure toy N32 parameter preparation with two bases,13 NTT-table constructors,26×32 vector Shoup entries plus26 scalar entries and three CtMeta constructors. These are the previously reviewed static counts. Table preparation is not an executed forward/inverse NTT. The32nd raw stderr reports `tiny_params=1`, transcript3, metadata2256, combined2259, matching the runtime report. No physical allocation-count, RSS, security or resource-fit claim follows.

The original review's I2 additions occur within retained exact filters8 and12, both observed passing. The runtime report's1135 boundary truncations,302 present-tag corruptions,119 policy/119 hash permutations and84 legal failure combinations describe the reviewed test-body coverage. Raw harness output establishes the named tests passed; it is not separate per-case instrumentation. The original and FIX1 source reviews supply the already-reviewed semantic interpretation without this outcome review repeating the code audit.

## QUALITY outcome and exact selection

Independently parsed all32 stdout files, separately from the controller's `expected_selected` flags. Every file has exactly:

- `running 1 test`;
- the one released fully qualified test name followed by `... ok`;
- one result line: passed1, failed0, ignored0, measured0, filtered253.

There are32 unique filters and32 unique child PIDs. Each recorded argv is exactly `cargo test -p ckks-train-bench --lib <released exact name> -- --exact --test-threads=1 --nocapture`; no broad prefix, feature override, `--list`, ignored-test selection or retry argv occurs. All32 exits are0 and all timeout flags arefalse. The default-feature scope is unchanged.

All32 launch records agree field-for-field with their corresponding outcome metadata, and each complete outcome record equals its corresponding summary entry. The summary is complete and ordered exactly as the release. Every child starts after the prior child's recorded end. The first child spans3.883seconds including the initial compile; subsequent spans are approximately0.093–0.153seconds; P3 spans0.147seconds. All are below their120second bounds, or P3's separate60second bound. These are controller wall intervals including Cargo/harness work, not benchmarks or constructor timing.

The controller opens streams exclusively, creates a new output directory, waits sequentially, owns each new process group, stops on a bad exit/wrong selection/timeout and contains no retry path. Timeout termination is confined to that owned group. This inspection is not an exercised timeout/fault-injection test. The launch record preserves `CARGO_NET_OFFLINE=true` and automatic_retry=false. No warning, error, panic or failed-test marker was found in the complete raw streams. The initial compile lines and two existing P3 diagnostic lines are consistent with the reported run.

## EVIDENCE outcome and custody limits

All64 raw stream byte lengths and SHA256 values match child metadata, total12,490bytes. The directory contains exactly32 stdout,32 stderr,32 child launch and32 outcome files, together with controller launch, summary and tool-session receipt. The recorded receipt names original session89518, launch chunkca8b75, terminal chunk348568, terminal exit0, one controller invocation,32 Cargo children, zero timeout and no retry. It agrees with the release, controller behavior and complete child evidence.

The original tool invocation was not replayed or independently reacquired by this reviewer. This review accepts the supplied recorded tool-session receipt as custody evidence and independently verifies its pin and consistency with raw files; it does not claim an external attestation or prove the absence of unrelated historical runs. Within this recorded runtime1 package there is one result per released filter and no retry or missing selection.

The earlier lost intermediate compile-success log remains disclosed in the amended author report. Its content was not reconstructed and this successful run does not repair that historical gap. The current source/static reviews and runtime evidence retain their separate scopes.

Reviewer activity was read-only text/hash/stdlib inspection. Two inline inspection attempts failed because of reviewer harness mistakes: a broad case-insensitive `FAILED` search matched the expected text `0 failed`, and a JSON lookup used `filters` instead of the actual `exact_filters` key. Corrected independent inspections passed; neither event ran a test or changed artifacts. No source, Cargo, tests, `--list`, native, SSH, HE, subagent, commit or public-document operation occurred. Only this separate review report was written.

## Remaining boundary

The32 passing tests establish bounded local codec/representation and selected shared-regression outcomes on this pinned default-feature package. They do not establish genuine retained reader/gate acceptance, live authority, actual resource/model validity, current source/report applicability, filesystem custody/durable publication, Linux/native/GPU execution, HE/scientific results or full Task54 completion. The independently accepted linked Full oracle remains separate and is not integrated or exercised by this package. C1 still requires real current-and-prior same-target/reference/approved-policy composition; synthetic-domain rejection or unrelated predecessor tests do not satisfy it.

Full/Quick schemas and caps, all21 profiles, full2680+20+7/quick268, scientific/security policies, work holds, MIT/Apache2/BSD/ISC with the Unicode-table-only exception, and later explicit runtime/production gates remain unchanged. No additional test rerun is requested by this outcome review.
