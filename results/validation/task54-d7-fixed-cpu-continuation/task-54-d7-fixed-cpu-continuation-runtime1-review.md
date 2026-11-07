# Fixed CPU continuation runtime1 — independent outcome review

**Verdict: PASS for the three released exact unit-test outcomes and their bounded evidence claims.** Q1 is correctly applied. No outcome-report correction or rerun is required. The parent controller terminal exit remains **UNOBSERVED**; this is a retained orchestration limitation, not an observed exit0.

This was a focused read-only evidence review after the independent source review. No source re-audit, Cargo/test execution, SSH, agent delegation or source edit was performed. Only this review artifact was added.

## Evidence checked

Independently matched the supplied hashes:

- runtime1 report: `bcfadce49fa40325cb627094004d66fdebdbfeb4af49834310f28cfdaa763128`
- runtime1 release: `947d12c2a04bfdd7a2c522fc46f7fd968b65d9b655ee06deb3c07d45f7e921bd`
- runtime1 controller: `9c2e65274786d18a53880cc8cc793c3e388b1d09d17a331c3883a90a3f53ac21`

Read the complete report/controller, release semantics and inventory clarification, controller launch, root observation, complete summary and all three launch/meta/raw sets. Independently rehashed all **491 source/input pins**, all **5 release-input artifacts**, and the **6 raw streams totaling1222bytes**. They match their recorded digests. Each per-test meta equals its summary entry, and its pre-launch fields equal the corresponding launch record.

Each actual command is `cargo test -p ckks-train-bench --lib --features parallel <exact-filter> -- --exact --test-threads=1`. The controller uses `CARGO_NET_OFFLINE=true`, runs children sequentially, waits for each direct Cargo child, checks exact named success/counts, and has no retry branch. Existing output-directory creation and exclusive raw-stream creation prevent silently overwriting this invocation's records. These facts support the recorded bounded single invocation; they are not external attestation against unrecorded execution.

| Exact filter | Cargo child PID/exit | Raw selected outcome |
|---|---|---|
| `study_client::fixed_continuation_tests::d7_fixed_prekey_exact_parameters_and_rotations` |73951 /0|1 passed,0 failed,0 ignored; exact name matches|
| `study_client::fixed_continuation_tests::d7_fixed_prekey_bootstrap_owner_and_parameters` |74215 /0|1 passed,0 failed,0 ignored; exact name matches|
| `study_execution::cpu_continuation::tests::d7_fixed_retained_association_and_copy_refusal` |74217 /0|1 passed,0 failed,0 ignored; exact name matches|

All three outputs report226 filtered out. No broad test-suite result is implied. The recorded build used the `parallel` feature; this is not evidence of a separate default-feature runtime run. The summary is complete for all three children and records their direct `proc.wait()` exits.

## Inventory and claim consistency

The release explicitly supersedes the preserved author inventory's total interpretation of `rng_from_seed:4`: test3 has **4 direct refusal RNGs +8 internal materialization RNGs =12 total**, no entropy/forks on the reviewed paths. It also retains NTT table counts72/36/72, RNS basis counts4/2/4, the scalar DFT proof's65,535root checks, two full canonical Iris400-epoch materializations/F64/F32 initialization before Quick2 consumption, and the distinction between ordinary allocation/work and zero reachable HE constructors. The report repeats these corrections accurately. The embedded original inventory's `executed_tests:0`/`runtime_authorized:false` remains a preserved pre-release snapshot; top-level release status and actual results are separate and unambiguous.

Successful test bodies establish their parameter/P1/rotation refusal, original association and accounting assertions. Planning-work assertions and constructor totals are source-derived completed-body inventory, as the report states. Zero KeyGenerator/Encoder/Evaluator/Bootstrapper/session/refresh/training/native-observer calls follows the preceding independent source review of those reachable branches; it is not newly instrumented runtime allocation proof, RSS or timing evidence. No successful encrypted setup/drop, numerical I/NI loop or poison injection is claimed.

## Retained observation limit

`root-observation.json` records fixed controller PID73949 absent in `ps` (return1, empty output), three Cargo exits0 and the missing parent tool terminal observation. A process's absence cannot determine its exit status. Likewise, a complete three-test summary does not by itself prove the controller's later final check/print or terminal exit succeeded: summary publication precedes those operations. The report correctly does **not** claim parent exit0 or use absence to reconstruct it. The independently verified current pins and per-child raw/meta records support the narrower three-test result despite that missing parent observation. The initial wrapper/session-metadata omission is preserved explicitly and was not “repaired” by rerunning.

The report's conclusion is therefore appropriately limited to metadata/pre-key refusal/accounting validation. Production `MissingExecutionDependency`, genuine gate/admission/start integration, resource/security/precision evidence, native Linux/GPU execution, scientific rows, full-study requirements and existing holds remain outside this evidence. This review does not grant further runtime or campaign authority.
