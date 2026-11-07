## Spec Compliance

**SPEC: PASS for the released source package. QUALITY: Approved. No critical or important finding.** No runtime test, native observation, Linux integration or subsequent execution is accepted or released by this review.

- `study_reference/document/runtime_descriptors.rs:9–41` uses the exact ratified encoding/domain Text headers and existing nested document encodings. The physical and runtime bodies follow the prescribed order; hashes cover complete header/body bytes. Descriptor output exposes immutable bytes/hash inspection only, with no gate/start/Ready conversion.
- `runtime_descriptors.rs:67–226` requires available nonempty bounded CPU/OS/hostname, optional exact CUDA identity, raw16-byte nonzero UUID, ordinal/capability ranges and the actual ASCII/noncontrol PCI spelling. Ordinal0/compute minor0 remain valid. Physical bytes exclude PID, threading telemetry, environment and private process/pool/native identities.
- `runtime_descriptors.rs:239–494` preserves closed form/recipe/backend mapping, positive actual-worker claims/serial1, sequential0, CPU absent/CUDA stream1, mode/placement/bootstrap-engine route consistency, physical SHA256 linkage, complete CUDA paths/hashes/versions, exact four ordered kernel entries and two intended options. Driver/NVRTC major must be positive; NVRTC minor0 remains available. Non-UTF8 Unix module paths are retained without normalization or filesystem access by the inspection decoder.
- `runtime_descriptors.rs:495–617` is a fixed private cursor for these two bodies. It bounds input by A_MAX, checks length conversion and offset addition before slicing/allocation, bounds nested text/raw/hash values, uses little-endian U64, rejects unsupported tags, bad UTF8/hash spelling/counts, truncated/trailing bytes and semantic mismatch, then requires exact canonical reencoding. It adds no JSON schema or generic descriptor registry.
- `study_runtime.rs:201–254`, `study_runtime/cuda.rs:205–244` and `study_provenance/train_capture.rs:175–192` add borrowed observations from the actual retained lease/binding, bracketed with existing rechecks. Workers come from the retained executor; Host is the same retained object already used by the rechecked capture. CUDA accessors expose paths/hashes/versions/device/kernel/options only, without native addresses, module owners, file handles or process/pool identity.
- `runtime_descriptors.rs:201–226` produces physical bytes from the same RecheckedTrainCapture, checks its Host pointer, and compares against a second rechecked retained projection. `study_runtime.rs:342–354` calls it immediately after the existing Case component and still returns `MissingExecutionDependency`. No extra materialization, HE constructor, publication or readiness path is added.
- `runtime_descriptors.rs:363–494` provides policy inspection encoding and retained-observation comparison, but no production policy producer is connected. PolicySelection is explicitly a claim, not a checked route. The missing genuine implementation adapter must still bind actual sequential dispatch, intended one-stream construction and refresher/bootstrapper ownership, and later check the created device after durable start. Leaving that connection unavailable is required by this release, not a missing deliverable or permission to substitute CLI/constants.

**Cannot verify from diff/build evidence:** actual approved Linux Train physical capture, native CUDA observation/module/device behavior, a real checked runtime route, created streams/options, gates/admission/start and scientific execution remain unmeasured. The Linux-only test body is cfg-excluded on Mac and has neither Linux compile proof nor execution evidence here. Underlying unchanged document/runtime/native helpers remain dependencies rather than a fresh exhaustive audit.

## Strengths

- `runtime_descriptors.rs:201–226` and `train_capture.rs:175–192` preserve exact captured-owner linkage instead of rebuilding host facts or deriving workers from environment/request values.
- `runtime_descriptors.rs:296` compares the complete CUDA snapshot encoding with retained observations, including module/source/options identities; recomputing a claimant's digest does not make substituted bytes match. This comparison remains distinct from proving concrete route ownership.
- `runtime_descriptor_tests.rs:49–283` freezes all four complete independent literals and adds meaningful transport-prefix/trailing/domain/tag/length/UTF8, field-bound, path/version, kernel/order/options, route/alignment and cross-link negatives. Synthetic CUDA values require no native observer or executor.
- `runtime_descriptor_tests.rs:285–315` deliberately joins its one Linux test thread before assertions and avoids making global observed-task-count equality/inequality a stable identity condition. Its source tests only host projection, not an approved context or CUDA path, and is properly excluded from the proposed Mac inventory.

## Issues

### Critical (Must Fix)

None found in the bounded diff.

### Important (Should Fix)

None found in the bounded diff.

### Minor (Nice to Have)

No additional source change requested. Preserve the following limits in later outcome reports:

- `runtime_descriptors.rs:363–494`: successful encoding/inspection or even retained-observation comparison does not establish the missing sequential/stream/refresh implementation route. The codec intentionally accepts synthetic inspection structures; no production route claim follows from their private bytes/hash.
- `runtime_descriptor_tests.rs:285–315`: source inspection suggests a coherent Linux-only host/telemetry test, but neither Mac no-run/clippy nor a Mac `parallel,cuda` library check compiles this cfg-excluded body. A future native Linux build/test is still required; this review supplies no Linux compile result.

## Review checks and retained evidence

Read the parent brief first, release, complete descriptor contract/four literals, relevant runtime-design §§6–7/12 and target-design §5, source report and full diff. The1,393-line diff was read once in three consecutive chunks, including both complete new files. No Git command, source edit, Cargo/test, native call, network operation or subagent was used. Only this review-file write accompanies read/hash/text work.

Named-risk outside-diff checks were limited to:

1. **Nested encoder reuse might bypass A_MAX or change Observation/RawOs semantics.** Inspected unchanged document Encoder/append and Text/Hash/Option/Observation/RawOs encoders (`document.rs:450–485,519–572,723–757,1090–1110`). Every descriptor append is auxiliary/nonblob and contributes to unchanged A_MAX. The existing unavailable-reason type is Text256; Cursor inference retains that bound. New semantic checks cover manually assembled typed inspection values before encoding.
2. **Mode mapping might silently treat an unsupported CaseMode as Interactive.** Inspected the existing closed vocabulary (`study_profile/registered_case.rs:56–58`): CaseMode has exactly Interactive/Noninteractive, and unsupported backend/placement forms fail the explicit policy comparisons. No new registered route is created.
3. **CUDA fact access might bypass held-owner/version/device/kernel checks.** Inspected unchanged source-hash ordering and CudaBinding recheck/export path (`study_runtime/cuda.rs:279–329`) plus observer recheck/query bounds (`ckks-train-cuda/src/runtime_observation.rs:133–199,385–456`). Actual device count/ordinal availability stays in the retained observer, while only identity fields are serialized. The private view uses existing module/export/file checks and the NTT/pointwise/basis/keyswitch order. No observer or full-module read policy was rewritten. These were source reads, not native observation.

Artifact custody:

| Artifact | Verified SHA256 |
|---|---|
| Source release | `3efd2882e335dfd416b257532ce9f0d2d134eb9e0af5b3cb49676f8b42062c44` |
| Source report | `a15905ad10d0c7dbedff9a3193c395a0cb8483d7e0c9f81b017eeac0f6953636` |
| Complete review diff | `e9a9098542e4ee3666ffe88ab47a73cfcff1c944448ba9edc956867b4ae1421c` |

All6owned,5binding and89unchanged entries in `source-validation/source-pins.json` independently match current bytes. BASE is reported as `9f04ecd4a3b25280496aeafd86c7bb09c15804aa`; the intervening operational-only commit is outside this candidate. The initial tracked-file stat has four paths; the complete diff/pins correctly include the two new descriptor files as well. Current unchanged pins include the existing Case/Reference producer,82Case golden files, Reference golden source, native CUDA observer and Cargo files. This verifies those retained identities; no Git BASE comparison or whole-repository preservation check was rerun.

The four retained validation hex files exactly match the ratified document literals. Independent stdlib byte decoding/hash arithmetic confirms:

| Literal | Bytes | SHA256 |
|---|---:|---|
|P_CPU |111 |`5c5e1f65bdb89e5151e8422ef699d63a422f7d4bd7fb3ad288dcb5942e0d214f` |
|R_CPU |209 |`d9da1fc171c8f60e848c916d2926a7f7579903aac8157a9d273aa7eb01e80fb5` |
|P_CUDA |179 |`b2a0322e7aa8bf78bc68c618c36070daa955e2800adeb2e585e0e62fcb274693` |
|R_CUDA |816 |`def6285a4a1dd5276e7a28fbe8a851460a693f80f2fa2aa34a70172e2570a7d9` |

The diff's frozen test literals/assertions correspond to those complete examples, including byteFF and valid zero ordinal/minors. This is source/fixture arithmetic, not execution of the Rust codec.

Retained logs01–10 support genuine missing-module compile RED, preserved private-comparator compile failure, successful final parallel no-run/strictclippy, serial and parallel+CUDA library checks, and targeted formatting/diff checks. Log10 records final no-run/clippy/format success after explicit domain-corruption coverage was added. Final outputs are clean. No test binary or `--list` ran, so none of the descriptor or golden tests has a new runtime PASS in this source package.

The proposed Mac batch is9exact tests:6new synthetic descriptor tests plus3unchanged Case/Reference golden regressions. The six new tests use only bounded in-memory literal/structure/encoding/hash work: datasets/materializations/native/module/file reads/executors/spawned threads/subprocesses/HE/JIT/learning all0. Two Case regressions read checked-in golden files; the Reference regression uses embedded literals. The excluded Linux host test would perform2Host observations and1joined std thread, with no dataset/CUDA/module/executor/subprocess work. Existing executor/nonforgeability tests remain unchanged rather than being silently added to this inventory. This is a proposed bounded selection, not execution authorization.

## Assessment

**Task quality: Approved for this source dependency.** Exact private representation and actual retained physical plumbing are implemented without fabricating the absent runtime-route authority. Separate Linux/CUDA integration and reviewed consuming-route work remain necessary.

Full21profiles/full2680+20+7/quick268, Reference-only authority, unsupported native-CUDA-bootstrap/external routes, secure-profile/resource/range/precision/security and campaign holds remain unchanged. Other execution descriptors, live gates, admission/start, scientific output and full Task54 remain incomplete. No runtime, HE/scientific acceptance or subsequent invocation follows from this review.
