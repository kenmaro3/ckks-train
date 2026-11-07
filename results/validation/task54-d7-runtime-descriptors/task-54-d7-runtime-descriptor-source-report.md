# Task54 D7 descriptor source report

2026-11-07. Source-phase deliverable only; **no tests executed**, no native CUDA observation/JIT, HE/learning, remote/network, commit/push or subagents. Full Task54, actual approved Linux capture, checked runtime route, live gates, admission/start and scientific execution remain incomplete.

Source BASE `9f04ecd4a3b25280496aeafd86c7bb09c15804aa`; root's independent operational-only HEAD is `af9a33f08ac9e338b8605f21ca663bc67b846c5b`. Release SHA256 `3efd2882e335dfd416b257532ce9f0d2d134eb9e0af5b3cb49676f8b42062c44`. Read parent brief first, full ratified descriptors, runtime-design §§6–7/12 and target-design §5. No domain/tag/layout/cap/scientific-policy changes.

## Source changes and pins

All final source and binding pins, plus 89 unchanged source/golden pins, are in `task-54-d7-runtime-descriptor-source-validation/source-pins.json`.

- `crates/ckks-train-bench/src/study_reference/document.rs`: `616c24753c18a3c1da31845b8610beb476d3ce2eb2dd84aa927525fb3ec7df17`
- `crates/ckks-train-bench/src/study_reference/document/runtime_descriptors.rs`: `edad74318331721d11ef0563eccc425cba4686aed828c486878e7b8a4a02df7f`
- `crates/ckks-train-bench/src/study_reference/document/runtime_descriptor_tests.rs`: `3873f08466775c3ac9d9e50734680dd450d7e8a20971a62385ddd1bd2f653d03`
- `crates/ckks-train-bench/src/study_runtime.rs`: `8a85ae2230ef658e80dff6fce25f46e9649b690e84b3d8b5ae3125d7a0200f79`
- `crates/ckks-train-bench/src/study_runtime/cuda.rs`: `351a30ddccb584325275fa62044c9e0fad5b194589c3b4ad2f3aec0f6951f740`
- `crates/ckks-train-bench/src/study_provenance/train_capture.rs`: `dfb92702c8a3acb533b8596d12f8d6ec66155525f039c8be8d3c9497df1d26df`

Existing Case/Reference producer code, all 82 Case golden files, Reference golden source, CUDA native observer, Cargo manifests/lock were byte-compared to BASE and unchanged (89 files total). P4/P5, session/NN, input selectors, CLI, admission/start and native observer internals were not edited. Unowned `.vscode/`, `results/mnist_blocks*`, and root's operational files were not touched.

## Implemented boundaries

- One private `document/runtime_descriptors.rs` reuses existing Encoder, Text, Hash, Observation, Option, List and RawOs encodings. Every append counts against unchanged A_MAX; no new dependency, JSON schema, parser cap or generic codec framework. Its fixed private Cursor reads only these two ratified bodies, validates lengths/UTF8/tags/domains/hash spelling/semantics/trailing bytes and requires exact reencoding. This is private inspection decoding, not a new public artifact reader or authority constructor.
- Physical bytes/hash are immutable inspection values. Physical identity uses only available nonempty CPU/OS/hostname and optional actual device ordinal/raw UUID/PCI/compute tuple. PID, observed threads, available CPUs, environment, pool/process/file identity and native addresses are not duplicated into physical bytes.
- `CaseRuntimeLease::descriptor_facts` borrows its actual same Host/recipe/retained executor workers, and optional CUDA facts, bracketed by unchanged lease rechecks. `CudaBinding::facts` brackets a narrow borrowed view with its existing module/export/observer rechecks. Views expose raw retained module paths/hashes, successful versions, physical identity, ordered source hashes and intended options; no native address/FD/file-owner/pool identity escapes.
- Runtime policy codec preserves all fields, forms/closed recipes, CPU/CUDA and worker/stream/route alignment, physical hash link, raw paths (including nonUTF8), positive required versions/major versus valid ordinal/minor0, fixed kernel order/hashes and intended option order. Private `validate_retained` can compare a policy claim with actual runtime observations, including complete CUDA module/source/options snapshots. It is not called as a production route producer and does not prove sequential/stream/refresh implementation ownership.

## Actual production physical flow

`run_owned` still owns everything synchronously inside the installed retained executor. It loads the genuine Train context, pinned registry/registered case, resolves/materializes the selected input, calls `capture_train` then `finish_train_capture`, and encodes its existing captured Case component. **Immediately afterward** it calls that same `RecheckedTrainCapture::physical_descriptor`.

The new producer obtains `runtime_descriptor_facts` from the capture's exact retained Train context, bracketing with artifact rechecks. It requires pointer equality of the retained Host with the Host already used by Case capture, and identical retained execution worker count. Physical body is derived from that Host and optional actual retained CUDA identity, encoded, then checked against a second rechecked retained projection and the same Host pointer. It returns only bytes/hash, and `run_owned` still returns unchanged `MissingExecutionDependency` afterward.

This path exists in production source and compiles; it was **not executed** here. If executed later, existing context rechecks still read/hash source/executable/receipts and run their approved local helper; CUDA lease rechecks still perform actual observer/module/export checks and full retained-module reads. None of those effects occurs in the proposed synthetic runtime tests below.

## Runtime codec versus missing production route

Actual retained-fact view and exact runtime byte encoding are implemented. **No actual production runtime-policy producer is connected.** The missing dependency is the genuine checked concrete implementation-route adapter binding the reviewed sequential matvec dispatch, intended single-stream construction, and actual `ClientRefresher`/CPU `Bootstrapper` or CUDA `HostRefresher`/`HostBootstrapper` ownership to the registered mode/placement/bootstrap-engine. After durable start, its consumer must separately compare created-device stream count/physical identity/options and retain the synchronous independent host-evaluator/counter contract.

Minimal future connection: the separately owned genuine route adapter must supply those concrete ownership facts, use this capture's same rechecked runtime/physical descriptor, derive the policy inside the private document codec, invoke retained-fact validation, and retain exact bytes through the later D7 boundary. This package adds no empty route token, callback/trait, CLI-to-success mapping, gate/start/Ready constructor or capability conversion. Native CUDA bootstrap/external-b remain unsupported current routes. Other ExecutionBinding descriptors/live gates and the full runner remain required.

## Test-first and static evidence

Four complete literals were frozen in `runtime_descriptor_tests.rs` and validation `p_cpu.hex`, `r_cpu.hex`, `p_cuda.hex`, `r_cuda.hex` **before encoder source**. Independent Python stdlib decoding/hash calculations reproduced lengths 111/209/179/816 and the ratified hashes, without repository encoder imports, host/module reads or runtime observation. Fixtures are explicitly synthetic, including raw byte FF, CUDA ordinal0/compute minor0/NVRTC minor0.

All commands/output/exit codes are retained verbatim:

| Log | Command/result |
|---|---|
| `01-red.txt` | `cargo test -p ckks-train-bench --lib --features parallel --no-run`: expected RED exit101, unresolved missing private descriptor module after frozen tests |
| `02-compile-first.txt` | same no-run command: exit0 |
| `03-clippy-first.txt` | strict clippy: exit101; sibling test could not call private CUDA-snapshot comparator; corrected to document-parent-only visibility, without public API |
| `04-clippy.txt` | strict clippy after correction: exit0 |
| `05-no-default-check.txt` | `cargo check -p ckks-train-bench --lib --no-default-features`: exit0 |
| `06-cuda-check.txt` | `cargo check -p ckks-train-bench --lib --features parallel,cuda`: exit0 |
| `07-final-no-run.txt` | `cargo test -p ckks-train-bench --lib --features parallel --no-run`: exit0 |
| `08-final-clippy.txt` | `cargo clippy -p ckks-train-bench --lib --tests --features parallel -- -D warnings`: exit0 |
| `10-final-domain-test-static.txt` | Added explicit domain-Text byte corruption to existing transport test; final no-run, strict clippy and targeted rustfmt check all exit0 |
| `09-format-diff.txt` | targeted `rustfmt --check --config skip_children=true` for six owned source paths and `git diff --check`: both exit0 |

Compile success is not runtime test success. No test executable was invoked, including no `--list`. The Linux-only test body is cfg-excluded from this Mac build, and native Linux/CUDA integration is unverified.

## Root's proposed bounded future pure runtime inventory

Not execution authorization. Feature set `parallel`; use exact names with one test thread; no broad module filter. Each of the six new tests below performs only bounded in-memory literal/structure/encoding/SHA256 operations: **materializations0, datasets0, host/native/module/file reads0, executors0, spawned threads0, subprocesses0, HE/JIT/learning0**.

Prefix `study_reference::document::runtime_descriptor_tests::`:

1. `synthetic_descriptor_four_independent_literals`: exact complete bytes/length/hash for all four ratified examples, including valid zero values and nonUTF8 raw path.
2. `synthetic_descriptor_rejects_transport_corruption`: every strict prefix, trailing byte, encoding/domain mismatch, impossible lengths, A_MAX excess, unsupported tags and invalid UTF8. Covers the implemented private P/R inspection decoders, not a public artifact parser.
3. `synthetic_physical_identity_bounds_and_swaps`: required host/device fields, empty/unavailable identity, UUID, PCI ASCII/noncontrol/length, ordinal/compute-major range and substituted physical identity/link.
4. `synthetic_runtime_alignment_and_route_claims_rejected`: form/recipe/backend/worker/stream/route/mode/placement/bootstrap-engine mismatches; Rayon worker1 remains Rayon. Selection values are inspection claims, not checked route evidence.
5. `synthetic_cuda_paths_versions_kernels_options_and_hashes`: raw-path tag/canonical spelling/bounds/distinctness, required driver/NVRTC major, valid minor0, kernel inventory/order/hash, intended option order/exact compute spelling, wrong physical link and source-hash substitution through the same comparator used by retained CUDA validation.
6. `synthetic_runtime_other_supported_routes_and_missing_cuda_rejected`: CPU NI route1, CUDA I route2 and CPU Rayon worker2 positive inspection encodings; absent/unexpected CUDA facts reject.

Root-selected unchanged regressions (same `parallel`, zero materializations/HE/JIT/executors/spawned threads/subprocesses/native module reads):

7. `study_reference::document::case_provenance_tests::case_provenance_all_six_independent_components_exact`: regular checked-in Case golden fixture reads only, no datasets/native observations.
8. `study_reference::document::case_provenance_tests::case_provenance_reference_stays_scalar_and_unavailable`: checked-in Case golden fixture reads only, no datasets/native observations.
9. `study_reference::document::tests::independent_python_exact_canonical_bytes_cover_all_domains_and_timing_options`: embedded Reference/diagnostic/failure literals; no file/native reads.

Explicitly excluded from current Mac inventory: `study_reference::document::runtime_descriptor_tests::linux_physical_host_projection_excludes_live_thread_telemetry` (`cfg(target_os="linux")`). It performs two real Host observations (`/proc/cpuinfo`, `/proc/self/task`, available-parallelism query, uname/gethostname/PID); creates exactly one std thread held by channels and joins it before assertions. It checks stable host projection bytes while a known extra thread is live and deliberately makes no equality/inequality gate from global task-count telemetry. No approved context is fabricated; datasets/materializations/CUDA/modules/HE/executors/subprocesses0. This requires separate native Linux review/execution evidence.

Existing retained-executor pool1/2, env99, foreign same-size pool, thread ownership and capture nonforgeability tests were left unchanged. Their compile-time trait/privacy checks still compile, but this source package neither executes nor expands their runtime inventory. No materializing Iris test is proposed for this bounded run.
