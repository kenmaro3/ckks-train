# D7 runtime/physical inner descriptors

2026-11-07. **RATIFIED DESIGN; implementation and execution gates remain pending.** This fills only the byte-layout gap delegated by runtime-design §7.3 inside target-design §5's existing ExecutionBinding. Existing scientific policy, schemas, parser caps, five design pins, complete registration inventory and unresolved cases remain unchanged. No tests, constructors, HE, native queries, network or Git operations were run. Only read/text/hash work and the independent synthetic byte calculations below were performed.

## 1. Canonical conventions and domains

Use the existing `hekit-study-reference-encoding-v1` primitives exactly: U64 is eight little-endian bytes; Text is U64 UTF8 byte length then bytes; Hash is Text of exactly 64 lowercase hexadecimal SHA256 characters; enum is one byte; Option is absent00 or present01 followed by value; List is U64 count then ordered items; Bytes is U64 decoded byte length then raw bytes. No native-endian integers, JSON hashing, Debug spelling or implicit defaults.

Reuse document.rs `Text`, `Hash`, `Option<T>`, `List<T,N>`, `Observation<T>` and `RawOs` nested encodings without changing existing Reference/Case bytes. Observation is available00+value or unavailable01+Text256(reason), distinct from Option's tags. RawOs is Unix00+Bytes4096, preserving non-UTF8 path bytes. Do not reuse the full Host record for physical identity: it includes PID and changing telemetry. A narrow private codec module under the existing document module can use these primitives; do not introduce a second generic encoding framework.

Each descriptor is `Text(encoding), Text(domain), body` with no integrity/self-digest field. Proposed auxiliary domains are `hekit-study-physical-binding-v1` and `hekit-study-runtime-policy-v1`; these are inner descriptor domains, not new scientific artifact schemas. Hash every canonical byte, including both header Text values. Keep the existing 2,097,152-byte auxiliary canonical bound (`A_MAX`) and existing field bounds; nothing raises a cap. No new JSON document/envelope/parser is proposed. If descriptors later travel inside an existing bytes field, preserve its existing bounds as well.

Reject unsupported tags, missing/trailing bytes, bad lengths/UTF8/hash spelling, arithmetic overflow or semantic mismatch; never normalize decoded claims into captured facts. Descriptor bytes and hashes are immutable inspection values, never capabilities.

## 2. Physical binding: exact body order

| Order | Field | Exact encoding and constraint |
|---|---|---|
| 1 | cpu_model | Observation<Text4096>, available and nonempty required |
| 2 | os | Observation<Text4096>, available and nonempty required |
| 3 | hostname | Observation<Text4096>, available and nonempty required |
| 4 | cuda_device | Option<CudaDeviceIdentity>; absent for CPU, present for CUDA |

`CudaDeviceIdentity` is exactly: `ordinal:U64, uuid:Bytes(exactly16), pci_bus_id:Text4096, compute_major:U64, compute_minor:U64`.

Retain the existing observer semantics: ordinal fits nonnegative c_int and is below its observed positive device count; UUID has at least one nonzero byte; PCI is the actual nonempty ASCII/noncontrol spelling before NUL from the fixed32-byte query buffer (length1..31), without case/PCI normalization; compute major is positive and minor nonnegative, each at most i32::MAX. Ordinal0 and minor0 are valid. An individual zero UUID byte is valid; an all-zero UUID is rejected by the observer. The UUID is raw bytes, never formatted text. Device count remains a retained/rechecked availability fact, not a new physical identity field.

The three host values come from the SAME retained Host snapshot and must equal the corresponding captured Case host values. `CaseRuntimeLease::recheck` checks these stable identity values against fresh observations. Host identity here means the ratified cpu_model/OS/hostname tuple; it does not claim a globally unique machine UUID. No machine-ID/boot-ID field is invented.

`available_cpus`, `observed_threads`, threading environment and existing Case process_id stay in their unchanged CaseProvenance fields. Their snapshots are not serialized again here. In particular, `/proc/self/task` count must not become a stable-identity equality gate. Actual execution worker count is separately bound below. Private ProcessStamp, ThreadId, pool pointer/index, native addresses/bases and file identity stamps remain private recheck state, never new descriptor/provenance fields.

## 3. Runtime policy: exact body order

| Order | Field | Exact encoding and constraint |
|---|---|---|
| 1 | form | byte: cpu_serial=0, cpu_rayon=1, cuda_host_rayon=2 |
| 2 | recipe | Text4096, exactly the corresponding closed recipe below |
| 3 | backend | byte: cpu=0, cuda=1; must match form and RegisteredCase |
| 4 | execution_threads | U64, actual retained executor workers, positive; serial exactly1 |
| 5 | matvec_order | byte: sequential=0 only |
| 6 | stream_count | Option<U64>; CPU absent, CUDA present and exactly1 |
| 7 | refresh_route | byte: cpu_client_refresher=0, cpu_bootstrapper=1, cuda_host_refresher=2, cuda_host_bootstrapper=3 |
| 8 | physical_binding_sha256 | Hash of the complete physical descriptor above |
| 9 | cuda_runtime | Option<CudaRuntime>; CPU absent, CUDA present |

Recipe strings by form0/1/2: `study-train-cpu-serial-linux-clean-v1`, `study-train-cpu-rayon-linux-clean-v1`, `study-train-cuda-host-linux-clean-v1`. This is the runtime-design §3 closed mapping, not Rust enum-discriminant casting. Full current-build role/algorithm/features checks remain mandatory and unchanged. Rayon worker1 is valid and still form1/2, never rewritten to serial.

Route0/2 requires registered I/client-refresh semantics and actual ClientRefresher/HostRefresher ownership; route1/3 requires registered NI semantics and actual CPU Bootstrapper/HostBootstrapper ownership. CUDA routes retain the synchronous independent host evaluator/counter contract. Check route against the existing outer mode/placement/bootstrap_engine and reviewed implementation, not CLI alone. Here CPU/CUDA I corresponds to outer bootstrap_engine none, CPU/CUDA NI host; this does not reclassify a client refresh as bootstrap. No tag is assigned for native CUDA bootstrap or external-b: those remain registered/unavailable until their separately owned adapters and representation are reviewed, with no CPU fallback. Unsupported current routes fail rather than disappear from inventory.

`CudaRuntime` exact order is: `driver_path:RawOs4096, driver_sha256:Hash, driver_version:U64, nvrtc_path:RawOs4096, nvrtc_sha256:Hash, nvrtc_major:U64, nvrtc_minor:U64, kernels:List<Kernel,4>, intended_options:List<Text4096,2>`.

`Kernel` is exactly `kind:byte, source_sha256:Hash`, with exactly four entries in order ntt=0, pointwise=1, basis=2, keyswitch=3; reject omission/duplication/reordering. Hash the complete UTF8 bytes of the corresponding owned embedded source constant, not a filename. Intended options has exactly two strings in this order: `--fmad=false`, `--gpu-architecture=compute_{major}{minor}`. The decimal major/minor spellings have no leading zero except the number0; values must come from the same observed physical identity. These are the observer's actual intended options, not proof of compilation or PTX.

Paths are distinct, absolute, canonical raw Unix paths from the retained module descriptors. Driver version is the actual positive encoded integer from cuDriverGetVersion, not a split or toolkit version. NVRTC major is positive, minor may be0; all three fit i32::MAX. A successful observed version with minor0 is available, never null/unavailable. Existing observation source rejects driver0 or major0; do not weaken that check or invent nonzero replacements. Unknown/missing CUDA libraries, module hashes, device identity, options or versions reject a CUDA descriptor; Option absence means CPU inapplicability, never failed CUDA observation.

These projections describe two observed top-level modules and intended compilation only, not complete native/transitive dependency attestation. No new file-size/read cap replaces the existing 1GiB/module, 1MiB chunks, full-read/hash and identity checks.

## 4. Actual source mapping and missing adapters

| Required fact | Existing retained source | Required narrow implementation, not currently evidence |
|---|---|---|
| form/recipe/backend | study_runtime.rs CaseRuntimeLease.recipe, HostExecutor/HostPool; current checked recipe and registered selector | Private read-only projection derives closed tags; cross-check all three without accepting caller strings |
| actual workers | HostExecutor captured `pool.current_num_threads`; CaseRuntimeLease::execution_threads rechecks owner/pool/process | Use this count, never Host::execution_threads scalar helper, env value, global Rayon discovery or requested count |
| host identity | CaseRuntimeLease.host; context.rs Host::observe cpuinfo/uname/gethostname | Borrow retained fields, bracket capture with the unchanged lease/context rechecks |
| modules/hashes | cuda.rs CudaBinding.driver/compiler RetainedModule.path/hash/file/identity and retained exports | Private borrowed fact view after CudaBinding::recheck; do not reopen a caller path or export pointers |
| versions/device/options | CudaBinding.observation: CudaRuntimeObservation accessors | Borrow actual values; preserve observer/export/module rechecks; ordinal request alone is not device identity |
| four source hashes | CudaBinding.kernel_hashes, NTT/POINTWISE/BASIS/KEYSWITCH_KERNEL_SRC in that order | Fixed tag/hash projection; no new source discovery |
| sequential/stream/route | Ratified training/runtime contract; CPU study_client uses ClientRefresher/Bootstrapper; CUDA device exposes n_streams/physical_identity/intended_compile_options | Missing checked implementation adapter must bind selected concrete route and intended one-stream construction; lease alone cannot supply route readiness |

Before start, stream_count=1 binds the reviewed concrete construction requirement; it is not a claim that a CUDA stream already exists. After durable start the existing D7 consuming boundary must compare the created device's n_streams/identity/options with the retained requirement and bind sequential dispatch to its actual route. This proposal neither constructs that boundary nor grants pre-start allocation. Missing adapters remain MissingCode/availability errors, not successful descriptors from constants plus CLI. The runtime lease by itself does not establish the implementation route.

Outer linkage is unchanged: ExecutionBinding.physical_binding_sha256 = SHA256(P); runtime_policy_sha256 = SHA256(R); R's physical link must equal SHA256(P). Check the actual retained P/R bytes and same live context/runtime alongside the outer fields. No recursive hash: P has no link to R. Exact host/device/worker/route linkage must be checked even if a claimant recomputes every digest.

## 5. Independent literal transport examples

These are synthetic format examples assembled from literal byte prefixes/ASCII and hashed with Python hashlib, without repository encoder imports or runtime observation. They are NOT captured evidence, approved contexts or gate passes. Hex below is the complete byte sequence; whitespace is not data.

P_CPU: host cpu_model=`C`, os=`O`, hostname=`H`, each available; cuda_device absent. 111 bytes; SHA256 `5c5e1f65bdb89e5151e8422ef699d63a422f7d4bd7fb3ad288dcb5942e0d214f`.
```text
210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311f0000000000000068656b69742d73747564792d706879736963616c2d62696e64696e672d7631000100000000000000430001000000000000004f0001000000000000004800
```
R_CPU: form0, exact serial recipe, backend0, workers1, sequential0, no streams, route0, link=P_CPU hash, cuda_runtime absent. 209 bytes; SHA256 `d9da1fc171c8f60e848c916d2926a7f7579903aac8157a9d273aa7eb01e80fb5`.
```text
210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311d0000000000000068656b69742d73747564792d72756e74696d652d706f6c6963792d763100250000000000000073747564792d747261696e2d6370752d73657269616c2d6c696e75782d636c65616e2d763100010000000000000000000040000000000000003563356531663635626462383965353135316538343232656636393964363361343232663764346264376662336164323838646362353934326530643231346600
```
P_CUDA: same available C/O/H; device present, ordinal0, UUID bytes00..0f, PCI `0000:01:00.0`, compute8.0. 179 bytes; SHA256 `b2a0322e7aa8bf78bc68c618c36070daa955e2800adeb2e585e0e62fcb274693`.
```text
210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311f0000000000000068656b69742d73747564792d706879736963616c2d62696e64696e672d7631000100000000000000430001000000000000004f000100000000000000480100000000000000001000000000000000000102030405060708090a0b0c0d0e0f0c00000000000000303030303a30313a30302e3008000000000000000000000000000000
```

R_CUDA: complete synthetic runtime body, form2, exact `study-train-cuda-host-linux-clean-v1` recipe, backend1, workers2, sequential0, present stream1, route3 (cuda_host_bootstrapper), link exact P_CUDA hash above, cuda_runtime present. Driver RawOs bytes `/opt/cuda/lib/` + one byteFF + `driver.so`, hash=`a` repeated64, version13010; NVRTC RawOs `/opt/cuda/lib/libnvrtc.so`, hash=`b` repeated64, major13/minor0. Exactly four ordered kernel entries tags0/1/2/3 with hashes `c`/`d`/`e`/`f` each repeated64; exactly two ordered options `--fmad=false`, `--gpu-architecture=compute_80`. 816 bytes; SHA256 `def6285a4a1dd5276e7a28fbe8a851460a693f80f2fa2aa34a70172e2570a7d9`. The raw path contains one non-UTF8 byteFF, not four literal backslash/x/f/f bytes; no filesystem path was accessed. NVRTC minor0 is valid, and the linked P_CUDA retains valid ordinal0/compute minor0. These are format fixtures, not loaded modules/device observations or route readiness.
```text
210000000000000068656b69742d73747564792d7265666572656e63652d656e636f64696e672d76311d0000000000000068656b69742d73747564792d72756e74696d652d706f6c6963792d763102240000000000000073747564792d747261696e2d637564612d686f73742d6c696e75782d636c65616e2d76310102000000000000000001010000000000000003400000000000000062326130333232653761613862663738626336386336313863333630373064616139353565323830306164656232653538356530653632666362323734363933010018000000000000002f6f70742f637564612f6c69622fff6472697665722e736f400000000000000061616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161616161d2320000000000000019000000000000002f6f70742f637564612f6c69622f6c69626e767274632e736f4000000000000000626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262626262620d00000000000000000000000000000004000000000000000040000000000000006363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636363636301400000000000000064646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464646464024000000000000000656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565656565650340000000000000006666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666602000000000000000c000000000000002d2d666d61643d66616c73651d000000000000002d2d6770752d6172636869746563747572653d636f6d707574655f3830
```

## 6. Minimal follow-on and decisions

Root ratified these domain strings, exact field/tag order and identity-only physical projection after independent design and complete-vector review, before any hash-producing source. This is a representation decision, not a new user permission gate or execution release. No scientific decision remains for this narrow representation. Missing read-only fact views, route adapter and encoders are implementation dependencies. Other ExecutionBinding descriptor encodings remain outside this proposal; these two definitions alone cannot satisfy D7.

Minimal future ownership: one private `study_reference/document` descriptor module reusing existing nested codecs; narrow private fact-view methods in study_runtime.rs/cuda.rs and the Train context owner; later separately assigned checked implementation-route consumer. No public evidence constructor, new framework, revised Reference parser, CUDA observer rewrite, P5 changes or broad gate/protocol implementation is implied.

Focused future acceptance: use the four independently assembled literal vectors above, frozen before encoder source; preserve all existing Reference/Case vectors; negative unknown tag/order/size/path/option/hash/domain/cross-link cases; ordinal0 and NVRTC/compute minor0 positives versus zero required-major/driver negatives; actual pool1/2 versus environment99 and wrong same-size pool rejection; stable host/device/module substitution rejection while changed observed_threads alone does not fail identity; capture remains within retained owner and reaches no context/JIT/key constructor. Later real created-device checks remain separate execution evidence, not synthetic acceptance.

Rationale: two small domains make the already required hashes unambiguous while preserving the outer grammar and nested codecs. Rejected alternatives: hash(Debug/CLI/JSON), whole Host snapshot as physical identity, PID/boot/pool addresses as scientific identity, environment-derived workers, fake unavailable CUDA records, and embedding a new generic descriptor registry. All would either lose actual ownership facts, create unstable identities or exceed this missing-representation scope.

Historical design-review input pins (SHA256; not current execution authority): runtime-design `fe0106291a0da629f808d8faf3d25d9fc5aad27a4bc87587949ccad7b9582d1a`; target-design `7cbf299bb5829ee911a02ceb4439ea959a2676897d68fedcb2da605a16ec2222`; study_runtime.rs `e6011d502884777cadce46d7aac3117c18b47cdc20d7a97872994b5958222bd0`; runtime/cuda.rs `7bf6f780e64dbfb5ed03b6e9b3f89906a4378cf53cc51bdc75c66e22fcfc2dcb`; CUDA runtime_observation.rs `e43610e6c7b5d0ba0e672352c0e84a278f35863cd0a5243f10a551aaecf93c5b`; document.rs `eaa09efc927de5b67417ff91a8b5c5f8f71d6decd7d8e92859b57cde50d5477f`.
