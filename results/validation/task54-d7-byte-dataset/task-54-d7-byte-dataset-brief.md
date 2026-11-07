# Task54 next implementation preparation: dataset decoding from retained bytes

RELEASED after resource source acceptance and push. Fixed implementation BASE `edecc0a4424c7efc444cb025d5561f9e6b120a25`. You are the sole source writer; the parallel bootstrap derivation author edits only its ledger worksheet, never production source. Current source pins below have been reverified and are unchanged from preparation. This implements a required seam of Task54, not a substitute for the complete same-owner preacquisition integration.

Read task-54-brief.md and the accepted task-54-d7-typed-materializer-preallocation-map.md sections1–3. Existing resource design and DatasetInput contract stay binding. Original research matrix, science/security gates, full/quick inventories and actual-input receipt identities remain unchanged. No new dependencies, approvals, schema, owner permits, or positive runner authority.

## Concrete source evidence

At root commit5b50e841e65a0bad698b9a61621ba2e1fde7571d, the three source files below remain unchanged by the concurrent resource codec author. Their current digest inventory is appended. Inspected exact paths:

- profile_data.rs:273–313 verify_npy_manifest reads manifest, hashes every pathname, decodes label files, then hashes manifest again.
- profile_data.rs:315–460 load_dataset has Iris/WDBC/IDX/NPY branches. It loads numerical values and then reopens paths for identities. NPY invokes another complete pathname loader.
- data.rs:559–613 load_npy_dataset reads manifest and all four arrays, verifies hashes, decodes, clones image arrays into train/test; the numerical behavior and file-specific manifest defense must remain.
- data.rs:794–809 load_wdbc must retain the fixed WDBC_SHA256 verification before UTF8/parse, never expose a canonical route that bypasses it.
- data.rs:204–209 load_mnist already composes pure parse_idx_u8/mnist_from_idx through pathname reads.
- profile_data.rs:498ff LockedFile and seed/order/schedule parsing is a later compulsory seam, not completed by dataset-only factoring.

## Intended bounded scope

Implement one shared pure dataset decoding path receiving borrowed source bytes with a closed variant matching DataSource. Keep path identity labels from the actual request, never infer roles from supplied basenames. All four dataset routes must consume these exact slices for hashing, structured manifest verification and numerical parsing. A borrowed input structure is ordinary data, not a resource/custody capability. Missing/wrong variant must fail; there is no pathname fallback in the byte path.

Existing public pathname utilities needed by standalone baseline binaries may wrap the same shared decoders. They must acquire each required content once and compute reported file digests from the consumed bytes, preserving numerical order, scaling, fixed split, RNG stream, labels, shape and class checks. The future retained canonical path calls the pure helpers, not these wrappers. Preserve independent structured NPY per-file digest checks and vector label checks; do not weaken them to the permissive legacy scanner alone. Avoid duplicating preprocessing implementation or changing the legacy generic parser's accepted format without a recorded reason.

Scope will be data.rs, profile_data.rs and narrowly named new tests/helper modules only. Do not modify study_runtime/run_owned, live resource owner, records/corpus, context/approval/source evidence, Cargo/dependencies or scientific results. Further resolver/preflight/captured inputs/materialize/reference/target same-byte integration and genuine preacquisition physical producer remain explicit next obligations; no post-materialization audit or caller byte budget substitutes for them.

## Tests before implementation, review before commit

Before any first executable filter, write exact test names, fixture paths/count/aggregate bytes, maximum synthetic shapes/work, command and expected RED. Use small deterministic synthetic CSV/IDX/NPY values and existing separately verified numerical expectations; tests must detect a different source slice changing the consumed result/hash and reject manifest mismatch/shape/label/UTF8 errors. A fixture involving actual WDBC content must be specifically inventoried rather than broadly loading dataset directories. Do not call any runtime/context/store/HE/native/helper/SSH, perform network fetches or scan data/processes. No cap-sized inputs, new installs/cache trees or broad --workspace/--list suites. CARGO_BUILD_JOBS=2 existing cache; bounded new filter only. Retain original RED/failures/GREEN logs. Follow with `CARGO_BUILD_JOBS=2 cargo clippy -p ckks-train-bench --all-features --all-targets -- -D warnings` (compile/lint only) and `cargo fmt --all -- --check`. A plain cargo check does not replace required strict lint verification. No redundant same-code reruns.

Final report must map every removed pathname reopen to its pure replacement, preserve numeric provenance boundaries, and identify exact remaining canonical acquisition/materializer callsites. Supply complete diff against released BASE, six-or-fewer intended source pins unless concrete necessity recorded, exact test logs/results, and an explicit statement that no real fit/gate/scientific result was established. Root sends independent SPEC/QUALITY review, resolves findings, commits and pushes.

## Source pins reverified at release

- `crates/ckks-train-bench/src/data.rs`: 58637 bytes, SHA256 `548c647d832e860d6daadd48ac8832fb185c788365798b32941ee650ee752225`.
- `crates/ckks-train-bench/src/profile_data.rs`: 28565 bytes, SHA256 `1de6c85f1baaf4a292a66598e679125ba511f4cf50b340d3ae92693a54d957dd`.
- `crates/ckks-train-bench/src/study_profile.rs`: 79141 bytes, SHA256 `eaaf3ad085f1a347b664099f720a1ad2fb068ada66939baf60c3c6bbefe3670f`.

## Handoff and source ownership

Canonical report task-54-d7-byte-dataset-report.md, complete source-only diff task-54-d7-byte-dataset.diff and exact source/input/log pins task-54-d7-byte-dataset-pins.json in this ledger. Append test inventories before first new filter invocation. Keep initial RED and all actual failure/GREEN logs. Freeze final source for root independent review and do not commit/push. No other agents. The existing tests under crates/ckks-train-bench/tests may be inspected to select meaningful fixture coverage, but no broad existing suite or dataset scan may execute without finite inventory. New tests/helper modules stay narrowly scoped to the byte-dataset task. Public wrappers remain documented; this is shared production numerical logic, never a second preprocessing implementation.
