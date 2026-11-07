# D7 byte dataset bounded test inventory (before first filter)

BASE edecc0a4424c7efc444cb025d5561f9e6b120a25. Six exact tests in `crates/ckks-train-bench/tests/byte_dataset.rs`:

- `byte_dataset_iris_uses_supplied_slices_and_request_identity`
- `byte_dataset_idx_preserves_scaling_and_rejects_shape_and_labels`
- `byte_dataset_npy_checks_exact_manifest_arrays_and_vector_labels`
- `byte_dataset_wdbc_keeps_pinned_digest_and_fixed_split`
- `byte_dataset_rejects_missing_wrong_variant_utf8_and_declared_shape`
- `byte_dataset_path_wrappers_share_the_pure_decoders`

Command: `CARGO_BUILD_JOBS=2 cargo test -p ckks-train-bench --test byte_dataset byte_dataset_ -- --test-threads=1`.
Expected initial RED: missing `DatasetBytes`, `decode_dataset`, `data::wdbc_from_bytes` API; no tests can execute before the required borrowed-byte seam exists. Compile-error RED is explicitly distinguished from an assertion RED.

Fixture inventory (no directory scan/download):
- Only existing dataset file: `crates/ckks-train-bench/data/breast_cancer.csv`, exactly 124103 bytes, SHA256 `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`. One include_bytes fixture, also read by the WDBC pathname wrapper parity call; 569x30 source,455x30 train,114x30 test. Rejected mutation changes one byte with the same length; malformed WDBC one byte. No synthetic bypass of the UCI digest.
- In-memory Iris base: CSV 52 bytes + ordered partition 17 bytes = 69 bytes,2 inputs;3x4 source,2x4 train,1x4 test. Changed heldout CSV changes only 100/200/300/400 to 1000/2000/3000/4000; swapped split has the same length. UTF8/empty and declared-shape refusals use zero/one-byte replacements.
- In-memory IDX base:4 arrays, sizes24/10/20/9 =63 aggregate bytes;2x2x2 train,1x2x2 test;3 labels total. Variants replace heldout pixels, heldout axes1x1x4 (same byte count), one label (same byte count), or truncate train payload by one byte. Resize at most2x2->1x1;3 images, no large work.
- In-memory NPY base:4 arrays, sizes[102, 84, 86, 76], aggregate348, plus compact object manifest355 bytes = 703 bytes/5 inputs. Maximum individual synthetic matrix2x2;3 rows x2 features total,9 f64 scalars including labels. Refusals: matrix-shaped labels2x2, label3/fractional0.5, NaN image, wrong heldout width1x3, manifest replacement of one byte, or missing train-image digest; recomputed manifests for intended malformed arrays. A structured files-list manifest of4 records is also exercised. Each manifest <=600 bytes; each NPY <=106 bytes; maximum simultaneous synthetic source payload <2KiB. No generic parser format restriction is added.
- Path-wrapper parity test: a single task-specific temporary directory created by the test, exactly11 path names: `role-a`, `role-b`, the4 Split::files IDX names, the4 NPY_DATASET_FILES names and `data_manifest.json`. At base writes,11 files total835 bytes. Manifest replaced once by legacy text, no new path; cleanup only this unique test directory. Request paths in the pure tests use `absent/*` and must never be opened. No fixture directory enumeration occurs.

Finite filter runs only these6 tests; WDBC decoding bounded to 569x30 with sorting/shuffle, all other datasets <=4 scalar entries per row. No runtime/context/store/HE/native/helper calls, network, installs/cache trees, process scans, or broad test inventory/suite. Existing Cargo cache and CARGO_BUILD_JOBS=2. Strict Clippy compile/lint and fmt check follow GREEN. TDD skill used; brief's bounded test restriction takes precedence over its generic full-suite direction.

## Preexecution follow-up inventory: early request validation

After first GREEN(6/6), add one assertion to the existing exact test `byte_dataset_rejects_missing_wrong_variant_utf8_and_declared_shape`. Command: `CARGO_BUILD_JOBS=2 cargo test -p ckks-train-bench --test byte_dataset byte_dataset_rejects_missing_wrong_variant_utf8_and_declared_shape -- --exact --test-threads=1`. Same Iris in-memory69-byte base, no additional fixture/path; request.features=0. Expected assertion RED: load_dataset currently returns NotFound by opening the absent CSV before validating a zero-width request, whereas the original load_dataset returned InvalidData before acquisition. Preserve validation order by sharing the small request validator; no new decoding behavior or authority. Log suffix early-request-red. After this source correction rerun original six-test filter once, log green-2.

## Preexecution follow-up inventory: strict-lint test fixture writer

First strict Clippy failed on `clippy::format_collect` in the new legacy-manifest test fixture writer, not production source. Retain `task-54-d7-byte-dataset-clippy.log` unchanged. Replace test's format/map/collect with writeln! into one String (identical fixture text/bytes). Command: `CARGO_BUILD_JOBS=2 cargo test -p ckks-train-bench --test byte_dataset byte_dataset_path_wrappers_share_the_pure_decoders -- --exact --test-threads=1`; same sole unique temp directory,11 paths,835-byte base fixture aggregate above; no new dataset or work. Expected GREEN; log suffix wrapper-green. Repeat strict Clippy as clippy-2 and fmt as fmt-2 after this changed test helper. Production source unchanged after green-2.
