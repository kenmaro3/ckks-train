# Task 55 current packages: independent bounded review (2026-11-08)

**SPEC: PASS. QUALITY: PASS with one nonblocking reporting clarification.** The retained evidence supports the five successful offline/locked macOS package verification builds and the actual archive-content claims. No package-content or dependency correction is required. This review is pinned to source `2dc049a796949b685a5c23f8b05440ee2ce474aa`, not current HEAD or active Rust WIP.

## Evidence identity and review method

- Reviewed report: `task-55-current-packages-20261108-report.md`, SHA256 `79742bdd60ee386cf19cb678bf9e4dadf1708ad9798118b5462dc40d63e24056`.
- Evidence root: `tools/release/_ext/package-current-2dc049a/`.
- `freeze.json`: SHA256 `95eb695bd136bebd137d51c3b0b5859e1eb5f41862e893179a7ed27eb38e8b62`; its source, report, evidence-manifest and source-manifest pins agree with the reviewed inputs.
- `evidence.sha256.jsonl`: SHA256 `9a15d96b08f26c9360a139e561323f361dbbb7aec8ab8973127cbc783153920f`.
- Source-manifest pin: `c4b5f1e26055e85f72bbc08c626324e332cdd454ade3c0e5ef8a83f17ba01f79`. Root separately reviews the complete source/evidence manifests; this bounded review directly compared every actual nongenerated package member and original crate manifest with `git show` at the fixed commit.
- Read `commands.json`, `execution.json`, the complete five package stderr logs, `qualify.py`, `audit.py`, `finalize.py`, `write_report.py`, package audits, lock audit, local source configuration and cleanup record. Used independent standard-Python archive/TOML/hash reads; did not execute author scripts, rebuild, run tests, fetch, upload, use SSH, edit source or spawn agents.

## Confirmed results

All five recorded package commands have `--offline --locked -j 2`, fixed-export cwd, exit 0, and logs containing Packaging, Verifying, compilation of the extracted package path, and successful Finished output. The fifth command adds `--features cuda`; its log includes cudarc compilation. No recorded unlocked fallback occurred. The qualification script contains a possible lock-normalization fallback, but the executed command records contain only the five locked invocations. The task-specific target path is explicit and separate from the shared target.

| Actual archive | Members | Bytes | SHA256 |
|---|---:|---:|---|
| `ckks-train-core-1.0.0.crate` | 17 | 48115 | `797db8c4475ec3ee3fbe828114cee807bf202893bdac3cd0c06ebeeafaa1b017` |
| `ckks-train-ckks-1.0.0.crate` | 39 | 203387 | `1e0fed1861d922ed71fc6207cc924aafec88505881f6fb3723442564a3bed5ed` |
| `ckks-train-nn-1.0.0.crate` | 60 | 347802 | `79ab5c60f793f581a83009cedf7eb8963e23547d771383666cb39c5699d93808` |
| `ckks-train-cuda-1.0.0.crate` | 41 | 130075 | `eef928e6f78f822502109e3d34b1ea65e33f1daef14040a52d3d2d4468ccd495` |

The separately retained fifth archive, `ckks-train-cuda-1.0.0-cuda-feature.crate`, is byte-identical to the default CUDA archive. The historical counts are not substituted for these actual counts.

Every archive member is a regular file, has a unique safe package-relative name, and is either an exact fixed-commit crate file or an expected Cargo-generated `Cargo.toml`, `Cargo.toml.orig` or `Cargo.lock`. Each `Cargo.toml.orig` exactly matches the fixed original manifest. README bytes match the fixed crate README. No excluded external-tool checkout, `_ext`, `.git`, target, ledger, data/results tree or unowned nongenerated member is included. The reported credential-pattern scan is appropriately limited to packaged content; this is not a library-security or history audit.

Every normalized internal dependency/dev-dependency uses `ckks-train-*`, version `1.0.0`, and no local `path`; no old `hekit-*` dependency remains. The actual archive lockfiles have 72/66/24/27 external package entries respectively; every external version/source/checksum matches the fixed workspace lock. Their 0/1/2/3 internal dependency entries reference the exact newly retained dependency archive checksums. The generated lockfiles are inspected package inputs, not a registry-availability claim.

All four packages include the complete root LICENSE (1075 bytes, SHA256 `37b50b0dc885fe0e9306cca59347532477623f058c9c496d60f153c535e20f87`) and THIRD_PARTY_NOTICES (10077 bytes, SHA256 `fa1b18275f507a15f821863a0e239f0b3859b858d647cab497b0f072430a0769`), with regular archive mode 0644. The entire de-indented Unicode V3 text matches cached `unicode-ident 1.0.27/LICENSE-UNICODE` (1995 bytes, SHA256 `f7db81051789b729fea528a63ec4c938fdcb93d9d61d97dc8cc2e9df6d47f2a1`); the entire selected rustix MIT text matches cached `rustix 1.1.6/LICENSE-MIT` (1023 bytes, SHA256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`). Fixed `deny.toml` retains only the exact `unicode-ident =1.0.27` Unicode exception and the stated MIT/Apache/BSD/ISC general allow list. Packaged crate licensing remains MIT.

All five CUDA kernel assets match fixed Git bytes. Independently read hashes are:

- `kernels/basis.cu`: `f339bcb5492fee28e5041f7c3097aea6326b2024f0f9a4be2855fcd21bafef0a`.
- `kernels/keyswitch.cu`: `8001e378323e64173700f82aa734d042d75c08af8aaad906864120dcd0ea5014`.
- `kernels/ntt.cu`: `fc487dfe9190c712c4a27f862100019d57afe7f501ce17acbc31691155b315da`.
- `kernels/pointwise.cu`: `d7da2002c30ff52256ffed4ba8b88117effaf3f039f6e44ced84db7666b1541b`.
- `kernels/clang_check_shim.h`: `cf8322ae85616e938da4bbde961d312f44340e5eca7bbc2cde292f87ff4ac791`.

The Git export has no `.git`; the actual packages have no `.cargo_vcs_info.json`. Provenance is supplied by the standalone source/evidence manifests and fixed Git identity, not by invented in-archive VCS metadata.

The cleanup script deletes only its resolved, nonsymlink `build-cache` directory after archives/logs/audit evidence are retained. `cleanup.json` records 556 removed files, 269853342 logical bytes and 271339520 allocated bytes, and the cache path is currently absent. Its external vendor entries are references to prior cached directories; the script does not delete or rebuild that prior cache.

## Nonblocking clarification and limits

The report's “own peak was about519MB allocated” is stronger than the retained measurement. `cleanup.json` establishes a **pre-deletion observed allocation** of 519069696 bytes; it is not a continuous high-water measurement. Minimal correction: replace that sentence with “The observed working area immediately before deletion was about519MB allocated, below1GiB; transient peak allocation was not continuously measured.” No repeat build is needed. If the frozen report is revised, refresh its report/freeze/evidence pins rather than treating the present report hash as unchanged.

Correctness and package provenance: PASS. Package-scope security/exclusion checks: PASS within the stated narrow scan. Maintainability/reviewability: PASS, with retained logs, hashes and fixed inputs. Runtime performance: not assessed.

These are Cargo packaged-library compilation checks on Rust/Cargo 1.85.1 for aarch64-apple-darwin. They do not establish packaged test or benchmark execution, native kernel compilation/execution, GPU timing, training, scientific correctness, Linux CI at the final release SHA, crates.io availability/upload/publication, stable-v1.0 completion or a general security assessment. Active WIP was not qualified by this review.
