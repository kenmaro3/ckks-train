# Task 55 current packages: fixed 2dc049a qualification (2026-11-08)

Status: **PASS for current macOS package builds and actual archive content**. All four publishable crates and the CUDA-feature package verification completed with exit0, offline and locked. Root independent review is pending. No registry upload or publish command was run. This does not establish stable-v1.0 completion, exact-final-SHA Linux CI, native GPU execution, training, or a scientific result.

## Fixed input and isolation

- Source commit: `2dc049a796949b685a5c23f8b05440ee2ce474aa`.
- Working/evidence root: `/Users/marconi/Documents/research/hekit/tools/release/_ext/package-current-2dc049a` (ignored).
- Complete tracked export: `source.tar.gz`; extracted fixed input: `source/`.
- `manifests/source.sha256.jsonl`: **5616 entries, 158709078 logical bytes**, SHA256 `c4b5f1e26055e85f72bbc08c626324e332cdd454ade3c0e5ef8a83f17ba01f79`.
- Every manifest entry was checked against its exact Git blob identity and Git permission mode; no extra/missing tracked entry exists. Active writer WIP and untracked files were not copied. The source manifest was rechecked after packaging and cleanup.
- The four crate trees, `Cargo.lock` and `deny.toml` have no diff between preview `6fba46c` and this fixed commit (saved `preview-package-input-diff.json` and stdout/stderr). Later HEAD movement does not retarget this evidence.
- Toolchain: Cargo1.85.1 (`d73d2caf9`), rustc1.85.1 (`4eb161250`), aarch64-apple-darwin. `CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=0`, `CARGO_NET_OFFLINE=true`, and task-only `CARGO_TARGET_DIR=/Users/marconi/Documents/research/hekit/tools/release/_ext/package-current-2dc049a/build-cache`.

The prior Task55 root `/var/folders/l9/j0s8_fkd3b53h97864_x2kq40000gn/T/ckks-train-renamed-packages-itehj5x0` was read only. The new `vendor/` references its **86 external cached crate directories** through symlinks; prior internal ckks-train/hekit package directories were excluded. Each newly produced internal archive was extracted only into this task's `vendor/` with exact file hashes and archive checksum. All referenced external directory-source checksum files remained unchanged. No new dependency fetch, install or network update occurred. The prior target and shared repository target were not used or removed.

## Build commands and outcomes

The exact common command was:

```sh
cargo --config /Users/marconi/Documents/research/hekit/tools/release/_ext/package-current-2dc049a/local-source.toml package \
  -p <crate> --offline --locked -j 2
```

Serial order: `ckks-train-core`, `ckks-train-ckks`, `ckks-train-nn`, `ckks-train-cuda`; then CUDA again with `--features cuda`. All **5 exit0**, no warning/error lines, no lock fallback or source-manifest rewrite. `cargo package` compiled its extracted package verification source. These are packaged-library builds; packaged tests, benchmarks, GPU kernels and runtime were not executed. All external entries in all four generated package lockfiles match the fixed workspace lock's exact version/source/checksum. Internal package-lock checksums equal the newly created dependency archives.

All command argv/cwd/start/exit and separate stdout/stderr paths are retained in `commands.json` and `execution.json`, with every log in `logs/`. The complete packaging stderr logs were inspected. No failed Cargo or content-audit attempt occurred. Exploratory directory-inventory misses are not treated as qualification failures.

## Retained exact archives

| Package | Members | Compressed bytes | SHA256 |
|---|---:|---:|---|
| ckks-train-core | 17 | 48115 | `797db8c4475ec3ee3fbe828114cee807bf202893bdac3cd0c06ebeeafaa1b017` |
| ckks-train-ckks | 39 | 203387 | `1e0fed1861d922ed71fc6207cc924aafec88505881f6fb3723442564a3bed5ed` |
| ckks-train-nn | 60 | 347802 | `79ab5c60f793f581a83009cedf7eb8963e23547d771383666cb39c5699d93808` |
| ckks-train-cuda | 41 | 130075 | `eef928e6f78f822502109e3d34b1ea65e33f1daef14040a52d3d2d4468ccd495` |

All four reside in `archives/`. The fifth `ckks-train-cuda-1.0.0-cuda-feature.crate` is byte-identical to the CUDA default archive (same130075bytes/SHA256), retaining that feature-build artifact separately. The snapshot has no `.git`, so no `.cargo_vcs_info.json` member is generated; exact commit provenance is instead retained by the complete source manifest and Git-blob audit.

The historical17/34/31/39 counts are not reused: current counts are17/39/60/41. Newer tracked source is covered by the current archive-member audit.

## Actual member-byte audit

`package-audit.json`, per-crate `manifests/*.members.json`, normalized `*.Cargo.toml.normalized`, generated package `*.Cargo.lock`, `external-lock-audit.json`, and `checks.json` record:

- Every actual archive member is either an exact fixed-commit crate file or the expected Cargo-generated `Cargo.toml`, `Cargo.toml.orig`, or `Cargo.lock`; the original manifest itself equals the fixed commit. Every README matches its fixed current crate README.
- Full `LICENSE`:1075bytes, SHA256 `37b50b0dc885fe0e9306cca59347532477623f058c9c496d60f153c535e20f87`. Full `THIRD_PARTY_NOTICES`:10077bytes, SHA256 `fa1b18275f507a15f821863a0e239f0b3859b858d647cab497b0f072430a0769`. All eight package entries are regular files matching the complete root originals, with source mode0644.
- The complete1995-byte Unicode V3 notice exactly matches cached `unicode-ident1.0.27/LICENSE-UNICODE`, SHA256 `f7db81051789b729fea528a63ec4c938fdcb93d9d61d97dc8cc2e9df6d47f2a1`. The policy remains scoped to `unicode-ident =1.0.27` and the Unicode-table-only exception; the crates remain MIT. No policy expansion occurred.
- The new complete1023-byte selected rustix MIT notice exactly matches cached rustix1.1.6 `LICENSE-MIT`, SHA256 `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3`.
- Normalized internal dependencies/dev-dependencies use renamed `ckks-train-*`, version1.0.0 and registry-style declarations without local `path`; no old `hekit-*` dependency name remains.
- CUDA contains all5 exact kernel assets, including canonical `.cu` hashes: basis `f339bcb5492fee28e5041f7c3097aea6326b2024f0f9a4be2855fcd21bafef0a`; keyswitch `8001e378323e64173700f82aa734d042d75c08af8aaad906864120dcd0ea5014`; ntt `fc487dfe9190c712c4a27f862100019d57afe7f501ce17acbc31691155b315da`; pointwise `d7da2002c30ff52256ffed4ba8b88117effaf3f039f6e44ced84db7666b1541b`. Shim bytes also match the fixed source.
- No `_ext`, `.git`, target, `.superpowers`, data/results tree, `.env`, credential/private-key filename, external tool checkout or unowned input appears in any archive. Every nongenerated member has exact fixed Git-byte provenance. Scoped PEM/AWS/GitHub/OpenAI credential-pattern scans found0hits. This is package-content verification, not a library security or whole-history audit.

## Bounded cache cleanup and freeze

After all archives/logs/manifests and successful audit evidence were retained, removed **only** `/Users/marconi/Documents/research/hekit/tools/release/_ext/package-current-2dc049a/build-cache`. It contained556 files, 269853342 logical bytes / 271339520 allocated bytes. `build-cache-before-removal.json` inventories it; `cleanup.json` records deletion and confirms the path is absent.

Working area before deletion: 501914005 logical bytes / 519069696 allocated bytes. After deletion: 232060663 logical bytes / 247730176 allocated bytes (about232MB retained, minor final report/manifest additions excluded from that instant). The own peak was about519MB allocated, below the1GiB target. Host free space before/after deletion: 5733191680 → 5960351744bytes; concurrent host activity can affect that delta.

Retained source pins, exact5archives, logs, normalized manifests, checks and source/audit scripts are frozen for root review. `evidence.sha256.jsonl` hashes the evidence files outside the separately manifested source/vendor trees; `freeze.json` pins that manifest and this report. No authored tracked source/docs, git index, commit, push, registry, workflow, native/server or research state was changed by this task. The ledger report is the only task-authored document outside the ignored working area.
