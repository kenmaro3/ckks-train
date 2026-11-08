#!/usr/bin/env python3
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parent; REPO=ROOT.parents[3]
L=REPO/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5'
a=json.loads((ROOT/'package-audit.json').read_text());c=json.loads((ROOT/'cleanup.json').read_text());e=json.loads((ROOT/'execution.json').read_text())
e['archive_audit_final']={'argv':['python3','tools/release/_ext/package-current-2dc049a/audit.py'],'exit':0,'stdout':'logs/archive-audit-final.stdout','stderr':'logs/archive-audit-final.stderr'}
(ROOT/'execution.json').write_text(json.dumps(e,indent=2,sort_keys=True)+'\n')
x=json.loads((ROOT/'checks.json').read_text());x['all_four_external_package_locks_match_fixed_workspace_lock']=True;(ROOT/'checks.json').write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
rows='\n'.join('| '+p['crate']+' | '+str(p['members'])+' | '+str(p['bytes'])+' | `'+p['sha256']+'` |' for p in a['packages'])
external=len(json.loads((ROOT/'external-directory-source.json').read_text()))
report=f'''# Task 55 current packages: fixed 2dc049a qualification (2026-11-08)

Status: **PASS for current macOS package builds and actual archive content**. All four publishable crates and the CUDA-feature package verification completed with exit0, offline and locked. Root independent review is pending. No registry upload or publish command was run. This does not establish stable-v1.0 completion, exact-final-SHA Linux CI, native GPU execution, training, or a scientific result.

## Fixed input and isolation

- Source commit: `2dc049a796949b685a5c23f8b05440ee2ce474aa`.
- Working/evidence root: `{ROOT}` (ignored).
- Complete tracked export: `source.tar.gz`; extracted fixed input: `source/`.
- `manifests/source.sha256.jsonl`: **{a['source_entries']} entries, {a['source_bytes']} logical bytes**, SHA256 `{a['source_manifest_sha256']}`.
- Every manifest entry was checked against its exact Git blob identity and Git permission mode; no extra/missing tracked entry exists. Active writer WIP and untracked files were not copied. The source manifest was rechecked after packaging and cleanup.
- The four crate trees, `Cargo.lock` and `deny.toml` have no diff between preview `6fba46c` and this fixed commit (saved `preview-package-input-diff.json` and stdout/stderr). Later HEAD movement does not retarget this evidence.
- Toolchain: Cargo1.85.1 (`d73d2caf9`), rustc1.85.1 (`4eb161250`), aarch64-apple-darwin. `CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=0`, `CARGO_NET_OFFLINE=true`, and task-only `CARGO_TARGET_DIR={ROOT/'build-cache'}`.

The prior Task55 root `{pathlib.Path('/var/folders/l9/j0s8_fkd3b53h97864_x2kq40000gn/T/ckks-train-renamed-packages-itehj5x0')}` was read only. The new `vendor/` references its **{external} external cached crate directories** through symlinks; prior internal ckks-train/hekit package directories were excluded. Each newly produced internal archive was extracted only into this task's `vendor/` with exact file hashes and archive checksum. All referenced external directory-source checksum files remained unchanged. No new dependency fetch, install or network update occurred. The prior target and shared repository target were not used or removed.

## Build commands and outcomes

The exact common command was:

```sh
cargo --config {ROOT/'local-source.toml'} package \\
  -p <crate> --offline --locked -j 2
```

Serial order: `ckks-train-core`, `ckks-train-ckks`, `ckks-train-nn`, `ckks-train-cuda`; then CUDA again with `--features cuda`. All **5 exit0**, no warning/error lines, no lock fallback or source-manifest rewrite. `cargo package` compiled its extracted package verification source. These are packaged-library builds; packaged tests, benchmarks, GPU kernels and runtime were not executed. All external entries in all four generated package lockfiles match the fixed workspace lock's exact version/source/checksum. Internal package-lock checksums equal the newly created dependency archives.

All command argv/cwd/start/exit and separate stdout/stderr paths are retained in `commands.json` and `execution.json`, with every log in `logs/`. The complete packaging stderr logs were inspected. No failed Cargo or content-audit attempt occurred. Exploratory directory-inventory misses are not treated as qualification failures.

## Retained exact archives

| Package | Members | Compressed bytes | SHA256 |
|---|---:|---:|---|
{rows}

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

After all archives/logs/manifests and successful audit evidence were retained, removed **only** `{ROOT/'build-cache'}`. It contained{c['removed_cache_files']} files, {c['removed_cache_logical_bytes']} logical bytes / {c['removed_cache_allocated_bytes']} allocated bytes. `build-cache-before-removal.json` inventories it; `cleanup.json` records deletion and confirms the path is absent.

Working area before deletion: {c['working_area_before_logical_bytes']} logical bytes / {c['working_area_before_allocated_bytes']} allocated bytes. After deletion: {c['working_area_after_logical_bytes']} logical bytes / {c['working_area_after_allocated_bytes']} allocated bytes (about232MB retained, minor final report/manifest additions excluded from that instant). The own peak was about519MB allocated, below the1GiB target. Host free space before/after deletion: {c['free_before_bytes']} → {c['free_after_bytes']}bytes; concurrent host activity can affect that delta.

Retained source pins, exact5archives, logs, normalized manifests, checks and source/audit scripts are frozen for root review. `evidence.sha256.jsonl` hashes the evidence files outside the separately manifested source/vendor trees; `freeze.json` pins that manifest and this report. No authored tracked source/docs, git index, commit, push, registry, workflow, native/server or research state was changed by this task. The ledger report is the only task-authored document outside the ignored working area.
'''
report_path=L/'task-55-current-packages-20261108-report.md';report_path.write_text(report)
rows=[]
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.is_symlink():continue
    rel=p.relative_to(ROOT)
    if rel.parts[0] in ('source','vendor') or str(rel) in ('evidence.sha256.jsonl','freeze.json'):continue
    rows.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
m=ROOT/'evidence.sha256.jsonl';m.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
freeze={'fixed_source_commit':a['source_commit'],'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'evidence_manifest':str(m),'evidence_manifest_sha256':hashlib.sha256(m.read_bytes()).hexdigest(),'report':str(report_path),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),'source_manifest_sha256':a['source_manifest_sha256'],'independent_review':'pending'}
(ROOT/'freeze.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
print(json.dumps(freeze,indent=2))
