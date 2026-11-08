# Releasing ckks-train v1.0.0

State on 2026-11-06 (Task 55): the release is prepared, nothing is published. The GitHub
repository `kenmaro3/ckks-train` is **private**; the owner decides when it becomes public and
whether the crates go to crates.io. The user has authorized milestone pushes to the private
main branch and the ckks-train crate rename. Making the repository public, publishing crates,
and creating a public release remain separate publication decisions.

Current qualification (2026-11-08): the dated Task55 preparation below is historical.
Subsequent source changes and the ongoing Task54 implementation require validation of
the final release candidate; the earlier package/CI passes do not cover that candidate.
The two CI runs in the 07:09 JST observation below did not execute any test steps. No final release
readiness or publication is established by this document.

## 1. Checks before any release step

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features cuda -- -D warnings
cargo test --workspace
cargo deny check licenses bans
cargo doc --workspace --no-deps                          # must print no warning
cargo doc --workspace --no-deps --features ckks-train-cuda/cuda
```

## 2. Crate names (applied on 2026-11-06)

The workspace now uses `ckks-train-{core,ckks,nn,cuda,bench}` and Rust paths
`ckks_train_*`. The user-requested `scripts/rename.sh` migration changed 903 occurrences
on 847 lines in 158 tracked text files and moved five crate directories. Rustfmt then
normalized import order and line wrapping. The root checkout directory remains `hekit`;
`HekitError` and `HEKIT_*` remain compatible API/environment names.

Historical results, spikes, known-answer vectors and `docs/quadruped-log.md` retain their
original names. The historical `tools/reboot_layout/assemble.py` also retains its original
bytes: its provenance describes old commits, so rewriting those paths would be false. All 755 tracked files in those excluded paths have identical SHA-256
hashes before and after the migration. The earlier Task55 dry run (822 lines, 155 files)
was performed before the final paper and runner additions; it is not this migration's count.

The migration script is retained as the original `hekit-*` migration procedure. It does
not publish packages. Crate names on crates.io become permanent only when published.
The full workspace test caught two fixed-source-hash failures because five CUDA kernel
comment lines had changed their crate paths. The executable CUDA code was unchanged.
We retain the exact four previously reviewed kernel files and their existing hash pins,
and exclude `crates/*-cuda/kernels/*.cu` from the migration script. Their old paths in
comments are historical references. Repinning comment-only changes was an alternative,
but retaining the byte identity preserves the direct link to existing GPU evidence.
Post-rename macOS checks passed: workspace tests **462 passed, 12 ignored**; fmt;
all-target CUDA compilation; Clippy with/without CUDA; rustdoc with/without CUDA under
`RUSTDOCFLAGS="-D warnings"`; and cargo-deny licenses/bans (only the unused BSD-3-Clause
allowance warning). All **49 Python tests** passed, both shell scripts parsed, and
`run_all.sh --tables-only` reproduced the existing tables. Table/figure freshness and
all 755 historical-file hashes still match. The paper sources and generated artifacts
were unchanged by the migration, so their reviewed 9-page PDF builds remain applicable.
The renamed commit `09202cf5650aa76b9b30dd79a68dd615c75a91ae` has its own successful [Linux CI run42143028484](https://github.com/kenmaro3/ckks-train/actions/runs/42143028484), all six jobs. This result applies only to that SHA; queue commit `4b1c499` now also has run42145577051 completed/success for all six jobs (section 8). The pending state in section 7 remains the dated 10:49 observation.

## 3. Publishing to GitHub (owner's decision)

1. Review what becomes public. The whole history becomes visible, not only `main`'s tip:
   - no secrets in the history (`git log -p | grep -iE 'password|token|secret|BEGIN .*KEY'`,
     or a scanner such as gitleaks);
   - no GPL code: ReBoot is only run from a separate environment; `spikes/**/_ext/` (where the
     ReBoot checkout lives) is gitignored and no object under it is in the history (checked
     2026-11-06: 0 objects). Values from ReBoot's configuration are committed as data only;
   - no large files (checked 2026-11-06: no blob over 5 MB);
   - `LICENSE` names "hekit contributors" as the copyright holder; change it if the holder
     should be a person or an organization.
2. Decide the owner: keep it under the personal account `kenmaro3`, or transfer it to an
   organization (`gh api repos/kenmaro3/ckks-train/transfer -f new_owner=<org>`); GitHub
   redirects the old URL, but the `repository` field in the manifests should then be updated.
3. Set the description and topics, e.g.
   `gh repo edit kenmaro3/ckks-train --description "RNS-CKKS with bootstrapping and CUDA for encrypted neural-network training" --add-topic homomorphic-encryption --add-topic ckks --add-topic rust --add-topic cuda`.
4. Make it public: `gh repo edit kenmaro3/ckks-train --visibility public --accept-visibility-change-consequences`.
5. Check that the CI workflow runs green on GitHub (section 5).
6. Tag and release: `git tag -a v1.0.0 -m "ckks-train v1.0.0"`, `git push origin v1.0.0`,
   then `gh release create v1.0.0 --title "ckks-train v1.0.0" --notes-file <the 1.0.0 section of CHANGELOG.md>`.
   Replace "unreleased" in `CHANGELOG.md` with the release date first.
7. Optional, for the paper's artifact: archive the release on Zenodo (GitHub integration) to
   get a DOI, and fill in the citation in `README.md` once the author list is fixed.

## 4. Publishing to crates.io (owner's decision)

Order: `ckks-train-core`, `ckks-train-ckks`, `ckks-train-nn`, `ckks-train-cuda`.
`ckks-train-bench` has `publish = false`. Every path dependency carries `version = "1.0.0"`, and
each publishable crate ships `LICENSE`, `THIRD_PARTY_NOTICES`, and its own `README.md`.
Since commit `a29dcf6`, the eight license/notice entries are regular mode0644 copies,
not symbolic links. All eight currently match the root originals byte for byte.
Whenever a root notice changes, synchronize all four crate copies before packaging.
The [source-custody record](../results/validation/task54-d7-fixture-custody/README.md)
explains this change. The package checks below predate that conversion and subsequent
source changes; inspect and build the actual final archives again before release.

```sh
cargo publish -p ckks-train-core
cargo publish -p ckks-train-ckks     # after ckks-train-core 1.0.0 is visible in the index
cargo publish -p ckks-train-nn
cargo publish -p ckks-train-cuda
```

Pre-rename dry runs on 2026-11-06 (Rust 1.85.1, original `hekit-*` names):

| crate | `cargo publish --dry-run` | packaging check |
|---|---|---|
| hekit-core | **passes** (18 files, 166.3 KiB; the packaged crate builds) | — |
| hekit-ckks | fails: `no matching package named hekit-core found` (crates.io index) | `cargo package --list`: 35 files; builds from a local source (below) |
| hekit-nn | fails: `no matching package named hekit-ckks found` | `cargo package --list`: 32 files; builds from a local source |
| hekit-cuda | fails: `no matching package named hekit-ckks found` | `cargo package --list`: 40 files, kernels included; builds from a local source, with and without `cuda` |

The three failures are expected: cargo 1.85 resolves a package's dependencies against
crates.io when it packages it, and `hekit-core` / `hekit-ckks` are not published yet.
(Packaging several workspace crates together, `cargo publish --workspace`, needs a newer cargo
than the pinned 1.85.) To check the packages themselves, the chain was packaged in a
throwaway clone with crates.io replaced by a local directory source holding the vendored
third-party crates plus each `hekit-*` `.crate` as it was produced: `cargo package` (which
builds the packaged crate) passed for hekit-core, hekit-ckks (35 files), hekit-nn (32 files)
and hekit-cuda (40 files), and for hekit-cuda also with `--features cuda`. So each dry run
should pass once its dependencies are on crates.io, provided nothing else changes. The `hekit-cuda` dev-dependency on `hekit-nn` is versioned too, so its
packaged tests build once `hekit-nn` is published.

### Post-rename package verification (2026-11-06)

The renamed `ckks-train-core` registry publish dry run also succeeded and explicitly
aborted upload. The other three registry dry runs still fail because `ckks-train-core`
or `ckks-train-ckks` is unpublished. An isolated copy of the actual renamed source was
packaged using an offline directory source: all four extracted packages built, and
`ckks-train-cuda` also built with `--features cuda`. The snapshot excludes `.git`, so its
archives have 17 / 34 / 31 / 39 files respectively (no `.cargo_vcs_info.json`).

The archive audit checks the complete LICENSE and THIRD_PARTY_NOTICES, including the
approved Unicode V3 notice, the current crate READMEs, normalized internal dependency
names and the unchanged CUDA kernels. No external tool checkouts or credential-pattern
hits were found in those archives. This is an artifact-content check, not a security
audit of the library. No package was uploaded; actual registry publication remains a
separate operation in dependency order.

The docs.rs build of `ckks-train-cuda` enables the `cuda` feature (`[package.metadata.docs.rs]`);
`cudarc` loads CUDA at run time, so no toolkit is needed to build the documentation.

## 5. CI

`.github/workflows/ci.yml` runs fmt, clippy (with and without `cuda`), a compile check of the
`cuda` feature, the tests, and `cargo deny check licenses bans` on `ubuntu-latest` for every
push to `main` and every pull request. A research-tools job also runs the Python regression
tests, checks shell syntax and verifies generated tables/figures against archived inputs.
These checks do not launch training runs or claim hardware measurements.

`act` is not installed on the development machine (`which act`: not found on 2026-11-06), so
**the workflow was not run locally with `act`**. Instead, the commands of each job were run
directly on macOS (Rust 1.85.1) for Task 55, all passing: `cargo fmt --all --check`, both
clippy invocations, `cargo check --workspace --all-targets --features cuda`,
`cargo test --workspace -j 4` and `cargo deny check licenses bans`. That is not a run of the
workflow on Linux. The private-main push did start the real
[CI run for 0daf778](https://github.com/kenmaro3/ckks-train/actions/runs/42136309953):
all five jobs (fmt, clippy, cuda-check, test and deny) completed successfully. This run does not validate
later source changes or the eventual renamed tree. Record the final SHA and completed
workflow result before release. The later observation below updates the renamed SHA only.

## 6. Dependency policy audit (2026-11-06)

The general allow-list is MIT / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / ISC. The
inherited config also allowed Unicode-3.0 and Zlib globally. A metadata audit found
`unicode-ident 1.0.27` declares `(MIT OR Apache-2.0) AND Unicode-3.0`; the Unicode
component is a conjunctive requirement and cannot be dismissed by choosing MIT.

The user approved the Unicode character-property tables only. The global Unicode
and unused Zlib allowances were removed; a `licenses.exceptions` entry now permits
the Unicode requirement only for `unicode-ident =1.0.27`. Its full Unicode V3 notice
is in THIRD_PARTY_NOTICES. A future version or another crate needs a fresh audit.
The strict four-family check failed as expected before the exception; the active
scoped policy passes `cargo deny check licenses bans` (cargo-deny 0.20.2), with only
an unused BSD-3-Clause allowance warning. No GPL-family or BSD-3-Clause-Clear
exception was introduced. Dataset attribution is separate from code dependencies;
Iris/WDBC credits and separately downloaded MNIST/Fashion-MNIST sources are recorded
in the notices and data README.


## 7. Waiting measurement queue deployment (2026-11-06 10:49:44 JST)

The approved root controller uploaded and hash-verified the independently reviewed queue from source commit `4b1c49943533bb9ff79493daee33a27b8237a18c`, then observed PID1739838 / start ticks467041669 in **waiting** state, owner.json only and clones_created=false. No new build, training or measurement had started. The bundle HEAD is `09202cf5650aa76b9b30dd79a68dd615c75a91ae`; after both FULL successes and chain10 terminal/inactive identity validate, separate fresh clones use exact historical commits bd5cc49 / 0e07561 / 9305d2e for R47/R48/R52, sequentially. R52 is 21-update subsets only. R54 is not queued; Task54's original full matrix remains implementation-incomplete while its extension is designed.

The [operations record](post-full-queue.md) contains exact paths, full commits, config/script/bundle/source-record SHA256 and recovery rules. Canonical raw results remain unchanged. Waiting deployment is distinct from successful measurement and from publication. No public release, tag, crates.io publish or arXiv submission occurred.

The renamed SHA09202cf has Linux CI run42143028484 **success** for all six jobs (fmt, clippy, cuda-check, test, deny, research-tools), captured in task-55-rename-ci.json. Queue SHA4b1c499's [run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) is pending at this observation; the prior CI success does not apply to it. Neither CI result proves GPU execution or a research gate.


## 8. Follow-up observation and queue CI completion (2026-11-06 11:09:14 JST)

At Unix1793930954.5309272, the read-only key-auth observation recorded FULL-1 update144 (zero-based), **145/750 updates complete**, elapsed13851s. Its last three progress lines (142–144) all matched prediction; no post terminal or partial marker was recorded. FULL-2 had no progress or post terminal. Chain10 and the waiting controller retained start ticks465769819/467041669 respectively; state contained owner.json only and no clones existed. No new queued build/training/measurement had started. This supersedes the progress knowledge at 10:49 without rewriting that snapshot. R54 and full eMLP-2/3 runs remain unqueued.

Root subsequently verified queue commit `4b1c49943533bb9ff79493daee33a27b8237a18c` / [CI run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) **completed/success, all six jobs success**: fmt, test, deny, research-tools, cuda-check and clippy. The renamed SHA09202cf's earlier success remains a separate record. This replaces the current pending claim for 4b1c499; it does not validate later commits, GPU measurements or a research gate. No publication occurred.

Evidence retained in the ledger: `post-full-followup-observation.stdout` (SHA256 `eeae57cf21919e2011c61baf26bdccaf4486e502a325e1bdf15d5a58f277edb9`) and `task-post-full-ci.json` (SHA256 `fd1c7d740e6fa64461f0185640a5802b3149af2b7a8f600f01b56b691e5e1a09`). Root supplied the administrative CI run/SHA mapping. Original research JSON is unchanged.

## 9. rustix selected-MIT notice (2026-11-08)

THIRD_PARTY_NOTICES now records the MIT alternative selected for rustix1.1.6 and reproduces its complete pinned LICENSE-MIT text. Commit71d3f09 added the direct Linux std/fs edge to ckks-train-bench with default features disabled; the locked version/checksum already existed. Bench remains publish=false. Existing notices, dependency versions/features, deny policy and the Unicode-table-only exception are unchanged by this documentation update.

The [candidate and independent review](../results/validation/task55-rustix-notice/README.md) passed exact byte/manifest/history checks. This is repository notice preparation, not a new package build, final-SHA CI result, native Linux validation or registry publication. The latest source and distribution contents still require final release verification once implementation is complete.

## 2026-11-08 07:09 JST: CI未起動の確認

a29dcf6（run42416425934）と767fb52（run42418803177）は各6ジョブがcompleted/failureだったが、全ジョブのstepは0件だった。各checkのannotationは、支払い失敗または利用上限による未起動を示している。どちらの課金条件かはAPIから区別できない。[全ジョブの一次記録とSHA対応](../results/validation/ci-20261108-0709/README.md)を保存した。コードのテスト結果は得られておらず、以前のSHAのCI合格を引き継がない。課金設定・支払い・workflowの変更や再実行は行っていない。ローカルの実装・検証は継続できるが、最終公開候補のCI検証は未達として残す。

## Research preview preparation (2026-11-08)

The user has requested greater speed toward publication. Prepare a fixed research preview
in parallel with the remaining v1.0 and research work. The initial source candidate is
`90851d261b36b034d3b660ec4c95dd330eadc616`, isolated from uncommitted Task54 changes.
This is a candidate selection, not a validation result or permission to publish. The
1–3 day estimate is a planning target, subject to candidate validation.

The preview will identify measured, unmet and unmeasured results, include the existing
paper draft with its original build provenance, and distinguish table regeneration from
full experiment execution. Its existing1.0.0 version fields do not mean a stable v1.0
has been qualified or published. Final-candidate tests, license/package checks and public
history review still apply. No scientific gate or original full-matrix obligation is removed.

Waiting for every additional evaluation would delay access to existing work; including
mutable, unreviewed source would prevent a meaningful qualification. A fixed preview is
the intermediate deliverable. The cost is separate validation and explicit source/PDF
provenance, followed by validation again for the eventual final version. The existing
FULL-2 and post-full experiment sequence continues unchanged.

The initial preview candidate failed one integration-test assertion caused by shared test
counters. The minimal test-only correction has independent SPEC/QUALITY approval;
full qualification of the updated candidate remains pending. See
[preview preparation and evidence](research-preview.md). No initial archive is promoted
to a validated release by this correction.
