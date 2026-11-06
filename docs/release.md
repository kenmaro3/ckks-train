# Releasing ckks-train v1.0.0

State on 2026-11-06 (Task 55): the release is prepared, nothing is published. The GitHub
repository `kenmaro3/ckks-train` is **private**; the owner decides when it becomes public and
whether the crates go to crates.io. The user has authorized milestone pushes to the private
main branch and the ckks-train crate rename. Making the repository public, publishing crates,
and creating a public release remain separate publication decisions.

## 1. Checks before any release step

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features cuda -- -D warnings
cargo test --workspace
cargo deny check licenses bans
cargo doc --workspace --no-deps                          # must print no warning
cargo doc --workspace --no-deps --features hekit-cuda/cuda
```

## 2. Crate names (decided; apply before the first publication)

The user has chosen ckks-train and requested the crate rename; no further naming approval
is needed. The crates still carry the working name `hekit-*` while the current source writers
finish. Applying the rename and validating the final tree remain release preparation tasks.
`scripts/rename.sh` renames `hekit-{core,ckks,nn,cuda,bench}` to `ckks-train-*` in package
names, dependency keys and paths, `hekit_*` to `ckks_train_*` in Rust paths, and moves the
crate directories (`git mv`). It leaves `results/`, `spikes/`, `kat/` and
`docs/quadruped-log.md` alone (they record the names their results were produced with) and
other spellings such as `HekitError` or `HEKIT_*`.

```sh
scripts/rename.sh --dry-run | less     # every file and line that would change, and a summary
scripts/rename.sh                      # apply (refuses to run on a dirty working tree)
cargo check --workspace --all-targets && cargo test --workspace
```

On 2026-11-06, at the Task 55 commit, the dry run reported 822 lines (875 occurrences) in 155
files plus 5 directory moves; 979 lines keep another spelling of "hekit". Applied in a
throwaway clone (not in this repository), the renamed workspace passed
`cargo check --workspace --all-targets --features cuda`. Crate names on crates.io are
permanent once published. Apply the already decided rename before step 4.

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

Order: `hekit-core`, `hekit-ckks`, `hekit-nn`, `hekit-cuda` (or the renamed crates).
`hekit-bench` has `publish = false`. Every path dependency carries `version = "1.0.0"`, and
each publishable crate ships `LICENSE` and `THIRD_PARTY_NOTICES` (symbolic links to the
repository root, which cargo packages as files) and its own `README.md`.

```sh
cargo publish -p hekit-core
cargo publish -p hekit-ckks     # after hekit-core 1.0.0 is visible in the index
cargo publish -p hekit-nn
cargo publish -p hekit-cuda
```

Dry runs on 2026-11-06 (Rust 1.85.1):

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

The docs.rs build of `hekit-cuda` enables the `cuda` feature (`[package.metadata.docs.rs]`);
`cudarc` loads CUDA at run time, so no toolkit is needed to build the documentation.

## 5. CI

`.github/workflows/ci.yml` runs fmt, clippy (with and without `cuda`), a compile check of the
`cuda` feature, the tests, and `cargo deny check licenses bans` on `ubuntu-latest` for every
push to `main` and every pull request.

`act` is not installed on the development machine (`which act`: not found on 2026-11-06), so
**the workflow was not run locally with `act`**. Instead, the commands of each job were run
directly on macOS (Rust 1.85.1) for Task 55, all passing: `cargo fmt --all --check`, both
clippy invocations, `cargo check --workspace --all-targets --features cuda`,
`cargo test --workspace -j 4` and `cargo deny check licenses bans`. That is not a run of the
workflow on Linux. The private-main push did start the real
[CI run for 0daf778](https://github.com/kenmaro3/ckks-train/actions/runs/42136309953):
all five jobs (fmt, clippy, cuda-check, test and deny) completed successfully. This run does not validate
later source changes or the eventual renamed tree. Record the final SHA and completed
workflow result before release.

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
