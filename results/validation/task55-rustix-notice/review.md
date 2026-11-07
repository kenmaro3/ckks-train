# Task55 rustix selected-MIT notice — independent documentation review

SPEC: **PASS**. QUALITY: **PASS**. No Critical, Important or Minor finding. The root-authored candidate accurately records the selected MIT alternative and preserves the source license text. This review does not publish the notice, release a package, assess broader legal compliance, or change a dependency/exception.

## Exact reviewed inputs

Read the complete brief, candidate and diff. Inspected only fixed commit `71d3f0937741a4ff8d156b36540332ea82203d3c` and its parent for dependency history, the fixed section9 ratification/helper, and the installed pinned rustix manifest/MIT license. No concurrent retained-reader source was inspected. The author previously implemented the MIT-selected dependency edge; this notice candidate was authored by root and independently reviewed here.

| Input | Bytes | SHA256 |
|---|---:|---|
| `brief.md` | 701 | `8fe41932a32ea4bac5a6ec9509bf2aaf040c1d4b4b8784cfe320eb72220cc72d` |
| `pins.json` | 479 | `4fac83c908252475e8d4fbc64714f3a0d7bfc40d3483aa41b9311f6c8dca7693` |
| `THIRD_PARTY_NOTICES.candidate` | 10047 | `202a73802ec729312bd431da05c2b976a2544e61cbc245b227c8d6962555c711` |
| `notice.diff` | 2892 | `79a2dc12620f5d1d1aa47ac2d4e2cbd1faefaa56ce7f4a9bb62d7fc864e7caec` |
| Fixed/current actual `THIRD_PARTY_NOTICES` | 8106 | `618469340d937d6a88145a7a1f6e618220917f2624917830335a050ed3af9ba9` |
| Installed rustix1.1.6 `LICENSE-MIT` | 1023 | `23f18e03dc49df91622fe2a76176497404e46ced8a715d9d2b67a7446571cca3` |
| Fixed bench `Cargo.toml` | 1640 | `fc698bca60c78aa259180e6cd238514cb9e83bdb780a70e3c0287831b6a438b4` |
| Fixed `Cargo.lock` | 20584 | `e5059c771e93261c6fecf855f7a2d8ab052021f62cadafddbb23c43fa2f216c3` |
| Fixed `docs/study-gates-design.md` | 22604 | `073f9fec30a4818ae78b990e02439948886573a65804f03bb21a47b2f37baf5d` |
| Fixed `deny.toml` | 1263 | `62898cd841869e00b02aed52f7d966fd873b5d64b12152b01924c2704a4e8ee5` |
| Fixed `publication_anonymous.rs` | 4617 | `f6e8b4b4938ba3c30741b7d0f9fcc745395250d65abc8e3f4a91898c84c38038` |

Installed license path: `/Users/marconi/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rustix-1.1.6/LICENSE-MIT`. Its sibling installed `Cargo.toml` SHA256 is `43d4bcaf19bf857aaa17ada056e7561a50abdcb3201a1722770948ce050c77cb` and declares `Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT`.

## Verification and factual scope

Removing only the added rustix section from the candidate reproduces the entire8106-byte base notice exactly. All existing SEAL/data/acknowledgement/ISC/Unicode text is byte-identical. The supplied full diff exactly matches an independently generated eight-context-line unified diff of that base and candidate. The actual working-tree THIRD_PARTY_NOTICES still equals the fixed base; this reviewer did not apply the candidate.

The candidate's license reproduction, after removing the four-space presentation indentation and the preceding separator blank line, is byte-for-byte identical to all1023 bytes of the pinned LICENSE-MIT, including final newline and original line wrapping. No copyright/permission/disclaimer line was omitted or invented. The first read-only extraction included the separator newline (1024 bytes) and therefore failed a raw equality assertion; narrowing the extracted license to exclude that presentation separator yielded the exact1023-byte match. This was an extraction-boundary issue, not a content defect or a project test failure.

Fixed bench Cargo.toml confirms a normal dependency under `cfg(target_os = "linux")`, exact version `=1.1.6`, `default-features = false`, features `std` and `fs`, and `publish = false`. Section9 explicitly selects the MIT alternative. The fixed helper consumes safe rustix filesystem APIs for anonymous creation, held proc descriptors and linking; the notice accurately describes that use without claiming the surrounding historical code/dependencies are globally unsafe-free or that native behavior has been validated.

Comparing commit71d3f09 with its parent confirms the manifest edge was added in that commit and Cargo.lock changed only by adding rustix to the bench dependency list. The rustix package entry is identical before/after: version1.1.6, registry checksum `e5e76742e5294153d4dde1207dbec5bdd81eef6fabdc0d2118babadabfd858aa`. Thus “already present” and the precise direct-edge commit claim are supported. The new notice adds no dependency version, enabled feature, license exception or Unicode allowance.

The repository-versus-package language is appropriately bounded: bench remains unpublished, and the notice makes no crates.io-release or tested-binary-package claim. The independently written fixed API-consuming helper and the recorded implementation provenance support the statement that rustix implementation source was not copied; the license-text reproduction is clearly identified separately. This documentation review is not a fresh whole-repository source-provenance or legal audit.

Only read-only Git/file inspection, byte hashing/comparisons and TOML parsing were performed. No Cargo invocation, tests, build, runtime/native probe, SSH, subagent, source/Cargo/public-notice edit, commit or push occurred. The only write is this review.md. No test execution is needed for this notice-only review.
