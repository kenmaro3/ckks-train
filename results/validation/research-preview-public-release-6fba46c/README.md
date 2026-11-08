# Public research preview — 2026-11-08

**Published:** [v1.0.0-preview.1](https://github.com/kenmaro3/ckks-train/releases/tag/v1.0.0-preview.1),
release437943352, at2026-11-08T01:42:51Z (10:42:51JST).
This is a prerelease, not stable v1.0, arXiv submission or completion of the research plan.

Source/tag: `6fba46c38f1cdb5be88b9bf1eeecab0f4566b534`.
All5,316 archive files match Git content/mode. The source tar.gz is56,389,315bytes,
SHA256 `51a73fd728f80158b790a9fb0f04f60d53ccd77fab0ea4b24bb6346fe3d3ea0a`.
The eight public assets include source, PDF, arXiv source ZIP, manuscript notice,
preview notes, inventory, manifest and checksums. Their API sizes/digests match the
reviewed local bundle. The old0b2d17a draft bundle and records remain historical;
its superseded source asset was removed from this release after the new upload verified.

## Verification scope

- [Exact-source CI42433577945](https://github.com/kenmaro3/ckks-train/actions/runs/42433577945)
  completed/success, all6jobs successful. Actual workspace log:75 target summaries,
  1,028passed/0failed/47ignored (includes doctests; ignored cases unexecuted).
  Raw API and full log ZIP are in `ci/`; no earlier run's pass is inherited.
- `reproduction/` retains exact-source clean-checkout `run_all.sh --tables-only`,
  `tables.py --check`, `figs.py --check`: all exit0, no tracked changes.
  These regenerate archived tables; they do not rerun the research experiment matrix.
- `review.md` independently approves SPEC/QUALITY and required the subsequent exact
  CI gate. `ci/capture.json` and `upload-verification.json` prove that gate and the
  completed upload before publication. The preparation-time pending CI status in
  `integrity.json` and the review remains an honest historical snapshot.
- The19-member arXiv ZIP reuses18 matching repository manuscript inputs plus generated
  bibliography. The locally built/rendered10-page PDF is unchanged. Its named author
  is Kentaro Mihara; softwareMIT and the manuscript license remain separate.
- `publication/verification.json` records anonymous HTTP200 for the release and PDF,
  exact tag commit, all8asset API digests, and downloaded PDF hash/length equality.
  PDF:526,897bytes, SHA256 `15213f545e4553e7262f388dd57d8b296aef761b699205701840b0504503ca50`.

FULL-2, the full-setting ReBoot CPU timing, equal-security condition, GPU training/
ablations/bootstrap and full Task54 matrix remain pending, unmet or unmeasured as
specified in the release notes. Uncommitted supervisor work is excluded. The arXiv
server has not compiled the source package and no arXiv submission exists.

The initial packaging verification failed on the top-level tar directory's missing
trailing slash; the error and script are retained. Resumed verification skipped
directory records before checking file prefixes and verified the unchanged archive.
This is packaging validation, not a scientific-result JSON edit.
