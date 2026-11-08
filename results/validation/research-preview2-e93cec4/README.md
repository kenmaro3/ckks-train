# Preview 2 preparation and handoff — 2026-11-08

This is **local preparation**, not publication, a new experiment, or arXiv submission.
The user switched the immediate task to review and agent-handoff documentation before
release notes, asset manifest/checksums, upload or publication were performed.
The user explicitly retained responsibility for arXiv submission.

Source: `e93cec434652a8c9f8c48becc6eb3045a3f78a7f`.
Bundle: `/Volumes/Untitled/ckks-train-storage/archives/research-preview2-e93cec4/`.
No `v1.0.0-preview.2` tag/release was created by this session.

- `prepare.py` records the actual packaging procedure. It refuses an existing output
  directory; do not delete the candidate merely to rerun the script.
- `integrity.json`: 6,113 archive files match Git blob and mode; all 18 manuscript
  inputs match the previously built paper; 19 ZIP members match their retained hashes.
- `scan.json`: no match for limited known credential patterns in uncompressed source
  archive files. This is not exhaustive secret detection, compressed-interior scanning
  or a new full-history audit.
- `reproduction.json` / `tables-only.log`: all three table/figure commands succeeded
  on the isolated fixed-source clone, leaving a clean tree. No training was run.
- `ci-observation.json`: raw GitHub API observation captured at approximately
  2026-11-08 22:31 JST; five jobs succeeded, test remained in progress. Refresh before
  publication. Earlier source success does not qualify this SHA or uncommitted code.
- `handoff-snapshot/`: status and per-file hashes, including 50 tracked modified and
  41 untracked files outside this new record. Snapshot evidence is not code approval.

The PDF and ZIP reuse the exact reviewed artifacts described in
[the paper verification](../paper-full2-r47-20261108/README.md). This session checked
identity, member contents and the ten-page PDF metadata; it did not rebuild or
repeat the previous all-page visual review. The default paper has not been changed.

See [the new handoff](../../../docs/agent-handoff-2026-11-08.md) for current decisions,
scientific limits, the worktree boundary, next steps and exact resumption commands.
