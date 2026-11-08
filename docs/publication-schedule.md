# 48-hour publication schedule — 2026-11-08

The user explicitly requires a preview within one day and the paper/publication within
two days. Received at approximately2026-11-08 09:14 JST. Use conservative deadlines:

- **Preview ready: 2026-11-09 09:00 JST** (2026-11-09 00:00 UTC).
- **Code/release and paper PDF public: 2026-11-10 09:00 JST** (2026-11-10 00:00 UTC).

This supersedes the earlier1–3-day preview /1–2-week paper estimate. Publication intent
is explicit; preparation must produce concrete reviewable artifacts before publishing.
The exact additional destinations (arXiv/crates.io) have been asked asynchronously;
continue GitHub code/release/PDF preparation immediately. Formal venue submission,
authorship metadata, account access and third-party processing are separate constraints;
do not imply an arXiv announcement or conference acceptance by the deadline.

## Critical path

1. Complete exact candidate00f4409 verification. Workspace tests are already running
   on the isolated checkout; finish Clippy default/CUDA, rustdoc and necessary package/
   public-content review. Fix actual failures, avoid unrelated improvements. Keep the
   current supervisor WIP out of the candidate unless complete and reviewed in time.
2. By2026-11-09 09:00 JST assemble source/release notes, reproducible existing tables,
   PDF and validation summary. Initial90851d2 archive remains a failed qualification
   attempt, not the deliverable. The test-isolation correction has independent review.
3. Continue FULL-2 and the existing sequential GPU/additional-model queue. Freeze the
   paper's evidence selection by2026-11-09 18:00 JST, at the latest; remaining runs are
   reported as unmeasured and can be added later. Do not stop or restart existing jobs
   merely to meet this writing cutoff. Preserve raw binary JSON and existing gates.
4. Use the second day to finalize the actual contribution, comparison caveats, generated
   tables, limitations and PDF. Finish numerical/source checks and visual inspection,
   then publish the prepared GitHub artifacts before2026-11-10 09:00 JST.

## Scope and reasons

The original full research plan remains recorded and active. Its numerical gates are
not lowered; unmet/unmeasured results are not rewritten as passed. Full21-profile /
10-seed execution and unmeasured GPU bootstrap are not prerequisites for this first
public research artifact. Reproducibility claims cover what is actually executable;
regenerating archived tables is not a claim that every experiment has completed.

Reason: the user's new deadline prioritizes access to implemented code and an honest
paper over waiting for exhaustive evaluation and infrastructure completion. Waiting
for every optional/extended measurement risks missing the explicit deadline. Cost:
the first public version has narrower supported claims and clearly listed remaining
work; later results require an updated artifact/paper. This is staged publication,
not a declaration that all original Tasks42/50/54/57 gates have passed.

Author/placeholders, mainB and same-N interpretation remain unresolved as previously
instructed. Do not invent author consent, affiliation, a paper license or completed
third-party submission. Prepare the GitHub draft/PDF without blocking independent work.
