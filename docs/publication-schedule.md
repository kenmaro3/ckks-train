# 48-hour publication schedule — 2026-11-08

The user explicitly requires a preview within one day and the paper/publication within
two days. Received at approximately2026-11-08 09:14 JST. Use conservative deadlines:

- **Preview ready: 2026-11-09 09:00 JST** (2026-11-09 00:00 UTC).
- **GitHub code/release/PDF public and arXiv submitted: 2026-11-10 09:00 JST** (2026-11-10 00:00 UTC).

This supersedes the earlier1–3-day preview /1–2-week paper estimate. Publication intent
is explicit; preparation must produce concrete reviewable artifacts before publishing.
The user has confirmed GitHub plus arXiv submission. crates.io is not part of this
48-hour publication request. Continue GitHub preparation and prepare the arXiv LaTeX
source package in parallel. Authorship and paper license are confirmed below; account/endorsement
status must still be established for submission. arXiv announcement/moderation and conference
acceptance are controlled externally and are not a guaranteed deadline outcome.

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
   then publish the prepared GitHub artifacts and complete arXiv submission before
   2026-11-10 09:00 JST, subject to actual account/endorsement availability.

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

MainB and same-N interpretation remain unresolved as previously instructed.
The user confirmed sole authorship: Kentaro Mihara, miha.ken.19@gmail.com, as a personal
project unrelated to their employer. Use Independent Researcher in the named manuscript;
do not include a company affiliation or invent a city/country. The paper license is the
arXiv non-exclusive distribution license. Software remains MIT. No submission has occurred.

## arXiv preparation requirements verified on2026-11-08

Official [submission guidelines](https://info.arxiv.org/help/submit/index.html) require
registered authors, appropriate scientific content and moderation, and TeX source
rather than a PDF generated from TeX. Prepare the source, bibliography and figures;
keep the GitHub PDF as a separate public artifact. Verify the arXiv-generated output.
The [license page](https://info.arxiv.org/help/license/index.html) says the selected
license for a posted version is irrevocable. The user selected the arXiv non-exclusive
distribution license on2026-11-08; software MIT does not decide the paper license. The
[endorsement instructions](https://info.arxiv.org/help/endorsement.html) require checking
eligibility for the selected category. Do not contact endorsers or invent their consent.

The official [registration guidance](https://info.arxiv.org/help/registerhelp.html)
allows unaffiliated authors to enter Independent as their organization. This supports
the chosen personal-research attribution; account affiliation must still reflect the
actual account policy and is not changed here.

Account check update (2026-11-08): the user reports a previous arXiv submission and
will check the account personally. They explicitly asked the agent to skip account
checking for now. Continue manuscript/source/GitHub preparation; do not infer that
current login or category endorsement has been verified.
