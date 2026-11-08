# ckks-train interim paper draft

The named 10-page [preview PDF](https://github.com/kenmaro3/ckks-train/releases/download/v1.0.0-preview.1/ckks-train-paper-preview.pdf)
and [arXiv preparation ZIP](https://github.com/kenmaro3/ckks-train/releases/tag/v1.0.0-preview.1)
were published on 2026-11-08 as `v1.0.0-preview.1`, source `6fba46c`.
The sole author is Kentaro Mihara (Independent Researcher). This is the FULL-1
interim paper; FULL-2 and later results remain pending. No arXiv submission or
arXiv server build is claimed. The build notes below include earlier dated artifacts.

Task56 owns `main.tex`, `sections/*.tex`, `refs.bib`, `figs/*`, and `tools/figs.py` plus its focused test. Task54 owns `tables/*` and `tools/paper_tables.py`; the paper consumes those outputs unchanged. This is a full English draft, not a submission-ready claim of research success. FULL-1 is completed partial full-setting evidence at B=64 with one initial-weight seed. FULL-2 remains the registered accuracy gate; full-setting ReBoot CPU timing, GPU training ablations, GPU bootstrap measurements and peak GPU memory remain unmeasured. The FULL-1 source/table update was rebuilt on 2026-11-07: anonymous and author-placeholder variants each have 10 pages. The reviewed anonymous output is `tools/latex/_ext/task56/full1-20261107/anonymous/main.pdf`; the placeholder format check is `tools/latex/_ext/task56/full1-20261107/authors/author-wrapper.pdf`. Anonymous pages 1–10 and the placeholder first page were visually inspected; both final logs have no overflow, undefined-reference/citation or missing-character errors. These checks do not close the remaining scientific gates.

## Numerical sources and figures

Run from the repository root:

```sh
python3 -m unittest discover -s tools -p test_figs.py
python3 tools/paper_tables.py --check
python3 tools/paper_tables.py --sources
python3 tools/figs.py
python3 tools/figs.py --check
```

`tools/paper_tables.py` generates the seven tables under `paper/tables/`. Missing registered runs remain pending; invalid existing inputs are errors. Coordinate regeneration with its owner. `tools/figs.py` uses only the Python standard library and imports the table generator's existing validated source API. It does not run training or edit raw results. It selects the two faithful measured subset rows from the RQ1 table cells and creates:

- `figs/rq1_ratios.tex`: native vector TeX picture, directly input by the paper.
- `figs/rq1_ratios.svg`: standalone vector plot with title/description.
- `figs/rq1_ratios.sources.json`: source paths, fields, SHA-256, scope, and displayed precision.

The plot uses the same rounded ratio cells as the table. Each row is one seed; its four accounting conventions are derived comparisons, not four new runs. The full-setting rows and GPU results are never substituted with the subset. Literal performance results are included through generated tables/figures; design constants and symbolic equations are authored in the sections. A plot/table matching existing source does not independently validate the underlying experiment.

## Build anonymous PDF

The default `main.tex` uses unmodified `acmart` sigconf/review/anonymous formatting. It changes no font or margins. Author names and affiliations are omitted by default. A separate `AuthorDraft` toggle enables the confirmed sole author metadata; it must not be used for the anonymous artifact.

The project-local TeX installation is a development tool under ignored `tools/latex/_ext/`, not a Cargo dependency or a distributed library component. Use its absolute binary path with a process-local PATH. The following Python build snippet works from the repository root and does not change a shell profile:

```python
import os
import subprocess
from pathlib import Path

root = Path.cwd().resolve()
tex = root / "tools/latex/_ext/TinyTeX/bin/universal-darwin"
out = root / "tools/latex/_ext/task56/anonymous"
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["PATH"] = str(tex) + os.pathsep + env["PATH"]
subprocess.run([
    str(tex / "latexmk"), "-pdf", "-interaction=nonstopmode",
    "-halt-on-error", "-no-shell-escape", "-outdir=" + str(out), "main.tex"
], cwd=root / "paper", env=env, check=True)
```

Output: `tools/latex/_ext/task56/anonymous/main.pdf`. All compiler outputs remain outside the tracked paper source. Inspect `main.log` for missing references, characters, overflow, and warnings. With Poppler:

```sh
pdfinfo tools/latex/_ext/task56/anonymous/main.pdf
pdftoppm -png -scale-to 1400 tools/latex/_ext/task56/anonymous/main.pdf /tmp/ckks-train-paper
```

For the nonanonymous author variant, create an ignored wrapper containing `\def\AuthorDraft{1}\input{main.tex}` under `tools/latex/_ext/task56/authors/`. Build that wrapper with the same process-local toolchain, working directory `paper/`, and an output directory under `authors/`. The user confirmed Kentaro Mihara as sole author, with contact miha.ken.19@gmail.com, on 2026-11-08. This is a personal project unrelated to their employer; the manuscript uses Independent Researcher without a company affiliation or invented city/country. The selected paper license is the arXiv non-exclusive distribution license; the software remains MIT.

## Format constraint and verification

The primary [WAHC2026 CFP](https://homomorphicencryption.org/wahc-2026-14th-workshop-on-encrypted-computing-applied-homomorphic-cryptography/) was checked on 2026-11-06: English anonymous ACM sigconf, at most **12 pages including references and appendices**, with no font/margin changes. This is the provisional draft limit. WAHC2027 rules and deadlines are unconfirmed and must be checked before submission. Nothing has been submitted or uploaded.

The Task56 coordination report records the exact build, page count, log warnings, rendered-page review, tests, input hashes, and final source freeze. ACM publication reference metadata is intentionally suppressed for this venue-unassigned draft, which produces a nonfatal `printacmref` warning. Shell escape is deliberately disabled; epstopdf can warn even though no EPS conversion is used. Underfull-box warnings, especially long bibliography URLs, require visual review but do not themselves indicate missing content. The final build must have no overflowing boxes or undefined citations/references.

## Evidence and bibliography boundaries

Current limitations are in section7 and the appendix, including security-model assumptions and practical memory caveats, asymmetric KAT coverage, earlier performance misses, inconsistent prior timing totals, interactive/diagnostic boundaries, batch-accuracy tradeoffs, and the fixed-epoch Breast Cancer miss. The paper does not claim equal-security superiority, completion of the registered FULL-2 accuracy gate, GPU training speedup, or GPU bootstrap performance. FULL-1 accuracy is reported as a separate larger-batch observation.

References were verified against primary paper/publisher/author/project pages. The HEonGPU key-switch paper's metadata is verified; fetching its PDF returned HTTP403, so its numerical table attribution remains the inherited archived M4 evidence rather than a claimed fresh PDF check. Its bibliographic year is2025; no unverified manuscript-version date is added in authored text. HEonGPU's official bootstrap benchmark is a different remaining-level and main-secret workload. External ReBoot/OpenFHE implementation source was not copied into this artifact. The user approved one narrow Unicode-3.0 character-table exception for `unicode-ident=1.0.27`, with its complete Unicode notice, alongside the MIT/Apache/BSD/ISC policy. This is a package-and-version-specific exception, not general Unicode approval. The root release owner applies the cargo-deny rule and notices; no release policy is changed by this paper task.

For the first public artifact, follow `docs/publication-schedule.md`: freeze the available evidence, explicitly retain unmet/unmeasured comparisons, and review all generated artifacts. Before a conference submission, remove identifying artifact metadata as required and recheck the target CFP. Compilation and an intact pipeline do not close those research gates.

## Named draft, 2026-11-08

The confirmed single-author PDF is `tools/latex/_ext/task56/named-20261108/author-wrapper.pdf` (10 pages). It uses an Independent Researcher author note rather than an institutional affiliation: acmart requires a country for an affiliation, and no location was supplied. The class is unchanged. The build succeeded; the final log contains no overflowing boxes, undefined citations/references or missing characters. Page 1 was visually inspected at full size and pages 2–10 as a rendered contact sheet. Build logs and `verification.json` are alongside the PDF. The research evidence is unchanged from the interim FULL-1 draft; this is not a submitted or final paper.

## arXiv source preview, 2026-11-08

`ArxivDraft` enables the confirmed author and `sigconf,nonacm` formatting without review line numbers. Anonymous review and `AuthorDraft` modes remain available. The manuscript body, generated tables and figure are unchanged. This follows the [arXiv TeX guidance](https://info.arxiv.org/help/submit_tex.html) to prepare the necessary source and bibliography without unrelated files; arXiv must still compile and display its own output during submission.

Local package: `tools/release/_ext/arxiv-source-preview-20261108/ckks-train-arxiv-source-preview.zip` (19 members, 37,482 bytes). It contains `main.tex`, eight section files, seven tables, one TeX figure, `refs.bib` and `main.bbl`. Only `main.tex` is transformed by prepending `\def\ArxivDraft{1}`; `main.bbl` is the generated bibliography from the author preprint build. No raw results, logs, extra figure formats, external implementations or installed TeX style files are bundled. The sibling `manifest.json` records every member hash, the source-base commit and the exact main.tex worktree patch used for this preview.

The ZIP was read back, each member compared, and unpacked to `source/`. Building `main.tex` from that directory with the same project-local `latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape` succeeded. The isolated PDF is `build/main.pdf` (10 pages), and its extracted text matches the directly built preprint. The final log has no overflow, missing-character or undefined-reference/citation warning. All ten rendered pages were visually reviewed, with the first page also inspected separately. This verifies the local TeX Live 2026/acmart 2.21 toolchain; the arXiv server toolchain has not yet been tested. The package is a preliminary artifact and must be regenerated after the evidence cutoff, not submitted as a claim of final measurements.

The manuscript is covered by the separate [paper licensing notice](LICENSE). The software MIT license and data notices remain separate. The selected arXiv license will be recorded in submission metadata; this file does not assert that a submission exists.
