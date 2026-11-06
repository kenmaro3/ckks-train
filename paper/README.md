# ckks-train interim paper draft

Task56 owns `main.tex`, `sections/*.tex`, `refs.bib`, `figs/*`, and `tools/figs.py` plus its focused test. Task54 owns `tables/*` and `tools/paper_tables.py`; the paper consumes those outputs unchanged. This is a full English draft, not a submission-ready claim of research success. Registered full-setting encrypted runs, GPU training ablations, GPU bootstrap measurements and peak GPU memory remain pending.

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

The default `main.tex` uses unmodified `acmart` sigconf/review/anonymous formatting. It changes no font or margins. Author names and affiliations are omitted by default. A separate `AuthorDraft` toggle enables placeholder metadata; it must not be used for the anonymous artifact.

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

For the nonanonymous placeholder variant, create an ignored wrapper containing `\def\AuthorDraft{1}\input{main.tex}` under `tools/latex/_ext/task56/authors/`. Build that wrapper with the same process-local toolchain, working directory `paper/`, and an output directory under `authors/`. This exercises the toggle; it supplies no real author identity.

## Format constraint and verification

The primary [WAHC2026 CFP](https://homomorphicencryption.org/wahc-2026-14th-workshop-on-encrypted-computing-applied-homomorphic-cryptography/) was checked on 2026-11-06: English anonymous ACM sigconf, at most **12 pages including references and appendices**, with no font/margin changes. This is the provisional draft limit. WAHC2027 rules and deadlines are unconfirmed and must be checked before submission. Nothing has been submitted or uploaded.

The Task56 coordination report records the exact build, page count, log warnings, rendered-page review, tests, input hashes, and final source freeze. ACM publication reference metadata is intentionally suppressed for this venue-unassigned draft, which produces a nonfatal `printacmref` warning. Shell escape is deliberately disabled; epstopdf can warn even though no EPS conversion is used. Underfull-box warnings, especially long bibliography URLs, require visual review but do not themselves indicate missing content. The final build must have no overflowing boxes or undefined citations/references.

## Evidence and bibliography boundaries

Current limitations are in section7 and the appendix, including security-model assumptions and practical memory caveats, asymmetric KAT coverage, earlier performance misses, inconsistent prior timing totals, interactive/diagnostic boundaries, batch-accuracy tradeoffs, and the fixed-epoch Breast Cancer miss. The paper does not claim equal-security superiority, completed full-setting accuracy, GPU training speedup, or GPU bootstrap performance.

References were verified against primary paper/publisher/author/project pages. The HEonGPU key-switch paper's metadata is verified; fetching its PDF returned HTTP403, so its numerical table attribution remains the inherited archived M4 evidence rather than a claimed fresh PDF check. Its bibliographic year is2025; no unverified manuscript-version date is added in authored text. HEonGPU's official bootstrap benchmark is a different remaining-level and main-secret workload. External ReBoot/OpenFHE implementation source was not copied into this artifact. The user approved one narrow Unicode-3.0 character-table exception for `unicode-ident=1.0.27`, with its complete Unicode notice, alongside the MIT/Apache/BSD/ISC policy. This is a package-and-version-specific exception, not general Unicode approval. The root release owner applies the cargo-deny rule and notices; no release policy is changed by this paper task.

Before release: complete the registered measurements and same-condition comparisons, review all generated artifacts, replace author placeholders in an authorized nonanonymous version, remove identifying artifact metadata as required, and recheck the target CFP. Compilation and an intact pipeline do not close those research gates.
