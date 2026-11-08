# arXiv submission metadata draft

Prepared 2026-11-08 from the current interim manuscript. **Not submitted.**
Account checking is deferred at the user's request; login and category endorsement
have not been verified. Refresh this file after the evidence cutoff and compare
with the final source and arXiv-generated PDF before submission.

| Field | Prepared value |
|---|---|
| Title | Revisiting Diagonal Packing for Non-Interactive Encrypted Neural Network Training |
| Author | Kentaro Mihara |
| Author order | Sole author |
| Affiliation | Independent Researcher; no company affiliation |
| Contact | miha.ken.19@gmail.com |
| License | arXiv non-exclusive distribution license, as selected by the author |
| Top-level source / processor | `main.tex` / `pdflatex` |
| Proposed primary category | cs.CR (Cryptography and Security) |
| Optional cross-list proposal | cs.LG (Machine Learning) |
| Comments draft | 10 pages, research preprint. Code and evidence: https://github.com/kenmaro3/ckks-train |
| Journal reference / DOI / report number | Leave empty: none assigned |

The category choices are editorial proposals based on the manuscript's focus on
homomorphic-encryption algorithms and encrypted learning, using the
[official arXiv taxonomy](https://arxiv.org/category_taxonomy), checked 2026-11-08.
They are not an endorsement or moderation decision. Confirm the actual page count
and public repository URL before using the Comments field. WAHC2027 is a planned
venue, not an accepted or published journal reference.

## Abstract

Training a neural network under homomorphic encryption requires more than an efficient forward matrix product: transposes, batch reductions, optimizer state, and repeated refreshes can dominate the cost. We revisit diagonal packing for this setting. Starting from the earlier rectangular transpose, we combine a periodic representation using the smaller padded dimension's number of diagonals with feature-major SIMD mini-batch lanes and baby-step giant-step evaluation. Compact optimizer state amortizes bootstrapping across diagonals. We implement the resulting training path in a Rust CKKS stack with instrumented CPU and CUDA arithmetic. We compare key-switch accounting conventions, bootstrap placement, and measured operator latencies, and explicitly separate layout effects from bootstrap implementations. Completed encrypted subset experiments and the first single-seed full-setting run support a conditional key-switch advantage for a larger batch. The full-setting larger-batch result does not replace the registered baseline-batch accuracy experiment, which remains pending. Full-setting baseline CPU timing and an equal-security comparison are unresolved, as are GPU training ablations and physical GPU bootstrap measurements. We report these gaps together with negative results rather than interpreting implementation completeness as research success. This interim draft establishes the algorithms and reproducible evidence needed to finalize that comparison.

## Current preview artifacts and refresh boundary

The title and abstract above were extracted verbatim from `main.tex`
(SHA256 `ccaf2e223bb55ecae6e9d011cf91d1d5a39b4c5f67d44ca07f56d63b1a5ed68e`). They include
pending FULL-2 and unresolved comparisons deliberately; replace them only when
supported by the final retained evidence, then rebuild the PDF/source package.

The current locally tested upload preview is
`tools/release/_ext/arxiv-source-cutoff-readiness-20261108/ckks-train-arxiv-source.zip`.
It has 19 members, 38,430 bytes and SHA256
`7362cbf975bb0c0052831e2a2778c9bb7f1cebe3417916bb54baeef4e420d89a`.
The matching PDF in the isolated build is `build/main.pdf` in that directory.
These are preliminary artifacts, not a substitute for arXiv's own source-build
and preview check. This metadata file is submission assistance; it need not be
included in the TeX source ZIP. No account action, upload, license acceptance,
endorsement request or submission was performed while preparing it.

## Upload preparation check, 2026-11-08

The published 19-member preview ZIP was inspected without changing it: all
TeX input and bibliography paths resolve with exact case, names use supported
characters, `main.bbl` is present, and the entrypoint enables the named
`sigconf,nonacm` preprint. Its abstract still exactly matches the metadata above.
The package contains no compiler logs or output PDF. This is a static package
check, not another compilation or an arXiv server-build result.

The [official TeX Live guide](https://info.arxiv.org/help/faq/texlive.html),
checked 2026-11-08, lists TeX Live2025 as default and2023 as an alternative.
The local preview was built with2026, so select `main.tex` and `pdflatex`
and inspect arXiv's actual log/PDF before submission. Do not assume identical
page breaks or class versions. The existing `main.bbl` uses ordinary BibTeX,
not biblatex/Biber. Source and bibliography preparation follows the
[official TeX instructions](https://info.arxiv.org/help/submit_tex.html).

After the evidence cutoff, regenerate the upload ZIP, update this abstract and
page count to match that final paper, and repeat the package and PDF checks.
The public preview remains fixed; an updated final submission is a separate
artifact. Account/endorsement checks remain deferred at the author's request.

## Local manuscript update, 2026-11-08 12:31 JST

The current local package above incorporates the clarified encrypted/plaintext
weight-decay scope, explicit FULL-1 final-parameter diagnostic, compact released
protocol and current named-artifact instructions. All 19 members were read back
and matched; the isolated source build succeeded and its extracted PDF text
matches the direct 10-page build. This remains an interim package: FULL-2 and
queued hardware evidence are pending. The public preview ZIP remains unchanged;
no arXiv upload or server build occurred. Refresh again after the evidence cutoff.
