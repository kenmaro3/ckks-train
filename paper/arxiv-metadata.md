# arXiv submission metadata draft

Prepared 2026-11-08 for the evidence-frozen `v1.0.0-preview.2` manuscript. **Not submitted.**
The author will perform arXiv submission, including account/endorsement checks and
the final arXiv-generated PDF review. The agent has not logged in or submitted it.
Use this public [source ZIP](https://github.com/kenmaro3/ckks-train/releases/download/v1.0.0-preview.2/ckks-train-arxiv-source.zip)
and compare the server-built output with the [reviewed PDF](https://github.com/kenmaro3/ckks-train/releases/download/v1.0.0-preview.2/ckks-train-paper-preview.pdf).
Its checksum is recorded in the release's `SHA256SUMS`; no further experiment is
required for this version's evidence selection. Future evidence updates require
a separately reviewed package and matching metadata.

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

Training a neural network under homomorphic encryption requires more than an efficient forward matrix product: transposes, batch reductions, optimizer state, and repeated refreshes can dominate the cost. We revisit diagonal packing for this setting. Starting from the earlier rectangular transpose, we combine a periodic representation using the smaller padded dimension's number of diagonals with feature-major SIMD mini-batch lanes and baby-step giant-step evaluation. Compact optimizer state amortizes bootstrapping across diagonals. We implement the resulting training path in a Rust CKKS stack with instrumented CPU and CUDA arithmetic. We compare key-switch accounting conventions, bootstrap placement, and measured operator latencies, and explicitly separate layout effects from bootstrap implementations. Two completed full-setting encrypted runs, each with one initial-weight seed, support a larger-batch key-switch advantage and baseline-batch accuracy within the registered tolerance. At the baseline batch, however, normalizing both bootstrap costs to the baseline model misses the key-switch target. The released baseline also falls below the common security-estimator threshold, and its full-setting CPU time is unmeasured. Revised multi-stream execution improves an independent-product microbenchmark; GPU training ablations and physical GPU bootstrap measurements remain unresolved. We present a conditional layout comparison and its limitations, without combining successful criteria from different batches into a single all-conditions result.

## Current preview artifacts and refresh boundary

The title and abstract above match the current `main.tex`
(SHA256 `924d994b1f863a8fb2316e22b0eb263a111695800a66088667e043c72dd3e367`). They reflect the completed FULL-2
accuracy gate, its missed normalized-count target, and the R-47-3 independent-product
microbenchmark. GPU training and deeper encrypted-model results remain pending.

The current locally tested upload package is
`/Volumes/Untitled/ckks-train-storage/archives/arxiv-source-full2-r47-20261108/ckks-train-arxiv-source.zip`.
Its manifest and isolated-build result are archived with the
[paper verification](../results/validation/paper-full2-r47-20261108/README.md).
The exact same ZIP bytes are now published in preview 2; its evidence selection
is frozen for this version. This is not a substitute for arXiv's own source-build
and preview check. This metadata file is submission
assistance; it need not be included in the TeX source ZIP. No account action,
upload, license acceptance, endorsement request or submission was performed while
preparing it.

## Earlier upload preparation check, 2026-11-08

The published 19-member preview ZIP was inspected without changing it: all
TeX input and bibliography paths resolve with exact case, names use supported
characters, `main.bbl` is present, and the entrypoint enables the named
`sigconf,nonacm` preprint. Its abstract matched the earlier metadata; the current abstract above includes the later FULL-2/R-47-3 update.
The package contains no compiler logs or output PDF. This is a static package
check, not another compilation or an arXiv server-build result.

The [official TeX Live guide](https://info.arxiv.org/help/faq/texlive.html),
checked 2026-11-08, lists TeX Live2025 as default and2023 as an alternative.
The local preview was built with2026, so select `main.tex` and `pdflatex`
and inspect arXiv's actual log/PDF before submission. Do not assume identical
page breaks or class versions. The existing `main.bbl` uses ordinary BibTeX,
not biblatex/Biber. Source and bibliography preparation follows the
[official TeX instructions](https://info.arxiv.org/help/submit_tex.html).

For a future evidence update, regenerate the upload ZIP, update the abstract and
page count, and repeat the package and PDF checks. Preview 2 is fixed and ready
for the author's submission workflow; account/endorsement checks and the server
preview remain the author's responsibility.

## Local manuscript update, 2026-11-08 12:31 JST

The earlier 12:31 local package incorporated the clarified encrypted/plaintext
weight-decay scope, explicit FULL-1 final-parameter diagnostic, compact released
protocol and current named-artifact instructions. All 19 members were read back
and matched; the isolated source build succeeded and its extracted PDF text
matched the direct 10-page build. At that snapshot, FULL-2 and queued hardware
evidence were pending. The current package above supersedes that local snapshot;
the public preview ZIP remains unchanged. No arXiv upload or server build occurred.
Refresh again after the evidence cutoff.
