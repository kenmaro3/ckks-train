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
`tools/release/_ext/arxiv-source-preview-20261108/ckks-train-arxiv-source-preview.zip`.
It has 19 members and SHA256
`c9a99d53f6560fefb4b451d4011c68f22c3dc467b4e437813f4e5dd1331039f8`.
The matching PDF in the isolated build is `build/main.pdf` in that directory.
These are preliminary artifacts, not a substitute for arXiv's own source-build
and preview check. This metadata file is submission assistance; it need not be
included in the TeX source ZIP. No account action, upload, license acceptance,
endorsement request or submission was performed while preparing it.
