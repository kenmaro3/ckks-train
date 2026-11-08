# FULL-2 / R-47-3 paper update — 2026-11-08

This archive verifies a local named preprint after adoption of the completed FULL-2
and R-47-3 evidence. It is not a new scientific run, a final evidence cutoff,
a public-release update or an arXiv submission. Base Git commit: `b7c51dd3f03f416637e1379c0fc35bbd806c3b99`;
changed manuscript input hashes are recorded in `direct-build-meta.json`.

## Content and evidence

The paper reports FULL-2's baseline-batch accuracy pass, its convention(b) count
miss, the unmet common-security condition and the unmeasured full-setting ReBoot
CPU denominator. FULL-1's larger-batch count passes are kept separate. R-47-3 is
an independent-product, capped-level microbenchmark; its within-run repetitions
are not independent seeds or GPU-training measurements. R48/R52 remain pending.
The underlying results were already reviewed and adopted in
[Task42](../task42-full2-finalization-20261108/README.md) and
[R47](../task47-r47-3-adoption-20261108/README.md).

`initial-content-review.md` is the independent content review before the final
repository-notice shortening and build/metadata documentation. The follow-up
review in `final-content-and-artifact-review.md` independently verifies the later
changes and saved artifact consistency (SPEC/QUALITY PASS). Root performed the
visual inspection; the reviewer checked saved render identity. No scientific JSON
was edited. `freshness-checks.json` records the later root table/figure check and
metadata hash; it is additional to the reviewer's original 15-file archive.

## Local artifact checks

- Direct PDF: 10 pages, 532,092 bytes, SHA256
  `9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c`.
- Source ZIP: 19 members, 40,079 bytes, SHA256
  `2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2`.
- All ZIP members were read back and matched to selected sources. Only `main.tex`
  gains the ArxivDraft definition; the direct build's generated bibliography is
  included as `main.bbl`. Eight sections, seven tables, one figure and `refs.bib`
  are included. No raw results, logs, output PDF or installed TeX styles are bundled.
- Building the unpacked ZIP alone succeeded. Extracted text and all ten 1600-pixel
  page renders are identical to the direct PDF. PDF bytes differ because they are
  separate compiler outputs; their hashes are not claimed identical.
- Root inspected all initial pages. The initial eleventh page held only two lines
  of the final repository notice. Shortening that notice yielded ten pages without
  changing fonts, margins or scientific content. Final pages 1–9 have byte-identical
  PNG renders to reviewed initial pages; final page 10 was separately inspected.
- Both final logs have no overfull boxes, missing characters or undefined
  citations/references. Disabled shell escape and the float-page warning were
  inspected. Local toolchain: TeX Live2026 / acmart2.21. arXiv server is untested.
- The submission metadata abstract matches `paper/main.tex` verbatim. The page
  count and confirmed sole author remain consistent. No account operation occurred.

`verification.json`, the two build metadata records, logs, PDF information and
`package-manifest.json` retain exact commands, input hashes and output checks.
`initial-build-meta.json` preserves the earlier eleven-page build's input state.
Large PDF/PNG/ZIP files stay on the user-selected MicroSD; full paths are in the
records and [paper README](../../../paper/README.md). No internal-storage fallback
was used. The fixed public preview at `6fba46c` remains unchanged.

This content/build verification does not validate unfinished Task54 source,
complete the full experiment matrix, close M4/M5 research gates, establish GPU
training speedup, or replace arXiv's own compilation and submission checks.
