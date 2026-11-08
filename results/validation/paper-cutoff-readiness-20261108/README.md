# Interim paper content and local submission-package verification

Baseline source: `47d0035dd167d63b90f7552f8a394510f3d2deb5` plus the exact
eleven-file correction in [review.diff](review.diff) and [owned-pins.json](owned-pins.json).
The initial content review found four bounded clarity/reproducibility issues;
the [correction report](report.md) records their remedy and verification limits.
Raw measurements, thresholds, and the immutable public preview are unchanged.

The new generated note reports FULL-1's final weight/bias discrepancy against its
same-batch plaintext twin. It carries field-level provenance and is not a new
bootstrap-precision or accuracy gate. The new synthetic test first failed for
the absent note, then all14 table tests passed. Archived-table regeneration and
table/figure freshness checks passed. The preserved whole-diff whitespace failure
is a TSV empty final field; the other ten paths pass their scoped check.

The named PDF has10pages,528922bytes and SHA256
`5b2f73b9bff3645574dc1084cb9d38e325ddbcc4da18004387d95203ddd09900`.
Root inspected rendered pages3–10; pages1–2 have unchanged extracted text from
the preceding reviewed named artifact. Build logs have no overflow, undefined
reference/citation or missing-character errors. No new anonymous build is claimed.

The local arXiv ZIP has19members,38430bytes and SHA256
`7362cbf975bb0c0052831e2a2778c9bb7f1cebe3417916bb54baeef4e420d89a`.
Every member was read back and matched. A build using only the extracted source
succeeded and its extracted PDF text equals the direct build. The manifest and
command/output records identify both local builds. Local TeXLive2026/acmart2.21
is not arXiv's server toolchain. No upload, account check or submission occurred.

The base-CI observation records all6jobs successful for the baseline47d0035,
not this correction or the separate uncommitted Task54 Rust work. The complete
research matrix, FULL-2 and queued hardware measurements remain outstanding.
Refresh the final manuscript and upload package after the evidence cutoff.

The [scoped independent rereview](fix1-review.md) passes SPEC and QUALITY for
these four interim corrections. It independently checked all eleven file pins,
archive members and PDF hashes/text, and inspected rendered pages4/7/10. This
does not close the final-cutoff identity or any scientific/submission requirement.
