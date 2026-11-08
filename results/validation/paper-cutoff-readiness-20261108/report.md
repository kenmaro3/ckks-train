# Interim paper corrections after content review

Root read the complete paper-cutoff-readiness-review-20261108.md. All four
findings are addressed without changing measurements or numerical gates:
F1 scopes weight decay to encrypted training versus the plaintext reconstruction;
F2 adds a generated per-completed-FULL final-parameter error note and field provenance;
F3 replaces stale placeholder/submission guidance; F4 describes the actual released
protocol and adds a named-preprint-only repository link. The unfinished whole-matrix
runner is distinguished from archived-table reproduction. FULL-2 remains pending.

Eleven exact root-owned files are frozen in owned-pins.json and review.diff.
The same-batch maximum final weight/bias error is read from twin.max_abs_diff,
not authored into TeX or treated as single-bootstrap precision. The historical
producer ff2c2eb's mnist implementation computes this value from the final decrypted
weights/biases versus the plaintext twin. A .125 synthetic fixture first failed
because the note was absent; all14 table tests then passed. run_all --tables-only,
table check and figure check exited0. Three generated files changed: accuracy
Markdown, accuracy TeX and its sources.tsv provenance row. No raw result changed.

The full scoped git diff --check returned2 solely because sources.tsv retains its
required empty final formula field (trailing tab). Original stdout/stderr preserved;
do not strip a TSV field to satisfy whitespace checking. The ten other authored/
generated paths pass the scoped check. No failed command was silently retried.

Direct named build exited0; 10pages,528922bytes, SHA256
5b2f73b9bff3645574dc1084cb9d38e325ddbcc4da18004387d95203ddd09900.
PDF: tools/latex/_ext/task56/cutoff-readiness-20261108/arxiv-wrapper.pdf.
No overflow/undefined citations/references/missing characters in final log.
Root visually inspected rendered pages3–10. Extracted text on pages1–2 is unchanged
from the preceding reviewed named artifact; no new anonymous-mode build is claimed.

New local ZIP: tools/release/_ext/arxiv-source-cutoff-readiness-20261108/ckks-train-arxiv-source.zip,
19members,38430bytes,SHA256
7362cbf975bb0c0052831e2a2778c9bb7f1cebe3417916bb54baeef4e420d89a.
All members read back and matched to selected source/generated bibliography;
isolated unpacked-source build exited0 and extracted text equals direct build.
Both use existing local TeXLive2026/acmart2.21, not arXiv's server. Package manifest,
member/source pins, command/output/result records are retained here and alongside
the ignored package. Public preview6fba46c/tag/assets remain unchanged. No upload,
account check, submission, new experiment or Rust-source adoption occurred.

The latest metadata/README point to this local interim update. The abstract/main
source is unchanged. At evidence cutoff, refresh the actual results and final
source/PDF/package identities. Original research work and Task54 remain active.
