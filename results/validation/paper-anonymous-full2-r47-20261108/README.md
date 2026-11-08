# Anonymous paper format validation — 2026-11-08

Built the default `sigconf,review,anonymous` manuscript at committed source
`6910b149172b67b07e69b92132cb05152df8ba6e`, with the same18 TeX/BibTeX input hashes
as the [reviewed named preprint](../paper-full2-r47-20261108/README.md).
No scientific source, font or margin changed. The command, process-local toolchain,
start/end, input hashes and exit0 are recorded in `build-meta.json` and build logs.

The result is10 pages,534,419 bytes; `verification.json` records PDF and all ten
render hashes. Root inspected every page individually. Equations, tables, figure,
references and appendix display without clipping, overlap or an orphan last page.
The first-page author block is Anonymous Author(s); extracted text has neither the
author's email nor the repository-account URL. Bibliographic self-citation remains.
This is a bounded metadata/display check, not a complete deanonymization audit.

The final log has no overfull boxes, missing characters or undefined citations/
references. Disabled shell escape, a float-only text-page warning and the suppressed
ACM reference block warning are acknowledged for this unsubmitted draft. The local
TeX Live2026/acmart2.21 build does not verify WAHC2027's rules or a submission system.
Ten pages fit the existing provisional12-page formatting target; no venue acceptance
or conference submission is claimed. No new numerical experiment was run.

The PDF and page PNGs are on the verified MicroSD at the path in `verification.json`;
only these small logs and verification records are retained in Git. The earlier
public preview and named arXiv-source package are unchanged.

Independent [record-consistency review](independent-review.md) passed SPEC/QUALITY.
It verifies current input hashes and saved artifacts/logs; root's visual inspection
was not independently repeated. The final manifest adds this review to the original
seven reviewed archive members without changing their underlying build records.
