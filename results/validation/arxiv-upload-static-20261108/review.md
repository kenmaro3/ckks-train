# arXiv upload preparation — bounded independent review

SPEC: PASS. QUALITY: PASS. No findings within the requested scope.

Reviewed the full `arxiv-upload-readiness-review.diff`, the changed portions of `paper/arxiv-metadata.md` and `paper/README.md` with surrounding context, and both files in `results/validation/arxiv-upload-static-20261108/`. Also read the existing ZIP, manuscript entrypoint and retained local build-log toolchain header. No source WIP, network, account actions, upload, compilation or test reruns were performed; only this report was written.

## Verified evidence and boundaries

- Recomputed the ZIP's 37,482-byte size and SHA256 `c9a99d53f6560fefb4b451d4011c68f22c3dc467b4e437813f4e5dd1331039f8`. Its 19 unique members exactly match `inspection.json`: main.tex, eight sections, seven tables, one TeX figure, refs.bib and main.bbl. There are no PDF/compiler-log/aux outputs or bundled class files. Member names contain only the stated simple filename characters, with no parent traversal.
- Independently resolved all 17 active explicit input/bibliography references with exact case. The retained reference inventory matches; the two commented wrapper examples in main.tex are not active includes. This does not inspect all implicit dependencies of installed TeX classes/packages.
- The packaged main.tex is exactly the current paper main.tex prefixed with `\def\ArxivDraft{1}` and a newline. The original main.tex hash matches metadata: `ccaf2e223bb55ecae6e9d011cf91d1d5a39b4c5f67d44ca07f56d63b1a5ed68e`. Its ArxivDraft branch enables named `sigconf,nonacm`; the confirmed author/email match. The abstract is verbatim identical to the metadata paragraph. The supplied main.bbl declares BibTeX generation and contains an ordinary thebibliography environment.
- `paper/arxiv-metadata.md:16,50–70` correctly identifies main.tex/pdflatex and calls this a static check. The retained isolated `build/main.log` identifies pdfTeX/TeX Live 2026 and acmart 2.21; this was read, not rebuilt. The documented server default 2025 and alternative 2023 are attributed to the root's dated check of the [official TeX Live guide](https://info.arxiv.org/help/faq/texlive.html), supplied in the review brief. The [official TeX instructions](https://info.arxiv.org/help/submit_tex.html) are linked for source/bibliography and output inspection. These external pages were not independently fetched in this no-network review; neither server compilation nor a particular installed server acmart version is asserted.
- `paper/README.md:66` now correctly distinguishes the already public GitHub preview from an unsubmitted arXiv/conference paper. Its current prefix retains the exact public preview source/tag. Metadata preserves provisional comments/page count, final evidence refresh and regeneration, and deferred account/endorsement checks. The public preview remains fixed, with any later upload candidate a separate artifact; implementation and scientific gates are unchanged.
- `inspection.json:126` and the evidence README disclose the initial expected-SHA transcription error and the failed pre-inspection check. The successful static record uses the established correct archive hash. No false compiler/server success is substituted for that failure.

## Reviewed input pins

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| Ledger review diff | 4225 | `23736a5e480e7073acee357ce1ac7dcf50e754688fca94c5ea19013fd6ae8b5f` |
| paper/arxiv-metadata.md | 5328 | `199e6a5fcefee6e3534d58f8d1e18fad73d6055abe5008572f16ab3f31acd5d9` |
| paper/README.md | 11098 | `a65325bc8fbe01f8ee8be1688b7c878a320bd8615a934833ca35351d2e4c9bde` |
| Evidence README.md | 1465 | `7bb3988c532ca81399b000a407da38ebb8e0c4a307060e79fb822b6af485ddf0` |
| Evidence inspection.json | 3469 | `1b0680211c4a9d8ffaf340ccb722aa2f5a006744bb2de7bf18f9c17645ca728a` |
