# Independent HEonGPU primary-source citation review

Date: 2026-11-08. Scope: the eight-file citation correction frozen in `heongpu-primary-20261108/review.diff`, following `heongpu-primary-review-brief.md` and `change-rationale.md`. This is a literature/manuscript review, not a new HE benchmark or runtime-source review.

SPEC: PASS. QUALITY: PASS. No blocking or nonblocking findings in this bounded update.

## Primary-source verification

I independently checked the held author-deposited PDF against `download.json`: 530327 bytes, SHA256 `9223e5689b8326928f06fe8d83ed1dc74dba21f9658c9cee77f7cbbfbc19b7a9`. The recorded primary URL is `https://research.sabanciuniv.edu/id/eprint/53894/1/GPU%20Implementations.pdf`. Network access was not used during this review; HTTP 200 is the retained download record, not a new retrieval claim.

Direct read-only text extraction from PDF page 1 confirms November 2024 and the paper title/authors. Pages 9-11 identify the RTX 4090, 24 GB and CUDA 12.4, define I as a number of primes in a group, and contain the cited table and qualification. I also visually inspected the retained full-page `table3.png`, including the paragraph below the table.

Table 3, page 11, supports CKKS GPU relinearization, log N = 16, 43-bit precision, level 35, Method II, (I, l-tilde, l, r-prime) = (4, 40, 36, -), time 2581.8 microseconds. The adjacent Method I time is 6707.4 microseconds; the row is not confused with the 36-bit, level-41 Method II value 3458.4. The change preserves the cited 2581.8 value and bibliographic/ePrint year 2025 while recording the verified November 2024 manuscript date.

The following security paragraph explains that Q is kept constant for multiplication capacity while the exposed extended modulus varies. It describes Method II/III security as lower than the 128-bit Method I comparison and gives an approximate 110-bit minimum across all experimental configurations. That is an author-stated across-experiments lower bound, not a fresh security estimate and not an assertion that the exact 2581.8-microsecond row has exactly 110-bit security. The new Japanese report wording and English metadata/table notes preserve this distinction. The manuscript says the reference is not a matched-security speed comparison.

## Scope and consistency checks

Both eight-entry before/after manifests match the bytes and SHA256 of the saved before files and the current scoped files. I reconstructed each current file from the frozen patch and its saved before version in memory; all eight match exactly. Thus the reviewed patch is the supplied change, rather than unrelated concurrent work.

The only scoped JSON is literature metadata. Removing its changed `source` and `caveat` fields makes the before/after documents identical, including value_us = 2581.8 and the parameter string. No measured-result JSON is included in this patch. No runtime, test, dependency or generator source is included. This establishes the bounded patch's contents; it is not a workspace-wide raw-data audit.

The existing Q-prime-count difference (36 versus 32), four special primes, alpha = 4 versus dnum = 8 distinction, nominal timing ratio, and linear-in-l cost-model limitation remain. The separate ePrint 2024/1543 library paper remains explicitly unread/unverified after HTTP 403; this review does not close that gap. Existing unmeasured rows, later-fix retiming status, accuracy/security gates and GPU-bootstrap limitations remain in the reviewed text.

I compared the retained `generated-identity.json` and independently checked all 18 current listed artifact hashes. Only `paper/tables/gpu_vs_cpu_ops.tex` and `results/tables/gpu_vs_cpu_ops.md` differ between the recorded generated before/after sets, and both changes are citation-note prose. Table row bodies and numerical values are unchanged.

## Retained validation and visual quality

I inspected retained check-0/1/2 JSON, stdout and stderr: tables-only regeneration, table check and figure check each record exit 0. These are retained author-run checks, not reviewer reruns. I ran no tests, builds or generators.

The retained local latexmk build records exit 0 and a 10-page PDF. I independently verified its 527045-byte size and SHA256 `fe79d5ae1f552937e32fa2d5881368d59ca5be081aa9c4ff350ac4a18fb01c22`. The final TeX log contains no overfull boxes, undefined-reference/citation or missing-character diagnostics. Underfull-box messages and the disabled-shell-escape epstopdf warning are present; the initial missing-bbl message is followed by bibliography generation and final success.

I visually inspected retained page renders 6-10, including section 6.5, Table 7's full note and the bibliography entry. The added caveat and date are present, legible and do not introduce clipping or overlap. I did not render the PDF again or independently visually inspect pages 1-5; their unchanged extracted text is an author-report boundary. The already-noted appendix author-placeholder wording is outside this patch and is not a new review finding.

The new PDF is local subsequent drafting work. Published preview/tag/source ZIP identity at 6fba46c and pending arXiv metadata/ZIP refresh are retained report boundaries, not independently rechecked remote publication state. No upload, contact, commit or push was performed. Only this review report was written. Full-file byte count and SHA256 are provided in the reviewer completion message after the final write.
