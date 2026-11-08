# Paper cutoff readiness FIX1 scoped rereview — 2026-11-08

## Verdict

**SPEC: PASS. QUALITY: PASS for the frozen F1–F4 interim-paper corrections.** No blocking regression was found in the exact eleven-file diff. This is approval of the stated corrections and retained-evidence presentation, not final scientific success, a whole-matrix execution result, an arXiv server-build/submission verdict, or permission for public upload. FULL-2, equal-security comparison, full-setting ReBoot CPU timing, queued GPU training/ablation, physical GPU bootstrap, and whole-matrix execution remain unresolved.

Reviewed baseline `47d0035dd167d63b90f7552f8a394510f3d2deb5` to the frozen root-owned inputs below. Read the root report, owned pins, complete 27,086-byte diff, focused regression source, production extraction path, retained test/build/check logs, package manifest, current metadata, generated notes and TSV source row. Independently checked all eleven file byte counts/SHA-256 values, the diff hash, direct/isolated PDF hashes, nineteen archive member hashes, and existing direct/isolated extracted-text equality. Independently viewed existing rendered pages 4, 7 and 10. No tests, generation, build, rendering, SSH, native/HE execution, source mutation or subagent work was performed; this report is the only file written by this reviewer.

## Findings closed

- **F1 closed.** `paper/sections/noninteractive.tex:28` now scopes omitted weight decay to the evaluated encrypted trainer and distinguishes the separate plaintext reconstruction. `paper/sections/discussion.tex:19` preserves the same distinction. This agrees with FULL-1's zero decay and the plaintext paper-setting source's coupled decay. No result or baseline protocol changed.
- **F2 closed.** `tools/paper_tables.py` reads each completed FULL row's `twin.max_abs_diff` via `s.num` and records that exact field as the note's source. It runs only after an existing validated FULL result has been accepted; missing FULL-2 continues through the pending branch without inheriting FULL-1's error. The generated TeX/Markdown disclose **0.00682657**, correctly rounded from **0.006826572783266538**. The diagnostic is explicitly the final maximum absolute weight-or-bias difference versus the same-batch twin, and is explicitly neither per-bootstrap precision nor an accuracy gate. The prose references the generated table and does not paste a performance measurement into authored TeX. The existing FULL-1 B64 nongate verdict is unchanged; no unrelated NI1/256 threshold is retroactively applied.
- **F3 closed.** `paper/sections/appendix.tex` replaces author-placeholder guidance, confirms current preprint authorship, distinguishes anonymous/named-review/named-preprint modes, and separates arXiv source/metadata/server-PDF review from a future conference anonymity/CFP review. It no longer implies that all original research gates must pass before a candid preprint can be submitted. The narrower statement about author metadata/artifact links is consistent with the conditional new link; it does not certify global artifact anonymity.
- **F4 closed for this interim update.** Evaluation now defines the actual complete released setting: 4,800 examples over ten passes, 196→32→16, z²+z, frozen zero biases, sum-loss NAG with momentum 0.9 and released ordering/schedule. It distinguishes this from the 60,000-example paper-setting reconstruction and retains B64 FULL-1 versus B48-in-64 FULL-2. The public artifact URL appears only under `ArxivDraft`, preserving anonymous conference mode. The appendix now explicitly distinguishes archived-table regeneration from the unfinished whole-matrix runner.

## Regression and provenance evidence

The new `.125` synthetic fixture is meaningfully independent of the historical `.00682657` value: it would catch a copied/hardcoded note, asserts only one completed-FULL diagnostic while FULL-2 is absent, verifies the exact `(result path, twin.max_abs_diff)` field source, checks the precision/gate caveat, and verifies the FULL-1 nongate result remains intact. The retained RED log shows failure at `len(notes): 0 != 1`; the retained subsequent table-suite log shows fourteen tests and `OK`, with recorded exit 0. These are root-executed results inspected by this reviewer, not a fresh reviewer test run.

The retained `run_all.sh --tables-only`, `tools/tables.py --check` and `tools/figs.py --check` command records all show exit 0. The table diff changes only the error note and its provenance row; numeric measurements, existing accuracy values, original gates and other six tables are unchanged. The changed `results/tables/sources.tsv` row has the required seven-field TSV structure and an empty last formula field. Its trailing tab is the sole recorded `git diff --check` complaint (root-recorded exit 2); stripping it would change the tabular record. The ten remaining scoped files have a clean recorded whitespace check. This known format-specific whitespace result does not undermine provenance and was not hidden or reported as a globally clean diff.

The direct and isolated existing final logs contain no `Overfull`, undefined-reference/citation, or missing-character lines. Existing direct and isolated extracted text is byte-identical. On rendered page 4, the compact protocol paragraph fits without overlap or clipping. On page 7, the generated error note is visible immediately below the accuracy table and before the existing gate note; it is not clipped and does not replace the pending FULL-2 row. On page 10, the updated artifact/author text and conditional repository URL fit. The page remains visually sparse after the final paragraph, not overflowed. Root's recorded review covers pages 3–10 and unchanged extracted text on 1–2; this reviewer independently inspected only 4/7/10, not all ten pages or the anonymous variant.

## Artifact and submission boundary

README and arXiv metadata correctly identify a **new local interim** PDF/ZIP, retain the immutable earlier public preview, and explicitly preserve pending FULL-2/hardware evidence and absence of upload/server build. The title/abstract/main source remain unchanged. The package still contains nineteen necessary TeX/BibTeX/generated-table/figure members and no raw results, logs, compiler installation or PDF.

At the final evidence freeze, carry forward the previously registered requirement to attach the actual final revision/evidence identity. The local package manifest already records baseline plus exact paper diff and member hashes; that manifest is a sibling record, not one of the nineteen TeX-upload members. Thus the appendix's statement that the source identifies its revision/evidence must be made concrete in the final retained distribution (for example by an immutable release/manifest/source identity), and must not be interpreted as saying the current TeX ZIP itself embeds the sibling manifest. This is the existing cutoff-dependent final packaging task, not a new scientific requirement or a reason to block these interim corrections.

Do not change the public `v1.0.0-preview.1` or its assets. At cutoff, incorporate only accepted completed evidence, keep unavailable measurements unavailable, rebuild/read back the separate final artifacts, refresh metadata/hash/page count, and inspect arXiv's own log/PDF in the authorized submission workflow. The local TeX Live 2026/acmart 2.21 build is not arXiv-server evidence.

## Frozen identities

- Review diff: 27,086 bytes; SHA-256 `0467ad4baf86aabd7b30851f2a239211e399d6664f2ea879bc6d30c3cb2480d2`.
- Direct PDF: `tools/latex/_ext/task56/cutoff-readiness-20261108/arxiv-wrapper.pdf`; 528,922 bytes, 10 pages; SHA-256 `5b2f73b9bff3645574dc1084cb9d38e325ddbcc4da18004387d95203ddd09900`.
- Local ZIP: `tools/release/_ext/arxiv-source-cutoff-readiness-20261108/ckks-train-arxiv-source.zip`; 38,430 bytes, 19 members; SHA-256 `7362cbf975bb0c0052831e2a2778c9bb7f1cebe3417916bb54baeef4e420d89a`.
- Isolated PDF: `tools/release/_ext/arxiv-source-cutoff-readiness-20261108/build/main.pdf`; 528,911 bytes; SHA-256 `d8582bc0b9b402dacbbdb3ed496e9ed2dabb2a7447cbad7667373b43bbaa34da`. It differs in bytes from the direct PDF; extracted text equality, not PDF byte equality, is the verified comparison.
- Direct final log SHA-256 `86d1a670772623d7091d12c7ce566b00d7e62b6fb4edcb135844ccdbe54bcd1c`; isolated final log SHA-256 `6dedaa79b12590c1facc21afd48dbb42e7a2b8ced3f8d03d95e1034a75f8f337`.

| Frozen root-owned file | Bytes | SHA-256 |
|---|---:|---|
| `tools/paper_tables.py` | 50240 | `47468ae980ce082f99ea48daeb7c8d33cfedf3fa3255bef4a6f8a43ad5529741` |
| `tools/test_paper_tables.py` | 11705 | `3009bbe3f32a72c83dc2ff5f7ac4c412839f07d829773ec58fc2f050979f336d` |
| `paper/sections/noninteractive.tex` | 5031 | `ff680d5fbb65b6fb7bd3f4715593cd39b6c016fb8875651bee827848f5ff7d8f` |
| `paper/sections/discussion.tex` | 7011 | `9654e8761c609e8a915aa1e4d94e7d41c11989e8b4f367da94ae82c690772681` |
| `paper/sections/evaluation.tex` | 10592 | `e3e735afa94391127f07b86be352883505ccc269975869e97406e2762b98bae4` |
| `paper/sections/appendix.tex` | 6111 | `6cd5c4f1d2c3b1b8e66fd5f215ed5fc4b17cfa86e13a8ec528f3dd1507675b57` |
| `paper/tables/accuracy_vs_reboot.tex` | 3502 | `02c1b3694fde9fee2c3e7200a78d0d1f386b2cf5b1d0798e2b757dd7f8590a79` |
| `results/tables/accuracy_vs_reboot.md` | 3189 | `b4c2474771934af41b6e47387983de9f9b0ea3e241ade763cc2d53c615df76d0` |
| `results/tables/sources.tsv` | 152189 | `9f8104022e1968f76ff4844e9999b708dc2b87c9741a1b60079ddb51fe0c3c26` |
| `paper/README.md` | 12922 | `e6dcbd197f88bccec3e0fd286c2fb3476cef06faa8016a50ac8b2d7dac61811e` |
| `paper/arxiv-metadata.md` | 5951 | `bc8d0db46b50364a515c46aa9d7f94f874d779a1e2ea2bf239736777f1a8adc3` |

Identical direct/isolated extracted-text SHA-256: `2ce9967f5f5fdce22e576d0a585a0a9d228cc0b51c4963d767bb25c884382035`.
