# Task54 P4 pure cap provenance-header closure review

2026-11-07. **PASS: the minor invocation-label finding is closed.** This is a ledger-only closure of the publication correction identified in independent review `558f24e3e7fb29da2f6d01c2891a20d134e5d231ed022f640e9a8f38cbed5e22`. It is not a new source or numerical acceptance review. Only this closure report was written; no source, Cargo/build/test, Engine/forward, HE, Git, network or subagent operation was performed.

## Independently verified custody

Fully read correction report `task-54-p4-pure-cap-header-correction-report.md`, SHA256 `7bf33bb79171439c631ee334fc1018051ab285bfe35cbe555bd55632656f0304`, the complete extractor patch, correction proof and re-extraction output. Independently rehashed current v2 manifest `aacf36d37052d58b156f6b3abb108ef4dfc3075371aad4cf11a1a1ce76e38b38` and **all65 entries**. Every entry matches, including the correction report and preserved originals.

`header-before/` has exactly the seven required backups: initial manifest, extractor, four TSVs and original v2 report. The backed-up manifest retains `c8d0eb705d023f617478c60d704f75aba74c99ac5d2e7d44fb539cba49490927` and its complete53-entry inventory. Each extractor/TSV backup matches its initial manifest hash. The report backup is byte-identical to the live original report, which remains `3290a76c935d38f67699b48a25824605df2f9f9425b9cbd633bcd830422fd20a`; the prior independent review retains its original558f24e3 hash.

Against those53 original entries, exactly five live hashes changed: extractor and four TSVs. **All other48 are identical**, including raw stdout/stderr, terminal status files, source snapshots, pins/freezes, extracted summary and v1-preservation/comparison artifacts. Each of the53 before/after hash/classification rows in header-correction-proof.txt independently matches the actual bytes.

## Exact minimal correction

The extractor differs from its preserved copy by one comment-string replacement at line36. Independently reconstructed unified delta matches header-extractor-delta.patch exactly. The old label `sole invocation08` is replaced by `v2 invocation03 (03-single-numerical.stdout.log)`, accurately naming the retained v2 input. Parsing, data-field handling and computations are unchanged.

Each of the four TSVs differs by exactly its first comment line, with that same replacement. Every following byte is identical, including column header, record order, values/bits, coordinates, distances, statuses and line endings. Independently computed non-comment SHA256 values match the proof:

| TSV | Unchanged non-comment SHA256 |
|---|---|
| all-margins.tsv | `5da0781d6bca2b8578dcc602349f91c01376000eed85face2e25a46c6041089d` |
| failed-margins.tsv | `220b0ca255380e8b97bf02a409196bc8aad6364e20f16bb32476fd28ac020fd7` |
| all-controls.tsv | `17deba737708e522f7ebfa9a7e2f1afafbda08e789579b09b9e08489cd9b5ced` |
| all-hidden-ranges.tsv | `64b5735a9c749e67fc54a9a3c91e1007888dc15411c311c8ef121e0224b6c139` |

The retained re-extraction stdout exactly equals the unchanged extracted-summary content after its initial evidence marker, and stderr is empty. The correction report/proof record one `python3 extract_evidence.py` execution with exit0. Its previously reviewed read-only log parsing remains unchanged apart from the output comment; no numerical trainer path was introduced. This review did not execute that extractor again.

The requested preservation, exact label correction, non-comment identity proof and repinning are therefore complete. The sole minor evidence-label publication condition from the v2 review is satisfied. No further correction or numerical rerun is needed to close this finding. Root retains document/publication ownership; this closure does not itself publish or send anything.

## Unchanged result and authority

The prior source/SPEC/static conclusions remain attributed to the existing reviews and retained runs. v1 remains Original FAIL /Signal FAIL at I1e-5/NI5e-4. v2 remains Original FAIL /Signal PASS at I1e-5/NI4e-4, with minimum Signal NI separation8.78906249999998579. Pure Engine counts remain136 v1 +136 v2 =272 cumulative; the header correction and this review add0. These are retained earlier results, not a new numerical measurement in this closure.

Actual HE precision/parity and mutant separation, P4 client/session/factory/count/metadata/poison acceptance, P3 real consuming pre-key boundary, P5 blocks, production D7 admission/durable start, secure range/precision/resources and scientific campaigns remain separate and unresolved. Closing the provenance label neither releases those operations nor permits later cap loosening.
