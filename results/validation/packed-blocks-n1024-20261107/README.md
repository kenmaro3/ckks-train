# Packed-block N1024 correctness evidence, 2026-11-07

These are the first registered interactive (I) and noninteractive (NI) primary attempts on the Mac, on source `ad607daf3e53c3763c4c5db25701f2ede5115154`. Source review and separate I/NI releases preceded execution; the I independent audit preceded the NI release. Both exact single tests exited0 with source/executable hashes unchanged. No historical84/212 run, retry, seed search, fixture change or threshold adjustment occurred.

This is a **small insecure correctness fixture**, not a production-security, performance, memory-admission, GPU or full-dataset result. Full block campaigns and Task54's complete study matrix remain separate and unfinished.

| Retained gate | I | NI |
|---|---:|---:|
| Graph/configuration cases | 2 (G2/G3) | 6 (G2/G3 × AU/WN/AA) |
| Committed updates / final inferences | 8 /2 | 24 /6 |
| Completed and validated refresh calls | 116 client refreshes | 332 bootstraps (299 training+33 inference) |
| Independently inspected512-complex-slot snapshots | 710 | 4,050 |
| Largest independent scalar error | 3.775018180620039e-9 | 3.441401240719656e-8 |
| Original scalar-error gate | 2^-12 | 2^-12 |
| Largest actual input/output bootstrap error | Not applicable | 1.8763991003465872e-8 |
| Original per-bootstrap gate | Not applicable | 2^-20 |

All registered snapshot/role range and numerical gates pass within this scope. Counts of repeated snapshots/slots are coverage, not independent statistical samples. One held-out key seed167 and N1024/512 slots do not establish production N65536 correctness or security. The detailed independent reports retain all omitted-counter/intermediate/generation/seed/ciphertext-provenance limits. In particular, NI has24 independently matched seven-field training counts and332 measured bootstrap counter intervals; actual aggregate inference/full-phase seven-field totals were not printed. The final event counter is not a phase total.

## File custody and roles

`stdout.txt.gz` losslessly compresses the original binary/test stdout. `compression.json` records both original and compressed byte counts/hashes; decompression was independently hashed against each original. Original uncompressed logs remain in the local implementation ledger. `stderr.txt`, `runner.json`, `source-manifest.json`, `runner.py` and `phase-release.md` are unchanged copies of the execution/release records. Runner JSON is external operational provenance, not a binary-emitted scientific result.

`independent-audit.md`, `.py`, `.json` and `-summary.txt` are unchanged **post-run derived audit artifacts**, not new training-result JSON. They independently reconstruct the original scalar equations and inspect raw emitted bits without executing HE. Their numerical tables must not be admitted as production result rows. The scripts preserve their original local ledger paths and strict execution-identity assertions; they are archival audit sources, not portable entry points from this publication directory. Re-running their full identity checks requires the recorded checkout, frozen executable and original ledger inputs referenced by the manifests. This publication does not claim those private build/evidence inputs are recreated by a fresh clone or by run_all.sh.

To inspect the original program output without running any experiment:

```sh
gzip -dc interactive/stdout.txt.gz
gzip -dc noninteractive/stdout.txt.gz
```

The commands assume this evidence directory as cwd. SHA256SUMS covers every published file except itself. Dates/durations in runner metadata are execution provenance only; Mac values must not be mixed with quadruped timing. Source changes in unrelated bench files are disclosed by the NI runner's original working-tree status and do not change the frozen nine NN source files or executable. No secret key/raw ciphertext archive is published; independently audited numerical evidence is the complete decoded bit payload emitted by the reviewed harness.
