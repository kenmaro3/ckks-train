# Task54 P4 pure cap provenance-header correction

2026-11-07. **Artifact-only correction complete; numerical verdicts unchanged.** Root authorized the sole low-severity custody correction in independent review558f24e3e7fb29da2f6d01c2891a20d134e5d231ed022f640e9a8f38cbed5e22. No Rust/source edit, Cargo/build/test, Engine/forward call, numerical run, Git, network or subagent was used. Source/Cargo ownership remained with root.

## Preserved initial identities

Before changing anything, verified all53 entries of the initial v2 manifest c8d0eb705d023f617478c60d704f75aba74c99ac5d2e7d44fb539cba49490927. header-before/ preserves byte-exact copies of that manifest, extractor, all four initial TSVs and the original v2 report3290a76c935d38f67699b48a25824605df2f9f9425b9cbd633bcd830422fd20a. The live original v2 report and independent review are unchanged. The new manifest includes the backup manifest as an ordinary artifact, so the initial53-entry inventory remains inspectable.

## Exact correction

The extractor output comment changed only from the copied v1 label to the exact v2 source label:

```diff
--- header-before/extract_evidence.py
+++ extract_evidence.py
@@ -33,7 +33,7 @@
     row['actual_calls']=str(calls[(row['candidate'],row['family'],row['control'])])
 def write_tsv(name,rows,keys):
     with (root/name).open('w') as out:
-        out.write('# NONSCIENTIFIC_TEST_EVIDENCE extracted verbatim fields from sole invocation08\n')
+        out.write('# NONSCIENTIFIC_TEST_EVIDENCE extracted verbatim fields from v2 invocation03 (03-single-numerical.stdout.log)\n')
         writer=csv.DictWriter(out,fieldnames=keys,delimiter='\t',extrasaction='ignore')
         writer.writeheader();writer.writerows(rows)
 margin_keys=['candidate','family','control','update','required_layer','backend','epsilon','epsilon_bits','layer','tensor','coordinate','baseline','baseline_bits','control_bits','max_abs_difference','max_normalized_distance','required','status']
```

Ran only `python3 extract_evidence.py`, reading the existing03-single-numerical.stdout.log. Terminal exit0, with complete output retained in header-reextract.stdout.log/header-reextract.stderr.log. It regenerated the same four TSVs and extracted-summary.txt. The latter remains byte-identical; its regeneration did not change its hash.

Each TSV differs by exactly its first comment line. Every non-comment byte, including column header, record order, values/bits, coordinates, distances, status and line endings, is identical. No raw log, numerical value payload, status, source snapshot, source pin or freeze file changed.

## Before/after identities

| Artifact | Before SHA256 | After SHA256 |
|---|---|---|
| extract_evidence.py | `58b79f6be3ce7b3930e33a6cc7c4690fc5425ff83163738a40465508ac1c5cfe` | `bc69592581fd488c3183ac37463ea7f1f2f1161f6dadfc4ecb24bd81b4e6617b` |
| all-margins.tsv | `4042f7c8f2c16da1d147590ff2c5825d552c544141d8925eb293b1c19601bd52` | `9ccf089f8eea5fc05a179e79b6ea65aac3a89c01f31aad60370d66a65715d4fa` |
| failed-margins.tsv | `61b90c7997a02abaa73e803af8f037739b8a6928d86a749e6cf6f61e3efee6a6` | `23e566a3b61bcb555fd5bde626a960ae9a32e156ed00a9dd9a2869ca56e6876e` |
| all-controls.tsv | `7f4712737c554d146c7fd621f7a8988a2380dad9dd2e2b0eff64c3eba9d3668a` | `2ad08ef8bbca92e889e57603de5b1e7db4990df8d5deec81dc99b340b45c821c` |
| all-hidden-ranges.tsv | `d966494f5a9870e63c4823f6c13a1939b65456b5b668c36e9a3e2e505e672d93` | `f9dd4e97613fbf5d81be33eb1e4d14d1c93d60fe7fe8001bd975300c073ba265` |

header-correction-proof.txt lists pre/post hashes and classifications for all original53 entries: exactly the extractor/four TSVs changed, and every other48 entry is byte-identical. It additionally records each TSV non-comment SHA256 and proves that the full file delta is exactly the one comment replacement. The original report remains3290a76c935d38f67699b48a25824605df2f9f9425b9cbd633bcd830422fd20a; the review remains558f24e3e7fb29da2f6d01c2891a20d134e5d231ed022f640e9a8f38cbed5e22.

Current MANIFEST.sha256 inventories the original artifacts with the five corrected hashes, all seven header-before backups, correction delta/logs/proof and this correction report. Its own updated hash is returned separately to root; the initial manifest remains unchanged in header-before/MANIFEST.sha256. No initial source/report/raw evidence was overwritten.

## Unchanged measured state and closure

v1 remains Original FAIL /Signal FAIL at NI5e-4. v2 remains Original FAIL /Signal PASS at I1e-5/NI4e-4, with57 required witnesses/backend, minimum Signal NI distance8.78906249999998579, and108 Original failed backend rows. Pure training counts remain136 v1 +136 v2 =272 cumulative; this correction adds0 calls. All source/spec/static/numerical conclusions remain attributed to the existing independent reviews and retained runs. This report closes only the copied provenance-label defect for the same reviewer’s ledger-only check; it supplies no HE or future execution permission.
