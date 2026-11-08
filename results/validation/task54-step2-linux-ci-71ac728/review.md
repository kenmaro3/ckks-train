# Step2 actual Linux CI evidence — bounded independent review

SPEC: PASS. QUALITY: PASS. No findings in the reviewed scope.

Scope: the complete `step2-ci-doc-append.diff`, the appended paragraphs at `docs/handover.md:1173–1175` and `docs/m5-report.md:907–909`, and `results/validation/task54-step2-linux-ci-71ac728/`. This is evidence/documentation review, not source, Step3, native16, or scientific qualification. No network, source/WIP reads, builds, or test reruns were performed. Only this report was written.

## Independently checked evidence

- Retained `run.json` identifies run `42436964378`, attempt 1, exact head `71ac7280be8f29f42dba3908d8b4f8b9514ddd93`, status completed and conclusion success. All six jobs in `jobs.json` (deny, research-tools, test, fmt, clippy, cuda-check) have that exact head and completed/success. The retained watch result records exit 0 at `2026-11-08T02:22:41.168900+00:00`; API update precedes capture. These are retained records, not a new live API check.
- Recomputed every entry of `SHA256SUMS`; it covers exactly the other archive files. Recomputed ZIP hash/size and all 57 ZIP-member hashes/sizes; the inventory exactly matches `capture.json`.
- Parsed only the original `test/8_Run cargo test --workspace.txt`, avoiding double counting its aggregate-log copy. Its 76 successful target summaries total **1,101 passed, 0 failed, 47 ignored**, including doctests. Both named actual successes occur once: `study_supervisor::handshake::operational_fix2_private_aliases_are_cloexec` and `study_train_invalid_and_ordinary_builds_refuse_without_artifacts`.
- Read `capture.py` without executing it and compared `capture.json` and README against raw API/log data. The Linux workspace command/environment and exact-source conclusions agree with the appended claims.
- Both appended document bodies are identical. Removing the exact new heading and following append restores the previously reviewed hashes: handover `493958ee9e4385421940cd0213fc5b3119033a142cca927ebbd6e66bb73dbf21`; M5 `6cfdc0e4f2d8df635436b2d3ba7c873760a0c2eacd42d48ba0a0ec8043c8417b`. The earlier summary review PASS is retained.

## Claim boundaries

The paragraphs and archive README lines 5–7 correctly distinguish this actual Linux CI from earlier Mac checks and preserve the earlier regression/fix history. The descriptor case is explicitly a no-child Linux check; native16 inheritance/EOF/lock/pidfd, deployment and other native fault checks remain unexecuted. No Step3, GPU hardware, all-experiment, secure-profile or scientific-result approval is inferred. Six successful CI jobs and Linux test counts are supported for this exact commit only.

## Reviewed input pins

| Input | Bytes | SHA256 |
| --- | ---: | --- |
| `step2-ci-doc-append.diff` | 3578 | `9c5ed6490b3c75562c3fa0fb9173fcf698a04b3e778683f06cbcde95b3264b55` |
| Archive `README.md` | 1843 | `27e326109bce0f4834b5ba430557ff7c7b6e250f733c9382b82230ed0430dfbc` |
| `capture.json` | 10751 | `c3e6e516ec49786a5a9e251aaf2e89a6798955a5c46c5a65751b43febdbc5a8c` |
| `SHA256SUMS` | 775 | `3a9041d9eea1387062bd48fed8e4389dcc19ce3a753866a973620f08d3356cb3` |
| `logs.zip` | 206760 | `bd494f872f9dca7312479c995b83585862f24356c6096684cafa2fb94987eabd` |
| Workspace test log ZIP member | 194552 | `73bf153a76b9dae8fa51bc971f8f7a3aabd3c3319913ffc73983e6cc2107ef3c` |
