This is a research preview of ckks-train, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network training. It includes the CPU non-interactive training path, CUDA arithmetic/evaluator implementation, archived measurements, and a ten-page single-author preprint by Kentaro Mihara.

Source commit: `0b2d17a14cd305890d9eed771008d4f8617eff88`. Uncommitted Task54 work is excluded. The source archive has been read back and every tracked file and mode checked against Git. Asset hashes are in `SHA256SUMS`.

### Evidence and limits

- FULL-1 (196 inputs, B=64, one initial-weight seed) is archived. Its four key-switch ratios meet the registered numerical threshold. FULL-2 is still required for the registered comparison against ReBoot seed0 accuracy0.9041.
- Full-setting official ReBoot CPU timing and the equal-security condition remain unresolved. GPU training/ablations and physical GPU bootstrap measurements remain unmeasured. This preview does not declare the original research plan complete.
- Mac workspace tests:1,025 passed,19 ignored on00f4409; strict Clippy passed with and without CUDA. Subsequent code-path changes are a reviewed documentation link and notice wording. Strict rustdoc, formatting, license checks and archived table/figure regeneration passed oncf58897. Current Linux CI has not run because GitHub blocks runner startup for billing/limits.
- The paper and arXiv source ZIP were built and visually checked locally. No arXiv submission or identifier exists yet. The paper is an interim preprint; its separate licensing notice is attached. Software remains MIT.

### Reproduce the archived tables

Clone the repository, check out the source commit above, then run:

```sh
bash scripts/run_all.sh --tables-only
```

This regenerates tables from archived evidence. It does not rerun the incomplete full experiment matrix. Git metadata is needed for provenance-producing experiment commands; a source archive alone does not provide it.

Read `docs/research-preview.md`, `docs/m3-report.md`, `docs/m4-report.md`, and `docs/m5-report.md` for measured, unmet and unmeasured conditions. This is research code, not an audited deployment library.
