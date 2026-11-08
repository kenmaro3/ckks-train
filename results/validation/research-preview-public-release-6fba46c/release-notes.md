This is a research preview of ckks-train, an MIT-licensed Rust RNS-CKKS library for encrypted neural-network training. It includes the CPU non-interactive training path, CUDA arithmetic/evaluator implementation, archived measurements, and a ten-page single-author preprint by Kentaro Mihara.

Source commit: `6fba46c38f1cdb5be88b9bf1eeecab0f4566b534`. Uncommitted Task54 work is excluded. The source archive has been read back and every tracked file and mode checked against Git. Asset hashes are in `SHA256SUMS`.

### Evidence and limits

- FULL-1 (196 inputs, B=64, one initial-weight seed) is archived. Its four key-switch ratios meet the registered numerical threshold. FULL-2 is still required for the registered comparison against ReBoot seed0 accuracy0.9041.
- Full-setting official ReBoot CPU timing and the equal-security condition remain unresolved. GPU training/ablations and physical GPU bootstrap measurements remain unmeasured. This preview does not declare the original research plan complete.
- Exact-source Linux CI: [run42433577945](https://github.com/kenmaro3/ckks-train/actions/runs/42433577945). The workflow covers workspace tests, formatting, strict Clippy with/without CUDA, CUDA compilation, dependency policy, and research-tool checks. These are software checks, not GPU performance measurements.
- On a clean checkout of this exact commit, `run_all.sh --tables-only`, `tables.py --check` and `figs.py --check` all succeeded without tracked changes. Earlier Mac workspace verification at00f4409 passed1,025 tests (19 ignored); strict rustdoc and additional checks retain their source-specific evidence in the repository.
- The paper and arXiv source ZIP were built and visually checked locally. No arXiv submission or identifier exists yet. The paper is an interim preprint; its separate licensing notice is attached. Software remains MIT.

### Reproduce the archived tables

Clone the repository, check out the source commit above, then run:

```sh
bash scripts/run_all.sh --tables-only
```

This regenerates tables from archived evidence. It does not rerun the incomplete full experiment matrix. Git metadata is needed for provenance-producing experiment commands; a source archive alone does not provide it.

Read `docs/research-preview.md`, `docs/m3-report.md`, `docs/m4-report.md`, and `docs/m5-report.md` for measured, unmet and unmeasured conditions. This is research code, not an audited deployment library.
