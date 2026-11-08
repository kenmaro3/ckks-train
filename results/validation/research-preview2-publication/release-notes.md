This research preview updates the MIT-licensed ckks-train Rust library and its ten-page preprint, **Revisiting Diagonal Packing for Non-Interactive Encrypted Neural Network Training**, by Kentaro Mihara (Independent Researcher).

Fixed source: `e93cec434652a8c9f8c48becc6eb3045a3f78a7f`. This release freezes the reviewed FULL-2 and R-47-3 evidence. Later queued experiments and uncommitted Task54 implementation are excluded. Preview 1 remains available unchanged.

### Results and limits

- FULL-2 completed 4,800 examples over ten epochs at B=48, with one initial-weight seed. Encrypted test accuracy is 90.35%, versus 90.41% for the released ReBoot-code plaintext reference at seed 0: a difference of -0.06 percentage point, within the registered tolerance.
- FULL-2 meets key-switch criteria (a), (c), and (d), but misses criterion (b): 0.781062 versus the target of at most 0.70 when both bootstrap costs use the baseline model. FULL-1 meets all four count criteria at B=64; those separate-batch successes are not combined into an all-conditions result.
- The common-security comparison condition remains unmet, and full-setting ReBoot CPU timing is unmeasured. Baseline model counts are not instrumented baseline wall-clock results.
- R-47-3 measures four independent products at level 5 with three warmups and 30 repetitions. It is an operator microbenchmark, not GPU training speedup or 30 independent training seeds. Of 45 operator cases, 42 executed and three were capacity skips.
- GPU training/ablation terminal results, physical GPU bootstrap measurements, peak GPU memory, additional encrypted models and the full study matrix are not established by this release. This is a research preview, not stable v1.0 or completion of the research plan.

### Validation and artifacts

- [Exact-source Linux CI](https://github.com/kenmaro3/ckks-train/actions/runs/42484163158) completed successfully in all six jobs: research tools, formatting, workspace tests, strict Clippy, CUDA compilation and dependency policy. CUDA compilation is not hardware execution.
- All 6,113 source archive files were checked against Git content and modes. All 18 manuscript inputs match the reviewed build; all 19 ZIP members match the retained package manifest. The PDF is ten pages. Checksums and an asset manifest accompany this release.
- The retained clean-checkout verification regenerated archived tables and checked tables/figures without changes. This is not a rerun of all experiments. The PDF and ZIP reuse the locally compiled and visually reviewed FULL-2/R-47-3 artifacts.
- Software is MIT licensed. The manuscript has a separate licensing notice; the selected arXiv non-exclusive distribution license does not make the paper MIT. The author will perform arXiv submission. No arXiv upload, server build, identifier or conference acceptance is claimed.

### Reproduce the archived tables

```sh
git clone https://github.com/kenmaro3/ckks-train.git
cd ckks-train
git checkout e93cec434652a8c9f8c48becc6eb3045a3f78a7f
bash scripts/run_all.sh --tables-only
python3 tools/tables.py --check
python3 tools/figs.py --check
```

Git metadata is required for provenance-producing experiment commands; the source archive alone does not provide it. Read `docs/m3-report.md`, `docs/m4-report.md`, and `docs/m5-report.md` for the remaining research gates.

The fixed source snapshot retains older publication links and one stale README status sentence saying that FULL-2 is awaited. The generated tables, manuscript, and this release note record its completed result. Updated navigation/status documentation is maintained on main; the fixed archive is not silently rewritten.
