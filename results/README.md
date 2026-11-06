# results/

Every number in the reports and the paper comes from a file here, written by the program that
measured it. This page explains the layout, the provenance rule, and how to rerun the
experiments and regenerate the tables.

## Layout

| Path | What it holds |
|---|---|
| `*.json` | Mac runs (Apple M3 Pro): plaintext twins (`mnist196_plain_*`, `mnist_emlp1_plain_*`), Task 51 datasets (`plain_*`), the Task 48 ablation smoke, Iris (2020 method), the Task 33 ReBoot layout in hekit. Mac times are never compared with server times. |
| `logs/` | Logs of the Mac runs. |
| `quadruped/*.json` | Server (quadruped-gpu, i9-13900KF, RTX 4090) results, composed by `quadruped/scripts/compose.py`: the binary's output plus a `quadruped` section with runner facts. `quadruped.binary_output_unmodified` points to the raw file. |
| `quadruped/raw/<job>/` | The binary's own output, unmodified, plus the runner's records (`cmd.txt`, `meta.json`, `pre.txt`/`post.txt`, `rss.tsv`, `time.txt`, `synced-commit.txt`). The tables read these raw files. |
| `quadruped/requests.md` | The server queue (one block per request id; gpu-runner writes the status). |
| `quadruped/logs/`, `quadruped/scripts/` | Server logs and the runner's scripts. |
| `quadruped/data/` | ReBoot's 14x14 arrays and per-seed values (gitignored arrays; manifests with sha256 are committed). |
| `quadruped/raw/reboot_*` | ReBoot's released settings and initial weights, exported **as values only** (ReBoot is GPL-3.0; no ReBoot source is in this repository). |
| `security/` | lattice-estimator logs; the summary table is in `docs/security.md`. |
| `reference/literature.json` | Published values from other papers (HEonGPU), with their citation. The only hand-entered numbers under `results/`; never measurements. |
| `run_all/<mode>/` | Output of `scripts/run_all.sh` (`full` or `quick`): `<id>.json`, `logs/<id>.log`, `meta/<id>.run.json`. |
| `tables/` | Generated Markdown tables and `sources.tsv` (do not edit). The LaTeX versions are in `paper/tables/`. |

## Provenance rule

- The binaries (`mnist`, `plain`, `ablation`) write a `provenance` block themselves: git commit,
  modified tracked files, command line, build, machine, threads, load, peak RSS, start and finish.
  `iris`, `mnist --mode interactive` and `mnist_reboot_layout` do not; for their `run_all` runs,
  `meta/<id>.run.json` records the commit, command and host (runner metadata, not a result).
- **Numbers are never edited by hand.** Older files with a hand-added provenance block
  (`iris_*.json`, `mnist_reboot_layout.json`) say so in `provenance.note`; every other field is the
  binary's output.
- A server result is admissible for same-CPU comparisons only when its provenance names the
  server host; `run_all` refuses a dirty tree unless `--allow-dirty` is given.
- Quick runs (`run_all/quick/`) check the pipeline. They never feed `paper/tables/`.

## Rerun the experiments

```sh
scripts/run_all.sh --dry-run            # commands, status and estimated hours (full mode)
scripts/run_all.sh --quick --dry-run    # the quick matrix
scripts/run_all.sh --quick              # <= 2 h target on quadruped-gpu (subsets, 1 seed)
scripts/run_all.sh                      # the paper's runs (~83 h estimated on quadruped-gpu)
scripts/run_all.sh --only '^plain'      # a subset of run ids (extended regex)
```

Coverage (research plan §4.4 as narrowed by the M3 rulings): MNIST 14x14 (196 inputs) as the
headline, 784-input subset rows as supplementary, Fashion-MNIST and Breast Cancer as plaintext
twins only; eMLP-1 plus eMLP-2/3 as block MLPs; CPU plus GPU at level 7 only; interactive and
non-interactive with the WhenNeeded, AfterUpdate and AfterActivation placements; baselines
(a) the 2020 method on hekit (Iris, `seal_2020`), (c) ReBoot's layout in hekit. Baseline (b),
official ReBoot (OpenFHE), runs only on the server through gpu-runner and is not invoked by
`run_all.sh`.

Seeds: encrypted rows use **1 seed** in full mode (ReBoot's seed-0 initial weights; Iris seed
2020), because each costs hours (the two 196-input full-length runs take about 58 h together).
Plaintext twins use **10 seeds** (eMLP-2/3: seeds 0-2, all that ReBoot's code releases), and the
Task 51 configs run their 10 seeds each. Quick mode uses 1 seed everywhere.

The script is idempotent: a run is skipped when its meta file says `done` at the current HEAD
and the output's own `provenance.git_commit` (if any) matches. Delete the meta file to rerun.
Datasets: `scripts/fetch_mnist.sh`, `scripts/fetch_fashion_mnist.sh`, and ReBoot's 14x14 arrays
copied into `quadruped/data/reboot_emlp1_full/` (checked against `data_manifest.json`).

## Regenerate the tables

```sh
python3 tools/paper_tables.py              # paper/tables/*.tex and results/tables/*.md
python3 tools/paper_tables.py --check      # exit 1 if the committed tables are stale
python3 tools/paper_tables.py --sources    # file and JSON field behind every cell (TSV)
python3 tools/paper_tables.py --sources rq1_ks
python3 tools/paper_tables.py --from-run-all full   # rows from run_all/full/, written to run_all/full/tables/
```

Python 3, standard library only. Tables: `rq1_ks` (four-way key-switch ratios, both B, subset
and full labelled), `wallclock_same_cpu` (server rows only), `accuracy_vs_reboot` (R-M3-15),
`rq3_ablation`, `gpu_vs_cpu_ops` (with the HEonGPU column and its caveat), `security`,
`plain_datasets` (Task 51). Rows whose input does not exist yet print `pending (<request id>)`:
R-41-FULL-1/2, R-47-3, R-48-2/3/4, R-52-1/2. A present input that fails a consistency check stops
the generation. Mean ± sample standard deviation with the seed count n where there are several
seeds.

## Server timing of the runner

| Request | Mode | Wall-clock | `df -h ~` before / after | Status |
|---|---|---|---|---|
| R-54-1 | `--quick` | pending | pending | queued behind R-41-FULL-2 and R-48 |
| (not requested) | full | not run (about 83 h estimated; the FULL rows already run as R-41-FULL-1/2) | — | — |
