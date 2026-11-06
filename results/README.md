# results/

Measured values come from saved program outputs; literature values, security estimates and
model constants are labeled separately. Historical annotations retain their original evidence
limits. This page explains the layout, provenance rule, partial experiment coverage, and the
commands for experiments and table regeneration. Task54's real `--quick` run remains pending.

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

- Every **new** accepted output requires the binary's own `provenance`, including its writer
  and full git commit. `run_all` fails a row whose binary lacks it, including old versions of
  `iris`, interactive `mnist` and `mnist_reboot_layout`. The writer-provenance update for these
  paths is separate Rust work; wrapper metadata never substitutes for that implementation.
- **Numbers are never edited by hand.** Older files with a hand-added provenance block
  (`iris_*.json`, `mnist_reboot_layout.json`) say so in `provenance.note`; every other field is the
  binary's output.
- Same-CPU timing requires the same server host **and thread count**; a present mismatch
  fails table generation instead of being mixed into the comparison.
- `run_all` refuses tracked and untracked changes unless `--allow-dirty` is given. It records
  both separately, plus a SHA256 of tracked/untracked source bytes under `crates/`, `scripts/`,
  `tools/`, `.cargo/` and root Cargo/toolchain/config files. Workspace/ancestor and Cargo-home
  `config` / `config.toml` are included even when ignored/untracked. Ignored external trees
  are excluded. Config contents are hashed, not printed.
- Each build explicitly pins its target directory and native host triple; inherited target-dir
  or cross-target settings cannot redirect the build away from the receipt/execution path.
  Receipts are `target/<native>/release/run_all-build.json` or
  `target/run_all-cuda/<native>/release/run_all-build.json`. They bind HEAD, source/config
  identity, features, Cargo/Rust version information, relevant build environment and binary
  hashes. `--no-build` requires a matching receipt; command and requested thread count also
  participate in done-result reuse. Custom compiler/wrapper environment overrides are rejected;
  configured wrappers are disabled explicitly for this runner's native recipe.
- A runtime HEAD in writer provenance alone cannot establish which code was compiled. This
  receipt is not a hermetic build proof: it does not inventory external SDK/system-library
  contents or arbitrary inputs read by third-party build scripts. Rebuild after such changes.
- Quick runs (`run_all/quick/`) check the pipeline. They never feed `paper/tables/`.

## Rerun the experiments

```sh
scripts/run_all.sh --dry-run            # commands, status and estimated hours (full mode)
scripts/run_all.sh --quick --dry-run    # the quick matrix
scripts/run_all.sh --quick              # <= 2 h target on quadruped-gpu (subsets, 1 seed)
scripts/run_all.sh                      # the paper's runs (~83 h estimated on quadruped-gpu)
scripts/run_all.sh --only '^plain'      # a subset of run ids (extended regex)
scripts/run_all.sh --threads 12         # allowed 1..24; nice 10 for builds and runs
scripts/run_all.sh --no-build           # only with a matching prior build receipt
```

The runner implements a **partial** research-plan matrix; it does not implement or complete
all dataset × model × backend × interaction × seed combinations.

| Scope | Implemented entry / remaining coverage |
|---|---|
| MNIST 196, eMLP-1 NI | FULL-1/2 commands; one released weight seed. FULL-1 raw output and runner exit0 are archived at `quadruped/raw/R-41-FULL-1/`; FULL-2 remains pending. These commands repeat the registered long runs; do not launch them to regenerate draft tables. |
| MNIST 784, eMLP-1 NI | WN/AU at B32/64 and AA at B32, supplementary subset rows. No complete placement × batch matrix or ten encrypted seeds. |
| Interactive eMLP-1 | CPU and CUDA level7 / same-level CPU pair. GPU execution is unmeasured; full-key GPU rows remain memory-constrained. |
| RQ3 | NAG-only CUDA L7, CPU L7 and CPU L31. Conditional SGD fallback and optimizer propagation from R48 are **not implemented in this runner**. Use the registered R48 blocks for that policy; a CUDA memory SKIP stays unmeasured here. Table generation rejects unlike CPU/GPU conditions. |
| eMLP-2/3 | Plaintext released seeds and NI encrypted subset commands. Full encrypted training is a user decision; interactive/GPU block-MLP matrix is unsupported. |
| Fashion-MNIST / Breast Cancer | Task51 plaintext only. Encrypted CPU/GPU, interaction/NI and deeper model combinations are unsupported, not measured successes. Breast Cancer's fixed-epoch miss remains a miss. |
| Baseline (a), 2020 method | In-library Iris `seal_2020`; not a complete external/GPU baseline matrix. |
| Baseline (b), official ReBoot | External OpenFHE environment through gpu-runner and requests; never invoked here. |
| Baseline (c), ReBoot layout in hekit | CPU interactive Lazy/PerSample only. GPU baseline (c) and its NI/other-dataset combinations are unsupported. |
| GPU bootstrap / performance | Existing operation benchmarks are archived references. GPU bootstrap and GPU training adoption/peak gates remain unmeasured; compilation or CPU replay is not a GPU result. |

Seeds: encrypted rows use **1 seed** in full mode (ReBoot's seed-0 initial weights; Iris seed
2020), because each costs hours (the two 196-input full-length runs take about 58 h together).
Plaintext twins use **10 seeds** (eMLP-2/3: seeds 0-2, all that ReBoot's code releases), and the
Task 51 configs run their 10 seeds each. Quick mode uses 1 seed everywhere.

The script is idempotent for a matching HEAD, source identity, binary hash and binary writer.
Committing results changes HEAD and can require a rerun; this is not cross-commit cache reuse.
Missing-input-only selections and metadata persistence failures exit 1. A successful binary
whose runner record cannot be saved is not reported done. GPU capacity skips require exit 3 plus a SKIP log line;
device OOM remains a failure. Builds use nice 10 and `-j N`; plaintext jobs are capped by N.
Delete a meta file to force a rerun. The receipt does not replace writer provenance.
Datasets: `scripts/fetch_mnist.sh`, `scripts/fetch_fashion_mnist.sh`, and ReBoot's 14x14 arrays
copied into `quadruped/data/reboot_emlp1_full/` (checked against `data_manifest.json`).

## Regenerate the tables

```sh
scripts/run_all.sh --tables-only          # canonical archives, no experiment/build
scripts/run_all.sh --tables-only --from-run-all full  # explicit new full run directory
python3 tools/paper_tables.py              # paper/tables/*.tex and results/tables/*.md
python3 tools/tables.py                    # thin compatibility entrypoint, same generator
python3 tools/paper_tables.py --check      # exit 1 if the committed tables are stale
python3 tools/paper_tables.py --sources    # file and JSON field behind every cell (TSV)
python3 tools/paper_tables.py --sources rq1_ks
python3 tools/paper_tables.py --from-run-all full   # rows from run_all/full/, written to run_all/full/tables/
python3 tools/paper_tables.py --from-run-all quick  # smoke coverage only, mode-local output
```

Canonical regeneration reads archived registered inputs. Explicit `--from-run-all full`
replaces registered result rows with `results/run_all/full/<id>.json`; the fixed baseline,
literature, security, and historical model references remain archived. It does not fall back
to old registered results when a new output is missing. New results require writer provenance.
Experiments regenerate only their mode-local tables after execution. Quick outputs generate
`quick_coverage` only, never research tables or FULL validators, and never feed paper tables.

Python 3, standard library only. Tables: `rq1_ks` (four-way key-switch ratios, both B, subset
and full labelled), `wallclock_same_cpu` (server rows only), `accuracy_vs_reboot` (R-M3-15),
`rq3_ablation`, `gpu_vs_cpu_ops` (with the HEonGPU column and its caveat), `security`,
`plain_datasets` (Task 51). Rows whose input does not exist yet print `pending (<request id>)`:
R-41-FULL-1/2, R-47-3, R-48-2/3/4, R-52-1/2. A present input that fails a consistency check stops
the generation before any files are written. Every RQ1, accuracy and timing consumer validates
present FULL inputs independently. Tests use clearly marked **synthetic FULL fixtures**, not
real full-length evidence. Mean ± sample standard deviation includes the seed count n;
matching twin means use identical seed subsets. The independent interactive reference is
shown outside NI-derived (d) and its ways-pass column. All four RQ3 caveats remain in the notes.

HEonGPU's I=4 means four primes per digit (alpha=4), not four digits. The 2581.8 us literature
value is unchanged. Its previously cited Nov2024 manuscript version date is unverified.

## Server timing of the runner

| Request | Mode | Wall-clock | `df -h ~` before / after | Status |
|---|---|---|---|---|
| R-54-1 | `--quick` | pending | pending | requested, not queued on the server; runner stopped. Requires dispatch after FULL and the registered R-48 order |
| (not requested) | full | not run (about 83 h estimated; the FULL rows already run as R-41-FULL-1/2) | — | — |
