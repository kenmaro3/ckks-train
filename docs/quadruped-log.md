# quadruped-gpu run log

最新の追記は末尾「2026-11-06 11:09:14 JST」。10:49起動観測と09:30引継ぎは各時点のスナップショットとして残す。

## 現在の待機キュー（2026-11-06 10:49:44 JST 起動観測）

この節は下の09:30引継ぎスナップショットの「未投入」を更新する。過去の観測値・回収手順はそのまま残す。

2026-11-06 10:49:44 JST の起動後観測で、post-full コントローラは **waiting**、PID `1739838` / start ticks `467041669`、state ディレクトリは `owner.json` のみ、`clones_created=false` だった。これは待機コントローラの投入であり、R-47 / R-48 / R-52 のビルド・学習・測定開始ではない。R-54 は投入していない。

FULL の最後の実進捗観測は **2026-11-06 10:21:08 JST、FULL-1 update 113（0 始まり）、114/750 更新完了**、elapsed 10956 s。10:30 の gate identity と 10:49 の待機起動観測から後続更新数を推定しない。FULL-1/2 の完了・研究合格は未確認である。

chain10 は PID1700979 / start ticks465769819、開始Unix1793917065.19、stdout `/home/kmihara/hekit-runs/logs/chain10.log`（10:30のidentity観測）。待機キュー source `4b1c49943533bb9ff79493daee33a27b8237a18c`、bundle HEAD `09202cf5650aa76b9b30dd79a68dd615c75a91ae`。ゲート解放後だけ新規 clone で R47 `bd5cc49` → R48 `0e07561` → R52 `9305d2e` を逐次実行し、R52は各21更新のみ。両FULL成功 terminal、chain terminal と非稼働identityの照合前には clone/build/train を開始しない。FULL cloneは変更しない。

config・スクリプト・bundle・観測記録の完全SHA256、cloneパス、失敗/回復手順は [post-full-queue.md](post-full-queue.md) の運用記録を参照。R54は **not queued**。Task54の全実験行列は実装未完・拡張設計中。投入・待機は研究ゲート合格ではない。鍵認証、nice10、各12 threads/合計24以下、長時間バックグラウンド、他テナント非操作を継続する。

## 引き継ぎ: 回収手順 (handover, 2026-11-06 09:30 JST)

The remote-runner session stopped here. Nothing on the server was killed. This section is for someone with only
`ssh quadruped-gpu` (key auth; never a password).

**Queue state at 09:27 JST 2026-11-06**
- Running: `~/hekit-runs/chain10.sh` (pid 1700979) → R-41-FULL-1, at update 80 of 750 after 7733 s (about 96.6 s per
  update; partial check passed with a projection of 70596 s). Expected end about 03:20 JST 2026-11-07. Output:
  `~/hekit-r41full/results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json`, log
  `~/hekit-runs/logs/R-41-FULL-1.log`, provenance `~/hekit-runs/out/R-41-FULL-1/`.
- Then, in the same chain: R-41-FULL-2 (1000 updates, B = 48). Partial rule: after 10 updates it projects the total
  and stops at once if the projection exceeds 55 h. Expected about 38.6 h, so it should end around 18:00 JST
  2026-11-08. Output: `~/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json`, log
  `logs/R-41-FULL-2.log`, `out/R-41-FULL-2/`.
- **Not queued on the server:** R-47-3, R-48-1..4, R-52-1/2 and R-54-1 are in `results/quadruped/requests.md` but
  were never started; no chain exists for them. They need to be queued after chain10, or with a hold file.
- Finished on the server but not yet copied back: none (as of 09:27, every finished job is in `results/quadruped/`).

**Layout on the server**
- `~/hekit-runs/run_job.sh NAME WORKDIR THREADS GPU(0|1) -- CMD...`: runs CMD under `nice -n 10 /usr/bin/time -v`,
  sets OMP/RAYON/CARGO thread caps to THREADS, samples the RSS of its own process tree every 10 s and kills only that
  tree above 90 GiB. Env `SYNCED_COMMIT_FILE` picks the tree's `.synced-commit` (default `~/hekit/.synced-commit`).
- `~/hekit-runs/r41.sh ID OUT_JSON NUM_UPDATES -- mnist-args`: an `mnist` run through run_job, with the partial-run
  watcher (env `HEKIT_DIR`, `PARTIAL_AFTER`, `PARTIAL_STOP_AT`, `PARTIAL_LIMIT_S`). It waits while
  `~/hekit-runs/hold/ID` exists.
- `~/hekit-runs/chainN.sh` and `hekit_chain.sh`: the queues used so far (only chain10 is alive). Copies of every script
  are in `results/quadruped/scripts/`.
- `~/hekit-runs/logs/<job>.log` is the program output. `~/hekit-runs/out/<job>/` holds `cmd.txt`, `workdir.txt`,
  `synced-commit.txt`, `pre.txt`/`post.txt` (date, uptime, df, free, nvidia-smi), `time.txt` (time -v), `rss.tsv`,
  `meta.json`, plus `projection.txt`/`partial.txt` for r41 jobs.
- **Done marker:** the last line of `out/<job>/post.txt` is `JOB_DONE rc=<exit>`. If `time.txt` exists but
  `post.txt` does not, the runner died (see the Iris incident below): rebuild `meta.json` from `pre.txt`, `time.txt`
  and `rss.tsv`, and mark it recovered.
- Trees: `~/hekit` is 43f219f plus 6 dirty files (old R-41-1..5). `~/hekit-r44` is 3e0e99c (F1/F2). `~/hekit-r45` is
  ce4730a (hekit-cuda tests and bench). `~/hekit-r41full` is ff2c2eb plus the R-REB-2 .npy arrays, sha-checked (FULL
  runs). Each tree has `.synced-commit`. Do not modify a tree a running job uses.
- ReBoot stack (GPL-3.0, stays in `~/reboot`, never in a hekit tree): `~/reboot/install` (OpenFHE 1.2.1),
  `~/reboot/venv` (Python 3.11), `~/reboot/src/ReBoot` (3ac173f plus the build-only patch), and my wrappers
  `~/reboot/*.py`. Outputs may contain data only, never ReBoot source.

**Collecting a job** (from the Mac, in `hekit/`)
1. Pull one tar stream (ssh connect takes about 47 s, so use one call or a ControlMaster):
   `ssh quadruped-gpu 'cd ~/hekit-runs && t=$(mktemp -d) && mkdir $t/J && cp -r out/J/. $t/J/ && cp logs/J.log $t/J/program.log && tar -C $t -czf - . && rm -rf $t' | tar -C results/quadruped/raw -xzf -`
   (with J = the job name). For mnist jobs, also copy the binary's JSON from the tree:
   `ssh quadruped-gpu 'cat ~/hekit-r41full/results/quadruped/<name>.json' > results/quadruped/raw/J/<name>.json`, and
   copy `raw/J/program.log` to `results/quadruped/logs/J.log`.
2. Compose with `python3 results/quadruped/scripts/compose.py`:
   - `program_json JOB BINARY_JSON_IN_RAW OUT_NAME "note" "threads note"` copies the binary's JSON and adds a
     `quadruped` block (provenance from `meta.json`). Example:
     `compose.py program_json R-41-FULL-1 mnist196_ni_when_needed_b64_full_seed0.json mnist196_ni_when_needed_b64_full_seed0.json "requests.md R-41-FULL-1, clean clone ~/hekit-r41full at ff2c2eb" "12: rayon (--features parallel, RAYON_NUM_THREADS=12)"`
   - Other modes: `cuda`, `reboot JOB OUT`, `bootstrap`.
   - Never type a number in by hand.
3. Mark the block in `requests.md` (`status: done`, with result path, times, exit status, load, and SLAM presence from
   `pre.txt`/`post.txt`), and add an entry to this log.

**Syncing code** never uses a plain rsync of the tree: the gitignored `_ext/` directories are many GB. Use the file
list from `git ls-files -co --exclude-standard` (`results/quadruped/scripts/sync_hekit.sh`). For a new commit, prefer
`git bundle create x.bundle HEAD`, then on the server a fresh `git clone` (or `git fetch` + checkout in an idle clean
clone), and write `.synced-commit`.

**Shared-tenant rules:** run jobs at `nice -n 10`, with at most 24 threads in total (12 per job of ours), RAM under
90 GB, and `nvidia-smi` checked before any GPU job (record free memory; do not wait for memory). Never touch the
other agent's `persistent_ster*` SLAM processes or vLLM (pid 2791212, about 20.7 GB of GPU memory). Do not overlap a
timed run with our own CPU work; record SLAM load when it appears. Never stop a chain's bash while a job runs under
it (that lost Iris's done marker). To change a running queue, use a hold file or an atomic `mv` of the script.
Delete build objects that are no longer needed; disk had 25 GB free on 2026-11-06.


Runs on the Linux server `quadruped-gpu` (the remote runner session). Results live in `results/quadruped/`,
the scripts that produced them in `results/quadruped/scripts/` (copies of what runs on the server).
Server facts: Ubuntu 26.04, 32 cores, 123 GB RAM, RTX 4090 24 GB (driver 595.91, CUDA 13.1), shared with
another agent's SLAM job (four `persistent_ster*` processes, about 4 cores) and a vLLM process holding about
20.6 GB of GPU memory. Rules followed: `nice -n 10`, at most 24 threads, RAM under 90 GB (watchdog in
`run_job.sh` kills only its own job tree above 90 GiB), no process of others touched.

Every result JSON carries a `provenance` block written by `run_job.sh` (`meta.json`): synced hekit commit
(`~/hekit/.synced-commit`, including the number of dirty files and a hash of `git diff HEAD`), exact command,
thread cap, `uptime` before and after, `/usr/bin/time -v` output (peak RSS), sampled tree RSS every 10 s,
start/end time, and `nvidia-smi` snapshots.

## Setup (2026-11-05)

- 10:42 probe: load 4.02, 29 GB disk free, 110 GB RAM available, GPU 1819 MiB free (vLLM pid 2791212 holds
  20660 MiB, four SLAM processes 384 MiB each).
- SSH from the Mac takes about 47 s to connect; commands use a multiplexed control connection.
- Repo sync: `rsync` of tracked plus untracked-not-ignored files (the plain `rsync --exclude target` pulled in
  the gitignored multi-GB `tools/estimate/_ext/sage-env`, so it was stopped and replaced by a gitignore-aware
  file list). First sync: commit 43f219f, 6 dirty files (working-tree diff sha256/16 `f0f8ec53e649c5de`).
- Rust: rustup under `~/.cargo`/`~/.rustup` (stable, plus the pinned 1.85.1 from `rust-toolchain.toml`).
  `nvcc` 13.1 V13.1.115 at `/usr/local/cuda-13.1/bin`.
- MNIST: `scripts/fetch_mnist.sh` on the server (MD5 and SHA-256 verified), copied to
  `~/reboot/data/MNIST/raw/` for torchvision.
- ReBoot stack under `~/reboot/` (GPL code stays there): OpenFHE v1.2.1 (85a8f63, Release, NATIVEOPT,
  OpenMP, NATIVE_SIZE 64), openfhe-python v0.8.9 (2447d37), ReBoot 3ac173f with the same build-only patch as
  the Mac (`spikes/reboot/_ext/logs/reboot_patches.diff`), CPython 3.11.16 venv via uv, numpy 2.0.0,
  torch 2.15.0+cpu, scikit-learn 1.9.1, pybind11 3.1.0. GCC 15.2. Import test OK.

## Jobs

CPU of the server: 13th Gen Intel Core i9-13900KF, 1 socket, 32 hardware threads (8 performance cores with
2 threads each, 16 efficiency cores). Thread counts below are software threads; with 12 threads a job runs on a
mix of P and E cores, so per-thread speed is not that of a homogeneous Xeon (ReBoot paper: 2x Xeon Gold 5318S).

### 2026-11-05 11:14–11:18 ReBoot smoke run (`raw/reboot_smoke/`, not a result)
1-step run (`--subsample 48`, batch 48, 12 threads) to test the build and the wrapper. Load 4.24 before, 8.84
after (the hekit and spike release builds ran at the same time, so its step time is not reported anywhere).
rc 0, peak RSS 63.5 GB (time -v). Decoder check: decrypted initial weights match the plaintext initial weights
to 5.8e-11. One `bootstrap_array` call per step on 4 ciphertexts with `num_iterations=2`, precision 11.

### 2026-11-05 11:19–11:21 Job 4: CUDA NTT spike → `results/quadruped/cuda_ntt_spike.json`
`spikes/ntt-cuda` with cudarc 0.12.1. That version has no CUDA 13 feature (newest `cuda-12060`) and no
`dynamic-loading` feature (it dlopens libcuda/libnvrtc by default), so on the server copy only, `cuda-12040` was
changed to `cuda-12060` (diff in `raw/cuda_spike_bench/cargo_toml.diff`) and `LD_LIBRARY_PATH` pointed at
`/usr/local/cuda-13.1/lib64` (NVRTC 13.1). The repo file is unchanged. GPU free before: 1819 MiB, enough for
this spike (32 MiB of data plus the CUDA context). Load 4.83 before.
- Tests: 8/8 pass (6 emulation, plus 2 GPU-vs-CPU tests that ran on the GPU and were not skipped).
- RTX 4090 (sm_89), driver 595.91.07 (driver CUDA 13.2): N = 2^16 forward NTT, batch 64, median
  **2.608 µs/poly** (0.1669 ms per batch), bit-exact vs CPU; the 200-rep repeat gives the same median.
  Criterion ≤ 3.04 µs/poly (2 × GPU-NTT's 1.52 µs/poly on a 4090): **pass**.

Snapshots: every job's `raw/<job>/pre.txt` and `post.txt` hold `date`, `uptime`, `df -h ~`, `free -b` and
`nvidia-smi` (query plus process list) taken immediately before and after the run; the table below is read
from them. All jobs below used the server tree at 43f219f plus the 6 uncommitted Mac edits (none in
hekit-core/ckks/nn/bench sources; `.synced-commit` in each raw dir). The chain ran unattended from 12:22.
The four SLAM processes ended between 11:22 and 12:22: they are in the `nvidia-smi` process list before the
ReBoot subset run and absent after it, and GPU free memory rose from 1819 to 3376 MiB. vLLM (pid 2791212) still
holds its memory. Every job after the ReBoot subset run ran without the SLAM load.

| job | start → end (JST) | load 1/5/15 before | load after | disk free | GPU free |
|---|---|---|---|---|---|
| reboot_emlp1_official_subset | 11:21:59 → 12:22:41 | 4.21 5.82 5.43 | 4.03 5.41 6.55 | 26G / 26G | 1819 / 3376 MiB |
| mnist_interactive_b32 | 12:23:16 → 15:25:39 | 2.25 4.81 6.31 | 0.88 1.00 1.14 | 26G / 26G | 3376 / 3376 MiB |
| bootstrap_bench | 15:28:11 → 15:40:35 | 3.47 1.58 1.31 | 3.47 2.38 1.67 | 26G / 26G | 3376 / 3376 MiB |
| R-41-1 | 15:40:45 → 16:54:03 | 3.74 2.48 1.71 | 3.22 3.43 3.11 | 26G / 26G | 3376 / 3376 MiB |
| R-41-2 | 16:54:07 → 17:32:17 | 2.97 3.38 3.10 | 2.30 3.07 3.17 | 26G / 26G | 3376 / 3376 MiB |
| R-41-3 | 17:32:28 → 18:19:50 | 1.95 2.97 3.14 | 2.67 2.77 2.80 | 26G / 26G | 3376 / 3376 MiB |
| R-41-4 | 18:19:50 → 18:44:46 | 2.67 2.77 2.80 | 2.70 2.97 2.82 | 26G / 26G | 3376 / 3376 MiB |

### 2026-11-05 11:22–12:23 Job 1: official ReBoot eMLP-1, subset → `results/quadruped/reboot_emlp1_official_subset.json`
Command (via `reboot_launch.sh` → `run_job.sh`, wrapper `reboot_wrapper.py`): `training_encrypted.py MNIST
--subsample 1024 --num-layers 2 --num-hidden 32 --batch-size 48 --num-epochs 1 --seed 0`, 12 threads
("Detected 12 threads"), OMP_NUM_THREADS=12. ReBoot's own settings (parser and config, recorded in the JSON):
lr 0.0005 (script default; CosineLR constant for the first 5 epochs), momentum 0.9, Nesterov, weight decay 0,
Xavier-uniform init without bias, N = 2^16, depth 24, Δ = 2^49 (first modulus 54 bits), FLEXIBLEAUTOEXT,
HEStd_128_classic, bootstrap level budget [2,2], 2 iterations, precision 11, SPARSE_TERNARY.
- Data: `--subsample 1024` keeps the first 1024 MNIST training images (same images as hekit's `--subset 1024`);
  batch 48 gives 21 steps, so 1008 samples are trained and 16 dropped.
- Per step (ReBoot's own `step_time`, includes that step's weight bootstrap): mean 171.4 s, median 175.4 s,
  min 149.8 s, max 187.4 s. Sum of 21 steps 3599.9 s. Weight bootstrap (`bootstrap_array`, 4 ciphertexts W1 W2 V1
  V2, 2 iterations each) mean 124.1 s per step; 63.1 s on step 1, 112.6 s on step 21. Setup until the weights
  were encrypted: 29.5 s. Script wall 3636.4 s, process wall 3641.6 s.
- Bootstraps: 21 `bootstrap_array` calls, 84 `EvalBootstrap` calls (each with 2 iterations).
- Key switches: **not measured**. Neither ReBoot nor OpenFHE 1.2.1 logs rotation or key-switch counts.
- Test accuracy (decrypted trained weights, plaintext forward pass on the 10,000 test images): **0.7506**
  (initial weights 0.1456). Final train loss 34.56, train accuracy 57.94% (ReBoot's running estimate).
- Peak RSS 71.6 GB (time -v); the paper reports 76.07 GB.
- `WRAPPER_DONE` line kept verbatim in the JSON (`wrapper_done_line`).

**Full setting not run.** The paper protocol is 4800 samples × 10 epochs = 1000 steps. At the measured mean of
171.4 s/step that is about 47.6 h, beyond the 12 h limit (arithmetic, not measured). One epoch of that protocol
(100 steps) would take about 4.8 h plus setup. It is not run unless the controller asks for it in
`requests.md`; no `reboot_emlp1_official_full.json` exists.

### 2026-11-05 12:23–15:26 Job 2a: hekit interactive MNIST, Task 32 arguments → `results/quadruped/mnist_interactive_b32.json`
`cargo run --release -p hekit-bench --bin mnist -- --mode interactive --epochs 1 --subset 1024 --test-subset
10000 --batch 32 --hidden 32 --seed 32 --lr 0.001 --momentum 0.9 --params n16_train --transpose reuse`,
built without the `parallel` feature, so it ran on 1 thread.
- Counts are identical to the Mac run: 20256 key switches (18272 rot + 1984 relin), **19.78 KS/sample**.
  Test accuracy 0.3324, equal to the twin, as on the Mac.
- Time: **epoch 10893.6 s** on this CPU versus **5004.2 s** on the Mac (Apple M3 Pro, `results/mnist_emlp1.json`),
  ratio 2.18. Per batch 339.9 s (Mac 156.1 s, ratio 2.18); client refresh 31.4 s per batch (Mac 21.0 s,
  ratio 1.50). Keygen 20.8 s. Peak RSS 18.0 GB (time -v).
- Load history: the 1-minute load the binary logs after each of the 32 batches was 1.00–5.49 (mean 1.92). The
  10-second samples in `rss.tsv` (n = 1090) were 1.00–7.44 (mean 1.94). The SLAM processes were already gone
  (not in the `nvidia-smi` list in `pre.txt`). The higher load at the start (2.25 4.81 6.31) is the decaying
  average from the ReBoot run that had just ended. Per-batch time was flat (332.0–359.8 s), so contention from
  other jobs does not explain the gap.
- Not established: why one hekit thread is 2.2× slower here than on the M3 Pro. The job was not pinned, so the
  scheduler may have run it on an E-core (16 of the 24 cores), and it ran at nice 10. A rerun pinned to one
  P-core (`taskset`) would separate these factors; not done, because it is not in the brief.

### 2026-11-05 15:28–15:41 Job 3: n16_boot bootstrap bench → `results/quadruped/bootstrap_bench.txt`
`cargo bench -p hekit-ckks --features parallel --bench bootstrap`, `HEKIT_BOOT_THREADS=1,12`; criterion,
10 samples. Setup 24.6 s, 111 key switches per bootstrap, 33 Galois keys, 10.27 GB of keys.
- **1 thread: 45.414 s** [45.340, 45.501] (the requested measurement).
- 12 threads: 13.949 s [13.882, 14.012].
- Peak RSS 16.8 GB.

### 2026-11-05 15:41–18:45 Requests R-41-1..4: Task 41 Step 5, non-interactive eMLP-1 (requests.md)
Built with `--features parallel` and run with `RAYON_NUM_THREADS=12`, through `r41.sh`/`run_job.sh`, which
adds nice 10 and `time -v` and applies the partial-run rule. All four ran a full epoch with exit 0; the rule
did not trigger (each projection was under 5 h).

| block | placement, B | KS/sample incl. bootstraps (without) | bootstraps | epoch s | per update s (boot s) | test acc (twin) | peak RSS |
|---|---|---|---|---|---|---|---|
| R-41-1 | after_update, 32 | 39.469 (25.594) | 128 | 4339.1 | 135.3 (55.6) | 0.3322 (0.3324) | 28.6 GB |
| R-41-2 | after_update, 64 | 20.641 (13.703) | 64 | 2228.9 | 138.8 (55.5) | 0.1929 (0.1929) | 28.6 GB |
| R-41-3 | when_needed, 32 | 32.531 (25.594) | 64 | 2773.6 | 86.4 (27.9) | 0.3323 (0.3324) | 28.1 GB |
| R-41-4 | when_needed, 64 | 17.172 (13.703) | 32 | 1434.8 | 89.2 (27.8) | 0.1929 (0.1929) | 28.2 GB |

In every block the measured counts equal the prediction (`every_update_equals_prediction`). The JSONs are the
binary's output plus a `quadruped` block (threads, provenance); the unmodified binary output is in
`raw/R-41-x/`. Logs: `results/quadruped/logs/R-41-x.log`.

### Disk cleanup 19:55
Deleted the OpenFHE and openfhe-python build trees (91 MB) and the rustup installer. Kept the ReBoot install,
venv, reboot_cpp module and hekit `target/` for reruns. My total on the server is about 3.2 GB (`~/reboot` 1.5G,
`~/.rustup` 1.1G, `~/hekit` 395M, `~/.cargo` 150M). The disk has 26 GB free; the rest of the usage belongs to
other users.

### 2026-11-05 ~20:10 ReBoot config record replaced (controller request)
`raw/reboot_emlp1_official_subset/reboot_config_dump.txt` contained ReBoot (GPL-3.0) source excerpts and was
deleted. It is replaced by `reboot_config_values.json`, which holds plain values only, and
`reboot_emlp1_official_subset.json` was regenerated from it. The 1008 trained indices are in
`reboot_subsample_indices.txt` (one line per step, plus the 16 dropped). They were rebuilt by `scripts/reboot_indices.py`
from ReBoot's own shuffles and checked by a plaintext replay with ReBoot's plaintext Nesterov step: per-step batch
loss within 1.5e-6 (relative) of the encrypted run, batch accuracy equal on 21/21 steps (`reboot_indices.json`).

### 2026-11-05 20:08 R-REB-1: ReBoot initial weights → `raw/reboot_emlp1_official_subset/reboot_init_weights/`
`scripts/reboot_init_weights.py` (run through `run_job.sh`, 2 threads, wall 10 s, peak RSS 0.78 GB). It ran while
the single-threaded Iris run was going (epoch ~355); that overlap is noted here because it may have added a little
time to one or two Iris epochs. W1.npy (784 × 32) and W2.npy (32 × 16) are stored in × out as float64 (ReBoot holds
float32, so the upcast is exact); sha256 values are in `init_weights.json`. Replaying step 1 from the files
reproduces ReBoot's step-1 batch loss 49.33608627 and accuracy 0.1458 exactly (equal to `reboot_indices.json` and to
the encrypted run's step 1).

### 2026-11-05 20:12 Queue change for R-44-1
R-44-1 needs commit ≥ 8b39a57, while R-41-5 is pinned to 43f219f in `~/hekit`. So `~/hekit` was not re-synced.
Instead, the Mac HEAD (85bf330, committed state, 0 dirty files) was cloned from a git bundle into `~/hekit-r44`, and
R-44-1 runs there. The first chain was stopped (only its bash process; the Iris job it had launched continued).
`chain2.sh` waits for Iris, runs R-44-1 (GPU check, then an untimed build, then the tests), and then R-41-5. That way
no build overlaps a timed run. `run_job.sh` now takes `SYNCED_COMMIT_FILE`, so each job records the tree it ran in.

### 2026-11-05 20:20 Queue change: chain3
New blocks R-41-F1, R-41-F2 and R-REB-2 (requests at 3e0e99c). chain2, which was still waiting for Iris and had run
nothing, was stopped. `~/hekit-r44` was re-cloned at 3e0e99c (committed state, 0 dirty files); that commit contains
the R-REB-1 weights and the subset indices, so F1/F2 use the in-tree copies at the paths in the request command
(no flags point at `~/hekit`). `chain3.sh` waits for Iris, then runs R-44-1 → R-41-F1 → R-41-F2 → R-REB-2 → R-41-5
(R-41-5 from the 43f219f tree in `~/hekit`). `r41.sh` now takes `HEKIT_DIR` for the tree.
Finding for R-REB-2: upstream `experiments.sh` runs the full eMLP-1 setting with `precision_analysis.py`, not
`training_encrypted.py`. That script resizes inputs to 14 × 14 (196 features), fixes lr at 5e-4 in its config,
steps the cosine scheduler once per batch (so the lr decays after step 500), trains a plaintext twin alongside, and is
listed with seeds 0, 1 and 2.

### 2026-11-05 18:45–20:20 Job 2b: Iris → `results/quadruped/iris_20.json`
`cargo run --release -p hekit-bench --bin iris -- --epochs 400 --batch 20 --seed 2020`, 1 thread (release profile;
the Mac `results/iris_20.json` binary was built with the LTO bench profile). Load 2.37 2.89 2.80 at start.
- Final: test acc 0.9667, train acc 0.9833, twin equal; max weight error 2.7e-6. Same accuracies as the Mac run.
- Wall 5686.5 s (binary's own) versus 3355.8 s on the Mac (ratio 1.69, different build profile). Peak RSS 1.3 GB.
- **Runner incident:** `run_job.sh` for this job died between the end of its monitoring loop (20:19:38) and writing
  `post.txt`/`meta.json`. The program itself finished normally (time -v exit 0, 1:34:49). The cause is not
  determined; the job's parent chain bash had been stopped at 20:12 to reorder the queue. As a result, chain3 (which
  waited for `post.txt`) sat idle from 20:20 to 20:47. Recovery at 20:47: `post.txt` was written then and marked
  "recovered snapshot, taken late", and `meta.json` was rebuilt from `pre.txt`, `time.txt` and `rss.tsv` (end time =
  mtime of `time.txt`; field `recovered`), so `uptime_after`/`load_after` in the Iris provenance are from 20:47, not
  from the job end. Lesson applied: chains are only stopped while they wait, never while a job runs.

### 2026-11-05 20:50 Queue: chain4 (R-45-1 supersedes R-44-1)
R-45-1 (hekit-cuda Task 44 + 45 tests) runs in a new clean clone `~/hekit-r45` at 7519f77 (contains a6ab1e2,
0 dirty files). R-44-1 was not run and is marked superseded. F1/F2 stay on `~/hekit-r44` at 3e0e99c as announced.
Order: R-45-1 → R-41-F1 → R-41-F2 → R-REB-2 → R-41-5. A dry run of the extended R-REB-2 script (seeds 0–9,
twin replay, data dump) runs into `~/hekit-runs/scratch/` alongside R-45-1, which is a correctness test, not timed.

### 2026-11-05 20:48 R-45-1: hekit-cuda Task 44 + 45 GPU tests → `results/quadruped/hekit_cuda_task45_tests.txt`
Clean clone `~/hekit-r45` at 7519f77 (contains a6ab1e2), cudarc 0.19.0 with `cuda-13010` and `dynamic-loading`,
NVRTC 13.1 via `LD_LIBRARY_PATH`. GPU free before: 2208 MiB (used 21875 MiB, by vLLM and others). Untimed build
`raw/R-45-1-build/`, then `cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1`.
**18 passed, 0 failed, 0 SKIP** (5 Task 44: NTT, INTT round trip, upload, small params, bad ordinal → Device
error; 13 Task 45: pointwise, automorphism, rescale, mod_down, mod_up, QP layout for seal_2020 and n16_train, plus
shape errors). Each GPU test reported a need of ≤ 145 MiB with 1816.6 MiB free. Peak host RSS 0.45 GB.

R-REB-2 dry run (`~/hekit-runs/scratch/rebdry`, 20:47–20:48:36, outside run_job): all 10 seeds, 1000 steps each,
196 inputs, data identical across seeds, pre-scaling min/max −35.69 / 284.48 (bicubic overshoot), no accuracy
numbers in the repo's text files. Twin test accuracy over seeds 0–9: mean 0.9065, sample std 0.0027. It finished
21 s before R-41-F1 started, so it did not overlap the timed run. The provenance-recorded R-REB-2 run happens in its
queue slot and will be checked against this dry run.

### 2026-11-05 20:58 Queue: R-REB-3 slotted before R-41-5 (chain5)
`r41.sh` now waits while `~/hekit-runs/hold/<ID>` exists; it was replaced by an atomic `mv`, so the running
R-41-F1 is unaffected. `hold/R-41-5` was created. `chain5.sh` waits for R-REB-2's done marker, runs a 1-seed 1-epoch
smoke test of `scripts/reboot_paper_setting.py` into scratch, then the full R-REB-3 through `run_job.sh` (10 worker
processes, 1 thread each), and removes the hold. If the smoke test fails, R-REB-3 is skipped with a note and R-41-5 is
released.
Setting notes for R-REB-3: ReBoot's `*_decay_inv` model arguments are passed unchanged as the optimizer's weight
decay (gradient + wd·W), so 0.001 is used for both. The loader's seed-42 shuffle re-seeds the global RNG, and the
script re-seeds with S before building models, so the per-epoch orders do not depend on the data load.

### 2026-11-05 21:01 Queue: R-45-2 (chain6)
R-45-2 (hekit-cuda tests at ≥ 8b81f3d, 22 tests) runs in `~/hekit-r45`, re-synced to 44cb42b via a git bundle. The
first checkout aborted because R-45-1's untracked result copies collided with files now committed; those copies (already
on the Mac) were removed, the checkout repeated, and `.synced-commit` rewritten (0 dirty files). `chain6.sh` starts
when R-41-F2 is done, so the build runs beside the untimed R-REB-2/R-REB-3 jobs and not beside a timed run.

### 2026-11-05 21:08 Queue: R-46-1 supersedes R-45-2 (chain7)
chain6 was stopped while still waiting (no R-45-2 output exists). `~/hekit-r45` was re-synced to 8f01b09 (contains
f68c05f, 0 dirty files). `chain7.sh` starts R-46-1 (32 hekit-cuda tests) when R-41-F2 is done, and records the free GPU
memory at start in `R-46-1.gpufree`. GPU free at 21:08: 2987 MiB.

### 2026-11-05 20:49–21:40 R-41-F1: faithful NI eMLP-1, WhenNeeded, B = 32 → `results/quadruped/mnist_emlp1_ni_when_needed_b32_faithful.json`
Clean clone `~/hekit-r44` at 3e0e99c, in-tree ReBoot indices (1008 samples) and initial weights, `--features
parallel`, 12 threads. Load 4.02 3.94 4.41 before, 4.27 4.33 4.25 after; the other agent's SLAM processes were
running again (3 in `nvidia-smi` at start, 1 at end; 1-min load mean 4.53, max 7.95 over 307 samples), so this run's
wall time had more co-load than R-41-1..4. GPU free at start 2208 MiB.
- 33.488 KS/sample including bootstraps (26.0 without); 17 of 32 updates bootstrapped (68 bootstraps); every update
  equals the prediction. These match the request's expectation (33.49 KS/sample, 17 of 32).
- Epoch 3016.7 s, per update mean 94.0 s (61.6–142.5 s), bootstrap 31.2 s mean. Peak RSS 28.5 GB.
- Test accuracy 0.7577, equal to the plaintext twin (0.7577) and the twin value in the request. ReBoot's own encrypted
  subset run gave 0.7506 (`reboot_emlp1_official_subset.json`).
- 21:45 R-46-1 clone `~/hekit-r45` re-synced to 456a419 (contains b4b9dc3, the Task 46 fix round) before R-46-1 started; 0 dirty files.

### 2026-11-05 21:40–22:08 R-41-F2: faithful NI eMLP-1, WhenNeeded, B = 64 → `results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`
Clean clone `~/hekit-r44` at 3e0e99c, 12 threads. Load 4.27 4.33 4.25 before, 3.32 4.04 4.04 after; one SLAM process in
`nvidia-smi` at start and end, 1-min load mean 4.01 (max 5.72).
- 17.885 KS/sample including bootstraps (13.921 without); 9 of 16 updates bootstrapped (36 bootstraps); every update
  equals the prediction (the request expected 17.89 and 9 of 16).
- Epoch 1607.9 s, per update mean 100.0 s, bootstrap 32.9 s mean. Peak RSS 28.4 GB.
- Test accuracy 0.7358; twin 0.7359 (one test image apart).

### 2026-11-05 22:08 R-REB-2 → `raw/reboot_emlp1_full/`, `data/reboot_emlp1_full/`
Through `run_job.sh`, 2 threads, about 2 min. Output byte-identical to the 20:47 dry run (`diff -rq`). Data arrays
(14×14 train 4800 and test 10000, float32, labels uint8) are in `data/reboot_emlp1_full/` (gitignored) with
`data_manifest.json`; the sha256 values were re-checked on the Mac. Twin test accuracy over seeds 0–9: 0.9065 ± 0.0027.

### 2026-11-05 22:08–22:10 R-46-1: hekit-cuda tests at 456a419 → `results/quadruped/hekit_cuda_task46_tests.txt`
32 test functions ok, 0 failed. 31 ran on the GPU bit-exact; `hoisted_rotate_16_matches_cpu_n16_train` printed the
expected SKIP (needs 5816 MiB, 2595 MiB free). GPU free at start 2987 MiB. Peak host RSS 7.6 GB.

### 2026-11-05 22:08 R-REB-3 setback and R-41-5 start
chain5's smoke test of `reboot_paper_setting.py` (1 seed, 1 epoch) trained and evaluated, but my summary code crashed:
it asked for epochs 10/20/50 of a 1-epoch run. As designed, chain5 then skipped R-REB-3 and removed the hold, so
R-41-5 started at 22:09:08. Its first 44 s (setup and keygen, before any timed update) overlapped the R-46-1 GPU tests
(22:08:52–22:09:52), which run on one CPU thread. The summary is fixed (epochs beyond the run are left out). R-REB-3
is re-queued in `chain8.sh`, which waits for R-41-5, repeats the smoke test and then runs the full job.

### 2026-11-05 22:25 Queue: chain9 (R-47-1, R-47-2)
`~/hekit-r45` re-synced to ce4730a (contains feff7ed). chain9: R-47-1 after R-41-5 (alongside R-REB-3), then R-47-2 after R-REB-3 and R-47-1. Session stopped here (usage limit); results of R-41-5, R-REB-3, R-47-1, R-47-2 are not yet collected.


## 2026-11-06 07:14 Resume after the usage-limit stop
State found: R-41-5, R-REB-3, R-47-1 and R-47-2 had all finished overnight (exit 0); chains 4, 8 and 9 had ended; no job
of ours was running. Machine idle: load 0.11, no SLAM processes, vLLM 20660 MiB on the GPU (3376 MiB free), disk 26 GB
free, RAM 115 GB available.

### 2026-11-05 22:09–23:10 R-41-5: NI eMLP-1, AfterActivation, B = 32 → `results/quadruped/mnist_emlp1_ni_after_activation_b32.json`
43f219f tree in `~/hekit`, 12 threads. 36 KS/sample including bootstraps
(25.59375 without), 3 bootstraps per update; every update equals the prediction. Epoch
3606.6 s, per update mean 112.4 s. Test accuracy 0.3322 (twin 0.3324). Peak RSS
28.2 GB. Load 2.92, 3.80, 3.95 before, 3.59, 3.24, 3.02 after. The first 44 s overlapped the R-46-1 tests (setup/keygen only).

### 2026-11-05 23:10–23:11 R-REB-3: ReBoot paper setting, plaintext → `raw/reboot_emlp1_paper_setting/`, `data/reboot_emlp1_paper/`
chain8: the smoke test (1 seed, 1 epoch) passed, then the full run through `run_job.sh`: 10 worker processes, 1 thread
each, 20 runs (seeds 0–9 × const/sched) × 50 epochs × 1250 steps, 12–17 s per run (the plaintext 196-32-16 step takes
about 0.25 ms), wall 34 s. All 20 curves have 50 epochs with finite outputs. The 60k pre-scaling min/max were
−35.76 / 292.40. The data dir (110 MB, gitignored) holds the arrays, the per-seed init weights and the epoch orders, with
`manifest.json`; all 34 sha256 values were re-checked on the Mac.
Test accuracy over seeds 0–9 (mean ± sample std):
| schedule | best epoch | epoch 10 | epoch 20 | epoch 50 |
|---|---|---|---|---|
| const 0.001 | 0.9535 ± 0.0010 | 0.9494 ± 0.0020 | 0.9513 ± 0.0012 | 0.9515 ± 0.0019 |
| ReBoot scheduler | 0.9546 ± 0.0009 | 0.9495 ± 0.0023 | 0.9516 ± 0.0008 | 0.9525 ± 0.0008 |
Paper Table 4: 94.61 ± 0.27. Gap of the released plaintext path to that figure (mean over seeds, pt): +0.33 to +0.64 at fixed epochs 10/20/50 (both schedules), and +0.74 (const) / +0.85 (sched) at the best epoch. The best-epoch values are optimistic, because the best epoch is picked per seed on the test set.
(Corrected 2026-11-06; the earlier text said "about 0.8–0.9 pt", which overstated the fixed-epoch gap.)

### 2026-11-05 23:10–23:14 R-47-1: hekit-cuda tests at ce4730a → `results/quadruped/hekit_cuda_task47_tests.txt`
45 of 45 test functions ok, 0 failed. 42 ran bit-exact on the GPU; 3 printed expected SKIPs:
`matvec_d64_matches_cpu_n16_train_full` (8436 MiB), `sequence_and_counts_match_cpu_n16_train_full_keys` (2996 MiB, with
2984.6 MiB free, 11 MiB short) and `hoisted_rotate_16_matches_cpu_n16_train` (5816 MiB). The level-9 matvec ran and
passed. It took 3 min 14 s, not the expected 15–20 min. It ran alongside R-REB-3; both are untimed.

### 2026-11-05 23:14–23:22 R-47-2: cuda_ops benchmark → `results/quadruped/cuda_ops_r47.log`, `raw/R-47-2/criterion/`
Run after R-REB-3 and R-47-1 had ended, with no other job of ours; no SLAM processes before or after. Load 2.13 4.64
3.95 at start (decaying from the jobs just finished). Setup 30.9 s (N = 2^16, level 31, dnum 8, 21 Galois keys).
Medians from criterion (GPU / CPU 1 thread / CPU 12 threads):
- add 110.89 µs / 15.50 ms / 14.86 ms
- mul_relin_rescale 5.82 ms / 733.6 ms / 336.5 ms
- rotate1 3.99 ms / 633.0 ms / 224.1 ms
- hoisted_rotate_16 L9: 4.99 ms / 999.2 ms / 176.1 ms; at full level the GPU was skipped (needs 5580 MiB),
  CPU 3.748 s / 687.7 ms
- matvec_enc d64 L9: 16.12 ms / 2.551 s / 549.8 ms; L5 9.51 ms / 1.404 s / 314.9 ms; at full level the GPU was
  skipped (needs 7276 MiB), CPU 12.97 s / 2.319 s
- 4 matvecs at L5: batched on 4 streams 42.41 ms, sequential 34.96 ms
The program output was captured by `run_job.sh` and copied to `cuda_ops_r47.log` instead of using `tee`, which has
the same content.

### 2026-11-06 07:17 Queue: chain10 (R-41-FULL-1, R-41-FULL-2)
New clean clone `~/hekit-r41full` at ff2c2eb (contains ff3042d). The R-REB-2 14×14 arrays were copied in from the
server's original R-REB-2 output; all four sha256 values matched the committed `data_manifest.json`. `r41.sh` now
takes `PARTIAL_AFTER`, `PARTIAL_STOP_AT` and `PARTIAL_LIMIT_S`. For FULL-1 these are 10 / 10 / 108000 s (30 h); for
FULL-2, 10 / 10 / 198000 s (55 h); the projection goes to `projection.txt`. The swap happened with no `r41.sh` running.
chain10 runs an untimed build, then FULL-1 (750 updates), then FULL-2 (1000 updates), with nothing of ours alongside.
Headroom: 26 GB disk (these jobs write only logs and JSON), 115 GB RAM available against the ~28 GB per run seen in
the 784-input NI runs.


### 2026-11-06 R-REB-4: ReBoot's CKKS parameters → `results/quadruped/raw/reboot_crypto_params/params.json`
`scripts/reboot_crypto_params.{py,cpp,sh}` through `run_job.sh`, 1 thread, nice 19, peak RSS 0.56 GB, a few seconds
(07:30–07:31). Correction: R-41-FULL-1's timed updates had already begun at 07:17:56 (its build reused cached artifacts), so R-REB-4 overlapped about 1 minute of them, around updates 6–7. ReBoot's `CryptoContext` was built as in the subset run with
`openfhe.GenCryptoContext` intercepted: the CCParams were captured, the context generated, and the constructor stopped
before key generation and bootstrapping setup, which do not change the modulus chain. A C++ helper linked to the same
OpenFHE 1.2.1 rebuilt the context from those CCParams and read the chain. The enum settings match the captured ones.
Result: ring 2^16; Q has 26 towers (54-bit q0, 24 × 49-bit, and the extra FLEXIBLEAUTOEXT modulus 786433 ≈ 19.58 bits),
log2 Q = 1249.585; P has 8 × 60-bit, log2 P = 480; log2 PQ = 1729.585; HYBRID with dnum 3 (OpenFHE's
default for numLargeDigits = 0), 9 towers per digit; SPARSE_TERNARY with Hamming weight 192 counted on a generated key;
σ = 3.19. openfhe-python's `GetModulus()` overflows a double for this Q, so log Q could not be cross-checked from Python;
the ring dimension agrees.

- 07:40 R-41-FULL-1: projection after 10 updates 70596 s (19.6 h; limit 30 h), so the run continues. Updates alternate 68 s (no bootstrap, 925 KS) and 120 s (4 bootstraps, 1369 KS), matching the prediction. One SLAM process was back at 07:40 (load 6.98). Expected end about 02:55 on 11-07, then FULL-2.

### 2026-11-06 R-REB-5: ReBoot eMLP-2/3 full setting as data → `raw/reboot_emlp23_full/`
`scripts/reboot_emlp23.py` through `run_job.sh`, 1 thread, nice 19, 3.8 s, 0.9 GB, during R-41-FULL-1's timed updates
(negligible next to the 12-thread run). Structure read from the models built by ReBoot's constructor with the MLP-2/3
lines of `experiments.sh`; summary in `config_values.json:structure_summary`. Seed-0 twin test accuracy (recomputed):
eMLP-2 0.9156, eMLP-3 0.9175. Indices differ from eMLP-1's (different number of RNG draws before the epoch shuffles).


### 2026-11-06 11:09:14 JST: FULL-1 update144、post-full待機継続

2026-11-06 **11:09:14 JST**（Unix `1793930954.5309272`）の鍵認証による読み取り観測で、FULL-1 は **update144（0始まり）=145/750更新完了**、elapsed13851s。観測した直近3行（updates142〜144）はすべて `matches prediction true`、post終端はなく、partialマーカーもない。FULL-2には進捗行・post終端がない。FULL-1/2の完了・研究ゲート合格は未確認である。

chain10 PID1700979/start ticks465769819、待機controller PID1739838/start ticks467041669は同じidentityのまま。stateはowner.jsonのみ、clone未作成。R47/R48/R52は引き続き両FULL成功とchain terminal/非稼働identityの検証待ちで、追加のbuild/train/measurementは未開始。R54とeMLP-2/3全長は未投入。実行中cloneや他テナントのプロセスは変更していない。

その後rootが確認した queue commit `4b1c49943533bb9ff79493daee33a27b8237a18c` の [CI run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) は **completed/success、fmt/test/deny/research-tools/cuda-check/clippyの全6ジョブsuccess**。10:49時点のpending記録を更新する。これは当該SHAのLinux CIであり、後続SHA、GPU実測、FULL完了、研究合格を検証したものではない。

一次記録は ledger `post-full-followup-observation.stdout`（SHA256 `eeae57cf21919e2011c61baf26bdccaf4486e502a325e1bdf15d5a58f277edb9`）と `task-post-full-ci.json`（SHA256 `fd1c7d740e6fa64461f0185640a5802b3149af2b7a8f600f01b56b691e5e1a09`）。CI run/SHAの対応はrootの管理観測による。元の研究JSONは変更していない。

## 2026-11-08 10:04 JST: FULL-2の定期確認

読み取り専用の実観測は2026-11-08T10:04:13.560092+09:00。FULL-2はupdate759（0始まり）＝**760/1000更新**、ログ内elapsed110519s。直近6更新は各1415 KS・4 bootstrapでprediction一致。own chainと待機queueは期待start ticksと同じidentityで生存し、最終JSONはまだなく、queueはowner.jsonのみ。[一次記録](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1003/README.md)。本番結果の確定、RQ1精度判定、R47/R48/R52の開始は未確認。次の確認は2026-11-08T11:04:14+09:00以降のR-41-FULL2-OBS-20261108-1104として準備済み・未実行。ジョブ再起動・build・追加学習・queue変更は行っていない。

## 2026-11-08 11:05 JST: FULL-2の定期確認

読み取り専用の実観測は2026-11-08T11:05:37.021755+09:00。FULL-2はupdate784（0始まり）＝**785/1000更新**、ログ内elapsed114222s。直近6更新は各1415 KS・4 bootstrapでprediction一致。own chainと待機queueは期待start ticksと同じidentityで生存し、最終JSONはまだなく、queueはowner.jsonのみ。[一次記録](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1104/README.md)。本番結果の確定、RQ1精度判定、R47/R48/R52の開始は未確認。次の確認は2026-11-08T12:05:38+09:00以降のR-41-FULL2-OBS-20261108-1205として準備済み・未実行。ジョブ再起動・build・追加学習・queue変更は行っていない。

## 2026-11-08 12:05 JST: FULL-2の定期確認

読み取り専用の実観測は2026-11-08T12:05:57.733401+09:00。FULL-2はupdate809（0始まり）＝**810/1000更新**、ログ内elapsed117907s。直近6更新は各1415 KS・4 bootstrapでprediction一致。own chainと待機queueは期待start ticksと同じidentityで生存し、最終JSONはまだなく、queueはowner.jsonのみ。[一次記録](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1205/README.md)。本番結果の確定、RQ1精度判定、R47/R48/R52の開始は未確認。次の確認は2026-11-08T13:06:00+09:00以降のR-41-FULL2-OBS-20261108-1306として準備済み・未実行。ジョブ再起動・build・追加学習・queue変更は行っていない。

<a id="latest-evidence-20261108-1307"></a>


## 2026-11-08 13:07 JST: FULL-2の定期確認

読み取り専用の実観測は2026-11-08T13:07:05.356678+09:00。FULL-2はupdate834（0始まり）＝**835/1000更新**、ログ内elapsed121610s。直近6更新は各1415 KS・4 bootstrapでprediction一致。own chainと待機queueは期待start ticksと同じidentityで生存し、最終JSONはまだなく、queueはowner.jsonのみ。[一次記録](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1306/README.md)。1306は依頼の識別名で、実観測時刻とは区別する。本番結果の確定、RQ1精度判定、R47/R48/R52の開始は未確認。次の確認は2026-11-08T14:08:00+09:00以降のR-41-FULL2-OBS-20261108-1408として準備済み・未実行。ジョブ再起動・build・追加学習・queue変更は行っていない。


<a id="latest-evidence-20261108-1422"></a>

## 2026-11-08 14:22 JST: FULL-2の定期確認

読み取り専用の実観測は2026-11-08T14:22:14.580956+09:00。FULL-2はupdate864（0始まり）＝**865/1000更新**、ログ内elapsed126040s。直近6更新は各1415 KS・4 bootstrapでprediction一致。own chainと待機queueは期待start ticksと同じidentityで生存し、最終JSONはまだなく、queueはowner.jsonのみ。[一次記録](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1408/README.md)。1408は依頼の識別名で、実観測時刻とは区別する。本番結果の確定、RQ1精度判定、R47/R48/R52の開始は未確認。次の確認は2026-11-08T15:23:00+09:00以降のR-41-FULL2-OBS-20261108-1523として準備済み・未実行。ジョブ再起動・build・追加学習・queue変更・terminal collector実行は行っていない。
