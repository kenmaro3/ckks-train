# quadruped-gpu run log

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
