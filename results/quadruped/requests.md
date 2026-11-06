# Server requests for gpu-runner (quadruped-gpu)

Written by the hekit controller session. The user approved it in that session on 2026-11-05
("Yes, use the runner"). gpu-runner runs the blocks in order. After each block it writes the
result path and `status: done` (or `status: failed` with the reason) under the block. It never
commits; the controller commits `results/quadruped/` and `docs/quadruped-log.md` by path.

Common rules for every block:

- Sync the repo at the commit in the block, using the gitignore-aware rsync already in use.
- Build once per commit:
  `cargo build --release -p hekit-bench --features parallel --bins`
- Before and after each run, record `nvidia-smi`, `df -h ~` and `uptime` in docs/quadruped-log.md.
- Do not kill other users' processes. The SLAM job uses 4 cores, so cap threads at 12.
- Run with nohup. Write logs to `results/quadruped/logs/<id>.log`.
- Wrap each run in `/usr/bin/time -v`.

---

## R-41-1: Task 41 Step 5, non-interactive eMLP-1, AfterUpdate, B = 32

- **purpose:** RQ1 spec metric on the same CPU as the official ReBoot run. Bootstraps are counted.
- **commit to sync:** 43f219f
- **working dir:** `~/hekit` (repo root on the server)
- **command:**
  ```
  RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist --mode noninteractive --placement after_update --batch 32 --subset 1024 --epochs 1 --transpose reuse --out results/quadruped/mnist_emlp1_ni_after_update_b32.json
  ```
- **expected duration:**
  - The first batch is logged after about 5–10 min.
  - A full epoch takes about 2–4 h.
  - If the log projects more than 5 h from the first 2 batches, stop after batch 8 and mark the block `partial`. The JSON is written per run, so record the log path either way.
- **GPU memory need:** 0 (CPU only)
- **RAM need:** about 25 GB
- **output path:** `results/quadruped/mnist_emlp1_ni_after_update_b32.json` and `results/quadruped/logs/R-41-1.log`
- **status:** done (2026-11-05T15:40–16:54 JST, exit 0, full epoch, partial rule not triggered: projection after 2 updates: 4421 s for 32 updates). Result: `results/quadruped/mnist_emlp1_ni_after_update_b32.json`; log: `results/quadruped/logs/R-41-1.log`; raw/provenance: `results/quadruped/raw/R-41-1/`. KS/sample incl. bootstraps 39.46875, epoch 4339 s, test acc 0.3322, peak RSS 28.6 GB (kB/1e6). Run as `run_job.sh` (nice 10, /usr/bin/time -v, RAYON_NUM_THREADS=12) on the server tree at 43f219f + 6 uncommitted Mac edits outside hekit-bench/ckks/nn sources (export_params example, docs, tools/estimate).

## R-41-2: Task 41 Step 5, AfterUpdate, B = 64

- Same as R-41-1, with `--batch 64`.
- **output path:** `results/quadruped/mnist_emlp1_ni_after_update_b64.json`
- **expected duration:** about 1.5–3 h, with the same partial rule as R-41-1.
- **status:** done (2026-11-05T16:54–17:32 JST, exit 0, full epoch, partial rule not triggered: projection after 2 updates: 2282 s for 16 updates). Result: `results/quadruped/mnist_emlp1_ni_after_update_b64.json`; log: `results/quadruped/logs/R-41-2.log`; raw/provenance: `results/quadruped/raw/R-41-2/`. KS/sample incl. bootstraps 20.640625, epoch 2229 s, test acc 0.1929, peak RSS 28.6 GB (kB/1e6). Run as `run_job.sh` (nice 10, /usr/bin/time -v, RAYON_NUM_THREADS=12) on the server tree at 43f219f + 6 uncommitted Mac edits outside hekit-bench/ckks/nn sources (export_params example, docs, tools/estimate).

## R-41-3: Task 41 Step 5, WhenNeeded, B = 32

- Same as R-41-1, with `--placement when_needed`.
- **output path:** `results/quadruped/mnist_emlp1_ni_when_needed_b32.json`
- **status:** done (2026-11-05T17:32–18:19 JST, exit 0, full epoch, partial rule not triggered: projection after 2 updates: 2815 s for 32 updates). Result: `results/quadruped/mnist_emlp1_ni_when_needed_b32.json`; log: `results/quadruped/logs/R-41-3.log`; raw/provenance: `results/quadruped/raw/R-41-3/`. KS/sample incl. bootstraps 32.53125, epoch 2774 s, test acc 0.3323, peak RSS 28.1 GB (kB/1e6). Run as `run_job.sh` (nice 10, /usr/bin/time -v, RAYON_NUM_THREADS=12) on the server tree at 43f219f + 6 uncommitted Mac edits outside hekit-bench/ckks/nn sources (export_params example, docs, tools/estimate).

## R-41-4: Task 41 Step 5, WhenNeeded, B = 64

- Same as R-41-1, with `--placement when_needed --batch 64`.
- **output path:** `results/quadruped/mnist_emlp1_ni_when_needed_b64.json`
- **status:** done (2026-11-05T18:19–18:44 JST, exit 0, full epoch, partial rule not triggered: projection after 2 updates: 1478 s for 16 updates). Result: `results/quadruped/mnist_emlp1_ni_when_needed_b64.json`; log: `results/quadruped/logs/R-41-4.log`; raw/provenance: `results/quadruped/raw/R-41-4/`. KS/sample incl. bootstraps 17.171875, epoch 1435 s, test acc 0.1929, peak RSS 28.2 GB (kB/1e6). Run as `run_job.sh` (nice 10, /usr/bin/time -v, RAYON_NUM_THREADS=12) on the server tree at 43f219f + 6 uncommitted Mac edits outside hekit-bench/ckks/nn sources (export_params example, docs, tools/estimate).

## R-41-5: Task 41 Step 5, AfterActivation, B = 32

- Same as R-41-1, with `--placement after_activation`.
- **output path:** `results/quadruped/mnist_emlp1_ni_after_activation_b32.json`
- **note:** this one is lower priority. Run it only after R-41-1 to R-41-4 are done.

M4 GPU blocks (Tasks 44–49) and Task 54 blocks will be appended here as that code lands. Each GPU
block will state its GPU memory need, and the tests skip with a message when the need is not met.

---
- **status:** done (2026-11-05T22:09–23:10 JST, exit 0, full epoch, partial rule not triggered: projection 3850 s). Result: `results/quadruped/mnist_emlp1_ni_after_activation_b32.json`; log `results/quadruped/logs/R-41-5.log`; provenance `results/quadruped/raw/R-41-5/`. KS/sample incl. bootstraps 36 (without 25.59375), 3 bootstraps per update (96 total), every update equals prediction; epoch 3606.6 s, per update mean 112.4 s; test acc 0.3322 (twin 0.3324); peak RSS 28.2 GB. 43f219f tree in ~/hekit. Its first 44 s (setup/keygen) overlapped the single-thread R-46-1 GPU tests; one SLAM process in nvidia-smi at start, none at end.

## R-REB-1: ReBoot official eMLP-1, initial weights as plain data

- **purpose:** hekit's faithful twin, run on ReBoot's exact 1008 samples, lands at 0.71 test accuracy against ReBoot's 0.7506. The likely cause is the initial weights (torch seed 0 vs hekit's ChaCha20). With ReBoot's initial weights published as data, the twin and the encrypted runs can match step by step.
- **commit to sync:** none needed (ReBoot side only)
- **working dir:** `~/reboot`, using the same venv and the same instrumentation approach as the subset run (import by name, no ReBoot code copied)
- **command:** construct ReBoot's eMLP-1 model exactly as the subset run did (seed 0, same config) and save each layer's initial weight matrix before any training step. Save them as `.npy` (float64, shapes as ReBoot stores them), plus a JSON that records:
  - the shape and orientation of each matrix (in × out or out × in)
  - the seed
  - the sha256 of each file

  Check that replaying step 1 of ReBoot's plaintext Nesterov step from these weights reproduces the step-1 batch loss in reboot_indices.json.
- **expected duration:** < 5 min
- **GPU memory need:** 0 MiB. **RAM need:** < 2 GB
- **output path:** `results/quadruped/raw/reboot_emlp1_official_subset/reboot_init_weights/{W1.npy, W2.npy, init_weights.json}`
- **note:** data only. No ReBoot source in any output file.
- **status:** done (2026-11-05T20:08–20:08 JST, exit 0, wall 10 s, peak RSS 0.78 GB, 2 threads, ran while the single-threaded Iris run was going). Output: `results/quadruped/raw/reboot_emlp1_official_subset/reboot_init_weights/` (W1.npy [784, 32], W2.npy [32, 16], both in x out, float64 upcast of ReBoot's float32; init_weights.json has seed, orientation and sha256). Step-1 replay from the .npy files: batch loss 49.33608627319336 and acc 0.1458, equal to reboot_indices.json and to the encrypted run's step 1 (abs diff 0.0). W2 has 16 output columns: 10 classes plus 6 padding columns, which also have nonzero initial weights. Script: `results/quadruped/scripts/reboot_init_weights.py`; provenance: `results/quadruped/raw/R-REB-1/`.

## R-44-1: Task 44 hekit-cuda GPU tests (device, batched NTT/INTT)

- **purpose:** check that the GPU NTT and INTT match the CPU bit for bit (n16_train, 32 limbs), and that a bad ordinal returns a Device error. This is the first run of cudarc 0.19.0 (cuda-13010, dynamic-loading) on the server.
- **commit to sync:** 8b39a57 or later
- **working dir:** `~/hekit`
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  ```
- **pass criterion:** every test passes, and no `SKIP` line appears. Each GPU test prints `<test>: NVIDIA GeForce RTX 4090 (sm_89) (need X MiB, free Y MiB)`. Any SKIP line counts as not passed; record it as-is.
- **expected duration:** 1–2 min for the build, under 1 min for the tests
- **GPU memory need:** at most 104 MiB per test, about 512 MiB peak including the CUDA context. **RAM need:** < 4 GB
- **output path:** `results/quadruped/hekit_cuda_task44_tests.txt` (full stdout and stderr) and `results/quadruped/logs/R-44-1.log`
- **status:** superseded by R-45-1 (not run: the Iris run's runner stalled the queue until 20:47, by which time R-45-1 had arrived)

## R-41-F1: Task 41 faithful encrypted run, WhenNeeded, B = 32, ReBoot's samples and initial weights

- **purpose:** accuracy and KS on ReBoot's exact 1008-sample subset with ReBoot's hyperparameters and initial weights. This is the subset row of the ±0.5 pt check; the plaintext twin gives 0.7577.
- **commit to sync:** 85bf330 or later. The tree must include the R-REB-1 outputs.
- **working dir:** `~/hekit`
- **command:**
  ```
  RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist --mode noninteractive --placement when_needed --batch 32 --faithful --indices results/quadruped/raw/reboot_emlp1_official_subset/reboot_subsample_indices.txt --init-weights results/quadruped/raw/reboot_emlp1_official_subset/reboot_init_weights --epochs 1 --transpose reuse --out results/quadruped/mnist_emlp1_ni_when_needed_b32_faithful.json
  ```
  (Build first: `cargo build --release -p hekit-bench --features parallel --bins`.)
- **expected:** 33.49 KS/sample; 17 of 32 updates bootstrap; about 45–50 min
- **GPU memory need:** 0. **RAM need:** about 25 GB
- **output path:** `results/quadruped/mnist_emlp1_ni_when_needed_b32_faithful.json`, `results/quadruped/logs/R-41-F1.log`
- **status:** done (2026-11-05T20:48–21:40 JST, exit 0, full epoch, partial rule not triggered: projection 3038 s). Result: `results/quadruped/mnist_emlp1_ni_when_needed_b32_faithful.json`; log `results/quadruped/logs/R-41-F1.log`; provenance `results/quadruped/raw/R-41-F1/`. KS/sample incl. bootstraps 33.4881 (without 26), 17 of 32 updates bootstrapped (68 bootstraps), every update equals prediction; epoch 3016.7 s, per update mean 94.0 s; test acc 0.7577 (twin 0.7577); peak RSS 28.5 GB. Clean clone ~/hekit-r44 at 3e0e99c. Caveat for timing: the other agent's SLAM processes were running again (3 at start, 1 at end; mean 1-min load 4.53, max 7.95), unlike R-41-1..4.

## R-41-F2: same as R-41-F1, B = 64

- Same as R-41-F1, with `--batch 64` and `--out results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`.
- **expected:** 17.89 KS/sample; 9 of 16 updates bootstrap; about 25–30 min. Plaintext twin accuracy is 0.7359.
- **output path:** `results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`, `results/quadruped/logs/R-41-F2.log`
- **status:** done (2026-11-05T21:40–22:08 JST, exit 0, full epoch, partial rule not triggered: projection 1570 s). Result: `results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`; log `results/quadruped/logs/R-41-F2.log`; provenance `results/quadruped/raw/R-41-F2/`. KS/sample incl. bootstraps 17.8849 (without 13.9206), 9 of 16 updates bootstrapped (36 bootstraps), every update equals prediction; epoch 1607.9 s, per update mean 100.0 s; test acc 0.7358 (twin 0.7359); peak RSS 28.4 GB. Clean clone ~/hekit-r44 at 3e0e99c. One SLAM process in nvidia-smi at start and end; 1-min load mean 4.01, max 5.72.

## R-REB-2: ReBoot official eMLP-1 full training setting, as data

- **purpose:** the ±0.5 pt criterion is judged on ReBoot's full training setting. Before queueing a full-length faithful hekit run, publish what that setting is.
- **commit to sync:** none (ReBoot side only)
- **working dir:** `~/reboot` (same venv and approach; import by name, no ReBoot code copied)
- **what to produce:**
  - The plain values of ReBoot's full eMLP-1 training config: steps, epochs, batch size, lr schedule (constant or decayed), the number of training samples drawn, and the subsample or shuffle seeds.
  - The per-step training index lists for the full run, in the same format as reboot_subsample_indices.txt, computed by replaying ReBoot's shuffles. Do not train.
  - ReBoot's paper-reported eMLP-1 test accuracy, if it is printed or stored anywhere in the repo's data or logs. Otherwise say "not found".
- **expected duration:** < 10 min. **GPU memory need:** 0
- **output path:** `results/quadruped/raw/reboot_emlp1_full/{config_values.json, train_indices.txt}`
- **status:** done (2026-11-05T22:08–22:08 JST, exit 0; byte-identical to the 20:47 dry run). Outputs: `results/quadruped/raw/reboot_emlp1_full/` (config_values.json; train_indices.txt for seed 0 and train_indices_seed{1..9}.txt; init_weights_seed{0..9}/W1.npy (196×32), W2.npy (32×16); twin_steps_seed{0..9}.tsv) and `results/quadruped/data/reboot_emlp1_full/` (14×14 train/test arrays, gitignored, plus data_manifest.json with sha256). Full setting = precision_analysis.py, 4800 samples, 10 epochs × 100 steps, 196 inputs, lr 5e-4 constant through step 501 then cosine restarts, seeds 0–2 listed in experiments.sh (3–9 same mechanism). Recomputed plaintext twin test accuracy over seeds 0–9: mean 0.9065, sample std 0.0027. Paper-reported accuracy in the repo's text files: not found.

## R-45-1: hekit-cuda GPU tests at a6ab1e2 (Task 44 + Task 45, 18 tests)

- **purpose:** check that the GPU pointwise, automorphism, rescale, mod_down and mod_up kernels match the CPU bit for bit (seal_2020 and n16_train, 8 random inputs plus extremes), and that the Task 44 NTT tests pass at this commit too. If R-44-1 has not run yet, run this block instead and mark R-44-1 "superseded by R-45-1".
- **commit to sync:** a6ab1e2 or later (clean clone, as for R-44-1)
- **working dir:** `~/hekit-r44` (re-synced to a6ab1e2), or a fresh clean clone
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  ```
- **pass criterion:** all 18 GPU tests pass, and no `SKIP` line appears. Record any SKIP as-is.
- **expected duration:** 1–2 min to build, then 1–3 min of tests
- **GPU memory need:** at most 145 MiB per test, about 600 MiB peak. **RAM need:** < 4 GB
- **output path:** `results/quadruped/hekit_cuda_task45_tests.txt` and `results/quadruped/logs/R-45-1.log`
- **status:** done (2026-11-05T20:48–20:48 JST, exit 0): **18 tests passed, 0 failed, 0 SKIP lines** (from cargo's `test result:` lines; 18 tests ran) (5 Task 44 + 13 Task 45 tests; each GPU test printed `NVIDIA GeForce RTX 4090 (sm_89) (need ≤145 MiB, free 1816.6 MiB)`). Clean clone `~/hekit-r45` at 7519f77 (contains a6ab1e2), 12 build jobs, `--test-threads=1`. GPU free before: 2208 MiB. Output: `results/quadruped/hekit_cuda_task45_tests.txt` (full stdout+stderr), `results/quadruped/logs/R-45-1.log`; provenance `results/quadruped/raw/R-45-1/` (build: `raw/R-45-1-build/`).

## R-REB-3: ReBoot paper setting for Table 4 (94.61 ± 0.27), plaintext only

- **purpose:** find out whether the paper's eMLP-1 MNIST figure (Table 4: 94.61 ± 0.27, plaintext, 10 runs) can be reproduced from the released code at the paper's own stated setting. The released full setting gives 0.9065 ± 0.0027 (R-REB-2).
- **paper setting** (arXiv:2506.19693, read by the controller):
  - Table 8 (MNIST, eMLP-1, ReBoot row): lr γ = 0.001, weight decay η = 0.001, batch 48.
  - Table 2: MNIST train size 60,000, test size 10,000, input 14×14.
  - Epochs, lr schedule, preprocessing and seeds are not stated.
- **commit to sync:** none (ReBoot side only). Use the same venv and approach as before: import by name, no ReBoot code copied.
- **runs** (all plaintext, using ReBoot's released plaintext training path, the same one its twin uses):
  - Data: all 60,000 training images, 14×14 with ReBoot's own resize. Min-max scaling over the 60k training set, the same pipeline as R-REB-2 but on the full set. Full 10k test set.
  - Hyperparameters: lr 0.001, weight decay 0.001, batch 48, Nesterov momentum 0.9 (as the code does). Seeds 0–9.
  - Two lr schedules:
    - (i) constant 0.001;
    - (ii) the released code's own scheduler, scaled to the step count, as it would be if the config just set lr = 0.001.
  - Run up to 50 epochs and record test accuracy (argmax over the 10 classes) after every epoch. Do not apply early stopping; just record the curve.
  - Also record, per schedule, the best-epoch mean ± std over the 10 seeds, and the mean ± std at epochs 10, 20 and 50.
- **expected duration:** plaintext 196-32-16 on 60k samples is minutes per seed per schedule. Cap at 12 threads.
- **GPU memory need:** 0. **RAM need:** < 8 GB
- **output path:**
  - `results/quadruped/raw/reboot_emlp1_paper_setting/{config_values.json, acc_curves_seed{S}_{const,sched}.tsv, summary.json}`
  - For hekit's twin later: per-seed init weights and per-step indices for the 60k setting under `results/quadruped/data/reboot_emlp1_paper/`, with the `.npy` files gitignored and a manifest JSON. Include the 14×14 60k train and 10k test arrays as float32 there.
- **order:** after R-REB-2.
- **status:** done (2026-11-05T23:10–23:11 JST, exit 0; smoke test passed first; 20 runs in 10 single-thread workers, 12–17 s each, total wall 34 s). Outputs: `results/quadruped/raw/reboot_emlp1_paper_setting/` (config_values.json, acc_curves_seed{0..9}_{const,sched}.tsv, summary.json) and `results/quadruped/data/reboot_emlp1_paper/` (60k/10k 14×14 arrays, per-seed init weights, per-seed epoch orders as uint16 (50, 60000); gitignored, manifest.json with sha256, re-checked on the Mac). Test accuracy over seeds 0–9 (mean ± sample std): const best epoch 0.9535 ± 0.0010, epoch 10 0.9494 ± 0.0020, 20 0.9513 ± 0.0012, 50 0.9515 ± 0.0019; sched best epoch 0.9546 ± 0.0009, epoch 10 0.9495 ± 0.0023, 20 0.9516 ± 0.0008, 50 0.9525 ± 0.0008. Paper Table 4: 94.61 ± 0.27.

## R-45-2: hekit-cuda GPU tests at 8b81f3d (22 tests, including near-tie base conversion)

- **purpose:** after the Task 45 fix round. The new near-tie inputs for mod_down and mod_up catch FMA contraction, which random inputs cannot. NVRTC now compiles with `--fmad=false`. All 22 GPU tests must be bit-exact.
- **commit to sync:** 8b81f3d or later (use a clean clone)
- **working dir:** a clean clone such as `~/hekit-r45` re-synced to 8b81f3d
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  ```
- **pass criterion:** all 22 GPU tests pass, with no `SKIP` line
- **expected duration:** 1–2 min to build, then 2–4 min of tests
- **GPU memory need:** at most 145 MiB per test, about 600 MiB peak. **RAM need:** < 4 GB
- **output path:** `results/quadruped/hekit_cuda_task45_fix1_tests.txt`, `results/quadruped/logs/R-45-2.log`
- **order:** short; run it at the next free slot, ahead of R-REB-3 if possible
- **status:** superseded by R-46-1 (not run; R-46-1 covers all Task 44/45 tests at a later commit)

## R-46-1: hekit-cuda GPU tests at f68c05f (32 tests, adding GPU key switching, rotation and relinearization)

- **purpose:** check that the GPU hybrid key switch, relinearize, rotate, conjugate and hoisted_rotate all match the CPU bit for bit (Review Focus 5), on seal_2020 and n16_train. This run also covers all Task 44/45 tests, so if R-45-2 has not run yet, run this block in its place and mark R-45-2 "superseded by R-46-1".
- **commit to sync:** b4b9dc3 or later (Task 46 fix round: n16_train capped keys at level 9 cover digits 4+4+2). Use a clean clone.
- **working dir:** a clean clone, for example `~/hekit-r45`, re-synced to f68c05f
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  ```
- **pass criterion:** all 32 GPU tests pass. The only expected SKIP is `hoisted_rotate_16_matches_cpu_n16_train`, which needs 5816 MiB. Any other SKIP counts as not passed, so record it as-is.
- **expected duration:** 6–10 min (build 1–2 min, tests 5–8 min)
- **GPU memory need:** declared per test, peak about 1.4 GB plus the CUDA context; the largest test is rotate n16_train at 1336 MiB. If less than 1.6 GB is free at start, note it: the n16_train tests will then skip on their own.
- **RAM need:** about 7 GB
- **output path:** `results/quadruped/hekit_cuda_task46_tests.txt` and `results/quadruped/logs/R-46-1.log`
- **order:** next free slot. It is GPU-light on compute and can run alongside the untimed plaintext jobs.
- **status:** done (2026-11-05T22:08–22:09 JST, exit 0): **32 of 32 test functions ok, 0 failed; 31 ran bit-exact on the GPU and 1 printed the expected SKIP** (`hoisted_rotate_16_matches_cpu_n16_train: needs 5816.0 MiB of GPU memory, only 2595.2 MiB free`; cargo counts it as passed). No other SKIP. Clean clone `~/hekit-r45` at 456a419 (contains b4b9dc3). GPU free at start 2987 MiB. Output: `results/quadruped/hekit_cuda_task46_tests.txt`, `results/quadruped/logs/R-46-1.log`; provenance `results/quadruped/raw/R-46-1/`. Overlap note: its test phase (22:08:52–22:09:52) overlapped the first 44 s of R-41-5 (setup and keygen, before any timed update).

## R-47-1: hekit-cuda GPU tests at feff7ed (all 45 GPU tests of Tasks 44–47)

- **purpose:** check that `CudaEvaluator` (the HeOps impl, BSGS matvec, multi-stream batch, counters) matches the CPU bit for bit and count for count. Rerun the Task 44–46 tests at the same commit.
- **commit to sync:** feff7ed or later, in a clean clone
- **working dir:** a clean clone, e.g. `~/hekit-r45` re-synced to feff7ed
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  ```
- **pass criterion:** every test passes. Expected SKIPs:
  - `hoisted_rotate_16_matches_cpu_n16_train` (5816 MiB)
  - `matvec_d64_matches_cpu_n16_train_full` (8436 MiB)
  - `sequence_and_counts_match_cpu_n16_train_full_keys` (2996 MiB)
  - `matvec_d64_matches_cpu_n16_train_level9` (1948 MiB), only if free memory is short at that moment

  Record every SKIP line as-is.
- **expected duration:** 15–20 min, including about 55 n16_train CPU key generations
- **GPU memory need:** at most about 1.95 GiB per test. **RAM need:** about 10 GB.
- **timing:** heavy on CPU (key generation), so don't overlap a timed run. Run it after R-41-5, alongside R-REB-3 (untimed) if you like.
- **output path:** `results/quadruped/hekit_cuda_task47_tests.txt`, `results/quadruped/logs/R-47-1.log`
- **status:** done (2026-11-05T23:10–23:14 JST, exit 0, 3 min 14 s): **45 of 45 test functions ok, 0 failed; 42 ran bit-exact on the GPU, 3 printed expected SKIPs**: `matvec_d64_matches_cpu_n16_train_full` (needs 8436 MiB, 2984.6 free), `sequence_and_counts_match_cpu_n16_train_full_keys` (needs 2996 MiB, 2984.6 free), `hoisted_rotate_16_matches_cpu_n16_train` (needs 5816 MiB). `matvec_d64_matches_cpu_n16_train_level9` ran and passed. Clean clone `~/hekit-r45` at ce4730a (contains feff7ed); GPU free at start 3376 MiB; ran alongside R-REB-3 (untimed). Output `results/quadruped/hekit_cuda_task47_tests.txt`, `logs/R-47-1.log`; provenance `raw/R-47-1/`.

## R-47-2: Task 47 benchmark (CPU 1/12 threads vs GPU, n16_train)

- **purpose:** the M4 ops benchmark table: add, mul+relin+rescale, rotate, hoisted_rotate×16, matvec_enc d=64 (capped levels where full keys don't fit), and 4-stream vs sequential. These go next to HEonGPU's RTX 4090 values.
- **commit to sync:** feff7ed or later (same clean clone as R-47-1)
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  HEKIT_BENCH_THREADS=1,12 cargo bench -p hekit-bench --features cuda --bench cuda_ops 2>&1 | tee results/quadruped/cuda_ops_r47.log
  ```
  Afterwards, copy `target/criterion/cuda_ops_n16_train/` (only the `estimates.json` and `benchmark.json` files) into `results/quadruped/raw/R-47-2/criterion/`.
- **timing:** this is a timed CPU+GPU benchmark, so run it with no other hekit or ReBoot job on the machine: after R-REB-3 and R-41-5. Record the SLAM load before and after.
- **expected duration:** 20–30 min
- **GPU memory need:** at most about 1.6 GiB (rows that don't fit print "skipped: needs …"). **RAM need:** about 10 GB.
- **output path:** `results/quadruped/cuda_ops_r47.log` and `results/quadruped/raw/R-47-2/criterion/`
- **status:** done (2026-11-05T23:14–23:21 JST, exit 0), after R-REB-3 and R-47-1 had ended, with no other job of ours; untimed bench build before it. SLAM processes: none before or after (`raw/R-47-2/slam_{before,after}.txt`); load 2.13, 4.64, 3.95 before (decaying from the jobs that had just ended), 6.00, 3.38, 3.38 after. GPU rows `hoisted_rotate_16` (needs 5580 MiB) and `matvec_enc_d64` full (needs 7276 MiB) printed `skipped`; the capped L9/L5 rows ran. Output `results/quadruped/cuda_ops_r47.log` (full program output, captured by run_job instead of tee) and `results/quadruped/raw/R-47-2/criterion/` (estimates.json and benchmark.json only). Clean clone `~/hekit-r45` at ce4730a.

## R-41-FULL-1: faithful full-length encrypted run, 196 inputs, WhenNeeded B=64, seed 0 (RQ1 headline)

- **purpose:** the RQ1 headline at ReBoot's full released setting: 196 inputs, 4800 samples, 1000 ReBoot steps (48 000 samples), ReBoot's per-step lr schedule (sample-aligned to B=64), and ReBoot's seed-0 initial weights. The run measures KS/sample including bootstraps, epoch time, and test accuracy against the twin. The 196 twin already reproduces ReBoot's twin for all 10 seeds (test accuracy identical, seed 0 = 0.9041).
- **commit to sync:** ff3042d or later (clean clone)
- **working dir:** clean clone, e.g. `~/hekit-r41full`
- **build:** `cargo build --release -p hekit-bench --features parallel --bins`
- **command:**
  ```
  RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist --mode noninteractive --placement when_needed --batch 64 --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full --indices results/quadruped/raw/reboot_emlp1_full/train_indices.txt --init-weights results/quadruped/raw/reboot_emlp1_full/init_weights_seed0 --init-weights-manifest results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-schedule results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-key lr_per_step --lr-schedule-batch 48 --epochs 1 --transpose reuse --out results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json
  ```
  The data arrays must be present at `results/quadruped/data/reboot_emlp1_full/`. They are gitignored, so copy them in; their sha256 is checked against data_manifest.json.
- **expected:** 18.27 KS/sample (750 updates, every one checked against the closed form). About 18.5 h, extrapolated from the 784-input runs.
- **partial rule:** after the first 10 updates, write the projected total to the log. If it exceeds 30 h, stop and mark the block `partial` with the projection. Do not stop below that.
- **GPU memory need:** 0. **RAM need:** about 25 GB
- **output path:** `results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json`, `results/quadruped/logs/R-41-FULL-1.log`
- **order:** after R-47-2. It is a long timed run, so nothing of ours should overlap it. Untimed jobs may run only if you record them.
- **status:** running/queued in chain10 (started 07:17 JST 2026-11-06): untimed build, then R-41-FULL-1, then R-41-FULL-2, nothing else of ours alongside. Clean clone `~/hekit-r41full` at ff2c2eb (contains ff3042d, 0 dirty files); the R-REB-2 14×14 arrays were copied in from the server's original R-REB-2 output and their sha256 matched data_manifest.json. Same flags as the block (argument order differs: `--batch` comes after the shared flags). Disk 26 GB free, RAM 115 GB available, machine idle (load 0.1, no SLAM processes) at 07:14. Partial rule: projection after 10 updates to `raw/R-41-FULL-1/projection.txt`; stop at once if over 30 h (108000 s).
- **note:** R-REB-4 (1 thread, nice 19, including a g++ compile) ran 07:30–07:31 during this run's timed updates (around updates 6–7); it may add a little to those updates' times.
- **progress (07:40 JST 2026-11-06):** partial check passed: projection after 10 updates 70596 s (19.6 h) for 750 updates, under the 30 h limit, so the run continues (`raw/R-41-FULL-1/projection.txt` once collected). Updates alternate 68 s with 0 bootstraps (925 KS) and 120 s with 4 bootstraps (1369 KS), all matching the prediction (18.272 KS/sample). Expected end about 02:55 JST 2026-11-07. One SLAM process was running again at 07:40 (1-min load 6.98).

## R-41-FULL-2: faithful full-length encrypted run, 196 inputs, B=48 in 64 lanes, seed 0 (±0.5 pt accuracy check)

- **purpose:** ReBoot's own batch size. The in-binary twin is ReBoot's twin, so this checks encrypted test accuracy against seed 0's 0.9041.
- **commit and dir:** as R-41-FULL-1
- **command:**
  ```
  RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full --indices results/quadruped/raw/reboot_emlp1_full/train_indices.txt --init-weights results/quadruped/raw/reboot_emlp1_full/init_weights_seed0 --init-weights-manifest results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-schedule results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-key lr_per_step --epochs 1 --transpose reuse --out results/quadruped/mnist196_ni_b48_full_seed0.json
  ```
- **expected:** 28.99 KS/sample (1000 updates, every update bootstraps). About 38.6 h.
- **partial rule:** after the first 10 updates, log the projected total. If it exceeds 55 h, stop and mark `partial`.
- **GPU memory need:** 0. **RAM need:** about 25 GB
- **output path:** `results/quadruped/mnist196_ni_b48_full_seed0.json`, `results/quadruped/logs/R-41-FULL-2.log`
- **order:** after R-41-FULL-1
- **status:** running/queued in chain10 (started 07:17 JST 2026-11-06): untimed build, then R-41-FULL-1, then R-41-FULL-2, nothing else of ours alongside. Clean clone `~/hekit-r41full` at ff2c2eb (contains ff3042d, 0 dirty files); the R-REB-2 14×14 arrays were copied in from the server's original R-REB-2 output and their sha256 matched data_manifest.json. Same flags as the block (argument order differs: `--batch` comes after the shared flags). Disk 26 GB free, RAM 115 GB available, machine idle (load 0.1, no SLAM processes) at 07:14. Partial rule: projection after 10 updates; stop at once if over 55 h (198000 s).

## R-REB-4: ReBoot's actual CKKS modulus chain and secret distribution, as data (for the security comparison)

- **purpose:** estimate ReBoot's security with the same lattice-estimator models used for hekit (BDGL16/MATZOV), on ReBoot's actual parameters. The subset run's printout gives ring 2^16, mult depth 24, scaling 49 bits, first modulus 54 bits, FLEXIBLEAUTOEXT, HEStd_128_classic, SPARSE_TERNARY, and bootstrapping level budget [2,2] with 2 iterations. It does not give the full modulus chain or the key-switching modulus.
- **commit to sync:** none (ReBoot/OpenFHE side only). Use the same venv. Build ReBoot's crypto context for eMLP-1 exactly as the subset run does, including EvalBootstrapSetup, then query it through the OpenFHE Python API. Data only; no ReBoot code in the outputs.
- **what to record:**
  - the ring dimension;
  - every Q modulus (value and bit size) of the final context after bootstrapping setup, and its total log Q;
  - every P (key-switching) modulus and total log P, plus dnum (num large digits) and the key-switching technique;
  - log PQ;
  - the secret key distribution and its Hamming weight as OpenFHE implements SPARSE_TERNARY in v1.2.1. Read it from OpenFHE's documentation, or check it empirically by generating a key and counting nonzeros. OpenFHE is BSD-2, so reading its docs or source is fine.
  - the error standard deviation (σ).
- **expected duration:** < 5 min. **GPU memory need:** 0
- **output path:** `results/quadruped/raw/reboot_crypto_params/params.json`
- **order:** at the next free slot. It is light and must not disturb the timed FULL runs; run it at nice 19 in a few seconds of CPU.
- **status:** done (2026-11-06T07:30–07:31 JST, exit 0, 1 thread, nice 19, peak RSS 0.56 GB; overlapped about 1 min (07:30–07:31) of R-41-FULL-1's timed updates, around updates 6–7; it used 1 thread at nice 19, plus a g++ compile). Output: `results/quadruped/raw/reboot_crypto_params/params.json` (raw captures in `raw/R-REB-4/`). Ring 2^16; Q: 26 towers (54-bit q0, 24 × 49-bit, last tower 786433 ≈ 19.58 bits = FLEXIBLEAUTOEXT's extra modulus), log2 Q = 1249.585; P: 8 × 60-bit, log2 P = 480; log2 PQ = 1729.585; HYBRID, dnum = 3 (OpenFHE's choice for numLargeDigits = 0), 9 towers per digit; SPARSE_TERNARY with Hamming weight 192 measured on a generated key (97 × +1, 95 × −1); σ = 3.19. Method: ReBoot's own CryptoContext constructor with openfhe.GenCryptoContext intercepted (stopped before keygen, so no 44 GB context), then a C++ helper on the same OpenFHE 1.2.1 install reading the chain from those CCParams.

## R-47-3: Task 47 batch vs sequential re-measurement (after the fix round), plus all GPU tests at 37e80a9

- **purpose:** first, re-time the 4-product batch at n16_train L5 on 1, 2 and 4 streams after the fix (pool release threshold set, no per-call reserve, no threads); bit-identity is checked inside the example. Second, re-run all hekit-cuda GPU tests at the fix commit.
- **commit to sync:** exactly bd5cc493db166a282d26b05b1e069f2a0ee629ef in the post-full queue (fresh clone; the example times both pool thresholds, `[default]` 256 MiB and `[max]`)
- **working dir:** fresh `/home/kmihara/hekit-post-full-clones/hekit-post-full-r47`; no existing clone is re-synced
- **command:**
  ```
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
  cargo build -p hekit-cuda --features cuda --release --example batch_streams
  HEKIT_BATCH_REPS=30 target/release/examples/batch_streams 2>&1 | tee results/quadruped/batch_streams_r47_3.log
  test -x /usr/local/cuda-13.1/bin/nsys && HEKIT_BATCH_REPS=5 /usr/local/cuda-13.1/bin/nsys profile --trace=cuda --force-overwrite=true -o results/quadruped/raw/R-47-3/r47_3_batch target/release/examples/batch_streams
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1 2>&1 | tee results/quadruped/hekit_cuda_task47_fix1_tests.txt
  ```
- **pass criterion:** the example prints medians and does not panic (batch results bit-identical to sequential). In the tests, all pass, and the only expected SKIPs are hoisted_rotate_16 full and matvec full. The full-key sequence test now declares 2868 MiB and should run if that much memory is free.
- **expected duration:** about 5 min for the example (+1 min with nsys), then 3–5 min for the tests
- **GPU memory need:** at most about 2.8 GiB (the sequence test), about 1 GiB for the example. **RAM need:** about 10 GB
- **timing:** this block is timed (example) and CPU-heavy (key generation in the tests). Run it after R-41-FULL-1 and R-41-FULL-2, or in a gap between them if the gap is longer than 15 min. Never run it during a FULL run.
- **output path:** `results/quadruped/batch_streams_r47_3.log`, `results/quadruped/hekit_cuda_task47_fix1_tests.txt`, `results/quadruped/raw/R-47-3/`
- **status:** queued, waiting (2026-11-06 10:49:44 JST; post-full-v1 controller PID1739838 / start ticks467041669). No clone/build/run started at this observation. Both successful FULL terminals plus chain10 terminal/inactive identity must validate before release. Exact pinned fresh clones and source/config/script/bundle hashes: `docs/post-full-queue.md`. This supersedes the 09:30 not-queued handover snapshot; it is not a measured result.

## R-REB-5: ReBoot eMLP-2 / eMLP-3 block structure and training values, as data (for Task 52)

- **purpose:** Task 52 implements ReBoot's block MLP (RE/CE blocks, each with a local classifier and local loss, and no gradient between blocks). The paper (Table 3) gives eMLP-2 = RE(64)→CE(32) and eMLP-3 = RE(128)→CE(64)→RE(32). The block internals and the training setting must come from ReBoot's released code, recorded as values only.
- **commit to sync:** none (ReBoot side only). Use the same venv and approach: import by name, never copy code.
- **what to record** for eMLP-2 and eMLP-3 on MNIST, at the full released setting used by experiments.sh / precision_analysis.py, as plain values:
  - each block's kind (RE/CE); its forward map (which layers, widths, activation, and whether a CE block widens or narrows); the local classifier's shape and its output padding; and the input each block receives (previous block's activation, or the raw input);
  - the loss per block, its reduction (sum or mean over batch and outputs), and how the blocks' losses combine (independent updates, or summed);
  - whether anything flows between blocks: detached activations, a shared optimizer step;
  - which block's head is used at test time (last block, or an ensemble);
  - lr, momentum, weight decay per block, the lr schedule, batch size, step count, and seeds;
  - initial weights for every layer and head, as .npy (seed 0 is enough; 0–2 if cheap), with orientation and sha256;
  - per-step indices if they differ from the eMLP-1 full run (the same subsample seed probably gives the same indices; say so if they're identical);
  - ReBoot's plaintext twin, recomputed with ReBoot's plaintext path: per-step batch loss and accuracy, plus test accuracy, for seed 0.
- **expected duration:** < 15 min (plaintext only). **GPU memory need:** 0
- **output path:** `results/quadruped/raw/reboot_emlp23_full/{config_values.json, init_weights_emlp{2,3}_seed{S}/, twin_steps_emlp{2,3}_seed0.tsv}`
- **order:** next free slot, at nice 19. It must not disturb the timed FULL runs; if it would take more than a few CPU-minutes, run it after FULL-2.
- **status:** done (2026-11-06T07:55–07:55 JST, exit 0, 1 thread, nice 19, wall 3.8 s, peak RSS 0.90 GB; ran during R-41-FULL-1's timed updates, negligible). Output: `results/quadruped/raw/reboot_emlp23_full/` (config_values.json with `structure` read from the built models and `structure_summary`; train_indices_emlp{2,3}_seed{0,1,2}.txt; init_weights_emlp{2,3}_seed{0,1,2}/<layer>.npy with sha256; twin_steps_emlp{2,3}_seed0.tsv). Full setting = precision_analysis.py, 196 inputs, 4800 samples, 1000 steps, lr 5e-4 then ReBoot's cosine schedule, decay 0. eMLP-2 = local block 196→64 (head 64→16) + output stack 64→32→16; eMLP-3 = blocks 196→128 (head 128→16), 128→64 (head 64→16) + output stack 64→32→16; no gradient between blocks; test uses the output stack. Recomputed seed-0 twin test accuracy: eMLP-2 0.9156, eMLP-3 0.9175. Per-step indices differ from eMLP-1's for every seed (different number of init draws before the epoch shuffles), so they are given per model.

## R-48-1..4: Task 48 GPU training checks and RQ3 ablation (pinned to 0e07561; replaces the 626160e text)

Deployed post-full queue: fresh `/home/kmihara/hekit-post-full-clones/hekit-post-full-r48` at exactly `0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3`; original `hekit-*` names remain correct. Its runner records actual child post/meta status, and SGD fallback requires both exit 3 and a SKIP line. The manual setup below describes the request; the deployed controller performs clean bundle checkout and checked data copy after its FULL/chain gates.

Run these blocks **after R-41-FULL-2 and R-47-3**, in the order R-48-1 → R-48-2 → R-48-3 → R-48-4, each one alone, and never during a FULL run.

Run every R-48 block under **bash** (it uses `PIPESTATUS`).

Common setup:
```
cd ~/hekit-r48 && git fetch && git checkout --detach 0e07561 && git status --porcelain --untracked-files=no   # must print nothing (or rsync a clean tree at 0e07561)
export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}
export MNIST=$HOME/hekit-r41full/data/mnist   # or scripts/fetch_mnist.sh in the clone
export RAYON_NUM_THREADS=12
cargo build -p hekit-bench --features cuda --release --bins --tests   # untimed, ~3–5 min
mkdir -p results/quadruped/raw/R-48-1 results/quadruped/logs
```

Failure rule, for every block:
- **Exit 3 or a `SKIP` line:** not enough free GPU memory. Record `skipped: needs N MiB, M free`; that row stays unmeasured.
- **Exit 1 (or a failing test) with `device error` / `out of memory` in the log:** record it, keep the log, mark the block `failed: device OOM`, and continue with the next block. Do not retry, and do not switch to SGD; SGD is only for exit 3 in R-48-2.
- **Any other exit 1:** record it as `failed`, keep the log, and continue.

### R-48-1: Step 1 GPU-vs-CPU training tests
```
HEKIT_MNIST_DIR=$MNIST HEKIT_TASK48_OUT=results/quadruped/raw/R-48-1 nice -n 10 cargo test -p hekit-bench --features cuda --release --test mnist_cuda --test iris_cuda -- --ignored --nocapture --test-threads=1 2>&1 | tee results/quadruped/task48_gpu_tests.txt
```
- **Pass:** 5/5 ok. Skipped tests also pass, so record which variants printed SKIP, and record the `max |W_gpu - W_cpu|` lines.
- **Step 1 at n16_train is met only if `n16_train_level7` ran.** It needs 2456 MiB free.
- `n16_train_full_keys` (15560 MiB) is expected to SKIP.
- **GPU:** ≤ 2.4 GiB. **RAM:** ≤ 15 GB. **Duration:** 35–45 min.

### R-48-2: GPU ablation, n16_train level 7 (timed)
```
nice -n 10 target/release/ablation --backend cuda --params n16_train --top-level 7 --optimizer nag --iters 5 --warmup 1 --data $MNIST --out results/quadruped/ablation_cuda.json 2>&1 | tee results/quadruped/logs/R-48-2.log
rc=${PIPESTATUS[0]}; OPT=nag
if [ "$rc" -eq 3 ]; then
  OPT=sgd
  nice -n 10 target/release/ablation --backend cuda --params n16_train --top-level 7 --optimizer sgd --iters 5 --warmup 1 --data $MNIST --out results/quadruped/ablation_cuda_sgd.json 2>&1 | tee results/quadruped/logs/R-48-2-sgd.log
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 3 ] && OPT=nag   # both GPU runs skipped: the CPU row R-48-3 stays NAG (the more useful row)
fi
echo "$OPT" > results/quadruped/logs/R-48-2.optimizer
echo "R-48-2 optimizer=$OPT exit=$rc" | tee -a results/quadruped/logs/R-48-2.log
```
- **Pass:** exit 0, `all_checks_pass: true`, and the `provenance.gpu` field is present.
- **GPU:** 2456 MiB with NAG, 2072 MiB with SGD. **Duration:** 5–10 min. Run it alone on the GPU.

### R-48-3: CPU ablation, n16_train level 7, same optimizer as R-48-2
```
OPT=$(cat results/quadruped/logs/R-48-2.optimizer 2>/dev/null || echo nag)
OUT=results/quadruped/ablation_cpu_l7.json; [ "$OPT" = sgd ] && OUT=results/quadruped/ablation_cpu_l7_sgd.json
nice -n 10 target/release/ablation --backend cpu --params n16_train --top-level 7 --optimizer "$OPT" --iters 5 --warmup 1 --data $MNIST --out "$OUT" 2>&1 | tee results/quadruped/logs/R-48-3.log
echo "R-48-3 optimizer=$OPT exit=${PIPESTATUS[0]}" | tee -a results/quadruped/logs/R-48-3.log
```
- **Pass:** exit 0 and `all_checks_pass: true`.
- **Threads:** 12. **RAM:** ~5 GB. **Duration:** 15–25 min.

### R-48-4: CPU ablation, n16_train level 31 (the RQ3 CPU table, NAG)
```
nice -n 10 target/release/ablation --backend cpu --params n16_train --optimizer nag --iters 3 --warmup 1 --data $MNIST --out results/quadruped/ablation_cpu.json 2>&1 | tee results/quadruped/logs/R-48-4.log
echo "R-48-4 exit=${PIPESTATUS[0]}" | tee -a results/quadruped/logs/R-48-4.log
```
- **Pass:** exit 0 and `all_checks_pass: true`.
- **Threads:** 12. **RAM:** ~25 GB. **Duration:** ~60–90 min (estimate).
- **status:** queued, waiting (2026-11-06 10:49:44 JST; post-full-v1 controller PID1739838 / start ticks467041669). No clone/build/run started at this observation. Both successful FULL terminals plus chain10 terminal/inactive identity must validate before release. Exact pinned fresh clones and source/config/script/bundle hashes: `docs/post-full-queue.md`. This supersedes the 09:30 not-queued handover snapshot; it is not a measured result.

## R-52-1 / R-52-2: Task 52 encrypted block-MLP subset runs (eMLP-2, eMLP-3), ReBoot's first 21 steps (1008 samples), seed 0, B=48

- **purpose:** check the encrypted block MLP's KS against its closed form (every update) and its accuracy against the twin, on ReBoot's own subset. Full-length runs are not queued; they are on the user's decision list.
- **working dir:** fresh `/home/kmihara/hekit-post-full-clones/hekit-post-full-r52`; source bundle checkout is immutable before run outputs.
- **commit to sync:** exactly 9305d2e1f4f9999296cd4b4aaf55ea631fbca78e (fresh post-full clone). This needs the R-REB-2 data arrays in `results/quadruped/data/reboot_emlp1_full/`, as for FULL-1/2.
- **build:** `cargo build --release -p hekit-bench --features parallel --bins`
- **R-52-1 (eMLP-2):**
  ```
  RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist --mode noninteractive --placement when_needed --batch 48 --faithful --blocks 64 --hidden 32 --dataset-npy results/quadruped/data/reboot_emlp1_full --indices results/reboot_emlp2_seed0_steps1-21_indices.txt --init-weights results/quadruped/raw/reboot_emlp23_full/init_weights_emlp2_seed0 --init-weights-manifest results/quadruped/raw/reboot_emlp23_full/config_values.json --epochs 1 --transpose reuse --out results/quadruped/mnist196_emlp2_ni_b48_subset1008_seed0.json
  ```
  - **expected:** 71.35 KS/sample, about 1.7 h, about 27 GB RAM
- **R-52-2 (eMLP-3):** same command, with `--blocks 128,64`, `--indices results/reboot_emlp3_seed0_steps1-21_indices.txt`, `--init-weights results/quadruped/raw/reboot_emlp23_full/init_weights_emlp3_seed0`, and `--out results/quadruped/mnist196_emlp3_ni_b48_subset1008_seed0.json`
  - **expected:** 135.46 KS/sample, about 3.2 h, about 31 GB RAM
- **pass rule:** exit 0 and `all_checks_pass: true` in the output JSON. That means all of:
  - `checks.every_update_equals_prediction`
  - `checks.twin_weights_within_2^-10`
  - `checks.test_acc_diff_within_0.005`
  - `checks.handoff_err_within_2^-10`

  Also report `measured.measured_over_predicted` (expect 1.0), and for each boundary `twin.handoff_bootstraps[*].{max_abs_input_twin, inputs_above_1, log2_max_abs_err_vs_twin}`. On the 21-step subset the plaintext twin reaches max |a| ≈ 1.19 (eMLP-2) and 0.97 / 0.50 (eMLP-3). So `precondition_hit` should be true for eMLP-2 and false for eMLP-3; neither result is an anomaly. The ≈ 2.3 maximum appears only in full-length runs.
- **GPU memory need:** 0
- **timing:** timed CPU runs. Run them alone, after R-41-FULL-2 and before R-47-3 / R-48, or after those; nothing of ours may overlap. Partial rule: after 5 updates, log the projected total; if it is more than 2× the estimate, stop and mark `partial`.
- **output path:** as given in `--out`, plus `results/quadruped/logs/R-52-{1,2}.log`
- **status:** queued, waiting (2026-11-06 10:49:44 JST; post-full-v1 controller PID1739838 / start ticks467041669). No clone/build/run started at this observation. Both successful FULL terminals plus chain10 terminal/inactive identity must validate before release. Exact pinned fresh clones and source/config/script/bundle hashes: `docs/post-full-queue.md`. This supersedes the 09:30 not-queued handover snapshot; it is not a measured result.

## R-54-1: Task 54 one-shot runner, `scripts/run_all.sh --quick` (pipeline check, <= 2 h target)

Current scope: **not queued** and excluded from post-full-v1. The reviewed subset entry does not complete Task54's original full experiment matrix; that implementation remains incomplete while its extension design is prepared.

- **purpose:** run every group of the paper's experiment matrix once on subsets (1-4 updates, 1 seed): NI eMLP-1 at 196 and 784 inputs (WN / AU / AA, B = 32 / 48 / 64), interactive CPU, GPU and CPU at level 7, the RQ3 ablation (GPU L7, CPU L7 default config, CPU insecure `small` smoke), eMLP-2/3 NI, plaintext twins, Task 51 datasets, Iris (baseline a), ReBoot layout in hekit (baseline c). It checks that every row runs end to end with provenance and measures the real quick duration. Quick numbers are not paper numbers. Official ReBoot (baseline b) is not part of it.
- **commit to sync:** 962b7d2 or later (clean clone, e.g. `~/hekit-r54`; `git status --porcelain --untracked-files=no` must print nothing, the script refuses a dirty tree)
- **data:** copy in `results/quadruped/data/reboot_emlp1_full/*.npy` (as for R-41-FULL; sha256 checked against data_manifest.json by the binary), and `data/mnist`, `data/fashion_mnist` (`scripts/fetch_mnist.sh`, `scripts/fetch_fashion_mnist.sh`, or copy from `~/hekit-r41full/data`). A run whose inputs are missing is reported `missing-input`, not run.
- **command (bash):**
  ```
  cd ~/hekit-r54
  export LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH}; export PATH=/usr/local/cuda-13.1/bin:$PATH
  mkdir -p results/quadruped/logs
  scripts/run_all.sh --quick --dry-run > results/quadruped/logs/R-54-1.plan.txt
  /usr/bin/time -v scripts/run_all.sh --quick 2>&1 | tee results/quadruped/logs/R-54-1.log
  echo "R-54-1 exit=${PIPESTATUS[0]}" | tee -a results/quadruped/logs/R-54-1.log
  ```
  The script builds first (`cargo build --release -p hekit-bench --features parallel --bins`, then the CUDA build into `target/run_all-cuda`), then runs 24 jobs one after another, each under `nice -n 10` with `RAYON_NUM_THREADS=12`.
- **expected duration:** 1.71 h by the dry-run estimate (builds 0.2 h). Do not stop it for time below 4 h; above 4 h stop it, mark `partial`, and keep the outputs (the meta files say which runs finished).
- **GPU memory need:** at most about 2.4 GiB (GPU ablation at level 7 with NAG, 2456 MiB; mnist interactive at level 7). A GPU run that does not fit exits 3 with a `SKIP` line and is recorded `skipped-gpu-memory`; that is not a failure.
- **RAM need:** about 31 GB peak (one NI run at a time, 28-31 GB). **Disk:** about 3 GB (second target dir for the CUDA build, small outputs); check `df -h ~` first.
- **timing:** timed CPU and GPU jobs. Run it alone, after R-41-FULL-2 and the R-48 blocks; never during a FULL run.
- **pass:** exit 0 (every run `done` or `skipped-gpu-memory`). Exit 1 means a run failed or had missing inputs: list those ids with the last lines of `results/run_all/quick/logs/<id>.log`.
- **output path:** `results/run_all/quick/` (copy back `*.json`, `logs/`, `meta/`), `results/quadruped/logs/R-54-1.log`, `results/quadruped/logs/R-54-1.plan.txt`. Record total wall-clock (from `time -v`), `df -h ~`, `nvidia-smi` and `uptime` before and after; the controller puts the duration and `df -h` into `results/README.md`.
- **status:** not queued. The remote-runner session stopped on 2026-11-06 (handover) without starting this block; no chain exists for it on the server. See docs/quadruped-log.md, "引き継ぎ: 回収手順".


## R-REB-6-OBS: current original-producer environment inventory (read-only)

- **purpose:** diagnose Mac eMLP2 seed0 image parity failure without changing preprocessing, canonical data or the running FULL jobs. Current environment evidence is not retrospective attestation.
- **commit to sync:** none; no tree sync/build/install. Exporter source under investigation: `f0216160d8a74a644a12a227fb2ec915badc46b5`.
- **working dir:** read only `/home/kmihara/reboot` and the authoritative FULL-2 log.
- **command:** key-only SSH, `nice -n 10 python3 -`, with original stdlib inventory source supplied through stdin. Source SHA256 `a32f9ab9f498e6386eb53a4b167ab37adc516e1a0886f0e894cb326ed3f26e84`; exact retained source `raw/R-REB-6-OBS/observer.py`. No package/native/baseline import, training, CUDA, scientific capture or network inside observer. Large native files are stat-only, not hash-attested.
- **expected duration:** under 30 seconds; 1 process/thread, no background experiment.
- **GPU memory need:** 0. **RAM need:** under 100 MiB expected.
- **output path:** `results/quadruped/raw/R-REB-6-OBS/` (byte-exact inventory, observer, stderr, controller metadata, hashes and limitations); controller ledger retains originals. No accepted scientific result JSON.
- **status:** done (2026-11-07T04:32:19.620754+09:00, exit0; controller wall 40.113s including SSH). No scientific imports or job/queue mutation. Read-only untimed observation overlapped FULL-2. Raw README explicitly excludes the guessed final-result/config paths from evidence; this is current environment inventory only, not compatibility success or historical attestation.


## R-REB-7-STATIC: Linux exporter static prerequisites and FULL-2 observation

- **purpose:** observe actual NumPy SVE source/possible import edges, correct configuration paths and startup metadata before any Linux scientific import. Also read exact own chain/queue process identities and authoritative FULL-2 log/result path.
- **commit to sync:** none; no sync/build/install. Unchanged exporter reference f0216160d8a74a644a12a227fb2ec915badc46b5.
- **working dir:** read-only /home/kmihara/reboot and /home/kmihara/hekit-runs.
- **command:** key-only SSH, `nice -n 10 python3 -`, original stdlib observer through stdin; source SHA256 424a2315a1c2110542a9164f41d141290013118f08213692fe31748a83f20084. Source retained in controller ledger `task-54-linux-static-prerequisites-20261107.py`. No third-party source excerpts exported, scientific/native imports, full package/native hashing, data mutation or timed workload.
- **expected duration:** bounded observation under 2 minutes, one process/thread; a timeout is observation failure, not job termination.
- **GPU memory need:** 0. **RAM need:** under 100 MiB expected.
- **output path:** controller ledger `task-54-linux-static-prerequisites-20261107.{stdout,stderr,meta.json}`; later reviewed publication may retain exact raw copies.
- **pass criterion:** actual path/hash/static facts with limitations; neither import success nor compatibility. Lightweight untimed read-only observation may overlap FULL-2; full context hashing/capture remain deferred.
- **status:** done (2026-11-07T04:58:18.287347+09:00, exit0; controller wall 64.614 s including SSH). NumPy optional-query source matches Mac; unconditional module-level SVE query found. Scientific import reachability and target startup remain unverified. Chain and queue exact PID/start/own UID match live state S; FULL-2 log40/1000, authoritative result absent, queue owner.json only. No job mutation or scientific capture.


## R-REB-8-ARCHIVE: preserve mismatching original-producer config bytes

- **purpose:** compare the observed remote config hash9d6a2148… with the existing tracked/exporter-pinned artifact76c493e2… without changing either. Diagnose path/provenance before selecting the Linux comparison archive.
- **commit to sync:** none; no build/install/sync.
- **working dir:** read only `/home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full/`.
- **command:** key-only SSH `nice -n 10 cat /home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full/config_values.json`; exact stdout saved unmodified in the controller ledger.
- **expected duration:** under 2 minutes, one process; 191369 bytes expected. No scientific import, native execution or job mutation.
- **GPU memory need:** 0. **RAM need:** streaming file only.
- **output path:** ledger `task-54-linux-original-archive-20261107.{json,stderr,meta.json}` and separate local comparison report.
- **pass criterion:** hash equality to observed source or explicit drift, then exact local JSON-field comparison. No pin changes, normalization or replacement of historical results.
- **status:** done (exit0; controller 4.476 s including SSH). Original191369 bytes retained unmodified, SHA2569d6a21484856b3a5761f64fa139c1da226531d31e901eec2f5d72493ad0ce444 matches prior observation. Parsed field comparison is recorded in `raw/R-REB-8-ARCHIVE/`: only local structure_summary is additional; shared values including F64 bits match. Original file/pin remains unchanged; future comparison archive must be the verified pinned tracked copy.


## R-REB-9-BOOTSTRAP-STATIC: fixed startup/import-source prerequisites

- **purpose:** inspect actual executable .pth import targets, referenced startup-module facts, base site.py and selected known NumPy import-chain source identities before starting the target interpreter.
- **commit to sync:** none; no build/install/sync.
- **working dir:** read-only /home/kmihara/reboot and resolved CPython base library.
- **command:** key-only SSH `nice -n 10 python3 -`; original stdlib-only observer via stdin, SHA256 413e60c53b8753beb21cb2cae7c4ba00c15f6a22e12af5b6fc3a8fabd35f506f; ledger `task-54-linux-bootstrap-static-20261107.py`. Selected sources parsed as AST, never imported. Raw stdout also contained dependency string-literal receivers; publish a labelled sanitized derivative while retaining original stdout in the ledger.
- **expected duration:** under 2 minutes; one process/thread, fixed small source list.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** controller ledger `task-54-linux-bootstrap-static-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual source/hash/startup facts with explicit limitations, not dynamic reachability, target-startup approval or full consumer audit. No timed job/capture/queue mutation.
- **status:** done (2026-11-07T05:12:43.047940+09:00, exit0; controller8.685s including SSH). Actual _virtualenv.pth unconditionally imports a module under the governed site-packages root; its source installs a meta-path finder. Distutils .pth has an environment-dependent hook. Target interpreter was not started; static facts do not approve startup or full import reachability.


## R-REB-10-CREATOR-STATIC: actual base creator prerequisites

- **purpose:** resolve R9's missing actual base-site hook identity and conditional behavior; inspect base CPython executable, site/venv sources and applicable fixed customization candidates before proposing an exact creator command.
- **commit to sync:** none; no build/install/sync.
- **working dir:** read-only resolved CPython base and metadata for the absent proposed external venv path.
- **command:** key-only SSH `nice -n 10 python3 -`; original stdlib observer via stdin, SHA256 d4e09c78e2702251b70577a47c49370326a3b39c8abd94c107a032beaa101b25; ledger `task-54-linux-creator-static-20261107.py`. Only symbolic AST facts and short token literals are exported; no baseline source is read.
- **expected duration:** under 2 minutes; one process/thread, bounded source list and one interpreter file hash.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** controller ledger `task-54-linux-creator-static-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual base identities/conditions and explicit remaining startup/runtime limitations. No target-interpreter startup, environment creation, package-wide/native hashing, scientific imports, capture or job/queue mutation. Lightweight read-only untimed prerequisite observation may overlap FULL-2.
- **status:** done (2026-11-07T05:45:00.614568+09:00, exit0; controller54.011s including SSH). Actual base executable/hook/site/venv identities recorded; default-local distutils condition and shim insertion observed statically. Fixed customization/zip candidates and proposed destination absent. No target startup or creation; base pyvenv.cfg candidates and fresh precreation checks remain prerequisites.


## R-REB-11-CREATE-EMPTY-VENV: isolated empty creator-only phase

- **purpose:** create a separate stdlib venv using the observed original base interpreter, retaining the original scientific environment unchanged; static postcheck only.
- **commit to sync:** none; no repository sync/build/package install.
- **working dir:** creator `/home/kmihara/reboot`; new absent destination `/home/kmihara/reboot/exporter-venv-task54-v1` only.
- **command:** key-only SSH `nice -n 10 python3 -`, reviewed operational wrapper through stdin, SHA256 bac07100a6c8740e6df36ec99eb0d3463bf16a77d9d809006ca39c4105ed0873. Wrapper gates exact base Python `-I -B -m venv --without-pip --symlinks` with frozen replacement environment, all six input hashes, prefix/customization/.pth/owner/destination prechecks. Source/phase release and independent FIX1 review retained in controller ledger.
- **expected duration:** seconds, one creator process/thread; child timeout90s plus bounded5s cleanup/collection. No pip, ensurepip, upgrades or scientific work. The waiting controller is not another computational workload; lightweight preparation may overlap FULL-2.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** ledger `task-54-linux-create-empty-venv-attempt1.{stdout,stderr,meta.json}`; created destination retained even on failure.
- **pass criterion:** all fresh preflight and exact child exit/stream/15-entry static layout/config/link/empty-site checks; independently inspect any warning. New interpreter startup/context/native/scientific compatibility are explicitly not tested. No retry or automatic partial cleanup, no other process/queue mutation.
- **status:** done (2026-11-07T06:08:36.727462+09:00–2026-11-07T06:08:36.810504+09:00, wrapper/child exit0). Exact fresh preflight and15-entry static postchecks passed, empty own site and pinned-base interpreter links retained; config396bytes SHA2564af48d1b0d9fea3094d3c725918c55e52670536de714c30660f3fb37ec13c643. Child stdout/stderr empty, terminal status and complete streams observed. Creator-only; new environment interpreter was not started and no context/capture/scientific compatibility was tested.


## R-REB-12-LAUNCH-HELPERS: fixed Linux stdlib helper source facts

- **purpose:** observe the actual pinned Linux CPython helper source identities and import atoms named by the reviewed launcher API, before any new-venv startup/context/capture.
- **commit to sync:** none; no build/install/sync.
- **working dir:** read-only exact base `/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu` and fixed34 sources under its `lib/python3.11`.
- **command:** key-only SSH `nice -n 10 python3 -`; original fixed-list observer through stdin, SHA256 9ee70f1169b9bf02390c0777ac164ee6fb48bfc95f00531325d301ff8f5693a9; ledger `task-54-linux-helper-static-20261107.py`. No target interpreter or source is executed.
- **expected duration:** seconds after SSH; one observer process/thread, fixed34 source paths, per-source512KiB and aggregate4MiB caps, one pinned base executable hash. Lightweight read-only untimed preparation may overlap FULL-2.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** controller ledger `task-54-linux-helper-static-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual file/hash/import observations and explicit remaining helper-closure/startup limitations. No transitive-closure, startup-cache or runtime claim; no scientific imports/full native or package hashing/context/capture/job or queue mutation.
- **status:** done (2026-11-07T06:33:04.897637+09:00–2026-11-07T06:33:05.009331+09:00, exit0; controller14.689s including SSH). All34 fixed files observed,656069source bytes; base executable pin matched. Raw operational evidence is archived in `raw/R-REB-12-LAUNCH-HELPERS/`. This is an import-atom/source-identity observation only, not complete helper closure or target startup/context/capture approval.


## R-REB-13-LAUNCH-EXTRA-STATIC: venv root mode and typing source identity

- **purpose:** observe missing root-mode and typing.py identity prerequisites identified by bounded launcher review; no mode change or target startup.
- **commit to sync:** none; no build/install/sync.
- **working dir:** read-only exact new-venv root metadata and original base stdlib typing.py.
- **command:** key-only SSH `nice -n 10 python3 -`; fixed original observer SHA256 cc9eb2a387cd40a33ba9f3ce72093d942d5c4beb0d7773726b7ba9c3efff2f18, ledger `task-54-linux-launch-extra-static-20261107.py`; source syntax parsed locally only.
- **expected duration:** seconds; one observer process/thread, one source file cap512KiB. Lightweight read-only untimed preparation may overlap FULL-2.
- **GPU memory need:** 0. **RAM need:** under50MiB expected.
- **output path:** ledger `task-54-linux-launch-extra-static-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual root owner/mode metadata and exact typing source hash; no semantic/closure/startup/context/scientific claim or job/queue change.
- **status:** done (2026-11-07T06:47:25.938039+09:00, exit0; controller66.311s including SSH). Actual root uid1000/mode0775 recorded; typing.py120951bytes SHA256eff3df4f1d65e13d690b8f427cada4dd3b88431dd1e7f319fda1bd0062113203, uid1000/mode0664. This metadata/hash-only observation does not establish source semantics, helper closure, new-venv startup or later-phase readiness. Raw operational evidence: `raw/R-REB-13-LAUNCH-EXTRA-STATIC/`.


## R-REB-14-STARTUP-PREREQUISITES: bounded startup selectors and platform effects

- **purpose:** obtain fixed deployment/controller/package-root metadata and actual platform warmup call facts before selecting one startup-only release.
- **commit to sync:** none; no build/install/sync/deployment.
- **working dir:** read-only fixed candidates under own hekit-runs/reboot, pinned base platform.py, observer interpreter, and two explicit uname command candidates.
- **command:** key-only SSH `nice -n 10 python3 -`; original observer SHA256 004bf8a268791c57a677ea4785c2b5f70cd5325c41d363d579b9b76e0233d613, ledger `task-54-linux-startup-prerequisite-20261107.py`. Source parsed locally; target platform source only parsed as data.
- **expected duration:** seconds after SSH; one observer, platform source cap512KiB, observer executable cap64MiB, two candidate command files cap2MiB each, AST output cap512KiB. Untimed lightweight read-only preparation may overlap FULL-2.
- **GPU memory need:** 0. **RAM need:** under150MiB expected.
- **output path:** ledger `task-54-linux-startup-prerequisite-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** fixed metadata/source pin and derived call facts; unknown or absent values retained honestly. No target startup, command-candidate execution, package scan, scientific import, context/capture, deployment or job/queue mutation. Controller/bootstrap remains ordinary trusted local Python; hashes are observations, not self-issued runtime approval.
- **status:** failed (controller exit1, 2026-11-07T07:10:45+09:00); command-candidate read rejected its existing2MiB kind/size gate. Stdout empty, traceback retained; no candidate values inferred and no target command executed. Original source/streams/controller retained; corrected read-only observation follows separately.


## R-REB-14-FIX1: preserve bounded command candidate observations

- **purpose:** repeat the fixed read-only prerequisite observation after a terminal observer error; retain metadata for an oversized/nonregular command candidate without increasing the2MiB read cap, and retain partial evidence on later error.
- **commit to sync:** none; no install/build/sync/deployment.
- **working dir:** same fixed candidates and pinned platform.py as R-REB-14.
- **command:** key-only SSH `nice -n 10 python3 -`; original corrected observer SHA256 4841f1c1eaf4126919ea467037573c87da3c67380cfab5755c46b024a0abb9f7, ledger `task-54-linux-startup-prerequisite-fix1-20261107.py`; syntax parsed locally.
- **expected duration:** seconds after SSH, one observer; all source/executable/output caps unchanged.
- **GPU memory need:** 0. **RAM need:** under150MiB expected.
- **output path:** ledger `task-54-linux-startup-prerequisite-fix1-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual fixed metadata/derived source facts; an unavailable candidate hash remains unavailable, not silently approved. No target startup, command execution, scientific imports, package scan, deployment or job/queue change.
- **status:** done (2026-11-07T07:13:00.956467+09:00, exit0). Fixed deployment directories absent; own cwd/package root present; fictional warmup path absent. Actual platform AST call uses uname with -p; its candidate is an11352352byte root-owned coreutils executable exceeding the unchanged2MiB hash cap, so hash remains unmeasured. System observer interpreter identified separately. No command/target startup or deployment.


## R-REB-15-STARTUP-COMMAND-IDENTITY: exact known executable and host observations

- **purpose:** resolve R14's known11.35MB command identity with a justified16MiB bounded read; refresh known controller/base pins and ordinary host uname/libc metadata.
- **commit to sync:** none; no install/build/sync.
- **working dir:** exactly3 canonical executable paths from R14/R11; no directory scan.
- **command:** key-only SSH `nice -n 10 python3 -`; observer SHA256 d4c6b428b88bf60d1899b4716fc957cd9cf6b95c3a6a80463b0840b02dfbb106, ledger `task-54-linux-startup-command-identity-20261107.py`. Reads the three fixed executable files as bytes only; executes no target command or interpreter.
- **expected duration:** seconds after SSH, one observer;16MiB cap each for coreutils and system Python,32MiB for known21.74MB base executable. No full native/package hashing.
- **GPU memory need:** 0. **RAM need:** under150MiB expected.
- **output path:** ledger `task-54-linux-startup-command-identity-20261107.{stdout,stderr,meta.json}`.
- **pass criterion:** actual file identities and matching known pins, with host metadata explicitly observed by system observer. Not new-venv runtime/native isolation or scientific evidence. No command execution, deployment or job/queue mutation.
- **status:** done (2026-11-07T07:15:09.860511+09:00, exit0). Exact coreutils uname11352352bytes SHA25648893b0fb21436b54619db80486e83ef39dfccaf1aefe83dfa00c02d6146e8c0; system/base pins unchanged. System observer reports Linuxx86_64 and glibc2.43. No target startup or command execution; records archived with both R14 attempts in `raw/R-REB-14-15-STARTUP-PREREQUISITES/`.


## R-REB-16-STARTUP-FILES: fresh four-file deployment without interpreter startup

- **purpose:** deploy exact reviewed prelaunch/launcher/exporter copies and observation-only candidate sidecar into a new own directory.
- **commit to sync:** none; no checkout/build/package/environment mutation.
- **working dir:** fresh `/home/kmihara/hekit-runs/task54-launch/v1`; parent identity from R14 freshly checked.
- **command:** key-only SSH `nice -n 10 python3 -`; original deployment wrapper SHA256 d63de1ddda59b69fb183ad9507d0a7cdcda7ae28d4acd03d2550f3d9c0d38bca, ledger `task-54-linux-startup1-deploy.py`. Independent bounded spec/quality review5cb1f12bbd12b18783c585e3ef7330ac7044d2046bc9ab9fce219aa2f0335eb9 accepted.
- **expected duration:** seconds; one process,111278payload bytes, fixed4files/2newdirectories. No target process launch.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** ledger `task-54-linux-startup1-deploy.{stdout,stderr,meta.json}`, exact remote fresh files underv1.
- **pass criterion:** fresh-only O_EXCL writes, owner/mode/hash/readback/fsync and terminal receipt; preserve partial state/no retry or cleanup on failure. This installs inert copies only, does not start the new venv, append packages, load exporter, run context/capture or mutate FULL2/queue.
- **status:** done (2026-11-07T07:22:46.625793+09:00, exit0). Four exact files were written fresh, fsynced and hash/uid/mode0600 verified. No target process started. The existing deployment must not be recreated/retried; independent readback and separate startup invocation follow.


## R-REB-17-STARTUP-READBACK: independent deployed-byte and live-job observation

- **purpose:** fresh readback of R16's4files/2directories and absent startup evidence paths, alongside the existing read-only FULL2/queue observer before any startup release.
- **commit to sync:** none; no build/install/sync.
- **working dir:** exact own deployment and existing FULL2 log/result/ownedprocess/queue paths.
- **command:** key-only SSH `nice -n 10 python3 -`; observer SHA256 47af9533e23db875c392979ea91ab4079001734dd15ba95fbc8e3f6d4a850426, ledger `task-54-linux-startup1-readback.py`. Existing job observer semantics retained; only four cap1MiB readbacks and two directory metadata checks added.
- **expected duration:** seconds after SSH; one observer, no scientific workload.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** ledger `task-54-linux-startup1-readback.{stdout,stderr,meta.json}`.
- **pass criterion:** actual independent byte/hash/owner/mode/inode agreement and honest current ownedjob state. No process signaling, target startup, package scan, native/scientific imports, context/capture or mutation.
- **status:** done (2026-11-07T07:25:40.070290+09:00, exit0). All4deployed file bytes/hash/owner/mode/inode matched and2directories are own0700; startup evidence paths absent. FULL2 update101 zero-based (102/1000 complete), result absent, own chain+queue alive with matching start ticks; queue owner.json only. No process/queue changes.


## R-REB-18-STARTUP1: one trusted-only new-venv startup observation

- **purpose:** observe actual isolated venv identity and trusted stdlib helper preparation, then stop before external package append/exporter loading.
- **commit to sync:** none; frozen deployed copies only.
- **working dir:** controller/child `/home/kmihara/reboot`, exact deployed sidecar `/home/kmihara/hekit-runs/task54-launch/v1/startup1-release.json`.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`; original external verifier SHA25694563d06cc7cfe48bfc6ed3e19a9a6f4404418917d606044e986333bb7ea0c44 through stdin. It pins inputs then execs exact base3.11.16 `-I -B` prelaunch with external sidecar SHA256b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a. Prelaunch owns lexical new-venv `bin/python -I -B` child and the same sidecar transport.
- **expected duration:** seconds; child60s timeout plus5s bounded collection, outer SSH observation110s. One-thread environment caps, nice10 inherited, no HE/GPU or scientific imports. Lightweight untimed setup may overlap FULL2; no context/full package-native hash/capture in this phase.
- **GPU memory need:** 0. **RAM need:** under150MiB expected.
- **output path:** remote exact `startup1-parent.json` / `startup1-child.json` beside sidecar, no-clobber; controller ledger `task-54-linux-startup1-invoke.{stdout,stderr,meta.json}`.
- **pass criterion:** exact fresh static/identity/site/path/module/hook checks, matching parent/child release binding, actual terminal zero and complete streams. Pre-exec verification alone is not success. Platform warmup may execute the identified uname -p and ordinary trusted stdlib fallbacks; reachability is source inference, not a native allowlist. Direct-child timeout does not contain descendants; failure/unresolved streams are retained, no retry/guard patch or process-tree signaling.
- **status:** failed (2026-11-07T07:27:42.013940+09:00 child terminalexit1, complete streams/no cleanup errors). Static preflight passed; child rejected importer-cache key equal to its exact pinned launcher file path before recording startup stage. ChildPID1958934, no helper preparation/external package append/exporter/context/capture reached. Preserve records; no automatic retry. The cache entry value was not recorded, so its being None is not claimed as observed.


## R-REB-19-STARTUP1-EVIDENCE: collect exact failed-startup bytes

- **purpose:** preserve exact parent/child evidence from terminal R18 without rerunning it.
- **commit to sync:** none.
- **working dir:** exactly two known startup1 evidence files under own deployment.
- **command:** key-only SSH `nice -n 10 python3 -`; collector SHA2561f7668159f31d1b5df40b22da9de11f0499d13ab45f332bdb13d47d77d84b4b7, ledger `task-54-linux-startup1-collect.py`.
- **expected duration:** seconds, one observer;2MiB cap per regular own file, missing files explicit.
- **GPU memory need:** 0. **RAM need:** under50MiB expected.
- **output path:** ledger `task-54-linux-startup1-collect.{stdout,stderr,meta.json}` plus decoded byte-identical parent/child JSON and hashes.
- **pass criterion:** exact available bytes/identities retained, distinct from startup verdict. No target process invocation/retry, package/scientific import, mutation or signals.
- **status:** done (2026-11-07T07:29:37.018856+09:00, exit0). Both remote evidence files collected byte-for-byte and decoded locally with matching hashes; failed startup release binding and childterminal1/complete streams/empty stages confirmed. No retry or target invocation.


## R-REB-20-STARTUP2-FILES: fixed startup2 operation

- **purpose:** Deploy four fresh fixed startup2 bytes; preserve v1/startup1.
- **commit to sync:** none; isolated frozen copies, no checkout/build/install.
- **working dir:** `/home/kmihara/hekit-runs/task54-launch/v2`; invocation cwd `/home/kmihara/reboot`.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`; exact ledger `task-54-linux-startup2-deploy.py`, SHA256 ab1c616bea7230961ffcced5c8706c0b308bf75fe38b6feadc807044e948eccf on stdin. No password/sudo/other-process signals.
- **expected duration:** seconds after SSH; invocation child60s/cleanup5s, outer110s observation deadline. One-thread preparation only, no HE/scientific workload.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** ledger `task-54-linux-startup2-deploy.{stdout,stderr,meta.json}`; invocation creates fresh remote startup2-parent.json / startup2-child.json.
- **pass criterion:** Fresh-only absent v2, four exact hashes/uid/mode/fsync; no target startup. Preserve partial/failed records; no retry or replacement of v1/startup1. No job/queue mutation or native-isolation claim.
- **status:** done (2026-11-07T07:52:08.542554+09:00, exit0). Four exact files deployed exclusively under freshv2, hashes/uid/mode0600 verified; v1 unchanged. No target started.


## R-REB-21-STARTUP2-READBACK: fixed startup2 operation

- **purpose:** Independently rehash four deployed files and observe directory/evidence state.
- **commit to sync:** none; isolated frozen copies, no checkout/build/install.
- **working dir:** `/home/kmihara/hekit-runs/task54-launch/v2`; invocation cwd `/home/kmihara/reboot`.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`; exact ledger `task-54-linux-startup2-readback.py`, SHA256 6a45c62ca7196e771f0166e3afdae1878b9671683fa861009767e259387517a2 on stdin. No password/sudo/other-process signals.
- **expected duration:** seconds after SSH; invocation child60s/cleanup5s, outer110s observation deadline. One-thread preparation only, no HE/scientific workload.
- **GPU memory need:** 0. **RAM need:** under100MiB expected.
- **output path:** ledger `task-54-linux-startup2-readback.{stdout,stderr,meta.json}`; invocation creates fresh remote startup2-parent.json / startup2-child.json.
- **pass criterion:** Four file identities must match deployment receipt; own0700 directories and absent evidence. No FULL2 poll or target start. Preserve partial/failed records; no retry or replacement of v1/startup1. No job/queue mutation or native-isolation claim.
- **status:** done (2026-11-07T07:53:12.706105+09:00, exit0). Four complete file identity records match R20; commonparent matches R17, v2directory identity first observed66311/36309263 own0700; both evidencepaths absent. No job poll or mutation.


## R-REB-22-STARTUP2: fixed startup2 operation

- **purpose:** One FIX3 trusted-only startup observation after deployment/readback.
- **commit to sync:** none; isolated frozen copies, no checkout/build/install.
- **working dir:** `/home/kmihara/hekit-runs/task54-launch/v2`; invocation cwd `/home/kmihara/reboot`.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`; exact ledger `task-54-linux-startup2-invoke.py`, SHA256 8c313e0716ceb221077ae03491ef9ace8a676b87ea754214b046866013cb3ffc on stdin. No password/sudo/other-process signals.
- **expected duration:** seconds after SSH; invocation child60s/cleanup5s, outer110s observation deadline. One-thread preparation only, no HE/scientific workload.
- **GPU memory need:** 0. **RAM need:** under150MiB expected.
- **output path:** ledger `task-54-linux-startup2-invoke.{stdout,stderr,meta.json}`; invocation creates fresh remote startup2-parent.json / startup2-child.json.
- **pass criterion:** Exact sidecar ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd; terminalzero, complete streams, real stage evidence. No externalappend/exporter/context/capture. Preserve partial/failed records; no retry or replacement of v1/startup1. No job/queue mutation or native-isolation claim.
- **status:** done (2026-11-07T07:53:48.768885+09:00, childterminal0). ChildPID1963722, complete streams/cleanupempty. Parent/child release and argv match; actual startup/helpers snapshots observed, observation-only. No externalappend/exporter/context/capture. Originalstartup1failure unchanged.


## R-REB-23-STARTUP2-EVIDENCE: fixed startup2 operation

- **purpose:** Collect exact parent/child evidence without rerunning startup.
- **commit to sync:** none; isolated frozen copies, no checkout/build/install.
- **working dir:** `/home/kmihara/hekit-runs/task54-launch/v2`; invocation cwd `/home/kmihara/reboot`.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`; exact ledger `task-54-linux-startup2-collect.py`, SHA256 e52ff869ecd2d84f395f25f8459df7d3a010e24b51dcec3c1f1f1add6909f08b on stdin. No password/sudo/other-process signals.
- **expected duration:** seconds after SSH; invocation child60s/cleanup5s, outer110s observation deadline. One-thread preparation only, no HE/scientific workload.
- **GPU memory need:** 0. **RAM need:** under50MiB expected.
- **output path:** ledger `task-54-linux-startup2-collect.{stdout,stderr,meta.json}`; invocation creates fresh remote startup2-parent.json / startup2-child.json.
- **pass criterion:** Both fixed files at most2MiB each with original bytes/hash; missing/error explicit, no startup verdict inferred from collection. Preserve partial/failed records; no retry or replacement of v1/startup1. No job/queue mutation or native-isolation claim.
- **status:** done (exit0). Both original files collected byte-for-byte: parent2755B SHA2569d491997ed1476c98f5e1847354d30e344cb54bbb1f4ee2ecf480c16f71b5fc1; child98031B SHA256a1fbe98aa390b7305f7189dabf574c4d28b2422617bfeb010e261f2c82fd1f41. Parent childhash, terminal0/completestreams and startup-only stage membership verified. No retry.


## R-41-FULL2-OBS-20261107-0825: hourly read-only progress observation

- **purpose:** Check the existing FULL-2 chain and post-full queue without starting, stopping or changing either.
- **commit to sync:** none.
- **working dir:** existing own `/home/kmihara/hekit-runs`; fixed own PIDs1700979/start465769819 and1739838/start467041669.
- **command:** key-only SSH `nice -n 10 /usr/bin/python3.14 -I -B -`, unchanged ledger `post-full-observation-20261107-0600.remote.py` SHA25617550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243 supplied on stdin.
- **expected duration:** seconds after connection; single observer, bounded8KiB log tail. Connection timeout90s; local async controller retains stdout/stderr and terminal exit.
- **GPU memory need:** 0. **RAM need:** under50MiB expected.
- **output path:** ignored ledger `post-full-observation-20261107-0825.{stdout,stderr,meta.json}`.
- **pass criterion:** Actual process start/uid/state and direct-child observations plus log/result/queue state recorded. This is operational evidence, not a scientific result or final accuracy. No signals, restart, training, build, or other-user process inspection.
- **status:** done (2026-11-07T08:25:52.000152+09:00, SSH exit0). Same owned chain/queue live; update126 means127/1000 completed, last-six predictions match. Final result absent; queue owner.json only. Exact operational records: `raw/R-41-FULL2-OBS-20261107-0825/`. No job or queue mutation.
