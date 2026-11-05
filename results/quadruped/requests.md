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
- **commit to sync:** bd5cc49 or later (clean clone; the example now times both pool thresholds, `[default]` 256 MiB and `[max]`)
- **working dir:** clean clone, e.g. `~/hekit-r45` re-synced
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

Run these blocks **after R-41-FULL-2 and R-47-3**, in the order R-48-1 → R-48-2 → R-48-3 → R-48-4, each one alone, and never during a FULL run.

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
