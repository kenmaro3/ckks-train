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
- **status:** queued: starts after the Iris run (epoch ~300/400 at 19:55 JST) and a parallel rebuild; expected start ~20:20 JST

M4 GPU blocks (Tasks 44–49) and Task 54 blocks will be appended here as that code lands. Each GPU
block will state its GPU memory need, and the tests skip with a message when the need is not met.

---

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

## R-41-F2: same as R-41-F1, B = 64

- Same as R-41-F1, with `--batch 64` and `--out results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`.
- **expected:** 17.89 KS/sample; 9 of 16 updates bootstrap; about 25–30 min. Plaintext twin accuracy is 0.7359.
- **output path:** `results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json`, `results/quadruped/logs/R-41-F2.log`

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
