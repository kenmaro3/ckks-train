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
