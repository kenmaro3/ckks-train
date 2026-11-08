# RQ1 full-length evidence (Task 42 preparation)

Overall RQ1: pending. This summary cannot establish an unconditional pass.

Checks validate declared provenance and internal consistency. A recorded git HEAD does not attest the built binary; command paths do not prove input file contents.

ReBoot full CPU timing: unmeasured (no 784-input subset timing is reused).
Security comparability: unverified (requires matched estimator evidence and distributions).
ReBoot B=48, 4 bootstraps/update; nonboot KS/sample = 37 (model); bootstrap KS = 502 (model). hekit bootstrap KS = 111 (measured).
d is the measured NI count with bootstrap KS removed, not a separately executed interactive run.

ReBoot released config: values-only reference
- Input: `results/quadruped/raw/reboot_emlp1_full/config_values.json`; SHA256: `b89c9129177f5831cf044c775a2228fecace8214479aa54b1ad71b2d10a8ad18`
- seed-0 reference accuracy: 0.904100 (per_seed.0.twin.test_acc_argmax10)

## FULL-1: measured raw binary output
- Input: `results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json`; SHA256: `c18c7e217c710342164aad6d255f82f78089a306e7d8c05f2fc7c2778cd68acf`
- Writer: the mnist binary (raw output, not assembled); commit `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f`; tracked changes: `none`.
- Machine: kmihara-Z790-S01 / 13th Gen Intel(R) Core(TM) i9-13900KF / Linux 7.0.0-31-generic x86_64; threads: rayon (feature parallel), RAYON_NUM_THREADS = 12, 32 logical CPUs.
- Unix timestamps: 1793917076.4435072 → 1793989184.8170981.
- Command: `./target/release/mnist --mode noninteractive --placement when_needed --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full --indices results/quadruped/raw/reboot_emlp1_full/train_indices.txt --init-weights results/quadruped/raw/reboot_emlp1_full/init_weights_seed0 --init-weights-manifest results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-schedule results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-key lr_per_step --epochs 1 --transpose reuse --batch 64 --lr-schedule-batch 48 --out results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json`
- initial weights: seed 0 (command + released config)
- training.seed is the encryption RNG seed; it is not the initial-weight seed.
- 196 → 32 → 16; N=2^16; B=64; 750 updates; 48000 samples (4800 × 10 passes).
- binary epochs=1 traverses flattened indices; train_total covers all 10 passes.
- Training time: 72050.794763 s; 1.501058 s/sample; 7205.079476 s/pass (arithmetic mean).
- Measured integer counts: 877040 KS, 1500 bootstraps; 710540 nonboot KS; 2.000000 bootstraps/update.
- Schedule and every-update prediction checks: verified.

| Way | hekit KS/sample | ReBoot KS/sample (model) | Ratio | ≤0.70 KS gate |
|---|---:|---:|---:|---|
| a: own bootstrap costs | 18.271667 | 78.833333 | 0.231776 | PASS |
| b: both 502 (model) | 30.490417 | 78.833333 | 0.386771 | PASS |
| c: both 111 | 18.271667 | 46.250000 | 0.395063 | PASS |
| d: boot-excluded NI ablation | 14.802917 | 37.000000 | 0.400079 | PASS |

- FULL-1 accuracy (separate): 0.903600; twin: 0.903700; maximum weight/bias error: 0.0068265728.

## FULL-2: measured raw binary output
- Input: `results/quadruped/raw/R-41-FULL-2/mnist196_ni_b48_full_seed0.json`; SHA256: `921d5a7c315497ccc2f4d309b19edd6f5718013abeb38f0b63eb06da00f93a62`
- Writer: the mnist binary (raw output, not assembled); commit `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f`; tracked changes: `none`.
- Machine: kmihara-Z790-S01 / 13th Gen Intel(R) Core(TM) i9-13900KF / Linux 7.0.0-31-generic x86_64; threads: rayon (feature parallel), RAYON_NUM_THREADS = 12, 32 logical CPUs.
- Unix timestamps: 1793989198.9180489 → 1794135486.2308514.
- Command: `./target/release/mnist --mode noninteractive --placement when_needed --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full --indices results/quadruped/raw/reboot_emlp1_full/train_indices.txt --init-weights results/quadruped/raw/reboot_emlp1_full/init_weights_seed0 --init-weights-manifest results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-schedule results/quadruped/raw/reboot_emlp1_full/config_values.json --lr-key lr_per_step --epochs 1 --transpose reuse --batch 48 --out results/quadruped/mnist196_ni_b48_full_seed0.json`
- initial weights: seed 0 (command + released config)
- training.seed is the encryption RNG seed; it is not the initial-weight seed.
- 196 → 32 → 16; N=2^16; B=48; 1000 updates; 48000 samples (4800 × 10 passes).
- binary epochs=1 traverses flattened indices; train_total covers all 10 passes.
- Training time: 146229.853280 s; 3.046455 s/sample; 14622.985328 s/pass (arithmetic mean).
- Measured integer counts: 1391540 KS, 4000 bootstraps; 947540 nonboot KS; 4.000000 bootstraps/update.
- Schedule and every-update prediction checks: verified.

| Way | hekit KS/sample | ReBoot KS/sample (model) | Ratio | ≤0.70 KS gate |
|---|---:|---:|---:|---|
| a: own bootstrap costs | 28.990417 | 78.833333 | 0.367743 | PASS |
| b: both 502 (model) | 61.573750 | 78.833333 | 0.781062 | FAIL |
| c: both 111 | 28.990417 | 46.250000 | 0.626820 | PASS |
| d: boot-excluded NI ablation | 19.740417 | 37.000000 | 0.533525 | PASS |

- FULL-2 accuracy (measured): 0.903500; twin: 0.904100; maximum weight/bias error: 0.006625031.

FULL-2 accuracy gate: PASS; difference -0.060000 percentage points; tolerance ±0.5 percentage point.
