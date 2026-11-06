# R-41-FULL-1: immutable server output

Retrieved after runner exit0 / JOB_DONE rc=0, watchdog_killed=false, no partial marker. The11 files in SHA256SUMS are byte-exact server outputs, not reconstructed results. Program JSON SHA256 c18c7e217c710342164aad6d255f82f78089a306e7d8c05f2fc7c2778cd68acf; raw source/provenance remains unchanged, including legacy machine.threads wording. Actual host/threads are in producer provenance and runner metadata: i9-13900KF/Linux,12 threads,nice10.

The binary reports196→32→16, N65536, WN, B64,750updates/48000samples, initial weights seed0; training.seed32 is encryption RNG. Binary epochs1 consumes flattened10-pass indices. train_total72050.794763016s covers all10 passes; runner wall72114.93636155128s is a distinct interval. Accuracy0.9036 versus same-B twin0.9037, one seed only. FULL-2 remains the specified accuracy gate against ReBoot released seed0=0.9041.

Observed server source HEAD ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f, tracked status clean. Current binary SHA256 at retrieval3409cffcbd50471067829a3fc597976d3cac2cea1f4ed26255e03070cc02d5e7 is a retrieval observation, not retrospective build attestation. All8 command input files matched local archived hashes listed below. Root raw fetch/terminal observations are retained in the task ledger. No result JSON was edited and no run was repeated.

Reproduce aggregation with `python3 -B tools/rq1_summary.py --full-1 results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json --full-2 results/quadruped/raw/R-41-FULL-2/mnist196_ni_b48_full_seed0.json --reboot-config results/quadruped/raw/reboot_emlp1_full/config_values.json`. `scripts/run_all.sh --tables-only` regenerates the paper tables without training.

## Input hashes observed on server and matched locally

- `results/quadruped/data/reboot_emlp1_full/test_images_14x14.npy`: `9464f6e1544cd65d7387036bab2e71e3826b121b108d2bb4f1360e9bfb0d6d87`
- `results/quadruped/data/reboot_emlp1_full/test_labels.npy`: `ff7e84b144c037e7215dfa787d6773550c5db83029d9a4e7bae6e90f605f081d`
- `results/quadruped/data/reboot_emlp1_full/train_images_14x14.npy`: `7b1590d8884172b853d8037063b290d20b41112f6c0aa5309769639552b9276c`
- `results/quadruped/data/reboot_emlp1_full/train_labels.npy`: `1d63491b1c54835583c5ca6886305d7b83d7c9ca0c3ab24433c431b1f7e84856`
- `results/quadruped/raw/reboot_emlp1_full/config_values.json`: `b89c9129177f5831cf044c775a2228fecace8214479aa54b1ad71b2d10a8ad18`
- `results/quadruped/raw/reboot_emlp1_full/init_weights_seed0/W1.npy`: `92ab56cd70584de0410b43f06f014d9a1f474d0f57282dfc2b3a23a3e445bc69`
- `results/quadruped/raw/reboot_emlp1_full/init_weights_seed0/W2.npy`: `e2e5192130aea10834790f00d401d6a4a86fe508f5c472ddeb847b7b12db86f8`
- `results/quadruped/raw/reboot_emlp1_full/train_indices.txt`: `6c73caeebcec203bce24ffafaed6e6ad8a573fba8e28abd168169e391370e853`
