#!/bin/bash
# R-41-FULL-1 then R-41-FULL-2 (faithful full-length 196-input encrypted runs, clean clone ~/hekit-r41full >= ff3042d).
# Long timed runs: nothing else of ours is queued alongside. ~/hekit-runs/chain10.sh on quadruped-gpu.
set -u
R=$HOME/hekit-runs/run_job.sh
H=$HOME/hekit-r41full
export PATH=$HOME/.cargo/bin:$PATH SYNCED_COMMIT_FILE=$H/.synced-commit
cd $H
bash $R hekit_r41full_build $H 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
D=results/quadruped/raw/reboot_emlp1_full
COMMON="--mode noninteractive --placement when_needed --faithful --dataset-npy results/quadruped/data/reboot_emlp1_full --indices $D/train_indices.txt --init-weights $D/init_weights_seed0 --init-weights-manifest $D/config_values.json --lr-schedule $D/config_values.json --lr-key lr_per_step --epochs 1 --transpose reuse"
HEKIT_DIR=$H PARTIAL_AFTER=10 PARTIAL_STOP_AT=10 PARTIAL_LIMIT_S=108000 bash $HOME/hekit-runs/r41.sh R-41-FULL-1 \
  results/quadruped/mnist196_ni_when_needed_b64_full_seed0.json 750 -- $COMMON --batch 64 --lr-schedule-batch 48
HEKIT_DIR=$H PARTIAL_AFTER=10 PARTIAL_STOP_AT=10 PARTIAL_LIMIT_S=198000 bash $HOME/hekit-runs/r41.sh R-41-FULL-2 \
  results/quadruped/mnist196_ni_b48_full_seed0.json 1000 -- $COMMON --batch 48
echo CHAIN10_DONE
