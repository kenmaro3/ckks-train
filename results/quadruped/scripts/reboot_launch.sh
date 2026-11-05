#!/bin/bash
# Launch one ReBoot eMLP-1 run through run_job.sh. Lives at ~/reboot/reboot_launch.sh on quadruped-gpu.
# Usage: reboot_launch.sh NAME [training_encrypted.py args...]
# Same arguments as upstream experiments.sh / hekit spikes/reboot/run.sh (MNIST, 2 layers, 32 hidden, batch 48, seed 0,
# lr and weight decay at the script defaults), 12 threads (REBOOT_THREADS / OMP_NUM_THREADS).
set -eu
NAME=$1; shift
R=$HOME/reboot; EXP=$R/src/ReBoot/experiments/3_training_encrypted
OUT=$HOME/hekit-runs/out/$NAME
mkdir -p "$OUT" "$R/runs/$NAME"
rm -rf "$EXP/res"
bash $HOME/hekit-runs/run_job.sh "$NAME" "$R/runs/$NAME" 12 0 -- \
  env REBOOT_THREADS=12 REBOOT_SRC=$R/src/ReBoot REBOOT_STATS_OUT=$OUT/stats.json REBOOT_DATA=$R/data \
      LD_LIBRARY_PATH=$R/install/lib PYTHONUNBUFFERED=1 PYTHONPATH=$R/src/ReBoot \
      $R/venv/bin/python $R/reboot_wrapper.py MNIST "$@"
[ -d "$EXP/res" ] && mv "$EXP/res" "$OUT/res" || true
