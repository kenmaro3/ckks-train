#!/bin/bash
# Run ReBoot eMLP-1 on MNIST (ReBoot protocol: 4800-sample subsample, batch 48). Usage: run.sh <logname> [extra args]
set -e
EXT="$(cd "$(dirname "$0")" && pwd)/_ext"
LOG="$EXT/logs/$1.log"; shift
. "$EXT/venv/bin/activate"
export PYTHONUNBUFFERED=1 REBOOT_DATA="$EXT/data" OMP_NUM_THREADS=12 PYTHONPATH="$EXT/ReBoot" DYLD_LIBRARY_PATH="$EXT/install/lib"
mkdir -p "$EXT/data" "$EXT/run_cwd" && cd "$EXT/ReBoot/experiments/3_training_encrypted"
/usr/bin/time -l python training_encrypted.py MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48 --num-epochs 1 --seed 0 "$@" > "$LOG" 2>&1
