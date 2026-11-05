#!/bin/bash
# R-45-1 (clean clone ~/hekit-r45 at the Mac HEAD >= a6ab1e2; supersedes R-44-1), R-41-F1, R-41-F2 (clean clone
# ~/hekit-r44 at 3e0e99c), R-REB-2 (~/reboot, data only), then R-41-5 (43f219f tree in ~/hekit). ~/hekit-runs/chain4.sh.
set -u
R=$HOME/hekit-runs/run_job.sh
H=$HOME/hekit-r44
export PATH=$HOME/.cargo/bin:$PATH

# --- R-45-1 (GPU), clean clone ~/hekit-r45
H5=$HOME/hekit-r45
export SYNCED_COMMIT_FILE=$H5/.synced-commit
NEED=600
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
mkdir -p $H5/results/quadruped/logs
if [ "$FREE" -ge "$NEED" ]; then
  bash $R R-45-1-build $H5 12 0 -- cargo test -p hekit-cuda --features cuda --release --no-run
  bash $R R-45-1 $H5 12 1 -- env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
    cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  cp $HOME/hekit-runs/logs/R-45-1.log $H5/results/quadruped/hekit_cuda_task45_tests.txt
else
  echo "skipped: needs $NEED MiB GPU, $FREE MiB free at $(date -Is)" > $HOME/hekit-runs/R-45-1.skipped
fi

export SYNCED_COMMIT_FILE=$H/.synced-commit
# --- R-41-F1, R-41-F2 (faithful, in-tree indices and init weights at 3e0e99c)
cd $H
bash $R hekit_r44_build_release_parallel $H 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
D=results/quadruped/raw/reboot_emlp1_official_subset
FA="--mode noninteractive --placement when_needed --faithful --indices $D/reboot_subsample_indices.txt --init-weights $D/reboot_init_weights --epochs 1 --transpose reuse"
HEKIT_DIR=$H bash $HOME/hekit-runs/r41.sh R-41-F1 results/quadruped/mnist_emlp1_ni_when_needed_b32_faithful.json 32 -- $FA --batch 32
HEKIT_DIR=$H bash $HOME/hekit-runs/r41.sh R-41-F2 results/quadruped/mnist_emlp1_ni_when_needed_b64_faithful.json 16 -- $FA --batch 64
unset SYNCED_COMMIT_FILE

# --- R-REB-2 (ReBoot side, no training)
bash $R R-REB-2 $HOME/reboot 2 0 -- env REBOOT_SRC=$HOME/reboot/src/ReBoot REBOOT_DATA=$HOME/reboot/data \
  LD_LIBRARY_PATH=$HOME/reboot/install/lib PYTHONPATH=$HOME/reboot/src/ReBoot \
  $HOME/reboot/venv/bin/python $HOME/reboot/reboot_full_setting.py $HOME/hekit-runs/out/R-REB-2/reboot_emlp1_full

# --- R-41-5 (43f219f tree in ~/hekit)
cd $HOME/hekit
bash $R hekit_build_release_parallel2 $HOME/hekit 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
bash $HOME/hekit-runs/r41.sh R-41-5 results/quadruped/mnist_emlp1_ni_after_activation_b32.json 32 -- \
  --mode noninteractive --subset 1024 --epochs 1 --transpose reuse --placement after_activation --batch 32
echo CHAIN4_DONE
