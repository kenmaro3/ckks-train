#!/bin/bash
# After Iris: R-44-1 (hekit-cuda GPU tests, clean clone ~/hekit-r44 at the Mac HEAD it was bundled from), then
# R-41-5 (non-interactive AfterActivation B=32 at 43f219f in ~/hekit). ~/hekit-runs/chain2.sh on quadruped-gpu.
set -u
R=$HOME/hekit-runs/run_job.sh
export PATH=$HOME/.cargo/bin:$PATH
until grep -q JOB_DONE $HOME/hekit-runs/out/iris_20/post.txt 2>/dev/null; do sleep 20; done

# --- R-44-1
export SYNCED_COMMIT_FILE=$HOME/hekit-r44/.synced-commit
NEED=512
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
mkdir -p $HOME/hekit-r44/results/quadruped/logs
if [ "$FREE" -ge "$NEED" ]; then
  bash $R R-44-1-build $HOME/hekit-r44 12 0 -- cargo test -p hekit-cuda --features cuda --release --no-run
  LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} bash $R R-44-1 $HOME/hekit-r44 12 1 -- \
    env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
    cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  cp $HOME/hekit-runs/logs/R-44-1.log $HOME/hekit-r44/results/quadruped/hekit_cuda_task44_tests.txt
else
  echo "skipped: needs $NEED MiB GPU, $FREE MiB free at $(date -Is)" > $HOME/hekit-runs/R-44-1.skipped
fi
unset SYNCED_COMMIT_FILE

# --- R-41-5 (43f219f tree in ~/hekit)
cd $HOME/hekit
bash $R hekit_build_release_parallel2 $HOME/hekit 12 0 -- cargo build --release -p hekit-bench --features parallel --bins
bash $HOME/hekit-runs/r41.sh R-41-5 results/quadruped/mnist_emlp1_ni_after_activation_b32.json 32 -- \
  --mode noninteractive --subset 1024 --epochs 1 --transpose reuse --placement after_activation --batch 32
echo CHAIN2_DONE
