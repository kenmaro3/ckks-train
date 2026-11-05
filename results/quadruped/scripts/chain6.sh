#!/bin/bash
# R-45-2 (hekit-cuda tests at >= 8b81f3d, clean clone ~/hekit-r45): starts when R-41-F2 is done, i.e. alongside the
# untimed R-REB-2 / R-REB-3 plaintext jobs, so its build overlaps no timed run. ~/hekit-runs/chain6.sh.
set -u
R=$HOME/hekit-runs/run_job.sh
H5=$HOME/hekit-r45
export PATH=$HOME/.cargo/bin:$PATH SYNCED_COMMIT_FILE=$H5/.synced-commit
until grep -q JOB_DONE $HOME/hekit-runs/out/R-41-F2/post.txt 2>/dev/null; do sleep 30; done
NEED=600
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
mkdir -p $H5/results/quadruped/logs
if [ "$FREE" -ge "$NEED" ]; then
  bash $R R-45-2-build $H5 12 0 -- cargo test -p hekit-cuda --features cuda --release --no-run
  bash $R R-45-2 $H5 12 1 -- env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
    cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
  cp $HOME/hekit-runs/logs/R-45-2.log $H5/results/quadruped/hekit_cuda_task45_fix1_tests.txt
else
  echo "skipped: needs $NEED MiB GPU, $FREE MiB free at $(date -Is)" > $HOME/hekit-runs/R-45-2.skipped
fi
echo CHAIN6_DONE
