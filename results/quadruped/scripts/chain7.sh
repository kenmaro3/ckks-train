#!/bin/bash
# R-46-1 (hekit-cuda tests at >= f68c05f, clean clone ~/hekit-r45; supersedes R-46-1): starts when R-41-F2 is done, i.e. alongside the
# untimed R-REB-2 / R-REB-3 plaintext jobs, so its build overlaps no timed run. ~/hekit-runs/chain7.sh.
set -u
R=$HOME/hekit-runs/run_job.sh
H5=$HOME/hekit-r45
export PATH=$HOME/.cargo/bin:$PATH SYNCED_COMMIT_FILE=$H5/.synced-commit
until grep -q JOB_DONE $HOME/hekit-runs/out/R-41-F2/post.txt 2>/dev/null; do sleep 30; done
# Always run (requests.md: tests that do not fit skip themselves); record free GPU memory at start.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU free at start: $FREE MiB (block asks to note < 1600 MiB) $(date -Is)" > $HOME/hekit-runs/R-46-1.gpufree
mkdir -p $H5/results/quadruped/logs
bash $R R-46-1-build $H5 12 0 -- cargo test -p hekit-cuda --features cuda --release --no-run
bash $R R-46-1 $H5 12 1 -- env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
cp $HOME/hekit-runs/logs/R-46-1.log $H5/results/quadruped/hekit_cuda_task46_tests.txt
echo CHAIN7_DONE
