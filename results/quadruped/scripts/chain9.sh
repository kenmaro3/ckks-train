#!/bin/bash
# R-47-1 (hekit-cuda tests, clean clone ~/hekit-r45 >= feff7ed) after R-41-5, alongside R-REB-3 (chain8, untimed);
# then R-47-2 (timed CPU+GPU benchmark) once R-REB-3 (chain8) and R-47-1 are both done. ~/hekit-runs/chain9.sh.
set -u
R=$HOME/hekit-runs/run_job.sh
H5=$HOME/hekit-r45
export PATH=$HOME/.cargo/bin:$PATH SYNCED_COMMIT_FILE=$H5/.synced-commit
until grep -q JOB_DONE $HOME/hekit-runs/out/R-41-5/post.txt 2>/dev/null; do sleep 30; done

# --- R-47-1
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | sed "s/^/GPU free at start (MiB): /" > $HOME/hekit-runs/R-47-1.gpufree
mkdir -p $H5/results/quadruped/logs
bash $R R-47-1-build $H5 12 0 -- cargo test -p hekit-cuda --features cuda --release --no-run
bash $R R-47-1 $H5 12 1 -- env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} \
  cargo test -p hekit-cuda --features cuda --release -- --ignored --nocapture --test-threads=1
cp $HOME/hekit-runs/logs/R-47-1.log $H5/results/quadruped/hekit_cuda_task47_tests.txt

# --- R-47-2: needs a quiet machine (no other hekit/ReBoot job of ours)
until grep -q CHAIN8_DONE $HOME/hekit-runs/logs/chain8.log 2>/dev/null; do sleep 30; done
bash $R R-47-2-build $H5 12 0 -- cargo bench -p hekit-bench --features cuda --bench cuda_ops --no-run
slam() { { echo "date: $(date -Is)"; uptime; echo "--- SLAM processes (pid %cpu etime cmd)";
           ps -eo pid,pcpu,etime,args | grep -E "persistent_ster" | grep -v grep;
           echo "--- top CPU processes"; ps -eo pid,user,pcpu,args --sort=-pcpu | head -8; } ; }
mkdir -p $HOME/hekit-runs/out/R-47-2
slam > $HOME/hekit-runs/out/R-47-2/slam_before.txt
bash $R R-47-2 $H5 12 1 -- env LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64:${LD_LIBRARY_PATH:-} HEKIT_BENCH_THREADS=1,12 \
  cargo bench -p hekit-bench --features cuda --bench cuda_ops
slam > $HOME/hekit-runs/out/R-47-2/slam_after.txt
cp $HOME/hekit-runs/logs/R-47-2.log $H5/results/quadruped/cuda_ops_r47.log
mkdir -p $HOME/hekit-runs/out/R-47-2/criterion
( cd $H5/target/criterion && find cuda_ops_n16_train -name estimates.json -o -name benchmark.json | \
  tar cf - -T - ) | tar -C $HOME/hekit-runs/out/R-47-2/criterion -xf -
echo CHAIN9_DONE
