#!/bin/bash
# R-REB-3 (ReBoot paper setting, plaintext): after R-REB-2 (chain4), before R-41-5 (held by ~/hekit-runs/hold/R-41-5).
# ~/hekit-runs/chain5.sh on quadruped-gpu.
set -u
R=$HOME/hekit-runs/run_job.sh
until grep -q JOB_DONE $HOME/hekit-runs/out/R-REB-2/post.txt 2>/dev/null; do sleep 30; done
ENVS="REBOOT_SRC=$HOME/reboot/src/ReBoot REBOOT_DATA=$HOME/reboot/data LD_LIBRARY_PATH=$HOME/reboot/install/lib PYTHONPATH=$HOME/reboot/src/ReBoot"
# smoke test: 1 seed, 1 epoch, scratch output
rm -rf $HOME/hekit-runs/scratch/rreb3_smoke
if nice -n 10 env $ENVS OMP_NUM_THREADS=2 RREB3_EPOCHS=1 RREB3_SEEDS=0 $HOME/reboot/venv/bin/python $HOME/reboot/reboot_paper_setting.py \
     $HOME/hekit-runs/scratch/rreb3_smoke/raw $HOME/hekit-runs/scratch/rreb3_smoke/data 2 > $HOME/hekit-runs/scratch/rreb3_smoke.log 2>&1; then
  bash $R R-REB-3 $HOME/reboot 12 0 -- env $ENVS OMP_NUM_THREADS=1 $HOME/reboot/venv/bin/python $HOME/reboot/reboot_paper_setting.py \
    $HOME/hekit-runs/out/R-REB-3/reboot_emlp1_paper_setting $HOME/hekit-runs/out/R-REB-3/reboot_emlp1_paper 10
  echo "R-REB-3 finished $(date -Is)"
else
  echo "R-REB-3 smoke test failed $(date -Is); see scratch/rreb3_smoke.log; R-REB-3 not run" > $HOME/hekit-runs/R-REB-3.failed
fi
rm -f $HOME/hekit-runs/hold/R-41-5
echo CHAIN5_DONE
