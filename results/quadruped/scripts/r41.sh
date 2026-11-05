#!/bin/bash
# Run one Task 41 Step 5 request block (requests.md R-41-x) on quadruped-gpu (~/hekit-runs/r41.sh).
# HEKIT_DIR (default ~/hekit) selects the tree. Usage: r41.sh ID OUT_JSON_REL NUM_UPDATES -- mnist args...
# The request's `RAYON_NUM_THREADS=12 /usr/bin/time -v ./target/release/mnist ...` is run through run_job.sh, which
# sets RAYON_NUM_THREADS=12 and wraps the command in `nice -n 10 /usr/bin/time -v`.
# Partial rule (requests.md): after 2 updates, project setup + mean(update 0,1) * NUM_UPDATES; if > 5 h, stop the
# job after 8 updates have been logged and mark it partial (out/ID/partial.txt).
set -u
ID=$1; OUTJ=$2; NUPD=$3; shift 3; [ "$1" = "--" ] && shift
H=${HEKIT_DIR:-$HOME/hekit}
LOG=$HOME/hekit-runs/logs/$ID.log; OUT=$HOME/hekit-runs/out/$ID
mkdir -p "$OUT"; rm -f "$OUT/partial.txt" "$OUT/time.pid"
(
  # watcher
  while [ ! -f "$OUT/time.pid" ]; do sleep 5; done
  TPID=$(cat "$OUT/time.pid"); decided=0; partial=0
  while kill -0 "$TPID" 2>/dev/null; do
    n=$(grep -c "^mnist: epoch .* update " "$LOG" 2>/dev/null)
    if [ "$decided" = 0 ] && [ "${n:-0}" -ge 2 ]; then
      proj=$(grep "^mnist: epoch .* update " "$LOG" | head -2 | sed -E 's/.*\): ([0-9.]+) s,.*elapsed ([0-9]+) s.*/\1 \2/' |
        awk -v N="$NUPD" '{s+=$1; e=$2} END {mean=s/2; setup=e-s; printf "%.0f", setup + mean*N}')
      echo "projection after 2 updates: ${proj} s for ${NUPD} updates" > "$OUT/projection.txt"
      decided=1; [ "$proj" -gt 18000 ] && partial=1
    fi
    if [ "$partial" = 1 ] && [ "${n:-0}" -ge 8 ]; then
      echo "partial: projected > 5 h ($(cat $OUT/projection.txt)); stopped after $n updates at $(date -Is)" > "$OUT/partial.txt"
      for p in $(pgrep -P "$TPID") "$TPID"; do kill -TERM "$p" 2>/dev/null; done
      break
    fi
    sleep 15
  done
) &
bash $HOME/hekit-runs/run_job.sh "$ID" "$H" 12 0 -- ./target/release/mnist "$@" --out "$OUTJ"
wait
mkdir -p $H/results/quadruped/logs && cp "$LOG" "$H/results/quadruped/logs/$ID.log"
