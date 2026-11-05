#!/bin/bash
# Generic provenance-recording job runner for quadruped-gpu (lives at ~/hekit-runs/run_job.sh on the server).
# Usage (always launched under nohup by the caller):
#   run_job.sh NAME WORKDIR THREADS GPU(0|1) -- CMD [ARGS...]
# Writes ~/hekit-runs/out/NAME/{pre.txt,post.txt,time.txt,rss.tsv,meta.json,cmd.txt} and the program
# output to ~/hekit-runs/logs/NAME.log. Runs CMD under `nice -n 10 /usr/bin/time -v`, with
# OMP/RAYON/cargo thread caps = THREADS. A watchdog samples the RSS of the job's own process tree every
# 10 s and kills only that tree if it exceeds RSS_LIMIT_KB (default 90 GB). Never touches other processes.
set -u
NAME=$1; WORKDIR=$2; THREADS=$3; GPU=$4; shift 4
[ "$1" = "--" ] && shift
RSS_LIMIT_KB=${RSS_LIMIT_KB:-94371840}   # 90 GiB
OUT=$HOME/hekit-runs/out/$NAME; LOG=$HOME/hekit-runs/logs/$NAME.log
mkdir -p "$OUT" "$HOME/hekit-runs/logs"
export PATH=$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH
export OMP_NUM_THREADS=$THREADS RAYON_NUM_THREADS=$THREADS CARGO_BUILD_JOBS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

snap() {  # $1 = file
  {
    echo "date: $(date -Is)"
    echo "uptime: $(uptime)"
    echo "--- df -h ~"; df -h ~
    echo "--- free -b"; free -b
    echo "--- nvidia-smi query"
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free,memory.total,utilization.gpu --format=csv 2>&1
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1
    if [ "$GPU" = 1 ]; then echo "--- nvidia-smi"; nvidia-smi 2>&1; fi
  } > "$1"
}

printf '%q ' "$@" > "$OUT/cmd.txt"; echo >> "$OUT/cmd.txt"
echo "$WORKDIR" > "$OUT/workdir.txt"
cp -f "${SYNCED_COMMIT_FILE:-$HOME/hekit/.synced-commit}" "$OUT/synced-commit.txt" 2>/dev/null || true
snap "$OUT/pre.txt"
START=$(date -Is); START_EPOCH=$(date +%s.%N)
cd "$WORKDIR" || { echo "no workdir $WORKDIR" > "$LOG"; exit 2; }
nice -n 10 /usr/bin/time -v -o "$OUT/time.txt" "$@" > "$LOG" 2>&1 &
TPID=$!
echo "$TPID" > "$OUT/time.pid"
tree_pids() {  # all descendants of $1 (inclusive)
  local p=$1; echo "$p"
  for c in $(pgrep -P "$p" 2>/dev/null); do tree_pids "$c"; done
}
PEAK=0; KILLED=0
echo -e "epoch_s\trss_kb_tree\tload1" > "$OUT/rss.tsv"
while kill -0 "$TPID" 2>/dev/null; do
  RSS=0
  for p in $(tree_pids "$TPID"); do r=$(awk '/^VmRSS:/{print $2}' /proc/$p/status 2>/dev/null); RSS=$((RSS + ${r:-0})); done
  [ "$RSS" -gt "$PEAK" ] && PEAK=$RSS
  echo -e "$(date +%s)\t$RSS\t$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT/rss.tsv"
  if [ "$RSS" -gt "$RSS_LIMIT_KB" ]; then
    echo "WATCHDOG: tree RSS ${RSS} kB > ${RSS_LIMIT_KB} kB, killing own job tree" | tee -a "$LOG" >> "$OUT/rss.tsv"
    for p in $(tree_pids "$TPID" | tac); do kill -TERM "$p" 2>/dev/null; done
    KILLED=1; sleep 5
    for p in $(tree_pids "$TPID" | tac); do kill -KILL "$p" 2>/dev/null; done
  fi
  sleep 10
done
wait "$TPID"; RC=$?
END=$(date -Is); END_EPOCH=$(date +%s.%N)
snap "$OUT/post.txt"
python3 - "$OUT" "$NAME" "$WORKDIR" "$THREADS" "$GPU" "$START" "$END" "$START_EPOCH" "$END_EPOCH" "$RC" "$PEAK" "$KILLED" <<'PY'
import json, sys, re, os, platform, subprocess
out, name, wd, thr, gpu, start, end, s0, s1, rc, peak, killed = sys.argv[1:]
def rd(f):
    p = os.path.join(out, f)
    return open(p).read() if os.path.exists(p) else None
time_v = rd("time.txt") or ""
tv = {}
for line in time_v.splitlines():
    k, sep, v = line.strip().partition(": ")
    if sep:
        tv[k.strip()] = v.strip()
def up(txt):
    m = re.search(r"^uptime: (.*)$", txt or "", re.M); return m.group(1) if m else None
def load(txt):
    m = re.search(r"load average: ([\d.]+), ([\d.]+), ([\d.]+)", txt or ""); return [float(x) for x in m.groups()] if m else None
cpu = subprocess.run(["bash", "-c", "lscpu | grep -E 'Model name|^CPU\\(s\\)|Socket|Thread\\(s\\) per core'"], capture_output=True, text=True).stdout
meta = {
    "job": name, "host": platform.node(), "kernel": platform.release(), "cpu": cpu.strip().splitlines(),
    "workdir": wd, "command": (rd("cmd.txt") or "").strip(),
    "threads_cap": int(thr), "env_thread_caps": {"OMP_NUM_THREADS": thr, "RAYON_NUM_THREADS": thr, "CARGO_BUILD_JOBS": thr},
    "nice": 10, "gpu_job": gpu == "1",
    "hekit_synced_commit": (rd("synced-commit.txt") or "").strip().splitlines(),
    "start": start, "end": end, "wall_s_runner": float(s1) - float(s0),
    "exit_status": int(rc), "watchdog_killed": killed == "1", "watchdog_rss_limit_kb": int(os.environ.get("RSS_LIMIT_KB", "94371840")),
    "uptime_before": up(rd("pre.txt")), "uptime_after": up(rd("post.txt")),
    "load_before": load(rd("pre.txt")), "load_after": load(rd("post.txt")),
    "peak_rss_kb_time_v": int(tv["Maximum resident set size (kbytes)"]) if "Maximum resident set size (kbytes)" in tv else None,
    "peak_rss_kb_tree_sampled_10s": int(peak),
    "time_v": tv,
    "snapshot_before": rd("pre.txt"), "snapshot_after": rd("post.txt"),
}
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
PY
echo "JOB_DONE rc=$RC" >> "$OUT/post.txt"
