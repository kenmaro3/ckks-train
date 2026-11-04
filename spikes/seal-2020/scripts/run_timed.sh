#!/usr/bin/env bash
# Timed runs for Task 4 (SEAL 2020 reproduction). Holds a bench lock so that timed runs do not overlap.
# Usage: scripts/run_timed.sh [run-name]      (logs go to results/<run-name>/)
# BENCH_LOCK: lock directory (default: build/bench.lock inside this spike, gitignored). Point several
#             scripts at the same path to serialise their timed runs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="${BENCH_LOCK:-$HERE/build/bench.lock}"
OUT="$HERE/results/${1:-run}"
for bin in seal_ntt_bench he_train; do
    [ -x "$HERE/build/$bin" ] || { echo "error: $HERE/build/$bin missing; build first (see README)" >&2; exit 1; }
done
mkdir -p "$(dirname "$LOCK")" || { echo "error: cannot create lock dir $(dirname "$LOCK")" >&2; exit 1; }
mkdir -p "$OUT"
until mkdir "$LOCK" 2>/dev/null; do echo "waiting for bench lock $LOCK" >&2; sleep 30; done
trap 'rm -rf "$LOCK"' EXIT
echo task4 > "$LOCK/owner"
cd "$HERE"
{ date; uptime; } > "$OUT/load.txt"
./build/seal_ntt_bench 1000 > "$OUT/ntt_bench.txt" 2>&1
./build/he_train --transpose alg5 --keys exact --warmup 1 --iters 10 > "$OUT/time_alg5_exact.txt" 2>&1
./build/he_train --transpose alg4 --keys exact --warmup 1 --iters 10 > "$OUT/time_alg4_exact.txt" 2>&1
./build/he_train --transpose alg5 --keys pow2  --warmup 1 --iters 10 > "$OUT/time_alg5_pow2.txt" 2>&1
./build/seal_ntt_bench 1000 > "$OUT/ntt_bench_after.txt" 2>&1
{ date; uptime; } >> "$OUT/load.txt"
