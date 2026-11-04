#!/usr/bin/env bash
# Timed runs for Task 4 (SEAL 2020 reproduction). Holds the shared bench lock.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOCK=/Users/marconi/Documents/research/hekit/.superpowers/sdd/2026-11-04-he-training-implementation-plan/bench.lock
OUT="$HERE/results/${1:-run}"
mkdir -p "$OUT"
until mkdir "$LOCK" 2>/dev/null; do sleep 30; done
echo task4 > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
cd "$HERE"
{ date; uptime; } > "$OUT/load.txt"
./build/seal_ntt_bench 1000 > "$OUT/ntt_bench.txt" 2>&1
./build/he_train --transpose alg5 --keys exact --warmup 1 --iters 10 > "$OUT/time_alg5_exact.txt" 2>&1
./build/he_train --transpose alg4 --keys exact --warmup 1 --iters 10 > "$OUT/time_alg4_exact.txt" 2>&1
./build/he_train --transpose alg5 --keys pow2  --warmup 1 --iters 10 > "$OUT/time_alg5_pow2.txt" 2>&1
./build/seal_ntt_bench 1000 > "$OUT/ntt_bench_after.txt" 2>&1
{ date; uptime; } >> "$OUT/load.txt"
