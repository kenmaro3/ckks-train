#!/usr/bin/env bash
# Builds and runs the SEAL 4.1.2 operation micro-bench (see bench.cpp), recording the 1-min
# load average before and after each preset.
#
# Usage: tools/seal-kat/bench/run_bench.sh [presets...] (default: seal_2020 n16)
#        SEAL_BENCH_REPS=10 sets the timed repetitions per operation.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$ROOT/tools/seal-kat/bench
BUILD=$SRC/build
cmake_args=(-S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release)
SPIKE_SEAL=$ROOT/spikes/seal-2020/build/_deps/seal-src
if [ -d "$SPIKE_SEAL" ]; then
  cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_SEAL=$SPIKE_SEAL")
fi
mkdir -p "$BUILD"
cmake "${cmake_args[@]}" >"$BUILD/configure.log" 2>&1 || { cat "$BUILD/configure.log"; exit 1; }
cmake --build "$BUILD" -j --target seal_ops_bench >"$BUILD/build.log" 2>&1 || { tail -50 "$BUILD/build.log"; exit 1; }

presets=("$@")
[ ${#presets[@]} -eq 0 ] && presets=(seal_2020 n16)
for p in "${presets[@]}"; do
  echo "LOAD before $p: $(uptime)"
  "$BUILD/seal_ops_bench" "$p" "${SEAL_BENCH_REPS:-10}"
  echo "LOAD after $p: $(uptime)"
done
