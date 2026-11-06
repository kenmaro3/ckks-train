#!/usr/bin/env bash
# Known-answer interop between ckks-train-ckks and Microsoft SEAL 4.1.2 (seal_2020 preset).
#
# 1. builds tools/seal-kat/interop (Release; SEAL via FetchContent, offline from the spike's
#    checkout when it exists),
# 2. SEAL writes keys, ciphertexts, its decryptions and values.json to kat/seal_2020/
#    (gitignored, ~80 MB, regenerated on every run with fresh SEAL randomness),
# 3. cargo test -p ckks-train-ckks --test seal_kat -- --ignored (SEAL -> hekit; hekit writes
#    kat/seal_2020/hekit_*.bin),
# 4. SEAL decrypts hekit's files (hekit -> SEAL); exits non-zero on any failure.
#
# Usage: tools/seal-kat/run_kat.sh   (HEKIT_SEAL_KAT_DIR overrides the output directory)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC=$ROOT/tools/seal-kat/interop
BUILD=$SRC/build
KAT=${HEKIT_SEAL_KAT_DIR:-$ROOT/kat/seal_2020}

cmake_args=(-S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release)
SPIKE_SEAL=$ROOT/spikes/seal-2020/build/_deps/seal-src
if [ -d "$SPIKE_SEAL" ]; then
  cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_SEAL=$SPIKE_SEAL")
fi

echo "== [1/4] build SEAL 4.1.2 harness (Release)"
mkdir -p "$BUILD"
cmake "${cmake_args[@]}" >"$BUILD/configure.log" 2>&1 || { cat "$BUILD/configure.log"; exit 1; }
cmake --build "$BUILD" -j --target seal_interop_kat >"$BUILD/build.log" 2>&1 || { tail -50 "$BUILD/build.log"; exit 1; }
echo "built $BUILD/seal_interop_kat"

echo "== [2/4] SEAL generates KAT files"
mkdir -p "$KAT"
KAT=$(cd "$KAT" && pwd)
# Remove only the files this script and the KAT test write (the directory may be
# overridden by HEKIT_SEAL_KAT_DIR, so never delete it recursively).
find "$KAT" -maxdepth 1 -type f \( -name '*.bin' -o -name 'values.json' -o -name 'inputs.f64' \) -delete
"$BUILD/seal_interop_kat" generate "$KAT"

echo "== [3/4] hekit: cargo test -p ckks-train-ckks --test seal_kat -- --ignored"
(cd "$ROOT" && HEKIT_SEAL_KAT_DIR="$KAT" cargo test -p ckks-train-ckks --test seal_kat -- --ignored --nocapture --test-threads=1)

echo "== [4/4] SEAL decrypts hekit's ciphertexts"
"$BUILD/seal_interop_kat" verify "$KAT"
