# spikes/seal-2020 — Mihara et al. 2020 on SEAL 4.1 (Task 4)

Throwaway spike: one encrypted training iteration of the 2020 paper
(arXiv:2012.13552, "Neural Network Training With Homomorphic Encryption")
re-implemented on Microsoft SEAL 4.1.2 (CKKS), plus a SEAL NTT micro-bench.
Results are in `../RESULTS.md` (section "Task 4") and raw logs in `results/`.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches SEAL v4.1.2 into build/_deps (gitignored)
cmake --build build -j 6
(cd build && ctest)                               # = he_train --check (Alg.5 and Alg.4 layouts)
```

SEAL is pulled with `FetchContent` (tag `v4.1.2`, MIT). CMake flags set in
`CMakeLists.txt`: `SEAL_BUILD_DEPS=ON`, `SEAL_USE_MSGSL=OFF`, `SEAL_USE_ZLIB=OFF`,
`SEAL_USE_ZSTD=OFF`, `SEAL_USE_INTEL_HEXL=OFF`, `SEAL_BUILD_EXAMPLES/TESTS/BENCH/SEAL_C=OFF`,
`SEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON`; `CMAKE_BUILD_TYPE=Release`,
`CMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG"` (applies to SEAL too). SEAL's defaults
`SEAL_USE_INTRIN=ON`, `SEAL_USE___INT128` (auto-detected) stay on.

## Run (from this directory; the program reads `data/iris.csv`)

| command | what |
|---|---|
| `build/he_train --check` | 1 encrypted iteration, assert max weight error vs plaintext < 1e-3 |
| `build/he_train --iters 10 --warmup 1` | timing: 1 warm-up + 10 timed iterations (per-phase) |
| `build/he_train --transpose alg4` | plain diagonals + Alg. 4 transposition (default `alg5`: stepped diagonals + Alg. 5) |
| `build/he_train --keys pow2` | SEAL default Galois keys (±2^k only) instead of one key per used step |
| `build/he_train --check --dims 6,3,1` | the paper's Table-2 network (synthetic data) for op counts |
| `build/he_train --plain-train [--batch 20 --epochs 400 --seeds 10]` | plaintext training, accuracy |
| `build/seal_ntt_bench 1000` | SEAL forward Harvey NTT, N = 2^16, 60-bit q |
| `scripts/run_timed.sh [run-name]` | all timed runs under a bench lock, logs into `results/<run-name>/`. Lock dir: `$BENCH_LOCK`, default `build/bench.lock` (gitignored; parent created). Fails fast if `build/` binaries are missing |

Fixed seeds: split / weight init seed `2020` (`--seed`); sample order of the
timed iterations and epoch shuffling use `seed*7919+1`. All RNG is
`std::mt19937_64` with hand-written Box–Muller / Fisher–Yates (portable).

## Data

`data/iris.csv` is Fisher's Iris data (R. A. Fisher, 1936; UCI Machine
Learning Repository "Iris", CC BY 4.0, doi:10.24432/C56C76), here taken from
the copy bundled with scikit-learn (identical to R's `iris`; differs from the
UCI file in two known-erroneous entries). Columns: 4 features, label 0/1/2.
Inputs are z-scored with training-set statistics.
