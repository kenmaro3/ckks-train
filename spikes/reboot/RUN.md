# ReBoot baseline run (Task 5 / Spike 3)

Status: **build OK, eMLP-1 does NOT run to completion on this machine (36 GB RAM): the CKKS context alone needs ~44 GB.
No epoch time / accuracy could be measured.** Paper's own numbers are recorded below for reference.

## Environment
- Machine: Apple M3 Pro, 12 cores, 36 GB RAM (38,654,705,664 B), macOS 26 (Darwin 25.3), arm64.
- Compiler: Apple clang 21.0.0 (clang-2100.1.1.101) + Homebrew libomp (already installed; nothing newly brewed).
- ReBoot: https://github.com/AI-Tech-Research-Lab/ReBoot @ `3ac173fcea376ba5a2f733b2b8a2fbd4d9547aa2` (README pins OpenFHE 1.2.1, OpenFHE-Python 0.8.9, Python 3.10.12).
- OpenFHE: tag `v1.2.1` (commit `85a8f6325c50e3e39d07d253a1fe150bad2d49ce`), installed to `_ext/install`.
  CMake: `-DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DWITH_NATIVEOPT=ON -DCMAKE_BUILD_TYPE=Release`
  plus `-DOpenMP_{C,CXX}_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" -DOpenMP_*_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib`, `make -j10`. NATIVE_SIZE=64.
- OpenFHE-Python: tag `v0.8.9` (commit `2447d372d63a6ff5c54e1cd1f85ad6d1c5dafe19`), installed into `_ext/venv`.
- Python: 3.11.9 venv (`_ext/venv`; README says 3.10.12, 3.11 works). numpy 2.0.0, pybind11 3.1.0, torch 2.15.0, torchvision 0.29.2 (+ pyyaml, scikit-learn, imbalanced-learn, ucimlrepo, pandas, seaborn, matplotlib, wandb).
- reboot_cpp (pybind11 module) built with `cmake` + `make -j10`, Release, same OpenMP flags.
- Threads: `OMP_NUM_THREADS=12` and the script itself calls `reboot_cpp.set_num_threads(os.cpu_count())` (= 12), as the README/script do.
- Dataset: MNIST via torchvision download into `_ext/data` (not committed). Paper (arXiv:2506.19693) PDF in `_ext/paper`.

## Patches to ReBoot (build/portability only; algorithm untouched). Full diff: `_ext/logs/reboot_patches.diff`
1. `cpp/CMakeLists.txt`: hard-coded `/workspaces/ReBoot` paths -> `${CMAKE_CURRENT_SOURCE_DIR}`; dropped GNU-only linker flag `-Wl,--as-needed` (macOS ld).
2. `lib/utils/data.py`: default `data_path` `/data` -> `os.environ.get("REBOOT_DATA","/data")` (+ `import os`).
3. `experiments/3_training_encrypted/training_encrypted.py`: `optimizer=OptimizerName.ENCRYPTED_NESTEROV_SGD` -> `.name` (upstream bug: `OptimizerName[<enum>]` raised KeyError; the other experiment scripts already use `.name`).
Env fixes (venv only): torch's and sklearn's bundled `libomp.dylib` replaced by symlinks to Homebrew's libomp, otherwise `OMP: Error #15 ... libomp already initialized` aborts at import.
Note: CMake flags `-Ofast -fopenmp -march=native` in ReBoot's CMakeLists are only stored in unused variables (`C(XX)_COMPILE_FLAGS`), so the module is built with CMake Release flags (`-O3`) + OpenMP flags only.

## Command line (as in upstream `experiments/3_training_encrypted/experiments.sh`, MLP-1, 1 epoch)
`spikes/reboot/run.sh <logname>`, i.e. in `_ext/ReBoot/experiments/3_training_encrypted`:
```
OMP_NUM_THREADS=12 REBOOT_DATA=_ext/data PYTHONPATH=_ext/ReBoot DYLD_LIBRARY_PATH=_ext/install/lib \
/usr/bin/time -l python training_encrypted.py MNIST --subsample 4800 --num-layers 2 --num-hidden 32 --batch-size 48 --num-epochs 1 --seed 0
```
(ReBoot's protocol: 4800-sample MNIST subsample, batch 48 -> 100 iterations/epoch; full MNIST would be 1250 it/epoch. The script prints *training* loss/accuracy only; test accuracy is computed in the paper with the plaintext twin.)

## Result: out of memory
- Full-size runs (batch 48 x 12 threads; also batch 4/12 x 12 threads, batch 4 x 4 threads) were SIGKILLed or aborted during the first training iteration, in the weight-bootstrapping step (`model.bootstrap`, 64 ciphertexts = weights + velocities, `#pragma omp parallel for`), after forward/backward had finished. Batch 1 x 1 thread was stopped by me after >10 min without finishing the step (swapping).
- Root cause measured with `footprint` (RSS is misleading because macOS compresses/swaps): **right after `CryptoContext(...)` construction (keygen + `EvalBootstrapSetup/KeyGen` + sum keys) the process footprint is 44 GB** (all MALLOC_SMALL; N=2^16, l=24, 32768 slots, level budget [2,2]). Even with `OMP_NUM_THREADS=1`, batch 1, a single `EvalBootstrap` took 161-178 s while swapping (swap 15-35 GB used, free memory ~0), and a full step never completed.
- This matches the paper: eMLP-1 peak memory 76.07 GB (eMLP-2 181.45 GB, eMLP-3 213.63 GB) on 2x Xeon Gold 5318S / 384 GB RAM. 36 GB cannot hold the eMLP-1 key material plus working set. No upstream issue exists for this (it is hardware, not a bug); not pursued further per the failure ruling (~1 h of attempts).
- The timed-run lock (`bench.lock`) was acquired once for a run that crashed at import (libomp) and released immediately; no timed measurement is reported, so no contention caveat applies to any number below.
- Not measured: epoch time, accuracy, peak memory of a full run. The only measured memory figure is the 44 GB context footprint (lower bound for peak).
- Possible follow-ups (not done, would change the benchmark): reduce ring/levels, run on a larger-RAM host.

## Reference numbers from the paper (not measured here)
- Paper hardware: 2x Intel Xeon Gold 5318S, 384 GB RAM, Ubuntu 20.04; thread count not stated.
- Table 6: eMLP-1 190.32 s per training iteration (N=2^16); eMLP-2 629.65 s. Hence ReBoot-protocol epoch (100 it) ~ 5.3 h and full-MNIST epoch (1250 it) ~ 66 h at the paper's speed (arithmetic from the paper's per-iteration time, not measured).
- Fig. 2 (precision analysis) shows only 400 encrypted iterations; accuracy tables use plaintext twin training: Table 4 eMLP-1 94.61 +/- 0.27, eMLP-2 96.39 +/- 0.16, eMLP-3 96.77 +/- 0.13 (MNIST test, ReBoot column).

## Step 3: config transcription (`crates/hekit-bench/configs/reboot_emlp{1,2,3}.json`)
Sources: Table 3 (N, l, Delta, widths), Table 8 (lr gamma, weight decay eta, batch b; MNIST/ReBoot row), Appendix A (momentum 0.9).

| | log_n | levels l (= tau_ReBoot + tau_bs) | scale_bits | hidden widths | lr | batch |
|---|---|---|---|---|---|---|
| eMLP-1 | 16 | 24 = 8+16 | 49 | 32 | 0.001 | 48 |
| eMLP-2 | 17 | 27 = 11+16 | 59 | 64, 32 | 0.001 | 48 |
| eMLP-3 | 17 | 29 = 13+16 | 59 | 128, 64, 32 | 0.0005 | 48 |

- `layers`: 784 -> hidden widths -> 10; hidden layers `poly_relu` (EncryptedPolyReLU), final classifier `none`. (The code pads the 10 classes to 16 internally; the JSON uses the logical 10.)
- `momentum` 0.9 (paper Appendix A, repo `training_encrypted.py`), `dataset` "MNIST".
- **`epochs` = 10 is NOT stated in the paper**; taken from repo `experiments/3_training_encrypted/experiments.sh` (`--num-epochs 10`, 4800-sample subsample). The script's own default is 500; Fig. 2 shows 400 iterations (= 4 epochs of 100 it).
- Disagreement paper vs repo: the repo never passes `--lr`, so its runs use the parser default 5e-4; the paper's Table 8 gives 0.001 (eMLP-1/2) and 0.0005 (eMLP-3) -> JSON uses the paper's values. Also weight decay: paper Table 8 eta = 0.001 / 0.005 / 0.001 (eMLP-1/2/3, MNIST); repo default `--weight-decay 0.0` (not in JSON schema).
- Repo configs (`experiments/config/fhe_config_mlp{1,2,3}.yaml`) agree with the paper on N (65536/131072/131072), levels (mult_depth from model depth 8/11/13 + bootstrap depth 14 + 2 iterations = 24/27/29) and Delta (49/59/59).
- Extra parameters (not in JSON): security level HEStd_128_classic (paper: lambda = 128, HE Standard); secret-key distribution SPARSE_TERNARY (OpenFHE 1.2.1 uses Hamming weight 192 for it: `tug` with 192, base-pke.cpp; the paper does not state it); dnum / numLargeDigits **not set** by ReBoot -> OpenFHE default (HYBRID key switching, automatic digit count; value not recorded); first modulus 54/60/60 bits; rescaling FLEXIBLEAUTOEXT (eMLP-1) / FLEXIBLEAUTO (eMLP-2/3); bootstrap level budget [2,2], 2 iterations, precision 11/18/17; full slots N/2; weight decay as above.
