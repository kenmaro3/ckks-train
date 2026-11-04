# M0 spike results

Each task appends one `## Task N: ...` section below (edit only your own).

## Task 2: CPU NTT (Rust)

Crate: `spikes/ntt-cpu` (scalar Harvey NTT, Shoup precomputation, lazy reduction, no SIMD, single thread).

| item | value |
|---|---|
| machine | Apple M3 Pro (12 cores; bench uses 1 core) |
| rustc | 1.85.1 (4eb161250 2025-03-15) |
| build profile | `cargo bench` = release (`opt-level=3`, `lto=true`, `codegen-units=1`), default target-cpu |
| N | 65536 (2^16), negacyclic mod X^N+1 |
| q (60-bit, q ≡ 1 mod 2N) | 1152921504606584833 |
| forward NTT, median | **416.4 µs** (95% CI of median 416.2 – 416.5 µs; criterion mean CI 417.7 – 418.9 µs) |
| inverse NTT (incl. N^-1 scaling), median | 385.5 µs (95% CI 385.4 – 385.6 µs) |
| plan target band (1 Apple M-core) | 200 – 400 µs |
| inside band? | **No, slightly above (~4% over the upper edge)**; same order of magnitude, scalar code only |

Notes:
- Timing excludes the input copy (criterion `iter_batched_ref`, setup not timed). Two bench runs gave forward 423 µs and 417 µs (means of criterion's estimate), so run-to-run spread is ~1–2%.
- Twiddle/ordering convention is documented in the crate-level docs of `src/lib.rs` (forward: CT, natural in, bit-reversed out; inverse: GS, bit-reversed in, natural out).
- The SEAL 4.1 NTT on the same machine will be measured in Task 4 (SEAL is built there) to compute the SEAL ratio.

## Task 3: CUDA NTT

Crate: `spikes/ntt-cuda` (kernels `kernels/ntt.cu`, compiled at runtime by NVRTC through `cudarc` 0.12.1
with features `std, driver, nvrtc, cuda-12040`; CUDA libraries are `dlopen`ed at first use).

**状態: GPU 待ち(未実行)。** この Mac (Apple M3 Pro) には NVIDIA GPU も CUDA toolkit (`nvcc`, `libcuda`,
`libnvrtc`) も無い。クラウド GPU のレンタルは課金を伴うためユーザーの判断事項(本タスクでは Step 1 をスキップ)。
カーネル・ホストコード・bench・GPU テストは書いてあり、Mac 上では以下まで確認済み:

- `cargo build --all-targets` / `cargo clippy --all-targets -- -D warnings`: OK(cudarc は CUDA 無しでもビルドできる)。
- カーネル挙動の CPU エミュレーション(`src/lib.rs` の `emu`、`.cu` とスレッド/ブロック/ステージの添字計算と
  Shoup/lazy reduction を 1 対 1 に再現、`__umul64hi` = u128 の上位 64 bit)が `ntt_cpu::ntt_forward` と
  **ビット一致**: N = 2^16 × batch 64、N ∈ {2, 16, 256, 2048, 4096, 8192} × batch 3、全係数 q−1 / 0。
  各フェーズで「各要素にちょうど 1 回だけ触れる」ことも検査(GPU 上のデータ競合になる添字ミスを検出)。
- `.cu` は Homebrew clang 20 (NVPTX) で sm_80 / sm_90 の PTX まで警告なしでコンパイル可(NVRTC の代用、構文・型検査):
  `/opt/homebrew/opt/llvm/bin/clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_80 -nocudainc -nocudalib -O3 -Wall -Wextra -include kernels/clang_check_shim.h -S kernels/ntt.cu -o /tmp/ntt.ptx`
- GPU テスト(`#[ignore]`)は Mac では `SKIP: no usable CUDA device: ...` を出して通過、bench はメッセージを出して終了。

### 設計(ブリーフからの変更点)

ブリーフの「1 ブロック = 1 多項式、共有メモリ」は N = 2^16 では成立しない(u64 × 65536 = 512 KiB > SM あたり
共有メモリ上限 H100 228 KiB / A100 164 KiB)。そこで標準的な分割にした(radix-2 のまま):

| 部分 | カーネル | ステージ (N = 2^16) | launch 構成 (batch 64) |
|---|---|---|---|
| 大きいストライド (t ≥ 2048) | `ntt_fwd_global_stage` × 5 回 (1 ステージ 1 launch、global memory) | m = 1, 2, 4, 8, 16 | grid (128, 64), block 256、1 スレッド 1 butterfly |
| 残り (t = 1024 … 1) + 最終還元 | `ntt_fwd_shared` × 1 回 (チャンク 2048 係数 = 16 KiB を共有メモリへ) | m = 32 … 32768 (11 ステージ) | grid (32, 64), block 1024, dyn smem 16384 B |

batch は `blockIdx.y`(多項式 p は `a + p·N` に連続配置)。表・順序・算術は `ntt-cpu` の規約そのまま
(roots[i] = ψ^bitrev(i)、Shoup 表、CT natural-in / bit-reversed-out、[0, 4q) の lazy、出力 [0, q))。
q = 1152921504606584833(Task 2 と同じ)。計時は CUDA event、デバイス常駐データ(H2D/D2H は計時外)、
warm-up 5 回 + 50 回(≥ 20)、各回 event 対で計り中央値/最小/平均を per-poly µs で出す。計時前に CPU と全一致を確認。

### 測定値

| item | value |
|---|---|
| GPU (`nvidia-smi`) | **GPU 待ち(未実行)** |
| driver / CUDA / NVRTC version | GPU 待ち(未実行) |
| 時間単価 (provider, spot/on-demand) | GPU 待ち(未実行) |
| GPU vs CPU ビット一致 (N = 2^16, batch 64) | GPU 待ち(未実行)(CPU エミュレーションでは一致) |
| batch 64 forward NTT, per-poly µs (median / min) | **GPU 待ち(未実行)** |
| 参考: HEonGPU (GPU-NTT) A100, N = 2^16, batch 64, radix2-CT | 196.43 µs / batch = **3.07 µs / poly** |
| 判定基準 | 我々の per-poly µs ≤ 2 × 3.07 = **6.14 µs**(A100 で測った場合) |
| 判定 | GPU 待ち(未実行) |

### HEonGPU の公表値(Step 3 の比較対象)

HEonGPU の NTT は同じ著者の GPU-NTT ライブラリ(HEonGPU の README が third-party dependency / submodule として
リンク)。HEonGPU の README には NTT 単体の数値は無い(bootstrapping 等のみ)。HEonGPU 論文 (ePrint 2024/1543) と
GPU-NTT の ePrint 版 (2023/1410) は取得時に HTTP 403 で本文を確認できなかった。NTT 単体の数値は GPU-NTT の査読版
論文から転記する:

Source: A. Özcan, A. Javeed, E. Savaş, "High-Performance Number Theoretic Transform on GPU Through radix2-CT and
4-Step Algorithms", IEEE Access 13 (2025) 87862–87883, doi:10.1109/ACCESS.2025.3570024 (CC BY 4.0),
PDF https://research.sabanciuniv.edu/51889/1/High-Performance.pdf — accessed 2026-11-04.
Setup: CUDA 12.1、cudaEventRecord、50–100 回の平均、forward のみ。

| 表 | GPU | 条件 | radix2-CT | 4-Step |
|---|---|---|---|---|
| Table 10 (batch forward NTT, µs **per batch**) | A100 (80 GB, 1935 GB/s) | N = 2^16, batch 64 | 196.43 (3.07 /poly) | 188.83 (2.95 /poly) |
| Table 10 | RTX 4090 | N = 2^16, batch 64 | 97.16 (1.52 /poly) | 91.00 (1.42 /poly) |
| Table 10 | RTX 3060 Ti | N = 2^16, batch 64 | 422.00 (6.59 /poly) | 406.88 (6.36 /poly) |
| Table 9 (single NTT latency, 64-bit Goldilocks q) | A100 / RTX 4090 | N = 2^16, batch 1 | 15.93 / 8.66 | 16.34 / 9.04 |
| Table 11 (single NTT latency, 60-bit random q, Plantard) | A100 / RTX 4090 | N = 2^16, batch 1 | 16.15 / 8.86 | – |

注意: Table 10 のキャプションには法の種類が明記されていない(Table 9 は 64-bit Goldilocks 素数 2^64 − 2^32 + 1)。
Table 11 によれば 60-bit 一般素数でも単体 NTT は ~1% 差なので、比較上の影響は小さいとみなす。H100 の公表値は無い。
A100 を借りれば同じ GPU で直接比較できる(H100 で測る場合、A100 の基準 6.14 µs は甘めになる)。

参考の帯域見積(測定ではない): 本実装は batch あたり global memory を 6 往復(5 stage + 1 shared kernel)、
64 × 2^16 × 8 B = 32 MiB を読み書き → 約 403 MB。A100 のピーク 1935 GB/s で ~208 µs / batch ≈ 3.3 µs / poly が
下限の目安(加えて 64-bit 乗算のエミュレーションコスト)。

### GPU マシンでの実行手順

```sh
# 前提: NVIDIA driver (libcuda.so) と CUDA 12.x の NVRTC (libnvrtc.so.12) が LD_LIBRARY_PATH 上にあること。
#       rustup があれば rust-toolchain (1.85) は自動で入る。
cd hekit/spikes/ntt-cuda
nvidia-smi                                   # ヘッダ(Driver / CUDA Version)と GPU 名をそのまま上の表へ
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
cargo test --release -- --include-ignored --nocapture   # エミュレーション 6 本 + GPU 一致 2 本(SKIP が出ないこと)
cargo run --release --bin bench                          # N = 2^16, batch 64, 50 reps, warm-up 5
cargo run --release --bin bench -- 16 64 200 20          # 引数: log2N batch reps warmup(再現性確認用)
```

記録するもの: `nvidia-smi` の GPU 名・driver・CUDA version・メモリ・最大クロック、プロバイダ名と時間単価
(spot/on-demand の別)、bench 出力の device 行 (sm_XY)、CPU agreement 行、per-poly µs の median / min。

## Task 4: SEAL 2020 reproduction

Code: `spikes/seal-2020/` (C++17, SEAL **4.1.2** via FetchContent, Release `-O3`; flags in its README).
Paper: Mihara et al., arXiv:2012.13552 (Alg. 2/4/5, Tables 2/4). Raw logs: `spikes/seal-2020/results/`.

**Setup (as paper):** Iris 4-10-3 MLP with biases, square activation after both dense layers, MSE (mean over 3 outputs),
SGD lr 0.1, batch 1 for the timed iteration; CKKS N = 2^15, coeff modulus {60, 40×9, 60}, Δ = 2^40 (tc128),
weights **and** input/label encrypted, diagonal packing, transposition by rotations + additions only.
Interactive mode: after the SGD update the client decrypts and re-encrypts ("repacking", paper Fig. 7).

**Depth / levels:** one full iteration uses exactly the 9 available rescales, so no mid-iteration refresh is needed:
z1 (1) → a1 = z1² (2) → z2 (3) → a2 (4) → (a2−y)⊙z2 (5) → ×lr·4/3 (ct-pt, 6) → W2ᵀδ2 (7) → ⊙z1 (8) → ΔW1 = δ1⊗x (9).
The updated W1 is at level 0 (W2 at level 2) → client repacking after every iteration, exactly where the paper puts it.

**Layout choices** (documented in `src/he_train.cpp`): generalized (non-square) diagonals: W (out×in) → `in` diagonals of
length `out` (no padding to square / power of two needed — the input is replicated with period `in` instead: client packs
x replicated, server replicates a1 and δ2 with log-doubling rotations). Two variants:
`alg5` (default) = pre-rotated "stepped" diagonals (paper Fig. 6) + Alg. 5 transpose (M−1 rotations);
`alg4` = plain diagonals + Alg. 4 transpose (Σ_i (e_i+1) terms), with the forward rotations of x / a1 reused for ΔW.
Only W2 is transposed (W1ᵀ is not needed for training a 2-layer net; the paper's Table 2 mirrors both layers).
W2 diagonals are mod-switched down to the level where W2ᵀ is used *before* transposing. Galois keys: one per used step
("exact", 20 keys) or SEAL's default ±2^k keys ("pow2", closer to what SEAL 3.4 code typically used).

### Timing (Apple M3 Pro, single thread — SEAL is single-threaded; 1 warm-up + 10 timed iterations)

Machine was shared with other agents' jobs: load average **67.7 → 31.4** (1-min) during run 1; run 2 (load 17–37) was noisier
and is in `results/run2/`. Numbers below are run 1, mean ± sd over 10 iterations (seconds).

Paper Table 4, every row exactly as printed (seconds, batch 1, i7-8700K, SEAL 3.4), next to our measurements and the
phase of ours each row is compared with:

| Table 4 row (as printed) | paper row | paper diagonal | our phase compared with | ours alg5, exact keys | ours alg4, exact keys | ours alg5, pow2 keys |
|---|---|---|---|---|---|---|
| Feedforward | 12.42 | 3.10 | FF: z1, a1, a1 replication, z2, a2 | 0.713 ± 0.051 | 0.775 ± 0.058 | 0.806 ± 0.007 |
| Transition | 7.28 | 2.15 | TR: W2 mod-switch + transpose (W2 only) | 0.025 ± 0.002 | 0.090 ± 0.004 | 0.037 ± 0.001 |
| Backpropagation | 5.46 | 3.12 | BP: a2−y, δ2, lr scaling, δ2 replication, W2ᵀδ2, δ1, **and ΔW/Δb outer products** | 0.275 ± 0.005 | 0.213 ± 0.024 | 0.318 ± 0.003 |
| SGD Update | 0.23 | 0.23 | UPD: W − ΔW, b − Δb (subtractions + mod-switches) | 0.017 ± 0.002 | 0.019 ± 0.003 | 0.016 ± 0.000 |
| Repacking | 0.62 | 0.50 | client decrypt + re-encode + re-encrypt of 16 weight cts | 0.345 ± 0.009 | 0.345 ± 0.011 | 0.339 ± 0.005 |
| Total | 28.47 | 9.25 | stated total, compared below | | | |
| *sum of phases (computed here, not printed in the paper)* | *26.01* | *9.10* | sum of the five rows above | 1.376 | 1.442 | 1.515 |

**Note: the paper's stated totals exceed the sum of its own phases, and the paper doesn't explain the gap.**
Diagonal: stated 9.25 vs phase sum 9.10 (+0.15 s). Row: stated 28.47 vs phase sum 26.01 (+2.46 s). We don't guess
what the gap is. No server-side figure is derived by subtracting from the stated total.

Headline comparison, against the paper's **stated total** (9.25 s, diagonal):

| ours (run 1) | what it includes | value | paper stated total / ours |
|---|---|---|---|
| alg5, exact keys | server iteration + repacking + client encryption of x, y | 1.031 + 0.345 + 0.039 = **1.415 s** | **6.5×** |
| alg4, exact keys | same | 1.097 + 0.345 + 0.040 = 1.482 s | 6.2× |
| alg5, pow2 keys | same | 1.176 + 0.339 + 0.039 = 1.554 s | 6.0× |

Without the 0.039 s client encryption (server + repacking only): alg5 1.376 s → 6.7×.

Indicative only (phase sums, assuming the paper's phases mean what ours mean; see the mapping caveat below):
paper FF + Transition + BP + SGD Update = 8.60 s vs our server iteration 1.031 s (≈ 8.3×). Paper phase sum incl.
Repacking = 9.10 s vs our 1.376 s (≈ 6.6×).

Row packing (not implemented here): printed server-side rows sum to 25.39 s, all printed rows to 26.01 s.
(Run 2, noisier: alg5-exact server 1.166 ± 0.219 s, server + repacking 1.574 s.)
→ About 6.5× faster than the paper's stated total (ours = server iteration + repacking + client encryption of x, y; what the paper's total contains beyond its phases is unknown). All per-phase comparisons are **indicative**: our BP includes the gradient outer products, and the paper doesn't say whether its "Backpropagation" or its "SGD Update" (0.23 s) contains them. The paper's
2.15 s transition is ~90× our 0.025 s. We can't tell which of these explains it: hardware, SEAL 3.4 → 4.1, keys,
transposing at the top level, or transposing both layers (the paper doesn't give enough detail). pow2 keys only cost
+14 %, so the key set alone doesn't explain it.

### Correctness

Updated weights after one encrypted iteration vs. a dense double-precision reference on the same init/sample:
**max |W_he − W_ref| = 4.4e-6** (run-to-run 4e-6 – 1.1e-5; single-step update magnitude up to 0.107);
over 11 chained iterations (with repacking) it is ≤ 2.6e-5. `he_train --check` (CTest) asserts < 1e-3 for both layouts.
A slot-exact plaintext simulation of the packed algorithm matches the dense reference to 1e-16 (packing logic exact).
Main error source: SEAL refuses to add ciphertexts with unequal scales; we force-equalize (max relative mismatch 7.4e-5,
from scales drifting through 9 rescales by 40-bit primes ≠ 2^40), which perturbs ΔW by ~1e-5 relative.

### Op counts (server side, one iteration, counted at every evaluator call)

Iris 4-10-3 (both layouts; `relin` = `mul_cc`, rescale = mul_cc + mul_cp):

| layout | phase | ct-ct mul | ct-pt mul | relin | rot | rescale |
|---|---|---|---|---|---|---|
| alg5 | FF / TR / BP / UPD | 16 / 0 / 19 / 0 | 0 / 0 / 1 / 0 | 16 / 0 / 19 / 0 | 13 / 2 / 16 / 0 | 16 / 0 / 20 / 0 |
| alg5 | **total** | **35** | **1** | **35** | **31** | **36** |
| alg4 | FF / TR / BP / UPD | 16 / 0 / 19 / 0 | 0 / 0 / 1 / 0 | 16 / 0 / 19 / 0 | 13 / 11 / 4 / 0 | 16 / 0 / 20 / 0 |
| alg4 | **total** | **35** | **1** | **35** | **28** | **36** |

Paper's Table-2 network 6-3-1 (`--dims 6,3,1`, synthetic data), row by row against Table 2 (diagonal column):

| Table 2 row | paper mult / rot | ours alg5 mult / rot | ours alg4 mult / rot |
|---|---|---|---|
| FF Dense 6→3 | 6 / 3 | 6 / 5 | 6 / 5 |
| FF square | 1 / 0 | 1 / 0 | 1 / 0 |
| FF Dense 3→1 (a1 replication needs 0 rot here) | 3 / 1 | 3 / 2 | 3 / 2 |
| FF square | 1 / 0 | 1 / 0 | 1 / 0 |
| Transition W1 (6×3) | 0 / 3 | not done (not needed) | not done |
| Transition W2 (3×1) | 0 / 1 | 0 / 0 | 0 / 2 |
| BP output-square derivative (a2−y)⊙z2 | — (not listed) | 1 / 0 | 1 / 0 |
| BP lr scaling (ct-pt) | — | 1 (pt) / 0 | 1 (pt) / 0 |
| BP δ2 replication | — | 0 / 2 | 0 / 2 |
| BP Dense 1→3 (W2ᵀδ2) | 1 / 3 | 1 / 0 | 1 / 0 |
| BP square derivative ⊙z1 | 1 / – | 1 / 0 | 1 / 0 |
| BP Dense 3→6 + square (W1ᵀδ1, input gradient) | 3 / 6 + 1 / 0 | not done (not needed) | not done |
| ΔW2, ΔW1 (outer products; not in Table 2) | — | 9 / 7 | 9 / 0 |
| **total** | **17 / 17** | **23 + 1 pt / 16** | **23 + 1 pt / 11** |

Why they differ: (1) Table 2 omits the weight-gradient outer products (in·out entries → `in` ct-ct mults per layer) and
the output-layer derivative, and includes the unnecessary input-layer backprop (W1ᵀ, Dense 3→6). (2) Table 2's Dense
rotation count equals the **output** dim (6→3: 3 rot), whereas Alg. 2 as written needs `in − 1` rotations of x
(the paper's own text says "N multiplications and N rotations"); we follow Alg. 2. (3) Transition: Table 1 / Table 2
count M rotations for Alg. 5, we need M − 1 (the i = 0 diagonal needs none); Alg. 4 needs Σ_i (e_i + 1) minus trivial
ones = 11 for W2 of Iris (paper's q(M+1) = 12). (4) Replication rotations (log-doubling of a1 and δ2 so that rotations
wrap with period `in`) are not discussed in the paper. Mult counts of the rows both sides compute agree exactly.
Rule inferred from Table 2 for a cost model: Dense(a→b) = a mult, b rot; square = 1 mult; transition(a×b) = 0 mult, b rot.

### SEAL NTT (for Task 2's ratio)

`seal_ntt_bench`: `seal::util::ntt_negacyclic_harvey`, N = 2^16, q = 1152921504606584833 (accepted, 60-bit), 1000 reps,
input copy excluded, single thread. Same load conditions as above.

| | run 1 before / after | run 2 before / after |
|---|---|---|
| forward (fully reduced output), median | **368.0** / 364.2 µs | 376.4 / 379.1 µs |
| forward lazy (`_lazy`, output in [0,4q)), median | 330.3 / 331.0 µs | 335.9 / 334.0 µs |
| inverse (incl. N⁻¹), median | 355.1 / 357.2 µs | 362.3 / 356.7 µs |

**Ratio Rust (416.4 µs) / SEAL (368.0 µs) = 1.13** (vs SEAL lazy: 1.26; range over runs 1.10 – 1.14 for the non-lazy one).
SEAL 4.1 on arm64 has no SIMD NTT path either (HEXL is x86-only), so this is scalar vs scalar.

### Plaintext training (400 epochs, same network/hyperparameters, `he_train --plain-train`)

Init N(0, 0.2²) for W and b, z-scored inputs, 80:20 split (120/30), lr 0.1, MSE.

| batch | seeds | converged | train acc (mean) | test acc (mean ± sd) | train loss |
|---|---|---|---|---|---|
| 20 (paper's full-training batch) | 2020–2029 | 10/10 | **0.982** | 0.960 ± 0.021 | 0.0245 |
| 20 | 2020–2069 | 50/50 | 0.983 | 0.969 ± 0.029 | 0.0246 |
| 1, lr 0.1 | 2020–2029 | **0/10 (diverge in epoch 1–2)** | — | — | — |
| 1, lr 0.01 | 2020–2029 | 10/10 | 0.982 | 0.950 ± 0.028 | 0.0238 |

Paper: plaintext accuracy 0.9805, loss 0.0249 (which split is not stated) — matches our train accuracy/loss.
Note: batch 1 with lr 0.1 diverges (square∘square activations), so the paper's batch-1 setting is a timing-only setting.
Without biases or without standardization accuracy drops to 0.70–0.96 (exploration, not committed).
