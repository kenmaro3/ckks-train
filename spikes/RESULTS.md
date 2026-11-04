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
| **idle-machine rerun** (2026-11-04 16:17, load avg 1-min 2.97 → 3.20; run by Task 4, `cargo bench`, same binary) | forward median **417.8 µs** (CI 416.6 – 419.2; mean 424.0; min sample-mean 413.8), inverse median **408.9 µs** (CI 401.3 – 414.4; min sample-mean 384.8). Log: `spikes/seal-2020/results/run3_idle/rust_ntt_cpu_cargo_bench.txt` |
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
#       driver は CUDA >= 12.4 に対応していること(cudarc の `cuda-12040` バインディング。nvidia-smi の
#       "CUDA Version" が 12.4 以上)。NVRTC は driver より新しくてはいけない(新しい NVRTC の PTX は古い
#       driver が読めず、モジュールのロードで失敗する)。
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

**Idle-machine rerun (run 3, `results/run3_idle/`, headline):** load average (1-min) **3.61 → 2.97** (waited until ≤ 4).
Mean ± sd over 10 iterations (s):

| phase (ours) | alg5, exact keys | alg4, exact keys | alg5, pow2 keys |
|---|---|---|---|
| Feedforward | 0.689 ± 0.022 | 0.730 ± 0.005 | 0.800 ± 0.002 |
| Transition | 0.024 ± 0.002 | 0.085 ± 0.001 | 0.037 ± 0.000 |
| Backpropagation (incl. ΔW, Δb) | 0.269 ± 0.004 | 0.197 ± 0.002 | 0.316 ± 0.002 |
| SGD update | 0.015 ± 0.001 | 0.015 ± 0.000 | 0.015 ± 0.000 |
| **server iteration** | **0.997 ± 0.027** | 1.027 ± 0.006 | 1.168 ± 0.002 |
| Repacking (client) | 0.334 ± 0.001 | 0.335 ± 0.002 | 0.336 ± 0.002 |
| client encryption of x, y | 0.038 | 0.038 | 0.038 |
| **server + repacking + client enc** | **1.369** | 1.400 | 1.542 |
| paper stated total 9.25 s / ours | **6.8×** | 6.6× | 6.0× |

Idle, without client encryption: alg5 1.331 s → 7.0× vs 9.25 s. Indicative phase-sum ratios (idle, alg5): 8.60 / 0.997 =
8.6× (server phases), 9.10 / 1.331 = 6.8× (incl. repacking). Max |W_he − W_ref| over 11 chained iterations in run 3:
1.4e-5 (alg5), 3.6e-5 (alg4), 1.2e-4 (pow2), all < 1e-3. The error depends on the encryption randomness, so it varies
between runs.

**Under load (earlier runs, kept for reference):** the machine was shared with other agents' jobs. Load average (1-min)
was **67.7 → 31.4** during run 1; run 2 (load 17–37) was noisier and is in `results/run2/`. Numbers below are
run 1 **under load**, mean ± sd over 10 iterations (seconds).

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

Comparison against the paper's **stated total** (9.25 s, diagonal), run 1 **under load** (idle numbers above):

| ours (run 1, under load) | what it includes | value | paper stated total / ours |
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
→ Idle: about **6.8×** faster than the paper's stated total (under load: 6.5×; ours = server iteration + repacking + client encryption of x, y; what the paper's total contains beyond its phases is unknown). All per-phase comparisons are **indicative**: our BP includes the gradient outer products, and the paper doesn't say whether its "Backpropagation" or its "SGD Update" (0.23 s) contains them. The paper's
2.15 s transition is ~90× our 0.025 s. We can't tell which of these explains it: hardware, SEAL 3.4 → 4.1, keys,
transposing at the top level, or transposing both layers (the paper doesn't give enough detail). pow2 keys only cost
+14 %, so the key set alone doesn't explain it.

### Correctness

Updated weights after one encrypted iteration vs. a dense double-precision reference on the same init/sample:
**max |W_he − W_ref| = 4.4e-6** (run-to-run 4e-6 – 1.1e-5; single-step update magnitude up to 0.107);
over 11 chained iterations (with repacking) it is ≤ 1.2e-4 over runs 1–3 (run 3, idle: 1.4e-5 alg5 / 3.6e-5 alg4 /
1.2e-4 pow2; runs 1–2: 1.6e-5 – 4.3e-5; `[check]` lines of `results/run*/time_*.txt`). `he_train --check` (CTest)
asserts < 1e-3 for both layouts.
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
Rule inferred from Table 2 (it reproduces the paper's table, not true cost; the cost model follows Alg. 2 instead):
Dense(a→b) = a mult, b rot; square = 1 mult; transition(a×b) = 0 mult, b rot.

### SEAL NTT (for Task 2's ratio)

`seal_ntt_bench`: `seal::util::ntt_negacyclic_harvey`, N = 2^16, q = 1152921504606584833 (accepted, 60-bit), 1000 reps,
input copy excluded, single thread. Runs 1–2 under load (as above); run 3 idle (load 3.61 → 2.97).

| | run 1 before / after (under load) | run 2 before / after (under load) | **run 3 before / after (idle)** |
|---|---|---|---|
| forward (fully reduced output), median | 368.0 / 364.2 µs | 376.4 / 379.1 µs | **373.7** / 376.5 µs |
| forward (fully reduced output), min | 358.2 / 358.0 µs | 358.0 / 359.4 µs | **334.4** / 339.5 µs |
| forward lazy (`_lazy`, output in [0,4q)), median | 330.3 / 331.0 µs | 335.9 / 334.0 µs | 332.3 / 363.9 µs |
| inverse (incl. N⁻¹), median | 355.1 / 357.2 µs | 362.3 / 356.7 µs | 356.0 / 356.5 µs |

**Idle ratio (Rust and SEAL both rerun idle, ~1 min apart):**
- median-based: Rust forward median 417.8 µs / SEAL forward median 373.7 µs = **1.12**. Against the "after" SEAL run (376.5 µs) it is 1.11.
- min-based: Rust 413.8 µs / SEAL 334.4 µs = **1.24**. Criterion's min is the fastest *sample mean* (each sample averages many calls), whereas SEAL's min is a single call, so this ratio is biased upward.
- vs the SEAL lazy forward (median 332.3 µs): 1.26. Inverse medians: 408.9 / 356.0 = 1.15.

Under load (earlier): Rust (416.4 µs) / SEAL (368.0 µs) = 1.13; range over runs 1–2 is 1.10 – 1.14, and 1.26 vs SEAL lazy.
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

## Task 5: ReBoot baseline

ReBoot (OpenFHE 1.2.1, `3ac173f`) builds on this Mac, but **eMLP-1 cannot run on 36 GB RAM**: CKKS context (N=2^16, l=24, full-slot bootstrapping keys) alone has a 44 GB footprint; training steps die in weight bootstrapping (OOM/swap). Details: `spikes/reboot/RUN.md`.

| item | value |
|---|---|
| epoch time (this machine) | not measurable (OOM) |
| accuracy (this machine) | not measurable |
| memory (this machine) | >= 44 GB after context creation (36 GB RAM) |
| paper: eMLP-1 iteration time / peak memory | 190.32 s / 76.07 GB (2x Xeon 5318S, 384 GB) |
| paper-derived epoch (100 it, 4800 samples) / full MNIST (1250 it) | ~5.3 h / ~66 h (arithmetic, not measured) |
| paper: MNIST test acc eMLP-1/2/3 (plaintext twin) | 94.61 / 96.39 / 96.77 % |
| configs (paper Table 3/8) | `crates/hekit-bench/configs/reboot_emlp{1,2,3}.json`: N 2^16/2^17/2^17, l 24/27/29, Δ 49/59/59 bits, batch 48, lr 0.001/0.001/0.0005, momentum 0.9; epochs=10 from repo script (not in paper) |

## Task 6: Cost model (RQ1)

Code: `spikes/costmodel/` (Python 3.11 + numpy):
- `costmodel.py`: closed-form op counts per layout.
- `slotsim.py`: slot-exact plaintext simulator with an op counter; it refuses to rotate or multiply a degree-2 ciphertext.
- `test_costmodel.py`: 48 tests pass, plus 1 strict xfail.
- `table.py`: generates the tables below from `crates/hekit-bench/configs/reboot_emlp1.json` (Task 5).

To reproduce, run `python table.py` and `pytest` in `spikes/costmodel/`. The docstrings carry the full derivations, with a source for every formula.

**Verdict (RQ1, spec §4.3.2):**
- **Under the spec metric at ReBoot's own setting (N = 2^16, B = 48), RQ1 is not met.**
  - KS_spec (rotations + relins + bootstraps × 502) ratio is **0.744**.
  - The literal spec metric (rotations + ct-ct mults + bootstraps × 502) gives **0.786**.
  - Both miss the ≤ 0.70 bar.
  - At this setting diag_batched also needs **2 ciphertexts per batch**, because 1024 (784 padded) × 48 > N/2.
- The only way to pass at B = 48 is single-iteration bootstrapping (0.65), and it does not count. ReBoot runs a 2-iteration
  Meta-BTS for bootstrap precision, so one iteration changes the precision and conflicts with 「同一精度」.
- Without bootstraps, diag_batched wins clearly at B ≥ 32 for both N: 0.45 (rot + relin) / 0.54 (rot + ct-ct) at (2^16, 48).
- The spec metric (≤ 0.70, both variants) is met only at **B ≥ 64**:
  - (64, 2^16) gives 0.64 / 0.68 and needs 2 ciphertexts per batch.
  - (64, 2^17) gives 0.67 / 0.68 against ReBoot as shipped.
  - (128, both N) gives 0.43–0.49 and needs 2 or 4 ciphertexts per batch.
- **(64, 2^17) depends on the ReBoot baseline.** It is the only pass that satisfies §4.3.3's d × B ≤ N/2 (1 ciphertext
  per batch). ReBoot as shipped fills the slots, with r = 2048 at 2^17 (39 KS per sample). A generous ReBoot whose
  sumRows covers only the 1024 used rows costs 37 KS per sample (`custom_sumrows=True`). Against that baseline,
  (64, 2^17) gives KS_spec **0.691** (pass, 1.3% margin) and literal_spec **0.702 (fail)**.
  - So (64, 2^17) passes the literal spec metric only against ReBoot as shipped.
  - Against the generous baseline, no point with d × B ≤ N/2 passes the literal spec metric. The remaining literal
    passes, (64, 2^16)* and (128, both N)*, all need ≥ 2 ciphertexts per batch.
- Those passes rest on three things:
  - (a) the compact W/V re-layout, which keeps bootstraps at 2 per layer;
  - (b) BOOT_KS ≈ 502;
  - (c) B = 64/128 reaching the same accuracy as ReBoot's B = 48 (untested).
- **RQ1's spec metric is met only if the compact W/V re-layout works in CKKS.** Without it, diag_batched bootstraps
  2·(32 + 16) = 96 ciphertexts per update instead of 4, and the KS_spec ratio becomes **12.9** at (2^16, 48). It stays
  above 7 at every (B, N).
- **Interactive mode (M2 Task 30/32).** M2 trains interactively: the client decrypts and re-encrypts the weights after
  every update, so there is no bootstrap. The client re-encodes the expanded diagonals, so the server never compacts
  or unpacks. At (2^16, 48) that removes the update-phase unpack (7.62 KS per sample) and the gradient-phase lane
  shifts (0.96), about 51% of the 16.8.
  - Model: `interactive=True`. KS is **8.2** per sample (ratio **0.222**) when the server transposes with rotations
    (the default, the paper's point). With `client_transpose=True`, where the client also supplies W^T, it is
    7.9 (0.214).
  - **M2 Task 32's 10% check must compare against this interactive variant**, not the non-interactive 16.8.
  - M2 can settle only this no-bootstrap KS. The spec metric needs M3 or an explicit CKKS test of the compact
    re-layout: its bootstrap term, premise (a) compaction, and premise (b) BOOT_KS.
- Primary metric: rot + relin, not rot + ct-ct. A key switch happens at the relinearisation, not at the tensor product.
  With one relin per product (ReBoot, 2020) the two metrics coincide; under lazy relinearisation several tensor products
  share one key switch. The literal metric is reported alongside.
- The advantage comes from spreading per-update work (batch reduction, weight re-layout, transpose, bootstraps) over B
  samples. ReBoot pays 30 rotations per sample whatever B is: log2(r) forward + 2·log2(c) forward + log2(r) backward.

**Checks against published or measured counts:**
- 2020 paper Table 2 (6-3-1 net):
  - `row_2020` reproduces 38 mults / 63 rotations, every row matching.
  - `diag_2020` reproduces the 17 / 17 total.
  - The per-row rotations are a strict xfail. Table 2 counts the output dimension; Alg. 2 and §2 count the input dimension.
    The totals agree only because forward and backward swap in and out. Task 4 found the same.
- `diag_2020` in the "seal" convention matches Task 4's measured SEAL counts on 4-10-3: 35 ct-ct, 35 relin, 31 rot.
- ReBoot eMLP-1 has (r, c) = (1024, 32), the same as ReBoot's own log. It costs 30 rot + 7 relin per sample, plus 4
  bootstraps per mini-batch (W and V for 2 layers).
- diag_batched and ReBoot were checked in the simulator over one NAG step. diag_batched ran with compaction, without
  compaction, and interactive (server or client transpose). ReBoot ran as shipped and with custom sumRows.
  - The decoded forward output, W^Tδ, dW, W' and V' equal dense numpy.
  - The op counts equal the formulas phase by phase. That holds on 6 + 3 + 4 × 2 test networks for diag_batched and
    2 + 2 for ReBoot.

Config `reboot_emlp1.json`: reboot_emlp1 on MNIST, layers 784 -> 32 (poly_relu) -> 10 (none), ReBoot batch 48, ReBoot log_n 16, momentum 0.9.

### Verdict at ReBoot's own setting (N = 2^16, B = 48; diag_batched needs 2 ciphertext(s) per batch)

| metric | diag_batched | reboot | ratio | ratio without compaction | <= 0.70? |
|---|---|---|---|---|---|
| KS = rot + relin (primary) | 16.8 | 37.0 | **0.454** | 0.403 | yes |
| KS_spec = KS + boots x BOOT_KS | 58.6 | 78.8 | **0.744** | 12.925 | no |
| literal: rot + ct-ct mults | 20.1 | 37.0 | **0.544** | 0.493 | yes |
| literal_spec: rot + ct-ct + boots x BOOT_KS | 62.0 | 78.8 | **0.786** | 12.967 | no |
| KS, interactive (server transposes; M2 Task 32 mode, no bootstraps) | 8.2 | 37.0 | **0.222** | - | yes |
| KS, interactive (client supplies W^T) | 7.9 | 37.0 | **0.214** | - | yes |

Interactive rows: the client decrypts and re-encrypts W and V after every update (Task 30 ClientRefresher), so there is no bootstrap, no compaction and no unpack; ReBoot's KS has no bootstrap term either. They are the reference for M2 Task 32's 10% check, not the spec metric.

### Primary metric: KS = rotations + relins, per sample per epoch

| N | B | cts | diag ks | reboot ks | **ratio** | diag ks_spec | reboot ks_spec | **ratio** | ratio ks_spec without compaction | diag ks interactive | ratio ks interactive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2^16 | 16 | 1 | 44.6 | 37.0 | **1.20** | 170.1 | 162.5 | **1.05** | 18.77 | 18.8 | 0.51 |
| 2^16 | 32 | 1 | 23.8 | 37.0 | **0.64** | 86.5 | 99.8 | **0.87** | 15.31 | 10.9 | 0.29 |
| 2^16 | 48 (b0) | 2* | 16.8 | 37.0 | **0.45** | 58.6 | 78.8 | **0.74** | 12.92 | 8.2 | 0.22 |
| 2^16 | 64 | 2* | 12.6 | 37.0 | **0.34** | 44.0 | 68.4 | **0.64** | 11.18 | 6.2 | 0.17 |
| 2^16 | 128 | 4* | 7.0 | 37.0 | **0.19** | 22.7 | 52.7 | **0.43** | 7.27 | 3.8 | 0.10 |
| 2^17 | 16 | 1 | 47.6 | 39.0 | **1.22** | 205.1 | 196.5 | **1.04** | 19.45 | 18.8 | 0.48 |
| 2^17 | 32 | 1 | 25.3 | 39.0 | **0.65** | 104.0 | 117.8 | **0.88** | 16.24 | 10.9 | 0.28 |
| 2^17 | 48 (b0) | 1 | 17.9 | 39.0 | **0.46** | 70.4 | 91.5 | **0.77** | 13.94 | 8.3 | 0.21 |
| 2^17 | 64 | 1 | 13.4 | 39.0 | **0.34** | 52.8 | 78.4 | **0.67** | 12.21 | 6.2 | 0.16 |
| 2^17 | 128 | 2* | 7.1 | 39.0 | **0.18** | 26.7 | 58.7 | **0.46** | 8.16 | 3.5 | 0.09 |

### Literal spec metric: rotations + ct-ct mults, per sample per epoch

| N | B | cts | diag ks_literal | reboot ks_literal | **ratio** | diag ks_literal_spec | reboot ks_literal_spec | **ratio** | ratio ks_literal_spec without compaction |
|---|---|---|---|---|---|---|---|---|---|
| 2^16 | 16 | 1 | 48.1 | 37.0 | **1.30** | 173.6 | 162.5 | **1.07** | 18.80 |
| 2^16 | 32 | 1 | 25.5 | 37.0 | **0.69** | 88.3 | 99.8 | **0.89** | 15.32 |
| 2^16 | 48 (b0) | 2* | 20.1 | 37.0 | **0.54** | 62.0 | 78.8 | **0.79** | 12.97 |
| 2^16 | 64 | 2* | 15.1 | 37.0 | **0.41** | 46.5 | 68.4 | **0.68** | 11.21 |
| 2^16 | 128 | 4* | 9.9 | 37.0 | **0.27** | 25.6 | 52.7 | **0.49** | 7.32 |
| 2^17 | 16 | 1 | 51.1 | 39.0 | **1.31** | 208.6 | 196.5 | **1.06** | 19.47 |
| 2^17 | 32 | 1 | 27.0 | 39.0 | **0.69** | 105.8 | 117.8 | **0.90** | 16.26 |
| 2^17 | 48 (b0) | 1 | 19.0 | 39.0 | **0.49** | 71.5 | 91.5 | **0.78** | 13.96 |
| 2^17 | 64 | 1 | 14.3 | 39.0 | **0.37** | 53.6 | 78.4 | **0.68** | 12.22 |
| 2^17 | 128 | 2* | 8.3 | 39.0 | **0.21** | 28.0 | 58.7 | **0.48** | 8.18 |

`ratio` = diag_batched / reboot at the same N and B. `cts` = diag_batched ciphertexts per mini-batch; `*` = d_pad*B > N/2 (more than one). BOOT_KS = key switches of one EvalBootstrap in ReBoot's configuration (2 iterations): 502 at N = 2^16, 630 at N = 2^17.

- ratio <= 0.7 under ks: (B=32, N=2^16), (B=48, N=2^16)*, (B=64, N=2^16)*, (B=128, N=2^16)*, (B=32, N=2^17), (B=48, N=2^17), (B=64, N=2^17), (B=128, N=2^17)*. Best 0.181 at (B=128, N=2^17).
- ratio <= 0.7 under ks_spec: (B=64, N=2^16)*, (B=128, N=2^16)*, (B=64, N=2^17), (B=128, N=2^17)*. Best 0.431 at (B=128, N=2^16).
- ratio <= 0.7 under ks_spec (no compaction): none. Best 7.266 at (B=128, N=2^16).
- ratio <= 0.7 under ks_literal: (B=32, N=2^16), (B=48, N=2^16)*, (B=64, N=2^16)*, (B=128, N=2^16)*, (B=32, N=2^17), (B=48, N=2^17), (B=64, N=2^17), (B=128, N=2^17)*. Best 0.213 at (B=128, N=2^17).
- ratio <= 0.7 under ks_literal_spec: (B=64, N=2^16)*, (B=128, N=2^16)*, (B=64, N=2^17), (B=128, N=2^17)*. Best 0.477 at (B=128, N=2^17).
- ratio <= 0.7 under ks_literal_spec (no compaction): none. Best 7.320 at (B=128, N=2^16).
- ratio <= 0.7 under ks interactive: (B=16, N=2^16), (B=32, N=2^16), (B=48, N=2^16)*, (B=64, N=2^16)*, (B=128, N=2^16)*, (B=16, N=2^17), (B=32, N=2^17), (B=48, N=2^17), (B=64, N=2^17), (B=128, N=2^17)*. Best 0.089 at (B=128, N=2^17).

### N = 2^17: ReBoot as shipped (slot-filling r = 2048) vs generous ReBoot (custom sumRows, r = 1024)

| B | cts | reboot ks shipped / generous | ratio ks | ratio ks_spec | ratio ks_literal | ratio ks_literal_spec | ratio ks interactive |
|---|---|---|---|---|---|---|---|
| 16 | 1 | 39.0 / 37.0 | 1.220 / **1.285** | 1.044 / **1.054** | 1.309 / **1.380** | 1.061 / **1.072** | 0.482 / **0.508** |
| 32 | 1 | 39.0 / 37.0 | 0.648 / **0.683** | 0.883 / **0.899** | 0.693 / **0.731** | 0.898 / **0.914** | 0.280 / **0.295** |
| 48 (b0) | 1 | 39.0 / 37.0 | 0.458 / **0.483** | 0.769 / **0.786** | 0.488 / **0.514** | 0.782 / **0.799** | 0.212 / **0.224** |
| 64 | 1 | 39.0 / 37.0 | 0.343 / **0.362** | 0.673 / **0.691** | 0.366 / **0.386** | 0.684 / **0.702** | 0.159 / **0.168** |
| 128 | 2* | 39.0 / 37.0 | 0.181 / **0.191** | 0.456 / **0.472** | 0.213 / **0.224** | 0.477 / **0.494** | 0.089 / **0.094** |

Cells: ratio vs shipped / **ratio vs generous**. The generous baseline replicates ReBoot's r x c matrix twice so that sumRows needs only log2(1024) = 10 rotations; it is not what ReBoot ships.

- ratio <= 0.7 under ks_spec vs generous ReBoot at N = 2^17: (B=64), (B=128)*.
- ratio <= 0.7 under ks_literal_spec vs generous ReBoot at N = 2^17: (B=128)*.

### Sensitivity of KS_spec ratio to BOOT_KS (same B)

| N | BOOT_KS | B=16 | B=32 | B=48 | B=64 | B=128 |
|---|---|---|---|---|---|---|
| 2^16 | 0 (= KS) | 1.20 | 0.64 | 0.45 | 0.34 | 0.19 |
| 2^16 | 251 (1 iteration) | 1.08 | 0.81 | 0.65 | 0.54 | 0.33 |
| 2^16 | 502 (ReBoot: 2 iterations) | 1.05 | 0.87 | 0.74 | 0.64 | 0.43 |
| 2^16 | 1004 (2x) | 1.03 | 0.92 | 0.83 | 0.76 | 0.56 |
| 2^17 | 0 (= KS) | 1.22 | 0.65 | 0.46 | 0.34 | 0.18 |
| 2^17 | 315 (1 iteration) | 1.07 | 0.82 | 0.68 | 0.56 | 0.35 |
| 2^17 | 630 (ReBoot: 2 iterations) | 1.04 | 0.88 | 0.77 | 0.67 | 0.46 |
| 2^17 | 1260 (2x) | 1.02 | 0.93 | 0.85 | 0.78 | 0.59 |

### Per-phase key switches per sample (N = 2^16)

| layout | B | forward | transpose | backward | gradient | update | bootstraps/sample |
|---|---|---|---|---|---|---|---|
| reboot | 48 | 23.00 | 0.00 | 12.00 | 2.00 | 0.00 | 0.083 |
| diag_batched | 32 | 0.97 | 0.47 | 0.34 | 10.56 | 11.44 | 0.125 |
| diag_batched | 64 | 0.97 | 0.23 | 0.34 | 5.34 | 5.72 | 0.062 |
| diag_batched | 48 | 1.29 | 0.31 | 0.46 | 7.12 | 7.62 | 0.083 |
| diag_batched interactive | 48 | 1.29 | 0.31 | 0.46 | 6.17 | 0.00 | 0.000 |

### Reference rows (N = 2^16, KS per sample)

| layout | B | rotations | relins | KS | bootstraps/sample |
|---|---|---|---|---|---|
| reboot | 48 | 30.0 | 7.0 | 37.0 | 0.083 |
| reboot (lazy relin, hypothetical) | 48 | 30.0 | 5.0 | 35.0 | 0.083 |
| diag_2020 (seal convention) | 1 | 1650.0 | 1644.0 | 3294.0 | 1632.000 |
| row_2020 | 1 | 26460.0 | 118.0 | 26578.0 | 84.000 |
| diag_batched | 1 | 463.0 | 58.0 | 521.0 | 4.000 |

The reference rows at B = 1 are the 2020 algorithms on the eMLP-1 net, without batching.

**Modelling assumptions**
1. **Metric.** Every rotation and every relinearisation counts as one key switch, hoisted or not and at any level.
   Hoisting helps time, not count; BSGS reduces the count. The literal spec metric is reported as well.
2. **BOOT_KS** is one OpenFHE 1.2.1 EvalBootstrap in ReBoot's configuration: level budget [2,2], sparse ternary secret,
   num_iterations = 2, which makes it a Meta-BTS of two full bootstraps. That is 502 at N = 2^16 and 630 at 2^17.
   Per bootstrap at N = 2^16 (2^15 slots):
   - CoeffsToSlots + SlotsToCoeffs: 218 rotations (109 + 109), counted by enumerating the code's rotation-offset
     tables. Baby offsets are centred, so none is zero. One giant offset wraps to 0 mod N/2 and is skipped.
   - Plus 1 conjugation.
   - Plus 2 EvalMods of about 16 relins each, an **estimate** from OpenFHE's Paterson–Stockmeyer cost formula
     (about 13% of the total).
   - The sensitivity table above covers 0, ½×, 1× and 2×.
3. **Equal setup.** Both layouts get the same N, depth budget and bootstrap configuration. Neither bootstraps
   activations, which holds for 1-hidden-layer eMLP-1. Both bootstrap W and V (NAG) once per mini-batch.
4. **ReBoot model.** Taken from its paper (§3 eq. 4–5, §4.2.1, §4.3) and its code (`packing.cpp`, `nn.cpp`,
   `cryptocontext.py`, OpenFHE `EvalSumRows`/`EvalSumCols`): one sample per ciphertext, no hoisting, a relin after every
   product. ReBoot is GPL-3.0. Its code was read to count operations only; no ReBoot code is copied into hekit.
   - A hypothetical lazy-relin ReBoot (35 KS per sample) is listed as a stronger baseline.
   - Multi-sample packing is impossible for this net at N = 2^16, because r·c = 1024·32 = N/2.
   - At N = 2^17 the main baseline follows ReBoot's slot-filling rule, r = N/2 / c = 2048: 11 sumRows rotations,
     39 KS per sample.
   - The generous variant (`custom_sumrows=True`) uses r = 1024. The client replicates the matrix twice, so sumRows
     takes 10 rotations, giving 37 KS per sample. It is in the N = 2^17 table above, and it flips the literal-spec
     verdict at (64, 2^17).
5. **diag_batched layout:**
   - Feature-major `slot = j·B' + b`, stored S-periodically; dims padded to powers of two (784 → 1024, 10 → 16).
   - BSGS mat-vec over m = min(d_in, d_out) diagonals, with a fold for wide layers and free replication for tall ones.
   - The transpose rotates the encrypted weight diagonals, one rotation each (m − 1 per update).
   - The gradient reuses the forward's baby-step rotations and reduces over the batch (log2 B').
   - Then comes the compact re-layout: pack into ceil(m·L/S) W and V ciphertexts, run NAG, bootstrap, and unpack and
     replicate for the next forward.
   - `interactive=True` (M2) drops the compaction, the unpack and the bootstraps. NAG runs on the expanded diagonals,
     and the client reads lane 0 back at re-encryption. The client only decrypts, permutes or replicates slots, and
     re-encrypts; every arithmetic op, including the batch reduction, stays on the server.
6. Client-side encodings are free (x and y in either layout, initial weights). Plaintext/scalar products and
   additions are not key switches.

**Limitations / open concerns**
- **BSGS needs n2 relins, not the brief's 1.** A giant-step rotation acts on a degree-2 sum of tensor products, so it
  must be relinearised first. One relin is only possible with no BSGS (n2 = 1). The simulator enforces this.
- **At B = 1, diag_batched is worse than the 2020 diagonal method:** 80 vs 43 key switches per step on the Table 2 net,
  and 521 vs ReBoot's 37 per sample on eMLP-1. Its per-update re-layout only pays off when amortised over B.
- **KS_spec depends on the unbuilt compact re-layout.** It is verified only in the plaintext simulator, not in CKKS.
  Without it, the ratio is 7–19 (column "without compaction"). Its extra cost is 2m masks, about 2m rotations, and
  2 extra levels per update. Whether that fits the depth budget and the precision target has to be shown in CKKS.
- **The EvalMod part of BOOT_KS is estimated**, not counted from a run (ReBoot OOMs on this machine, see Task 5).
- **Accuracy at large B is untested.** Every spec-metric pass needs B ≥ 64, while ReBoot trains eMLP-1 with B = 48.
- The count metric ignores how expensive each key switch is (level, dnum) and ignores rotation-key memory.
  diag_batched needs more distinct rotation keys.
- Unmodelled diag_batched optimisations could only lower its cost: batched reduction and replication of several
  diagonals at once, and storage with fewer periods.
