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
