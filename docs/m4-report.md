# M4 報告(下書き、2026-11-05)

M4(CUDA バックエンド、Task 43〜50)の報告の下書き。現時点では Task 47 の表の骨組みだけを置く。測定値の列は quadruped-gpu での実行(R-47-x)待ち。

## 1. GPU 対 CPU 対 HEonGPU(n16_train、Task 47 Step 5)

**目標(ゲートではない): hekit GPU が HEonGPU の 2 倍以内。** 比較できる HEonGPU の公表値は §1.2 に記した 1 つ(relinearize)しか見つかっていない。その 1 つは名目 1.81 倍、素数の本数をそろえると約 2.0 倍で、目標のちょうど境目にある。

測定条件(R-47-2、2026-11-05、commit ce4730a = Task 47 の feff7ed と同じコード、`results/quadruped/cuda_ops_r47.log`、`results/quadruped/raw/R-47-2/`):
- GPU: quadruped-gpu の RTX 4090(sm_89、24083 MiB、CUDA 13.1、NVRTC で実行時コンパイル、`--fmad=false`)。GPU は他プロセスと共有で、測定開始時の空きは 2889 MiB。
- CPU: 同じサーバの **i9-13900KF**(32 論理 CPU)。`cuda_ops` ベンチは `parallel` 機能(rayon)付きで、**1 スレッドと 12 スレッド**で測った。SLAM ジョブが 4 コアを使うので 12 スレッドで止めた(brief の「Mac 1 スレッド、サーバ 32 スレッド」からの逸脱)。nice 10、開始時 load 2.13。
- 値は criterion の中央値(10 サンプル、flat sampling、warm-up 1 s)。GPU は 1 反復ごとに `CudaDevice::sync` で完了を待つ(キューに積む時間ではなく遅延を測る)。

### 1.1 表(ms、中央値)

`_L9` / `_L5` の行は、入力をレベル 9 / 5 に落とし、鍵をそのレベル以下用に切り詰めた(capped key: n16_train の全鍵 288 MiB に対し、L9 で 42 MiB、L5 で 20 MiB)測定である。**そのレベルでの演算としては忠実**(切り詰めた鍵は、そのレベルでは全鍵と同じ計算をする)。ただし最上位レベル(31)の値ではない。CPU は全鍵のまま同じレベルで測った(レベル l の鍵切替は l 以下の limb と桁しか読まないので、仕事量は同じ)。

| 演算 | レベル | GPU 鍵メモリ | hekit GPU | サーバ CPU(i9-13900KF)1 thr | サーバ CPU 12 thr | GPU 対 CPU(1 / 12 thr) | HEonGPU(RTX 4090) | GPU / HEonGPU |
|---|---|---|---|---|---|---|---|---|
| add | 31 | なし | 0.111 | 15.67 | 14.61 | 141× / 132× | 無し | — |
| relinearize | 31 | 288 MiB | 4.671 | 639.6 | 240.1 | 137× / 51× | 2.582(ℓ = 36、§1.2) | **1.81×(名目。ℓ = 36 対 32 の差を補正すると約 2.0×)** |
| mul + relin + rescale | 31 | 288 MiB | 5.823 | 725.2 | 337.7 | 125× / 58× | 無し | — |
| rotate(1) | 31 | 288 MiB | 3.990 | 622.0 | 223.9 | 156× / 56× | 無し | — |
| hoisted_rotate × 16 | 31 | 16 × 288 MiB | skipped: needs 5580 MiB(2377 MiB free) | 3765 | 689.5 | — | 無し | — |
| hoisted_rotate × 16(capped) | 9 | 16 × 42 MiB | 4.984 | 1005 | 172.9 | 202× / 35× | 無し | — |
| matvec_enc(d = 64) | 31 | 15 × 288 MiB | skipped: needs 7276 MiB(1929 MiB free) | 12938 | 2325 | — | 無し | — |
| matvec_enc(d = 64、capped) | 9 | 15 × 42 MiB | 16.12 | 2523 | 547.8 | 157× / 34× | 無し | — |
| matvec_enc(d = 64、capped) | 5 | 15 × 20 MiB | 9.509 | 1391 | 314.6 | 146× / 33× | 無し | — |
| matvec_enc × 4、1 ストリーム(逐次) | 5 | 15 × 20 MiB | 34.99(sd 0.07) | — | — | — | 無し | — |
| matvec_enc × 4、4 ストリーム(`matvec_enc_batch`) | 5 | 15 × 20 MiB | **41.67(sd 2.12)、逐次より 19% 遅い** | — | — | — | 無し | — |

- **relinearize の HEonGPU 比**: HEonGPU の値は Q 素数 36 本(ℓ = 36、特別素数 4 本、I = 4、43 bit 精度)の構成で、n16_train は 32 本。名目では 4.671 / 2.582 = 1.81 倍。鍵切替のコストはこの大きさでほぼ ℓ に比例するとして 32 本に換算すると、HEonGPU は約 2.30 ms で、比は約 2.0 倍になる。**パラメータが同一ではないので、目標の判定は参考値**。
- 参考(Mac M3 Pro、M1 報告 §1、1 スレッド、parallel 機能無し、最上位レベル): add 1.85、mul + relin + rescale 302.7、rotate 256.9、hoisted × 16 1569、matvec_enc(d = 64)5711 ms。サーバ CPU の 1 スレッドは Mac の 2〜8 倍遅い(E コアに載った可能性、ピン留めなし。M3 の Iris の比較と同じ傾向)。

### 1.1a マルチストリーム(`matvec_enc_batch`)は n16_train では逐次より遅い

R-47-2 で、同じ 4 つの積を 4 ストリームに分けると 41.67 ms、1 ストリームで逐次に流すと 34.99 ms だった(逐次 = 9.51 ms × 3.68。ホスト側の待ちは逐次でも既に隠れている)。

- **理由: 1 つの積で既にデバイスが埋まっている。** `gpu/add`(L31)は約 96 MiB を 110.9 µs で動かしており、約 0.9 TB/s、RTX 4090 のピーク帯域(1.01 TB/s)の約 90% にあたる。N = 2^16 では要素ごとのカーネルも帯域律速で、占有率も満杯なので、ストリームを増やしてもスループットは増えない。
- 増えたのはバッチ側だけの固定費である。呼び出しごとのワーカースレッド 4 本、約 480 MiB の作業領域の確保と解放(メモリプールの解放閾値が 0 だったので、同期のたびに物理メモリを OS に返して再マップしていた)、4 つ同時の作業領域。バッチの sd(2.1 ms)が逐次(0.07 ms)より大きいのもこれと合う。
- **Task 47 の修正 1 回目(fix round 1)で変えたこと**: デバイス生成時にプールの解放閾値を `u64::MAX` にした(`CudaDevice::trim_pool` で返せる)。作業領域の仮確保をやめ、空き容量の確認だけにした(他テナントのメモリを一時的にでも取らない)。スレッドを使わず、呼び出しスレッドから各ストリームに順に積むようにした。再測定は R-47-3(1 / 2 / 4 ストリーム、任意で nsys)。
- **Task 48 への指示: 再測定で利得が出るまで、訓練ステップでは逐次の `matvec_enc` を使う。** 小さい積(低レベルの末尾、seal_2020 程度の大きさ)では効く可能性があるが、まだ何も測っていない。

### 1.2 HEonGPU の公表値の出典

読んだのは論文・README・公式文書だけ(コードは読んでいない、HEonGPU は Apache-2.0)。

1. **Özcan, Savaş, "GPU Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes"**(IACR ePrint 2025/124、2024 年 11 月版、HEonGPU に統合された実装)。
   - Table 2: GeForce RTX 4090(24 GB)、CUDA 12.4。
   - **Table 3**(CKKS の relinearization、GPU、µs): log N = 16、43 bit 精度、I = 4(桁あたり 4 素数 = n16_train の α)、Method II(hybrid)、(I, ℓ̃, ℓ) = (4, 40, 36)、level 35 で **2581.8 µs**。n16_train は ℓ = 32、特別素数 4 本(ℓ̃ = 36)、dnum 8 なので、素数の本数が約 1 割少ない。同じ表の 36 bit 精度、(4, 46, 42) は 3458.4 µs。
   - Table 5: CKKS、log N = 16 の GPU relinearization は Method I 9446.2 / II 3458.4 / III 3798.2 µs(36 bit 精度の行)。SEAL(CPU、Ryzen 9 7950X3D)の Method I は 1,215,753 µs。
   - add・mul・rescale・rotate・hoisted rotation・行列ベクトル積の RTX 4090 の値は、この論文には無い。
2. **HEonGPU の README と公式文書の Appendix "Performance Benchmarks"**(github.com/Alisah-Ozcan/HEonGPU、heongpu.readthedocs.io): RTX 4090 の表は CKKS bootstrapping(例: N = 2^16 の Regular で 109.28 ms)と TFHE だけ。個々の CKKS 演算の値は無い。
3. **Özcan, Savaş, "HEonGPU: a GPU-based Fully Homomorphic Encryption Library 1.0"**(IACR ePrint 2024/1543): **本文は取得できなかった**(eprint.iacr.org の PDF が 403 / ボット確認を返した)。そのため、この論文の表番号と値は未確認。手で PDF を開いて、RTX 4090・N = 2^16 の演算の表があれば追記する(要確認事項)。

表の「無し」は、上の 1 と 2 に RTX 4090・N = 2^16 の値が無いことを示す。3 は未確認。

### 1.3 実装状況(Task 47)

- `CudaEvaluator`(`hekit_ckks::HeOps` の実装)。CPU と同じ検査順・エラー種別・`OpCounter`。結果は CPU とビット一致する(matvec を含む。BSGS の手順は CPU と共通の `hekit_cuda::matvec` にある)。
- RQ3 の 3 フラグ(hoist / bsgs / lazy_relin)は CPU と同じ意味。d = 16 の 8 通りすべてで `EvalConfig::matvec_rotations` / `matvec_enc_relins` と一致することを GPU テストで確かめる。
- `matvec_enc_batch`: 独立な積を呼び出しスレッドから各ストリームに積む(§1.1a のとおり、n16_train では逐次より遅い)。開始前に各ストリームの作業領域が空き容量に収まるか確かめ、収まらなければ何も確保せずに `Device` エラーを返す。
- 小さな表(内積の桁アドレス、スカラー定数)は、ストリームごとの pinned 転送リングで非同期に送る。

## 2. 未記入

Task 48〜50(RQ3 の表、GPU 上の訓練、まとめ)はこの後に追記する。
