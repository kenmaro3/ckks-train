# M4 報告(下書き、2026-11-05)

M4(CUDA バックエンド、Task 43〜50)の報告の下書き。現時点では Task 47 の表の骨組みだけを置く。測定値の列は quadruped-gpu での実行(R-47-x)待ち。

## 1. GPU 対 CPU 対 HEonGPU(n16_train、Task 47 Step 5)

**目標(ゲートではない): hekit GPU が HEonGPU の 2 倍以内。** 比較できる HEonGPU の公表値は §1.2 に記した 1 つ(relinearize)しか見つかっていない。

測定条件(予定):
- GPU: quadruped-gpu の RTX 4090(sm_89、24 GB、CUDA 13.1、NVRTC で実行時コンパイル、`--fmad=false`)。GPU は他プロセスと共有で、空きは約 1.8〜3 GB。
- CPU: 同じサーバ。`cuda_ops` ベンチは `parallel` 機能(rayon)付きで、1 スレッドと 12 スレッドを測る。SLAM ジョブが 4 コアを使うので 12 スレッドで止める。
- 値は criterion の中央値(10 サンプル、flat sampling、warm-up 1 s)。GPU は 1 反復ごとに `CudaDevice::sync` で完了を待つ(キューに積む時間ではなく遅延を測る)。

### 1.1 表(ms)

`_L9` / `_L5` の行は、入力をレベル 9 / 5 に落とし、鍵をそのレベル以下用に切り詰めた(capped key: n16_train の全鍵 288 MiB に対し、L9 で 42 MiB、L5 で 20 MiB)測定である。**そのレベルでの演算としては忠実**(切り詰めた鍵は、そのレベルでは全鍵と同じ計算をする)。ただし最上位レベル(31)の値ではない。CPU も同じレベル・同じラベルで測る。

| 演算 | レベル | GPU 鍵メモリ | hekit GPU | hekit CPU 1 thr | hekit CPU 12 thr | HEonGPU(RTX 4090) | GPU / HEonGPU |
|---|---|---|---|---|---|---|---|
| add | 31 | なし | pending R-47-2 | pending R-47-2 | pending R-47-2 | 無し | — |
| relinearize | 31 | 288 MiB | pending R-47-2 | pending R-47-2 | pending R-47-2 | 2.582 ms(§1.2、パラメータ差あり) | pending |
| mul + relin + rescale | 31 | 288 MiB | pending R-47-2 | pending R-47-2 | pending R-47-2 | 無し | — |
| rotate(1) | 31 | 288 MiB | pending R-47-2 | pending R-47-2 | pending R-47-2 | 無し | — |
| hoisted_rotate × 16 | 31 | 16 × 288 MiB | skipped: needs 5580 MiB(見込み) | pending R-47-2 | pending R-47-2 | 無し | — |
| hoisted_rotate × 16(capped) | 9 | 16 × 42 MiB | pending R-47-2 | pending R-47-2 | pending R-47-2 | 無し | — |
| matvec_enc(d = 64) | 31 | 15 × 288 MiB | skipped: needs 7276 MiB(見込み) | pending R-47-2 | pending R-47-2 | 無し | — |
| matvec_enc(d = 64、capped) | 9 | 15 × 42 MiB | pending R-47-2(1536 MiB、空き次第) | pending R-47-2 | pending R-47-2 | 無し | — |
| matvec_enc(d = 64、capped) | 5 | 15 × 20 MiB | pending R-47-2 | pending R-47-2 | pending R-47-2 | 無し | — |
| matvec_enc × 4、4 ストリーム(batch) | 5 | 15 × 20 MiB | pending R-47-2 | — | — | 無し | — |
| matvec_enc × 4、1 ストリーム(逐次) | 5 | 15 × 20 MiB | pending R-47-2 | — | — | 無し | — |

GPU メモリの必要量(見込み)は、鍵 + 入力 + 作業領域 + 40 MiB で、NTT 表の 72 MiB は別。ベンチは各演算の前に空きを確かめ、足りなければ `gpu/<op>: skipped: needs X MiB (Y MiB free)` と表示して飛ばす。

参考(Mac、M1 報告 §1、1 スレッド、parallel 機能無し): add 1.85、mul + relin + rescale 302.7、rotate 256.9、hoisted × 16 1569、matvec_enc(d = 64)5711 ms。

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
- `matvec_enc_batch`: 独立な積をストリームごとのワーカースレッドで並列に流す。開始前に各ストリームの作業領域を確保できるか確かめる。足りなければ `Device` エラーを返す。
- 小さな表(内積の桁アドレス、スカラー定数)は、ストリームごとの pinned 転送リングで非同期に送る。

## 2. 未記入

Task 48〜50(RQ3 の表、GPU 上の訓練、まとめ)はこの後に追記する。
