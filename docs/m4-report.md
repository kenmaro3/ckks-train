# M4 報告(測定待ち、2026-11-06)

M4(CUDA バックエンド、Task 43〜50)の現状報告。Task 47 の演算ベンチ(R-47-2)と GPU 演算の一致確認は回収済み。Task 48 の訓練・RQ3 ランナーと provenance 修正はレビュー済みで、CPU smoke の計数を確認した。Task 49 の GPU ブートストラップはコード準備・レビュー・CPU 検証を完了した。**GPU 訓練、RQ3 のサーバ表、修正後のバッチ再測定、GPU ブートストラップの精度・速度・ピークメモリは未計測で、M4 全体の合格は未判定**である。Task 50 は報告準備の段階であり、欠測を埋めた最終報告ではない。

測定待ちは FULL-1 → FULL-2 → R-47-3 → R-48-1〜4 の順を維持する。Task 48 は既存担当の実装・修正レビュー完了後に引き継ぐ。実装・測定・判定の対応は §4、未決事項は §5 にまとめる。

## 1. GPU 対 CPU 対 HEonGPU(n16_train、Task 47 Step 5)

**目標(ゲートではない): hekit GPU が HEonGPU の 2 倍以内。** 比較できる HEonGPU の公表値は §1.2 に記した 1 つ(relinearize)しか見つかっていない。その 1 つは名目 1.81 倍である。素数本数 36→32 の線形コストモデルによる外挿は約 2.0 倍となるが、同一パラメータを測った結果ではなく、目標の合否は確定しない。

測定条件(R-47-2、2026-11-05、commit ce4730a = Task 47 の feff7ed と同じコード、`results/quadruped/cuda_ops_r47.log`、`results/quadruped/raw/R-47-2/`):

- GPU: quadruped-gpu の RTX 4090(sm_89、24083 MiB、CUDA 13.1、NVRTC で実行時コンパイル、`--fmad=false`)。GPU は他プロセスと共有で、測定開始時の空きは 2889 MiB。
- CPU: 同じサーバの **i9-13900KF**(32 論理 CPU)。`cuda_ops` ベンチは `parallel` 機能(rayon)付きで、**1 スレッドと 12 スレッド**で測った。共有ホストでの上限を 12 スレッドとした(brief の「Mac 1 スレッド、サーバ 32 スレッド」からの逸脱)。実行前後の SLAM プロセス観測は無し(`raw/R-47-2/slam_{before,after}.txt`)、他の hekit / ReBoot ジョブは終了後。nice 10、開始時 load 2.13。
- 値は criterion の中央値(10 サンプル、flat sampling、warm-up 1 s)。GPU は 1 反復ごとに `CudaDevice::sync` で完了を待つ(キューに積む時間ではなく遅延を測る)。

### 1.1 表(ms、中央値)

`_L9` / `_L5` の行は、入力をレベル 9 / 5 に落とし、鍵をそのレベル以下用に切り詰めた(capped key: n16_train の全鍵 288 MiB に対し、L9 で 42 MiB、L5 で 20 MiB)測定である。**そのレベルでの演算としては忠実**(切り詰めた鍵は、そのレベルでは全鍵と同じ計算をする)。ただし最上位レベル(31)の値ではない。CPU は全鍵のまま同じレベルで測った(レベル l の鍵切替は l 以下の limb と桁しか読まないので、仕事量は同じ)。

| 演算 | レベル | GPU 鍵メモリ | hekit GPU | サーバ CPU(i9-13900KF)1 thr | サーバ CPU 12 thr | GPU 対 CPU(1 / 12 thr) | HEonGPU(RTX 4090) | GPU / HEonGPU |
|---|---|---|---|---|---|---|---|---|
| add | 31 | なし | 0.111 | 15.67 | 14.61 | 141× / 132× | 無し | — |
| relinearize | 31 | 288 MiB | 4.671 | 639.6 | 240.1 | 137× / 51× | 2.582(ℓ = 36、§1.2) | **1.81×(名目。36→32 本のモデル外挿で約 2.0×)** |
| mul + relin + rescale | 31 | 288 MiB | 5.823 | 725.2 | 337.7 | 125× / 58× | 無し | — |
| rotate(1) | 31 | 288 MiB | 3.990 | 622.0 | 223.9 | 156× / 56× | 無し | — |
| hoisted_rotate × 16 | 31 | 16 × 288 MiB | skipped: needs 5580 MiB(2377 MiB free) | 3765 | 689.5 | — | 無し | — |
| hoisted_rotate × 16(capped) | 9 | 16 × 42 MiB | 4.984 | 1005 | 172.9 | 202× / 35× | 無し | — |
| matvec_enc(d = 64) | 31 | 15 × 288 MiB | skipped: needs 7276 MiB(1929 MiB free) | 12938 | 2325 | — | 無し | — |
| matvec_enc(d = 64、capped) | 9 | 15 × 42 MiB | 16.12 | 2523 | 547.8 | 157× / 34× | 無し | — |
| matvec_enc(d = 64、capped) | 5 | 15 × 20 MiB | 9.509 | 1391 | 314.6 | 146× / 33× | 無し | — |
| matvec_enc × 4、1 ストリーム(逐次) | 5 | 15 × 20 MiB | 34.99(sd 0.07) | — | — | — | 無し | — |
| matvec_enc × 4、4 ストリーム(`matvec_enc_batch`) | 5 | 15 × 20 MiB | **41.67(sd 2.12)、逐次より 19% 遅い** | — | — | — | 無し | — |

- **relinearize の HEonGPU 比**: HEonGPU の値は Q 素数 36 本(ℓ = 36、特別素数 4 本、I = 4、43 bit 精度)の構成で、n16_train は 32 本。名目では 4.671 / 2.582 = 1.81 倍。鍵切替コストがこの大きさで ℓ に比例するという仮定で 36→32 本に外挿すると、HEonGPU は約 2.30 ms、比は約 2.0 倍になる。これは**コストモデルによる換算**であり、32 本での HEonGPU 実測値ではない。**パラメータが同一ではないので、目標の判定は参考値**。
- 参考(Mac M3 Pro、M1 報告 §1、1 スレッド、parallel 機能無し、最上位レベル): add 1.85、mul + relin + rescale 302.7、rotate 256.9、hoisted × 16 1569、matvec_enc(d = 64)5711 ms。サーバ CPU の 1 スレッドは Mac の 2〜8.5 倍遅い(多くの演算で 2.3〜2.4 倍、add だけ 8.5 倍。E コアに載った可能性、ピン留めなし。M3 の Iris の比較と同じ傾向)。

### 1.1a 修正前 R-47-2 のマルチストリームは逐次より遅い

R-47-2 で、同じ 4 つの積を 4 ストリームに分けると 41.67 ms、1 ストリームで逐次に流すと 34.99 ms だった(逐次 = 単発 9.51 ms の約 3.68 倍)。ホスト側の待ちの内訳はプロファイルで確認していない。

- **未確認の仮説: 単一の積でも帯域や実行資源が競合し、並行化の利得が小さい可能性がある。** `gpu/add`(L31)の約 96 MiB / 110.9 µs は約 0.9 TB/s という概算になる。ただし、これは add の転送量モデルから求めた有効帯域であり、matvec の帯域律速・GPU 占有率・全資源の飽和を測ったものではない。**単一 matvec が GPU を使い切るという説明は未確認**で、nsys 等のプロファイルが必要である。R-47-2 で直接観測したのはバッチの 19% 遅延である。
- バッチ側だけにある固定費が遅さを足していると考えられる(**仮説。プロファイルで未確認**)。呼び出しごとのワーカースレッド 4 本、約 480 MiB の作業領域の確保と解放、4 つ同時の作業領域である。メモリプールの解放閾値が 0(CUDA の既定)だったので、同期のたびに物理メモリを返して再マップしていた可能性がある。バッチの sd(2.1 ms)が逐次(0.07 ms)より大きいことはこの仮説と矛盾しない。R-47-3(と任意の nsys)で確かめる。
- **Task 47 の修正で変えたこと**(fix round 1 と 2):
  - 作業領域の仮確保をやめ、空き容量の確認だけにした(他テナントのメモリを一時的にでも取らない)。
  - スレッドを使わず、呼び出しスレッドから各ストリームに順に積むようにした。
  - プールの解放閾値を `CudaDevice` の設定にした。既定は **256 MiB**(`CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD`)。CUDA の解放閾値は、使用中の分も含めたプールの**予約量**の上限である。同期のたびにプールは、予約量が `max(使用中, 閾値)` 以下になるまで未使用分を返す。n16_train では鍵と入力だけで 256 MiB を超える(全鍵 1 本で 288 MiB)ので、**既定は n16_train ではほぼ閾値 0 と同じに働く**(一時領域は同期ごとに返され、再マップされる)。それでも共有 GPU の保護は保たれる。同期後の hekit の使用量は常に「使用中 + 256 MiB」以下で、過去のピークは残らない。
  - `set_pool_release_threshold(u64::MAX)` にすると、hekit のピーク使用量を `trim_pool` まで保持する(fix round 1 の設定。共有 GPU では他テナントを圧迫しうる)。
  - **Task 48 の選択肢(既定は変えない)**: n16_train で一時領域を再利用したいときは、(a) 鍵のアップロード後に `set_pool_release_threshold(pool_used_memory() + 256 MiB)` とする(鍵と入力に加えて最大 256 MiB の一時領域を保持)か、(b) `u64::MAX` にして下の `trim_pool(0)` の規則を守る。
  - 再測定 R-47-3 は 1 / 2 / 4 ストリームを、既定(256 MiB。**n16_train ではほぼ閾値 0**、出力の `[default 256 MiB, ~threshold 0 at n16_train]` 行)と `u64::MAX`(`[max]` 行)の両方の閾値で測る(任意で nsys)。
- **Task 48 への指示: 再測定で利得が出るまで、訓練ステップでは逐次の `matvec_enc` を使う。** 小さい積(低レベルの末尾、seal_2020 程度の大きさ)では効く可能性があるが、まだ何も測っていない。
- **Task 48 への指示(GPU メモリ): 訓練ループは各エポックの境目と、一度きりのピーク(鍵のアップロード、`matvec_enc_batch`、最上位レベルの大きな演算)の後に `CudaDevice::trim_pool(0)` を呼ぶ。** 保存済み観測では GPU は他テナント(約 21 GB を使用)と共有で、空きは 1.5〜3 GB だった。プールに残ったメモリは hekit が使っていなくても他テナントからは使えない。`cuda_ops` ベンチは各ブロックの前と最後に `trim_pool(0)` を呼び、空き容量は `CudaDevice::available_memory`(ドライバの空き + プールの未使用分)で判定する。

### 1.2 HEonGPU の公表値の出典

読んだのは論文・README・公式文書だけ(コードは読んでいない、HEonGPU は Apache-2.0)。

1. **Özcan, Savaş, "GPU Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes"**(IACR ePrint 2025/124、2024 年 11 月版、HEonGPU に統合された実装)。
   - Table 2: GeForce RTX 4090(24 GB)、CUDA 12.4。
   - **Table 3**(CKKS の relinearization、GPU、µs): log N = 16、43 bit 精度、I = 4(桁あたり 4 素数 = n16_train の α)、Method II(hybrid)、(I, ℓ̃, ℓ) = (4, 40, 36)、level 35 で **2581.8 µs**。n16_train は ℓ = 32、特別素数 4 本(ℓ̃ = 36)、dnum 8 なので、素数の本数が約 1 割少ない。同じ表の 36 bit 精度、(4, 46, 42) は 3458.4 µs。
   - Table 5: CKKS、log N = 16 の GPU relinearization は Method I 9446.2 / II 3458.4 / III 3798.2 µs(36 bit 精度の行)。SEAL(CPU、Ryzen 9 7950X3D)の Method I は 1,215,753 µs。
   - add・mul・rescale・rotate・hoisted rotation・行列ベクトル積の RTX 4090 の値は、この論文には無い。
2. **HEonGPU の README と公式文書の [Appendix "Performance Benchmarks"](https://heongpu.readthedocs.io/en/latest/appendix.html#performance-benchmarks)**: RTX 4090 の表は CKKS bootstrapping と TFHE で、個々の CKKS 演算の値は無い。2026-11-06 に計画責任者が一次資料で確認した Regular の条件は N = 2^16、32768 スロット、残り 9 レベル、5 回の中央値 **109.28 ms**。Task 49 の比較境界は §3.3。
3. **Özcan, Savaş, "HEonGPU: a GPU-based Fully Homomorphic Encryption Library 1.0"**(IACR ePrint 2024/1543): **本文は取得できなかった**(eprint.iacr.org の PDF が 403 / ボット確認を返した)。そのため、この論文の表番号と値は未確認。手で PDF を開いて、RTX 4090・N = 2^16 の演算の表があれば追記する(要確認事項)。

表の「無し」は、上の 1 と 2 に RTX 4090・N = 2^16 の値が無いことを示す。3 は未確認。

### 1.3 実装状況(Task 47)

- `CudaEvaluator`(`ckks_train_ckks::HeOps` の実装)。CPU と同じ検査順・エラー種別・`OpCounter`。R-47-1(`ce4730a`、`results/quadruped/hekit_cuda_task47_tests.txt`、`results/quadruped/raw/R-47-1/`)は 45 テスト関数が ok、うち 42 実行・3 SKIP。実行した演算・matvec の CPU とのビット一致を確認した。全鍵 matvec、全鍵 sequence、hoisted ×16 はメモリ不足で SKIP しており、その行の一致は未確認。BSGS の手順は CPU と共通の `ckks_train_cuda::matvec` にある。
- RQ3 の 3 フラグ(hoist / bsgs / lazy_relin)は CPU と同じ意味。d = 16 の 8 通りすべてで `EvalConfig::matvec_rotations` / `matvec_enc_relins` と一致することを R-47-1 の GPU テストで確認済み。
- `matvec_enc_batch`: 修正後は独立な積を呼び出しスレッドから各ストリームに積む。§1.1a の遅延値は修正前の R-47-2 の結果で、修正後の性能は R-47-3 待ち。開始前に各ストリームの作業領域が空き容量に収まるか確かめ、収まらなければ何も確保せずに `Device` エラーを返す。
- 小さな表(内積の桁アドレス、スカラー定数)は、ストリームごとの pinned 転送リングで非同期に送る。

## 2. GPU 上の訓練と RQ3 アブレーション(Task 48)

**状態(2026-11-06): 実装・provenance 修正・再レビューは完了(実装 `626160e`、修正 `0e07561`、smoke 再生成 `b1d1813`、実行依頼修正 `bd61e9b`)。CPU での確認は済み。GPU 上の訓練と、サーバでの RQ3 の表は未計測。** 直近の保存済みサーバ観測では他テナントが GPU を約 20.7 GB 使い、空きは 2〜3 GB だった。n16_train の全鍵での GPU 訓練(事前必要量の見積もり約 15.2 GiB、ピーク実測値ではない)は、GPU メモリが空くまで**未計測**のままとする(計画責任者の裁定: 未計測の行はそう書く)。

### 2.1 実装

- **バックエンド切り替え**: `mnist` と `iris` に `--backend cpu|cuda` を足した(`cuda` は `cuda` 機能付きのビルドが必要)。共通部分は `ckks_train_bench::backend`。暗号化・復号・エンコード・クライアント再暗号化・ブートストラップはホストに残し、GPU では次のように分ける(「レベル演算は GPU、ブートストラップは CPU」)。
  - 対話モードのクライアント再暗号化は `HostRefresher` の後ろで行う(ダウンロード、ホストで再暗号化、アップロード)。
  - 非対話モードのブートストラップは `HostBootstrapper` の後ろで行う。ホスト側の計数は GPU 評価器の `OpCounter` に足し込む。
  - GPU ブートストラップ(Task 49)は §3 の別実装である。現在の訓練ベンチへの接続はしておらず、非対話 GPU 訓練の時間には引き続き CPU の `HostBootstrapper` と転送を分けて記録する。
- **RQ3 のフラグ**: `--no-hoist --no-bsgs --no-lazy-relin`(`EvalConfig`)を `mnist`・`iris` の両モードに通した。
- **閉形式の照合(Task 43 M5)**: フラグを考慮するようにした。
  - 対話モード: `predict` の行列ベクトル積を `EvalConfig::matvec_rotations` / `matvec_enc_relins` で数える。8 通りすべてで実測と一致する(`tests/mnist_flags.rs`、`small`)。
  - 非対話モード: `predict_update_with` を使う。全オフで実測と一致する(`mini_boot`)。
  - コストモデルとその補正項は既定フラグのもの、と出力 JSON に明記した。
- **`HostRefresher` の N1/N2(Task 43 レビュー)**: `HostBootstrapper` と同じにそろえた。単体テストあり(`ckks-train-nn` `refresh.rs`)。
  - ホスト評価器が評価器 E 自身のときは、何もせずに `InvalidParams` を返す(アドレスで判定)。
  - 計数はアップロードが成功した後でだけ足す。
- **行列ベクトル積は逐次**: `--matvec sequential` が既定で、訓練に通っている唯一のモード。`--matvec batch` は理由を示して拒否する。
  - R-47-2 で 4 ストリームの `matvec_enc_batch` は逐次より 19% 遅かった。再測定 R-47-3 はまだ。
  - 1 つの訓練ステップの matvec は前の出力に依存するので、まとめられる独立な積が無い。
- **GPU メモリ**:
  - `trim_pool(0)` は次の時点で呼ぶ: 鍵のアップロード後、ネットワークのアップロード後、各エポックの境目、アブレーションの各構成の後。
  - `--gpu-pool after-keys` で解放閾値を「使用中 + 256 MiB」に上げられる(Task 47 で文書化した選択肢。既定は 256 MiB のまま)。
  - 必要量は鍵生成の前に見積もる(`mlp_gpu_need`: 鍵、表、常駐状態、作業領域。作業領域は開始レベルで数えるので多めに出る)。空きが足りなければ `SKIP` 行を出して終了コード 3 で終わる。iris だけは鍵生成の後に確かめる。
- **時間の分け方**: GPU では演算が非同期に積まれるので、最初のダウンロードがステップ全体を待ってしまう。そこで再暗号化の前にデバイスを同期し、その待ち時間をステップ側に数える(`SyncedRefresher`)。CPU では待ちはゼロで、Task 32 と同じ分け方になる。
- **レベルの上限 `--top-level L`(対話モード)**: ネットワーク、バッチ、クライアント再暗号化をレベル L に置き、GPU の鍵もレベル L 以下の分だけ上げる(`ClientRefresher::with_output_level`)。
  - パラメータと安全性は全レベルの場合と同じで、暗号文の limb が少ないだけである。
  - eMLP-1 の 1 ステップは、満杯のバッチで 7 レベルを使う(`W1'` がレベル 0 に着く)。端数のバッチは線形出力層の有効レーンのマスクで 1 レベル多く、8 レベルを使う。
- **CPU の経路は変わっていない**: 変更前のバイナリと比べた。
  - `mnist` の `small` 対話と `mini_boot` 非対話: 双子との差、計数、評価がビット一致。
  - `iris` の timing: コミット済みの `results/iris_1_timing.json` と、反復ごとの重み誤差がビット一致。

### 2.2 Step 1: GPU と CPU の訓練一致テスト(未実行、サーバ待ち)

`crates/ckks-train-bench/tests/mnist_cuda.rs` と `iris_cuda.rs` に置いた(どちらも `#[ignore]`、`--features cuda`)。

- mnist の 3 テストは、実 MNIST の subset 1024 で同じ対話訓練を `--backend cuda` と `--backend cpu` で走らせ、次を確かめる。GPU を先に走らせ、メモリ不足でスキップしたら CPU は走らせない。
  - **バッチごとの `OpCounter` が完全に一致する**(回転、relin、鍵切替、積、rescale、ブートストラップ)。
  - 復号した最終の重みとバイアスが 2^-12 以内で一致する。
- iris の 2 テストは、seal_2020 の対話 timing と mini_boot の非対話(ブートストラップは CPU)で同じことを確かめる。

| テスト | パラメータ | GPU メモリ(事前見積もり) | 実測状態 / 見込み |
|---|---|---|---|
| mnist `n16_train_full_keys` | n16_train、レベル 31、全鍵(288 MiB × 32) | 15560 MiB | 未実行。現状の空きでは SKIP 見込み |
| mnist `n16_train_level7` | n16_train、`--top-level 7`、鍵はレベル 7 以下(24 MiB × 32) | 2456 MiB | 未実行。空き次第で、足りなければ SKIP |
| mnist `small_insecure` | `small`(N = 2^12、安全でない)、d = 1024、B = 2、実データと 784-32-10 | 397 MiB | 未実行。容量上は実行見込み |
| iris seal_2020 対話 | seal_2020、B = 20、timing 6 反復 | 約 1.3 GiB | 未実行。容量上は実行見込み |
| iris mini_boot 非対話 | mini_boot(N = 2^10、安全でない)、1 エポック | 約 0.1 GiB | 未実行。容量上は実行見込み |

どの行が実行されたかを R-48-1 のログで確認する。**n16_train の Step 1 ゲートは `n16_train_level7` が実際に走り、計数と精度の検査を通った場合だけ満たす**。テストは SKIP でも Cargo 上 ok になるため、5/5 ok や終了コード 0 だけではゲート合格としない。`small_insecure` や iris の成功だけでも代用しない。

### 2.3 RQ3 アブレーションの表

`crates/ckks-train-bench/src/bin/ablation.rs` は、対話モードの eMLP-1 の 1 反復(784 → 32(z^2 + z)→ 10(線形)、既定 NAG、容量不足時の登録済み代替は SGD、1 バッチ 1 暗号文、クライアント再暗号化)を、8 通りの `EvalConfig` で測る。出力先は CLI の `--out` で指定する。登録した本番の出力先は下表の `results/quadruped/` 配下であり、smoke JSON と分ける。

- 鍵、初期ネットワーク、バッチは全構成で同じ。ウォームアップの後に計測反復を回し、中央値を取る。
- 時間は「ステップ」(サーバ側、デバイスの待ちを含む)と「再暗号化」(ホスト。フラグは影響しない)に分ける。
- 計数は 1 反復分。閉形式は「既定の実測 + 各 matvec のフラグ差分」で、全構成で照合する。
- 重みは既定構成との差を出す。

**この表が述べなければならないこと**(`ckks_train_ckks::ops` の "What the RQ3 table must state" をそのまま写す。`ablation` の出力 JSON の `caveats` にも同じ文がある):

1. **Definition of `bsgs = false`.** The giant rotation is not shared; the weights keep
   the same pre-rotated (BSGS) layout and the same Galois keys. Every product with a
   nonzero giant shift is finished (relinearized) and rotated on its own; the products of
   the zero-shift block are summed as with BSGS. Per matvec of dimension `d` (offset 0):
   `d - 1` rotations, of which only the `baby - 1` rotations of `x` can be hoisted; with
   `lazy_relin` `1 + d - baby` relins, without it `d`
   ([`EvalConfig::matvec_rotations`], [`EvalConfig::matvec_enc_relins`]). At `d = 16`
   (`baby = giant = 4`), as (hoistable rotations / individual rotations / relins):

   | form | hoistable rot. | individual rot. | relins (lazy) | relins (not lazy) |
   |---|---|---|---|---|
   | default BSGS | 3 | 3 | 4 | 16 |
   | `bsgs = false` (this library) | 3 | 12 | 13 | 16 |
   | textbook diagonal method (not implemented) | 15 | 0 | 1 | 16 |

   The textbook (Halevi-Shoup) diagonal method rotates `x` by all `d - 1` steps (all
   hoistable) and sums the products lazily; it needs un-pre-rotated diagonals, i.e.
   another weight layout and gradient form, so it is not implemented. The all-off
   configuration has the same counts in both forms (15 rotations, 16 relins at `d = 16`).
2. **Bias.** Against the strongest naive baseline (the textbook method with hoisting and
   lazy relinearization), the cost of the `bsgs = false` rows overstates what BSGS gains:
   the measured BSGS gain is an **upper bound**. Conversely, the marginal gains of `hoist`
   and `lazy_relin` measured inside `bsgs = false` configurations are **lower bounds** (only
   `baby - 1` rotations can be hoisted and only the zero-shift block summed lazily: e.g.
   `lazy_relin` saves 3 relins at `d = 16` instead of 15).
3. **`hoist` changes no [`OpCounter`] value.** The counter does not distinguish hoisted from
   individual rotations, so the effect of `hoist` shows only in wall-clock time, never in
   key-switch counts.
4. **`bsgs` does not touch the gradient's outer product** (`Dense::weight_grad_scaled`,
   `Dense::weight_grad_reduced`): its `(baby - 1) + (giant - 1)` input rotations are what
   the pre-rotated weight layout requires, not a BSGS choice; only `hoist` changes how they
   run. The interactive gradient multiplies each diagonal with one [`Evaluator::mul`]
   (nothing to sum lazily), so `lazy_relin` does not change it either.

**閉形式の鍵切替回数(n16_train、eMLP-1、B = 32、1 反復、`mnist --plan` のフラグ付き予測)**: CPU で 8 通りすべて実測と一致することを `small` で確かめた式による。`hoist` は数を変えないので 4 通りに分かれる。

| hoist / bsgs / lazy_relin | 回転 | relin | 鍵切替 | 1 サンプルあたり |
|---|---|---|---|---|
| * / 1 / 1(既定) | 571 | 62 | 633 | 19.781 |
| * / 1 / 0 | 571 | 114 | 685 | 21.406 |
| * / 0 / 1 | 610 | 101 | 711 | 22.219 |
| * / 0 / 0(全オフ) | 610 | 114 | 724 | 22.625 |

1 ステップの matvec は 3 つある(層 1 の順伝播 d = 32、層 2 の順伝播と逆伝播 d = 16)。勾配の外積(回転 `(baby - 1) + (giant - 1) + 2 m log B`)はフラグで変わらない。上の caveat 4 のとおりである。

**Mac での CPU スモーク(`results/ablation_cpu_smoke_small.json`。安全でない `small` の N = 2^12、d = 1024、B = 2、実 MNIST、NAG、seed 48 の 1 シード、1 スレッド、ウォームアップ 1 + 計測 3)。RQ3 の結果ではなく、ランナーの動作確認である**。Task 48 fix round 1 で、provenance を書くバイナリにより、commit 0e07561 の clean な作業ツリーで再生成した(`provenance` に記録)。**このときの Mac は他のセッションの作業で負荷が高かった**(load average は開始時 { 33.93 41.15 28.58 }、終了時 { 22.09 37.37 27.70 })。**そのためステップ時間は構成間の比較に使えず、意味を持つのは計数と検査だけである**。

| hoist / bsgs / lazy_relin | 回転 | relin | 鍵切替 | 既定との重みの差 |
|---|---|---|---|---|
| 1 / 1 / 1 | 171 | 62 | 233 | 0(同一) |
| 1 / 1 / 0 | 171 | 114 | 285 | 2^-26.9 |
| 1 / 0 / 1 | 210 | 101 | 311 | 2^-27.1 |
| 1 / 0 / 0 | 210 | 114 | 324 | 2^-27.0 |
| 0 / 1 / 1 | 171 | 62 | 233 | 0(同一) |
| 0 / 1 / 0 | 171 | 114 | 285 | 2^-26.9 |
| 0 / 0 / 1 | 210 | 101 | 311 | 2^-27.1 |
| 0 / 0 / 0 | 210 | 114 | 324 | 2^-27.0 |

- 全構成で、閉形式が実測と一致し、計数は反復間で一定だった。
- `hoist` オフの行は、対応する `hoist` オンの行と計数が同じで、重みもビット一致した。以上で `all_checks_pass: true`。
- 高負荷下の時間・速度比は表から除いた。元 JSON の時間を構成間の性能比較に使わない。provenance が無かった修正前の出力は `626160e:results/ablation_cpu_smoke_small.json` に履歴として残るが、論文表や速度の根拠として採用しない。
- `small` は dnum 11、α = 1 なので、ホイストの利得の大きさは n16_train(dnum 8、α = 4)とは違うはずである。

**本番の行(サーバ、未計測。出力先は実行依頼に登録した予定パスで、現時点では結果ファイル未回収)**:

| 依頼 / 出力先 | バックエンド | 条件 | 状態 |
|---|---|---|---|
| R-48-2: `results/quadruped/ablation_cuda.json` | RTX 4090 | n16_train、レベル 7、NAG、B = 32、同一レイアウト | 未計測。事前必要量 2456 MiB |
| R-48-2 の exit 3 時のみ: `results/quadruped/ablation_cuda_sgd.json` | RTX 4090 | 同上、SGD に変更 | 未計測。事前必要量 2072 MiB |
| R-48-3: `results/quadruped/ablation_cpu_l7.json` または `ablation_cpu_l7_sgd.json`(同じディレクトリ) | CPU 12 スレッド | n16_train、レベル 7、R-48-2 と同じ optimizer / layout / B | 未計測。対応する GPU 行と比較する |
| R-48-4: `results/quadruped/ablation_cpu.json` | CPU 12 スレッド | n16_train、レベル 31、全鍵、NAG、B = 32 | 未計測。CPU の RQ3 表用 |
| GPU 全サイズ(別途測定枠が必要) | RTX 4090 | n16_train、レベル 31、全鍵 | 未計測。事前必要量約 15.2 GiB |

GPU 行はレベル 7 の値であり、レベル 31 の CPU 行から CPU/GPU 速度比を作らない。比較は同じ optimizer、レベル、レイアウト、バッチで行う。NAG が容量チェックで exit 3 の場合だけ SGD を試す。両 GPU 行が SKIP なら R-48-3 は NAG とし、GPU 速度比は出さない。実際の device OOM(終了コード 1)は失敗としてログを保存し、SGD へ切り替えたり再試行したりしない。

R-48-2/3 は warm-up 1 + 計測 5 反復、R-48-4 は warm-up 1 + 計測 3 反復である。これらは同一初期条件・同一シード内の反復で、独立 10 シードの評価ではない。CPU/GPU の時間を一つの分布にプールせず、構成別・バックエンド別に中央値、反復数、共有負荷を示す。計数の正しさと RQ3 の性能結論を分け、**RQ3 の実時間の寄与は未判定**とする。

## 3. GPU ブートストラップ(Task 49)

### 3.1 実装と訓練ベンチの境界

`f64a617` (`feat(cuda): GPU bootstrapping`) は独立レビュー承認後にコミット済みで、**コード準備完了、実機測定は保留**である。実装前の設計判断と代替案は [gpu-bootstrap-design.md](gpu-bootstrap-design.md) に記録した。既存 CPU の多項式評価・丸め・演算順を狭い `BootstrapArithmetic: HeOps` で共用し、新しい外部依存を加えていない。

`CudaBootstrapper` は SSE(main→h32)、centered mod-raise、SSE(h32→main)、CtS、EvalMod、StC をデバイス上で実行する。暗号文を CPU にダウンロードせず、既存の `BootstrapOps` / `BootstrapRefresher` に接続する。鍵と変換対角は一度アップロードする。公開整数定数のホストでの展開・転送は残っており、将来の GPU 時間に含める。計数は全処理と最終同期が成功した後だけ合算する。

この API の接続確認は、§2 の訓練ベンチが GPU ブートストラップへ切り替わったことを意味しない。現在の非対話 CUDA 訓練は `HostBootstrapper` を使うため、レベル演算、CPU ブートストラップ、転送の時間を引き続き分ける。

### 3.2 確認済みの範囲と未計測項目

以下は既存の Task49 実装報告・独立レビューと保存済みログを照合した結果であり、この報告更新で新たに実行した GPU 測定ではない。

| 項目 | 確認結果 | 証拠と範囲 |
|---|---|---|
| CKKS の CPU 回帰 | 158 件合格、6 件 ignored | `/tmp/hekit-task49-ckks-tests.log`、Task49 実装報告。CPU の既存精度・EvalMod・SSE を含む |
| CUDA クレートのローカル検査 | 61 件合格、GPU 依存 47 件 ignored | `/tmp/hekit-task49-cuda-tests.log`。CPU リプレイとコンパイルの証拠 |
| 小さい全経路の CPU リプレイ | N = 1024、全 512 スロット、26.73 bit、CPU 残差・scale・level・全計数が一致 | `/tmp/hekit-task49-focused.log`。物理 GPU 上の一致ではない |
| n16_boot の CPU 全スロット精度 | N = 2^16、32768 スロット、入力 [-1,1]、**17.46 bit、出力レベル 16、111 KS = 回転 69 + relin 40 + SSE 2** | `/tmp/hekit-task49-n16-precision.log`。回転は共役 2 回を含む。GPU 時間ではない |
| CUDA 実機テストの構築 | `gpu_bootstrap` の ignored テストがコンパイル成功 | `/tmp/hekit-task49-gpu-compile.log`。実行の証拠ではない |
| GPU 全スロット精度・CPU/GPU ビット一致・計数一致 | **未計測** | 実機 n16 テスト待ち |
| GPU メモリピーク・実機 allocation failure | **未計測** | 事前容量チェックはピーク測定でも確保保証でもない |
| GPU 遅延・≤220 ms 目標 | **未計測・未判定** | CPU 時間や CUDA コンパイルから補完しない |

実装報告とレビューは `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-49-{report,review}.md`。`/tmp` のログはローカル確認用であり、論文の最終証拠として使う前に安定した結果ディレクトリへログと provenance を保存する。共通 EvalMod のラッパー比較は独立した数値オラクルではなく、独立した精度の証拠は既存精度テストと上の CPU 全スロットゲートが担う。

### 3.3 HEonGPU と固定目標

| 対象 | デバイス / N / スロット | ブート後の残りレベル | 秘密鍵分布 | 時間と扱い |
|---|---|---|---|---|
| HEonGPU Regular(公式 Appendix、2026-11-06 確認) | RTX 4090 / 2^16 / 32768 | 9 | 主鍵 h = 192、temporary h = 32 | **109.28 ms**、5 回の中央値、公表値 |
| 本実装 n16_boot / SSE h32 | GPU 実測待ち / 2^16 / 32768 | 16 | 主鍵は密三値、temporary h = 32 | **未計測**。計画の固定目標 **≤220 ms** |

出典は [HEonGPU の一次資料](https://heongpu.readthedocs.io/en/latest/appendix.html#performance-benchmarks)。残りレベル数と主鍵分布が違うため、220 ms は計画に固定された目標であり、同一仕事量で「HEonGPU の 2 倍以内」を保証する換算ではない。同じ N と temporary h32 だけから安全性の同等性も主張しない。将来の結果にはこの条件差を併記する。

## 4. Task 50 の証拠・ゲート・測定受け入れ

### 4.1 現在地

| 対象 | 現時点の状態 | 証拠 / 残る条件 |
|---|---|---|
| Task 47 GPU 演算の一致 | 実行した行は一致、全鍵の一部は SKIP | R-47-1、§1.3。SKIP を含む 45 ok を全行実測と数えない |
| Task 47 演算ベンチ | R-47-2 は測定済み | §1 の Criterion 中央値。修正前コードの結果を保持 |
| Task 47 修正後バッチ | **未計測** | R-47-3。1 / 2 / 4 ストリーム、既定と max のプール閾値を別々に測る |
| Task 48 訓練経路 / provenance | 実装・修正レビュー済み | `626160e` / `0e07561`、smoke `b1d1813`。再レビューで Important findings 解消 |
| Task 48 n16_train 訓練一致 | **未計測、Step 1 未判定** | R-48-1 で `n16_train_level7` の実行が必須 |
| RQ3 の 8 構成 | CPU small の計数・検査は確認済み、サーバ実時間は**未計測** | R-48-2/3/4 と四つの定義上の caveat(§2.3)を保持 |
| Task 49 | コード準備・レビュー・CPU 検証済み、GPU は**未計測** | §3。deferred は実機検証を指し、未実装とは扱わない |
| GPU メモリピーク | **未計測** | 必要量の見積もりとチェックポイント観測からピークを作らない |
| Task 50 / M4 総合判定 | 報告準備中、**最終判定は保留** | 未計測の RQ3・訓練一致・修正後性能を残して全体合格としない |

### 4.2 測定待ちの出力と実行条件

依頼の正本は [results/quadruped/requests.md](../results/quadruped/requests.md) の R-47-3 / R-48-1〜4。現時点では以下は予定出力である。

| 依頼 | 予定する証拠 | 受け入れ時に見るもの |
|---|---|---|
| R-47-3 | `results/quadruped/batch_streams_r47_3.log`、`hekit_cuda_task47_fix1_tests.txt`(同じディレクトリ)、`raw/R-47-3/`(同じディレクトリ) | clean な修正後 commit、全ストリームのビット一致、閾値別の中央値と SKIP、任意の profile |
| R-48-1 | `results/quadruped/task48_gpu_tests.txt`、`results/quadruped/raw/R-48-1/` | 5 テストの実行 / SKIP 内訳、`n16_train_level7` 実行、バッチごとの全計数と最終重み・bias の許容差 |
| R-48-2 | §2.3 の CUDA JSON、`results/quadruped/logs/R-48-2{,-sgd}.log`、`R-48-2.optimizer`(同じディレクトリ) | 終了コード、optimizer、provenance.gpu、8 構成と全検査 |
| R-48-3 / 4 | §2.3 の CPU JSON、`results/quadruped/logs/R-48-{3,4}.log` | 12 スレッド、レベル、optimizer、同一比較条件、8 構成と全検査 |

FULL-1/2 の後に R-47-3、続いて R-48-1 → 2 → 3 → 4 を各ブロック単独で実行する。既存依頼の pin は `0e07561`、追跡ファイル clean を必須とする。pin には Task52 の `b8e267c` も含まれるが、Task48 再レビューで対象の対話・ablation 経路は旧実装と同じ結果になることを確認済みである。対象外の共有プロセスの停止・設定変更は行わない。訓練の matvec は逐次、各 epoch と文書化したピーク後の `trim_pool(0)` を維持する。

### 4.3 受け入れチェック

1. **出所と完了を確かめる。** バイナリが書いた raw JSON とログ、commit / clean 状態、CLI、build/features、機械・CPU スレッド・GPU、データと seed、開始終了時の負荷、終了コードを保存する。後から raw JSON に metadata を手で追記して測定済みにしない。
2. **SKIP と失敗を別に扱う。** R-48 の exit 3 / SKIP は容量不足による未計測、device OOM は失敗として元ログを保持する。Cargo ok だけから実行を推定しない。Task49 も `SKIP UNMEASURED` で成功終了するため、実機採用判定前に `MEASURED:` 等の明示的な実行完了マーカーを検証する仕組み、または要求した実機テストが資源不足で失敗する仕組みを必須にする。
3. **同じ量を比較する。** R-48-1 は全 `OpCounter` のバッチごとの完全一致と最終重み・bias の差 ≤2^-12、さらに n16_train level7 の実行を要求する。演算のビット一致、訓練の許容差一致、Task49 の GPU 全スロット精度 ≥16 bit / CPU 残差・計数一致は別の証拠として残す。
4. **時間の母集団を分ける。** 同じ optimizer / level / layout / B の CPU と GPU 行だけを比較し、CPU と GPU の時間を混ぜない。ステップには GPU 同期待ちを含め、クライアント再暗号化や CPU bootstrap と分離する。反復中央値、独立シード数、共有負荷を記録し、1 シードの反復や Criterion の 10 サンプルを独立 10 シードと数えない。
5. **RQ3 の意味を保つ。** 全 8 フラグ構成、閉形式一致、反復間計数一定、hoist 対応行の計数・重み一致、`all_checks_pass: true` を確認する。§2.3 の四つの caveat を結果表から落とさない。small の高負荷 CPU smoke は計数・検査の証拠に限り、GPU 本番の速度や各技法の性能寄与を代用しない。
6. **メモリと目標を実測で閉じる。** `gpu_need_estimate` / `mlp_gpu_need` は容量のヒューリスティックで、`pool_used_memory` / reserved や `memory_after_trims` はチェックポイントの観測である。どちらもピークではない。ピークの測定方法・対象プロセス・プールの扱い・観測期間を明記した実測が無ければ、ピークは未計測と記す。Task49 の 20 GiB free は保守的な事前条件で、ピーク測定値ではない。遅延は同期まで含めて測り、≤220 ms は実測後に合否を記す。

## 5. 限界と本人の未決事項

現時点の証拠から主張できるのは、測定済み R-47-2 の演算遅延、実行済み GPU 演算の一致、CPU で検査した RQ3 の計数、Task49 のコード準備と CPU 精度である。GPU 訓練の速度向上、RQ3 の各技法の実時間寄与、修正後のマルチストリーム利得、GPU ブートストラップ目標達成、GPU メモリピークは未判定である。論文の見出しや M4 全体合格は、欠測を埋めてから条件付きの主張を更新する。CPU smoke、モデル外挿、コンパイル成功を実機測定へ読み替えない。

本人の未決事項は [M3 報告](m3-report.md) §9 と [引き継ぎ](handover.md) §7 を参照する。B ≥64 を主設定にするか、同一 N の解釈、秘密鍵分布と安全性の比較、wall-clock の基準、長期重み誤差、WAHC 2027 の著者情報は、未取得の測定結果と区別して残す。ライブラリ名は ckks-train に決定済みである。これらの未決事項で既存の測定・実装作業を止めない。
