# M4 報告(下書き、2026-11-05)

M4(CUDA バックエンド、Task 43〜50)の報告の下書き。§1 には Task 47 の測定値(R-47-2、quadruped-gpu)を記入済み。§2 は Task 48(GPU 上の訓練と RQ3 アブレーション)の実装と Mac での確認で、サーバの行(GPU 訓練、RQ3 の本番の表)は未計測。バッチ対逐次の再測定(R-47-3)と Task 49〜50 の節はまだ無い。

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
- 参考(Mac M3 Pro、M1 報告 §1、1 スレッド、parallel 機能無し、最上位レベル): add 1.85、mul + relin + rescale 302.7、rotate 256.9、hoisted × 16 1569、matvec_enc(d = 64)5711 ms。サーバ CPU の 1 スレッドは Mac の 2〜8.5 倍遅い(多くの演算で 2.3〜2.4 倍、add だけ 8.5 倍。E コアに載った可能性、ピン留めなし。M3 の Iris の比較と同じ傾向)。

### 1.1a マルチストリーム(`matvec_enc_batch`)は n16_train では逐次より遅い

R-47-2 で、同じ 4 つの積を 4 ストリームに分けると 41.67 ms、1 ストリームで逐次に流すと 34.99 ms だった(逐次 = 9.51 ms × 3.68。ホスト側の待ちは逐次でも既に隠れている)。

- **理由: 1 つの積で既にデバイスが埋まっている。** `gpu/add`(L31)は約 96 MiB を 110.9 µs で動かしており、約 0.9 TB/s、RTX 4090 のピーク帯域(1.01 TB/s)の約 90% にあたる。N = 2^16 では要素ごとのカーネルも帯域律速で、占有率も満杯なので、ストリームを増やしてもスループットは増えない。
- バッチ側だけにある固定費が遅さを足していると考えられる(**仮説。プロファイルで未確認**)。呼び出しごとのワーカースレッド 4 本、約 480 MiB の作業領域の確保と解放、4 つ同時の作業領域である。メモリプールの解放閾値が 0(CUDA の既定)だったので、同期のたびに物理メモリを返して再マップしていた可能性がある。バッチの sd(2.1 ms)が逐次(0.07 ms)より大きいことはこの仮説と矛盾しない。R-47-3(と任意の nsys)で確かめる。
- **Task 47 の修正で変えたこと**(fix round 1 と 2):
  - 作業領域の仮確保をやめ、空き容量の確認だけにした(他テナントのメモリを一時的にでも取らない)。
  - スレッドを使わず、呼び出しスレッドから各ストリームに順に積むようにした。
  - プールの解放閾値を `CudaDevice` の設定にした。既定は **256 MiB**(`CudaDevice::DEFAULT_POOL_RELEASE_THRESHOLD`)。CUDA の解放閾値は、使用中の分も含めたプールの**予約量**の上限である。同期のたびにプールは、予約量が `max(使用中, 閾値)` 以下になるまで未使用分を返す。n16_train では鍵と入力だけで 256 MiB を超える(全鍵 1 本で 288 MiB)ので、**既定は n16_train ではほぼ閾値 0 と同じに働く**(一時領域は同期ごとに返され、再マップされる)。それでも共有 GPU の保護は保たれる。同期後の hekit の使用量は常に「使用中 + 256 MiB」以下で、過去のピークは残らない。
  - `set_pool_release_threshold(u64::MAX)` にすると、hekit のピーク使用量を `trim_pool` まで保持する(fix round 1 の設定。共有 GPU では他テナントを圧迫しうる)。
  - **Task 48 の選択肢(既定は変えない)**: n16_train で一時領域を再利用したいときは、(a) 鍵のアップロード後に `set_pool_release_threshold(pool_used_memory() + 256 MiB)` とする(鍵と入力に加えて最大 256 MiB の一時領域を保持)か、(b) `u64::MAX` にして下の `trim_pool(0)` の規則を守る。
  - 再測定 R-47-3 は 1 / 2 / 4 ストリームを、既定(256 MiB。**n16_train ではほぼ閾値 0**、出力の `[default 256 MiB, ~threshold 0 at n16_train]` 行)と `u64::MAX`(`[max]` 行)の両方の閾値で測る(任意で nsys)。
- **Task 48 への指示: 再測定で利得が出るまで、訓練ステップでは逐次の `matvec_enc` を使う。** 小さい積(低レベルの末尾、seal_2020 程度の大きさ)では効く可能性があるが、まだ何も測っていない。
- **Task 48 への指示(GPU メモリ): 訓練ループは各エポックの境目と、一度きりのピーク(鍵のアップロード、`matvec_enc_batch`、最上位レベルの大きな演算)の後に `CudaDevice::trim_pool(0)` を呼ぶ。** GPU は他テナント(約 21 GB を使用)と共有で、空きは 1.5〜3 GB しかない。プールに残ったメモリは hekit が使っていなくても他テナントからは使えない。`cuda_ops` ベンチは各ブロックの前と最後に `trim_pool(0)` を呼び、空き容量は `CudaDevice::available_memory`(ドライバの空き + プールの未使用分)で判定する。

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

## 2. GPU 上の訓練と RQ3 アブレーション(Task 48)

**状態(2026-11-06): 実装と CPU での確認は済み。GPU 上の訓練と、サーバでの RQ3 の表は未計測。** サーバの GPU は他テナントが約 20.7 GB を使っており、空きは 2〜3 GB しかない。n16_train の全鍵での GPU 訓練(約 15.2 GiB 必要)は、GPU メモリが空くまで**未計測**のままとする(計画責任者の裁定: 未計測の行はそう書く)。

### 2.1 実装

- **バックエンド切り替え**: `mnist` と `iris` に `--backend cpu|cuda` を足した(`cuda` は `cuda` 機能付きのビルドが必要)。共通部分は `hekit_bench::backend`。暗号化・復号・エンコード・クライアント再暗号化・ブートストラップはホストに残し、GPU では次のように分ける(「レベル演算は GPU、ブートストラップは CPU」)。
  - 対話モードのクライアント再暗号化は `HostRefresher` の後ろで行う(ダウンロード、ホストで再暗号化、アップロード)。
  - 非対話モードのブートストラップは `HostBootstrapper` の後ろで行う。ホスト側の計数は GPU 評価器の `OpCounter` に足し込む。
  - GPU ブートストラップ(Task 49)はこのタスクの範囲外。
- **RQ3 のフラグ**: `--no-hoist --no-bsgs --no-lazy-relin`(`EvalConfig`)を `mnist`・`iris` の両モードに通した。
- **閉形式の照合(Task 43 M5)**: フラグを考慮するようにした。
  - 対話モード: `predict` の行列ベクトル積を `EvalConfig::matvec_rotations` / `matvec_enc_relins` で数える。8 通りすべてで実測と一致する(`tests/mnist_flags.rs`、`small`)。
  - 非対話モード: `predict_update_with` を使う。全オフで実測と一致する(`mini_boot`)。
  - コストモデルとその補正項は既定フラグのもの、と出力 JSON に明記した。
- **`HostRefresher` の N1/N2(Task 43 レビュー)**: `HostBootstrapper` と同じにそろえた。単体テストあり(`hekit-nn` `refresh.rs`)。
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

### 2.2 Step 1: GPU と CPU の一致テスト(未実行、サーバ待ち)

`crates/hekit-bench/tests/mnist_cuda.rs` と `iris_cuda.rs` に置いた(どちらも `#[ignore]`、`--features cuda`)。

- mnist の 3 テストは、実 MNIST の subset 1024 で同じ対話訓練を `--backend cuda` と `--backend cpu` で走らせ、次を確かめる。GPU を先に走らせ、メモリ不足でスキップしたら CPU は走らせない。
  - **バッチごとの `OpCounter` が完全に一致する**(回転、relin、鍵切替、積、rescale、ブートストラップ)。
  - 復号した最終の重みとバイアスが 2^-12 以内で一致する。
- iris の 2 テストは、seal_2020 の対話 timing と mini_boot の非対話(ブートストラップは CPU)で同じことを確かめる。

| テスト | パラメータ | GPU メモリ(申告) | 見込み |
|---|---|---|---|
| mnist `n16_train_full_keys` | n16_train、レベル 31、全鍵(288 MiB × 32) | 15560 MiB | **スキップ**(空き 2〜3 GB)。GPU 全サイズ訓練は未計測 |
| mnist `n16_train_level7` | n16_train、`--top-level 7`、鍵はレベル 7 以下(24 MiB × 32) | 2456 MiB | 空き次第。足りなければスキップ |
| mnist `small_insecure` | `small`(N = 2^12、安全でない)、d = 1024、B = 2、実データと 784-32-10 | 397 MiB | 走る |
| iris seal_2020 対話 | seal_2020、B = 20、timing 6 反復 | 約 1.3 GiB | 走る |
| iris mini_boot 非対話 | mini_boot(N = 2^10、安全でない)、1 エポック | 約 0.1 GiB | 走る |

どの行が走ったかは、サーバの結果で埋める。

### 2.3 RQ3 アブレーションの表

`crates/hekit-bench/src/bin/ablation.rs` は、対話モードの eMLP-1 の 1 反復(784 → 32(z^2 + z)→ 10(線形)、NAG、1 バッチ 1 暗号文、クライアント再暗号化)を、8 通りの `EvalConfig` で測る。出力は `results/ablation_<backend>.json`。

- 鍵、初期ネットワーク、バッチは全構成で同じ。ウォームアップの後に計測反復を回し、中央値を取る。
- 時間は「ステップ」(サーバ側、デバイスの待ちを含む)と「再暗号化」(ホスト。フラグは影響しない)に分ける。
- 計数は 1 反復分。閉形式は「既定の実測 + 各 matvec のフラグ差分」で、全構成で照合する。
- 重みは既定構成との差を出す。

**この表が述べなければならないこと**(`hekit_ckks::ops` の "What the RQ3 table must state" をそのまま写す。`ablation` の出力 JSON の `caveats` にも同じ文がある):

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

**Mac での CPU スモーク(`results/ablation_cpu_smoke_small.json`。安全でない `small` の N = 2^12、d = 1024、B = 2、実 MNIST、1 スレッド、ウォームアップ 1 + 計測 3)。RQ3 の結果ではなく、ランナーの動作確認である**:

| hoist / bsgs / lazy_relin | 回転 | relin | 鍵切替 | ステップ(s、中央値) | 全オフ比の速さ | 既定との重みの差 |
|---|---|---|---|---|---|---|
| 1 / 1 / 1 | 171 | 62 | 233 | 0.560 | 1.81× | 0(同一) |
| 1 / 1 / 0 | 171 | 114 | 285 | 0.762 | 1.33× | 2^-26.9 |
| 1 / 0 / 1 | 210 | 101 | 311 | 0.859 | 1.18× | 2^-27.1 |
| 1 / 0 / 0 | 210 | 114 | 324 | 0.994 | 1.02× | 2^-27.0 |
| 0 / 1 / 1 | 171 | 62 | 233 | 0.606 | 1.67× | 0(ビット一致) |
| 0 / 1 / 0 | 171 | 114 | 285 | 0.793 | 1.28× | 2^-26.9 |
| 0 / 0 / 1 | 210 | 101 | 311 | 0.893 | 1.13× | 2^-27.1 |
| 0 / 0 / 0 | 210 | 114 | 324 | 1.013 | 1.00× | 2^-27.0 |

- 全構成で、閉形式が一致し、計数は反復間で一定だった(`all_checks_pass: true`)。
- 再暗号化(ホスト)は全構成で 0.25〜0.27 s で、フラグの影響を受けない。
- `small` は dnum 11、α = 1 なので、ホイストの利得の大きさは n16_train(dnum 8、α = 4)とは違うはずである。

**本番の行(サーバ、未計測)**:

| 行 | バックエンド | パラメータ | 状態 |
|---|---|---|---|
| `results/ablation_cpu.json` | CPU 12 スレッド | n16_train、レベル 31(全鍵) | 未計測(サーバ待ち、R-48-4) |
| `results/ablation_cpu_l7.json` | CPU 12 スレッド | n16_train、`--top-level 7` | 未計測(サーバ待ち、R-48-3)。GPU 行と同じ条件の比較用 |
| `results/ablation_cuda.json` | GPU(RTX 4090) | n16_train、`--top-level 7`、鍵はレベル 7 以下(NAG で 2456 MiB 申告、SGD なら 2072 MiB) | 未計測(サーバ待ち、R-48-2)。空きが足りなければスキップ |
| GPU 全サイズ | GPU | n16_train、レベル 31、全鍵(約 15.2 GiB) | **未計測**。GPU メモリが空くまで測れない |

GPU 行はレベル 7 の値で、最上位レベル 31 の値ではない。最上位レベルの GPU の値は、GPU メモリが空くまで未計測とする。
