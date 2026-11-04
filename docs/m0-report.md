# M0 報告(2026-11-04)

## 0. 判定(冒頭)

**RQ1(コストモデル、M0 時点): 条件付きで「30% 以上少ない (B, N) は存在する」。ただし ReBoot 自身の設定 (N = 2^16, B = 48) では spec §4.3.2 の基準を満たさない。**

| 指標 | (2^16, 48) の比 diag_batched / reboot | ≤ 0.70 を満たす (B, N) |
|---|---|---|
| (i) 計画の KS = 回転 + relin(ブートストラップ除く) | **0.454** | B ≥ 32 の全 8 点(最良 0.181 @ (128, 2^17)) |
| (ii) spec の KS_spec = 回転 + relin + ブートストラップ × BOOT_KS | **0.744(不合格)** | **B ≥ 64 のみ**: (64, 2^16) 0.64*, (64, 2^17) 0.67, (128, 2^16) 0.43*, (128, 2^17) 0.46* |
| (ii') 字義どおり = 回転 + ct-ct 乗算 + ブートストラップ × BOOT_KS | **0.786(不合格)** | B ≥ 64 のみ: 0.68* / 0.68 / 0.49* / 0.48* |
| (ii) を compact 再配置なしで | 12.9 | なし(全点 ≥ 7.27) |

`*` = diag_batched が 1 ミニバッチに 2 暗号文以上必要(d_pad·B > N/2)。BOOT_KS = 502 (2^16) / 630 (2^17)。

(ii)/(ii') の合格は次の 3 つの前提に依存する:
1. **compact W/V 再配置が CKKS で動くこと。** 平文シミュレータでのみ検証済み。これが無いと (2^16, 48) で 12.9×、全 (B, N) で 7.27× 以上悪くなる。
2. **BOOT_KS ≈ 502 の一部は推定。** EvalMod 部分(約 13%)は OpenFHE の式からの推定で、実行して数えたものではない。BOOT_KS が 2 倍(1004 / 1260)だと、B = 64 は 0.76 / 0.78(2^16 / 2^17)で不合格になり、B = 128 だけが残る(0.56 / 0.59)。
3. **B ≥ 64 で ReBoot (B = 48) と同精度(±0.5 pt)が出ること。** 未検証。

このほか §4.3.2 の wall-clock 条件(同一 CPU で ReBoot 公式以下)は M0 では評価できていない。

**Plan B: Step 2 の条件「(b) が偽」は字義どおりには成立しない。** (b) は、計画の KS では無条件に、spec 指標では上の前提 1〜3 の下で真になる。**したがって M0 の規則では Plan B は発動しない。** 推奨は、今は発動せずに M1 へ進み、§4.3.2 のとおり 2027-02-15(M2 完了時)に hekit 上の実測 (Task 32/33/34) で最終判定すること。M1(CKKS コア)の中身は本計画でも Plan B でも同じなので、M1 に進む判断は RQ1 の結果に左右されない。**判断は本人。**

判定が「Plan B 寄り」に変わる条件(どれか 1 つで spec 指標の合格が消えうる):
- compact 再配置が深さ予算(+2 レベル/更新)や精度目標に収まらない。
- B = 64/128 で ReBoot B = 48 比 −0.5 pt 以内の精度が出ない(B = 48 では合格点が無い)。
- 実測した BOOT_KS や EvalMod の回数が推定より大きい(2 倍で B = 64 が脱落)。
- 「同一 N」を ReBoot eMLP-1 の 2^16 に限定すると判断した場合、合格点は 2 暗号文構成 (64, 2^16)* と (128, 2^16)* だけになる。§4.3.3 の基本則 d × B ≤ N/2 を満たす合格点は (64, 2^17) だけ。
- 同一 CPU の wall-clock で ReBoot に負ける。

## 1. 数字の表(Task 2〜6)

出典はすべて `spikes/RESULTS.md`(以下 R)、`spikes/reboot/RUN.md`、`spikes/costmodel/table.py`。table.py を 2026-11-04 に再実行し、R の表と完全に一致することを確認した。

| スパイク | 項目 | 値 | 比較・基準 | 備考 |
|---|---|---|---|---|
| 1a CPU NTT (Rust) | forward, N = 2^16, q 60 bit, 1 core, 中央値 | **417.8 µs**(アイドル)/ 416.4 µs(負荷下) | 計画の帯 200–400 µs → **約 4% 超過** | スカラー Harvey、SIMD なし。inverse 408.9 / 385.5 µs |
| | SEAL 4.1.2 forward 中央値 | **373.7 µs**(アイドル)/ 368.0 µs(負荷下) | — | `ntt_negacyclic_harvey`、arm64 も SIMD なし |
| | Rust / SEAL | **1.12**(中央値)/ 1.24(min ベース、上方バイアスあり)/ 1.26(SEAL lazy 比) | M1 完了基準「SEAL 比性能 0.5〜1.0 倍」→ 0.89 倍で帯内 | 負荷下 1.10–1.14 |
| 1b CUDA NTT | batch 64 forward, per-poly | **GPU 待ち(未実行)** | 基準 ≤ 2 × 3.07 = **6.14 µs/poly**(A100) | カーネルの CPU エミュレーションが ntt-cpu とビット一致。sm_80/90 の PTX までコンパイル可 |
| | 参照値 GPU-NTT (HEonGPU の NTT) | A100, N = 2^16, batch 64, radix2-CT: 196.43 µs/batch = **3.07 µs/poly**(論文値) | 4-Step 2.95、RTX 4090 1.52 µs/poly | 出典は Özcan et al., IEEE Access 13 (2025) Table 10。法の種類はキャプションに無い。数値は画像の表から読んだもので、独立には確認していない。HEonGPU / GPU-NTT の ePrint は 403 で読めなかった |
| 2 SEAL 2020 再現 (Iris 4-10-3, N = 2^15) | 1 反復(server、alg5、exact keys) | **0.997 ± 0.027 s**(アイドル)/ 1.031 s(負荷下) | — | alg4 1.027 s、pow2 keys 1.168 s(アイドル) |
| | server + 再パック + クライアント暗号化 | **1.369 s**(アイドル)/ 1.415 s(負荷下) | 論文 Table 4 の記載合計 9.25 s → **6.8×**(負荷下 6.5×) | 再パック 0.334 s、暗号化 0.038 s |
| | 論文 Table 4 自体の不整合 | 対角: 記載 9.25 s ≠ 相の和 9.10 s。行: 28.47 s ≠ 26.01 s | 差の理由は論文に無い | 記載合計との比を主にし、相ごとの比は参考扱い(相和比 8.6×) |
| | 演算数(6-3-1 net)と論文 Table 2 | ours alg5: 23 ct-ct + 1 pt / 16 rot。alg4: 23 + 1 pt / 11 | 論文 17 mult / 17 rot | 両方が計算している行の乗算数は一致する。差は計上範囲(ΔW を除外、W1ᵀ を含む)と、回転数の数え方(出力次元か入力次元か)による。行ごとには食い違う(§6) |
| | 演算数(Iris 4-10-3、alg5) | 35 ct-ct / 35 relin / 31 rot / 36 rescale | コストモデル diag_2020 と一致 | — |
| | 暗号化学習の誤差 | 1 反復 max\|W_he − W_ref\| = **4.4e-6**。11 反復連鎖 (run 3): 1.4e-5 (alg5) / 3.6e-5 (alg4) / 1.2e-4 (pow2) | 合格線 < 1e-3 | 誤差の主因はスケールの強制等化 |
| | 平文精度 (400 epoch, batch 20) | train **0.982** / test 0.960 ± 0.021(10 seeds) | 論文 0.9805 / loss 0.0249 | batch 1 + lr 0.1 は発散(10/10) |
| 3 ReBoot (eMLP-1, MNIST) | ビルド | **OK**(OpenFHE 1.2.1、ReBoot `3ac173f`、移植用の 3 パッチ) | — | — |
| | 実行 | **メモリ不足 (OOM) で未計測** | 機械は 36 GB RAM | context 生成直後の footprint ≥ **44 GB**。`time -l` の peak footprint は 4 回の実行で **51–54 GB**。最初の反復の重みブートストラップで kill |
| | epoch 時間 / 精度 | **メモリ不足で未計測** | 論文値: **190.32 s/iter**、ピーク **76.07 GB**(2× Xeon 5318S, 384 GB) | 論文値からの算術: 100 it ≈ 5.3 h、全 MNIST 1250 it ≈ 66 h。test acc 94.61 %(論文値、平文ツイン) |
| | 設定の転記 | `crates/hekit-bench/configs/reboot_emlp{1,2,3}.json` | 784-32-10、2^16、l 24、Δ 2^49、B 48、lr 1e-3、mom 0.9 | epochs = 10 はリポジトリのスクリプト由来で論文には無い。weight decay は JSON に無い |
| 4 コストモデル | ReBoot eMLP-1 / sample | 30 rot + 7 relin = **37 KS**、ブートストラップ 4 / mini-batch | lazy relin にした仮想 ReBoot は 35 | (r, c) = (1024, 32) は ReBoot のログと一致 |
| | diag_batched (2^16, 48) / sample | KS 16.8、KS_spec 58.6(ReBoot 78.8) | 比 0.454 / **0.744** | 2 暗号文/batch |
| | diag_batched (2^16, 64) | KS_spec 44.0 vs 68.4 → **0.64**。字義どおりの指標で 0.68 | — | 2 暗号文/batch |
| | diag_batched (2^17, 64) | KS_spec 52.8 vs 78.4 → **0.67**。字義どおりで 0.68 | — | 1 暗号文 |
| | 検証 | 2020 Table 2 の合計 17/17 と row 38/63 を再現。Task 4 の SEAL 実測 35/35/31 とも一致 | 行ごとの回転数は strict xfail | pytest 35 pass + 1 xfail |

## 2. (a) CUDA NTT は HEonGPU の 2 倍以内か

**未判定(GPU 待ち)。** 判定には A100 80GB を 1 枚、約 1 時間借りれば足りる(下の価格で約 $1–5)。H100 で測ると A100 基準の 6.14 µs は甘めになる。

```sh
cd hekit/spikes/ntt-cuda
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
cargo test --release -- --include-ignored --nocapture   # GPU 一致テストで SKIP が出ないこと
cargo run --release --bin bench                          # N=2^16, batch 64, 50 reps
cargo run --release --bin bench -- 16 64 200 20          # 再現性の確認
```

前提: driver の CUDA バージョンが NVRTC 12.x 以上であること。既知のリスクとして、`ntt_fwd_shared` に `__launch_bounds__(1024)` が無いので LAUNCH_OUT_OF_RESOURCES が出る可能性がある(§6)。

## 3. (c) GPU 費用見積り(M4 = 200 GPU 時間)

公開価格ページだけを読んだ(2026-11-04 アクセス。登録・ログイン・購入はしていない)。価格は 1 GPU 1 時間あたりの USD。

| プロバイダ | GPU(構成) | オンデマンド $/h | spot / community $/h | 出典 |
|---|---|---|---|---|
| Vast.ai(マーケット) | A100 SXM4 80GB ×1 | 0.63 – 1.60(中央値 1.09、11 件) | 最低入札 0.27〜(中央値 0.93) | 公開検索 API `https://console.vast.ai/api/v0/bundles/`。`https://vast.ai/pricing` は数値が描画されず読めなかった |
| Vast.ai | A100 PCIe 80GB ×1 | 0.87 – 1.00(5 件) | 0.13〜 | 同上 |
| Vast.ai | H100 SXM 80GB ×1 | 1.34 – 3.41(中央値 2.40、7 件) | 0.40〜 | 同上 |
| RunPod | A100 80GB PCIe / SXM | 1.79 / 1.79(Secure) | 1.19 / 1.39(Community) | https://www.runpod.io/pricing |
| RunPod | H100 80GB PCIe / SXM / NVL | 2.89 / 3.99 / 3.19(Secure) | 1.99 / 2.69 / 2.59(Community) | 同上 |
| Lambda | H100 80GB PCIe / SXM ×1 | 3.29 / 4.29 | — | https://lambda.ai/pricing(1 枚構成の A100 は 40GB の $1.99 だけで、80GB は無い) |
| GCP us-central1 | A100 80GB (a2-ultragpu-1g, VM 込み) | 5.07 | 3.01 (spot) | https://gcloud-compute.com/a2-ultragpu-1g.html(GCP 価格の集計サイト、データ更新 2026-11-01)。GCP 公式ページは JS アプリに転送されて読めなかった |
| GCP us-central1 | H100 80GB (a3-highgpu-1g) | 11.06 | 6.59 (spot) | https://gcloud-compute.com/a3-highgpu-1g.html |
| AWS us-east-1 | H100 80GB (p5.4xlarge) | 6.88 | 4.51 (spot) | https://instances.vantage.sh/aws/ec2/p5.4xlarge(公式の p5 ページには価格が無い)。A100 80GB の 1 枚構成は AWS に無い(p4de は 8 枚) |

200 時間の費用:

| GPU | 最安(マーケット / community) | 専用クラウド(Secure / Lambda) | ハイパースケーラ spot〜オンデマンド |
|---|---|---|---|
| A100 80GB | **$126 – 320**(Vast)/ $238 – 278(RunPod Community) | $358(RunPod Secure) | $602 – 1,014(GCP) |
| H100 80GB | $267 – 681(Vast)/ $398 – 538(RunPod Community) | $578 – 858(RunPod Secure, Lambda) | $902 – 1,376(AWS)/ $1,318 – 2,212(GCP) |

**推奨: A100 80GB を RunPod か Vast.ai で借りる。予算上限は約 $400(200 h × $1.4–1.8 に 1 割程度の余裕)。** 理由は次の 3 つ。
- 比較基準 (GPU-NTT) が A100 で測られているので、同じ GPU で直接比べられる。
- ハイパースケーラより 3〜5 倍安い。
- spike 1b も同じアカウントで約 1 時間で済む。

H100 を使うなら $500–900 を見込む。spot や Vast は中断と品質のばらつきがあるので、長時間の bench は Secure / オンデマンドで回す。M4b の基準は「RTX 4090 で 110 ms の 2 倍以内」なので、4090 も短時間借りる必要がある(本表では価格を取っていない)。

**大メモリ CPU ホスト(ReBoot ベースライン用):** eMLP-1 のピークは 76 GB(論文値)、eMLP-2 は 181 GB、eMLP-3 は 214 GB。

| ホスト | vCPU / RAM | オンデマンド $/h | spot $/h | 10 h | 30 h | 出典 |
|---|---|---|---|---|---|---|
| AWS r7i.4xlarge | 16 / 128 GiB | 1.058 | 0.391 | $10.6 | $32 | https://instances.vantage.sh/aws/ec2/r7i.4xlarge |
| AWS r7i.8xlarge | 32 / 256 GiB | 2.117 | 0.553 | $21.2 | $64 | https://instances.vantage.sh/aws/ec2/r7i.8xlarge |
| GCP n2-highmem-16 | 16 / 128 GB | 1.048 | 0.629 | $10.5 | $31 | https://gcloud-compute.com/n2-highmem-16.html |

- 論文のマシンは 2× Xeon 5318S(48 コア)。16 vCPU ではおそらく数倍遅くなるので、eMLP-1 の 1 epoch(論文速度で 5.3 h)には 10〜30 h を見込む。
- 推奨は **r7i.8xlarge のオンデマンド(約 $20–65)**。256 GiB あれば eMLP-2/3 まで載る。ジョブにチェックポイントが無いので spot は避ける。
- §4.3.2 の「同一 CPU での wall-clock」比較もこのホストで行うことになる(36 GB の Mac では ReBoot が動かない)。

## 4. M1 に進む前に本人が決めること

1. **GPU レンタル**: プロバイダと予算上限(推奨は RunPod / Vast の A100 80GB で ≤ $400)。最初の用途は spike 1b の判定(約 1 h)。
2. **大メモリ CPU ホスト**: ReBoot ベースラインと同一 CPU での比較用。プロバイダと時間(推奨は r7i.8xlarge で 30 h、約 $65)。やるのは M0 の追加測定か、M2 の比較時か。
3. **条件付きの RQ1 判定を受け入れて M1 に進むか**(推奨: 進む。Plan B の最終判定は 2027-02-15)。
4. **B ≥ 64 を論文の主設定にしてよいか**: §4.3.3 の「ReBoot の値も併走」は維持し、B = 48 では spec 指標を満たさないと正直に書くことになる。
5. **「同一 N」の解釈**: eMLP-1 は 2^16 に固定するか、2^17 での比較も合格に数えるか。
6. **M2 の最初に compact W/V 再配置を CKKS 上で検証することを優先するか**(spec 合格の前提 1)。BSGS で giant step 1 回につき relin が n2 回必要になる点(設計の前提)も M1 の BSGS API に反映する。
7. CUDA カーネルは未実行。M1 の途中から CUDA に着手する計画(§5 M4)の前に spike 1b を回すかどうか。

## 5. M0 中の判断(ルーリング)

- **T1**: ワークスペースで `exclude = ["spikes"]`。cargo-deny は既定の toolchain で入れた。deny.toml で `tfhe` / `concrete*` を禁止。
- **T3**: GPU のレンタルはしなかった(課金を伴うため)。「1 ブロック = 1 多項式を共有メモリ」は N = 2^16 で不可能(512 KiB > 228 KiB)なので、大きいストライドの 5 ステージを global memory、残り 11 ステージを共有メモリで処理する分割に変えた。カーネルの CPU エミュレーションを ntt-cpu とビット一致で検査した。
- **T4**: 追加で SEAL NTT と SEAL の演算数を測った。論文 Table 4 の合計と相の和が合わないので、比は記載合計に対して出した。負荷下(load 17–68)の計測をアイドル時 (load ≈ 3) に再実行し、表はアイドル値を主にした。
- **T5**: ビルドや実行に失敗したら原因を記録して先へ進む。1 epoch が 2 h を超える見込みなら有限バッチで外挿する、という上限を設けた。実際には OOM で外挿もできなかった。
- **T6**:
  - `diag_2020` を独立したレイアウトキーにした。計画の「diag_batched の B = 1」とは式が両立しないため。
  - d_pad·B > N/2 のときは ceil 個の暗号文で数え、表で印を付ける。
  - KS(計画の定義)と KS_spec(BOOT_KS 込み)を併記する。ReBoot は寛大にモデル化する。
  - 修正ラウンドで次を追加した: compact 再配置なしの列、字義どおりの指標、判定文を ReBoot 自身の (N, B) の spec 結果から書き出す形への変更。

## 6. 既知の小さな問題(M1 前に知っておくこと)

- **CI**: deny ジョブは `check licenses` しか走らず、tfhe / concrete* の禁止 (bans) は CI では強制されない(ローカルでは `cargo deny check bans` が ok)。cargo-deny-action@v2 はタグが固定されていない。
- **NTT CPU**: bit-reversed 出力順がテストで固定されていない。最悪ケースの入力(全係数 q−1 や 0)のテストも無い。
- **NTT CUDA**: `ntt_fwd_shared` に `__launch_bounds__(1024)` が無い。計時窓に host の enqueue が入る(約 0.1 µs/poly 悲観側)。A100 の 196.43 µs は画像の表から読んだ値。ntt.cu:398 のコメントが誤解を招く。
- **SEAL 2020**:
  - スケールを強制等化する手口(1e-4 の guard、余裕は 1.36× しかない)を M1/M2 に持ち込まないこと。
  - pow2 鍵モードは鍵切替の回数ではなく論理回転の回数を数えている。
  - Table 2 は**行ごとには**食い違い、合計だけが一致する(出力次元と入力次元の数え方の違い)。Table 2 から推定した規則は「論文の表を再現するだけで、真のコストではない」。
  - `run_timed.sh` は gitignore された lock のパスを固定で使っている(新規 clone では止まる)。
- **ReBoot**: Table 8 のハイパーパラメータは平文ツインの精度実験用で、190.32 s の計時実験に使われたとは限らない。44 GB の根拠になるログは保存されていない。weight decay は RUN.md にだけ書かれている。
- **コストモデル**:
  - ブートストラップの回転数が 216 か 218 か(比への影響 < 0.005)。
  - N = 2^17 の ReBoot 基準は最大限には寛大でない(約 5%)。
  - compact 再配置なしの数はあくまで下限。
  - 鍵切替 1 回の重さ(レベルや dnum による差)と回転鍵のメモリは、数え方の指標に入っていない。
- **RESULTS.md 内の食い違い(報告のみ、元は変更していない)**: Task 4 の Correctness に「11 反復連鎖で ≤ 2.6e-5」とあるが、同じ節の run 3 は alg4 3.6e-5、pow2 1.2e-4 で、この上限を超える。本報告は run 3 の値を使った(いずれも < 1e-3 なので合否には影響しない)。
- **spec 内の定義の揺れ**: §4.2 の RQ1 は「1 サンプル・1 層あたり」、§4.3.2 は「1 サンプル・1 epoch あたり」で、指標も「回転 + ct-ct 乗算」と書かれている。コストモデルはネット全体の 1 サンプルあたり(1 epoch 平均)で数えた。主指標を relin にし、字義どおりの指標を併記している。

## 7. 完了の定義の確認

2026-11-04、リポジトリのルートで `cargo test --workspace` を実行して全 ok(scaffold のため 0 tests)、`cargo deny check licenses` を実行して `licenses ok`(未使用の許可ライセンスについての warning だけ)を確認した。M0 の完了条件「3 つの数字が表にある」のうち、CUDA NTT と ReBoot の epoch 時間は未計測で、その理由(GPU 待ち / メモリ不足)を表に書いた。
