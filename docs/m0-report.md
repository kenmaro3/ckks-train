# M0 報告(2026-11-04)

## 0. 判定(冒頭)

**RQ1(コストモデル、M0 時点): 条件付きで「30% 以上少ない (B, N) は存在する」。ただし ReBoot 自身の設定 (N = 2^16, B = 48) では spec §4.3.2 の基準を満たさない。§4.3.3 の d × B ≤ N/2 を満たす合格点は (64, 2^17) だけで、字義どおりの spec 指標では出荷版 ReBoot に対してだけ合格する。寛大な ReBoot 基準(custom sumRows)に対しては 0.702 で不合格になる。**

**Plan B: M0 の規則では発動しない(Step 2 の「(b) が偽」は成立しない)。ただし spec 指標での合格は、下の前提 1〜3 と ReBoot 基準の選び方に依存する。また M2 の実測(対話版、ブートストラップなし)では決着しない。判断は本人。**

| 指標 | (2^16, 48) の比 diag_batched / reboot | ≤ 0.70 を満たす (B, N)(出荷版 ReBoot に対して) | N = 2^17 で寛大な ReBoot(custom sumRows、37 KS/sample)に対して |
|---|---|---|---|
| (i) 計画の KS = 回転 + relin(ブートストラップ除く) | **0.454** | B ≥ 32 の全 8 点(最良 0.181 @ (128, 2^17)) | B ≥ 32 で合格(0.683 / 0.483 / 0.362 / 0.191) |
| (ii) spec の KS_spec = 回転 + relin + ブートストラップ × BOOT_KS | **0.744(不合格)** | **B ≥ 64 のみ**: (64, 2^16) 0.64*, (64, 2^17) 0.67, (128, 2^16) 0.43*, (128, 2^17) 0.46* | (64, 2^17) **0.691**(合格、余裕 1.3%)、(128, 2^17)* 0.472 |
| (ii') 字義どおり = 回転 + ct-ct 乗算 + ブートストラップ × BOOT_KS | **0.786(不合格)** | B ≥ 64 のみ: 0.68* / 0.68 / 0.49* / 0.48* | (64, 2^17) **0.702(不合格)**、(128, 2^17)* 0.494 |
| (ii) を compact 再配置なしで | 12.9 | なし(全点 ≥ 7.27) | — |
| (iii) 対話版の KS(M2 Task 30/32 のモード。ブートストラップ、compact 再配置、unpack なし)† | **0.222**(W^T をクライアントが渡すと 0.214) | 全 10 点(B ≥ 16。最良 0.089 @ (128, 2^17)) | B ≥ 16 で合格(0.508 〜 0.094) |

† M2 での追記(Task 34): (iii) は M0 の対話版**モデル値**。M2 の実測(Task 32、2^16、B = 32)は 19.78 KS/sample で、ReBoot 37 に対して **0.535**(同じ (B, N) のモデル値 0.29 の 1.81 倍)。補正後の式では B = 48 で約 0.39。差の大部分はモデルに無かったサーバ側の複製(m log B)。同じライブラリ上の ReBoot レイアウトの実測(Task 33)は Lazy 35.06 / PerSample 37 KS/sample。詳細は `docs/m2-report.md` §1・§2・§4。

`*` = diag_batched が 1 ミニバッチに 2 暗号文以上必要(d_pad·B > N/2)。(64, 2^16) と (128, 2^17) は 2 暗号文、(128, 2^16) は 4 暗号文。BOOT_KS = 502 (2^16) / 630 (2^17)。出荷版 ReBoot は 2^17 でスロットを埋める r = 2048(39 KS/sample)。寛大版は使う 1024 行だけを sumRows する(行列を 2 回複製、37 KS/sample。ReBoot の実装ではない)。2^16 では両者は同じ(37)。

(ii)/(ii') の合格は次の 3 つの前提に依存する:
1. **compact W/V 再配置が CKKS で動くこと。** 平文シミュレータでのみ検証済み。これが無いと (2^16, 48) で 12.9×、全 (B, N) で 7.27× 以上悪くなる。
2. **BOOT_KS ≈ 502 の一部は推定。** EvalMod 部分(約 13%)は OpenFHE の式からの推定で、実行して数えたものではない。BOOT_KS が 2 倍(1004 / 1260)だと、B = 64 は 0.76 / 0.78(2^16 / 2^17)で不合格になり、B = 128 だけが残る(0.56 / 0.59)。
3. **B ≥ 64 で ReBoot (B = 48) と同精度(±0.5 pt)が出ること。** 未検証。

このほか §4.3.2 の wall-clock 条件(同一 CPU で ReBoot 公式以下)は、M0 では評価できていない。

**M0 の完了状況:** 完了条件「3 つの数字が表にある」は一部しか満たしていない。
- 表にあるもの: スパイク 1 の CPU NTT(SEAL 比)と、スパイク 2 の SEAL 2020 反復時間。加えてコストモデル (RQ1) と GPU 費用見積り。
- スパイク 1 の CUDA NTT は **GPU 待ち(未実行)**。
- スパイク 3 の ReBoot epoch 時間は **メモリ不足 (OOM) で未計測**。理由は §1 の表に書いた。

**M2 で決着できること・できないこと:**
- M2(Task 30〜33)は対話版で、クライアントが更新のたびに重みを復号・再暗号化する。そのため M2 ではブートストラップも、compact 再配置も、unpack も走らない。(2^16, 48) では、非対話版の 16.8 KS/sample のうち約 51% がこの再配置分(unpack 7.62 + lane shift 0.96)にあたる。
- M2 の実測 (Task 32/33) で決着できるのは、**ブートストラップを除いた KS を対話版モデルと比べること**だけ。対話版モデル (`interactive=True`) は (2^16, 48) で 8.2 KS/sample、比 0.222。Task 32 Step 3 の「10% 以内」の確認も、この対話版モデルと比べる(非対話版の 16.8 とは比べない)。
- spec 指標(ブートストラップ込み、前提 1 の compaction と前提 2 の BOOT_KS を含む)の判定には、M3(ブートストラップ)か、compact 再配置の CKKS 単体テストが要る。§4.3.2 は Plan B の判定を 2027-02-15(M2 完了時)に置くが、その時点の実測だけでは spec 指標を判定できない。

**推奨:** 今は Plan B を発動せず、M1 へ進む。M1(CKKS コア)の中身は本計画でも Plan B でも同じなので、M1 に進む判断は RQ1 の結果に左右されない(この点は筆者の推測)。2027-02-15 の判定に向けては、compact 再配置の CKKS 単体テストを M2 に入れるか、spec 指標の判定を M3 まで延ばすかを決める必要がある(§4 の 6)。**判断は本人。**

判定が「Plan B 寄り」に変わる条件(どれか 1 つで spec 指標の合格が消えうる):
- compact 再配置が、深さ予算(+2 レベル/更新)や精度目標に収まらない。
- B = 64/128 で、ReBoot B = 48 比 −0.5 pt 以内の精度が出ない(B = 48 では合格点が無い)。
- 実測した BOOT_KS や EvalMod の回数が推定より大きい(2 倍で B = 64 が脱落する)。
- 「同一 N」を ReBoot eMLP-1 の 2^16 に限定すると判断した場合、合格点は (64, 2^16)* と (128, 2^16)* だけになる。(64, 2^16) は 2 暗号文、(128, 2^16) は 4 暗号文の構成で、§4.3.3 の基本則 d × B ≤ N/2 を満たす合格点は (64, 2^17) だけ。
- ReBoot 基準を寛大版(custom sumRows)にすると、(64, 2^17) は字義どおりの spec 指標で 0.702 となり不合格、KS_spec でも 0.691(余裕 1.3%)にとどまる。このとき d × B ≤ N/2 を満たし、しかも字義どおりの指標で合格する点は無くなる。
- 同一 CPU の wall-clock で ReBoot に負ける。

## 1. 数字の表(Task 2〜6)

出典はすべて `spikes/RESULTS.md`(以下 R)、`spikes/reboot/RUN.md`、`spikes/reboot/logs-summary.txt`、`spikes/costmodel/table.py`。table.py は最終レビュー修正(2026-11-04)で寛大な ReBoot 基準と対話版の列を加えて再実行し、出力を R の Task 6 節にそのまま貼った。

| スパイク | 項目 | 値 | 比較・基準 | 備考 |
|---|---|---|---|---|
| 1a CPU NTT (Rust) | forward, N = 2^16, q 60 bit, 1 core, 中央値 | **417.8 µs**(アイドル)/ 416.4 µs(負荷下) | 計画の帯 200–400 µs → **約 4% 超過** | スカラー Harvey、SIMD なし。inverse 408.9 / 385.5 µs |
| | SEAL 4.1.2 forward 中央値 | **373.7 µs**(アイドル)/ 368.0 µs(負荷下) | — | `ntt_negacyclic_harvey`、arm64 も SIMD なし |
| | Rust / SEAL(時間の比) | **1.12**(中央値)/ 1.24(min ベース、上方バイアスあり)/ 1.26(SEAL lazy 比) | NTT 単体だけの目安: 性能 = 1/時間 と読むと Rust は SEAL の 373.7 / 417.8 = **0.89 倍**の速さで、M1 完了基準「SEAL 比性能 0.5〜1.0 倍」の帯内 | M1 の基準はライブラリ全体の性能なので、NTT 単体の比は目安にすぎない。負荷下 1.10–1.14 |
| 1b CUDA NTT | batch 64 forward, per-poly | **GPU 待ち(未実行)** | 基準 ≤ 2 × 3.07 = **6.14 µs/poly**(A100) | カーネルの CPU エミュレーションが ntt-cpu とビット一致。sm_80/90 の PTX までコンパイル可 |
| | 参照値 GPU-NTT (HEonGPU の NTT) | A100, N = 2^16, batch 64, radix2-CT: 196.43 µs/batch = **3.07 µs/poly**(論文値) | 4-Step 2.95、RTX 4090 1.52 µs/poly | 出典は Özcan et al., IEEE Access 13 (2025) Table 10。法の種類はキャプションに無い。数値は画像の表から読んだもので、独立には確認していない。HEonGPU / GPU-NTT の ePrint は 403 で読めなかった |
| 2 SEAL 2020 再現 (Iris 4-10-3, N = 2^15) | 1 反復(server、alg5、exact keys) | **0.997 ± 0.027 s**(アイドル)/ 1.031 s(負荷下) | — | alg4 1.027 s、pow2 keys 1.168 s(アイドル) |
| | server + 再パック + クライアント暗号化 | **1.369 s**(アイドル)/ 1.415 s(負荷下) | 論文 Table 4 の記載合計 9.25 s → **6.8×**(負荷下 6.5×) | 再パック 0.334 s、暗号化 0.038 s |
| | 論文 Table 4 自体の不整合 | 対角: 記載 9.25 s ≠ 相の和 9.10 s。行: 28.47 s ≠ 26.01 s | 差の理由は論文に無い | 記載合計との比を主にし、相ごとの比は参考扱い(相和比 8.6×) |
| | 演算数(6-3-1 net)と論文 Table 2 | ours alg5: 23 ct-ct + 1 pt / 16 rot。alg4: 23 + 1 pt / 11 | 論文 17 mult / 17 rot | 両方が計算している行の乗算数は一致する。差は計上範囲(ΔW を除外、W1ᵀ を含む)と、回転数の数え方(出力次元か入力次元か)による。行ごとには食い違う(§6) |
| | 演算数(Iris 4-10-3、alg5) | 35 ct-ct / 35 relin / 31 rot / 36 rescale | コストモデル diag_2020 と一致 | — |
| | 暗号化学習の誤差 | 1 反復 max\|W_he − W_ref\| = **4.4e-6**。11 反復連鎖 (run 3): 1.4e-5 (alg5) / 3.6e-5 (alg4) / 1.2e-4 (pow2) | 合格線 < 1e-3 | 誤差の主因はスケールの強制等化 |
| | 平文精度 (400 epoch, batch 20) | train **0.982** / test 0.960 ± 0.021(10 seeds) | 論文 0.9805 / loss 0.0249 | batch 1 + lr 0.1 は発散(10/10) |
| 3 ReBoot (eMLP-1, MNIST) | ビルド | **OK**(OpenFHE 1.2.1、ReBoot `3ac173f`、移植用の 3 パッチ。パッチは gitignore 下にだけ置く) | — | ReBoot は **GPL-3.0**。実行するだけのベースラインで、コードは hekit に一切コピーしていない |
| | 実行 | **メモリ不足 (OOM) で未計測** | 機械は 36 GB RAM | context 生成直後の footprint ≥ **44 GB**(`footprint` で対話的に読んだ値で、ログは無い)。`/usr/bin/time -l` の peak footprint は 4 回の実行で **51–54 GB**(`spikes/reboot/logs-summary.txt`)。最初の反復の重みブートストラップ(eMLP-1 は W 2 + V 2 = 4 暗号文)で kill |
| | epoch 時間 / 精度 | **メモリ不足で未計測** | 論文値: **190.32 s/iter**、ピーク **76.07 GB**(2× Xeon 5318S, 384 GB) | 論文値からの算術: 100 it ≈ 5.3 h、全 MNIST 1250 it ≈ 66 h。test acc 94.61 %(論文値、平文ツイン) |
| | 設定の転記 | `crates/hekit-bench/configs/reboot_emlp{1,2,3}.json` | 784-32-10、2^16、l 24、Δ 2^49、B 48、lr 1e-3、mom 0.9 | epochs = 10 はリポジトリのスクリプト由来で論文には無い。weight decay は JSON に無い。emlp2/3 はブロック構造を持たない(§6) |
| 4 コストモデル | ReBoot eMLP-1 / sample | 30 rot + 7 relin = **37 KS**、ブートストラップ 4 / mini-batch | lazy relin にした仮想 ReBoot は 35。2^17 は出荷版 39、寛大版 (custom sumRows) 37 | (r, c) = (1024, 32) は ReBoot のログと一致 |
| | diag_batched (2^16, 48) / sample | KS 16.8、KS_spec 58.6(ReBoot 78.8) | 比 0.454 / **0.744** | 2 暗号文/batch |
| | diag_batched (2^16, 48) 対話版 / sample | KS **8.2**(サーバが転置)/ 7.9(クライアントが W^T を渡す) | 比 **0.222** / 0.214 | M2 Task 32 の 10% 確認の比較対象 |
| | diag_batched (2^16, 64) | KS_spec 44.0 vs 68.4 → **0.64**。字義どおりの指標で 0.68 | — | 2 暗号文/batch |
| | diag_batched (2^17, 64) | KS_spec 52.8 vs 78.4 → **0.67**。字義どおりで 0.68 | 寛大な ReBoot (76.375) に対しては KS_spec 0.691、字義どおり **0.702(不合格)** | 1 暗号文 |
| | 検証 | 2020 Table 2 の合計 17/17 と row 38/63 を再現。Task 4 の SEAL 実測 35/35/31 とも一致 | 行ごとの回転数は strict xfail | pytest 48 pass + 1 xfail。対話版と custom sumRows もシミュレータで式と一致 |

## 2. (a) CUDA NTT は HEonGPU の 2 倍以内か

**未判定(GPU 待ち)。** 判定には A100 80GB を 1 枚、約 1 時間借りれば足りる(下の価格で約 $1–5)。H100 で測ると A100 基準の 6.14 µs は甘めになる。

```sh
cd hekit/spikes/ntt-cuda
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit --format=csv
cargo test --release -- --include-ignored --nocapture   # GPU 一致テストで SKIP が出ないこと
cargo run --release --bin bench                          # N=2^16, batch 64, 50 reps
cargo run --release --bin bench -- 16 64 200 20          # 再現性の確認
```

前提: driver が CUDA ≥ 12.4 に対応していること(cudarc の `cuda-12040` バインディング)。また NVRTC が driver より新しくないこと。`ntt_fwd_shared` には最終レビュー修正で `__launch_bounds__(1024)` を付けた(LAUNCH_OUT_OF_RESOURCES 対策。GPU ではまだ未確認)。

## 3. (c) GPU 費用見積り(M4 = 200 GPU 時間)

公開価格ページと公開 API だけを読んだ(2026-11-04 アクセス。登録・ログイン・購入はしていない)。価格は 1 GPU 1 時間あたりの USD。

| プロバイダ | GPU(構成) | オンデマンド $/h | spot / community $/h | 出典 |
|---|---|---|---|---|
| Vast.ai(マーケット) | A100 SXM4 80GB ×1 | 0.63 – 1.60(中央値 1.09、11 件) | 最低入札 0.27〜(中央値 0.93) | 公開検索 API のスナップショット(下記)。`https://vast.ai/pricing` は数値が描画されず読めなかった |
| Vast.ai | A100 PCIe 80GB ×1 | 0.87 – 1.00(5 件) | 0.13〜 | 同上 |
| Vast.ai | H100 SXM 80GB ×1 | 1.34 – 3.41(中央値 2.40、7 件) | 0.40〜 | 同上 |
| RunPod | A100 80GB PCIe / SXM | 1.79 / 1.79(Secure) | 1.19 / 1.39(Community) | https://www.runpod.io/pricing |
| RunPod | H100 80GB PCIe / SXM / NVL | 2.89 / 3.99 / 3.19(Secure) | 1.99 / 2.69 / 2.59(Community) | 同上 |
| Lambda | H100 80GB PCIe / SXM ×1 | 3.29 / 4.29 | — | https://lambda.ai/pricing(1 枚構成の A100 は 40GB の $1.99 だけで、80GB は無い) |
| GCP us-central1 | A100 80GB (a2-ultragpu-1g, VM 込み) | 5.07 | 3.01 (spot) | https://gcloud-compute.com/us-central1/a2-ultragpu-1g.html(GCP 価格の集計サイト、データ更新 2026-11-01)。GCP 公式ページは JS アプリに転送されて読めなかった |
| GCP us-central1 | H100 80GB (a3-highgpu-1g) | 11.06 | 6.59 (spot) | https://gcloud-compute.com/us-central1/a3-highgpu-1g.html |
| AWS us-east-1 | H100 80GB (p5.4xlarge) | 6.88 | 4.51 (spot) | https://instances.vantage.sh/aws/ec2/p5.4xlarge(公式の p5 ページには価格が無い)。A100 80GB の 1 枚構成は AWS に無い(p4de は 8 枚) |

Vast.ai の数値は、ログイン不要の公開検索 API `https://console.vast.ai/api/v0/bundles/` を **2026-11-04 16:32 JST に 1 回だけ叩いた時点のスナップショット**で、時間とともに変わる。
- クエリ: GPU 名ごとに 1 回(A100 SXM4 / A100 PCIE / H100 SXM / H100 PCIE)。条件は 1 GPU、rentable、on-demand。クエリ文字列そのものは保存しておらず、条件は Task 7 の作業記録による(応答の全件が 1 GPU・rentable であることは確認した)。
- 取得後に gpu_ram ≥ 70 GB で絞り、`dph_total` を時間単価、`min_bid` を最低入札とした。
- 生の出力はリポジトリに入れず、M0 の作業記録(Task 7 の報告)に置いた。

200 時間の費用(表の 2 桁に丸めた単価 × 200):

| GPU | 最安(マーケット / community) | 専用クラウド(Secure / Lambda) | ハイパースケーラ spot〜オンデマンド |
|---|---|---|---|
| A100 80GB | **$126 – 320**(Vast)/ $238 – 278(RunPod Community) | $358(RunPod Secure) | $602 – 1,014(GCP) |
| H100 80GB | $268 – 682(Vast)/ $398 – 538(RunPod Community) | $578 – 858(RunPod Secure, Lambda) | $902 – 1,376(AWS)/ $1,318 – 2,212(GCP) |

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
| GCP n2-highmem-16 | 16 / 128 GB | 1.048 | 0.629 | $10.5 | $31 | https://gcloud-compute.com/us-central1/n2-highmem-16.html |

- 論文のマシンは 2× Xeon 5318S(48 コア)。16 vCPU ではおそらく数倍遅くなるので、eMLP-1 の 1 epoch(論文速度で 5.3 h)には 10〜30 h を見込む。
- 推奨は **r7i.8xlarge のオンデマンド(約 $20–65)**。256 GiB あれば eMLP-2/3 まで載る。ジョブにチェックポイントが無いので spot は避ける。
- §4.3.2 の「同一 CPU での wall-clock」比較もこのホストで行うことになる(36 GB の Mac では ReBoot が動かない)。

## 4. M1 に進む前に本人が決めること

1. **GPU レンタル**: プロバイダと予算上限(推奨は RunPod / Vast の A100 80GB で ≤ $400)。最初の用途は spike 1b の判定(約 1 h)。
2. **大メモリ CPU ホスト**: ReBoot ベースラインと同一 CPU での比較用。プロバイダと時間(推奨は r7i.8xlarge で 30 h、約 $65)。やるのは M0 の追加測定か、M2 の比較時か。
3. **条件付きの RQ1 判定を受け入れて M1 に進むか**(推奨: 進む)。
4. **B ≥ 64 を論文の主設定にしてよいか**: §4.3.3 の「ReBoot の値も併走」は維持し、B = 48 では spec 指標を満たさないと正直に書くことになる。
5. **「同一 N」の解釈**: eMLP-1 は 2^16 に固定するか、2^17 での比較も合格に数えるか。
6. **spec 指標をいつ、何で判定するか**: M2 の対話版ではブートストラップが無いので、M2 の実測で判定できるのは対話版 KS だけ(§0)。選択肢は 2 つ。(a) M2 の最初に compact W/V 再配置を CKKS 上で単体テストする(spec 合格の前提 1)。(b) spec 指標の判定を M3(ブートストラップ)まで延ばす。どちらにするかで、§4.3.2 の 2027-02-15 判定の中身が変わる。BSGS で giant step 1 回につき relin が n2 回必要になる点(設計の前提)も、M1 の BSGS API に反映する。
7. **ReBoot 基準をどちらで数えるか**: 出荷版 ReBoot(2^17 で r = 2048、39 KS/sample)か、寛大版(custom sumRows、37 KS/sample)か。寛大版にすると、(64, 2^17) は字義どおりの spec 指標で不合格(0.702)になる。
8. CUDA カーネルは未実行。M1 の途中から CUDA に着手する計画(§5 M4)の前に、spike 1b を回すかどうか。

## 5. M0 中の判断(ルーリング)

- **T1**: ワークスペースで `exclude = ["spikes"]`。cargo-deny は既定の toolchain で入れた。deny.toml で `tfhe` / `concrete*` を禁止(最終レビュー修正で `tfhe-*` も追加)。
- **T3**: GPU のレンタルはしなかった(課金を伴うため)。「1 ブロック = 1 多項式を共有メモリ」は N = 2^16 で不可能(512 KiB > 228 KiB)なので、大きいストライドの 5 ステージを global memory、残り 11 ステージを共有メモリで処理する分割に変えた。カーネルの CPU エミュレーションを ntt-cpu とビット一致で検査した。
- **T4**: 追加で SEAL NTT と SEAL の演算数を測った。論文 Table 4 の合計と相の和が合わないので、比は記載合計に対して出した。負荷下(load 17–68)の計測をアイドル時 (load ≈ 3) に再実行し、表はアイドル値を主にした。
- **T5**: ビルドや実行に失敗したら原因を記録して先へ進む。1 epoch が 2 h を超える見込みなら有限バッチで外挿する、という上限を設けた。実際には OOM で外挿もできなかった。
- **T6**:
  - `diag_2020` を独立したレイアウトキーにした。計画の「diag_batched の B = 1」とは式が両立しないため。
  - d_pad·B > N/2 のときは ceil 個の暗号文で数え、表で印を付ける。
  - KS(計画の定義)と KS_spec(BOOT_KS 込み)を併記する。ReBoot は寛大にモデル化する。
  - 修正ラウンドで次を追加した: compact 再配置なしの列、字義どおりの指標、判定文を ReBoot 自身の (N, B) の spec 結果から書き出す形への変更。
- **ReBoot のライセンス**: ReBoot は **GPL-3.0** なので、実行するだけのベースラインとして扱う。コードは hekit に一切コピーしない。ビルド用パッチも gitignore 下にだけ置き、コミットしない。コストモデルが ReBoot のファイルを引くのは、演算数を読んだ場所を示すためだけ。**M2 Task 33 は ReBoot のレイアウトを論文(§3、式 4–5)から実装し直す。コードからは実装しない。**
- **最終レビュー修正(2026-11-04)**: コストモデルに 2 つの変種を足した。1 つは寛大な ReBoot 基準 (`custom_sumrows=True`)、もう 1 つは対話版 (`interactive=True`、既定ではサーバが回転で転置し、`client_transpose=True` も選べる)。どちらもシミュレータで式と一致することを確かめた。§0 の判定はこの再生成した表から書き直した。CI の deny ジョブは `licenses bans` を走らせるようにした。

## 6. 既知の小さな問題(M1 前に知っておくこと)

「(レビュー由来)」は M0 のタスク別レビューで指摘された項目。行番号は現在のファイルの行。

- **CI**: deny ジョブは `cargo deny check licenses bans` を走らせ、tfhe / concrete* の禁止を強制する(**修正済み**)。cargo-deny-action@v2 のタグは固定していない。CI は一度も走っていない(remote が無い)。
- **NTT CPU**(レビュー由来): bit-reversed 出力順がテストで固定されていない。最悪ケースの入力(全係数 q−1 や 0)のテストも無い。
- **NTT CUDA**(レビュー由来):
  - `ntt_fwd_shared` に `__launch_bounds__(1024)` を付けた(**修正済み**。GPU では未確認)。
  - ntt.cu:53 の「requires t >= blockDim.x」というコメントは、正しさの前提ではなく性能上の性質だと書き直した(**修正済み**)。
  - 計時窓に host の enqueue が入る(約 0.1 µs/poly 悲観側)。
  - A100 の 196.43 µs は画像の表から読んだ値。
- **SEAL 2020**:
  - スケールを強制等化する手口(1e-4 の guard、余裕は 1.36× しかない)を M1/M2 に持ち込まないこと。
  - pow2 鍵モードは、鍵切替の回数ではなく論理回転の回数を数えている。
  - Table 2 は**行ごとには**食い違い、合計だけが一致する(出力次元と入力次元の数え方の違い)。R にある Table 2 から推定した規則には、「論文の表を再現するだけで、真のコストではない」と明記した(**修正済み**)。
  - `run_timed.sh` の lock パスは `BENCH_LOCK` で変えられる。既定は spike 内の gitignore された `build/bench.lock` で、親ディレクトリも作る。`set -euo pipefail` を入れた(**修正済み**、レビュー由来)。
  - he_train.cpp:891 の古い「lock-step」コメントを、実際の動作(参照モデルは HE モデルに合わせ直さない)に合わせた(**修正済み**、レビュー由来)。
- **ReBoot**:
  - ReBoot は GPL-3.0(§5)。
  - Table 8 のハイパーパラメータは平文ツインの精度実験(論文 Table 4)用で、190.32 s の計時実験と同じとは限らない。RUN.md にも書いた(**修正済み**)。
  - 44 GB の footprint は対話的に読んだ値で、ログは無い(レビュー由来)。代わりに、自前の `/usr/bin/time -l` の要約行(peak footprint 51–54 GB)を `spikes/reboot/logs-summary.txt` としてコミットした(**修正済み**)。
  - weight decay は RUN.md にだけ書かれている。
  - **`reboot_emlp2/3.json` は eMLP-2/3 に忠実ではない。** 論文 Table 3 では、eMLP-2 = RE-Block(64) → CE-Block(32) で、ブロックごとに局所分類器(CE-FC / RE-FC)と局所損失を持ち、ブロック間に勾配は流れない。JSON は素の BP の形(784-64-32-10 など)しか表せない。スキーマにブロックのフィールドが入るまでは eMLP-2/3 として使わないこと。eMLP-1 は単一ブロックなので影響を受けない(RUN.md に記載)。
  - RUN.md にあった「重みブートストラップは 64 暗号文」は誤りで、正しくは 4 暗号文(W 2 + V 2、`models.py:121-133`)。訂正した(**修正済み**)。
- **コストモデル**:
  - ブートストラップの回転数が 216 か 218 か(レビュー由来)は、修正ラウンド 1 で 218 に確定した(**修正済み**)。
  - N = 2^17 の寛大な ReBoot 基準(約 5% 安い 37 KS/sample)は、モデルと表に入れた(**修正済み**)。それにより判定が 1 点変わった(§0)。
  - R:485 の「Per bootstrap at N = 2^16」は 2^15 スロットのことだと明記した(**修正済み**、レビュー由来)。
  - compact 再配置なしの数は、あくまで下限。
  - 鍵切替 1 回の重さ(レベルや dnum による差)と回転鍵のメモリは、数え方の指標に入っていない。
- **RESULTS.md 内の食い違い**: Task 4 の Correctness にあった「11 反復連鎖で ≤ 2.6e-5」は、同じ節の run 3(alg4 3.6e-5、pow2 1.2e-4)と矛盾していた。元の文を「runs 1–3 で ≤ 1.2e-4」に直し、run ごとの値を併記した(**修正済み**)。合否には影響しない(いずれも < 1e-3)。
- **spec 内の定義の揺れ**: §4.2 の RQ1 は「1 サンプル・1 層あたり」、§4.3.2 は「1 サンプル・1 epoch あたり」で、指標も「回転 + ct-ct 乗算」と書かれている。コストモデルはネット全体の 1 サンプルあたり(1 epoch 平均)で数えた。主指標を relin にし、字義どおりの指標を併記している。

## 7. 完了の定義の確認

2026-11-04 の最終レビュー修正の後、リポジトリのルートで次を実行し、すべて通った。
- `cargo test --workspace`: 全 ok(scaffold のため 0 tests)。
- `cargo fmt --all --check`。
- `cargo clippy --workspace --all-targets -- -D warnings`。
- `cargo deny check licenses bans`: `bans ok, licenses ok`(未使用の許可ライセンスについての warning だけ)。

`cargo deny check licenses bans` は、`spikes/ntt-cpu` と `spikes/ntt-cuda` の Cargo.lock に対しても手で実行し、どちらも `bans ok, licenses ok` だった(ルートの deny.toml を使う)。CI は一度も走っていない(git の remote が無い)ので、上記はすべてローカルでの確認。

M0 の完了条件「3 つの数字が表にある」のうち、CUDA NTT と ReBoot の epoch 時間は未計測で、その理由(GPU 待ち / メモリ不足)を表に書いた(§0)。
