# ckks-train 引き継ぎ書(旧 hekit、2026-11-07 時点)

最新の設計: [CPU準備の所有権・計上](#latest-evidence-20261107-d7-preparation-design)。最新の実装: [D7 ExecutionBinding内部descriptor](#latest-evidence-20261107-d7-execution-descriptors)。実行環境: [D7実行環境descriptor](#latest-evidence-20261107-d7-descriptors)。入力provenance: [Train接続](#latest-evidence-20261107-d7-train-capture)。CPU学習経路: [P5限定実測](#latest-evidence-20261107-p5-cpu)。比較テストの準備: [P5複数層平文感度検証](#latest-evidence-20261107-p5-preflight)。最新の運用記録: [配置・読戻し](#latest-evidence-20261107-r29)、[FULL-2の定期確認](#latest-evidence-20261107-1952)。

現在の実装・運用状態は末尾の追記を優先する。§5、§11、§12.1〜12.4は各時点の記録であり、後続キューの現在状態や新しいCI結果をそこから推定しない。

次の担当エージェントが、この文書と参照先だけで残りのタスク(Task 42 の仕上げ、47 の再計測、48〜58)を進められるように書く。日本語、技術用語は英語。

## 0. 30 秒で分かる現在地

- **目的**: 2020 年論文(Mihara et al., arXiv:2012.13552、CKKS で NN 学習)を、自作の MIT ライセンス Rust RNS-CKKS ライブラリ **ckks-train**(bootstrapping + CUDA)で拡張し、論文 1 本(WAHC 2027、arXiv + v1.0 を 2027-05-31 まで)を書く。
- **主ベースライン**: ReBoot(AAAI-26、OpenFHE、GPL-3.0。実行のみ、コードは一切取り込まない)。
- **RQ1**(研究計画 §4.3.2): 同じ N・セキュリティ・モデル・精度 ±0.5 pt で、1 サンプルあたりの key switch が ReBoot より 30% 以上少なく(比 ≤ 0.70)、同一 CPU の wall-clock が ReBoot 公式以下。
- **進捗**: FULL-1の結果をM3へ反映済み。Task42/RQ1はFULL-2と未達・未計測項目の裁定待ち、R47/R48/R52は既存post-fullキュー待ち。Task54は全実験行列の基盤を実装中。crate改名・公開準備とpaper草稿は存在するが、M4/M5全体・v1.0公開の完了は主張しない。
- **暫定判定**(784 入力 subset、サーバー実測): WhenNeeded B=64 は key switch 3 通りすべてと wall-clock(0.447 倍)で合格。精度 ±0.5 pt の確定は本番走行 R-41-FULL-2 待ち(2026-11-07 深夜〜11-08 に完了見込み)。
- **GitHub**: `kenmaro3/ckks-train`(private、2026-11-06 に初回 push。ローカルは `/Users/marconi/Documents/research/hekit`、branch `main`)。

## 1. 必ず読むファイル(順番どおり)

以下の `plans/` は、リポジトリの親にある `/Users/marconi/Documents/research/plans/` を指す。`hekit/` はこのリポジトリである。

| 順 | ファイル | 何が分かるか |
|---|---|---|
| 1 | `plans/2026-11-04-he-training-research-plan.md` | 研究計画(spec)。RQ1〜3、評価基準、制約 |
| 2 | `plans/2026-11-04-he-training-implementation-plan.md` | Task 1〜34(M0〜M2)。Global Constraints は M3 以降も有効 |
| 3 | `plans/2026-11-04-he-training-implementation-plan-m3-m5.md` | Task 35〜58。M3 ルーリングの注記入り |
| 4 | `hekit/docs/m3-report.md` | **最重要**。RQ1 表、4 通りの比、裁定 R-M3-1〜15 の全文(§8)、未決事項(§9) |
| 5 | `hekit/docs/m2-report.md`, `m1-report.md`, `m0-report.md` | 各フェーズの数字、逸脱、ルーリング |
| 6 | `hekit/docs/layout.md` | スロット配置、転置、fold、コストモデル閉形式(§7.5)、鍵の本数 |
| 7 | `hekit/docs/bootstrap.md` | n16_boot の鎖、SSE、EvalMod、精度の論証、実測 |
| 8 | `hekit/docs/security.md` | lattice-estimator の全結果(hekit + ReBoot) |
| 9 | `hekit/docs/m4-report.md` | M4 草稿。GPU 対 CPU 対 HEonGPU の表 |
| 10 | `hekit/docs/quadruped-log.md`, `hekit/results/quadruped/requests.md` | サーバーで何を走らせたか、キューの状態、依頼の書式 |
| 11 | `hekit/.superpowers/sdd/*/progress.md` | 実装セッションの ledger(タスクごとの dispatch / review / ruling) |

## 2. リポジトリとビルド

```
hekit/
  crates/ckks-train-core   RNS 算術、NTT、サンプラ(ChaCha20 のみ)
  crates/ckks-train-ckks   RNS-CKKS、hybrid key switching、hoisting、BSGS、bootstrapping(bootstrap/)、OpCounter
  crates/ckks-train-nn     学習層: layout、Dense、転置、勾配、NAG、Trainer(対話)、trainer_ni(非対話)、compact、layout_reboot(ベースライン c)
  crates/ckks-train-cuda   CUDA バックエンド(cudarc 0.19.0、dynamic-loading、NVRTC)。feature `cuda` で有効、無くても check は通る
  crates/ckks-train-bench  bins: iris, mnist, mnist_reboot_layout, plain。configs/*.json。コストモデルの移植
  docs/  results/  results/quadruped/  spikes/(M0、workspace 外)  tools/(seal-kat, estimate, bootstrap)  kat/
```

- toolchain: `rust-toolchain.toml`(1.85.1)、edition 2024。`[profile.test]` opt-level 3。`cargo deny check licenses` 必須(allow: MIT, Apache-2.0, BSD-*, ISC。GPL と BSD-3-Clause-Clear は拒否)。
- 通常のテスト: `cargo test --workspace`(約 400 件)。n16_boot の 10 GB 超のテストは `#[ignore]`(`cargo test -- --ignored` で手動)。
- 並列: feature `parallel`(rayon)。ブートストラップは `HEKIT_BOOT_THREADS` で本数指定。学習の非ブートストラップ経路は部分的にしか並列化されていない(サーバーの F1/F2 で CPU 使用率 272〜308%、ReBoot は 518%。M4 の改善対象)。
- GPU テスト: `cargo test -p ckks-train-cuda --features cuda`。各テストは必要 GPU メモリを宣言し、足りなければ SKIP 行を出す。
- データ: `scripts/fetch_mnist.sh`、`scripts/fetch_fashion_mnist.sh`(gitignore 下の `data/`)。Iris は同梱(CC BY 4.0)。
- 主な実行例(引数は各 bin の `--help`):
  - `cargo run --release -p ckks-train-bench --bin iris -- --epochs 400 --batch 20 --seed 2020 --out results/iris_20.json`
  - `cargo run --release -p ckks-train-bench --bin mnist -- --config crates/ckks-train-bench/configs/reboot_emlp1_faithful.json --mode noninteractive --placement when_needed --batch 64 ...`(`--dataset-npy`, `--init-weights`, `--lr-schedule`, `--lr-schedule-batch 48`, `--epoch-orders`, `--plan` あり)
- 結果 JSON の規則: バイナリが自分で `provenance`(git commit、dirty files、コマンド行、スレッド数、load、RSS、時刻)を書く。**数字を手で書き換えない**。

## 3. 確定した仕様(計画と違う点を含む)

### 3.1 パラメータプリセット(`ckks-train-ckks/src/params.rs`)

| preset | N | q(レベル 0 →) | P | dnum | log PQ | 用途 |
|---|---|---|---|---|---|---|
| `seal_2020` | 2^15 | 60, 40×9 | 60 | 10 | 480 | 2020 年再現(Iris)。SEAL 4.1.2 と bit-exact |
| `n16_train` | 2^16 | 58, 42×31 | 58×4 | 8 | 1592 | 対話モードの MNIST。BDGL16 143.9 / MATZOV 141.2 bit |
| `n16_boot` | 2^16 | 54, 42×16(計算), 34×4(StC), 58×7(EvalMod), 56×4(CtS) | 60×4 | 8 | 1732 | 非対話(bootstrapping)。ブートストラップ後 16 レベル。密鍵 131.5 / 129.3 bit、SSE 一時鍵(h'=32)133.8 / 134.2 bit |
| `n16_boot_sparse` | 同上、主鍵 h=192 | | | | | **使わない**(122.2 / 121.7 bit < 128)。代替として記録のみ |
| `test_tiny` | 2^5 | | | | | テスト専用、安全でない |

- 素数の割り当て順は SEAL の `CoeffModulus::Create` と同じ(`pop_back`)。Galois 生成元は 3(SEAL と同じ。計画の 5^k ではない)。
- `add` のスケール許容 2^-10(黙って付け替える)。学習コードは厳密スケール経路(`mul_scalar` の目標スケール引数、平文定数は `ct.scale()` で符号化)を使い、この誤差を入れない。
- `Arc<CkksParams>` は 1 つだけ作って共有(別 Arc は `InvalidParams`)。

### 3.2 ブートストラップ(Task 35〜38)

- SSE(sparse-secret encapsulation、Bossuat et al. ACNS 2022): level 0 で h'=32 の一時鍵に切り替え → mod-raise → CoeffToSlot(4 レベル、56 bit 素数)→ EvalMod(7 レベル、Chebyshev 次数 127、double angle なし、level-optimal scale-invariant 評価)→ SlotToCoeff(4 レベル、34 bit 素数 × 4)→ 元の鍵へ戻す。
- 実測: 精度 **17.46 bit**(ゲート 16、モデル 17.4)、**111 key switch / boot**(回転 69、relin 40、SSE 2)、出力は level 16・scale ちょうど Δ。鍵 33 本 10.27 GB + 対角 ≈ 3 GB。時間: Mac 23.8 s(1 スレッド)/ 7.70 s(12)、サーバー 45.4 s / 13.95 s。
- `OpCounter` は `rotations, relins, sse_switches, ct_ct_tensor, ct_pt_mults, rescales, bootstraps` を持ち、`key_switches() = rotations + relins + sse_switches`。ブートストラップの中身も全部数える。

### 3.3 学習レイアウト(Task 25〜41)

- 特徴メジャー `slot(j, b) = j·stride + b`。BSGS は stride 付き、giant step ごとに relin 1 回(`matvec_enc` の relins == giant)。転置は回転と加算だけ(鍵は Reuse、+12 KS/更新)。
- 複製ブロックは refresh 時にブロック平均を取らないと指数的に乖離する(M2 で発見)。非対話版は単一ブロック + compact 配置で回避。
- compact repacking(Task 40): 1 層の対角 32 本を 1 暗号文に詰めて 1 回でブートストラップ。pack 31 回転、unpack 191 回転、relin 0。**融合経路(R-M3-11)**: 勾配の有効レーンマスクと pack のマスクを 1 回の平文乗算に統合(別々だと +1 レベル、+54% KS)。
- ブートストラップ配置(`BootPlacement`): `AfterUpdate`(AU、毎更新 4 回)、`WhenNeeded`(WN、平均 2 回。計画に無い hekit の選択肢、**見出し**)、`AfterActivation`(AA、3 回)。
- B は 2 の冪のみ(32 = 1 暗号文、64 = 32 レーン × 2 マイクロバッチを足してから 1 更新)。B=48 は 64 レーンに 48 を載せて走らせる(FULL-2)。
- NAG の速度は基準 lr(スケジュール最大)で保持し、各ステップは倍率 ≤ 1 だけ掛ける(lr が 843 倍まで下がる ReBoot のスケジュールで、ブートストラップ誤差が増幅されないため。+0.72 KS/サンプル)。

### 3.4 ReBoot に忠実な設定(R-M3-12〜15)

- `configs/reboot_emlp1_faithful.json`: 入力 **196**(14×14。784 ではない)→ 32(z²+z)→ 10(16 にパディング)。勾配はバッチで**和**(lr_mean = 5e-4·B)、momentum 0.9、バイアスなし(暗号化した 0 を凍結)、Xavier 初期値。初期重み・サンプル順・学習率スケジュール(5 epoch 一定 → cosine with restarts)は ReBoot の公開コードから**データとして**取り出したもの(`results/quadruped/raw/reboot_emlp1_full/`、`data/reboot_emlp1_full/`)。
- hekit の平文双子は ReBoot の公開コードのフル設定(4800 サンプル × 10 epoch、10 シード)を**完全再現**(1000 バッチの accuracy 一致、test 平均 0.90648 ± 0.00271)。
- 精度 ±0.5 pt の参照値は、この公開コードの値(seed 0: 0.9041)。論文の Table 4 の 94.61 ± 0.27 は公開コードの既定設定では出ず、論文に書かれた設定(60,000 枚、lr 0.001、wd 0.001)で回すと epoch 固定で +0.33〜+0.64 pt(最良 epoch で +0.74〜+0.85)。報告には注記付きで併記する。
- ReBoot の実パラメータ(R-REB-4): N=2^16、Q 26 本(log Q 1249.6)、P 60×8、dnum 3、SPARSE_TERNARY h=192、σ 3.19、log PQ 1729.6。同じ推定器での推定は実行中(hekit セッション)。

### 3.5 CUDA(Task 43〜47)

- `HeOps` トレイト(Task 43)で evaluator を抽象化。`EvalConfig { hoist, bsgs, lazy_relin }` で RQ3 の ablation。d=16 の回数(`ops.rs` の「What the RQ3 table must state」が正本): 既定 = hoist 可能な回転 3 + 個別回転 3、relin 4。bsgs=false = 3 + 12、relin 13(事前回転レイアウトのまま。教科書版は 15 + 0、relin 1 で未実装)。全オフ = 回転 15、relin 16。bsgs の利得は上限、bsgs オフ内の hoist / lazy の利得は下限。hoist は OpCounter を変えない(wall-clock のみ)。
- `ckks-train-cuda`: `CudaDevice`、`DevicePoly`、`DeviceKsk`(capped keys: level 9 で 42 MiB、level 5 で 20 MiB、full 288 MiB)、`CudaEvaluator`(BSGS、マルチストリーム)。NVRTC は `--fmad=false`。CPU と **bit-exact**(45/45、想定スキップ 3)。
- GPU メモリプールは使用中 + 256 MiB までしか保持しない(`DEFAULT_POOL_RELEASE_THRESHOLD`、共有 GPU のため)。Task 48 の学習は epoch 境界で `trim_pool(0)`。
- 4 ストリームのバッチは逐次より遅い(41.7 対 35.0 ms、N=2^16 では 1 演算で GPU が飽和)。Task 48 は逐次で進める。
- HEonGPU 比較: 4090 の公表値は ePrint 2025/124 Table 3 の relin 2581.8 µs のみ。hekit は 4.67 ms(1.81 倍、同 ℓ で約 2.0 倍、目標の「2 倍以内」ぎりぎり)。ePrint 2024/1543 は PDF が 403 で未確認。4090 の CKKS bootstrapping は HEonGPU docs で Regular 109.28 ms / Slim 38.55 ms(Task 49 の目標値)。

## 4. 計測結果の要約

### 4.1 key switch / サンプル(eMLP-1、N=2^16、サーバー i9-13900KF 12 スレッド)

| 行 | 入力 | 設定 | KS/サンプル(boot 込み) | boot/更新 | epoch 時間 | test acc |
|---|---|---|---|---|---|---|
| 対話(M2) | 784 | lr 0.001、B=32、1 スレッド | 19.78(boot 無し) | 0 | 10894 s(Mac 5004 s) | 0.332 |
| AU B=32 | 784 | 旧設定 | 39.47 | 4 | 4339 s | 0.332 |
| AU B=64 | 784 | 旧設定 | 20.64 | 4 | 2229 s | 0.193 |
| WN B=32 | 784 | 旧設定 | 32.53 | 2 | 2774 s | 0.3323 |
| WN B=64 | 784 | 旧設定 | 17.17 | 2 | 1435 s | 0.1929 |
| AA B=32 | 784 | 旧設定 | 36.0 | 3 | 3607 s | 0.332 |
| **WN B=32 忠実** | 784 | ReBoot 添字・初期重み | 33.49 | 17/32 更新 | 3017 s | 0.7577(双子一致) |
| **WN B=64 忠実** | 784 | 同上 | 17.89 | 9/16 更新 | 1608 s | 0.7358(双子一致) |
| ReBoot 公式 subset | 784 | OpenFHE、B=48、12 スレッド | 37(モデル)+ 4×502/48 | 4(502 KS/boot はモデル値) | 約 149 s/step × 21 + 112.6 s | 0.7506 |
| **FULL-1** WN B=64 | 196 | 4800×10 epoch、seed 0 | 予測 18.27 | | 約 19.6 h(走行中) | pending |
| **FULL-2** B=48 in 64 | 196 | 同上 | 予測 28.99 | 毎更新 | 約 38.6 h | pending(±0.5 pt 判定) |

- 全走行で実測 KS は閉形式(コストモデル + Task 32/41 の補正)と**完全一致**。M0 のコストモデルそのものは 1.81 倍の過小予測だった(補正項は `layout.md` §7.5)。
- 比の 4 通り(M3 報告 §2): (a) 各自の boot コスト(hekit 111 実測、ReBoot 502 モデル)、(b) 両側 502、(c) 両側 111、(d) 両側 boot 除外(同じ非対話実行の実測カウンタから bootstrap 分を除く: ((K − 111·H)/S)/ReBoot 側。F1 0.742、F2 0.397)。M2 の対話版 0.564 は (d) ではなく別の参考行。WN B=64: 0.233 / 0.414 / 0.404。WN B=32: 0.435 / 0.779 / 0.756。
- wall-clock(同一 CPU、サンプルあたり): WN B=64 0.447 倍、AU B=64 0.609、WN B=32 0.838、AU B=32 1.187。**優位はすべてブートストラップ由来**(14.6 s 対 31.0 s、回数も少ない)。非ブートストラップ部分は hekit が遅い(B=32 で 2.03 倍、B=64 で 1.09 倍。単一スレッド対 OpenMP)。
- 精度差(B=64 で −1.48 pt)は平文の双子でも同じ(−1.47)なのでバッチサイズの効果。暗号化誤差ではない。

### 4.2 そのほか

- Iris(2020 年再現、seal_2020): batch 1 の 1 反復 0.856 s(SEAL 版の 0.63 倍、論文の 10.8 分の 1)。batch 20 × 400 epoch: test 0.967(ゲート 0.95)、56 分(Mac)。サーバーでも 0.9667。
- SEAL 比(M1): mul+relin+rescale 1.20 倍、rotate 1.17 倍(未達、ゲートではない)。NTT 1.03 倍。KAT は SEAL 4.1.2 と bit-exact。
- GPU: NTT 2.608 µs/poly(基準 3.04)。演算は CPU 1 スレッドの 125〜202 倍、12 スレッドの 33〜132 倍(`m4-report.md` §1)。

## 5. サーバー(quadruped-gpu)の運用

- `ssh quadruped-gpu`(鍵認証。**パスワードは絶対に使わない・探さない**。本人がチャットに貼ったことがあり、ローテーションを勧めた)。Ubuntu 26.04、i9-13900KF(P8+E16)、123 GB、RTX 4090 24 GB、CUDA 13.1(`/usr/local/cuda-13.1/bin` を PATH に)、sudo 不可、ディスク残り約 26 GB。
- 他人のジョブ: SLAM(`persistent_ster*`、約 4 コア)と vLLM(pid 2791212、GPU 20.6 GB 占有、利用率 0%)。**触らない**。GPU の空きは約 2〜3 GB なので、フルサイズの GPU 鍵(288 MiB × 30 本超)やブートストラップは載らない。capped keys(level 9/5)で測る。
- 自分のジョブ: `nice -n 10`、24 スレッド以内、RAM 90 GB 以内、2 分超はバックグラウンド(`~/hekit-runs/run_job.sh`、ログ `~/hekit-runs/logs/`、出力 `~/hekit-runs/out/<job>/`)。前後に `df -h ~`、`nvidia-smi`、`uptime` を記録。
- 依頼の書式: `hekit/results/quadruped/requests.md` にブロック(id、目的、コマンド、作業ディレクトリ、所要時間、GPU メモリ、出力先、同期するコミット)を追記 → runner が順に実行し `status: done/skipped` と結果パスを書く。結果は `results/quadruped/`、ログは `docs/quadruped-log.md`。runner は commit しない。実装側が path 指定で commit する。
- 同期: rsync(gitignore を考慮した file list。`tools/estimate/_ext/sage-env` などの巨大な gitignore 物を送らない)。`~/hekit/.synced-commit` に記録。
- ReBoot のスタックは `~/reboot/`(OpenFHE v1.2.1、openfhe-python 0.8.9、ReBoot 3ac173f + build-only patch)。GPL なので `~/hekit` に入れない。ReBoot の設定値を書き出すときは**値だけ**(ソースの抜粋は commit しない)。
- 現在のキュー(chain10): R-41-FULL-1(07:17 開始、ETA 2026-11-07 02:55 JST)→ R-41-FULL-2(約 38.6 h)。R-47-3(Task 47 再計測 + GPU テスト全部、bd5cc49 以降)は依頼済みだが未投入。FULL の時間計測を乱さないよう FULL の後に投入する。R-REB-4(ReBoot の暗号パラメータ)は完了。
- セッションは 5 時間の利用上限で止まることがある。サーバーのジョブは nohup で走り続けるので、再開後にログと done マーカーを見て回収する。待機は「ジョブ完了で exit するローカルのバックグラウンド Bash」を 1 ジョブ 1 本置く。
- Mac(M3 Pro、36 GB)は 1 スレッドでサーバーの約 2 倍速い。**同一 CPU 比較はサーバーの行だけ**を使う。Mac の走行は KS と精度の参照用。

## 6. 残タスクと着手条件

| Task | 内容 | 状態 / 条件 |
|---|---|---|
| 42 | M3 報告の仕上げ | 草稿 09206bd。FULL-1/2 の 196 行を入れ、§0 の判定を確定。ReBoot のセキュリティ推定(実行中)を §3.3 に入れる。4 通りの比 + subset/full の明記 |
| 47 | CudaEvaluator の再計測 | R-47-3(bd5cc49)の結果を `m4-report.md` §1 に入れる。batch 行は「untuned」のまま |
| 48 | GPU 学習 + RQ3 ablation | 実装中(2026-11-06 着手)。HostRefresher の N1/N2(アドレス確認、upload 成功後に計数を合算)もこのタスクで入れる。逐次実行、epoch 境界で `trim_pool(0)`。GPU メモリ 2〜3 GB の制約下では capped keys の小さい構成で。入らない行は「未計測」と明記 |
| 49(stretch) | GPU ブートストラップ | コード準備完了・レビュー承認(`f64a617`)。CPU 回帰・再現テストと CUDA コンパイルを確認。GPU 精度・速度・メモリは未計測。目標 ≤ 220 ms。詳細は §10 と `gpu-bootstrap-design.md` |
| 50 | M4 報告 | 47/48 の数字 + HEonGPU 比較 + ablation 表 |
| 51 | データセット追加 | 実装済み(09f16f2、8bd9cba)、レビュー中。平文双子 eMLP-1、10 シード、epoch 50: MNIST 95.00(T4 94.61、合格)、Fashion-MNIST 85.22(T4 85.12、合格)、Breast Cancer は固定 epoch で −1.3〜−2.6 pt の未達(テストで選んだ最良 epoch のみ 99.30)。Breast Cancer は z-score だと z²+z で発散するので min-max(訓練データだけで決定) |
| 52 | eMLP-2/3 のブロック構造(`BlockMlp`) | M0 で emlp2/3 の JSON は忠実でないと記録。ReBoot のコードから値を取り直す(runner に依頼) |
| 53(stretch) | 4 次活性化 | 削減候補 |
| 54 | `scripts/run_all.sh` + `tools/tables.py` | Mac は `--quick`、本走行はサーバー。論文の表を JSON から生成 |
| 55 | v1.0 公開準備 | 名前は **ckks-train**(GitHub `kenmaro3/ckks-train`、private。本人決定 2026-11-06)。README の Status が M1 のままなので更新。`scripts/rename.sh` は dry-run のみ。THIRD_PARTY_NOTICES に SEAL(MIT)、Lattigo(アルゴリズムの謝辞、コードは未参照)、libloading(ISC)、Iris/WDBC/MNIST のデータ出典 |
| 56 | 論文ドラフト(acmart、WAHC 2027) | 著者欄は仮置き(本人未決)。R-M3-15 の精度の書き方、4 通りの比、「速さはブートストラップ由来」の正直な分解、ReBoot の論文値の再現性(R-REB-3)を入れる |
| 57 | M5 報告 + arXiv チェックリスト | |
| 58(最後) | PyO3 バインディング | Python toolchain が無ければ省略して明記 |

削減の順序(計画どおり): Metal → Task 49 → 53 → 51 の追加データセット → 58。

## 7. 本人の未決事項(作業は止めない。M3 報告 §9 と同じ番号)

11. **B ≥ 64 を論文の主設定にするか**。KS と wall-clock を 3 通りすべてで満たすのは B=64 だけだが、平文でも B=48 より 1.47 pt 低い。計画担当の推奨: B=64 を主、B=32 と B=48-in-64(精度行)を併記。FULL の結果で確定。
12. **秘密鍵分布の揃え方**(hekit 密三値 + SSE、ReBoot SPARSE_TERNARY h=192)。ReBoot 側の同モデル推定が 128 bit 以上なら「両側 128 bit 以上」で閉じる。
13. **wall-clock の比べ方**(12 スレッド、ピン留めなし、共有負荷あり)。ReBoot 公式のフル走行(約 47.6 h)を回すか。
14. **「同一 N」の解釈**(実測はすべて 2^16。推奨は 2^16 固定)。
15. **重み誤差の蓄積の許容基準**(非対話版は 1 epoch で 2^-13〜2^-15)。FULL で 10 epoch の値を見る。
17. **ライブラリの名前**(決定済み): **決定(2026-11-06): ckks-train**。GitHub は rename 済み。crate 名(`hekit-*` → `ckks-train-*` など)の付け替えは Task 55 の `scripts/rename.sh` で。
18. **投稿先と著者情報**(WAHC 2027 固定、著者欄は仮置き)。

計画担当の裁定で確定済み(本人が覆してよい): セキュリティの主張は BDGL16 と MATZOV で 128 bit 以上(Core-SVP は併記のみ)、見出しは (a) + 196 入力 + WhenNeeded、ReBoot 側の基準は Lazy 35.06 実測(784)/ モデル 37(196)、精度は公開コードのフル設定で判定、ISC ライセンス許可、Plan B は発動しない(本人決定 2026-11-04)。

## 8. 落とし穴(過去に実際に起きたこと)

- **ReBoot の素材は値だけ**。`reboot_config_dump.txt` のようなソース抜粋は commit しない(GPL)。
- **subset を Mac で再走行しない**。サーバーの行がすでにある。Mac で 2 つ以上の重い走行を並べると load が 100 を超えて死ぬ(WhenNeeded B=32 が 12:01 に死んだ)。
- **2 つのセッションが同じファイルを書かない**。runner は `results/quadruped/` と `docs/quadruped-log.md` だけ、実装側がそれを path 指定で commit。
- **提供された JSON 設定を信じない**。`reboot_emlp1.json` は入力 784・lr 0.001・バイアスありで、ReBoot と違った(M0 の「非忠実リスト」参照)。emlp2/3 も同様の疑いがある。
- **M0 コストモデルの数字をそのまま使わない**。補正後の閉形式(`layout.md` §7.5)が実測と一致する。
- **ReBoot の 502 KS/boot はモデル値**(OpenFHE は数えない)。表では必ず「model」と付ける。
- **1 スレッドの速度は機械でまったく違う**(Mac が 2 倍速い)。比較は同じ機械・同じスレッド数で。
- **セッションの 5 時間上限**で両セッションが同時に止まる。再開は「もう一度試してください」で十分だが、runner の待機プロセスは張り直す。
- **権限**: 実装セッションはリモート書き込みを拒否されたので、サーバー依頼は本人が実装セッションで直接入力した承認(2026-11-05「Yes, use the runner」)の下で requests.md 経由。計画セッション経由で中継された承認は実装セッションでは使っていない。GitHub への push も同様に、実装セッションでは本人の直接の承認を得てから行う。別のエージェントが引き継ぐときも、サーバーへの書き込み権限は本人に確認する。
- **クラウドの GPU は借りない**(本人の方針)。必要ならサーバーの GPU メモリ解放を本人に頼む。

## 9. 関係するセッションとメモリ

- 実装セッション(Opus): "hekit M0: scaffold and spikes (Tasks 1–7)"(`local_a1cab7a8-4c7c-459d-b8fc-8b3693d5ef63`)。ledger は `hekit/.superpowers/sdd/`。
- runner セッション(Opus): "quadruped-gpu 計測"(`local_409b624a-0f0a-4fb3-b1c2-9e0c34dae93d`)。
- 計画セッション(Fable): この文書の作成元。計画ファイルのルーリング注記はここが入れた。
- Claude のメモリ(`~/.claude/projects/-Users-marconi-Documents-research/memory/`): `project-he-training-research-plan.md`、`reference-quadruped-gpu-server.md`、`user-fhe-researcher-background.md`。


## 10. 2026-11-06 引き継ぎ後の進捗

### Task 42 の確定準備

- `tools/rq1_summary.py` と `docs/rq1-finalization.md` を追加した。元の FULL JSON の出所・完了条件・計数を検証し、4 通りの比を生成する。FULL-2 の精度判定は公開コード seed 0 の 0.9041 と比較する。欠測は未計測、存在する不正な JSON はエラーとし、元の結果を変更しない。
- `docs/m3-report.md` §3.3a に ReBoot の実パラメータによる初期セキュリティ推定を反映した(`9ee79c4`)。初期値は BDGL16 122.3 / MATZOV 121.9 bit。追加探索と Task 39 追補レビューの後の最終値は MITM ありで 121.8 / 121.5 bit、MITM なしで 127.8 / 126.2 bit で、128 bit 条件は満たしていない(`docs/security.md`)。元ログは `results/security/reboot-initial-20261106/`。同じ N でも同じセキュリティとは主張しない。
- FULL-1 / FULL-2 の完了 JSON と R-47-3 はまだ回収していない。Task 42 は未確定。SSH の観測タイムアウトはジョブ終了・失敗の証拠にしない。

### Task 49 のコード準備

`f64a617` (`feat(cuda): GPU bootstrapping`) は独立レビュー承認後にコミットした。`CudaBootstrapper` は SSE、mod-raise、CtS、EvalMod、StC を GPU の暗号文で実行し、既存の `BootstrapOps` / `BootstrapRefresher` に接続する。CPU との多項式評価の共用について、理由と代替案は実装前に `docs/gpu-bootstrap-design.md` に記録した。新しい外部依存は追加していない。

確認済みの範囲:

- CKKS の CPU 回帰テスト 158 件、CUDA クレートのローカルテスト 61 件が合格。GPU が必要な 47 件は未実行。
- CPU の小さいパラメータで全経路のビット・計数一致を確認。本番サイズ N=2^16 / 32768 スロットの CPU 精度テストは 17.46 bit、残り16レベル、111 KS = 回転69 + relin40 + SSE2。これは GPU の精度測定ではない。
- CUDA テストのコンパイル、CUDA クレートの全ターゲットの clippy、対象ファイルの整形検査が合格。最終の workspace 全体の整形検査は、別担当が編集中のファイルで差分が出たため、この節目で全体合格とは記録しない。

**未計測:** GPU 上の精度・CPU とのビット一致、実機でのメモリ不足時の挙動、GPU メモリのピーク、レイテンシと 220 ms 目標。現状の空き GPU メモリでは実行しない。公開整数定数をホストで展開して転送する費用も将来の GPU 時間に含める。

レビューの軽微事項は ledger に残した。特に `gpu_bootstrap` のテストはメモリ不足時に `SKIP UNMEASURED` を出して正常終了するため、**Cargo の終了コードだけを GPU ゲートの合格証拠にしない**。実機の採用判定前に、実行完了を示す明示的なマーカーを確認する仕組み、または資源不足時に要求した実機テストを失敗させる仕組みを追加する。共通 EvalMod の比較テストはラッパーの整合性確認であり、独立した数値検証は既存の精度テストと上記 CPU 全スロットのゲートが担う。

### 並行作業の担当境界

本人の回答により、Task 48 は既存担当の完了後に引き継ぐ。provenance が欠けていた ablation JSON は既存担当が `0e07561` で修正し、再レビュー中。Task 52 も既存担当が実装中。この段落は引き継ぎ開始時の記録で、現在は両タスクともレビュー済み（§11〜§12）。GPU 計測の順は FULL-1 → FULL-2 → R-47-3 → R-48-1〜4 を維持する。

v1.0.0 のリリース準備(Task 55、名前 ckks-train、公開・改名は本人の判断): `docs/release.md`。

## 11. 旧実装セッションの停止記録(2026-11-06)

以下は旧実装セッションが把握していた範囲の記録である。並行して進めた Codex の Task 49・50・56・57 の状態は §10 と末尾の §12 を優先する。本人の指示(計画セッション経由、2026-11-06)で、この実装セッションはここで止めた。以後は Codex がこのチェックアウトで作業する。ledger は `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/progress.md`(gitignore 下、各タスクの dispatch / review / ruling の記録)。

### 11.1 旧セッションの実装状況（個別レビュー状態は表を参照）

| Task | コミット | 要点 |
|---|---|---|
| 35〜38 | 505b597, f40a3a3, 85fecfd, 67f5ec2 ほか | n16_boot、SSE、EvalMod 7 レベル、17.46 bit、111 KS/boot |
| 39 | 6c72082, ee4aa27, 0330f20, 8420a96 | hekit (a) 131.5/129.3、(b) 133.8/134.2。ReBoot 実パラメータは非 MITM 127.8/126.2、MITM 121.8/121.5(BDGL16/MATZOV、MITM はメモリ 2^104〜110) |
| 40 | 58b293b | compact 再配置 |
| 41 | cc007a8 … 9d132c1, ff3042d | 非対話 Trainer、融合経路、忠実設定、196 入力、per-step lr(基準 lr で速度保持) |
| 43 | 340f487, 37d8d6f | HeOps、EvalConfig、add_counts |
| 44〜47 | 8b39a57, a6ab1e2, 8b81f3d, f68c05f, b4b9dc3, feff7ed, 37e80a9, bd5cc49, 940c191 | CUDA。RTX 4090 で bit-exact(R-45-1 18/18、R-46-1 31/31、R-47-1 45/45 中 42 実行) |
| 48 | 626160e, 0e07561, b1d1813 | GPU 学習パス(level 7 capped keys)、RQ3 ablation runner(provenance 付き) |
| 51 | 09f16f2, 8bd9cba | MNIST/F-MNIST は ±1 pt 内、Breast Cancer は固定 epoch で未達(min-max 採用) |
| 52 | b8e267c, 91efa6e, 9305d2e | BlockMlp(局所ブロック + 出力スタック)、双子は R-REB-5 と 1000/1000 一致 |
| 55 | 91219e9, 49126d4 | v1.0 準備(publish dry-run は core のみ通過、他は未公開依存のため。rename.sh dry-run 822 行 / 155 ファイル、未適用)。**レビューは停止時に中断(未完)** |

### 11.2 途中で止めたもの

- **Task 54**(run_all.sh、`tools/paper_tables.py`): 実装 962b7d2 / 1134a9b / a502429 はコミット済み。レビューで Important 3 件: (I1) missing-input だけ残ると exit 0、(I2) (d) 列に対話版 0.564 を入れている → 裁定済みの定義(同じ非対話実行からブートストラップ分を除く。§4.1 参照)に直す、(I3) FULL 経路のテストが無い。**修正ラウンドは途中で止め、未完の編集はコミット版に戻した**。
- **Task 55** のレビュー: 途中で停止(結果なし)。
- **Task 42**(M3 報告): 草稿 09206bd + 追補(ba54f9d で (d) 列)。FULL-1/2 の 196 行を入れて確定する必要あり。`docs/rq1-finalization.md` は Codex の提案文書。計画担当の方針: 計画と m3-report が優先、生き残る内容は Task 42 確定時に m3-report へ統合し、その旨を rq1-finalization.md の冒頭に書く。
- **Task 49**: 旧セッションでは延期としたが、Codex がコードを実装・レビュー済み(`f64a617`、§10)。GPU 実機測定だけが未実施。**Task 53**(4 次活性化)は保留。
- **Task 50、56、57**: 旧セッションでは未着手。Codex の進捗は §12。**Task 58** は未着手。

### 11.3 サーバーのキュー(`results/quadruped/requests.md`、gpu-runner が実行)

R-41-FULL-1(07:17 開始、ETA 2026-11-07 02:55 JST)→ R-41-FULL-2(約 38.6 h)→ R-47-3(bd5cc49 以降)→ R-48-1..4(0e07561、bash、失敗規則付き)→ R-52-1/2(9305d2e、合格規則付き)→ R-54-1(962b7d2 以降、`--quick`)。完了したもの: R-REB-1〜5、R-41-1〜5、F1/F2、R-45-1、R-46-1、R-47-1/2、R-REB-4。

### 11.4 未決事項(本人)

§7 の各項目に加えて: ReBoot が同じ推定器で 128 bit 未満であることの論文での扱い(表として示すか、等セキュリティ比較を併記するか、書き方)、eMLP-2/3 の全長暗号化走行を回すか(eMLP-2: WN B=64 約 51 h、B=48 約 82 h。eMLP-3: 約 98 h / 152 h、±30%)、LICENSE の著作権者名("hekit contributors" のまま)。

### 11.5 作業ツリーの注意(停止時)

- `results/mnist_emlp1_plain_faithful_b{32,48,64}.json` は作業ツリーで上書きされている(09:13、`--plan --mode plain` の実行、この実装セッションではない)。Task 41 が引用するのは**コミット版**(f195124 の provenance 付き)。
- 未追跡・未コミットの `docs/rq1-finalization.md` の変更、`docs/m5-report.md`、`paper/`、`tools/figs.py`、`tools/test_figs.py`、`results/mnist_blocks_*_plain_b48.json`、`.vscode/` は Codex のもので、この実装セッションは触っていない。

## 12. Codex 統合状況（2026-11-06、継続中）

- 旧実装セッションの停止を `b4712f5` で確認。以後の残件は Codex が担当する。Task 48/51/52 の実装・修正レビューは完了しており、実機測定の待ちは残る。
- Task 42 の ReBoot 推定追補を `0daf778` で統合し、元 grid ログ3本と capture 記録を不変のまま保存、独立レビュー後に push 済み。M3 は依然 FULL-1/2 待ち。比 (d) は `ba54f9d` の裁定を採用し、同じ NI 実行から bootstrap を引く。M2 の対話行は別の参考行。
- Task 50 の報告準備は `5acc442`、Task 57 の現状報告は `fd49eac`。どちらも独立レビュー済み。報告が存在することと、測定・研究ゲートが完了したことは別である。
- Task 54 の旧レビュー指摘（欠落入力だけで exit 0、対話比を (d) に混入、FULL のテスト不足）を引き継いで修正中。加えて `run_all.sh --tables-only`、新規出力の provenance、対応範囲の明記を整備する。`tools/paper_tables.py` を本体に維持し、`tools/tables.py` は互換入口とする予定。旧 Codex prototype は ledger 下 `task54-frozen-prototype/` に全バイトを保存して退避した。
- Task 56 は全7節・付録と自作図を作成中。匿名版の初回 build は9ページで通過したが、表の修正反映・最終レイアウト確認・独立レビュー前。著者欄は仮置き。TeX 配布物は ignored `tools/latex/_ext/` に置き、配布物に含めない。
- Task 55 の旧レビューは中断されていたため、公開準備を再監査中。実 checkout の crate rename はまだ未適用。名称 ckks-train は本人が決定済みで、改名の再承認は不要。編集中の Rust・表・論文を確定した後に適用し、テストとレビューを行う。
- 依存監査で `unicode-ident 1.0.27` の `(MIT OR Apache-2.0) AND Unicode-3.0` を確認。本人が 2026-11-06 に Unicode 文字属性表だけの例外を承認した。`deny.toml` は一般許可を MIT / Apache-2.0 / BSD / ISC に戻し、`unicode-ident =1.0.27` だけに例外を限定。旧設定の Unicode-3.0 / Zlib の一般許可は削除した。Unicode の通知を `THIRD_PARTY_NOTICES` に追加し、スコープ付きの licenses / bans 検査が合格した（未使用 BSD-3-Clause 許可の警告のみ）。
- `results/mnist_emlp1_plain_faithful_b{32,48,64}.json` の誤上書きは、上書き後の全バイトと SHA-256 を ledger 下 `preserved-plan-side-effects-20261106/` に保存してから、引用元の HEAD のバイト列（writer commit `f195124`）へ復旧した。数値や provenance を手修正していない。`--plan --mode plain` を実行前に拒否する修正と新規出力 provenance を `d611c9d` で実装・独立レビュー・commit 済み。未追跡の `results/mnist_blocks_*_plain_b48.json` と `.vscode/` は変更していない。
- 最新の鍵認証のみの SSH 観測は接続タイムアウト。ジョブの終了・失敗とは判定せず、再起動も追加ジョブ投入もしていない。登録済みの逐次実行を維持する。

### 12.1 サーバー担当も停止済み（0c819d4 の回収手順を優先）

`docs/quadruped-log.md` 冒頭に旧 runner の 09:27 JST 観測と回収手順がある。chain10（当時 PID 1700979）で FULL-1 → FULL-2 のみ投入済み。FULL-1 は当時 80/750 更新。R-47-3、R-48-1〜4、R-52-1/2、R-54-1 は **依頼書のみ・サーバー未投入**。§11.3 の矢印は希望する順序であり、後続キューが存在する意味ではない。FULL 完了後に Codex が登録済みブロックを明示的に投入する必要がある。

SSH の接続に約47秒かかる記録がある。5秒の connect timeout は到達不能を判断するには短かった。鍵認証のみで待ち時間を適切に設定し、長い観測はバックグラウンドで待つ。実行中の clone を変更しない。ログは `~/hekit-runs/logs/`、runner provenance は `~/hekit-runs/out/`、DONE は `post.txt` 最終行の `JOB_DONE rc=...`。元の学習 JSON は `~/hekit-r41full/results/quadruped/` にある。

Codex の 55 秒・90 秒の接続待ち、その後の60秒の読み取り専用再接続も SSH はタイムアウトした。ローカルの経路と alias の宛先は確認したが、認証成功・サーバー状態の再確認には至っていない。最後に確認された FULL の状態は旧 runner の 09:27 観測である。追加キューの投入や実行中ジョブの変更は行っていない。


### 12.2 論文・再現入口の確定と crate 改名（2026-11-06）

- `e2d27b6`: Task56 の論文ドラフトを独立再レビュー後に commit。匿名・著者仮置きとも最終表込み9ページ。寸法の2冪 padding と先行する矩形転置への帰属を明記した。FULL/GPU 欠測と安全性の比較限界を残す。
- `7752459`: Task54 の表生成・runner 修正を独立再レビュー後に commit。49 Python tests が合格。古いバイナリの誤採用、Cargo設定やスレッド数変更の見落とし、メタデータ書込失敗の握り潰しを防ぐ。`--tables-only` と `tools/tables.py` で保存済み結果から表を再生成できる。
- `c820638`: Python 回帰テストと表・図の freshness を CI に追加。`0daf778` の実 Linux CI 全5ジョブ成功は、後続 SHA の成功とは分ける。
- `scripts/rename.sh` を実 checkout に適用し、5 crate を `ckks-train-*` に変更した。903箇所・847行・158ファイルとディレクトリ移動、続いて rustfmt の整形。結果・spikes・kat・サーバーログの755ファイルは改名前後で同一バイト。改名後462 Rust tests合格（12 ignored）、49 Python tests合格、CUDA compile・Clippy両feature・rustdoc警告0・deny・4パッケージ連鎖ビルドが通過。最終 SHA の Linux CI は別途確認する。変更は独立レビュー後に commit する。
- Task58（PyO3）は計画の削減枠として今回は削減する。Rust API と再現手順、未完の研究ゲートを優先し、binding 提供済みとはしない。理由と後日の代替案は M5 報告に記載した。

改名時の例外: CUDA kernel のコメント置換がソース固定ハッシュ検査に掛かったため、4 kernel は旧版と同一バイトを保持した。`tools/reboot_layout/assemble.py` も過去コミットの provenance を記述するため同一バイトを保持し、両者を rename.sh の対象外にした。ハッシュの付け替えや過去の結果の書換えは行っていない。


### 12.3 post-full 待機コントローラ投入と CI 更新（2026-11-06 10:49:44 JST）

2026-11-06 10:49:44 JST の起動後観測で、post-full コントローラは **waiting**、PID `1739838` / start ticks `467041669`、state ディレクトリは `owner.json` のみ、`clones_created=false` だった。これは待機コントローラの投入であり、R-47 / R-48 / R-52 のビルド・学習・測定開始ではない。R-54 は投入していない。

FULL の最後の実進捗観測は **2026-11-06 10:21:08 JST、FULL-1 update 113（0 始まり）、114/750 更新完了**、elapsed 10956 s。10:30 の gate identity と 10:49 の待機起動観測から後続更新数を推定しない。FULL-1/2 の完了・研究合格は未確認である。

10:30 のidentity照合で chain10 PID1700979 / start ticks465769819、FULL clone `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f` を確認した。レビュー済み queue source `4b1c499` はprivate mainへpush済みで、root controllerが4ファイルのSHA256を照合して待機起動した。R47 `bd5cc49` → R48 `0e07561` → R52 `9305d2e`（各21更新subset）を両FULL成功+chain terminal/非稼働ゲート後にだけ新規cloneで逐次実行する。既存FULL cloneには触らない。実PID・完全commit・config/script/bundle/source-record SHA256と回復手順は [post-full-queue.md](post-full-queue.md) を参照する。§11.3のR54までの矢印は投入済みキューを意味しない。

R54は未投入のまま。Task54の縮小版入口・表生成はレビュー済みだが、研究計画の全実験行列の実装は未完で、拡張設計中。eMLP-2/3全長も投入していない。待機コントローラの存在をFULL完了、GPU実測、研究合格と呼ばない。

改名commit `09202cf5650aa76b9b30dd79a68dd615c75a91ae` の [Linux CI run42143028484](https://github.com/kenmaro3/ckks-train/actions/runs/42143028484) は fmt/clippy/cuda-check/test/deny/research-tools **全6ジョブsuccess**（ledger `task-55-rename-ci.json`）。この成功は当該SHAのみ。後続queue commit `4b1c499` の [run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) はこの記録時点でpendingであり、成功を継承させない。ローカル公開準備・private push・CI・待機投入は、公開release/crates.io/arXivやGPU研究測定とは別である。


### 12.4 FULL進捗と待機継続、queue CI確定（2026-11-06 11:09:14 JST観測）

2026-11-06 **11:09:14 JST**（Unix `1793930954.5309272`）の鍵認証による読み取り観測で、FULL-1 は **update144（0始まり）=145/750更新完了**、elapsed13851s。観測した直近3行（updates142〜144）はすべて `matches prediction true`、post終端はなく、partialマーカーもない。FULL-2には進捗行・post終端がない。FULL-1/2の完了・研究ゲート合格は未確認である。

chain10 PID1700979/start ticks465769819、待機controller PID1739838/start ticks467041669は同じidentityのまま。stateはowner.jsonのみ、clone未作成。R47/R48/R52は引き続き両FULL成功とchain terminal/非稼働identityの検証待ちで、追加のbuild/train/measurementは未開始。R54とeMLP-2/3全長は未投入。実行中cloneや他テナントのプロセスは変更していない。

その後rootが確認した queue commit `4b1c49943533bb9ff79493daee33a27b8237a18c` の [CI run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) は **completed/success、fmt/test/deny/research-tools/cuda-check/clippyの全6ジョブsuccess**。10:49時点のpending記録を更新する。これは当該SHAのLinux CIであり、後続SHA、GPU実測、FULL完了、研究合格を検証したものではない。

一次記録は ledger `post-full-followup-observation.stdout`（SHA256 `eeae57cf21919e2011c61baf26bdccaf4486e502a325e1bdf15d5a58f277edb9`）と `task-post-full-ci.json`（SHA256 `fd1c7d740e6fa64461f0185640a5802b3149af2b7a8f600f01b56b691e5e1a09`）。CI run/SHAの対応はrootの管理観測による。元の研究JSONは変更していない。


## Task54 全実験範囲への接続と最新観測（2026-11-06 12:01:40 JST）

既存の `run_all.sh` は保存済み表を再生成できるが、原計画 Task54 の全データセット・モデル・CPU/GPU・I/NI・baseline(a/b/c)・10 seedsを実行できる段階ではない。部分対応を完了扱いにしないため、`8eabcdd` で [全実験行の登録](experiment-matrix.md) と機械可読 registry を独立レビュー後に確定した。21 profiles / 17 familiesを登録し、欠落入力・未実装・メモリ不足・保留も必要な行として残す。B128の境界更新は実際の4864/4736サンプル区間で集計し、4800サンプルへ按分しない。新しいFP32基準と目標値の固定はこれからであり、既存FULLの評価基準を変更しない。

接続の基礎として、`0b1298d` は class-aware データ読込・固定分割・訓練集合だけで求める前処理・宣言済みseed入力検証を共通化した。関連33テスト合格、外部データ2件ignored。`1a2d23e` はブロック学習のAA handoffで既存の正規化済みactivation bootstrap出力を再利用し、二重bootstrapを除いた。NN108テスト合格、重い4件ignored。いずれも独立レビュー後のcommitで、実データ全実験やGPU実測の完了を意味しない。実際の設定・入力への接続、暗号化学習入口、packed baseline(c)、FP32・統計・全行runnerは引き続き実装中。

`8eabcdd4cd46b3144c5a222ddb0ce533ee627b49` の [CI run42150264921](https://github.com/kenmaro3/ckks-train/actions/runs/42150264921) は6ジョブすべて成功。これは当該SHAのCIであり、後続実装やGPU計測の検証ではない。

12:01:40 JST（Unix1793934100.240689）の鍵認証による読み取りでは、FULL-1はupdate177（0始まり）、**178/750更新完了**、elapsed17000s。直近3更新175〜177は予測演算数と一致した。FULL-1のpost終端・partialはなく、FULL-2の進捗・postもまだない。chain10 PID1700979/start465769819と待機queue PID1739838/start467041669は同一identityのまま、stateはowner.jsonのみ、後続cloneは未作成。R47/R48/R52は待機中、R54は未投入。ジョブや実行中cloneの変更はしていない。これは11:09の観測より新しい稼働記録であり、FULL完了や研究ゲート合格ではない。

一次記録はledger `post-full-observation-20261106-1200.stdout`（SHA256 `eb747177089ddae13281bfcbdb782222958afcabb0559dcde89d3739323d4a4f`）と `task-54-d0-ci.json`。研究結果JSONは変更していない。Task42の四つの比とFULL-2対0.9041の±0.5pt判定、M4の実機ゲートは未確定のまま残る。

## Task54 基礎コードの追加確定と観測更新（2026-11-06）

この節の最新のコード統合点は private main に push 済みの [`9610822`](https://github.com/kenmaro3/ckks-train/commit/9610822f0ccbf254138b5ec1aef049240583ce0f)。次の限定範囲は実装・ローカル検証・独立レビュー済みである。

- [profile 入力・更新計画](study-profile-inputs.md): [`8f3a968`](https://github.com/kenmaro3/ckks-train/commit/8f3a9686314ce09397eb8d0a1223bd9bb40cbc61)。[不変の共通入力](study-training-design.md#10-d6a-local-implementation-evidence-independently-reviewed-after-fix1): [`da24811`](https://github.com/kenmaro3/ckks-train/commit/da248115e20c79128dafb32e1732d74972799457)。D6a は62 tests合格・外部データ依存2件ignored。Normal初期化の演算順を修正し、legacy Iris の83パラメータの全ビットと400 epoch順序を厳密照合した。
- [重み付き I microbatch](interactive-microbatch-design.md): [`9a04fcc`](https://github.com/kenmaro3/ckks-train/commit/9a04fcccc37cfb93821639b7256403bafb57fa55)。[I block 合成・全stageのrate設定とNI reference設定の原子性](block-profile-design.md): [`cf9241e`](https://github.com/kenmaro3/ckks-train/commit/cf9241e6e410e93670ec633290856bac3e38b735)。各stageの局所lossを保持する。class-aware本体のCPU/GPU実行入口への接続完了を意味しない。
- [packed設計](packed-training-design.md): geometry/primitives [`ef39ea2`](https://github.com/kenmaro3/ckks-train/commit/ef39ea220477026c15152ecb6eeb4d9b87b8b888)、SGD/NAG optimizer [`71de370`](https://github.com/kenmaro3/ckks-train/commit/71de3709986e1cbe511d425792539d0aea5a3921)、暗号化2層stage `0388282`。25 focused testsと共有作業ツリーのNN回帰157件が合格、重い4件ignored。暗号化数値検証は非安全な小規模CPU fixture・許容差1e-7であり、本番精度・性能の証拠ではない。

- [実際のF32/F64参照計算](study-training-design.md#11-d6b-concrete-rationale-and-api-bounded-implementation): [`b897f69`](https://github.com/kenmaro3/ckks-train/commit/b897f696eb97f9fc093d3d8b95abe1266d9c7d37)。演算そのものを各dtypeで行い、各stageの局所loss・旧hidden・固定referenceのNAGを保持する。11 unit testsと全4 Iris graphの2更新×2dtype検証、bench回帰119件が合格、外部データ2件ignored。全長seed0の基準結果やtarget lockを作成した証拠ではない。
- [純粋なscale/scalar metadata](ckks-metadata-design.md): [`1ec12c8`](https://github.com/kenmaro3/ckks-train/commit/1ec12c8a49f51ef7241093e411b38617d288e854)、14 focused tests合格。新APIの有限値領域と、既存runtimeが許す非有限の特殊ケースを明確に区別し、既存の端点挙動は変更していない。
- [packed client refreshと状態projection](packed-refresh-design.md#9-c1-concrete-api-2026-11-06-recorded-before-implementation): [`9ff2504`](https://github.com/kenmaro3/ckks-train/commit/9ff25045f8be4aa793da6cb49a25d0300157496c)。W/U/bias・更新indexを保ち、全応答検証後に一括で状態を置き換える。23 tests・strict Clippy・formatを独立レビューでも再実行して合格。小規模CPUでのprotocol/学習一致の検証であり、NI bootstrapやGPU計測ではない。

- [共有条件とseed別入力のidentity](study-reference-design.md): [`6aba8f4`](https://github.com/kenmaro3/ckks-train/commit/6aba8f4f4a986946e69293697e179b4873cfcdaa)。全profile共通の条件と、各seedの全順序・初期値・rate・更新計画を別の不変receiptに保存する。独立レビューでも74 tests合格・既存の外部データ2件ignored、Clippy・format合格。全長参照結果・target lock・保存形式はまだ実装していない。
- [pre-key bootstrap metadata](bootstrap-metadata-design.md): [`090f8fb`](https://github.com/kenmaro3/ckks-train/commit/090f8fb46aef908e658c7608862a85695c72b17a)。鍵を作る前に構造上の演算数・必要な鍵・境界scaleを導出する。既定設定の計数 `(69,40,46,343,56,2,1)`、111 key switches、出力level16/Delta、32 rotation keysを確認した。これは構造の検証であり性能実測ではない。修正後の独立再レビューでMacの9 integration +12 metadata tests・Clippy・formatが合格した。Linuxではproof未対応を明示して拒否する。encoding・EvalMod内部scale・range・memory・securityを含む全面的admissionは未達である。検証filterが意図せずN=2^16のEvalMod鍵生成テストを含んだ事実と、元の「未実行」記述の訂正は同設計書のF1/F2節に残した。元ログは保持し、以後の対象filterから明示的に除外する。

[事前admission](study-diagonal-admission-design.md)と[参照結果・profile別target lock](study-reference-design.md)は設計確定済みだが実装は未完。[参照結果の保存形式](study-reference-document-design.md)も `2ff6e06` で独立レビュー後に確定し、parserを実装中である。

[packed NIの小規模精度検証](packed-ni-validation-plan.md)は事前登録した21固定入力×2鍵×2回を、レビュー済みプログラム `9610822` で初回実行した。2026-11-06 22:14:36–22:15:05 JSTのMac実行で **84/84回が合格**、終了コード0、ソース・実行ファイルの前後SHA256は一致した。全512 complex slotsを検査し、1回のbootstrap誤差の最大値は約5.89e-8（各回の閾値2^-20）、2回後の累積誤差最大値は約1.13e-7（各入力の初期誤差+2×2^-20が閾値）。N1024・full chain・出力level16の固定corpusについての経験的検証であり、本番N65536、主学習の各event/最終snapshot、GPU、securityの合格ではない。各回79 key switchesで、本番metadataの111とは区別する。元のprogram stdout/stderrと別記のrunner provenanceを[保存した](../results/calibration/packed-ni-n1024-20261106/)。研究JSONの手書き・再試行・閾値変更は行っていない。

**原計画Task54の全実験行列は未完。** class-aware本体のstatic admissionとCPU/GPU bridge、全長F32参照artifact・target lock、packed NI/AA/block/GPU、baseline(a/b) adapter、全行runner・metrics・統計が残る。共有条件・seed別identityは上記の限定範囲で完成したが、参照session・保存・provenance・検証・target固定への接続は残る。全長暗号化blockは投入していない。

サーバーの最新の保存済み観測は **2026-11-06 21:45:13.892846 JST**（Unix `1793969113.8928456`）。FULL-1はupdate542（0始まり）= **543/750**、elapsed51932s、直近3更新はすべて予測計数と一致。FULL-1のpost終端はなく、FULL-2の進捗・postもない。同じ観測で両FULLの`partial.txt`不存在を確認した。chain PID1700979/ticks465769819、queue PID1739838/ticks467041669は同一identityでstate S、stateディレクトリはowner.jsonのみ・後続clone未作成。R47/R48/R52は両FULL成功とchain終端/非稼働の検証待ち、R54は未投入。これは現在時刻の状態や完了予測ではない。一次記録はledger `post-full-observation-20261106-2142.stdout` と対応meta/stderr。ジョブ変更・新しいFULL JSONはなく、Task42の四つの比・FULL-2対0.9041の±0.5pt判定とM4ゲートは未確定のまま。

CIはSHAごとに区別する。`1b340d2` のshallow checkout失敗を保存し、`3b5022c` はresearch-toolsだけfetch-depthを0に修正した（過去pin・source・入力は不変）。その [run42160446178](https://github.com/kenmaro3/ckks-train/actions/runs/42160446178) は6ジョブすべてsuccessになった。後続の `0388282` [run42162930752](https://github.com/kenmaro3/ckks-train/actions/runs/42162930752)、`cf9241e` [run42162578386](https://github.com/kenmaro3/ckks-train/actions/runs/42162578386)、`b897f69` [run42164161300](https://github.com/kenmaro3/ckks-train/actions/runs/42164161300)、`567cb89` [run42164016832](https://github.com/kenmaro3/ckks-train/actions/runs/42164016832)もcompleted/success。21:43:01 JSTまでの一覧観測では `46d616b` / `1ec12c8` / `e018bd7` / `9ff2504` / `db4c5fe` / `6ad2679` / `6aba8f4` / `493239d` もcompleted/successを確認した。`6aba8f4` は [run42168750096](https://github.com/kenmaro3/ckks-train/actions/runs/42168750096)、`493239d` は [run42170007123](https://github.com/kenmaro3/ckks-train/actions/runs/42170007123)。新しい `090f8fb` の [run42221820811](https://github.com/kenmaro3/ckks-train/actions/runs/42221820811) は22:01:02 JST観測時点でin_progressであり、成功を継承させない。一次記録はledger `task-54-ci-history-observation-1338.json`、`task-54-ci-observation-1448.json`、`task-54-ci-observation-2144.json`、`task-54-ci-observation-p1b.json`。一覧でのrun成功と実機研究ゲートは別である。 22:19:08 JSTの追加観測でも `090f8fb`、`02a26a5`、`9610822` はin_progressだった。calibrationプログラムの [run42224207703](https://github.com/kenmaro3/ckks-train/actions/runs/42224207703) のCI合格はまだ主張しない（ledger `task-54-ci-observation-calibration.json`）。

Mac運用メモ: 本人の依頼により、再生成可能なignored build artifacts、download済みpackage archive394件、改名前の`hekit_*`/`libhekit_*`生成物、未使用のrelease/package/doc生成物を削除した。累計11.391GiB相当の割当領域、全削除pathはledger `mac-cleanup-20261106.json`。現在使うdebug生成物、dataset・元結果・ledger・PDF・source・外部baseline環境・installed Sage環境と展開済みpackageは保持した。削除量は空き容量の差ではなく、21:44:42 JSTの`df`は28GiB free（ledger `mac-disk-observation-2148.json`）。この観測後は追加削除していない。ローカル `CARGO_INCREMENTAL=0` はcache増加の制御であり、Rust profileや研究上の最適化条件の変更ではない。

<a id="latest-evidence-20261106-2321"></a>

## Task54 parser・Linux構造検証の確定とFULL観測更新（2026-11-06）

コード統合点は private main にpush済みの [`8d3a649`](https://github.com/kenmaro3/ckks-train/commit/8d3a649db4b84e35f24eb598429782278b4c248c)。[参照結果のparser](study-reference-document-design.md)を実装し、独立レビューと指摘1件の修正・再レビューを完了した。不正なフィールド名が大きな文字列の例外扱いを得ないよう、object keyとarray要素の構造を診断表示から分離した。修正後はparser20件、正規パス2件、既存canonical golden1件、再現入力5件の拒否、Clippy・formatを確認した。初回の全21 profiles×10 seeds・既存入力ビット一致等の検証は元の証拠として保持し、修正後に全件を再実行したとは扱わない。読み込んだ値は不変の未検証claimであり、全長参照結果・実行証明・target lockには変換しない。

[`23ac79d`](https://github.com/kenmaro3/ckks-train/commit/23ac79d5332086e6237f9d6b548d084417a5c376)のLinux拡張は、実際のGNU x86_64 Linuxで **test / releaseの各profileが13 library +9 integration tests合格**。65,535個のroot検査、実際のTARGET/HOST、compilerと実行環境を記録し、証跡も独立レビュー後に [`f9a8eed`](https://github.com/kenmaro3/ckks-train/commit/f9a8eedf63c55567c80dc81883ee9e7d0dbe9980) で保存した。[元ログとCI provenance](../results/validation/bootstrap-linux-20261106/README.md)を参照。rustc1.85.1/LLVM19.1.7、Ubuntu24.04.5/glibc2.39の記録された環境について、条件付きの構造metadata検証を満たした。encoding・EvalMod内部scale・暗号化した値のrange/precision・実際の鍵・security・memory・本番server admissionは引き続き別の未達項目で、将来のserver実行ファイルも自身のprocessで検査する。小規模runtime fixtureとn16のmetadata検証を、本番N65536の学習・GPU性能測定として扱わない。

[参照session・provenance・保存の契約](study-reference-session-design.md)は独立レビュー後に確定した。Macでの実際の2更新診断と、Linuxでの全長参照結果を生成する能力を分け、前者から本番参照やtargetを作れない設計である。この契約の実装・全長参照結果・target lockは未完。packed NI/AA planner・runnerも実装中であり、主学習の暗号化数値検証はまだ実行していない。既存の84/84 calibration合格は固定corpusの証拠として保持し、主学習や全行runnerの合格に読み替えない。**Task54の全実験行列とTask42/M4の最終判定は未完**である。

サーバーの **2026-11-06 23:21:25.182956 JST**（Unix1793974885.182956）の鍵認証・読み取り観測では、FULL-1はupdate602（0始まり）= **603/750更新完了**、elapsed57771s。直近3更新は予測計数と一致し、FULL-1のpost終端と両FULLのpartialマーカーはない。FULL-2には進捗・post終端がない。chain PID1700979/start ticks465769819とqueue PID1739838/start ticks467041669は同じidentityでstate S、owner.jsonのみ・後続clone未作成。R47/R48/R52は両FULL成功とchain終了検証を待ち、R54と全長暗号化blockは未投入。ジョブ変更・再起動・新しいFULL結果JSONはない。Task42の四つの比、FULL-2対ReBoot seed0=0.9041の±0.5pt判定は結果待ち。一次記録はledger `post-full-observation-20261106-232047.stdout` と対応meta/stderrである。

CIの **23:20:49.439744 JST** 一覧観測では `090f8fb` / `02a26a5` / `9610822` / `878438c` / `edeffa7` はcompleted/success。`23ac79d` [run42230241832](https://github.com/kenmaro3/ckks-train/actions/runs/42230241832) と `8d3a649` [run42231800043](https://github.com/kenmaro3/ckks-train/actions/runs/42231800043) はin_progressだった。前者のfocused Linux stepsの合格と、workspace全体・workflow全体の完了は区別する。一次記録はledger `task-54-ci-observation-progress5.json`。後続の報告・設計commitへCI成功を継承させない。

Macの容量対策では、未使用のビルド成果物・改名前の成果物・取得済みアーカイブ等を累計 **11.3908 GiB（削除前の割当量）**削除した。データセット・測定結果・source・再現用環境と作業中のbuild出力は保持した。23:16:04 JSTの観測で空き約13 GiB、project target約3.7 GiB。削除累計とfilesystem空き量の増減は同じ値ではない。一次記録はledger `mac-cleanup-20261106.json`、`mac-cleanup-summary-20261106.json`、`mac-disk-observation-parser-integrated.json`。


<a id="latest-evidence-20261107-0020"></a>
## 2026-11-07追記: packed NI/AAの小型数値検証と残る研究ゲート

[`f53f34f`](https://github.com/kenmaro3/ckks-train/commit/f53f34f0396e05ecae3d670fee21140454123349)でpacked NI/AA planner・runnerを統合・pushした。独立レビューで見つかった将来batchのparameter/scale照合とHost経由の対象評価器の鍵構造照合を修正し、再レビューを通過した。元の46件の限定検証と修正後10件の限定検証は、それぞれの実行記録として保持する。修正後も2,496行の予測traceは不変で、主検証の閾値・入力・seed・学習率は変えていない。

Macで登録済みの **N=1024・12ケース×4更新** を初めて実行し、48更新・欠測0、bootstrap実行212/212（上限228）で終了コード0となった。全更新の7種の演算数は予測と一致した。独立した生ログ監査もcoverage/result/integrity合格。全complex誤差の最大は2.53557821117368795e-8（E=2^-12以内）、incremental bootstrap誤差の最大は1.26730185407340126e-8（epsilon=2^-20以内）で、全roleの入力範囲も合格した。原ログ・source/executableのidentity・監査・残る制限は[保存した検証証跡](../results/validation/packed-ni-n1024-20261107/README.md)を参照する。これは一つのkey seed153によるLazyの決定的な有限fixtureであり、12独立seed、暗号学的に安全なN2^16、GPU性能や本番メモリ適合性の証拠ではない。512 complex slotsの検査はプログラムが行い、保存した集約値から全slotを独立再復号したとは扱わない。元の84回calibrationは再実行していない。

全長FP32参照結果のbuild/context・保存基盤（D6c2a）は実装中。session/reader、target固定、c3 local blocks、profile/backend接続、全21 profiles×10 seedsとCPU/GPU・I/NI・各ablationの実験行列は引き続き未完である。今回の有限fixture合格をTask54全体の合格に置き換えない。研究計画の範囲や数値ゲートは削減していない。

サーバーの **2026-11-07 00:02:25.381083 JST** の鍵認証・読み取り観測では、FULL-1はupdate627（0始まり）= **628/750更新完了**、elapsed60228s。直近3更新の計数は予測と一致。FULL-2に進捗はなく、両FULLのpost終端・partialマーカーはない。chain PID1700979/start ticks465769819、queue PID1739838/start ticks467041669は同じidentityでstate S、owner.jsonのみ・後続clone未作成。ジョブ変更・再起動・追加投入はない。R47/R48/R52は両FULL成功とchain終了の検証待ちで、RQ1の4比とFULL-2対ReBoot seed0=0.9041の±0.5pt判定は未確定。一次記録はledger `post-full-observation-20261107-000142.*`。

CIではLinux候補`23ac79d`のworkflow全体が成功した。一方、parserの`8d3a649`はlib-testのhex helperに対するClippy警告で失敗し、他5ジョブは成功した。この失敗を保持し、[`5c9c257`](https://github.com/kenmaro3/ckks-train/commit/5c9c257994056cb6356745e5a68923dd501cd669)で同じ16進表記を保つ修正をレビュー・pushした。修正後のローカル対象4テストとlib-testを含むClippyは合格。**00:12:20.389774 JST** の[run42238449687](https://github.com/kenmaro3/ckks-train/actions/runs/42238449687)ではfmt/clippy/research-tools/cuda-check/denyが成功、testはin_progressで、workflow全体や後続commitへ成功を継承しない。

Macの不要生成物は累計11.3908 GiB（割当量）を削除済み。その後の観測で空きは約5.0 GiB、project targetは約3.7 GiBのままで、追加削除は行っていない。空き容量の変動を削除量と同一視せず、ソース・データ・計測結果・再現環境・現在の検証成果物を保持する。

<a id="latest-evidence-20261107-context"></a>
## 2026-11-07追記: 参照結果のprovenance基盤とブロック設計

[`8e66cc6`](https://github.com/kenmaro3/ckks-train/commit/8e66cc69e21637d46486020614789be817cf04e8)で、全長FP32参照結果に必要なbuild/context・内部capture/recheck・保存処理の基盤（D6c2a）を独立レビュー後に統合・pushした。通常のビルドは本番用capsuleを持たず、本番contextを取得できない。参照結果を信頼するための明示的なlocal approval、source/toolchain/環境/実行ファイルの照合と、上書きを禁止する保存処理を実装した。[確定した契約](study-reference-session-design.md)の範囲であり、学習の完了を表す型や全長結果・targetはまだ生成していない。

初回の限定検証はRust unit14件・integration1件・Python16件とClippy/formatが合格した。独立レビューで、ディレクトリへのsymlinkより先に`..`を畳み込むと、実際のリンク先を記録せず対象root外も受理する不備が判明した。2個のPythonファイルを修正し、失敗する元の再現ケースを保存したうえでPython20件が合格、再レビューで指摘を閉じた。修正後のRust sourceは不変で、元のRust検証を再実行したとは扱わない。統合後にはlib-testを含む`cargo clippy -p ckks-train-bench --lib --tests --no-default-features --no-deps -- -D warnings`も合格した。実際のLinux本番build、承認済み実行ファイルによるcapture、ext4/XFSでの保存・復旧・durabilityは未検証である。portable fixtureや状態遷移の模擬検証を、実機での保存成功の証拠にはしない。

[`8be3fc9`](https://github.com/kenmaro3/ckks-train/commit/8be3fc94fe328a1b37fbe20b5d8b522a150ee987)で[packed local blocksと推論専用処理の設計](packed-blocks-design.md)を独立レビュー後に確定した。ブロック全体を一括更新し、更新前のhiddenを受け渡し、AAでは既存のpost-hiddenをそのまま使う。小型検証案の平文計算で最終推論のhiddenが学習時の上限を超えることが分かり、暗号化実行前に学習用0.5と推論用0.6の範囲を分けた。元の入力・学習率・seed・E/epsilon・presetは保持した。I/NIの数値検証はignored testとして別途明示的に実行する設計であり、ブロック実装・metadataによる正確なschedule確認・I/NI実行は未完。NI上限372回は設計上の上限で、実測件数ではない。

次のD6c2bでは学習session・serializer・readerを実装中で、Macでの実行範囲はcanonical Irisの2更新診断に限る。この節の時点でその診断結果の合格は主張しない。全長FP32参照結果、target固定、packed blocks、profile/backend接続、全21 profiles×10 seedsのCPU/GPU・I/NI・各ablationの行列は引き続き未完。**Task42/RQ1、M4、Task54の完了条件は満たしていない。**

サーバーの **2026-11-07 00:58:45.319284 JST** の鍵認証・読み取り観測では、FULL-1はupdate662（0始まり）= **663/750更新完了**、elapsed63610s。直近3更新は予測計数と一致し、両FULLのpost終端・partialマーカーはない。FULL-2に進捗はなく、chain PID1700979/start ticks465769819とqueue PID1739838/start ticks467041669は同一identityでstate S、owner.jsonのみ・後続clone未作成。ジョブ変更・再起動・追加投入はない。一次記録はledger `post-full-observation-20261107-005813.*`。R47/R48/R52は両FULLとchain終端の検証待ちで、4比率とFULL-2対ReBoot seed0=0.9041の±0.5pt判定は未確定である。

CIはSHAごとに区別する。01:01:20 JSTの観測で`f53f34f`の[run42241920369](https://github.com/kenmaro3/ckks-train/actions/runs/42241920369)、01:25:11 JSTの観測で`79da074`の[run42245610391](https://github.com/kenmaro3/ckks-train/actions/runs/42245610391)は、いずれも6ジョブすべてcompleted/success。一次記録はledger `task-54-ci-ni-source-observation.json`、`task-54-ci-ni-publication-final-observation.json`。新しい`8be3fc9`/`8e66cc6`や後続sourceへCI成功を継承しない。

Macの容量対策は累計11.3908 GiB（削除前の割当量）のままで、この追記までに追加削除はしていない。古いbuild成果物と取得済みアーカイブ等を整理し、source・dataset・測定結果・現在の検証成果物を保持している。filesystem空き容量の変動は、この削除累計やprojectの増減と同一視しない。

<a id="latest-evidence-20261107-session"></a>
## 2026-11-07追記: 参照sessionの統合と不足seedの取得方針

[`d76ef05`](https://github.com/kenmaro3/ckks-train/commit/d76ef05034d078b459500f83dc9271aabe88e157)で、参照学習session・結果serializer・reader・保存adapter（D6c2b）を独立レビュー後に統合・pushした。sessionが実際の更新と評価を保持し、全長完了・2更新診断・F64診断・読み取った参照結果を別の型にする。readerは再生成した二つのcanonical identityの全byte列と、全長schedule・分母・provenanceを照合する。設計は[参照session契約](study-reference-session-design.md)を参照。

初回の限定検証は29 unit、既存保存基盤8件、compile-fail14件とClippy/formatが合格。独立レビューで開始時の時計取得失敗を終了時まで持ち越す不備が見つかり、開始直後に失敗してidentity/capture/model生成へ進まないよう修正した。元の失敗テストを保存し、修正後の11 lifecycleテストと実Iris診断・Clippyが合格、同じレビュー担当の再レビューで指摘を閉じた。無変更のreader/digest/保存基盤やcompile-fail検証を再実行したとは扱わない。元のdoctest記録にはbuild lock待機の1行があるが14件は合格し、記録を消すための再実行はしていない。

Macの実Iris診断では、canonical seed0をF32/F64で各2更新し、内部評価とメモリ内JSONの再読込を確認した。成功した各sessionは40 samples、held-out 3/30を報告し、別の2更新sessionでは3回目のadvanceが演算前に拒否された。修正を含む限定回帰全体で34 optimizer updatesを実行したが、いずれも新しいsessionの2更新以内で、連続した全長学習ではない。JSONのproduction provenanceは明示的にunavailableで、科学用結果ファイルやtargetは生成していない。これは性能・全長精度の測定ではない。

**未計測・未完の区分:** 実際の承認済みLinux build/context、全長FP32参照結果、Linux ext4/XFSでの保存・復旧・durability、全長Fashionのresource/serialization確認は未計測。target lockと開始記録は具体設計の独立レビュー中で、thin binary、packed blocks実装、profile/backend接続、全21 profiles×10 seedsとCPU/GPU・I/NI・各ablationの実験行列は未完。既存の84回calibrationと212回の小型NI主検証は再実行していない。Task42/RQ1・M4・Task54全体の合格を主張しない。

不足するeMLP2/3 seed3–9は14個のmodel/seed取得データで、B48/B64の28 profile/seed行に影響する。既存入力のhashとローカル前提ファイルを監査したが、baselineのimport・再取得・pin変更はまだ行っていない。[実装前の取得方針](study-profile-inputs.md#seed-capture-extension-20261107)として、既存0–2を完全一致で再取得できることを先に確認し、新しい別バンドルへ0–9を実際のbaseline APIから出力する方式を選んだ。既存rawやFULL条件は保持し、未実行のtwin学習結果は出力しない。値取得専用exporterはsource-only実装段階で、実行前に独立レビューする。MissingInputは取得・新入力versionの検証が済むまで残す。

サーバーの **2026-11-07 02:06:46.112982 JST** の読み取り観測では、FULL-1はupdate704（0始まり）= **705/750更新完了**、elapsed67705s。直近3更新の計数は予測と一致した。FULL-2には進捗がなく、両FULLのpost終端・partialマーカーはない。chain PID1700979/start ticks465769819、queue PID1739838/start ticks467041669は同一identityでstate S、owner.jsonのみ・後続clone未作成。追加投入・再起動はしていない。一次記録はledger `post-full-observation-20261107-020552.*`。R47/R48/R52は両FULLとchain終了の検証待ちで、RQ1の4比率とFULL-2対ReBoot seed0=0.9041の±0.5ptは未確定である。

CIは **02:18:58.306748 JST** の観測で、`8e66cc6` [run42253471185](https://github.com/kenmaro3/ckks-train/actions/runs/42253471185) と `30043d8` [run42255283140](https://github.com/kenmaro3/ckks-train/actions/runs/42255283140) が全6ジョブcompleted/success。`8be3fc9` [run42251269857](https://github.com/kenmaro3/ckks-train/actions/runs/42251269857)も別の観測で成功した。一次記録はledger `task-54-ci-context-later-observation-20261107.json` と `task-54-ci-context-publication-observation-20261107.json`。後続の`d76ef05`へCI成功を継承しない。

Macでは未使用Sage package cacheの独立割当分489,848,832 bytesを追加削除し、累計削除量は12,720,607,232 bytes（約11.8470 GiB）となった。環境に残る共有hardlink分は削除量へ加算していない。Sage環境109,413項目のmetadata digestは不変で、起動確認は最初の60秒timeout後、再試行で成功した。ソース・dataset・実験結果・再現環境・現在の検証成果物は保持している。削除量とfilesystem全体の空き容量変動は区別する。


<a id="latest-evidence-20261107-full1"></a>
## 2026-11-07追記: FULL-1回収と入力取得コード

FULL-1は03:19:51 JSTにrunner exit0で終了し、元JSON・ログ・終了記録11ファイルを無変更で回収した。[M3報告](m3-report.md)に196入力・B=64の750更新/48000サンプル行を追加。KS比(a/b/c/d)=0.231776/0.386771/0.395063/0.400079はいずれも≤0.70、精度0.9036（自身の双子0.9037、n=1）、学習時間72050.794763s。全長ReBoot時間は未計測、セキュリティ同条件は未達、FULL-2対公開コードseed0=0.9041の±0.5ptゲートは未確定。03:20:57 JSTの観測でFULL-2のsetup開始を確認した。後続R47/R48/R52は両FULLとchain終了の検証待ちで、ジョブの追加・再起動はしていない。

Task54の[具体target/admission契約](study-target-design.md)は33f08aeでレビュー後に確定し、独立した期待値も数値拒否例の修正・再レビュー後に受理した。target lockのコードは実装中で、実際の全長参照・lock・admission・HE開始は未検証。[入力取得用exporter](../tools/export_reboot_block_seed_inputs.py)は2b09ca7で独立再レビュー後に統合し、33件の限定テストが合格。Pythonの既存キャッシュが承認済みsourceと異なるコードを実行する問題を修正した。Macでの実compatibility取得はライブラリ検索パスと依存ライブラリの起動処理で停止しており、seed0–2の完全一致・seed3–9取得・入力version移行はまだ未達。失敗記録を保持し、閾値や既存入力を変更していない。


<a id="latest-evidence-20261107-static"></a>
## 2026-11-07追記: target基盤・取得前提と現在の運用状態

Target lockは`66a4b36`で独立レビュー後に統合済み。[StudyTrainの実行・build契約](study-runtime-design.md)は`50a6dee`で確定した。Referenceと学習のbuild/権限を分け、CPU直列・専用Rayon pool・CUDA＋host poolの固定recipeを定める。実際のLinux承認済みcontext、全長FP32参照、保存・復旧、学習runtime、D7 descriptor/gateは未検証または未実装。Packed-block compositionの9ファイルを独立レビューし、失敗時の応答・途中観測の保持と構造テストに不足を認め、元の実装担当が修正中。通常テスト11件合格・主検証2件ignoredであり、新しい実HE主検証は実行していない。既存84回calibration/212回主検証も再実行していない。

登録ケースの独立期待値は全2,680 family tuples、別枠20 prospective Iris、全7補足記述子、19 canonical vectorsの照合でSPEC/QUALITYレビューに合格した。これは今後のcase projection実装用であり、2,700を研究全体の分母とは扱わない。入力・executor・resource不足やfull block campaignの保留で登録行を削除しない。全21 profiles×10 seeds、CPU/GPU・I/NI・ablationを含むTask54は未完。

入力exporterは`f021616`まで独立レビュー後に統合済み。Mac実compatibilityはeMLP2 seed0の画像payload不一致で停止し、別診断でtrain4,149/940,800、test8,694/1,960,000要素の差、最大4 ULP/2^-22を観測した。ラベルはbyte一致、同seedの重み・順序も一致。他seed/eMLP3、seed3–9取得、新入力versionは未達であり、許容誤差や置換で受理していない。不一致原因は未確定。

[R-REB-7-STATIC](../results/quadruped/raw/R-REB-7-STATIC/README.md)は04:58:18 JST、exit0の読み取り専用観測。Linux NumPyの該当source hashはMacと一致し、module直下から条件分岐のないSVE照会を呼ぶことをASTで確認した。動的import到達性、`.pth`のstartupコード、完全なcontext・native hash・network isolationは未検証で、Linux科学用import/captureは未実行。baselineコードの抜粋は持ち込んでいない。

同観測でFULL-2はupdate39（0始まり）=40/1000更新、elapsed5780s、直近6更新の計数は予測と一致。chain PID1700979/start465769819、queue PID1739838/start467041669は自UIDでstate S、queueはowner.jsonのみ。正しいFULL-2結果pathにはまだファイルがない。後続R47/R48/R52は両FULL成功とchain終了の検証待ちで、追加投入・再起動・キュー変更はない。FULL-2対ReBoot seed0=0.9041の±0.5ptは未確定。

Mac容量対策として、稼働していないSage環境を削除した。394個のinstalled packageのURL・version・hashとconda historyをledger `sage-env-reinstall-20261107/`へ保存し、見積もり結果・実行スクリプト・estimator checkoutは保持。追加削除4,441,407,488 bytes、累計17,264,930,816 bytes（独立割当量）。削除直後の空き容量は9,881,452,544 bytesで、空き容量変動を削除量と同一視しない。Sage再計算には再インストールが必要。現在使うRust/ReBoot/TinyTeXとデータ・結果・原ログは保持した。


<a id="latest-evidence-20261107-bootstrap"></a>
## 2026-11-07追記: Linux起動前提とCIの未実行

[R-REB-8-ARCHIVE](../results/quadruped/raw/R-REB-8-ARCHIVE/README.md)でLinuxの元configを無変更で回収した。固定済みのtracked configとの違いはローカル側のstructure_summaryだけで、共通する6,048個のF64値はbit一致。固定pinを変えず、今後の比較用archiveには新しいcheckout内の検証済み既存trackedコピーを使う。元の出力は別証跡として保持し、remote重み・順序・contextの照合完了は主張しない。

[R-REB-9-BOOTSTRAP-STATIC](../results/quadruped/raw/R-REB-9-BOOTSTRAP-STATIC/README.md)は05:12:43 JSTの固定source読み取り。既存venvの起動時_virtualenv importは、現在の「検証前に読み込んだ管理対象sourceを拒否する」ルールと衝突すると予測される。target起動や科学用importは実行していない。新しい空のstdlib venvと明示した既存package pathを用いる案を設計中で、元環境の変更・新環境作成・guard例外は行っていない。実行前にcreator/startup・launcher・context・native/isolationと完全一致ゲートを検証する。

CIは`50a6dee`までの直近記録では成功。一方、`c6ab260` [run42285149048](https://github.com/kenmaro3/ckks-train/actions/runs/42285149048)と`2434c4b` [run42286055904](https://github.com/kenmaro3/ckks-train/actions/runs/42286055904)は6ジョブすべてstepなしのfailureで終了した。各runのtest注釈は、アカウントの支払い失敗または利用上限によりジョブを開始しなかったと報告している。コード検証の失敗とは区別し、この2commitのCIは未実行として扱う。請求設定・上限・支払いは変更していない。一次API記録はledger `task-54-ci-billing-block-observation-20261107.json`。ローカル限定検証・レビューを続け、直前の成功を後続commitへ継承しない。


<a id="latest-evidence-20261107-blocks"></a>
## 2026-11-07追記: packed-blockの実I/NI検証と全ケースprojection

Packed-block compositionは指摘F1/F2/Q1の修正・独立再レビュー後に`ad607da`で統合した。失敗時のraw応答と途中の観測を保持し、旧U/bias・元generationと局所lossの構造検査を追加。修正後の通常21 tests、metadata、strict Clippy・formatが合格した。各主検証を別にreleaseし、Iの独立監査合格後にNIを1回だけ実行した。[元ログ・runner provenance・独立監査](../results/validation/packed-blocks-n1024-20261107/README.md)を保存した。

MacのIは05:27:05–05:27:09 JST、2 graphs・8更新・2推論・116 client refreshでexit0。全710×512 complex slotsを独立再計算し、最大スカラー誤差3.775018180620039e-9は元の2^-12以内。NIは05:36:23–05:38:26 JST、G2/G3×AU/WN/AAの6構成・24更新・6推論・332 bootstrapsでexit0。全4,050×512 slotsの最大スカラー誤差3.441401240719656e-8は2^-12以内、全332回の実入力対出力の最大増分誤差1.8763991003465872e-8は2^-20以内。固定role range、24更新の7項目計数、332回の計数区間も独立照合した。生ログは無変更で保持し、公開用gzipは展開後のSHA256一致を確認した。

**合格範囲はN1024・512 slots・held-out key seed167の小規模な正しさだけ。** source/executableの前後hashは一致するが、独立監査はプログラムが出力した復号bit列の計算であり、暗号文・秘密鍵から再復号した証明ではない。NIの推論／phase全体の7項目総計数は未出力で、最後のbootstrap後カウンタを総計に流用しない。seed・generation等のsource依存限界も監査に残した。本番N65536/security、GPU、memory、性能、全長blockデータセットの合格ではない。既存84/212検証は再実行していない。

登録case projectionは独立SPEC/QUALITYレビュー後に`cf3f822`で統合した。既存のhash検証済みregistryから全21 profilesの2,680 family cases、別枠20 prospective Iris、全7補足記述子を保持し、入力やexecutorが未整備でも行を削らない。独立期待値との全14項目tuple比較・19 canonical vectors・拒否／公開API境界など44件の限定検証とClippy・formatが合格。全長データをmaterializeする既存1件は明示して除外した。2,700を研究全体の分母や実行済み行数とは扱わず、quick・archival・未解決の補足行から実行権限を作らない。

[R-REB-10-CREATOR-STATIC](../results/quadruped/raw/R-REB-10-CREATOR-STATIC/README.md)は05:45:00 JSTの実base interpreter／hook／site／venvの読み取り観測。base側のdistutils条件とhook identityを確認したが、target起動・新venv作成・入力取得はまだ未実行。Linux optional queryへの限定的なsource対応を次に進める。全長FP32参照、学習runtime、admission／実行handoff、baseline adapter、全行runnerと実測は未完。Task42のFULL-2ゲートとM4後続ジョブも未確定で、上記の小規模検証から合格を継承しない。


<a id="latest-evidence-20261107-linux-launch"></a>
## 2026-11-07追記: Linux入力取得のsource対応と空環境作成

入力exporterのLinux対応は`010bb3c`で独立SPEC/QUALITYレビュー後に統合・pushした。同一の検証済みNumPy source内の任意host照会だけをLinux/x86_64でもEPERMで拒否し、childを実行させず依存自身の既存fallbackへ戻す。Darwinの不在条件は保持し、実際のsys_platform/machineとsource/code/globals/frame・argv等を結び付けたpolicy v2へ更新。古いpolicy/contextはoutput/backend作成前に拒否する。Mac上のfictional fixtureでfocused6件と全exporter43件が合格し、科学設定・seed・前処理・比較・receiptは変更していない。実Linuxのimport・互換性の合格ではない。

[R-REB-11](../results/quadruped/raw/R-REB-11-CREATE-EMPTY-VENV/README.md)で06:08:36 JSTに独立した空のstdlib venvを1回作成した。実行前レビューの2指摘を修正し、子プロセス回収／固定layout検査の8テストと再レビュー後にrelease。wrapper/childはexit0、警告なし、15-entry layout・system-site無効・空のown site・pinned baseへの3 interpreter linksを静的に検査し、原記録の独立監査も合格した。新venvのinterpreter起動、package-path追加、context/native hash、科学用import/captureはまだ行っていない。既存科学環境、FULL-2、後続queueは変更していない。

最新のジョブ観測は05:58:57 JST、FULL-2 update65（0始まり）=66/1000、elapsed9501s、直近6計数は予測一致。chain PID1700979/start465769819、queue PID1739838/start467041669は自UID/state Sでlive。chainの直接child1910150もliveで、その子1910153/1910154は一覧のみ確認した。正しい最終結果pathは不存在、queueはowner.jsonのみ。これは最終精度・完了の証拠ではない。一次記録はledger post-full-observation-20261107-0600.*。後続R47/R48/R52とTask42の最終判定は引き続き待機中。

学習roleのbuild/approval用独立期待値と、取得用のprelaunch/launcher実装準備を進める。全長参照・実学習runtime・全行runner・実測の残項目は前節どおり。新commitのhosted CI成功は未確認で、以前の成功や請求理由で未起動だった別runを継承しない。公開release・課金設定の変更は行っていない。


<a id="latest-evidence-20261107-startup1"></a>
## 2026-11-07追記: Linux起動の失敗記録と学習role実装のレビュー

[R14/R15](../results/quadruped/raw/R-REB-14-15-STARTUP-PREREQUISITES/README.md)の追加観測を独立レビュー後に`450e51f`で統合・pushした。最初のcommand-file読み取りは2MiB上限で失敗し、失敗記録と後続の限定観測を別々に保持した。system observerとbase interpreterを区別し、実際のuname実体とhashを取得した。完全なhelper/native closureや科学用importの合格ではない。

[R16〜19の原記録](../results/quadruped/raw/R-REB-16-19-STARTUP1/README.md)では専用ディレクトリへ固定4ファイルを配置し、別の読み取りで全hash・所有者・mode・inodeを照合した。その後、07:27:42 JSTの**実startup1は未達**。静的preflightは通ったが、子プロセスが自分の固定launcherファイルに対応するimporter-cacheキーを拒否してexit1となった。子の終了・全stream回収は確認済み、startup stageは空、helper preparationや外部package追加・exporter/context/captureには到達していない。Linuxの当該finder値は未記録で、Noneだったとは主張しない。原失敗・sidecar・source・親子JSONを保持し、自動再試行はしていない。限定修正には新しいテスト・レビュー・固定copy/sidecar/evidence・実行releaseが必要。

07:25:40 JSTのFULL-2はupdate101（0始まり）=102/1000更新、直近6計数は予測一致。chain PID1700979/start465769819とqueue PID1739838/start467041669は同じ自UIDでlive、最終結果は未出力、queueはowner.jsonのみ。既存ジョブは変更していない。この観測で07:30の定例確認を兼ね、次の定例は08:25 JST以降。Task42の精度・同一CPU timing等とM4後続キューの判定は未確定。

学習role/buildの変更はRust35件・限定integration1件・Python32 methodsとstrict Clippy/formatを通したが、独立レビューでPythonの非選択emitted pathに`..`が入った場合のcontainment不足が見つかり、修正中・未統合である。Rust側は拒否するため両言語の規則を揃える。試験数は科学的case数ではなく、native build、Train current context/runtime、全行runner、HE実測の完了は主張しない。全登録行・未解決補足・campaign holdは維持する。

Macの不要生成物は累計19,298,938,880 bytes（約17.97GiB）を整理済み。直近の追加対象は非稼働Rust中間objectだけで、primary binaryのhashは不変。現在の空きは別途変動し、約1.1GiBのため既存target・単一Cargo・限定検証を維持する。科学結果・データ・証跡は保持し、Sage再実行時は保存した394 pinsから再導入する。新commitのhosted CIは未確認、以前の請求理由による未起動を成功扱いせず、課金・公開設定は変更していない。

<a id="latest-evidence-20261107-role-build"></a>
## 2026-11-07追記: 学習用build統合と起動修正のsourceレビュー

`f1bdc3e`でReference／学習用の固定build recipe、approval-v2、Rust/Pythonの整合検査を統合・pushした。Pythonの未選択emitted pathに含まれる`.`／`..`はPath構築前に拒否するよう修正し、元の逸脱例を含む限定7テストと独立再レビューが合格した。前節の「修正中・未統合」はこの追記で更新する。元のRust35件・integration1件・Python32 methods・Clippy/formatの検証と修正後の限定検証を区別する。build-script検証ハーネスのmtimeによる実行ファイル選択は軽微な残課題で、レビュー済みsourceと当該実行ファイルの同一性を証明したとは扱わない。

Linux launcherのFIX3は固定された自分自身のファイルキーに対する`finder is None`だけを許す限定修正で、8件のテストと同じ独立レビュアーによる再レビューが合格した。実startup1の失敗とLinux finder値の未観測は変わらない。新しいstartup2用の固定配置・sidecarを準備中で、現時点では新しい配置・起動・context／captureは未実行である。

`f1bdc3e`の[CI run42306740510](https://github.com/kenmaro3/ckks-train/actions/runs/42306740510)は全6ジョブのstepsが空で、各注記に支払い／利用上限による起動拒否がある。CI検証は未実施であり、コード試験の失敗とも成功とも扱わない。課金設定は変更していない。

次の実装は、学習側の参照結果読み取りとReference専用の公開・復旧操作を分ける制御、既存target storeの読み取り専用open、入力の再検証の共通化である。実runtime、全行runner、native reference／admission／HE、Task42／M4の最終測定は引き続き未完。FULL-2の最新実観測は前節の07:25:40 JST（102/1000更新）で、新しい進捗を推定していない。全登録範囲とcampaign holdを維持する。

<a id="latest-evidence-20261107-startup2"></a>
## 2026-11-07追記: Linux startup2の実起動成功

[R20〜23の原記録](../results/quadruped/raw/R-REB-20-23-STARTUP2/README.md)では新しいv2配置の4ファイルを別の読み取りで照合し、FIX3の実startup2が07:53:48.768885 JSTに子PID1963722／exit0で終了した。親子のrelease／argv／child hashが一致し、stream回収完了・cleanup errorなしを確認した。実際のstartup／helpersの2段階を記録し、固定launcherのimporter-cacheキーは両段階でnullだった。これは今回の観測であり、未記録だったstartup1の値を補完しない。

実起動記録の独立レビューも合格した。合格範囲は隔離venvのidentityと標準ライブラリ準備の起動観測まで。外部package追加、exporter読込み、科学用context／capture、seed取得は未実行で、完全なhelper/native closureや数値互換性を主張しない。先行する失敗記録を保持し、新しい段階の実行条件は別途確認する。FULL-2／GPU後続キューは変更も再観測もしておらず、最新の本番進捗は07:25:40 JSTの102/1000更新のままである。

学習側の読み取りと公開・復旧操作を分ける変更は限定55テスト・strict library Clippy／formatと独立sourceレビューに合格し、`2fa380f`で統合・pushした。実際のTrain contextによる読み取り／書込み拒否の統合検証はruntime実装後に残す。全行runnerと研究ゲートの未完事項、全登録範囲・campaign holdを維持する。

<a id="latest-evidence-20261107-r2-runtime"></a>
## 2026-11-07追記: 検証ハーネス修正とruntime実装準備

`aedb16d`でbuild-script検証のmtime選択を廃止し、現在の`build.rs`と同一のバイト列から一度だけコンパイルした実行ファイルを使うよう修正した。3件の限定テスト（4 recipe・計12 synthetic dispatchを含む）と独立再レビューが合格し、R2は今回のハーネスと実行記録について解消した。過去のmtime選択による証跡を遡って認証するものではなく、Linuxの学習用build／context／実行結果を得たという意味でもない。

`4be5aa3`でruntimeの単一core内lease、CUDA観測adapterを先行する実装順、native moduleの1GiB上限・1MiB chunk読み取りと既存SHA-256のprivate streaming化を、理由・代替案とともに実装前の設計として記録した。独立設計レビュー済み。CUDA側の実関数ポインタ・実デバイス識別・共有compile optionsのAPI範囲も確認したが、実装／実機測定は未完である。

Macの容量低下に対し、非稼働objectと24時間超の古い重複rlib/rmetaを499ファイル、678,801,408 allocated bytes整理した。各crateの新しい2 variantと全実行ファイル・dylibを残し、primary packed-block binaryのhash不変を確認した。累計削除は19,977,740,288 bytes（約18.61GiB）。ソース・実験結果・データ・PDFは保持した。削除したcache variantが将来必要になれば再生成される。直後の空きは1,305,198,592 bytesで、以後のビルドも0.75GiB floorを都度確認する。

ReBootの次段階は、既知の標準ライブラリsource・installed metadata形式・固定候補pathを調べる限定読み取りの準備中。全package/native内容hash、context／capture、学習の追加実行はまだ行っていない。標準ライブラリsource本文は監査用の非追跡入力として扱い、公開MIT配布物にコピーしない。最新のFULL-2実観測は07:25:40 JSTの102/1000更新で、既存GPUキューの状態を新たに推定しない。全登録範囲とcampaign holdは維持する。

<a id="latest-evidence-20261107-0825"></a>
## 2026-11-07 08:25 JST追記: FULL-2の定期確認と実装状況

[今回の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-0825/README.md)では、08:25:52 JSTに既存chainと後続queueの同一PID／開始時刻／所有者を確認した。FULL-2は127/1000更新まで進み、末尾6更新の予測一致はtrue。最終結果JSONはまだなく、queueにはowner.jsonのみだった。処理は変更していない。これは途中経過であり、精度判定やTask42／M4の完了を意味しない。

CUDA runtime観測adapterの実装を開始した。ReBoot前提確認ツールは限定テスト16件に合格したが、独立レビューで読取り失敗時の候補情報保持と未確認を区別する集計表示に修正事項が見つかり、実行前に修正する。新たなcontext／captureやGPU測定結果は得ていない。全登録範囲・campaign holdを維持する。

<a id="latest-evidence-20261107-cuda-r24"></a>
## 2026-11-07追記: CUDA runtime観測APIの統合とR24の未完了記録

CUDA側の固定15 native export・実singleton・device識別情報を保持して再確認する観測APIと、共有compile options／2つのimmutable device accessorを`3bfbed3`で統合した。独立sourceレビュー後、仕様とのquery順序差C1を修正し、実保存ログで3件のREDから限定10テスト全件GREENを確認、再レビューでも合格した。初回の通常構成4件・doctest5件・Clippy／formatは実装担当の実行報告があるが、初回raw transcriptは保存されていないという証跡上の制限を残す。4 kernelのバイト列は不変。Mac上の限定検証であり、実GPU観測・Train context・native module照合・D7 admission／全行runnerの完成を意味しない。

[R24の派生運用記録](../results/quadruped/observations/R-REB-24/README.md)では、08:59:43 JSTに標準ライブラリ23ファイルの監査用本文を取得した（18既知pin一致・5新規identity）。本文は非追跡ledgerにだけ保持し、公開側には識別情報の派生summaryと自作observer／controllerを置く。summaryは元stdoutではなく、科学測定結果でもない。

R24はinstalled metadataのlogical header上限8192 bytesで止まり、exit1／未完了だった。metadataを550058 bytes読んだが、対象package／file／headerの識別情報は出力に残らず、その内訳を推定しない。checkout／native／input／context／captureの各段階は未到達。原記録を保存し、失敗を成功扱いしたり上限を自動で広げたりしていない。

次の診断は、同じ上限のまま読み取れたmetadata fileのhashとheader名・長さ・行番号だけを失敗情報に残す。対象が未特定なのでpackageの推測や上限変更を採らず、取得済み23本文の再読取りも避け、同じroot／childに限定したmetadata専用確認を別途レビュー・実行判断する方針を理由・代替案とともに記録した。現在はその限定コードを準備中で、追加のremote確認は未実行。FULL-2の最新実観測は08:25:52 JSTの127/1000更新であり、R24から学習／後続queueの進捗を推定しない。全登録範囲とcampaign holdを維持する。

<a id="latest-evidence-20261107-0947"></a>
## 2026-11-07 09:47 JST追記: FULL-2と残る実装

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-0950/README.md)では、09:47:06 JSTに同一所有者・開始時刻のchain／queueが稼働中であることを確認した。FULL-2は162/1000更新、末尾6更新の予測一致はtrue。最終JSONはなく、後続queueはowner.jsonのみ。パスの0950は識別用ラベルで、実観測時刻ではない。最終精度・Task42／M4の合格は未確定で、ジョブは変更していない。

CUDA観測APIに続くTrain runtime／contextの実装を開始した。serial CPU・専用Rayon pool・CUDA hostの3構成、process識別、保持したnative moduleの照合を対象とする。全行runnerと実機通し試験は引き続き未完了。R24の失敗情報を保持する限定修正はローカル6テストと既存6テストに合格し、独立レビュー中で、metadata専用remote確認はまだ実行していない。全登録範囲とcampaign holdを維持する。

<a id="latest-evidence-20261107-r25"></a>
## 2026-11-07 09:50 JST追記: metadata停止箇所の特定

限定修正FIX2は独立sourceレビューに合格した。[R25の元運用記録](../results/quadruped/observations/R-REB-25/README.md)では、09:50:13 JSTのmetadata専用確認がexit1／未完了となり、停止候補を保持できた。対象はmatplotlib-3.11.2のMETADATA（80264 bytes、元記録に実hashあり）で、Licenseフィールドが物理171行目で累積8219 bytesとなり、据え置いた8192 bytes上限を超えた。8219は停止時の長さであり、License全体の長さではない。本文・値は出力していない。

metadataの読取りは550058 bytes、source／smallは0。失敗前の部分inventoryと原因を保存し、context／captureや後続collectorには進んでいない。R24の欠落情報を過去にさかのぼって補完せず、今回の観測と分ける。上限の自動変更や成功扱いはせず、実際の形式に基づく次の扱いを理由・代替案とともに検討する。FULL-2の最新実観測は09:47:06 JSTの162/1000更新のままで、GPU計測待ち・全登録範囲・campaign holdを維持する。

<a id="latest-evidence-20261107-runtime"></a>
## 2026-11-07追記: Train runtime／contextの統合

`5e763f0`で、serial CPU・専用Rayon pool・CUDA hostの3構成を保持するprivate runtimeとTrain contextを統合した。現在のprocess／thread／poolを再確認し、CUDA側の観測APIと保持したnative module fileを照合する。Linux限定ProcessStampと、既存SHA-256のstreaming化も含む11ファイルの変更で、独立sourceレビューは仕様・品質とも合格、指摘事項なしだった。

保存した実行ログでは、各構成のprovenance限定47件、Reference関連52件、専用pool・module fileの拒否例、独立SHA期待値、7doctests、拒否理由を照合した7つのrustc compile negative、3構成のstrict Clippy、formatが合格した。構成間で同じテストを含むため件数を単純合算しない。途中で検出したReference拒否順序の回帰は修正し、失敗ログも保存した。Macの限定検証であり、実LinuxのTrain公開読取り／書込み拒否統合とfork確認、実GPU観測／JITは未計測。

現時点のdispatcherはcontext確認後にMissingExecutionDependencyを返す。CaseProvenance、具体的な実行条件の記録・admission／durable start・HE開始／checkpoint・全行runnerはまだ未完了で、この統合を科学実験の実行成功とは扱わない。CaseProvenanceの独立literal fixturesは準備・レビュー中。ReBoot側はR25の観測を根拠に、長い非選択headerを既存file上限内の形式情報として記録する修正を進める。物理行8192／選択値256と既存read/output上限は保持するが、非選択headerの一律logical8192拒否は変更する。理由と代替案は実装前に記録した。

FULL-2の最新実観測は09:47:06 JSTの162/1000更新のまま。最終精度／GPU実測／Task42・M4・M5の全体判定、全登録範囲、campaign holdは変えていない。

<a id="latest-evidence-20261107-r26"></a>
## 2026-11-07 10:13 JST追記: 残る前提情報の限定確認

[R26の派生運用記録](../results/quadruped/observations/R-REB-26/README.md)では、10:13:35 JSTに限定確認がexit0／bounded_observation_completeで終了した。追加4標準ライブラリ本文は既知pinに一致し、非追跡ledgerだけに保持した。公開用summaryは本文4項目を除いた派生物で、元stdoutや科学測定結果ではない。

metadataは66distribution・198file entriesを記録し、必要名の欠落・重複・未対応形式は報告されなかった。長いLicenseの最終正規化長はmatplotlib／numpy／pandas／scipyで59883／46523／50660／45750 bytesであり、値は出力していない。2つのstdlib rootは206／5項目の浅い走査を完了し、metadata候補なしだった。これは形式・所在の観測であり、target実行やhelperの検証完了を意味しない。

exporter用の正規checkout候補は存在せず、歴史producer出力の実hash 9d6a2148…は期待76c493e2…と不一致で、既存R8が確認したtracked版のstructure_summary有無による差を保持する。期待pinのtracked archiveを新しいcheckoutで確認する必要があり、元JSONを直して合わせない。native 3種類は1／1／9候補のmetadataのみで、内容hash・loadは未実行。未圧縮MNIST4ファイル、canonical NPY4prefix、既存の重み／順序36pathは観測され、現在のexporter contextが必須とするgzip4ファイルは不在だった。完了したのは限定確認であり、必要条件の承認・context／captureは未完了。既存ファイルの修復やpinの差替えはしていない。

CaseProvenanceの独立literal fixturesはレビュー後に採用し、固定codecとthin Train入口の実装を開始した。実測に結び付いたcapture／admission／全行runnerは依然として未完了。FULL-2の最新実観測は09:47:06 JSTの162/1000更新で、R26から後続GPUキューの進捗を推測しない。全登録範囲・campaign holdを維持する。

<a id="latest-evidence-20261107-case-inputs"></a>
## 2026-11-07追記: CaseProvenanceのcodec・入口と入力原本の準備

CaseProvenanceの固定codecとthin `study_train` 入口を実装し、独立sourceレビューで仕様・品質とも合格した。6つの正例の完全なバイト列と67の負例、明示的なruntime／worker指定、raw OS path、登録条件の選択と拒否経路を確認した。Referenceの既存形式は変更していない。serial／parallel／CUDA-featureの限定テスト、既存Reference関連58件、privacy doctestsとstrict Clippyも保存ログで合格している。構成間で重なるテストを合算せず、CUDA-feature検証を実GPU測定とは扱わない。

今回の入口は実runtime／context／登録条件を確認した後、未実装の実行依存を明示して終了する。実測に結び付いた `capture_case`／`recheck_case`、具体的な実行条件・gate・admission／durable start／HE開始、全行runnerは未完了である。2,680 family行、20 Iris行、7 supplementの登録と未解決項目を保持する。次の実装対象は、既存の暗号演算と一致する学習前のレベル・スケール・演算回数計画である。

ReBoot再現用のMNIST gzip原本4点を既存の配布元からMacの隔離領域に取得し、exporterの固定SHA-256との完全一致を確認した。既存の未圧縮4点も含む計8点（66,544,770 bytes）は、読取り前後のファイル識別情報と全体hashが一致した。これはローカル入力の保全確認で、科学測定結果ではない。サーバーの既存データは変更しておらず、新しい正規checkoutと専用入力rootの準備を進める。追加helperの限定sourceレビューは合格したが、実context／capture・全package／native内容の確認は未完了である。

FULL-2の最新実観測は09:47:06 JSTの162/1000更新のまま。最終精度・GPU計測・Task42／M4／M5全体の合格は未確定で、既存キューとfull encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-1048"></a>
## 2026-11-07 10:48 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1047/README.md)では、10:48:05 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は187/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1047は依頼時刻のラベルで、実観測時刻とは区別する。ジョブ・キューは変更していない。

この途中経過を最終精度やTask42／M4の合格とは扱わない。学習前のレベル・スケール・演算回数計画器は実装を開始し、実暗号比較は限定テストの条件を確認してから実施する。新しいexporter checkout／入力rootの配置ツールはローカル検証と独立sourceレビューを通過したが、サーバーへの配置・readbackはまだ未実行である。全登録範囲とfull encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-r27"></a>
## 2026-11-07 10:54 JST追記: exporter専用checkoutと入力8点の配置完了

[R27の運用記録](../results/quadruped/raw/R-REB-27/README.md)では、preflight・転送・作成後照合の3段階がexit0で終了し、10:54:10 JSTに`staging_verified`となった。専用checkout `/home/kmihara/hekit-exporter-010bb3c-task54-v1` のdetached HEADは`010bb3c`で、全1,293項目のGit tree inventory、exporter・tracked config・重み／順序を合わせた38ファイル、MNIST原本8ファイルのサイズとhashが固定値に一致した。exporter限定と全体のstatus／diffはいずれも空だった。元のproducer出力は修正せず、tracked版のconfigを正規checkoutに配置した。

専用data rootは`/home/kmihara/hekit-stage-task54-010bb3c-v1/payload/data`で、8ファイルはその`MNIST/raw`以下にある。既存パスの24項目の限定確認は前後で一致し、元のbaseline側でgzip4点がない状態も維持した。両方の新規rootは既に存在するため、作成処理を再実行しない。独立sourceレビューと実記録レビューは配置の範囲で合格した。controllerは入力hash・サイズ・安定確認booleanを保持するが、比較したfd／pathの生stat情報までは保存しておらず、そこからの再検証はできないという制限を残す。

完了したのは入力とcheckoutの配置であり、exporter context／capture、全package・native・canonical内容の照合、追加seedの取得や学習実行は未完了である。既存の計時ジョブ終了待ち条件を維持する。FULL-2の最新実観測は10:48:05 JSTの187/1000更新のままで、今回の配置から学習・GPUキューの進捗を推定しない。Task42／M4／M5の全体判定、全登録範囲とfull encrypted-block campaign holdは変えていない。

<a id="latest-evidence-20261107-1149"></a>
## 2026-11-07 11:49 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1148/README.md)では、11:49:19 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は213/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1148は依頼時刻のラベルで、実観測時刻とは区別する。完了したcontroller記録を回収し、監視を再実行せず、ジョブ・キューも変更していない。

学習前の計画器と実暗号演算の比較テストでは、loss deltaとbackward deltaが同じ観測名を使う不具合を検出した。純粋テストで再現後、識別名を分け、重複を明示的に拒否する修正を行った。静的検査は合格したが、11:54:35 JSTに限定再実行は3件合格・5件失敗で終了した。5件はいずれもNIのbias gradient中間値を記録するフックの欠落を検出したもので、後続の2回帰テストは未実行である。最初の失敗記録を保持し、計画器全体の合格とは扱わない。FULL-2の最終精度、後続GPU計測、Task42／M4／M5の全体判定は未確定である。全登録範囲とfull encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-r28"></a>
## 2026-11-07 12:29 JST追記: 配置停止と権限解釈の訂正

[R28の運用記録](../results/quadruped/raw/R-REB-28/README.md)では、12:26:02 JSTのsidecar配置がexit1で停止した。既存2プログラムの照合後、exporterの期待属性と一致しないとして停止し、レビューした処理順序からsidecar書込みには未到達と確認した。12:29:07 JSTの別途許可した読取り専用診断もexit1で、exporterの実permission値436（0664）を保持した。

原因は、こちらの期待値への換算ミスである。R27のfull stat mode 33204は0o100664で、permission bitsは今回と同じ0664となる。過去記録と今回のpath・size・hash・uid・device・inode・正規化したpermissionの7項目は一致し、権限が変更されたという証拠はない。誤った0644解釈を含む元の準備・レビュー・依頼記録を改変せず、独立診断レビューで明示的に訂正した。

診断はsidecar・directory・証拠ファイル不在の確認に進んでいないため、readback合格や現在の不在を示すものではない。再試行、chmod、入力修復、context／capture、ジョブ変更は行っていない。元の一回限りの実行記録を保持し、実記録に一致する期待値と新しい固定ローカル記録名の限定修正を進める。修正コードの独立レビュー後に、別途具体的な配置操作を判断する。Task42／M4／M5の全体合格、全登録範囲とfull encrypted-block campaign holdは変えていない。

<a id="latest-evidence-20261107-1249"></a>
## 2026-11-07 12:49 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1249/README.md)では、12:49:42 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は238/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1回の観測を同じcontroller handleで終端まで回収し、再試行やジョブ・キューの変更はしていない。

これは途中経過で、最終精度やTask42／M4の合格を意味しない。次の通常確認は13:49 JST頃とする。全登録範囲とfull encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-r29"></a>
## 2026-11-07 12:55 JST追記: 修正後の配置と読戻し完了

[R29の運用記録](../results/quadruped/raw/R-REB-29/README.md)では、期待permissionを実記録どおり0664に直した限定修正が独立sourceレビューを通過した。元のR28失敗記録と一回限りの実行記録を保持し、新しい固定ローカル記録名で12:54:43 JSTに設定sidecarの配置、12:55:55 JSTに独立した読戻しがそれぞれexit0で完了した。サーバーの既存ファイルの権限や内容は変更していない。

新規sidecarは309421 bytesで固定hashに一致し、0600で作成された。読戻しの4ファイルのpath・size・hash・uid・mode・device・inodeは配置記録と完全に一致し、2つのdirectory識別情報も固定値に一致した。後続のcontext証拠3パスは、その観測時点で存在しなかった。両処理は完全なstdout／空stderrを保持し、cutoffや再試行なしだった。

完了したのは配置・読戻しであり、context起動、追加seed取得、学習やGPU計測は未実行である。invoke／collectは未許可で、既存の計時ジョブ終了待ち条件を維持する。FULL-2の最新実観測は12:49:42 JSTの238/1000更新のまま。Task42／M4／M5全体の合格、全登録範囲、full encrypted-block campaign holdは変えていない。

<a id="latest-evidence-20261107-p2"></a>
## 2026-11-07追記: 学習前の正確な計画器P2と小規模実暗号検証

学習前に実際の素数列に基づいてレベル・スケール・演算回数を計画するNN APIを実装した。既存の暗号学習演算に限定的なテスト観測を加え、観測位置・件数・係数の対応、入力時と整列後のmetadata、更新前後の重み／bias／velocity、block境界を計画値と双方向に照合する。暗号演算の順序、preset、許容誤差、bootstrap方針は変更していない。

[Mac上の検証記録](../results/validation/task54-p2/README.md)では、純粋unit16件・外部pure13件、strict Clippy・format・コンパイルが合格した。修正後の比較処理を使った既存8テスト群を1回実行し、8件すべて合格した。生ログで93論理結果（99 stage更新）と220回の実bootstrap／正常returnを確認し、実行前後の9ソースhashも一致した。bootstrap内部の演算はstage／blockの計測値に含まれており、二重加算していない。初期レビューと限定修正レビュー、最終実行結果の独立レビューはいずれも確認済みで、P2の実装・限定検証範囲を受け入れる。

以前のrole重複・NI観測フック欠落は失敗記録を保持して修正した。追加の純粋回帰例は、到達可能な状態で構造的にはbootstrap不要と判定された後、正確なplaintext scaleの最小値条件に失敗することを示す。これは専用のテスト用パラメータでのInvalidParamsであり、ULP探索失敗や本番設定の不具合を意味しない。先の672軌道・2688遷移の探索で該当例が見つからなかった結果も変更していない。理由・代替案と実行前の判断を検証記録に残した。

今回の合格は選択した小規模CPUケースに限り、全条件の直積、secure profile、精度・学習品質、GPU性能、resource／security判定には広げない。P3の具体的な実行前判定、P4／P5の実行ブリッジ、全行runnerは引き続き未完了である。FULL-2の最新実観測は12:49:42 JSTの238/1000更新のままで、最終精度・後続GPU計測・Task42／M4／M5全体の合格は未確定。全登録範囲とfull encrypted-block campaign holdを維持する。


<a id="latest-evidence-20261107-1349"></a>
## 2026-11-07 13:49 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1349/README.md)では、13:49:44 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は263/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1回の観測を同じcontroller handleで終端まで回収し、再試行やジョブ・キューの変更はしていない。

これは途中経過で、最終精度やTask42／M4の合格を意味しない。次の通常確認は14:49 JST頃とする。context／captureの起動は行わず、既存計時ジョブ終了待ち条件、全登録範囲、full encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-p3"></a>
## 2026-11-07追記: 入力・設定を結び付ける実行計画P3の限定検証

P3のソース段階を独立レビュー後に受け入れた。登録済みcaseと実際の入力、完全な学習率列・参照値、素数／bootstrap設定、順序付きmicro partition、更新前後のmetadataを結び付け、全21 profilesとI／WN／AU／AAの候補を、欠落・不正・未証明の状態も残して扱う。実際のIris materializationでは2400更新分の完全な計画を保持し、別に区別した2更新のprefixだけを計画する。暗号学習や精度の測定ではない。

[Macの純粋検証記録](../results/validation/task54-p3/README.md)には、fix1のunit11件・integration5件の合格と実行前後のsource hashを保持した。その後のfix2は既存テスト内81行だけの追加で、製品コードと他のテストは同一のまま、変更した1件を実行して合格した。strict Clippy・formatと独立した修正レビューも確認済み。最終保存の容量超過で完了済み2更新が失われる問題と、失敗stageが欠ける問題は修正前に再現し、元の失敗ログを残した。構築時の上限値と実際の使用量を区別し、各係数・速度状態の値の省略や、異なるstageの教師ラベルの取り違えを検出するテストを追加した。

この範囲の検証ではkeys／Encoder／evaluator／HEは0であり、full profileのメモリ適合・暗号精度・range・security・性能を証明しない。実際のP4／D7消費側での到達可能なkey factory spy、真正なgate／startへの接続は必須・未完了で、P3全体、Task54、M4／M5全体の完了を主張しない。全実験行列、full encrypted-block hold、FULL-2の評価条件は変更していない。

<a id="latest-evidence-20261107-1449"></a>
## 2026-11-07 14:49 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1449/README.md)では、14:49:56 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は289/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1回の観測を同じcontroller handle62197で終端まで回収し、再試行やジョブ・キューの変更はしていない。

これは途中経過で、最終精度やTask42／M4の合格を意味しない。次の通常確認は15:49 JST頃とする。context／captureの起動は行わず、既存計時ジョブ終了待ち条件、全登録範囲、full encrypted-block campaign holdを維持する。

<a id="latest-evidence-20261107-p4-preflight"></a>
## 2026-11-07追記: CPU暗号学習との比較に使う検証データの感度確認

[平文テストの記録](../results/validation/task54-p4-preflight/README.md)では、既存D6b F64学習器を使い、誤った損失係数・学習率・母数・末尾micro・教師ラベル・velocity処理を、最初に違いが出る更新で検出できるか確認した。元の小さい初期値は識別力不足だった。信号の強い固定データも、最初のNI許容誤差5e-4では57条件中5条件が必要な8倍の差に届かなかった。この実測FAILと元のソース・ログは保持した。

HEを実行する前に、同じデータと8倍の条件を保ち、NI許容誤差をより厳しい4e-4へ固定した。Iは1e-5のまま。別版v2を1回実行し、信号の強いデータでは各backendの57条件がすべて合格、最小分離は8.789倍だった。元の小さい初期値は引き続きFAIL。初回136回・v2で136回、合計272回の実際の平文更新を行い、両版の入力・状態のbits、range、誤実装との差の絶対値が同一であることを独立レビューで照合した。数値や条件を混ぜて元のFAILを上書きしていない。

検証ハーネスのsource／strict Clippy／formatと、v2の限定した感度検証は合格した。派生TSVの古い実行名コメントは元を保存して修正し、数値payloadと生ログは変えていない。これは暗号学習の実測ではなく、keys／Encoder／evaluator／HEは0である。暗号計算が4e-4を達成できるか、P4のclient／session／packet／poison処理、実際のkey factory前の検証、P5／D7と全行実行は未検証・未完了。科学実験の精度基準、preset、全登録範囲、計時ジョブの待機条件は変更していない。

<a id="latest-evidence-20261107-1549"></a>

## 2026-11-07 15:49 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1549/README.md)では、15:49:51 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は314/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。1回の観測を同じcontroller handle43642で終端まで回収し、再試行やジョブ・キューの変更はしていない。

<a id="latest-evidence-20261107-p4-cpu"></a>

## 2026-11-07追記: P4単一段CPU学習経路の限定実測

[実装・検証証跡](../results/validation/task54-p4-cpu/README.md)を保存した。clientが秘密鍵と入力暗号化を所有し、private packetを既存のinteractive／non-interactive trainerへ渡す単一段CPU経路を実装した。計画に結び付いた設定・状態・七つのinclusive計数・更新進捗を検査し、処理途中の失敗後は再実行を拒否する。optimizerの漸化式は既存実装を使う。

合格範囲は小規模synthetic CPU contractである。実際に選択したignored testは1件実行・1件合格、20 sessionsで32更新成功と2件の意図した途中失敗、実bootstrapは71回だった。全W/bの平文比較はIで12更新、NIで20更新。最大絶対誤差はそれぞれ7.64926433305390674e-9／1.89993002996935445e-8で、事前固定の1e-5／4e-4以内。拒否・replay・poison後retryについて268回の状態不変比較も完走した。source／strict Clippy／formatと実行証跡の独立レビューが合格した。

初回実行は、syntheticのcurrent .04／reference .08をcanonicalのfull-maximum検査に渡したため、鍵生成前に失敗した。失敗記録を保持し、private fixtureだけを既存field codecへ接続する修正をレビューした。canonical検査、rate／reference、誤差基準を変えず、別の2回目の実行で上記を確認した。報告では初回の消費した準備処理も累積に含める。Macのテスト時間は性能比較やサーバー計測に使用しない。

未達・未計測はcanonical owned D7判定／durable startとproduction run_owned、full pre-key受入、secure profileのresource／range／security／precision、P5複数層、GPU、全行・quick科学実験である。NI内部kappa／maskと完全なtyped P3 traceは未観測、foreign-basis負例はIのみ、wrong-owner負例はprivate association tokenである。本結果をTask54／M4／M5全体の合格や本番精度・性能に拡張しない。全登録範囲、full encrypted-block hold、FULL-2の判定条件を維持する。

<a id="latest-evidence-20261107-1649"></a>

## 2026-11-07 16:50 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1649/README.md)では、16:50:04 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は339/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。16:49以降の予定枠から1回観測し、同じcontroller handle69535で終端まで回収した。再試行やジョブ・キューの変更はしていない。最終精度・Task42／M4の判定は引き続き未確定である。

<a id="latest-evidence-20261107-p5-preflight"></a>

## 2026-11-07追記: P5複数層比較データの感度検証

[平文検証と失敗を含む証跡](../results/validation/task54-p5-preflight/README.md)を保存した。既存F64 Engineの2段／3段・4系列で、local headや更新後hiddenの誤使用、local update欠落、schedule／tail／row再開の誤り、stage別targetの局所性を検査した。最初のv1はIで36/36合格、NIでは許容誤差4e-4の8倍に届かない7項目がありFAILだった。元の出力・失敗判定を保持している。

HEを実行する前に、入力・初期値・rate・reference・control・必要倍率8を維持し、将来のNI誤差上限を3e-5へ厳しくした。Iは1e-5のまま。理由・代替案を設計報告に記録し、source review後に別のv2を1回実行した。I／NIの各36項目すべてが合格し、最小分離はそれぞれ31.782457081491742倍／10.594152360497247倍だった。30,836件の入力・状態・range等の出力が順序も含め一致し、全72比較の元値・bits・絶対差も不変であることを独立レビューで確認した。同じF64差を二つの誤差上限で評価した結果で、backend別の暗号実測ではない。

各実行は156 Engine更新・61 Engine生成・179明示forwardを実際に完走し、失敗を含む累積は312／122／358となる。各698 CHECKは合格、baseline hidden最大絶対値は0.652500000000000080で0.75以内。source／strict Clippy／formatと限定したv2感度検証は合格した。別のempty-phase testと将来のpaired-oracle phaseは未実行である。

P5のkeys／Encoder／evaluator／HEは0であり、暗号学習が3e-5を達成できるかは未検証。複数層のclient／session／packet／retention、canonical D7とdurable start、secure profile、GPU、全行・quick科学実験、P5／Task54／M4／M5全体は未完了である。全登録範囲、full encrypted-block hold、FULL-2の判定条件は維持する。

<a id="latest-evidence-20261107-1750"></a>

## 2026-11-07 17:51 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1750/README.md)では、17:51:09 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は365/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。17:50:05以降の予定枠から1回観測し、同じcontroller handle58137で終端まで回収した。再試行やジョブ・キューの変更はしていない。最終精度・Task42／M4の判定は引き続き未確定である。次の通常確認は18:51:10 JST以降とする。

<a id="latest-evidence-20261107-p5-cpu"></a>

## 2026-11-07追記: P5複数層CPU学習経路の限定実測

[実装・検証証跡](../results/validation/task54-p5-cpu/README.md)を保存した。既存BlockMlp／BlockTrainer／BlockNiTrainerを使う2段・3段のclient／暗号packet／sessionを接続し、更新前hiddenの層間受け渡し、stage別target、AAの同一暗号文再利用、全stageの状態・inclusive七計数・途中失敗後の再実行拒否を検査した。共通処理を変更したため既存P4も1回再実行し、32更新成功＋2意図した失敗、71実bootstrap、268状態不変比較が合格した。

P5は小規模synthetic CPUの16session・42attemptを実行し、40更新成功＋2意図した途中失敗を含む1件のテストが合格した。実bootstrap313回、client refresh357回、成功handoff84個。重み2,304・bias784・old hidden1,800、計4,888項目の全要素を既存F64 Engineと比較し、Iの11endpoint最大絶対誤差は8.56300169660428878e-9、NIの29endpointでは9.06713890636012820e-8だった。事前固定の上限I1e-5／NI3e-5以内であり、元のpure v1失敗・v2厳格化の記録は変更していない。

独立レビューで、全比較の期待値とoracle出力bits、全40endpointの要素網羅・最大値、実測七計数とstage＋handoffの合計、全guard列と68状態不変比較の完了を確認した。最初のguardがharness行に付いていたため報告の総件数を1,376から1,377へ修正したが、元ログ・各処理の件数・数値・判定は不変。元報告も保持した。source／compile／strict Clippy／formatと、P4回帰・P5実行証跡の独立レビューが合格した。Macのテスト時間を性能比較には使用しない。

snapshot公開APIは成功直後の使用だけをこの検証で確認した。初回成功前やpoison後は一貫したendpointとは限らず、広い用途向けの利用条件明記／拒否処理は軽微な残項目である。negative mutationの一時コピーは未観測の診断workspaceに含める。retentionは保持payloadと推定・未観測領域の区分であり、実ピーク・RSS・本番メモリ適合ではない。canonical D7とdurable start／全行runner、secure profile、GPU、全登録科学実験、P5／Task54／M4／M5全体は引き続き未完了で、full encrypted-block holdとFULL-2の判定条件を維持する。

<a id="latest-evidence-20261107-d7-train-capture"></a>

## 2026-11-07追記: 全実験runnerの入力・実行環境provenance接続

[実装・検証証跡](../results/validation/task54-d7-train-capture/README.md)を保存した。保持した実行所有者の中で、選択したcaseのprofile／seedを解決し、canonical inputを1回生成して、Train専用のcapture／recheck／CaseProvenance符号化へ接続した。実際のexecutorのworker数と同じHostを使い、Reference専用経路の固定1や環境変数を学習側のスレッド数として流用しない。context／case／inputの借用と同一性、source／receipt／executableの検査を維持する。

compile REDから実装し、最終no-run／strict Clippy／逐次CPU・CUDA feature check／formatと独立source reviewが合格した。Macの限定テストは31parent（新規5・回帰26）と隔離child1が合格、doctest9（compile-pass1・compile-fail8）も合格した。Iris emlp1のseed0／7だけを各1回生成し、別入力・別所有者・同サイズ別poolの拒否、実pool2対environment99、Caseの独立6正例／67負例とReferenceの既存bytesを確認した。

全32Cargo呼出しはexit0だったが、controllerはdoctestの最後のsuite8だけを総数9と比べ、exit1になった。元controller・ログ・metadataを保持し、実際の1＋8件の成功を結果報告で区別した。数値JSONの書換えや再実行はしていない。synthetic predicate／codecとcompile-only検査を、本物の承認済みLinux Train contextの実行証拠へ読み替えない。

Linuxでの正式なcontextを使うpositive integrationは未計測である。production runnerはこの接続後もMissingExecutionDependencyを返す。残りのdescriptor／route、実資源等のgate、target／reference要求、durable admission／start、科学結果publisherと全実験への接続は未完了で、Task54／M4／M5全体の合格ではない。HE／学習／GPU計測はこの検証では行っていない。全登録範囲と既存hold、FULL-2の判定条件を維持する。

<a id="latest-evidence-20261107-1851"></a>

## 2026-11-07 18:52 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1851/README.md)では、18:52:11 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は390/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。18:51:10以降の予定枠から1回観測し、同じcontroller handle66436で終端まで回収した。再試行やジョブ・キューの変更はしていない。最終精度・Task42／M4の判定は引き続き未確定である。次の通常確認は19:52:12 JST以降とする。

<a id="latest-evidence-20261107-d7-descriptors"></a>

## 2026-11-07追記: 実行環境descriptorの実装と限定検証

[実装・検証証跡](../results/validation/task54-d7-runtime-descriptors/README.md)を保存した。確定済みのphysical／runtime-policyの形式を既存のdocument primitivesと上限で実装し、保持したCPU／CUDAの実情報を借用するfact viewと、CaseProvenanceの直後にphysical descriptorを生成するproduction経路を接続した。同じHost・実executor worker数・context／CUDA再検証を維持する。

compile REDから実装し、最終no-run／strict Clippy／逐次CPU・CUDA feature check／formatと独立source reviewが合格した。Macでは新規6件と既存Case／Reference互換性3件、計9件の限定テストが全件合格し、controllerもexit0で終了した。独立した4つの完全なバイト列・長さ・hashと一致し、切断／余分なbytes／domain／tag／上限／UTF8、実行条件や物理identity・source hashの取り違えを拒否した。CUDAのordinal・minorが0の場合や非UTF8 module pathも形式として検証した。入力はsyntheticであり、GPUの実測ではない。

選択したテストではdataset生成・学習・HE・実機／native moduleの観測・executor／thread／subprocess生成は行っていない。既存goldenのうち2件だけが同梱fixtureを読んだ。正式なLinux contextを使うphysical captureのpositive integrationは未計測で、Linux限定のhost／thread検証もMacではcompile対象外・未実行である。

runtime-policyのcodecと実情報viewは実装したが、具体的な学習経路の所有権を保証するadapterとproduction runtime-policy生成は未接続である。descriptorのbytes／hashや登録値だけから実行可としない。残りのExecutionBinding、live gate、durable admission／start、科学結果出力と全実験runnerが必要で、Task54／M4／M5全体は未完了。全登録範囲・既存hold・FULL-2の判定条件を維持する。

<a id="latest-evidence-20261107-d7-execution-design"></a>

## 2026-11-07追記: ExecutionBindingと実行開始後の所有権設計

[設計](study-execution-design.md)と[独立レビュー・完全なテスト値](../results/validation/task54-d7-execution-design/README.md)を確定した。実パラメータ、物理配置、鍵の生成・使用方針、秘密鍵分布の宣言を別々のdescriptorへ結び、既存P1／P3／physical／runtimeのbytesを維持する。新しい4形式の14正例・56負例は実装から独立したliteralで、レビューで内容・順序・分岐を確認した。syntheticな形式例であり、実パラメータや安全性の証拠ではない。

実行開始後は所有するdriverを返し、借用するP4／P5 sessionはその固定処理内で生成・破棄する。durable start後・鍵生成前のplan再検証、同じcontext／入力／計画の保持、暗号オブジェクト破棄までの資源lease保持を必須とした。初回レビューの重要2件を修正し、再レビューのSPEC／QUALITYが合格した。設計のみの確定で、4形式の実装、実経路からのExecutionBinding生成、live gate、durable admission／start、科学結果出力は未完了。全登録範囲・既存hold・FULL-2判定条件を維持する。

<a id="latest-evidence-20261107-1952"></a>

## 2026-11-07 19:53 JST追記: FULL-2の定期確認

[読み取り専用の運用記録](../results/quadruped/raw/R-41-FULL2-OBS-20261107-1952/README.md)では、19:53:01 JSTに同一所有者・開始時刻のchain／queueが稼働中と確認した。FULL-2は416/1000更新、末尾6更新は各1369 key switches／4 bootstrapsで予測と一致した。最終結果JSONはまだなく、後続queueはowner.jsonのみ。19:52:12以降の予定枠から1回観測し、同じcontroller handle88946で終端まで回収した。再試行やジョブ・キューの変更はしていない。最終精度・Task42／M4の判定は引き続き未確定である。次の通常確認は20:53:02 JST以降とする。

<a id="latest-evidence-20261107-d7-execution-descriptors"></a>

## 2026-11-07追記: ExecutionBindingの内部descriptor実装と限定検証

[実装・検証証跡](../results/validation/task54-d7-execution-descriptors/README.md)を保存した。対話パラメータ・物理配置・鍵方針・秘密鍵分布の宣言の4形式を実装し、canonicalなchecked preparationから同じplan／parameter／P1／geometryを借用するviewと、現在のTrain receiptのsource manifest参照を追加した。P3配置のエンコード本体を共有し、既存P1／P3／physical／runtimeの形式を維持する。

tests-firstのcompile REDから実装し、no-run／strict Clippy／逐次CPU・CUDA feature check／formatが合格した。初回レビューで負例テストが一部のparseエラーを見逃す点を指摘され、30形式エラー・9パラメータ条件違反・17形式上有効な不一致例を明示的に区別するよう修正した。元テスト・報告・失敗ログを保持し、sourceと修正の独立レビューが合格した。

Macで10件の限定テストを1回実行し、全件合格、全Cargoとcontrollerがexit0で終了した。独立した14の完全なbytes／hash、56負例、6,314の途中切断、6段のP3配置bytes、宣言weightの範囲と同一性の違いを検査した。実行証跡の独立レビューも合格した。syntheticな形式検証であり、パラメータ生成・planner・dataset・HE・GPU／native実行は行っていない。Macの時間を性能比較に使用しない。

実際のcanonical sourceを用いるpositive integrationは未計測で、runnerのMissingExecutionDependencyは維持している。残りは保持したCPU経路、実ExecutionBinding／runtime-policy生成、追加payloadの一括計上、live gate、durable admission／startと科学結果出力である。この限定合格をTask54／M4／M5全体、セキュリティ、学習精度の合格とはしない。全登録範囲・既存hold・FULL-2判定条件を維持する。

<a id="latest-evidence-20261107-d7-preparation-design"></a>

## 2026-11-07追記: CPU準備処理の所有権とメモリ計上設計

[設計の補足](study-execution-design.md#5-retained-preparation-accounting-refinement)と[独立レビュー記録](../results/validation/task54-d7-preparation-design/README.md)を実装前に確定した。既存callbackからchecked配置が外へ出ると計上用ownerが先に破棄される点と、入力・パラメータのlifetimeを一つにした型の不整合を修正した。新しいD7経路は配置・入力参照・元の計上ownerをまとめて保持する。既存互換経路の旧P3範囲と区別し、将来の固定CPU処理では暗号化後のinitial state／block binding／sessionが配置を持つ間も同じ計上ownerを保持する。修正後の独立SPEC／QUALITYレビューが合格した。

追加payloadは元のplan料金と合算し、元の上限を変えず、実際の一時bufferの重なりと保持分を数える。descriptorやbindingのサイズは出力を先に作らず共通の符号化処理から求める。既存P2生成、別管理の入力・パラメータ、library sort内部scratch等の対象外を明記し、全準備メモリやRSSを制限したとは主張しない。固定最大値による代用や、配置を複製してサイズを調べる案は採用しなかった。

これは設計の限定合格である。canonicalなI／NI・1〜3段の6ケースは検証計画であり、まだ実行結果ではない。保持準備の実装、実学習経路・gate・durable start・科学結果出力と全実験は未完了。Task54／M4／M5全体の合格ではなく、既存hold・全登録範囲・FULL-2の判定条件を維持する。
