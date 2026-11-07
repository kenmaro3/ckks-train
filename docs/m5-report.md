# M5 報告（進行中、2026-11-07）

最新の追記: [FULL-2の09:47確認と残る実装](#latest-evidence-20261107-0947)。

**研究全体は未完了。** M3 のFULL-2精度判定と全長比較、M4 の修正後 GPU 再計測・訓練、M5 の再現スクリプトの実機通し試験が残る。実装済み、測定済み、研究ゲート合格を分けて記録する。論文は WAHC 2027 向けのレビュー済みドラフトで、投稿・arXiv 公開・crates.io 公開・リポジトリの公開化は行っていない。

この報告は Task 57 の現状整理である。後続の測定を「完了」と見なす代わりにはならない。数値の一次資料と条件は [M3](m3-report.md)、[M4](m4-report.md)、[再現手順](../results/README.md)、[セキュリティ推定](security.md)を参照する。

## 1. 実装・測定・判定

| Task | 実装とレビュー | 測定・検証 | 現時点の判定 / 残り |
|---|---|---|---|
| 42: M3 / RQ1 | 草稿、4 比率の検証付き集計器、ReBoot 安全性追補をレビュー済み | 784 入力 subset は測定済み。196入力FULL-1は回収・4比率合格。FULL-2は未回収 | **未確定**。FULL-2 と公開コード seed 0 の 0.9041 を ±0.5 pt で比較。FULL-1 の B = 64 精度は別に示す |
| 47: CUDA 演算 | プールとバッチ処理の修正をレビュー済み | R-47-2 回収済み、R-47-3 は10:49観測でqueued/waiting・未計測 | 修正前のバッチ時間を修正後の性能と呼ばない |
| 48: GPU 訓練 / RQ3 | provenance 修正を含めレビュー済み | 小さい CPU smoke の計数のみ確認。R-48-1〜4 は10:49観測でqueued/waiting・未計測 | GPU 訓練の速さ・精度・実ピークメモリは未判定 |
| 49: GPU bootstrap（stretch） | `f64a617`、独立レビュー済み | CPU 全スロット精度・計数と CUDA コンパイルを確認。GPU 実機は未実行 | **コードのみ**。220 ms 目標・GPU 精度・ビット一致・メモリは未計測 |
| 50: M4 報告 | `docs/m4-report.md` の報告準備をレビュー済み | 測定済みと欠測を分離 | **報告あり、M4 合格未判定** |
| 51: データセット | `09f16f2` / `8bd9cba`、レビュー済み | eMLP-1 平文 10 seeds | MNIST / Fashion-MNIST の ±1 pt 目標は合格。Breast Cancer は固定 epoch で未達 |
| 52: eMLP-2/3 | `b8e267c` / `91efa6e` / `9305d2e`、修正レビュー済み | ReBoot から値として取得した構成と seed-0 平文軌跡を照合。小さい暗号化 hand-off の試験済み | 本番パラメータの暗号化 subset R-52-1/2 は10:49観測でqueued/waiting・未計測。全長の暗号化学習は未実行 |
| 53: 4 次活性化（stretch） | 保留 | 未計測 | 計画の削減候補。精度改善を主張しない |
| 54: 実験・表生成 | `7752459`、引継ぎ修正と独立再レビュー済み | 合計49件の Python 回帰テスト、表・図の freshness、shell syntax が合格。`--tables-only` で再生成可能 | **縮小版の再現入口のみ実装・レビュー済み。計画の全実験行列は実装未完**。拡張設計中。R-54-1 は未投入で post-full-v1 対象外 |
| 55: v1.0 準備 | README・Cargo metadata・notices を整備、実 checkout の crate rename を適用 | 755ファイルのバイト不変、改名後462 Rust tests合格（12 ignored）、49 Python tests、CUDA compile、Clippy、rustdoc、deny、4パッケージのローカル連鎖ビルド合格 | **ローカル公開準備済み**。改名SHA09202cfのLinux CI全6ジョブ成功。queue SHA4b1c499のLinux CIも全6ジョブsuccess（後続SHAには継承しない）。公開操作は未実施 |
| 56: 論文 | `e2d27b6`、全7節・付録・参考文献・自作図、独立再レビュー済み | 初版は両版9ページ。FULL-1反映後に両版10ページで再ビルド成功。匿名版全10ページ・仮著者版先頭を表示確認し、両版の最終ログにoverflow・未定義参照・欠落文字なし | **ドラフトあり**。研究結果の欠測を明記。投稿・公開は未実施 |
| 57: M5 報告 | `fd49eac` の現状報告はレビュー済み | 後続変更を再照合中 | 論文・表・rename の検証結果を照合した現状報告。研究全体は未完了 |
| 58: PyO3（最後、削減可） | 今回の範囲から削減 | 未検証 | Rust API と再現手順を優先。Python binding は提供しない |

## 2. 研究ゲートと主張の範囲

### RQ1: 全長評価とセキュリティ同条件は未確定

M3 の subset では B = 64 の鍵切替と同一サーバ CPU 時間の基準を満たす。一方、これは 196 入力の全長結果ではない。FULL-1/2 についてはバイナリが書いた元 JSON の provenance、モデル、入力数、更新数、初期値・学習スケジュール、計数の整合性を `tools/rq1_summary.py` で検証してから採用する。比 (a)〜(d) を併記し、ReBoot の 502 KS/bootstrap はモデル値、ckks-train の 111 は実測として区別する。(d) は非対話測定から bootstrap を引いた値であり、独立した対話版の実測ではない。

ReBoot 公開設定を同じ推定モデルに入れると、非 MITM ハイブリッドでも 128 bit を下回る。MITM を含む最小演算量はさらに低いが、巨大なメモリ要件を伴う推定であり、実用攻撃の実証ではない。原因は取得した構成と OpenFHE の表による水準検査の扱いに帰属させる。**同じ N であっても、同じ推定で双方 128 bit 以上という条件は満たしていない。** FULL-2 の精度が合格しても、この条件が解消したことにはならない。[M3 §3.3a](m3-report.md#33a-reboot-の実パラメータと追加探索2026-11-06-更新)と [security.md](security.md)に探索・モデル・メモリ・1 鍵で観測した h の限界を残す。

### RQ2 / RQ3 と GPU

WN / AU / AA の比較は M3 に測定済みの範囲を記載した。AA の bootstrap 入力が想定範囲を超える観測や、ブロック hand-off の誤差も残す。WN が eMLP-1 で良いことから、すべての深いモデルで最適とは主張しない。

RQ3 は CPU smoke の演算数だけでは GPU の寄与を決められない。小さい非安全なプリセット・高い Mac 負荷下の smoke 時間を性能比較に使わない。サーバ側の同じ optimizer・同じレベルの CPU/GPU 行を比較し、NAG が GPU メモリに収まらず SGD に移る場合は CPU の対照も SGD に合わせる。メモリ見積もりを実測ピークと呼ばず、スキップを成功件数に含めない。四つの ablation の解釈上の注意は [M4 §2](m4-report.md)を継承する。

GPU 演算は実機で測定済みだが、GPU 訓練と GPU bootstrap は別のゲートである。訓練経路は現時点で CPU の HostBootstrapper を使う。Task 49 の device bootstrap API があることを、GPU 常駐の全学習を測定した証拠にしない。HEonGPU の公表値との比較も、残存レベル・素数本数・鍵分布等が異なる名目比較である。

### データセットと既存マイルストーンの未達

- Task 51 の固定 epoch の平文精度は、MNIST と Fashion-MNIST が計画の ±1 pt 内、Breast Cancer は未達。テスト集合で選んだ最良 epoch を主結果に取り替えない。Breast Cancer の min-max 化は z-score 入力での訓練発散を根拠に決めた変更で、論文と同一の前処理を確認したという意味ではない。
- eMLP-2/3 は local block と個別 head に加え、独立した output stack を持つ。ReBoot のソースを取り込まず、外部実行から取得した設定値・初期重み・軌跡で照合した。平文軌跡の一致を本番の暗号化学習精度の証拠にしない。
- M1 の seal_2020 では、鍵切替系の性能目標に未達が残る。SEAL と異なる分解設定の n16 比較を同一アルゴリズムの速度比と呼ばない。
- M0 の CPU エミュレーション、理論コスト、旧時点の OOM は、その時点の記録として残す。後続の実機結果と混ぜない。M2 の対話版はクライアントの復号・再暗号化を要し、非対話の RQ1 合格の代用にはならない。

## 3. 再現と未実行の範囲

元の計測 JSON を不変の入力とする。Mac と quadruped-gpu の時間は分け、単一 seed に標準偏差 0 を付けず、未計測とする統計量を明示する。runner のメタデータは、バイナリが書いた provenance の代わりにはならない。古い結果に provenance を後付けしない。`d611c9d` で Iris・対話 MNIST・ReBoot-layout の新規出力にバイナリ自身の provenance を追加し、平文の `--plan` が学習・ファイル書き込みへ進まない回帰テストを通した。既存の元結果は変更していない。

現時点の表生成は `tools/paper_tables.py`、実験入口は `scripts/run_all.sh`。Task 54 の実装は、長い暗号化実験を 1 seed、平文を複数 seed にした縮小行列である。研究計画の全組合せ・10 seeds が完走したとは記載しない。未対応・資源不足・入力欠落・保留を coverage として残し、公式 ReBoot の実行は外部環境と requests の手順に従う。実機の `--quick` が完了するまでは「2 時間以内」は目標・見積もりである。

旧 runner の09:30停止記録（`0c819d4`）では後続は未投入だったが、2026-11-06 10:49:44 JSTの起動後観測でpost-full-v1 PID1739838 / start ticks467041669はwaiting、owner.jsonのみ、clones_created=falseだった。R-47-3 → R-48-1〜4 → R-52-1/2（各21更新subset）は両FULL成功terminalとchain10 terminal/非稼働identityの検証待ちとして投入済み。ビルド・測定は未開始で、R54は未投入。計画の全実験行列は実装未完で拡張設計中。実stateと完全commit・SHA256は [運用記録](post-full-queue.md) を参照する。この10:49記録時点で確認済みだったFULL進捗は10:21:08のupdate113（0始まり、114/750）。11:09の新しい実観測は下の追記に分け、観測間の更新数を推定しない。別テナントを停止せず、GPU に入らない行は未計測と記す。鍵認証、nice 10、24 スレッド以内、長時間処理のバックグラウンド化を維持する。SSH のタイムアウトはジョブの完了・失敗を意味しない。新たなクラウド GPU は使わない。

## 4. v1.0 の残作業

名称は **ckks-train**、リポジトリは `kenmaro3/ckks-train` で確定している。名称の再承認は不要。`scripts/rename.sh` で実 checkout の5 crate を `ckks-train-*` に変更した。改名後のビルド・テスト・rustdoc・ライセンス検査・生成手順と配布パッケージの検証が合格した。歴史的な結果 JSON と外部ベースラインを改名の対象にしない。

改名前の Task 55 報告ではローカルの workspace テスト、fmt、Clippy、rustdoc、cargo-deny が通過した。旧名 `hekit-core` の publish dry run は成功し、後続 crate の dry run は未公開の内部依存を crates.io が解決できず失敗した。ローカル directory source を使ったパッケージ連鎖のビルド確認は別に成功している。**実際の crates.io publish 成功を意味しない。** 詳細は [release.md](release.md)。

改名後は462 Rust testsが合格（12 ignored）、49 Python testsと表・図の freshness も合格。CUDA はコンパイル検証であり、実機計測ではない。4パッケージの隔離環境での連鎖ビルドと CUDA feature のビルドが通った。改名後の core の registry dry run も成功したが、残り3つは未公開内部依存のため失敗する。README の最終修正はローカルの梱包・ビルドで再検証し、registry dry run を再実行したとは扱わない。詳細は release.md に区別して記す。

GitHub CI は実際に起動している。`0daf778` の [CI run](https://github.com/kenmaro3/ckks-train/actions/runs/42136309953) は fmt / clippy / cuda-check / test / deny の全5ジョブが成功した。この結果は `0daf778` の検証であり、後続の provenance 修正・表生成・crate rename を検証したものではない。改名SHA `09202cf5650aa76b9b30dd79a68dd615c75a91ae` の [Linux CI run42143028484](https://github.com/kenmaro3/ckks-train/actions/runs/42143028484) は全6ジョブsuccess。後続queue SHA `4b1c499` の [run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) は10:49記録時点ではpendingだったが、その後のroot確認でcompleted/success・全6ジョブsuccess（`task-post-full-ci.json`）。この成功は4b1c499だけの検証で、後続SHAへ継承しない。このCIはGPU実機・研究測定ではない。

公開前には最終 SHA の CI、配布パッケージの内容・notices、履歴内の機密や外部コード混入、README の状態、CHANGELOG の日付を確認する。依存の一般許可リストは MIT / Apache-2.0 / BSD / ISC。`unicode-ident =1.0.27` の Unicode 文字属性表だけは、本人が 2026-11-06 に承認した Unicode-3.0 例外を設け、通知を付ける。別の crate やバージョンへの一般許可ではない。データセットのライセンスと帰属は `THIRD_PARTY_NOTICES` に別途記す。公開先を変更したり、公開化・tag/release・crates.io publish を実行したりした事実はない。

## 5. 論文・arXiv チェックリスト

| 項目 | 状態 / 完了に必要なもの |
|---|---|
| 原稿 | `paper/` にレビュー済みドラフト。7節と付録、2020年手法からの差分、RQ1〜RQ3、Limitations を含む |
| 数値・表・図 | 表は generator の出力を input、図も元結果から生成。欠測を埋める手入力をしない。source trace を維持 |
| 著者・所属・連絡先 | 仮置き。著者順・所属・連絡先・貢献の確認は本人の未決事項 |
| 匿名版 | 匿名 acmart sigconf と、仮置き著者を表示する版を分ける。投稿前に自己参照と artifact URL も匿名性を確認 |
| ページ制限 | 前年 WAHC 2026 の公式 CFP は参考文献・付録込み 12 ページ。2027 の規則は未確認。前年規則を暫定上限とし、2027 CFP 公開後に再確認 |
| build / 表示 | `latexmk -pdf`、ページ数、未解決引用・参照、はみ出し、全ページの図表の可読性を確認。最終表反映後、両版9ページで build・表示確認と独立レビュー済み |
| 参考文献 | 原論文・公式資料で書誌を確認。ベースラインのソース抜粋を取り込まない |
| 図の権利 | 自作図と結果から生成した図を使い、外部図の転載を前提にしない。データ・文献の帰属を付記 |
| arXiv ライセンス | ソフトウェアの MIT と原稿のライセンスは別。本人が選ぶ。著者全員の公開同意、投稿アカウント・endorsement が必要なら確認 |
| 提出物 | TeX・BibTeX・生成図表を含む再現可能なソース一式。ローカル TeX 配布物、機密、データセット本体を混入させない |
| 投稿・公開 | 未実施。ドラフト作成は投稿済みを意味しない |

ページ制限の一次資料: [WAHC 2026 Call for Papers](https://homomorphicencryption.org/wahc-2026-14th-workshop-on-encrypted-computing-applied-homomorphic-cryptography/)。2027 の締切や上限が確定したとは扱わない。

Task 58 は計画と本人の指示で削減可能な最後の項目であるため、今回の公開準備では削減する。Rust API・論文の再現手順・未完の研究ゲートを優先する判断であり、代替は研究評価の確定後に PyO3 を別途追加すること。Python binding の実装・検証済みという主張はしない。

## 6. 本人の未決事項（作業は継続）

1. 見出しの主設定 B と「同一 N」の解釈。FULL-1/2 と四つの比を残したまま、結果に即して判断できるようにする。
2. ReBoot 公開構成の比較を安全性差付きで報告するか、別途 128 bit を満たす構成で再推定・再測定するか。後者は公開設定そのものとの比較ではなくなる。
3. eMLP-2/3 全長暗号化実験の実施範囲。現在は subset を登録済みで、全長時間は見積もりにとどまる。利用中の共有サーバの実測を根拠に判断する。
4. 著者・所属・順序、arXiv ライセンス、公開時期・公開先。名称 ckks-train は決定済み。
5. stretch の 4 次活性化・PyO3 を採用するか。必須報告・再現・公開準備を優先し、削減時は明記する。


## 7. 後続観測（2026-11-06 11:09:14 JST）

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

<a id="latest-evidence-20261106-2321"></a>

## Task54 parser・Linux構造検証の確定とFULL観測更新（2026-11-06）

コード統合点は private main にpush済みの [`8d3a649`](https://github.com/kenmaro3/ckks-train/commit/8d3a649db4b84e35f24eb598429782278b4c248c)。[参照結果のparser](study-reference-document-design.md)を実装し、独立レビューと指摘1件の修正・再レビューを完了した。不正なフィールド名が大きな文字列の例外扱いを得ないよう、object keyとarray要素の構造を診断表示から分離した。修正後はparser20件、正規パス2件、既存canonical golden1件、再現入力5件の拒否、Clippy・formatを確認した。初回の全21 profiles×10 seeds・既存入力ビット一致等の検証は元の証拠として保持し、修正後に全件を再実行したとは扱わない。読み込んだ値は不変の未検証claimであり、全長参照結果・実行証明・target lockには変換しない。

[`23ac79d`](https://github.com/kenmaro3/ckks-train/commit/23ac79d5332086e6237f9d6b548d084417a5c376)のLinux拡張は、実際のGNU x86_64 Linuxで **test / releaseの各profileが13 library +9 integration tests合格**。65,535個のroot検査、実際のTARGET/HOST、compilerと実行環境を記録し、証跡も独立レビュー後に [`f9a8eed`](https://github.com/kenmaro3/ckks-train/commit/f9a8eedf63c55567c80dc81883ee9e7d0dbe9980) で保存した。[元ログとCI provenance](../results/validation/bootstrap-linux-20261106/README.md)を参照。rustc1.85.1/LLVM19.1.7、Ubuntu24.04.5/glibc2.39の記録された環境について、条件付きの構造metadata検証を満たした。encoding・EvalMod内部scale・暗号化した値のrange/precision・実際の鍵・security・memory・本番server admissionは引き続き別の未達項目で、将来のserver実行ファイルも自身のprocessで検査する。小規模runtime fixtureとn16のmetadata検証を、本番N65536の学習・GPU性能測定として扱わない。

[参照session・provenance・保存の契約](study-reference-session-design.md)は独立レビュー後に確定した。Macでの実際の2更新診断と、Linuxでの全長参照結果を生成する能力を分け、前者から本番参照やtargetを作れない設計である。この契約の実装・全長参照結果・target lockは未完。packed NI/AA planner・runnerも実装中であり、主学習の暗号化数値検証はまだ実行していない。既存の84/84 calibration合格は固定corpusの証拠として保持し、主学習や全行runnerの合格に読み替えない。**Task54の全実験行列とTask42/M4の最終判定は未完**である。

サーバーの **2026-11-06 23:21:25.182956 JST**（Unix1793974885.182956）の鍵認証・読み取り観測では、FULL-1はupdate602（0始まり）= **603/750更新完了**、elapsed57771s。直近3更新は予測計数と一致し、FULL-1のpost終端と両FULLのpartialマーカーはない。FULL-2には進捗・post終端がない。chain PID1700979/start ticks465769819とqueue PID1739838/start ticks467041669は同じidentityでstate S、owner.jsonのみ・後続clone未作成。R47/R48/R52は両FULL成功とchain終了検証を待ち、R54と全長暗号化blockは未投入。ジョブ変更・再起動・新しいFULL結果JSONはない。Task42の四つの比、FULL-2対ReBoot seed0=0.9041の±0.5pt判定は結果待ち。一次記録はledger `post-full-observation-20261106-232047.stdout` と対応meta/stderrである。

CIの **23:20:49.439744 JST** 一覧観測では `090f8fb` / `02a26a5` / `9610822` / `878438c` / `edeffa7` はcompleted/success。`23ac79d` [run42230241832](https://github.com/kenmaro3/ckks-train/actions/runs/42230241832) と `8d3a649` [run42231800043](https://github.com/kenmaro3/ckks-train/actions/runs/42231800043) はin_progressだった。前者のfocused Linux stepsの合格と、workspace全体・workflow全体の完了は区別する。一次記録はledger `task-54-ci-observation-progress5.json`。後続の報告・設計commitへCI成功を継承させない。

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

## 2026-11-07追記: 実行主体の契約と取得時の不一致

[StudyTrainの実行・build契約](study-runtime-design.md)を独立レビュー後に確定した。Referenceのscalar1・既存形式を保持し、学習実行のbuildと参照artifactのbuildを別々に検証する。CPU直列、専用Rayon pool、CUDA＋host poolの固定recipe、approval-v2、読み取り／生成権限、実際のworker数の取得方法を定めた。理由・代替案・費用は同報告に記載。実装、独立期待値、実runtime、D7のdescriptor／gateは未完であり、この設計は実行許可やTask54の合格ではない。

Target lockのコードは`66a4b36`で統合済みで、そのcommitのCI全6項目の成功を確認した。これは実際の承認済みLinux context・全長FP32参照・保存／復旧の実運用検証を代替しない。Packed blockの更新・forward-only APIは実装中で、今回の実HE主検証は未実行。

入力exporterは`f021616`までの修正を独立レビュー後に統合した。Macの実compatibility検証はeMLP2 seed0の画像payloadが保存済みcanonical dataと一致せず停止した。同seedの重みと順序はbyte一致、ラベル・設定・scheduleに不一致は報告されていないが、他seed／eMLP3の比較まで到達していない。許容誤差で通さず、失敗・別診断出力を保持する。Seed3–9取得、入力version移行とMissingInput解消は未達。

[R-REB-6-OBS](../results/quadruped/raw/R-REB-6-OBS/README.md)で元のLinux環境を読み取り専用で観測した。現在の前処理Python source hashはMac側と一致し、Torch／torchvisionはLinux用buildだった。過去のnative実体や不一致原因を証明する観測ではない。04:32:19 JSTのFULL-2ログはupdate28（0始まり）=29/1000更新完了、直近4更新の計数は予測と一致。誤った推測pathの2項目はrawを変更せずREADMEで証拠から除外した。最終精度・現在のprocess liveness・終了はこの観測から主張しない。既存FULL／post-full queueを変更せず、追加したのは記録済みの軽量・未計時の読み取り観測のみ。

## 2026-11-07追記: ケース期待値とLinux静的前提

全2,680 family tuples、別枠20 Iris、全7補足記述子、19 canonical vectorsを独立照合し、case projection用期待値のSPEC/QUALITYレビューが合格した。Rust projectionは未実装、whole-study denominatorは未確定のまま。Packed-block実装の独立レビューで失敗時の応答・途中観測の保持と構造テストに不足が見つかり修正中。新しい実HEのI/NI主検証は未実行。

[R-REB-7-STATIC](../results/quadruped/raw/R-REB-7-STATIC/README.md)でLinuxのNumPy sourceと正しいconfigパス、実行ファイルのsymlink先、`.pth`の存在を観測した。Macと同じSVE照会sourceで、現行Darwin専用の拒否処理はLinuxに適用されない。動的import到達性・startupコード・完全なcontext/native/isolationは未検証。04:58:18 JSTのFULL-2は40/1000更新で直近6計数は予測一致、chain/queueの同一PID/start/自UID/state Sを確認した。これは最終精度や終了の証拠ではなく、後続キューは変更していない。

本人の容量整理依頼により、完了済みセキュリティ見積もり用の非稼働Sage環境を削除し、394 packageの再インストール情報を保持した。追加4,441,407,488 bytes、累計17,264,930,816 bytesの独立割当量を整理。見積もり結果・原ログ・コード・データ・現在使用中の環境は残した。以前の「Sage環境保持」は過去時点の記録であり、再計算には保存したpinsからの再インストールが必要。

## 2026-11-07追記: 取得経路とCIの制限

[R-REB-8](../results/quadruped/raw/R-REB-8-ARCHIVE/README.md)は元configと固定済みarchiveの差をstructure_summaryだけと確認した。共有F64値はbit一致だが、byte pinは異なるため元出力をそのまま比較archiveとして受理しない。今後の新checkoutで既存pinnedコピーを完全検証して使い、pin・元出力は保持する。[R-REB-9](../results/quadruped/raw/R-REB-9-BOOTSTRAP-STATIC/README.md)では既存venvの起動時importとsource guardの衝突が静的に予測された。新しい起動経路は設計中で、環境作成・実import・compatibilityは未実行。

`c6ab260`と`2434c4b`のCIは6ジョブすべてstepなしで終了し、test注釈はアカウントの支払いまたは利用上限による起動拒否を示した。この2commitのCIは未実行であり、ローカル検証や前commitの成功で代替したとは扱わない。請求・上限を変更せず、実装と独立レビューは継続する。

## 2026-11-07追記: block正しさと登録行の実装

`ad607da`のpacked-block compositionは独立再レビューを経て統合した。[I/NIの実検証証跡](../results/validation/packed-blocks-n1024-20261107/README.md)ではIが2 graphs／8更新／2推論／116 client refresh、NIがG2/G3×AU/WN/AAの6構成／24更新／6推論／332 bootstrapsで各1回・exit0。全710／4,050個の512-complex-slot snapshotを独立再計算し、最大スカラー誤差は3.775018180620039e-9／3.441401240719656e-8（各上限2^-12）。NI全332回の最大増分誤差1.8763991003465872e-8は2^-20以内で、固定range、24更新の7計数、各bootstrap区間も合格した。

これはMac・N1024・key seed167の小規模な正しさの合格。実験精度・速度・GPU・本番security・full block campaignの合格ではない。原stdout/stderrと外部runner記録を保持し、gzip化したstdoutは展開hashで同一性を確認。監査JSONは明示した事後派生物で、科学結果行にしない。復号bit列の独立計算と、sourceに依存するseed／generation／暗号文由来を区別し、未出力の推論／phase総計数を最後のeventカウンタで代用しない。

`cf3f822`のcase projectionは独立SPEC/QUALITYレビューと44件の限定検証・Clippy・formatが合格。全2,680 family cases、別枠20 prospective Iris、7補足記述子、19独立canonical vectorsを照合し、MissingInputや未実装executorによって登録行を削除しない。全長dataset materializationの既存1テストは明示的に除外した。これは完全なmembershipの実装で、実行準備・admission・科学結果の証拠ではない。2,700をwhole-study denominatorにしない。

[R-REB-10](../results/quadruped/raw/R-REB-10-CREATOR-STATIC/README.md)でbase creatorの実行ファイルとhook／site／venvを静的に観測した。新環境作成や科学用importは未実行。学習role/runtime、Linux起動／context、全長参照・target、実行handoff、baseline adapter、全行runnerとTask42/M4の最終測定は残る。これらを本追記の限定合格で置き換えない。

## 2026-11-07追記: Linux source対応と取得準備

`010bb3c`の限定Linux optional-query対応は独立レビューと全exporter43テストに合格した。実際のsource/code/platformを結び付けたv2 EPERM拒否でchild実行を許可せず、Darwin条件・科学的値・完全一致ゲートを保持する。Macのfictional検証であり、実Linuxの依存import／数値互換性は未計測。

[R-REB-11](../results/quadruped/raw/R-REB-11-CREATE-EMPTY-VENV/README.md)は06:08:36 JSTに新しい空のstdlib venvを作成して静的検査だけを行い、exit0・警告なし・15-entry layout／空のown site／pinned-base linksを確認した。異常時回収と検査順の2指摘は実行前に修正・8テスト・独立再レビュー済み。新venvの起動、context、科学用import、入力取得は未実行で、作成成功をその代替にしない。原出力と独立監査を保存した。

05:58:57 JSTのFULL-2は66/1000更新、直近6計数は予測一致。chain/queueは同一PID/start/自UIDでlive、最終結果は未出力、queueはowner.jsonのみ。ジョブ変更はなく、RQ1/M4の最終判定は未確定。Task54の学習role／実行接続と全行runnerも未完である。

## 2026-11-07追記: 起動準備の実測と未達

[R14/R15](../results/quadruped/raw/R-REB-14-15-STARTUP-PREREQUISITES/README.md)の静的観測をレビュー・保存し、[R16〜19](../results/quadruped/raw/R-REB-16-19-STARTUP1/README.md)で固定コピーの配置と独立readbackは合格した。一方、実startup1は静的preflight後にimporter-cacheの固定launcherキーを拒否してexit1となり未達。親子の原記録・終了・完全なstream回収を保持し、helper preparation、外部package追加、科学用import/context/captureには進んでいない。失敗したLinux記録からfinderの値を補完しない。修正と別の実行releaseが必要で、環境作成やsourceレビューの合格を起動成功に置き換えない。

07:25:40 JSTのFULL-2は102/1000更新、既存chain/queueは同じstart ticksでlive、最終結果なし。学習role/buildの限定Rust35・integration1・Python32 methodsとClippy/formatは通ったが、独立レビューのPython emitted-path containment指摘を修正中で未統合。native/context/runtime・全行runner・科学的matrix・Task42/M4/M5全体は未完である。既存holdと全登録範囲を保持し、公開・課金設定は変更していない。

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

## 2026-11-07追記: runtime実装前の具体化

[Runtime設計 §11](study-runtime-design.md#11-concrete-ownership-and-bounded-module-reads-2026-11-07)に、単一core内での実行lease保持、CUDA観測adapterを先行する実装順、native moduleを1GiB上限・1MiB chunkで読む方針と代替案を記録した。自己参照や実行権限の複製を避け、既存SHA-256のprivate streaming化でファイル全体のメモリ複製を避ける。いずれも実装前の選択で、実runtime・CUDA実機・Train統合検証や研究ゲートの合格ではない。既存の科学的設定・artifact上限・全登録範囲は変更しない。

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

本人の未決事項: 主設定B、比較における「同一N」の扱い、著者欄（現状は仮置き）は未決のまま残す。これらの回答待ちで実装を止めず、現在の登録条件と既定の比較区分を維持する。
