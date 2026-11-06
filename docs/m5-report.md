# M5 報告（進行中、2026-11-06）

**研究全体は未完了。** M3 の全長 196 入力行、M4 の修正後 GPU 再計測・訓練、M5 の再現スクリプトの実機通し試験が残る。実装済み、測定済み、研究ゲート合格を分けて記録する。論文は WAHC 2027 向けのレビュー済みドラフトで、投稿・arXiv 公開・crates.io 公開・リポジトリの公開化は行っていない。

この報告は Task 57 の現状整理である。後続の測定を「完了」と見なす代わりにはならない。数値の一次資料と条件は [M3](m3-report.md)、[M4](m4-report.md)、[再現手順](../results/README.md)、[セキュリティ推定](security.md)を参照する。

## 1. 実装・測定・判定

| Task | 実装とレビュー | 測定・検証 | 現時点の判定 / 残り |
|---|---|---|---|
| 42: M3 / RQ1 | 草稿、4 比率の検証付き集計器、ReBoot 安全性追補をレビュー済み | 784 入力 subset は測定済み。196 入力 FULL-1/2 は未回収 | **未確定**。FULL-2 と公開コード seed 0 の 0.9041 を ±0.5 pt で比較。FULL-1 の B = 64 精度は別に示す |
| 47: CUDA 演算 | プールとバッチ処理の修正をレビュー済み | R-47-2 回収済み、R-47-3 は10:49観測でqueued/waiting・未計測 | 修正前のバッチ時間を修正後の性能と呼ばない |
| 48: GPU 訓練 / RQ3 | provenance 修正を含めレビュー済み | 小さい CPU smoke の計数のみ確認。R-48-1〜4 は10:49観測でqueued/waiting・未計測 | GPU 訓練の速さ・精度・実ピークメモリは未判定 |
| 49: GPU bootstrap（stretch） | `f64a617`、独立レビュー済み | CPU 全スロット精度・計数と CUDA コンパイルを確認。GPU 実機は未実行 | **コードのみ**。220 ms 目標・GPU 精度・ビット一致・メモリは未計測 |
| 50: M4 報告 | `docs/m4-report.md` の報告準備をレビュー済み | 測定済みと欠測を分離 | **報告あり、M4 合格未判定** |
| 51: データセット | `09f16f2` / `8bd9cba`、レビュー済み | eMLP-1 平文 10 seeds | MNIST / Fashion-MNIST の ±1 pt 目標は合格。Breast Cancer は固定 epoch で未達 |
| 52: eMLP-2/3 | `b8e267c` / `91efa6e` / `9305d2e`、修正レビュー済み | ReBoot から値として取得した構成と seed-0 平文軌跡を照合。小さい暗号化 hand-off の試験済み | 本番パラメータの暗号化 subset R-52-1/2 は10:49観測でqueued/waiting・未計測。全長の暗号化学習は未実行 |
| 53: 4 次活性化（stretch） | 保留 | 未計測 | 計画の削減候補。精度改善を主張しない |
| 54: 実験・表生成 | `7752459`、引継ぎ修正と独立再レビュー済み | 合計49件の Python 回帰テスト、表・図の freshness、shell syntax が合格。`--tables-only` で再生成可能 | **縮小版の再現入口のみ実装・レビュー済み。計画の全実験行列は実装未完**。拡張設計中。R-54-1 は未投入で post-full-v1 対象外 |
| 55: v1.0 準備 | README・Cargo metadata・notices を整備、実 checkout の crate rename を適用 | 755ファイルのバイト不変、改名後462 Rust tests合格（12 ignored）、49 Python tests、CUDA compile、Clippy、rustdoc、deny、4パッケージのローカル連鎖ビルド合格 | **ローカル公開準備済み**。改名SHA09202cfのLinux CI全6ジョブ成功。後続queue SHAのCIはpending。公開操作は未実施 |
| 56: 論文 | `e2d27b6`、全7節・付録・参考文献・自作図、独立再レビュー済み | 最終表を含む匿名版・著者仮置き版とも9ページ、全ページ表示と修正影響ページを確認 | **ドラフトあり**。研究結果の欠測を明記。投稿・公開は未実施 |
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

旧 runner の09:30停止記録（`0c819d4`）では後続は未投入だったが、2026-11-06 10:49:44 JSTの起動後観測でpost-full-v1 PID1739838 / start ticks467041669はwaiting、owner.jsonのみ、clones_created=falseだった。R-47-3 → R-48-1〜4 → R-52-1/2（各21更新subset）は両FULL成功terminalとchain10 terminal/非稼働identityの検証待ちとして投入済み。ビルド・測定は未開始で、R54は未投入。計画の全実験行列は実装未完で拡張設計中。実stateと完全commit・SHA256は [運用記録](post-full-queue.md) を参照する。FULLの最終実進捗は10:21:08のupdate113（0始まり、114/750）、後の観測から更新数を推定しない。別テナントを停止せず、GPU に入らない行は未計測と記す。鍵認証、nice 10、24 スレッド以内、長時間処理のバックグラウンド化を維持する。SSH のタイムアウトはジョブの完了・失敗を意味しない。新たなクラウド GPU は使わない。

## 4. v1.0 の残作業

名称は **ckks-train**、リポジトリは `kenmaro3/ckks-train` で確定している。名称の再承認は不要。`scripts/rename.sh` で実 checkout の5 crate を `ckks-train-*` に変更した。改名後のビルド・テスト・rustdoc・ライセンス検査・生成手順と配布パッケージの検証が合格した。歴史的な結果 JSON と外部ベースラインを改名の対象にしない。

改名前の Task 55 報告ではローカルの workspace テスト、fmt、Clippy、rustdoc、cargo-deny が通過した。旧名 `hekit-core` の publish dry run は成功し、後続 crate の dry run は未公開の内部依存を crates.io が解決できず失敗した。ローカル directory source を使ったパッケージ連鎖のビルド確認は別に成功している。**実際の crates.io publish 成功を意味しない。** 詳細は [release.md](release.md)。

改名後は462 Rust testsが合格（12 ignored）、49 Python testsと表・図の freshness も合格。CUDA はコンパイル検証であり、実機計測ではない。4パッケージの隔離環境での連鎖ビルドと CUDA feature のビルドが通った。改名後の core の registry dry run も成功したが、残り3つは未公開内部依存のため失敗する。README の最終修正はローカルの梱包・ビルドで再検証し、registry dry run を再実行したとは扱わない。詳細は release.md に区別して記す。

GitHub CI は実際に起動している。`0daf778` の [CI run](https://github.com/kenmaro3/ckks-train/actions/runs/42136309953) は fmt / clippy / cuda-check / test / deny の全5ジョブが成功した。この結果は `0daf778` の検証であり、後続の provenance 修正・表生成・crate rename を検証したものではない。改名SHA `09202cf5650aa76b9b30dd79a68dd615c75a91ae` の [Linux CI run42143028484](https://github.com/kenmaro3/ckks-train/actions/runs/42143028484) は全6ジョブsuccess。後続queue SHA `4b1c499` の [run42145577051](https://github.com/kenmaro3/ckks-train/actions/runs/42145577051) はこの記録時点でpending。このCIはGPU実機・研究測定ではない。

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
