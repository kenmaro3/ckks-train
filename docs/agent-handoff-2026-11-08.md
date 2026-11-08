# エージェント切替用レビュー・再開計画 — 2026-11-08

**再開後の更新:** 本人の再開依頼を受け、固定e93cec4の全6 CI成功と配布物を再照合し、
preview 2を公開した。[公開・匿名取得の検証](../results/validation/research-preview2-publication/README.md)。
この版の証拠選択はFULL-2/R47までで確定。arXivは本人担当、Task54の未commit実装は維持。
以下は22:31時点の引き継ぎ原本であり、未公開・CI実行中の記述は過去の状態である。

この文書を次のエージェントの入口とする。基準時刻は **2026-11-08 22:31 JST**、レビュー対象の確定ソースは `e93cec434652a8c9f8c48becc6eb3045a3f78a7f`。後から変わるCI・作業tree・サーバー状態は再確認する。従来の [handover.md](handover.md) と実装ledgerは詳細履歴として残す。

## 1. 最新の本人指示と今回の停止位置

- 目的は **MITでOSS公開し、論文を出すこと**。完成度の最後の10〜20%を待つより、約1日で実用的な初回公開を優先する。未達の研究基準を緩めたり、未計測を合格と書く許可ではない。
- **arXiv投稿は本人が行う**と、このセッションで明確に回答済み。エージェントはログイン、ライセンス受諾、投稿ボタン操作、endorserへの連絡をしない。本人用のZIP・PDF・投稿情報まで用意する。
- その後、本人がエージェント切替を希望し、「ここまでのレビュー」「今後の計画」「しっかりしたドキュメンテーション」を依頼した。これを受けて公開操作を止め、引き継ぎに切り替えた。
- このセッションでは **新しいtag/releaseの作成、asset upload、push、arXiv操作、サーバーへのSSH・実験追加・再起動を実施していない**。
- `docs/publication-schedule.md` の旧48時間計画は履歴。現在は上記のスピード優先と本人投稿の分担を優先する。初回MIT公開は既に達成しており、全研究計画の完了を初回公開の条件に追加しない。

## 2. すぐ分かる現在地

| 対象 | 現在地 | 確認範囲 |
|---|---|---|
| 公開リポジトリ | [kenmaro3/ckks-train](https://github.com/kenmaro3/ckks-train)、PUBLIC、MIT | 今回 `gh repo view` で確認 |
| 公開リリース | [v1.0.0-preview.1](https://github.com/kenmaro3/ckks-train/releases/tag/v1.0.0-preview.1)、固定 `6fba46c` | 今回release一覧とremote tagを確認。既存公開検証は別記録 |
| main | local / originとも `e93cec4` | 今回 `git log` / `git ls-remote` で照合 |
| 公開添付論文 | FULL-1中心の旧10ページ版 | 現在のmainにある論文とは内容が異なる |
| 新しい論文 | FULL-2 / R-47-3を反映した10ページ、著者入りPDFと19ファイルのZIP | 以前のビルド・目視記録あり。今回18論文入力、19 ZIP memberと出力hashを再照合 |
| 次の配布候補 | 固定 `e93cec4` からローカル作成 | 6,113 source fileのGit blob・mode一致。**未公開・配布manifest未作成** |
| 最新CI | `e93cec4` / [run42484163158](https://github.com/kenmaro3/ckks-train/actions/runs/42484163158) | 22:31時点で5ジョブ成功、workspace test実行中。全体成功ではない |
| 未コミット作業 | 50 tracked modified + 41 untracked files（自分の新しい引継ぎ記録を除く） | ファイル展開後の数。`git status`のdirectory表示件数と違う |
| arXiv | **未投稿、本人が担当** | アカウント・endorsement・arXiv上の組版は未確認 |

引き継ぎ文書だけのローカルcommitをこの後作成する場合、local HEADは上記基準SHAの後続になる。配布candidateは引き続きe93cec4で固定し、文書保存のためにcandidateを作り直さない。今回の引き継ぎcommitはpushしない。

`v1.0.0-preview.2` のremote tagは今回の確認で存在しなかった。次の公開名の候補として使用しただけで、予約・作成していない。既存preview.1のassetを上書きしない。

## 3. 研究内容のレビュー結論

現在の最も説明しやすい成果は、既存CKKS手法を学習向けに組み合わせた **diagonal packing / feature-major mini-batch / BSGS / compact optimizer stateの実装と、数え方を明示した条件付き比較**。非対話学習そのものの発明、同一安全性での優越、GPU学習高速化を主張する段階ではない。

| 項目 | 採用済み結果と限界 | 一次入口 |
|---|---|---|
| FULL-1 | 196入力、B=64、初期重みseed0、全長完了。4種類のKS比基準を満たす。FULL-2とは別バッチ | [M3](m3-report.md) |
| FULL-2 | B=48、精度0.9035 対 ReBoot公開コードseed0=0.9041、差−0.06 percentage pointで登録許容内。KS比(a/c/d)達成、(b)=0.781062で0.70以下の基準を未達 | [Task42最終化](../results/validation/task42-full2-finalization-20261108/README.md) |
| 総合RQ1 | 共通安全性条件未達、全長ReBoot CPU時間未計測。FULL-1の計数合格とFULL-2の精度合格を合算して全条件合格にしない | [M3](m3-report.md) |
| R-47-3 | level5の独立4積microbenchmark、3 warmup・30反復。GPU学習の速度ではなく、反復は独立seedでもない。テスト45 ok中3容量SKIP、実行42 | [R47採用](../results/validation/task47-r47-3-adoption-20261108/README.md) |
| GPU | operator一致には実機証拠あり。GPU学習・ablation終端回収、実機bootstrap、peak memoryは未完または未測定 | [M4](m4-report.md) |
| Task54 | 保存済み結果からの表生成は動作。全実験行列、production start/実環境gate/CPU・GPU接続は未完 | [M5](m5-report.md) |

今回のレビューは **公開状態・論文入力と配布物の一致・残作業の整合性のレビュー**。暗号実装全行の監査や、新しい実験の再実行ではない。既存の独立レビューを参照したが、このセッションで独立レビュー担当を新たに起動していない。

## 4. 今回実施した検証と検出した注意点

### 実施済み

1. GitHub repositoryのPUBLIC、既存preview.1、mainとtagのSHAを照合。
2. 固定HEADを独立clone `/private/tmp/ckks-train-preview2-e93cec4` にcheckout。共有mainのWIPを取り込んでいない。
3. そのcloneで `scripts/run_all.sh --tables-only`、`tools/tables.py --check`、`tools/figs.py --check` がすべてexit0。実行後 `git status --porcelain` は空。**保存結果からの再生成**であり、学習の再実行ではない。
4. source tar.gzを読み戻し、全6,113ファイルの内容をGit blob、modeをGit tracked modeに照合。
5. 論文ビルド記録の18入力が固定HEADのbytesと同じこと、既存PDF・ZIPのSHA256、ZIPの全19 memberのサイズ・SHA256を照合。`pdfinfo`で10ページを確認。
6. source archiveの非圧縮trackedファイルを限定的な既知資格情報パターンで検査し、該当0。任意secret検出や圧縮ファイル内部・全履歴の再監査ではない。公開済み履歴の以前の検査は [preview記録](research-preview.md) を参照。
7. 今回確認時点の `git diff --check` は成功。新しいRust全体テスト・Clippy・fmt・Linux native試験は実施していない。

### 次に誤解しやすい点

- READMEの概要にはFULL-2完了が反映されている一方、末尾StatusのM3には `assessment awaits FULL-2` が残る。**説明の内部不一致**。次の公開文書更新で修正する。研究結果そのものの不一致ではない。
- 既存 `paper/README.md`、release文書は公開preview.1とローカル更新を区別している。新candidateを公開したら冒頭リンク・現在状態だけを更新し、過去の失敗・旧SHAの検証を消さない。
- `paper/arxiv-metadata.md` のアカウント確認保留という説明は今回の「投稿は本人」に更新する。タイトル・abstractは更新論文と合っているが、最終版の締切・採用結果が変われば再照合する。
- 最新CIの成功をまだ宣言できない。直前 `465b32f` の全6ジョブ成功を `e93cec4` やWIPへ転用しない。
- 50 tracked変更にはbenchだけでなくcore/ckks/nnの資源量観測helperも含まれる。`Vec::capacity()`等による論理payload量とRSS/peak memoryは別物。今回その一部差分を読んだが、全差分の品質承認はしていない。
- `prepare.py` は準備directoryが既存なら失敗する。これは上書き防止。既存候補を削除して再実行せず、再作成が必要なら別directory・別記録にする。

## 5. 作業中コードの扱い

元の実装ledgerは `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/progress.md`。これはローカルの詳細記録であり、GitHub cloneだけで存在すると仮定しない。

最後の記録では、Task54 Step3のsole writerがoriginal acquisition/error workspaceと読取・資源管理の接続を継続していた。**このセッションのagent一覧にはrootしか見えず、別スレッドの担当agentの現在の生存・停止は確認できない**。過去のagent名、PID、tool session IDを現在有効とみなして再起動・pollしない。今回Cargo/rustc/latexmkプロセス名の確認時には一致なしだったが、別担当の今後の書込停止を保証しない。

- `task-54-step3-u5-source-acceptance.md` は、U5のretained/prospective sizingとF1の限定修正を採用した記録。
- F1はcounter拒否経路のBoxを除去し、private stage表現を変えて診断とpublic Option viewを保つもの。記録上の10 testsと特定 `result_large_err` lintは成功。ただし188個の別warningが残り、strict Clippy全体成功とは違う。
- 13ファイルのfreezeはその採用後に解除済み。**今あるbytes全体をその過去レビューが保証するわけではない**。
- full Step3/4、positive kind7 dispatcher、U6環境capture、genuine gates、durable admission/start、recovery、native execution、科学的出力は未完。
- 現在のdiffを丸ごとcommitして「Step3完了」にしない。元担当が存在する場合はファイル所有・作業区切りを確認し、source pinsを固定してから残経路をレビューする。
- `.vscode/` と2件の `results/mnist_blocks_*_plain_b48.json` もuntracked。今回の配布には含めていない。由来・採用が未確認の結果をgenerated tableに取り込まない。

今回のファイル別snapshotは [worktree-inventory.json](../results/validation/research-preview2-e93cec4/handoff-snapshot/worktree-inventory.json)、status/diff statは同じdirectory。変更trackedファイルのbinary diffはMicroSDの `/Volumes/Untitled/ckks-train-storage/archives/research-preview2-wip-handoff.patch`。このpatchは **untracked内容を含まない**。元の作業treeを削除・resetしてよいバックアップではない。snapshotと現在のhashが違えば、後続編集として扱う。

## 6. 実験キュー：次に回収するもの

このセッションではSSHしていない。以下はリポジトリ内の **既存観測**。

- 22:06 JST: queue PID1739838/start467041669、child2393682/start488014488が生存、R48/R52の回収に必要な終端は未確認。
- 22:12 JST: [bounded observation](../results/quadruped/raw/R-48-PROGRESS-20261108-2210/README.md)でR-48-4 CPU ablation logの末尾12行を確認。R52の2つのlogは不在。部分logは計時結果や科学的合格に使わない。
- 22:45以降の次回observerは **準備済み・このセッションでは未実行**。登録は `results/quadruped/requests.md` の `R-41-FULL2-OBS-20261108-2245`。既に他担当が使っていないか必ず先に確認する。
- ledgerにある `post-full-observation-20261108-2245-controller.py` / command JSON / observerのpinとfresh output条件を確認し、登録済みのbounded手順で観測する。時刻到来だけで終端になったと判断しない。
- [collector準備記録](../results/validation/postqueue-terminal-collection-preparation-20261108/README.md) と [回収map](post-full-queue.md) が手順の入口。R48/R52 collectorは、実際のterminal observationをレビューするまで起動しない。
- 回収は `task-48-terminal-controller.py OBSERVATION_PREFIX` / `task-52-terminal-controller.py OBSERVATION_PREFIX`。`OBSERVATION_PREFIX`は実観測に置き換え、placeholderのまま実行しない。fresh-only、key-only SSH、1回、元byte保存、timeout時に既存jobをkillしない等の既存条件を保つ。
- R48はn16_train_level7の実行、batchごとのOpCounter一致、最終weight/bias誤差≤2^-12、8 flag rows、optimizer/level/layout/B/計時範囲を照合。Cargo exit0だけでは容量SKIPを実行済みにできない。
- R52は21更新のsubset。失敗exit前にJSONを書くproducerがあるため、JSONの存在やaccuracy値だけで成功扱いしない。
- 新結果は終端原本→runner/queue/provenance→科学的基準→表生成→論文の順に取り込む。締切までに揃わなければ未計測行のまま公開し、後で更新する。キュー再起動や重複投入は不要。

## 7. 配布物の場所と再開コマンド

準備directory:

`/Volumes/Untitled/ckks-train-storage/archives/research-preview2-e93cec4/`

| ファイル | 状態 |
|---|---|
| `ckks-train-e93cec4-source.tar.gz` | 固定HEAD、全6,113ファイル照合済み |
| `source-inventory.json` | 各ファイルのGit blob / mode / SHA256 |
| `ckks-train-paper-preview.pdf` | 532,092 bytes、10ページ、既存レビュー済みPDFと同一 |
| `ckks-train-arxiv-source.zip` | 40,079 bytes、19 member、既存isolated build済みZIPと同一 |
| `PAPER_LICENSE.txt` | 論文用の別license notice |
| `SOFTWARE_LICENSE.txt` | MIT本文 |

PDF SHA256: `9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c`

ZIP SHA256: `2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2`

今回の検証・準備script: [results/validation/research-preview2-e93cec4](../results/validation/research-preview2-e93cec4/README.md)。**release notes、全asset manifest、SHA256SUMS、upload後の照合はまだ未実施**。

```sh
cd /Users/marconi/Documents/research/hekit
git status --short
git log -3 --oneline
gh run view 42484163158 --repo kenmaro3/ckks-train \
  --json status,conclusion,headSha,jobs
gh release list --repo kenmaro3/ckks-train --limit 5
git ls-remote origin refs/heads/main refs/tags/v1.0.0-preview.2
python3 tools/release/_ext/storage_root.py archives
```

固定候補の再生成を確認する場合は独立cloneで行う。すでに同一SHAで成功済みなので、入力やコードが変わらなければ同じチェックを繰り返す必要はない。

```sh
cd /private/tmp/ckks-train-preview2-e93cec4
git rev-parse HEAD
bash scripts/run_all.sh --tables-only
python3 tools/tables.py --check
python3 tools/figs.py --check
git status --porcelain
```

このcloneは `--shared` で元repositoryのobject storeを参照する。独立バックアップではない。旧 `/private/tmp/claude-501/wt55` worktreeは今回作成したものではなく、勝手に削除しない。

MicroSDは `tools/release/_ext/storage_root.py` でUUIDと実mountを検査する。PDF/ZIP/大きなarchiveは既存方針どおり同カードに置き、不在なら内部SSDへ黙ってfallbackしない。

## 8. 次の担当の優先順と完了条件

### P0：再開後まず15分

- 本文の最新本人指示、Git status、origin、candidate、CIを確認。新agentの最初の仕事を全Rust WIPの完成にしない。
- 別担当が作業を続けているか確認できる範囲で確認し、source所有を確定。新規workerを暗黙に増やさない。
- 本人のarXiv投稿分担を維持。現時点で提出可能なPDF・ZIPの場所をすぐ案内できるようにする。

### P1：再開後1〜2時間を目安に公開候補を確定

1. `e93cec4` CIが全6成功ならその証拠を保存。失敗ならjob/step/logを読み、インフラ失敗とコード失敗を区別する。以前の6910b14はcargo-deny actionのダウンロード展開に失敗した記録があり、そのSHAの失敗を消さない。
2. R48/R52が終端ならbounded回収して採否を決める。時間がかかる場合は現在の論文で凍結。全21-profile/10-seedやGPU bootstrap完了を待たない。
3. 固定candidateをそのまま使うか、新しい採用済み結果を含むSHAに更新するかを一度決める。更新するならsource・CI・論文inputの対応を取り直す。
4. release notesにFULL-2の精度合格、(b)未達、共通安全性未達、GPU訓練等の未測定を明記。別licenseの論文、MITソフトウェア、研究previewであることを説明。
5. 全assetのmanifest/SHA256SUMSを作成し、実ファイルを読み戻して照合。旧preview.1は保存する。

### P2：公開と本人への引き渡し

- 本人の公開意図はあるが、**今回の最後の指示は引き継ぎで区切ること**。この文書作成中には実行していない。次の担当は再開依頼の範囲でcandidateを公開する。
- 新tagは固定SHAへ向け、GitHub release/PDF/source/ZIPを公開。asset digest/size、tag SHA、匿名HTTP取得、downloaded PDF hashを確認してから「公開済み」とする。
- README、paper README、release現状のリンクを実在する新releaseへ更新。安定版v1.0・研究計画の全達成は別判断。
- 本人には **ZIP、PDF、[投稿metadata](../paper/arxiv-metadata.md)** を渡す。著者Kentaro Mihara単著、Independent Researcher、contact `miha.ken.19@gmail.com`、論文licenseはarXiv non-exclusive distribution。タイトル・abstract・page countを最終artifactと照合する。
- category案はcs.CR、cross-list案cs.LG。投稿・endorsement・arXiv生成PDFの最終確認は本人担当。arXiv IDや投稿完了を推定しない。

### P3：初回公開後

Task54 full Step3/4、native gates、全実験行列、GPU学習/bootstrapping、複数seed、共通安全性比較、全長baseline CPU時間を残課題として進める。元の研究gateを変更する場合は別の研究判断として記録する。公開できたことと、仮説がすべて立証できたことを分ける。

## 9. 次エージェントへの開始メッセージ

> `/Users/marconi/Documents/research/hekit` の `docs/agent-handoff-2026-11-08.md` を最初に読んでください。MITコードとpreview.1は公開済み、最新FULL-2/R47論文の配布candidateはe93cec4でローカル準備済み・未公開です。まずCIと作業treeの最新状態を確認し、1日以内の公開優先で候補のmanifest・公開・検証を仕上げてください。arXiv投稿は本人が行います。Task54の50 tracked変更等を無断でまとめてcommitせず、旧reviewを現在の全WIPへ適用しないでください。R48/R52は終端確認後にだけ回収し、未計測が残っても論文の主張を絞って公開を進めてください。
