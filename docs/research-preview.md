# Research preview preparation — 2026-11-08

**公開準備中。公開済み・安定版v1.0・研究計画完了ではない。**
速度を優先する希望に合わせ、未完の研究全体とは別に、固定したコードと既存の論文ドラフトを確認できる候補を準備する。元のTask42/50/54/57と全実験条件は残す。公開操作はまだ行っていない。

## 初回候補（全体テスト不合格、公開対象として未認定）

- Source: `90851d261b36b034d3b660ec4c95dd330eadc616`。mainの作業中のTask54変更を含めない。
- Source archive: `ckks-train-90851d2-source.tar.gz`、56,209,245 bytes、SHA256 `276e0e08b923e1a9eea285d434511cc854e3f6e41d1934b387f4a7444d57489c`。
- Gitにある全5,071ファイルの内容とmodeをarchiveから読み戻して照合した。ソースarchiveにはGitの履歴・metadataを含めないため、provenanceを必要とするコマンドには同じcommitのGit checkoutを使う。
- PDF: 既存のFULL-1反映版、匿名10ページ、529,509 bytes、SHA256 `57f0eb179a44ad8b9b0224c717bfa664ea8ddbc73bee2811d8e47ee1ebb99c75`。元の `tools/latex/_ext/task56/full1-20261107/anonymous/main.pdf` と同一bytesであり、今回の候補から再ビルドしたという意味ではない。

ローカル一式は `tools/release/_ext/research-preview-bundle-90851d2/` に置く。`PREVIEW.md`、`manifest.json`、`source-inventory.json`と上記2ファイルを含む。大きな配布archiveをGitへ追加せず、元ソースとhashを区別する。

## この候補で確認したこと

| 検証 | 状態・範囲 |
|---|---|
| Python回帰 | 115件合格。保存済み入力・合成fixtureによる検証で、実学習ではない |
| 表の再生成 | `scripts/run_all.sh --tables-only` が成功。生成後も候補checkoutのtracked変更なし |
| 表・図の一致 | `tools/tables.py --check`、`tools/figs.py --check` 成功 |
| Shell構文 | run_allとrenameを別々に `bash -n` で確認、両方成功 |
| Rust整形 | `cargo fmt --all --check` 成功 |
| 依存 | この候補の `cargo deny --offline --locked check licenses bans` 成功。承認済みUnicode表の例外は維持 |
| Rust全体テスト | **未達**。407.426秒でexit101。study_admissionの共有処理回数のassertionが1件失敗し、後続targetは未実行。Mac、2 build jobs / 2 test threads、incrementalなし。共有fixtureのカウンタ干渉を再現し、下記修正を採用。修正後の全体検証は別途必要 |
| 最終Clippy・rustdoc・配布package | この候補では未実施 |
| Linux CI | この候補の成功は未確認。直近の別SHAでは課金条件による未起動があり、過去の成功を継承しない |

検証は候補専用checkoutとbuild directoryで行い、作業中の実装・cacheと分ける。実機GPU評価やサーバー時間の代わりにはしない。

## 公開履歴の点検範囲

候補HEADから到達可能な5,298 blob、179,731,216 bytesを限定的な資格情報パターンで検査し、該当0件だった。private-key header、GitHub token、AWS access-key ID、Slack token、OpenAI project keyの既知形式が対象で、任意形式・entropy・全機密情報の検出を保証しない。

圧縮された3 blobも別途読み、gzipログとzip内ファイルの計114,648,570 bytesを同じパターンで検査し、該当0件だった。展開先への書込は行っていない。履歴に `_ext/` 配下のpathは0件だったが、それだけで外部コード混入がないと結論しない。5MBを超えるblobは4件あり、最大は32,158,222 bytesの既存の検証ログである。旧公開準備時の「5MB超なし」はこの候補には適用しない。

これは最終の公開履歴・ライセンス・文書内容のレビューを終えたという意味ではない。

## 初回公開と残る研究の区別

FULL-1の実測を保持する。FULL-2の精度、全長ReBoot CPU時間、同じ安全性条件での比較、GPU学習の評価、全実験行列の実行は未確定である。表が生成できることと全実験が再現できることを分ける。候補のversion fieldが1.0.0であっても、正式版の合格とは呼ばない。

最終候補の検証・公開資料のレビューを終え、公開対象が具体的に確認できる段階まで準備する。著者は2026-11-08にKentaro Mihara単著・個人研究として確定した。主設定B、「同一N」の扱いは未決のまま残す。これらの回答待ちで実装・計測を止めない。

## 全体テストで見つかった問題と修正

[元の失敗・切分け・修正・独立レビュー](../results/validation/research-preview-test-isolation/README.md)を保存した。4つの統合テストが共有する処理予算に、canonicalテストが自身の上限を累積値として適用していた。並列2スレッドで再現し、単独では合格した。

共有fixtureを使う4テストだけを同一lockで保護し、canonicalの元の判定値を自身の処理差分へ適用する。全体の148/148/8/0/8上限、他のテストの並列性、本番コードは変更していない。修正後の通常並列と2スレッド指定は各5/5合格し、仕様・品質の独立レビューも合格した。これは修正1件の検証であり、修正後の最終候補の全体合格ではない。初回archiveを合格版へ読み替えず、候補を更新して検証する。

## 修正候補の全体検証（2026-11-08）

固定`00f4409c585fe3395b93aabffdbe020053fb01e1`で全体テストがexit0、74 targetの集計は1,025合格・0失敗・19 ignoredだった。ignoredは未実行である。Mac、2 build jobs / 2 test threadsで731.299秒。Clippyは通常・CUDA有効の両方で警告をエラーにして合格した。

厳格rustdocでは`plain`の説明文の未解決リンク1件を検出した。公開関数への完全修飾リンクへ1行だけ修正し、独立レビュー合格、修正後の通常・CUDA有効のrustdocも合格した。本番コードの変更はない。修正前の失敗と、base commit＋修正diffを明記したGREENログを[検証記録](../results/validation/research-preview-fixed-verification/README.md)に保持した。進行中Task54の未commit実装を検証したものではない。

著者入りのarXiv用source previewも作成した。必要19ファイルをZIP化し、展開先だけから10ページのPDFを再ビルドできた。現時点の本文はFULL-1中間版で、arXiv側のビルド・投稿は未実施。[paperの手順](../paper/README.md)を参照。アカウント確認はユーザー自身が行う。

## ローカル配布一式 cf58897

`tools/release/_ext/research-preview-bundle-cf58897/`に、固定source archive・著者入りPDF・arXiv source ZIP・manifestを準備した。sourceは5,211ファイル、56,313,450 bytesで、全ファイルの内容をGit blob、modeをtracked modeに照合済み。古い不合格候補のarchiveとは別物である。

このcommitのclean checkoutでfmt、依存ライセンス、表の再生成、表・図一致、厳格rustdoc通常/CUDAが合格した。全体テストとClippyは前記00f4409の証跡を参照し、以降のコード差分が説明文リンク1行だけであることを確認した。[今回の記録と配布hash](../results/validation/research-preview-cf58897/README.md)を保持する。公開内容の最終レビューと研究の残ゲートは未完で、GitHub公開・arXiv投稿はまだ行っていない。
