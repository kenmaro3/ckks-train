# Task 42 確定前の検証手順

2026-11-06。M3 報告の確定作業用。これは測定結果でも Task 42 の完了報告でもない。

本書は集計の検証手順であり、研究計画・本人の指示・M3 の裁定を置き換える仕様ではない。比 (d) の定義は M3 §2 の 2026-11-06 裁定（`ba54f9d`）に採用されたもの。今後矛盾が見つかった場合は記録し、M3 確定時に計画と裁定へ整合させる。

## 入力と完了条件

`results/quadruped/requests.md` の R-41-FULL-1 / FULL-2 が終了し、runner の終了コードと、バイナリが出した元の JSON を回収してから実施する。予測、途中ログ、別の走行を完了 JSON の代わりにしない。元 JSON は変更しない。

| 行 | 入力 | 学習サンプル延べ数 | 更新数 | 精度の扱い |
|---|---:|---:|---:|---|
| FULL-1 | 196、WN、B=64 | 48000 | 750 | B=64 の精度を独立に記載 |
| FULL-2 | 196、WN、B=48 in 64 lanes | 48000 | 1000 | ReBoot 公開コード seed 0 の 0.9041 と比較 |

両行とも初期重みは ReBoot seed 0。JSON の `training.seed` は暗号化などの乱数 seed であり、初期重みの seed と混同しない。公開コードの参照は `results/quadruped/raw/reboot_emlp1_full/config_values.json` の `per_seed` にある値を使う。データと初期重みの manifest、lr schedule、インデックス列も照合する。

現行コマンドは 10 epoch 分のインデックス列を平坦化して `--epochs 1` で渡す。このため `epoch_measured` を通常の 4800 サンプルの 1 epoch 時間として引用しない。実測の総学習時間と、総学習時間を 10 で割った「10 epoch の平均」を区別する。

## 4 通りの比

実測総 KS を K、実測 bootstrap 回数を H、延べサンプル数を S とする。hekit の bootstrap は実測 111 KS/回、ReBoot はモデル 502 KS/回、ReBoot の非 bootstrap 部分は 196 入力ではモデル 37 KS/サンプルを使う。ReBoot 側は常に自身の B=48、4 bootstrap/更新。

| 比 | 計算 |
|---|---|
| (a) 各自の bootstrap | `(K / S) / (37 + 4*502/48)` |
| (b) 両側 502 | `((K - 111*H + 502*H) / S) / (37 + 4*502/48)` |
| (c) 両側 111 | `(K / S) / (37 + 4*111/48)` |
| (d) 両側 bootstrap 除外 | `((K - 111*H) / S) / 37` |

(b)〜(d) は実測カウンタからの派生計算であり、別のブートストラップ実装を実行した時間・精度の測定ではない。(d) は非対話実行の bootstrap 分を除いた値で、compact 再配置等の費用を含む。M2 の対話版や、その閉形式の値をこの列に代入しない。既存 M3 表の「対話版」の参考値は、別の行として出所を維持する。

`tools/rq1_summary.py` は元 JSON を読み、出所・完了条件・カウンタ整合性を検証して Markdown を生成する。欠測を数値で埋めない。出力にある入力 SHA-256 を保存し、最終表の元ファイルを特定できるようにする。

```sh
python3 tools/rq1_summary.py \
  --full-1 results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json \
  --full-2 results/quadruped/raw/R-41-FULL-2/mnist196_ni_b48_full_seed0.json \
  --reboot-config results/quadruped/raw/reboot_emlp1_full/config_values.json
```

runner が別の場所に元 JSON を保存した場合は、その実在するパスを指定する。provenance の記載と SHA-256 は出所の追跡に用いるもので、内容の真正性を暗号学的に保証するものではない。runner のビルド・終了記録との照合は別途必要である。

## 合否の分離

- KS の基準は比 ≤0.70。ReBoot 側の 37 と 502 は常に「model」と明記する。
- 精度の基準は FULL-2 と公開コード seed 0 の差が ±0.5 percentage point 以内。FULL-1 の精度差を FULL-2 の精度で置き換えない。
- 784 subset の同一 CPU 比較と 196 全長の時間を別表にする。196 全長の ReBoot 実測がなければ、その条件の wall-clock は未計測。47.6 h 等の外挿を実測の分母にしない。
- セキュリティは R-REB-4 の実パラメータと同一推定器の結果を参照し、BDGL16 / MATZOV の 128 bit ゲートと、秘密鍵分布の差を記載する。推定途中や未達を合格にしない。
- 精度と費用を異なる B で示す採用済みの報告方針は、その対応を明記する。異なる構成の合格項目を集めて「同一構成で全条件合格」とは書かない。

## 確定版で更新する場所

`docs/m3-report.md` の §0、§1.1–1.2、§2、§3、§5 の前提 3、§6.2 を更新する。§8 の裁定と §9 の未決事項を残し、`docs/handover.md` に進捗を追記する。Task 42 のレビューと commit の後、ユーザー指示どおり `git push origin main` を行う。

主設定 B、「同一 N」、著者欄の決定待ちは作業を止める条件にしない。ただし測定の欠落と未達は、未決事項とは別に報告する。
