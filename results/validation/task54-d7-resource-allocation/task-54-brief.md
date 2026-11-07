### Task 54: 全実験スクリプトと表の生成

**Files:**
- Create: `scripts/run_all.sh`, `tools/tables.py`, `results/README.md`

- [ ] **Step 1: `run_all.sh` が研究計画 §4.4 の全組合せ(データセット 4 × モデル × {CPU, GPU} × 対話/非対話、ベースライン a/b/c)を、シード 10 個で実行し `results/*.json` に出す。長い実験は `--quick`(subset、シード 1)で 2 時間以内に全体を通せるようにする。**
- [ ] **Step 2: `tables.py` が `results/` から論文の表(鍵切替、時間、精度、GPU メモリ、セキュリティ)を LaTeX と Markdown で生成する。平均 ± 標準偏差、シード数を列に入れる。**
- [ ] **Step 3: quadruped-gpu で `run_all.sh --quick` を通し、次に本番(ReBoot 公式のベースライン b は Task 5 の環境で、同一 CPU)を nohup で回す。所要時間と `df -h` を `results/README.md` に記録する。**
- [ ] **Step 4: Commit** `feat(bench): one-shot experiment runner and table generator`

