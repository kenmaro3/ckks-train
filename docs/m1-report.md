# M1 報告(2026-11-05)

M1(CKKS コア、Task 8〜24)の完了報告。正当性ゲート(KAT・精度)は合格。性能目標(SEAL 比)は seal_2020 の鍵切替系 3 演算で未達(1.17〜1.20 倍)。

## 1. SEAL 比(性能、目標 0.5〜1.0 倍)

**定義: SEAL 比 = hekit の時間 / SEAL の時間(小さいほど hekit が速い)。** 1.0 以下を合格とする。0.5 未満は「帯より速い」で、これも合格扱い。どちらも 1 スレッド(SEAL は内部スレッド無し、hekit は rayon 等を使っていないので常に 1 スレッド)。マルチスレッドの副次列は無い。

測定条件: Apple M3 Pro、macOS、rustc 1.85.1(`[profile.bench]` opt 3 + LTO)、SEAL 4.1.2 Release(-O3)。1 分平均 load ≤ 4 になってから開始した(load はベンチ自身の 1 スレッドを含む)。値は中央値。hekit は criterion 0.5 で 10 サンプル(flat sampling、warm-up 1 s)、SEAL は warm-up 1 回 + 10 回。

### seal_2020(N = 2^15、{60, 40×9, 60}、Δ = 2^40、レベル 9。hekit と SEAL で同一パラメータ)

load: SEAL 3.76 → 3.20、hekit 3.20 → 3.37。

| 演算 | hekit (ms) | SEAL (ms) | SEAL 比 | 判定 |
|---|---|---|---|---|
| encrypt(公開鍵) | 14.65 | 16.47 | **0.89** | 合格 |
| add | 0.213 | 0.383 | **0.56** | 合格 |
| mul + relin + rescale | 41.19 | 34.30 | **1.20** | **未達** |
| rotate(1) | 34.39 | 29.29 | **1.17** | **未達** |
| rotate × 16(通常、ステップ 1..16) | 550.9 | 469.1 | **1.17** | **未達** |
| hoisted_rotate × 16 vs SEAL の通常回転 16 回 | 221.2 | 469.1 | **0.47** | 合格(帯より速い。SEAL には公開の hoisting が無い) |
| matvec_plain(d = 64)vs SEAL 素朴対角法(63 回転 + 64 multiply_plain) | 437.8 | 1922.6 | **0.23** | 合格(帯より速い) |
| 同上 vs SEAL BSGS 8×8(14 回転、hoisting 無し) | 437.8 | 497.2 | **0.88** | 合格 |
| matvec_enc(d = 64)vs SEAL 素朴対角法(63 回転 + 64 multiply + 1 relin) | 790.9 | 2028.5 | **0.39** | 合格(帯より速い) |
| 同上 vs SEAL BSGS 8×8(14 回転 + 64 multiply + 8 relin) | 790.9 | 803.2 | **0.985** | 合格(余裕 1.5%) |

ルーリングで指定された matvec の比較相手は素朴対角法(63 回転)。同じアルゴリズム(BSGS)どうしの行も載せた。こちらの方が実装の速さの比較として公平で、差は hekit の baby 側 7 回転の hoisting から来ている。

### n16_train(N = 2^16、q = {58, 42×31}、レベル 31)

load: hekit 3.53 → 4.17、SEAL 4.17 → 3.29。

SEAL には複数素数の特別法が無い。そのため SEAL 側は `{58, 42×31, 58}` を使った。データ素数 Q は hekit と同じ 32 本で、特別素数は 1 本、鍵切替は素数ごとの分解(dnum 32、k 1)、sec_level none。一方 hekit は dnum 8、k 4 の hybrid 鍵切替を使う。**つまりこの表は同じアルゴリズムの比較ではなく、「n16_train の構成」対「SEAL ネイティブの構成」の比較である。** 回転の差の大部分は dnum の違い(分解する NTT の数が約 1/3)による。

| 演算 | hekit (ms) | SEAL (ms) | SEAL 比 |
|---|---|---|---|
| encrypt(公開鍵) | 105.2 | 115.9 | 0.91 |
| add | 1.85 | 2.87 | 0.65 |
| mul + relin + rescale | 302.7 | 542.7 | 0.56 |
| rotate(1) | 256.9 | 501.2 | 0.51 |
| rotate × 16 | 4121 | 8031 | 0.51 |
| hoisted_rotate × 16 vs SEAL の通常回転 16 回 | 1569 | 8031 | 0.20 |
| matvec_plain(d = 64)vs 素朴 / BSGS | 3141 | 32114 / 7573 | 0.10 / 0.41 |
| matvec_enc(d = 64)vs 素朴 / BSGS | 5711 | 33147 / 12098 | 0.17 / 0.47 |

SEAL の n16 では、鍵メモリを抑えるため Galois 鍵を 2 本(ステップ 1 と 8、1 本 1.1 GB)だけ作った。そのため rotate × 16 はステップ 1 を 16 回、素朴法はステップ 1 の累積回転、BSGS の giant 側はステップ 8 の Horner 法で計算している。回転のコストはステップに依存しない。SEAL の matvec は 4 通りとも復号して W x と照合した(最大誤差 2^-14.2 〜 2^-16.9)。

### 本タスクの性能変更とその効果(seal_2020)

| 演算 | 変更前(`--quick`、load ≈ 4.5) | 変更後(上の表) | 変更前の SEAL 比 |
|---|---|---|---|
| encrypt | 22.78 | 14.65 | 1.38 |
| mul + relin + rescale | 43.96 | 41.19 | 1.28 |
| rotate(1) | 36.60 | 34.39 | 1.25 |
| hoisted_rotate × 16 | 229.9 | 221.2 | 0.49 |
| matvec_enc(d = 64) | 837.8 | 790.9 | 1.04(BSGS 比) |

- NTT のループを書き換えた(2152d02): N = 2^16 で forward 386.4 µs、inverse 356.6 µs(load 3.4〜3.6)。SEAL の forward 373.7 µs(M0)の 1.03 倍。変更前は 435.5 µs(負荷下)。
- 離散ガウスを CDT 法に変えた(e2326a3): 1 係数あたりの乱数を約 10 回から 1 回に減らした。暗号化が 1.4 倍速くなった。
- 1 素数の桁の持ち上げを 1 パスにした(69225da)。
- 試して効果が無かったもの: BaseConverter のキャッシュ(構築コストは k^2·l 語演算で、N に比べて無視できる)、内積の係数ブロック化、別の 1 素数持ち上げ(いずれも測定誤差内か、遅くなった)。
- 回転の内訳(seal_2020): 分解 ≈ 23.5 ms(NTT 110 回 ≈ 20.4 ms)+ 内積と mod_down ≈ 10 ms。SEAL は同じ NTT 回数で約 29 ms なので、残る差は NTT 以外(変換・内積・メモリ確保)にある。

再現方法: `cargo bench -p hekit-bench --bench ckks_ops`(`HEKIT_BENCH_PRESET=seal_2020|n16_train` で絞れる。n16_train は約 10 GB のメモリを使う)、`tools/seal-kat/bench/run_bench.sh [seal_2020] [n16]`。NTT と hoisting の再測定は `docs/bench.md` にある。


## 2. KAT(SEAL 4.1.2 との既知解相互運用、正当性ゲート)

**結果: 合格(両方向)。** `tools/seal-kat/run_kat.sh`(2026-11-05、Task 24 の性能変更後のコード)の出力 `PASS: 0 failure(s)`。

- **SEAL → hekit**: SEAL が作った 8 個の暗号文(対称暗号 x, y、公開鍵暗号 x、add、multiply の 3 成分、relinearize、rescale_to_next、rotate_vector(1))について、hekit の復号は SEAL の復号と**ビット一致**した(hekit と SEAL の decode の差は 2^-inf。スケール 2^80 の 2 個だけは浮動小数点の decode の差で 2^-51)。hekit の add / mul_no_relin / rescale を SEAL の暗号文にかけた結果も、SEAL の add / multiply / rescale_to_next と**ビット一致**した。
- **hekit → SEAL**: hekit が作った 6 個(対称暗号 x のレベル 9 とレベル 3、SEAL の公開鍵での暗号、add、3 成分の積、SEAL の積を hekit で rescale したもの)を、SEAL がすべて読み込んで上限内で復号した。この方向はビット一致ではなく誤差上限だけで検査している(§5 の未達項目)。
- **誤差(最大スロット絶対誤差)と上限**: 対称暗号 / add / テンソル積 / relin は 2^-28.7 〜 2^-29.5(上限 2^-25)。公開鍵暗号は 2^-24.8 〜 2^-24.9、rescale は 2^-24.8(上限 2^-23)。SEAL の回転は 2^-19.1(上限 2^-15)。
- **上限がブリーフの一律 2^-25 でない理由**: 公開鍵暗号と rescale の誤差は丸め r_0 + r_1·s が支配する。SEAL の一様三値鍵では最大誤差の中央値が 2^-24.8 で、2^-25 だと SEAL 自身の復号が約半分の確率で落ちる。SEAL の回転は非中心化の素数ごとの分解なので、スロット 0 付近に約 2^-17.5 の鍵切替誤差が乗る。そこで主ゲートを「ビット一致 + SEAL 自身と同じ誤差」とし、2^-23 / 2^-15 は副次的な上限にした(ルーリング 9)。導出は `crates/hekit-ckks/tests/seal_kat.rs` の `bound_log2` にある。
- hekit の回転(厳密中心化の持ち上げ)は、設計上 SEAL の回転(非中心化)とビット一致しない。比べているのは復号結果だけ。

## 3. 精度(正当性ゲート)

**結果: 合格。** 閾値は seal_2020 ≥ 20 bit、n16_train ≥ 24 bit(`crates/hekit-ckks/tests/precision.rs`)。bits = −log2(全 N/2 スロットの最大絶対誤差)。入力は [−1, 1) の一様乱数、固定シード。Task 24 の CDT サンプラ導入後に再実行した値(`docs/precision.md`、カッコ内は Task 20 の値)。

| op | seal_2020(level) | n16_train(level) |
|---|---|---|
| encode/decode | 32.14 (9) | 33.70 (31) |
| encrypt | 24.81 (9) | 25.72 (31) |
| add | 24.32 (9) | 25.19 (31) |
| mul+rescale | 24.13 (8) | 25.38 (30) |
| rotate | 22.34 (9) | 25.23 (31) |
| matvec(16×16、平文対角) | 22.26 (8) | 24.71 (30) |
| matvec_enc(16×16、暗号対角) | 22.33 (8) | **24.47 (30)**(余裕 0.47 bit) |

公開鍵暗号の往復(Task 16)は seal_2020 で 2^-24 を上限にした(計画は 2^-25、ルーリング 5)。深さ 9 の連鎖は x ∈ [0.9, 1.025] で相対 2^-10 以内(ルーリング 8)。


## 4. セキュリティ(lattice-estimator、`docs/security.md`)

| preset | N | log PQ | BDGL16(= HE Standard の BKZ.sieve) | MATZOV(estimator の既定) | Core-SVP(ADPS16、2^0.292β) |
|---|---|---|---|---|---|
| seal_2020 | 2^15 | 480 | 252.9 | 245.6 | 217.8 |
| n16_train | 2^16 | 1592 | **143.9** | **141.2** | **107.7(< 128)** |

- 値は usvp / bdd / dual / dual_hybrid の最小(log2 rop)。秘密鍵は SparseTernary(n/4, n/4, n)、誤差は離散ガウス σ 3.2、m = n(m = ∞ でも同じ値)。
- **HE Standard v1.1 (2018)**: n = 32768 の 128 bit 上限は log Q 881(Table 1、BKZ.sieve、秘密鍵 (−1, 1))で、seal_2020 の 480 はその範囲内。N = 2^16 の表は無い。HE Standard の BKZ.sieve のコスト式は estimator の BDGL16 と同じで、その手法では n16_train は 143.9 bit。
- **Core-SVP では n16_train は 107.7 bit で 128 に届かない**(同じ β = 369 の primal 攻撃で、低次項を除いた値)。HE Standard は ADPS16 ではなく sieve のコストモデルを選んでいるので、プリセットはそのままにした。Core-SVP ≥ 128 を要求するなら log PQ を下げる必要がある(必要な値は計算していない)。
- **疎な秘密鍵(ハミング重み指定)の推定はまだ粗い**: h = 192 で rough モデルだけ(seal_2020 215.2、n16_train 107.2)。hybrid / MITM を含む完全な推定は数分では終わらなかった。M3 の疎なブートストラップ鍵の前に、完全な推定が必要(**M3 の前提条件**)。既定のプリセットはハミング重みを指定していない。



## 5. 未達項目(正直に)

**性能(目標であって正当性ゲートではない)**
- seal_2020 の mul+relin+rescale 1.20 倍、rotate 1.17 倍、rotate × 16 1.17 倍は、目標 1.0 以下に届かない。鍵切替 1 回が SEAL より約 15〜20% 遅い。NTT は SEAL の 1.03 倍まで詰めたので、残りは桁変換・内積・メモリ確保にある。安く安全に直せる範囲では、これ以上の改善は出なかった。
- matvec_enc を SEAL の BSGS と比べると 0.985 倍で、余裕は 1.5% しかない。
- 別の読み方: M0 報告は「SEAL 比性能 = SEAL の時間 / hekit の時間 が 0.5〜1.0」と読んでいた。この読み方なら seal_2020 の全演算が合格になる(最も遅い mul でも 0.83)。本報告はルーリングの定義(時間の比、1.0 以下)で判定している。
- n16_train の SEAL 比は、同じアルゴリズムの比較ではない(§1)。SEAL を n16_train と同じ 32+4 素数で動かすことはできなかった(SEAL の特別素数は最後の 1 本だけ)。
- hoisting の速度向上は 2.58 倍(rotate × 32 → hoisted 32、`docs/bench.md`)。目標の 2〜3 倍の範囲内だが、Task 18 の 2.74 倍からは下がった(分解が速くなり、ステップごとの部分が相対的に重くなったため)。
- マルチスレッドの列は無い(hekit は並列化していない)。

**計画からの逸脱(ルーリング済み)**
- Galois 生成元は 3^k(計画は 5^k)。SEAL 4.1 との相互運用のため。
- 公開鍵暗号の往復誤差の上限は 2^-24(計画は 2^-25)。秘密鍵暗号は 2^-25。
- KAT の副次的な上限は 2^-23(pk・rescale)と 2^-15(SEAL 回転)。ブリーフは一律 2^-25 だった。主ゲートはビット一致。
- 深さ 9 の連鎖テストの入力範囲は [0.9, 1.025](計画は [0.9, 1.1]。2^70 に溢れるため)。
- n16_train の L / k / dnum は 31 / 4 / 8(計画は 30 / 3 / 4)。log PQ は 1592。
- 素数の順序を修正した(SEAL の `pop_back` 順。Task 14 の判断は誤りで、KAT で発覚して ab53dfd で修正)。
- Task 21 の「matvec_enc の relins == 1」は「relins == giant」に置き換えた。
- 本タスクで離散ガウスを CDT 法に変えたので、乱数の消費の仕方が変わり、固定シードでも雑音の値が変わった(精度表は再実行した)。

**セキュリティの注意**
- n16_train は Core-SVP(ADPS16)で 107.7 bit で、128 bit 未満。BDGL16(= HE Standard)では 143.9 bit、MATZOV では 141.2 bit。
- 疎な秘密鍵の推定は rough だけ。完全な推定は **M3 の前提条件**。
- 2018 年の HE Standard の表は N = 2^16 を含まない。

**KAT の限界**
- hekit → SEAL の方向は誤差上限だけで検査していて、ビット一致は見ていない。
- hekit の回転は SEAL の回転とビット一致しない(中心化の持ち上げ。設計どおり)。
- KAT は seal_2020 だけ。n16_train の parms_id は Python の hashlib と照合して固定した(本タスクのテスト)が、SEAL 自体では動かしていない。

**知っておくべき先送りの軽微な指摘**(ledger より抜粋)
- サンプラは定数時間ではない(CDT も二分探索)。秘密鍵は `Debug` を伏せ字にしているが zeroize しない。`from_entropy` は panic する。`gaussian_poly` に σ の上限チェックが無い。
- NTT の x < q と、`QpPoly::new` の前提は `debug_assert` だけ。リリースビルドでは検査されない。
- automorphism の置換キャッシュ(`Mutex`)は上限が無い。`GaloisKeys::insert` は次数しか検査しない。
- エンコーダの Q/2 境界チェックは f64 の近似で、log Q > 1025 では inf になる(文書化されていない)。decode は O(N·L^2)。
- `CkksParams::new` は、独自パラメータで桁のビット数 ≤ log P かどうかを検査しない(プリセットは満たしている)。
- n16_train の精度テストは余裕 ≈ 0.5 bit(固定シード)。既定のテストスイートで約 19 s かかる。
- カウンタのテストに、matvec_plain の ct_pt_mults == d、共役、rescales == 0 のケースが無い。
- KAT: SEAL の minor version の検査が緩い。不正入力のテストは暗号文についてしか無い。

**本タスクで片付けた軽微な指摘**: `run_kat.sh` が環境変数で上書きできるパスを `rm -rf` しないようにした(既知の KAT ファイルだけ消す)。`primes.rs` の `skip` に関する古い doc を直した。n16_train の BLAKE2b parms_id を固定するテストを足した。rustdoc の警告(private item へのリンク)を直した。`docs/precision.md` の「約 4 bit」を「約 2 bit」に直した。


## 6. M1 中の判断(ルーリング)

ledger から、結果に効いたものだけを挙げる。

1. **n16_train は L 31(q 32 本: 58 + 31×42)、dnum 8(α 4)、k 4(特別素数 58×4)、log PQ 1592。** 計画の L 30 / k 3 / dnum 4 は、計画自身の割り切れ条件(dnum | L+1)に反する。さらに k 3 では P(174 bit)が桁(184 bit)より小さく、KS 雑音が約 2^160 倍になる。Task 22 で 128 bit(BDGL16 / MATZOV)を確認して確定した。
2. **素数の割り当ては SEAL の `CoeffModulus::Create` と同じにした。** ビット幅ごとに降順で生成し、`pop_back` で小さい方から順に渡す。Task 14 のルーリング(大きい方から渡す)は誤りで、Task 23 の KAT で発覚し ab53dfd で修正した(n16_train の順序も変わった)。KAT を正とする。
3. **NTT の根は SEAL と同じ**(最小の原始 2N 乗根、ビット反転順)。暗号文と鍵は NTT 形式で持つ。
4. **Galois の生成元は 3**(SEAL 4.1 の `GaloisTool::generator_`)。計画の 5^k ではない。実 SEAL で確認した。共役は 2N−1。
5. **公開鍵暗号の往復誤差の上限は 2^-24 とした(計画は 2^-25)。秘密鍵暗号は 2^-25 のまま。** Δ = 2^40、N = 2^15 では v·e + e0 + s·e1 の最大スロット誤差が本質的に ≈ 2^-25 になる(SEAL も同じ)。シードを選んで通すことはしない。
6. **桁の持ち上げは中心化した(Task 17)。α > 1 でも厳密に中心化する(Task 18)。** 非中心化だと seal_2020 の回転誤差が 2^-22.7 から 2^-19.8 に悪化する。厳密な中心化により、hoisted 回転と通常の回転の結果がビット一致する。
7. **mod_down の厳密丸め(7944f6f、依頼外の変更)を採用した。** k > 1 で生じる偏った誤差を取り除き、n16_train の mul+rescale の余裕(0.76 bit しかなかった)を取り戻した。この版は k = 1 で SEAL とのビット一致を壊していたので、8f10ad2 で直した。
8. **深さ 9 の連鎖テストは x ∈ [0.9, 1.025]、相対誤差 2^-10·max(1, |want|) とした。** 計画の [0.9, 1.1] では 1.1^512 ≈ 2^70 となり、レベル 0 の容量(< 2^19)を超える(計画の不備)。
9. **KAT のゲートは SEAL とのビット一致とした。** 対象は復号、直列化、add / multiply / rescale。加えて、誤差が SEAL 自身の復号と同じであること。ブリーフの一律 2^-25 は SEAL 自身でも満たせない(pk 暗号 / rescale / 回転)ので、2^-23 / 2^-15 を副次的な上限にした。
10. **matvec_enc の relins は giant 回(d = 16 で 4)、matvec_plain は 0 回。** Task 21 の「relins == 1」は、親が Task 19 を更新したことで置き換わった。
11. **lattice-estimator は conda-forge の SageMath で動かし、開発ツールとしてだけ使う(依存にしない)。** 秘密鍵の分布は SparseTernary(n/4, n/4, n) でモデル化した(既定のサンプラは −1/0/1 を確率 1/4, 1/2, 1/4 で出す)。
12. **`[profile.test]` は opt-level 3 にした**(N = 2^16 のテストを現実的な時間で回すため)。debug-assertions と overflow-checks は残している。
13. **Task 24: SEAL の演算マイクロベンチを新しく作った。** hoisted×16 は SEAL の通常回転 16 回と比べる。主な比較はシングルスレッドで行う。hekit は rayon を使っていないので全部シングルスレッドになり、マルチスレッドの副次列は無い。

## 7. M2 への持ち込み(Task 25 以降が知っておくこと)

- **Evaluator の API**(`Evaluator::new(params, Option<RelinKey>, Option<GaloisKeys>)`)
  - 演算: `add / sub / negate / add_plain / mul_plain / mul_no_relin / relinearize / mul / rescale / mod_switch_to_level / rotate / conjugate / hoisted_rotate / matvec_plain / matvec_enc`。カウンタは `counter() / reset_counter()`。
  - 規則は検査するだけで、黙って直すことはない。レベルが違えば `LevelMismatch`(`mod_switch_to_level` で明示的に下げる)。スケールの相対差が 2^-10 以上なら `ScaleMismatch`(強制的に揃えない)。レベル 0 での `mul` / `rescale` は `NoLevelsLeft`。鍵が無いか別パラメータの鍵なら `InvalidParams`。
  - `mul` は relin 込みで、結果は rescale 前。3 成分暗号文どうしを足してから 1 回だけ relin する(遅延 relin)こともできる。
- **カウンタの意味**(`OpCounter`)
  - `rotations` は鍵切替をする回転だけを数える。slots の倍数の回転はコピーなので 0。共役は 1 回と数える。hoisted は 1 ステップを 1 回と数える。
  - `relins` は `relinearize` を呼ぶたびに 1(`mul` 1 回で 1)。
  - `matvec_plain(d)`: ct_pt が d、rotations が baby + giant − 2、relins が 0。
  - `matvec_enc(d)`: ct_ct_tensor が d、**relins は giant**、rotations は baby + giant − 2。
  - rescale は呼び出し側が呼んだ分だけ数える。`key_switches()` = rotations + relins(+ bootstraps × `BOOT_KS_EQUIV`、M1 では 0)。
- **matvec のレイアウト**
  - d は 2 の冪で、slots を割り切ること。
  - x は**複製して**暗号化する(スロット j に x[j mod d])。
  - 対角は `diagonals_from_matrix` → `Diagonals::encode / encrypt` で作る(BSGS 用に事前回転済み)。
  - 出力も複製レイアウト(スロット j に (Wx)[j mod d])なので、次の層にそのまま入れられる。
  - BSGS の分割は baby = 2^ceil(log2 d / 2)(d = 64 なら 8×8)。
- **matvec の出力は rescale 前**: レベルは x.level()、スケールは s_x × s_W。呼び出し側が 1 回 rescale する(対角を Δ で符号化しておけば、rescale 後のスケールは ≈ Δ に戻る)。
- **Galois 鍵のステップ**
  - `kg.galois_keys(&bsgs_rotation_steps(d))`(ステップ 1..baby と baby·k)に、ほかに使うステップを足して作る。生成元は 3 で、負のステップも使える。
  - **n16_train の Galois 鍵は 1 本 302 MB**(dnum 8 × 2 × 36 limb × 2^16 × 8 B)で、Relin 鍵も同じ大きさ。本タスクのベンチは 21 本(6.3 GB)を作った。
- **精度の余裕**
  - n16_train の最小は matvec_enc の約 24.5 bit。テストの閾値は 24 bit なので、固定シードで余裕は約 0.5 bit(≈ 2σ)しかない。
  - seal_2020 の最小は約 22 bit(閾値 20 bit)。
  - 学習で層を重ねると誤差は積もる。M2 の精度目標は、この余裕から逆算して決めること。
- **演算コスト**: n16_train の 1 演算あたりの時間は §1 の表にある(1 スレッド: 回転 257 ms、mul+relin+rescale 303 ms、matvec_enc d = 64 で 5.7 s)。鍵切替 1 回は回転 1 回とほぼ同じ。hoisted 回転は 1 ステップあたり通常回転の約 0.4 倍かかる。鍵生成は、Galois 鍵 21 本 + relin 鍵 + 対角 64 本の暗号化で約 22 s。
- **乱数**: `Rng::from_seed`(ChaCha20)が唯一の乱数源。本タスクで離散ガウスのサンプラを CDT 法に変えたため、同じシードでも Task 24 以前とは雑音の値が違う(テストはすべて通る)。
