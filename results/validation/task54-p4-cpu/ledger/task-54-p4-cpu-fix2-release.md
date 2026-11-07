# P4 source fix round2 — pre-runtime diagnostic inventory correction

Root read independent fix1 review in full: S1/S2/Q1/Q2 addressed; F1 is a deterministic new diagnostic-count mismatch. No runtime has been invoked. Original source/report/evidence remain preserved.

## New important finding F1:264 unchanged assertions cannot cover the retained matrix

**Locations:** `crates/ckks-train-bench/src/study_diagonal/p4_tests.rs:386-392` (hard cap264), `512-515` (one reorder assertion for every two-micro first update), `619-633` (smoke/scheduled case assembly), `857-863` (all first-update groups), `865-869`, `897-902`, `924-931`, `945` (final264 assertion). The appended source-report diagnostic inventory contains the same undercount.

The report counts only4 two-micro reorder checks. There are **8**:

- SquareC10 and PolyC10 smoke in I and NI:4 first-update groups with `[8,4]`.
- PolyFourRate in I/WN/AU/AA:4 more first-update groups with `[8,4]`.

All18 successful sessions call `prework_negatives` at update0. The retained helper has12 unconditional rejects, plus reorder when two-micro, plus foreign-basis for8 successful I sessions. Therefore:

| Read-only work | Correct source count |
|---|---:|
| First-update unchanged assertions |18*12 +8 reorders +8 I basis =232 |
| Later changing-rate unchanged assertions |2 |
| Success replay/exhaustion unchanged assertions |32 |
| Poison retry unchanged assertions |2 |
| Total unchanged assertions |268 |
| Snapshot captures |54, unchanged |
| Configuration collection calls |54+268 =322 |

The guard stops at assertion265, before the complete intended matrix can finish; even removing its cap without correcting the final assertion would end in a264-versus268 failure. This is a source-derived deterministic breakage, not an observed runtime failure. Source compilation cannot expose it.

**Required correction:** root must explicitly correct the cheap diagnostic inventory/cap/final assertion from264 to268, first-update count228 to232, and configuration collection318 to322; then freeze/recheck the tiny correction before runtime. Retain all8 reorder checks and the full original matrix. No parameter/key/Encoder/plan/oracle/trainer/bootstrap cap needs to increase, and none should be changed. Do not reduce rejection coverage merely to fit264. This reviewer has not changed any cap or source.


## Root ruling and source-only release

Correct the analytical diagnostic inventory before HE:18*12+8two-micro reorder+8I-basis=232 first-update comparisons; +2later-rate+32replay+2poison=268total;54captures+268comparisons=322config collections. This corrects a count, not numerical tolerance or work scope. Alternative deleting four assertions would lose prescribed coverage and is rejected. Keep all expensive caps,54captures,68clientdecrypt endpoints and122encrypted snapshot copies unchanged. Cost if wrong: finite test stops/rework, never silent widening.

Only crates/ckks-train-bench/src/study_diagonal/p4_tests.rs is writable source: change assertion guard264->268 and final expected264->268, associated exact count comments if any. No other source logic or file changes. Append fix2 correction to source report with historical prefix preserved; do not rewrite earlier report/evidence. Save task-54-p4-cpu-fix2-validation raw/JSON plus all8source pins. Same original source-only commands/environment/disk/current-target rules; final focused no-run and strict scoped Clippy once on amended source, assigned rustfmt if needed. No runtime/HE/Engine/planner/constructors/network/subagents/Git staging or commit. Root will independently review tiny delta before any execution. Return actual hashes and terminal evidence, then release source/Cargo.
