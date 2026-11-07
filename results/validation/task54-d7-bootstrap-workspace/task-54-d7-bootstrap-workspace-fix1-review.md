# Bootstrap workspace FIX1 scoped independent review

SPEC: **PASS, scoped to F1 correction**.

QUALITY: **PASS, scoped to F1 correction**. Original P1 finding F1 is closed by the corrected report identified below. No new finding in this delta. This is conditional source/algebra review only; no physical-model acceptance, memory-fit decision, execution authority, scientific validation or Task54 completion follows.

## Closure evidence

Fixed BASE remains `5b50e841e65a0bad698b9a61621ba2e1fde7571d`. Reinspection of `crates/ckks-train-ckks/src/linalg.rs:809–887` confirms that xs owns the outer row allocation and inner Option rows throughout block collection and reduction. Blocks borrow xs; neither successful output nor a block Err consumes it. Row construction failures have only a prefix of these rows, already dominated by the unchanged saturated BabyBuild term.

The corrected expression explicitly includes `Rows(t,b_s) + X + max(Collect,Reduce,U_mul_plain(2,l)+C2(l))` as the post-build branch. Therefore both phases contain the known retained row owner. Writing BabyBuild as `rho+B` and post-build work as C gives `max(rho+B,rho+C)=rho+max(B,C)` in each nonnegative additive coordinate. This directly implements the original bounded remedy without relying on disjoint row coordinates or an unsupported build-phase dominance assumption.

The new explanation correctly retains rows on success and error prefixes, preserves the existing error-result payload envelope and IntoIter storage requirements, and prevents silently assigning rows to ChecksAndCollectionMetadata. It distinguishes Option slot backing from separately owned ciphertext component/limb backing and explains that inline ciphertext storage inside a slot must not be counted twice during physical layout lowering. It also correctly distinguishes lowering a conservative componentwise vector bound from lowering every complete retained-owner branch before taking max; it does not assume arbitrary lowering commutes with max.

No primitive envelope is evaluated or assigned zero. No physical byte total or guessed multiplier is introduced. All original primitive/collection/layout/allocator/runtime/parallel-mode and external-owner limits remain active.

## Unchanged scope and integrity

The exact diff against the preserved original report was independently regenerated in memory with stdlib difflib and matched the submitted narrow diff byte-for-byte. The only formula change is the explicit Rows addition to W_BSGS. Other changes are the bounded FIX1 explanation and the historical correction from “only these two output files are written” to “the original derivation wrote only these two output files.” All remaining formulas, cache/degree worksheet values and original limits are unchanged. The structural integer worksheet was not rerun or recertified.

The original review remains byte-identical to its recorded hash. The canonical source-pin JSON equals the preserved original byte-for-byte. Before calculations, the finite inventory was declared: eight named local inputs/outputs below and the one fixed linalg blob. All recorded bytes/SHA256 values were verified; the fixed blob's Git identity also matched. The source-pin ledger and FIX1 pin ledger are provenance inputs, not expanded semantic review of every listed source.

All local paths are in `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/`; `before/` in this table abbreviates `task-54-d7-bootstrap-workspace-before-fix1/`.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| task-54-d7-bootstrap-workspace-derivation-report.md | 33388 | `e7493f3c62e8dfa69290f9895582226a9dab322c15f1a990b5420266cabbacaf` |
| task-54-d7-bootstrap-workspace-fix1.diff | 5889 | `44bd9a07c4a1767c9e8e4f40f8af5910c19c1fc610806e0d26857f74974e3d87` |
| task-54-d7-bootstrap-workspace-fix1-pins.json | 3189 | `1cbb822bc76d33695dcd00aaf72cf03b28bccffff5d9e1381ccea81461ec281e` |
| task-54-d7-bootstrap-workspace-fix1-brief.md | 1606 | `21a5367f9188dca20ad5f331a8a10d677b1f14d2b3baf0c04e23a847eb76d12a` |
| task-54-d7-bootstrap-workspace-review.md | 12521 | `e76661e52aa65cf2df043522fd5a097331b6e40d806408e1c93bb8b614cfae50` |
| task-54-d7-bootstrap-workspace-source-pins.json | 10199 | `9bd058ebff48465845e4000cb9c758b648bbd42a1b6521eb4970ecf2b277e86e` |
| before/task-54-d7-bootstrap-workspace-source-pins.json | 10199 | `9bd058ebff48465845e4000cb9c758b648bbd42a1b6521eb4970ecf2b277e86e` |
| before/task-54-d7-bootstrap-workspace-derivation-report.md | 30309 | `75e847b3086266d0ccf2a2c1af3ac5bd567096e530c32248de2beb9f7e9f7853` |

Actually reconsulted source: `crates/ckks-train-ckks/src/linalg.rs`, BASE above, 71884 bytes, SHA256 `4eef1dccf8b651c1d536ab6da9f323b6aa4e3a9bc7c0d70b7560bb25c241db01`, Git blob `51ee94b98a0395822f9cba92f118a3beca10faaf`; semantic range 809–887. No other production source was reopened for FIX1 review. The broader original review's source coverage and limitations remain recorded in the unchanged original review rather than being claimed as a fresh audit.

Performed only the specified local artifact reads, fixed source read, preinventoried stdlib hash/diff comparisons and this new canonical review write. No current parser/source changes, project helpers/imports, Cargo/tests/build/--list, native/HE/CUDA/SSH, data/process/credential access, network/install, subagents, commit or push. The original review and report artifacts were not edited by this reviewer.
