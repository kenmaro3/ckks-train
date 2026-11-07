# Task54 P2 — local code-validation evidence

This directory contains **Mac miniature validation**, not scientific training/performance results, secure-profile admission or GPU measurements. Raw test logs/controller records and original review/release reports are preserved byte-for-byte. `runtime-summary.json` is an explicitly derived parsing of `runtime-matrix.log`, not a binary-emitted scientific result. Recipe/literal JSON files are pure-test fixtures. Nothing here replaces the provenance-bearing production result JSONs.

The exact level/scale/count planner is checked against real existing CPU encrypted training, using runtime-observed values rather than values copied from the planner. Current source adds two-way observation identity/cardinality, branch-specific coefficients, original x/y versus aligned metadata and full final-state/boundary comparisons. Source arithmetic, presets, tolerance and bootstrap policy are unchanged.

Validation completed on2026-11-07:

- 16 pure unit tests and13 external pure tests; strict Clippy, format and runtime compilation pass.
- Exactly one final eight-function runtime matrix:8PASS,0FAIL,93 logical results comprising99 stage calls and220 actual bootstrap attempts/Ok returns. The released99-stage/266-bootstrap limits are bounds, not reported execution counts.
- The final command uses existing target with CARGO_INCREMENTAL=0,CARGO_BUILD_JOBS=8,RAYON_NUM_THREADS=8 and testthreads1. It appears below and in the exact controller record. Its orchestration duration is not a training-speed measurement.
- Source reviews for F1/F2/one-valid and the final F3 pure witness pass. Final actual-outcome review also passes (`runtime-outcome-review.md`). Root accepts the bounded P2 implementation/validation scope; no overall project or secure-profile acceptance is implied.

```sh
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 cargo test -p ckks-train-nn --lib p2_runtime_ -- --test-threads=1 --nocapture
```

The encompassing stage/block counters are rotations18783,relins10750,ct-ct12916,ct-pt47862,rescales16176,SSE440,bootstraps220. The220 bootstrap subrecords are already included; adding their counters again is incorrect. The selected matrix covers the registered activation/class families, placements, optimizer branches, configurations and bounded block controls; it is not an exhaustive Cartesian encrypted campaign. One-valid tails, multicontainer negative checks and all21 geometry profiles remain pure-only coverage. Runtime source pins match before/after.

The F3 pure-only counterexample uses a valid fresh actual-prime state. Structural must_boot(false) is followed by an exact-target plaintext-minimum rejection at weight_unpacked (q10=1021441<2^20). It is **InvalidParams after an available exact label**, not ScaleMismatch/ULP exhaustion, and proves no registered secure profile failure. The source-derived proposal, pre-code root disposition and exact WN-only release explain why this meets the original structural-versus-exact-admissibility requirement. The earlier672-trajectory/2688-transition ULP-filtered search found no witness and is still recorded as such in the correction report.

Historical reports intentionally retain their then-current pending/unmet statuses. Read `initial-review.md`, `correction-review.md`, `f3-review.md` and the later runtime outcome review in order; later evidence discharges only its named scope. Earlier role-collision/NI-hook failures and the unsuccessful search remain preserved in the implementation ledger. No failed run was changed into a success.

P3 admission, P4/P5 execution bridges, full runner, range/precision/resource/security checks, full registered encrypted-block feasibility, final scientific measurements and Task42/M4/M5 acceptance remain separate. The existing full encrypted-block hold remains in force. SHA256SUMS binds this finalized archive except itself.
