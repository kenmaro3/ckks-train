# Task54 CPU wire FIX1 scoped re-review

This file is prepared before the author's freeze. It is not a review dispatch until root supplies the final report/diff/pin hashes. Review only the frozen FIX1 when released; no second writer.

Read `task-54-d7-cpu-wire-fix1-brief.md` first. It contains the original findings verbatim, the binding original/adopted-contract references, and root's CW06 process disposition made before correction. Read the frozen FIX1 report `task-54-d7-cpu-wire-fix1-report.md`, `task-54-d7-cpu-wire-fix1-pins.json`, and FIX1-only diff `task-54-d7-cpu-wire-fix1.diff`. Original review is `task-54-d7-cpu-wire-review.md`; the exact previously reviewed source and evidence are retained in `cpu-wire-review-snapshot/`. Source BASE is abd6c3393b843d7d80fff4ebc890526866682c3b; intervening main commits only publish documentation/operational observation. FIX1 compares against that preserved reviewed source, not an invented committed intermediate revision.

Verdict each CW01–CW07 and inspect new breakage in FIX1 only. CW01–05 are concrete supplied-record correctness findings; CW07 is the coverage-claim correction. For CW06, verify the documented historical limitation and the new correction chronology against preserved tests, fixtures, and logs; do not call the original ordering retrospectively satisfied. Assess whether the root disposition and this correction's evidence support adoption of the corrected source, explicitly retaining any limitation. No broad re-review of untouched code. Out-of-scope observations belong in a separate non-blocking section for final review.

Root has independently reconstructed all137 expanded FIX1 fixture members using forward immutable-base slices, checked every name/length/SHA against the manifest, rejected duplicate keys/cycles/invalid bounds, and checked unchanged original78. `root-cpu-wire-fix1-byte-custody.json` is this evidence. It establishes custody, not semantic correctness; review the independent writer and counterexample construction as needed. `root-cpu-wire-iris-schedule-derivation.md` records the actual registered Full schedule derived from matrix/profile source, not executed HE work.

Review is read-only except the requested review report. No source/index/HEAD changes, no commit/push, no agents, no native/HE/context/materialization/registry/parameter/dataset execution, no SSH/network/dependencies/new cache. Do not rerun already-recorded covering tests or broad suites. If source reading exposes a specific doubt not answered by existing evidence, report it with exact lines and a proposed narrow check; root will route it to the author.

Write `task-54-d7-cpu-wire-fix1-review.md` with per-finding ADDRESSED / NOT ADDRESSED (and the explicit historical CW06 status/disposition), evidence file:line, new breakage and severity, out-of-scope observations, and separate SPEC and QUALITY verdicts. Return only verdicts, open IDs, report path/size/SHA256. No live authority, real runtime/native release, scientific result, or Task54 completion follows from this source review alone.

## Frozen release, 2026-11-07 20:50 UTC

Author confirmed source frozen and handoff complete. Root fully read8905B report, verified all56 current file pins, read all named correction RED/GREEN and final regression/Clippy/fmt logs. Review may now proceed.

- report8905B SHA2568b8596d3b21d5b1503d85cab0046dbf20134c73f6c3dc4e928455f9c40db9e00
- pins37731B SHA25643bee70ab13071e85da9bbded3f6ff773a2240fdfac57acba1074298dd6ff0d0
- completeFIX1diff2652094B SHA256018f2a97cc834e3747972c968abcc25729d379929abd1d8418a6faed4df7944c
- originalBASEfullsource3125754B SHA25694ff5a839210a77d7cc98b09aa86aa777eb12b082cc3e1b735b8599a1db342d8
- `root-cpu-wire-fix1-review-code.diff`25907B SHA2561c7801487de7831959b3773b60ac34430a935366e88c030f71401d01082cd320: every production/test code hunk preserved; only the single giant FIX1 literal JSON hunk excluded and explicitly referenced. Full diff/fixture/writer/manifest remain available. Prefer this code diff for reading, then inspect independent fixture grammar or raw members relevant to the findings. This does not exempt fixture correctness from review.
