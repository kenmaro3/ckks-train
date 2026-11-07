# D7 retained preparation/accounting design evidence

Design only, 2026-11-07. Eight artifacts are exact ledger copies. No source, constructor, planner, canonical-case test or HE execution is established by this archive.

The initial proposal (`e51db678`) received one Important finding (`1fa64626`): a checked object could outlive callback accounting, and input/parameter lifetimes were improperly tied. FIX1 (`6de4d485`, report `0ecf7e6a`) separates those lifetimes and legacy compatibility from the new retained D7 owner. It also traces geometry through encrypted initial state, block binding and sessions: future fixed continuation must transfer the same accounting until crypto teardown. Independent scoped re-review (`3a129a3e`) accepts I1 at design level.

The new accounting covers retained and explicitly sized temporary application payload under unchanged original limits. Preexisting P2 creation, legacy outputs and library stable-sort scratch are explicitly outside this boundary; it is not an all-preparation allocation/RSS guarantee. Exact binding-workspace sizing and module-confined source-derived digest caching remain implementation checks. The six canonical preparation cases are proposed test inventory, not measured success.

See [readable decision and alternatives](../../../docs/study-execution-design.md#5-retained-preparation-accounting-refinement). Source implementation, fixed CPU bodies, actual gates/start, production wiring and full Task54 experiments remain unfinished. SHA256SUMS covers the eight exact copies and this README; archived unified diffs are intentionally preserved byte-for-byte.
