# Gate evidence custody design

Root ratifies proposal `fc938561` after independent SPEC/QUALITY review `31d9fb50`. This closes the live-gate design's carried M1 custody requirement: fixed branch paths/names/caps, exact binary evidence bytes, guarded sidecar-before-admission ordering, retained file identity, partial states and recovery. The [readable design](../../../docs/study-gates-design.md#6-evidence-custody-and-recovery) supplements the full exact proposal here.

Full orphan/named-attempt refusal and explicit Retry semantics remain. Quick has a separate tree and Initial-only lifecycle; a separate explicitly selected authorized namespace is needed to repeat a case, without automatic rotation or a new user approval ceremony. Recovery inspects/resyncs complete verified finals, never creates missing evidence, live authority or a fabricated end.

This is design acceptance only. Exact new evidence/Quick schemas, independent literal vectors and C01–C10 operation/fault inventories remain required before their source/execution. No source, tests, native publication, gate or HE campaign was run for this design. Three evidence files are exact ledger copies; SHA256SUMS covers them and README.
