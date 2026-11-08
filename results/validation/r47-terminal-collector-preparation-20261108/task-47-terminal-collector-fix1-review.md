# R47 terminal collector fix1: scoped re-review

SPEC: PASS. QUALITY: PASS. Both prior P2 findings: ADDRESSED. No fix-induced finding within this scope. Source approval does not release a controller, request or remote invocation.

Reviewed `task-47-terminal-collector-fix1.diff`, `task-47-terminal-collector-fix1-report.json`, the five added covering fixtures, retained corrected RED/ GREEN stderr, preparation precision/portability note and actual final source. Final source: 11201 bytes, independently confirmed SHA256 `8e40fb7efa1cfdb90f826fe4b8b8c6046a876616f695790d5e5c27975c2a0d5a`.

- Timestamp P2: the lower bound now uses `math.floor(gate_at)`, matching whole-second runner precision. Start/end ordering and current-observation upper bound remain intact. Fixtures cover same-recorded-second acceptance, a start before the gate floor, and end before start. The comment and preparation note explicitly preserve unresolved subsecond ordering and the limit on historical-execution claims.
- Refusal-retention P2: `RecursionError` is explicitly included in the existing refusal handler. The path now records error stage/path/type/message and returns the capture containing already acquired raw entries, so the unchanged main block can serialize it. No broad catch or stable-read changes were introduced. Fixtures cover actual over-nested JSON refusal and a separately injected parser `RecursionError`, checking owner/gate and offending-state retention.

The portability distinction is accurate: the retained local Python 3.14 run reports excessive nesting through `ValueError`, which the original collector already handled. The new exception path is tested by explicit injection, not represented as an actual local parser `RecursionError` or remote reproduction. Corrected RED records the fractional-gate failure and uncaught injected error; GREEN records 15 tiny synthetic tests passing in 0.113 s. These are retained author results, read here without rerunning tests.

Reversing only the two source changes and explanatory comment yields SHA256 `60ee6293bdac5ead622fd6e8560667bdb89ce6cc1a3403c4a354845919b807e9`, exactly the initial reviewed source. Thus helper bytes, fixed paths, caps, owner/gate handling, missing-state and nsys semantics, exit/provenance checks and scientific-acceptance labeling remain unchanged. No further audit was performed.

Only local artifact reads, a byte-level hash comparison and this report write were performed. No tests rerun, SSH, research/native execution, source edits, controller/request release, commit or push. The prior review's acquisition limits and separate future transport-envelope requirements remain applicable; this result establishes no current server state or scientific measurement.
