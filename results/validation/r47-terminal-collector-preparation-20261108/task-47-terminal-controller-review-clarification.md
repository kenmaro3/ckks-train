# R47 controller review: correction of erroneous P2

SPEC: PASS. QUALITY: PASS for the reviewed operations-source draft. The sole P2 in `task-47-terminal-controller-review.md` is WITHDRAWN as incorrect. No source change or weaker process-inactivity fallback is required. Initial report SHA256 `359739a1f1229a22d795292f8e3e0d7699762221aded40236e8168157201f959` is preserved unchanged; this clarification supersedes its verdict, finding and proposed amendment.

Root identified the enclosing `queue_lock` context manager that the initial review failed to trace. Independent reread confirms the exact unchanged source path:

1. `post_full_queue.py` lines 441–442 enter `with queue_lock(cfg['state'])` around the whole `Queue(cfg).execute()` call.
2. The R47 `job()` writes `post-full-r47-build.json` at lines 351–353; a failed build raises `QueueError` at lines 355–359.
3. That exception first unwinds through `queue_lock`, whose `yield` is inside `try` at lines 109–110. Its `except BaseException` at lines 111–113 atomically writes `terminal.json` with failed status, error and ended time, then re-raises.
4. Only afterwards does the outer handler at lines 448–450 print the stop and exit 2.

Therefore the existing controller's released-gate + R47-build-state + queue-terminal-file branch does cover the ordinary failed-build exception path. The synthetic positive fixture represents a source-reachable file combination. The initial statement that `terminal.json` is written only at normal queue completion was false because it omitted the context manager's second writer. I retract both that statement and the proposed inactive-process fallback.

The controller remains unchanged at 7653 bytes, independently reconfirmed SHA256 `4de62d14af065aee39ae9c74e9e82b6bd71e39f921a0936901961beb7cc8a6d6`; queue source remains SHA256 `1a5f780070e004ec826f21b7dcc187d9e9a65d44e694f2cd5e2f1831589a78f9`. No additional blocking finding remains in the scoped review. Abrupt death or failure to write a terminal file does not grant this collector's normal precondition; it should continue to refuse rather than gain a weaker fallback from the withdrawn finding.

The initial report's other source checks and evidence limits remain applicable: approved collector byte/hash pin, exact R47 outputs, unchanged observer/chain/source checks, retained raw outputs, explicit post-capture 64 MiB envelope validation rather than a streaming hard cap, separate remote original-byte bound, no automatic retry and no scientific acceptance. Recorded six pure synthetic tests and actual pending check-only refusal are the available executed evidence. Positive SSH, timeout/launch-error and envelope-excess paths remain unexecuted.

This correction uses source inspection and local hashes only. No tests rerun, SSH, source/controller edits, request release, experiment, commit or push. Only this clarification was written. Source-review PASS does not itself authorize remote invocation; the exact registered request and root inspection of an actual qualifying observation remain separate prerequisites.
