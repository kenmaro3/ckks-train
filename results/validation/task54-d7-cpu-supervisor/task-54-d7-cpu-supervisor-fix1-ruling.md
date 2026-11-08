# Step2 review fix round 1 ruling

Recorded 2026-11-08; before correction source edits. Binding frozen review:
`task-54-d7-cpu-supervisor-spec-review.md`, 10608 bytes,
SHA256 c427b13e30bcca5a3e0d96bcc76eac476c5606d38af19dff54be2bdedd6ffa4e.
Root read the entire report and inspected probe_index, HeldDirectory::recheck,
Monitor::sample and Session::complete. All four findings accepted for correction;
none is waived. Existing submitted source stays an unadopted Step2 attempt.

## Ruling: F01 history traversal scope

The 64MiB/4096 raw-storage limits apply to newly added operation storage, not
lifetime replay. Remove the accidental cumulative read-byte and entry charges.
Preserve 8MiB observation, 64KiB command and 128-byte name bounds, original/copy
byte equality, all retained identities, account lock and malformed/incomplete
record refusal. Keep per-operation storage/capacity accounting unchanged.

Prior history must be streamed one probe/check pair at a time, with bounded live
record buffers. Do not accumulate all probe names, observations or parsed pairs.
A check group has only the fixed 16 allowed check names; a completed pair has
exactly command.json/observations.bin, and the active exception has command.json
only. Those local cardinalities supply bounds without a permanent account
lifetime budget. Exhaustion in a *new* operation remains a real refusal.

The outer history has finite filesystem contents but no newly invented fixed
lifetime record count. Preserve supported-filesystem and retained-name rules.
Document how iteration rejects mutation/replacement/unknown names and preserves
safe directory custody under the account lock; do not claim the lock prevents
arbitrary same-user filesystem mutation. If constant-memory traversal changes a
previous duplicate check, explicitly justify equivalent detection based on the
actual directory API and filesystem assumptions before implementation. Do not
silently skip history, delete records, repair markers, accept partial scans, or
substitute a hash-only comparison.

Reason: legitimate resolved reuse must not become permanently blocked merely
because original and indexed bytes were read twice. Alternatives rejected:
raising the lifetime cap merely postpones the same bug; deleting history loses
custody evidence; loading the full namespace removes the memory bound. Cost:
O(history) validation time and bounded per-record memory. This does not promise
constant latency or a hard OS wall-clock bound. Cover histories beyond the old
64MiB replay/4096 entry limits using the real traversal/completion decision where
feasible; do not count a synthetic boolean as native filesystem execution.

## F02 final sample failure

Accept the review's concrete trace. Explicitly gate final completion on returned
sample validity and non-threshold outcome, with sticky final-teardown failure
before any original/account completion or production closure. Preserve original
stop reason separately: deliberate native stop is not itself proof of final
sampling success or failure. Share the actual completion decision in covering
negative tests for RSS failure and late final sampling after child exit; retain
marker/lock handling and no unknown-process signal. No schema or gate relaxation.

## F03 native clock fault source

Implement the already required scoped, raw-labeled clock fault paths under the
existing common-clock and TERM/grace/KILL checks. No seventeenth check and no
production clock override. Exercise the real latch/fallback/action path with
legal outer custody where intentional inner failure remains unresolved. Cover
loss before/after TERM and recovered clock after latched loss. Never serialize
invented valid clock values or describe injection as a kernel outage. Source
and local covering tests only; native execution remains a later gate.

## F04 and evidence preservation

Correct the report to describe selected pinned v3 account anchor acquisition,
not passwd lookup. Snapshot all frozen source and review metadata before edits;
preserve original RED/GREEN/compile failures. Append fix report with exact
covering commands/results and scope. Freeze a fix-only diff and new pins for
scoped review, leaving the original reviewed package immutable. Do not commit,
push, run server/native jobs, alter billing, or touch unrelated root artifacts.
