# FULL-2 terminal collector preparation

Scope: a finite read-only collector for the existing terminal collection runbook.
This is an unexecuted operations draft, not a new experiment or source acceptance.
No SSH/server call, result retrieval, runner control, scientific JSON edit, or
Git mutation was made. Root owns this operations script; the sole Rust writer
continues separately. Independent review is required before using the draft.

The collector selects only the fixed R-41-FULL-2 result, log and nine runner
artifacts. Terminal meta must identify the exact job/workdir/command with explicit
end/exit/watchdog values. Command order may differ; duplicate/extra/missing flags,
changed values and executable refuse. Nonzero/watchdog terminal evidence remains
capturable with exit3 and runner_success=false. Missing mandatory artifacts also
produce exit3; absent optional artifacts are explicit. It never starts/stops jobs.

Each fixed path is opened using own regular-file leaf + no-symlink directory
walk. Per-read file identities/size/times and original directory identity chain
are rechecked. Each file is bounded8MiB, total original bytes32MiB; base64 is an
operational transport wrapper, not a scientific producer. Terminal metadata is
reread and compared across collection. This is read-time stability, not a
historical build/input attestation or an atomic multi-file filesystem snapshot.
Actual output is still independently reviewed and passed to rq1_summary only
following provenance/config checks in the runbook.

Validation: `python3 -B <ledger>/task-42-full2-terminal-collector-tests.py`, original
exec59fff3 exit0. Seven local unittest cases cover command reorder/identity,
terminal fields/time/types, failed terminal preservation, duplicate JSON fields,
actual own temporary bytes, leaf/directory symlinks and sparse oversize refusal.
stdout empty; stderr contains full7-test PASS result. These are local parser/IO
checks, not remote execution or scientific validation. The operations draft was
written first and then these checks were added; this is not a claimed RED/GREEN
TDD sequence. No Rust research source is modified by this preparation.

Files and exact hashes are in task-42-full2-terminal-collector-pins.json; full new
source/test diff is task-42-full2-terminal-collector.diff. Root must inspect the
next registered FULL-2 observation and independent terminal state before release.
A future one-shot controller must retain command/source pins, stdout/stderr/exit,
use key-onlySSH/nice10, fresh-only output paths and no automatic retry. No such
controller has been invoked or terminal observation claimed here.

## Fix round1 — F1/F2

Root read the complete independent review8635B SHA1627db5a1e9174bef75a2d3b0635af2e45f023f4f84a90714a9396853b413b09.
Preserved the initial reviewed source/test/report/pins/diff unchanged under
`task-42-full2-terminal-collector-initial/`. Root continued ownership of this
small operations draft (not Rust implementation) and returned the exact fix to
the independent reviewer; no competing research-source writer was started.

F1: retained evidence envelope exists before first metadata parse. Every bounded
successful read is inserted before semantic comparisons; refusal contains exact
prior bytes and stage/path/type/error. Final metadata has separate `meta.final.json`
observation with its original remote_path, preserving both observations. Non-object
JSON explicitly refuses. Remaining aggregate budget is checked against held file
size before payload allocation. Missing files remain explicit without invented
bytes. No collection proceeds past an identity refusal.

F2: leaf open uses O_NONBLOCK with O_NOFOLLOW, then immediately checks held regular
file/type/UID. A local own FIFO without a writer now refuses promptly.

Covering tests were added before the fix. RED originalexec ee3164 exit1:12 tests,
8 errors across refusal subcases/FIFO timeout; old7 cases still pass. FIFO test
uses its own bounded Python child, with a2-second subprocess timeout; no research
processes are involved. GREEN originalexec c44607 exit0:12 tests passed in0.061s
(exact timing in stderr). Added coverage for active/malformed/non-object metadata,
offending command, late IO refusal and final metadata drift preserving bytes, plus
actual local FIFO nonblocking refusal. Synthetic mocks return plain operational
capture dictionaries only, not scientific JSON or trusted execution capabilities.

Exact fix diff and six source/test/output pins are in the `fix1.diff` and
`fix1-pins.json` files. No SSH/server execution or automatic retry occurred.
