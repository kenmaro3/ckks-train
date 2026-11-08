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
