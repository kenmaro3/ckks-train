# Step2 fix round 2/5 — whole-source quality findings

Recorded before source edits. Root fully read the13292-byte quality report
SHA256 a644d4d30d8bf4d043ae76f812c97a737bd474d4d2965e44cea51a46e962f33d,
and independently inspected handshake.rs duplicate acquisition. Accept Q01/Q02;
no source/native acceptance yet. The earlier scoped F01–F04 pass remains valid.

## Q01 accepted: private aliases must not leak through exec

Use safe atomic close-on-exec duplication for private fd0/fd1 aliases. Keep original
intentionally inherited standard descriptors and same OFD lock semantics; explicit
Stdio/ancillary handoff is the only intended transfer. Audit all private aliases in
the fixed P→M→L route. Do not relax EOF/drain/reap or replace EOF with exit status.
Reason: a live leaf can retain M's old P–M socket, preventing the actual M EOF that
P needs before permitting leaf shutdown. Closing only at normal exit is too late;
non-atomic flag-setting adds a needless inheritance interval when an atomic safe
API already exists. Cost: explicit descriptor ownership/flags and covering tests.
No new package, project unsafe, child hierarchy or wire format is needed.

Add a narrow test of the actual duplication helper and FD flags where supported,
not a projection-only model. Preserve the native assertion that M EOF is observed
while L remains alive and before L teardown. Linux-only tests may be authored and
cross-compiled here; disclose any that cannot run on the Mac. Existing authorized
Linux CI can later execute a bounded descriptor-only unit test after reviewed
source is committed, separately from native16 runtime/provisioning. No new local
Linux environment/server job/process experiment is authorized by this ruling.

## Q02 accepted: live framed control loss must initiate cleanup

In the actual pump/controller lifecycle, distinguish unexpected frame-boundary EOF
from terminal status plus required drains. Feed unexpected framed EOF from a live
owned child into the existing protocol-failure stop and owned TERM/grace/KILL path,
even with no queued write. Never infer process exit/reap or discard ownership from
EOF. Preserve original successful terminal protocol phases, raw Git stdout mode,
and existing stop/drain semantics. Document the finite allowed-phase decision and
its source basis before editing; do not add a broad EOF exemption or new timeout
as a substitute for detecting control loss.

Cover the actual shared decision and scheduler transition from live framed EOF
without an explicit test call to fail(). Include raw-output and legitimate terminal
controls. Keep the native both-EOF/queued-output case as a genuine stopping-wait
check; adapt its ordering/labels if automatic failure now occurs sooner, rather
than leaving it dependent on a manual failure that bypasses production detection.
Reason: an alive process with a closed control channel cannot execute the promised
protocol; normal RSS is irrelevant. A global timeout merely delays or masks the
missing transition. Cost: distinguish control mode and permitted terminal states.

## Correction procedure and retained scope

Snapshot all71 FIX1 source files/review metadata before edits. Preserve original and
FIX1 reports/pins/archives/raw failures. Test expectations -> implementation ->
covering tests, strict Linux/Mac source checks as affected, then freeze FIX2-only
diff/newpins/command index/report. Name unexecuted Linux tests and compile-only RED
honestly. Preserve F01 streaming limits, F02 final gate and F03 fault cleanup/latch.
No source/operational-design change outside these findings without a recorded reason.
No commits/push/network/server/native16/spending. Root owns review and adoption.
Step3 and full matrix remain required later, not silently waived or claimed here.
