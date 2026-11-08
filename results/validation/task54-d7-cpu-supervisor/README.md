# Task54 CPU supervisor — reviewed Step2 source

This commit adopts the reviewed Step2 source: the Linux supervisor/worker and
broker, retained process/resource ownership, sampled-RSS cleanup, compatible
operational evidence plumbing, native-probe driver source, and bounded frontends.
**Native16 execution, Linux kernel behavior, the finalized clean build receipt,
Step3 retained Input/Reviewed readers and five gates, Step4 endpoints, and the full
registered experiment matrix remain incomplete.** No scientific row is added.
The published preview remains separately fixed at6fba46c.

## Review and corrections

The original source package received a specification review and correction round1:
history streaming instead of unsupported lifetime caps, final-sample failure
retention, native clock-fault cleanup source, and the pinned account description.
The scoped FIX1 specification review passed. Whole-Step2 quality review then found
private FD inheritance and missing automatic stop on live framed EOF. Round2
corrected both; scoped quality review passed. The Linux-only actual descriptor
flag test was authored/cross-compiled, not executed on the Mac.

Root then ran the full bench crate including legacy integration tests on the frozen
FIX2 worktree:509passed/1failed/6ignored across33 target summaries, exit101.
The sole failing CLI test expected an obsolete generic error for a legacy profile
that now correctly refuses earlier as outside the closed CPU slice. Round3 changes
only that exact expected message and a fixture comment. All production files are
unchanged from FIX2; nonzero exit, empty stdout, exact diagnostic and no-artifact
assertions remain. Default/parallel/CUDA focused tests each pass2/2. Independent
FIX3 SPEC/QUALITY review passes. The original full-suite failure is retained;
there is no claim of a new complete-suite rerun after this test-only correction.

Final authored set:72paths (71 prior files plus the existing CLI integration test),
selected source inputs453/512. This selection is not a physical inventory or a
clean build identity. The additional operational-design document was already
pinned and read by the source reviews. Recorded execution HEAD7578673 plus frozen
WIP must not be confused with CI for the old public preview or this adoption commit.

## Evidence layout and verification

- Final FIX3 report/pins/diff and scoped review are direct files here; earlier
  reports and specification/quality reviews are also copied unchanged.
- `reviewed-evidence-through-fix2.tar.gz` contains959 original ledger-relative
  members, including all211 earlier author's commands and source/review snapshots.
  The accompanying manifest maps original ledger paths to archive members with
  exact sizes/hashes. All members were read back and matched before adoption.
- FIX3's pre-edit snapshot is194members and preserves all71 prior sources, the
  existing test, review metadata and root regression. It is separate from the
  older snapshots. The one-file diff reconstructs the final reviewed test.
- `root-integration-red/` retains the complete root command/stdout/stderr/exit.
  `fix3-evidence/` retains the seven new focused commands at their original
  ledger-relative paths. Metadata in these immutable reports uses the original
  execution paths; extract archives relative to that ledger when resolving them.
- Root verified the final529source/input pin occurrences,17review pins and28log
  pins. Independent reviewer separately repeated those checks and confirmed all
  prior71 source files unchanged. Source tests:FIX2 operational43 and supervisor30,
  Linux/Mac strictClippy/fmt; FIX3 focused frontend2tests in each feature branch,
  focused strictClippy/fmt and isolated patch verification. All raw failures remain.

Compiler success is not actual pidfd/subreaper/flock/EOF/clock behavior. Mac CLI
smoke tests are validation, not server timings or new paper measurements. The
registered native validation and later actual I/NI scientific paths remain held
until their genuine prerequisites and sequential server queue permit them.
