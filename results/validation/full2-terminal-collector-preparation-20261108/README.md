# FULL-2 terminal collector preparation (not executed remotely)

The read-only collector for R-41-FULL-2 is prepared and independently reviewed.
This archive is operations source and local validation, **not an experiment
result, terminal observation, retrieval, or RQ1 acceptance**. FULL-2 was still
running at the last registered observation; no new server call was made here.

Final source SHA256: `50f2cf22ca3f540a86117c06664c6606f23fd9e2532f72cfa2eaabea168058ef`.
Independent fix1 review: SPEC PASS / QUALITY PASS, F1 and F2 addressed;
SHA256 `7c4cee20e5f42b03c8d3fb3a6c882aec3777c35b4231bb998a2a0d2609ada05f`.
Root read the complete review and verified these identities before adoption.

The fixed collector reads only the result, actual runner log and named runner
artifacts. It checks exact job/workdir/command and terminal metadata; command
flag order may differ, but duplicate/extra/missing/changed flags refuse. It
preserves acquired original bytes on semantic refusal, retains initial/final
metadata observations separately, rejects symlinks/nonregular files, and bounds
per-file/aggregate reads. Scientific acceptance is always false. A successful
capture still requires independent input/config/provenance and RQ1 validation.

Validation: initial seven local checks passed. Added regression tests reproduced
F1/F2 (RED: 12 tests, eight errors including subcases); after the fix all 12 local
checks passed (GREEN, 0.061s). Captures in tests are explicitly synthetic parser
inputs or own temporary files; the FIFO child has a two-second timeout. These
checks are not remote or native research evidence. Exact commands, exit evidence,
source/test diff, retained initial draft and reviews are included unchanged.

No SSH, process control, experiment, scientific JSON edit or collector launch
occurred. A future one-shot controller must still pin source/command, use key-only
SSH and nice10, bound its lifetime, retain stdout/stderr/exit, create fresh-only
outputs and avoid automatic retries. The runbook requires an actual terminal
observation before release. The published preview/tag/assets remain unchanged.

The whole staged whitespace check reported preserved whitespace in the raw RED
stderr and unified fix diff. Those evidence bytes were kept unchanged. The
whitespace check passed for all other archived files; this is an explicit
evidence-format exception, not an unreported all-files pass.
