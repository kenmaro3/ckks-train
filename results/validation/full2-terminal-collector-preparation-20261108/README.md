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


## One-shot controller preparation

The controller was subsequently independently reviewed: SPEC PASS / QUALITY PASS,
review SHA256 `d8f78ed4547ddf87f4b9e1d17040efae206073809402cce8f678188b59decbc3`.
Pinned controller6461B SHA256
`decdd1ed38f15b9d84cec1df03f54584309d9e359abbe39f98373246e238d1e6`.
Root read the complete review and registered R-41-FULL2-COLLECT-1 as prepared only.
The archived controller is an exact copy of its original ledger execution source;
its explicit observer prefix must belong to that execution ledger.

Actual `--check-only` on the retained1306 pending observation exited2 before
output creation or SSH. Positive SSH/timeout branches remain untested. The
controller requires a qualifying root-reviewed actual observation, pins/holds the
collector bytes, reserves six fresh outputs, uses key-only nice10 SSH and a110s
local wait, and preserves originals/status without automatic retry. Timeout may
terminate only its new local SSH child; it cannot prove remote completion.
No remote launch or terminal/scientific result is claimed by this archive.
