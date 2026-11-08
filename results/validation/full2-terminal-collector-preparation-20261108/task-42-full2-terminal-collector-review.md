# FULL-2 terminal collector independent review

Reviewed 2026-11-08. Scope: the unexecuted operations draft and its retained local tests, against the terminal collection plan and registered R-41-FULL-2 command. This is not remote release, a terminal observation, result acceptance or Rust source review.

**SPEC: CHANGES REQUIRED. QUALITY: CHANGES REQUIRED.** Two bounded findings prevent use of this draft. The current tests are retained evidence of seven local passes, not coverage of the two refusal/IO paths below. No tests were rerun.

## F1 — Refusals discard original bytes already collected (P2)

Source `task-42-full2-terminal-collector.remote.py`, lines 135–137, 161–167 and 180–184.

The first meta.json is read successfully into `raw` and `meta_entry`, then parsed before any capture is emitted. If metadata is active, malformed, has duplicate keys, or disagrees on job/workdir/command, the outer handler outputs only `operational_collection_refusal` and a string. The original metadata is lost from the collected evidence. Likewise a mismatching cmd.txt/workdir.txt is checked before its entry is added, and a later refusal (including final metadata drift, an unreadable optional file, or a limit failure) discards all earlier file entries. Final changed meta bytes are also discarded. This violates the plan's explicit preservation of observation/error evidence, especially its missing-or-active metadata path. A future controller retaining stdout/stderr cannot recover bytes the collector never emitted.

Required bounded correction: retain each successfully bounded read in a non-success evidence envelope before parsing or comparing it; emit that envelope with structured stage/path/error state on refusal. Preserve both initial and final metadata observations under distinct observation identities when the final read differs; never overwrite the first with the second or label the set stable. Active metadata must remain pending/refused, with its actual bytes retained. Preserve offending cmd/workdir bytes and already collected files on later refusal. Missing or unsafe/unreadable files need explicit error records, not invented bytes. Keep the existing caps: oversized payloads need only their available bounded diagnostic facts, not an unbounded read. Do not continue into result acceptance or expand the fixed path inventory after an identity refusal.

The metadata decoder also assumes a JSON object: valid JSON `[]`, `null` or a scalar reaches `.get` and can raise AttributeError outside the stated handler. Explicitly reject a non-object at this same boundary so malformed metadata follows the structured refusal/preservation path. This is part of the same finding, not a request for a generic exception framework.

Finite follow-up checks: initial active/malformed/non-object metadata retains exact bounded raw bytes and refuses; mismatching command retains its original bytes; final metadata mismatch retains both observations plus earlier files with non-success status; later IO refusal retains earlier captures. No remote execution is needed to verify those decisions.

## F2 — Leaf open can block before rejecting a non-regular file (P2)

Source `task-42-full2-terminal-collector.remote.py`, lines 93–97.

The leaf uses `os.open(..., O_RDONLY | O_NOFOLLOW)` before fstat establishes that it is an own regular file. If a fixed leaf is a FIFO without a writer, opening for read blocks indefinitely; the regular-file rejection and finite read loop are never reached. The source/test bounds cover regular-file sizes and symlinks, but do not make this path finite. This is relevant to the stated finite collector behavior even though no such remote file is claimed to exist.

Required bounded correction: use a nonblocking leaf open (O_NONBLOCK together with the existing no-follow flags), immediately validate the held fd's regular-file type and UID, then retain the existing bounded regular-file read and name/ancestor checks. A pre-open lstat alone is insufficient because the leaf can change between lstat and open. Preserve the refusal through F1's evidence/error path. The eventual one-shot controller should still bound its SSH/read lifetime; that does not replace fixing the avoidable FIFO block and must never control the research process.

Finite follow-up check: an own FIFO at a local temporary fixed test path refuses promptly without a writer, alongside the existing regular-file/symlink checks. No server run is needed. This is not a demand for filesystem-wide or hostile-kernel guarantees.

## Confirmed scope and correct behavior

The command table matches the FULL-2 executable and registered application flags, including B=48 and absence of FULL-1's `--lr-schedule-batch 48`. The sibling meta/cmd convention records the bare application command, with thread/nice settings outside that string; the draft correctly compares that representation rather than requiring the request's shell `/usr/bin/time` prefix. Parsed argv permits reordering while rejecting repeated, extra, missing or changed flags. Both original command strings remain available on the successful capture path.

Terminal metadata requires the actual `job` key, exact workdir, command identity, timezone-bearing start/end with end not before start, integer exit_status and boolean watchdog_killed. Nonzero exit/watchdog terminal evidence is capturable as runner_success=false/exit 3. The result filename alone cannot set runner_success: that flag derives from terminal metadata, while scientific_acceptance remains false. Result bytes are opaque and preserved; an error/failure JSON is not rewritten or promoted to scientific success by this code. Scientific JSON/config/input/provenance checks and rq1_summary remain separate required runbook work.

Acquisition selects only the exact result, actual runner log, and nine named runner artifacts including meta.json. It does not traverse unrelated content, inspect processes, execute the recorded command, write remotely, or start/stop jobs. Per-file bytes are limited to 8 MiB and accepted aggregate original bytes to 32 MiB. Stable regular-file reads compare held fd identity/size/times, rewalk no-symlink directory identities, and check the named leaf. The final metadata comparison adds a collection-wide terminal-state check; it does not establish an atomic multi-file snapshot. The report correctly distinguishes read-time stability from historical source/build/input attestation. No additional historical attestation or global snapshot mechanism is requested here.

Key-only SSH, nice 10, source/command pins, finite controller lifetime, fresh-only local output creation, stdout/stderr/exit retention and no automatic retry belong to the explicitly deferred controller. Their absence from this standalone remote body is not an additional draft-source finding. This review does not authorize that controller or any remote execution. The existing queue remains untouched.

## Evidence inspected and verification limits

Read the complete 187-line source and 89-line test diff once, the collection plan, report, pins JSON and retained test stderr; checked the registered request and actual FULL-1 sibling meta.json/cmd.txt. Local read-only hashing confirmed all four pin entries, and reconstruction confirmed both full added-file diff bodies equal the pinned source/test bytes. No collector module was imported or executed, no tests rerun, and no server/SSH/native/scientific action occurred.

Inspected pins:

- Plan: 4,202 bytes, SHA256 `e38d8fcd3b9ce383404d723b39f19839b81426d9218e186b3a4a7be52a665a15`.
- Report: 2,641 bytes, SHA256 `13ec285860c8ef63e7066811a89a449d297315e473b9dd511bd4c058a1aadc02`.
- Full diff: 12,108 bytes, SHA256 `992cbbc4469cfbec4b566c8c6f28cd4bc0016732798ac94722a50134238bdc87`.
- Pins JSON: 672 bytes, SHA256 `9f1aa8c1e381365cde46032aa7afc378967dcb21330c8084f5e71b8023e7c7c0`.
- Remote source: 7,647 bytes, SHA256 `00f5184275ff3c7b521756f778a0ba5c86a6eda545814c4b9cfd1bcb40f44718`.
- Tests: 4,029 bytes, SHA256 `c603f948e9b75bd6c01fb8655e86187f7ca6b78b3f50f7bb60a68883f4fa33d6`.
- Test stdout: 0 bytes, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Test stderr: 918 bytes, SHA256 `cf06578ab9bb8186cbfc1c6e627d4b28899edc10a1eefc245b72be12422b66a2`; reports seven tests, OK. That is inspected prior evidence, not a new test result.

Only this review note was written. No source/test/result edits, test reruns, subagents, commit or push. Resolve F1/F2 and review the exact revised operations artifact before considering release; no broader framework or scientific rerun is needed for these corrections.
