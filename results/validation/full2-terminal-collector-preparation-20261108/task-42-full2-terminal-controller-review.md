# FULL-2 terminal controller independent review

Reviewed 2026-11-08 against the terminal collection runbook, accepted collector fix1, controller report/diff/pins and the actual registered observer body and 1306 observation. Scope is this finite one-shot controller's integration and custody. No source/controller/collector execution, tests or SSH were performed by this review.

**SPEC: PASS. QUALITY: PASS for the reviewed operations draft.** No load-bearing finding. This is source review acceptance only: positive invocation, SSH and timeout branches remain untested, and no terminal collection or scientific acceptance is established. Root's review of an actual qualifying observation remains a prerequisite to invocation.

## Source and observation prerequisites

The controller reads the collector once, requires its exact 8,408-byte length and accepted SHA256 `50f2cf22ca3f540a86117c06664c6606f23fd9e2532f72cfa2eaabea168058ef`, and sends those same held bytes through stdin. A later source-path change cannot change the submitted body. The collector source remains the retained, separately reviewed local artifact; launch/final metadata record its length and hash and the literal SSH command. No arbitrary remote source or command is accepted from observation JSON.

The observation prefix is explicitly supplied, restricted to the existing ledger and registered prefix convention. Transport returncode must be zero, stdout hash and byte count must match the held original stdout, and the source hash must equal the actual registered 1,560-byte observer. The output kind and exact FULL-2 result path are checked and result existence must be true. These checks link the selected local records; they do not authenticate arbitrary caller-authored files. The explicit root review of an actual registered capture supplies the operational selection, and the report does not claim a generic provenance service or historical execution attestation.

The actual observer emits the fields the controller consumes. Original chain PID 1700979 and expected start ticks 465769819 must occur exactly once. The controller permits absence, terminal Z/X with the same start identity, or a different numeric start identity showing that the old PID identity is gone. A same-identity live process refuses. Accepting a reused PID does not assert that its new process terminated, and sends no command to it. The separate queue process is not controlled or required to stop by this collector. The collector independently checks the actual original runner metadata, so observation file existence never grants runner or scientific success.

The retained 1306 transport/output pair has matching bytes/hash and exit 0, but reports result_exists=false and the original chain still S with its original start ticks. The recorded check-only refusal is consistent with the exact first failing guard. No inference of later completion follows from this record.

## One-shot launch, retention and timeout

The six fixed output files are all opened in exclusive binary-create mode before Popen. The preliminary exists check is not the sole protection: exclusive creation rejects an intervening file or symlink, and partial reservations are left in place, preventing automatic reuse. Launch metadata and both held observation originals are written and flushed before SSH. The launch copies therefore preserve the actual inputs reviewed by this invocation even if the original observation paths later change.

The command uses the known quadruped-gpu target, BatchMode, disabled password and keyboard-interactive authentication, one connection attempt, fixed connection/keepalive settings, and `nice -n 10 /usr/bin/python3.14 -I -B -`. ControlMaster=no uses the established multiplex path without creating a new master. It invokes only the pinned read-only collector. No process listing/control, experiment, queue mutation, remote write, retry or result-editing command is introduced.

stdout/stderr go directly to the exclusively reserved files. communicate has a 110-second timeout; its timeout handler calls kill only on the Popen object for this new local SSH child and then reaps it. It never targets a research PID, process group or multiplex master. The report correctly says this does not prove remote process termination. Transport metadata retains the actual child returncode, timeout flag, normalized controller status and stdout/stderr lengths/hashes. Timeout status 124, normal remote 0/2/3 and signalled-child normalization remain distinguishable through those fields. A launch/local OS error records its text with default controller status 125; no retry occurs.

The existing collector's refusal envelope and original byte payloads pass through untouched. The controller neither interprets nor replaces scientific JSON. A completed transport remains subject to review of collector status/evidence and the runbook's original config/input/provenance and scientific checks. Fresh raw-result publication is separate from this ledger transport capture.

## Verification and exact identities

Read the entire 130-line new controller diff, report and pins, the actual observer source and 1306 controller/metadata/stdout. Read-only local hashing confirmed all three controller pin entries and the actual observer stdout transport pin. Reconstruction of the full added source from the diff matches the pinned controller bytes. No module was imported, no syntax check or check-only invocation was rerun, and no source or result file was edited.

The report records exec 4ff7a3 running check-only against actual pending data and exiting 2 before output reservations/SSH. Retained stderr is exactly the corresponding refusal; stdout is empty. The report states that only signal-status normalization changed afterwards and that final syntax was checked. That prior check is pending-path evidence only; it is not execution coverage of the final controller's positive launch, exclusive-reservation failures, SSH or timeout behavior. Those limits do not turn source inspection into a tested-branch claim.

- Controller source: 6,461 bytes; SHA256 `decdd1ed38f15b9d84cec1df03f54584309d9e359abbe39f98373246e238d1e6`.
- Controller diff: 6,664 bytes; SHA256 `8821e676c547f8f003e008510d76721cb20c7dac4001d6fd3982b339da79776b`.
- Controller pins: 517 bytes; SHA256 `a5947afc15a5a23e8c71113af37acc4b1c2c8661e3ae7a7b3a89fde58cc9a910`.
- Controller report: 2,246 bytes; SHA256 `4c11e560515f91fea5b4f7ca39eb03122745022528abb3ae5d7de42cc2bf708c`.
- Pending-check stderr: 77 bytes; SHA256 `a919e698e1ee2eee8405ecccfb44a24e0ec015c6f7a6fe0910c6d788f12c5ae2`; stdout zero bytes, SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Actual observer `post-full-observation-20261107-0600.remote.py`: 1,560 bytes; SHA256 `17550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243`.
- Actual `post-full-observation-20261108-1306.stdout`: 2,418 bytes; SHA256 `2ac9b4835e727f594495e508a9d8ce90c6a13a95a2c1de1b537be2e7983c7a8c`.
- Actual `post-full-observation-20261108-1306.meta.json`: 1,203 bytes; SHA256 `92a31d4f5e4518daad4cee4b3c6c003052e3cda25953df3aee45ae86d6f5ecaf`.

Only this review note was written. No source/result edits, reruns, native/SSH/server execution, subagents, commit or push. No controller or collector remote release is performed or implied by this review.
