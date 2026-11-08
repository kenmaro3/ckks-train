# FULL-2 terminal collector fix1 independent review

Reviewed 2026-11-08. Scope: prior F1/F2 and new breakage introduced by the exact fix1 source/test diff. The collection runbook and original review remain the requirements; this is a local operations-draft review, not controller approval, server release, terminal observation or scientific acceptance.

**SPEC: PASS. QUALITY: PASS. F1: ADDRESSED. F2: ADDRESSED.** No remaining load-bearing finding in the reviewed correction. No tests were rerun.

## F1 — ADDRESSED: refusal preserves acquired evidence

The capture envelope and its shared files/missing collections now exist before the initial metadata read and parse. The local take helper inserts each successful bounded stable read before any semantic parse/comparison. A refusal returns that same envelope with exit_code=2 and a structured stage/path/type/message error. Initial active/malformed metadata therefore retains its exact original bytes, and identity refusal stops further acquisition. An offending cmd.txt/workdir.txt is retained before its check; later IO errors preserve earlier files. Non-object metadata explicitly raises ValueError instead of escaping through AttributeError.

Final metadata uses the distinct `meta.final.json` observation key while retaining its actual remote_path. Both original byte observations survive final-byte/stamp disagreement, and the envelope remains operational_collection_refusal without scientific acceptance. Missing files remain explicit; neither missing nor unsafe reads invent original bytes. The remaining aggregate allowance is now passed into read_stable and checked against the held file size before payload allocation; the second metadata observation also consumes budget. This preserves the existing per-file/aggregate bounds and can legitimately refuse when there is insufficient room for the final observation.

The added local cases exercise active/malformed/non-object initial metadata, offending command retention, later IO refusal and both final metadata observations. They use mock operational captures, not real runner provenance or scientific results. Static inspection also confirms the analogous workdir refusal follows the same retain-before-check order. No unrelated evidence/decoder framework was added.

## F2 — ADDRESSED: nonblocking leaf type check

The leaf open adds O_NONBLOCK to the existing O_RDONLY/O_NOFOLLOW flags, followed immediately by held-fd regular-file and UID validation. An own FIFO without a writer can now reach the refusal rather than block waiting for a writer. Directory no-follow walking, bounded regular-file reading and identity/stamp rechecks remain unchanged. F1's handler records the error and preserves earlier successful captures.

The added FIFO case uses a separate local Python child with a two-second timeout and asserts a nonzero result containing the regular-file refusal. Retained RED evidence shows that child timing out before the fix; retained GREEN evidence shows the case passing. This establishes coverage for the reported FIFO issue, not an arbitrary-filesystem latency guarantee. The future controller still needs its already required finite lifetime and must not control research processes.

## Regression scope and evidence

Read the full fix1 diff and report's appended Fix round1. Read-only reconstruction of the unified source/test diff from the preserved initial files to current files exactly matches fix1.diff. All six declared source/test/RED/GREEN byte-length and SHA256 pins match. No collector/test module was imported or executed by this review.

The correction does not change the fixed remote path inventory, expected command, job/workdir/terminal checks, read-only behavior or scientific_acceptance=false. Successful captures still derive runner_success from exit/watchdog metadata and exit 0 only when required artifacts are present; failed terminal or missing required-artifact captures use exit 3. Refusals use exit 2 and retain evidence. Result payloads remain opaque original bytes. Adding a second metadata observation is an operational capture distinction, not a result edit or an atomic-snapshot claim.

Retained prior test evidence: RED stderr reports 12 tests in 2.029 seconds and eight errors including subcases and the FIFO timeout; GREEN stderr reports 12 tests in 0.061 seconds, OK. The report identifies RED exec ee3164 exit 1 and GREEN exec c44607 exit 0, with tests added before the fix. These are inspected recorded outcomes, not a new execution or remote validation.

Exact reviewed identities:

- Remote source: 8,408 bytes; SHA256 `50f2cf22ca3f540a86117c06664c6606f23fd9e2532f72cfa2eaabea168058ef`.
- Tests: 7,831 bytes; SHA256 `95d0ad7c9adc04aa9e753b84453a6da326bc6682c12331acd3e626b05b24b587`.
- Fix1 diff: 11,091 bytes; SHA256 `7014c8b9639b9aaba81a4cd89c0706eacab2483e8700b0474f4478dab56611a1`.
- Fix1 pins: 1,027 bytes; SHA256 `f49cf237c78f3f203bad282b4ccc5fadaf76f8c69dc41065d809dfe5445832bd`.
- Appended report: 4,600 bytes; SHA256 `26b869e850ddd7bfc17e4fdbc802a654148020198296ad7dbc777d7b484a6476`.
- RED stderr: 14,346 bytes; SHA256 `97599cc4cc217327db97fdc4fbd0d7b372fb00bfa75859e35baeb999a1dc10ce`.
- GREEN stderr: 1,507 bytes; SHA256 `385bdbee219604bbe4a75999e57a029ff9d02046c973ec425b4b90626c8c66f0`.
- Both RED/GREEN stdout files: zero bytes; SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Only this review was written. No source/test/result edits, tests, SSH/server/native execution, subagents, commit or push. The future key-only SSH/nice-10 controller, fresh local output custody, terminal stdout/stderr/exit retention and separate provenance/scientific checks remain unreleased work; this scoped PASS does not authorize or claim them.
