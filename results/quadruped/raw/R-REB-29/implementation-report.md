# Task54 CONTEXT1 recovery1 — implementation/source freeze

2026-11-07 JST. **SOURCE FREEZE. Ready for independent source review; no remote phase released or executed.** Implemented exactly four new fixed sources and a fixed new manifest. Original sources/controller/candidate, failed-deploy sentinel/receipts, diagnostic receipts and prior maps remain unchanged. Reused the original collector without copying or editing it. Root retains archive/docs ownership and every subsequent action release.

Binding source-only release: `task-54-linux-context1-recovery1-source-release.md`, SHA `6c7bdddcd0037295d2a66b73fcb0048bb8dcf4b9fbf2e88f39e85a50b7de5ace`; recovery-map-v2 SHA `b386f64c99b88749effda9e640836180856b873d3ea18d613f09ab793b71f2ac`. Read both fully before source work, then the required independent diagnostic review SHA `d46f6e423aed374025603ae6a585500c6ff809afbecf45cc8a5c7368f8c23718` and failed-deploy no-write source review SHA `aa151e147083e1b68357c18801e9cf7fda8e7a4c875449a29b852c793556cf3a`.

## Corrected fact

R27's full stat mode33204 normalizes with `stat.S_IMODE` to436=0o664. Diagnostic1 already records normalized permission436=0o664. Their seven comparable recorded fields agree. **No permission difference or server permission-change event is observed.** The original outer0644 expectation came from the wrong decimal-to-octal interpretation. This implementation corrects the expectation to existing retained evidence; it does not chmod E or accept an observed server change. Historical erroneous materials remain immutable with the v2/independent correction.

The source-derived no-write proof for the failed original deploy remains valid; `files: []` alone is not the basis. Its sentinel remains consumed. No new deployment, readback, invoke or collection has occurred.

## Frozen artifacts

| New artifact under L | Bytes | SHA256 |
|---|---:|---|
| `task-54-linux-context1-recovery1-deploy.py` | 417905 | `36264a9166f24e1b3e19e33a0ee37b35ba9ab640b2deafde68ef3860c1b0889f` |
| `task-54-linux-context1-recovery1-readback.py` | 3391 | `1f74a2d74c33795a7a7c6dd2d37d20dd6a92ad1469d3512b27108f379313e099` |
| `task-54-linux-context1-recovery1-invoke.py` | 7754 | `f597d4c3f2f92152578a86e3490b0a4e31f0f7ecdb1f91d89a5b4deac7627029` |
| `task-54-linux-context1-recovery1-controller.py` | 13346 | `6b48d9dd617ffe8010f7e6e1103d8b72020b15b4a14795cf2ab149de8eee3b29` |
| `task-54-linux-context1-recovery1-transport-command-manifest.json` | 5299 | `94e6eaca18287235f638f953ed107a05d5ae71e28fa49d36c02ad7d95dfbd5aa` |
| `task-54-linux-context1-recovery1-transport-source-delta.patch` | 16107 | `1d802a61c6fd6fec8813a7bbb41a35eee4b60e461b46ce4e06e1fc7a2675eb78` |

Reused collector: `task-54-linux-context1-collect.py`,2327bytes/SHA `b886eb5a8d2adc889db46b86b6e1533fe484a7e7097e7be5c9c2e5780c8fc2de`. Candidate remains309421bytes/SHA `6137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e`, DATA only.

The full source+manifest unified diff is `task-54-linux-context1-recovery1-transport-source-delta.patch`; it includes all four original-to-new source changes plus the original-to-new manifest. Generated with difflib; no Git operation. Source differences are exactly two deploy lines, two readback lines, one invoke line and three controller lines. The static verifier reverses only that finite allowlist and requires byte-for-byte equality to each original source.

## Implemented changes and preserved behavior

- Deploy/readback: only genuine E's expected permission changed from decimal420 to literal0o664. E's prior link exemption is now tied to its one exact absolute path instead of0644. Launcher/prelaunch/sidecar remain0600/nlink1. No new E link-count precondition or device/inode pin. E path, uid1000,61243bytes, hash and canonical/regular/stable-read checks remain unchanged.
- Invoke: only the independent E permission check changed from0o644 to0o664. All ten invoke pins, E size, exact payload.paths, eight-entry ENV, cwd, before-base guards, three evidence-absence checks, child1800/cleanup5 and existing prelaunch exec remain unchanged.
- Controller: only its fixed manifest path, fixed four-phase source selection and local receipt prefix changed. Deploy/readback/invoke select new `task-54-linux-context1-recovery1-{phase}.py`; collect selects the exact original collector. Every local phase prefix is `task-54-linux-context1-recovery1-{phase}`. Original consumed sentinels and normal/diagnostic receipts are neither reused nor overwritten. New sentinels remain permanent. There is no arbitrary namespace/command/source/retry option or automatic phase chaining.
- Remote V and sidecar/evidence names remain `context1-release.json`, `context1-parent.json`, `context1-child.json`, `context1-context.json`. Sidecar still uses0600 O_CREAT/O_EXCL/O_NOFOLLOW and exactly the accepted candidate bytes in the existing V. No mkdir, chmod, cleanup, overwrite or new checkout.
- Exact SSH argv, raw-stream custody, deadline/cutoff/error handling and strict local decoder are unchanged. Deadline remains2100seconds; invoke stdout1MiB/stderr32MiB, collection stdout100MiB/stderr32MiB, raw parent32MiB/child8MiB/context32MiB. Exact separate-role base64 maximum100663300 and1GiB memory planning remain unchanged. These are operational cutoffs/planning, not producer bounds or measured RSS. A cutoff kills only the local SSH process, leaves remote terminal unknown and never authorizes a retry.

The fixed manifest records the evidence correction, original manifest hash, exact new source/controller pins, unchanged collector/candidate/command/settings and future local command templates. Later action-release values remain placeholders because no remote action is authorized by this source freeze. Decoder local prefix is `task-54-linux-context1-recovery1-decoded`; remote names do not derive from the new local custody prefix.

## Test-first evidence and scope

Local commands:

```text
python3 -I -B .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-linux-context1-recovery1-transport-tests.py --baseline
python3 -I -B .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-linux-context1-recovery1-transport-tests.py
python3 -I -B .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-linux-context1-recovery1-transport-static-verify.py
```

Before implementation, `--baseline` ran the four evidence-gate tests against original wrappers and failed as intended (10 failing subcases): old gates reject the actual retained0664 and accept the erroneous0644. This is an actual gate regression, not merely a missing-file check. The full new suite was also run before source creation and retained its missing-source red log. After the eight-line source correction, the full suite passed **11/11 tests in0.511seconds**, exit0; static verification also exited0. Logs are retained as `...tests-baseline-red.log`, `...tests-missing-red.log`, `...tests-green.log`; static results are `...static-verification.json`.

The test reads and pins actual retained R27 stdout SHA `2f4894b5dfd0123db5dc531cdea04efe68f68ba2ba08aaa6c8c72a579743e7cd`, selected `checkout.selected_files["tools/export_reboot_block_seed_inputs.py"].stat.mode`, and diagnostic1 stdout SHA `88e9adb7f56ec02e2b309544ea2b0d82ad7ca2d07c0a89b477f7f7669545f7cb`, selected `deployment_files[2].mode`. It normalizes R27 via stat.S_IMODE and uses independent literal0o664. No copied436-only substitute serves as the evidence input.

For each of deploy/readback/invoke, the regression AST-extracts and evaluates the actual production gate expressions with small in-memory identity/stat records: both retained forms pass, old0o644 and different0o666 reject. Reverting any actual E mode gate to0o644 breaks the positive evidence regression. Actual exact-path link expressions accept E with synthetic nlink3 and reject every selected program/sidecar plus an E-like alternate path with extra links. Invoke's program/sidecar and deploy's post-write sidecar gates retain nlink1. Changed owner/hash/size rejection is exercised where those fields are checked. No wrapper main or real remote path is executed.

The seven bounded local-custody tests use temporary directories and harmless local Python subprocesses in place of SSH: all four exact phase/source names and fresh custody prefixes; wrong names rejected before launch; original synthetic consumed sentinel unchanged; one-shot zero result and no second child; nonzero unknown terminal; deadline with retained prefix/unknown terminal/no reuse; stream cutoff with bounded prefix; final retained-file hash after a synthetic short persisted-file failure. No large fixture or generalized retry framework was introduced.

Static checks cover exact allowlisted source scope, unchanged ten pins/ENV/payload, identical candidate and one embedded sidecar, four readback identities with E0664 and the other three0600, unchanged collector/controller budgets/decoder, fixed manifest hashes/command and24 unchanged input hashes. The exact source reversal proves canonical/stable-read checks, before-exec guard ordering and decoder logic are untouched. All four new sources compile without executing them. No actual recovery1 operation receipt exists under the real ledger.

A bounded fresh source inspection also checked immutable prelaunch/launcher's E read path: candidate.exporter contains only path/hash; prelaunch loads the pinned launcher; validate_static and load_exporter_exact call verify_file for E, and verify_file/read_regular enforce canonical/regular/size/hash (optional owner) without an E0644 permission gate. The separate directory/venv layout mode checks remain unrelated and unchanged. No immutable target/helper module was imported or executed.

## Preservation and limits

`task-54-linux-context1-recovery1-transport-input-pins-before.json` and `...input-pins-after.json` are identical24-input records: original five sources, original manifest/tests/candidate, immutable launcher/prelaunch/exporter, source release/maps/reviews, R27 receipt and original deploy/diagnostic custody. `task-54-linux-context1-recovery1-transport-freeze-verification.json` inventories the four new sources, manifest, complete delta, tests, logs, static evidence and before/after pins.

No NN/tracked-source/history/memory edit, Cargo/HE, Git operation, target import, SSH/network, remote mutation, deployment, readback, invocation, collection or subagent was performed in this recovery implementation. Root archive/docs work is independent and not claimed as this author's work. Local gate/custody tests are not Linux execution, present remote-state verification or context-content validation.

Independent source review remains required. Each later remote phase requires a separately concrete root action release naming new pins and fixed custody. Before any invoke, successful independent readback must join all four identities with deploy records and confirm all three remote evidence paths absent, and the full timed-job gate remains in force. Any mismatch/partial/unknown outcome preserves evidence and stops the chain. Complete transport/parent success does not validate the separate context file: exact child hash, both release bindings, purpose, terminal/streams/cleanup and actual context content still need review. Capture/scientific/full encrypted-block holds remain unchanged.
