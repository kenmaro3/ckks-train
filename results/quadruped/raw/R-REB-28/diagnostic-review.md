# CONTEXT1 diagnostic1 — independent actual-evidence review

2026-11-07 JST. **Material correction: diagnostic1 E permissions match retained R27; there is no observed permission-bit difference between these two receipts. R27 full stat mode33204 is0o100664, whose permission bits are0664 (decimal436), not0644.** All seven comparable recorded fields agree. The frozen0644 expectation rests on an incorrect decimal-to-octal interpretation in the preparation material. This report corrects that factual interpretation; it does not accept a future mode-gate change or release recovery.

This is a failed-partial diagnostic, not deployment success or a passed admission/readback gate. No cause, actor, permission-change event, change time or repair is inferred. I inspected the exact diagnostic release/stdout/stderr/metadata, compared R27's selected-E record, rehashed the original five sources and checked normal readback custody paths. Only this report was written. No SSH, tests, target imports, source edits or remote changes occurred. A read-only comparison assertion expecting the difference described in root's handoff failed; correcting the numeric interpretation produced exact agreement.

Diagnostic action release is2000 bytes / `df5156a8daf7f968d0aefe8a53565c720619e2e58585cea27dfcd2cdf240b6e6`. It authorizes one read-only diagnostic with unchanged readback source `efac95957b5b7a6d02d44a5cc4fea8122a1f7d7569066eba5ea472fcde06cb75` under distinct custody. Actual metadata binds that exact release and3305-byte source, and its SSH argv exactly matches the frozen manifest. Stdout is1257 bytes / `88e9adb7f56ec02e2b309544ea2b0d82ad7ca2d07c0a89b477f7f7669545f7cb`; stderr is empty / `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Byte counts and hashes match terminal metadata:1312 bytes / `9310aa452dd3558cbc8bd37815b626e12a311c1449567615d139db62e560a931`, returncode1, automatic_retry=false. The release itself repeats the historical0644 interpretation; its exact hash preserves that original statement rather than making it correct.

R27 comparison source is `results/quadruped/raw/R-REB-27/remote-create-readback.stdout`,246858 bytes / `2f4894b5dfd0123db5dc531cdea04efe68f68ba2ba08aaa6c8c72a579743e7cd`, at `checkout.selected_files["tools/export_reboot_block_seed_inputs.py"]`. Diagnostic source is `deployment_files[2]`.

| Field | R27 retained actual | Diagnostic1 actual |
|---|---|---|
| Path | `/home/kmihara/hekit-exporter-010bb3c-task54-v1/tools/export_reboot_block_seed_inputs.py` | Same |
| Bytes |61243|61243|
| SHA256 | `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce` | Same |
| UID |1000|1000|
| Device |66311|66311|
| Inode |36310811|36310811|
| Permission bits |0664, S_IMODE(33204)=436|0664, recorded permission value436|

Independent numeric checks: `oct(33204) == "0o100664"`; `stat.S_IMODE(33204) == 436 == 0o664`. Conversely,0644 permissions equal decimal420, and a regular-file full stat mode100644 equals decimal33188. The group-write bit0020 is present in BOTH retained observations. It is the difference from the frozen expected0644, not evidence of a change between R27 and diagnostic1. Diagnostic1 does not record ctime/mtime, gid, ACLs or a change actor. No conclusions about those fields or the intervening history follow.

Actual stdout retains two pinned program records, then E's actual record, followed by status `failed_partial` and RuntimeError `deployment identity drift`. In unchanged readback line23, the actual record is appended before the expected-identity assertion. E passed preceding canonical/regular/bounded stable-read checks. Its actual permission value436 is the only mismatch against the source's expected record (mode420); bytes/hash/uid/path match. This explains diagnostic1's rejection. It is consistent with the earlier aggregate deploy failure, while not retroactively measuring all E attributes at the exact deploy instant. The diagnostic source contains no chmod/repair operation.

The sidecar record was not reached, deployment_directories is empty and evidence_absent is absent because those checks occur later. The diagnostic therefore supplies neither a current sidecar/evidence-absence check nor a complete directory/admission join. Normal `task-54-linux-context1-readback.{launch.json,stdout,stderr,meta.json,cutoff.json}` were all absent at this local review, consistent with distinct diagnostic custody and no normal sentinel consumption.

All five original sources still match reviewed pins: deploy `f0e747cd7c99b92cee6b3dedcf3b9499b059fdc000bc6330b27a5219cb9ef523`; readback `efac95957b5b7a6d02d44a5cc4fea8122a1f7d7569066eba5ea472fcde06cb75`; invoke `721ba884db49410dc41158c61d15172ff5b2f255e8ef20a7f4b5442ef4aa32d0`; collect `b886eb5a8d2adc889db46b86b6e1533fe484a7e7097e7be5c9c2e5780c8fc2de`; controller `c5c4e8d428d0895cf8f1220a285eb20848e4fb39d95fcb6b8dad52eedd7224bd`.

This finding supersedes the historical0644 interpretation carried through the preparation map, original source review and later release/handoff language. It does not negate the earlier control-flow proof that deploy stopped before writing. Preserve the immutable originals and this correction, do not automatically retry or chmod, and require a concrete root recovery proposal before any source/gate change or remote mutation. No new-source acceptance is given here. Context/capture and the full timed-job gate remain unchanged.
