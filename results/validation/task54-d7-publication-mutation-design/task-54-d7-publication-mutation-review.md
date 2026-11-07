# Task54 D7 publication mutation design independent SPEC + QUALITY review

Reviewed 2026-11-07. **SPEC: PASS. QUALITY: PASS.** No Critical, Important or Minor findings. The proposed mechanism is acceptable as a design with the explicit implementation/native proof obligations below. This is not a production source release, a passing native experiment, or confirmation that the current publisher's mutation defect has already been fixed.

## Scope and exact reviewed artifacts

Read the complete mutation design brief, full proposal/report and complete added-file diff. Reviewed the shared publication/target boundary, accepted custody inventory FIX1 and actual locked dependency sources. The requested shared scope includes Reference, Diagnostic, Failure, Target and future D7 publication; a D7-only adapter is not the accepted remedy.

- Proposal/report,35848 bytes: `714880b778511c3be75d1ac0e6712e77e24d2564ba52f61cdb36f9da288509f3`.
- Complete diff,36283 bytes: `ecb38f613e453ebe61319d343412b5d89f5490e674b12189645b5c79d2241e5d`.
- Current publication source: `8cfabba4dcf7daae93e90ce40a0bc1bcf13c59c713bb5c9d65964b97a4a2c004`.
- Current publication tests: `10f640a0961da262ddcd15627a5a1c59455b1fddbee7cc5c9a4711b6eff33fd6`.
- Bench manifest: `189d5d50799949561ccf6ff6696b7f5b89e1aa860d9f88a641abddca9e096d98`.
- Cargo.lock: `7250dbf896c207b4e838cabea88c5a4110e46803e3c8e8fe6460f8a7f60c97b1`.

Independent read-only inspection confirmed all four source/dependency pins, exact diff contents and operation/count arithmetic. The concurrently edited codec files were not reviewed or modified as part of this task.

## Mechanism and primary-source assessment

The anonymous-inode choice addresses the original mutation defect rather than narrowing away the named-entry actor. Linux documents O_TMPFILE as unnamed inode creation and permits later linking when O_EXCL is absent; its documented proc-FD alternative avoids the capability needed by AT_EMPTY_PATH. Existing final names are not overwritten by linkat. The proposal uses the correct SYMLINK_FOLLOW behavior and never reads a textual symlink target back into a pathname. [open(2)](https://man7.org/linux/man-pages/man2/open.2.html), [link(2)](https://man7.org/linux/man-pages/man2/link.2.html).

The procfs capture sequence is coherent: fixed nofollow `/proc`, procfs validation, then held-root `openat2("self/fd")` with BENEATH/NO_XDEV/NO_MAGICLINKS. Linux's `self` implementation returns an ordinary relative PID symlink, so allowing ordinary symlinks while rejecting magic links during this directory capture is intentional. The later numeric FD link is deliberately followed only after capture, through the held directory. openat2's Linux5.6 API floor and unsupported/restricted-host refusal are openly acknowledged; no weaker fallback is authorized. [openat2(2)](https://man7.org/linux/man-pages/man2/openat2.2.html), [Linux v6.12 proc self](https://raw.githubusercontent.com/torvalds/linux/v6.12/fs/proc/self.c).

The mutation source is the held File's kernel FD entry. The kernel proc FD-link implementation obtains the referenced file path from the task FD table. A normal temp-name replacement cannot redirect that reference while the correct descriptor remains owned and open. This is distinct from the preliminary inode comparison, which is only a consistency check. The proposal explicitly excludes arbitrary in-process raw-FD close/dup2 and privileged mount injection from its proof while retaining the original named-entry actor. [Linux v6.12 proc FD lookup](https://raw.githubusercontent.com/torvalds/linux/v6.12/fs/proc/fd.c).

Ownership rules are appropriately strict: derive the decimal FD only within a helper retaining the File borrow throughout the syscall; keep original File and proc directory through publication and late checks; do not rely on DecInt itself to retain a borrow. The shared PID check refuses inherited fork use, and D7 retains stronger process/context/thread checks. CLOEXEC is not confused with fork exclusion. Proc access/main-thread-lifetime failures refuse rather than weakening capture. [Rust OwnedFd](https://doc.rust-lang.org/std/os/fd/struct.OwnedFd.html), [proc_pid_fd(5)](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html).

The six failure/cleanup paths now drop an anonymous owner instead of mutating a temp pathname. The actor sentinel correctly represents X/R, never pretends to be the unnamed O, and remains subject to unrestricted replacement. No publisher unlink/rename is needed. Pre-link drop or process exit cannot create a final; after a successful link, drop cannot remove that final. Close errors and delayed last-reference reclamation are not misreported as a cleanup receipt or immediate storage-reclamation proof. [close(2)](https://man7.org/linux/man-pages/man2/close.2.html).

## Rust API, feature, MSRV and license assessment

Read installed rustix1.1.6 Cargo.toml/build.rs and actual fs/openat2.rs, fs/at.rs, fs module exports and path/dec_int.rs. The safe open/openat/openat2/fchmod/fstatfs/linkat APIs and typed flags exist; DecInt::from_fd returns a formatted value rather than lifetime ownership. The Linux API export is available for the normal Linux configuration with the proposed `std`/`fs` features. No proposed helper requires a new project unsafe block or raw-FD ownership constructor. Safe OwnedFd→File conversion is appropriate at workspace Rust1.85.

Cargo.lock actually contains rustix1.1.6 with the stated registry checksum. Its declared MSRV1.65 and MIT alternative match the proposal; selecting MIT avoids needing the LLVM exception. A new direct Linux normal dependency and lock dependency edge still require source/build provenance review even though the package version is already locked. Feature unification and final lock/build results remain source-verification obligations; no build was inferred from manifest inspection. [rustix1.1.6 manifest](https://raw.githubusercontent.com/bytecodealliance/rustix/v1.1.6/Cargo.toml), [safe openat2 source](https://raw.githubusercontent.com/bytecodealliance/rustix/v1.1.6/src/fs/openat2.rs), [safe at APIs](https://github.com/bytecodealliance/rustix/blob/v1.1.6/src/fs/at.rs).

Primary Linux/kernel sources were inspected for semantics only. No GPL or other external implementation was copied into the repository. `#![forbid(unsafe_code)]` in the new private helper module is a concrete project-code constraint, not a claim that std/dependencies or the existing parent module contain no unsafe code.

## Visibility, durability and legacy compatibility

The five-operation sequence C/W/F/L/D is sound for the selected mechanism. Before successful L, failures remain BeforeLink and drop only owned descriptors. L must return success immediately after the successful link: late checks cannot be folded into L and misclassified. D or later final/anchor/context/clock failures are VisibleUnconfirmed and preserve visible finals. The proposal correctly adds final inode/anchor confirmation to the shared Store path, while retaining Target guard/source confirmations.

File sync precedes link and parent sync follows link. Removing temp unlink also removes the second namespace mutation that required a second parent sync; directory-creation and recovery ancestor syncing remain. The review accepts this software fsync boundary, not power-loss or storage-device behavior proof. [fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html).

Recovery does not require procfs capture or anonymous-creation support. Existing recognized temp grammar/caps, unknown-entry rejection, nlink2 historical final/temp states, Reference profile-empty rules, Train existing-only behavior and recovery permissions stay intact. Named remnants remain immutable inspection fixtures; none is adopted or cleaned. A new writer's lack of a temp does not reset an orphaned named attempt or permit Initial/Quick Retry. Every shared production writer migrates together, preserving formats, caps and canonical goldens.

## Exact test-contract deltas and remaining proof

The proposed numbers are internally consistent and openly replace old counts:15×6 primitive rows=90; six actor scenarios×15=90; legacy prefixes15×65=975; C substeps8×15=120 plus one explicitly separate ownership-transfer scenario; file crash cuts15×5×2=150; before/after c5/c6 cuts4×15=60. ENTER becomes30 Ops, START/END5 each, Full35, Quick40; R/H/G and directory accounting stay separately defined. Shared four-caller additions correctly total24 fault cases,24 actor cases,4 no-clobber cases and260 legacy-prefix controls.

The native minimum totals21 cases per actually available supported mount (1+3+6+2+4+2+2+1). It uses bounded scratch payloads, ordinary unprivileged execution and coordinated barriers, with injected errors distinguished from real OS failures. ext4 evidence cannot silently become XFS evidence. Existing guard/fork/mount tests remain separate, and no native scratch test can construct genuine gates or establish a gate→begin lifecycle.

Before claiming remediation complete, source review must verify the actual shared call graph, no surviving named fallback/cleanup path, retained proc/File borrows, changed-process refusal, final inode confirmation and error mapping. Compile/target/feature/license/lock checks and revised deterministic tests must run under their authorized source inventory. Separately bounded native evidence must demonstrate the unprivileged proc-FD link, exact inode/no-clobber behavior, absence of publisher temp mutation, failure/drop behavior and late-final replacement refusal. These are mandatory pending obligations, not extra design findings or a request to reapprove already-authorized ordinary source work.

No Cargo/compiler/tests/--list/native/runtime/SSH/HE/generator/subagents/commit or production/public-doc edit occurred. Tools performed read-only local inspection, primary-document browsing and file/hash/arithmetic checks; only this review was written. Current authority, all21/full2680+20+7/quick268, holds, original science/security thresholds, baselines/Task5 and MIT/Apache2/BSD/ISC plus Unicode-tables-only exception remain unchanged.
