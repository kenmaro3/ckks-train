# Resource producer and retained typed input

Status: DESIGN RATIFIED, 2026-11-08. Root accepts proposal `e72b99a7` after full read and independent scoped SPEC/QUALITY PASS `2474c2f7`. The [proposal, initial findings, FIX1 and dependency map](../results/validation/task54-d7-resource-design/README.md) preserve the exact decision. No physical model, native behavior or positive execution gate is validated by this design acceptance.

One private resource owner must precede context expansion, canonical input acquisition and typed materialization. Train and nested reference/target operations share it. Standalone reference read, recovery, snapshot and target operations use their actual preparation-only work selection, without inventing an HE request. Large verified snapshots remain inside the resource lifetime. The existing caller-supplied input and the independently reconstructed reference/selected-seed receipts still must agree.

## 1. Ordinary-user cooperative exclusion

Use one stable per-host lock under the trusted account anchor, shared across participating checkouts and campaigns. Its exact location, owner/modes, establishment and identity checks are frozen in the proposal. No sudo or administrator-installed `/run` inode is required. The earlier proposal's Unix-root requirement was unnecessary for the existing cooperative claim and contradicted the task's operating constraints.

This lock serializes participating runners; it does not reserve RAM against other users or resist arbitrary malicious same-UID replacement. Independently trusted host/global-cgroup origin remains separate evidence. Missing origin cannot be replaced by a hostname, visible `/`, equal namespace identifiers or a worker-authored flag. A multi-claim registry would add reservation and recovery complexity without helping the required first serial path.

Publication/profile guards are released during HE; the distinct resource lock remains through the work it covers. This narrowly refines custody's earlier “no flock held” wording. Lock order is resource first, then short publication guards. Nested operations never acquire another resource lock.

## 2. Bootstrap and actual capacity

The small fixed bootstrap anchor is acquired and verified before any capacity decision relies on it. A fixed-scratch probe then selects supported kernel/allocator/filesystem profiles and checks capacity. Only afterward may the larger policy/model/context be parsed under the compiled bootstrap envelope. Full policy bytes must match the retained anchor's identity. The existing allocating process/context parsers and Python helpers are not the minimal bootstrap implementation.

The observer checks actual host memory, every applicable cgroup ancestor, filesystem capacity and live claim state. A proved global cgroup-v2 structural root has no fabricated zero-usage row; hidden ancestors, missing files and unknown mappings remain missing evidence. This clarifies the existing cgroup list semantics without changing the outer Resource bytes. Native namespace/procfs/cgroupfs behavior still needs validation.

All22 applicable allocation components require source/lifetime bounds and applicable validation, including raw plus decoded arrays, independent reconstructions, parser/hash buffers, helpers, allocator/stacks, retained evidence and failure paths. Requested payload is not RSS. No guessed margin, unknown-as-zero term, provisional final identity or post-allocation permission is introduced. Numerical bootstrap/model bounds remain unfinished.

## 3. Helpers and teardown

The actual runtime hash-file/snapshot Python/Git tree needs the same lock description while it performs covered work. The proposed narrow Rust stdin/Python duplicated-FD/pass_fds route keeps exclusion through parent death and allowed descendant lifetime. It creates no child authority to resume the parent's frame. Exact trusted helper/Git behavior must prohibit early close/unlock and be validated with bounded crash/competitor controls; arbitrary Git behavior is not presumed safe.

Parent-only RAII or waiting on the normal success path is insufficient. Rewriting all provenance logic or adding a daemon would cost more unrelated work. The selected inherited hold requires later source/build renewal and native verification. No helper changes or crash tests are executed by this design decision.

## 4. Required implementation sequence

Freeze independent literal vectors for the new closed bootstrap and source-record formats before encoder implementation. Existing outer Resource/Requested/Requirements/Observation/Claim/FileRef and Full/Quick formats and caps remain unchanged. Trusted policy selects finalized model/validation bytes after the actual build, avoiding a source/model hash cycle.

Then wire the actual runtime, standalone/nested readers, captured preflight and typed loaders to the SAME retained bytes and resource owner. The current path still validates files and reopens them in the loader. Merely adding unused parsers or leaving all public wrappers refused is an intermediate state, not completion. Preserve numerical preprocessing, splits/RNG, order/init/rates, F32 conversion and complete source identity. The later retained-store Rc, original/cloned guard FDs and mount checks at `d1aa3af` must be included in the actual model; the earlier map's counts do not cover that delta.

Model calibration, Linux observer/lock/guard/helper tests and scientific experiments remain separate bounded work after FULL2. This design does not authorize new server operations. All21 profiles, Full2680+20+7 and Quick268, CPU/GPU/baseline/external requirements, correctness/security thresholds and existing encrypted-block hold remain unchanged. CPU evidence cannot stand for GPU fit or matrix completion.
