# R-REB-20–23: startup2 operational evidence

These are original operational records and frozen source inputs, not scientific result JSON or training measurements. Files were copied byte-for-byte from the local controller ledger; `SHA256SUMS` binds the archive. No recorded value was rewritten.

- R20 deployed four fixed files (111,477 bytes) into the previously absent `/home/kmihara/hekit-runs/task54-launch/v2` at 2026-11-07 07:52:08 JST. Existing v1/startup1 was preserved.
- R21 independently rehashed/read all four files at 07:53:12 JST. Complete file path/size/hash/uid/mode/device/inode records equal R20. The common parent matches its R17 identity; v2 directory inode 36309263 on device 66311 was first observed here, not matched to a nonexistent deployment directory observation. Both evidence paths were absent.
- R22 used the reviewed fixed invoker, explicit system observer and pinned base controller. Its lexical venv child PID1963722 terminated with exit0 at 07:53:48.768885 JST. Parent reports complete streams and no cleanup errors. Parent/child release/argv and child-file hash match.
- R23 collected both exact files without retry: parent2755 bytes SHA256 `9d491997ed1476c98f5e1847354d30e344cb54bbb1f4ee2ecf480c16f71b5fc1`; child98031 bytes SHA256 `a1fbe98aa390b7305f7189dabf574c4d28b2422617bfeb010e261f2c82fd1f41`.

The child records exactly startup and helpers stages. Both identify CPython3.11.16, the intended venv prefix and base executable, isolated mode, disabled user site and bytecode writes. The exact v2 launcher importer-cache key is present with null in these two actual snapshots. This does not retroactively establish the unrecorded startup1 finder value. Startup has77 modules; helpers has162 after trusted preparation. Counts are snapshot observations, not proof of complete semantic or native dependency closure.

Purpose remains `startup-only`, with status `observation_only_not_runtime_approval`: no external-package append, exporter loading, scientific context/capture, seed export, training or GPU computation occurred. No FULL2/queue poll or mutation occurred in these operations. The original startup1 failure remains separately archived. Source-only review, actual startup success and any future scientific compatibility gates remain distinct.

The archive includes exact standalone payloads, candidate sidecar, wrappers, manifests, pre-action release records, source review and controller stdout/stderr/metadata. Package-manifest names are ledger-relative originals; the archive SHA256SUMS is the archive-local inventory. Preparation-time pending statements describe their earlier stage, not the later observed outcome. Direct-child60s/cleanup5s limits do not establish descendant containment; platform preparation may invoke the observed uname executable and ordinary trusted fallbacks. No native isolation or complete import closure is claimed.

The [independent actual-evidence review](task-54-linux-startup2-actual-review.md) accepts these bounded observations. It leaves context/capture and full helper/native closure unapproved.

Current-source clarification: the tracked exporter at source commit2fa380f and the current local file both hash to `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce`, exactly matching the preserved deployment payload. The preparation report's statement that live exporter development had changed is not evidence of a current byte difference. Its original bytes are preserved as historical preparation, with this explicit clarification; no exporter substitution occurred.
