# R-REB-11-CREATE-EMPTY-VENV

**Operational preparation, not a scientific result or capture.** On2026-11-07 at06:08:36 JST, the released wrapper created `/home/kmihara/reboot/exporter-venv-task54-v1` once with the observed base CPython3.11.16, `-I -B -m venv --without-pip --symlinks`, nice10 and a fixed replacement environment. The original scientific venv was not modified. No package was installed and no interpreter under the new venv was started.

Both child and controller exited0. Child terminal state and complete streams were observed, with no warnings, output, timeout or cleanup errors. The child ran for about0.028s, the wrapper0.083s and the controller/SSH interval9.690s; these are operational provenance, not benchmark measurements. Its saved proc-stat sample records nice10 and one thread at that instant, not a continuous peak.

Fresh preflight rechecked the six base executable/hook/site/venv hashes, exact `.pth` inventory, both base-side pyvenv.cfg absence, fixed customization/zip candidates, owned canonical parent and absent destination. Static postcheck accepted exactly15 entries (six directories/five files/four links), an empty own site directory, three interpreter links resolving to pinned base SHA2562874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174 and lib64→lib. New pyvenv.cfg is396 bytes, SHA2564af48d1b0d9fea3094d3c725918c55e52670536de714c30660f3fb37ec13c643, with system-site inclusion false. Exact config bytes are retained as hex in the raw record and independently rehashed.

The operational wrapper's initial static review found two issues: child collection could be skipped after an exception, and tree traversal/content reads preceded the intended bound/layout checks. FIX1 preserved the original, added guaranteed owned-child lifecycle handling with bounded collection, and validated only the fixed shallow layout before reading content. Eight isolated helper tests passed before the scoped re-review and release. The original unexecuted wrapper and both source-review reports remain in the local ledger; the published wrapper is the exact reviewed and executed FIX1 source.

## Custody and use

Eight files are byte-exact archived copies: released wrapper, isolated test program/log, raw operational stdout JSON, empty stderr, controller metadata, phase release and independent actual-record audit. README and SHA256SUMS are separate publication metadata. JSON roles are explicit operational records, not binary-emitted training results. No result value or original output was normalized or edited.

The helper tests extract only four definitions from the sibling wrapper source and do not execute its remote top level. To repeat that local helper check from this directory:

```sh
python3 -B task-54-linux-create-empty-venv-fix1-tests.py
```

They create tiny original local Python child snippets and temporary fictional layouts. They do not recreate the remote environment or prove its present state. **Do not rerun the creator wrapper into the existing destination.** Its single-attempt release has ended, and preflight rejects reuse.

The independent audit accepts static creation only. Normal base-interpreter bootstrap is trusted; `-B` does not force source-only reads, and the exporter's source guard was not exercised. Before every later new-venv child, recheck config/interpreter/own-site/startup paths and bind the exact launcher command/source/environment. Actual prefix/path/module/hooks, package/native inventory, isolation, consumer analysis, context, archive provenance and exact six-comparison compatibility remain separate unmeasured gates. FULL-2 and its queue were not changed by this operation; this record is not a new queue observation.
