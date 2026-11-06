# Task54 startup2 preparation — frozen, not released

2026-11-07 JST. Local preparation only. FIX3 independent scoped source acceptance is pending at preparation time. Neither deployment nor startup execution is released by this package. Every file created for this work has a new startup2 ledger name; existing originals and startup1 artifacts were not modified. No source/wrapper execution, SSH, remote observation, Git mutation, Cargo/build, scientific import, subagent or FULL2 poll occurred.

## Four exact deployment byte strings

Candidate destination: `/home/kmihara/hekit-runs/task54-launch/v2`. It has NOT been observed absent in this task: the deployment wrapper must prove absence immediately before creating it. Existing `/home/kmihara/hekit-runs/task54-launch` is required to match saved R17 metadata uid1000, mode0700, device66311, inode36309255. Its parent retains the reviewed identity uid1000, mode0775, device66311, inode36072090. No mkdir, chmod, cleanup or file write targets v1 or its files.

| Remote basename | Local exact original | Bytes | SHA-256 |
|---|---|---:|---|
| `task-54-exporter-linux-prelaunch-v1.py` | `task-54-linux-startup2-prelaunch-original.py` | 7044 | `c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0` |
| `task-54-exporter-linux-launch-v1.py` | `task-54-linux-startup2-launcher-original.py` | 29357 | `17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c` |
| `export_reboot_block_seed_inputs.py` | `task-54-linux-startup2-exporter-original.py` | 61243 | `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce` |
| `startup2-release.json` | `task-54-linux-startup2-candidate-release.json` | 13833 | `ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd` |

Total payload: 111477 bytes. The two program basenames deliberately retain their established `-v1.py` API names; remote directory v2 and pinned launcher bytes distinguish this deployment. These standalone originals are review copies, not local invocation targets. The deploy wrapper encodes exactly these four byte strings as base64 and validates each length/hash before any write. The local verifier decoded all four from its AST literal and compared bytes to the standalone copies without executing the wrapper.

The prelaunch and exporter originals were recovered from the exact verified startup1 deployment payload. The frozen exporter is copied unchanged even though separately owned development work has since changed the live production file. No current development exporter bytes are substituted. The launcher is an exact copy of the frozen FIX3 original, whose source acceptance is pending. The candidate record retains the same fixed schema and every reviewed R10–R15 input except the six leaves below.

## Exact narrow changes from startup1

1. Sidecar: three deployed input paths (`programs.prelaunch.path`, `programs.launcher.path`, `exporter.path`) change from v1 to v2. Two evidence paths change to v2/startup2-parent.json and v2/startup2-child.json. Only `programs.launcher.sha256` changes content identity, from FIX2 to FIX3. Recursive structural comparison found exactly these six changed leaves.
2. All 35 helper pins, base/venv/config/layout/root_mode509, interpreter lexical path, package root, startup paths/customization/absence inventories, base bootstrap pins, cwd, exact replacement environment and 60/5 deadlines remain identical to the reviewed startup1 sidecar. Purpose remains `startup-only`, payload `{}`, context evidence null, all six stage pins null for observation only. There is no external append, exporter code loading, context/capture argument, override or full-closure claim.
3. Deployment: fixed payload, v2 destination and startup2 sidecar name change. The now-existing task54-launch directory is validated against actual R17 metadata rather than required absent/created again. Only v2 is fresh-created. Both v2 and its new parent/child evidence names are checked absent before writes. Existing O_EXCL/O_NOFOLLOW,0600,single-link,hash,fsync,no-retry/partial-preservation behavior remains. Parent identity checks, system observer pin and no-concurrent-writer boundary remain. Mode0600 is owner-writable; no filesystem-enforced immutability claim is added.
4. Invoker: exact text substitution only for v1/v2 and startup1/startup2 paths, launcher digest and sidecar digest. Ten fixed input pins, cap/uid checks, canonical/O_NOFOLLOW reads, base bootstrap gates, uname identity, environment/cwd, one direct execve and failure evidence semantics are unchanged.
5. Independent readback: reuse the prior bounded four-file byte reader and directory/evidence checks, remove the unrelated FULL2 process/log/queue poll. Pin four exact path/size/hash/uid/mode records. New v2 device/inode values cannot be invented before deployment; readback records them and root must compare them to the separately collected deployment receipt before execution release. This is an explicit pending evidence join, not silent replacement of inode checking by a claim of equality. No output from deployment is executed or treated as a new source pin.
6. Collector: only v1/v2 and startup1/startup2 substitutions. It reads at most2MiB each from the two fixed evidence files, retaining available exact bytes as base64/hash; missing files remain explicitly missing. A successful collection is not a startup verdict and never retries an invocation.

## Exact package inventory

All paths below are relative to this ledger directory. Hashes identify prepared bytes, not installed or executed files.

| File | Bytes | SHA-256 |
|---|---:|---|
| `task-54-linux-startup2-prelaunch-original.py` | 7044 | `c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0` |
| `task-54-linux-startup2-launcher-original.py` | 29357 | `17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c` |
| `task-54-linux-startup2-exporter-original.py` | 61243 | `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce` |
| `task-54-linux-startup2-candidate-release.json` | 13833 | `ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd` |
| `task-54-linux-startup2-deploy.py` | 152920 | `ab1c616bea7230961ffcced5c8706c0b308bf75fe38b6feadc807044e948eccf` |
| `task-54-linux-startup2-invoke.py` | 6485 | `8c313e0716ceb221077ae03491ef9ace8a676b87ea754214b046866013cb3ffc` |
| `task-54-linux-startup2-readback.py` | 3247 | `6a45c62ca7196e771f0166e3afdae1878b9671683fa861009767e259387517a2` |
| `task-54-linux-startup2-collect.py` | 1761 | `e52ff869ecd2d84f395f25f8459df7d3a010e24b51dcec3c1f1f1add6909f08b` |
| `task-54-linux-startup2-deploy-manifest.json` | 6007 | `3d6246af15545b4147d7da0549acebffafb73dcd0aa45d29b8d7a1019b792c07` |
| `task-54-linux-startup2-static-verification.json` | 2085 | `9cd0b140e074ba0e315f7f7a6741fa091621a7b2222552a78b87ad129dff6c10` |

`task-54-linux-startup2-package-manifest.json` additionally binds this preparation report and all ten files above, excluding only itself to avoid a digest cycle. The deployment manifest records the four local-to-remote byte mappings and all27 unchanged startup1 input-file hashes. `static-verification.json` records the performed AST/hash/literal comparisons and six exact sidecar differences. It explicitly reports `wrapper_execution: false` and `remote_access: false`.

## Review and later action inventory — not execution authority

1. Finish independent scoped FIX3 source rereview and root disposition; review this exact startup2 deploy/sidecar/invoker/readback/collector package. No existing startup1 deployment/execution release applies to these new digests or paths.
2. If separately released by root, run exactly the fixed deployment wrapper under the already accepted key-only observer transport, remote command `nice -n 10 /usr/bin/python3.14 -I -B -`, supplying only `task-54-linux-startup2-deploy.py` bytes on stdin. Bind its exact source and deployment-only release digest in NEW local invocation metadata; preserve terminal status/stdout/stderr, including partial deployment failure. No automatic retry or rollback. Suggested new evidence names are `task-54-linux-startup2-deploy.stdout`, `.stderr`, `.meta.json`; these do not exist as fabricated outcomes in this package.
3. Separately read back with the same fixed observer command and `task-54-linux-startup2-readback.py` stdin, retaining NEW `.stdout/.stderr/.meta.json`. Require four independent rehashes, required directory metadata, absent parent/child evidence, and an explicit device/inode/size/hash/uid/mode join to the completed deployment receipt. On any discrepancy, stop without execution. This wrapper intentionally performs no FULL2 poll. No extra workload poll before08:25 is proposed; root can use its established scheduling gate and decide later if an observation is due.
4. After actual deployment/readback evidence and workload gates are satisfied, root may issue a separate one-attempt startup-only execution release binding the exact invoker source, sidecar and fresh evidence paths. The candidate remote observer command is again `nice -n 10 /usr/bin/python3.14 -I -B -` with `task-54-linux-startup2-invoke.py` as stdin. That wrapper's checked argv is exactly:

```text
/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
-I
-B
/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-prelaunch-v1.py
--release
/home/kmihara/hekit-runs/task54-launch/v2/startup2-release.json
--release-sha256
ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd
```

Cwd remains `/home/kmihara/reboot`. Exact replacement environment: `LANG=C`, `LC_ALL=C`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PATH=/usr/bin:/bin`, `RAYON_NUM_THREADS=1`, `SETUPTOOLS_USE_DISTUTILS=local`. It uses the same base controller and lexical candidate-venv child, unchanged direct-child60s/cleanup5s bounds. An outer110s timeout remains an observation failure, not proof of remote termination or permission to restart. Persist NEW invoke stdout/stderr/meta and inspect the same operation after timeout/failure.

5. If separately authorized to collect the resulting evidence, supply `task-54-linux-startup2-collect.py` to the fixed observer command. Save its stdout/stderr/meta unchanged, decode only the exact reported startup2-parent.json and startup2-child.json into NEW ledger files, verify recorded hashes/lengths, then join release SHA/purpose/argv/terminal streams/stages to the invoke metadata. Missing/error/incomplete states remain failures or unresolved facts. No generated report may replace raw bytes or promote the prior startup1 failure.

The remote v2 parent/child inode values, actual absent state, installed file identities, invocation terminal result and successful startup/helper snapshots are unknown until those separate operations occur. This preparation does not manufacture them. Source acceptance, deployment, readback and execution are distinct dispositions.

## Retained runtime limitations

FIX3's exact script-key/None exception preserves snapshots and rejects non-None; Linux startup1 did not record the finder value. Thus startup2 may still fail, and any later issue requires its own evidence and review. Known helper pins are provenance, not complete semantic closure. The accepted trusted-only startup boundary remains normal trusted local CPython/bootstrap/cache with no concurrent writer; no ancestry/native-isolation framework is added. Trusted platform preparation may invoke uname with no per-command timeout, and direct-child cleanup is not descendant containment. Source/metadata observations and successful startup stages would not authorize external append, context/capture or any scientific/performance claim.

The complete R16–19 startup1 wrapper sources and deployment/invocation reviews, separate prior release records, successful R16 receipt and R17 readback, and failed R18/R19 evidence provided the pattern. Existing startup1 inputs were rehashed unchanged; no old artifact was regenerated. Work stops with this local package frozen for review.
