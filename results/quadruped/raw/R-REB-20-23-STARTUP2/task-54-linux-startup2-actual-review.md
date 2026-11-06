# Task54 startup2 — independent actual-evidence review

2026-11-07. **SPEC: PASS. QUALITY: PASS for the saved R20–R23 startup-only observation and its evidence chain.** No actionable inconsistency found in this scope. The child completed startup and trusted helper preparation with terminal exit 0. Its status remains `observation_only_not_runtime_approval`; this review grants no context/capture release, full semantic closure, scientific result, or native isolation claim.

## Evidence identity and joins

Independently read the deployment and execution releases, all four stdout/stderr/metadata triplets, the candidate sidecar, parent and child records, and the relevant unchanged launcher predicates. Rehashed every triplet against its metadata, each wrapper against its recorded source digest, and each release against its metadata binding. All four controllers report terminal return code 0 and empty stderr. Their recorded transport is key-only SSH to `quadruped-gpu`, `nice -n 10 /usr/bin/python3.14 -I -B -`. No new remote operation was performed for this review.

| Artifact | SHA256 |
| --- | --- |
| Deployment release | `6018298069f83f574622973eea8aa8d4c5ae3cb7c11e2bb08c8a3a9989de1fb2` |
| Execution release | `8e0f99d8a0b59647afeb9c0e129ea65e4b210c81b6ba3317ab70753cec4725f4` |
| R20 deployment stdout | `775a6dd696babf34f5f111e1d040939805df995dbc000510694d97300e120169` |
| R21 readback stdout | `13816a2702cc02fede96c5da0b7f9a696798175cb42daa2d5f83d22884a82499` |
| R22 invocation stdout | `eab0f1002a135812448760db466d462b6712f0b3e579122050458d66eeef5b6c` |
| R23 collection stdout | `8f90d2794deddf3d59988fd1222dc7b1c34771966ee8fff7dddcf1b5f9bed618` |
| Candidate/deployed sidecar | `ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd` |
| Raw parent, 2755 bytes | `9d491997ed1476c98f5e1847354d30e344cb54bbb1f4ee2ecf480c16f71b5fc1` |
| Raw child, 98031 bytes | `a1fbe98aa390b7305f7189dabf574c4d28b2422617bfeb010e261f2c82fd1f41` |

The four deployed records match R21 readback and the corresponding R22 pre-exec pins exactly in path, byte count, SHA256, uid, mode, device and inode. All are under `/home/kmihara/hekit-runs/task54-launch/v2`, uid 1000, mode 0600, device 66311:

| File | Bytes | Inode | SHA256 |
| --- | ---: | ---: | --- |
| task-54-exporter-linux-prelaunch-v1.py | 7044 | 36309264 | `c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0` |
| task-54-exporter-linux-launch-v1.py | 29357 | 36309265 | `17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c` |
| export_reboot_block_seed_inputs.py | 61243 | 36309266 | `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce` |
| startup2-release.json | 13833 | 36309267 | `ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd` |

R21 common-parent facts agree with saved R17: uid 1000, mode 0700, device 66311, inode 36309255. V2 directory inode 36309263, same uid/mode/device, is first observed by R21; deployment did not record its inode. Both startup2 evidence paths were absent at R21. These are observations at those times, not a claim of present filesystem state.

Strict base64 decoding of each R23 file matches the corresponding local raw file byte-for-byte, its length and digest. R23 observes parent inode 36309269 and child inode 36309268, both device 66311, uid 1000, mode 0664. These evidence-file modes are separate from deployed-source mode 0600 and agree with the unchanged exclusive evidence writer; no source-mode assertion is transferred to them.

Parent `child_evidence_sha256` matches the raw child. Parent/child release records and child argv agree exactly; the parent controller argv agrees with the external verifier's planned direct exec. The sidecar digest matches actual local candidate bytes. Parent and verifier environment/cwd agree with the sidecar. All child stream hex/digest pairs are consistent and empty.

## Terminal result and phase boundary

Parent static validation reports passed. Child PID 1963722 started at Unix 1794005628.696766 and was observed terminal 0 at 1794005628.768885, **2026-11-07 07:53:48.768885 JST**. Parent records `child_terminal_observed: true`, `child_streams_complete: true`, `terminal_and_streams_collected`, no cleanup errors, `success: true`, and `child_terminal_zero_collected`. The 60-second child and 5-second cleanup limits remain recorded. These timestamps are operational custody, not a benchmark.

The child has exactly `startup` and `helpers` stages, purpose `startup-only`, and status `observation_only_not_runtime_approval`. The sidecar has empty payload, null context evidence, and all six expected-stage fields null. Thus these are observed snapshots, not preapproved stage-equality matches. The unchanged dispatch returns after helpers for this purpose; append/exporter/dispatch/context_done stages are absent. The exporter file was verified as an input, not imported or executed by this run.

## Actual stage findings

Both snapshots have identical interpreter identity: lexical executable `/home/kmihara/reboot/exporter-venv-task54-v1/bin/python`, resolved base executable `/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`, venv prefix and exec_prefix, and matching base prefix/base_exec_prefix. Version is `3.11.16 (main, Aug 14 2026, 15:35:36) [Clang 22.1.3 ]`. Flags are isolated 1, no_user_site 1, no_site 0, dont_write_bytecode true. Parent controller runs the pinned base interpreter with isolated 1, no_site 0, bytecode disabled and nice 10; it is correctly distinguished from the lexical venv child.

Both path lists equal the four sidecar startup entries, in order: base `lib/python311.zip`, base `lib/python3.11`, base `lib/python3.11/lib-dynload`, and the empty venv's `lib/python3.11/site-packages`. The baseline package root `/home/kmihara/reboot/venv/lib/python3.11/site-packages`, base site-packages, cwd and launcher source are not added to sys.path. Both site records have ENABLE_USER_SITE false, site_packages containing only the own venv site directory, and PREFIXES containing only the venv. The accepted predicate also checks that own site directory is empty at each snapshot; this is not inferred merely from its path string.

Both meta_path records are BuiltinImporter, FrozenImporter, PathFinder in order. Both path_hooks records are zipimport.zipimporter followed by the frozen FileFinder.path_hook. The accepted running source validates their supported identities/details before recording stable names; JSON labels alone are not an independent object-identity mechanism.

| Stage | Module entries | SourceFileLoader | BuiltinImporter | FrozenImporter | Trusted typing aliases | Cache entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| startup | 77 | 34 | 24 | 19 | 0 | 10 |
| helpers | 162 | 97 | 44 | 19 | 2 | 14 |

All 77 shared module records remain identical; 85 entries are added during preparation, none removed. Every recorded ordinary file/origin/package-search path is inside the trusted base stdlib root, excluding base site-packages, except `__main__`, which is exactly the pinned v2 launcher. Built-in/frozen records and the two explicit trusted-alias origins account for the other origin forms. There are no ExtensionFileLoader or SourcelessFileLoader records in these snapshots. `_task54_exact_exporter`, sitecustomize, usercustomize, _distutils_hack, numpy and torch are absent. Inspection of every origin/file/search field found no baseline package or exporter source origin.

`typing` appears after preparation with the pinned base `typing.py` path and SourceFileLoader. `typing.io` and `typing.re` are the narrowly recognized `typing._DeprecatedType` aliases, origins `trusted-alias:typing.io` and `trusted-alias:typing.re`, same trusted parent file, null search. These are accepted through the existing explicit parent/alias identity predicate, not a general spec-less exemption.

**The exact key `/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-launch-v1.py` is present with JSON null in both cache snapshots.** Under the pinned FIX3 serializer and `finder is None` guard, this records the actual negative cache value for startup2. The key is absent from sys.path. The only other null cache entry is the absent base python311.zip; all other entries record supported FileFinder. The four added cache keys are trusted stdlib collections, email, importlib/metadata and importlib/resources directories. All ten shared cache entries remain unchanged. This resolves startup2's finder observation only: startup1's unrecorded finder value remains unknown, and startup1 remains failed.

Preparation reports the exact 35 HELPERS names. All 35 finite helper-source pins correspond to files present in the helper snapshot. The pinned dispatch verifies those files before explicit imports and runs the reviewed fictional in-memory metadata/RECORD, email, CSV and argparse warmups, platform calls, and pathlib operations before the helpers snapshot. The 97 SourceFileLoader entries include the launcher and more stdlib files than those 35 pins; therefore neither the count nor the successful warmup proves an exhaustive source/cache/semantic closure. Platform warmup may invoke uname; these records do not give a complete native-command execution trace or descendant containment proof.

## Permissible conclusions and remaining prerequisites

It is now supportable to retain R20 deployment, R21 independent identity readback, R22 terminal-zero startup/helper observation, and R23 exact-byte collection as actual Linux evidence for these exact inputs. The reviewed FIX3 narrow negative-cache case was actually encountered and accepted. There is no evidence-chain blocker to publishing that bounded outcome together with the original startup1 failure.

Before any context execution, root must separately review and release a concrete context-purpose sidecar, selected helper/source and native warmup scope, and actual supported stage expectations. Startup2 supplies candidates for startup/helpers only. Append/exporter/dispatch expectations are still absent and must be established through an explicitly reviewed method; they cannot be invented, copied from another platform, or obtained by bypassing the existing schema/guards. Exact context input/native/git paths and pins, fresh non-clobbering output/evidence paths, repeated prelaunch static checks, and the existing workload/resource gate remain required for that next operation. This review authorizes neither a preparatory remote operation nor context execution.

Successful context observation and review would still be separate from capture approval and the pinned unchanged exporter-main capture path. No external package execution, baseline import, context artifact, approval-bound capture, full semantic closure, complete native isolation or scientific performance/result is established here. The existing trusted interpreter/bootstrap/cache and no-concurrent-writer boundary remains. Saved R20–R23 observations do not refresh unrelated workload liveness or progress.

Only this review file was written. Review operations were local reads, hashes, strict data decoding and comparisons. No child/source import or execution, test rerun, SSH/network operation, deployment, source edit, Cargo/build, Git mutation or subagent was performed. A local audit predicate initially omitted the two known `trusted-alias:` origin labels and stopped on typing.io; correcting that review-only classification completed the full check. This was not a launcher failure or a change to accepted policy.
