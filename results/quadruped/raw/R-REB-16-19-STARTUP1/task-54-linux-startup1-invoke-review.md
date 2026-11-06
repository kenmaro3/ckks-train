# Task54 startup1 external invocation wrapper review

2026-11-07. **SPEC: PASS. QUALITY: PASS for the exact new external invocation wrapper. No load-bearing issue found in the assigned scope.** Startup remains unreleased pending root disposition and fresh independent deployment/readback/workload evidence. This source review is not a startup success record or context/capture authorization. Unchanged FIX2 code is not reopened.

Reviewed complete new source `task-54-linux-startup1-invoke.py`, SHA256 **`94563d06cc7cfe48bfc6ed3e19a9a6f4404418917d606044e986333bb7ea0c44`**, against the accepted candidate sidecar `b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a`, FIX2 launcher/prelaunch contract, R15 command identities and the collected deployment receipt. Performed only local source/AST/JSON inspection, hashes and literal-data comparisons. No wrapper execution, tests/reruns, SSH, target runtime, source edits, Git, build or subagents occurred. Only this review file was written.

## Invocation and pin consistency

All10 explicit pin tuples agree with the accepted candidate/deployment source identities and R15 executable observations. Caps accommodate the observed files: system Python7477160 bytes, native uname11352352 bytes and base Python21740000 bytes. Owner expectations match those observations (system/native uid0, base and deployed inputs uid1000). The exact replacement environment equals the accepted candidate/R11 environment; cwd is `/home/kmihara/reboot`. The sidecar and both launcher program digests are fixed, not discovered as fresh authority from their own contents.

The exact prelaunch invocation is:

```text
/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11
-I
-B
/home/kmihara/hekit-runs/task54-launch/v1/task-54-exporter-linux-prelaunch-v1.py
--release
/home/kmihara/hekit-runs/task54-launch/v1/startup1-release.json
--release-sha256
b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a
```

This is an argv vector passed directly to `os.execve`; no shell string, command override or extra transport argument exists. `os.chdir` uses the pinned candidate cwd. `execve` replaces the verifier process, retaining its PID and inherited nice level, with exactly the supplied environment and no inherited overlay. It does not create another parent/cleanup framework. The frozen prelaunch then performs its own static validation before creating the one fresh lexical-venv child, with the candidate's 60-second timeout and 5-second cleanup.

The wrapper checks startup-only purpose, empty payload, environment/cwd and 60/5 limits, while the verified sidecar bytes bind all remaining schema/stage fields. No context/capture fallback is present. It verifies the exporter bytes as an input; it does not execute them.

## Pre-exec checks and failure semantics

Before the base interpreter starts, the wrapper checks uid1000/nice>=10, the outer system interpreter's resolved path/hash, and every pinned input. Each read rejects noncanonical/nonregular/wrong-owner/oversized paths, opens without following a final symlink, checks opened device/inode, reads at most cap+1 bytes, compares before/after identity/size/mtime and verifies the digest. This is consistent with the ordinary trusted-interpreter/cache and no-concurrent-writer boundary; it does not claim hostile-filesystem race containment.

It checks canonical owned0700 deployment directories,0600 single-link deployed files, absent parent/child output evidence, and absent fictional warmup path. The selected executable `/usr/bin/uname` must resolve to the separately pinned native command and remain executable. It verifies base prefix/zip candidates, fixed site/user customization forms and exact base `.pth` inventory before exec. The base-site hook source identities and environment match the already reviewed controller bootstrap; no assertion is made that the parent itself has the target child's clean guard state. Fresh exec into the target child remains the isolation of Python process state specified by the contract.

The flushed stdout record is explicitly `verified_before_exec_not_startup_result`. If chdir or exec fails, the wrapper reports `verifier_failed_no_retry` on stderr and exits1. A pre-exec stdout record can therefore coexist with a failed exec and is never sufficient for success. No automatic retry, cleanup of evidence, guard patching, foreign-path append, capture invocation or baseline import exists.

Once exec succeeds, SSH terminal status and the matching remote parent/child evidence must establish the startup outcome, including child terminal/stream facts. An outer110-second observation timeout is neither remote termination nor permission to retry. Native command descendants are not contained by this design, and no such containment or native-isolation claim is made.

## Deployment evidence and remaining gate

Locally verified the saved deployment stdout SHA256 `b3412785633ee4290df0348036cc403d4ea011bb33518006c39213e46a9bd90e` and empty stderr against `task-54-linux-startup1-deploy.meta.json`. The metadata records exit0 and source `d63de1ddda59b69fb183ad9507d0a7cdcda7ae28d4acd03d2550f3d9c0d38bca`; its separate deployment-release digest `69e1a7865621049de25fb34e69758d0c1dec74fb253075ba174898b94681a7bb` matches the deployment-only Markdown release, not the candidate sidecar. All four receipt file hashes/uid/mode0600 agree with the new verifier's pins.

This establishes consistency of the saved successful deployment record. Fresh independent remote readback and workload observation are still pending and were not inferred from it. Root must bind this exact verifier source/command in its separate startup disposition after those checks. The source PASS adds no full helper semantic-closure requirement to this trusted-only observation and does not release external package append, context, capture, scientific acceptance or whole Task54/full-matrix completion.
