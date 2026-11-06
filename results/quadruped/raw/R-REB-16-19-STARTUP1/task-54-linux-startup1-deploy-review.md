# Task54 startup1 four-file deployment — bounded review

2026-11-07. **SPEC: PASS. QUALITY: PASS for the exact new deployment wrapper and candidate record. No load-bearing issue found in this scope.** Actual remote deployment, collected deployment evidence and independent remote readback remain pending. This is not approval to start the target interpreter, run context/capture or infer scientific results. The unchanged FIX2 implementation is not reopened.

## Exact reviewed artifacts

| Artifact | SHA256 |
| --- | --- |
| `task-54-linux-startup1-deploy.py` | `d63de1ddda59b69fb183ad9507d0a7cdcda7ae28d4acd03d2550f3d9c0d38bca` |
| `task-54-linux-startup1-deploy-manifest.json` | `a74c527632ac4c6f03b5deba2f038e6ea05e3a13ebe89227b385adc28a499752` |
| `task-54-linux-startup1-candidate-release.json` | `b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a` |

Read the entire executable wrapper body. Parsed its AST, extracted only the string literal passed to `json.loads` for PAYLOAD, decoded the four base64 items, and compared every resulting byte with its pinned local original and manifest entry. The wrapper itself was never executed. All four comparisons passed:

| Remote basename | Bytes | SHA256 |
| --- | ---: | --- |
| `task-54-exporter-linux-prelaunch-v1.py` | 7044 | `c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0` |
| `task-54-exporter-linux-launch-v1.py` | 29158 | `e6ff0ab0e45fc680442bc184ecf665cdc63888bba13c673d5cb18441d9fecd84` |
| `export_reboot_block_seed_inputs.py` | 61243 | `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce` |
| `startup1-release.json` | 13833 | `b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a` |

Total payload is 111278 bytes. The exporter comparison used the actual local file at `tools/export_reboot_block_seed_inputs.py`; its bytes matched the frozen pin at review time. No development revision was substituted.

## Mutation and evidence assessment

Before any write, the wrapper requires uid1000, nice at least10, resolved controller `/usr/bin/python3.14` and its exact observed hash. It decodes/validates the complete finite payload, checks exactly four permitted distinct basenames, and verifies the canonical parent directory's kind/uid/mode/device/inode against R14: `/home/kmihara/hekit-runs`, uid1000, 0775, device66311, inode36072090.

It then requires both `/home/kmihara/hekit-runs/task54-launch` and its `v1` child to be absent. `mkdir` is fresh-only, with no `exist_ok` or existing-directory reuse. Four writes use `O_CREAT|O_EXCL|O_NOFOLLOW` at fixed basenames, so no path traversal or overwrite route exists in the pinned payload. Each file is flushed/fsynced and rechecked for uid1000, mode0600, link count1 and exact hash. The final destination entry set must match exactly; destination/root/parent directories are fsynced before the success label.

Ordinary exceptions produce failure with partial state preserved and no retry/rollback/cleanup mutation. A later invocation refuses the already-created root. The receipt distinguishes created directories and verified files; success is assigned only after all checks/fsyncs. If the outer observer is interrupted or loses transport, absence of a complete successful receipt is not success; outer collection/readback remains necessary. This is not a race-containment or crash-transactionality claim, and retains the existing no-concurrent-writer assumption.

The wrapper contains no subprocess launch, shell call, target module import, venv mutation, checkout write, queue/job change or scientific operation. File mode0600 is owner-writable, not filesystem-enforced immutability; the three copies' immutable identity here means their fixed reviewed bytes/hash and no replacement by this wrapper. Subsequent source/release pin checks and separate operational authority remain required.

## Candidate record and observed inputs

The record is consistent with the frozen FIX2 schema and the selected startup-only purpose: empty payload, six null stages, null context output, lexical new-venv `bin/python`, fixed program/exporter identities, 60-second child limit and 5-second cleanup limit. No capture approval, mode, artifact output, context paths or dispatch override is present. All deployed program/exporter paths point to the exact new directory; parent/child evidence filenames are distinct fresh siblings, outside protected interpreter/package trees.

The following data comparisons were performed locally against the saved observations:

- **R10:** exact base executable/prefix, site source hash, base `.pth` name/hash and `_distutils_hack` source hash match. Inventory SHA256 `b353a62e49466d5239851fc5c99399b4273c7ea843cb91424295c16aec51f27d`.
- **R11:** exact config hex/hash, all15 layout entries (kind/uid/mode and file bytes/hash or link target/resolution), environment map and cwd match the accepted creation record. Its stdout SHA256 remains `01c8dfd367c273b87777b71213d102933b1fb3eced0ce465d300d1e19c49f675`. Octal mode strings were compared numerically without changing file bytes. Cwd is `/home/kmihara/reboot`; PATH is `/usr/bin:/bin`, LANG/LC_ALL are C, one-thread caps are1 and SETUPTOOLS_USE_DISTUTILS is local.
- **R12/R13:** all34 helper source pins plus the exact typing.py pin match, with no additional or missing entry. Venv root uid1000/mode509 (0775) matches R13. R12 SHA256 `2dc5e1c90a9df0b078b2a7f7cb73f38ceef165860ba8e9c82649daac04235b12`; R13 SHA256 `85a430e3bc0a9e72231b396d5456419e2846167b16948cef279d51dd01829fdc`.
- **Latest R14 FIX1/R15:** parent metadata, absent root/destination, resolved cwd/package root and controller Python identity agree with the wrapper/record. Both saved stdout/stderr hashes were checked against controller metadata. R14 stdout `1eee1546a55ae482aaa27f85d87dff4d7c92e12c640e84c9d8707d333ac3ef9a`; R15 stdout `94ca5a2c95c8b83d7e0531ae70e2d25baef474ba3e520804b3a4bf0e8d94c6be`. Observer Python hash is `52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41`.

The fixed startup path/customization/absence arrays are schema expectations, not observed child runtime state. Fresh prelaunch validation remains mandatory before any future child. The 35 helper pins identify known sources; this review makes no complete semantic-closure claim. Such closure is not required to copy four inert files or to represent an observation-only candidate, and no full-closure condition was added to this deployment gate.

## Verification boundary

Only this review file was written. Local actions were source/JSON/AST reading, literal decoding, hashing and data comparison. An initial AST extractor assumed PAYLOAD was a direct literal; inspection showed a `json.loads` call around a literal string, and extraction was corrected without executing it. A first inventory lookup omitted R10's separate base_pth array; combining its explicit source and base_pth inventories completed the comparison. Neither was a wrapper/runtime failure.

No wrapper execution, mutation fixture, target runtime, test suite, SSH, build, Git action or subagent ran. The review does not report remote files as installed. Root's separate exact-command/deployment action must collect its terminal/stream evidence and read back actual file identities; startup release remains a later independent decision. No unchanged FIX2 source concern or broader host-security framework is introduced.
