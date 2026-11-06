# Task54 empty-venv attempt1 — retained operational evidence audit

2026-11-07. **Evidence integrity: PASS. Static-creation disposition: ACCEPTED for this creator-only phase.** The retained record supports successful creation and static inspection of `/home/kmihara/reboot/exporter-venv-task54-v1` by the exact released FIX1 wrapper. No contradiction, warning, timeout, cleanup error or missing terminal result was found. This is not a release or successful test of the new interpreter, a package-path launcher, exporter context, native imports or scientific compatibility.

This review read the complete saved stdout/stderr/controller metadata and creator-only release, rehashed their bytes and the wrapper, decoded and rehashed the retained config bytes, and checked the saved tree/process/timing fields. No remote access, rerun, new-interpreter startup, source edit, tests, scientific imports or live host mutation occurred. Only this review was written. Findings describe the saved attempt, not the current live filesystem.

## Identity and chronology

The wrapper rehash is `bac07100a6c8740e6df36ec99eb0d3463bf16a77d9d809006ca39c4105ed0873`, matching controller metadata and the released source. Release SHA256 is `12295ca68761bec151a9df7d6b4b3e1b2680cc7185ebee2dc15029f1b8f98c7e`, also matching metadata. The recorded independent FIX1 review identity is the previously accepted `2eaa90c33aac7edb3d61de065399272b8d21d1a870248cd36ce9038058ea0bb9`.

Recorded JST chronology is ordered correctly: root release at 06:08:28.812940; controller start at 06:08:28.898094; wrapper start at 06:08:36.727462; creator start at 06:08:36.749325; creator terminal collection at 06:08:36.777275; wrapper finish at 06:08:36.810504; controller finish at 06:08:38.588270. Approximate intervals are 0.028 seconds for the creator, 0.083 seconds for the wrapper, and 9.690 seconds for the full controller/SSH interval. These are operational timings, not scientific performance results.

Controller stdout is exactly 10430 bytes with SHA256 `01c8dfd367c273b87777b71213d102933b1fb3eced0ce465d300d1e19c49f675`; controller stderr is zero bytes with the empty-content hash. Both independently rehash to the values recorded in metadata. Controller returncode is 0. The key-only SSH command uses `nice -n 10 python3 -` and names the exact released wrapper hash; its record does not authorize any additional remote operation.

## Command, preflight and owned child

The retained argv is exactly the pinned base `/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11`, followed by `-I -B -m venv --without-pip --symlinks` and the new destination. Cwd is `/home/kmihara/reboot`. The replacement environment matches the reviewed mapping: PATH `/usr/bin:/bin`, LANG/LC_ALL `C`, SETUPTOOLS_USE_DISTUTILS `local`, and OMP/MKL/OPENBLAS/RAYON thread-limit variables each `1`. No loader/activation/PYTHONPATH override is present in that mapping.

The record contains matching hashes and sizes for all six pinned executable/source inputs, including the 21740000-byte base executable `2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174`. Parent identity is canonical uid1000/mode0775; destination `lexists` was false. Both base `pyvenv.cfg` candidates, the zip and fixed customization candidates are recorded absent. The reviewed wrapper additionally requires the exact `.pth` set and no customization package/extension candidates before dispatch; its passed path attests those checks, though stdout does not separately enumerate those successful set checks.

Owned child PID1939496 has a saved `/proc` stat sample naming `python3.11`, PPID1939495, start_ticks473994888, nice10 and one thread at that sample. The wrapper itself records nice10. The one-thread sample and configured caps support the bounded lightweight operation; they are not a continuous peak-thread measurement.

Child returncode is 0; `child_terminal_observed` and `child_streams_complete` are true; `child_cleanup_status` is `terminal_and_streams_collected`; cleanup_errors is empty; no timeout or controller/proc-stat error is reported. Child stdout/stderr hex strings are empty and both hashes are the empty-content SHA256. There are therefore no retained creator warnings to resolve. The final wrapper status is `created_static_checks_passed_new_interpreter_not_started`.

## Static output verification

The saved tree contains exactly 15 expected entries: six directories, five regular files and four symlinks, all uid1000. It has no entry below the own site directory. Together with the passed explicit empty-site check in the pinned wrapper, this supports an empty `/home/kmihara/reboot/exporter-venv-task54-v1/lib/python3.11/site-packages` at postcheck.

The three interpreter links resolve to the exact pinned base executable. `bin/python` and `bin/python3` point relatively to `python3.11`; `bin/python3.11` points to the absolute base executable. `lib64` points to `lib` and resolves within the new destination. The pinned wrapper separately hashed all three resolved interpreter links against the base pin before its success status; the tree records their targets/resolutions rather than repeating those hashes per link.

The retained `config_hex` decodes to exactly 396 bytes. Independently hashing those bytes gives `4af48d1b0d9fea3094d3c725918c55e52670536de714c30660f3fb37ec13c643`, equal to both `config_sha256` and the tree's pyvenv.cfg hash. Its five fields record the expected base bin home, `include-system-site-packages = false`, version3.11.16, the exact base executable and the correct destination with `--without-pip` in the generated command.

The generated config command omits `-I -B` and the explicit default symlink option. This is not conflicting execution evidence: the controller/wrapper argv is authoritative, and the actual symlink layout is separately recorded and checked. All five expected regular files are comfortably below the reviewed 65536-byte cap. The retained activation-script hashes describe files written as data; no activation script was invoked.

## Scope and remaining gates

The reviewed command starts the base creator, not any executable under the newly created destination. Its no-pip path and wrapper then perform static inspection only. The retained evidence and reviewed control flow support the stated no-new-interpreter-start scope. They do not constitute a general process trace or prove that all bootstrap cache bytes equaled the hashed sources: normal creator site/stdlib/bootstrap remains the explicitly accepted trust boundary, and `-B` does not disable bytecode reads. The exporter source guard was not installed or exercised here.

The wrapper has no FULL2/queue mutation operations and none is recorded in this attempt. This audit is not a fresh workload-owner or queue observation. No original scientific environment repair, package install, scientific import, full package/native hashing, context observation or capture follows from static creation.

Before any later new-venv startup, retain the separate prelaunch checks for config/interpreter/empty own site and applicable startup paths; bind the actual launcher source/command/environment and observe truthful live prefix/flags/path/module/hook identity. Explicit package-path setup, unchanged source guard, full runtime/native identities and isolation, optional-query policy conditions, archive provenance and fresh exact six-comparison compatibility remain separate gates. Successful creation does not close any of those gates or admit diagnostic arrays. Root must separately release a concrete next phase.

## Rehashed local records

| Record | SHA256 |
| --- | --- |
| task-54-linux-create-empty-venv-attempt1.stdout | 01c8dfd367c273b87777b71213d102933b1fb3eced0ce465d300d1e19c49f675 |
| task-54-linux-create-empty-venv-attempt1.stderr | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| task-54-linux-create-empty-venv-attempt1.meta.json | f16aaf07ab79fcf1fb41dceecd40422f185bd1c6d43664775edb034feec23b48 |
| task-54-linux-create-empty-venv-release.md | 12295ca68761bec151a9df7d6b4b3e1b2680cc7185ebee2dc15029f1b8f98c7e |
| task-54-linux-create-empty-venv-20261107.py | bac07100a6c8740e6df36ec99eb0d3463bf16a77d9d809006ca39c4105ed0873 |

Ledger filenames are relative to this directory. No later launch release is issued by this report.
