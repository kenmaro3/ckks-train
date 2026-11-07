# FIX1 finite attempt inventory, frozen before execution

Attempt `task54-fixture-custody-fix1-attempt-01`: exactly one fresh pure `support.preflight_manifest(0, rows)` call. Read/hash exactly410 selected current source files (12934199 bytes), assembled from frozen565 minus158 named payload paths plus3 adopted paths. Require full path/length/hash equality with original handoff410 pins and absence/presence of the exact deleted/added paths. No raw filesystem-tree scan or filtering; historic Mac physical-equality failure remains. Expected manifest size54531 bytes; measured outcome is authoritative.

Driver: 3701 bytes SHA256 `05bddd182529abe44071954e5749f5250926f10f304e770977daa906d8d11c05`.
Controller: 2363 bytes SHA256 `5d3aca3c7e1a3e633d6c82d022c10948cc61ff143c70427db19557e1fb329858`.
Structured inventory: task-54-d7-fixture-custody-fix1-test-inventory.json.

Command: `python3 -B .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-fixture-custody-fix1-controller.py`. Controller preserves the exact resolved child command/cwd and driver/controller/Python/inventory hashes, uses exclusive creation for command/stdout/stderr/outcome and driver result JSON, and exits with the actual child status. Each outcome records stdout/stderr/result hashes and lengths. No later command can replace that status.

Postcheck re-reads all43 frozen original files and43 snapshot copies, verifies snapshot-manifest identity, and verifies410 selected source hashes. All historical stale JSON/failed log/scaffold RED artifacts remain unchanged. No Cargo/test/Clippy/fmt rerun or source/native/network/cache work.
