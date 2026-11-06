# Task54 startup2 package — scoped source-only review

2026-11-07. **SPEC: PASS. QUALITY: PASS for the exact prepared package. No source-level blocker found in the assigned delta.** This is not deployment/startup authority or a claim about current remote v2 state. Root has now accepted FIX3 source review `eae95054bbc1211a0c5221fd0dd3d596165235b45d8b61167624803aa2ad8565`; preparation-time statements that it was pending remain historical, not a new blocker. Separate concrete deployment and execution dispositions are still required.

## Reviewed identity and custody

| Artifact | SHA256 |
| --- | --- |
| Preparation report | `23a61c60a63c9cf3c4c102c45ca249e3b4cb3de6b4cb3cb9c4c35380cb19d401` |
| Package manifest | `ab6094f70b51da5a0b5a713e30827dde34d4c37fb182416067b95e875ebde57f` |
| Candidate sidecar | `ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd` |
| Deployment wrapper | `ab1c616bea7230961ffcced5c8706c0b308bf75fe38b6feadc807044e948eccf` |
| Invocation wrapper | `8c313e0716ceb221077ae03491ef9ace8a676b87ea754214b046866013cb3ffc` |
| Independent readback | `6a45c62ca7196e771f0166e3afdae1878b9671683fa861009767e259387517a2` |
| Evidence collector | `e52ff869ecd2d84f395f25f8459df7d3a010e24b51dcec3c1f1f1add6909f08b` |

Read the complete report, four wrapper bodies, candidate, manifests and source differences against accepted startup1 predecessors. Independently rehashed all11 package files and all27 listed unchanged startup1 input files. Parsed the deployment AST literal without executing it, decoded all4 payloads and compared their exact bytes with the standalone originals and declared size/hash mappings. All matched,111477 bytes total. The prelaunch and exporter additionally equal their exact decoded accepted startup1 payloads; the launcher equals accepted FIX3 source.

The source pins are prelaunch `c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0`, FIX3 launcher `17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c`, and preserved exporter `0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce`. The separately owned live development exporter is not substituted or characterized by this package.

## Delta assessment

**Sidecar:** recursive comparison confirms exactly6 changed leaves: prelaunch/launcher/exporter deployed paths, parent/child evidence paths, and launcher SHA256. Everything else remains byte-value equivalent to the accepted startup1 record:35 helper pins, base/venv/config/layout/root_mode, lexical interpreter, package root, bootstrap candidates/pins, exact environment/cwd,60/5 limits, startup-only purpose, empty payload and null stages/context evidence. No unrelated policy or authority change is hidden in the migration.

**Deployment:** the existing `/home/kmihara/hekit-runs/task54-launch` parent is now checked against R17 uid1000/mode0700/device66311/inode36309255 instead of recreated. The higher parent retains its exact prior identity check. Only absent `/home/kmihara/hekit-runs/task54-launch/v2` is created; fixed files use the unchanged exclusive/no-follow0600 writes, hash/uid/mode/link checks and fsync sequence. Failure preserves partial state with no retry or cleanup. No mkdir/chmod/content-write targets v1. Syncing the common ancestor does not rewrite v1 evidence.

**Invoker and collector:** mechanically verified their complete sources equal the accepted startup1 sources after only the declared v1→v2, startup1→startup2 and launcher/sidecar hash substitutions. Fixed argv, replacement environment/cwd, trusted input checks, direct exec, pre-exec-not-success label, child60s/cleanup5s limits and no-retry semantics are retained. The collector remains a fixed two-file bounded read; it preserves missing files and exact available bytes without treating collection success as startup success.

**Readback:** removes the unrelated FULL2 process/log/queue observation and retains the bounded canonical regular-file reader, no-follow open, before/after identity checks and4 fixed hash/length/uid/mode comparisons. All4 expected records agree with the payload and invocation pins. It records actual device/inode, checks both owned0700 directories and absent startup2 evidence, and reports only readback completion. New file inode values are deliberately not invented in source: root must join each returned file identity to the actual deployment receipt before execution release. This is an explicit later gate, not automated proof already supplied by this script. No new FULL poll is included.

The deployment receipt records file device/inode, while its created-directories list contains names, not directory inode metadata. Therefore the later receipt join is specifically for the4 files. The existing common parent's readback identity can also be checked against retained R17 identity; v2 directory device/inode is a new readback observation with the required canonical kind/uid/mode checks, not a fabricated earlier measurement. No directory-identity equality absent from the records should be claimed.

The preserved root-mode and exact module-loader/cache policies are unchanged. The FIX3 branch remains limited to the exact pinned launcher key with finder None by identity. No broad foreign-None allowance, cache deletion, external package append or context/capture permission is introduced.

## Permissible subsequent steps under separate root dispositions

1. Bind the exact deployment source digest above in a new deployment-only record and run it once using the selected key-only transport and `nice -n 10 /usr/bin/python3.14 -I -B -`. Preserve new startup2 deploy stdout/stderr/meta. It must establish actual parent identity and v2 absence; neither is asserted current by this review. Partial failure stops without reuse/retry/rollback.
2. Run the exact independent readback source with the same fixed observer command and new stdout/stderr/meta. Require terminal success, all4 identities, required directory facts and absent evidence. Compare each file's path/bytes/hash/uid/mode/device/inode with the separate completed deployment receipt. Any mismatch prevents execution. Complete the established workload/scheduling gate independently; this readback does not observe workloads.
3. Only after those joins and root's separate one-attempt startup-only disposition, run the pinned invocation source using that fixed observer command. It execs base CPython3.11.16 `-I -B`, the v2 prelaunch path, and `--release /home/kmihara/hekit-runs/task54-launch/v2/startup2-release.json --release-sha256 ec4cd24694e5484ed4192470b0186229f4cd4bda2ebe4ffc30108bb678394dbd`. Cwd and replacement environment remain the accepted ones. An outer110s timeout is unresolved observation, not remote termination or permission to restart.
4. Collect only the resulting startup2 parent/child files with the exact collector. Save raw stdout/stderr/meta; verify base64 lengths/digests and retain byte-identical decoded files. Join release/purpose/argv, terminal exit, streams/cleanup and stage facts before assigning any outcome. Pre-exec success or successful collection alone is insufficient. Missing, failed or incomplete evidence remains so.

These are the supported next steps, not a claim they occurred or that old startup1 releases authorize them. No context/capture or science follows automatically from a successful startup-only observation.

## Review boundary and retained uncertainty

The actual remote v2 absence, installed files, file/directory identities, startup2 exit/stages and finder value remain unknown here. Startup1 remains failed and immutable. Normal trusted CPython/bootstrap/cache and no-concurrent-writer assumptions remain; platform warmup can invoke uname without a per-command timeout, and direct-child cleanup does not contain descendants. Known helper pins do not prove full semantic closure or native isolation.

Only this review file was written. All checks were local reading, hashing, AST/literal decoding and data/text comparison. No prepared source/wrapper execution, test suite, SSH, remote observation, deployment, scientific import, Cargo/build, Git mutation or subagent occurred. Unchanged accepted policy was not reopened.
