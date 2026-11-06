# R-REB-16–19: deployed startup1 and retained failed startup

These are byte-identical operational records, not scientific result JSON and not paper-table inputs. No original JSON values were repaired or normalized. The deployment contains the reviewed three source copies plus a startup-only sidecar. Never rerun its fresh-only deployment or overwrite its evidence paths.

| Phase | Observed result | Limit |
| --- | --- | --- |
| R16 deployment | exit0; four files111278bytes total, own uid1000/mode0600, fsynced/hash checked | Deployment alone does not prove interpreter startup. |
| R17 independent readback | exit0; four byte/hash/owner/mode/inode matches, two own0700 directories, evidence paths absent | Point-in-time readback and process observation only. |
| R18 startup1 | Static preflight passed, then new-venv child1958934 exited1 with complete streams and no cleanup error | Startup failed; neither stage snapshot nor trusted-helper preparation completed. |
| R19 evidence collection | Both remote parent/child files copied byte-for-byte with SHA256 verification | Collection is not a successful startup or a retry. |

At07:25:40 JST, R17 also observed the same own FULL2 chain and post-full queue alive with matching start ticks. FULL2 log update101 is zero-based, hence102/1000 updates completed; result absent, queue owner.json only. This is not final accuracy or job completion, nor a continuous workload/thread measurement. No observer/deployer changed those jobs or their queue.

R18 used the separately released exact external verifier, which replaced itself with pinned base CPython3.11.16 prelaunch via execve. Prelaunch checked static inputs and started the lexical new-venv interpreter with -I -B and the bound sidecar. Its owned child terminated1 at07:27:42.013940 JST. The failure is `foreign importer cache path: /home/kmihara/hekit-runs/task54-launch/v1/task-54-exporter-linux-launch-v1.py`. The child stages object is empty. The actual finder value was not recorded; these Linux records do not prove it was None. The raw child file is824bytes SHA256c35c6aacb3f2c3a6fd10af7c46fe8b9720b838ef2f9adbe0f81555050a8fedea; parent is4224bytes SHA256119b5447185f61fb4a5f578315c347445604b25adebbdc9a6288435768ebaa9f.

No helper preparation, external package append, exporter-code load, context, capture or scientific import was reached. Ordinary system/base/child stdlib imports did occur; unqualified “package import” wording in the deployment receipt's fixed scope text means no external scientific package import. The source reviews in this archive apply only to their exact source/deployment/invocation contracts and do not override the actual failed outcome. There was no automatic retry, guard patch, evidence deletion or broader process signaling.

The candidate's60s direct-child timeout and5s stream cleanup do not claim descendant containment. Known source/executable pins and ordinary trusted local/bootstrap/cache assumptions do not prove full native/helper closure. Future corrected code must receive separate tests/review, new frozen source/sidecar/evidence identity and a fresh exact-purpose release. Context, numerical compatibility and all scientific gates remain unmeasured by this archive.

The deployment wrapper embeds the exact four copied payloads as base64. This is retained execution input, not a suggestion to deploy again. Collection stdout likewise wraps original parent/child bytes; decoded files match their remote hashes. SHA256SUMS covers24 originals plus this README; its own hash is held in the controller publication manifest.
