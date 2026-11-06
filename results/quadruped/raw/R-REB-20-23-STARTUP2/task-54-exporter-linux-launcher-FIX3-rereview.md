# Task54 launcher FIX3 — scoped same-reviewer rereview

2026-11-07. **SPEC: PASS. QUALITY: PASS, source-only.** The minimal cache-entry change implements root ruling R-LAUNCH-FIX3 exactly; no new fix-induced defect was found. This does not establish the failed Linux entry's value, repair the historical startup1 outcome, approve deployment/retry, or establish target/helper/runtime success. Unchanged launcher policy and previously closed findings are not reopened.

## Reviewed identity

| Artifact | SHA256 |
| --- | --- |
| FIX3 root ruling | `dcf9550660f0524d69c9b3b8597b6a7488b22da2ebb36f0323c293a64725eb8b` |
| FIX3 delta patch | `15f0699ae677a0028a5af5ebc02070e1aa43e8dcf330a870a9e6605bb57fe7de` |
| FIX3 full patch | `9d0bdefe0a03038b841188cc5a188156138dab18ed28a2a8439d64e811877fd9` |
| Current report | `d5386386149e9c8fbe709474b936c0c6e0fc9dbe3bb93d31f37cf85d417fae30` |
| FIX3 manifest | `85de4c4fed012fb4a688e77e4e47455cab4f526fc89c633e2df54bf72d48461e` |
| Launcher | `17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c` |
| Tests | `e4a0e4a51ec73feb8bfec62ce58ca3a6c9e0656973425c21b8bf771dd0387279` |

Independently rehashed all33 current manifest entries,79 immutable predecessor entries and2 unchanged external evidence files. Verified the three modified-file delta sections against preserved FIX2-to-current differences and all postimage hunks in the31-file full patch and11-file delta. All matched. Read the root ruling, separate API addendum, source/test change and immediate control-flow context, appended FIX3 report, meaningful RED, final GREEN and both retained intermediate-failure logs. No patch was applied.

## Contract and behavior

`task-54-exporter-linux-launch-v1.py:267-277` first requires an exact string cache key. If that key equals the release's canonical `programs.launcher.path`, it requires **`finder is None`**. This branch precedes the general trusted/permitted path rule, so any non-None value at the exact script key—including a genuine FileFinder or a false/equality lookalike—still fails even if that key overlaps a generally permitted location. Other keys use the original trusted/permitted policy and existing finder-type/path/loader-detail checks.

The snapshot assignment preserves the exact observed key and JSON null; no key is inserted when absent. There is no cache deletion, path normalization, new directory trust, sys.path change, schema/transport override or helper/guard relaxation. Existing full stage equality continues to distinguish presence from absence. The small branch adds no unbounded work or separate authority mechanism.

The inherited canonical release/source identity, launcher hash and actual argv verification still run before dispatch (`main:408-413`). The internal checker is not a standalone authentication API, as the addendum explicitly states. The exact-key exception therefore operates only after those unchanged checks on the production invocation path.

## Focused evidence

The six new methods exercise the actual importer checker with fictional cache mappings and genuine bootstrap hooks. They cover exact-key None and absence preservation; non-None identity denial with generic-permission overlap; other foreign negative entries including sibling/directory/exporter/prelaunch/lexical alias; ordinary permitted finder and loader details; pinned snapshot presence/absence; and child-main self-hash rejection before dispatch/helper side effects. Cache-state preservation is checked inside the fixture, and patched state is restored. Two existing regressions preserve startup-only phase separation and stale-stage denial.

The retained meaningful RED reaches the precise original failure: FIX2 rejects exact pinned launcher-key None as a foreign importer-cache path. The earlier fixture-hook failure is correctly retained separately and not counted as meaningful RED. The first eight-method run's only failure was a test diagnostic expectation (`hash mismatch` versus existing `pin mismatch`); the source denial and absent dispatch marker already occurred. That assertion was corrected without changing the implementation. Final saved evidence reports **8/8 PASS on Mac CPython3.11.13**, comprising six new methods and two relevant regressions. No historical40/19/9 suite is represented as rerun.

These tests adequately cover the bounded ruling and the demonstrated failure mode. They are not an actual Linux launcher stage, scientific import, helper-closure validation or successful startup. No additional probe or passed-suite rerun was necessary for this review.

## Preserved operational boundary

The original startup1 parent remains SHA256 `119b5447185f61fb4a5f578315c347445604b25adebbdc9a6288435768ebaa9f`; child remains `c35c6aacb3f2c3a6fd10af7c46fe8b9720b838ef2f9adbe0f81555050a8fedea`. Their terminal failed outcome, empty stages and unobserved finder value are unchanged. The local Mac plain-script observation is only a hypothesis-supporting observation; it does not fill the missing Linux value. A future non-None Linux entry will still reject, and another startup issue may appear.

FIX2 originals remain preserved, and the historical sample sidecar remains labeled FIX2 fictional rather than being silently promoted to FIX3 deployment input. Current separately owned production exporter state is outside this rereview. Any later attempt needs separately reviewed frozen copies, a new exact sidecar, unused evidence paths and a root startup-only release. No context/capture, external-package, native-isolation, scientific acceptance or Task54 completion follows from this source PASS.

Only this review file was written. Local work was reading, hashing and patch/data comparison. No tests, subprocess probes, SSH, deployment, target runtime, source edits, Cargo/build, Git mutation or subagents occurred.
