# Post-FULL queue operation

## 2026-11-06 10:49:44 JST: deployed waiting controller

2026-11-06 10:49:44 JST の起動後観測で、post-full コントローラは **waiting**、PID `1739838` / start ticks `467041669`、state ディレクトリは `owner.json` のみ、`clones_created=false` だった。これは待機コントローラの投入であり、R-47 / R-48 / R-52 のビルド・学習・測定開始ではない。R-54 は投入していない。

ユーザー承認に基づき、独立レビュー後の source commit `4b1c49943533bb9ff79493daee33a27b8237a18c` のスクリプトを root controller がアップロードし、4 ファイルの SHA256 照合後にバックグラウンド起動した。source bundle の HEAD は `09202cf5650aa76b9b30dd79a68dd615c75a91ae`（5076076 bytes）で、スクリプトの source commit とは区別する。

| 項目 | 観測値 / 固定値 |
|---|---|
| 起動観測 Unix time | `1793929784.6580677` |
| owner 開始 Unix time | `1793929784.594468` |
| controller | `/home/kmihara/hekit-runs/post_full_queue.py --config /home/kmihara/hekit-runs/post-full-config.json` |
| state / log | `/home/kmihara/hekit-runs/post-full-v1` / `/home/kmihara/hekit-runs/post-full-v1.log` |
| FULL clone | `/home/kmihara/hekit-r41full`, commit `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f` |
| chain10 identity | PID `1700979`, start ticks `465769819`, chain start Unix `1793917065.19` |
| chain terminal log | `/home/kmihara/hekit-runs/logs/chain10.log` |

FULL の最後の実進捗観測は **2026-11-06 10:21:08 JST、FULL-1 update 113（0 始まり）、114/750 更新完了**、elapsed 10956 s。10:30 の gate identity と 10:49 の待機起動観測から後続更新数を推定しない。FULL-1/2 の完了・研究合格は未確認である。

重い処理は、両 FULL の新鮮で一致する成功 post/meta（partial・watchdog なし）と `CHAIN10_DONE`、確認済み chain identity の非稼働をすべて検証するまで開始しない。PID 消失だけでは開始しない。実行中の FULL clone を変更しない。鍵認証のみ、sudo・新規クラウド・他テナントのプロセス操作なし、各ジョブ nice 10 / 12 threads、全体24 threads以下、逐次実行を維持する。2分を超える処理はバックグラウンドで実行する。

ゲート解放後の固定順序と新規 clone（この観測時点ではいずれも未作成）:

| 順 | 依頼 | 完全固定 commit | 新規 clone |
|---|---|---|---|
| 1 | R-47-3 example / optional nsys / original GPU tests | `bd5cc493db166a282d26b05b1e069f2a0ee629ef` | `/home/kmihara/hekit-post-full-clones/hekit-post-full-r47` |
| 2 | R-48-1 → 2 → 条件付きSGD → 3 → 4 | `0e075618ab6e28b0cf260f98feb7f80ee1ccf1f3` | `/home/kmihara/hekit-post-full-clones/hekit-post-full-r48` |
| 3 | R-52-1 → 2、各21更新の subset | `9305d2e1f4f9999296cd4b4aaf55ea631fbca78e` | `/home/kmihara/hekit-post-full-clones/hekit-post-full-r52` |

これらは改名前の `hekit-*` バイナリと依頼を再現する。R-54 と eMLP-2/3 全長は対象外。Task54 の縮小入口はレビュー済みだが、計画の全実験行列は実装未完で、拡張設計中である。データは完了した FULL clone からだけ読み、新規 clone で manifest SHA256 を検証する。GPU SKIP・OOM・partial・出力 provenance 不整合を測定合格に変換しない。raw JSON と canonical `~/hekit-runs/out/J` / `logs/J.log` を保存し、既存結果を編集しない。

アップロード・起動観測の SHA256:

| ファイル | SHA256 |
|---|---|
| `post_full_queue.sh` | `e104bb557a8bcf0795874f236f7d041711af7a95be1edbaebd964549d85046de` |
| `post_full_queue.py` | `1a5f780070e004ec826f21b7dcc187d9e9a65d44e694f2cd5e2f1831589a78f9` |
| `post-full-config.json` | `6cb87837324ddcf42c709a33fadb86c8ac6e73302b90dbe8835783adbd13e377` |
| `post-full-09202cf.bundle` | `bf6d2a5c5414fd7ba9f0aaacf2223addf56691483b10de7022fc734e88bdf91d` |

一次記録（gitignored ledger、値だけを公開文書へ転記）と SHA256:

| 記録 | SHA256 |
|---|---|
| `post-full-launch.stdout` | `af8b15748685c281b2733b51aedd8bcd5840ab47bf04f96f1ed64bc1456fab2f` |
| `post-full-config.json` | `6cb87837324ddcf42c709a33fadb86c8ac6e73302b90dbe8835783adbd13e377` |
| `post-full-bundle.json` | `0af531dc13953599610af4480ff1a0e943186dc7521afffc4820658693a67546` |
| `task-post-full-remote-gates.json` | `6cc6be87ab73aa2d40b1f8ad4c0e21949ee5ada0e3bde39a7909f20f720f1ac4` |
| `task-42-transport-readback.log` | `c0e77d321be654792cc53c9765d2cc2eda29bf082b5bb1b18cf7a91545fdeab7` |
| `task-55-rename-ci.json` | `c37cd1b6292beb6a5b636f4f5f6f2f77cff12dd8cd27035e71e832eae9c57864` |

起動後観測の一次記録は上記 ledger ディレクトリ `.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/` に保持する。この状態は観測時点の値であり、その後の完了や性能を意味しない。

## Controller behavior and configuration

The controller was prepared and independently reviewed locally; the observed deployment state is recorded above. The implementation fixtures did not run research jobs. R-54 is excluded. Historical package names intentionally remain `hekit-*` at the three request versions.

Order: wait -> R47 clone/build/example/optional nsys/original GPU tests -> R48 clone/data/build/tests/NAG GPU/(conditional SGD GPU)/matched CPU L7/CPU L31 -> R52 clone/build/data/eMLP2 subset/eMLP3 subset. Only one `run_job.sh` child exists at a time. It applies nice 10, all five thread caps = 12, GPU snapshots and the existing 90 GiB own-tree RSS watchdog. Clone/checkouts also use nice 10. There are no remote calls or broad process controls.

Pins are code constants: R47 `bd5cc49`, R48 `0e07561`, R52 `9305d2e`. All clones are new paths below configured `clone_root`; no existing clone is changed. The bundle must contain all three commits, and its configured SHA256 is checked before each clone. Disk must have at least 12 GiB free before each clone/build (configurable `min_free_bytes`).

Gate requires both FULL post.txt last lines `JOB_DONE rc=0`, matching meta exit 0 and no RSS watchdog, no `partial.txt`, exact job/workdir, expected synced commit prefix, start >= configured chain start and plausible end, fresh post markers; then the observed chain PID/start ticks must no longer be active, and its log must end `CHAIN10_DONE` with mtime after both FULL endings. Missing evidence waits (maximum 5 days); contradictory, failed, partial, stale or PID-reuse evidence stops without a clone/build. Missing PID by itself is never sufficient. The current time is deliberately not a suitable `not_before_epoch`: FULL-1 is already running.

The queue claims its state directory using atomic mkdir before waiting; existing state is always refused, including after interruption or completion. Terminal records persist and no automatic restart occurs. A distinct state/clone configuration is a distinct operator action and must not be used to bypass a claimed queue. Every job refuses pre-existing runner output. No active FULL clone writes occur; raw IDX files and .npy data are read only after gates pass, copied to new clones, and checked against SHA256 manifests before and after copying.

`run_job.sh`'s shell return is recorded but never used as the child outcome: fresh matching post/meta exit status is required. R48 failures/OOM are recorded and later blocks continue; only NAG exit 3 **plus a SKIP line** triggers SGD. NAG OOM does not trigger fallback. Both GPU exit-3 skips choose CPU NAG; a GPU SGD attempt otherwise chooses CPU SGD. Output rows additionally require `all_checks_pass: true`; GPU output requires `provenance.gpu`. SKIPs, comparison lines and complete canonical logs are retained. A build failure or missing/inconsistent runner terminal provenance stops the queue conservatively.

R52 uses exactly the pinned requests' 21-update indices and arguments. Its own Python watcher projects setup + mean(first five update times) * 21. Limits: eMLP2 12240 s (2*1.7h), eMLP3 23040 s (2*3.2h). An excessive projection writes `partial.txt`, then SIGTERMs only captured children below its verified time parent, with Linux start-tick checks against PID reuse. The time parent is allowed to finish naturally, and all captured identities must disappear within 60 seconds before advancing. It never signals chain10, the FULL trees, or other process trees. It does not reuse r41.sh because that script's fixed HOME paths and unbounded watcher wait are unnecessary here. The wrapper remains alive to write terminal provenance. A partial result is not a successful research result.

Production config (JSON, absolute paths; replace every placeholder with captured evidence):

```json
{
  "runs": "/home/USER/hekit-runs",
  "full_tree": "/home/USER/hekit-r41full",
  "full_commit": "ff2c2eb",
  "not_before_epoch": 0,
  "chain_pid": 1700979,
  "chain_start_ticks": "CAPTURE_FROM_PROC_STAT_FIELD22",
  "chain_log": "/home/USER/hekit-runs/ACTUAL_CHAIN10_STDOUT_LOG",
  "bundle": "/home/USER/hekit-runs/post-full.bundle",
  "bundle_sha256": "CAPTURE_BUNDLE_SHA256",
  "clone_root": "/home/USER/hekit-post-full-clones",
  "state": "/home/USER/hekit-runs/post-full-v1"
}
```

`not_before_epoch = /proc/stat btime + chain start ticks / getconf CLK_TCK`, captured while the actual chain PID still exists. Verify `/proc/PID/cmdline` and stdout fd path independently before filling the config. `0` above is a placeholder, never a deploy-ready value. The producer must validate nonempty JSON and captured fields; the queue must not consume an empty file from a failed SSH redirect.

Optional `runner` overrides the local run_job path for fixtures; `poll_seconds` defaults 30, `max_wait_seconds` 432000, `job_poll_seconds` 5. Fixture HOME and runs match the runner's hard-coded HOME layout. Production requires Linux /proc and absolute mutable paths outside the FULL clone.

Root-only eventual command, after review, commit, validated config, verified bundle and upload:

```bash
nohup bash "$HOME/hekit-runs/post_full_queue.sh" --config "$HOME/hekit-runs/post-full-config.json" > "$HOME/hekit-runs/post-full-v1.log" 2>&1 < /dev/null &
```

The command above is a template; do not run a second copy of the deployed controller. Keep `post_full_queue.py` adjacent to the shell entry. Copy the unchanged existing `run_job.sh` to the configured runs directory if needed. Record the background PID, config/script/bundle SHA256 and state directory; confirm it waits without starting any build. The actual root-controller launch is recorded in the deployed-state section; no duplicate launch is needed.


## Failure and recovery states

- Waiting: no clone/build/train has started; missing FULL or chain terminal evidence keeps the queue waiting until its bounded deadline.
- Gate failure/timeout: terminal.json records failure; no research work starts. Revalidate FULL job identity, exit/meta markers and chain evidence before preparing a new reviewed queue configuration.
- Child failure: canonical run_job log/meta/post files and per-job queue JSON preserve the actual child status. R48 continues after a recorded child failure; build or missing/inconsistent terminal provenance failures stop the queue.
- Partial: R52 records the projection and partial.txt, signals only verified children below its time parent, then waits for all captured identities and runner terminal provenance. This is incomplete research evidence.
- Interrupted/claimed: the atomic state directory persists. Never delete the state or rename it to force an automatic restart. Inspect owned runner processes and existing job output first; avoid starting a second copy. Recovery requires a separately reviewed configuration and distinct job identifiers or an explicitly reviewed collection-only action.
- Completed: queue completion describes execution state only. A successful process exit does not establish a measurement pass; inspect JSON checks, SKIP variants, provenance, coverage and actual recorded metrics before reporting findings.

The source bundle, validated config, scripts and state must be retained together with their SHA256 values. Do not fill a missing observation using an estimate. The example config and launch command above are templates, not evidence of a deployment or measured result.


## Independent review fix round (local only)

Three P2 findings were reproduced with RED fixtures before fixes: mismatched runner identity was accepted, output all_checks_pass without pinned raw provenance was accepted, and partial termination included the time parent without waiting for captured children to disappear.

Runner completion now requires exact job/workdir/full pinned commit/dirty_files 0/nice10/thread12/GPU flag/command identity. Both command records, meta and post must belong to this invocation and have fresh timestamps; runner start/end must lie within the invocation window (run_job date fields have one-second precision).

Raw output validation follows the actual historical `provenance.rs` and binary writers at `0e07561` and `9305d2e`: exact full git_commit, written_by, command_line, clean marker `git_modified_tracked_files: "none"`, build flags/version/target and parallel_feature, and started_unix/finished_unix within the actual job interval. R48 build = package 0.0.0, release, parallel+cuda; R52 = package 1.0.0, release, parallel without cuda. The pinned helper maps empty git status through cmd_out's "unknown" sentinel to "none"; an empty string is not substituted. GPU ablation must contain its actual name/free_mib_at_start/total_mib object; CPU ablation has gpu:null. The queue never edits raw JSON to fill a missing field or convert it into a pass.

Partial stopping retains each captured PID and Linux start ticks in partial-identities.json. Only descendants below the time process receive TERM; the time parent receives no signal and can finish its time/provenance output naturally. The queue waits for all captured identities, including the time parent, to disappear before advancing. The bounded wait defaults to 60 seconds (`partial_stop_timeout_seconds`); survivors cause terminal failure and no next job. Reused PIDs are no longer captured identities and receive no signal. A timeout requires operator review of the recorded identities; it is not permission to force a restart.

Verification after this fix round: 19/19 synthetic fixtures, shell syntax and Python compile checks pass. Tests cover valid actual R48/R52 output schemas, wrong/missing CPU/GPU provenance, wrong writer/commit/command/build/time, truthy fake GPU provenance, terminal identity mismatches, bounded survivor waiting, PID reuse, and child-only signalling. These are local synthetic checks, not remote GPU/training measurements.

## Result collection map prepared 2026-11-08

This is a collection plan checked against the deployed queue source above and the pinned test/binary sources. It is not a new server observation, an executed collector, or a measurement result. At the last actual observation (2026-11-08 14:22:14 JST), FULL-2 had completed 865/1000 updates, its final JSON was absent, and the queue state contained only `owner.json`. See the retained [observation](../results/quadruped/raw/R-41-FULL2-OBS-20261108-1408/). Do not launch another queue or alter the existing jobs while preparing collection.

The canonical server roots are `R=/home/kmihara/hekit-runs`, `S=R/post-full-v1`, and `C=/home/kmihara/hekit-post-full-clones`. These are notation for the table, not shell variables to execute. For each actual runner job `J`, preserve `R/logs/J.log` and the original files in `R/out/J/`: `meta.json`, `post.txt`, `cmd.txt`, `workdir.txt`, `synced-commit.txt`, `pre.txt`, `time.txt`, `rss.tsv`, and `time.pid` when present. Preserve `projection.txt`, `partial.txt`, and `partial-identities.json` when produced. A missing artifact is recorded as missing; it is never replaced with another job's record.

| Block | Exact queue/runner job names | Additional original evidence to collect |
|---|---|---|
| R47 | `post-full-r47-build`, `R-47-3-example`, optional `R-47-3-nsys`, `R-47-3-tests` | Corresponding `S/J.json`; optional profiler files under `C/hekit-post-full-r47/results/quadruped/raw/R-47-3/` |
| R48 build/tests | `post-full-r48-build`, `R-48-1` | `S/J.json`; test JSON files listed below under `C/hekit-post-full-r48/results/quadruped/raw/R-48-1/` |
| R48 ablation | `R-48-2`, conditional `R-48-2-sgd`, `R-48-3`, `R-48-4` | `S/J.json`, `S/R-48-optimizer.json`; selected binary JSON files listed below |
| R52 | `post-full-r52-build`, `R-52-1`, `R-52-2` | `S/J.json`; the two subset JSON files below, if written; partial/projection records remain evidence of incomplete runs |

Also preserve `S/owner.json`, `S/gate-released.json` when present, the observed `S/terminal.json` when present, and `R/post-full-v1.log`. A job record can be collected after that job's terminal provenance is complete while a later job continues; the whole queue need not finish first. An actively growing log is an observation at collection time, not a terminal log. The optional nsys-unavailable record has only a skip status and no runner job; do not invent a missing runner failure for this documented branch. Likewise, an unselected SGD branch is not an attempted measurement.

The anticipated filenames in M4 are presentation/archive destinations, not the names written by this deployed queue. In particular, the actual optimizer record is `S/R-48-optimizer.json`, not a generated `R-48-2.optimizer`; the actual R47 example and GPU test evidence is in the two canonical runner logs. If copies use the report's shorter names, retain a source-path/size/SHA256 mapping and copy original bytes unchanged. Never assemble a replacement binary-result JSON from queue summaries.

For R48 tests, the finite possible producer files are `{n16_train_full_keys,n16_train_level7,small_insecure}_{cuda,cpu}.json` and `iris_{seal_2020_timing,mini_boot_ni}_{cuda,cpu}.json` under the test directory above (ten files at most). GPU SKIP returns before the paired CPU run, so both expected outputs can be absent without having been run. Record each test's execution/SKIP/failure separately. Cargo's `5 passed` can include early SKIP returns. The mandatory level7 row requires its actual GPU and CPU JSON, successful test assertions, matching per-batch counters and total, predictor checks, and the final parameter comparison at the registered tolerance. In the pinned MNIST test, the final comparison flattens the complete `final_weights` object; in the Iris tests the assertions compare the emitted weight/bias error metrics. Report those actual validation scopes separately.

The R48 ablation files, relative to `C/hekit-post-full-r48/results/quadruped/`, are `ablation_cuda.json`, conditional `ablation_cuda_sgd.json`, selected `ablation_cpu_l7.json` or `ablation_cpu_l7_sgd.json`, and `ablation_cpu.json`. The R52 files, relative to `C/hekit-post-full-r52/results/quadruped/`, are `mnist196_emlp2_ni_b48_subset1008_seed0.json` and `mnist196_emlp3_ni_b48_subset1008_seed0.json`. These are producer paths, not a claim that the files currently exist.

Collection must be read-only, key-authenticated, nice 10, bounded in time and bytes, limited to these owned evidence paths, and use fresh local destinations. Before collecting, register the exact collection command and bounds in [requests.md](../results/quadruped/requests.md). Preserve raw stdout/stderr, exit status, file identities and hashes; check regular-file ownership and path/identity stability around reads. On refusal or changed files, preserve the observation and do not automatically retry or restart a job. Read-time hashes do not attest historical execution. Small original result/log evidence stays in the local repository; large optional profiler archives go directly to the verified MicroSD storage root under the existing storage policy, with no silent SSD fallback. Do not recursively copy build trees, datasets, or unrelated files.

A fixed R47-only collector has now passed independent source review and 15 tiny local synthetic checks; its [preparation evidence](../results/validation/r47-terminal-collector-preparation-20261108/) preserves the original failures and fixes. It has not been invoked remotely. The source limits original acquired bytes to 8 MiB per file and 32 MiB total, including rereads; a future controller must separately budget the encoded transport. It excludes large profiler files, the growing whole-queue log and R48/R52 artifacts from this first capture. Those remain part of the eventual intake above. Source approval alone does not release a controller or collection request, or establish that R47 has run.

### Acceptance after collection

1. Match each actual job's command, workdir, full pinned commit, clean marker, nice/thread limits, GPU flag, terminal timestamps, `JOB_DONE rc`, and watchdog state. The wrapper return code is not the child exit status. Preserve any contradiction as a failure to establish provenance. Keep all original raw artifacts unchanged.
2. Interpret queue statuses as execution summaries only. `done`/`completed` do not imply scientific acceptance; `done_with_skips` requires per-test coverage inspection. Exit 3 plus an actual SKIP line denotes an unmeasured capacity row; OOM, watchdog, assertion failures and invalid output remain failures. A missing future job is pending or unstarted after an earlier stop, not a zero-valued result.
3. For R47, separate default/max pool thresholds, the one-stream sequential x4 baseline, and the two-/four-stream batch x4 variants. Use the example's actual 30-repetition median/min/SD and equality checks. The optional five-repetition nsys run is separate profiling evidence. Preserve comparison with the older R-47-2 code as historical, not a substitute for R-47-3.
4. For R48, independently inspect binary provenance and all eight unique `(hoist,bsgs,lazy_relin)` configurations, checks and recorded samples. Preserve the four caveats in M4 §2.3. Compare CPU/GPU speed only for matching optimizer, level, dimensions/layout, batch, seed, data and timing definition; the level31 CPU row is a separate ablation. Level7 rows have one warmup plus five timed iterations, level31 one plus three, all within a seed. `iteration_s_median` in this writer is the sum of the step and refresh medians, not an independently measured median of paired total times. GPU pool checkpoints and capacity estimates are not memory peaks.
5. For R52, require the original binary provenance, actual update/sample coverage and registered data/index/initialization identities. A complete 21-update subset still does not establish the full ten-pass architecture result. Preserve partial projection and termination evidence without extrapolating it into a full timing or final accuracy. No automatic longer retry is authorized by collection.
6. Generate derived tables from the accepted originals, retaining source hashes, then update M4/M5 and the paper with separate pass/unmet/unmeasured outcomes. The scheduled publication cutoff can leave honest missing rows; it does not convert an incomplete experiment into a passing one. Review the exact data/report diff before commit/push. Keep the published `v1.0.0-preview.1` tag and its assets unchanged.
