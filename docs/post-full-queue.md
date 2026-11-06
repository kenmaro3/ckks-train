# Post-FULL queue operation

Scope: prepare one background queue, locally. No server work, SSH, research runs, source-tree mutation, staging, commit, or deployment is performed by this implementation task. R-54 is excluded. Historical package names intentionally remain `hekit-*` at the three request versions.

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

Keep `post_full_queue.py` adjacent to the shell entry. Copy the unchanged existing `run_job.sh` to the configured runs directory if needed. Record the background PID, config/script/bundle SHA256 and state directory; confirm it waits without starting any build. This command is documented, not executed by this task.


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
