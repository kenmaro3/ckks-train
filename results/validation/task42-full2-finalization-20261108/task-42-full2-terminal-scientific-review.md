# Task42 FULL-2 terminal evidence — independent scientific review

2026-11-08. **SPEC: PASS for terminal evidence, extraction identity, and the specified numerical/accuracy evaluation. QUALITY: PASS within the provenance limits below. RQ1 unconditional scientific acceptance: NOT established.** FULL-2 is complete, but its normalized-both-502 KS criterion fails; same-security remains UNMET and full-setting ReBoot CPU timing remains UNMEASURED. This review does not edit or replace original measurements or existing M3 rulings.

## Terminal and custody conclusion

All12 local files under `results/quadruped/raw/R-41-FULL-2/` are exactly the parent capture's decoded original bytes, including sizes and SHA256; no extra local file is present. The capture is765,520 B, SHA256 `e9c318a83e6392a9589c5abad97918083fdfed710120370a65c2a67ca22e8571`. Original result is19,259 B, SHA256 `921d5a7c315497ccc2f4d309b19edd6f5718013abeb38f0b63eb06da00f93a62`. The raw capture retains `scientific_acceptance=false`; this review does not rewrite that operational field.

Runner meta and its final reread are byte-identical, with exit0, watchdog=false, start2026-11-07T03:19:58+09:00 and end2026-11-08T19:58:07+09:00. `post.txt` ends `JOB_DONE rc=0`; GNU time exit is0. Snapshot bytes and source marker match metadata. The command token sequences in cmd.txt, meta and raw provenance agree exactly. The worker source declared by both raw provenance and synced marker is `ff2c2ebc9d7946490bcf63f8b6be9f45fe67d49f`, tracked changes none. Workdir is /home/kmihara/hekit-r41full.

The archived `chain10.sh`, `r41.sh`, `run_job.sh` source is consistent with the command,12-thread caps, nice10,90GiB own-tree watchdog,55h projection limit, time/RSS fields and post terminal marker. The chain uses `set -u`, not fail-fast; chain completion alone is not substituted for this job's own exit/log/result evidence. The local script copies and source marker are recorded provenance, not a cryptographic identity of the executed ELF or deployed script at each instant.

The fresh observation records the old chain PID absent and includes the same final log messages. Collection metadata records one invocation with automatic_retry=false, timeout=false, SSH/controller exit0, matching capture length/hash, and empty stderr. No SSH was performed by this review. Meta.final is a second read of the same remote meta file, not an independent second run.

The collector's selected files do **not** include partial.txt. Thus missing_artifacts=[] is not proof that no partial marker ever existed. Positive full completion is supported directly by the complete ordered log, terminal exit, result and below-limit projection; no assertion about uncollected files is necessary.

## Measurements and prescribed comparison

- Registered condition:196→32→16, N=2^16, n16_boot, WhenNeeded, B48 in64 lanes, one ciphertext microbatch, layout_d256, reuse transpose,48,000 samples and1,000 updates. All log update indices0–999 occur once, each with48 samples,4 bootstraps and prediction=true. KS histogram is510×1369 +490×1415 =1,391,540. Counters independently sum1,161,540 rotations +222,000 relins +8,000 SSE =1,391,540 KS;4,000 bootstraps. The1000-entry measured and predicted boot schedules match.
- Initial weights are the declared ReBoot model/shuffle seed0, distinct from raw `training.seed=32` (RNG). Local seed0 W1/W2 hashes match the released manifest. The local index table contains1,000 correctly numbered48-sample rows, ten permutations of0..4799; the reference has1,000 LR values. Config uses B48,10 passes,4800 training examples,196 features,zero initial biases and10,000 test examples. The in-binary twin0.9041 equals the released per_seed.0 twin0.9041.
- Encrypted-trained accuracy **0.9035**, reference **0.9041**: **−0.06 percentage point**, within±0.5pp, **PASS**. This is n=1; no encrypted multi-seed mean/SD is measured. Maximum weight/bias difference is0.0066250310331289475.
- Measured training total146229.853279576s (40.6194h),3.046455277s/sample. Its ten-pass arithmetic mean is14622.9853279576s/pass, not a separately measured single-epoch time. The command flattens10 passes into binary epochs=1. Program wall146287.312211238s and runner wall146288.39505529404s are distinct from training total.
- GNU-time peak28694844 KiB equals raw memory29383520256bytes. Sampled tree peak28622228 KiB from14,568 RSS rows is a separate sampled maximum. Projection after10 updates independently recomputes142968s <198000s; it is not the actual final duration.
- CPU context is Linux/i9-13900KF, Rayon12, parallel feature enabled,32 logical CPUs; shared load exists. This is not a new GPU training result or a pinned/isolated CPU comparison.

Use the adopted196-input baseline37 KS/sample and502 KS/bootstrap, both **ReBoot model**, with ReBoot's own B48 and4 bootstraps/update. Ours111 KS/bootstrap is measured. K=1391540,H=4000,S=48000 gives nonboot K−111H=947540.

| Case | Own KS/sample | ReBoot KS/sample | Ratio | ≤0.70 |
|---|---:|---:|---:|---|
| a, own bootstrap costs |28.990416667|78.833333333 model|0.367743129|PASS|
| b, both502 |61.573750000 derived|78.833333333 model|0.781062368|**FAIL**|
| c, both111 |28.990416667|46.250000000 derived model|0.626819820|PASS|
| d, both boot-excluded |19.740416667 derived NI|37 model|0.533524775|PASS|

(b)–(d) are counter-derived alternatives, not timings/accuracies from separately run bootstrap implementations. (d) retains compact/repacking costs; it is not the interactive run. FULL-1's B64 KS results and FULL-2's B48 accuracy must stay explicitly paired to their separate configurations. Their successful components cannot be assembled into “one setting passed all conditions”.

## Required interpretation and limitations

1. Preserve original JSON unchanged. Its legacy `rq1.headline_reboot_side=lazy_measured` points to the784-input Task33 comparison. For196 use the adopted M0 model37 (also present in the raw m0_model branch), not35.0625. Its legacy `machine.threads` string mentions “this Mac” / quadruped pending; it is stale text. Use the consistent structured `provenance.machine`, `provenance.threads` and runner metadata for the actual Linux/Rayon12 statement.
2. A source HEAD/clean marker and command path are not executed-binary attestation or a run-time hash inventory for every input. The deployment record says the four arrays were checked when copied. Local weight/index/config/manifest checks support source consistency, but cannot prove the server consumed those exact bytes throughout this run. The raw result does not contain full run-time data/index/LR content hashes. Current source inspection is not silently treated as historical ff2c2eb code verification; no Git access/build was performed.
3. Existing M3/security rulings remain: ReBoot actual sparse h192 parameters fall below128 bits under the same estimator (without MITM127.8/126.2; with MITM121.8/121.5 BDGL16/MATZOV). Ours uses dense main + sparse SSE keys; same N does not establish same security. **Same-security UNMET**, not merely unknown because the summary tool says “unverified”.
4. Full196 ReBoot same-CPU timing is still unmeasured. Neither784 subset timings nor47.6h projection supplies it. Summary stdout's overall “pending” means the aggregator cannot establish unconditional acceptance; it is not a claim FULL-2 remains unfinished, and it does not override known security failure or ratio(b) failure.
5. No new finding blocks adoption of this bounded numerical result with those labels. This review does not close all Task42 documentation/archival work, Task54, stable release, or scientific gates outside the named criterion.

## Executed bounded checks

`python3 -B tools/rq1_summary.py --full-1 results/quadruped/raw/R-41-FULL-1/mnist196_ni_when_needed_b64_full_seed0.json --full-2 results/quadruped/raw/R-41-FULL-2/mnist196_ni_b48_full_seed0.json --reboot-config results/quadruped/raw/reboot_emlp1_full/config_values.json` → **exit0**. Both inputs pass the aggregator's declared-provenance/internal-consistency checks; FULL-2 accuracy gate PASS and(a,c,d)PASS/(b)FAIL, as above. Stdout only; no generated output file written.

Independent Python audit → **exit0** (raw output below). It separately compares capture bytes, commands, source/snapshots, all log rows, projection, RSS, local initial-weight hashes, index rows, and four ratios. Two initial audit attempts failed only in the review's index parser: first tried to parse a comment as int (`ValueError: invalid literal for int() with base 10: '#'`), second included the three leading step/epoch/batch columns (`AssertionError`). After reading the file's documented format, the parser excludes comments and validates those three columns separately. No input was changed and no result failure was suppressed. A first guessed data_manifest path under raw/ was absent; the actual manifest is under data/.

No test suite, constructor, native/HE/training/build, SSH, Git operation or subagent. Source/docs/results and all captured artifacts remain unchanged; only this review file was written.

```text
BYTE_MATCH meta.json 5088 fdd3c88ab5088fd9cd2d60d2e67b6d217a0e5db68acb6441c0bd9b5b52df6b85
BYTE_MATCH cmd.txt 551 7c52015307aefda9ce65567b46b18d00826aa903f606bcb6e0c61250c7fd6557
BYTE_MATCH workdir.txt 28 8a79a74525030b18d26a82514400e50f67ab2a0870ff75b9ac60eaa6333380f1
BYTE_MATCH pre.txt 704 a5d5d2453425c93985421320257fdc79c021c5850b2122b6c8fbddfd4e7a48fb
BYTE_MATCH post.txt 718 3ff94bb9a0255907cb58f38730899a29bcaa279b6178073ca456bd7a4e953957
BYTE_MATCH rss.tsv 364221 2468281efa0c64959e27247a044951375b531138a731dfbe4cbb81334c515196
BYTE_MATCH time.txt 1299 cfdd5b63683792512a15d5bccd2d4869109c78750970df7cca9b01b48c789823
BYTE_MATCH synced-commit.txt 290 e0c8969be845b272d6d194650a2ea5977529ebccd9996c0da0b01f6e94f9de4d
BYTE_MATCH projection.txt 72 87664f098ad30a520d4a81478c07ab336580f3fd502234ff089da37523c0080b
BYTE_MATCH program.log 174163 da70b366de933e1abe242ad468fec8f9dc80bc0c6abce57e1e728fed396930cc
BYTE_MATCH mnist196_ni_b48_full_seed0.json 19259 921d5a7c315497ccc2f4d309b19edd6f5718013abeb38f0b63eb06da00f93a62
BYTE_MATCH meta.final.json 5088 fdd3c88ab5088fd9cd2d60d2e67b6d217a0e5db68acb6441c0bd9b5b52df6b85
TERMINAL/COMMAND/SOURCE/SNAPSHOTS PASS
LOG 1000 ordered updates; samples 48000 KS 1391540 boots 4000 KS histogram {1369: 510, 1415: 490}
PROJECTION 142968 < 198000 RSS_ROWS 14568 maxKB 28622228
LOCAL_WEIGHT_MATCH init_weights_seed0/W1.npy 50304 92ab56cd70584de0410b43f06f014d9a1f474d0f57282dfc2b3a23a3e445bc69
LOCAL_WEIGHT_MATCH init_weights_seed0/W2.npy 4224 e2e5192130aea10834790f00d401d6a4a86fe508f5c472ddeb847b7b12db86f8
LOCAL_INDICES 48000; ten permutations of0..4799; LR1000; model/shuffle seed 0 JSON RNG 32
INDEPENDENT_RATIOS [(0.3677431289640592, True), (0.7810623678646933, False), (0.6268198198198198, True), (0.5335247747747748, True)]
ACCURACY_DELTA_PP -0.060000000000004494
train_seconds 146229.853279576 mean_ten_pass 14622.9853279576 runner_seconds 146288.39505529404
NO FILES WRITTEN; local data checks are not runtime all-input attestation
```

## Additional reviewed pins

| Input | SHA256 |
|---|---|
| docs/rq1-finalization.md |97499159b367e25dd3968083766e00405273188bee11869c317ee24286a5e12c|
| docs/m3-report.md, observed pre-finalization |3f8b551813d2889aa78a536153694a917f2c3a9956dbb58c79d1d14a4e00a31a|
| tools/rq1_summary.py |9380e740f5a7333607dfdce5966d0d793bf8b71fc03d1df5327390ce1b342c79|
| scripts/chain10.sh under results/quadruped |e9afd280f863bc52b5d3885c043030ca3d2ec069104f817bfef59c40d18bb3e1|
| scripts/r41.sh under results/quadruped |86c4e671002e01ccdd0f23a20dbb1e2feb6d7e9e0eecd9800cc505fe4f07bc9b|
| scripts/run_job.sh under results/quadruped |46d2e83cc3b7ec175813eab9072b49f3eebc7b592125b71c6512315a2146c736|
| reboot_emlp1_full/config_values.json |b89c9129177f5831cf044c775a2228fecace8214479aa54b1ad71b2d10a8ad18|
| reboot_emlp1_full/train_indices.txt |6c73caeebcec203bce24ffafaed6e6ad8a573fba8e28abd168169e391370e853|
| data/reboot_emlp1_full/data_manifest.json |0ddec87635b5c0e6326c49932a79e0cf8a035d25705b22d6f7d176fb1ee2dd98|
| fetch-v1.meta.json |3720756238c34da2cc7ec5e77849ce4b48f2bf6449525c8b5bcf5b6b6c51d889|
| fetch-v1.observation.stdout |e5db42adfb75c97115cea04f76a7b52677999dbb168886eafaf9a32a0ab0bf02|
| fetch-v1.observation.meta.json |a39ec81992ea84e9340c5bfae50e07c0a59e63c195bbfbe6121c721ad26fef57|

Last three refer to L/task-42-full2-terminal-fetch-v1.*. Original12 file pins are in the raw audit output above. M3 remains mutable under the root's separate finalization task; this is the inspected snapshot, not a freeze.

## Exact independent audit command

```sh
python3 -B - <<'PY'
from pathlib import Path
import json,hashlib,base64,re,shlex,datetime,collections
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');D=Path('results/quadruped/raw/R-41-FULL-2');P=L/'task-42-full2-terminal-fetch-v1.stdout';raw=P.read_bytes();c=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=='e9c318a83e6392a9589c5abad97918083fdfed710120370a65c2a67ca22e8571'
assert set(c['files'])=={p.name for p in D.iterdir()} and len(c['files'])==12
for n,r in c['files'].items():
 b=base64.b64decode(r['base64'],validate=True);assert b==(D/n).read_bytes() and len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256'];print('BYTE_MATCH',n,len(b),hashlib.sha256(b).hexdigest())
j=json.loads((D/'mnist196_ni_b48_full_seed0.json').read_bytes());m=json.loads((D/'meta.json').read_bytes());f=json.loads((D/'meta.final.json').read_bytes())
assert m==f and m['exit_status']==0 and m['watchdog_killed'] is False and c['scientific_acceptance'] is False
assert shlex.split((D/'cmd.txt').read_text())==shlex.split(m['command'])==shlex.split(j['provenance']['command_line'])
assert m['hekit_synced_commit']==(D/'synced-commit.txt').read_text().strip().splitlines();assert 'full '+j['provenance']['git_commit'] in m['hekit_synced_commit']
assert m['snapshot_before']==(D/'pre.txt').read_text();assert (D/'post.txt').read_text()==m['snapshot_after']+'JOB_DONE rc=0\n'
rows=re.findall(r'^mnist: epoch (\d+) update (\d+) \((\d+) samples, (\d+) cts\): ([0-9.]+) s, (\d+) KS \((\d+) bootstraps, weights booted\), matches prediction (true|false); elapsed (\d+) s,', (D/'program.log').read_text(),re.M)
assert len(rows)==1000 and [int(r[1]) for r in rows]==list(range(1000));assert all(r[0]=='0' and r[2]=='48' and r[3]=='1' and r[6]=='4' and r[7]=='true' for r in rows)
assert sum(int(r[5]) for r in rows)==j['measured']['total']['key_switches']==1391540;assert sum(int(r[6]) for r in rows)==4000
projection=int(float(rows[9][8])-sum(float(r[4]) for r in rows[:10])+sum(float(r[4]) for r in rows[:10])/10*1000+0.5);assert projection==142968
rss=[line.split() for line in (D/'rss.tsv').read_text().splitlines()[1:]];assert max(int(r[1]) for r in rss)==m['peak_rss_kb_tree_sampled_10s'];assert int(m['time_v']['Maximum resident set size (kbytes)'])*1024==j['memory']['peak_rss_bytes']
print('TERMINAL/COMMAND/SOURCE/SNAPSHOTS PASS');print('LOG',len(rows),'ordered updates; samples',sum(int(r[2]) for r in rows),'KS',sum(int(r[5]) for r in rows),'boots',sum(int(r[6]) for r in rows),'KS histogram',dict(collections.Counter(int(r[5]) for r in rows)));print('PROJECTION',projection,'<',198000,'RSS_ROWS',len(rss),'maxKB',max(int(r[1]) for r in rss))
refdir=Path('results/quadruped/raw/reboot_emlp1_full');ref=json.loads((refdir/'config_values.json').read_bytes());s=ref['per_seed']['0']
for w in s['init_weights']:
 b=(refdir/w['file']).read_bytes();assert hashlib.sha256(b).hexdigest()==w['sha256'];print('LOCAL_WEIGHT_MATCH',w['file'],len(b),hashlib.sha256(b).hexdigest())
index_rows=[[int(x) for x in line.split()] for line in (refdir/'train_indices.txt').read_text().splitlines() if line and not line.startswith('#')];assert len(index_rows)==1000 and all(len(r)==51 and r[:3]==[n+1,n//100+1,n%100+1] for n,r in enumerate(index_rows));ix=[x for r in index_rows for x in r[3:]];assert len(ix)==48000 and all(sorted(ix[n:n+4800])==list(range(4800)) for n in range(0,48000,4800));assert len(s['lr_per_step'])==1000
print('LOCAL_INDICES 48000; ten permutations of0..4799; LR1000; model/shuffle seed',s['seeds']['model_and_epoch_shuffle_seed'],'JSON RNG',j['training']['seed'])
K=1391540;H=4000;S=48000;N=K-111*H
vals=[(K/S)/(37+4*502/48),((N+502*H)/S)/(37+4*502/48),(K/S)/(37+4*111/48),(N/S)/37]
print('INDEPENDENT_RATIOS',[(x,x<=.70) for x in vals]);print('ACCURACY_DELTA_PP',100*(j['evaluation']['test_acc_encrypted_trained']-s['twin']['test_acc_argmax10']));print('train_seconds',j['time_s']['train_total'],'mean_ten_pass',j['time_s']['train_total']/10,'runner_seconds',m['wall_s_runner'])
print('NO FILES WRITTEN; local data checks are not runtime all-input attestation')
PY
```

