# Task 47 R-47-3 terminal scientific review

Date: 2026-11-08. Repository: /Users/marconi/Documents/research/hekit.

SPEC: **PASS for captured terminal evidence, bounded measurement interpretation and generated-table attribution.**
QUALITY: **PASS for this scope.** No blocking finding in the captured results. This does not mean every Task47 GPU case ran, or that end-to-end training, exclusive-device peak performance, or a profiler trace analysis has been validated.

## Evidence and interpretation

The once-collected transport is 161433 bytes SHA256 `97678b9deed3b12f07982566a5bfd6b840833eb82431fd2eb2086c31ab8b8df9`. All 56 original file entries decode exactly to local files under `results/quadruped/raw/R-47-3/`, with matching recorded size and SHA256, totalling 108427 bytes. All four meta/state first-versus-final copies match. Controller records exit0, no timeout, no automatic retry, empty stderr; capture has no missing required artifacts and deliberately retains `scientific_acceptance=false`. That operational field is not rewritten by this review.

All four original jobs agree across child exit, wrapper exit, GNU time exit, post terminal `JOB_DONE rc=0`, queue state, command, workdir, and declared synced commit `bd5cc493db166a282d26b05b1e069f2a0ee629ef` / `dirty_files 0`. Build/example/nsys are `done`; tests are `done_with_skips`. Post gate release and build begin after FULL2's independently audited 19:58:07 JST completion. The runner wrapper wall time includes overhead and is not the measured matvec latency.

The original example, n16_train/d64/level5/four products, reports:

| Pool | Mode | Median ms | Min ms | Population SD ms |
|---|---|---:|---:|---:|
| default256MiB | sequential,1 stream |36.773|36.693|0.203|
| default256MiB | batch,2 streams |32.080|31.841|0.652|
| default256MiB | batch,4 streams |32.889|32.087|0.235|
| max u64 | sequential,1 stream |34.784|34.664|0.209|
| max u64 | batch,2 streams |29.173|28.940|0.222|
| max u64 | batch,4 streams |28.821|28.502|0.227|

These are 30 timed within-run repetitions after three warmups, not 30 independent seeds or replicates of the whole setup. The already-adopted pinned-producer review establishes device synchronization inside each timing, upper-middle sorted element as the 30-sample median, population SD, and residue equality assertions after timing prints. Exit0 is therefore relevant beyond the presence of timing lines. Individual sample timings were not emitted, so distribution, SD, and rounding cannot be independently recomputed from this log. CPU setup14.7s is not a CPU matvec latency.

Against each pool's own sequential median, rounded values imply approximately 1.146x/1.118x for default2/4 streams and 1.192x/1.207x for max2/4 streams. This is descriptive within-run improvement; no statistical significance claim or universal optimal stream/pool policy follows. Four streams are slower than two under default, slightly faster under max. Shared RTX4090 snapshots show VLLM using20660MiB, about3134MiB initially free and example1980MiB after evaluator setup; these are not peak/exclusive-device or GPU training results.

Tests print 5+17+13+10=45 harness passed, zero failed, but three successful-return test functions explicitly SKIP due to memory capacity. Only42 non-skipped test entries provide execution evidence. Unexecuted cases: full n16_train d64 matvec (8436.0MiB needed), full-key sequence/counts (2868.0MiB), full hoisted rotate16 (5816.0MiB), each against2485.9MiB free. Capped level5/9 matvec, capped batch/sequential cases, counts, other kernels, and error cases do execute. The batch oversized-stream test reports a preflight Device error (172x146MiB over total device capacity); the separate key-switch test reports a CUDA allocation OOM error. Neither justifies claiming successful execution of skipped full cases.

nsys is a separate5-repetition profiled run, with approximately46–49ms medians. Command metadata agrees with original cmd/state. The log reports CPU IP/backtrace sampling unsupported and reports generation of the .nsys-rep path. The profiler binary is intentionally excluded and absent locally: no timeline, kernel overlap, or profiler internals were reviewed. Its printed times must not replace, average into, or be compared as equivalent to the unprofiled30-repetition results.

## Generated tables and source boundary

The six R-47-3 example rows in Markdown/TeX preserve medians and min/SD/repetition notes. Historical R-47-2 Criterion rows remain separately labeled. No R-47-3 CPU speedup or HEonGPU ratio is manufactured. All233 R-47-3 provenance TSV lines point to the example originals, never nsys/tests. `python3 -B tools/paper_tables.py --check` independently exits0 with empty stdout/stderr. No generated output was written during this review.

Reviewed current finite reader `tools/paper_tables.py:1237–1375`, current table, and prior independent pinned-source record `task-54-r47-table-integration-review.md`. That prior review inspected the exact producer at bd5cc493, SHA256 `e416c28d7a0c6b033d880afe5e0a64e097e18436716be2897cc896a37f9ca8e3`, and device source SHA256 `630c4fb623ca96b9b1fd7d631f195a41b82ceb5562301169904f8dd462c8c755`. This turn did not invoke Git or retrieve the historical source again. A read attempt at the old current-tree `crates/hekit-cuda/examples/batch_streams.rs` path found it absent; it is not substituted with a newer source. The prior exact-source review, rather than a fresh source inspection, supplies those implementation semantics.

Capture hashes prove byte identity at collection/review time, not a historical ELF/input/runtime attestation. Synced clean-source metadata supplies provenance but is not a binary hash proof. No independent native rerun, SSH, training, build, PDF, Git operation, or source/docs/results edit was performed. Historical external HEonGPU claims and other historical table rows were not independently re-audited in this result intake.

## Independent bounded command and raw output

One initial audit-harness attempt used an incorrect expected post marker `exit_status: 0`, failed an assertion after successfully checking all56 files, and exited1. Inspection showed the actual established marker is `JOB_DONE rc=0`; the corrected script below passed. This was a reviewer probe correction, not a result/source change.

Initial output:
```text
56 original files exact, 108427 bytes; capture pins all match
Traceback (most recent call last):
  File "<stdin>", line 9, in <module>
AssertionError

```

Corrected command:
```sh
python3 -B - <<'PY'
from pathlib import Path
import json,hashlib,base64,re,subprocess
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');root=Path('results/quadruped/raw/R-47-3');d=json.loads((L/'task-47-terminal-fetch-v1.stdout').read_bytes());pins=[]
for name,item in sorted(d['files'].items()):
 b=base64.b64decode(item['base64'],validate=True);p=root/name;assert b==p.read_bytes();assert len(b)==item['bytes'];assert hashlib.sha256(b).hexdigest()==item['sha256'];pins.append(f'{item["sha256"]} {len(b)} {name}')
assert len(pins)==56;assert sum(x['bytes'] for x in d['files'].values())==d['original_bytes_collected']==108427
print('56 original files exact, 108427 bytes; capture pins all match')
for job in d['jobs']:
 p=root/job;m=json.loads((p/'meta.json').read_bytes());s=json.loads((p/'state.json').read_bytes());assert (p/'meta.json').read_bytes()==(p/'meta.final.json').read_bytes();assert (p/'state.json').read_bytes()==(p/'state.final.json').read_bytes();assert m['exit_status']==s['rc']==s['wrapper_rc']==0 and m['watchdog_killed'] is False;assert m['time_v']['Exit status']=='0';assert (p/'post.txt').read_text().splitlines()[-1]=='JOB_DONE rc=0';assert m['hekit_synced_commit']==(p/'synced-commit.txt').read_text().splitlines();assert m['hekit_synced_commit']==['commit bd5cc49','full bd5cc493db166a282d26b05b1e069f2a0ee629ef','dirty_files 0'];print(job,s['status'],m['start'],m['end'],'runner seconds',m['wall_s_runner'],'GNU time seconds',m['time_v']['Elapsed (wall clock) time (h:mm:ss or m:ss)'])
log=(root/'R-47-3-tests/program.log').read_text();summ=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed;',log);assert sum(int(x[0]) for x in summ)==45;assert all(int(x[1])==0 for x in summ);skips=[x for x in log.splitlines() if 'SKIP' in x];assert len(skips)==3;print('harness 45 ok, 3 capacity skips, 42 non-skipped');print('\n'.join(skips))
for label,a,b in [('default 2',36.773,32.080),('default 4',36.773,32.889),('max 2',34.784,29.173),('max 4',34.784,28.821)]:print(label,'speedup',a/b,'latency reduction %',100*(1-b/a))
r=subprocess.run(['python3','-B','tools/paper_tables.py','--check'],capture_output=True,text=True,timeout=60);print('TABLE CHECK',r.returncode,'STDOUT',repr(r.stdout),'STDERR',repr(r.stderr));assert r.returncode==0
print('ORIGINAL PINS');print('\n'.join(pins))
for p in [L/'task-47-terminal-fetch-v1.stdout',L/'task-47-terminal-fetch-v1.meta.json',L/'task-54-r47-table-integration-review.md',Path('tools/paper_tables.py'),Path('results/tables/gpu_vs_cpu_ops.md'),Path('paper/tables/gpu_vs_cpu_ops.tex'),Path('results/tables/sources.tsv')]:
 b=p.read_bytes();print('REVIEW PIN',len(b),hashlib.sha256(b).hexdigest(),p)
PY
```

Exit: 0. Raw output:
```text
56 original files exact, 108427 bytes; capture pins all match
post-full-r47-build done 2026-11-08T19:58:17+09:00 2026-11-08T19:58:27+09:00 runner seconds 10.049123525619507 GNU time seconds 0:04.82
R-47-3-example done 2026-11-08T19:58:32+09:00 2026-11-08T19:59:02+09:00 runner seconds 30.122881412506104 GNU time seconds 0:23.57
R-47-3-nsys done 2026-11-08T19:59:07+09:00 2026-11-08T20:00:07+09:00 runner seconds 60.36249303817749 GNU time seconds 0:52.09
R-47-3-tests done_with_skips 2026-11-08T20:00:12+09:00 2026-11-08T20:03:33+09:00 runner seconds 201.08256840705872 GNU time seconds 3:20.74
harness 45 ok, 3 capacity skips, 42 non-skipped
test matvec_d64_matches_cpu_n16_train_full ... SKIP matvec_d64_matches_cpu_n16_train_full: needs 8436.0 MiB of GPU memory, only 2485.9 MiB free
test sequence_and_counts_match_cpu_n16_train_full_keys ... SKIP sequence_and_counts_match_cpu_n16_train_full_keys: needs 2868.0 MiB of GPU memory, only 2485.9 MiB free
test hoisted_rotate_16_matches_cpu_n16_train ... SKIP hoisted_rotate_16_matches_cpu_n16_train: needs 5816.0 MiB of GPU memory, only 2485.9 MiB free
default 2 speedup 1.1462905236907732 latency reduction % 12.762080874554716
default 4 speedup 1.1180941956277175 latency reduction % 10.562097190873743
max 2 speedup 1.1923353786035034 latency reduction % 16.130979760809574
max 4 speedup 1.2068977481697372 latency reduction % 17.142939282428692
TABLE CHECK 0 STDOUT '' STDERR ''
ORIGINAL PINS
f60c0ae8a42744499de58ff24c95db487dd8af0afc63c49feca3b5313d68f54c 39 R-47-3-example/cmd.txt
af432f72f440560911e16feebfc047258c822f756fdb6cc234c713eaad100a49 7852 R-47-3-example/meta.final.json
af432f72f440560911e16feebfc047258c822f756fdb6cc234c713eaad100a49 7852 R-47-3-example/meta.json
87927caf5dd61c87a02f350f687e37f490a709e4dd5d090aae28bd2e3fb76a9d 2698 R-47-3-example/post.txt
96e4c1aa00a4e06453afc6a8e06288ea93e00bc4774638185bd1c9eead85ed16 2684 R-47-3-example/pre.txt
7ce4fe8d3a63f3391117ae050e1cfbf200d0131d5902315cb59ddf3fdd54e3a5 1112 R-47-3-example/program.log
5fc91711fcb9966845c6c5419cb7ff68acae14d5ae3ec729ef2860f8f2021adb 96 R-47-3-example/rss.tsv
4e6a087ccd329bf1e25585c4f44fa36abcdc484f42eadc92a01597053e5b475a 339 R-47-3-example/state.final.json
4e6a087ccd329bf1e25585c4f44fa36abcdc484f42eadc92a01597053e5b475a 339 R-47-3-example/state.json
2a1436691b69d15575a5ab6043b384440744d76f387bf4199a632b8295363944 75 R-47-3-example/synced-commit.txt
b18d54ef65a14bc501a4e3b317aded863a193c5dc9138f8b6544bce640a9c733 8 R-47-3-example/time.pid
0861e71a488d5df1792de21f47cec398efa7d3b66f4a877f362a082d4e7c68d4 768 R-47-3-example/time.txt
e16f71bde438e00f69ae85988b7feeac709d5cc63ebe1a310b62619d6d110e07 57 R-47-3-example/workdir.txt
db76140d1e0b7dbb89f50c26c40bc4dea3733652f7799600b4936137d8be2826 157 R-47-3-nsys/cmd.txt
c366972a4a3e5d154ed9f2c0bf700b1cfaaf85d883391311ba67bcb91a0501fe 8077 R-47-3-nsys/meta.final.json
c366972a4a3e5d154ed9f2c0bf700b1cfaaf85d883391311ba67bcb91a0501fe 8077 R-47-3-nsys/meta.json
6108869e189c06193bc5fdc4abcbc3f69603a8b3f0706d78af0eba7cb26203aa 2698 R-47-3-nsys/post.txt
526c5cecdbda9d409e526d3f0cc67be60b9886b18911fe254cc54af0d66cd500 2684 R-47-3-nsys/pre.txt
eb4ace9da2e192c6e9d32daadc84a4f162f089eb17a9f3f68ea82103bc62ce37 7279 R-47-3-nsys/program.log
c2b81e6d2010edd6136a7fbf2bfd6efc4626e0303a398d3a8aab3c7ebe77627c 161 R-47-3-nsys/rss.tsv
346510963371480ae42aae7c86186823c3fb02194c1a38af69e893f5c8f2ede4 493 R-47-3-nsys/state.final.json
346510963371480ae42aae7c86186823c3fb02194c1a38af69e893f5c8f2ede4 493 R-47-3-nsys/state.json
2a1436691b69d15575a5ab6043b384440744d76f387bf4199a632b8295363944 75 R-47-3-nsys/synced-commit.txt
037dcd0fc178a70491fa1b451c78b86918a501af8c7d4a23a7713cd29aed93d8 8 R-47-3-nsys/time.pid
ed9fc1debdb160287538212e3b532a154986d5d75009b16b628954e6370206bf 880 R-47-3-nsys/time.txt
e16f71bde438e00f69ae85988b7feeac709d5cc63ebe1a310b62619d6d110e07 57 R-47-3-nsys/workdir.txt
fc2b869d8b19fb05ab8180946edc639a79966d725ac96f79a6661d1a5fcdfb7f 94 R-47-3-tests/cmd.txt
d9e6a198db47af782a510741854c9c9716e9a041ed2056972658ceff0d669dd6 7970 R-47-3-tests/meta.final.json
d9e6a198db47af782a510741854c9c9716e9a041ed2056972658ceff0d669dd6 7970 R-47-3-tests/meta.json
aa157c007f8fc1630cae7674c378b7e230086b30c8c7bb4357296eb3d683cf54 2698 R-47-3-tests/post.txt
6ae8faf671a354919672a98e3cd55cfe7a43fb9cfa75d4a764c26001ed272a2f 2684 R-47-3-tests/pre.txt
39e6a5321f50104bf4472c7a45b067d8e83f4aae70f6000080d0972417a649b1 12697 R-47-3-tests/program.log
961c389db7462c331eb7c026f784f38ad870cfbba29d6be452191ca841f91e24 503 R-47-3-tests/rss.tsv
9f99192773039f64ee1ef348d5c51dd1b414408f5c25ff5041a0afcc95a2435a 2151 R-47-3-tests/state.final.json
9f99192773039f64ee1ef348d5c51dd1b414408f5c25ff5041a0afcc95a2435a 2151 R-47-3-tests/state.json
2a1436691b69d15575a5ab6043b384440744d76f387bf4199a632b8295363944 75 R-47-3-tests/synced-commit.txt
00332d0d2223a1780f4d4dba59dcf799ad39a60424b0f9068b73f04fe483f398 8 R-47-3-tests/time.pid
98a049aac7b67ff5c5e7c45103a8f3f0af787a6499f4e05c606689b5bf5a6c9e 831 R-47-3-tests/time.txt
e16f71bde438e00f69ae85988b7feeac709d5cc63ebe1a310b62619d6d110e07 57 R-47-3-tests/workdir.txt
30a69ffbbcad879c29fe1b8a8ac5efb94e8801cdf7fcf0d35e64e09893c23de4 633 gate-released.final.json
30a69ffbbcad879c29fe1b8a8ac5efb94e8801cdf7fcf0d35e64e09893c23de4 633 gate-released.json
4cfb85c83bf1578242c3858d5e6ec0c8e119adef85d033664afe6166327edb79 53 owner.final.json
4cfb85c83bf1578242c3858d5e6ec0c8e119adef85d033664afe6166327edb79 53 owner.json
24ea2bae74cfe91517d9322bbb5aa2784146d27a8044ec2c47025f051b3d47c1 77 post-full-r47-build/cmd.txt
e929038c22db430c7ae17a962a307182c84a296f93d25d4ec350b0337f67ca6a 3926 post-full-r47-build/meta.final.json
e929038c22db430c7ae17a962a307182c84a296f93d25d4ec350b0337f67ca6a 3926 post-full-r47-build/meta.json
2356501b33a32d935284ae723873ec9f641020081f07465a6d3433128969046f 718 post-full-r47-build/post.txt
3c24c87ad713d0b3afcf6a4d05230f1815e5378892779810497132edf5c46466 704 post-full-r47-build/pre.txt
156121c722e99edf2d22351f3806e9334e94649229bff46c5ad843e84d94f1c3 874 post-full-r47-build/program.log
75e5f682d68bbdeade08d724234c11903e3cda8da50acc57699b53ea58a19e0d 48 post-full-r47-build/rss.tsv
608b3ba305936a4331407764515d9777aa10a0f719207552f085546dbfc7309f 443 post-full-r47-build/state.final.json
608b3ba305936a4331407764515d9777aa10a0f719207552f085546dbfc7309f 443 post-full-r47-build/state.json
2a1436691b69d15575a5ab6043b384440744d76f387bf4199a632b8295363944 75 post-full-r47-build/synced-commit.txt
f430184044d1828597a0f75c1225ff2163c6604156f29645cb3256926760fa7f 8 post-full-r47-build/time.pid
6fc7aa807b659ad7df5c7a6bdbdb6ef65782dc8dbf7c9edfd3970aea0a5724b2 809 post-full-r47-build/time.txt
e16f71bde438e00f69ae85988b7feeac709d5cc63ebe1a310b62619d6d110e07 57 post-full-r47-build/workdir.txt
REVIEW PIN 161433 97678b9deed3b12f07982566a5bfd6b840833eb82431fd2eb2086c31ab8b8df9 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-47-terminal-fetch-v1.stdout
REVIEW PIN 1679 cb60c6a8012b87d04919d0c0df5d94ffdbdc4ba1cdf588b54e67db1cdfdfa534 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-47-terminal-fetch-v1.meta.json
REVIEW PIN 13727 feb3792d3ab83a478f5835102a9ec54a02978c6f21ac4c7bafde5e4c90f675c4 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-r47-table-integration-review.md
REVIEW PIN 106815 f7e3774f7124fe24e5a7a882df902ea7584aaa62ece7855a43def5d880af6b38 tools/paper_tables.py
REVIEW PIN 5347 082916963d7da22dd5b559cca82a75e8a5fdc44ba4e087880181982f5cfff1c0 results/tables/gpu_vs_cpu_ops.md
REVIEW PIN 5598 3baea336bd2cb9e2904c89e8aed89d9b2092eb1493df9a58e2f9237533e3ccb8 paper/tables/gpu_vs_cpu_ops.tex
REVIEW PIN 195246 ad80f369f0cd6e5cff153583d7049d0b90b594a75a14ae33a0b638a9908f8f34 results/tables/sources.tsv

```

## Final documentation, archive and status review

SPEC: **PASS** and QUALITY: **PASS** for the parent-supplied final M4/table diff, raw README/manifest, validation archive, and added handover/M5/request status diff. No unresolved blocking finding.

M4 §1.1b reproduces all six values, min/SD, timing conditions and same-pool ratios correctly. Its rounded latency reductions12.8/10.6/16.1/17.1% match independent calculations. It preserves pre-fix R47-2 separately, three unexecuted full cases, the42-versus45 distinction, profiler intervention and missing binary, available-after versus peak, and provenance versus executed-ELF attestation. Training remains sequential; default256MiB/trim rules remain; no R48/R52/Task49 performance or M4-wide success is promoted.

One minor stale sentence was reported: §1.1a quoted a proposed `[default 256 MiB, ~threshold 0 at n16_train]` log line absent from the actual output, and described remeasurement in future tense. Parent corrected it to actual `pool release threshold: default (268435456 bytes)` / `[default]` syntax and completed tense. The corrected current sentence was read and the final diff hunks match current files. This finding is closed.

Raw SHA256SUMS covers exactly the56 captured originals. Validation SHA256SUMS covers9 files (README plus8 transport/source copies), excluding itself. Every hash matches and all8 copies exactly match ledger originals. Raw README is intentionally outside the original-only56-file manifest; its reviewed hash is recorded below. No profile binary is represented as recovered. Repeated .final files are explicitly repeated observations, not independent jobs. A later addition of this review to the archive would require the parent to update its manifest; this review does not claim to have audited a future archive copy of itself.

The added handover/M5/request status records accurately distinguish the20:06 triggering observation from the20:24 collection; old timestamped statements remain historical. They preserve R48/R52 terminal absence from this intake, Task54 unfinished source work, incomplete M4/M5/stable/arXiv status, and limits of prior CI. All added local Markdown destinations exist; M4 §1.1b fragment names match its heading-derived slug. External URLs were not fetched in this bounded local review.

Independent archive/diff command, exit0:
```sh
python3 -B - <<'PY'
from pathlib import Path
import json,hashlib,re
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');raw=Path('results/quadruped/raw/R-47-3');arc=Path('results/validation/task47-r47-3-adoption-20261108');d=json.loads((L/'task-47-terminal-fetch-v1.stdout').read_bytes())
for directory in [raw,arc]:
 entries=[]
 for line in (directory/'SHA256SUMS').read_text().splitlines():
  h,name=line.split(None,1);name=name.lstrip('*');b=(directory/name).read_bytes();assert hashlib.sha256(b).hexdigest()==h,name;entries.append(name)
 if directory==raw:assert set(entries)==set(d['files'])
 else:assert set(entries)=={x.name for x in directory.iterdir() if x.is_file() and x.name!='SHA256SUMS'}
 print('MANIFEST PASS',directory,len(entries))
for p in arc.iterdir():
 if p.name not in ['README.md','SHA256SUMS']:
  assert p.read_bytes()==(L/p.name).read_bytes(),p;print('LEDGER BYTES EXACT',p.name)
for p in [raw/'README.md',arc/'README.md']:
 for dest in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  target,_,anchor=dest.partition('#');q=p.parent/target;assert q.exists(),(p,dest);print('LINK EXISTS',p.name,dest)
for filename in ['task-47-r47-3-report.diff','task-47-status-records.diff']:
 txt=(L/filename).read_text();sections=re.split(r'(?=^diff --git )',txt,flags=re.M)
 for sec in sections:
  if not sec:continue
  dest=re.search(r'^\+\+\+ b/(.+)$',sec,re.M)[1];lines=Path(dest).read_text().splitlines(keepends=True)
  chunks=re.split(r'(?=^@@ )',sec,flags=re.M)[1:]
  for chunk in chunks:
   hdr,*body=chunk.splitlines(keepends=True);m=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',hdr);start=int(m[1])-1;n=int(m[2] or 1);new=[x[1:] for x in body if x.startswith((' ','+'))];assert lines[start:start+n]==new,(dest,hdr)
  print('DIFF CURRENT HUNKS MATCH',filename,dest,len(chunks))
paths=[Path('docs/m4-report.md'),Path('docs/handover.md'),Path('docs/m5-report.md'),Path('results/quadruped/requests.md'),L/'task-47-r47-3-report.diff',L/'task-47-status-records.diff',raw/'README.md',raw/'SHA256SUMS']+sorted(arc.iterdir())
for p in paths:
 b=p.read_bytes();print('PIN',len(b),hashlib.sha256(b).hexdigest(),p)
PY
```

Raw output:
```text
MANIFEST PASS results/quadruped/raw/R-47-3 56
MANIFEST PASS results/validation/task47-r47-3-adoption-20261108 9
LEDGER BYTES EXACT task-47-terminal-fetch-v1.observation.stdout
LEDGER BYTES EXACT task-47-terminal-fetch-v1.stdout
LEDGER BYTES EXACT task-47-terminal-controller.py
LEDGER BYTES EXACT task-47-terminal-fetch-v1.stderr
LEDGER BYTES EXACT task-47-terminal-fetch-v1.observation.meta.json
LEDGER BYTES EXACT task-47-terminal-collector.remote.py
LEDGER BYTES EXACT task-47-terminal-fetch-v1.launch.json
LEDGER BYTES EXACT task-47-terminal-fetch-v1.meta.json
LINK EXISTS README.md ../../../../docs/m4-report.md#11b-修正後r-47-3-低レベル独立4積の再測定
LINK EXISTS README.md ../../../validation/task47-r47-3-adoption-20261108/README.md
LINK EXISTS README.md ../../quadruped/raw/R-47-3/README.md
DIFF CURRENT HUNKS MATCH task-47-r47-3-report.diff docs/m4-report.md 6
DIFF CURRENT HUNKS MATCH task-47-r47-3-report.diff paper/tables/gpu_vs_cpu_ops.tex 2
DIFF CURRENT HUNKS MATCH task-47-r47-3-report.diff results/tables/gpu_vs_cpu_ops.md 2
DIFF CURRENT HUNKS MATCH task-47-r47-3-report.diff results/tables/sources.tsv 1
DIFF CURRENT HUNKS MATCH task-47-status-records.diff docs/handover.md 2
DIFF CURRENT HUNKS MATCH task-47-status-records.diff docs/m5-report.md 3
DIFF CURRENT HUNKS MATCH task-47-status-records.diff results/quadruped/requests.md 1
PIN 43602 046cf855b3900b5b6672fefb96a6a192b0187a3c3cd4daca013de879f5560f4d docs/m4-report.md
PIN 217654 3d253facfe250ea3556008bb6d8d668d2e35c6ac3e916aa1a22a3091fa3cd187 docs/handover.md
PIN 180255 48d20825891f1e415e0a4995b25bf18d9be172724c5c16ad56e397d0c96a821f docs/m5-report.md
PIN 165659 eea9c5bd0e7ae8e8b08f820add6de598fd63bb1b15e6fb12732b645b63b7874a results/quadruped/requests.md
PIN 64781 24db8dcc80864d9b9e7424bdbe7ec7a06bd00736f72940d876b962cd7000e568 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-47-r47-3-report.diff
PIN 14099 f3291647bd445e9fdffaf1c18c0e1d294bd6caf9cb12ebdbcffe4d0d8afbd4bc .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-47-status-records.diff
PIN 2177 a81fc85d25f1c16004124a86e80a0e3b36ea6d27a759ea9d7a5aac0e8147966b results/quadruped/raw/R-47-3/README.md
PIN 5136 a99bd64e2b381b3097ec2c86e1056b69b5ffd67e7a2c405274daa5ac4b6bebe3 results/quadruped/raw/R-47-3/SHA256SUMS
PIN 1166 915f66adf41e38c99711e4cdeff298e93ee71dce4807312ef51bc6118fc0a18b results/validation/task47-r47-3-adoption-20261108/README.md
PIN 905 b6bf83486a5d506f8f49d7a484d9ab9cd120feb77a786d3e5d3526c1830230f4 results/validation/task47-r47-3-adoption-20261108/SHA256SUMS
PIN 11201 8e40fb7efa1cfdb90f826fe4b8b8c6046a876616f695790d5e5c27975c2a0d5a results/validation/task47-r47-3-adoption-20261108/task-47-terminal-collector.remote.py
PIN 7653 4de62d14af065aee39ae9c74e9e82b6bd71e39f921a0936901961beb7cc8a6d6 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-controller.py
PIN 1253 02bde4021d5f663ac9395c98b3eaec772d0f153e8cdef51c9b40394ad3016778 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.launch.json
PIN 1679 cb60c6a8012b87d04919d0c0df5d94ffdbdc4ba1cdf588b54e67db1cdfdfa534 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.meta.json
PIN 1201 a39ec81992ea84e9340c5bfae50e07c0a59e63c195bbfbe6121c721ad26fef57 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.observation.meta.json
PIN 2317 e5db42adfb75c97115cea04f76a7b52677999dbb168886eafaf9a32a0ab0bf02 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.observation.stdout
PIN 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.stderr
PIN 161433 97678b9deed3b12f07982566a5bfd6b840833eb82431fd2eb2086c31ab8b8df9 results/validation/task47-r47-3-adoption-20261108/task-47-terminal-fetch-v1.stdout

```

Additional read-only inline Python checks: all four cmd/meta/state argv and workdir records agree; all233 R47 TSV references are example-only; no .nsys-rep exists under the local archive; all added local Markdown destinations and M4 §1.1b anchor slugs resolve. Exit0. No table regeneration, tests, native execution or Git operation was used for this final documentation pass. Only this review file was written.
