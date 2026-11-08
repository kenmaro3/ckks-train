# Operational review: 21:05 queue observation and cargo-deny installation failure

Date: 2026-11-08. Reviewed frozen CI archive `results/validation/ci-deny-download-20261108/` including its later6910b14 terminal observation, and `results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/`.

SPEC: **PASS**. QUALITY: **PASS** for record integrity and bounded operational interpretation. No actionable findings. This is not a scientific-result acceptance or whole-current-CI certification.

## CI diagnosis

The original6910b14 job log128720918474 shows the resolved EmbarkStudios action SHA `3c6349835b2b7b196a839186cb8b78e02f7b5f25`, cargo-deny0.20.2 release URL, three Docker build invocations, two automatic backoff/retry warnings, and `gzip: invalid magic` followed by unsuccessful tar/Docker build. It never reaches the action's Run/audit step. The job therefore provides **no license or forbidden-dependency verdict**. Repeated error-summary copies are not counted as extra attempts.

The40211e5 job128722911533 uses the same action SHA/version, has one initial failed Docker build and one automatic retry, then reaches the action and prints `bans ok, licenses ok` at11:57:22 UTC. This supports an intermittent installation/download failure rather than a demonstrated policy failure. The upstream curl pipeline records neither response status nor body: a precise HTTP/transport/service cause cannot be inferred. No retry-loop, workflow patch, dependency exemption or policy relaxation is warranted by this evidence.

The diagnosis records identical workflow/deny.toml/Cargo.lock/workspace Cargo.toml object IDs at b7c51dd,6910b14,40211e5. The three recorded maps are equal; independently hashing current file bytes in Git-blob format also matches all four recorded IDs. No Git lookup was performed, so historical commit-to-object associations remain supported by the parent's diagnosis rather than a fresh independent history query.

The later6910b14 terminal observation matches the ledger original and establishes overall failure: deny failed, the other five jobs including test succeeded. The preceding2f7b8bd and b7c51dd observations show all six successful. The saved40211e5 observation still has five successes and test in progress with no overall conclusion. Its successful deny job is neither a retroactive6910b14 pass nor a completed40211e5 whole-CI pass. README preserves these distinctions and limits them to exact observed sources; active Task54 WIP, GPU measurements and stable release are not validated.

Job API metadata records read-only `gh api .../actions/jobs/.../logs` retrieval commands with exit0 and their observation times. No API/network command was executed by this reviewer. The README's watcher exit1/time and bounded credential-pattern scan are parent-recorded operational context; the archived completed-run JSON independently supports the failure conclusion, but does not itself encode watcher process exit/time. This review did not repeat a secret audit or prove absence of all manual actions; the saved logs specifically establish automatic Docker retries.

## Queue observation and interpretation

All eight source/command/controller/launch/transport/outcome records exactly match ledger originals; stdout2401B SHA256 `43205f41e1377604bd7e0d1747c38bbe644723215c88185df891bb6b9c633d64`. Command/source digests, exclusive launch record and final transport metadata agree. Controller code invokes its supplied read-only source once, preserves raw bytes and exits with SSH's recorded0; it contains no automatic retry. The remote source inspects only fixed parent identities/own children, fixed FULL2 tail/existence and queue filenames. It does not read R48/R52 result content or change scientific files/processes.

Actual timestamp is2026-11-08T21:06:23.838264+09:00;2105 is request identity/earliest time, not the actual observation minute. Old chain1700979 is absent; expected queue1739838/start467041669 is own/live. Direct child2393682/start488014488 is the observed own child, not a separately pre-pinned historical identity. Its executable is bash, so no particular scientific stage or progress quantity is inferred from it.

Against the earlier2031 snapshot, exactly R-48-1.json, R-48-2.json, R-48-3.json and R-48-optimizer.json newly appear. R-48-4.json, R-52-2.json and terminal.json are absent. This does not meet the retained terminal-collector trigger and supplies no job-content success/failure, optimizer selection, numerical or scientific verdict. Outcome explicitly leaves both collection qualifications and scientific_acceptance false. The retained FULL2 tail is an observation of already-collected work, not a new measurement or another terminal collection. README appropriately says neither R48 nor R52 collector was invoked; this is consistent with the bounded source/action record and does not generalize to unobserved activity.

CI manifest covers12 entries and observation manifest9, each excluding itself and including README. All hashes and coverage match. All11 CI original/diagnosis records and8 observation originals match ledger bytes. No source, workflow, raw scientific JSON, README or manifest was edited during this review; only this report was written. No Git, SSH, build, tests, scientific/native execution or subagent was used.

## Independent read-only check

Command, exit0:
```sh
python3 -B - <<'PY'
from pathlib import Path
import json,hashlib,re,datetime
H=lambda b:hashlib.sha256(b).hexdigest()
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');A=Path('results/validation/ci-deny-download-20261108');Q=Path('results/quadruped/raw/R-41-FULL2-OBS-20261108-2105')
for directory in [A,Q]:
 names=[]
 for line in (directory/'SHA256SUMS').read_text().splitlines():
  h,name=line.split(None,1);name=name.lstrip('*');assert H((directory/name).read_bytes())==h,name;names.append(name)
 assert set(names)=={p.name for p in directory.iterdir() if p.is_file() and p.name!='SHA256SUMS'}
 copied=0
 for p in directory.iterdir():
  if p.name in ['README.md','SHA256SUMS']:continue
  assert p.read_bytes()==(L/p.name).read_bytes(),p;copied+=1
 print('MANIFEST+LEDGER COPY PASS',directory,'manifest',len(names),'copied',copied)
d=json.loads((A/'ci-deny-download-diagnosis-20261108.json').read_text());assert len({json.dumps(x,sort_keys=True) for x in d['configuration_objects'].values()})==1
for p,h in d['configuration_objects']['6910b14'].items():
 b=Path(p).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==h,p
print('All3 recorded configuration-object maps equal; current4file blob hashes equal record (no historical Git lookup)')
for source,count,retries in [('6910b14',3,2),('40211e5',2,1)]:
 p=next(A.glob(f'ci-deny-{source}-*api.stdout'));s=re.sub(r'\x1b\[[0-9;]*m','',p.read_text());assert s.count('##[command]/usr/bin/docker build')==count;assert s.count('before retry.')==retries;assert 'gzip: invalid magic' in s;assert d['resolved_action_sha'] in s;assert 'cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz' in s
 audit='bans ok, licenses ok' in s;assert audit==(source=='40211e5');assert ('##[group]Run EmbarkStudios/cargo-deny-action@v2' in s)==audit;print(source,'Docker attempts',count,'automatic retry warnings',retries,'audit reached/passed',audit)
for p in sorted(A.glob('ci-observation*.json')):
 x=json.loads(p.read_text());jobs=[(j['name'],j['status'],j['conclusion']) for j in x['jobs']];print(p.name,x['headSha'],x['status'],x['conclusion'],jobs)
 if '42477151156' in p.name:assert x['conclusion']=='failure' and sum(j['conclusion']=='success' for j in x['jobs'])==5 and [j['name'] for j in x['jobs'] if j['conclusion']=='failure']==['deny']
 if '42477797335' in p.name:assert x['status']=='in_progress' and sum(j['conclusion']=='success' for j in x['jobs'])==5 and [j['name'] for j in x['jobs'] if j['status']=='in_progress']==['test']
prefix='post-full-observation-20261108-2105';b=(Q/(prefix+'.stdout')).read_bytes();o=json.loads(b);meta=json.loads((Q/(prefix+'.meta.json')).read_text());cmd=json.loads((Q/(prefix+'-command.json')).read_text());outcome=json.loads((Q/(prefix+'-root-outcome.json')).read_text());assert len(b)==meta['stdout_bytes']==2401 and H(b)==meta['stdout_sha256']==outcome['raw_stdout_sha256'];assert meta['returncode']==0 and meta['automatic_retry'] is False;assert not (Q/(prefix+'.stderr')).read_bytes();assert meta['command']==cmd['command'];assert H((Q/(prefix+'-command.json')).read_bytes())==meta['command_manifest_sha256'];assert H((Q/'post-full-observation-20261107-0600.remote.py').read_bytes())==meta['source_sha256']==cmd['source_sha256']
prev=json.loads(Path('results/quadruped/raw/R-41-FULL2-OBS-20261108-2031/post-full-observation-20261108-2031.stdout').read_bytes());new=sorted(set(o['queue_entries'])-set(prev['queue_entries']));assert new==outcome['new_records_since_2031'];assert not {'R-48-4.json','R-52-2.json','terminal.json'}&set(o['queue_entries']);assert all(outcome[x] is False for x in ['r48_collection_qualified','r52_collection_qualified','scientific_acceptance']);assert o['processes'][0]['exists'] is False and o['processes'][1]['matching_own_live_process'] is True
print('OBSERVER identity/command/source/stdout hashes pass; new entries',new);print('observed UTC',datetime.datetime.fromtimestamp(o['observed_unix'],datetime.timezone.utc).isoformat());print('No R48-4/R52-2/terminal; no numerical acceptance')
for p in sorted(A.iterdir())+sorted(Q.iterdir()):
 b=p.read_bytes();print('PIN',len(b),H(b),p)
PY
```

Raw output with exact reviewed pins:
```text
MANIFEST+LEDGER COPY PASS results/validation/ci-deny-download-20261108 manifest 12 copied 11
MANIFEST+LEDGER COPY PASS results/quadruped/raw/R-41-FULL2-OBS-20261108-2105 manifest 9 copied 8
All3 recorded configuration-object maps equal; current4file blob hashes equal record (no historical Git lookup)
6910b14 Docker attempts 3 automatic retry warnings 2 audit reached/passed False
40211e5 Docker attempts 2 automatic retry warnings 1 audit reached/passed True
ci-observation-42475544891-completed.json 2f7b8bd5d97973823219bf12e90eaafbd8027c10 completed success [('clippy', 'completed', 'success'), ('deny', 'completed', 'success'), ('research-tools', 'completed', 'success'), ('test', 'completed', 'success'), ('cuda-check', 'completed', 'success'), ('fmt', 'completed', 'success')]
ci-observation-42476002092-120202.json b7c51dd3f03f416637e1379c0fc35bbd806c3b99 completed success [('fmt', 'completed', 'success'), ('cuda-check', 'completed', 'success'), ('deny', 'completed', 'success'), ('test', 'completed', 'success'), ('research-tools', 'completed', 'success'), ('clippy', 'completed', 'success')]
ci-observation-42477151156-completed.json 6910b149172b67b07e69b92132cb05152df8ba6e completed failure [('test', 'completed', 'success'), ('clippy', 'completed', 'success'), ('research-tools', 'completed', 'success'), ('cuda-check', 'completed', 'success'), ('fmt', 'completed', 'success'), ('deny', 'completed', 'failure')]
ci-observation-42477797335-120201.json 40211e564c2ef6e2a94f19527710ef99ec54196d in_progress  [('test', 'in_progress', ''), ('research-tools', 'completed', 'success'), ('deny', 'completed', 'success'), ('fmt', 'completed', 'success'), ('cuda-check', 'completed', 'success'), ('clippy', 'completed', 'success')]
OBSERVER identity/command/source/stdout hashes pass; new entries ['R-48-1.json', 'R-48-2.json', 'R-48-3.json', 'R-48-optimizer.json']
observed UTC 2026-11-08T12:06:23.838264+00:00
No R48-4/R52-2/terminal; no numerical acceptance
PIN 2388 d3eded419bd4d5e422f8962f7a4fc790163235b358a55192c834ef50640d67f5 results/validation/ci-deny-download-20261108/README.md
PIN 1269 ff691d69b23f7bf73053b7710a63bebf59069d78b2a63cb15c07b3586d695ca6 results/validation/ci-deny-download-20261108/SHA256SUMS
PIN 173 a96bd55b10b601c859cba45ee6312685be34afab5c8b6e13015ce2c6e78fd2ef results/validation/ci-deny-download-20261108/ci-deny-40211e5-job128722911533-api.meta.json
PIN 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 results/validation/ci-deny-download-20261108/ci-deny-40211e5-job128722911533-api.stderr
PIN 37270 dd21cb8bd1473f16b5085149b6d102139216ed310bf5553e4eaeee7d78e5b7c5 results/validation/ci-deny-download-20261108/ci-deny-40211e5-job128722911533-api.stdout
PIN 173 34306007ad55cd1cc3a575e659d85d4758f7f98e7eef82b5b3eae172ae102145 results/validation/ci-deny-download-20261108/ci-deny-6910b14-job128720918474-api.meta.json
PIN 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 results/validation/ci-deny-download-20261108/ci-deny-6910b14-job128720918474-api.stderr
PIN 28959 1ed8314e6f6630aa1af8e003c1a4c4fc815b44def136fc35f5d32303eab52fae results/validation/ci-deny-download-20261108/ci-deny-6910b14-job128720918474-api.stdout
PIN 2601 1b138e117e9ecf30313dc169a226b3a6c64fc819fc73b03e5145fe926798490c results/validation/ci-deny-download-20261108/ci-deny-download-diagnosis-20261108.json
PIN 9311 0d71e181f97c43a698f80bb4273ef4d721eb11476dab8758f8ffcce450f4a9c0 results/validation/ci-deny-download-20261108/ci-observation-42475544891-completed.json
PIN 9311 66ace560cbd049517dfabc818ca9a5d726c3db504d31023c0e4a2f0ec7307aaf results/validation/ci-deny-download-20261108/ci-observation-42476002092-120202.json
PIN 9144 9e0691ae1b9bb978e21491d2e45a885f77bdc7742c9ac8eea4204114ad6ee02f results/validation/ci-deny-download-20261108/ci-observation-42477151156-completed.json
PIN 9135 f76741c71e0237bb8c24a4928a5fb1df5ab44e39f15aa68d1ed1262406ca5ef9 results/validation/ci-deny-download-20261108/ci-observation-42477797335-120201.json
PIN 1073 35025b1012ce1aa9dad87cbfcce427557b726e4c7a6e0a5d2c16303ef2774919 results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/README.md
PIN 983 78640cd3166f7619d4363f3fdf23559bcb03e4265362d2ba05f6c5757cec19ac results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/SHA256SUMS
PIN 1560 17550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243 results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261107-0600.remote.py
PIN 813 3ea8d73747aaa436f06ef1649de1d21d6ae78070f467a0fca7fa49c81377bfec results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105-command.json
PIN 1547 f07dc66eb0a1d979916425ab7f7944ed9c70902ac047bb9e33c7a17e51a9922f results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105-controller.py
PIN 578 96c27c7128be16e987d06536e0d57759d9eedbff6d64c7b1fdc1ddb47203237b results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105-root-outcome.json
PIN 901 cbb748208d42d4cefb4839da302918927de44fcf03b0a82917714c56b8bed397 results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105.launch.json
PIN 1203 315209f8b8afb30858abc5adba012bb4e21ad4049216fbdff9d9c25aca82e0df results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105.meta.json
PIN 0 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105.stderr
PIN 2401 43205f41e1377604bd7e0d1747c38bbe644723215c88185df891bb6b9c633d64 results/quadruped/raw/R-41-FULL2-OBS-20261108-2105/post-full-observation-20261108-2105.stdout

```

