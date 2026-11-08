# Task42 final report/archive/tables — independent review

2026-11-08. **SPEC: PASS. QUALITY: PASS.** One Plan B wording issue was raised and corrected by the root during review; no unresolved finding remains in the final reviewed scope. This approves the final documentation's bounded evidence/interpretation, not unconditional RQ1 acceptance, new measurements, or a paper PDF build.

Final M3:67,250 B, SHA256 `e80dc9fa40eaf0b12f8bfb47d3b9fed25f96dfdd71405444937d771e0dee6aed`. Root-supplied final diff:61,873 B, SHA256 `44d8de6d2cd89b9d871beb20bdf6dbc87a60ccec473c23f74eb5103056c5509b`. I read this supplied diff and reversed its hunks in memory; no Git command was used. Every new-side hunk matched the actual eight changed tracked files. Reconstructed pre-edit M3 SHA256 is `3f8b551813d2889aa78a536153694a917f2c3a9956dbb58c79d1d14a4e00a31a`, the snapshot used by the preceding scientific review.

## Findings resolved and scope accepted

**Plan B — closed.** The first reviewed text at M3 §0/§9.1 said the old selection A continued, although A deferred judgment only until M3. With this report now making that judgment, it could imply a newly approved indefinite deferral. Research plan §4.3.2 explicitly links failure to a change in paper emphasis; Task42 brief Step2 returns judgment to the user. The root removed the continuation wording, preserved A as a historical decision, stated that the deferred M3 judgment has now occurred, and added §9.19. The final text reports RQ1 as conditional model comparison/re-evaluation and leaves treatment of the full Plan B emphasis to the user. It does not turn unmeasured GPU training into a result or block authorized document/publication work. No automatic new scientific waiver or interpretation of the old decision as permanent nonactivation remains.

**Numbers and gates — PASS.** The final M3 and three changed MD/TeX table pairs agree with the independently audited original FULL-2 result:196 inputs, B48 in64 lanes,1000 updates/48000 samples,1391540 KS,4000 boots,28.990416667 KS/sample, training146229.853279576s. ReBoot nonboot37 and bootstrap502 remain model values; ours111 remains measured. Ratios(a,b,c,d) are0.367743129/0.781062368/0.626819820/0.533524775. The table's `acd` pass set excludes(b). Accuracy0.9035 versus seed0 reference0.9041 is−0.06pp, PASS; B64 remains “not the gate”. Full-setting ReBoot timing is not measured and same-security is unmet. Ten-pass arithmetic mean, RNG32 versus initial seed0, n=1, and the accumulated parameter-error diagnostic are accurately distinguished. No mixed-B all-conditions pass appears.

**Historical decisions — PASS.** M3 §8.1–8.2 are exactly unchanged (13,224 B, SHA256 `dee5f3aaad29b620bdd89d75dcc633eb6bd2f9617ffa0059daf7ae0cb056471b`). §8.3 changes only the completed FULL status, retaining deviations. All existing §9 items1–18 remain; scientific choices such as B, security matching, timing comparison, same N and parameter-error tolerance are not silently resolved. The renamed library and existing sole-author/publication choices are labeled as separately settled. §9.19 records the Plan B distinction above.

**Archive and links — PASS.** Raw SHA256SUMS contains exactly12 original server-byte files, and all12 equal the base64 parent capture. The19,259-byte original JSON remains SHA256 `921d5a7c315497ccc2f4d309b19edd6f5718013abeb38f0b63eb06da00f93a62`. Archive SHA256SUMS covers all13 other archive files, including its README. All9 task42 source/controller/fetch/review copies equal their L originals; no compression/re-encoding loss or changed scientific_acceptance flag. The archived5,392-byte derived summary is exactly the preceding independent aggregation stdout (SHA256 `2304b2becc437f0f16f71314ef4c36ee9e7436ddaf863f2f4dbca8168667b749`), with command exit0 and empty stderr. Its generic pending/unverified text is properly qualified by the archive README, rather than overriding known completion/failure. All actual relative links in M3 and both new READMEs resolve.

**Generated files — PASS.** Changes are confined to accuracy, RQ1 KS, wall-clock MD/TeX pairs and their sources.tsv records. The new TSV dependencies point to FULL-2 actual counts/timing/accuracy, seed0 reference, and the existing labeled model constants; they do not use the raw legacy lazy_measured headline for196. Read-only `python3 -B tools/paper_tables.py --check` returned0 with empty stdout/stderr. No table generation was run by this reviewer. The other four MD/TeX pairs have no changes in the supplied diff. No PDF layout/render claim is made.

## Actual verification and limits

Python standard-library read-only checks validated both hash manifests, capture equality, ledger/archive equality, relative-link existence and final file pins; exit0. In-memory reverse-diff verification validated all hunks, the previous M3 identity, unchanged §8.1–8.2 and retained §9 numbering; exit0. Archived summary equality to prior independent stdout was true. The initial lightweight link scanner mistook a bracketed modulus list followed by parenthesized prose for a link and stopped; restricting the check to actual no-whitespace local destinations resolved that scanner false positive. No target was edited.

The preceding scientific review's executed-binary/all-input provenance limits remain explicit in the final documents. This review does not revalidate the old experiment matrix, rerun tests/training, or inspect unrelated active Rust source. The root subsequently added a bounded handover/M5/requests/observation review, recorded below. No source/docs/results writes, SSH, native/HE/build, Git operation or subagent; only this report was written.

Root's final M3 amendment changed no table bytes, so the successful check-only table result remains applicable. Reviewed plan SHA256: `2872931846f37465192e76b6e5daf8f23625ed225b480b44853de0762eefb5a9` (../plans/2026-11-04-he-training-research-plan.md).

### Raw check results and final pins

```text
DIFF_PIN 61873 44d8de6d2cd89b9d871beb20bdf6dbc87a60ccec473c23f74eb5103056c5509b
DIFF_NEW_MATCH docs/m3-report.md
M3_OLD_PIN 63600 3f8b551813d2889aa78a536153694a917f2c3a9956dbb58c79d1d14a4e00a31a
HISTORICAL_8.1_8.2_BYTE_EQUAL 13224 dee5f3aaad29b620bdd89d75dcc633eb6bd2f9617ffa0059daf7ae0cb056471b
SECTION9_ITEMS1_TO18_PRESERVED;19_ADDED
DIFF_NEW_MATCH paper/tables/accuracy_vs_reboot.tex
DIFF_NEW_MATCH paper/tables/rq1_ks.tex
DIFF_NEW_MATCH paper/tables/wallclock_same_cpu.tex
DIFF_NEW_MATCH results/tables/accuracy_vs_reboot.md
DIFF_NEW_MATCH results/tables/rq1_ks.md
DIFF_NEW_MATCH results/tables/sources.tsv
DIFF_NEW_MATCH results/tables/wallclock_same_cpu.md
NO_FILES_WRITTEN

MANIFEST_PASS results/quadruped/raw/R-41-FULL-2 12 entries
MANIFEST_PASS results/validation/task42-full2-finalization-20261108 13 entries
CAPTURE_RAW_12_EQUAL
LEDGER_ARCHIVE_TASK42_COPIES_EQUAL 9
RELATIVE_LINKS_PASS docs/m3-report.md
RELATIVE_LINKS_PASS results/quadruped/raw/R-41-FULL-2/README.md
RELATIVE_LINKS_PASS results/validation/task42-full2-finalization-20261108/README.md
PIN docs/m3-report.md 67250 e80dc9fa40eaf0b12f8bfb47d3b9fed25f96dfdd71405444937d771e0dee6aed
PIN results/quadruped/raw/R-41-FULL-2/README.md 2443 48eba3f307970208a34c5e17a8f0a15be9dc9d3140930ba361a81be1030f781c
PIN results/quadruped/raw/R-41-FULL-2/SHA256SUMS 949 0e642eafeff9b0b364951a62ff3347e8abdbd4ffa6f805ac94cfc6d5c046b5b0
PIN results/validation/task42-full2-finalization-20261108/README.md 1467 c99e1cababa0cd7f2c546fd2ba125a5c4d4299411ff731afe9b927cb01c0545a
PIN results/validation/task42-full2-finalization-20261108/SHA256SUMS 1320 b5be8c6c5cbc17d1919dfc2b636a1689f1ea623b340c60411ee5be920fd5f321
PIN results/tables/accuracy_vs_reboot.md 3403 2070c0ec36e1ef2df710d6b339b0fc5a4a016d1237fa727e0a49d7d87421c238
PIN results/tables/gpu_vs_cpu_ops.md 2822 af5f19dd3f4cc993b43643a49c3bcbc0600854ddee638adfa402f4fffd6ff242
PIN results/tables/plain_datasets.md 1279 0fa907d7b9d3d382e9ad7a169f4c68309f13be01e295e45b020dfb8125e6bd65
PIN results/tables/rq1_ks.md 2916 a21ab0ae2633e32393042c85e3a5cec0245dd80414c087226b89b95afac17465
PIN results/tables/rq3_ablation.md 4260 3f276671badfb198dc54a725537dad27d256cc26f9fcbd015cc34b69f3d7fac8
PIN results/tables/security.md 1050 b7ba5e0bd84756808ac3581a3a39f001e4aedf919bead0da1c35c6f2608d473e
PIN results/tables/sources.tsv 162402 878fbcbf90a897194dd3bc3af75f294ccd218dac68dff3a2f03ff39e378743bc
PIN results/tables/wallclock_same_cpu.md 1844 f2972f81115f29e71f5a0f0d5ee652274c4832b36613fd2b3ccd35c0716d198c
PIN paper/tables/accuracy_vs_reboot.tex 3718 08eafc7dec34a0bb791a77789c02369fe61c5a54a6aad03c1651c50ea3a33270
PIN paper/tables/gpu_vs_cpu_ops.tex 3051 c9104367559251ee43275272c219c2e632b59225c1d56eda7170422afe3043cf
PIN paper/tables/plain_datasets.tex 1535 1667a75790194a9dc0fef7b36af3fc6cc5aeda22ac87b395ef19d82d9db7a6fe
PIN paper/tables/rq1_ks.tex 3124 a8d57e0199cf1c6927f51ed946baed0ed98a5834031bed690ac5d1b99b9a8e9d
PIN paper/tables/rq3_ablation.tex 4488 de38d3727aecf170b18a3ce0ed4db28f5984bf6938934081236e53d7f6db7cba
PIN paper/tables/security.tex 1361 3b30d20da49c757d274b6b30501bb524ab448150301d0f153b3e2d60d31c019e
PIN paper/tables/wallclock_same_cpu.tex 2092 292f2cdb74aa92101a8f68974c7fc11010e3e6581f78080fe455461aff8fbb32
NO_WRITES
```

Manifest and diff verification used these exact read-only commands; no temporary or generated artifact was written:

```sh
python3 -B - <<'PY'
from pathlib import Path
import base64,hashlib,json,re
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');A=Path('results/validation/task42-full2-finalization-20261108');D=Path('results/quadruped/raw/R-41-FULL-2')
for d in [D,A]:
 entries={}
 for line in (d/'SHA256SUMS').read_text().splitlines():
  h,n=line.split('  ',1);b=(d/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h,n;entries[n]=h
 expected={p.name for p in d.iterdir() if p.is_file()}-({'SHA256SUMS','README.md'} if d==D else {'SHA256SUMS'})
 assert set(entries)==expected,(d,set(entries)^expected)
 print('MANIFEST_PASS',d,len(entries),'entries')
c=json.loads((A/'task-42-full2-terminal-fetch-v1.stdout').read_bytes())
for n,x in c['files'].items():assert base64.b64decode(x['base64'],validate=True)==(D/n).read_bytes(),n
print('CAPTURE_RAW_12_EQUAL')
for p in A.glob('task-42-*'):
 assert p.read_bytes()==(L/p.name).read_bytes(),p.name
print('LEDGER_ARCHIVE_TASK42_COPIES_EQUAL',len(list(A.glob('task-42-*'))))
for p in [Path('docs/m3-report.md'),D/'README.md',A/'README.md']:
 for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' not in ref and not ref.startswith('#') and not any(c.isspace() for c in ref):
   target=p.parent/ref.split('#',1)[0];assert target.exists(),(p,ref)
 print('RELATIVE_LINKS_PASS',p)
for p in [Path('docs/m3-report.md'),D/'README.md',D/'SHA256SUMS',A/'README.md',A/'SHA256SUMS']+sorted(Path('results/tables').glob('*'))+sorted(Path('paper/tables').glob('*')):
 b=p.read_bytes();print('PIN',p,len(b),hashlib.sha256(b).hexdigest())
print('NO_WRITES')
PY
```

```sh
python3 -B - <<'PY'
from pathlib import Path
import re,hashlib
p=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-42-final-report.diff');b=p.read_bytes();text=b.decode();print('DIFF_PIN',len(b),hashlib.sha256(b).hexdigest())
for part in text.split('diff --git ')[1:]:
 lines=part.splitlines(keepends=True);path=lines[0].split(' b/',1)[1].strip();cur=Path(path).read_text().splitlines(keepends=True);old=[];pos=0;i=0
 while i<len(lines):
  m=re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',lines[i])
  if not m:i+=1;continue
  start=int(m.group(3))-1;old.extend(cur[pos:start]);pos=start;i+=1
  while i<len(lines) and not lines[i].startswith('@@ '):
   z=lines[i];i+=1
   if z.startswith('\\'):continue
   if z[0] in ' +':assert cur[pos]==z[1:],(path,pos+1);pos+=1
   if z[0] in ' -':old.append(z[1:])
 old.extend(cur[pos:]);old=''.join(old);new=''.join(cur);print('DIFF_NEW_MATCH',path)
 if path=='docs/m3-report.md':
  print('M3_OLD_PIN',len(old.encode()),hashlib.sha256(old.encode()).hexdigest())
  for label,end in [('### 8.1','### 8.3')]:
   a=old[old.index(label):old.index(end)];z=new[new.index(label):new.index(end)];assert a==z;print('HISTORICAL_8.1_8.2_BYTE_EQUAL',len(a.encode()),hashlib.sha256(a.encode()).hexdigest())
  assert re.findall(r'^\d+\.',old[old.index('## 9.'):],re.M)==re.findall(r'^\d+\.',new[new.index('## 9.'):],re.M)[:18];print('SECTION9_ITEMS1_TO18_PRESERVED;19_ADDED')
print('NO_FILES_WRITTEN')
PY
```

## Added bounded status/20:05 observation review — PASS

The root-supplied `task-42-status-records.diff` was fully read in bounded sections and all new-side hunks match the three current files. The handover/M5/requests updates accurately replace the old FULL-2 pending state with the audited terminal/count/accuracy result. They preserve RQ1 failure/unmeasured distinctions, incomplete Task54/native work, and separate fixed-preview validation. The new handover anchor exists; all42 added local destinations/custom latest-evidence anchors resolve.

The20:05 request is distinguished from actual observation20:06:14.668975 JST. The2005 archive's9 manifest entries cover its README and8 exact ledger artifacts; all hashes and ledger bytes match. Its observation stdout is exactly the triggering observation copied into the FULL-2 capture record. The old chain is absent, expected queue identity is live, and R47 test/example/nsys plus R47/R48 build filenames are present. Documents expressly avoid promoting those names to scientific acceptance or R48/R52 terminal success. This review does not independently adopt any R47/R48/R52 measurement.

The status documents' “final documentation review/commit in progress” language is accurate for this review snapshot; completion of this review does not retroactively change their recorded state. The pending root commit itself is outside this review. No additional blocker was found.

Read-only supplemental checks exited0:

```text
DIFF_PIN 25179 dd02422ebdd26691d934bb5f9d23ef45661ad0cc5214052436636ffc0aaebf04
DIFF_NEW_MATCH docs/handover.md
DIFF_NEW_MATCH docs/m5-report.md
DIFF_NEW_MATCH results/quadruped/requests.md
NO_FILES_WRITTEN

ADDED_STATUS_LINKS_PASS 42
OBS_MANIFEST_PASS 9 entries; covered ['README.md', 'post-full-observation-20261107-0600.remote.py', 'post-full-observation-20261108-2005-command.json', 'post-full-observation-20261108-2005-controller.py', 'post-full-observation-20261108-2005-root-outcome.json', 'post-full-observation-20261108-2005.launch.json', 'post-full-observation-20261108-2005.meta.json', 'post-full-observation-20261108-2005.stderr', 'post-full-observation-20261108-2005.stdout']
OBS_LEDGER_COPIES_EQUAL
OBS_TRIGGER_CAPTURE_EQUAL 1794135974.6689746 ['R-47-3-example.json', 'R-47-3-nsys.json', 'R-47-3-tests.json', 'gate-released.json', 'owner.json', 'post-full-r47-build.json', 'post-full-r48-build.json']
PIN docs/handover.md 216720 e69d4424e90d93f62c93966c31674b39beba97e0bd72c7d9b8a37f7f9bafbb7f
PIN docs/m5-report.md 179658 8a7c25c61f10e0c4efe9083a81b62ba2efd7254e502441ca409c1ea6f900e608
PIN results/quadruped/requests.md 164069 44f39fdeea74bfb9d37eb9ff7a385859e94035d4ba39b5dc988f965a66e2e989
PIN results/quadruped/raw/R-41-FULL2-OBS-20261108-2005/README.md 1244 2c9faac62c2a8bcacfda23cf8b2b2192a8040a49ac59b5f7b202188601bd4b1a
PIN results/quadruped/raw/R-41-FULL2-OBS-20261108-2005/SHA256SUMS 983 44632c6bfc2999c69d71774b5f7a44749eb012f5ce8632cd8643bd89abf2fde1
PIN .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-42-status-records.diff 25179 dd02422ebdd26691d934bb5f9d23ef45661ad0cc5214052436636ffc0aaebf04
NO_FILES_WRITTEN
```


