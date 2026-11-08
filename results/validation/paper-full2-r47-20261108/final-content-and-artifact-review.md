# Task56 FULL2 / R47 paper adoption review

Date: 2026-11-08. Scope: parent-provided `task56-full2-r47-paper.diff`, covering paper/README.md, main.tex and introduction/gpu/evaluation/discussion sections. The generated tables are consumed unchanged.

SPEC: **PASS for the manuscript's adoption of already-reviewed FULL2 and R47 evidence.**
QUALITY: **PASS for the same content scope.** No actionable finding or scientific overclaim identified.

The abstract, introduction and conclusion consistently present a conditional comparison. FULL2's registered baseline-batch accuracy pass is separated from its failed convention(b) count target. FULL1's all-four count pass at B64 is not combined with FULL2's accuracy pass at B48 into a synthetic all-conditions result. Overall RQ1 is explicitly unmet; the common-security condition remains unmet and full-setting ReBoot CPU time remains unmeasured.

Evaluation identifies196 inputs,4800 examples over ten passes, B48 in64 physical lanes versus B64, and initial-weight seed0 distinct from the encryption RNG seed. The command's one epoch is correctly distinguished from one conventional pass. Its accuracy reference is the individual released-code seed0 result, not the ten-seed mean or published paper setting. Adopted tables give90.35% versus90.41%, delta−0.06 percentage points within±0.5; FULL2 counts yield(a/c/d)PASS and(b)0.781062>0.70. Baseline non-bootstrap37 and bootstrap502 remain models, versus measured ours111 per bootstrap; the unchanged surrounding explanation and table notes retain that boundary.

The final parameter differences are retained as trajectory diagnostics against respective same-batch twins, not standalone bootstrap precision, a new tolerance, or a multi-seed guarantee. The full-setting runs do not substitute for previously distinct784-input subsets or deeper encrypted-model results.

R47 claims are limited to four independent capped-level5 products, three warmups and30 synchronized within-run repetitions per condition. The six adopted medians support within-pool improvement, with the smaller median's stream count depending on pool threshold. The manuscript does not attribute the gain to individual changes, infer statistical significance, or call it a training-step speedup. Sequential training, shared-device caveats, missing peak-memory measurements and unfinished GPU bootstrap evidence remain. The separate profiled5-repetition run is excluded from timing rows; its trace remains unanalyzed. Test accounting explicitly preserves45 harness successes including3 memory SKIPs,42 executed functions including expected-error paths.

R48/RQ3 server rows and R52 encrypted subset rows remain pending. No subsequent queue artifact is accepted by this paper edit. Security discussion retains estimator-model and attack-memory caveats rather than claiming a practical break or proof. No citation key was added or removed in the reviewed files; no new external-source verification was needed for this bounded adoption review.

The README distinguishes fixed public6fba46c FULL1 preview assets from newer local source. Earlier dated build records remain historical and do not validate this updated source. The12:31 dated paragraph's former FULL2-pending state is historical, not the current status. A new build/visual verification record is a separate root activity and is not independently established by this content verdict.

## Verification and boundaries

Read the complete supplied diff, current changed sections and relevant generated RQ1/accuracy/GPU tables; checked for stale current FULL2-pending or repaired-R47-pending claims. Compared against the already-completed independent FULL2 numeric/final-document and R47 terminal/document reviews, whose exact hashes are below. No raw scientific data was changed or reinterpreted, and no numerical/native experiment was rerun.

The read-only Python probe below verifies that every supplied diff hunk matches the current file and that citation sets are unchanged; it records all seven table hashes. Exit0. No Git invocation, build, SSH, tests, source edit or table generation was performed. Only this ledger report is written. PDF layout and arXiv server acceptance are outside this content check.

```sh
python3 -B - <<'PY'
from pathlib import Path
import re,hashlib
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');d=L/'task56-full2-r47-paper.diff';files=[]
for sec in re.split(r'(?=^diff --git )',d.read_text(),flags=re.M):
 if not sec:continue
 path=re.search(r'^\+\+\+ b/(.+)$',sec,re.M)[1];files.append(path);lines=Path(path).read_text().splitlines(keepends=True);old=list(lines)
 chunks=re.split(r'(?=^@@ )',sec,flags=re.M)[1:]
 for chunk in reversed(chunks):
  header,*body=chunk.splitlines(keepends=True);m=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',header);i=int(m[1])-1;n=int(m[2] or 1);new=[x[1:] for x in body if x.startswith((' ','+'))];before=[x[1:] for x in body if x.startswith((' ','-'))];assert lines[i:i+n]==new,(path,header);old[i:i+n]=before
 new=''.join(lines);old=''.join(old)
 citations=lambda x:set(re.findall(r'\\cite\w*\{([^}]+)\}',x))
 assert citations(new)==citations(old),path
 print('CURRENT HUNKS MATCH; citation set unchanged',path,len(chunks))
assert set(files)=={'paper/README.md','paper/main.tex','paper/sections/introduction.tex','paper/sections/gpu.tex','paper/sections/evaluation.tex','paper/sections/discussion.tex'}
for p in [d]+[Path(x) for x in files]+[Path('docs/m3-report.md'),Path('docs/m4-report.md'),L/'task-42-full2-terminal-scientific-review.md',L/'task-42-final-report-review.md',L/'task-47-terminal-scientific-review.md']+sorted(Path('paper/tables').glob('*.tex')):
 b=p.read_bytes();print('PIN',len(b),hashlib.sha256(b).hexdigest(),p)
print('No source changes, tests, builds, Git, or table generation performed')
PY
```

Raw output:
```text
CURRENT HUNKS MATCH; citation set unchanged paper/README.md 2
CURRENT HUNKS MATCH; citation set unchanged paper/main.tex 1
CURRENT HUNKS MATCH; citation set unchanged paper/sections/discussion.tex 2
CURRENT HUNKS MATCH; citation set unchanged paper/sections/evaluation.tex 3
CURRENT HUNKS MATCH; citation set unchanged paper/sections/gpu.tex 1
CURRENT HUNKS MATCH; citation set unchanged paper/sections/introduction.tex 1
PIN 36299 021af5fa0cd896a0c2910ea5ea322315597f40d53fe4457b0bbf0c5d40efd4a5 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task56-full2-r47-paper.diff
PIN 13956 190d915a3c34a0585eb3d4d5b058881354a2646d7d4e1bbcc090b6a6e04ee304 paper/README.md
PIN 3934 924d994b1f863a8fb2316e22b0eb263a111695800a66088667e043c72dd3e367 paper/main.tex
PIN 7444 e2a419ae4bde6776e70ac70c136fa021e5fa2798121e8aa5191c133b67babeb3 paper/sections/discussion.tex
PIN 12025 20d9030013780b736a3db31cbbe344dfce8b525e2289628ca5e1e6aef192f2d3 paper/sections/evaluation.tex
PIN 3232 82f7be2adcc0c45362e5d463597721bdfeaf92e4062cef40da0e543b651f9006 paper/sections/gpu.tex
PIN 3306 f233d8b3dd745f98c79e989d2833c499e30dc4a5cb0c148b52128fd113d2acdb paper/sections/introduction.tex
PIN 67250 e80dc9fa40eaf0b12f8bfb47d3b9fed25f96dfdd71405444937d771e0dee6aed docs/m3-report.md
PIN 43602 046cf855b3900b5b6672fefb96a6a192b0187a3c3cd4daca013de879f5560f4d docs/m4-report.md
PIN 16773 03b8bd650fc94ebcb137ae219f24379382958f4cafa437694e1bc4fb5ce624ea .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-42-full2-terminal-scientific-review.md
PIN 15611 61af732e725fdb273fef038a96c2c4caabd93914a0e97a8c5796829a96e8d490 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-42-final-report-review.md
PIN 25865 3a8467a3416c7dde6b279676b126559c1ddc23a5378472fe23a8260ccd3deb7e .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-47-terminal-scientific-review.md
PIN 3718 08eafc7dec34a0bb791a77789c02369fe61c5a54a6aad03c1651c50ea3a33270 paper/tables/accuracy_vs_reboot.tex
PIN 5598 3baea336bd2cb9e2904c89e8aed89d9b2092eb1493df9a58e2f9237533e3ccb8 paper/tables/gpu_vs_cpu_ops.tex
PIN 1535 1667a75790194a9dc0fef7b36af3fc6cc5aeda22ac87b395ef19d82d9db7a6fe paper/tables/plain_datasets.tex
PIN 3124 a8d57e0199cf1c6927f51ed946baed0ed98a5834031bed690ac5d1b99b9a8e9d paper/tables/rq1_ks.tex
PIN 4488 de38d3727aecf170b18a3ce0ed4db28f5984bf6938934081236e53d7f6db7cba paper/tables/rq3_ablation.tex
PIN 1361 3b30d20da49c757d274b6b30501bb524ab448150301d0f153b3e2d60d31c019e paper/tables/security.tex
PIN 2092 292f2cdb74aa92101a8f68974c7fc11010e3e6581f78080fe455461aff8fbb32 paper/tables/wallclock_same_cpu.tex
No source changes, tests, builds, Git, or table generation performed

```


## Appendix shortening: current-source observation

Read `paper/sections/appendix.tex` in the follow-up review. Its current 5982 bytes have SHA256 `2860523513668d311e41774a58227d381c21893fae7cbfaa2975af02de4d1537`. The sole ArxivDraft-gated final notice says: “Code and evidence:” followed by the repository URL, then explicitly distinguishes release v1.0.0-preview.1 as an earlier snapshot from this preprint's source revision. This wording is consistent with the already-reviewed README and does not add a publication/submission claim. The preceding paragraph retains final-source/metadata/server-PDF checks and unavailable-measurement boundaries; all preceding scientific limitations remain present.

No issue in this current notice was identified. At this observation, the supplied paper.diff still lacked an appendix hunk, so exact old/new scope verification and the upcoming README/build-metadata appendix remain pending a later supplied diff. No PDF build or visual validation is claimed here.

## Final appendix, metadata, packaging and status follow-up

SPEC: **PASS**. QUALITY: **PASS**. The earlier pending appendix-diff and build-documentation checks are now closed. No remaining actionable finding.

Reviewed the complete later `task56-full2-r47-paper-final.diff` (59943B, SHA256 `123ef18391958ea64d5532b60dd372a57c88253e7d97e2531855d21a63760035`) against current files, and the15-file validation archive at `results/validation/paper-full2-r47-20261108/`. The previous scientific text remains byte-identical to the content-reviewed inputs. The appendix hunk changes only the one ArxivDraft-gated final repository notice, preserving the URL, earlier-release boundary and surrounding scientific/submission limitations. Comparing18 initial/final TeX input pins finds only appendix changed; tables, main, references, figure and other sections did not change. No font/margin source change is present.

README/arxiv-metadata/handover/M5 consistently label this as a newer **local** named preprint and package, not a public-release replacement, final evidence cutoff, arXiv server build or submission. The metadata abstract matches current main.tex verbatim and quotes its correct hash. The12:31 metadata paragraph now explicitly describes an earlier snapshot superseded by the present local package. Earlier published preview and historical evidence remain distinct. M5/handover retain R48/R52 and research incompleteness. The cited20:32 observation README explicitly does not infer training progress or scientific success. Added local Markdown links exist (9 checked). Git commit/push statements are parent-recorded operational facts; this review did not query Git or remote state.

Independently read the actual MicroSD PDF and ZIP bytes. Direct PDF532092B SHA256 `9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c`; ZIP40079B SHA256 `2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2`. All19 unique ZIP members match manifest and unpacked source. Eighteen source inputs match current files; main.tex adds only the ArxivDraft definition and main.bbl matches the direct generated bibliography. No raw data, logs, styles or PDF are in the19-member set. Direct/package metadata and recorded outputs show exit0 builds with shell escape disabled and no arXiv server action.

All10 saved direct/isolated page PNG pairs are byte-identical and match verification.json hashes. Initial/final saved page renders1–9 match, and page10 differs as recorded. The saved extracted texts also match exactly, including the recorded text hash. Both stored pdfinfo records say10 pages. Both archived final TeX logs match SD logs and contain no overfull/missing-character/undefined-reference/citation patterns; their acknowledged disabled-shell-escape and float-page warnings do not contradict the document claims. Ten archived build/meta/log files exactly match their original SD copies. The initial8581B content review copy is exact, preserving the original PASS before this follow-up.

Visual inspection itself was performed by root, not repeated by this reviewer. This review independently verifies saved render identity and consistency of its custody/inspection record; it does not claim to have regenerated or visually inspected the pages, rebuilt the package, or independently repeated source-to-PDF rendering. Likewise, it does not revalidate installed toolchain/runtime provenance or external publication status. No new scientific result is created by PDF verification.

The following read-only probe completed exit0. Its raw output records exact final document/archive pins; no Git, SSH, build, native experiment, table generation or source/document/result mutation was run by this reviewer. Only this report was updated.

```sh
python3 -B - <<'PY'
from pathlib import Path
import hashlib,json,re,zipfile
H=lambda b:hashlib.sha256(b).hexdigest()
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');A=Path('results/validation/paper-full2-r47-20261108');d=json.loads((A/'direct-build-meta.json').read_text());m=json.loads((A/'package-manifest.json').read_text());v=json.loads((A/'verification.json').read_text());initial=json.loads((A/'initial-build-meta.json').read_text());isol=json.loads((A/'isolated-build-result.json').read_text());out=Path(d['output_directory']);pack=Path(m['zip']['path']).parent
assert d['exit_code']==isol['exit_code']==0;assert d['paper_inputs_sha256']==m['source_inputs_sha256'];assert len(d['paper_inputs_sha256'])==18
for p,h in d['paper_inputs_sha256'].items():assert H(Path(p).read_bytes())==h,p
changes=[p for p,h in initial['paper_inputs_sha256'].items() if d['paper_inputs_sha256'][p]!=h];assert changes==['paper/sections/appendix.tex'];print('18 current build inputs match; only appendix input changed between initial/final')
for k in ['zip','direct_pdf']:
 x=m[k];b=Path(x['path']).read_bytes();assert len(b)==x['bytes'] and H(b)==x['sha256'];print(k,len(b),H(b))
with zipfile.ZipFile(m['zip']['path']) as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(m['members'])==19
 for entry in m['members']:
  name=entry['name'];b=z.read(name);assert len(b)==entry['bytes'] and H(b)==entry['sha256'];expected=(out/'arxiv-wrapper.bbl').read_bytes() if name=='main.bbl' else (b'\\def\\ArxivDraft{1}\n'+Path('paper/main.tex').read_bytes() if name=='main.tex' else (Path('paper')/name).read_bytes());assert b==expected,name;assert b==(pack/'source'/name).read_bytes(),name
print('19 unique ZIP members match manifest/current selected inputs/unpacked source; main has only ArxivDraft prefix; bbl matches direct generated bibliography')
assert H((pack/'build/main.pdf').read_bytes())==isol['pdf_sha256'];assert (pack/'build/main.txt').read_bytes()==(out/'arxiv-wrapper.txt').read_bytes();assert H((out/'arxiv-wrapper.txt').read_bytes())==isol['text_sha256']
for page in v['pages']:
 n=page['page'];b=(out/f'page-{n:02}.png').read_bytes();assert H(b)==page['direct_png_sha256'];assert b==(pack/'build'/f'page-{n:02}.png').read_bytes();old=(Path(initial['output_directory'])/f'page-{n:02}.png').read_bytes();assert (b==old)==page['previous_reviewed_render_identical']
assert [x['page'] for x in v['pages']]==list(range(1,11));print('10 saved direct/isolated PNG pairs exact; initial/final pages1-9 exact, page10 differs; saved extracted text exact')
for archived,actual in [('direct-build-meta.json',out/'build-meta.json'),('direct-build.stdout',out/'build.stdout'),('direct-build.stderr',out/'build.stderr'),('direct-tex.log',out/'arxiv-wrapper.log'),('initial-build-meta.json',Path(initial['output_directory'])/'build-meta.json'),('package-manifest.json',pack/'manifest.json'),('isolated-build-result.json',pack/'isolated-build-result.json'),('isolated-build.stdout',pack/'build.stdout'),('isolated-build.stderr',pack/'build.stderr'),('isolated-tex.log',pack/'build/main.log')]:assert (A/archived).read_bytes()==actual.read_bytes(),archived
for f in ['direct-tex.log','isolated-tex.log']:
 s=(A/f).read_text();assert not re.search(r'Overfull|Missing character:|undefined references|Citation .*undefined|Reference .*undefined',s,re.I);print(f,'no blocking warning patterns')
for f in ['direct-pdfinfo.txt','isolated-pdfinfo.txt']:assert re.search(r'^Pages:\s+10$',(A/f).read_text(),re.M)
main=Path('paper/main.tex').read_text();meta=Path('paper/arxiv-metadata.md').read_text();abstract=re.search(r'\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}',main,re.S)[1];assert abstract==meta.split('## Abstract\n\n',1)[1].split('\n\n## ',1)[0];assert H(Path('paper/main.tex').read_bytes()) in meta;print('metadata abstract exact and main hash current; stored pdfinfo10pages each')
assert H((A/'initial-content-review.md').read_bytes())=='4cdba2512d1b0e50643a02dafaa273c4e15620e8aac62f4a92d03aac0bc02d9b';print('initial review snapshot exact')
diff=L/'task56-full2-r47-paper-final.diff';paths=[]
for sec in re.split(r'(?=^diff --git )',diff.read_text(),flags=re.M):
 if not sec:continue
 p=Path(re.search(r'^\+\+\+ b/(.+)$',sec,re.M)[1]);paths.append(p);lines=p.read_text().splitlines(keepends=True)
 for chunk in re.split(r'(?=^@@ )',sec,flags=re.M)[1:]:
  header,*body=chunk.splitlines(keepends=True);mm=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',header);i=int(mm[1])-1;n=int(mm[2] or 1);assert lines[i:i+n]==[x[1:] for x in body if x.startswith((' ','+'))],(p,header)
 print('CURRENT DIFF MATCH',p)
for p in [diff]+paths+sorted(A.iterdir()):
 b=p.read_bytes();print('PIN',len(b),H(b),p)
PY
```

Raw output:
```text
18 current build inputs match; only appendix input changed between initial/final
zip 40079 2fa463bb471257b89d83414694de18f758bb40459e1b3e5e0f813297252a01c2
direct_pdf 532092 9d896a45296754a22de0ba4a4daa5f832a7900f689b62363588a02d2eb22699c
19 unique ZIP members match manifest/current selected inputs/unpacked source; main has only ArxivDraft prefix; bbl matches direct generated bibliography
10 saved direct/isolated PNG pairs exact; initial/final pages1-9 exact, page10 differs; saved extracted text exact
direct-tex.log no blocking warning patterns
isolated-tex.log no blocking warning patterns
metadata abstract exact and main hash current; stored pdfinfo10pages each
initial review snapshot exact
CURRENT DIFF MATCH docs/handover.md
CURRENT DIFF MATCH docs/m5-report.md
CURRENT DIFF MATCH paper/README.md
CURRENT DIFF MATCH paper/arxiv-metadata.md
CURRENT DIFF MATCH paper/main.tex
CURRENT DIFF MATCH paper/sections/appendix.tex
CURRENT DIFF MATCH paper/sections/discussion.tex
CURRENT DIFF MATCH paper/sections/evaluation.tex
CURRENT DIFF MATCH paper/sections/gpu.tex
CURRENT DIFF MATCH paper/sections/introduction.tex
PIN 59943 123ef18391958ea64d5532b60dd372a57c88253e7d97e2531855d21a63760035 .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task56-full2-r47-paper-final.diff
PIN 219064 7dcc5362d0137165c9e1816546ee0735a3c23db4379c43e749ca0a7ed17fd199 docs/handover.md
PIN 180334 6fbcacf0fcc1f7aa1d3b8822b9861021e47170e2fda39d44bf42601dc2a5e2fa docs/m5-report.md
PIN 15788 6a051dbf36d2407ceaa19bfe4a392f56320e6c7962efadfcd32a344f5bce7461 paper/README.md
PIN 6165 9f9cdf9a3faa3f804c7f748a38a6ddbf7ff13f9f7635ed33b13422a02b928773 paper/arxiv-metadata.md
PIN 3934 924d994b1f863a8fb2316e22b0eb263a111695800a66088667e043c72dd3e367 paper/main.tex
PIN 5982 2860523513668d311e41774a58227d381c21893fae7cbfaa2975af02de4d1537 paper/sections/appendix.tex
PIN 7444 e2a419ae4bde6776e70ac70c136fa021e5fa2798121e8aa5191c133b67babeb3 paper/sections/discussion.tex
PIN 12025 20d9030013780b736a3db31cbbe344dfce8b525e2289628ca5e1e6aef192f2d3 paper/sections/evaluation.tex
PIN 3232 82f7be2adcc0c45362e5d463597721bdfeaf92e4062cef40da0e543b651f9006 paper/sections/gpu.tex
PIN 3306 f233d8b3dd745f98c79e989d2833c499e30dc4a5cb0c148b52128fd113d2acdb paper/sections/introduction.tex
PIN 3486 58c0049c11bddfd0a1157d2a85fbc1c3d3e646a1e46f617b3bdf1c2c4cdaa01e results/validation/paper-full2-r47-20261108/README.md
PIN 2750 0a665956d40508949e5774d8e7f898a4212c99fe72c3f715ff6786e571930d2e results/validation/paper-full2-r47-20261108/direct-build-meta.json
PIN 320 192b658511770a0b03653a844a1f154692861d1ad6fc8c02bcef8a3f6ebcf1c2 results/validation/paper-full2-r47-20261108/direct-build.stderr
PIN 72070 cef8514719d7b389f33fc3ca4c3ba85865238788067be3ea9771abc7f6a39d6b results/validation/paper-full2-r47-20261108/direct-build.stdout
PIN 883 0a8ddfa31a2bae486c4a47e0256375ce78bf9aed2e9fbdb639268aacd07e7fc7 results/validation/paper-full2-r47-20261108/direct-pdfinfo.txt
PIN 62195 641c3c860ec8a28f3fda91141fe54a640e6846d3ff59e847fb1232aeed70d7f0 results/validation/paper-full2-r47-20261108/direct-tex.log
PIN 2674 31b130cf261d38c1d1ca2acf1fbe65f7c57af11a4ee8fde1e1c050633e40299b results/validation/paper-full2-r47-20261108/initial-build-meta.json
PIN 8581 4cdba2512d1b0e50643a02dafaa273c4e15620e8aac62f4a92d03aac0bc02d9b results/validation/paper-full2-r47-20261108/initial-content-review.md
PIN 786 9f85f3884017c2e055a8543c7a7abb8edc8b92d7fb44f04d80de508e9ca37d16 results/validation/paper-full2-r47-20261108/isolated-build-result.json
PIN 104 e9fd09dc0839d7100cac4d9ee46d5eb96f8d57bbbc62c2f70451c5b05c171b2d results/validation/paper-full2-r47-20261108/isolated-build.stderr
PIN 47753 543461e7f0ce5acc3d9c7c6ab9d6efc3dacd56f11afd719fe9d0e2fb4a1cc517 results/validation/paper-full2-r47-20261108/isolated-build.stdout
PIN 883 7f0921da9a02c49d61d56903059807403930bf3a7c23115c7b564948e557ba8e results/validation/paper-full2-r47-20261108/isolated-pdfinfo.txt
PIN 61980 4ef58e441b1601f934b599ab38c85bd174ff41d34a7e60ccf0813b97aa2ac108 results/validation/paper-full2-r47-20261108/isolated-tex.log
PIN 43836 ad478a70837938ee45e8cbe6bcc137dbdf5f81a240081ca5ae392f27d66df917 results/validation/paper-full2-r47-20261108/package-manifest.json
PIN 3418 71c25509f94ed978e997548b48d13c032966db54e08e8822993773ad610e0f28 results/validation/paper-full2-r47-20261108/verification.json

```
