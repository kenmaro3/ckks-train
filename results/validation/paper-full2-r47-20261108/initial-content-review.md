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

