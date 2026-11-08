# Task56 anonymous format validation review

Date: 2026-11-08. Scope: `task56-anonymous-validation.diff` (paper/README.md append only), `results/validation/paper-anonymous-full2-r47-20261108/`, current18 TeX/Bib inputs and their recorded SD artifacts.

SPEC: **PASS**. QUALITY: **PASS**. No actionable findings.

The added README and archive accurately limit this validation to the default anonymous display/metadata variant. The current main.tex selects `sigconf,review,anonymous` without AuthorDraft/ArxivDraft; the recorded command builds main.tex directly, with shell escape disabled. All18 input hashes equal the reviewed named build's hashes and current files, so no scientific source, figure/table, font/margin source change accompanies this validation. The recorded6910b149172b67b07e69b92132cb05152df8ba6e commit is build provenance, not independently queried Git evidence.

Independently read the SD PDF:534419 bytes, SHA256 `63d89a196977ce827323b9a6f42c6384fc0c2377956fcd9e963d15d6eb8c767f`. Stored pdfinfo states10 pages. All ten stored page PNGs match verification hashes. Saved extracted text matches its hash, has Anonymous Author(s) on page1, contains neither the author's email nor repository-account URL, and retains the earlier bibliographic surname. This matches the stated narrow identity check, not a complete anonymity/deanonymization audit.

All7 manifest entries match and cover the archive except SHA256SUMS itself. Build-meta/stdout/stderr/main.log exactly match SD originals. Recorded build exit0 and final-log absence of overfull/missing-character/undefined-reference/citation patterns are coherent with the README. The acknowledged shell-escape, float-only-page and suppressed ACM reference-block warnings are not concealed. TeX Live2026/acmart2.21 and the provisional12-page target are local formatting context; neither WAHC2027 compliance, venue acceptance nor submission is claimed.

The README-only diff matches the current file and its local verification link exists. Earlier public preview and named package remain distinct; no new scientific experiment or research-gate result is asserted.

Limits: Root performed visual inspection of all pages. This reviewer verified stored artifact identity and record consistency without re-rendering, rebuilding or visually re-inspecting. No broader PDF-metadata/deanonymization audit or live publication-state check was performed. No Git, SSH, build, native/scientific execution, tests, source/docs/results edits or subagents were used; only this review report was written.

## Independent read-only command and raw evidence

Exit0:
```sh
python3 -B - <<'PY'
from pathlib import Path
import hashlib,json,re
H=lambda b:hashlib.sha256(b).hexdigest()
L=Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5');A=Path('results/validation/paper-anonymous-full2-r47-20261108');v=json.loads((A/'verification.json').read_text());m=json.loads((A/'build-meta.json').read_text());named=json.loads(Path('results/validation/paper-full2-r47-20261108/direct-build-meta.json').read_text());out=Path(m['output_directory'])
assert m['paper_inputs_sha256']==named['paper_inputs_sha256'];assert len(m['paper_inputs_sha256'])==18
for p,h in m['paper_inputs_sha256'].items():assert H(Path(p).read_bytes())==h,p
print('18 named/anonymous/current TeX/Bib inputs match')
entries=[]
for line in (A/'SHA256SUMS').read_text().splitlines():
 h,p=line.split(None,1);p=p.lstrip('*');assert H((A/p).read_bytes())==h,p;entries.append(p)
assert set(entries)=={p.name for p in A.iterdir() if p.is_file() and p.name!='SHA256SUMS'};print('All',len(entries),'archive manifest entries match')
for p in ['build-meta.json','build.stdout','build.stderr','main.log']:assert (A/p).read_bytes()==(out/p).read_bytes(),p
print('4 archived build/meta/log files exact to SD originals')
b=Path(v['pdf']['path']).read_bytes();assert len(b)==v['pdf']['bytes']==534419;assert H(b)==v['pdf']['sha256'];print('PDF',len(b),H(b))
text=(out/'main.txt').read_bytes();assert H(text)==v['text_sha256'];s=text.decode();assert 'Anonymous Author(s)' in s.split('\f')[0];assert 'miha.ken.19@gmail.com' not in s and 'github.com/kenmaro3' not in s;assert 'Mihara' in s;print('Saved text hash and scoped identity checks PASS; bibliographic surname retained')
assert [p['page'] for p in v['page_renders']]==list(range(1,11))
for p in v['page_renders']:assert H((out/f'page-{p["page"]:02}.png').read_bytes())==p['sha256']
print('10 saved PNG hashes match; visual inspection not repeated')
log=(A/'main.log').read_text();assert not re.search(r'Overfull|Missing character:|undefined references|Citation .*undefined|Reference .*undefined',log,re.I);assert m['exit_code']==0;assert re.search(r'^Pages:\s+10$',(A/'pdfinfo.txt').read_text(),re.M);print('Recorded build exit0,10pages,no blocking log patterns')
diff=L/'task56-anonymous-validation.diff';d=diff.read_text();assert re.findall(r'^\+\+\+ b/(.+)$',d,re.M)==['paper/README.md'];lines=Path('paper/README.md').read_text().splitlines(keepends=True)
for chunk in re.split(r'(?=^@@ )',d,flags=re.M)[1:]:
 header,*body=chunk.splitlines(keepends=True);mm=re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@',header);i=int(mm[1])-1;n=int(mm[2] or 1);assert lines[i:i+n]==[x[1:] for x in body if x.startswith((' ','+'))]
assert (Path('paper')/'../results/validation/paper-anonymous-full2-r47-20261108/README.md').exists();print('README-only diff current and local link exists')
for p in [diff,Path('paper/README.md'),Path('paper/main.tex')]+sorted(A.iterdir()):
 b=p.read_bytes();print('PIN',len(b),H(b),p)
PY
```

```text
18 named/anonymous/current TeX/Bib inputs match
All 7 archive manifest entries match
4 archived build/meta/log files exact to SD originals
PDF 534419 63d89a196977ce827323b9a6f42c6384fc0c2377956fcd9e963d15d6eb8c767f
Saved text hash and scoped identity checks PASS; bibliographic surname retained
10 saved PNG hashes match; visual inspection not repeated
Recorded build exit0,10pages,no blocking log patterns
README-only diff current and local link exists
PIN 1099 f93b58a1c74ed35da47ee9fd903b818c7083e987e8ea442dc3c1abda4a45053b .superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task56-anonymous-validation.diff
PIN 16435 b87a6126c923687559fd8d16ebfc6ea5998abaaafa2e6b8d6f02f4bf5dfbd995 paper/README.md
PIN 3934 924d994b1f863a8fb2316e22b0eb263a111695800a66088667e043c72dd3e367 paper/main.tex
PIN 1669 8dfea54b3ea20711781410b359bc3c948e5401a05fa8c01feac2dfa28a7d72fb results/validation/paper-anonymous-full2-r47-20261108/README.md
PIN 553 fc46522d5031b6a3ba3826c6586a52f5b4293906b543de3859e928c4c7dbbf25 results/validation/paper-anonymous-full2-r47-20261108/SHA256SUMS
PIN 2684 579f88b34f10d4eb9eca13ea28edbbfa4a0e09ad81a6d0988da0f31434af2869 results/validation/paper-anonymous-full2-r47-20261108/build-meta.json
PIN 306 efecf6cef939904b0ff21485bfa9bf7ed1ac7d2fdc113f3200276b8fb4b37f48 results/validation/paper-anonymous-full2-r47-20261108/build.stderr
PIN 71950 0cd420d8d03b5c7c260b8123aa254795668a672132a7700043083a7092f33224 results/validation/paper-anonymous-full2-r47-20261108/build.stdout
PIN 60167 bf4630690b74bfcc191f67b83bcd8107dcefc9ba2ed3781d9870c06bef17e787 results/validation/paper-anonymous-full2-r47-20261108/main.log
PIN 883 9b2c6f16d028283a65e644c6050a2ae204de285b56cd4ae5781f07c5eba2ff64 results/validation/paper-anonymous-full2-r47-20261108/pdfinfo.txt
PIN 2327 b3d760871bd4e6db0935c2378d430c41b319330949576128c34a4679c3fad38c results/validation/paper-anonymous-full2-r47-20261108/verification.json

```

