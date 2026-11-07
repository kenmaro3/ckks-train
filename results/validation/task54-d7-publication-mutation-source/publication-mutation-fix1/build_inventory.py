"""Read source spans and archived pins; no test discovery or execution."""
from pathlib import Path
import json,re,hashlib,difflib
root=Path(__file__).parent
ledger=root.parent
before=ledger/'publication-mutation-before-fix1'
expected=json.loads((root/'expected-opt-in-functions.json').read_text())
oldpins=json.loads((root/'before-source-pins.json').read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def functions(text):
 # Mask ordinary Rust strings and comments without shifting source offsets.
 masked=re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',lambda m:' '*(m.end()-m.start()),text,flags=re.S)
 found={}
 for m in re.finditer(r'\bfn (\w+)\s*(?:<[^\n]*>)?\s*\(',masked):
  start=masked.find('{',m.end())
  if start<0:continue
  depth=1;end=start+1
  while depth:
   depth+=(masked[end]=='{')-(masked[end]=='}');end+=1
  found[m[1]]=(text[m.start():end],m.start(),end)
 return found
changed={};diff=[]
for f,h in oldpins.items():
 old=(before/f).read_bytes();new=Path(f).read_bytes()
 assert sha(old)==h,f
 if old!=new:
  changed[f]={'before':h,'after':sha(new)}
  assert f in expected,f
  diff += difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='before-fix1/'+f,tofile='after-fix1/'+f)
  oldfn=functions(old.decode());newfn=functions(new.decode())
  altered=[name for name,v in oldfn.items() if name in newfn and v[0]!=newfn[name][0]]
  assert set(altered)<=set(expected[f])|{'run_child'},(f,altered)
  changed[f]['changed_bodies']=altered
assert len(changed)==2,changed
(root/'scoped-source.diff').write_text(''.join(diff))
(root/'source-change-inventory.json').write_text(json.dumps(changed,indent=2)+'\n')
(root/'final-source-pins.json').write_text(json.dumps({f:sha(Path(f).read_bytes()) for f in oldpins},indent=2)+'\n')
release=json.loads((ledger/'task-54-d7-publication-mutation-runtime1-release.json').read_text())
mac=[]
for name in release['tests']:
 short=name.rsplit('::',1)[1]
 if name.startswith('study_reference::'):
  f='crates/ckks-train-bench/src/study_reference/target_tests.rs'
  text=Path(f).read_text()
  assert sha(text.encode())==release['source_pins'][f]
  body=functions(text)[short][0]
 else:
  f='crates/ckks-train-bench/src/study_provenance/'+('publication_anonymous_tests.rs' if 'anonymous_contract::' in name else 'publication_tests.rs')
  old=(before/f).read_text();new=Path(f).read_text()
  oldbody=functions(old)[short][0];body=functions(new)[short][0]
  assert body==oldbody,name
  assert short not in expected[f],name
 mac.append({'filter':name,'source':f,'body_sha256':sha(body.encode()),'body_unchanged':True,'ignored':False})
assert len(mac)==25
(root/'mac25-applicability.json').write_text(json.dumps(mac,indent=2)+'\n')
oldfilters=json.loads((ledger/'publication-mutation-source-author1/exact-source-test-filters.json').read_text())
newfilters=[]
for row in oldfilters:
 f=row['source'];name=row['filter'].rsplit('::',1)[1]
 text=Path(f).read_text();fn=functions(text)[name]
 ignored=name in expected.get(f,[])
 newfilters.append(dict(row,line=text[:fn[1]].count('\n')+1,ignored=ignored,
  child_only=name in ['publication_native_child','target_native_linux_publication_child']))
assert sum(r['ignored'] for r in newfilters)==23
(root/'exact-filter-inventory.json').write_text(json.dumps(newfilters,indent=2)+'\n')
print('Only2 test files changed;5 production/dependency files identical. All25 accepted Mac test bodies/default selection unchanged.48 source filters;23 opt-in, including2 affected child-only entrypoints.')
