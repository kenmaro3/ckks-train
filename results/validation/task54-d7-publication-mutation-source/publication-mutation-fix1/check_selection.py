"""Static source/harness audit only: never imports or executes project/test code."""
from pathlib import Path
import json,re
root=Path(__file__).parent
expected=json.loads((root/'expected-opt-in-functions.json').read_text())
missing=[]
for file,names in expected.items():
 text=Path(file).read_text()
 for name in names:
  m=re.search(r'((?:[ \t]*#\[[^\n]*\]\n)+)[ \t]*fn '+name+r'\(',text)
  assert m,(file,name)
  if not re.search(r'#\[ignore = "[^"\n]+"\]',m[1]):missing.append(name)
assert not missing, 'native entrypoints lacking explicit opt-in: '+', '.join(missing)
for file,helper in [
 ('crates/ckks-train-bench/src/study_provenance/publication_tests.rs','study_provenance::publication::tests::target_native_linux_publication_child'),
 ('crates/ckks-train-bench/src/study_provenance/publication_anonymous_tests.rs','study_provenance::publication::tests::anonymous_contract::linux::publication_native_child')]:
 text=Path(file).read_text()
 arrays=re.findall(r'\.args\(\[(.*?)\]\)',text,re.S)
 selected=[a for a in arrays if '"'+helper+'"' in a]
 assert len(selected)==1,helper
 assert '"--exact"' in selected[0] and '"--ignored"' in selected[0],helper
print('23 scratch-dependent parent/helper entrypoints explicitly ignored; both affected child argv select --exact --ignored.')
