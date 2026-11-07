"""Finite direct-byte inventory; calls only the existing pure preflight function."""
import hashlib, importlib.util, json, pathlib, sys
root=pathlib.Path(__file__).resolve().parents[3]
ledger=pathlib.Path(__file__).parent
inventory=json.loads((ledger/'task-54-d7-fixture-custody-original-inventory.json').read_text())
paths={r['path'] for r in inventory['selected']}
if sys.argv[1]=='green':
    paths-={r['path'] for r in inventory['original_family_files'] if not r['path'].endswith('/ADOPTION.md')}
    paths|={'crates/ckks-train-bench/src/study_reference/target_goldens/literals.json', 'crates/ckks-train-bench/src/study_reference/document/case_provenance_goldens/literals.json', 'crates/ckks-train-bench/src/study_reference/document/fixture_literals.rs'}
if sys.argv[1]=='green':
    physical={str(p.relative_to(root)) for folder in ('crates','scripts','tools') for p in (root/folder).rglob('*') if p.is_file() or p.is_symlink()}
    physical|=set(inventory['fixed'])
    assert physical==paths, ('unexpected or missing physical source records', sorted(physical^paths))
rows=[]
records=[]
for name in sorted(paths):
    data=(root/name).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    rows.append((name.encode(),1,digest))
    records.append(dict(path=name,byte_length=len(data),sha256=digest))
spec=importlib.util.spec_from_file_location('support',root/'scripts/study_reference_build_support.py')
support=importlib.util.module_from_spec(spec); spec.loader.exec_module(support)
result=dict(mode=sys.argv[1],source_records=len(rows),source_bytes=sum(r['byte_length'] for r in records),maximum_source_file_bytes=max(r['byte_length'] for r in records),records=records)
try:
    size=support.preflight_manifest(0,rows)
    result.update(accepted=True,manifest_bytes=size,manifest_cap=support.LIMIT,source_record_cap=512)
except support.Invalid as e:
    result.update(accepted=False,error=str(e),source_record_cap=512)
(ledger/f'task-54-d7-fixture-custody-preflight-{sys.argv[1]}.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='records'})
assert result['accepted']==(sys.argv[1]=='green')
assert len(rows)==(565 if sys.argv[1]=='red' else 410)
