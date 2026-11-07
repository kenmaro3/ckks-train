"""Independent exact byte custody against frozen BASE Git archive and inventory."""
import hashlib,json,pathlib,tarfile
root=pathlib.Path(__file__).resolve().parents[3]; ledger=pathlib.Path(__file__).parent
original=json.loads((ledger/'task-54-d7-fixture-custody-original-inventory.json').read_text())['original_family_files']
def unique(pairs):
    obj={}
    for key,value in pairs:
        assert key not in obj, ('duplicate',key)
        obj[key]=value
    return obj
with tarfile.open(ledger/'task-54-d7-fixture-custody-base-families.tar') as archive:
    base={m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
assert set(base)=={r['path'] for r in original}
result=[]
for family,count,expected in [('crates/ckks-train-bench/src/study_reference/target_goldens/',77,'60f2292de60d2409884bf5c7d00f56f482b4c3245eda5eddac935611f0a87cd3'),('crates/ckks-train-bench/src/study_reference/document/case_provenance_goldens/',81,'3c300767ab0d47364e5f49843766d62734b4d0688c992f6b4fd22ff72b5e337a')]:
    bundle=(root/family/'literals.json').read_bytes()
    assert hashlib.sha256(bundle).hexdigest()==expected
    obj=json.loads(bundle,object_pairs_hook=unique)
    assert obj['original_directory']==family
    files=obj['files']; rows=[r for r in original if r['path'].startswith(family) and not r['path'].endswith('/ADOPTION.md')]
    assert len(files)==len(rows)==count
    assert set(files)=={r['path'][len(family):] for r in rows}
    for row in rows:
        member=files[row['path'][len(family):]]; data=member['text'].encode('utf8')
        assert data==base[row['path']]
        assert len(data)==member['byte_length']==row['byte_length']
        assert hashlib.sha256(data).hexdigest()==member['sha256']==row['sha256']
    assert len(bundle)<8*1024*1024
    result.append(dict(path=family+'literals.json',members=len(rows),original_bytes=sum(r['byte_length'] for r in rows),bundle_bytes=len(bundle),sha256=expected,all_base_bytes_names_lengths_hashes_equal=True))
assert sum(r['members'] for r in result)==158
# Combined bundles and their complete literal payloads fit the inherited8MiB manifest transport cap as well.
assert sum(r['bundle_bytes'] for r in result)<8*1024*1024
(ledger/'task-54-d7-fixture-custody-check-green.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
