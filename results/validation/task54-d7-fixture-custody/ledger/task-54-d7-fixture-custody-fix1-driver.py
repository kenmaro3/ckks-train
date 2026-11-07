"""FIX1 attempt01: pure preflight on explicit proposed selected records only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

parser=argparse.ArgumentParser()
parser.add_argument('--attempt-id',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
assert args.attempt_id=='task54-fixture-custody-fix1-attempt-01'
ledger=Path(__file__).resolve().parent
root=ledger.parents[2]
def sha(data): return hashlib.sha256(data).hexdigest()
def load_pinned(name,expected):
    data=(ledger/name).read_bytes()
    assert sha(data)==expected, name
    return json.loads(data)
original=load_pinned('task-54-d7-fixture-custody-original-inventory.json','64a894ffda82248843cab0b35ee25b88e288efa02a6b76d889b26835282d802b')
handoff=load_pinned('task-54-d7-fixture-custody-pins.json','a17d3ccd5f1ae6b9965933a7883cb9d2070daf6d33f8cce08f92fdb4d89758e5')
delta=load_pinned('task-54-d7-fixture-custody-physical-delta.json','f60574104dee75e54caf446e16c307c9b3c2fcc8c6caf7ccb7e8689c5a22fbb7')
base={row['path'] for row in original['selected']}
assert len(base)==len(original['selected'])==565
removed={row['path'] for row in original['original_family_files'] if not row['path'].endswith('/ADOPTION.md')}
added={
    'crates/ckks-train-bench/src/study_reference/document/case_provenance_goldens/literals.json',
    'crates/ckks-train-bench/src/study_reference/document/fixture_literals.rs',
    'crates/ckks-train-bench/src/study_reference/target_goldens/literals.json',
}
assert len(removed)==158 and removed<=base
assert set(delta['deleted'])==removed and set(delta['added'])==added
assert not added&base
selected=(base-removed)|added
pinned={row['path']:row for row in handoff['source_files']}
assert len(pinned)==len(handoff['source_files'])==410 and set(pinned)==selected
assert all(not os.path.lexists(root/name) for name in removed)
assert all((root/name).is_file() for name in added)
records=[]
rows=[]
for name in sorted(selected):
    path=root/name
    assert stat.S_ISREG(path.lstat().st_mode), name
    data=path.read_bytes()
    record=dict(path=name,byte_length=len(data),sha256=sha(data))
    assert record==pinned[name], name
    records.append(record)
    rows.append((name.encode('utf8'),1,record['sha256']))
# Import only the unchanged support module, then invoke only its pure preflight.
support_path=root/'scripts/study_reference_build_support.py'
spec=importlib.util.spec_from_file_location('support',support_path)
support=importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)
encoded_bytes=support.preflight_manifest(0,rows)
result=dict(
    attempt_id=args.attempt_id,
    kind='fresh_proposed_selected_inventory_pure_preflight_not_physical_tree_acceptance',
    driver_sha256=sha(Path(__file__).read_bytes()),
    input_handoff_pins_sha256='a17d3ccd5f1ae6b9965933a7883cb9d2070daf6d33f8cce08f92fdb4d89758e5',
    support_sha256=sha(support_path.read_bytes()),
    accepted=True,
    original_selected_count=565,
    removed_payload_count=158,
    added_source_count=3,
    selected_count=len(records),
    selected_bytes=sum(row['byte_length'] for row in records),
    manifest_encoded_bytes=encoded_bytes,
    source_record_cap=512,
    manifest_byte_cap=support.LIMIT,
    current_source_pins_unchanged=True,
    actual_mac_physical_equality='historically_failed_not_retested_or_established',
    clean_fixed_recipe_source_inventory_build='not_performed',
    records=records,
)
with args.output.open('x') as output:
    json.dump(result,output,indent=2)
    output.write('\n')
print(json.dumps({key:value for key,value in result.items() if key!='records'},sort_keys=True))
