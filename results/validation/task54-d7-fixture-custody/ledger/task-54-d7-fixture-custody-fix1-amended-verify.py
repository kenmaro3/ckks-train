"""Final bounded preservation/type/output checks; no preflight or test rerun."""
from pathlib import Path
import hashlib,json,stat,subprocess
ledger=Path(__file__).resolve().parent
root=ledger.parents[2]
snapshot=ledger/'fixture-custody-review-snapshot'
def pin(path):
    data=path.read_bytes()
    return dict(path=str(path),byte_length=len(data),sha256=hashlib.sha256(data).hexdigest())
def matches(actual,expected):
    return Path(actual['path']).resolve()==Path(expected['path']).resolve() and all(actual[key]==expected[key] for key in ('byte_length','sha256'))
inventory=json.loads((ledger/'task-54-d7-fixture-custody-fix1-amendment-test-inventory.json').read_text())
for row in inventory['interim_fix1_files_preserved']:
    assert matches(pin(root/row['path']),row),row['path']
manifest=snapshot/'snapshot-pins.json'
assert pin(manifest)['sha256']=='8e212112f663f98956512cbd4d2e8cab0b16fc318e83e4b9715543a64a30e784'
for row in json.loads(manifest.read_text())['files']:
    for path in [root/row['original'],snapshot/row['snapshot']]:
        actual=pin(path)
        assert actual['byte_length']==row['bytes'] and actual['sha256']==row['sha256'],str(path)
licenses=[]
for row in inventory['checks']:
    path=root/row['path']; origin=root/row['root_origin']
    assert stat.S_ISREG(path.lstat().st_mode) and stat.S_IMODE(path.lstat().st_mode)==0o644
    assert path.read_bytes()==origin.read_bytes()
    actual=pin(path)
    assert actual['byte_length']==row['byte_length'] and actual['sha256']==row['sha256']
    licenses.append(dict(**actual,before_git_mode='120000',before_link_target=row['before_link_target'],after_git_mode='100644',after_permissions='0644',root_origin_pin=pin(origin),root_bytes_equal=True))
handoff=json.loads((ledger/'task-54-d7-fixture-custody-pins.json').read_text())
source=[]
for row in handoff['source_files']:
    path=root/row['path']; actual=pin(path)
    assert matches(actual,row) and stat.S_ISREG(path.lstat().st_mode),row['path']
    source.append(dict(**row,file_type='regular',permissions=oct(stat.S_IMODE(path.lstat().st_mode)),git_mode='100755' if path.lstat().st_mode&stat.S_IXUSR else '100644'))
coherence=[]
for attempt in ('01','02'):
    prefix='task-54-d7-fixture-custody-fix1-attempt-'+attempt
    outcome=json.loads((ledger/(prefix+'-process-outcome.json')).read_text())
    command=json.loads((ledger/(prefix+'-command.json')).read_text())
    assert all(outcome[key]==command[key] for key in ('command','cwd','attempt_id'))
    for key in ('stdout','stderr','command_metadata','driver'):
        assert matches(pin(Path(outcome[key]['path'])),outcome[key]),(attempt,key)
    if attempt=='01':
        assert outcome['actual_exit_status']==1 and outcome['result'] is None and not (ledger/(prefix+'-result.json')).exists()
    else:
        result_path=ledger/(prefix+'-result.json'); result=json.loads(result_path.read_text())
        assert outcome['actual_exit_status']==0 and matches(pin(result_path),outcome['result'])
        assert result['driver_sha256']==outcome['driver']['sha256']
        assert result['records']==handoff['source_files']
        assert result['selected_count']==410 and result['manifest_encoded_bytes']==54531 and result['accepted']
        assert json.loads((ledger/(prefix+'-stdout.log')).read_text())=={key:value for key,value in result.items() if key!='records'}
        assert (ledger/(prefix+'-stderr.log')).stat().st_size==0
    coherence.append(dict(attempt_id=outcome['attempt_id'],actual_exit_status=outcome['actual_exit_status'],coherent_command_driver_output_hashes=True,pure_preflight_calls=0 if attempt=='01' else 1,result_present=outcome['result'] is not None))
result=dict(kind='FIX1_final_preservation_and_type_content_checks',original43_unchanged=True,snapshot43_unchanged=True,snapshot_manifest_unchanged=True,all_interim_fix1_artifacts_unchanged=True,all410_content_pins_unchanged=True,source_types_changed=8,all410_lstat_regular=True,source_type_changes=licenses,source_files=source,attempt_coherence=coherence,raw_mac_tree_equality='not tested; historical failure remains',native_build='not run')
with (ledger/'task-54-d7-fixture-custody-fix1-amended-verification.json').open('x') as output:
    json.dump(result,output,indent=2);output.write('\n')
base='ec2d2656f21e9139d3dde12abdce13d770ea2757'
license_paths=[row['path'] for row in inventory['checks']]
type_patch=subprocess.check_output(['git','diff','--binary',base,'--',*license_paths],cwd=root)
with (ledger/'task-54-d7-fixture-custody-fix1-source.diff').open('xb') as output:output.write(type_patch)
original=json.loads((ledger/'task-54-d7-fixture-custody-original-inventory.json').read_text())
delta=json.loads((ledger/'task-54-d7-fixture-custody-physical-delta.json').read_text())
full_patch=subprocess.check_output(['git','diff','--binary',base,'--','crates','scripts','tools',*original['fixed']],cwd=root)
for path in delta['added']:
    completed=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',path],cwd=root,stdout=subprocess.PIPE)
    assert completed.returncode==1
    full_patch+=completed.stdout
with (ledger/'task-54-d7-fixture-custody-fix1-full-source.diff').open('xb') as output:output.write(full_patch)
print('PASS:43 originals+43 snapshots+interim FIX1 files unchanged;410 content pins unchanged;410 regular;8 exact root-byte0644 copies; attempt01/02 output hashes coherent.')
