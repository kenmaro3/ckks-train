from pathlib import Path
import datetime, hashlib, json, os, re, subprocess, time
L=Path(__file__).resolve().parent
ROOT=L.parents[2]
auth_path=L/'task-54-d7-train-capture-runtime1-release.json'
auth=json.loads(auth_path.read_text())
assert auth['released'] is True
assert 'CKKS_RUNTIME_POOL_TEST_CHILD' not in os.environ
assert Path.cwd().resolve()==ROOT
for name, digest in auth['inputs'].items():
    assert hashlib.sha256((L/name).read_bytes()).hexdigest()==digest, name
manifest=json.loads((L/auth['inventory']).read_text())
pins=json.loads((L/auth['source_pins']).read_text())['files']
def check_pins():
    for name,digest in pins.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
check_pins()
out=L/'task-54-d7-train-capture-runtime1'
out.mkdir()
(out/'launch.json').write_text(json.dumps({'kind':'local_unit_validation_not_scientific_result','pid':os.getpid(),'started':time.time(),'release_sha256':hashlib.sha256(auth_path.read_bytes()).hexdigest(),'automatic_retry':False},indent=2)+'\n')
commands=[]
for t in manifest['tests']:
    commands.append((t['name'],[t['name'] if x=='EXACT_NAME' else x for x in manifest['invocation_pattern']],1))
commands.append(('doctests',manifest['proposed_doctest_command'],manifest['doctest_count_from_source']))
results=[]
for i,(name,cmd,expected) in enumerate(commands,1):
    check_pins()
    item={'index':i,'name':name,'argv':cmd,'start':time.time(),'expected_parent_tests':expected}
    with (out/f'{i:02d}.stdout').open('xb') as stdout,(out/f'{i:02d}.stderr').open('xb') as stderr:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr)
        item['pid']=proc.pid
        (out/f'{i:02d}.launch.json').write_text(json.dumps(item,indent=2)+'\n')
        item['exit_code']=proc.wait()
    item['end']=time.time()
    raw=(out/f'{i:02d}.stdout').read_bytes()
    for ext in ['stdout','stderr']:
        data=(out/f'{i:02d}.{ext}').read_bytes()
        item[ext+'_sha256']=hashlib.sha256(data).hexdigest()
        item[ext+'_bytes']=len(data)
    matches=re.findall(rb'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',raw)
    item['result_counts']=[[int(v) for v in m] for m in matches]
    item['expected_selected']=bool(matches) and [int(v) for v in matches[-1]]==[expected,0,0]
    check_pins()
    (out/f'{i:02d}.meta.json').write_text(json.dumps(item,indent=2)+'\n')
    results.append(item)
    (out/'summary.json').write_text(json.dumps({'kind':'local_unit_validation_not_scientific_result','results':results,'complete':i==len(commands)},indent=2)+'\n')
    print(json.dumps({'index':i,'name':name,'exit':item['exit_code'],'counts':item['result_counts']}),flush=True)
    if item['exit_code']!=0 or not item['expected_selected']:
        raise SystemExit('failed command or wrong selection; no retry')
check_pins()
print('ALL_SELECTED_CHECKS_PASS',flush=True)
