from pathlib import Path
import hashlib,json,os,re,signal,subprocess,time,sys
L=Path(__file__).resolve().parent
ROOT=L.parents[2]
release_path=L/'task-54-d7-retained-acquisition-runtime2-release.json'
release=json.loads(release_path.read_text())
assert release['released'] is True and Path.cwd().resolve()==ROOT
assert sys.platform=='darwin' and os.environ.get('CARGO_BUILD_TARGET', 'aarch64-apple-darwin')=='aarch64-apple-darwin'
assert len(release['tests'])==release['expected_test_count']==14 and len(set(release['tests']))==14
env_overrides={'CARGO_NET_OFFLINE':'true'}
child_env=dict(os.environ,**env_overrides)
for name,digest in release['inputs'].items():
    assert hashlib.sha256((L/name).read_bytes()).hexdigest()==digest,name
pins=release['source_pins']
def check_pins():
    for name,digest in pins.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
check_pins()
out=L/'task-54-d7-retained-acquisition-runtime2'
out.mkdir()
(out/'launch.json').write_text(json.dumps({'kind':'unit_validation_not_scientific','pid':os.getpid(),'start':time.time(),'release_sha256':hashlib.sha256(release_path.read_bytes()).hexdigest(),'automatic_retry':False,'env_overrides':env_overrides},indent=2)+'\n')
results=[]
for i,name in enumerate(release['tests'],1):
    check_pins()
    target_flags=['--lib']
    cmd=['cargo','test','-p','ckks-train-bench']+target_flags+[name,'--','--exact','--test-threads=1','--nocapture']
    item={'index':i,'name':name,'argv':cmd,'start':time.time()}
    with (out/f'{i:02d}.stdout').open('xb') as stdout,(out/f'{i:02d}.stderr').open('xb') as stderr:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=stdout,stderr=stderr,env=child_env,start_new_session=True)
        item['pid']=proc.pid
        (out/f'{i:02d}.launch.json').write_text(json.dumps(item,indent=2)+'\n')
        item['timeout_seconds']=release['per_filter_timeout_seconds'][name]
        item['timed_out']=False
        try:
            item['exit_code']=proc.wait(timeout=item['timeout_seconds'])
        except subprocess.TimeoutExpired:
            item['timed_out']=True
            try: os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: item['exit_code']=proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try: os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                item['exit_code']=proc.wait()
    item['end']=time.time()
    raw=(out/f'{i:02d}.stdout').read_text()
    named=re.findall(r'^test (.+?) \.\.\. ok$',raw,re.M)
    counts=[[int(v) for v in row] for row in re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',raw)]
    item['named_successes']=named
    item['result_counts']=counts
    item['expected_selected']=named==[name] and bool(counts) and [sum(r[j] for r in counts) for j in range(3)]==[1,0,0]
    for ext in ['stdout','stderr']:
        data=(out/f'{i:02d}.{ext}').read_bytes();item[ext+'_bytes']=len(data);item[ext+'_sha256']=hashlib.sha256(data).hexdigest()
    check_pins()
    (out/f'{i:02d}.meta.json').write_text(json.dumps(item,indent=2)+'\n')
    results.append(item)
    (out/'summary.json').write_text(json.dumps({'kind':'unit_validation_not_scientific','complete':i==len(release['tests']),'results':results},indent=2)+'\n')
    print(json.dumps({'index':i,'exit':item['exit_code'],'expected_selected':item['expected_selected']}),flush=True)
    if item['timed_out'] or item['exit_code']!=0 or not item['expected_selected']:raise SystemExit('test failure or wrong selection; no retry')
check_pins()
print('ALL_SELECTED_CHECKS_PASS',flush=True)
