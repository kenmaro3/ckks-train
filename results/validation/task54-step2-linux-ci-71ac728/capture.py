import pathlib,subprocess,json,hashlib,zipfile,re,datetime
l=pathlib.Path('.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5')
watch=json.loads((l/'ci-step2-71ac728-watch.result.json').read_text());assert watch['exit_code']==0
out=l/'ci-step2-71ac728-complete';out.mkdir(exist_ok=False)
for name,url in [('run','repos/kenmaro3/ckks-train/actions/runs/42436964378'),('jobs','repos/kenmaro3/ckks-train/actions/runs/42436964378/jobs'),('logs','repos/kenmaro3/ckks-train/actions/runs/42436964378/logs')]:
 dest=out/(name+('.zip' if name=='logs' else '.json'))
 with dest.open('xb') as so,(out/(name+'.stderr')).open('xb') as se:r=subprocess.run(['gh','api',url],stdout=so,stderr=se)
 assert r.returncode==0,(name,r.returncode)
r=json.loads((out/'run.json').read_text());j=json.loads((out/'jobs.json').read_text());sha='71ac7280be8f29f42dba3908d8b4f8b9514ddd93'
assert r['head_sha']==sha and r['status']=='completed' and r['conclusion']=='success'
assert len(j['jobs'])==6 and all(x['status']=='completed' and x['conclusion']=='success' for x in j['jobs'])
with zipfile.ZipFile(out/'logs.zip') as z:
 raw=z.read('test/8_Run cargo test --workspace.txt');tests=re.findall(rb'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',raw)
 assert tests and all(x[0]==b'ok' and x[2]==b'0' for x in tests)
 descriptor_lines=[line.decode('utf-8') for line in raw.splitlines() if b'operational_fix2_private_aliases_are_cloexec' in line]
 assert len(descriptor_lines)==1 and descriptor_lines[0].endswith(' ... ok'), descriptor_lines
 members=[{'name':n,'bytes':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()]
summary={'kind':'ci_validation_not_scientific_result','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':sha,'run_id':r['id'],'run_attempt':r['run_attempt'],'run_url':r['html_url'],'conclusion':r['conclusion'],'jobs':{x['name']:x['conclusion'] for x in j['jobs']},'workspace_summary_count':len(tests),'workspace_passed':sum(int(x[1]) for x in tests),'workspace_failed':sum(int(x[2]) for x in tests),'workspace_ignored':sum(int(x[3]) for x in tests),'linux_descriptor_test':descriptor_lines,'native16_status':'not established by this CI','logs_sha256':hashlib.sha256((out/'logs.zip').read_bytes()).hexdigest(),'members':members}
(out/'capture.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='members'},indent=2))
