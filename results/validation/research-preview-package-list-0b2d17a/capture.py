import datetime,json,os,pathlib,subprocess,hashlib
root=pathlib.Path('/Users/marconi/Documents/research/hekit')
cwd=root/'tools/release/_ext/research-preview-candidate'
sha='0b2d17a14cd305890d9eed771008d4f8617eff88'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()==sha
assert not subprocess.check_output(['git','status','--porcelain'],cwd=cwd)
out=root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-package-list-0b2d17a'
out.mkdir(exist_ok=False)
rows=[]
for name in ['core','ckks','nn','cuda']:
 argv=['nice','-n','10','cargo','package','--offline','--locked','--list','-p','ckks-train-'+name]
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 result=subprocess.run(argv,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 for suffix,data in [('stdout',result.stdout),('stderr',result.stderr)]:
  (out/(name+'.'+suffix)).write_bytes(data)
 files=result.stdout.decode().splitlines()
 rows.append({'crate':'ckks-train-'+name,'argv':argv,'cwd':str(cwd),'source_commit':sha,'started_utc':start,'exit_code':result.returncode,'files':files,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'license_included':'LICENSE' in files,'notices_included':'THIRD_PARTY_NOTICES' in files,'readme_included':'README.md' in files})
 print(name,result.returncode,len(files),flush=True)
(out/'results.json').write_text(json.dumps({'kind':'package-list-only-not-build-or-upload','rows':rows},indent=2)+'\n')
assert all(r['exit_code']==0 and r['license_included'] and r['notices_included'] and r['readme_included'] for r in rows)
