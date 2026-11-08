import datetime, gzip, hashlib, io, json, pathlib, re, subprocess, time
root=pathlib.Path('/Users/marconi/Documents/research/hekit')
cwd=root
ledger=root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5'
output=ledger/'research-preview-history-scan-684d7fb.json'
assert not output.exists()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()=='684d7fbffa69c9d5d3aa80a8070c5f82e5d84f1e'
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=subprocess.check_output(['git','rev-list','--objects','HEAD'],cwd=cwd).splitlines()
paths={r.split(b' ',1)[0]:r.split(b' ',1)[1].decode(errors='replace') if b' ' in r else '' for r in rows}
meta=subprocess.run(['git','cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],cwd=cwd,input=b'\n'.join(paths)+b'\n',stdout=subprocess.PIPE,check=True).stdout.splitlines()
blobs=[r.split() for r in meta if r.split()[1]==b'blob']
patterns={
 'private-key-header':rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
 'github-classic-token':rb'gh[pousr]_[A-Za-z0-9]{36,}',
 'github-finegrained-token':rb'github_pat_[A-Za-z0-9_]{60,}',
 'aws-access-key-id':rb'(?:AKIA|ASIA)[A-Z0-9]{16}',
 'slack-token':rb'xox[baprs]-[0-9A-Za-z-]{20,}',
 'openai-project-key':rb'sk-(?:proj|svcacct)-[A-Za-z0-9_-]{40,}',
}
patterns={k:re.compile(v) for k,v in patterns.items()}
findings=[];compressed=[];external=[];large=[];total=0
proc=subprocess.Popen(['git','cat-file','--batch'],cwd=cwd,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
for oid,kind,size in blobs:
 n=int(size);path=paths[oid]
 proc.stdin.write(oid+b'\n');proc.stdin.flush()
 header=proc.stdout.readline().split();assert header==[oid,kind,size]
 data=proc.stdout.read(n);assert len(data)==n and proc.stdout.read(1)==b'\n'
 total+=n
 if n>5000000:large.append({'object':oid.decode(),'path':path,'bytes':n})
 if '/_ext/' in path or path.startswith('_ext/'):external.append({'object':oid.decode(),'path':path})
 if data[:2]==b'\x1f\x8b' or data[:4]==b'PK\x03\x04':compressed.append({'object':oid.decode(),'path':path,'bytes':n,'status':'container-content-not-inspected-by-this-pass'})
 for label,pattern in patterns.items():
  count=sum(1 for _ in pattern.finditer(data))
  if count:findings.append({'object':oid.decode(),'path':path,'rule':label,'count':count})
proc.stdin.close();assert proc.wait()==0
report={'source_commit':'684d7fbffa69c9d5d3aa80a8070c5f82e5d84f1e','started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'blob_count':len(blobs),'blob_bytes':total,'patterns':list(patterns),'findings':findings,'containers_requiring_separate_inspection':compressed,'large_blobs_over_5000000_bytes':large,'external_directory_paths':external,'scope':'Limited credential-signature scan of all reachable raw blobs at candidate HEAD. Not entropy detection, a comprehensive secret audit, source-license audit, archive-member inspection or public-release approval.'}
with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'report':str(output),'blobs':len(blobs),'findings':len(findings),'containers_not_inspected':len(compressed),'large_blobs':len(large),'external_directory_paths':len(external)}))
