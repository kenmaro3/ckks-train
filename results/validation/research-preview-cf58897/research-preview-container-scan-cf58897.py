import ast, datetime, gzip, hashlib, io, json, pathlib, re, subprocess, zipfile
root=pathlib.Path('/Users/marconi/Documents/research/hekit');l=root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5';cwd=root/'tools/release/_ext/research-preview-candidate'
original=json.loads((l/'research-preview-history-scan-cf58897.json').read_text())
source=(l/'research-preview-history-scan-cf58897.py').read_text();patterns=None
for node in ast.parse(source).body:
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='patterns' and isinstance(node.value,ast.Dict):patterns={k:re.compile(v) for k,v in ast.literal_eval(node.value).items()};break
assert patterns
report={'source_commit':original['source_commit'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rules_from_script_sha256':hashlib.sha256(source.encode()).hexdigest(),'entries':[],'scope':'Same limited credential signatures, decompressed gzip logs and zip members. No extraction, entropy detection, comprehensive audit or release approval.'}
total=0
def scan(stream,label):
 global total
 counts={};length=0;digest=hashlib.sha256();tail=b''
 while True:
  chunk=stream.read(1024*1024)
  if not chunk:break
  length+=len(chunk);total+=len(chunk)
  if length>1024**3 or total>2*1024**3:raise RuntimeError('decompression cap exceeded')
  digest.update(chunk);data=tail+chunk
  for name,pat in patterns.items():
   n=sum(1 for m in pat.finditer(data) if m.end()>len(tail))
   if n:counts[name]=counts.get(name,0)+n
  tail=data[-4096:]
 return {'member':label,'bytes':length,'sha256':digest.hexdigest(),'findings':counts}
for row in original['containers_requiring_separate_inspection']:
 data=subprocess.check_output(['git','cat-file','blob',row['object']],cwd=cwd);assert len(data)==row['bytes']
 item={'object':row['object'],'path':row['path'],'members':[]}
 if data[:2]==b'\x1f\x8b':
  with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:item['members'].append(scan(f,'gzip-log-stream'))
 else:
  with zipfile.ZipFile(io.BytesIO(data)) as z:
   members=z.infolist();assert len(members)<=4096
   for m in members:
    if m.is_dir():continue
    assert not m.flag_bits&1 and m.file_size<=1024**3
    with z.open(m) as f:item['members'].append(scan(f,m.filename))
 item['status']='limited-signature-scan-complete';report['entries'].append(item)
report['decompressed_bytes']=total;report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
path=l/'research-preview-container-scan-cf58897.json'
with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'containers':len(report['entries']),'decompressed_bytes':total,'matching_members':sum(bool(m['findings']) for e in report['entries'] for m in e['members']),'report':str(path)}))
