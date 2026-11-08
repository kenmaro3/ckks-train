import hashlib,json,pathlib,shutil,subprocess,tarfile
from datetime import datetime,timezone
root=pathlib.Path.cwd()
sha='6fba46c38f1cdb5be88b9bf1eeecab0f4566b534'
old=root/'tools/release/_ext/research-preview-bundle-0b2d17a'
out=root/'tools/release/_ext/research-preview-bundle-6fba46c'
archive=out/'ckks-train-6fba46c-source.tar.gz'
prefix='ckks-train-6fba46c/'
changed=''
def git(*args): return subprocess.check_output(['git',*args])
entries={}
for row in git('ls-tree','-rz',sha).split(b'\0'):
 if not row: continue
 info,name=row.split(b'\t',1); mode,kind,oid=info.decode().split()
 assert kind=='blob',(mode,kind,name)
 entries[name.decode()]={'mode':mode,'oid':oid}
inventory=[]
with tarfile.open(archive,'r:gz') as tar:
 for m in tar:
  if m.isdir(): continue
  assert m.name.startswith(prefix)
  rel=m.name[len(prefix):]; expected=entries.pop(rel)
  assert m.isfile(),(rel,m.type)
  data=tar.extractfile(m).read()
  oid=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  assert oid==expected['oid'],rel
  assert m.mode==int(expected['mode'],8)&0o777,(rel,m.mode,expected)
  inventory.append({'path':rel,'mode':expected['mode'],'git_blob':oid,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
assert not entries,entries
(out/'source-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
proof={'kind':'distribution_integrity_not_scientific_result','prepared_utc':datetime.now(timezone.utc).isoformat(),'source_commit':sha,'files':len(inventory),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_git_blobs_and_modes_match':True,'manuscript_input_git_delta':changed,'ci_status':'pending; no inherited pass','prior_bundle_preserved':str(old.relative_to(root))}
(root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/research-preview-6fba46c-integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof,indent=2))
