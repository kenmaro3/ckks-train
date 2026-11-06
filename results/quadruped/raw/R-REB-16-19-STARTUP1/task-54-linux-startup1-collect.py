"""Read exact startup1 evidence bytes only; never invoke or retry the startup."""
from pathlib import Path
import base64,hashlib,json,os,stat,time
root=Path('/home/kmihara/hekit-runs/task54-launch/v1')
r={'kind':'startup_evidence_byte_collection_not_scientific_result','started_unix':time.time(),'uid':os.getuid(),'nice':os.nice(0),'files':[]}
def need(ok,why):
 if not ok:raise RuntimeError(why)
try:
 need(r['uid']==1000 and r['nice']>=10,'uid/nice')
 for name in ['startup1-parent.json','startup1-child.json']:
  p=root/name;entry={'path':str(p),'lexists':os.path.lexists(p)};r['files'].append(entry)
  if not entry['lexists']:continue
  need(p.resolve()==p,'canonical');a=p.lstat();entry.update(uid=a.st_uid,mode=stat.S_IMODE(a.st_mode),bytes=a.st_size,device=a.st_dev,inode=a.st_ino)
  need(stat.S_ISREG(a.st_mode) and a.st_uid==1000 and a.st_size<=2*1024*1024,'kind/owner/size')
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   b=os.fstat(fd);need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and stat.S_ISREG(b.st_mode) and b.st_size<=2*1024*1024,'opened identity')
   with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(2*1024*1024+1)
   c=os.fstat(fd);need(len(data)==b.st_size and len(data)<=2*1024*1024 and (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_size,c.st_mtime_ns),'read changed')
  finally:os.close(fd)
  entry.update(sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode())
 r['status']='collected_available_bytes_not_startup_verdict'
except Exception as e:r['status']='failed_partial';r['error']={'type':type(e).__name__,'message':str(e)}
r['finished_unix']=time.time();print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='collected_available_bytes_not_startup_verdict' else 1)
