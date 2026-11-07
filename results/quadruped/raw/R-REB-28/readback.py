"""Independently read four pinned context1 files; no job polling or target execution.

New device/inode observations must be joined to the separate deployment receipt.
"""
from pathlib import Path
import json,os,time
import hashlib,stat
expected=[{'path': '/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-prelaunch-v1.py', 'bytes': 7044, 'sha256': 'c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0', 'uid': 1000, 'mode': 384}, {'path': '/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-launch-v1.py', 'bytes': 29357, 'sha256': '17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c', 'uid': 1000, 'mode': 384}, {'path': '/home/kmihara/hekit-exporter-010bb3c-task54-v1/tools/export_reboot_block_seed_inputs.py', 'bytes': 61243, 'sha256': '0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce', 'uid': 1000, 'mode': 420}, {'path': '/home/kmihara/hekit-runs/task54-launch/v2/context1-release.json', 'bytes': 309421, 'sha256': '6137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e', 'uid': 1000, 'mode': 384}]
record={'kind':'context_sidecar_readback_not_scientific_result','observed_unix':time.time(),'nice':os.nice(0),'uid':os.getuid()}
record['deployment_files']=[];record['deployment_directories']=[]
def need(ok,why):
 if not ok:raise RuntimeError(why)
try:
 need(os.getuid()==1000 and os.nice(0)>=10,'uid/nice')
 for item in expected:
  p=Path(item['path']);need(p.resolve()==p,'canonical');a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<=1024*1024 and (item['mode']==0o644 or a.st_nlink==1),'kind/size/links')
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   b=os.fstat(fd);need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and stat.S_ISREG(b.st_mode) and b.st_size<=1024*1024,'opened identity')
   with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(1024*1024+1)
   c=os.fstat(fd);need(len(data)==b.st_size and len(data)<=1024*1024 and (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_size,c.st_mtime_ns),'read changed')
  finally:os.close(fd)
  actual={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'uid':b.st_uid,'mode':stat.S_IMODE(b.st_mode),'device':b.st_dev,'inode':b.st_ino};record['deployment_files'].append(actual);need({key:actual[key] for key in item}==item,'deployment identity drift')
 for name in ['/home/kmihara/hekit-runs/task54-launch','/home/kmihara/hekit-runs/task54-launch/v2']:
  p=Path(name);a=p.lstat();need(p.resolve()==p and stat.S_ISDIR(a.st_mode) and a.st_uid==1000 and stat.S_IMODE(a.st_mode)==0o700,'deployment directory')
  record['deployment_directories'].append({'path':name,'uid':a.st_uid,'mode':stat.S_IMODE(a.st_mode),'device':a.st_dev,'inode':a.st_ino})
 record['evidence_absent']={str(Path(expected[0]['path']).parent/n):not os.path.lexists(Path(expected[0]['path']).parent/n) for n in ['context1-parent.json','context1-child.json','context1-context.json']};need(all(record['evidence_absent'].values()),'evidence already exists')
 record['status']='readback_complete'
except Exception as e:record['status']='failed_partial';record['error']={'type':type(e).__name__,'message':str(e)}
record['finished_unix']=time.time();print(json.dumps(record,indent=2));raise SystemExit(0 if record['status']=='readback_complete' else 1)
