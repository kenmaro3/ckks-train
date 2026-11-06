from pathlib import Path
import json,os,time

def proc(pid,expected=None):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text(); f=raw[raw.rindex(')')+2:].split(); uid=p.stat().st_uid
  d={'pid':pid,'exists':True,'state':f[0],'start_ticks':f[19],'uid':uid,'expected_start_ticks':expected,'matching_own_live_process':uid==os.getuid() and (expected is None or f[19]==expected) and f[0] not in ('Z','X')}
  if d['matching_own_live_process']:
   try:d['executable']=os.readlink(p/'exe')
   except OSError:d['executable']=None
   d['children']=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()]
  return d
 except FileNotFoundError:return {'pid':pid,'exists':False,'expected_start_ticks':expected}
parents=[proc(1700979,'465769819'),proc(1739838,'467041669')]
children=[proc(c) for d in parents for c in d.get('children',[])]
root=Path('/home/kmihara/hekit-runs'); log=root/'logs/R-41-FULL-2.log'
with log.open('rb') as f:
 f.seek(max(0,log.stat().st_size-8192));tail=f.read().decode(errors='replace').splitlines()[-6:]
result=Path('/home/kmihara/hekit-r41full/results/quadruped/mnist196_ni_b48_full_seed0.json')
queue=root/'post-full-v1'
record={'kind': 'read_only_job_observation_not_scientific_result', 'observed_unix': time.time(), 'nice': os.nice(0), 'uid': os.getuid(), 'processes': parents, 'direct_children': children, 'full2_log_path': str(log), 'full2_log_tail': tail, 'full2_result_path': str(result), 'full2_result_exists': result.exists(), 'queue_entries': sorted((p.name for p in queue.iterdir())) if queue.exists() else None}
import hashlib,stat
expected=[{'path': '/home/kmihara/hekit-runs/task54-launch/v1/task-54-exporter-linux-prelaunch-v1.py', 'bytes': 7044, 'sha256': 'c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0', 'uid': 1000, 'mode': 384, 'device': 66311, 'inode': 36309257}, {'path': '/home/kmihara/hekit-runs/task54-launch/v1/task-54-exporter-linux-launch-v1.py', 'bytes': 29158, 'sha256': 'e6ff0ab0e45fc680442bc184ecf665cdc63888bba13c673d5cb18441d9fecd84', 'uid': 1000, 'mode': 384, 'device': 66311, 'inode': 36309258}, {'path': '/home/kmihara/hekit-runs/task54-launch/v1/export_reboot_block_seed_inputs.py', 'bytes': 61243, 'sha256': '0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce', 'uid': 1000, 'mode': 384, 'device': 66311, 'inode': 36309259}, {'path': '/home/kmihara/hekit-runs/task54-launch/v1/startup1-release.json', 'bytes': 13833, 'sha256': 'b1840858ab2e95c256c094cbce0faaedf28b6e4b9daa23fe0bcad8e3060bcb3a', 'uid': 1000, 'mode': 384, 'device': 66311, 'inode': 36309260}]
record['kind']='startup_deployment_readback_and_job_observation_not_scientific_result'
record['deployment_files']=[];record['deployment_directories']=[]
def need(ok,why):
 if not ok:raise RuntimeError(why)
try:
 need(os.getuid()==1000 and os.nice(0)>=10,'uid/nice')
 for item in expected:
  p=Path(item['path']);need(p.resolve()==p,'canonical');a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<=1024*1024 and a.st_nlink==1,'kind/size/links')
  fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
  try:
   b=os.fstat(fd);need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and stat.S_ISREG(b.st_mode) and b.st_size<=1024*1024,'opened identity')
   with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(1024*1024+1)
   c=os.fstat(fd);need(len(data)==b.st_size and len(data)<=1024*1024 and (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_size,c.st_mtime_ns),'read changed')
  finally:os.close(fd)
  actual={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'uid':b.st_uid,'mode':stat.S_IMODE(b.st_mode),'device':b.st_dev,'inode':b.st_ino};record['deployment_files'].append(actual);need(actual==item,'deployment identity drift')
 for name in ['/home/kmihara/hekit-runs/task54-launch','/home/kmihara/hekit-runs/task54-launch/v1']:
  p=Path(name);a=p.lstat();need(p.resolve()==p and stat.S_ISDIR(a.st_mode) and a.st_uid==1000 and stat.S_IMODE(a.st_mode)==0o700,'deployment directory')
  record['deployment_directories'].append({'path':name,'uid':a.st_uid,'mode':stat.S_IMODE(a.st_mode),'device':a.st_dev,'inode':a.st_ino})
 record['evidence_absent']={str(Path(expected[0]['path']).parent/n):not os.path.lexists(Path(expected[0]['path']).parent/n) for n in ['startup1-parent.json','startup1-child.json']};need(all(record['evidence_absent'].values()),'evidence already exists')
 record['status']='readback_complete'
except Exception as e:record['status']='failed_partial';record['error']={'type':type(e).__name__,'message':str(e)}
record['finished_unix']=time.time();print(json.dumps(record,indent=2));raise SystemExit(0 if record['status']=='readback_complete' else 1)
