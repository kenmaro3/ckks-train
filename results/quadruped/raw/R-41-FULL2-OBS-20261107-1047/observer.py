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
print(json.dumps({'kind':'read_only_job_observation_not_scientific_result','observed_unix':time.time(),'nice':os.nice(0),'uid':os.getuid(),'processes':parents,'direct_children':children,'full2_log_path':str(log),'full2_log_tail':tail,'full2_result_path':str(result),'full2_result_exists':result.exists(),'queue_entries':sorted(p.name for p in queue.iterdir()) if queue.exists() else None},indent=2))
