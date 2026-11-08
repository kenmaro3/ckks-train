from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
L=Path(__file__).resolve().parent
prefix=L/'post-full-observation-20261108-1104'
mp=L/'post-full-observation-20261108-1104-command.json'
mb=mp.read_bytes(); m=json.loads(mb)
assert datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(m['earliest_jst'])
src=Path(m['source_path']); b=src.read_bytes()
assert len(b)==1560 and hashlib.sha256(b).hexdigest()==m['source_sha256']=='17550e5d285a03329782d8ca9f3af7ad5b4e39468596d41781607a942572c243'
meta={'kind':'operational_observation_not_scientific_result','started_unix':time.time(),'local_controller_pid':os.getpid(),'command':m['command'],'source_sha256':m['source_sha256'],'source_bytes':len(b),'input_bytes_held_in_memory':True,'command_manifest_sha256':hashlib.sha256(mb).hexdigest(),'automatic_retry':False}
with Path(str(prefix)+'.launch.json').open('x') as f:json.dump(meta,f,indent=2)
with Path(str(prefix)+'.stdout').open('xb') as out,Path(str(prefix)+'.stderr').open('xb') as err:
 p=subprocess.Popen(m['command'],stdin=subprocess.PIPE,stdout=out,stderr=err)
 meta['local_ssh_pid']=p.pid
 p.communicate(b)
 meta['returncode']=p.returncode
meta['finished_unix']=time.time()
for ext in ['stdout','stderr']:
 data=Path(str(prefix)+'.'+ext).read_bytes();meta[ext+'_bytes']=len(data);meta[ext+'_sha256']=hashlib.sha256(data).hexdigest()
with Path(str(prefix)+'.meta.json').open('x') as f:json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2),flush=True)
raise SystemExit(meta['returncode'])
