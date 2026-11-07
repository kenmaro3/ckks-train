from pathlib import Path
import os,subprocess,json,hashlib,shutil,time
root=Path.cwd()
b=Path(__file__).resolve().parent
a=json.loads((b/'task-54-p5-pure-runtime1-authorization.json').read_text())
for name,key in [('task-54-p5-pure-fix1-review.md','review_sha256'),('task-54-p5-pure-runtime1-release.md','release_sha256'),('task-54-p5-pure-runtime1-source-pins.json','pins_sha256'),('task-54-p5-pure-source-report.md','source_report_sha256')]:
 assert hashlib.sha256((b/name).read_bytes()).hexdigest()==a[key],name
assert a['invocations']==1 and a['source_review_pass_root_read'] and not a['automatic_retry']
pins=json.loads((b/'task-54-p5-pure-runtime1-source-pins.json').read_text())
current=lambda:{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in pins}
assert current()==pins
free=shutil.disk_usage(root).free
assert free>=805306368
v=b/'task-54-p5-pure-runtime1-validation';v.mkdir()
cmd=['cargo','test','-p','ckks-train-bench','--lib','study_plain::p5_fixtures::p5_pure_block_sensitivity_gate','--','--ignored','--exact','--nocapture','--test-threads=1']
bound_env={'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'8','RAYON_NUM_THREADS':'8','CARGO_TARGET_DIR':str(root/'target')}
env=os.environ.copy();env.update(bound_env)
record={'kind':'nonscientific_test_execution_custody','command':cmd,'environment':bound_env,'free_bytes_before':free,'source_before':pins,'authorization_sha256':hashlib.sha256((b/'task-54-p5-pure-runtime1-authorization.json').read_bytes()).hexdigest(),'started_unix':time.time(),'local_controller_pid':os.getpid(),'automatic_retry':False}
with (v/'launch.json').open('x') as f:json.dump(record,f,indent=2)
with (v/'output.raw.txt').open('xb') as out:
 process=subprocess.Popen(cmd,cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 record['cargo_pid']=process.pid
 with (v/'process.json').open('x') as f:json.dump({'cargo_pid':process.pid,'controller_pid':os.getpid()},f,indent=2)
 record['test_exit']=process.wait()
record['finished_unix']=time.time();record['source_after']=current();record['source_unchanged']=record['source_after']==pins
raw=(v/'output.raw.txt').read_bytes();record['raw_bytes']=len(raw);record['raw_sha256']=hashlib.sha256(raw).hexdigest()
record['controller_exit']=record['test_exit'] if record['source_unchanged'] else 3
with (v/'terminal.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record,indent=2),flush=True)
raise SystemExit(record['controller_exit'])
