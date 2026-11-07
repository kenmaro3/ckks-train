import datetime, hashlib, json, os, pathlib, shutil, subprocess, sys
root=pathlib.Path.cwd()
out=root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-p5-pure-cap-revision-validation'
name=sys.argv[1]; command=sys.argv[2:]
files=['crates/ckks-train-bench/src/study_plain.rs','crates/ckks-train-bench/src/study_plain/p5_fixtures.rs']
def pins(): return {f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files}
record={'command':command,'cwd':str(root),'environment':{'CARGO_INCREMENTAL':'0','CARGO_BUILD_JOBS':'8','RAYON_NUM_THREADS':'8'},'before':pins(),'start':datetime.datetime.now(datetime.timezone.utc).isoformat(),'free_bytes':shutil.disk_usage(root).free,'custody':'source/static only; child cargo test always --no-run; no test binary invoked'}
assert record['free_bytes']>=805306368
if command[0]=='cargo' and command[1]=='test': assert '--no-run' in command
with (out/(name+'.stdout')).open('wb') as stdout, (out/(name+'.stderr')).open('wb') as stderr:
 process=subprocess.Popen(command,stdout=stdout,stderr=stderr,env={**os.environ,**record['environment']}); record['pid']=process.pid; record['exit']=process.wait()
record['end']=datetime.datetime.now(datetime.timezone.utc).isoformat(); record['after']=pins()
(out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record)); print((out/(name+'.stdout')).read_text()); print((out/(name+'.stderr')).read_text())
