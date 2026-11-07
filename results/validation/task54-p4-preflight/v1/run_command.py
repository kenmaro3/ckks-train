import os,sys,subprocess,time,shutil,json
from pathlib import Path
label=sys.argv[1]
command=sys.argv[2:]
root=Path(__file__).parent
env=os.environ.copy()
env.update(CARGO_INCREMENTAL="0",CARGO_BUILD_JOBS="8",RAYON_NUM_THREADS="8")
free=shutil.disk_usage(".").free
if free<805306368: raise SystemExit("disk floor violation")
meta={"command":command,"disk_free_before":free,"CARGO_INCREMENTAL":"0","CARGO_BUILD_JOBS":"8","RAYON_NUM_THREADS":"8","start_unix":time.time()}
with (root/(label+".stdout.log")).open("wb") as out,(root/(label+".stderr.log")).open("wb") as err:
 result=subprocess.run(command,env=env,stdout=out,stderr=err)
meta.update(exit_code=result.returncode,end_unix=time.time(),disk_free_after=shutil.disk_usage(".").free)
(root/(label+".status.json")).write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta))
print((root/(label+".stdout.log")).read_text()[-2000:])
print((root/(label+".stderr.log")).read_text()[-4000:])
sys.exit(result.returncode)
