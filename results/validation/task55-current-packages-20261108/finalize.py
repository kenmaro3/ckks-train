#!/usr/bin/env python3
import datetime,hashlib,json,os,pathlib,shutil,subprocess
ROOT=pathlib.Path(__file__).resolve().parent; REPO=ROOT.parents[3]
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def inventory(base):
    rows=[]
    for p in sorted(base.rglob('*')):
        if p.is_dir() and not p.is_symlink():continue
        st=p.lstat(); rows.append({'path':str(p.relative_to(base)),'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'kind':'symlink' if p.is_symlink() else 'file'})
    return {'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'allocated_bytes':sum(r['allocated_bytes'] for r in rows),'entries':rows}
audit=json.loads((ROOT/'package-audit.json').read_text()); commands=json.loads((ROOT/'commands.json').read_text())
assert len([r for r in commands if '-package-' in r['label']])==5
assert all(r['exit']==0 for r in commands)
assert json.loads((ROOT/'cuda-feature-archive.json').read_text())['identical_to_default']
assert not (ROOT/'logs/archive-audit.stderr').read_bytes()
assert json.loads((ROOT/'logs/archive-audit.stdout').read_text())['status']=='PASS'
external=json.loads((ROOT/'external-directory-source.json').read_text())
assert all(hashlib.sha256((pathlib.Path(x['source'])/'.cargo-checksum.json').read_bytes()).hexdigest()==x['checksum_sha256'] for x in external)
logs=[]
for p in sorted((ROOT/'logs').glob('*.stderr')):
    data=p.read_text(errors='replace'); logs.append({'path':str(p.relative_to(ROOT)),'warning_lines':sum('warning:' in l for l in data.splitlines()),'error_lines':sum('error:' in l for l in data.splitlines())})
assert not any(r['warning_lines'] or r['error_lines'] for r in logs)
save('execution.json',{'qualify_driver':{'argv':['python3','tools/release/_ext/package-current-2dc049a/qualify.py'],'exit':0,'stdout':'driver.stdout','stderr':'driver.stderr'},'archive_audit':{'argv':['python3','tools/release/_ext/package-current-2dc049a/audit.py'],'exit':0,'stdout':'logs/archive-audit.stdout','stderr':'logs/archive-audit.stderr'},'stderr_counts':logs})
cache=ROOT/'build-cache'
assert cache.is_dir() and not cache.is_symlink() and cache.parent==ROOT
cacheinv=inventory(cache); save('build-cache-before-removal.json',cacheinv)
before=inventory(ROOT); free_before=shutil.disk_usage(ROOT).free
shutil.rmtree(cache)
after=inventory(ROOT); free_after=shutil.disk_usage(ROOT).free
save('cleanup.json',{'removed_path':str(cache),'scope':'only this task reproducible CARGO_TARGET_DIR; archives/logs/source pins retained; prior temp and shared target untouched','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'removed_cache_files':cacheinv['files'],'removed_cache_logical_bytes':cacheinv['bytes'],'removed_cache_allocated_bytes':cacheinv['allocated_bytes'],'working_area_before_logical_bytes':before['bytes'],'working_area_before_allocated_bytes':before['allocated_bytes'],'working_area_after_logical_bytes':after['bytes'],'working_area_after_allocated_bytes':after['allocated_bytes'],'free_before_bytes':free_before,'free_after_bytes':free_after,'free_delta_bytes':free_after-free_before,'free_space_note':'host free-space delta may include concurrent activity','path_absent_after':not cache.exists()})
with (ROOT/'logs/git-status-after.stdout').open('wb') as out,(ROOT/'logs/git-status-after.stderr').open('wb') as err:
    p=subprocess.run(['git','status','--porcelain=v1'],cwd=REPO,stdout=out,stderr=err)
assert p.returncode==0
save('git-status-after.json',{'argv':['git','status','--porcelain=v1'],'exit':p.returncode,'note':'concurrent writer changes are outside this task; this task authored no tracked changes'})
save('checks.json',{'commit':audit['source_commit'],'packages_default':4,'cuda_feature_build':True,'all_five_offline_locked_exit_zero':True,'all_source_git_blob_bytes_modes_match':True,'archive_content_audit':'PASS','external_directory_source_checksum_files_unchanged':True,'build_cache_removed':not cache.exists(),'retained_archives':5,'uploaded_or_published':False,'linux_ci_run':False,'native_cuda_or_scientific_measurement':False,'new_external_dependencies_fetched':False})
print(json.dumps({'status':'PASS','cache_removed_bytes':cacheinv['bytes'],'retained_area_bytes':after['bytes'],'free_after_bytes':free_after}))
