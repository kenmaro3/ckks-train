import datetime, json, os, pathlib, subprocess, sys, time
root=pathlib.Path('/Users/marconi/Documents/research/hekit')
ledger=root/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5'
cwd=root/'tools/release/_ext/research-preview-candidate'
commands={
 'python-tests':['python3','-m','unittest','discover','-s','tools','-p','test_*.py'],
 'tables':['bash','scripts/run_all.sh','--tables-only'],
 'table-check':['python3','tools/tables.py','--check'],
 'figure-check':['python3','tools/figs.py','--check'],
 'shell-runall':['bash','-n','scripts/run_all.sh'],
 'shell-rename':['bash','-n','scripts/rename.sh'],
 'fmt':['cargo','fmt','--all','--check'],
 'study-parallel':['cargo','test','--offline','--locked','-p','ckks-train-bench','--test','study_admission','--','--test-threads=2','--nocapture'],
 'study-isolated':['cargo','test','--offline','--locked','-p','ckks-train-bench','--test','study_admission','p3_canonical_quick_full_identity_and_revalidation','--','--exact','--test-threads=2','--nocapture'],
 'tests':['cargo','test','--offline','--locked','--workspace','--no-fail-fast','--','--test-threads=2'],
 'clippy':['cargo','clippy','--offline','--locked','--workspace','--all-targets','--','-D','warnings'],
 'clippy-cuda':['cargo','clippy','--offline','--locked','--workspace','--all-targets','--features','cuda','--','-D','warnings'],
 'deny':['cargo','deny','--offline','--locked','check','licenses','bans'],
 'doc':['cargo','doc','--offline','--locked','--workspace','--no-deps'],
 'doc-cuda':['cargo','doc','--offline','--locked','--workspace','--no-deps','--features','ckks-train-cuda/cuda'],
}
label=sys.argv[1];argv=['nice','-n','10',*commands[label]]
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=cwd,text=True).strip()=='cf588971063874d4e8e400987a4c80eb9d0f30ed'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
folder=ledger/'research-preview-cf58897-evidence'/f'{stamp}-{label}';folder.mkdir(parents=True,exist_ok=False)
overrides={'CARGO_BUILD_JOBS':'2','CARGO_INCREMENTAL':'0','RAYON_NUM_THREADS':'2','RUST_TEST_THREADS':'2','CARGO_TARGET_DIR':str(root/'tools/release/_ext/preview-target-90851d2'),'PYTHONDONTWRITEBYTECODE':'1'}
if label.startswith('doc'):overrides['RUSTDOCFLAGS']='-D warnings'
env=os.environ.copy();env.update(overrides)
(folder/'command.json').write_text(json.dumps({'argv':argv,'cwd':str(cwd),'source_commit':'cf588971063874d4e8e400987a4c80eb9d0f30ed','overrides':overrides,'started_utc':stamp},indent=2)+'\n')
start=time.monotonic_ns()
print(str(folder),flush=True)
with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
 process=subprocess.Popen(argv,cwd=cwd,env=env,stdout=out,stderr=err)
 (folder/'launch.json').write_text(json.dumps({'controller_pid':os.getpid(),'child_pid':process.pid})+'\n')
 code=process.wait()
result={'exit_code':code,'elapsed_ns':time.monotonic_ns()-start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
sys.exit(code)
