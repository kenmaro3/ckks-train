#!/usr/bin/env python3
import datetime, hashlib, json, os, pathlib, shutil, subprocess, tarfile
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parents[3]
COMMIT='2dc049a796949b685a5c23f8b05440ee2ce474aa'
PRIOR=pathlib.Path('/var/folders/l9/j0s8_fkd3b53h97864_x2kq40000gn/T/ckks-train-renamed-packages-itehj5x0')
SRC=ROOT/'source'; TARGET=ROOT/'build-cache'; VENDOR=ROOT/'vendor'
ENV=os.environ|{'CARGO_TARGET_DIR':str(TARGET),'CARGO_BUILD_JOBS':'2','CARGO_INCREMENTAL':'0','CARGO_NET_OFFLINE':'true','RUSTUP_TOOLCHAIN':'1.85','CARGO_TERM_COLOR':'never'}
for d in ('logs','archives','manifests','vendor'): (ROOT/d).mkdir(exist_ok=True)
commands=[]
def save(name,obj): (ROOT/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def run(label,args,cwd=REPO):
    start=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'logs'/f'{label}.stdout').open('wb') as out,(ROOT/'logs'/f'{label}.stderr').open('wb') as err:
        p=subprocess.run(args,cwd=cwd,env=ENV,stdout=out,stderr=err)
    rec={'label':label,'argv':args,'cwd':str(cwd),'started_utc':start,'exit':p.returncode,'stdout':f'logs/{label}.stdout','stderr':f'logs/{label}.stderr'}
    commands.append(rec); save('commands.json',commands)
    print(json.dumps(rec),flush=True)
    return p.returncode
if SRC.exists(): raise SystemExit('refuse existing snapshot')
save('inventory-before.json',{'commit':COMMIT,'free_bytes':shutil.disk_usage(ROOT).free,'environment':{k:ENV[k] for k in ('CARGO_TARGET_DIR','CARGO_BUILD_JOBS','CARGO_INCREMENTAL','CARGO_NET_OFFLINE','RUSTUP_TOOLCHAIN')},'prior_vendor':str(PRIOR/'vendor')})
assert run('git-head',['git','rev-parse','HEAD'])==0
assert (ROOT/'logs/git-head.stdout').read_text().strip()==COMMIT
assert run('git-status-before',['git','status','--porcelain=v1'])==0
assert run('source-export',['git','archive','--format=tar.gz','--output='+str(ROOT/'source.tar.gz'),COMMIT])==0
SRC.mkdir()
with tarfile.open(ROOT/'source.tar.gz') as tf: tf.extractall(SRC,filter='data')
assert run('git-tree',['git','ls-tree','-rz',COMMIT])==0
manifest=[]
for p in sorted(SRC.rglob('*')):
    if p.is_dir(): continue
    rel=p.relative_to(SRC).as_posix(); data=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
    manifest.append({'path':rel,'kind':'symlink' if p.is_symlink() else 'file','bytes':len(data),'mode':oct(p.lstat().st_mode&0o777),'sha256':hashlib.sha256(data).hexdigest()})
(ROOT/'manifests/source.sha256.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in manifest))
external=[]
for p in sorted((PRIOR/'vendor').iterdir()):
    if p.is_dir() and not p.name.startswith(('ckks-train-','hekit-')):
        (VENDOR/p.name).symlink_to(p,target_is_directory=True)
        external.append({'directory':p.name,'source':str(p),'checksum_sha256':hashlib.sha256((p/'.cargo-checksum.json').read_bytes()).hexdigest()})
save('external-directory-source.json',external)
(ROOT/'local-source.toml').write_text('[source.crates-io]\nreplace-with = "current-packages-local"\n[source.current-packages-local]\ndirectory = '+json.dumps(str(VENDOR))+'\n')
assert run('cargo-version',['cargo','--version'],SRC)==0
assert run('rustc-version',['rustc','--version','--verbose'],SRC)==0
archives=[]
for name in ('core','ckks','nn','cuda'):
    crate='ckks-train-'+name
    args=['cargo','--config',str(ROOT/'local-source.toml'),'package','-p',crate,'--offline','--locked','-j','2']
    code=run(name+'-package-locked',args,SRC)
    if code:
        err=(ROOT/f'logs/{name}-package-locked.stderr').read_text()
        if 'lock file' not in err and 'lockfile' not in err: raise SystemExit(f'{name} concrete failure; inspect logs')
        args.remove('--locked')
        assert run(name+'-package-unlocked-normalization',args,SRC)==0
    package=TARGET/'package'/f'{crate}-1.0.0.crate'
    archive=ROOT/'archives'/package.name; shutil.copy2(package,archive)
    data=archive.read_bytes(); digest=hashlib.sha256(data).hexdigest()
    archives.append({'crate':crate,'archive':str(archive.relative_to(ROOT)),'bytes':len(data),'sha256':digest})
    save('archives.json',archives)
    with tarfile.open(archive) as tf: tf.extractall(VENDOR,filter='data')
    vend=VENDOR/(crate+'-1.0.0')
    hashes={str(p.relative_to(vend)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(vend.rglob('*')) if p.is_file() and p.name!='.cargo-checksum.json'}
    (vend/'.cargo-checksum.json').write_text(json.dumps({'files':hashes,'package':digest},sort_keys=True))
    if name=='cuda':
        args=['cargo','--config',str(ROOT/'local-source.toml'),'package','-p',crate,'--offline','--locked','-j','2','--features','cuda']
        code=run('cuda-feature-package-locked',args,SRC)
        if code:
            err=(ROOT/'logs/cuda-feature-package-locked.stderr').read_text()
            if 'lock file' not in err and 'lockfile' not in err: raise SystemExit('cuda feature concrete failure')
            args.remove('--locked'); assert run('cuda-feature-package-unlocked-normalization',args,SRC)==0
        feature=ROOT/'archives'/f'{crate}-1.0.0-cuda-feature.crate'; shutil.copy2(package,feature)
        save('cuda-feature-archive.json',{'archive':str(feature.relative_to(ROOT)),'bytes':feature.stat().st_size,'sha256':hashlib.sha256(feature.read_bytes()).hexdigest(),'identical_to_default':feature.read_bytes()==archive.read_bytes()})
print('PACKAGES BUILT; audit and cleanup remain',flush=True)
