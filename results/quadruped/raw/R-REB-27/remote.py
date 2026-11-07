"""Fixed Task54 stdlib-only staging. Generated release is reviewed before SSH.

No target module import, scientific work, external network command or old-tree write.
The existing root-owned controller supplies key-only SSH and retains terminal state.
"""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import time

RELEASE = {'archive_relative': 'results/quadruped/raw/reboot_emlp23_full', 'bundle_head': '1220bd0d76225e312bbaca9e78d1a22093abc3b5', 'checkout': '/home/kmihara/hekit-exporter-010bb3c-task54-v1', 'disk_accounting': {'extracted_members': 112636165, 'git_object_allowance': 268435456, 'margin': 268435456, 'minimum_free': 805306368, 'retained_tar': 112650240, 'simultaneous_bytes': 556169205, 'target_tree_blobs': 62447344}, 'members': {'data/MNIST/raw/t10k-images-idx3-ubyte': {'bytes': 7840016, 'sha256': '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7'}, 'data/MNIST/raw/t10k-images-idx3-ubyte.gz': {'bytes': 1648877, 'sha256': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'}, 'data/MNIST/raw/t10k-labels-idx1-ubyte': {'bytes': 10008, 'sha256': 'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2'}, 'data/MNIST/raw/t10k-labels-idx1-ubyte.gz': {'bytes': 4542, 'sha256': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'}, 'data/MNIST/raw/train-images-idx3-ubyte': {'bytes': 47040016, 'sha256': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db'}, 'data/MNIST/raw/train-images-idx3-ubyte.gz': {'bytes': 9912422, 'sha256': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'}, 'data/MNIST/raw/train-labels-idx1-ubyte': {'bytes': 60008, 'sha256': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5'}, 'data/MNIST/raw/train-labels-idx1-ubyte.gz': {'bytes': 28881, 'sha256': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'}, 'manifest.json': {'bytes': 254985, 'sha256': '302ed14d0a443d4f23cee23ee60d5c75bb5090b48f0cf8c80aa6e36964eadee5'}, 'repo.bundle': {'bytes': 45836410, 'sha256': 'da76cd40cb8d37c872323a4c8d5ed4b4601045b29788908e24e00c2365698244'}}, 'old_git_roots': ['/home/kmihara/reboot/src/ReBoot', '/home/kmihara/reboot/src/openfhe-development', '/home/kmihara/reboot/src/openfhe-python'], 'old_producer_config': '/home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full/config_values.json', 'package': {'bytes': 112650240, 'sha256': '373efd0a284fdcfe7997162bd5d8ef1db633a238c068c740ea6e11bbb2959cc0'}, 'preserve_roots': ['/home/kmihara/reboot', '/home/kmihara/reboot/src/ReBoot', '/home/kmihara/reboot/src/openfhe-development', '/home/kmihara/reboot/src/openfhe-python', '/home/kmihara/reboot/data/MNIST/raw', '/home/kmihara/hekit-runs', '/home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full', '/home/kmihara/hekit-runs/out/R-REB-2/reboot_emlp1_full/data', '/home/kmihara/hekit-runs/task54-launch/v2'], 'raw': {'t10k-images-idx3-ubyte': {'bytes': 7840016, 'sha256': '0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7'}, 't10k-images-idx3-ubyte.gz': {'bytes': 1648877, 'sha256': '8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6'}, 't10k-labels-idx1-ubyte': {'bytes': 10008, 'sha256': 'ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2'}, 't10k-labels-idx1-ubyte.gz': {'bytes': 4542, 'sha256': 'f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6'}, 'train-images-idx3-ubyte': {'bytes': 47040016, 'sha256': 'ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db'}, 'train-images-idx3-ubyte.gz': {'bytes': 9912422, 'sha256': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'}, 'train-labels-idx1-ubyte': {'bytes': 60008, 'sha256': '65a50cbbf4e906d70832878ad85ccda5333a97f0f4c3dd2ef09a8a9eef7101c5'}, 'train-labels-idx1-ubyte.gz': {'bytes': 28881, 'sha256': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'}}, 'receiver': '/home/kmihara/hekit-stage-task54-010bb3c-v1', 'release_id': 'task54-checkout-010bb3c-v1-373efd0a284fdcfe', 'remote_git': {'bytes': 4547768, 'sha256': 'ed8b2df973b28c0fb773b9c0dc0816d3ee2b3835bb932d9153c21417bb51b362'}, 'required_free_bytes': 824604661, 'required_remaining_free_bytes': 711954421, 'target': '010bb3c82627a0b8809e0308cdcbf76615bf9b39', 'tree': '509078781b1d7933f644c515957da6c54659dfed'}  # FROZEN_RELEASE
DEADLINE = None
GIT_ENV = {'PATH':'/usr/bin:/bin','LC_ALL':'C','GIT_CONFIG_NOSYSTEM':'1',
           'GIT_CONFIG_GLOBAL':'/dev/null','GIT_CONFIG_SYSTEM':'/dev/null',
           'GIT_ATTR_NOSYSTEM':'1','GIT_OPTIONAL_LOCKS':'0','GIT_TERMINAL_PROMPT':'0'}
GIT_ARGS = ['-c','core.fsmonitor=false','-c','core.untrackedCache=false',
            '-c','core.hooksPath=/dev/null','-c','submodule.recurse=false']
PREP_ARGS = ['-c','pack.threads=1','-c','index.threads=1','-c','maintenance.auto=false','-c','gc.auto=0']


def require(ok, message):
    if not ok: raise ValueError(message)


def within_budget():
    require(DEADLINE is None or time.monotonic()<=DEADLINE,'cooperative phase deadline')


def meta(s):
    return {'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,
            'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def file_identity(path, cap=209715201):
    path=Path(path)
    # O_NOFOLLOW covers the final component; fixed trusted parent identities are
    # checked separately. Same-uid no-concurrent-writer remains the trust boundary.
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd); require(stat.S_ISREG(before.st_mode),'not regular file')
        require(before.st_size<=cap,'file size cap')
        h=hashlib.sha256(); n=0
        while True:
            within_budget()
            b=os.read(fd,min(1048576,cap+1-n))
            if not b:break
            n+=len(b); require(n<=cap,'file read cap');h.update(b)
        after=os.fstat(fd); path_after=os.lstat(path)
        require(meta(before)==meta(after)==meta(path_after),'file changed during hash')
        require(n==before.st_size,'short file')
        return {'path':str(path),'bytes':n,'sha256':h.hexdigest(),'stat':meta(after)}
    finally:os.close(fd)


def check_file(path, expected):
    x=file_identity(path,expected['bytes'])
    require(x['bytes']==expected['bytes'] and x['sha256']==expected['sha256'],'file identity mismatch '+str(path))
    return x


def absent(path):
    try:os.lstat(path)
    except FileNotFoundError:return
    raise FileExistsError(str(path))


def new_json(path, value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,sort_keys=True,indent=2);f.write('\n')


def unpack(package, output, members, expected_bytes, expected_sha):
    x=file_identity(package)
    require(x['bytes']==expected_bytes,'package size')
    require(x['sha256']==expected_sha,'package hash')
    # Complete format/member/content validation precedes creation of output.
    with tarfile.open(package,'r:') as t, open(package,'rb') as raw:
        infos=t.getmembers();require(len(infos)==len(members),'member count')
        names=[v.name for v in infos];require(len(names)==len(set(names)),'duplicate member')
        require(set(names)==set(members),'member allowlist')
        for v in infos:
            within_budget()
            p=PurePosixPath(v.name)
            require(not p.is_absolute() and '..' not in p.parts and str(p)==v.name,'unsafe member path')
            require(v.isfile() and not v.pax_headers and v.mode==0o644,'member form')
            raw.seek(v.offset); header=raw.read(512)
            require(header[257:265]==b'ustar\x0000' and header[156:157] in (b'0',b'\x00'),'not ordinary USTAR member')
            require(v.size==members[v.name]['bytes'],'member size')
            h=hashlib.sha256();f=t.extractfile(v)
            for b in iter(lambda:f.read(1048576),b''):h.update(b)
            f.close();require(h.hexdigest()==members[v.name]['sha256'],'member hash')
        absent(output);Path(output).mkdir(mode=0o700)
        # Explicit ordinary-file copy; never tar.extract/extractall or chmod/chown.
        for v in infos:
            within_budget()
            dst=Path(output)/v.name;dst.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with dst.open('xb') as out,t.extractfile(v) as src:
                shutil.copyfileobj(src,out,1048576)
            check_file(dst,members[v.name])
    return x


def safe_config(data):
    for item in data.split(b'\0'):
        if item:
            key=item.split(b'\n',1)[0].decode('ascii').lower()
            require(not key.startswith(('filter.','include.','includeif.')) and key!='core.worktree','unsafe Git config')


def git(root, args, records, prep=False, expected_rc=0, executable='/usr/bin/git'):
    within_budget()
    command=[executable,*GIT_ARGS,'-C',str(root),*(PREP_ARGS if prep else []),*args]
    started=time.time()
    # Ordinary local Git only. No hard-timeout/descendant guarantee is claimed;
    # controller retains a live handle and unknown state if it has not returned.
    x=subprocess.run(command,env=dict(GIT_ENV),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    require(len(x.stdout)+len(x.stderr)<=1048576,'Git output cap')
    records.append({'argv':command,'environment':GIT_ENV,'returncode':x.returncode,
                    'started_unix':started,'finished_unix':time.time(),
                    'stdout_b64':base64.b64encode(x.stdout).decode(),
                    'stderr_b64':base64.b64encode(x.stderr).decode(),
                    'stdout_sha256':hashlib.sha256(x.stdout).hexdigest(),
                    'stderr_sha256':hashlib.sha256(x.stderr).hexdigest()})
    require(x.returncode==expected_rc,'Git returncode')
    return x.stdout


def checkout(destination,bundle,release,records,executable='/usr/bin/git'):
    d=Path(destination);absent(d);d.mkdir(mode=0o700)
    g=lambda args,rc=0:git(d,args,records,True,rc,executable)
    g(['init','--template='])
    g(['bundle','verify',str(bundle)])
    require(g(['bundle','list-heads',str(bundle)]).decode()==release['bundle_head']+' HEAD\n','bundle heads')
    g(['-c','protocol.allow=never','-c','protocol.file.allow=always','fetch','--no-tags','--no-recurse-submodules',str(bundle),'HEAD'])
    g(['merge-base','--is-ancestor',release['target'],'FETCH_HEAD'])
    g(['checkout','--detach',release['target']])


def readback(d,manifest,records,executable='/usr/bin/git'):
    d=Path(d);require(d.resolve()==d and d.is_dir(),'checkout canonical root')
    gd=d/'.git'; require(stat.S_ISDIR(os.lstat(gd).st_mode) and gd.resolve()==gd,'real independent .git')
    for p in ('commondir','objects/info/alternates'):absent(gd/p)
    require((gd/'HEAD').read_text()==manifest['target']+'\n','detached HEAD file')
    g=lambda args,rc=0:git(d,args,records,False,rc,executable)
    cfg=g(['config','--null','--list','--includes']);safe_config(cfg)
    require(g(['rev-parse','HEAD']).decode().strip()==manifest['target'],'HEAD')
    exporter='tools/export_reboot_block_seed_inputs.py'
    require(g(['status','--porcelain=v1','--untracked-files=all','--ignore-submodules=all','--',exporter])==b'','exporter status')
    require(g(['diff','--binary','--no-ext-diff','--no-textconv','--no-renames','--ignore-submodules=all','HEAD','--',exporter])==b'','exporter diff')
    g(['symbolic-ref','-q','HEAD'],1)
    require(g(['rev-parse','HEAD^{tree}']).decode().strip()==manifest['tree'],'tree')
    tree=g(['ls-tree','-r','-l','HEAD']).decode()
    require(hashlib.sha256(tree.encode()).hexdigest()==manifest['tree_inventory_sha256'],'tree inventory')
    require(g(['status','--porcelain=v1','--untracked-files=all','--ignore-submodules=all'])==b'','repository status')
    require(g(['diff','--binary','--no-ext-diff','--no-textconv','--no-renames','--ignore-submodules=all','HEAD','--'])==b'','repository diff')
    files={p:check_file(d/p,e) for p,e in manifest['selected_files'].items()}
    return {'config_sha256':hashlib.sha256(cfg).hexdigest(),'selected_files':files,
            'head':manifest['target'],'tree':manifest['tree'],'status':'clean_observed'}


def sentinels(release):
    result={}
    for root in release['preserve_roots']:
        p=Path(root);s=os.lstat(p)
        require(stat.S_ISDIR(s.st_mode) and p.resolve()==p,'existing root identity')
        result[root]={'device':s.st_dev,'inode':s.st_ino,'mode':s.st_mode,'uid':s.st_uid,'resolved':str(p.resolve())}
    for root in release['old_git_roots']:
        for n in ('HEAD','config'):
            p=Path(root)/'.git'/n;result[str(p)]=file_identity(p,65536)
    p=Path(release['old_producer_config']);result[str(p)]=file_identity(p,200000)
    require(result[str(p)]['sha256']=='9d6a21484856b3a5761f64fa139c1da226531d31e901eec2f5d72493ad0ce444','producer config changed')
    for n in release['raw']:
        p=Path('/home/kmihara/reboot/data/MNIST/raw')/n
        try:result[str(p)]={'stat':meta(os.lstat(p))}
        except FileNotFoundError:result[str(p)]={'absent':True}
    return result


def fixed_runtime(release):
    require(sys.platform=='linux' and os.getuid()==1000,'remote Linux uid')
    require(os.getpriority(os.PRIO_PROCESS,0)>=10,'nice10 required')
    expected={'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','OMP_NUM_THREADS':'1',
              'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','RAYON_NUM_THREADS':'1'}
    require(dict(os.environ)==expected,'fixed preparation environment')
    p=Path('/home/kmihara');s=os.lstat(p)
    require(p.resolve()==p and stat.S_ISDIR(s.st_mode) and s.st_uid==1000,'trusted parent')
    check_file('/usr/bin/git',release['remote_git'])
    s=os.lstat('/usr/bin/git');require(s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o755,'Git ownership/mode')


def main():
    global DEADLINE
    DEADLINE=time.monotonic()+180
    r=RELEASE;require(isinstance(r,dict),'unfrozen template')
    require(len(sys.argv)==2 and sys.argv[1] in ('preflight','create-readback'),'fixed action')
    phase=sys.argv[1];fixed_runtime(r);recv=Path(r['receiver']);dest=Path(r['checkout'])
    records=[];out={'kind':'staging_not_scientific_result','phase':phase,'state':'partial',
                    'started_unix':time.time(),'commands':records,'release_id':r['release_id']}
    save=None
    try:
        if phase=='preflight':
            absent(recv);absent(dest)
            available=shutil.disk_usage('/home/kmihara').free
            require(available>=r['required_free_bytes'],'insufficient free disk')
            out['free_bytes']=available;out['static_sentinels']=sentinels(r)
            recv.mkdir(mode=0o700);save=recv/'pre.json'
            out['state']='preflight_complete'
        else:
            require(recv.resolve()==recv and stat.S_ISDIR(os.lstat(recv).st_mode),'receiver identity')
            absent(recv/'readback.json');absent(dest);save=recv/'readback.json'
            pre=json.loads((recv/'pre.json').read_text())
            require(pre['state']=='preflight_complete' and pre['release_id']==r['release_id'],'preflight receipt')
            before=sentinels(r);require(before==pre['static_sentinels'],'preflight sentinels changed')
            require(shutil.disk_usage('/home/kmihara').free>=r['required_remaining_free_bytes'],'insufficient remaining disk')
            out['package']=unpack(recv/'package.tar.part',recv/'payload',r['members'],r['package']['bytes'],r['package']['sha256'])
            manifest_path=recv/'payload/manifest.json';check_file(manifest_path,r['members']['manifest.json'])
            manifest=json.loads(manifest_path.read_text())
            require(manifest['target']==r['target'] and manifest['bundle_head']==r['bundle_head'],'manifest commits')
            checkout(dest,recv/'payload/repo.bundle',r,records)
            out['checkout']=readback(dest,manifest,records)
            rawroot=recv/'payload/data/MNIST/raw'
            require(set(p.name for p in rawroot.iterdir())==set(r['raw']),'raw member names')
            out['raw']={n:check_file(rawroot/n,e) for n,e in r['raw'].items()}
            after=sentinels(r);out['static_sentinels_before']=before;out['static_sentinels_after']=after
            require(before==after,'static sentinels changed')
            out.update(state='staging_verified',data_root=str(recv/'payload/data'),archive=str(dest/r['archive_relative']))
    except BaseException as e:
        out.update(state='failed_partial',error=type(e).__name__+': '+str(e));raise
    finally:
        out['finished_unix']=time.time()
        if save is not None:new_json(save,out)
        encoded=json.dumps(out,sort_keys=True).encode();require(len(encoded)<=1048576,'receipt output cap')
        print(encoded.decode(),flush=True)


if __name__=='__main__':main()
