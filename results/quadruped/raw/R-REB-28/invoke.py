"""Verify fixed context inputs, then replace this observer with the pinned controller."""
from pathlib import Path
import hashlib,json,os,stat,sys,time
ROOT=Path('/home/kmihara/hekit-runs/task54-launch/v2')
RELEASE=ROOT/'context1-release.json'
RELEASE_SHA='6137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e'
BASE=Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu')
PINS=[('/usr/bin/python3.14', '52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41', 16777216, 0), ('/usr/lib/cargo/bin/coreutils/uname', '48893b0fb21436b54619db80486e83ef39dfccaf1aefe83dfa00c02d6146e8c0', 16777216, 0), ('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11', '2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174', 33554432, 1000), ('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib/python3.11/site.py', '0577e4b379d8ef4aa0ecf88a264ef18c10f72ea72119afae36524d1aae77c9bf', 1048576, 1000), ('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib/python3.11/site-packages/_distutils_hack/__init__.py', 'df81e6bcba34ee3e3952f776551fb669143b9490fdd6c4caeb32609f97e985b4', 1048576, 1000), ('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib/python3.11/site-packages/distutils-precedence.pth', '2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224', 1048576, 1000), ('/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-launch-v1.py', '17164fd9ab47db3d566b82aea0c9ac24955132c21b363d444f98700ec3ac019c', 1048576, 1000), ('/home/kmihara/hekit-runs/task54-launch/v2/task-54-exporter-linux-prelaunch-v1.py', 'c63287a05a19a9743a2eabf316aba4bf4de35ff78aff5be4f0c4914bdbc14ff0', 1048576, 1000), ('/home/kmihara/hekit-exporter-010bb3c-task54-v1/tools/export_reboot_block_seed_inputs.py', '0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce', 1048576, 1000), ('/home/kmihara/hekit-runs/task54-launch/v2/context1-release.json', '6137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e', 1048576, 1000)]
PAYLOAD_PATHS={'archive': '/home/kmihara/hekit-exporter-010bb3c-task54-v1/results/quadruped/raw/reboot_emlp23_full', 'baseline': '/home/kmihara/reboot/src/ReBoot', 'canonical': '/home/kmihara/hekit-runs/out/R-REB-2/reboot_emlp1_full/data', 'data': '/home/kmihara/hekit-stage-task54-010bb3c-v1/payload/data', 'git': {'executable': '/usr/bin/git', 'sha256': 'ed8b2df973b28c0fb773b9c0dc0816d3ee2b3835bb932d9153c21417bb51b362'}, 'native': {'openfhe_libraries': ['/home/kmihara/reboot/install/lib/libOPENFHEbinfhe.so.1.2.1', '/home/kmihara/reboot/install/lib/libOPENFHEcore.so.1.2.1', '/home/kmihara/reboot/install/lib/libOPENFHEpke.so.1.2.1'], 'openfhe_python': '/home/kmihara/reboot/venv/lib/python3.11/site-packages/openfhe.cpython-311-x86_64-linux-gnu.so', 'reboot': '/home/kmihara/reboot/src/ReBoot/reboot_cpp.cpython-311-x86_64-linux-gnu.so'}, 'openfhe': '/home/kmihara/reboot/src/openfhe-development', 'openfhe_python': '/home/kmihara/reboot/src/openfhe-python'}
ENV={'LANG': 'C', 'LC_ALL': 'C', 'MKL_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'PATH': '/usr/bin:/bin', 'RAYON_NUM_THREADS': '1', 'SETUPTOOLS_USE_DISTUTILS': 'local'}
r={'kind':'context_only_external_verifier_before_exec','started_unix':time.time(),'uid':os.getuid(),'nice':os.nice(0),'pins':[]}
def need(ok,why):
    if not ok:raise RuntimeError(why)
def read_pin(path,pin,cap,uid):
    p=Path(path);need(p.resolve()==p,'noncanonical input');s=p.lstat();need(stat.S_ISREG(s.st_mode) and s.st_uid==uid and s.st_size<=cap,'input kind/owner/size')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        a=os.fstat(fd);need((a.st_dev,a.st_ino)==(s.st_dev,s.st_ino) and stat.S_ISREG(a.st_mode) and a.st_size<=cap,'opened identity changed')
        with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(cap+1)
        b=os.fstat(fd);need(len(data)==a.st_size and len(data)<=cap and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'input read changed')
    finally:os.close(fd)
    h=hashlib.sha256(data).hexdigest();need(h==pin,'input pin changed')
    r['pins'].append({'path':str(p),'sha256':h,'bytes':len(data),'uid':a.st_uid,'mode':stat.S_IMODE(a.st_mode),'device':a.st_dev,'inode':a.st_ino})
    return data
try:
    need(r['uid']==1000 and r['nice']>=10,'uid/nice')
    need(str(Path(sys.executable).resolve())=='/usr/bin/python3.14','outer interpreter drift')
    r['outer_interpreter']={'lexical':sys.executable,'resolved':str(Path(sys.executable).resolve()),'argv':sys.orig_argv}
    release_bytes=None
    for path,pin,cap,uid in PINS:
        raw=read_pin(path,pin,cap,uid)
        if path==str(RELEASE):release_bytes=raw
    for folder in (ROOT.parent,ROOT):
        s=folder.lstat();need(folder.resolve()==folder and stat.S_ISDIR(s.st_mode) and s.st_uid==1000 and stat.S_IMODE(s.st_mode)==0o700,'deployment directory drift')
    for name in ('task-54-exporter-linux-prelaunch-v1.py','task-54-exporter-linux-launch-v1.py','context1-release.json'):
        s=(ROOT/name).lstat();need(stat.S_IMODE(s.st_mode)==0o600 and s.st_nlink==1,'deployment mode/linkcount drift')
    e=Path('/home/kmihara/hekit-exporter-010bb3c-task54-v1/tools/export_reboot_block_seed_inputs.py').lstat();need(stat.S_IMODE(e.st_mode)==0o644 and e.st_size==61243,'genuine exporter mode/size drift')
    need(all(not os.path.lexists(ROOT/name) for name in ('context1-parent.json','context1-child.json','context1-context.json')),'prior evidence exists; no retry')
    need(not os.path.lexists('/nonexistent-task54-fictional'),'fictional warmup path now exists')
    need(str(Path('/usr/bin/uname').resolve())=='/usr/lib/cargo/bin/coreutils/uname' and os.access('/usr/bin/uname',os.X_OK),'selected uname path drift')
    s=json.loads(release_bytes);need(s['purpose']=='context' and s['payload']=={'paths':PAYLOAD_PATHS} and s['environment']==ENV and s['cwd']=='/home/kmihara/reboot' and s['timeout_seconds']==1800 and s['cleanup_seconds']==5,'release purpose/command drift')
    # Gate ordinary base site/bootstrap candidates BEFORE exec into base Python.
    for absent in s['startup']['absent']:need(not os.path.lexists(absent),'prefix/zip candidate now present')
    for name in s['startup']['customization_dirs']:
        folder=Path(name);need(folder.resolve()==folder and folder.is_dir(),'customization directory drift')
        for base in ('sitecustomize','usercustomize'):
            for suffix in ('','.py','.pyc','.pyo'):need(not os.path.lexists(folder/(base+suffix)),'base customization present')
            need(not list(folder.glob(base+'*.so')) and not list(folder.glob(base+'*.pyd')),'base extension customization present')
    need(sorted(p.name for p in Path(s['startup']['base_site']).glob('*.pth'))==s['startup']['pth_inventory'],'base pth inventory drift')
    # The reviewed sidecar hash already binds schema, all stages and every remaining field.
    cmd=[str(BASE/'bin/python3.11'),'-I','-B',str(ROOT/'task-54-exporter-linux-prelaunch-v1.py'),'--release',str(RELEASE),'--release-sha256',RELEASE_SHA]
    r.update(status='verified_before_exec_not_context_result',controller_argv=cmd,environment=ENV,cwd=s['cwd'],finished_unix=time.time(),scope='One direct exec into pinned base prelaunch; it owns a single context-only child with1800s/5s deadlines. No capture/scientific approval or native isolation claim.')
    os.chdir(s['cwd']);print(json.dumps(r,indent=2),flush=True)
    os.execve(cmd[0],cmd,ENV)
except Exception as e:
    r.update(status='verifier_failed_no_retry',error={'type':type(e).__name__,'message':str(e)},finished_unix=time.time());print(json.dumps(r,indent=2),file=sys.stderr,flush=True);raise SystemExit(1)
