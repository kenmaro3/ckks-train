"""Read three fixed executable identities and host libc/kernel facts; execute none."""
from pathlib import Path
import hashlib,json,os,stat,sys,time
r={'kind':'fixed_startup_command_identity_not_scientific_result','started_unix':time.time(),'uid':os.getuid(),'nice':os.nice(0),'files':[]}
def need(ok,why):
    if not ok:raise RuntimeError(why)
items=[('/usr/lib/cargo/bin/coreutils/uname',16777216,None),('/usr/bin/python3.14',16777216,'52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41'),('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11',33554432,'2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174')]
try:
    need(r['nice']>=10,'nice below10')
    r['host']={'sys_platform':sys.platform,'uname':dict(zip(('sysname','nodename','release','version','machine'),os.uname())),'confstr_CSGNULIBCVERSION':os.confstr('CS_GNU_LIBC_VERSION')}
    for name,cap,pin in items:
        p=Path(name);need(p.resolve()==p,'noncanonical');a=p.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'kind/size')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            b=os.fstat(fd);need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and stat.S_ISREG(b.st_mode) and b.st_size<=cap,'opened changed')
            with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(cap+1)
            c=os.fstat(fd);need(len(data)==b.st_size and len(data)<=cap and (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_size,c.st_mtime_ns),'read changed')
        finally:os.close(fd)
        h=hashlib.sha256(data).hexdigest();r['files'].append({'path':name,'bytes':len(data),'sha256':h,'uid':b.st_uid,'mode':stat.S_IMODE(b.st_mode),'device':b.st_dev,'inode':b.st_ino,'mtime_ns':b.st_mtime_ns,'expected_sha256':pin,'spawned_by_observer':False})
        need(pin is None or h==pin,'known pin changed')
    r['status']='observed_no_target_startup'
except Exception as e:r['status']='failed_partial';r['error']={'type':type(e).__name__,'message':str(e)}
r['finished_unix']=time.time();print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='observed_no_target_startup' else 1)
