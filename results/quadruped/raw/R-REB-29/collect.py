"""Read three exact context evidence files only; never invoke or retry context."""
from pathlib import Path
import base64,hashlib,json,os,stat,time
root=Path('/home/kmihara/hekit-runs/task54-launch/v2')
OWNER=1000
CAPS={'context1-parent.json':32*1024*1024,'context1-child.json':8*1024*1024,'context1-context.json':32*1024*1024}
def need(ok,why):
    if not ok:raise RuntimeError(why)
def collect():
    r={'kind':'context_evidence_byte_collection_not_scientific_result','started_unix':time.time(),'uid':os.getuid(),'nice':os.nice(0),'files':[],'bundle_complete':False}
    try:
        need(r['uid']==OWNER and r['nice']>=10,'uid/nice')
        for name,cap in CAPS.items():
            p=root/name;entry={'path':str(p),'lexists':os.path.lexists(p),'cap_bytes':cap};r['files'].append(entry)
            if not entry['lexists']:continue
            a=p.lstat();entry.update(uid=a.st_uid,mode=stat.S_IMODE(a.st_mode),bytes=a.st_size,device=a.st_dev,inode=a.st_ino)
            need(p.resolve()==p,'canonical')
            need(stat.S_ISREG(a.st_mode) and a.st_uid==OWNER and a.st_size<=cap,'kind/owner/size')
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                b=os.fstat(fd);need((a.st_dev,a.st_ino,a.st_uid,a.st_mode,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_uid,b.st_mode,b.st_size,b.st_mtime_ns) and stat.S_ISREG(b.st_mode) and b.st_uid==OWNER and b.st_size<=cap,'opened identity')
                with os.fdopen(fd,'rb',closefd=False) as f:data=f.read(cap+1)
                c=os.fstat(fd);need(len(data)==b.st_size and len(data)<=cap and (b.st_dev,b.st_ino,b.st_uid,b.st_mode,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_uid,c.st_mode,c.st_size,c.st_mtime_ns),'read changed')
            finally:os.close(fd)
            entry.update(sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode())
        r['status']='collected_available_bytes_not_context_verdict'
        r['bundle_complete']=all('base64' in entry for entry in r['files'])
    except Exception as e:r['status']='failed_partial';r['error']={'type':type(e).__name__,'message':str(e)}
    r['finished_unix']=time.time()
    return r
if __name__=='__main__':
    r=collect();print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='collected_available_bytes_not_context_verdict' else 1)
