"""Retained local custody for four fixed CONTEXT1 operations; no automatic next step.

Run only after a separate concrete root action release. A local cutoff kills only
this Popen SSH process, never a process group or a remote PID. Its remote terminal
state stays unknown. The exclusive launch receipt forbids automatic reuse/retry.
"""
from pathlib import Path
import argparse,base64,hashlib,json,os,selectors,stat,subprocess,tempfile,time
L=Path(__file__).resolve().parent
CANDIDATE_SHA='6137e3df2fe20a27147a54e3b385406aa11957ed9065cd251e6818c7e1940d8e'
COMMAND=['ssh','-o','BatchMode=yes','-o','PasswordAuthentication=no','-o','KbdInteractiveAuthentication=no','-o','ConnectionAttempts=1','-o','ConnectTimeout=90','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath=/tmp/ckks-train-ssh-%C','quadruped-gpu','nice -n 10 /usr/bin/python3.14 -I -B -']
DEADLINE_SECONDS=2100
LIMITS={'deploy':(1024*1024,32*1024*1024),'readback':(1024*1024,32*1024*1024),'invoke':(1024*1024,32*1024*1024),'collect':(100*1024*1024,32*1024*1024)}
CAPS={'context1-parent.json':32*1024*1024,'context1-child.json':8*1024*1024,'context1-context.json':32*1024*1024}
REMOTE_ROOT='/home/kmihara/hekit-runs/task54-launch/v2/'

def need(ok,why):
    if not ok:raise ValueError(why)

def unique_object(pairs):
    out={}
    for key,value in pairs:
        need(key not in out,'duplicate JSON key');out[key]=value
    return out

def strict_json(raw):
    return json.loads(raw,object_pairs_hook=unique_object,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def read_regular(path,cap):
    path=Path(path);need(path.resolve()==path,'noncanonical local input')
    a=path.lstat();need(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'local input kind/size')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        b=os.fstat(fd);need((a.st_dev,a.st_ino)==(b.st_dev,b.st_ino) and stat.S_ISREG(b.st_mode) and b.st_size<=cap,'local opened identity')
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(cap+1)
        c=os.fstat(fd);need(len(raw)==b.st_size and len(raw)<=cap and (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns)==(c.st_dev,c.st_ino,c.st_size,c.st_mtime_ns),'local read changed')
    finally:os.close(fd)
    return raw

def sha(raw):return hashlib.sha256(raw).hexdigest()

def write_json_exclusive(path,record):
    with Path(path).open('x') as f:
        json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def run_once(phase,release_path,release_sha,manifest_sha):
    need(phase in LIMITS,'unknown operation')
    manifest_path=L/'task-54-linux-context1-recovery1-transport-command-manifest.json'
    manifest_raw=read_regular(manifest_path,1024*1024);need(sha(manifest_raw)==manifest_sha,'command manifest pin')
    spec=strict_json(manifest_raw)
    need(spec['command']==COMMAND and spec['candidate_sha256']==CANDIDATE_SHA,'command/candidate drift')
    controller_sha=sha(read_regular(Path(__file__).resolve(),1024*1024));need(controller_sha==spec['controller_sha256'],'controller pin')
    name=('task-54-linux-context1-collect.py' if phase=='collect' else f'task-54-linux-context1-recovery1-{phase}.py');item=spec['sources'][phase];need(item['name']==name,'source selection')
    source=read_regular(L/name,1024*1024);need(sha(source)==item['sha256'],'source pin')
    release_path=Path(release_path).resolve();release=read_regular(release_path,1024*1024);need(sha(release)==release_sha,'concrete action release pin')
    prefix=L/f'task-54-linux-context1-recovery1-{phase}'
    paths={k:Path(str(prefix)+s) for k,s in [('launch','.launch.json'),('stdout','.stdout'),('stderr','.stderr'),('meta','.meta.json'),('cutoff','.cutoff.json')]}
    # This fixed sentinel is never removed, including on pre-spawn or terminal failure.
    meta={'kind':'context1_local_operation_custody_not_context_verdict','phase':phase,'started_unix':time.time(),'local_controller_pid':os.getpid(),'command':COMMAND,'source_path':str(L/name),'source_bytes':len(source),'source_sha256':sha(source),'candidate_sha256':CANDIDATE_SHA,'controller_sha256':controller_sha,'command_manifest_sha256':manifest_sha,'action_release_path':str(release_path),'action_release_sha256':release_sha,'observation_deadline_seconds':DEADLINE_SECONDS,'stdout_limit_bytes':LIMITS[phase][0],'stderr_limit_bytes':LIMITS[phase][1],'observation_deadline_expired':False,'cutoff_reason':None,'remote_terminal':'unknown','streams_complete':False,'returncode':None,'automatic_retry':False}
    write_json_exclusive(paths['launch'],meta)
    need(all(not os.path.lexists(paths[k]) for k in ('stdout','stderr','meta','cutoff')),'prior local custody exists; no reuse')
    counts={'stdout':0,'stderr':0};seen={'stdout':0,'stderr':0}
    proc=None;cutoff_at=None;stream_eof=set();started=time.monotonic()
    try:
        with paths['stdout'].open('xb') as out,paths['stderr'].open('xb') as err,tempfile.TemporaryFile() as src,selectors.DefaultSelector() as selector:
            src.write(source);src.seek(0)
            proc=subprocess.Popen(COMMAND,stdin=src,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            meta['local_process_pid']=proc.pid
            for key,pipe in [('stdout',proc.stdout),('stderr',proc.stderr)]:
                os.set_blocking(pipe.fileno(),False);selector.register(pipe,selectors.EVENT_READ,key)
            outputs={'stdout':out,'stderr':err};limits=dict(zip(('stdout','stderr'),LIMITS[phase]))
            while selector.get_map() or proc.poll() is None:
                now=time.monotonic()
                if meta['cutoff_reason'] is None and now-started>=DEADLINE_SECONDS:
                    meta['observation_deadline_expired']=True;meta['cutoff_reason']='observation_deadline'
                if meta['cutoff_reason'] is not None and cutoff_at is None:
                    cutoff_at=now
                    write_json_exclusive(paths['cutoff'],{'observed_unix':time.time(),'reason':meta['cutoff_reason'],'remote_terminal':'unknown','local_process_pid':proc.pid,'action':'kill local SSH only if still running; never remote PID/group; no retry'})
                    if proc.poll() is None:proc.kill()
                if cutoff_at is not None and now-cutoff_at>=5:break
                for key,_ in selector.select(.05):
                    stream=key.data
                    try:chunk=os.read(key.fileobj.fileno(),65536)
                    except BlockingIOError:continue
                    if not chunk:
                        stream_eof.add(stream);selector.unregister(key.fileobj);key.fileobj.close();continue
                    seen[stream]+=len(chunk);kept=chunk[:max(0,limits[stream]-counts[stream])]
                    outputs[stream].write(kept);counts[stream]+=len(kept)
                    if len(kept)<len(chunk) and meta['cutoff_reason'] is None:meta['cutoff_reason']=stream+'_limit'
            # A final chunk may set overflow when Popen and both pipes already ended.
            if meta['cutoff_reason'] is not None and cutoff_at is None:
                write_json_exclusive(paths['cutoff'],{'observed_unix':time.time(),'reason':meta['cutoff_reason'],'remote_terminal':'unknown','local_process_pid':proc.pid,'action':'local process already terminal; no retry'})
            if proc.poll() is None:
                proc.kill()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:meta['local_terminal_unresolved']=True
            meta['returncode']=proc.poll()
            meta['streams_complete']=meta['cutoff_reason'] is None and stream_eof=={'stdout','stderr'}
            if meta['returncode']==0 and meta['streams_complete']:meta['remote_terminal']='ssh_command_returned_zero_not_context_verdict'
            for f in outputs.values():f.flush();os.fsync(f.fileno())
    except Exception as e:
        meta['error']={'type':type(e).__name__,'message':str(e)}
        meta['streams_complete']=False;meta['remote_terminal']='unknown'
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:meta['local_terminal_unresolved']=True
        if proc is not None:meta['returncode']=proc.poll()
    finally:
        if proc is not None:
            for pipe in (proc.stdout,proc.stderr):
                if pipe is not None:pipe.close()
    # Re-read the closed files: a write/fsync exception can invalidate in-memory counts.
    final_hashes={}
    for key,limit in zip(('stdout','stderr'),LIMITS[phase]):
        try:
            retained=read_regular(paths[key],limit);counts[key]=len(retained);final_hashes[key]=sha(retained);del retained
        except Exception as e:
            counts[key]=None;final_hashes[key]=None;meta['streams_complete']=False;meta['remote_terminal']='unknown'
            meta.setdefault('custody_errors',[]).append({'stream':key,'type':type(e).__name__,'message':str(e)})
    meta.update(finished_unix=time.time(),retained_bytes=counts,observed_bytes=seen,retained_prefix_sha256=final_hashes)
    meta['hash_scope']='exact retained files; a cutoff hash is a prefix hash, not a complete stream hash'
    write_json_exclusive(paths['meta'],meta)
    return meta

def decode_collection(envelope_path,prefix):
    # Raw stdout/stderr and controller metadata already exist. Never rewrite them.
    raw=read_regular(Path(envelope_path).resolve(),100*1024*1024);r=strict_json(raw)
    need(r['kind']=='context_evidence_byte_collection_not_scientific_result','wrong envelope kind')
    need(r['status'] in ('collected_available_bytes_not_context_verdict','failed_partial'),'wrong collection status')
    need(type(r['bundle_complete']) is bool and isinstance(r['files'],list),'invalid completeness/files')
    allowed={REMOTE_ROOT+n:(n,cap) for n,cap in CAPS.items()};seen=set();available=[]
    for entry in r['files']:
        path=entry['path'];need(path in allowed and path not in seen,'duplicate/unknown path');seen.add(path)
        name,cap=allowed[path];need(type(entry['lexists']) is bool,'invalid existence')
        if 'base64' not in entry:
            need('sha256' not in entry,'hash without exact bytes')
            need(r['status']=='failed_partial' or entry['lexists'] is False,'available status omitted existing bytes')
            continue
        need(entry['lexists'] is True and type(entry['bytes']) is int and 0<=entry['bytes']<=cap,'invalid byte count')
        encoded=entry['base64'];need(isinstance(encoded,str) and len(encoded)<=4*((cap+2)//3),'base64 size')
        data=base64.b64decode(encoded,validate=True)
        need(base64.b64encode(data).decode()==encoded,'noncanonical base64')
        need(len(data)==entry['bytes'] and sha(data)==entry['sha256'],'decoded size/hash mismatch')
        available.append((name,entry))
        del data
    complete=len(available)==3 and seen==set(allowed)
    need(r['bundle_complete']==complete,'claimed completeness mismatch')
    if r['status']=='collected_available_bytes_not_context_verdict':need(seen==set(allowed),'available-byte envelope missing role metadata')
    need(not complete or r['status']=='collected_available_bytes_not_context_verdict','failed envelope claiming complete')
    destinations=[Path(str(prefix)+'-'+name) for name,_ in available]
    need(Path(prefix).parent.resolve()==Path(prefix).parent,'noncanonical decoded directory')
    for p in destinations:
        if os.path.lexists(p):raise FileExistsError(str(p))
    decoded=[]
    for (name,entry),dest in zip(available,destinations):
        data=base64.b64decode(entry['base64'],validate=True)
        with dest.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        need(sha(read_regular(dest,CAPS[name]))==entry['sha256'],'saved decoded hash mismatch')
        decoded.append({'path':str(dest),'bytes':len(data),'sha256':entry['sha256']})
        del data
    return {'kind':'context1_strict_local_decode_not_context_verdict','envelope_sha256':sha(raw),'collection_status':r['status'],'bundle_complete':complete,'decoded':decoded,'semantic_review':'Required separately: parent child_evidence_sha256 versus exact child; both release bindings, purpose, statuses, terminal/streams/cleanup; actual context content. Parent does not hash context.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    run=sub.add_parser('run');run.add_argument('phase',choices=tuple(LIMITS));run.add_argument('--action-release',type=Path,required=True);run.add_argument('--action-release-sha256',required=True);run.add_argument('--command-manifest-sha256',required=True)
    decode=sub.add_parser('decode');decode.add_argument('--stdout',type=Path,required=True);decode.add_argument('--prefix',type=Path,required=True)
    a=parser.parse_args()
    if a.mode=='run':
        result=run_once(a.phase,a.action_release,a.action_release_sha256,a.command_manifest_sha256)
        print(json.dumps(result,indent=2));return 0 if result['returncode']==0 and result['streams_complete'] else 1
    result=decode_collection(a.stdout,a.prefix);write_json_exclusive(Path(str(a.prefix)+'.decode.json'),result);print(json.dumps(result,indent=2));return 0 if result['bundle_complete'] else 1
if __name__=='__main__':raise SystemExit(main())
