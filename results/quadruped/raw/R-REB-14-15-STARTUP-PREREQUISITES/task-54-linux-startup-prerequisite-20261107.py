"""Read-only fixed startup candidates and derived platform call facts; no target import."""
from pathlib import Path
import ast, hashlib, json, os, stat, sys, time
BASE = Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu')
SOURCE = BASE / 'lib/python3.11/platform.py'
PIN = '4a3cc68ba7354f8d5cc28c55315a7d77b77dcdbbaa3a092572a636b5fb5b6938'
OUT = {'kind':'startup_prerequisite_read_only_not_scientific_result', 'started_unix':time.time(), 'uid':os.getuid(), 'nice':os.nice(0)}
def need(ok, message):
    if not ok: raise RuntimeError(message)
def metadata(p):
    p=Path(p); r={'path':str(p),'lexists':os.path.lexists(p)}
    if not r['lexists']: return r
    s=p.lstat(); r.update(resolved=str(p.resolve()),uid=s.st_uid,mode=stat.S_IMODE(s.st_mode),bytes=s.st_size,device=s.st_dev,inode=s.st_ino,kind='symlink' if stat.S_ISLNK(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'other')
    if stat.S_ISLNK(s.st_mode):r['target']=os.readlink(p)
    return r
def read(p, cap):
    p=Path(p); need(p.resolve()==p,'noncanonical read'); before=p.lstat();need(stat.S_ISREG(before.st_mode) and before.st_size<=cap,'read kind/size')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        a=os.fstat(fd); need((a.st_dev,a.st_ino)==(before.st_dev,before.st_ino) and stat.S_ISREG(a.st_mode) and a.st_size<=cap,'opened file changed')
        with os.fdopen(fd,'rb',closefd=False) as f: data=f.read(cap+1)
        b=os.fstat(fd);need(len(data)<=cap and (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns),'changed/oversized read')
    finally:os.close(fd)
    return data
need(OUT['nice']>=10,'nice below10')
paths=['/home/kmihara/hekit-runs','/home/kmihara/hekit-runs/task54-launch','/home/kmihara/hekit-runs/task54-launch/v1','/home/kmihara/reboot','/home/kmihara/reboot/venv/lib/python3.11/site-packages','/nonexistent-task54-fictional','/usr/bin/uname','/bin/uname']
OUT['fixed_candidates']=[metadata(p) for p in paths]
exe=Path(sys.executable).resolve(); data=read(exe,64*1024*1024)
OUT['observer_interpreter']={'lexical':sys.executable,**metadata(exe),'sha256':hashlib.sha256(data).hexdigest(),'version':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'argv':sys.orig_argv if hasattr(sys,'orig_argv') else sys.argv}
OUT['command_candidates']=[]
for name in ['/usr/bin/uname','/bin/uname']:
    p=Path(name)
    if not os.path.lexists(p):continue
    resolved=p.resolve(); data=read(resolved,2*1024*1024)
    OUT['command_candidates'].append({'candidate':name,**metadata(resolved),'sha256':hashlib.sha256(data).hexdigest(),'executed':False})
data=read(SOURCE,524288);need(hashlib.sha256(data).hexdigest()==PIN,'platform source pin changed')
OUT['platform_source']={**metadata(SOURCE),'sha256':PIN}
tree=ast.parse(data,filename=str(SOURCE)); events=[]
def expr(n,depth=0):
    if n is None:return None
    need(depth<=35,'expression depth cap')
    if isinstance(n,ast.Constant):
        v=n.value
        if isinstance(v,(str,bytes)) and len(v)>160:return {'literal_type':type(v).__name__,'length':len(v)}
        if isinstance(v,bytes):return {'bytes_hex':v.hex()}
        if v is None or isinstance(v,(str,int,float,bool)):return {'literal':v}
        return {'literal_type':type(v).__name__}
    if isinstance(n,ast.Name):return {'name':n.id}
    if isinstance(n,ast.Attribute):return {'attribute':n.attr,'on':expr(n.value,depth+1)}
    # Derived AST structure only; no source bodies, comments, or prose text.
    if isinstance(n,(ast.expr,ast.comprehension,ast.keyword,ast.arguments,ast.arg)):
        d={'node':type(n).__name__}
        for k,v in ast.iter_fields(n):
            if k in ('ctx','type_comment'):continue
            if isinstance(v,ast.AST):d[k]=expr(v,depth+1)
            elif isinstance(v,list):d[k]=[expr(x,depth+1) if isinstance(x,ast.AST) else x for x in v]
            elif isinstance(v,(str,int,float,bool)) or v is None:d[k]=v
        return d
    return {'node':type(n).__name__}
selected={'platform','uname','uname_result','_Processor','_node','libc_ver','architecture','_syscmd_file','_syscmd_ver'}
def visit(n,scope,enabled=False):
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
        scope=scope+[n.name];enabled=enabled or n.name in selected
        if enabled:
            e={'line':n.lineno,'scope':scope,'kind':type(n).__name__}
            if hasattr(n,'args'):e['arguments']=expr(n.args)
            if hasattr(n,'bases'):e['bases']=[expr(x) for x in n.bases]
            e['decorators']=[expr(x) for x in n.decorator_list];events.append(e)
    elif enabled and isinstance(n,(ast.Call,ast.If,ast.Return,ast.Assign,ast.AnnAssign,ast.ExceptHandler,ast.Import,ast.ImportFrom)):
        e={'line':n.lineno,'scope':scope,'kind':type(n).__name__}
        if isinstance(n,ast.Call):e['expression']=expr(n)
        elif isinstance(n,ast.If):e['condition']=expr(n.test)
        elif isinstance(n,ast.Return):e['value']=expr(n.value)
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):e['targets']=[expr(x) for x in (n.targets if isinstance(n,ast.Assign) else [n.target])];e['value']=expr(n.value)
        elif isinstance(n,ast.ExceptHandler):e['exception']=expr(n.type)
        else:e['module']=getattr(n,'module',None);e['imports']=[{'name':x.name,'asname':x.asname} for x in n.names]
        events.append(e)
    need(len(events)<=1500,'event cap')
    for child in ast.iter_child_nodes(n):visit(child,scope,enabled)
visit(tree,[])
OUT['platform_initial_cache_assignments']=[{'line':n.lineno,'targets':[expr(x) for x in n.targets],'value':expr(n.value)} for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_uname_cache','_platform_cache','_default_architecture') for t in n.targets)]
OUT['platform_ast_facts']=events
OUT['scope']='Fixed candidate metadata/hash and selected derived platform AST facts only. No observed command execution, target startup, package scan, source execution, full helper closure, scientific import, job/queue mutation, or deployment.'
OUT['finished_unix']=time.time()
raw=json.dumps(OUT,indent=2,allow_nan=False);need(len(raw.encode())<=512*1024,'output cap');print(raw)
