"""Task54 exact-purpose Linux launcher; stdlib only, no import-time dispatch.

External root review, a trusted controller bootstrap, and no concurrent writers
are prerequisites. This module is also the pinned prelaunch validation library.
"""
from __future__ import annotations
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import types

STAGES = ('startup', 'helpers', 'append', 'exporter', 'dispatch', 'context_done')
EXPORTER_NAME = '_task54_exact_exporter'
MAX_RELEASE = 8 * 1024 * 1024
HELPERS = ('__future__', 'argparse', 'ast', 'datetime', 'errno', 'hashlib',
           'importlib', 'importlib.abc', 'importlib.machinery', 'importlib.metadata',
           'io', 'json', 'math', 'os', 'pathlib', 'platform', 're', 'shlex', 'struct',
           'subprocess', 'sys', 'types', 'stat', 'time', 'site', 'csv', 'email.parser',
           'email.feedparser', 'email.policy', 'email.message', 'encodings.ascii',
           'encodings.utf_8', 'encodings.latin_1', 'gettext', 'locale')
DIRS = {'bin', 'include', 'include/python3.11', 'lib', 'lib/python3.11', 'lib/python3.11/site-packages'}
FILES = {'pyvenv.cfg', 'bin/Activate.ps1', 'bin/activate', 'bin/activate.csh', 'bin/activate.fish'}
LINKS = {'lib64', 'bin/python', 'bin/python3', 'bin/python3.11'}

class LaunchError(RuntimeError):
    pass

def require(ok, message):
    if not ok:
        raise LaunchError(message)

def keys(value, expected, label):
    require(type(value) is dict and set(value) == set(expected.split()), 'wrong keys: ' + label)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value)) == value and '..' not in Path(value).parts, 'nonabsolute/nonlexical path')
    return Path(value)

def canonical(value):
    p = absolute(value)
    require(str(p.resolve()) == value, 'noncanonical path: ' + value)
    return p

def read_regular(path, limit=MAX_RELEASE, owner=None):
    p = canonical(str(path)); s = p.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_size <= limit, 'nonregular/oversized file: ' + str(p))
    if owner is not None:
        require(s.st_uid == owner, 'wrong file owner: ' + str(p))
    raw = p.read_bytes()
    require(len(raw) == s.st_size and len(raw) <= limit, 'file changed while reading')
    return raw

def verify_file(path, expected, limit=MAX_RELEASE, owner=None):
    require(valid_hash(expected), 'invalid SHA256')
    raw = read_regular(path, limit, owner)
    require(sha(raw) == expected, 'pin mismatch: ' + str(path))
    return raw

def strict_json(raw):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, 'duplicate JSON key'); out[k]=v
        return out
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(LaunchError('nonfinite JSON')))
    except (ValueError, UnicodeError) as exc:
        raise LaunchError('invalid JSON') from exc

def parse_transport(argv):
    require(len(argv) == 4 and argv[0] == '--release' and argv[2] == '--release-sha256', 'exact release transport required')
    canonical(argv[1]); require(valid_hash(argv[3]), 'invalid release digest')
    return argv[1], argv[3]

def validate_schema(s):
    keys(s, 'schema purpose programs interpreter base venv package_root exporter startup stages environment cwd timeout_seconds cleanup_seconds evidence payload', 'release')
    require(s['schema'] == 'hekit-linux-launch-release-v1' and s['purpose'] in ('startup-only','context','capture'), 'unsupported release')
    keys(s['programs'], 'prelaunch launcher', 'programs')
    for item in (*s['programs'].values(), s['exporter']):
        keys(item, 'path sha256', 'program identity'); canonical(item['path']); require(valid_hash(item['sha256']), 'bad program pin')
    keys(s['base'], 'prefix executable sha256 site_source site_sha256 version uid', 'base')
    b=s['base']; canonical(b['prefix']); canonical(b['executable']); canonical(b['site_source'])
    require(b['version']=='3.11.16' and type(b['uid']) is int and b['uid'] >= 0, 'unsupported base')
    require(valid_hash(b['sha256']) and valid_hash(b['site_sha256']), 'base pin missing')
    v=s['venv']; keys(v, 'path uid root_mode config_hex config_sha256 layout', 'venv'); canonical(v['path'])
    require(type(v['uid']) is int and v['uid'] >= 0 and valid_hash(v['config_sha256']), 'bad venv identity')
    require(type(v['root_mode']) is int and 0 <= v['root_mode'] <= 0o7777, 'invalid venv root mode bits')
    try: cfg=bytes.fromhex(v['config_hex'])
    except (ValueError,TypeError) as exc: raise LaunchError('bad config hex') from exc
    require(len(cfg)<=65536 and sha(cfg)==v['config_sha256'], 'config identity inconsistent')
    require(s['interpreter']==str(Path(v['path'])/'bin/python'), 'wrong lexical interpreter')
    require(type(v['layout']) is dict and set(v['layout'])==DIRS|FILES|LINKS, 'wrong 15-entry layout')
    for name,e in v['layout'].items():
        kind='directory' if name in DIRS else 'file' if name in FILES else 'symlink'
        keys(e, 'kind uid mode' + (' sha256 bytes' if kind=='file' else ' target resolved' if kind=='symlink' else ''), 'layout entry')
        require(e['kind']==kind and e['uid']==v['uid'] and type(e['mode']) is int and 0<=e['mode']<=0o777, 'wrong layout kind/owner/mode')
        if kind=='file': require(valid_hash(e['sha256']) and type(e['bytes']) is int and 0<=e['bytes']<=65536, 'invalid file layout')
    canonical(s['package_root']); canonical(s['cwd'])
    require(not Path(s['package_root']).is_relative_to(Path(v['path'])) and not Path(s['package_root']).is_relative_to(Path(b['prefix'])), 'external root overlaps interpreter roots')
    st=s['startup']; keys(st,'paths trusted_roots absent customization_dirs base_site pth_inventory base_site_pins helper_sources','startup')
    for field in ('paths','trusted_roots','absent','customization_dirs'):
        require(type(st[field]) is list and len(st[field])==len(set(st[field])), 'bad startup list')
        for p in st[field]: canonical(p)
    base=Path(b['prefix']); lib=base/'lib/python3.11'; own=Path(v['path'])/'lib/python3.11/site-packages'
    require(st['paths']==[str(base/'lib/python311.zip'),str(lib),str(lib/'lib-dynload'),str(own)], 'wrong startup path contract')
    require(st['trusted_roots']==[str(lib)] and st['base_site']==str(lib/'site-packages'), 'wrong trusted/base-site roots')
    require(st['absent']==[str(base/'pyvenv.cfg'),str(base/'bin/pyvenv.cfg'),str(base/'lib/python311.zip'),str(Path(v['path'])/'bin/pyvenv.cfg')], 'wrong prefix/zip candidate inventory')
    require(st['customization_dirs']==[str(lib),str(lib/'lib-dynload'),str(lib/'site-packages')], 'wrong customization directory inventory')
    require(b['site_source']==str(lib/'site.py') and b['executable']==str(base/'bin/python3.11'), 'wrong base source/executable path')
    require(type(st['pth_inventory']) is list and st['pth_inventory']==sorted(set(st['pth_inventory'])) and all(type(n) is str and Path(n).name==n and n.endswith('.pth') for n in st['pth_inventory']), 'bad pth inventory')
    for field in ('base_site_pins','helper_sources'):
        require(type(st[field]) is dict, 'bad source inventory')
        for p,h in st[field].items():
            require(valid_hash(h),'bad source pin')
            if field=='helper_sources':
                canonical(p); require(Path(p).is_relative_to(lib) and not Path(p).is_relative_to(lib/'site-packages'), 'foreign helper source')
            else: require(not Path(p).is_absolute() and '..' not in Path(p).parts, 'bad base-site relative file')
    require(set(st['pth_inventory']) <= set(st['base_site_pins']), 'unhashed base pth')
    keys(s['stages'],' '.join(STAGES),'stages')
    for stage,value in s['stages'].items():
        if value is not None: keys(value,'identity paths site modules meta_path path_hooks path_importers','stage '+stage)
    env=s['environment']; require(type(env) is dict and all(type(k) is str and type(v) is str and '\0' not in k+v and '=' not in k for k,v in env.items()),'bad environment')
    require(env.get('PATH')=='/usr/bin:/bin' and all(env.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS')), 'explicit PATH/one-thread caps required')
    require(not any(k in env for k in ('PYTHONPATH','PYTHONHOME','PYTHONSTARTUP','PYTHONUSERBASE')), 'forbidden Python path environment')
    for field in ('timeout_seconds','cleanup_seconds'):
        require(type(s[field]) in (int,float) and 0<s[field]<=604800,'bad timeout')
    require(s['cleanup_seconds']<=30 and (s['purpose']!='startup-only' or s['timeout_seconds']<=120),'unbounded startup/cleanup')
    keys(s['evidence'],'parent child context','evidence')
    for k,p in s['evidence'].items():
        if k=='context' and s['purpose']!='context': require(p is None,'cross-purpose context output')
        else: canonical(p)
    payload=s['payload']
    if s['purpose']=='startup-only':
        keys(payload,'','startup payload')
        require(all(s['stages'][n] is None for n in ('append','exporter','dispatch','context_done')), 'startup cross-purpose stage')
    else:
        require(st['helper_sources'] and str(lib/'typing.py') in st['helper_sources'] and all(s['stages'][n] is not None for n in STAGES[:5]), 'unreviewed helper closure/stage pins; operational release unavailable')
        if s['purpose']=='context':
            keys(payload,'paths','context payload'); p=payload['paths']
            keys(p,'baseline data archive canonical openfhe openfhe_python native git','context paths')
            for k,value in p.items():
                if k not in ('native','git'): canonical(value)
            keys(p['native'],'reboot openfhe_python openfhe_libraries','native')
            require(type(p['native']['openfhe_libraries']) is list and len(p['native']['openfhe_libraries'])>=3,'native libraries')
            for value in (p['native']['reboot'],p['native']['openfhe_python'],*p['native']['openfhe_libraries']): canonical(value)
            keys(p['git'],'executable sha256','git'); canonical(p['git']['executable']); require(valid_hash(p['git']['sha256']),'git digest')
        else:
            keys(payload,'approval approval_sha256 mode out','capture payload'); canonical(payload['approval']); canonical(payload['out'])
            require(valid_hash(payload['approval_sha256']) and payload['mode']=='compatibility','capture mode/hash')
            require(s['stages']['context_done'] is None,'capture context_done stage')
    outputs=[p for p in s['evidence'].values() if p is not None]+([payload['out']] if s['purpose']=='capture' else [])
    require(len(outputs)==len(set(outputs)),'duplicate output/evidence paths')
    inputs=[x['path'] for x in s['programs'].values()]+[s['exporter']['path'], b['executable'],b['site_source'],str(Path(v['path'])/'pyvenv.cfg')]
    require(not set(outputs)&set(inputs),'output/input alias')
    protected=[v['path'],b['prefix'],s['package_root']]
    if s['purpose']=='context': protected += [value for key,value in payload['paths'].items() if key not in ('native','git')]
    require(not any(Path(output).is_relative_to(Path(root)) for output in outputs for root in protected),'output inside protected input tree')
    if s['purpose']=='capture': require(not any(Path(output).is_relative_to(Path(payload['out'])) for output in s['evidence'].values() if output is not None),'evidence would change captured artifact tree')

def load_release(path, digest):
    raw=verify_file(path,digest)
    s=strict_json(raw); validate_schema(s)
    # Transport metadata is implementation-owned and never accepted from JSON.
    s['_transport']={'path':str(path),'sha256':digest}
    return s

def validate_static(s):
    clean={k:v for k,v in s.items() if k!='_transport'}; validate_schema(clean)
    b=s['base']; v=s['venv']; root=canonical(v['path'])
    root_metadata=root.lstat()
    require(stat.S_ISDIR(root_metadata.st_mode) and root_metadata.st_uid==v['uid'] and stat.S_IMODE(root_metadata.st_mode)==v['root_mode'],'wrong venv root owner/kind/mode')
    for item in s['programs'].values(): verify_file(item['path'],item['sha256'])
    verify_file(s['exporter']['path'],s['exporter']['sha256'])
    verify_file(b['executable'],b['sha256'],64*1024*1024,b['uid'])
    verify_file(b['site_source'],b['site_sha256'],1024*1024,b['uid'])
    seen=set(); files=[]
    for rel in ['']+sorted(DIRS,key=lambda p:(p.count('/'),p)):
        if rel: require(rel in seen,'missing directory: '+rel)
        folder=root/rel; meta=folder.lstat(); require(stat.S_ISDIR(meta.st_mode),'directory kind changed')
        with os.scandir(folder) as entries:
            for entry in entries:
                name=str(Path(entry.path).relative_to(root)); meta=entry.stat(follow_symlinks=False)
                require(name in v['layout'] and name not in seen,'unexpected layout entry: '+name)
                e=v['layout'][name]; seen.add(name)
                actual='directory' if stat.S_ISDIR(meta.st_mode) else 'file' if stat.S_ISREG(meta.st_mode) else 'symlink' if stat.S_ISLNK(meta.st_mode) else 'special'
                require(actual==e['kind'] and meta.st_uid==e['uid'] and meta.st_mode&0o777==e['mode'],'wrong layout kind/owner/mode: '+name)
                if actual=='file':
                    require(meta.st_size==e['bytes'] and meta.st_size<=65536,'wrong/oversized layout file'); files.append((root/name,e))
                elif actual=='symlink':
                    require(os.readlink(root/name)==e['target'] and str((root/name).resolve())==e['resolved'],'retargeted link')
    require(seen==set(v['layout']),'incomplete layout')
    for p,e in files: verify_file(p,e['sha256'],65536,v['uid'])
    cfgraw=verify_file(root/'pyvenv.cfg',v['config_sha256'],65536,v['uid'])
    require(cfgraw.hex()==v['config_hex'],'config bytes differ')
    cfg={}
    for line in cfgraw.decode('utf-8').splitlines():
        k,sep,value=line.partition('='); k=k.strip(); require(bool(sep) and k not in cfg,'malformed config'); cfg[k]=value.strip()
    require(cfg.get('include-system-site-packages')=='false' and cfg.get('home')==str(Path(b['prefix'])/'bin') and cfg.get('executable')==b['executable'] and cfg.get('version')==b['version'],'wrong venv config/system site')
    require('--without-pip' in cfg.get('command','') and v['path'] in cfg.get('command',''),'wrong config creation command')
    require(os.readlink(root/'lib64')=='lib','wrong lib64')
    for n in ('python','python3','python3.11'): require(str((root/'bin'/n).resolve())==b['executable'],'wrong interpreter resolution')
    st=s['startup']
    for p in st['absent']: require(not os.path.lexists(p),'unexpected prefix/zip candidate: '+p)
    for folder in st['customization_dirs']:
        p=canonical(folder); require(p.is_dir(),'missing startup folder')
        for name in ('sitecustomize','usercustomize'):
            for suffix in ('','.py','.pyc','.pyo'):
                require(not os.path.lexists(p/(name+suffix)),'startup customization present')
            require(not list(p.glob(name+'*.so')) and not list(p.glob(name+'*.pyd')),'startup extension customization present')
    base_site=canonical(st['base_site'])
    require(sorted(p.name for p in base_site.glob('*.pth'))==st['pth_inventory'],'base pth inventory changed')
    for name,h in st['base_site_pins'].items(): verify_file(base_site/name,h,1024*1024,b['uid'])
    for p,h in st['helper_sources'].items(): verify_file(p,h,4*1024*1024,b['uid'])
    for p in s['evidence'].values():
        if p is not None: require(not os.path.lexists(p),'evidence exists')
    if s['purpose']=='capture':
        verify_file(s['payload']['approval'],s['payload']['approval_sha256'])
        require(not os.path.lexists(s['payload']['out']),'capture output exists')
    require(canonical(s['cwd']).is_dir(),'cwd missing')
    return {'static':'passed','assumption':'no concurrent writer; not hostile-filesystem race containment'}

def build_child_command(s):
    t=s['_transport']
    return [s['interpreter'],'-I','-B',s['programs']['launcher']['path'],'--release',t['path'],'--release-sha256',t['sha256']]

def trusted_path(s,path):
    try:
        p=canonical(path)
    except (LaunchError,OSError): return False
    return any(p.is_relative_to(Path(root)) for root in s['startup']['trusted_roots']) and not p.is_relative_to(Path(s['startup']['base_site']))

def class_fact(obj):
    cls=obj if isinstance(obj,type) else type(obj)
    return {'module':cls.__module__,'qualname':cls.__qualname__}

def importer_facts(s):
    expected=[importlib.machinery.BuiltinImporter,importlib.machinery.FrozenImporter,importlib.machinery.PathFinder]
    require(sys.meta_path==expected,'foreign meta finder')
    hooks=[]
    import zipimport
    for hook in sys.path_hooks:
        if hook is zipimport.zipimporter: hooks.append({'kind':'zipimport.zipimporter'})
        else:
            require(isinstance(hook,types.FunctionType) and hook.__module__=='_frozen_importlib_external' and hook.__qualname__=='FileFinder.path_hook.<locals>.path_hook_for_FileFinder' and hook.__code__.co_filename=='<frozen importlib._bootstrap_external>', 'foreign path hook')
            closure=hook.__closure__ or (); cells=[c.cell_contents for c in closure]
            require(importlib.machinery.FileFinder in cells,'path hook class changed')
            details=[x for x in cells if type(x) is tuple]
            expected_details=((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES))
            require(len(details)==1 and details[0]==expected_details,'path hook loader details changed')
            hooks.append({'kind':'frozen FileFinder.path_hook'})
    require(hooks==[{'kind':'zipimport.zipimporter'},{'kind':'frozen FileFinder.path_hook'}],'unexpected path hook sequence')
    importers={}
    permitted=set(s['startup']['paths'])|{s['package_root']}
    for p, finder in sorted(sys.path_importer_cache.items()):
        require(type(p) is str,'foreign importer cache path: '+str(p))
        if p == s['programs']['launcher']['path']:
            require(finder is None,'nonnegative launcher importer cache entry')
        else:
            require(trusted_path(s,p) or p in permitted,'foreign importer cache path: '+str(p))
        require(finder is None or type(finder) is importlib.machinery.FileFinder,'foreign cached finder')
        if finder is not None:
            require(finder.path==p, 'cached finder path differs')
            expected_loaders=[(suffix,loader) for loader,suffixes in ((importlib.machinery.ExtensionFileLoader,importlib.machinery.EXTENSION_SUFFIXES),(importlib.machinery.SourceFileLoader,importlib.machinery.SOURCE_SUFFIXES),(importlib.machinery.SourcelessFileLoader,importlib.machinery.BYTECODE_SUFFIXES)) for suffix in suffixes]
            require(finder._loaders==expected_loaders,'cached finder loader details changed')
        importers[p]=None if finder is None else class_fact(finder)
    return [class_fact(x) for x in expected],hooks,importers

def module_facts(s):
    result={}
    allowed={s['programs']['launcher']['path'],s['exporter']['path']}
    for name,module in sorted(sys.modules.items()):
        if name in ('typing.io','typing.re'):
            parent=sys.modules.get('typing'); parent_spec=getattr(parent,'__spec__',None)
            parent_file=getattr(parent,'__file__',None)
            require(isinstance(parent,types.ModuleType) and 'typing' in result and parent_spec is not None and type(parent_spec.loader) is importlib.machinery.SourceFileLoader and parent_spec.origin==parent_file and type(parent_file) is str and Path(parent_file).name=='typing.py' and trusted_path(s,parent_file),'typing alias lacks trusted source parent')
            require(module is getattr(parent,name.split('.')[1],None) and type(module) is getattr(parent,'_DeprecatedType',None) and getattr(module,'__module__',None)=='typing' and getattr(module,'__name__',None)==name,'replaced/foreign typing alias')
            result[name]={'origin':'trusted-alias:'+name,'file':parent_file,'loader':class_fact(type(module)),'search':None}
            continue
        require(isinstance(module,types.ModuleType),'nonmodule sys.modules entry: '+name)
        spec=getattr(module,'__spec__',None); origin=getattr(spec,'origin',None); file=getattr(module,'__file__',None)
        if name=='__main__':
            require(file==s['programs']['launcher']['path'],'wrong main source'); origin=file
        elif name==EXPORTER_NAME:
            require(file==s['exporter']['path'] and origin==file,'wrong exporter module')
        else:
            require(spec is not None,'module without spec: '+name)
            if origin not in ('built-in','frozen'):
                require(type(origin) is str and trusted_path(s,origin),'foreign/namespace module: '+name)
            if file is not None: require(trusted_path(s,file),'foreign module file: '+name)
        loader=getattr(spec,'loader',None) if spec else getattr(module,'__loader__',None)
        supported_loader=(loader is importlib.machinery.BuiltinImporter or
                          loader is importlib.machinery.FrozenImporter or
                          type(loader) is importlib.machinery.SourceFileLoader or
                          type(loader) is importlib.machinery.SourcelessFileLoader or
                          type(loader) is importlib.machinery.ExtensionFileLoader or
                          loader is None and name in ('__main__',EXPORTER_NAME))
        require(supported_loader,'foreign module loader: '+name)
        loader_fact=None if loader is None else class_fact(loader)
        locations=getattr(spec,'submodule_search_locations',None) if spec else None
        if locations is not None:
            require(all(trusted_path(s,p) for p in locations),'foreign module search path')
        result[name]={'origin':origin,'file':file,'loader':loader_fact,'search':None if locations is None else list(locations)}
    return result

def check_identity(s):
    v=s['venv']['path']; b=s['base']['prefix']
    identity={'executable':sys.executable,'resolved_executable':str(Path(sys.executable).resolve()),'prefix':sys.prefix,'exec_prefix':sys.exec_prefix,'base_prefix':sys.base_prefix,'base_exec_prefix':sys.base_exec_prefix,'version':sys.version,'isolated':sys.flags.isolated,'no_user_site':sys.flags.no_user_site,'no_site':sys.flags.no_site,'dont_write_bytecode':sys.dont_write_bytecode}
    require(sys.prefix==sys.exec_prefix==v and sys.base_prefix==sys.base_exec_prefix==b,'wrong actual prefixes')
    require(sys.executable==s['interpreter'] and identity['resolved_executable']==s['base']['executable'],'wrong actual executable')
    require(sys.flags.isolated==1 and sys.flags.no_user_site==1 and sys.flags.no_site==0 and sys.dont_write_bytecode is True,'wrong actual flags')
    require(dict(os.environ)==s['environment'] and str(Path.cwd())==s['cwd'],'actual environment/cwd differs')
    return identity

def match_stage(s,stage,snapshot):
    expected=s['stages'][stage]
    if expected is not None: require(snapshot==expected,'stage identity mismatch: '+stage)
    else: require(s['purpose']=='startup-only' or stage=='context_done','unreviewed stage')
    return snapshot

def check_stage(s,stage):
    require(stage in STAGES,'unknown stage')
    import site
    identity=check_identity(s); v=s['venv']['path']
    paths=s['startup']['paths']+([s['package_root']] if stage in ('append','exporter','dispatch','context_done') else [])
    require(sys.path==paths,'wrong actual startup/path append state')
    own=Path(v)/'lib/python3.11/site-packages'
    require(own.is_dir() and not own.is_symlink() and not list(own.iterdir()),'own site contaminated')
    require(site.ENABLE_USER_SITE is False and site.getsitepackages()==[str(own)],'wrong site state')
    modules=module_facts(s); meta,hooks,importers=importer_facts(s)
    snapshot={'identity':identity,'paths':list(sys.path),'site':{'ENABLE_USER_SITE':site.ENABLE_USER_SITE,'site_packages':site.getsitepackages(),'PREFIXES':list(site.PREFIXES)},'modules':modules,'meta_path':meta,'path_hooks':hooks,'path_importers':importers}
    return match_stage(s,stage,snapshot)

def preload_stdlib():
    for name in HELPERS: importlib.import_module(name)
    import argparse, csv, email.parser, io, platform
    # Exercise the concrete lazy metadata parser and RECORD consumer in memory.
    class FictionalDistribution(importlib.metadata.Distribution):
        def read_text(self,filename):
            return {'METADATA':'Name: task54-fictional\nVersion: 0\nSummary: trusted helper preparation\n','RECORD':'fictional.py,sha256=AAAA,1\n'}.get(filename)
        def locate_file(self,path): return Path('/nonexistent-task54-fictional')/path
    d=FictionalDistribution(); require(d.metadata['Name']=='task54-fictional' and d.version=='0' and len(d.files)==1,'metadata helper warmup failed')
    email.parser.Parser().parsestr('Subject: =?utf-8?q?fixture?=\n\nfixture')
    list(csv.reader(io.StringIO('a,b\n')))
    parser=argparse.ArgumentParser(add_help=True); parser.add_argument('--fixture'); parser.parse_args([]); parser.format_help()
    platform.platform(); platform.uname(); Path('/').resolve(); list(Path('/nonexistent-task54-fictional').glob('*.py'))
    return {'helpers':list(HELPERS),'warmup':'fictional metadata, RECORD, parser, platform, pathlib; not Linux closure proof'}

def append_package_root(s):
    root=str(canonical(s['package_root'])); require(Path(root).is_dir() and root not in sys.path,'external package root missing/already appended')
    sys.path.append(root)  # Never site.addsitedir/main: .pth files remain inert.

def load_exporter_exact(path,digest):
    require(EXPORTER_NAME not in sys.modules,'exporter already loaded')
    raw=verify_file(path,digest)
    code=compile(raw,path,'exec',dont_inherit=True)
    module=types.ModuleType(EXPORTER_NAME); module.__file__=path; module.__package__=''
    module.__spec__=importlib.machinery.ModuleSpec(EXPORTER_NAME,loader=None,origin=path)
    sys.modules[EXPORTER_NAME]=module
    try: exec(code,module.__dict__)
    except BaseException:
        sys.modules.pop(EXPORTER_NAME,None); raise
    return module

def write_new(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with open(path,'xb') as stream: stream.write(raw)

def dispatch(s,record=None):
    if record is None: record={}
    record['purpose']=s['purpose']; record['stages']={}
    record['stages']['startup']=check_stage(s,'startup')
    for p,h in s['startup']['helper_sources'].items(): verify_file(p,h,4*1024*1024,s['base']['uid'])
    record['preparation']=preload_stdlib()
    record['stages']['helpers']=check_stage(s,'helpers')
    if s['purpose']=='startup-only':
        record['status']='observation_only_not_runtime_approval'; return 0
    require(s['startup']['helper_sources'] and all(s['stages'][n] is not None for n in STAGES[:5]),'unreviewed helper/stage prerequisites')
    append_package_root(s); record['stages']['append']=check_stage(s,'append')
    exporter=load_exporter_exact(s['exporter']['path'],s['exporter']['sha256'])
    record['stages']['exporter']=check_stage(s,'exporter')
    record['stages']['dispatch']=check_stage(s,'dispatch')
    if s['purpose']=='context':
        result=exporter.observe_context(s['payload']['paths'])
        record['stages']['context_done']=check_stage(s,'context_done')
        write_new(s['evidence']['context'],result); record['status']='context_observed_not_capture_approval'; return 0
    require(s['purpose']=='capture','unsupported dispatch')
    p=s['payload']; argv=['--approval',p['approval'],'--approval-sha256',p['approval_sha256'],'--mode','compatibility','--out',p['out']]
    record['exporter_argv']=argv
    code=exporter.main(argv)
    require(type(code) is int and code==0,'exporter main failed')
    record['status']='exporter_main_returned_zero_not_scientific_claim'; return code

def main(argv=None):
    record={'kind':'task54_child_operational_evidence','status':'failed'}; spec=None
    try:
        path,h=parse_transport(sys.argv[1:] if argv is None else argv); spec=load_release(path,h)
        require(str(Path(__file__).resolve())==spec['programs']['launcher']['path'],'wrong launcher identity')
        verify_file(__file__,spec['programs']['launcher']['sha256'])
        record['release']={'path':path,'sha256':h}; record['actual_argv']=list(sys.orig_argv)
        require(record['actual_argv']==build_child_command(spec),'actual child argv differs from derived transport')
        return dispatch(spec,record)
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc)); print(json.dumps(record),file=sys.stderr); return 1
    finally:
        if spec is not None:
            write_new(spec['evidence']['child'],record)

if __name__=='__main__':
    raise SystemExit(main())
