"""Fixed read-only prerequisites. Raw stdout is ignored-ledger audit input only.

Never publish the stdlib source bodies in tracked/public archives. No target
imports, subprocesses, package/native content inventory or context authority.
Trusted system bootstrap and no concurrent writers are explicit assumptions.
"""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

MIB=1024*1024
OWNER=1000
BASE=Path('/home/kmihara/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu')
SOURCE_ROOT=BASE/'lib/python3.11'
PACKAGE=Path('/home/kmihara/reboot/venv/lib/python3.11/site-packages')
BASELINE=Path('/home/kmihara/reboot/src/ReBoot')
NATIVE_LIB=Path('/home/kmihara/reboot/install/lib')
ARCHIVE=Path('/home/kmihara/hekit-runs/out/R-REB-5/reboot_emlp23_full')
CANONICAL=Path('/home/kmihara/hekit-runs/out/R-REB-2/reboot_emlp1_full/data')
RAW=Path('/home/kmihara/reboot/data/MNIST/raw')
CHECKOUTS=(Path('/home/kmihara/hekit-exporter-f021616'),BASELINE,
           Path('/home/kmihara/reboot/src/openfhe-development'),
           Path('/home/kmihara/reboot/src/openfhe-python'))
SYSTEM=Path('/usr/bin/python3.14')
SYSTEM_PIN='52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41'
BASE_PIN='2874a0b9344d06b7767aebb1e6e25a759ffcbdb544e99400ecc74dc6092d1174'
UNAME=Path('/usr/lib/cargo/bin/coreutils/uname')
UNAME_PIN='48893b0fb21436b54619db80486e83ef39dfccaf1aefe83dfa00c02d6146e8c0'
EXPORTER_PIN='0d8eaff394f2c0fa53c2cdd4f6cb130e4255f80c0f9ed13ff185b1da2ec733ce'
ARCHIVE_PIN='76c493e28692531d07d54337dfcf89a21d8d78fa2979e02b20dccad47fb303c5'
SOURCES=('importlib/metadata/__init__.py','importlib/metadata/_adapters.py',
 'importlib/metadata/_text.py','email/parser.py','email/feedparser.py',
 'email/message.py','email/header.py','email/charset.py','email/policy.py',
 'email/_policybase.py','csv.py','encodings/__init__.py','encodings/aliases.py',
 'codecs.py','argparse.py','gettext.py','locale.py','copy.py','shutil.py',
 'textwrap.py','pathlib.py','zipfile.py','typing.py')
SOURCE_PINS={'importlib/metadata/__init__.py': 'bc6d98cd94459ad3b6ea5cb75cde38383647655054f065b897176f7635d2d8bc', 'importlib/metadata/_adapters.py': '07a7c28b9fbc98b543154663de4ac8e67028fa62a9d5d1ffa886afc88c85ac9b', 'importlib/metadata/_text.py': '1c2b0592c66924b7933f734493f9e0ac079755146d4ebb7287d78e001a113f80', 'email/parser.py': '7a665bc106e9d4f6f8f2c8de53b2c1d5b97f5105cc7da638b82d731f6f9e354e', 'email/feedparser.py': 'ca333e0462c96505d8c510f260192256306fa889c011c6f2eaa8d50c70503cce', 'email/message.py': '275f2d0a31e1331cd90014039c4c150fe5822c3e981638cc31d4973ee0fb27eb', 'email/header.py': '4d9baa908ad5288dd8fad8cf20b3802ffac77ba1642727804a633b201c56e5ca', 'email/charset.py': '3ccb2e860390656664d271f08da8c0bae9a4297cc7e90be664d7c82d364e442f', 'email/policy.py': '56a2a90d973c06668f6f113d67f26cf348af10b33eece2a6b812e34da45258d8', 'email/_policybase.py': '3db4bf3be5bdba13ab9a78ce30784c330d6dbc657b4e9142210dc8b264cfd424', 'csv.py': '05c688620a98076289b09943788f840175cac8d299956e33fe9b4502a6526525', 'encodings/__init__.py': '78c4744d407690f321565488710b5aaf6486b5afa8d185637aa1e7633ab59cd8', 'encodings/aliases.py': '6fdcc49ba23a0203ae6cf28e608f8e6297d7c4d77d52e651db3cb49b9564c6d2', 'argparse.py': 'e878677e158d0bc0fb5119e0df74d94236f3653777bc9f2ea3fee49c167fe001', 'gettext.py': 'a5c249a522b6b8e3aa6f1b12a8bcc09508b99ad612b58d2fc973db27ea3b7cc3', 'locale.py': 'ac828df41b2fac93ba3e66a2735216c98c0e1853b8dfb9b859e301fe64153feb', 'pathlib.py': '3b21ac188fd9ac20bacc8ccd976fabd5d8ccf5000f8e157d3dfff005b228fb3d', 'typing.py': 'eff3df4f1d65e13d690b8f427cada4dd3b88431dd1e7f319fda1bd0062113203'}
PRIOR={'R12': {'ledger_path': 'task-54-linux-helper-static-20261107.stdout', 'sha256': '2dc5e1c90a9df0b078b2a7f7cb73f38ceef165860ba8e9c82649daac04235b12'}, 'R13': {'ledger_path': 'task-54-linux-launch-extra-static-20261107.stdout', 'sha256': '85a430e3bc0a9e72231b396d5456419e2846167b16948cef279d51dd01829fdc'}, 'R14': {'ledger_path': 'task-54-linux-startup-prerequisite-fix1-20261107.stdout', 'sha256': '1eee1546a55ae482aaa27f85d87dff4d7c92e12c640e84c9d8707d333ac3ef9a'}, 'R15': {'ledger_path': 'task-54-linux-startup-command-identity-20261107.stdout', 'sha256': '94ca5a2c95c8b83d7e0531ae70e2d25baef474ba3e520804b3a4bf0e8d94c6be'}, 'startup2_child': {'ledger_path': 'startup2-child.json', 'sha256': 'a1fbe98aa390b7305f7189dabf574c4d28b2422617bfeb010e261f2c82fd1f41'}, 'root_ruling': {'ledger_path': 'task-54-linux-after-startup2-root-ruling.md', 'sha256': '40029329ba296da3762b0ff877618c89858fe8e9d9b423ff7be7d607cd920530'}, 'source_release': {'ledger_path': 'task-54-linux-context-prerequisite-observer-root-release.md', 'sha256': '0f7a636f4dd696ba5ae20a71788826010f93c5847fd61fe36fb52458cc2e7932'}}
SOURCE_CAP=256*1024
BUDGETS={'source':2*MIB,'metadata':16*MIB,'small':24*MIB,'self':16*MIB}
OUTPUT_CAP=8*MIB
ROW_CAP=65536
LINE_CAP=8192
NPY_NAMES=('train_images_14x14.npy','train_labels.npy','test_images_14x14.npy','test_labels.npy')
RAW_NAMES=tuple(stem+suffix for stem in ('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte','train-images-idx3-ubyte','train-labels-idx1-ubyte') for suffix in ('','.gz'))
LAYERS={'emlp2':('linear_fwd_00','linear_lrn_00','linear_01','linear_02'),
        'emlp3':('linear_fwd_00','linear_lrn_00','linear_fwd_01','linear_lrn_01','linear_02','linear_03')}
REQUIRED={'numpy','torch','torchvision','pybind11','pyyaml','scikit-learn','imbalanced-learn','ucimlrepo','wandb','pandas','matplotlib','seaborn'}

class ObservationError(ValueError):pass
class LimitError(ObservationError):pass
def need(ok,why):
    if not ok:raise ObservationError(why)
def bound(ok,why):
    if not ok:raise LimitError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def verify_pin(raw,pin):need(sha(raw)==pin,'previously pinned source changed')
def stat_values(s):
    return {'uid':s.st_uid,'mode':stat.S_IMODE(s.st_mode),'device':s.st_dev,
            'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
def identity(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns
def metadata(path):
    p=Path(path);r={'path':str(p),'sha256':None,'read_scope':'metadata_only'}
    try:
        s=p.lstat();r['stat']=stat_values(s)
        r['kind']='symlink' if stat.S_ISLNK(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'special'
        if stat.S_ISLNK(s.st_mode):
            link=os.readlink(p);bound(len(os.fsencode(link))<=4096,'symlink text cap');r['link']=link
        resolved=p.resolve();r['resolved']=str(resolved)
        if resolved!=p:r['resolved_stat']=stat_values(resolved.stat())
        r['status']='observed'
    except FileNotFoundError:r['status']='missing_or_dangling'
    except OSError as e:r.update(status='metadata_error',error=type(e).__name__)
    return r

def content_policy(p):
    """The entire content-read whitelist. No path obtained from RECORD is allowed."""
    p=Path(p)
    need(p.is_absolute() and '..' not in p.parts,'nonabsolute/traversal content path')
    if p in {SOURCE_ROOT/n for n in SOURCES}:return 'source',SOURCE_CAP,OWNER,False
    if p==SYSTEM:return 'self',16*MIB,0,False
    if p==Path('/usr/bin/git'):return 'small',16*MIB,0,False
    if p==ARCHIVE/'config_values.json':return 'small',2*MIB,OWNER,False
    if p in {CANONICAL/n for n in NPY_NAMES}:return 'small',256,OWNER,True
    for root in CHECKOUTS:
        if p==root/'.git/HEAD':return 'small',4096,OWNER,False
        if p==root/'.git/packed-refs':return 'small',MIB,OWNER,False
        try:parts=p.relative_to(root/'.git/refs/heads').parts
        except ValueError:parts=()
        if parts and all(part not in ('','.','..') and '\0' not in part for part in parts):return 'small',4096,OWNER,False
    if p==CHECKOUTS[0]/'tools/export_reboot_block_seed_inputs.py':return 'small',MIB,OWNER,False
    try:parts=p.relative_to(PACKAGE).parts
    except ValueError:parts=()
    if len(parts)==1 and parts[0].endswith('.egg-info'):return 'metadata',512*1024,OWNER,False
    if len(parts)==2:
        folder,name=parts
        if folder.endswith('.dist-info') and name in ('METADATA','RECORD','WHEEL'):
            return 'metadata',{'METADATA':512*1024,'RECORD':2*MIB,'WHEEL':16384}[name],OWNER,False
        if folder.endswith('.egg-info') and name in ('PKG-INFO','SOURCES.txt','installed-files.txt'):
            return 'metadata',512*1024 if name=='PKG-INFO' else 2*MIB,OWNER,False
    raise ObservationError('content path outside fixed whitelist')

class Reader:
    def __init__(self):
        self.used={key:0 for key in BUDGETS};self.deadline=time.monotonic()+60
        self.last=None;self.failure=None
    def tick(self):
        need(self.failure is None,'previous optional candidate read refused or failed')
        bound(time.monotonic()<self.deadline,'internal observation deadline')
    def read(self,path):
        self.tick();p=Path(path);group,cap,uid,prefix=content_policy(p)
        need(p.resolve()==p,'noncanonical content path')
        initial=p.lstat();need(stat.S_ISREG(initial.st_mode) and initial.st_uid==uid,'content kind/owner')
        bound(prefix or initial.st_size<=cap,'per-file content cap')
        request=min(initial.st_size,cap)+1 if not prefix else min(initial.st_size,cap)
        bound(self.used[group]+request<=BUDGETS[group] and sum(self.used.values())+request<=58*MIB,'aggregate read cap')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            before=os.fstat(fd)
            need(identity(before)==identity(initial) and stat.S_ISREG(before.st_mode) and before.st_uid==uid,'opened content identity changed')
            with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(request)
            self.used[group]+=len(raw)
            after=os.fstat(fd)
            need(identity(before)==identity(after),'content changed during read')
            need(len(raw)==min(before.st_size,cap) if prefix else len(raw)==before.st_size,'content length changed')
            self.last=stat_values(after)
            return raw
        finally:os.close(fd)

def optional_read(reader,path,item):
    """Retain the refused candidate before returning its partial collector output.

    The next tick stops the operation; neither a cap nor an identity failure is
    retried. Required source/self reads still propagate directly as before.
    """
    try:
        group,cap,uid,prefix=content_policy(path)
        item.update(read_cap=cap,read_scope='prefix_only' if prefix else group+'_content',sha256=None)
        raw=reader.read(path);item['stat']=reader.last;return raw
    except Exception as exc:
        item.update(status='read_refused_or_failed',sha256=None,
                    error={'type':type(exc).__name__,'message':str(exc)[:1024]})
        reader.failure=item
        return None

def shallow(root,cap,reader):
    reader.tick();need(root.resolve()==root,'noncanonical directory')
    entries=[]
    with os.scandir(root) as it:
        for entry in it:
            reader.tick();bound(len(entries)<cap,'shallow enumeration cap')
            entries.append((entry.name,entry.stat(follow_symlinks=False).st_mode))
    return sorted(entries)

def metadata_summary(raw,name):
    try:text=raw.decode('utf-8')
    except UnicodeError:return {'format':name,'utf8':False,'ascii':False,'status':'unsupported_encoding'}
    lines=text.splitlines();bound(all(len(x.encode())<=LINE_CAP for x in lines),'metadata physical line cap')
    result={'format':name,'utf8':True,'ascii':raw.isascii(),'status':'parsed_forms',
            'crlf':b'\r\n' in raw,'bare_lf':b'\n' in raw.replace(b'\r\n',b''),'non_ascii':not raw.isascii()}
    if name=='RECORD':
        old=csv.field_size_limit(LINE_CAP);hist={};algs=set();counts={'rows':0,'empty_hash':0,'empty_size':0,'absolute_paths':0,'parent_paths':0,'embedded_newline_rows':0}
        try:
            for row in csv.reader(io.StringIO(text,newline=''),strict=True):
                counts['rows']+=1;bound(counts['rows']<=ROW_CAP,'metadata row cap')
                bound(sum(len(x.encode()) for x in row)<=LINE_CAP,'metadata logical row cap')
                hist[str(len(row))]=hist.get(str(len(row)),0)+1
                if len(row)!=3:continue
                path,h,size=row;counts['empty_hash']+=not h;counts['empty_size']+=not size
                counts['absolute_paths']+=path.startswith('/');counts['parent_paths']+='..' in path.split('/')
                counts['embedded_newline_rows']+=any('\n' in x or '\r' in x for x in row)
                if h:
                    alg=h.split('=',1)[0];bound(len(alg)<=128 and len(algs)<128 or alg in algs,'RECORD algorithm cap');algs.add(alg)
        except csv.Error as e:raise ObservationError('malformed CSV RECORD') from e
        finally:csv.field_size_limit(old)
        result.update(counts,column_counts=hist,hash_algorithms=sorted(algs),quoted_fields='"' in text);return result
    if name in ('SOURCES.txt','installed-files.txt'):
        bound(len(lines)<=ROW_CAP,'metadata list row cap');result.update(rows=len(lines),absolute_paths=sum(x.startswith('/') for x in lines),parent_paths=sum('..' in x.split('/') for x in lines));return result
    headers={};current=None;folded=0;separator=False
    for line_number,line in enumerate(lines,1):
        if not line:separator=True;break
        if line[:1] in (' ','\t'):
            folded+=1
            if current is not None:headers[current][-1]+=' '+line.strip()
        else:
            key,sep,value=line.partition(':')
            if not sep:result['status']='malformed_header';continue
            need(bool(re.fullmatch(r'[A-Za-z0-9-]+',key)),'invalid header name')
            current=key.lower();headers.setdefault(current,[]).append(value.strip())
        bound(len(headers)<=512,'header name cap')
        if current is not None:
            logical_bytes=len(headers[current][-1].encode())
            if logical_bytes>LINE_CAP:
                exc=LimitError('logical header cap')
                exc.diagnostic={'header_name':current,'logical_bytes':logical_bytes,'cap':LINE_CAP,'line':line_number}
                raise exc
    header_text='\n'.join(v for values in headers.values() for v in values)
    encodings=set(re.findall(r'=\?([^?\s]+)\?',header_text))
    encodings.update(re.findall(r'charset\s*=\s*["\x27]?([A-Za-z0-9_.-]+)',header_text,re.I))
    bound(len(encodings)<=128 and all(len(x.encode())<=128 for x in encodings),'encoding label cap')
    result.update(header_names=sorted(headers),header_count=sum(map(len,headers.values())),folded_headers=folded,separator=separator,encoding_labels=sorted(encodings),encoded_words='=?' in header_text)
    if name=='WHEEL':
        result['wheel']={k:headers.get(k,[]) for k in ('wheel-version','root-is-purelib','tag')}
        bound(all(len(v.encode())<=256 for values in result['wheel'].values() for v in values),'wheel value cap')
    else:
        for key in ('name','version'):
            values=headers.get(key,[]);result[key+'_count']=len(values)
            bound(all(len(v.encode())<=256 for v in values),'name/version cap')
            result[key]=values[0].lower().replace('_','-') if key=='name' and values else values[0] if values else None
    return result

def collect_sources(reader):
    values=[]
    for name in SOURCES:
        path=SOURCE_ROOT/name;raw=reader.read(path);pin=SOURCE_PINS.get(name)
        if pin:verify_pin(raw,pin)
        values.append({'path':str(path),'resolved':str(path),'stat':reader.last,'sha256':sha(raw),'pin_expected':pin,'pin_match':True if pin else None,'source_text':raw.decode('utf-8'),'status':'observed_source_not_semantic_closure'})
    return values

def collect_metadata(reader):
    entries=shallow(PACKAGE,2048,reader);results=[];unsupported=[];names=[];encodings=set();header_names=set();count=0
    result={'root':str(PACKAGE),'scan_count':len(entries),'distribution_count':0,
            'complete':False,'observation_status':'incomplete_read',
            'unsupported_forms':unsupported,'entries':results}
    for name,mode in entries:
        reader.tick();p=PACKAGE/name
        if not name.endswith(('.dist-info','.egg-info','.egg','.zip')):continue
        count+=1;result['distribution_count']=count;bound(count<=256,'distribution candidate cap')
        if stat.S_ISLNK(mode) or name.endswith(('.egg','.zip')):
            unsupported.append(metadata(p));continue
        children=('METADATA','RECORD','WHEEL') if name.endswith('.dist-info') else ('PKG-INFO','SOURCES.txt','installed-files.txt')
        paths=[(p/n,n) for n in children] if stat.S_ISDIR(mode) else [(p,'PKG-INFO')] if stat.S_ISREG(mode) and name.endswith('.egg-info') else []
        if not paths:unsupported.append(metadata(p));continue
        for path,kind in paths:
            info=metadata(path);results.append(info)
            if info['status']!='observed':continue
            raw=optional_read(reader,path,info)
            if raw is None:return result,entries
            info.update(stat=reader.last,sha256=sha(raw),read_scope='metadata_file_only')
            try:forms=metadata_summary(raw,kind)
            except ObservationError as exc:
                info.update(status='parse_refused_or_failed',error={'type':type(exc).__name__,'message':str(exc)[:1024]})
                if hasattr(exc,'diagnostic'):info['parse_diagnostic']=exc.diagnostic
                reader.failure=info;result['observation_status']='incomplete_parse'
                return result,entries
            info['forms']=forms
            if forms.get('name'):names.append(forms['name'])
            encodings.update(forms.get('encoding_labels',[]));header_names.update(forms.get('header_names',[]))
            bound(len(encodings)<=128 and len(header_names)<=512,'global metadata labels cap')
    errors=any(x['status']=='metadata_error' for x in results+unsupported)
    missing=any(x['status']=='missing_or_dangling' for x in results)
    bad_forms=bool(unsupported) or any(x['status']=='observed' and x.get('forms',{}).get('status')!='parsed_forms' for x in results)
    outcome='metadata_error' if errors else 'unsupported_forms' if bad_forms else 'missing_files' if missing else 'filesystem_forms_complete'
    result.update(complete=outcome=='filesystem_forms_complete',observation_status=outcome,
                  missing_required=sorted(REQUIRED-set(names)),duplicate_names=sorted({n for n in names if names.count(n)>1}),encoding_labels=sorted(encodings))
    return result,entries

def head_ref(raw):
    text=raw.decode('ascii').strip()
    if re.fullmatch('[0-9a-f]{40}|[0-9a-f]{64}',text):return None,text
    need(text.startswith('ref: refs/heads/'),'unsupported HEAD format')
    ref=text[5:];parts=ref.split('/')
    need(all(x not in ('','.','..') and '\0' not in x for x in parts) and all(not c.isspace() for c in ref),'unsafe HEAD ref')
    return ref,None

def collect_checkouts(reader):
    out=[]
    for root in CHECKOUTS:
        reader.tick();item={'root':str(root),'metadata':metadata(root),'git_form':metadata(root/'.git')};out.append(item)
        g=item['git_form']
        if g.get('kind')=='directory' and g.get('resolved')==str(root/'.git'):
            hp=root/'.git/HEAD';h=metadata(hp);item['head_identity']=h
            if h['status']=='observed':
                raw=optional_read(reader,hp,h)
                if raw is None:return out
                h.update(sha256=sha(raw),read_scope='HEAD');ref,oid=head_ref(raw);h['ref']=ref
                if ref:
                    rp=root/'.git'/ref
                    if rp.exists():
                        h['ref_identity']=metadata(rp);rr=optional_read(reader,rp,h['ref_identity'])
                        if rr is None:return out
                        _,oid=head_ref(rr);need(oid is not None,'symbolic branch ref');h['ref_sha256']=sha(rr)
                    elif (root/'.git/packed-refs').exists():
                        pp=root/'.git/packed-refs';h['packed_refs_identity']=metadata(pp)
                        rr=optional_read(reader,pp,h['packed_refs_identity'])
                        if rr is None:return out
                        h['packed_refs_sha256']=sha(rr)
                        for row in rr.decode('ascii').splitlines():
                            if row.startswith(('#','^')):continue
                            fields=row.split()
                            if len(fields)==2 and fields[1]==ref:
                                need(oid is None,'duplicate packed HEAD ref');_,oid=head_ref(fields[0].encode());need(oid is not None,'bad packed object')
                h['object_id']=oid
        if root==CHECKOUTS[0]:
            p=root/'tools/export_reboot_block_seed_inputs.py';e=metadata(p);item['exporter_identity']=e
            if e['status']=='observed':
                raw=optional_read(reader,p,e)
                if raw is None:return out
                e.update(sha256=sha(raw),expected_sha256=EXPORTER_PIN,pin_match=sha(raw)==EXPORTER_PIN,read_scope='exact_exporter_hash_only')
    return out

def collect_native(reader,package_entries):
    result=[];roles=[]
    roots=((PACKAGE,package_entries,'openfhe_extension'),(BASELINE,None,'reboot_extension'),(NATIVE_LIB,None,'openfhe_library'))
    for root,entries,role in roots:
        reader.tick()
        info=metadata(root);role_info={'role':role,'root':info,'status':'no_matching_candidate','candidate_count':0};roles.append(role_info)
        if info['status']=='missing_or_dangling':role_info['status']='missing_root';continue
        if info['status']=='metadata_error':role_info['status']='metadata_error';continue
        if info.get('kind')!='directory' or info.get('resolved')!=str(root):role_info['status']='unsupported_root';continue
        if entries is None:
            try:entries=shallow(root,512 if root==BASELINE else 128,reader)
            except OSError as exc:
                role_info.update(status='metadata_error',error=type(exc).__name__);continue
        for name,mode in entries:
            pattern=r'openfhe.*(?:\.so.*|\.pyd)' if root==PACKAGE else r'reboot_cpp.*(?:\.so.*|\.pyd)' if root==BASELINE else r'libOPENFHE.*\.so.*'
            if re.fullmatch(pattern,name) and (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                bound(len(result)<32,'native candidate cap');item=metadata(root/name);item.update(role=role,sha256=None,hash_status='not_read_native_content');result.append(item)
                role_info['candidate_count']+=1
                if item['status']!='observed':role_info['status']='candidate_unavailable'
        if role_info['candidate_count'] and role_info['status']=='no_matching_candidate':role_info['status']='candidates_observed_not_native_identity'
    return {'candidates':result,'roles':roles}

def archive_files():
    return [f'init_weights_{m}_seed{s}/{layer}.npy' for m,layers in LAYERS.items() for s in range(3) for layer in layers]+[f'train_indices_{m}_seed{s}.txt' for m in LAYERS for s in range(3)]
def archive_summary(raw):
    data=json.loads(raw);need(type(data) is dict and type(data.get('models')) is dict,'archive model mapping')
    result={'model_keys':sorted(data['models']),'models':{},'expected_sha256':ARCHIVE_PIN,'actual_sha256':sha(raw),'pin_match':sha(raw)==ARCHIVE_PIN}
    allowed=set(archive_files())
    for model in LAYERS:
        item=data['models'].get(model,{ });seeds=item.get('per_seed',{});need(type(seeds) is dict,'archive seeds mapping')
        result['models'][model]={'seed_keys':sorted(seeds),'files':{}}
        for seed in ('0','1','2'):
            if seed not in seeds:continue
            e=seeds[seed];order=e['indices_file'];need(order==f'train_indices_{model}_seed{seed}.txt','foreign archive order')
            weights=[]
            for weight in e['init_weights']:
                rel=weight['file'];need(rel in allowed and rel==f'init_weights_{model}_seed{seed}/{weight["layer"]}.npy' and weight['layer'] in LAYERS[model],'foreign archive weight')
                h=weight['sha256'];need(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h),'archive weight digest')
                weights.append({'file':rel,'sha256':h})
            bound(len(weights)<=len(LAYERS[model]),'archive weight count');result['models'][model]['files'][seed]={'indices_file':order,'init_weights':weights}
    return result

def npy_prefix(raw):
    need(raw[:6]==b'\x93NUMPY' and len(raw)>=10,'NPY magic/prefix')
    major,minor=raw[6:8];need((major,minor) in ((1,0),(2,0),(3,0)),'NPY version')
    start=10 if major==1 else 12;need(len(raw)>=start,'NPY header prefix')
    size=int.from_bytes(raw[8:start],'little');out={'version':[major,minor],'header_bytes':size,'complete_header':start+size<=len(raw)}
    if not out['complete_header']:return out
    try:header=ast.literal_eval(raw[start:start+size].decode('utf-8' if major==3 else 'latin1').strip())
    except (ValueError,SyntaxError,UnicodeError) as e:raise ObservationError('invalid literal NPY header') from e
    need(type(header) is dict and set(header)=={'descr','fortran_order','shape'} and type(header['descr']) is str and type(header['fortran_order']) is bool and type(header['shape']) is tuple and all(type(n) is int and n>=0 for n in header['shape']),'unsupported NPY header values')
    out.update(dtype=header['descr'],shape=list(header['shape']),fortran_order=header['fortran_order']);return out

def collect_inputs(reader):
    items=[]
    for root,names,role in ((RAW,RAW_NAMES,'raw_mnist'),(ARCHIVE,archive_files(),'historical_artifact')):
        for name in names:reader.tick();item=metadata(root/name);item['role']=role;items.append(item)
    for name in NPY_NAMES:
        reader.tick();p=CANONICAL/name;item=metadata(p);item['role']='canonical_prefix';items.append(item)
        if item['status']=='observed':
            raw=optional_read(reader,p,item)
            if raw is None:return items,{'status':'not_reached_after_incomplete_input'}
            item.update(prefix_sha256=sha(raw),prefix_bytes=len(raw),read_scope='at_most_256_prefix_bytes',header=npy_prefix(raw))
    p=ARCHIVE/'config_values.json';summary=metadata(p)
    if summary['status']=='observed':
        raw=optional_read(reader,p,summary)
        if raw is None:return items,summary
        summary.update(archive_summary(raw));summary['read_scope']='complete_small_config_hash_and_values'
    return items,summary

def collect_commands(reader):
    out=[]
    for p,pin in ((Path('/usr/bin/git'),None),(BASE/'bin/python3.11',BASE_PIN),(Path('/usr/bin/uname'),UNAME_PIN),(UNAME,UNAME_PIN)):
        reader.tick();item=metadata(p);item.update(previous_pin=pin,hash_status='not_rehashed_prior_pin_only' if pin else 'not_read');out.append(item)
        if p==Path('/usr/bin/git') and item.get('resolved')==str(p) and item.get('kind')=='file':
            raw=optional_read(reader,p,item)
            if raw is None:return out
            item.update(sha256=sha(raw),hash_status='observed_candidate_not_approved')
    return out

def emit(record,stream):
    record['publication']='ignored_audit_ledger_only_not_public_archive'
    raw=(json.dumps(record,ensure_ascii=True,allow_nan=False,sort_keys=True)+'\n')
    if len(raw.encode())>OUTPUT_CAP:
        raw=json.dumps({'kind':'context_prerequisite_observation_not_context_or_runtime_release','status':'incomplete_observation','error':'serialized output cap exceeded; source bodies omitted, not sanitized original','publication':'ignored_audit_ledger_only_not_public_archive'})+'\n'
        stream.write(raw);stream.flush();return 1
    stream.write(raw);stream.flush();return 0 if record.get('status')=='bounded_observation_complete' else 1

PRIOR=dict(PRIOR,R24={'ledger_path':'task-54-linux-context-prerequisite-observe-FIX1.stdout','sha256':'051fe79ac9ea4aef6e5de8c8bc8d4064fc1b32e7a1009b2cc6251f4f09556781'})

def main():
    reader=Reader();r={'kind':'metadata_prerequisite_diagnostic_not_context_or_runtime_release','started_unix':time.time(),'status':'incomplete_observation','prior_evidence':PRIOR,'section_status':{},'errors':[],'limits':{'read_groups':BUDGETS,'read_total':58*MIB,'output_bytes':OUTPUT_CAP,'source_count':0,'source_per_file':SOURCE_CAP,'metadata_rows':ROW_CAP,'metadata_line_bytes':LINE_CAP,'wall_seconds':60},'scope':'Fixed metadata-only diagnostic; source and small reads omitted. R24 source bodies are dated prior review inputs, not reobserved here; normal trusted system bootstrap/no concurrent writer. No target imports, subprocess, package/native closure, scientific/context/capture or public source-body publication.'}
    try:
        need(os.getuid()==OWNER and os.nice(0)>=10,'observer uid/nice')
        need(Path(sys.executable).resolve()==SYSTEM,'observer interpreter path')
        raw=reader.read(SYSTEM);verify_pin(raw,SYSTEM_PIN)
        r['observer']={'lexical':sys.executable,'resolved':str(SYSTEM),'version':sys.version,'uid':os.getuid(),'nice':os.nice(0),'sha256':sha(raw)}
        r['metadata_inventory'],entries=collect_metadata(reader);r['section_status']['metadata']=r['metadata_inventory']['observation_status']
        reader.tick()
        reader.tick();r['status']='bounded_observation_complete'
    except Exception as exc:
        r['errors'].append({'type':type(exc).__name__,'message':str(exc)[:1024]})
        if reader.failure is not None:r['failed_candidate']=reader.failure
    r['read_bytes']=reader.used;r['finished_unix']=time.time()
    return emit(r,sys.stdout)

if __name__=='__main__':raise SystemExit(main())
