#!/usr/bin/env python3
import hashlib,json,os,pathlib,re,subprocess,tarfile,tomllib
ROOT=pathlib.Path(__file__).resolve().parent; SRC=ROOT/'source'; REPO=ROOT.parents[3]
def sha(b):return hashlib.sha256(b).hexdigest()
def save(p,x): (ROOT/p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
source=[json.loads(x) for x in (ROOT/'manifests/source.sha256.jsonl').read_text().splitlines()]
by_path={r['path']:r for r in source}
tree={}
for r in (ROOT/'logs/git-tree.stdout').read_bytes().split(b'\0'):
    if not r: continue
    h,p=r.split(b'\t',1); mode,kind,oid=h.decode().split(); tree[p.decode()]={'mode':mode,'kind':kind,'oid':oid}
assert tree.keys()==by_path.keys(),{'missing':list(tree.keys()-by_path.keys()),'extra':list(by_path.keys()-tree.keys())}
for rel,r in by_path.items():
    p=SRC/rel; b=os.readlink(p).encode() if p.is_symlink() else p.read_bytes()
    assert sha(b)==r['sha256'],rel
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==tree[rel]['oid'],rel
    assert tree[rel]['kind']=='blob',rel
    assert int(tree[rel]['mode'],8)&0o777==p.lstat().st_mode&0o777,rel
policy=tomllib.loads((SRC/'deny.toml').read_text())
assert policy['licenses']['exceptions']==[{'name':'unicode-ident','version':'=1.0.27','allow':['MIT','Apache-2.0','Unicode-3.0']}]
assert set(policy['licenses']['allow'])=={'MIT','Apache-2.0','BSD-2-Clause','BSD-3-Clause','ISC'}
assert (SRC/'Cargo.toml').read_bytes()==(REPO/'Cargo.toml').read_bytes()
for label,paths in [('preview-package-input-diff',['deny.toml','Cargo.lock','crates/ckks-train-core','crates/ckks-train-ckks','crates/ckks-train-nn','crates/ckks-train-cuda'])]:
    args=['git','diff','--exit-code','6fba46c','2dc049a796949b685a5c23f8b05440ee2ce474aa','--']+paths
    with (ROOT/'logs'/f'{label}.stdout').open('wb') as out,(ROOT/'logs'/f'{label}.stderr').open('wb') as err: p=subprocess.run(args,cwd=REPO,stdout=out,stderr=err)
    save('preview-package-input-diff.json',{'argv':args,'exit':p.returncode})
    assert p.returncode==0
notice=(SRC/'THIRD_PARTY_NOTICES').read_text()
unicode=notice[notice.index('    UNICODE LICENSE V3'):]
unicode=''.join(line[4:] if line.startswith('    ') else line for line in unicode.splitlines(keepends=True))
u=(ROOT/'vendor/unicode-ident-1.0.27/LICENSE-UNICODE').read_bytes()
assert unicode.encode()==u,'Unicode complete notice differs'
rustix=notice.split('The following text is reproduced verbatim from rustix 1.1.6 LICENSE-MIT:\n\n',1)[1].split('\n\n\nUnicode character data',1)[0]
rustix=''.join(line[4:] if line.startswith('    ') else line for line in rustix.splitlines(keepends=True)).rstrip('\n')+'\n'
rb=(ROOT/'vendor/rustix-1.1.6/LICENSE-MIT').read_bytes()
assert rustix.encode()==rb,'rustix MIT complete notice differs'
archives=json.loads((ROOT/'archives.json').read_text()); internal_hashes={a['crate']:a['sha256'] for a in archives}
canonical={'basis.cu':'f339bcb5492fee28e5041f7c3097aea6326b2024f0f9a4be2855fcd21bafef0a','keyswitch.cu':'8001e378323e64173700f82aa734d042d75c08af8aaad906864120dcd0ea5014','ntt.cu':'fc487dfe9190c712c4a27f862100019d57afe7f501ce17acbc31691155b315da','pointwise.cu':'d7da2002c30ff52256ffed4ba8b88117effaf3f039f6e44ced84db7666b1541b'}
secret_patterns={'private-key':rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----','aws-access':rb'(?:AKIA|ASIA)[A-Z0-9]{16}','github-token':rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}','openai-token':rb'sk-(?:proj-)?[A-Za-z0-9_-]{40,}'}
results=[]
for archive in archives:
    crate=archive['crate']; file=ROOT/archive['archive']; assert sha(file.read_bytes())==archive['sha256']
    members=[]; checks={}; payload={}
    with tarfile.open(file) as tf:
        for member in tf.getmembers():
            rel=pathlib.PurePosixPath(member.name); assert rel.parts[0]==crate+'-1.0.0'
            path='/'.join(rel.parts[1:]); assert path and '..' not in rel.parts
            assert member.isfile(),member.name
            b=tf.extractfile(member).read(); payload[path]=b
            members.append({'path':path,'bytes':len(b),'sha256':sha(b),'mode':oct(member.mode)})
    assert len(payload)==len(members),'duplicate archive member'
    manifest=tomllib.loads(payload['Cargo.toml'].decode()); info=manifest['package']
    assert info['name']==crate and info['version']=='1.0.0' and info['license']=='MIT'
    assert info['repository']=='https://github.com/kenmaro3/ckks-train'
    assert payload['Cargo.toml.orig']==(SRC/'crates'/crate/'Cargo.toml').read_bytes()
    (ROOT/'manifests'/f'{crate}.Cargo.toml.normalized').write_bytes(payload['Cargo.toml'])
    (ROOT/'manifests'/f'{crate}.Cargo.lock').write_bytes(payload['Cargo.lock'])
    generated={'Cargo.toml','Cargo.toml.orig','Cargo.lock'}
    for path,b in payload.items():
        if path not in generated:
            orig=SRC/'crates'/crate/path
            assert orig.is_file() and not orig.is_symlink(),str(orig)
            assert orig.read_bytes()==b,(crate,path)
        assert not set(pathlib.PurePosixPath(path).parts)&{'_ext','.git','target','.superpowers','data','results'},path
        assert not re.search(r'(^|/)(\.env(?:\.|$)|credentials?|id_rsa|id_ed25519)',path,re.I),path
        for kind,pattern in secret_patterns.items():assert not re.search(pattern,b),(crate,path,kind)
    for name in ('LICENSE','THIRD_PARTY_NOTICES'):
        assert payload[name]==(SRC/name).read_bytes()
        assert by_path[f'crates/{crate}/{name}']['kind']=='file'
        checks[name]={'bytes':len(payload[name]),'sha256':sha(payload[name]),'complete_root_match':True,'source_regular_mode':by_path[f'crates/{crate}/{name}']['mode']}
    assert payload['README.md']==(SRC/'crates'/crate/'README.md').read_bytes()
    internal=[]
    def walk_tables(t,prefix=''):
        for k,v in t.items():
            if k in ('dependencies','dev-dependencies','build-dependencies'):
                for name,dep in v.items():
                    assert not name.startswith('hekit-')
                    if name.startswith('ckks-train-'):
                        assert dep['version']=='1.0.0' and 'path' not in dep,(name,dep)
                        internal.append({'section':prefix+k,'name':name,'version':dep['version']})
            elif isinstance(v,dict):walk_tables(v,prefix+k+'.')
    walk_tables(manifest)
    lock=tomllib.loads(payload['Cargo.lock'].decode())
    for package in lock['package']:
        if package['name'].startswith('ckks-train-') and package['name']!=crate:
            assert package['checksum']==internal_hashes[package['name']],package
    kernelchecks={}
    if crate.endswith('-cuda'):
        for name,hashvalue in canonical.items(): assert sha(payload['kernels/'+name])==hashvalue
        for path,b in payload.items():
            if path.startswith('kernels/'): kernelchecks[path]=sha(b)
        assert len(kernelchecks)==5
    assert b'.git' not in b'\n'.join(x['path'].encode() for x in members)
    (ROOT/'manifests'/f'{crate}.members.json').write_text(json.dumps(members,indent=2,sort_keys=True)+'\n')
    results.append(archive|{'members':len(members),'manifest_internal_dependencies':internal,'license_notices':checks,'all_other_members_exact_commit_owned':True,'README_exact_commit_match':True,'kernel_sha256':kernelchecks,'excluded_paths_or_secret_pattern_hits':0})
save('package-audit.json',{'source_commit':'2dc049a796949b685a5c23f8b05440ee2ce474aa','source_entries':len(source),'source_bytes':sum(r['bytes'] for r in source),'source_manifest_sha256':sha((ROOT/'manifests/source.sha256.jsonl').read_bytes()),'git_blob_identity_and_modes_exact':True,'preview_package_inputs_identical':True,'policy_exception':policy['licenses']['exceptions'],'unicode_notice':{'bytes':len(u),'sha256':sha(u),'complete_exact_match':True},'rustix_selected_MIT_notice':{'bytes':len(rb),'sha256':sha(rb),'complete_exact_match':True},'packages':results})
workspace_lock=tomllib.loads((SRC/'Cargo.lock').read_text())
root_lock={(p['name'],p['version']):p for p in workspace_lock['package']}
external_locks=[]
for file in sorted((ROOT/'manifests').glob('*.Cargo.lock')):
    dependencies=[]
    for dependency in tomllib.loads(file.read_text())['package']:
        if dependency['name'].startswith('ckks-train-'): continue
        key=(dependency['name'],dependency['version']); assert key in root_lock,key
        for field in ('source','checksum'): assert dependency.get(field)==root_lock[key].get(field),(file,key,field)
        dependencies.append({k:dependency.get(k) for k in ('name','version','source','checksum')})
    external_locks.append({'package_lock':file.name,'external_dependencies_match_fixed_workspace_lock':True,'dependencies':dependencies})
save('external-lock-audit.json',external_locks)
print(json.dumps({'status':'PASS','source_entries':len(source),'archives':len(archives),'external_package_locks_match':len(external_locks)}))
