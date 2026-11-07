"""FIX1 independent finite oracle, frozen before CW01/02/05 production changes.
No project imports/calls, HE, native process or dataset operations. Grammar is
u64LE; Text/Bytes=u64length+bytes; tags=u8; lists=u64count+items; Option=0/1.
All137 members named in the pre-execution inventory are authored below.
Historical generic Full provenance template is reused as fictional input only;
its full canonical traversal is independently transcribed, never Rust encoded.
"""
from pathlib import Path
import copy,hashlib,json,struct
ROOT=Path(__file__).resolve().parents[3];LEDGER=ROOT/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5'
SRC=ROOT/'crates/ckks-train-bench/src/study_reference/document/evidence/observed'
OLDRAW=(SRC/'literals.json').read_bytes();OLD={k:bytes.fromhex(v) for k,v in json.loads(OLDRAW).items()}
TEMPLATE=json.loads((ROOT/'crates/ckks-train-bench/src/study_reference/target_goldens/initial.json').read_text())
ENC='hekit-study-reference-encoding-v1';REV='c'*40;P='d7-resource-observed-rss-v1';H=lambda c:c*64
NAMES=['inputs','implementation','layout','resource','user'];CK=['rotations','relins','ct_ct_tensor','ct_pt_mults','rescales','sse_switches','bootstraps'];TK=['setup','server','bootstrap','refresh','input_encryption','diagnostic','fp32','end_to_end','publish']
VALUES={};PACKED={}
def n(x):return struct.pack('<Q',x)
def q(x):return bytes([x])
def t(x):x=x.encode() if isinstance(x,str) else x;return n(len(x))+x
def sha(x):return hashlib.sha256(x).hexdigest()
def hd(x):return t(ENC)+t('hekit-test-'+x)+b'\1'
def ls(xs):return n(len(xs))+b''.join(xs)
def opt(x):return b'\0' if x is None else b'\1'+x
def js(v):return json.dumps(v,ensure_ascii=True,separators=(',',':')).encode()
def cj(x):return {'wall_source':0,'unix_ns':100+x,'monotonic_source':0,'elapsed_since_operation_start_ns':x}
def cl(v):return b'\0'+n(v['unix_ns'])+b'\0'+n(v['elapsed_since_operation_start_ns'])
def f(path='fixture',data=b'x',root=2):return {'root':root,'path':path,'length':len(data),'sha256':sha(data)}
def fb(v):return q(v['root'])+t(v['path'])+n(v['length'])+t(v['sha256'])
def link(data,canonical):return {'file_sha256':sha(data),'payload_sha256':sha(canonical)}
def lb(v):return t(v['file_sha256'])+t(v['payload_sha256'])
def store(key,data,base=None):
 assert key not in VALUES;VALUES[key]=data
 if base is None:PACKED[key]=data.hex();return data
 before=OLD[base[4:]] if base.startswith('old:') else VALUES[base]
 if len(before)!=len(data):
  a=0
  while a<min(len(before),len(data)) and before[a]==data[a]:a+=1
  z=0
  while z<min(len(before)-a,len(data)-a) and before[-1-z]==data[-1-z]:z+=1
  patches=[[a,len(before)-a-z,data[a:len(data)-z if z else len(data)].hex()]]
 else:
  spans=[];i=0
  while i<len(data):
   if data[i]==before[i]:i+=1;continue
   a=i;i+=1
   while i<len(data) and (data[i]!=before[i] or any(data[j]!=before[j] for j in range(i,min(i+8,len(data))))):i+=1
   spans.append([a,i-a,data[a:i].hex()])
  patches=spans
 PACKED[key]={'base':base,'patches':patches};return data

def endpoint(u,full):return {'updates':u,'samples':u*20,'crossed':[u//6] if full else [],'counter':dict(zip(CK,range(1,8))),'times':dict(zip(TK,[3,u*10,1,0,2,3,4,u*20,0])),'he_correct':2,'fp32_correct':2,'test_count':30,'weight_max_abs_diff_bits':0,'prediction_max_abs_diff_bits':0}
def epb(e):return n(e['updates'])+n(e['samples'])+ls([n(x) for x in e['crossed']])+b''.join(n(e['counter'][k]) for k in CK)+b''.join(n(e['times'][k]) for k in TK)+b''.join(n(e[k]) for k in ['he_correct','fp32_correct','test_count','weight_max_abs_diff_bits','prediction_max_abs_diff_bits'])
def result_b(r):
 b=hd('cpu-result-v1')+q(r['scope'])+t(bytes.fromhex(r['case']))+t(bytes.fromhex(r['execution']))+t(r['attempt'])+fb(r['admission_or_preflight'])+fb(r['start'])+b''.join(fb(x) for x in r['evidence'])+t(r['policy'])
 b+=b''.join(t(r[k]) for k in ['source_manifest','source_revision','executable','contract','receipt','materialized','consumed_plan'])+q(r['status'])+n(r['completed_updates'])+n(r['processed_samples'])+q(r['poisoned'])+b'\0'+ls([epb(x) for x in r['checkpoints']])+opt(epb(r['final']) if r['final'] else None)+opt(t(r['model_digest']) if r['model_digest'] else None)
 target=r['target']
 b+=opt(fb(target['lock'])+n(target['threshold_bits'])+q(target['reached'])+opt(n(target['first_checkpoint']) if target['first_checkpoint'] is not None else None)+t(target['time_basis'])+opt(n(target['time_ns']) if target['time_ns'] is not None else None)+n(target['censor_ns']) if target else None)
 return b+cl(r['ended'])
def exported(r,rb):
 eid=sha(t(ENC)+t('hekit-test-result-export-id-v1')+q(r['scope'])+t(r['start']['sha256']))
 raw=js({'schema':'hekit-test-cpu-result-json-v1','synthetic':True,'export_id':eid,'canonical_result':f('result.bin',rb),'payload':r});return raw,eid

def evjson(kind,b):return js({'schema':'hekit-test-quick-'+kind+'-v2','kind':'quick_'+kind,'synthetic':True,'canonical_bytes':b.hex(),'integrity':{'encoding':ENC,'hash':'sha256','payload_sha256':sha(b)}})
def gates_b(prefix):return b''.join(t('test-'+P if k=='resource' else 'test-d7-'+k+'-v1')+t(sha(OLD[prefix+k])) for k in NAMES)
def quick_end(identity,gates,pre,preb,start,startb,result,resultjson,eid):return hd('quick-end-v2')+t(sha(identity))+gates+lb(link(pre,preb))+lb(link(start,startb))+b'\0'+n(2)+b'\0'+cl(cj(9))+b'\1'+fb(f('result.bin',result))+b'\1'+fb(f('result.json',resultjson))+b'\1'+fb(f('study-cpu-'+eid+'.json',resultjson,5))+n(7)
def supervisor(claim,start,result,end,resolved=True):return hd('supervisor-end-v1')+fb(f('pending.bin',OLD['pending']))+b'\1'+fb(f('claim.bin',claim))+b'\1'+fb(f('start.json',start))+b'\1'+fb(f('result.bin',result))+b'\1'+fb(f('end.json',end))+b'\1'+fb(f('samples'))+n(0)+t('peak_rss_kb_tree_sampled_10s')+b'\1'+n(9)+q(0 if resolved else 7)+b'\1'+n(0)+q(resolved)+cl(cj(10))
# Four omitted Quick equalities; independent dependencies are rebuilt, bundle/gates unchanged.
for variant in ['attempt','approval','receipt','executable']:
 identity=OLD['0-1-identity']
 if variant=='attempt':identity=identity.replace(t('attempt1'),t('attempt2'))
 context=[sha(b'x')]*3
 if variant!='attempt':context[['approval','receipt','executable'].index(variant)]=sha(b'foreign-'+variant.encode())
 g=gates_b('0-1-')
 pb=hd('quick-preflight-v2')+t(identity)+t(sha(identity))+g+ls([t(b'fictional-study-train'),t(b'\xff')])+t(sha(b'x'))+t(REV)+b''.join(t(x) for x in context)+cl(cj(3));pr=evjson('preflight',pb)
 sb=hd('quick-start-v2')+t(sha(identity))+g+lb(link(pr,pb))+cl(cj(4));sr=evjson('start',sb)
 oldresult=OLD['0-1-result'];oldjson=OLD['0-1-result-json'];eid=json.loads(oldjson)['export_id']
 eb=quick_end(identity,g,pr,pb,sr,sb,oldresult,oldjson,eid);er=evjson('end',eb)
 for member,data in [('preflight-json',pr),('start-json',sr),('end-json',er)]:store('quick-'+variant+'-'+member,data,'old:0-1-'+member)
# Clock defect: complete Quick result and all downstream file links remain coherent.
r=json.loads(OLD['0-1-result-json'])['payload'];r['ended']={'wall_source':0,'unix_ns':0,'monotonic_source':0,'elapsed_since_operation_start_ns':0};rb=result_b(r);rj,eid=exported(r,rb)
pb=OLD['0-1-preflight'];pr=OLD['0-1-preflight-json'];sb=OLD['0-1-start'];sr=OLD['0-1-start-json'];eb=quick_end(OLD['0-1-identity'],gates_b('0-1-'),pr,pb,sr,sb,rb,rj,eid);er=evjson('end',eb)
for member,data in [('preflight-json',pr),('start-json',sr),('result',rb),('result-json',rj),('result-export',rj),('end-json',er),('supervisor-end',supervisor(OLD['0-1-claim'],sr,rb,er))]:store('quick-clock-'+member,data,'old:0-1-'+('result-json' if member=='result-export' else member))
# Generic Full grammar, unchanged; known finite fictional captured provenance.
def obs(v,encode):return (b'\0'+encode(v['value'])) if v['status']=='available' else (b'\1'+t(v['reason']))
def rawos(v):return b'\0'+t(bytes.fromhex(v['bytes']))
def provenance(p):
 v=p['value'];a=v['build'];b=b'\0'+t(v['schema'])+b'\1'
 b+=b''.join(t(a[k]) for k in ['compiler_vv','target','host','profile','opt_level'])+q(a['debug'])+ls([t(x) for x in sorted(a['features'])])+b''.join(t(a[k]) for k in ['package_version','cargo_lock_sha256','source_manifest_sha256','config_manifest_sha256','build_environment_sha256','recipe_version','receipt_file_sha256'])
 a=v['source'];b+=rawos(a['root'])+obs(a['commit'],t)+q(0)+q(0)+ls([t(x['path'])+t(x['sha256']) for x in sorted(a['manifest'],key=lambda x:x['path'])])
 a=v['executable'];b+=rawos(a['path'])+t(a['start_sha256'])+t(a['end_sha256']);a=v['invocation'];b+=ls([rawos(x) for x in a['argv']])+rawos(a['cwd']);a=v['host']
 b+=b''.join(obs(a[k],t) for k in ['cpu_model','os','hostname'])+b''.join(obs(a[k],n) for k in ['available_cpus','process_id','observed_threads'])+n(a['execution_threads'])
 b+=ls([t(x['key'])+obs(x['value'],t) for x in sorted(v['thread_environment'],key=lambda x:x['key'])])+t(v['recheck']['source_manifest_sha256'])+t(v['recheck']['build_receipt_file_sha256']);return b
PROV=copy.deepcopy(TEMPLATE['provenance']);v=PROV['value'];v['build']['source_manifest_sha256']=sha(b'x');v['build']['receipt_file_sha256']=sha(b'x');v['source']['commit']={'status':'available','value':REV};v['executable']['start_sha256']=v['executable']['end_sha256']=sha(b'x');v['recheck']['source_manifest_sha256']=v['recheck']['build_receipt_file_sha256']=sha(b'x');PROVB=provenance(PROV)
LOCKID=sha(t(ENC)+t('hekit-study-target-lock-id-v1')+t('campaign1')+t('iris__emlp1'))
CONTRACT=b'fix1-contract';RECEIPT=b'fix1-receipt';CH=sha(CONTRACT);RH=sha(RECEIPT)
POLICY=copy.deepcopy(TEMPLATE['policy']);POLICY['registry_sha256']=H('a');POLICY['approval_sha256']=sha(b'x')
def policy_b(p):return t(p['scientific_revision'])+t(p['scientific_sha256'])+t(p['registry_sha256'])+t(p['portable_lock_sha256'])+ls([t(x['path'])+t(x['sha256']) for x in p['design_documents']])+t(p['approval_sha256'])
def identity_doc(b,receipt=False):return {'schema':'hekit-study-seed-input-receipt-v1' if receipt else 'hekit-study-profile-contract-v1','encoding':ENC,'canonical_bytes':b.hex(),'sha256':sha(b)}
def idb(v):return t(v['schema'])+t(v['encoding'])+t(bytes.fromhex(v['canonical_bytes']))+t(v['sha256'])
def oldclock(c):return {'wall_source':'unix_system_time','unix_ns':c['unix_ns'],'monotonic_source':'std_instant','elapsed_since_operation_start_ns':c['elapsed_since_operation_start_ns']}
def full_canon(a,isstart,case,execution):
 schema=a['schema'];b=t(ENC)+t(schema)+t(schema)+q(2 if isstart else 1)+b'\0'+t(a['writer']['api_version'])+t(a['writer']['algorithm_revision'])+b'\1'+t(a['campaign_id'])+t(a['profile_id'])+t(a['lock_id'])+policy_b(a['policy'])+lb(a['lock'])+lb(a['reference'])
 if not isstart:b+=idb(a['contract'])+n(a['selected_input']['seed'])+t(a['selected_input']['materialized_sha256'])+idb(a['selected_input']['seed_receipt'])+case[len(t(ENC)+t('hekit-study-registered-case-v1')):]+t(a['case_id'])+execution[len(t(ENC)+t('hekit-study-case-binding-v1')):]+t(a['execution_binding_sha256'])+t(a['attempt_id'])+b'\0'+b''.join(t(a['gates'][k]['policy_revision'])+t(a['gates'][k]['evidence_sha256']) for k in NAMES)+t(a['gates_sha256'])+cl(a['admitted'])
 else:b+=t(a['case_id'])+t(a['execution_binding_sha256'])+n(a['seed'])+t(a['seed_receipt_sha256'])+t(a['materialized_sha256'])+t(a['attempt_id'])+lb(a['admission'])+cl(a['started'])
 return b+PROVB+t(ENC)+t('sha256')

def full(mode,variant=None):
 label='ni' if mode else 'i';prefix='full-'+label+'-';case=OLD['case-ni' if mode else 'case-i']
 request=OLD[f'{mode}-0-request'];consumption=hd('gate-consumption-v1')+b'\0'+n(0)+t(sha(request))+t(H('d'))+t(RH)+n(2400)
 execution=t(ENC)+t('hekit-study-case-binding-v1')+b'\0'+t(REV)+t(sha(b'x'))+t(H('a'))+opt(t(H('b')) if mode else None)+t(H('c'))+t(H('d'))+b'\1\1\1'+q(mode)+t(H('e'))+t(H('f'))+t(H('a'))+t(H('b'))
 common=lambda policy:t(policy)+b'\0'+t(execution)+t(CH)+t(RH)+t(H('a'))+t('campaign1')+t('iris__emlp1')+t(sha(b'x'))*4+t(REV)
 branch=t(case)+t(sha(case))+t(sha(b'x'))*4+t(LOCKID)+n(0)+t('attempt1')+b'\0'
 app=t(sha(b'x'))+t(REV)+t(H('a'))+opt(t(H('b')) if mode else None)+t(H('e'))+t(H('f'))+t(H('c'))+t(H('d'))+t(H('b'))+b'\0\0'+q(mode)+b'\0'
 claim=OLD[f'{mode}-0-claim'];idref=lambda receipt:t('hekit-study-seed-input-receipt-v1' if receipt else 'hekit-study-profile-contract-v1')+t(ENC)+n(len(RECEIPT if receipt else CONTRACT))+t(RH if receipt else CH)
 inputs=ls([fb(f('input'))])+idref(False)+idref(True)+t(H('a'))+t(consumption)+t(sha(consumption))+idref(True)+t(H('a'))
 correctness=[q(i)+fb(f('report'))+fb(f('command'))+fb(f('manifest',root=1))+t(REV)+fb(f('criterion'))+app for i in range(1+mode)]
 security=fb(f('policy'))+fb(f('parameter'))+t(REV)+ls([fb(f('estimator'))])+fb(f('environment'))+fb(f('command'))+fb(f('model'))+ls([q(i)+fb(f('problem'))+fb(f('bdgl'))+fb(f('matzov')) for i in range(1+mode)])
 implementation=app+ls(correctness)+security+ls([q(i) for i in range(4+mode)])
 layout=t(H('a'))+opt(t(H('b')) if mode else None)+t(H('c'))+t(H('d'))+t(H('e'))+t(consumption)+t(sha(consumption))+t(request)+t(sha(request))
 resource=app+t(H('a'))+fb(f('policy'))+fb(f('native'))+t('d7-cpu-source-sizing-v1')+OLD['demand']+b''.join(n(x) for x in [67108864,1000000,10,20,0,5])+t(H('a'))+cl(cj(3))+n(10000)+n(0)+n(10000)+fb(f('host'))+fb(f('disk'))+OLD['supervisor-binding']+fb(f('claim.bin',claim))
 user_tail=OLD[f'{mode}-0-user'];# User tail is retained exactly after its old Common+branch; no changed user authority fields.
 oldcommon=t('test-d7-user-v1')+b'\0'+t(execution)+t(H('e'))+t(H('f'))+t(H('a'))+t('campaign1')+t('iris__emlp1')+t(sha(b'x'))*4+t(REV)
 oldbranch=t(case)+t(sha(case))+t(sha(b'x'))*4+t(H('b'))+n(0)+t('attempt1')+b'\0'
 user_tail=user_tail[len(hd('gate-user-evidence-v1')+oldcommon+oldbranch):]
 sides={k:hd('resource-observed-evidence-v1' if k=='resource' else 'gate-'+k+'-evidence-v1')+common('test-'+P if k=='resource' else 'test-d7-'+k+'-v1')+branch+b for k,b in zip(NAMES,[inputs,implementation,layout,resource,user_tail])}
 gates={k:{'policy_revision':'test-'+P if k=='resource' else 'test-d7-'+k+'-v1','evidence_sha256':sha(sides[k])} for k in NAMES}
 casej={'registry_sha256':H('a'),'origin':'family','registration_id':'main_ni' if mode else 'main_i','profile_id':'iris__emlp1','seed':0,'scope':'full','role':'main','backend':'cpu','mode':'noninteractive' if mode else 'interactive','placement':'when_needed' if mode else 'client_refresh','packing':'batched_diagonal','security_policy':'n16_boot' if mode else 'n16_train','logical_batch':20,'eval_config':None}
 xj={'implementation':'main_diagonal','implementation_revision':REV,'implementation_manifest_sha256':sha(b'x'),'parameter_identity_sha256':H('a'),'bootstrap_descriptor_sha256':H('b') if mode else None,'layout_plan_sha256':H('c'),'trainer_plan_sha256':H('d'),'eval_config':{'hoist':True,'bsgs':True,'lazy_relin':True},'bootstrap_engine':'host' if mode else 'none','key_policy_sha256':H('e'),'security_policy_sha256':H('f'),'physical_binding_sha256':H('a'),'runtime_policy_sha256':H('b')}
 a={'schema':'hekit-test-he-admission-v1','kind':'he_admission','status':'complete','writer':TEMPLATE['writer'],'synthetic':True,'campaign_id':'campaign1','profile_id':'iris__emlp1','lock_id':LOCKID,'policy':POLICY,'lock':link(b'x',b'x'),'reference':link(b'x',b'x'),'contract':identity_doc(CONTRACT),'selected_input':{'seed':0,'materialized_sha256':H('a'),'seed_receipt':identity_doc(RECEIPT,True)},'case':casej,'case_id':sha(case),'execution':xj,'execution_binding_sha256':sha(execution),'attempt_id':'attempt2' if variant=='attempt' else 'attempt1','predecessor':None,'gates':gates,'admitted':oldclock(cj(3)),'provenance':PROV}
 if variant=='tuple':a['gates']['resource']['policy_revision']='test-d7-resource-v1'
 gb=t(ENC)+t('hekit-study-admission-gates-v1')+b''.join(t(a['gates'][k]['policy_revision'])+t(a['gates'][k]['evidence_sha256']) for k in NAMES);a['gates_sha256']=sha(gb)
 ac=full_canon(a,False,case,execution);a['integrity']={'encoding':ENC,'hash':'sha256','payload_sha256':sha(ac)};ar=js(a)
 s={'schema':'hekit-test-he-start-v1','kind':'he_start','status':'complete','writer':TEMPLATE['writer'],'synthetic':True,'campaign_id':'campaign1','profile_id':'iris__emlp1','lock_id':LOCKID,'policy':POLICY,'lock':a['lock'],'reference':a['reference'],'case_id':sha(case),'execution_binding_sha256':sha(execution),'seed':0,'seed_receipt_sha256':RH,'materialized_sha256':H('a'),'attempt_id':a['attempt_id'],'admission':link(ar,ac),'started':oldclock(cj(4)),'provenance':PROV}
 if variant=='admission-start':s['admission']['payload_sha256']=H('0')
 sc=full_canon(s,True,case,execution);s['integrity']={'encoding':ENC,'hash':'sha256','payload_sha256':sha(sc)};sr=js(s)
 checkpoints=[endpoint(k*6,True) for k in range(1,401)];target={'lock':f('target-lock.json'),'threshold_bits':0,'reached':True,'first_checkpoint':6,'time_basis':'cumulative_update_ns','time_ns':60,'censor_ns':24000}
 r={'scope':0,'case':case.hex(),'execution':execution.hex(),'attempt':a['attempt_id'],'admission_or_preflight':f('admission.json',ar),'start':f('start.json',sr),'evidence':[f(k+'.bin',sides[k]) for k in NAMES],'policy':P,'source_manifest':sha(b'x'),'source_revision':REV,'executable':sha(b'x'),'contract':CH,'receipt':RH,'materialized':H('a'),'consumed_plan':sha(consumption),'status':0,'completed_updates':2400,'processed_samples':48000,'poisoned':False,'failed_prefix':None,'checkpoints':checkpoints,'final':checkpoints[-1],'model_digest':H('c'),'target':target,'ended':cj(8)}
 if variant=='target-lock':r['target']['lock']=f('target-lock.json',b'y')
 if variant=='target-threshold':r['target']['threshold_bits']=0x3fb0000000000000
 if variant=='source':r['source_manifest']=sha(b'y')
 if variant=='case':r['case']=OLD['case-ni' if mode==0 else 'case-i'].hex()
 if variant=='evidence':r['evidence'][0]=f('inputs.bin',b'y')
 if variant=='clock':r['ended']=cj(3)
 if variant=='scope':r.update(scope=1,completed_updates=2,processed_samples=40,checkpoints=[endpoint(1,False),endpoint(2,False)],final=endpoint(2,False),target=None)
 rb=result_b(r);rj,eid=exported(r,rb)
 endrefs=[f('result.bin',rb),f('result.json',rj),f('study-cpu-'+eid+'.json',rj,5)]
 if variant=='output':endrefs[1]=f('result.json',b'y')
 er=hd('he-end-v1')+t(a['attempt_id'])+fb(f('admission.json',ar))+fb(f('start.json',sr))+b''.join(b'\1'+fb(x) for x in endrefs)+b'\0'+n(2400)+n(48000)+n(7)+cl(cj(9))
 sup=supervisor(claim,sr,rb,er,variant!='closure')
 members={'admission':ar,'start':sr,'result':rb,'result-json':rj,'result-export':rj,'end':er,'supervisor-end':sup}
 if variant is None:
  for k,b in dict(request=request,consumption=consumption,claim=claim,**sides,**{'admission-canonical':ac,'start-canonical':sc},**members).items():store(prefix+k,b)
 else:
  for k,b in members.items():store('full-'+variant+'-'+k,b,'full-i-'+k)
for mode in [0,1]:full(mode)
for variant in ['tuple','attempt','admission-start','target-lock','target-threshold','source','case','output','evidence','closure','clock','scope']:full(0,variant)
assert len(VALUES)==137,len(VALUES)
assert max(map(len,VALUES.values()))<=524288 and sum(map(len,VALUES.values()))<=8388608
raw=(json.dumps(PACKED,sort_keys=True,separators=(',',':'))+'\n').encode();assert len(raw)<=4194304
assert (SRC/'literals.json').read_bytes()==OLDRAW
(SRC/'fix1_literals.json').write_bytes(raw)
manifest={'classification':'independent fictional supplied-record FIX1, no authority/science','members':len(VALUES),'aggregate_expanded_bytes':sum(map(len,VALUES.values())),'max_member_bytes':max(map(len,VALUES.values())),'compact_bytes':len(raw),'compact_sha256':sha(raw),'original78_sha256':sha(OLDRAW),'member_inventory':{k:{'bytes':len(b),'sha256':sha(b)} for k,b in VALUES.items()}}
(LEDGER/'task-54-d7-cpu-wire-fix1-fixture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='member_inventory'}))
