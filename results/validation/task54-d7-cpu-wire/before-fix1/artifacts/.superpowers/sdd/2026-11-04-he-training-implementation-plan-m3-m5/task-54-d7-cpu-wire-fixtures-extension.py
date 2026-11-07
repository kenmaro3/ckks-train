"""Independent finite extension; source is a complete closed input/output ledger.
No project imports/encoders/execution. Retains all original bytes verbatim.
Exactly four coherent inert bundles (main I/NI x Full/Quick), two Quick v2
chains, and one fixed observed policy JSON; no live owner or measured science.
"""
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
LEDGER=ROOT/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5'
OUT=ROOT/'crates/ckks-train-bench/src/study_reference/document/evidence/observed/literals.json'
B={k:bytes.fromhex(v) for k,v in json.loads((LEDGER/'d7-cpu-wire-logs/literals-initial.json').read_text()).items()}
ENC='hekit-study-reference-encoding-v1';REV='c'*40;P='d7-resource-observed-rss-v1';H=lambda x:x*64
NAMES=['inputs','implementation','layout','resource','user'];CK=['rotations','relins','ct_ct_tensor','ct_pt_mults','rescales','sse_switches','bootstraps'];TK=['setup','server','bootstrap','refresh','input_encryption','diagnostic','fp32','end_to_end','publish']
def n(x):return struct.pack('<Q',x)
def tag(x):return bytes([x])
def t(x):x=x.encode() if isinstance(x,str) else x;return n(len(x))+x
def sha(x):return hashlib.sha256(x).hexdigest()
def hd(x,v=1):return t(ENC)+t('hekit-test-'+x)+tag(v)
def f(path='fixture',root=2,data=b'x',hash=None):return tag(root)+t(path)+n(len(data))+t(hash or sha(data))
def fj(path='fixture',root=2,data=b'x',hash=None):return {'root':root,'path':path,'length':len(data),'sha256':hash or sha(data)}
def clock(x=1):return b'\0'+n(100+x)+b'\0'+n(x)
def cj(x=1):return {'wall_source':0,'unix_ns':100+x,'monotonic_source':0,'elapsed_since_operation_start_ns':x}
def ls(*v):return n(len(v))+b''.join(v)
def opt(x):return b'\0' if x is None else b'\1'+x
def js(v):return json.dumps(v,separators=(',',':')).encode()
def put(k,v):assert k not in B;B[k]=v;return v
def evjson(kind,b):return js({'schema':'hekit-test-quick-'+kind+'-v2','kind':'quick_'+kind,'synthetic':True,'canonical_bytes':b.hex(),'integrity':{'encoding':ENC,'hash':'sha256','payload_sha256':sha(b)}})
for mode in range(2):
 for scope in range(2):
  pre=f'{mode}-{scope}-';name=lambda s:pre+s
  case=B['case-i' if mode==0 else 'case-ni'];work=B[f'W{1+mode*2+scope:02}']
  request=put(name('request'),hd('gate-request-v1')+tag(scope)+b'\0\0\1\1\1'+b''.join(n(x) for x in [67108864,1000000,2,2,2400 if scope==0 else 2,0,1]))
  execution=t(ENC)+t('hekit-study-case-binding-v1')+b'\0'+t(REV)+t(sha(b'x'))+t(H('a'))+opt(t(H('b')) if mode else None)+t(H('c'))+t(H('d'))+b'\1\1\1'+tag(mode)+t(H('e'))+t(H('f'))+t(H('a'))+t(H('b'))
  consumed=hd('gate-consumption-v1')+tag(scope)+n(0)+t(sha(request))+t(H('d'))+t(H('f'))
  if scope==0:consumed+=n(2400)
  else:
   consumed+=n(2)
   for i in range(2):consumed+=n(i)+n(0)+n(i)+ls(*(n(x) for x in range(i*20,(i+1)*20)))+ls(n(0)+n(20))
  put(name('consumption'),consumed)
  identity=hd('quick-identity-v1')+t(case)+t(sha(case))+t(execution)+t(sha(execution))+t(H('e'))+t(H('f'))+t(H('a'))+t('campaign1')+t('iris__emlp1')+n(0)+b'\1'+t(consumed)+t(sha(consumed))+t(request)+t(sha(request))+t(H('d'))+t('attempt1')
  if scope:put(name('identity'),identity)
  common=lambda policy:t(policy)+tag(scope)+t(execution)+t(H('e'))+t(H('f'))+t(H('a'))+t('campaign1')+t('iris__emlp1')+t(sha(b'x'))*4+t(REV)
  branch=(t(case)+t(sha(case))+t(sha(b'x'))*4+t(H('b'))+n(0)+t('attempt1')+b'\0') if scope==0 else (t(case)+t(sha(case))+t(identity)+t(sha(identity))+n(0)+b'\1'+t(sha(consumed))+t(sha(request))+t('attempt1'))
  app=t(sha(b'x'))+t(REV)+t(H('a'))+opt(t(H('b')) if mode else None)+t(H('e'))+t(H('f'))+t(H('c'))+t(H('d'))+t(H('b'))+b'\0\0'+tag(mode)+tag(scope)
  claim=put(name('claim'),hd('resource-observed-claim-v1')+f('pending.bin',data=B['pending'])+t(work)+t(sha(b'x'))+t(sha(b'x'))+t(REV)+clock(2)+B['supervisor-binding'])
  idref=lambda receipt:t('hekit-study-seed-input-receipt-v1' if receipt else 'hekit-study-profile-contract-v1')+t(ENC)+n(1)+t(H('f') if receipt else H('e'))
  inputs=ls(f('input'))+idref(False)+idref(True)+t(H('a'))+t(consumed)+t(sha(consumed))
  if scope==0:inputs+=idref(True)+t(H('a'))
  correctness=[]
  for i in range(1+mode):correctness.append(tag(i)+f('report')+f('command')+f('manifest',1)+t(REV)+f('criterion')+app)
  security=f('policy')+f('parameter')+t(REV)+ls(f('estimator'))+f('environment')+f('command')+f('model')+ls(*(tag(i)+f('problem')+f('bdgl')+f('matzov') for i in range(1+mode)))
  implementation=app+ls(*correctness)+security+ls(*(tag(i) for i in range(4+mode)))
  layout=t(H('a'))+opt(t(H('b')) if mode else None)+t(H('c'))+t(H('d'))+t(H('e'))+t(consumed)+t(sha(consumed))+t(request)+t(sha(request))
  observation=t(H('a'))+clock(3)+n(10000)+n(0)+n(10000)+f('host')+f('disk')
  resource=app+t(H('a'))+f('policy')+f('native')+t('d7-cpu-source-sizing-v1')+B['demand']+b''.join(n(x) for x in [67108864,1000000,10,20,0,5])+observation+B['supervisor-binding']+f('claim.bin',data=claim)
  authority=hd('user-scope-v1')+b'\0'+t('conversation1')+t('message1')+t(b'fictional instruction, not authority')+n(1)+ls(tag(scope)+t('campaign1')+t('iris__emlp1')+t(sha(case))+t('attempt1')+b'\0\0')+n(0)
  current=hd('user-current-v1')+t('conversation1')+t('message1')+n(1)+t(sha(authority))+n(0)+b'\0'+clock(3)
  user=t(authority)+t(sha(authority))+t(current)+t(sha(current))+f('scope',4,authority)+f('current',4,current)+n(0)+clock(3)
  for gate,tail in zip(NAMES,[inputs,implementation,layout,resource,user]):
   put(name(gate),hd('resource-observed-evidence-v1' if gate=='resource' else 'gate-'+gate+'-evidence-v1')+common('test-'+P if gate=='resource' else 'test-d7-'+gate+'-v1')+branch+tail)
  if scope:
   gates=b''.join(t('test-'+P if gate=='resource' else 'test-d7-'+gate+'-v1')+t(sha(B[name(gate)])) for gate in NAMES)
   preflight=put(name('preflight'),hd('quick-preflight-v2')+t(identity)+t(sha(identity))+gates+ls(t(b'fictional-study-train'),t(b'\xff'))+t(sha(b'x'))+t(REV)+t(sha(b'x'))*3+clock(3))
   prejson=put(name('preflight-json'),evjson('preflight',preflight))
   start=put(name('start'),hd('quick-start-v2')+t(sha(identity))+gates+t(sha(prejson))+t(sha(preflight))+clock(4))
   startjson=put(name('start-json'),evjson('start',start))
   def ej(u):return {'updates':u,'samples':20*u,'crossed':[],'counter':dict(zip(CK,range(1,8))),'times':dict(zip(TK,[3,u*10,1,0,2,3,4,u*20,0])),'he_correct':2,'fp32_correct':2,'test_count':30,'weight_max_abs_diff_bits':0,'prediction_max_abs_diff_bits':0}
   def endpoint(u):return n(u)+n(20*u)+n(0)+b''.join(n(x) for x in range(1,8))+b''.join(n(x) for x in [3,u*10,1,0,2,3,4,u*20,0])+n(2)+n(2)+n(30)+n(0)+n(0)
   result=put(name('result'),hd('cpu-result-v1')+b'\1'+t(case)+t(execution)+t('attempt1')+f('preflight.json',data=prejson)+f('start.json',data=startjson)+b''.join(f(g+'.bin',data=B[name(g)]) for g in NAMES)+t(P)+t(sha(b'x'))+t(REV)+t(sha(b'x'))+t(H('e'))+t(H('f'))+t(H('a'))+t(sha(consumed))+b'\0'+n(2)+n(40)+b'\0\0'+ls(endpoint(1),endpoint(2))+b'\1'+endpoint(2)+b'\1'+t(H('c'))+b'\0'+clock(8))
   exportid=sha(t(ENC)+t('hekit-test-result-export-id-v1')+b'\1'+t(sha(startjson)))
   payload={'scope':1,'case':case.hex(),'execution':execution.hex(),'attempt':'attempt1','admission_or_preflight':fj('preflight.json',data=prejson),'start':fj('start.json',data=startjson),'evidence':[fj(g+'.bin',data=B[name(g)]) for g in NAMES],'policy':P,'source_manifest':sha(b'x'),'source_revision':REV,'executable':sha(b'x'),'contract':H('e'),'receipt':H('f'),'materialized':H('a'),'consumed_plan':sha(consumed),'status':0,'completed_updates':2,'processed_samples':40,'poisoned':False,'failed_prefix':None,'checkpoints':[ej(1),ej(2)],'final':ej(2),'model_digest':H('c'),'target':None,'ended':cj(8)}
   resultjson=put(name('result-json'),js({'schema':'hekit-test-cpu-result-json-v1','synthetic':True,'export_id':exportid,'canonical_result':fj('result.bin',data=result),'payload':payload}))
   end=put(name('end'),hd('quick-end-v2')+t(sha(identity))+gates+t(sha(prejson))+t(sha(preflight))+t(sha(startjson))+t(sha(start))+b'\0'+n(2)+b'\0'+clock(9)+b'\1'+f('result.bin',data=result)+b'\1'+f('result.json',data=resultjson)+b'\1'+f('study-cpu-'+exportid+'.json',5,resultjson)+n(7))
   endjson=put(name('end-json'),evjson('end',end))
   put(name('supervisor-end'),hd('supervisor-end-v1')+f('pending.bin',data=B['pending'])+b'\1'+f('claim.bin',data=claim)+b'\1'+f('start.json',data=startjson)+b'\1'+f('result.bin',data=result)+b'\1'+f('end.json',data=endjson)+b'\1'+f('samples')+n(0)+t('peak_rss_kb_tree_sampled_10s')+b'\1'+n(9)+b'\0\1'+n(0)+b'\1'+clock(10))
policy={'schema':'hekit-test-resource-observed-policy-v1','synthetic':True,'host_origin':fj('host-origin'),'native_validation':fj('native'),'source_manifest':H('b'),'source_revision':REV,'executable':H('d'),'recipe':'study-train-cpu-serial-linux-clean-v2','sizing_revision':'d7-cpu-source-sizing-v1','claim_domain':'task54-host-serial-v1','rss_limit_kib':94371840,'sample_interval_ns':10000000000,'term_grace_ns':5000000000}
put('policy-json',js(policy))
assert len(B)<=128 and max(map(len,B.values()))<=16384 and sum(map(len,B.values()))<=262144
raw=(json.dumps({k:v.hex() for k,v in B.items()},sort_keys=True,separators=(',',':'))+'\n').encode();OUT.write_bytes(raw)
manifest={'classification':'fictional inert wire only','bundle_files':1,'bundle_bytes':len(raw),'bundle_sha256':sha(raw),'literal_count':len(B),'aggregate_decoded_bytes':sum(map(len,B.values())),'literals':{k:{'bytes':len(v),'sha256':sha(v)} for k,v in B.items()}}
(LEDGER/'task-54-d7-cpu-wire-fixture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='literals'}))
