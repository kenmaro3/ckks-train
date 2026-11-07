"""Independent finite fictional oracle. No project imports, native calls or science.
Grammar: t/b=u64LE byte count+bytes; n=u64LE; tag=u8; list=n+items;
option=0 or1+value; header=t(ENC)+t(domain)+tag(synthetic). FileRef is
root:u8,path:t,length:n,sha256:t. Clock=0,n(unix),0,n(elapsed).
All outputs below are the COMPLETE finite input/output inventory. Never import
this in production. Writes one compact source literal bundle plus ledger manifest.
"""
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'crates/ckks-train-bench/src/study_reference/document/evidence/observed/literals.json'
ENC='hekit-study-reference-encoding-v1'
def n(x):return struct.pack('<Q',x)
def t(x):
 x=x.encode() if isinstance(x,str) else x
 return n(len(x))+x
def tag(x):return bytes([x])
def sha(x):return hashlib.sha256(x).hexdigest()
def h(x):return t(x*64)
def header(x,v=1):return t(ENC)+t('hekit-test-'+x)+tag(v)
def clock(x=1):return b'\0'+n(100+x)+b'\0'+n(x)
def f(path='fixture',root=2,data=b'x'):return tag(root)+t(path)+n(len(data))+t(sha(data))
def files(*xs):return n(len(xs))+b''.join(xs)
B={}
def put(k,x):B[k]=x;return x
request={}
for mode in (0,1):
 for scope in (0,1):
  key=f'W{1+mode*2+scope:02}'
  req=tag(scope)+b'\0\0\1\1\1'+b''.join(n(x) for x in (67108864,1000000,2,2,2400 if scope==0 else 2,0,1))
  request[mode,scope]=req
  put(key,header('resource-requested-work-v1')+b'\0\0'+t('main_i' if mode==0 else 'main_ni')+t('iris__emlp1')+n(0)+b'\0\0'+tag(mode)+tag(mode)+b'\0'+t('study-train-cpu-serial-linux-clean-v2')+t(req)+b'\0')
for op in range(6):put(f'W{op+5:02}',header('resource-requested-work-v1')+b'\1'+tag(op)+t('iris__emlp1')+n(1)+n(0))
# Fake whole-file/payload identity stubs are descriptive ONLY, never authorities.
sup=t('linux-pidfd-broker-v1')+h('a')+t('00000000-0000-0000-0000-000000000001')+n(10)+n(20)+n(11)+n(21)+t('claim1')+f('initial-sample.bin')+n(94371840)+n(10000000000)+n(5000000000)+n(4)
put('supervisor-binding',sup)
put('bootstrap',header('resource-observed-bootstrap-v1')+f('resource/linux-cpu-observed-v1/policy.json')+h('a')+t('00000000-0000-0000-0000-000000000001')+f('namespace')+t('task54-host-serial-v1')+h('b')+t('c'*40)+h('d')+t('study-train-cpu-serial-linux-clean-v2'))
pending=put('pending',header('resource-observed-pending-v1')+t('claim1')+f('policy.json')+h('a')+t('00000000-0000-0000-0000-000000000001')+n(10)+n(20)+clock())
claim=put('claim',header('resource-observed-claim-v1')+f('pending.bin',data=pending)+t(B['W01'])+h('e')+h('b')+t('c'*40)+clock(2)+sup)
put('host-pending',header('host-claim-pending-v1')+t('claim1')+t('00000000-0000-0000-0000-000000000001')+n(10)+n(20)+t(b'/fictional/observations')+n(1)+n(2))
put('host-closed',header('host-claim-closed-v1')+h('f')+f('supervisor.end.bin'))
sample=n(0)+clock(1)+clock(2)+tag(0)+files(n(10)+n(20)+n(4),n(11)+n(21)+n(5))+n(9)+b'\1\0'
put('initial-sample',header('resource-observed-initial-sample-v1')+sample+f('pending.bin',data=pending))
put('samples',header('resource-observed-samples-v1')+f('claim.bin',data=claim)+struct.pack('<I',len(sample))+sample)
put('supervisor-end',header('supervisor-end-v1')+f('pending.bin',data=pending)+b'\1'+f('claim.bin',data=claim)+b'\1'+f('start.bin')+b'\1'+f('result.bin')+b'\1'+f('end.bin')+b'\1'+f('samples.bin')+n(0)+t('peak_rss_kb_tree_sampled_10s')+b'\1'+n(9)+tag(0)+b'\1'+n(0)+b'\1'+clock(10))
put('full-end',header('he-end-v1')+t('attempt1')+f('admission.bin')+f('start.bin')+b'\1'+f('result.bin')+b'\1'+f('result.json')+b'\1'+f('export.json',5)+tag(0)+n(2400)+n(48000)+n(5)+clock(9))
put('export-id',t(ENC)+t('hekit-test-result-export-id-v1')+tag(1)+h('d'))
# Isolated demand, arithmetic attribution is specified by retained/additional total.
put('demand',tag(2)+n(24)+n(16)+files(t('raw_0')+n(3)+n(8),t('result_0')+n(2)+n(8)))
# Result fixture holds no scientific measurements: fictional complete Quick with 2 endpoints.
case=t(ENC)+t('hekit-study-registered-case-v1')+h('a')+tag(0)+t('main_i')+t('iris__emlp1')+n(0)+b'\0\0\0\0\0\0\0'+n(20)+b'\0'
put('case-i',case)
case_ni=case.replace(t('main_i'),t('main_ni'))
# mode/placement/security in case grammar are independent fixed tags.
off=len(t(ENC)+t('hekit-study-registered-case-v1')+h('a')+tag(0)+t('main_ni')+t('iris__emlp1')+n(0))
case_ni=case_ni[:off]+b'\0\0\0\1\1\0\1'+case_ni[off+7:]
put('case-ni',case_ni)
# Execution bytes will be supplied to strict same-case linkage tests from unchanged corpus;
# here result validates syntax and requires supplied actual evidence/start at closure.
def endpoint(u):return n(u)+n(u*20)+n(0)+b''.join(n(x) for x in (1,2,3,4,5,6,7))+b''.join(n(x) for x in (3,u*10,1,0,2,3,4,u*20,0))+n(2)+n(2)+n(30)+n(0)+n(0)
e1,e2=endpoint(1),endpoint(2)
put('result',header('cpu-result-v1')+tag(1)+t(case)+t(b'fictional-execution')+t('attempt1')+f('preflight.bin')+f('start.bin')+b''.join(f(x+'.bin') for x in ('inputs','implementation','layout','resource','user'))+t('d7-resource-observed-rss-v1')+h('b')+t('c'*40)+h('d')+h('e')+h('f')+h('a')+h('b')+tag(0)+n(2)+n(40)+b'\0\0'+files(e1,e2)+b'\1'+e2+b'\1'+h('c')+b'\0'+clock(8))
assert len(B)<=128 and max(map(len,B.values()))<=16384 and sum(map(len,B.values()))<=131072
raw=(json.dumps({k:v.hex() for k,v in B.items()},sort_keys=True,separators=(',',':'))+'\n').encode()
OUT.write_bytes(raw)
manifest={'classification':'fictional inert wire only','bundle_files':1,'bundle_bytes':len(raw),'bundle_sha256':sha(raw),'literal_count':len(B),'aggregate_decoded_bytes':sum(map(len,B.values())),'literals':{k:{'bytes':len(v),'sha256':sha(v)} for k,v in B.items()}}
(ROOT/'.superpowers/sdd/2026-11-04-he-training-implementation-plan-m3-m5/task-54-d7-cpu-wire-fixture-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='literals'}))
