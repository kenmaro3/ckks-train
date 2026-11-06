"""Independent stdlib-only audit of retained NI evidence; never launches HE/Cargo.

Writes only its derived JSON next to this script. Expectations are reconstructed
from the normative original arrays/equations, not imported from Rust or scalar JSON.
"""
from pathlib import Path
import ast
import collections
import copy
import hashlib
import json
import math
import re
import struct

ROOT = Path(__file__).resolve().parents[3]
LEDGER = Path(__file__).resolve().parent
RUN = LEDGER / 'task-54-d4c3-ni-primary-attempt-1'
E = 2.0 ** -12
SCALE = 4796333603149578240
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

meta = json.loads((RUN / 'runner.json').read_text())
manifest = json.loads((RUN / 'source-manifest.json').read_text())
assert meta['returncode'] == 0 and meta['source_commit'] == 'ad607daf3e53c3763c4c5db25701f2ede5115154'
assert meta['command'] == ['cargo', 'test', '-p', 'ckks-train-nn', '--test', 'packed_blocks', 'block_ni_original_fixture', '--', '--ignored', '--exact', '--nocapture', '--test-threads=1']
assert meta['environment'] == dict(CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='8', RAYON_NUM_THREADS='8')
assert meta['cwd'] == str(ROOT)
assert sha(RUN / 'source-manifest.json') == meta['source_manifest_sha256'] == '4788e0af8128c514b36a11cb651fa018ec39db8b7e140425d08be16c4915796e'
assert sha(LEDGER / 'task-54-d4c3-ni-runner.py') == meta['runner_sha256'] == '5c5b8307f9e3206dc926d5a81e98dba5d0088de33eb096526322d8d6948a15e8'
assert sha(LEDGER / 'task-54-d4c3-fix1-rereview.md') == meta['review_sha256'] == '6bb24e383bfd8a82acb6e6335f41db2456b970f3daba6e8f3fb1e2dcbe682f3b'
assert sha(RUN / 'frozen-test-executable') == meta['executable_before_sha256'] == meta['executable_after_sha256'] == '91dfd7ef7bd451cc8e423c1fc7cd4241c379c22e777dcf30c6e15c73b26ee5c8'
assert meta['source_unchanged'] and meta['executable_unchanged']
verified = []
def walk(obj):
    if isinstance(obj, dict):
        if 'path' in obj and 'sha256' in obj:
            p = ROOT / obj['path']
            assert sha(p) == obj['sha256'], str(p)
            if 'bytes' in obj:
                assert p.stat().st_size == obj['bytes']
            verified.append(obj['path'])
        for x in obj.values():
            walk(x)
    elif isinstance(obj, list):
        for x in obj:
            walk(x)
walk(manifest)
for name in ['stdout.txt', 'stderr.txt']:
    assert sha(RUN / name) == meta[name + '_sha256']
    assert (RUN / name).stat().st_size == meta[name + '_bytes']
assert sha(RUN / 'stdout.txt') == 'd8a9f62fb828a2b9f27ce60adb83d626f5835739d4546317d1ddd0012c8a9678'

EPS=2.0**-20
W = [[[5/16, 1/16, 1/16], [1/16, 5/16, 1/16], [1/16, 1/8, 1/4]],
     [[5/16, 1/8, -1/16], [1/16, 5/16, 1/16], [-1/16, 1/16, 1/4]]]
X = [[3/4,3/8,1/2], [3/8,3/4,1/2], [1/2,3/8,3/4]]
Y = [[3/4,0.,0.], [0.,3/4,0.], [3/8,3/8,0.]]
IDS = [[0,1,2], [0], [0,2], [0,1,2]]
RATES = [.04,.01,.02,.08]
def mv(w,x):
    return [sum(w[o][i]*x[i] for i in range(3)) for o in range(3)]
def snap(state, inputs, ids, final=True):
    z = [mv(state['w'][0], x) for x in inputs]
    h = [[v*v+v for v in row] for row in z]
    out = [mv(state['w'][1], row) for row in h] if final else copy.deepcopy(h)
    return dict(input=copy.deepcopy(inputs), z=z, hidden=h, output=out)
def evolve(old,t):
    # Capture all old forward values first; only then mutate a separate candidate.
    candidate = copy.deepcopy(old)
    inputs = [X[i] for i in IDS[t]]
    snaps = []
    for state in old:
        s = snap(state, inputs, IDS[t])
        snaps.append(s)
        inputs = s['hidden']
    for kstage in reversed(range(len(old))):
        state, s, new = old[kstage], snaps[kstage], candidate[kstage]
        d2 = [[row[o]-Y[r][o] for o in range(3)] for row,r in zip(s['output'], IDS[t])]
        d1 = [[sum(state['w'][1][o][i]*d[o] for o in range(3))*(2*z[i]+1) for i in range(3)] for d,z in zip(d2,s['z'])]
        for k,(xx,dd) in enumerate([(s['input'],d1),(s['hidden'],d2)]):
            for o in range(3):
                for i in range(3):
                    g = sum(d[o]*x[i]/len(IDS[t]) for x,d in zip(xx,dd))
                    v = .9*state['v'][k][o][i]+g
                    new['v'][k][o][i] = v
                    new['w'][k][o][i] = state['w'][k][o][i]-RATES[t]*(g+.9*v)
    return candidate, snaps
def vec(v,expanded):
    return [v[s//4 if expanded else s%4] if (s//4 if expanded else s%4)<3 else 0. for s in range(512)]
def matrix(m,row):
    return [(m[s%4][s//4] if row else m[s//4][s%4]) if s//4<3 and s%4<3 else 0. for s in range(512)]

FIELDS=['rotations','relins','ct_ct_tensor','ct_pt_mults','rescales','sse_switches','bootstraps']
BOOT=dict(zip(FIELDS,[37,40,46,207,56,2,1]))
def counter(text):
    m=re.search(r'OpCounter \{ ([^}]+) \}',text)
    assert m,text
    d={k:int(v) for k,v in re.findall(r'(\w+): (\d+)',m[1])}
    assert list(d)==FIELDS
    return d
def plus(a,b): return {k:a[k]+b[k] for k in FIELDS}
def minus(a,b): return {k:a[k]-b[k] for k in FIELDS}
ZERO=dict.fromkeys(FIELDS,0)
md={}; frozen_roles=[]; forward=False
metadata_text=(LEDGER/'task-54-d4c3-fix1-metadata-trace.log').read_text()
for line in metadata_text.splitlines():
    if line.startswith('CASE '):
        m=re.fullmatch(r'CASE G(\d) (Au|Wn|Aa)',line)
        case=(int(m[1]),m[2]);md[case]={'updates':{},'forward':{}}
    elif line.startswith('UPDATE '):
        t=int(line.split()[1]);forward=False
        md[case]['updates'][t]={'counts':counter(line),'stages':{}}
    elif line.startswith('STAGE '):
        m=re.match(r'STAGE (\d+) refresh=(true|false)',line);s=int(m[1])
        levels=[int(v) for v in re.findall(r'level: (\d+)',line)]
        assert len(levels)==4
        md[case]['updates'][t]['stages'][s]=dict(refresh=m[2]=='true',levels=levels,trace={})
    elif line.startswith('INFERENCE '):
        forward=True;md[case]['inference_counts']=counter(line)
    elif line.startswith('FORWARD '):
        s=int(line.split()[1]);md[case]['forward'][s]={'counts':counter(line),'trace':{}}
    elif line.startswith(('TRACE ','FORWARD_TRACE ')):
        m=re.search(r'sample: (None|Some\((\d+)\)), name: "([^"]+)", level: (\d+)',line)
        target=md[case]['forward'][s] if forward else md[case]['updates'][t]['stages'][s]
        target['trace'][(int(m[2]) if m[2] else None,m[3])]=int(m[4])
    elif line.startswith('ROLE '): frozen_roles.append(line)
assert list(md)==[(g,p) for g in [2,3] for p in ['Au','Wn','Aa']]
assert len(frozen_roles)==332

protocol=[]; expected=[]; attempts=[]; events={}; current_case=None
def literal(l):protocol.append(('literal',l))
def numeric(kind,label):protocol.append((kind,str(label)))
def add(label,values,shape,family,level,bound=None,parity=True):
    protocol.append(('SLOTS',label))
    if parity:protocol.append(('PARITY',label))
    expected.append(dict(label=label,expected=values,shape=shape,family=family,level=level,bound=bound,case=current_case,parity=parity))
def role_values(role,ss,snaps,ids):
    m=re.fullmatch(r'(Weight|Velocity)\((\d+)\)',role)
    if m:
        k=int(m[2]);s=k//2; layer=k%2
        matrix_values=ss[s]['w'][layer] if m[1]=='Weight' else [[.08*1.9*x for x in r] for r in ss[s]['v'][layer]]
        return matrix(matrix_values,(s+layer)%2==0),None,('W' if m[1]=='Weight' else 'U'),(.4 if m[1]=='Weight' else .1)
    m=re.fullmatch(r'Hidden \{ stage: (\d+), record: (\d+) \}',role)
    s,r=map(int,m.groups());return vec(snaps[s]['hidden'][ids.index(r)],s%2==1),s%2==1,'hidden',None
def identity(a):
    update='None' if a['update'] is None else f"Some({a['update']})"
    record='None' if a['record'] is None else f"Some({a['record']})"
    return f"AttemptIdentity {{ ordinal: {a['ordinal']}, graph: {a['graph']}, placement: {a['placement']}, phase: {a['phase']}, update: {update}, stage: {a['stage']}, record: {record}, role: {a['role']} }}"
def new_attempt(role,t,s,record,ss,snaps,ids):
    values,shape,family,bound=role_values(role,ss,snaps,ids)
    phase='Training' if t is not None else 'Inference'
    if family=='hidden':bound=.5 if phase=='Training' else .6
    a=dict(ordinal=len(attempts),graph=current_case[0],placement=current_case[1],phase=phase,update=t,stage=s,record=record,role=role,values=values,shape=shape,family=family,bound=bound)
    attempts.append(a);return a
def attempt_diagnostics(group):
    for a in group:
        i=a['ordinal'];fam=a['phase'].lower()+'_'+a['family']
        add(f'attempt={i} input',a['values'],a['shape'],'boot_input_'+fam,0,a['bound'],False)
        numeric('ATTEMPT_INTERVAL',i);numeric('ATTEMPT',i)
        add(f'attempt={i} output',a['values'],a['shape'],'boot_output_'+fam,16,a['bound'],False)
        numeric('INCREMENTAL',i)
def state(label,ss,levels):
    for s,st in enumerate(ss):
        for k in range(2):
            add(f'{label} stage={s} W{k}',matrix(st['w'][k],(s+k)%2==0),None,'W',levels[s][k],.4)
            if st['has_u']:
                u=[[.08*1.9*x for x in row] for row in st['v'][k]]
                add(f'{label} stage={s} U{k}',matrix(u,(s+k)%2==0),None,'U',levels[s][2+k],.1)
def event(label,index,a,sourcelevel):
    name=f"{label} event={index} global=Some({identity(a)}) role={a['role']}"
    assert a['ordinal'] not in events;events[a['ordinal']]=name
    fam='event_'+a['phase'].lower()+'_'+a['family']
    add(name+' source',a['values'],a['shape'],fam+'_source',sourcelevel,a['bound'])
    if a['family']!='hidden':
        add(name+' projection_source',a['values'],a['shape'],fam+'_projection_source',sourcelevel,a['bound'],False)
        add(name+' projected_raw',a['values'],a['shape'],fam+'_projected_raw',sourcelevel-1,a['bound'],False)
        numeric('PROJECTION',name)
        add(name+' projected',a['values'],a['shape'],fam+'_projected',sourcelevel-1,a['bound'])
    add(name+' input',a['values'],a['shape'],fam+'_input',0,a['bound'])
    add(name+' output',a['values'],a['shape'],fam+'_output',16,a['bound'])
    add(name+' range',a['values'],a['shape'],fam+'_range',0,a['bound'],False)
    numeric('EVENT',name)

committed=0
for current_case in md:
    g,placement=current_case
    ss=[dict(w=copy.deepcopy(W),v=[[[0.]*3 for _ in range(3)] for _ in range(2)],has_u=False) for _ in range(g)]
    levels=[[16]*4 for _ in range(g)]
    for t,ids in enumerate(IDS):
        label=f'NI G{g} {placement} t={t}';literal('BEGIN '+label)
        old=copy.deepcopy(ss);oldlevels=levels
        ss,snaps=evolve(old,t)
        for st in ss:st['has_u']=True
        plan=md[current_case]['updates'][t];child_events=[[] for _ in range(g)];boundary_events={};group=[]
        for s in range(g):
            stageplan=plan['stages'][s]
            assert stageplan['refresh']==(placement!='Aa' or t in [1,3])
            if placement=='Aa':
                for r in ids:
                    a=new_attempt(f'Hidden {{ stage: {s}, record: {r} }}',t,s,r,ss,snaps,ids);group.append(a);child_events[s].append(a)
            if stageplan['refresh']:
                for k in range(2):
                    for kind in ['Weight','Velocity']:
                        a=new_attempt(f'{kind}({2*s+k})',t,s,None,ss,snaps,ids);group.append(a);child_events[s].append(a)
            if s+1<g and placement!='Aa':
                for r in ids:
                    a=new_attempt(f'Hidden {{ stage: {s}, record: {r} }}',t,s,r,ss,snaps,ids);group.append(a);boundary_events[(s,r)]=a
        assert len(group)==plan['counts']['bootstraps']
        attempt_diagnostics(group);state(label+' old',old,oldlevels)
        for s in range(g):
            trace=plan['stages'][s]['trace']
            literal(f'TRACE_FLAGS {label} stage={s} head_affines={len(ids)} residuals={len(ids)} outer_sums=2 state_updates=2')
            for j,r in enumerate(ids):
                add(f'{label} s={s} id={r} hidden_pre',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'training_hidden_pre',trace[(r,'hidden')],.5)
                if placement=='Aa':add(f'{label} s={s} id={r} hidden_post',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'training_hidden_post',16,.5)
                add(f'{label} s={s} id={r} output',vec(snaps[s]['output'][j],s%2==0),s%2==0,'training_output',trace[(r,'output')])
            for k in range(2):
                for role,word in [('W','weight'),('U','velocity')]:
                    value=ss[s]['w'][k] if role=='W' else [[.08*1.9*x for x in row] for row in ss[s]['v'][k]]
                    add(f'{label} candidate stage={s} {role}{k}',matrix(value,(s+k)%2==0),None,role,trace[(None,word+str(k))],.4 if role=='W' else .1)
            for j,a in enumerate(child_events[s]):
                sourcelevel=trace[(a['record'],'hidden')] if a['family']=='hidden' else trace[(None,('weight' if a['family']=='W' else 'velocity')+str(int(re.search(r'\d+',a['role'])[0])%2))]
                event(label,j,a,sourcelevel)
        for s in range(g-1):
            trace=plan['stages'][s]['trace']
            for j,r in enumerate(ids):
                for suffix in ['source','output']:
                    level=16 if placement=='Aa' or suffix=='output' else trace[(r,'hidden')]
                    add(f'{label} handoff s={s} id={r} {suffix}',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'training_handoff',level,.5)
                if placement!='Aa':event(label,0,boundary_events[(s,r)],trace[(r,'hidden')])
        levels=[plan['stages'][s]['levels'] for s in range(g)]
        state(label+' committed_candidate',ss,levels)
        numeric('UPDATE_COUNTS',label);committed+=1
        literal(f'GRAPH_COMMIT {label} count={committed} indices={[t+1]*g}')
    snaps=[];inputs=X
    for s,st in enumerate(ss):
        z=snap(st,inputs,[0,1,2],s+1==g);snaps.append(z);inputs=z['hidden']
    group=[];child_events=[[] for _ in range(g)]
    for s in range(g):
        if s+1<g or placement=='Aa':
            for r in range(3):
                a=new_attempt(f'Hidden {{ stage: {s}, record: {r} }}',None,s,r,ss,snaps,[0,1,2]);group.append(a);child_events[s].append(a)
    assert len(group)==md[current_case]['inference_counts']['bootstraps'];attempt_diagnostics(group)
    label='NI inference'
    for s in range(g):
        trace=md[current_case]['forward'][s]['trace'];boot=bool(child_events[s])
        literal(f'INFERENCE_TRACE_FLAGS {label} stage={s} head_affines={3 if s+1==g else 0} loss_outer_update_derivative_boundaries=0 graph_commits=0')
        for r in range(3):
            for suffix,key,expanded,level in [('input','input',s%2==0,16),('z','z',s%2==1,trace[(r,'z1')]),('path_output','output',s%2==0 if s+1==g else s%2==1,trace[(r,'output')] if s+1==g else (16 if boot else trace[(r,'hidden')])),('hidden_pre','hidden',s%2==1,trace[(r,'hidden')])]:
                bound=.6 if key=='hidden' or (key=='output' and s+1<g) else None
                add(f'{label} s={s} id={r} {suffix}',vec(snaps[s][key][r],expanded),expanded,'inference_'+suffix,level,bound)
            if boot:add(f'{label} s={s} id={r} hidden_post',vec(snaps[s]['hidden'][r],s%2==1),s%2==1,'inference_hidden_post',16,.6)
        for j,a in enumerate(child_events[s]):event(label,j,a,trace[(a['record'],'hidden')])
    for r in range(3):add(f'{label} final id={r}',vec(snaps[-1]['output'][r],(g-1)%2==0),(g-1)%2==0,'inference_final',md[current_case]['forward'][g-1]['trace'][(r,'output')])
    state('post inference unchanged',ss,levels)
assert len(attempts)==332 and len(expected)==4050 and len(events)==332

# Cross-check independently generated roles against the preregistered metadata log.
for a,l in zip(attempts,frozen_roles):
    i=a['ordinal'];g=a['graph'];p=a['placement'];s=a['stage'];r=a['record'];t=a['update']
    if a['phase']=='Inference':prefix=f'ROLE ordinal={i} G{g} {p} inference stage={s} record={r} cost='
    elif a['family']=='hidden':prefix=f'ROLE ordinal={i} G{g} {p} update={t} stage={s} '+('hidden_record=' if p=='Aa' else 'boundary_record=')+str(r)+' cost='
    else:prefix=f"ROLE ordinal={i} G{g} {p} update={t} {a['role']} inclusive_cost="
    assert l.startswith(prefix),(a,l)
    expected_cost=BOOT if a['family']=='hidden' else plus(BOOT,dict(zip(FIELDS,[0,0,0,1,1,0,0])))
    assert counter(l)==expected_cost

slot_re=re.compile(r'SLOTS (.+) context=(0x[0-9a-f]+) level=(\d+) scale_bits=(\d+) components=(\d+) values=(.+)')
parity_re=re.compile(r'PARITY (.+) max=(\S+) worst=(\d+) active=(\S+) padding=(\S+) replica=(\S+) imaginary=(\S+)')
records=[];actual_protocol=[];numeric_lines={};contexts=set();linecounts=collections.Counter();params=None;harness=[]
for line in (RUN/'stdout.txt').open():
    line=line.rstrip('\n');kind=line.split(' ',1)[0];linecounts[kind]+=1
    assert not re.search(r'FAILED|UNDECODABLE|panicked|\bNaN\b|\binf\b',line)
    if not line:continue
    if kind=='SLOTS':
        m=slot_re.fullmatch(line);assert m
        label,ctx,level,scale,components,payload=m.groups()
        assert int(scale)==SCALE and int(components)==2
        contexts.add(ctx);bits=ast.literal_eval(payload)
        assert len(bits)==512 and all(isinstance(z,tuple) and len(z)==2 and all(isinstance(b,int) and 0<=b<2**64 for b in z) for z in bits)
        flat=[b for z in bits for b in z];packed=struct.pack('>1024Q',*flat);doubles=struct.unpack('>1024d',packed)
        assert all(math.isfinite(x) for x in doubles)
        values=[complex(doubles[i],doubles[i+1]) for i in range(0,1024,2)]
        records.append(dict(label=label,values=values,payload_sha256=hashlib.sha256(packed).hexdigest(),level=int(level)))
        actual_protocol.append(('SLOTS',label))
    elif kind=='PARITY':
        m=parity_re.fullmatch(line);assert m and records[-1]['label']==m[1]
        summary=dict(zip(['max','worst','active','padding','replica','imaginary'],[float(m[2]),int(m[3]),float(m[4]),float(m[5]),float(m[6]),float(m[7])]))
        assert all(math.isfinite(x) for x in summary.values()) and summary['max']<=E
        records[-1]['summary']=summary;actual_protocol.append(('PARITY',m[1]))
    elif kind in ['BEGIN','TRACE_FLAGS','GRAPH_COMMIT','INFERENCE_TRACE_FLAGS']:actual_protocol.append(('literal',line))
    elif kind in ['ATTEMPT_INTERVAL','ATTEMPT','INCREMENTAL']:
        label=line.split(' ',2)[1];actual_protocol.append((kind,label));assert (kind,label) not in numeric_lines;numeric_lines[(kind,label)]=line
    elif kind in ['PROJECTION','EVENT','UPDATE_COUNTS']:
        split={'PROJECTION':' drift=','EVENT':' range=','UPDATE_COUNTS':' OpCounter '}[kind]
        label=line[len(kind)+1:].split(split)[0];actual_protocol.append((kind,label));assert (kind,label) not in numeric_lines;numeric_lines[(kind,label)]=line
    else:
        assert kind in ['running','test','NI_END','ok'],line[:150]
        harness.append(line)
        if 'PARAMS ' in line:params=line.split('PARAMS ',1)[1]
assert actual_protocol==protocol,next(((i,a,b) for i,(a,b) in enumerate(zip(actual_protocol,protocol)) if a!=b),('length',len(actual_protocol),len(protocol)))
assert len(records)==4050 and len(contexts)==1
assert len(harness)==5 and harness[0]=='running 1 test' and harness[1].startswith('test block_ni_original_fixture ... PARAMS ')
assert harness[2]=='NI_END success=true cases=6 committed=24 inferences=6 attempted=332 completed=332 validated=332 expected=332 cap=372'
assert harness[3]=='ok' and re.fullmatch(r'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 22 filtered out; finished in [\d.]+s',harness[4])
assert params.startswith('ni=true N=1024 scale_bits=4796333603149578240 dnum=8 ') and params.endswith('held_out=167')
assert 'bp=BootParams { log_slots: 9, ctos_levels: 4, evalmod_levels: 7, stoc_levels: 4, k_intervals: 13, cheb_degree: 127, double_angle: 0, sparse_hw: None, ephemeral_hw: Some(32) }' in params
for name,bits in [('q',[54]+[42]*16+[34]*4+[58]*7+[56]*4),('p',[60]*4)]:
    values=ast.literal_eval(re.search(r'\b'+name+r'=(\[[^]]+\])',params)[1]);frozen=ast.literal_eval(re.search(r'\b'+name+r'=(\[[^]]+\])',metadata_text)[1])
    assert values==frozen and [v.bit_length() for v in values]==bits and all(v%2048==1 for v in values)

rows=[];summary_diff=dict.fromkeys(['max','active','padding','replica','imaginary'],0.);worst_disagreements=[];by_label={}
for index,(a,e) in enumerate(zip(records,expected)):
    assert a['label']==e['label'] and a['level']==e['level'],(index,a['label'],a['level'],e['level'])
    assert ('summary' in a)==e['parity']
    v=a['values'];shape=e['shape'];errors=[math.hypot(z.real-w,z.imag) for z,w in zip(v,e['expected'])]
    used=[(s//4<3 and s%4<3) if shape is None else ((s//4 if shape else s%4)<3) for s in range(512)]
    metrics=dict(max=max(errors),worst=max(range(512),key=errors.__getitem__),active=max(x for x,u in zip(errors,used) if u),padding=max(x for x,u in zip(errors,used) if not u),imaginary=max(abs(z.imag) for z in v),replica=0.)
    if shape is not None:
        metrics['replica']=max(math.hypot(z.real-v[4*(s//4) if shape else s%4].real,z.imag-v[4*(s//4) if shape else s%4].imag) for s,z in enumerate(v) if used[s])
    magnitude=max(math.hypot(z.real,z.imag) for z in v)
    assert metrics['max']<=E,(a['label'],metrics['max'])
    if e['bound'] is not None:assert magnitude<=e['bound']
    if 'summary' in a:
        for k in summary_diff:summary_diff[k]=max(summary_diff[k],abs(metrics[k]-a['summary'][k]))
        if metrics['worst']!=a['summary']['worst']:worst_disagreements.append(index)
    row=dict(index=index,case=list(e['case']),label=a['label'],level=a['level'],family=e['family'],bound=e['bound'],magnitude=magnitude,has_reported_parity=e['parity'],payload_sha256=a['payload_sha256'],**metrics)
    rows.append(row);a['derived']=row
    assert (e['case'],a['label']) not in by_label,(e['case'],a['label'])
    by_label[(e['case'],a['label'])]=a
assert not worst_disagreements and summary_diff['imaginary']==0.
def get(case,label):return by_label[(case,label)]
def distance(a,b):return max(math.hypot(x.real-y.real,x.imag-y.imag) for x,y in zip(a['values'],b['values']))
identity_checks=0
def identical(a,b):
    global identity_checks
    assert a['payload_sha256']==b['payload_sha256'] and a['level']==b['level'],(a['label'],b['label'])
    identity_checks+=1
attempt_results=[];projection_results=[];last_after=ZERO;scalar_diffs={k:0. for k in ['range','incremental','projection']}
for a in attempts:
    i=a['ordinal'];case=(a['graph'],a['placement']);name=events[i]
    inp=get(case,f'attempt={i} input');out=get(case,f'attempt={i} output')
    inc=distance(inp,out);assert inc<=EPS
    line=numeric_lines[('INCREMENTAL',str(i))]
    m=re.fullmatch(r'INCREMENTAL \d+ error=(\S+) epsilon=(\S+)',line);assert float(m[2])==EPS and float(m[1])<=EPS
    scalar_diffs['incremental']=max(scalar_diffs['incremental'],abs(float(m[1])-inc))
    interval=counter(numeric_lines[('ATTEMPT_INTERVAL',str(i))]);assert interval==BOOT
    line=numeric_lines[('ATTEMPT',str(i))];prefix=f'ATTEMPT {i} identity={identity(a)} '
    assert line.startswith(prefix) and line.endswith('completed=true validated=true')
    m=re.search(r'bound=(\S+) range=(\S+) before=(OpCounter \{[^}]+\}) after=(OpCounter \{[^}]+\})',line)
    bound,value=float(m[1]),float(m[2]);before,after=counter(m[3]),counter(m[4])
    assert bound==a['bound'] and value<=bound and value<=1.
    scalar_diffs['range']=max(scalar_diffs['range'],abs(value-inp['derived']['magnitude']))
    assert minus(after,before)==interval and all(before[k]>=last_after[k] for k in FIELDS)
    assert before['bootstraps']==i and after['bootstraps']==i+1;last_after=after
    line=numeric_lines[('EVENT',name)];m=re.search(r' range=(\S+) bound=(\S+)',line)
    assert float(m[2])==bound and counter(line)==BOOT
    scalar_diffs['range']=max(scalar_diffs['range'],abs(float(m[1])-inp['derived']['magnitude']))
    for suffix in ['input','range']:identical(inp,get(case,name+' '+suffix))
    identical(out,get(case,name+' output'))
    source=get(case,name+' source')
    if a['family']!='hidden':
        identical(source,get(case,name+' projection_source'));identical(get(case,name+' projected_raw'),get(case,name+' projected'))
        drift=distance(source,get(case,name+' projected_raw'));line=numeric_lines[('PROJECTION',name)]
        scalar_diffs['projection']=max(scalar_diffs['projection'],abs(float(line.rsplit('=',1)[1])-drift))
        projection_results.append(dict(ordinal=i,drift=drift))
        layer=int(re.search(r'\d+',a['role'])[0])%2;role=a['family']+str(layer);base=f"NI G{case[0]} {case[1]} t={a['update']}"
        identical(source,get(case,f"{base} candidate stage={a['stage']} {role}"))
        identical(out,get(case,f"{base} committed_candidate stage={a['stage']} {role}"))
    else:
        s=a['stage'];r=a['record'];base=f"NI G{case[0]} {case[1]} t={a['update']}" if a['phase']=='Training' else 'NI inference'
        identical(source,get(case,f'{base} s={s} id={r} hidden_pre'))
        if a['phase']=='Inference' or case[1]=='Aa':identical(out,get(case,f'{base} s={s} id={r} hidden_post'))
        if a['phase']=='Training' and s+1<case[0]:
            identical(out,get(case,f'{base} handoff s={s} id={r} output'))
            identical(out if case[1]=='Aa' else source,get(case,f'{base} handoff s={s} id={r} source'))
    attempt_results.append(dict(**{k:v for k,v in a.items() if k not in ['values','shape']},input_range=inp['derived']['magnitude'],input_parity=inp['derived']['max'],output_parity=out['derived']['max'],incremental_max=inc,before=before,after=after,interval=interval))

update_results=[]
for case,plan in md.items():
    g,p=case
    for t in range(4):
        label=f'NI G{g} {p} t={t}';actual=counter(numeric_lines[('UPDATE_COUNTS',label)])
        assert actual==plan['updates'][t]['counts']
        calls=[a for a in attempt_results if (a['graph'],a['placement'],a['update'])==(g,p,t)]
        assert len(calls)==actual['bootstraps']
        update_results.append(dict(graph=g,placement=p,update=t,actual_counts=actual))
        for s in range(g):
            for role in ['W0','U0','W1','U1']:
                committed=get(case,f'{label} committed_candidate stage={s} {role}')
                next_label=f'NI G{g} {p} t={t+1} old stage={s} {role}' if t<3 else f'post inference unchanged stage={s} {role}'
                identical(committed,get(case,next_label))
                if not plan['updates'][t]['stages'][s]['refresh']:identical(committed,get(case,f'{label} candidate stage={s} {role}'))
    # Inference hands each selected payload to the next child without a separate
    # handoff record in this path. Verify decoded payload identity directly.
    for s in range(g-1):
        for r in range(3):
            selected=get(case,f'NI inference s={s} id={r} hidden_post')
            identical(selected,get(case,f'NI inference s={s} id={r} path_output'))
            identical(selected,get(case,f'NI inference s={s+1} id={r} input'))

families={};case_stats={};role_stats={}
for row in rows:
    for group,key in [(families,row['family']),(case_stats,f'G{row["case"][0]} {row["case"][1]}')]:
        item=group.setdefault(key,dict(count=0,max_error=0.,max_magnitude=0.,max_imaginary=0.,max_padding=0.,max_replica=0.))
        item['count']+=1
        for name,field in [('max_error','max'),('max_magnitude','magnitude'),('max_imaginary','imaginary'),('max_padding','padding'),('max_replica','replica')]:item[name]=max(item[name],row[field])
for a in attempt_results:
    key=f"G{a['graph']} {a['placement']} {a['phase']} stage{a['stage']} {a['family']}"
    st=role_stats.setdefault(key,dict(count=0,bound=a['bound'],max_range=0.,max_incremental=0.,max_input_parity=0.,max_output_parity=0.));st['count']+=1
    for field,src in [('max_range','input_range'),('max_incremental','incremental_max'),('max_input_parity','input_parity'),('max_output_parity','output_parity')]:st[field]=max(st[field],a[src])
result=dict(status='PASS_retained_NI_scope',source_commit=meta['source_commit'],manifest_entries_verified=len(verified),owned_source_entries=len(manifest['owned_source']),script_sha256=sha(Path(__file__)),stdout_sha256=sha(RUN/'stdout.txt'),stdout_bytes=(RUN/'stdout.txt').stat().st_size,parameter_line=params,E=E,epsilon_boot=EPS,line_counts=dict(linecounts),slots=4050,complex_slots=4050*512,finite_f64_components=4050*1024,parity_summaries=2654,independently_checked_raw_without_summary=1396,attempts=attempt_results,projections=projection_results,updates=update_results,exact_payload_identity_checks=identity_checks,global_max_error=max(rows,key=lambda x:x['max']),global_max_incremental=max(attempt_results,key=lambda x:x['incremental_max']),summary_max_absolute_differences=summary_diff,summary_worst_index_disagreements=worst_disagreements,other_summary_max_absolute_differences=scalar_diffs,level_histogram=dict(collections.Counter(r['level'] for r in rows)),families=families,cases=case_stats,roles=role_stats,rows=rows)
output=LEDGER/'task-54-d4c3-ni-runtime-audit.json';output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['status','manifest_entries_verified','slots','complex_slots','finite_f64_components','line_counts','global_max_error','global_max_incremental','summary_max_absolute_differences','other_summary_max_absolute_differences','exact_payload_identity_checks','cases']},indent=2))
print('derived_json_sha256',sha(output))

