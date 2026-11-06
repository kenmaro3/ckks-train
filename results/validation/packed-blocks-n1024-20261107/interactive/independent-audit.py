"""Independent stdlib-only audit of retained I evidence; never launches HE/Cargo.

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
RUN = LEDGER / 'task-54-d4c3-i-primary-attempt-1'
E = 2.0 ** -12
SCALE = 4796333603149578240
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

meta = json.loads((RUN / 'runner.json').read_text())
manifest = json.loads((RUN / 'source-manifest.json').read_text())
assert meta['returncode'] == 0 and meta['source_commit'] == 'ad607daf3e53c3763c4c5db25701f2ede5115154'
assert meta['command'] == ['cargo', 'test', '-p', 'ckks-train-nn', '--test', 'packed_blocks', 'block_interactive_original_fixture', '--', '--ignored', '--exact', '--nocapture', '--test-threads=1']
assert meta['environment'] == dict(CARGO_INCREMENTAL='0', CARGO_BUILD_JOBS='8', RAYON_NUM_THREADS='8')
assert meta['cwd'] == str(ROOT)
assert sha(RUN / 'source-manifest.json') == meta['source_manifest_sha256'] == '4788e0af8128c514b36a11cb651fa018ec39db8b7e140425d08be16c4915796e'
assert sha(LEDGER / 'task-54-d4c3-i-runner.py') == meta['runner_sha256'] == 'd8770ace57c2cd6de0ca1a7f17892cdf3de14e1616d425b3aa9dd13f6cc9962c'
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
assert sha(RUN / 'stdout.txt') == '7256d4572d8da7e8b87dad7f193106500a998d3d245f116eb7365a2ef8593afd'
raw = (RUN / 'stdout.txt').read_text()
lines = raw.splitlines()
assert not re.search(r'FAILED|UNDECODABLE|panicked|\bNaN\b|\binf\b', raw)
assert lines.count('running 1 test') == 1
assert len([l for l in lines if l.startswith('test block_')]) == 1
assert len([l for l in lines if l.startswith('test result: ok. 1 passed; 0 failed; 0 ignored;')]) == 1
assert lines.count('ok') == 1
assert 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 22 filtered out; finished in 3.27s' in lines
assert all(not l or l.startswith(('running 1 test','test block_interactive_original_fixture ... PARAMS ','BEGIN ','SLOTS ','PARITY ','TRACE_FLAGS ','GRAPH_COMMIT ','INFERENCE_TRACE_FLAGS ','I_END ','test result: ')) or l=='ok' for l in lines)
assert lines.count('I_END success=true graphs=2 committed=8 inferences=2 attempted=116 completed=116 validated=116') == 1
params = next(l.split('PARAMS ', 1)[1] for l in lines if 'PARAMS ' in l)
assert params.startswith('ni=false N=1024 scale_bits=4796333603149578240 dnum=8 ')
assert params.endswith('held_out=167')
assert 'bp=BootParams { log_slots: 9, ctos_levels: 4, evalmod_levels: 7, stoc_levels: 4, k_intervals: 13, cheb_degree: 127, double_angle: 0, sparse_hw: None, ephemeral_hw: Some(32) }' in params
frozen_metadata = (LEDGER / 'task-54-d4c3-fix1-metadata-trace.log').read_text()
for name, bits in [('q', [54]+[42]*16+[34]*4+[58]*7+[56]*4), ('p', [60]*4)]:
    vals = ast.literal_eval(re.search(r'\b'+name+r'=(\[[^]]+\])', params)[1])
    frozen = ast.literal_eval(re.search(r'\b'+name+r'=(\[[^]]+\])', frozen_metadata)[1])
    assert vals == frozen and [v.bit_length() for v in vals] == bits
    assert all(v % 2048 == 1 for v in vals)

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

# Build the entire expected ordered diagnostic protocol, preserving intentional
# duplicate candidate records and graph-less inference labels via enclosing order.
expected=[]
protocol=[]
client_specs=[]
current_g=None
def add(label, values, shape, family, bound=None):
    expected.append(dict(label=label, expected=values, shape=shape, family=family, graph=current_g, bound=bound))
    protocol.append(('slot',label))
def literal(line):
    protocol.append(('line',line))
def stage(label,state,s):
    for k in range(2):
        add(f'{label} W{k}', matrix(state['w'][k],(s+k)%2==0),None,'W',.4)
        if state['has_u']:
            u=[[.08*1.9*x for x in row] for row in state['v'][k]]
            add(f'{label} U{k}',matrix(u,(s+k)%2==0),None,'U',.1)
def states(label,ss):
    for s,st in enumerate(ss):
        stage(f'{label} stage={s}',st,s)
def call(role,values,shape,family,bound,phase,t,s,r=None):
    index=len(client_specs)
    client_specs.append(dict(index=index,graph=current_g,phase=phase,update=t,stage=s,record=r,role=role,family=family,bound=bound))
    add(f'client={index} role={role} input',values,shape,'client_'+phase+'_'+family+'_input',bound)
    add(f'client={index} output',values,shape,'client_'+phase+'_'+family+'_output',bound)
def hidden_calls(snaps,s,ids,phase,t):
    for j,r in enumerate(ids):
        call(f'Hidden {{ stage: {s}, record: {r} }}',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'hidden',.5 if phase=='training' else .6,phase,t,s,r)
committed=0
for current_g in [2,3]:
    ss=[dict(w=copy.deepcopy(W),v=[[[0.]*3 for _ in range(3)] for _ in range(2)],has_u=False) for _ in range(current_g)]
    for t,ids in enumerate(IDS):
        label=f'I G{current_g} t={t}'
        literal('BEGIN '+label)
        old=copy.deepcopy(ss)
        ss,snaps=evolve(old,t)
        for st in ss: st['has_u']=True
        for s in range(current_g):
            for k in range(2):
                call(f'Weight({2*s+k})',matrix(ss[s]['w'][k],(s+k)%2==0),None,'W',.4,'training',t,s)
                u=[[.08*1.9*x for x in row] for row in ss[s]['v'][k]]
                call(f'Velocity({2*s+k})',matrix(u,(s+k)%2==0),None,'U',.1,'training',t,s)
            if s+1<current_g: hidden_calls(snaps,s,ids,'training',t)
        states(label+' old',old)
        for s in range(current_g):
            literal(f'TRACE_FLAGS {label} stage={s} head_affines={len(ids)} residuals={len(ids)} outer_sums=2 state_updates=2')
            for j,r in enumerate(ids):
                add(f'{label} s={s} id={r} hidden_pre',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'training_hidden',.5)
                add(f'{label} s={s} id={r} output',vec(snaps[s]['output'][j],s%2==0),s%2==0,'training_output')
            stage(f'{label} candidate stage={s}',ss[s],s)
        for s in range(current_g-1):
            for j,r in enumerate(ids):
                for suffix in ['source','output']:
                    add(f'{label} handoff s={s} id={r} {suffix}',vec(snaps[s]['hidden'][j],s%2==1),s%2==1,'training_handoff',.5)
        states(label+' candidate',ss)
        committed+=1
        literal(f'GRAPH_COMMIT {label} count={committed} indices={[t+1]*current_g}')
    snaps=[]
    inputs=X
    for s,st in enumerate(ss):
        snap_s=snap(st,inputs,[0,1,2],s+1==current_g)
        snaps.append(snap_s)
        inputs=snap_s['hidden']
    for s in range(current_g-1): hidden_calls(snaps,s,[0,1,2],'inference',None)
    label='I inference'
    for s in range(current_g):
        literal(f'INFERENCE_TRACE_FLAGS {label} stage={s} head_affines={3 if s+1==current_g else 0} loss_outer_update_derivative_boundaries=0 graph_commits=0')
        for r in range(3):
            for suffix,key,expanded in [('input','input',s%2==0),('z','z',s%2==1),('path_output','output',s%2==0 if s+1==current_g else s%2==1),('hidden_pre','hidden',s%2==1)]:
                add(f'{label} s={s} id={r} {suffix}',vec(snaps[s][key][r],expanded),expanded,'inference_'+suffix,.6 if key=='hidden' or (key=='output' and s+1<current_g) else None)
    for s in range(current_g-1):
        for r in range(3):
            for suffix in ['source','out']:
                add(f'{label} handoff s={s} id={r} {suffix}',vec(snaps[s]['hidden'][r],s%2==1),s%2==1,'inference_handoff',.6)
    for r in range(3):
        add(f'{label} final id={r}',vec(snaps[-1]['output'][r],(current_g-1)%2==0),(current_g-1)%2==0,'inference_final')
    states(label+' unchanged',ss)
assert len(expected)==710 and len(client_specs)==116

slot_re=re.compile(r'SLOTS (.+) context=(0x[0-9a-f]+) level=(\d+) scale_bits=(\d+) components=(\d+) values=(.+)')
parity_re=re.compile(r'PARITY (.+) max=(\S+) worst=(\d+) active=(\S+) padding=(\S+) replica=(\S+) imaginary=(\S+)')
def f64(x):
    assert isinstance(x,int) and 0<=x<2**64
    return struct.unpack('>d',struct.pack('>Q',x))[0]
actual_protocol=[]
records=[]
contexts=set()
levels=collections.Counter()
for i,l in enumerate(lines):
    if l.startswith(('BEGIN ','TRACE_FLAGS ','GRAPH_COMMIT ','INFERENCE_TRACE_FLAGS ')):
        actual_protocol.append(('line',l))
    if not l.startswith('SLOTS '): continue
    m=slot_re.fullmatch(l)
    assert m, i
    label,ctx,level,scale,components,payload=m.groups()
    assert int(scale)==SCALE and int(components)==2 and 0<=int(level)<=31
    contexts.add(ctx); levels[int(level)]+=1
    bits=ast.literal_eval(payload)
    assert len(bits)==512 and all(isinstance(x,tuple) and len(x)==2 for x in bits)
    a=[(f64(x),f64(y)) for x,y in bits]
    assert all(math.isfinite(x) and math.isfinite(y) for x,y in a)
    p=parity_re.fullmatch(lines[i+1])
    assert p and p[1]==label
    summary=dict(zip(['max','worst','active','padding','replica','imaginary'],[float(p[2]),int(p[3]),float(p[4]),float(p[5]),float(p[6]),float(p[7])]))
    assert all(math.isfinite(v) for v in summary.values()) and 0<=summary['worst']<512
    assert summary['max']<=E
    actual_protocol.append(('slot',label))
    records.append(dict(label=label,values=a,bits=bits,level=int(level),summary=summary))
assert actual_protocol==protocol, next(((i,a,b) for i,(a,b) in enumerate(zip(actual_protocol,protocol)) if a!=b), ('length',len(actual_protocol),len(protocol)))
assert len(records)==710 and len([l for l in lines if l.startswith('PARITY ')])==710 and len(contexts)==1
stats={}
summary_diff={k:0. for k in ['max','active','padding','replica','imaginary']}
worst_disagreements=[]
rows=[]
for i,(a,e) in enumerate(zip(records,expected)):
    assert a['label']==e['label']
    v=a['values']; shape=e['shape']
    errors=[math.hypot(z[0]-want,z[1]) for z,want in zip(v,e['expected'])]
    used=[(s//4<3 and s%4<3) if shape is None else ((s//4 if shape else s%4)<3) for s in range(512)]
    metrics=dict(max=max(errors),worst=max(range(512),key=errors.__getitem__),active=max(x for x,u in zip(errors,used) if u),padding=max(x for x,u in zip(errors,used) if not u),imaginary=max(abs(z[1]) for z in v),replica=0.)
    if shape is not None:
        metrics['replica']=max(math.hypot(z[0]-v[4*(s//4) if shape else s%4][0],z[1]-v[4*(s//4) if shape else s%4][1]) for s,z in enumerate(v) if used[s])
    magnitude=max(math.hypot(*z) for z in v)
    assert metrics['max']<=E, (a['label'],metrics)
    if e['bound'] is not None: assert magnitude<=e['bound']
    # Summary agreement is reported without adding or adjusting a numeric gate.
    for key in summary_diff:
        summary_diff[key]=max(summary_diff[key],abs(metrics[key]-a['summary'][key]))
    if metrics['worst']!=a['summary']['worst']: worst_disagreements.append(i)
    family=e['family']
    st=stats.setdefault(family,dict(count=0,max_error=0.,max_magnitude=0.,max_imaginary=0.,max_replica=0.,max_padding=0.))
    st['count']+=1
    for target,value in [('max_error',metrics['max']),('max_magnitude',magnitude),('max_imaginary',metrics['imaginary']),('max_replica',metrics['replica']),('max_padding',metrics['padding'])]:
        st[target]=max(st[target],value)
    rows.append(dict(index=i,graph=e['graph'],label=a['label'],level=a['level'],family=family,bound=e['bound'],magnitude=magnitude,**metrics))

# Exact callback ordering and canonical repack: fresh ciphertext can differ by
# encryption error, so only report measured output-vs-repack error (no new gate).
clients=[]
for spec in client_specs:
    inp=next(r for r in records if r['label'].startswith(f"client={spec['index']} role="))
    out=next(r for r in records if r['label']==f"client={spec['index']} output")
    assert out['level']==16
    expected_input_level=12 if spec['family']=='hidden' else (4 if int(re.search(r'\((\d+)\)',spec['role'])[1])%2==0 else 7)
    assert inp['level']==expected_input_level
    packed=[0.]*512
    if spec['family'] in ['W','U']:
        for j in range(512):
            if j//4<3 and j%4<3: packed[j]=inp['values'][j][0]
    else:
        expanded=spec['stage']%2==1
        for k in range(3):
            indices=[j for j in range(512) if (j//4 if expanded else j%4)==k]
            mean=sum(inp['values'][j][0] for j in indices)/len(indices)
            for j in indices: packed[j]=mean
    err=max(math.hypot(z[0]-w,z[1]) for z,w in zip(out['values'],packed))
    clients.append(dict(**spec,input_level=inp['level'],output_level=out['level'],output_vs_canonical_repack_max=err))

assert not worst_disagreements and summary_diff['imaginary']==0.
assert [(c['index'],c['graph']) for c in clients]==[(i,2 if i<44 else 3) for i in range(116)]

# Both candidate records share a label but are different: child pre-refresh,
# then whole-graph post-refresh. Bind each to its respective callback payload.
identity_checks=0
for g in [2,3]:
    by_label=collections.defaultdict(list)
    for a,e in zip(records,expected):
        if e['graph']==g: by_label[a['label']].append(a)
    for t in range(4):
        for s in range(g):
            for role in ['W0','U0','W1','U1']:
                group=by_label[f'I G{g} t={t} candidate stage={s} {role}']
                assert len(group)==2
                kind='Weight' if role[0]=='W' else 'Velocity'
                spec=next(c for c in client_specs if c['graph']==g and c['update']==t and c['role']==f'{kind}({2*s+int(role[1])})')
                for ix,label in enumerate([f"client={spec['index']} role={spec['role']} input",f"client={spec['index']} output"]):
                    assert group[ix]['bits']==by_label[label][0]['bits']
                    identity_checks+=1
                next_label=f'I G{g} t={t+1} old stage={s} {role}' if t<3 else f'I inference unchanged stage={s} {role}'
                assert group[1]['bits']==by_label[next_label][0]['bits']
                identity_checks+=1
    for spec in [c for c in client_specs if c['graph']==g and c['family']=='hidden']:
        label=f"I G{g} t={spec['update']}" if spec['phase']=='training' else 'I inference'
        for direction,suffix in [('input','source'),('output','output' if spec['phase']=='training' else 'out')]:
            a=next(r for r in records if r['label']==(f"client={spec['index']} output" if direction=='output' else f"client={spec['index']} role={spec['role']} input"))
            b=by_label[f"{label} handoff s={spec['stage']} id={spec['record']} {suffix}"][0]
            assert a['bits']==b['bits']; identity_checks+=1

result=dict(status='PASS_for_retained_I_snapshot_scope',source_commit=meta['source_commit'],manifest_entries_verified=len(verified),owned_source_entries=len(manifest['owned_source']),stdout_sha256=sha(RUN/'stdout.txt'),stdout_bytes=len(raw.encode()),script_sha256=sha(Path(__file__)),parameter_line=params,graph_count=2,commits=8,inferences=2,client_calls=len(clients),client_protocol=clients,slots_records=len(rows),complex_slots=len(rows)*512,finite_f64_components=len(rows)*1024,context_count=len(contexts),level_histogram=dict(levels),E=E,global_max_error=max(rows,key=lambda x:x['max']),summary_max_absolute_differences=summary_diff,summary_worst_index_disagreements=worst_disagreements,exact_repeated_payload_checks=identity_checks,families=stats,rows=rows)
output=LEDGER/'task-54-d4c3-i-runtime-audit.json'
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ['status','manifest_entries_verified','slots_records','complex_slots','finite_f64_components','level_histogram','global_max_error','summary_max_absolute_differences','summary_worst_index_disagreements','exact_repeated_payload_checks','families']},indent=2))
print('derived_json_sha256',sha(output))
