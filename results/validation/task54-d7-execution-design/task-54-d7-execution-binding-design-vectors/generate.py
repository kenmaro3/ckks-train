#!/usr/bin/env python3
"""Independent literal wire assembler. Python stdlib only; no repository imports.
No CkksParams, planner, prime generation, runtime, keys, or HE constructors.
Inputs are synthetic transport examples, not checked implementation objects.
Run from any cwd; all writes stay alongside this script.
"""
import copy
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENCODING = 'hekit-study-reference-encoding-v1'
DOMAINS = {
    'parameters': 'hekit-study-interactive-parameters-v1',
    'layout': 'hekit-study-diagonal-layout-v1',
    'keys': 'hekit-study-cpu-diagonal-keys-v1',
    'security': 'hekit-study-main-security-policy-v1',
}
A_MAX = 2097152

class LiteralBytes:
    def __init__(self):
        self.b = bytearray()
        self.offsets = {}
    def mark(self, name):
        self.offsets[name] = len(self.b)
    def u8(self, name, value):
        self.mark(name)
        self.b += struct.pack('<B', value)
    def u64(self, name, value):
        self.mark(name)
        self.b += struct.pack('<Q', value)
    def i32(self, name, value):
        self.mark(name)
        self.b += struct.pack('<i', value)
    def text(self, name, value):
        raw = value.encode('utf-8')
        self.u64(name+'.length', len(raw))
        self.mark(name+'.bytes')
        self.b += raw
    def seq(self, name, values, put):
        self.u64(name+'.count', len(values))
        for i, value in enumerate(values):
            put(name+'.'+str(i), value)
    def opt(self, name, value, put):
        self.u8(name+'.tag', int(value is not None))
        if value is not None:
            put(name+'.value', value)
    def secret(self, name, value):
        self.u8(name+'.tag', int(value is not None))
        if value is not None:
            self.u64(name+'.weight', value)
    def compact(self, name, values):
        for field in ['d_out','d_in','batch','diags_per_ct','lane_shift']:
            self.u64(name+'.'+field, values[field])
    def stage(self, name, v):
        self.seq(name+'.dims', v['dims'], self.u64)
        for field in ['slots','d','batch']:
            self.u64(name+'.'+field, v[field])
        def shape(n, s):
            for field, value in zip(['d_out','d_in','n_out','n_in','l','m','baby','giant'], s):
                self.u64(n+'.'+field, value)
        self.seq(name+'.shapes', v['shapes'], shape)
        def mask(n, m):
            for field in ['canonical_period','gradient_period','bias_period','lanes']:
                self.u64(n+'.'+field, m[field])
            self.seq(n+'.reduce_steps', m['reduce_steps'], self.i32)
            self.seq(n+'.replication_steps', m['replication_steps'], self.i32)
            self.u8(n+'.compact_first_period', m['compact_first_period'])
        self.seq(name+'.masks', v['masks'], mask)
        def recipe(n, rows):
            def row(k, entry):
                self.u64(k+'.source', entry['source'])
                self.seq(k+'.steps', entry['steps'], self.i32)
            self.seq(n, rows, row)
        self.seq(name+'.transpose', v['transpose'], recipe)
        self.seq(name+'.rotations', v['rotations'], self.i32)
        def occupancy(n, c):
            self.seq(n+'.plans', c['plans'], self.compact)
            def container(k, x):
                self.opt(k+'.layer', x['layer'], self.u64)
                self.u64(k+'.index', x['index'])
                self.compact(k+'.plan', x['plan'])
            self.seq(n+'.containers', c['containers'], container)
            def bias(k, x):
                self.u64(k+'.container', x['container'])
                self.u64(k+'.position', x['position'])
            self.seq(n+'.biases', c['biases'], bias)
        self.opt(name+'.compact', v['compact'], occupancy)

def assemble(kind, v):
    w = LiteralBytes()
    w.text('encoding', ENCODING)
    w.text('domain', DOMAINS[kind])
    if kind == 'parameters':
        for field in ['q','p','q_bits','p_bits']:
            w.seq(field, v[field], w.u64)
        for field in ['log_n','scale_bits','dnum']:
            w.u64(field, v[field])
        w.opt('main_weight', v['main_weight'], w.u64)
    elif kind == 'layout':
        w.text('parameter_identity', v['parameter_identity'])
        for field in ['consumption','mode']:
            w.u8(field, v[field])
        for field in ['hoist','bsgs','lazy_relin']:
            w.u8('eval.'+field, v['eval'][field])
        w.seq('stages', v['stages'], w.stage)
        def update(n, parts):
            def part(k, p):
                w.u64(k+'.start', p['start'])
                w.u64(k+'.valid', p['valid'])
            w.seq(n, parts, part)
        w.seq('parts', v['parts'], update)
    elif kind == 'keys':
        w.text('parameter_identity', v['parameter_identity'])
        w.opt('bootstrap_descriptor', v['bootstrap_descriptor'], w.text)
        w.u8('transpose.tag', int(v['transpose'] is not None))
        if v['transpose'] is not None:
            w.seq('transpose.pool', v['transpose'], w.i32)
        w.u64('training_level', v['training_level'])
        w.seq('training_rotations', v['training_rotations'], w.i32)
        for field in ['training_relin','training_conjugation','training_sse']:
            w.u8(field, v[field])
        for field in ['bootstrap_generated','bootstrap_execution']:
            w.seq(field, v[field], w.i32)
        for field in ['bootstrap_conjugation','bootstrap_sse','separate_bootstrap_bundle','owner_policy']:
            w.u8(field, v[field])
    elif kind == 'security':
        w.u8('registered_policy', v['registered_policy'])
        w.text('parameter_identity', v['parameter_identity'])
        w.opt('bootstrap_descriptor', v['bootstrap_descriptor'], w.text)
        w.secret('main', v['main'])
        # Explicit {weight:null} is Some(DenseTernary), null is None.
        w.opt('ephemeral', v['ephemeral'], lambda n,s: w.secret(n, s['weight']))
        w.u8('assurance', v['assurance'])
        w.opt('bootstrap_scope', v['bootstrap_scope'], w.u8)
        w.seq('bootstrap_unvalidated', v['bootstrap_unvalidated'], w.u8)
    return bytes(w.b), w.offsets

def digest(b):
    return {'byte_count':len(b), 'sha256':hashlib.sha256(b).hexdigest()}

def put_json(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)+'\n')

# Constants below are literal specimens. No semantic geometry or prime computation.
p = {'q':[97,193], 'p':[257], 'q_bits':[7,8], 'p_bits':[9],
     'log_n':4, 'scale_bits':0x4050000000000000, 'dnum':1, 'main_weight':None}
mask = {'canonical_period':2,'gradient_period':2,'bias_period':2,'lanes':2,
        'reduce_steps':[1], 'replication_steps':[-1], 'compact_first_period':0}
stage = {'dims':[2,2,2], 'slots':8,'d':2,'batch':2,
         'shapes':[[2,2,2,2,2,2,1,2],[2,2,2,2,2,2,1,2]],
         'masks':[copy.deepcopy(mask),copy.deepcopy(mask)],
         'transpose':[[{'source':0,'steps':[]},{'source':1,'steps':[-2,1]}],
                      [{'source':0,'steps':[1,-2]},{'source':1,'steps':[]}]],
         'rotations':[1,2,6], 'compact':None}
compact = {'d_out':2,'d_in':2,'batch':2,'diags_per_ct':2,'lane_shift':4}
ni_stage = copy.deepcopy(stage)
for m in ni_stage['masks']:
    m['compact_first_period'] = 1
ni_stage['compact'] = {'plans':[copy.deepcopy(compact),copy.deepcopy(compact)],
    'containers':[{'layer':0,'index':0,'plan':copy.deepcopy(compact)},
                  {'layer':None,'index':1,'plan':copy.deepcopy(compact)}],
    'biases':[{'container':0,'position':0},{'container':1,'position':1}]}
layout = {'parameter_identity':'a'*64, 'consumption':0,'mode':0,
          'eval':{'hoist':1,'bsgs':1,'lazy_relin':0}, 'stages':[stage],
          'parts':[[{'start':0,'valid':2},{'start':2,'valid':1}]]}
keys = {'parameter_identity':'a'*64,'bootstrap_descriptor':None,'transpose':None,
        'training_level':1,'training_rotations':[1,2,6],
        'training_relin':1,'training_conjugation':0,'training_sse':0,
        'bootstrap_generated':[],'bootstrap_execution':[],
        'bootstrap_conjugation':0,'bootstrap_sse':0,'separate_bootstrap_bundle':0,'owner_policy':0}
security = {'registered_policy':0,'parameter_identity':'a'*64,'bootstrap_descriptor':None,
            'main':None,'ephemeral':None,'assurance':0,'bootstrap_scope':None,
            'bootstrap_unvalidated':[]}
fixtures = []
def add(name, kind, values, branches, unsupported='All are transport-only synthetic fixtures; no live owned producer or secure profile.'):
    fixtures.append({'name':name,'kind':kind,'input':copy.deepcopy(values),
                     'covers':branches,'production_status':unsupported})
add('parameter_dense','parameters',p,['main_weight=None'])
p_fixed=copy.deepcopy(p); p_fixed['main_weight']=3
add('parameter_fixed','parameters',p_fixed,['main_weight=Some(3)'],'Fixed main weight is explicitly unsupported by current main I policy; transport-only fixture.')
add('layout_i_single','layout',layout,['Full','I','single stage','compact=None','partial partition','negative ordered I32 chains','empty/nonempty nested lists'])
layout_ni=copy.deepcopy(layout); layout_ni.update(consumption=1,mode=1)
layout_ni['eval']={'hoist':0,'bsgs':1,'lazy_relin':1}
layout_ni['stages']=[copy.deepcopy(ni_stage),copy.deepcopy(ni_stage)]
layout_ni['parts']=[[{'start':0,'valid':2}],[{'start':0,'valid':1}]]
add('layout_ni_two','layout',layout_ni,['Quick','NI','two stages','compact=Some','container layer=None/Some','partial final update','Bool0/1'])
layout_three=copy.deepcopy(layout); layout_three['stages']=[copy.deepcopy(stage) for _ in range(3)]
add('layout_i_three','layout',layout_three,['three stages','common layout'])
add('keys_i_dedicated','keys',keys,['Dedicated','bootstrap=None','empty bootstrap lists','separate=false'])
keys_ir=copy.deepcopy(keys); keys_ir['transpose']=[6,1,2]
add('keys_i_reuse','keys',keys_ir,['Reuse','exact ordered pool'])
keys_ni=copy.deepcopy(keys); keys_ni.update(bootstrap_descriptor='b'*64,
    bootstrap_generated=[1,2,4],bootstrap_execution=[1,4],bootstrap_conjugation=1,separate_bootstrap_bundle=1)
add('keys_ni_dedicated','keys',keys_ni,['bootstrap=Some','Dedicated','generated != executed','bootstrap SSE=false'])
keys_nir=copy.deepcopy(keys_ni); keys_nir.update(transpose=[6,1,2],bootstrap_sse=1)
add('keys_ni_reuse','keys',keys_nir,['Reuse','bootstrap SSE=true','separate=true'])
add('security_i_dense','security',security,['I','DenseTernary','ephemeral=None','bootstrap=None','scope=None','empty limitations'])
sec_fixed=copy.deepcopy(security); sec_fixed['main']=3
add('security_i_fixed','security',sec_fixed,['main FixedWeight(3)'],'Fixed main weight is unsupported by current main I policy; transport-only secret-tag fixture.')
sec_ni=copy.deepcopy(security); sec_ni.update(registered_policy=1,bootstrap_descriptor='b'*64,bootstrap_scope=0,bootstrap_unvalidated=list(range(9)))
add('security_ni_absent','security',sec_ni,['NI','scope=Some(0)','all limitations0..8','ephemeral=None'],'Absent ephemeral may differ from registered default NI BootParams; transport-only, exact owned-default equality still required.')
sec_nif=copy.deepcopy(sec_ni); sec_nif['ephemeral']={'weight':3}
add('security_ni_fixed','security',sec_nif,['ephemeral=Some(FixedWeight(3))'],'Ephemeral weight3 is synthetic, not a default_n16 declaration or security evidence.')
sec_nid=copy.deepcopy(sec_ni); sec_nid['ephemeral']={'weight':None}
add('security_ni_dense','security',sec_nid,['ephemeral=Some(DenseTernary)'],'Some(DenseTernary) is a grammar branch not produced by current P1 ephemeral_hw mapping; never a production candidate.')
put_json('inputs.json',{'notice':'Literal transport-only fixtures; all digests a/b are literal placeholders, not evidence.', 'vectors':fixtures})
assembled={}; summaries=[]
for f in fixtures:
    b,off=assemble(f['kind'],f['input']); assert len(b)<=A_MAX
    assembled[f['name']]=(f,b,off)
    (ROOT/(f['name']+'.hex')).write_text(b.hex()+'\n')
    summaries.append({'name':f['name'],'domain':DOMAINS[f['kind']],**digest(b),
                      'hex_file':f['name']+'.hex','offsets':off,'covers':f['covers']})
put_json('manifest.json',{'encoding':ENCODING,'A_MAX':A_MAX,'vectors':summaries})

negatives=[]
def mutated(name,base,path,raw,expected):
    _,b,off=assembled[base]; pos=off[path]
    out=b[:pos]+raw+b[pos+len(raw):]
    negatives.append({'name':name,'base':base,'mutation':{'offset':pos,'old_hex':b[pos:pos+len(raw)].hex(),'new_hex':raw.hex()},
                      'expected':expected,**digest(out),'complete_hex':out.hex()})
def values(name,base,change,expected):
    f,_,_=assembled[base]; v=copy.deepcopy(f['input']); change(v)
    out,_=assemble(f['kind'],v)
    negatives.append({'name':name,'base':base,'replacement_input':v,'expected':expected,
                      **digest(out),'complete_hex':out.hex()})
def byte(name,base,path,value,expected): mutated(name,base,path,bytes([value]),expected)
def u64(name,base,path,value,expected): mutated(name,base,path,struct.pack('<Q',value),expected)
wire='Reject malformed representation before allocation/owned projection; no capability.'
policy='Wire may be well-formed; reject semantic/actual-owned binding even after recomputing digest; no capability.'
for base in ['parameter_dense','layout_i_single','keys_i_dedicated','security_i_dense']:
    byte(base+'-wrong-domain',base,'domain.bytes',ord('X'),wire)
u64('parameter-count-u64-overflow','parameter_dense','q.count',2**64-1,wire)
u64('parameter-count-exceeds-A_MAX','parameter_dense','q.count',A_MAX//8+1,wire)
u64('parameter-wrong-text-length','parameter_dense','domain.length',1,wire)
values('parameter-empty-chain','parameter_dense',lambda v:v.update(q=[],q_bits=[]),policy)
values('parameter-bit-vector-count','parameter_dense',lambda v:v.update(q_bits=[7]),policy)
values('parameter-dnum-zero','parameter_dense',lambda v:v.update(dnum=0),policy)
values('parameter-dnum-not-divisor','parameter_dense',lambda v:v.update(dnum=3),policy)
values('parameter-logN-overflow','parameter_dense',lambda v:v.update(log_n=64),policy)
values('parameter-zero-scale','parameter_dense',lambda v:v.update(scale_bits=0),policy)
values('parameter-main-weight-zero','parameter_fixed',lambda v:v.update(main_weight=0),policy)
values('parameter-main-weight-overN','parameter_fixed',lambda v:v.update(main_weight=17),policy)
byte('parameter-option-tag2','parameter_dense','main_weight.tag',2,wire)
values('parameter-swapped-bit-vectors','parameter_dense',lambda v:v.update(q_bits=[9],p_bits=[7,8]),policy)
byte('layout-mode-tag2','layout_i_single','mode',2,wire)
byte('layout-consumption-tag2','layout_i_single','consumption',2,wire)
byte('layout-bool2','layout_i_single','eval.hoist',2,wire)
u64('layout-count-zero','layout_i_single','stages.count',0,wire)
u64('layout-count-four','layout_i_single','stages.count',4,wire)
byte('layout-compact-option2','layout_ni_two','stages.0.compact.tag',2,wire)
byte('layout-container-layer-option2','layout_ni_two','stages.0.compact.value.containers.0.layer.tag',2,wire)
u64('layout-valid-zero','layout_i_single','parts.0.0.valid',0,policy)
u64('layout-valid-over-lanes','layout_i_single','parts.0.0.valid',3,policy)
u64('layout-start-discontinuous','layout_i_single','parts.0.1.start',1,policy)
values('layout-ordered-chain-swap','layout_i_single',lambda v:v['stages'][0]['transpose'][0][1].update(steps=[1,-2]),policy)
byte('keys-bootstrap-option2','keys_i_dedicated','bootstrap_descriptor.tag',2,wire)
byte('keys-transpose-tag2','keys_i_dedicated','transpose.tag',2,wire)
byte('keys-owner-tag1','keys_i_dedicated','owner_policy',1,wire)
byte('keys-uppercase-hash','keys_i_dedicated','parameter_identity.bytes',ord('A'),wire)
values('keys-zero-rotation','keys_i_dedicated',lambda v:v.update(training_rotations=[0,1,2,6]),policy)
values('keys-duplicate-rotation','keys_i_dedicated',lambda v:v.update(training_rotations=[1,1,2,6]),policy)
values('keys-superset-not-equal','keys_i_dedicated',lambda v:v.update(training_rotations=[1,2,3,6]),policy)
values('keys-generated-executed-swap','keys_ni_dedicated',lambda v:v.update(bootstrap_generated=[1,4],bootstrap_execution=[1,2,4]),policy)
values('keys-bootstrap-absent-with-bundle','keys_ni_dedicated',lambda v:v.update(bootstrap_descriptor=None),policy)
values('keys-reuse-pool-reorder','keys_i_reuse',lambda v:v.update(transpose=[1,2,6]),policy)
byte('security-policy-tag2','security_i_dense','registered_policy',2,wire)
byte('security-main-secret-tag2','security_i_dense','main.tag',2,wire)
byte('security-ephemeral-option2','security_i_dense','ephemeral.tag',2,wire)
byte('security-ephemeral-secret-tag2','security_ni_fixed','ephemeral.value.tag',2,wire)
byte('security-assurance1','security_i_dense','assurance',1,wire)
byte('security-scope-option2','security_ni_absent','bootstrap_scope.tag',2,wire)
byte('security-scope-value1','security_ni_absent','bootstrap_scope.value',1,wire)
values('security-weight-zero','security_ni_fixed',lambda v:v.update(ephemeral={'weight':0}),policy)
values('security-weight-overN','security_ni_fixed',lambda v:v.update(ephemeral={'weight':17}),policy)
values('security-weight-change','security_ni_fixed',lambda v:v.update(ephemeral={'weight':4}),policy)
values('security-limitations-order','security_ni_fixed',lambda v:v.update(bootstrap_unvalidated=[1,0,2,3,4,5,6,7,8]),policy)
values('security-limitations-count','security_ni_fixed',lambda v:v.update(bootstrap_unvalidated=list(range(8))),policy)
values('security-NI-no-bootstrap','security_ni_fixed',lambda v:v.update(bootstrap_descriptor=None),policy)
values('security-absent-not-dense','security_ni_absent',lambda v:v.update(ephemeral={'weight':None}),policy)
for base in ['parameter_dense','layout_i_single','keys_i_dedicated','security_i_dense']:
    _,b,_=assembled[base]
    out=b+b'\x00'
    negatives.append({'name':base+'-trailing-byte','base':base,'mutation':{'append_hex':'00'},
                      'expected':wire,**digest(out),'complete_hex':out.hex()})
put_json('negative-vectors.json',{'notice':'Expected failures, not observed Rust test outcomes. Security weight fixtures bind synthetic N=16. Semantic cases require original retained fixture facts.', 'vectors':negatives,
    'families':[{'name':'every-proper-prefix','bases':[f['name'] for f in fixtures],
                 'exact_inputs':'For each base canonical byte array B, B[:n] for every integer 0<=n<len(B).',
                 'expected':wire},
                {'name':'total-size-guard','exact_numeric_inputs':[A_MAX-1,A_MAX,A_MAX+1,2**64-1],
                 'expected':['size guard only accepts; grammar/owner validation still required','size guard only accepts; grammar/owner validation still required','reject size before reserve','reject size before reserve']},
                {'name':'checked-size-arithmetic','exact_numeric_inputs':{'u64_add':[2**64-1,1],'u64_mul':[2**64-1,8],'u64_to_32bit_usize':2**32},
                 'expected':'checked arithmetic/conversion rejects before any reserve; no wrap or truncation'}]})
lines=['# Frozen independent wire vectors','',
       'Synthetic format examples only. No repository encoders, parameter/planner constructors, HE, native calls or evidence sources were used.',
       'The complete inputs are in inputs.json; complete canonical bytes in each named .hex; manifest.json fixes offsets, byte lengths and SHA256.',
       'All a/b hashes are literal placeholders. Positive means a byte-oracle example, not a production-valid object. Every production producer still derives fields from actual retained objects.',
       'Security Some(DenseTernary), fixed main, and non-default ephemeral examples explicitly test transport branches unsupported by current production policy. Layout values are hand-written format specimens, not derived StageGeometry evidence.',
       '', '| Vector | Bytes | SHA256 |','|---|---:|---|']
for s in summaries: lines.append('| '+s['name']+' | '+str(s['byte_count'])+' | `'+s['sha256']+'` |')
lines += ['', 'negative-vectors.json supplies '+str(len(negatives))+' complete mutation inputs with exact bytes/length/digest and distinguishes wire failures from live/source binding failures. It additionally fixes every proper truncation, exact A_MAX boundary checks, and checked-arithmetic/conversion failures without allocating multi-megabyte filler artifacts.',
          'A digest change alone is never a rejection oracle: semantic mutations must be refused against the unchanged retained source/route facts even if every outer digest is recomputed.',
          '', 'Reproduce: `python3 generate.py`. This writes only files beside this script; no project imports or constructors. Independent review must freeze these artifacts before hash-producing source. These are not test execution results or ratification.']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'positive_vectors':len(fixtures),'complete_negative_vectors':len(negatives),
    'positive_wire_bytes':sum(x['byte_count'] for x in summaries),'max_positive_bytes':max(x['byte_count'] for x in summaries)}))
