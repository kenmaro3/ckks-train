"""Read retained NONSCIENTIFIC_TEST_EVIDENCE only; no trainer execution."""
from pathlib import Path
import re, csv, ast, math, collections
root=Path(__file__).parent
lines=(root/'03-single-numerical.stdout.log').read_text().splitlines()
margin=[]; controls=[]; ranges=[]; calls=collections.Counter(); snapshot_count=0; values_count=0; frozen_count=0
for line in lines:
    if 'NONSCIENTIFIC_TEST_EVIDENCE ' not in line: continue
    line=line.split('NONSCIENTIFIC_TEST_EVIDENCE ',1)[1]
    kind,_,rest=line.partition(' ')
    if kind in ('MARGIN','CONTROL','RANGE','ACTUAL_TRAIN'):
        tokens=rest.split(); candidate,family=tokens[0].split('/')
        fields=dict(token.split('=',1) for token in tokens[1:] if '=' in token)
        fields.update(candidate=candidate,family=family)
        if kind=='MARGIN':
            fields['control_value']=fields['control']
            fields['control']=re.search(r' control=([^ ]+)',rest).group(1)
            margin.append(fields)
        if kind=='CONTROL': controls.append(fields)
        if kind=='RANGE': fields['phase']='before' if ' before ' in rest else 'after'; ranges.append(fields)
        if kind=='ACTUAL_TRAIN': calls[(candidate,family,fields['control'])]+=1
    if kind=='SNAPSHOT':
        snapshot_count+=1
        vals=ast.literal_eval(rest.split(' values=',1)[1].split(' bits=',1)[0])
        bits=ast.literal_eval(rest.split(' bits=',1)[1])
        assert all(math.isfinite(value) for value in vals)
        assert len(vals)==len(bits)
        values_count+=len(vals)
        if '/poly-' in rest and (' tensor=b ' in rest or ' tensor=Vb ' in rest):
            assert all(bit==0 for bit in bits)
            frozen_count+=len(bits)
for row in controls:
    row['actual_calls']=str(calls[(row['candidate'],row['family'],row['control'])])
def write_tsv(name,rows,keys):
    with (root/name).open('w') as out:
        out.write('# NONSCIENTIFIC_TEST_EVIDENCE extracted verbatim fields from sole invocation08\n')
        writer=csv.DictWriter(out,fieldnames=keys,delimiter='\t',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
margin_keys=['candidate','family','control','update','required_layer','backend','epsilon','epsilon_bits','layer','tensor','coordinate','baseline','baseline_bits','control_bits','max_abs_difference','max_normalized_distance','required','status']
margin_keys.insert(margin_keys.index('control_bits'),'control_value')
write_tsv('all-margins.tsv',margin,margin_keys)
write_tsv('failed-margins.tsv',[row for row in margin if row['status']=='FAIL'],margin_keys)
write_tsv('all-controls.tsv',controls,['candidate','family','control','status','first_affected_update','actual_calls'])
write_tsv('all-hidden-ranges.tsv',ranges,['candidate','family','control','update','phase','hidden_activation_max','hidden_preactivation_max','activation_min','activation_max','preactivation_min','preactivation_max','baseline_gate','limit'])
summary=[]
for candidate in ('Original','Signal'):
    cm=[row for row in margin if row['candidate']==candidate]
    cr=[row for row in ranges if row['candidate']==candidate and row['baseline_gate']=='true']
    cc=[row for row in controls if row['candidate']==candidate]
    summary.append(f'{candidate}: calls={sum(count for key,count in calls.items() if key[0]==candidate)} controls={len(cc)} statuses={dict(collections.Counter(row["status"] for row in cc))} margin_rows={len(cm)} failures={dict(collections.Counter(row["backend"] for row in cm if row["status"]=="FAIL"))} baseline_a_max={max(float(row["hidden_activation_max"]) for row in cr):.17e} baseline_z_max={max(float(row["hidden_preactivation_max"]) for row in cr):.17e}')
    for family in sorted(set(row['family'] for row in cr)):
        fr=[row for row in cr if row['family']==family]
        summary.append(f'  {family}: a_max={max(float(row["hidden_activation_max"]) for row in fr):.17e} z_max={max(float(row["hidden_preactivation_max"]) for row in fr):.17e}')
summary.append(f'Finite all logged W/b/velocity values: PASS; snapshot tensors={snapshot_count}; scalar entries={values_count}; exact frozen b/Vb zero entries={frozen_count}')
(root/'extracted-summary.txt').write_text('NONSCIENTIFIC_TEST_EVIDENCE\n'+'\n'.join(summary)+'\n')
print('\n'.join(summary))
