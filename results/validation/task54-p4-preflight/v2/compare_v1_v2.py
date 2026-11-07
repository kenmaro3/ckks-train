"""Read-only exact comparison of retained v1/v2 test evidence; no trainer calls."""
from pathlib import Path
import collections,csv,re,hashlib
root=Path(__file__).parent
v1=root.parent/'task-54-p4-pure-preflight-validation'
def records(path):
 result=collections.defaultdict(list)
 for line in path.read_text().splitlines():
  if 'NONSCIENTIFIC_TEST_EVIDENCE ' not in line:continue
  entry=line.split('NONSCIENTIFIC_TEST_EVIDENCE ',1)[1]
  kind=entry.split(' ',1)[0];result[kind].append(entry)
 return result
a=records(v1/'08-single-numerical.stdout.log');b=records(root/'03-single-numerical.stdout.log')
assert a.keys()==b.keys()
summary=['NONSCIENTIFIC_TEST_EVIDENCE READ_ONLY_COMPARISON']
for kind in a:
 assert len(a[kind])==len(b[kind]),kind
 summary.append(f'{kind}: v1={len(a[kind])} v2={len(b[kind])}')
 if kind in ('CALL_INPUT','SNAPSHOT','RANGE','CONTROL','ACTUAL_TRAIN','TOTAL'):
  assert a[kind]==b[kind],kind
  summary.append('  PASS exact full text identity, including every retained value/bit/ordinal')
 elif kind=='MANIFEST':
  assert [s.replace('task54-p4-pure-v1','task54-p4-pure-v2') for s in a[kind]]==b[kind]
  summary.append('  PASS VERSION is the only manifest difference')
 elif kind=='MARGIN':
  for old,new in zip(a[kind],b[kind]):
   if ' backend=I ' in old:assert old==new
   else:
    trim=lambda s:re.sub(r' (epsilon|epsilon_bits|max_normalized_distance|status)=[^ ]+','',s)
    assert trim(old)==trim(new),(old,new)
  summary.append('  PASS all raw I margins exact; NI only epsilon/bits, normalized distance/status differ')
 elif kind=='CANDIDATE':
  for old,new in zip(a[kind],b[kind]):
   assert old.split(' gate_failures=')[0]==new.split(' gate_failures=')[0]
  assert b[kind][0].endswith('gate_status=FAIL')
  assert b[kind][1].endswith('gate_failures=[] gate_status=PASS')
  summary.append('  PASS candidate identities/counts/harness fields unchanged; numerical gate fields reflect revised cap')
 else:raise AssertionError(kind)
def rows(path):return list(csv.DictReader(path.read_text().splitlines()[1:],delimiter='\t'))
old=rows(v1/'all-margins.tsv');new=rows(root/'all-margins.tsv')
assert len(old)==len(new)==228
flips=[]
for left,right in zip(old,new):
 unchanged=set(left)-{'epsilon','epsilon_bits','max_normalized_distance','status'}
 assert all(left[k]==right[k] for k in unchanged)
 if left['backend']=='I':assert left==right
 else:
  assert float(left['epsilon'])==5e-4 and float(right['epsilon'])==4e-4
  assert float(right['max_normalized_distance'])==float(right['max_abs_difference'])/4e-4
  assert (right['status']=='PASS')==(float(right['max_normalized_distance'])>=8)
  if left['status']!=right['status']:flips.append(right)
assert len(flips)==5 and all(r['candidate']=='Signal' and r['status']=='PASS' for r in flips)
assert rows(v1/'all-controls.tsv')==rows(root/'all-controls.tsv')
assert rows(v1/'all-hidden-ranges.tsv')==rows(root/'all-hidden-ranges.tsv')
for line in (root/'v1-preservation-before.sha256').read_text().splitlines():
 digest,name=line.split('  ',1);assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
summary.extend(['PASS all input/state bits, ranges, absolute witness distances and coordinates identical.', 'PASS exactly five Signal NI FAIL->PASS flips; Original FAIL retained; I fields unchanged.', 'PASS all original v1 evidence files/report hashes unchanged; v1 remains FAIL.', 'No Engine, forward, materializer, planner, parameter/proof/descriptor/Encoder/key/evaluator or HE operation executed by comparison.'])
(root/'v1-v2-comparison.txt').write_text('\n'.join(summary)+'\n')
print('\n'.join(summary))
