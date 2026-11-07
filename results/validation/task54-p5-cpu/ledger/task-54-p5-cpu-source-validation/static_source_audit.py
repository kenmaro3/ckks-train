import pathlib,subprocess,re,hashlib,json
root=pathlib.Path('.')
out=pathlib.Path(__file__).parent
base='2ea00601c5d6f4c10a98f4e0ddb2405935c34022'
def old(path):return subprocess.check_output(['git','show',base+':'+path],text=True)
def method(source,marker):
 a=source.index(marker);opening=source.index('{',a);depth=1
 for b in range(opening+1,len(source)):
  depth+=(source[b]=='{')-(source[b]=='}')
  if not depth:return source[opening+1:b]
def norm(s):return re.sub(r'\s+','',s)
path='crates/ckks-train-bench/src/study_diagonal.rs';before=old(path);after=pathlib.Path(path).read_text();original=method(before,'fn check_state_without_progress(')
common='check_mlp(context, description, mlp, state)?; check_batch(batch, state.batch())?;'
result={}
for variant,helper in [('Interactive','check_interactive_stage'),('NonInteractive','check_ni_stage')]:
 arm=method(original,'Training::'+variant+'(t) =>')
 arm=arm.replace('self.context','context').replace('self.mlp','mlp').replace('self.description','description')
 result['P4_'+variant+'_expressions_preserved']=norm(common+arm+'Ok(())')==norm(method(after,'fn '+helper))
 assert result['P4_'+variant+'_expressions_preserved'],variant
for name in ['check_batch','check_optimizer','check_mlp','validate_initial_packet','check_ct']:
 result['P4_'+name+'_body_preserved']=norm(method(before,'fn '+name+'('))==norm(method(after,'fn '+name+'('));assert result['P4_'+name+'_body_preserved']
for marker in ['fn check_shape(','fn step(','fn validate_packet(','fn from_owner(','fn check_coefficients(','fn check_hidden_trace(']:
 result['P4_'+marker+'_preserved']=norm(method(before,marker))==norm(method(after,marker));assert result['P4_'+marker+'_preserved'],marker
for path in ['crates/ckks-train-bench/src/study_diagonal/p4_tests.rs','crates/ckks-train-bench/src/study_admission/p4_fixtures.rs','crates/ckks-train-bench/src/study_plain/p4_fixtures.rs','crates/ckks-train-bench/src/study_plain/p5_fixtures.rs','crates/ckks-train-bench/src/study_plain.rs','crates/ckks-train-nn/src/block.rs','Cargo.toml','crates/ckks-train-bench/Cargo.toml']:
 actual=pathlib.Path(path).read_bytes();reference=subprocess.check_output(['git','show',base+':'+path]);result['unchanged_'+path]=actual==reference;assert actual==reference,path
 result['sha256_'+path]=hashlib.sha256(actual).hexdigest()
# Prospective closed matrix arithmetic only; no planner/Engine/HE invocation.
rows=[('A',4,2,4,6),('B',4,3,4,6),('C',4,2,1,2),('D',2,2,2,2),('Poison',2,2,1,2)]
result['prospective_sessions']=sum(s for _,s,_,_,_ in rows)
result['prospective_plans']=len(range(result['prospective_sessions']))
result['prospective_geometries_initial']=sum(s*stages for _,s,stages,_,_ in rows)
result['prospective_updates']=sum(s*stages*updates for _,s,stages,updates,_ in rows)
result['prospective_handoffs']=sum(s*(stages-1)*updates for _,s,stages,updates,_ in rows)
result['prospective_transitions']=result['prospective_updates']+result['prospective_handoffs']
result['prospective_packets']=sum(s*updates for _,s,_,updates,_ in rows)
result['prospective_first_input_encryptions']=sum(s*micros for _,s,_,_,micros in rows)
result['prospective_stage_target_encryptions']=sum(s*stages*micros for _,s,stages,_,micros in rows)
result['prospective_success_stage_inspections']=sum(s*stages*updates for family,s,stages,updates,_ in rows if family!='Poison')
result['prospective_success_handoff_inspections']=sum(s*(stages-1)*micros for family,s,stages,_,micros in rows if family!='Poison')
result['prospective_captures']={'full_session':46,'poison_initial_model':2,'combined_state_model':48,'configuration_only':3,'unchanged_comparisons':68,'configuration_comparisons':4}
(out/'static-source-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
