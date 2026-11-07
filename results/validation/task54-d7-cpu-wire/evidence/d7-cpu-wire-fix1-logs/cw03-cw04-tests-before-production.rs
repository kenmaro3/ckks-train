//! FIX1 contract-derived counterexamples, no live or numerical constructors.
use super::*;
use super::observed::*;
fn old(name:&str)->Vec<u8>{let v:serde_json::Value=serde_json::from_str(include_str!("observed/literals.json")).unwrap();unhex(v[name].as_str().unwrap())}
fn unhex(s:&str)->Vec<u8>{s.as_bytes().chunks_exact(2).map(|p|u8::from_str_radix(std::str::from_utf8(p).unwrap(),16).unwrap()).collect()}
#[test]
fn cw03_preparation_child_selected_seed(){
 let mut accepted=Vec::new();
 for(op,id)in[(4,"W09"),(5,"W10")]{let b=old(id);let w=RequestedWork::decode(&b,Mode::Oracle).unwrap();w.bind_preparation_root(op,"iris__emlp1",&[0]).unwrap();if w.check_child(op,2,"iris__emlp1",1).is_ok(){accepted.push(op);}}
 let b=old("W08");let mut b=b[..b.len()-16].to_vec();for n in[2u64,0,1]{b.extend_from_slice(&n.to_le_bytes());}let w=RequestedWork::decode(&b,Mode::Oracle).unwrap();w.bind_preparation_root(3,"iris__emlp1",&[0,1]).unwrap();w.check_child(3,2,"iris__emlp1",1).unwrap();assert!(accepted.is_empty(),"seed0-only roots accepted selected1: {accepted:?}");
}
#[test]
fn cw04_registered_full_checkpoint_prefix(){
 let b=old("0-1-result");let rec=ObservedRecord::decode(RecordKind::Result,&b,Mode::Oracle).unwrap();let RecordBody::Result(mut r)=rec.body else{unreachable!()};r.scope=Tag(0);r.status=Tag(2);r.completed_updates=6;r.processed_samples=120;r.checkpoints.truncate(1);r.checkpoints[0].updates=6;r.checkpoints[0].samples=120;r.checkpoints[0].crossed=vec![1];r.final_endpoint=None;r.model_digest=None;r.target=Some(TargetOutcome{lock:r.start.clone(),threshold_bits:0,reached:true,first_checkpoint:Some(6),time_basis:Id("cumulative_update_ns"),time_ns:Some(10),censor_ns:10});r.validate(Mode::Oracle).unwrap();
 let mut accepted=Vec::new();for(updates,crossed)in[(1,vec![1]),(12,vec![2]),(6,vec![2]),(6,vec![])]{let mut wrong=r.clone();wrong.completed_updates=updates;wrong.processed_samples=updates*20;wrong.checkpoints[0].updates=updates;wrong.checkpoints[0].samples=updates*20;wrong.checkpoints[0].crossed=crossed;wrong.target.as_mut().unwrap().first_checkpoint=Some(updates);if wrong.validate(Mode::Oracle).is_ok(){accepted.push((updates,wrong.checkpoints[0].crossed.clone()));}}assert!(accepted.is_empty(),"invalid prefixes accepted: {accepted:?}");
}
