//! D4c3: ordinary tests are parameter-only or controlled metadata runtime, never HE.
use ckks_train_ckks::{CkksParams, metadata::CtMeta};
use ckks_train_nn::packed_training::*;
use std::sync::Arc;

#[test]
fn block_partition_and_target_preflight() {
    let (q, aux) = ckks_train_ckks::params::n16_boot_bits();
    let p = Arc::new(CkksParams::new(10, &q, &aux, 42, 8).unwrap());
    let g = PackedGeometryPlan::new(3, &[StageShape::new(3, 3, 2).unwrap(); 2], 512).unwrap();
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let build = |ids: Vec<usize>, parts: Vec<PackedPartition>, targets: Vec<Vec<CtMeta>>| {
        PackedGraphBatchMeta::new(
            p.clone(),
            &g,
            0,
            ids.clone(),
            vec![h; ids.len()],
            targets,
            parts,
        )
    };
    let good = build(
        vec![0, 1, 2],
        vec![PackedPartition::new(0, 2, 2), PackedPartition::new(2, 4, 1)],
        vec![vec![h; 3]; 2],
    )
    .unwrap();
    assert_eq!(good.valid_ids(), &[0, 1, 2]);
    assert_eq!(good.valid_count(), 3);
    assert!(
        build(
            vec![0, 2, 1],
            vec![PackedPartition::new(0, 4, 3)],
            vec![vec![h; 3]; 2]
        )
        .is_err()
    );
    assert!(
        build(
            vec![0, 1, 2],
            vec![PackedPartition::new(0, 3, 2), PackedPartition::new(2, 4, 1)],
            vec![vec![h; 3]; 2]
        )
        .is_err()
    );
    assert!(
        build(
            vec![0, 1, 2],
            vec![PackedPartition::new(0, 4, 3)],
            vec![vec![h; 3]]
        )
        .is_err()
    );
    let forty_eight = build(
        (0..48).collect(),
        vec![
            PackedPartition::new(0, 32, 32),
            PackedPartition::new(32, 48, 16),
        ],
        vec![vec![h; 48]; 2],
    )
    .unwrap();
    assert_eq!(forty_eight.valid_count(), 48);
}
mod packed_blocks_helpers;
use ckks_train_ckks::{HeOps, OpCounter};
use ckks_train_nn::layout_reboot::GradRelin;
use packed_blocks_helpers::*;

#[test]
fn block_failure_controls() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let batch = batch(&ev, &old, &[0, 2]);
    let boot = boot(c);
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Aa).unwrap();
    let plan = server
        .plan_update(
            &old,
            &batch,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    let equivalent = packed_blocks_helpers::batch(&ev, &old, &[0, 2]);
    let failed = server
        .prepare_planned(&old, &equivalent, &plan)
        .unwrap_err();
    assert_eq!(failed.phase(), PackedGraphPhase::Preflight);
    assert_eq!(ev.counter(), OpCounter::default());
    let first = old.stages()[0].layers()[0].weights().ciphertext().tag;
    let old_tags = old
        .stages()
        .iter()
        .flat_map(|stage| {
            stage
                .layers()
                .iter()
                .map(|layer| layer.weights().ciphertext().tag)
        })
        .collect::<Vec<_>>();
    boot.bad_at.set(Some(7));
    let failed = server.prepare_planned(&old, &batch, &plan).unwrap_err();
    assert_eq!(failed.stage(), 1);
    assert_eq!(failed.children().len(), 1);
    assert!(ev.counter().bootstraps > 0);
    assert_eq!(
        old.stages()[0].layers()[0].weights().ciphertext().tag,
        first
    );
    for s in old.stages() {
        for layer in s.layers() {
            assert_eq!(layer.update_index(), 0);
            assert!(layer.velocity().is_none());
        }
        assert!(s.biases().is_none());
    }
    assert_eq!(
        old_tags,
        old.stages()
            .iter()
            .flat_map(|stage| stage
                .layers()
                .iter()
                .map(|layer| layer.weights().ciphertext().tag))
            .collect::<Vec<_>>()
    );
    assert_eq!(failed.phase(), PackedGraphPhase::Stage);
    assert_eq!(
        failed.role(),
        Some(PackedRefreshRole::Hidden {
            stage: 1,
            record: 0
        })
    );
    let Some(PackedGraphChildFailure::Ni(child)) = failed.child_failure() else {
        panic!("typed NI child failure missing")
    };
    assert_eq!(child.phase(), PackedNiPhase::Update);
    assert!(child.events().is_empty());
    assert!(child.updated_stage().is_none());
    let mut child_delta = boot.contract.cost_per_boot().clone();
    child_delta += &OpCounter {
        rotations: 4,
        relins: 2,
        ct_ct_tensor: 2,
        ct_pt_mults: 3,
        rescales: 5,
        sse_switches: 0,
        bootstraps: 0,
    };
    assert_eq!(child.counter_delta(), Some(&child_delta));
    let mut total = plan.children()[0].predicted_counts().clone();
    total += &child_delta;
    assert_eq!(failed.counter_delta(), Some(&total));
    assert_eq!(ev.counter(), total);
    // One explicitly bounded controlled execution of the ORIGINAL plan proves its tokens
    // still bind. This fake operation is not an actual-HE retry or a new metadata plan.
    boot.bad_at.set(None);
    let before = ev.counter();
    let (_, receipt) = server.prepare_planned(&old, &batch, &plan).unwrap();
    assert_eq!(receipt.actual_counts(), plan.predicted_counts());
    let mut expected = before;
    expected += plan.predicted_counts();
    assert_eq!(ev.counter(), expected);
}

#[test]
fn block_forward_only_excludes_local_heads_and_updates() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let boot = boot(c);
    let records = (0..4)
        .map(|id| {
            PackedForwardRecord::new(
                id,
                PackedVector::new(
                    &ev,
                    old.geometry().layers()[0].input(),
                    ev.ct(16, ev.params.scale()),
                )
                .unwrap(),
                id < 3,
            )
        })
        .collect();
    let inputs = PackedForwardBatch::new(
        &ev,
        &old,
        records,
        vec![PackedPartition::new(0, 2, 2), PackedPartition::new(2, 4, 1)],
    )
    .unwrap();
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au).unwrap();
    old.stages()[0].layers()[1]
        .weights()
        .ciphertext()
        .forbid_reads
        .store(true, std::sync::atomic::Ordering::Relaxed);
    let plan = server
        .plan_forward(&old, &inputs, PackedInferencePlan::BoundaryBootstrap)
        .unwrap();
    let (out, receipt) = server.prepare_forward(&old, &inputs, &plan).unwrap();
    assert_eq!(receipt.children()[0].inputs().len(), 3);
    assert_eq!(receipt.children()[0].preactivations().len(), 3);
    assert_eq!(receipt.children()[0].outputs().len(), 3);
    assert_eq!(out.valid_ids(), &[0, 1, 2]);
    assert_eq!(out.values().len(), 3);
    let local_head = old.stages()[0].layers()[1].weights().ciphertext().tag;
    assert!(
        !ev.muls
            .borrow()
            .iter()
            .any(|&(a, b)| a == local_head || b == local_head)
    );
    for s in old.stages() {
        for layer in s.layers() {
            assert_eq!(layer.update_index(), 0);
            assert!(layer.velocity().is_none());
        }
    }
    let mut expected = boot.contract.cost_per_boot().clone();
    expected += boot.contract.cost_per_boot();
    expected += boot.contract.cost_per_boot();
    expected += &OpCounter {
        rotations: 54,
        relins: 15,
        ct_ct_tensor: 15,
        ct_pt_mults: 21,
        rescales: 36,
        sse_switches: 0,
        bootstraps: 0,
    };
    assert_eq!(receipt.actual_counts(), &expected);
    assert_eq!(plan.predicted_counts(), &expected);
    assert_eq!(boot.calls.get(), 3);
}
#[test]
fn block_interactive_replay_is_not_a_handoff() {
    let (p, enc, _) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let data = batch(&ev, &old, &[0, 2]);
    let mut client = Client {
        ev: &ev,
        contract: PackedRefreshContract::new(&ev, 16).unwrap(),
        calls: 0,
        replay_at: Some(5),
        previous: None,
    };
    let server = PackedBlockInteractiveTrainer::new(&ev, &enc).unwrap();
    let f = server
        .prepare_update(&old, &data, 0.04, GradRelin::Lazy, &mut client)
        .unwrap_err();
    assert_eq!(f.phase(), PackedGraphPhase::Handoff);
    assert_eq!(f.children().len(), 1);
    assert_eq!(f.handoffs().len(), 1);
    assert!(f.handoffs()[0].completed());
    assert!(!f.handoffs()[0].validated());
    assert_eq!(client.calls, 5);
    assert_eq!(old.stages()[0].layers()[0].update_index(), 0);
}
#[test]
fn block_metadata_schedule() {
    let (p, enc, c) = parameters();
    let schedules = fixture::schedules(&p, &enc, &c);
    println!(
        "PARAMS N={} q={:?} p={:?} delta_bits={} dnum={} descriptor={:?}",
        p.n(),
        p.q(),
        p.p(),
        p.scale().to_bits(),
        p.dnum(),
        c.metadata().descriptor_identity_bytes()
    );
    let mut total = 0;
    let mut ordinal = 0;
    for (case_index, case) in schedules.into_iter().enumerate() {
        assert_eq!(case.total, fixture::CASE_COUNTS[case_index]);
        println!("CASE G{} {:?}", case.g, case.placement);
        for (t, update) in case.updates.iter().enumerate() {
            println!("UPDATE {t} counts {:?}", update.predicted_counts());
            for (s, child) in update.children().iter().enumerate() {
                println!(
                    "STAGE {s} refresh={} counts={:?} W={:?} U={:?}",
                    child.refresh_state(),
                    child.predicted_counts(),
                    child.committed().weights(),
                    child.committed().velocities()
                );
                if case.placement == PackedNiPlacement::Aa {
                    for &id in fixture::IDS[t] {
                        println!(
                            "ROLE ordinal={ordinal} G{} {:?} update={t} stage={s} hidden_record={id} cost={:?}",
                            case.g,
                            case.placement,
                            c.cost_per_boot()
                        );
                        ordinal += 1;
                    }
                }
                if child.refresh_state() {
                    for role in [
                        PackedRefreshRole::Weight(2 * s),
                        PackedRefreshRole::Velocity(2 * s),
                        PackedRefreshRole::Weight(2 * s + 1),
                        PackedRefreshRole::Velocity(2 * s + 1),
                    ] {
                        let mut cost = c.cost_per_boot().clone();
                        cost += &OpCounter {
                            ct_pt_mults: 1,
                            rescales: 1,
                            ..Default::default()
                        };
                        println!(
                            "ROLE ordinal={ordinal} G{} {:?} update={t} {role:?} inclusive_cost={cost:?}",
                            case.g, case.placement
                        );
                        ordinal += 1;
                    }
                }
                if s + 1 < case.g && case.placement != PackedNiPlacement::Aa {
                    for &id in fixture::IDS[t] {
                        println!(
                            "ROLE ordinal={ordinal} G{} {:?} update={t} stage={s} boundary_record={id} cost={:?}",
                            case.g,
                            case.placement,
                            c.cost_per_boot()
                        );
                        ordinal += 1;
                    }
                }
                for trace in &child.dag().step().trace {
                    println!("TRACE {trace:?}");
                }
            }
        }
        println!("INFERENCE {:?}", case.inference.predicted_counts());
        for (s, child) in case.inference.children().iter().enumerate() {
            println!("FORWARD {s} {:?}", child.predicted_counts());
            if s + 1 < case.g || case.placement == PackedNiPlacement::Aa {
                for id in 0..3 {
                    println!(
                        "ROLE ordinal={ordinal} G{} {:?} inference stage={s} record={id} cost={:?}",
                        case.g,
                        case.placement,
                        c.cost_per_boot()
                    );
                    ordinal += 1;
                }
            }
            for trace in child.trace() {
                println!("FORWARD_TRACE {trace:?}");
            }
        }
        println!("TOTAL {:?}", case.total);
        total += case.total.bootstraps;
    }
    println!("SUCCESSFUL_SCHEDULE_BOOTSTRAPS {total}");
    assert_eq!(total, 332);
    assert_eq!(ordinal, 332);
    assert!(total <= 372);
    let mut client_calls = 0;
    for g in [2, 3] {
        let mut counts = OpCounter::default();
        for (t, plans) in fixture::interactive_schedule(&p, &enc, g)
            .iter()
            .enumerate()
        {
            for (s, plan) in plans.iter().enumerate() {
                counts += &plan.step().predicted_counts;
                println!(
                    "I G{g} update={t} stage={s} counts={:?}",
                    plan.step().predicted_counts
                );
                for tr in &plan.step().trace {
                    println!("I_TRACE {tr:?}");
                }
                for role in [
                    PackedRefreshRole::Weight(2 * s),
                    PackedRefreshRole::Velocity(2 * s),
                    PackedRefreshRole::Weight(2 * s + 1),
                    PackedRefreshRole::Velocity(2 * s + 1),
                ] {
                    println!(
                        "I_ROLE ordinal={client_calls} G{g} update={t} {role:?} server_delta=zero"
                    );
                    client_calls += 1;
                }
                if s + 1 < g {
                    for &id in fixture::IDS[t] {
                        println!(
                            "I_ROLE ordinal={client_calls} G{g} update={t} stage={s} hidden_record={id} server_delta=zero"
                        );
                        client_calls += 1;
                    }
                }
            }
        }
        let inference = OpCounter {
            rotations: if g == 2 { 54 } else { 66 },
            relins: if g == 2 { 15 } else { 21 },
            ct_ct_tensor: if g == 2 { 15 } else { 21 },
            ct_pt_mults: if g == 2 { 21 } else { 30 },
            rescales: if g == 2 { 36 } else { 51 },
            sse_switches: 0,
            bootstraps: 0,
        };
        let mut planned_inference = OpCounter::default();
        for (s, plan) in fixture::interactive_forward_schedule(&p, &enc, g)
            .iter()
            .enumerate()
        {
            planned_inference += plan.predicted_counts();
            println!("I_FORWARD G{g} stage={s} {:?}", plan.predicted_counts());
            for tr in plan.trace() {
                println!("I_FORWARD_TRACE {tr:?}");
            }
        }
        assert_eq!(planned_inference, inference);
        counts += &inference;
        println!("I_INFERENCE G{g} {inference:?}");
        for s in 0..g - 1 {
            for id in 0..3 {
                println!(
                    "I_ROLE ordinal={client_calls} G{g} inference stage={s} record={id} server_delta=zero"
                );
                client_calls += 1;
            }
        }
        println!("I_TOTAL G{g} {counts:?}");
    }
    assert_eq!(client_calls, 116);
    println!("SUCCESSFUL_I_CLIENT_CALLS {client_calls}");
}
#[test]
fn block_scalar_old_forward_and_local_target_isolation() {
    let mut states = vec![fixture::ScalarStage::new(); 3];
    let targets = vec![fixture::Y.to_vec(); 3];
    let mut changed = states.clone();
    let mut altered = targets.clone();
    altered[2][0][0] = 0.25;
    let old = fixture::scalar_step(&mut states, 0, &targets);
    let alternate = fixture::scalar_step(&mut changed, 0, &altered);
    assert_eq!(states[0].w, changed[0].w);
    assert_eq!(states[1].w, changed[1].w);
    assert_ne!(states[2].w, changed[2].w);
    assert_eq!(old[1].input, old[0].hidden);
    assert_eq!(old[2].input, old[1].hidden);
    assert_eq!(old[2].input, alternate[2].input);
    let mut first = vec![fixture::ScalarStage::new(); 3];
    let mut first_targets = targets.clone();
    first_targets[0][0][0] = 0.25;
    let first_snapshots = fixture::scalar_step(&mut first, 0, &first_targets);
    assert_ne!(first[0].w, states[0].w);
    assert_eq!(first[1].w, states[1].w);
    assert_eq!(first[2].w, states[2].w);
    assert_eq!(first_snapshots[0].hidden, old[0].hidden);
    assert_eq!(first_snapshots[1].input, old[1].input);

    for t in 1..4 {
        let snaps = fixture::scalar_step(&mut states, t, &targets);
        assert!(
            snaps
                .iter()
                .flat_map(|s| s.hidden.iter().flatten())
                .all(|v| v.abs() < 0.5)
        );
    }
    let inf = fixture::scalar_forward(&states);
    let max = inf[0].hidden.iter().flatten().copied().fold(0.0, f64::max);
    assert!((max - 0.5292030932466849).abs() < 1e-14);
    assert!(max > 0.5 && max < 0.6);
    for state in &states {
        for k in 0..2 {
            assert!(
                state.w[k]
                    .iter()
                    .flatten()
                    .zip(fixture::W[k].iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max)
                    > 4. * fixture::E
            );
            assert!(
                state
                    .u(k)
                    .iter()
                    .flatten()
                    .map(|v| v.abs())
                    .fold(0., f64::max)
                    > 4. * fixture::E
            );
        }
    }
}

#[test]
#[ignore = "D4c3 bounded I primary: requires reviewed source, frozen metadata and explicit root I phase release"]
fn block_interactive_original_fixture() {
    packed_blocks_helpers::primary::interactive();
}
#[test]
#[ignore = "D4c3 true NI primary: requires reviewed source, frozen metadata and explicit root NI phase release"]
fn block_ni_original_fixture() {
    packed_blocks_helpers::primary::ni();
}

#[test]
fn block_aa_exact_selected_payload_and_known_next_runtime() {
    let (p, enc, contract) = parameters();
    let frozen = fixture::schedules(&p, &enc, &contract);
    for schedule in frozen {
        let ev = Tags::new(p.clone());
        let mut graph = packed_blocks_helpers::graph(&ev, schedule.g);
        let boot = boot(contract.clone());
        let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, schedule.placement).unwrap();
        for (t, &rate) in fixture::RATES.iter().enumerate() {
            let batch = batch(&ev, &graph, fixture::IDS[t]);
            let plan = server
                .plan_update(
                    &graph,
                    &batch,
                    rate,
                    GradRelin::Lazy,
                    fixture::next(&p, graph.geometry(), t),
                )
                .unwrap();
            assert_eq!(
                plan.predicted_counts(),
                schedule.updates[t].predicted_counts()
            );
            let start = ev.muls.borrow().len();
            let (candidate, receipt) = server.prepare_planned(&graph, &batch, &plan).unwrap();
            assert_eq!(receipt.valid_ids(), fixture::IDS[t]);
            for (s, child) in receipt.children().iter().enumerate() {
                assert_eq!(child.step().plan.valid_samples, fixture::IDS[t].len());
                if schedule.placement == PackedNiPlacement::Aa {
                    for (j, post) in child
                        .step()
                        .forward()
                        .hidden_post()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let tag = post.ciphertext().tag;
                        let uses = ev.muls.borrow()[start..]
                            .iter()
                            .filter(|&&(a, b)| a == tag || b == tag)
                            .count();
                        assert_eq!(uses, if s + 1 < schedule.g { 4 } else { 2 });
                        if s + 1 < schedule.g {
                            let handoff = &receipt.handoffs()[s * fixture::IDS[t].len() + j];
                            assert_eq!(handoff.source().ciphertext().tag, tag);
                            assert_eq!(handoff.output().unwrap().ciphertext().tag, tag);
                            assert_eq!(handoff.counter_delta(), Some(&OpCounter::default()));
                            assert!(handoff.bootstrap().is_none());
                            let w = graph.stages()[s + 1].layers()[0].weights().ciphertext().tag;
                            assert!(
                                ev.muls.borrow()[start..]
                                    .iter()
                                    .any(|&(a, b)| (a == tag && b == w) || (a == w && b == tag))
                            );
                        }
                    }
                }
            }
            let next = candidate.into_graph();
            let before = ev.counter();
            assert!(server.prepare_planned(&next, &batch, &plan).is_err());
            assert_eq!(ev.counter(), before);
            graph = next;
        }
        let records = (0..3)
            .map(|id| {
                PackedForwardRecord::new(
                    id,
                    PackedVector::new(
                        &ev,
                        graph.geometry().layers()[0].input(),
                        ev.ct(16, p.scale()),
                    )
                    .unwrap(),
                    true,
                )
            })
            .collect();
        let input =
            PackedForwardBatch::new(&ev, &graph, records, vec![PackedPartition::new(0, 3, 3)])
                .unwrap();
        for (s, stage) in graph.stages().iter().enumerate() {
            for layer in stage.layers() {
                layer
                    .velocity()
                    .unwrap()
                    .ciphertext()
                    .forbid_reads
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            if s + 1 < schedule.g {
                stage.layers()[1]
                    .weights()
                    .ciphertext()
                    .forbid_reads
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        let plan = server
            .plan_forward(&graph, &input, schedule.inference.policy())
            .unwrap();
        let (_, receipt) = server.prepare_forward(&graph, &input, &plan).unwrap();
        assert_eq!(
            receipt.actual_counts(),
            schedule.inference.predicted_counts()
        );
        assert_eq!(ev.counter(), schedule.total);
    }
}
#[test]
fn block_complete_preflight_rejects_bad_targets_ids_partitions_and_future() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p.clone());
    let graph = graph(&ev, 2);
    let b = boot(c);
    let server = PackedBlockNiTrainer::new(&ev, &enc, &b, PackedNiPlacement::Wn).unwrap();
    for bad in 0..6 {
        let mut records = records(&ev, &graph, &[0, 1, 2]);
        let r = records[3].clone();
        let mut targets = r.targets().to_vec();
        match bad {
            0 => {
                targets.pop();
            }
            1 => targets.push(targets[0].clone()),
            2 => {
                targets[1] = PackedVector::new(
                    &ev,
                    graph.geometry().layers()[1].output(),
                    ev.ct(16, p.scale()),
                )
                .unwrap()
            }
            _ => {}
        }
        if bad <= 2 {
            records[3] = PackedGraphRecord::new(3, r.input().clone(), targets, false);
        }
        if bad == 3 {
            records.swap(0, 1);
        }
        let parts = match bad {
            4 => vec![PackedPartition::new(0, 3, 2), PackedPartition::new(2, 4, 1)],
            5 => vec![PackedPartition::new(0, 4, 4)],
            _ => vec![PackedPartition::new(0, 4, 3)],
        };
        assert!(PackedGraphBatch::new(&ev, &graph, records, parts).is_err());
        assert_eq!(ev.counter(), OpCounter::default());
    }
    let batch = batch(&ev, &graph, &[0, 2]);
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let wrong = PackedGraphBatchMeta::new(
        p.clone(),
        graph.geometry(),
        7,
        vec![0, 2],
        vec![h; 2],
        vec![vec![h; 2]; 2],
        vec![PackedPartition::new(0, 4, 2)],
    )
    .unwrap();
    assert!(
        server
            .prepare_update(
                &graph,
                &batch,
                0.04,
                GradRelin::Lazy,
                PackedGraphNext::Known(
                    wrong,
                    PackedUpdateMeta::new(0.01, GradRelin::Lazy, true).unwrap()
                )
            )
            .is_err()
    );
    let next = fixture::meta_batch(&p, graph.geometry(), 1);
    assert!(
        server
            .prepare_update(
                &graph,
                &batch,
                0.04,
                GradRelin::Lazy,
                PackedGraphNext::Known(
                    next,
                    PackedUpdateMeta::new(1e-100, GradRelin::Lazy, true).unwrap()
                )
            )
            .is_err()
    );
    ev.missing.set(Some(-2));
    assert!(
        server
            .plan_update(
                &graph,
                &batch,
                0.04,
                GradRelin::Lazy,
                PackedGraphNext::Unknown
            )
            .is_err()
    );
    assert_eq!(ev.counter(), OpCounter::default());
    assert_eq!(b.calls.get(), 0);
}
#[test]
fn block_interactive_full_schedule_and_inference_are_116_callbacks() {
    let (p, enc, _) = parameters();
    let ev = Tags::new(p);
    let server = PackedBlockInteractiveTrainer::new(&ev, &enc).unwrap();
    let mut client = Client {
        ev: &ev,
        contract: PackedRefreshContract::new(&ev, 16).unwrap(),
        calls: 0,
        replay_at: None,
        previous: None,
    };
    for n in [2, 3] {
        let mut graph = graph(&ev, n);
        for (t, &rate) in fixture::RATES.iter().enumerate() {
            let batch = batch(&ev, &graph, fixture::IDS[t]);
            let before = client.calls;
            let (candidate, receipt) = server
                .prepare_update(&graph, &batch, rate, GradRelin::Lazy, &mut client)
                .unwrap();
            assert_eq!(
                client.calls - before,
                4 * n + fixture::IDS[t].len() * (n - 1)
            );
            assert_eq!(receipt.handoffs().len(), fixture::IDS[t].len() * (n - 1));
            graph = candidate.into_graph();
        }
        let records = (0..3)
            .map(|id| {
                PackedForwardRecord::new(
                    id,
                    PackedVector::new(
                        &ev,
                        graph.geometry().layers()[0].input(),
                        ev.ct(16, ev.params.scale()),
                    )
                    .unwrap(),
                    true,
                )
            })
            .collect();
        let input =
            PackedForwardBatch::new(&ev, &graph, records, vec![PackedPartition::new(0, 3, 3)])
                .unwrap();
        let (_, r) = server
            .prepare_forward(
                &graph,
                &input,
                PackedInferencePlan::ClientBoundary,
                &mut client,
            )
            .unwrap();
        assert_eq!(r.handoffs().len(), 3 * (n - 1));
        for stage in graph.stages() {
            assert_eq!(stage.layers()[0].update_index(), 4);
        }
    }
    assert_eq!(client.calls, 116);
    assert_eq!(ev.counter().bootstraps, 0);
}
#[test]
fn block_relin_controls_and_unrestored_inference_are_explicit() {
    let (p, enc, c) = parameters();
    let mut ev = Tags::new(p);
    let graph = graph(&ev, 2);
    let batch = batch(&ev, &graph, &[0, 1, 2]);
    let boot = boot(c);
    let lazy = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au)
        .unwrap()
        .plan_update(
            &graph,
            &batch,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    let per = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au)
        .unwrap()
        .plan_update(
            &graph,
            &batch,
            0.04,
            GradRelin::PerSample,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    assert_eq!(
        per.predicted_counts().relins - lazy.predicted_counts().relins,
        8
    );
    ev.cfg.lazy_relin = false;
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au).unwrap();
    let disabled = server
        .plan_update(
            &graph,
            &batch,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    assert_eq!(disabled.predicted_counts(), per.predicted_counts());
    let _ = server.prepare_planned(&graph, &batch, &disabled).unwrap();
    let input = PackedForwardBatch::new(
        &ev,
        &graph,
        (0..3)
            .map(|id| {
                PackedForwardRecord::new(
                    id,
                    PackedVector::new(
                        &ev,
                        graph.geometry().layers()[0].input(),
                        ev.ct(16, ev.params.scale()),
                    )
                    .unwrap(),
                    true,
                )
            })
            .collect(),
        vec![PackedPartition::new(0, 3, 3)],
    )
    .unwrap();
    let plan = server
        .plan_forward(&graph, &input, PackedInferencePlan::NoRefresh)
        .unwrap();
    assert_eq!(
        plan.predicted_counts(),
        &OpCounter {
            rotations: 54,
            relins: 15,
            ct_ct_tensor: 15,
            ct_pt_mults: 21,
            rescales: 36,
            sse_switches: 0,
            bootstraps: 0
        }
    );
    let before = boot.calls.get();
    server.prepare_forward(&graph, &input, &plan).unwrap();
    assert_eq!(boot.calls.get(), before);
}
#[test]
fn block_context_representation_and_generation_preflight() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p.clone());
    let original = graph(&ev, 2);
    let bootstrap = boot(c);
    let trainer = PackedBlockNiTrainer::new(&ev, &enc, &bootstrap, PackedNiPlacement::Au).unwrap();
    let (foreign_p, _, _) = parameters();
    let foreign = Tags::new(foreign_p);
    let spec = original.geometry().layers()[0].input();
    for kind in 0..3 {
        let mut r = records(&ev, &original, &[0, 1, 2]);
        let input = if kind == 0 {
            PackedVector::new(&foreign, spec, foreign.ct(16, foreign.params.scale())).unwrap()
        } else {
            PackedVector::new(
                &ev,
                spec,
                ev.ct(
                    if kind == 1 { 0 } else { 16 },
                    if kind == 2 { p.scale() * 2. } else { p.scale() },
                ),
            )
            .unwrap()
        };
        r[3] = PackedGraphRecord::new(3, input, r[3].targets().to_vec(), false);
        let result = PackedGraphBatch::new(&ev, &original, r, vec![PackedPartition::new(0, 4, 3)]);
        if kind == 1 {
            assert!(
                result.is_ok(),
                "invalid holes may be level zero but must be valid representations"
            );
        } else {
            assert!(result.is_err());
        }
    }
    let mut wrong = ev.ct(16, p.scale());
    wrong.size = 3;
    assert!(PackedVector::new(&ev, spec, wrong).is_err());
    let data = batch(&ev, &original, &[0, 2]);
    let plan = trainer
        .plan_update(
            &original,
            &data,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Unknown,
        )
        .unwrap();
    assert!(matches!(plan.next(), PackedGraphNext::Unknown));
    let other = graph(&ev, 2);
    assert!(trainer.prepare_planned(&other, &data, &plan).is_err());
    assert_eq!(ev.counter(), OpCounter::default());
    let g = geometry(2);
    let swapped = (0..2)
        .rev()
        .map(|s| {
            PackedStage::new(
                &ev,
                &g,
                s,
                std::array::from_fn(|k| {
                    PackedWeights::new(&ev, g.layers()[2 * s + k].clone(), ev.ct(16, p.scale()))
                        .unwrap()
                }),
                None,
                PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
            )
            .unwrap()
        })
        .collect();
    assert!(PackedBlockGraph::new(&ev, g, swapped).is_err());
    assert_eq!(ev.counter(), OpCounter::default());
}

#[test]
fn fix1_returned_invalid_graph_handoff_is_completed_not_validated() {
    let (p, enc, c) = parameters();
    for placement in [PackedNiPlacement::Au, PackedNiPlacement::Wn] {
        let ev = Tags::new(p.clone());
        let old = graph(&ev, 2);
        let data = batch(&ev, &old, &[0, 2]);
        let boot = boot(c.clone());
        boot.bad_at.set(Some(5));
        let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, placement).unwrap();
        let plan = server
            .plan_update(
                &old,
                &data,
                0.04,
                GradRelin::Lazy,
                PackedGraphNext::Terminal,
            )
            .unwrap();
        let f = server.prepare_planned(&old, &data, &plan).unwrap_err();
        let mut expected = plan.children()[0].predicted_counts().clone();
        expected += c.cost_per_boot();
        assert_eq!(f.phase(), PackedGraphPhase::Handoff);
        assert_eq!(f.stage(), 0);
        assert_eq!(
            f.role(),
            Some(PackedRefreshRole::Hidden {
                stage: 0,
                record: 0
            })
        );
        assert_eq!(f.children().len(), 1);
        assert!(f.child_failure().is_none());
        assert_eq!(f.counter_delta(), Some(&expected));
        assert_eq!(ev.counter(), expected);
        assert_eq!(f.handoffs().len(), 1);
        let h = &f.handoffs()[0];
        assert!(h.attempted());
        assert!(
            h.completed(),
            "the capability returned a raw wrong-level ciphertext"
        );
        assert!(!h.validated());
        assert!(h.output().is_none());
        assert!(h.bootstrap().is_none());
        assert_eq!(h.counter_delta(), Some(c.cost_per_boot()));
        let raw = h.raw_attempt().unwrap();
        assert_eq!(
            raw.role(),
            PackedRefreshRole::Hidden {
                stage: 0,
                record: 0
            }
        );
        assert_eq!(raw.input().level, 0);
        assert_eq!(raw.output().unwrap().level, 15);
        assert!(raw.completed());
        assert!(!raw.validated());
        assert_eq!(raw.counter_delta(), Some(c.cost_per_boot()));
    }
}
#[test]
fn fix1_partial_forward_observations_keep_physical_ids_and_raw_failed_response() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let inputs = forward_batch(&ev, &old, &[0, 2]);
    let boot = boot(c.clone());
    boot.bad_at.set(Some(3));
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Aa).unwrap();
    let plan = server
        .plan_forward(&old, &inputs, PackedInferencePlan::AllHiddenBootstrap)
        .unwrap();
    let f = server.prepare_forward(&old, &inputs, &plan).unwrap_err();
    assert_eq!(f.stage(), 1);
    assert_eq!(f.children().len(), 1);
    assert_eq!(
        f.children()[0]
            .observations()
            .iter()
            .map(|o| o.id())
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    let child = f.child_failure().unwrap();
    assert_eq!(
        child.role(),
        Some(PackedRefreshRole::Hidden {
            stage: 1,
            record: 0
        })
    );
    let observations = child.observations();
    assert_eq!(observations.len(), 1);
    let observation = &observations[0];
    assert_eq!(observation.id(), 0);
    assert!(observation.preactivation().is_some());
    assert!(observation.hidden_pre().is_some());
    assert!(observation.hidden_post().is_none());
    assert!(observation.output().is_none());
    let raw = observation.bootstrap_attempt().unwrap();
    assert!(raw.completed());
    assert!(!raw.validated());
    assert_eq!(raw.output().unwrap().level, 15);
}
#[test]
fn fix1_late_forward_failure_keeps_completed_and_partial_work() {
    let (p, enc, _) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let inputs = forward_batch(&ev, &old, &[0, 2]);
    let mut client = Client {
        ev: &ev,
        contract: PackedRefreshContract::new(&ev, 16).unwrap(),
        calls: 0,
        replay_at: None,
        previous: None,
    };
    ev.fail_tensor.set(Some(8));
    let server = PackedBlockInteractiveTrainer::new(&ev, &enc).unwrap();
    let f = server
        .prepare_forward(
            &old,
            &inputs,
            PackedInferencePlan::ClientBoundary,
            &mut client,
        )
        .unwrap_err();
    assert_eq!(f.stage(), 1);
    assert_eq!(f.children().len(), 1);
    assert_eq!(f.handoffs().len(), 2);
    let partial = f.child_failure().unwrap().observations();
    assert_eq!(
        partial.iter().map(|o| o.id()).collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert!(partial[0].output().is_some());
    assert!(partial[1].affine().is_none());
    assert!(partial[1].preactivation().is_none());
    assert!(partial[1].output().is_none());
    assert_eq!(
        f.counter_delta(),
        Some(&OpCounter {
            rotations: 25,
            relins: 7,
            ct_ct_tensor: 8,
            ct_pt_mults: 10,
            rescales: 17,
            sse_switches: 0,
            bootstraps: 0
        })
    );
}
#[test]
fn fix1_frozen_attempt_roles_cover_failed_inputs_and_phase_bounds() {
    use packed_blocks_helpers::primary::{
        AttemptPhase, RecordedBoot, attempt_range, frozen_attempts,
    };
    let (p, enc, c) = parameters();
    let frozen = fixture::schedules(&p, &enc, &c);
    let identities = frozen_attempts(&frozen);
    assert_eq!(identities.len(), 332);
    assert_eq!(
        identities[4].role,
        PackedRefreshRole::Hidden {
            stage: 0,
            record: 0
        }
    );
    assert_eq!(identities[4].phase, AttemptPhase::Training);
    for (ordinal, id) in identities.iter().enumerate() {
        assert_eq!(id.ordinal, ordinal);
        let bound = id.bound();
        assert!(attempt_range(*id, &vec![(bound, 0.0); 512]).2);
        assert!(!attempt_range(*id, &vec![(bound + fixture::EPS, 0.0); 512]).2);
        assert!(!attempt_range(*id, &vec![(0.0, f64::NAN); 512]).2);
    }
    assert!(
        identities
            .iter()
            .any(|id| id.phase == AttemptPhase::Inference && id.bound() == 0.6)
    );
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let data = batch(&ev, &old, &[0, 1, 2]);
    let fake = boot(c);
    fake.bad_at.set(Some(5));
    let recorded = RecordedBoot::new(&fake, identities);
    let server = PackedBlockNiTrainer::new(&ev, &enc, &recorded, PackedNiPlacement::Au).unwrap();
    let f = server
        .prepare_update(
            &old,
            &data,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap_err();
    let attempts = recorded.attempts.borrow();
    let failed = &attempts[4];
    assert_eq!(failed.identity.role, f.role().unwrap());
    assert_eq!(failed.identity.stage, 0);
    assert_eq!(failed.identity.record, Some(0));
    assert_eq!(failed.identity.update, Some(0));
    assert_eq!(failed.input.level, 0);
    assert_eq!(failed.output.as_ref().unwrap().level, 15);
    assert!(!failed.response_validated.get());
    assert_eq!(failed.identity.bound(), 0.5);
    assert_eq!(attempts.len(), 5);
}
#[test]
fn fix1_external_failure_walk_covers_i_and_ni_partial_observations() {
    use packed_blocks_helpers::primary::visit_forward_failure;
    let (p, enc, c) = parameters();
    for ni in [false, true] {
        let ev = Tags::new(p.clone());
        let old = graph(&ev, 2);
        let input = forward_batch(&ev, &old, &[0, 2]);
        let fake = boot(c.clone());
        let f = if ni {
            fake.bad_at.set(Some(3));
            let server =
                PackedBlockNiTrainer::new(&ev, &enc, &fake, PackedNiPlacement::Aa).unwrap();
            let plan = server
                .plan_forward(&old, &input, PackedInferencePlan::AllHiddenBootstrap)
                .unwrap();
            server.prepare_forward(&old, &input, &plan).unwrap_err()
        } else {
            ev.fail_tensor.set(Some(8));
            let mut client = Client {
                ev: &ev,
                contract: PackedRefreshContract::new(&ev, 16).unwrap(),
                calls: 0,
                replay_at: None,
                previous: None,
            };
            PackedBlockInteractiveTrainer::new(&ev, &enc)
                .unwrap()
                .prepare_forward(
                    &old,
                    &input,
                    PackedInferencePlan::ClientBoundary,
                    &mut client,
                )
                .unwrap_err()
        };
        let mut visited = vec![];
        visit_forward_failure(&f, |s, id, kind, ct| {
            visited.push((s, id, kind.to_owned(), ct.tag))
        });
        assert!(visited.iter().all(|(_, id, _, _)| *id == 0 || *id == 2));
        assert_eq!(visited.len(), if ni { 22 } else { 26 });
        assert!(
            visited
                .iter()
                .any(|(s, id, k, _)| *s == 0 && *id == 2 && k == "output")
        );
        assert!(
            visited
                .iter()
                .any(|(s, id, k, _)| *s == 1 && *id == 0 && k == "input")
        );
        if ni {
            let raw = f.child_failure().unwrap().observations()[0]
                .bootstrap_attempt()
                .unwrap();
            assert!(visited.iter().any(|(s, id, k, tag)| *s == 1
                && *id == 0
                && k == "raw_boot_output"
                && *tag == raw.output().unwrap().tag));
        } else {
            assert!(
                visited
                    .iter()
                    .any(|(s, id, k, _)| *s == 1 && *id == 2 && k == "input")
            );
            assert_eq!(
                visited
                    .iter()
                    .filter(|(_, _, k, _)| k == "handoff_response")
                    .count(),
                2
            );
        }
    }
}
#[test]
fn fix1_initialized_u_and_present_bias_survive_final_stage_failure() {
    let (p, enc, c) = parameters();
    for square in [false, true] {
        let ev = Tags::new(p.clone());
        let initial = if square {
            square_graph(&ev, 2)
        } else {
            graph(&ev, 2)
        };
        let boot = boot(c.clone());
        let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au).unwrap();
        let init = batch(&ev, &initial, &[0, 1, 2]);
        let (old, _) = server
            .prepare_update(
                &initial,
                &init,
                0.04,
                GradRelin::Lazy,
                PackedGraphNext::Terminal,
            )
            .unwrap();
        let old = old.into_graph();
        let tags = state_tags(&old);
        if square {
            assert!(
                tags.iter()
                    .all(|(w, u, b, i)| u.is_none() && b.is_some_and(|b| b != *w) && *i == 1)
            );
        } else {
            assert!(
                tags.iter()
                    .all(|(w, u, b, i)| u.is_some_and(|u| u != *w) && b.is_none() && *i == 1)
            );
            let mut scalar = vec![fixture::ScalarStage::new(); 2];
            fixture::scalar_step(&mut scalar, 0, &vec![fixture::Y.to_vec(); 2]);
            assert!(
                scalar
                    .iter()
                    .all(|s| s.u(0).iter().flatten().any(|u| u.abs() > 4.0 * fixture::E))
            );
        }
        let data = batch(&ev, &old, &[0]);
        let plan = server
            .plan_update(
                &old,
                &data,
                0.01,
                GradRelin::Lazy,
                PackedGraphNext::Terminal,
            )
            .unwrap();
        boot.bad_at.set(Some(boot.calls.get() + 7));
        let before = ev.counter();
        let failure = server.prepare_planned(&old, &data, &plan).unwrap_err();
        assert_eq!(failure.phase(), PackedGraphPhase::Stage);
        assert_eq!(failure.stage(), 1);
        assert_eq!(failure.children().len(), 1);
        assert_eq!(failure.handoffs().len(), 1);
        assert!(failure.handoffs()[0].validated());
        let role = if square {
            PackedRefreshRole::Bias(2)
        } else {
            PackedRefreshRole::Velocity(2)
        };
        assert_eq!(failure.role(), Some(role));
        let Some(PackedGraphChildFailure::Ni(child)) = failure.child_failure() else {
            panic!("typed partial child required")
        };
        assert_eq!(child.phase(), PackedNiPhase::Refresh);
        assert_eq!(child.role(), Some(role));
        assert_eq!(child.events().len(), 1);
        assert_eq!(child.events()[0].role(), PackedRefreshRole::Weight(2));
        assert_eq!(child.updated_stage().unwrap().layers()[0].update_index(), 2);
        let mut partial = plan.children()[1].dag().step().predicted_counts.clone();
        partial += &OpCounter {
            ct_pt_mults: 2,
            rescales: 2,
            ..Default::default()
        };
        partial += c.cost_per_boot();
        partial += c.cost_per_boot();
        assert_eq!(child.counter_delta(), Some(&partial));
        let mut total = plan.children()[0].predicted_counts().clone();
        total += c.cost_per_boot();
        total += &partial;
        assert_eq!(failure.counter_delta(), Some(&total));
        let mut expected = before;
        expected += &total;
        assert_eq!(ev.counter(), expected);
        assert_eq!(state_tags(&old), tags);
        boot.bad_at.set(None);
        let (candidate, receipt) = server.prepare_planned(&old, &data, &plan).unwrap();
        assert_eq!(receipt.actual_counts(), plan.predicted_counts());
        assert_eq!(state_tags(&old), tags);
        assert!(
            candidate
                .graph()
                .stages()
                .iter()
                .all(|s| s.layers()[0].update_index() == 2)
        );
    }
}
#[test]
fn fix1_independent_captured_reverse_backbone_and_unequal_group_mean_proofs() {
    for g in [2, 3] {
        let mut sequential = vec![fixture::ScalarStage::new(); g];
        let targets = vec![fixture::Y.to_vec(); g];
        for t in 0..4 {
            let old = sequential.clone();
            let (reversed, snapshots) = fixture::reverse_step(&old, t, &targets);
            let forward = fixture::scalar_step(&mut sequential, t, &targets);
            for s in 0..g {
                assert_eq!(reversed[s].w, sequential[s].w);
                assert_eq!(reversed[s].v, sequential[s].v);
                assert_eq!(snapshots[s].input, forward[s].input);
                assert_eq!(snapshots[s].z, forward[s].z);
                assert_eq!(snapshots[s].hidden, forward[s].hidden);
                assert_eq!(snapshots[s].output, forward[s].output);
                assert_eq!(snapshots[s].grad, forward[s].grad);
                if s + 1 < g {
                    assert_eq!(snapshots[s + 1].input, snapshots[s].hidden);
                }
            }
        }
        let old = vec![fixture::ScalarStage::new(); g];
        let (_, base) = fixture::reverse_step(&old, 0, &targets);
        let mut changed = old.clone();
        changed[0].w[0][0][0] += 1.0 / 16.0;
        let (_, propagated) = fixture::reverse_step(&changed, 0, &targets);
        assert_ne!(base[1].input, propagated[1].input);
        let first = fixture::capture_local(&old[0], &fixture::X[..2], &fixture::Y[..2]);
        let tail = fixture::capture_local(&old[0], &fixture::X[2..3], &fixture::Y[2..3]);
        let mut wrong_difference = 0.0_f64;
        for k in 0..2 {
            for o in 0..3 {
                for i in 0..3 {
                    let correct = base[0].grad[k][o][i];
                    let weighted = (2.0 * first.grad[k][o][i] + tail.grad[k][o][i]) / 3.0;
                    let wrong = (first.grad[k][o][i] + tail.grad[k][o][i]) / 2.0;
                    assert!((correct - weighted).abs() < 1e-16);
                    wrong_difference = wrong_difference.max((correct - wrong).abs());
                }
            }
        }
        assert!((wrong_difference - 0.00866062636487).abs() < 1e-14);
    }
}
#[test]
fn fix1_capability_error_keeps_attempt_identity_input_and_partial_interval() {
    let (p, enc, c) = parameters();
    let identities =
        packed_blocks_helpers::primary::frozen_attempts(&fixture::schedules(&p, &enc, &c));
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let data = batch(&ev, &old, &[0, 1, 2]);
    let fake = boot(c.clone());
    fake.error_at.set(Some(5));
    let recorded = packed_blocks_helpers::primary::RecordedBoot::new(&fake, identities);
    let server = PackedBlockNiTrainer::new(&ev, &enc, &recorded, PackedNiPlacement::Au).unwrap();
    let plan = server
        .plan_update(
            &old,
            &data,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    let f = server.prepare_planned(&old, &data, &plan).unwrap_err();
    let h = &f.handoffs()[0];
    assert!(h.attempted());
    assert!(!h.completed());
    assert!(!h.validated());
    let raw = h.raw_attempt().unwrap();
    assert_eq!(raw.input().level, 0);
    assert!(raw.output().is_none());
    assert_eq!(raw.counter_delta(), Some(&partial_boot_cost()));
    let mut expected = plan.children()[0].predicted_counts().clone();
    expected += &partial_boot_cost();
    assert_eq!(f.counter_delta(), Some(&expected));
    assert_eq!(ev.counter(), expected);
    let attempts = recorded.attempts.borrow();
    assert_eq!(attempts.last().unwrap().identity.role, f.role().unwrap());
    assert!(attempts.last().unwrap().output.is_none());
    assert_eq!(attempts.last().unwrap().identity.bound(), 0.5);
}
#[test]
fn fix1_final_forward_verification_failure_retains_all_completed_records() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let input = forward_batch(&ev, &old, &[0, 2]);
    let boot = boot(c);
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au).unwrap();
    let plan = server
        .plan_forward(&old, &input, PackedInferencePlan::NoRefresh)
        .unwrap();
    ev.extra_count_tensor.set(Some(4));
    let failure = server.prepare_forward(&old, &input, &plan).unwrap_err();
    assert!(failure.children().is_empty());
    let child = failure.child_failure().unwrap();
    assert_eq!(child.trace(), plan.children()[0].trace());
    let mut expected = plan.children()[0].predicted_counts().clone();
    expected.sse_switches += 1;
    assert_eq!(child.counter_delta(), Some(&expected));
    assert_eq!(failure.counter_delta(), Some(&expected));
    assert_eq!(
        child
            .observations()
            .iter()
            .map(|o| o.id())
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert!(child.observations().iter().all(|o| o.output().is_some()));
    let mut outputs = 0;
    packed_blocks_helpers::primary::visit_forward_failure(&failure, |_, _, kind, _| {
        outputs += usize::from(kind == "output")
    });
    assert_eq!(outputs, 2);
}
#[test]
fn fix1_wrong_boundary_counter_retains_metadata_valid_raw_output() {
    let (p, enc, c) = parameters();
    let ev = Tags::new(p);
    let old = graph(&ev, 2);
    let data = batch(&ev, &old, &[0, 2]);
    let boot = boot(c.clone());
    boot.wrong_count_at.set(Some(5));
    let server = PackedBlockNiTrainer::new(&ev, &enc, &boot, PackedNiPlacement::Au).unwrap();
    let plan = server
        .plan_update(
            &old,
            &data,
            0.04,
            GradRelin::Lazy,
            PackedGraphNext::Terminal,
        )
        .unwrap();
    let failure = server.prepare_planned(&old, &data, &plan).unwrap_err();
    let h = &failure.handoffs()[0];
    assert!(h.completed());
    assert!(!h.validated());
    assert!(h.output().is_none());
    assert!(h.bootstrap().is_none());
    let raw = h.raw_attempt().unwrap();
    assert_eq!(raw.output().unwrap().level, 16);
    let mut cost = c.cost_per_boot().clone();
    cost.sse_switches += 1;
    assert_eq!(raw.counter_delta(), Some(&cost));
    let mut expected = plan.children()[0].predicted_counts().clone();
    expected += &cost;
    assert_eq!(failure.counter_delta(), Some(&expected));
}
