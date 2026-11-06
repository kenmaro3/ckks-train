//! Metadata-only NI admission; no keys, encrypted data or precision claim.
use ckks_train_ckks::{CkksParams, Encoder, OpCounter, metadata::CtMeta};
use ckks_train_nn::{layout_reboot::GradRelin, packed_training::*};
use std::sync::Arc;

fn setup(
    square: bool,
    level: usize,
) -> (Arc<CkksParams>, Encoder, PackedStageMeta, PackedBatchMeta) {
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    let p = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
    let e = Encoder::new(p.clone());
    let g = PackedGeometryPlan::new(
        3,
        &[
            StageShape::new(3, 3, if square { 3 } else { 2 }).unwrap(),
            StageShape::new(3, 3, if square { 3 } else { 2 }).unwrap(),
        ],
        512,
    )
    .unwrap();
    let h = CtMeta::new(&p, level, p.scale()).unwrap();
    let config = if square {
        PackedStageConfig::square_sgd()
    } else {
        PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
    };
    let s = PackedStageMeta::new(
        p.clone(),
        &g,
        0,
        config,
        [h; 2],
        [None; 2],
        square.then_some([h; 2]),
        [0; 2],
    )
    .unwrap();
    let b = PackedBatchMeta::new(
        p.clone(),
        0,
        g.layers()[0].input(),
        g.layers()[1].output(),
        vec![
            PackedRecordMeta::new(0, h, h),
            PackedRecordMeta::new(2, h, h),
        ],
    )
    .unwrap();
    (p, e, s, b)
}
fn update(rate: f64, relin: GradRelin) -> PackedUpdateMeta {
    PackedUpdateMeta::new(rate, relin, true).unwrap()
}

#[test]
fn pure_fresh_depth_counter_and_actual_prime_witnesses() {
    for square in [false, true] {
        let (p, e, s, b) = setup(square, 16);
        let d = plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), None).unwrap();
        assert_eq!(d.step().minimum_state_level, if square { 1 } else { 4 });
        let counts = if square {
            OpCounter {
                rotations: 36,
                relins: 16,
                ct_ct_tensor: 18,
                ct_pt_mults: 20,
                rescales: 36,
                sse_switches: 0,
                bootstraps: 0,
            }
        } else {
            OpCounter {
                rotations: 36,
                relins: 12,
                ct_ct_tensor: 14,
                ct_pt_mults: 18,
                rescales: 30,
                sse_switches: 0,
                bootstraps: 0,
            }
        };
        assert_eq!(d.step().predicted_counts, counts);
        assert_eq!(d.step().trace[0].level, 15);
        assert_eq!(
            d.step().trace[0].scale.to_bits(),
            (p.scale() * p.scale() / p.q()[16] as f64).to_bits()
        );
        assert_eq!(
            d.step()
                .trace
                .iter()
                .filter_map(|x| x.sample)
                .collect::<std::collections::BTreeSet<_>>(),
            [0, 2].into()
        );
        assert!(!d.candidate().is_executable());
        let per = plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::PerSample), None).unwrap();
        let mut expected = counts;
        expected.relins += 2;
        assert_eq!(per.step().predicted_counts, expected);
    }
}
#[test]
fn scalar_underflow_and_low_target_reject_without_he() {
    let (p, e, s, b) = setup(false, 16);
    assert!(plan_stage_dag(&e, &s, &b, update(1e-100, GradRelin::Lazy), None).is_err());
    let low = CtMeta::new(&p, 0, p.scale()).unwrap();
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let b = PackedBatchMeta::new(
        p.clone(),
        0,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(2, h, low)],
    )
    .unwrap();
    assert!(plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), None).is_err());
}
#[test]
fn metadata_rejects_duplicate_record_ids_and_wrong_schedule() {
    let (p, e, s, b) = setup(false, 16);
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    assert!(
        PackedBatchMeta::new(
            p.clone(),
            0,
            b.input_spec(),
            b.target_spec(),
            vec![
                PackedRecordMeta::new(2, h, h),
                PackedRecordMeta::new(2, h, h)
            ]
        )
        .is_err()
    );
    let bad = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(0, h, h)],
    )
    .unwrap();
    assert!(plan_stage_dag(&e, &s, &bad, update(0.04, GradRelin::Lazy), None).is_err());
}
fn boot(p: Arc<CkksParams>) -> PackedBootstrapContract {
    let proof = ckks_train_ckks::bootstrap::metadata::DftSupportProof::verify_current().unwrap();
    let bp = ckks_train_ckks::bootstrap::BootParams {
        log_slots: 9,
        ..ckks_train_ckks::bootstrap::BootParams::default_n16()
    };
    PackedBootstrapContract::new(p, &bp, &proof).unwrap()
}
#[test]
fn aa_fresh_and_carried_histories_keep_real_u_and_projection_reserve() {
    let (p, e, s, b) = setup(false, 16);
    let c = boot(p.clone());
    let first = plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), Some(&c)).unwrap();
    assert_eq!(first.candidate().weights().map(|h| h.level()), [8, 11]);
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let next = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(0, h, h)],
    )
    .unwrap();
    let second = plan_stage_dag(
        &e,
        first.candidate(),
        &next,
        update(0.01, GradRelin::Lazy),
        Some(&c),
    )
    .unwrap();
    assert_eq!(second.candidate().weights().map(|h| h.level()), [2, 6]);
    let levels: Vec<_> = second
        .step()
        .trace
        .iter()
        .filter(|x| {
            [
                "z1",
                "hidden",
                "hidden.post_boot",
                "z2",
                "loss",
                "upstream",
                "derivative1",
                "dz1",
            ]
            .contains(&x.name.as_str())
        })
        .map(|x| (x.name.as_str(), x.level))
        .collect();
    assert_eq!(
        levels,
        vec![
            ("z1", 6),
            ("hidden", 4),
            ("hidden.post_boot", 16),
            ("z2", 9),
            ("loss", 8),
            ("upstream", 6),
            ("derivative1", 5),
            ("dz1", 4)
        ]
    );
    assert_eq!(
        second.candidate().velocities().map(|h| h.unwrap().level()),
        [2, 6]
    );
}
#[test]
fn ni_future_invalid_scalar_is_rejected_even_when_depth_fits() {
    let (p, e, s, b) = setup(false, 16);
    let c = boot(p.clone());
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let next = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(2, h, h)],
    )
    .unwrap();
    assert!(
        PackedNiPlan::new(
            &e,
            &s,
            &b,
            update(0.04, GradRelin::Lazy),
            PackedNextStep::Known(next, update(1e-100, GradRelin::Lazy)),
            PackedNiPlacement::Aa,
            &c
        )
        .is_err()
    );
}
#[test]
fn ni_known_next_skips_refresh_only_with_actual_carried_dag_admission() {
    let (p, e, s, b) = setup(false, 16);
    let c = boot(p.clone());
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let next = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(2, h, h)],
    )
    .unwrap();
    let plan = PackedNiPlan::new(
        &e,
        &s,
        &b,
        update(0.04, GradRelin::Lazy),
        PackedNextStep::Known(next, update(0.01, GradRelin::Lazy)),
        PackedNiPlacement::Aa,
        &c,
    )
    .unwrap();
    assert!(!plan.refresh_state());
    assert_eq!(plan.predicted_counts().bootstraps, 2);
    let terminal = PackedNiPlan::new(
        &e,
        &s,
        &b,
        update(0.04, GradRelin::Lazy),
        PackedNextStep::Terminal,
        PackedNiPlacement::Aa,
        &c,
    )
    .unwrap();
    assert!(terminal.refresh_state());
    assert_eq!(terminal.predicted_counts().bootstraps, 6);
}
#[path = "packed_ni_helpers/mod.rs"]
mod fixture;
#[test]
fn registered_schedule_is_encoder_checked_without_keys_and_frozen_before_he() {
    let (p, e, _, _) = setup(false, 16);
    let c = boot(p.clone());
    let mut total = 0;
    for square in [true, false] {
        for ordinal in 0..2 {
            for placement in [
                PackedNiPlacement::Au,
                PackedNiPlacement::Wn,
                PackedNiPlacement::Aa,
            ] {
                let plans = fixture::schedule(p.clone(), &e, &c, square, ordinal, placement);
                let events: Vec<_> = plans
                    .iter()
                    .map(|p| p.predicted_counts().bootstraps)
                    .collect();
                let expected = if placement == PackedNiPlacement::Aa {
                    if square {
                        vec![7, 5, 6, 7]
                    } else {
                        vec![3, 5, 2, 7]
                    }
                } else {
                    vec![4; 4]
                };
                assert_eq!(events, expected);
                for (u, plan) in plans.iter().enumerate() {
                    println!(
                        "SCHEDULE square={square} ordinal={ordinal} placement={placement:?} update={u} state_refresh={} weights={:?} U={:?} bias={:?} counts={:?}",
                        plan.refresh_state(),
                        plan.dag().candidate().weights(),
                        plan.dag().candidate().velocities(),
                        plan.dag().candidate().biases(),
                        plan.predicted_counts()
                    );
                    for t in &plan.dag().step().trace {
                        println!(
                            "TRACE square={square} ordinal={ordinal} placement={placement:?} update={u} record={:?} name={} level={} scale_bits={}",
                            t.sample,
                            t.name,
                            t.level,
                            t.scale.to_bits()
                        );
                    }
                }
                total += events.iter().sum::<u64>();
            }
        }
    }
    assert_eq!(total, 212);
}
#[test]
fn independent_scalar_fixture_learning_visibility_and_slot_maps() {
    for square in [false, true] {
        let mut s = fixture::Scalar::new(square);
        for t in 0..4 {
            let (h, y) = s.step(t);
            assert_eq!(h.len(), [3, 1, 2, 3][t]);
            assert!(h.iter().flatten().all(|x| x.abs() < 0.5));
            assert!(y.iter().flatten().all(|x| x.abs() < 0.1));
        }
        for k in 0..2 {
            let movement = s.w[k]
                .iter()
                .flatten()
                .zip(fixture::W[k].iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(movement > 4.0 * fixture::E);
            if square {
                let movement = s.b[k]
                    .iter()
                    .zip(fixture::B[k])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                assert!(movement > 4.0 * fixture::E);
            } else {
                assert!(
                    s.u(k).iter().flatten().map(|x| x.abs()).fold(0.0, f64::max) > 4.0 * fixture::E
                );
            }
        }
        let expected = if square {
            [0.00156116594257, 0.00279636965903]
        } else {
            [0.0244288490083, 0.0345990416005]
        };
        for (k, want) in expected.into_iter().enumerate() {
            let moved = s.w[k]
                .iter()
                .flatten()
                .zip(fixture::W[k].iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!((moved - want).abs() < 1e-12);
        }
    }
    let row = fixture::matrix(&fixture::W[0], true);
    let col = fixture::matrix(&fixture::W[0], false);
    assert_eq!((row[1], col[1]), (-0.0625, -0.125));
    assert_eq!((row[4], col[4]), (-0.125, -0.0625));
}
#[test]
fn ni_low_input_target_reserve_and_bad_future_schedule_fail_closed() {
    let (p, e, s, b) = setup(false, 16);
    let c = boot(p.clone());
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let low = CtMeta::new(&p, 0, p.scale()).unwrap();
    for (input, target, index) in [(low, h, 1), (h, low, 1), (h, h, 2)] {
        let next = PackedBatchMeta::new(
            p.clone(),
            index,
            b.input_spec(),
            b.target_spec(),
            vec![PackedRecordMeta::new(2, input, target)],
        )
        .unwrap();
        for placement in [
            PackedNiPlacement::Au,
            PackedNiPlacement::Wn,
            PackedNiPlacement::Aa,
        ] {
            assert!(
                PackedNiPlan::new(
                    &e,
                    &s,
                    &b,
                    update(0.04, GradRelin::Lazy),
                    PackedNextStep::Known(next.clone(), update(0.01, GradRelin::Lazy)),
                    placement,
                    &c
                )
                .is_err()
            );
        }
    }
    let (p, e, s, b) = setup(true, 15);
    let c = boot(p);
    // Exactly the graph fits, but no state projection reserve remains.
    assert_eq!(
        plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), None)
            .unwrap()
            .step()
            .minimum_state_level,
        0
    );
    assert!(
        PackedNiPlan::new(
            &e,
            &s,
            &b,
            update(0.04, GradRelin::Lazy),
            PackedNextStep::Unknown,
            PackedNiPlacement::Wn,
            &c
        )
        .is_err()
    );
}
#[test]
fn first_equal_and_later_u_scalar_costs_are_literal_for_each_relin_policy() {
    let (p, e, s, b) = setup(false, 16);
    let c = boot(p.clone());
    for relin in [GradRelin::Lazy, GradRelin::PerSample] {
        let first = plan_stage_dag(&e, &s, &b, update(0.08, relin), None).unwrap();
        assert_eq!(
            first.step().predicted_counts,
            OpCounter {
                rotations: 36,
                relins: if relin == GradRelin::Lazy { 12 } else { 14 },
                ct_ct_tensor: 14,
                ct_pt_mults: 16,
                rescales: 28,
                sse_switches: 0,
                bootstraps: 0
            }
        );
        let plan = PackedNiPlan::new(
            &e,
            &s,
            &b,
            update(0.04, relin),
            PackedNextStep::Terminal,
            PackedNiPlacement::Au,
            &c,
        )
        .unwrap();
        let h = CtMeta::new(&p, 16, p.scale()).unwrap();
        let batch = PackedBatchMeta::new(
            p.clone(),
            1,
            b.input_spec(),
            b.target_spec(),
            vec![
                PackedRecordMeta::new(0, h, h),
                PackedRecordMeta::new(2, h, h),
            ],
        )
        .unwrap();
        let later =
            plan_stage_dag(&e, plan.committed(), &batch, update(0.01, relin), None).unwrap();
        assert_eq!(
            later.step().predicted_counts,
            OpCounter {
                rotations: 36,
                relins: if relin == GradRelin::Lazy { 12 } else { 14 },
                ct_ct_tensor: 14,
                ct_pt_mults: 22,
                rescales: 34,
                sse_switches: 0,
                bootstraps: 0
            }
        );
    }
}
#[test]
fn independent_n16_first_valid_record_full_trace_uses_each_actual_prime() {
    for square in [false, true] {
        let (p, e, s, b) = setup(square, 16);
        let plan = plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), None).unwrap();
        let d = p.scale();
        let q = |j: usize| p.q()[j] as f64;
        let mut expected = vec![
            ("affine1.product", 15, d * d / q(16)),
            ("affine1", 14, d),
            ("z1", 14, d),
            ("hidden.natural", 13, d * d / q(14)),
            ("hidden", 12, d),
            ("affine2.product", 11, d * d / q(12)),
            ("affine2", 10, d),
            ("z2", 10, d),
        ];
        if square {
            expected.extend([
                ("output.natural", 9, d * d / q(10)),
                ("output", 8, d),
                ("residual", 8, d),
                ("loss", 7, d),
                ("derivative2", 9, d),
                ("dz2", 6, d * d / q(7)),
                ("upstream.product", 5, (d * d / q(7)) * d / q(6)),
                ("upstream", 4, d),
                ("derivative1", 13, d),
                ("dz1", 3, d * d / q(4)),
            ]);
        } else {
            expected.extend([
                ("output", 10, d),
                ("residual", 10, d),
                ("loss", 9, d),
                ("dz2", 9, d),
                ("upstream.product", 8, d * d / q(9)),
                ("upstream", 7, d),
                ("derivative1", 13, d),
                ("dz1", 6, d * d / q(7)),
            ]);
        }
        let actual: Vec<_> = plan
            .step()
            .trace
            .iter()
            .filter(|t| t.sample == Some(0))
            .map(|t| (t.name.as_str(), t.level, t.scale.to_bits()))
            .collect();
        assert_eq!(
            actual,
            expected
                .into_iter()
                .map(|(n, l, s)| (n, l, s.to_bits()))
                .collect::<Vec<_>>()
        );
        let shared = if square {
            vec![
                ("outer0", 2, d * (d * d / q(4)) / q(3)),
                ("gradient0.0", 1, d),
                ("weight0", 1, d),
                ("bias_gradient0", 2, d),
                ("bias0", 2, d),
                ("outer1", 5, d * (d * d / q(7)) / q(6)),
                ("gradient1.0", 4, d),
                ("weight1", 4, d),
                ("bias_gradient1", 5, d),
                ("bias1", 5, d),
            ]
        } else {
            vec![
                ("outer0", 5, d * (d * d / q(7)) / q(6)),
                ("gradient0.0", 4, d),
                ("gradient0.1", 4, d),
                ("weight0", 4, d),
                ("velocity0", 4, d),
                ("outer1", 8, d * d / q(9)),
                ("gradient1.0", 7, d),
                ("gradient1.1", 7, d),
                ("weight1", 7, d),
                ("velocity1", 7, d),
            ]
        };
        let actual: Vec<_> = plan
            .step()
            .trace
            .iter()
            .filter(|t| t.sample.is_none())
            .map(|t| (t.name.as_str(), t.level, t.scale.to_bits()))
            .collect();
        assert_eq!(
            actual,
            shared
                .into_iter()
                .map(|(n, l, s)| (n, l, s.to_bits()))
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn hypothetical_first_weights_and_later_velocity_levels_are_not_fabricated() {
    let (p, e, s, b) = setup(false, 16);
    let g = fixture::geometry(false);
    let hi = CtMeta::new(&p, 16, p.scale()).unwrap();
    let lo = CtMeta::new(&p, 1, p.scale()).unwrap();
    let later = PackedStageMeta::new(
        p.clone(),
        &g,
        0,
        fixture::config(false),
        [hi; 2],
        [Some(lo), Some(hi)],
        None,
        [1; 2],
    )
    .unwrap();
    let batch = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        b.records().to_vec(),
    )
    .unwrap();
    assert!(plan_stage_dag(&e, &later, &batch, update(0.01, GradRelin::Lazy), None).is_err());
    let changed = PackedStageMeta::new(
        p.clone(),
        &g,
        0,
        fixture::config(false),
        [CtMeta::new(&p, 15, p.scale()).unwrap(), hi],
        [None; 2],
        None,
        [0; 2],
    )
    .unwrap();
    assert_eq!(
        plan_stage_dag(&e, &changed, &b, update(0.04, GradRelin::Lazy), None)
            .unwrap()
            .candidate()
            .weights()
            .map(|x| x.level()),
        [3, 6]
    );
    assert_eq!(
        plan_stage_dag(&e, &s, &b, update(0.04, GradRelin::Lazy), None)
            .unwrap()
            .candidate()
            .weights()
            .map(|x| x.level()),
        [4, 7]
    );
}
#[test]
fn future_old_u_scalar_underflow_rejects_an_otherwise_depth_fitting_aa_step() {
    let (p, e, _, b) = setup(false, 16);
    let c = boot(p.clone());
    let g = fixture::geometry(false);
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let state = PackedStageMeta::new(
        p.clone(),
        &g,
        0,
        PackedStageConfig::poly_relu2_nag(1e-7, 0.08).unwrap(),
        [h; 2],
        [None; 2],
        None,
        [0; 2],
    )
    .unwrap();
    let first = plan_stage_dag(&e, &state, &b, update(0.08, GradRelin::Lazy), Some(&c)).unwrap();
    assert_eq!(first.candidate().weights().map(|h| h.level()), [8, 11]);
    let batch = PackedBatchMeta::new(
        p.clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        vec![PackedRecordMeta::new(2, h, h)],
    )
    .unwrap();
    let failure = plan_stage_dag(
        &e,
        first.candidate(),
        &batch,
        update(0.08, GradRelin::Lazy),
        Some(&c),
    )
    .unwrap_err();
    assert!(failure.to_string().contains("U scalar"));
    assert!(
        PackedNiPlan::new(
            &e,
            &state,
            &b,
            update(0.08, GradRelin::Lazy),
            PackedNextStep::Known(batch, update(0.08, GradRelin::Lazy)),
            PackedNiPlacement::Aa,
            &c
        )
        .is_err()
    );
}

// Review F1: independent foreign parameter constructors, without keys or ciphertexts.
fn foreign_batch_rejected(scale_bits: u32, extra_level: bool) {
    let (p, enc, state, valid) = setup(false, 16);
    let (mut q, aux) = ckks_train_ckks::params::n16_boot_bits();
    if extra_level {
        q.extend([42; 8]);
    }
    let foreign = Arc::new(CkksParams::new(10, &q, &aux, scale_bits, 8).unwrap());
    let level = if extra_level { foreign.max_level() } else { 16 };
    if extra_level {
        assert!(level > p.max_level());
    }
    let h = CtMeta::new(&foreign, level, foreign.scale()).unwrap();
    let batch = PackedBatchMeta::new(
        foreign.clone(),
        0,
        valid.input_spec(),
        valid.target_spec(),
        valid
            .records()
            .iter()
            .map(|r| PackedRecordMeta::new(r.id(), h, h))
            .collect(),
    )
    .unwrap();
    assert_eq!(valid, valid.clone());
    assert_ne!(batch, valid);
    assert!(
        plan_stage_dag(&enc, &state, &batch, update(0.04, GradRelin::Lazy), None).is_err(),
        "foreign batch must fail at the consuming DAG boundary"
    );
}
#[test]
fn fix_round1_pure_foreign_delta_rejected() {
    foreign_batch_rejected(40, false);
}
#[test]
fn fix_round1_pure_foreign_long_chain_rejected() {
    foreign_batch_rejected(42, true);
}
#[test]
fn fix_round1_pure_same_shape_foreign_context_rejected() {
    foreign_batch_rejected(42, false);
}
