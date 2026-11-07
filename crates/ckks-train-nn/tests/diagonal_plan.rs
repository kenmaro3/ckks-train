//! Pure P2 contracts. No keys, Encoder or ciphertexts are constructed here.
use ckks_train_ckks::{CkksParams, EvalConfig};
use ckks_train_nn::diagonal_plan::*;
use ckks_train_nn::{dense::TransposeKeys, layout::Layout, trainer_ni::BootPlacement};

fn params() -> CkksParams {
    CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap()
}
fn geometry(width: usize, poly: bool) -> StageGeometry {
    plan_stage_geometry(
        &[2, 4, width],
        &Layout::new(512, 16, 8).unwrap(),
        &[
            if poly {
                DiagonalActivation::PolyRelu2
            } else {
                DiagonalActivation::Square
            },
            if poly {
                DiagonalActivation::Identity
            } else {
                DiagonalActivation::Square
            },
        ],
        &TransposeKeys::Dedicated,
        DiagonalMode::Interactive,
        EvalConfig::OPTIMIZED,
    )
    .unwrap()
}
#[test]
fn p2_geometry_is_checked_and_keeps_logical_class_width() {
    let g = geometry(3, false);
    assert_eq!(g.dims(), &[2, 4, 3]);
    assert_eq!((g.shapes()[1].d_out, g.shapes()[1].n_out), (3, 4));
    assert_eq!(geometry(4, true).dims(), &[2, 4, 4]);
    assert!(
        plan_stage_geometry(
            &[0, 4],
            &Layout::new(512, 16, 8).unwrap(),
            &[DiagonalActivation::Square],
            &TransposeKeys::Dedicated,
            DiagonalMode::Interactive,
            EvalConfig::NAIVE
        )
        .is_err()
    );
}
#[test]
fn p2_initial_labels_use_actual_primes_and_preserve_bias_products() {
    let p = params();
    let g = geometry(3, false);
    let s = initial_stage_metadata(
        &g,
        &p,
        None,
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Active,
    )
    .unwrap();
    let d = p.scale();
    let z = d * d / p.q()[9] as f64;
    let a = z * z / p.q()[8] as f64;
    assert_eq!(s.weights()[0].scale_bits(), d.to_bits());
    assert_eq!(s.biases()[0].scale_bits(), (d * d).to_bits());
    assert_eq!(s.biases()[1].scale_bits(), (a * d).to_bits());
    assert_eq!(s.batch().x.level(), 9);
    assert!(s.velocity().iter().all(Option::is_none));
}
#[test]
fn p2_interactive_schedule_carries_reference_velocity_and_seven_counts() {
    let p = params();
    let g = geometry(4, true);
    let mut s = initial_stage_metadata(
        &g,
        &p,
        None,
        DiagonalOptimizer::Nag {
            lr: 0.04,
            reference: 0.08,
            momentum: 0.9,
        },
        BiasPolicy::Frozen,
    )
    .unwrap();
    let mut counts = Vec::new();
    for lr in [0.04, 0.08, 0.02, 0.08] {
        let u = UpdateSpec::new(
            vec![
                MicroMetadata {
                    n_valid: 8,
                    x: s.batch().x,
                    y: s.batch().y,
                },
                MicroMetadata {
                    n_valid: 4,
                    x: s.batch().x,
                    y: s.batch().y,
                },
            ],
            12,
            lr,
            None,
        );
        let r = plan_stage_update(&g, &p, None, &s, &u).unwrap();
        assert_eq!(r.reference_branch(), lr != 0.08);
        assert_eq!(r.client_refreshes(), 14);
        assert_eq!(r.counts_nonbootstrap().bootstraps, 0);
        assert_eq!(r.counts_nonbootstrap().sse_switches, 0);
        assert!(r.post_state().velocity().iter().all(Option::is_some));
        counts.push(r.counts_nonbootstrap().clone());
        s = r.post_state().clone();
    }
    assert_eq!(counts[0].ct_ct_tensor, counts[1].ct_ct_tensor);
    assert_eq!(counts[0].ct_pt_mults, counts[1].ct_pt_mults);
    assert!(counts[2].ct_pt_mults > counts[3].ct_pt_mults);
}
#[test]
fn p2_invalid_partition_and_identity_fail_before_transition() {
    let p = params();
    let g = geometry(3, false);
    let s = initial_stage_metadata(
        &g,
        &p,
        None,
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Active,
    )
    .unwrap();
    for (ns, total) in [(vec![], 0), (vec![0], 0), (vec![9], 9), (vec![8, 4], 11)] {
        let u = UpdateSpec::new(
            ns.into_iter()
                .map(|n| MicroMetadata {
                    n_valid: n,
                    x: s.batch().x,
                    y: s.batch().y,
                })
                .collect(),
            total,
            0.08,
            None,
        );
        assert!(matches!(
            plan_stage_update(&g, &p, None, &s, &u).unwrap_err().kind,
            PlanFailure::InvalidMicroPartition
        ));
    }
    let other = geometry(4, true);
    let u = UpdateSpec::new(
        vec![MicroMetadata {
            n_valid: 8,
            x: s.batch().x,
            y: s.batch().y,
        }],
        8,
        0.08,
        None,
    );
    assert!(matches!(
        plan_stage_update(&other, &p, None, &s, &u)
            .unwrap_err()
            .kind,
        PlanFailure::IdentityMismatch
    ));
}
#[test]
fn p2_all_configs_and_reuse_chains_are_explicit() {
    let l = Layout::new(512, 16, 8).unwrap();
    for cfg in EvalConfig::all() {
        let g = plan_stage_geometry(
            &[2, 4, 10],
            &l,
            &[DiagonalActivation::Square; 2],
            &TransposeKeys::Dedicated,
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
            cfg,
        )
        .unwrap();
        assert_eq!(g.eval_config(), cfg);
        assert!(g.compact().is_some());
    }
    assert!(
        plan_stage_geometry(
            &[2, 4, 10],
            &l,
            &[DiagonalActivation::Square; 2],
            &TransposeKeys::Reuse(vec![]),
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED
        )
        .is_err()
    );
}

fn ni() -> &'static (
    CkksParams,
    ckks_train_ckks::bootstrap::metadata::BootstrapMetadata,
) {
    use ckks_train_ckks::bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, DftSupportProof},
    };
    use std::sync::OnceLock;
    static N: OnceLock<(CkksParams, BootstrapMetadata)> = OnceLock::new();
    N.get_or_init(|| {
        let (q, sp) = ckks_train_ckks::params::n16_boot_bits();
        let p = CkksParams::new(10, &q, &sp, 42, 8).unwrap();
        let proof = DftSupportProof::verify_current().unwrap();
        let b = BootstrapMetadata::new_with_verified_support(
            &p,
            &BootParams {
                log_slots: 9,
                ..BootParams::default_n16()
            },
            &proof,
        )
        .unwrap();
        (p, b)
    })
}
#[test]
fn p2_ni_all_classes_placements_and_carried_schedule() {
    let (p, b) = ni();
    let l = Layout::new(512, 16, 8).unwrap();
    for placement in [
        BootPlacement::WhenNeeded,
        BootPlacement::AfterUpdate,
        BootPlacement::AfterActivation,
    ] {
        for (width, poly) in [
            (3, false),
            (2, false),
            (10, false),
            (4, true),
            (2, true),
            (16, true),
        ] {
            let acts = if poly {
                [DiagonalActivation::PolyRelu2, DiagonalActivation::Identity]
            } else {
                [DiagonalActivation::Square; 2]
            };
            let g = plan_stage_geometry(
                &[2, 4, width],
                &l,
                &acts,
                &TransposeKeys::Dedicated,
                DiagonalMode::NonInteractive(placement),
                EvalConfig::OPTIMIZED,
            )
            .unwrap();
            let mut state = initial_stage_metadata(
                &g,
                p,
                Some(b),
                DiagonalOptimizer::Nag {
                    lr: 0.04,
                    reference: 0.08,
                    momentum: 0.9,
                },
                BiasPolicy::Frozen,
            )
            .unwrap();
            for (step, rate) in [0.04, 0.08, 0.02, 0.08].into_iter().enumerate() {
                let parts = match step {
                    0 | 3 => vec![8, 4],
                    1 => vec![8],
                    _ => vec![5],
                };
                let spec = UpdateSpec::new(
                    parts
                        .iter()
                        .map(|&n| MicroMetadata {
                            n_valid: n,
                            x: state.batch().x,
                            y: state.batch().y,
                        })
                        .collect(),
                    parts.iter().sum(),
                    rate,
                    match step {
                        0 => Some(true),
                        1 | 2 => Some(false),
                        _ => None,
                    },
                );
                let r = plan_stage_update(&g, p, Some(b), &state, &spec).unwrap_or_else(|e| {
                    panic!("width{width} poly{poly} {placement:?} step{step}: {e:?}")
                });
                assert_eq!(r.first_nag_update(), step == 0);
                assert_eq!(r.reference_branch(), rate != 0.08);
                assert_eq!(
                    r.bootstrap_events().len() as u64,
                    r.counts_including_bootstrap().bootstraps
                );
                assert_eq!(r.post_state().completed_updates(), step + 1);
                assert!(r.trace().iter().any(|e| e.role == "bias_extracted"));
                state = r.post_state().clone();
            }
        }
    }
}
#[test]
fn p2_ni_predictor_and_counter_projection_all_eight_configs() {
    use ckks_train_nn::trainer_ni::{ActCost, predict_update_with, ref_shift_rotations};
    let (p, b) = ni();
    let l = Layout::new(512, 16, 8).unwrap();
    let dims = [2, 4, 4];
    for cfg in EvalConfig::all() {
        let g = plan_stage_geometry(
            &dims,
            &l,
            &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
            cfg,
        )
        .unwrap();
        let s = initial_stage_metadata(
            &g,
            p,
            Some(b),
            DiagonalOptimizer::Nag {
                lr: 0.04,
                reference: 0.08,
                momentum: 0.9,
            },
            BiasPolicy::Frozen,
        )
        .unwrap();
        let u = UpdateSpec::new(
            vec![
                MicroMetadata {
                    n_valid: 8,
                    x: s.batch().x,
                    y: s.batch().y,
                },
                MicroMetadata {
                    n_valid: 4,
                    x: s.batch().x,
                    y: s.batch().y,
                },
            ],
            12,
            0.04,
            None,
        );
        let r = plan_stage_update(&g, p, Some(b), &s, &u).unwrap();
        let old = predict_update_with(
            &dims,
            &[ActCost::QUADRATIC, ActCost::IDENTITY],
            &l,
            &TransposeKeys::Dedicated,
            BootPlacement::AfterUpdate,
            2,
            true,
            true,
            cfg,
        )
        .unwrap();
        assert_eq!(
            r.counts_nonbootstrap().rotations,
            old.rotations() + ref_shift_rotations(&dims, &l).unwrap()
        );
        assert_eq!(r.counts_nonbootstrap().relins, old.relins());
        assert_eq!(
            r.counts_nonbootstrap().ct_ct_tensor,
            2 * (2 + 4) + 2 + 2 * 4 + 2 + 2 * (2 + 4)
        ); // dense, square, backward, dz, gradient tensors
        assert_eq!(
            r.counts_nonbootstrap().ct_pt_mults,
            2 + 2 + 2 * (2 + 4) + 2 + 6 + 2 + 1
        ); // partial Identity mask
    }
}
#[test]
fn p2_all21_registered_geometry_and_params() {
    let mut profiles = vec![];
    for (input, classes, padded, lanes) in [
        (4, 3, 4, 32),
        (30, 2, 2, 16),
        (196, 10, 16, 64),
        (196, 10, 16, 64),
    ] {
        for model in 0..4 {
            profiles.push((input, classes, padded, lanes, model));
        }
    }
    for model in 0..4 {
        profiles.push((196, 10, 16, 64, model));
    }
    profiles.push((196, 10, 16, 128, 1));
    assert_eq!(profiles.len(), 21);
    use ckks_train_ckks::bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, DftSupportProof},
    };
    let p = CkksParams::n16_boot();
    let interactive = CkksParams::n16_train();
    let proof = DftSupportProof::verify_current().unwrap();
    let boot = BootstrapMetadata::new_with_verified_support(&p, &BootParams::default_n16(), &proof)
        .unwrap();
    for (input, classes, padded, lanes, model) in profiles {
        let stages = match model {
            0 => vec![vec![input, 10, classes]],
            1 => vec![vec![input, 32, padded]],
            2 => vec![vec![input, 64, padded], vec![64, 32, padded]],
            _ => vec![
                vec![input, 128, padded],
                vec![128, 64, padded],
                vec![64, 32, padded],
            ],
        };
        let d = stages
            .iter()
            .flatten()
            .map(|n| usize::next_power_of_two(*n))
            .max()
            .unwrap();
        let l = Layout::new(p.slots(), d, lanes).unwrap();
        for dims in &stages {
            for mode in [
                DiagonalMode::Interactive,
                DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
                DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
                DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
            ] {
                let acts = if model == 0 {
                    [DiagonalActivation::Square; 2]
                } else {
                    [DiagonalActivation::PolyRelu2, DiagonalActivation::Identity]
                };
                let g = plan_stage_geometry(
                    dims,
                    &l,
                    &acts,
                    &TransposeKeys::Dedicated,
                    mode,
                    EvalConfig::OPTIMIZED,
                )
                .unwrap();
                assert_eq!(g.dims(), dims);
                let (parameters, descriptor) = if mode == DiagonalMode::Interactive {
                    (&interactive, None)
                } else {
                    (&p, Some(&boot))
                };
                let state = initial_stage_metadata(
                    &g,
                    parameters,
                    descriptor,
                    DiagonalOptimizer::Sgd { lr: 0.08 },
                    if model == 0 {
                        BiasPolicy::Active
                    } else {
                        BiasPolicy::Frozen
                    },
                )
                .unwrap();
                assert_eq!(
                    state.batch().x.level(),
                    if mode == DiagonalMode::Interactive {
                        31
                    } else {
                        15
                    }
                );
                assert_eq!(state.training_keys().parameters.q(), parameters.q());
                assert_eq!(state.training_keys().rotation_steps, g.rotation_steps());
                assert!(state.batch().y.scale().is_finite());
                if let Some(cl) = g.compact() {
                    assert_eq!(cl.containers.len(), 2);
                    assert!(cl.containers.iter().all(|c| c.layer.is_some()));
                }
            }
        }
    }
}

#[test]
fn p2_first_reference_and_heterogeneous_inputs_are_distinct_branches() {
    let (p, b) = ni();
    let l = Layout::new(512, 16, 8).unwrap();
    for mode in [
        DiagonalMode::Interactive,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
        DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
    ] {
        let g = plan_stage_geometry(
            &[2, 4, 4],
            &l,
            &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            mode,
            EvalConfig::OPTIMIZED,
        )
        .unwrap();
        let ip = params();
        let (param, boot) = if mode == DiagonalMode::Interactive {
            (&ip, None)
        } else {
            (p, Some(b))
        };
        let state = initial_stage_metadata(
            &g,
            param,
            boot,
            DiagonalOptimizer::Nag {
                lr: 0.08,
                reference: 0.08,
                momentum: 0.9,
            },
            BiasPolicy::Active,
        )
        .unwrap();
        let u = UpdateSpec::new(
            vec![MicroMetadata {
                n_valid: 8,
                x: state.batch().x,
                y: state.batch().y,
            }],
            8,
            0.08,
            Some(true),
        );
        let first = plan_stage_update(&g, param, boot, &state, &u).unwrap();
        assert!(first.first_nag_update());
        assert!(!first.reference_branch());
        assert!(
            !first
                .trace()
                .iter()
                .any(|e| e.role == "mu_velocity" || e.role.ends_with("_mu_velocity"))
        );
        let second = plan_stage_update(&g, param, boot, first.post_state(), &u).unwrap();
        assert!(!second.first_nag_update());
        assert!(!second.reference_branch());
        assert!(
            second
                .trace()
                .iter()
                .any(|e| e.role == "mu_velocity" || e.role.ends_with("_mu_velocity"))
        );
    }
    let g = plan_stage_geometry(
        &[2, 4, 4],
        &l,
        &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
        &TransposeKeys::Dedicated,
        DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let state = initial_stage_metadata(
        &g,
        p,
        Some(b),
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Frozen,
    )
    .unwrap();
    let low = ckks_train_ckks::metadata::CtMeta::new(p, 14, p.scale()).unwrap();
    let u = UpdateSpec::new(
        vec![
            MicroMetadata {
                n_valid: 8,
                x: state.batch().x,
                y: state.batch().y,
            },
            MicroMetadata {
                n_valid: 4,
                x: low,
                y: state.batch().y,
            },
        ],
        12,
        0.08,
        None,
    );
    let error = plan_stage_update(&g, p, Some(b), &state, &u).unwrap_err();
    assert_eq!(error.kind, PlanFailure::ScaleInadmissible);
    assert_eq!(error.operation, "dense bias exact product");
    assert_eq!(
        error.source_scale_bits,
        Some(state.biases()[1].scale_bits())
    );
    assert!(error.target_scale_bits.is_some());
    assert_eq!(error.prime, Some(p.q()[12]));
    assert_eq!(
        (error.micro, error.layer, error.update),
        (Some(1), Some(1), Some(0))
    );
    assert!(!error.trace.is_empty());
}
#[test]
fn p2_arithmetic_parameter_and_rate_rejections_are_typed() {
    let l = Layout::new(512, 16, 8).unwrap();
    assert_eq!(
        plan_stage_geometry(
            &[usize::MAX, 2],
            &l,
            &[DiagonalActivation::Identity],
            &TransposeKeys::Dedicated,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED
        )
        .unwrap_err()
        .kind,
        PlanFailure::ArithmeticOverflow
    );
    let p = params();
    let g = geometry(4, true);
    let s = initial_stage_metadata(
        &g,
        &p,
        None,
        DiagonalOptimizer::Nag {
            lr: 0.04,
            reference: 0.08,
            momentum: 0.9,
        },
        BiasPolicy::Frozen,
    )
    .unwrap();
    for rate in [0., -1., f64::NAN, f64::INFINITY, 0.09] {
        let u = UpdateSpec::new(
            vec![MicroMetadata {
                n_valid: 8,
                x: s.batch().x,
                y: s.batch().y,
            }],
            8,
            rate,
            None,
        );
        assert_eq!(
            plan_stage_update(&g, &p, None, &s, &u).unwrap_err().kind,
            PlanFailure::RateConflict
        );
    }
    let changed =
        CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 41, 10).unwrap();
    let u = UpdateSpec::new(
        vec![MicroMetadata {
            n_valid: 8,
            x: s.batch().x,
            y: s.batch().y,
        }],
        8,
        0.04,
        None,
    );
    assert_eq!(
        plan_stage_update(&g, &changed, None, &s, &u)
            .unwrap_err()
            .kind,
        PlanFailure::IdentityMismatch
    );
}

#[test]
fn p2_bias_only_container_never_acquires_velocity() {
    let (p, b) = ni();
    let l = Layout::new(512, 32, 16).unwrap();
    let g = plan_stage_geometry(
        &[32, 16],
        &l,
        &[DiagonalActivation::Identity],
        &TransposeKeys::Dedicated,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let state = initial_stage_metadata(
        &g,
        p,
        Some(b),
        DiagonalOptimizer::Nag {
            lr: 0.04,
            reference: 0.08,
            momentum: 0.9,
        },
        BiasPolicy::Active,
    )
    .unwrap();
    assert_eq!(g.compact().unwrap().containers.len(), 2);
    assert!(g.compact().unwrap().containers[1].layer.is_none());
    let u = UpdateSpec::new(
        vec![MicroMetadata {
            n_valid: 16,
            x: state.batch().x,
            y: state.batch().y,
        }],
        16,
        0.04,
        None,
    );
    let r = plan_stage_update(&g, p, Some(b), &state, &u).unwrap();
    assert!(r.post_state().velocity()[0].is_some());
    assert!(r.post_state().velocity()[1].is_none());
    assert_eq!(r.bootstrap_events().len(), 3);
}

#[test]
fn p2_one_valid_tail_keeps_denominator_order_and_partial_identity_work() {
    let (np, boot) = ni();
    let ip = params();
    for poly in [false, true] {
        for mode in [
            DiagonalMode::Interactive,
            DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
            DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
        ] {
            let (p, b) = if mode == DiagonalMode::Interactive {
                (&ip, None)
            } else {
                (np, Some(boot))
            };
            let acts = if poly {
                [DiagonalActivation::PolyRelu2, DiagonalActivation::Identity]
            } else {
                [DiagonalActivation::Square; 2]
            };
            let g = plan_stage_geometry(
                &[2, 4, if poly { 4 } else { 3 }],
                &Layout::new(512, 16, 8).unwrap(),
                &acts,
                &TransposeKeys::Dedicated,
                mode,
                EvalConfig::OPTIMIZED,
            )
            .unwrap();
            for nag in [false, true] {
                let rate = if nag { 0.04 } else { 0.08 };
                let opt = if nag {
                    DiagonalOptimizer::Nag {
                        lr: rate,
                        reference: 0.08,
                        momentum: 0.9,
                    }
                } else {
                    DiagonalOptimizer::Sgd { lr: rate }
                };
                let s = initial_stage_metadata(&g, p, b, opt, BiasPolicy::Active).unwrap();
                for parts in [vec![8, 1], vec![1]] {
                    let total: usize = parts.iter().sum();
                    let u = UpdateSpec::new(
                        parts
                            .iter()
                            .map(|&n| MicroMetadata {
                                n_valid: n,
                                x: s.batch().x,
                                y: s.batch().y,
                            })
                            .collect(),
                        total,
                        rate,
                        None,
                    );
                    let r = plan_stage_update(&g, p, b, &s, &u).unwrap();
                    assert_eq!(
                        r.coefficients().current_bits,
                        (if nag {
                            (rate * (1. + 0.9)) / total as f64
                        } else {
                            rate / total as f64
                        })
                        .to_bits()
                    );
                    assert_eq!(r.coefficients().bias_bits, (rate / total as f64).to_bits());
                    assert_eq!(
                        r.coefficients().reference_bits,
                        nag.then_some(((0.08_f64 * (1. + 0.9)) / total as f64).to_bits())
                    );
                    assert_eq!(
                        r.trace()
                            .iter()
                            .filter(|e| e.role == "entry_input")
                            .map(|e| e.micro)
                            .collect::<Vec<_>>(),
                        (0..parts.len()).map(Some).collect::<Vec<_>>()
                    );
                    for (mb, &n) in parts.iter().enumerate() {
                        let dz = r
                            .trace()
                            .iter()
                            .find(|e| e.role == "dz" && e.micro == Some(mb) && e.layer == Some(1))
                            .unwrap();
                        let loss = r
                            .trace()
                            .iter()
                            .find(|e| e.role == "loss_delta" && e.micro == Some(mb))
                            .unwrap();
                        if poly {
                            assert_eq!(
                                dz.metadata.level(),
                                loss.metadata.level() - usize::from(n != 8)
                            );
                            assert_eq!(dz.counts.ct_pt_mults, u64::from(n != 8));
                            assert_eq!(dz.counts.rescales, u64::from(n != 8));
                        }
                        let entry = r
                            .trace()
                            .iter()
                            .find(|e| e.role == "entry_input" && e.micro == Some(mb))
                            .unwrap();
                        assert_eq!(entry.metadata, s.batch().x);
                        assert_eq!(entry.multiplicity, 1);
                    }
                }
            }
        }
    }
}

#[test]
fn p2_raw_entries_remain_distinct_when_alignment_merges_labels() {
    let (p, boot) = ni();
    let g = plan_stage_geometry(
        &[2, 4, 4],
        &Layout::new(512, 16, 8).unwrap(),
        &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
        &TransposeKeys::Dedicated,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let s = initial_stage_metadata(
        &g,
        p,
        Some(boot),
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Frozen,
    )
    .unwrap();
    let mut entries = vec![];
    let mut aligned = vec![];
    for level in [15, 16] {
        let u = UpdateSpec::new(
            vec![MicroMetadata {
                n_valid: 8,
                x: ckks_train_ckks::metadata::CtMeta::new(p, level, s.batch().x.scale()).unwrap(),
                y: ckks_train_ckks::metadata::CtMeta::new(p, level, p.scale()).unwrap(),
            }],
            8,
            0.08,
            None,
        );
        let r = plan_stage_update(&g, p, Some(boot), &s, &u).unwrap();
        let at = |role: &str| {
            r.trace()
                .iter()
                .find(|e| e.role == role && e.micro == Some(0))
                .unwrap()
                .metadata
        };
        entries.push((at("entry_input"), at("entry_target")));
        aligned.push((at("input"), at("aligned_target")));
    }
    assert_ne!(entries[0], entries[1]);
    assert_eq!(aligned[0], aligned[1]);
    assert_eq!((entries[0].0.level(), entries[1].0.level()), (15, 16));
    assert_eq!((entries[0].1.level(), entries[1].1.level()), (15, 16));
}
