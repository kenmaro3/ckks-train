use ckks_train_ckks::CkksParams;
use ckks_train_ckks::bootstrap::metadata::DftSupportProof;
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use ckks_train_ckks::bootstrap::metadata::{BootstrapMetadata, MetadataLimits};

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn default_n16_structural_receipt_without_he() {
    let proof = DftSupportProof::verify_current().unwrap();
    assert_eq!(proof.root_count(), 65_535);
    let p = CkksParams::n16_boot();
    let bp = BootParams::default_n16();
    let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
    assert_eq!(
        m.evalmod().coefficient_bits(),
        ckks_train_ckks::bootstrap::evalmod_coeffs(&bp)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let c = m.cost_per_boot();
    assert_eq!(
        (
            c.rotations,
            c.relins,
            c.ct_ct_tensor,
            c.ct_pt_mults,
            c.rescales,
            c.sse_switches,
            c.bootstraps
        ),
        (69, 40, 46, 343, 56, 2, 1)
    );
    assert_eq!(m.key_switches_per_boot(), 111);
    assert_eq!(m.output().level(), 16);
    assert_eq!(m.output().scale_bits(), p.scale().to_bits());
    assert_eq!(m.generated_keys().rotation_steps().len(), 32);
    assert!(m.generated_keys().conjugation().is_some());
    let shapes: Vec<_> = m
        .stoc_stages()
        .iter()
        .map(|s| (s.d(), s.stride(), s.offset()))
        .collect();
    assert_eq!(
        shapes,
        [(16, 1, -8), (32, 8, -16), (32, 128, -16), (16, 2048, 0)]
    );
    let again = BootstrapMetadata::with_limits_and_verified_support(
        &p,
        &bp,
        MetadataLimits::default(),
        &proof,
    )
    .unwrap();
    assert_eq!(
        m.descriptor_identity_bytes(),
        again.descriptor_identity_bytes()
    );
}

use ckks_train_ckks::Encoder;
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use ckks_train_ckks::KeyGenerator;
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use ckks_train_ckks::bootstrap::metadata::{
    BootstrapMetadataError, MetadataResource, UnvalidatedRequirement,
};
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper, Direction, dft_stages};
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use ckks_train_core::sampler::Rng;
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
use std::collections::BTreeSet;
use std::sync::Arc;

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
fn small(
    log_slots: u32,
    ctos: usize,
    stoc: usize,
    degree: usize,
    r: usize,
    sse: bool,
) -> (CkksParams, BootParams) {
    let em = (degree + 1).next_power_of_two().trailing_zeros() as usize + r;
    let mut q = vec![54, 42];
    q.extend(vec![34; stoc]);
    q.extend(vec![58; em]);
    q.extend(vec![56; ctos]);
    let p = CkksParams::new(log_slots + 1, &q, &[60], 42, q.len()).unwrap();
    let bp = BootParams {
        log_slots,
        ctos_levels: ctos,
        evalmod_levels: em,
        stoc_levels: stoc,
        k_intervals: 32,
        cheb_degree: degree,
        double_angle: r,
        sparse_hw: None,
        ephemeral_hw: if sse { Some(1) } else { None },
    };
    (p, bp)
}

/// Independent connectivity oracle: visit row-to-column bit choices explicitly. Also
/// independently form stored offsets by Minkowski addition of each butterfly's keys.
#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn every_small_group_matches_connectivity_minkowski_and_actual_dft() {
    let proof = DftSupportProof::verify_current().unwrap();
    for log in 1..=6 {
        let n = 1usize << log;
        for levels in 1..=log as usize {
            let (p, bp) = small(log, levels, levels, 3, 0, false);
            let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
            for (direction, stages) in [
                (Direction::CoeffToSlot, m.ctos_stages()),
                (Direction::SlotToCoeff, m.stoc_stages()),
            ] {
                let actual = dft_stages(log, levels, direction == Direction::CoeffToSlot).unwrap();
                for (stage, matrix) in stages.iter().zip(actual) {
                    let mut stored = BTreeSet::from([0]);
                    for bit in stage.low_bit()..=stage.high_bit() {
                        let h = 1usize << bit;
                        stored = stored
                            .iter()
                            .flat_map(|&o| [o, (o + h) % n, (o + n - h) % n])
                            .collect();
                    }
                    assert_eq!(
                        stage.stored_support().offsets().collect::<BTreeSet<_>>(),
                        stored
                    );
                    assert_eq!(
                        matrix.offsets().into_iter().collect::<BTreeSet<_>>(),
                        stored
                    );
                    let mut reachable = vec![vec![false; n]; n];
                    for (row, cells) in reachable.iter_mut().enumerate() {
                        let mut positions = BTreeSet::from([row]);
                        for bit in stage.low_bit()..=stage.high_bit() {
                            let h = 1usize << bit;
                            positions = positions.iter().flat_map(|&c| [c, c ^ h]).collect();
                        }
                        for column in positions {
                            cells[column] = true;
                        }
                    }
                    let window: BTreeSet<_> = (0..stage.d())
                        .map(|i| {
                            ((stage.offset() + i as i64) * stage.stride() as i64)
                                .rem_euclid(n as i64) as usize
                        })
                        .collect();
                    assert!(
                        stored.is_subset(&window),
                        "every stored offset must be in the window"
                    );
                    assert_eq!(window.len(), stage.d());
                    assert!(stage.d().is_power_of_two());
                    assert_eq!(stage.offset() % (stage.baby() as i64), 0);
                    let dense = matrix.to_dense();
                    for row in 0..n {
                        for column in 0..n {
                            assert_eq!(
                                dense[(row, column)] != num_complex::Complex64::new(0.0, 0.0),
                                reachable[row][column]
                            );
                        }
                    }
                    for i in 0..stage.d() {
                        let o = ((stage.offset() + i as i64) * stage.stride() as i64)
                            .rem_euclid(n as i64) as usize;
                        let main = (0..n).any(|row| {
                            reachable[row][(row + o) % n]
                                && (direction == Direction::SlotToCoeff
                                    || stage.low_bit() != 0
                                    || row % 2 == (row + o) % n % 2)
                        });
                        assert_eq!(stage.main_kept_indices().contains(&i), main);
                        if let Some(kept) = stage.second_kept_indices() {
                            let second = (0..n).any(|row| {
                                reachable[row][(row + o) % n]
                                    && (direction == Direction::SlotToCoeff
                                        || row % 2 != (row + o) % n % 2)
                            });
                            assert_eq!(kept.contains(&i), second);
                        }
                        // Actual scalar matrix diagonals supply a separate zero witness.
                        let actual_nonzero = matrix
                            .diagonal(o)
                            .is_some_and(|v| v.iter().any(|z| z.re != 0.0 || z.im != 0.0));
                        assert_eq!(actual_nonzero, stored.contains(&o));
                    }
                }
            }
        }
    }
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn small_runtime_receipts_match_encoding_scales_keys_and_all_counters() {
    let proof = DftSupportProof::verify_current().unwrap();
    for (sse, r, degree, cts, stc) in [
        (false, 0, 3, 1, 3),
        (true, 0, 4, 3, 1),
        (false, 1, 3, 2, 3),
        (true, 1, 4, 3, 2),
    ] {
        let (p, bp) = small(4, cts, stc, degree, r, sse);
        let p = Arc::new(p);
        let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
        assert_eq!(
            m.evalmod().coefficient_bits(),
            ckks_train_ckks::bootstrap::evalmod_coeffs(&bp)
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut kg = KeyGenerator::new(Arc::clone(&p), Rng::from_seed([41; 32])).unwrap();
        let encoder = Encoder::new(Arc::clone(&p));
        let runtime = Bootstrapper::new(Arc::clone(&p), bp, &encoder, &mut kg).unwrap();
        assert_eq!(&runtime.cost_per_boot(), m.cost_per_boot());
        assert_eq!(
            runtime.keys().rotation_steps(),
            m.generated_keys().rotation_steps()
        );
        assert_eq!(runtime.output_level(), m.output().level());
        for (metadata, encoded) in m
            .ctos_stages()
            .iter()
            .zip(runtime.coeff_to_slot_stages())
            .chain(m.stoc_stages().iter().zip(runtime.slot_to_coeff_stages()))
        {
            assert_eq!(encoded.scale.to_bits(), metadata.output().scale_bits());
            assert_eq!(encoded.level, metadata.input().level());
            for (i, pt) in encoded.diags.entries().iter().enumerate() {
                assert_eq!(pt.is_some(), metadata.main_kept_indices().contains(&i));
                if let Some(pt) = pt {
                    assert_eq!(pt.scale().to_bits(), metadata.diagonal_scale_bits());
                }
            }
            match (&encoded.second, metadata.second_kept_indices()) {
                (None, None) => {}
                (Some(term), Some(kept)) => {
                    for (i, pt) in term.entries().iter().enumerate() {
                        assert_eq!(pt.is_some(), kept.contains(&i));
                    }
                }
                _ => panic!("second term mismatch"),
            }
        }
        assert!(
            m.unvalidated_requirements()
                .contains(&UnvalidatedRequirement::EncodingQOverTwoFeasibility)
        );
        assert!(
            m.unvalidated_requirements()
                .contains(&UnvalidatedRequirement::EvalModInternalScaleUnvalidated)
        );
        if sse {
            let role = m.generated_keys().sse_to_sparse().unwrap();
            assert_eq!(
                (
                    role.level(),
                    role.q_prefix_len(),
                    role.digits(),
                    role.alpha()
                ),
                (0, 1, 1, 1)
            );
            let role = m.generated_keys().sse_to_main().unwrap();
            assert_eq!(
                (
                    role.level(),
                    role.q_prefix_len(),
                    role.digits(),
                    role.alpha()
                ),
                (p.max_level(), p.q().len(), p.dnum(), p.alpha())
            );
        }
    }
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn malformed_sizes_weights_packing_and_budgets_fail_without_panics() {
    let proof = DftSupportProof::verify_current().unwrap();
    let p = CkksParams::n16_boot();
    let base = BootParams::default_n16();
    let bad = [
        BootParams {
            ctos_levels: usize::MAX,
            ..base.clone()
        },
        BootParams {
            stoc_levels: usize::MAX,
            ..base.clone()
        },
        BootParams {
            evalmod_levels: usize::MAX,
            ..base.clone()
        },
        BootParams {
            cheb_degree: usize::MAX,
            ..base.clone()
        },
        BootParams {
            cheb_degree: usize::MAX / 2 + 1,
            ..base.clone()
        },
        BootParams {
            double_angle: 64,
            ..base.clone()
        },
        BootParams {
            double_angle: usize::MAX,
            ..base.clone()
        },
        BootParams {
            ctos_levels: 16,
            ..base.clone()
        },
        BootParams {
            log_slots: 14,
            ..base.clone()
        },
        BootParams {
            log_slots: 0,
            ..base.clone()
        },
        BootParams {
            log_slots: u32::MAX,
            ..base.clone()
        },
        BootParams {
            ctos_levels: 0,
            ..base.clone()
        },
        BootParams {
            stoc_levels: 0,
            ..base.clone()
        },
        BootParams {
            evalmod_levels: 0,
            ..base.clone()
        },
        BootParams {
            cheb_degree: 0,
            ..base.clone()
        },
        BootParams {
            k_intervals: 0,
            ..base.clone()
        },
        BootParams {
            k_intervals: 1,
            ..base.clone()
        },
        BootParams {
            ephemeral_hw: Some(0),
            ..base.clone()
        },
        BootParams {
            ephemeral_hw: Some(usize::MAX),
            ..base.clone()
        },
        BootParams {
            ephemeral_hw: Some(p.n() / 2),
            ..base.clone()
        },
        BootParams {
            sparse_hw: Some(1),
            ..base.clone()
        },
    ];
    for bp in bad {
        assert!(
            BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).is_err(),
            "{bp:?}"
        );
    }
    for h in [0, usize::MAX] {
        let p = p.clone().with_hamming_weight(h);
        let bp = BootParams {
            sparse_hw: Some(h),
            ..base.clone()
        };
        assert!(BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).is_err());
    }
    for (limits, resource) in [
        (
            MetadataLimits {
                max_coefficients: 127,
                ..MetadataLimits::default()
            },
            MetadataResource::Coefficients,
        ),
        (
            MetadataLimits {
                max_interpolation_terms: 16383,
                ..MetadataLimits::default()
            },
            MetadataResource::InterpolationTerms,
        ),
        (
            MetadataLimits {
                max_polynomial_work: 1,
                ..MetadataLimits::default()
            },
            MetadataResource::PolynomialWork,
        ),
        (
            MetadataLimits {
                max_peak_bytes: 1,
                ..MetadataLimits::default()
            },
            MetadataResource::RequestedBytes,
        ),
    ] {
        assert!(
            matches!(BootstrapMetadata::with_limits_and_verified_support(&p,&base,limits,&proof),Err(BootstrapMetadataError::MetadataBudgetExceeded{resource:r,..}) if r==resource)
        );
    }
    let valid = BootstrapMetadata::new_with_verified_support(&p, &base, &proof).unwrap();
    let bytes = valid.resource_bounds().requested_bytes();
    assert!(
        BootstrapMetadata::with_limits_and_verified_support(
            &p,
            &base,
            MetadataLimits {
                max_peak_bytes: bytes - 1,
                ..MetadataLimits::default()
            },
            &proof
        )
        .is_err()
    );
    let same = BootstrapMetadata::with_limits_and_verified_support(
        &p,
        &base,
        MetadataLimits {
            max_peak_bytes: bytes,
            ..MetadataLimits::default()
        },
        &proof,
    )
    .unwrap();
    assert_eq!(
        valid.descriptor_identity_bytes(),
        same.descriptor_identity_bytes()
    );
    eprintln!(
        "default requested bytes={bytes}, work={}, compiler={:?}, target={:?}",
        valid.resource_bounds().polynomial_work(),
        proof.configured_compiler(),
        proof.target()
    );
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn identities_bind_contents_not_arc_or_presets() {
    let proof = DftSupportProof::verify_current().unwrap();
    let p = CkksParams::n16_boot();
    let bp = BootParams::default_n16();
    let base = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
    let clone =
        BootstrapMetadata::new_with_verified_support(&p.clone(), &bp.clone(), &proof).unwrap();
    assert_eq!(
        base.parameter_identity_bytes(),
        clone.parameter_identity_bytes()
    );
    for changed in [
        BootParams {
            k_intervals: 14,
            ..bp.clone()
        },
        BootParams {
            cheb_degree: 126,
            ..bp.clone()
        },
        BootParams {
            ephemeral_hw: Some(31),
            ..bp.clone()
        },
        BootParams {
            ctos_levels: 3,
            ..bp.clone()
        },
        BootParams {
            stoc_levels: 3,
            ..bp.clone()
        },
    ] {
        let m = BootstrapMetadata::new_with_verified_support(&p, &changed, &proof).unwrap();
        assert_ne!(
            m.parameter_identity_bytes(),
            base.parameter_identity_bytes()
        );
    }
    let sparse = CkksParams::n16_boot_sparse();
    let m =
        BootstrapMetadata::new_with_verified_support(&sparse, &BootParams::sparse_n16(), &proof)
            .unwrap();
    assert_ne!(
        base.parameter_identity_bytes(),
        m.parameter_identity_bytes()
    );
    assert_eq!(m.cost_per_boot(), base.cost_per_boot());
}

#[test]
fn nonzero_unencoded_diagonal_remains_some_when_plaintext_rounds_to_zero() {
    use ckks_train_ckks::Plaintext;
    use ckks_train_ckks::linalg::Diagonals;
    let p = Arc::new(CkksParams::test_tiny());
    let encoder = Encoder::new(Arc::clone(&p));
    let tiny = vec![num_complex::Complex64::new(1e-30, 0.0); p.slots()];
    let zero = vec![num_complex::Complex64::new(0.0, 0.0); p.slots()];
    let diags = Diagonals::<Option<Plaintext>>::encode_complex_sparse(
        &encoder,
        &[tiny, zero],
        1,
        0,
        2f64.powi(20),
        1,
    )
    .unwrap();
    let pt = diags.entries()[0]
        .as_ref()
        .expect("pre-encoding nonzero remains Some");
    assert!(pt.poly().limbs().iter().flatten().all(|&v| v == 0));
    assert!(diags.entries()[1].is_none());
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn small_observed_bootstrap_counter_equals_structural_receipt() {
    use ckks_train_ckks::{Encryptor, Evaluator};
    let proof = DftSupportProof::verify_current().unwrap();
    let (p, bp) = small(4, 4, 4, 3, 0, true);
    let p = Arc::new(p);
    let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
    let mut rng = Rng::from_seed([67; 32]);
    let mut kg = KeyGenerator::new(Arc::clone(&p), Rng::from_seed([68; 32])).unwrap();
    let encoder = Encoder::new(Arc::clone(&p));
    let encryptor = Encryptor::new(Arc::clone(&p), kg.public_key());
    let evaluator = Evaluator::new(Arc::clone(&p), Some(kg.relin_key()), None);
    let runtime = Bootstrapper::new(Arc::clone(&p), bp, &encoder, &mut kg).unwrap();
    let pt = encoder
        .encode(&vec![0.01; p.slots()], p.scale(), p.max_level())
        .unwrap();
    let ct = encryptor.encrypt(&pt, &mut rng);
    let ct = evaluator.mod_switch_to_level(&ct, 0).unwrap();
    let before = evaluator.counter();
    let output = runtime.bootstrap(&evaluator, &ct).unwrap();
    assert_eq!(&(&evaluator.counter() - &before), m.cost_per_boot());
    assert_eq!(output.level(), m.output().level());
    assert_eq!(output.scale().to_bits(), m.output().scale_bits());
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn stage_cost_places_final_ctos_conjugation_only_in_direction_total() {
    let proof = DftSupportProof::verify_current().unwrap();
    let (p, bp) = small(4, 3, 2, 3, 0, true);
    let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
    let mut stage_rotations = 0;
    for stage in m.ctos_stages() {
        let expected = stage.main_baby_steps().len()
            + stage.second_baby_steps().map_or(0, <[i32]>::len)
            + stage.shared_giant_steps().len()
            + usize::from(stage.second_kept_indices().is_some());
        assert_eq!(stage.cost().rotations, expected as u64);
        stage_rotations += stage.cost().rotations;
    }
    assert_eq!(m.ctos_cost().rotations, stage_rotations + 1);
}

#[cfg(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
))]
#[test]
fn key_requirements_carry_parameter_binding_and_max_scalar_budget_is_operational() {
    let proof = DftSupportProof::verify_current().unwrap();
    let p = CkksParams::n16_boot();
    let bp = BootParams {
        cheb_degree: 4095,
        evalmod_levels: 12,
        ..BootParams::default_n16()
    };
    let m = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
    assert_eq!(m.evalmod().coefficient_bits().len(), 4096);
    assert_eq!(m.resource_bounds().interpolation_terms(), 1 << 24);
    assert_eq!(
        m.generated_keys().parameter_identity_bytes(),
        m.parameter_identity_bytes()
    );
    let too_large = BootParams {
        cheb_degree: 4096,
        evalmod_levels: 13,
        ..bp
    };
    assert!(matches!(
        BootstrapMetadata::with_limits_and_verified_support(
            &p,
            &too_large,
            MetadataLimits {
                max_coefficients: 8192,
                ..MetadataLimits::default()
            },
            &proof
        ),
        Err(BootstrapMetadataError::MetadataBudgetExceeded {
            resource: MetadataResource::InterpolationTerms,
            ..
        })
    ));
}

// Unsupported environments must retain the public named boundary, not mint a token.
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_os = "macos",
        target_pointer_width = "64"
    ),
    all(
        target_arch = "x86_64",
        target_vendor = "unknown",
        target_os = "linux",
        target_env = "gnu",
        target_pointer_width = "64"
    )
)))]
#[test]
fn unsupported_target_returns_named_dft_kept_support_boundary() {
    use ckks_train_ckks::bootstrap::metadata::{
        BootstrapMetadataError, DftProofFailure, MissingRequirement,
    };
    let error =
        DftSupportProof::verify_current().expect_err("unsupported target must not mint a proof");
    let expected = if usize::BITS != 64 {
        DftProofFailure::UnsupportedPointerWidth
    } else {
        DftProofFailure::UnsupportedTargetEnvironment
    };
    assert!(matches!(error, BootstrapMetadataError::MissingCode {
        requirement: MissingRequirement::DftKeptSupportProof,
        reason,
    } if reason == expected));
}
