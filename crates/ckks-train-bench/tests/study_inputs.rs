//! Canonical materialization checks, without training or HE allocation.
use ckks_train_bench::{
    study_inputs::{CastPolicy, Dtype, StageInput, StudyInputError, materialize},
    study_profile::{ProfileError, StudyRegistry},
};
use ckks_train_core::sampler::Rng;
use std::path::Path;

fn registry() -> StudyRegistry {
    StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
}

// Catches a different fork, draw order, second transpose, or epoch reseeding.
#[test]
fn iris_matches_unchanged_legacy_normal_draws_and_in_place_epoch_order() {
    let r = registry();
    let p = materialize(&r.resolve("iris__mlp2020", 7).unwrap()).unwrap();
    let mut key = [0; 32];
    key[..8].copy_from_slice(&7u64.to_le_bytes());
    let mut master = Rng::from_seed(key);
    let mut init = master.fork();
    let mut shuffle = master.fork();
    let mut mismatches = Vec::new();
    let mut compared = 0;
    for (layer, (out, input)) in p.initial().layers().iter().zip([(10, 4), (3, 10)]) {
        assert_eq!(layer.weights().dim(), (out, input));
        for actual in layer.weights().iter().chain(layer.biases()) {
            // The unchanged iris.rs initializer calls sigma * normal(r): finish the
            // unscaled helper before multiplying. The fixed bits below independently
            // guard against reproducing the materializer's association in this oracle.
            let expected = 0.2 * legacy_unscaled_normal(&mut init);
            if actual.to_bits() != expected.to_bits() {
                mismatches.push((compared, actual.to_bits(), expected.to_bits()));
            }
            compared += 1;
        }
    }
    assert_eq!(compared, 83);
    assert!(
        mismatches.is_empty(),
        "legacy Normal mismatch_count={}/{}; first={:?}",
        mismatches.len(),
        compared,
        mismatches.first()
    );
    assert_eq!(
        p.initial().layers()[0].weights()[[1, 0]].to_bits(),
        0x3fb5_370e_f368_f5a7
    );
    let mut order: Vec<u32> = (0..120).collect();
    for epoch in 0..400 {
        for i in (1..order.len()).rev() {
            order.swap(i, shuffle.below(i as u64 + 1) as usize);
        }
        assert_eq!(p.order().pass_rows(epoch).unwrap(), order);
    }
    assert_eq!(p.update_rows(0).unwrap().rows(), &order_for_first_epoch());
    let last = p.update_rows(2399).unwrap();
    assert_eq!(last.rows(), &order[100..120]);
    assert_eq!(last.planned().processed_start, 47980);
    assert_eq!(last.crossed_boundaries(), &[400]);
    assert_eq!(p.dataset().train_indices.len(), 120);
    assert_ne!(p.dataset().train_indices[0], 0); // canonical rows and source rows are distinct.
}

// Verbatim evaluation sequence of the unchanged iris.rs normal helper; sigma is
// deliberately applied by the caller as in Net::init, after this product returns.
fn legacy_unscaled_normal(r: &mut Rng) -> f64 {
    let u1 = ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)).max(1e-300);
    let u2 = (r.next_u64() >> 11) as f64 * 2f64.powi(-53);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn order_for_first_epoch() -> Vec<u32> {
    let mut key = [0; 32];
    key[..8].copy_from_slice(&7u64.to_le_bytes());
    let mut master = Rng::from_seed(key);
    let _ = master.fork();
    let mut r = master.fork();
    let mut rows: Vec<u32> = (0..120).collect();
    for i in (1..120).rev() {
        rows.swap(i, r.below(i as u64 + 1) as usize);
    }
    rows[..20].to_vec()
}

// Catches drawing fresh f32 weights/preprocessing or a dtype-free digest.
#[test]
fn f32_is_once_cast_and_identity_binds_seed_and_dtype() {
    let r = registry();
    let p = materialize(&r.resolve("iris__emlp1", 0).unwrap()).unwrap();
    let q = materialize(&r.resolve("iris__emlp1", 0).unwrap()).unwrap();
    let other = materialize(&r.resolve("iris__emlp1", 1).unwrap()).unwrap();
    let a = p.numeric_f32();
    let b = p.numeric_f32();
    let wide = p.numeric_f64();
    assert!(std::ptr::eq(a.train_x(), b.train_x()));
    assert_eq!(a.dtype(), Dtype::F32);
    assert_eq!(wide.dtype(), Dtype::F64);
    assert_eq!(a.cast_policy(), CastPolicy::RoundToF32Once);
    for (f, d) in a.train_x().iter().zip(wide.train_x()) {
        assert_eq!(f.to_bits(), (*d as f32).to_bits());
    }
    for (f, d) in a.initial().layers().iter().zip(wide.initial().layers()) {
        for (x, y) in f.weights().iter().zip(d.weights()) {
            assert_eq!(x.to_bits(), (*y as f32).to_bits());
        }
        for (x, y) in f.biases().iter().zip(d.biases()) {
            assert_eq!(x.to_bits(), (*y as f32).to_bits());
        }
    }
    assert_eq!(p.identity().digest(), q.identity().digest());
    assert_ne!(p.identity().digest(), other.identity().digest());
    assert_ne!(
        p.identity().f32_inputs_sha256(),
        p.identity().f64_inputs_sha256()
    );
    assert!(p.identity().canonical_prepared());
    assert!(p.reference_kernel_rate_f64() >= p.plan().updates[0].mean_gradient_rate);
    assert!(p.update_rows(p.plan().updates.len()).is_err());
}

// Catches flattening block heads into the inference chain or reading updated weights.
#[test]
fn all_four_stage_graphs_keep_named_local_heads_and_pre_update_handoffs() {
    let r = registry();
    for (model, count) in [("mlp2020", 1), ("emlp1", 1), ("emlp2", 2), ("emlp3", 3)] {
        let p = materialize(&r.resolve(&format!("iris__{model}"), 0).unwrap()).unwrap();
        assert_eq!(p.graph().stages().len(), count);
        for (i, stage) in p.graph().stages().iter().enumerate() {
            assert_eq!(stage.layers(), 2 * i..2 * i + 2);
            assert_eq!(
                stage.input(),
                if i == 0 {
                    StageInput::Dataset
                } else {
                    StageInput::PreUpdateActivation { layer: 2 * (i - 1) }
                }
            );
            assert_eq!(
                stage.representation_layer(),
                if i + 1 < count { Some(2 * i) } else { None }
            );
            assert_eq!(
                p.initial().layers()[2 * i].name(),
                p.metadata().layers[2 * i].name
            );
        }
    }
}

#[test]
fn missing_captured_block_seed_cannot_be_materialized_with_a_fallback() {
    let r = registry();
    assert!(matches!(
        materialize(&r.resolve("mnist_released__emlp3", 3).unwrap()),
        Err(StudyInputError::Preparation(ProfileError::MissingInput {
            seed: 3,
            ..
        }))
    ));
}
