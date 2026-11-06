//! Bounded canonical Iris integration, never a full reference/target campaign.
use ckks_train_bench::{
    study_inputs::{CastPolicy, Dtype, materialize},
    study_plain::{PlainState, evaluate_plain, step_plain},
    study_profile::StudyRegistry,
};
use std::path::Path;

// Catches dtype recasting, skipped rows/rates, graph misdispatch and frozen bias drift.
#[test]
fn all_iris_graphs_bind_initial_cast_and_consume_two_exact_updates() {
    let registry =
        StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
    for model in ["mlp2020", "emlp1", "emlp2", "emlp3"] {
        let profile =
            materialize(&registry.resolve(&format!("iris__{model}"), 0).unwrap()).unwrap();
        let identity = profile.identity().digest().to_string();
        let first_rows = profile.order().rows()[..40].to_vec();
        let mut narrow = PlainState::<f32>::new(&profile).unwrap();
        let mut wide = PlainState::<f64>::new(&profile).unwrap();
        assert_eq!(narrow.input_identity(), wide.input_identity());
        assert_eq!(narrow.input_identity(), profile.identity().digest());
        assert_eq!(narrow.dtype(), Dtype::F32);
        assert_eq!(wide.dtype(), Dtype::F64);
        assert_eq!(narrow.cast_policy(), CastPolicy::RoundToF32Once);
        for (f, d) in narrow.layers().iter().zip(wide.layers()) {
            for (x, y) in f.weights().iter().zip(d.weights()) {
                assert_eq!(x.to_bits(), (*y as f32).to_bits());
            }
            for (x, y) in f.biases().iter().zip(d.biases()) {
                assert_eq!(x.to_bits(), (*y as f32).to_bits());
            }
            assert!(f.weight_velocity().iter().all(|&x| x == 0.0));
        }
        for index in 0..2 {
            let expected = profile.update_rows(index).unwrap();
            let a = step_plain(&mut narrow).unwrap();
            let b = step_plain(&mut wide).unwrap();
            assert_eq!(a.update_index, index);
            assert_eq!(a.valid_samples, expected.rows().len());
            assert_eq!(a, b);
        }
        assert_eq!(profile.identity().digest(), identity);
        assert_eq!(profile.order().rows()[..40], first_rows);
        for layer in narrow.layers() {
            assert!(
                layer
                    .weights()
                    .iter()
                    .chain(layer.weight_velocity())
                    .chain(layer.biases())
                    .chain(layer.bias_velocity())
                    .all(|x| x.is_finite())
            );
        }
        for layer in wide.layers() {
            assert!(
                layer
                    .weights()
                    .iter()
                    .chain(layer.weight_velocity())
                    .chain(layer.biases())
                    .chain(layer.bias_velocity())
                    .all(|x| x.is_finite())
            );
        }
        assert_eq!(narrow.completed_updates(), 2);
        assert_eq!(narrow.processed_samples(), 40);
        let a = evaluate_plain(&narrow).unwrap();
        let b = evaluate_plain(&wide).unwrap();
        assert_eq!(a.test_count, 30);
        assert_eq!(b.test_count, 30);
        assert!(a.correct <= a.test_count);
        assert!(b.correct <= b.test_count);
        assert!(a.accuracy().is_finite());
        assert_eq!(
            profile.order().rows()[..20],
            *profile.update_rows(0).unwrap().rows()
        );
        if model != "mlp2020" {
            for layer in narrow.layers() {
                assert!(
                    layer
                        .biases()
                        .iter()
                        .chain(layer.bias_velocity())
                        .all(|&x| x.to_bits() == 0)
                );
            }
        }
    }
}
