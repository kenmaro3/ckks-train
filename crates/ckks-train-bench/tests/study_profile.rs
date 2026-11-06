//! Scientific input-binding checks; no training, HE allocation or external execution.
use ckks_train_bench::{
    data::sha256_hex,
    profile_data::{DataSource, LockedFile, OrderCoverage},
    study_profile::*,
};
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn registry() -> StudyRegistry {
    StudyRegistry::load(root()).unwrap()
}
fn order_lock(p: &ResolvedProfile) -> LockedFile {
    let path = p.metadata().captured_order.as_ref().unwrap().path.clone();
    LockedFile {
        sha256: sha256_hex(&std::fs::read(&path).unwrap()),
        path,
    }
}
#[test]
fn all_twenty_one_profiles_bind_actual_source_graph_and_dataset() {
    let r = registry();
    assert_eq!(r.profile_ids().len(), 21);
    for id in r.profile_ids() {
        for seed in 0..=9 {
            let p = r.resolve(&id, seed).unwrap();
            let m = p.metadata();
            assert_eq!(m.seed, seed);
            assert_eq!(m.identity.registry.sha256, APPROVED_REGISTRY_SHA256);
            assert_eq!(p.data_request().classes, m.classes);
            assert_eq!(m.layers.last().unwrap().output, m.output_width);
            if m.model == Model::Mlp2020 {
                assert_eq!(m.layers.len(), 2);
                assert!(m.layers.iter().all(|l| l.bias));
                assert_eq!(m.loss, Loss::MeanOutputMse);
            } else {
                assert_eq!(m.output_width, m.classes.next_power_of_two());
                assert!(m.layers.iter().all(|l| !l.bias));
            }
            match m.model {
                Model::Emlp2 => {
                    assert_eq!(m.layers.len(), 4);
                    assert_eq!(m.layers[0].output, 64)
                }
                Model::Emlp3 => {
                    assert_eq!(m.layers.len(), 6);
                    assert_eq!(m.layers[0].output, 128);
                    assert_eq!(m.layers[2].input, 128)
                }
                _ => {}
            }
            assert!(!m.identity.sources.is_empty());
        }
    }
    let p = r.resolve("iris__emlp3", 7).unwrap();
    let m = p.metadata();
    assert_eq!((m.classes, m.output_width, m.layers[0].input), (3, 4, 4));
    assert_eq!(m.layers[1].role, LayerRole::LocalHead { block: 0 });
    assert_eq!(m.layers[4].role, LayerRole::OutputHidden);
    assert!(matches!(
        p.data_request().source,
        DataSource::IrisCsv { .. }
    ));
    let p = r.resolve("breast_cancer__emlp2", 4).unwrap();
    assert_eq!(
        (
            p.data_request().train_samples,
            p.data_request().test_samples
        ),
        (Some(455), Some(114))
    );
    assert!(matches!(
        p.data_request().source,
        DataSource::Wdbc {
            split_seed: 0,
            test_size: 114,
            ..
        }
    ));
    let p = r.resolve("fashion_mnist__emlp1", 4).unwrap();
    assert!(matches!(
        p.data_request().source,
        DataSource::Idx {
            resize: Some(14),
            ..
        }
    ));
}
#[test]
fn captured_selected_model_seed_shapes_orientation_and_missing_seed() {
    let r = registry();
    let p = r.resolve("mnist_released__emlp3", 2).unwrap();
    let m = p.metadata();
    let a = match &m.initialization {
        Initialization::Captured(a) => a,
        _ => panic!("captured"),
    };
    assert_eq!(a.len(), 6);
    assert_eq!(
        m.schedule.as_ref().unwrap().json_pointer,
        "/models/emlp3/per_seed/2/lr_per_step"
    );
    assert_eq!(a[0].logical_shape, (128, 196));
    assert_eq!(a[0].source_shape, (196, 128));
    assert_eq!(a[0].orientation, Orientation::InputByOutput);
    assert!(
        a[0].file
            .path
            .to_string_lossy()
            .contains("emlp3_seed2/linear_fwd_00.npy")
    );
    let lock = order_lock(&p);
    assert_eq!(
        m.captured_order.as_ref().unwrap().sha256.as_deref(),
        Some(lock.sha256.as_str())
    );
    let v = p.preflight(Some(&lock)).unwrap();
    assert_eq!(
        v.captured.as_ref().unwrap().initialization[0].dim(),
        (128, 196)
    );
    assert_eq!(v.plan.updates.len(), 1000);
    let missing = r.resolve("mnist_released__emlp3", 3).unwrap();
    assert!(
        missing
            .metadata()
            .captured_order
            .as_ref()
            .unwrap()
            .sha256
            .is_none()
    );
    assert!(
        missing
            .metadata()
            .captured_order
            .as_ref()
            .unwrap()
            .path
            .to_string_lossy()
            .contains("emlp3_seed3")
    );
    assert!(matches!(
        missing.preflight(None),
        Err(ProfileError::MissingInput { seed: 3, .. })
    ));
    let p2 = r.resolve("mnist_released__emlp2", 2).unwrap();
    let v2 = p2.preflight(None).unwrap();
    assert_eq!(
        v2.captured.as_ref().unwrap().initialization[0].dim(),
        (64, 196)
    );
    assert_ne!(
        v2.captured.as_ref().unwrap().order,
        v.captured.as_ref().unwrap().order
    );
    let raw =
        ckks_train_bench::data::parse_npy_f64(&std::fs::read(&a[0].file.path).unwrap()).unwrap();
    assert_eq!(
        v.captured.as_ref().unwrap().initialization[0][[0, 1]],
        raw[[1, 0]]
    );
    assert!(matches!(
        p2.preflight(Some(&lock)),
        Err(ProfileError::Conflict(_))
    ));
    let mut wrong = order_lock(&p2);
    wrong.sha256 = "0".repeat(64);
    assert!(p2.preflight(Some(&wrong)).is_err());
}
#[test]
fn captured_flat_order_and_sample_aligned_schedule_conserve_actual_boundaries() {
    let r = registry();
    for (id, n, b) in [
        ("mnist_released__emlp1", 1000, 48),
        ("mnist_released__emlp1__b64", 750, 64),
        ("mnist_released__emlp1__b128_candidate", 375, 128),
    ] {
        let p = r.resolve(id, 1).unwrap();
        let v = p.preflight(Some(&order_lock(&p))).unwrap();
        let plan = v.plan;
        assert_eq!(plan.updates.len(), n);
        assert_eq!(plan.processed_samples, 48000);
        assert_eq!(
            v.captured.as_ref().unwrap().order.as_ref().unwrap().len(),
            48000
        );
        assert_eq!(
            plan.updates.iter().map(|u| u.valid_samples).sum::<usize>(),
            48000
        );
        let schedule = v.captured.as_ref().unwrap().schedule.as_ref().unwrap();
        for u in &plan.updates {
            let i = u.processed_start / 48;
            assert_eq!(u.source_schedule_index, Some(i));
            if u.index > 0 {
                assert_eq!(
                    u.processed_start,
                    plan.updates[u.index - 1].processed_start
                        + plan.updates[u.index - 1].valid_samples
                );
            }
            assert_eq!(u.source_coefficient, schedule[i]);
            assert_eq!(u.mean_gradient_rate, schedule[i] * b as f64);
        }
        assert_eq!(plan.checkpoints.len(), 10);
        if b == 128 {
            assert_eq!(
                plan.checkpoints
                    .iter()
                    .map(|c| c.completed_updates)
                    .collect::<Vec<_>>(),
                vec![38, 75, 113, 150, 188, 225, 263, 300, 338, 375]
            );
            assert_eq!(plan.checkpoints[0].processed_samples, 4864);
            assert_eq!(plan.checkpoints[1].interval_samples, 4736);
            assert!(!plan.checkpoints[0].exact_logical_boundary);
            assert_eq!(plan.checkpoints[0].crossed_logical_boundaries, vec![1]);
            assert!(plan.checkpoints[1].exact_logical_boundary);
        }
        assert_eq!(
            plan.checkpoints
                .iter()
                .map(|c| c.interval_samples)
                .sum::<usize>(),
            48000
        );
        assert_eq!(
            plan.checkpoints
                .iter()
                .map(|c| c.interval_updates)
                .sum::<usize>(),
            n
        );
        assert!(matches!(
            p.metadata().captured_order.as_ref().unwrap().coverage,
            OrderCoverage::FullPermutation
        ));
    }
}
#[test]
fn local_recipes_and_adapted_rates_preserve_fixed_splits_and_tail_policy() {
    let r = registry();
    let p = r.resolve("iris__mlp2020", 9).unwrap();
    let Initialization::Recipe(recipe) = &p.metadata().initialization else {
        panic!("recipe")
    };
    assert_eq!(recipe.rng, RngConvention::LegacyIrisMasterFork);
    assert_eq!(recipe.seed, 9);
    assert_eq!(recipe.distribution, Distribution::Normal { sigma: 0.2 });
    let v = p.preflight(None).unwrap();
    assert!(v.captured.is_none());
    assert_eq!(v.plan.updates.len(), 2400);
    assert_eq!(v.plan.updates[0].mean_gradient_rate, 0.1);
    assert_eq!(v.plan.updates[0].delta_multiplier, 2.0 / 3.0);
    let p = r.resolve("breast_cancer__emlp1", 9).unwrap();
    let v = p.preflight(None).unwrap();
    assert_eq!(v.plan.updates.len(), 2800);
    assert_eq!(v.plan.processed_samples, 44800);
    assert_eq!(v.plan.dropped_samples, 700);
    assert!(matches!(
        p.metadata().order,
        OrderPolicy::PerEpochRecipe {
            coverage: OrderCoverage::UniqueSubset {
                samples_per_pass: 448
            },
            ..
        }
    ));
    let p = r.resolve("mnist_released__mlp2020__b64", 9).unwrap();
    let v = p.preflight(Some(&order_lock(&p))).unwrap();
    assert!(matches!(
        p.metadata().initialization,
        Initialization::Recipe(_)
    ));
    assert_eq!(v.plan.updates[0].mean_gradient_rate, 0.0005 * 64.0);
    assert_eq!(v.plan.updates[0].delta_multiplier, 0.2);
    assert!(v.captured.as_ref().unwrap().initialization.is_empty());
}
#[test]
fn unknown_profile_seed_and_conflicting_additional_order_lock_fail_closed() {
    let r = registry();
    assert!(matches!(
        r.resolve("FULL-1", 0),
        Err(ProfileError::UnsupportedProfile(_))
    ));
    assert!(matches!(
        r.resolve("iris__mlp2020", 10),
        Err(ProfileError::UnsupportedSeed(10))
    ));
    let p = r.resolve("mnist_released__emlp1", 0).unwrap();
    let mut lock = order_lock(&p);
    lock.sha256 = "0".repeat(64);
    assert!(matches!(
        p.preflight(Some(&lock)),
        Err(ProfileError::Conflict(_))
    ));
}

#[test]
fn canonical_dataset_load_proves_profile_class_and_fixed_partition_identity() {
    let p = registry().resolve("iris__emlp1", 8).unwrap();
    let prepared = p.prepare_dataset().unwrap();
    assert_eq!(prepared.dataset.profile_id, "iris__emlp1");
    assert_eq!(prepared.dataset.classes, 3);
    assert_eq!(prepared.dataset.train_x.dim(), (120, 4));
    assert_eq!(prepared.dataset.test_x.dim(), (30, 4));
    assert_eq!(prepared.dataset.files.len(), 2);
    assert!(prepared.dataset.prep.contains("training mean"));
    let p0 = registry().resolve("iris__emlp1", 0).unwrap();
    assert_eq!(
        prepared.dataset.train_indices,
        p0.prepare_dataset().unwrap().dataset.train_indices
    );
}

#[test]
fn portable_lock_and_bound_files_reject_working_copy_corruption_without_repair() {
    let r = registry();
    let p = r.resolve("mnist_released__emlp1", 0).unwrap();
    assert_eq!(
        p.metadata().identity.integrity_lock.sha256,
        APPROVED_INPUT_LOCK_SHA256
    );
    let lock: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root().join("crates/ckks-train-bench/configs/experiment_input_locks.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(lock["files"].as_array().unwrap().len(), 18);
    for f in lock["files"].as_array().unwrap() {
        let path = f["path"].as_str().unwrap();
        assert_eq!(
            sha256_hex(&std::fs::read(root().join(path)).unwrap()),
            f["sha256"].as_str().unwrap()
        );
    }
    let temp = std::env::temp_dir().join(format!("hekit-study-lock-{}", std::process::id()));
    std::fs::create_dir_all(temp.join("crates/ckks-train-bench/configs")).unwrap();
    for name in ["experiment_matrix.json", "experiment_input_locks.json"] {
        std::fs::copy(
            root().join("crates/ckks-train-bench/configs").join(name),
            temp.join("crates/ckks-train-bench/configs").join(name),
        )
        .unwrap();
    }
    let mut reauthorized = lock.clone();
    reauthorized["files"][0]["sha256"] = serde_json::Value::String("0".repeat(64));
    std::fs::write(
        temp.join("crates/ckks-train-bench/configs/experiment_input_locks.json"),
        serde_json::to_vec(&reauthorized).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        StudyRegistry::load(&temp),
        Err(ProfileError::Conflict(_))
    ));
    std::fs::write(
        temp.join("crates/ckks-train-bench/configs/experiment_input_locks.json"),
        b"{}",
    )
    .unwrap();
    assert!(matches!(
        StudyRegistry::load(&temp),
        Err(ProfileError::Conflict(_))
    ));
    assert_eq!(
        std::fs::read(temp.join("crates/ckks-train-bench/configs/experiment_input_locks.json"))
            .unwrap(),
        b"{}"
    );
    std::fs::copy(
        root().join("crates/ckks-train-bench/configs/experiment_input_locks.json"),
        temp.join("crates/ckks-train-bench/configs/experiment_input_locks.json"),
    )
    .unwrap();
    std::fs::write(
        temp.join("crates/ckks-train-bench/configs/experiment_matrix.json"),
        b"{broken",
    )
    .unwrap();
    assert!(matches!(
        StudyRegistry::load(&temp),
        Err(ProfileError::Conflict(_))
    ));
    std::fs::remove_dir_all(temp).unwrap();
}
