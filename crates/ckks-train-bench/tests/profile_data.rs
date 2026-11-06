//! Shared profile input behavior on synthetic files; no benchmark runs or downloads.
use ckks_train_bench::data::{Tabular, sha256_hex, write_idx_u8};
use ckks_train_bench::profile_data::*;
use ndarray::{Array2, arr2};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "hekit-profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }
    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }
    fn locked(&self, name: &str, bytes: &[u8]) -> LockedFile {
        LockedFile {
            path: self.write(name, bytes),
            sha256: sha256_hex(bytes),
        }
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn npy(rows: usize, cols: usize, vals: &[f64]) -> Vec<u8> {
    let h = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({rows}, {cols}), }}\n");
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((h.len() as u16).to_le_bytes());
    b.extend(h.as_bytes());
    b.extend(vals.iter().flat_map(|v| v.to_le_bytes()));
    b
}
fn request(id: &str, classes: usize, features: usize, source: DataSource) -> DataRequest {
    DataRequest {
        profile_id: format!("{id}__fixture"),
        dataset_id: id.into(),
        classes,
        features,
        train_samples: None,
        test_samples: None,
        source,
    }
}
fn write_npy_dataset(t: &Temp, bad_label: f64, nan: bool) {
    let mut manifest = serde_json::Map::new();
    for (name, rows, cols, vals) in [
        (
            "train_images_14x14.npy",
            2,
            196,
            vec![if nan { f64::NAN } else { 0.25 }; 392],
        ),
        ("train_labels.npy", 1, 2, vec![0.0, bad_label]),
        ("test_images_14x14.npy", 1, 196, vec![1.5; 196]),
        ("test_labels.npy", 1, 1, vec![9.0]),
    ] {
        let b = npy(rows, cols, &vals);
        t.write(name, &b);
        manifest.insert(name.into(), serde_json::json!(sha256_hex(&b)));
    }
    t.write(
        "data_manifest.json",
        serde_json::to_string(&manifest).unwrap().as_bytes(),
    );
}

#[test]
fn iris_fixed_partition_preserves_row_order_and_train_only_zscore() {
    // Removing training-only scaling or sorting the declared split changes these values.
    let t = Temp::new();
    let csv = t.write(
        "iris.csv",
        b"a,b,c,d,label\n1,2,3,4,0\n3,4,5,6,1\n100,200,300,400,2\n",
    );
    let split = t.write("split.txt", b"train:1,0\ntest:2\n");
    let q = request(
        "iris",
        3,
        4,
        DataSource::IrisCsv {
            csv: csv.clone(),
            split,
        },
    );
    let a = load_dataset(&q).unwrap();
    assert_eq!(a.train_y, [1, 0]);
    assert_eq!(a.train_x, arr2(&[[1., 1., 1., 1.], [-1., -1., -1., -1.]]));
    assert_eq!(a.test_x[[0, 0]], 98.);
    assert_eq!(a.train_indices, [1, 0]);
    assert_eq!(a.files.len(), 2);
    fs::write(
        csv,
        b"a,b,c,d,label\n1,2,3,4,0\n3,4,5,6,1\n1000,2000,3000,4000,2\n",
    )
    .unwrap();
    assert_eq!(load_dataset(&q).unwrap().train_x, a.train_x);
}

#[test]
fn tabular_scaling_uses_training_rows_and_rejects_bad_partitions() {
    // A held-out row entering fitted min/max or std breaks the literal train values.
    let mut table = Tabular {
        features: Array2::from_shape_fn((3, 30), |(i, j)| [2., 4., 100.][i] + j as f64),
        labels: vec![0, 1, 1],
        ids: vec![1, 2, 3],
    };
    let a = tabular_dataset(
        "bc__fixture",
        "breast_cancer",
        &table,
        &[0, 1],
        &[2],
        2,
        Scaling::MinMax,
    )
    .unwrap();
    assert_eq!(a.train_x.row(0).to_vec(), vec![0.; 30]);
    assert_eq!(a.train_x.row(1).to_vec(), vec![1.; 30]);
    assert_eq!(a.test_x[[0, 0]], 49.);
    table.features.row_mut(2).fill(10000.);
    assert_eq!(
        tabular_dataset("b", "bc", &table, &[0, 1], &[2], 2, Scaling::MinMax)
            .unwrap()
            .train_x,
        a.train_x
    );
    let z = tabular_dataset("b", "bc", &table, &[0, 1], &[2], 2, Scaling::Standardize).unwrap();
    assert_eq!(z.train_x[[0, 0]], -1.);
    assert_eq!(z.train_x[[1, 0]], 1.);
    for (tr, te) in [
        (vec![0, 1], vec![1]),
        (vec![0], vec![2]),
        (vec![0, 1], vec![3]),
        (vec![], vec![0, 1, 2]),
    ] {
        assert!(tabular_dataset("b", "bc", &table, &tr, &te, 2, Scaling::MinMax).is_err());
    }
}

#[test]
fn four_dataset_shapes_and_class_padding_are_real_class_aware() {
    // Accidentally treating every source as MNIST or scoring padded channels fails here.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let iris = load_dataset(&request(
        "iris",
        3,
        4,
        DataSource::IrisCsv {
            csv: root.join("data/iris.csv"),
            split: root.join("data/iris_split_seed2020.txt"),
        },
    ))
    .unwrap();
    assert_eq!(
        (iris.train_x.dim(), iris.test_x.dim(), iris.classes),
        ((120, 4), (30, 4), 3)
    );
    let bc = load_dataset(&request(
        "breast_cancer",
        2,
        30,
        DataSource::Wdbc {
            path: root.join("data/breast_cancer.csv"),
            test_size: 114,
            split_seed: 0,
            scaling: Scaling::MinMax,
        },
    ))
    .unwrap();
    assert_eq!(
        (bc.train_x.dim(), bc.test_x.dim(), bc.classes),
        ((455, 30), (114, 30), 2)
    );
    let t = Temp::new();
    for (split, n) in [
        (ckks_train_bench::data::Split::Train, 2),
        (ckks_train_bench::data::Split::Test, 1),
    ] {
        let (im, la) = split.files();
        t.write(
            im,
            &write_idx_u8(
                &[n, 28, 28],
                &(0..n * 784).map(|i| (i % 256) as u8).collect::<Vec<_>>(),
            ),
        );
        t.write(la, &write_idx_u8(&[n], &vec![1; n]));
    }
    let fashion = load_dataset(&request(
        "fashion_mnist",
        10,
        196,
        DataSource::Idx {
            dir: t.0.clone(),
            resize: Some(14),
        },
    ))
    .unwrap();
    assert_eq!((fashion.train_x.dim(), fashion.classes), ((2, 196), 10));
    write_npy_dataset(&t, 1., false);
    let mn = load_dataset(&request(
        "mnist_released",
        10,
        196,
        DataSource::Npy { dir: t.0.clone() },
    ))
    .unwrap();
    assert_eq!((mn.train_x.dim(), mn.classes), ((2, 196), 10));
    assert_eq!(mn.test_x[[0, 0]], 1.5, "captured values unchanged");
    let target = padded_targets(&[2, 0], 3, 4).unwrap();
    assert_eq!(target, arr2(&[[0., 0., 1., 0.], [1., 0., 0., 0.]]));
    assert_eq!(
        class_accuracy(&arr2(&[[0., 1., 3., 999.], [2., 1., 0., 999.]]), &[2, 0], 3).unwrap(),
        1.
    );
    assert!(padded_targets(&[3], 3, 4).is_err());
    assert!(padded_targets(&[0], 3, 2).is_err());
    assert!(class_accuracy(&arr2(&[[0., f64::NAN, 1.]]), &[2], 3).is_err());
}

#[test]
fn bad_labels_shapes_nonfinite_and_manifest_inputs_are_refused() {
    let t = Temp::new();
    write_npy_dataset(&t, 10., false);
    let mut q = request(
        "mnist_released",
        10,
        196,
        DataSource::Npy { dir: t.0.clone() },
    );
    assert!(load_dataset(&q).is_err());
    write_npy_dataset(&t, 1., true);
    assert!(load_dataset(&q).is_err());
    write_npy_dataset(&t, 1., false);
    q.features = 4;
    assert!(load_dataset(&q).is_err());
    q.features = 196;
    q.train_samples = Some(4800);
    assert!(load_dataset(&q).is_err());
    q.train_samples = None;
    t.write("data_manifest.json",b"{\"test_labels.npy\":\"0000000000000000000000000000000000000000000000000000000000000000\"}");
    assert!(
        load_dataset(&q).is_err(),
        "missing file-specific manifest entry"
    );
    write_npy_dataset(&t, 1., false);
    t.write("train_labels.npy", b"bad");
    assert!(load_dataset(&q).is_err());
    fs::remove_file(t.0.join("data_manifest.json")).unwrap();
    assert!(load_dataset(&q).is_err());
    assert!(parse_iris("a,b,c,d,label\nNaN,1,2,3,0").is_err());
    assert!(parse_iris("a,b,c,d,label\n0,1,2,3,3").is_err());
    assert!(parse_partition("train:0,1\ntrain:2\ntest:3", 4).is_err());
    assert!(parse_partition("train:0,1\ntest:1,2", 3).is_err());
}

#[test]
fn captured_seed_inputs_require_exact_seed_hash_shape_order_and_schedule() {
    // Any seed0 fallback, unchecked hash/shape, non-permutation order or schedule omission fails.
    let t = Temp::new();
    let init = t.locked("weights.npy", &npy(2, 3, &[0.; 6]));
    let order = t.locked("order.txt", b"2 0 1 1 2 0");
    let schedule = t.locked(
        "config.json",
        b"{\"per_seed\":{\"0\":{\"lr_per_step\":[0.1,0.2]}}}",
    );
    let files = SeedFiles {
        initialization: vec![ArrayInput {
            file: init,
            shape: (2, 3),
        }],
        order: Some(OrderInput {
            file: order,
            population: 3,
            passes: 2,
            coverage: OrderCoverage::FullPermutation,
        }),
        schedule: Some(ScheduleInput {
            file: schedule,
            json_pointer: "/per_seed/0/lr_per_step".into(),
            steps: 2,
        }),
    };
    let mut contract = CapturedInputs {
        profile_id: "mnist_released__emlp2".into(),
        per_seed: BTreeMap::from([(0, files)]),
    };
    let good = preflight_seed(&contract, 0).unwrap();
    assert_eq!(good.order, Some(vec![2, 0, 1, 1, 2, 0]));
    assert_eq!(good.schedule, Some(vec![0.1, 0.2]));
    assert_eq!(good.files.len(), 3);
    assert!(
        preflight_seed(&contract, 3)
            .unwrap_err()
            .to_string()
            .contains("missing")
    );
    contract.per_seed.get_mut(&0).unwrap().initialization[0].shape = (3, 2);
    assert!(preflight_seed(&contract, 0).is_err());
    contract.per_seed.get_mut(&0).unwrap().initialization[0].shape = (2, 3);
    contract
        .per_seed
        .get_mut(&0)
        .unwrap()
        .order
        .as_mut()
        .unwrap()
        .file = t.locked("bad-order.txt", b"0 1 1 0 1 2");
    assert!(preflight_seed(&contract, 0).is_err());
    contract.per_seed.get_mut(&0).unwrap().order = Some(OrderInput {
        file: t.locked("subset.txt", b"2 0 1 2"),
        population: 3,
        passes: 2,
        coverage: OrderCoverage::UniqueSubset {
            samples_per_pass: 2,
        },
    });
    assert_eq!(
        preflight_seed(&contract, 0).unwrap().order,
        Some(vec![2, 0, 1, 2])
    );
    contract
        .per_seed
        .get_mut(&0)
        .unwrap()
        .order
        .as_mut()
        .unwrap()
        .coverage = OrderCoverage::FullPermutation;
    assert!(
        preflight_seed(&contract, 0).is_err(),
        "coverage is never inferred from length"
    );
    contract.per_seed.get_mut(&0).unwrap().order = None;
    contract
        .per_seed
        .get_mut(&0)
        .unwrap()
        .schedule
        .as_mut()
        .unwrap()
        .json_pointer = "/per_seed/3/lr_per_step".into();
    assert!(preflight_seed(&contract, 0).is_err());
    contract.per_seed.get_mut(&0).unwrap().schedule = None;
    fs::write(
        &contract.per_seed[&0].initialization[0].file.path,
        b"tampered",
    )
    .unwrap();
    assert!(preflight_seed(&contract, 0).is_err());
}

#[test]
fn idx_training_scale_is_invariant_to_heldout_pixels_and_handles_bad_shapes() {
    let t = Temp::new();
    let (tr_img, tr_lab) = ckks_train_bench::data::Split::Train.files();
    let (te_img, te_lab) = ckks_train_bench::data::Split::Test.files();
    t.write(
        tr_img,
        &write_idx_u8(&[2, 2, 2], &[0, 20, 40, 60, 80, 100, 120, 140]),
    );
    t.write(tr_lab, &write_idx_u8(&[2], &[0, 1]));
    t.write(te_img, &write_idx_u8(&[1, 2, 2], &[255; 4]));
    t.write(te_lab, &write_idx_u8(&[1], &[1]));
    let q = request(
        "fashion_fixture",
        2,
        4,
        DataSource::Idx {
            dir: t.0.clone(),
            resize: Some(2),
        },
    );
    let a = load_dataset(&q).unwrap();
    assert_eq!(a.train_x[[0, 0]], 0.);
    assert_eq!(a.train_x[[1, 3]], 1.);
    assert!(
        a.test_x[[0, 0]] > 1.,
        "heldout values do not expand fitted range"
    );
    t.write(te_img, &write_idx_u8(&[1, 2, 2], &[0; 4]));
    assert_eq!(load_dataset(&q).unwrap().train_x, a.train_x);
    t.write(te_img, &write_idx_u8(&[1, 1, 4], &[0; 4]));
    assert!(
        load_dataset(&q).is_err(),
        "equal flattened widths do not excuse wrong image axes"
    );
    t.write(te_img, &write_idx_u8(&[1, 2, 2], &[0; 4]));
    t.write(te_lab, &write_idx_u8(&[1], &[2]));
    assert!(
        load_dataset(&q).is_err(),
        "labels checked against declared real classes"
    );
}

#[test]
fn captured_preflight_rejects_schedule_length_sign_and_nonfinite_weights() {
    let t = Temp::new();
    let inputs = SeedFiles {
        initialization: vec![ArrayInput {
            file: t.locked("w.npy", &npy(1, 1, &[1.])),
            shape: (1, 1),
        }],
        order: Some(OrderInput {
            file: t.locked("o.txt", b"0 3 1"),
            population: 3,
            passes: 1,
            coverage: OrderCoverage::FullPermutation,
        }),
        schedule: None,
    };
    let mut c = CapturedInputs {
        profile_id: "fixture".into(),
        per_seed: BTreeMap::from([(2, inputs)]),
    };
    assert!(preflight_seed(&c, 2).is_err(), "out-of-range order");
    c.per_seed.get_mut(&2).unwrap().order = None;
    for bytes in [
        b"{\"lr\":[0.1]}".as_slice(),
        b"{\"lr\":[0.1,-0.2]}",
        b"{\"lr\":[0.1,null]}",
    ] {
        c.per_seed.get_mut(&2).unwrap().schedule = Some(ScheduleInput {
            file: t.locked("s.json", bytes),
            json_pointer: "/lr".into(),
            steps: 2,
        });
        assert!(preflight_seed(&c, 2).is_err());
    }
    c.per_seed.get_mut(&2).unwrap().schedule = None;
    c.per_seed.get_mut(&2).unwrap().initialization[0].file =
        t.locked("w.npy", &npy(1, 1, &[f64::INFINITY]));
    assert!(preflight_seed(&c, 2).is_err());
}

#[test]
fn pure_tabular_preprocessing_requires_declared_profile_and_dataset_identity() {
    // The pure API must not let an adapter drop identity while bypassing DataRequest.
    let table = Tabular {
        features: arr2(&[[1.], [3.], [9.]]),
        labels: vec![0, 1, 0],
        ids: vec![0, 1, 2],
    };
    for (profile, dataset) in [("", "iris"), ("iris__fixture", "")] {
        assert!(
            tabular_dataset(
                profile,
                dataset,
                &table,
                &[0, 1],
                &[2],
                2,
                Scaling::Standardize
            )
            .is_err()
        );
    }
}
