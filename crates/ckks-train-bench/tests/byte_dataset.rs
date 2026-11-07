//! Bounded exact-byte dataset seam tests; no training/runtime/context work.
use ckks_train_bench::data::{self, NPY_DATASET_FILES, Split, sha256_hex, write_idx_u8};
use ckks_train_bench::profile_data::{
    DataRequest, DataSource, Dataset, DatasetBytes, Scaling, decode_dataset, load_dataset,
};
use ndarray::arr2;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const CSV: &[u8] = b"a,b,c,d,label\n1,2,3,4,0\n3,4,5,6,1\n100,200,300,400,2\n";
const SPLIT: &[u8] = b"train:1,0\ntest:2\n";
const WDBC: &[u8] = include_bytes!("../data/breast_cancer.csv");

fn request(source: DataSource, classes: usize, features: usize) -> DataRequest {
    DataRequest {
        profile_id: "byte-fixture".into(),
        dataset_id: "declared-dataset".into(),
        classes,
        features,
        train_samples: None,
        test_samples: None,
        source,
    }
}
fn iris_request() -> DataRequest {
    request(
        DataSource::IrisCsv {
            csv: "absent/labels.npy".into(),
            split: "absent/images.npy".into(),
        },
        3,
        4,
    )
}
fn idx_arrays() -> [Vec<u8>; 4] {
    [
        write_idx_u8(&[2, 2, 2], &[0, 0, 0, 0, 255, 255, 255, 255]),
        write_idx_u8(&[2], &[0, 1]),
        write_idx_u8(&[1, 2, 2], &[128; 4]),
        write_idx_u8(&[1], &[2]),
    ]
}
fn idx_bytes(a: &[Vec<u8>; 4]) -> DatasetBytes<'_> {
    DatasetBytes::Idx {
        train_images: &a[0],
        train_labels: &a[1],
        test_images: &a[2],
        test_labels: &a[3],
    }
}
fn npy(shape: &str, values: &[f64]) -> Vec<u8> {
    let h = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({shape}), }}\n");
    let mut b = b"\x93NUMPY\x01\x00".to_vec();
    b.extend((h.len() as u16).to_le_bytes());
    b.extend(h.as_bytes());
    b.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    b
}
fn npy_arrays() -> [Vec<u8>; 4] {
    [
        npy("2, 2", &[1., 2., 3., 4.]),
        npy("2,", &[0., 1.]),
        npy("1, 2", &[5., 6.]),
        npy("1,", &[2.]),
    ]
}
fn manifest(a: &[Vec<u8>; 4]) -> Vec<u8> {
    let v: serde_json::Map<String, serde_json::Value> = NPY_DATASET_FILES
        .into_iter()
        .zip(a)
        .map(|(n, b)| (n.into(), sha256_hex(b).into()))
        .collect();
    serde_json::to_vec(&v).unwrap()
}
fn npy_bytes<'a>(m: &'a [u8], a: &'a [Vec<u8>; 4]) -> DatasetBytes<'a> {
    DatasetBytes::Npy {
        manifest: m,
        arrays: a.each_ref().map(Vec::as_slice),
    }
}
fn same(a: &Dataset, b: &Dataset) {
    assert_eq!(a.train_x, b.train_x);
    assert_eq!(a.test_x, b.test_x);
    assert_eq!(a.train_y, b.train_y);
    assert_eq!(a.test_y, b.test_y);
    assert_eq!(a.files, b.files);
    assert_eq!(a.prep, b.prep);
    assert_eq!(a.split_identity, b.split_identity);
    assert_eq!(a.train_indices, b.train_indices);
    assert_eq!(a.test_indices, b.test_indices);
    assert_eq!(a.profile_id, b.profile_id);
    assert_eq!(a.dataset_id, b.dataset_id);
    assert_eq!(a.classes, b.classes);
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!(
            "hekit-byte-dataset-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn byte_dataset_iris_uses_supplied_slices_and_request_identity() {
    // A path reopen, filename role inference, training refit on heldout, or row sorting fails.
    let q = iris_request();
    let a = decode_dataset(
        &q,
        DatasetBytes::IrisCsv {
            csv: CSV,
            split: SPLIT,
        },
    )
    .unwrap();
    assert_eq!(a.train_x, arr2(&[[1.; 4], [-1.; 4]]));
    assert_eq!(a.train_y, [1, 0]);
    assert_eq!(a.train_indices, [1, 0]);
    assert_eq!(a.test_x[[0, 0]], 98.);
    assert_eq!(
        a.files,
        [
            ("absent/labels.npy".into(), sha256_hex(CSV)),
            ("absent/images.npy".into(), sha256_hex(SPLIT))
        ]
    );
    let changed = b"a,b,c,d,label\n1,2,3,4,0\n3,4,5,6,1\n1000,2000,3000,4000,2\n";
    let b = decode_dataset(
        &q,
        DatasetBytes::IrisCsv {
            csv: changed,
            split: SPLIT,
        },
    )
    .unwrap();
    assert_eq!(a.train_x, b.train_x);
    assert_eq!(b.test_x[[0, 0]], 998.);
    assert_ne!(a.files[0].1, b.files[0].1);
    let b = decode_dataset(
        &q,
        DatasetBytes::IrisCsv {
            csv: CSV,
            split: b"train:0,1\ntest:2\n",
        },
    )
    .unwrap();
    assert_eq!(b.train_y, [0, 1]);
    assert_ne!(a.files[1].1, b.files[1].1);
}

#[test]
fn byte_dataset_idx_preserves_scaling_and_rejects_shape_and_labels() {
    // Missing /255 conversion, test-fitted resize range, and unchecked axes/classes fail.
    let mut a = idx_arrays();
    let mut q = request(
        DataSource::Idx {
            dir: "absent/idx".into(),
            resize: None,
        },
        3,
        4,
    );
    let raw = decode_dataset(&q, idx_bytes(&a)).unwrap();
    assert_eq!(raw.train_x, arr2(&[[0.; 4], [1.; 4]]));
    assert_eq!(raw.test_x[[0, 0]], 128. / 255.);
    assert_eq!(raw.files[2].1, sha256_hex(&a[2]));
    q.source = DataSource::Idx {
        dir: "absent/idx".into(),
        resize: Some(1),
    };
    q.features = 1;
    let resized = decode_dataset(&q, idx_bytes(&a)).unwrap();
    assert_eq!(resized.train_x, arr2(&[[0.], [1.]]));
    assert!((resized.test_x[[0, 0]] - 128. / 255.).abs() < 1e-15);
    a[2] = write_idx_u8(&[1, 2, 2], &[255; 4]);
    let changed = decode_dataset(&q, idx_bytes(&a)).unwrap();
    assert_eq!(changed.train_x, resized.train_x);
    assert_eq!(changed.test_x[[0, 0]], 1.);
    assert_ne!(changed.files[2].1, resized.files[2].1);
    a[2] = write_idx_u8(&[1, 1, 4], &[0; 4]);
    assert!(decode_dataset(&q, idx_bytes(&a)).is_err());
    a = idx_arrays();
    a[3] = write_idx_u8(&[1], &[3]);
    assert!(decode_dataset(&q, idx_bytes(&a)).is_err());
    a[3] = write_idx_u8(&[1], &[10]);
    assert!(decode_dataset(&q, idx_bytes(&a)).is_err());
    a = idx_arrays();
    a[0].pop();
    assert!(decode_dataset(&q, idx_bytes(&a)).is_err());
}

#[test]
fn byte_dataset_npy_checks_exact_manifest_arrays_and_vector_labels() {
    // A permissive scanner substituting another file's hash or flattening matrix labels fails.
    let q = request(
        DataSource::Npy {
            dir: "absent/npy".into(),
        },
        3,
        2,
    );
    let mut a = npy_arrays();
    let m = manifest(&a);
    let d = decode_dataset(&q, npy_bytes(&m, &a)).unwrap();
    assert_eq!(d.train_x, arr2(&[[1., 2.], [3., 4.]]));
    assert_eq!(d.test_x, arr2(&[[5., 6.]]));
    assert_eq!(d.train_y, [0, 1]);
    assert_eq!(
        d.files[4],
        ("absent/npy/data_manifest.json".into(), sha256_hex(&m))
    );
    a[2] = npy("1, 2", &[50., 60.]);
    assert!(decode_dataset(&q, npy_bytes(&m, &a)).is_err());
    let m2 = manifest(&a);
    let changed = decode_dataset(&q, npy_bytes(&m2, &a)).unwrap();
    assert_eq!(changed.test_x[[0, 0]], 50.);
    assert_ne!(changed.files[2].1, d.files[2].1);
    assert_ne!(changed.files[4].1, d.files[4].1);
    let mut missing: serde_json::Value = serde_json::from_slice(&m).unwrap();
    missing
        .as_object_mut()
        .unwrap()
        .remove(NPY_DATASET_FILES[0]);
    assert!(
        decode_dataset(
            &q,
            npy_bytes(&serde_json::to_vec(&missing).unwrap(), &npy_arrays())
        )
        .is_err()
    );
    for bad in [
        npy("2, 2", &[0., 1., 0., 1.]),
        npy("2,", &[0., 3.]),
        npy("2,", &[0., 0.5]),
    ] {
        a = npy_arrays();
        a[1] = bad;
        assert!(decode_dataset(&q, npy_bytes(&manifest(&a), &a)).is_err());
    }
    a = npy_arrays();
    a[0] = npy("2, 2", &[1., 2., f64::NAN, 4.]);
    assert!(decode_dataset(&q, npy_bytes(&manifest(&a), &a)).is_err());
    a = npy_arrays();
    a[2] = npy("1, 3", &[1., 2., 3.]);
    assert!(decode_dataset(&q, npy_bytes(&manifest(&a), &a)).is_err());
    assert!(decode_dataset(&q, npy_bytes(&[255], &a)).is_err());
    let a = npy_arrays();
    let listed = serde_json::json!({"files": NPY_DATASET_FILES.into_iter().zip(&a).map(|(file,b)| serde_json::json!({"file":file,"sha256_npy_file":sha256_hex(b)})).collect::<Vec<_>>()});
    assert_eq!(
        decode_dataset(&q, npy_bytes(&serde_json::to_vec(&listed).unwrap(), &a))
            .unwrap()
            .train_x,
        d.train_x
    );
}

#[test]
fn byte_dataset_wdbc_keeps_pinned_digest_and_fixed_split() {
    // Bypassing the fixed UCI pin, changing stream4, or losing training-only scaling fails.
    assert_eq!(WDBC.len(), 124103);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/breast_cancer.csv");
    let q = request(
        DataSource::Wdbc {
            path,
            test_size: 114,
            split_seed: 0,
            scaling: Scaling::MinMax,
        },
        2,
        30,
    );
    let a = decode_dataset(&q, DatasetBytes::Wdbc { csv: WDBC }).unwrap();
    assert_eq!((a.train_x.dim(), a.test_x.dim()), ((455, 30), (114, 30)));
    assert_eq!(a.files[0].1, data::WDBC_SHA256);
    assert!(a.train_indices.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(a.train_y.iter().filter(|&&c| c == 0).count(), 170);
    assert_eq!(a.test_y.iter().filter(|&&c| c == 0).count(), 42);
    same(&a, &load_dataset(&q).unwrap());
    let mut changed = WDBC.to_vec();
    changed[0] ^= 1;
    assert!(decode_dataset(&q, DatasetBytes::Wdbc { csv: &changed }).is_err());
    assert!(data::wdbc_from_bytes(&changed).is_err());
    assert!(data::wdbc_from_bytes(&[255]).is_err());
}

#[test]
fn byte_dataset_rejects_missing_wrong_variant_utf8_and_declared_shape() {
    // A fallback pathname load or unchecked bytes/declared shape fails these refusals.
    let mut q = iris_request();
    for input in [
        DatasetBytes::Wdbc { csv: CSV },
        DatasetBytes::IrisCsv {
            csv: &[],
            split: SPLIT,
        },
        DatasetBytes::IrisCsv {
            csv: &[255],
            split: SPLIT,
        },
        DatasetBytes::IrisCsv {
            csv: CSV,
            split: &[255],
        },
        DatasetBytes::IrisCsv {
            csv: CSV,
            split: &[],
        },
    ] {
        assert!(decode_dataset(&q, input).is_err());
    }
    q.train_samples = Some(3);
    assert!(
        decode_dataset(
            &q,
            DatasetBytes::IrisCsv {
                csv: CSV,
                split: SPLIT
            }
        )
        .is_err()
    );
    q.train_samples = None;
    q.features = 5;
    assert!(
        decode_dataset(
            &q,
            DatasetBytes::IrisCsv {
                csv: CSV,
                split: SPLIT
            }
        )
        .is_err()
    );
    q.features = 0;
    assert_eq!(
        load_dataset(&q).unwrap_err().kind(),
        std::io::ErrorKind::InvalidData,
        "invalid declarations are refused before any pathname acquisition"
    );
}

#[test]
fn byte_dataset_path_wrappers_share_the_pure_decoders() {
    // Divergent wrapper preprocessing, source identities, or legacy NPY format break parity.
    let t = Temp::new();
    let q = request(
        DataSource::IrisCsv {
            csv: t.write("role-a", CSV),
            split: t.write("role-b", SPLIT),
        },
        3,
        4,
    );
    same(
        &load_dataset(&q).unwrap(),
        &decode_dataset(
            &q,
            DatasetBytes::IrisCsv {
                csv: CSV,
                split: SPLIT,
            },
        )
        .unwrap(),
    );
    let a = idx_arrays();
    for (name, bytes) in [
        Split::Train.files().0,
        Split::Train.files().1,
        Split::Test.files().0,
        Split::Test.files().1,
    ]
    .into_iter()
    .zip(&a)
    {
        t.write(name, bytes);
    }
    let q = request(
        DataSource::Idx {
            dir: t.0.clone(),
            resize: Some(1),
        },
        3,
        1,
    );
    same(
        &load_dataset(&q).unwrap(),
        &decode_dataset(&q, idx_bytes(&a)).unwrap(),
    );
    let train = data::load_mnist(&t.0, Split::Train, Some(1)).unwrap();
    assert_eq!(train.images, arr2(&[[0.; 4]]));
    let a = npy_arrays();
    let m = manifest(&a);
    for (name, b) in NPY_DATASET_FILES.into_iter().zip(&a) {
        t.write(name, b);
    }
    t.write("data_manifest.json", &m);
    let q = request(DataSource::Npy { dir: t.0.clone() }, 3, 2);
    same(
        &load_dataset(&q).unwrap(),
        &decode_dataset(&q, npy_bytes(&m, &a)).unwrap(),
    );
    let (tr, te) = data::load_npy_dataset(&t.0).unwrap();
    assert_eq!(tr.images, arr2(&[[1., 2.], [3., 4.]]));
    assert_eq!(te.labels, [2]);
    let mut legacy = String::new();
    for (name, b) in NPY_DATASET_FILES.into_iter().zip(&a) {
        writeln!(legacy, "{name}: {}", sha256_hex(b)).unwrap();
    }
    t.write("data_manifest.json", legacy.as_bytes());
    assert!(data::load_npy_dataset(&t.0).is_ok());
    assert!(load_dataset(&q).is_err());
}
