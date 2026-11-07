//! Class-aware dataset loading and declared captured-input preflight, without HE allocation.
//!
//! This module does not interpret the experiment matrix or derive model weight shapes. Its
//! callers declare source/profile identity and expected dimensions; D2 must bind those
//! declarations to approved model profiles and call preflight before allocating HE state.
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use ndarray::{Array2, Axis};
use serde_json::Value;

use crate::data::{self, Split, Tabular};
use crate::twin::rng;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

/// Training-only tabular scaling; supplementary WDBC z-score studies remain explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// Per-column training minimum and maximum.
    MinMax,
    /// Training mean and population standard deviation.
    Standardize,
}

/// The concrete files and preprocessing to consume (no dataset-name inference).
#[derive(Debug, Clone)]
pub enum DataSource {
    /// Iris CSV and a fixed, complete `train:` / `test:` partition.
    IrisCsv {
        /// CSV with four features and integer labels.
        csv: PathBuf,
        /// Ordered partition indices.
        split: PathBuf,
    },
    /// UCI WDBC, preserving the production pinned UCI hash.
    Wdbc {
        /// Original UCI CSV.
        path: PathBuf,
        /// Number of stratified test rows.
        test_size: usize,
        /// Explicit split seed; matrix primary is zero, supplementary plain may differ.
        split_seed: u8,
        /// Training-only scaling.
        scaling: Scaling,
    },
    /// Published IDX train/test files, optionally resized with Task51 preprocessing.
    Idx {
        /// Directory containing the four unpacked files.
        dir: PathBuf,
        /// Output side, or byte/255 inputs when absent.
        resize: Option<usize>,
    },
    /// Captured arrays consumed unchanged, with file-specific manifest verification.
    Npy {
        /// Directory containing four arrays and `data_manifest.json`.
        dir: PathBuf,
    },
}

/// Borrowed source content with closed roles matching [`DataSource`]. The request supplies
/// path identity labels; these slices supply both the decoded values and reported hashes.
/// This ordinary data view confers no acquisition, custody, resource or runner authority.
#[derive(Debug, Clone, Copy)]
pub enum DatasetBytes<'a> {
    /// Iris CSV and fixed ordered partition bytes.
    IrisCsv {
        /// CSV content.
        csv: &'a [u8],
        /// Partition content.
        split: &'a [u8],
    },
    /// Original pinned UCI CSV bytes.
    Wdbc {
        /// CSV content, subject to the mandatory UCI digest.
        csv: &'a [u8],
    },
    /// Four IDX roles; names are never inferred from content or supplied basenames.
    Idx {
        /// Training images.
        train_images: &'a [u8],
        /// Training labels.
        train_labels: &'a [u8],
        /// Held-out images.
        test_images: &'a [u8],
        /// Held-out labels.
        test_labels: &'a [u8],
    },
    /// Captured NPY roles, in [`data::NPY_DATASET_FILES`] order.
    Npy {
        /// Structured per-file manifest content.
        manifest: &'a [u8],
        /// Train images, train labels, test images, test labels.
        arrays: [&'a [u8]; 4],
    },
}

/// Explicit expected data contract; declaring a profile does not execute the matrix.
#[derive(Debug, Clone)]
pub struct DataRequest {
    /// Concrete primary or supplementary profile identity.
    pub profile_id: String,
    /// Dataset identity, independent from source file format.
    pub dataset_id: String,
    /// Real classes (output padding is a separate model property).
    pub classes: usize,
    /// Exact feature width after preprocessing.
    pub features: usize,
    /// Expected training rows when fixed by the caller's profile.
    pub train_samples: Option<usize>,
    /// Expected held-out rows when fixed by the caller's profile.
    pub test_samples: Option<usize>,
    /// Source and preprocessing policy.
    pub source: DataSource,
}

/// Shared numerical inputs for plaintext and future encrypted adapters.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Declared profile identity, never inferred from class count.
    pub profile_id: String,
    /// Declared dataset identity.
    pub dataset_id: String,
    /// Training features, in the original declared split order.
    pub train_x: Array2<f64>,
    /// Real training labels.
    pub train_y: Vec<u8>,
    /// Held-out features.
    pub test_x: Array2<f64>,
    /// Real held-out labels.
    pub test_y: Vec<u8>,
    /// Number of real classes.
    pub classes: usize,
    /// File paths and SHA256 digests, including partition/manifest when used.
    pub files: Vec<(String, String)>,
    /// Preprocessing description (legacy plain wording retained).
    pub prep: String,
    /// Split semantics; captured order is a separate seed input.
    pub split_identity: String,
    /// Source-row indices for the training split, before any training-order input.
    pub train_indices: Vec<usize>,
    /// Source-row indices for the held-out split.
    pub test_indices: Vec<usize>,
}

fn bytes_sha(path: &Path, bytes: &[u8]) -> (String, String) {
    (path.display().to_string(), data::sha256_hex(bytes))
}
fn utf8(bytes: &[u8]) -> io::Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| invalid("dataset text is not UTF-8"))
}
fn validate_arrays(x: &Array2<f64>, labels: &[u8], classes: usize) -> io::Result<()> {
    if !(2..=256).contains(&classes) || x.nrows() == 0 || x.ncols() == 0 {
        return Err(invalid(
            "nonempty features and 2..=256 real classes required",
        ));
    }
    if labels.len() != x.nrows() || labels.iter().any(|&c| c as usize >= classes) {
        return Err(invalid("label count or real-class label range mismatch"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(invalid("non-finite feature value"));
    }
    Ok(())
}
fn validate_dataset(d: &Dataset) -> io::Result<()> {
    validate_arrays(&d.train_x, &d.train_y, d.classes)?;
    validate_arrays(&d.test_x, &d.test_y, d.classes)?;
    if d.train_x.ncols() != d.test_x.ncols() {
        return Err(invalid("training and test feature widths differ"));
    }
    Ok(())
}
fn validate_partition(train: &[usize], test: &[usize], n: usize) -> io::Result<()> {
    if train.is_empty() || test.is_empty() || train.len().checked_add(test.len()) != Some(n) {
        return Err(invalid("split must be a nonempty train/test partition"));
    }
    let mut seen = vec![false; n];
    for &i in train.iter().chain(test) {
        if i >= n || seen[i] {
            return Err(invalid(
                "split contains out-of-range, duplicate or overlapping indices",
            ));
        }
        seen[i] = true;
    }
    Ok(())
}

/// Parses Iris CSV at a pure layer, rejecting malformed labels and non-finite features.
pub fn parse_iris(text: &str) -> io::Result<Tabular> {
    let mut values = Vec::new();
    let mut labels = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let cells: Vec<_> = line.split(',').map(str::trim).collect();
        if cells.len() != 5 {
            return Err(invalid("Iris row must contain four features and one label"));
        }
        for cell in &cells[..4] {
            values.push(
                cell.parse::<f64>()
                    .map_err(|_| invalid("invalid Iris feature"))?,
            );
        }
        labels.push(
            cells[4]
                .parse::<u8>()
                .map_err(|_| invalid("invalid Iris label"))?,
        );
    }
    let features = Array2::from_shape_vec((labels.len(), 4), values)
        .map_err(|_| invalid("invalid Iris shape"))?;
    validate_arrays(&features, &labels, 3)?;
    Ok(Tabular {
        features,
        ids: (0..labels.len() as u64).collect(),
        labels,
    })
}

/// Parses ordered partition indices; rejects missing/repeated keys and incomplete partitions.
pub fn parse_partition(text: &str, samples: usize) -> io::Result<(Vec<usize>, Vec<usize>)> {
    let (mut train, mut test) = (None, None);
    for line in text
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
    {
        let (key, values) = line
            .split_once(':')
            .ok_or_else(|| invalid("invalid split line"))?;
        let slot = match key {
            "train" => &mut train,
            "test" => &mut test,
            _ => return Err(invalid("unknown split key")),
        };
        if slot.is_some() {
            return Err(invalid("duplicate split key"));
        }
        *slot = Some(
            values
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<usize>()
                        .map_err(|_| invalid("invalid split index"))
                })
                .collect::<io::Result<Vec<_>>>()?,
        );
    }
    let train = train.ok_or_else(|| invalid("missing train split"))?;
    let test = test.ok_or_else(|| invalid("missing test split"))?;
    validate_partition(&train, &test, samples)?;
    Ok((train, test))
}

/// Selects a checked partition and fits scaling only on training rows.
///
/// Pure preprocessing permits synthetic fixtures without weakening production WDBC hashes.
pub fn tabular_dataset(
    profile_id: &str,
    dataset_id: &str,
    table: &Tabular,
    train: &[usize],
    test: &[usize],
    classes: usize,
    scaling: Scaling,
) -> io::Result<Dataset> {
    if profile_id.is_empty() || dataset_id.is_empty() {
        return Err(invalid("profile and dataset identity required"));
    }
    validate_arrays(&table.features, &table.labels, classes)?;
    validate_partition(train, test, table.labels.len())?;
    let mut train_x = table.features.select(Axis(0), train);
    let mut test_x = table.features.select(Axis(0), test);
    let prep = match scaling {
        Scaling::MinMax => {
            data::min_max_columns(&mut train_x, &mut test_x);
            "per-column min-max scaling with the training ranges (data::min_max_columns)"
        }
        Scaling::Standardize => {
            data::standardize(&mut train_x, &mut test_x);
            "standardization with the training mean and population std (data::standardize)"
        }
    };
    let d = Dataset {
        profile_id: profile_id.into(),
        dataset_id: dataset_id.into(),
        train_x,
        train_y: train.iter().map(|&i| table.labels[i]).collect(),
        test_x,
        test_y: test.iter().map(|&i| table.labels[i]).collect(),
        classes,
        files: vec![],
        prep: prep.into(),
        split_identity: "explicit ordered complete partition".into(),
        train_indices: train.to_vec(),
        test_indices: test.to_vec(),
    };
    validate_dataset(&d)?;
    Ok(d)
}

/// Pathname utility for standalone callers. Acquires each required file once, then uses
/// [`decode_dataset`] for all numerical processing and consumed-byte identities. Canonical
/// retained acquisition must call the pure decoder directly after its own resource checks.
pub fn load_dataset(q: &DataRequest) -> io::Result<Dataset> {
    validate_request(q)?;
    match &q.source {
        DataSource::IrisCsv { csv, split } => {
            let csv = fs::read(csv)?;
            let split = fs::read(split)?;
            decode_dataset(
                q,
                DatasetBytes::IrisCsv {
                    csv: &csv,
                    split: &split,
                },
            )
        }
        DataSource::Wdbc { path, .. } => {
            let csv = fs::read(path)?;
            decode_dataset(q, DatasetBytes::Wdbc { csv: &csv })
        }
        DataSource::Idx { dir, .. } => {
            let (tr_images, tr_labels) = Split::Train.files();
            let (te_images, te_labels) = Split::Test.files();
            let train_images = fs::read(dir.join(tr_images))?;
            let train_labels = fs::read(dir.join(tr_labels))?;
            let test_images = fs::read(dir.join(te_images))?;
            let test_labels = fs::read(dir.join(te_labels))?;
            decode_dataset(
                q,
                DatasetBytes::Idx {
                    train_images: &train_images,
                    train_labels: &train_labels,
                    test_images: &test_images,
                    test_labels: &test_labels,
                },
            )
        }
        DataSource::Npy { dir } => {
            let manifest = fs::read(dir.join("data_manifest.json"))?;
            let arrays = [
                fs::read(dir.join(data::NPY_DATASET_FILES[0]))?,
                fs::read(dir.join(data::NPY_DATASET_FILES[1]))?,
                fs::read(dir.join(data::NPY_DATASET_FILES[2]))?,
                fs::read(dir.join(data::NPY_DATASET_FILES[3]))?,
            ];
            decode_dataset(
                q,
                DatasetBytes::Npy {
                    manifest: &manifest,
                    arrays: arrays.each_ref().map(Vec::as_slice),
                },
            )
        }
    }
}

fn validate_request(q: &DataRequest) -> io::Result<()> {
    if q.profile_id.is_empty() || q.dataset_id.is_empty() || q.features == 0 {
        return Err(invalid(
            "profile/dataset identity and feature width required",
        ));
    }
    Ok(())
}

/// Pure dataset decoding, verification and preprocessing of the exact supplied slices.
///
/// Missing/malformed content or a variant differing from the request fails; no pathname
/// fallback exists. Request paths are identity labels, not evidence of source custody.
/// Captured NPY order, structured manifest defenses, and the fixed WDBC pin are preserved.
pub fn decode_dataset(q: &DataRequest, bytes: DatasetBytes<'_>) -> io::Result<Dataset> {
    validate_request(q)?;
    let mut d = match (&q.source, bytes) {
        (
            DataSource::IrisCsv { csv, split },
            DatasetBytes::IrisCsv {
                csv: csv_bytes,
                split: split_bytes,
            },
        ) => {
            let table = parse_iris(utf8(csv_bytes)?)?;
            let (tr, te) = parse_partition(utf8(split_bytes)?, table.labels.len())?;
            let mut d = tabular_dataset(
                &q.profile_id,
                &q.dataset_id,
                &table,
                &tr,
                &te,
                q.classes,
                Scaling::Standardize,
            )?;
            d.files = vec![bytes_sha(csv, csv_bytes), bytes_sha(split, split_bytes)];
            d.split_identity = "fixed ordered CSV partition; no study-seed redraw".into();
            d
        }
        (
            DataSource::Wdbc {
                path,
                test_size,
                split_seed,
                scaling,
            },
            DatasetBytes::Wdbc { csv },
        ) => {
            let table = data::wdbc_from_bytes(csv)?;
            if *test_size == 0 || *test_size >= table.labels.len() {
                return Err(invalid("WDBC split requires nonempty train/test"));
            }
            let (tr, te) =
                data::stratified_split(&table.labels, *test_size, &mut rng(*split_seed, 4));
            let mut d = tabular_dataset(
                &q.profile_id,
                &q.dataset_id,
                &table,
                &tr,
                &te,
                q.classes,
                *scaling,
            )?;
            d.files = vec![bytes_sha(path, csv)];
            d.prep = format!(
                "stratified split ({} train / {} test, ChaCha20 rng(split_seed = {split_seed}, 4), data::stratified_split), then {}",
                tr.len(),
                te.len(),
                d.prep
            );
            d.split_identity =
                format!("stratified source-row partition; split seed {split_seed}, stream4");
            d
        }
        source @ ((DataSource::Idx { .. }, DatasetBytes::Idx { .. })
        | (DataSource::Npy { .. }, DatasetBytes::Npy { .. })) => {
            let (tr, te, prep, files) = match source {
                (
                    DataSource::Idx { dir, resize },
                    DatasetBytes::Idx {
                        train_images,
                        train_labels,
                        test_images,
                        test_labels,
                    },
                ) => {
                    let tr = data::mnist_from_bytes(train_images, train_labels, None)?;
                    let te = data::mnist_from_bytes(test_images, test_labels, None)?;
                    validate_arrays(&tr.images, &tr.labels, q.classes)?;
                    validate_arrays(&te.images, &te.labels, q.classes)?;
                    if tr.rows == 0 || tr.cols == 0 || (tr.rows, tr.cols) != (te.rows, te.cols) {
                        return Err(invalid("IDX image shapes differ or are empty"));
                    }
                    let mut files = Vec::new();
                    for (split, images, labels) in [
                        (Split::Train, train_images, train_labels),
                        (Split::Test, test_images, test_labels),
                    ] {
                        let (a, b) = split.files();
                        files.push(bytes_sha(&dir.join(a), images));
                        files.push(bytes_sha(&dir.join(b), labels));
                    }
                    match resize {
                        Some(side) => {
                            if *side == 0 || side.checked_mul(*side) != Some(q.features) {
                                return Err(invalid("invalid resized feature width"));
                            }
                            // Reject constant training images before the legacy primitive's assert.
                            let mut resized =
                                data::resize_bicubic_aa(&tr.images, tr.rows, tr.cols, *side, *side);
                            let lo = resized.iter().copied().fold(f64::INFINITY, f64::min);
                            let hi = resized.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                                return Err(invalid("training resize has no finite scaling range"));
                            }
                            let prep = format!(
                                "{0}x{1} -> {side}x{side} antialiased bicubic (a = -0.5), then min-max scaling over the training set (data::reboot_preprocess)",
                                tr.rows, tr.cols
                            );
                            let mut heldout =
                                data::resize_bicubic_aa(&te.images, te.rows, te.cols, *side, *side);
                            data::min_max_scale(&mut resized, &mut heldout);
                            let a = data::Mnist {
                                images: resized,
                                labels: tr.labels,
                                rows: *side,
                                cols: *side,
                            };
                            let b = data::Mnist {
                                images: heldout,
                                labels: te.labels,
                                rows: *side,
                                cols: *side,
                            };
                            (a, b, prep, files)
                        }
                        None => (tr, te, "pixels byte / 255".into(), files),
                    }
                }
                (DataSource::Npy { dir }, DatasetBytes::Npy { manifest, arrays }) => {
                    let (a, b) = data::npy_dataset_from_bytes(
                        manifest,
                        arrays,
                        data::NpyManifestPolicy::Structured,
                    )?;
                    let mut files = data::NPY_DATASET_FILES
                        .into_iter()
                        .zip(arrays)
                        .map(|(name, bytes)| bytes_sha(&dir.join(name), bytes))
                        .collect::<Vec<_>>();
                    files.push(bytes_sha(&dir.join("data_manifest.json"), manifest));
                    (
                        a,
                        b,
                        "captured NPY arrays unchanged; source-manifest hashes verified; training order separate".into(),
                        files,
                    )
                }
                _ => unreachable!("matched image source and byte variants above"),
            };
            let train_indices = (0..tr.labels.len()).collect();
            let test_indices = (0..te.labels.len()).collect();
            Dataset {
                profile_id: q.profile_id.clone(),
                dataset_id: q.dataset_id.clone(),
                train_x: tr.images,
                train_y: tr.labels,
                test_x: te.images,
                test_y: te.labels,
                classes: q.classes,
                files,
                prep,
                split_identity: "source train/test row order unchanged; captured training indices applied separately".into(),
                train_indices,
                test_indices,
            }
        }
        _ => {
            return Err(invalid(
                "dataset byte variant differs from requested source",
            ));
        }
    };
    validate_dataset(&d)?;
    if d.train_x.ncols() != q.features
        || q.train_samples.is_some_and(|n| n != d.train_x.nrows())
        || q.test_samples.is_some_and(|n| n != d.test_x.nrows())
    {
        return Err(invalid("dataset shape differs from declared profile"));
    }
    // Keep declarations explicit for adapters; no inferred model/profile remapping.
    d.profile_id.clone_from(&q.profile_id);
    d.dataset_id.clone_from(&q.dataset_id);
    Ok(d)
}

/// Checked one-hot targets, zero in every padded output channel.
pub fn padded_targets(
    labels: &[u8],
    classes: usize,
    output_width: usize,
) -> io::Result<Array2<f64>> {
    if !(2..=256).contains(&classes)
        || output_width < classes
        || labels.iter().any(|&c| c as usize >= classes)
    {
        return Err(invalid("invalid classes, output padding or target label"));
    }
    Ok(data::one_hot(labels, output_width))
}

/// Accuracy argmax over real classes only; rejects mismatched or non-finite predictions.
pub fn class_accuracy(outputs: &Array2<f64>, labels: &[u8], classes: usize) -> io::Result<f64> {
    if outputs.ncols() < classes {
        return Err(invalid("prediction width below real classes"));
    }
    validate_arrays(outputs, labels, classes)?;
    Ok(crate::twin::accuracy_over(outputs, labels, classes))
}

/// A required content digest for one declared input file.
#[derive(Debug, Clone)]
pub struct LockedFile {
    /// Exact file to read.
    pub path: PathBuf,
    /// Required lowercase or uppercase SHA256 hex; never optional.
    pub sha256: String,
}
impl LockedFile {
    fn read(&self) -> io::Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        if self.sha256.len() != 64
            || !self.sha256.bytes().all(|c| c.is_ascii_hexdigit())
            || !data::sha256_hex(&bytes).eq_ignore_ascii_case(&self.sha256)
        {
            return Err(invalid(format!(
                "{}: locked sha256 mismatch",
                self.path.display()
            )));
        }
        Ok(bytes)
    }
}
/// An initialized NPY array with exact logical shape, supplied by the model adapter.
#[derive(Debug, Clone)]
pub struct ArrayInput {
    /// Required file and digest.
    pub file: LockedFile,
    /// Exact rows/columns; 1D reader arrays are `(1,n)`.
    pub shape: (usize, usize),
}
/// Explicit per-pass coverage, never guessed from a file length or drop-last flag.
#[derive(Debug, Clone, Copy)]
pub enum OrderCoverage {
    /// Every population row appears exactly once per pass.
    FullPermutation,
    /// Exactly this many distinct population rows per pass (a declared dropped tail).
    UniqueSubset {
        /// Required processed rows per pass, at most the population size.
        samples_per_pass: usize,
    },
}
/// Captured text indices, with explicit pass boundaries and coverage policy.
#[derive(Debug, Clone)]
pub struct OrderInput {
    /// Locked indices: whitespace/comma tokens, optional comment lines and TSV metadata.
    pub file: LockedFile,
    /// Source training population size.
    pub population: usize,
    /// Exact number of captured passes (do not repeat a flattened file).
    pub passes: usize,
    /// Required per-pass coverage.
    pub coverage: OrderCoverage,
}
/// Schedule array in a locked JSON source, selected without seed fallback.
#[derive(Debug, Clone)]
pub struct ScheduleInput {
    /// Locked JSON file.
    pub file: LockedFile,
    /// Exact seed/model pointer to a numeric array; derived by the caller.
    pub json_pointer: String,
    /// Required source schedule length; resampling is a downstream model responsibility.
    pub steps: usize,
}
/// All declared captured inputs for one explicit model/order seed.
#[derive(Debug, Clone)]
pub struct SeedFiles {
    /// Exact initialized arrays and shapes; must not be empty.
    pub initialization: Vec<ArrayInput>,
    /// Captured order when required by the caller's profile.
    pub order: Option<OrderInput>,
    /// Captured schedule when required by the caller's profile.
    pub schedule: Option<ScheduleInput>,
}
/// Seed-keyed input contract; absent seeds are missing inputs, never encryption repetitions.
#[derive(Debug, Clone)]
pub struct CapturedInputs {
    /// Concrete profile identity.
    pub profile_id: String,
    /// Per-seed manifests; callers must derive these from the approved profile sources.
    pub per_seed: BTreeMap<u8, SeedFiles>,
}
/// Verified source values and digests; no training or HE state is created.
#[derive(Debug)]
pub struct VerifiedSeed {
    /// Declared profile identity.
    pub profile_id: String,
    /// Selected model/order seed.
    pub seed: u8,
    /// Initialized arrays in declaration order.
    pub initialization: Vec<Array2<f64>>,
    /// Flat captured order, preserving all declared passes.
    pub order: Option<Vec<usize>>,
    /// Captured source schedule, before downstream sample-aligned resampling.
    pub schedule: Option<Vec<f64>>,
    /// Verified file paths and content digests.
    pub files: Vec<(String, String)>,
}

/// Verifies the selected seed's declared hashes, dimensions, order and schedule.
///
/// This generic layer does not decide which arrays/inputs a model requires. D2 must derive
/// that complete contract from the profile, then enforce preflight before HE allocation.
pub fn preflight_seed(contract: &CapturedInputs, seed: u8) -> io::Result<VerifiedSeed> {
    let inputs = contract.per_seed.get(&seed).ok_or_else(|| {
        invalid(format!(
            "missing captured inputs for {} seed {seed}",
            contract.profile_id
        ))
    })?;
    if contract.profile_id.is_empty() || inputs.initialization.is_empty() {
        return Err(invalid("profile identity and initialized arrays required"));
    }
    let mut files = Vec::new();
    let mut read = |f: &LockedFile| -> io::Result<Vec<u8>> {
        let bytes = f.read()?;
        files.push((f.path.display().to_string(), data::sha256_hex(&bytes)));
        Ok(bytes)
    };
    let mut initialization = Vec::new();
    for item in &inputs.initialization {
        let a = data::parse_npy_f64(&read(&item.file)?)?;
        if a.dim() != item.shape || a.is_empty() || a.iter().any(|v| !v.is_finite()) {
            return Err(invalid("initialized array shape or finite-value mismatch"));
        }
        initialization.push(a);
    }
    let checked = verify_order_schedule(inputs.order.as_ref(), inputs.schedule.as_ref())?;
    files.extend(checked.2);
    let (order, schedule) = (checked.0, checked.1);
    Ok(VerifiedSeed {
        profile_id: contract.profile_id.clone(),
        seed,
        initialization,
        order,
        schedule,
        files,
    })
}

/// Verifies exactly required captured order and schedule for a locally initialized model.
///
/// No initialized-array dependency is introduced; the caller must bind model/seed paths.
pub fn preflight_order_schedule(
    profile_id: &str,
    seed: u8,
    order: &OrderInput,
    schedule: &ScheduleInput,
) -> io::Result<VerifiedSeed> {
    if profile_id.is_empty() {
        return Err(invalid("profile identity required"));
    }
    let (order, schedule, files) = verify_order_schedule(Some(order), Some(schedule))?;
    Ok(VerifiedSeed {
        profile_id: profile_id.into(),
        seed,
        initialization: Vec::new(),
        order,
        schedule,
        files,
    })
}

type CheckedOrderSchedule = (Option<Vec<usize>>, Option<Vec<f64>>, Vec<(String, String)>);
fn verify_order_schedule(
    order: Option<&OrderInput>,
    schedule: Option<&ScheduleInput>,
) -> io::Result<CheckedOrderSchedule> {
    let mut files = Vec::new();
    let mut read = |f: &LockedFile| -> io::Result<Vec<u8>> {
        let bytes = f.read()?;
        files.push((f.path.display().to_string(), data::sha256_hex(&bytes)));
        Ok(bytes)
    };
    let order = order
        .map(|o| -> io::Result<Vec<usize>> {
            let n = match o.coverage {
                OrderCoverage::FullPermutation => o.population,
                OrderCoverage::UniqueSubset { samples_per_pass } => samples_per_pass,
            };
            if n == 0 || n > o.population || o.passes == 0 {
                return Err(invalid("invalid declared order coverage"));
            }
            let bytes = read(&o.file)?;
            let text = std::str::from_utf8(&bytes).map_err(|_| invalid("order is not UTF8"))?;
            let indices = text
                .lines()
                .filter(|line| !line.trim_start().starts_with('#'))
                .map(|line| line.rsplit_once('\t').map_or(line, |(_, indices)| indices))
                .flat_map(|line| line.split(|c: char| c.is_whitespace() || c == ','))
                .filter(|token| !token.is_empty())
                .map(|s| {
                    s.parse::<usize>()
                        .map_err(|_| invalid("invalid order index"))
                })
                .collect::<io::Result<Vec<_>>>()?;
            if n.checked_mul(o.passes) != Some(indices.len()) {
                return Err(invalid(
                    "captured order length differs from declared passes/coverage",
                ));
            }
            for pass in indices.chunks_exact(n) {
                let mut seen = vec![false; o.population];
                for &i in pass {
                    if i >= o.population || seen[i] {
                        return Err(invalid(
                            "captured pass has out-of-range or duplicate indices",
                        ));
                    }
                    seen[i] = true;
                }
            }
            Ok(indices)
        })
        .transpose()?;
    let schedule = schedule
        .map(|s| -> io::Result<Vec<f64>> {
            let json: Value = serde_json::from_slice(&read(&s.file)?)
                .map_err(|e| invalid(format!("invalid schedule JSON: {e}")))?;
            let values = json
                .pointer(&s.json_pointer)
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("missing selected schedule array"))?;
            if s.steps == 0 || values.len() != s.steps {
                return Err(invalid("schedule length differs from declared steps"));
            }
            values
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|x| x.is_finite() && *x >= 0.)
                        .ok_or_else(|| invalid("invalid schedule coefficient"))
                })
                .collect()
        })
        .transpose()?;
    Ok((order, schedule, files))
}
