//! Pure resolution of reviewed study profiles, locked inputs and actual update boundaries.
//!
//! No HE objects, datasets, initialized local arrays, training or processes are created.
use crate::{
    data::sha256_hex,
    profile_data::{
        self, ArrayInput, CapturedInputs, DataRequest, DataSource, Dataset, LockedFile,
        OrderCoverage, OrderInput, Scaling, ScheduleInput, SeedFiles, VerifiedSeed,
    },
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fmt, fs, io,
    path::{Path, PathBuf},
};

/// SHA256 of the independently reviewed D0 registry at commit 8eabcdd.
pub const APPROVED_REGISTRY_SHA256: &str =
    "164dbc67fddb87bed814bfa9bcbad1f97508a4a2de265b7a8ca899d8fab1b4cc";
/// SHA256 of the reviewed portable input lock generated from exact D0 Git blobs.
pub const APPROVED_INPUT_LOCK_SHA256: &str =
    "cc04f6f7451b0ec99416e10d14c2bf6b2eaf5f703c8de43d500cac3e40da126b";
const REGISTRY: &str = "crates/ckks-train-bench/configs/experiment_matrix.json";
const LOCKS: &str = "crates/ckks-train-bench/configs/experiment_input_locks.json";
const IRIS_DATA_SHA256: &str = "874d28c2148c94ac8bed1b98ab1d93d27a473f72b2a712b06833f49b3259e54d";
const IRIS_SPLIT_SHA256: &str = "9da646715cfa3f2bd0e800680ff987fe066a4e0403cc2acf25ca90b8cf02cf12";
const FASHION_SOURCE_SHA256: &str =
    "1ecc5cfa25ea1d16d93c8df533e24ead408cf35fdee2eee9a18527a11753c3c0";
const LOCK_COMMIT: &str = "8eabcdd4cd46b3144c5a222ddb0ce533ee627b49";

/// Fail-closed profile/input errors, before any HE allocation.
#[derive(Debug)]
pub enum ProfileError {
    /// The selected ID is not a registered library profile.
    UnsupportedProfile(String),
    /// Statistical seeds are exactly 0 through 9.
    UnsupportedSeed(u8),
    /// An export or integrity-lock entry is absent; no fallback is permitted.
    MissingInput {
        /// Selected profile.
        profile: String,
        /// Selected seed.
        seed: u8,
        /// Missing source.
        input: String,
    },
    /// Conflicting graph, source, path, seed, shape, orientation or digest.
    Conflict(String),
    /// Malformed declared source metadata.
    Malformed(String),
    /// Source read/parse/preflight error.
    Io(io::Error),
}
impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ProfileError {}
impl From<io::Error> for ProfileError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
type Result<T> = std::result::Result<T, ProfileError>;
fn malformed(s: impl Into<String>) -> ProfileError {
    ProfileError::Malformed(s.into())
}
fn conflict(s: impl Into<String>) -> ProfileError {
    ProfileError::Conflict(s.into())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| malformed(format!("missing string {key}")))
}
fn number(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| malformed(format!("missing positive integer {key}")))
}
fn real(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .filter(|n| n.is_finite() && *n >= 0.)
        .ok_or_else(|| malformed(format!("invalid nonnegative coefficient {key}")))
}
fn boolean(v: &Value, key: &str) -> Result<bool> {
    v.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| malformed(format!("missing bool {key}")))
}
fn array<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| malformed(format!("missing array {key}")))
}
fn json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes).map_err(|e| malformed(e.to_string()))
}
fn read_locked(f: &LockedFile) -> Result<Vec<u8>> {
    let b = fs::read(&f.path)?;
    if sha256_hex(&b) != f.sha256 {
        return Err(conflict(format!(
            "{}: locked SHA256 mismatch",
            f.path.display()
        )));
    }
    Ok(b)
}
fn source(root: &Path, path: &str) -> Result<LockedFile> {
    let p = root.join(path);
    Ok(LockedFile {
        sha256: sha256_hex(&fs::read(&p)?),
        path: p,
    })
}
fn eq<T: PartialEq + fmt::Debug>(a: T, b: T, what: &str) -> Result<()> {
    if a != b {
        return Err(conflict(format!("{what}: {a:?} != {b:?}")));
    }
    Ok(())
}

/// Registered learning graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Exact/adapted 2020 MLP.
    Mlp2020,
    /// Two-layer eMLP.
    Emlp1,
    /// One local block plus output stack.
    Emlp2,
    /// Two local blocks plus output stack.
    Emlp3,
}
/// Dense activation in the registered mathematical graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// z squared.
    Square,
    /// z squared plus z.
    SquarePlusLinear,
    /// No activation.
    Identity,
}
/// Independent loss/stage role, in forward parameter order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRole {
    /// Ordinary two-layer hidden layer.
    Hidden,
    /// Ordinary/output final layer.
    Output,
    /// Local block representation, no gradient below stage.
    LocalForward {
        /// Zero-based block.
        block: usize,
    },
    /// Independent local classification head.
    LocalHead {
        /// Zero-based block.
        block: usize,
    },
    /// Separate output-stack hidden layer, no gradient below stage.
    OutputHidden,
}
/// Logical dense layer, independent of encrypted padding/layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    /// Source logical layer name.
    pub name: String,
    /// Input feature width.
    pub input: usize,
    /// Output feature width.
    pub output: usize,
    /// Activation applied to this layer's output.
    pub activation: Activation,
    /// Trainable bias exists.
    pub bias: bool,
    /// Independent stage role.
    pub role: LayerRole,
}
/// Loss/gradient convention (padded RSS outputs remain trained).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    /// Mean over real classes, delta multiplier 2/classes.
    MeanOutputMse,
    /// RSS over all logical padded outputs, delta multiplier 1.
    RssDelta,
}
/// Learning-state optimizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Optimizer {
    /// Ordinary SGD.
    Sgd,
    /// Nesterov, buffer v'=mu*v+g; step g+mu*v'.
    Nesterov {
        /// Momentum coefficient.
        momentum: f64,
    },
}
/// Rate conversion applied to a batch-mean gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateUnits {
    /// Already in batch-mean units.
    Mean,
    /// Source sum-gradient rate times actual valid batch size.
    SumTimesValid,
    /// Adapted2020 fixed nominal logical batch convention.
    SumTimesNominal,
}
/// Captured matrix storage orientation; logical library shape is always out by in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Captured ReBoot arrays store in by out and require transpose.
    InputByOutput,
}
/// One exact captured parameter binding, never inferred from class count alone.
#[derive(Debug, Clone)]
pub struct InitializedArray {
    /// Exact layer name.
    pub name: String,
    /// Logical (output,input) shape.
    pub logical_shape: (usize, usize),
    /// Captured physical (input,output) shape.
    pub source_shape: (usize, usize),
    /// Explicit conversion required.
    pub orientation: Orientation,
    /// Required source file hash.
    pub file: LockedFile,
}
/// RNG seed construction; generating values is a later adapter duty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RngConvention {
    /// ChaCha20 seed is little-endian u64 in bytes0..8, other bytes0;
    /// first master.fork initializes, second master.fork shuffles; preserve existing controls.
    LegacyIrisMasterFork,
    /// ChaCha20 key bytes [role,seed,seed,...]; role1 keys,2 init,
    /// 3 shuffle,4 encryption,5 refresh. Persistent per-role streams.
    DomainSeparated,
}
/// Deterministic local weight/bias distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Distribution {
    /// Box-Muller normal, cosine draw only, sigma scaling.
    Normal {
        /// Standard deviation.
        sigma: f64,
    },
    /// U[-sqrt(1/(in+out)),sqrt(1/(in+out))), 53-bit uniforms.
    XavierUniform,
}
/// A recipe, explicitly not materialized initialized arrays.
#[derive(Debug, Clone)]
pub struct GenerationRecipe {
    /// Statistical model/order seed.
    pub seed: u8,
    /// Exact stream key convention.
    pub rng: RngConvention,
    /// Distribution including exact scale.
    pub distribution: Distribution,
    /// Forward layer order; each weight is out-by-in row-major,
    /// followed by output-width bias draws. Disabled Xavier biases are drawn then discarded,
    /// matching twin::init_net. Normal draws biases when enabled.
    pub layers: Vec<Layer>,
}
/// Initialization availability, independent of metadata resolution.
#[derive(Debug, Clone)]
pub enum Initialization {
    /// Exact captured values, requiring preflight.
    Captured(Vec<InitializedArray>),
    /// No arrays exist yet; deterministic recipe only.
    Recipe(GenerationRecipe),
    /// Registered captured seed has no export.
    MissingCaptured,
}
/// Explicit training-order policy.
#[derive(Debug, Clone)]
pub enum OrderPolicy {
    /// A single already-flattened multi-pass file.
    Captured,
    /// Persistent shuffle stream; new domains use a fresh 0..population vector
    /// each epoch. Legacy Iris shuffles the previous ordered source-index vector in-place.
    PerEpochRecipe {
        /// Seed convention, shares the declared recipe's seed.
        rng: RngConvention,
        /// Retained rows after the epoch permutation.
        coverage: OrderCoverage,
    },
}
/// Captured order path and exact pass coverage.
#[derive(Debug, Clone)]
pub struct CapturedOrder {
    /// Exact selected model/seed path.
    pub path: PathBuf,
    /// Required snapshot digest; absent only when the selected seed export is missing.
    pub sha256: Option<String>,
    /// Population before captured shuffle.
    pub population: usize,
    /// Number of passes already in this file.
    pub passes: usize,
    /// Every released pass is a complete permutation.
    pub coverage: OrderCoverage,
}
/// Locked schedule source; source JSON is verified before paths are derived.
#[derive(Debug, Clone)]
pub struct ScheduleBinding {
    /// Source JSON digest/path.
    pub source: LockedFile,
    /// Exact selected model/seed pointer.
    pub json_pointer: String,
    /// Unresampled source length.
    pub steps: usize,
    /// Source logical batch, independent of actual study batch.
    pub reference_batch: usize,
}
/// Input provenance, preserving source rather than execution claims.
#[derive(Debug, Clone)]
pub struct InputIdentity {
    /// Independently reviewed registry bytes.
    pub registry: LockedFile,
    /// Portable snapshot integrity lock.
    pub integrity_lock: LockedFile,
    /// Consumed config/value sources, including local recipe code.
    pub sources: Vec<LockedFile>,
    /// Exact preprocessing declaration.
    pub preprocessing: String,
    /// Exact fixed split declaration.
    pub split_policy: String,
    /// Expected dataset file hashes; verified only by combined preparation.
    pub dataset_files: Vec<LockedFile>,
}
/// Fully typed metadata for one explicit profile and statistical seed.
#[derive(Debug, Clone)]
pub struct ProfileMetadata {
    /// Registry ID.
    pub id: String,
    /// Selected study seed.
    pub seed: u8,
    /// Dataset key.
    pub dataset: String,
    /// Selected graph.
    pub model: Model,
    /// Real classes (accuracy domain).
    pub classes: usize,
    /// Logical padded trainable output width.
    pub output_width: usize,
    /// Ordered layers and independent local heads.
    pub layers: Vec<Layer>,
    /// Loss convention.
    pub loss: Loss,
    /// Optimizer convention.
    pub optimizer: Optimizer,
    /// Decay coefficient, registered zero.
    pub weight_decay: f64,
    /// Raw rate value (schedule overrides per update).
    pub rate: f64,
    /// Raw units description from registry.
    pub rate_description: String,
    /// Mean-gradient conversion.
    pub rate_units: RateUnits,
    /// Nominal logical batch.
    pub batch: usize,
    /// Logical passes/epochs.
    pub logical_epochs: usize,
    /// Drop tail within each local epoch, or flat captured stream.
    pub drop_last: bool,
    /// Captured values, recipe, or explicit missing state.
    pub initialization: Initialization,
    /// Explicit order semantics.
    pub order: OrderPolicy,
    /// Captured order if applicable.
    pub captured_order: Option<CapturedOrder>,
    /// Selected-seed schedule if applicable.
    pub schedule: Option<ScheduleBinding>,
    /// Source identities.
    pub identity: InputIdentity,
}

/// Loaded approved registry plus portable captured-file lock; no execution state.
pub struct StudyRegistry {
    root: PathBuf,
    value: Value,
    registry: LockedFile,
    lock: LockedFile,
    locked: BTreeMap<String, String>,
}
impl StudyRegistry {
    /// Loads the exact approved registry and portable lock, with no Git/runtime repair.
    pub fn load(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let registry = LockedFile {
            path: root.join(REGISTRY),
            sha256: APPROVED_REGISTRY_SHA256.into(),
        };
        let value = json(&read_locked(&registry)?)?;
        eq(value["schema_version"].as_u64(), Some(1), "registry schema")?;
        eq(
            value["artifact_type"].as_str(),
            Some("design_registration"),
            "registry artifact",
        )?;
        let lock = LockedFile {
            path: root.join(LOCKS),
            sha256: APPROVED_INPUT_LOCK_SHA256.into(),
        };
        let lv = json(&read_locked(&lock)?)?;
        eq(
            string(&lv, "artifact_type")?,
            "input_integrity_lock",
            "lock type",
        )?;
        eq(
            string(&lv, "source_commit")?,
            LOCK_COMMIT,
            "lock source commit",
        )?;
        let mut locked = BTreeMap::new();
        for f in array(&lv, "files")? {
            let path = string(f, "path")?;
            let sha = string(f, "sha256")?;
            if locked.insert(path.into(), sha.into()).is_some() {
                return Err(malformed("duplicate integrity-lock path"));
            }
        }
        Ok(Self {
            root,
            value,
            registry,
            lock,
            locked,
        })
    }
    /// All registered library profile IDs; this list does not authorize execution.
    pub fn profile_ids(&self) -> Vec<String> {
        self.value["profiles"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|p| p["id"].as_str().map(str::to_owned))
            .collect()
    }
    fn locked_file(&self, path: &str, profile: &str, seed: u8) -> Result<LockedFile> {
        let sha = self
            .locked
            .get(path)
            .ok_or_else(|| ProfileError::MissingInput {
                profile: profile.into(),
                seed,
                input: format!("integrity lock {path}"),
            })?;
        Ok(LockedFile {
            path: self.root.join(path),
            sha256: sha.clone(),
        })
    }
    /// Resolves metadata even when the selected captured block seed export is absent.
    pub fn resolve(&self, id: &str, seed: u8) -> Result<ResolvedProfile> {
        if seed > 9 {
            return Err(ProfileError::UnsupportedSeed(seed));
        }
        let profiles = array(&self.value, "profiles")?;
        let p = profiles
            .iter()
            .find(|p| p["id"].as_str() == Some(id))
            .ok_or_else(|| ProfileError::UnsupportedProfile(id.into()))?;
        let dataset = string(p, "dataset")?;
        let model_id = string(p, "model")?;
        let d = &self.value["datasets"][dataset];
        let model = &self.value["models"][model_id];
        let classes = number(d, "classes")?;
        let features = number(d, "features")?;
        let output_width = number(p, "output_width")?;
        eq(number(p, "classes")?, classes, "real classes")?;
        let model_kind = match model_id {
            "mlp2020" => Model::Mlp2020,
            "emlp1" => Model::Emlp1,
            "emlp2" => Model::Emlp2,
            "emlp3" => Model::Emlp3,
            _ => return Err(malformed("unsupported model graph")),
        };
        eq(
            output_width,
            if model_kind == Model::Mlp2020 {
                classes
            } else {
                classes.next_power_of_two()
            },
            "output padding",
        )?;
        eq(p["bias"].as_bool(), model["bias"].as_bool(), "bias")?;
        eq(p["loss"].as_str(), model["loss_id"].as_str(), "loss")?;
        eq(
            p["optimizer"].as_str(),
            model["optimizer"].as_str(),
            "optimizer",
        )?;
        let layers = graph(model, p, features, output_width)?;
        let mut sources = Vec::new();
        // Source paths come from the registry, never from guessed upstream defaults.
        let mut refs = Vec::new();
        for owner in [d, model, p] {
            for r in array(owner, "source_refs")? {
                let r = r
                    .as_str()
                    .ok_or_else(|| malformed("invalid source reference"))?;
                if !refs.contains(&r) {
                    refs.push(r);
                }
            }
        }
        for r in refs {
            let path = string(&self.value["sources"][r], "path")?;
            if !path.ends_with(".json") && r != "iris" {
                continue;
            }
            let f = if self.locked.contains_key(path) {
                self.locked_file(path, id, seed)?
            } else {
                source(&self.root, path)?
            };
            read_locked(&f)?;
            sources.push(f);
        }
        check_source_graph(
            &self.root,
            &self.value,
            model_kind,
            dataset,
            &layers,
            p,
            &mut sources,
        )?;
        for path in [
            "crates/ckks-train-bench/src/twin.rs",
            "crates/ckks-train-core/src/sampler.rs",
        ] {
            sources.push(source(&self.root, path)?);
        }
        let train = number(d, "train_samples")?;
        let test = number(d, "test_samples")?;
        let batch = number(p, "batch")?;
        let epochs = number(p, "logical_epochs")?;
        let drop_last = boolean(p, "drop_last")?;
        let data_path = self.root.join(string(d, "data_path")?);
        let data_source = match dataset {
            "iris" => DataSource::IrisCsv {
                csv: data_path,
                split: self.root.join(string(d, "split_path")?),
            },
            "breast_cancer" => DataSource::Wdbc {
                path: data_path,
                test_size: test,
                split_seed: 0,
                scaling: Scaling::MinMax,
            },
            "mnist_released" => DataSource::Npy { dir: data_path },
            "fashion_mnist" => DataSource::Idx {
                dir: data_path,
                resize: Some(features.isqrt()),
            },
            _ => return Err(malformed("unsupported dataset adapter")),
        };
        let data_request = DataRequest {
            profile_id: id.into(),
            dataset_id: dataset.into(),
            classes,
            features,
            train_samples: Some(train),
            test_samples: Some(test),
            source: data_source,
        };
        let init = &p["initialization"];
        let legacy = init["existing_iris_rng_compatibility"].as_bool() == Some(true);
        let rng = if legacy {
            RngConvention::LegacyIrisMasterFork
        } else {
            RngConvention::DomainSeparated
        };
        let mut initialization = match string(init, "kind")? {
            "local" => Initialization::Recipe(GenerationRecipe {
                seed,
                rng,
                distribution: match string(&init["distribution"], "kind")? {
                    "normal" => Distribution::Normal {
                        sigma: real(&init["distribution"], "sigma")?,
                    },
                    "xavier_uniform" => Distribution::XavierUniform,
                    _ => return Err(malformed("unsupported initialization")),
                },
                layers: layers.clone(),
            }),
            "captured_weights" => Initialization::MissingCaptured,
            _ => return Err(malformed("unsupported init kind")),
        };
        let mut schedule = None;
        let mut captured_order = None;
        let mut seed_files = None;
        let order = match string(&p["order"], "kind")? {
            "local_per_epoch" => OrderPolicy::PerEpochRecipe {
                rng,
                coverage: if drop_last && train % batch != 0 {
                    OrderCoverage::UniqueSubset {
                        samples_per_pass: train / batch * batch,
                    }
                } else {
                    OrderCoverage::FullPermutation
                },
            },
            "captured_flattened_10_pass" => {
                let source_ref = string(&p["schedule"], "source_ref")?;
                eq(
                    string(&p["order"], "source_ref")?,
                    source_ref,
                    "order/schedule source",
                )?;
                let path = string(&self.value["sources"][source_ref], "path")?;
                let f = self.locked_file(path, id, seed)?;
                let captured = json(&read_locked(&f)?)?;
                let captured_model = if model_kind == Model::Mlp2020 || model_kind == Model::Emlp1 {
                    "emlp1"
                } else {
                    model_id
                };
                let base = if source_ref == "blocks_released" {
                    format!("/models/{captured_model}")
                } else {
                    String::new()
                };
                let seed_pointer = format!("{base}/per_seed/{seed}");
                let pointer =
                    string(&p["schedule"], "json_pointer")?.replace("{seed}", &seed.to_string());
                eq(
                    pointer.clone(),
                    format!("{seed_pointer}/lr_per_step"),
                    "schedule selected seed/model",
                )?;
                let config = if base.is_empty() {
                    &captured["config_values"]
                } else {
                    &captured["models"][captured_model]["config"]
                };
                let reference_batch = number(config, "batch_size")?;
                eq(boolean(config, "bias")?, false, "captured bias")?;
                eq(real(config, "momentum")?, 0.9, "captured momentum")?;
                eq(
                    real(config, if base.is_empty() { "lr_base" } else { "lr" })?,
                    real(&p["learning_rate"], "value")?,
                    "captured base rate",
                )?;
                if base.is_empty() {
                    let hidden = array(config, "hidden")?;
                    eq(
                        hidden.as_slice(),
                        array(&self.value["models"]["emlp1"]["graph"], "hidden_widths")?.as_slice(),
                        "captured eMLP1 hidden",
                    )?;
                }
                eq(number(config, "epochs")?, epochs, "captured logical passes")?;
                eq(number(config, "subsample_K")?, train, "captured population")?;
                if !base.is_empty() {
                    validate_structure(&captured["models"][captured_model]["structure"], &layers)?;
                }
                let steps = train / reference_batch * epochs;
                schedule = Some(ScheduleBinding {
                    source: f.clone(),
                    json_pointer: pointer.clone(),
                    steps,
                    reference_batch,
                });
                if let Some(s) = captured.pointer(&seed_pointer) {
                    validate_seed_argv(s, seed, captured_model)?;
                    let dir = f.path.parent().ok_or_else(|| malformed("source parent"))?;
                    let order_name = string(s, "indices_file")?;
                    let order_path = dir.join(order_name);
                    let indices_name = if seed == 0 {
                        "train_indices.txt".into()
                    } else {
                        format!("train_indices_seed{seed}.txt")
                    };
                    let expected = string(&p["order"], "path_template")?
                        .replace("{seed}", &seed.to_string())
                        .replace("{indices_name}", &indices_name);
                    eq(
                        order_path.clone(),
                        self.root.join(expected),
                        "selected order path",
                    )?;
                    let rel = order_path
                        .strip_prefix(&self.root)
                        .map_err(|_| conflict("order path outside repository"))?
                        .to_str()
                        .ok_or_else(|| malformed("order path encoding"))?;
                    let order_file = self.locked_file(rel, id, seed)?;
                    let o = OrderInput {
                        file: order_file,
                        population: train,
                        passes: epochs,
                        coverage: OrderCoverage::FullPermutation,
                    };
                    captured_order = Some(CapturedOrder {
                        path: order_path,
                        sha256: Some(o.file.sha256.clone()),
                        population: train,
                        passes: epochs,
                        coverage: OrderCoverage::FullPermutation,
                    });
                    let mut arrays = Vec::new();
                    if matches!(initialization, Initialization::MissingCaptured) {
                        eq(string(init, "source_ref")?, source_ref, "init/order source")?;
                        arrays = bind_arrays(
                            s,
                            &layers,
                            dir,
                            &self.root.join(
                                string(init, "path_template")?.replace("{seed}", &seed.to_string()),
                            ),
                        )?;
                        initialization = Initialization::Captured(arrays.clone());
                    }
                    seed_files = Some(SeedFiles {
                        initialization: arrays
                            .iter()
                            .map(|a| ArrayInput {
                                file: a.file.clone(),
                                shape: a.source_shape,
                            })
                            .collect(),
                        order: Some(o),
                        schedule: Some(ScheduleInput {
                            file: f.clone(),
                            json_pointer: pointer,
                            steps,
                        }),
                    });
                }
                if captured_order.is_none() {
                    // Missing exports still resolve their required path from the registered
                    // model/seed template. No digest or available export is invented.
                    let path =
                        string(&p["order"], "path_template")?.replace("{seed}", &seed.to_string());
                    if path.contains('{') {
                        return Err(malformed("unresolved missing order template"));
                    }
                    captured_order = Some(CapturedOrder {
                        path: self.root.join(path),
                        sha256: None,
                        population: train,
                        passes: epochs,
                        coverage: OrderCoverage::FullPermutation,
                    });
                }
                if !sources.iter().any(|s| s.path == f.path) {
                    sources.push(f);
                }
                OrderPolicy::Captured
            }
            _ => return Err(malformed("unsupported order kind")),
        };
        let loss = if model_kind == Model::Mlp2020 {
            Loss::MeanOutputMse
        } else {
            Loss::RssDelta
        };
        let rate_units = if model_kind != Model::Mlp2020 {
            RateUnits::SumTimesValid
        } else if dataset == "iris" {
            RateUnits::Mean
        } else {
            RateUnits::SumTimesNominal
        };
        let optimizer = if model_kind == Model::Mlp2020 {
            Optimizer::Sgd
        } else {
            Optimizer::Nesterov {
                momentum: real(p, "momentum")?,
            }
        };
        let dataset_files = dataset_bindings(
            &self.root,
            &self.value,
            dataset,
            id,
            seed,
            &self.locked,
            &mut sources,
        )?;
        let metadata = ProfileMetadata {
            id: id.into(),
            seed,
            dataset: dataset.into(),
            model: model_kind,
            classes,
            output_width,
            layers,
            loss,
            optimizer,
            weight_decay: real(p, "weight_decay")?,
            rate: real(&p["learning_rate"], "value")?,
            rate_description: string(&p["learning_rate"], "units")?.into(),
            rate_units,
            batch,
            logical_epochs: epochs,
            drop_last,
            initialization,
            order,
            captured_order,
            schedule,
            identity: InputIdentity {
                registry: self.registry.clone(),
                integrity_lock: self.lock.clone(),
                sources,
                preprocessing: string(d, "preprocessing")?.into(),
                split_policy: string(d, "split_policy")?.into(),
                dataset_files,
            },
        };
        Ok(ResolvedProfile {
            metadata,
            data_request,
            seed_files,
        })
    }
}

fn graph(model: &Value, p: &Value, input: usize, output: usize) -> Result<Vec<Layer>> {
    let bias = boolean(p, "bias")?;
    let g = &model["graph"];
    let mut layers = Vec::new();
    let mut previous = input;
    let hidden = match string(g, "hidden_activation")? {
        "square" => Activation::Square,
        "z2_plus_z" => Activation::SquarePlusLinear,
        _ => return Err(malformed("unknown hidden activation")),
    };
    if let Some(widths) = g["hidden_widths"].as_array() {
        for (i, w) in widths.iter().enumerate() {
            let width = w.as_u64().ok_or_else(|| malformed("hidden width"))? as usize;
            layers.push(Layer {
                name: format!("W{}", i + 1),
                input: previous,
                output: width,
                activation: hidden,
                bias,
                role: LayerRole::Hidden,
            });
            previous = width;
        }
        let activation = match string(g, "output_activation")? {
            "square" => Activation::Square,
            "identity" => Activation::Identity,
            _ => return Err(malformed("output activation")),
        };
        layers.push(Layer {
            name: format!("W{}", layers.len() + 1),
            input: previous,
            output,
            activation,
            bias,
            role: LayerRole::Output,
        });
    } else {
        eq(
            g["propagate_below_local_block"].as_bool(),
            Some(false),
            "local gradient barrier",
        )?;
        for (i, w) in array(g, "local_block_widths")?.iter().enumerate() {
            let width = w.as_u64().ok_or_else(|| malformed("block width"))? as usize;
            layers.push(Layer {
                name: format!("linear_fwd_{i:02}"),
                input: previous,
                output: width,
                activation: hidden,
                bias,
                role: LayerRole::LocalForward { block: i },
            });
            layers.push(Layer {
                name: format!("linear_lrn_{i:02}"),
                input: width,
                output,
                activation: Activation::Identity,
                bias,
                role: LayerRole::LocalHead { block: i },
            });
            previous = width;
        }
        let i = layers.len() / 2;
        let width = number(g, "output_stack_hidden_width")?;
        layers.push(Layer {
            name: format!("linear_{i:02}"),
            input: previous,
            output: width,
            activation: hidden,
            bias,
            role: LayerRole::OutputHidden,
        });
        layers.push(Layer {
            name: format!("linear_{:02}", i + 1),
            input: width,
            output,
            activation: Activation::Identity,
            bias,
            role: LayerRole::Output,
        });
    }
    Ok(layers)
}
fn validate_seed_argv(s: &Value, seed: u8, model: &str) -> Result<()> {
    let args = array(s, "argv")?;
    let val = |flag: &str| {
        args.iter()
            .position(|v| v.as_str() == Some(flag))
            .and_then(|i| args.get(i + 1))
            .and_then(Value::as_str)
    };
    eq(
        val("--seed"),
        Some(seed.to_string().as_str()),
        "captured argv seed",
    )?;
    let n = match model {
        "emlp1" => "2",
        "emlp2" => "3",
        "emlp3" => "4",
        _ => return Err(malformed("captured model")),
    };
    eq(val("--num-layers"), Some(n), "captured argv model")
}
fn validate_structure(v: &Value, layers: &[Layer]) -> Result<()> {
    let mut linear = Vec::new();
    for b in array(v, "blocks")? {
        for group in ["forward_layers", "local_head"] {
            for l in array(b, group)? {
                if l.get("in_features").is_some() {
                    linear.push(l)
                }
            }
        }
    }
    for l in array(v, "output_stack")? {
        if l.get("in_features").is_some() {
            linear.push(l)
        }
    }
    eq(linear.len(), layers.len(), "captured structure layer count")?;
    for (l, s) in layers.iter().zip(linear) {
        eq(
            string(s, "name")?,
            l.name.as_str(),
            "captured logical layer name",
        )?;
        eq(number(s, "in_features")?, l.input, "captured input")?;
        eq(number(s, "out_features")?, l.output, "captured output")?;
        let propagate = matches!(l.role, LayerRole::LocalHead { .. } | LayerRole::Output);
        eq(
            boolean(s, "propagate_backward")?,
            propagate,
            "captured stage gradient barrier",
        )?;
        eq(real(s, "momentum")?, 0.9, "captured layer momentum")?;
        eq(
            real(s, "optimizer_weight_decay")?,
            0.,
            "captured layer decay",
        )?;
    }
    Ok(())
}
fn bind_arrays(
    s: &Value,
    layers: &[Layer],
    dir: &Path,
    expected_dir: &Path,
) -> Result<Vec<InitializedArray>> {
    let arrays = array(s, "init_weights")?;
    eq(arrays.len(), layers.len(), "complete initialized arrays")?;
    layers
        .iter()
        .zip(arrays)
        .map(|(l, a)| {
            let path = dir.join(string(a, "file")?);
            eq(
                path.parent(),
                Some(expected_dir),
                "initialized array model/seed directory",
            )?;
            eq(
                path.file_stem().and_then(|s| s.to_str()),
                Some(l.name.as_str()),
                "initialized logical layer",
            )?;
            if let Some(name) = a["layer"].as_str() {
                eq(name, l.name.as_str(), "source layer name")?;
            }
            eq(string(a, "orientation")?, "in x out", "array orientation")?;
            let shape = array(a, "shape")?;
            eq(
                shape.as_slice(),
                [Value::from(l.input), Value::from(l.output)].as_slice(),
                "initialized shape",
            )?;
            let sha = string(a, "sha256")?;
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(malformed("initialized digest"));
            }
            Ok(InitializedArray {
                name: l.name.clone(),
                logical_shape: (l.output, l.input),
                source_shape: (l.input, l.output),
                orientation: Orientation::InputByOutput,
                file: LockedFile {
                    path,
                    sha256: sha.into(),
                },
            })
        })
        .collect()
}
fn check_source_graph(
    root: &Path,
    registry: &Value,
    model: Model,
    dataset: &str,
    layers: &[Layer],
    p: &Value,
    sources: &mut Vec<LockedFile>,
) -> Result<()> {
    let key = match model {
        Model::Mlp2020 => "iris",
        Model::Emlp1 => "emlp1",
        Model::Emlp2 => "emlp2",
        Model::Emlp3 => "emlp3",
    };
    let file = source(root, string(&registry["sources"][key], "path")?)?;
    let bytes = read_locked(&file)?;
    if model == Model::Mlp2020 {
        // Existing Rust source is a provenance source, not an executable configuration format.
        let text = std::str::from_utf8(&bytes).map_err(|_| malformed("iris source encoding"))?;
        if !text.contains("const DIMS: [usize; 3] = [4, 10, 3]") || !text.contains("sigma: 0.2") {
            return Err(conflict("Iris source control graph/init changed"));
        }
        eq(layers[0].output, 10, "2020 hidden")?;
    } else {
        let s = json(&bytes)?;
        if model == Model::Emlp1 {
            let a = array(&s, "layers")?;
            eq(a.len(), layers.len(), "eMLP1 complete source layers")?;
            // The reviewed faithful source is the original28x28 MNIST subset;
            // selected Iris/WDBC/released196 adaptations do not redefine its endpoints.
            eq(number(&a[0], "in")?, 28 * 28, "eMLP1 original subset input")?;
            eq(
                number(&s, "output_width")?,
                number(&registry["datasets"]["mnist_released"], "classes")?.next_power_of_two(),
                "eMLP1 original MNIST padded output domain",
            )?;
            eq(
                number(&a[0], "out")?,
                layers[0].output,
                "eMLP1 source hidden",
            )?;
            eq(
                number(&a[1], "out")?,
                number(&s, "output_width")?,
                "eMLP1 source padded output",
            )?;
            eq(
                number(&a[1], "in")?,
                number(&a[0], "out")?,
                "eMLP1 source connection",
            )?;
            for (i, l) in a.iter().enumerate() {
                eq(
                    boolean(l, "bias")?,
                    boolean(p, "bias")?,
                    "eMLP1 source bias",
                )?;
                eq(
                    string(l, "activation")?,
                    if i == 0 { "poly_relu" } else { "none" },
                    "eMLP1 source activation",
                )?;
            }
            eq(
                string(&s, "init")?,
                "xavier_uniform",
                "eMLP1 source initializer",
            )?;
        } else {
            let blocks = array(&s, "blocks")?;
            eq(blocks.len(), layers.len() / 2 - 1, "source block count")?;
            let mut source_input = number(&s, "input")?;
            eq(
                source_input,
                number(&registry["datasets"]["mnist_released"], "features")?,
                "source original input",
            )?;
            eq(
                number(&s, "output_width")?,
                number(&registry["datasets"]["mnist_released"], "classes")?.next_power_of_two(),
                "source original padded outputs",
            )?;
            for (i, b) in blocks.iter().enumerate() {
                eq(number(b, "in")?, source_input, "source local connection")?;
                source_input = number(b, "width")?;
                eq(
                    number(b, "width")?,
                    layers[2 * i].output,
                    "source block width",
                )?;
                eq(
                    boolean(b, "bias")?,
                    boolean(p, "bias")?,
                    "source local block bias",
                )?;
                eq(
                    string(b, "activation")?,
                    "poly_relu",
                    "source local block activation",
                )?;
                eq(
                    boolean(b, "backward_below")?,
                    false,
                    "source local gradient barrier",
                )?;
                eq(
                    number(b, "head")?,
                    number(&s, "output_width")?,
                    "source local head",
                )?;
                let names = array(b, "reboot_layers")?;
                eq(names.len(), 2, "source local layer names")?;
                eq(
                    names[0].as_str(),
                    Some(layers[2 * i].name.as_str()),
                    "source forward name",
                )?;
                eq(
                    names[1].as_str(),
                    Some(layers[2 * i + 1].name.as_str()),
                    "source head name",
                )?;
            }
            let out = array(&s, "output_stack")?;
            eq(out.len(), 2, "complete source output stack")?;
            eq(
                number(&out[0], "in")?,
                source_input,
                "source output-stack connection",
            )?;
            eq(
                number(&out[1], "in")?,
                number(&out[0], "out")?,
                "source final-layer connection",
            )?;
            eq(
                number(&out[1], "out")?,
                number(&s, "output_width")?,
                "source final output width",
            )?;
            eq(
                number(&out[0], "out")?,
                layers[layers.len() - 2].output,
                "source output hidden",
            )?;
            eq(
                boolean(&out[0], "backward_below")?,
                false,
                "source output gradient barrier",
            )?;
            for (i, l) in out.iter().enumerate() {
                eq(
                    boolean(l, "bias")?,
                    boolean(p, "bias")?,
                    "source output bias",
                )?;
                eq(
                    string(l, "activation")?,
                    if i == 0 { "poly_relu" } else { "none" },
                    "source output activation",
                )?;
            }
            eq(
                string(&s, "init")?,
                "xavier_uniform",
                "source block initializer",
            )?;
        }
        eq(real(&s, "momentum")?, 0.9, "source momentum")?;
        eq(
            real(&s, "weight_decay")?,
            real(p, "weight_decay")?,
            "source decay",
        )?;
    }
    if !sources.iter().any(|s| s.path == file.path) {
        sources.push(file);
    }
    if matches!(dataset, "breast_cancer" | "fashion_mnist") {
        let k = if dataset == "breast_cancer" {
            "bc"
        } else {
            "fmnist"
        };
        let file = source(root, string(&registry["sources"][k], "path")?)?;
        let s = json(&read_locked(&file)?)?;
        eq(
            number(&s, "classes")?,
            number(p, "classes")?,
            "dataset source classes",
        )?;
        eq(
            real(&s["train"], "lr_sum")?,
            real(&p["learning_rate"], "value")?,
            "source learning rate",
        )?;
        eq(
            number(&s["train"], "epochs")?,
            number(p, "logical_epochs")?,
            "source epochs",
        )?;
        eq(
            number(&s["train"], "batch")?,
            number(p, "batch")?,
            "source batch",
        )?;
        eq(
            boolean(&s["train"], "drop_last")?,
            boolean(p, "drop_last")?,
            "source tail",
        )?;
    }
    Ok(())
}

/// Immutable resolved binding; callers cannot replace its selected class/model/seed mapping.
pub struct ResolvedProfile {
    metadata: ProfileMetadata,
    data_request: DataRequest,
    seed_files: Option<SeedFiles>,
}
impl ResolvedProfile {
    /// Typed immutable learning and provenance metadata.
    pub fn metadata(&self) -> &ProfileMetadata {
        &self.metadata
    }
    /// D1 dataset request, derived from registered identity/preprocessing/fixed population.
    pub fn data_request(&self) -> DataRequest {
        self.data_request.clone()
    }
    /// Loads the canonical registered dataset and verifies its source hashes, captured
    /// inputs and actual update plan. This is the combined host-side readiness boundary;
    /// metadata resolution or captured-only preflight alone does not establish data readiness.
    pub fn prepare_dataset(&self) -> Result<PreparedDataset> {
        let inputs = self.preflight(None)?;
        for f in &self.metadata.identity.dataset_files {
            read_locked(f).map_err(|e| match e {
                ProfileError::Io(ref io) if io.kind() == io::ErrorKind::NotFound => {
                    ProfileError::MissingInput {
                        profile: self.metadata.id.clone(),
                        seed: self.metadata.seed,
                        input: f.path.display().to_string(),
                    }
                }
                other => other,
            })?;
        }
        let dataset = profile_data::load_dataset(&self.data_request)?;
        for f in &self.metadata.identity.dataset_files {
            if !dataset.files.iter().any(|(p, hash)| {
                p == &f.path.display().to_string() && hash.eq_ignore_ascii_case(&f.sha256)
            }) {
                return Err(conflict(
                    "loaded dataset source hashes differ from registered binding",
                ));
            }
        }
        Ok(PreparedDataset { dataset, inputs })
    }
    /// Verifies captured/source files and selected seed, then plans update boundaries.
    /// Dataset bytes/preprocessing remain unverified until `prepare_dataset` succeeds.
    /// Optional additional order lock must agree with the portable snapshot lock; it can
    /// strengthen caller pairing but never replace the registered hash or path.
    pub fn preflight(&self, additional_order_lock: Option<&LockedFile>) -> Result<PreparedInputs> {
        let m = &self.metadata;
        if matches!(m.initialization, Initialization::MissingCaptured)
            || matches!(m.order, OrderPolicy::Captured) && self.seed_files.is_none()
        {
            return Err(ProfileError::MissingInput {
                profile: m.id.clone(),
                seed: m.seed,
                input: "selected captured model/seed export".into(),
            });
        }
        read_locked(&m.identity.registry)?;
        read_locked(&m.identity.integrity_lock)?;
        for f in &m.identity.sources {
            read_locked(f)?;
        }
        let captured = if let Some(inputs) = &self.seed_files {
            if let Some(extra) = additional_order_lock {
                let expected = &inputs
                    .order
                    .as_ref()
                    .ok_or_else(|| malformed("order binding"))?
                    .file;
                eq(
                    extra.path.clone(),
                    expected.path.clone(),
                    "additional order selected model/seed",
                )?;
                eq(
                    extra.sha256.to_lowercase(),
                    expected.sha256.to_lowercase(),
                    "additional order digest",
                )?;
            }
            let mut verified = if matches!(m.initialization, Initialization::Captured(_)) {
                profile_data::preflight_seed(
                    &CapturedInputs {
                        profile_id: m.id.clone(),
                        per_seed: BTreeMap::from([(m.seed, inputs.clone())]),
                    },
                    m.seed,
                )?
            } else {
                profile_data::preflight_order_schedule(
                    &m.id,
                    m.seed,
                    inputs
                        .order
                        .as_ref()
                        .ok_or_else(|| malformed("required order"))?,
                    inputs
                        .schedule
                        .as_ref()
                        .ok_or_else(|| malformed("required schedule"))?,
                )?
            };
            // D1 validates physical source arrays; expose logical out-by-in matrices explicitly.
            verified.initialization = verified
                .initialization
                .into_iter()
                .map(|a| a.reversed_axes().to_owned())
                .collect();
            Some(verified)
        } else {
            if additional_order_lock.is_some() {
                return Err(conflict("order lock supplied for local-order profile"));
            }
            None
        };
        let plan = build_plan(
            m,
            self.data_request
                .train_samples
                .ok_or_else(|| malformed("train count"))?,
            captured.as_ref().and_then(|c| c.schedule.as_deref()),
        )?;
        Ok(PreparedInputs { captured, plan })
    }
}
/// One actual complete optimizer update, no synthetic fractional epoch.
#[derive(Debug, Clone)]
pub struct PlannedUpdate {
    /// Zero-based update index.
    pub index: usize,
    /// Cumulative processed samples before update.
    pub processed_start: usize,
    /// Valid logical sample count.
    pub valid_samples: usize,
    /// Source schedule index, never an interpolation.
    pub source_schedule_index: Option<usize>,
    /// Raw captured/constant coefficient.
    pub source_coefficient: f64,
    /// Rate multiplying batch-mean gradient.
    pub mean_gradient_rate: f64,
    /// Loss delta multiplier; distinct from learning rate.
    pub delta_multiplier: f64,
}
/// First actual update boundary at/after the declared logical boundary.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Completed update count.
    pub completed_updates: usize,
    /// Actual cumulative processed samples.
    pub processed_samples: usize,
    /// Logical boundaries crossed by the final update.
    pub crossed_logical_boundaries: Vec<usize>,
    /// Endpoint lies exactly on logical boundary.
    pub exact_logical_boundary: bool,
    /// Disjoint interval's complete updates.
    pub interval_updates: usize,
    /// Disjoint interval's actual processed samples.
    pub interval_samples: usize,
}
/// A dataset and input plan that both passed the canonical registered bindings.
#[derive(Debug)]
pub struct PreparedDataset {
    /// Canonical loaded data, source hashes and fixed partition.
    pub dataset: Dataset,
    /// Captured values/recipes and actual schedule plan.
    pub inputs: PreparedInputs,
}
/// Host-only validated update and checkpoint accounting.
#[derive(Debug, Clone)]
pub struct BatchPlan {
    /// Actual updates in order.
    pub updates: Vec<PlannedUpdate>,
    /// Actual update-boundary checkpoints (origin is0/0).
    pub checkpoints: Vec<Checkpoint>,
    /// Conserved consumed samples.
    pub processed_samples: usize,
    /// Registered drop-last rows, summed across local epochs.
    pub dropped_samples: usize,
}
/// Verified captured values (if needed) and numerical schedule plan; local init stays a recipe.
#[derive(Debug)]
pub struct PreparedInputs {
    /// Logical out-by-in initialized matrices only for captured init;
    /// empty initialization for locally initialized models borrowing captured order/schedule.
    pub captured: Option<VerifiedSeed>,
    /// Actual update endpoints and conversion coefficients.
    pub plan: BatchPlan,
}
fn build_plan(
    m: &ProfileMetadata,
    population: usize,
    schedule: Option<&[f64]>,
) -> Result<BatchPlan> {
    let flat = matches!(m.order, OrderPolicy::Captured);
    let per_pass = if flat || !m.drop_last {
        population
    } else {
        population / m.batch * m.batch
    };
    if per_pass == 0 {
        return Err(conflict("no complete logical batch"));
    }
    let total = per_pass
        .checked_mul(m.logical_epochs)
        .ok_or_else(|| malformed("sample count overflow"))?;
    let mut updates = Vec::new();
    let mut checkpoints = Vec::new();
    let mut processed = 0;
    let mut last_update = 0;
    let mut last_samples = 0;
    let mut next_boundary = 1;
    let segments = if flat { 1 } else { m.logical_epochs };
    let segment_samples = if flat { total } else { per_pass };
    for _ in 0..segments {
        let mut segment_done = 0;
        while segment_done < segment_samples {
            let valid = m.batch.min(segment_samples - segment_done);
            let source_index = m.schedule.as_ref().map(|s| processed / s.reference_batch);
            let coefficient = if let Some(i) = source_index {
                *schedule
                    .and_then(|s| s.get(i))
                    .ok_or_else(|| conflict("missing selected-seed source schedule coefficient"))?
            } else {
                m.rate
            };
            let scale = match m.rate_units {
                RateUnits::Mean => 1.,
                RateUnits::SumTimesValid => valid as f64,
                RateUnits::SumTimesNominal => m.batch as f64,
            };
            updates.push(PlannedUpdate {
                index: updates.len(),
                processed_start: processed,
                valid_samples: valid,
                source_schedule_index: source_index,
                source_coefficient: coefficient,
                mean_gradient_rate: coefficient * scale,
                delta_multiplier: if m.loss == Loss::MeanOutputMse {
                    2. / m.classes as f64
                } else {
                    1.
                },
            });
            processed += valid;
            segment_done += valid;
            let mut crossed = Vec::new();
            while next_boundary <= m.logical_epochs && processed >= next_boundary * per_pass {
                crossed.push(next_boundary);
                next_boundary += 1;
            }
            if !crossed.is_empty() {
                checkpoints.push(Checkpoint {
                    completed_updates: updates.len(),
                    processed_samples: processed,
                    crossed_logical_boundaries: crossed,
                    exact_logical_boundary: processed % per_pass == 0,
                    interval_updates: updates.len() - last_update,
                    interval_samples: processed - last_samples,
                });
                last_update = updates.len();
                last_samples = processed;
            }
        }
    }
    eq(processed, total, "processed sample conservation")?;
    eq(
        checkpoints
            .iter()
            .map(|c| c.interval_samples)
            .sum::<usize>(),
        total,
        "checkpoint sample conservation",
    )?;
    eq(
        checkpoints
            .iter()
            .map(|c| c.interval_updates)
            .sum::<usize>(),
        updates.len(),
        "checkpoint update conservation",
    )?;
    Ok(BatchPlan {
        updates,
        checkpoints,
        processed_samples: total,
        dropped_samples: (population - per_pass) * m.logical_epochs,
    })
}

fn dataset_bindings(
    root: &Path,
    registry: &Value,
    dataset: &str,
    id: &str,
    seed: u8,
    locked: &BTreeMap<String, String>,
    sources: &mut Vec<LockedFile>,
) -> Result<Vec<LockedFile>> {
    let d = &registry["datasets"][dataset];
    let dir = root.join(string(d, "data_path")?);
    let result = match dataset {
        "iris" => vec![
            LockedFile {
                path: dir,
                sha256: IRIS_DATA_SHA256.into(),
            },
            LockedFile {
                path: root.join(string(d, "split_path")?),
                sha256: IRIS_SPLIT_SHA256.into(),
            },
        ],
        "breast_cancer" => vec![LockedFile {
            path: dir,
            sha256: crate::data::WDBC_SHA256.into(),
        }],
        "fashion_mnist" => {
            let path = "scripts/fetch_fashion_mnist.sh";
            let f = LockedFile {
                path: root.join(path),
                sha256: FASHION_SOURCE_SHA256.into(),
            };
            let bytes = read_locked(&f)?;
            sources.push(f);
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| malformed("Fashion source hash metadata encoding"))?;
            let list = text
                .split("done <<LIST\n")
                .nth(1)
                .and_then(|s| s.split("\nLIST").next())
                .ok_or_else(|| malformed("Fashion source hash metadata absent"))?;
            let mut files = Vec::new();
            for line in list.lines() {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if fields.len() != 3
                    || fields[2].len() != 64
                    || !fields[2].bytes().all(|b| b.is_ascii_hexdigit())
                {
                    return Err(malformed("Fashion source hash metadata row"));
                }
                files.push(LockedFile {
                    path: dir.join(fields[0]),
                    sha256: fields[2].into(),
                });
            }
            eq(files.len(), 4, "Fashion source file count")?;
            files
        }
        "mnist_released" => {
            let path = string(&registry["sources"]["mnist_released"], "path")?;
            let f = LockedFile {
                path: root.join(path),
                sha256: locked
                    .get(path)
                    .ok_or_else(|| ProfileError::MissingInput {
                        profile: id.into(),
                        seed,
                        input: format!("integrity lock {path}"),
                    })?
                    .clone(),
            };
            let captured = json(&read_locked(&f)?)?;
            let manifest_path =
                root.join(string(&registry["sources"]["mnist_released_data"], "path")?);
            let manifest_file = source(
                root,
                string(&registry["sources"]["mnist_released_data"], "path")?,
            )?;
            let manifest = json(&read_locked(&manifest_file)?)?;
            for key in ["files", "pipeline", "row_order", "scaling"] {
                eq(
                    &manifest[key],
                    &captured["data"][key],
                    &format!("captured data {key}"),
                )?;
            }
            if !sources.iter().any(|s| s.path == f.path) {
                sources.push(f);
            }
            if !sources.iter().any(|s| s.path == manifest_path) {
                sources.push(manifest_file);
            }
            let expected = array(&captured["data"], "files")?;
            eq(expected.len(), 4, "captured dataset files")?;
            let mut files = Vec::new();
            for a in expected {
                let name = string(a, "file")?;
                if Path::new(name).components().count() != 1 {
                    return Err(conflict("dataset array path"));
                }
                files.push(LockedFile {
                    path: dir.join(name),
                    sha256: string(a, "sha256_npy_file")?.into(),
                });
            }
            files
        }
        _ => return Err(malformed("dataset hashes")),
    };
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    #[test]
    fn conflicting_source_bias_activation_shape_and_stage_barriers_fail_closed() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let r = StudyRegistry::load(root()).unwrap();
        let p = r.resolve("iris__emlp2", 0).unwrap();
        let original: Value = serde_json::from_slice(
            &fs::read(root().join("crates/ckks-train-bench/configs/reboot_emlp2.json")).unwrap(),
        )
        .unwrap();
        let temp = std::env::temp_dir().join(format!(
            "hekit-source-conflict-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let path = temp.join("crates/ckks-train-bench/configs/reboot_emlp2.json");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        for (pointer, value) in [
            ("/blocks/0/bias", Value::Bool(true)),
            ("/blocks/0/activation", Value::String("none".into())),
            ("/blocks/0/width", Value::from(63)),
            ("/blocks/0/backward_below", Value::Bool(true)),
            ("/output_stack/0/bias", Value::Bool(true)),
            ("/blocks/0/in", Value::from(195)),
            ("/output_stack", serde_json::json!([])),
        ] {
            let mut changed = original.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            assert!(
                check_source_graph(
                    &temp,
                    &r.value,
                    Model::Emlp2,
                    "iris",
                    &p.metadata.layers,
                    &r.value["profiles"][2],
                    &mut Vec::new()
                )
                .is_err(),
                "accepted conflicting {pointer}"
            );
        }
        fs::remove_dir_all(temp).unwrap();
    }
    #[test]
    fn partial_batch_rate_and_local_epoch_tail_accounting_use_actual_valid_samples() {
        let r = StudyRegistry::load(root()).unwrap();
        let p = r.resolve("iris__emlp1", 0).unwrap();
        let mut m = p.metadata.clone();
        m.batch = 19;
        m.logical_epochs = 2;
        let plan = build_plan(&m, 120, None).unwrap();
        assert_eq!(plan.updates.len(), 14);
        assert_eq!(plan.updates[6].valid_samples, 6);
        assert_eq!(plan.updates[6].mean_gradient_rate, m.rate * 6.);
        assert_eq!(plan.processed_samples, 240);
        assert_eq!(plan.checkpoints[0].processed_samples, 120);
        m.rate_units = RateUnits::SumTimesNominal;
        assert_eq!(
            build_plan(&m, 120, None).unwrap().updates[6].mean_gradient_rate,
            m.rate * 19.
        );
    }
    #[test]
    fn faithful_emlp1_source_original_endpoints_are_independent_of_selected_adaptation() {
        let r = StudyRegistry::load(root()).unwrap();
        let source_path = "crates/ckks-train-bench/configs/reboot_emlp1_faithful.json";
        let original: Value =
            serde_json::from_slice(&fs::read(root().join(source_path)).unwrap()).unwrap();
        let temp =
            std::env::temp_dir().join(format!("hekit-emlp1-endpoints-{}", std::process::id()));
        let path = temp.join(source_path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Each selected profile is a valid adaptation, while the referenced source stays
        // the original784-input MNIST subset with10 real classes padded to16.
        let mut accepted_conflicts = Vec::new();
        for id in [
            "iris__emlp1",
            "breast_cancer__emlp1",
            "mnist_released__emlp1",
            "fashion_mnist__emlp1",
        ] {
            let profile = r.value["profiles"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["id"].as_str() == Some(id))
                .unwrap();
            let p = r.resolve(id, 0).unwrap();
            fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
            if matches!(id, "breast_cancer__emlp1" | "fashion_mnist__emlp1") {
                let other = if id.starts_with("breast_cancer") {
                    "bc_mlp.json"
                } else {
                    "fmnist_emlp1.json"
                };
                fs::copy(
                    root().join("crates/ckks-train-bench/configs").join(other),
                    temp.join("crates/ckks-train-bench/configs").join(other),
                )
                .unwrap();
            }
            assert!(
                check_source_graph(
                    &temp,
                    &r.value,
                    Model::Emlp1,
                    &p.metadata.dataset,
                    &p.metadata.layers,
                    profile,
                    &mut Vec::new()
                )
                .is_ok(),
                "valid source rejected for {id}"
            );
            let mut missing = original.clone();
            missing["layers"][0].as_object_mut().unwrap().remove("in");
            let mut zero = original.clone();
            zero["layers"][0]["in"] = Value::from(0);
            let mut wrong = original.clone();
            wrong["layers"][0]["in"] = Value::from(196);
            let mut wrong_outputs = original.clone();
            wrong_outputs["layers"][1]["out"] = Value::from(8);
            wrong_outputs["output_width"] = Value::from(8);
            for (case, changed) in [
                ("missing original input", missing),
                ("zero original input", zero),
                ("wrong original input", wrong),
                (
                    "self-consistent wrong original output domain",
                    wrong_outputs,
                ),
            ] {
                fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
                if check_source_graph(
                    &temp,
                    &r.value,
                    Model::Emlp1,
                    &p.metadata.dataset,
                    &p.metadata.layers,
                    profile,
                    &mut Vec::new(),
                )
                .is_ok()
                {
                    accepted_conflicts.push(format!("{case} for {id}"));
                }
            }
        }
        fs::remove_dir_all(temp).unwrap();
        assert!(
            accepted_conflicts.is_empty(),
            "accepted source conflicts: {accepted_conflicts:?}"
        );
    }
    #[test]
    fn initialized_array_binding_rejects_shape_orientation_name_and_seed_directory() {
        let r = StudyRegistry::load(root()).unwrap();
        let p = r.resolve("mnist_released__emlp2", 2).unwrap();
        let path = root().join("results/quadruped/raw/reboot_emlp23_full/config_values.json");
        let source: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let seed = &source["models"]["emlp2"]["per_seed"]["2"];
        let dir = path.parent().unwrap();
        let expected = dir.join("init_weights_emlp2_seed2");
        assert!(bind_arrays(seed, &p.metadata.layers, dir, &expected).is_ok());
        for (pointer, value) in [
            ("/init_weights/0/shape", serde_json::json!([64, 196])),
            (
                "/init_weights/0/orientation",
                Value::String("out x in".into()),
            ),
            (
                "/init_weights/0/layer",
                Value::String("other_same_shape_layer".into()),
            ),
            (
                "/init_weights/0/file",
                Value::String("init_weights_emlp2_seed0/linear_fwd_00.npy".into()),
            ),
            ("/init_weights/0/sha256", Value::String("".into())),
        ] {
            let mut changed = seed.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(
                bind_arrays(&changed, &p.metadata.layers, dir, &expected).is_err(),
                "accepted {pointer}"
            );
        }
    }
}
