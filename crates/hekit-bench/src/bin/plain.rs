//! Task 51: the plaintext twin of ReBoot's eMLP-1 on several data sets, driven by a JSON config
//! (`crates/hekit-bench/configs/{mnist_emlp1_paper, fmnist_emlp1, bc_mlp}.json`), to reproduce
//! the ReBoot column of the paper's Table 4 (arXiv:2506.19693; plaintext, 10 runs).
//!
//! The training math is [`hekit_bench::twin`], the same code as `mnist --mode plain`: the
//! network `d_in -> hidden (z^2 + z) -> output_width (linear)`, MSE against one-hot targets
//! padded with zeros to `output_width`, Nesterov momentum, coupled weight decay (torch's
//! `g += wd W` before the momentum buffer), no biases, Xavier-uniform init
//! `U(±sqrt(1 / (fan_in + fan_out)))` drawn from ChaCha20 seeded by the run seed (not ReBoot's
//! draw). The accuracy is the argmax over the first `classes` outputs.
//!
//! The lr is given as ReBoot's `lr_sum` (the step on the batch-SUMMED gradient, Table 8's
//! γ); our gradient is the batch mean, so the step uses `lr = lr_sum * batch` and the decay
//! `weight_decay / batch` (the identical update, `configs/reboot_emlp1_faithful.json`
//! "lr_conversion"). Every epoch visits the training set in a fresh ChaCha20 permutation
//! (stream `rng(seed, 3)`); with `drop_last` the last partial batch is skipped, as ReBoot's
//! loader does (its subset run trained 21 x 48 of 1024 samples).
//!
//! # Config
//!
//! ```json
//! {
//!   "name": "fmnist_emlp1",
//!   "dataset": { "kind": "mnist_idx", "path": "data/fashion_mnist", "resize": 14 },
//!   "classes": 10,
//!   "model": { "hidden": 32, "output_width": 16 },
//!   "train": { "lr_sum": 0.0005, "weight_decay": 0.0, "momentum": 0.9, "batch": 48,
//!              "drop_last": true, "epochs": 50, "report_epochs": [10, 20, 50] },
//!   "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
//!   "target": { "source": "...", "test_acc_mean": 0.8512, "test_acc_std": 0.0020 },
//!   "out": "results/plain_fmnist_emlp1.json"
//! }
//! ```
//!
//! - `dataset.kind = "mnist_idx"`: the four IDX files in `path` (MNIST or Fashion-MNIST); with
//!   `resize: S`, ReBoot's preprocessing [`reboot_preprocess`] (antialiased bicubic resize to
//!   `S x S`, min-max scaling over the training set), else pixels `byte / 255`.
//! - `dataset.kind = "wdbc"`: UCI WDBC at `path` (sha256-checked), a stratified split with
//!   `test_size` test samples drawn from ChaCha20 stream `rng(split_seed, 4)` (`split_seed`: a
//!   number, or `"per_run"` for the run seed), then `scaling` with the training statistics:
//!   `"min_max"` (per column, default) or `"standardize"` (z-score).
//! - Other keys (`notes`, ...) are free text and are copied into the output.
//!
//! Usage: `plain --config FILE [--seeds 0,1,2 | --seeds 0-9] [--epochs N] [--jobs 4]
//! [--split-seed N|per_run] [--scaling min_max|standardize] [--out FILE]` (flags override the config). Seeds run in parallel
//! on `--jobs` threads (default 4), each single-threaded and deterministic. Use `cargo run
//! --release`: a debug build gives bit-identical accuracies but is 60-200x slower (about 25 s
//! per Fashion-MNIST epoch and seed instead of 0.14 s).
//!
//! Output: per seed the test accuracy after every epoch, and over seeds the mean, sample std
//! (ddof 1, as R-REB-3), min and max at each `report_epochs` entry, at the last epoch and at
//! each seed's best epoch (test-set selected, so optimistic), each against the target's ±1 pt
//! gate.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hekit_bench::data::{
    Mnist, Split, load_mnist, load_wdbc, min_max_columns, one_hot, reboot_preprocess, sha256_hex,
    standardize, stratified_split,
};
use hekit_bench::twin::{
    WeightDecay, accuracy_over, forward_plain, init_net, permutation, rng, twin_step,
};
use ndarray::{Array2, Axis};
use serde_json::{Value, json};

/// The ±gate of Table 4 reproduction, in accuracy (1 pt).
const GATE: f64 = 0.01;

/// A loaded data set: features and labels of the training and test sets, and what was read.
struct Data {
    train_x: Array2<f64>,
    train_y: Vec<u8>,
    test_x: Array2<f64>,
    test_y: Vec<u8>,
    /// File names and SHA-256 digests read.
    files: Vec<(String, String)>,
    /// A description of the preprocessing.
    prep: String,
}

/// How the WDBC split is seeded.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SplitSeed {
    Fixed(u8),
    PerRun,
}

/// The run configuration (the config file with the flag overrides).
struct Cfg {
    raw: Value,
    name: String,
    kind: String,
    path: PathBuf,
    resize: Option<usize>,
    test_size: usize,
    /// `wdbc` only: `min_max` (per column) or `standardize`.
    scaling: String,
    split_seed: SplitSeed,
    classes: usize,
    hidden: usize,
    output_width: usize,
    lr_sum: f64,
    weight_decay: f64,
    momentum: f64,
    batch: usize,
    drop_last: bool,
    epochs: usize,
    report_epochs: Vec<usize>,
    seeds: Vec<u8>,
    jobs: usize,
    out: PathBuf,
    target: Option<(f64, f64, String)>,
}

fn get<'a>(v: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('.').try_fold(v, |v, k| v.get(k))
}

fn need_f64(v: &Value, path: &str) -> Result<f64, String> {
    get(v, path)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("config: {path} must be a number"))
}

fn need_usize(v: &Value, path: &str) -> Result<usize, String> {
    get(v, path)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| format!("config: {path} must be a non-negative integer"))
}

fn opt_usize(v: &Value, path: &str) -> Result<Option<usize>, String> {
    match get(v, path) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => need_usize(v, path).map(Some),
    }
}

fn need_str(v: &Value, path: &str) -> Result<String, String> {
    get(v, path)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("config: {path} must be a string"))
}

fn parse_seeds(s: &str) -> Result<Vec<u8>, String> {
    let one = |t: &str| t.trim().parse::<u8>().map_err(|e| format!("seed {t}: {e}"));
    let mut out = Vec::new();
    for part in s.split(',').filter(|t| !t.trim().is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => out.extend(one(a)?..=one(b)?),
            None => out.push(one(part)?),
        }
    }
    if out.is_empty() {
        return Err("no seeds".into());
    }
    Ok(out)
}

fn parse_split_seed(v: &Value) -> Result<SplitSeed, String> {
    match v {
        Value::String(s) if s == "per_run" => Ok(SplitSeed::PerRun),
        Value::String(s) => s
            .parse::<u8>()
            .map(SplitSeed::Fixed)
            .map_err(|_| format!("split_seed {s}: a seed 0..=255 or \"per_run\"")),
        Value::Number(n) => n
            .as_u64()
            .and_then(|x| u8::try_from(x).ok())
            .map(SplitSeed::Fixed)
            .ok_or_else(|| format!("split_seed {n}: a seed 0..=255 or \"per_run\"")),
        _ => Err("split_seed: a seed 0..=255 or \"per_run\"".into()),
    }
}

fn parse_cfg(argv: &[String]) -> Result<Cfg, String> {
    let mut flags = std::collections::HashMap::new();
    let mut it = argv.iter();
    while let Some(f) = it.next() {
        let v = it.next().ok_or_else(|| format!("{f} needs a value"))?;
        match f.as_str() {
            "--config" | "--seeds" | "--epochs" | "--jobs" | "--split-seed" | "--scaling"
            | "--out" => {
                flags.insert(f.as_str(), v.clone());
            }
            _ => return Err(format!("unknown flag {f}")),
        }
    }
    let path = flags.get("--config").ok_or("--config FILE is required")?;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let raw: Value = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    let kind = need_str(&raw, "dataset.kind")?;
    if !matches!(kind.as_str(), "mnist_idx" | "wdbc") {
        return Err(format!("dataset.kind {kind} (mnist_idx | wdbc)"));
    }
    let split_seed = match flags.get("--split-seed") {
        Some(s) => parse_split_seed(&Value::String(s.clone()))?,
        None => match get(&raw, "dataset.split_seed") {
            Some(v) => parse_split_seed(v)?,
            None => SplitSeed::Fixed(0),
        },
    };
    let seeds = match flags.get("--seeds") {
        Some(s) => parse_seeds(s)?,
        None => get(&raw, "seeds")
            .and_then(Value::as_array)
            .ok_or("config: seeds must be an array")?
            .iter()
            .map(|s| {
                s.as_u64()
                    .and_then(|x| u8::try_from(x).ok())
                    .ok_or_else(|| format!("seed {s}: 0..=255"))
            })
            .collect::<Result<_, _>>()?,
    };
    let num = |k: &str, d: usize| -> Result<usize, String> {
        flags
            .get(k)
            .map_or(Ok(d), |v| v.parse().map_err(|e| format!("{k} {v}: {e}")))
    };
    let epochs = num("--epochs", need_usize(&raw, "train.epochs")?)?;
    let mut report_epochs: Vec<usize> = get(&raw, "train.report_epochs")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|e| e as usize)
                .collect()
        })
        .unwrap_or_default();
    report_epochs.retain(|&e| (1..=epochs).contains(&e));
    let target = match get(&raw, "target") {
        Some(t) => Some((
            need_f64(t, "test_acc_mean")?,
            need_f64(t, "test_acc_std")?,
            need_str(t, "source")?,
        )),
        None => None,
    };
    let cfg = Cfg {
        name: need_str(&raw, "name")?,
        path: PathBuf::from(need_str(&raw, "dataset.path")?),
        resize: opt_usize(&raw, "dataset.resize")?,
        test_size: opt_usize(&raw, "dataset.test_size")?.unwrap_or(0),
        scaling: flags.get("--scaling").cloned().unwrap_or_else(|| {
            get(&raw, "dataset.scaling")
                .and_then(Value::as_str)
                .unwrap_or("min_max")
                .to_string()
        }),
        split_seed,
        classes: need_usize(&raw, "classes")?,
        hidden: need_usize(&raw, "model.hidden")?,
        output_width: need_usize(&raw, "model.output_width")?,
        lr_sum: need_f64(&raw, "train.lr_sum")?,
        weight_decay: need_f64(&raw, "train.weight_decay")?,
        momentum: need_f64(&raw, "train.momentum")?,
        batch: need_usize(&raw, "train.batch")?,
        drop_last: get(&raw, "train.drop_last")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        epochs,
        report_epochs,
        seeds,
        jobs: num("--jobs", 4)?.max(1),
        out: PathBuf::from(match flags.get("--out") {
            Some(o) => o.clone(),
            None => need_str(&raw, "out")?,
        }),
        target,
        kind,
        raw,
    };
    if cfg.epochs == 0 || cfg.batch == 0 || cfg.hidden == 0 || cfg.classes < 2 {
        return Err("epochs, batch and hidden must be positive and classes >= 2".into());
    }
    if cfg.output_width < cfg.classes {
        return Err(format!(
            "output_width {} below {} classes",
            cfg.output_width, cfg.classes
        ));
    }
    if !matches!(cfg.scaling.as_str(), "min_max" | "standardize") {
        return Err(format!(
            "dataset.scaling {} (min_max | standardize)",
            cfg.scaling
        ));
    }
    if cfg.kind == "wdbc" && cfg.test_size == 0 {
        return Err("dataset.test_size is required for wdbc".into());
    }
    Ok(cfg)
}

/// The SHA-256 of a file, for the output's record of what was read.
fn file_sha(path: &Path) -> Result<(String, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((path.display().to_string(), sha256_hex(&bytes)))
}

fn load(cfg: &Cfg, split_seed: u8) -> Result<Data, String> {
    match cfg.kind.as_str() {
        "mnist_idx" => {
            let hint = |e: std::io::Error| {
                format!(
                    "{}: {e} (scripts/fetch_mnist.sh or scripts/fetch_fashion_mnist.sh)",
                    cfg.path.display()
                )
            };
            let tr = load_mnist(&cfg.path, Split::Train, None).map_err(hint)?;
            let te = load_mnist(&cfg.path, Split::Test, None).map_err(hint)?;
            let mut files = Vec::new();
            for split in [Split::Train, Split::Test] {
                let (a, b) = split.files();
                files.push(file_sha(&cfg.path.join(a))?);
                files.push(file_sha(&cfg.path.join(b))?);
            }
            let (tr, te, prep): (Mnist, Mnist, String) = match cfg.resize {
                Some(side) => {
                    let (a, b) = reboot_preprocess(&tr, &te, side);
                    let prep = format!(
                        "{0}x{1} -> {side}x{side} antialiased bicubic (a = -0.5), then min-max \
                         scaling over the training set (data::reboot_preprocess)",
                        tr.rows, tr.cols
                    );
                    (a, b, prep)
                }
                None => (tr, te, "pixels byte / 255".into()),
            };
            Ok(Data {
                train_x: tr.images,
                train_y: tr.labels,
                test_x: te.images,
                test_y: te.labels,
                files,
                prep,
            })
        }
        _ => {
            let t = load_wdbc(&cfg.path).map_err(|e| e.to_string())?;
            let (tr, te) = stratified_split(&t.labels, cfg.test_size, &mut rng(split_seed, 4));
            let mut train_x = t.features.select(Axis(0), &tr);
            let mut test_x = t.features.select(Axis(0), &te);
            let scaling = if cfg.scaling == "standardize" {
                standardize(&mut train_x, &mut test_x);
                "standardization with the training mean and population std (data::standardize)"
            } else {
                min_max_columns(&mut train_x, &mut test_x);
                "per-column min-max scaling with the training ranges (data::min_max_columns)"
            };
            Ok(Data {
                train_x,
                train_y: tr.iter().map(|&i| t.labels[i]).collect(),
                test_x,
                test_y: te.iter().map(|&i| t.labels[i]).collect(),
                files: vec![file_sha(&cfg.path)?],
                prep: format!(
                    "stratified split ({} train / {} test, ChaCha20 rng(split_seed = {split_seed}, \
                     4), data::stratified_split), then {scaling}",
                    tr.len(),
                    te.len()
                ),
            })
        }
    }
}

/// One seed's training: the test and train accuracy and the test MSE after every epoch.
struct RunResult {
    seed: u8,
    split_seed: u8,
    test_acc: Vec<f64>,
    train_acc: Vec<f64>,
    test_mse: Vec<f64>,
    test_acc_init: f64,
    updates: usize,
    seconds: f64,
}

fn train_one(cfg: &Cfg, data: &Data, seed: u8, split_seed: u8) -> RunResult {
    let t0 = Instant::now();
    let dims = [data.train_x.ncols(), cfg.hidden, cfg.output_width];
    let mut net = init_net(&dims, &mut rng(seed, 2), "xavier", true);
    let mut vel: Vec<Array2<f64>> = net.iter().map(|l| Array2::zeros(l.w.dim())).collect();
    let y_train = one_hot(&data.train_y, cfg.output_width);
    let y_test = one_hot(&data.test_y, cfg.output_width);
    let lr = cfg.lr_sum * cfg.batch as f64;
    let wd = WeightDecay {
        mean: cfg.weight_decay / cfg.batch as f64,
        decoupled: false,
    };
    let n = data.train_x.nrows();
    let steps = if cfg.drop_last {
        n / cfg.batch
    } else {
        n.div_ceil(cfg.batch)
    };
    let mut shuffle = rng(seed, 3);
    let test_acc_init = accuracy_over(
        &forward_plain(&net, &data.test_x),
        &data.test_y,
        cfg.classes,
    );
    let (mut test_acc, mut train_acc, mut test_mse) = (Vec::new(), Vec::new(), Vec::new());
    let mut updates = 0;
    for _ in 0..cfg.epochs {
        let order = permutation(n, &mut shuffle);
        for s in 0..steps {
            let idx = &order[s * cfg.batch..((s + 1) * cfg.batch).min(n)];
            let x = data.train_x.select(Axis(0), idx);
            let y = y_train.select(Axis(0), idx);
            // Biases are absent (zero, never updated): bias lr 0.
            twin_step(&mut net, &mut vel, &x, &y, lr, cfg.momentum, 0.0, wd);
            updates += 1;
        }
        let out = forward_plain(&net, &data.test_x);
        test_acc.push(accuracy_over(&out, &data.test_y, cfg.classes));
        test_mse.push((&out - &y_test).mapv(|v| v * v).mean().unwrap_or(f64::NAN));
        let tr = forward_plain(&net, &data.train_x);
        train_acc.push(accuracy_over(&tr, &data.train_y, cfg.classes));
    }
    RunResult {
        seed,
        split_seed,
        test_acc,
        train_acc,
        test_mse,
        test_acc_init,
        updates,
        seconds: t0.elapsed().as_secs_f64(),
    }
}

/// One seed's result and, when the data set was loaded for that seed only, its data.
type SeedOutcome = Result<(RunResult, Option<Data>), String>;

/// Mean, sample std (ddof 1), min and max.
fn summary(v: &[f64]) -> Value {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let std = if v.len() > 1 {
        (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    json!({
        "mean": mean,
        "std_sample": std,
        "min": v.iter().copied().fold(f64::INFINITY, f64::min),
        "max": v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

/// `summary` plus the ±1 pt gate against the target mean.
fn gated(v: &[f64], target: &Option<(f64, f64, String)>) -> Value {
    let mut s = summary(v);
    if let Some((t, _, _)) = target {
        let mean = s["mean"].as_f64().unwrap_or(f64::NAN);
        s["minus_target_pt"] = json!((mean - t) * 100.0);
        s["within_1pt"] = json!((mean - t).abs() <= GATE);
    }
    s
}

fn cmd_out(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn run(cfg: &Cfg) -> Result<(), String> {
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let t0 = Instant::now();
    let per_run_data = cfg.kind == "wdbc" && cfg.split_seed == SplitSeed::PerRun;
    let fixed = match cfg.split_seed {
        SplitSeed::Fixed(s) => s,
        SplitSeed::PerRun => 0,
    };
    let shared = if per_run_data {
        None
    } else {
        Some(load(cfg, fixed)?)
    };
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<SeedOutcome>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..cfg.jobs.min(cfg.seeds.len()) {
            s.spawn(|| {
                loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&seed) = cfg.seeds.get(k) else { break };
                    let r = match &shared {
                        Some(d) => Ok((train_one(cfg, d, seed, fixed), None)),
                        None => load(cfg, seed).map(|d| (train_one(cfg, &d, seed, seed), Some(d))),
                    };
                    if let Ok((r, _)) = &r {
                        eprintln!(
                            "plain {}: seed {}: last epoch test acc {:.4}, best {:.4} ({:.1} s)",
                            cfg.name,
                            r.seed,
                            r.test_acc.last().copied().unwrap_or(f64::NAN),
                            r.test_acc.iter().copied().fold(0.0, f64::max),
                            r.seconds
                        );
                    }
                    results.lock().expect("no poisoned lock").push(r);
                }
            });
        }
    });
    let mut runs = Vec::new();
    let mut data_desc = shared
        .as_ref()
        .map(|d| (d.files.clone(), d.prep.clone(), d.train_x.ncols()));
    for r in results.into_inner().expect("no poisoned lock") {
        let (run, d) = r?;
        if data_desc.is_none() {
            data_desc = d.map(|d| {
                let n = d.train_x.ncols();
                (d.files, d.prep, n)
            });
        }
        runs.push(run);
    }
    runs.sort_by_key(|r| r.seed);
    let (files, prep, d_in) = data_desc.unwrap_or_default();
    let col = |e: usize| -> Vec<f64> { runs.iter().map(|r| r.test_acc[e - 1]).collect() };
    let best: Vec<f64> = runs
        .iter()
        .map(|r| r.test_acc.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        .collect();
    let best_epoch: Vec<usize> = runs
        .iter()
        .map(|r| {
            // The first epoch reaching the best accuracy (1-based).
            let b = r.test_acc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            r.test_acc.iter().position(|&a| a == b).map_or(0, |p| p + 1)
        })
        .collect();
    let at_epoch: serde_json::Map<String, Value> = cfg
        .report_epochs
        .iter()
        .map(|&e| (e.to_string(), gated(&col(e), &cfg.target)))
        .collect();
    let mean_curve: Vec<f64> = (1..=cfg.epochs)
        .map(|e| col(e).iter().sum::<f64>() / runs.len() as f64)
        .collect();
    let dirty = cmd_out("git", &["status", "--porcelain", "--untracked-files=no"]);
    let out = json!({
        "task": "Task 51: plaintext twin of ReBoot eMLP-1 against the paper's Table 4 (ReBoot column)",
        "provenance": {
            "written_by": "the plain binary (raw output, not assembled)",
            "git_commit": cmd_out("git", &["rev-parse", "HEAD"]),
            "git_modified_tracked_files": if dirty.is_empty() { Value::Array(vec![]) } else {
                Value::Array(dirty.lines().map(|l| json!(l.trim())).collect())
            },
            "started_unix": started,
            "wall_seconds": t0.elapsed().as_secs_f64(),
            "cpu": cmd_out("sysctl", &["-n", "machdep.cpu.brand_string"]),
            "jobs": cfg.jobs,
            "debug_assertions": cfg!(debug_assertions),
            "argv": std::env::args().collect::<Vec<_>>(),
        },
        "config_file": cfg.raw,
        "effective": {
            "network": format!(
                "{} -> {} (z^2 + z) -> {} (linear, {} classes), no biases, Xavier uniform, MSE",
                d_in,
                cfg.hidden, cfg.output_width, cfg.classes
            ),
            "lr_sum": cfg.lr_sum,
            "lr_on_batch_mean_gradient": cfg.lr_sum * cfg.batch as f64,
            "weight_decay_reboot": cfg.weight_decay,
            "weight_decay_on_batch_mean_gradient": cfg.weight_decay / cfg.batch as f64,
            "weight_decay_form": "coupled (torch SGD: g += wd W before momentum and Nesterov)",
            "momentum": cfg.momentum,
            "batch": cfg.batch,
            "drop_last": cfg.drop_last,
            "epochs": cfg.epochs,
            "shuffle": "a fresh ChaCha20 permutation per epoch, rng(seed, 3)",
            "init": "Xavier uniform from ChaCha20 rng(seed, 2) (not ReBoot's draw)",
            "lr_schedule": "constant",
            "split_seed": match cfg.split_seed {
                SplitSeed::Fixed(s) => json!(s),
                SplitSeed::PerRun => json!("per_run (= run seed)"),
            },
            "preprocessing": prep,
            "files_sha256": files.iter().map(|(f, h)| json!({"file": f, "sha256": h})).collect::<Vec<_>>(),
            "train_samples": runs.first().map(|r| r.updates / cfg.epochs * cfg.batch),
        },
        "target": cfg.target.as_ref().map(|(m, s, src)| json!({
            "test_acc_mean": m, "test_acc_std": s, "source": src, "gate": "|mean - target| <= 1 pt",
        })),
        "summary": {
            "seeds": runs.iter().map(|r| r.seed).collect::<Vec<_>>(),
            "last_epoch": gated(&col(cfg.epochs), &cfg.target),
            "at_epoch": at_epoch,
            "best_epoch_test_selected": gated(&best, &cfg.target),
            "best_epoch_per_seed": best_epoch,
            "mean_curve": mean_curve,
        },
        "runs": runs.iter().map(|r| json!({
            "seed": r.seed,
            "split_seed": r.split_seed,
            "test_acc_init": r.test_acc_init,
            "test_acc": r.test_acc,
            "train_acc": r.train_acc,
            "test_mse": r.test_mse,
            // The first epoch (1-based) whose test MSE is not finite: training diverged.
            "nonfinite_from_epoch": r.test_mse.iter().position(|m| !m.is_finite()).map(|p| p + 1),
            "updates": r.updates,
            "seconds": r.seconds,
        })).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&out).map_err(|e| e.to_string())? + "\n";
    if let Some(dir) = cfg.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&cfg.out, text).map_err(|e| format!("{}: {e}", cfg.out.display()))?;
    let s = &out["summary"];
    let pct = |v: &Value| {
        format!(
            "{:.2} ± {:.2}",
            v["mean"].as_f64().unwrap_or(f64::NAN) * 100.0,
            v["std_sample"].as_f64().unwrap_or(f64::NAN) * 100.0
        )
    };
    eprintln!(
        "plain {}: last epoch {} %, best epoch {} %{}; wrote {}",
        cfg.name,
        pct(&s["last_epoch"]),
        pct(&s["best_epoch_test_selected"]),
        cfg.target
            .as_ref()
            .map_or(String::new(), |(m, sd, _)| format!(
                " (target {:.2} ± {:.2} %)",
                m * 100.0,
                sd * 100.0
            )),
        cfg.out.display()
    );
    Ok(())
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let r = parse_cfg(&argv).and_then(|cfg| run(&cfg));
    if let Err(e) = r {
        eprintln!("plain: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_lists_and_ranges() {
        assert_eq!(parse_seeds("0-3").unwrap(), [0, 1, 2, 3]);
        assert_eq!(parse_seeds("5,1, 2").unwrap(), [5, 1, 2]);
        assert!(parse_seeds("").is_err());
        assert!(parse_seeds("300").is_err());
    }

    #[test]
    fn split_seed_forms() {
        assert_eq!(parse_split_seed(&json!(3)).unwrap(), SplitSeed::Fixed(3));
        assert_eq!(parse_split_seed(&json!("7")).unwrap(), SplitSeed::Fixed(7));
        assert_eq!(
            parse_split_seed(&json!("per_run")).unwrap(),
            SplitSeed::PerRun
        );
        assert!(parse_split_seed(&json!(256)).is_err());
        assert!(parse_split_seed(&json!(true)).is_err());
    }

    #[test]
    fn summary_uses_the_sample_std() {
        let s = summary(&[1.0, 2.0, 3.0]);
        assert_eq!(s["mean"], json!(2.0));
        assert_eq!(s["std_sample"], json!(1.0));
        let g = gated(&[0.9, 0.91], &Some((0.9, 0.0, String::new())));
        assert_eq!(g["within_1pt"], json!(true));
        let g = gated(&[0.9, 0.9], &Some((0.92, 0.0, String::new())));
        assert_eq!(g["within_1pt"], json!(false));
    }

    /// The committed configs parse, and the WDBC one trains end to end (2 epochs, 2 seeds).
    #[test]
    fn configs_parse_and_wdbc_trains() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs");
        for name in ["mnist_emlp1_paper", "fmnist_emlp1", "bc_mlp"] {
            let p = dir.join(format!("{name}.json"));
            let cfg = parse_cfg(&["--config".into(), p.display().to_string()]).unwrap();
            assert_eq!(cfg.name, name);
            assert_eq!(cfg.seeds, (0..10).collect::<Vec<u8>>());
        }
        let p = dir.join("bc_mlp.json");
        let mut cfg = parse_cfg(&["--config".into(), p.display().to_string()]).unwrap();
        cfg.path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/breast_cancer.csv");
        cfg.epochs = 2;
        let data = load(&cfg, 0).unwrap();
        assert_eq!((data.train_x.nrows(), data.test_x.nrows()), (455, 114));
        assert_eq!(data.train_x.ncols(), 30);
        let a = train_one(&cfg, &data, 1, 0);
        let b = train_one(&cfg, &data, 1, 0);
        assert_eq!(a.test_acc, b.test_acc, "deterministic");
        assert_eq!(a.updates, 2 * (455 / 16));
        assert!(a.test_acc[1] > 0.8, "{:?}", a.test_acc);
    }
}
