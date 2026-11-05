//! Task 32: interactive encrypted training of ReBoot's eMLP-1 on MNIST and the measured
//! key-switch counts that decide RQ1 (`docs/m2-report.md` "MNIST eMLP-1").
//!
//! # Network and training (`crates/hekit-bench/configs/reboot_emlp1.json`)
//!
//! The config's values are transcribed into the defaults below, not parsed (the crate has no
//! JSON dependency): dims, activations, `lr` and `momentum` as in the config; `batch` 48 → 32
//! lanes and `epochs` 10 → 1 per the Task 32 rulings.
//!
//! `784 -> 32` with ReBoot's `z^2 + z` ([`PolyRelu2`]), then `32 -> 10` linear ("none",
//! [`Identity`]); MSE against one-hot labels (the trainer's loss gradient is `a - y`); NAG
//! with `lr = 0.001`, `momentum = 0.9` per the config (Table 8 hyperparameters; weight decay
//! appears only in `spikes/reboot/RUN.md` and is not applied). Biases take plain SGD steps
//! with the same `lr` (the NAG API of Task 28 has no bias momentum: a recorded deviation).
//! Initial weights and biases are PyTorch `nn.Linear`'s default, uniform in
//! `±1/sqrt(fan_in)`, from ChaCha20 seeded by `--seed`. Pixels are `byte / 255`. Batches are
//! the first `--subset` training images in file order (no shuffling); a last partial batch is
//! trained with the trainer's valid-lane mask.
//!
//! # Parameters, layout and keys
//!
//! `--params n16_train` (default): N = 2^16, 2^15 slots, 31 levels, scale 2^42. The layout
//! has `d = 1024` (784 padded) feature positions and `B = 32` lanes, so `d B = 2^15`: one
//! ciphertext per batch. ReBoot's `B = 48` would need two ciphertexts per batch (not run; see
//! the report for its formula cost). Galois keys: exactly
//! [`mlp_rotation_steps`]; `--transpose reuse` (default) builds the `W2^T` transpose from the
//! keys the step already has (0 extra keys, +12 rotations per update for eMLP-1,
//! `docs/layout.md` §2.3), `--transpose dedicated` generates its own keys (+14 keys of
//! ≈ 302 MB each).
//!
//! `--params small` (N = 2^12, insecure, for the smoke test only) runs the same code path on
//! data sets with smaller images (e.g. 8 x 8 synthetic IDX files).
//!
//! # What is measured
//!
//! - The evaluator's [`OpCounter`] per batch (rotations and relinearizations are the key
//!   switches), per sample = total / samples trained.
//! - The closed-form prediction of the same count ([`predict`]): the spike cost model's
//!   interactive variant (`spikes/costmodel/costmodel.py`, ported in [`cost_model_interactive`])
//!   plus the terms by which this implementation differs from it (replicated batch sum,
//!   recomputed baby rotations, bias gradients, BSGS split, transpose keys).
//! - Wall-clock per batch (server step and client refresh), keygen and encryption times.
//! - A plaintext f64 NAG twin with the same initial weights and batches: the max weight and
//!   bias differences after training, and the test accuracy of both.
//!
//! Usage: `mnist [--epochs 1] [--subset 1024] [--test-subset 10000] [--batch 32]
//! [--hidden 32] [--params n16_train|small] [--transpose reuse|dedicated] [--seed 32]
//! [--lr 0.001] [--momentum 0.9] [--data data/mnist] [--out results/mnist_emlp1.json]
//! [--plan]`. `--plan` prints the layout, key count and predicted counts without generating
//! keys.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use hekit_bench::data::{Mnist, Split, load_mnist, one_hot};
use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Identity, PolyRelu2};
use hekit_nn::dense::{TransposeKeys, transpose_rotation_plan};
use hekit_nn::layout::{LayerShape, Layout, unpack_activations};
use hekit_nn::optim::Nag;
use hekit_nn::refresh::ClientRefresher;
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_key_pool, mlp_rotation_steps};
use hekit_nn::trainer_ni::{
    ActCost, BootPlacement, CompactLayout, NiOptimizer, NonInteractiveTrainer, UpdateCost,
    boot_schedule, levels_per_iter, ni_rotation_steps, predict_update_for,
};
use ndarray::{Array2, Axis, s};

const CLASSES: usize = 10;

// ---------------------------------------------------------------------------------------------
// Arguments.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Args {
    epochs: usize,
    subset: usize,
    test_subset: usize,
    batch: usize,
    hidden: usize,
    params: String,
    transpose: String,
    seed: u8,
    lr: f64,
    momentum: f64,
    data: PathBuf,
    out: Option<PathBuf>,
    plan: bool,
    mode: String,
    placement: String,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        epochs: 1,
        subset: 1024,
        test_subset: 10_000,
        batch: 32,
        hidden: 32,
        params: String::new(),
        transpose: "reuse".into(),
        seed: 32,
        lr: 0.001,
        momentum: 0.9,
        data: PathBuf::from("data/mnist"),
        out: None,
        plan: false,
        mode: "interactive".into(),
        placement: "after_update".into(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--plan" {
            a.plan = true;
            continue;
        }
        let v = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag} {v}: {e}"));
        let real = |v: &str| v.parse::<f64>().map_err(|e| format!("{flag} {v}: {e}"));
        match flag.as_str() {
            "--epochs" => a.epochs = num(&v)?,
            "--subset" => a.subset = num(&v)?,
            "--test-subset" => a.test_subset = num(&v)?,
            "--batch" => a.batch = num(&v)?,
            "--hidden" => a.hidden = num(&v)?,
            "--params" => a.params = v,
            "--transpose" => a.transpose = v,
            "--seed" => a.seed = v.parse().map_err(|e| format!("--seed {v}: {e}"))?,
            "--lr" => a.lr = real(&v)?,
            "--momentum" => a.momentum = real(&v)?,
            "--data" => a.data = PathBuf::from(v),
            "--out" => a.out = Some(PathBuf::from(v)),
            "--mode" => a.mode = v,
            "--placement" => a.placement = v,
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if a.epochs == 0 || a.subset == 0 || a.batch == 0 || a.hidden == 0 {
        return Err("--epochs, --subset, --batch and --hidden must be positive".into());
    }
    let ni = match a.mode.as_str() {
        "interactive" => false,
        "noninteractive" => true,
        m => return Err(format!("unknown --mode {m} (interactive | noninteractive)")),
    };
    placement_named(&a.placement)?;
    if a.params.is_empty() {
        a.params = if ni { "n16_boot" } else { "n16_train" }.into();
    }
    Ok(a)
}

fn placement_named(name: &str) -> Result<BootPlacement, String> {
    match name {
        "after_update" => Ok(BootPlacement::AfterUpdate),
        "after_activation" => Ok(BootPlacement::AfterActivation),
        "when_needed" => Ok(BootPlacement::WhenNeeded),
        p => Err(format!(
            "unknown --placement {p} (after_update | after_activation | when_needed)"
        )),
    }
}

fn params_named(name: &str) -> Result<CkksParams, String> {
    match name {
        "n16_train" => Ok(CkksParams::n16_train()),
        "n16_boot" => Ok(CkksParams::n16_boot()),
        // Insecure (N = 2^10): n16_boot's chain in miniature, the non-interactive smoke test's
        // stand-in for n16_boot (512 slots, bootstrapping with BootParams::default_n16 at 9
        // log-slots).
        "mini_boot" => {
            let (q, p) = hekit_ckks::params::n16_boot_bits();
            CkksParams::new(10, &q, &p, 42, 8).map_err(|e| e.to_string())
        }
        // Insecure (N = 2^12 with 11 primes): the smoke test's fast stand-in for n16_train.
        "small" => {
            let mut q = vec![58];
            q.extend([42; 10]);
            CkksParams::new(12, &q, &[60], 42, 11).map_err(|e| e.to_string())
        }
        _ => Err(format!(
            "unknown --params {name} (n16_train | small | n16_boot | mini_boot)"
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Closed-form counts.
// ---------------------------------------------------------------------------------------------

/// Rotations and relinearizations (key switches = their sum).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Ks {
    rot: u64,
    relin: u64,
}

impl Ks {
    fn add(&mut self, rot: usize, relin: usize) {
        self.rot += rot as u64;
        self.relin += relin as u64;
    }
    fn total(&self) -> u64 {
        self.rot + self.relin
    }
}

fn log2(x: usize) -> usize {
    x.trailing_zeros() as usize
}

/// The spike's BSGS split (`costmodel.bsgs_split`): minimises `(n1-1) + (n2-1) + n2`, ties to
/// the larger `n1`.
fn spike_split(m: usize) -> (usize, usize) {
    let mut best: Option<(usize, usize, usize)> = None;
    let mut n1 = m;
    while n1 >= 1 {
        let n2 = m / n1;
        let ks = (n1 - 1) + (n2 - 1) + n2;
        if best.is_none_or(|b| ks < b.0) {
            best = Some((ks, n1, n2));
        }
        n1 /= 2;
    }
    let b = best.expect("m >= 1");
    (b.1, b.2)
}

/// `spikes/costmodel/costmodel.py` `network_cost("diag_batched", layers, batch, slots,
/// interactive=True)` for one update on one ciphertext (`d B <= slots`): per layer
/// forward `(n1-1)+(n2-1)+fold` rotations and `n2` relins (+1 relin with an activation);
/// for every layer but the first a transpose of `m - 1` rotations and the backward matvec;
/// the activation derivative product (1 relin) of every activated layer; the gradient
/// `(n2-1) + m log B` rotations and `m` relins. `acts[i]`: layer `i` has an activation.
fn cost_model_interactive(dims: &[usize], acts: &[bool], batch: usize) -> Ks {
    let mut k = Ks::default();
    for (i, p) in dims.windows(2).enumerate() {
        let (din, dout) = (p[0].next_power_of_two(), p[1].next_power_of_two());
        let m = din.min(dout);
        let (n1, n2) = spike_split(m);
        let f_fwd = if din > dout { log2(din / dout) } else { 0 };
        let f_bwd = if dout > din { log2(dout / din) } else { 0 };
        k.add((n1 - 1) + (n2 - 1) + f_fwd, n2);
        if acts[i] {
            k.add(0, 2); // forward activation product and derivative product
        }
        if i > 0 {
            k.add(m - 1, 0); // transpose
            k.add((n1 - 1) + (n2 - 1) + f_bwd, n2);
        }
        k.add((n2 - 1) + m * log2(batch), m);
    }
    k
}

/// The documented corrections from the cost model to this implementation (one update):
/// `docs/layout.md` §5 items 1, 2 and 5, plus the bias gradients and the BSGS split.
#[derive(Debug, Clone, Copy, Default)]
struct Terms {
    /// §5.1: the server replicates the batch sum back to every lane: `+ m log B` rotations.
    replication: u64,
    /// §5.2: the gradient recomputes the layer input's baby rotations: `+ (n1 - 1)`.
    babies: u64,
    /// Not in the model: the bias gradient's batch sum, `2 log B` rotations per layer.
    bias: u64,
    /// The library splits BSGS as `baby = 2^ceil(log2 m / 2)`, the model minimises key
    /// switches with ties to the larger `n1`: equal matvec key switches, different gradient
    /// rotation counts (`(baby-1)+(giant-1)` vs `(n1-1)+(n2-1)`). Signed.
    split: i64,
    /// §5.5: the transpose's rotations beyond the model's `m - 1` (0 with dedicated keys).
    transpose: u64,
}

impl Terms {
    fn total(&self) -> i64 {
        (self.replication + self.babies + self.bias + self.transpose) as i64 + self.split
    }
}

/// The exact key switches of one [`Trainer::train_batch_valid`] step of the MLP `dims` with
/// activations `acts` (`true`: a polynomial activation with a ciphertext derivative, `false`:
/// the identity), from the library's layer shapes and transpose plan, and the corrections
/// from [`cost_model_interactive`] to it.
fn predict(
    dims: &[usize],
    acts: &[bool],
    layout: &Layout,
    keys: &TransposeKeys,
) -> Result<(Ks, Terms), String> {
    let b = layout.batch;
    let lb = log2(b);
    let mut k = Ks::default();
    let mut t = Terms::default();
    for (i, p) in dims.windows(2).enumerate() {
        let sh = LayerShape::new(p[1], p[0], layout);
        let (n1, n2) = spike_split(sh.m);
        // forward
        k.add(sh.baby + sh.giant - 2 + sh.fold_steps(), sh.giant);
        if acts[i] {
            k.add(0, 1);
        }
        // dz (ct-ct only for a ciphertext derivative)
        if acts[i] {
            k.add(0, 1);
        }
        if i > 0 {
            let plan =
                transpose_rotation_plan(p[1], p[0], layout, keys).map_err(|e| e.to_string())?;
            let tr: usize = plan.iter().map(|(_, s)| s.len()).sum();
            k.add(tr, 0);
            t.transpose += (tr - (sh.m - 1)) as u64;
            let st = sh.transposed();
            k.add(st.baby + st.giant - 2 + st.fold_steps(), st.giant);
        }
        // weight gradient and bias gradient
        k.add((sh.baby - 1) + (sh.giant - 1) + 2 * sh.m * lb, sh.m);
        k.add(2 * lb, 0);
        t.replication += (sh.m * lb) as u64;
        t.babies += (n1 - 1) as u64;
        t.bias += (2 * lb) as u64;
        t.split += ((sh.baby - 1) + (sh.giant - 1)) as i64 - ((n1 - 1) + (n2 - 1)) as i64;
    }
    Ok((k, t))
}

// ---------------------------------------------------------------------------------------------
// The plaintext twin.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Layer {
    w: Array2<f64>,
    b: Vec<f64>,
}

fn affine(a: &Array2<f64>, l: &Layer) -> Array2<f64> {
    let mut z = a.dot(&l.w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
    }
    z
}

/// `z^2 + z` on every layer but the last (identity).
fn forward_plain(net: &[Layer], x: &Array2<f64>) -> Array2<f64> {
    let mut a = x.clone();
    for (i, l) in net.iter().enumerate() {
        let z = affine(&a, l);
        a = if i + 1 == net.len() {
            z
        } else {
            z.mapv(|v| v * v + v)
        };
    }
    a
}

/// One plaintext NAG step, the trainer's update rule: `delta = a - y`, `G = dz^T a_in / n`,
/// `V' = mu V + G`, `W' = W - lr (G + mu V')`, biases `b' = b - lr sum(dz) / n`.
fn twin_step(
    net: &mut [Layer],
    vel: &mut [Array2<f64>],
    x: &Array2<f64>,
    y: &Array2<f64>,
    lr: f64,
    mu: f64,
) {
    let n = x.nrows() as f64;
    let last = net.len() - 1;
    let mut acts = vec![x.clone()];
    let mut zs = Vec::new();
    for (i, l) in net.iter().enumerate() {
        let z = affine(acts.last().expect("input"), l);
        acts.push(if i == last {
            z.clone()
        } else {
            z.mapv(|v| v * v + v)
        });
        zs.push(z);
    }
    let mut delta = acts.last().expect("output") - y;
    for i in (0..net.len()).rev() {
        let dz = if i == last {
            delta.clone()
        } else {
            &delta * &zs[i].mapv(|v| 2.0 * v + 1.0)
        };
        let next = dz.dot(&net[i].w);
        let g = dz.t().dot(&acts[i]) / n;
        vel[i] = mu * &vel[i] + &g;
        net[i].w = &net[i].w - &(lr * &(&g + &(mu * &vel[i])));
        let gb = dz.sum_axis(Axis(0)) / n;
        net[i].b.iter_mut().zip(&gb).for_each(|(b, g)| *b -= lr * g);
        delta = next;
    }
}

fn accuracy(out: &Array2<f64>, labels: &[u8]) -> f64 {
    let hits = out
        .rows()
        .into_iter()
        .zip(labels)
        .filter(|(row, l)| {
            let arg = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(j, _)| j)
                .expect("nonempty row");
            arg == **l as usize
        })
        .count();
    hits as f64 / labels.len().max(1) as f64
}

fn mse(out: &Array2<f64>, y: &Array2<f64>) -> f64 {
    (out - y).mapv(|v| v * v).mean().unwrap_or(0.0)
}

fn max_abs(a: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter().map(f64::abs).fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------------------------
// Output.
// ---------------------------------------------------------------------------------------------

/// A minimal JSON value (no dependency).
enum J {
    N(f64),
    I(i64),
    S(String),
    B(bool),
    A(Vec<J>),
    O(Vec<(String, J)>),
}

impl J {
    fn write(&self, out: &mut String, ind: usize) {
        let pad = |n: usize| "  ".repeat(n);
        match self {
            J::N(v) if v.is_finite() => write!(out, "{v}").unwrap(),
            J::N(_) => out.push_str("null"),
            J::I(v) => write!(out, "{v}").unwrap(),
            J::S(s) => json_string(out, s),
            J::B(b) => write!(out, "{b}").unwrap(),
            J::A(v) => {
                if v.iter().all(|x| matches!(x, J::N(_) | J::I(_))) {
                    out.push('[');
                    for (i, x) in v.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        x.write(out, ind);
                    }
                    out.push(']');
                } else {
                    out.push_str("[\n");
                    for (i, x) in v.iter().enumerate() {
                        out.push_str(&pad(ind + 1));
                        x.write(out, ind + 1);
                        out.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                    }
                    out.push_str(&pad(ind));
                    out.push(']');
                }
            }
            J::O(kv) => {
                out.push_str("{\n");
                for (i, (k, v)) in kv.iter().enumerate() {
                    write!(out, "{}\"{k}\": ", pad(ind + 1)).unwrap();
                    v.write(out, ind + 1);
                    out.push_str(if i + 1 < kv.len() { ",\n" } else { "\n" });
                }
                out.push_str(&pad(ind));
                out.push('}');
            }
        }
    }
}

/// `s` as a JSON string literal (RFC 8259 escapes; Rust's `Debug` would write `\u{..}`).
fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn obj(kv: Vec<(&str, J)>) -> J {
    J::O(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn ks_json(k: &Ks, samples: f64) -> J {
    obj(vec![
        ("rotations", J::I(k.rot as i64)),
        ("relins", J::I(k.relin as i64)),
        ("key_switches", J::I(k.total() as i64)),
        ("rotations_per_sample", J::N(k.rot as f64 / samples)),
        ("relins_per_sample", J::N(k.relin as f64 / samples)),
        ("key_switches_per_sample", J::N(k.total() as f64 / samples)),
    ])
}

fn counter_json(c: &OpCounter) -> J {
    obj(vec![
        ("rotations", J::I(c.rotations as i64)),
        ("relins", J::I(c.relins as i64)),
        ("key_switches", J::I(c.key_switches() as i64)),
        ("ct_ct_tensor", J::I(c.ct_ct_tensor as i64)),
        ("ct_pt_mults", J::I(c.ct_pt_mults as i64)),
        ("rescales", J::I(c.rescales as i64)),
    ])
}

/// The load averages: `sysctl -n vm.loadavg` (macOS), else `/proc/loadavg` (Linux), else
/// `"unknown"`.
fn loadavg() -> String {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .or_else(|| std::fs::read_to_string("/proc/loadavg").ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// This process's peak resident set size in bytes (`getrusage(RUSAGE_SELF)`; `ru_maxrss` is
/// in bytes on macOS and in KiB on Linux), or `None` if the call fails.
fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: `getrusage` only writes the `rusage` struct it is given, which is plain data
    // and fully initialised by `zeroed`.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    if rc != 0 {
        return None;
    }
    let raw = u64::try_from(ru.ru_maxrss).ok()?;
    Some(if cfg!(target_os = "macos") {
        raw
    } else {
        raw * 1024
    })
}

/// How this binary was built (from compile-time facts only).
fn build_info() -> J {
    obj(vec![
        ("debug_assertions", J::B(cfg!(debug_assertions))),
        ("package_version", J::S(env!("CARGO_PKG_VERSION").into())),
        ("target_os", J::S(std::env::consts::OS.into())),
        ("target_arch", J::S(std::env::consts::ARCH.into())),
    ])
}

fn stats(v: &[f64]) -> J {
    let n = v.len().max(1) as f64;
    obj(vec![
        ("mean", J::N(v.iter().sum::<f64>() / n)),
        ("min", J::N(v.iter().copied().fold(f64::INFINITY, f64::min))),
        (
            "max",
            J::N(v.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        ),
    ])
}

// ---------------------------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------------------------

fn rng(seed: u8, role: u8) -> Rng {
    let mut s = [seed; 32];
    s[0] = role;
    Rng::from_seed(s)
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn init_net(dims: &[usize], r: &mut Rng) -> Vec<Layer> {
    dims.windows(2)
        .map(|p| {
            let bound = 1.0 / (p[0] as f64).sqrt();
            Layer {
                w: Array2::from_shape_fn((p[1], p[0]), |_| bound * uni(r)),
                b: (0..p[1]).map(|_| bound * uni(r)).collect(),
            }
        })
        .collect()
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("mnist: {e}");
            std::process::exit(2);
        }
    };
    let r = if args.mode == "noninteractive" {
        run_ni(&args)
    } else {
        run(&args)
    };
    if let Err(e) = r {
        eprintln!("mnist: {e}");
        std::process::exit(1);
    }
}

fn load(args: &Args) -> Result<(Mnist, Mnist), String> {
    let hint = |e: std::io::Error| format!("{e} (run scripts/fetch_mnist.sh)");
    let train = load_mnist(&args.data, Split::Train, Some(args.subset)).map_err(hint)?;
    let test = load_mnist(&args.data, Split::Test, Some(args.test_subset)).map_err(hint)?;
    if train.features() != test.features() {
        return Err("train and test images differ in size".into());
    }
    Ok((train, test))
}

fn run(args: &Args) -> Result<(), String> {
    let load_start = loadavg();
    let t_all = Instant::now();
    // `--plan` does not need the data: without the files it plans for 28 x 28 images.
    let loaded = load(args);
    let features = match (&loaded, args.plan) {
        (Ok((train, _)), _) => train.features(),
        (Err(e), true) => {
            eprintln!("mnist: --plan: data not loaded ({e}); planning for 28 x 28 images");
            28 * 28
        }
        (Err(e), false) => return Err(e.clone()),
    };
    let params = Arc::new(params_named(&args.params)?);
    let dims = [features, args.hidden, CLASSES];
    let acts_flags = [true, false];
    let d = dims
        .iter()
        .map(|x| x.next_power_of_two())
        .max()
        .expect("dims");
    let layout = Layout::new(params.slots(), d, args.batch).map_err(|e| e.to_string())?;
    let keys = match args.transpose.as_str() {
        "reuse" => TransposeKeys::Reuse(mlp_key_pool(&dims, &layout).map_err(|e| e.to_string())?),
        "dedicated" => TransposeKeys::Dedicated,
        t => return Err(format!("unknown --transpose {t} (reuse | dedicated)")),
    };
    let steps = mlp_rotation_steps(&dims, &layout, &keys).map_err(|e| e.to_string())?;
    let limbs = params.q().len() + params.p().len();
    let key_bytes = params.dnum() * 2 * limbs * params.n() * 8;
    let (pred, terms) = predict(&dims, &acts_flags, &layout, &keys)?;
    let model = cost_model_interactive(&dims, &acts_flags, layout.batch);
    // docs/layout.md §5 items 1, 2 and 5 (written before this task): replication, recomputed
    // babies, transpose keys. The bias term (§3) and the BSGS split are not in §5.
    let pre_registered = model.total() + terms.replication + terms.babies + terms.transpose;
    let b = layout.batch as f64;
    eprintln!(
        "mnist: {} N = 2^{} ({} slots, {} levels), dims {dims:?}, layout d = {d}, B = {}, \
         transpose {}: {} Galois keys + relin ({:.1} GB at {:.0} MB each)",
        args.params,
        params.log_n(),
        params.slots(),
        params.max_level(),
        layout.batch,
        args.transpose,
        steps.len(),
        ((steps.len() + 1) * key_bytes) as f64 / 1e9,
        key_bytes as f64 / 1e6
    );
    eprintln!(
        "mnist: predicted per update {} rot + {} relin = {} KS ({:.3}/sample); cost model \
         (interactive) {} KS ({:.3}/sample); corrections {terms:?}",
        pred.rot,
        pred.relin,
        pred.total(),
        pred.total() as f64 / b,
        model.total(),
        model.total() as f64 / b
    );
    let layout_json = obj(vec![
        ("params", J::S(args.params.clone())),
        ("log_n", J::I(params.log_n() as i64)),
        ("slots", J::I(params.slots() as i64)),
        ("max_level", J::I(params.max_level() as i64)),
        ("scale_bits", J::N(params.scale().log2())),
        ("dims", J::A(dims.iter().map(|&x| J::I(x as i64)).collect())),
        ("layout_d", J::I(d as i64)),
        ("lanes_b", J::I(layout.batch as i64)),
        ("ciphertexts_per_batch", J::I(1)),
        ("transpose_keys", J::S(args.transpose.clone())),
        ("galois_keys", J::I(steps.len() as i64)),
        (
            "galois_steps",
            J::A(steps.iter().map(|&s| J::I(s as i64)).collect()),
        ),
        ("key_bytes_each", J::I(key_bytes as i64)),
        (
            "key_memory_gb",
            J::N(((steps.len() + 1) * key_bytes) as f64 / 1e9),
        ),
    ]);
    let prediction_json = obj(vec![
        ("cost_model_interactive", ks_json(&model, b)),
        (
            "corrections",
            obj(vec![
                ("replication_m_logB", J::I(terms.replication as i64)),
                ("recomputed_babies", J::I(terms.babies as i64)),
                ("bias_gradients", J::I(terms.bias as i64)),
                ("bsgs_split", J::I(terms.split)),
                ("transpose_beyond_m_minus_1", J::I(terms.transpose as i64)),
            ]),
        ),
        (
            "replication_adjusted_key_switches",
            J::I((model.total() + terms.replication) as i64),
        ),
        (
            "pre_registered_adjusted_key_switches",
            J::I(pre_registered as i64),
        ),
        (
            "cost_model_plus_all_corrections",
            J::I(model.total() as i64 + terms.total()),
        ),
        ("exact", ks_json(&pred, b)),
    ]);
    if args.plan {
        let mut s = String::new();
        obj(vec![
            ("setup", layout_json),
            ("prediction_per_update", prediction_json),
        ])
        .write(&mut s, 0);
        println!("{s}");
        return Ok(());
    }
    let (train, test) = loaded?;

    // Keys and the client.
    let t_keys = Instant::now();
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), rng(args.seed, 1)).map_err(|e| e.to_string())?;
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
    let encoder = Encoder::new(Arc::clone(&params));
    let ev = Evaluator::new(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    drop(kg);
    let keygen_s = t_keys.elapsed().as_secs_f64();
    eprintln!("mnist: keys in {keygen_s:.1} s");

    // Network and twin.
    let mut r = rng(args.seed, 2);
    let init = init_net(&dims, &mut r);
    let plain: Vec<(Array2<f64>, Vec<f64>)> =
        init.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(PolyRelu2), Box::new(Identity)];
    let t_enc = Instant::now();
    let mut mlp = Mlp::encrypt(&ev, &plain, acts, &layout, &enc, &encoder, &mut r)
        .map_err(|e| e.to_string())?;
    let spec = mlp.batch_spec(&ev, &encoder).map_err(|e| e.to_string())?;
    let encrypt_net_s = t_enc.elapsed().as_secs_f64();
    let client = ClientRefresher::new(
        dec.clone(),
        enc.clone(),
        Encoder::new(Arc::clone(&params)),
        rng(args.seed, 3),
    );
    let optims: Vec<Box<dyn LayerOptimizer>> = vec![
        Box::new(Nag::new(args.lr, args.momentum)),
        Box::new(Nag::new(args.lr, args.momentum)),
    ];
    let mut trainer = Trainer::new(&ev, Box::new(client), optims, layout)
        .map_err(|e| e.to_string())?
        .with_transpose_keys(keys);
    let mut twin = init.clone();
    let mut vel: Vec<Array2<f64>> = twin.iter().map(|l| Array2::zeros(l.w.dim())).collect();
    let y_train = one_hot(&train.labels, CLASSES);

    // Training.
    let n = train.len();
    let mut batch_s = Vec::new();
    let mut refresh_s = Vec::new();
    let mut client_enc_s = Vec::new();
    let mut per_batch: Vec<OpCounter> = Vec::new();
    let mut total = OpCounter::default();
    let mut samples = 0usize;
    let t_train = Instant::now();
    for epoch in 0..args.epochs {
        for start in (0..n).step_by(layout.batch) {
            let end = (start + layout.batch).min(n);
            let x = train.images.slice(s![start..end, ..]).to_owned();
            let y = y_train.slice(s![start..end, ..]).to_owned();
            let t0 = Instant::now();
            let (xct, yct) = spec
                .encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
                .map_err(|e| e.to_string())?;
            client_enc_s.push(t0.elapsed().as_secs_f64());
            let log = trainer
                .train_epoch_valid(&mut mlp, &[(xct, yct, end - start)])
                .map_err(|e| format!("batch at {start}: {e}"))?;
            twin_step(&mut twin, &mut vel, &x, &y, args.lr, args.momentum);
            batch_s.push(log.seconds);
            refresh_s.push(log.refresh_seconds);
            let c = log.counter;
            total += &c;
            samples += end - start;
            eprintln!(
                "mnist: epoch {epoch} batch {} ({} samples): {:.1} s (refresh {:.1} s), \
                 {} rot + {} relin; elapsed {:.0} s, load {}",
                per_batch.len(),
                end - start,
                log.seconds,
                log.refresh_seconds,
                c.rotations,
                c.relins,
                t_all.elapsed().as_secs_f64(),
                loadavg()
            );
            per_batch.push(c);
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();

    // Results.
    let got: Vec<Layer> = mlp
        .layers
        .iter()
        .map(|(dn, _)| Layer {
            w: dn.w.decrypt(&dec, &encoder, &layout),
            b: unpack_activations(&encoder.decode(&dec.decrypt(&dn.b)), &layout, dn.w.d_out())
                .row(0)
                .to_vec(),
        })
        .collect();
    let w_diff: Vec<f64> = got
        .iter()
        .zip(&twin)
        .map(|(g, t)| max_abs(g.w.iter().zip(&t.w).map(|(a, b)| a - b)))
        .collect();
    let b_diff: Vec<f64> = got
        .iter()
        .zip(&twin)
        .map(|(g, t)| max_abs(g.b.iter().zip(&t.b).map(|(a, b)| a - b)))
        .collect();
    let w_moved: Vec<f64> = twin
        .iter()
        .zip(&init)
        .map(|(t, i)| max_abs(t.w.iter().zip(&i.w).map(|(a, b)| a - b)))
        .collect();
    let max_diff = w_diff.iter().chain(&b_diff).copied().fold(0.0, f64::max);
    let y_test = one_hot(&test.labels, CLASSES);
    let out_he = forward_plain(&got, &test.images);
    let out_twin = forward_plain(&twin, &test.images);
    let out_init = forward_plain(&init, &test.images);
    let samples_f = samples as f64;
    let batches = per_batch.len();
    let constant = per_batch.iter().all(|c| {
        (c.rotations, c.relins) == (pred.rot, pred.relin) && c.key_switches() == pred.total()
    });
    // Every key switch (OpCounter::key_switches: rotations, relins and SSE switches; there are
    // no bootstraps in interactive mode).
    let measured = Ks {
        rot: total.rotations,
        relin: total.key_switches() - total.rotations,
    };
    let replication_adjusted = (model.total() + terms.replication) as f64 * batches as f64;
    let pre_registered_total = pre_registered as f64 * batches as f64;
    eprintln!(
        "mnist: {batches} batches, {samples} samples in {train_s:.1} s; {} KS ({:.3}/sample), \
         prediction per batch matched: {constant}; max |W_he - W_twin| = {max_diff:e}",
        measured.total(),
        measured.total() as f64 / samples_f
    );
    let result = obj(vec![
        (
            "task",
            J::S("Task 32: MNIST eMLP-1 interactive training (RQ1 counts)".into()),
        ),
        (
            "network",
            J::S(format!(
                "{} -> {} (z^2 + z) -> {} (linear), MSE, NAG (biases SGD)",
                dims[0], dims[1], dims[2]
            )),
        ),
        ("setup", layout_json),
        (
            "training",
            obj(vec![
                ("epochs", J::I(args.epochs as i64)),
                ("subset", J::I(n as i64)),
                ("batches", J::I(batches as i64)),
                ("samples", J::I(samples as i64)),
                ("lr", J::N(args.lr)),
                ("momentum", J::N(args.momentum)),
                ("weight_decay", J::N(0.0)),
                ("seed", J::I(args.seed as i64)),
                ("init", J::S("uniform(+-1/sqrt(fan_in)) for W and b".into())),
            ]),
        ),
        (
            "measured",
            obj(vec![
                ("total", counter_json(&total)),
                ("per_sample", ks_json(&measured, samples_f)),
                ("per_batch_first", counter_json(&per_batch[0])),
                ("every_batch_equals_prediction", J::B(constant)),
            ]),
        ),
        ("prediction_per_update", prediction_json),
        (
            "measured_over_replication_adjusted",
            J::N(measured.total() as f64 / replication_adjusted),
        ),
        (
            "measured_over_pre_registered_adjusted",
            J::N(measured.total() as f64 / pre_registered_total),
        ),
        (
            "measured_over_cost_model_interactive",
            J::N(measured.total() as f64 / (model.total() as f64 * batches as f64)),
        ),
        (
            "time_s",
            obj(vec![
                ("keygen", J::N(keygen_s)),
                ("encrypt_network", J::N(encrypt_net_s)),
                ("epoch_total", J::N(train_s)),
                ("per_batch_step_incl_refresh", stats(&batch_s)),
                ("per_batch_refresh", stats(&refresh_s)),
                ("per_batch_client_encrypt", stats(&client_enc_s)),
                ("wall_total", J::N(t_all.elapsed().as_secs_f64())),
            ]),
        ),
        (
            "twin",
            obj(vec![
                (
                    "max_abs_w_diff_per_layer",
                    J::A(w_diff.iter().map(|&v| J::N(v)).collect()),
                ),
                (
                    "max_abs_b_diff_per_layer",
                    J::A(b_diff.iter().map(|&v| J::N(v)).collect()),
                ),
                ("max_abs_diff", J::N(max_diff)),
                ("log2_max_abs_diff", J::N(max_diff.log2())),
                (
                    "max_abs_w_update_per_layer",
                    J::A(w_moved.iter().map(|&v| J::N(v)).collect()),
                ),
            ]),
        ),
        (
            "evaluation",
            obj(vec![
                ("test_samples", J::I(test.len() as i64)),
                (
                    "test_acc_encrypted_trained",
                    J::N(accuracy(&out_he, &test.labels)),
                ),
                ("test_acc_twin", J::N(accuracy(&out_twin, &test.labels))),
                ("test_acc_init", J::N(accuracy(&out_init, &test.labels))),
                ("test_mse_encrypted_trained", J::N(mse(&out_he, &y_test))),
                ("test_mse_twin", J::N(mse(&out_twin, &y_test))),
                ("test_mse_init", J::N(mse(&out_init, &y_test))),
            ]),
        ),
        (
            "levels_last_step",
            J::O(
                trainer
                    .trace()
                    .iter()
                    .map(|(k, v)| (k.clone(), J::I(*v as i64)))
                    .collect(),
            ),
        ),
        (
            "memory",
            obj(vec![
                (
                    "peak_rss_bytes",
                    peak_rss_bytes().map_or(J::N(f64::NAN), |b| J::I(b as i64)),
                ),
                (
                    "source",
                    J::S("getrusage(RUSAGE_SELF) at the end of the run".into()),
                ),
            ]),
        ),
        (
            "machine",
            obj(vec![
                ("load_start", J::S(load_start)),
                ("load_end", J::S(loadavg())),
                ("build", build_info()),
                (
                    "threads",
                    J::S("single (the library is single-threaded)".into()),
                ),
            ]),
        ),
    ]);
    let mut s = String::new();
    result.write(&mut s, 0);
    s.push('\n');
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from("results/mnist_emlp1.json"));
    write_out(&out, &s)
}

fn write_out(out: &std::path::Path, s: &str) -> Result<(), String> {
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(out, s).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!("mnist: wrote {}", out.display());
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Task 41: non-interactive mode.
// ---------------------------------------------------------------------------------------------

/// The stdout of `cmd args` (trimmed), or `"unknown"`.
fn cmd_out(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Seconds since the Unix epoch.
fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(f64::NAN, |d| d.as_secs_f64())
}

/// Where a result comes from, written by the binary itself: the git commit and the files that
/// differ from it, the command line, the build, the machine, the threads and the load.
fn provenance(load_start: &str, started_unix: f64) -> J {
    let threads = std::thread::available_parallelism().map_or(0, |n| n.get());
    let rayon = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    let dirty = cmd_out("git", &["status", "--porcelain", "--untracked-files=no"]);
    obj(vec![
        (
            "written_by",
            J::S("the mnist binary (raw output, not assembled)".into()),
        ),
        ("git_commit", J::S(cmd_out("git", &["rev-parse", "HEAD"]))),
        (
            "git_modified_tracked_files",
            J::S(if dirty == "unknown" {
                "none".into()
            } else {
                dirty
            }),
        ),
        (
            "command_line",
            J::S(std::env::args().collect::<Vec<_>>().join(" ")),
        ),
        ("build", build_info()),
        ("parallel_feature", J::B(cfg!(feature = "parallel"))),
        ("rustc_on_path", J::S(cmd_out("rustc", &["-V"]))),
        (
            "machine",
            obj(vec![
                (
                    "cpu",
                    J::S(cmd_out("sysctl", &["-n", "machdep.cpu.brand_string"])),
                ),
                ("logical_cpus", J::I(threads as i64)),
                (
                    "memory_bytes",
                    J::S(cmd_out("sysctl", &["-n", "hw.memsize"])),
                ),
                ("os", J::S(cmd_out("uname", &["-srm"]))),
                ("host", J::S(cmd_out("hostname", &["-s"]))),
            ]),
        ),
        (
            "threads",
            J::S(if cfg!(feature = "parallel") {
                format!(
                    "rayon (feature parallel), RAYON_NUM_THREADS = {rayon}, {threads} logical CPUs"
                )
            } else {
                "1 (no parallel feature)".into()
            }),
        ),
        ("load_start", J::S(load_start.into())),
        ("load_end", J::S(loadavg())),
        ("started_unix", J::N(started_unix)),
        ("finished_unix", J::N(unix_now())),
    ])
}

/// ReBoot's bootstrap key switches in the M0 cost model: OpenFHE `EvalBootstrap` with
/// Meta-BTS, 2 iterations (`spikes/costmodel/costmodel.py` `bootstrap_key_switches(2^15)`:
/// 436 linear-transform rotations + 2 conjugations + 64 EvalMod relins; `spikes/RESULTS.md`
/// Task 6, `docs/bootstrap.md` §11.2).
const REBOOT_BOOT_KS: u64 = 502;
/// hekit's measured bootstrap key switches on n16_boot (`docs/bootstrap.md` §11.2), the
/// model's `HEKIT_BOOT_KS` for the prediction before keys exist.
const HEKIT_BOOT_KS: u64 = 111;
/// ReBoot's key switches per sample without bootstraps (M0 cost model, ReBoot as shipped at
/// N = 2^16, eMLP-1; independent of B) and its bootstraps per update (W and V of 2 layers).
const REBOOT_KS_PER_SAMPLE: f64 = 37.0;
const REBOOT_BOOTS_PER_UPDATE: f64 = 4.0;
/// ReBoot's own batch size (Table 8).
const REBOOT_BATCH: usize = 48;

/// The spike's transpose count (`costmodel._transpose_rotations`) with its BSGS split.
fn spike_transpose_rotations(din: usize, dout: usize, n1: usize) -> usize {
    let (m, l) = (din.min(dout), din.max(dout));
    (0..m)
        .filter(|&i| {
            let (c, t) = if din >= dout {
                let c = (dout - i % dout) % dout;
                (c, -(c as i64))
            } else {
                let c = (din - i % din) % din;
                (c, i as i64)
            };
            let amount = (t - ((i / n1) * n1) as i64 + ((c / n1) * n1) as i64).rem_euclid(l as i64);
            amount != 0
        })
        .count()
}

/// `spikes/costmodel/costmodel.py` `network_cost("diag_batched", layers, batch, slots)`
/// (non-interactive, `compact=True`, `momentum=True`) for one update: rotations, relins,
/// bootstraps, the ciphertexts per batch `k` and the lanes `B'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NiModel {
    ks: Ks,
    boots: u64,
    k: usize,
    lanes: usize,
}

fn cost_model_ni(dims: &[usize], acts: &[bool], batch: usize, slots: usize) -> NiModel {
    let lmax = dims
        .iter()
        .map(|d| d.next_power_of_two())
        .max()
        .expect("dims");
    let bp = batch.next_power_of_two().min(slots / lmax);
    let k = batch.div_ceil(bp);
    let mut ks = Ks::default();
    let mut boots = 0;
    for (i, p) in dims.windows(2).enumerate() {
        let (din, dout) = (p[0].next_power_of_two(), p[1].next_power_of_two());
        let (m, l) = (din.min(dout), din.max(dout));
        let (n1, n2) = spike_split(m);
        let f_fwd = if din > dout { log2(din / dout) } else { 0 };
        let f_bwd = if dout > din { log2(dout / din) } else { 0 };
        let nw = m.div_ceil(slots / l);
        ks.add(k * ((n1 - 1) + (n2 - 1) + f_fwd), k * n2);
        if acts[i] {
            ks.add(0, 2 * k); // forward activation and derivative product
        }
        if i > 0 {
            ks.add(spike_transpose_rotations(din, dout, n1), 0);
            ks.add(k * ((n1 - 1) + (n2 - 1) + f_bwd), k * n2);
        }
        ks.add(k * (n2 - 1) + m * log2(bp) + (m - nw), m);
        ks.add((m - nw) + m * log2(slots / l), 0);
        boots += 2 * nw as u64;
    }
    NiModel {
        ks,
        boots,
        k,
        lanes: bp,
    }
}

/// The corrections from [`cost_model_ni`] to the implementation's closed form
/// ([`predict_update_for`]), one update, key switches (Task 32's terms where they apply).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct NiTerms {
    /// The gradient recomputes the layer input's baby rotations: `k (n1 - 1)` per layer.
    babies: i64,
    /// The library's BSGS split against the model's (gradient `(b-1)+(g-1)` against
    /// `(n1-1)+(n2-1)`; the matvec key switches are equal).
    split: i64,
    /// The transpose's rotations beyond the model's count (Reuse keys).
    transpose: i64,
    /// Not in the model: the bias gradients (reduce `log2 B` and the shift to the compact
    /// position) and the bias extraction (shift and `log2(S / L_host)` replications).
    bias: i64,
    /// Bootstraps beyond the model's `2 nW` per update (activation bootstraps, bias-only
    /// containers, the lazy schedule), times the bootstrap's key switches.
    boots: i64,
}

impl NiTerms {
    fn total(&self) -> i64 {
        self.babies + self.split + self.transpose + self.bias + self.boots
    }
}

fn lib_acts(acts: &[bool]) -> Vec<ActCost> {
    acts.iter()
        .map(|&a| {
            if a {
                ActCost::QUADRATIC
            } else {
                ActCost::IDENTITY
            }
        })
        .collect()
}

/// The closed form of one update and its corrections from the model, for `boots_per_update`
/// weight bootstraps (2 nW for AfterUpdate; the lazy average otherwise) at `boot_ks`.
#[allow(clippy::too_many_arguments)]
fn ni_terms(
    dims: &[usize],
    acts: &[bool],
    layout: &Layout,
    keys: &TransposeKeys,
    placement: BootPlacement,
    model: &NiModel,
    weight_boots_per_update: f64,
    boot_ks: u64,
) -> Result<(UpdateCost, NiTerms, f64), String> {
    let cost = predict_update_for(
        dims,
        &lib_acts(acts),
        layout,
        keys,
        placement,
        model.k,
        true,
        true,
    )
    .map_err(|e| e.to_string())?;
    let k = model.k as i64;
    let mut t = NiTerms::default();
    let lb = layout.log_batch() as i64;
    let cl = CompactLayout::new(dims, layout).map_err(|e| e.to_string())?;
    for (i, p) in dims.windows(2).enumerate() {
        let sh = LayerShape::new(p[1], p[0], layout);
        let (n1, n2) = spike_split(sh.m);
        t.babies += k * (n1 as i64 - 1);
        let lib_grad = (sh.baby - 1 + sh.giant - 1) as i64;
        t.split += k * (lib_grad - (n1 + n2 - 2) as i64);
        // Forward and backward matvecs: key switches equal under both splits (asserted).
        let fwd = |b: usize, g: usize, f: usize| (b + g - 2 + f + g) as i64;
        assert_eq!(
            fwd(sh.baby, sh.giant, sh.fold_steps()),
            fwd(n1, n2, sh.fold_steps()),
            "matvec key switches differ between the splits"
        );
        if i > 0 {
            let plan =
                transpose_rotation_plan(p[1], p[0], layout, keys).map_err(|e| e.to_string())?;
            let tr: usize = plan.iter().map(|(_, s)| s.len()).sum();
            let (din, dout) = (p[0].next_power_of_two(), p[1].next_power_of_two());
            t.transpose += tr as i64 - spike_transpose_rotations(din, dout, n1) as i64;
        }
        let bp = cl.bias_plan(i);
        let o = i64::from(bp.offset(cl.biases[i].position) != 0);
        t.bias += lb + o + o + bp.diags_per_ct.trailing_zeros() as i64;
    }
    let lib_boots = weight_boots_per_update + cost.activation_bootstraps as f64;
    let extra = (lib_boots - model.boots as f64) * boot_ks as f64;
    t.boots = extra.round() as i64;
    Ok((cost, t, lib_boots))
}

fn ni_cost_json(c: &UpdateCost) -> J {
    obj(vec![
        ("forward_rot", J::I(c.forward_rot as i64)),
        ("forward_relin", J::I(c.forward_relin as i64)),
        ("transpose_rot", J::I(c.transpose_rot as i64)),
        ("backward_rot", J::I(c.backward_rot as i64)),
        ("backward_relin", J::I(c.backward_relin as i64)),
        ("gradient_rot", J::I(c.gradient_rot as i64)),
        ("gradient_relin", J::I(c.gradient_relin as i64)),
        ("shift_rot", J::I(c.shift_rot as i64)),
        ("bias_grad_rot", J::I(c.bias_grad_rot as i64)),
        ("unpack_rot", J::I(c.unpack_rot as i64)),
        ("bias_extract_rot", J::I(c.bias_extract_rot as i64)),
        ("rotations", J::I(c.rotations() as i64)),
        ("relins", J::I(c.relins() as i64)),
        (
            "weight_bootstraps_when_booted",
            J::I(c.weight_bootstraps as i64),
        ),
        (
            "activation_bootstraps",
            J::I(c.activation_bootstraps as i64),
        ),
    ])
}

fn ni_counter_json(c: &OpCounter) -> J {
    obj(vec![
        ("rotations", J::I(c.rotations as i64)),
        ("relins", J::I(c.relins as i64)),
        ("sse_switches", J::I(c.sse_switches as i64)),
        ("key_switches", J::I(c.key_switches() as i64)),
        ("bootstraps", J::I(c.bootstraps as i64)),
        ("ct_ct_tensor", J::I(c.ct_ct_tensor as i64)),
        ("ct_pt_mults", J::I(c.ct_pt_mults as i64)),
        ("rescales", J::I(c.rescales as i64)),
    ])
}

/// The RQ1 ratio three ways (parent's reporting rule): hekit's key switches per sample with
/// its bootstraps priced at `p_he`, over ReBoot's at `p_rb`, against ReBoot at its own B = 48
/// and at hekit's B.
/// One ReBoot-side baseline of the RQ1 ratio: its key switches per sample without bootstraps
/// and where the number comes from.
struct RebootSide {
    name: &'static str,
    ks_per_sample: f64,
    source: String,
}

/// The ReBoot-side baselines: Task 33's measured in-library ReBoot layout (Lazy, the headline,
/// and PerSample), read from `results/mnist_reboot_layout.json` when present, and the M0 cost
/// model's ReBoot as shipped. Bootstraps are not in any of them (Task 33 ran with hekit's
/// interactive client): the ratio adds ReBoot's 4 bootstraps per update (W and V of 2 layers,
/// M0 model) at the way's bootstrap cost.
fn reboot_sides() -> Vec<RebootSide> {
    let mut out = Vec::new();
    if let Ok(json) = std::fs::read_to_string(REBOOT_MEASURED) {
        for (name, mode) in [
            ("lazy_measured", "\"lazy\""),
            ("persample_measured", "\"persample\""),
        ] {
            let v = json.find(mode).and_then(|at| {
                let key = "\"key_switches_per_sample\":";
                let k = json[at..].find(key)? + at + key.len();
                let end = json[k..].find([',', '\n', '}'])? + k;
                json[k..end].trim().parse::<f64>().ok()
            });
            if let Some(ks) = v {
                out.push(RebootSide {
                    name,
                    ks_per_sample: ks,
                    source: format!(
                        "measured (Task 33, {REBOOT_MEASURED}, mode {mode}): ReBoot's per-sample \
                         layout in hekit (in-library layout, hekit's interactive client, no \
                         bootstraps); ReBoot official (OpenFHE) not measured"
                    ),
                });
            }
        }
    }
    out.push(RebootSide {
        name: "m0_model",
        ks_per_sample: REBOOT_KS_PER_SAMPLE,
        source: "M0 cost model, ReBoot as shipped at N = 2^16 (spikes/RESULTS.md Task 6)".into(),
    });
    out
}

/// The RQ1 ratio three ways (parent's reporting rule) for every ReBoot-side baseline: hekit's
/// key switches per sample with its bootstraps priced at `p_he`, over ReBoot's with its 4
/// bootstraps per update priced at `p_rb`, against ReBoot at its own B = 48 and at hekit's B.
fn rq1_json(
    nonboot_per_sample: f64,
    boots_per_sample: f64,
    batch: usize,
    he_boot: u64,
    sides: &[RebootSide],
) -> J {
    let he = |p: f64| nonboot_per_sample + boots_per_sample * p;
    let way = |side: &RebootSide, name: &str, p_he: u64, p_rb: u64| {
        let (ph, pr) = (p_he as f64, p_rb as f64);
        let rb = |b: usize| side.ks_per_sample + REBOOT_BOOTS_PER_UPDATE * pr / b as f64;
        obj(vec![
            ("way", J::S(name.into())),
            ("hekit_boot_ks", J::I(p_he as i64)),
            ("reboot_boot_ks", J::I(p_rb as i64)),
            ("hekit_ks_per_sample", J::N(he(ph))),
            ("reboot_ks_per_sample_b48", J::N(rb(REBOOT_BATCH))),
            ("ratio_vs_reboot_b48", J::N(he(ph) / rb(REBOOT_BATCH))),
            ("reboot_ks_per_sample_same_b", J::N(rb(batch))),
            ("ratio_vs_reboot_same_b", J::N(he(ph) / rb(batch))),
        ])
    };
    let per_side = sides
        .iter()
        .map(|side| {
            (
                side.name.to_string(),
                obj(vec![
                    ("source", J::S(side.source.clone())),
                    ("ks_per_sample_without_bootstraps", J::N(side.ks_per_sample)),
                    (
                        "a_own_bootstrap_costs",
                        way(
                            side,
                            &format!(
                                "(a) own bootstrap costs: hekit {he_boot} (measured), ReBoot 502"
                            ),
                            he_boot,
                            REBOOT_BOOT_KS,
                        ),
                    ),
                    (
                        "b_both_502",
                        way(
                            side,
                            "(b) both 502: layout/packing gain only (conservative)",
                            REBOOT_BOOT_KS,
                            REBOOT_BOOT_KS,
                        ),
                    ),
                    (
                        "c_both_hekit",
                        way(
                            side,
                            &format!("(c) both {he_boot}: ReBoot's layout on hekit's bootstrapper"),
                            he_boot,
                            he_boot,
                        ),
                    ),
                ]),
            )
        })
        .collect();
    obj(vec![
        (
            "metric",
            J::S("key switches per sample (rotations + relins + SSE switches), bootstraps included".into()),
        ),
        ("headline_reboot_side", J::S(sides[0].name.into())),
        ("by_reboot_side", J::O(per_side)),
        (
            "reboot_bootstraps",
            obj(vec![
                ("bootstraps_per_update", J::N(REBOOT_BOOTS_PER_UPDATE)),
                ("source", J::S("M0 cost model: ReBoot bootstraps W and V of both layers once per update".into())),
                ("own_batch", J::I(REBOOT_BATCH as i64)),
                (
                    "boot_ks_502_provenance",
                    J::S("OpenFHE EvalBootstrap, Meta-BTS with 2 iterations, as configured by ReBoot \
                          (level budget [2, 2]): 2 x (218 linear-transform rotations + 1 conjugation \
                          + 32 EvalMod relins) = 502, counted from the OpenFHE code by the M0 cost \
                          model (spikes/costmodel/costmodel.py bootstrap_key_switches; \
                          spikes/RESULTS.md Task 6; docs/bootstrap.md 11.2). A model count, not a \
                          measurement.".into()),
                ),
            ]),
        ),
    ])
}

/// Where Task 33's measured ReBoot-layout baseline would be (not parsed here: the crate has no
/// JSON dependency; when it exists the report states its values next to these).
const REBOOT_MEASURED: &str = "results/mnist_reboot_layout.json";

#[allow(clippy::too_many_lines)]
fn run_ni(args: &Args) -> Result<(), String> {
    let started_unix = unix_now();
    let load_start = loadavg();
    let t_all = Instant::now();
    let placement = placement_named(&args.placement)?;
    let loaded = load(args);
    let features = match (&loaded, args.plan) {
        (Ok((train, _)), _) => train.features(),
        (Err(e), true) => {
            eprintln!("mnist: --plan: data not loaded ({e}); planning for 28 x 28 images");
            28 * 28
        }
        (Err(e), false) => return Err(e.clone()),
    };
    let params = Arc::new(params_named(&args.params)?);
    let dims = [features, args.hidden, CLASSES];
    let acts_flags = [true, false];
    let d = dims
        .iter()
        .map(|x| x.next_power_of_two())
        .max()
        .expect("dims");
    // B lanes per ciphertext; a larger batch is k micro-batch ciphertexts (ruling R-M3-4).
    let lanes = args.batch.next_power_of_two().min(params.slots() / d);
    let micro = args.batch.div_ceil(lanes);
    if micro * lanes != args.batch {
        return Err(format!(
            "--batch {} is not a whole number of {lanes}-lane ciphertexts (power-of-two lanes, \
             ruling R-M3-4: B = 48 is not run)",
            args.batch
        ));
    }
    let layout = Layout::new(params.slots(), d, lanes).map_err(|e| e.to_string())?;
    let keys = match args.transpose.as_str() {
        "reuse" => TransposeKeys::Reuse(mlp_key_pool(&dims, &layout).map_err(|e| e.to_string())?),
        "dedicated" => TransposeKeys::Dedicated,
        t => return Err(format!("unknown --transpose {t} (reuse | dedicated)")),
    };
    let steps = ni_rotation_steps(&dims, &layout, &keys).map_err(|e| e.to_string())?;
    let limbs = params.q().len() + params.p().len();
    let key_bytes = params.dnum() * 2 * limbs * params.n() * 8;
    let bp = BootParams {
        log_slots: params.slots().trailing_zeros(),
        ..BootParams::default_n16()
    };
    let cl = CompactLayout::new(&dims, &layout).map_err(|e| e.to_string())?;

    // The prediction, before any key exists (stated in the report before the run).
    let model = cost_model_ni(&dims, &acts_flags, args.batch, params.slots());
    let boot_ks_he = HEKIT_BOOT_KS;
    let b = args.batch as f64;
    eprintln!(
        "mnist: noninteractive {} on {} (N = 2^{}, {} slots), dims {dims:?}, d = {d}, {lanes} lanes \
         x {micro} micro-batches, transpose {}: {} training Galois keys + relin ({:.1} GB) + \
         bootstrap keys",
        placement.name(),
        args.params,
        params.log_n(),
        params.slots(),
        args.transpose,
        steps.len(),
        ((steps.len() + 1) * key_bytes) as f64 / 1e9
    );

    let act_objs: [&dyn Activation; 2] = [&PolyRelu2, &Identity];
    // Per-update fullness of the run (all epochs): a group is full when every lane of its
    // ciphertexts is valid (its sample count is a multiple of `lanes`), whatever the number of
    // ciphertexts: the trainer's definition (review N1).
    let fullness = |n: usize| -> Vec<bool> {
        (0..args.epochs)
            .flat_map(|_| {
                (0..n)
                    .step_by(args.batch)
                    .map(move |s| ((s + args.batch).min(n) - s) % lanes == 0)
            })
            .collect()
    };
    let schedule_of = |full: &[bool], start: usize, out: usize| {
        boot_schedule(&dims, &act_objs, &layout, placement, true, start, out, full)
            .map_err(|e| e.to_string())
    };
    let weight_boots_avg = |sched: &[bool]| {
        sched.iter().filter(|&&x| x).count() as f64 * cl.bootstraps(true) as f64
            / sched.len().max(1) as f64
    };
    if args.plan {
        let out_level = hekit_ckks::bootstrap::levels_after_boot(&params, &bp);
        let sched = schedule_of(&fullness(args.subset), out_level - 1, out_level)?;
        let (cost, terms, boots) = ni_terms(
            &dims,
            &acts_flags,
            &layout,
            &keys,
            placement,
            &model,
            weight_boots_avg(&sched),
            boot_ks_he,
        )?;
        let pred_ks = cost.rotations() + cost.relins();
        let pred_total = pred_ks as f64 + boots * boot_ks_he as f64;
        let mut s = String::new();
        obj(vec![
            (
                "cost_model_ni_key_switches_without_boots",
                J::I(model.ks.total() as i64),
            ),
            ("cost_model_ni_bootstraps", J::I(model.boots as i64)),
            (
                "cost_model_ni_ks_per_sample_111",
                J::N((model.ks.total() + model.boots * boot_ks_he) as f64 / b),
            ),
            ("closed_form", ni_cost_json(&cost)),
            ("terms", J::S(format!("{terms:?}"))),
            ("bootstraps_per_update_avg", J::N(boots)),
            ("predicted_ks_per_sample", J::N(pred_total / b)),
            (
                "boot_schedule",
                J::A(sched.iter().map(|&x| J::I(i64::from(x))).collect()),
            ),
            ("galois_keys", J::I(steps.len() as i64)),
        ])
        .write(&mut s, 0);
        println!("{s}");
        return Ok(());
    }
    let (train, test) = loaded?;

    // Keys: the client generates everything, keeps the secret key, and hands the server the
    // evaluation keys only.
    let t_keys = Instant::now();
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), rng(args.seed, 1)).map_err(|e| e.to_string())?;
    let encoder = Encoder::new(Arc::clone(&params));
    let bs =
        Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg).map_err(|e| e.to_string())?;
    let boot_setup_s = t_keys.elapsed().as_secs_f64();
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let sk = kg.secret_key();
    let ev = Evaluator::new(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    drop(kg);
    let keygen_s = t_keys.elapsed().as_secs_f64();
    let per_boot = bs.cost_per_boot();
    let boot_ks = bs.key_switches_per_boot();
    eprintln!(
        "mnist: bootstrapper {boot_setup_s:.1} s, all keys {keygen_s:.1} s; {boot_ks} KS per \
         bootstrap; bootstrap keys {:.2} GB",
        bs.keys().bytes() as f64 / 1e9
    );

    // Network, compact state and twin (client side).
    let mut r = rng(args.seed, 2);
    let init = init_net(&dims, &mut r);
    let plain: Vec<(Array2<f64>, Vec<f64>)> =
        init.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(PolyRelu2), Box::new(Identity)];
    let t_enc = Instant::now();
    let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
        &ev, &bs, &plain, acts, &layout, placement, &enc, &encoder, &mut r,
    )
    .map_err(|e| e.to_string())?;
    let encrypt_net_s = t_enc.elapsed().as_secs_f64();
    let optim = NiOptimizer::Nag {
        lr: args.lr,
        momentum: args.momentum,
    };
    let lpi = levels_per_iter(&mlp, placement);
    let mut trainer = NonInteractiveTrainer::new(&ev, &bs, optim, layout, placement, &mlp, state)
        .map_err(|e| e.to_string())?
        .with_transpose_keys(keys.clone())
        .with_micro_batches(micro);
    let spec = trainer.batch_spec(&mlp).map_err(|e| e.to_string())?;
    let n = train.len();
    let run_full = fullness(n);
    let schedule = schedule_of(&run_full, trainer.start_level(), bs.output_level())?;
    let (cost, terms, boots_avg) = ni_terms(
        &dims,
        &acts_flags,
        &layout,
        &keys,
        placement,
        &model,
        weight_boots_avg(&schedule),
        boot_ks,
    )?;
    let pred_nonboot = cost.rotations() + cost.relins();
    let pred_per_sample = (pred_nonboot as f64 + boots_avg * boot_ks as f64) / b;
    let model_per_sample = (model.ks.total() + model.boots * boot_ks) as f64 / b;
    eprintln!(
        "mnist: levels per step {lpi} (of {} after a bootstrap and an unpack); predicted per \
         update {pred_nonboot} KS + {boots_avg:.2} bootstraps = {pred_per_sample:.3} KS/sample; \
         cost model {model_per_sample:.3} KS/sample; corrections {terms:?}",
        trainer.start_level()
    );
    let mut twin = init.clone();
    let mut vel: Vec<Array2<f64>> = twin.iter().map(|l| Array2::zeros(l.w.dim())).collect();
    let y_train = one_hot(&train.labels, CLASSES);

    // Training.
    let mut update_s = Vec::new();
    let mut client_enc_s = Vec::new();
    let mut per_update: Vec<OpCounter> = Vec::new();
    let mut booted = Vec::new();
    let mut matches = true;
    let mut total = OpCounter::default();
    let mut samples = 0usize;
    let mut max_hidden: f64 = 0.0;
    // Client-side diagnostic of the activation bootstraps (AfterActivation): the client
    // decrypts the bootstrapped activations after each update, outside the trainer (which holds
    // no key) and outside the timed step, and compares them with the twin's.
    let diag_dec = Decryptor::new(Arc::clone(&params), sk.clone());
    let mut act_boot_err: f64 = 0.0;
    let mut act_boot_max_out: f64 = 0.0;
    let mut act_boot_inputs = 0usize;
    let mut act_boot_inputs_above_1 = 0usize;
    let mut update_index = 0usize;
    let mut pred_run_ks = 0u64;
    let mut boot_s = Vec::new();
    let t_train = Instant::now();
    for epoch in 0..args.epochs {
        for start in (0..n).step_by(args.batch) {
            let end = (start + args.batch).min(n);
            trainer.set_next_batch_full(run_full.get(update_index + 1).copied());
            update_index += 1;
            let boot_s0 = trainer.bootstrap_seconds();
            let t0 = Instant::now();
            let mut cts = Vec::new();
            for sub in (start..end).step_by(lanes) {
                let se = (sub + lanes).min(end);
                let x = train.images.slice(s![sub..se, ..]).to_owned();
                let y = y_train.slice(s![sub..se, ..]).to_owned();
                let (xc, yc) = spec
                    .encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
                    .map_err(|e| e.to_string())?;
                cts.push((xc, yc, se - sub));
            }
            client_enc_s.push(t0.elapsed().as_secs_f64());
            let x = train.images.slice(s![start..end, ..]).to_owned();
            let y = y_train.slice(s![start..end, ..]).to_owned();
            let h = affine(&x, &twin[0]).mapv(|v| v * v + v);
            max_hidden = max_hidden.max(max_abs(h.iter().copied()));
            let refs: Vec<_> = cts.iter().map(|(x, y, k)| (x, y, *k)).collect();
            let before = ev.counter();
            let t1 = Instant::now();
            trainer
                .train_step_valid(&mut mlp, &refs)
                .map_err(|e| format!("update at {start}: {e}"))?;
            let secs = t1.elapsed().as_secs_f64();
            let c = &ev.counter() - &before;
            boot_s.push(trainer.bootstrap_seconds() - boot_s0);
            for (mb, outs) in trainer.last_activation_bootstraps().iter().enumerate() {
                let (sub, nv) = (mb * lanes, refs[mb].2);
                let dec_a = unpack_activations(
                    &encoder.decode(&diag_dec.decrypt(&outs[0])),
                    &layout,
                    args.hidden,
                );
                for i in 0..nv {
                    for j in 0..args.hidden {
                        let (he, tw) = (dec_a[[i, j]], h[[sub + i, j]]);
                        act_boot_err = act_boot_err.max((he - tw).abs());
                        act_boot_max_out = act_boot_max_out.max(he.abs());
                        act_boot_inputs += 1;
                        act_boot_inputs_above_1 += usize::from(tw.abs() > 1.0);
                    }
                }
            }
            twin_step(&mut twin, &mut vel, &x, &y, args.lr, args.momentum);
            let did_boot = trainer.last_update_booted();
            // The closed form of this update (any fullness: valid lanes change no key switch;
            // the micro-batch count does) with the a-priori schedule, checked against the
            // counter and the observed bootstrap decision.
            let planned = schedule[update_index - 1];
            let p = predict_update_for(
                &dims,
                &lib_acts(&acts_flags),
                &layout,
                &keys,
                placement,
                refs.len(),
                true,
                planned,
            )
            .map_err(|e| e.to_string())?;
            let nb = p.bootstraps();
            pred_run_ks += p.key_switches(boot_ks);
            matches &= did_boot == planned
                && c.bootstraps == nb
                && c.rotations == p.rotations() + nb * per_boot.rotations
                && c.relins == p.relins() + nb * per_boot.relins
                && c.key_switches() == p.key_switches(boot_ks);
            total += &c;
            samples += end - start;
            update_s.push(secs);
            booted.push(did_boot);
            eprintln!(
                "mnist: epoch {epoch} update {} ({} samples, {} cts): {secs:.1} s, {} KS \
                 ({} bootstraps{}), matches prediction {matches}; elapsed {:.0} s, load {}",
                per_update.len(),
                end - start,
                refs.len(),
                c.key_switches(),
                c.bootstraps,
                if did_boot { ", weights booted" } else { "" },
                t_all.elapsed().as_secs_f64(),
                loadavg()
            );
            per_update.push(c);
        }
    }
    let train_s = t_train.elapsed().as_secs_f64();

    // Results (the client decrypts only now).
    let dec = Decryptor::new(Arc::clone(&params), sk);
    let got: Vec<Layer> = mlp
        .layers
        .iter()
        .map(|(dn, _)| Layer {
            w: dn.w.decrypt(&dec, &encoder, &layout),
            b: unpack_activations(&encoder.decode(&dec.decrypt(&dn.b)), &layout, dn.w.d_out())
                .row(0)
                .to_vec(),
        })
        .collect();
    let w_diff: Vec<f64> = got
        .iter()
        .zip(&twin)
        .map(|(g, t)| max_abs(g.w.iter().zip(&t.w).map(|(a, b)| a - b)))
        .collect();
    let b_diff: Vec<f64> = got
        .iter()
        .zip(&twin)
        .map(|(g, t)| max_abs(g.b.iter().zip(&t.b).map(|(a, b)| a - b)))
        .collect();
    let w_moved: Vec<f64> = twin
        .iter()
        .zip(&init)
        .map(|(t, i)| max_abs(t.w.iter().zip(&i.w).map(|(a, b)| a - b)))
        .collect();
    let max_diff = w_diff.iter().chain(&b_diff).copied().fold(0.0, f64::max);
    let y_test = one_hot(&test.labels, CLASSES);
    let out_he = forward_plain(&got, &test.images);
    let out_twin = forward_plain(&twin, &test.images);
    let out_init = forward_plain(&init, &test.images);
    let sf = samples as f64;
    let updates = per_update.len();
    let ks = total.key_switches();
    let boots = total.bootstraps;
    let nonboot = ks - boots * boot_ks;
    let measured_per_sample = ks as f64 / sf;
    let epoch_full_s = if samples == n * args.epochs {
        train_s
    } else {
        f64::NAN
    };
    let per_update_mean = update_s.iter().sum::<f64>() / updates.max(1) as f64;
    let extrapolated_1024 = per_update_mean * 1024f64.div_euclid(b).max(1.0);
    eprintln!(
        "mnist: {updates} updates, {samples} samples in {train_s:.1} s; {ks} KS = \
         {measured_per_sample:.3}/sample incl. {boots} bootstraps (predicted \
         {pred_per_sample:.3}, model {model_per_sample:.3}); matches {matches}; max |W_he - W_twin| \
         = {max_diff:e}"
    );
    let sides = reboot_sides();
    let result = obj(vec![
        ("provenance", provenance(&load_start, started_unix)),
        (
            "task",
            J::S("Task 41: MNIST eMLP-1 non-interactive training with bootstrapping (RQ1)".into()),
        ),
        (
            "network",
            J::S(format!(
                "{} -> {} (z^2 + z) -> {} (linear), MSE, NAG (biases SGD)",
                dims[0], dims[1], dims[2]
            )),
        ),
        (
            "setup",
            obj(vec![
                ("params", J::S(args.params.clone())),
                ("log_n", J::I(params.log_n() as i64)),
                ("slots", J::I(params.slots() as i64)),
                ("max_level", J::I(params.max_level() as i64)),
                ("bootstrap_output_level", J::I(bs.output_level() as i64)),
                (
                    "start_level_after_unpack",
                    J::I(trainer.start_level() as i64),
                ),
                ("levels_per_iter", J::I(lpi as i64)),
                ("placement", J::S(placement.name().into())),
                ("batch", J::I(args.batch as i64)),
                ("lanes_per_ciphertext", J::I(lanes as i64)),
                ("micro_batches_per_update", J::I(micro as i64)),
                ("layout_d", J::I(d as i64)),
                ("transpose_keys", J::S(args.transpose.clone())),
                ("training_galois_keys", J::I(steps.len() as i64)),
                ("key_bytes_each", J::I(key_bytes as i64)),
                ("bootstrap_key_bytes", J::I(bs.keys().bytes() as i64)),
                ("boot_ks", J::I(boot_ks as i64)),
                ("compact_containers", J::I(cl.containers.len() as i64)),
                ("bias_slots", J::S(format!("{:?}", cl.biases))),
            ]),
        ),
        (
            "training",
            obj(vec![
                ("epochs", J::I(args.epochs as i64)),
                ("subset", J::I(n as i64)),
                ("updates", J::I(updates as i64)),
                ("samples", J::I(samples as i64)),
                ("lr", J::N(args.lr)),
                ("momentum", J::N(args.momentum)),
                ("weight_decay", J::N(0.0)),
                ("seed", J::I(args.seed as i64)),
            ]),
        ),
        (
            "prediction",
            obj(vec![
                ("cost_model_ni_without_boots", ks_json(&model.ks, b)),
                (
                    "cost_model_ni_bootstraps_per_update",
                    J::I(model.boots as i64),
                ),
                (
                    "cost_model_ni_ks_per_sample_at_hekit_boot",
                    J::N(model_per_sample),
                ),
                ("closed_form_per_update", ni_cost_json(&cost)),
                ("bootstraps_per_update_avg", J::N(boots_avg)),
                (
                    "corrections_per_update",
                    obj(vec![
                        ("recomputed_babies", J::I(terms.babies)),
                        ("bsgs_split", J::I(terms.split)),
                        ("transpose_beyond_model", J::I(terms.transpose)),
                        ("bias_gradients_and_extraction", J::I(terms.bias)),
                        ("bootstraps_beyond_model_x_boot_ks", J::I(terms.boots)),
                        ("total", J::I(terms.total())),
                    ]),
                ),
                ("predicted_ks_per_sample", J::N(pred_per_sample)),
                (
                    "boot_schedule",
                    J::A(schedule.iter().map(|&x| J::I(i64::from(x))).collect()),
                ),
            ]),
        ),
        (
            "measured",
            obj(vec![
                ("total", ni_counter_json(&total)),
                ("ks_per_sample_incl_bootstraps", J::N(measured_per_sample)),
                (
                    "ks_per_sample_without_bootstraps",
                    J::N(nonboot as f64 / sf),
                ),
                ("bootstraps_per_update", J::N(boots as f64 / updates as f64)),
                ("bootstraps_per_sample", J::N(boots as f64 / sf)),
                (
                    "weight_booted",
                    J::A(booted.iter().map(|&x| J::I(i64::from(x))).collect()),
                ),
                ("every_update_equals_prediction", J::B(matches)),
                ("predicted_ks_this_run", J::I(pred_run_ks as i64)),
                (
                    "measured_over_predicted",
                    J::N(ks as f64 / pred_run_ks as f64),
                ),
                (
                    "measured_over_predicted_full_batch_per_sample",
                    J::N(measured_per_sample / pred_per_sample),
                ),
                (
                    "measured_over_model_plus_corrections",
                    J::N(
                        measured_per_sample
                            / (model_per_sample + terms.total() as f64 / b),
                    ),
                ),
                (
                    "measured_over_cost_model",
                    J::N(measured_per_sample / model_per_sample),
                ),
                ("per_update_first", ni_counter_json(&per_update[0])),
            ]),
        ),
        (
            "rq1",
            rq1_json(
                nonboot as f64 / sf,
                boots as f64 / sf,
                args.batch,
                boot_ks,
                &sides,
            ),
        ),
        (
            "time_s",
            obj(vec![
                ("bootstrapper_setup", J::N(boot_setup_s)),
                ("keygen_total", J::N(keygen_s)),
                ("encrypt_network", J::N(encrypt_net_s)),
                ("train_total", J::N(train_s)),
                ("epoch_measured", J::N(epoch_full_s)),
                (
                    "epoch_1024_extrapolated_from_mean_update",
                    J::N(extrapolated_1024),
                ),
                ("per_update", stats(&update_s)),
                ("per_update_client_encrypt", stats(&client_enc_s)),
                (
                    "bootstrap_time_total",
                    J::N(trainer.bootstrap_seconds()),
                ),
                ("per_update_bootstrap", stats(&boot_s)),
                ("wall_total", J::N(t_all.elapsed().as_secs_f64())),
            ]),
        ),
        (
            "twin",
            obj(vec![
                (
                    "max_abs_w_diff_per_layer",
                    J::A(w_diff.iter().map(|&v| J::N(v)).collect()),
                ),
                (
                    "max_abs_b_diff_per_layer",
                    J::A(b_diff.iter().map(|&v| J::N(v)).collect()),
                ),
                ("max_abs_diff", J::N(max_diff)),
                ("log2_max_abs_diff", J::N(max_diff.log2())),
                (
                    "max_abs_w_update_per_layer",
                    J::A(w_moved.iter().map(|&v| J::N(v)).collect()),
                ),
                ("max_abs_hidden_activation", J::N(max_hidden)),
                (
                    "activation_bootstraps",
                    obj(vec![
                        (
                            "note",
                            J::S("AfterActivation only: the bootstrapped hidden activations of every \
                                  update, decrypted by the client after the update, against the \
                                  twin's activations from the same pre-update weights (bootstrap \
                                  error plus the accumulated training error, which the weight \
                                  diff bounds). Inputs above 1 violate the bootstrap's |m| <= 1 \
                                  precondition; they are not clipped."
                                .into()),
                        ),
                        ("slots_compared", J::I(act_boot_inputs as i64)),
                        ("max_abs_input_twin", J::N(max_hidden)),
                        ("inputs_above_1", J::I(act_boot_inputs_above_1 as i64)),
                        ("precondition_hit", J::B(act_boot_inputs_above_1 > 0)),
                        ("max_abs_output", J::N(act_boot_max_out)),
                        ("max_abs_err_vs_twin", J::N(act_boot_err)),
                        ("log2_max_abs_err_vs_twin", J::N(act_boot_err.log2())),
                    ]),
                ),
            ]),
        ),
        (
            "evaluation",
            obj(vec![
                ("test_samples", J::I(test.len() as i64)),
                (
                    "test_acc_encrypted_trained",
                    J::N(accuracy(&out_he, &test.labels)),
                ),
                ("test_acc_twin", J::N(accuracy(&out_twin, &test.labels))),
                ("test_acc_init", J::N(accuracy(&out_init, &test.labels))),
                ("test_mse_encrypted_trained", J::N(mse(&out_he, &y_test))),
                ("test_mse_twin", J::N(mse(&out_twin, &y_test))),
            ]),
        ),
        (
            "levels_last_step",
            J::O(
                trainer
                    .trace()
                    .iter()
                    .map(|(k, v)| (k.clone(), J::I(*v as i64)))
                    .collect(),
            ),
        ),
        (
            "memory",
            obj(vec![(
                "peak_rss_bytes",
                peak_rss_bytes().map_or(J::N(f64::NAN), |b| J::I(b as i64)),
            )]),
        ),
        (
            "machine",
            obj(vec![
                ("load_start", J::S(load_start)),
                ("load_end", J::S(loadavg())),
                ("build", build_info()),
                ("parallel_feature", J::B(cfg!(feature = "parallel"))),
                (
                    "threads",
                    J::S(if cfg!(feature = "parallel") {
                        format!(
                            "rayon, {} available (this Mac); quadruped-gpu 32-thread figure: pending (user-driven session)",
                            std::thread::available_parallelism().map_or(0, |n| n.get())
                        )
                    } else {
                        "single".into()
                    }),
                ),
            ]),
        ),
    ]);
    let mut out = String::new();
    result.write(&mut out, 0);
    out.push('\n');
    let path = args.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "results/mnist_emlp1_ni_{}_b{}.json",
            placement.name(),
            args.batch
        ))
    });
    write_out(&path, &out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMLP1: [usize; 3] = [784, 32, 10];

    /// The port equals `spikes/costmodel` (`network_cost("diag_batched", [(784, 32,
    /// "poly_relu"), (32, 10, "none")], 32, 2**15, interactive=True)`: 291 rotations and 58
    /// relins per update, 10.906 KS per sample).
    #[test]
    fn cost_model_port_matches_the_spike() {
        assert_eq!(spike_split(32), (8, 4));
        assert_eq!(spike_split(16), (8, 2));
        let k = cost_model_interactive(&EMLP1, &[true, false], 32);
        assert_eq!((k.rot, k.relin), (291, 58));
    }

    #[test]
    fn json_strings_are_escaped() {
        let mut out = String::new();
        json_string(&mut out, "a\"b\\c\n\u{1}é");
        assert_eq!(out, "\"a\\\"b\\\\c\\n\\u0001é\"");
        let mut st = String::new();
        stats(&[]).write(&mut st, 0);
        assert!(
            st.contains("\"min\": null") && st.contains("\"max\": null"),
            "{st}"
        );
        assert!(peak_rss_bytes().is_some_and(|b| b > 0));
    }

    /// eMLP-1 on n16_train's 2^15 slots with 32 lanes: the exact count is the model plus the
    /// corrections, for both transpose key sets (`docs/layout.md` §2.3: 31 vs 45 keys).
    #[test]
    fn emlp1_prediction_is_model_plus_corrections() {
        let layout = Layout::new(1 << 15, 1024, 32).unwrap();
        let model = cost_model_interactive(&EMLP1, &[true, false], 32);
        let reuse = TransposeKeys::Reuse(mlp_key_pool(&EMLP1, &layout).unwrap());
        for (keys, ks, n_keys) in [(reuse, 633, 31), (TransposeKeys::Dedicated, 621, 45)] {
            let (k, t) = predict(&EMLP1, &[true, false], &layout, &keys).unwrap();
            assert_eq!(k.total(), ks);
            assert_eq!(model.total() as i64 + t.total(), ks as i64);
            assert_eq!(
                (t.replication, t.babies, t.bias, t.split),
                (240, 14, 20, -2)
            );
            // Pre-registered in docs/layout.md §5 (items 1, 2, 5): 615 reuse, 603 dedicated.
            let pre = model.total() + t.replication + t.babies + t.transpose;
            assert_eq!(
                pre,
                ks - 18,
                "pre-registered = exact - bias (20) - split (-2)"
            );
            assert_eq!(
                mlp_rotation_steps(&EMLP1, &layout, &keys).unwrap().len(),
                n_keys
            );
        }
    }

    /// The non-interactive port equals `spikes/costmodel` (`network_cost("diag_batched",
    /// layers, B, slots, boot_ks=111)`, compact, NAG): rotations, relins and bootstraps per
    /// update, from a run of the spike.
    #[test]
    fn cost_model_ni_port_matches_the_spike() {
        let em = [true, false];
        for (dims, b, slots, rot, relin, boots) in [
            (&EMLP1[..], 32, 1 << 15, 703, 58, 4),
            (&EMLP1[..], 64, 1 << 15, 739, 68, 4),
            (&EMLP1[..], 48, 1 << 15, 739, 68, 4),
            (&[64, 32, 10][..], 8, 512, 437, 58, 10),
            (&[64, 32, 10][..], 16, 512, 469, 68, 10),
        ] {
            let m = cost_model_ni(dims, &em, b, slots);
            assert_eq!(
                (m.ks.rot, m.ks.relin, m.boots),
                (rot, relin, boots),
                "{dims:?} B {b}"
            );
        }
        // The spike's KS_spec per sample at 111: 37.656 (B 32), 19.547 (B 64), 26.063 (B 48).
        for (b, ks) in [(32, 37.65625), (64, 19.546875), (48, 26.0625)] {
            let m = cost_model_ni(&EMLP1, &em, b, 1 << 15);
            assert_eq!((m.ks.total() + 111 * m.boots) as f64 / b as f64, ks);
        }
    }

    /// eMLP-1 non-interactive (AfterUpdate) on 2^15 slots: the library's closed form is the
    /// model plus the listed corrections, term by term, for B = 32 (one ciphertext) and
    /// B = 64 (two micro-batches), with both transpose key sets.
    #[test]
    fn emlp1_ni_closed_form_is_model_plus_corrections() {
        let layout = Layout::new(1 << 15, 1024, 32).unwrap();
        let em = [true, false];
        let reuse = TransposeKeys::Reuse(mlp_key_pool(&EMLP1, &layout).unwrap());
        for (b, keys, terms, per_sample) in [
            (32, reuse.clone(), (14, -2, 12, 34), 39.46875),
            (64, reuse, (28, -4, 12, 34), 20.640625),
            (32, TransposeKeys::Dedicated, (14, -2, 0, 34), 39.09375),
        ] {
            let model = cost_model_ni(&EMLP1, &em, b, 1 << 15);
            let (cost, t, boots) = ni_terms(
                &EMLP1,
                &em,
                &layout,
                &keys,
                BootPlacement::AfterUpdate,
                &model,
                4.0,
                111,
            )
            .unwrap();
            assert_eq!(
                (t.babies, t.split, t.transpose, t.bias, t.boots),
                (terms.0, terms.1, terms.2, terms.3, 0)
            );
            assert_eq!(boots, 4.0);
            let closed = (cost.rotations() + cost.relins() + 4 * 111) as i64;
            assert_eq!(
                closed,
                (model.ks.total() + 111 * model.boots) as i64 + t.total()
            );
            assert_eq!(closed as f64 / b as f64, per_sample, "B {b}");
            // Within 10% of the model (plan owner's gate).
            let model_ps = (model.ks.total() + 111 * model.boots) as f64 / b as f64;
            assert!(per_sample / model_ps < 1.10, "{}", per_sample / model_ps);
        }
    }
}
