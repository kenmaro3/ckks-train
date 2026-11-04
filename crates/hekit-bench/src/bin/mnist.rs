//! Task 32: interactive encrypted training of ReBoot's eMLP-1 on MNIST and the measured
//! key-switch counts that decide RQ1 (`docs/m2-report.md` "MNIST eMLP-1").
//!
//! # Network and training (`crates/hekit-bench/configs/reboot_emlp1.json`)
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
use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Identity, PolyRelu2};
use hekit_nn::dense::{TransposeKeys, transpose_rotation_plan};
use hekit_nn::layout::{LayerShape, Layout, unpack_activations};
use hekit_nn::optim::Nag;
use hekit_nn::refresh::ClientRefresher;
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_key_pool, mlp_rotation_steps};
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
    out: PathBuf,
    plan: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        epochs: 1,
        subset: 1024,
        test_subset: 10_000,
        batch: 32,
        hidden: 32,
        params: "n16_train".into(),
        transpose: "reuse".into(),
        seed: 32,
        lr: 0.001,
        momentum: 0.9,
        data: PathBuf::from("data/mnist"),
        out: PathBuf::from("results/mnist_emlp1.json"),
        plan: false,
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
            "--out" => a.out = PathBuf::from(v),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if a.epochs == 0 || a.subset == 0 || a.batch == 0 || a.hidden == 0 {
        return Err("--epochs, --subset, --batch and --hidden must be positive".into());
    }
    Ok(a)
}

fn params_named(name: &str) -> Result<CkksParams, String> {
    match name {
        "n16_train" => Ok(CkksParams::n16_train()),
        // Insecure (N = 2^12 with 11 primes): the smoke test's fast stand-in for n16_train.
        "small" => {
            let mut q = vec![58];
            q.extend([42; 10]);
            CkksParams::new(12, &q, &[60], 42, 11).map_err(|e| e.to_string())
        }
        _ => Err(format!("unknown --params {name} (n16_train | small)")),
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
            J::S(s) => write!(out, "{s:?}").unwrap(),
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

fn obj(kv: Vec<(&str, J)>) -> J {
    J::O(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn ks_json(k: &Ks, samples: f64) -> J {
    obj(vec![
        ("rotations", J::I(k.rot as i64)),
        ("relins", J::I(k.relin as i64)),
        ("key_switches", J::I(k.total() as i64)),
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

fn loadavg() -> String {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn stats(v: &[f64]) -> J {
    let n = v.len().max(1) as f64;
    obj(vec![
        ("mean", J::N(v.iter().sum::<f64>() / n)),
        ("min", J::N(v.iter().copied().fold(f64::INFINITY, f64::min))),
        ("max", J::N(v.iter().copied().fold(0.0, f64::max))),
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
    if let Err(e) = run(&args) {
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
    let (train, test) = load(args)?;
    let params = Arc::new(params_named(&args.params)?);
    let dims = [train.features(), args.hidden, CLASSES];
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
            "documented_adjusted_key_switches",
            J::I((model.total() + terms.replication) as i64),
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
            total.rotations += c.rotations;
            total.relins += c.relins;
            total.ct_ct_tensor += c.ct_ct_tensor;
            total.ct_pt_mults += c.ct_pt_mults;
            total.rescales += c.rescales;
            total.bootstraps += c.bootstraps;
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
    let constant = per_batch
        .iter()
        .all(|c| (c.rotations, c.relins) == (pred.rot, pred.relin));
    let measured = Ks {
        rot: total.rotations,
        relin: total.relins,
    };
    let documented = (model.total() + terms.replication) as f64 * batches as f64;
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
            "measured_over_documented_adjusted",
            J::N(measured.total() as f64 / documented),
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
            "machine",
            obj(vec![
                ("load_start", J::S(load_start)),
                ("load_end", J::S(loadavg())),
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
    if let Some(dir) = args.out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&args.out, &s).map_err(|e| format!("{}: {e}", args.out.display()))?;
    eprintln!("mnist: wrote {}", args.out.display());
    Ok(())
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
            assert_eq!(
                mlp_rotation_steps(&EMLP1, &layout, &keys).unwrap().len(),
                n_keys
            );
        }
    }
}
