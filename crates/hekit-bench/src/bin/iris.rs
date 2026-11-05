//! Task 31: end-to-end encrypted training on Iris, reproducing the 2020 paper's setup
//! (Mihara et al., arXiv:2012.13552): a 4-10-3 MLP with biases, Square after both dense layers,
//! MSE (mean over the 3 outputs), SGD with lr 0.1, CKKS `seal_2020` (N = 2^15, 9 levels),
//! interactive mode (the client decrypts and re-encrypts every weight after each update,
//! [`ClientRefresher`]).
//!
//! # Data, split and preprocessing
//!
//! The same as the M0 SEAL reproduction (`spikes/seal-2020`, `spikes/RESULTS.md` "Task 4"), for
//! comparability:
//! - `data/iris.csv` is that spike's file (Fisher's Iris, UCI CC BY 4.0, via scikit-learn's copy;
//!   attribution in the repository's root `README.md`, "Data");
//! - `data/iris_split_seed2020.txt` is that spike's 80:20 split for its default seed 2020
//!   (120 train / 30 test indices, exported from its `split_80_20`, `std::mt19937_64`);
//! - inputs are z-scored with the training-set mean and population standard deviation;
//! - weights and biases are initialised from N(0, 0.2^2), labels are one-hot.
//!
//! The initial weights and the epoch shuffling come from ChaCha20 (`hekit_core::sampler::Rng`,
//! the project's only random source) seeded by `--seed`, so they are not the spike's draws.
//!
//! # Loss and learning rate
//!
//! The paper's (and the spike's) loss is `L = (1/3) sum_k (a_k - y_k)^2`, averaged over the
//! batch. The trainer's loss gradient is `a - y` (`hekit_nn::loss::Mse`), so the factor
//! `2 / d_out` is folded into the learning rate the encrypted optimizer sees:
//! `lr_eff = lr * 2 / 3`. The update is then exactly the spike's `W -= lr * dL/dW`.
//!
//! # Shapes and layout
//!
//! `d_in = 4`, hidden `10 -> 16` (padded), out `3 -> 4` (padded); padded rows and columns are 0.
//! The layout has `d = 16` feature positions and `B = batch.next_power_of_two()` lanes (20 -> 32:
//! 12 lanes are masked out of the gradient by the trainer's valid-lane mask; 1 -> 1).
//!
//! # Plaintext twin
//!
//! Next to the encrypted network the program trains a dense f64 copy with the same initial
//! weights, the same batches and the same update. Every epoch it logs the max |W_he - W_twin|
//! over all logical weights and biases (noise accumulation over the run), and the loss and
//! accuracy of both. It also logs the copy spread (max deviation of the redundant lane and
//! block copies of every weight and bias, stderr only).
//!
//! # Refresh
//!
//! The client refresh re-lays the replicated blocks (`ClientRefresher::with_layout`). Without
//! it the 32 block copies of the batch-20 layout drift apart exponentially and the run leaves
//! the plaintext trajectory after about 15 epochs (overflow at epoch 30, `docs/m2-report.md`).
//!
//! # Modes
//!
//! - training (default): `--epochs E --batch B`, output `results/iris_<B>.json`;
//! - timing: `--time-iters K [--warmup W]`: `W + K` single iterations, each on its own batch of
//!   `--batch` samples, timed as client encryption of `(x, y)`, server step (everything but the
//!   refresh) and client refresh; output `results/iris_<B>_timing.json`.
//!
//! Usage: `iris [--epochs 400] [--batch 20] [--preset seal_2020] [--seed 2020] [--lr 0.1]
//! [--optimizer sgd|nag] [--momentum 0.9] [--time-iters K] [--warmup W] [--out PATH]
//! [--data PATH] [--split PATH]`.

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use hekit_ckks::bootstrap::{BootParams, Bootstrapper};
use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::error::Result;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Square};
use hekit_nn::dense::TransposeKeys;
use hekit_nn::layout::{Layout, unpack_activations};
use hekit_nn::optim::{Nag, Sgd};
use hekit_nn::refresh::ClientRefresher;
use hekit_nn::trainer::{BatchSpec, EpochLog, LayerOptimizer, Mlp, Trainer, mlp_rotation_steps};
use hekit_nn::trainer_ni::{BootPlacement, NiOptimizer, NonInteractiveTrainer, ni_rotation_steps};
use ndarray::{Array1, Array2, Axis};

const DIMS: [usize; 3] = [4, 10, 3];
const CLASSES: usize = 3;

// ---------------------------------------------------------------------------------------------
// Options.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Opts {
    epochs: usize,
    batch: usize,
    preset: String,
    seed: u64,
    lr: f64,
    optimizer: String,
    momentum: f64,
    sigma: f64,
    time_iters: usize,
    warmup: usize,
    out: Option<String>,
    data: String,
    split: String,
    checkpoint_every: usize,
    mode: String,
    placement: String,
}

impl Default for Opts {
    fn default() -> Opts {
        Opts {
            epochs: 400,
            batch: 20,
            preset: "seal_2020".into(),
            seed: 2020,
            lr: 0.1,
            optimizer: "sgd".into(),
            momentum: 0.9,
            sigma: 0.2,
            time_iters: 0,
            warmup: 1,
            out: None,
            data: concat!(env!("CARGO_MANIFEST_DIR"), "/data/iris.csv").into(),
            split: concat!(env!("CARGO_MANIFEST_DIR"), "/data/iris_split_seed2020.txt").into(),
            checkpoint_every: 10,
            mode: "interactive".into(),
            placement: "after_update".into(),
        }
    }
}

fn usage() -> ! {
    eprintln!(
        "usage: iris [--epochs E] [--batch B] [--preset seal_2020] [--seed S] \
         [--lr LR] [--optimizer sgd|nag] [--momentum MU] [--sigma SIG] [--time-iters K] \
         [--warmup W] [--out PATH] [--data PATH] [--split PATH] [--checkpoint-every E] \
         [--mode interactive|noninteractive] \
         [--placement after_update|after_activation|when_needed] \
         (noninteractive: --preset n16_boot | mini_boot)"
    );
    std::process::exit(2)
}

fn parse_opts() -> Opts {
    let mut o = Opts::default();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| usage());
        fn num<T: std::str::FromStr>(s: String) -> T {
            s.parse().unwrap_or_else(|_| usage())
        }
        match a.as_str() {
            "--epochs" => o.epochs = num(val()),
            "--batch" => o.batch = num(val()),
            "--preset" => o.preset = val(),
            "--seed" => o.seed = num(val()),
            "--lr" => o.lr = num(val()),
            "--optimizer" => o.optimizer = val(),
            "--momentum" => o.momentum = num(val()),
            "--sigma" => o.sigma = num(val()),
            "--time-iters" => o.time_iters = num(val()),
            "--warmup" => o.warmup = num(val()),
            "--out" => o.out = Some(val()),
            "--data" => o.data = val(),
            "--split" => o.split = val(),
            "--checkpoint-every" => o.checkpoint_every = num(val()),
            "--mode" => o.mode = val(),
            "--placement" => o.placement = val(),
            _ => usage(),
        }
    }
    if o.batch == 0
        || !matches!(o.optimizer.as_str(), "sgd" | "nag")
        || !matches!(o.mode.as_str(), "interactive" | "noninteractive")
        || placement(&o.placement).is_none()
    {
        usage();
    }
    o
}

fn preset(name: &str) -> CkksParams {
    match name {
        "seal_2020" => CkksParams::seal_2020(),
        "n16_boot" => CkksParams::n16_boot(),
        // Insecure (N = 2^10): n16_boot's chain in miniature, tests and smoke runs only.
        "mini_boot" => {
            let (q, p) = hekit_ckks::params::n16_boot_bits();
            CkksParams::new(10, &q, &p, 42, 8).expect("mini_boot")
        }
        _ => {
            eprintln!(
                "unknown preset {name} (seal_2020 interactive; n16_boot or mini_boot \
                 non-interactive)"
            );
            std::process::exit(2)
        }
    }
}

fn placement(name: &str) -> Option<BootPlacement> {
    match name {
        "after_update" => Some(BootPlacement::AfterUpdate),
        "after_activation" => Some(BootPlacement::AfterActivation),
        "when_needed" => Some(BootPlacement::WhenNeeded),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Data.
// ---------------------------------------------------------------------------------------------

struct Data {
    x: Array2<f64>,
    labels: Vec<usize>,
    train: Vec<usize>,
    test: Vec<usize>,
}

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
}

/// The CSV (header, 4 features, label 0/1/2), the split file (`train:` / `test:` lines of
/// comma-separated row indices) and z-scoring with training-set statistics (population sd).
fn load(opts: &Opts) -> Data {
    let mut rows = Vec::new();
    let mut labels = Vec::new();
    for line in read(&opts.data)
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 5, "iris row: {line}");
        rows.extend(
            f[..4]
                .iter()
                .map(|v| v.trim().parse::<f64>().expect("feature")),
        );
        labels.push(f[4].trim().parse::<usize>().expect("label"));
    }
    let n = labels.len();
    let mut x = Array2::from_shape_vec((n, 4), rows).expect("shape");
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for line in read(&opts.split).lines().filter(|l| !l.starts_with('#')) {
        let parse = |s: &str| -> Vec<usize> {
            s.split(',')
                .map(|v| v.trim().parse().expect("split index"))
                .collect()
        };
        if let Some(s) = line.strip_prefix("train:") {
            train = parse(s);
        } else if let Some(s) = line.strip_prefix("test:") {
            test = parse(s);
        }
    }
    let mut all: Vec<usize> = train.iter().chain(&test).copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..n).collect::<Vec<_>>(), "the split is a partition");
    for k in 0..4 {
        let mu = train.iter().map(|&i| x[[i, k]]).sum::<f64>() / train.len() as f64;
        let var = train.iter().map(|&i| (x[[i, k]] - mu).powi(2)).sum::<f64>() / train.len() as f64;
        let sd = var.sqrt();
        x.column_mut(k).mapv_inplace(|v| (v - mu) / sd);
    }
    Data {
        x,
        labels,
        train,
        test,
    }
}

fn onehot(labels: &[usize], idx: &[usize]) -> Array2<f64> {
    let mut y = Array2::zeros((idx.len(), CLASSES));
    for (r, &i) in idx.iter().enumerate() {
        y[[r, labels[i]]] = 1.0;
    }
    y
}

fn rows(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    x.select(Axis(0), idx)
}

// ---------------------------------------------------------------------------------------------
// Randomness (ChaCha20 only).
// ---------------------------------------------------------------------------------------------

fn uni(r: &mut Rng) -> f64 {
    (r.next_u64() >> 11) as f64 * 2f64.powi(-53)
}

/// Box–Muller.
fn normal(r: &mut Rng) -> f64 {
    let u1 = uni(r).max(1e-300);
    let u2 = uni(r);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Fisher–Yates.
fn shuffle(v: &mut [usize], r: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, r.below(i as u64 + 1) as usize);
    }
}

// ---------------------------------------------------------------------------------------------
// The plaintext twin (dense f64, the spike's `sgd_step` with the same loss).
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Net {
    layers: Vec<(Array2<f64>, Array1<f64>)>,
}

impl Net {
    fn init(r: &mut Rng, sigma: f64) -> Net {
        Net {
            layers: DIMS
                .windows(2)
                .map(|p| {
                    let w = Array2::from_shape_fn((p[1], p[0]), |_| sigma * normal(r));
                    let b = Array1::from_shape_fn(p[1], |_| sigma * normal(r));
                    (w, b)
                })
                .collect(),
        }
    }

    /// `(z_l, a_l)` per layer, rows are samples.
    fn forward(&self, x: &Array2<f64>) -> Vec<(Array2<f64>, Array2<f64>)> {
        let mut a = x.clone();
        let mut out = Vec::new();
        for (w, b) in &self.layers {
            let z = a.dot(&w.t()) + b;
            a = z.mapv(|v| v * v);
            out.push((z, a.clone()));
        }
        out
    }

    fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        self.forward(x).pop().expect("layers").1
    }

    /// One SGD step on the batch: `L = mean_batch (1/d_out) sum_k (a_k - y_k)^2`.
    fn sgd_step(&mut self, x: &Array2<f64>, y: &Array2<f64>, lr: f64) {
        let n = x.nrows() as f64;
        let fw = self.forward(x);
        let d_out = y.ncols() as f64;
        let a_l = &fw.last().expect("layers").1;
        let mut delta = (a_l - y) * (2.0 / d_out);
        let mut grads = Vec::new();
        for l in (0..self.layers.len()).rev() {
            let z = &fw[l].0;
            let dz = &delta * &z.mapv(|v| 2.0 * v);
            let input = if l == 0 { x } else { &fw[l - 1].1 };
            let gw = dz.t().dot(input) / n;
            let gb = dz.sum_axis(Axis(0)) / n;
            if l > 0 {
                delta = dz.dot(&self.layers[l].0);
            }
            grads.push((l, gw, gb));
        }
        for (l, gw, gb) in grads {
            self.layers[l].0 -= &(gw * lr);
            self.layers[l].1 -= &(gb * lr);
        }
    }
}

/// `(mean-over-outputs MSE averaged over samples, accuracy)`.
fn metrics(pred: &Array2<f64>, labels: &[usize], idx: &[usize]) -> (f64, f64) {
    let y = onehot(labels, idx);
    let loss = (pred - &y).mapv(|v| v * v).sum() / (pred.nrows() * CLASSES) as f64;
    let ok = pred
        .rows()
        .into_iter()
        .zip(idx)
        .filter(|(row, i)| {
            let am = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, _)| k)
                .expect("nonempty");
            am == labels[**i]
        })
        .count();
    (loss, ok as f64 / idx.len() as f64)
}

// ---------------------------------------------------------------------------------------------
// The encrypted side.
// ---------------------------------------------------------------------------------------------

struct Client {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
}

/// Copy spread of every refreshed vector (weight diagonals and biases): each is meant to be
/// the same in every lane and every replicated block, so the max deviation from its lane-0,
/// block-0 value over the other lanes (`lane`) and the other blocks (`block`) measures how far
/// the redundant copies have drifted apart.
fn copy_spread(c: &Client, mlp: &Mlp, layout: &Layout) -> (f64, f64) {
    let (bk, lanes) = (layout.block(), layout.batch);
    let mut spread = (0.0f64, 0.0f64);
    for (d, _) in &mlp.layers {
        for ct in
            d.w.diagonals()
                .entries()
                .iter()
                .chain(std::iter::once(&d.b))
        {
            let v = c.encoder.decode(&c.dec.decrypt(ct));
            for (sl, x) in v.iter().enumerate() {
                let j = (sl % bk) / lanes;
                let r = v[j * lanes];
                let dev = (x - r).abs();
                if sl < bk {
                    spread.0 = spread.0.max(dev);
                } else if sl % lanes == 0 {
                    spread.1 = spread.1.max(dev);
                }
            }
        }
    }
    spread
}

/// The decrypted network (logical shapes).
fn decrypt_net(c: &Client, mlp: &Mlp, layout: &Layout) -> Net {
    Net {
        layers: mlp
            .layers
            .iter()
            .map(|(d, _)| {
                let w = d.w.decrypt(&c.dec, &c.encoder, layout);
                let b = unpack_activations(
                    &c.encoder.decode(&c.dec.decrypt(&d.b)),
                    layout,
                    d.w.d_out(),
                )
                .row(0)
                .to_owned();
                (w, b)
            })
            .collect(),
    }
}

/// `(max |W - W'|, max |b - b'|)` over all layers.
fn net_error(a: &Net, b: &Net) -> (f64, f64) {
    let mut e = (0.0f64, 0.0f64);
    for ((wa, ba), (wb, bb)) in a.layers.iter().zip(&b.layers) {
        e.0 = wa
            .iter()
            .zip(wb)
            .fold(e.0, |m, (x, y)| m.max((x - y).abs()));
        e.1 = ba
            .iter()
            .zip(bb)
            .fold(e.1, |m, (x, y)| m.max((x - y).abs()));
    }
    e
}

fn optims(o: &Opts, lr_eff: f64) -> Vec<Box<dyn LayerOptimizer>> {
    (0..DIMS.len() - 1)
        .map(|_| -> Box<dyn LayerOptimizer> {
            if o.optimizer == "nag" {
                Box::new(Nag::new(lr_eff, o.momentum))
            } else {
                Box::new(Sgd { lr: lr_eff })
            }
        })
        .collect()
}

/// The 1-, 5- and 15-minute load averages: `/proc/loadavg` on Linux, `sysctl vm.loadavg` on
/// macOS; `""` if neither is available.
fn load_avg() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
    }
    std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().trim_matches(['{', '}']).trim().to_string())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------------
// JSON (hand-written: a handful of numbers, no serde dependency).
// ---------------------------------------------------------------------------------------------

fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{v:e}")
    } else {
        "null".into()
    }
}

fn jcounter(c: &OpCounter) -> String {
    format!(
        "{{\"rotations\": {}, \"relins\": {}, \"sse_switches\": {}, \"key_switches\": {}, \
         \"ct_ct_tensor\": {}, \"ct_pt_mults\": {}, \"rescales\": {}, \"bootstraps\": {}}}",
        c.rotations,
        c.relins,
        c.sse_switches,
        c.key_switches(),
        c.ct_ct_tensor,
        c.ct_pt_mults,
        c.rescales,
        c.bootstraps
    )
}

fn jstr(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn write_json(path: &str, body: &str) {
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).expect("create output dir");
    }
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, body).expect("write json");
    std::fs::rename(&tmp, path).expect("rename json");
}

// ---------------------------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------------------------

struct Setup {
    client: Client,
    ev: Evaluator,
    layout: Layout,
    keys: usize,
    keygen_seconds: f64,
    lanes: usize,
}

fn setup(o: &Opts, rng: &mut Rng) -> Setup {
    let params = Arc::new(preset(&o.preset));
    let lanes = o.batch.next_power_of_two();
    let d = DIMS
        .iter()
        .map(|d| d.next_power_of_two())
        .max()
        .expect("dims");
    let layout = Layout::new(params.slots(), d, lanes).expect("layout");
    let steps = mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).expect("steps");
    let t0 = Instant::now();
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng.fork()).expect("keygen");
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(&steps);
    let keygen_seconds = t0.elapsed().as_secs_f64();
    let client = Client {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        params: Arc::clone(&params),
    };
    eprintln!(
        "# preset {} (N = 2^{}, max_level {}), layout d = {d}, B = {lanes}, {} Galois keys \
         {steps:?}, keygen {keygen_seconds:.1} s",
        o.preset,
        params.n().trailing_zeros(),
        params.max_level(),
        steps.len()
    );
    Setup {
        ev: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
        client,
        layout,
        keys: steps.len(),
        keygen_seconds,
        lanes,
    }
}

fn config_json(o: &Opts, s: &Setup, lr_eff: f64, data: &Data) -> String {
    format!(
        "{{\"network\": \"4-10-3\", \"activation\": \"square\", \"loss\": \"mse_mean_over_outputs\", \
         \"optimizer\": {}, \"lr\": {}, \"lr_eff\": {}, \"momentum\": {}, \"batch\": {}, \
         \"lanes\": {}, \"layout_d\": {}, \"padded_dims\": [4, 16, 4], \"epochs\": {}, \
         \"seed\": {}, \"init\": \"N(0, {}^2) ChaCha20\", \"preset\": {}, \"n\": {}, \
         \"max_level\": {}, \"galois_keys\": {}, \"keygen_seconds\": {}, \"train\": {}, \
         \"test\": {}, \"split\": \"spikes/seal-2020 split_80_20 seed 2020\", \
         \"preprocessing\": \"z-score, training-set mean and population sd\"}}",
        jstr(&o.optimizer),
        o.lr,
        lr_eff,
        o.momentum,
        o.batch,
        s.lanes,
        s.layout.d,
        // Timing mode runs single iterations, no epochs.
        if o.time_iters > 0 { 0 } else { o.epochs },
        o.seed,
        o.sigma,
        jstr(&o.preset),
        s.client.params.n(),
        s.client.params.max_level(),
        s.keys,
        jnum(s.keygen_seconds),
        data.train.len(),
        data.test.len()
    )
}

fn main() -> Result<()> {
    let o = parse_opts();
    let data = load(&o);
    if o.mode == "noninteractive" {
        return run_ni(&o, &data);
    }
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&o.seed.to_le_bytes());
    let mut master = Rng::from_seed(seed);
    let mut init_rng = master.fork();
    let mut shuffle_rng = master.fork();
    let mut enc_rng = master.fork();
    let refresh_rng = master.fork();

    let load_start = load_avg();
    let s = setup(&o, &mut master);
    let c = &s.client;
    let lr_eff = o.lr * 2.0 / CLASSES as f64;

    let twin0 = Net::init(&mut init_rng, o.sigma);
    let plain: Vec<(Array2<f64>, Vec<f64>)> = twin0
        .layers
        .iter()
        .map(|(w, b)| (w.clone(), b.to_vec()))
        .collect();
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(Square), Box::new(Square)];
    let mut mlp = Mlp::encrypt(
        &s.ev,
        &plain,
        acts,
        &s.layout,
        &c.enc,
        &c.encoder,
        &mut enc_rng,
    )?;
    let spec: BatchSpec = mlp.batch_spec(&s.ev, &c.encoder)?;
    let refresher = ClientRefresher::new(
        c.dec.clone(),
        c.enc.clone(),
        Encoder::new(Arc::clone(&c.params)),
        refresh_rng,
    )
    .with_layout(s.layout)?;
    let mut trainer = Trainer::new(&s.ev, Box::new(refresher), optims(&o, lr_eff), s.layout)?;
    let mut twin = twin0.clone();
    let x_train = rows(&data.x, &data.train);
    let x_test = rows(&data.x, &data.test);
    let e0 = net_error(&decrypt_net(c, &mlp, &s.layout), &twin);
    eprintln!(
        "# init: encrypted vs twin W 2^{:.1}, b 2^{:.1}; x at level {}, y at level {}",
        e0.0.log2(),
        e0.1.log2(),
        spec.x_level,
        spec.y_level
    );

    let wall = Instant::now();
    let body = if o.time_iters > 0 {
        timing(
            &o,
            &s,
            &data,
            &spec,
            &mut mlp,
            &mut trainer,
            &mut twin,
            &mut shuffle_rng,
            &mut enc_rng,
        )?
    } else {
        let mut epochs_json = Vec::new();
        let mut order = data.train.clone();

        let out = o
            .out
            .clone()
            .unwrap_or_else(|| format!("results/iris_{}.json", o.batch));
        let mut summary = String::new();
        for epoch in 0..o.epochs {
            shuffle(&mut order, &mut shuffle_rng);
            let t_enc = Instant::now();
            let mut batches = Vec::new();
            let mut plain_batches = Vec::new();
            for chunk in order.chunks(o.batch) {
                let xb = rows(&data.x, chunk);
                let yb = onehot(&data.labels, chunk);
                let (xc, yc) =
                    spec.encrypt(&c.enc, &c.encoder, &s.layout, &xb, &yb, &mut enc_rng)?;
                batches.push((xc, yc, chunk.len()));
                plain_batches.push((xb, yb));
            }
            let enc_seconds = t_enc.elapsed().as_secs_f64();
            let log: EpochLog = trainer.train_epoch_valid(&mut mlp, &batches)?;
            for (xb, yb) in &plain_batches {
                twin.sgd_step(xb, yb, o.lr);
            }
            let he = decrypt_net(c, &mlp, &s.layout);
            let (spread_lane, spread_block) = copy_spread(c, &mlp, &s.layout);
            eprintln!(
                "  copies: lane spread 2^{:.1}, block spread 2^{:.1}",
                spread_lane.log2(),
                spread_block.log2()
            );
            let (err_w, err_b) = net_error(&he, &twin);
            let (tr_loss, tr_acc) = metrics(&he.predict(&x_train), &data.labels, &data.train);
            let (te_loss, te_acc) = metrics(&he.predict(&x_test), &data.labels, &data.test);
            let (tw_loss, tw_acc) = metrics(&twin.predict(&x_train), &data.labels, &data.train);
            let (tw_te_loss, tw_te_acc) = metrics(&twin.predict(&x_test), &data.labels, &data.test);
            eprintln!(
                "epoch {epoch:3} {:.1}s (refresh {:.2}s, enc {:.2}s) loss {tr_loss:.5} \
                 acc {tr_acc:.4}/{te_acc:.4} | twin {tw_loss:.5} {tw_acc:.4}/{tw_te_acc:.4} | \
                 err W 2^{:.1} b 2^{:.1} | wall {:.0}s load {}",
                log.seconds,
                log.refresh_seconds,
                enc_seconds,
                err_w.log2(),
                err_b.log2(),
                wall.elapsed().as_secs_f64(),
                if epoch % 20 == 0 {
                    load_avg()
                } else {
                    String::new()
                }
            );
            epochs_json.push(format!(
                "{{\"epoch\": {}, \"seconds\": {}, \"refresh_seconds\": {}, \
                 \"client_encrypt_seconds\": {}, \"batches\": {}, \"refreshes\": {}, \
                 \"counter\": {}, \"train_loss\": {}, \"train_acc\": {}, \"test_loss\": {}, \
                 \"test_acc\": {}, \"twin_train_loss\": {}, \"twin_train_acc\": {}, \
                 \"twin_test_loss\": {}, \"twin_test_acc\": {}, \"weight_err\": {}, \
                 \"bias_err\": {}}}",
                log.epoch,
                jnum(log.seconds),
                jnum(log.refresh_seconds),
                jnum(enc_seconds),
                log.batches,
                log.refreshes,
                jcounter(&log.counter),
                jnum(tr_loss),
                tr_acc,
                jnum(te_loss),
                te_acc,
                jnum(tw_loss),
                tw_acc,
                jnum(tw_te_loss),
                tw_te_acc,
                jnum(err_w),
                jnum(err_b)
            ));
            let diverged = !tr_loss.is_finite() || tr_loss > 1e6;
            let last = epoch + 1 == o.epochs || diverged;
            summary = format!(
                "\"complete\": {last}, \"diverged\": {diverged}, \"epochs_done\": {}, \
                 \"wall_seconds\": {}, \"load_start\": {}, \"load_now\": {}, \
                 \"final\": {{\"train_loss\": {}, \"train_acc\": {tr_acc}, \"test_loss\": {}, \
                 \"test_acc\": {te_acc}, \"twin_train_loss\": {}, \"twin_train_acc\": {tw_acc}, \
                 \"twin_test_loss\": {}, \"twin_test_acc\": {tw_te_acc}, \"weight_err\": {}, \
                 \"bias_err\": {}}}",
                epoch + 1,
                jnum(wall.elapsed().as_secs_f64()),
                jstr(&load_start),
                jstr(&load_avg()),
                jnum(tr_loss),
                jnum(te_loss),
                jnum(tw_loss),
                jnum(tw_te_loss),
                jnum(err_w),
                jnum(err_b)
            );
            if !last && o.checkpoint_every > 0 && (epoch + 1) % o.checkpoint_every == 0 {
                write_json(
                    &out,
                    &train_json(&o, &s, lr_eff, &data, &summary, &trainer, &epochs_json),
                );
            }
            if diverged {
                break;
            }
        }
        let body = train_json(&o, &s, lr_eff, &data, &summary, &trainer, &epochs_json);
        write_json(&out, &body);
        eprintln!("# wrote {out}; wall {:.0} s", wall.elapsed().as_secs_f64());
        return Ok(());
    };
    let out = o
        .out
        .clone()
        .unwrap_or_else(|| format!("results/iris_{}_timing.json", o.batch));
    let full = format!(
        "{{\"task\": \"31\", \"mode\": \"timing\", \"config\": {}, \"load_start\": {}, \
         \"load_end\": {}, {body}, \"trace\": {}}}\n",
        config_json(&o, &s, lr_eff, &data),
        jstr(&load_start),
        jstr(&load_avg()),
        trace_json(&trainer)
    );
    write_json(&out, &full);
    eprintln!("# wrote {out}");
    Ok(())
}

fn train_json(
    o: &Opts,
    s: &Setup,
    lr_eff: f64,
    data: &Data,
    summary: &str,
    trainer: &Trainer<'_>,
    epochs: &[String],
) -> String {
    format!(
        "{{\"task\": \"31\", \"mode\": \"train\", \"config\": {}, {summary}, \"trace\": {}, \
         \"epochs\": [\n  {}\n]}}\n",
        config_json(o, s, lr_eff, data),
        trace_json(trainer),
        epochs.join(",\n  ")
    )
}

fn trace_json(t: &Trainer<'_>) -> String {
    let items: Vec<String> = t
        .trace()
        .iter()
        .map(|(k, l)| format!("[{}, {l}]", jstr(k)))
        .collect();
    format!("[{}]", items.join(", "))
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let mu = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mu, var.sqrt())
}

/// `warmup + time_iters` single iterations; returns the JSON fields.
#[allow(clippy::too_many_arguments)]
fn timing(
    o: &Opts,
    s: &Setup,
    data: &Data,
    spec: &BatchSpec,
    mlp: &mut Mlp,
    trainer: &mut Trainer<'_>,
    twin: &mut Net,
    shuffle_rng: &mut Rng,
    enc_rng: &mut Rng,
) -> Result<String> {
    let c = &s.client;
    let mut order = data.train.clone();
    shuffle(&mut order, shuffle_rng);
    let (mut enc_t, mut server_t, mut refresh_t, mut total_t) = (vec![], vec![], vec![], vec![]);
    let mut iters = Vec::new();
    let mut counter = OpCounter::default();
    let mut max_err: f64 = 0.0;
    for it in 0..o.warmup + o.time_iters {
        let start = (it * o.batch) % order.len();
        let idx: Vec<usize> = (0..o.batch)
            .map(|k| order[(start + k) % order.len()])
            .collect();
        let xb = rows(&data.x, &idx);
        let yb = onehot(&data.labels, &idx);
        let t0 = Instant::now();
        let (xc, yc) = spec.encrypt(&c.enc, &c.encoder, &s.layout, &xb, &yb, enc_rng)?;
        let enc = t0.elapsed().as_secs_f64();
        let log = trainer.train_epoch_valid(mlp, &[(xc, yc, idx.len())])?;
        twin.sgd_step(&xb, &yb, o.lr);
        let (ew, eb) = net_error(&decrypt_net(c, mlp, &s.layout), twin);
        max_err = max_err.max(ew.max(eb));
        let server = log.seconds - log.refresh_seconds;
        eprintln!(
            "iter {it:2}{} enc {enc:.4}s server {server:.4}s refresh {:.4}s ({} cts) | \
             err W 2^{:.1} b 2^{:.1} | {:?}",
            if it < o.warmup { " (warm-up)" } else { "" },
            log.refresh_seconds,
            log.refreshes,
            ew.log2(),
            eb.log2(),
            log.counter
        );
        iters.push(format!(
            "{{\"iter\": {it}, \"warmup\": {}, \"client_encrypt_seconds\": {}, \
             \"server_seconds\": {}, \"refresh_seconds\": {}, \"refreshes\": {}, \
             \"weight_err\": {}, \"bias_err\": {}}}",
            it < o.warmup,
            jnum(enc),
            jnum(server),
            jnum(log.refresh_seconds),
            log.refreshes,
            jnum(ew),
            jnum(eb)
        ));
        if it >= o.warmup {
            enc_t.push(enc);
            server_t.push(server);
            refresh_t.push(log.refresh_seconds);
            total_t.push(enc + log.seconds);
            counter = log.counter;
        }
    }
    let (e, se) = mean_sd(&enc_t);
    let (sv, ssv) = mean_sd(&server_t);
    let (r, sr) = mean_sd(&refresh_t);
    let (t, st) = mean_sd(&total_t);
    eprintln!(
        "# {} timed iterations: server {sv:.4} ± {ssv:.4} s, refresh {r:.4} ± {sr:.4} s, \
         client encryption {e:.4} ± {se:.4} s, total {t:.4} ± {st:.4} s; max weight error vs \
         twin over all iterations {max_err:e}",
        o.time_iters
    );
    let mut out = String::new();
    write!(
        out,
        "\"timed_iterations\": {}, \"warmup_iterations\": {}, \
         \"server_seconds\": {{\"mean\": {}, \"sd\": {}}}, \
         \"refresh_seconds\": {{\"mean\": {}, \"sd\": {}}}, \
         \"client_encrypt_seconds\": {{\"mean\": {}, \"sd\": {}}}, \
         \"total_seconds\": {{\"mean\": {}, \"sd\": {}}}, \"counter_per_iteration\": {}, \
         \"max_err_vs_twin\": {}, \"iterations\": [\n  {}\n]",
        o.time_iters,
        o.warmup,
        jnum(sv),
        jnum(ssv),
        jnum(r),
        jnum(sr),
        jnum(e),
        jnum(se),
        jnum(t),
        jnum(st),
        jcounter(&counter),
        jnum(max_err),
        iters.join(",\n  ")
    )
    .expect("write to string");
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Task 41: non-interactive mode (bootstrapping, no client in the loop).
// ---------------------------------------------------------------------------------------------

/// The training run of `main` without a client: [`NonInteractiveTrainer`] with a
/// [`Bootstrapper`] on `--preset n16_boot` (or the insecure `mini_boot`), the same data, split,
/// initialisation, batches and plaintext twin. Per epoch: the twin error, losses and accuracies,
/// key switches per sample (bootstraps included) and bootstraps. The secret key is used only
/// after each epoch, for the evaluation.
fn run_ni(o: &Opts, data: &Data) -> Result<()> {
    let pl = placement(&o.placement).expect("checked");
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&o.seed.to_le_bytes());
    let mut master = Rng::from_seed(seed);
    let mut init_rng = master.fork();
    let mut shuffle_rng = master.fork();
    let mut enc_rng = master.fork();
    let load_start = load_avg();
    let params = Arc::new(preset(&o.preset));
    let d = DIMS
        .iter()
        .map(|d| d.next_power_of_two())
        .max()
        .expect("dims");
    let lanes = o.batch.next_power_of_two().min(params.slots() / d);
    if lanes < o.batch {
        eprintln!(
            "batch {} does not fit one ciphertext of {lanes} lanes",
            o.batch
        );
        std::process::exit(2);
    }
    let layout = Layout::new(params.slots(), d, lanes)?;
    let steps = ni_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated)?;
    let t0 = Instant::now();
    let mut kg = KeyGenerator::new(Arc::clone(&params), master.fork())?;
    let encoder = Encoder::new(Arc::clone(&params));
    let bp = BootParams {
        log_slots: params.slots().trailing_zeros(),
        ..BootParams::default_n16()
    };
    let bs = Bootstrapper::new(Arc::clone(&params), bp, &encoder, &mut kg)?;
    let ev = Evaluator::new(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    let c = Client {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), kg.public_key()),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        params: Arc::clone(&params),
    };
    drop(kg);
    let keygen_seconds = t0.elapsed().as_secs_f64();
    let lr_eff = o.lr * 2.0 / CLASSES as f64;
    let optim = if o.optimizer == "nag" {
        NiOptimizer::Nag {
            lr: lr_eff,
            momentum: o.momentum,
        }
    } else {
        NiOptimizer::Sgd { lr: lr_eff }
    };
    let twin0 = Net::init(&mut init_rng, o.sigma);
    let plain: Vec<(Array2<f64>, Vec<f64>)> = twin0
        .layers
        .iter()
        .map(|(w, b)| (w.clone(), b.to_vec()))
        .collect();
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(Square), Box::new(Square)];
    let (mut mlp, state) = NonInteractiveTrainer::encrypt_network(
        &ev,
        &bs,
        &plain,
        acts,
        &layout,
        pl,
        &c.enc,
        &c.encoder,
        &mut enc_rng,
    )?;
    // The server side: evaluation keys and the bootstrapper only.
    let mut trainer = NonInteractiveTrainer::new(&ev, &bs, optim, layout, pl, &mlp, state)?;
    let spec = trainer.batch_spec(&mlp)?;
    eprintln!(
        "# non-interactive {} on {} (N = 2^{}), layout d = {d}, B = {lanes}, {} Galois keys + \
         bootstrap keys, keygen {keygen_seconds:.1} s, {} levels per step of {}",
        pl.name(),
        o.preset,
        params.n().trailing_zeros(),
        steps.len(),
        trainer.levels_per_iter(),
        trainer.start_level()
    );
    let mut twin = twin0.clone();
    let x_train = rows(&data.x, &data.train);
    let x_test = rows(&data.x, &data.test);
    let mut order = data.train.clone();
    let mut epochs_json = Vec::new();
    let wall = Instant::now();
    for epoch in 0..o.epochs {
        shuffle(&mut order, &mut shuffle_rng);
        let mut batches = Vec::new();
        let mut plain_batches = Vec::new();
        for chunk in order.chunks(o.batch) {
            let xb = rows(&data.x, chunk);
            let yb = onehot(&data.labels, chunk);
            let (xc, yc) = spec.encrypt(&c.enc, &c.encoder, &layout, &xb, &yb, &mut enc_rng)?;
            batches.push((xc, yc, chunk.len()));
            plain_batches.push((xb, yb));
        }
        let before = ev.counter();
        let mut secs = 0.0;
        // Activation-bootstrap diagnostic (AfterActivation; client side, after each update and
        // outside the timed step): the bootstrapped hidden activations against the twin's.
        let (mut act_err, mut act_max, mut act_above_1) = (0.0f64, 0.0f64, 0usize);
        for (i, ((xc, yc, n), (xb, yb))) in batches.iter().zip(&plain_batches).enumerate() {
            let hidden = twin.forward(xb)[0].1.clone();
            act_max = act_max.max(hidden.iter().fold(0.0, |m, v| m.max(v.abs())));
            act_above_1 += hidden.iter().filter(|v| v.abs() > 1.0).count();
            trainer.set_next_batch_full(batches.get(i + 1).map(|b| b.2 == lanes));
            let t_b = Instant::now();
            trainer.train_batch_valid(&mut mlp, xc, yc, *n)?;
            secs += t_b.elapsed().as_secs_f64();
            if let Some(outs) = trainer.last_activation_bootstraps().first() {
                let a = unpack_activations(
                    &c.encoder.decode(&c.dec.decrypt(&outs[0])),
                    &layout,
                    DIMS[1],
                );
                for r in 0..*n {
                    for j in 0..DIMS[1] {
                        act_err = act_err.max((a[[r, j]] - hidden[[r, j]]).abs());
                    }
                }
            }
            twin.sgd_step(xb, yb, o.lr);
        }
        let counter = &ev.counter() - &before;
        let he = decrypt_net(&c, &mlp, &layout);
        let (err_w, err_b) = net_error(&he, &twin);
        let (tr_loss, tr_acc) = metrics(&he.predict(&x_train), &data.labels, &data.train);
        let (te_loss, te_acc) = metrics(&he.predict(&x_test), &data.labels, &data.test);
        let samples = data.train.len() as f64;
        eprintln!(
            "epoch {epoch:3} {secs:.1}s loss {tr_loss:.5} acc {tr_acc:.4}/{te_acc:.4} | err W \
             2^{:.1} b 2^{:.1} | {:.1} KS/sample, {} bootstraps | wall {:.0}s",
            err_w.log2(),
            err_b.log2(),
            counter.key_switches() as f64 / samples,
            counter.bootstraps,
            wall.elapsed().as_secs_f64()
        );
        epochs_json.push(format!(
            "{{\"epoch\": {epoch}, \"seconds\": {}, \"counter\": {}, \
             \"key_switches_per_sample\": {}, \"train_loss\": {}, \"train_acc\": {tr_acc}, \
             \"test_loss\": {}, \"test_acc\": {te_acc}, \"weight_err\": {}, \"bias_err\": {}, \
             \"max_abs_hidden_activation\": {}, \"hidden_activations_above_1\": {act_above_1}, \
             \"activation_bootstrap_err_vs_twin\": {}}}",
            jnum(secs),
            jcounter(&counter),
            jnum(counter.key_switches() as f64 / samples),
            jnum(tr_loss),
            jnum(te_loss),
            jnum(err_w),
            jnum(err_b),
            jnum(act_max),
            jnum(act_err)
        ));
    }
    let out = o
        .out
        .clone()
        .unwrap_or_else(|| format!("results/iris_ni_{}_{}.json", pl.name(), o.batch));
    write_json(
        &out,
        &format!(
            "{{\"task\": \"41\", \"mode\": \"noninteractive\", \"placement\": {}, \"preset\": {}, \
             \"batch\": {}, \"lanes\": {lanes}, \"lr\": {}, \"lr_eff\": {}, \"optimizer\": {}, \
             \"twin_optimizer\": \"sgd (the twin is SGD for every --optimizer, as in interactive \
             mode)\", \"activation_bootstrap_note\": \"after_activation only: bootstrapped hidden \
             activations decrypted by the client after each update against the twin's; inputs \
             above 1 violate the bootstrap's |m| <= 1 precondition and are not clipped\", \
             \"levels_per_iter\": {}, \"galois_keys\": {}, \"boot_ks\": {}, \"load_start\": {}, \
             \"load_end\": {}, \"wall_seconds\": {}, \"epochs\": [\n  {}\n]}}\n",
            jstr(pl.name()),
            jstr(&o.preset),
            o.batch,
            o.lr,
            lr_eff,
            jstr(&o.optimizer),
            trainer.levels_per_iter(),
            steps.len(),
            bs.key_switches_per_boot(),
            jstr(&load_start),
            jstr(&load_avg()),
            jnum(wall.elapsed().as_secs_f64()),
            epochs_json.join(",\n  ")
        ),
    );
    eprintln!("# wrote {out}");
    Ok(())
}
