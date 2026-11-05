//! Task 48 Step 2: the RQ3 ablation runner. One interactive eMLP-1 training iteration
//! (784 -> 32 (z^2 + z) -> 10 (linear), MSE, NAG, one batch ciphertext, client refresh) under
//! each of the 8 [`EvalConfig`] combinations (`hoist` x `bsgs` x `lazy_relin`) on one backend,
//! with the time of an iteration and its key-switch counts, written to
//! `results/ablation_<backend>.json`.
//!
//! # Protocol
//!
//! * Keys are generated once; every configuration gets its own evaluator over the same keys
//!   (the CPU clones them; the GPU uploads them once and shares them,
//!   `CudaEvaluator::with_config`), the same initial network (same seed) and the same batches
//!   (the first `(warmup + iters) * batch` MNIST training images, or deterministic uniform
//!   pixels when the IDX files are absent: time and counts do not depend on the values).
//! * Per configuration: `--warmup` untimed iterations, then `--iters` timed ones. An iteration
//!   is one `Trainer::train_epoch_valid` on one batch. Its **step** time is the server side
//!   (forward, backward, gradients, optimizer) including the wait for the device queue
//!   (`hekit_bench::backend::SyncedRefresher`); its **refresh** time is the host-side client
//!   refresh (download, decrypt, re-encrypt, upload), which the flags do not change. The
//!   reported times are medians over the timed iterations.
//! * Counts: the `OpCounter` of one iteration (identical in every iteration; checked). The
//!   closed form: the default configuration's measured count plus, for every matrix-vector
//!   product of the step (forward of each layer, backward of every layer but the first), the
//!   difference of [`EvalConfig::matvec_rotations`] / [`EvalConfig::matvec_enc_relins`] from
//!   the default flags (nothing else in an interactive step depends on the flags); checked for
//!   every configuration (`closed_form_matches`).
//! * Values: the decrypted weights after the last iteration against the default
//!   configuration's (the flags change costs, never the message: equal up to CKKS noise,
//!   bit-identical for `hoist`).
//! * On the GPU the pool is trimmed (`trim_pool(0)`) after the key upload, after every
//!   network upload and after every configuration.
//!
//! # What the RQ3 table must state (copied from `hekit_ckks::ops`)
//!
//! The output's `caveats` carry the four statements of `hekit_ckks::ops`, "What the RQ3 table
//! must state", verbatim: the definition of `bsgs = false` with its `d = 16` counts; that the
//! measured BSGS gain is an upper bound and the `hoist` / `lazy_relin` gains inside
//! `bsgs = false` configurations are lower bounds; that `hoist` changes no `OpCounter` value
//! (wall-clock only); and that `bsgs` does not touch the gradient's outer product.
//!
//! # GPU memory
//!
//! The run estimates its device-memory need before any key exists
//! (`hekit_bench::backend::mlp_gpu_need`) and exits with code 3 and a `SKIP` line if the free
//! memory is below it. n16_train with `--top-level 7` (keys for levels <= 7, 24 MiB each)
//! declares about 2.4 GiB with NAG and about 2.0 GiB with `--optimizer sgd`; n16_train at the
//! full level 31 declares about 15.2 GiB and does not fit the shared card.
//!
//! Usage: `ablation [--backend cpu|cuda] [--params n16_train|small] [--top-level L]
//! [--iters 3] [--warmup 1] [--batch 32] [--hidden 32] [--seed 48] [--lr 0.001]
//! [--momentum 0.9] [--optimizer nag|sgd] [--gpu-pool default|after-keys] [--data data/mnist]
//! [--configs all|default] [--out results/ablation_<backend>.json]`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use hekit_bench::backend::{
    Backend, BackendOpts, CpuHooks, DeviceHooks, SKIP_EXIT_CODE, SKIP_PREFIX, SyncedRefresher,
    config_label, mlp_gpu_need,
};
use hekit_bench::data::{Mnist, Split, load_mnist, one_hot};
use hekit_bench::twin::{Layer, init_net, rng, uni};
use hekit_ckks::{
    CkksParams, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, HeOps, KeyGenerator,
    OpCounter,
};
use hekit_nn::activation::{Activation, Identity, PolyRelu2};
use hekit_nn::dense::TransposeKeys;
use hekit_nn::layout::{LayerShape, Layout, unpack_activations};
use hekit_nn::optim::{Nag, Sgd};
use hekit_nn::refresh::{ClientRefresher, Refresher};
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_key_pool, mlp_rotation_steps};
use ndarray::{Array2, s};
use serde_json::{Value, json};

const CLASSES: usize = 10;

/// The four statements of `hekit_ckks::ops`, "What the RQ3 table must state", verbatim.
const CAVEATS: [&str; 4] = [
    "1. **Definition of `bsgs = false`.** The giant rotation is not shared; the weights keep \
     the same pre-rotated (BSGS) layout and the same Galois keys. Every product with a nonzero \
     giant shift is finished (relinearized) and rotated on its own; the products of the \
     zero-shift block are summed as with BSGS. Per matvec of dimension `d` (offset 0): `d - 1` \
     rotations, of which only the `baby - 1` rotations of `x` can be hoisted; with \
     `lazy_relin` `1 + d - baby` relins, without it `d` ([`EvalConfig::matvec_rotations`], \
     [`EvalConfig::matvec_enc_relins`]). At `d = 16` (`baby = giant = 4`), as (hoistable \
     rotations / individual rotations / relins): | form | hoistable rot. | individual rot. | \
     relins (lazy) | relins (not lazy) | |---|---|---|---|---| | default BSGS | 3 | 3 | 4 | 16 | \
     | `bsgs = false` (this library) | 3 | 12 | 13 | 16 | | textbook diagonal method (not \
     implemented) | 15 | 0 | 1 | 16 | The textbook (Halevi-Shoup) diagonal method rotates `x` \
     by all `d - 1` steps (all hoistable) and sums the products lazily; it needs \
     un-pre-rotated diagonals, i.e. another weight layout and gradient form, so it is not \
     implemented. The all-off configuration has the same counts in both forms (15 rotations, \
     16 relins at `d = 16`).",
    "2. **Bias.** Against the strongest naive baseline (the textbook method with hoisting and \
     lazy relinearization), the cost of the `bsgs = false` rows overstates what BSGS gains: the \
     measured BSGS gain is an **upper bound**. Conversely, the marginal gains of `hoist` and \
     `lazy_relin` measured inside `bsgs = false` configurations are **lower bounds** (only \
     `baby - 1` rotations can be hoisted and only the zero-shift block summed lazily: e.g. \
     `lazy_relin` saves 3 relins at `d = 16` instead of 15).",
    "3. **`hoist` changes no [`OpCounter`] value.** The counter does not distinguish hoisted \
     from individual rotations, so the effect of `hoist` shows only in wall-clock time, never \
     in key-switch counts.",
    "4. **`bsgs` does not touch the gradient's outer product** (`Dense::weight_grad_scaled`, \
     `Dense::weight_grad_reduced`): its `(baby - 1) + (giant - 1)` input rotations are what the \
     pre-rotated weight layout requires, not a BSGS choice; only `hoist` changes how they run. \
     The interactive gradient multiplies each diagonal with one [`Evaluator::mul`] (nothing to \
     sum lazily), so `lazy_relin` does not change it either.",
];

#[derive(Debug, Clone)]
struct Opts {
    be: BackendOpts,
    params: String,
    iters: usize,
    warmup: usize,
    batch: usize,
    hidden: usize,
    seed: u8,
    lr: f64,
    momentum: f64,
    optimizer: String,
    data: PathBuf,
    configs: String,
    out: Option<PathBuf>,
}

fn parse(argv: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        be: BackendOpts::default(),
        params: "n16_train".into(),
        iters: 3,
        warmup: 1,
        batch: 32,
        hidden: 32,
        seed: 48,
        lr: 0.001,
        momentum: 0.9,
        optimizer: "nag".into(),
        data: PathBuf::from("data/mnist"),
        configs: "all".into(),
        out: None,
    };
    let mut it = argv.iter().cloned();
    while let Some(flag) = it.next() {
        if o.be.take(&flag, &mut || it.next())? {
            continue;
        }
        let v = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag} {v}: {e}"));
        let real = |v: &str| v.parse::<f64>().map_err(|e| format!("{flag} {v}: {e}"));
        match flag.as_str() {
            "--params" => o.params = v,
            "--iters" => o.iters = num(&v)?,
            "--warmup" => o.warmup = num(&v)?,
            "--batch" => o.batch = num(&v)?,
            "--hidden" => o.hidden = num(&v)?,
            "--seed" => o.seed = v.parse().map_err(|e| format!("--seed {v}: {e}"))?,
            "--lr" => o.lr = real(&v)?,
            "--momentum" => o.momentum = real(&v)?,
            "--optimizer" => o.optimizer = v,
            "--data" => o.data = PathBuf::from(v),
            "--configs" => o.configs = v,
            "--out" => o.out = Some(PathBuf::from(v)),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    o.be.validate()?;
    if o.be.config != EvalConfig::default() {
        return Err("the ablation runs every flag combination itself; drop --no-*".into());
    }
    if o.iters == 0 || o.batch == 0 || o.hidden == 0 {
        return Err("--iters, --batch and --hidden must be positive".into());
    }
    if !matches!(o.optimizer.as_str(), "nag" | "sgd") {
        return Err(format!("unknown --optimizer {} (nag | sgd)", o.optimizer));
    }
    if !matches!(o.configs.as_str(), "all" | "default") {
        return Err(format!("unknown --configs {} (all | default)", o.configs));
    }
    Ok(o)
}

fn params_named(name: &str) -> Result<CkksParams, String> {
    match name {
        "n16_train" => Ok(CkksParams::n16_train()),
        // Insecure (N = 2^12 with 11 primes): the mnist bin's smoke preset (d = 1024 needs
        // --batch 2).
        "small" => {
            let mut q = vec![58];
            q.extend([42; 10]);
            CkksParams::new(12, &q, &[60], 42, 11).map_err(|e| e.to_string())
        }
        _ => Err(format!("unknown --params {name} (n16_train | small)")),
    }
}

fn main() {
    let o = match parse(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("ablation: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&o) {
        eprintln!("ablation: {e}");
        std::process::exit(if e.starts_with(SKIP_PREFIX) {
            SKIP_EXIT_CODE
        } else {
            1
        });
    }
}

/// The training batches: real MNIST if the IDX files are there, else deterministic uniform
/// pixels (time and counts do not depend on the values).
fn batches(o: &Opts, n: usize) -> (Mnist, &'static str) {
    if let Ok(m) = load_mnist(&o.data, Split::Train, Some(n)) {
        if m.len() == n {
            return (m, "MNIST training images (first in file order)");
        }
    }
    let mut r = rng(o.seed, 9);
    let images = Array2::from_shape_fn((n, 784), |_| 0.5 * (uni(&mut r) + 1.0));
    let labels = (0..n).map(|i| (i % CLASSES) as u8).collect();
    (
        Mnist {
            images,
            labels,
            rows: 28,
            cols: 28,
        },
        "synthetic uniform pixels (MNIST IDX files not found)",
    )
}

/// What the configurations share: keys, client, data, layout.
struct Ctx<'a> {
    o: &'a Opts,
    params: Arc<CkksParams>,
    dims: [usize; 3],
    layout: Layout,
    keys: TransposeKeys,
    top: usize,
    enc: Encryptor,
    dec: Decryptor,
    encoder: Encoder,
    data: Mnist,
    init: Vec<Layer>,
}

/// One configuration's measurements.
struct Measured {
    cfg: EvalConfig,
    counters: Vec<OpCounter>,
    step_s: Vec<f64>,
    refresh_s: Vec<f64>,
    weights: Vec<Layer>,
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n == 0 {
        f64::NAN
    } else if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Trains `warmup + iters` iterations on `ev` from the shared initial network; returns the
/// timed iterations' counters and times and the final decrypted weights.
fn run_config<'e, E: HeOps, R: Refresher<E> + 'e>(
    ctx: &Ctx<'_>,
    ev: &'e E,
    refresher: R,
    hooks: &'e dyn DeviceHooks,
) -> Result<Measured, String> {
    let o = ctx.o;
    let e2s = |e: hekit_core::error::HekitError| e.to_string();
    let label = config_label(ev.config());
    let plain: Vec<(Array2<f64>, Vec<f64>)> = ctx
        .init
        .iter()
        .map(|l| (l.w.clone(), l.b.clone()))
        .collect();
    let acts: Vec<Box<dyn Activation<E>>> = vec![Box::new(PolyRelu2), Box::new(Identity)];
    let mut r = rng(o.seed, 4);
    let mut mlp = Mlp::encrypt_at(
        ev,
        &plain,
        acts,
        &ctx.layout,
        ctx.top,
        &ctx.enc,
        &ctx.encoder,
        &mut r,
    )
    .map_err(e2s)?;
    let spec = mlp.batch_spec_at(ev, &ctx.encoder, ctx.top).map_err(e2s)?;
    hooks.trim(&format!("{label}: network upload"))?;
    let (synced, wait) = SyncedRefresher::new(refresher, hooks);
    let optims: Vec<Box<dyn LayerOptimizer<E> + 'e>> = (0..2)
        .map(|_| -> Box<dyn LayerOptimizer<E> + 'e> {
            if o.optimizer == "nag" {
                Box::new(Nag::<E::Ct>::new_for(o.lr, o.momentum))
            } else {
                Box::new(Sgd { lr: o.lr })
            }
        })
        .collect();
    let mut trainer = Trainer::new(ev, Box::new(synced), optims, ctx.layout)
        .map_err(e2s)?
        .with_transpose_keys(ctx.keys.clone());
    let y_all = one_hot(&ctx.data.labels, CLASSES);
    let mut m = Measured {
        cfg: *ev.config(),
        counters: Vec::new(),
        step_s: Vec::new(),
        refresh_s: Vec::new(),
        weights: Vec::new(),
    };
    for it in 0..o.warmup + o.iters {
        let (a, b) = (it * o.batch, (it + 1) * o.batch);
        let x = ctx.data.images.slice(s![a..b, ..]).to_owned();
        let y = y_all.slice(s![a..b, ..]).to_owned();
        let (xc, yc) = spec
            .encrypt(&ctx.enc, &ctx.encoder, &ctx.layout, &x, &y, &mut r)
            .map_err(e2s)?;
        let xc = ev.upload_ct_owned(xc).map_err(e2s)?;
        let yc = ev.upload_ct_owned(yc).map_err(e2s)?;
        hooks.sync()?;
        let w0 = wait.get();
        let log = trainer
            .train_epoch_valid(&mut mlp, &[(xc, yc, o.batch)])
            .map_err(|e| format!("{label}: iteration {it}: {e}"))?;
        let w = wait.get() - w0;
        let (step, refresh) = (
            log.seconds - log.refresh_seconds + w,
            log.refresh_seconds - w,
        );
        eprintln!(
            "ablation: {label} iter {it}{}: step {step:.3} s, refresh {refresh:.3} s, {} rot + \
             {} relin",
            if it < o.warmup { " (warm-up)" } else { "" },
            log.counter.rotations,
            log.counter.relins
        );
        if it >= o.warmup {
            m.counters.push(log.counter);
            m.step_s.push(step);
            m.refresh_s.push(refresh);
        }
    }
    for (dn, _) in &mlp.layers {
        let h = dn.download(ev).map_err(e2s)?;
        m.weights.push(Layer {
            w: h.w.decrypt(&ctx.dec, &ctx.encoder, &ctx.layout),
            b: unpack_activations(
                &ctx.encoder.decode(&ctx.dec.decrypt(&h.b)),
                &ctx.layout,
                h.w.d_out(),
            )
            .row(0)
            .to_vec(),
        });
    }
    drop(trainer);
    drop(mlp);
    hooks.trim(&format!("{label}: end"))?;
    Ok(m)
}

fn configs(o: &Opts) -> Vec<EvalConfig> {
    if o.configs == "default" {
        vec![EvalConfig::default()]
    } else {
        EvalConfig::all().to_vec()
    }
}

fn run(o: &Opts) -> Result<(), String> {
    let t_all = Instant::now();
    let params = Arc::new(params_named(&o.params)?);
    let n = (o.warmup + o.iters) * o.batch;
    let (data, data_note) = batches(o, n);
    let dims = [data.features(), o.hidden, CLASSES];
    let d = dims
        .iter()
        .map(|x| x.next_power_of_two())
        .max()
        .expect("dims");
    let layout = Layout::new(params.slots(), d, o.batch).map_err(|e| e.to_string())?;
    let keys = TransposeKeys::Reuse(mlp_key_pool(&dims, &layout).map_err(|e| e.to_string())?);
    let steps = mlp_rotation_steps(&dims, &layout, &keys).map_err(|e| e.to_string())?;
    let top = o.be.start_level(&params)?;
    let shapes: Vec<LayerShape> = dims
        .windows(2)
        .map(|p| LayerShape::new(p[1], p[0], &layout))
        .collect();
    let m: Vec<usize> = shapes.iter().map(|s| s.m).collect();
    // The matvecs of one step: the forward of every layer, the backward of every layer but
    // the first.
    let matvecs: Vec<usize> = m.iter().copied().chain(m[1..].iter().copied()).collect();
    let need = mlp_gpu_need(&params, steps.len(), top, &m, o.optimizer == "nag");
    eprintln!(
        "ablation: backend {}, {} (N = 2^{}, start level {top}), dims {dims:?}, d = {d}, \
         B = {}, {} Galois keys, matvecs {matvecs:?}; GPU need {}; data: {data_note}",
        o.be.backend.name(),
        o.params,
        params.log_n(),
        o.batch,
        steps.len(),
        need.json()
    );
    #[cfg(feature = "cuda")]
    let dev = match o.be.backend {
        Backend::Cuda => Some(hekit_bench::backend::open_gpu("ablation", &params, &need)?),
        Backend::Cpu => None,
    };

    let t_keys = Instant::now();
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), rng(o.seed, 1)).map_err(|e| e.to_string())?;
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
    let rk = kg.relin_key();
    let gks = kg.galois_keys(&steps);
    drop(kg);
    let keygen_s = t_keys.elapsed().as_secs_f64();
    let init = init_net(&dims, &mut rng(o.seed, 2), "pytorch", false);
    let ctx = Ctx {
        o,
        params: Arc::clone(&params),
        dims,
        layout,
        keys,
        top,
        enc,
        dec,
        encoder: Encoder::new(Arc::clone(&params)),
        data,
        init,
    };
    let client = |seed_role: u8| -> Result<ClientRefresher, String> {
        ClientRefresher::new(
            ctx.dec.clone(),
            ctx.enc.clone(),
            Encoder::new(Arc::clone(&ctx.params)),
            rng(o.seed, seed_role),
        )
        .with_output_level(top)
        .map_err(|e| e.to_string())
    };

    let (results, device, memory, facts) = match o.be.backend {
        Backend::Cpu => {
            let mut res = Vec::new();
            for cfg in configs(o) {
                let ev = Evaluator::with_config(
                    Arc::clone(&params),
                    Some(rk.clone()),
                    Some(gks.clone()),
                    cfg,
                );
                res.push(run_config(&ctx, &ev, client(3)?, &CpuHooks)?);
            }
            (res, "cpu".to_string(), "[]".to_string(), json!({}))
        }
        Backend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                cuda_configs(
                    &ctx,
                    dev.as_ref().expect("opened"),
                    &rk,
                    &gks,
                    &steps,
                    &client,
                )?
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = (&rk, &gks, &client);
                return Err("--backend cuda needs the `cuda` feature".into());
            }
        }
    };
    report(
        &ctx, &results, &matvecs, &steps, keygen_s, &need, data_note, &device, &memory, facts,
        t_all,
    )
}

/// The 8 configurations on the GPU: the keys are uploaded once (levels <= `top`) and shared by
/// every configuration's evaluator (`CudaEvaluator::with_config`).
#[cfg(feature = "cuda")]
#[allow(clippy::type_complexity)]
fn cuda_configs(
    ctx: &Ctx<'_>,
    dev: &hekit_cuda::CudaDevice,
    rk: &hekit_ckks::RelinKey,
    gks: &hekit_ckks::GaloisKeys,
    steps: &[i32],
    client: &dyn Fn(u8) -> Result<ClientRefresher, String>,
) -> Result<(Vec<Measured>, String, String, Value), String> {
    let hooks = hekit_bench::backend::GpuHooks::new(dev);
    let keys = steps
        .iter()
        .map(|&s| {
            gks.get(s)
                .ok_or_else(|| format!("no Galois key for step {s}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let t = Instant::now();
    let base = hekit_cuda::CudaEvaluator::with_keys(
        dev,
        Arc::clone(&ctx.params),
        Some(rk),
        &keys,
        EvalConfig::default(),
        ctx.top,
    )
    .map_err(|e| e.to_string())?;
    hooks.sync()?;
    let upload_s = t.elapsed().as_secs_f64();
    hooks.trim("key upload")?;
    hooks.apply_pool_policy(ctx.o.be.pool)?;
    let mut res = Vec::new();
    for cfg in configs(ctx.o) {
        let ev = base.with_config(cfg);
        let refresher = hekit_nn::refresh::HostRefresher::new(&ev, client(3)?);
        res.push(run_config(ctx, &ev, refresher, &hooks)?);
    }
    let facts = json!({
        "device_key_level": base.key_level(),
        "device_keys": base.num_galois_keys() + 1,
        "device_key_bytes": base.key_bytes(),
        "device_key_upload_s": upload_s,
    });
    Ok((res, hooks.describe(), hooks.memory_json(), facts))
}

fn counter_value(c: &OpCounter) -> Value {
    json!({
        "rotations": c.rotations,
        "relins": c.relins,
        "key_switches": c.key_switches(),
        "ct_ct_tensor": c.ct_ct_tensor,
        "ct_pt_mults": c.ct_pt_mults,
        "rescales": c.rescales,
        "bootstraps": c.bootstraps,
    })
}

fn max_weight_diff(a: &[Layer], b: &[Layer]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            x.w.iter()
                .zip(&y.w)
                .chain(x.b.iter().zip(&y.b))
                .map(|(p, q)| (p - q).abs())
        })
        .fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn report(
    ctx: &Ctx<'_>,
    results: &[Measured],
    matvecs: &[usize],
    steps: &[i32],
    keygen_s: f64,
    need: &hekit_bench::backend::GpuNeed,
    data_note: &str,
    device: &str,
    memory: &str,
    facts: Value,
    t_all: Instant,
) -> Result<(), String> {
    let o = ctx.o;
    let def = EvalConfig::default();
    let base = results
        .iter()
        .find(|m| m.cfg == def)
        .ok_or("the default configuration did not run")?;
    let base_c = base.counters[0].clone();
    let naive_step = results
        .iter()
        .find(|m| m.cfg == EvalConfig::NAIVE)
        .map(|m| median(&m.step_s));
    let base_step = median(&base.step_s);
    let mut rows = Vec::new();
    let mut all_ok = true;
    for m in results {
        let c = &m.counters[0];
        let constant = m.counters.iter().all(|x| x == c);
        // Closed form: the default count plus the flag deltas of every matvec.
        let d_rot: i64 = matvecs
            .iter()
            .map(|&d| m.cfg.matvec_rotations(d) as i64 - def.matvec_rotations(d) as i64)
            .sum();
        let d_relin: i64 = matvecs
            .iter()
            .map(|&d| m.cfg.matvec_enc_relins(d) as i64 - def.matvec_enc_relins(d) as i64)
            .sum();
        let pred_rot = base_c.rotations as i64 + d_rot;
        let pred_relin = base_c.relins as i64 + d_relin;
        let matches = c.rotations as i64 == pred_rot && c.relins as i64 == pred_relin;
        // hoist changes no count.
        let hoist_pair = results
            .iter()
            .find(|x| {
                x.cfg
                    == EvalConfig {
                        hoist: true,
                        ..m.cfg
                    }
            })
            .map(|x| &x.counters[0] == c);
        let wdiff = max_weight_diff(&m.weights, &base.weights);
        all_ok &= constant && matches && hoist_pair != Some(false);
        let step = median(&m.step_s);
        rows.push(json!({
            "config": config_label(&m.cfg),
            "hoist": m.cfg.hoist,
            "bsgs": m.cfg.bsgs,
            "lazy_relin": m.cfg.lazy_relin,
            "counter_per_iteration": counter_value(c),
            "counter_constant_over_iterations": constant,
            "closed_form": {"rotations": pred_rot, "relins": pred_relin},
            "closed_form_matches": matches,
            "counts_equal_hoist_on_counterpart": hoist_pair,
            "step_s_median": step,
            "step_s": m.step_s,
            "refresh_s_median": median(&m.refresh_s),
            "iteration_s_median": step + median(&m.refresh_s),
            "step_speedup_over_all_off": naive_step.map(|n| n / step),
            "step_time_over_default": step / base_step,
            "max_abs_weight_diff_vs_default": wdiff,
            "log2_max_abs_weight_diff_vs_default": wdiff.log2(),
        }));
        eprintln!(
            "ablation: {}: {} rot + {} relin = {} KS (closed form {}), step {step:.3} s \
             (x{:.3} of default), weights 2^{:.1} from default",
            config_label(&m.cfg),
            c.rotations,
            c.relins,
            c.key_switches(),
            if matches { "ok" } else { "MISMATCH" },
            step / base_step,
            wdiff.log2()
        );
    }
    let result = json!({
        "task": "Task 48: RQ3 ablation (hoist x bsgs x lazy_relin), one interactive eMLP-1 iteration",
        "backend": {
            "options": serde_json::from_str::<Value>(&o.be.json()).map_err(|e| e.to_string())?,
            "device": device,
            "gpu_need_estimate": serde_json::from_str::<Value>(&need.json()).map_err(|e| e.to_string())?,
            "memory_after_trims": serde_json::from_str::<Value>(memory).map_err(|e| e.to_string())?,
            "facts": facts,
            "bootstrap": "none (interactive: client refresh on the host, behind HostRefresher on the GPU)",
        },
        "setup": {
            "params": o.params,
            "log_n": ctx.params.log_n(),
            "max_level": ctx.params.max_level(),
            "start_level": ctx.top,
            "level_capped": ctx.top < ctx.params.max_level(),
            "dims": ctx.dims,
            "layout_d": ctx.layout.d,
            "lanes_b": ctx.layout.batch,
            "galois_keys": steps.len(),
            "transpose_keys": "reuse (mlp_key_pool)",
            "matvec_dims_per_step": matvecs,
            "optimizer": o.optimizer,
            "lr": o.lr,
            "momentum": o.momentum,
            "seed": o.seed,
            "data": data_note,
            "warmup_iterations": o.warmup,
            "timed_iterations": o.iters,
            "keygen_s": keygen_s,
        },
        "protocol": "per configuration: same keys, same initial network and batches; warm-up \
                     iterations untimed; step = server side incl. the wait for the device queue; \
                     refresh = host client refresh (unchanged by the flags); medians over the \
                     timed iterations; closed form = default count + per-matvec deltas of \
                     EvalConfig::matvec_rotations / matvec_enc_relins",
        "caveats": CAVEATS,
        "all_checks_pass": all_ok,
        "configs": rows,
        "machine": {
            "threads": if cfg!(feature = "parallel") {
                format!(
                    "rayon (parallel feature), RAYON_NUM_THREADS={}",
                    std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset (all cores)".into())
                )
            } else {
                "single (no parallel feature)".into()
            },
            "wall_total_s": t_all.elapsed().as_secs_f64(),
        },
    });
    let out = o
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("results/ablation_{}.json", o.be.backend.name())));
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())? + "\n";
    std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
    eprintln!(
        "ablation: wrote {} (all checks pass: {all_ok})",
        out.display()
    );
    if all_ok {
        Ok(())
    } else {
        Err(
            "a count check failed (see closed_form_matches / counter_constant_over_iterations)"
                .into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caveats_are_the_ops_docs_verbatim() {
        // Every caveat's text (whitespace-normalised, without markdown table layout) appears in
        // the hekit_ckks::ops module docs.
        let ops = include_str!("../../../hekit-ckks/src/ops.rs");
        let norm = |s: &str| {
            s.replace("//!", " ")
                .replace('|', " ")
                .replace("---", " ")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
        let doc = norm(ops);
        for c in CAVEATS {
            assert!(doc.contains(&norm(c)), "not verbatim: {c}");
        }
    }

    #[test]
    fn options() {
        let p = |a: &[&str]| parse(&a.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let o = p(&[]).unwrap();
        assert_eq!((o.iters, o.warmup, o.batch), (3, 1, 32));
        assert!(p(&["--no-bsgs"]).is_err());
        assert!(p(&["--matvec", "batch"]).is_err());
        assert!(p(&["--optimizer", "adam"]).is_err());
        assert_eq!(p(&["--top-level", "7"]).unwrap().be.top_level, Some(7));
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
    }
}
