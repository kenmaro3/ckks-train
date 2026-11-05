//! Task 33 Step 5: ReBoot's per-sample layout (`hekit_nn::layout_reboot`, written from the
//! paper arXiv:2506.19693 only) trained interactively on MNIST eMLP-1 on hekit, the RQ1
//! baseline (c): same library, parameters, data, subset, batch, learning rate and seed as
//! `mnist` (Task 32), only the packing differs.
//!
//! # Network and training
//!
//! `784 -> 32` with `z^2 + z` ([`PolyRelu2`]) then `32 -> 10` linear, MSE against one-hot
//! labels (`delta = y - t`), NAG (`lr = 0.001`, `momentum = 0.9`). Initial weights are drawn
//! exactly as `mnist` does (PyTorch default, same seed), **but there are no biases** (the
//! ReBoot layout here has no bias path; biases would add only plaintext additions, no key
//! switch, so the counts are unchanged). The plaintext twin is therefore bias-free too and the
//! test accuracies are not comparable with `mnist`'s.
//!
//! # Layout and one training step
//!
//! `(r, c) = (1024, 32)` (paper §4.1 rule; `(16, 2048)` cannot hold 784 inputs). Layer 1 is
//! row-encoded (`W1`, 784 x 32), layer 2 column-encoded (`W2`, 32 x 10). One sample per
//! ciphertext. Per sample (server): `z1 = vecmat(x_exp, W1)`, `a1 = f(z1)`,
//! `y = vecmat_col(a1, W2)`, `g = y - t`, `da1 = vecmat(g, W2)`, `dz1 = f'(z1) da1`. Per batch:
//! `dW1 = outer_sum(x, dz1)`, `dW2 = outer_sum(g, a1)` with the chosen relinearisation policy
//! (`--mode lazy|persample`). Interactive update: the client decrypts `dW1`, `dW2`, applies
//! NAG in plaintext (`G = dW / n`, `V' = mu V + G`, `W' = W - lr (G + mu V')`; Alg. 1 line
//! 10's delayed gradient is not modelled) and encrypts the new weights on the top level. The
//! update has no key switch. Weights and inputs are encrypted at `max_level`.
//!
//! Usage: `mnist_reboot_layout --mode lazy|persample [--subset 1024] [--batch 32]
//! [--r 1024] [--c 32] [--params n16_train|small] [--seed 32] [--lr 0.001] [--momentum 0.9]
//! [--test-subset 10000] [--data data/mnist] [--out results/mnist_reboot_<mode>.json]`.
//! Run it under `/usr/bin/time -l` for the peak RSS.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use hekit_bench::data::{Split, load_mnist, one_hot};
use hekit_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, PolyRelu2};
use hekit_nn::layout_reboot::{GradRelin, RebootLayout};
use ndarray::{Array2, s};

const CLASSES: usize = 10;

#[derive(Debug, Clone)]
struct Args {
    mode: GradRelin,
    mode_name: String,
    subset: usize,
    test_subset: usize,
    batch: usize,
    hidden: usize,
    r: usize,
    c: usize,
    params: String,
    seed: u8,
    lr: f64,
    momentum: f64,
    data: PathBuf,
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        mode: GradRelin::Lazy,
        mode_name: String::new(),
        subset: 1024,
        test_subset: 10_000,
        batch: 32,
        hidden: 32,
        r: 1024,
        c: 32,
        params: "n16_train".into(),
        seed: 32,
        lr: 0.001,
        momentum: 0.9,
        data: PathBuf::from("data/mnist"),
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|e| format!("{flag} {v}: {e}"));
        let real = |v: &str| v.parse::<f64>().map_err(|e| format!("{flag} {v}: {e}"));
        match flag.as_str() {
            "--mode" => {
                a.mode = match v.as_str() {
                    "lazy" => GradRelin::Lazy,
                    "persample" => GradRelin::PerSample,
                    _ => return Err(format!("unknown --mode {v} (lazy | persample)")),
                };
                a.mode_name = v;
            }
            "--subset" => a.subset = num(&v)?,
            "--test-subset" => a.test_subset = num(&v)?,
            "--batch" => a.batch = num(&v)?,
            "--hidden" => a.hidden = num(&v)?,
            "--r" => a.r = num(&v)?,
            "--c" => a.c = num(&v)?,
            "--params" => a.params = v,
            "--seed" => a.seed = v.parse().map_err(|e| format!("--seed {v}: {e}"))?,
            "--lr" => a.lr = real(&v)?,
            "--momentum" => a.momentum = real(&v)?,
            "--data" => a.data = PathBuf::from(v),
            "--out" => a.out = Some(PathBuf::from(v)),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if a.mode_name.is_empty() {
        return Err("--mode lazy|persample is required".into());
    }
    if a.subset == 0 || a.batch == 0 || a.hidden == 0 {
        return Err("--subset, --batch and --hidden must be positive".into());
    }
    Ok(a)
}

fn params_named(name: &str) -> Result<CkksParams, String> {
    match name {
        "n16_train" => Ok(CkksParams::n16_train()),
        // Insecure smoke-test parameters (N = 2^12, 11 primes), as in `mnist`.
        "small" => {
            let mut q = vec![58];
            q.extend([42; 10]);
            CkksParams::new(12, &q, &[60], 42, 11).map_err(|e| e.to_string())
        }
        _ => Err(format!("unknown --params {name} (n16_train | small)")),
    }
}

fn rng(seed: u8, role: u8) -> Rng {
    let mut s = [seed; 32];
    s[0] = role;
    Rng::from_seed(s)
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

/// Weights `d_in x d_out` drawn exactly as `mnist`'s `init_net` (its `(d_out, d_in)` matrix
/// transposed; the biases are drawn and discarded to keep the stream identical).
fn init_net(dims: &[usize], r: &mut Rng) -> Vec<Array2<f64>> {
    dims.windows(2)
        .map(|p| {
            let bound = 1.0 / (p[0] as f64).sqrt();
            let w = Array2::from_shape_fn((p[1], p[0]), |_| bound * uni(r));
            for _ in 0..p[1] {
                uni(r);
            }
            w.t().to_owned()
        })
        .collect()
}

/// Plaintext forward, `W` as `d_in x d_out`, `z^2 + z` on the hidden layer, no biases.
fn forward_plain(net: &[Array2<f64>], x: &Array2<f64>) -> Array2<f64> {
    let z = x.dot(&net[0]).mapv(|v| v * v + v);
    z.dot(&net[1])
}

/// One plaintext NAG step with the library update rule used here.
fn nag_step(w: &mut Array2<f64>, vel: &mut Array2<f64>, g: &Array2<f64>, lr: f64, mu: f64) {
    *vel = mu * &*vel + g;
    *w = &*w - &(lr * &(g + &(mu * &*vel)));
}

fn twin_step(
    net: &mut [Array2<f64>],
    vel: &mut [Array2<f64>],
    x: &Array2<f64>,
    y: &Array2<f64>,
    lr: f64,
    mu: f64,
) {
    let n = x.nrows() as f64;
    let z1 = x.dot(&net[0]);
    let a1 = z1.mapv(|v| v * v + v);
    let delta = a1.dot(&net[1]) - y;
    let dz1 = delta.dot(&net[1].t()) * z1.mapv(|v| 2.0 * v + 1.0);
    let g2 = a1.t().dot(&delta) / n;
    let g1 = x.t().dot(&dz1) / n;
    nag_step(&mut net[0], &mut vel[0], &g1, lr, mu);
    nag_step(&mut net[1], &mut vel[1], &g2, lr, mu);
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

fn max_abs(a: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter().map(f64::abs).fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------------------------
// JSON output (minimal, no dependency).
// ---------------------------------------------------------------------------------------------

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
                out.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    x.write(out, ind);
                }
                out.push(']');
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

fn loadavg() -> String {
    Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn counter_json(c: &OpCounter) -> J {
    obj(vec![
        ("rotations", J::I(c.rotations as i64)),
        ("relins", J::I(c.relins as i64)),
        ("key_switches", J::I(c.key_switches() as i64)),
        ("ct_ct_tensor", J::I(c.ct_ct_tensor as i64)),
        ("ct_pt_mults", J::I(c.ct_pt_mults as i64)),
        ("rescales", J::I(c.rescales as i64)),
        ("bootstraps", J::I(c.bootstraps as i64)),
    ])
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("mnist_reboot_layout: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&args) {
        eprintln!("mnist_reboot_layout: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    let e = |x: hekit_core::error::HekitError| x.to_string();
    let load_start = loadavg();
    let t_all = Instant::now();
    let hint = |x: std::io::Error| format!("{x} (run scripts/fetch_mnist.sh)");
    let train = load_mnist(&args.data, Split::Train, Some(args.subset)).map_err(hint)?;
    let test = load_mnist(&args.data, Split::Test, Some(args.test_subset)).map_err(hint)?;
    let params = Arc::new(params_named(&args.params)?);
    let dims = [train.features(), args.hidden, CLASSES];
    let lay = RebootLayout::new(args.r, args.c, params.slots()).map_err(e)?;
    let steps = lay.galois_steps();
    let limbs = params.q().len() + params.p().len();
    let key_bytes = params.dnum() * 2 * limbs * params.n() * 8;
    eprintln!(
        "mnist_reboot_layout: {} N = 2^{} ({} slots), dims {dims:?}, (r, c) = ({}, {}), mode {}, \
         {} Galois keys + relin ({:.1} GB)",
        args.params,
        params.log_n(),
        params.slots(),
        args.r,
        args.c,
        args.mode_name,
        steps.len(),
        ((steps.len() + 1) * key_bytes) as f64 / 1e9
    );

    let t_keys = Instant::now();
    let mut kg =
        KeyGenerator::new(Arc::clone(&params), rng(args.seed, 1)).map_err(|x| x.to_string())?;
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
    eprintln!("mnist_reboot_layout: keys in {keygen_s:.1} s");

    let mut r = rng(args.seed, 2);
    let init = init_net(&dims, &mut r);
    let mut net = init.clone();
    let mut vel: Vec<Array2<f64>> = net.iter().map(|w| Array2::zeros(w.dim())).collect();
    let mut twin = init.clone();
    let mut twin_vel = vel.clone();
    let y_train = one_hot(&train.labels, CLASSES);
    let act = PolyRelu2;
    let top = params.max_level();
    let delta = params.scale();
    let encrypt = |v: &[f64], r: &mut Rng| -> Result<Ciphertext, String> {
        let pt = encoder.encode(v, delta, top).map_err(|x| x.to_string())?;
        Ok(enc.encrypt(&pt, r))
    };
    let decrypt = |ct: &Ciphertext| encoder.decode(&dec.decrypt(ct));

    let n = train.len();
    let mut batch_server_s = Vec::new();
    let mut batch_client_s = Vec::new();
    let mut per_batch: Vec<OpCounter> = Vec::new();
    let mut total = OpCounter::default();
    let mut samples = 0usize;
    let t_train = Instant::now();
    for start in (0..n).step_by(args.batch) {
        let end = (start + args.batch).min(n);
        let nb = end - start;
        // Client: encrypt the weights (fresh, top level) and the inputs.
        let tc = Instant::now();
        let w1e = encrypt(&lay.pack_row(&net[0]).map_err(e)?, &mut r)?;
        let w2e = encrypt(&lay.pack_col(&net[1]).map_err(e)?, &mut r)?;
        let mut xs = Vec::with_capacity(nb);
        for i in start..end {
            let x: Vec<f64> = train.images.row(i).to_vec();
            xs.push(encrypt(&lay.pack_expanded(&x).map_err(e)?, &mut r)?);
        }
        let mut client_s = tc.elapsed().as_secs_f64();

        // Server.
        ev.reset_counter();
        let ts = Instant::now();
        let mut g_cts = Vec::with_capacity(nb);
        let mut a1_cts = Vec::with_capacity(nb);
        let mut dz1_cts = Vec::with_capacity(nb);
        for (k, xe) in xs.iter().enumerate() {
            let z1 = lay.vecmat(&ev, xe, &w1e).map_err(e)?;
            let a1 = act.forward(&ev, &encoder, &z1).map_err(e)?;
            let y = lay.vecmat_col(&ev, &encoder, &a1, &w2e).map_err(e)?;
            let neg_t: Vec<f64> = y_train.row(start + k).iter().map(|v| -v).collect();
            let pt = encoder
                .encode(&lay.pack_expanded(&neg_t).map_err(e)?, y.scale(), y.level())
                .map_err(e)?;
            let g = ev.add_plain(&y, &pt).map_err(e)?;
            let da1 = lay.vecmat(&ev, &g, &w2e).map_err(e)?;
            let dz1 = act.backward(&ev, &encoder, &z1, &da1).map_err(e)?;
            g_cts.push(g);
            a1_cts.push(a1);
            dz1_cts.push(dz1);
        }
        let p1: Vec<(&Ciphertext, &Ciphertext)> = xs.iter().zip(&dz1_cts).collect();
        let p2: Vec<(&Ciphertext, &Ciphertext)> = g_cts.iter().zip(&a1_cts).collect();
        let dw1 = lay.outer_sum(&ev, &p1, args.mode).map_err(e)?;
        let dw2 = lay.outer_sum(&ev, &p2, args.mode).map_err(e)?;
        let server_s = ts.elapsed().as_secs_f64();
        let c = ev.counter();

        // Client: decrypt the gradients and update the weights (NAG, plaintext).
        let tc = Instant::now();
        let nf = nb as f64;
        let g1 = lay.unpack_row(&decrypt(&dw1), dims[0], dims[1]) / nf;
        let g2 = lay.unpack_col(&decrypt(&dw2), dims[1], dims[2]) / nf;
        nag_step(&mut net[0], &mut vel[0], &g1, args.lr, args.momentum);
        nag_step(&mut net[1], &mut vel[1], &g2, args.lr, args.momentum);
        client_s += tc.elapsed().as_secs_f64();
        let x = train.images.slice(s![start..end, ..]).to_owned();
        let y = y_train.slice(s![start..end, ..]).to_owned();
        twin_step(&mut twin, &mut twin_vel, &x, &y, args.lr, args.momentum);

        total.rotations += c.rotations;
        total.relins += c.relins;
        total.ct_ct_tensor += c.ct_ct_tensor;
        total.ct_pt_mults += c.ct_pt_mults;
        total.rescales += c.rescales;
        total.bootstraps += c.bootstraps;
        samples += nb;
        batch_server_s.push(server_s);
        batch_client_s.push(client_s);
        eprintln!(
            "mnist_reboot_layout: batch {} ({nb} samples): server {server_s:.1} s, client \
             {client_s:.1} s; {} rot + {} relin; elapsed {:.0} s, load {}",
            per_batch.len(),
            c.rotations,
            c.relins,
            t_all.elapsed().as_secs_f64(),
            loadavg()
        );
        per_batch.push(c);
    }
    let train_s = t_train.elapsed().as_secs_f64();

    let w_diff: Vec<f64> = net
        .iter()
        .zip(&twin)
        .map(|(g, t)| max_abs(g.iter().zip(t).map(|(a, b)| a - b)))
        .collect();
    let w_moved: Vec<f64> = twin
        .iter()
        .zip(&init)
        .map(|(t, i)| max_abs(t.iter().zip(i).map(|(a, b)| a - b)))
        .collect();
    let max_diff = w_diff.iter().copied().fold(0.0, f64::max);
    let out_he = forward_plain(&net, &test.images);
    let out_twin = forward_plain(&twin, &test.images);
    let out_init = forward_plain(&init, &test.images);
    let sf = samples as f64;
    let batches = per_batch.len();
    let server_total: f64 = batch_server_s.iter().sum();
    let client_total: f64 = batch_client_s.iter().sum();
    let constant = per_batch
        .iter()
        .all(|c| (c.rotations, c.relins) == (per_batch[0].rotations, per_batch[0].relins));
    eprintln!(
        "mnist_reboot_layout: {batches} batches, {samples} samples in {train_s:.1} s (server \
         {server_total:.1} s); {} KS ({:.3}/sample); max |W_he - W_twin| = {max_diff:e}",
        total.key_switches(),
        total.key_switches() as f64 / sf
    );
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let result = obj(vec![
        ("mode", J::S(args.mode_name.clone())),
        (
            "network",
            J::S(format!(
                "{} -> {} (z^2 + z) -> {} (linear), MSE, NAG, no biases",
                dims[0], dims[1], dims[2]
            )),
        ),
        (
            "setup",
            obj(vec![
                ("params", J::S(args.params.clone())),
                ("log_n", J::I(params.log_n() as i64)),
                ("slots", J::I(params.slots() as i64)),
                ("max_level", J::I(top as i64)),
                ("r", J::I(args.r as i64)),
                ("c", J::I(args.c as i64)),
                ("ciphertexts_per_sample", J::I(1)),
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
            ]),
        ),
        (
            "training",
            obj(vec![
                ("epochs", J::I(1)),
                ("subset", J::I(n as i64)),
                ("batch", J::I(args.batch as i64)),
                ("batches", J::I(batches as i64)),
                ("samples", J::I(samples as i64)),
                ("lr", J::N(args.lr)),
                ("momentum", J::N(args.momentum)),
                ("seed", J::I(args.seed as i64)),
            ]),
        ),
        (
            "measured",
            obj(vec![
                ("total", counter_json(&total)),
                ("per_batch_first", counter_json(&per_batch[0])),
                ("every_batch_equal", J::B(constant)),
                ("rotations_per_sample", J::N(total.rotations as f64 / sf)),
                ("relins_per_sample", J::N(total.relins as f64 / sf)),
                (
                    "key_switches_per_sample",
                    J::N(total.key_switches() as f64 / sf),
                ),
            ]),
        ),
        (
            "time_s",
            obj(vec![
                ("keygen", J::N(keygen_s)),
                ("epoch_total_incl_client", J::N(train_s)),
                ("server_total", J::N(server_total)),
                ("client_total", J::N(client_total)),
                ("server_per_batch_mean", J::N(mean(&batch_server_s))),
                ("server_per_sample_mean", J::N(server_total / sf)),
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
            ]),
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
    let mut text = String::new();
    result.write(&mut text, 0);
    text.push('\n');
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("results/mnist_reboot_{}.json", args.mode_name)));
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|x| x.to_string())?;
    }
    std::fs::write(&out, &text).map_err(|x| format!("{}: {x}", out.display()))?;
    eprintln!("mnist_reboot_layout: wrote {}", out.display());
    Ok(())
}
