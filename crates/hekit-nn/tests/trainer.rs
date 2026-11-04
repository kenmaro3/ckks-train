//! Task 30: the trainer in interactive mode (`hekit_nn::trainer`, `hekit_nn::refresh`).
//!
//! - One `train_batch` of the Task 30 network (4 -> 8 -> 2, Square, batch 8, seal_2020) gives
//!   the plaintext training step of `hekit_nn::sim` within 2^-10; every intermediate level
//!   equals `max_level - depth` of the simulator's level table (W1' on level 0: the step uses
//!   the 9 levels of seal_2020 without an error), and after the refresh everything is back on
//!   the top level at its exact scale.
//! - An epoch of two more batches (one with 5 valid samples of 8) keeps matching the
//!   simulator applied in sequence; the epoch counter equals the simulator's counts.
//! - NAG over two batches (zero-new-key transpose) matches an independent plaintext NAG: the
//!   refreshed velocity is carried to the second step.
//! - `predict_plain` is the plaintext forward of the decrypted weights.
//!
//! One shared `Arc<CkksParams>` per context, Galois keys for exactly the steps the network
//! needs, fixed ChaCha20 seeds.

use std::sync::Arc;
use std::time::Instant;

use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::activation::{Activation, Poly, PolyRelu2, Square};
use hekit_nn::dense::{
    TransposeKeys, backward_rotation_steps, forward_rotation_steps, grad_rotation_steps,
    transpose_rotation_steps,
};
use hekit_nn::layout::{Layout, unpack_activations};
use hekit_nn::optim::{Nag, Sgd};
use hekit_nn::refresh::{ClientRefresher, Refresher};
use hekit_nn::sim::{SimLayer, sim_train_step_sgd};
use hekit_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_key_pool, mlp_rotation_steps};
use ndarray::Array2;

/// The brief's gate: every weight within 2^-10 of the plaintext step.
const TOL: f64 = 1.0 / 1024.0;

const DIMS: [usize; 3] = [4, 8, 2];

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

/// Uniform in `[-1, 1)`.
fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn rand_mat(r: &mut Rng, rows: usize, cols: usize, amp: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| amp * uni(r))
}

fn mat_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn vec_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    layout: Layout,
}

impl Ctx {
    fn client(&self, seed: u8) -> ClientRefresher {
        ClientRefresher::new(
            self.dec.clone(),
            self.enc.clone(),
            Encoder::new(Arc::clone(&self.params)),
            rng(seed),
        )
    }
}

/// seal_2020 (one shared `Arc`), batch-8 layout with `d = 8`, Galois keys for exactly `steps`.
fn ctx(steps: &[i32], seed: u8) -> Ctx {
    let params = Arc::new(CkksParams::seal_2020());
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(steps);
    assert_eq!(gks.len(), steps.len(), "one key per step");
    Ctx {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        ev: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
        layout: Layout::new(params.slots(), 8, 8).unwrap(),
        params,
    }
}

fn random_net(r: &mut Rng) -> Vec<SimLayer> {
    DIMS.windows(2)
        .map(|p| SimLayer {
            w: rand_mat(r, p[1], p[0], 0.8 / (p[0] as f64).sqrt()),
            b: (0..p[1]).map(|_| 0.2 * uni(r)).collect(),
        })
        .collect()
}

fn squares() -> Vec<Box<dyn Activation>> {
    vec![Box::new(Square), Box::new(Square)]
}

fn encrypt_mlp(c: &Ctx, net: &[SimLayer], r: &mut Rng) -> Mlp {
    let layers: Vec<(Array2<f64>, Vec<f64>)> =
        net.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
    Mlp::encrypt(&c.ev, &layers, squares(), &c.layout, &c.enc, &c.encoder, r).unwrap()
}

/// The decrypted layers of `mlp` (weights and lane-0 bias).
fn decrypt_layers(c: &Ctx, mlp: &Mlp) -> Vec<SimLayer> {
    mlp.layers
        .iter()
        .map(|(d, _)| SimLayer {
            w: d.w.decrypt(&c.dec, &c.encoder, &c.layout),
            b: unpack_activations(
                &c.encoder.decode(&c.dec.decrypt(&d.b)),
                &c.layout,
                d.w.d_out(),
            )
            .row(0)
            .to_vec(),
        })
        .collect()
}

/// Max weight and bias error of `got` against `want`, per layer.
fn layer_errors(got: &[SimLayer], want: &[SimLayer]) -> Vec<(f64, f64)> {
    got.iter()
        .zip(want)
        .map(|(g, w)| (mat_diff(&g.w, &w.w), vec_diff(&g.b, &w.b)))
        .collect()
}

fn assert_close(what: &str, got: &[SimLayer], want: &[SimLayer]) {
    for (i, (ew, eb)) in layer_errors(got, want).into_iter().enumerate() {
        eprintln!(
            "{what}: layer {} W error 2^{:.2}, b error 2^{:.2}",
            i + 1,
            ew.log2(),
            eb.log2()
        );
        assert!(
            ew <= TOL && eb <= TOL,
            "{what} layer {}: {ew:e} {eb:e}",
            i + 1
        );
    }
}

/// Every weight, bias (and the trainer's velocity, via the next step) on the top level at
/// its exact scale after the refresh.
fn assert_fresh(c: &Ctx, mlp: &Mlp, bias_scales: &[f64]) {
    let top = c.params.max_level();
    for ((d, _), &bs) in mlp.layers.iter().zip(bias_scales) {
        assert_eq!(d.w.level(), top, "weights refreshed to the top");
        assert_eq!(d.w.scale(), c.params.scale(), "weights keep scale Delta");
        assert_eq!(d.b.level(), top, "bias refreshed to the top");
        assert_eq!(d.b.scale(), bs, "bias keeps its exact scale");
    }
}

/// Plain forward of Square layers (for `predict_plain`).
fn plain_forward(net: &[SimLayer], x: &Array2<f64>) -> Array2<f64> {
    let mut a = x.clone();
    for l in net {
        let mut z = a.dot(&l.w.t());
        for mut row in z.rows_mut() {
            row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
        }
        a = z.mapv(|v| v * v);
    }
    a
}

fn sgd_optims(lr: f64) -> Vec<Box<dyn LayerOptimizer>> {
    vec![Box::new(Sgd { lr }), Box::new(Sgd { lr })]
}

fn add_counts(a: &OpCounter, b: &OpCounter) -> OpCounter {
    OpCounter {
        rotations: a.rotations + b.rotations,
        relins: a.relins + b.relins,
        ct_ct_tensor: a.ct_ct_tensor + b.ct_ct_tensor,
        ct_pt_mults: a.ct_pt_mults + b.ct_pt_mults,
        rescales: a.rescales + b.rescales,
        bootstraps: a.bootstraps + b.bootstraps,
    }
}

/// Task 30 Step 1: one `train_batch` (SGD) equals the simulator's plaintext step within
/// 2^-10, every level equals the simulator's level table (W1' on level 0, no error before
/// the refresh), and the refresh puts everything back on the top level at its exact scale.
/// Then an epoch of two batches (8 and 5 valid samples) keeps matching the simulator, with
/// the epoch counter equal to the simulator's counts.
#[test]
fn train_batch_matches_sim_and_fits_seal_2020() {
    let lr = 0.1;
    let keys = TransposeKeys::Dedicated;
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, 8).unwrap();
    let steps = mlp_rotation_steps(&DIMS, &layout, &keys).unwrap();
    eprintln!("dedicated: {} Galois keys {steps:?}", steps.len());
    let c = ctx(&steps, 30);
    let mut r = rng(130);
    let net = random_net(&mut r);
    let mut mlp = encrypt_mlp(&c, &net, &mut r);
    let bias_scales: Vec<f64> = mlp.layers.iter().map(|(d, _)| d.b.scale()).collect();
    assert_fresh(&c, &mlp, &bias_scales);

    // predict_plain on the fresh network is the plaintext forward of the initial weights.
    let xs = rand_mat(&mut r, 8, 4, 1.0);
    let pe = mat_diff(
        &mlp.predict_plain(&c.dec, &c.encoder, &c.layout, &xs),
        &plain_forward(&net, &xs),
    );
    eprintln!("predict_plain (fresh) error 2^{:.2}", pe.log2());
    assert!(pe <= TOL);

    let spec = mlp.batch_spec(&c.ev, &c.encoder).unwrap();
    assert_eq!(spec.x_level, c.params.max_level());
    assert_eq!(spec.x_scale, c.params.scale());
    let x = rand_mat(&mut r, 8, 4, 1.0);
    let y = rand_mat(&mut r, 8, 2, 1.0);
    let (xct, yct) = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x, &y, &mut r)
        .unwrap();
    assert_eq!((yct.level(), yct.scale()), (spec.y_level, spec.y_scale));

    let mut trainer = Trainer::new(&c.ev, Box::new(c.client(31)), sgd_optims(lr), c.layout)
        .unwrap()
        .with_transpose_keys(keys);
    let before = c.ev.counter();
    let t0 = Instant::now();
    trainer.train_batch(&mut mlp, &xct, &yct).unwrap();
    let secs = t0.elapsed().as_secs_f64();
    let step_count = {
        let after = c.ev.counter();
        OpCounter {
            rotations: after.rotations - before.rotations,
            relins: after.relins - before.relins,
            ct_ct_tensor: after.ct_ct_tensor - before.ct_ct_tensor,
            ct_pt_mults: after.ct_pt_mults - before.ct_pt_mults,
            rescales: after.rescales - before.rescales,
            bootstraps: after.bootstraps - before.bootstraps,
        }
    };
    eprintln!("train_batch (seal_2020, SGD, dedicated): {secs:.3} s, counter {step_count:?}");

    let sim = sim_train_step_sgd(&net, &x, &y, lr, &layout);
    assert_eq!(sim.depth, 9, "the step uses the 9 levels of seal_2020");
    assert_eq!(
        step_count,
        sim.count.to_op_counter(),
        "train_batch counter == sim"
    );
    // Every recorded level equals max_level - depth of the simulator's level table.
    let top = c.params.max_level();
    let trace = trainer.trace().to_vec();
    eprintln!("trace {trace:?}");
    assert!(trace.len() >= 14, "the trace records the whole step");
    for (label, level) in &trace {
        assert_eq!(*level, top - sim.depth_of(label), "level of {label}");
    }
    let w1 = trace.iter().find(|(l, _)| l == "W1'").unwrap().1;
    assert_eq!(w1, 0, "W1' on level 0: 9 levels, zero margin, no error");
    assert_fresh(&c, &mlp, &bias_scales);
    assert_eq!(
        trainer.refreshes(),
        4 + 2 + 2,
        "m1 + m2 weight entries, two biases"
    );
    assert_close(
        "one train_batch vs sim",
        &decrypt_layers(&c, &mlp),
        &sim.layers,
    );

    // An epoch of two more batches: one full, one with 5 valid samples (Task 31's batch 20 in
    // 32 lanes works the same way).
    let x2 = rand_mat(&mut r, 8, 4, 1.0);
    let y2 = rand_mat(&mut r, 8, 2, 1.0);
    let x3 = rand_mat(&mut r, 5, 4, 1.0);
    let y3 = rand_mat(&mut r, 5, 2, 1.0);
    let b2 = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x2, &y2, &mut r)
        .unwrap();
    let b3 = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x3, &y3, &mut r)
        .unwrap();
    let log = trainer
        .train_epoch_valid(&mut mlp, &[(b2.0, b2.1, 8), (b3.0, b3.1, 5)])
        .unwrap();
    let sim2 = sim_train_step_sgd(&sim.layers, &x2, &y2, lr, &layout);
    let sim3 = sim_train_step_sgd(&sim2.layers, &x3, &y3, lr, &layout);
    eprintln!("epoch log {log:?}");
    assert_eq!(log.epoch, 0);
    assert_eq!(log.batches, 2);
    assert_eq!(log.refreshes, 16);
    assert!(log.loss.is_none(), "the server cannot see the loss");
    assert!(log.seconds > 0.0 && log.refresh_seconds > 0.0 && log.refresh_seconds < log.seconds);
    assert_eq!(
        log.counter,
        add_counts(&sim2.count.to_op_counter(), &sim3.count.to_op_counter()),
        "epoch counter == sim counts"
    );
    assert_eq!(trainer.log.len(), 1);
    assert_eq!(trainer.log[0], log);
    assert_fresh(&c, &mlp, &bias_scales);
    let got = decrypt_layers(&c, &mlp);
    assert_close("three batches vs sim", &got, &sim3.layers);

    // predict_plain after training is the plaintext forward of the trained weights.
    let pe = mat_diff(
        &mlp.predict_plain(&c.dec, &c.encoder, &c.layout, &xs),
        &plain_forward(&sim3.layers, &xs),
    );
    eprintln!("predict_plain (trained) error 2^{:.2}", pe.log2());
    assert!(pe <= TOL);
}

/// Plaintext NAG reference (independent of the simulator): Square MLP, MSE gradient
/// `a - y`, `G = dz^T a_in / n`; weights `V' = mu V + G`, `W' = W - lr (G + mu V')`
/// (`V_0 = 0`), biases plain SGD `b' = b - lr sum(dz) / n`.
fn nag_reference(
    net: &mut [SimLayer],
    vel: &mut [Array2<f64>],
    x: &Array2<f64>,
    y: &Array2<f64>,
    lr: f64,
    mu: f64,
) {
    let n = x.nrows() as f64;
    let mut acts = vec![x.clone()];
    let mut zs = Vec::new();
    for l in net.iter() {
        let mut z = acts.last().unwrap().dot(&l.w.t());
        for mut row in z.rows_mut() {
            row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
        }
        acts.push(z.mapv(|v| v * v));
        zs.push(z);
    }
    let mut delta = acts.last().unwrap() - y;
    for i in (0..net.len()).rev() {
        let dz = &delta * &zs[i].mapv(|v| 2.0 * v);
        let next = dz.dot(&net[i].w);
        let g = dz.t().dot(&acts[i]) / n;
        vel[i] = mu * &vel[i] + &g;
        net[i].w = &net[i].w - &(lr * &(&g + &(mu * &vel[i])));
        let gb = dz.sum_axis(ndarray::Axis(0)) / n;
        for (b, gbj) in net[i].b.iter_mut().zip(&gb) {
            *b -= lr * gbj;
        }
        delta = next;
    }
}

/// NAG over two consecutive `train_batch` calls with the zero-new-key transpose: matches the
/// plaintext NAG. The second step needs the velocity refreshed with the weights (without
/// it, layer 1's velocity would sit on level 0, not above the gradient: `LevelMismatch`).
#[test]
fn nag_two_batches_carry_refreshed_velocity() {
    let (lr, mu) = (0.05, 0.9);
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, 8).unwrap();
    let pool = mlp_key_pool(&DIMS, &layout).unwrap();
    let keys = TransposeKeys::Reuse(pool.clone());
    let steps = mlp_rotation_steps(&DIMS, &layout, &keys).unwrap();
    let dedicated = mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).unwrap();
    eprintln!(
        "reuse: {} Galois keys {steps:?} (dedicated {})",
        steps.len(),
        dedicated.len()
    );
    assert_eq!(steps, pool, "the transpose needs no key beyond the pool");
    assert!(steps.len() < dedicated.len());
    let c = ctx(&steps, 32);
    let mut r = rng(132);
    let net = random_net(&mut r);
    let mut mlp = encrypt_mlp(&c, &net, &mut r);
    let bias_scales: Vec<f64> = mlp.layers.iter().map(|(d, _)| d.b.scale()).collect();
    let spec = mlp.batch_spec(&c.ev, &c.encoder).unwrap();
    let optims: Vec<Box<dyn LayerOptimizer>> =
        vec![Box::new(Nag::new(lr, mu)), Box::new(Nag::new(lr, mu))];
    let mut trainer = Trainer::new(&c.ev, Box::new(c.client(33)), optims, c.layout)
        .unwrap()
        .with_transpose_keys(keys);

    let mut want = net.clone();
    let mut vel: Vec<Array2<f64>> = net.iter().map(|l| Array2::zeros(l.w.dim())).collect();
    for k in 0..2 {
        let x = rand_mat(&mut r, 8, 4, 1.0);
        let y = rand_mat(&mut r, 8, 2, 1.0);
        let (xct, yct) = spec
            .encrypt(&c.enc, &c.encoder, &c.layout, &x, &y, &mut r)
            .unwrap();
        let t0 = Instant::now();
        trainer.train_batch(&mut mlp, &xct, &yct).unwrap();
        eprintln!(
            "nag train_batch {} (reuse keys): {:.3} s",
            k + 1,
            t0.elapsed().as_secs_f64()
        );
        nag_reference(&mut want, &mut vel, &x, &y, lr, mu);
        assert_fresh(&c, &mlp, &bias_scales);
        assert_close(
            &format!("nag step {}", k + 1),
            &decrypt_layers(&c, &mlp),
            &want,
        );
    }
    // Weights, biases and both velocities: (4 + 1 + 4) + (2 + 1 + 2) per step after the first.
    assert_eq!(trainer.refreshes(), 2 * 14);
}

/// `ClientRefresher` restores the top level at the ciphertext's exact scale and keeps the
/// values (any slot layout: decrypt, decode, re-encode, encrypt).
#[test]
fn client_refresher_restores_level_and_scale() {
    let c = ctx(&[], 34);
    let mut client = c.client(35);
    let mut r = rng(134);
    let v: Vec<f64> = (0..c.params.slots()).map(|_| uni(&mut r)).collect();
    let scale = c.params.scale() * 1.375;
    let ct = c
        .enc
        .encrypt(&c.encoder.encode(&v, scale, 2).unwrap(), &mut r);
    let fresh = client.refresh(&ct).unwrap();
    assert_eq!(fresh.level(), c.params.max_level());
    assert_eq!(fresh.scale(), scale, "exactly the old scale");
    let e = vec_diff(&c.encoder.decode(&c.dec.decrypt(&fresh)), &v);
    eprintln!("refresh error 2^{:.2}", e.log2());
    assert!(e < 2f64.powi(-20));
    // At a chosen scale.
    let other = client.refresh_to_scale(&ct, c.params.scale()).unwrap();
    assert_eq!(other.scale(), c.params.scale());
    assert!(vec_diff(&c.encoder.decode(&c.dec.decrypt(&other)), &v) < 2f64.powi(-20));
    assert_eq!(client.refreshed(), 2);
    // A ciphertext of other parameters is an error, not a panic.
    let p2 = Arc::new(CkksParams::seal_2020());
    let mut kg2 = KeyGenerator::new(Arc::clone(&p2), rng(36)).unwrap();
    let enc2 = Encryptor::new(Arc::clone(&p2), kg2.public_key());
    let foreign = enc2.encrypt(
        &Encoder::new(Arc::clone(&p2))
            .encode(&v, p2.scale(), 1)
            .unwrap(),
        &mut r,
    );
    assert!(matches!(
        client.refresh(&foreign).unwrap_err(),
        HekitError::InvalidParams(_)
    ));
    assert_eq!(client.refreshed(), 2);
}

/// Input errors are reported before the network changes.
#[test]
fn train_batch_errors_leave_the_network_unchanged() {
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, 8).unwrap();
    let steps = mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).unwrap();
    let c = ctx(&steps, 37);
    let mut r = rng(137);
    let net = random_net(&mut r);
    let mut mlp = encrypt_mlp(&c, &net, &mut r);
    let spec = mlp.batch_spec(&c.ev, &c.encoder).unwrap();
    let x = rand_mat(&mut r, 8, 4, 1.0);
    let y = rand_mat(&mut r, 8, 2, 1.0);
    let (xct, yct) = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x, &y, &mut r)
        .unwrap();
    let is_invalid = |e: HekitError| matches!(e, HekitError::InvalidParams(_));

    // One optimizer for two layers.
    let mut t1 = Trainer::new(
        &c.ev,
        Box::new(c.client(38)),
        vec![Box::new(Sgd { lr: 0.1 }) as Box<dyn LayerOptimizer>],
        c.layout,
    )
    .unwrap();
    let before = c.ev.counter();
    assert!(is_invalid(
        t1.train_batch(&mut mlp, &xct, &yct).unwrap_err()
    ));
    assert_eq!(c.ev.counter(), before, "nothing done");
    // No valid sample, more valid samples than lanes.
    let mut t = Trainer::new(&c.ev, Box::new(c.client(39)), sgd_optims(0.1), c.layout).unwrap();
    for n in [0, 9] {
        assert!(is_invalid(
            t.train_batch_valid(&mut mlp, &xct, &yct, n).unwrap_err()
        ));
    }
    assert_eq!(c.ev.counter(), before, "nothing done");
    // y not at the output's exact scale: the forward runs, then Mse refuses; the network is
    // untouched (still fresh, same weights).
    let bad_y = c.enc.encrypt(
        &c.encoder
            .encode(&[0.0; 16], spec.y_scale * (1.0 + 1e-9), spec.y_level)
            .unwrap(),
        &mut r,
    );
    assert_eq!(
        t.train_batch(&mut mlp, &xct, &bad_y).unwrap_err(),
        HekitError::ScaleMismatch
    );
    let bias_scales: Vec<f64> = mlp.layers.iter().map(|(d, _)| d.b.scale()).collect();
    assert_fresh(&c, &mlp, &bias_scales);
    assert_close("after errors", &decrypt_layers(&c, &mlp), &net);
    assert!(t.log.is_empty());
    assert_eq!(t.refreshes(), 0);
    // A layout of other slots.
    let small = Layout::new(1 << 10, 8, 8).unwrap();
    assert!(is_invalid(
        Trainer::new(&c.ev, Box::new(c.client(40)), sgd_optims(0.1), small)
            .err()
            .unwrap()
    ));
}

/// The key set: the union of the forward, backward (layers above the first), gradient and
/// transpose steps; the reuse pool is the same set without the transpose-only keys.
#[test]
fn mlp_rotation_steps_are_the_union_of_the_layer_steps() {
    let layout = Layout::new(1 << 14, 8, 8).unwrap();
    let mut want: Vec<i32> = [
        forward_rotation_steps(8, 4, &layout),
        forward_rotation_steps(2, 8, &layout),
        backward_rotation_steps(2, 8, &layout),
        grad_rotation_steps(8, 4, &layout),
        grad_rotation_steps(2, 8, &layout),
    ]
    .concat();
    want.sort_unstable();
    want.dedup();
    let pool = mlp_key_pool(&DIMS, &layout).unwrap();
    assert_eq!(pool, want);
    let tr = transpose_rotation_steps(2, 8, &layout, &TransposeKeys::Dedicated).unwrap();
    let mut ded = [want.clone(), tr].concat();
    ded.sort_unstable();
    ded.dedup();
    assert_eq!(
        mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).unwrap(),
        ded
    );
    assert_eq!(
        mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Reuse(pool.clone())).unwrap(),
        pool
    );
    // docs/layout.md §2.3: 11 keys for the full step, 1 of them used only by the transpose.
    assert_eq!(ded.len(), 11);
    assert_eq!(pool.len(), 10);
    // Bad shapes are errors.
    assert!(mlp_rotation_steps(&[4], &layout, &TransposeKeys::Dedicated).is_err());
    assert!(mlp_rotation_steps(&[4, 16, 2], &layout, &TransposeKeys::Dedicated).is_err());
}

/// `Activation::eval_plain` (used by `predict_plain`).
#[test]
fn eval_plain_is_the_polynomial() {
    assert_eq!(Square.eval_plain(-1.5), 2.25);
    assert_eq!(PolyRelu2.eval_plain(-1.5), 0.75);
    let p = Poly::new(vec![0.5, -1.0, 0.25, 0.0, 2.0]).unwrap();
    let z: f64 = 0.75;
    let want = 0.5 - z + 0.25 * z * z + 2.0 * z.powi(4);
    assert!((p.eval_plain(z) - want).abs() < 1e-15);
}
