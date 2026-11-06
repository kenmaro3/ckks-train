//! Task 32: the identity activation (ReBoot's linear output layer, `"none"`) and the trainer's
//! constant-derivative path (`ckks_train_nn::activation::Identity`, `ckks_train_nn::trainer`).
//!
//! - `Identity` forwards `z` unchanged (depth 0), has the constant derivative 1, and its
//!   `backward` is the upstream gradient itself (no level, no op).
//! - A PolyRelu2 → Identity MLP (4 → 8 → 3, NAG, seal_2020) trained over a full and a partial
//!   batch matches an independent plaintext NAG within 2^-10. The identity layer's
//!   `dz = delta * valid` costs no relinearization: the step's relins are exactly the
//!   forward / backward giants, the gradient diagonals and the one hidden-layer `dz` and
//!   activation products.

use std::sync::Arc;

use ckks_train_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::activation::{Activation, Identity, PolyRelu2};
use ckks_train_nn::dense::TransposeKeys;
use ckks_train_nn::layout::{LayerShape, Layout, unpack_activations};
use ckks_train_nn::optim::Nag;
use ckks_train_nn::refresh::ClientRefresher;
use ckks_train_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_key_pool, mlp_rotation_steps};
use ndarray::{Array2, Axis};

const TOL: f64 = 1.0 / 1024.0;
const DIMS: [usize; 3] = [4, 8, 3];

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn rand_mat(r: &mut Rng, rows: usize, cols: usize, amp: f64) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| amp * uni(r))
}

fn max_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn identity_is_depth_zero_with_unit_derivative() {
    let params = Arc::new(CkksParams::seal_2020());
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(1)).unwrap();
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
    let encoder = Encoder::new(Arc::clone(&params));
    let ev = Evaluator::new(Arc::clone(&params), Some(kg.relin_key()), None);
    let mut r = rng(2);
    let v: Vec<f64> = (0..params.slots()).map(|_| uni(&mut r)).collect();
    let z = enc.encrypt(
        &encoder
            .encode(&v, params.scale(), params.max_level())
            .unwrap(),
        &mut r,
    );
    let id = Identity;
    assert_eq!(id.depth(), 0);
    assert_eq!(id.derivative_depth(), 0);
    assert_eq!(id.constant_derivative(), Some(1.0));
    assert_eq!(PolyRelu2.constant_derivative(), None);
    assert_eq!(id.eval_plain(-0.75), -0.75);
    let before = ev.counter();
    let a = id.forward(&ev, &encoder, &z).unwrap();
    let u = id.backward(&ev, &encoder, &z, &a).unwrap();
    assert_eq!(
        ev.counter(),
        before,
        "identity forward / backward do no work"
    );
    for ct in [&a, &u] {
        assert_eq!(ct.level(), z.level());
        assert_eq!(ct.scale(), z.scale());
        assert_eq!(
            encoder.decode(&dec.decrypt(ct)),
            encoder.decode(&dec.decrypt(&z))
        );
    }
    assert!(
        id.derivative_factor(&ev, &encoder, &z, 1.0, None).is_err(),
        "a constant derivative is not a ciphertext"
    );
}

struct Layer {
    w: Array2<f64>,
    b: Vec<f64>,
}

/// Plaintext NAG of a PolyRelu2 → Identity MLP: MSE gradient `a - y`, `G = dz^T a_in / n`,
/// `V' = mu V + G`, `W' = W - lr (G + mu V')`, biases plain SGD.
fn nag_reference(
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
        let mut z = acts.last().unwrap().dot(&l.w.t());
        for mut row in z.rows_mut() {
            row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
        }
        acts.push(if i == last {
            z.clone()
        } else {
            z.mapv(|v| v * v + v)
        });
        zs.push(z);
    }
    let mut delta = acts.last().unwrap() - y;
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
        for (b, gbj) in net[i].b.iter_mut().zip(&gb) {
            *b -= lr * gbj;
        }
        delta = next;
    }
}

#[test]
fn poly_relu_identity_mlp_trains_like_plaintext_nag() {
    let (lr, mu) = (0.05, 0.9);
    let params = Arc::new(CkksParams::seal_2020());
    let layout = Layout::new(params.slots(), 8, 8).unwrap();
    let keys = TransposeKeys::Reuse(mlp_key_pool(&DIMS, &layout).unwrap());
    let steps = mlp_rotation_steps(&DIMS, &layout, &keys).unwrap();
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(40)).unwrap();
    let enc = Encryptor::new(Arc::clone(&params), kg.public_key());
    let dec = Decryptor::new(Arc::clone(&params), kg.secret_key());
    let encoder = Encoder::new(Arc::clone(&params));
    let ev = Evaluator::new(
        Arc::clone(&params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    let mut r = rng(41);
    let mut want: Vec<Layer> = DIMS
        .windows(2)
        .map(|p| Layer {
            w: rand_mat(&mut r, p[1], p[0], 0.8 / (p[0] as f64).sqrt()),
            b: (0..p[1]).map(|_| 0.2 * uni(&mut r)).collect(),
        })
        .collect();
    let plain: Vec<(Array2<f64>, Vec<f64>)> =
        want.iter().map(|l| (l.w.clone(), l.b.clone())).collect();
    let acts: Vec<Box<dyn Activation>> = vec![Box::new(PolyRelu2), Box::new(Identity)];
    let mut mlp = Mlp::encrypt(&ev, &plain, acts, &layout, &enc, &encoder, &mut r).unwrap();
    let spec = mlp.batch_spec(&ev, &encoder).unwrap();
    assert_eq!(
        spec.y_level,
        params.max_level() - 3,
        "identity output adds no level"
    );
    let client = ClientRefresher::new(
        dec.clone(),
        enc.clone(),
        Encoder::new(Arc::clone(&params)),
        rng(42),
    );
    let optims: Vec<Box<dyn LayerOptimizer>> =
        vec![Box::new(Nag::new(lr, mu)), Box::new(Nag::new(lr, mu))];
    let mut trainer = Trainer::new(&ev, Box::new(client), optims, layout)
        .unwrap()
        .with_transpose_keys(keys);

    // Relins of one step: forward giants + the PolyRelu2 square, the W2^T giants, the one
    // hidden dz product, and m relins per gradient. The identity dz adds none.
    let sh1 = LayerShape::new(DIMS[1], DIMS[0], &layout);
    let sh2 = LayerShape::new(DIMS[2], DIMS[1], &layout);
    let sh2t = sh2.transposed();
    let relins = (sh1.giant + 1) + sh2.giant + sh2t.giant + 1 + sh1.m + sh2.m;

    let mut vel: Vec<Array2<f64>> = want.iter().map(|l| Array2::zeros(l.w.dim())).collect();
    for n_valid in [8, 5] {
        let x = rand_mat(&mut r, n_valid, DIMS[0], 1.0);
        let y = rand_mat(&mut r, n_valid, DIMS[2], 1.0);
        let (xct, yct) = spec
            .encrypt(&enc, &encoder, &layout, &x, &y, &mut r)
            .unwrap();
        let before = ev.counter();
        trainer
            .train_batch_valid(&mut mlp, &xct, &yct, n_valid)
            .unwrap();
        let after = ev.counter();
        assert_eq!(
            (after.relins - before.relins) as usize,
            relins,
            "n_valid {n_valid}: identity dz is relin-free"
        );
        nag_reference(&mut want, &mut vel, &x, &y, lr, mu);
        for ((d, _), l) in mlp.layers.iter().zip(&want) {
            let w = d.w.decrypt(&dec, &encoder, &layout);
            let b = unpack_activations(&encoder.decode(&dec.decrypt(&d.b)), &layout, d.w.d_out());
            let ew = max_diff(&w, &l.w);
            let eb = b
                .row(0)
                .iter()
                .zip(&l.b)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max);
            eprintln!(
                "n_valid {n_valid}: W err 2^{:.2}, b err 2^{:.2}",
                ew.log2(),
                eb.log2()
            );
            assert!(ew <= TOL && eb <= TOL, "{ew:e} {eb:e}");
        }
    }
    let labels: Vec<&str> = trainer.trace().iter().map(|(s, _)| s.as_str()).collect();
    assert!(
        !labels.contains(&"s2"),
        "no derivative factor for the identity layer"
    );
    assert!(labels.contains(&"s1"));
}
