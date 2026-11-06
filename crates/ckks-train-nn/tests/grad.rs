//! Task 28: encrypted weight / bias gradients and the SGD / NAG updates (`ckks_train_nn::dense`,
//! `ckks_train_nn::optim`).
//!
//! - `weight_grad` is compared slot for slot with the packed plaintext `delta^T A / batch`
//!   (2^-14) and its counter with `sim_outer_batchsum` (exactly); the result has exactly the
//!   weights' scale, so `W - dW` is a plain `sub` with identical scales.
//! - `Sgd::step` gives `W - lr dW`; `Nag` two steps match the plaintext NAG.
//! - The whole Task 30 training step (4 -> 8 -> 2, Square, batch 8, seal_2020) is run
//!   encrypted with every constant folded: the level of every value is read from the
//!   ciphertexts and must equal `max_level - depth` of the simulator's level table, the whole
//!   counter must equal the simulator's, and the updated layers must match it.
//!
//! One shared `Arc<CkksParams>` per context, Galois keys for exactly the steps a test needs,
//! fixed ChaCha20 seeds.

use std::collections::BTreeSet;
use std::sync::Arc;

use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Diagonals, Encoder, Encryptor, Evaluator, KeyGenerator,
    OpCounter,
};
use ckks_train_core::error::HekitError;
use ckks_train_core::sampler::Rng;
use ckks_train_nn::dense::{
    Dense, EncWeights, TransposeKeys, backward_rotation_steps, bias_grad_rotation_steps,
    forward_rotation_steps, grad_rotation_steps, transpose_rotation_steps,
};
use ckks_train_nn::layout::{
    LayerShape, Layout, pack_activations, pack_bias, pack_weight_diagonals, unpack_activations,
};
use ckks_train_nn::optim::{Nag, Optimizer, Sgd};
use ckks_train_nn::sim::{
    SimCounter, SimLayer, sim_batchsum, sim_outer_batchsum, sim_train_step_sgd, valid_lane_mask,
};
use ndarray::Array2;

/// The brief's gate for `weight_grad`: every slot within 2^-14.
const TOL: f64 = 1.0 / 16384.0;

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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn mat_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn union(lists: &[Vec<i32>]) -> Vec<i32> {
    lists
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
}

impl Ctx {
    fn encrypt(&self, v: &[f64], scale: f64, level: usize, r: &mut Rng) -> Ciphertext {
        self.enc
            .encrypt(&self.encoder.encode(v, scale, level).unwrap(), r)
    }

    fn decrypt(&self, ct: &Ciphertext) -> Vec<f64> {
        self.encoder.decode(&self.dec.decrypt(ct))
    }

    fn decrypt_diags(&self, d: &Diagonals<Ciphertext>) -> Vec<Vec<f64>> {
        d.entries().iter().map(|c| self.decrypt(c)).collect()
    }
}

/// seal_2020 (one shared `Arc`), Galois keys for exactly `steps`.
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
        params,
    }
}

/// A random layer, encrypted at the top level for inputs of scale `Delta`.
fn random_layer(
    c: &Ctx,
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    r: &mut Rng,
) -> (Array2<f64>, Vec<f64>, Dense) {
    let w = rand_mat(r, d_out, d_in, 1.0);
    let b: Vec<f64> = (0..d_out).map(|_| uni(r)).collect();
    let layer = Dense::encrypt(&w, &b, layout, c.params.scale(), &c.enc, &c.encoder, r).unwrap();
    (w, b, layer)
}

/// `weight_grad` and `bias_grad` of a `d_out x d_in` layer: values, counter, level and scale.
fn check_grads(d_out: usize, d_in: usize, d: usize, seed: u8) {
    let batch = 8;
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), d, batch).unwrap();
    let sh = LayerShape::new(d_out, d_in, &layout);
    let steps = grad_rotation_steps(d_out, d_in, &layout);
    let c = ctx(&steps, seed);
    let mut r = rng(seed.wrapping_add(100));
    let (_, _, layer) = random_layer(&c, d_out, d_in, &layout, &mut r);
    assert_eq!(layer.grad_rotation_steps(&layout), steps);
    let delta = rand_mat(&mut r, batch, d_out, 1.0);
    let a = rand_mat(&mut r, batch, d_in, 1.0);
    let (dp, ap) = (
        pack_activations(&delta, &layout),
        pack_activations(&a, &layout),
    );
    let (scale, top) = (c.params.scale(), c.params.max_level());
    // delta one level below A (A is mod-switched down to delta's level).
    let dct = c.encrypt(&dp, scale, top - 1, &mut r);
    let act = c.encrypt(&ap, scale, top, &mut r);

    // Plaintext reference and simulator.
    let dw = delta.t().dot(&a) / batch as f64;
    let want = pack_weight_diagonals(&dw, &layout, (sh.baby, sh.giant));
    let mut sc = SimCounter::default();
    let sim = sim_outer_batchsum(&dp, &ap, d_out, d_in, &layout, 1.0 / batch as f64, &mut sc);
    for (s, w) in sim.iter().zip(&want) {
        assert!(max_abs_diff(s, w) < 1e-12, "simulator vs reference");
    }

    let ev = c.ev.clone();
    ev.reset_counter();
    let g = layer.weight_grad(&ev, &dct, &act, &layout).unwrap();
    let cnt = ev.counter();
    let what = format!("{d_out} x {d_in}, d = {d}, B = {batch}");
    assert_eq!(
        cnt,
        sc.to_op_counter(),
        "weight_grad counter == sim ({what})"
    );
    let lb = layout.log_batch();
    assert_eq!(
        cnt.rotations,
        ((sh.baby - 1) + (sh.giant - 1) + 2 * sh.m * lb) as u64,
        "(baby - 1) + (giant - 1) + 2 m log2 B"
    );
    assert_eq!(
        (cnt.ct_ct_tensor, cnt.relins, cnt.ct_pt_mults, cnt.rescales),
        (sh.m as u64, sh.m as u64, sh.m as u64, 2 * sh.m as u64)
    );
    assert_eq!((g.d(), g.stride()), (sh.m, batch));
    for e in g.entries() {
        assert_eq!(e.level(), top - 3, "two levels below delta");
        assert_eq!(e.scale(), layer.w.scale(), "exactly the weights' scale");
    }
    let mut err: f64 = 0.0;
    for (i, (got, w)) in c.decrypt_diags(&g).iter().zip(&want).enumerate() {
        let e = max_abs_diff(got, w);
        assert!(e <= TOL, "{what}: diagonal {i} error {e:e} > 2^-14");
        err = err.max(e);
    }
    // W - dW is a plain sub with identical scales.
    let wl = layer
        .w
        .diagonals()
        .mod_switch_to_level(&ev, top - 3)
        .unwrap();
    for (we, ge) in wl.entries().iter().zip(g.entries()) {
        assert_eq!(we.scale(), ge.scale());
        ev.sub(we, ge).unwrap();
    }

    // Bias gradient: steps 3-5 of the batch sum, at exactly the bias's scale.
    let mut sb = SimCounter::default();
    let simb = sim_batchsum(&dp, &layout, 1.0 / batch as f64, &mut sb);
    let sums: Vec<f64> = (0..d_out)
        .map(|j| delta.column(j).sum() / batch as f64)
        .collect();
    assert!(max_abs_diff(&simb, &pack_bias(&sums, &layout)) < 1e-12);
    ev.reset_counter();
    let gb = layer.bias_grad(&ev, &dct, &layout).unwrap();
    assert_eq!(ev.counter(), sb.to_op_counter(), "bias_grad counter == sim");
    assert_eq!(gb.level(), top - 2);
    assert_eq!(gb.scale(), layer.b.scale(), "exactly the bias's scale");
    let eb = max_abs_diff(&c.decrypt(&gb), &pack_bias(&sums, &layout));
    assert!(eb <= TOL, "bias grad error {eb:e}");
    eprintln!(
        "{what}: weight_grad 2^{:.2}, bias_grad 2^{:.2}, counter {cnt:?}",
        err.log2(),
        eb.log2()
    );
}

/// Iris W1: tall 8 x 4 (m = 4, split 2 x 2).
#[test]
fn grads_tall_8x4() {
    check_grads(8, 4, 8, 1);
}

/// Iris W2: wide 2 x 8 (m = 2, split 2 x 1).
#[test]
fn grads_wide_2x8() {
    check_grads(2, 8, 8, 2);
}

/// Square 16 x 16 (m = 16, split 4 x 4).
#[test]
fn grads_square_16x16() {
    check_grads(16, 16, 16, 3);
}

/// The Galois steps: gradient babies `k2 B`, giants `-g k1 B`, and the batch reduce and
/// replicate `+-2^t`.
#[test]
fn grad_rotation_steps_are_babies_giants_and_lanes() {
    let layout = Layout::new(1 << 14, 16, 8).unwrap();
    assert_eq!(bias_grad_rotation_steps(&layout), vec![-4, -2, -1, 1, 2, 4]);
    assert_eq!(
        grad_rotation_steps(16, 16, &layout),
        vec![-96, -64, -32, -4, -2, -1, 1, 2, 4, 8, 16, 24]
    );
    assert_eq!(
        grad_rotation_steps(2, 8, &layout),
        vec![-4, -2, -1, 1, 2, 4, 8]
    );
    let one = Layout::new(1 << 14, 16, 1).unwrap();
    assert!(bias_grad_rotation_steps(&one).is_empty());
}

/// `Sgd::step` with the gradient carrying `lr / batch` (folded into the lane-0 mask) gives
/// `W - lr dW` at the gradient's level and the weights' scale, in the layout the next forward
/// uses (the forward is run on the updated weights).
#[test]
fn sgd_step_gives_w_minus_lr_dw() {
    let (d_out, d_in, batch) = (8, 4, 8);
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, batch).unwrap();
    let steps = union(&[
        grad_rotation_steps(d_out, d_in, &layout),
        forward_rotation_steps(d_out, d_in, &layout),
    ]);
    let c = ctx(&steps, 4);
    let mut r = rng(104);
    let (w, b, mut layer) = random_layer(&c, d_out, d_in, &layout, &mut r);
    let delta = rand_mat(&mut r, batch, d_out, 1.0);
    let a = rand_mat(&mut r, batch, d_in, 1.0);
    let (scale, top) = (c.params.scale(), c.params.max_level());
    let dct = c.encrypt(&pack_activations(&delta, &layout), scale, top - 2, &mut r);
    let act = c.encrypt(&pack_activations(&a, &layout), scale, top - 2, &mut r);
    let mut sgd = Sgd { lr: 0.1 };
    assert_eq!(sgd.grad_scale(), 0.1);
    let g = layer
        .weight_grad_scaled(&c.ev, &dct, &act, &layout, sgd.grad_scale() / batch as f64)
        .unwrap();
    let ev = c.ev.clone();
    ev.reset_counter();
    sgd.step(&ev, &mut layer.w, &g).unwrap();
    assert_eq!(
        ev.counter(),
        OpCounter::default(),
        "SGD update is subtractions only"
    );
    assert_eq!(layer.w.level(), top - 4, "the update costs no level");
    assert_eq!(layer.w.scale(), scale, "the weights keep their scale");
    let want = &w - &(0.1 * &(delta.t().dot(&a) / batch as f64));
    let got = layer.w.decrypt(&c.dec, &c.encoder, &layout);
    let err = mat_diff(&got, &want);
    eprintln!("sgd: W - lr dW error 2^{:.2}", err.log2());
    assert!(err <= TOL, "{err:e}");
    // Every lane of every diagonal is the packed W' (the next forward needs no repacking).
    let sh = LayerShape::new(d_out, d_in, &layout);
    let packed = pack_weight_diagonals(&want, &layout, (sh.baby, sh.giant));
    for (got, w) in c.decrypt_diags(layer.w.diagonals()).iter().zip(&packed) {
        assert!(max_abs_diff(got, w) <= TOL);
    }
    // The next forward works on the updated weights.
    let x = rand_mat(&mut r, batch, d_in, 1.0);
    let xct = c.encrypt(&pack_activations(&x, &layout), scale, top - 4, &mut r);
    let z = layer.forward(&c.ev, &xct, &layout).unwrap();
    let mut zw = x.dot(&want.t());
    for mut row in zw.rows_mut() {
        row.iter_mut().zip(&b).for_each(|(v, bb)| *v += bb);
    }
    let ez = max_abs_diff(&c.decrypt(&z), &pack_activations(&zw, &layout));
    assert!(ez <= TOL, "forward after update {ez:e}");
}

/// Two NAG steps match the plaintext NAG `V' = mu V + g`, `W' = W - lr (g + mu V')`
/// (`g = dW`, `V_0 = 0`). The encrypted optimizer keeps the scaled velocity
/// `U = lr (1 + mu) V`; the gradient carries `lr (1 + mu) / batch` (folded into the mask), and
/// the multiplications by `mu` act only on `U`, which sits a level above the gradient: the
/// updated weights land at the gradient's level.
#[test]
fn nag_two_steps_match_plaintext() {
    let (d_out, d_in, batch) = (2, 8, 8);
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, batch).unwrap();
    let c = ctx(&grad_rotation_steps(d_out, d_in, &layout), 5);
    let mut r = rng(105);
    let (w0, _, mut layer) = random_layer(&c, d_out, d_in, &layout, &mut r);
    let (lr, mu) = (0.1, 0.9);
    let mut nag = Nag::new(lr, mu);
    assert!(nag.velocity().is_none());
    assert!((nag.grad_scale() - lr * (1.0 + mu)).abs() < 1e-15);
    let scale = c.params.scale();

    let (mut w_ref, mut v_ref) = (w0.clone(), Array2::<f64>::zeros((d_out, d_in)));
    // Step 1 gradient from level 7 (lands at 5); step 2 from level 4 (lands at 2, the
    // velocity at 5 is above it).
    for (k, level) in [7usize, 4].into_iter().enumerate() {
        let delta = rand_mat(&mut r, batch, d_out, 1.0);
        let a = rand_mat(&mut r, batch, d_in, 1.0);
        let dct = c.encrypt(&pack_activations(&delta, &layout), scale, level, &mut r);
        let act = c.encrypt(&pack_activations(&a, &layout), scale, level, &mut r);
        let h = layer
            .weight_grad_scaled(&c.ev, &dct, &act, &layout, nag.grad_scale() / batch as f64)
            .unwrap();
        let ev = c.ev.clone();
        ev.reset_counter();
        nag.step(&ev, &mut layer.w, &h).unwrap();
        let cnt = ev.counter();
        let m = LayerShape::new(d_out, d_in, &layout).m as u64;
        if k == 0 {
            assert_eq!(cnt, OpCounter::default(), "first step: U = h, W - h");
        } else {
            // mu U and (mu^2 / (1 + mu)) U per diagonal, each with its rescale.
            assert_eq!(
                (cnt.ct_pt_mults, cnt.rescales, cnt.rotations),
                (2 * m, 2 * m, 0)
            );
        }
        assert_eq!(
            layer.w.level(),
            level - 2,
            "the update costs no level on the path"
        );
        assert_eq!(layer.w.scale(), scale);
        let u = nag.velocity().unwrap();
        assert_eq!(u.entries()[0].level(), level - 2);
        // Plaintext NAG.
        let g = delta.t().dot(&a) / batch as f64;
        v_ref = mu * &v_ref + &g;
        w_ref = &w_ref - &(lr * &(&g + &(mu * &v_ref)));
        let got = layer.w.decrypt(&c.dec, &c.encoder, &layout);
        let err = mat_diff(&got, &w_ref);
        let gu = EncWeights::from_diagonals(u.clone(), d_out, d_in, &layout)
            .unwrap()
            .decrypt(&c.dec, &c.encoder, &layout);
        let eu = mat_diff(&gu, &(lr * (1.0 + mu) * &v_ref));
        eprintln!(
            "nag step {}: W error 2^{:.2}, U error 2^{:.2}",
            k + 1,
            err.log2(),
            eu.log2()
        );
        assert!(err <= TOL, "step {}: {err:e}", k + 1);
        assert!(eu <= TOL, "step {}: velocity {eu:e}", k + 1);
    }
    // A third step with a gradient at the velocity's level needs a refresh first.
    let delta = rand_mat(&mut r, batch, d_out, 1.0);
    let dct = c.encrypt(&pack_activations(&delta, &layout), scale, 4, &mut r);
    let a = rand_mat(&mut r, batch, d_in, 1.0);
    let act = c.encrypt(&pack_activations(&a, &layout), scale, 4, &mut r);
    let h = layer
        .weight_grad_scaled(&c.ev, &dct, &act, &layout, 0.01)
        .unwrap();
    let before = layer.w.clone();
    let ev = c.ev.clone();
    ev.reset_counter();
    assert_eq!(
        nag.step(&ev, &mut layer.w, &h).unwrap_err(),
        HekitError::LevelMismatch
    );
    assert_eq!(ev.counter(), OpCounter::default(), "nothing done");
    assert_eq!(layer.w.level(), before.level());
    // After a refresh of the velocity (client re-encrypts U at the top), it works.
    let u = nag.velocity().unwrap();
    let uw = EncWeights::from_diagonals(u.clone(), d_out, d_in, &layout)
        .unwrap()
        .decrypt(&c.dec, &c.encoder, &layout);
    let fresh = EncWeights::encrypt(&uw, &layout, &c.enc, &c.encoder, &mut r).unwrap();
    nag.set_velocity(Some(fresh.into_diagonals()));
    nag.step(&c.ev, &mut layer.w, &h).unwrap();
    assert_eq!(layer.w.level(), 2);
}

/// A rate change must not reinterpret the velocity already stored by NAG.
#[test]
fn nag_changing_rate_keeps_the_original_velocity_basis() {
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, 8).unwrap();
    let c = ctx(&grad_rotation_steps(2, 8, &layout), 71);
    let mut r = rng(171);
    let (mut want, _, mut layer) = random_layer(&c, 2, 8, &layout, &mut r);
    let mut v = Array2::<f64>::zeros((2, 8));
    let mut nag = Nag::new(0.04, 0.9);
    nag.set_lr_reference(0.08).unwrap();
    for (&lr, level) in [0.04, 0.00008, 0.02, 0.08].iter().zip([9, 7, 5, 3]) {
        if level == 5 {
            nag.lr = lr; // Valid legacy public field writes use the same fixed basis.
        } else {
            nag.set_lr(lr).unwrap();
        }
        let delta = rand_mat(&mut r, 8, 2, 1.0);
        let a = rand_mat(&mut r, 8, 8, 1.0);
        let dc = c.encrypt(
            &pack_activations(&delta, &layout),
            c.params.scale(),
            level,
            &mut r,
        );
        let ac = c.encrypt(
            &pack_activations(&a, &layout),
            c.params.scale(),
            level,
            &mut r,
        );
        let mut constants = vec![nag.grad_scale() / 8.0];
        if let Some(c_ref) = nag.reference_grad_scale().unwrap() {
            constants.push(c_ref / 8.0);
        }
        let hs = layer
            .weight_grad_scaled_consts(&c.ev, &dc, &ac, &layout, &constants)
            .unwrap();
        let before = c.ev.counter();
        nag.step_with_reference_gradient(&c.ev, &mut layer.w, &hs[0], hs.get(1))
            .unwrap();
        let count = &c.ev.counter() - &before;
        let products = if level == 9 {
            0
        } else {
            2 * hs[0].entries().len() as u64
        };
        assert_eq!(count.ct_pt_mults, products);
        assert_eq!(count.rescales, products);
        assert_eq!(layer.w.level(), level - 2);
        let g = delta.t().dot(&a) / 8.0;
        v = 0.9 * &v + &g;
        want = &want - &(lr * (&g + 0.9 * &v));
        let error = mat_diff(&layer.w.decrypt(&c.dec, &c.encoder, &layout), &want);
        assert!(
            error <= TOL,
            "lr={lr}, changing-rate weight error {error:e}"
        );
        let u = EncWeights::from_diagonals(nag.velocity().unwrap().clone(), 2, 8, &layout).unwrap();
        assert!(mat_diff(&u.decrypt(&c.dec, &c.encoder, &layout), &(0.08 * 1.9 * &v)) <= TOL);
        assert!(
            nag.set_lr_reference(0.1).is_err(),
            "basis locked after update"
        );
    }
    let state_before = c.decrypt_diags(nag.velocity().unwrap());
    let before = c.ev.counter();
    for lr in [0.0, -0.01, f64::NAN, f64::INFINITY, 0.09] {
        assert!(nag.set_lr(lr).is_err());
        assert_eq!(nag.lr, 0.08);
        assert_eq!(c.decrypt_diags(nag.velocity().unwrap()), state_before);
    }
    assert_eq!(c.ev.counter(), before);
    let state = nag.take_velocity();
    assert!(
        nag.set_lr_reference(0.1).is_err(),
        "taking state cannot unlock its basis"
    );
    nag.momentum = 0.8;
    assert!(nag.reference_grad_scale().is_err());
    assert!(nag.set_lr(0.02).is_err());
    let dw = layer.w.diagonals().clone();
    assert!(matches!(
        nag.step(&c.ev, &mut layer.w, &dw),
        Err(HekitError::InvalidParams(_))
    ));
    assert_eq!(c.ev.counter(), before);
    nag.momentum = 0.9;
    nag.set_velocity(state);
    assert!(nag.reference_grad_scale().is_ok());
}

#[test]
fn two_gradient_constants_share_products_and_reduction() {
    for batch in [1, 8] {
        let params = CkksParams::seal_2020();
        let layout = Layout::new(params.slots(), 8, batch).unwrap();
        let c = ctx(&grad_rotation_steps(2, 8, &layout), 72);
        let mut r = rng(172);
        let (_, _, layer) = random_layer(&c, 2, 8, &layout, &mut r);
        let d = rand_mat(&mut r, batch, 2, 1.0);
        let a = rand_mat(&mut r, batch, 8, 1.0);
        let dc = c.encrypt(&pack_activations(&d, &layout), c.params.scale(), 7, &mut r);
        let ac = c.encrypt(&pack_activations(&a, &layout), c.params.scale(), 7, &mut r);
        let before = c.ev.counter();
        let h = layer
            .weight_grad_scaled(&c.ev, &dc, &ac, &layout, 0.03)
            .unwrap();
        let single = &c.ev.counter() - &before;
        let href = layer
            .weight_grad_scaled(&c.ev, &dc, &ac, &layout, 0.12)
            .unwrap();
        let before = c.ev.counter();
        let both = layer
            .weight_grad_scaled_consts(&c.ev, &dc, &ac, &layout, &[0.03, 0.12])
            .unwrap();
        let mut expected = single;
        let m = h.entries().len() as u64;
        expected.ct_pt_mults += m;
        expected.rescales += m;
        expected.rotations += m * layout.log_batch() as u64;
        assert_eq!(&c.ev.counter() - &before, expected);
        for (actual, wanted) in both.iter().zip([&h, &href]) {
            assert_eq!(c.decrypt_diags(actual), c.decrypt_diags(wanted));
            assert!(
                actual
                    .entries()
                    .iter()
                    .all(|ct| ct.level() == 5 && ct.scale() == layer.w.scale())
            );
        }
        for constants in [vec![], vec![0.03, f64::NAN], vec![f64::INFINITY]] {
            let before = c.ev.counter();
            assert!(
                layer
                    .weight_grad_scaled_consts(&c.ev, &dc, &ac, &layout, &constants)
                    .is_err()
            );
            assert_eq!(c.ev.counter(), before);
        }
        let mut nag = Nag::new(0.03, 0.9);
        nag.set_lr_reference(0.12).unwrap();
        let mut w = layer.w.clone();
        let before = c.ev.counter();
        assert!(
            nag.step(&c.ev, &mut w, &h).is_err(),
            "missing reference must not silently use h"
        );
        let low = href.mod_switch_to_level(&c.ev, 4).unwrap();
        assert!(
            nag.step_with_reference_gradient(&c.ev, &mut w, &h, Some(&low))
                .is_err()
        );
        assert_eq!(c.ev.counter(), before);
        assert!(nag.velocity().is_none());
        assert_eq!(
            c.decrypt_diags(w.diagonals()),
            c.decrypt_diags(layer.w.diagonals())
        );
    }
}

#[test]
fn checked_rates_preserve_configuration_on_errors() {
    let mut nag = Nag::new(0.04, 0.9);
    let mut sgd = Sgd { lr: 0.04 };
    for lr in [0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(nag.set_lr(lr).is_err());
        assert!(nag.set_lr_reference(lr).is_err());
        assert!(sgd.set_lr(lr).is_err());
        assert_eq!(nag.lr, 0.04);
        assert_eq!(nag.lr_reference(), 0.04);
        assert_eq!(sgd.lr, 0.04);
    }
    assert!(nag.set_lr(0.08).is_err());
    assert!(nag.set_lr_reference(0.02).is_err());
    nag.set_lr_reference(0.08).unwrap();
    nag.set_lr(0.08).unwrap();
    sgd.set_lr(0.8).unwrap();
    assert_eq!(sgd.grad_scale(), 0.8);
    assert_eq!(nag.reference_grad_scale().unwrap(), None);
    nag.lr = f64::NAN;
    assert!(nag.reference_grad_scale().is_err());
}

/// Errors are reported before any homomorphic work (the counter does not move).
#[test]
fn errors_before_work() {
    let (d_out, d_in, batch) = (8, 4, 8);
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, batch).unwrap();
    let steps = grad_rotation_steps(d_out, d_in, &layout);
    let c = ctx(&steps, 6);
    let mut r = rng(106);
    let (_, _, layer) = random_layer(&c, d_out, d_in, &layout, &mut r);
    let (scale, top) = (c.params.scale(), c.params.max_level());
    let dp = pack_activations(&rand_mat(&mut r, batch, d_out, 1.0), &layout);
    let ap = pack_activations(&rand_mat(&mut r, batch, d_in, 1.0), &layout);
    let dct = c.encrypt(&dp, scale, top, &mut r);
    let act = c.encrypt(&ap, scale, top, &mut r);
    let is_invalid = |e: HekitError| matches!(e, HekitError::InvalidParams(_));

    // A missing Galois key (the lane step 1, used by both gradients), a missing relin key,
    // a foreign layout.
    let no_lane1: Vec<i32> = steps.iter().copied().filter(|&s| s != 1).collect();
    let mut kg = KeyGenerator::new(Arc::clone(&c.params), rng(6)).unwrap();
    let _ = (kg.public_key(), kg.relin_key());
    let few = Evaluator::new(
        Arc::clone(&c.params),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&no_lane1)),
    );
    assert!(is_invalid(
        layer.weight_grad(&few, &dct, &act, &layout).unwrap_err()
    ));
    assert!(is_invalid(
        layer.bias_grad(&few, &dct, &layout).unwrap_err()
    ));
    let bare = Evaluator::new(Arc::clone(&c.params), None, c.ev.galois_keys().cloned());
    assert!(is_invalid(
        layer.weight_grad(&bare, &dct, &act, &layout).unwrap_err()
    ));
    assert_eq!(few.counter(), OpCounter::default());
    assert_eq!(bare.counter(), OpCounter::default());
    let ev = c.ev.clone();
    ev.reset_counter();
    let other = Layout::new(params.slots(), 8, 4).unwrap();
    assert!(is_invalid(
        layer.weight_grad(&ev, &dct, &act, &other).unwrap_err()
    ));
    assert!(is_invalid(layer.bias_grad(&ev, &dct, &other).unwrap_err()));
    // Two levels are needed for the weight gradient, one for the bias gradient.
    let d1 = c.encrypt(&dp, scale, 1, &mut r);
    let a1 = c.encrypt(&ap, scale, 1, &mut r);
    let d0 = c.encrypt(&dp, scale, 0, &mut r);
    assert_eq!(
        layer.weight_grad(&ev, &d1, &a1, &layout).unwrap_err(),
        HekitError::NoLevelsLeft
    );
    assert_eq!(
        layer.bias_grad(&ev, &d0, &layout).unwrap_err(),
        HekitError::NoLevelsLeft
    );
    assert!(is_invalid(
        layer
            .weight_grad_scaled(&ev, &dct, &act, &layout, f64::NAN)
            .unwrap_err()
    ));
    assert!(is_invalid(
        layer
            .bias_grad_scaled(&ev, &dct, &layout, f64::INFINITY)
            .unwrap_err()
    ));
    assert_eq!(ev.counter(), OpCounter::default());

    // Optimizer shape / level / scale checks.
    let g = layer
        .weight_grad(
            &ev,
            &c.encrypt(&dp, scale, 3, &mut r),
            &c.encrypt(&ap, scale, 3, &mut r),
            &layout,
        )
        .unwrap();
    assert_eq!(g.entries()[0].level(), 1);
    ev.reset_counter();
    let mut w_low = EncWeights::from_diagonals(
        layer.w.diagonals().mod_switch_to_level(&ev, 0).unwrap(),
        d_out,
        d_in,
        &layout,
    )
    .unwrap();
    assert_eq!(
        Sgd { lr: 0.1 }.step(&ev, &mut w_low, &g).unwrap_err(),
        HekitError::LevelMismatch
    );
    assert_eq!(
        Nag::new(0.1, 0.9).step(&ev, &mut w_low, &g).unwrap_err(),
        HekitError::LevelMismatch
    );
    let zeros = vec![vec![0.0; params.slots()]; 4];
    let mut w_scaled = EncWeights::from_diagonals(
        Diagonals::encrypt(&c.encoder, &c.enc, &zeros, batch, 2.0 * scale, top, &mut r).unwrap(),
        d_out,
        d_in,
        &layout,
    )
    .unwrap();
    assert_eq!(
        Sgd { lr: 0.1 }.step(&ev, &mut w_scaled, &g).unwrap_err(),
        HekitError::ScaleMismatch
    );
    assert_eq!(
        Nag::new(0.1, 0.9).step(&ev, &mut w_scaled, &g).unwrap_err(),
        HekitError::ScaleMismatch
    );
    let mut w = layer.w.clone();
    let g_wrong = Diagonals::from_prerotated(2, batch, g.entries()[..2].to_vec()).unwrap();
    assert!(is_invalid(
        Sgd { lr: 0.1 }.step(&ev, &mut w, &g_wrong).unwrap_err()
    ));
    assert!(is_invalid(
        Nag::new(0.1, 0.9).step(&ev, &mut w, &g_wrong).unwrap_err()
    ));
    let mut g_mixed = g.clone();
    g_mixed.entries_mut()[1] = ev.mod_switch_to_level(&g.entries()[1], 0).unwrap();
    assert_eq!(
        Sgd { lr: 0.1 }.step(&ev, &mut w, &g_mixed).unwrap_err(),
        HekitError::LevelMismatch
    );
    assert_eq!(ev.counter(), OpCounter::default());
    assert_eq!(w.level(), top, "weights untouched by failed steps");
}

/// Task 26 review: `EncWeights::from_diagonals` checks that all entries share one level and
/// one scale (its `level()` / `scale()` read entry 0).
#[test]
fn from_diagonals_checks_uniform_level_and_scale() {
    let layout = Layout::new(1 << 14, 8, 8).unwrap();
    let c = ctx(&[], 7);
    let mut r = rng(107);
    let (_, _, layer) = random_layer(&c, 8, 4, &layout, &mut r);
    let good = layer.w.diagonals().clone();
    assert!(EncWeights::from_diagonals(good.clone(), 8, 4, &layout).is_ok());
    let mut low = good.clone();
    low.entries_mut()[2] = c.ev.mod_switch_to_level(&good.entries()[2], 3).unwrap();
    assert_eq!(
        EncWeights::from_diagonals(low, 8, 4, &layout).unwrap_err(),
        HekitError::LevelMismatch
    );
    let mut scaled = good.clone();
    let top = c.params.max_level();
    scaled.entries_mut()[3] = c.encrypt(&[0.0], c.params.scale() * (1.0 + 1e-9), top, &mut r);
    assert_eq!(
        EncWeights::from_diagonals(scaled, 8, 4, &layout).unwrap_err(),
        HekitError::ScaleMismatch
    );
}

/// The Task 30 network, 4 -> 8 -> 2, Square, batch 8, one SGD step on seal_2020 with every
/// constant folded (`c = lr / n` into the lane-0 masks, `2 valid` into the derivative
/// plaintexts). Every value's level is read from its ciphertext and must equal
/// `max_level - depth` of `sim_train_step_sgd`'s level table (`docs/layout.md` §4); `W1'` lands
/// on level 0, so the step fits the 9 levels with zero margin. The whole counter equals the
/// simulator's, and the updated layers match it.
#[test]
fn task30_step_levels_fit_seal_2020() {
    let (batch, lr) = (8, 0.1);
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), 8, batch).unwrap();
    let dims = [4usize, 8, 2];
    let keys = TransposeKeys::Dedicated;
    let steps = union(&[
        forward_rotation_steps(8, 4, &layout),
        forward_rotation_steps(2, 8, &layout),
        transpose_rotation_steps(2, 8, &layout, &keys).unwrap(),
        backward_rotation_steps(2, 8, &layout),
        grad_rotation_steps(8, 4, &layout),
        grad_rotation_steps(2, 8, &layout),
    ]);
    eprintln!("task 30 step: {} Galois keys {steps:?}", steps.len());
    let c = ctx(&steps, 8);
    let mut r = rng(108);
    let net: Vec<SimLayer> = dims
        .windows(2)
        .map(|p| SimLayer {
            w: rand_mat(&mut r, p[1], p[0], 0.8 / (p[0] as f64).sqrt()),
            b: (0..p[1]).map(|_| 0.2 * uni(&mut r)).collect(),
        })
        .collect();
    let x = rand_mat(&mut r, batch, 4, 1.0);
    let y = rand_mat(&mut r, batch, 2, 1.0);
    let sim = sim_train_step_sgd(&net, &x, &y, lr, &layout);
    assert_eq!(sim.depth, 9);
    let top = c.params.max_level();
    let scale = c.params.scale();
    let level_of = |label: &str| top - sim.depth_of(label);
    let ev = &c.ev;
    ev.reset_counter();
    let mut seen: Vec<(String, usize)> = Vec::new();
    let mut check = |label: &str, level: usize| {
        assert_eq!(level, level_of(label), "level of {label}");
        seen.push((label.to_string(), level));
    };

    // Forward. Layer 2's bias is encrypted for a1's scale (the client knows the chain).
    let xct = c.encrypt(&pack_activations(&x, &layout), scale, top, &mut r);
    let mut l1 = Dense::encrypt(
        &net[0].w, &net[0].b, &layout, scale, &c.enc, &c.encoder, &mut r,
    )
    .unwrap();
    let z1 = l1.forward(ev, &xct, &layout).unwrap();
    check("z1", z1.level());
    let a1 = ev.rescale(&ev.mul(&z1, &z1).unwrap()).unwrap();
    check("a1", a1.level());
    let mut l2 = Dense::encrypt(
        &net[1].w,
        &net[1].b,
        &layout,
        a1.scale(),
        &c.enc,
        &c.encoder,
        &mut r,
    )
    .unwrap();
    let z2 = l2.forward(ev, &a1, &layout).unwrap();
    check("z2", z2.level());
    let a2 = ev.rescale(&ev.mul(&z2, &z2).unwrap()).unwrap();
    check("a2", a2.level());
    // Loss gradient: y encrypted at a2's exact scale and level.
    let yct = c.encrypt(
        &pack_activations(&y, &layout),
        a2.scale(),
        a2.level(),
        &mut r,
    );
    let delta2 = ev.sub(&a2, &yct).unwrap();
    check("delta2", delta2.level());

    // Derivative factors s = z * (2 valid): plaintexts at z's level (off the critical path).
    let two_valid: Vec<f64> = valid_lane_mask(&layout, batch)
        .iter()
        .map(|m| 2.0 * m)
        .collect();
    let s_of = |z: &Ciphertext| {
        let pt = c.encoder.encode(&two_valid, scale, z.level()).unwrap();
        ev.rescale(&ev.mul_plain(z, &pt).unwrap()).unwrap()
    };
    let n = batch as f64;
    let sgd = Sgd { lr };
    let cfold = sgd.grad_scale() / n;

    // Layer 2.
    let s2 = s_of(&z2);
    check("s2", s2.level());
    let dz2 = ev.rescale(&ev.mul(&delta2, &s2).unwrap()).unwrap();
    check("dz2", dz2.level());
    let wt2 = l2.w.transpose(ev, &layout).unwrap();
    let delta1 = l2.backward_input(ev, &dz2, &wt2, &layout).unwrap();
    check("delta1", delta1.level());
    let g2 = l2
        .weight_grad_scaled(ev, &dz2, &a1, &layout, cfold)
        .unwrap();
    let mut sgd2 = sgd;
    sgd2.step(ev, &mut l2.w, &g2).unwrap();
    check("W2'", l2.w.level());
    let gb2 = l2.bias_grad_scaled(ev, &dz2, &layout, cfold).unwrap();
    let b2 = ev
        .sub(&ev.mod_switch_to_level(&l2.b, gb2.level()).unwrap(), &gb2)
        .unwrap();
    assert_eq!(b2.scale(), l2.b.scale(), "bias keeps its exact scale");
    check("b2'", b2.level());

    // Layer 1.
    let s1 = s_of(&z1);
    check("s1", s1.level());
    let s1d = ev.mod_switch_to_level(&s1, delta1.level()).unwrap();
    let dz1 = ev.rescale(&ev.mul(&delta1, &s1d).unwrap()).unwrap();
    check("dz1", dz1.level());
    let g1 = l1
        .weight_grad_scaled(ev, &dz1, &xct, &layout, cfold)
        .unwrap();
    let mut sgd1 = sgd;
    sgd1.step(ev, &mut l1.w, &g1).unwrap();
    check("W1'", l1.w.level());
    assert_eq!(
        l1.w.level(),
        0,
        "W1' on the last level: 9 levels, zero margin"
    );
    let gb1 = l1.bias_grad_scaled(ev, &dz1, &layout, cfold).unwrap();
    let b1 = ev
        .sub(&ev.mod_switch_to_level(&l1.b, gb1.level()).unwrap(), &gb1)
        .unwrap();
    assert_eq!(b1.scale(), l1.b.scale());
    check("b1'", b1.level());
    let cnt = ev.counter();
    eprintln!("task 30 step levels {seen:?}");
    eprintln!("task 30 step counter {cnt:?} (sim {:?})", sim.count);
    assert_eq!(
        cnt,
        sim.count.to_op_counter(),
        "whole step counter == simulator"
    );

    // Values: the updated layers match the simulator's.
    for (i, (wt, bt)) in [(&l1.w, &b1), (&l2.w, &b2)].into_iter().enumerate() {
        let got = wt.decrypt(&c.dec, &c.encoder, &layout);
        let ew = mat_diff(&got, &sim.layers[i].w);
        let gb = unpack_activations(&c.decrypt(bt), &layout, sim.layers[i].b.len());
        let eb = max_abs_diff(gb.row(0).as_slice().unwrap(), &sim.layers[i].b);
        eprintln!(
            "task 30 step layer {}: W' error 2^{:.2}, b' error 2^{:.2}",
            i + 1,
            ew.log2(),
            eb.log2()
        );
        assert!(ew <= TOL && eb <= TOL, "layer {}: {ew:e} {eb:e}", i + 1);
    }
}
