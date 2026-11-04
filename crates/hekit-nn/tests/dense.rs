//! Task 26: encrypted dense forward (`hekit_nn::dense`). Every case decrypts the forward output
//! and compares every slot with the plaintext `X W^T + b` in canonical form, runs the Task 25
//! simulator on the same data and requires the evaluator's counter to equal the simulator's
//! exactly (`relins == giant`). Galois keys are generated for exactly
//! [`forward_rotation_steps`]. Fixed seeds.

use std::sync::Arc;

use hekit_ckks::{CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter};
use hekit_core::error::HekitError;
use hekit_core::sampler::Rng;
use hekit_nn::dense::{Dense, EncWeights, encrypt_bias, forward_rotation_steps};
use hekit_nn::layout::{LayerShape, Layout, pack_activations, pack_bias, pack_weight_diagonals};
use hekit_nn::sim::{SimCounter, sim_dense_forward};
use ndarray::Array2;

/// The brief's precision gate: every slot within 2^-15 of the plaintext reference.
const TOL: f64 = 1.0 / 32768.0;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

/// Uniform in `[-1, 1)`.
fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

fn rand_mat(r: &mut Rng, rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |_| uni(r))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
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
}

/// One shared `Arc<CkksParams>`; Galois keys for exactly `steps`.
fn ctx(params: CkksParams, steps: &[i32], seed: u8) -> Ctx {
    let params = Arc::new(params);
    let mut kg = KeyGenerator::new(Arc::clone(&params), rng(seed)).unwrap();
    let pk = kg.public_key();
    let rk = kg.relin_key();
    let gks = kg.galois_keys(steps);
    assert_eq!(gks.len(), steps.len(), "one key per forward step");
    Ctx {
        encoder: Encoder::new(Arc::clone(&params)),
        enc: Encryptor::new(Arc::clone(&params), pk),
        dec: Decryptor::new(Arc::clone(&params), kg.secret_key()),
        ev: Evaluator::new(Arc::clone(&params), Some(rk), Some(gks)),
        params,
    }
}

/// `small`: N = 2^12 (2048 slots), levels 3.
fn small_params() -> CkksParams {
    CkksParams::new(12, &[60, 40, 40, 40], &[60], 40, 4).unwrap()
}

/// Encrypted forward of a random `d_out x d_in` layer on `batch` random samples, checked
/// against the plaintext reference and the simulator (values and counts). Returns the max
/// slot error.
fn check_forward(
    params: CkksParams,
    d_out: usize,
    d_in: usize,
    d: usize,
    batch: usize,
    seed: u8,
) -> f64 {
    let slots = params.slots();
    let layout = Layout::new(slots, d, batch).unwrap();
    let sh = LayerShape::new(d_out, d_in, &layout);
    let steps = forward_rotation_steps(d_out, d_in, &layout);
    let c = ctx(params, &steps, seed);
    let mut r = rng(seed.wrapping_add(100));
    let w = rand_mat(&mut r, d_out, d_in);
    let bias: Vec<f64> = (0..d_out).map(|_| uni(&mut r)).collect();
    let x = rand_mat(&mut r, batch, d_in);

    // Plaintext reference X W^T + b, in canonical form.
    let mut z = x.dot(&w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(&bias).for_each(|(v, b)| *v += b);
    }
    let want = pack_activations(&z, &layout);

    // Simulator on the same data.
    let mut sc = SimCounter::default();
    let diags = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
    let xp = pack_activations(&x, &layout);
    let sim = sim_dense_forward(
        &diags,
        &pack_bias(&bias, &layout),
        &xp,
        d_out,
        d_in,
        &layout,
        &mut sc,
    );
    assert!(max_abs_diff(&sim, &want) < 1e-9, "simulator vs reference");

    // Encrypted forward.
    let (delta, top) = (c.params.scale(), c.params.max_level());
    let xct = c
        .enc
        .encrypt(&c.encoder.encode(&xp, delta, top).unwrap(), &mut r);
    let layer =
        Dense::encrypt(&w, &bias, &layout, xct.scale(), &c.enc, &c.encoder, &mut r).unwrap();
    assert_eq!(layer.w.d_out(), d_out);
    assert_eq!(layer.w.d_in(), d_in);
    assert_eq!(layer.forward_rotation_steps(&layout), steps);
    c.ev.reset_counter();
    let y = layer.forward(&c.ev, &xct, &layout).unwrap();
    let cnt = c.ev.counter();

    assert_eq!(y.level(), top - 1, "forward costs one level");
    assert_eq!(y.len(), 2);
    assert_eq!(
        cnt,
        sc.to_op_counter(),
        "counter == simulator ({d_out} x {d_in}, B = {batch})"
    );
    assert_eq!(cnt.relins, sh.giant as u64, "relins == giant");
    assert_eq!(cnt.ct_ct_tensor, sh.m as u64);
    assert_eq!(
        cnt.rotations,
        (sh.baby + sh.giant - 2 + sh.fold_steps()) as u64,
        "baby + giant - 2 + fold"
    );
    assert_eq!(cnt.rescales, 1);
    assert_eq!(cnt.ct_pt_mults, 0);

    let got = c.encoder.decode(&c.dec.decrypt(&y));
    let err = max_abs_diff(&got, &want);
    let err_sim = max_abs_diff(&got, &sim);
    eprintln!(
        "dense {d_out} x {d_in}, d = {d}, B = {batch}, slots = {slots}: max error 2^{:.2} \
         (vs sim 2^{:.2}), counter {cnt:?}",
        err.log2(),
        err_sim.log2()
    );
    assert!(err <= TOL, "{d_out} x {d_in}: max error {err:e} > 2^-15");
    err
}

#[test]
fn rotation_steps_are_bsgs_plus_fold() {
    let layout = Layout::new(1 << 14, 64, 8).unwrap();
    // Square 16 x 16: bsgs_rotation_steps(16, 8), no fold.
    assert_eq!(
        forward_rotation_steps(16, 16, &layout),
        vec![8, 16, 24, 32, 64, 96]
    );
    // Wide 16 x 64: m = 16, plus the fold steps 16 * 8 and 32 * 8.
    assert_eq!(
        forward_rotation_steps(16, 64, &layout),
        vec![8, 16, 24, 32, 64, 96, 128, 256]
    );
    // Tall 64 x 16: m = 16, no fold.
    assert_eq!(
        forward_rotation_steps(64, 16, &layout),
        vec![8, 16, 24, 32, 64, 96]
    );
    // Iris-like 3 x 4 (padded 4 x 4: split 2 x 2) and 2 x 8 (m = 2, fold 2 * 8 and 4 * 8).
    assert_eq!(forward_rotation_steps(3, 4, &layout), vec![8, 16]);
    assert_eq!(forward_rotation_steps(2, 8, &layout), vec![8, 16, 32]);
}

#[test]
fn square_16x16_batch8_seal2020() {
    check_forward(CkksParams::seal_2020(), 16, 16, 16, 8, 1);
}

#[test]
fn wide_16x64_batch8_seal2020() {
    check_forward(CkksParams::seal_2020(), 16, 64, 64, 8, 2);
}

#[test]
fn tall_64x16_batch8_seal2020() {
    check_forward(CkksParams::seal_2020(), 64, 16, 64, 8, 3);
}

/// `d * B == slots` (a single block, no replica): N = 2^12, (16, 16), B = 128.
#[test]
fn full_block_16x16_batch128_small() {
    check_forward(small_params(), 16, 16, 16, 128, 4);
}

/// Non-power-of-two logical dimensions (padded 4 x 8, wide): padded features stay 0.
#[test]
fn padded_3x5_batch4_small() {
    check_forward(small_params(), 3, 5, 8, 4, 5);
}

/// The weights and bias are switched down to a lower-level input; `EncWeights::decrypt`
/// round-trips the matrix.
#[test]
fn lower_level_input_and_weight_roundtrip() {
    let layout = Layout::new(2048, 16, 8).unwrap();
    let (d_out, d_in) = (8, 16);
    let c = ctx(
        small_params(),
        &forward_rotation_steps(d_out, d_in, &layout),
        6,
    );
    let mut r = rng(106);
    let w = rand_mat(&mut r, d_out, d_in);
    let bias: Vec<f64> = (0..d_out).map(|_| uni(&mut r)).collect();
    let x = rand_mat(&mut r, 8, d_in);
    let weights = EncWeights::encrypt(&w, &layout, &c.enc, &c.encoder, &mut r).unwrap();
    assert_eq!(weights.level(), c.params.max_level());
    assert_eq!(weights.scale(), c.params.scale());
    let back = weights.decrypt(&c.dec, &c.encoder, &layout);
    let werr = max_abs_diff(back.as_slice().unwrap(), w.as_slice().unwrap());
    // Encryption noise only (~2^-28 at Delta = 2^40).
    assert!(werr < 2f64.powi(-24), "weight round trip {werr:e}");

    let level = 2;
    let xp = pack_activations(&x, &layout);
    let xct = c.enc.encrypt(
        &c.encoder.encode(&xp, c.params.scale(), level).unwrap(),
        &mut r,
    );
    let b = encrypt_bias(
        &bias,
        &layout,
        weights.product_scale(xct.scale()),
        c.params.max_level(),
        &c.enc,
        &c.encoder,
        &mut r,
    )
    .unwrap();
    let layer = Dense { w: weights, b };
    let y = layer.forward(&c.ev, &xct, &layout).unwrap();
    assert_eq!(y.level(), level - 1);
    let mut z = x.dot(&w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(&bias).for_each(|(v, b)| *v += b);
    }
    let err = max_abs_diff(
        &c.encoder.decode(&c.dec.decrypt(&y)),
        &pack_activations(&z, &layout),
    );
    assert!(err <= TOL, "lower level: max error {err:e}");
}

/// Errors are reported before any homomorphic work (the counter does not move): a bias not at
/// the exact product scale, a missing fold key, a layout of another slot count or batch.
#[test]
fn errors_before_work() {
    let layout = Layout::new(2048, 16, 8).unwrap();
    let (d_out, d_in) = (4, 16);
    let steps = forward_rotation_steps(d_out, d_in, &layout);
    // Keys without the last fold step.
    let c = ctx(small_params(), &steps[..steps.len() - 1], 7);
    let mut r = rng(107);
    let w = rand_mat(&mut r, d_out, d_in);
    let bias: Vec<f64> = (0..d_out).map(|_| uni(&mut r)).collect();
    let xp = pack_activations(&rand_mat(&mut r, 8, d_in), &layout);
    let (delta, top) = (c.params.scale(), c.params.max_level());
    let xct = c
        .enc
        .encrypt(&c.encoder.encode(&xp, delta, top).unwrap(), &mut r);
    let zero = OpCounter::default();

    let layer =
        Dense::encrypt(&w, &bias, &layout, xct.scale(), &c.enc, &c.encoder, &mut r).unwrap();
    c.ev.reset_counter();
    let e = layer.forward(&c.ev, &xct, &layout).unwrap_err();
    assert!(
        matches!(e, HekitError::InvalidParams(ref m) if m.contains("missing galois key")),
        "{e:?}"
    );
    assert_eq!(c.ev.counter(), zero);

    // Full keys, bias at Delta (not the exact product scale Delta^2): ScaleMismatch.
    let c = ctx(small_params(), &steps, 8);
    let xct = c
        .enc
        .encrypt(&c.encoder.encode(&xp, delta, top).unwrap(), &mut r);
    let weights = EncWeights::encrypt(&w, &layout, &c.enc, &c.encoder, &mut r).unwrap();
    let b = encrypt_bias(&bias, &layout, delta, top, &c.enc, &c.encoder, &mut r).unwrap();
    let layer = Dense { w: weights, b };
    c.ev.reset_counter();
    let e = layer.forward(&c.ev, &xct, &layout).unwrap_err();
    assert!(matches!(e, HekitError::ScaleMismatch), "{e:?}");
    assert_eq!(c.ev.counter(), zero);

    // Bias below the input level: LevelMismatch.
    let weights = layer.w;
    let b = encrypt_bias(
        &bias,
        &layout,
        weights.product_scale(xct.scale()),
        top - 1,
        &c.enc,
        &c.encoder,
        &mut r,
    )
    .unwrap();
    let layer = Dense { w: weights, b };
    let e = layer.forward(&c.ev, &xct, &layout).unwrap_err();
    assert!(matches!(e, HekitError::LevelMismatch), "{e:?}");
    assert_eq!(c.ev.counter(), zero);

    // A layout with another batch (stride) or slot count.
    for other in [
        Layout::new(2048, 16, 4).unwrap(),
        Layout::new(4096, 16, 8).unwrap(),
    ] {
        let e = layer.forward(&c.ev, &xct, &other).unwrap_err();
        assert!(
            matches!(e, HekitError::InvalidParams(_)),
            "{other:?}: {e:?}"
        );
        assert_eq!(c.ev.counter(), zero);
    }

    // A layer that does not fit the layout is an error, not a panic.
    let big = rand_mat(&mut r, 32, 16);
    let e = EncWeights::encrypt(&big, &layout, &c.enc, &c.encoder, &mut r).unwrap_err();
    assert!(matches!(e, HekitError::InvalidParams(_)), "{e:?}");
}
