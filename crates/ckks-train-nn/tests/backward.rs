//! Task 27: rotation-only encrypted transpose and backward pass (`ckks_train_nn::dense`).
//!
//! Every encrypted case decrypts all `m` transposed diagonals and compares every slot with
//! `pack_weight_diagonals(W^T)`, requires the transpose to cost rotations only (relins, tensors,
//! ct-pt and rescales stay 0) with the simulator's count, then runs `backward_input` and compares
//! every slot with the plaintext `delta W` in canonical form and its counter with
//! `sim_backward_input`. Both Galois-key modes (`TransposeKeys::Dedicated` and `Reuse`). Galois
//! keys are generated for exactly the steps the case needs. Fixed seeds.

use std::collections::BTreeSet;
use std::sync::Arc;

use ckks_train_ckks::{
    CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::error::HekitError;
use ckks_train_core::sampler::Rng;
use ckks_train_nn::dense::{
    Dense, EncWeights, TransposeKeys, backward_rotation_steps, forward_rotation_steps,
    training_key_pool, transpose_rotation_plan, transpose_rotation_steps,
};
use ckks_train_nn::layout::{
    LayerShape, Layout, pack_activations, pack_weight_diagonals, rotations_for_transpose,
    transpose_galois_steps, transpose_sources,
};
use ckks_train_nn::sim::{SimCounter, sim_backward_input, sim_rotate_counted, sim_transpose};
use ndarray::Array2;

/// The brief's gate for the transposed diagonals: every slot within 2^-18.
const TOL_T: f64 = 1.0 / 262144.0;
/// The brief's gate for `delta W`: every slot within 2^-15.
const TOL_B: f64 = 1.0 / 32768.0;

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

fn union(a: &[i32], b: &[i32]) -> Vec<i32> {
    a.iter()
        .chain(b)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Runs a rotation plan on plaintext diagonals with the simulator's counting rules.
fn sim_plan(plan: &[(usize, Vec<i32>)], diags: &[Vec<f64>], cnt: &mut SimCounter) -> Vec<Vec<f64>> {
    plan.iter()
        .map(|(c, steps)| {
            steps
                .iter()
                .fold(diags[*c].clone(), |v, &s| sim_rotate_counted(&v, s, cnt))
        })
        .collect()
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
    assert_eq!(gks.len(), steps.len(), "one key per step");
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

#[derive(Clone, Copy, Debug)]
enum Mode {
    Dedicated,
    Reuse,
}

fn mode_for(mode: Mode, d_out: usize, d_in: usize, layout: &Layout) -> TransposeKeys {
    match mode {
        Mode::Dedicated => TransposeKeys::Dedicated,
        Mode::Reuse => TransposeKeys::Reuse(training_key_pool(d_out, d_in, layout)),
    }
}

/// Encrypted transpose (in `mode`) and backward input of a random `d_out x d_in` layer on
/// `batch` random `delta` rows, checked against the plaintext references and the simulator
/// (values and counts). Returns `(transpose error, backward error, transpose rotations)`.
fn check_transpose_backward(
    d_out: usize,
    d_in: usize,
    d: usize,
    batch: usize,
    mode: Mode,
    seed: u8,
) -> (f64, f64, u64) {
    let params = CkksParams::seal_2020();
    let layout = Layout::new(params.slots(), d, batch).unwrap();
    let sh = LayerShape::new(d_out, d_in, &layout);
    let keys = mode_for(mode, d_out, d_in, &layout);
    let t_steps = transpose_rotation_steps(d_out, d_in, &layout, &keys).unwrap();
    let b_steps = backward_rotation_steps(d_out, d_in, &layout);
    let c = ctx(params, &union(&t_steps, &b_steps), seed);
    let mut r = rng(seed.wrapping_add(100));
    let w = rand_mat(&mut r, d_out, d_in);
    let bias: Vec<f64> = (0..d_out).map(|_| uni(&mut r)).collect();
    let delta = rand_mat(&mut r, batch, d_out);

    // Plaintext: the transposed diagonals and the plan's simulated count.
    let wd = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
    let want_t = pack_weight_diagonals(&w.t().to_owned(), &layout, (sh.baby, sh.giant));
    let plan = transpose_rotation_plan(d_out, d_in, &layout, &keys).unwrap();
    let mut sc_t = SimCounter::default();
    let sim_t = sim_plan(&plan, &wd, &mut sc_t);
    for (i, (s, want)) in sim_t.iter().zip(&want_t).enumerate() {
        assert_eq!(s, want, "plaintext plan entry {i} is exact");
    }
    if let Mode::Dedicated = mode {
        let mut sc = SimCounter::default();
        let st = sim_transpose(&wd, d_out, d_in, &layout, &mut sc);
        assert_eq!(st, sim_t, "dedicated plan == sim_transpose");
        assert_eq!(sc, sc_t);
        assert_eq!(sc.rotations, (sh.m - 1) as u64, "m - 1 rotations");
    }

    // Encrypted transpose.
    let (scale, top) = (c.params.scale(), c.params.max_level());
    let layer = Dense::encrypt(&w, &bias, &layout, scale, &c.enc, &c.encoder, &mut r).unwrap();
    assert_eq!(
        layer.w.transpose_rotation_steps(&layout, &keys).unwrap(),
        t_steps
    );
    c.ev.reset_counter();
    let wt = layer.w.transpose_with(&c.ev, &layout, &keys).unwrap();
    let cnt_t = c.ev.counter();
    assert_eq!(
        cnt_t,
        sc_t.to_op_counter(),
        "transpose counter == simulator ({d_out} x {d_in}, {mode:?})"
    );
    assert_eq!(cnt_t.relins, 0, "transpose adds no relin");
    assert_eq!(cnt_t.ct_ct_tensor, 0, "transpose adds no tensor");
    assert_eq!(cnt_t.ct_pt_mults, 0);
    assert_eq!(cnt_t.rescales, 0);
    assert_eq!((wt.d_out(), wt.d_in()), (d_in, d_out));
    assert_eq!(wt.level(), layer.w.level(), "transpose costs no level");
    assert_eq!(wt.scale(), layer.w.scale(), "transpose keeps the scale");
    let mut err_t: f64 = 0.0;
    for (i, (ct, want)) in wt.diagonals().entries().iter().zip(&want_t).enumerate() {
        let got = c.encoder.decode(&c.dec.decrypt(ct));
        let e = max_abs_diff(&got, want);
        assert!(e <= TOL_T, "entry {i}: transpose error {e:e} > 2^-18");
        err_t = err_t.max(e);
    }
    // The unpacked matrix is W^T.
    let back = wt.decrypt(&c.dec, &c.encoder, &layout);
    let werr = max_abs_diff(
        back.as_standard_layout().as_slice().unwrap(),
        w.t().as_standard_layout().as_slice().unwrap(),
    );
    assert!(werr <= TOL_T, "decrypt(W^T) error {werr:e}");

    // Backward input delta W, delta one level below the weights (as in training).
    let level = top - 1;
    let dp = pack_activations(&delta, &layout);
    let dct = c
        .enc
        .encrypt(&c.encoder.encode(&dp, scale, level).unwrap(), &mut r);
    let want_b = pack_activations(&delta.dot(&w), &layout);
    let mut sc_b = SimCounter::default();
    let sim_b = sim_backward_input(&sim_t, &dp, d_out, d_in, &layout, &mut sc_b);
    assert!(
        max_abs_diff(&sim_b, &want_b) < 1e-9,
        "simulator vs reference"
    );
    assert_eq!(layer.backward_rotation_steps(&layout), b_steps);
    c.ev.reset_counter();
    let v = layer.backward_input(&c.ev, &dct, &wt, &layout).unwrap();
    let cnt_b = c.ev.counter();
    assert_eq!(
        cnt_b,
        sc_b.to_op_counter(),
        "backward counter == simulator ({d_out} x {d_in})"
    );
    let sht = sh.transposed();
    assert_eq!(cnt_b.relins, sh.giant as u64, "relins == giant");
    assert_eq!(cnt_b.ct_ct_tensor, sh.m as u64);
    assert_eq!(
        cnt_b.rotations,
        (sh.baby + sh.giant - 2 + sht.fold_steps()) as u64
    );
    assert_eq!(cnt_b.rescales, 1);
    assert_eq!(v.level(), level - 1, "backward costs one level");
    let q = c.params.q()[level] as f64;
    assert_eq!(v.scale(), dct.scale() * wt.scale() / q, "one rescale");
    let got = c.encoder.decode(&c.dec.decrypt(&v));
    let err_b = max_abs_diff(&got, &want_b);
    eprintln!(
        "{d_out} x {d_in}, d = {d}, B = {batch}, {mode:?}: transpose 2^{:.2} \
         ({} rotations, {} keys), backward 2^{:.2}, counter {cnt_b:?}",
        err_t.log2(),
        cnt_t.rotations,
        t_steps.len(),
        err_b.log2()
    );
    assert!(err_b <= TOL_B, "backward error {err_b:e} > 2^-15");
    (err_t, err_b, cnt_t.rotations)
}

#[test]
fn square_16x16_dedicated() {
    let (_, _, rot) = check_transpose_backward(16, 16, 16, 8, Mode::Dedicated, 11);
    assert_eq!(rot, 15);
}

#[test]
fn wide_16x64_dedicated() {
    let (_, _, rot) = check_transpose_backward(16, 64, 64, 8, Mode::Dedicated, 12);
    assert_eq!(rot, 15);
}

#[test]
fn tall_64x16_dedicated() {
    let (_, _, rot) = check_transpose_backward(64, 16, 64, 8, Mode::Dedicated, 13);
    assert_eq!(rot, 15);
}

#[test]
fn square_16x16_reuse() {
    let (_, _, rot) = check_transpose_backward(16, 16, 16, 8, Mode::Reuse, 14);
    assert_eq!(rot, 24);
}

#[test]
fn wide_16x64_reuse() {
    let (_, _, rot) = check_transpose_backward(16, 64, 64, 8, Mode::Reuse, 15);
    assert_eq!(rot, 30);
}

#[test]
fn tall_64x16_reuse() {
    let (_, _, rot) = check_transpose_backward(64, 16, 64, 8, Mode::Reuse, 16);
    assert_eq!(rot, 24);
}

/// The plans in plaintext, for many shapes and batches: the dedicated plan is
/// `rotations_for_transpose` (one rotation per entry, `m - 1` keys at most), the reuse plan
/// needs no key outside its pool, and both are exact.
#[test]
fn plans_are_exact_and_reuse_needs_no_new_key() {
    let mut r = rng(17);
    for batch in [1, 8] {
        for (d_out, d_in) in [
            (1, 4),
            (4, 1),
            (2, 8),
            (8, 4),
            (3, 5),
            (16, 16),
            (16, 64),
            (64, 16),
            (8, 64),
            (10, 32),
            (32, 8),
        ] {
            let layout = Layout::new(1 << 12, 64, batch).unwrap();
            let sh = LayerShape::new(d_out, d_in, &layout);
            let w = rand_mat(&mut r, d_out, d_in);
            let wd = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
            let want = pack_weight_diagonals(&w.t().to_owned(), &layout, (sh.baby, sh.giant));

            let ded =
                transpose_rotation_plan(d_out, d_in, &layout, &TransposeKeys::Dedicated).unwrap();
            let expect: Vec<(usize, Vec<i32>)> = transpose_sources(d_out, d_in, &layout)
                .into_iter()
                .zip(rotations_for_transpose(d_out, d_in, &layout))
                .map(|(c, a)| (c, if a == 0 { vec![] } else { vec![a] }))
                .collect();
            assert_eq!(ded, expect, "{d_out} x {d_in}, B = {batch}");
            assert_eq!(
                transpose_rotation_steps(d_out, d_in, &layout, &TransposeKeys::Dedicated).unwrap(),
                transpose_galois_steps(d_out, d_in, &layout)
            );

            let pool = training_key_pool(d_out, d_in, &layout);
            let reuse = TransposeKeys::Reuse(pool.clone());
            let plan = transpose_rotation_plan(d_out, d_in, &layout, &reuse).unwrap();
            let mut cnt = SimCounter::default();
            assert_eq!(
                sim_plan(&plan, &wd, &mut cnt),
                want,
                "{d_out} x {d_in} reuse"
            );
            let used = transpose_rotation_steps(d_out, d_in, &layout, &reuse).unwrap();
            assert!(
                used.iter().all(|s| pool.contains(s)),
                "{d_out} x {d_in}: reuse uses {used:?} outside {pool:?}"
            );
            // The pool is forward(W) + forward(W^T) + the gradient giants -g k1 B.
            let grad: Vec<i32> = (1..sh.giant)
                .map(|k1| -((sh.baby * k1 * batch) as i32))
                .collect();
            let fb = union(
                &forward_rotation_steps(d_out, d_in, &layout),
                &backward_rotation_steps(d_out, d_in, &layout),
            );
            assert_eq!(pool, union(&fb, &grad));
            // Each source diagonal is used once.
            let mut src: Vec<usize> = plan.iter().map(|(c, _)| *c).collect();
            src.sort_unstable();
            assert_eq!(src, (0..sh.m).collect::<Vec<_>>());
        }
    }
}

/// `docs/layout.md` §2.3's zero-new-key totals, with the pool of the whole training step
/// (all layers' forward, backward and gradient-giant keys): Iris 4 → 8 → 2 (B = 8),
/// Iris 4 → 8 → 3 (B = 32) and eMLP-1 W2 (B = 32 on 2^15 slots).
/// `(slots, d, batch, [(d_out, d_in)] of the network, reuse rotations of layer 2's transpose)`.
type DocCase = (usize, usize, usize, &'static [(usize, usize)], usize);

#[test]
fn reuse_totals_match_layout_doc() {
    let cases: [DocCase; 3] = [
        (1 << 14, 8, 8, &[(8, 4), (2, 8)], 2),
        (1 << 14, 8, 32, &[(8, 4), (3, 8)], 5),
        (1 << 15, 1024, 32, &[(32, 784), (10, 32)], 27),
    ];
    for (slots, d, batch, layers, want) in cases {
        let layout = Layout::new(slots, d, batch).unwrap();
        let pool = layers.iter().fold(Vec::new(), |p, &(o, i)| {
            union(&p, &training_key_pool(o, i, &layout))
        });
        let (o, i) = layers[1];
        let plan =
            transpose_rotation_plan(o, i, &layout, &TransposeKeys::Reuse(pool.clone())).unwrap();
        let rot: usize = plan.iter().map(|(_, s)| s.len()).sum();
        let ded = transpose_galois_steps(o, i, &layout);
        let new_keys = ded.iter().filter(|s| !pool.contains(s)).count();
        eprintln!(
            "{o} x {i}, B = {batch}: dedicated {} rotations ({new_keys} new keys), reuse {rot}",
            LayerShape::new(o, i, &layout).m - 1
        );
        assert_eq!(rot, want, "{o} x {i}, B = {batch}");
    }
}

/// Errors before any homomorphic work (the counter does not move).
#[test]
fn errors_before_work() {
    let layout = Layout::new(2048, 16, 8).unwrap();
    let (d_out, d_in) = (4, 16);
    let t_steps =
        transpose_rotation_steps(d_out, d_in, &layout, &TransposeKeys::Dedicated).unwrap();
    let b_steps = backward_rotation_steps(d_out, d_in, &layout);
    // Keys without the last transpose step.
    let last = *t_steps.last().unwrap();
    assert!(!b_steps.contains(&last));
    let c = ctx(
        small_params(),
        &union(&t_steps[..t_steps.len() - 1], &b_steps),
        18,
    );
    let mut r = rng(118);
    let w = rand_mat(&mut r, d_out, d_in);
    let bias = vec![0.0; d_out];
    let (scale, top) = (c.params.scale(), c.params.max_level());
    let layer = Dense::encrypt(&w, &bias, &layout, scale, &c.enc, &c.encoder, &mut r).unwrap();
    let zero = OpCounter::default();

    c.ev.reset_counter();
    let e = layer.w.transpose(&c.ev, &layout).unwrap_err();
    assert!(
        matches!(e, HekitError::InvalidParams(ref m) if m.contains("missing galois key")),
        "{e:?}"
    );
    assert_eq!(c.ev.counter(), zero);
    // A reuse pool without a usable step for some entry is reported, not a panic.
    let e = layer
        .w
        .transpose_with(&c.ev, &layout, &TransposeKeys::Reuse(vec![16]))
        .unwrap_err();
    assert!(matches!(e, HekitError::InvalidParams(_)), "{e:?}");
    // Another batch or slot count.
    for other in [
        Layout::new(2048, 16, 4).unwrap(),
        Layout::new(4096, 16, 8).unwrap(),
    ] {
        let e = layer.w.transpose(&c.ev, &other).unwrap_err();
        assert!(matches!(e, HekitError::InvalidParams(_)), "{e:?}");
    }
    assert_eq!(c.ev.counter(), zero);

    // Backward with weights that are not the transpose (W itself, 4 x 16): InvalidParams.
    let dm = rand_mat(&mut r, 8, d_out);
    let dp = pack_activations(&dm, &layout);
    let dct = c
        .enc
        .encrypt(&c.encoder.encode(&dp, scale, top - 1).unwrap(), &mut r);
    let e = layer
        .backward_input(&c.ev, &dct, &layer.w, &layout)
        .unwrap_err();
    assert!(matches!(e, HekitError::InvalidParams(_)), "{e:?}");
    assert_eq!(c.ev.counter(), zero);

    // A transpose with full keys; then W^T below delta's level: LevelMismatch.
    let c = ctx(small_params(), &union(&t_steps, &b_steps), 19);
    let layer = Dense::encrypt(&w, &bias, &layout, scale, &c.enc, &c.encoder, &mut r).unwrap();
    let wt = layer.w.transpose(&c.ev, &layout).unwrap();
    let low = EncWeights::from_diagonals(
        wt.diagonals().mod_switch_to_level(&c.ev, 1).unwrap(),
        d_in,
        d_out,
        &layout,
    )
    .unwrap();
    let dct = c
        .enc
        .encrypt(&c.encoder.encode(&dp, scale, 2).unwrap(), &mut r);
    c.ev.reset_counter();
    let e = layer
        .backward_input(&c.ev, &dct, &low, &layout)
        .unwrap_err();
    assert!(matches!(e, HekitError::LevelMismatch), "{e:?}");
    assert_eq!(c.ev.counter(), zero);
    // W^T above delta's level is switched down.
    let v = layer.backward_input(&c.ev, &dct, &wt, &layout).unwrap();
    assert_eq!(v.level(), 1);
    let err = max_abs_diff(
        &c.encoder.decode(&c.dec.decrypt(&v)),
        &pack_activations(&dm.dot(&w), &layout),
    );
    assert!(err <= TOL_B, "small backward error {err:e}");
}
