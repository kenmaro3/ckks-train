//! Task 25: slot layout, rotation-only transpose and the plaintext simulator
//! (`docs/layout.md`). Every check is against a dense ndarray reference, fixed seeds.

use std::sync::Arc;

use hekit_ckks::linalg::{Diagonals, bsgs_split, diagonals_from_matrix};
use hekit_ckks::{CkksParams, Encoder, Plaintext};
use hekit_core::sampler::Rng;
use hekit_nn::layout::{
    LayerShape, Layout, pack_activations, pack_bias, pack_weight_diagonals,
    rotations_for_transpose, transpose_diagonals, transpose_galois_steps, transpose_sources,
    unpack_activations, unpack_weight_diagonals,
};
use hekit_nn::sim::{
    SimCounter, SimLayer, lane0_mask, sim_backward_input, sim_batch_reduce, sim_batchsum,
    sim_dense_forward, sim_fold, sim_matvec, sim_outer_batchsum, sim_rotate, sim_rotate_counted,
    sim_train_step_sgd, sim_transpose, valid_lane_mask,
};
use ndarray::Array2;

const TOL: f64 = 1e-9;
const SHAPES: [(usize, usize); 3] = [(16, 16), (16, 64), (64, 16)];
const BATCHES: [usize; 2] = [1, 8];

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

fn assert_mat_close(got: &Array2<f64>, want: &Array2<f64>, tol: f64, what: &str) {
    assert_eq!(got.dim(), want.dim(), "{what}: shape");
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let err = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        err <= tol * scale,
        "{what}: max error {err:e} (scale {scale})"
    );
}

fn assert_diags_close(got: &[Vec<f64>], want: &[Vec<f64>], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: number of diagonals");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = max_abs_diff(g, w);
        assert!(e <= tol, "{what}: diagonal {i} error {e:e}");
    }
}

/// The layouts of the brief's grid: d = 64, two replicas of the block (and one single-block
/// layout, `d * batch == slots`).
fn layouts(batch: usize) -> Vec<Layout> {
    vec![
        Layout::new(64 * batch * 2, 64, batch).unwrap(),
        Layout::new(64 * batch, 64, batch).unwrap(),
    ]
}

/// M1's BSGS pre-rotation as documented in `hekit_ckks::linalg`:
/// `pre[k1 * baby + k2] = rot(diag[g k1 + k2], -g k1 stride)`.
fn m1_prerotate(diags: &[Vec<f64>], stride: usize) -> Vec<Vec<f64>> {
    let (baby, giant) = bsgs_split(diags.len());
    let slots = diags[0].len();
    let mut out = Vec::new();
    for k1 in 0..giant {
        let shift = (baby * k1 * stride) % slots;
        for k2 in 0..baby {
            let v = &diags[baby * k1 + k2];
            out.push((0..slots).map(|j| v[(j + slots - shift) % slots]).collect());
        }
    }
    out
}

#[test]
fn layout_checks_its_parameters() {
    assert!(Layout::new(128, 16, 8).is_ok());
    assert!(Layout::new(1 << 14, 16, 8).is_ok()); // replicated blocks
    assert!(Layout::new(128, 12, 8).is_err());
    assert!(Layout::new(128, 16, 6).is_err());
    assert!(Layout::new(64, 16, 8).is_err());
    let l = Layout::new(256, 16, 8).unwrap();
    assert_eq!(
        (l.slot(3, 5), l.block(), l.features(), l.log_batch()),
        (29, 128, 32, 3)
    );
}

#[test]
fn rotation_convention_and_counting() {
    let v: Vec<f64> = (0..8).map(f64::from).collect();
    assert_eq!(sim_rotate(&v, 3), vec![3., 4., 5., 6., 7., 0., 1., 2.]);
    assert_eq!(sim_rotate(&v, -1), vec![7., 0., 1., 2., 3., 4., 5., 6.]);
    let mut cnt = SimCounter::default();
    let _ = sim_rotate_counted(&v, 8, &mut cnt); // multiple of slots: a copy
    let _ = sim_rotate_counted(&v, 0, &mut cnt);
    assert_eq!(cnt.rotations, 0);
    let _ = sim_rotate_counted(&v, -3, &mut cnt);
    assert_eq!(cnt.rotations, 1);
}

#[test]
fn pack_is_canonical_and_unpacks() {
    let mut r = rng(1);
    let layout = Layout::new(256, 16, 8).unwrap();
    let x = rand_mat(&mut r, 6, 3, 1.0); // 6 of 8 lanes, 3 of n = 4 features
    let v = pack_activations(&x, &layout);
    assert_eq!(v.len(), layout.slots);
    for (s, &got) in v.iter().enumerate() {
        let (j, b) = ((s / 8) % 4, s % 8);
        let want = if b < 6 && j < 3 { x[[b, j]] } else { 0.0 };
        assert_eq!(got, want, "slot {s}");
    }
    let u = unpack_activations(&v, &layout, 3);
    assert_eq!(u.slice(ndarray::s![..6, ..]), x);
    let bias = pack_bias(&[1.0, 2.0, 3.0], &layout);
    assert_eq!(bias[layout.slot(2, 7)], 3.0);
    assert_eq!(bias[layout.slot(3, 0)], 0.0);
    assert_eq!(bias[layout.slot(6, 4)], 3.0); // period 4
}

#[test]
fn square_diagonals_equal_m1() {
    let mut r = rng(2);
    for &batch in &BATCHES {
        for layout in layouts(batch) {
            for n in [16usize, 64] {
                let w = rand_mat(&mut r, n, n, 1.0);
                let ours = pack_weight_diagonals(&w, &layout, bsgs_split(n));
                let m1 = m1_prerotate(&diagonals_from_matrix(&w, layout.slots, batch), batch);
                // Bitwise: both read the same matrix entries.
                assert_eq!(
                    ours, m1,
                    "n = {n}, batch = {batch}, slots = {}",
                    layout.slots
                );
            }
        }
    }
}

/// Against M1's own pre-rotation (`Diagonals::<Plaintext>::encode`), decoded: seal_2020 slots.
#[test]
fn square_diagonals_equal_m1_encoded() {
    let params = Arc::new(CkksParams::seal_2020());
    let encoder = Encoder::new(params.clone());
    let layout = Layout::new(params.slots(), 16, 8).unwrap();
    let w = rand_mat(&mut rng(3), 16, 16, 1.0);
    let raw = diagonals_from_matrix(&w, layout.slots, 8);
    let encoded =
        Diagonals::<Plaintext>::encode(&encoder, &raw, 8, params.scale(), params.max_level())
            .unwrap();
    let ours = pack_weight_diagonals(&w, &layout, bsgs_split(16));
    for (i, (pt, v)) in encoded.entries().iter().zip(&ours).enumerate() {
        let e = max_abs_diff(&encoder.decode(pt), v);
        assert!(e < 1e-8, "entry {i}: {e:e}");
    }
}

#[test]
fn matvec_and_fold_give_x_wt() {
    let mut r = rng(4);
    for &(d_out, d_in) in &SHAPES {
        for &batch in &BATCHES {
            for layout in layouts(batch) {
                let sh = LayerShape::new(d_out, d_in, &layout);
                let w = rand_mat(&mut r, d_out, d_in, 1.0);
                let x = rand_mat(&mut r, batch, d_in, 1.0);
                let diags = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
                let mut cnt = SimCounter::default();
                let y = sim_matvec(&diags, &pack_activations(&x, &layout), &layout, &mut cnt);
                let y = sim_fold(&y, d_out, d_in, &layout, &mut cnt);
                let want = x.dot(&w.t());
                let what = format!("({d_out},{d_in}) batch {batch} slots {}", layout.slots);
                assert_mat_close(&unpack_activations(&y, &layout, d_out), &want, TOL, &what);
                // The output is in canonical form over all slots (ready for the next layer).
                let e = max_abs_diff(&y, &pack_activations(&want, &layout));
                assert!(e <= TOL * 64.0, "{what}: canonical form {e:e}");
                let fold = sh.fold_steps() as u64;
                assert_eq!(
                    cnt.rotations,
                    (sh.baby + sh.giant - 2) as u64 + fold,
                    "{what}"
                );
                assert_eq!(cnt.ct_ct_tensor, sh.m as u64, "{what}");
                assert_eq!(cnt.relins, sh.giant as u64, "{what}");
                assert_eq!((cnt.ct_pt_mults, cnt.rescales), (0, 0), "{what}");
            }
        }
    }
}

#[test]
fn dense_forward_adds_bias_and_rescales() {
    let mut r = rng(5);
    let layout = Layout::new(1024, 64, 8).unwrap();
    let (d_out, d_in) = (10, 50); // non-powers of two: padded to 16 x 64
    let sh = LayerShape::new(d_out, d_in, &layout);
    let w = rand_mat(&mut r, d_out, d_in, 1.0);
    let b: Vec<f64> = (0..d_out).map(|_| uni(&mut r)).collect();
    let x = rand_mat(&mut r, 8, d_in, 1.0);
    let mut cnt = SimCounter::default();
    let z = sim_dense_forward(
        &pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant)),
        &pack_bias(&b, &layout),
        &pack_activations(&x, &layout),
        d_out,
        d_in,
        &layout,
        &mut cnt,
    );
    let mut want = x.dot(&w.t());
    want.rows_mut().into_iter().for_each(|mut row| {
        row.iter_mut().zip(&b).for_each(|(v, bb)| *v += bb);
    });
    assert_mat_close(&unpack_activations(&z, &layout, d_out), &want, TOL, "dense");
    assert_eq!(cnt.rescales, 1);
    assert_eq!(cnt.rotations, (sh.baby + sh.giant - 2 + 2) as u64);
}

#[test]
fn transpose_is_rotation_only_and_exact() {
    let mut r = rng(6);
    let extra = [(8, 4), (2, 8), (10, 32), (32, 50)];
    for &(d_out, d_in) in SHAPES.iter().chain(&extra) {
        for &batch in &BATCHES {
            for layout in layouts(batch) {
                let sh = LayerShape::new(d_out, d_in, &layout);
                let w = rand_mat(&mut r, d_out, d_in, 1.0);
                let diags = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
                let wt = w.t().to_owned();
                let want = pack_weight_diagonals(&wt, &layout, (sh.baby, sh.giant));
                let what = format!("({d_out},{d_in}) batch {batch} slots {}", layout.slots);
                // Element for element (pure slot permutations of the same values).
                assert_eq!(
                    transpose_diagonals(&diags, d_out, d_in, &layout),
                    want,
                    "{what}"
                );
                let mut cnt = SimCounter::default();
                let t = sim_transpose(&diags, d_out, d_in, &layout, &mut cnt);
                assert_eq!(t, want, "{what}");
                // Rotations only, m - 1 of them, no products of any kind.
                assert_eq!(cnt.rotations, (sh.m - 1) as u64, "{what}");
                assert_eq!(
                    (cnt.relins, cnt.ct_ct_tensor, cnt.ct_pt_mults, cnt.rescales),
                    (0, 0, 0, 0),
                    "{what}"
                );
                let steps = rotations_for_transpose(d_out, d_in, &layout);
                assert_eq!(
                    steps.iter().filter(|&&s| s != 0).count(),
                    sh.m - 1,
                    "{what}"
                );
                assert!(steps.iter().all(|&s| s >= 0 && (s as usize) < sh.l * batch));
                let keys = transpose_galois_steps(d_out, d_in, &layout);
                assert!(keys.windows(2).all(|p| p[0] < p[1]) && !keys.contains(&0));
                let src = transpose_sources(d_out, d_in, &layout);
                let mut sorted = src.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..sh.m).collect::<Vec<_>>(), "{what}: permutation");
                // Transposing back gives W.
                assert_eq!(
                    transpose_diagonals(&t, d_in, d_out, &layout),
                    diags,
                    "{what}"
                );
            }
        }
    }
}

#[test]
fn backward_input_gives_delta_w() {
    let mut r = rng(7);
    for &(d_out, d_in) in &SHAPES {
        for &batch in &BATCHES {
            let layout = Layout::new(64 * batch * 2, 64, batch).unwrap();
            let sh = LayerShape::new(d_out, d_in, &layout);
            let w = rand_mat(&mut r, d_out, d_in, 1.0);
            let delta = rand_mat(&mut r, batch, d_out, 1.0);
            let diags = pack_weight_diagonals(&w, &layout, (sh.baby, sh.giant));
            let wt = transpose_diagonals(&diags, d_out, d_in, &layout);
            let mut cnt = SimCounter::default();
            let v = sim_backward_input(
                &wt,
                &pack_activations(&delta, &layout),
                d_out,
                d_in,
                &layout,
                &mut cnt,
            );
            let what = format!("({d_out},{d_in}) batch {batch}");
            assert_mat_close(
                &unpack_activations(&v, &layout, d_in),
                &delta.dot(&w),
                TOL,
                &what,
            );
            let fold = sh.transposed().fold_steps();
            assert_eq!(
                cnt.rotations,
                (sh.baby + sh.giant - 2 + fold) as u64,
                "{what}"
            );
            assert_eq!(cnt.rescales, 1);
        }
    }
}

#[test]
fn outer_batchsum_gives_delta_t_a() {
    let mut r = rng(8);
    for &(d_out, d_in) in &SHAPES {
        for &batch in &BATCHES {
            for layout in layouts(batch) {
                let sh = LayerShape::new(d_out, d_in, &layout);
                let delta = rand_mat(&mut r, batch, d_out, 1.0);
                let a = rand_mat(&mut r, batch, d_in, 1.0);
                let mut cnt = SimCounter::default();
                let g = sim_outer_batchsum(
                    &pack_activations(&delta, &layout),
                    &pack_activations(&a, &layout),
                    d_out,
                    d_in,
                    &layout,
                    1.0,
                    &mut cnt,
                );
                let dw = delta.t().dot(&a);
                let want = pack_weight_diagonals(&dw, &layout, (sh.baby, sh.giant));
                let what = format!("({d_out},{d_in}) batch {batch} slots {}", layout.slots);
                // Replicated in every lane and every block: equal to the packed dW itself.
                assert_diags_close(&g, &want, TOL * batch as f64, &what);
                assert_mat_close(
                    &unpack_weight_diagonals(&g, d_out, d_in, &layout, batch - 1),
                    &dw,
                    TOL,
                    &what,
                );
                let lb = layout.log_batch();
                let rot = (sh.baby - 1) + (sh.giant - 1) + 2 * sh.m * lb;
                assert_eq!(cnt.rotations, rot as u64, "{what}");
                assert_eq!(cnt.ct_ct_tensor, sh.m as u64, "{what}");
                assert_eq!(cnt.relins, sh.m as u64, "{what}");
                assert_eq!(cnt.ct_pt_mults, sh.m as u64, "{what}");
                assert_eq!(cnt.rescales, 2 * sh.m as u64, "{what}");
            }
        }
    }
}

#[test]
fn batch_reduce_and_batchsum() {
    let mut r = rng(9);
    let layout = Layout::new(512, 16, 8).unwrap();
    let dz = rand_mat(&mut r, 8, 16, 1.0);
    let v = pack_activations(&dz, &layout);
    let sums: Vec<f64> = (0..16).map(|j| dz.column(j).sum()).collect();
    let mut cnt = SimCounter::default();
    let red = sim_batch_reduce(&v, &layout, &mut cnt);
    assert_eq!(cnt.rotations, 3);
    for (j, s) in sums.iter().enumerate() {
        assert!(
            (red[layout.slot(j, 0)] - s).abs() < TOL,
            "lane 0 of feature {j}"
        );
    }
    let mut cnt = SimCounter::default();
    let bs = sim_batchsum(&v, &layout, 0.25, &mut cnt);
    let scaled: Vec<f64> = sums.iter().map(|s| 0.25 * s).collect();
    assert!(max_abs_diff(&bs, &pack_bias(&scaled, &layout)) < TOL);
    assert_eq!((cnt.rotations, cnt.ct_pt_mults, cnt.rescales), (6, 1, 1));
    // Masks.
    let m0 = lane0_mask(&layout);
    assert_eq!(m0.iter().sum::<f64>(), 64.0);
    assert_eq!((m0[0], m0[1], m0[8]), (1.0, 0.0, 1.0));
    let mv = valid_lane_mask(&layout, 6);
    assert_eq!((mv[5], mv[6], mv[13], mv[14]), (1.0, 0.0, 1.0, 0.0));
}

/// Dense reference of one SGD step: Square activations, MSE gradient `a_L - y`,
/// `dW = dz^T A / n`, `db = sum_b dz / n`.
fn reference_step(layers: &[SimLayer], x: &Array2<f64>, y: &Array2<f64>, lr: f64) -> Vec<SimLayer> {
    let n = x.nrows() as f64;
    let mut acts = vec![x.clone()];
    let mut zs = Vec::new();
    for l in layers {
        let mut z = acts.last().unwrap().dot(&l.w.t());
        for mut row in z.rows_mut() {
            row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
        }
        acts.push(z.mapv(|v| v * v));
        zs.push(z);
    }
    let mut delta = acts.last().unwrap() - y;
    let mut out = layers.to_vec();
    for i in (0..layers.len()).rev() {
        let dz = &delta * &zs[i].mapv(|v| 2.0 * v);
        let g = dz.t().dot(&acts[i]) / n;
        let gb = dz.sum_axis(ndarray::Axis(0)) / n;
        if i > 0 {
            delta = dz.dot(&layers[i].w);
        }
        out[i].w = &layers[i].w - &(lr * &g);
        out[i].b = layers[i]
            .b
            .iter()
            .zip(&gb)
            .map(|(b, g)| b - lr * g)
            .collect();
    }
    out
}

fn random_net(r: &mut Rng, dims: &[usize]) -> Vec<SimLayer> {
    dims.windows(2)
        .map(|p| SimLayer {
            w: rand_mat(r, p[1], p[0], 0.8 / (p[0] as f64).sqrt()),
            b: (0..p[1]).map(|_| 0.2 * uni(r)).collect(),
        })
        .collect()
}

#[test]
fn train_step_matches_dense_reference() {
    let mut r = rng(10);
    // (dims, valid samples, layout d, lanes): the Task 30 net, a padded batch with
    // non-power-of-two dims (Task 31 style), and wide / tall layers.
    let cases: [(&[usize], usize, usize, usize); 4] = [
        (&[4, 8, 2], 8, 8, 8),
        (&[3, 5, 2], 6, 8, 8),
        (&[16, 64, 16], 8, 64, 8),
        (&[64, 16, 8, 4], 5, 64, 8),
    ];
    for (dims, n_valid, d, batch) in cases {
        let layout = Layout::new(d * batch * 2, d, batch).unwrap();
        let net = random_net(&mut r, dims);
        let x = rand_mat(&mut r, n_valid, dims[0], 1.0);
        let y = rand_mat(&mut r, n_valid, *dims.last().unwrap(), 1.0);
        let step = sim_train_step_sgd(&net, &x, &y, 0.1, &layout);
        let want = reference_step(&net, &x, &y, 0.1);
        for (i, (got, w)) in step.layers.iter().zip(&want).enumerate() {
            let what = format!("{dims:?} layer {i}");
            assert_mat_close(&got.w, &w.w, TOL, &what);
            let b = Array2::from_shape_vec((1, w.b.len()), got.b.clone()).unwrap();
            let wb = Array2::from_shape_vec((1, w.b.len()), w.b.clone()).unwrap();
            assert_mat_close(&b, &wb, TOL, &format!("{what} bias"));
            // The updated diagonals are replicated in every lane: the packed W' itself.
            let sh = LayerShape::new(w.w.nrows(), w.w.ncols(), &layout);
            let packed = pack_weight_diagonals(&w.w, &layout, (sh.baby, sh.giant));
            assert_diags_close(&step.weight_diags[i], &packed, 1e-8, &what);
            assert!(max_abs_diff(&step.bias_slots[i], &pack_bias(&w.b, &layout)) < 1e-8);
        }
    }
}

/// The level table of `docs/layout.md` §4 (Task 30 network on seal_2020, 9 levels).
#[test]
fn train_step_level_table() {
    let mut r = rng(11);
    let layout = Layout::new(1 << 14, 8, 8).unwrap();
    let net = random_net(&mut r, &[4, 8, 2]);
    let x = rand_mat(&mut r, 8, 4, 1.0);
    let y = rand_mat(&mut r, 8, 2, 1.0);
    let step = sim_train_step_sgd(&net, &x, &y, 0.1, &layout);
    let table = [
        ("z1", 1),
        ("a1", 2),
        ("z2", 3),
        ("a2", 4),
        ("delta2", 4),
        ("s2", 4),
        ("dz2", 5),
        ("delta1", 6),
        ("outer2", 6),
        ("W2'", 7),
        ("b2'", 6),
        ("s1", 2),
        ("dz1", 7),
        ("outer1", 8),
        ("W1'", 9),
        ("b1'", 8),
    ];
    for (label, depth) in table {
        assert_eq!(step.depth_of(label), depth, "{label}");
    }
    assert_eq!(step.depth, 9, "fits seal_2020's 9 levels exactly");
    // Op counts of the whole step (W1: 8 x 4 tall, m = 4, 2 x 2; W2: 2 x 8 wide, m = 2, 2 x 1;
    // lb = 3):
    // forward 2 + (1 + 2 fold) = 5; transpose W2: 1; backward W2^T (8 x 2, tall): 1;
    // gradients: W2 (1 + 0 + 2*2*3) = 13, W1 (1 + 1 + 2*4*3) = 26; biases 2 * 6 = 12.
    assert_eq!(step.count.rotations, 5 + 1 + 1 + 13 + 26 + 12);
    // Relins: forward matvecs (giant) 2 + 1; W2^T matvec (m = 2, giant 1) 1; squares 2;
    // dz products 2; outer products m = 2 + 4.
    assert_eq!(step.count.relins, (2 + 1) + 1 + 2 + 2 + (2 + 4));
    // Deeper nets: 4L + 1.
    let net3 = random_net(&mut r, &[4, 8, 8, 2]);
    let s3 = sim_train_step_sgd(&net3, &x, &y, 0.1, &layout);
    assert_eq!(s3.depth, 13);
}

/// Research-lead ruling item 1: a slot-domain 0/1 mask encoded at scale 1 is rounded away
/// unless it is constant (its constant coefficient is the slot mean, 2k/N).
#[test]
fn masks_at_scale_one_are_destroyed() {
    let params = Arc::new(CkksParams::seal_2020());
    let encoder = Encoder::new(params.clone());
    let layout = Layout::new(params.slots(), 16, 8).unwrap();
    let err_at_scale_one = |mask: &[f64]| {
        let pt = encoder.encode(mask, 1.0, 0).unwrap();
        max_abs_diff(&encoder.decode(&pt), mask)
    };
    let lane0 = lane0_mask(&layout);
    let valid = valid_lane_mask(&layout, 6);
    let ones = vec![1.0; params.slots()];
    assert!(err_at_scale_one(&lane0) > 0.5, "lane-0 mask survives?");
    assert!(err_at_scale_one(&valid) > 0.5, "valid-lane mask survives?");
    assert!(err_at_scale_one(&ones) < 1e-9, "constant mask is integral");
}
