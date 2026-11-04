//! Plaintext slot simulator of the encrypted training ops (`docs/layout.md`).
//!
//! A "ciphertext" is a `Vec<f64>` of `layout.slots` values. Every `sim_*` function runs the
//! same op sequence as the encrypted code of Tasks 26–30 will run: rotations are cyclic
//! shifts ([`sim_rotate`]: `rot(v, k)[s] = v[(s + k) mod slots]`, as `Evaluator::rotate`) and
//! products are slot-wise. Each function counts what it does in a [`SimCounter`] with the
//! semantics of `hekit_ckks::OpCounter`. So a test can assert that the encrypted counter equals
//! the simulator's ([`SimCounter::to_op_counter`]).
//!
//! Counting rules (as `OpCounter`):
//! - A rotation counts 1 unless its step is a multiple of `slots` (a copy). Hoisted rotations
//!   count 1 per step.
//! - A ct-ct product is 1 tensor. `relins` counts each relinearisation (`sim_mul` relinearises
//!   once; the matvec relinearises once per giant step, M1's lazy relin).
//! - `ct_pt_mults` counts plaintext and scalar products.
//! - `rescales` counts the rescales the encrypted function itself performs. The matvec returns
//!   its result unrescaled, like `matvec_enc`.
//!
//! Additions, subtractions and level switches are free and not counted.

use hekit_ckks::OpCounter;
use hekit_ckks::linalg::bsgs_split;
use ndarray::Array2;

use crate::layout::{
    LayerShape, Layout, pack_activations, pack_bias, pack_weight_diagonals,
    rotations_for_transpose, transpose_sources, unpack_activations, unpack_weight_diagonals,
};

/// Simulated operation counts (the fields of `hekit_ckks::OpCounter` except bootstraps).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SimCounter {
    /// Key-switching rotations (trivial rotations excluded).
    pub rotations: u64,
    /// Relinearisations.
    pub relins: u64,
    /// Ciphertext-ciphertext tensor products.
    pub ct_ct_tensor: u64,
    /// Ciphertext-plaintext (incl. scalar) multiplications.
    pub ct_pt_mults: u64,
    /// Rescales.
    pub rescales: u64,
}

impl SimCounter {
    /// The same counts as an `OpCounter` (`bootstraps = 0`), to compare with the evaluator.
    pub fn to_op_counter(&self) -> OpCounter {
        OpCounter {
            rotations: self.rotations,
            relins: self.relins,
            ct_ct_tensor: self.ct_ct_tensor,
            ct_pt_mults: self.ct_pt_mults,
            rescales: self.rescales,
            bootstraps: 0,
        }
    }

    /// Key switches: `rotations + relins`.
    pub fn key_switches(&self) -> u64 {
        self.rotations + self.relins
    }
}

/// `v` rotated left by `steps` (negative: right): `out[s] = v[(s + steps) mod len]`. Not
/// counted; see [`sim_rotate_counted`].
pub fn sim_rotate(v: &[f64], steps: i32) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let k = (steps as i64).rem_euclid(n as i64) as usize;
    let mut out = Vec::with_capacity(n);
    out.extend_from_slice(&v[k..]);
    out.extend_from_slice(&v[..k]);
    out
}

/// [`sim_rotate`], counting one rotation unless `steps` is a multiple of the length (a copy, as
/// in `Evaluator::rotate`).
pub fn sim_rotate_counted(v: &[f64], steps: i32, cnt: &mut SimCounter) -> Vec<f64> {
    if (steps as i64).rem_euclid(v.len().max(1) as i64) != 0 {
        cnt.rotations += 1;
    }
    sim_rotate(v, steps)
}

/// Slot-wise `a + b` (free).
///
/// # Panics
/// If the lengths differ.
pub fn sim_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Slot-wise `a - b` (free).
///
/// # Panics
/// If the lengths differ.
pub fn sim_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Slot-wise product without counting (shared by the counted ops).
fn hadamard(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

/// `Evaluator::mul` followed by `rescale`: slot-wise `a * b`. Counts 1 tensor, 1 relin and
/// 1 rescale (1 level).
pub fn sim_mul(a: &[f64], b: &[f64], cnt: &mut SimCounter) -> Vec<f64> {
    cnt.ct_ct_tensor += 1;
    cnt.relins += 1;
    cnt.rescales += 1;
    hadamard(a, b)
}

/// `mul_plain` (or `mul_scalar`) by the plaintext slot vector `p`, then `rescale`. Counts
/// 1 ct-pt and 1 rescale (1 level).
pub fn sim_mul_plain(a: &[f64], p: &[f64], cnt: &mut SimCounter) -> Vec<f64> {
    cnt.ct_pt_mults += 1;
    cnt.rescales += 1;
    hadamard(a, p)
}

/// M1's `Evaluator::matvec_enc` on pre-rotated diagonals with `d = diags.len()` and
/// `stride = layout.batch`:
/// `sum_{k1} rot(relin(sum_{k2} diags[k1 baby + k2] * rot(x, k2 batch)), baby k1 batch)`.
/// Counts `baby - 1` hoisted and `giant - 1` giant rotations, `d` tensors and `giant` relins.
/// The result is not rescaled.
///
/// # Panics
/// If `diags.len()` is not a power of two or a length is not `layout.slots`.
pub fn sim_matvec(
    diags: &[Vec<f64>],
    x: &[f64],
    layout: &Layout,
    cnt: &mut SimCounter,
) -> Vec<f64> {
    assert_eq!(x.len(), layout.slots, "x length must be the slot count");
    let (baby, giant) = bsgs_split(diags.len());
    let s = layout.batch;
    // Hoisted baby steps of x (step 0 is a copy).
    let xs: Vec<Vec<f64>> = (0..baby)
        .map(|k2| sim_rotate_counted(x, (k2 * s) as i32, cnt))
        .collect();
    let mut acc = vec![0.0; layout.slots];
    for k1 in 0..giant {
        let mut inner = vec![0.0; layout.slots];
        for (k2, xr) in xs.iter().enumerate() {
            let w = &diags[k1 * baby + k2];
            assert_eq!(
                w.len(),
                layout.slots,
                "diagonal length must be the slot count"
            );
            cnt.ct_ct_tensor += 1;
            inner = sim_add(&inner, &hadamard(w, xr));
        }
        cnt.relins += 1;
        let block = sim_rotate_counted(&inner, (baby * k1 * s) as i32, cnt);
        acc = sim_add(&acc, &block);
    }
    acc
}

/// The fold of a wide layer (`n_in > n_out`) after the matvec:
/// `for t < log2(n_in / n_out): y = y + rot(y, n_out 2^t batch)`. Rotations and additions only.
/// The result is the canonical `n_out`-periodic output. A no-op for square or tall layers.
pub fn sim_fold(
    y: &[f64],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    cnt: &mut SimCounter,
) -> Vec<f64> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let mut y = y.to_vec();
    for t in 0..sh.fold_steps() {
        let r = sim_rotate_counted(&y, (sh.n_out * (1 << t) * layout.batch) as i32, cnt);
        y = sim_add(&y, &r);
    }
    y
}

/// Task 26's dense forward: `rescale(fold(matvec(diags, x)) + bias)` for `W` (`d_out x d_in`).
/// `bias` is packed in canonical form ([`pack_bias`]). Counts the matvec, the fold and
/// 1 rescale (1 level).
pub fn sim_dense_forward(
    diags: &[Vec<f64>],
    bias: &[f64],
    x: &[f64],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    cnt: &mut SimCounter,
) -> Vec<f64> {
    let y = sim_matvec(diags, x, layout, cnt);
    let y = sim_add(&sim_fold(&y, d_out, d_in, layout, cnt), bias);
    cnt.rescales += 1;
    y
}

/// Task 27's backward input `delta W` (lane-wise `W^T delta`) for `W` (`d_out x d_in`), from
/// the transposed diagonals `wt` ([`sim_transpose`]):
/// `rescale(fold_{W^T}(matvec(wt, delta)))`. The fold applies when `n_out > n_in`. The output
/// is canonical with period `n_in`. Counts the matvec, the fold and 1 rescale.
pub fn sim_backward_input(
    wt: &[Vec<f64>],
    delta: &[f64],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    cnt: &mut SimCounter,
) -> Vec<f64> {
    let y = sim_matvec(wt, delta, layout, cnt);
    let y = sim_fold(&y, d_in, d_out, layout, cnt);
    cnt.rescales += 1;
    y
}

/// Task 27's encrypted transpose, by rotations only:
/// `out[i] = rot(diags[transpose_sources[i]], rotations_for_transpose[i])`. Counts the nonzero
/// rotations (`m - 1`), with no products, relins or levels.
///
/// # Panics
/// If `diags.len() != m` or a diagonal is not `slots` long.
pub fn sim_transpose(
    diags: &[Vec<f64>],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    cnt: &mut SimCounter,
) -> Vec<Vec<f64>> {
    let sh = LayerShape::new(d_out, d_in, layout);
    assert_eq!(diags.len(), sh.m, "expected {} diagonals", sh.m);
    transpose_sources(d_out, d_in, layout)
        .into_iter()
        .zip(rotations_for_transpose(d_out, d_in, layout))
        .map(|(c, step)| {
            assert_eq!(
                diags[c].len(),
                layout.slots,
                "diagonal length must be the slot count"
            );
            sim_rotate_counted(&diags[c], step, cnt)
        })
        .collect()
}

/// The lane-0 mask: 1 in slots `s` with `s mod batch == 0`, else 0. It is not integral in
/// coefficient form (`docs/layout.md` §3.1), so it is encoded at an exact scale and costs the
/// level of the constant it is folded with.
pub fn lane0_mask(layout: &Layout) -> Vec<f64> {
    valid_lane_mask(layout, 1)
}

/// The valid-lane mask for `n_valid <= batch` real samples: 1 in lanes `b < n_valid`.
///
/// # Panics
/// If `n_valid > layout.batch`.
pub fn valid_lane_mask(layout: &Layout, n_valid: usize) -> Vec<f64> {
    assert!(
        n_valid <= layout.batch,
        "n_valid {n_valid} > batch {}",
        layout.batch
    );
    (0..layout.slots)
        .map(|s| if s % layout.batch < n_valid { 1.0 } else { 0.0 })
        .collect()
}

/// Batch reduction: `for t < log2(batch): v = v + rot(v, 2^t)`. Lane 0 of each feature then
/// holds the sum over the `batch` lanes. The other lanes hold partial sums that mix adjacent
/// features. Counts `log2(batch)` rotations and no level.
pub fn sim_batch_reduce(v: &[f64], layout: &Layout, cnt: &mut SimCounter) -> Vec<f64> {
    let mut v = v.to_vec();
    for t in 0..layout.log_batch() {
        let r = sim_rotate_counted(&v, 1 << t, cnt);
        v = sim_add(&v, &r);
    }
    v
}

/// Replication of lane 0 to every lane, `for t < log2(batch): v = v + rot(v, -2^t)`. The other
/// lanes must be 0 on input (after [`lane0_mask`]). Counts `log2(batch)` rotations.
pub fn sim_batch_replicate(v: &[f64], layout: &Layout, cnt: &mut SimCounter) -> Vec<f64> {
    let mut v = v.to_vec();
    for t in 0..layout.log_batch() {
        let r = sim_rotate_counted(&v, -(1 << t), cnt);
        v = sim_add(&v, &r);
    }
    v
}

/// Reduce, multiply by `c * lane0_mask` (the constant folded into the mask: 1 ct-pt and
/// 1 rescale), then replicate. The result holds `c * sum_b v_b[j]` in every lane of feature
/// `j`. This is the bias gradient (Task 28 `bias_grad` with `c = lr / n` or `1 / n`). Counts
/// `2 log2(batch)` rotations and 1 level.
pub fn sim_batchsum(v: &[f64], layout: &Layout, c: f64, cnt: &mut SimCounter) -> Vec<f64> {
    let red = sim_batch_reduce(v, layout, cnt);
    let mask: Vec<f64> = lane0_mask(layout).into_iter().map(|m| c * m).collect();
    let masked = sim_mul_plain(&red, &mask, cnt);
    sim_batch_replicate(&masked, layout, cnt)
}

/// The weight gradient `c * delta^T A` for `W` (`d_out x d_in`) as pre-rotated diagonals, in
/// the layout of [`pack_weight_diagonals`] (replicated in every lane). `delta` is canonical with
/// period `n_out`, `a` with period `n_in`. For `c = g k1 + k2`:
/// `entry = batchsum(rot(delta, -g k1 batch) * rot(a, k2 batch))`, run as:
/// 1. `baby - 1` hoisted rotations of `a` and `giant - 1` rotations of `delta`;
/// 2. per diagonal, 1 tensor, 1 relin and 1 rescale;
/// 3. [`sim_batchsum`] with the constant `c` folded into the lane-0 mask.
///
/// Counts `(baby - 1) + (giant - 1) + 2 m log2(batch)` rotations, `m` tensors, `m` relins,
/// `m` ct-pt and `2m` rescales (2 levels on the path).
pub fn sim_outer_batchsum(
    delta: &[f64],
    a: &[f64],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    c: f64,
    cnt: &mut SimCounter,
) -> Vec<Vec<f64>> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let s = layout.batch;
    assert_eq!(
        delta.len(),
        layout.slots,
        "delta length must be the slot count"
    );
    assert_eq!(a.len(), layout.slots, "a length must be the slot count");
    let a_rot: Vec<Vec<f64>> = (0..sh.baby)
        .map(|k2| sim_rotate_counted(a, (k2 * s) as i32, cnt))
        .collect();
    let d_rot: Vec<Vec<f64>> = (0..sh.giant)
        .map(|k1| sim_rotate_counted(delta, -((sh.baby * k1 * s) as i32), cnt))
        .collect();
    (0..sh.m)
        .map(|e| {
            let p = sim_mul(&d_rot[e / sh.baby], &a_rot[e % sh.baby], cnt);
            sim_batchsum(&p, layout, c, cnt)
        })
        .collect()
}

/// One dense layer of [`sim_train_step_sgd`]: `w` (`d_out x d_in`) and bias `b` (`d_out`).
#[derive(Debug, Clone, PartialEq)]
pub struct SimLayer {
    /// Weights, `d_out x d_in`.
    pub w: Array2<f64>,
    /// Bias, `d_out` values.
    pub b: Vec<f64>,
}

/// The result of [`sim_train_step_sgd`].
#[derive(Debug, Clone)]
pub struct SimStep {
    /// The updated layers, decoded from lane 0 of the slot vectors.
    pub layers: Vec<SimLayer>,
    /// The updated pre-rotated weight diagonals per layer (slot vectors).
    pub weight_diags: Vec<Vec<Vec<f64>>>,
    /// The updated packed biases per layer (slot vectors).
    pub bias_slots: Vec<Vec<f64>>,
    /// The forward output `a_L` (valid samples x `d_out` of the last layer).
    pub output: Array2<f64>,
    /// Ops of the whole step.
    pub count: SimCounter,
    /// `(label, depth)` in execution order. Depth is the levels consumed on the longest path
    /// from the fresh inputs. Labels (1-based layer `l`, `L` layers): `z{l}`, `a{l}`,
    /// `delta{L}` (loss gradient), `s{l}` (derivative factor), `dz{l}`, `delta{l-1}`,
    /// `outer{l}`, `W{l}'`, `b{l}'`.
    pub depths: Vec<(String, usize)>,
    /// The maximum depth: the levels one iteration needs (≤ `max_level` to fit without a
    /// mid-iteration refresh).
    pub depth: usize,
}

impl SimStep {
    /// The recorded depth of `label`.
    ///
    /// # Panics
    /// If `label` was not recorded.
    pub fn depth_of(&self, label: &str) -> usize {
        self.depths
            .iter()
            .find(|(l, _)| l == label)
            .unwrap_or_else(|| panic!("no value labelled {label}"))
            .1
    }
}

/// One SGD training step of an MLP with Square activations on every layer, MSE gradient
/// `a_L - y`, `dW = dz^T A / n`, `db = sum dz / n` (`n = x.nrows()` valid samples), in the
/// slot layout. This is the op sequence of Task 30's `train_batch` with every constant folded
/// (`docs/layout.md` §4):
/// - forward: [`sim_dense_forward`], then `sim_mul(z, z)`;
/// - loss gradient: `delta = a_L - y`;
/// - per layer, top down:
///   - `s = sim_mul_plain(z, 2 * valid)`, then `dz = sim_mul(delta, s)`;
///   - [`sim_transpose`] and [`sim_backward_input`] with the old `W` (not for layer 1);
///   - [`sim_outer_batchsum`] and [`sim_batchsum`] with `c = lr / n`;
///   - `W' = W - g`, `b' = b - g_b`.
///
/// # Panics
/// If the shapes do not chain, `x` and `y` row counts differ, or a dimension exceeds the
/// layout.
pub fn sim_train_step_sgd(
    layers: &[SimLayer],
    x: &Array2<f64>,
    y: &Array2<f64>,
    lr: f64,
    layout: &Layout,
) -> SimStep {
    assert!(!layers.is_empty(), "no layers");
    assert_eq!(x.nrows(), y.nrows(), "x and y sample counts differ");
    let n_valid = x.nrows();
    let c = lr / n_valid as f64;
    let mut cnt = SimCounter::default();
    let mut depths: Vec<(String, usize)> = Vec::new();
    let mut rec = |label: String, d: usize| depths.push((label, d));

    let mut diags: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut biases: Vec<Vec<f64>> = Vec::new();
    for l in layers {
        let sh = LayerShape::new(l.w.nrows(), l.w.ncols(), layout);
        assert_eq!(l.b.len(), sh.d_out, "bias length");
        diags.push(pack_weight_diagonals(&l.w, layout, (sh.baby, sh.giant)));
        biases.push(pack_bias(&l.b, layout));
    }

    // Forward. acts[i] is the input of layer i (acts[0] = x), with its depth.
    let mut acts: Vec<(Vec<f64>, usize)> = vec![(pack_activations(x, layout), 0)];
    let mut zs: Vec<(Vec<f64>, usize)> = Vec::new();
    for (i, l) in layers.iter().enumerate() {
        let (a, da) = acts.last().expect("nonempty");
        assert_eq!(
            l.w.ncols(),
            if i == 0 {
                x.ncols()
            } else {
                layers[i - 1].w.nrows()
            }
        );
        let (d_out, d_in) = l.w.dim();
        let z = sim_dense_forward(&diags[i], &biases[i], a, d_out, d_in, layout, &mut cnt);
        let dz = da + 1;
        rec(format!("z{}", i + 1), dz);
        let a_next = sim_mul(&z, &z, &mut cnt);
        rec(format!("a{}", i + 1), dz + 1);
        zs.push((z, dz));
        acts.push((a_next, dz + 1));
    }
    let n_layers = layers.len();
    let (a_out, d_aout) = acts.last().expect("nonempty").clone();
    assert_eq!(y.ncols(), layers[n_layers - 1].w.nrows(), "y width");
    let mut delta = sim_sub(&a_out, &pack_activations(y, layout));
    let mut d_delta = d_aout;
    rec(format!("delta{n_layers}"), d_delta);

    let two_valid: Vec<f64> = valid_lane_mask(layout, n_valid)
        .into_iter()
        .map(|m| 2.0 * m)
        .collect();
    let mut new_diags = diags.clone();
    let mut new_biases = biases.clone();
    for i in (0..n_layers).rev() {
        let (d_out, d_in) = layers[i].w.dim();
        let ln = i + 1;
        let (z, d_z) = &zs[i];
        let s = sim_mul_plain(z, &two_valid, &mut cnt);
        rec(format!("s{ln}"), d_z + 1);
        let dz = sim_mul(&delta, &s, &mut cnt);
        let d_dz = d_delta.max(d_z + 1) + 1;
        rec(format!("dz{ln}"), d_dz);

        let next = if i > 0 {
            let wt = sim_transpose(&diags[i], d_out, d_in, layout, &mut cnt);
            let v = sim_backward_input(&wt, &dz, d_out, d_in, layout, &mut cnt);
            rec(format!("delta{i}"), d_dz + 1);
            Some((v, d_dz + 1))
        } else {
            None
        };

        let (a_in, d_a) = &acts[i];
        let g = sim_outer_batchsum(&dz, a_in, d_out, d_in, layout, c, &mut cnt);
        let d_outer = d_dz.max(*d_a) + 1;
        rec(format!("outer{ln}"), d_outer);
        for (w, gd) in new_diags[i].iter_mut().zip(&g) {
            *w = sim_sub(w, gd);
        }
        rec(format!("W{ln}'"), d_outer + 1);
        let gb = sim_batchsum(&dz, layout, c, &mut cnt);
        new_biases[i] = sim_sub(&biases[i], &gb);
        rec(format!("b{ln}'"), d_dz + 1);

        if let Some((v, dv)) = next {
            delta = v;
            d_delta = dv;
        }
    }

    let out_layers = layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let (d_out, d_in) = l.w.dim();
            SimLayer {
                w: unpack_weight_diagonals(&new_diags[i], d_out, d_in, layout, 0),
                b: unpack_activations(&new_biases[i], layout, d_out)
                    .row(0)
                    .to_vec(),
            }
        })
        .collect();
    let d_last = layers[n_layers - 1].w.nrows();
    let output = unpack_activations(&a_out, layout, d_last)
        .slice(ndarray::s![..n_valid, ..])
        .to_owned();
    let depth = depths.iter().map(|(_, d)| *d).max().unwrap_or(0);
    SimStep {
        layers: out_layers,
        weight_diags: new_diags,
        bias_slots: new_biases,
        output,
        count: cnt,
        depths,
        depth,
    }
}
