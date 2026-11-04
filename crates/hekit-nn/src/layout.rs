//! Slot layout of activations and weights for batched encrypted training (`docs/layout.md`).
//!
//! # Layout
//!
//! [`Layout`] `{ slots, d, batch }`: feature-major, `slot(j, b) = j * batch + b` for feature
//! `j < d` and lane (sample) `b < batch`. The block of `d * batch` slots repeats
//! `slots / (d * batch)` times. This is M1's strided BSGS layout (`hekit_ckks::linalg`) with
//! `stride = batch`.
//!
//! A vector of logical dimension `d_v` (padded to `n = d_v.next_power_of_two() <= d`) is stored
//! in **canonical form**, `n`-periodic in the feature index: slot `j * batch + b` holds
//! `x_b[j mod n]` (0 for padded features and unused lanes). Inputs, labels and biases are packed
//! this way by the client, and every layer output comes out of the matvec in this form.
//!
//! # Weights
//!
//! `W` (`d_out x d_in`, zero-padded to `n_out x n_in`, `L = max`, `m = min`) is stored as its
//! `m` generalised diagonals `D_c[j] = W[j mod n_out][(j + c) mod n_in]` (`L`-periodic in `j`,
//! the same in every lane). They are kept pre-rotated for BSGS with
//! `(baby, giant) = bsgs_split(m)`: `entries[k1 * baby + k2] = rot(D_{g k1 + k2}, -g k1 batch)`,
//! with `g = baby` ([`pack_weight_diagonals`]). For square `W` this is exactly M1's
//! `diagonals_from_matrix(w, slots, batch)` followed by M1's pre-rotation, so
//! `Diagonals::from_prerotated(m, batch, entries)` and `Evaluator::matvec_enc` apply directly.
//! A wide `W` (`n_in > n_out`) needs a fold after the matvec ([`crate::sim::sim_fold`]). A tall
//! `W` needs nothing, because its canonical input is already `n_in`-periodic.
//!
//! # Transpose
//!
//! Output entry `i` of `W^T` is `rot(entries[transpose_sources[i]], rotations_for_transpose[i])`.
//! That is one rotation per entry, with no addition and no mask. See [`transpose_diagonals`] and
//! `docs/layout.md` §2.3.

use hekit_ckks::linalg::bsgs_split;
use hekit_core::error::{HekitError, Result};
use ndarray::Array2;

use crate::sim::sim_rotate;

/// The slot layout: `slots` slots, `d` feature positions per block and `batch` lanes,
/// `slot(j, b) = j * batch + b`. `d` and `batch` are powers of two and `d * batch` divides
/// `slots` (the block of `d * batch` slots is replicated; `d * batch == slots` is the
/// single-block case).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    /// Slot count of the ciphertext (`N / 2`).
    pub slots: usize,
    /// Feature positions per block: the largest padded layer dimension of the network.
    pub d: usize,
    /// Lanes per block (the padded mini-batch size; the M1 matvec stride).
    pub batch: usize,
}

impl Layout {
    /// Checks and builds a layout.
    ///
    /// Errors (`InvalidParams`): `d` or `batch` zero or not a power of two, or `d * batch` not
    /// dividing `slots`.
    pub fn new(slots: usize, d: usize, batch: usize) -> Result<Layout> {
        let ok = d.is_power_of_two()
            && batch.is_power_of_two()
            && d.checked_mul(batch)
                .is_some_and(|blk| slots > 0 && slots % blk == 0);
        if !ok {
            return Err(HekitError::InvalidParams(format!(
                "layout needs powers of two d = {d}, batch = {batch} with d * batch dividing \
                 slots = {slots}"
            )));
        }
        Ok(Layout { slots, d, batch })
    }

    /// The slot of feature `j` in lane `b` (in the first block).
    pub fn slot(&self, j: usize, b: usize) -> usize {
        j * self.batch + b
    }

    /// The block size `d * batch` (the period of every packed vector).
    pub fn block(&self) -> usize {
        self.d * self.batch
    }

    /// The number of feature positions over all slots, `slots / batch` (a multiple of `d`).
    pub fn features(&self) -> usize {
        self.slots / self.batch
    }

    /// `log2(batch)`: the number of rotations of one batch reduction (and of one replication).
    pub fn log_batch(&self) -> usize {
        self.batch.trailing_zeros() as usize
    }
}

/// The geometry of one dense layer `W` (`d_out x d_in`) in a [`Layout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShape {
    /// Logical output dimension (rows of `W`).
    pub d_out: usize,
    /// Logical input dimension (columns of `W`).
    pub d_in: usize,
    /// Padded output dimension, `d_out.next_power_of_two()`.
    pub n_out: usize,
    /// Padded input dimension, `d_in.next_power_of_two()`.
    pub n_in: usize,
    /// Diagonal length (period in features), `max(n_out, n_in)`.
    pub l: usize,
    /// Number of diagonals (the BSGS dimension), `min(n_out, n_in)`.
    pub m: usize,
    /// BSGS baby steps of `bsgs_split(m)`.
    pub baby: usize,
    /// BSGS giant steps of `bsgs_split(m)`.
    pub giant: usize,
}

impl LayerShape {
    /// The shape of a `d_out x d_in` layer in `layout`.
    ///
    /// # Panics
    /// If a dimension is 0 or its padded size exceeds `layout.d`.
    pub fn new(d_out: usize, d_in: usize, layout: &Layout) -> LayerShape {
        assert!(d_out > 0 && d_in > 0, "layer dimensions must be positive");
        let (n_out, n_in) = (d_out.next_power_of_two(), d_in.next_power_of_two());
        assert!(
            n_out <= layout.d && n_in <= layout.d,
            "layer {d_out} x {d_in} (padded {n_out} x {n_in}) exceeds layout d = {}",
            layout.d
        );
        let (l, m) = (n_out.max(n_in), n_out.min(n_in));
        let (baby, giant) = bsgs_split(m);
        LayerShape {
            d_out,
            d_in,
            n_out,
            n_in,
            l,
            m,
            baby,
            giant,
        }
    }

    /// The shape of `W^T` (`d_in x d_out`): same `l`, `m` and split.
    pub fn transposed(&self) -> LayerShape {
        LayerShape {
            d_out: self.d_in,
            d_in: self.d_out,
            n_out: self.n_in,
            n_in: self.n_out,
            ..*self
        }
    }

    /// Fold rotations after the matvec: `log2(n_in / n_out)` for a wide layer, else 0.
    pub fn fold_steps(&self) -> usize {
        if self.n_in > self.n_out {
            (self.n_in / self.n_out).trailing_zeros() as usize
        } else {
            0
        }
    }

    /// The (padded) value `W~[r][c]` read by diagonal entries: `w[r][c]` inside `w`, else 0.
    fn padded(w: &Array2<f64>, r: usize, c: usize) -> f64 {
        if r < w.nrows() && c < w.ncols() {
            w[[r, c]]
        } else {
            0.0
        }
    }
}

/// Packs `x` (`rows x d_in`, `rows <= batch` samples) in canonical form: slot `j * batch + b`
/// holds `x[b][j mod n_in]` (`n_in = d_in.next_power_of_two()`), 0 for padded features
/// (`j mod n_in >= d_in`) and for unused lanes (`b >= rows`).
///
/// # Panics
/// If `x` has more rows than `layout.batch`, or `n_in > layout.d`.
pub fn pack_activations(x: &Array2<f64>, layout: &Layout) -> Vec<f64> {
    let (rows, cols) = x.dim();
    assert!(
        rows <= layout.batch,
        "{rows} samples exceed batch {}",
        layout.batch
    );
    let n = cols.next_power_of_two();
    assert!(
        cols > 0 && n <= layout.d,
        "dimension {cols} exceeds d = {}",
        layout.d
    );
    (0..layout.slots)
        .map(|s| {
            let (j, b) = ((s / layout.batch) % n, s % layout.batch);
            if b < rows && j < cols { x[[b, j]] } else { 0.0 }
        })
        .collect()
}

/// Packs a bias (or any per-feature vector) in canonical form, the same in every lane: slot
/// `j * batch + b` holds `bias[j mod n]` (`n = bias.len().next_power_of_two()`), 0 for padded
/// features.
///
/// # Panics
/// If `bias` is empty or `n > layout.d`.
pub fn pack_bias(bias: &[f64], layout: &Layout) -> Vec<f64> {
    let rows = Array2::from_shape_fn((layout.batch, bias.len()), |(_, j)| bias[j]);
    pack_activations(&rows, layout)
}

/// Reads lane `b < batch`, feature `j < d_valid` from slot `j * batch + b`: a
/// `batch x d_valid` matrix (rows of unused lanes are whatever the slots hold).
///
/// # Panics
/// If `v.len() != layout.slots` or `d_valid > layout.d`.
pub fn unpack_activations(v: &[f64], layout: &Layout, d_valid: usize) -> Array2<f64> {
    assert_eq!(
        v.len(),
        layout.slots,
        "vector length must be the slot count"
    );
    assert!(
        d_valid <= layout.d,
        "d_valid = {d_valid} exceeds d = {}",
        layout.d
    );
    Array2::from_shape_fn((layout.batch, d_valid), |(b, j)| v[layout.slot(j, b)])
}

/// The `m` generalised diagonals of `w` (`d_out x d_in`) in BSGS order, pre-rotated and
/// replicated over all lanes: `entries[k1 * baby + k2][j * batch + b] = D_c[j - g k1]` with
/// `c = g k1 + k2`, `g = baby`, `D_c[j] = W~[j mod n_out][(j + c) mod n_in]` (`W~` is `w`
/// zero-padded to `n_out x n_in`), feature indices mod `slots / batch`.
///
/// For square `w` this equals `hekit_ckks::linalg::diagonals_from_matrix(w, slots, batch)`
/// followed by M1's BSGS pre-rotation.
///
/// # Panics
/// If `bsgs != bsgs_split(m)` (M1 fixes the split) or the shape does not fit the layout
/// ([`LayerShape::new`]).
pub fn pack_weight_diagonals(
    w: &Array2<f64>,
    layout: &Layout,
    bsgs: (usize, usize),
) -> Vec<Vec<f64>> {
    let sh = LayerShape::new(w.nrows(), w.ncols(), layout);
    assert_eq!(
        bsgs,
        (sh.baby, sh.giant),
        "bsgs must be bsgs_split({}) = {:?}",
        sh.m,
        (sh.baby, sh.giant)
    );
    let f = layout.features();
    (0..sh.m)
        .map(|c| {
            let shift = sh.baby * (c / sh.baby);
            (0..layout.slots)
                .map(|s| {
                    let j = (s / layout.batch + f - shift) % f;
                    LayerShape::padded(w, j % sh.n_out, (j + c) % sh.n_in)
                })
                .collect()
        })
        .collect()
}

/// Inverse of [`pack_weight_diagonals`], reading lane `lane`: the `d_out x d_in` matrix whose
/// pre-rotated diagonals are `diags`. Every padded entry is visited exactly once (the map
/// `(c, j < l) -> (j mod n_out, (j + c) mod n_in)` is a bijection), so the result is exact.
///
/// # Panics
/// If `diags.len() != m`, a diagonal is not `slots` long, or `lane >= batch`.
pub fn unpack_weight_diagonals(
    diags: &[Vec<f64>],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    lane: usize,
) -> Array2<f64> {
    let sh = LayerShape::new(d_out, d_in, layout);
    assert_eq!(diags.len(), sh.m, "expected {} diagonals", sh.m);
    assert!(lane < layout.batch, "lane {lane} >= batch {}", layout.batch);
    let f = layout.features();
    let mut w = Array2::zeros((sh.n_out, sh.n_in));
    for (c, v) in diags.iter().enumerate() {
        assert_eq!(
            v.len(),
            layout.slots,
            "diagonal length must be the slot count"
        );
        let shift = sh.baby * (c / sh.baby);
        for j in 0..sh.l {
            w[[j % sh.n_out, (j + c) % sh.n_in]] = v[layout.slot((j + shift) % f, lane)];
        }
    }
    w.slice(ndarray::s![..d_out, ..d_in]).to_owned()
}

/// For each output entry `i` of `W^T` (BSGS order), the entry `c(i)` of `W` it is rotated
/// from: `c = (-i) mod n_out` for a wide-or-square `W` (`n_in >= n_out`), `c = (-i) mod n_in`
/// for a tall one.
pub fn transpose_sources(d_out: usize, d_in: usize, layout: &Layout) -> Vec<usize> {
    let sh = LayerShape::new(d_out, d_in, layout);
    (0..sh.m).map(|i| (sh.m - i % sh.m) % sh.m).collect()
}

/// For each output entry `i` of `W^T` (BSGS order), the rotation (in slots, in
/// `[0, l * batch)`) applied to entry [`transpose_sources`]`[i]` of `W`:
/// `a_i * batch`, `a_i = (t - g floor(i/g) + g floor(c/g)) mod l`, `t = -c` (wide or square)
/// or `t = i` (tall). `0` means a copy (no key switch). Exactly `m - 1` steps are nonzero.
pub fn rotations_for_transpose(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let g = sh.baby as i64;
    let l = sh.l as i64;
    transpose_sources(d_out, d_in, layout)
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            let (i, c) = (i as i64, c as i64);
            let t = if sh.n_in >= sh.n_out { -c } else { i };
            let a = (t - g * (i / g) + g * (c / g)).rem_euclid(l);
            (a * layout.batch as i64) as i32
        })
        .collect()
}

/// The distinct nonzero steps of [`rotations_for_transpose`], sorted: the Galois keys the
/// encrypted transpose needs (in addition to `bsgs_rotation_steps(m, batch)` for the products).
pub fn transpose_galois_steps(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let mut steps: Vec<i32> = rotations_for_transpose(d_out, d_in, layout)
        .into_iter()
        .filter(|&s| s != 0)
        .collect();
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// The pre-rotated diagonals of `W^T` from those of `W` (`d_out x d_in`), by **rotations
/// only**: `out[i] = rot(diags[transpose_sources[i]], rotations_for_transpose[i])`. The result
/// equals `pack_weight_diagonals(W^T)` exactly; see [`crate::sim::sim_transpose`] for the
/// counted version.
///
/// # Panics
/// If `diags.len() != m` or a diagonal is not `slots` long.
pub fn transpose_diagonals(
    diags: &[Vec<f64>],
    d_out: usize,
    d_in: usize,
    layout: &Layout,
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
            sim_rotate(&diags[c], step)
        })
        .collect()
}
