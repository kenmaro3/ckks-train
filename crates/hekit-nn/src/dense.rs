//! Encrypted dense layer, forward pass (Task 26; layout in `docs/layout.md` §2).
//!
//! # Op sequence (= [`crate::sim::sim_dense_forward`], op for op)
//!
//! For `W` (`d_out x d_in`) in a [`Layout`] with `B = layout.batch` lanes,
//! `(baby, giant) = bsgs_split(m)`, `m = min(n_out, n_in)`:
//!
//! 1. `y = matvec_enc(W, x)`: M1's [`Evaluator::matvec_enc`] on the `m` generalised diagonals
//!    ([`crate::layout::pack_weight_diagonals`], already pre-rotated for BSGS) wrapped by
//!    `Diagonals::from_prerotated(m, B, ·)`. The generalised diagonals need no variant of the
//!    BSGS loop: M1's loop with `d = m`, `stride = B` is exactly [`crate::sim::sim_matvec`]
//!    (`baby - 1` hoisted + `giant - 1` giant rotations, `m` tensors, `giant` relins). The
//!    result is unrescaled: level `x.level()`, scale `x.scale() * W.scale()`.
//! 2. [`fold`] for a wide layer (`n_in > n_out`): `log2(n_in / n_out)` rotations and additions
//!    ([`crate::sim::sim_fold`]). Nothing for square or tall layers.
//! 3. `y + b`: the bias ciphertext must have **exactly** the scale `x.scale() * W.scale()` of
//!    the unrescaled product (M1 §7 exact-scale path; [`EncWeights::product_scale`]), so the
//!    addition relabels nothing. No relin.
//! 4. One rescale: the output is at `x.level() - 1` with scale `x.scale() W.scale() / q_l`, in
//!    canonical form (`n_out`-periodic), ready for the next layer.
//!
//! So a forward costs `baby + giant - 2 + fold_steps` rotations, `m` tensors, `giant` relins,
//! no ct-pt products and 1 rescale (1 level), and needs Galois keys for exactly
//! [`forward_rotation_steps`].
//!
//! # Levels
//!
//! Weights and bias are usually encrypted at the top level. [`Dense::forward`] switches them
//! down (free: primes dropped, scale unchanged) to the level of `x` when they are above it.
//! An operand below `x.level()` is `LevelMismatch`.
//!
//! # Transpose and backward pass (Task 27; `docs/layout.md` §2.3)
//!
//! [`EncWeights::transpose`] builds the diagonals of `W^T` by **rotations only** (no addition,
//! no mask, no product, no level): entry `i` of `W^T` is entry
//! [`crate::layout::transpose_sources`]`[i]` of `W` rotated by
//! [`crate::layout::rotations_for_transpose`]`[i]`, op for op
//! [`crate::sim::sim_transpose`] (`m - 1` rotations, one distinct ciphertext each, so plain
//! [`Evaluator::rotate`]; there is nothing to hoist). [`TransposeKeys::Reuse`] is the
//! zero-new-key form: each amount is split into a shortest chain of steps whose keys the
//! training step already has ([`training_key_pool`]), usually 2 rotations per entry.
//!
//! [`Dense::backward_input`] is `delta W` (lane-wise `W^T delta`) =
//! `rescale(fold_{W^T}(matvec_enc(W^T, delta)))`, op for op
//! [`crate::sim::sim_backward_input`]: the forward of [`Dense::forward`] on `W^T` without a
//! bias. The output is canonical with period `n_in`, one level below `delta`.

use std::borrow::Cow;
use std::collections::VecDeque;

use hekit_ckks::linalg::{Diagonals, bsgs_rotation_steps};
use hekit_ckks::{Ciphertext, Decryptor, Encoder, Encryptor, Evaluator};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;
use ndarray::Array2;

use crate::layout::{
    LayerShape, Layout, pack_bias, pack_weight_diagonals, rotations_for_transpose,
    transpose_sources, unpack_weight_diagonals,
};

/// The shape of a `d_out x d_in` layer in `layout`, as an error instead of
/// [`LayerShape::new`]'s panic.
fn checked_shape(d_out: usize, d_in: usize, layout: &Layout) -> Result<LayerShape> {
    let fits = |n: usize| n > 0 && n.next_power_of_two() <= layout.d;
    if !(fits(d_out) && fits(d_in)) {
        return Err(HekitError::InvalidParams(format!(
            "layer {d_out} x {d_in} does not fit layout d = {}",
            layout.d
        )));
    }
    Ok(LayerShape::new(d_out, d_in, layout))
}

/// `InvalidParams` unless `layout` has the slot count of the parameters.
fn check_slots(layout: &Layout, slots: usize) -> Result<()> {
    if layout.slots != slots {
        return Err(HekitError::InvalidParams(format!(
            "layout of {} slots for parameters with {slots} slots",
            layout.slots
        )));
    }
    Ok(())
}

/// The fold rotation steps of a layer: `n_out * 2^t * batch` for `t < fold_steps()`.
fn fold_rotation_steps(sh: &LayerShape, layout: &Layout) -> Vec<i32> {
    (0..sh.fold_steps())
        .map(|t| (sh.n_out * (1 << t) * layout.batch) as i32)
        .collect()
}

/// The Galois steps of one forward of a `d_out x d_in` layer, sorted and distinct:
/// `bsgs_rotation_steps(m, batch)` (the matvec) and the fold steps `n_out * 2^t * batch`,
/// `t < log2(n_in / n_out)` (wide layers only). Generate exactly these keys (n16_train keys
/// are 302 MB each).
///
/// # Panics
/// If the layer does not fit the layout ([`LayerShape::new`]).
pub fn forward_rotation_steps(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let mut steps = bsgs_rotation_steps(sh.m, layout.batch);
    steps.extend(fold_rotation_steps(&sh, layout));
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// The fold of a wide layer (`n_in > n_out`) after the matvec, encrypted
/// ([`crate::sim::sim_fold`]): `for t < log2(n_in / n_out): y = y + rot(y, n_out 2^t batch)`.
/// Rotations and additions only (no level, no relin); a copy for square or tall layers. Also
/// the fold of the backward `W^T` matvec (Task 27: call with the dimensions of `W^T`).
///
/// Errors: `InvalidParams` if the layer does not fit `layout` or a fold key is missing, and
/// those of [`Evaluator::rotate`].
pub fn fold(
    ev: &Evaluator,
    y: &Ciphertext,
    d_out: usize,
    d_in: usize,
    layout: &Layout,
) -> Result<Ciphertext> {
    let sh = checked_shape(d_out, d_in, layout)?;
    let steps = fold_rotation_steps(&sh, layout);
    check_keys(ev, &steps)?;
    let mut y = y.clone();
    for step in steps {
        let r = ev.rotate(&y, step)?;
        y = ev.add(&y, &r)?;
    }
    Ok(y)
}

/// `InvalidParams("missing galois key")` unless `ev` has a key for every step.
fn check_keys(ev: &Evaluator, steps: &[i32]) -> Result<()> {
    let have = |s: i32| ev.galois_keys().is_some_and(|g| g.get(s).is_some());
    match steps.iter().find(|&&s| !have(s)) {
        Some(s) => Err(HekitError::InvalidParams(format!(
            "missing galois key for step {s}"
        ))),
        None => Ok(()),
    }
}

/// Encrypts `bias` (`d_out` values) in canonical form ([`pack_bias`]) at `scale` and `level`.
/// For [`Dense::forward`], `scale` must be [`EncWeights::product_scale`]`(x.scale())`
/// exactly.
///
/// Errors: `InvalidParams` if the layout's slot count is not the parameters', `bias` does not
/// fit the layout, and the [`Encoder::encode`] errors.
pub fn encrypt_bias(
    bias: &[f64],
    layout: &Layout,
    scale: f64,
    level: usize,
    enc: &Encryptor,
    encoder: &Encoder,
    rng: &mut Rng,
) -> Result<Ciphertext> {
    check_slots(layout, encoder.params().slots())?;
    checked_shape(bias.len(), 1, layout)?;
    let pt = encoder.encode(&pack_bias(bias, layout), scale, level)?;
    Ok(enc.encrypt(&pt, rng))
}

/// The encrypted weights of a `d_out x d_in` layer: its `m` generalised diagonals, pre-rotated
/// for BSGS, as M1 [`Diagonals`] with `d = m` and `stride = batch` (`docs/layout.md` §2).
#[derive(Debug, Clone)]
pub struct EncWeights {
    diags: Diagonals<Ciphertext>,
    d_out: usize,
    d_in: usize,
}

impl EncWeights {
    /// Packs `w` (`d_out x d_in`) with [`pack_weight_diagonals`] and encrypts every diagonal at
    /// `params.scale()` and the top level.
    ///
    /// Errors: `InvalidParams` if `w` does not fit `layout` (a padded dimension above
    /// `layout.d`) or the layout's slot count is not the parameters', and the
    /// [`Encoder::encode`] errors (e.g. a non-finite weight).
    pub fn encrypt(
        w: &Array2<f64>,
        layout: &Layout,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<EncWeights> {
        let (d_out, d_in) = w.dim();
        let sh = checked_shape(d_out, d_in, layout)?;
        let params = encoder.params();
        check_slots(layout, params.slots())?;
        let entries = pack_weight_diagonals(w, layout, (sh.baby, sh.giant))
            .iter()
            .map(|v| Ok(enc.encrypt(&encoder.encode(v, params.scale(), params.max_level())?, rng)))
            .collect::<Result<Vec<_>>>()?;
        EncWeights::from_diagonals(
            Diagonals::from_prerotated(sh.m, layout.batch, entries)?,
            d_out,
            d_in,
            layout,
        )
    }

    /// Wraps encrypted pre-rotated diagonals of a `d_out x d_in` layer (e.g. weights updated by
    /// the training step, or produced by the transpose).
    ///
    /// Errors (`InvalidParams`): the layer does not fit `layout`, `diags.d() != m` or
    /// `diags.stride() != layout.batch`.
    pub fn from_diagonals(
        diags: Diagonals<Ciphertext>,
        d_out: usize,
        d_in: usize,
        layout: &Layout,
    ) -> Result<EncWeights> {
        let sh = checked_shape(d_out, d_in, layout)?;
        if diags.d() != sh.m || diags.stride() != layout.batch {
            return Err(HekitError::InvalidParams(format!(
                "diagonals (d = {}, stride = {}) for a {d_out} x {d_in} layer need d = {}, \
                 stride = {}",
                diags.d(),
                diags.stride(),
                sh.m,
                layout.batch
            )));
        }
        Ok(EncWeights { diags, d_out, d_in })
    }

    /// The encrypted diagonals.
    pub fn diagonals(&self) -> &Diagonals<Ciphertext> {
        &self.diags
    }

    /// Consumes the weights and returns the diagonals.
    pub fn into_diagonals(self) -> Diagonals<Ciphertext> {
        self.diags
    }

    /// Output dimension (rows of `W`).
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Input dimension (columns of `W`).
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// The level of the diagonals (all equal, checked at construction).
    pub fn level(&self) -> usize {
        self.diags.entries()[0].level()
    }

    /// The scale of the diagonals (all exactly equal, checked at construction).
    pub fn scale(&self) -> f64 {
        self.diags.entries()[0].scale()
    }

    /// The scale of the unrescaled matvec output for an input of scale `x_scale`,
    /// `x_scale * self.scale()`: the exact scale the bias must have.
    pub fn product_scale(&self, x_scale: f64) -> f64 {
        x_scale * self.scale()
    }

    /// The layer's shape in `layout`.
    ///
    /// Errors (`InvalidParams`): the layer does not fit `layout`.
    pub fn shape(&self, layout: &Layout) -> Result<LayerShape> {
        checked_shape(self.d_out, self.d_in, layout)
    }

    /// Decrypts and unpacks the weights (lane 0, [`unpack_weight_diagonals`]): the
    /// `d_out x d_in` matrix. For tests and diagnostics.
    ///
    /// # Panics
    /// If `layout` is not the layout of the weights, or the keys are not for these parameters.
    pub fn decrypt(&self, dec: &Decryptor, encoder: &Encoder, layout: &Layout) -> Array2<f64> {
        let vals: Vec<Vec<f64>> = self
            .diags
            .entries()
            .iter()
            .map(|c| encoder.decode(&dec.decrypt(c)))
            .collect();
        unpack_weight_diagonals(&vals, self.d_out, self.d_in, layout, 0)
    }
}

/// An encrypted dense layer `z = x W^T + b` (lane-wise `W x_b + b`).
#[derive(Debug, Clone)]
pub struct Dense {
    /// The weights.
    pub w: EncWeights,
    /// The bias in canonical form, at the exact scale of the unrescaled matvec output
    /// ([`EncWeights::product_scale`]).
    pub b: Ciphertext,
}

impl Dense {
    /// Encrypts `w` ([`EncWeights::encrypt`]) and `bias` (`d_out` values, [`encrypt_bias`]) at
    /// the top level, the bias at `x_scale * Delta` for inputs of scale `x_scale`.
    ///
    /// Errors: those of [`EncWeights::encrypt`] and [`encrypt_bias`], and `InvalidParams` if
    /// `bias.len() != d_out`.
    pub fn encrypt(
        w: &Array2<f64>,
        bias: &[f64],
        layout: &Layout,
        x_scale: f64,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<Dense> {
        if bias.len() != w.nrows() {
            return Err(HekitError::InvalidParams(format!(
                "bias of {} values for {} outputs",
                bias.len(),
                w.nrows()
            )));
        }
        let w = EncWeights::encrypt(w, layout, enc, encoder, rng)?;
        let top = encoder.params().max_level();
        let b = encrypt_bias(
            bias,
            layout,
            w.product_scale(x_scale),
            top,
            enc,
            encoder,
            rng,
        )?;
        Ok(Dense { w, b })
    }

    /// The forward pass `rescale(fold(matvec_enc(W, x)) + b)` (see the
    /// [module docs](self)). `x` is canonical with period `n_in` (e.g.
    /// [`crate::layout::pack_activations`], or the previous layer's output). The result is
    /// canonical with period `n_out`, one level below `x`.
    ///
    /// Errors, all reported before any homomorphic work (the counter does not move):
    /// `InvalidParams` if `layout` is not the weights' (slot count, batch) or a forward key is
    /// missing; `LevelMismatch` if `W` or `b` is below `x.level()`; `ScaleMismatch` unless
    /// `b.scale() == x.scale() * W.scale()` exactly; and the checks of
    /// [`Evaluator::matvec_enc`] (`NoLevelsLeft` at level 0, a missing relin key, ...).
    pub fn forward(&self, ev: &Evaluator, x: &Ciphertext, layout: &Layout) -> Result<Ciphertext> {
        let sh = self.w.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        if self.w.diags.stride() != layout.batch {
            return Err(HekitError::InvalidParams(format!(
                "weights packed for batch {}, layout batch {}",
                self.w.diags.stride(),
                layout.batch
            )));
        }
        check_keys(ev, &forward_rotation_steps(sh.d_out, sh.d_in, layout))?;
        let level = x.level();
        if self.w.level() < level || self.b.level() < level {
            return Err(HekitError::LevelMismatch);
        }
        if self.b.scale() != self.w.product_scale(x.scale()) {
            return Err(HekitError::ScaleMismatch);
        }

        let diags = if self.w.level() > level {
            Cow::Owned(self.w.diags.mod_switch_to_level(ev, level)?)
        } else {
            Cow::Borrowed(&self.w.diags)
        };
        let b = if self.b.level() > level {
            Cow::Owned(ev.mod_switch_to_level(&self.b, level)?)
        } else {
            Cow::Borrowed(&self.b)
        };
        let y = ev.matvec_enc(&diags, x)?;
        let y = fold(ev, &y, sh.d_out, sh.d_in, layout)?;
        let y = ev.add(&y, &b)?;
        ev.rescale(&y)
    }

    /// The Galois steps [`Dense::forward`] needs: [`forward_rotation_steps`] of this layer.
    ///
    /// # Panics
    /// If the layer does not fit `layout`.
    pub fn forward_rotation_steps(&self, layout: &Layout) -> Vec<i32> {
        forward_rotation_steps(self.w.d_out, self.w.d_in, layout)
    }
}

// ---------------------------------------------------------------------------------------------
// Task 27: rotation-only transpose and backward pass.
// ---------------------------------------------------------------------------------------------

/// Which Galois keys [`EncWeights::transpose_with`] rotates with (`docs/layout.md` §2.3,
/// "Galois-key cost of the transpose").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransposeKeys {
    /// One rotation per entry by [`crate::layout::rotations_for_transpose`]: `m - 1` key
    /// switches (the cost model's count), but up to `m - 1` keys no other op uses
    /// ([`crate::layout::transpose_galois_steps`]). The default ([`EncWeights::transpose`]).
    Dedicated,
    /// Zero new keys: every amount is split into a shortest chain of steps from this pool
    /// (e.g. [`training_key_pool`], or its union over all layers of the network). Steps that
    /// are not multiples of `batch` (they would mix lanes) are ignored. With the layer's own
    /// [`training_key_pool`] an entry costs at most 2 rotations for square, tall and
    /// `n_in = 2 n_out` layers, and up to 3 for wider ones (16 x 64: 30 rotations instead of
    /// 15); the network pool can do better (eMLP-1 W2: 27 instead of 15, 0 of 14 new keys).
    /// Choose it when key memory binds (one n16_train key is about 302 MB).
    Reuse(Vec<i32>),
}

/// The Galois steps of [`Dense::backward_input`] for a `d_out x d_in` layer `W`: the forward
/// steps of `W^T`, [`forward_rotation_steps`]`(d_in, d_out)` (BSGS steps of `m`, and the fold
/// `n_in 2^t batch` when `W` is tall).
///
/// # Panics
/// If the layer does not fit the layout ([`LayerShape::new`]).
pub fn backward_rotation_steps(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    forward_rotation_steps(d_in, d_out, layout)
}

/// The multiple-of-`batch` Galois steps one training step of a `d_out x d_in` layer already
/// needs keys for, sorted and distinct: [`forward_rotation_steps`] (forward),
/// [`backward_rotation_steps`] (backward) and the gradient's giant steps `-g k1 batch`,
/// `1 <= k1 < giant` (`crate::sim::sim_outer_batchsum`). The pool of
/// [`TransposeKeys::Reuse`] for a transpose without new keys. (The batch reduction's steps
/// `±2^t` are not multiples of `batch` and cannot serve the transpose.)
///
/// # Panics
/// If the layer does not fit the layout ([`LayerShape::new`]).
pub fn training_key_pool(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let mut steps = forward_rotation_steps(d_out, d_in, layout);
    steps.extend(backward_rotation_steps(d_out, d_in, layout));
    steps.extend((1..sh.giant).map(|k1| -((sh.baby * k1 * layout.batch) as i32)));
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// The rotation plan of the transpose of a `d_out x d_in` layer: for each output entry `i` of
/// `W^T` (BSGS order), `(c, steps)`: entry `c` of `W` rotated by each of `steps` in turn
/// (empty: a copy). Each `c < m` appears once.
///
/// - [`TransposeKeys::Dedicated`]: `(transpose_sources[i], [rotations_for_transpose[i]])`,
///   the amount dropped when it is 0; exactly [`crate::sim::sim_transpose`].
/// - [`TransposeKeys::Reuse`]: the same source, and a shortest chain of pool steps whose sum is
///   the amount modulo `l * batch` (the period of every diagonal, so the result is the same).
///   Breadth-first over the residues, pool steps tried in ascending order (deterministic).
///
/// Errors: `InvalidParams` if the layer does not fit `layout`, or (reuse) some amount is not
/// a sum of pool steps.
pub fn transpose_rotation_plan(
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    keys: &TransposeKeys,
) -> Result<Vec<(usize, Vec<i32>)>> {
    let sh = checked_shape(d_out, d_in, layout)?;
    let plan = transpose_sources(d_out, d_in, layout)
        .into_iter()
        .zip(rotations_for_transpose(d_out, d_in, layout));
    match keys {
        TransposeKeys::Dedicated => Ok(plan
            .map(|(c, a)| (c, if a == 0 { Vec::new() } else { vec![a] }))
            .collect()),
        TransposeKeys::Reuse(pool) => {
            let chains = shortest_chains(pool, sh.l, layout.batch);
            plan.map(|(c, a)| {
                let u = a as usize / layout.batch;
                chains[u].clone().map(|s| (c, s)).ok_or_else(|| {
                    HekitError::InvalidParams(format!(
                        "transpose amount {a} is not a sum of the reuse pool steps {pool:?}"
                    ))
                })
            })
            .collect()
        }
    }
}

/// For every residue `u < l` (in units of `batch` slots), a shortest chain of `pool` steps
/// summing to `u * batch` modulo `l * batch`, or `None` if unreachable. Pool steps that are not
/// multiples of `batch` or are multiples of `l * batch` are skipped.
fn shortest_chains(pool: &[i32], l: usize, batch: usize) -> Vec<Option<Vec<i32>>> {
    let mut steps: Vec<(i32, usize)> = pool
        .iter()
        .filter(|&&s| s as i64 % batch as i64 == 0)
        .map(|&s| (s, (s as i64 / batch as i64).rem_euclid(l as i64) as usize))
        .filter(|&(_, u)| u != 0)
        .collect();
    steps.sort_unstable();
    steps.dedup();
    let mut prev: Vec<Option<(usize, i32)>> = vec![None; l];
    let mut seen = vec![false; l];
    seen[0] = true;
    let mut queue = VecDeque::from([0usize]);
    while let Some(r) = queue.pop_front() {
        for &(s, u) in &steps {
            let next = (r + u) % l;
            if !seen[next] {
                seen[next] = true;
                prev[next] = Some((r, s));
                queue.push_back(next);
            }
        }
    }
    (0..l)
        .map(|u| {
            if !seen[u] {
                return None;
            }
            let mut chain = Vec::new();
            let mut r = u;
            while let Some((p, s)) = prev[r] {
                chain.push(s);
                r = p;
            }
            chain.reverse();
            Some(chain)
        })
        .collect()
}

/// The Galois steps the transpose of a `d_out x d_in` layer rotates by, sorted and distinct
/// (generate exactly these keys, plus [`backward_rotation_steps`] for the backward pass).
/// For [`TransposeKeys::Dedicated`] this is [`crate::layout::transpose_galois_steps`]; for
/// [`TransposeKeys::Reuse`] a subset of the pool.
///
/// Errors as [`transpose_rotation_plan`].
pub fn transpose_rotation_steps(
    d_out: usize,
    d_in: usize,
    layout: &Layout,
    keys: &TransposeKeys,
) -> Result<Vec<i32>> {
    let mut steps: Vec<i32> = transpose_rotation_plan(d_out, d_in, layout, keys)?
        .into_iter()
        .flat_map(|(_, s)| s)
        .collect();
    steps.sort_unstable();
    steps.dedup();
    Ok(steps)
}

impl EncWeights {
    /// The encrypted `W^T` (`d_in x d_out`) with [`TransposeKeys::Dedicated`]: `m - 1`
    /// rotations, nothing else (see [`EncWeights::transpose_with`]).
    ///
    /// Errors as [`EncWeights::transpose_with`].
    pub fn transpose(&self, ev: &Evaluator, layout: &Layout) -> Result<EncWeights> {
        self.transpose_with(ev, layout, &TransposeKeys::Dedicated)
    }

    /// The encrypted `W^T` (`d_in x d_out`), by rotations only: entry `i` is entry `c` of `W`
    /// rotated by the steps of [`transpose_rotation_plan`] (one [`Evaluator::rotate`] per step;
    /// each entry is a distinct ciphertext, so there is nothing to hoist). No addition, product,
    /// relin or rescale: the result has the level and the scale of `W`, and its diagonals are
    /// `pack_weight_diagonals(W^T)` up to the key-switch noise of the rotations.
    ///
    /// Key switching costs fewer limbs at a lower level: to transpose for a `delta` below the
    /// weights' level, switch the diagonals down first ([`Diagonals::mod_switch_to_level`]).
    ///
    /// Errors, all reported before any homomorphic work (the counter does not move):
    /// `InvalidParams` if `layout` is not the weights' (slot count, batch), a reuse amount is
    /// not reachable from the pool, or a key of [`transpose_rotation_steps`] is missing; and
    /// those of [`Evaluator::rotate`].
    pub fn transpose_with(
        &self,
        ev: &Evaluator,
        layout: &Layout,
        keys: &TransposeKeys,
    ) -> Result<EncWeights> {
        let sh = self.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        self.check_batch(layout)?;
        let plan = transpose_rotation_plan(self.d_out, self.d_in, layout, keys)?;
        check_keys(
            ev,
            &transpose_rotation_steps(self.d_out, self.d_in, layout, keys)?,
        )?;
        let src = self.diags.entries();
        let entries = plan
            .iter()
            .map(|(c, steps)| match steps.split_first() {
                None => Ok(src[*c].clone()),
                Some((&s0, rest)) => rest
                    .iter()
                    .try_fold(ev.rotate(&src[*c], s0)?, |ct, &s| ev.rotate(&ct, s)),
            })
            .collect::<Result<Vec<_>>>()?;
        EncWeights::from_diagonals(
            Diagonals::from_prerotated(sh.m, layout.batch, entries)?,
            self.d_in,
            self.d_out,
            layout,
        )
    }

    /// [`transpose_rotation_steps`] of these weights.
    ///
    /// Errors as [`transpose_rotation_plan`].
    pub fn transpose_rotation_steps(
        &self,
        layout: &Layout,
        keys: &TransposeKeys,
    ) -> Result<Vec<i32>> {
        transpose_rotation_steps(self.d_out, self.d_in, layout, keys)
    }

    /// `InvalidParams` unless the diagonals are packed for `layout.batch` lanes.
    fn check_batch(&self, layout: &Layout) -> Result<()> {
        if self.diags.stride() != layout.batch {
            return Err(HekitError::InvalidParams(format!(
                "weights packed for batch {}, layout batch {}",
                self.diags.stride(),
                layout.batch
            )));
        }
        Ok(())
    }
}

impl Dense {
    /// The backward pass to the layer input, `delta W` (lane-wise `W^T delta_b`), from the
    /// transposed weights `wt` ([`EncWeights::transpose`] of `self.w`):
    /// `rescale(fold_{W^T}(matvec_enc(W^T, delta)))`, op for op
    /// [`crate::sim::sim_backward_input`]. `delta` is canonical with period `n_out` (e.g. the
    /// layer's `dz`). The fold runs when `W^T` is wide (`W` tall). The result is canonical with
    /// period `n_in`, at `delta.level() - 1` with scale `delta.scale() * wt.scale() / q_l`
    /// (one rescale of the exact product scale; no addition of other histories).
    ///
    /// Costs `baby + giant - 2 + fold` rotations, `m` tensors, `giant` relins, 1 rescale.
    /// `wt` above `delta.level()` is switched down (free).
    ///
    /// Errors, all reported before any homomorphic work (the counter does not move):
    /// `InvalidParams` if `wt` is not `d_in x d_out`, `layout` is not the weights' or a key of
    /// [`backward_rotation_steps`] is missing; `LevelMismatch` if `wt` is below
    /// `delta.level()`; and the checks of [`Evaluator::matvec_enc`].
    pub fn backward_input(
        &self,
        ev: &Evaluator,
        delta: &Ciphertext,
        wt: &EncWeights,
        layout: &Layout,
    ) -> Result<Ciphertext> {
        let (d_out, d_in) = (self.w.d_out, self.w.d_in);
        if (wt.d_out, wt.d_in) != (d_in, d_out) {
            return Err(HekitError::InvalidParams(format!(
                "backward of a {d_out} x {d_in} layer needs {d_in} x {d_out} weights, got {} x {}",
                wt.d_out, wt.d_in
            )));
        }
        wt.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        wt.check_batch(layout)?;
        check_keys(ev, &backward_rotation_steps(d_out, d_in, layout))?;
        let level = delta.level();
        if wt.level() < level {
            return Err(HekitError::LevelMismatch);
        }
        let diags = if wt.level() > level {
            Cow::Owned(wt.diags.mod_switch_to_level(ev, level)?)
        } else {
            Cow::Borrowed(&wt.diags)
        };
        let y = ev.matvec_enc(&diags, delta)?;
        let y = fold(ev, &y, d_in, d_out, layout)?;
        ev.rescale(&y)
    }

    /// The Galois steps [`Dense::backward_input`] needs: [`backward_rotation_steps`] of this
    /// layer.
    ///
    /// # Panics
    /// If the layer does not fit `layout`.
    pub fn backward_rotation_steps(&self, layout: &Layout) -> Vec<i32> {
        backward_rotation_steps(self.w.d_out, self.w.d_in, layout)
    }
}
