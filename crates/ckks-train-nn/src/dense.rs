//! Encrypted dense layer: forward (Task 26), transpose and backward (Task 27), weight and
//! bias gradients (Task 28); layout in `docs/layout.md` §2–§3.
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
use std::sync::Arc;

#[cfg(doc)]
use ckks_train_ckks::Evaluator;
use ckks_train_ckks::linalg::{Diagonals, bsgs_rotation_steps};
use ckks_train_ckks::{
    Ciphertext, Decryptor, DiagonalEntry, Encoder, Encryptor, HeCiphertext, HeOps,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::sampler::Rng;
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
pub fn fold<E: HeOps>(
    ev: &E,
    y: &E::Ct,
    d_out: usize,
    d_in: usize,
    layout: &Layout,
) -> Result<E::Ct> {
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
fn check_keys<E: HeOps>(ev: &E, steps: &[i32]) -> Result<()> {
    let have = |s: i32| ev.has_galois_key(s);
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
///
/// `C` is the ciphertext type of the backend ([`HeOps::Ct`]; [`Ciphertext`] for the CPU
/// [`ckks_train_ckks::Evaluator`], the default). The constructors that encrypt are CPU-side;
/// [`EncWeights::upload`] / [`EncWeights::download`] move the weights to and from another
/// backend.
#[derive(Debug, Clone)]
pub struct EncWeights<C = Ciphertext> {
    diags: Diagonals<C>,
    d_out: usize,
    d_in: usize,
}

impl EncWeights {
    /// Packs `w` (`d_out x d_in`) with [`pack_weight_diagonals`] and encrypts every diagonal at
    /// `params.scale()` and the top level: [`EncWeights::encrypt_at`] at `max_level`.
    ///
    /// Errors as [`EncWeights::encrypt_at`].
    pub fn encrypt(
        w: &Array2<f64>,
        layout: &Layout,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<EncWeights> {
        let top = encoder.params().max_level();
        EncWeights::encrypt_at(w, layout, top, enc, encoder, rng)
    }

    /// [`EncWeights::encrypt`] at `level` instead of the top level (e.g. the level a refresher
    /// returns weights on, [`crate::refresh::Refresher::output_level`], or the non-interactive
    /// trainer's start level, [`crate::trainer_ni::NonInteractiveTrainer::start_level`]).
    ///
    /// Errors: `InvalidParams` if `w` does not fit `layout` (a padded dimension above
    /// `layout.d`) or the layout's slot count is not the parameters', and the
    /// [`Encoder::encode`] errors (e.g. a non-finite weight, a level above `max_level`).
    pub fn encrypt_at(
        w: &Array2<f64>,
        layout: &Layout,
        level: usize,
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
            .map(|v| Ok(enc.encrypt(&encoder.encode(v, params.scale(), level)?, rng)))
            .collect::<Result<Vec<_>>>()?;
        EncWeights::from_diagonals(
            Diagonals::from_prerotated(sh.m, layout.batch, entries)?,
            d_out,
            d_in,
            layout,
        )
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

    /// The weights on the backend of `ev` ([`HeOps::upload_ct`] of every diagonal).
    ///
    /// Errors: those of the upload.
    pub fn upload<E: HeOps>(&self, ev: &E) -> Result<EncWeights<E::Ct>> {
        Ok(EncWeights {
            diags: self.diags.try_map(|c| ev.upload_ct(c))?,
            d_out: self.d_out,
            d_in: self.d_in,
        })
    }
}

impl EncWeights {
    /// [`EncWeights::upload`] by value (a move for the CPU [`ckks_train_ckks::Evaluator`]).
    ///
    /// Errors: those of the upload.
    pub fn upload_owned<E: HeOps>(self, ev: &E) -> Result<EncWeights<E::Ct>> {
        Ok(EncWeights {
            diags: self.diags.try_map_owned(|c| ev.upload_ct_owned(c))?,
            d_out: self.d_out,
            d_in: self.d_in,
        })
    }
}

impl<C: HeCiphertext + Clone> EncWeights<C> {
    /// The weights on the host ([`HeOps::download_ct`] of every diagonal), e.g. to decrypt
    /// weights trained on another backend.
    ///
    /// Errors: those of the download.
    pub fn download<E: HeOps<Ct = C>>(&self, ev: &E) -> Result<EncWeights> {
        Ok(EncWeights {
            diags: self.diags.try_map(|c| ev.download_ct(c))?,
            d_out: self.d_out,
            d_in: self.d_in,
        })
    }

    /// Wraps encrypted pre-rotated diagonals of a `d_out x d_in` layer (e.g. weights updated by
    /// the training step, or produced by the transpose).
    ///
    /// Errors: `InvalidParams` if the layer does not fit `layout`, `diags.d() != m` or
    /// `diags.stride() != layout.batch`; `LevelMismatch` / `ScaleMismatch` unless all entries
    /// share one level and exactly one scale ([`EncWeights::level`] and
    /// [`EncWeights::scale`] read entry 0; [`Diagonals::entries_mut`] can break this).
    pub fn from_diagonals(
        diags: Diagonals<C>,
        d_out: usize,
        d_in: usize,
        layout: &Layout,
    ) -> Result<EncWeights<C>> {
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
        check_uniform(&diags)?;
        Ok(EncWeights { diags, d_out, d_in })
    }

    /// The encrypted diagonals.
    pub fn diagonals(&self) -> &Diagonals<C> {
        &self.diags
    }

    /// Consumes the weights and returns the diagonals.
    pub fn into_diagonals(self) -> Diagonals<C> {
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
}

/// An encrypted dense layer `z = x W^T + b` (lane-wise `W x_b + b`).
///
/// `C` is the ciphertext type of the backend ([`Ciphertext`] by default); the operations
/// take any evaluator `E: HeOps<Ct = C>`.
#[derive(Debug, Clone)]
pub struct Dense<C = Ciphertext> {
    /// The weights.
    pub w: EncWeights<C>,
    /// The bias in canonical form, at the exact scale of the unrescaled matvec output
    /// ([`EncWeights::product_scale`]).
    pub b: C,
}

impl Dense {
    /// Encrypts `w` ([`EncWeights::encrypt`]) and `bias` (`d_out` values, [`encrypt_bias`]) at
    /// the top level, the bias at `x_scale * Delta` for inputs of scale `x_scale`:
    /// [`Dense::encrypt_at`] at `max_level`.
    ///
    /// Errors as [`Dense::encrypt_at`].
    pub fn encrypt(
        w: &Array2<f64>,
        bias: &[f64],
        layout: &Layout,
        x_scale: f64,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<Dense> {
        let top = encoder.params().max_level();
        Dense::encrypt_at(w, bias, layout, x_scale, top, enc, encoder, rng)
    }

    /// [`Dense::encrypt`] at `level` instead of the top level.
    ///
    /// Errors: those of [`EncWeights::encrypt_at`] and [`encrypt_bias`], and `InvalidParams` if
    /// `bias.len() != d_out`.
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_at(
        w: &Array2<f64>,
        bias: &[f64],
        layout: &Layout,
        x_scale: f64,
        level: usize,
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
        let w = EncWeights::encrypt_at(w, layout, level, enc, encoder, rng)?;
        let b = encrypt_bias(
            bias,
            layout,
            w.product_scale(x_scale),
            level,
            enc,
            encoder,
            rng,
        )?;
        Ok(Dense { w, b })
    }

    /// The layer on the backend of `ev` ([`EncWeights::upload`] and the bias).
    ///
    /// Errors: those of the upload.
    pub fn upload<E: HeOps>(&self, ev: &E) -> Result<Dense<E::Ct>> {
        Ok(Dense {
            w: self.w.upload(ev)?,
            b: ev.upload_ct(&self.b)?,
        })
    }

    /// [`Dense::upload`] by value (a move for the CPU [`ckks_train_ckks::Evaluator`]).
    ///
    /// Errors: those of the upload.
    pub fn upload_owned<E: HeOps>(self, ev: &E) -> Result<Dense<E::Ct>> {
        Ok(Dense {
            w: self.w.upload_owned(ev)?,
            b: ev.upload_ct_owned(self.b)?,
        })
    }
}

impl<C: HeCiphertext + Clone> Dense<C> {
    /// The layer on the host ([`EncWeights::download`] and the bias).
    ///
    /// Errors: those of the download.
    pub fn download<E: HeOps<Ct = C>>(&self, ev: &E) -> Result<Dense> {
        Ok(Dense {
            w: self.w.download(ev)?,
            b: ev.download_ct(&self.b)?,
        })
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
    pub fn forward<E: HeOps<Ct = C>>(&self, ev: &E, x: &C, layout: &Layout) -> Result<C> {
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

impl<C: HeCiphertext + Clone> EncWeights<C> {
    /// The encrypted `W^T` (`d_in x d_out`) with [`TransposeKeys::Dedicated`]: `m - 1`
    /// rotations, nothing else (see [`EncWeights::transpose_with`]).
    ///
    /// Errors as [`EncWeights::transpose_with`].
    pub fn transpose<E: HeOps<Ct = C>>(&self, ev: &E, layout: &Layout) -> Result<EncWeights<C>> {
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
    pub fn transpose_with<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        layout: &Layout,
        keys: &TransposeKeys,
    ) -> Result<EncWeights<C>> {
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

impl<C: HeCiphertext + Clone> Dense<C> {
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
    pub fn backward_input<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        delta: &C,
        wt: &EncWeights<C>,
        layout: &Layout,
    ) -> Result<C> {
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

// ---------------------------------------------------------------------------------------------
// Task 28: weight and bias gradients (`docs/layout.md` §3).
// ---------------------------------------------------------------------------------------------

/// `LevelMismatch` / `ScaleMismatch` unless all entries share one level and exactly one scale.
pub(crate) fn check_uniform<C: DiagonalEntry>(diags: &Diagonals<C>) -> Result<()> {
    let first = &diags.entries()[0];
    for e in diags.entries() {
        if e.level() != first.level() {
            return Err(HekitError::LevelMismatch);
        }
        if e.scale() != first.scale() {
            return Err(HekitError::ScaleMismatch);
        }
    }
    Ok(())
}

impl<C: HeCiphertext + Clone> EncWeights<C> {
    /// Replaces the diagonals by `diags` of the same shape (an optimizer update).
    ///
    /// Errors: `InvalidParams` if `d` or the stride differ; as [`check_uniform`].
    pub(crate) fn set_diagonals(&mut self, diags: Diagonals<C>) -> Result<()> {
        self.check_same_shape(&diags)?;
        check_uniform(&diags)?;
        self.diags = diags;
        Ok(())
    }

    /// `InvalidParams` unless `other` has the shape (`d`, stride) of these diagonals.
    pub(crate) fn check_same_shape<T>(&self, other: &Diagonals<T>) -> Result<()> {
        if other.d() != self.diags.d() || other.stride() != self.diags.stride() {
            return Err(HekitError::InvalidParams(format!(
                "diagonals (d = {}, stride = {}) for weights with d = {}, stride = {}",
                other.d(),
                other.stride(),
                self.diags.d(),
                self.diags.stride()
            )));
        }
        Ok(())
    }
}

/// `InvalidParams` unless `ct` is a 2-component ciphertext over the evaluator's data basis
/// (what the rotations and products of the gradient need), checked before any work.
fn check_input<E: HeOps>(ev: &E, ct: &E::Ct) -> Result<()> {
    let ours = ev.owns(ct);
    if !ours || ct.size() != 2 {
        return Err(HekitError::InvalidParams(format!(
            "gradient inputs must be 2-component ciphertexts over the evaluator's basis \
             (got {} components{})",
            ct.size(),
            if ours { "" } else { ", foreign basis" }
        )));
    }
    Ok(())
}

/// The lane steps of the batch sum: reduce `2^t` and replicate `-2^t`, `t < log2(batch)`.
fn lane_steps(layout: &Layout) -> (Vec<i32>, Vec<i32>) {
    let reduce: Vec<i32> = (0..layout.log_batch()).map(|t| 1 << t).collect();
    let replicate = reduce.iter().map(|s| -s).collect();
    (reduce, replicate)
}

/// The Galois steps of [`Dense::bias_grad`], sorted: the batch reduce `2^t` and replicate
/// `-2^t`, `t < log2(batch)` (none for `batch = 1`).
pub fn bias_grad_rotation_steps(layout: &Layout) -> Vec<i32> {
    let (mut steps, rep) = lane_steps(layout);
    steps.extend(rep);
    steps.sort_unstable();
    steps
}

/// The Galois steps of [`Dense::weight_grad`] (and [`Dense::bias_grad`]) for a `d_out x d_in`
/// layer, sorted and distinct: the hoisted baby rotations of the layer input `k2 batch`
/// (`1 <= k2 < baby`), the giant rotations of `delta` `-g k1 batch` (`1 <= k1 < giant`,
/// `g = baby`) and [`bias_grad_rotation_steps`].
///
/// # Panics
/// If the layer does not fit the layout ([`LayerShape::new`]).
pub fn grad_rotation_steps(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let b = layout.batch;
    let mut steps: Vec<i32> = (1..sh.baby).map(|k2| (k2 * b) as i32).collect();
    steps.extend((1..sh.giant).map(|k1| -((sh.baby * k1 * b) as i32)));
    steps.extend(bias_grad_rotation_steps(layout));
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// Steps 3–5 of the batch sum (`docs/layout.md` §3, [`crate::sim::sim_batchsum`]) of `v`
/// (2 components, level `l >= 1`): reduce (`log2 B` rotations), multiply by the lane-0 mask
/// `mask` (encoded at [`Evaluator::target_plain_scale`]`(v_reduced, target)`; the folded
/// constant is inside it) and rescale (1 ct-pt, 1 level), replicate (`log2 B` rotations). The
/// result is at level `l - 1` with scale exactly `target`.
fn batch_sum<E: HeOps>(
    ev: &E,
    v: &E::Ct,
    mask: &E::Pt,
    target: f64,
    layout: &Layout,
) -> Result<E::Ct> {
    let (reduce, replicate) = lane_steps(layout);
    let mut v = v.clone();
    for s in reduce {
        let r = ev.rotate(&v, s)?;
        v = ev.add(&v, &r)?;
    }
    let mut v = ev.rescale(&ev.mul_plain_to_scale(&v, mask, target)?)?;
    for s in replicate {
        let r = ev.rotate(&v, s)?;
        v = ev.add(&v, &r)?;
    }
    Ok(v)
}

/// `c * lane0_mask` encoded at `scale` and `level`.
fn lane0_plaintext<E: HeOps>(
    ev: &E,
    layout: &Layout,
    c: f64,
    scale: f64,
    level: usize,
) -> Result<E::Pt> {
    if !c.is_finite() {
        return Err(HekitError::InvalidParams(format!(
            "gradient constant {c} must be finite"
        )));
    }
    let mask: Vec<f64> = (0..layout.slots)
        .map(|s| if s % layout.batch == 0 { c } else { 0.0 })
        .collect();
    ev.upload_pt_owned(Encoder::new(Arc::clone(ev.params())).encode(&mask, scale, level)?)
}

impl<C: HeCiphertext + Clone> Dense<C> {
    /// The Galois steps [`Dense::weight_grad`] needs: [`grad_rotation_steps`] of this layer.
    ///
    /// # Panics
    /// If the layer does not fit `layout`.
    pub fn grad_rotation_steps(&self, layout: &Layout) -> Vec<i32> {
        grad_rotation_steps(self.w.d_out, self.w.d_in, layout)
    }

    /// The weight gradient `delta^T A / batch` (the mean over the `B = layout.batch` lanes) as
    /// pre-rotated, replicated diagonals: [`Dense::weight_grad_scaled`] with `c = 1 / B`.
    ///
    /// Errors as [`Dense::weight_grad_scaled`].
    pub fn weight_grad<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        delta: &C,
        a_prev: &C,
        layout: &Layout,
    ) -> Result<Diagonals<C>> {
        self.weight_grad_scaled(ev, delta, a_prev, layout, 1.0 / layout.batch as f64)
    }

    /// The weight gradient `c * delta^T A` (`d_out x d_in`, summed over the lanes) of this
    /// layer, op for op [`crate::sim::sim_outer_batchsum`] (`docs/layout.md` §3). `delta` is
    /// canonical with period `n_out` (the layer's `dz`), `a_prev` with period `n_in` (the
    /// layer input). The constant `c` (e.g. `lr / n` for SGD, [`crate::optim::Optimizer::grad_scale`]
    /// `/ n` in general, or 1 when the caller folded it into the derivative factor `s_L`) is
    /// folded into the lane-0 mask plaintext: **it costs no level**.
    ///
    /// For `c = g k1 + k2`, entry `k1 baby + k2` is
    /// `batchsum(rot(delta, -g k1 B) * rot(A, k2 B))`:
    /// 1. both inputs are switched down (free) to the lower of their levels `l`; `baby - 1`
    ///    hoisted rotations of `A` and `giant - 1` hoisted rotations of `delta`;
    /// 2. per diagonal `mul` (1 tensor, 1 relin) and `rescale` (level `l - 1`);
    /// 3. reduce `v += rot(v, 2^t)`, `t < log2 B`;
    /// 4. [`Evaluator::mul_plain_to_scale`] by `c * lane0_mask` (encoded at
    ///    [`Evaluator::target_plain_scale`], target `self.w.scale()`) and `rescale`
    ///    (level `l - 2`, scale **exactly** `self.w.scale()`);
    /// 5. replicate `v += rot(v, -2^t)`.
    ///
    /// Costs `(baby - 1) + (giant - 1) + 2 m log2 B` rotations, `m` tensors, `m` relins, `m`
    /// ct-pt and `2 m` rescales (2 levels). The result is in exactly the packed, pre-rotated,
    /// replicated form of [`EncWeights`] ([`crate::layout::pack_weight_diagonals`] of
    /// `c delta^T A` in every lane), at level `l - 2` and scale `self.w.scale()`: the update
    /// `W - dW` is a plain `sub` of identical scales after switching `W` down (free).
    ///
    /// An [`Encoder`] for the mask is built per call (`O(N)` tables, small next to the
    /// rotations).
    ///
    /// Errors, all reported before any homomorphic work (the counter does not move):
    /// `InvalidParams` if `layout` is not the weights' (slot count, batch), a key of
    /// [`grad_rotation_steps`] or the relinearization key is missing, `c` is not finite, or
    /// the inputs are not 2-component ciphertexts over the evaluator's basis; `NoLevelsLeft`
    /// if `min(delta.level(), a_prev.level()) < 2`; `ScaleMismatch` if `self.w.scale()` is
    /// not reachable by a rescale from level `l - 1`
    /// ([`Evaluator::exact_product_scale`]; never for power-of-two weight scales).
    pub fn weight_grad_scaled<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        delta: &C,
        a_prev: &C,
        layout: &Layout,
        c: f64,
    ) -> Result<Diagonals<C>> {
        let sh = self.w.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        self.w.check_batch(layout)?;
        check_keys(ev, &grad_rotation_steps(sh.d_out, sh.d_in, layout))?;
        if !ev.has_relin_key() {
            return Err(HekitError::InvalidParams(
                "evaluator has no relinearization key".into(),
            ));
        }
        for ct in [delta, a_prev] {
            check_input(ev, ct)?;
        }
        let level = delta.level().min(a_prev.level());
        if level < 2 {
            return Err(HekitError::NoLevelsLeft);
        }
        // The scale of every product after its rescale (rotations keep scales), and the
        // mask plaintext that lands it exactly on the weights' scale.
        let prod_scale = delta.scale() * a_prev.scale() / ev.params().q()[level] as f64;
        let target = self.w.scale();
        let mask_scale = ev.exact_product_scale(level - 1, target)? / prod_scale;
        let mask = lane0_plaintext(ev, layout, c, mask_scale, level - 1)?;

        let b = layout.batch;
        let delta = ev.mod_switch_to_level(delta, level)?;
        let a = ev.mod_switch_to_level(a_prev, level)?;
        let a_steps: Vec<i32> = (0..sh.baby).map(|k2| (k2 * b) as i32).collect();
        let d_steps: Vec<i32> = (0..sh.giant)
            .map(|k1| -((sh.baby * k1 * b) as i32))
            .collect();
        let a_rot = ev.hoisted_rotate(&a, &a_steps)?;
        let d_rot = ev.hoisted_rotate(&delta, &d_steps)?;
        let entries = (0..sh.m)
            .map(|e| {
                let p = ev.rescale(&ev.mul(&d_rot[e / sh.baby], &a_rot[e % sh.baby])?)?;
                batch_sum(ev, &p, &mask, target, layout)
            })
            .collect::<Result<Vec<_>>>()?;
        Diagonals::from_prerotated(sh.m, b, entries)
    }

    /// The bias gradient `sum_b delta_b / batch` in canonical form (every lane):
    /// [`Dense::bias_grad_scaled`] with `c = 1 / layout.batch`.
    ///
    /// Errors as [`Dense::bias_grad_scaled`].
    pub fn bias_grad<E: HeOps<Ct = C>>(&self, ev: &E, delta: &C, layout: &Layout) -> Result<C> {
        self.bias_grad_scaled(ev, delta, layout, 1.0 / layout.batch as f64)
    }

    /// The bias gradient `c * sum_b delta_b` (`d_out` values) in canonical form, replicated in
    /// every lane, op for op [`crate::sim::sim_batchsum`]: reduce (`log2 B` rotations),
    /// [`Evaluator::mul_plain_to_scale`] by `c * lane0_mask` with target `self.b.scale()` and
    /// `rescale` (1 ct-pt, 1 level), replicate (`log2 B` rotations). The result is at
    /// `delta.level() - 1` with scale **exactly** `self.b.scale()`, so `b - db` (after
    /// switching `b` down) is a plain `sub` and the updated bias still has the exact scale
    /// [`Dense::forward`] requires. `c` folds at no level, as in [`Dense::weight_grad_scaled`].
    ///
    /// Errors, all before any homomorphic work: `InvalidParams` if `layout` is not the
    /// weights', a key of [`bias_grad_rotation_steps`] is missing, `c` is not finite or
    /// `delta` is not a 2-component ciphertext over the evaluator's basis; `NoLevelsLeft` at
    /// level 0; `ScaleMismatch` if `self.b.scale()` is not reachable by a rescale from
    /// `delta.level()` ([`Evaluator::exact_product_scale`]; possible but rare for a bias
    /// encrypted for a rescaled input scale: switch `delta` down a level and retry).
    pub fn bias_grad_scaled<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        delta: &C,
        layout: &Layout,
        c: f64,
    ) -> Result<C> {
        self.w.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        self.w.check_batch(layout)?;
        check_keys(ev, &bias_grad_rotation_steps(layout))?;
        check_input(ev, delta)?;
        let target = self.b.scale();
        let mask_scale = ev.target_plain_scale(delta, target)?;
        let mask = lane0_plaintext(ev, layout, c, mask_scale, delta.level())?;
        batch_sum(ev, delta, &mask, target, layout)
    }
}

// ---------------------------------------------------------------------------------------------
// Task 41: reduce-only gradients for the non-interactive (compact) update.
// ---------------------------------------------------------------------------------------------

/// The Galois steps of [`Dense::weight_grad_reduced`] for a `d_out x d_in` layer, sorted and
/// distinct: the baby rotations of the layer input `k2 batch` (`1 <= k2 < baby`), the giant
/// rotations of `delta` `-g k1 batch` (`1 <= k1 < giant`) and the batch reduce `2^t`
/// (`t < log2 batch`). No replication steps (the gradient stays in lane 0). The bias's
/// [`Dense::bias_grad_reduced`] needs only the reduce steps.
///
/// # Panics
/// If the layer does not fit the layout ([`LayerShape::new`]).
pub fn grad_reduced_rotation_steps(d_out: usize, d_in: usize, layout: &Layout) -> Vec<i32> {
    let sh = LayerShape::new(d_out, d_in, layout);
    let b = layout.batch;
    let mut steps: Vec<i32> = (1..sh.baby).map(|k2| (k2 * b) as i32).collect();
    steps.extend((1..sh.giant).map(|k1| -((sh.baby * k1 * b) as i32)));
    steps.extend(lane_steps(layout).0);
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// `c` on lane 0 of the first `period` features (slots `j batch`, `j < period`), 0 elsewhere,
/// encoded at `scale` and `level`: the batch-sum mask that is also the compact pack mask
/// (`docs/layout.md` §6.4).
fn first_period_plaintext<E: HeOps>(
    ev: &E,
    layout: &Layout,
    period: usize,
    c: f64,
    scale: f64,
    level: usize,
) -> Result<E::Pt> {
    if !c.is_finite() {
        return Err(HekitError::InvalidParams(format!(
            "gradient constant {c} must be finite"
        )));
    }
    let span = period * layout.batch;
    let mask: Vec<f64> = (0..layout.slots)
        .map(|s| {
            if s < span && s % layout.batch == 0 {
                c
            } else {
                0.0
            }
        })
        .collect();
    ev.upload_pt_owned(Encoder::new(Arc::clone(ev.params())).encode(&mask, scale, level)?)
}

/// `InvalidParams` unless `xs` is nonempty with one level and exactly one scale for all.
fn check_micro<C: DiagonalEntry>(xs: &[&C], what: &str) -> Result<()> {
    let Some(first) = xs.first() else {
        return Err(HekitError::InvalidParams(format!("no {what} ciphertexts")));
    };
    if xs
        .iter()
        .any(|x| x.level() != first.level() || x.scale() != first.scale())
    {
        return Err(HekitError::InvalidParams(format!(
            "the {what} micro-batches must share one level and one scale"
        )));
    }
    Ok(())
}

impl<C: HeCiphertext + Clone> Dense<C> {
    /// The weight gradient `c * sum_mb delta_mb^T A_mb` for the compact update
    /// (`docs/layout.md` §6.4, ruling R-M3-11): the outer product of
    /// [`Dense::weight_grad_scaled`], summed over the micro-batches `mb` of one update before
    /// a single relinearization per diagonal, then the batch **reduce only** and the
    /// first-period lane-0 mask with `c` folded in. Entry `e` (BSGS order) is zero outside slots
    /// `j B`, `j < L`, where it holds `c * G_e[j]` (the pre-rotated diagonal of §3): exactly the
    /// precondition of [`crate::compact::compact_shift`]. No replication.
    ///
    /// 1. all inputs are switched down (free) to their lowest level `l`; per micro-batch
    ///    `baby - 1` hoisted rotations of `A` and `giant - 1` hoisted rotations of `delta`;
    /// 2. per diagonal: one [`Evaluator::mul_no_relin`] per micro-batch, summed (degree 2),
    ///    then one relin and one rescale (level `l - 1`);
    /// 3. reduce `v += rot(v, 2^t)`, `t < log2 B` (lane 0 of each feature now holds the sum);
    /// 4. [`Evaluator::mul_plain_to_scale`] by `c * mask0_L` (encoded at
    ///    [`Evaluator::target_plain_scale`], target `self.w.scale()`) and rescale (level
    ///    `l - 2`, scale **exactly** `self.w.scale()`).
    ///
    /// Costs `k ((baby - 1) + (giant - 1)) + m log2 B` rotations, `k m` tensors, `m` relins,
    /// `m` ct-pt and `2 m` rescales (2 levels) for `k` micro-batches. Against
    /// [`Dense::weight_grad_scaled`] (k = 1): `m log2 B` fewer rotations (no replication).
    ///
    /// Errors, all before any homomorphic work: `InvalidParams` if `deltas` and `a_prevs` are
    /// empty or of different lengths, a micro-batch differs in level or scale from the first,
    /// `layout` is not the weights', a key of [`grad_reduced_rotation_steps`] or the
    /// relinearization key is missing, `c` is not finite, or an input is not a 2-component
    /// ciphertext over the evaluator's basis; `NoLevelsLeft` if `l < 2`; `ScaleMismatch` if
    /// `self.w.scale()` is not reachable by a rescale from level `l - 1`.
    pub fn weight_grad_reduced<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        deltas: &[&C],
        a_prevs: &[&C],
        layout: &Layout,
        c: f64,
    ) -> Result<Vec<C>> {
        let mut out = self.weight_grad_reduced_consts(ev, deltas, a_prevs, layout, &[c])?;
        Ok(out.pop().expect("one constant"))
    }

    /// [`Dense::weight_grad_reduced`] for several folded constants at once: the products, the
    /// relinearizations and the reduce are shared, and each constant gets its own lane-0 mask
    /// (`m` ct-pt and `m` rescales per constant, the same level). Output `k` carries `cs[k]`.
    /// The non-interactive trainer uses two constants when a per-step learning rate differs
    /// from the velocity's reference lr (`docs/layout.md` §7.1).
    ///
    /// Errors as [`Dense::weight_grad_reduced`], and `InvalidParams` for no constant.
    pub fn weight_grad_reduced_consts<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        deltas: &[&C],
        a_prevs: &[&C],
        layout: &Layout,
        cs: &[f64],
    ) -> Result<Vec<Vec<C>>> {
        if cs.is_empty() {
            return Err(HekitError::InvalidParams("no gradient constant".into()));
        }
        let sh = self.w.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        self.w.check_batch(layout)?;
        if deltas.len() != a_prevs.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} delta and {} input micro-batches",
                deltas.len(),
                a_prevs.len()
            )));
        }
        check_micro(deltas, "delta")?;
        check_micro(a_prevs, "layer input")?;
        check_keys(ev, &grad_reduced_rotation_steps(sh.d_out, sh.d_in, layout))?;
        if !ev.has_relin_key() {
            return Err(HekitError::InvalidParams(
                "evaluator has no relinearization key".into(),
            ));
        }
        for ct in deltas.iter().chain(a_prevs) {
            check_input(ev, ct)?;
        }
        let level = deltas[0].level().min(a_prevs[0].level());
        if level < 2 {
            return Err(HekitError::NoLevelsLeft);
        }
        let prod_scale = deltas[0].scale() * a_prevs[0].scale() / ev.params().q()[level] as f64;
        let target = self.w.scale();
        let mask_scale = ev.exact_product_scale(level - 1, target)? / prod_scale;
        let masks = cs
            .iter()
            .map(|&c| first_period_plaintext(ev, layout, sh.l, c, mask_scale, level - 1))
            .collect::<Result<Vec<_>>>()?;

        let b = layout.batch;
        let a_steps: Vec<i32> = (0..sh.baby).map(|k2| (k2 * b) as i32).collect();
        let d_steps: Vec<i32> = (0..sh.giant)
            .map(|k1| -((sh.baby * k1 * b) as i32))
            .collect();
        let mut rots = Vec::with_capacity(deltas.len());
        for (d, a) in deltas.iter().zip(a_prevs) {
            let d = ev.mod_switch_to_level(d, level)?;
            let a = ev.mod_switch_to_level(a, level)?;
            rots.push((
                ev.hoisted_rotate(&d, &d_steps)?,
                ev.hoisted_rotate(&a, &a_steps)?,
            ));
        }
        let (reduce, _) = lane_steps(layout);
        let mut out: Vec<Vec<C>> = cs.iter().map(|_| Vec::with_capacity(sh.m)).collect();
        for e in 0..sh.m {
            let mut acc: Option<C> = None;
            for (d_rot, a_rot) in &rots {
                // Lazy (default): degree-2 terms, one relin below; naive: relin each.
                let p = ev.mul_lazy(&d_rot[e / sh.baby], &a_rot[e % sh.baby])?;
                acc = Some(match acc {
                    None => p,
                    Some(s) => ev.add(&s, &p)?,
                });
            }
            let p = acc.expect("at least one micro-batch");
            let mut v = ev.rescale(&ev.relin_lazy(p)?)?;
            for &s in &reduce {
                let r = ev.rotate(&v, s)?;
                v = ev.add(&v, &r)?;
            }
            for (o, mask) in out.iter_mut().zip(&masks) {
                o.push(ev.rescale(&ev.mul_plain_to_scale(&v, mask, target)?)?);
            }
        }
        Ok(out)
    }

    /// The bias gradient `c * sum_mb sum_b dz_mb,b` for the compact update: the micro-batch
    /// `dz`s summed (no op beyond additions), the batch **reduce only** (`log2 B` rotations)
    /// and the first-`period` lane-0 mask with `c` folded in (1 ct-pt, 1 level), landing on
    /// exactly `target_scale`. Slot `j B` (`j < period`) holds `c * sum dz[j mod n_out]`, every
    /// other slot is 0: the input of one compact position of period `period >= n_out`
    /// (`docs/layout.md` §7.3).
    ///
    /// Errors, all before any homomorphic work: `InvalidParams` if `dzs` is empty or a
    /// micro-batch differs in level or scale, `layout` is not the weights', a reduce key is
    /// missing, `c` is not finite, `period` is not a power of two in `n_out..=layout.d`, or an
    /// input is not a 2-component ciphertext over the evaluator's basis; and those of
    /// [`Evaluator::target_plain_scale`] (`NoLevelsLeft` at level 0, `ScaleMismatch`).
    pub fn bias_grad_reduced<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        dzs: &[&C],
        layout: &Layout,
        c: f64,
        period: usize,
        target_scale: f64,
    ) -> Result<C> {
        let sh = self.w.shape(layout)?;
        check_slots(layout, ev.params().slots())?;
        self.w.check_batch(layout)?;
        check_micro(dzs, "dz")?;
        let (reduce, _) = lane_steps(layout);
        check_keys(ev, &reduce)?;
        if !(period.is_power_of_two() && period >= sh.n_out && period <= layout.d) {
            return Err(HekitError::InvalidParams(format!(
                "bias period {period} for {} outputs in layout d = {}",
                sh.n_out, layout.d
            )));
        }
        for ct in dzs {
            check_input(ev, ct)?;
        }
        let mask_scale = ev.target_plain_scale(dzs[0], target_scale)?;
        let mask = first_period_plaintext(ev, layout, period, c, mask_scale, dzs[0].level())?;
        let mut v = dzs[0].clone();
        for d in &dzs[1..] {
            v = ev.add(&v, d)?;
        }
        for s in reduce {
            let r = ev.rotate(&v, s)?;
            v = ev.add(&v, &r)?;
        }
        ev.rescale(&ev.mul_plain_to_scale(&v, &mask, target_scale)?)
    }
}
