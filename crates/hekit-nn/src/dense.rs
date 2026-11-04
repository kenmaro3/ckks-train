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

use std::borrow::Cow;

use hekit_ckks::linalg::{Diagonals, bsgs_rotation_steps};
use hekit_ckks::{Ciphertext, Decryptor, Encoder, Encryptor, Evaluator};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;
use ndarray::Array2;

use crate::layout::{
    LayerShape, Layout, pack_bias, pack_weight_diagonals, unpack_weight_diagonals,
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
