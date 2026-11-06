//! The encrypted MLP and its trainer in interactive mode (Task 30; `docs/layout.md` §4).
//!
//! # One training step ([`Trainer::train_batch`])
//!
//! The default SGD op sequence is [`crate::sim::sim_train_step_sgd`]'s, op for op (the
//! counters are equal; tested), generalised to any [`Activation`] and [`LayerOptimizer`]:
//!
//! 1. forward, per layer: `z = Dense::forward(a)`, `a' = act.forward(z)`;
//! 2. loss gradient `delta_L = a_L - y` ([`Mse::grad`]: `y` must be encrypted at exactly
//!    `a_L`'s scale, see [`BatchSpec`]);
//! 3. per layer, top down:
//!    - derivative factor `s = act'(z) * valid` ([`Activation::derivative_factor_at`] with
//!      constant 1 and the valid-lane mask; the activation's own factor, e.g. 2 for Square,
//!      is inside it). `z` is on a high level, so this is off the critical path;
//!    - `dz = rescale(delta * s)` (the one level on the critical path). An activation with a
//!      constant derivative `k` ([`Activation::constant_derivative`], e.g. the identity of a
//!      linear output layer, Task 32) has no `s`: `dz = delta` for `k = 1` on a full batch
//!      (no op, no level), else `rescale(delta * (k * valid))` (one plaintext multiply);
//!    - for every layer but the first: `W^T` by rotations only
//!      ([`EncWeights::transpose_with`] on the **old** weights, switched down to `dz`'s level
//!      first: fewer limbs to key-switch) and `delta_{l-1} = Dense::backward_input(dz)`;
//!    - `dW = Dense::weight_grad_scaled(dz, a_{l-1}, c)` with `c = grad_scale / n` folded into
//!      the lane-0 mask (no level; [`crate::optim`] module docs), and the bias gradient
//!      `Dense::bias_grad_scaled(dz, c_b)` with `c_b = bias_grad_scale / n` (zero for
//!      [`Trainer::with_frozen_biases`]). Scheduled NAG additionally uses a reference-rate
//!      gradient from shared products/reduction when current rate differs from reference:
//!      `m` extra ct-pt products, `m` rescales and `m log2(B)` rotations, no extra depth;
//! 4. the optimizer steps (`W' = W - dW` for SGD; NAG's scaled-velocity form) and
//!    `b' = b - db`; all at exactly the old scales;
//! 5. the refresh: every weight diagonal, every bias and every optimizer state (the NAG
//!    velocity) goes through the [`Refresher`] to its [`Refresher::output_level`]
//!    ([`Trainer::refresh_level`]: the top level for [`crate::refresh::ClientRefresher`], the
//!    bootstrap output level for [`crate::refresh::BootstrapRefresher`]) at its exact scale.
//!
//! On seal_2020 the Task 30 network (4 → 8 → 2, Square, batch 8) uses exactly the 9 levels:
//! `W1'` lands on level 0 before the refresh ([`Trainer::trace`] records the level of every
//! value, labelled as the simulator's level table). No `seal_2020_l10` is needed.
//!
//! [`Trainer::set_lr`] validates every optimizer before committing any rate change.
//! Errors in steps 1–3 (levels, scales, keys, shapes) leave the network and the optimizer
//! states unchanged: nothing is written before all gradients exist. An optimizer error can
//! leave earlier optimizers updated while the network is still uncommitted; there is no
//! cross-optimizer rollback. A refresh failure can leave a partially refreshed updated
//! network. Rate-change validation is atomic independently of these training-step errors.
//!
//! # Deviations from the brief's interface
//!
//! - One optimizer **per layer** (`Vec<Box<dyn LayerOptimizer>>` instead of one
//!   `Box<dyn Optimizer>`): [`Nag`] holds one layer's velocity (Task 28). [`LayerOptimizer`]
//!   adds what the trainer needs on top of [`Optimizer`]: the bias learning rate and the
//!   refresh of the optimizer state.
//! - [`Mlp::predict_plain`] takes the [`Layout`] (the weights' unpacking needs it), and
//!   [`EpochLog`] also records the batch count, the refreshed ciphertexts and the refresh time.
//!
//! # Backends and ablation flags (Task 43)
//!
//! [`Mlp`], [`Trainer`], [`LayerOptimizer`], [`crate::optim::Optimizer`],
//! [`crate::refresh::Refresher`] and [`Activation`] are generic over the evaluator
//! `E: `[`HeOps`] (default [`Evaluator`], so existing CPU code reads unchanged); ciphertexts
//! are `E::Ct`. Encryption, decryption and encoding stay on the host ([`Ciphertext`],
//! [`ckks_train_ckks::Plaintext`]) and are uploaded ([`HeOps::upload_ct_owned`],
//! [`HeOps::upload_pt_owned`]: moves for the CPU). The evaluator's
//! [`ckks_train_ckks::EvalConfig`] flags apply to the step: `hoist` to the hoisted rotations of the
//! matvecs and the gradient, `bsgs` to the forward and backward matvecs, `lazy_relin` to the
//! matvec relinearizations (the interactive gradient multiplies one product per diagonal and
//! has nothing to relinearize lazily); see [`ckks_train_ckks::ops`]. The simulator's counts
//! ([`crate::sim`]) are those of the default flags.
//!
//! # Galois keys
//!
//! [`mlp_rotation_steps`] is the exact key set of a training step: the forward steps of every
//! layer, the gradient steps of every layer, and the backward and transpose steps of every
//! layer but the first (the first layer has no backward pass). With
//! [`TransposeKeys::Reuse`]`(`[`mlp_key_pool`]`)` the transpose adds no key at all.

use std::sync::Arc;
use std::time::Instant;

use ckks_train_ckks::{
    Ciphertext, Decryptor, DiagonalEntry, Diagonals, Encoder, Encryptor, Evaluator, HeOps,
    OpCounter,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::poly::RnsPoly;
use ckks_train_core::sampler::Rng;
use ndarray::Array2;

use crate::activation::Activation;
use crate::dense::{
    Dense, EncWeights, TransposeKeys, backward_rotation_steps, forward_rotation_steps,
    grad_rotation_steps, transpose_rotation_steps,
};
use crate::layout::{Layout, pack_activations, unpack_activations};
use crate::loss::Mse;
use crate::optim::{Nag, Optimizer, Sgd, check_lr};
use crate::refresh::Refresher;
use crate::sim::valid_lane_mask;

// ---------------------------------------------------------------------------------------------
// Optimizers per layer.
// ---------------------------------------------------------------------------------------------

/// An [`Optimizer`] as the trainer uses it: one instance per layer.
pub trait LayerOptimizer<E: HeOps = Evaluator>: Optimizer<E> {
    /// The factor the bias gradient carries (folded into its lane-0 mask, divided by the
    /// valid sample count): the bias update is `b' = b - bias_grad_scale() * mean(dz)`.
    /// [`Sgd`] and [`Nag`]: `lr` (biases take plain SGD steps, Task 28).
    fn bias_grad_scale(&self) -> f64;

    /// Refreshes the optimizer's encrypted state (e.g. the NAG velocity, which lands on the
    /// gradient's level) to the refresher's [`Refresher::output_level`]; returns the number
    /// of ciphertexts refreshed. On an error the state is unchanged.
    fn refresh_state(&mut self, refresher: &mut dyn Refresher<E>) -> Result<usize>;

    /// Read-only rate-change validation. Defaults to unsupported so existing custom
    /// optimizers remain usable without silently claiming schedule support.
    /// Implement together with [`LayerOptimizer::apply_lr`].
    fn validate_lr(&self, _lr: f64) -> Result<()> {
        Err(HekitError::InvalidParams(
            "layer optimizer does not support learning-rate changes".into(),
        ))
    }

    /// Infallible commit of a rate accepted by [`LayerOptimizer::validate_lr`].
    /// The trainer validates every optimizer before calling this on any layer. An opt-in
    /// custom implementation must assign the validated value without failure or panic and
    /// without resetting state. The default is unreachable through the trainer's setter
    /// because default validation always rejects; callers must not bypass validation.
    fn apply_lr(&mut self, _lr: f64) {}
}

impl<E: HeOps> LayerOptimizer<E> for Sgd {
    fn validate_lr(&self, lr: f64) -> Result<()> {
        check_lr(lr)
    }

    fn apply_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn bias_grad_scale(&self) -> f64 {
        Sgd::bias_grad_scale(self)
    }

    /// SGD has no state: nothing to refresh.
    fn refresh_state(&mut self, _refresher: &mut dyn Refresher<E>) -> Result<usize> {
        Ok(0)
    }
}

impl<E: HeOps> LayerOptimizer<E> for Nag<E::Ct> {
    fn validate_lr(&self, lr: f64) -> Result<()> {
        Nag::validate_rate(self, lr)
    }

    fn apply_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    fn bias_grad_scale(&self) -> f64 {
        Nag::bias_grad_scale(self)
    }

    /// Refreshes every entry of the scaled velocity `U` (scale `Delta` kept exactly).
    fn refresh_state(&mut self, refresher: &mut dyn Refresher<E>) -> Result<usize> {
        let Some(u) = self.velocity() else {
            return Ok(0);
        };
        let entries = u
            .entries()
            .iter()
            .map(|c| refresher.refresh(c))
            .collect::<Result<Vec<_>>>()?;
        let fresh = Diagonals::from_prerotated(u.d(), u.stride(), entries)?;
        let n = fresh.entries().len();
        self.set_velocity(Some(fresh));
        Ok(n)
    }
}

// ---------------------------------------------------------------------------------------------
// Galois keys.
// ---------------------------------------------------------------------------------------------

/// `InvalidParams` unless `dims` has at least two entries, all nonzero with padded dimension
/// at most `layout.d`.
pub(crate) fn check_dims(dims: &[usize], layout: &Layout) -> Result<()> {
    if dims.len() < 2 {
        return Err(HekitError::InvalidParams(format!(
            "an MLP needs at least 2 dimensions (got {dims:?})"
        )));
    }
    if let Some(d) = dims
        .iter()
        .find(|&&d| d == 0 || d.next_power_of_two() > layout.d)
    {
        return Err(HekitError::InvalidParams(format!(
            "dimension {d} does not fit layout d = {}",
            layout.d
        )));
    }
    Ok(())
}

pub(crate) fn sorted(mut v: Vec<i32>) -> Vec<i32> {
    v.sort_unstable();
    v.dedup();
    v
}

/// The Galois steps of a training step of the MLP `dims[0] → dims[1] → …` **without** the
/// transpose, sorted and distinct: every layer's [`forward_rotation_steps`] and
/// [`grad_rotation_steps`], and the [`backward_rotation_steps`] of every layer but the first.
/// Pass it as [`TransposeKeys::Reuse`] for a transpose without new keys.
///
/// Errors: `InvalidParams` for fewer than 2 dimensions or a dimension that does not fit
/// `layout`.
pub fn mlp_key_pool(dims: &[usize], layout: &Layout) -> Result<Vec<i32>> {
    check_dims(dims, layout)?;
    let mut steps = Vec::new();
    for (i, p) in dims.windows(2).enumerate() {
        let (d_in, d_out) = (p[0], p[1]);
        steps.extend(forward_rotation_steps(d_out, d_in, layout));
        steps.extend(grad_rotation_steps(d_out, d_in, layout));
        if i > 0 {
            steps.extend(backward_rotation_steps(d_out, d_in, layout));
        }
    }
    Ok(sorted(steps))
}

/// The exact Galois steps of [`Trainer::train_batch`] with `keys`: [`mlp_key_pool`] and the
/// [`transpose_rotation_steps`] of every layer but the first. Generate exactly these keys
/// (one seal_2020 key ≈ 58 MB, one n16_train key ≈ 302 MB).
///
/// Errors: those of [`mlp_key_pool`], and of [`transpose_rotation_steps`] (a reuse pool that
/// cannot reach a transpose amount).
pub fn mlp_rotation_steps(
    dims: &[usize],
    layout: &Layout,
    keys: &TransposeKeys,
) -> Result<Vec<i32>> {
    let mut steps = mlp_key_pool(dims, layout)?;
    for p in dims.windows(2).skip(1) {
        steps.extend(transpose_rotation_steps(p[1], p[0], layout, keys)?);
    }
    Ok(sorted(steps))
}

// ---------------------------------------------------------------------------------------------
// Scale chain (what the client must know to encrypt biases and labels exactly).
// ---------------------------------------------------------------------------------------------

/// `(level, scale)` of [`Dense::forward`]'s output for an input at `(level, x_scale)` and
/// weights of scale `w_scale`: one rescale of the exact product scale (the bias has exactly
/// that scale, so the addition keeps it).
pub(crate) fn dense_out<E: HeOps>(
    ev: &E,
    level: usize,
    x_scale: f64,
    w_scale: f64,
) -> Result<(usize, f64)> {
    if level == 0 {
        return Err(HekitError::NoLevelsLeft);
    }
    Ok((
        level - 1,
        (x_scale * w_scale) / ev.params().q()[level] as f64,
    ))
}

/// Runs activations on zero ciphertexts to read their output `(level, scale)`: the scale is a
/// deterministic function of the input scale and the primes, and this reads it from the
/// activation's own code. A fork of the training evaluator ([`HeOps::fork`]): it shares the
/// keys (no key copy) and has its own counter, so the training evaluator's counter does not
/// move.
pub(crate) struct ScaleProbe<E: HeOps> {
    ev: E,
}

/// A zero ciphertext at `level` and `scale` on `ev`'s backend (built on the host, uploaded).
pub(crate) fn zero_ct<E: HeOps>(ev: &E, level: usize, scale: f64) -> Result<E::Ct> {
    let mut p = RnsPoly::zero(ev.params().basis_q(), level);
    p.to_ntt();
    ev.upload_ct_owned(Ciphertext::new(vec![p.clone(), p], scale)?)
}

impl<E: HeOps> ScaleProbe<E> {
    pub(crate) fn new(ev: &E) -> ScaleProbe<E> {
        ScaleProbe { ev: ev.fork() }
    }

    pub(crate) fn activation(
        &self,
        act: &dyn Activation<E>,
        encoder: &Encoder,
        level: usize,
        scale: f64,
    ) -> Result<(usize, f64)> {
        let ct = zero_ct(&self.ev, level, scale)?;
        let a = act.forward(&self.ev, encoder, &ct)?;
        Ok((a.level(), a.scale()))
    }
}

/// Where the client encrypts a batch: `x` at `(x_level, x_scale)` and `y` at exactly the
/// output's `(y_level, y_scale)` ([`Mse::grad`] requires the scales to be equal). Computed by
/// [`Mlp::batch_spec`] once (it is the same for every batch while the weights are refreshed
/// to the same scales).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchSpec {
    /// Level of the input batch (the top level, or the refresher's output level).
    pub x_level: usize,
    /// Scale of the input batch (`Delta`).
    pub x_scale: f64,
    /// Level of the network's output, where `y` is encrypted.
    pub y_level: usize,
    /// Exact scale of the network's output.
    pub y_scale: f64,
}

impl BatchSpec {
    /// Packs ([`pack_activations`]) and encrypts `x` (`n x d_0`) and `y` (`n x d_L`) for
    /// `n <= layout.batch` samples; train with `n` valid samples
    /// ([`Trainer::train_batch_valid`]).
    ///
    /// Errors: `InvalidParams` if the row counts differ, exceed the batch, or a width does not
    /// fit the layout, and the [`Encoder::encode`] errors.
    pub fn encrypt(
        &self,
        enc: &Encryptor,
        encoder: &Encoder,
        layout: &Layout,
        x: &Array2<f64>,
        y: &Array2<f64>,
        rng: &mut Rng,
    ) -> Result<(Ciphertext, Ciphertext)> {
        let fits = |a: &Array2<f64>| {
            a.nrows() <= layout.batch && a.ncols() > 0 && a.ncols().next_power_of_two() <= layout.d
        };
        if x.nrows() != y.nrows() || !fits(x) || !fits(y) {
            return Err(HekitError::InvalidParams(format!(
                "batch x {:?} / y {:?} does not fit layout (d = {}, batch = {})",
                x.dim(),
                y.dim(),
                layout.d,
                layout.batch
            )));
        }
        let xp = encoder.encode(&pack_activations(x, layout), self.x_scale, self.x_level)?;
        let yp = encoder.encode(&pack_activations(y, layout), self.y_scale, self.y_level)?;
        Ok((enc.encrypt(&xp, rng), enc.encrypt(&yp, rng)))
    }
}

// ---------------------------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------------------------

/// An encrypted MLP: dense layers, each followed by its activation (the last one included).
///
/// `E` is the evaluator the network is trained on ([`Evaluator`] by default); the layers hold
/// its ciphertexts ([`HeOps::Ct`]).
pub struct Mlp<E: HeOps = Evaluator> {
    /// The layers in forward order.
    #[allow(clippy::type_complexity)]
    pub layers: Vec<(Dense<E::Ct>, Box<dyn Activation<E>>)>,
}

impl<E: HeOps> Mlp<E> {
    /// Encrypts the plaintext layers `(W_l, b_l)` (`W_l` is `d_l x d_{l-1}`) at the top level
    /// for inputs encrypted at the top level with scale `Delta` (as [`BatchSpec`] does):
    /// [`Mlp::encrypt_at`] at `max_level`, the level [`crate::refresh::ClientRefresher`]
    /// refreshes to. For any other refresher, start at its
    /// [`crate::refresh::Refresher::output_level`] with [`Mlp::encrypt_at`].
    ///
    /// Errors as [`Mlp::encrypt_at`].
    pub fn encrypt(
        ev: &E,
        layers: &[(Array2<f64>, Vec<f64>)],
        acts: Vec<Box<dyn Activation<E>>>,
        layout: &Layout,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<Mlp<E>> {
        let top = encoder.params().max_level();
        Mlp::encrypt_at(ev, layers, acts, layout, top, enc, encoder, rng)
    }

    /// Encrypts the plaintext layers `(W_l, b_l)` (`W_l` is `d_l x d_{l-1}`) at `level` for
    /// inputs encrypted at `level` with scale `Delta` (as [`Mlp::batch_spec_at`] does): the
    /// weights at `Delta`, each bias at the exact product scale its forward adds it to (the
    /// input scale of layer `l` follows from the chain; the activation scales are read with a
    /// zero-ciphertext probe on a clone of `ev`, which shares its keys). `level` is where the
    /// refreshed values land between steps ([`crate::refresh::Refresher::output_level`]), so
    /// the first step sees the same levels and scales as every later one.
    ///
    /// Errors: `InvalidParams` if `layers` is empty, `acts` has another length, the shapes do
    /// not chain or do not fit `layout`, or `level > max_level`; `NoLevelsLeft` if the chain
    /// runs out of levels; and the errors of [`Dense::encrypt_at`] and of the activations.
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_at(
        ev: &E,
        layers: &[(Array2<f64>, Vec<f64>)],
        acts: Vec<Box<dyn Activation<E>>>,
        layout: &Layout,
        level: usize,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<Mlp<E>> {
        if layers.is_empty() || acts.len() != layers.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} layers and {} activations",
                layers.len(),
                acts.len()
            )));
        }
        let mut dims = vec![layers[0].0.ncols()];
        for (w, _) in layers {
            if w.ncols() != *dims.last().expect("nonempty") {
                return Err(HekitError::InvalidParams(
                    "layer shapes do not chain".into(),
                ));
            }
            dims.push(w.nrows());
        }
        check_dims(&dims, layout)?;
        let params = encoder.params();
        if level > params.max_level() {
            return Err(HekitError::InvalidParams(format!(
                "level {level} above max_level {}",
                params.max_level()
            )));
        }
        let delta = params.scale();
        let probe = ScaleProbe::new(ev);
        let (start, mut scale) = (level, delta);
        let mut level = start;
        let mut out = Vec::with_capacity(layers.len());
        for ((w, b), act) in layers.iter().zip(acts) {
            let dense = Dense::encrypt_at(w, b, layout, scale, start, enc, encoder, rng)?
                .upload_owned(ev)?;
            let (l, s) = dense_out(ev, level, scale, dense.w.scale())?;
            (level, scale) = probe.activation(act.as_ref(), encoder, l, s)?;
            out.push((dense, act));
        }
        Ok(Mlp { layers: out })
    }

    /// The dimensions `[d_0, d_1, …, d_L]`.
    pub fn dims(&self) -> Vec<usize> {
        let mut dims: Vec<usize> = self
            .layers
            .iter()
            .map(|(d, _)| d.w.d_in())
            .take(1)
            .collect();
        dims.extend(self.layers.iter().map(|(d, _)| d.w.d_out()));
        dims
    }

    /// Where the client encrypts batches for this network ([`BatchSpec`]): `x` on the top
    /// level at `Delta`, `y` at the output's exact level and scale: [`Mlp::batch_spec_at`] at
    /// `max_level` (the interactive client refresh's level).
    ///
    /// Errors as [`Mlp::batch_spec_at`].
    pub fn batch_spec(&self, ev: &E, encoder: &Encoder) -> Result<BatchSpec> {
        self.batch_spec_at(ev, encoder, ev.params().max_level())
    }

    /// [`Mlp::batch_spec`] for inputs at `x_level` (the level of the refreshed weights,
    /// [`crate::refresh::Refresher::output_level`]), computed from the public parameters, the
    /// weights' scales and the activations (zero-ciphertext probe, see [`Mlp::encrypt_at`];
    /// the probe shares `ev`'s keys, so this is cheap).
    ///
    /// Errors: `InvalidParams` for an empty network or `x_level > max_level`; `NoLevelsLeft`
    /// if the forward does not fit the levels; the activations' errors.
    pub fn batch_spec_at(&self, ev: &E, encoder: &Encoder, x_level: usize) -> Result<BatchSpec> {
        if self.layers.is_empty() {
            return Err(HekitError::InvalidParams("empty network".into()));
        }
        let params = ev.params();
        if x_level > params.max_level() {
            return Err(HekitError::InvalidParams(format!(
                "level {x_level} above max_level {}",
                params.max_level()
            )));
        }
        let probe = ScaleProbe::new(ev);
        let x_scale = params.scale();
        let (mut level, mut scale) = (x_level, x_scale);
        for (dense, act) in &self.layers {
            let (l, s) = dense_out(ev, level, scale, dense.w.scale())?;
            (level, scale) = probe.activation(act.as_ref(), encoder, l, s)?;
        }
        Ok(BatchSpec {
            x_level,
            x_scale,
            y_level: level,
            y_scale: scale,
        })
    }

    /// [`Mlp::predict_plain`] for a network on any backend: every layer is downloaded
    /// ([`Dense::download`]) and evaluated on the host.
    ///
    /// Errors: those of the download.
    ///
    /// # Panics
    /// As [`Mlp::predict_plain`].
    pub fn predict_plain_on(
        &self,
        ev: &E,
        dec: &Decryptor,
        encoder: &Encoder,
        layout: &Layout,
        x: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let host = self
            .layers
            .iter()
            .map(|(d, _)| d.download(ev))
            .collect::<Result<Vec<_>>>()?;
        Ok(predict_layers(
            host.iter()
                .zip(&self.layers)
                .map(|(d, (_, a))| (d, a.as_ref())),
            dec,
            encoder,
            layout,
            x,
        ))
    }
}

impl Mlp {
    /// Plaintext inference with the decrypted weights (for evaluation): decrypts every layer
    /// (weights from their diagonals, the bias from lane 0) and runs
    /// `a = act(a W^T + b)` row by row with [`Activation::eval_plain`]. `x` is `n x d_0`; the
    /// result is `n x d_L`. For a network on another backend, [`Mlp::predict_plain_on`].
    ///
    /// # Panics
    /// If `x` does not have `d_0` columns, `layout` is not the network's, or the keys are not
    /// for these parameters.
    pub fn predict_plain(
        &self,
        dec: &Decryptor,
        encoder: &Encoder,
        layout: &Layout,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        predict_layers(
            self.layers.iter().map(|(d, a)| (d, a.as_ref())),
            dec,
            encoder,
            layout,
            x,
        )
    }
}

/// `a = act(a W^T + b)` over host layers (see [`Mlp::predict_plain`]).
pub(crate) fn predict_layers<'l, E: HeOps + 'l>(
    layers: impl Iterator<Item = (&'l Dense, &'l dyn Activation<E>)>,
    dec: &Decryptor,
    encoder: &Encoder,
    layout: &Layout,
    x: &Array2<f64>,
) -> Array2<f64> {
    let mut a = x.clone();
    for (dense, act) in layers {
        let w = dense.w.decrypt(dec, encoder, layout);
        assert_eq!(a.ncols(), w.ncols(), "input width");
        let b = unpack_activations(&encoder.decode(&dec.decrypt(&dense.b)), layout, w.nrows());
        let mut z = a.dot(&w.t());
        for mut row in z.rows_mut() {
            row.iter_mut()
                .zip(b.row(0))
                .for_each(|(v, bj)| *v = act.eval_plain(*v + bj));
        }
        a = z;
    }
    a
}

// ---------------------------------------------------------------------------------------------
// The trainer.
// ---------------------------------------------------------------------------------------------

/// What one epoch cost ([`Trainer::train_epoch`]).
#[derive(Debug, Clone, PartialEq)]
pub struct EpochLog {
    /// The epoch index (0-based, the position in [`Trainer::log`]).
    pub epoch: usize,
    /// The evaluator's operations during the epoch (the difference of its counter; the
    /// client's decryptions and encryptions are not evaluator operations).
    pub counter: OpCounter,
    /// Wall-clock seconds of the epoch, refreshes included.
    pub seconds: f64,
    /// Wall-clock seconds spent in the refreshes (client side in interactive mode).
    pub refresh_seconds: f64,
    /// The training loss, if known. The server cannot compute it (it never sees plaintexts):
    /// `None` from the trainer; a client may fill it in (e.g. from [`Mlp::predict_plain`]).
    pub loss: Option<f64>,
    /// Batches trained.
    pub batches: usize,
    /// Ciphertexts refreshed (weights, biases, optimizer states).
    pub refreshes: usize,
}

/// Mini-batch training of an encrypted [`Mlp`] with one [`LayerOptimizer`] per layer and a
/// [`Refresher`] that restores the levels after every update (see the [module docs](self)).
///
/// `E` is the evaluator ([`Evaluator`] by default): the network, the refresher and the
/// optimizers work on its ciphertexts.
pub struct Trainer<'a, E: HeOps = Evaluator> {
    ev: &'a E,
    refresher: Box<dyn Refresher<E> + 'a>,
    optims: Vec<Box<dyn LayerOptimizer<E> + 'a>>,
    layout: Layout,
    transpose_keys: TransposeKeys,
    encoder: Encoder,
    trace: Vec<(String, usize)>,
    refreshes: usize,
    refresh_seconds: f64,
    frozen_biases: bool,
    /// One entry per finished epoch.
    pub log: Vec<EpochLog>,
}

/// `after - before`, field by field.
fn counter_delta(before: &OpCounter, after: &OpCounter) -> OpCounter {
    after - before
}

/// [`Dense::bias_grad_scaled`]; if the bias's scale is not reachable by a rescale from
/// `dz`'s level (rare, see there), retried once from one level lower.
fn bias_grad<E: HeOps>(
    ev: &E,
    dense: &Dense<E::Ct>,
    dz: &E::Ct,
    layout: &Layout,
    c: f64,
) -> Result<E::Ct> {
    match dense.bias_grad_scaled(ev, dz, layout, c) {
        Err(HekitError::ScaleMismatch) if dz.level() >= 2 => {
            let lower = ev.mod_switch_to_level(dz, dz.level() - 1)?;
            dense.bias_grad_scaled(ev, &lower, layout, c)
        }
        r => r,
    }
}

/// `dz = delta * (k * valid)` for an activation with the constant derivative `k`
/// ([`Activation::constant_derivative`], e.g. the identity): `delta` itself when `k = 1` and
/// every lane is valid (no op, no level); otherwise one plaintext multiply at `delta`'s exact
/// scale and a rescale (one level, no relinearization).
pub(crate) fn constant_dz<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    delta: &E::Ct,
    k: f64,
    mask: &[f64],
    full: bool,
) -> Result<E::Ct> {
    if full && k == 1.0 {
        return Ok(delta.clone());
    }
    let target = delta.scale();
    let v: Vec<f64> = mask.iter().map(|m| k * m).collect();
    let pt = ev.upload_pt_owned(enc.encode(
        &v,
        ev.target_plain_scale(delta, target)?,
        delta.level(),
    )?)?;
    ev.rescale(&ev.mul_plain_to_scale(delta, &pt, target)?)
}

impl<'a, E: HeOps> Trainer<'a, E> {
    /// A trainer on `ev` (which must hold the relinearization key and the Galois keys of
    /// [`mlp_rotation_steps`]) with one optimizer per layer, in `layout`, with the
    /// [`TransposeKeys::Dedicated`] transpose ([`Trainer::with_transpose_keys`] to change it).
    /// Binds `refresher` to `layout` ([`Refresher::bind_layout`]), so every refreshed weight,
    /// bias and optimizer state is re-laid in the canonical form of the layout (Task 31: the
    /// replicated block copies drift apart otherwise).
    ///
    /// Errors: `InvalidParams` if `layout` does not have the parameters' slot count; and the
    /// refresher's [`Refresher::bind_layout`] errors.
    pub fn new(
        ev: &'a E,
        mut refresher: Box<dyn Refresher<E> + 'a>,
        optims: Vec<Box<dyn LayerOptimizer<E> + 'a>>,
        layout: Layout,
    ) -> Result<Trainer<'a, E>> {
        if layout.slots != ev.params().slots() {
            return Err(HekitError::InvalidParams(format!(
                "layout of {} slots for parameters with {} slots",
                layout.slots,
                ev.params().slots()
            )));
        }
        refresher.bind_layout(&layout)?;
        Ok(Trainer {
            ev,
            refresher,
            optims,
            layout,
            transpose_keys: TransposeKeys::Dedicated,
            encoder: Encoder::new(Arc::clone(ev.params())),
            trace: Vec::new(),
            refreshes: 0,
            refresh_seconds: 0.0,
            frozen_biases: false,
            log: Vec::new(),
        })
    }

    /// Uses `keys` for the transposes (e.g. [`TransposeKeys::Reuse`]`(`[`mlp_key_pool`]`)`
    /// when key memory binds).
    pub fn with_transpose_keys(mut self, keys: TransposeKeys) -> Trainer<'a, E> {
        self.transpose_keys = keys;
        self
    }

    /// Sets the rate of subsequent updates without clearing optimizer state. Every layer
    /// is validated before any rate is committed: malformed rates, NAG reference ceilings,
    /// or unsupported custom optimizers leave the entire rate set unchanged. NAG reference
    /// maxima must be configured on each optimizer before constructing the trainer.
    pub fn set_lr(&mut self, lr: f64) -> Result<()> {
        check_lr(lr)?;
        for opt in &self.optims {
            opt.validate_lr(lr)?;
        }
        for opt in &mut self.optims {
            opt.apply_lr(lr);
        }
        Ok(())
    }

    /// Keeps the initialized bias values by folding a zero coefficient into their gradient.
    /// Initialize biases as zero for a bias-free profile. Like NI, the bias operations and
    /// refreshes remain, preserving counts/levels; decoded values are approximate CKKS.
    /// The default is to train biases using the optimizer's bias-gradient scale.
    pub fn with_frozen_biases(mut self) -> Trainer<'a, E> {
        self.frozen_biases = true;
        self
    }

    /// The layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The levels of the last successful step **before** the refresh, labelled as
    /// [`crate::sim::SimStep::depths`] (`z{l}`, `a{l}`, `delta{L}`, `s{l}`, `dz{l}`,
    /// `delta{l-1}`, `W{l}'`, `b{l}'`; 1-based layers). On seal_2020's Task 30 network every
    /// entry equals `max_level - depth`, and `W1'` is on level 0.
    pub fn trace(&self) -> &[(String, usize)] {
        &self.trace
    }

    /// The level of every weight, bias and optimizer state after a step: the refresher's
    /// [`Refresher::output_level`] (`max_level` for the interactive client refresh).
    pub fn refresh_level(&self) -> usize {
        self.refresher.output_level()
    }

    /// Ciphertexts refreshed so far.
    pub fn refreshes(&self) -> usize {
        self.refreshes
    }

    /// One training step on a full batch: [`Trainer::train_batch_valid`] with
    /// `n_valid = layout.batch`.
    ///
    /// Errors as [`Trainer::train_batch_valid`].
    pub fn train_batch(&mut self, mlp: &mut Mlp<E>, x: &E::Ct, y: &E::Ct) -> Result<()> {
        self.train_batch_valid(mlp, x, y, self.layout.batch)
    }

    /// One training step (see the [module docs](self)) on a batch whose first `n_valid` lanes
    /// are samples: `x` and `y` encrypted as [`BatchSpec::encrypt`] does. The gradients are
    /// means over the `n_valid` samples; the padded lanes are masked out of the derivative
    /// factor (0 extra levels). Afterwards every weight, bias and optimizer state is on
    /// [`Trainer::refresh_level`] at its old exact scale.
    ///
    /// Input and gradient errors, before network/optimizer writes: `InvalidParams` if the
    /// network is empty, there is not one optimizer per
    /// layer, the shapes do not chain, `n_valid` is not in `1..=batch`, or a key is missing;
    /// `ScaleMismatch` if `y` is not at the output's exact scale; `NoLevelsLeft` if the step
    /// does not fit the levels; and the errors of the layer operations. NAG configuration
    /// errors are checked before forward work. An optimizer failure may leave earlier
    /// optimizer states updated (the network is committed only after all optimizers succeed).
    /// A subsequent refresher failure may leave the updated network partially refreshed.
    pub fn train_batch_valid(
        &mut self,
        mlp: &mut Mlp<E>,
        x: &E::Ct,
        y: &E::Ct,
        n_valid: usize,
    ) -> Result<()> {
        self.train_batch_valid_with_hidden(mlp, x, y, n_valid)
            .map(|_| ())
    }

    /// [`Trainer::train_batch_valid`], returning the step's hidden activations: the outputs
    /// of the activations of layers `1 .. L - 1` (1-based; the inputs of layers `2 .. L`),
    /// computed by the forward pass from the weights **before** the update, at their own
    /// levels and scales. A block of [`crate::block::BlockMlp`] hands the first one to the
    /// next stage (Task 52).
    ///
    /// Errors as [`Trainer::train_batch_valid`].
    pub fn train_batch_valid_with_hidden(
        &mut self,
        mlp: &mut Mlp<E>,
        x: &E::Ct,
        y: &E::Ct,
        n_valid: usize,
    ) -> Result<Vec<E::Ct>> {
        let n_layers = mlp.layers.len();
        if n_layers == 0 || self.optims.len() != n_layers {
            return Err(HekitError::InvalidParams(format!(
                "{n_layers} layers and {} optimizers",
                self.optims.len()
            )));
        }
        if n_valid == 0 || n_valid > self.layout.batch {
            return Err(HekitError::InvalidParams(format!(
                "{n_valid} valid samples for a batch of {}",
                self.layout.batch
            )));
        }
        if mlp
            .layers
            .windows(2)
            .any(|p| p[1].0.w.d_in() != p[0].0.w.d_out())
        {
            return Err(HekitError::InvalidParams(
                "layer shapes do not chain".into(),
            ));
        }
        // Validate all fixed-basis configurations before any forward work. The optional
        // extra gradient is computed at the same level as the current-rate gradient.
        let reference_scales = self
            .optims
            .iter()
            .map(|opt| opt.reference_grad_scale())
            .collect::<Result<Vec<_>>>()?;
        let ev = self.ev;
        let enc = &self.encoder;
        let layout = &self.layout;
        let mut trace: Vec<(String, usize)> = Vec::new();

        // 1. Forward. acts[i] is the input of layer i.
        let mut acts = vec![x.clone()];
        let mut zs = Vec::with_capacity(n_layers);
        for (i, (dense, act)) in mlp.layers.iter().enumerate() {
            let z = dense.forward(ev, &acts[i], layout)?;
            trace.push((format!("z{}", i + 1), z.level()));
            let a = act.forward(ev, enc, &z)?;
            trace.push((format!("a{}", i + 1), a.level()));
            zs.push(z);
            acts.push(a);
        }

        // 2. Loss gradient.
        let mut delta = Mse.grad(ev, &acts[n_layers], y)?;
        trace.push((format!("delta{n_layers}"), delta.level()));

        // 3. Backward and gradients, top down; nothing is written yet.
        let mask = valid_lane_mask(layout, n_valid);
        let n = n_valid as f64;
        let mut updates = Vec::with_capacity(n_layers);
        for i in (0..n_layers).rev() {
            let ln = i + 1;
            let (dense, act) = &mlp.layers[i];
            let dz = match act.constant_derivative() {
                Some(k) => constant_dz(ev, enc, &delta, k, &mask, n_valid == layout.batch)?,
                None => {
                    let s = act.derivative_factor_at(ev, enc, &zs[i], 1.0, Some(&mask), None)?;
                    trace.push((format!("s{ln}"), s.level()));
                    let lv = s.level().min(delta.level());
                    ev.rescale(&ev.mul(
                        &ev.mod_switch_to_level(&delta, lv)?,
                        &ev.mod_switch_to_level(&s, lv)?,
                    )?)?
                }
            };
            trace.push((format!("dz{ln}"), dz.level()));
            let next = if i > 0 {
                let w = &dense.w;
                let low = EncWeights::from_diagonals(
                    w.diagonals().mod_switch_to_level(ev, dz.level())?,
                    w.d_out(),
                    w.d_in(),
                    layout,
                )?;
                let wt = low.transpose_with(ev, layout, &self.transpose_keys)?;
                let v = dense.backward_input(ev, &dz, &wt, layout)?;
                trace.push((format!("delta{i}"), v.level()));
                Some(v)
            } else {
                None
            };
            let opt = &self.optims[i];
            let (dw, reference) = if let Some(c_ref) = reference_scales[i] {
                let mut gs = dense.weight_grad_scaled_consts(
                    ev,
                    &dz,
                    &acts[i],
                    layout,
                    &[opt.grad_scale() / n, c_ref / n],
                )?;
                let reference = gs.pop().expect("two gradient constants");
                (gs.pop().expect("two gradient constants"), Some(reference))
            } else {
                (
                    dense.weight_grad_scaled(ev, &dz, &acts[i], layout, opt.grad_scale() / n)?,
                    None,
                )
            };
            let c_b = if self.frozen_biases {
                0.0
            } else {
                opt.bias_grad_scale() / n
            };
            let db = bias_grad(ev, dense, &dz, layout, c_b)?;
            let b_new = ev.sub(&ev.mod_switch_to_level(&dense.b, db.level())?, &db)?;
            updates.push((i, dw, reference, b_new));
            if let Some(v) = next {
                delta = v;
            }
        }

        // 4. Optimizer steps (on copies), then commit.
        let mut new_layers = Vec::with_capacity(n_layers);
        for (i, dw, reference, b_new) in updates {
            let mut w = mlp.layers[i].0.w.clone();
            self.optims[i].step_with_reference_gradient(ev, &mut w, &dw, reference.as_ref())?;
            trace.push((format!("W{}'", i + 1), w.level()));
            trace.push((format!("b{}'", i + 1), b_new.level()));
            new_layers.push((i, w, b_new));
        }
        for (i, w, b) in new_layers {
            mlp.layers[i].0.w = w;
            mlp.layers[i].0.b = b;
        }
        self.trace = trace;

        // 5. Refresh everything the next step consumes.
        self.refresh_all(mlp)?;
        acts.truncate(n_layers);
        Ok(acts.split_off(1))
    }

    /// Refreshes every weight diagonal, bias and optimizer state.
    fn refresh_all(&mut self, mlp: &mut Mlp<E>) -> Result<()> {
        let t0 = Instant::now();
        let r = self.refresher.as_mut();
        let mut count = 0;
        let result = (|| -> Result<()> {
            for ((dense, _), opt) in mlp.layers.iter_mut().zip(self.optims.iter_mut()) {
                let w = &dense.w;
                let entries = w
                    .diagonals()
                    .entries()
                    .iter()
                    .map(|c| r.refresh(c))
                    .collect::<Result<Vec<_>>>()?;
                count += entries.len();
                let diags =
                    Diagonals::from_prerotated(w.diagonals().d(), w.diagonals().stride(), entries)?;
                dense.w = EncWeights::from_diagonals(diags, w.d_out(), w.d_in(), &self.layout)?;
                dense.b = r.refresh(&dense.b)?;
                count += 1;
                count += opt.refresh_state(r)?;
            }
            Ok(())
        })();
        self.refreshes += count;
        self.refresh_seconds += t0.elapsed().as_secs_f64();
        result
    }

    /// One epoch over full batches: [`Trainer::train_batch`] on each `(x, y)` in order, then
    /// an [`EpochLog`] appended to [`Trainer::log`] and returned.
    ///
    /// Errors: the first error of a batch (the epoch is not logged; the batches before it
    /// stay trained).
    pub fn train_epoch(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(E::Ct, E::Ct)],
    ) -> Result<EpochLog> {
        let b = self.layout.batch;
        self.run_epoch(mlp, batches.iter().map(|(x, y)| (x, y, b)))
    }

    /// [`Trainer::train_epoch`] with the valid sample count of each batch
    /// ([`Trainer::train_batch_valid`]; e.g. a last, partial batch).
    ///
    /// Errors as [`Trainer::train_epoch`].
    pub fn train_epoch_valid(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(E::Ct, E::Ct, usize)],
    ) -> Result<EpochLog> {
        self.run_epoch(mlp, batches.iter().map(|(x, y, n)| (x, y, *n)))
    }

    fn run_epoch<'b>(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: impl Iterator<Item = (&'b E::Ct, &'b E::Ct, usize)>,
    ) -> Result<EpochLog>
    where
        E::Ct: 'b,
    {
        let before = self.ev.counter();
        let (r0, rs0) = (self.refreshes, self.refresh_seconds);
        let t0 = Instant::now();
        let mut count = 0;
        for (x, y, n) in batches {
            self.train_batch_valid(mlp, x, y, n)?;
            count += 1;
        }
        let log = EpochLog {
            epoch: self.log.len(),
            counter: counter_delta(&before, &self.ev.counter()),
            seconds: t0.elapsed().as_secs_f64(),
            refresh_seconds: self.refresh_seconds - rs0,
            loss: None,
            batches: count,
            refreshes: self.refreshes - r0,
        };
        self.log.push(log.clone());
        Ok(log)
    }
}
