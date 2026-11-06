//! ReBoot's eMLP-2/3 block structure (Task 52; ReBoot's structure as data, R-REB-5,
//! `results/quadruped/raw/reboot_emlp23_full/config_values.json`).
//!
//! # Structure
//!
//! A [`BlockMlp`] is a list of **local blocks** followed by a separate **output stack**:
//!
//! - a local block is a linear layer that sends no gradient below itself, then `z^2 + z`,
//!   then a local head (linear, `width -> 16`). The block is trained by its own squared-error
//!   loss (`delta = head output - label`) and by nothing else;
//! - the output stack is linear, `z^2 + z`, linear (`-> 16`), trained by the main loss; its
//!   first linear sends no gradient down either;
//! - the activation of every block's first layer is the next stage's input. Losses are never
//!   summed: every layer has its own optimizer (NAG at the same lr in ReBoot), updated from
//!   its own stage's loss only;
//! - test time uses the output stack ([`BlockMlp::predict_plain`]).
//!
//! eMLP-2 is block `196 -> 64` (head `64 -> 16`) and the output stack `64 -> 32 -> 16`;
//! eMLP-3 is blocks `196 -> 128` and `128 -> 64` (heads `-> 16`) and the same output stack.
//!
//! Every stage (block or output stack) is therefore a two-layer [`Mlp`] trained on its own
//! loss with no backward pass below its first layer, which is exactly what the existing
//! trainers do for an [`Mlp`] (the first layer has no backward pass). [`BlockMlp`] holds one
//! [`Mlp`] per stage, and the block trainers ([`BlockTrainer`], interactive;
//! [`BlockNiTrainer`], non-interactive) run one [`Trainer`] / [`NonInteractiveTrainer`] per
//! stage, with its own optimizers and state.
//!
//! # One step
//!
//! For each stage `s` in forward order: train the stage on its input and its labels, taking
//! the hidden activation `a_s` of its first layer from the stage's forward pass (computed from
//! the weights **before** the update); hand `a_s` to stage `s + 1`. ReBoot runs the whole
//! forward first, then the backward of the output stack, then the blocks from the last to the
//! first. The two orders give the same update: no stage reads another stage's new weights (the
//! inputs are all computed from the old ones) and no gradient crosses a stage boundary.
//!
//! **The hand-off.** `a_s` is on a low level at the product scale of its history. The next
//! stage expects its input like a batch: at `Delta` on its trainer's start level (its biases
//! are encrypted at the product scale of a `Delta` input). So `a_s` is multiplied by 1 onto
//! exactly `Delta` (one ct-pt, one level, no key switch) and then refreshed: by the client in
//! interactive mode (any [`Refresher`], e.g. [`crate::refresh::ClientRefresher`]), by one
//! bootstrap per micro-batch ciphertext in non-interactive mode. With
//! [`BootPlacement::AfterActivation`], the stage already produced this canonical-Delta
//! bootstrap output: the hand-off reuses it without another bootstrap or scale conversion.
//! The next stage's forward pass mod-switches its input to the actual layer level as needed.
//! The bootstrap's precondition
//! `|m| <= 1` applies to the activation (as for [`BootPlacement::AfterActivation`]); it is not
//! clipped.
//!
//! # RE / CE
//!
//! ReBoot's encrypted model alternates row-packed (RE) and column-packed (CE) layers along
//! the chain (eMLP-2: RE(64) -> CE(32)). hekit's layout is one diagonal, batched layout for
//! every layer (`docs/layout.md`): every linear layer is the same diagonal matvec, and the
//! backward pass transposes by rotations. A layer's position in the chain changes nothing in
//! its packing, so RE vs CE has no counterpart here; the math (which products, which losses,
//! which gradients) is the same.
//!
//! # Levels and bootstraps (non-interactive)
//!
//! Each stage is a separate [`NonInteractiveTrainer`] with its own compact state and its own
//! bootstrap schedule. A stage's step starts from its hand-off at the bootstrap output level,
//! so it needs the levels of a two-layer network ([`crate::trainer_ni::levels_per_iter`], 7 for
//! `z^2 + z` then linear with WhenNeeded/AfterUpdate, 5 with AA), whatever its position in the
//! chain. Additional hand-off bootstraps are `(stages - 1) * k` per update for `k` micro-batches
//! with [`BootPlacement::WhenNeeded`] or [`BootPlacement::AfterUpdate`], and zero with
//! [`BootPlacement::AfterActivation`], whose stage activation bootstraps are counted once.
//! Mixed schedules choose this behavior per source stage at each boundary; the closed-form
//! predictor takes one placement and describes homogeneous schedules only.
//! Chaining the stages without them does not fit the lazy schedule: the input of stage `s`
//! sits 2 levels per stage below the batch, so every later stage would need its weights and
//! its predecessors' bootstrapped every update ([`predict_block_update_with`] gives the
//! closed form of the chosen design).

use std::time::Instant;

use ckks_train_ckks::bootstrap::Bootstrapper;
use ckks_train_ckks::{BootstrapOps, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, HeOps};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::sampler::Rng;
use ndarray::Array2;

use crate::activation::Activation;
use crate::dense::TransposeKeys;
use crate::layout::Layout;
use crate::refresh::Refresher;
use crate::trainer::{Mlp, Trainer, mlp_rotation_steps, predict_layers, sorted};
use crate::trainer_ni::{
    ActCost, BootPlacement, CompactState, NonInteractiveTrainer, UpdateCost, ni_rotation_steps,
    predict_update_with,
};

/// One micro-batch of a non-interactive block update: the input `x`, one label ciphertext per
/// stage (`ys[s]` at stage `s`'s [`NonInteractiveTrainer::batch_spec`]) and the valid sample
/// count.
pub type BlockMicroBatch<'b, C> = (&'b C, &'b [&'b C], usize);

// ---------------------------------------------------------------------------------------------
// The network.
// ---------------------------------------------------------------------------------------------

/// Local blocks plus a separate output stack (see the [module docs](self)). Every stage is an
/// [`Mlp`]: a block has at least two layers (its first layer, whose activation feeds the next
/// stage, then its local head), the output stack at least one.
pub struct BlockMlp<E: HeOps = Evaluator> {
    /// The local blocks in forward order.
    pub blocks: Vec<Mlp<E>>,
    /// The output stack (the model output; test time uses it).
    pub output: Mlp<E>,
}

/// The dimensions of a stage, `InvalidParams` if its layers do not chain.
fn stage_dims<E: HeOps>(m: &Mlp<E>) -> Result<Vec<usize>> {
    if m.layers.is_empty() {
        return Err(HekitError::InvalidParams("empty stage".into()));
    }
    if m.layers
        .windows(2)
        .any(|p| p[1].0.w.d_in() != p[0].0.w.d_out())
    {
        return Err(HekitError::InvalidParams(
            "stage layer shapes do not chain".into(),
        ));
    }
    Ok(m.dims())
}

impl<E: HeOps> BlockMlp<E> {
    /// The network of `blocks` and `output`.
    ///
    /// Errors (`InvalidParams`): a block with fewer than two layers (no head), an empty output
    /// stack, a stage whose layers do not chain, or a stage whose input is not the width of the
    /// previous block's first layer.
    pub fn new(blocks: Vec<Mlp<E>>, output: Mlp<E>) -> Result<BlockMlp<E>> {
        let mut prev: Option<usize> = None;
        for (i, b) in blocks.iter().enumerate() {
            if b.layers.len() < 2 {
                return Err(HekitError::InvalidParams(format!(
                    "block {i} has {} layer(s): a block is a layer and a local head",
                    b.layers.len()
                )));
            }
            let d = stage_dims(b)?;
            if prev.is_some_and(|p| p != d[0]) {
                return Err(HekitError::InvalidParams(format!(
                    "block {i} takes {} inputs, the previous block's width is {}",
                    d[0],
                    prev.expect("some")
                )));
            }
            prev = Some(d[1]);
        }
        let d = stage_dims(&output)?;
        if prev.is_some_and(|p| p != d[0]) {
            return Err(HekitError::InvalidParams(format!(
                "the output stack takes {} inputs, the last block's width is {}",
                d[0],
                prev.expect("some")
            )));
        }
        Ok(BlockMlp { blocks, output })
    }

    /// Encrypts the plaintext stages (`stages[s]` = `(W_l, b_l)` per layer, the last stage the
    /// output stack) with [`Mlp::encrypt_at`] at `level` (every stage's input arrives at
    /// `level` at `Delta`: the batch, or a refreshed hand-off).
    ///
    /// Errors: `InvalidParams` if `stages` is empty or `acts` has another length; those of
    /// [`Mlp::encrypt_at`] and [`BlockMlp::new`].
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_at(
        ev: &E,
        stages: &[Vec<(Array2<f64>, Vec<f64>)>],
        acts: Vec<Vec<Box<dyn Activation<E>>>>,
        layout: &Layout,
        level: usize,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<BlockMlp<E>> {
        if stages.is_empty() || acts.len() != stages.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} stages and {} activation lists",
                stages.len(),
                acts.len()
            )));
        }
        let mut mlps = stages
            .iter()
            .zip(acts)
            .map(|(st, a)| Mlp::encrypt_at(ev, st, a, layout, level, enc, encoder, rng))
            .collect::<Result<Vec<_>>>()?;
        let output = mlps.pop().expect("nonempty");
        BlockMlp::new(mlps, output)
    }

    /// Blocks plus the output stack.
    pub fn num_stages(&self) -> usize {
        self.blocks.len() + 1
    }

    /// Stage `s`: block `s`, or the output stack for `s = blocks.len()`.
    ///
    /// # Panics
    /// If `s >= num_stages()`.
    pub fn stage(&self, s: usize) -> &Mlp<E> {
        if s == self.blocks.len() {
            &self.output
        } else {
            &self.blocks[s]
        }
    }

    /// [`BlockMlp::stage`], mutable.
    ///
    /// # Panics
    /// If `s >= num_stages()`.
    pub fn stage_mut(&mut self, s: usize) -> &mut Mlp<E> {
        if s == self.blocks.len() {
            &mut self.output
        } else {
            &mut self.blocks[s]
        }
    }

    /// Every stage's dimensions ([`Mlp::dims`]), blocks first.
    pub fn stage_dims(&self) -> Vec<Vec<usize>> {
        (0..self.num_stages())
            .map(|s| self.stage(s).dims())
            .collect()
    }

    /// One interactive training step on a full batch: [`BlockTrainer::train_batch_valid`] with
    /// `n_valid = layout.batch`. `ys[s]` are stage `s`'s labels (in ReBoot all equal: the
    /// one-hot labels), each encrypted at the stage's [`Mlp::batch_spec`].
    ///
    /// Errors as [`BlockTrainer::train_batch_valid`].
    pub fn train_batch(
        &mut self,
        trainer: &mut BlockTrainer<'_, E>,
        x: &E::Ct,
        ys: &[&E::Ct],
    ) -> Result<()> {
        let b = trainer.layout().batch;
        trainer.train_batch_valid(self, x, ys, b)
    }

    /// [`BlockMlp::train_batch`] with `n_valid` samples.
    ///
    /// Errors as [`BlockTrainer::train_batch_valid`].
    pub fn train_batch_valid(
        &mut self,
        trainer: &mut BlockTrainer<'_, E>,
        x: &E::Ct,
        ys: &[&E::Ct],
        n_valid: usize,
    ) -> Result<()> {
        trainer.train_batch_valid(self, x, ys, n_valid)
    }

    /// One non-interactive update: [`BlockNiTrainer::train_step_valid`].
    ///
    /// Errors as [`BlockNiTrainer::train_step_valid`].
    pub fn train_step_ni<B: BootstrapOps<E>>(
        &mut self,
        trainer: &mut BlockNiTrainer<'_, E, B>,
        batches: &[BlockMicroBatch<'_, E::Ct>],
    ) -> Result<()> {
        trainer.train_step_valid(self, batches)
    }
}

impl BlockMlp {
    /// Plaintext inference with the decrypted weights: the input through the first layer of
    /// every block (`act(a W^T + b)`), then the whole output stack (the model output; the
    /// local heads are not used at test time). `x` is `n x d_0`.
    ///
    /// # Panics
    /// As [`Mlp::predict_plain`].
    pub fn predict_plain(
        &self,
        dec: &Decryptor,
        encoder: &Encoder,
        layout: &Layout,
        x: &Array2<f64>,
    ) -> Array2<f64> {
        let mut a = x.clone();
        for b in &self.blocks {
            let (d, act) = &b.layers[0];
            a = predict_layers(std::iter::once((d, act.as_ref())), dec, encoder, layout, &a);
        }
        self.output.predict_plain(dec, encoder, layout, &a)
    }
}

/// `a` multiplied by 1 onto exactly `Delta` (one ct-pt, one level): the hand-off's input scale.
fn to_delta<E: HeOps>(ev: &E, a: &E::Ct) -> Result<E::Ct> {
    ev.rescale(&ev.mul_scalar(a, 1.0, ev.params().scale())?)
}

/// `InvalidParams` unless there is one label ciphertext per stage and one trainer per stage.
fn check_counts(stages: usize, trainers: usize, ys: usize) -> Result<()> {
    if stages != trainers || ys != stages {
        return Err(HekitError::InvalidParams(format!(
            "{stages} stages, {trainers} stage trainers and {ys} label ciphertexts"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Galois keys.
// ---------------------------------------------------------------------------------------------

/// `InvalidParams` unless there is one transpose-key choice per stage.
fn check_keys(stage_dims: &[Vec<usize>], keys: &[TransposeKeys]) -> Result<()> {
    if stage_dims.is_empty() || keys.len() != stage_dims.len() {
        return Err(HekitError::InvalidParams(format!(
            "{} stages and {} transpose-key choices",
            stage_dims.len(),
            keys.len()
        )));
    }
    Ok(())
}

/// The Galois steps of an interactive block step: the union of every stage's
/// [`mlp_rotation_steps`] with its transpose keys (the hand-off has no rotation).
///
/// Errors: `InvalidParams` for no stage or not one key choice per stage; those of
/// [`mlp_rotation_steps`].
pub fn block_rotation_steps(
    stage_dims: &[Vec<usize>],
    layout: &Layout,
    keys: &[TransposeKeys],
) -> Result<Vec<i32>> {
    check_keys(stage_dims, keys)?;
    let mut steps = Vec::new();
    for (d, k) in stage_dims.iter().zip(keys) {
        steps.extend(mlp_rotation_steps(d, layout, k)?);
    }
    Ok(sorted(steps))
}

/// The Galois steps of a non-interactive block update: the union of every stage's
/// [`ni_rotation_steps`], one per Galois element (the bootstrapper brings its own keys).
///
/// Errors: as [`block_rotation_steps`], with those of [`ni_rotation_steps`].
pub fn ni_block_rotation_steps(
    stage_dims: &[Vec<usize>],
    layout: &Layout,
    keys: &[TransposeKeys],
) -> Result<Vec<i32>> {
    check_keys(stage_dims, keys)?;
    let mut steps = Vec::new();
    for (d, k) in stage_dims.iter().zip(keys) {
        steps.extend(ni_rotation_steps(d, layout, k)?);
    }
    let s = layout.slots as i64;
    let mut seen = std::collections::HashSet::new();
    let mut out = sorted(steps);
    out.retain(|&k| seen.insert((k as i64).rem_euclid(s)));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// Interactive.
// ---------------------------------------------------------------------------------------------

/// Interactive block training: one [`Trainer`] per stage (its own optimizers and refresher)
/// and a [`Refresher`] for the hand-offs (see the [module docs](self)).
pub struct BlockTrainer<'a, E: HeOps = Evaluator> {
    ev: &'a E,
    stages: Vec<Trainer<'a, E>>,
    handoff: Box<dyn Refresher<E> + 'a>,
    layout: Layout,
    handoffs: usize,
}

impl<'a, E: HeOps> BlockTrainer<'a, E> {
    /// A block trainer from one stage trainer per stage (blocks, then the output stack) and the
    /// hand-off refresher, which is bound to the stages' layout.
    ///
    /// Errors (`InvalidParams`): no stage, stages with different layouts or refresh levels, a
    /// hand-off refresher whose output level is not the stages' refresh level (the hand-off is
    /// the next stage's input, which must arrive where a batch does); the hand-off's
    /// [`Refresher::bind_layout`] errors.
    pub fn new(
        ev: &'a E,
        stages: Vec<Trainer<'a, E>>,
        mut handoff: Box<dyn Refresher<E> + 'a>,
    ) -> Result<BlockTrainer<'a, E>> {
        let first = stages
            .first()
            .ok_or_else(|| HekitError::InvalidParams("no stage trainer".into()))?;
        let layout = *first.layout();
        let level = first.refresh_level();
        if stages
            .iter()
            .any(|t| *t.layout() != layout || t.refresh_level() != level)
        {
            return Err(HekitError::InvalidParams(
                "stage trainers differ in layout or refresh level".into(),
            ));
        }
        if handoff.output_level() != level {
            return Err(HekitError::InvalidParams(format!(
                "hand-off refresher outputs level {}, the stages start at {level}",
                handoff.output_level()
            )));
        }
        handoff.bind_layout(&layout)?;
        Ok(BlockTrainer {
            ev,
            stages,
            handoff,
            layout,
            handoffs: 0,
        })
    }

    /// The stage trainers (blocks, then the output stack).
    pub fn stages(&self) -> &[Trainer<'a, E>] {
        &self.stages
    }

    /// The layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Hand-offs refreshed so far (one per stage boundary per step).
    pub fn handoffs(&self) -> usize {
        self.handoffs
    }

    /// One step (see the [module docs](self)): every stage, in forward order, trains on its
    /// input and `ys[s]` with `n_valid` samples ([`Trainer::train_batch_valid_with_hidden`]),
    /// and hands its first layer's activation (from the old weights) to the next stage.
    ///
    /// Errors: `InvalidParams` before any work if `net` does not have one stage per trainer or
    /// `ys` one ciphertext per stage; then the stage trainers' errors (a stage's error leaves
    /// that stage unchanged, but the stages before it are already updated) and the hand-off's.
    pub fn train_batch_valid(
        &mut self,
        net: &mut BlockMlp<E>,
        x: &E::Ct,
        ys: &[&E::Ct],
        n_valid: usize,
    ) -> Result<()> {
        let n = net.num_stages();
        check_counts(n, self.stages.len(), ys.len())?;
        let mut input = x.clone();
        for (s, trainer) in self.stages.iter_mut().enumerate() {
            let hidden =
                trainer.train_batch_valid_with_hidden(net.stage_mut(s), &input, ys[s], n_valid)?;
            if s + 1 < n {
                let a = hidden.first().ok_or_else(|| {
                    HekitError::InvalidParams("a block needs a hidden activation".into())
                })?;
                input = self.handoff.refresh(&to_delta(self.ev, a)?)?;
                self.handoffs += 1;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Non-interactive.
// ---------------------------------------------------------------------------------------------

/// Non-interactive block training: one [`NonInteractiveTrainer`] per stage (its own compact
/// state, optimizer and bootstrap schedule). A stage boundary reuses an existing AA activation
/// bootstrap or performs one new bootstrap per micro-batch (see the [module docs](self)).
/// Holds no secret key.
pub struct BlockNiTrainer<'a, E: HeOps = Evaluator, B: BootstrapOps<E> = Bootstrapper> {
    ev: &'a E,
    bs: &'a B,
    stages: Vec<NonInteractiveTrainer<'a, E, B>>,
    handoff_boots: u64,
    handoff_seconds: f64,
    last_handoffs: Vec<Vec<E::Ct>>,
}

impl<'a, E: HeOps, B: BootstrapOps<E>> BlockNiTrainer<'a, E, B> {
    /// Encrypts (client side) every stage with [`NonInteractiveTrainer::encrypt_network`]
    /// (`stages[s]` = `(W_l, b_l)` per layer, the last stage the output stack): the
    /// [`BlockMlp`] at the start level and one [`CompactState`] per stage.
    ///
    /// Errors: `InvalidParams` if `stages` is empty or `acts` has another length; those of
    /// [`NonInteractiveTrainer::encrypt_network`] and [`BlockMlp::new`].
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn encrypt_network(
        ev: &E,
        bs: &B,
        stages: &[Vec<(Array2<f64>, Vec<f64>)>],
        acts: Vec<Vec<Box<dyn Activation<E>>>>,
        layout: &Layout,
        placement: BootPlacement,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<(BlockMlp<E>, Vec<CompactState<E::Ct>>)> {
        if stages.is_empty() || acts.len() != stages.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} stages and {} activation lists",
                stages.len(),
                acts.len()
            )));
        }
        let mut mlps = Vec::with_capacity(stages.len());
        let mut states = Vec::with_capacity(stages.len());
        for (st, a) in stages.iter().zip(acts) {
            let (m, s) = NonInteractiveTrainer::encrypt_network(
                ev, bs, st, a, layout, placement, enc, encoder, rng,
            )?;
            mlps.push(m);
            states.push(s);
        }
        let output = mlps.pop().expect("nonempty");
        Ok((BlockMlp::new(mlps, output)?, states))
    }

    /// A block trainer from one stage trainer per stage (blocks, then the output stack).
    ///
    /// Stages may use different placements: each boundary reuses its source stage's first
    /// activation bootstrap for [`BootPlacement::AfterActivation`], otherwise it performs a
    /// new bootstrap. The homogeneous [`predict_block_update_with`] formula does not describe
    /// mixed schedules; sum their per-stage costs and new non-AA boundary bootstraps instead.
    ///
    /// Errors (`InvalidParams`): no stage or stages with different layouts.
    pub fn new(
        ev: &'a E,
        bs: &'a B,
        stages: Vec<NonInteractiveTrainer<'a, E, B>>,
    ) -> Result<BlockNiTrainer<'a, E, B>> {
        let first = stages
            .first()
            .ok_or_else(|| HekitError::InvalidParams("no stage trainer".into()))?;
        let layout = *first.layout();
        if stages.iter().any(|t| *t.layout() != layout) {
            return Err(HekitError::InvalidParams(
                "stage trainers differ in layout".into(),
            ));
        }
        Ok(BlockNiTrainer {
            ev,
            bs,
            stages,
            handoff_boots: 0,
            handoff_seconds: 0.0,
            last_handoffs: Vec::new(),
        })
    }

    /// The stage trainers (blocks, then the output stack).
    pub fn stages(&self) -> &[NonInteractiveTrainer<'a, E, B>] {
        &self.stages
    }

    /// The stage trainers, mutable (e.g. [`NonInteractiveTrainer::with_frozen_biases`] is a
    /// builder; set such options before [`BlockNiTrainer::new`]).
    pub fn stages_mut(&mut self) -> &mut [NonInteractiveTrainer<'a, E, B>] {
        &mut self.stages
    }

    /// [`NonInteractiveTrainer::set_lr`] on every stage (ReBoot: one lr for every layer).
    ///
    /// Errors: the first stage's error (the stages before it keep the new lr).
    pub fn set_lr(&mut self, lr: f64) -> Result<()> {
        self.stages.iter_mut().try_for_each(|t| t.set_lr(lr))
    }

    /// [`NonInteractiveTrainer::set_lr_reference`] on every stage.
    ///
    /// Errors: the first stage's error.
    pub fn set_lr_reference(&mut self, lr_ref: f64) -> Result<()> {
        self.stages
            .iter_mut()
            .try_for_each(|t| t.set_lr_reference(lr_ref))
    }

    /// [`NonInteractiveTrainer::set_next_batch_full`] on every stage.
    pub fn set_next_batch_full(&mut self, full: Option<bool>) {
        self.stages
            .iter_mut()
            .for_each(|t| t.set_next_batch_full(full));
    }

    /// Newly performed hand-off bootstraps so far. Reusing a stage's AA output adds zero.
    pub fn handoff_bootstraps(&self) -> u64 {
        self.handoff_boots
    }

    /// The hand-off bootstraps of the last update: `[boundary][micro-batch]` is the bootstrap
    /// output of block `boundary`'s activation (stage `boundary + 1`'s input), at `Delta` on
    /// the bootstrap output level, canonical with the block's width. For a client-side
    /// diagnostic of the hand-off's precision (its input is not range-checked: the bootstrap's
    /// `|m| <= 1` precondition is the caller's; the trainer itself never decrypts), as
    /// [`NonInteractiveTrainer::last_activation_bootstraps`] for
    /// [`BootPlacement::AfterActivation`]. With AA, these are clones of the source stage's
    /// recorded first activation outputs, not additional bootstraps; they add neither hand-off
    /// count nor hand-off bootstrap time. Empty before the first update or after a failed one.
    pub fn last_handoff_bootstraps(&self) -> &[Vec<E::Ct>] {
        &self.last_handoffs
    }

    /// Wall-clock seconds of all bootstraps so far (the stages' and the hand-offs').
    pub fn bootstrap_seconds(&self) -> f64 {
        self.handoff_seconds
            + self
                .stages
                .iter()
                .map(|t| t.bootstrap_seconds())
                .sum::<f64>()
    }

    /// One update (see the [module docs](self)) on the micro-batches `(x, ys, n_valid)`, `ys[s]`
    /// stage `s`'s labels at that stage's [`NonInteractiveTrainer::batch_spec`]: every stage,
    /// in forward order, takes its update
    /// ([`NonInteractiveTrainer::train_step_valid_with_hidden`]), then its first layer's
    /// activation of every micro-batch is passed as the next stage's input. AA reuses
    /// [`NonInteractiveTrainer::last_activation_bootstraps`] at `[micro][0]`; other placements
    /// switch the pre-bootstrap hidden value to `Delta` and perform a new counted bootstrap.
    /// The next stage handles any required level alignment in its forward pass.
    ///
    /// Errors: `InvalidParams` before any work if `batches` is empty, `net` does not have one
    /// stage per trainer or a micro-batch not one label ciphertext per stage; then the stage
    /// trainers' errors (the stages before a failing one are already updated) and the hand-off
    /// bootstrap's.
    pub fn train_step_valid(
        &mut self,
        net: &mut BlockMlp<E>,
        batches: &[BlockMicroBatch<'_, E::Ct>],
    ) -> Result<()> {
        let n = net.num_stages();
        if batches.is_empty() {
            return Err(HekitError::InvalidParams("no micro-batches".into()));
        }
        for b in batches {
            check_counts(n, self.stages.len(), b.1.len())?;
        }
        self.last_handoffs.clear();
        let mut handoffs = Vec::with_capacity(n - 1);
        let mut inputs: Vec<E::Ct> = batches.iter().map(|b| b.0.clone()).collect();
        for s in 0..n {
            let mb: Vec<(&E::Ct, &E::Ct, usize)> = inputs
                .iter()
                .zip(batches)
                .map(|(x, b)| (x, b.1[s], b.2))
                .collect();
            let hidden = self.stages[s].train_step_valid_with_hidden(net.stage_mut(s), &mb)?;
            if s + 1 < n {
                let next = if self.stages[s].placement() == BootPlacement::AfterActivation {
                    self.stages[s]
                        .last_activation_bootstraps()
                        .iter()
                        .map(|micro| {
                            micro.first().cloned().ok_or_else(|| {
                                HekitError::InvalidParams(
                                    "a block needs a hidden activation".into(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    let t0 = Instant::now();
                    let mut next = Vec::with_capacity(hidden.len());
                    for h in &hidden {
                        let a = h.first().ok_or_else(|| {
                            HekitError::InvalidParams("a block needs a hidden activation".into())
                        })?;
                        let low = self.ev.mod_switch_to_level(&to_delta(self.ev, a)?, 0)?;
                        next.push(self.bs.bootstrap(self.ev, &low)?);
                        self.handoff_boots += 1;
                    }
                    self.handoff_seconds += t0.elapsed().as_secs_f64();
                    next
                };
                handoffs.push(next.clone());
                inputs = next;
            }
        }
        self.last_handoffs = handoffs;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Closed form.
// ---------------------------------------------------------------------------------------------

/// The closed-form counts of one non-interactive block update ([`predict_block_update_with`]):
/// every stage's [`UpdateCost`] and the hand-off bootstraps.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockUpdateCost {
    /// One per stage (blocks, then the output stack).
    pub stages: Vec<UpdateCost>,
    /// Newly performed boundary bootstraps: zero for AA (reuse), `(stages - 1) * micro`
    /// for WhenNeeded/AfterUpdate. Existing AA activation bootstraps belong to `stages`.
    pub handoff_bootstraps: u64,
}

impl BlockUpdateCost {
    /// All rotations outside the bootstraps (the hand-off's switch to `Delta` is one ct-pt).
    pub fn rotations(&self) -> u64 {
        self.stages.iter().map(UpdateCost::rotations).sum()
    }

    /// All relinearizations outside the bootstraps.
    pub fn relins(&self) -> u64 {
        self.stages.iter().map(UpdateCost::relins).sum()
    }

    /// All bootstraps: the stages' (weights, velocities and activations) and new hand-offs.
    pub fn bootstraps(&self) -> u64 {
        self.stages.iter().map(UpdateCost::bootstraps).sum::<u64>() + self.handoff_bootstraps
    }

    /// The key switches with every bootstrap at `boot_ks`.
    pub fn key_switches(&self, boot_ks: u64) -> u64 {
        self.rotations() + self.relins() + self.bootstraps() * boot_ks
    }
}

/// The closed-form counts of one [`BlockNiTrainer::train_step_valid`] with `micro` micro-batch
/// ciphertexts: every stage `s` as [`predict_update_with`] (dimensions `stage_dims[s]`,
/// activations `stage_acts[s]`, transpose keys `keys[s]`, its weights bootstrapped iff
/// `booted[s]`) plus newly performed hand-off bootstraps: zero for AA (the stage's first
/// activation bootstrap is reused), `(stages - 1) * micro` for WhenNeeded/AfterUpdate.
/// All stages use `placement`; a mixed-placement trainer requires per-stage predictions
/// plus one new bootstrap per micro-batch at each non-AA source-stage boundary.
///
/// Errors: `InvalidParams` unless every per-stage slice has one entry per stage; those of
/// [`predict_update_with`].
#[allow(clippy::too_many_arguments)]
pub fn predict_block_update_with(
    stage_dims: &[Vec<usize>],
    stage_acts: &[Vec<ActCost>],
    layout: &Layout,
    keys: &[TransposeKeys],
    placement: BootPlacement,
    micro: usize,
    nag: bool,
    booted: &[bool],
    cfg: EvalConfig,
) -> Result<BlockUpdateCost> {
    check_keys(stage_dims, keys)?;
    if stage_acts.len() != stage_dims.len() || booted.len() != stage_dims.len() {
        return Err(HekitError::InvalidParams(format!(
            "{} stages, {} activation lists, {} bootstrap flags",
            stage_dims.len(),
            stage_acts.len(),
            booted.len()
        )));
    }
    let stages = stage_dims
        .iter()
        .zip(stage_acts)
        .zip(keys)
        .zip(booted)
        .map(|(((d, a), k), &b)| {
            predict_update_with(d, a, layout, k, placement, micro, nag, b, cfg)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BlockUpdateCost {
        handoff_bootstraps: if placement == BootPlacement::AfterActivation {
            0
        } else {
            (stage_dims.len() as u64 - 1) * micro as u64
        },
        stages,
    })
}
