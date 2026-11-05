//! Non-interactive encrypted training with CKKS bootstrapping (M3, Task 41;
//! `docs/layout.md` §6.4 and §7).
//!
//! [`NonInteractiveTrainer`] trains an [`Mlp`] with no client in the loop: the levels are
//! restored by bootstrapping ([`Bootstrapper`]), never by the key owner. The trainer holds an
//! [`Evaluator`] (public evaluation keys) and a [`Bootstrapper`] (its evaluation keys), and no
//! [`hekit_ckks::Decryptor`]: no secret key can be used during training, by construction of the
//! type.
//!
//! # The fused update path (ruling R-M3-11)
//!
//! The weights, the NAG velocity and the biases live between steps in **compact** form
//! ([`crate::compact`], `docs/layout.md` §6): the [`CompactState`], laid out by a
//! [`CompactLayout`]. One update (`k` micro-batches of `B` lanes, one ciphertext each):
//!
//! 1. forward and backward per micro-batch, as the interactive [`crate::trainer::Trainer`]
//!    (the transpose once per update);
//! 2. the weight gradient **reduce only** ([`crate::dense::Dense::weight_grad_reduced`]): the products of
//!    all micro-batches summed before one relinearization per diagonal, the batch reduce, and
//!    the first-period lane-0 mask with `c = grad_scale / n` folded in, which is also the pack
//!    mask; then the mask-free lane shift into compact form ([`compact_shift`]);
//! 3. the bias gradient likewise ([`crate::dense::Dense::bias_grad_reduced`]), shifted to the bias's compact
//!    position ([`BiasSlot`]);
//! 4. the optimizer on the persistent compact ciphertexts (NAG on `W` and `U`, plain SGD on the
//!    biases, see [`NiOptimizer`]);
//! 5. bootstrap every compact ciphertext (weights and velocity) when the placement says so
//!    ([`BootPlacement`]);
//! 6. unpack the weights only ([`compact_unpack`]) and extract the biases
//!    ([`compact_extract`]) for the next forward.
//!
//! This is the cost model's non-interactive compact path (`spikes/costmodel/costmodel.py`,
//! `_diag_batched`, `compact=True`): its gradient, lane shift, unpack and bootstrap counts are
//! reproduced exactly, and [`predict_update`] gives the closed form of every count with the
//! differences from the model listed term by term (`docs/layout.md` §7.5).
//!
//! # Biases
//!
//! Biases are bootstrapped inside the weight containers: each bias takes one free position of
//! a compact ciphertext whose period `L` is at least the bias's padded length (e.g. eMLP-1's
//! two biases sit in W2's container, which holds 16 of its 1024 positions). Only when no
//! container has room does a bias-only container get its own bootstrap
//! ([`CompactLayout::new`]). The bias positions of the velocity stay zero: biases take plain
//! SGD steps under NAG (the M2 deviation, Task 28), so `W' = W - h_W - h_b - kappa U` and
//! `U' = mu U + h_W`.
//!
//! # Levels and placements
//!
//! [`levels_per_iter`] is the exact level count of one step from the unpacked weights to the
//! updated compact state; [`NonInteractiveTrainer::new`] refuses a network whose step does not
//! fit the levels left after a bootstrap and an unpack (`InvalidParams`). See
//! [`BootPlacement`] for what each placement bootstraps when.
//!
//! # Backends and the bootstrapper (Task 43)
//!
//! [`NonInteractiveTrainer`] is generic over the evaluator `E: HeOps` and the bootstrapper
//! `B: `[`BootstrapOps<E>`] (defaults [`Evaluator`] and [`Bootstrapper`]). The CPU
//! [`Bootstrapper`] is not generic: it implements `BootstrapOps<Evaluator>` only. A device
//! backend supplies its own `BootstrapOps` (a device bootstrapper, Task 49, or a host adapter
//! that downloads, bootstraps on the CPU and uploads). The trainer reads only
//! [`BootstrapOps::output_level`] for its level schedule and calls
//! [`BootstrapOps::bootstrap`] on its state. The evaluator's ablation flags apply as in the
//! interactive trainer, plus `lazy_relin` on the micro-batch gradient sum
//! ([`Dense::weight_grad_reduced`]); [`predict_update`] follows the evaluator's flags
//! ([`predict_update_with`]).

use std::sync::Arc;
use std::time::Instant;

use hekit_ckks::bootstrap::Bootstrapper;
use hekit_ckks::{
    BootstrapOps, Ciphertext, DiagonalEntry, Encoder, Encryptor, EvalConfig, Evaluator, HeOps,
};
use hekit_core::error::{HekitError, Result};
use hekit_core::sampler::Rng;
use ndarray::Array2;

use crate::activation::Activation;
use crate::compact::{
    CompactPlan, compact_extract, compact_pack_plain, compact_rotation_steps, compact_shift,
    compact_unpack, extract_steps, place_plain,
};
use crate::dense::{
    Dense, EncWeights, TransposeKeys, backward_rotation_steps, forward_rotation_steps,
    grad_reduced_rotation_steps, transpose_rotation_plan, transpose_rotation_steps,
};
use crate::layout::{LayerShape, Layout, pack_bias, pack_weight_diagonals};
use crate::loss::Mse;
use crate::sim::valid_lane_mask;
use crate::trainer::{
    BatchSpec, EpochLog, Mlp, ScaleProbe, check_dims, constant_dz, dense_out, sorted, zero_ct,
};

// ---------------------------------------------------------------------------------------------
// Placement and optimizer.
// ---------------------------------------------------------------------------------------------

/// When the trainer bootstraps (`docs/layout.md` §7.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootPlacement {
    /// After **every** update: every compact weight and velocity ciphertext is bootstrapped,
    /// and the next step starts from the bootstrap output level (minus the unpack). The cost
    /// model's placement (one bootstrap of `W` and `V` per update).
    AfterUpdate,
    /// The output of every hidden activation (`a_l`, `l < L`, each micro-batch ciphertext) is
    /// bootstrapped right after the activation (first switched to exactly `Delta` by a
    /// multiplication by 1, one ct-pt and one level, so the next layer's bias scale is
    /// `Delta^2`), and the next layer runs from the bootstrapped value. The weights are
    /// bootstrapped only when the next step would not fit their remaining levels. Each step
    /// then consumes fewer weight levels (the forward of the upper layers no longer descends
    /// from the lower ones), so the weights are bootstrapped less often, at the price of
    /// `(L - 1) k` activation bootstraps per update.
    ///
    /// **Precondition of the activation bootstraps: `|a_l| <= 1` in every slot**
    /// ([`Bootstrapper::bootstrap`]'s EvalMod range). Larger activations are **not clipped**:
    /// the bootstrap still runs, and its error grows (quadratically in `|m|` above 1; Task 38
    /// measured 17.38 bits absolute at `|m| = 4` on n16_boot, against 17.46 at `|m| <= 1`,
    /// `docs/bootstrap.md` §11.4). The caller is responsible for the range (e.g. the input
    /// scaling or the initialisation) and should measure it: the client can decrypt
    /// [`NonInteractiveTrainer::last_activation_bootstraps`] after an update and compare with
    /// a plaintext twin, as the `mnist` and `iris` benches do.
    AfterActivation,
    /// Not in the brief: no activation bootstrap, and the weights are bootstrapped only when the
    /// next step would not fit their remaining levels (the schedule of
    /// [`BootPlacement::AfterActivation`] without its activation bootstraps). For a network
    /// whose step takes less than half the levels after a bootstrap this skips bootstraps.
    WhenNeeded,
}

impl BootPlacement {
    fn boots_activations(self) -> bool {
        self == BootPlacement::AfterActivation
    }

    fn lazy(self) -> bool {
        self != BootPlacement::AfterUpdate
    }

    /// The command-line / JSON name: `after_update`, `after_activation`, `when_needed`.
    pub fn name(self) -> &'static str {
        match self {
            BootPlacement::AfterUpdate => "after_update",
            BootPlacement::AfterActivation => "after_activation",
            BootPlacement::WhenNeeded => "when_needed",
        }
    }
}

/// The update rule on the compact state, for every layer. The gradient constant is folded into
/// the batch-sum mask ([`NiOptimizer::grad_scale`] divided by the valid sample count), so the
/// update is subtractions and, for NAG, two scalar products of the velocity (off the critical
/// path, as [`crate::optim::Nag`]). Biases take plain SGD steps with `lr` in both cases (the
/// M2 deviation: no bias momentum).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NiOptimizer {
    /// `W' = W - lr dW`.
    Sgd {
        /// The learning rate.
        lr: f64,
    },
    /// Nesterov in the scaled-velocity form of [`crate::optim::Nag`]: `h = lr (1 + mu) dW`,
    /// `U' = mu U + h`, `W' = W - h - (mu^2 / (1 + mu)) U` (`U = 0` before the first step).
    Nag {
        /// The learning rate.
        lr: f64,
        /// The momentum `mu`.
        momentum: f64,
    },
}

impl NiOptimizer {
    /// The factor the weight gradient carries: `lr` (SGD) or `lr (1 + mu)` (NAG).
    pub fn grad_scale(&self) -> f64 {
        match *self {
            NiOptimizer::Sgd { lr } => lr,
            NiOptimizer::Nag { lr, momentum } => lr * (1.0 + momentum),
        }
    }

    /// The factor the bias gradient carries: `lr` (plain SGD for the biases).
    pub fn bias_grad_scale(&self) -> f64 {
        match *self {
            NiOptimizer::Sgd { lr } | NiOptimizer::Nag { lr, .. } => lr,
        }
    }

    /// `true` for NAG (a velocity per weight container, bootstrapped with it).
    pub fn has_velocity(&self) -> bool {
        matches!(self, NiOptimizer::Nag { .. })
    }

    /// The learning rate.
    pub fn lr(&self) -> f64 {
        match *self {
            NiOptimizer::Sgd { lr } | NiOptimizer::Nag { lr, .. } => lr,
        }
    }

    /// The same optimizer with learning rate `lr`.
    pub fn with_lr(self, lr: f64) -> NiOptimizer {
        match self {
            NiOptimizer::Sgd { .. } => NiOptimizer::Sgd { lr },
            NiOptimizer::Nag { momentum, .. } => NiOptimizer::Nag { lr, momentum },
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Compact layout of the state.
// ---------------------------------------------------------------------------------------------

/// One compact ciphertext of the state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContainerSpec {
    /// The layer whose weight diagonals it holds (`None`: a bias-only container).
    pub layer: Option<usize>,
    /// The index among the layer's compact ciphertexts ([`CompactPlan::ciphertexts`]).
    pub index: usize,
    /// The plan that lays out its positions (the layer's, or `L = layout.d` for a bias-only
    /// container).
    pub plan: CompactPlan,
}

/// Where a layer's bias lives: position `position` of container `container`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiasSlot {
    /// Index into [`CompactLayout::containers`].
    pub container: usize,
    /// The compact position `t` (slots `offset(t) + j B`, `j < L` of the container's plan).
    pub position: usize,
}

/// The compact layout of a network's trainable state (`docs/layout.md` §7.3): per layer its
/// [`CompactPlan`] and its compact weight ciphertexts (in layer order), then any bias-only
/// containers, and per layer the [`BiasSlot`] of its bias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactLayout {
    /// The layers' plans.
    pub plans: Vec<CompactPlan>,
    /// Every compact ciphertext: the weight containers of layer 0, of layer 1, ..., then the
    /// bias-only containers.
    pub containers: Vec<ContainerSpec>,
    /// One slot per layer.
    pub biases: Vec<BiasSlot>,
}

impl CompactLayout {
    /// The layout of the MLP `dims[0] → dims[1] → …` in `layout`. Each bias (padded length
    /// `n = pow2(d_out)`) takes a free position of a weight container whose plan has
    /// `L >= n`: among those, the containers with the largest `L` first (fewest replication
    /// rotations at the extraction, `log2(S / L)`), and within a container a free
    /// power-of-two position first (its offset is a tree step, whose Galois keys the step
    /// already has), else the lowest free position. A bias that finds no room goes to a
    /// bias-only container with `L = layout.d` (one more bootstrap per update).
    ///
    /// Errors: `InvalidParams` for fewer than 2 dimensions or a dimension that does not fit
    /// `layout`.
    pub fn new(dims: &[usize], layout: &Layout) -> Result<CompactLayout> {
        check_dims(dims, layout)?;
        let plans: Vec<CompactPlan> = dims
            .windows(2)
            .map(|p| CompactPlan::for_layer(layout, p[1], p[0]))
            .collect();
        let mut containers = Vec::new();
        let mut used: Vec<Vec<bool>> = Vec::new();
        for (l, plan) in plans.iter().enumerate() {
            for i in 0..plan.ciphertexts() {
                containers.push(ContainerSpec {
                    layer: Some(l),
                    index: i,
                    plan: *plan,
                });
                let mut u = vec![false; plan.diags_per_ct];
                u[..plan.diagonals_in(i)].iter_mut().for_each(|x| *x = true);
                used.push(u);
            }
        }
        let mut biases = Vec::with_capacity(plans.len());
        for &d_out in &dims[1..] {
            let n = d_out.next_power_of_two();
            let mut cands: Vec<usize> = (0..containers.len())
                .filter(|&c| containers[c].plan.diag_len() >= n && used[c].iter().any(|u| !u))
                .collect();
            cands.sort_by_key(|&c| (std::cmp::Reverse(containers[c].plan.diag_len()), c));
            let pow2 = |c: usize| {
                (0..used[c].len().trailing_zeros())
                    .map(|k| 1usize << k)
                    .find(|&t| !used[c][t])
            };
            let pick = cands
                .iter()
                .find_map(|&c| pow2(c).map(|t| (c, t)))
                .or_else(|| {
                    cands
                        .first()
                        .map(|&c| (c, used[c].iter().position(|u| !u).expect("free")))
                });
            let (c, t) = match pick {
                Some(ct) => ct,
                None => {
                    let plan = CompactPlan::for_layer(layout, layout.d, layout.d);
                    containers.push(ContainerSpec {
                        layer: None,
                        index: 0,
                        plan,
                    });
                    used.push(vec![false; plan.diags_per_ct]);
                    (containers.len() - 1, 0)
                }
            };
            used[c][t] = true;
            biases.push(BiasSlot {
                container: c,
                position: t,
            });
        }
        Ok(CompactLayout {
            plans,
            containers,
            biases,
        })
    }

    /// The containers of layer `l` (a range into [`CompactLayout::containers`]).
    pub fn layer_containers(&self, l: usize) -> std::ops::Range<usize> {
        let start: usize = self.plans[..l].iter().map(|p| p.ciphertexts()).sum();
        start..start + self.plans[l].ciphertexts()
    }

    /// The plan of the container holding layer `l`'s bias.
    pub fn bias_plan(&self, l: usize) -> &CompactPlan {
        &self.containers[self.biases[l].container].plan
    }

    /// The Galois steps of the compact path: [`compact_rotation_steps`] of every layer, and
    /// for every bias the gradient shift `-offset(t)` and [`extract_steps`] of its position.
    pub fn rotation_steps(&self) -> Vec<i32> {
        let mut steps: Vec<i32> = self.plans.iter().flat_map(compact_rotation_steps).collect();
        for (l, s) in self.biases.iter().enumerate() {
            let plan = self.bias_plan(l);
            let o = plan.offset(s.position) as i32;
            if o != 0 {
                steps.push(-o);
            }
            steps.extend(extract_steps(plan, s.position));
        }
        sorted(steps)
    }

    /// Bootstraps per update when the weights are bootstrapped: every container, plus a
    /// velocity for every weight container with NAG.
    pub fn bootstraps(&self, nag: bool) -> usize {
        self.containers
            .iter()
            .map(|c| 1 + usize::from(nag && c.layer.is_some()))
            .sum()
    }
}

/// The Galois steps of [`NonInteractiveTrainer`]'s steps for the MLP `dims` in `layout`, sorted,
/// one per Galois element (steps equal modulo the slot count share a key): every layer's
/// [`forward_rotation_steps`] and [`grad_reduced_rotation_steps`], the
/// [`backward_rotation_steps`] and [`transpose_rotation_steps`] of every layer but the first,
/// and [`CompactLayout::rotation_steps`]. The bootstrapper brings its own keys.
///
/// Errors: those of [`CompactLayout::new`] and of [`transpose_rotation_steps`].
pub fn ni_rotation_steps(
    dims: &[usize],
    layout: &Layout,
    keys: &TransposeKeys,
) -> Result<Vec<i32>> {
    let cl = CompactLayout::new(dims, layout)?;
    let mut steps = Vec::new();
    for (i, p) in dims.windows(2).enumerate() {
        let (d_in, d_out) = (p[0], p[1]);
        steps.extend(forward_rotation_steps(d_out, d_in, layout));
        steps.extend(grad_reduced_rotation_steps(d_out, d_in, layout));
        if i > 0 {
            steps.extend(backward_rotation_steps(d_out, d_in, layout));
            steps.extend(transpose_rotation_steps(d_out, d_in, layout, keys)?);
        }
    }
    steps.extend(cl.rotation_steps());
    let s = layout.slots as i64;
    let mut seen = std::collections::HashSet::new();
    let mut out: Vec<i32> = sorted(steps)
        .into_iter()
        .filter(|&k| (k as i64).rem_euclid(s) != 0)
        .collect();
    out.retain(|&k| seen.insert((k as i64).rem_euclid(s)));
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The compact state.
// ---------------------------------------------------------------------------------------------

/// The trainable state between steps: one ciphertext per container of the [`CompactLayout`]
/// (weights and biases) and, with NAG, the velocity of every weight container (`None` before
/// the first step and for bias-only containers). All at `Delta` exactly. `C` is the
/// backend's ciphertext type ([`Ciphertext`] by default).
#[derive(Debug, Clone)]
pub struct CompactState<C = Ciphertext> {
    /// One ciphertext per container.
    pub weights: Vec<C>,
    /// One velocity per container (`None`: zero).
    pub velocity: Vec<Option<C>>,
}

impl CompactState {
    /// The state on the backend of `ev` (a move for the CPU [`Evaluator`]).
    ///
    /// Errors: those of the upload.
    pub fn upload_owned<E: HeOps>(self, ev: &E) -> Result<CompactState<E::Ct>> {
        Ok(CompactState {
            weights: self
                .weights
                .into_iter()
                .map(|c| ev.upload_ct_owned(c))
                .collect::<Result<_>>()?,
            velocity: self
                .velocity
                .into_iter()
                .map(|u| u.map(|c| ev.upload_ct_owned(c)).transpose())
                .collect::<Result<_>>()?,
        })
    }
}

impl CompactState {
    /// Encrypts (client side) the compact state of the plaintext layers `(W_l, b_l)` at `level`
    /// and `Delta`, zero velocity: weight diagonal `c` of layer `l` at its position
    /// ([`compact_pack_plain`]), each bias in canonical form at its [`BiasSlot`].
    ///
    /// Errors: `InvalidParams` if the layers do not match `cl` or `layout`; the
    /// [`hekit_ckks::Encoder::encode`] errors.
    pub fn encrypt(
        cl: &CompactLayout,
        layers: &[(Array2<f64>, Vec<f64>)],
        layout: &Layout,
        level: usize,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<CompactState> {
        if layers.len() != cl.plans.len() {
            return Err(HekitError::InvalidParams(format!(
                "{} layers for a compact layout of {}",
                layers.len(),
                cl.plans.len()
            )));
        }
        let mut slots: Vec<Vec<f64>> = vec![vec![0.0; layout.slots]; cl.containers.len()];
        for (l, ((w, b), plan)) in layers.iter().zip(&cl.plans).enumerate() {
            if w.dim() != (plan.d_out, plan.d_in) || b.len() != plan.d_out {
                return Err(HekitError::InvalidParams(format!(
                    "layer {l}: {:?} weights and {} biases for a plan of {} x {}",
                    w.dim(),
                    b.len(),
                    plan.d_out,
                    plan.d_in
                )));
            }
            let sh = LayerShape::new(plan.d_out, plan.d_in, layout);
            let diags = pack_weight_diagonals(w, layout, (sh.baby, sh.giant));
            for (k, v) in compact_pack_plain(&diags, plan).into_iter().enumerate() {
                let c = cl.layer_containers(l).start + k;
                slots[c].iter_mut().zip(&v).for_each(|(s, x)| *s += x);
            }
            let slot = cl.biases[l];
            let canon = pack_bias(b, layout);
            place_plain(
                &mut slots[slot.container],
                &canon,
                &cl.containers[slot.container].plan,
                slot.position,
            );
        }
        let delta = encoder.params().scale();
        let weights = slots
            .iter()
            .map(|v| Ok(enc.encrypt(&encoder.encode(v, delta, level)?, rng)))
            .collect::<Result<Vec<_>>>()?;
        Ok(CompactState {
            velocity: vec![None; weights.len()],
            weights,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Levels (exact, structural).
// ---------------------------------------------------------------------------------------------

/// What the level arithmetic needs of an activation.
#[derive(Debug, Clone, Copy)]
struct ActMeta {
    depth: i64,
    dderiv: i64,
    constant: Option<f64>,
}

fn metas<E: HeOps>(mlp: &Mlp<E>) -> Vec<ActMeta> {
    let acts: Vec<&dyn Activation<E>> = mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
    metas_of(&acts)
}

fn metas_of<E: HeOps>(acts: &[&dyn Activation<E>]) -> Vec<ActMeta> {
    acts.iter()
        .map(|a| ActMeta {
            depth: a.depth() as i64,
            dderiv: a.derivative_depth() as i64,
            constant: a.constant_derivative(),
        })
        .collect()
}

/// The levels of one step (signed: a negative level means the step does not fit).
#[derive(Debug, Clone)]
struct Pass {
    /// `(label, level)` of every value, as [`NonInteractiveTrainer::trace`].
    trace: Vec<(String, i64)>,
    /// Input level of every layer's forward (after alignment).
    inputs: Vec<i64>,
    /// Level of every layer's compact weight gradient.
    h: Vec<i64>,
    /// Level of every layer's bias gradient.
    hb: Vec<i64>,
    /// The lowest level of the step.
    min: i64,
}

/// The level of every value of one step, op for op the trainer's: weights of layer `l` at
/// `w[l]`, its bias at `b[l]`, the input batch at `x`, a bootstrap output at `boot`.
fn pass_levels(
    meta: &[ActMeta],
    boot_acts: bool,
    w: &[i64],
    b: &[i64],
    x: i64,
    boot: i64,
    full: bool,
) -> Pass {
    let n = meta.len();
    let mut trace = Vec::new();
    let mut inputs = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut acts = Vec::with_capacity(n + 1);
    let mut prev = x;
    for l in 0..n {
        let input = prev.min(w[l]).min(b[l]);
        inputs.push(input);
        if l == 0 {
            acts.push(input);
        }
        let z = input - 1;
        let a = z - meta[l].depth;
        trace.push((format!("z{}", l + 1), z));
        trace.push((format!("a{}", l + 1), a));
        zs.push(z);
        if boot_acts && l + 1 < n {
            trace.push((format!("a{}_to_delta", l + 1), a - 1));
            trace.push((format!("a{}_boot", l + 1), boot));
            acts.push(boot);
            prev = boot;
        } else {
            acts.push(a);
            prev = a;
        }
    }
    let mut delta = prev;
    trace.push((format!("delta{n}"), delta));
    let mut h = vec![0; n];
    let mut hb = vec![0; n];
    for i in (0..n).rev() {
        let ln = i + 1;
        let dz = match meta[i].constant {
            Some(k) if k == 1.0 && full => delta,
            Some(_) => delta - 1,
            None => {
                let s = zs[i] - meta[i].dderiv;
                trace.push((format!("s{ln}"), s));
                s.min(delta) - 1
            }
        };
        trace.push((format!("dz{ln}"), dz));
        if i > 0 {
            delta = dz - 1;
            trace.push((format!("delta{i}"), delta));
        }
        h[i] = dz.min(acts[i]) - 2;
        hb[i] = dz - 1;
        trace.push((format!("h{ln}"), h[i]));
        trace.push((format!("hb{ln}"), hb[i]));
    }
    let min = trace.iter().map(|(_, v)| *v).min().unwrap_or(0).min(x);
    Pass {
        trace,
        inputs,
        h,
        hb,
        min,
    }
}

/// The levels one training step consumes, exactly: from the unpacked weights (and biases and
/// input batch, all on one level `w`) to the lowest level of the step, which is the updated
/// compact state (the weight and bias gradients land there; the update itself costs no level).
/// With [`BootPlacement::AfterActivation`] the bootstrapped activations restart above `w`, so
/// the forward of the upper layers no longer descends from the lower ones. A full batch is
/// assumed (a partial batch costs one more level at a linear output layer, see
/// [`crate::activation::Activation::constant_derivative`]).
///
/// eMLP-1 (784 → 32 z² + z → 10 linear): 7 for [`BootPlacement::AfterUpdate`] and
/// [`BootPlacement::WhenNeeded`], 5 for [`BootPlacement::AfterActivation`].
/// [`NonInteractiveTrainer::new`] requires `levels_per_iter` to be at most the levels left
/// after a bootstrap and an unpack (`output_level - 1`, 15 on n16_boot).
pub fn levels_per_iter<E: HeOps>(mlp: &Mlp<E>, placement: BootPlacement) -> usize {
    let acts: Vec<&dyn Activation<E>> = mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
    levels_per_iter_of(&acts, placement)
}

/// [`levels_per_iter`] of a network with the activations `acts` (one per layer; the level count
/// does not depend on the dimensions), without an encrypted network (e.g. for a plan).
pub fn levels_per_iter_of<E: HeOps>(
    acts: &[&dyn Activation<E>],
    placement: BootPlacement,
) -> usize {
    let meta = metas_of(acts);
    const BIG: i64 = 1 << 30;
    let n = meta.len();
    let p = pass_levels(
        &meta,
        placement.boots_activations(),
        &vec![BIG; n],
        &vec![BIG; n],
        BIG,
        BIG + 1,
        true,
    );
    (BIG - p.min) as usize
}

/// The level state between steps (signed for the feasibility arithmetic).
#[derive(Debug, Clone)]
struct LevelState {
    /// Unpacked weights per layer.
    w: Vec<i64>,
    /// Extracted bias per layer.
    b: Vec<i64>,
    /// Compact weights per container.
    c: Vec<i64>,
    /// Velocity per container.
    u: Vec<Option<i64>>,
}

/// The container levels after the update of `pass` from `st` (NAG lands `W'` and `U'` on the
/// same level, as [`crate::optim::Nag`]).
fn update_levels(cl: &CompactLayout, st: &LevelState, pass: &Pass, nag: bool) -> Vec<i64> {
    cl.containers
        .iter()
        .enumerate()
        .map(|(c, spec)| {
            let mut lv = st.c[c];
            if let Some(l) = spec.layer {
                lv = lv.min(pass.h[l]);
                if let (true, Some(u)) = (nag, st.u[c]) {
                    lv = lv.min(u - 1);
                }
            }
            for (l, s) in cl.biases.iter().enumerate() {
                if s.container == c {
                    lv = lv.min(pass.hb[l]);
                }
            }
            lv
        })
        .collect()
}

/// The unpacked levels from container levels `c`: each layer's weights one below the lowest
/// of its containers, each bias one below its container.
fn unpacked(cl: &CompactLayout, c: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let w = (0..cl.plans.len())
        .map(|l| cl.layer_containers(l).map(|i| c[i]).min().expect("one") - 1)
        .collect();
    let b = cl.biases.iter().map(|s| c[s.container] - 1).collect();
    (w, b)
}

// ---------------------------------------------------------------------------------------------
// Closed-form counts.
// ---------------------------------------------------------------------------------------------

/// The key switches of one update by phase (closed form, [`predict_update`]); bootstraps are
/// counted as events, their own key switches are not in the other fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpdateCost {
    /// Forward matvecs and folds (rotations), per micro-batch.
    pub forward_rot: u64,
    /// Forward relinearizations (matvec giants and activations), per micro-batch.
    pub forward_relin: u64,
    /// Transposes (rotations, once per update).
    pub transpose_rot: u64,
    /// Backward matvecs (rotations).
    pub backward_rot: u64,
    /// Backward relinearizations (matvec giants, derivative factors and the `dz` products).
    pub backward_relin: u64,
    /// Weight-gradient rotations: hoisted babies and giants per micro-batch, the reduce.
    pub gradient_rot: u64,
    /// Weight-gradient relinearizations (`m` per layer per update, lazy over micro-batches).
    pub gradient_relin: u64,
    /// Lane shifts into compact form ([`compact_shift`], `m - n_ct` per layer).
    pub shift_rot: u64,
    /// Bias gradients: the reduce and the shift to the bias position.
    pub bias_grad_rot: u64,
    /// Unpack of the weights ([`compact_unpack`]).
    pub unpack_rot: u64,
    /// The second lane shift of an update whose lr differs from the velocity's reference lr
    /// ([`NonInteractiveTrainer::set_lr`]; 0 otherwise): [`ref_shift_rotations`].
    pub ref_shift_rot: u64,
    /// Bias extraction ([`compact_extract`]).
    pub bias_extract_rot: u64,
    /// Weight (and velocity) bootstraps.
    pub weight_bootstraps: u64,
    /// Activation bootstraps ([`BootPlacement::AfterActivation`]).
    pub activation_bootstraps: u64,
}

impl UpdateCost {
    /// All rotations outside the bootstraps.
    pub fn rotations(&self) -> u64 {
        self.forward_rot
            + self.transpose_rot
            + self.backward_rot
            + self.gradient_rot
            + self.shift_rot
            + self.ref_shift_rot
            + self.bias_grad_rot
            + self.unpack_rot
            + self.bias_extract_rot
    }

    /// All relinearizations outside the bootstraps.
    pub fn relins(&self) -> u64 {
        self.forward_relin + self.backward_relin + self.gradient_relin
    }

    /// All bootstraps.
    pub fn bootstraps(&self) -> u64 {
        self.weight_bootstraps + self.activation_bootstraps
    }

    /// The key switches with every bootstrap at `boot_ks` (e.g.
    /// [`Bootstrapper::key_switches_per_boot`], 111 on n16_boot).
    pub fn key_switches(&self, boot_ks: u64) -> u64 {
        self.rotations() + self.relins() + self.bootstraps() * boot_ks
    }
}

/// What the closed form needs of an activation: the relinearizations of its forward and of
/// its derivative factor, and whether its derivative is a constant (no `dz` product).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActCost {
    /// Relinearizations of [`Activation::forward`] (1 for `Square` and `PolyRelu2`).
    pub forward_relins: u64,
    /// Relinearizations of [`Activation::derivative_factor_at`] (0 for `Square` and
    /// `PolyRelu2`, whose derivatives are linear).
    pub derivative_relins: u64,
    /// [`Activation::constant_derivative`] is `Some` (e.g. the identity): no `dz` product.
    pub constant_derivative: bool,
}

impl ActCost {
    /// `z^2 + z` or `z^2`: one relin forward, a linear derivative, one `dz` product.
    pub const QUADRATIC: ActCost = ActCost {
        forward_relins: 1,
        derivative_relins: 0,
        constant_derivative: false,
    };
    /// The identity (a linear layer).
    pub const IDENTITY: ActCost = ActCost {
        forward_relins: 0,
        derivative_relins: 0,
        constant_derivative: true,
    };

    /// Reads the counts of `act` by running it on a zero ciphertext at the top level with a
    /// fork of `ev` ([`HeOps::fork`]: shared keys, own counter: `ev`'s counter does not move).
    ///
    /// Errors: the activation's errors (e.g. no relinearization key).
    pub fn probe<E: HeOps>(ev: &E, act: &dyn Activation<E>) -> Result<ActCost> {
        let constant = act.constant_derivative().is_some();
        if constant && act.depth() == 0 {
            return Ok(ActCost::IDENTITY);
        }
        let probe = ev.fork();
        let enc = Encoder::new(Arc::clone(ev.params()));
        let z = zero_ct(ev, ev.params().max_level(), ev.params().scale())?;
        act.forward(&probe, &enc, &z)?;
        let forward_relins = probe.counter().relins;
        probe.reset_counter();
        if !constant {
            act.derivative_factor_at(&probe, &enc, &z, 1.0, None, None)?;
        }
        Ok(ActCost {
            forward_relins,
            derivative_relins: probe.counter().relins,
            constant_derivative: constant,
        })
    }
}

/// The closed-form counts of one update of `mlp` with `micro` micro-batch ciphertexts on
/// `ev`: [`predict_update_with`] with the activations' [`ActCost::probe`] and `ev`'s ablation
/// flags ([`HeOps::config`]; the default flags give [`predict_update_for`]).
///
/// Errors: those of [`predict_update_for`] and of the probe.
#[allow(clippy::too_many_arguments)]
pub fn predict_update<E: HeOps>(
    ev: &E,
    mlp: &Mlp<E>,
    layout: &Layout,
    keys: &TransposeKeys,
    placement: BootPlacement,
    micro: usize,
    nag: bool,
    weights_booted: bool,
) -> Result<UpdateCost> {
    let acts = mlp
        .layers
        .iter()
        .map(|(_, a)| ActCost::probe(ev, a.as_ref()))
        .collect::<Result<Vec<_>>>()?;
    predict_update_with(
        &mlp.dims(),
        &acts,
        layout,
        keys,
        placement,
        micro,
        nag,
        weights_booted,
        *ev.config(),
    )
}

/// The closed-form counts of one update of the MLP `dims` (activations `acts`) with `micro`
/// micro-batch ciphertexts (`docs/layout.md` §7.5), op for op
/// [`NonInteractiveTrainer::train_step`]: forward, transpose, backward, gradient (lazy
/// relinearization over the micro-batches), lane shift, bias gradient, unpack, bias
/// extraction, and the bootstraps (activations if `placement` bootstraps them; weights and
/// velocities if `weights_booted`).
///
/// Errors: `InvalidParams` if `acts` does not have one entry per layer; those of
/// [`CompactLayout::new`] and of [`transpose_rotation_plan`].
#[allow(clippy::too_many_arguments)]
pub fn predict_update_for(
    dims: &[usize],
    acts: &[ActCost],
    layout: &Layout,
    keys: &TransposeKeys,
    placement: BootPlacement,
    micro: usize,
    nag: bool,
    weights_booted: bool,
) -> Result<UpdateCost> {
    predict_update_with(
        dims,
        acts,
        layout,
        keys,
        placement,
        micro,
        nag,
        weights_booted,
        EvalConfig::default(),
    )
}

/// [`predict_update_for`] for an evaluator with the ablation flags `cfg` (Task 43, RQ3): the
/// forward and backward matvecs cost [`EvalConfig::matvec_rotations`] and
/// [`EvalConfig::matvec_enc_relins`], and the micro-batch gradient `m` relins with
/// `lazy_relin` (`micro * m` without). `hoist` changes no count. The activations' counts come
/// from `acts` (probe them on an evaluator with the same flags, as [`predict_update`] does).
///
/// Errors as [`predict_update_for`].
#[allow(clippy::too_many_arguments)]
pub fn predict_update_with(
    dims: &[usize],
    acts: &[ActCost],
    layout: &Layout,
    keys: &TransposeKeys,
    placement: BootPlacement,
    micro: usize,
    nag: bool,
    weights_booted: bool,
    cfg: EvalConfig,
) -> Result<UpdateCost> {
    let cl = CompactLayout::new(dims, layout)?;
    let n = cl.plans.len();
    if acts.len() != n {
        return Err(HekitError::InvalidParams(format!(
            "{} activations for {n} layers",
            acts.len()
        )));
    }
    let k = micro as u64;
    let lb = layout.log_batch() as u64;
    let mut c = UpdateCost::default();
    for (i, p) in dims.windows(2).enumerate() {
        let (d_in, d_out) = (p[0], p[1]);
        let sh = LayerShape::new(d_out, d_in, layout);
        let act = acts[i];
        let m = sh.m as u64;
        c.forward_rot += k * (cfg.matvec_rotations(sh.m) + sh.fold_steps() as u64);
        c.forward_relin += k * (cfg.matvec_enc_relins(sh.m) + act.forward_relins);
        if !act.constant_derivative {
            c.backward_relin += k * (act.derivative_relins + 1);
        }
        if i > 0 {
            let plan = transpose_rotation_plan(d_out, d_in, layout, keys)?;
            c.transpose_rot += plan.iter().map(|(_, s)| s.len() as u64).sum::<u64>();
            let st = sh.transposed();
            c.backward_rot += k * (cfg.matvec_rotations(st.m) + st.fold_steps() as u64);
            c.backward_relin += k * cfg.matvec_enc_relins(st.m);
        }
        c.gradient_rot += k * ((sh.baby - 1) + (sh.giant - 1)) as u64 + m * lb;
        c.gradient_relin += if cfg.lazy_relin { m } else { k * m };
        let plan = &cl.plans[i];
        c.shift_rot += plan.pack_rotations() as u64;
        c.unpack_rot += plan.unpack_rotations() as u64;
        let slot = cl.biases[i];
        let bp = cl.bias_plan(i);
        let o = u64::from(bp.offset(slot.position) != 0);
        c.bias_grad_rot += lb + o;
        c.bias_extract_rot += o + bp.diags_per_ct.trailing_zeros() as u64;
        if placement.boots_activations() && i + 1 < n {
            c.activation_bootstraps += k;
        }
    }
    if weights_booted {
        c.weight_bootstraps = cl.bootstraps(nag) as u64;
    }
    Ok(c)
}

/// The extra rotations of an update whose lr differs from the velocity's reference lr
/// ([`NonInteractiveTrainer::set_lr`], `docs/layout.md` §7.1): the reference-lr gradient's own
/// lane shift, `m - n_ct` per layer ([`crate::compact::CompactPlan::pack_rotations`]). Set it as
/// [`UpdateCost::ref_shift_rot`] when [`NonInteractiveTrainer::last_update_rescaled`].
///
/// Errors: those of [`CompactLayout::new`].
pub fn ref_shift_rotations(dims: &[usize], layout: &Layout) -> Result<u64> {
    let cl = CompactLayout::new(dims, layout)?;
    Ok(cl.plans.iter().map(|p| p.pack_rotations() as u64).sum())
}

/// The bootstrap decision after an update that leaves the levels `next` (the one rule of both
/// [`NonInteractiveTrainer`] and [`boot_schedule`]): always for [`BootPlacement::AfterUpdate`];
/// for a lazy placement, unless the next step fits from `next` (every container unpackable and
/// every level of the step non-negative). `next_full` is what is known of the next step:
/// `Some(true)` a full batch; `Some(false)` or `None` (unknown) a partial one, which costs one
/// more level at a linear output layer ([`crate::activation::Activation::constant_derivative`]),
/// so an unknown next step is planned conservatively.
fn must_boot(
    meta: &[ActMeta],
    placement: BootPlacement,
    next: &LevelState,
    start: i64,
    out: i64,
    next_full: Option<bool>,
) -> bool {
    if !placement.lazy() {
        return true;
    }
    let fits_unpack = next.c.iter().all(|&c| c >= 1);
    let p = pass_levels(
        meta,
        placement.boots_activations(),
        &next.w,
        &next.b,
        start,
        out,
        next_full == Some(true),
    );
    !(fits_unpack && p.min >= 0)
}

/// Which updates bootstrap the weights, from a fresh (bootstrapped) state, for updates whose
/// batches are full (`full[i]`) or partial: always for [`BootPlacement::AfterUpdate`]; for the
/// lazy placements the trainer's rule ([`NonInteractiveTrainer::set_next_batch_full`]): an
/// update bootstraps when the next step would not fit, the next step's fullness being
/// `full[i + 1]` (the trainer's `train_epoch_valid` look-ahead) and unknown after the last
/// update (planned as partial). `start` is the level of the unpacked weights after a bootstrap
/// ([`NonInteractiveTrainer::start_level_for`]), `out` the bootstrap output level.
///
/// Errors: `InvalidParams` if `acts` does not have one entry per layer; as
/// [`CompactLayout::new`].
#[allow(clippy::too_many_arguments)]
pub fn boot_schedule<E: HeOps>(
    dims: &[usize],
    acts: &[&dyn Activation<E>],
    layout: &Layout,
    placement: BootPlacement,
    nag: bool,
    start: usize,
    out: usize,
    full: &[bool],
) -> Result<Vec<bool>> {
    let cl = CompactLayout::new(dims, layout)?;
    if acts.len() != cl.plans.len() {
        return Err(HekitError::InvalidParams(format!(
            "{} activations for {} layers",
            acts.len(),
            cl.plans.len()
        )));
    }
    let meta = metas_of(acts);
    let n = meta.len();
    let (s, o) = (start as i64, out as i64);
    let fresh = |u: bool| LevelState {
        w: vec![s; n],
        b: vec![s; n],
        c: vec![o; cl.containers.len()],
        u: vec![u.then_some(o); cl.containers.len()],
    };
    let mut st = fresh(false);
    let mut out_v = Vec::with_capacity(full.len());
    for (i, &f) in full.iter().enumerate() {
        let p = pass_levels(&meta, placement.boots_activations(), &st.w, &st.b, s, o, f);
        let c = update_levels(&cl, &st, &p, nag);
        let mut next = LevelState {
            w: Vec::new(),
            b: Vec::new(),
            u: cl
                .containers
                .iter()
                .zip(&c)
                .map(|(spec, &lv)| (nag && spec.layer.is_some()).then_some(lv))
                .collect(),
            c,
        };
        (next.w, next.b) = unpacked(&cl, &next.c);
        let boot = must_boot(&meta, placement, &next, s, o, full.get(i + 1).copied());
        out_v.push(boot);
        st = if boot { fresh(nag) } else { next };
    }
    Ok(out_v)
}

// ---------------------------------------------------------------------------------------------
// The trainer.
// ---------------------------------------------------------------------------------------------

/// Non-interactive mini-batch training of an encrypted [`Mlp`] (see the [module docs](self)).
/// Holds no secret key.
///
/// Generic over the evaluator `E` ([`Evaluator`] by default) and the bootstrapper `B`
/// ([`Bootstrapper`] by default; any [`BootstrapOps<E>`], e.g. a device bootstrapper or a
/// host adapter, Task 49). The level schedule only reads `B`'s output level.
pub struct NonInteractiveTrainer<'a, E: HeOps = Evaluator, B: BootstrapOps<E> = Bootstrapper> {
    ev: &'a E,
    bs: &'a B,
    optim: NiOptimizer,
    layout: Layout,
    placement: BootPlacement,
    transpose_keys: TransposeKeys,
    encoder: Encoder,
    cl: CompactLayout,
    state: CompactState<E::Ct>,
    meta: Vec<ActMeta>,
    levels_per_iter: usize,
    micro: usize,
    trace: Vec<(String, usize)>,
    weight_boots: u64,
    act_boots: u64,
    boot_seconds: f64,
    last_booted: bool,
    next_full: Option<bool>,
    act_outputs: Vec<Vec<E::Ct>>,
    frozen_biases: bool,
    /// The reference lr of the stored velocity: `U = lr_ref (1 + mu) V` at every step
    /// (`docs/layout.md` §7.1); the step's lr must not exceed it, so no stored quantity is ever
    /// scaled up.
    lr_ref: f64,
    /// Whether the last update used a second gradient mask (its lr differed from `lr_ref`).
    last_rescaled: bool,
    /// One entry per finished epoch (`refreshes` counts bootstraps, `refresh_seconds` their
    /// time, `batches` the updates).
    pub log: Vec<EpochLog>,
}

impl<E: HeOps, B: BootstrapOps<E>> std::fmt::Debug for NonInteractiveTrainer<'_, E, B> {
    /// A summary without ciphertexts.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NonInteractiveTrainer")
            .field("optim", &self.optim)
            .field("layout", &self.layout)
            .field("placement", &self.placement)
            .field("levels_per_iter", &self.levels_per_iter)
            .field("micro", &self.micro)
            .finish()
    }
}

/// The input scale of every layer's next forward (`Delta` for the batch; the product chain of
/// the activations, or `Delta` after an activation bootstrap): the bias of layer `l` is
/// encrypted / extracted at `inputs_scale[l] * Delta`.
fn input_scales<E: HeOps>(
    ev: &E,
    encoder: &Encoder,
    mlp_acts: &[&dyn Activation<E>],
    w_scales: &[f64],
    inputs: &[i64],
    boot_acts: bool,
) -> Result<Vec<f64>> {
    let probe = ScaleProbe::new(ev);
    let delta = ev.params().scale();
    let mut out = vec![delta];
    for l in 0..mlp_acts.len() - 1 {
        if boot_acts {
            out.push(delta);
            continue;
        }
        let level = usize::try_from(inputs[l]).map_err(|_| HekitError::NoLevelsLeft)?;
        let (zl, zs) = dense_out(ev, level, out[l], w_scales[l])?;
        let (_, s) = probe.activation(mlp_acts[l], encoder, zl, zs)?;
        out.push(s);
    }
    Ok(out)
}

/// `InvalidParams` unless every level is non-negative (the step fits).
fn check_fits(p: &Pass) -> Result<()> {
    if p.min < 0 {
        return Err(HekitError::InvalidParams(format!(
            "the step needs {} levels more than are left",
            -p.min
        )));
    }
    Ok(())
}

/// The start level after a bootstrap and an unpack: `output_level - 1`.
fn start_after(output_level: usize) -> usize {
    output_level.saturating_sub(1)
}

impl NonInteractiveTrainer<'_> {
    /// The level of the unpacked weights after a bootstrap: `bs.output_level() - 1` (15 on
    /// n16_boot). Encrypt the initial network and the input batches here
    /// ([`NonInteractiveTrainer::encrypt_network`], [`NonInteractiveTrainer::batch_spec`]).
    pub fn start_level_for(bs: &Bootstrapper) -> usize {
        start_after(bs.output_level())
    }
}

impl<'a, E: HeOps, B: BootstrapOps<E>> NonInteractiveTrainer<'a, E, B> {
    /// Encrypts (client side) the initial network for non-interactive training with
    /// `placement`: the expanded [`Mlp`] at [`NonInteractiveTrainer::start_level_for`] (the
    /// first step sees the levels and scales of every later one, Task 40 carry-in) with each
    /// bias at the exact product scale of its first forward, and the [`CompactState`] at
    /// `bs.output_level()` (as a bootstrap leaves it) with zero velocity.
    ///
    /// Errors: `InvalidParams` if `layers` is empty, `acts` has another length, or the shapes
    /// do not chain or fit `layout`; those of [`Dense::encrypt_at`] and [`CompactState::encrypt`];
    /// `NoLevelsLeft` if the forward does not fit.
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_network(
        ev: &E,
        bs: &B,
        layers: &[(Array2<f64>, Vec<f64>)],
        acts: Vec<Box<dyn Activation<E>>>,
        layout: &Layout,
        placement: BootPlacement,
        enc: &Encryptor,
        encoder: &Encoder,
        rng: &mut Rng,
    ) -> Result<(Mlp<E>, CompactState<E::Ct>)> {
        let start = start_after(bs.output_level());
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
        // The input scale of every layer's first forward on the placement's chain (the
        // bootstrapped activations restart at Delta); each bias is encrypted once, at its exact
        // product scale (the weights are at Delta).
        let refs: Vec<&dyn Activation<E>> = acts.iter().map(|a| a.as_ref()).collect();
        let meta = metas_of(&refs);
        let n = meta.len();
        let s = start as i64;
        let p = pass_levels(
            &meta,
            placement.boots_activations(),
            &vec![s; n],
            &vec![s; n],
            s,
            bs.output_level() as i64,
            true,
        );
        let delta = encoder.params().scale();
        let w_scales = vec![delta; n];
        let scales = input_scales(
            ev,
            encoder,
            &refs,
            &w_scales,
            &p.inputs,
            placement.boots_activations(),
        )?;
        drop(refs);
        let mlp = Mlp {
            layers: layers
                .iter()
                .zip(acts)
                .zip(&scales)
                .map(|(((w, b), act), &sx)| {
                    Ok((
                        Dense::encrypt_at(w, b, layout, sx, start, enc, encoder, rng)?
                            .upload_owned(ev)?,
                        act,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let cl = CompactLayout::new(&mlp.dims(), layout)?;
        let state =
            CompactState::encrypt(&cl, layers, layout, bs.output_level(), enc, encoder, rng)?
                .upload_owned(ev)?;
        Ok((mlp, state))
    }

    /// A trainer on `ev` (relinearization key and the Galois keys of [`ni_rotation_steps`]) and
    /// `bs` (its own keys), for `mlp` (the expanded network at the start level) and its compact
    /// `state` (both from [`NonInteractiveTrainer::encrypt_network`]), with
    /// [`TransposeKeys::Dedicated`] ([`NonInteractiveTrainer::with_transpose_keys`]) and one
    /// micro-batch per update in [`NonInteractiveTrainer::train_epoch`]
    /// ([`NonInteractiveTrainer::with_micro_batches`]).
    ///
    /// Errors (`InvalidParams`): `layout` not of the parameters' slot count; `ev` and `bs` over
    /// different parameters; the network does not chain or does not fit `layout`; `state` not
    /// of the network's [`CompactLayout`]; **[`levels_per_iter`] above the levels left after a
    /// bootstrap and an unpack** (`bs.output_level() - 1`); or the levels of `mlp` and `state`
    /// do not allow the first step.
    pub fn new(
        ev: &'a E,
        bs: &'a B,
        optim: NiOptimizer,
        layout: Layout,
        placement: BootPlacement,
        mlp: &Mlp<E>,
        state: CompactState<E::Ct>,
    ) -> Result<NonInteractiveTrainer<'a, E, B>> {
        if layout.slots != ev.params().slots() {
            return Err(HekitError::InvalidParams(format!(
                "layout of {} slots for parameters with {} slots",
                layout.slots,
                ev.params().slots()
            )));
        }
        if bs.output_level() > ev.params().max_level() || bs.output_level() == 0 {
            return Err(HekitError::InvalidParams(
                "bootstrapper output level does not fit the evaluator's parameters".into(),
            ));
        }
        let dims = mlp.dims();
        if mlp
            .layers
            .windows(2)
            .any(|p| p[1].0.w.d_in() != p[0].0.w.d_out())
        {
            return Err(HekitError::InvalidParams(
                "layer shapes do not chain".into(),
            ));
        }
        let cl = CompactLayout::new(&dims, &layout)?;
        if state.weights.len() != cl.containers.len() || state.velocity.len() != cl.containers.len()
        {
            return Err(HekitError::InvalidParams(format!(
                "compact state of {} ciphertexts for a layout of {}",
                state.weights.len(),
                cl.containers.len()
            )));
        }
        let lpi = levels_per_iter(mlp, placement);
        let after_boot = start_after(bs.output_level());
        if lpi > after_boot {
            return Err(HekitError::InvalidParams(format!(
                "one step needs {lpi} levels, {after_boot} are left after a bootstrap and an \
                 unpack ({} placement)",
                placement.name()
            )));
        }
        let t = NonInteractiveTrainer {
            ev,
            bs,
            optim,
            layout,
            placement,
            transpose_keys: TransposeKeys::Dedicated,
            encoder: Encoder::new(Arc::clone(ev.params())),
            cl,
            state,
            meta: metas(mlp),
            levels_per_iter: lpi,
            micro: 1,
            trace: Vec::new(),
            weight_boots: 0,
            act_boots: 0,
            boot_seconds: 0.0,
            last_booted: false,
            next_full: None,
            act_outputs: Vec::new(),
            frozen_biases: false,
            lr_ref: optim.lr(),
            last_rescaled: false,
            log: Vec::new(),
        };
        let st = t.level_state(mlp);
        check_fits(&t.pass(&st, t.start_level() as i64, true))?;
        Ok(t)
    }

    /// Uses `keys` for the transposes (e.g. [`TransposeKeys::Reuse`] when key memory binds).
    pub fn with_transpose_keys(mut self, keys: TransposeKeys) -> NonInteractiveTrainer<'a, E, B> {
        self.transpose_keys = keys;
        self
    }

    /// The optimizer.
    pub fn optimizer(&self) -> NiOptimizer {
        self.optim
    }

    /// Sets the learning rate of the following updates (a per-step schedule, e.g. ReBoot's
    /// scheduler). With NAG it must not exceed the reference lr ([`NonInteractiveTrainer::set_lr_reference`],
    /// by default the optimizer's initial lr). The weight and bias gradients take it through
    /// their folded constants; with NAG the velocity stays stored at the reference lr and the
    /// update scales its contribution by `s = lr / lr_ref <= 1` inside the existing scalar
    /// product (`docs/layout.md` §7.1). When `s != 1` the velocity needs the gradient at the
    /// reference lr too: a second lane-0 mask on the same reduced products (`m` ct-pt, same
    /// level) and a second lane shift (`m - n_ct` rotations per layer,
    /// [`UpdateCost::ref_shift_rot`]). Setting `lr = lr_ref` is bit-identical to never setting
    /// it.
    ///
    /// Errors (`InvalidParams`): `lr` not finite and positive, or (NAG) above the reference lr.
    pub fn set_lr(&mut self, lr: f64) -> Result<()> {
        if !(lr.is_finite() && lr > 0.0) {
            return Err(HekitError::InvalidParams(format!(
                "learning rate {lr} must be finite and positive"
            )));
        }
        // SGD has no velocity: no reference to stay below (review n2).
        if self.optim.has_velocity() && lr > self.lr_ref {
            return Err(HekitError::InvalidParams(format!(
                "learning rate {lr} above the velocity's reference lr {} (set_lr_reference)",
                self.lr_ref
            )));
        }
        self.optim = self.optim.with_lr(lr);
        Ok(())
    }

    /// Sets the reference lr of the stored velocity (e.g. the schedule's maximum), before the
    /// first update (afterwards the stored velocity is formed at the old reference).
    ///
    /// Errors (`InvalidParams`): `lr_ref` not finite and positive, below the current lr, or a
    /// velocity already exists.
    pub fn set_lr_reference(&mut self, lr_ref: f64) -> Result<()> {
        if !(lr_ref.is_finite() && lr_ref > 0.0) || lr_ref < self.optim.lr() {
            return Err(HekitError::InvalidParams(format!(
                "reference lr {lr_ref} must be finite, positive and at least the lr {}",
                self.optim.lr()
            )));
        }
        if self.state.velocity.iter().any(Option::is_some) {
            return Err(HekitError::InvalidParams(
                "the reference lr can only be set before the first update".into(),
            ));
        }
        self.lr_ref = lr_ref;
        Ok(())
    }

    /// The velocity's reference lr.
    pub fn lr_reference(&self) -> f64 {
        self.lr_ref
    }

    /// Whether the last update used the reference-lr gradient besides the step's (its lr
    /// differed from the reference lr): add [`ref_shift_rotations`] to its closed form.
    pub fn last_update_rescaled(&self) -> bool {
        self.last_rescaled
    }

    /// `s = lr / lr_ref` (exactly 1 at the reference lr).
    fn lr_factor(&self) -> f64 {
        if self.optim.lr() == self.lr_ref {
            1.0
        } else {
            self.optim.lr() / self.lr_ref
        }
    }

    /// Never updates the biases: the bias gradient's folded constant is 0, so every op and every
    /// count is unchanged and the biases keep their encrypted values (encrypt them as zeros for
    /// a bias-free network, ReBoot's `bias=False`; ruling R-M3-12).
    pub fn with_frozen_biases(mut self) -> NonInteractiveTrainer<'a, E, B> {
        self.frozen_biases = true;
        self
    }

    /// Groups `k` consecutive batches of [`NonInteractiveTrainer::train_epoch`] into one update
    /// (`k` micro-batch ciphertexts, e.g. `B = 64` as 2 x 32 lanes, ruling R-M3-4).
    pub fn with_micro_batches(mut self, k: usize) -> NonInteractiveTrainer<'a, E, B> {
        self.micro = k.max(1);
        self
    }

    /// The layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// The compact layout of the state.
    pub fn compact_layout(&self) -> &CompactLayout {
        &self.cl
    }

    /// The compact state (weights, biases, velocity).
    pub fn state(&self) -> &CompactState<E::Ct> {
        &self.state
    }

    /// The placement.
    pub fn placement(&self) -> BootPlacement {
        self.placement
    }

    /// [`levels_per_iter`] of the network.
    pub fn levels_per_iter(&self) -> usize {
        self.levels_per_iter
    }

    /// The start level: [`NonInteractiveTrainer::start_level_for`] of the bootstrapper.
    pub fn start_level(&self) -> usize {
        start_after(self.bs.output_level())
    }

    /// Weight and velocity bootstraps so far.
    pub fn weight_bootstraps(&self) -> u64 {
        self.weight_boots
    }

    /// Activation bootstraps so far.
    pub fn activation_bootstraps(&self) -> u64 {
        self.act_boots
    }

    /// Whether the last update bootstrapped the weights.
    pub fn last_update_booted(&self) -> bool {
        self.last_booted
    }

    /// Wall-clock seconds spent in bootstraps so far (weights, velocities and activations).
    pub fn bootstrap_seconds(&self) -> f64 {
        self.boot_seconds
    }

    /// The bootstrapped hidden activations of the last update ([`BootPlacement::AfterActivation`]
    /// only; empty otherwise): `[micro-batch][l]` is the bootstrap output of `a_{l+1}` at
    /// `Delta`, canonical with period `pow2(d_{l+1})`. For a client-side diagnostic of the
    /// activation bootstraps' precision (the trainer itself never decrypts).
    pub fn last_activation_bootstraps(&self) -> &[Vec<E::Ct>] {
        &self.act_outputs
    }

    /// Declares whether the **next** step's micro-batches are all full (one shot: consumed by
    /// the next update's bootstrap decision). A lazy placement skips the weight bootstrap only
    /// if the step after it fits; without a declaration (`None`) that step is planned as a
    /// partial batch, which costs one more level at a linear output layer, so the decision is
    /// conservative. [`NonInteractiveTrainer::train_epoch_valid`] declares it for every update
    /// but the last of the epoch. A declared full step that turns out partial and does not fit
    /// is refused before any work (`InvalidParams`).
    pub fn set_next_batch_full(&mut self, full: Option<bool>) {
        self.next_full = full;
    }

    /// The levels of the last step (labels: `z{l}`, `a{l}`, `a{l}_to_delta`, `a{l}_boot`,
    /// `delta{L}`, `s{l}`, `dz{l}`, `delta{l-1}`, `h{l}`, `hb{l}`, then `c{i}'` for every
    /// container after the update), micro-batch 0.
    pub fn trace(&self) -> &[(String, usize)] {
        &self.trace
    }

    /// The structural prediction of the levels of the next step's trace (as
    /// [`NonInteractiveTrainer::trace`], from the current levels of `mlp` and the state, `x` at
    /// the start level), for tests.
    pub fn predicted_trace(&self, mlp: &Mlp<E>) -> Vec<(String, i64)> {
        let st = self.level_state(mlp);
        let p = self.pass(&st, self.start_level() as i64, true);
        let c = update_levels(&self.cl, &st, &p, self.optim.has_velocity());
        let mut t = p.trace;
        t.extend(c.iter().enumerate().map(|(i, v)| (format!("c{i}'"), *v)));
        t
    }

    /// Where the client encrypts a batch: `x` on [`NonInteractiveTrainer::start_level`] at
    /// `Delta`, `y` at exactly the output's level and scale of a step that starts from a
    /// bootstrap. (Steps of a lazy placement that start lower switch `y` to their output's
    /// scale with one ct-pt product on `y`.)
    ///
    /// Errors: `NoLevelsLeft` if the forward does not fit; the probe's errors.
    pub fn batch_spec(&self, mlp: &Mlp<E>) -> Result<BatchSpec> {
        let s = self.start_level() as i64;
        let n = self.meta.len();
        let p = pass_levels(
            &self.meta,
            self.placement.boots_activations(),
            &vec![s; n],
            &vec![s; n],
            s,
            self.bs.output_level() as i64,
            true,
        );
        let acts: Vec<&dyn Activation<E>> = mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
        let w: Vec<f64> = mlp.layers.iter().map(|(d, _)| d.w.scale()).collect();
        let scales = input_scales(
            self.ev,
            &self.encoder,
            &acts,
            &w,
            &p.inputs,
            self.placement.boots_activations(),
        )?;
        let last = n - 1;
        let level = usize::try_from(p.inputs[last]).map_err(|_| HekitError::NoLevelsLeft)?;
        let (zl, zs) = dense_out(self.ev, level, scales[last], w[last])?;
        let (y_level, y_scale) =
            ScaleProbe::new(self.ev).activation(acts[last], &self.encoder, zl, zs)?;
        Ok(BatchSpec {
            x_level: self.start_level(),
            x_scale: self.ev.params().scale(),
            y_level,
            y_scale,
        })
    }

    fn level_state(&self, mlp: &Mlp<E>) -> LevelState {
        LevelState {
            w: mlp.layers.iter().map(|(d, _)| d.w.level() as i64).collect(),
            b: mlp.layers.iter().map(|(d, _)| d.b.level() as i64).collect(),
            c: self
                .state
                .weights
                .iter()
                .map(|c| c.level() as i64)
                .collect(),
            u: self
                .state
                .velocity
                .iter()
                .map(|u| u.as_ref().map(|u| u.level() as i64))
                .collect(),
        }
    }

    fn pass(&self, st: &LevelState, x: i64, full: bool) -> Pass {
        pass_levels(
            &self.meta,
            self.placement.boots_activations(),
            &st.w,
            &st.b,
            x,
            self.bs.output_level() as i64,
            full,
        )
    }

    /// One update on one full micro-batch: [`NonInteractiveTrainer::train_step_valid`].
    ///
    /// Errors as [`NonInteractiveTrainer::train_step_valid`].
    pub fn train_batch(&mut self, mlp: &mut Mlp<E>, x: &E::Ct, y: &E::Ct) -> Result<()> {
        let b = self.layout.batch;
        self.train_step_valid(mlp, &[(x, y, b)])
    }

    /// One update on one micro-batch with `n_valid` samples.
    ///
    /// Errors as [`NonInteractiveTrainer::train_step_valid`].
    pub fn train_batch_valid(
        &mut self,
        mlp: &mut Mlp<E>,
        x: &E::Ct,
        y: &E::Ct,
        n_valid: usize,
    ) -> Result<()> {
        self.train_step_valid(mlp, &[(x, y, n_valid)])
    }

    /// One update on full micro-batches `(x, y)` (gradients accumulated before one update and
    /// at most one bootstrap of the state).
    ///
    /// Errors as [`NonInteractiveTrainer::train_step_valid`].
    pub fn train_step(&mut self, mlp: &mut Mlp<E>, batches: &[(E::Ct, E::Ct)]) -> Result<()> {
        let b = self.layout.batch;
        let v: Vec<_> = batches.iter().map(|(x, y)| (x, y, b)).collect();
        self.train_step_valid(mlp, &v)
    }

    /// One update (see the [module docs](self)) on the micro-batches `(x, y, n_valid)`, each
    /// encrypted as [`NonInteractiveTrainer::batch_spec`] says, with `n_valid` samples in its
    /// first lanes. The gradients are means over all valid samples. Afterwards the weights and
    /// biases of `mlp` are unpacked for the next step, and the compact state is updated (and
    /// bootstrapped when the placement says so).
    ///
    /// Errors, reported before anything is written (`mlp` and the state are unchanged; the
    /// counter may have moved): `InvalidParams` if `batches` is empty, `mlp` is not the
    /// trainer's network, `n_valid` is not in `1..=batch`, a key is missing, or the step does
    /// not fit the current levels; `ScaleMismatch` if a `y` cannot be brought to the output's
    /// exact scale; and the errors of the layer operations and of the bootstrapper.
    pub fn train_step_valid(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(&E::Ct, &E::Ct, usize)],
    ) -> Result<()> {
        self.step(mlp, batches, false).map(|_| ())
    }

    /// [`NonInteractiveTrainer::train_step_valid`], returning the update's hidden activations:
    /// per micro-batch the outputs of the activations of layers `1 .. L - 1` (1-based), from
    /// the forward pass with the weights **before** the update, at their own levels and scales
    /// (not the bootstrapped copies of [`BootPlacement::AfterActivation`]). A block of
    /// [`crate::block::BlockMlp`] hands the first one to the next stage (Task 52).
    ///
    /// Errors as [`NonInteractiveTrainer::train_step_valid`].
    pub fn train_step_valid_with_hidden(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(&E::Ct, &E::Ct, usize)],
    ) -> Result<Vec<Vec<E::Ct>>> {
        self.step(mlp, batches, true)
    }

    fn step(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(&E::Ct, &E::Ct, usize)],
        keep_hidden: bool,
    ) -> Result<Vec<Vec<E::Ct>>> {
        let n_layers = mlp.layers.len();
        if batches.is_empty() {
            return Err(HekitError::InvalidParams("no micro-batches".into()));
        }
        if n_layers != self.cl.plans.len()
            || mlp
                .layers
                .iter()
                .zip(&self.cl.plans)
                .any(|((d, _), p)| (d.w.d_out(), d.w.d_in()) != (p.d_out, p.d_in))
        {
            return Err(HekitError::InvalidParams(
                "network is not the trainer's".into(),
            ));
        }
        let lanes = self.layout.batch;
        if batches.iter().any(|b| b.2 == 0 || b.2 > lanes) {
            return Err(HekitError::InvalidParams(format!(
                "valid sample counts {:?} for batches of {lanes}",
                batches.iter().map(|b| b.2).collect::<Vec<_>>()
            )));
        }
        let full = batches.iter().all(|b| b.2 == lanes);
        let st = self.level_state(mlp);
        let x_level = batches.iter().map(|b| b.0.level()).min().expect("nonempty") as i64;
        let pass = self.pass(&st, x_level, full);
        check_fits(&pass)?;

        let ev = self.ev;
        let enc = &self.encoder;
        let layout = self.layout;
        let delta_scale = ev.params().scale();
        let boot_acts = self.placement.boots_activations();
        let mut trace: Vec<(String, usize)> = Vec::new();
        let mut t_boot = 0.0;
        let mut act_boots = 0u64;

        // 1. Forward per micro-batch. acts[mb][l] is the input of layer l.
        let mut acts: Vec<Vec<E::Ct>> = Vec::with_capacity(batches.len());
        let mut zs: Vec<Vec<E::Ct>> = Vec::with_capacity(batches.len());
        let mut hidden: Vec<Vec<E::Ct>> = Vec::new();
        for (mb, (x, _, _)) in batches.iter().enumerate() {
            let mut h_mb = Vec::new();
            let mut a_in = Vec::with_capacity(n_layers + 1);
            let mut z_mb = Vec::with_capacity(n_layers);
            let mut prev = (*x).clone();
            for (l, (dense, act)) in mlp.layers.iter().enumerate() {
                let level = prev.level().min(dense.w.level()).min(dense.b.level());
                let input = ev.mod_switch_to_level(&prev, level)?;
                if l == 0 {
                    a_in.push(input.clone());
                }
                let z = dense.forward(ev, &input, &layout)?;
                let a = act.forward(ev, enc, &z)?;
                if mb == 0 {
                    trace.push((format!("z{}", l + 1), z.level()));
                    trace.push((format!("a{}", l + 1), a.level()));
                }
                z_mb.push(z);
                if keep_hidden && l + 1 < n_layers {
                    h_mb.push(a.clone());
                }
                if boot_acts && l + 1 < n_layers {
                    let a1 = ev.rescale(&ev.mul_scalar(&a, 1.0, delta_scale)?)?;
                    let t0 = Instant::now();
                    let ab = self.bs.bootstrap(ev, &ev.mod_switch_to_level(&a1, 0)?)?;
                    t_boot += t0.elapsed().as_secs_f64();
                    act_boots += 1;
                    if mb == 0 {
                        trace.push((format!("a{}_to_delta", l + 1), a1.level()));
                        trace.push((format!("a{}_boot", l + 1), ab.level()));
                    }
                    a_in.push(ab.clone());
                    prev = ab;
                } else {
                    a_in.push(a.clone());
                    prev = a;
                }
            }
            acts.push(a_in);
            zs.push(z_mb);
            if keep_hidden {
                hidden.push(h_mb);
            }
        }

        // 2. Loss gradients (y brought to the output's exact scale if a lazy step starts low).
        let mut deltas = Vec::with_capacity(batches.len());
        for (mb, (_, y, _)) in batches.iter().enumerate() {
            let out = &acts[mb][n_layers];
            let y = align_target(ev, y, out)?;
            deltas.push(Mse.grad(ev, out, &y)?);
        }
        trace.push((format!("delta{n_layers}"), deltas[0].level()));

        // 3. Backward and the compact gradients, top down; nothing is written yet.
        let n_total: usize = batches.iter().map(|b| b.2).sum();
        let n = n_total as f64;
        let c_w = self.optim.grad_scale() / n;
        let c_b = if self.frozen_biases {
            0.0
        } else {
            self.optim.bias_grad_scale() / n
        };
        let masks: Vec<Vec<f64>> = batches
            .iter()
            .map(|b| valid_lane_mask(&layout, b.2))
            .collect();
        let mut h_w: Vec<Vec<E::Ct>> = vec![Vec::new(); n_layers];
        let rescale = self.optim.has_velocity() && self.lr_factor() != 1.0;
        let mut h_w_ref: Vec<Option<Vec<E::Ct>>> = vec![None; n_layers];
        let mut h_b: Vec<Option<E::Ct>> = vec![None; n_layers];
        for i in (0..n_layers).rev() {
            let ln = i + 1;
            let (dense, act) = &mlp.layers[i];
            let mut dzs = Vec::with_capacity(batches.len());
            for (mb, b) in batches.iter().enumerate() {
                let dz = match act.constant_derivative() {
                    Some(k) => constant_dz(ev, enc, &deltas[mb], k, &masks[mb], b.2 == lanes)?,
                    None => {
                        let s = act.derivative_factor_at(
                            ev,
                            enc,
                            &zs[mb][i],
                            1.0,
                            Some(&masks[mb]),
                            None,
                        )?;
                        if mb == 0 {
                            trace.push((format!("s{ln}"), s.level()));
                        }
                        let lv = s.level().min(deltas[mb].level());
                        ev.rescale(&ev.mul(
                            &ev.mod_switch_to_level(&deltas[mb], lv)?,
                            &ev.mod_switch_to_level(&s, lv)?,
                        )?)?
                    }
                };
                dzs.push(dz);
            }
            // Partial and full micro-batches can differ by one level at a linear layer.
            let low = dzs.iter().map(|d| d.level()).min().expect("nonempty");
            let dzs = dzs
                .iter()
                .map(|d| ev.mod_switch_to_level(d, low))
                .collect::<Result<Vec<_>>>()?;
            trace.push((format!("dz{ln}"), dzs[0].level()));
            if i > 0 {
                let w = &dense.w;
                let lowered = EncWeights::from_diagonals(
                    w.diagonals().mod_switch_to_level(ev, low)?,
                    w.d_out(),
                    w.d_in(),
                    &layout,
                )?;
                let wt = lowered.transpose_with(ev, &layout, &self.transpose_keys)?;
                let mut next = Vec::with_capacity(batches.len());
                for dz in &dzs {
                    next.push(dense.backward_input(ev, dz, &wt, &layout)?);
                }
                trace.push((format!("delta{i}"), next[0].level()));
                deltas = next;
            }
            let dz_refs: Vec<&E::Ct> = dzs.iter().collect();
            let a_refs: Vec<&E::Ct> = acts.iter().map(|a| &a[i]).collect();
            // The velocity's gradient at the reference lr (a second mask) when the step's lr
            // differs from it (NAG only; docs/layout.md §7.1).
            let hw = if rescale {
                let c_ref = self.optim.with_lr(self.lr_ref).grad_scale() / n;
                let mut gs = dense.weight_grad_reduced_consts(
                    ev,
                    &dz_refs,
                    &a_refs,
                    &layout,
                    &[c_w, c_ref],
                )?;
                let g_ref = gs.pop().expect("two");
                let g = gs.pop().expect("two");
                h_w_ref[i] = Some(compact_shift(ev, &g_ref, &self.cl.plans[i])?);
                compact_shift(ev, &g, &self.cl.plans[i])?
            } else {
                let g = dense.weight_grad_reduced(ev, &dz_refs, &a_refs, &layout, c_w)?;
                compact_shift(ev, &g, &self.cl.plans[i])?
            };
            trace.push((format!("h{ln}"), hw[0].level()));
            let slot = self.cl.biases[i];
            let bplan = *self.cl.bias_plan(i);
            let gb = dense.bias_grad_reduced(
                ev,
                &dz_refs,
                &layout,
                c_b,
                bplan.diag_len(),
                delta_scale,
            )?;
            let gb = ev.rotate(&gb, -(bplan.offset(slot.position) as i32))?;
            trace.push((format!("hb{ln}"), gb.level()));
            h_w[i] = hw;
            h_b[i] = Some(gb);
        }

        // 4. The optimizer on the compact state (on copies).
        let (mut weights, mut velocity) = self.update_state(&h_w, &h_w_ref, &h_b)?;
        for (i, w) in weights.iter().enumerate() {
            trace.push((format!("c{i}'"), w.level()));
        }

        // 5. Bootstrap the state if the placement says so.
        let nag = self.optim.has_velocity();
        let mut next = LevelState {
            w: Vec::new(),
            b: Vec::new(),
            c: weights.iter().map(|c| c.level() as i64).collect(),
            u: velocity
                .iter()
                .map(|u| u.as_ref().map(|u| u.level() as i64))
                .collect(),
        };
        (next.w, next.b) = unpacked(&self.cl, &next.c);
        let start = self.start_level() as i64;
        let next_full = self.next_full.take();
        let boot = must_boot(
            &self.meta,
            self.placement,
            &next,
            start,
            self.bs.output_level() as i64,
            next_full,
        );
        let mut weight_boots = 0;
        if boot {
            let t0 = Instant::now();
            for ct in weights.iter_mut().chain(velocity.iter_mut().flatten()) {
                *ct = self.bs.bootstrap(ev, &ev.mod_switch_to_level(ct, 0)?)?;
                weight_boots += 1;
            }
            t_boot += t0.elapsed().as_secs_f64();
            next.c = weights.iter().map(|c| c.level() as i64).collect();
            (next.w, next.b) = unpacked(&self.cl, &next.c);
        }
        debug_assert!(
            !nag || velocity
                .iter()
                .zip(&self.cl.containers)
                .all(|(u, c)| u.is_some() == c.layer.is_some())
        );

        // 6. Unpack the weights and extract the biases for the next forward.
        let mut new_w = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let range = self.cl.layer_containers(l);
            let low = range
                .clone()
                .map(|c| weights[c].level())
                .min()
                .expect("one");
            let cts = range
                .map(|c| ev.mod_switch_to_level(&weights[c], low))
                .collect::<Result<Vec<_>>>()?;
            new_w.push(compact_unpack(ev, &cts, &self.cl.plans[l])?);
        }
        let next_pass = self.pass(&next, start, true);
        let acts_dyn: Vec<&dyn Activation<E>> =
            mlp.layers.iter().map(|(_, a)| a.as_ref()).collect();
        let w_scales: Vec<f64> = new_w.iter().map(|w| w.scale()).collect();
        let scales = input_scales(ev, enc, &acts_dyn, &w_scales, &next_pass.inputs, boot_acts)?;
        let mut new_b = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let slot = self.cl.biases[l];
            let plan = self.cl.containers[slot.container].plan;
            let need = next_pass.inputs[l].max(0) as usize;
            new_b.push(extract_bias(
                ev,
                &weights[slot.container],
                &plan,
                slot.position,
                scales[l] * w_scales[l],
                need,
            )?);
        }

        // Commit.
        for (l, (w, b)) in new_w.into_iter().zip(new_b).enumerate() {
            let layer = &mut mlp.layers[l].0;
            layer.w = EncWeights::from_diagonals(
                w.into_diagonals(),
                layer.w.d_out(),
                layer.w.d_in(),
                &layout,
            )?;
            layer.b = b;
        }
        self.state = CompactState { weights, velocity };
        self.weight_boots += weight_boots;
        self.act_boots += act_boots;
        self.boot_seconds += t_boot;
        self.last_booted = boot;
        self.last_rescaled = rescale;
        self.act_outputs = if boot_acts {
            acts.into_iter().map(|a| a[1..n_layers].to_vec()).collect()
        } else {
            Vec::new()
        };
        self.trace = trace;
        Ok(hidden)
    }

    /// The optimizer on every container (see the [module docs](self)): per container the level
    /// `min(W, h_W, h_b of its biases, U - 1)`, `W' = W - h_W - sum h_b - kappa U` and
    /// `U' = mu U + h_W` (NAG), `W' = W - h_W - sum h_b` (SGD; bias-only containers).
    #[allow(clippy::type_complexity)]
    fn update_state(
        &self,
        h_w: &[Vec<E::Ct>],
        h_w_ref: &[Option<Vec<E::Ct>>],
        h_b: &[Option<E::Ct>],
    ) -> Result<(Vec<E::Ct>, Vec<Option<E::Ct>>)> {
        let ev = self.ev;
        let delta = ev.params().scale();
        let mut weights = Vec::with_capacity(self.cl.containers.len());
        let mut velocity = Vec::with_capacity(self.cl.containers.len());
        for (c, spec) in self.cl.containers.iter().enumerate() {
            let w = &self.state.weights[c];
            let hw = spec.layer.map(|l| &h_w[l][spec.index]);
            let hbs: Vec<&E::Ct> = self
                .cl
                .biases
                .iter()
                .enumerate()
                .filter(|(_, s)| s.container == c)
                .map(|(l, _)| h_b[l].as_ref().expect("every layer's bias gradient"))
                .collect();
            let u = self.state.velocity[c]
                .as_ref()
                .filter(|_| spec.layer.is_some());
            let mut lv = w.level();
            if let Some(h) = hw {
                lv = lv.min(h.level());
            }
            for h in &hbs {
                lv = lv.min(h.level());
            }
            let nag = match self.optim {
                NiOptimizer::Nag { momentum, .. } if spec.layer.is_some() => Some(momentum),
                _ => None,
            };
            if let (Some(_), Some(u)) = (nag, u) {
                if u.level() == 0 {
                    return Err(HekitError::LevelMismatch);
                }
                lv = lv.min(u.level() - 1);
            }
            let down = |ct: &E::Ct| ev.mod_switch_to_level(ct, lv);
            let mut w_new = down(w)?;
            if let Some(h) = hw {
                w_new = ev.sub(&w_new, &down(h)?)?;
            }
            for h in &hbs {
                w_new = ev.sub(&w_new, &down(h)?)?;
            }
            // The velocity's gradient term at the reference lr: the second mask's output when
            // the step's lr differs from it, else the step's own gradient.
            let h_vel = spec
                .layer
                .and_then(|l| h_w_ref[l].as_ref().map(|v| &v[spec.index]));
            let s_lr = self.lr_factor();
            let u_new = match (nag, hw) {
                (Some(mu), Some(h)) => {
                    let h = down(h_vel.unwrap_or(h))?;
                    Some(match u {
                        None => h,
                        Some(u) => {
                            // U is stored at the reference lr; this step's W takes
                            // s = lr / lr_ref <= 1 of it, folded into kappa (no extra op or level;
                            // s = 1 exactly at the reference lr, so the products are
                            // bit-identical to the constant-lr trainer).
                            let kappa = mu * mu / (1.0 + mu) * s_lr;
                            let ue = ev.mod_switch_to_level(u, lv + 1)?;
                            let mu_u = ev.rescale(&ev.mul_scalar(&ue, mu, delta)?)?;
                            let k_u = ev.rescale(&ev.mul_scalar(&ue, kappa, delta)?)?;
                            w_new = ev.sub(&w_new, &k_u)?;
                            ev.add(&mu_u, &h)?
                        }
                    })
                }
                _ => None,
            };
            weights.push(w_new);
            velocity.push(u_new);
        }
        Ok((weights, velocity))
    }

    /// One epoch: [`NonInteractiveTrainer::train_step`] on every group of
    /// [`NonInteractiveTrainer::with_micro_batches`] consecutive full `(x, y)` (a last,
    /// shorter group is one update too), then an [`EpochLog`] appended to
    /// [`NonInteractiveTrainer::log`] and returned (`batches` = updates, `refreshes` =
    /// bootstraps, `refresh_seconds` = their time).
    ///
    /// Errors as [`NonInteractiveTrainer::train_epoch_valid`].
    pub fn train_epoch(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(E::Ct, E::Ct)],
    ) -> Result<EpochLog> {
        let b = self.layout.batch;
        let v: Vec<_> = batches
            .iter()
            .map(|(x, y)| (x.clone(), y.clone(), b))
            .collect();
        self.train_epoch_valid(mlp, &v)
    }

    /// [`NonInteractiveTrainer::train_epoch`] with the valid sample count of every batch (e.g.
    /// a last, partial one). Before every update but the last it declares whether the next
    /// group is full ([`NonInteractiveTrainer::set_next_batch_full`]), so a lazy placement plans
    /// its bootstraps around the partial batches.
    ///
    /// Errors: the first error of an update (the epoch is not logged; the updates before it
    /// stay trained).
    pub fn train_epoch_valid(
        &mut self,
        mlp: &mut Mlp<E>,
        batches: &[(E::Ct, E::Ct, usize)],
    ) -> Result<EpochLog> {
        let before = self.ev.counter();
        let (b0, bs0) = (self.weight_boots + self.act_boots, self.boot_seconds);
        let t0 = Instant::now();
        let (lanes, micro) = (self.layout.batch, self.micro);
        let groups: Vec<_> = batches.chunks(micro).collect();
        let full = |g: &[(E::Ct, E::Ct, usize)]| {
            // Full = every lane valid, whatever the number of ciphertexts: the definition of
            // the level pass (`train_step_valid`) and of `boot_schedule`'s `full`.
            g.iter().all(|b| b.2 == lanes)
        };
        for (i, group) in groups.iter().enumerate() {
            self.next_full = groups.get(i + 1).map(|g| full(g));
            let refs: Vec<_> = group.iter().map(|(x, y, n)| (x, y, *n)).collect();
            self.train_step_valid(mlp, &refs)?;
        }
        let log = EpochLog {
            epoch: self.log.len(),
            counter: &self.ev.counter() - &before,
            seconds: t0.elapsed().as_secs_f64(),
            refresh_seconds: self.boot_seconds - bs0,
            loss: None,
            batches: groups.len(),
            refreshes: (self.weight_boots + self.act_boots - b0) as usize,
        };
        self.log.push(log.clone());
        Ok(log)
    }
}

/// `y` at `out`'s exact scale and at most its level: switched down if the scales already agree,
/// else multiplied by 1 at `out.level() + 1` and rescaled onto exactly `out.scale()` (one ct-pt,
/// the level is `y`'s, off the critical path).
fn align_target<E: HeOps>(ev: &E, y: &E::Ct, out: &E::Ct) -> Result<E::Ct> {
    if y.scale() == out.scale() {
        return Ok(y.clone());
    }
    if y.level() <= out.level() {
        return Err(HekitError::ScaleMismatch);
    }
    let y = ev.mod_switch_to_level(y, out.level() + 1)?;
    ev.rescale(&ev.mul_scalar(&y, 1.0, out.scale())?)
}

/// [`compact_extract`] of a bias at exactly `target`, from the container's level down to
/// `need` (the level its next forward runs at): a target the rescale of one level cannot reach
/// (`ScaleMismatch`) is retried one level lower.
fn extract_bias<E: HeOps>(
    ev: &E,
    ct: &E::Ct,
    plan: &CompactPlan,
    t: usize,
    target: f64,
    need: usize,
) -> Result<E::Ct> {
    let mut level = ct.level();
    loop {
        let src = ev.mod_switch_to_level(ct, level)?;
        match compact_extract(ev, &src, plan, t, target) {
            Err(HekitError::ScaleMismatch) if level > need + 1 => level -= 1,
            r => return r,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::mlp_key_pool;

    const EMLP1: [usize; 3] = [784, 32, 10];

    fn emlp1_meta() -> Vec<ActMeta> {
        vec![
            ActMeta {
                depth: 1,
                dderiv: 1,
                constant: None,
            },
            ActMeta {
                depth: 0,
                dderiv: 0,
                constant: Some(1.0),
            },
        ]
    }

    /// The eMLP-1 numbers of `docs/layout.md` §7.3, §7.4 and §7.6 (no encryption).
    #[test]
    fn emlp1_compact_layout_levels_and_keys() {
        let layout = Layout::new(1 << 15, 1024, 32).unwrap();
        let cl = CompactLayout::new(&EMLP1, &layout).unwrap();
        assert_eq!(
            cl.containers.len(),
            2,
            "one container per layer, no bias-only one"
        );
        let slots: Vec<(usize, usize, usize)> = cl
            .biases
            .iter()
            .enumerate()
            .map(|(l, s)| (s.container, s.position, cl.bias_plan(l).offset(s.position)))
            .collect();
        assert_eq!(slots, vec![(1, 16, 16), (1, 32, 1024)]);
        assert_eq!(cl.bootstraps(true), 4);
        let reuse = TransposeKeys::Reuse(mlp_key_pool(&EMLP1, &layout).unwrap());
        assert_eq!(
            ni_rotation_steps(&EMLP1, &layout, &reuse).unwrap().len(),
            35
        );

        let meta = emlp1_meta();
        let p = pass_levels(&meta, false, &[15, 15], &[15, 15], 15, 16, true);
        let get = |k: &str| p.trace.iter().find(|(l, _)| l == k).unwrap().1;
        for (k, v) in [
            ("z1", 14),
            ("a1", 13),
            ("z2", 12),
            ("a2", 12),
            ("delta2", 12),
            ("dz2", 12),
            ("delta1", 11),
            ("s1", 13),
            ("dz1", 10),
            ("h1", 8),
            ("hb1", 9),
            ("h2", 10),
            ("hb2", 11),
        ] {
            assert_eq!(get(k), v, "{k}");
        }
        assert_eq!(15 - p.min, 7);
        let st = LevelState {
            w: vec![15; 2],
            b: vec![15; 2],
            c: vec![16; 2],
            u: vec![Some(16); 2],
        };
        assert_eq!(update_levels(&cl, &st, &p, true), vec![8, 9]);
        let a = pass_levels(&meta, true, &[15, 15], &[15, 15], 15, 16, true);
        assert_eq!(15 - a.min, 5);
        // The lazy schedule: a second step from the unpacked state fits, a third does not.
        let c1 = update_levels(&cl, &st, &p, true);
        let (w, b) = unpacked(&cl, &c1);
        assert_eq!((w.clone(), b.clone()), (vec![7, 8], vec![8, 8]));
        let p2 = pass_levels(&meta, false, &w, &b, 15, 16, true);
        assert_eq!(p2.min, 0);
    }
}
