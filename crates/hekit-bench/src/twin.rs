//! The plaintext f64 twin of the encrypted trainers (Tasks 32 and 41, shared by the `mnist`
//! and `plain` bins since Task 51).
//!
//! The network is a stack of affine layers with ReBoot's `z^2 + z` activation
//! ([`hekit_nn::activation::PolyRelu2`]) on every layer but the last (identity), trained on the
//! MSE against one-hot targets (output delta `a - y`) with Nesterov momentum: the update rule of
//! the encrypted trainers ([`twin_step`]).

use hekit_core::sampler::Rng;
use ndarray::{Array2, Axis};

/// One affine layer: `z = a W^T + b`.
#[derive(Debug, Clone)]
pub struct Layer {
    /// `d_out x d_in`.
    pub w: Array2<f64>,
    /// `d_out` biases.
    pub b: Vec<f64>,
}

/// `a W^T + b` for every row of `a`.
pub fn affine(a: &Array2<f64>, l: &Layer) -> Array2<f64> {
    let mut z = a.dot(&l.w.t());
    for mut row in z.rows_mut() {
        row.iter_mut().zip(&l.b).for_each(|(v, b)| *v += b);
    }
    z
}

/// `z^2 + z` on every layer but the last (identity).
pub fn forward_plain(net: &[Layer], x: &Array2<f64>) -> Array2<f64> {
    let mut a = x.clone();
    for (i, l) in net.iter().enumerate() {
        let z = affine(&a, l);
        a = if i + 1 == net.len() {
            z
        } else {
            z.mapv(|v| v * v + v)
        };
    }
    a
}

/// Weight decay of the twin (plain mode, ruling R-M3-14), in our batch-mean units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightDecay {
    /// The decay on the batch-mean gradient: ReBoot's `weight_decay / B` when the lr is given
    /// in `--lr-sum` units (ReBoot sums the gradient), else `--weight-decay` as is.
    pub mean: f64,
    /// `false`: coupled (torch SGD `weight_decay`: `g += wd W` before the momentum buffer and
    /// Nesterov); `true`: decoupled (`W -= lr wd W` beside the step).
    pub decoupled: bool,
}

impl WeightDecay {
    /// No decay.
    pub const NONE: WeightDecay = WeightDecay {
        mean: 0.0,
        decoupled: false,
    };
}

/// One plaintext NAG step, the trainer's update rule: `delta = a - y`, `G = dz^T a_in / n`,
/// `V' = mu V + G`, `W' = W - lr (G + mu V')`, biases `b' = b - bias_lr sum(dz) / n`, with the
/// weight decay `wd` on the weights only.
#[allow(clippy::too_many_arguments)]
pub fn twin_step(
    net: &mut [Layer],
    vel: &mut [Array2<f64>],
    x: &Array2<f64>,
    y: &Array2<f64>,
    lr: f64,
    mu: f64,
    bias_lr: f64,
    wd: WeightDecay,
) {
    let n = x.nrows() as f64;
    let last = net.len() - 1;
    let mut acts = vec![x.clone()];
    let mut zs = Vec::new();
    for (i, l) in net.iter().enumerate() {
        let z = affine(acts.last().expect("input"), l);
        acts.push(if i == last {
            z.clone()
        } else {
            z.mapv(|v| v * v + v)
        });
        zs.push(z);
    }
    let mut delta = acts.last().expect("output") - y;
    for i in (0..net.len()).rev() {
        let dz = if i == last {
            delta.clone()
        } else {
            &delta * &zs[i].mapv(|v| 2.0 * v + 1.0)
        };
        let next = dz.dot(&net[i].w);
        let mut g = dz.t().dot(&acts[i]) / n;
        let w_old = net[i].w.clone();
        if wd.mean != 0.0 && !wd.decoupled {
            // Coupled L2 (torch SGD weight_decay): on the weights, not the biases.
            g = &g + &(wd.mean * &w_old);
        }
        vel[i] = mu * &vel[i] + &g;
        net[i].w = &net[i].w - &(lr * &(&g + &(mu * &vel[i])));
        if wd.mean != 0.0 && wd.decoupled {
            net[i].w = &net[i].w - &((lr * wd.mean) * &w_old);
        }
        let gb = dz.sum_axis(Axis(0)) / n;
        net[i]
            .b
            .iter_mut()
            .zip(&gb)
            .for_each(|(b, g)| *b -= bias_lr * g);
        delta = next;
    }
}

/// The activation a block hands to the next stage (Task 52, ReBoot's eMLP-2/3): `z^2 + z` of
/// the stage's first layer.
pub fn block_hidden(stage: &[Layer], x: &Array2<f64>) -> Array2<f64> {
    affine(x, &stage[0]).mapv(|v| v * v + v)
}

/// The model output of a block network (`stages`: the local blocks, then the output stack):
/// the input through every block's first layer ([`block_hidden`]), then the whole output stack
/// ([`forward_plain`]). The local heads are not used at test time.
pub fn block_forward_plain(stages: &[Vec<Layer>], x: &Array2<f64>) -> Array2<f64> {
    let (output, blocks) = stages.split_last().expect("an output stack");
    let mut a = x.clone();
    for b in blocks {
        a = block_hidden(b, &a);
    }
    forward_plain(output, &a)
}

/// One step of ReBoot's block training (R-REB-5): every stage's input is computed first from
/// the weights before the step (the batch, then each block's [`block_hidden`]); then every
/// stage takes its own [`twin_step`] on its own labels `ys[s]` (local loss for a block, the
/// main loss for the output stack). No gradient crosses a stage (`twin_step` sends none below
/// a network's first layer), losses are not summed, and every layer has its own velocity, all
/// at the same lr. The update order is irrelevant (no stage reads another's new weights).
#[allow(clippy::too_many_arguments)]
pub fn block_twin_step(
    stages: &mut [Vec<Layer>],
    vels: &mut [Vec<Array2<f64>>],
    x: &Array2<f64>,
    ys: &[&Array2<f64>],
    lr: f64,
    mu: f64,
    bias_lr: f64,
    wd: WeightDecay,
) {
    assert_eq!(ys.len(), stages.len(), "one label matrix per stage");
    let mut inputs = vec![x.clone()];
    for st in &stages[..stages.len() - 1] {
        let h = block_hidden(st, inputs.last().expect("input"));
        inputs.push(h);
    }
    for (s, st) in stages.iter_mut().enumerate() {
        twin_step(st, &mut vels[s], &inputs[s], ys[s], lr, mu, bias_lr, wd);
    }
}

/// The accuracy with the argmax over the first `cols` outputs (a padded output layer has extra
/// rows trained to 0).
pub fn accuracy_over(out: &Array2<f64>, labels: &[u8], cols: usize) -> f64 {
    let hits = out
        .rows()
        .into_iter()
        .zip(labels)
        .filter(|(row, l)| {
            let arg = row
                .iter()
                .take(cols)
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(j, _)| j)
                .expect("nonempty row");
            arg == **l as usize
        })
        .count();
    hits as f64 / labels.len().max(1) as f64
}

/// The bins' ChaCha20 stream for `seed` and `role` (key bytes `[role, seed, seed, ...]`).
pub fn rng(seed: u8, role: u8) -> Rng {
    let mut s = [seed; 32];
    s[0] = role;
    Rng::from_seed(s)
}

/// A uniform draw in `[-1, 1)` (53 random bits).
pub fn uni(r: &mut Rng) -> f64 {
    2.0 * ((r.next_u64() >> 11) as f64 * 2f64.powi(-53)) - 1.0
}

/// The initial network: `pytorch` = `nn.Linear`'s default, W and b uniform in
/// `±1/sqrt(fan_in)`; `xavier` = W uniform in `±sqrt(1/(fan_in + fan_out))` (ReBoot's printed
/// bound) and b from the same draw order. `no_bias` zeroes the biases (they are still drawn, so
/// the weights do not depend on it).
pub fn init_net(dims: &[usize], r: &mut Rng, init: &str, no_bias: bool) -> Vec<Layer> {
    dims.windows(2)
        .map(|p| {
            let bound = if init == "xavier" {
                (1.0 / (p[0] + p[1]) as f64).sqrt()
            } else {
                1.0 / (p[0] as f64).sqrt()
            };
            let w = Array2::from_shape_fn((p[1], p[0]), |_| bound * uni(r));
            let b: Vec<f64> = (0..p[1]).map(|_| bound * uni(r)).collect();
            Layer {
                w,
                b: if no_bias { vec![0.0; p[1]] } else { b },
            }
        })
        .collect()
}

/// A uniformly random permutation of `0..n` (Fisher-Yates with [`Rng::below`]).
pub fn permutation(n: usize, r: &mut Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = r.below(i as u64 + 1) as usize;
        p.swap(i, j);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The twin's coupled weight decay is torch's SGD convention: `g += wd W` before the
    /// momentum buffer and the Nesterov combination (checked against the explicit formula).
    #[test]
    fn twin_weight_decay_is_coupled_l2_before_nesterov() {
        let mut r = rng(3, 4);
        let net0 = init_net(&[4, 3, 2], &mut r, "xavier", true);
        let x = Array2::from_shape_fn((5, 4), |_| uni(&mut r));
        let y = Array2::from_shape_fn((5, 2), |_| uni(&mut r));
        let (lr, mu, wd) = (0.1, 0.9, 0.05);
        let mut net = net0.clone();
        let mut vel: Vec<Array2<f64>> = net.iter().map(|l| Array2::zeros(l.w.dim())).collect();
        let decay = WeightDecay {
            mean: wd,
            decoupled: false,
        };
        // Two steps (the second uses a nonzero velocity).
        for _ in 0..2 {
            twin_step(&mut net, &mut vel, &x, &y, lr, mu, 0.0, decay);
        }
        // The explicit formula.
        let mut want = net0.clone();
        let mut v: Vec<Array2<f64>> = want.iter().map(|l| Array2::zeros(l.w.dim())).collect();
        for _ in 0..2 {
            let z1 = affine(&x, &want[0]);
            let a1 = z1.mapv(|t| t * t + t);
            let o = affine(&a1, &want[1]);
            let d2 = &o - &y;
            let d1 = d2.dot(&want[1].w) * z1.mapv(|t| 2.0 * t + 1.0);
            let n = x.nrows() as f64;
            let g = [d1.t().dot(&x) / n, d2.t().dot(&a1) / n];
            for i in 0..2 {
                let gi = &g[i] + &(wd * &want[i].w);
                v[i] = mu * &v[i] + &gi;
                want[i].w = &want[i].w - &(lr * &(&gi + &(mu * &v[i])));
            }
        }
        for (a, b) in net.iter().zip(&want) {
            let e =
                a.w.iter()
                    .zip(&b.w)
                    .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
            assert!(e < 1e-15, "{e}");
        }
    }

    /// The block step equals the explicit form: each stage's local NAG step on the input
    /// formed from the old weights; block 0 does not see the output stack's labels.
    #[test]
    fn block_twin_step_is_stagewise() {
        let mut r = rng(5, 4);
        let s0 = init_net(&[4, 6, 3], &mut r, "xavier", true);
        let s1 = init_net(&[6, 5, 3], &mut r, "xavier", true);
        let x = Array2::from_shape_fn((7, 4), |_| uni(&mut r));
        let y = Array2::from_shape_fn((7, 3), |_| uni(&mut r));
        let y2 = Array2::from_shape_fn((7, 3), |_| uni(&mut r));
        let run = |yo: &Array2<f64>| {
            let mut st = vec![s0.clone(), s1.clone()];
            let mut v: Vec<Vec<Array2<f64>>> = st
                .iter()
                .map(|n| n.iter().map(|l| Array2::zeros(l.w.dim())).collect())
                .collect();
            block_twin_step(
                &mut st,
                &mut v,
                &x,
                &[&y, yo],
                0.1,
                0.9,
                0.0,
                WeightDecay::NONE,
            );
            st
        };
        let a = run(&y);
        let b = run(&y2);
        assert_eq!(a[0][0].w, b[0][0].w);
        assert_ne!(a[1][0].w, b[1][0].w);
        // Stage 1 = a plain twin step on the block activation of the old weights.
        let mut want = s1.clone();
        let mut v: Vec<Array2<f64>> = want.iter().map(|l| Array2::zeros(l.w.dim())).collect();
        let h = block_hidden(&s0, &x);
        twin_step(&mut want, &mut v, &h, &y, 0.1, 0.9, 0.0, WeightDecay::NONE);
        assert_eq!(a[1][0].w, want[0].w);
        assert_eq!(a[1][1].w, want[1].w);
        // The model output goes through block 0's first layer and the whole output stack.
        let o = block_forward_plain(&a, &x);
        assert_eq!(o, forward_plain(&a[1], &block_hidden(&a[0], &x)));
    }

    #[test]
    fn permutation_is_a_seeded_permutation() {
        let p = permutation(100, &mut rng(1, 3));
        let mut s = p.clone();
        s.sort_unstable();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
        assert_ne!(p, (0..100).collect::<Vec<_>>(), "shuffled");
        assert_eq!(p, permutation(100, &mut rng(1, 3)), "deterministic");
        assert_ne!(p, permutation(100, &mut rng(2, 3)), "seed-dependent");
        assert!(permutation(0, &mut rng(1, 3)).is_empty());
    }

    #[test]
    fn accuracy_over_takes_the_first_columns() {
        let out = ndarray::arr2(&[[0.1, 0.9, 5.0], [0.8, 0.2, 5.0]]);
        assert_eq!(accuracy_over(&out, &[1, 0], 2), 1.0);
        assert_eq!(accuracy_over(&out, &[1, 0], 3), 0.0);
    }
}
