//! Synthetic CPU correctness only; the small CKKS parameters are not scientific presets.
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use ckks_train_ckks::Diagonals;
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::activation::{Activation, Identity, PolyRelu2, Square};
use ckks_train_nn::block::{BlockMlp, BlockTrainer};
use ckks_train_nn::dense::EncWeights;
use ckks_train_nn::dense::TransposeKeys;
use ckks_train_nn::layout::{Layout, unpack_activations};
use ckks_train_nn::optim::{Nag, Optimizer, Sgd};
use ckks_train_nn::refresh::{ClientRefresher, Refresher};
use ckks_train_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_rotation_steps};
use ndarray::{Array2, Axis, s};

const TOL: f64 = 1.0 / 1024.0;
const DIMS: [usize; 3] = [2, 2, 2];
type Net = Vec<(Array2<f64>, Vec<f64>)>;

fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}

struct Ctx {
    params: Arc<CkksParams>,
    encoder: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    layout: Layout,
}
impl Ctx {
    fn new() -> Self {
        Self::with_levels(10)
    }
    fn with_levels(levels: usize) -> Self {
        let mut q = vec![40; levels];
        q[0] = 60;
        let params = Arc::new(CkksParams::new(10, &q, &[60], 40, levels).unwrap());
        let layout = Layout::new(params.slots(), 4, 8).unwrap();
        let steps = mlp_rotation_steps(&DIMS, &layout, &TransposeKeys::Dedicated).unwrap();
        let mut kg = KeyGenerator::new(params.clone(), rng(51)).unwrap();
        let pk = kg.public_key();
        let rk = kg.relin_key();
        let gks = kg.galois_keys(&steps);
        Self {
            encoder: Encoder::new(params.clone()),
            enc: Encryptor::new(params.clone(), pk),
            dec: Decryptor::new(params.clone(), kg.secret_key()),
            ev: Evaluator::new(params.clone(), Some(rk), Some(gks)),
            params,
            layout,
        }
    }
    fn client(&self) -> ClientRefresher {
        ClientRefresher::new(
            self.dec.clone(),
            self.enc.clone(),
            self.encoder.clone(),
            rng(53),
        )
    }
    fn model(&self, initial: &Net, square: bool) -> Mlp {
        let acts: Vec<Box<dyn Activation>> = if square {
            vec![Box::new(Square), Box::new(Square)]
        } else {
            vec![Box::new(PolyRelu2), Box::new(Identity)]
        };
        Mlp::encrypt(
            &self.ev,
            initial,
            acts,
            &self.layout,
            &self.enc,
            &self.encoder,
            &mut rng(52),
        )
        .unwrap()
    }
    fn decoded(&self, mlp: &Mlp) -> Net {
        mlp.layers
            .iter()
            .map(|(d, _)| {
                (
                    d.w.decrypt(&self.dec, &self.encoder, &self.layout),
                    unpack_activations(
                        &self.encoder.decode(&self.dec.decrypt(&d.b)),
                        &self.layout,
                        d.w.d_out(),
                    )
                    .row(0)
                    .to_vec(),
                )
            })
            .collect()
    }
}
fn initial(frozen: bool) -> Net {
    DIMS.windows(2)
        .enumerate()
        .map(|(l, p)| {
            (
                Array2::from_shape_fn((p[1], p[0]), |(o, i)| 0.12 + 0.035 * (o + i + l) as f64),
                vec![if frozen { 0.0 } else { 0.04 }; p[1]],
            )
        })
        .collect()
}
fn data(step: usize, n: usize) -> (Array2<f64>, Array2<f64>) {
    (
        Array2::from_shape_fn((n, 2), |(r, c)| {
            if r < 8 {
                0.12 + 0.018 * (r + c + step) as f64
            } else {
                0.8 + 0.035 * (r + c + step) as f64
            }
        }),
        Array2::from_shape_fn((n, 2), |(r, c)| {
            if r < 8 {
                0.3 + 0.02 * (r + c) as f64
            } else {
                -0.35 - 0.01 * (r + c) as f64
            }
        }),
    )
}
fn forward(net: &Net, x: &Array2<f64>, square: bool) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let mut acts = vec![x.clone()];
    let mut zs = Vec::new();
    for (l, (w, b)) in net.iter().enumerate() {
        let mut z = acts.last().unwrap().dot(&w.t());
        for mut row in z.rows_mut() {
            for (v, b) in row.iter_mut().zip(b) {
                *v += b;
            }
        }
        acts.push(if square {
            z.mapv(|v| v * v)
        } else if l == 0 {
            z.mapv(|v| v * v + v)
        } else {
            z.clone()
        });
        zs.push(z);
    }
    (acts, zs)
}
// Independent f64 model: concatenate valid rows, differentiate once and update once.
fn reference(
    net: &mut Net,
    vel: &mut [Array2<f64>],
    x: &Array2<f64>,
    y: &Array2<f64>,
    (lr, mu): (f64, f64),
    square: bool,
    frozen: bool,
) {
    let (acts, zs) = forward(net, x, square);
    let mut delta = acts.last().unwrap() - y;
    for l in (0..net.len()).rev() {
        let dz = if square {
            &delta * &zs[l].mapv(|v| 2.0 * v)
        } else if l == 0 {
            &delta * &zs[l].mapv(|v| 2.0 * v + 1.0)
        } else {
            delta.clone()
        };
        let next = dz.dot(&net[l].0);
        let g = dz.t().dot(&acts[l]) / x.nrows() as f64;
        vel[l] = mu * &vel[l] + &g;
        net[l].0 = &net[l].0 - &(lr * (&g + mu * &vel[l]));
        if !frozen {
            for (b, gb) in net[l].1.iter_mut().zip(dz.sum_axis(Axis(0))) {
                *b -= lr * gb / x.nrows() as f64;
            }
        }
        delta = next;
    }
}
fn difference(a: &Net, b: &Net) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|((aw, ab), (bw, bb))| {
            aw.iter()
                .zip(bw)
                .chain(ab.iter().zip(bb))
                .map(|(x, y)| (x - y).abs())
        })
        .fold(0.0, f64::max)
}

struct Capture {
    inner: ClientRefresher,
    seen: Rc<RefCell<Vec<Ciphertext>>>,
}
impl Refresher for Capture {
    fn refresh(&mut self, ct: &Ciphertext) -> Result<Ciphertext> {
        let out = self.inner.refresh(ct)?;
        self.seen.borrow_mut().push(out.clone());
        Ok(out)
    }
    fn bind_layout(&mut self, layout: &Layout) -> Result<()> {
        self.inner.bind_layout(layout)
    }
    fn output_level(&self) -> usize {
        self.inner.output_level()
    }
}
fn stage_initial(n: usize) -> Vec<Net> {
    (0..n)
        .map(|stage| {
            let mut net = initial(true);
            for (w, _) in &mut net {
                *w *= 1.0 + 0.1 * stage as f64;
            }
            net
        })
        .collect()
}
fn model(c: &Ctx, stages: &[Net]) -> BlockMlp {
    let mut models: Vec<_> = stages.iter().map(|s| c.model(s, false)).collect();
    let output = models.pop().unwrap();
    BlockMlp::new(models, output).unwrap()
}
fn stage_trainers(c: &Ctx, n: usize, rate: f64) -> Vec<Trainer<'_>> {
    (0..n)
        .map(|_| {
            let opts: Vec<Box<dyn LayerOptimizer>> = (0..2)
                .map(|_| {
                    let mut opt = Nag::new(rate, 0.9);
                    opt.set_lr_reference(0.08).unwrap();
                    Box::new(opt) as Box<dyn LayerOptimizer>
                })
                .collect();
            Trainer::new(&c.ev, Box::new(c.client()), opts, c.layout)
                .unwrap()
                .with_frozen_biases()
        })
        .collect()
}
fn padded(a: &Array2<f64>, row: usize, n: usize, dirty: f64) -> Array2<f64> {
    let mut out = Array2::from_elem((8, a.ncols()), dirty);
    out.slice_mut(s![..n, ..])
        .assign(&a.slice(s![row..row + n, ..]));
    out
}
fn encode_parts(
    c: &Ctx,
    net: &BlockMlp,
    x: &Array2<f64>,
    ys: &[Array2<f64>],
    parts: &[usize],
) -> Vec<(Ciphertext, Vec<Ciphertext>, usize)> {
    let specs: Vec<_> = (0..net.num_stages())
        .map(|s| net.stage(s).batch_spec(&c.ev, &c.encoder).unwrap())
        .collect();
    let mut row = 0;
    let mut r = rng(54);
    parts
        .iter()
        .map(|&n| {
            let xm = padded(x, row, n, 2.5);
            let mut xc = None;
            let ycs = ys
                .iter()
                .zip(&specs)
                .map(|(y, spec)| {
                    let ym = padded(y, row, n, -3.0);
                    let (xct, yct) = spec
                        .encrypt(&c.enc, &c.encoder, &c.layout, &xm, &ym, &mut r)
                        .unwrap();
                    if xc.is_none() {
                        xc = Some(xct);
                    }
                    yct
                })
                .collect();
            row += n;
            (xc.unwrap(), ycs, n)
        })
        .collect()
}
#[derive(Debug, PartialEq)]
struct Receipt {
    trajectory: Vec<Vec<Net>>,
    counts: Vec<OpCounter>,
    traces: Vec<Vec<Vec<(String, usize)>>>,
    boundary_bits: Vec<Vec<u64>>,
    refreshes: Vec<usize>,
    handoffs: usize,
}
#[derive(Clone, Copy)]
enum Mode {
    Block,
    Singleton,
    Manual,
}

fn run(
    c: &Ctx,
    nstages: usize,
    rates: &[f64],
    parts: &[usize],
    mode: Mode,
    changed_target: Option<usize>,
) -> Receipt {
    let mut want = stage_initial(nstages);
    let init = want.clone();
    let mut vel: Vec<Vec<_>> = want
        .iter()
        .map(|st| st.iter().map(|(w, _)| Array2::zeros(w.dim())).collect())
        .collect();
    let mut net = model(c, &want);
    let seen = Rc::new(RefCell::new(Vec::new()));
    let capture = || Capture {
        inner: c.client(),
        seen: Rc::clone(&seen),
    };
    let mut block = BlockTrainer::new(
        &c.ev,
        stage_trainers(c, nstages, rates[0]),
        Box::new(capture()),
    )
    .unwrap();
    let mut manual = stage_trainers(c, nstages, rates[0]);
    let mut handoff = capture();
    handoff.bind_layout(&c.layout).unwrap();
    let mut receipt = Receipt {
        trajectory: vec![],
        counts: vec![],
        traces: vec![],
        boundary_bits: vec![],
        refreshes: vec![],
        handoffs: 0,
    };
    for (step, &rate) in rates.iter().enumerate() {
        block.set_lr(rate).unwrap();
        for stage in &mut manual {
            stage.set_lr(rate).unwrap();
        }
        let (x, y) = data(step, parts.iter().sum());
        let ys: Vec<_> = (0..nstages)
            .map(|s| &y + 0.04 * s as f64 + if changed_target == Some(s) { 0.8 } else { 0.0 })
            .collect();
        let mut old_inputs = vec![x.clone()];
        for st in &want[..nstages - 1] {
            old_inputs.push(forward(st, old_inputs.last().unwrap(), false).0[1].clone());
        }
        // Include dirty inactive lanes in the independent boundary oracle, although their
        // gradients must never contribute to the logical update.
        let mut row = 0;
        let mut boundary_inputs: Vec<_> = parts
            .iter()
            .map(|&n| {
                let xm = padded(&x, row, n, 2.5);
                row += n;
                xm
            })
            .collect();
        let mut expected_boundary = Vec::new();
        for st in &want[..nstages - 1] {
            boundary_inputs = boundary_inputs
                .iter()
                .map(|x| forward(st, x, false).0[1].clone())
                .collect();
            expected_boundary.extend(boundary_inputs.clone());
        }
        let encoded = encode_parts(c, &net, &x, &ys, parts);
        let targets: Vec<Vec<_>> = encoded
            .iter()
            .map(|(_, ys, _)| ys.iter().collect())
            .collect();
        let batches: Vec<_> = encoded
            .iter()
            .zip(&targets)
            .map(|((x, _, n), ys)| (x, &ys[..], *n))
            .collect();
        seen.borrow_mut().clear();
        let before = c.ev.counter();
        let mut expected_counts = OpCounter::default();
        match mode {
            Mode::Block => block.train_step_valid(&mut net, &batches).unwrap(),
            Mode::Singleton => {
                assert_eq!(batches.len(), 1);
                net.train_batch_valid(&mut block, batches[0].0, batches[0].1, batches[0].2)
                    .unwrap();
            }
            Mode::Manual => {
                let mut inputs: Vec<_> = batches.iter().map(|b| b.0.clone()).collect();
                for (s, trainer) in manual.iter_mut().enumerate() {
                    let mb: Vec<_> = inputs
                        .iter()
                        .zip(&batches)
                        .map(|(x, b)| (x, b.1[s], b.2))
                        .collect();
                    let before_stage = c.ev.counter();
                    // Legacy singleton control calls the preexisting API explicitly.
                    let hidden = if mb.len() == 1 {
                        vec![
                            trainer
                                .train_batch_valid_with_hidden(
                                    net.stage_mut(s),
                                    mb[0].0,
                                    mb[0].1,
                                    mb[0].2,
                                )
                                .unwrap(),
                        ]
                    } else {
                        trainer
                            .train_step_valid_with_hidden(net.stage_mut(s), &mb)
                            .unwrap()
                    };
                    expected_counts += &(&c.ev.counter() - &before_stage);
                    if s + 1 < nstages {
                        inputs = hidden
                            .iter()
                            .map(|h| {
                                let canonical =
                                    c.ev.rescale(
                                        &c.ev.mul_scalar(&h[0], 1.0, c.params.scale()).unwrap(),
                                    )
                                    .unwrap();
                                handoff.refresh(&canonical).unwrap()
                            })
                            .collect();
                    }
                }
                expected_counts.ct_pt_mults += ((nstages - 1) * parts.len()) as u64;
                expected_counts.rescales += ((nstages - 1) * parts.len()) as u64;
            }
        }
        let counts = &c.ev.counter() - &before;
        if matches!(mode, Mode::Manual) {
            assert_eq!(
                counts, expected_counts,
                "all counter fields = stage work + literal boundary cost"
            );
        }
        receipt.counts.push(counts);
        let seen_ref = seen.borrow();
        assert_eq!(seen_ref.len(), (nstages - 1) * parts.len());
        for (h, expected) in seen_ref.iter().zip(&expected_boundary) {
            assert_eq!(h.scale(), c.params.scale());
            assert_eq!(h.level(), c.params.max_level());
            let decoded = c.encoder.decode(&c.dec.decrypt(h));
            let packed = ckks_train_nn::layout::pack_activations(expected, &c.layout);
            let err = decoded
                .iter()
                .zip(&packed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err < TOL, "all-slot old-weight handoff error {err:e}");
            receipt
                .boundary_bits
                .push(decoded.iter().map(|v| v.to_bits()).collect());
        }
        for s in 0..nstages {
            reference(
                &mut want[s],
                &mut vel[s],
                &old_inputs[s],
                &ys[s],
                (rate, 0.9),
                false,
                true,
            );
        }
        let got: Vec<_> = (0..nstages).map(|s| c.decoded(net.stage(s))).collect();
        for s in 0..nstages {
            assert!(
                difference(&got[s], &want[s]) < TOL,
                "stage {s}, update {step}"
            );
            for (_, b) in &got[s] {
                assert!(b.iter().all(|v| v.abs() < TOL), "every stage bias frozen");
            }
        }
        let active = if matches!(mode, Mode::Manual) {
            &manual[..]
        } else {
            block.stages()
        };
        receipt
            .traces
            .push(active.iter().map(|t| t.trace().to_vec()).collect());
        receipt.refreshes = active.iter().map(|t| t.refreshes()).collect();
        let per_update: Vec<_> = (0..nstages)
            .map(|s| {
                net.stage(s)
                    .layers
                    .iter()
                    .map(|(d, _)| 2 * d.w.diagonals().entries().len() + 1)
                    .sum::<usize>()
            })
            .collect();
        assert_eq!(
            receipt.refreshes,
            per_update
                .iter()
                .map(|n| n * (step + 1))
                .collect::<Vec<_>>(),
            "one refresh pass per stage per logical update"
        );
        receipt.trajectory.push(got);
    }
    receipt.handoffs = if matches!(mode, Mode::Manual) {
        (nstages - 1) * parts.len() * rates.len()
    } else {
        block.handoffs()
    };
    assert_eq!(receipt.handoffs, (nstages - 1) * parts.len() * rates.len());
    let moved = want
        .iter()
        .zip(&init)
        .map(|(a, b)| difference(a, b))
        .fold(0.0, f64::max);
    let error = want
        .iter()
        .zip(receipt.trajectory.last().unwrap())
        .map(|(a, b)| difference(a, b))
        .fold(0.0, f64::max);
    assert!(
        moved > 1e-4 && moved > 16.0 * error,
        "nontrivial training signal {moved:e}, error {error:e}"
    );
    // Independent plaintext inference: local heads do not participate.
    let (x, _) = data(0, 3);
    let mut expected = x.clone();
    for st in &want[..nstages - 1] {
        expected = forward(st, &expected, false).0[1].clone();
    }
    expected = forward(&want[nstages - 1], &expected, false)
        .0
        .pop()
        .unwrap();
    let prediction = net.predict_plain(&c.dec, &c.encoder, &c.layout, &x);
    assert!(
        prediction
            .iter()
            .zip(&expected)
            .all(|(a, b)| (a - b).abs() < TOL)
    );
    receipt
}

#[test]
fn unequal_microbatches_scheduled_frozen_blocks_match_plaintext_and_stage_costs() {
    let c = Ctx::new();
    for stages in [2, 3] {
        let args = &[0.03, 0.08, 0.04, 0.01];
        let block = run(&c, stages, args, &[8, 4], Mode::Block, None);
        let manual = run(&c, stages, args, &[8, 4], Mode::Manual, None);
        assert_eq!(block, manual, "complete state/count/trace/boundary receipt");
    }
}
#[test]
fn one_stage_has_no_boundary_overhead_and_matches_the_stage_api() {
    let c = Ctx::new();
    let block = run(&c, 1, &[0.03, 0.08, 0.04, 0.01], &[8, 4], Mode::Block, None);
    assert_eq!(block.handoffs, 0);
    assert_eq!(
        block,
        run(
            &c,
            1,
            &[0.03, 0.08, 0.04, 0.01],
            &[8, 4],
            Mode::Manual,
            None
        )
    );
}
#[test]
fn singleton_wrapper_preserves_legacy_values_counts_traces_and_refreshes() {
    let c = Ctx::new();
    for n in [8, 4] {
        let block = run(&c, 3, &[0.08, 0.08], &[n], Mode::Block, None);
        assert_eq!(
            block,
            run(&c, 3, &[0.08, 0.08], &[n], Mode::Singleton, None)
        );
        assert_eq!(block, run(&c, 3, &[0.08, 0.08], &[n], Mode::Manual, None));
    }
}
#[test]
fn each_local_target_changes_only_its_stage_for_old_forward_inputs() {
    let c = Ctx::new();
    let base = run(&c, 3, &[0.03], &[8, 4], Mode::Block, None);
    for changed in 0..3 {
        let other = run(&c, 3, &[0.03], &[8, 4], Mode::Block, Some(changed));
        assert_eq!(
            base.boundary_bits, other.boundary_bits,
            "targets cannot change old hidden inputs"
        );
        for s in 0..3 {
            if s == changed {
                assert!(difference(&base.trajectory[0][s], &other.trajectory[0][s]) > TOL);
            } else {
                assert_eq!(
                    base.trajectory[0][s], other.trajectory[0][s],
                    "no cross-stage loss gradient"
                );
            }
        }
    }
}

// External observer proves no earlier commit; it is not a mutable public test hook.
struct ObservedSgd {
    value: Rc<RefCell<f64>>,
}
impl Optimizer for ObservedSgd {
    fn grad_scale(&self) -> f64 {
        *self.value.borrow()
    }
    fn step(
        &mut self,
        ev: &Evaluator,
        w: &mut EncWeights,
        dw: &Diagonals<Ciphertext>,
    ) -> Result<()> {
        Sgd {
            lr: *self.value.borrow(),
        }
        .step(ev, w, dw)
    }
}
impl LayerOptimizer for ObservedSgd {
    fn bias_grad_scale(&self) -> f64 {
        *self.value.borrow()
    }
    fn refresh_state(&mut self, _: &mut dyn Refresher) -> Result<usize> {
        Ok(0)
    }
    fn validate_lr(&self, lr: f64) -> Result<()> {
        if lr.is_finite() && lr > 0.0 {
            Ok(())
        } else {
            Err(HekitError::InvalidParams("rate".into()))
        }
    }
    fn apply_lr(&mut self, lr: f64) {
        *self.value.borrow_mut() = lr;
    }
}
struct Unsupported;
impl Optimizer for Unsupported {
    fn grad_scale(&self) -> f64 {
        0.03
    }
    fn step(&mut self, _: &Evaluator, _: &mut EncWeights, _: &Diagonals<Ciphertext>) -> Result<()> {
        unreachable!()
    }
}
impl LayerOptimizer for Unsupported {
    fn bias_grad_scale(&self) -> f64 {
        0.03
    }
    fn refresh_state(&mut self, _: &mut dyn Refresher) -> Result<usize> {
        Ok(0)
    }
}
#[test]
fn rejected_interactive_block_rates_leave_earlier_supported_layers_untouched() {
    let c = Ctx::new();
    for unsupported in [false, true] {
        let value = Rc::new(RefCell::new(0.03));
        let first: Vec<Box<dyn LayerOptimizer>> = vec![Box::new(ObservedSgd {
            value: Rc::clone(&value),
        })];
        let second: Vec<Box<dyn LayerOptimizer>> = vec![if unsupported {
            Box::new(Unsupported)
        } else {
            Box::new(Nag::new(0.03, 0.9))
        }];
        let stages = [first, second]
            .into_iter()
            .map(|opts| Trainer::new(&c.ev, Box::new(c.client()), opts, c.layout).unwrap())
            .collect();
        let mut block = BlockTrainer::new(&c.ev, stages, Box::new(c.client())).unwrap();
        let before = c.ev.counter();
        for rate in [0.04, 0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert!(block.set_lr(rate).is_err());
            assert_eq!(*value.borrow(), 0.03);
            assert_eq!(c.ev.counter(), before);
            assert_eq!(block.handoffs(), 0);
            assert!(
                block
                    .stages()
                    .iter()
                    .all(|s| s.refreshes() == 0 && s.trace().is_empty())
            );
        }
        if !unsupported {
            block.set_lr(0.02).unwrap();
            assert_eq!(*value.borrow(), 0.02);
        }
    }
}
#[test]
fn malformed_parts_reject_before_any_stage_work_and_leave_future_state_unchanged() {
    let c = Ctx::new();
    let init = stage_initial(3);
    let mut net = model(&c, &init);
    let mut block =
        BlockTrainer::new(&c.ev, stage_trainers(&c, 3, 0.03), Box::new(c.client())).unwrap();
    let (x, y) = data(0, 12);
    let encoded = encode_parts(&c, &net, &x, &[y.clone(), y.clone(), y], &[8, 4]);
    let yr: Vec<Vec<_>> = encoded.iter().map(|b| b.1.iter().collect()).collect();
    let first = (&encoded[0].0, &yr[0][..], 8);
    let before_net: Vec<_> = (0..3).map(|s| c.decoded(net.stage(s))).collect();
    let before = c.ev.counter();
    for bad in [
        vec![],
        vec![first, (&encoded[1].0, &yr[1][..2], 4)],
        vec![first, (&encoded[1].0, &yr[1][..], 0)],
        vec![first, (&encoded[1].0, &yr[1][..], 9)],
    ] {
        assert!(block.train_step_valid(&mut net, &bad).is_err());
        assert_eq!(c.ev.counter(), before);
        assert_eq!(block.handoffs(), 0);
        assert!(
            block
                .stages()
                .iter()
                .all(|s| s.refreshes() == 0 && s.trace().is_empty())
        );
        assert_eq!(
            before_net,
            (0..3).map(|s| c.decoded(net.stage(s))).collect::<Vec<_>>()
        );
    }
    let mut wrong_stage_count = model(&c, &init[..2]);
    assert!(
        block
            .train_step_valid(&mut wrong_stage_count, &[first])
            .is_err()
    );
    assert_eq!(c.ev.counter(), before);
    assert_eq!(block.handoffs(), 0);
    // A subsequent valid step remains exactly equal to a clean trainer, including NAG.
    let mut clean_net = model(&c, &init);
    let mut clean =
        BlockTrainer::new(&c.ev, stage_trainers(&c, 3, 0.03), Box::new(c.client())).unwrap();
    let good = [first, (&encoded[1].0, &yr[1][..], 4)];
    for lr in [0.03, 0.08] {
        block.set_lr(lr).unwrap();
        clean.set_lr(lr).unwrap();
        block.train_step_valid(&mut net, &good).unwrap();
        clean.train_step_valid(&mut clean_net, &good).unwrap();
        assert_eq!(
            (0..3).map(|s| c.decoded(net.stage(s))).collect::<Vec<_>>(),
            (0..3)
                .map(|s| c.decoded(clean_net.stage(s)))
                .collect::<Vec<_>>()
        );
    }
}
