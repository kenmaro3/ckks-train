//! Synthetic CPU correctness only; the small CKKS parameters are not scientific presets.
use std::sync::Arc;

use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::activation::{Activation, Identity, PolyRelu2, Square};
use ckks_train_nn::dense::TransposeKeys;
use ckks_train_nn::layout::{Layout, unpack_activations};
use ckks_train_nn::optim::{Nag, Sgd};
use ckks_train_nn::refresh::ClientRefresher;
use ckks_train_nn::trainer::{LayerOptimizer, Mlp, Trainer, mlp_rotation_steps};
use ndarray::{Array2, Axis, s};

const TOL: f64 = 1.0 / 1024.0;
const DIMS: [usize; 3] = [2, 4, 2];
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
fn encrypted(
    c: &Ctx,
    mlp: &Mlp,
    x: &Array2<f64>,
    y: &Array2<f64>,
    parts: &[usize],
) -> Vec<(Ciphertext, Ciphertext, usize)> {
    let spec = mlp.batch_spec(&c.ev, &c.encoder).unwrap();
    let mut row = 0;
    let mut r = rng(54);
    parts
        .iter()
        .map(|&n| {
            // Deliberately large nonzero inactive inputs AND targets; masks must exclude them.
            let mut xm = Array2::from_elem((8, 2), 2.5);
            let mut ym = Array2::from_elem((8, 2), -3.0);
            xm.slice_mut(s![..n, ..])
                .assign(&x.slice(s![row..row + n, ..]));
            ym.slice_mut(s![..n, ..])
                .assign(&y.slice(s![row..row + n, ..]));
            row += n;
            let (xc, yc) = spec
                .encrypt(&c.enc, &c.encoder, &c.layout, &xm, &ym, &mut r)
                .unwrap();
            (xc, yc, n)
        })
        .collect()
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
#[derive(Debug, PartialEq)]
struct Receipt {
    counts: Vec<OpCounter>,
    traces: Vec<Vec<(String, usize)>>,
    bits: Vec<Vec<u64>>,
    hidden: Vec<Vec<u64>>,
    refreshes: usize,
}

fn run(
    c: &Ctx,
    rates: &[f64],
    nag: bool,
    square: bool,
    frozen: bool,
    parts: &[usize],
    legacy: bool,
) -> Receipt {
    let mut want = initial(frozen);
    let mut vel: Vec<_> = want.iter().map(|(w, _)| Array2::zeros(w.dim())).collect();
    let mut mlp = c.model(&want, square);
    let opts: Vec<Box<dyn LayerOptimizer>> = (0..2)
        .map(|_| {
            if nag {
                let mut n = Nag::new(rates[0], 0.9);
                n.set_lr_reference(0.08).unwrap();
                Box::new(n) as Box<dyn LayerOptimizer>
            } else {
                Box::new(Sgd { lr: rates[0] }) as Box<dyn LayerOptimizer>
            }
        })
        .collect();
    let mut trainer = Trainer::new(&c.ev, Box::new(c.client()), opts, c.layout).unwrap();
    if frozen {
        trainer = trainer.with_frozen_biases();
    }
    let mut receipt = Receipt {
        counts: vec![],
        traces: vec![],
        bits: vec![],
        hidden: vec![],
        refreshes: 0,
    };
    for (step, &lr) in rates.iter().enumerate() {
        trainer.set_lr(lr).unwrap();
        let (x, y) = data(step, parts.iter().sum());
        let old_hidden = forward(&want, &x, square).0[1].clone();
        let batches = encrypted(c, &mlp, &x, &y, parts);
        let refs: Vec<_> = batches.iter().map(|(x, y, n)| (x, y, *n)).collect();
        let before = c.ev.counter();
        let hidden = if legacy {
            vec![
                trainer
                    .train_batch_valid_with_hidden(&mut mlp, refs[0].0, refs[0].1, refs[0].2)
                    .unwrap(),
            ]
        } else {
            trainer
                .train_step_valid_with_hidden(&mut mlp, &refs)
                .unwrap()
        };
        receipt.counts.push(&c.ev.counter() - &before);
        receipt.traces.push(trainer.trace().to_vec());
        assert_eq!(hidden.len(), parts.len());
        let mut offset = 0;
        for (h, &n) in hidden.iter().zip(parts) {
            assert_eq!(h.len(), 1);
            let values = c.encoder.decode(&c.dec.decrypt(&h[0]));
            let a = unpack_activations(&values, &c.layout, 4);
            for r in 0..n {
                for j in 0..4 {
                    assert!(
                        (a[[r, j]] - old_hidden[[offset + r, j]]).abs() < TOL,
                        "old-weight hidden micro offset {offset}"
                    );
                }
            }
            receipt
                .hidden
                .push(values.iter().map(|v| v.to_bits()).collect());
            offset += n;
        }
        reference(
            &mut want,
            &mut vel,
            &x,
            &y,
            (lr, if nag { 0.9 } else { 0.0 }),
            square,
            frozen,
        );
        let got = c.decoded(&mlp);
        let error = difference(&got, &want);
        assert!(error < TOL, "step {step} error {error:e}");
        for (d, _) in &mlp.layers {
            assert_eq!(d.w.level(), c.params.max_level());
            assert_eq!(d.b.level(), c.params.max_level());
            receipt.bits.extend(
                d.w.diagonals()
                    .entries()
                    .iter()
                    .chain(std::iter::once(&d.b))
                    .map(|ct| {
                        c.encoder
                            .decode(&c.dec.decrypt(ct))
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    }),
            );
        }
    }
    receipt.refreshes = trainer.refreshes();
    assert_eq!(receipt.refreshes, rates.len() * if nag { 10 } else { 6 });
    receipt
}

#[test]
fn unequal_dirty_micros_scheduled_nag_hidden_and_exact_counts() {
    let c = Ctx::new();
    let rates = [0.04, 0.00008, 0.02, 0.08];
    let scheduled = run(&c, &rates, true, false, true, &[8, 4], false);
    let constant = run(&c, &[0.08; 4], true, false, true, &[8, 4], false);
    let sgd = run(&c, &rates, false, false, true, &[8, 4], false);
    assert_eq!(scheduled.traces, constant.traces);
    for (i, &rate) in rates.iter().enumerate() {
        let mut expected = constant.counts[i].clone();
        if rate != 0.08 {
            expected.ct_pt_mults += 8;
            expected.rescales += 8;
            expected.rotations += 24;
        }
        assert_eq!(
            scheduled.counts[i], expected,
            "reference branch at step {i}"
        );
        let mut nag_expected = sgd.counts[i].clone();
        if i > 0 {
            nag_expected.ct_pt_mults += 8;
            nag_expected.rescales += 8;
        }
        assert_eq!(
            constant.counts[i], nag_expected,
            "exactly one NAG state update"
        );
    }
    let t = &scheduled.traces[0];
    let lv = |label: &str| t.iter().find(|(s, _)| s == label).unwrap().1;
    assert_eq!(
        lv("micro0/dz2"),
        lv("micro1/dz2") + 1,
        "partial Identity consumes one mask level"
    );
}
#[test]
fn square_sgd_unequal_and_one_valid_tail_match_one_logical_update() {
    let c = Ctx::new();
    run(&c, &[0.04, 0.2], false, true, false, &[8, 4], false);
    run(&c, &[0.04, 0.2], false, true, false, &[8, 1], false);
}
#[test]
fn singleton_is_seeded_bit_count_trace_and_refresh_identical() {
    let c = Ctx::new();
    for parts in [&[8][..], &[1][..]] {
        assert_eq!(
            run(&c, &[0.04, 0.08], true, false, true, parts, true),
            run(&c, &[0.04, 0.08], true, false, true, parts, false)
        );
    }
}
#[test]
fn malformed_micro_counts_reject_before_work_or_model_changes() {
    let c = Ctx::new();
    let mut mlp = c.model(&initial(false), false);
    let (x, y) = data(0, 8);
    let b = encrypted(&c, &mlp, &x, &y, &[8]);
    let mut trainer = Trainer::new(
        &c.ev,
        Box::new(c.client()),
        vec![Box::new(Sgd { lr: 0.04 }), Box::new(Sgd { lr: 0.04 })],
        c.layout,
    )
    .unwrap();
    let before = c.ev.counter();
    let model = c.decoded(&mlp);
    for counts in [vec![], vec![0], vec![9], vec![8, 0], vec![8, 9]] {
        let refs: Vec<_> = counts.into_iter().map(|n| (&b[0].0, &b[0].1, n)).collect();
        assert!(trainer.train_step_valid(&mut mlp, &refs).is_err());
        assert_eq!(c.ev.counter(), before);
        assert_eq!(c.decoded(&mlp), model);
        assert_eq!(trainer.refreshes(), 0);
        assert!(trainer.trace().is_empty());
    }
}
#[test]
fn fixture_distinguishes_sequential_steps_and_unweighted_micro_means() {
    let (x, y) = data(0, 12);
    let init = initial(true);
    let zero = || {
        init.iter()
            .map(|(w, _)| Array2::zeros(w.dim()))
            .collect::<Vec<_>>()
    };
    let mut correct = init.clone();
    reference(&mut correct, &mut zero(), &x, &y, (0.04, 0.9), false, true);
    let mut sequential = init.clone();
    let mut v = zero();
    for (start, end) in [(0, 8), (8, 12)] {
        reference(
            &mut sequential,
            &mut v,
            &x.slice(s![start..end, ..]).to_owned(),
            &y.slice(s![start..end, ..]).to_owned(),
            (0.04, 0.9),
            false,
            true,
        );
    }
    let mut means = Vec::new();
    for (start, end) in [(0, 8), (8, 12)] {
        let mut m = init.clone();
        reference(
            &mut m,
            &mut zero(),
            &x.slice(s![start..end, ..]).to_owned(),
            &y.slice(s![start..end, ..]).to_owned(),
            (0.04, 0.9),
            false,
            true,
        );
        means.push(m);
    }
    let equal: Net = means[0]
        .iter()
        .zip(&means[1])
        .map(|((w, b), (v, _))| ((w + v) * 0.5, b.clone()))
        .collect();
    assert!(difference(&correct, &sequential) > 10.0 * TOL);
    assert!(difference(&correct, &equal) > 10.0 * TOL);
}

#[test]
fn sgd_counts_are_sum_of_micro_passes_and_frozen_bias_has_no_count_effect() {
    let c = Ctx::new();
    let multi = run(&c, &[0.04, 0.02], false, false, false, &[8, 4], false);
    let full = run(&c, &[0.04, 0.02], false, false, false, &[8], true);
    let partial = run(&c, &[0.04, 0.02], false, false, false, &[4], true);
    let frozen = run(&c, &[0.04, 0.02], false, false, true, &[8, 4], false);
    assert_eq!(multi.counts, frozen.counts);
    assert_eq!(multi.traces, frozen.traces);
    for i in 0..2 {
        let mut sum = full.counts[i].clone();
        sum += &partial.counts[i];
        assert_eq!(
            multi.counts[i], sum,
            "SGD updates and accumulation add no fields in OpCounter"
        );
    }
    // Accumulator already partial/low when the next full/high contribution arrives.
    run(&c, &[0.04, 0.00008], true, false, true, &[4, 8, 1], false);
}

#[test]
fn late_collection_failure_keeps_model_and_velocity_untouched() {
    let c = Ctx::new();
    let init = initial(true);
    let mut model = c.model(&init, false);
    let mut untouched = c.model(&init, false);
    let make_opts = || {
        (0..2)
            .map(|_| {
                let mut opt = Nag::new(0.04, 0.9);
                opt.set_lr_reference(0.08).unwrap();
                Box::new(opt) as Box<dyn LayerOptimizer>
            })
            .collect()
    };
    let mut trainer = Trainer::new(&c.ev, Box::new(c.client()), make_opts(), c.layout)
        .unwrap()
        .with_frozen_biases();
    let mut fresh = Trainer::new(&c.ev, Box::new(c.client()), make_opts(), c.layout)
        .unwrap()
        .with_frozen_biases();
    let (x, y) = data(0, 12);
    let bs = encrypted(&c, &model, &x, &y, &[8, 4]);
    let spec = model.batch_spec(&c.ev, &c.encoder).unwrap();
    let bad_pt = c
        .encoder
        .encode(
            &vec![0.0; c.params.slots()],
            spec.y_scale * 2.0,
            spec.y_level,
        )
        .unwrap();
    let bad = c.enc.encrypt(&bad_pt, &mut rng(58));
    let before = c.ev.counter();
    assert!(
        trainer
            .train_step_valid(&mut model, &[(&bs[0].0, &bs[0].1, 8), (&bs[1].0, &bad, 4)])
            .is_err()
    );
    assert_eq!(c.decoded(&model), c.decoded(&untouched));
    assert_eq!(trainer.refreshes(), 0);
    assert!(trainer.trace().is_empty());
    assert_ne!(
        c.ev.counter(),
        before,
        "completed primitives remain counted"
    );
    let refs: Vec<_> = bs.iter().map(|(x, y, n)| (x, y, *n)).collect();
    let before = c.ev.counter();
    trainer.train_step_valid(&mut model, &refs).unwrap();
    let actual = &c.ev.counter() - &before;
    let before = c.ev.counter();
    let hidden = fresh
        .train_step_valid_with_hidden(&mut untouched, &refs)
        .unwrap();
    assert_eq!(hidden.len(), 2);
    assert_eq!(
        actual,
        &c.ev.counter() - &before,
        "failure did not initialize NAG velocity"
    );
    assert_eq!(
        c.decoded(&model),
        c.decoded(&untouched),
        "no-hidden and hidden paths have identical state"
    );
    assert_eq!(trainer.trace(), fresh.trace());
    assert_eq!(trainer.refreshes(), fresh.refreshes());
}

#[test]
fn invalid_later_optimizer_preflights_before_any_micro_work() {
    let c = Ctx::new();
    let mut model = c.model(&initial(false), false);
    let (x, y) = data(0, 12);
    let bs = encrypted(&c, &model, &x, &y, &[8, 4]);
    let mut invalid = Nag::new(0.08, 0.9);
    invalid.lr = 0.1;
    let mut trainer = Trainer::new(
        &c.ev,
        Box::new(c.client()),
        vec![Box::new(Sgd { lr: 0.04 }), Box::new(invalid)],
        c.layout,
    )
    .unwrap();
    let before = c.ev.counter();
    let old = c.decoded(&model);
    let refs: Vec<_> = bs.iter().map(|(x, y, n)| (x, y, *n)).collect();
    assert!(trainer.train_step_valid(&mut model, &refs).is_err());
    assert_eq!(before, c.ev.counter());
    assert_eq!(old, c.decoded(&model));
    assert_eq!(trainer.refreshes(), 0);
}

#[test]
fn hidden_receipt_orders_multiple_layers_and_micros_from_old_weights() {
    // Three partial Identity masks need more levels than the two-layer fixture.
    let c = Ctx::with_levels(12);
    // Existing key set also covers the repeated 2 -> 4 -> 2 -> 4 geometry.
    let net: Net = [2, 4, 2, 4]
        .windows(2)
        .enumerate()
        .map(|(l, p)| {
            (
                Array2::from_elem((p[1], p[0]), 0.1 + 0.05 * l as f64),
                vec![0.02; p[1]],
            )
        })
        .collect();
    let mut mlp = Mlp::encrypt(
        &c.ev,
        &net,
        vec![Box::new(Identity), Box::new(Identity), Box::new(Identity)],
        &c.layout,
        &c.enc,
        &c.encoder,
        &mut rng(59),
    )
    .unwrap();
    let spec = mlp.batch_spec(&c.ev, &c.encoder).unwrap();
    let x1 = Array2::from_elem((8, 2), 0.3);
    let x2 = Array2::from_elem((8, 2), 0.8);
    let y = Array2::from_elem((8, 4), -0.4);
    let (a, b) = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x1, &y, &mut rng(60))
        .unwrap();
    let (d, e) = spec
        .encrypt(&c.enc, &c.encoder, &c.layout, &x2, &y, &mut rng(61))
        .unwrap();
    let mut trainer = Trainer::new(
        &c.ev,
        Box::new(c.client()),
        (0..3)
            .map(|_| Box::new(Sgd { lr: 0.08 }) as Box<dyn LayerOptimizer>)
            .collect(),
        c.layout,
    )
    .unwrap();
    let hidden = trainer
        .train_step_valid_with_hidden(&mut mlp, &[(&a, &b, 8), (&d, &e, 4)])
        .unwrap();
    assert_eq!(hidden.len(), 2);
    for (micro, x) in [&x1, &x2].iter().enumerate() {
        assert_eq!(hidden[micro].len(), 2);
        let mut want = (*x).clone();
        for (layer, (w, b)) in net.iter().take(2).enumerate() {
            want = want.dot(&w.t());
            for mut row in want.rows_mut() {
                for (v, b) in row.iter_mut().zip(b) {
                    *v += b;
                }
            }
            let h = &hidden[micro][layer];
            assert_eq!(h.level(), c.params.max_level() - layer - 1);
            let got =
                unpack_activations(&c.encoder.decode(&c.dec.decrypt(h)), &c.layout, w.nrows());
            assert!(got.iter().zip(&want).all(|(g, w)| (g - w).abs() < TOL));
        }
    }
}
