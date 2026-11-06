//! Insecure small CPU full-stage learning oracles; no bootstrap/NI/security evidence.
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, HeOps, KeyGenerator,
    OpCounter,
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::{
    layout_reboot::GradRelin,
    packed_training::{
        PackedGeometryPlan, PackedRecord, PackedStage, PackedStageConfig, PackedVector,
        PackedWeights, StageShape, VectorSpec,
    },
};
use ndarray::Array2;
use std::sync::Arc;
struct Ctx {
    p: Arc<CkksParams>,
    e: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
    plan: PackedGeometryPlan,
}
impl Ctx {
    fn new(square: bool) -> Self {
        let mut bits = vec![40; 32];
        bits[0] = 60;
        let p = Arc::new(CkksParams::new(8, &bits, &[60], 40, 32).unwrap());
        let out = if square { 3 } else { 4 };
        let plan = PackedGeometryPlan::new(
            5,
            &[
                StageShape::new(3, out, 3).unwrap(),
                StageShape::new(5, out, 3).unwrap(),
            ],
            p.slots(),
        )
        .unwrap();
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([91; 32])).unwrap();
        Self {
            e: Encoder::new(p.clone()),
            enc: Encryptor::new(p.clone(), kg.public_key()),
            dec: Decryptor::new(p.clone(), kg.secret_key()),
            ev: Evaluator::new(
                p.clone(),
                Some(kg.relin_key()),
                Some(kg.galois_keys(&plan.layout().galois_steps())),
            ),
            p,
            plan,
        }
    }
    fn ct(&self, v: &[f64], level: usize) -> Ciphertext {
        self.enc.encrypt(
            &self.e.encode(v, self.p.scale(), level).unwrap(),
            &mut Rng::from_seed([92; 32]),
        )
    }
    fn vec(&self, s: VectorSpec, v: &[f64], l: usize) -> PackedVector<Ciphertext> {
        PackedVector::new(&self.ev, s, self.ct(&s.pack(v).unwrap(), l)).unwrap()
    }
    fn vals(&self, c: &Ciphertext) -> Vec<f64> {
        self.e.decode(&self.dec.decrypt(c))
    }
    fn stage(
        &self,
        s: usize,
        square: bool,
    ) -> (PackedStage<Ciphertext>, [Array2<f64>; 2], [Vec<f64>; 2]) {
        let ls = &self.plan.layers()[2 * s..2 * s + 2];
        let w = std::array::from_fn(|k| {
            Array2::from_shape_fn(
                (ls[k].output().logical_len(), ls[k].input().logical_len()),
                |(o, i)| 0.03 + 0.01 * (o + i) as f64,
            )
        });
        let b = std::array::from_fn(|k| {
            vec![if square { 0.04 } else { 0.0 }; ls[k].output().logical_len()]
        });
        let ws = std::array::from_fn(|k| {
            PackedWeights::new(
                &self.ev,
                ls[k].clone(),
                self.ct(&ls[k].pack_weights_out_by_in(&w[k]).unwrap(), 31),
            )
            .unwrap()
        });
        let bs = square.then(|| std::array::from_fn(|k| self.vec(ls[k].output(), &b[k], 31)));
        let config = if square {
            PackedStageConfig::square_sgd()
        } else {
            PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
        };
        (
            PackedStage::new(&self.ev, &self.plan, s, ws, bs, config).unwrap(),
            w,
            b,
        )
    }
    fn records(&self, s: usize, n: usize, target_shift: f64) -> Vec<PackedRecord<Ciphertext>> {
        let ls = &self.plan.layers()[2 * s..2 * s + 2];
        (0..n)
            .map(|j| {
                PackedRecord::new(
                    self.vec(
                        ls[0].input(),
                        &(0..ls[0].input().logical_len())
                            .map(|i| 0.1 + 0.02 * (i + j) as f64)
                            .collect::<Vec<_>>(),
                        31,
                    ),
                    self.vec(
                        ls[1].output(),
                        &(0..ls[1].output().logical_len())
                            .map(|o| if o == j % 3 { 0.25 + target_shift } else { 0.0 })
                            .collect::<Vec<_>>(),
                        31,
                    ),
                    true,
                )
            })
            .collect()
    }
}
fn near(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 1e-7, "slot {i}: {a} != {b}");
    }
}
// Explicit output-by-input matrix loops and unscaled V; never calls encrypted helper math.
fn reference(
    w: &mut [Array2<f64>; 2],
    b: &mut [Vec<f64>; 2],
    v: &mut [Array2<f64>; 2],
    n: usize,
    lr: f64,
    square: bool,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut g = w.clone();
    for x in &mut g {
        x.fill(0.0)
    }
    let mut gb = b.clone();
    for x in &mut gb {
        x.fill(0.0)
    }
    let mut hs = vec![];
    let mut ys = vec![];
    for j in 0..n {
        let x: Vec<_> = (0..w[0].ncols())
            .map(|i| 0.1 + 0.02 * (i + j) as f64)
            .collect();
        let z: Vec<_> = (0..w[0].nrows())
            .map(|o| b[0][o] + (0..x.len()).map(|i| w[0][[o, i]] * x[i]).sum::<f64>())
            .collect();
        let h: Vec<_> = z
            .iter()
            .map(|z| if square { z * z } else { z * z + z })
            .collect();
        let z2: Vec<_> = (0..w[1].nrows())
            .map(|o| b[1][o] + (0..h.len()).map(|i| w[1][[o, i]] * h[i]).sum::<f64>())
            .collect();
        let y: Vec<_> = z2.iter().map(|z| if square { z * z } else { *z }).collect();
        let dz2: Vec<_> = y
            .iter()
            .enumerate()
            .map(|(o, y)| {
                (y - if o == j % 3 { 0.25 } else { 0.0 })
                    * if square { 2.0 / 3.0 * 2.0 * z2[o] } else { 1.0 }
            })
            .collect();
        let dz1: Vec<_> = (0..h.len())
            .map(|i| {
                (0..y.len()).map(|o| w[1][[o, i]] * dz2[o]).sum::<f64>()
                    * if square { 2.0 * z[i] } else { 2.0 * z[i] + 1.0 }
            })
            .collect();
        for k in 0..2 {
            let (a, d) = if k == 0 { (&x, &dz1) } else { (&h, &dz2) };
            for (o, &dz) in d.iter().enumerate() {
                gb[k][o] += dz / n as f64;
                for (i, &input) in a.iter().enumerate() {
                    g[k][[o, i]] += input * dz / n as f64;
                }
            }
        }
        hs.push(h);
        ys.push(y);
    }
    for k in 0..2 {
        for o in 0..w[k].nrows() {
            for i in 0..w[k].ncols() {
                let grad = g[k][[o, i]];
                if square {
                    w[k][[o, i]] -= lr * grad;
                } else {
                    v[k][[o, i]] = 0.9 * v[k][[o, i]] + grad;
                    w[k][[o, i]] -= lr * (grad + 0.9 * v[k][[o, i]]);
                }
            }
            if square {
                b[k][o] -= lr * gb[k][o];
            }
        }
    }
    (hs, ys)
}
#[test]
fn whole_stage_learning_old_snapshots_and_exact_counts_in_both_directions() {
    for square in [false, true] {
        let c = Ctx::new(square);
        for s in 0..2 {
            let (mut stage, mut w, mut b) = c.stage(s, square);
            let mut v = std::array::from_fn(|k| Array2::zeros(w[k].dim()));
            for (t, lr) in if square { vec![0.04] } else { vec![0.04, 0.08] }
                .into_iter()
                .enumerate()
            {
                let n = if t == 0 { 3 } else { 1 };
                let records = c.records(s, n, 0.0);
                c.ev.reset_counter();
                let plan = stage
                    .plan_step(&c.ev, &c.e, &records, lr, GradRelin::Lazy)
                    .unwrap();
                assert_eq!(c.ev.counter(), OpCounter::default());
                let old = c.vals(stage.layers()[0].weights().ciphertext());
                let (candidate, receipt) = stage
                    .prepare_step(&c.ev, &c.e, &records, lr, GradRelin::Lazy)
                    .unwrap();
                near(&c.vals(stage.layers()[0].weights().ciphertext()), &old);
                assert_eq!(stage.layers()[0].update_index(), t as u64);
                assert_eq!(receipt.actual_counts, plan.predicted_counts);
                assert_eq!(receipt.trace, plan.trace);
                let (hs, ys) = reference(&mut w, &mut b, &mut v, n, lr, square);
                let ls = &c.plan.layers()[2 * s..2 * s + 2];
                for j in 0..n {
                    near(
                        &c.vals(receipt.hidden[j].ciphertext()),
                        &ls[0].output().pack(&hs[j]).unwrap(),
                    );
                    near(
                        &c.vals(receipt.output[j].ciphertext()),
                        &ls[1].output().pack(&ys[j]).unwrap(),
                    );
                }
                let a = c.plan.layout().r().ilog2() as u64;
                let cc = 2 * c.plan.layout().c().ilog2() as u64;
                let branches = if square || lr == 0.08 { 2 } else { 4 };
                let extra =
                    branches + if square { 2 } else { 0 } + if t > 0 && !square { 4 } else { 0 };
                assert_eq!(
                    receipt.actual_counts,
                    OpCounter {
                        rotations: n as u64 * if s == 0 { 2 * a + cc } else { a + 2 * cc },
                        ct_ct_tensor: n as u64 * if square { 9 } else { 7 },
                        relins: n as u64 * if square { 7 } else { 5 } + 2,
                        ct_pt_mults: n as u64 * if square { 8 } else { 7 } + extra,
                        rescales: n as u64 * if square { 15 } else { 12 } + 2 + extra,
                        ..Default::default()
                    }
                );
                stage = candidate.into_stage();
                for k in 0..2 {
                    near(
                        &c.vals(stage.layers()[k].weights().ciphertext()),
                        &ls[k].pack_weights_out_by_in(&w[k]).unwrap(),
                    );
                    if square {
                        near(
                            &c.vals(stage.biases().unwrap()[k].ciphertext()),
                            &ls[k].output().pack(&b[k]).unwrap(),
                        );
                    } else {
                        near(
                            &c.vals(stage.layers()[k].velocity().unwrap().ciphertext()),
                            &ls[k]
                                .pack_weights_out_by_in(&v[k].mapv(|v| v * 0.08 * 1.9))
                                .unwrap(),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn changed_encrypted_target_changes_both_updates_but_never_old_forward() {
    let c = Ctx::new(false);
    let (stage, _, _) = c.stage(0, false);
    let (a, ra) = stage
        .prepare_step(&c.ev, &c.e, &c.records(0, 1, 0.0), 0.04, GradRelin::Lazy)
        .unwrap();
    let (b, rb) = stage
        .prepare_step(&c.ev, &c.e, &c.records(0, 1, 0.4), 0.04, GradRelin::Lazy)
        .unwrap();
    near(
        &c.vals(ra.hidden[0].ciphertext()),
        &c.vals(rb.hidden[0].ciphertext()),
    );
    near(
        &c.vals(ra.output[0].ciphertext()),
        &c.vals(rb.output[0].ciphertext()),
    );
    let a = a.into_stage();
    let b = b.into_stage();
    for k in 0..2 {
        assert!(
            c.vals(a.layers()[k].weights().ciphertext())
                .iter()
                .zip(c.vals(b.layers()[k].weights().ciphertext()))
                .any(|(x, y)| (x - y).abs() > 1e-5)
        );
    }
    // Logical padded class 3 (real classes=3) trains against zero; it is not structural padding.
    let l = &c.plan.layers()[1];
    let initial = 0.03 + 0.01 * 3.0;
    let trained = c.vals(a.layers()[1].weights().ciphertext());
    let slot = 3 * l.layout().c();
    assert!((trained[slot] - initial).abs() > 1e-6);
}

#[test]
fn unequal_partition_and_invalid_records_keep_one_actual_mean_and_policy_counts() {
    let c = Ctx::new(false);
    let (stage, _, _) = c.stage(1, false);
    let flat = c.records(1, 3, 0.0);
    let mut grouped = flat[..2].to_vec();
    grouped.push(PackedRecord::new(
        c.vec(c.plan.layers()[2].input(), &[0.1; 3], 31),
        c.vec(c.plan.layers()[3].output(), &[0.0; 4], 31),
        false,
    ));
    grouped.extend_from_slice(&flat[2..]);
    let (a, ra) = stage
        .prepare_step(&c.ev, &c.e, &flat, 0.08, GradRelin::Lazy)
        .unwrap();
    let (b, rb) = stage
        .prepare_step(&c.ev, &c.e, &grouped, 0.08, GradRelin::PerSample)
        .unwrap();
    let a = a.into_stage();
    let b = b.into_stage();
    assert_eq!(rb.plan.valid_samples, 3);
    for k in 0..2 {
        near(
            &c.vals(a.layers()[k].weights().ciphertext()),
            &c.vals(b.layers()[k].weights().ciphertext()),
        );
    }
    let mut expected = ra.actual_counts;
    expected.relins += 4;
    assert_eq!(rb.actual_counts, expected);
}

#[test]
fn malformed_stage_and_entire_dag_reject_before_counted_work() {
    let c = Ctx::new(false);
    let (stage, _, _) = c.stage(0, false);
    let ls = c.plan.layers();
    let records = c.records(0, 1, 0.0);
    let x = c.vec(ls[0].input(), &[0.1; 5], 31);
    let y = c.vec(ls[1].output(), &[0.0; 4], 31);
    let old = c.vals(stage.layers()[0].weights().ciphertext());
    let wrong_scale = PackedVector::new(
        &c.ev,
        ls[1].output(),
        c.ev.relabel(y.ciphertext().clone(), c.p.scale() * 2.0)
            .unwrap(),
    )
    .unwrap();
    let wrong_repr = c.vec(ls[0].output(), &[0.0; 3], 31);
    let wrong_width = c.vec(
        VectorSpec::new(ls[1].layout(), ls[1].output().repr(), 3).unwrap(),
        &[0.0; 3],
        31,
    );
    let low = c.vec(ls[0].input(), &[0.1; 5], 11);
    let other = Ctx::new(false);
    let foreign = other.vec(ls[1].output(), &[0.0; 4], 31);
    let cases = [
        vec![],
        vec![PackedRecord::new(x.clone(), y.clone(), false)],
        vec![PackedRecord::new(x.clone(), wrong_scale, true)],
        vec![PackedRecord::new(x.clone(), wrong_repr, true)],
        vec![PackedRecord::new(x.clone(), wrong_width, true)],
        vec![PackedRecord::new(low, y, true)],
        vec![PackedRecord::new(x, foreign, true)],
    ];
    c.ev.reset_counter();
    for bad in cases {
        assert!(
            stage
                .prepare_step(&c.ev, &c.e, &bad, 0.04, GradRelin::Lazy)
                .is_err()
        );
        assert_eq!(c.ev.counter(), OpCounter::default());
    }
    for rate in [0.0, -0.1, f64::NAN, f64::INFINITY, 0.09, 1e-100] {
        assert!(
            stage
                .prepare_step(&c.ev, &c.e, &records, rate, GradRelin::Lazy)
                .is_err(),
            "rate {rate}"
        );
        assert_eq!(c.ev.counter(), OpCounter::default());
    }
    let no_keys = Evaluator::new(c.p.clone(), None, None);
    assert!(
        stage
            .prepare_step(&no_keys, &c.e, &records, 0.04, GradRelin::Lazy)
            .is_err()
    );
    assert_eq!(no_keys.counter(), OpCounter::default());
    assert!(
        stage
            .prepare_step(&c.ev, &other.e, &records, 0.04, GradRelin::Lazy)
            .is_err()
    );
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(stage.layers()[0].update_index(), 0);
    near(&c.vals(stage.layers()[0].weights().ciphertext()), &old);
}

#[test]
fn independent_first_step_trace_proves_canonical_twelve_and_fifteen_level_dags() {
    use ckks_train_nn::packed_training::PackedTraceEntry;
    for square in [false, true] {
        let c = Ctx::new(square);
        let (stage, _, _) = c.stage(0, square);
        let p = stage
            .plan_step(&c.ev, &c.e, &c.records(0, 1, 0.0), 0.04, GradRelin::Lazy)
            .unwrap();
        let d = c.p.scale();
        let q = |l: usize| c.p.q()[l] as f64;
        let mut tuples = vec![
            ("affine1.product", 30, d * d / q(31)),
            ("affine1", 29, d),
            ("z1", 29, d),
            ("hidden.natural", 28, d * d / q(29)),
            ("hidden", 27, d),
            ("affine2.product", 26, d * d / q(27)),
            ("affine2", 25, d),
            ("z2", 25, d),
        ];
        if square {
            tuples.extend([
                ("output.natural", 24, d * d / q(25)),
                ("output", 23, d),
                ("residual", 23, d),
                ("loss", 22, d),
                ("derivative2", 24, d),
                ("dz2", 21, d * d / q(22)),
                ("upstream.product", 20, (d * d / q(22)) * d / q(21)),
                ("upstream", 19, d),
                ("derivative1", 28, d),
                ("dz1", 18, d * d / q(19)),
            ]);
        } else {
            tuples.extend([
                ("output", 25, d),
                ("residual", 25, d),
                ("loss", 24, d),
                ("dz2", 24, d),
                ("upstream.product", 23, d * d / q(24)),
                ("upstream", 22, d),
                ("derivative1", 28, d),
                ("dz1", 21, d * d / q(22)),
            ]);
        }
        let mut expected: Vec<_> = tuples
            .into_iter()
            .map(|(name, level, scale)| PackedTraceEntry {
                sample: Some(0),
                name: name.into(),
                level,
                scale,
            })
            .collect();
        let shared = if square {
            vec![
                ("outer0", 17, d * (d * d / q(19)) / q(18)),
                ("gradient0.0", 16, d),
                ("weight0", 16, d),
                ("bias_gradient0", 17, d),
                ("bias0", 17, d),
                ("outer1", 20, d * (d * d / q(22)) / q(21)),
                ("gradient1.0", 19, d),
                ("weight1", 19, d),
                ("bias_gradient1", 20, d),
                ("bias1", 20, d),
            ]
        } else {
            vec![
                ("outer0", 20, d * (d * d / q(22)) / q(21)),
                ("gradient0.0", 19, d),
                ("gradient0.1", 19, d),
                ("weight0", 19, d),
                ("velocity0", 19, d),
                ("outer1", 23, d * d / q(24)),
                ("gradient1.0", 22, d),
                ("gradient1.1", 22, d),
                ("weight1", 22, d),
                ("velocity1", 22, d),
            ]
        };
        expected.extend(
            shared
                .into_iter()
                .map(|(name, level, scale)| PackedTraceEntry {
                    sample: None,
                    name: name.into(),
                    level,
                    scale,
                }),
        );
        assert_eq!(p.trace, expected);
        assert_eq!(p.minimum_state_level, 31 - if square { 15 } else { 12 });
        assert_eq!(c.ev.counter(), OpCounter::default());
    }
}

#[test]
fn bias_domain_and_geometry_are_checked_before_stage_creation() {
    let c = Ctx::new(true);
    let (stage, _, _) = c.stage(0, true);
    let ls = c.plan.layers();
    let ws = || std::array::from_fn(|k| stage.layers()[k].weights().clone());
    let bs = stage.biases().unwrap().clone();
    let bad_scale = PackedVector::new(
        &c.ev,
        ls[0].output(),
        c.ev.relabel(bs[0].ciphertext().clone(), c.p.scale() * 2.0)
            .unwrap(),
    )
    .unwrap();
    c.ev.reset_counter();
    assert!(
        PackedStage::new(
            &c.ev,
            &c.plan,
            0,
            ws(),
            None,
            PackedStageConfig::square_sgd()
        )
        .is_err()
    );
    assert!(
        PackedStage::new(
            &c.ev,
            &c.plan,
            0,
            ws(),
            Some(bs.clone()),
            PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
        )
        .is_err()
    );
    assert!(
        PackedStage::new(
            &c.ev,
            &c.plan,
            1,
            ws(),
            Some(bs.clone()),
            PackedStageConfig::square_sgd()
        )
        .is_err()
    );
    assert!(
        PackedStage::new(
            &c.ev,
            &c.plan,
            0,
            ws(),
            Some([bs[1].clone(), bs[0].clone()]),
            PackedStageConfig::square_sgd()
        )
        .is_err()
    );
    assert!(
        PackedStage::new(
            &c.ev,
            &c.plan,
            0,
            ws(),
            Some([bad_scale, bs[1].clone()]),
            PackedStageConfig::square_sgd()
        )
        .is_err()
    );
    assert_eq!(c.ev.counter(), OpCounter::default());
    let mut kg = KeyGenerator::new(c.p.clone(), Rng::from_seed([93; 32])).unwrap();
    let no_rot = Evaluator::new(c.p.clone(), Some(kg.relin_key()), None);
    assert!(
        stage
            .prepare_step(&no_rot, &c.e, &c.records(0, 1, 0.0), 0.04, GradRelin::Lazy)
            .is_err()
    );
    assert_eq!(no_rot.counter(), OpCounter::default());
}

#[test]
fn mixed_input_target_and_bias_levels_are_planned_using_minima_without_value_changes() {
    let c = Ctx::new(true);
    let (stage, mut w, mut b) = c.stage(0, true);
    let ls = c.plan.layers();
    let ws = std::array::from_fn(|k| stage.layers()[k].weights().clone());
    let bs = std::array::from_fn(|k| {
        PackedVector::new(
            &c.ev,
            ls[k].output(),
            c.ev.mod_switch_to_level(
                stage.biases().unwrap()[k].ciphertext(),
                if k == 0 { 28 } else { 23 },
            )
            .unwrap(),
        )
        .unwrap()
    });
    let stage = PackedStage::new(
        &c.ev,
        &c.plan,
        0,
        ws,
        Some(bs),
        PackedStageConfig::square_sgd(),
    )
    .unwrap();
    let records: Vec<_> = (0..3)
        .map(|j| {
            PackedRecord::new(
                c.vec(
                    ls[0].input(),
                    &(0..5)
                        .map(|i| 0.1 + 0.02 * (i + j) as f64)
                        .collect::<Vec<_>>(),
                    31 - j,
                ),
                c.vec(
                    ls[1].output(),
                    &(0..3)
                        .map(|o| if o == j { 0.25 } else { 0.0 })
                        .collect::<Vec<_>>(),
                    25 - j,
                ),
                true,
            )
        })
        .collect();
    let (next, receipt) = stage
        .prepare_step(&c.ev, &c.e, &records, 0.04, GradRelin::Lazy)
        .unwrap();
    let mut v = std::array::from_fn(|k| Array2::zeros(w[k].dim()));
    reference(&mut w, &mut b, &mut v, 3, 0.04, true);
    let next = next.into_stage();
    for k in 0..2 {
        near(
            &c.vals(next.layers()[k].weights().ciphertext()),
            &ls[k].pack_weights_out_by_in(&w[k]).unwrap(),
        );
        near(
            &c.vals(next.biases().unwrap()[k].ciphertext()),
            &ls[k].output().pack(&b[k]).unwrap(),
        );
    }
    let find = |name: &str| {
        receipt
            .trace
            .iter()
            .find(|t| t.sample == Some(0) && t.name == name)
            .unwrap()
            .level
    };
    assert_eq!(
        (
            find("z1"),
            find("hidden"),
            find("z2"),
            find("output"),
            find("loss"),
            find("dz1")
        ),
        (27, 25, 23, 21, 20, 16)
    );
    assert_eq!(receipt.plan.minimum_state_level, 14);
}

// A backend error after layer0's candidate exists verifies whole-stage immutability.
// This wrapper delegates real CPU arithmetic and injects only one sub failure.
struct FailSub {
    ev: Evaluator,
    at: usize,
    subs: std::cell::Cell<usize>,
}
macro_rules! delegate {
 ($name:ident($($a:ident:$t:ty),*)->$r:ty) => {fn $name(&self,$($a:$t),*)->$r {HeOps::$name(&self.ev,$($a),*)}};
}
impl HeOps for FailSub {
    type Ct = Ciphertext;
    type Pt = ckks_train_ckks::Plaintext;
    delegate!(params()->&Arc<CkksParams>);
    delegate!(config()->&ckks_train_ckks::EvalConfig);
    delegate!(counter()->OpCounter);
    delegate!(reset_counter()->());
    delegate!(add_counts(counts:&OpCounter)->());
    fn fork(&self) -> Self {
        Self {
            ev: HeOps::fork(&self.ev),
            at: self.at,
            subs: std::cell::Cell::new(0),
        }
    }
    delegate!(has_relin_key()->bool);
    delegate!(has_galois_key(step:i32)->bool);
    delegate!(owns(ct:&Ciphertext)->bool);
    delegate!(upload_ct(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(download_ct(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(upload_pt(pt:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<ckks_train_ckks::Plaintext>);
    delegate!(add(a:&Ciphertext,b:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> ckks_train_core::error::Result<Ciphertext> {
        let n = self.subs.get() + 1;
        self.subs.set(n);
        if n == self.at {
            return Err(ckks_train_core::error::HekitError::InvalidParams(
                "injected late subtraction failure".into(),
            ));
        }
        self.ev.sub(a, b)
    }
    delegate!(negate(a:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(add_plain(a:&Ciphertext,p:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_plain(a:&Ciphertext,p:&ckks_train_ckks::Plaintext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_scalar(ct:&Ciphertext,value:f64,target_scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_plain_to_scale(a:&Ciphertext,p:&ckks_train_ckks::Plaintext,target_scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul_no_relin(a:&Ciphertext,b:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(relinearize(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mul(a:&Ciphertext,b:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(rescale(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(mod_switch_to_level(ct:&Ciphertext,level:usize)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(relabel(ct:Ciphertext,scale:f64)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(rotate(ct:&Ciphertext,steps:i32)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(conjugate(ct:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(hoisted_rotate(ct:&Ciphertext,steps:&[i32])->ckks_train_core::error::Result<Vec<Ciphertext>>);
    delegate!(matvec_plain(diag:&ckks_train_ckks::Diagonals<ckks_train_ckks::Plaintext>,x:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
    delegate!(matvec_enc(diag:&ckks_train_ckks::Diagonals<Ciphertext>,x:&Ciphertext)->ckks_train_core::error::Result<Ciphertext>);
}
#[test]
fn late_backend_failure_drops_all_candidates_without_partial_stage_mutation() {
    let c = Ctx::new(true);
    let (stage, _, _) = c.stage(0, true);
    let records = c.records(0, 1, 0.0);
    let old_w: Vec<_> = stage
        .layers()
        .iter()
        .map(|s| c.vals(s.weights().ciphertext()))
        .collect();
    let old_b: Vec<_> = stage
        .biases()
        .unwrap()
        .iter()
        .map(|b| c.vals(b.ciphertext()))
        .collect();
    // loss subtraction, first weight, first bias, then second weight fails.
    let fail = FailSub {
        ev: HeOps::fork(&c.ev),
        at: 4,
        subs: std::cell::Cell::new(0),
    };
    assert!(
        stage
            .prepare_step(&fail, &c.e, &records, 0.04, GradRelin::Lazy)
            .is_err()
    );
    assert_eq!(fail.subs.get(), 4);
    assert!(fail.counter().ct_ct_tensor > 0);
    for k in 0..2 {
        near(&c.vals(stage.layers()[k].weights().ciphertext()), &old_w[k]);
        near(&c.vals(stage.biases().unwrap()[k].ciphertext()), &old_b[k]);
        assert_eq!(stage.layers()[k].update_index(), 0);
    }
    // Same old stage remains usable after the failed attempt, with its original generation.
    assert!(
        stage
            .prepare_step(&c.ev, &c.e, &records, 0.04, GradRelin::Lazy)
            .is_ok()
    );
}

#[test]
fn later_unrepresentable_velocity_scalar_fails_before_all_stage_work() {
    let c = Ctx::new(false);
    let (initial, _, _) = c.stage(0, false);
    let ws = std::array::from_fn(|k| initial.layers()[k].weights().clone());
    let stage = PackedStage::new(
        &c.ev,
        &c.plan,
        0,
        ws,
        None,
        PackedStageConfig::poly_relu2_nag(1e-7, 0.08).unwrap(),
    )
    .unwrap();
    let records = c.records(0, 1, 0.0);
    let (next, _) = stage
        .prepare_step(&c.ev, &c.e, &records, 0.08, GradRelin::Lazy)
        .unwrap();
    let next = next.into_stage();
    let before = c.vals(next.layers()[0].weights().ciphertext());
    c.ev.reset_counter();
    assert!(
        next.prepare_step(&c.ev, &c.e, &records, 0.08, GradRelin::Lazy)
            .is_err()
    );
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(next.layers()[0].update_index(), 1);
    near(&c.vals(next.layers()[0].weights().ciphertext()), &before);
}

#[test]
fn requested_lazy_respects_disabled_flag_without_changing_learning() {
    let mut c = Ctx::new(false);
    // Same seeded secret as Ctx::new; flags only change effective outer-sum relinearization.
    let mut kg = KeyGenerator::new(c.p.clone(), Rng::from_seed([91; 32])).unwrap();
    c.ev = Evaluator::with_config(
        c.p.clone(),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&c.plan.layout().galois_steps())),
        ckks_train_ckks::EvalConfig {
            lazy_relin: false,
            ..Default::default()
        },
    );
    let (stage, mut w, mut b) = c.stage(0, false);
    let mut v = std::array::from_fn(|k| Array2::zeros(w[k].dim()));
    let (next, receipt) = stage
        .prepare_step(&c.ev, &c.e, &c.records(0, 3, 0.0), 0.04, GradRelin::Lazy)
        .unwrap();
    assert!(!receipt.plan.effective_lazy);
    assert_eq!(receipt.actual_counts.relins, 21);
    assert_eq!(receipt.actual_counts.ct_ct_tensor, 21);
    reference(&mut w, &mut b, &mut v, 3, 0.04, false);
    let next = next.into_stage();
    for (k, expected) in w.iter().enumerate() {
        near(
            &c.vals(next.layers()[k].weights().ciphertext()),
            &c.plan.layers()[k].pack_weights_out_by_in(expected).unwrap(),
        );
    }
}
