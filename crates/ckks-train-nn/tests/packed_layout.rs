//! Independent geometry and encrypted primitive oracles; no training campaign.
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, HeOps, KeyGenerator,
    OpCounter,
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::layout_reboot::GradRelin;
use ckks_train_nn::packed_training::{
    Orientation, PackedGeometryPlan, PackedVector, PackedWeights, StageShape, VectorRepr, affine,
    outer_sum_scaled_masks,
};
use ndarray::Array2;
use std::sync::Arc;

fn plan(
    input: usize,
    hidden: &[usize],
    output: usize,
    real: usize,
    slots: usize,
) -> PackedGeometryPlan {
    let stages: Vec<_> = hidden
        .iter()
        .map(|&h| StageShape::new(h, output, real).unwrap())
        .collect();
    PackedGeometryPlan::new(input, &stages, slots).unwrap()
}

#[test]
fn registered_complete_graph_geometry() {
    for (_dataset, input, real, padded) in [
        ("iris", 4, 3, 4),
        ("breast_cancer", 30, 2, 2),
        ("mnist", 196, 10, 16),
        ("fashion", 196, 10, 16),
    ] {
        for (hidden, output, rows, cols) in [
            (&[10][..], real, 2048, 16),
            (&[32][..], padded, 1024, 32),
            (&[64, 32][..], padded, 512, 64),
            (&[128, 64, 32][..], padded, 256, 128),
        ] {
            let p = plan(input, hidden, output, real, 32768);
            assert_eq!((p.layout().r(), p.layout().c()), (rows, cols));
            assert_eq!(p.layers().len(), 2 * hidden.len());
            for (s, pair) in p.layers().chunks_exact(2).enumerate() {
                assert_eq!(
                    pair[0].orientation(),
                    if s % 2 == 0 {
                        Orientation::Row
                    } else {
                        Orientation::Column
                    }
                );
                assert_ne!(pair[0].orientation(), pair[1].orientation());
                assert_eq!(pair[0].output(), pair[1].input());
                assert_eq!(pair[1].output().logical_len(), output);
                assert_eq!(p.stages()[s].real_classes(), real);
                if s + 1 < hidden.len() {
                    assert_eq!(pair[0].output(), p.layers()[2 * (s + 1)].input());
                }
                assert_eq!(
                    pair[1].matrix_mask().iter().sum::<f64>(),
                    (hidden[s] * output) as f64
                );
            }
            // Batch count is not an argument or a lattice axis: every logical B shares this plan.
            for b in [1, 20, 48, 64, 128] {
                assert_eq!(p.sample_ciphertexts(b).unwrap(), b);
            }
        }
    }
    assert_eq!(
        (
            plan(784, &[32], 16, 10, 32768).layout().r(),
            plan(784, &[32], 16, 10, 32768).layout().c()
        ),
        (1024, 32)
    );
}

#[test]
fn nonsquare_geometry_mapping_and_rejections() {
    let p = plan(5, &[3, 7, 2], 6, 4, 128);
    assert_eq!((p.layout().r(), p.layout().c()), (16, 8));
    for l in p.layers() {
        let w = Array2::from_shape_fn(
            (l.output().logical_len(), l.input().logical_len()),
            |(o, i)| (100 * o + i + 1) as f64,
        );
        let packed = l.pack_weights_out_by_in(&w).unwrap();
        for i in 0..p.layout().r() {
            for j in 0..p.layout().c() {
                let (input, output) = match l.orientation() {
                    Orientation::Row => (i, j),
                    Orientation::Column => (j, i),
                };
                let want = if input < w.ncols() && output < w.nrows() {
                    w[[output, input]]
                } else {
                    0.0
                };
                assert_eq!(packed[i * p.layout().c() + j], want);
            }
        }
        assert!(l.pack_weights_out_by_in(&Array2::zeros((1, 1))).is_err());
        for spec in [l.input(), l.output()] {
            let values: Vec<_> = (0..spec.logical_len()).map(|i| i as f64 + 1.0).collect();
            let packed = spec.pack(&values).unwrap();
            for (s, &v) in packed.iter().enumerate() {
                let k = match spec.repr() {
                    VectorRepr::Expanded => s / p.layout().c(),
                    VectorRepr::Repeated => s % p.layout().c(),
                };
                assert_eq!(v, values.get(k).copied().unwrap_or(0.0));
            }
        }
    }
    assert!(StageShape::new(0, 4, 3).is_err());
    assert!(StageShape::new(2, 2, 3).is_err());
    assert!(PackedGeometryPlan::new(0, &[StageShape::new(3, 2, 2).unwrap()], 128).is_err());
    assert!(PackedGeometryPlan::new(5, &[], 128).is_err());
    assert!(PackedGeometryPlan::new(5, &[StageShape::new(3, 2, 2).unwrap()], 127).is_err());
    // Local head CE output needs rows; final output RE needs columns. Both must fit.
    assert!(PackedGeometryPlan::new(9, &[StageShape::new(9, 9, 9).unwrap()], 64).is_err());
    assert!(
        PackedGeometryPlan::new(
            5,
            &[
                StageShape::new(3, 33, 33).unwrap(),
                StageShape::new(7, 3, 3).unwrap()
            ],
            128
        )
        .is_err()
    );
    let varied = PackedGeometryPlan::new(
        5,
        &[
            StageShape::new(3, 6, 4).unwrap(),
            StageShape::new(7, 5, 3).unwrap(),
            StageShape::new(2, 9, 7).unwrap(),
        ],
        128,
    )
    .unwrap();
    assert_eq!((varied.layout().r(), varied.layout().c()), (16, 8));
    for (stage, (model, real)) in [(6, 4), (5, 3), (9, 7)].into_iter().enumerate() {
        assert_eq!(varied.stages()[stage].real_classes(), real);
        assert_eq!(varied.layers()[2 * stage + 1].output().logical_len(), model);
    }
    let fits = PackedGeometryPlan::new(
        5,
        &[
            StageShape::new(3, 17, 17).unwrap(),
            StageShape::new(7, 3, 3).unwrap(),
        ],
        128,
    )
    .unwrap();
    assert_eq!((fits.layout().r(), fits.layout().c()), (32, 4));
    assert!(p.sample_ciphertexts(0).is_err());
}

struct Ctx {
    p: Arc<CkksParams>,
    e: Encoder,
    enc: Encryptor,
    dec: Decryptor,
    ev: Evaluator,
}
impl Ctx {
    fn new() -> Self {
        let p = Arc::new(CkksParams::new(10, &[60, 40, 40, 40], &[60], 40, 4).unwrap());
        let geometry = plan(5, &[3], 4, 2, p.slots());
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([71; 32])).unwrap();
        Self {
            e: Encoder::new(p.clone()),
            enc: Encryptor::new(p.clone(), kg.public_key()),
            dec: Decryptor::new(p.clone(), kg.secret_key()),
            ev: Evaluator::new(
                p.clone(),
                Some(kg.relin_key()),
                Some(kg.galois_keys(&geometry.layout().galois_steps())),
            ),
            p,
        }
    }
    fn ct(&self, v: &[f64]) -> Ciphertext {
        self.enc.encrypt(
            &self.e.encode(v, self.p.scale(), 3).unwrap(),
            &mut Rng::from_seed([72; 32]),
        )
    }
    fn values(&self, c: &Ciphertext) -> Vec<f64> {
        self.e.decode(&self.dec.decrypt(c))
    }
}
fn near(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (s, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() < 1.0 / 32768.0, "slot {s}: {a} vs {b}");
    }
}

#[test]
fn encrypted_affine_both_orientations_masks_levels_and_counts() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    for l in p.layers() {
        let x: Vec<_> = (0..l.input().logical_len())
            .map(|i| 0.1 * (i + 1) as f64)
            .collect();
        let w = Array2::from_shape_fn((l.output().logical_len(), x.len()), |(o, i)| {
            0.025 * (2 * o + i + 1) as f64
        });
        let mut storage = l.pack_weights_out_by_in(&w).unwrap();
        // Deliberate structural garbage at inactive output coordinates must be removed.
        for (s, v) in storage.iter_mut().enumerate() {
            if l.output().mask()[s] == 0.0 {
                *v = 0.4;
            }
        }
        let xct = PackedVector::new(&c.ev, l.input(), c.ct(&l.input().pack(&x).unwrap())).unwrap();
        let wct = PackedWeights::new(&c.ev, l.clone(), c.ct(&storage)).unwrap();
        c.ev.reset_counter();
        let out = affine(&c.ev, &c.e, &wct, &xct).unwrap();
        let ys: Vec<f64> = (0..w.nrows())
            .map(|o| (0..w.ncols()).map(|i| w[[o, i]] * x[i]).sum())
            .collect();
        let expected: Vec<_> = (0..c.p.slots())
            .map(|s| {
                let k = if l.orientation() == Orientation::Row {
                    s % p.layout().c()
                } else {
                    s / p.layout().c()
                };
                ys.get(k).copied().unwrap_or(0.0)
            })
            .collect();
        near(&c.values(out.ciphertext()), &expected);
        assert_eq!(
            (out.ciphertext().level(), out.ciphertext().scale()),
            (1, c.p.scale())
        );
        let rotations = match l.orientation() {
            Orientation::Row => p.layout().r().trailing_zeros() as u64,
            Orientation::Column => 2 * p.layout().c().trailing_zeros() as u64,
        };
        assert_eq!(
            c.ev.counter(),
            OpCounter {
                rotations,
                ct_ct_tensor: 1,
                relins: 1,
                ct_pt_mults: 1,
                rescales: 2,
                ..Default::default()
            }
        );
    }
}

#[test]
fn encrypted_shared_outer_branches_use_only_valid_pairs() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    for l in p.layers() {
        let input = l.input();
        let output = l.output();
        let xs: Vec<Vec<f64>> = (0..3)
            .map(|b| {
                (0..input.logical_len())
                    .map(|i| (b + i + 1) as f64 * 0.1)
                    .collect()
            })
            .collect();
        let ys: Vec<Vec<f64>> = (0..3)
            .map(|b| {
                (0..output.logical_len())
                    .map(|i| (2 * b + i + 1) as f64 * -0.15)
                    .collect()
            })
            .collect();
        let xv: Vec<_> = xs
            .iter()
            .map(|x| {
                let mut values = input.pack(x).unwrap();
                for (v, m) in values.iter_mut().zip(input.mask()) {
                    if m == 0.0 {
                        *v = 0.7;
                    }
                }
                PackedVector::new(&c.ev, input, c.ct(&values)).unwrap()
            })
            .collect();
        let yv: Vec<_> = ys
            .iter()
            .map(|y| {
                let mut values = output.pack(y).unwrap();
                for (v, m) in values.iter_mut().zip(output.mask()) {
                    if m == 0.0 {
                        *v = 0.8;
                    }
                }
                PackedVector::new(&c.ev, output, c.ct(&values)).unwrap()
            })
            .collect();
        let pairs = vec![(&xv[0], &yv[0]), (&xv[2], &yv[2])]; // middle public-invalid record skipped by caller
        for (policy, relins) in [(GradRelin::Lazy, 1), (GradRelin::PerSample, 2)] {
            for constants in [vec![0.25], vec![0.25, 0.4]] {
                c.ev.reset_counter();
                let branches =
                    outer_sum_scaled_masks(&c.ev, &c.e, l, &pairs, &constants, policy).unwrap();
                assert_eq!(branches.len(), constants.len());
                for (g, &factor) in branches.iter().zip(&constants) {
                    assert_eq!(g.valid_samples(), 2);
                    assert_eq!(g.factor(), factor);
                    assert_eq!(g.layer(), l);
                    assert_eq!(
                        (g.ciphertext().level(), g.ciphertext().scale()),
                        (1, c.p.scale())
                    );
                    let expected: Vec<_> = (0..c.p.slots())
                        .map(|s| {
                            let (i, o) = if l.orientation() == Orientation::Row {
                                (s / p.layout().c(), s % p.layout().c())
                            } else {
                                (s % p.layout().c(), s / p.layout().c())
                            };
                            if i < input.logical_len() && o < output.logical_len() {
                                factor * (xs[0][i] * ys[0][o] + xs[2][i] * ys[2][o])
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    near(&c.values(g.ciphertext()), &expected);
                }
                assert_eq!(
                    c.ev.counter(),
                    OpCounter {
                        ct_ct_tensor: 2,
                        relins,
                        ct_pt_mults: constants.len() as u64,
                        rescales: 1 + constants.len() as u64,
                        ..Default::default()
                    }
                );
            }
        }
    }
}

#[test]
fn structural_failures_precede_counted_work() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    let l = &p.layers()[0];
    let x = PackedVector::new(&c.ev, l.input(), c.ct(&l.input().pack(&[0.1; 5]).unwrap())).unwrap();
    let y = PackedVector::new(
        &c.ev,
        l.output(),
        c.ct(&l.output().pack(&[0.2; 3]).unwrap()),
    )
    .unwrap();
    let w = PackedWeights::new(&c.ev, l.clone(), c.ct(&l.matrix_mask())).unwrap();
    let pairs = [(&x, &y)];
    c.ev.reset_counter();
    for constants in [
        vec![],
        vec![f64::NAN],
        vec![0.1, f64::INFINITY],
        vec![0.1, f64::MAX],
    ] {
        assert!(
            outer_sum_scaled_masks(&c.ev, &c.e, l, &pairs, &constants, GradRelin::Lazy).is_err()
        );
    }
    assert!(outer_sum_scaled_masks(&c.ev, &c.e, l, &[], &[0.1], GradRelin::Lazy).is_err());
    assert!(outer_sum_scaled_masks(&c.ev, &c.e, l, &[(&y, &x)], &[0.1], GradRelin::Lazy).is_err());
    assert!(affine(&c.ev, &c.e, &w, &y).is_err());
    let no_keys = Evaluator::new(c.p.clone(), None, None);
    assert!(affine(&no_keys, &c.e, &w, &x).is_err());
    assert!(outer_sum_scaled_masks(&no_keys, &c.e, l, &pairs, &[0.1], GradRelin::Lazy).is_err());
    assert_eq!(no_keys.counter(), OpCounter::default());
    let low = PackedVector::new(
        &c.ev,
        l.input(),
        c.ev.mod_switch_to_level(x.ciphertext(), 1).unwrap(),
    )
    .unwrap();
    assert!(affine(&c.ev, &c.e, &w, &low).is_err());
    let huge = PackedVector::new(
        &c.ev,
        l.input(),
        c.ev.relabel(x.ciphertext().clone(), 1e300).unwrap(),
    )
    .unwrap();
    assert!(affine(&c.ev, &c.e, &w, &huge).is_err());
    assert!(
        outer_sum_scaled_masks(&c.ev, &c.e, l, &[(&huge, &y)], &[0.1], GradRelin::Lazy).is_err()
    );
    let wrong_scale =
        c.ev.relabel(w.ciphertext().clone(), 2.0 * c.p.scale())
            .unwrap();
    assert!(PackedWeights::new(&c.ev, l.clone(), wrong_scale).is_err());
    let other = Ctx::new();
    assert!(PackedVector::new(&other.ev, l.input(), x.ciphertext().clone()).is_err());
    assert!(affine(&c.ev, &other.e, &w, &x).is_err());
    assert_eq!(c.ev.counter(), OpCounter::default());
}

#[test]
fn masked_expand_preserves_legacy_all_rows_and_rejects_before_work() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    let l = p.layout();
    let values: Vec<_> = (0..c.p.slots()).map(|s| 0.01 * (s % 11) as f64).collect();
    let ct = c.ct(&values);
    c.ev.reset_counter();
    let legacy = l
        .expand_to_scale(&c.ev, &c.e, &ct, 0.5, c.p.scale())
        .unwrap();
    let legacy_counts = c.ev.counter();
    c.ev.reset_counter();
    let masked = l
        .expand_valid_to_scale(&c.ev, &c.e, &ct, 3, 0.5, c.p.scale())
        .unwrap();
    assert_eq!(c.ev.counter(), legacy_counts);
    assert_eq!(
        legacy_counts,
        OpCounter {
            rotations: 2 * l.c().trailing_zeros() as u64,
            ct_pt_mults: 1,
            rescales: 1,
            ..Default::default()
        }
    );
    let expected: Vec<_> = (0..c.p.slots())
        .map(|s| {
            0.5 * (0..l.c())
                .map(|j| values[s / l.c() * l.c() + j])
                .sum::<f64>()
        })
        .collect();
    near(&c.values(&legacy), &expected);
    near(
        &c.values(&masked),
        &expected
            .iter()
            .enumerate()
            .map(|(s, &v)| if s / l.c() < 3 { v } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    assert_eq!((masked.level(), masked.scale()), (2, c.p.scale()));
    c.ev.reset_counter();
    for (rows, factor, target) in [
        (0, 1.0, c.p.scale()),
        (l.r() + 1, 1.0, c.p.scale()),
        (3, f64::NAN, c.p.scale()),
        (3, f64::MAX, c.p.scale()),
        (3, 1.0, f64::NAN),
        (3, 1.0, 1.0),
    ] {
        assert!(
            l.expand_valid_to_scale(&c.ev, &c.e, &ct, rows, factor, target)
                .is_err()
        );
    }
    let no_rotation = Evaluator::new(c.p.clone(), None, None);
    assert!(
        l.expand_valid_to_scale(&no_rotation, &c.e, &ct, 3, 1.0, c.p.scale())
            .is_err()
    );
    assert_eq!(no_rotation.counter(), OpCounter::default());
    assert_eq!(c.ev.counter(), OpCounter::default());
}

#[test]
fn mixed_histories_components_and_missing_rotation_keys_fail_before_work() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    let l = &p.layers()[0];
    let x = PackedVector::new(&c.ev, l.input(), c.ct(&l.input().pack(&[0.1; 5]).unwrap())).unwrap();
    let y = PackedVector::new(
        &c.ev,
        l.output(),
        c.ct(&l.output().pack(&[0.2; 3]).unwrap()),
    )
    .unwrap();
    let low = PackedVector::new(
        &c.ev,
        l.input(),
        c.ev.mod_switch_to_level(x.ciphertext(), 2).unwrap(),
    )
    .unwrap();
    let scale = PackedVector::new(
        &c.ev,
        l.input(),
        c.ev.relabel(x.ciphertext().clone(), c.p.scale() * 2.0)
            .unwrap(),
    )
    .unwrap();
    let degree2 = c.ev.mul_no_relin(x.ciphertext(), x.ciphertext()).unwrap();
    let w = PackedWeights::new(&c.ev, l.clone(), c.ct(&l.matrix_mask())).unwrap();
    let mut kg = KeyGenerator::new(c.p.clone(), Rng::from_seed([71; 32])).unwrap();
    let no_rotation = Evaluator::new(c.p.clone(), Some(kg.relin_key()), None);
    c.ev.reset_counter();
    assert!(affine(&no_rotation, &c.e, &w, &x).is_err());
    assert_eq!(no_rotation.counter(), OpCounter::default());
    assert!(PackedVector::new(&c.ev, l.input(), degree2.clone()).is_err());
    assert!(PackedWeights::new(&c.ev, l.clone(), degree2).is_err());
    for bad in [&low, &scale] {
        assert!(
            outer_sum_scaled_masks(
                &c.ev,
                &c.e,
                l,
                &[(&x, &y), (bad, &y)],
                &[0.2],
                GradRelin::Lazy
            )
            .is_err()
        );
    }
    let other_slots = plan(5, &[3], 4, 2, 256);
    assert!(
        PackedVector::new(
            &c.ev,
            other_slots.layers()[0].input(),
            x.ciphertext().clone()
        )
        .is_err()
    );
    assert_eq!(c.ev.counter(), OpCounter::default());
}

#[test]
fn lazy_disabled_and_zero_negative_branches_keep_explicit_counts() {
    use ckks_train_ckks::EvalConfig;
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    let l = &p.layers()[0];
    let mut kg = KeyGenerator::new(c.p.clone(), Rng::from_seed([71; 32])).unwrap();
    let ev = Evaluator::with_config(
        c.p.clone(),
        Some(kg.relin_key()),
        None,
        EvalConfig {
            lazy_relin: false,
            ..Default::default()
        },
    );
    let x = PackedVector::new(&ev, l.input(), c.ct(&l.input().pack(&[0.1; 5]).unwrap())).unwrap();
    let y = PackedVector::new(&ev, l.output(), c.ct(&l.output().pack(&[0.2; 3]).unwrap())).unwrap();
    let out = outer_sum_scaled_masks(
        &ev,
        &c.e,
        l,
        &[(&x, &y), (&x, &y)],
        &[0.0, -0.5],
        GradRelin::Lazy,
    )
    .unwrap();
    near(&c.values(out[0].ciphertext()), &vec![0.0; c.p.slots()]);
    let expected: Vec<_> = (0..c.p.slots())
        .map(|s| {
            if s / p.layout().c() < 5 && s % p.layout().c() < 3 {
                -0.02
            } else {
                0.0
            }
        })
        .collect();
    near(&c.values(out[1].ciphertext()), &expected);
    assert_eq!(
        ev.counter(),
        OpCounter {
            ct_ct_tensor: 2,
            relins: 2,
            ct_pt_mults: 2,
            rescales: 3,
            ..Default::default()
        }
    );
}

#[test]
fn natural_input_scale_and_mixed_operand_levels_normalize_without_relabeling() {
    let c = Ctx::new();
    let p = plan(5, &[3], 4, 2, c.p.slots());
    for l in p.layers() {
        let input = l.input();
        let output = l.output();
        let xvals = vec![0.2; input.logical_len()];
        let pt =
            c.e.encode(&input.pack(&xvals).unwrap(), 2.0 * c.p.scale(), 2)
                .unwrap();
        let raw = c.enc.encrypt(&pt, &mut Rng::from_seed([77; 32]));
        let x = PackedVector::new(&c.ev, input, raw).unwrap();
        let y = PackedVector::new(
            &c.ev,
            output,
            c.ct(&output.pack(&vec![0.3; output.logical_len()]).unwrap()),
        )
        .unwrap();
        let w = PackedWeights::new(
            &c.ev,
            l.clone(),
            c.ct(&l
                .pack_weights_out_by_in(&Array2::from_elem(
                    (output.logical_len(), input.logical_len()),
                    0.1,
                ))
                .unwrap()),
        )
        .unwrap();
        c.ev.reset_counter();
        let z = affine(&c.ev, &c.e, &w, &x).unwrap();
        let expected: Vec<_> = (0..c.p.slots())
            .map(|s| {
                let coordinate = if l.orientation() == Orientation::Row {
                    s % p.layout().c()
                } else {
                    s / p.layout().c()
                };
                if coordinate < output.logical_len() {
                    0.02 * input.logical_len() as f64
                } else {
                    0.0
                }
            })
            .collect();
        near(&c.values(z.ciphertext()), &expected);
        assert_eq!(
            (z.ciphertext().level(), z.ciphertext().scale()),
            (0, c.p.scale())
        );
        c.ev.reset_counter();
        let g =
            outer_sum_scaled_masks(&c.ev, &c.e, l, &[(&x, &y)], &[0.5], GradRelin::Lazy).unwrap();
        let expected: Vec<_> = (0..c.p.slots())
            .map(|s| {
                let (i, o) = if l.orientation() == Orientation::Row {
                    (s / p.layout().c(), s % p.layout().c())
                } else {
                    (s % p.layout().c(), s / p.layout().c())
                };
                if i < input.logical_len() && o < output.logical_len() {
                    0.03
                } else {
                    0.0
                }
            })
            .collect();
        near(&c.values(g[0].ciphertext()), &expected);
        assert_eq!(g[0].valid_samples(), 1);
        assert_eq!(
            (g[0].ciphertext().level(), g[0].ciphertext().scale()),
            (0, c.p.scale())
        );
        assert_eq!(
            c.ev.counter(),
            OpCounter {
                ct_ct_tensor: 1,
                relins: 1,
                ct_pt_mults: 1,
                rescales: 2,
                ..Default::default()
            }
        );
    }
}
