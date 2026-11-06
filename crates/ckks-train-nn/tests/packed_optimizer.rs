//! Small insecure CPU correctness oracles, not training or bootstrap evidence.
//! The reference advances unscaled V, independently of the encrypted U recurrence.
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, KeyGenerator, OpCounter,
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::{
    layout_reboot::GradRelin,
    packed_training::{
        LayerGeometry, Orientation, PackedGeometryPlan, PackedGradient, PackedLayerState,
        PackedOptimizer, PackedVector, PackedWeights, StageShape, outer_sum_scaled_masks,
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
}
impl Ctx {
    fn new() -> Self {
        let p = Arc::new(
            CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap(),
        );
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([81; 32])).unwrap();
        Self {
            e: Encoder::new(p.clone()),
            enc: Encryptor::new(p.clone(), kg.public_key()),
            dec: Decryptor::new(p.clone(), kg.secret_key()),
            ev: Evaluator::new(p.clone(), Some(kg.relin_key()), None),
            p,
        }
    }
    fn ct(&self, values: &[f64], level: usize) -> Ciphertext {
        self.enc.encrypt(
            &self.e.encode(values, self.p.scale(), level).unwrap(),
            &mut Rng::from_seed([82; 32]),
        )
    }
    fn values(&self, ct: &Ciphertext) -> Vec<f64> {
        self.e.decode(&self.dec.decrypt(ct))
    }
    fn plan(&self) -> PackedGeometryPlan {
        PackedGeometryPlan::new(5, &[StageShape::new(3, 4, 2).unwrap()], self.p.slots()).unwrap()
    }
    fn state(&self, l: &LayerGeometry, policy: PackedOptimizer) -> PackedLayerState<Ciphertext> {
        let w = Array2::from_shape_fn(
            (l.output().logical_len(), l.input().logical_len()),
            |(o, i)| 0.1 + 0.02 * (o + i) as f64,
        );
        PackedLayerState::new(
            &self.ev,
            PackedWeights::new(
                &self.ev,
                l.clone(),
                self.ct(&l.pack_weights_out_by_in(&w).unwrap(), 9),
            )
            .unwrap(),
            policy,
        )
        .unwrap()
    }
    fn branches(
        &self,
        l: &LayerGeometry,
        n: usize,
        step: usize,
        level: usize,
        factors: &[f64],
    ) -> Vec<PackedGradient<Ciphertext>> {
        let pairs: Vec<_> = (0..n)
            .map(|b| {
                let x: Vec<_> = (0..l.input().logical_len())
                    .map(|i| xval(step, b, i))
                    .collect();
                let dz: Vec<_> = (0..l.output().logical_len())
                    .map(|o| dzval(step, b, o))
                    .collect();
                (
                    PackedVector::new(
                        &self.ev,
                        l.input(),
                        self.ct(&l.input().pack(&x).unwrap(), level),
                    )
                    .unwrap(),
                    PackedVector::new(
                        &self.ev,
                        l.output(),
                        self.ct(&l.output().pack(&dz).unwrap(), level),
                    )
                    .unwrap(),
                )
            })
            .collect();
        outer_sum_scaled_masks(
            &self.ev,
            &self.e,
            l,
            &pairs.iter().map(|(a, b)| (a, b)).collect::<Vec<_>>(),
            factors,
            GradRelin::Lazy,
        )
        .unwrap()
    }
}
fn xval(t: usize, b: usize, i: usize) -> f64 {
    0.08 * (1 + t + b + i) as f64
}
fn dzval(t: usize, b: usize, o: usize) -> f64 {
    0.06 * (1 + b + o) as f64 * if t % 2 == 0 { 1.0 } else { -0.7 }
}
fn near(a: &[f64], b: &[f64]) {
    for (slot, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() < 1e-7, "slot {slot}: {a} != {b}");
    }
}
fn packed_reference(l: &LayerGeometry, a: &Array2<f64>) -> Vec<f64> {
    (0..l.layout().slots())
        .map(|s| {
            let (i, o) = match l.orientation() {
                Orientation::Row => (s / l.layout().c(), s % l.layout().c()),
                Orientation::Column => (s % l.layout().c(), s / l.layout().c()),
            };
            if o < a.nrows() && i < a.ncols() {
                a[[o, i]]
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn four_rate_nag_matches_unscaled_velocity_and_counts_in_both_orientations() {
    let c = Ctx::new();
    for l in c.plan().layers() {
        let mut state = c.state(l, PackedOptimizer::nag(0.9, 0.08).unwrap());
        let mut w = Array2::from_shape_fn(
            (l.output().logical_len(), l.input().logical_len()),
            |(o, i)| 0.1 + 0.02 * (o + i) as f64,
        );
        let mut v = Array2::zeros(w.dim());
        for (t, lr) in [0.04, 0.00008, 0.02, 0.08].into_iter().enumerate() {
            let n = [3, 1, 2, 3][t];
            let level = 9 - 2 * t;
            let req = state.request(lr, n).unwrap();
            assert_eq!(req.update_index(), t as u64);
            assert_eq!(req.factors().len(), if lr == 0.08 { 1 } else { 2 });
            c.ev.reset_counter();
            let raw = c.branches(l, n, t, level, req.factors());
            let branches = req.factors().len() as u64;
            assert_eq!(
                c.ev.counter(),
                OpCounter {
                    ct_ct_tensor: n as u64,
                    relins: 1,
                    ct_pt_mults: branches,
                    rescales: 1 + branches,
                    ..Default::default()
                }
            );
            let bound = req.bind(raw).unwrap();
            c.ev.reset_counter();
            let old_values = c.values(state.weights().ciphertext());
            let candidate = state.prepare(&c.ev, &bound).unwrap();
            assert_eq!(
                c.ev.counter(),
                OpCounter {
                    ct_pt_mults: if t == 0 { 0 } else { 2 },
                    rescales: if t == 0 { 0 } else { 2 },
                    ..Default::default()
                }
            );
            near(&c.values(state.weights().ciphertext()), &old_values);
            assert_eq!(state.update_index(), t as u64);
            for o in 0..w.nrows() {
                for i in 0..w.ncols() {
                    let g = (0..n).map(|b| xval(t, b, i) * dzval(t, b, o)).sum::<f64>() / n as f64;
                    v[[o, i]] = 0.9 * v[[o, i]] + g;
                    w[[o, i]] -= lr * (g + 0.9 * v[[o, i]]);
                }
            }
            state = candidate.into_state();
            near(
                &c.values(state.weights().ciphertext()),
                &packed_reference(l, &w),
            );
            let u = state.velocity().unwrap();
            let expected_u = v.mapv(|v| v * 0.08 * 1.9);
            near(&c.values(u.ciphertext()), &packed_reference(l, &expected_u));
            // Check unscaled V separately, not only its encrypted representation.
            near(
                &c.values(u.ciphertext())
                    .iter()
                    .map(|x| x / (0.08 * 1.9))
                    .collect::<Vec<_>>(),
                &packed_reference(l, &v),
            );
            assert_eq!(
                (u.ciphertext().level(), u.ciphertext().scale()),
                (level - 2, c.p.scale())
            );
            assert_eq!(
                (
                    state.weights().ciphertext().level(),
                    state.weights().ciphertext().scale()
                ),
                (level - 2, c.p.scale())
            );
            c.ev.reset_counter();
            assert!(state.prepare(&c.ev, &bound).is_err());
            assert_eq!(c.ev.counter(), OpCounter::default());
        }
    }
}

#[test]
fn sgd_increasing_rates_use_actual_sample_mean_without_reference_ceiling() {
    let c = Ctx::new();
    for l in c.plan().layers() {
        let mut state = c.state(l, PackedOptimizer::sgd());
        let mut w = Array2::from_shape_fn(
            (l.output().logical_len(), l.input().logical_len()),
            |(o, i)| 0.1 + 0.02 * (o + i) as f64,
        );
        for (t, lr) in [0.001, 0.04, 0.3].into_iter().enumerate() {
            let n = t + 1;
            let req = state.request(lr, n).unwrap();
            assert_eq!(req.factors(), &[lr / n as f64]);
            let bound = req.bind(c.branches(l, n, t, 7, req.factors())).unwrap();
            c.ev.reset_counter();
            state = state.prepare(&c.ev, &bound).unwrap().into_state();
            assert_eq!(c.ev.counter(), OpCounter::default());
            assert!(state.velocity().is_none());
            for o in 0..w.nrows() {
                for i in 0..w.ncols() {
                    w[[o, i]] -=
                        lr * (0..n).map(|b| xval(t, b, i) * dzval(t, b, o)).sum::<f64>() / n as f64;
                }
            }
            near(
                &c.values(state.weights().ciphertext()),
                &packed_reference(l, &w),
            );
        }
    }
}

#[test]
fn invalid_metadata_and_stale_or_unrelated_generation_fail_without_work() {
    let c = Ctx::new();
    let plan = c.plan();
    let l = &plan.layers()[0];
    let state = c.state(l, PackedOptimizer::nag(0.9, 0.08).unwrap());
    for bad in [0.0, -0.1, f64::NAN, f64::INFINITY, 0.081, f64::MIN_POSITIVE] {
        assert!(state.request(bad, 3).is_err());
    }
    assert!(state.request(0.04, 0).is_err());
    for (mu, r) in [
        (f64::NAN, 0.08),
        (f64::INFINITY, 0.08),
        (-0.1, 0.08),
        (1.0, 0.08),
        (0.9, 0.0),
        (0.9, f64::INFINITY),
        (0.9, f64::MAX),
    ] {
        assert!(PackedOptimizer::nag(mu, r).is_err());
    }
    let req = state.request(0.04, 3).unwrap();
    let raw = c.branches(l, 3, 0, 7, req.factors());
    let wrong_layer = c.branches(&plan.layers()[1], 3, 0, 7, req.factors());
    let wrong_n = c.branches(l, 2, 0, 7, req.factors());
    let wrong_factor = c.branches(l, 3, 0, 7, &[0.03, 0.05]);
    let missing = c.branches(l, 3, 0, 7, &req.factors()[..1]);
    let mut wrong_level = raw.clone();
    wrong_level[1] = c.branches(l, 3, 0, 5, &req.factors()[1..])[0].clone();
    let equal = state.request(0.08, 3).unwrap();
    let extra = c.branches(l, 3, 0, 7, &[equal.factors()[0], equal.factors()[0]]);
    c.ev.reset_counter();
    for bad in [wrong_layer, wrong_n, wrong_factor, missing] {
        assert!(req.bind(bad).is_err());
    }
    assert!(equal.bind(extra).is_err());
    let bound = req.bind(raw).unwrap();
    let unrelated = c.state(l, PackedOptimizer::nag(0.9, 0.08).unwrap());
    assert!(unrelated.prepare(&c.ev, &bound).is_err());
    if let Ok(bad) = req.bind(wrong_level) {
        assert!(state.prepare(&c.ev, &bad).is_err());
    }
    assert_eq!(c.ev.counter(), OpCounter::default());
    let next = state.prepare(&c.ev, &bound).unwrap().into_state();
    let req2 = next.request(0.04, 3).unwrap();
    // U is at level 5; this gradient also at 5 leaves no scalar/rescale level.
    let high = req2.bind(c.branches(l, 3, 1, 7, req2.factors())).unwrap();
    c.ev.reset_counter();
    assert!(next.prepare(&c.ev, &high).is_err());
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(next.update_index(), 1);
    let other = Ctx::new();
    assert!(next.prepare(&other.ev, &high).is_err());
    assert_eq!(other.ev.counter(), OpCounter::default());
}

#[test]
fn weight_context_scale_components_and_gradient_context_are_checked() {
    use ckks_train_ckks::HeOps;
    let c = Ctx::new();
    let other = Ctx::new();
    let plan = c.plan();
    let l = &plan.layers()[0];
    let ct = c.ct(&l.matrix_mask(), 9);
    let wrong_scale = c.ev.relabel(ct.clone(), c.p.scale() * 2.0).unwrap();
    let degree2 = c.ev.mul_no_relin(&ct, &ct).unwrap();
    let foreign = PackedWeights::new(&other.ev, l.clone(), other.ct(&l.matrix_mask(), 9)).unwrap();
    let state = c.state(l, PackedOptimizer::nag(0.9, 0.08).unwrap());
    let req = state.request(0.04, 2).unwrap();
    let foreign_grad = req.bind(other.branches(l, 2, 0, 7, req.factors())).unwrap();
    let different_graph = PackedGeometryPlan::new(
        5,
        &[
            StageShape::new(3, 4, 2).unwrap(),
            StageShape::new(3, 4, 2).unwrap(),
        ],
        c.p.slots(),
    )
    .unwrap();
    let wrong_identity = c.branches(&different_graph.layers()[0], 2, 0, 7, req.factors());
    c.ev.reset_counter();
    assert!(PackedWeights::new(&c.ev, l.clone(), wrong_scale).is_err());
    assert!(PackedWeights::new(&c.ev, l.clone(), degree2).is_err());
    assert!(PackedLayerState::new(&c.ev, foreign, PackedOptimizer::sgd()).is_err());
    assert!(state.prepare(&c.ev, &foreign_grad).is_err());
    assert!(req.bind(wrong_identity).is_err());
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(state.update_index(), 0);
}

#[test]
fn low_weight_level_alignment_and_zero_momentum_keep_exact_delta() {
    let c = Ctx::new();
    let plan = c.plan();
    for l in plan.layers() {
        let w = Array2::from_shape_fn((l.output().logical_len(), l.input().logical_len()), |_| {
            0.25
        });
        let weights = PackedWeights::new(
            &c.ev,
            l.clone(),
            c.ct(&l.pack_weights_out_by_in(&w).unwrap(), 1),
        )
        .unwrap();
        let mut state =
            PackedLayerState::new(&c.ev, weights, PackedOptimizer::nag(0.0, 0.08).unwrap())
                .unwrap();
        let mut expected = w;
        for t in 0..2 {
            let req = state.request(0.08, 1).unwrap();
            assert_eq!(req.factors(), &[0.08]);
            let bound = req
                .bind(c.branches(l, 1, t, 7 - 2 * t, req.factors()))
                .unwrap();
            c.ev.reset_counter();
            state = state.prepare(&c.ev, &bound).unwrap().into_state();
            assert_eq!(
                c.ev.counter(),
                OpCounter {
                    ct_pt_mults: if t == 0 { 0 } else { 2 },
                    rescales: if t == 0 { 0 } else { 2 },
                    ..Default::default()
                }
            );
            for o in 0..expected.nrows() {
                for i in 0..expected.ncols() {
                    expected[[o, i]] -= 0.08 * xval(t, 0, i) * dzval(t, 0, o);
                }
            }
            near(
                &c.values(state.weights().ciphertext()),
                &packed_reference(l, &expected),
            );
            assert_eq!(
                (
                    state.weights().ciphertext().level(),
                    state.weights().ciphertext().scale()
                ),
                (1, c.p.scale())
            );
        }
    }
}

#[test]
fn scalar_rounding_to_zero_rejects_later_update_before_either_scalar_branch() {
    let c = Ctx::new();
    let plan = c.plan();
    let l = &plan.layers()[0];
    // kappa is positive in f64, but rounds to zero at the actual 40-bit scalar scale.
    let state = c.state(l, PackedOptimizer::nag(1e-7, 0.08).unwrap());
    let req = state.request(0.08, 1).unwrap();
    let first = req.bind(c.branches(l, 1, 0, 9, req.factors())).unwrap();
    let state = state.prepare(&c.ev, &first).unwrap().into_state();
    let req = state.request(0.08, 1).unwrap();
    let second = req.bind(c.branches(l, 1, 1, 7, req.factors())).unwrap();
    let before = c.values(state.weights().ciphertext());
    c.ev.reset_counter();
    assert!(state.prepare(&c.ev, &second).is_err());
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(state.update_index(), 1);
    near(&c.values(state.weights().ciphertext()), &before);
}
