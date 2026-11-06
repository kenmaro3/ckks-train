//! Original analytic CPU refresh fixture. Insecure N=256; no NI/bootstrap evidence.
//! Construction and independent loop-oracle pattern follows packed_training.rs.
use ckks_train_ckks::{
    Ciphertext, CkksParams, Decryptor, Encoder, Encryptor, Evaluator, HeOps, KeyGenerator,
    OpCounter,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::packed_training::*;
use ckks_train_nn::{
    layout_reboot::GradRelin,
    packed_training::{
        PackedGeometryPlan, PackedRecord, PackedStage, PackedStageConfig, PackedVector,
        PackedWeights, StageShape, VectorSpec,
    },
};
use ndarray::Array2;
use std::sync::Arc;
use std::time::Instant;
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
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([111; 32])).unwrap();
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
            &mut Rng::from_seed([112; 32]),
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
                |(o, i)| 0.025 + 0.005 * (o + i) as f64,
            )
        });
        let b = std::array::from_fn(|k| {
            vec![if square { 0.03 } else { 0.0 }; ls[k].output().logical_len()]
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
                            .map(|i| 0.12 + 0.01 * (i + j) as f64)
                            .collect::<Vec<_>>(),
                        31,
                    ),
                    self.vec(
                        ls[1].output(),
                        &(0..ls[1].output().logical_len())
                            .map(|o| if o == j % 3 { 0.2 + target_shift } else { 0.0 })
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
            .map(|i| 0.12 + 0.01 * (i + j) as f64)
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
                (y - if o == j % 3 { 0.2 } else { 0.0 })
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

// Client alone has decryptor/encryptor/RNG. Server receives only this trait capability.
struct Client<'a> {
    c: &'a Ctx,
    contract: PackedRefreshContract,
    rng: Rng,
    fail_at: Option<usize>,
    calls: usize,
}
impl<'a> Client<'a> {
    fn new(c: &'a Ctx) -> Self {
        Self {
            c,
            contract: PackedRefreshContract::new(&c.ev, 31).unwrap(),
            rng: Rng::from_seed([113; 32]),
            fail_at: None,
            calls: 0,
        }
    }
}
impl<E: HeOps<Ct = Ciphertext>> PackedClientRefresh<E> for Client<'_> {
    fn output_contract(&self) -> &PackedRefreshContract {
        &self.contract
    }
    fn refresh(
        &mut self,
        r: &PackedRefreshRequest<Ciphertext>,
    ) -> Result<(PackedRefreshResponse<Ciphertext>, ClientRefreshTiming)> {
        let call = self.calls;
        self.calls += 1;
        if self.fail_at == Some(call) {
            return Err(HekitError::InvalidParams("injected client failure".into()));
        }
        let start = Instant::now();
        let values = self.c.vals(r.ciphertext());
        let decrypt_decode = start.elapsed();
        let start = Instant::now();
        let packed = match r.shape() {
            PackedRefreshShape::Matrix(l) => {
                let w = Array2::from_shape_fn(
                    (l.output().logical_len(), l.input().logical_len()),
                    |(o, i)| {
                        values[match l.orientation() {
                            Orientation::Row => i * l.layout().c() + o,
                            Orientation::Column => o * l.layout().c() + i,
                        }]
                    },
                );
                l.pack_weights_out_by_in(&w)?
            }
            PackedRefreshShape::Vector(s) => {
                let mut sums = vec![0.0; s.logical_len()];
                let mut counts = vec![0usize; s.logical_len()];
                for (slot, value) in values.iter().enumerate() {
                    let k = match s.repr() {
                        VectorRepr::Expanded => slot / s.layout().c(),
                        VectorRepr::Repeated => slot % s.layout().c(),
                    };
                    if k < sums.len() {
                        sums[k] += value;
                        counts[k] += 1;
                    }
                }
                s.pack(
                    &sums
                        .iter()
                        .zip(counts)
                        .map(|(v, n)| v / n as f64)
                        .collect::<Vec<_>>(),
                )?
            }
        };
        let pt = self.c.e.encode(
            &packed,
            r.output_contract().delta(),
            r.output_contract().level(),
        )?;
        let ct = self.c.enc.encrypt(&pt, &mut self.rng);
        let canonical_repack_encrypt = start.elapsed();
        Ok((
            PackedRefreshResponse::new(&self.c.ev, r, ct)?,
            ClientRefreshTiming {
                decrypt_decode: Some(decrypt_decode),
                canonical_repack_encrypt: Some(canonical_repack_encrypt),
                ..Default::default()
            },
        ))
    }
}
fn responses(
    c: &Ctx,
    batch: &PackedRefreshBatch<'_, Ciphertext>,
) -> Vec<PackedRefreshResponse<Ciphertext>> {
    let mut client = Client::new(c);
    batch
        .requests()
        .iter()
        .map(|r| {
            <Client as PackedClientRefresh<Evaluator>>::refresh(&mut client, r)
                .unwrap()
                .0
        })
        .collect()
}
fn snapshot(c: &Ctx, stage: &PackedStage<Ciphertext>) -> Vec<Vec<f64>> {
    let mut out = vec![];
    for k in 0..2 {
        out.push(c.vals(stage.layers()[k].weights().ciphertext()));
        if let Some(u) = stage.layers()[k].velocity() {
            out.push(c.vals(u.ciphertext()));
        }
        if let Some(b) = stage.biases() {
            out.push(c.vals(b[k].ciphertext()));
        }
    }
    out
}

// Break caught: erased U, reference/index reset, updated hidden recomputation, duplicate
// update, loss/mean drift, missing client canonicalization or fake bootstrap counts.
#[test]
fn interactive_four_updates_preserve_unscaled_velocity_bias_old_forward_and_counts() {
    for square in [false, true] {
        let c = Ctx::new(square);
        let server = PackedInteractiveStage::new(&c.ev, &c.e).unwrap();
        for s in 0..2 {
            let (mut stage, mut w, mut b) = c.stage(s, square);
            let mut v: [Array2<f64>; 2] = std::array::from_fn(|k| Array2::zeros(w[k].dim()));
            let mut client = Client::new(&c);
            for (t, lr) in [0.04, 0.00008, 0.02, 0.08].into_iter().enumerate() {
                let n = [3, 1, 2, 3][t];
                let mut records = c.records(s, n, 0.0);
                let ls = &c.plan.layers()[2 * s..2 * s + 2];
                records.insert(
                    1,
                    PackedRecord::new(
                        c.vec(ls[0].input(), &vec![0.9; ls[0].input().logical_len()], 31),
                        c.vec(ls[1].output(), &vec![0.8; ls[1].output().logical_len()], 31),
                        false,
                    ),
                );
                let old = snapshot(&c, &stage);
                let plan = stage
                    .plan_step(&c.ev, &c.e, &records, lr, GradRelin::Lazy)
                    .unwrap();
                let before = c.ev.counter();
                let (candidate, receipt) = server
                    .prepare_step(&stage, &records, lr, GradRelin::Lazy, &mut client)
                    .unwrap();
                assert_eq!(&c.ev.counter() - &before, plan.predicted_counts);
                assert_eq!(receipt.actual_counts(), &plan.predicted_counts);
                assert_eq!(snapshot(&c, &stage), old);
                let expected_roles = if square {
                    vec![
                        PackedRefreshRole::Weight(2 * s),
                        PackedRefreshRole::Bias(2 * s),
                        PackedRefreshRole::Weight(2 * s + 1),
                        PackedRefreshRole::Bias(2 * s + 1),
                    ]
                } else {
                    vec![
                        PackedRefreshRole::Weight(2 * s),
                        PackedRefreshRole::Velocity(2 * s),
                        PackedRefreshRole::Weight(2 * s + 1),
                        PackedRefreshRole::Velocity(2 * s + 1),
                    ]
                };
                assert_eq!(
                    receipt
                        .events()
                        .iter()
                        .map(|e| e.role())
                        .collect::<Vec<_>>(),
                    expected_roles
                );
                assert_eq!(receipt.events().len(), 4);
                let (hs, ys) = reference(&mut w, &mut b, &mut v, n, lr, square);
                let ls = &c.plan.layers()[2 * s..2 * s + 2];
                for j in 0..n {
                    near(
                        &c.vals(receipt.step().hidden[j].ciphertext()),
                        &ls[0].output().pack(&hs[j]).unwrap(),
                    );
                    near(
                        &c.vals(receipt.step().output[j].ciphertext()),
                        &ls[1].output().pack(&ys[j]).unwrap(),
                    );
                }
                for k in 0..2 {
                    for st in [receipt.updated_stage(), candidate.stage()] {
                        assert_eq!(st.layers()[k].update_index(), t as u64 + 1);
                        near(
                            &c.vals(st.layers()[k].weights().ciphertext()),
                            &ls[k].pack_weights_out_by_in(&w[k]).unwrap(),
                        );
                        if square {
                            near(
                                &c.vals(st.biases().unwrap()[k].ciphertext()),
                                &ls[k].output().pack(&b[k]).unwrap(),
                            );
                        } else {
                            near(
                                &c.vals(st.layers()[k].velocity().unwrap().ciphertext()),
                                &ls[k]
                                    .pack_weights_out_by_in(&v[k].mapv(|x| 0.08 * 1.9 * x))
                                    .unwrap(),
                            );
                        }
                    }
                    assert_eq!(
                        candidate.stage().layers()[k].weights().ciphertext().level(),
                        31
                    );
                    assert_eq!(
                        candidate.stage().layers()[k].weights().ciphertext().scale(),
                        c.p.scale()
                    );
                    assert!(candidate.stage().layers()[k].request(0.081, 1).is_err() || square);
                }
                for event in receipt.events() {
                    assert_eq!(event.counter_delta(), Some(&OpCounter::default()));
                    assert_eq!(event.output_level(), 31);
                    assert_eq!(event.output_scale(), c.p.scale());
                    assert!(event.timing().download.is_none());
                    assert!(event.timing().upload.is_none());
                }
                stage = candidate.into_stage();
            }
        }
    }
}

// Break caught: cross-transaction/role replay accepted, partial or extra completion.
#[test]
fn batch_identity_cardinality_and_output_metadata_are_checked() {
    let c = Ctx::new(true);
    let (stage, _, _) = c.stage(0, true);
    let contract = PackedRefreshContract::new(&c.ev, 31).unwrap();
    let a = PackedRefreshBatch::new(&c.ev, &stage, contract.clone()).unwrap();
    let b = PackedRefreshBatch::new(&c.ev, &stage, contract).unwrap();
    assert!(a.complete(&c.ev, responses(&c, &b)).is_err());
    let mut r = responses(&c, &a);
    r.swap(0, 2);
    assert!(a.complete(&c.ev, r).is_err());
    let mut r = responses(&c, &a);
    r[1] = r[0].clone();
    assert!(a.complete(&c.ev, r).is_err());
    let mut r = responses(&c, &a);
    r.pop();
    assert!(a.complete(&c.ev, r).is_err());
    let mut r = responses(&c, &a);
    r.push(r[0].clone());
    assert!(a.complete(&c.ev, r).is_err());
    let request = &a.requests()[0];
    let ct = request.ciphertext();
    assert!(
        PackedRefreshResponse::new(&c.ev, request, c.ev.mod_switch_to_level(ct, 30).unwrap())
            .is_err()
    );
    assert!(
        PackedRefreshResponse::new(
            &c.ev,
            request,
            Ciphertext::new(ct.polys().to_vec(), c.p.scale() * 2.0).unwrap()
        )
        .is_err()
    );
    let mut polys = ct.polys().to_vec();
    polys.push(polys[0].clone());
    assert!(
        PackedRefreshResponse::new(&c.ev, request, Ciphertext::new(polys, c.p.scale()).unwrap())
            .is_err()
    );
    let other = Ctx::new(true);
    assert!(
        PackedRefreshResponse::new(
            &other.ev,
            request,
            other.ct(&vec![0.0; other.p.slots()], 31)
        )
        .is_err()
    );
    assert!(PackedRefreshContract::new(&c.ev, 32).is_err());
    assert!(
        PackedRefreshBatch::new(
            &c.ev,
            &stage,
            PackedRefreshContract::new(&other.ev, 31).unwrap()
        )
        .is_err()
    );
    a.complete(&c.ev, responses(&c, &a)).unwrap();
}

// Break caught: free mod-switch replacing projection, wrong support, level0 recovery,
// extra canonical-hidden mask, or treating range obligations as verified magnitude.
#[test]
fn state_projection_and_canonical_hidden_have_distinct_costs_and_support() {
    for square in [false, true] {
        let c = Ctx::new(square);
        for s in 0..2 {
            let ls = &c.plan.layers()[2 * s..2 * s + 2];
            let ws = std::array::from_fn(|k| {
                PackedWeights::new(&c.ev, ls[k].clone(), c.ct(&vec![0.125; c.p.slots()], 1))
                    .unwrap()
            });
            let bs = square.then(|| {
                std::array::from_fn(|k| {
                    PackedVector::new(&c.ev, ls[k].output(), c.ct(&vec![0.125; c.p.slots()], 1))
                        .unwrap()
                })
            });
            let stage = PackedStage::new(
                &c.ev,
                &c.plan,
                s,
                ws,
                bs,
                if square {
                    PackedStageConfig::square_sgd()
                } else {
                    PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
                },
            )
            .unwrap();
            let batch = PackedRefreshBatch::new(
                &c.ev,
                &stage,
                PackedRefreshContract::new(&c.ev, 31).unwrap(),
            )
            .unwrap();
            for r in batch.requests() {
                let before = c.ev.counter();
                let boot = project_state(&c.ev, &c.e, r).unwrap();
                assert_eq!(
                    &c.ev.counter() - &before,
                    OpCounter {
                        ct_pt_mults: 1,
                        rescales: 1,
                        ..Default::default()
                    }
                );
                assert_eq!(boot.ciphertext().level(), 0);
                assert_eq!(boot.ciphertext().scale(), c.p.scale());
                assert_eq!(boot.required_all_slot_absolute_bound(), 1.0);
                let mask = match r.shape() {
                    PackedRefreshShape::Matrix(l) => l.matrix_mask(),
                    PackedRefreshShape::Vector(v) => v.mask(),
                };
                near(
                    &c.vals(boot.ciphertext()),
                    &mask.iter().map(|v| v * 0.125).collect::<Vec<_>>(),
                );
            }
            let low_ws = std::array::from_fn(|k| {
                PackedWeights::new(&c.ev, ls[k].clone(), c.ct(&vec![0.125; c.p.slots()], 0))
                    .unwrap()
            });
            let low_bs = square.then(|| {
                std::array::from_fn(|k| {
                    c.vec(
                        ls[k].output(),
                        &vec![0.125; ls[k].output().logical_len()],
                        0,
                    )
                })
            });
            let low = PackedStage::new(
                &c.ev,
                &c.plan,
                s,
                low_ws,
                low_bs,
                if square {
                    PackedStageConfig::square_sgd()
                } else {
                    PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
                },
            )
            .unwrap();
            let batch = PackedRefreshBatch::new(
                &c.ev,
                &low,
                PackedRefreshContract::new(&c.ev, 31).unwrap(),
            )
            .unwrap();
            let before = c.ev.counter();
            assert!(project_state(&c.ev, &c.e, &batch.requests()[0]).is_err());
            assert_eq!(c.ev.counter(), before);
            batch.complete(&c.ev, responses(&c, &batch)).unwrap(); // I can refresh level0
            let hv = c.vec(ls[0].output(), &vec![0.2; ls[0].output().logical_len()], 2);
            let natural = PackedVector::new(
                &c.ev,
                hv.spec(),
                Ciphertext::new(hv.ciphertext().polys().to_vec(), c.p.scale() * 2.0).unwrap(),
            )
            .unwrap();
            assert!(CanonicalHidden::new(&c.ev, &ls[0], 77, &natural).is_err());
            let hidden = CanonicalHidden::new(&c.ev, &ls[0], 77, &hv).unwrap();
            let r = hidden
                .request(&c.ev, PackedRefreshContract::new(&c.ev, 31).unwrap())
                .unwrap();
            assert_eq!(
                r.role(),
                PackedRefreshRole::Hidden {
                    stage: s,
                    record: 77
                }
            );
            let before = c.ev.counter();
            assert!(project_state(&c.ev, &c.e, &r).is_err());
            let boot = hidden.boot_input(&c.ev).unwrap();
            assert_eq!(c.ev.counter(), before);
            near(
                &c.vals(boot.ciphertext()),
                &ls[0]
                    .output()
                    .pack(&vec![0.2; ls[0].output().logical_len()])
                    .unwrap(),
            );
            assert!(CanonicalHidden::new(&c.ev, &ls[1], 77, &hv).is_err());
        }
    }
}

// Break caught: same-index refresh reuses generations, U presence changes, or failure
// mutates the old state before every client result exists.
#[test]
fn same_index_refresh_invalidates_bound_gradients_and_late_failures_are_atomic() {
    for square in [false, true] {
        let c = Ctx::new(square);
        let server = PackedInteractiveStage::new(&c.ev, &c.e).unwrap();
        let (stage, _, _) = c.stage(0, square);
        let records = c.records(0, 1, 0.0);
        let (first, _) = server
            .prepare_step(
                &stage,
                &records,
                0.04,
                GradRelin::Lazy,
                &mut Client::new(&c),
            )
            .unwrap();
        let stage = first.into_stage();
        let mut bound = vec![];
        for layer in stage.layers() {
            let l = layer.weights().layer();
            let req = layer.request(0.02, 1).unwrap();
            let x = c.vec(l.input(), &vec![0.1; l.input().logical_len()], 10);
            let dz = c.vec(l.output(), &vec![0.1; l.output().logical_len()], 10);
            bound.push(
                req.bind(
                    outer_sum_scaled_masks(
                        &c.ev,
                        &c.e,
                        l,
                        &[(&x, &dz)],
                        req.factors(),
                        GradRelin::Lazy,
                    )
                    .unwrap(),
                )
                .unwrap(),
            );
        }
        let batch = PackedRefreshBatch::new(
            &c.ev,
            &stage,
            PackedRefreshContract::new(&c.ev, 31).unwrap(),
        )
        .unwrap();
        // Includes persistent U in NAG, not merely the initially absent-U state.
        for request in batch.requests() {
            let before = c.ev.counter();
            let projected = project_state(&c.ev, &c.e, request).unwrap();
            assert_eq!(
                &c.ev.counter() - &before,
                OpCounter {
                    ct_pt_mults: 1,
                    rescales: 1,
                    ..Default::default()
                }
            );
            near(
                &c.vals(projected.ciphertext()),
                &c.vals(request.ciphertext()),
            );
        }
        let refreshed = batch.complete(&c.ev, responses(&c, &batch)).unwrap();
        for (k, gradient) in bound.iter().enumerate() {
            assert!(stage.layers()[k].prepare(&c.ev, gradient).is_ok());
            assert!(
                refreshed.stage().layers()[k]
                    .prepare(&c.ev, gradient)
                    .is_err()
            );
            assert_eq!(refreshed.stage().layers()[k].update_index(), 1);
            assert_eq!(refreshed.stage().layers()[k].velocity().is_some(), !square);
        }
        let old = snapshot(&c, &stage);
        for fail in [2, 3] {
            let mut client = Client::new(&c);
            client.fail_at = Some(fail);
            let before = c.ev.counter();
            let err = server
                .prepare_step(&stage, &records, 0.02, GradRelin::Lazy, &mut client)
                .unwrap_err();
            assert_eq!(err.phase(), PackedRefreshPhase::Client);
            assert_eq!(err.events().len(), fail);
            assert_eq!(err.counter_delta().unwrap(), &(&c.ev.counter() - &before));
            assert!(err.counter_delta().unwrap().ct_ct_tensor > 0);
            assert_eq!(snapshot(&c, &stage), old);
            for (k, gradient) in bound.iter().enumerate() {
                assert_eq!(stage.layers()[k].update_index(), 1);
                assert!(stage.layers()[k].prepare(&c.ev, gradient).is_ok());
            }
        }
        let (clean, _) = server
            .prepare_step(
                &stage,
                &records,
                0.02,
                GradRelin::Lazy,
                &mut Client::new(&c),
            )
            .unwrap();
        let (retry, _) = server
            .prepare_step(
                &stage,
                &records,
                0.02,
                GradRelin::Lazy,
                &mut Client::new(&c),
            )
            .unwrap();
        for (a, b) in snapshot(&c, clean.stage())
            .iter()
            .zip(snapshot(&c, retry.stage()))
        {
            near(a, &b);
        }
    }
}

// Real evaluator forwarding; only a specific late subtraction is fault-injected.
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

// Break caught: update failure emits success, starts client work, changes old state,
// or loses the partial operation delta after first-layer arithmetic has completed.
#[test]
fn late_backend_update_failure_keeps_original_state_and_receipt_counts() {
    let c = Ctx::new(true);
    let (stage, _, _) = c.stage(0, true);
    let records = c.records(0, 1, 0.0);
    let old = snapshot(&c, &stage);
    // loss, W0, b0, W1, b1: fail late in both the W and bias paths.
    for at in [4, 5] {
        let ev = FailSub {
            ev: HeOps::fork(&c.ev),
            at,
            subs: std::cell::Cell::new(0),
        };
        let server = PackedInteractiveStage::new(&ev, &c.e).unwrap();
        let mut client = Client::new(&c);
        let err = server
            .prepare_step(&stage, &records, 0.04, GradRelin::Lazy, &mut client)
            .unwrap_err();
        assert_eq!(err.phase(), PackedRefreshPhase::Update);
        assert_eq!(err.events().len(), 0);
        assert_eq!(err.counter_delta().unwrap(), &ev.counter());
        assert!(ev.counter().ct_ct_tensor > 0);
        assert_eq!(client.calls, 0);
        assert_eq!(snapshot(&c, &stage), old);
        assert_eq!(stage.layers()[0].update_index(), 0);
    }
    let server = PackedInteractiveStage::new(&c.ev, &c.e).unwrap();
    let (next, _) = server
        .prepare_step(
            &stage,
            &records,
            0.04,
            GradRelin::Lazy,
            &mut Client::new(&c),
        )
        .unwrap();
    assert_eq!(next.stage().layers()[0].update_index(), 1);
}

// A replaying transport uses real client ciphertexts but returns an earlier role last.
struct Replay<'a> {
    client: Client<'a>,
    first: Option<PackedRefreshResponse<Ciphertext>>,
}
impl PackedClientRefresh<Evaluator> for Replay<'_> {
    fn output_contract(&self) -> &PackedRefreshContract {
        &self.client.contract
    }
    fn refresh(
        &mut self,
        req: &PackedRefreshRequest<Ciphertext>,
    ) -> Result<(PackedRefreshResponse<Ciphertext>, ClientRefreshTiming)> {
        let call = self.client.calls;
        let (res, time) =
            <Client as PackedClientRefresh<Evaluator>>::refresh(&mut self.client, req)?;
        if call == 0 {
            self.first = Some(res.clone());
        }
        Ok((
            if call == 3 {
                self.first.as_ref().unwrap().clone()
            } else {
                res
            },
            time,
        ))
    }
}
#[test]
fn late_response_replay_reports_role_and_completed_events_without_commit() {
    for square in [false, true] {
        let c = Ctx::new(square);
        let (stage, _, _) = c.stage(0, square);
        let old = snapshot(&c, &stage);
        let server = PackedInteractiveStage::new(&c.ev, &c.e).unwrap();
        let mut client = Replay {
            client: Client::new(&c),
            first: None,
        };
        let err = server
            .prepare_step(
                &stage,
                &c.records(0, 1, 0.0),
                0.04,
                GradRelin::Lazy,
                &mut client,
            )
            .unwrap_err();
        assert_eq!(err.phase(), PackedRefreshPhase::Response);
        assert_eq!(
            err.role(),
            Some(if square {
                PackedRefreshRole::Bias(1)
            } else {
                PackedRefreshRole::Velocity(1)
            })
        );
        assert_eq!(err.events().len(), 3);
        assert_eq!(snapshot(&c, &stage), old);
    }
}

// Break caught: client selects one redundant copy instead of averaging, clips active
// magnitudes, preserves structural garbage, or removes trained padded-class channels.
#[test]
fn client_restores_structural_zeros_averages_copies_and_never_clips() {
    let c = Ctx::new(true);
    for s in 0..2 {
        let ls = &c.plan.layers()[2 * s..2 * s + 2];
        let weights = std::array::from_fn(|k| {
            PackedWeights::new(&c.ev, ls[k].clone(), c.ct(&vec![1.25; c.p.slots()], 0)).unwrap()
        });
        let biases = std::array::from_fn(|k| {
            let spec = ls[k].output();
            // Each logical value has repeated copies 1.25 + replica*0.01.
            let values = (0..c.p.slots())
                .map(|slot| {
                    let replica = match spec.repr() {
                        VectorRepr::Expanded => slot % spec.layout().c(),
                        VectorRepr::Repeated => slot / spec.layout().c(),
                    };
                    1.25 + 0.01 * replica as f64
                })
                .collect::<Vec<_>>();
            PackedVector::new(&c.ev, spec, c.ct(&values, 0)).unwrap()
        });
        let stage = PackedStage::new(
            &c.ev,
            &c.plan,
            s,
            weights,
            Some(biases),
            PackedStageConfig::square_sgd(),
        )
        .unwrap();
        let batch = PackedRefreshBatch::new(
            &c.ev,
            &stage,
            PackedRefreshContract::new(&c.ev, 31).unwrap(),
        )
        .unwrap();
        let next = batch.complete(&c.ev, responses(&c, &batch)).unwrap();
        for (k, layer) in ls.iter().enumerate() {
            near(
                &c.vals(next.stage().layers()[k].weights().ciphertext()),
                &ls[k]
                    .matrix_mask()
                    .iter()
                    .map(|m| 1.25 * m)
                    .collect::<Vec<_>>(),
            );
            let spec = layer.output();
            let repeats = match spec.repr() {
                VectorRepr::Expanded => spec.layout().c(),
                VectorRepr::Repeated => spec.layout().r(),
            };
            let mean = 1.25 + 0.005 * (repeats - 1) as f64;
            near(
                &c.vals(next.stage().biases().unwrap()[k].ciphertext()),
                &spec.pack(&vec![mean; spec.logical_len()]).unwrap(),
            );
        }
    }
}
