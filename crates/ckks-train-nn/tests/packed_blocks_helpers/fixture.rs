//! Independent scalar model and literal slot maps for the ratified original fixture.
#![allow(dead_code)]
use super::*;
use ckks_train_ckks::metadata::CtMeta;
use ckks_train_nn::layout_reboot::GradRelin;
pub type V = [f64; 3];
pub type M = [V; 3];
pub const W: [M; 2] = [
    [
        [5. / 16., 1. / 16., 1. / 16.],
        [1. / 16., 5. / 16., 1. / 16.],
        [1. / 16., 1. / 8., 1. / 4.],
    ],
    [
        [5. / 16., 1. / 8., -1. / 16.],
        [1. / 16., 5. / 16., 1. / 16.],
        [-1. / 16., 1. / 16., 1. / 4.],
    ],
];
pub const X: [V; 4] = [
    [3. / 4., 3. / 8., 1. / 2.],
    [3. / 8., 3. / 4., 1. / 2.],
    [1. / 2., 3. / 8., 3. / 4.],
    [-1. / 2., 1. / 2., -1. / 2.],
];
pub const Y: [V; 4] = [
    [3. / 4., 0., 0.],
    [0., 3. / 4., 0.],
    [3. / 8., 3. / 8., 0.],
    [7. / 8., -7. / 8., 7. / 8.],
];
pub const IDS: [&[usize]; 4] = [&[0, 1, 2], &[0], &[0, 2], &[0, 1, 2]];
pub const RATES: [f64; 4] = [0.04, 0.01, 0.02, 0.08];
pub const E: f64 = 1. / 4096.;
pub const EPS: f64 = 1. / 1048576.;
pub fn seed(domain: u8, g: usize, mode: u8, t: u8, s: u8, r: u8, role: u8) -> [u8; 32] {
    let mut out = [0; 32];
    out[..8].copy_from_slice(&[195, domain, g as u8, mode, t, s, r, role]);
    out
}
pub fn client_seed(g: usize, call: u64) -> [u8; 32] {
    let mut out = [0; 32];
    out[..4].copy_from_slice(&[195, 2, g as u8, 0]);
    out[4..12].copy_from_slice(&call.to_le_bytes());
    out
}
pub fn vector(v: &V, expanded: bool) -> Vec<f64> {
    (0..512)
        .map(|s| {
            let k = if expanded { s / 4 } else { s % 4 };
            if k < 3 { v[k] } else { 0. }
        })
        .collect()
}
pub fn matrix(m: &M, row: bool) -> Vec<f64> {
    (0..512)
        .map(|s| {
            let (i, j) = (s / 4, s % 4);
            if i < 3 && j < 3 {
                if row { m[j][i] } else { m[i][j] }
            } else {
                0.
            }
        })
        .collect()
}
pub fn mv(w: &M, x: &V) -> V {
    std::array::from_fn(|o| (0..3).map(|i| w[o][i] * x[i]).sum())
}
#[derive(Clone, Debug)]
pub struct ScalarStage {
    pub w: [M; 2],
    pub v: [M; 2],
}
impl ScalarStage {
    pub fn new() -> Self {
        Self {
            w: W,
            v: [[[0.; 3]; 3]; 2],
        }
    }
    pub fn u(&self, k: usize) -> M {
        self.v[k].map(|row| row.map(|x| 0.08 * 1.9 * x))
    }
}
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub input: Vec<V>,
    pub z: Vec<V>,
    pub hidden: Vec<V>,
    pub output: Vec<V>,
    pub grad: [M; 2],
}
pub fn scalar_step(states: &mut [ScalarStage], t: usize, targets: &[Vec<V>]) -> Vec<Snapshot> {
    let ids = IDS[t];
    let mut input = ids.iter().map(|&i| X[i]).collect::<Vec<_>>();
    let mut snapshots = vec![];
    for (s, state) in states.iter_mut().enumerate() {
        let mut z = vec![];
        let mut hidden = vec![];
        let mut outputs = vec![];
        let mut grad = [[[0.; 3]; 3]; 2];
        for (j, &id) in ids.iter().enumerate() {
            let x = input[j];
            let z1 = mv(&state.w[0], &x);
            let h = z1.map(|z| z * z + z);
            let y = mv(&state.w[1], &h);
            let d2 = std::array::from_fn::<_, 3, _>(|o| y[o] - targets[s][id][o]);
            let d1 = std::array::from_fn::<_, 3, _>(|i| {
                (0..3).map(|o| state.w[1][o][i] * d2[o]).sum::<f64>() * (2. * z1[i] + 1.)
            });
            for (k, gk) in grad.iter_mut().enumerate() {
                let (a, d) = if k == 0 { (&x, &d1) } else { (&h, &d2) };
                for (o, &dz) in d.iter().enumerate() {
                    for (i, &ai) in a.iter().enumerate() {
                        gk[o][i] += dz * ai / ids.len() as f64;
                    }
                }
            }
            z.push(z1);
            hidden.push(h);
            outputs.push(y);
        }
        for (k, gk) in grad.iter().enumerate() {
            for (o, grow) in gk.iter().enumerate() {
                for (i, &value) in grow.iter().enumerate() {
                    state.v[k][o][i] = 0.9 * state.v[k][o][i] + value;
                    state.w[k][o][i] -= RATES[t] * (value + 0.9 * state.v[k][o][i]);
                }
            }
        }
        snapshots.push(Snapshot {
            input: input.clone(),
            z,
            hidden: hidden.clone(),
            output: outputs,
            grad,
        });
        input = hidden;
    }
    snapshots
}
pub fn scalar_forward(states: &[ScalarStage]) -> Vec<Snapshot> {
    let mut input = X[..3].to_vec();
    let mut out = vec![];
    for (s, state) in states.iter().enumerate() {
        let z = input.iter().map(|x| mv(&state.w[0], x)).collect::<Vec<_>>();
        let hidden = z.iter().map(|z| z.map(|v| v * v + v)).collect::<Vec<_>>();
        let output = if s + 1 == states.len() {
            hidden.iter().map(|h| mv(&state.w[1], h)).collect()
        } else {
            hidden.clone()
        };
        out.push(Snapshot {
            input,
            z,
            hidden: hidden.clone(),
            output,
            grad: [[[0.; 3]; 3]; 2],
        });
        input = hidden;
    }
    out
}
pub fn meta_batch(p: &Arc<CkksParams>, g: &PackedGeometryPlan, t: usize) -> PackedGraphBatchMeta {
    let h = CtMeta::new(p, 16, p.scale()).unwrap();
    let ids = IDS[t];
    PackedGraphBatchMeta::new(
        p.clone(),
        g,
        t as u64,
        ids.to_vec(),
        vec![h; ids.len()],
        vec![vec![h; ids.len()]; g.stages().len()],
        vec![
            PackedPartition::new(0, 2, ids.iter().filter(|&&i| i < 2).count()),
            PackedPartition::new(2, 4, ids.iter().filter(|&&i| i >= 2).count()),
        ],
    )
    .unwrap()
}
pub fn next(p: &Arc<CkksParams>, g: &PackedGeometryPlan, t: usize) -> PackedGraphNext {
    if t == 3 {
        PackedGraphNext::Terminal
    } else {
        PackedGraphNext::Known(
            meta_batch(p, g, t + 1),
            PackedUpdateMeta::new(RATES[t + 1], GradRelin::Lazy, true).unwrap(),
        )
    }
}
#[derive(Clone, Debug)]
pub struct Schedule {
    pub g: usize,
    pub placement: PackedNiPlacement,
    pub updates: Vec<PackedGraphNiPlan>,
    pub inference: PackedGraphForwardPlan,
    pub total: OpCounter,
}
pub fn schedules(p: &Arc<CkksParams>, enc: &Encoder, c: &PackedBootstrapContract) -> Vec<Schedule> {
    let mut out = vec![];
    for n in [2, 3] {
        let g = geometry(n);
        for placement in [
            PackedNiPlacement::Au,
            PackedNiPlacement::Wn,
            PackedNiPlacement::Aa,
        ] {
            let h = CtMeta::new(p, 16, p.scale()).unwrap();
            let mut states = (0..n)
                .map(|s| {
                    PackedStageMeta::new(
                        p.clone(),
                        &g,
                        s,
                        PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
                        [h; 2],
                        [None; 2],
                        None,
                        [0; 2],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let mut updates = vec![];
            let mut total = OpCounter::default();
            for (t, &rate) in RATES.iter().enumerate() {
                let plan = PackedGraphNiPlan::new(
                    enc,
                    &g,
                    &states,
                    &meta_batch(p, &g, t),
                    PackedUpdateMeta::new(rate, GradRelin::Lazy, true).unwrap(),
                    next(p, &g, t),
                    placement,
                    c,
                )
                .unwrap();
                states = plan.committed();
                total += plan.predicted_counts();
                updates.push(plan);
            }
            // These tags have no encrypted payload. Only final committed W histories are used.
            let ev = Tags::new(p.clone());
            let stages = states
                .iter()
                .enumerate()
                .map(|(s, m)| {
                    PackedStage::new(
                        &ev,
                        &g,
                        s,
                        std::array::from_fn(|k| {
                            PackedWeights::new(
                                &ev,
                                g.layers()[2 * s + k].clone(),
                                ev.ct(m.weights()[k].level(), m.weights()[k].scale()),
                            )
                            .unwrap()
                        }),
                        None,
                        PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
                    )
                    .unwrap()
                })
                .collect();
            let graph = PackedBlockGraph::new(&ev, g.clone(), stages).unwrap();
            let records = (0..3)
                .map(|id| {
                    PackedForwardRecord::new(
                        id,
                        PackedVector::new(&ev, g.layers()[0].input(), ev.ct(16, p.scale()))
                            .unwrap(),
                        true,
                    )
                })
                .collect();
            let batch =
                PackedForwardBatch::new(&ev, &graph, records, vec![PackedPartition::new(0, 3, 3)])
                    .unwrap();
            let boot = boot(c.clone());
            let server = PackedBlockNiTrainer::new(&ev, enc, &boot, placement).unwrap();
            let inference = server
                .plan_forward(
                    &graph,
                    &batch,
                    if placement == PackedNiPlacement::Aa {
                        PackedInferencePlan::AllHiddenBootstrap
                    } else {
                        PackedInferencePlan::BoundaryBootstrap
                    },
                )
                .unwrap();
            total += inference.predicted_counts();
            assert_eq!(ev.counter(), OpCounter::default());
            out.push(Schedule {
                g: n,
                placement,
                updates,
                inference,
                total,
            });
        }
    }
    out
}

// Frozen from exact-Q/Encoder metadata before any block key or HE execution.
pub const CASE_COUNTS: [OpCounter; 6] = [
    OpCounter {
        rotations: 1979,
        relins: 1881,
        ct_ct_tensor: 2165,
        ct_pt_mults: 9339,
        rescales: 2816,
        sse_switches: 88,
        bootstraps: 44,
    },
    OpCounter {
        rotations: 1979,
        relins: 1881,
        ct_ct_tensor: 2165,
        ct_pt_mults: 9339,
        rescales: 2816,
        sse_switches: 88,
        bootstraps: 44,
    },
    OpCounter {
        rotations: 1831,
        relins: 1721,
        ct_ct_tensor: 1981,
        ct_pt_mults: 8495,
        rescales: 2576,
        sse_switches: 80,
        bootstraps: 40,
    },
    OpCounter {
        rotations: 3189,
        relins: 3060,
        ct_ct_tensor: 3522,
        ct_pt_mults: 15249,
        rescales: 4557,
        sse_switches: 144,
        bootstraps: 72,
    },
    OpCounter {
        rotations: 3189,
        relins: 3060,
        ct_ct_tensor: 3522,
        ct_pt_mults: 15249,
        rescales: 4557,
        sse_switches: 144,
        bootstraps: 72,
    },
    OpCounter {
        rotations: 2745,
        relins: 2580,
        ct_ct_tensor: 2970,
        ct_pt_mults: 12741,
        rescales: 3861,
        sse_switches: 120,
        bootstraps: 60,
    },
];
pub fn interactive_schedule(
    p: &Arc<CkksParams>,
    enc: &Encoder,
    g: usize,
) -> Vec<Vec<PackedDagPlan>> {
    let geometry = geometry(g);
    let h = CtMeta::new(p, 16, p.scale()).unwrap();
    let mut out = vec![];
    let mut states = (0..g)
        .map(|s| {
            PackedStageMeta::new(
                p.clone(),
                &geometry,
                s,
                PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
                [h; 2],
                [None; 2],
                None,
                [0; 2],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    for (t, &rate) in RATES.iter().enumerate() {
        let mut plans = vec![];
        for (s, state) in states.iter_mut().enumerate() {
            let batch = PackedBatchMeta::new(
                p.clone(),
                t as u64,
                geometry.layers()[2 * s].input(),
                geometry.layers()[2 * s + 1].output(),
                IDS[t]
                    .iter()
                    .map(|&id| PackedRecordMeta::new(id, h, h))
                    .collect(),
            )
            .unwrap();
            let plan = plan_stage_dag(
                enc,
                state,
                &batch,
                PackedUpdateMeta::new(rate, GradRelin::Lazy, true).unwrap(),
                None,
            )
            .unwrap();
            *state = PackedStageMeta::new(
                p.clone(),
                &geometry,
                s,
                PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
                [h; 2],
                [Some(h); 2],
                None,
                plan.candidate().update_indices(),
            )
            .unwrap();
            plans.push(plan);
        }
        out.push(plans);
    }
    out
}
pub fn interactive_forward_schedule(
    p: &Arc<CkksParams>,
    enc: &Encoder,
    g: usize,
) -> Vec<PackedForwardPlan> {
    let ev = Tags::new(p.clone());
    let graph = super::graph(&ev, g);
    let plans = graph
        .stages()
        .iter()
        .enumerate()
        .map(|(s, stage)| {
            let records = (0..3)
                .map(|id| {
                    PackedForwardRecord::new(
                        id,
                        PackedVector::new(
                            &ev,
                            graph.geometry().layers()[2 * s].input(),
                            ev.ct(16, p.scale()),
                        )
                        .unwrap(),
                        true,
                    )
                })
                .collect::<Vec<_>>();
            stage
                .plan_forward(
                    &ev,
                    enc,
                    &records,
                    if s + 1 == g {
                        PackedForwardPart::OutputStack
                    } else {
                        PackedForwardPart::Backbone
                    },
                    None,
                )
                .unwrap()
        })
        .collect();
    assert_eq!(ev.counter(), OpCounter::default());
    plans
}

// A second, captured-forward oracle: capture every old stage before any update,
// then apply independent local gradients in reverse order. Never calls scalar_step.
pub fn capture_local(state: &ScalarStage, input: &[V], targets: &[V]) -> Snapshot {
    let z = input.iter().map(|x| mv(&state.w[0], x)).collect::<Vec<_>>();
    let hidden = z.iter().map(|z| z.map(|v| v * v + v)).collect::<Vec<_>>();
    let output = hidden
        .iter()
        .map(|h| mv(&state.w[1], h))
        .collect::<Vec<_>>();
    let residual = output
        .iter()
        .zip(targets)
        .map(|(y, t)| std::array::from_fn::<_, 3, _>(|o| y[o] - t[o]))
        .collect::<Vec<_>>();
    let upstream = residual
        .iter()
        .zip(&z)
        .map(|(d, z)| {
            std::array::from_fn::<_, 3, _>(|i| {
                let mut sum = 0.;
                for (o, &delta) in d.iter().enumerate() {
                    sum += state.w[1][o][i] * delta;
                }
                sum * (2. * z[i] + 1.)
            })
        })
        .collect::<Vec<_>>();
    let mut grad = [[[0.; 3]; 3]; 2];
    for (k, gk) in grad.iter_mut().enumerate() {
        let (xs, ds) = if k == 0 {
            (input, &upstream)
        } else {
            (hidden.as_slice(), &residual)
        };
        for (o, row) in gk.iter_mut().enumerate() {
            for (i, g) in row.iter_mut().enumerate() {
                for (x, d) in xs.iter().zip(ds) {
                    *g += d[o] * x[i] / input.len() as f64;
                }
            }
        }
    }
    Snapshot {
        input: input.to_vec(),
        z,
        hidden,
        output,
        grad,
    }
}
pub fn reverse_step(
    old: &[ScalarStage],
    t: usize,
    targets: &[Vec<V>],
) -> (Vec<ScalarStage>, Vec<Snapshot>) {
    let mut input = IDS[t].iter().map(|&id| X[id]).collect::<Vec<_>>();
    let mut snapshots = vec![];
    for (s, state) in old.iter().enumerate() {
        let target = IDS[t].iter().map(|&id| targets[s][id]).collect::<Vec<_>>();
        let snapshot = capture_local(state, &input, &target);
        input = snapshot.hidden.clone();
        snapshots.push(snapshot);
    }
    let mut candidate = old.to_vec();
    for s in (0..old.len()).rev() {
        for k in 0..2 {
            for o in 0..3 {
                for i in 0..3 {
                    let g = snapshots[s].grad[k][o][i];
                    let velocity = 0.9 * old[s].v[k][o][i] + g;
                    candidate[s].v[k][o][i] = velocity;
                    candidate[s].w[k][o][i] = old[s].w[k][o][i] - RATES[t] * (g + 0.9 * velocity);
                }
            }
        }
    }
    (candidate, snapshots)
}
