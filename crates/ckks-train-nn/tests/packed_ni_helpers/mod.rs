//! Private independent scalar/packing fixtures and registered public metadata schedule.
#![allow(dead_code)]
use ckks_train_ckks::{CkksParams, Encoder, metadata::CtMeta};
use ckks_train_nn::{layout_reboot::GradRelin, packed_training::*};
use std::sync::Arc;
pub const W: [[[f64; 3]; 3]; 2] = [
    [
        [0.25, -0.125, 0.0625],
        [-0.0625, 0.1875, 0.125],
        [0.125, 0.0625, -0.1875],
    ],
    [
        [0.1875, -0.125, 0.0625],
        [0.0625, 0.25, -0.125],
        [-0.125, 0.0625, 0.1875],
    ],
];
pub const B: [[f64; 3]; 2] = [[0.25, 0.1875, -0.125], [0.25, -0.1875, 0.125]];
pub const X: [[f64; 3]; 4] = [
    [0.75, -0.375, 0.375],
    [-0.375, 0.75, 0.1875],
    [0.375, 0.1875, -0.75],
    [0.5, -0.5, 0.5],
];
pub const Y: [[f64; 3]; 4] = [
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.25, 0.25, 0.0],
    [0.875, -0.875, 0.875],
];
pub const VALID: [[bool; 4]; 4] = [
    [true, true, true, false],
    [true, false, false, false],
    [true, false, true, false],
    [true, true, true, false],
];
pub const RATE: [f64; 4] = [0.04, 0.01, 0.02, 0.08];
pub const E: f64 = 1.0 / 4096.0;
pub const EPS: f64 = 1.0 / 1048576.0;
pub fn geometry(square: bool) -> PackedGeometryPlan {
    let classes = if square { 3 } else { 2 };
    PackedGeometryPlan::new(
        3,
        &[
            StageShape::new(3, 3, classes).unwrap(),
            StageShape::new(3, 3, classes).unwrap(),
        ],
        512,
    )
    .unwrap()
}
pub fn config(square: bool) -> PackedStageConfig {
    if square {
        PackedStageConfig::square_sgd()
    } else {
        PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap()
    }
}
// Independent literal s=4*i+j packing; no production pack/mask helper call.
pub fn matrix(m: &[[f64; 3]; 3], row: bool) -> Vec<f64> {
    (0..512)
        .map(|s| {
            let (i, j) = (s / 4, s % 4);
            if i < 3 && j < 3 {
                if row { m[j][i] } else { m[i][j] }
            } else {
                0.0
            }
        })
        .collect()
}
pub fn vector(v: &[f64; 3], expanded: bool) -> Vec<f64> {
    (0..512)
        .map(|s| {
            let i = if expanded { s / 4 } else { s % 4 };
            if i < 3 { v[i] } else { 0.0 }
        })
        .collect()
}
#[derive(Clone, Debug)]
pub struct Scalar {
    pub w: [[[f64; 3]; 3]; 2],
    pub b: [[f64; 3]; 2],
    pub v: [[[f64; 3]; 3]; 2],
    pub square: bool,
}
impl Scalar {
    pub fn new(square: bool) -> Self {
        Self {
            w: W,
            b: if square { B } else { [[0.0; 3]; 2] },
            v: [[[0.0; 3]; 3]; 2],
            square,
        }
    }
    pub fn u(&self, k: usize) -> [[f64; 3]; 3] {
        self.v[k].map(|row| row.map(|v| 0.08 * (1.0 + 0.9) * v))
    }
    pub fn step(&mut self, t: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let n = VALID[t].iter().filter(|&&x| x).count() as f64;
        let mut g = [[[0.0; 3]; 3]; 2];
        let mut gb = [[0.0; 3]; 2];
        let mut hidden = vec![];
        let mut outputs = vec![];
        for record in 0..4 {
            if !VALID[t][record] {
                continue;
            }
            let x = X[record];
            let mut z1 = self.b[0];
            let mut z2 = self.b[1];
            for (o, z) in z1.iter_mut().enumerate() {
                for (i, &xi) in x.iter().enumerate() {
                    *z += self.w[0][o][i] * xi;
                }
            }
            let h = z1.map(|z| if self.square { z * z } else { z * z + z });
            for (o, z) in z2.iter_mut().enumerate() {
                for (i, &hi) in h.iter().enumerate() {
                    *z += self.w[1][o][i] * hi;
                }
            }
            let y = z2.map(|z| if self.square { z * z } else { z });
            let d2: [f64; 3] = std::array::from_fn(|o| {
                if self.square {
                    (y[o] - Y[record][o]) * (2.0 / 3.0) * (2.0 * z2[o])
                } else {
                    y[o] - Y[record][o]
                }
            });
            let mut d1 = [0.0; 3];
            for (i, di) in d1.iter_mut().enumerate() {
                for (o, &d) in d2.iter().enumerate() {
                    *di += self.w[1][o][i] * d;
                }
                *di *= if self.square {
                    2.0 * z1[i]
                } else {
                    2.0 * z1[i] + 1.0
                };
            }
            for k in 0..2 {
                let (a, d) = if k == 0 { (&x, &d1) } else { (&h, &d2) };
                for (o, &dz) in d.iter().enumerate() {
                    gb[k][o] += dz / n;
                    for (i, &ai) in a.iter().enumerate() {
                        g[k][o][i] += dz * ai / n;
                    }
                }
            }
            hidden.push(h);
            outputs.push(y);
        }
        for k in 0..2 {
            for o in 0..3 {
                for i in 0..3 {
                    if self.square {
                        self.w[k][o][i] -= RATE[t] * g[k][o][i];
                    } else {
                        self.v[k][o][i] = 0.9 * self.v[k][o][i] + g[k][o][i];
                        self.w[k][o][i] -= RATE[t] * (g[k][o][i] + 0.9 * self.v[k][o][i]);
                    }
                }
                if self.square {
                    self.b[k][o] -= RATE[t] * gb[k][o];
                }
            }
        }
        (hidden, outputs)
    }
}
pub fn batch(
    p: &Arc<CkksParams>,
    g: &PackedGeometryPlan,
    ordinal: usize,
    t: usize,
) -> PackedBatchMeta {
    let h = CtMeta::new(p, 16, p.scale()).unwrap();
    PackedBatchMeta::new(
        p.clone(),
        t as u64,
        g.layers()[2 * ordinal].input(),
        g.layers()[2 * ordinal + 1].output(),
        VALID[t]
            .iter()
            .enumerate()
            .filter(|(_, v)| **v)
            .map(|(id, _)| PackedRecordMeta::new(id, h, h))
            .collect(),
    )
    .unwrap()
}
pub fn next(
    p: &Arc<CkksParams>,
    g: &PackedGeometryPlan,
    ordinal: usize,
    t: usize,
) -> PackedNextStep {
    if t == 3 {
        PackedNextStep::Terminal
    } else {
        PackedNextStep::Known(
            batch(p, g, ordinal, t + 1),
            PackedUpdateMeta::new(RATE[t + 1], GradRelin::Lazy, true).unwrap(),
        )
    }
}
pub fn schedule(
    p: Arc<CkksParams>,
    enc: &Encoder,
    contract: &PackedBootstrapContract,
    square: bool,
    ordinal: usize,
    placement: PackedNiPlacement,
) -> Vec<PackedNiPlan> {
    let g = geometry(square);
    let h = CtMeta::new(&p, 16, p.scale()).unwrap();
    let mut state = PackedStageMeta::new(
        p.clone(),
        &g,
        ordinal,
        config(square),
        [h; 2],
        [None; 2],
        square.then_some([h; 2]),
        [0; 2],
    )
    .unwrap();
    let mut result = vec![];
    for (t, &rate) in RATE.iter().enumerate() {
        // Host Encoder checks the literal ingestion values at the registered level.
        for r in 0..4 {
            enc.encode(&vector(&X[r], ordinal == 0), p.scale(), 16)
                .unwrap();
            enc.encode(&vector(&Y[r], ordinal == 0), p.scale(), 16)
                .unwrap();
        }
        let plan = PackedNiPlan::new(
            enc,
            &state,
            &batch(&p, &g, ordinal, t),
            PackedUpdateMeta::new(rate, GradRelin::Lazy, true).unwrap(),
            next(&p, &g, ordinal, t),
            placement,
            contract,
        )
        .unwrap();
        state = plan.committed().clone();
        result.push(plan);
    }
    result
}
