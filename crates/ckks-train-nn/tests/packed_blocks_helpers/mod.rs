//! Test-only metadata/tag evaluator. No ciphertext polynomials, keys or numeric HE.
#![allow(dead_code)]
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, CkksParams, DiagonalEntry, Diagonals, Encoder, EvalConfig,
    HeCiphertext, HeOps, OpCounter, Plaintext,
};
use ckks_train_core::error::{HekitError, Result};
use ckks_train_nn::packed_training::*;
use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};
#[derive(Clone, Debug)]
pub struct TagCt {
    pub params: Arc<CkksParams>,
    pub level: usize,
    pub scale: f64,
    pub size: usize,
    pub tag: usize,
    pub forbid_reads: Arc<std::sync::atomic::AtomicBool>,
}
impl DiagonalEntry for TagCt {
    fn level(&self) -> usize {
        assert!(
            !self.forbid_reads.load(std::sync::atomic::Ordering::Relaxed),
            "unused inference ciphertext metadata inspected"
        );
        self.level
    }
    fn scale(&self) -> f64 {
        assert!(
            !self.forbid_reads.load(std::sync::atomic::Ordering::Relaxed),
            "unused inference ciphertext metadata inspected"
        );
        self.scale
    }
    fn slots(&self) -> usize {
        self.params.slots()
    }
}
impl HeCiphertext for TagCt {
    fn size(&self) -> usize {
        self.size
    }
}
pub struct Tags {
    pub params: Arc<CkksParams>,
    pub cfg: EvalConfig,
    pub counts: RefCell<OpCounter>,
    pub next: Cell<usize>,
    pub muls: RefCell<Vec<(usize, usize)>>,
    pub missing: Cell<Option<i32>>,
    pub fail_tensor: Cell<Option<u64>>,
    pub extra_count_tensor: Cell<Option<u64>>,
}
impl Tags {
    pub fn new(p: Arc<CkksParams>) -> Self {
        Self {
            params: p,
            cfg: EvalConfig::default(),
            counts: RefCell::new(OpCounter::default()),
            next: Cell::new(0),
            muls: RefCell::new(vec![]),
            missing: Cell::new(None),
            fail_tensor: Cell::new(None),
            extra_count_tensor: Cell::new(None),
        }
    }
    pub fn ct(&self, level: usize, scale: f64) -> TagCt {
        let tag = self.next.get();
        self.next.set(tag + 1);
        TagCt {
            params: self.params.clone(),
            level,
            scale,
            size: 2,
            tag,
            forbid_reads: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    fn output(&self, a: &TagCt, scale: f64, size: usize) -> TagCt {
        let mut x = self.ct(a.level, scale);
        x.size = size;
        x
    }
}
fn unsupported<T>() -> Result<T> {
    Err(HekitError::InvalidParams(
        "test metadata evaluator has no HE transport/matvec".into(),
    ))
}
impl HeOps for Tags {
    type Ct = TagCt;
    type Pt = Plaintext;
    fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }
    fn config(&self) -> &EvalConfig {
        &self.cfg
    }
    fn counter(&self) -> OpCounter {
        self.counts.borrow().clone()
    }
    fn reset_counter(&self) {
        *self.counts.borrow_mut() = OpCounter::default();
    }
    fn add_counts(&self, c: &OpCounter) {
        *self.counts.borrow_mut() += c;
    }
    fn fork(&self) -> Self {
        panic!("production must not fork metadata planning")
    }
    fn has_relin_key(&self) -> bool {
        true
    }
    fn has_galois_key(&self, s: i32) -> bool {
        self.missing.get() != Some(s)
    }
    fn owns(&self, c: &TagCt) -> bool {
        Arc::ptr_eq(&self.params, &c.params)
    }
    fn upload_ct(&self, _: &Ciphertext) -> Result<TagCt> {
        unsupported()
    }
    fn download_ct(&self, _: &TagCt) -> Result<Ciphertext> {
        unsupported()
    }
    fn upload_pt(&self, p: &Plaintext) -> Result<Plaintext> {
        Ok(p.clone())
    }
    fn add(&self, a: &TagCt, b: &TagCt) -> Result<TagCt> {
        assert_eq!(a.level, b.level);
        assert_eq!(a.scale.to_bits(), b.scale.to_bits());
        Ok(self.output(a, a.scale, a.size))
    }
    fn sub(&self, a: &TagCt, b: &TagCt) -> Result<TagCt> {
        self.add(a, b)
    }
    fn negate(&self, a: &TagCt) -> Result<TagCt> {
        Ok(self.output(a, a.scale, a.size))
    }
    fn add_plain(&self, a: &TagCt, p: &Plaintext) -> Result<TagCt> {
        assert_eq!(a.level, p.level());
        assert_eq!(a.scale.to_bits(), p.scale().to_bits());
        Ok(self.output(a, a.scale, a.size))
    }
    fn mul_plain(&self, a: &TagCt, p: &Plaintext) -> Result<TagCt> {
        self.counts.borrow_mut().ct_pt_mults += 1;
        Ok(self.output(a, a.scale * p.scale(), a.size))
    }
    fn mul_scalar(&self, a: &TagCt, _v: f64, s: f64) -> Result<TagCt> {
        self.counts.borrow_mut().ct_pt_mults += 1;
        Ok(self.output(a, self.exact_product_scale(a.level, s)?, a.size))
    }
    fn mul_plain_to_scale(&self, a: &TagCt, _p: &Plaintext, s: f64) -> Result<TagCt> {
        self.counts.borrow_mut().ct_pt_mults += 1;
        Ok(self.output(a, self.exact_product_scale(a.level, s)?, a.size))
    }
    fn mul_no_relin(&self, a: &TagCt, b: &TagCt) -> Result<TagCt> {
        assert_eq!(a.level, b.level);
        self.muls.borrow_mut().push((a.tag, b.tag));
        self.counts.borrow_mut().ct_ct_tensor += 1;
        if self.extra_count_tensor.get() == Some(self.counter().ct_ct_tensor) {
            self.counts.borrow_mut().sse_switches += 1;
        }
        if self.fail_tensor.get() == Some(self.counter().ct_ct_tensor) {
            return Err(HekitError::InvalidParams("injected tensor failure".into()));
        }
        Ok(self.output(a, a.scale * b.scale, 3))
    }
    fn relinearize(&self, a: &TagCt) -> Result<TagCt> {
        self.counts.borrow_mut().relins += 1;
        Ok(self.output(a, a.scale, 2))
    }
    fn mul(&self, a: &TagCt, b: &TagCt) -> Result<TagCt> {
        self.relinearize(&self.mul_no_relin(a, b)?)
    }
    fn rescale(&self, a: &TagCt) -> Result<TagCt> {
        if a.level == 0 {
            return Err(HekitError::NoLevelsLeft);
        }
        self.counts.borrow_mut().rescales += 1;
        let mut x = self.output(a, a.scale / self.params.q()[a.level] as f64, a.size);
        x.level -= 1;
        Ok(x)
    }
    fn mod_switch_to_level(&self, a: &TagCt, l: usize) -> Result<TagCt> {
        if l > a.level {
            return Err(HekitError::LevelMismatch);
        }
        let mut x = a.clone();
        x.level = l;
        Ok(x)
    }
    fn relabel(&self, mut a: TagCt, s: f64) -> Result<TagCt> {
        a.scale = s;
        Ok(a)
    }
    fn rotate(&self, a: &TagCt, _s: i32) -> Result<TagCt> {
        self.counts.borrow_mut().rotations += 1;
        Ok(self.output(a, a.scale, a.size))
    }
    fn conjugate(&self, _: &TagCt) -> Result<TagCt> {
        unsupported()
    }
    fn hoisted_rotate(&self, a: &TagCt, s: &[i32]) -> Result<Vec<TagCt>> {
        s.iter().map(|&k| self.rotate(a, k)).collect()
    }
    fn matvec_plain(&self, _: &Diagonals<Plaintext>, _: &TagCt) -> Result<TagCt> {
        unsupported()
    }
    fn matvec_enc(&self, _: &Diagonals<TagCt>, _: &TagCt) -> Result<TagCt> {
        unsupported()
    }
}
pub struct Boot {
    pub contract: PackedBootstrapContract,
    pub calls: Cell<usize>,
    pub bad_at: Cell<Option<usize>>,
    pub error_at: Cell<Option<usize>>,
    pub wrong_count_at: Cell<Option<usize>>,
    pub outputs: RefCell<Vec<usize>>,
}
impl BootstrapOps<Tags> for Boot {
    fn bootstrap(&self, ev: &Tags, ct: &TagCt) -> Result<TagCt> {
        assert_eq!(ct.level, 0);
        self.calls.set(self.calls.get() + 1);
        if self.error_at.get() == Some(self.calls.get()) {
            ev.add_counts(&partial_boot_cost());
            return Err(HekitError::InvalidParams(
                "controlled bootstrap returned error after partial work".into(),
            ));
        }
        ev.add_counts(self.contract.cost_per_boot());
        if self.wrong_count_at.get() == Some(self.calls.get()) {
            ev.add_counts(&OpCounter {
                sse_switches: 1,
                ..Default::default()
            });
        }
        let level = if self.bad_at.get() == Some(self.calls.get()) {
            15
        } else {
            self.output_level()
        };
        let x = ev.ct(level, ev.params.scale());
        self.outputs.borrow_mut().push(x.tag);
        Ok(x)
    }
    fn output_level(&self) -> usize {
        self.contract.output().level()
    }
}
impl PlannedBootstrap<Tags> for Boot {
    fn contract(&self) -> Result<PackedBootstrapContract> {
        Ok(self.contract.clone())
    }
}
pub fn parameters() -> (Arc<CkksParams>, Encoder, PackedBootstrapContract) {
    let (q, aux) = ckks_train_ckks::params::n16_boot_bits();
    let p = Arc::new(CkksParams::new(10, &q, &aux, 42, 8).unwrap());
    let enc = Encoder::new(p.clone());
    let proof = ckks_train_ckks::bootstrap::metadata::DftSupportProof::verify_current().unwrap();
    let contract = PackedBootstrapContract::new(
        p.clone(),
        &ckks_train_ckks::bootstrap::BootParams {
            log_slots: 9,
            ..ckks_train_ckks::bootstrap::BootParams::default_n16()
        },
        &proof,
    )
    .unwrap();
    (p, enc, contract)
}
pub fn geometry(n: usize) -> PackedGeometryPlan {
    PackedGeometryPlan::new(3, &vec![StageShape::new(3, 3, 2).unwrap(); n], 512).unwrap()
}
pub fn graph(ev: &Tags, n: usize) -> PackedBlockGraph<TagCt> {
    let g = geometry(n);
    let stages = (0..n)
        .map(|s| {
            PackedStage::new(
                ev,
                &g,
                s,
                std::array::from_fn(|k| {
                    PackedWeights::new(
                        ev,
                        g.layers()[2 * s + k].clone(),
                        ev.ct(16, ev.params.scale()),
                    )
                    .unwrap()
                }),
                None,
                PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
            )
            .unwrap()
        })
        .collect();
    PackedBlockGraph::new(ev, g, stages).unwrap()
}
pub fn records(
    ev: &Tags,
    g: &PackedBlockGraph<TagCt>,
    ids: &[usize],
) -> Vec<PackedGraphRecord<TagCt>> {
    (0..4)
        .map(|id| {
            PackedGraphRecord::new(
                id,
                PackedVector::new(
                    ev,
                    g.geometry().layers()[0].input(),
                    ev.ct(16, ev.params.scale()),
                )
                .unwrap(),
                (0..g.stages().len())
                    .map(|s| {
                        PackedVector::new(
                            ev,
                            g.geometry().layers()[2 * s + 1].output(),
                            ev.ct(16, ev.params.scale()),
                        )
                        .unwrap()
                    })
                    .collect(),
                ids.contains(&id),
            )
        })
        .collect()
}
pub fn batch(ev: &Tags, g: &PackedBlockGraph<TagCt>, ids: &[usize]) -> PackedGraphBatch<TagCt> {
    PackedGraphBatch::new(
        ev,
        g,
        records(ev, g, ids),
        vec![
            PackedPartition::new(0, 2, ids.iter().filter(|&&i| i < 2).count()),
            PackedPartition::new(2, 4, ids.iter().filter(|&&i| i >= 2).count()),
        ],
    )
    .unwrap()
}
pub fn boot(c: PackedBootstrapContract) -> Boot {
    Boot {
        contract: c,
        calls: Cell::new(0),
        bad_at: Cell::new(None),
        error_at: Cell::new(None),
        wrong_count_at: Cell::new(None),
        outputs: RefCell::new(vec![]),
    }
}
pub struct Client<'a> {
    pub ev: &'a Tags,
    pub contract: PackedRefreshContract,
    pub calls: usize,
    pub replay_at: Option<usize>,
    pub previous: Option<PackedRefreshResponse<TagCt>>,
}
impl PackedClientRefresh<Tags> for Client<'_> {
    fn output_contract(&self) -> &PackedRefreshContract {
        &self.contract
    }
    fn refresh(
        &mut self,
        r: &PackedRefreshRequest<TagCt>,
    ) -> Result<(PackedRefreshResponse<TagCt>, ClientRefreshTiming)> {
        self.calls += 1;
        let response = if self.replay_at == Some(self.calls) {
            self.previous.clone().unwrap()
        } else {
            PackedRefreshResponse::new(
                self.ev,
                r,
                self.ev.ct(self.contract.level(), self.ev.params.scale()),
            )?
        };
        self.previous = Some(response.clone());
        Ok((response, ClientRefreshTiming::default()))
    }
}
pub mod fixture;
pub mod primary;
pub fn forward_batch(
    ev: &Tags,
    graph: &PackedBlockGraph<TagCt>,
    ids: &[usize],
) -> PackedForwardBatch<TagCt> {
    let records = (0..4)
        .map(|id| {
            PackedForwardRecord::new(
                id,
                PackedVector::new(
                    ev,
                    graph.geometry().layers()[0].input(),
                    ev.ct(16, ev.params.scale()),
                )
                .unwrap(),
                ids.contains(&id),
            )
        })
        .collect();
    PackedForwardBatch::new(
        ev,
        graph,
        records,
        vec![PackedPartition::new(0, 4, ids.len())],
    )
    .unwrap()
}
pub fn square_graph(ev: &Tags, n: usize) -> PackedBlockGraph<TagCt> {
    let g = PackedGeometryPlan::new(3, &vec![StageShape::new(3, 3, 3).unwrap(); n], 512).unwrap();
    let stages = (0..n)
        .map(|s| {
            PackedStage::new(
                ev,
                &g,
                s,
                std::array::from_fn(|k| {
                    PackedWeights::new(
                        ev,
                        g.layers()[2 * s + k].clone(),
                        ev.ct(16, ev.params.scale()),
                    )
                    .unwrap()
                }),
                Some(std::array::from_fn(|k| {
                    PackedVector::new(
                        ev,
                        g.layers()[2 * s + k].output(),
                        ev.ct(16, ev.params.scale()),
                    )
                    .unwrap()
                })),
                PackedStageConfig::square_sgd(),
            )
            .unwrap()
        })
        .collect();
    PackedBlockGraph::new(ev, g, stages).unwrap()
}
pub fn state_tags(
    graph: &PackedBlockGraph<TagCt>,
) -> Vec<(usize, Option<usize>, Option<usize>, u64)> {
    graph
        .stages()
        .iter()
        .flat_map(|s| {
            s.layers().iter().enumerate().map(|(k, l)| {
                (
                    l.weights().ciphertext().tag,
                    l.velocity().map(|u| u.ciphertext().tag),
                    s.biases().map(|b| b[k].ciphertext().tag),
                    l.update_index(),
                )
            })
        })
        .collect()
}

pub fn partial_boot_cost() -> OpCounter {
    OpCounter {
        rotations: 2,
        relins: 1,
        ct_ct_tensor: 1,
        ct_pt_mults: 3,
        rescales: 2,
        sse_switches: 1,
        bootstraps: 1,
    }
}
