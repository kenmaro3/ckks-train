//! Explicitly selected control tests and separately gated real-bootstrap primary program.
//! Metadata doubles below supply no numerical NI evidence.
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, CkksParams, Encoder, Encryptor, Evaluator, HeOps, KeyGenerator,
    OpCounter,
};
use ckks_train_core::{error::Result, sampler::Rng};
use ckks_train_nn::{layout_reboot::GradRelin, packed_training::*};
use std::{cell::Cell, sync::Arc};
struct Control {
    ev: Evaluator,
    enc: Encoder,
    stage: PackedStage<Ciphertext>,
    records: Vec<PackedRecord<Ciphertext>>,
    boot: MetadataDouble,
}
struct MetadataDouble {
    contract: PackedBootstrapContract,
    output: Ciphertext,
    calls: Cell<usize>,
    bad_at: usize,
}
impl<E: HeOps<Ct = Ciphertext>> BootstrapOps<E> for MetadataDouble {
    fn bootstrap(&self, ev: &E, _ct: &Ciphertext) -> Result<Ciphertext> {
        self.calls.set(self.calls.get() + 1);
        ev.add_counts(self.contract.cost_per_boot());
        if self.calls.get() == self.bad_at {
            return ev.mod_switch_to_level(&self.output, 15);
        }
        Ok(self.output.clone())
    }
    fn output_level(&self) -> usize {
        self.contract.output().level()
    }
}
impl<E: HeOps<Ct = Ciphertext>> PlannedBootstrap<E> for MetadataDouble {
    fn contract(&self) -> Result<PackedBootstrapContract> {
        Ok(self.contract.clone())
    }
}
fn control() -> Control {
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    let p = Arc::new(CkksParams::new(8, &q, &p, 42, 8).unwrap());
    let enc = Encoder::new(p.clone());
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([91; 32])).unwrap();
    let g = PackedGeometryPlan::new(3, &[StageShape::new(3, 3, 2).unwrap()], p.slots()).unwrap();
    let ev = Evaluator::new(
        p.clone(),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&g.layout().galois_steps())),
    );
    let encrypt = Encryptor::new(p.clone(), kg.public_key());
    let ct = |v: &[f64]| {
        encrypt.encrypt(
            &enc.encode(v, p.scale(), 16).unwrap(),
            &mut Rng::from_seed([92; 32]),
        )
    };
    let weights = std::array::from_fn(|k| {
        PackedWeights::new(&ev, g.layers()[k].clone(), ct(&vec![0.0; p.slots()])).unwrap()
    });
    let stage = PackedStage::new(
        &ev,
        &g,
        0,
        weights,
        None,
        PackedStageConfig::poly_relu2_nag(0.9, 0.08).unwrap(),
    )
    .unwrap();
    let records = (0..3)
        .map(|j| {
            PackedRecord::new(
                PackedVector::new(
                    &ev,
                    g.layers()[0].input(),
                    ct(&g.layers()[0].input().pack(&[0.1, 0.2, 0.3]).unwrap()),
                )
                .unwrap(),
                PackedVector::new(
                    &ev,
                    g.layers()[1].output(),
                    ct(&g.layers()[1].output().pack(&[0.5, 0.0, 0.0]).unwrap()),
                )
                .unwrap(),
                j != 1,
            )
        })
        .collect();
    let proof = ckks_train_ckks::bootstrap::metadata::DftSupportProof::verify_current().unwrap();
    let contract = PackedBootstrapContract::new(
        p.clone(),
        &ckks_train_ckks::bootstrap::BootParams {
            log_slots: 7,
            ..ckks_train_ckks::bootstrap::BootParams::default_n16()
        },
        &proof,
    )
    .unwrap();
    let boot = MetadataDouble {
        contract,
        output: ct(&vec![0.03125; p.slots()]),
        calls: Cell::new(0),
        bad_at: usize::MAX,
    };
    Control {
        ev,
        enc,
        stage,
        records,
        boot,
    }
}
#[test]
fn control_aa_preserves_hole_ids_and_single_hidden_events() {
    let c = control();
    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
    let (candidate, receipt) = trainer
        .prepare_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::Lazy,
            PackedNextStep::Terminal,
        )
        .unwrap();
    assert_eq!(receipt.step().forward().valid_ids(), &[0, 2]);
    assert_eq!(receipt.step().forward().hidden_pre().len(), 2);
    assert_eq!(receipt.step().forward().hidden_post().unwrap().len(), 2);
    assert_eq!(
        receipt
            .events()
            .iter()
            .map(|e| e.role())
            .collect::<Vec<_>>(),
        vec![
            PackedRefreshRole::Hidden {
                stage: 0,
                record: 0
            },
            PackedRefreshRole::Hidden {
                stage: 0,
                record: 2
            },
            PackedRefreshRole::Weight(0),
            PackedRefreshRole::Velocity(0),
            PackedRefreshRole::Weight(1),
            PackedRefreshRole::Velocity(1)
        ]
    );
    assert_eq!(c.boot.calls.get(), 6);
    assert_eq!(receipt.actual_counts().bootstraps, 6);
    assert_eq!(candidate.stage().layers()[0].update_index(), 1);
    assert_eq!(c.stage.layers()[0].update_index(), 0);
}
#[test]
fn control_invalid_future_has_zero_current_counts_and_zero_boots() {
    let c = control();
    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
    let b = c.stage.batch_metadata(&c.ev, &c.records).unwrap();
    let next = PackedBatchMeta::new(
        c.ev.params().clone(),
        1,
        b.input_spec(),
        b.target_spec(),
        b.records().to_vec(),
    )
    .unwrap();
    let fail = trainer
        .prepare_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::Lazy,
            PackedNextStep::Known(
                next,
                PackedUpdateMeta::new(1e-100, GradRelin::Lazy, true).unwrap(),
            ),
        )
        .unwrap_err();
    assert_eq!(fail.counter_delta(), Some(&OpCounter::default()));
    assert_eq!(c.ev.counter(), OpCounter::default());
    assert_eq!(c.boot.calls.get(), 0);
}
#[test]
fn control_stale_and_hypothetical_plans_cannot_execute_after_refresh() {
    let c = control();
    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Au).unwrap();
    let plan = trainer
        .plan_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::Lazy,
            PackedNextStep::Terminal,
        )
        .unwrap();
    let (next, _) = trainer
        .prepare_planned(&c.stage, &c.records, &plan)
        .unwrap();
    let before = c.ev.counter();
    assert!(
        trainer
            .prepare_planned(next.stage(), &c.records, &plan)
            .is_err()
    );
    assert_eq!(c.ev.counter(), before);
    // Same-index fresh-generation reconstruction must also invalidate the old plan.
    let batch = PackedRefreshBatch::new(
        &c.ev,
        &c.stage,
        PackedRefreshContract::new(&c.ev, 16).unwrap(),
    )
    .unwrap();
    let responses = batch
        .requests()
        .iter()
        .map(|r| PackedRefreshResponse::new(&c.ev, r, r.ciphertext().clone()).unwrap())
        .collect();
    let refreshed = batch.complete(&c.ev, responses).unwrap().into_stage();
    assert!(
        trainer
            .prepare_planned(&refreshed, &c.records, &plan)
            .is_err()
    );
    assert_eq!(c.ev.counter(), before);
}
#[test]
fn control_wrong_bootstrap_metadata_preserves_old_state_and_records_observed_work() {
    for kind in 0..4 {
        let mut c = control();
        c.boot.output = match kind {
            0 => c.ev.mod_switch_to_level(&c.boot.output, 15).unwrap(),
            1 => {
                c.ev.relabel(c.boot.output.clone(), c.ev.params().scale() * 2.0)
                    .unwrap()
            }
            2 => control().boot.output,
            _ => Ciphertext::new(
                vec![c.boot.output.polys()[0].clone(); 3],
                c.ev.params().scale(),
            )
            .unwrap(),
        };
        let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
        let fail = trainer
            .prepare_update(
                &c.stage,
                &c.records,
                0.04,
                GradRelin::Lazy,
                PackedNextStep::Terminal,
            )
            .unwrap_err();
        assert_eq!(fail.phase(), PackedNiPhase::Update);
        assert_eq!(fail.events().len(), 0);
        assert_eq!(
            fail.role(),
            Some(PackedRefreshRole::Hidden {
                stage: 0,
                record: 0
            })
        );
        assert_eq!(fail.counter_delta().unwrap().bootstraps, 1);
        assert_eq!(c.stage.layers()[0].update_index(), 0);
        assert!(c.stage.layers().iter().all(|l| l.velocity().is_none()));
    }
}
#[path = "packed_ni_helpers/mod.rs"]
mod fixture;
#[path = "packed_ni_helpers/primary.rs"]
mod primary;
#[test]
#[ignore = "primary HE requires fresh independent review and root release"]
fn packed_ni_primary_12_cases_max_228_calls() {
    primary::run();
}
#[test]
fn control_host_adapter_merges_exactly_once_miniature() {
    let mut q = vec![54, 42, 42];
    q.extend([34; 4]);
    q.extend([58; 7]);
    q.extend([56; 4]);
    let p = Arc::new(CkksParams::new(10, &q, &[60, 60], 42, 9).unwrap());
    let enc = Encoder::new(p.clone());
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([47; 32])).unwrap();
    let bp = ckks_train_ckks::bootstrap::BootParams {
        log_slots: 9,
        ..ckks_train_ckks::bootstrap::BootParams::default_n16()
    };
    let bs = ckks_train_ckks::bootstrap::Bootstrapper::new(p.clone(), bp.clone(), &enc, &mut kg)
        .unwrap();
    let ev = Evaluator::new(p.clone(), Some(kg.relin_key()), None);
    let host = HeOps::fork(&ev);
    let encrypt = Encryptor::new(p.clone(), kg.public_key());
    let ct = encrypt.encrypt(
        &enc.encode(&vec![0.1; 512], p.scale(), 0).unwrap(),
        &mut Rng::from_seed([48; 32]),
    );
    let adapter = PackedHostBootstrap::new(&bs, &host).unwrap();
    <PackedHostBootstrap<'_> as PlannedBootstrap<Evaluator>>::validate_binding(
        &adapter,
        &ev,
        &bs.contract().unwrap(),
    )
    .unwrap();
    println!(
        "HOST_CONTROL N={} q={:?} p={:?} dnum={} scale_bits={} bp={bp:?} seed=47 output_level={} expected={:?}",
        p.n(),
        p.q(),
        p.p(),
        p.dnum(),
        p.scale().to_bits(),
        bs.output_level(),
        bs.cost_per_boot()
    );
    let out = adapter.bootstrap(&ev, &ct).unwrap();
    println!(
        "HOST_CONTROL actual_host={:?} actual_target={:?} calls=1 output_level={}",
        host.counter(),
        ev.counter(),
        out.level()
    );
    assert_eq!(out.level(), 2);
    assert_eq!(host.counter(), bs.cost_per_boot());
    assert_eq!(ev.counter(), bs.cost_per_boot());
    assert_eq!(ev.counter().bootstraps, 1);
}

#[path = "packed_ni_helpers/trace_ops.rs"]
mod trace_ops;
#[test]
fn control_aa_second_affine_and_outer_use_actual_post_ciphertext_coefficients() {
    let c = control();
    let ev = trace_ops::TraceOps {
        ev: HeOps::fork(&c.ev),
        sentinel: c.boot.output.clone(),
        affine_uses: Cell::new(0),
        outer_uses: Cell::new(0),
    };
    let trainer = PackedNiTrainer::new(&ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
    let (_, receipt) = trainer
        .prepare_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::Lazy,
            PackedNextStep::Terminal,
        )
        .unwrap();
    // Each valid post ciphertext is used once by affine2 and once by outer1.
    // Equal approximate plaintexts cannot satisfy this all-RNS-prefix witness.
    assert_eq!(ev.affine_uses.get(), 2);
    assert_eq!(ev.outer_uses.get(), 2);
    for selected in &receipt.step().hidden {
        for (a, b) in selected
            .ciphertext()
            .polys()
            .iter()
            .zip(c.boot.output.polys())
        {
            assert_eq!(a.limbs(), b.limbs());
        }
    }
    assert!(
        receipt
            .step()
            .trace
            .iter()
            .filter(|x| x.name == "derivative1")
            .all(|x| x.level == 13)
    );
}

#[test]
fn control_late_velocity_boot_failure_retains_complete_candidate_and_old_generation() {
    let mut c = control();
    c.boot.bad_at = 4;
    let before: Vec<_> = c
        .stage
        .layers()
        .iter()
        .map(|l| l.weights().ciphertext().clone())
        .collect();
    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
    let fail = trainer
        .prepare_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::Lazy,
            PackedNextStep::Terminal,
        )
        .unwrap_err();
    assert_eq!(fail.phase(), PackedNiPhase::Refresh);
    assert_eq!(fail.role(), Some(PackedRefreshRole::Velocity(0)));
    assert_eq!(fail.events().len(), 3);
    assert_eq!(fail.counter_delta().unwrap().bootstraps, 4);
    assert_eq!(fail.updated_stage().unwrap().layers()[0].update_index(), 1);
    for (k, old) in before.iter().enumerate() {
        assert_eq!(c.stage.layers()[k].update_index(), 0);
        for (a, b) in c.stage.layers()[k]
            .weights()
            .ciphertext()
            .polys()
            .iter()
            .zip(old.polys())
        {
            assert_eq!(a.limbs(), b.limbs());
        }
    }
}
#[test]
fn control_cpu_capability_checks_real_key_context_without_bootstrap() {
    let mut q = vec![54, 42, 42];
    q.extend([34; 4]);
    q.extend([58; 7]);
    q.extend([56; 4]);
    let make = || Arc::new(CkksParams::new(10, &q, &[60, 60], 42, 9).unwrap());
    let p = make();
    let enc = Encoder::new(p.clone());
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([49; 32])).unwrap();
    let bs = ckks_train_ckks::bootstrap::Bootstrapper::new(
        p.clone(),
        ckks_train_ckks::bootstrap::BootParams {
            log_slots: 9,
            ..ckks_train_ckks::bootstrap::BootParams::default_n16()
        },
        &enc,
        &mut kg,
    )
    .unwrap();
    let ev = Evaluator::new(p.clone(), Some(kg.relin_key()), None);
    let contract = bs.contract().unwrap();
    bs.validate_binding(&ev, &contract).unwrap();
    let mut other = KeyGenerator::new(make(), Rng::from_seed([50; 32])).unwrap();
    let wrong = Evaluator::new(p, Some(other.relin_key()), None);
    assert!(bs.validate_binding(&wrong, &contract).is_err());
    assert_eq!(ev.counter(), OpCounter::default());
    assert_eq!(wrong.counter(), OpCounter::default());
}
#[test]
fn control_aa_per_sample_has_exact_all_seven_counts() {
    let c = control();
    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, PackedNiPlacement::Aa).unwrap();
    let (_, receipt) = trainer
        .prepare_update(
            &c.stage,
            &c.records,
            0.04,
            GradRelin::PerSample,
            PackedNextStep::Terminal,
        )
        .unwrap();
    let boot = c.boot.contract.cost_per_boot();
    let expected = OpCounter {
        rotations: 28 + 6 * boot.rotations,
        relins: 14 + 6 * boot.relins,
        ct_ct_tensor: 14 + 6 * boot.ct_ct_tensor,
        ct_pt_mults: 22 + 6 * boot.ct_pt_mults,
        rescales: 34 + 6 * boot.rescales,
        sse_switches: 6 * boot.sse_switches,
        bootstraps: 6,
    };
    assert_eq!(receipt.actual_counts(), &expected);
    assert!(!receipt.step().plan.effective_lazy);
}

#[test]
fn fix_round1_foreign_known_next_rejects_before_current_work() {
    let c = control();
    let b = c.stage.batch_metadata(&c.ev, &c.records).unwrap();
    for (scale, extra) in [(40, false), (42, true), (42, false)] {
        let (mut q, aux) = ckks_train_ckks::params::n16_boot_bits();
        if extra {
            q.extend([42; 8]);
        }
        let foreign = Arc::new(CkksParams::new(8, &q, &aux, scale, 8).unwrap());
        let level = if extra { foreign.max_level() } else { 16 };
        let h = ckks_train_ckks::metadata::CtMeta::new(&foreign, level, foreign.scale()).unwrap();
        let next = PackedBatchMeta::new(
            foreign.clone(),
            1,
            b.input_spec(),
            b.target_spec(),
            vec![PackedRecordMeta::new(0, h, h)],
        )
        .unwrap();
        for placement in [
            PackedNiPlacement::Au,
            PackedNiPlacement::Wn,
            PackedNiPlacement::Aa,
        ] {
            let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &c.boot, placement).unwrap();
            let failure = match trainer.prepare_update(
                &c.stage,
                &c.records,
                0.04,
                GradRelin::Lazy,
                PackedNextStep::Known(
                    next.clone(),
                    PackedUpdateMeta::new(0.01, GradRelin::Lazy, true).unwrap(),
                ),
            ) {
                Err(failure) => failure,
                Ok(_) => panic!("foreign next must reject before current work"),
            };
            assert_eq!(failure.counter_delta(), Some(&OpCounter::default()));
            assert_eq!(c.ev.counter(), OpCounter::default());
            assert_eq!(c.boot.calls.get(), 0);
            assert_eq!(c.stage.layers()[0].update_index(), 0);
        }
    }
}

#[test]
fn fix_round1_host_target_foreign_keys_reject_without_bootstrap() {
    // A downstream-local binder can target a foreign evaluator type without a newtype.
    struct LocalBinding<'a>(&'a Cell<usize>);
    impl PackedTargetKeyBinding<Evaluator> for LocalBinding<'_> {
        fn validate_packed_target_keys(&self, ev: &Evaluator, p: &Arc<CkksParams>) -> Result<()> {
            self.0.set(self.0.get() + 1);
            PackedCpuTargetKeyBinding.validate_packed_target_keys(ev, p)
        }
    }
    let mut q = vec![54, 42, 42];
    q.extend([34; 4]);
    q.extend([58; 7]);
    q.extend([56; 4]);
    let make = || Arc::new(CkksParams::new(10, &q, &[60, 60], 42, 9).unwrap());
    let p = make();
    let enc = Encoder::new(p.clone());
    let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([49; 32])).unwrap();
    let bs = ckks_train_ckks::bootstrap::Bootstrapper::new(
        p.clone(),
        ckks_train_ckks::bootstrap::BootParams {
            log_slots: 9,
            ..ckks_train_ckks::bootstrap::BootParams::default_n16()
        },
        &enc,
        &mut kg,
    )
    .unwrap();
    let host = Evaluator::new(p.clone(), Some(kg.relin_key()), None);
    let adapter = PackedHostBootstrap::new(&bs, &host).unwrap();
    let checks = Cell::new(0);
    let explicit =
        PackedHostBootstrap::with_target_binding(&bs, &host, LocalBinding(&checks)).unwrap();
    let g = PackedGeometryPlan::new(3, &[StageShape::new(3, 3, 2).unwrap()], p.slots()).unwrap();
    let steps = g.layout().galois_steps();
    let good = Evaluator::new(
        p.clone(),
        Some(kg.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    let mut other = KeyGenerator::new(make(), Rng::from_seed([50; 32])).unwrap();
    let wrong_relin = Evaluator::new(
        p.clone(),
        Some(other.relin_key()),
        Some(kg.galois_keys(&steps)),
    );
    let wrong_rotation = Evaluator::new(p, Some(kg.relin_key()), Some(other.galois_keys(&steps)));
    PackedNiTrainer::new(&good, &enc, &adapter, PackedNiPlacement::Aa).unwrap();
    PackedNiTrainer::new(&good, &enc, &explicit, PackedNiPlacement::Aa).unwrap();
    for wrong in [&wrong_relin, &wrong_rotation] {
        assert!(wrong.has_relin_key());
        assert!(steps.iter().all(|&s| wrong.has_galois_key(s)));
        assert!(PackedNiTrainer::new(wrong, &enc, &adapter, PackedNiPlacement::Aa).is_err());
        assert!(PackedNiTrainer::new(wrong, &enc, &explicit, PackedNiPlacement::Aa).is_err());
        assert_eq!(wrong.counter(), OpCounter::default());
    }
    assert_eq!(checks.get(), 3);
    assert_eq!(host.counter(), OpCounter::default());
    assert_eq!(good.counter(), OpCounter::default());
}
