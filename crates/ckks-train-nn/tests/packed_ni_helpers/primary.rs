//! Registered held-out primary PROGRAM. Compile-only until root's separate execution release.
use super::fixture::*;
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, Decryptor, Encryptor, Evaluator, KeyGenerator,
    bootstrap::Bootstrapper,
};
use ckks_train_ckks::{CkksParams, Encoder};
use ckks_train_core::{
    error::{HekitError, Result},
    sampler::Rng,
};
use ckks_train_nn::{layout_reboot::GradRelin, packed_training::*};
use std::sync::Arc;
type Complex64 = (f64, f64);
use std::cell::Cell;

struct Calls<'a> {
    bs: &'a Bootstrapper,
    attempted: Cell<u64>,
    completed: Cell<u64>,
}
impl BootstrapOps<Evaluator> for Calls<'_> {
    fn bootstrap(&self, ev: &Evaluator, ct: &Ciphertext) -> Result<Ciphertext> {
        if self.attempted.get() >= 228 {
            return Err(HekitError::InvalidParams(
                "registered primary call cap reached".into(),
            ));
        }
        self.attempted.set(self.attempted.get() + 1);
        let out = self.bs.bootstrap(ev, ct)?;
        self.completed.set(self.completed.get() + 1);
        Ok(out)
    }
    fn output_level(&self) -> usize {
        self.bs.output_level()
    }
}
impl PlannedBootstrap<Evaluator> for Calls<'_> {
    fn contract(&self) -> Result<PackedBootstrapContract> {
        self.bs.contract()
    }
    fn validate_binding(&self, ev: &Evaluator, c: &PackedBootstrapContract) -> Result<()> {
        self.bs.validate_binding(ev, c)
    }
}
struct Context {
    p: Arc<CkksParams>,
    enc: Encoder,
    ev: Evaluator,
    encrypt: Encryptor,
    diagnostic: Decryptor,
    bs: Bootstrapper,
}
impl Context {
    fn new() -> Self {
        let (q, p) = ckks_train_ckks::params::n16_boot_bits();
        let p = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
        // Frozen construction order, held-out seed153. No seed or preset search.
        let mut kg = KeyGenerator::new(p.clone(), Rng::from_seed([153; 32])).unwrap();
        let enc = Encoder::new(p.clone());
        let bp = ckks_train_ckks::bootstrap::BootParams {
            log_slots: 9,
            ..ckks_train_ckks::bootstrap::BootParams::default_n16()
        };
        let bs = Bootstrapper::new(p.clone(), bp.clone(), &enc, &mut kg).unwrap();
        let steps = geometry(true).layout().galois_steps();
        assert_eq!(steps, [-2, -1, 1, 2, 4, 8, 16, 32, 64, 128, 256]);
        let galois = kg.galois_keys(&steps);
        let relin = kg.relin_key();
        let public = kg.public_key();
        let secret = kg.secret_key();
        println!(
            "PRIMARY params N={} log_n={} scale_bits={} dnum={} q={:?} p={:?} q_bits={:?} p_bits={:?} bp={bp:?} seed=153 training_keys={steps:?} cost={:?}",
            p.n(),
            p.log_n(),
            p.scale().to_bits(),
            p.dnum(),
            p.q(),
            p.p(),
            p.q_bits(),
            p.p_bits(),
            bs.cost_per_boot()
        );
        Self {
            enc,
            ev: Evaluator::new(p.clone(), Some(relin), Some(galois)),
            encrypt: Encryptor::new(p.clone(), public),
            diagnostic: Decryptor::new(p.clone(), secret),
            bs,
            p,
        }
    }
    fn ct(&self, v: &[f64], seed: u8) -> Ciphertext {
        self.encrypt.encrypt(
            &self.enc.encode(v, self.p.scale(), 16).unwrap(),
            &mut Rng::from_seed([seed; 32]),
        )
    }
    fn decode(&self, c: &Ciphertext) -> Vec<Complex64> {
        self.enc
            .decode_complex(&self.diagnostic.decrypt(c))
            .into_iter()
            .map(|z| (z.re, z.im))
            .collect()
    }
    fn stage(&self, square: bool, ordinal: usize) -> PackedStage<Ciphertext> {
        let g = geometry(square);
        let layers = &g.layers()[2 * ordinal..2 * ordinal + 2];
        let weights = std::array::from_fn(|k| {
            PackedWeights::new(
                &self.ev,
                layers[k].clone(),
                self.ct(&matrix(&W[k], (ordinal + k) % 2 == 0), 200 + k as u8),
            )
            .unwrap()
        });
        let bias = square.then(|| {
            std::array::from_fn(|k| {
                PackedVector::new(
                    &self.ev,
                    layers[k].output(),
                    self.ct(&vector(&B[k], (ordinal + k) % 2 == 1), 202 + k as u8),
                )
                .unwrap()
            })
        });
        PackedStage::new(&self.ev, &g, ordinal, weights, bias, config(square)).unwrap()
    }
    fn records(&self, square: bool, ordinal: usize, t: usize) -> Vec<PackedRecord<Ciphertext>> {
        let g = geometry(square);
        (0..4)
            .map(|r| {
                PackedRecord::new(
                    PackedVector::new(
                        &self.ev,
                        g.layers()[2 * ordinal].input(),
                        self.ct(&vector(&X[r], ordinal == 0), (160 + 8 * t + 2 * r) as u8),
                    )
                    .unwrap(),
                    PackedVector::new(
                        &self.ev,
                        g.layers()[2 * ordinal + 1].output(),
                        self.ct(&vector(&Y[r], ordinal == 0), (161 + 8 * t + 2 * r) as u8),
                    )
                    .unwrap(),
                    VALID[t][r],
                )
            })
            .collect()
    }
}
fn max_error(a: &[Complex64], b: &[Complex64]) -> (f64, usize) {
    assert_eq!(a.len(), 512);
    assert_eq!(b.len(), 512);
    let mut worst = (0.0, 0);
    for (s, (a, b)) in a.iter().zip(b).enumerate() {
        if !(a.0.is_finite() && a.1.is_finite() && b.0.is_finite() && b.1.is_finite()) {
            return (f64::INFINITY, s);
        }
        let x = (a.0 - b.0).hypot(a.1 - b.1);
        if x > worst.0 {
            worst = (x, s);
        }
    }
    worst
}
fn metric(
    a: &[Complex64],
    expected: &[f64],
    support: &[bool],
    expanded: Option<bool>,
    label: &str,
) -> f64 {
    let ideal: Vec<_> = expected.iter().map(|&x| (x, 0.0)).collect();
    let (error, slot) = max_error(a, &ideal);
    let mut active = 0.0_f64;
    let mut padding = 0.0_f64;
    let mut imaginary = 0.0_f64;
    let mut replica = 0.0_f64;
    for s in 0..512 {
        let err = (a[s].0 - ideal[s].0).hypot(a[s].1 - ideal[s].1);
        if support[s] {
            active = active.max(err);
        } else {
            padding = padding.max(err);
        }
        imaginary = imaginary.max(a[s].1.abs());
        if let Some(expanded) = expanded {
            let i = if expanded { s / 4 } else { s % 4 };
            if i < 3 {
                let first = if expanded { 4 * i } else { i };
                replica = replica.max((a[s].0 - a[first].0).hypot(a[s].1 - a[first].1));
            }
        }
    }
    println!(
        "PARITY {label} error={error:.17e} worst_slot={slot} active={active:.17e} padding={padding:.17e} replica={replica:.17e} imaginary={imaginary:.17e}"
    );
    error
}
fn matrix_support() -> Vec<bool> {
    (0..512).map(|s| s / 4 < 3 && s % 4 < 3).collect()
}
fn vector_support(expanded: bool) -> Vec<bool> {
    (0..512)
        .map(|s| if expanded { s / 4 < 3 } else { s % 4 < 3 })
        .collect()
}
fn stage_parity(
    c: &Context,
    stage: &PackedStage<Ciphertext>,
    scalar: &Scalar,
    ordinal: usize,
    label: &str,
) {
    for k in 0..2 {
        let a = c.decode(stage.layers()[k].weights().ciphertext());
        assert!(
            metric(
                &a,
                &matrix(&scalar.w[k], (ordinal + k) % 2 == 0),
                &matrix_support(),
                None,
                &format!("{label} W{k}")
            ) <= E
        );
        if let Some(u) = stage.layers()[k].velocity() {
            assert!(
                metric(
                    &c.decode(u.ciphertext()),
                    &matrix(&scalar.u(k), (ordinal + k) % 2 == 0),
                    &matrix_support(),
                    None,
                    &format!("{label} U{k}")
                ) <= E
            );
        }
        if let Some(b) = stage.biases() {
            let expanded = (ordinal + k) % 2 == 1;
            assert!(
                metric(
                    &c.decode(b[k].ciphertext()),
                    &vector(&scalar.b[k], expanded),
                    &vector_support(expanded),
                    Some(expanded),
                    &format!("{label} b{k}")
                ) <= E
            );
        }
    }
}
fn norm(a: &[Complex64]) -> (f64, usize) {
    let mut result = (0.0, 0);
    for (s, z) in a.iter().enumerate() {
        if !(z.0.is_finite() && z.1.is_finite()) {
            return (f64::INFINITY, s);
        }
        if z.0.hypot(z.1) > result.0 {
            result = (z.0.hypot(z.1), s);
        }
    }
    result
}
fn events(
    c: &Context,
    produced: &[PackedBootstrapEvent<Ciphertext>],
    scalar: &Scalar,
    ordinal: usize,
    hidden: &[[f64; 3]],
    ids: &[usize],
    label: &str,
) -> Option<String> {
    // Emit every produced event's raw metric record before evaluating ordered hard gates.
    // Thus a failure preserves evidence for later same-update calls already performed.
    let mut gates = vec![];
    for (j, event) in produced.iter().enumerate() {
        let (expected, support, expanded, bound) = match event.role() {
            PackedRefreshRole::Weight(layer) => {
                let k = layer - 2 * ordinal;
                (
                    matrix(&scalar.w[k], (ordinal + k) % 2 == 0),
                    matrix_support(),
                    None,
                    0.4,
                )
            }
            PackedRefreshRole::Velocity(layer) => {
                let k = layer - 2 * ordinal;
                (
                    matrix(&scalar.u(k), (ordinal + k) % 2 == 0),
                    matrix_support(),
                    None,
                    0.1,
                )
            }
            PackedRefreshRole::Bias(layer) => {
                let k = layer - 2 * ordinal;
                let expanded = (ordinal + k) % 2 == 1;
                (
                    vector(&scalar.b[k], expanded),
                    vector_support(expanded),
                    Some(expanded),
                    0.4,
                )
            }
            PackedRefreshRole::Hidden { stage, record } => {
                assert_eq!(stage, ordinal);
                let i = ids.iter().position(|&id| id == record as usize).unwrap();
                let expanded = ordinal == 1;
                (
                    vector(&hidden[i], expanded),
                    vector_support(expanded),
                    Some(expanded),
                    0.5,
                )
            }
        };
        let prefix = format!("{label} event={j} role={:?}", event.role());
        let source = c.decode(event.source());
        let input = c.decode(event.input());
        let output = c.decode(event.output());
        let source_error = metric(
            &source,
            &expected,
            &support,
            expanded,
            &format!("{prefix} pre_projection"),
        );
        let projected_error = if let Some(projected) = event.projected() {
            let x = c.decode(projected);
            let (drift, slot) = max_error(&x, &source);
            println!("PROJECTION {prefix} drift={drift:.17e} worst_slot={slot}");
            metric(
                &x,
                &expected,
                &support,
                expanded,
                &format!("{prefix} post_projection"),
            )
        } else {
            assert!(matches!(event.role(), PackedRefreshRole::Hidden { .. }));
            source_error
        };
        let input_error = metric(
            &input,
            &expected,
            &support,
            expanded,
            &format!("{prefix} boot_input"),
        );
        let output_error = metric(
            &output,
            &expected,
            &support,
            expanded,
            &format!("{prefix} boot_output"),
        );
        let (incremental, incremental_slot) = max_error(&output, &input);
        let (range, worst_slot) = norm(&input);
        println!(
            "BOOT {prefix} norm={range:.17e} worst_slot={worst_slot} bound={bound:.17e} incremental={incremental:.17e} incremental_worst={incremental_slot} input_level={} output_level={} input_scale_bits={} output_scale_bits={} counts={:?}",
            event.input().level(),
            event.output().level(),
            event.input().scale().to_bits(),
            event.output().scale().to_bits(),
            event.counter_delta()
        );
        gates.push((
            prefix,
            range <= bound
                && range <= 1.0
                && incremental <= EPS
                && source_error <= E
                && projected_error <= E
                && input_error <= E
                && output_error <= E,
        ));
    }
    gates
        .into_iter()
        .find_map(|(label, ok)| (!ok).then_some(label))
}
pub fn run() {
    println!(
        "PRIMARY_START main_program held_out_seed=153 numerical_coverage=Lazy_only detection_boundary=after_prepare_before_commit cap=228"
    );
    // Prospective metadata/Encoder check precedes held-out key construction.
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    let p = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
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
    let mut predicted = 0;
    for square in [true, false] {
        for ordinal in 0..2 {
            for placement in [
                PackedNiPlacement::Au,
                PackedNiPlacement::Wn,
                PackedNiPlacement::Aa,
            ] {
                predicted += schedule(p.clone(), &enc, &contract, square, ordinal, placement)
                    .iter()
                    .map(|p| p.predicted_counts().bootstraps)
                    .sum::<u64>();
            }
        }
    }
    assert_eq!(predicted, 212);
    let c = match std::panic::catch_unwind(Context::new) {
        Ok(c) => c,
        Err(error) => {
            println!(
                "PRIMARY_END status=FAIL construction attempted=0 completed=0 validated_updates=0 missing_updates=48"
            );
            std::panic::resume_unwind(error)
        }
    };
    let calls = Calls {
        bs: &c.bs,
        attempted: Cell::new(0),
        completed: Cell::new(0),
    };
    let mut validated_updates = 0_u64;
    let mut current = String::from("before first case");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for square in [true, false] {
            for ordinal in 0..2 {
                for placement in [
                    PackedNiPlacement::Au,
                    PackedNiPlacement::Wn,
                    PackedNiPlacement::Aa,
                ] {
                    let trainer = PackedNiTrainer::new(&c.ev, &c.enc, &calls, placement).unwrap();
                    let g = geometry(square);
                    let mut stage = c.stage(square, ordinal);
                    let mut scalar = Scalar::new(square);
                    for t in 0..4 {
                        current = format!(
                            "square={square} ordinal={ordinal} placement={placement:?} update={t}"
                        );
                        println!(
                            "BEGIN {current} attempted={} completed={} validated_updates={validated_updates}",
                            calls.attempted.get(),
                            calls.completed.get()
                        );
                        let old = scalar.clone();
                        stage_parity(&c, &stage, &old, ordinal, &format!("{current} old"));
                        let (h, y) = scalar.step(t);
                        let ids: Vec<_> = VALID[t]
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| **v)
                            .map(|(i, _)| i)
                            .collect();
                        let records = c.records(square, ordinal, t);
                        let result = trainer.prepare_update(
                            &stage,
                            &records,
                            RATE[t],
                            GradRelin::Lazy,
                            next(&c.p, &g, ordinal, t),
                        );
                        let (candidate, receipt) = match result {
                            Ok(done) => done,
                            Err(failure) => {
                                println!(
                                    "EXECUTION_FAILURE {current} phase={:?} role={:?} attempted={} completed={} counters={:?} error={}",
                                    failure.phase(),
                                    failure.role(),
                                    calls.attempted.get(),
                                    calls.completed.get(),
                                    failure.counter_delta(),
                                    failure.error()
                                );
                                let diagnostic_failure = events(
                                    &c,
                                    failure.events(),
                                    &scalar,
                                    ordinal,
                                    &h,
                                    &ids,
                                    &current,
                                );
                                println!(
                                    "EXECUTION_FAILURE_RETAINED {current} validated_events={} diagnostic_failure={diagnostic_failure:?}",
                                    failure.events().len()
                                );
                                panic!(
                                    "registered execution failure before commit: {}",
                                    failure.error()
                                );
                            }
                        };
                        println!(
                            "PREPARED {current} attempted={} completed={} counts={:?} diagnostic_boundary=after_prepare_before_commit",
                            calls.attempted.get(),
                            calls.completed.get(),
                            receipt.actual_counts()
                        );
                        if let Some(failed) =
                            events(&c, receipt.events(), &scalar, ordinal, &h, &ids, &current)
                        {
                            panic!(
                                "registered event gate failed: {failed}; detection after prepare_update before commit"
                            );
                        }
                        stage_parity(
                            &c,
                            &stage,
                            &old,
                            ordinal,
                            &format!("{current} old_retained"),
                        );
                        stage_parity(
                            &c,
                            receipt.updated_stage(),
                            &scalar,
                            ordinal,
                            &format!("{current} candidate"),
                        );
                        stage_parity(
                            &c,
                            candidate.stage(),
                            &scalar,
                            ordinal,
                            &format!("{current} committed_candidate"),
                        );
                        assert_eq!(receipt.step().forward().valid_ids(), ids);
                        for (i, &id) in ids.iter().enumerate() {
                            let expanded = ordinal == 1;
                            let expected = vector(&h[i], expanded);
                            let support = vector_support(expanded);
                            assert!(
                                metric(
                                    &c.decode(
                                        receipt.step().forward().hidden_pre()[i].ciphertext()
                                    ),
                                    &expected,
                                    &support,
                                    Some(expanded),
                                    &format!("{current} record={id} hidden.pre")
                                ) <= E
                            );
                            if let Some(post) = receipt.step().forward().hidden_post() {
                                assert!(
                                    metric(
                                        &c.decode(post[i].ciphertext()),
                                        &expected,
                                        &support,
                                        Some(expanded),
                                        &format!("{current} record={id} hidden.post")
                                    ) <= E
                                );
                            }
                            assert!(
                                metric(
                                    &c.decode(receipt.step().hidden[i].ciphertext()),
                                    &expected,
                                    &support,
                                    Some(expanded),
                                    &format!("{current} record={id} hidden.selected")
                                ) <= E
                            );
                            let expanded = ordinal == 0;
                            assert!(
                                metric(
                                    &c.decode(receipt.step().output[i].ciphertext()),
                                    &vector(&y[i], expanded),
                                    &vector_support(expanded),
                                    Some(expanded),
                                    &format!("{current} record={id} output")
                                ) <= E
                            );
                        }
                        stage = candidate.into_stage();
                        validated_updates += 1;
                    }
                    for k in 0..2 {
                        assert!(
                            scalar.w[k]
                                .iter()
                                .flatten()
                                .zip(W[k].iter().flatten())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0, f64::max)
                                > 4.0 * E
                        );
                        if square {
                            assert!(
                                scalar.b[k]
                                    .iter()
                                    .zip(B[k])
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0.0, f64::max)
                                    > 4.0 * E
                            );
                        } else {
                            assert!(
                                scalar
                                    .u(k)
                                    .iter()
                                    .flatten()
                                    .map(|x| x.abs())
                                    .fold(0.0, f64::max)
                                    > 4.0 * E
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(validated_updates, 48);
        assert_eq!(calls.attempted.get(), 212);
        assert_eq!(calls.completed.get(), 212);
    }));
    println!(
        "PRIMARY_END status={} current={current} attempted={} completed={} validated_updates={validated_updates} missing_updates={} registered_cap=228 predicted_calls=212",
        if result.is_ok() { "PASS" } else { "FAIL" },
        calls.attempted.get(),
        calls.completed.get(),
        48 - validated_updates
    );
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
fn control_all_complex_slots_and_nonfinite_diagnostics_fail_closed() {
    let zero = vec![(0.0, 0.0); 512];
    let mut actual = zero.clone();
    actual[511] = (0.0, 2.0 * EPS);
    assert_eq!(max_error(&actual, &zero), (2.0 * EPS, 511));
    assert_eq!(norm(&actual), (2.0 * EPS, 511));
    actual[37] = (f64::NAN, 0.0);
    assert_eq!(max_error(&actual, &zero), (f64::INFINITY, 37));
    assert_eq!(norm(&actual), (f64::INFINITY, 37));
}
