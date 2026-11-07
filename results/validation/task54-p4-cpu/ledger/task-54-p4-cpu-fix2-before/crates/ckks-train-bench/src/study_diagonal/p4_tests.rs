//! NONSCIENTIFIC_TEST_EVIDENCE: one explicitly released future CPU matrix.
use super::*;
use crate::{
    study_admission::{
        MicroPart, PlanningBudget, PlanningParameters, PlanningUsage, p4_fixtures::SyntheticCpuPlan,
    },
    study_client::{
        CpuBootstrapDescription, PreparationInput, StudyClient, context_parameters, make_owned,
        prepare_then_make,
    },
    study_plain::p4_fixtures::{SignalFamily, SignalOracle, signal_oracle},
};
use ckks_train_ckks::bootstrap::metadata::DftSupportProof;
use ckks_train_core::sampler::Rng;
use ckks_train_nn::trainer_ni::BootPlacement;
use std::{cell::Cell, rc::Rc};

thread_local! { static ACTIVE: RefCell<Option<Rc<RuntimeBudget>>> = const { RefCell::new(None) }; }
#[derive(Default)]
pub(crate) struct RuntimeBudget {
    parameters: Cell<usize>,
    keys: Cell<usize>,
    proofs: Cell<usize>,
    descriptors: Cell<usize>,
    bootstrappers: Cell<usize>,
    encoders: Cell<usize>,
    attempts: Cell<usize>,
    boots: Cell<usize>,
    f64_updates: Cell<usize>,
    oracles: Cell<usize>,
    sessions: Cell<usize>,
    spy: Cell<usize>,
    state_snapshots: Cell<usize>,
    state_assertions: Cell<usize>,
}
fn reserve(cell: &Cell<usize>, cap: usize, name: &str) {
    let next = cell.get().checked_add(1).expect("guard overflow");
    eprintln!(
        "P4_RESERVE operation={name} before={} next={next} cap={cap}",
        cell.get()
    );
    assert!(next <= cap, "P4 {name} cap");
    cell.set(next);
}
impl RuntimeBudget {
    pub(crate) fn reserve_f64(&self) {
        reserve(&self.f64_updates, 32, "actual Engine::train");
    }
    pub(crate) fn reserve_oracle(&self) {
        reserve(&self.oracles, 18, "SignalOracle");
    }
    pub(crate) fn reserve_boot(&self) {
        reserve(&self.boots, 85, "actual bootstrap");
    }
    pub(crate) fn reserve_key(&self) {
        reserve(&self.keys, 2, "KeyGenerator");
    }
    pub(crate) fn reserve_bootstrapper(&self) {
        reserve(&self.bootstrappers, 1, "Bootstrapper");
    }
}
pub(crate) fn reserve_encoder() {
    ACTIVE.with(|b| {
        if let Some(b) = b.borrow().as_ref() {
            reserve(&b.encoders, 22, "Encoder");
        }
    });
}
pub(crate) fn reserve_attempt() {
    ACTIVE.with(|b| {
        reserve(
            &b.borrow().as_ref().expect("active budget").attempts,
            34,
            "trainer attempt",
        )
    });
}
fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}
struct Case {
    family: SignalFamily,
    mode: DiagonalMode,
    plan: Arc<SyntheticCpuPlan>,
    poison: bool,
}
fn add_case(
    cases: &mut Vec<Case>,
    family: SignalFamily,
    mode: DiagonalMode,
    p: PlanningParameters<'_>,
    budget: &Arc<PlanningBudget>,
    poison: bool,
) {
    let plan =
        Arc::new(SyntheticCpuPlan::build(&family.recipe(), mode, p, budget.clone()).unwrap());
    cases.push(Case {
        family,
        mode,
        plan,
        poison,
    });
}
fn spy_contract(
    plan: &Arc<SyntheticCpuPlan>,
    p: PlanningParameters<'_>,
    other_actual: PlanningParameters<'_>,
    budget: &RuntimeBudget,
) {
    let claim = plan.claim();
    let marker = prepare_then_make(
        PreparationInput::Synthetic {
            plan: plan.clone(),
            claim: &claim,
        },
        p,
        &plan.parts,
        |_| {
            reserve(&budget.spy, 1, "positive terminal factory spy");
            Ok("terminal-marker")
        },
    )
    .unwrap();
    assert_eq!(marker, "terminal-marker");
    let mut negatives = 0;
    let mut rejected = |claim: &crate::study_admission::p4_fixtures::SyntheticClaim,
                        actual: PlanningParameters<'_>,
                        parts: &[Vec<MicroPart>],
                        field: &str| {
        let calls = Cell::new(0);
        let result = prepare_then_make(
            PreparationInput::Synthetic {
                plan: plan.clone(),
                claim,
            },
            actual,
            parts,
            |_| {
                calls.set(calls.get() + 1);
                Ok(())
            },
        );
        assert_eq!(result.unwrap_err(), mismatch(field));
        assert_eq!(calls.get(), 0);
        negatives += 1;
    };
    let mut wrong = plan.claim();
    wrong.identity.push('x');
    rejected(&wrong, p, &plan.parts, "identity");
    // Existing I parameters against the NI-bound plan; claim bytes agree with the
    // actual I description. Removing actual.bytes != self.parameters makes this
    // otherwise-valid negative reach the shared factory and fail this assertion.
    let actual_claim = plan.claim_for_actual_parameters(other_actual);
    rejected(&actual_claim, other_actual, &plan.parts, "parameters");
    for which in 0..7 {
        rejected(
            &plan.bad_parameter_claim(p, which),
            p,
            &plan.parts,
            "parameters",
        );
    }
    let variants = [
        vec![],
        vec![vec![
            MicroPart { start: 0, valid: 4 },
            MicroPart { start: 0, valid: 4 },
        ]],
        vec![vec![MicroPart { start: 0, valid: 0 }]],
        vec![vec![MicroPart { start: 0, valid: 9 }]],
        vec![vec![MicroPart { start: 0, valid: 7 }]],
        vec![vec![MicroPart { start: 1, valid: 8 }]],
        vec![vec![
            MicroPart { start: 4, valid: 4 },
            MicroPart { start: 0, valid: 4 },
        ]],
    ];
    for parts in &variants {
        rejected(&claim, p, parts, "partition");
    }
    assert_eq!(negatives, 16);
}
fn arrays(oracle: &SignalOracle) -> Vec<(ndarray::Array2<f64>, Vec<f64>)> {
    oracle
        .initial()
        .iter()
        .map(|l| (l.weights().clone(), l.biases().to_vec()))
        .collect()
}
fn prepare_initial(
    client: &StudyClient,
    context: &CpuEvaluationContext,
    plan: &Arc<SyntheticCpuPlan>,
    layers: &[(ndarray::Array2<f64>, Vec<f64>)],
) -> EncryptedInitialState {
    let claim = plan.claim();
    let checked = prepare_then_make(
        PreparationInput::Synthetic {
            plan: plan.clone(),
            claim: &claim,
        },
        context_parameters(context),
        &plan.parts,
        Ok,
    )
    .unwrap();
    client.encrypt_initial(checked, context, layers).unwrap()
}
fn compare(
    client: &StudyClient,
    context: &CpuEvaluationContext,
    s: &CpuDiagonalSession<'_>,
    oracle: &SignalOracle,
    update: usize,
    ni: bool,
) {
    let before = context.counter();
    let snapshot = s.snapshot();
    let actual = client.inspect_snapshot(&snapshot, context).unwrap();
    assert_eq!(before, context.counter());
    let expected = oracle.post(update).unwrap();
    assert_eq!(actual.layers().len(), expected.len());
    let tolerance = if ni { 4e-4 } else { 1e-5 };
    let mut maximum = 0_f64;
    for ((w, b), e) in actual.layers().iter().zip(expected) {
        assert_eq!(w.dim(), e.weights().dim());
        assert_eq!(b.len(), e.biases().len());
        for (&a, &e) in w.iter().chain(b).zip(e.weights().iter().chain(e.biases())) {
            assert!(a.is_finite() && e.is_finite());
            maximum = maximum.max((a - e).abs());
            assert!(
                (a - e).abs() <= tolerance,
                "all-entry error={} cap={tolerance}",
                (a - e).abs()
            );
        }
        if !oracle.recipe().square() {
            assert!(e.biases().iter().all(|v| v.to_bits() == 0_f64.to_bits()));
        }
    }
    eprintln!(
        "P4_PARITY update={update} NI={ni} maximum_abs={maximum:.17e} frozen_cap={tolerance:.17e}"
    );
}
// Test-owned read-only snapshots. Copies of encrypted state are compared exactly;
// this performs no HE work, decryption, encoding, planner call or optimizer recurrence.
#[derive(Debug, PartialEq, Eq)]
struct OptimizerConfiguration {
    current: u64,
    reference: Option<u64>,
    momentum: Option<u64>,
    weight_factor: u64,
    bias_factor: u64,
    reference_factor: Option<u64>,
}
fn optimizer_configurations(s: &CpuDiagonalSession<'_>) -> Vec<OptimizerConfiguration> {
    match &s.trainer {
        Training::Interactive(t) => (0..s.mlp.layers.len())
            .map(|layer| {
                let view = t.optimizer_view(layer).unwrap();
                OptimizerConfiguration {
                    current: view.lr().to_bits(),
                    reference: view.reference().map(f64::to_bits),
                    momentum: view.momentum().map(f64::to_bits),
                    weight_factor: view.weight_factor().to_bits(),
                    bias_factor: view.bias_factor().to_bits(),
                    reference_factor: view.reference_factor().map(f64::to_bits),
                }
            })
            .collect(),
        Training::NonInteractive(t) => {
            let optimizer = t.optimizer();
            let momentum = match optimizer {
                NiOptimizer::Nag { momentum, .. } => Some(momentum),
                _ => None,
            };
            // Existing helper computation from actual configured state, not observed mask/kappa.
            let reference_factor = (optimizer.has_velocity()
                && optimizer.lr() / t.lr_reference() != 1.)
                .then(|| optimizer.with_lr(t.lr_reference()).grad_scale());
            vec![OptimizerConfiguration {
                current: optimizer.lr().to_bits(),
                reference: Some(t.lr_reference().to_bits()),
                momentum: momentum.map(f64::to_bits),
                weight_factor: optimizer.grad_scale().to_bits(),
                bias_factor: optimizer.bias_grad_scale().to_bits(),
                reference_factor: reference_factor.map(f64::to_bits),
            }]
        }
    }
}
enum EncryptedOptimizerSnapshot {
    Interactive(Vec<Option<ckks_train_ckks::Diagonals<Ciphertext>>>),
    NonInteractive {
        layout: ckks_train_nn::trainer_ni::CompactLayout,
        weights: Vec<Ciphertext>,
        velocity: Vec<Option<Ciphertext>>,
    },
}
#[derive(Debug, PartialEq)]
struct FailureSnapshot {
    update: usize,
    phase: &'static str,
    measured: OpCounter,
    trace: Vec<(String, usize)>,
}
impl From<&FailedUpdate> for FailureSnapshot {
    fn from(f: &FailedUpdate) -> Self {
        Self {
            update: f.update,
            phase: f.phase,
            measured: f.measured.clone(),
            trace: f.trace.clone(),
        }
    }
}
struct ReadOnlySessionSnapshot {
    counter: OpCounter,
    completed: usize,
    samples: usize,
    poisoned: bool,
    failure: Option<FailureSnapshot>,
    trace: Vec<(String, usize)>,
    batch: BatchSpec,
    model: EncryptedModelSnapshot,
    configurations: Vec<OptimizerConfiguration>,
    optimizer: EncryptedOptimizerSnapshot,
}
fn assert_ciphertext_exact(expected: &Ciphertext, actual: &Ciphertext) {
    assert_eq!(expected.len(), actual.len());
    assert_eq!(expected.level(), actual.level());
    assert_eq!(expected.scale().to_bits(), actual.scale().to_bits());
    assert_eq!(expected.polys().len(), actual.polys().len());
    for (expected, actual) in expected.polys().iter().zip(actual.polys()) {
        assert!(Arc::ptr_eq(expected.basis(), actual.basis()));
        assert_eq!(expected.level(), actual.level());
        assert_eq!(expected.is_ntt_form(), actual.is_ntt_form());
        assert_eq!(expected.limbs(), actual.limbs());
    }
}
fn assert_diagonals_exact(
    expected: &ckks_train_ckks::Diagonals<Ciphertext>,
    actual: &ckks_train_ckks::Diagonals<Ciphertext>,
) {
    assert_eq!(expected.d(), actual.d());
    assert_eq!(expected.stride(), actual.stride());
    assert_eq!(expected.entries().len(), actual.entries().len());
    for (expected, actual) in expected.entries().iter().zip(actual.entries()) {
        assert_ciphertext_exact(expected, actual);
    }
}
impl ReadOnlySessionSnapshot {
    fn capture(s: &CpuDiagonalSession<'_>) -> Self {
        ACTIVE.with(|b| {
            reserve(
                &b.borrow().as_ref().unwrap().state_snapshots,
                54,
                "encrypted state snapshot",
            )
        });
        let optimizer = match &s.trainer {
            Training::Interactive(t) => EncryptedOptimizerSnapshot::Interactive(
                (0..s.mlp.layers.len())
                    .map(|layer| t.optimizer_view(layer).unwrap().velocity().cloned())
                    .collect(),
            ),
            Training::NonInteractive(t) => EncryptedOptimizerSnapshot::NonInteractive {
                layout: t.compact_layout().clone(),
                weights: t.state().weights.clone(),
                velocity: t.state().velocity.clone(),
            },
        };
        Self {
            counter: s.context.counter(),
            completed: s.completed_updates(),
            samples: s.processed_samples(),
            poisoned: s.is_poisoned(),
            failure: s.failed_update().map(FailureSnapshot::from),
            trace: s.trace().to_vec(),
            batch: s.batch,
            model: s.snapshot(),
            configurations: optimizer_configurations(s),
            optimizer,
        }
    }
    fn assert_unchanged(&self, s: &CpuDiagonalSession<'_>) {
        ACTIVE.with(|b| {
            reserve(
                &b.borrow().as_ref().unwrap().state_assertions,
                264,
                "encrypted state unchanged assertion",
            )
        });
        assert_eq!(self.counter, s.context.counter());
        assert_eq!(self.completed, s.completed_updates());
        assert_eq!(self.samples, s.processed_samples());
        assert_eq!(self.poisoned, s.is_poisoned());
        assert_eq!(self.failure, s.failed_update().map(FailureSnapshot::from));
        assert_eq!(self.trace, s.trace());
        assert_eq!(self.batch, s.batch);
        assert_eq!(self.configurations, optimizer_configurations(s));
        assert!(Arc::ptr_eq(&self.model.owner, &s.context.owner));
        assert_eq!(self.model.layout, s.description.geometry.layout());
        assert_eq!(self.model.layers.len(), s.mlp.layers.len());
        for (expected, (actual, _)) in self.model.layers.iter().zip(&s.mlp.layers) {
            assert_eq!(expected.w.d_in(), actual.w.d_in());
            assert_eq!(expected.w.d_out(), actual.w.d_out());
            assert_diagonals_exact(expected.w.diagonals(), actual.w.diagonals());
            assert_ciphertext_exact(&expected.b, &actual.b);
        }
        match (&self.optimizer, &s.trainer) {
            (EncryptedOptimizerSnapshot::Interactive(expected), Training::Interactive(actual)) => {
                assert_eq!(expected.len(), s.mlp.layers.len());
                for (layer, expected) in expected.iter().enumerate() {
                    let view = actual.optimizer_view(layer).unwrap();
                    match (expected.as_ref(), view.velocity()) {
                        (None, None) => {}
                        (Some(expected), Some(actual)) => assert_diagonals_exact(expected, actual),
                        _ => panic!("interactive velocity presence changed during rejection"),
                    }
                }
            }
            (
                EncryptedOptimizerSnapshot::NonInteractive {
                    layout,
                    weights,
                    velocity,
                },
                Training::NonInteractive(actual),
            ) => {
                assert_eq!(layout, actual.compact_layout());
                assert_eq!(weights.len(), actual.state().weights.len());
                assert_eq!(velocity.len(), actual.state().velocity.len());
                for (expected, actual) in weights.iter().zip(&actual.state().weights) {
                    assert_ciphertext_exact(expected, actual);
                }
                for (expected, actual) in velocity.iter().zip(&actual.state().velocity) {
                    match (expected, actual) {
                        (None, None) => {}
                        (Some(expected), Some(actual)) => assert_ciphertext_exact(expected, actual),
                        _ => panic!("NI velocity presence changed during rejection"),
                    }
                }
            }
            _ => panic!("trainer mode changed during rejection"),
        }
    }
    fn assert_velocity_present(&self) {
        match &self.optimizer {
            EncryptedOptimizerSnapshot::Interactive(v) => {
                assert!(!v.is_empty() && v.iter().all(Option::is_some))
            }
            EncryptedOptimizerSnapshot::NonInteractive { velocity, .. } => {
                assert!(!velocity.is_empty() && velocity.iter().all(Option::is_some))
            }
        }
    }
}
fn later_rate_rejection(s: &mut CpuDiagonalSession<'_>, packet: &mut EncryptedLogicalUpdate) {
    assert_eq!(s.completed_updates(), 1);
    let snapshot = ReadOnlySessionSnapshot::capture(s);
    snapshot.assert_velocity_present();
    let next_rate = s.plan.rates()[packet.index].kernel.to_bits();
    assert!(
        snapshot
            .configurations
            .iter()
            .all(|c| c.current != next_rate)
    );
    let target = packet.micros[0].targets.pop().unwrap();
    assert_eq!(
        s.step(packet).unwrap_err(),
        mismatch("stage target cardinality")
    );
    snapshot.assert_unchanged(s);
    assert!(!s.is_poisoned());
    packet.micros[0].targets.push(target);
}
fn prework_negatives(
    s: &mut CpuDiagonalSession<'_>,
    packet: &mut EncryptedLogicalUpdate,
    other: &CpuEvaluationContext,
    client: &StudyClient,
    other_plan: &Arc<SyntheticCpuPlan>,
) {
    let state = ReadOnlySessionSnapshot::capture(s);
    let snapshot = client.inspect_snapshot(&s.snapshot(), s.context).unwrap();
    let reject = |s: &mut CpuDiagonalSession<'_>, p: &EncryptedLogicalUpdate, expected: &str| {
        let error = s.step(p).unwrap_err();
        assert_eq!(error, mismatch(expected));
        state.assert_unchanged(s);
        assert!(!s.is_poisoned());
    };
    packet.index += 1;
    reject(s, packet, "update index or replay");
    packet.index -= 1;
    // Association-token mutation only: this is not a third real-key experiment.
    let original = packet.owner.clone();
    packet.owner = Arc::new(());
    reject(s, packet, "packet context identity");
    packet.owner = original;
    let original = packet.plan.clone();
    packet.plan = PlanOwner::Synthetic(other_plan.clone());
    reject(s, packet, "packet plan identity");
    packet.plan = original;
    packet.micros[0].part.start += 1;
    reject(s, packet, "ordered partition");
    packet.micros[0].part.start -= 1;
    packet.micros[0].part.valid += 9;
    reject(s, packet, "ordered partition");
    packet.micros[0].part.valid -= 9;
    if packet.micros.len() > 1 {
        packet.micros.swap(0, 1);
        reject(s, packet, "ordered partition");
        packet.micros.swap(0, 1);
    }
    packet.total += 1;
    reject(s, packet, "logical valid denominator");
    packet.total -= 1;
    packet.target_width += 1;
    reject(s, packet, "target/input shape");
    packet.target_width -= 1;
    let target = packet.micros[0].targets.pop().unwrap();
    reject(s, packet, "stage target cardinality");
    packet.micros[0].targets.push(target.clone());
    packet.micros[0].targets.push(target);
    reject(s, packet, "stage target cardinality");
    packet.micros[0].targets.pop();
    let original = packet.micros[0].x.clone();
    packet.micros[0].x = Ciphertext::new(original.polys().to_vec(), original.scale() * 2.).unwrap();
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].x = original.clone();
    packet.micros[0].x = s
        .context
        .evaluator
        .fork()
        .mod_switch_to_level(&original, original.level() - 1)
        .unwrap();
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].x = original.clone();
    let mut polys = original.polys().to_vec();
    polys.push(polys[0].clone());
    packet.micros[0].x = Ciphertext::new(polys, original.scale()).unwrap();
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].x = original.clone();
    // This foreign-basis negative executes only for I: its level9 fits the NI
    // context; NI level15 exceeds the other I max9. Both modes share check_ct,
    // but no separate NI foreign-basis negative execution is claimed.
    if original.level() <= other.parameters().max_level() {
        let mut foreign =
            ckks_train_core::poly::RnsPoly::zero(other.parameters().basis_q(), original.level());
        foreign.to_ntt();
        packet.micros[0].x =
            Ciphertext::new(vec![foreign.clone(), foreign], original.scale()).unwrap();
        reject(s, packet, "ciphertext basis");
    }
    packet.micros[0].x = original;
    let after = client.inspect_snapshot(&s.snapshot(), s.context).unwrap();
    for ((aw, ab), (bw, bb)) in snapshot.layers().iter().zip(after.layers()) {
        assert_eq!(aw, bw);
        assert_eq!(ab, bb);
    }
}
#[test]
#[ignore = "HE/F64/planner runtime requires a separate frozen-source release"]
fn p4_cpu_bridge_contract() {
    assert_eq!(CpuDiagonalSession::capability(), CpuCapability::SingleStage);
    assert_eq!(
        require_single_stage(2).unwrap_err(),
        mismatch("missing block-stage execution capability")
    );
    run_contract_matrix();
}
fn run_contract_matrix() {
    let runtime = Rc::new(RuntimeBudget::default());
    ACTIVE.with(|b| *b.borrow_mut() = Some(runtime.clone()));
    reserve(&runtime.parameters, 2, "CkksParams");
    let ip = Arc::new(
        CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap(),
    );
    reserve(&runtime.parameters, 2, "CkksParams");
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    let np = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
    let bp = BootParams {
        log_slots: 9,
        ..BootParams::default_n16()
    };
    reserve(&runtime.proofs, 1, "DftSupportProof");
    let proof = DftSupportProof::verify_current().unwrap();
    reserve(&runtime.descriptors, 1, "BootstrapMetadata");
    let bm = Arc::new(BootstrapMetadata::new_with_verified_support(&np, &bp, &proof).unwrap());
    let pure = Arc::new(PlanningBudget::new(PlanningUsage {
        geometries: 28,
        initial_states: 28,
        updates: 42,
        handoffs: 0,
        wrapper_calls: 0,
    }));
    let i = DiagonalMode::Interactive;
    let wn = DiagonalMode::NonInteractive(BootPlacement::WhenNeeded);
    let au = DiagonalMode::NonInteractive(BootPlacement::AfterUpdate);
    let aa = DiagonalMode::NonInteractive(BootPlacement::AfterActivation);
    let params = |mode| {
        if mode == i {
            PlanningParameters {
                params: &ip,
                bootstrap: None,
                boot_params: None,
            }
        } else {
            PlanningParameters {
                params: &np,
                bootstrap: Some(&bm),
                boot_params: Some(&bp),
            }
        }
    };
    let mut cases = vec![];
    for (n, family) in SignalFamily::SMOKE.into_iter().enumerate() {
        add_case(&mut cases, family, i, params(i), &pure, false);
        let mode = [wn, au, aa][n % 3];
        add_case(&mut cases, family, mode, params(mode), &pure, false);
    }
    for mode in [i, wn, au, aa] {
        add_case(
            &mut cases,
            SignalFamily::PolyFourRate,
            mode,
            params(mode),
            &pure,
            false,
        );
    }
    for mode in [i, au] {
        add_case(
            &mut cases,
            SignalFamily::PolyFirstReference,
            mode,
            params(mode),
            &pure,
            false,
        );
    }
    for mode in [i, au] {
        add_case(
            &mut cases,
            SignalFamily::PolyC3,
            mode,
            params(mode),
            &pure,
            true,
        );
    }
    assert_eq!(cases.len(), 20);
    assert_eq!(pure.usage().geometries, 20);
    assert_eq!(pure.usage().initial_states, 20);
    assert_eq!(pure.usage().updates, 34);
    assert_eq!(pure.usage().handoffs, 0);
    spy_contract(&cases[1].plan, params(wn), params(i), &runtime);
    let keys = |ni: bool| {
        let mut keys = cases
            .iter()
            .filter(|c| (c.mode != i) == ni)
            .flat_map(|c| c.plan.geometry.rotation_steps().iter().copied())
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        keys
    };
    let setup = |case: &Case,
                 p: Arc<CkksParams>,
                 boot: Option<CpuBootstrapDescription>,
                 steps: Vec<i32>| {
        let claim = case.plan.claim();
        prepare_then_make(
            PreparationInput::Synthetic {
                plan: case.plan.clone(),
                claim: &claim,
            },
            params(case.mode),
            &case.plan.parts,
            |checked| {
                make_owned(
                    &checked,
                    p,
                    &steps,
                    boot,
                    rng(51),
                    rng(52),
                    Some(runtime.clone()),
                )
            },
        )
        .unwrap()
    };
    let (ic, ictx) = setup(&cases[0], ip.clone(), None, keys(false));
    let (nc, nctx) = setup(
        &cases[1],
        np.clone(),
        Some(CpuBootstrapDescription {
            parameters: bp.clone(),
            metadata: bm.clone(),
        }),
        keys(true),
    );
    let mut successful = 0usize;
    let mut failure_layers: Option<Vec<(ndarray::Array2<f64>, Vec<f64>)>> = None;
    let mut failure_rows: Option<(ndarray::Array2<f64>, Vec<u8>)> = None;
    for case in &cases {
        reserve(&runtime.sessions, 20, "session");
        let ni = case.mode != i;
        let (client, base) = if ni { (&nc, &nctx) } else { (&ic, &ictx) };
        let mut context = base.fork();
        let oracle = if case.poison {
            None
        } else {
            Some(signal_oracle(case.family, &runtime).unwrap())
        };
        let layers = oracle
            .as_ref()
            .map(arrays)
            .unwrap_or_else(|| failure_layers.as_ref().unwrap().clone());
        if matches!(case.family, SignalFamily::PolyC3) && !case.poison {
            failure_layers = Some(layers.clone());
            let (x, y) = oracle.as_ref().unwrap().rows(0).unwrap();
            failure_rows = Some((x.clone(), y.to_vec()));
        }
        if case.poison && ni {
            context.bootstrap.as_mut().unwrap().fail_after = Some(1);
        }
        let mut refresh = client.refresh_service(rng(53));
        if case.poison && !ni {
            refresh.fail_after = Some(1);
        }
        let mut initial = prepare_initial(client, &context, &case.plan, &layers);
        // All initial-packet negatives share the same validator before trainer construction.
        let owner = PlanOwner::Synthetic(case.plan.clone());
        let token = initial.owner.clone();
        initial.owner = Arc::new(());
        assert_eq!(
            validate_initial_packet(&context, &owner, &initial).unwrap_err(),
            mismatch("initial context identity")
        );
        initial.owner = token;
        let different = cases
            .iter()
            .find(|c| !Arc::ptr_eq(&c.plan, &case.plan))
            .unwrap();
        assert_eq!(
            validate_initial_packet(
                &context,
                &PlanOwner::Synthetic(different.plan.clone()),
                &initial
            )
            .unwrap_err(),
            mismatch("initial plan identity")
        );
        let removed = initial.mlp.layers.pop().unwrap();
        assert!(validate_initial_packet(&context, &owner, &initial).is_err());
        initial.mlp.layers.push(removed);
        let original_batch = initial.batch;
        initial.batch.y_scale *= 2.;
        assert!(validate_initial_packet(&context, &owner, &initial).is_err());
        initial.batch = original_batch;
        if let Some(compact) = &mut initial.compact {
            let c = compact.weights.pop().unwrap();
            assert!(validate_initial_packet(&context, &owner, &initial).is_err());
            initial.compact.as_mut().unwrap().weights.push(c);
            let c = initial.compact.as_ref().unwrap().weights[0].clone();
            initial.compact.as_mut().unwrap().velocity[0] = Some(c);
            assert!(validate_initial_packet(&context, &owner, &initial).is_err());
            initial.compact.as_mut().unwrap().velocity[0] = None;
        }
        let original_evaluator = context.evaluator.fork();
        context.evaluator = Evaluator::with_config(
            context.parameters().clone(),
            original_evaluator.relin_key().cloned(),
            original_evaluator.galois_keys().cloned(),
            ckks_train_ckks::EvalConfig::NAIVE,
        );
        assert_eq!(
            validate_initial_packet(&context, &owner, &initial).unwrap_err(),
            mismatch("evaluation configuration")
        );
        context.evaluator = original_evaluator;
        let original = initial.mlp.layers[0].0.b.clone();
        initial.mlp.layers[0].0.b =
            Ciphertext::new(original.polys().to_vec(), original.scale() * 2.).unwrap();
        assert!(validate_initial_packet(&context, &owner, &initial).is_err());
        initial.mlp.layers[0].0.b = original;
        let mut session =
            CpuDiagonalSession::from_owner(&context, owner, initial, (!ni).then_some(&refresh))
                .unwrap();
        for update in 0..case.plan.parts.len() {
            let (x, y) = oracle
                .as_ref()
                .map(|o| o.rows(update).unwrap())
                .unwrap_or_else(|| {
                    let (x, y) = failure_rows.as_ref().unwrap();
                    (x, y.as_slice())
                });
            let mut packet = client
                .encrypt_rows(
                    PlanOwner::Synthetic(case.plan.clone()),
                    update,
                    &context,
                    x,
                    y,
                    case.family.recipe().classes(),
                    case.family.recipe().width(),
                )
                .unwrap();
            if update == 0 && !case.poison {
                let before = context.counter();
                let mut labels = y.to_vec();
                labels[0] = u8::MAX;
                assert!(
                    client
                        .encrypt_rows(
                            PlanOwner::Synthetic(case.plan.clone()),
                            update,
                            &context,
                            x,
                            &labels,
                            case.family.recipe().classes(),
                            case.family.recipe().width()
                        )
                        .is_err()
                );
                assert!(
                    client
                        .encrypt_rows(
                            PlanOwner::Synthetic(case.plan.clone()),
                            update,
                            &context,
                            x,
                            &y[..y.len() - 1],
                            case.family.recipe().classes(),
                            case.family.recipe().width()
                        )
                        .is_err()
                );
                assert_eq!(before, context.counter());
                let targets = crate::profile_data::padded_targets(
                    y,
                    case.family.recipe().classes(),
                    case.family.recipe().width(),
                )
                .unwrap();
                assert!(
                    targets
                        .columns()
                        .into_iter()
                        .skip(case.family.recipe().classes())
                        .all(|c| c.iter().all(|v| *v == 0.))
                );
                prework_negatives(
                    &mut session,
                    &mut packet,
                    if ni { &ictx } else { &nctx },
                    client,
                    &different.plan,
                );
            }
            if update == 1
                && matches!(case.family, SignalFamily::PolyFourRate)
                && (case.mode == i || case.mode == au)
            {
                later_rate_rejection(&mut session, &mut packet);
            }
            if case.poison {
                let before = context.counter();
                assert!(session.step(&packet).is_err());
                assert!(session.is_poisoned());
                assert_eq!(session.completed_updates(), 0);
                assert_eq!(session.processed_samples(), 0);
                let failure = session.failed_update().unwrap();
                assert_eq!(failure.update, 0);
                assert_eq!(failure.phase, "trainer");
                if ni {
                    // NI publishes self.trace only at final commit. This bootstrap
                    // failure precedes commit, so no attempted trace is exposed.
                    assert!(session.trace().is_empty());
                    assert!(failure.trace.is_empty());
                } else {
                    // I publishes its attempted trace before client refresh begins.
                    assert!(!failure.trace.is_empty());
                    assert_eq!(failure.trace, session.trace());
                }
                assert_eq!(failure.measured, &context.counter() - &before);
                assert_ne!(failure.measured, OpCounter::default());
                if ni {
                    assert_eq!(context.bootstrap.as_ref().unwrap().successful.get(), 1);
                } else {
                    assert_eq!(refresh.successful.get(), 1);
                }
                let stopped = ReadOnlySessionSnapshot::capture(&session);
                assert_eq!(
                    session.step(&packet).unwrap_err(),
                    mismatch("poisoned session")
                );
                stopped.assert_unchanged(&session);
                break;
            }
            let completion = session.step(&packet).unwrap();
            successful += 1;
            assert_eq!(completion.completed_updates, update + 1);
            assert_eq!(session.completed_updates(), update + 1);
            let expected_samples: usize = case.plan.parts[..=update]
                .iter()
                .flatten()
                .map(|p| p.valid)
                .sum();
            assert_eq!(completion.processed_samples, expected_samples);
            assert_eq!(session.processed_samples(), expected_samples);
            compare(
                client,
                &context,
                &session,
                oracle.as_ref().unwrap(),
                update,
                ni,
            );
            let state = ReadOnlySessionSnapshot::capture(&session);
            let expected = if session.completed_updates() == case.plan.parts.len() {
                "exhausted session"
            } else {
                "update index or replay"
            };
            assert_eq!(session.step(&packet).unwrap_err(), mismatch(expected));
            state.assert_unchanged(&session);
            assert!(!session.is_poisoned());
        }
    }
    assert_eq!(successful, 32);
    assert_eq!(runtime.attempts.get(), 34);
    assert_eq!(runtime.f64_updates.get(), 32);
    assert_eq!(runtime.oracles.get(), 18);
    assert_eq!(runtime.sessions.get(), 20);
    assert_eq!(runtime.parameters.get(), 2);
    assert_eq!(runtime.keys.get(), 2);
    assert_eq!(runtime.proofs.get(), 1);
    assert_eq!(runtime.descriptors.get(), 1);
    assert_eq!(runtime.state_snapshots.get(), 54);
    assert_eq!(runtime.state_assertions.get(), 264);
    assert_eq!(runtime.bootstrappers.get(), 1);
    assert_eq!(runtime.encoders.get(), 22);
    assert!(runtime.boots.get() <= 85);
    eprintln!(
        "P4_FINAL successes={successful} attempts={} actual_boots={} F64={} planner={:?}",
        runtime.attempts.get(),
        runtime.boots.get(),
        runtime.f64_updates.get(),
        pure.usage()
    );
    ACTIVE.with(|b| *b.borrow_mut() = None);
}
