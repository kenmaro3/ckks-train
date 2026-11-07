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
fn spy_contract(plan: &Arc<SyntheticCpuPlan>, p: PlanningParameters<'_>, budget: &RuntimeBudget) {
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
                        parts: &[Vec<MicroPart>],
                        field: &str| {
        let calls = Cell::new(0);
        let result = prepare_then_make(
            PreparationInput::Synthetic {
                plan: plan.clone(),
                claim,
            },
            p,
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
    rejected(&wrong, &plan.parts, "identity");
    let mut wrong = plan.claim();
    wrong.binding.push(0);
    rejected(&wrong, &plan.parts, "identity");
    for which in 0..7 {
        rejected(
            &plan.bad_parameter_claim(p, which),
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
        rejected(&claim, parts, "partition");
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
fn prework_negatives(
    s: &mut CpuDiagonalSession<'_>,
    packet: &mut EncryptedLogicalUpdate,
    other: &CpuEvaluationContext,
    client: &StudyClient,
    other_plan: &Arc<SyntheticCpuPlan>,
) {
    let before = s.context.counter();
    let progress = (s.completed_updates(), s.processed_samples());
    let snapshot = client.inspect_snapshot(&s.snapshot(), s.context).unwrap();
    let reject = |s: &mut CpuDiagonalSession<'_>, p: &EncryptedLogicalUpdate, expected: &str| {
        let error = s.step(p).unwrap_err();
        assert_eq!(error, mismatch(expected));
        assert_eq!(s.context.counter(), before);
        assert!(!s.is_poisoned());
        assert_eq!((s.completed_updates(), s.processed_samples()), progress);
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
    spy_contract(&cases[1].plan, params(wn), &runtime);
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
            if case.poison {
                let before = context.counter();
                assert!(session.step(&packet).is_err());
                assert!(session.is_poisoned());
                assert_eq!(session.completed_updates(), 0);
                assert_eq!(session.processed_samples(), 0);
                let failure = session.failed_update().unwrap();
                assert_eq!(failure.phase, "trainer");
                assert_eq!(failure.measured, &context.counter() - &before);
                assert_ne!(failure.measured, OpCounter::default());
                if ni {
                    assert_eq!(context.bootstrap.as_ref().unwrap().successful.get(), 1);
                } else {
                    assert_eq!(refresh.successful.get(), 1);
                }
                let stopped = context.counter();
                assert_eq!(
                    session.step(&packet).unwrap_err(),
                    mismatch("poisoned session")
                );
                assert_eq!(stopped, context.counter());
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
            let before = context.counter();
            assert!(session.step(&packet).is_err());
            assert_eq!(before, context.counter());
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
