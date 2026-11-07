//! Real CPU fixture definitions. No test here is authorized merely by compiling this file.
use super::*;
use crate::{
    activation::{Activation, Identity, PolyRelu2, Square},
    optim::{Nag, Sgd},
    refresh::ClientRefresher,
    trainer::{LayerOptimizer, Mlp, Trainer},
    trainer_ni::{NiOptimizer, NonInteractiveTrainer},
};
use ckks_train_ckks::bootstrap::metadata::DftSupportProof;
use ckks_train_ckks::bootstrap::{BootParams, Bootstrapper};
use ckks_train_ckks::{
    BootstrapOps, Ciphertext, Decryptor, Encoder, Encryptor, Evaluator, HeOps, KeyGenerator,
};
use ckks_train_core::sampler::Rng;
use ndarray::Array2;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

pub(crate) fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}
pub(crate) struct Context {
    pub p: Arc<CkksParams>,
    pub enc: Encoder,
    pub encrypt: Encryptor,
    pub dec: Decryptor,
    pub ev: Evaluator,
    pub bs: Option<Bootstrapper>,
    pub bm: Option<BootstrapMetadata>,
    pub layout: Layout,
}
impl Context {
    fn create(ni: bool, coarse: bool) -> Self {
        let p = Arc::new(if coarse {
            CkksParams::new(12, &[60, 20, 20], &[60], 20, 3).unwrap()
        } else if ni {
            let (q, p) = ckks_train_ckks::params::n16_boot_bits();
            CkksParams::new(10, &q, &p, 42, 8).unwrap()
        } else {
            CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap()
        });
        let layout = Layout::new(p.slots(), if coarse { 4 } else { 16 }, 8).unwrap();
        let enc = Encoder::new(p.clone());
        let mut kg = KeyGenerator::new(p.clone(), rng(51)).unwrap();
        let mut keys = vec![];
        for width in if coarse {
            vec![2, 4]
        } else {
            vec![3, 2, 10, 4, 16]
        } {
            let dims = if coarse {
                vec![2, 2]
            } else {
                vec![2, 4, width]
            };
            let acts = vec![DiagonalActivation::Square; dims.len() - 1];
            for trans in [
                TransposeKeys::Dedicated,
                TransposeKeys::Reuse(crate::trainer::mlp_key_pool(&dims, &layout).unwrap()),
            ] {
                let mode = if ni {
                    DiagonalMode::NonInteractive(BootPlacement::AfterUpdate)
                } else {
                    DiagonalMode::Interactive
                };
                let g =
                    plan_stage_geometry(&dims, &layout, &acts, &trans, mode, EvalConfig::OPTIMIZED)
                        .unwrap();
                keys.extend_from_slice(g.rotation_steps());
            }
        }
        if !coarse {
            let g = plan_stage_geometry(
                &[4, 4, 4],
                &layout,
                &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
                &TransposeKeys::Dedicated,
                if ni {
                    DiagonalMode::NonInteractive(BootPlacement::AfterUpdate)
                } else {
                    DiagonalMode::Interactive
                },
                EvalConfig::OPTIMIZED,
            )
            .unwrap();
            keys.extend_from_slice(g.rotation_steps());
        }
        if coarse {
            keys.extend(crate::compact::extract_steps(
                &crate::compact::CompactPlan::for_layer(&layout, 2, 2),
                0,
            ));
        }
        keys = keys
            .into_iter()
            .map(|s| (s as i64).rem_euclid(p.slots() as i64) as i32)
            .filter(|&s| s != 0)
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let (bs, bm) = if ni {
            let bp = BootParams {
                log_slots: 9,
                ..BootParams::default_n16()
            };
            let proof = DftSupportProof::verify_current().unwrap();
            let bm = BootstrapMetadata::new_with_verified_support(&p, &bp, &proof).unwrap();
            let bs = Bootstrapper::new(p.clone(), bp, &enc, &mut kg).unwrap();
            assert_eq!(&bs.cost_per_boot(), bm.cost_per_boot());
            (Some(bs), Some(bm))
        } else {
            (None, None)
        };
        let encrypt = Encryptor::new(p.clone(), kg.public_key());
        let dec = Decryptor::new(p.clone(), kg.secret_key());
        let ev = Evaluator::new(p.clone(), Some(kg.relin_key()), Some(kg.galois_keys(&keys)));
        assert_eq!(ev.counter(), OpCounter::default());
        Self {
            p,
            enc,
            encrypt,
            dec,
            ev,
            bs,
            bm,
            layout,
        }
    }
    fn evaluator(&self, cfg: EvalConfig) -> Evaluator {
        Evaluator::with_config(
            self.p.clone(),
            self.ev.relin_key().cloned(),
            self.ev.galois_keys().cloned(),
            cfg,
        )
    }
    pub(crate) fn encrypt_at(&self, l: usize, s: f64, value: f64) -> Ciphertext {
        self.encrypt.encrypt(
            &self.enc.encode(&vec![value; self.p.slots()], s, l).unwrap(),
            &mut rng(52),
        )
    }
    fn client(&self) -> ClientRefresher {
        ClientRefresher::new(
            self.dec.clone(),
            self.encrypt.clone(),
            self.enc.clone(),
            rng(53),
        )
    }
}
pub(crate) fn context(ni: bool) -> &'static Context {
    static I: OnceLock<Context> = OnceLock::new();
    static N: OnceLock<Context> = OnceLock::new();
    if ni {
        N.get_or_init(|| Context::create(true, false))
    } else {
        I.get_or_init(|| Context::create(false, false))
    }
}
pub(crate) fn coarse() -> &'static Context {
    static C: OnceLock<Context> = OnceLock::new();
    C.get_or_init(|| Context::create(false, true))
}
static TOTAL_BOOTS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_STAGE_CALLS: AtomicUsize = AtomicUsize::new(0);
static TOTAL_HELPER_CALLS: AtomicUsize = AtomicUsize::new(0);
fn reserve_stage_calls(count: usize) {
    let before = TOTAL_STAGE_CALLS.fetch_add(count, Ordering::SeqCst);
    eprintln!(
        "P2_RESERVE_STAGE test={} before={before} requested={count} after={} cap=99",
        std::thread::current().name().unwrap_or("unnamed"),
        before.saturating_add(count)
    );
    assert!(
        before.checked_add(count).is_some_and(|n| n <= 99),
        "new-matrix stage-call cap"
    );
}
pub(crate) fn reserve_helper_calls(count: usize) {
    let before = TOTAL_HELPER_CALLS.fetch_add(count, Ordering::SeqCst);
    eprintln!(
        "P2_RESERVE_HELPER test={} before={before} requested={count} after={} cap=15",
        std::thread::current().name().unwrap_or("unnamed"),
        before.saturating_add(count)
    );
    assert!(
        before.checked_add(count).is_some_and(|n| n <= 15),
        "new-matrix helper-call cap"
    );
}

struct Guard<'a> {
    bs: &'a Bootstrapper,
    used: Cell<usize>,
    limit: usize,
}
impl BootstrapOps<Evaluator> for Guard<'_> {
    fn output_level(&self) -> usize {
        self.bs.output_level()
    }
    fn bootstrap(
        &self,
        e: &Evaluator,
        c: &Ciphertext,
    ) -> ckks_train_core::error::Result<Ciphertext> {
        // Both caps are checked before invoking the real bootstrap, including failed calls.
        eprintln!(
            "P2_BOOT_RESERVE test={} local_before={} local_cap={} global_before={} global_cap=266",
            std::thread::current().name().unwrap_or("unnamed"),
            self.used.get(),
            self.limit,
            TOTAL_BOOTS.load(Ordering::SeqCst)
        );
        assert!(self.used.get() < self.limit, "per-case bootstrap cap");
        let before = TOTAL_BOOTS.fetch_add(1, Ordering::SeqCst);
        assert!(before < 266, "new-matrix total bootstrap cap");
        self.used.set(self.used.get() + 1);
        eprintln!(
            "P2_BOOT_ATTEMPT global_ordinal={} local_ordinal={} input_level={} input_scale_bits={:#018x}",
            before + 1,
            self.used.get(),
            c.level(),
            c.scale().to_bits()
        );
        let counter_origin = e.counter();
        let result = self.bs.bootstrap(e, c);
        eprintln!(
            "P2_BOOT_RETURN global_ordinal={} local_ordinal={} ok={} error={:?} counter_delta={:?}",
            before + 1,
            self.used.get(),
            result.is_ok(),
            result.as_ref().err(),
            &e.counter() - &counter_origin
        );
        result
    }
}
fn acts(poly: bool) -> Vec<Box<dyn Activation>> {
    if poly {
        vec![Box::new(PolyRelu2), Box::new(Identity)]
    } else {
        vec![Box::new(Square), Box::new(Square)]
    }
}
fn layers(dims: &[usize], frozen: bool) -> Vec<(Array2<f64>, Vec<f64>)> {
    dims.windows(2)
        .map(|d| {
            (
                Array2::from_shape_fn((d[1], d[0]), |(i, j)| {
                    (((i + 3 * j + 1) % 7) as f64 - 3.) * 2f64.powi(-8)
                }),
                (0..d[1])
                    .map(|i| {
                        if frozen {
                            0.
                        } else {
                            ((i % 3) as f64 - 1.) * 2f64.powi(-8)
                        }
                    })
                    .collect(),
            )
        })
        .collect()
}
fn input(
    c: &Context,
    s: BatchMetadata,
    valid: usize,
    width: usize,
    real: usize,
) -> (Ciphertext, Ciphertext) {
    let x = Array2::from_shape_fn((valid, 2), |(i, j)| {
        (((i + 2 * j) % 5) as f64 - 2.) * 2f64.powi(-4)
    });
    let y = Array2::from_shape_fn((valid, width), |(i, j)| f64::from(j == i % real));
    let mut xp = crate::layout::pack_activations(&x, &c.layout);
    let mut yp = crate::layout::pack_activations(&y, &c.layout);
    for at in 0..c.p.slots() {
        if at % c.layout.batch >= valid {
            xp[at] = 2f64.powi(-5);
            yp[at] = 2f64.powi(-5);
        }
    }
    let mut r = rng(52);
    (
        c.encrypt.encrypt(
            &c.enc.encode(&xp, s.x.scale(), s.x.level()).unwrap(),
            &mut r,
        ),
        c.encrypt.encrypt(
            &c.enc.encode(&yp, s.y.scale(), s.y.level()).unwrap(),
            &mut r,
        ),
    )
}
fn ct_eq(m: CtMeta, c: &Ciphertext) {
    assert_eq!(
        (m.level(), m.scale_bits()),
        (c.level(), c.scale().to_bits())
    );
}
fn label(c: &Ciphertext) -> (usize, u64) {
    (c.level(), c.scale().to_bits())
}
fn meta_label(m: CtMeta) -> (usize, u64) {
    (m.level(), m.scale_bits())
}

fn compare_expanded(plan: &PlannedStageUpdate, actual: &Mlp) {
    let expected_weights = plan
        .state
        .w
        .iter()
        .enumerate()
        .map(|(i, &m)| {
            let dims = &plan.state.geometry.dims;
            let diagonals = dims[i]
                .next_power_of_two()
                .min(dims[i + 1].next_power_of_two());
            vec![meta_label(m); diagonals]
        })
        .collect::<Vec<_>>();
    let actual_weights = actual
        .layers
        .iter()
        .map(|(d, _)| d.w.diagonals().entries().iter().map(label).collect())
        .collect::<Vec<Vec<_>>>();
    compare_nested_labels(&expected_weights, &actual_weights);
    assert_eq!(
        actual
            .layers
            .iter()
            .map(|(d, _)| label(&d.b))
            .collect::<Vec<_>>(),
        plan.state
            .b
            .iter()
            .copied()
            .map(meta_label)
            .collect::<Vec<_>>(),
        "complete bias metadata"
    );
}

fn compare_nested_labels(expected: &[Vec<(usize, u64)>], actual: &[Vec<(usize, u64)>]) {
    assert_eq!(
        actual, expected,
        "complete outer/inner lengths and metadata"
    );
}

#[test]
fn p2_pure_nested_state_and_handoff_cardinality() {
    let expected = vec![vec![(7, 42); 2], vec![(8, 43); 4]];
    compare_nested_labels(&expected, &expected);
    for actual in [
        vec![],
        vec![expected[0].clone()],
        vec![expected[0][..1].to_vec(), expected[1].clone()],
        vec![expected[0].clone(), vec![(8, 43); 5]],
        vec![expected[0].clone(), expected[1].clone(), vec![]],
    ] {
        assert!(std::panic::catch_unwind(|| compare_nested_labels(&expected, &actual)).is_err());
    }
}
// A concrete metadata key identifies exactly one producer; ordering never resolves ambiguity.
fn planned_metadata_event<'a>(
    events: &'a [MetadataEvent],
    role: &str,
    micro: Option<usize>,
    layer: Option<usize>,
) -> &'a MetadataEvent {
    let mut matches = events
        .iter()
        .filter(|e| e.role == role && e.micro == micro && e.layer == layer);
    let expected = matches.next().unwrap_or_else(|| {
        panic!("missing planned metadata key role={role} micro={micro:?} layer={layer:?}")
    });
    assert!(
        matches.next().is_none(),
        "ambiguous planned metadata key role={role} micro={micro:?} layer={layer:?}"
    );
    expected
}

#[test]
fn p2_pure_loss_backward_role_identity() {
    let p = CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap();
    let g = plan_stage_geometry(
        &[2, 4, 4],
        &Layout::new(512, 16, 8).unwrap(),
        &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
        &TransposeKeys::Dedicated,
        DiagonalMode::Interactive,
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let state = initial_stage_metadata(
        &g,
        &p,
        None,
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Frozen,
    )
    .unwrap();
    let u = UpdateSpec::new(
        vec![MicroMetadata {
            n_valid: 8,
            x: state.batch.x,
            y: state.batch.y,
        }],
        8,
        0.08,
        None,
    );
    let plan = plan_stage_update(&g, &p, None, &state, &u).unwrap();
    let deltas = plan
        .trace()
        .iter()
        .filter(|e| matches!(e.role.as_str(), "delta" | "loss_delta" | "backward_delta"))
        .collect::<Vec<_>>();
    assert_eq!(deltas.len(), 2);
    assert_ne!(
        (&deltas[0].role, deltas[0].micro, deltas[0].layer),
        (&deltas[1].role, deltas[1].micro, deltas[1].layer),
        "distinct loss/backward values need distinct role keys"
    );
    let loss = planned_metadata_event(plan.trace(), "loss_delta", Some(0), Some(1));
    let backward = planned_metadata_event(plan.trace(), "backward_delta", Some(0), Some(1));
    assert_eq!((loss.metadata.level(), backward.metadata.level()), (6, 5));
    assert_ne!(loss.metadata.scale_bits(), backward.metadata.scale_bits());
}

#[test]
#[should_panic(expected = "ambiguous planned metadata key")]
fn p2_pure_lookup_rejects_ambiguous_role() {
    let p = CkksParams::new(10, &[60, 40, 40], &[60], 40, 3).unwrap();
    let e = MetadataEvent {
        role: "delta".into(),
        micro: Some(0),
        layer: Some(1),
        metadata: CtMeta::new(&p, 2, p.scale()).unwrap(),
        multiplicity: 1,
        counts: OpCounter::default(),
    };
    let duplicate = vec![e.clone(), e];
    let _ = planned_metadata_event(&duplicate, "delta", Some(0), Some(1));
}

#[test]
#[should_panic(expected = "missing planned metadata key")]
fn p2_pure_lookup_rejects_missing_role() {
    let _ = planned_metadata_event(&[], "loss_delta", Some(0), Some(1));
}

// Numeric fixtures exercise the observation comparator only; no HE is constructed.
fn pure_observation_plan(mode: DiagonalMode, nag: bool, later: bool) -> PlannedStageUpdate {
    pure_observation_plan_at(mode, nag, later, &[2, 4, 4], 8)
}
fn pure_observation_plan_at(
    mode: DiagonalMode,
    nag: bool,
    later: bool,
    dims: &[usize],
    lanes: usize,
) -> PlannedStageUpdate {
    use ckks_train_ckks::bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, DftSupportProof},
    };
    let (q, sp) = ckks_train_ckks::params::n16_boot_bits();
    let p = CkksParams::new(10, &q, &sp, 42, 8).unwrap();
    let boot = if mode == DiagonalMode::Interactive {
        None
    } else {
        Some(
            BootstrapMetadata::new_with_verified_support(
                &p,
                &BootParams {
                    log_slots: 9,
                    ..BootParams::default_n16()
                },
                &DftSupportProof::verify_current().unwrap(),
            )
            .unwrap(),
        )
    };
    let g = plan_stage_geometry(
        dims,
        &Layout::new(
            512,
            dims.iter()
                .map(|n| n.next_power_of_two())
                .max()
                .unwrap()
                .max(16),
            lanes,
        )
        .unwrap(),
        &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
        &TransposeKeys::Dedicated,
        mode,
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let opt = if nag {
        DiagonalOptimizer::Nag {
            lr: 0.04,
            reference: 0.08,
            momentum: 0.9,
        }
    } else {
        DiagonalOptimizer::Sgd { lr: 0.04 }
    };
    let mut state = initial_stage_metadata(&g, &p, boot.as_ref(), opt, BiasPolicy::Active).unwrap();
    let spec = |s: &StageState| {
        UpdateSpec::new(
            vec![MicroMetadata {
                n_valid: lanes,
                x: s.batch.x,
                y: s.batch.y,
            }],
            lanes,
            0.04,
            None,
        )
    };
    if later {
        state = plan_stage_update(&g, &p, boot.as_ref(), &state, &spec(&state))
            .unwrap()
            .state;
    }
    plan_stage_update(&g, &p, boot.as_ref(), &state, &spec(&state)).unwrap()
}

fn numeric_observations(plan: &PlannedStageUpdate) -> observation::Recorder {
    use observation::{Attempt, Coefficient, Observed};
    let mut r = observation::Recorder::default();
    let roles = [
        "input",
        "z",
        "a",
        "s",
        "dz",
        "dz_aligned",
        "aligned_target",
        "loss_delta",
        "backward_delta",
        "transpose",
        "gradient",
        "reference_gradient",
        "bias_gradient",
        "accumulated_gradient",
        "accumulated_reference",
        "accumulated_bias",
        "weight_updated",
        "velocity_updated",
        "mu_velocity",
        "kappa_velocity",
        "bias_updated",
        "weight_refreshed",
        "bias_refreshed",
        "velocity_refreshed",
        "compact_gradient",
        "compact_reference",
        "compact_bias_gradient",
        "weight_unpacked",
        "bias_extracted",
        "a_to_delta",
        "a_boot",
        "entry_input",
        "entry_target",
    ];
    for e in &plan.events {
        let (role, positions): (&str, Vec<usize>) =
            if let Some(tail) = e.role.strip_prefix("container") {
                let (ci, suffix) = tail.split_once('_').unwrap();
                let role = match suffix {
                    "updated" => "container_updated",
                    "velocity" => "container_velocity",
                    "mu_velocity" => "mu_velocity",
                    "kappa_velocity" => "kappa_velocity",
                    _ => continue,
                };
                (role, vec![ci.parse().unwrap()])
            } else if roles.contains(&e.role.as_str()) {
                (e.role.as_str(), (0..e.multiplicity).collect())
            } else {
                continue;
            };
        // This fixture owns numeric records, not runtime ciphertexts.
        let role: &'static str = roles
            .iter()
            .copied()
            .chain(["container_updated", "container_velocity"])
            .find(|r| *r == role)
            .unwrap();
        for position in positions {
            r.values.push(Observed {
                role,
                micro: e.micro,
                layer: e.layer,
                position,
                level: e.metadata.level(),
                scale_bits: e.metadata.scale_bits(),
            });
        }
    }
    for layer in 0..plan.state.w.len() {
        for (role, bits) in [
            ("current", Some(plan.coefficients.current_bits)),
            ("bias", Some(plan.coefficients.bias_bits)),
            ("reference", plan.coefficients.reference_bits),
        ] {
            if let Some(bits) = bits {
                r.coefficients.push(Coefficient {
                    role,
                    layer: Some(layer),
                    position: 0,
                    bits,
                });
            }
        }
    }
    let velocity_positions = if plan.state.geometry.mode == DiagonalMode::Interactive {
        (0..plan.state.w.len())
            .map(|l| (Some(l), 0))
            .collect::<Vec<_>>()
    } else if plan.first {
        vec![]
    } else {
        plan.events
            .iter()
            .filter_map(|e| {
                e.role
                    .strip_prefix("container")
                    .and_then(|tail| tail.strip_suffix("_mu_velocity"))
                    .map(|ci| (e.layer, ci.parse::<usize>().unwrap()))
            })
            .collect()
    };
    for (layer, position) in velocity_positions {
        for (role, bits) in [
            ("mu", plan.coefficients.velocity_mu_bits),
            ("kappa", plan.coefficients.velocity_kappa_bits),
        ] {
            if let Some(bits) = bits {
                r.coefficients.push(Coefficient {
                    role,
                    layer,
                    position,
                    bits,
                });
            }
        }
    }
    for a in &plan.attempts {
        let observed_role = if plan.state.geometry.mode == DiagonalMode::Interactive {
            "bias_gradient"
        } else {
            "bias_extracted"
        };
        if a.operation != observed_role {
            continue;
        }
        let role = match a.operation.as_str() {
            "bias_gradient" => "bias_gradient",
            "bias_extracted" => "bias_extracted",
            _ => continue,
        };
        r.attempts.push(Attempt {
            role,
            micro: a.micro,
            layer: a.layer,
            source_level: a.source.level(),
            source_scale_bits: a.source.scale_bits(),
            target_bits: a.target_bits,
            error: a.error.clone(),
        });
    }
    r
}

fn rejected(plan: &PlannedStageUpdate, r: &observation::Recorder) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compare_observed(plan, r))).is_err()
}

#[test]
fn p2_pure_comparator_rejects_missing_duplicate_and_wrong_positions() {
    for mode in [
        DiagonalMode::Interactive,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
    ] {
        let p = pure_observation_plan(mode, true, true);
        let fixture = numeric_observations(&p);
        compare_observed(&p, &fixture);
        let at = fixture
            .values
            .iter()
            .position(|v| v.role == "gradient")
            .unwrap();
        for mutation in 0..3 {
            let mut r = numeric_observations(&p);
            match mutation {
                0 => {
                    r.values.remove(at);
                }
                1 => r.values.push(r.values[at].clone()),
                _ => r.values[at].position = usize::MAX,
            }
            assert!(
                rejected(&p, &r),
                "accepted position mutation{mutation} {mode:?}"
            );
        }
    }
}

#[test]
fn p2_pure_comparator_requires_coefficients_retries_and_prefresh_values() {
    for mode in [
        DiagonalMode::Interactive,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
    ] {
        for later in [false, true] {
            let p = pure_observation_plan(mode, true, later);
            compare_observed(&p, &numeric_observations(&p));
            for role in ["current", "bias", "reference", "mu", "kappa"] {
                let mut r = numeric_observations(&p);
                if let Some(at) = r.coefficients.iter().position(|v| v.role == role) {
                    r.coefficients.remove(at);
                    assert!(
                        rejected(&p, &r),
                        "accepted missing coefficient {role} {mode:?} later{later}"
                    );
                }
            }
            let mut r = numeric_observations(&p);
            r.coefficients
                .iter_mut()
                .find(|v| v.role == "current" && v.layer == Some(1))
                .unwrap()
                .layer = Some(0);
            assert!(rejected(&p, &r), "accepted duplicate coefficient layer");
            for mutation in 0..3 {
                let mut r = numeric_observations(&p);
                match mutation {
                    0 => {
                        r.attempts.remove(0);
                    }
                    1 => r.attempts.push(r.attempts[0].clone()),
                    _ => r.attempts.swap(0, 1),
                }
                assert!(rejected(&p, &r), "accepted retry mutation{mutation}");
            }
            for role in ["bias_updated", "container_updated", "container_velocity"] {
                let mut r = numeric_observations(&p);
                if r.values.iter().any(|v| v.role == role) {
                    r.values.retain(|v| v.role != role);
                    assert!(rejected(&p, &r), "accepted missing pre-refresh {role}");
                }
            }
        }
    }
}

#[test]
fn p2_pure_receipt_vector_multiplicities_and_entry_roles() {
    let p = pure_observation_plan(DiagonalMode::Interactive, true, false);
    for (layer, m) in [(0, 2), (1, 4)] {
        for role in ["accumulated_gradient", "accumulated_reference"] {
            assert_eq!(
                planned_metadata_event(p.trace(), role, Some(0), Some(layer)).multiplicity,
                m
            );
        }
    }
    for role in ["entry_input", "entry_target"] {
        assert_eq!(
            planned_metadata_event(p.trace(), role, Some(0), None).multiplicity,
            1
        );
    }
}

#[test]
fn p2_pure_entry_receipt_is_required_separately() {
    for mode in [
        DiagonalMode::Interactive,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
    ] {
        let p = pure_observation_plan(mode, false, false);
        for role in ["entry_input", "entry_target"] {
            let e = planned_metadata_event(p.trace(), role, Some(0), None);
            assert_eq!(e.multiplicity, 1);
            let mut r = numeric_observations(&p);
            r.values.retain(|v| v.role != role);
            assert!(rejected(&p, &r), "accepted missing {role}");
        }
        let mut r = numeric_observations(&p);
        let x = r
            .values
            .iter()
            .position(|v| v.role == "entry_input")
            .unwrap();
        let y = r
            .values
            .iter()
            .position(|v| v.role == "entry_target")
            .unwrap();
        let x_label = (r.values[x].level, r.values[x].scale_bits);
        r.values[x].level = r.values[y].level;
        r.values[x].scale_bits = r.values[y].scale_bits;
        (r.values[y].level, r.values[y].scale_bits) = x_label;
        assert!(rejected(&p, &r), "accepted swapped raw entry labels");
    }
}

#[test]
fn p2_pure_missing_retry_is_not_equality() {
    let p = pure_observation_plan(DiagonalMode::Interactive, false, false);
    let mut r = numeric_observations(&p);
    assert_eq!(r.attempts.len(), 2);
    r.attempts.remove(0);
    assert!(rejected(&p, &r));
}

#[test]
fn p2_pure_missing_container_is_not_equality() {
    let p = pure_observation_plan(
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        true,
        true,
    );
    for role in ["container_updated", "container_velocity"] {
        let mut r = numeric_observations(&p);
        assert!(r.values.iter().any(|v| v.role == role));
        r.values.retain(|v| v.role != role);
        assert!(rejected(&p, &r), "accepted missing {role}");
    }
}

#[test]
fn p2_pure_coefficient_branch_and_container_positions_are_exact() {
    for mode in [
        DiagonalMode::Interactive,
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
    ] {
        for nag in [false, true] {
            for later in [false, true] {
                let p = pure_observation_plan(mode, nag, later);
                let mut r = numeric_observations(&p);
                compare_observed(&p, &r);
                let mu = r.coefficients.iter().filter(|v| v.role == "mu").count();
                assert_eq!(
                    mu,
                    if nag && (mode == DiagonalMode::Interactive || later) {
                        2
                    } else {
                        0
                    }
                );
                if let Some(v) = r.coefficients.iter_mut().find(|v| v.role == "mu") {
                    v.position = usize::MAX;
                } else {
                    r.coefficients.push(observation::Coefficient {
                        role: "mu",
                        layer: Some(0),
                        position: 0,
                        bits: 0.9f64.to_bits(),
                    });
                }
                assert!(
                    rejected(&p, &r),
                    "accepted wrong velocity coefficient branch/position"
                );
            }
        }
    }
    //32 diagonals /16 positions per compact ciphertext: two W containers per layer, plus bias-only storage.
    let p = pure_observation_plan_at(
        DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        true,
        true,
        &[32, 32, 32],
        8,
    );
    let r = numeric_observations(&p);
    assert_eq!(
        r.values.iter().filter(|v| v.role == "mu_velocity").count(),
        4
    );
    assert_eq!(r.coefficients.iter().filter(|v| v.role == "mu").count(), 4);
    assert_eq!(
        p.events
            .iter()
            .filter(|e| e.role.ends_with("_mu_velocity"))
            .count(),
        4
    );
    compare_observed(&p, &r);
    let mut bad = numeric_observations(&p);
    let at = bad
        .coefficients
        .iter()
        .position(|v| v.role == "mu" && v.position == 1)
        .unwrap();
    bad.coefficients[at].position = 0;
    assert!(rejected(&p, &bad));
}

// Literal pure-only witness discovered once under the frozen F3 recipe. No HE/profile admission.
#[test]
fn p2_pure_structural_no_boot_then_exact_plain_minimum() {
    use ckks_train_ckks::bootstrap::metadata::BootstrapMetadata;
    let q_bits = [
        vec![31],
        vec![20; 16],
        vec![34; 4],
        vec![58; 7],
        vec![56; 4],
    ]
    .concat();
    let p = CkksParams::new(8, &q_bits, &[60; 4], 19, 8).unwrap();
    assert_eq!(
        p.q(),
        &[
            2147483137,
            986113,
            993793,
            994817,
            995329,
            997889,
            1004033,
            1009153,
            1013249,
            1017857,
            1021441,
            1022977,
            1025537,
            1032193,
            1038337,
            1043969,
            1047041,
            17179848193,
            17179855361,
            17179858433,
            17179859969,
            288230376151647233,
            288230376151661569,
            288230376151683073,
            288230376151687681,
            288230376151690241,
            288230376151691777,
            288230376151697921,
            72057594037895681,
            72057594037896193,
            72057594037897217,
            72057594037913089
        ]
    );
    assert_eq!(
        p.p(),
        &[
            1152921504606815233,
            1152921504606827009,
            1152921504606830593,
            1152921504606844417
        ]
    );
    assert_eq!(p.scale().to_bits(), 4692750811720056832);
    assert!(
        p.q()[1..=16]
            .iter()
            .all(|&q| q as f64 >= 1.5 * p.scale() && q < (1 << 20))
    );
    let proof = DftSupportProof::verify_current().unwrap();
    let boot = BootstrapMetadata::new_with_verified_support(
        &p,
        &BootParams {
            log_slots: 7,
            ..BootParams::default_n16()
        },
        &proof,
    )
    .unwrap();
    assert_eq!(boot.output().level(), 16);
    let g = plan_stage_geometry(
        &[2, 4, 4],
        &Layout::new(128, 16, 8).unwrap(),
        &[DiagonalActivation::Identity; 2],
        &TransposeKeys::Dedicated,
        DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
        EvalConfig::OPTIMIZED,
    )
    .unwrap();
    let state = initial_stage_metadata(
        &g,
        &p,
        Some(&boot),
        DiagonalOptimizer::Sgd { lr: 0.08 },
        BiasPolicy::Active,
    )
    .unwrap();
    assert_eq!(meta_label(state.batch.x), (15, 4692750811720056832));
    assert_eq!(meta_label(state.batch.y), (13, 4683808092495075774));
    let spec = UpdateSpec::new(
        vec![MicroMetadata {
            n_valid: 8,
            x: state.batch.x,
            y: state.batch.y,
        }],
        8,
        0.08,
        Some(true),
    );
    let e = plan_stage_update(&g, &p, Some(&boot), &state, &spec).unwrap_err();
    let minimum = HekitError::InvalidParams("plaintext scale 1.021441e6 is below 2^20: the target scale is too small for this ciphertext's scale and level (the constant would be rounded to a few bits)".into());
    assert_eq!(e.kind, PlanFailure::ScaleInadmissible);
    assert_eq!(
        (e.operation.as_str(), e.micro, e.layer, e.update),
        ("weight_unpacked", None, Some(0), Some(0))
    );
    assert_eq!(e.source, Some(minimum.clone()));
    assert_eq!(e.prime, Some(1021441));
    assert_eq!(
        (e.source_scale_bits, e.target_scale_bits),
        (Some(4692750811720056832), Some(4692750811720056832))
    );
    let expected_trace = vec![
        (
            "entry_input",
            Some(0),
            None,
            15,
            4692750811720056832,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "entry_target",
            Some(0),
            None,
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "input",
            Some(0),
            Some(0),
            15,
            4692750811720056832,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "z",
            Some(0),
            Some(0),
            14,
            4688267086326004850,
            1,
            (1, 1, 2, 0, 1, 0, 0),
        ),
        (
            "a",
            Some(0),
            Some(0),
            14,
            4688267086326004850,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "input",
            Some(0),
            Some(1),
            14,
            4688267086326004850,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "z",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (2, 2, 4, 0, 1, 0, 0),
        ),
        (
            "a",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "aligned_target",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "loss_delta",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "dz",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "dz_aligned",
            Some(0),
            Some(1),
            13,
            4683808092495075774,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "transpose",
            None,
            Some(1),
            13,
            4692750811720056832,
            4,
            (3, 0, 0, 0, 0, 0, 0),
        ),
        (
            "backward_delta",
            Some(0),
            Some(1),
            12,
            4679376997572758275,
            1,
            (2, 2, 4, 0, 1, 0, 0),
        ),
        (
            "gradient_product",
            None,
            Some(1),
            12,
            4674893876687722636,
            4,
            (14, 4, 4, 0, 4, 0, 0),
        ),
        (
            "gradient",
            None,
            Some(1),
            11,
            4692750811720056832,
            4,
            (0, 0, 0, 4, 4, 0, 0),
        ),
        (
            "compact_gradient",
            None,
            Some(1),
            11,
            4692750811720056832,
            1,
            (3, 0, 0, 0, 0, 0, 0),
        ),
        (
            "bias_gradient",
            None,
            Some(1),
            12,
            4692750811720056832,
            1,
            (3, 0, 0, 1, 1, 0, 0),
        ),
        (
            "compact_bias_gradient",
            None,
            Some(1),
            12,
            4692750811720056832,
            1,
            (1, 0, 0, 0, 0, 0, 0),
        ),
        (
            "dz",
            Some(0),
            Some(0),
            12,
            4679376997572758275,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "dz_aligned",
            Some(0),
            Some(0),
            12,
            4679376997572758275,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "gradient_product",
            None,
            Some(0),
            11,
            4674977650081703040,
            2,
            (7, 2, 2, 0, 2, 0, 0),
        ),
        (
            "gradient",
            None,
            Some(0),
            10,
            4692750811720056832,
            2,
            (0, 0, 0, 2, 2, 0, 0),
        ),
        (
            "compact_gradient",
            None,
            Some(0),
            10,
            4692750811720056832,
            1,
            (1, 0, 0, 0, 0, 0, 0),
        ),
        (
            "bias_gradient",
            None,
            Some(0),
            11,
            4692750811720056832,
            1,
            (3, 0, 0, 1, 1, 0, 0),
        ),
        (
            "compact_bias_gradient",
            None,
            Some(0),
            11,
            4692750811720056832,
            1,
            (1, 0, 0, 0, 0, 0, 0),
        ),
        (
            "container0_updated",
            None,
            Some(0),
            10,
            4692750811720056832,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
        (
            "container1_updated",
            None,
            Some(1),
            11,
            4692750811720056832,
            1,
            (0, 0, 0, 0, 0, 0, 0),
        ),
    ];
    let trace = e
        .trace
        .iter()
        .map(|e| {
            (
                e.role.as_str(),
                e.micro,
                e.layer,
                e.metadata.level(),
                e.metadata.scale_bits(),
                e.multiplicity,
                (
                    e.counts.rotations,
                    e.counts.relins,
                    e.counts.ct_ct_tensor,
                    e.counts.ct_pt_mults,
                    e.counts.rescales,
                    e.counts.sse_switches,
                    e.counts.bootstraps,
                ),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        trace, expected_trace,
        "complete ordered reachable pre-error trace"
    );
    let expected_attempts = vec![
        (
            "gradient",
            None,
            Some(1),
            12,
            4674893876687722636,
            4692750811720056832,
            None,
        ),
        (
            "bias_gradient",
            None,
            Some(1),
            13,
            4683808092495075774,
            4692750811720056832,
            None,
        ),
        (
            "gradient",
            None,
            Some(0),
            11,
            4674977650081703040,
            4692750811720056832,
            None,
        ),
        (
            "bias_gradient",
            None,
            Some(0),
            12,
            4679376997572758275,
            4692750811720056832,
            None,
        ),
        (
            "weight_unpacked",
            None,
            Some(0),
            10,
            4692750811720056832,
            4692750811720056832,
            Some(minimum.clone()),
        ),
    ];
    assert_eq!(
        e.attempts
            .iter()
            .map(|a| (
                a.operation.as_str(),
                a.micro,
                a.layer,
                a.source.level(),
                a.source.scale_bits(),
                a.target_bits,
                a.error.clone()
            ))
            .collect::<Vec<_>>(),
        expected_attempts
    );
    let cl = g.compact.as_ref().unwrap();
    let c = (0..cl.containers.len())
        .map(|ci| {
            planned_metadata_event(
                &e.trace,
                &format!("container{ci}_updated"),
                None,
                cl.containers[ci].layer,
            )
            .metadata
            .level() as i64
        })
        .collect::<Vec<_>>();
    let u = (0..cl.containers.len())
        .map(|ci| {
            e.trace
                .iter()
                .find(|ev| ev.role == format!("container{ci}_velocity"))
                .map(|ev| ev.metadata.level() as i64)
        })
        .collect::<Vec<_>>();
    let (w, b) = unpacked(cl, &c);
    let structural = LevelState { w, b, c, u };
    assert_eq!(
        (&structural.w, &structural.b, &structural.c, &structural.u),
        (&vec![9, 10], &vec![9, 9], &vec![10, 11], &vec![None, None])
    );
    let acts = [DiagonalActivation::Identity; 2]
        .into_iter()
        .map(act_meta)
        .collect::<Vec<_>>();
    assert!(!must_boot(
        &acts,
        BootPlacement::WhenNeeded,
        &structural,
        15,
        16,
        Some(true)
    ));
    assert!(!e.trace.iter().any(|ev| (ev.role.starts_with("container")
        || ev.role.starts_with("velocity"))
        && ev.role.ends_with("_boot")));
    // The exact label EXISTS. The subsequent plaintext minimum, not a ULP miss, rejects it.
    let product = ckks_train_ckks::metadata::exact_product_scale_at(&p, 10, p.scale()).unwrap();
    assert_eq!(product / p.scale(), 1021441.0);
    assert!(product / p.scale() < ckks_train_ckks::evaluator::MIN_PLAIN_SCALE);
    assert_eq!(
        target_plain_scale_at(&p, 10, p.scale(), p.scale()),
        Err(minimum)
    );
}

fn compare_observed(plan: &PlannedStageUpdate, observed: &observation::Recorder) {
    use std::collections::BTreeMap;
    let roles = [
        "entry_input",
        "entry_target",
        "input",
        "z",
        "a",
        "s",
        "dz",
        "dz_aligned",
        "aligned_target",
        "loss_delta",
        "backward_delta",
        "transpose",
        "gradient",
        "reference_gradient",
        "bias_gradient",
        "accumulated_gradient",
        "accumulated_reference",
        "accumulated_bias",
        "weight_updated",
        "velocity_updated",
        "bias_updated",
        "mu_velocity",
        "kappa_velocity",
        "weight_refreshed",
        "bias_refreshed",
        "velocity_refreshed",
        "compact_gradient",
        "compact_reference",
        "compact_bias_gradient",
        "weight_unpacked",
        "bias_extracted",
        "a_to_delta",
        "a_boot",
    ];
    let mode = plan.state.geometry.mode;
    let (slots, d, batch) = plan.state.geometry.layout;
    let compact = if mode == DiagonalMode::Interactive {
        None
    } else {
        Some(
            CompactLayout::new(
                &plan.state.geometry.dims,
                &Layout::new(slots, d, batch).unwrap(),
            )
            .unwrap(),
        )
    };
    let mut expected = BTreeMap::new();
    for e in plan.trace() {
        let (role, positions) = if let Some(tail) = e.role.strip_prefix("container") {
            let (index, suffix) = tail.split_once('_').expect("container receipt suffix");
            let role = match suffix {
                "updated" => "container_updated",
                "velocity" => "container_velocity",
                "mu_velocity" => "mu_velocity",
                "kappa_velocity" => "kappa_velocity",
                _ => continue,
            };
            let ci = index.parse::<usize>().expect("numeric container position");
            let c = &compact.as_ref().expect("NI container").containers[ci];
            assert_eq!(e.layer, c.layer, "container owner layer");
            assert_eq!(e.multiplicity, 1);
            if role != "container_updated" {
                assert!(c.layer.is_some(), "bias-only U");
            }
            (role, vec![ci])
        } else if roles.contains(&e.role.as_str()) {
            assert!(e.multiplicity > 0);
            (e.role.as_str(), (0..e.multiplicity).collect())
        } else {
            continue;
        };
        for position in positions {
            let key = (role.to_string(), e.micro, e.layer, position);
            assert!(
                expected
                    .insert(key.clone(), (e.metadata.level(), e.metadata.scale_bits()))
                    .is_none(),
                "duplicate planned observation {key:?}"
            );
        }
    }
    let mut actual = BTreeMap::new();
    for v in &observed.values {
        let key = (v.role.to_string(), v.micro, v.layer, v.position);
        assert!(
            actual
                .insert(key.clone(), (v.level, v.scale_bits))
                .is_none(),
            "duplicate actual observation {key:?}"
        );
    }
    assert_eq!(
        actual, expected,
        "complete observation key/position/cardinality/metadata equality"
    );

    let mut expected_coefficients = BTreeMap::new();
    for layer in 0..plan.state.w.len() {
        for (role, bits) in [
            ("current", Some(plan.coefficients.current_bits)),
            ("bias", Some(plan.coefficients.bias_bits)),
            ("reference", plan.coefficients.reference_bits),
        ] {
            if let Some(bits) = bits {
                expected_coefficients.insert((role, Some(layer), 0), bits);
            }
        }
    }
    if let (Some(mu), Some(kappa)) = (
        plan.coefficients.velocity_mu_bits,
        plan.coefficients.velocity_kappa_bits,
    ) {
        if mode == DiagonalMode::Interactive {
            for layer in 0..plan.state.w.len() {
                expected_coefficients.insert(("mu", Some(layer), 0), mu);
                expected_coefficients.insert(("kappa", Some(layer), 0), kappa);
            }
        } else if !plan.first {
            for (ci, spec) in compact.as_ref().unwrap().containers.iter().enumerate() {
                if spec.layer.is_some() && plan.state.u[ci].is_some() {
                    expected_coefficients.insert(("mu", spec.layer, ci), mu);
                    expected_coefficients.insert(("kappa", spec.layer, ci), kappa);
                }
            }
        }
    }
    let mut actual_coefficients = BTreeMap::new();
    for c in &observed.coefficients {
        let key = (c.role, c.layer, c.position);
        assert!(
            actual_coefficients.insert(key, c.bits).is_none(),
            "duplicate coefficient {key:?}"
        );
    }
    assert_eq!(
        actual_coefficients, expected_coefficients,
        "complete branch-specific coefficients"
    );

    let retry_role = if mode == DiagonalMode::Interactive {
        "bias_gradient"
    } else {
        "bias_extracted"
    };
    let expected_attempts = plan
        .attempts
        .iter()
        .filter(|a| a.operation == retry_role)
        .map(|a| {
            (
                a.operation.as_str(),
                a.micro,
                a.layer,
                a.source.level(),
                a.source.scale_bits(),
                a.target_bits,
                a.error.clone(),
            )
        })
        .collect::<Vec<_>>();
    let actual_attempts = observed
        .attempts
        .iter()
        .map(|a| {
            (
                a.role,
                a.micro,
                a.layer,
                a.source_level,
                a.source_scale_bits,
                a.target_bits,
                a.error.clone(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_attempts, expected_attempts,
        "complete ordered observed retry roles"
    );
}

// This fixture deliberately lists the frozen independent matrix axes.
#[allow(clippy::too_many_arguments)]
fn run_case(
    width: usize,
    poly: bool,
    mode: DiagonalMode,
    cfg: EvalConfig,
    reuse: bool,
    frozen: bool,
    rates: &[f64],
    parts: &[Vec<usize>],
    nag: bool,
) {
    assert!(rates.len() <= 4 && rates.len() == parts.len());
    eprintln!(
        "P2_CASE test={} mode={mode:?} width={width} poly={poly} cfg={cfg:?} reuse={reuse} frozen={frozen} nag={nag} updates={} parts={parts:?}",
        std::thread::current().name().unwrap_or("unnamed"),
        rates.len()
    );
    let ni = matches!(mode, DiagonalMode::NonInteractive(_));
    let ctx = context(ni);
    let ev = ctx.evaluator(cfg);
    let dims = [2, 4, width];
    let trans = if reuse {
        TransposeKeys::Reuse(crate::trainer::mlp_key_pool(&dims, &ctx.layout).unwrap())
    } else {
        TransposeKeys::Dedicated
    };
    let g = plan_stage_geometry(
        &dims,
        &ctx.layout,
        &[
            if poly {
                DiagonalActivation::PolyRelu2
            } else {
                DiagonalActivation::Square
            },
            if poly {
                DiagonalActivation::Identity
            } else {
                DiagonalActivation::Square
            },
        ],
        &trans,
        mode,
        cfg,
    )
    .unwrap();
    let optim = if nag {
        DiagonalOptimizer::Nag {
            lr: rates[0],
            reference: 0.08,
            momentum: 0.9,
        }
    } else {
        DiagonalOptimizer::Sgd { lr: rates[0] }
    };
    let policy = if frozen {
        BiasPolicy::Frozen
    } else {
        BiasPolicy::Active
    };
    let mut state = initial_stage_metadata(&g, &ctx.p, ctx.bm.as_ref(), optim, policy).unwrap();
    let recorder = Rc::new(RefCell::new(observation::Recorder::default()));
    let setup = layers(&dims, frozen);
    let guard = ctx.bs.as_ref().map(|bs| Guard {
        bs,
        used: Cell::new(0),
        limit: rates.len() * if nag { 4 } else { 2 }
            + if mode == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) {
                parts.iter().map(Vec::len).sum()
            } else {
                0
            },
    });
    let (mut mlp, compact) = if ni {
        let (m, s) = NonInteractiveTrainer::encrypt_network(
            &ev,
            guard.as_ref().unwrap(),
            &setup,
            acts(poly),
            &ctx.layout,
            match mode {
                DiagonalMode::NonInteractive(p) => p,
                _ => unreachable!(),
            },
            &ctx.encrypt,
            &ctx.enc,
            &mut rng(52),
        )
        .unwrap();
        (m, Some(s))
    } else {
        (
            Mlp::encrypt(
                &ev,
                &setup,
                acts(poly),
                &ctx.layout,
                &ctx.encrypt,
                &ctx.enc,
                &mut rng(52),
            )
            .unwrap(),
            None,
        )
    };
    for (i, (d, _)) in mlp.layers.iter().enumerate() {
        for w in d.w.diagonals().entries() {
            ct_eq(state.w[i], w)
        }
        ct_eq(state.b[i], &d.b)
    }
    let mut it = if !ni {
        let opts = (0..2)
            .map(|i| {
                if nag {
                    let mut n = Nag::new(rates[0], 0.9);
                    n.set_lr_reference(0.08).unwrap();
                    n.observer = Some(recorder.clone());
                    n.observer_layer = Some(i);
                    Box::new(n) as Box<dyn LayerOptimizer>
                } else {
                    Box::new(Sgd { lr: rates[0] }) as Box<dyn LayerOptimizer>
                }
            })
            .collect();
        let mut t = Trainer::new(&ev, Box::new(ctx.client()), opts, ctx.layout)
            .unwrap()
            .with_transpose_keys(trans.clone());
        if frozen {
            t = t.with_frozen_biases()
        }
        t.observer = Some(recorder.clone());
        Some(t)
    } else {
        None
    };
    let mut nt = if ni {
        let opt = if nag {
            NiOptimizer::Nag {
                lr: rates[0],
                momentum: 0.9,
            }
        } else {
            NiOptimizer::Sgd { lr: rates[0] }
        };
        let mut t = NonInteractiveTrainer::new(
            &ev,
            guard.as_ref().unwrap(),
            opt,
            ctx.layout,
            match mode {
                DiagonalMode::NonInteractive(p) => p,
                _ => unreachable!(),
            },
            &mlp,
            compact.unwrap(),
        )
        .unwrap()
        .with_transpose_keys(trans.clone());
        if nag {
            t.set_lr_reference(0.08).unwrap()
        }
        if frozen {
            t = t.with_frozen_biases()
        }
        t.observer = Some(recorder.clone());
        Some(t)
    } else {
        None
    };
    for (update, &rate) in rates.iter().enumerate() {
        recorder.borrow_mut().values.clear();
        recorder.borrow_mut().attempts.clear();
        recorder.borrow_mut().coefficients.clear();
        let before_probe = ev.counter();
        let runtime_batch = if let Some(t) = &nt {
            t.batch_spec(&mlp).unwrap()
        } else {
            mlp.batch_spec(&ev, &ctx.enc).unwrap()
        };
        assert_eq!(runtime_batch.x_level, state.batch.x.level());
        assert_eq!(runtime_batch.x_scale.to_bits(), state.batch.x.scale_bits());
        assert_eq!(runtime_batch.y_level, state.batch.y.level());
        assert_eq!(runtime_batch.y_scale.to_bits(), state.batch.y.scale_bits());
        assert_eq!(before_probe, ev.counter());
        let next_full = parts.get(update + 1).map(|p| p.iter().all(|&n| n == 8));
        let spec = UpdateSpec::new(
            parts[update]
                .iter()
                .map(|&n| MicroMetadata {
                    n_valid: n,
                    x: state.batch.x,
                    y: state.batch.y,
                })
                .collect(),
            parts[update].iter().sum(),
            rate,
            next_full,
        );
        let plan = plan_stage_update(&g, &ctx.p, ctx.bm.as_ref(), &state, &spec).unwrap();
        let batches = parts[update]
            .iter()
            .map(|&n| {
                let (x, y) = input(
                    ctx,
                    state.batch,
                    n,
                    width,
                    if poly {
                        match width {
                            4 => 3,
                            16 => 10,
                            _ => 2,
                        }
                    } else {
                        width
                    },
                );
                (x, y, n)
            })
            .collect::<Vec<_>>();
        let refs = batches
            .iter()
            .map(|(x, y, n)| (x, y, *n))
            .collect::<Vec<_>>();
        let before = ev.counter();
        reserve_stage_calls(1);
        eprintln!(
            "P2_STAGE_PRE mode={mode:?} width={width} cfg={cfg:?} update={update} rate_bits={:#018x} parts={:?} next_full={next_full:?} reserved_stage_calls={} guard_used={}",
            rate.to_bits(),
            parts[update],
            TOTAL_STAGE_CALLS.load(Ordering::SeqCst),
            guard.as_ref().map_or(0, |g| g.used.get())
        );
        if let Some(t) = &mut it {
            t.set_lr(rate).unwrap();
            let refresh_origin = t.refreshes();
            let result = t.train_step_valid_with_hidden(&mut mlp, &refs);
            eprintln!(
                "P2_STAGE_RESULT mode={mode:?} width={width} cfg={cfg:?} update={update} parts={:?} ok={} error={:?} counter_delta={:?} guard_used={} client_refresh_delta={}",
                parts[update],
                result.is_ok(),
                result.as_ref().err(),
                &ev.counter() - &before,
                guard.as_ref().map_or(0, |g| g.used.get()),
                t.refreshes() - refresh_origin
            );
            let hidden = result.unwrap();
            assert_eq!(t.refreshes() - refresh_origin, plan.refreshes);
            assert_eq!(hidden.len(), plan.hidden.len());
            for (k, h) in hidden.iter().enumerate() {
                assert_eq!(h.len(), plan.hidden[k].len());
                for (j, a) in h.iter().enumerate() {
                    ct_eq(plan.hidden[k][j], a)
                }
            }
        } else {
            let t = nt.as_mut().unwrap();
            t.set_lr(rate).unwrap();
            t.set_next_batch_full(next_full);
            let result = t.train_step_valid_with_hidden(&mut mlp, &refs);
            eprintln!(
                "P2_STAGE_RESULT mode={mode:?} width={width} cfg={cfg:?} update={update} parts={:?} ok={} error={:?} counter_delta={:?} guard_used={} weight_boots={} activation_boots={}",
                parts[update],
                result.is_ok(),
                result.as_ref().err(),
                &ev.counter() - &before,
                guard.as_ref().map_or(0, |g| g.used.get()),
                t.weight_bootstraps(),
                t.activation_bootstraps()
            );
            let hidden = result.unwrap();
            assert_eq!(hidden.len(), plan.hidden.len());
            for (k, h) in hidden.iter().enumerate() {
                assert_eq!(h.len(), plan.hidden[k].len());
                for (j, a) in h.iter().enumerate() {
                    ct_eq(plan.hidden[k][j], a)
                }
            }
            assert_eq!(t.last_update_booted(), plan.booted);
            assert_eq!(t.last_update_rescaled(), plan.reference);
            assert_eq!(
                t.state().weights.iter().map(label).collect::<Vec<_>>(),
                plan.state
                    .c
                    .iter()
                    .copied()
                    .map(meta_label)
                    .collect::<Vec<_>>(),
                "complete compact W"
            );
            assert_eq!(
                t.state()
                    .velocity
                    .iter()
                    .map(|v| v.as_ref().map(label))
                    .collect::<Vec<_>>(),
                plan.state
                    .u
                    .iter()
                    .map(|v| v.map(meta_label))
                    .collect::<Vec<_>>(),
                "complete compact U/absence"
            );
            assert_eq!(t.last_activation_bootstraps().len(), plan.canonical.len());
            for (k, hh) in t.last_activation_bootstraps().iter().enumerate() {
                assert_eq!(hh.len(), plan.canonical[k].len());
                for (j, a) in hh.iter().enumerate() {
                    ct_eq(plan.canonical[k][j], a)
                }
            }
        }
        assert_eq!(
            &ev.counter() - &before,
            plan.including_boot,
            "full seven counters {mode:?} width{width} update{update}"
        );
        compare_observed(&plan, &recorder.borrow());
        compare_expanded(&plan, &mlp);
        let after = ev.counter();
        let probe = ev.fork();
        let _ = mlp.batch_spec(&probe, &ctx.enc).unwrap();
        assert_eq!(after, ev.counter());
        state = plan.state;
    }
}
#[test]
fn p2_runtime_i_class_metadata_counts() {
    for (i, (w, poly)) in [
        (3, false),
        (2, false),
        (10, false),
        (4, true),
        (2, true),
        (16, true),
    ]
    .into_iter()
    .enumerate()
    {
        run_case(
            w,
            poly,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
            false,
            poly,
            &[0.08],
            &[match i % 3 {
                0 => vec![8],
                1 => vec![5],
                _ => vec![8, 4],
            }],
            false,
        )
    }
}
#[test]
fn p2_runtime_i_nag_decrease_restart() {
    for (w, poly) in [(3, false), (4, true)] {
        run_case(
            w,
            poly,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
            poly,
            poly,
            &[0.04, 0.08, 0.02, 0.08],
            &[vec![8, 4], vec![8], vec![5], vec![8, 4]],
            true,
        )
    }
}
#[test]
fn p2_runtime_i_nag_first_reference() {
    for (w, poly) in [(3, false), (4, true)] {
        run_case(
            w,
            poly,
            DiagonalMode::Interactive,
            EvalConfig::OPTIMIZED,
            false,
            false,
            &[0.08, 0.08],
            &[vec![8], vec![8]],
            true,
        )
    }
}
#[test]
fn p2_runtime_ni_class_placements() {
    for p in [
        BootPlacement::WhenNeeded,
        BootPlacement::AfterUpdate,
        BootPlacement::AfterActivation,
    ] {
        for (i, (w, poly)) in [
            (3, false),
            (2, false),
            (10, false),
            (4, true),
            (2, true),
            (16, true),
        ]
        .into_iter()
        .enumerate()
        {
            run_case(
                w,
                poly,
                DiagonalMode::NonInteractive(p),
                EvalConfig::OPTIMIZED,
                false,
                poly,
                &[0.08],
                &[match i % 3 {
                    0 => vec![8],
                    1 => vec![5],
                    _ => vec![8, 4],
                }],
                false,
            )
        }
    }
}
#[test]
fn p2_runtime_ni_nag_decrease_restart() {
    for p in [
        BootPlacement::WhenNeeded,
        BootPlacement::AfterUpdate,
        BootPlacement::AfterActivation,
    ] {
        for (w, poly) in [(3, false), (4, true)] {
            run_case(
                w,
                poly,
                DiagonalMode::NonInteractive(p),
                EvalConfig::OPTIMIZED,
                poly,
                poly,
                &[0.04, 0.08, 0.02, 0.08],
                &[vec![8, 4], vec![8], vec![5], vec![8, 4]],
                true,
            )
        }
    }
}
#[test]
fn p2_runtime_ni_nag_first_reference() {
    for p in [
        BootPlacement::WhenNeeded,
        BootPlacement::AfterUpdate,
        BootPlacement::AfterActivation,
    ] {
        for (w, poly) in [(3, false), (4, true)] {
            run_case(
                w,
                poly,
                DiagonalMode::NonInteractive(p),
                EvalConfig::OPTIMIZED,
                false,
                false,
                &[0.08, 0.08],
                &[vec![8], vec![8]],
                true,
            )
        }
    }
}
#[test]
fn p2_runtime_eval_config_grid() {
    for cfg in EvalConfig::all() {
        for mode in [
            DiagonalMode::Interactive,
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
        ] {
            run_case(
                4,
                true,
                mode,
                cfg,
                false,
                true,
                &[0.04],
                &[vec![8, 4]],
                true,
            )
        }
    }
}

#[test]
fn p2_runtime_old_hidden_boundary_counts() {
    use crate::block::{BlockMlp, BlockNiTrainer, BlockTrainer};
    let cases = vec![
        vec![DiagonalMode::Interactive; 2],
        vec![DiagonalMode::NonInteractive(BootPlacement::WhenNeeded); 2],
        vec![DiagonalMode::NonInteractive(BootPlacement::AfterUpdate); 2],
        vec![DiagonalMode::NonInteractive(BootPlacement::AfterActivation); 2],
        vec![
            DiagonalMode::NonInteractive(BootPlacement::AfterActivation),
            DiagonalMode::NonInteractive(BootPlacement::AfterUpdate),
            DiagonalMode::NonInteractive(BootPlacement::WhenNeeded),
        ],
    ];
    for modes in cases {
        eprintln!(
            "P2_BLOCK_CASE test={} modes={modes:?} width=4 cfg={:?} update=0 parts=[8,4] stages={}",
            std::thread::current().name().unwrap_or("unnamed"),
            EvalConfig::OPTIMIZED,
            modes.len()
        );
        let ni = modes[0] != DiagonalMode::Interactive;
        let c = context(ni);
        let ev = c.evaluator(EvalConfig::OPTIMIZED);
        let guards = c.bs.as_ref().map(|bs| Guard {
            bs,
            used: Cell::new(0),
            limit: match modes.len() {
                3 => 10,
                _ if modes[0] == DiagonalMode::NonInteractive(BootPlacement::AfterActivation) => 8,
                _ => 6,
            },
        });
        let geometries = modes
            .iter()
            .enumerate()
            .map(|(i, &mode)| {
                plan_stage_geometry(
                    &[if i == 0 { 2 } else { 4 }, 4, 4],
                    &c.layout,
                    &[DiagonalActivation::PolyRelu2, DiagonalActivation::Identity],
                    &TransposeKeys::Dedicated,
                    mode,
                    EvalConfig::OPTIMIZED,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let states = geometries
            .iter()
            .map(|g| {
                initial_stage_metadata(
                    g,
                    &c.p,
                    c.bm.as_ref(),
                    DiagonalOptimizer::Sgd { lr: 0.08 },
                    BiasPolicy::Frozen,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut models = vec![];
        let mut compacts = vec![];
        for (i, g) in geometries.iter().enumerate() {
            let init = layers(g.dims(), true);
            if ni {
                let (m, s) = NonInteractiveTrainer::encrypt_network(
                    &ev,
                    guards.as_ref().unwrap(),
                    &init,
                    acts(true),
                    &c.layout,
                    match modes[i] {
                        DiagonalMode::NonInteractive(p) => p,
                        _ => unreachable!(),
                    },
                    &c.encrypt,
                    &c.enc,
                    &mut rng(52),
                )
                .unwrap();
                models.push(m);
                compacts.push(s);
            } else {
                models.push(
                    Mlp::encrypt(
                        &ev,
                        &init,
                        acts(true),
                        &c.layout,
                        &c.encrypt,
                        &c.enc,
                        &mut rng(52),
                    )
                    .unwrap(),
                )
            }
        }
        let output = models.pop().unwrap();
        let mut net = BlockMlp::new(models, output).unwrap();
        let recorders = (0..modes.len())
            .map(|_| Rc::new(RefCell::new(observation::Recorder::default())))
            .collect::<Vec<_>>();
        let mut it = if !ni {
            let trainers = (0..modes.len())
                .map(|i| {
                    let mut t = Trainer::new(
                        &ev,
                        Box::new(c.client()),
                        vec![Box::new(Sgd { lr: 0.08 }), Box::new(Sgd { lr: 0.08 })],
                        c.layout,
                    )
                    .unwrap()
                    .with_frozen_biases();
                    t.observer = Some(recorders[i].clone());
                    t
                })
                .collect();
            Some(BlockTrainer::new(&ev, trainers, Box::new(c.client())).unwrap())
        } else {
            None
        };
        let mut nt = if ni {
            let trainers = compacts
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut t = NonInteractiveTrainer::new(
                        &ev,
                        guards.as_ref().unwrap(),
                        NiOptimizer::Sgd { lr: 0.08 },
                        c.layout,
                        match modes[i] {
                            DiagonalMode::NonInteractive(p) => p,
                            _ => unreachable!(),
                        },
                        net.stage(i),
                        s,
                    )
                    .unwrap()
                    .with_frozen_biases();
                    t.observer = Some(recorders[i].clone());
                    t
                })
                .collect();
            Some(BlockNiTrainer::new(&ev, guards.as_ref().unwrap(), trainers).unwrap())
        } else {
            None
        };
        let mut plans = vec![];
        let mut handoffs = vec![];
        let mut xs = vec![states[0].batch.x; 2];
        let mut expected = OpCounter::default();
        for i in 0..modes.len() {
            let u = UpdateSpec::new(
                [8, 4]
                    .iter()
                    .enumerate()
                    .map(|(k, &n)| MicroMetadata {
                        n_valid: n,
                        x: xs[k],
                        y: states[i].batch.y,
                    })
                    .collect(),
                12,
                0.08,
                None,
            );
            let plan =
                plan_stage_update(&geometries[i], &c.p, c.bm.as_ref(), &states[i], &u).unwrap();
            merge(&mut expected, &plan.including_boot).unwrap();
            if i + 1 < modes.len() {
                let h = plan_stage_handoff(
                    &geometries[i],
                    &c.p,
                    c.bm.as_ref(),
                    &plan,
                    &geometries[i + 1],
                )
                .unwrap();
                merge(&mut expected, &h.counts_including_bootstrap).unwrap();
                xs = h.outputs.clone();
                handoffs.push(h);
            }
            plans.push(plan);
        }
        let batches = [8, 4]
            .iter()
            .map(|&n| {
                let (x, _) = input(c, states[0].batch, n, 4, 3);
                let ys = states
                    .iter()
                    .map(|s| input(c, s.batch, n, 4, 3).1)
                    .collect::<Vec<_>>();
                (x, ys, n)
            })
            .collect::<Vec<_>>();
        let targets = batches
            .iter()
            .map(|(_, ys, _)| ys.iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let refs = batches
            .iter()
            .zip(&targets)
            .map(|((x, _, n), ys)| (x, ys.as_slice(), *n))
            .collect::<Vec<_>>();
        let before = ev.counter();
        reserve_stage_calls(modes.len());
        eprintln!(
            "P2_BLOCK_PRE modes={modes:?} update=0 parts=[8,4] reserved_stage_calls={} guard_used={}",
            TOTAL_STAGE_CALLS.load(Ordering::SeqCst),
            guards.as_ref().map_or(0, |g| g.used.get())
        );
        if let Some(t) = &mut it {
            let result = t.train_step_valid(&mut net, &refs);
            eprintln!(
                "P2_BLOCK_RESULT modes={modes:?} update=0 parts=[8,4] ok={} error={:?} counter_delta={:?} guard_used={}",
                result.is_ok(),
                result.as_ref().err(),
                &ev.counter() - &before,
                guards.as_ref().map_or(0, |g| g.used.get())
            );
            result.unwrap();
            assert_eq!(t.handoffs(), 2 * (modes.len() - 1));
        } else {
            let t = nt.as_mut().unwrap();
            let result = t.train_step_valid(&mut net, &refs);
            eprintln!(
                "P2_BLOCK_RESULT modes={modes:?} update=0 parts=[8,4] ok={} error={:?} counter_delta={:?} guard_used={}",
                result.is_ok(),
                result.as_ref().err(),
                &ev.counter() - &before,
                guards.as_ref().map_or(0, |g| g.used.get())
            );
            result.unwrap();
            compare_nested_labels(
                &handoffs
                    .iter()
                    .map(|h| h.outputs.iter().copied().map(meta_label).collect())
                    .collect::<Vec<Vec<_>>>(),
                &t.last_handoff_bootstraps()
                    .iter()
                    .map(|hs| hs.iter().map(label).collect())
                    .collect::<Vec<Vec<_>>>(),
            );
            assert_eq!(t.stages().len(), plans.len());
            for (stage, plan) in t.stages().iter().zip(&plans) {
                assert_eq!(
                    stage.state().weights.iter().map(label).collect::<Vec<_>>(),
                    plan.state
                        .c
                        .iter()
                        .copied()
                        .map(meta_label)
                        .collect::<Vec<_>>(),
                    "block compact W"
                );
                assert_eq!(
                    stage
                        .state()
                        .velocity
                        .iter()
                        .map(|v| v.as_ref().map(label))
                        .collect::<Vec<_>>(),
                    plan.state
                        .u
                        .iter()
                        .map(|v| v.map(meta_label))
                        .collect::<Vec<_>>(),
                    "block compact U"
                );
            }
        }
        assert_eq!(&ev.counter() - &before, expected, "boundary modes{modes:?}");
        assert_eq!(net.num_stages(), plans.len());
        for i in 0..modes.len() {
            compare_observed(&plans[i], &recorders[i].borrow());
            compare_expanded(&plans[i], net.stage(i));
        }
    }
}
