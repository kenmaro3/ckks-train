//! NONSCIENTIFIC_TEST_EVIDENCE. Compile-visible complete fixed matrix, no automatic release.
use super::*;
use crate::{
    study_admission::{
        PlanningBudget, PlanningParameters, PlanningUsage, p5_fixtures::SyntheticBlockPlan,
    },
    study_client::{
        CpuBootstrapDescription, StudyClient,
        blocks::{BlockPreparationInput, make_blocks_owned, prepare_blocks_then_make},
        context_parameters,
    },
    study_plain::p5_fixtures::{BlockFamily, BlockOracle, BlockRecipe, PairedOraclePhase},
};
use ckks_train_ckks::{
    CkksParams,
    bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, DftSupportProof},
    },
};
use ckks_train_core::sampler::Rng;
use ckks_train_nn::optim::Nag;
use ndarray::Array2;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
#[repr(usize)]
enum Work {
    Params,
    Keys,
    Proof,
    Descriptor,
    Bootstrapper,
    Encoder,
    Attempt,
    Boot,
    Refresh,
    Packet,
    X,
    Y,
    Endpoint,
    Inspection,
    HandoffInspection,
    Session,
    Snapshot,
    Unchanged,
    ConfigSnapshot,
    ConfigComparison,
    PrefixCapture,
    Spy,
}
const CAPS: [usize; 22] = [
    2, 2, 1, 1, 1, 40, 42, 361, 357, 42, 64, 152, 40, 96, 84, 16, 46, 68, 3, 4, 2, 1,
];
#[derive(Default)]
struct Budget {
    counts: [Cell<usize>; 22],
}
thread_local! {static ACTIVE:RefCell<Option<Rc<Budget>>>=const {RefCell::new(None)};}
fn reserve(work: Work) {
    ACTIVE.with(|active| {
        if let Some(b) = active.borrow().as_ref() {
            let cell = &b.counts[work as usize];
            let next = cell.get().checked_add(1).unwrap();
            eprintln!(
                "P5_BEFORE_CALL kind={work:?} ordinal={next} cap={}",
                CAPS[work as usize]
            );
            assert!(next <= CAPS[work as usize], "P5 {work:?} cap");
            cell.set(next);
        }
    });
}
macro_rules! guards {($($name:ident:$kind:ident),*$(,)?)=>{$(pub(crate) fn $name(){reserve(Work::$kind);})*};}
guards! {reserve_key:Keys,reserve_bootstrapper:Bootstrapper,reserve_encoder:Encoder,reserve_boot:Boot,reserve_refresh:Refresh,reserve_attempt:Attempt,reserve_packet:Packet,reserve_x:X,reserve_y:Y,reserve_endpoint:Endpoint,reserve_inspection:Inspection,reserve_handoff_inspection:HandoffInspection}
struct Active;
impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|active| *active.borrow_mut() = None);
    }
}
fn rng(seed: u8) -> Rng {
    Rng::from_seed([seed; 32])
}
struct Case {
    family: BlockFamily,
    mode: DiagonalMode,
    plan: Arc<SyntheticBlockPlan>,
    poison: bool,
}
fn arrays(recipe: &BlockRecipe) -> Vec<Vec<(Array2<f64>, Vec<f64>)>> {
    recipe
        .initial()
        .chunks_exact(2)
        .map(|stage| {
            stage
                .iter()
                .map(|l| (l.weights().clone(), l.biases().to_vec()))
                .collect()
        })
        .collect()
}
fn setup(
    case: &Case,
    parameters: Arc<CkksParams>,
    boot: Option<CpuBootstrapDescription>,
    steps: &[i32],
) -> (StudyClient, CpuEvaluationContext) {
    let bound = boot.clone();
    let actual = PlanningParameters {
        params: &parameters,
        bootstrap: bound.as_ref().map(|b| b.metadata.as_ref()),
        boot_params: bound.as_ref().map(|b| &b.parameters),
    };
    let claim = case.plan.claim();
    prepare_blocks_then_make(
        BlockPreparationInput::Synthetic {
            plan: case.plan.clone(),
            claim: &claim,
        },
        actual,
        &case.plan.parts,
        |checked| make_blocks_owned(&checked, parameters.clone(), steps, boot, rng(61), rng(62)),
    )
    .unwrap()
}
fn spy_contract(
    plan: &Arc<SyntheticBlockPlan>,
    actual: PlanningParameters<'_>,
    other: PlanningParameters<'_>,
) {
    let claim = plan.claim();
    let marker = prepare_blocks_then_make(
        BlockPreparationInput::Synthetic {
            plan: plan.clone(),
            claim: &claim,
        },
        actual,
        &plan.parts,
        |checked| {
            reserve(Work::Spy);
            assert!(checked.plan.same(&BlockPlanOwner::Synthetic(plan.clone())));
            Ok("checked-private-block-factory")
        },
    )
    .unwrap();
    assert_eq!(marker, "checked-private-block-factory");
    let calls = Cell::new(0usize);
    let reject = |claim: &crate::study_admission::p5_fixtures::SyntheticBlockClaim,
                  p: PlanningParameters<'_>,
                  parts: &[Vec<MicroPart>],
                  field: &str| {
        let result = prepare_blocks_then_make(
            BlockPreparationInput::Synthetic {
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
    };
    let mut wrong = plan.claim();
    wrong.identity.push('x');
    reject(&wrong, actual, &plan.parts, "identity");
    reject(
        &plan.claim_for_actual_parameters(other),
        other,
        &plan.parts,
        "parameters",
    );
    for field in 0..7 {
        reject(
            &plan.bad_parameter_claim(actual, field),
            actual,
            &plan.parts,
            "parameters",
        );
    }
    for parts in [
        vec![],
        vec![vec![MicroPart { start: 0, valid: 0 }]],
        vec![vec![MicroPart { start: 1, valid: 8 }]],
        vec![vec![MicroPart { start: 0, valid: 9 }]],
        vec![vec![MicroPart { start: 0, valid: 7 }]],
        vec![vec![
            MicroPart { start: 4, valid: 4 },
            MicroPart { start: 0, valid: 4 },
        ]],
        vec![vec![
            MicroPart { start: 0, valid: 4 },
            MicroPart { start: 0, valid: 4 },
        ]],
    ] {
        reject(&claim, actual, &parts, "partition");
    }
    eprintln!(
        "P5_FACTORY_SPY positive=1 negative=16 key_factory_reached=0 canonical_owned_start=false"
    );
}
#[derive(Debug, PartialEq)]
struct Config {
    lr: u64,
    reference: Option<u64>,
    momentum: Option<u64>,
    weight: u64,
    bias: u64,
    reference_factor: Option<u64>,
}
fn i_config(t: &Trainer<'_>, layers: usize) -> Vec<Config> {
    (0..layers)
        .map(|l| {
            let v = t.optimizer_view(l).unwrap();
            Config {
                lr: v.lr().to_bits(),
                reference: v.reference().map(f64::to_bits),
                momentum: v.momentum().map(f64::to_bits),
                weight: v.weight_factor().to_bits(),
                bias: v.bias_factor().to_bits(),
                reference_factor: v.reference_factor().map(f64::to_bits),
            }
        })
        .collect()
}
fn configs(s: &CpuBlockSession<'_>) -> Vec<Vec<Config>> {
    match &s.trainer {
        BlockTraining::Interactive(t) => t
            .stages()
            .iter()
            .enumerate()
            .map(|(stage, t)| i_config(t, s.model.stage(stage).layers.len()))
            .collect(),
        BlockTraining::NonInteractive(t) => t
            .stages()
            .iter()
            .map(|t| {
                let o = t.optimizer();
                let momentum = match o {
                    NiOptimizer::Nag { momentum, .. } => Some(momentum),
                    _ => None,
                };
                let reference = (o.has_velocity() && o.lr() / t.lr_reference() != 1.)
                    .then(|| o.with_lr(t.lr_reference()).grad_scale());
                vec![Config {
                    lr: o.lr().to_bits(),
                    reference: Some(t.lr_reference().to_bits()),
                    momentum: momentum.map(f64::to_bits),
                    weight: o.grad_scale().to_bits(),
                    bias: o.bias_grad_scale().to_bits(),
                    reference_factor: reference.map(f64::to_bits),
                }]
            })
            .collect(),
    }
}
fn visit_ciphertexts(s: &CpuBlockSession<'_>, mut visit: impl FnMut(&Ciphertext)) {
    for stage in 0..s.model.num_stages() {
        for (d, _) in &s.model.stage(stage).layers {
            for c in d.w.diagonals().entries() {
                visit(c);
            }
            visit(&d.b);
        }
    }
    match &s.trainer {
        BlockTraining::Interactive(t) => {
            for (stage, t) in t.stages().iter().enumerate() {
                for layer in 0..s.model.stage(stage).layers.len() {
                    let view = t.optimizer_view(layer).unwrap();
                    if let Some(v) = view.velocity() {
                        for c in v.entries() {
                            visit(c);
                        }
                    }
                }
            }
        }
        BlockTraining::NonInteractive(t) => {
            for t in t.stages() {
                for c in &t.state().weights {
                    visit(c);
                }
                for c in t.state().velocity.iter().flatten() {
                    visit(c);
                }
                for c in t.last_activation_bootstraps().iter().flatten() {
                    visit(c);
                }
            }
            for c in t.last_handoff_bootstraps().iter().flatten() {
                visit(c);
            }
        }
    }
}
#[derive(Debug, PartialEq)]
struct RawMetadata {
    dims: Vec<Vec<usize>>,
    batches: Vec<BatchSpec>,
    velocity: Vec<Vec<bool>>,
    boot_refresh: Vec<(u64, u64)>,
    handoffs: u64,
    reference_flags: Vec<(bool, bool)>,
    compact_layouts: Vec<ckks_train_nn::trainer_ni::CompactLayout>,
}
fn raw_metadata(s: &CpuBlockSession<'_>) -> RawMetadata {
    let (velocity, boot_refresh, handoffs, reference_flags) = match &s.trainer {
        BlockTraining::Interactive(t) => (
            t.stages()
                .iter()
                .enumerate()
                .map(|(stage, t)| {
                    (0..s.model.stage(stage).layers.len())
                        .map(|layer| t.optimizer_view(layer).unwrap().velocity().is_some())
                        .collect()
                })
                .collect(),
            t.stages()
                .iter()
                .map(|t| (t.refreshes() as u64, 0))
                .collect(),
            t.handoffs() as u64,
            vec![],
        ),
        BlockTraining::NonInteractive(t) => (
            t.stages()
                .iter()
                .map(|t| t.state().velocity.iter().map(Option::is_some).collect())
                .collect(),
            t.stages()
                .iter()
                .map(|t| (t.weight_bootstraps(), t.activation_bootstraps()))
                .collect(),
            t.handoff_bootstraps(),
            t.stages()
                .iter()
                .map(|t| (t.last_update_rescaled(), t.last_update_booted()))
                .collect(),
        ),
    };
    let compact_layouts = match &s.trainer {
        BlockTraining::Interactive(_) => vec![],
        BlockTraining::NonInteractive(t) => t
            .stages()
            .iter()
            .map(|t| t.compact_layout().clone())
            .collect(),
    };
    RawMetadata {
        compact_layouts,
        dims: s.model.stage_dims(),
        batches: s.batches.clone(),
        velocity,
        boot_refresh,
        handoffs,
        reference_flags,
    }
}
struct RawSnapshot {
    cts: Vec<Ciphertext>,
    i_boundary: Vec<Ciphertext>,
    counter: OpCounter,
    progress: (usize, usize, bool),
    failure: Option<BlockFailure>,
    traces: Vec<Vec<(String, usize)>>,
    config: Vec<Vec<Config>>,
    metadata: RawMetadata,
}
impl RawSnapshot {
    fn capture(s: &CpuBlockSession<'_>) -> Self {
        reserve(Work::Snapshot);
        let mut cts = Vec::new();
        visit_ciphertexts(s, |c| cts.push(c.clone()));
        let i_boundary = s.facade.boundary.borrow().clone();
        let mut receipt = s.retention_receipt(None, None).unwrap();
        receipt
            .include_diagnostic_copy(
                "full-session encrypted snapshot; through rejection comparisons only",
                cts.iter().chain(&i_boundary),
            )
            .unwrap();
        eprintln!("P5_RETENTION full_snapshot_live {receipt:?}");
        Self {
            cts,
            i_boundary,
            counter: s.facade.base.counter(),
            progress: (s.completed, s.samples, s.poisoned),
            failure: s.failed.clone(),
            traces: s.traces(),
            config: configs(s),
            metadata: raw_metadata(s),
        }
    }
    fn assert_unchanged(&self, s: &CpuBlockSession<'_>) {
        reserve(Work::Unchanged);
        assert_eq!(self.counter, s.facade.base.counter());
        assert_eq!(self.progress, (s.completed, s.samples, s.poisoned));
        assert_eq!(self.failure, s.failed);
        assert_eq!(self.traces, s.traces());
        assert_eq!(self.config, configs(s));
        assert_eq!(self.metadata, raw_metadata(s));
        let mut ordinal = 0;
        visit_ciphertexts(s, |b| {
            assert!(ordinal < self.cts.len());
            assert!(ciphertext_exact(&self.cts[ordinal], b));
            ordinal += 1;
        });
        assert_eq!(ordinal, self.cts.len());
        let boundary = s.facade.boundary.borrow();
        assert_eq!(boundary.len(), self.i_boundary.len());
        for (a, b) in self.i_boundary.iter().zip(boundary.iter()) {
            assert!(ciphertext_exact(a, b));
        }
    }
}
fn prework_negatives(
    s: &mut CpuBlockSession<'_>,
    packet: &mut EncryptedBlockUpdate,
    other_plan: &Arc<SyntheticBlockPlan>,
) {
    let state = RawSnapshot::capture(s);
    let reject = |s: &mut CpuBlockSession<'_>, p: &EncryptedBlockUpdate, field: &str| {
        assert_eq!(s.step(p).unwrap_err(), mismatch(field));
        state.assert_unchanged(s);
        assert!(!s.poisoned);
    };
    let owner = packet.owner.clone();
    packet.owner = Arc::new(());
    reject(s, packet, "block packet context identity");
    packet.owner = owner;
    let plan = packet.plan.clone();
    packet.plan = BlockPlanOwner::Synthetic(other_plan.clone());
    reject(s, packet, "block packet plan identity");
    packet.plan = plan;
    packet.index += 1;
    reject(s, packet, "block update index/replay");
    packet.index -= 1;
    packet.total += 1;
    reject(s, packet, "block logical denominator");
    packet.total -= 1;
    packet.micros[0].part.start += 1;
    reject(s, packet, "block ordered partition");
    packet.micros[0].part.start -= 1;
    let valid = packet.micros[0].part.valid;
    packet.micros[0].part.valid = 0;
    reject(s, packet, "block ordered partition");
    packet.micros[0].part.valid = valid;
    let micro = packet.micros.pop().unwrap();
    reject(s, packet, "block micro cardinality");
    packet.micros.push(micro);
    let target = packet.micros[0].targets.pop().unwrap();
    reject(s, packet, "block stage target cardinality");
    packet.micros[0].targets.push(target);
    let duplicate = packet.micros[0].targets[0].clone();
    packet.micros[0].targets.push(duplicate);
    reject(s, packet, "block stage target cardinality");
    packet.micros[0].targets.pop();
    packet.micros[0].targets.swap(0, 1);
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].targets.swap(0, 1);
    let original = packet.micros[0].x.clone();
    packet.micros[0].x = s
        .facade
        .base
        .evaluator
        .fork()
        .mod_switch_to_level(&original, original.level() - 1)
        .unwrap();
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].x = original;
    let original = packet.micros[0].targets[1].clone();
    packet.micros[0].targets[1] =
        Ciphertext::new(original.polys().to_vec(), original.scale() * 2.).unwrap();
    reject(s, packet, "ciphertext level/scale/components");
    packet.micros[0].targets[1] = original;
}
fn later_rate_negative(s: &mut CpuBlockSession<'_>, packet: &mut EncryptedBlockUpdate) {
    let state = RawSnapshot::capture(s);
    assert!(state.metadata.velocity.iter().flatten().all(|v| *v));
    let rate = s.plan.rates()[packet.index].kernel.to_bits();
    assert!(state.config.iter().flatten().all(|c| c.lr != rate));
    if let BlockTraining::NonInteractive(t) = &mut s.trainer {
        assert!(t.set_lr_reference(s.plan.reference()).is_err());
    }
    let target = packet.micros[0].targets.pop().unwrap();
    assert_eq!(
        s.step(packet).unwrap_err(),
        mismatch("block stage target cardinality")
    );
    state.assert_unchanged(s);
    packet.micros[0].targets.push(target);
}
fn config_only_i(context: &CpuEvaluationContext, refresh: &ClientRefreshService) {
    let layout = Layout::new(512, 16, 8).unwrap();
    let mut stages = Vec::new();
    for reference in [0.08, 0.04] {
        let mut nag = Nag::new(0.04, 0.9);
        nag.set_lr_reference(reference).unwrap();
        reserve_encoder();
        stages.push(
            Trainer::new(
                &context.evaluator,
                Box::new(refresh.handle()),
                vec![Box::new(nag)],
                layout,
            )
            .unwrap(),
        );
    }
    let mut block =
        BlockTrainer::new(&context.evaluator, stages, Box::new(refresh.handle())).unwrap();
    reserve(Work::ConfigSnapshot);
    let before = block
        .stages()
        .iter()
        .map(|t| i_config(t, 1))
        .collect::<Vec<_>>();
    let counter = context.counter();
    assert!(block.set_lr(0.06).is_err());
    reserve(Work::ConfigComparison);
    assert_eq!(
        before,
        block
            .stages()
            .iter()
            .map(|t| i_config(t, 1))
            .collect::<Vec<_>>()
    );
    assert_eq!(counter, context.counter());
    block.set_lr(0.04).unwrap();
    reserve(Work::ConfigComparison);
    assert_eq!(
        before,
        block
            .stages()
            .iter()
            .map(|t| i_config(t, 1))
            .collect::<Vec<_>>()
    );
    assert_eq!(counter, context.counter());
}
fn config_ni(s: &mut CpuBlockSession<'_>) {
    reserve(Work::ConfigSnapshot);
    let original = configs(s);
    let counter = s.facade.base.counter();
    if let BlockTraining::NonInteractive(t) = &mut s.trainer {
        t.stages_mut()
            .last_mut()
            .unwrap()
            .set_lr_reference(0.04)
            .unwrap();
    } else {
        panic!("NI atomic test");
    }
    reserve(Work::ConfigSnapshot);
    let lowered = configs(s);
    if let BlockTraining::NonInteractive(t) = &mut s.trainer {
        assert!(t.set_lr(0.06).is_err());
    }
    reserve(Work::ConfigComparison);
    assert_eq!(lowered, configs(s));
    assert_eq!(counter, s.facade.base.counter());
    if let BlockTraining::NonInteractive(t) = &mut s.trainer {
        t.stages_mut()
            .last_mut()
            .unwrap()
            .set_lr_reference(0.08)
            .unwrap();
    }
    reserve(Work::ConfigComparison);
    assert_eq!(original, configs(s));
    assert_eq!(counter, s.facade.base.counter());
}
fn compare(
    client: &StudyClient,
    s: &CpuBlockSession<'_>,
    oracle: &BlockOracle,
    update: usize,
    ni: bool,
) -> usize {
    let before = s.facade.base.counter();
    let snapshot = s.snapshot();
    let retention = s.retention_receipt(None, Some(&snapshot)).unwrap();
    eprintln!("P5_RETENTION endpoint={update} {retention:?}");
    let actual = client
        .inspect_block_snapshot(&snapshot, s.facade.base)
        .unwrap();
    assert_eq!(before, s.facade.base.counter());
    let expected = oracle.post(update).unwrap();
    let cap = if ni { 3e-5 } else { 1e-5 };
    assert_eq!(actual.stages().len(), s.descriptions.len());
    let mut maximum = 0_f64;
    for (stage, model) in actual.stages().iter().enumerate() {
        assert_eq!(model.layers().len(), 2);
        for (layer, ((w, b), expected)) in model
            .layers()
            .iter()
            .zip(&expected[stage * 2..stage * 2 + 2])
            .enumerate()
        {
            assert_eq!(w.dim(), expected.weights().dim());
            assert_eq!(b.len(), expected.biases().len());
            for ((row, col), value) in w.indexed_iter() {
                let truth = expected.weights()[[row, col]];
                let error = (value - truth).abs();
                eprintln!(
                    "P5_VALUE update={update} stage={stage} layer={layer} tensor=W row={row} col={col} actual={value:.17e} expected={truth:.17e} error={error:.17e} cap={cap}"
                );
                assert!(value.is_finite() && truth.is_finite() && error <= cap);
                maximum = maximum.max(error);
            }
            for (entry, (&value, &truth)) in b.iter().zip(expected.biases()).enumerate() {
                assert_eq!(truth.to_bits(), 0_f64.to_bits());
                let error = (value - truth).abs();
                eprintln!(
                    "P5_VALUE update={update} stage={stage} layer={layer} tensor=b entry={entry} actual={value:.17e} expected={truth:.17e} error={error:.17e} cap={cap}"
                );
                assert!(value.is_finite() && error <= cap);
                maximum = maximum.max(error);
            }
            for ct in s.model.stage(stage).layers[layer]
                .0
                .w
                .diagonals()
                .entries()
                .iter()
                .chain(std::iter::once(&s.model.stage(stage).layers[layer].0.b))
            {
                eprintln!(
                    "P5_METADATA update={update} stage={stage} layer={layer} components={} level={} scale_bits={} limb_lengths={:?}",
                    ct.len(),
                    ct.level(),
                    ct.scale().to_bits(),
                    ct.polys()
                        .iter()
                        .map(|p| p.limbs().iter().map(Vec::len).collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    let hidden = oracle.old_hidden(update).unwrap();
    assert_eq!(actual.handoffs().len(), hidden.len() - 1);
    for (boundary, parts) in actual.handoffs().iter().enumerate() {
        let mut offset = 0;
        for (micro, values) in parts.iter().enumerate() {
            for ((row, col), value) in values.indexed_iter() {
                let truth = hidden[boundary][[offset + row, col]];
                let error = (value - truth).abs();
                eprintln!(
                    "P5_VALUE update={update} boundary={boundary} micro={micro} tensor=old_hidden row={row} col={col} actual={value:.17e} expected={truth:.17e} error={error:.17e} cap={cap}"
                );
                assert!(value.is_finite() && truth.is_finite() && error <= cap);
                maximum = maximum.max(error);
            }
            offset += values.nrows();
        }
        assert_eq!(offset, hidden[boundary].nrows());
    }
    eprintln!(
        "P5_ENDPOINT update={update} maximum={maximum:.17e} cap={cap} config={:?} metadata={:?} counter={:?}",
        configs(s),
        raw_metadata(s),
        s.facade.base.counter()
    );
    actual.handoffs().iter().map(Vec::len).sum()
}
fn capture_prefix_initial(initial: &EncryptedBlockInitial) -> Vec<Vec<Ciphertext>> {
    reserve(Work::PrefixCapture);
    let captures = (0..initial.model.num_stages())
        .map(|stage| {
            initial
                .model
                .stage(stage)
                .layers
                .iter()
                .flat_map(|(d, _)| {
                    d.w.diagonals()
                        .entries()
                        .iter()
                        .chain(std::iter::once(&d.b))
                })
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    eprintln!(
        "P5_POISON_INITIAL_CAPTURE role=full_initial_expanded_model count={} bytes={} live_until=corresponding_failure_comparison",
        captures.iter().map(Vec::len).sum::<usize>(),
        captures
            .iter()
            .flatten()
            .map(|c| ciphertext_bytes(c).unwrap())
            .sum::<usize>()
    );
    captures
}
fn poison_contract(
    s: &mut CpuBlockSession<'_>,
    packet: &EncryptedBlockUpdate,
    initial: Vec<Vec<Ciphertext>>,
    ni: bool,
) {
    let mut receipt = s.retention_receipt(Some(packet), None).unwrap();
    receipt
        .include_diagnostic_copy(
            "poison initial expanded model; until corresponding failure comparison",
            initial.iter().flatten(),
        )
        .unwrap();
    eprintln!("P5_RETENTION poison_initial_live {receipt:?}");
    let before = s.facade.base.counter();
    assert!(s.step(packet).is_err());
    assert!(s.poisoned);
    assert_eq!(s.completed, 0);
    assert_eq!(s.samples, 0);
    let failure = s.failed.as_ref().unwrap();
    assert_eq!(failure.phase, "trainer");
    assert_eq!(failure.measured, &s.facade.base.counter() - &before);
    assert!(!failure.traces[0].is_empty());
    assert!(failure.traces[1].is_empty());
    let source = s.plan.transition(0, 0).update_metadata().unwrap();
    s.check_state(0, source.post_state(), 1).unwrap();
    s.check_state(1, &s.plan.initial()[1], 0).unwrap();
    let current = (0..s.model.num_stages())
        .map(|stage| {
            s.model
                .stage(stage)
                .layers
                .iter()
                .flat_map(|(d, _)| {
                    d.w.diagonals()
                        .entries()
                        .iter()
                        .chain(std::iter::once(&d.b))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(initial.len(), current.len());
    let mut source_changed = false;
    for (stage, (expected, actual)) in initial.iter().zip(current).enumerate() {
        assert_eq!(expected.len(), actual.len());
        for (a, b) in expected.iter().zip(actual) {
            if stage == 0 {
                source_changed |= !ciphertext_exact(a, b);
            } else {
                assert!(ciphertext_exact(a, b));
            }
        }
    }
    assert!(source_changed);
    if ni {
        assert_eq!(s.facade.boots.get(), 5);
        assert_eq!(failure.measured.bootstraps, 5);
        if let BlockTraining::NonInteractive(t) = &s.trainer {
            assert!(t.last_handoff_bootstraps().is_empty());
        }
        assert_eq!(failure.retained_boundary_outputs, 0);
    } else {
        assert_eq!(s.facade.refreshes.get(), 15);
        assert_eq!(failure.retained_boundary_outputs, 1);
    }
    // Addendum: these two model captures are distinct from the 46 full snapshots.
    drop(initial);
    let snapshot = RawSnapshot::capture(s);
    assert_eq!(
        s.step(packet).unwrap_err(),
        mismatch("poisoned block session")
    );
    snapshot.assert_unchanged(s);
    eprintln!("P5_POISON_FAILURE ni={ni} actual={:?}", s.failed);
}
#[test]
#[ignore = "requires independent root frozen-source CPU runtime release"]
fn p5_cpu_block_bridge_contract() {
    assert!(std::mem::size_of::<BlockPlanOwner>() > 0);
    assert!(std::mem::size_of::<CpuBlockSession<'static>>() > 0);
    let budget = Rc::new(Budget::default());
    ACTIVE.with(|active| {
        assert!(active.borrow().is_none());
        *active.borrow_mut() = Some(budget.clone());
    });
    let _active = Active;
    reserve(Work::Params);
    let ip = Arc::new(
        CkksParams::new(10, &[60, 40, 40, 40, 40, 40, 40, 40, 40, 40], &[60], 40, 10).unwrap(),
    );
    reserve(Work::Params);
    let (q, p) = ckks_train_ckks::params::n16_boot_bits();
    let np = Arc::new(CkksParams::new(10, &q, &p, 42, 8).unwrap());
    let bp = BootParams {
        log_slots: 9,
        ..BootParams::default_n16()
    };
    reserve(Work::Proof);
    let proof = DftSupportProof::verify_current().unwrap();
    reserve(Work::Descriptor);
    let bm = Arc::new(BootstrapMetadata::new_with_verified_support(&np, &bp, &proof).unwrap());
    let planning = Arc::new(PlanningBudget::new(PlanningUsage {
        geometries: 36,
        initial_states: 36,
        updates: 100,
        handoffs: 58,
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
    let mut cases = Vec::new();
    for family in [BlockFamily::A, BlockFamily::B, BlockFamily::C] {
        for mode in [i, wn, au, aa] {
            let plan = Arc::new(
                SyntheticBlockPlan::build(
                    &family.recipe(),
                    mode,
                    params(mode),
                    planning.clone(),
                    false,
                )
                .unwrap(),
            );
            cases.push(Case {
                family,
                mode,
                plan,
                poison: false,
            });
        }
    }
    for mode in [i, au] {
        let family = BlockFamily::D;
        let plan = Arc::new(
            SyntheticBlockPlan::build(
                &family.recipe(),
                mode,
                params(mode),
                planning.clone(),
                false,
            )
            .unwrap(),
        );
        cases.push(Case {
            family,
            mode,
            plan,
            poison: false,
        });
    }
    for mode in [i, au] {
        let family = BlockFamily::A;
        let plan = Arc::new(
            SyntheticBlockPlan::build(&family.recipe(), mode, params(mode), planning.clone(), true)
                .unwrap(),
        );
        cases.push(Case {
            family,
            mode,
            plan,
            poison: true,
        });
    }
    assert_eq!(cases.len(), 16);
    assert_eq!(
        planning.usage(),
        PlanningUsage {
            geometries: 36,
            initial_states: 36,
            updates: 100,
            handoffs: 58,
            wrapper_calls: 0
        }
    );
    assert_eq!(cases.iter().map(|c| c.plan.index.len()).sum::<usize>(), 158);
    spy_contract(&cases[1].plan, params(wn), params(i));
    let union = |ni: bool| {
        let mut keys = cases
            .iter()
            .filter(|c| (c.mode != i) == ni)
            .flat_map(|c| {
                c.plan
                    .descriptions
                    .iter()
                    .flat_map(|d| d.geometry.rotation_steps().iter().copied())
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        keys
    };
    let (ic, ictx) = setup(&cases[0], ip.clone(), None, &union(false));
    let (nc, nctx) = setup(
        &cases[1],
        np.clone(),
        Some(CpuBootstrapDescription {
            parameters: bp.clone(),
            metadata: bm.clone(),
        }),
        &union(true),
    );
    let config_refresh = ic.refresh_service(rng(63));
    config_only_i(&ictx, &config_refresh);
    let mut phase = PairedOraclePhase::new();
    let mut successes = 0;
    let mut failures = 0;
    let mut returned_handoffs = [0usize; 2];
    for case in &cases {
        reserve(Work::Session);
        let ni = case.mode != i;
        let (client, base) = if ni { (&nc, &nctx) } else { (&ic, &ictx) };
        let context = base.fork();
        let facade = CpuBlockEvaluation::new(&context);
        let refresh = client.refresh_service(rng(64));
        let recipe = case.family.recipe();
        let oracle = if case.poison {
            None
        } else {
            Some(phase.oracle(case.family).unwrap())
        };
        let claim = case.plan.claim();
        let checked = prepare_blocks_then_make(
            BlockPreparationInput::Synthetic {
                plan: case.plan.clone(),
                claim: &claim,
            },
            context_parameters(&context),
            &case.plan.parts,
            Ok,
        )
        .unwrap();
        let (initial, binding) = client
            .encrypt_block_initial(checked, &context, &arrays(&recipe))
            .unwrap();
        let prefix = case.poison.then(|| capture_prefix_initial(&initial));
        let mut session = CpuBlockSession::from_owner(
            &facade,
            BlockPlanOwner::Synthetic(case.plan.clone()),
            initial,
            (!ni).then_some(&refresh),
        )
        .unwrap();
        if matches!(case.family, BlockFamily::A) && case.mode == au && !case.poison {
            config_ni(&mut session);
        }
        if case.poison {
            if ni {
                facade.fail_boot_after.set(Some(5));
            } else {
                facade.fail_refresh_after.set(Some(15));
            }
            let (x, y) = recipe.rows(0).unwrap();
            let packet = client
                .encrypt_block_rows(&binding, 0, &context, x, y)
                .unwrap();
            poison_contract(&mut session, &packet, prefix.unwrap(), ni);
            failures += 1;
            continue;
        }
        for update in 0..case.plan.parts.len() {
            let (x, y) = recipe.rows(update).unwrap();
            let mut packet = client
                .encrypt_block_rows(&binding, update, &context, x, y)
                .unwrap();
            if matches!(case.family, BlockFamily::C) && (case.mode == i || case.mode == au) {
                prework_negatives(&mut session, &mut packet, &cases[0].plan);
            }
            if matches!(case.family, BlockFamily::A)
                && update == 1
                && (case.mode == i || case.mode == au)
            {
                later_rate_negative(&mut session, &mut packet);
            }
            let completion = session.step(&packet).unwrap();
            assert_eq!(completion.completed_updates, update + 1);
            assert_eq!(session.completed_updates(), update + 1);
            assert!(!session.is_poisoned());
            eprintln!(
                "P5_COMPLETION family={:?} mode={:?} plan={} {completion:?}",
                case.family,
                case.mode,
                session.plan.identity()
            );
            let receipt = session.retention_receipt(Some(&packet), None).unwrap();
            eprintln!("P5_RETENTION packet_live {receipt:?}");
            compare(client, &session, oracle.as_ref().unwrap(), update, ni);
            let unchanged = RawSnapshot::capture(&session);
            let field = if update + 1 == case.plan.parts.len() {
                "exhausted block session"
            } else {
                "block update index/replay"
            };
            assert_eq!(session.step(&packet).unwrap_err(), mismatch(field));
            unchanged.assert_unchanged(&session);
            successes += 1;
        }
    }
    assert_eq!((successes, failures), (40, 2));
    assert_eq!(returned_handoffs, [22, 62]);
    eprintln!(
        "P5_SUCCESS_HANDOFFS I={} NI={} total=84",
        returned_handoffs[0], returned_handoffs[1]
    );
    phase.check_complete().unwrap();
    assert_eq!(phase.actual_counts(), [14, 14, 40, 128]);
    for (ordinal, cap) in CAPS.iter().enumerate() {
        let actual = budget.counts[ordinal].get();
        if ordinal == Work::Boot as usize || ordinal == Work::Refresh as usize {
            assert!(actual <= *cap);
        } else {
            assert_eq!(
                actual, *cap,
                "P5 expensive/diagnostic count ordinal={ordinal}"
            );
        }
        eprintln!("P5_FINAL_COUNT ordinal={ordinal} actual={actual} cap={cap}");
    }
    eprintln!(
        "NONSCIENTIFIC_TEST_EVIDENCE P5 matrix16 sessions42 attempts40 successes2 poison; all-entry caps I1e-5 NI3e-5; canonical owned D7/start/resource/security/Task54 acceptance unclaimed"
    );
}
