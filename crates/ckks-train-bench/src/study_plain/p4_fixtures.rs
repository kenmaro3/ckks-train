//! NONSCIENTIFIC_TEST_EVIDENCE: bounded synthetic sensitivity, never HE admission.

use super::{Engine, PlainLayer, RateCoefficients, StageSpec, forward};
use crate::study_profile::{Activation, Layer, LayerRole};
use ndarray::Array2;

const VERSION: &str = "task54-p4-pure-v2";
const CAPS: [(&str, f64); 2] = [("I", 1e-5), ("NI", 4e-4)];

#[derive(Clone, Copy, Debug)]
enum Candidate {
    Original,
    Signal,
}

#[derive(Clone, Debug)]
struct Family {
    name: &'static str,
    classes: usize,
    width: usize,
    square: bool,
    partitions: Vec<Vec<usize>>,
    means: Vec<f64>,
}

impl Family {
    fn delta(&self) -> f64 {
        if self.square {
            2. / self.classes as f64
        } else {
            1.
        }
    }

    fn engine(&self, candidate: Candidate) -> Engine<f64> {
        let specs = vec![
            Layer {
                name: "hidden".into(),
                input: 2,
                output: 4,
                activation: if self.square {
                    Activation::Square
                } else {
                    Activation::SquarePlusLinear
                },
                bias: self.square,
                role: LayerRole::Hidden,
            },
            Layer {
                name: "output".into(),
                input: 4,
                output: self.width,
                activation: if self.square {
                    Activation::Square
                } else {
                    Activation::Identity
                },
                bias: self.square,
                role: LayerRole::Output,
            },
        ];
        let layers = specs
            .iter()
            .enumerate()
            .map(|(layer, spec)| {
                let weights =
                    Array2::from_shape_fn((spec.output, spec.input), |(o, i)| match candidate {
                        Candidate::Original => (((o + 3 * i + 1) % 7) as i64 - 3) as f64 / 256.,
                        Candidate::Signal => match (self.square, layer, i) {
                            (true, 0, 0) => 0.5,
                            (false, 0, 0) => 0.25,
                            (_, 0, _) => 0.,
                            (true, _, _) => 0.875,
                            (false, _, _) => 0.25,
                        },
                    });
                let biases = (0..spec.output)
                    .map(|o| match candidate {
                        Candidate::Original if spec.bias => ((o % 3) as i64 - 1) as f64 / 256.,
                        _ => 0.,
                    })
                    .collect();
                PlainLayer::from_arrays(spec.name.clone(), weights, biases)
            })
            .collect();
        Engine {
            layers,
            specs,
            stages: vec![StageSpec {
                first: 0,
                input: None,
                representation: None,
            }],
            classes: self.classes,
            momentum: if self.square { None } else { Some(0.9) },
        }
    }
}

fn families() -> Vec<Family> {
    let mut result = Vec::new();
    for (square, names) in [
        (true, ["square-c3", "square-c2", "square-c10"]),
        (false, ["poly-c3-p4", "poly-c2-p2", "poly-c10-p16"]),
    ] {
        for (i, (classes, padded, parts)) in
            [(3, 4, vec![8]), (2, 2, vec![5]), (10, 16, vec![8, 4])]
                .into_iter()
                .enumerate()
        {
            result.push(Family {
                name: names[i],
                classes,
                width: if square { classes } else { padded },
                square,
                partitions: vec![parts],
                means: vec![if square { 0.06 } else { 0.04 }],
            });
        }
    }
    result.push(Family {
        name: "poly-four-rate",
        classes: 3,
        width: 4,
        square: false,
        partitions: vec![vec![8, 4], vec![8], vec![5], vec![8, 4]],
        means: vec![0.04, 0.08, 0.02, 0.08],
    });
    result.push(Family {
        name: "poly-first-reference",
        classes: 3,
        width: 4,
        square: false,
        partitions: vec![vec![8], vec![8]],
        means: vec![0.08, 0.08],
    });
    result
}

#[derive(Clone, Copy, Debug)]
enum Control {
    Baseline,
    NoUpdate,
    OmittedFold,
    DoubleFold,
    WrongWidth,
    TenClasses,
    PhysicalDenominator,
    LostTail,
    WrongTargets,
    ConstantFirst,
    ConstantReference,
    LagRate,
    SkipDecrease,
    SkipRestart,
    ResetVelocity,
    Sgd,
    DropSecond,
    HalfMeans,
}

impl Control {
    fn first_affected(self, family: &Family) -> usize {
        match self {
            Self::ConstantFirst | Self::LagRate | Self::ResetVelocity | Self::DropSecond => 1,
            Self::SkipDecrease => 2,
            Self::SkipRestart => 3,
            _ => {
                assert!(!family.means.is_empty());
                0
            }
        }
    }

    fn status(self, family: &Family) -> &'static str {
        match self {
            Self::OmittedFold | Self::DoubleFold if family.delta() == 1. => "EQUIVALENT",
            Self::WrongWidth if family.square && family.classes.is_power_of_two() => "EQUIVALENT",
            Self::WrongWidth if !family.square && family.width == 2 => "EQUIVALENT",
            Self::TenClasses if family.delta() == 0.2 => "EQUIVALENT",
            Self::PhysicalDenominator | Self::LostTail if family.partitions[0].len() == 1 => {
                "INAPPLICABLE"
            }
            _ => "APPLICABLE",
        }
    }
}

fn controls(family: &Family) -> Vec<Control> {
    match family.means.len() {
        1 => vec![
            Control::NoUpdate,
            Control::OmittedFold,
            Control::DoubleFold,
            Control::WrongWidth,
            Control::TenClasses,
            Control::PhysicalDenominator,
            Control::LostTail,
            Control::WrongTargets,
        ],
        4 => vec![
            Control::NoUpdate,
            Control::ConstantFirst,
            Control::ConstantReference,
            Control::LagRate,
            Control::SkipDecrease,
            Control::SkipRestart,
            Control::ResetVelocity,
            Control::Sgd,
        ],
        2 => vec![
            Control::NoUpdate,
            Control::DropSecond,
            Control::HalfMeans,
            Control::ResetVelocity,
        ],
        _ => unreachable!("fixed finite families"),
    }
}

#[derive(Debug, Default)]
struct CandidateReport {
    actual_calls: usize,
    baseline_families: usize,
    control_rows: usize,
    harness_failures: Vec<String>,
    gate_failures: Vec<String>,
}

#[derive(Debug)]
struct Report {
    candidates: Vec<CandidateReport>,
    actual_calls: usize,
}

#[derive(Clone)]
struct Snapshot {
    layers: Vec<PlainLayer<f64>>,
}

struct Trajectory {
    initial: Snapshot,
    steps: Vec<Snapshot>,
}

fn input(candidate: Candidate, rows: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, 2), |(r, j)| match candidate {
        Candidate::Original => (((r + 2 * j) % 5) as i64 - 2) as f64 / 16.,
        Candidate::Signal if j == 0 => 1.,
        Candidate::Signal => ((r % 3) as i64 - 1) as f64 / 4.,
    })
}

fn snapshot(engine: &Engine<f64>, context: &str, report: &mut CandidateReport) -> Snapshot {
    for (layer, (state, spec)) in engine.layers.iter().zip(&engine.specs).enumerate() {
        for (tensor, entries) in [
            ("W", state.weights.iter().copied().collect::<Vec<_>>()),
            ("b", state.biases.clone()),
            ("Vw", state.weight_velocity.iter().copied().collect()),
            ("Vb", state.bias_velocity.clone()),
        ] {
            if entries.iter().any(|v| !v.is_finite()) {
                report
                    .gate_failures
                    .push(format!("{context} nonfinite {layer}/{tensor}"));
            }
            if !spec.bias
                && matches!(tensor, "b" | "Vb")
                && entries.iter().any(|v| v.to_bits() != 0_f64.to_bits())
            {
                report
                    .harness_failures
                    .push(format!("{context} frozen zero {layer}/{tensor}"));
            }
            println!(
                "NONSCIENTIFIC_TEST_EVIDENCE SNAPSHOT {context} layer={layer} tensor={tensor} shape={}x{} values={entries:?} bits={:?}",
                state.weights.nrows(),
                if matches!(tensor, "W" | "Vw") {
                    state.weights.ncols()
                } else {
                    1
                },
                entries.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
    Snapshot {
        layers: engine.layers.clone(),
    }
}

fn hidden_range(
    engine: &Engine<f64>,
    x: &Array2<f64>,
    context: &str,
    baseline: bool,
    report: &mut CandidateReport,
) {
    match forward(&engine.layers[0], &engine.specs[0], x) {
        Ok(cache) => {
            let a = cache.a.iter().fold(0_f64, |max, v| max.max(v.abs()));
            let z = cache.z.iter().fold(0_f64, |max, v| max.max(v.abs()));
            let a_min = cache.a.iter().copied().fold(f64::INFINITY, f64::min);
            let a_max = cache.a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let z_min = cache.z.iter().copied().fold(f64::INFINITY, f64::min);
            let z_max = cache.z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "NONSCIENTIFIC_TEST_EVIDENCE RANGE {context} hidden_activation_max={a:.17e} hidden_preactivation_max={z:.17e} activation_min={a_min:.17e} activation_max={a_max:.17e} preactivation_min={z_min:.17e} preactivation_max={z_max:.17e} baseline_gate={baseline} limit=7.5e-1"
            );
            if baseline && a > 0.75 {
                report
                    .gate_failures
                    .push(format!("{context} activation range {a:.17e}"));
            }
        }
        Err(error) => report
            .gate_failures
            .push(format!("{context} forward {error}")),
    }
}

fn trajectory(
    candidate: Candidate,
    family: &Family,
    control: Control,
    report: &mut CandidateReport,
) -> Trajectory {
    trajectory_guarded(candidate, family, control, report, None)
}
fn trajectory_guarded(
    candidate: Candidate,
    family: &Family,
    control: Control,
    report: &mut CandidateReport,
    guard: Option<&crate::study_diagonal::p4_tests::RuntimeBudget>,
) -> Trajectory {
    let mut engine = family.engine(candidate);
    if matches!(control, Control::Sgd) {
        engine.momentum = None;
    }
    let initial = snapshot(
        &engine,
        &format!("{candidate:?}/{} control={control:?} initial", family.name),
        report,
    );
    let mut steps = Vec::new();
    for (index, (&base_mean, partitions)) in family.means.iter().zip(&family.partitions).enumerate()
    {
        let rows: usize = partitions.iter().sum();
        let kept = if matches!(control, Control::LostTail) {
            partitions[0]
        } else {
            rows
        };
        assert!((1..=12).contains(&rows));
        assert!(family.width <= 16);
        assert_eq!(engine.layers.len(), 2);
        let (x, mut labels, baseline_rate) = baseline_rows(candidate, family, kept, base_mean);
        if matches!(control, Control::WrongTargets) {
            labels.fill(0);
        }
        let mut mean = baseline_rate.mean;
        let mut delta = baseline_rate.delta;
        match control {
            Control::OmittedFold => delta = 1.,
            Control::DoubleFold => mean *= delta,
            Control::WrongWidth => {
                delta = 2.
                    / if family.square {
                        family.classes.next_power_of_two()
                    } else {
                        family.width
                    } as f64
            }
            Control::TenClasses => delta = 0.2,
            Control::PhysicalDenominator => mean *= rows as f64 / partitions[0] as f64,
            Control::ConstantFirst => mean = family.means[0],
            Control::ConstantReference => mean = 0.08,
            Control::LagRate => mean = family.means[index.saturating_sub(1)],
            Control::SkipDecrease if index == 2 => mean = 0.08,
            Control::SkipRestart if index == 3 => mean = 0.02,
            Control::HalfMeans => mean *= 0.5,
            _ => {}
        }
        if matches!(control, Control::ResetVelocity) && index > 0 {
            for layer in &mut engine.layers {
                layer.weight_velocity.fill(0.);
                layer.bias_velocity.fill(0.);
            }
        }
        let rate = RateCoefficients {
            source: baseline_rate.source,
            mean,
            delta,
            kernel: mean * delta,
        };
        let context = format!(
            "{candidate:?}/{} control={control:?} update={index}",
            family.name
        );
        println!(
            "NONSCIENTIFIC_TEST_EVIDENCE CALL_INPUT {context} shape=[2,4,{}] classes={} partitions={partitions:?} rows={rows} kept={kept} x={x:?} labels={labels:?} source_bits={} mean_bits={} delta_bits={} kernel_bits={} reference_bits={} momentum_bits={:?}",
            family.width,
            family.classes,
            rate.source.to_bits(),
            mean.to_bits(),
            delta.to_bits(),
            rate.kernel.to_bits(),
            (if family.square {
                0.06 * family.delta()
            } else {
                0.08
            })
            .to_bits(),
            engine.momentum.map(f64::to_bits)
        );
        let baseline = matches!(control, Control::Baseline);
        hidden_range(&engine, &x, &format!("{context} before"), baseline, report);
        let skip = matches!(control, Control::NoUpdate)
            || (matches!(control, Control::DropSecond) && index == 1);
        if !skip {
            assert!(report.actual_calls < 100, "fixed candidate actual-call cap");
            report.actual_calls += 1;
            println!(
                "NONSCIENTIFIC_TEST_EVIDENCE ACTUAL_TRAIN {context} candidate_call={}",
                report.actual_calls
            );
            if let Some(guard) = guard {
                guard.reserve_f64();
            }
            if let Err(error) = engine.train(&x, &[&labels], &rate) {
                report
                    .gate_failures
                    .push(format!("{context} train failed {error}"));
                println!(
                    "NONSCIENTIFIC_TEST_EVIDENCE TRAIN_FAILURE {context} error={error:?} actual_calls={}",
                    report.actual_calls
                );
                break;
            }
        }
        hidden_range(&engine, &x, &format!("{context} after"), baseline, report);
        steps.push(snapshot(&engine, &context, report));
    }
    Trajectory { initial, steps }
}

#[derive(Debug)]
struct Witness {
    layer: usize,
    tensor: &'static str,
    row: usize,
    col: usize,
    baseline: f64,
    control: f64,
    difference: f64,
}

fn witness(a: &Snapshot, b: &Snapshot, family: &Family, only_layer: Option<usize>) -> Witness {
    let mut result = Witness {
        layer: 0,
        tensor: "W",
        row: 0,
        col: 0,
        baseline: 0.,
        control: 0.,
        difference: -1.,
    };
    for (layer, (left, right)) in a.layers.iter().zip(&b.layers).enumerate() {
        if only_layer.is_some_and(|wanted| wanted != layer) {
            continue;
        }
        for ((row, col), &value) in left.weights.indexed_iter() {
            let other = right.weights[[row, col]];
            let difference = (value - other).abs();
            if difference > result.difference {
                result = Witness {
                    layer,
                    tensor: "W",
                    row,
                    col,
                    baseline: value,
                    control: other,
                    difference,
                };
            }
        }
        if family.square && only_layer.is_none() {
            for (row, (&value, &other)) in left.biases.iter().zip(&right.biases).enumerate() {
                let difference = (value - other).abs();
                if difference > result.difference {
                    result = Witness {
                        layer,
                        tensor: "b",
                        row,
                        col: 0,
                        baseline: value,
                        control: other,
                        difference,
                    };
                }
            }
        }
    }
    result
}

fn check_margin(context: &str, witness: Witness, report: &mut CandidateReport) {
    for (backend, epsilon) in CAPS {
        let distance = witness.difference / epsilon;
        let pass = distance.is_finite() && distance >= 8.;
        println!(
            "NONSCIENTIFIC_TEST_EVIDENCE MARGIN {context} backend={backend} epsilon={epsilon:.17e} epsilon_bits={} layer={} tensor={} coordinate=[{},{}] baseline={:.17e} baseline_bits={} control={:.17e} control_bits={} max_abs_difference={:.17e} max_normalized_distance={distance:.17e} required=8 status={}",
            epsilon.to_bits(),
            witness.layer,
            witness.tensor,
            witness.row,
            witness.col,
            witness.baseline,
            witness.baseline.to_bits(),
            witness.control,
            witness.control.to_bits(),
            witness.difference,
            if pass { "PASS" } else { "FAIL" }
        );
        if !pass {
            report
                .gate_failures
                .push(format!("{context} backend={backend} D={distance:.17e}"));
        }
    }
}

fn run_preflight() -> Report {
    println!(
        "NONSCIENTIFIC_TEST_EVIDENCE MANIFEST version={VERSION} candidates=[Original,Signal] cap_per_candidate=100 cap_total=200 hidden_activation_limit=0.75 no_HE=true update_indices=zero_based"
    );
    let mut reports = Vec::new();
    for candidate in [Candidate::Original, Candidate::Signal] {
        let mut report = CandidateReport::default();
        for family in families() {
            let baseline = trajectory(candidate, &family, Control::Baseline, &mut report);
            report.baseline_families += 1;
            if let Some(first) = baseline.steps.first() {
                for layer in 0..2 {
                    check_margin(
                        &format!(
                            "{candidate:?}/{} control=NoUpdatePerLayer update=0 required_layer={layer}",
                            family.name
                        ),
                        witness(first, &baseline.initial, &family, Some(layer)),
                        &mut report,
                    );
                }
            }
            for control in controls(&family) {
                report.control_rows += 1;
                let status = control.status(&family);
                let index = control.first_affected(&family);
                let affected_label = if status == "APPLICABLE" {
                    index.to_string()
                } else {
                    "NONE".into()
                };
                println!(
                    "NONSCIENTIFIC_TEST_EVIDENCE CONTROL {candidate:?}/{} control={control:?} status={status} first_affected_update={affected_label}",
                    family.name
                );
                if status != "APPLICABLE" {
                    continue;
                }
                let mutant = if matches!(control, Control::NoUpdate) {
                    None
                } else {
                    Some(trajectory(candidate, &family, control, &mut report))
                };
                let other = mutant.as_ref().and_then(|t| t.steps.get(index)).or({
                    if matches!(control, Control::NoUpdate) {
                        Some(&baseline.initial)
                    } else {
                        None
                    }
                });
                if let (Some(correct), Some(other)) = (baseline.steps.get(index), other) {
                    check_margin(
                        &format!(
                            "{candidate:?}/{} control={control:?} update={index}",
                            family.name
                        ),
                        witness(correct, other, &family, None),
                        &mut report,
                    );
                } else {
                    report.gate_failures.push(format!(
                        "{candidate:?}/{} {control:?} missing first-affected snapshot {index}",
                        family.name
                    ));
                }
            }
        }
        println!(
            "NONSCIENTIFIC_TEST_EVIDENCE CANDIDATE {candidate:?} actual_calls={} baseline_families={} control_rows={} harness_failures={:?} gate_failures={:?} gate_status={}",
            report.actual_calls,
            report.baseline_families,
            report.control_rows,
            report.harness_failures,
            report.gate_failures,
            if report.gate_failures.is_empty() {
                "PASS"
            } else {
                "FAIL"
            }
        );
        reports.push(report);
    }
    let actual_calls = reports.iter().map(|r| r.actual_calls).sum();
    println!(
        "NONSCIENTIFIC_TEST_EVIDENCE TOTAL actual_calls={actual_calls} candidates=2 constructors_HE=0"
    );
    Report {
        candidates: reports,
        actual_calls,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn p4_pure_sensitivity_gate() {
        let report = super::run_preflight();
        assert_eq!(report.candidates.len(), 2);
        assert!(report.actual_calls <= 200);
        for candidate in &report.candidates {
            assert!(candidate.actual_calls <= 100);
            assert_eq!(candidate.baseline_families, 8);
            assert_eq!(candidate.control_rows, 60);
            assert!(candidate.harness_failures.is_empty(), "{candidate:#?}");
        }
        assert!(report.candidates[1].gate_failures.is_empty(), "{report:#?}");
    }
}

// Closed sibling-consumer API. The existing Family::engine and trajectory remain the
// sole initial-array and arithmetic owners; no mutant trajectory is invoked here.
#[derive(Clone, Copy, Debug)]
pub(crate) enum SignalFamily {
    SquareC3,
    SquareC2,
    SquareC10,
    PolyC3,
    PolyC2,
    PolyC10,
    PolyFourRate,
    PolyFirstReference,
}
impl SignalFamily {
    pub(crate) const SMOKE: [Self; 6] = [
        Self::SquareC3,
        Self::SquareC2,
        Self::SquareC10,
        Self::PolyC3,
        Self::PolyC2,
        Self::PolyC10,
    ];
    pub(crate) fn recipe(self) -> SignalRecipe {
        let index = match self {
            Self::SquareC3 => 0,
            Self::SquareC2 => 1,
            Self::SquareC10 => 2,
            Self::PolyC3 => 3,
            Self::PolyC2 => 4,
            Self::PolyC10 => 5,
            Self::PolyFourRate => 6,
            Self::PolyFirstReference => 7,
        };
        SignalRecipe {
            family: families().remove(index),
        }
    }
}
fn baseline_rows(
    candidate: Candidate,
    family: &Family,
    rows: usize,
    mean: f64,
) -> (Array2<f64>, Vec<u8>, RateCoefficients<f64>) {
    // r spans the complete logical update, including every subsequent micro.
    (
        input(candidate, rows),
        (0..rows).map(|r| (r % family.classes) as u8).collect(),
        RateCoefficients {
            source: mean,
            mean,
            delta: family.delta(),
            kernel: mean * family.delta(),
        },
    )
}
pub(crate) struct SignalRecipe {
    family: Family,
}
impl SignalRecipe {
    pub(crate) fn classes(&self) -> usize {
        self.family.classes
    }
    pub(crate) fn width(&self) -> usize {
        self.family.width
    }
    pub(crate) fn square(&self) -> bool {
        self.family.square
    }
    pub(crate) fn partitions(&self) -> &[Vec<usize>] {
        &self.family.partitions
    }
    pub(crate) fn rates(&self) -> Vec<RateCoefficients<f64>> {
        self.family
            .means
            .iter()
            .map(|&mean| baseline_rows(Candidate::Signal, &self.family, 0, mean).2)
            .collect()
    }
    pub(crate) fn reference(&self) -> f64 {
        if self.family.square {
            0.06 * self.family.delta()
        } else {
            0.08
        }
    }
}
pub(crate) struct SignalOracle {
    recipe: SignalRecipe,
    initial: Snapshot,
    steps: Vec<Snapshot>,
    rows: Vec<(Array2<f64>, Vec<u8>)>,
}
impl SignalOracle {
    pub(crate) fn recipe(&self) -> &SignalRecipe {
        &self.recipe
    }
    pub(crate) fn initial(&self) -> &[PlainLayer<f64>] {
        &self.initial.layers
    }
    pub(crate) fn post(&self, update: usize) -> Option<&[PlainLayer<f64>]> {
        self.steps.get(update).map(|s| s.layers.as_slice())
    }
    pub(crate) fn rows(&self, update: usize) -> Option<(&Array2<f64>, &[u8])> {
        self.rows.get(update).map(|(x, y)| (x, y.as_slice()))
    }
}
pub(crate) fn signal_oracle(
    family: SignalFamily,
    guard: &crate::study_diagonal::p4_tests::RuntimeBudget,
) -> Result<SignalOracle, String> {
    guard.reserve_oracle();
    let recipe = family.recipe();
    let mut report = CandidateReport::default();
    let trajectory = trajectory_guarded(
        Candidate::Signal,
        &recipe.family,
        Control::Baseline,
        &mut report,
        Some(guard),
    );
    if !report.harness_failures.is_empty()
        || !report.gate_failures.is_empty()
        || report.actual_calls != recipe.family.means.len()
        || trajectory.steps.len() != recipe.family.means.len()
    {
        return Err(format!(
            "Signal baseline failed: {:?} {:?}",
            report.harness_failures, report.gate_failures
        ));
    }
    let rows = recipe
        .family
        .partitions
        .iter()
        .zip(&recipe.family.means)
        .map(|(parts, &mean)| {
            let (x, y, _) =
                baseline_rows(Candidate::Signal, &recipe.family, parts.iter().sum(), mean);
            (x, y)
        })
        .collect();
    Ok(SignalOracle {
        recipe,
        initial: trajectory.initial,
        steps: trajectory.steps,
        rows,
    })
}
