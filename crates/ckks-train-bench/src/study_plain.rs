//! Deterministic actual-f32/f64 plaintext arithmetic over immutable study inputs.
//!
//! Scalar loops and update order are specified in `docs/study-training-design.md` §11.
//! This library neither runs a full campaign nor creates a reference artifact or target.
use crate::{
    study_inputs::{
        CastPolicy, Dtype, MaterializedProfile, NumericInputs, RateCoefficients, StageInput,
    },
    study_profile::{Activation, Layer, Loss, Model, Optimizer},
};
use ndarray::Array2;
use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, Div, Mul, Sub},
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}
/// Scalar arithmetic restricted to native binary32 and binary64.
///
/// Coefficient conversion occurs once at construction; counts convert directly to T.
/// No arithmetic operation in the engine widens to another floating-point type.
pub trait PlainScalar:
    sealed::Sealed
    + Copy
    + fmt::Debug
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// Zero in the chosen arithmetic type.
    #[doc(hidden)]
    fn zero() -> Self;
    /// One in the chosen arithmetic type.
    #[doc(hidden)]
    fn one() -> Self;
    /// Convert an integer count for native arithmetic.
    #[doc(hidden)]
    fn count(value: usize) -> Self;
    /// Convert a metadata coefficient once.
    #[doc(hidden)]
    fn coefficient(value: f64) -> Self;
    /// Reject infinities and NaNs.
    #[doc(hidden)]
    fn finite(self) -> bool;
    /// D1's floating total order, including signed zeros.
    #[doc(hidden)]
    fn total_order(self, other: Self) -> Ordering;
    /// Select the matching already materialized numeric view.
    #[doc(hidden)]
    fn inputs(profile: &MaterializedProfile) -> NumericInputs<'_, Self>;
}
macro_rules! scalar {
    ($t:ty, $view:ident) => {
        impl PlainScalar for $t {
            fn zero() -> Self {
                0.
            }
            fn one() -> Self {
                1.
            }
            fn count(value: usize) -> Self {
                value as Self
            }
            fn coefficient(value: f64) -> Self {
                value as Self
            }
            fn finite(self) -> bool {
                self.is_finite()
            }
            fn total_order(self, other: Self) -> Ordering {
                self.total_cmp(&other)
            }
            fn inputs(profile: &MaterializedProfile) -> NumericInputs<'_, Self> {
                profile.$view()
            }
        }
    };
}
scalar!(f32, numeric_f32);
scalar!(f64, numeric_f64);

/// Explicit reference failures; none represents a successful scientific observation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlainError {
    /// The bound complete plan has no next update; state remains usable for evaluation.
    Exhausted,
    /// Numerical/data conventions, labels or shapes are incompatible.
    IncompatibleInputs(&'static str),
    /// A coefficient/count/intermediate/result was invalid or nonfinite.
    InvalidArithmetic(&'static str),
    /// A previous failed step made this state terminal; counters did not advance.
    Poisoned,
}
impl fmt::Display for PlainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PlainError {}
type Result<T> = std::result::Result<T, PlainError>;
fn checked<T: PlainScalar>(value: T, where_: &'static str) -> Result<T> {
    if value.finite() {
        Ok(value)
    } else {
        Err(PlainError::InvalidArithmetic(where_))
    }
}
fn positive<T: PlainScalar>(value: T, where_: &'static str) -> Result<T> {
    if value.finite() && value > T::zero() {
        Ok(value)
    } else {
        Err(PlainError::InvalidArithmetic(where_))
    }
}

/// Read-only parameters and persistent unscaled optimizer buffers for one named layer.
#[derive(Debug, Clone)]
pub struct PlainLayer<T> {
    name: String,
    weights: Array2<T>,
    biases: Vec<T>,
    weight_velocity: Array2<T>,
    bias_velocity: Vec<T>,
}
impl<T: PlainScalar> PlainLayer<T> {
    fn from_arrays(name: String, weights: Array2<T>, biases: Vec<T>) -> Self {
        let weight_velocity = Array2::from_elem(weights.dim(), T::zero());
        let bias_velocity = vec![T::zero(); biases.len()];
        Self {
            name,
            weights,
            biases,
            weight_velocity,
            bias_velocity,
        }
    }
    /// Registered canonical layer name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Current output-by-input logical weights.
    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }
    /// Current active biases, or exact frozen zeros.
    pub fn biases(&self) -> &[T] {
        &self.biases
    }
    /// Persistent unscaled NAG velocity; SGD retains zero.
    pub fn weight_velocity(&self) -> &Array2<T> {
        &self.weight_velocity
    }
    /// Persistent bias velocity; frozen biases retain zero.
    pub fn bias_velocity(&self) -> &[T] {
        &self.bias_velocity
    }
}
#[derive(Debug)]
struct StageSpec {
    first: usize,
    input: Option<usize>,
    representation: Option<usize>,
}
#[derive(Debug)]
struct Engine<T> {
    layers: Vec<PlainLayer<T>>,
    specs: Vec<Layer>,
    stages: Vec<StageSpec>,
    classes: usize,
    momentum: Option<T>,
}
/// A private mutable model bound for its lifetime to one immutable materialized profile.
///
/// No row/rate/index argument is accepted by [`step_plain`]. A failed step poisons this
/// state; arrays remain inspectable but further steps/predictions/evaluation fail.
#[derive(Debug)]
pub struct PlainState<'a, T: PlainScalar> {
    profile: &'a MaterializedProfile,
    inputs: NumericInputs<'a, T>,
    engine: Engine<T>,
    completed: usize,
    processed: usize,
    poisoned: bool,
}
impl<'a, T: PlainScalar> PlainState<'a, T> {
    /// Clone the bound initial model once and allocate zero unscaled velocity.
    ///
    /// Only the registered zero-decay Square/SGD and PolyRelu2/NAG conventions are admitted.
    pub fn new(profile: &'a MaterializedProfile) -> Result<Self> {
        let metadata = profile.metadata();
        let supported = match metadata.model {
            Model::Mlp2020 => {
                metadata.loss == Loss::MeanOutputMse && metadata.optimizer == Optimizer::Sgd
            }
            _ => {
                metadata.loss == Loss::RssDelta
                    && metadata.optimizer == Optimizer::Nesterov { momentum: 0.9 }
            }
        };
        if !supported || metadata.weight_decay != 0. {
            return Err(PlainError::IncompatibleInputs(
                "unsupported loss/optimizer/decay",
            ));
        }
        let inputs = T::inputs(profile);
        if inputs.rates().len() != profile.plan().updates.len() {
            return Err(PlainError::IncompatibleInputs("rate/plan count"));
        }
        let momentum = match metadata.optimizer {
            Optimizer::Sgd => None,
            Optimizer::Nesterov { momentum } => {
                Some(positive(T::coefficient(momentum), "momentum")?)
            }
        };
        let layers = inputs
            .initial()
            .layers()
            .iter()
            .map(|l| {
                PlainLayer::from_arrays(l.name().into(), l.weights().clone(), l.biases().to_vec())
            })
            .collect();
        let stages = profile
            .graph()
            .stages()
            .iter()
            .map(|s| StageSpec {
                first: s.layers().start,
                input: match s.input() {
                    StageInput::Dataset => None,
                    StageInput::PreUpdateActivation { layer } => Some(layer),
                },
                representation: s.representation_layer(),
            })
            .collect();
        let engine = Engine {
            layers,
            specs: metadata.layers.clone(),
            stages,
            classes: metadata.classes,
            momentum,
        };
        Ok(Self {
            profile,
            inputs,
            engine,
            completed: 0,
            processed: 0,
            poisoned: false,
        })
    }
    /// Canonical content/source identity shared by both numerical policies.
    pub fn input_identity(&self) -> &str {
        self.profile.identity().digest()
    }
    /// Actual arithmetic and storage dtype.
    pub fn dtype(&self) -> Dtype {
        self.inputs.dtype()
    }
    /// Canonical-input casting policy, independent from actual arithmetic.
    pub fn cast_policy(&self) -> CastPolicy {
        self.inputs.cast_policy()
    }
    /// Canonical ordered parameters and unscaled velocity; failure inspection is allowed.
    pub fn layers(&self) -> &[PlainLayer<T>] {
        &self.engine.layers
    }
    /// Successfully completed logical updates; also the next plan index.
    pub fn completed_updates(&self) -> usize {
        self.completed
    }
    /// Actual valid samples in successfully completed logical updates.
    pub fn processed_samples(&self) -> usize {
        self.processed
    }
    /// Whether a failed step has made this state terminal.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    fn usable(&self) -> Result<()> {
        if self.poisoned {
            Err(PlainError::Poisoned)
        } else {
            Ok(())
        }
    }
}
/// Successful exact planned-update accounting, without timing or performance claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlainStep {
    /// Consumed zero-based planned update.
    pub update_index: usize,
    /// Actual sample count of this logical update.
    pub valid_samples: usize,
    /// Total successfully consumed samples including this update.
    pub processed_samples: usize,
}
/// Consume precisely the next planned logical update and its once-cast rate.
///
/// Exhaustion is non-poisoning. Every other step failure poisons the state without
/// advancing counters; partial parameter mutation is possible, so retry is forbidden.
pub fn step_plain<T: PlainScalar>(state: &mut PlainState<'_, T>) -> Result<PlainStep> {
    state.usable()?;
    if state.completed == state.profile.plan().updates.len() {
        return Err(PlainError::Exhausted);
    }
    state.poisoned = true;
    let rows = state
        .profile
        .update_rows(state.completed)
        .map_err(|_| PlainError::IncompatibleInputs("planned rows"))?;
    let planned = rows.planned();
    if planned.index != state.completed
        || planned.processed_start != state.processed
        || planned.valid_samples != rows.rows().len()
    {
        return Err(PlainError::IncompatibleInputs("planned advancement"));
    }
    let next_processed = state
        .processed
        .checked_add(planned.valid_samples)
        .ok_or(PlainError::IncompatibleInputs("sample counter overflow"))?;
    let mut x = Array2::from_elem(
        (rows.rows().len(), state.inputs.train_x().ncols()),
        T::zero(),
    );
    let mut labels = Vec::with_capacity(rows.rows().len());
    for (i, &row) in rows.rows().iter().enumerate() {
        let row = row as usize;
        for j in 0..x.ncols() {
            x[[i, j]] = state.inputs.train_x()[[row, j]];
        }
        labels.push(state.inputs.train_y()[row]);
    }
    let stage_labels = vec![labels.as_slice(); state.engine.stages.len()];
    state
        .engine
        .train(&x, &stage_labels, &state.inputs.rates()[state.completed])?;
    let result = PlainStep {
        update_index: state.completed,
        valid_samples: planned.valid_samples,
        processed_samples: next_processed,
    };
    state.completed += 1;
    state.processed = next_processed;
    state.poisoned = false;
    Ok(result)
}
/// Predict one feature row in T, following representation layers and ignoring local heads.
pub fn predict_plain<T: PlainScalar>(state: &PlainState<'_, T>, features: &[T]) -> Result<Vec<T>> {
    state.usable()?;
    state.engine.predict(features)
}
/// Held-out integer counts; ratio conversion is reporting only, after classification in T.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlainEvaluation {
    /// Correct real-class argmax predictions.
    pub correct: usize,
    /// Exact held-out denominator.
    pub test_count: usize,
}
impl PlainEvaluation {
    /// Optional binary64 reporting ratio; no logits or arithmetic feed back into training.
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.test_count as f64
    }
}
/// Evaluate the bound held-out split without changing model, velocity or progress.
///
/// Argmax uses D1's last total-order maximum over real classes; every logical logit
/// (including padded channels) must be finite. No loss scalar is reported.
pub fn evaluate_plain<T: PlainScalar>(state: &PlainState<'_, T>) -> Result<PlainEvaluation> {
    state.usable()?;
    let test_count = state.inputs.test_y().len();
    if test_count == 0 || test_count != state.inputs.test_x().nrows() {
        return Err(PlainError::IncompatibleInputs("held-out count"));
    }
    let mut correct = 0;
    for (i, &label) in state.inputs.test_y().iter().enumerate() {
        let row = state.inputs.test_x().row(i).to_vec();
        let outputs = state.engine.predict(&row)?;
        if class_index(&outputs, state.engine.classes)? == usize::from(label) {
            correct += 1;
        }
    }
    Ok(PlainEvaluation {
        correct,
        test_count,
    })
}

struct Cache<T> {
    z: Array2<T>,
    a: Array2<T>,
}
struct StageCache<T> {
    input: Array2<T>,
    hidden: Cache<T>,
    output: Cache<T>,
}
struct Gradient<T> {
    weights: Array2<T>,
    biases: Vec<T>,
}
fn activate<T: PlainScalar>(z: T, activation: Activation) -> Result<T> {
    match activation {
        Activation::Identity => checked(z, "identity"),
        Activation::Square => checked(z * z, "square"),
        Activation::SquarePlusLinear => {
            checked(checked(z * z, "square")? + z, "square plus linear")
        }
    }
}
fn derivative<T: PlainScalar>(z: T, activation: Activation) -> Result<T> {
    let two = T::one() + T::one();
    match activation {
        Activation::Identity => Ok(T::one()),
        Activation::Square => checked(two * z, "square derivative"),
        Activation::SquarePlusLinear => checked(
            checked(two * z, "square derivative")? + T::one(),
            "poly derivative",
        ),
    }
}
fn forward<T: PlainScalar>(layer: &PlainLayer<T>, spec: &Layer, x: &Array2<T>) -> Result<Cache<T>> {
    if x.ncols() != spec.input
        || layer.weights.dim() != (spec.output, spec.input)
        || layer.biases.len() != spec.output
    {
        return Err(PlainError::IncompatibleInputs("dense shape"));
    }
    let mut z = Array2::from_elem((x.nrows(), spec.output), T::zero());
    let mut a = z.clone();
    for n in 0..x.nrows() {
        for o in 0..spec.output {
            let mut sum = T::zero();
            for i in 0..spec.input {
                let product = checked(layer.weights[[o, i]] * x[[n, i]], "forward product")?;
                sum = checked(sum + product, "forward sum")?;
            }
            let value = checked(sum + layer.biases[o], "forward bias")?;
            z[[n, o]] = value;
            a[[n, o]] = activate(value, spec.activation)?;
        }
    }
    Ok(Cache { z, a })
}
fn gradient<T: PlainScalar>(
    delta: &Array2<T>,
    input: &Array2<T>,
    active_bias: bool,
) -> Result<Gradient<T>> {
    let denominator = positive(T::count(input.nrows()), "actual valid sample count")?;
    let mut weights = Array2::from_elem((delta.ncols(), input.ncols()), T::zero());
    let mut biases = vec![T::zero(); delta.ncols()];
    for n in 0..input.nrows() {
        for o in 0..delta.ncols() {
            for i in 0..input.ncols() {
                let p = checked(delta[[n, o]] * input[[n, i]], "gradient product")?;
                weights[[o, i]] = checked(weights[[o, i]] + p, "gradient sum")?;
            }
            if active_bias {
                biases[o] = checked(biases[o] + delta[[n, o]], "bias gradient sum")?;
            }
        }
    }
    for w in &mut weights {
        *w = checked(*w / denominator, "mean gradient")?;
    }
    for b in &mut biases {
        *b = checked(*b / denominator, "mean bias gradient")?;
    }
    Ok(Gradient { weights, biases })
}
fn stage_gradients<T: PlainScalar>(
    engine: &Engine<T>,
    stage: &StageSpec,
    cache: &StageCache<T>,
    labels: &[u8],
    delta_scale: T,
) -> Result<[Gradient<T>; 2]> {
    let first = stage.first;
    let hidden_spec = &engine.specs[first];
    let output_spec = &engine.specs[first + 1];
    if labels.len() != cache.input.nrows()
        || labels.iter().any(|&y| usize::from(y) >= engine.classes)
    {
        return Err(PlainError::IncompatibleInputs("training labels"));
    }
    let mut output_delta = Array2::from_elem(cache.output.a.dim(), T::zero());
    let mut hidden_delta = Array2::from_elem(cache.hidden.a.dim(), T::zero());
    for n in 0..cache.input.nrows() {
        for o in 0..output_spec.output {
            let target = if o == usize::from(labels[n]) {
                T::one()
            } else {
                T::zero()
            };
            let raw = checked(cache.output.a[[n, o]] - target, "output residual")?;
            let scaled = checked(raw * delta_scale, "explicit delta scale")?;
            output_delta[[n, o]] = checked(
                scaled * derivative(cache.output.z[[n, o]], output_spec.activation)?,
                "output delta",
            )?;
        }
        for h in 0..hidden_spec.output {
            let mut sum = T::zero();
            for o in 0..output_spec.output {
                let p = checked(
                    engine.layers[first + 1].weights[[o, h]] * output_delta[[n, o]],
                    "backward product",
                )?;
                sum = checked(sum + p, "backward sum")?;
            }
            hidden_delta[[n, h]] = checked(
                sum * derivative(cache.hidden.z[[n, h]], hidden_spec.activation)?,
                "hidden delta",
            )?;
        }
    }
    Ok([
        gradient(&hidden_delta, &cache.input, hidden_spec.bias)?,
        gradient(&output_delta, &cache.hidden.a, output_spec.bias)?,
    ])
}
fn update_parameter<T: PlainScalar>(
    parameter: &mut T,
    velocity: &mut T,
    g: T,
    momentum: Option<T>,
    rate: T,
) -> Result<()> {
    let direction = if let Some(mu) = momentum {
        let retained = checked(mu * *velocity, "momentum product")?;
        let next = checked(retained + g, "unscaled velocity")?;
        let lookahead = checked(mu * next, "lookahead product")?;
        let direction = checked(g + lookahead, "NAG direction")?;
        *velocity = next;
        direction
    } else {
        g
    };
    let change = checked(rate * direction, "parameter change")?;
    *parameter = checked(*parameter - change, "parameter update")?;
    Ok(())
}
fn apply_gradient<T: PlainScalar>(
    layer: &mut PlainLayer<T>,
    gradient: &Gradient<T>,
    active_bias: bool,
    momentum: Option<T>,
    rate: T,
) -> Result<()> {
    for o in 0..layer.weights.nrows() {
        for i in 0..layer.weights.ncols() {
            update_parameter(
                &mut layer.weights[[o, i]],
                &mut layer.weight_velocity[[o, i]],
                gradient.weights[[o, i]],
                momentum,
                rate,
            )?;
        }
    }
    if active_bias {
        for o in 0..layer.biases.len() {
            update_parameter(
                &mut layer.biases[o],
                &mut layer.bias_velocity[o],
                gradient.biases[o],
                momentum,
                rate,
            )?;
        }
    }
    Ok(())
}
impl<T: PlainScalar> Engine<T> {
    fn train(&mut self, x: &Array2<T>, labels: &[&[u8]], rate: &RateCoefficients<T>) -> Result<()> {
        positive(rate.mean, "mean rate")?;
        positive(rate.delta, "delta multiplier")?;
        if x.nrows() == 0 || labels.len() != self.stages.len() {
            return Err(PlainError::IncompatibleInputs("empty batch/stage targets"));
        }
        // All stage inputs/forwards are captured before even the first parameter mutates.
        let mut caches: Vec<StageCache<T>> = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let input = if let Some(layer) = stage.input {
                let position = self
                    .stages
                    .iter()
                    .position(|s| s.representation == Some(layer))
                    .ok_or(PlainError::IncompatibleInputs("representation source"))?;
                caches
                    .get(position)
                    .ok_or(PlainError::IncompatibleInputs("representation order"))?
                    .hidden
                    .a
                    .clone()
            } else {
                x.clone()
            };
            let hidden = forward(&self.layers[stage.first], &self.specs[stage.first], &input)?;
            let output = forward(
                &self.layers[stage.first + 1],
                &self.specs[stage.first + 1],
                &hidden.a,
            )?;
            caches.push(StageCache {
                input,
                hidden,
                output,
            });
        }
        let mut gradients = Vec::with_capacity(self.layers.len());
        for (i, stage) in self.stages.iter().enumerate() {
            gradients.extend(stage_gradients(
                self, stage, &caches[i], labels[i], rate.delta,
            )?);
        }
        for (i, gradient) in gradients.iter().enumerate() {
            apply_gradient(
                &mut self.layers[i],
                gradient,
                self.specs[i].bias,
                self.momentum,
                rate.mean,
            )?;
        }
        Ok(())
    }
    fn predict(&self, features: &[T]) -> Result<Vec<T>> {
        if features.len() != self.specs[0].input || features.iter().any(|v| !v.finite()) {
            return Err(PlainError::IncompatibleInputs(
                "prediction feature shape/nonfinite",
            ));
        }
        let mut representations: Vec<(usize, Array2<T>)> = Vec::new();
        let dataset = Array2::from_shape_vec((1, features.len()), features.to_vec())
            .map_err(|_| PlainError::IncompatibleInputs("prediction shape"))?;
        for stage in &self.stages {
            let input = if let Some(layer) = stage.input {
                &representations
                    .iter()
                    .find(|(index, _)| *index == layer)
                    .ok_or(PlainError::IncompatibleInputs("inference representation"))?
                    .1
            } else {
                &dataset
            };
            let hidden = forward(&self.layers[stage.first], &self.specs[stage.first], input)?;
            if let Some(layer) = stage.representation {
                representations.push((layer, hidden.a));
            } else {
                return Ok(forward(
                    &self.layers[stage.first + 1],
                    &self.specs[stage.first + 1],
                    &hidden.a,
                )?
                .a
                .row(0)
                .to_vec());
            }
        }
        Err(PlainError::IncompatibleInputs("no inference output stage"))
    }
}
fn class_index<T: PlainScalar>(logits: &[T], classes: usize) -> Result<usize> {
    if classes == 0 || classes > logits.len() {
        return Err(PlainError::IncompatibleInputs("accuracy class width"));
    }
    for &value in logits {
        checked(value, "prediction logit")?;
    }
    let mut best = 0;
    for i in 1..classes {
        if logits[i].total_order(logits[best]) != Ordering::Less {
            best = i;
        }
    }
    Ok(best)
}

#[cfg(test)]
pub(crate) mod p4_fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study_inputs::materialize;
    use crate::study_profile::{LayerRole, StudyRegistry};
    use ndarray::{Array2, arr2};
    use std::path::Path;

    // Private synthetic models cannot construct a scientific MaterializedProfile.
    fn fixture(stages: usize, square: bool, bias: bool) -> Engine<f64> {
        let mut layers = Vec::new();
        let mut specs = Vec::new();
        let mut graph = Vec::new();
        for s in 0..stages {
            specs.push(Layer {
                name: format!("h{s}"),
                input: 1,
                output: 1,
                activation: if square {
                    Activation::Square
                } else {
                    Activation::SquarePlusLinear
                },
                bias,
                role: if s + 1 < stages {
                    LayerRole::LocalForward { block: s }
                } else {
                    LayerRole::Hidden
                },
            });
            specs.push(Layer {
                name: format!("o{s}"),
                input: 1,
                output: 3,
                activation: if square {
                    Activation::Square
                } else {
                    Activation::Identity
                },
                bias,
                role: if s + 1 < stages {
                    LayerRole::LocalHead { block: s }
                } else {
                    LayerRole::Output
                },
            });
            layers.push(PlainLayer::from_arrays(
                format!("h{s}"),
                arr2(&[[0.5]]),
                vec![0.],
            ));
            layers.push(PlainLayer::from_arrays(
                format!("o{s}"),
                arr2(&[[0.5], [1.], [0.]]),
                vec![0.; 3],
            ));
            graph.push(StageSpec {
                first: 2 * s,
                input: if s == 0 { None } else { Some(2 * (s - 1)) },
                representation: (s + 1 < stages).then_some(2 * s),
            });
        }
        Engine {
            layers,
            specs,
            stages: graph,
            classes: 3,
            momentum: if square { None } else { Some(0.9) },
        }
    }
    fn rate(mean: f64, delta: f64) -> RateCoefficients<f64> {
        RateCoefficients {
            source: mean,
            mean,
            delta,
            kernel: mean * delta,
        }
    }
    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a:.17e} != {b:.17e}");
    }

    // Catches wrong Square derivatives, missing 2/C, bias drift and rate/delta double folding.
    #[test]
    fn square_sgd_has_independent_hand_calculated_gradients() {
        let mut e = fixture(1, true, true);
        e.train(&arr2(&[[2.]]), &[&[0]], &rate(0.1, 2. / 3.))
            .unwrap();
        close(e.layers[0].weights[[0, 0]], 0.5 - 0.1 * (13. / 3.), 1e-15);
        close(e.layers[0].biases[0], -0.1 * (13. / 6.), 1e-15);
        for (i, (w, b)) in [(0.55, 0.05), (1. - 0.1 * 4. / 3., -0.1 * 4. / 3.), (0., 0.)]
            .into_iter()
            .enumerate()
        {
            close(e.layers[1].weights[[i, 0]], w, 1e-15);
            close(e.layers[1].biases[i], b, 1e-15);
        }
        assert!(
            e.layers
                .iter()
                .flat_map(|l| l.weight_velocity.iter())
                .all(|v| *v == 0.)
        );
    }

    // Independent central differences of sample/class MSE catch backward association and mean-N errors.
    #[test]
    fn square_sgd_matches_finite_difference_and_actual_partial_count() {
        let mut e = fixture(1, true, true);
        e.layers[1].weights[[2, 0]] = 0.2;
        let x = arr2(&[[0.7], [1.2]]);
        let objective = |engine: &Engine<f64>| {
            let mut total = 0.;
            for (row, label) in [(0.7, 0usize), (1.2, 1)] {
                let z = row * engine.layers[0].weights[[0, 0]] + engine.layers[0].biases[0];
                for c in 0..3 {
                    let q = z * z * engine.layers[1].weights[[c, 0]] + engine.layers[1].biases[c];
                    total += (q * q - f64::from(c == label)).powi(2);
                }
            }
            total / 6.
        };
        let original = e.layers.clone();
        let eps = 1e-6;
        let mut want = Vec::new();
        for l in 0..2 {
            for j in 0..e.layers[l].weights.len() {
                let c = j;
                e.layers[l].weights[[c, 0]] += eps;
                let plus = objective(&e);
                e.layers[l].weights[[c, 0]] -= 2. * eps;
                let minus = objective(&e);
                e.layers[l].weights[[c, 0]] += eps;
                want.push((plus - minus) / (2. * eps));
            }
            for j in 0..e.layers[l].biases.len() {
                e.layers[l].biases[j] += eps;
                let plus = objective(&e);
                e.layers[l].biases[j] -= 2. * eps;
                let minus = objective(&e);
                e.layers[l].biases[j] += eps;
                want.push((plus - minus) / (2. * eps));
            }
        }
        e.layers = original.clone();
        e.train(&x, &[&[0, 1]], &rate(0.01, 2. / 3.)).unwrap();
        let mut k = 0;
        for (old, new) in original.iter().zip(&e.layers) {
            for (a, b) in old
                .weights
                .iter()
                .zip(&new.weights)
                .chain(old.biases.iter().zip(&new.biases))
            {
                close((a - b) / 0.01, want[k], 1e-9);
                k += 1;
            }
        }
    }

    // Literal rational oracle: first below full reference .1, increase, drop, restart; V never rate-scaled.
    #[test]
    fn nag_four_rates_preserve_unscaled_velocity_and_active_or_frozen_bias() {
        let rates = [0.01, 0.1, 0.001, 0.05];
        let gs = [
            [0.5, -0.25, 0.2],
            [-0.2, 0.4, -0.1],
            [0.1, 0.2, 0.3],
            [-0.3, 0.1, -0.2],
        ];
        let ws = [
            [0.1905, -0.29525, 0.0962],
            [0.188, -0.351, 0.099],
            [0.1876075, -0.35152175, 0.0983652],
            [0.202945, -0.3755005, 0.1022992],
        ];
        let vs = [
            [0.5, -0.25, 0.2],
            [0.25, 0.175, 0.08],
            [0.325, 0.3575, 0.372],
            [-0.0075, 0.42175, 0.1348],
        ];
        for active in [false, true] {
            let mut l = PlainLayer::from_arrays(
                "p".into(),
                arr2(&[[0.2, -0.3]]),
                vec![if active { 0.1 } else { 0. }],
            );
            for i in 0..4 {
                apply_gradient(
                    &mut l,
                    &Gradient {
                        weights: arr2(&[[gs[i][0], gs[i][1]]]),
                        biases: vec![gs[i][2]],
                    },
                    active,
                    Some(0.9),
                    rates[i],
                )
                .unwrap();
                for j in 0..2 {
                    close(l.weights[[0, j]], ws[i][j], 2e-16);
                    close(l.weight_velocity[[0, j]], vs[i][j], 2e-16);
                }
                if active {
                    close(l.biases[0], ws[i][2], 2e-16);
                    close(l.bias_velocity[0], vs[i][2], 2e-16);
                } else {
                    assert_eq!(l.biases[0].to_bits(), 0);
                    assert_eq!(l.bias_velocity[0].to_bits(), 0);
                }
            }
        }
    }

    // A changed local target cannot change another stage, even via its updated representation.
    #[test]
    fn all_topologies_stop_gradients_and_use_pre_update_representations() {
        for (stages, square) in [(1, true), (1, false), (2, false), (3, false)] {
            let mut baseline = fixture(stages, square, square);
            let old_prediction = baseline.predict(&[1.]).unwrap();
            let mut altered = fixture(stages, square, square);
            for (i, spec) in altered.specs.iter().enumerate() {
                if matches!(spec.role, LayerRole::LocalHead { .. }) {
                    altered.layers[i].weights.fill(123.);
                }
            }
            assert_eq!(old_prediction, altered.predict(&[1.]).unwrap());
            let base_labels = vec![&[0u8][..]; stages];
            baseline
                .train(
                    &arr2(&[[1.]]),
                    &base_labels,
                    &rate(0.01, if square { 2. / 3. } else { 1. }),
                )
                .unwrap();
            for selected in 0..stages {
                let mut changed = fixture(stages, square, square);
                let mut labels = base_labels.clone();
                labels[selected] = &[1];
                changed
                    .train(
                        &arr2(&[[1.]]),
                        &labels,
                        &rate(0.01, if square { 2. / 3. } else { 1. }),
                    )
                    .unwrap();
                for other in 0..stages {
                    let a = &baseline.layers[2 * other];
                    let b = &changed.layers[2 * other];
                    if other == selected {
                        assert_ne!(a.weights, b.weights);
                    } else {
                        assert_eq!(a.weights, b.weights);
                        assert_eq!(
                            baseline.layers[2 * other + 1].weights,
                            changed.layers[2 * other + 1].weights
                        );
                    }
                }
            }
            if stages > 1 {
                // Stage1 receives .75 from OLD .5, even after stage0 mutates.
                // Its z=.375; h=.515625; d=[-.7421875,.515625,0].
                // d_hidden=(.5*d0+d1)*(2*.375+1)=.2529296875.
                let expected = 0.5 - 0.01 * 1.9 * (0.2529296875 * 0.75);
                close(baseline.layers[2].weights[[0, 0]], expected, 1e-15);
            }
        }
    }

    // Logical padded channel is trained, although real-class accuracy excludes it.
    #[test]
    fn padding_is_trained_and_argmax_follows_d1_last_total_order_maximum() {
        let mut e = fixture(1, false, false);
        e.classes = 2;
        e.layers[1].weights[[2, 0]] = 2.;
        e.train(&arr2(&[[1.]]), &[&[0]], &rate(0.01, 1.)).unwrap();
        assert_ne!(e.layers[1].weights[[2, 0]], 2.);
        assert_eq!(class_index(&[1f64, 2., 100.], 2).unwrap(), 1);
        assert_eq!(class_index(&[2f64, 2., 100.], 2).unwrap(), 1);
        assert_eq!(class_index(&[0f64, -0., 100.], 2).unwrap(), 0);
        assert!(class_index(&[1f64, 2., f64::NAN], 2).is_err());
        assert!(class_index(&[1f64, f64::INFINITY, 0.], 2).is_err());
    }

    // Three ascending products cancel to zero in f32 but one in f64; widening then cast fails.
    #[test]
    fn actual_f32_forward_reduction_and_nag_arithmetic_are_not_widened() {
        let spec = Layer {
            name: "cancellation".into(),
            input: 3,
            output: 1,
            activation: Activation::Identity,
            bias: false,
            role: LayerRole::Hidden,
        };
        let layer = PlainLayer::from_arrays(
            "cancellation".into(),
            arr2(&[[16777216f32, 1., -16777216.]]),
            vec![0.],
        );
        assert_eq!(
            forward(&layer, &spec, &arr2(&[[1f32, 1., 1.]])).unwrap().a[[0, 0]],
            0.
        );
        assert_eq!((16777216f64 + 1. - 16777216.) as f32, 1.);
        let mut l = PlainLayer::from_arrays("nag".into(), arr2(&[[3f32]]), vec![0.]);
        l.weight_velocity[[0, 0]] = 1234567.;
        apply_gradient(
            &mut l,
            &Gradient {
                weights: arr2(&[[-0.1f32]]),
                biases: vec![0.],
            },
            false,
            Some(0.9f32),
            0.03f32,
        )
        .unwrap();
        assert_eq!(l.weight_velocity[[0, 0]].to_bits(), 0x4987a231);
        assert_eq!(l.weights[[0, 0]].to_bits(), 0xc6ea59ef);
        let mu = 0.9f32 as f64;
        let g = -0.1f32 as f64;
        let r = 0.03f32 as f64;
        let wide = (3. - r * (g + mu * (mu * 1234567. + g))) as f32;
        assert_ne!(l.weights[[0, 0]].to_bits(), wide.to_bits());
        let mut cancel = PlainLayer::from_arrays("v".into(), arr2(&[[3f32]]), vec![0.]);
        cancel.weight_velocity[[0, 0]] = 16777215.;
        apply_gradient(
            &mut cancel,
            &Gradient {
                weights: arr2(&[[-15099493f32]]),
                biases: vec![0.],
            },
            false,
            Some(0.9f32),
            0.03f32,
        )
        .unwrap();
        assert_eq!(cancel.weight_velocity[[0, 0]], 0.);
        assert_ne!(((0.9f32 as f64) * 16777215. - 15099493.) as f32, 0.);
    }

    // Mean gradients must reduce in T too, independently of forward and optimizer tests.
    #[test]
    fn actual_f32_gradient_cancellation_and_valid_mean_are_not_widened() {
        let g = gradient(
            &arr2(&[[16777216f32], [1.], [-16777216.]]),
            &arr2(&[[1f32], [1.], [1.]]),
            true,
        )
        .unwrap();
        assert_eq!(g.weights[[0, 0]], 0.);
        assert_eq!(g.biases[0], 0.);
        assert_ne!(((16777216f64 + 1. - 16777216.) / 3.) as f32, 0.);
        let partial = gradient(&arr2(&[[1f32], [3.]]), &arr2(&[[2f32], [4.]]), true).unwrap();
        assert_eq!(partial.weights[[0, 0]], 7.);
        assert_eq!(partial.biases[0], 2.);
    }

    fn canonical_iris() -> MaterializedProfile {
        let registry =
            StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        materialize(&registry.resolve("iris__mlp2020", 0).unwrap()).unwrap()
    }

    // Test-only synthetic positioning at the final index is NOT a trained final model.
    // Catches wrong plan endpoint/exhaustion and accidental advancement on exhausted calls.
    #[test]
    fn bound_final_planned_update_then_exhaustion_does_not_replay_or_poison() {
        let profile = canonical_iris();
        let mut state = PlainState::<f64>::new(&profile).unwrap();
        let last = profile.plan().updates.last().unwrap();
        state.completed = last.index;
        state.processed = last.processed_start;
        let receipt = step_plain(&mut state).unwrap();
        assert_eq!(receipt.update_index, last.index);
        assert_eq!(receipt.valid_samples, last.valid_samples);
        assert_eq!(state.completed_updates(), profile.plan().updates.len());
        assert_eq!(state.processed_samples(), profile.plan().processed_samples);
        let before = state.layers()[0].weights().clone();
        assert_eq!(step_plain(&mut state), Err(PlainError::Exhausted));
        assert_eq!(step_plain(&mut state), Err(PlainError::Exhausted));
        assert_eq!(state.layers()[0].weights(), &before);
        assert!(!state.is_poisoned());
        assert!(evaluate_plain(&state).is_ok());
    }

    // Test-only parameter corruption cannot be supplied through the production API.
    #[test]
    fn failed_step_poison_is_terminal_and_success_counters_do_not_advance() {
        let profile = canonical_iris();
        let mut state = PlainState::<f32>::new(&profile).unwrap();
        step_plain(&mut state).unwrap();
        state.engine.layers[0].weights[[0, 0]] = f32::MAX;
        assert!(matches!(
            step_plain(&mut state),
            Err(PlainError::InvalidArithmetic(_))
        ));
        assert_eq!(state.completed_updates(), 1);
        assert_eq!(state.processed_samples(), 20);
        assert!(state.is_poisoned());
        assert_eq!(step_plain(&mut state), Err(PlainError::Poisoned));
        assert_eq!(evaluate_plain(&state), Err(PlainError::Poisoned));
        assert_eq!(predict_plain(&state, &[0.; 4]), Err(PlainError::Poisoned));
    }

    #[test]
    fn invalid_prediction_is_read_only_and_does_not_poison_training() {
        let profile = canonical_iris();
        let mut state = PlainState::<f64>::new(&profile).unwrap();
        assert!(predict_plain(&state, &[0.]).is_err());
        assert!(predict_plain(&state, &[f64::NAN; 4]).is_err());
        assert!(!state.is_poisoned());
        assert_eq!(state.completed_updates(), 0);
        step_plain(&mut state).unwrap();
    }

    // Nonfinite arithmetic and invalid batch/rates are explicit failures, never clamped.
    #[test]
    fn rejects_nonfinite_forward_gradient_and_invalid_arithmetic() {
        let mut e = fixture(1, true, true);
        assert!(
            e.train(&arr2(&[[f64::MAX]]), &[&[0]], &rate(0.1, 2. / 3.))
                .is_err()
        );
        assert!(e.predict(&[f64::NAN]).is_err());
        assert!(
            e.train(&Array2::zeros((0, 1)), &[&[]], &rate(0.1, 1.))
                .is_err()
        );
        assert!(e.train(&arr2(&[[1.]]), &[&[0]], &rate(0., 1.)).is_err());
        assert!(e.train(&arr2(&[[1.]]), &[&[3]], &rate(0.1, 1.)).is_err());
    }
}
