//! Pure geometric admission; no keys, ciphertexts or training-policy admission.
use super::invalid;
use crate::layout_reboot::RebootLayout;
use ckks_train_core::error::Result;
use ndarray::Array2;
use std::sync::Arc;

/// Matrix storage orientation (the logical source array is always output-by-input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// RE edge: expanded input, repeated output; slot `(i,j)` is source `[j,i]`.
    Row,
    /// CE edge: repeated input, expanded output; slot `(i,j)` is source `[i,j]`.
    Column,
}

/// Replication direction of a logical vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorRepr {
    /// Slot `(i,j)` holds logical coordinate `i`.
    Expanded,
    /// Slot `(i,j)` holds logical coordinate `j`.
    Repeated,
}

/// Checked vector shape on one lattice. Logical width includes trained padded-class channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSpec {
    layout: RebootLayout,
    repr: VectorRepr,
    logical_len: usize,
}
impl VectorSpec {
    /// Checks a nonempty vector against its actual row or column axis.
    pub fn new(layout: RebootLayout, repr: VectorRepr, logical_len: usize) -> Result<Self> {
        let bound = match repr {
            VectorRepr::Expanded => layout.r(),
            VectorRepr::Repeated => layout.c(),
        };
        if logical_len == 0 || logical_len > bound {
            return invalid("packed vector exceeds its representation axis");
        }
        Ok(Self {
            layout,
            repr,
            logical_len,
        })
    }
    /// Common lattice.
    pub fn layout(&self) -> RebootLayout {
        self.layout
    }
    /// Replication direction.
    pub fn repr(&self) -> VectorRepr {
        self.repr
    }
    /// Model width, not the real-class count.
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }
    /// Zeroes structural padding only.
    pub fn mask(&self) -> Vec<f64> {
        (0..self.layout.slots())
            .map(|s| {
                let k = match self.repr {
                    VectorRepr::Expanded => s / self.layout.c(),
                    VectorRepr::Repeated => s % self.layout.c(),
                };
                if k < self.logical_len { 1.0 } else { 0.0 }
            })
            .collect()
    }
    /// Packs exactly the model-width finite values, with zero structural padding.
    pub fn pack(&self, values: &[f64]) -> Result<Vec<f64>> {
        if values.len() != self.logical_len || values.iter().any(|v| !v.is_finite()) {
            return invalid("packed vector values have wrong length or nonfinite entries");
        }
        match self.repr {
            VectorRepr::Expanded => self.layout.pack_expanded(values),
            VectorRepr::Repeated => self.layout.pack_repeated(values),
        }
    }
}

/// One two-layer stage: hidden backbone edge plus classifier edge. All but the last stage
/// hand their hidden activation to the next stage; the last is the output stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StageShape {
    hidden: usize,
    output: usize,
    real_classes: usize,
}
impl StageShape {
    /// Checks nonzero widths and `real_classes <= output` (padded outputs remain trained).
    pub fn new(hidden: usize, output: usize, real_classes: usize) -> Result<Self> {
        if hidden == 0 || output == 0 || real_classes == 0 || real_classes > output {
            return invalid("invalid packed stage widths or real classes");
        }
        Ok(Self {
            hidden,
            output,
            real_classes,
        })
    }
    /// Hidden backbone width.
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    /// Classifier model width including trained padded channels.
    pub fn output(&self) -> usize {
        self.output
    }
    /// Real classes used for accuracy, not structural masking.
    pub fn real_classes(&self) -> usize {
        self.real_classes
    }
}

/// A checked layer tied to a particular graph definition and ordinal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerGeometry {
    graph: Arc<[StageShape]>,
    graph_input: usize,
    ordinal: usize,
    input: VectorSpec,
    output: VectorSpec,
    orientation: Orientation,
}
impl LayerGeometry {
    /// Input vector representation and width.
    pub fn input(&self) -> VectorSpec {
        self.input
    }
    /// Output vector representation and model width.
    pub fn output(&self) -> VectorSpec {
        self.output
    }
    /// Matrix orientation.
    pub fn orientation(&self) -> Orientation {
        self.orientation
    }
    /// Layer ordinal: backbone then classifier for each stage.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
    /// Common lattice.
    pub fn layout(&self) -> RebootLayout {
        self.input.layout
    }
    /// Matrix structural support, independent from real-class count.
    pub fn matrix_mask(&self) -> Vec<f64> {
        let l = self.layout();
        (0..l.slots())
            .map(|s| {
                let (i, o) = match self.orientation {
                    Orientation::Row => (s / l.c(), s % l.c()),
                    Orientation::Column => (s % l.c(), s / l.c()),
                };
                if i < self.input.logical_len && o < self.output.logical_len {
                    1.0
                } else {
                    0.0
                }
            })
            .collect()
    }
    /// Maps the original logical `[output,input]` array into row or column storage.
    pub fn pack_weights_out_by_in(&self, w: &Array2<f64>) -> Result<Vec<f64>> {
        if w.dim() != (self.output.logical_len, self.input.logical_len)
            || w.iter().any(|v| !v.is_finite())
        {
            return invalid("packed weights require finite logical output-by-input array");
        }
        let l = self.layout();
        let mut out = vec![0.0; l.slots()];
        for ((o, i), &value) in w.indexed_iter() {
            let s = match self.orientation {
                Orientation::Row => i * l.c() + o,
                Orientation::Column => o * l.c() + i,
            };
            out[s] = value;
        }
        Ok(out)
    }
}

/// Geometry-only plan for a complete local-stage graph, including its final output stack.
/// Does not admit level budgets, memory, optimizer semantics, batch tails or refresh modes.
#[derive(Clone, Debug)]
pub struct PackedGeometryPlan {
    layout: RebootLayout,
    stages: Arc<[StageShape]>,
    layers: Vec<LayerGeometry>,
}
impl PackedGeometryPlan {
    /// Chooses the smallest feasible power-of-two column bound, checking every edge.
    /// The first backbone edge is RE; subsequent stage backbone edges alternate. Each
    /// classifier is opposite its backbone edge. Capacity gaps are errors before HE work.
    pub fn new(input: usize, stages: &[StageShape], slots: usize) -> Result<Self> {
        if input == 0 || stages.is_empty() || !slots.is_power_of_two() || slots > (1usize << 30) {
            return invalid("invalid packed graph input, stages or supported slot count");
        }
        let mut rows = input;
        let mut cols = 1;
        let mut previous = input;
        for (s, stage) in stages.iter().enumerate() {
            if s % 2 == 0 {
                rows = rows.max(previous).max(stage.output);
                cols = cols.max(stage.hidden);
            } else {
                cols = cols.max(previous).max(stage.output);
                rows = rows.max(stage.hidden);
            }
            previous = stage.hidden;
        }
        let c = cols.checked_next_power_of_two().ok_or_else(|| {
            ckks_train_core::error::HekitError::InvalidParams("packed column bound overflow".into())
        })?;
        if c > slots || rows > slots / c {
            return invalid(
                "packed graph capacity gap: no common RE/CE lattice fits every backbone and head axis",
            );
        }
        // Larger columns can only decrease rows: if the minimal c fails, no factorization fits.
        let layout = RebootLayout::new(slots / c, c, slots)?;
        let graph: Arc<[StageShape]> = stages.into();
        let mut layers = Vec::with_capacity(stages.len() * 2);
        let mut previous = input;
        for (s, stage) in stages.iter().enumerate() {
            let (a, b, orientation) = if s % 2 == 0 {
                (VectorRepr::Expanded, VectorRepr::Repeated, Orientation::Row)
            } else {
                (
                    VectorRepr::Repeated,
                    VectorRepr::Expanded,
                    Orientation::Column,
                )
            };
            let x = VectorSpec::new(layout, a, previous)?;
            let h = VectorSpec::new(layout, b, stage.hidden)?;
            let y = VectorSpec::new(layout, a, stage.output)?;
            layers.push(LayerGeometry {
                graph: graph.clone(),
                graph_input: input,
                ordinal: 2 * s,
                input: x,
                output: h,
                orientation,
            });
            layers.push(LayerGeometry {
                graph: graph.clone(),
                graph_input: input,
                ordinal: 2 * s + 1,
                input: h,
                output: y,
                orientation: if orientation == Orientation::Row {
                    Orientation::Column
                } else {
                    Orientation::Row
                },
            });
            previous = stage.hidden;
        }
        Ok(Self {
            layout,
            stages: graph,
            layers,
        })
    }
    /// Shared lattice for all layers and handoffs.
    pub fn layout(&self) -> RebootLayout {
        self.layout
    }
    /// Complete ordered two-edge stages, last one the output stack.
    pub fn stages(&self) -> &[StageShape] {
        &self.stages
    }
    /// Backbone then classifier of each stage, including the output stack.
    pub fn layers(&self) -> &[LayerGeometry] {
        &self.layers
    }
    /// One input ciphertext per valid sample; no sample padding or memory admission.
    pub fn sample_ciphertexts(&self, valid_samples: usize) -> Result<usize> {
        if valid_samples == 0 {
            invalid("empty packed sample batch")
        } else {
            Ok(valid_samples)
        }
    }
}
