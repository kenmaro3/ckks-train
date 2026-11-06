//! Session-owned prefix/state encodings; no decoder for canonical identities.
use super::session::{Result, SessionError};
use crate::{
    data::sha256_hex,
    study_inputs::MaterializedProfile,
    study_plain::{PlainScalar, PlainState},
    study_profile::{BatchPlan, Optimizer},
};
use ndarray::{ArrayView2, Axis};
pub(super) trait Scalar: PlainScalar {
    const WIDTH: u8;
    fn bits(self, out: &mut Vec<u8>);
}
impl Scalar for f32 {
    const WIDTH: u8 = 32;
    fn bits(self, out: &mut Vec<u8>) {
        out.extend(self.to_bits().to_le_bytes());
    }
}
impl Scalar for f64 {
    const WIDTH: u8 = 64;
    fn bits(self, out: &mut Vec<u8>) {
        out.extend(self.to_bits().to_le_bytes());
    }
}
struct Stream(Vec<u8>);
impl Stream {
    fn new(domain: &str) -> Result<Self> {
        let mut s = Self(Vec::new());
        s.text(super::ENCODING_VERSION)?;
        s.text(domain)?;
        Ok(s)
    }
    fn n(&mut self, n: usize) -> Result<()> {
        self.0.extend(
            u64::try_from(n)
                .map_err(|_| SessionError("count_overflow"))?
                .to_le_bytes(),
        );
        Ok(())
    }
    fn text(&mut self, t: &str) -> Result<()> {
        self.n(t.len())?;
        self.0
            .len()
            .checked_add(t.len())
            .ok_or(SessionError("length_overflow"))?;
        self.0.extend(t.as_bytes());
        Ok(())
    }
}
pub(super) fn consumed(plan: &BatchPlan, full: &str, n: usize) -> Result<String> {
    if n > plan.updates.len() {
        return Err(SessionError("prefix_exhausted"));
    }
    if n == plan.updates.len() {
        return Ok(full.into());
    }
    Ok(sha256_hex(&prefix_bytes(plan, full, n)?))
}
fn prefix_bytes(plan: &BatchPlan, full: &str, n: usize) -> Result<Vec<u8>> {
    if n >= plan.updates.len() {
        return Err(SessionError("strict_prefix_required"));
    }
    let mut s = Stream::new("hekit-study-consumed-prefix-v1")?;
    s.text(full)?;
    s.n(plan.updates.len())?;
    s.n(n)?;
    s.n(plan.updates[..n].iter().try_fold(0usize, |a, u| {
        a.checked_add(u.valid_samples)
            .ok_or(SessionError("sample_overflow"))
    })?)?;
    s.n(plan
        .checkpoints
        .iter()
        .filter(|c| c.completed_updates <= n)
        .count())?;
    Ok(s.0)
}
struct LayerView<'a, T> {
    name: &'a str,
    tensors: [ArrayView2<'a, T>; 4],
}
fn state_bytes<T: Scalar>(
    materialized: &str,
    receipt: &str,
    n: usize,
    samples: usize,
    layers: &[LayerView<'_, T>],
) -> Result<Vec<u8>> {
    let mut s = Stream::new("hekit-study-plain-state-v1")?;
    s.0.push(T::WIDTH);
    s.text(materialized)?;
    s.text(receipt)?;
    s.n(n)?;
    s.n(samples)?;
    s.n(layers.len())?;
    if layers.is_empty() {
        return Err(SessionError("state_layers"));
    }
    for (i, l) in layers.iter().enumerate() {
        if l.name.is_empty() || layers[..i].iter().any(|p| p.name == l.name) {
            return Err(SessionError("state_layer_name"));
        }
        s.text(l.name)?;
        s.n(4)?;
        if l.tensors[0].dim() != l.tensors[2].dim()
            || l.tensors[1].dim() != l.tensors[3].dim()
            || l.tensors[1].nrows() != 1
            || l.tensors[1].ncols() != l.tensors[0].nrows()
        {
            return Err(SessionError("state_shape"));
        }
        for (role, t) in l.tensors.iter().enumerate() {
            if t.is_empty() || t.nrows().checked_mul(t.ncols()) != Some(t.len()) {
                return Err(SessionError("state_shape"));
            }
            s.0.push(role as u8);
            s.n(t.nrows())?;
            s.n(t.ncols())?;
            s.n(t.len())?;
            s.0.len()
                .checked_add(
                    t.len()
                        .checked_mul(usize::from(T::WIDTH) / 8)
                        .ok_or(SessionError("length_overflow"))?,
                )
                .ok_or(SessionError("length_overflow"))?;
            // Explicit logical row/column order, independent of ndarray strides.
            for row in t.rows() {
                for &x in row {
                    if !x.finite() {
                        return Err(SessionError("state_nonfinite"));
                    }
                    x.bits(&mut s.0);
                }
            }
        }
    }
    Ok(s.0)
}
pub(super) fn state_digest<T: Scalar>(
    p: &MaterializedProfile,
    state: &PlainState<'_, T>,
    receipt: &str,
) -> Result<String> {
    if state.layers().len() != p.metadata().layers.len() || state.is_poisoned() {
        return Err(SessionError("state_invalid"));
    }
    let mut views = Vec::new();
    for (layer, spec) in state.layers().iter().zip(&p.metadata().layers) {
        if layer.name() != spec.name
            || layer.weights().dim() != (spec.output, spec.input)
            || layer.biases().len() != spec.output
        {
            return Err(SessionError("state_graph"));
        }
        if !spec.bias
            && layer
                .biases()
                .iter()
                .chain(layer.bias_velocity())
                .any(|&x| x != T::zero())
        {
            return Err(SessionError("disabled_bias_changed"));
        }
        if p.metadata().optimizer == Optimizer::Sgd
            && layer
                .weight_velocity()
                .iter()
                .chain(layer.bias_velocity())
                .any(|&x| x != T::zero())
        {
            return Err(SessionError("sgd_velocity_changed"));
        }
        views.push(LayerView {
            name: layer.name(),
            tensors: [
                layer.weights().view(),
                ndarray::ArrayView1::from(layer.biases()).insert_axis(Axis(0)),
                layer.weight_velocity().view(),
                ndarray::ArrayView1::from(layer.bias_velocity()).insert_axis(Axis(0)),
            ],
        });
    }
    Ok(sha256_hex(&state_bytes(
        p.identity().digest(),
        receipt,
        state.completed_updates(),
        state.processed_samples(),
        &views,
    )?))
}
#[cfg(test)]
#[path = "digests_tests.rs"]
mod tests;
