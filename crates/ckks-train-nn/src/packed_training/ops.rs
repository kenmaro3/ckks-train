//! Exact-Delta masked affine and shared encrypted outer-sum branches.
use super::types::check_ct;
use super::{LayerGeometry, Orientation, PackedGradient, PackedVector, PackedWeights, invalid};
use crate::layout_reboot::GradRelin;
use ckks_train_ckks::{DiagonalEntry, Encoder, HeOps, Plaintext};
use ckks_train_core::error::{HekitError, Result};
use std::sync::Arc;

type ProductHistory = (usize, f64);
type TracedVector<C> = (PackedVector<C>, ProductHistory);
type TracedGradients<C> = (Vec<PackedGradient<C>>, ProductHistory);

/// References to one valid sample's input and output-gradient vectors, in that order.
pub type PackedSamplePair<'a, C> = (&'a PackedVector<C>, &'a PackedVector<C>);

fn context<E: HeOps>(ev: &E, enc: &Encoder) -> Result<()> {
    if !Arc::ptr_eq(ev.params(), enc.params()) {
        return invalid("packed encoder belongs to another parameter context");
    }
    if !ev.has_relin_key() {
        return invalid("packed product missing relinearization key");
    }
    Ok(())
}

/// Predicts the actual product/rescale scale without evaluating or relabelling a ciphertext.
fn product_history<E: HeOps>(ev: &E, a: &E::Ct, b: &E::Ct) -> Result<(usize, f64)> {
    let level = a.level().min(b.level());
    if level < 2 {
        return Err(HekitError::NoLevelsLeft);
    }
    let product = a.scale() * b.scale();
    let natural = product / ev.params().q()[level] as f64;
    if !product.is_finite() || !natural.is_finite() || natural <= 0.0 {
        return Err(HekitError::ScaleMismatch);
    }
    Ok((level - 1, natural))
}

/// Encode before counted work, including range/precision checks at the predicted natural scale.
pub(super) fn mask_plain<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    mask: &[f64],
    level: usize,
    natural: f64,
) -> Result<Plaintext> {
    let ps = ev.exact_product_scale(level, ev.params().scale())? / natural;
    if !ps.is_finite() || ps < ckks_train_ckks::evaluator::MIN_PLAIN_SCALE {
        return invalid(
            "packed normalization plaintext scale is nonfinite or below precision floor",
        );
    }
    enc.encode(mask, ps, level)
}

/// Encrypted affine without bias, normalized to exact Delta and projected to model output
/// support. Both orientations consume two levels, one tensor/relin and one ct-pt mask.
/// RE uses `log2(rows)` rotations, CE uses `2*log2(cols)`. Positive natural input scales
/// are accepted (for later backward adapters); weights must be exact Delta.
/// All structural, key, scale and encoding validation precedes counted operations.
/// Device failures after validation do not roll back previously completed operations.
pub fn affine<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    w: &PackedWeights<E::Ct>,
    x: &PackedVector<E::Ct>,
) -> Result<PackedVector<E::Ct>> {
    directed_affine(ev, enc, w, x, false).map(|(out, _)| out)
}

// Reverse direction uses the SAME matrix ciphertext, never a transpose or diagonal cast.
pub(super) fn directed_affine<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    w: &PackedWeights<E::Ct>,
    x: &PackedVector<E::Ct>,
    reverse: bool,
) -> Result<TracedVector<E::Ct>> {
    context(ev, enc)?;
    let layer = w.layer();
    let (input, output, orientation) = if reverse {
        (
            layer.output(),
            layer.input(),
            match layer.orientation() {
                Orientation::Row => Orientation::Column,
                Orientation::Column => Orientation::Row,
            },
        )
    } else {
        (layer.input(), layer.output(), layer.orientation())
    };
    let layout = layer.layout();
    check_ct(ev, w.ciphertext(), layout.slots())?;
    check_ct(ev, x.ciphertext(), layout.slots())?;
    if x.spec() != input {
        return invalid("packed affine input shape mismatch");
    }
    if w.ciphertext().scale() != ev.params().scale() {
        return Err(HekitError::ScaleMismatch);
    }
    let steps = match orientation {
        Orientation::Row => layout.row_sum_steps(),
        Orientation::Column => layout.expand_steps(),
    };
    if steps.iter().any(|&s| !ev.has_galois_key(s)) {
        return invalid("packed affine missing galois key");
    }
    let (level, natural) = product_history(ev, x.ciphertext(), w.ciphertext())?;
    let mask = match orientation {
        Orientation::Row => output.mask(),
        Orientation::Column => (0..layout.slots())
            .map(|s| {
                if s % layout.c() == 0 && s / layout.c() < output.logical_len() {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    };
    let pt = mask_plain(ev, enc, &mask, level, natural)?;
    let (out, history) = match orientation {
        Orientation::Row => {
            let pt = ev.upload_pt_owned(pt)?;
            let product = layout.vecmat(ev, x.ciphertext(), w.ciphertext())?;
            let history = (product.level(), product.scale());
            (
                ev.rescale(&ev.mul_plain_to_scale(&product, &pt, ev.params().scale())?)?,
                history,
            )
        }
        Orientation::Column => {
            // Validate encoding before tensor work. The helper encodes/uploads the same mask
            // at runtime; this redundant host encode is included in primitive wall time.
            let x = ev.mod_switch_to_level(x.ciphertext(), level + 1)?;
            let w = ev.mod_switch_to_level(w.ciphertext(), level + 1)?;
            let product = ev.rescale(&ev.mul(&x, &w)?)?;
            let history = (product.level(), product.scale());
            let out = layout.expand_valid_to_scale(
                ev,
                enc,
                &product,
                output.logical_len(),
                1.0,
                ev.params().scale(),
            )?;
            (out, history)
        }
    };
    Ok((PackedVector::new(ev, output, out)?, history))
}

/// Forms the encrypted sum over precisely the supplied valid sample pairs, then emits
/// `constant * sum` for each constant with the layer's active matrix mask at exact Delta.
/// Constants may include zero or negative finite factors; no averaging is implicit.
/// All pairs must have identical aligned level and actual product scale. One shared
/// product sum/relin/rescale feeds every branch; no plaintext gradient is constructed.
/// All shapes, contexts, levels, scales, relin key and branch encodings are validated first.
/// Upload/allocation failures can occur later; counted work is not transactionally rolled back.
pub fn outer_sum_scaled_masks<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    layer: &LayerGeometry,
    pairs: &[PackedSamplePair<'_, E::Ct>],
    constants: &[f64],
    relin: GradRelin,
) -> Result<Vec<PackedGradient<E::Ct>>> {
    outer_traced(ev, enc, layer, pairs, constants, relin).map(|(out, _)| out)
}

pub(super) fn outer_traced<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    layer: &LayerGeometry,
    pairs: &[PackedSamplePair<'_, E::Ct>],
    constants: &[f64],
    relin: GradRelin,
) -> Result<TracedGradients<E::Ct>> {
    context(ev, enc)?;
    if pairs.is_empty() || constants.is_empty() || constants.iter().any(|c| !c.is_finite()) {
        return invalid("packed outer sum needs valid pairs and finite branch constants");
    }
    let layout = layer.layout();
    let mut expected = None;
    let mut refs = Vec::with_capacity(pairs.len());
    for &(x, y) in pairs {
        check_ct(ev, x.ciphertext(), layout.slots())?;
        check_ct(ev, y.ciphertext(), layout.slots())?;
        if x.spec() != layer.input() || y.spec() != layer.output() {
            return invalid("packed outer pair shape mismatch");
        }
        let history = product_history(ev, x.ciphertext(), y.ciphertext())?;
        // Check exact pre-rescale scales too: distinct products must not merge by rounding.
        let key = (history.0, x.ciphertext().scale() * y.ciphertext().scale());
        if let Some((level, scale)) = expected {
            if key.0 != level {
                return Err(HekitError::LevelMismatch);
            }
            if key.1 != scale {
                return Err(HekitError::ScaleMismatch);
            }
        }
        expected = Some(key);
        refs.push(match layer.orientation() {
            Orientation::Row => (x.ciphertext(), y.ciphertext()),
            Orientation::Column => (y.ciphertext(), x.ciphertext()),
        });
    }
    let (level, product) = expected.expect("nonempty validated pairs");
    let natural = product / ev.params().q()[level + 1] as f64;
    let mask = layer.matrix_mask();
    let host_pts: Vec<_> = constants
        .iter()
        .map(|&c| {
            mask_plain(
                ev,
                enc,
                &mask.iter().map(|&m| m * c).collect::<Vec<_>>(),
                level,
                natural,
            )
        })
        .collect::<Result<_>>()?;
    // Upload every branch before any counted product; no partially validated branch list.
    let pts: Vec<_> = host_pts
        .into_iter()
        .map(|pt| ev.upload_pt_owned(pt))
        .collect::<Result<_>>()?;
    let sum = layout.outer_sum(ev, &refs, relin)?;
    let history = (sum.level(), sum.scale());
    let gradients = pts
        .iter()
        .zip(constants)
        .map(|(pt, &factor)| {
            let ct = ev.rescale(&ev.mul_plain_to_scale(&sum, pt, ev.params().scale())?)?;
            Ok(PackedGradient {
                ct,
                layer: layer.clone(),
                valid_samples: pairs.len(),
                factor,
            })
        })
        .collect::<Result<_>>()?;
    Ok((gradients, history))
}
