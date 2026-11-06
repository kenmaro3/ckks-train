//! Pure exact-target scale metadata, requiring no evaluator, keys or ciphertexts.
//!
//! These receipts validate scale/level arithmetic only. They do not prove basis/context
//! identity, key availability, encoded content range or decryption precision. See
//! `docs/ckks-metadata-design.md` for the supported domain and legacy-runtime boundary.

use ckks_train_core::error::{HekitError, Result};

use crate::CkksParams;
use crate::evaluator::{check_plain_scale, exact_product_scale, scalar_product_parts};

/// Validated ciphertext level and exact positive finite F64 scale bits.
///
/// This is not a parameter identity or a ciphertext. Level0 is valid metadata, but cannot
/// be used for a scalar multiplication followed by rescale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CtMeta {
    level: usize,
    scale_bits: u64,
}

impl CtMeta {
    /// Validates the level against `params` and requires a positive finite scale.
    /// Returns `InvalidParams` on failure. Does not validate any runtime basis or contents.
    pub fn new(params: &CkksParams, level: usize, scale: f64) -> Result<Self> {
        if level > params.max_level() {
            return Err(HekitError::InvalidParams(format!(
                "level {level} exceeds max_level {}",
                params.max_level()
            )));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(HekitError::InvalidParams(
                "source scale must be positive and finite".into(),
            ));
        }
        Ok(Self {
            level,
            scale_bits: scale.to_bits(),
        })
    }

    /// Data-prime level (`level + 1` active q limbs).
    pub fn level(&self) -> usize {
        self.level
    }

    /// Exact stored scale as F64.
    pub fn scale(&self) -> f64 {
        f64::from_bits(self.scale_bits)
    }

    /// Exact F64 bits, without logarithmic approximation.
    pub fn scale_bits(&self) -> u64 {
        self.scale_bits
    }
}

/// Checked exact-target scalar label and actual rounded coefficient metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarProductMeta {
    pre_rescale: CtMeta,
    post_rescale: CtMeta,
    plain_scale: f64,
    rounded_integer: f64,
}

impl ScalarProductMeta {
    /// Product label/level before the caller's rescale; not an executed product.
    pub fn pre_rescale(&self) -> CtMeta {
        self.pre_rescale
    }

    /// Exact target label at source level minus1; not an executed rescale.
    pub fn post_rescale(&self) -> CtMeta {
        self.post_rescale
    }

    /// Plaintext scale `label / source_scale` used by the scalar encoding.
    pub fn plain_scale(&self) -> f64 {
        self.plain_scale
    }

    /// Actual `round(value * plain_scale)`, retaining signed zero as F64.
    pub fn rounded_integer(&self) -> f64 {
        self.rounded_integer
    }

    /// Actual represented coefficient in F64 (`rounded_integer / plain_scale`).
    /// This is not the ideal requested coefficient or a decryption-error bound.
    pub fn represented_value(&self) -> f64 {
        self.rounded_integer / self.plain_scale
    }
}

/// Exact pre-rescale label from the existing nearest-first ±8-ULP search.
///
/// Uses actual `params.q()[level] as f64` and division equality without tolerance.
/// Errors and their order match [`crate::Evaluator::exact_product_scale`]: invalid level,
/// invalid target, level0 (`NoLevelsLeft`), product overflow, unreachable (`ScaleMismatch`).
pub fn exact_product_scale_at(params: &CkksParams, level: usize, target: f64) -> Result<f64> {
    exact_product_scale(params, level, target)
}

/// Checked plaintext scale for exact-target multiplication, using actual source scale.
///
/// Requires positive finite source and target scales, an available rescale level, and the
/// runtime minimum plaintext scale `2^20`. Nonfinite intermediate scales are rejected,
/// unlike the legacy runtime helper on tiny-source overflow. No basis/contents are checked.
pub fn target_plain_scale_at(
    params: &CkksParams,
    level: usize,
    source_scale: f64,
    target: f64,
) -> Result<f64> {
    CtMeta::new(params, level, source_scale)?;
    let sp = exact_product_scale_at(params, level, target)? / source_scale;
    if !sp.is_finite() {
        return Err(HekitError::InvalidParams(
            "metadata plaintext scale must be finite".into(),
        ));
    }
    check_plain_scale(sp)?;
    Ok(sp)
}

/// Pure scalar encoding and exact pre/post-rescale metadata, without operations/counts.
///
/// Revalidates `source` against `params`. Preserves runtime rounding order, minimum `2^20`
/// plaintext scale and strict `abs(k) < 2^62` bound. Zero, negative and rounded-zero scalar
/// values are supported; invalid scalars/metadata return existing typed errors. Nonfinite
/// intermediate scales/integers are rejected even where legacy runtime accepts them.
pub fn scalar_metadata_at(
    params: &CkksParams,
    source: CtMeta,
    value: f64,
    target: f64,
) -> Result<ScalarProductMeta> {
    CtMeta::new(params, source.level(), source.scale())?;
    let (label, plain_scale, rounded_integer) =
        scalar_product_parts(params, source.level(), source.scale(), value, target)?;
    if !(plain_scale.is_finite() && rounded_integer.is_finite()) {
        return Err(HekitError::InvalidParams(
            "metadata plaintext scale and rounded integer must be finite".into(),
        ));
    }
    Ok(ScalarProductMeta {
        pre_rescale: CtMeta::new(params, source.level(), label)?,
        post_rescale: CtMeta::new(params, source.level() - 1, target)?,
        plain_scale,
        rounded_integer,
    })
}
