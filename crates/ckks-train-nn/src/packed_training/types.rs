//! Checked ciphertext metadata. Constructors cannot inspect encrypted slot contents.
use super::{LayerGeometry, VectorSpec, invalid};
use ckks_train_ckks::{DiagonalEntry, HeCiphertext, HeOps};
use ckks_train_core::error::{HekitError, Result};

pub(crate) fn check_ct<E: HeOps>(ev: &E, ct: &E::Ct, slots: usize) -> Result<()> {
    if slots != ev.params().slots()
        || ct.slots() != slots
        || !ev.owns(ct)
        || ct.size() != 2
        || ct.level() > ev.params().max_level()
    {
        return invalid("packed ciphertext context, slots, components or level mismatch");
    }
    if !ct.scale().is_finite() || ct.scale() <= 0.0 {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(())
}

/// Encrypted vector with validated structural metadata. Positive natural scales are allowed
/// for derivative products; affine/gradient outputs return exact canonical Delta.
#[derive(Clone, Debug)]
pub struct PackedVector<C> {
    ct: C,
    spec: VectorSpec,
}
impl<C: HeCiphertext> PackedVector<C> {
    /// Validates context, slots, components, finite positive scale and level.
    /// The caller asserts that the encrypted values have this vector representation.
    pub fn new<E: HeOps<Ct = C>>(ev: &E, spec: VectorSpec, ct: C) -> Result<Self> {
        check_ct(ev, &ct, spec.layout().slots())?;
        Ok(Self { ct, spec })
    }
    /// Ciphertext, without mutable metadata access.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
    /// Vector geometry.
    pub fn spec(&self) -> VectorSpec {
        self.spec
    }
}

/// One matrix ciphertext in its own checked graph layer and orientation, at exact Delta.
#[derive(Clone, Debug)]
pub struct PackedWeights<C> {
    ct: C,
    layer: LayerGeometry,
}
impl<C: HeCiphertext> PackedWeights<C> {
    /// Validates context, shape capacity and exact Delta. The caller asserts packed contents.
    pub fn new<E: HeOps<Ct = C>>(ev: &E, layer: LayerGeometry, ct: C) -> Result<Self> {
        check_ct(ev, &ct, layer.layout().slots())?;
        if ct.scale() != ev.params().scale() {
            return Err(HekitError::ScaleMismatch);
        }
        Ok(Self { ct, layer })
    }
    /// Encrypted weights.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
    /// Immutable layer identity and geometry.
    pub fn layer(&self) -> &LayerGeometry {
        &self.layer
    }
}

/// Exact-Delta masked matrix equal to `factor * sum_valid(input outer dz)`.
/// No implicit averaging, optimizer-rate interpretation or update generation is assigned.
#[derive(Clone, Debug)]
pub struct PackedGradient<C> {
    pub(crate) ct: C,
    pub(crate) layer: LayerGeometry,
    pub(crate) valid_samples: usize,
    pub(crate) factor: f64,
}
impl<C> PackedGradient<C> {
    /// Encrypted active matrix in its weight orientation.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
    /// Layer to which this gradient belongs.
    pub fn layer(&self) -> &LayerGeometry {
        &self.layer
    }
    /// Exact number of supplied valid sample pairs.
    pub fn valid_samples(&self) -> usize {
        self.valid_samples
    }
    /// Public coefficient applied to the sum (not automatically a mean coefficient).
    pub fn factor(&self) -> f64 {
        self.factor
    }
}
