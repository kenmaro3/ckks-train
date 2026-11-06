//! Checked whole-graph geometry and encrypted masked primitives for one sample per ciphertext.
//!
//! Includes matrix optimizer preparation, but not stages, refresh, memory or full level admission.
//! Structural metadata is checked; encrypted slot contents cannot be verified without a secret.
//! Ingestion must supply the declared representation. Primitive outputs project structural padding.
mod ops;
mod optimizer;
mod plan;
mod types;

pub use ops::{PackedSamplePair, affine, outer_sum_scaled_masks};
pub use optimizer::{
    PackedLayerState, PackedOptimizer, PackedUpdateCandidate, PackedUpdateGradients,
    PackedUpdateRequest,
};
pub use plan::{
    LayerGeometry, Orientation, PackedGeometryPlan, StageShape, VectorRepr, VectorSpec,
};
pub use types::{PackedGradient, PackedVector, PackedWeights};

pub(crate) fn invalid<T>(message: &str) -> ckks_train_core::error::Result<T> {
    Err(ckks_train_core::error::HekitError::InvalidParams(
        message.into(),
    ))
}
