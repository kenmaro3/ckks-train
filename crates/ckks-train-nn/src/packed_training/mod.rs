//! Checked whole-graph geometry and encrypted masked primitives for one sample per ciphertext.
//!
//! Includes immutable encrypted stage preparation, typed client refresh and state projection.
//! Memory, range and NI admission remain separate.
//! Structural metadata is checked; encrypted slot contents cannot be verified without a secret.
//! Ingestion must supply the declared representation. Primitive outputs project structural padding.
mod ops;
mod optimizer;
mod plan;
mod refresh;
mod stage;
mod types;

pub use ops::{PackedSamplePair, affine, outer_sum_scaled_masks};
pub use optimizer::{
    PackedLayerState, PackedOptimizer, PackedUpdateCandidate, PackedUpdateGradients,
    PackedUpdateRequest,
};
pub use plan::{
    LayerGeometry, Orientation, PackedGeometryPlan, StageShape, VectorRepr, VectorSpec,
};
pub use refresh::{
    CanonicalHidden, ClientRefreshTiming, PackedBootInput, PackedClientRefresh,
    PackedInteractiveReceipt, PackedInteractiveResult, PackedInteractiveStage, PackedRefreshBatch,
    PackedRefreshCandidate, PackedRefreshContract, PackedRefreshEvent, PackedRefreshFailure,
    PackedRefreshPhase, PackedRefreshRequest, PackedRefreshResponse, PackedRefreshRole,
    PackedRefreshShape, project_state,
};
pub use types::{PackedGradient, PackedVector, PackedWeights};

pub(crate) fn invalid<T>(message: &str) -> ckks_train_core::error::Result<T> {
    Err(ckks_train_core::error::HekitError::InvalidParams(
        message.into(),
    ))
}

pub use stage::{
    PackedRecord, PackedStage, PackedStageCandidate, PackedStageConfig, PackedStepPlan,
    PackedStepReceipt, PackedStepResult, PackedTraceEntry,
};
