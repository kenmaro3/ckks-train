//! ckks-train-bench: benchmarks and cost-model harness for hekit (temporary crate name).
#![deny(missing_docs)]

pub mod backend;
pub mod data;
pub mod profile_data;
pub mod provenance;
pub mod study_admission;
pub mod study_inputs;
pub mod study_plain;
pub mod study_profile;
pub mod study_provenance;
pub mod study_reference;
pub mod twin;

mod study_runtime;

/// Closed study request frontend; grants no readiness authority.
pub mod study_run;

/// Client-owned CPU study encryption and diagnostics; grants no production admission.
pub mod study_client;
/// Single-stage CPU consuming bridge; production dispatch remains disconnected.
pub mod study_diagonal;
