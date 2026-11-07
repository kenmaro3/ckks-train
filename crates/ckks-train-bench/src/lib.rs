//! ckks-train-bench: benchmarks and cost-model harness for hekit (temporary crate name).
#![deny(missing_docs)]

pub mod backend;
pub mod data;
pub mod profile_data;
pub mod provenance;
pub mod study_inputs;
pub mod study_plain;
pub mod study_profile;
pub mod study_provenance;
pub mod study_reference;
pub mod twin;

mod study_runtime;
