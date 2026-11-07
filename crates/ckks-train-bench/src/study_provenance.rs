//! Approved local build context for the study-reference producer.
//!
//! A context is not a completed reference or scientific execution authorization.
//! Ordinary nonrecipe builds carry no capsule and cannot acquire this capability.
#![allow(dead_code)] // Foundation getters are consumed by the separately assigned D6c2b.
mod codec;
mod context;
pub(crate) mod process;
pub(crate) mod publication;
pub(crate) mod role;

#[allow(unused_imports)]
pub(crate) use context::{BuildFacts, Host, Observation, RuntimeCapture, RuntimeRecheck};
pub use context::{ContextRequest, VerifiedStudyContext, VerifiedStudyTrainContext};
pub(crate) use context::{check_runtime_environment, check_train_preflight};

/// A bounded rejection of unavailable, inconsistent or unsupported evidence.
#[derive(Debug)]
pub struct ProvenanceError(pub(crate) &'static str);
impl std::fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for ProvenanceError {}
impl From<std::io::Error> for ProvenanceError {
    fn from(_: std::io::Error) -> Self {
        Self("provenance I/O unavailable")
    }
}
pub(crate) type Result<T> = std::result::Result<T, ProvenanceError>;
fn ensure(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ProvenanceError(why))
    }
}
include!(concat!(env!("OUT_DIR"), "/study_capsule.rs"));

#[cfg(test)]
mod role_tests;
