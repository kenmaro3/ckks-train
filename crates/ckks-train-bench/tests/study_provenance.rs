//! Production context cannot be acquired from an ordinary development build.
use ckks_train_bench::study_provenance::{ContextRequest, VerifiedStudyContext};
use std::path::PathBuf;
#[test]
fn ordinary_build_never_grants_production_context() {
    let request = ContextRequest::new(
        PathBuf::from("/not-a-source"),
        PathBuf::from("/not-an-approval"),
        "0".repeat(64),
        vec![PathBuf::from("/not-a-receipt")],
    );
    assert!(VerifiedStudyContext::load(request).is_err());
}
