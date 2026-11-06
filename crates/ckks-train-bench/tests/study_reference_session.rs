//! Actual canonical Iris seed0 two-update diagnostics only; no production completion.
use ckks_train_bench::{
    study_inputs::materialize,
    study_profile::StudyRegistry,
    study_reference::{
        DiagnosticContext, QuickFp32DiagnosticSession, QuickFp64DiagnosticSession,
        derive_identities, parse_diagnostic_document,
    },
};
#[test]
fn actual_iris_two_update_diagnostics_reload_and_third_advance_is_terminal() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let p = materialize(
        &StudyRegistry::load(&root)
            .unwrap()
            .resolve("iris__mlp2020", 0)
            .unwrap(),
    )
    .unwrap();
    let ctx = DiagnosticContext::new(&root).unwrap();
    let (contract, receipt) = derive_identities(&p).unwrap();
    let mut f32 = QuickFp32DiagnosticSession::start(&p, &ctx, "actual_iris_f32").unwrap();
    assert_eq!(f32.advance().unwrap().completed_updates(), 1);
    assert_eq!(f32.advance().unwrap().completed_updates(), 2);
    let done = f32.finish_quick().unwrap();
    let bytes = done.document_bytes().unwrap();
    let parsed = parse_diagnostic_document(&bytes).unwrap();
    assert_eq!(
        parsed.claimed_identity().contract().canonical_bytes(),
        contract.canonical_bytes()
    );
    assert_eq!(
        parsed.claimed_identity().seed_receipt().canonical_bytes(),
        receipt.canonical_bytes()
    );
    assert_eq!(
        parsed.claimed_last_observation().claimed_correct(),
        done.correct()
    );
    assert_eq!(
        parsed.claimed_last_observation().claimed_test_count(),
        done.test_count()
    );
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["synthetic"], false);
    assert_eq!(
        v["provenance"]["reason"],
        "local_diagnostic_no_production_build_receipt"
    );
    assert_eq!(v["progress"]["consumed_updates"], 2);
    assert!(v["progress"]["planned_updates"].as_u64().unwrap() > 2);
    let mut f64 = QuickFp64DiagnosticSession::start(&p, &ctx, "actual_iris_f64").unwrap();
    f64.advance().unwrap();
    f64.advance().unwrap();
    let done64 = f64.finish_quick().unwrap();
    let bytes64 = done64.document_bytes().unwrap();
    let parsed64 = parse_diagnostic_document(&bytes64).unwrap();
    assert_eq!(
        parsed64.claimed_last_observation().claimed_correct(),
        done64.correct()
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes64).unwrap()["arithmetic"]["dtype"],
        "f64"
    );
    for (dtype, bytes) in [("f32", &bytes), ("f64", &bytes64)] {
        let parsed = parse_diagnostic_document(bytes).unwrap();
        let v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        println!(
            "{}",
            serde_json::json!({"evidence":"ACTUAL_CANONICAL_IRIS_SEED0_TWO_UPDATE_DIAGNOSTIC_ONLY","dtype":dtype,"synthetic":false,"updates":v["progress"]["consumed_updates"],"samples":v["progress"]["consumed_samples"],"correct":parsed.claimed_last_observation().claimed_correct(),"test_count":parsed.claimed_last_observation().claimed_test_count(),"contract":parsed.claimed_identity().contract().sha256(),"receipt":parsed.claimed_identity().seed_receipt().sha256(),"full_plan":v["progress"]["full_plan_sha256"],"prefix":v["progress"]["consumed_plan_sha256"],"state":v["last_state_sha256"],"file":parsed.file_sha256(),"payload":parsed.payload_sha256(),"byte_length":bytes.len(),"timing":v["timing"],"provenance":v["provenance"]})
        );
    }
    let mut exceeded = QuickFp32DiagnosticSession::start(&p, &ctx, "third_rejected").unwrap();
    exceeded.advance().unwrap();
    exceeded.advance().unwrap();
    assert!(exceeded.advance().is_err());
    assert!(exceeded.advance().is_err());
    let failure = exceeded.finish_quick().unwrap_err();
    assert_eq!(failure.completed_updates(), Some(2));
    assert!(
        ckks_train_bench::study_reference::parse_failure_document(
            &failure.document_bytes().unwrap()
        )
        .is_ok()
    );
    let mut exceeded64 = QuickFp64DiagnosticSession::start(&p, &ctx, "third_rejected_f64").unwrap();
    exceeded64.advance().unwrap();
    exceeded64.advance().unwrap();
    assert!(exceeded64.advance().is_err());
    let failure64 = exceeded64.finish_quick().unwrap_err();
    assert_eq!(failure64.completed_updates(), Some(2));
    let bytes = failure64.document_bytes().unwrap();
    ckks_train_bench::study_reference::parse_failure_document(&bytes).unwrap();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["arithmetic"]["dtype"],
        "f64"
    );
}
