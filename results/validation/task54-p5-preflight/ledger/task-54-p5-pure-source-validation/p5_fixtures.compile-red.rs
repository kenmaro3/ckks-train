//! NONSCIENTIFIC_TEST_EVIDENCE: closed synthetic block sensitivity, no HE admission.

#[test]
#[ignore = "requires separate root pure-runtime release; no HE"]
fn p5_pure_block_sensitivity_gate() {
    let report = run_pure_sensitivity();
    assert_eq!(report.counts.train, 156);
    assert_eq!(report.counts.engines, 61);
    assert_eq!(report.counts.forward, 179);
    assert!(report.failures.is_empty(), "{report:#?}");
}
