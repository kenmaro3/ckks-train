use super::*;
fn args(operation: &str) -> Vec<OsString> {
    let mut a = [
        "--root",
        "/fictional",
        "--approval",
        "/fictional/approval",
        "--approval-sha256",
        &"a".repeat(64),
        "--receipt",
        "/fictional/receipt",
        "--operation",
        operation,
        "--profile",
        "iris__emlp1",
        "--seed",
        "0",
        "--reference-root",
        "/fictional/reviewed",
        "--reference-id",
        "reference0",
    ]
    .into_iter()
    .map(Into::into)
    .collect::<Vec<_>>();
    if operation.starts_with("target-") || operation == "lock-target" {
        a.extend(
            [
                "--target-root",
                "/fictional/targets",
                "--campaign",
                "campaign0",
            ]
            .into_iter()
            .map(Into::into),
        );
    }
    if operation.ends_with("recover") {
        a.extend(["--expected-sha256".into(), "b".repeat(64).into()]);
    }
    a
}
#[test]
fn operational_preparation_entry_has_six_closed_roots() {
    for (operation, tag) in [
        ("reference-read", 0),
        ("reference-recover", 1),
        ("reference-snapshot", 2),
        ("target-read", 3),
        ("target-recover", 4),
        ("lock-target", 5),
    ] {
        let raw = args(operation);
        let r = parse(raw.clone()).unwrap();
        assert_eq!(r.operation, tag);
        assert_eq!(r.seeds(), [0]);
        assert!(r.work().is_ok());
        let mut bad = raw;
        let at = bad.iter().position(|s| s == "--seed").unwrap() + 1;
        bad[at] = "3".into();
        if tag == 3 {
            assert_eq!(parse(bad).unwrap().seeds(), [0, 3]);
        } else {
            assert!(parse(bad).is_err());
        }
    }
    let mut bad = args("reference-recover");
    bad.truncate(bad.len() - 2);
    assert!(parse(bad).is_err());
    let mut bad = args("reference-read");
    bad.extend(["--seed".into(), "0".into()]);
    assert!(parse(bad).is_err());
    assert!(parse(args("unknown")).is_err());
}
