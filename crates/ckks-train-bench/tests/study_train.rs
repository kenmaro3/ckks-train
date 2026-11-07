//! Negative frontend integration only. No approved context or HE is fabricated.
use std::{ffi::OsString, fs, process::Command};
fn valid() -> Vec<OsString> {
    let mut a: Vec<OsString> = [
        "--root",
        "/nonexistent",
        "--approval",
        "/nonexistent/approval",
        "--approval-sha256",
        &"a".repeat(64),
        "--receipt",
        "/nonexistent/receipt",
        "--origin",
        "family",
        "--registration",
        "main_i",
        "--profile",
        "iris__mlp2020",
        "--seed",
        "0",
        "--scope",
        "full",
        "--backend",
        if cfg!(feature = "cuda") {
            "cuda"
        } else {
            "cpu"
        },
        "--mode",
        "interactive",
        "--placement",
        "client_refresh",
        "--packing",
        "batched_diagonal",
        "--runtime",
        if cfg!(feature = "cuda") {
            "cuda-host"
        } else if cfg!(feature = "parallel") {
            "cpu-rayon"
        } else {
            "cpu-serial"
        },
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    if cfg!(feature = "parallel") {
        a.extend(["--workers".into(), "2".into()]);
    }
    if cfg!(feature = "cuda") {
        a.extend(["--device-ordinal".into(), "0".into()]);
    }
    a
}
#[test]
fn study_train_invalid_and_ordinary_builds_refuse_without_artifacts() {
    let dir =
        std::env::temp_dir().join(format!("hekit-study-train-refusal-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    for (args, expected) in [
        (vec![], "missing explicit option"),
        (vec!["--output".into(), "old.json".into()], "unknown option"),
        (
            valid(),
            "study runtime/context/registration unavailable or inconsistent",
        ),
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_study_train"))
            .args(args)
            .current_dir(&dir)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert_eq!(String::from_utf8(result.stderr).unwrap().trim(), expected);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
    }
    fs::remove_dir(dir).unwrap();
}
#[test]
fn study_train_facade_excludes_argv0_and_returns_no_context() {
    let mut args = valid();
    args.insert(0, "study_train".into());
    assert!(ckks_train_bench::study_run::run(args).is_err());
    assert!(ckks_train_bench::study_run::run(valid()).is_err());
}
