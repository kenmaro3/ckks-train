use super::*;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
fn valid(runtime: &str) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "--root",
        "/fictional",
        "--approval",
        "/fictional/approval",
        "--approval-sha256",
        &"a".repeat(64),
        "--receipt",
        "/fictional/receipt",
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
        if runtime == "cuda-host" {
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
        runtime,
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    if runtime != "cpu-serial" {
        args.extend(["--workers".into(), "2".into()]);
    }
    if runtime == "cuda-host" {
        args.extend(["--device-ordinal".into(), "0".into()]);
    }
    args
}
fn replace(args: &mut [OsString], key: &str, value: &str) {
    let i = args.iter().position(|x| x == key).unwrap();
    args[i + 1] = value.into();
}
#[test]
fn study_run_explicit_options_and_runtime_forms() {
    for runtime in ["cpu-serial", "cpu-rayon", "cuda-host"] {
        assert!(parse(valid(runtime)).is_ok(), "{runtime}");
    }
    let mut args = valid("cpu-serial");
    args.extend(["--receipt".into(), "/fictional/two".into()]);
    assert!(parse(args).is_ok());
    let mut args = valid("cpu-serial");
    replace(&mut args, "--seed", "9");
    assert_eq!(parse(args).unwrap().case.seed, 9);
}
#[test]
fn study_run_no_defaults_ignored_options_or_duplicate_singletons() {
    let args = valid("cpu-serial");
    for i in (0..args.len()).step_by(2) {
        let mut missing = args.clone();
        missing.drain(i..i + 2);
        assert!(parse(missing).is_err(), "missing {:?}", args[i]);
    }
    for i in (0..args.len()).step_by(2) {
        if args[i] != "--receipt" {
            let mut duplicate = args.clone();
            duplicate.extend([args[i].clone(), args[i + 1].clone()]);
            assert!(parse(duplicate).is_err());
        }
    }
    for tail in [
        vec!["positional"],
        vec!["--output", "old.json"],
        vec!["--campaign", "c"],
        vec!["--workers", "1"],
        vec!["--device-ordinal", "0"],
        vec!["--unknown"],
    ] {
        let mut a = args.clone();
        a.extend(tail.into_iter().map(Into::into));
        assert!(parse(a).is_err());
    }
    assert!(parse(vec![]).is_err());
}
#[test]
fn study_run_rejects_bad_vocabulary_bounds_and_incompatible_runtime() {
    for (key, values) in [
        ("--origin", vec!["unknown"]),
        (
            "--seed",
            vec!["10", "-1", "1.0", "1e0", "18446744073709551616"],
        ),
        ("--scope", vec!["quick", "FULL"]),
        ("--backend", vec!["external_cpu", "cuda"]),
        ("--mode", vec!["bad"]),
        ("--placement", vec!["bad"]),
        ("--packing", vec!["bad"]),
        ("--runtime", vec!["arbitrary"]),
        (
            "--approval-sha256",
            vec![
                "a",
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            ],
        ),
        ("--registration", vec!["", "../FULL"]),
        ("--profile", vec!["", "../old"]),
    ] {
        for value in values {
            let mut a = valid("cpu-serial");
            replace(&mut a, key, value);
            assert!(parse(a).is_err(), "{key}={value}");
        }
    }
    for value in ["0", "-1", "1.0", "1e1", "01", "18446744073709551616"] {
        let mut a = valid("cpu-rayon");
        replace(&mut a, "--workers", value);
        assert!(parse(a).is_err(), "workers={value}");
    }
    let mut a = valid("cpu-rayon");
    let i = a.iter().position(|x| x == "--workers").unwrap();
    a.drain(i..i + 2);
    assert!(parse(a).is_err());
    let mut a = valid("cuda-host");
    let i = a.iter().position(|x| x == "--device-ordinal").unwrap();
    a.drain(i..i + 2);
    assert!(parse(a).is_err());
    let mut a = valid("cuda-host");
    replace(&mut a, "--device-ordinal", "-1");
    assert!(parse(a).is_err());
}
#[test]
fn study_run_preserves_raw_paths_but_rejects_non_utf8_semantics() {
    let mut a = valid("cpu-serial");
    let i = a.iter().position(|x| x == "--root").unwrap();
    let raw = OsString::from_vec(b"/fictional/\xff".to_vec());
    a[i + 1] = raw.clone();
    let request = parse(a).unwrap();
    assert_eq!(
        request.root.as_os_str().as_bytes(),
        raw.as_os_str().as_bytes()
    );
    let mut a = valid("cpu-serial");
    let i = a.iter().position(|x| x == "--mode").unwrap();
    a[i + 1] = OsString::from_vec(vec![255]);
    assert!(parse(a).is_err());
    let mut a = valid("cpu-serial");
    for _ in 1..64 {
        a.extend(["--receipt".into(), "/fictional/r".into()]);
    }
    assert!(parse(a.clone()).is_ok());
    a.extend(["--receipt".into(), "/fictional/r".into()]);
    assert!(parse(a).is_err());
}

#[test]
fn study_run_existing_argument_byte_count_and_path_bounds() {
    let mut a = valid("cpu-serial");
    let i = a.iter().position(|x| x == "--root").unwrap();
    a[i + 1] = format!("/{}", "x".repeat(4095)).into();
    assert!(parse(a.clone()).is_ok());
    a[i + 1] = format!("/{}", "x".repeat(4096)).into();
    assert!(parse(a).is_err());
    let mut a = valid("cpu-serial");
    let i = a.iter().position(|x| x == "--root").unwrap();
    a[i + 1] = OsString::from_vec(b"/bad\0path".to_vec());
    assert!(parse(a).is_err());
    assert!(parse(vec![OsString::from("x"); 256]).is_err());
    let mut a = valid("cpu-serial");
    a.push("--workers".into());
    assert!(parse(a).is_err());
}

#[test]
fn study_run_parsed_request_selects_real_registration_without_readiness() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned();
    let registry = crate::study_profile::StudyRegistry::load(root).unwrap();
    let cases = registry.registered_cases().unwrap();
    for form in ["cpu-serial", "cpu-rayon", "cuda-host"] {
        let request = parse(valid(form)).unwrap();
        let selected = cases.select(&request.case).unwrap();
        assert_eq!(selected.selector(), &request.case);
        let mut wrong = request.case;
        wrong.registration_id = "unknown".into();
        assert!(cases.select(&wrong).is_err());
    }
    assert_eq!(cases.family_cases().len(), 2680);
    assert_eq!(cases.exact_iris_cases().len(), 20);
    assert_eq!(cases.supplements().len(), 7);
}
