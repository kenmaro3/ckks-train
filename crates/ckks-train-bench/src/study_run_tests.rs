use super::*;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
#[test]
fn operational_train_entry_compiles_only_exact_slice_work() {
    let mut args = valid("cpu-serial");
    args.extend(planning_args());
    for (k, v) in [
        ("--profile", "iris__emlp1"),
        ("--receipt-max-bytes", "67108864"),
        ("--receipt-max-records", "1000000"),
        ("--work-geometries", "2"),
        ("--work-initial-states", "2"),
        ("--work-updates", "2"),
        ("--work-handoffs", "0"),
    ] {
        replace(&mut args, k, v);
    }
    assert!(observed_work(&parse(args.clone()).unwrap()).is_ok());
    for (k, v) in [
        ("--profile", "iris__mlp2020"),
        ("--seed", "1"),
        ("--work-updates", "3"),
        ("--receipt-max-records", "1"),
    ] {
        let mut bad = args.clone();
        replace(&mut bad, k, v);
        if let Ok(r) = parse(bad) {
            assert!(observed_work(&r).is_err());
        }
    }
    replace(&mut args, "--consumption", "full");
    replace(&mut args, "--work-updates", "2400");
    assert!(observed_work(&parse(args.clone()).unwrap()).is_ok());
    replace(&mut args, "--registration", "main_ni");
    replace(&mut args, "--mode", "noninteractive");
    replace(&mut args, "--placement", "when_needed");
    assert!(observed_work(&parse(args).unwrap()).is_ok());
    assert!(observed_work(&parse(valid("cpu-rayon")).unwrap()).is_err());
}
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

fn planning_args() -> Vec<OsString> {
    [
        "--consumption",
        "quick",
        "--physical-lanes",
        "auto",
        "--transpose",
        "dedicated",
        "--transpose-pool",
        "none",
        "--eval-hoist",
        "1",
        "--eval-bsgs",
        "1",
        "--eval-lazy-relin",
        "1",
        "--partitions",
        "contiguous",
        "--receipt-max-bytes",
        "1234567",
        "--receipt-max-records",
        "456",
        "--work-geometries",
        "6",
        "--work-initial-states",
        "6",
        "--work-updates",
        "6",
        "--work-handoffs",
        "4",
        "--work-wrappers",
        "1",
    ]
    .into_iter()
    .map(Into::into)
    .collect()
}
#[test]
fn d7_request_group_is_explicit_all_or_none_and_preserves_caps() {
    assert!(parse(valid("cpu-serial")).unwrap().planning.is_none());
    let group = planning_args();
    let mut args = valid("cpu-serial");
    args.extend(group.clone());
    let request = parse(args).unwrap().planning.unwrap();
    assert_eq!(
        request.consumption,
        crate::study_admission::Consumption::Quick
    );
    assert_eq!(request.physical_lanes, None);
    assert!(request.partitions.is_none());
    assert_eq!(request.limits.max_bytes, 1234567);
    assert_eq!(request.limits.max_records, 456);
    assert_eq!(
        request.work_budget.usage(),
        crate::study_admission::PlanningUsage::default()
    );
    for i in (0..group.len()).step_by(2) {
        let mut args = valid("cpu-serial");
        let mut partial = group.clone();
        partial.drain(i..i + 2);
        args.extend(partial);
        assert!(parse(args).is_err(), "omitted {:?}", group[i]);
        let mut args = valid("cpu-serial");
        args.extend(group.clone());
        args.extend([group[i].clone(), group[i + 1].clone()]);
        assert!(parse(args).is_err(), "duplicate {:?}", group[i]);
    }
}
#[test]
fn d7_request_canonical_values_and_quick_seed() {
    let mut base = valid("cpu-serial");
    base.extend(planning_args());
    for (key, values) in [
        ("--consumption", vec!["Quick", "none"]),
        (
            "--physical-lanes",
            vec!["0", "01", "+1", "-1", "18446744073709551616"],
        ),
        ("--transpose", vec!["other"]),
        ("--transpose-pool", vec!["", "1", "0"]),
        ("--eval-hoist", vec!["true", "2"]),
        ("--eval-bsgs", vec!["false", "01"]),
        ("--eval-lazy-relin", vec!["-1"]),
        ("--partitions", vec!["none", "explicit"]),
        (
            "--receipt-max-bytes",
            vec!["01", "-1", "1.0", "18446744073709551616"],
        ),
        ("--work-updates", vec!["+1", "1e0"]),
    ] {
        for value in values {
            let mut args = base.clone();
            replace(&mut args, key, value);
            assert!(parse(args).is_err(), "{key}={value}");
        }
    }
    let mut args = base.clone();
    replace(&mut args, "--seed", "1");
    assert!(parse(args.clone()).is_err());
    replace(&mut args, "--consumption", "full");
    assert!(parse(args).is_ok());
    let mut args = base;
    replace(&mut args, "--transpose", "reuse");
    for bad in [
        "none",
        "",
        "+1",
        "-0",
        "01",
        "-01",
        "1,",
        ",1",
        "2147483648",
        "-2147483649",
        "1, 2",
    ] {
        replace(&mut args, "--transpose-pool", bad);
        assert!(parse(args.clone()).is_err(), "{bad}");
    }
    replace(
        &mut args,
        "--transpose-pool",
        "-2147483648,0,2,-1,2147483647",
    );
    let request = parse(args).unwrap().planning.unwrap();
    match request.transpose {
        ckks_train_nn::dense::TransposeKeys::Reuse(pool) => {
            assert_eq!(pool, vec![i32::MIN, 0, 2, -1, i32::MAX])
        }
        _ => panic!("explicit ordered reuse pool"),
    }
}

#[test]
fn d7_request_zero_caps_are_original_requests_without_default_reset() {
    let mut args = valid("cpu-serial");
    args.extend(planning_args());
    for key in [
        "--receipt-max-bytes",
        "--receipt-max-records",
        "--work-geometries",
        "--work-initial-states",
        "--work-updates",
        "--work-handoffs",
        "--work-wrappers",
    ] {
        replace(&mut args, key, "0");
    }
    let request = parse(args).unwrap().planning.unwrap();
    assert_eq!(request.limits.max_bytes, 0);
    assert_eq!(request.limits.max_records, 0);
    assert_eq!(
        request.work_budget.usage(),
        crate::study_admission::PlanningUsage::default()
    );
}
