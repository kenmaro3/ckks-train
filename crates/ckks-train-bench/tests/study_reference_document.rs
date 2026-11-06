//! Adversarial schema claims, never scientific reference results.
use ckks_train_bench::study_reference::{
    parse_diagnostic_document, parse_failure_document, parse_reference_document,
};
#[allow(dead_code)]
#[path = "../src/study_reference/document_goldens.rs"]
mod golden;
use serde_json::{Value, json};
fn parse(s: &str) -> Result<(), String> {
    let v: Value = serde_json::from_str(s).unwrap();
    match v["kind"].as_str().unwrap() {
        "full_fp32_final" => parse_reference_document(s.as_bytes()).map(|_| ()),
        "diagnostic_quick" | "diagnostic_full" => {
            parse_diagnostic_document(s.as_bytes()).map(|_| ())
        }
        _ => parse_failure_document(s.as_bytes()).map(|_| ()),
    }
    .map_err(|e| format!("{e:?}"))
}
fn reject(base: &str, path: &str, replacement: Value, code: &str) {
    let mut d: Value = serde_json::from_str(base).unwrap();
    *d.pointer_mut(path).unwrap() = replacement;
    let err = parse(&d.to_string()).unwrap_err();
    assert!(err.contains(code), "{path}: expected {code}, got {err}");
}
#[test]
fn independent_schema_goldens_are_untrusted_and_have_distinct_file_digests() {
    for s in [
        golden::REFERENCE,
        golden::CAPTURED_REFERENCE,
        golden::QUICK,
        golden::QUICK_PRIOR,
        golden::QUICK_F64,
        golden::DIAGNOSTIC_FULL,
        golden::FAILURE_NULL,
        golden::FAILURE_PRESENT,
        golden::FAILURE_ENDPOINT_NULL,
        golden::FAILURE_ENDPOINT_PRESENT,
    ] {
        parse(s).unwrap();
    }
    let a = parse_reference_document(golden::REFERENCE.as_bytes()).unwrap();
    let reformatted =
        serde_json::to_string_pretty(&serde_json::from_str::<Value>(golden::REFERENCE).unwrap())
            .unwrap();
    let b = parse_reference_document(reformatted.as_bytes()).unwrap();
    assert_eq!(a.payload_sha256(), golden::REFERENCE_SHA);
    assert_eq!(a.payload_sha256(), b.payload_sha256());
    assert_ne!(a.file_sha256(), b.file_sha256());
    assert_eq!(a.claimed_identity().contract().canonical_bytes(), &[255, 0]);
    assert_eq!(
        a.claimed_identity().seed_receipt().canonical_bytes(),
        &[254, 1, 2]
    );
    assert!(parse_failure_document(golden::REFERENCE.as_bytes()).is_err());
    assert!(parse_diagnostic_document(golden::REFERENCE.as_bytes()).is_err());
    assert!(parse_reference_document(golden::QUICK.as_bytes()).is_err());
    assert!(parse_reference_document(golden::FAILURE_NULL.as_bytes()).is_err());
    assert!(parse_failure_document(golden::QUICK.as_bytes()).is_err());
    assert!(parse_diagnostic_document(golden::FAILURE_NULL.as_bytes()).is_err());
}
#[test]
fn arithmetic_and_variant_matrices_reject_mixed_claims_before_hash() {
    for source in [
        golden::REFERENCE,
        golden::QUICK,
        golden::DIAGNOSTIC_FULL,
        golden::FAILURE_NULL,
    ] {
        for (dtype, acc, cast) in [
            ("f32", "scalar_f32", "canonical_f64"),
            ("f32", "scalar_f64", "f64_to_f32_once"),
            ("f32", "scalar_f64", "canonical_f64"),
            ("f64", "scalar_f32", "f64_to_f32_once"),
            ("f64", "scalar_f32", "canonical_f64"),
            ("f64", "scalar_f64", "f64_to_f32_once"),
        ] {
            let mut d: Value = serde_json::from_str(source).unwrap();
            d["arithmetic"]["dtype"] = json!(dtype);
            d["arithmetic"]["accumulation"] = json!(acc);
            d["arithmetic"]["cast"] = json!(cast);
            assert!(
                parse(&d.to_string())
                    .unwrap_err()
                    .contains("UnsupportedTag")
            );
        }
    }
    reject(
        golden::REFERENCE,
        "/arithmetic",
        json!({"dtype":"f64","accumulation":"scalar_f64","cast":"canonical_f64","reduction":"x","classification":"x"}),
        "UnsupportedTag",
    );
    reject(
        golden::QUICK,
        "/scope",
        json!("full_diagnostic"),
        "UnsupportedTag",
    );
    reject(
        golden::DIAGNOSTIC_FULL,
        "/scope",
        json!("quick_two_updates"),
        "UnsupportedTag",
    );
    reject(
        golden::FAILURE_NULL,
        "/requested_scope",
        json!("full_f64_diagnostic"),
        "UnsupportedTag",
    );
    reject(
        golden::QUICK,
        "/target_eligible",
        json!(true),
        "UnsupportedTag",
    );
}
#[test]
fn full_quick_and_standalone_failure_observation_rules_are_distinct() {
    for (s, p, v, c) in [
        (
            golden::REFERENCE,
            "/completion/actual_updates",
            json!(11),
            "OuterProgressMismatch",
        ),
        (
            golden::REFERENCE,
            "/completion/origin_processed_samples",
            json!(1),
            "OuterProgressMismatch",
        ),
        (
            golden::REFERENCE,
            "/completion/final_endpoint_index",
            json!(1),
            "OuterProgressMismatch",
        ),
        (
            golden::DIAGNOSTIC_FULL,
            "/endpoints",
            json!([]),
            "OuterProgressMismatch",
        ),
        (
            golden::DIAGNOSTIC_FULL,
            "/progress/consumed_samples",
            json!(23),
            "OuterProgressMismatch",
        ),
        (
            golden::DIAGNOSTIC_FULL,
            "/progress/consumed_plan_sha256",
            json!("1".repeat(64)),
            "CrossLinkMismatch",
        ),
        (
            golden::QUICK_PRIOR,
            "/endpoints/0/completed_updates",
            json!(3),
            "OuterProgressMismatch",
        ),
        (
            golden::QUICK_PRIOR,
            "/endpoints/0/processed_samples",
            json!(5),
            "OuterProgressMismatch",
        ),
        (
            golden::QUICK_PRIOR,
            "/endpoints/0/interval_updates",
            json!(2),
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/identity",
            Value::Null,
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/progress",
            Value::Null,
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/progress/completed_updates",
            json!(11),
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/progress/processed_samples",
            json!(23),
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/last_successful_endpoint/interval_updates",
            json!(0),
            "OuterProgressMismatch",
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            "/last_successful_endpoint/interval_samples",
            json!(25),
            "OuterProgressMismatch",
        ),
    ] {
        reject(s, p, v, c);
    }
    let mut d: Value = serde_json::from_str(golden::QUICK_PRIOR).unwrap();
    d["endpoints"][0]["completed_updates"] = json!(2);
    d["endpoints"][0]["interval_updates"] = json!(2);
    d["endpoints"][0]["processed_samples"] = json!(4);
    d["endpoints"][0]["interval_samples"] = json!(4);
    assert!(
        parse(&d.to_string())
            .unwrap_err()
            .contains("ObservationMismatch")
    );
}
#[test]
fn timing_is_required_and_failure_null_never_means_zero() {
    for phase in [
        "resolve",
        "materialize",
        "start",
        "update",
        "checkpoint",
        "finalize",
        "publish",
    ] {
        let s = golden::FAILURE_ENDPOINT_NULL.replace("\"publish\"", &format!("\"{phase}\""));
        let err = parse(&s);
        if phase == "publish" {
            err.unwrap();
        } else {
            assert!(err.unwrap_err().contains("DigestMismatch"));
        }
    }
    reject(golden::REFERENCE, "/timing", Value::Null, "BadJson");
    reject(golden::QUICK, "/timing", Value::Null, "BadJson");
    reject(
        golden::FAILURE_ENDPOINT_PRESENT,
        "/last_successful_endpoint/update_elapsed_ns/value",
        json!(9),
        "TimingMismatch",
    );
    reject(
        golden::FAILURE_PRESENT,
        "/timing/end_to_end_ns",
        json!(10),
        "TimingMismatch",
    );
    reject(
        golden::FAILURE_PRESENT,
        "/timing/start_unix_ns",
        json!(18446744073709551615u64),
        "DigestMismatch",
    );
    for s in [golden::REFERENCE, golden::QUICK, golden::FAILURE_NULL] {
        let mut d: Value = serde_json::from_str(s).unwrap();
        d.as_object_mut().unwrap().remove("timing");
        assert!(parse(&d.to_string()).unwrap_err().contains("MissingField"));
    }
}
#[test]
fn identity_and_payload_integrity_are_separate_from_semantics() {
    for (value, code) in [
        ("", "OutOfRange"),
        ("0", "BadHex"),
        ("FF00", "BadHex"),
        ("fg", "BadHex"),
        ("fe00", "DigestMismatch"),
    ] {
        reject(
            golden::REFERENCE,
            "/identity/contract/canonical_bytes",
            json!(value),
            code,
        );
    }
    reject(
        golden::REFERENCE,
        "/identity/contract/sha256",
        json!(golden::REFERENCE_SHA),
        "DigestMismatch",
    );
    reject(
        golden::REFERENCE,
        "/integrity/payload_sha256",
        json!("0".repeat(64)),
        "DigestMismatch",
    );
    reject(
        golden::REFERENCE,
        "/synthetic",
        json!(true),
        "UnsupportedTag",
    );
    reject(
        golden::REFERENCE,
        "/schema",
        json!("hekit-test-reference-document-v1"),
        "UnsupportedSchema",
    );
    assert!(
        parse_reference_document(golden::REFERENCE.replace("ff00", "\\u0066f00").as_bytes())
            .is_err()
    );
}
#[test]
fn lexical_numbers_accuracy_bits_and_count_ranges_fail_closed() {
    for tok in ["-0", "1.0", "1e0", "18446744073709551616", "01"] {
        let s = golden::REFERENCE.replacen("\"seed\":7", &format!("\"seed\":{tok}"), 1);
        assert!(parse_reference_document(s.as_bytes()).is_err(), "{tok}");
    }
    for (p, v, c) in [
        (
            "/final_observation/value",
            json!(-0.0),
            "ObservationMismatch",
        ),
        ("/final_observation/correct", json!(4), "OutOfRange"),
        ("/final_observation/test_count", json!(0), "OutOfRange"),
        (
            "/final_observation/test_count",
            json!(9007199254740993u64),
            "OutOfRange",
        ),
        (
            "/final_observation/bits",
            json!("3fd5555555555554"),
            "ObservationMismatch",
        ),
    ] {
        reject(golden::REFERENCE, p, v, c);
    }
    for tok in ["1e999", "NaN", "Infinity"] {
        assert!(
            parse_reference_document(
                golden::REFERENCE
                    .replace("0.3333333333333333", tok)
                    .as_bytes()
            )
            .is_err()
        );
    }
}
#[test]
fn unknown_missing_duplicate_escaped_keys_and_document_syntax_reject() {
    for s in [golden::REFERENCE, golden::QUICK, golden::FAILURE_NULL] {
        let mut d: Value = serde_json::from_str(s).unwrap();
        d["target"] = json!(0.9);
        assert!(parse(&d.to_string()).unwrap_err().contains("UnknownField"));
        let dup = s.replacen(
            "\"run_id\":",
            "\"run_id\":\"ADVERSARIAL_ONLY\",\"run_id\":",
            1,
        );
        assert!(parse(&dup).unwrap_err().contains("DuplicateField"));
        let dup = s.replacen(
            "\"run_id\":",
            "\"\\u0072un_id\":\"ADVERSARIAL_ONLY\",\"run_id\":",
            1,
        );
        assert!(parse(&dup).unwrap_err().contains("DuplicateField"));
    }
    for tail in ["{}", ",", "/*x*/"] {
        assert!(
            parse_reference_document(format!("{}{tail}", golden::REFERENCE).as_bytes()).is_err()
        );
    }
    for key in [
        "seed",
        "encoding",
        "dtype",
        "correct",
        "status",
        "api_version",
        "planned_updates",
        "wall_source",
        "hash",
    ] {
        let mut d: Value = serde_json::from_str(golden::REFERENCE).unwrap();
        fn duplicate(v: &mut Value, key: &str) -> bool {
            if let Some(m) = v.as_object_mut() {
                if m.contains_key(key) {
                    m.insert(format!("DUPLICATE_{key}"), m[key].clone());
                    return true;
                }
                for x in m.values_mut() {
                    if duplicate(x, key) {
                        return true;
                    }
                }
            } else if let Some(a) = v.as_array_mut() {
                for x in a {
                    if duplicate(x, key) {
                        return true;
                    }
                }
            }
            false
        }
        assert!(duplicate(&mut d, key));
        let s = d.to_string().replace(&format!("DUPLICATE_{key}"), key);
        assert!(parse(&s).unwrap_err().contains("DuplicateField"), "{key}");
    }
}
#[test]
fn provenance_transport_has_fixed_fields_paths_and_unique_sorted_maps() {
    for (p, v, c) in [
        (
            "/provenance/value/host/execution_threads",
            json!(2),
            "OutOfRange",
        ),
        (
            "/provenance/value/host/available_cpus/value",
            json!(0),
            "OutOfRange",
        ),
        (
            "/provenance/value/source/manifest/0/path",
            json!("a/../b"),
            "OutOfRange",
        ),
        (
            "/provenance/value/thread_environment/0/key",
            json!("SECRET"),
            "UnsupportedTag",
        ),
        (
            "/provenance/value/executable/end_sha256",
            json!("1".repeat(64)),
            "CrossLinkMismatch",
        ),
        (
            "/provenance/value/build/source_manifest_sha256",
            json!("1".repeat(64)),
            "CrossLinkMismatch",
        ),
        (
            "/provenance/value/build/receipt_file_sha256",
            json!("1".repeat(64)),
            "CrossLinkMismatch",
        ),
        (
            "/provenance/value/build/features",
            json!(["a", "a"]),
            "DuplicateField",
        ),
    ] {
        reject(golden::CAPTURED_REFERENCE, p, v, c);
    }
}
#[test]
fn preflight_bounds_depth_tokens_text_and_collections_before_semantic_decode() {
    let deep = format!("{}0{}", "[".repeat(17), "]".repeat(17));
    assert!(
        format!(
            "{:?}",
            parse_reference_document(deep.as_bytes()).unwrap_err()
        )
        .contains("DepthLimit")
    );
    reject(
        golden::REFERENCE,
        "/run_id",
        json!("a".repeat(129)),
        "OutOfRange",
    );
    reject(
        golden::REFERENCE,
        "/writer/api_version",
        json!("a".repeat(4097)),
        "TokenLimit",
    );
    reject(
        golden::REFERENCE,
        "/endpoints",
        json!(vec![
            serde_json::from_str::<Value>(golden::REFERENCE)
                .unwrap()["endpoints"][0]
                .clone();
            401
        ]),
        "OutOfRange",
    );
    let mut d: Value = serde_json::from_str(golden::REFERENCE).unwrap();
    d["x".repeat(65)] = json!(0);
    assert!(parse(&d.to_string()).unwrap_err().contains("TokenLimit"));
    let s = golden::REFERENCE.replace("\"seed\":7", &format!("\"seed\":{}", "1".repeat(65)));
    assert!(
        format!("{:?}", parse_reference_document(s.as_bytes()).unwrap_err()).contains("TokenLimit")
    );
}
#[test]
fn wrong_container_at_large_string_path_is_an_error_not_a_panic() {
    reject(
        golden::REFERENCE,
        "/identity/contract/canonical_bytes",
        json!({"xx":"ff00"}),
        "BadJson",
    );
}
#[path = "../src/study_reference/document_variant_goldens.rs"]
mod variant_golden;
#[test]
fn independent_rehashed_variants_cover_every_failure_phase_and_allowed_scope() {
    for (name, s) in variant_golden::POSITIVE_VARIANTS {
        parse(s).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
#[test]
fn every_nested_object_rejects_unknown_missing_and_duplicate_decoded_keys() {
    fn objects(v: &Value, p: String, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                out.push(p.clone());
                for (k, v) in m {
                    objects(v, format!("{p}/{k}"), out);
                }
            }
            Value::Array(a) => {
                for (i, v) in a.iter().enumerate() {
                    objects(v, format!("{p}/{i}"), out);
                }
            }
            _ => {}
        }
    }
    for source in [
        golden::CAPTURED_REFERENCE,
        golden::FAILURE_ENDPOINT_PRESENT,
        golden::QUICK,
    ] {
        let original: Value = serde_json::from_str(source).unwrap();
        let mut paths = Vec::new();
        objects(&original, String::new(), &mut paths);
        for p in paths {
            let object = original.pointer(&p).unwrap().as_object().unwrap();
            let key = object.keys().next().unwrap();
            let mut v = original.clone();
            v.pointer_mut(&p)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key);
            let e = parse(&v.to_string()).unwrap_err();
            assert!(e.contains("MissingField"), "{p}/{key}: {e}");
            let mut v = original.clone();
            v.pointer_mut(&p).unwrap()["not_allowed"] = json!(true);
            let e = parse(&v.to_string()).unwrap_err();
            assert!(e.contains("UnknownField"), "{p}: {e}");
            let mut v = original.clone();
            v.pointer_mut(&p).unwrap()["DUPLICATE_MARKER"] = object[key].clone();
            let s = v
                .to_string()
                .replace("\"DUPLICATE_MARKER\"", &format!("\"{key}\""));
            let e = parse(&s).unwrap_err();
            assert!(e.contains("DuplicateField"), "{p}/{key}: {e}");
            let escaped_key = format!("\"\\u{:04x}{}\"", key.as_bytes()[0], &key[1..]);
            let escaped = v.to_string().replace("\"DUPLICATE_MARKER\"", &escaped_key);
            let e = parse(&escaped).unwrap_err();
            assert!(e.contains("DuplicateField"), "escaped {p}/{key}: {e}");
        }
    }
}
#[test]
fn collection_caps_total_boundaries_and_duration_conservation_are_enforced() {
    let captured: Value = serde_json::from_str(golden::CAPTURED_REFERENCE).unwrap();
    for (path, n) in [
        ("/provenance/value/source/manifest", 513),
        ("/provenance/value/invocation/argv", 257),
        ("/provenance/value/build/features", 65),
        ("/provenance/value/thread_environment", 33),
    ] {
        let first = captured.pointer(path).unwrap()[0].clone();
        reject(
            golden::CAPTURED_REFERENCE,
            path,
            json!(vec![first; n]),
            "OutOfRange",
        );
    }
    reject(
        golden::REFERENCE,
        "/endpoints/0/crossed_logical_boundaries",
        json!(vec![1; 401]),
        "OutOfRange",
    );
    let mut v: Value = serde_json::from_str(golden::REFERENCE).unwrap();
    let mut first = v["endpoints"][0].clone();
    first["crossed_logical_boundaries"] = json!(vec![1; 201]);
    let mut second = first.clone();
    second["completed_updates"] = json!(24);
    second["processed_samples"] = json!(48);
    v["endpoints"] = json!([first, second]);
    assert!(parse(&v.to_string()).unwrap_err().contains("OutOfRange"));
    reject(
        golden::FAILURE_PRESENT,
        "/timing/setup_ns/value",
        json!(18446744073709551615u64),
        "TimingMismatch",
    );
    reject(
        golden::FAILURE_PRESENT,
        "/timing/end_unix_ns",
        json!(0),
        "DigestMismatch",
    );
    let mut v: Value = serde_json::from_str(golden::FAILURE_PRESENT).unwrap();
    v["timing"]["start_unix_ns"] = json!(1);
    v["timing"]["end_unix_ns"] = json!(0);
    assert!(
        parse(&v.to_string())
            .unwrap_err()
            .contains("TimingMismatch")
    );
    let mut v: Value = serde_json::from_str(golden::REFERENCE).unwrap();
    let first = v["endpoints"][0].clone();
    let mut second = first.clone();
    second["completed_updates"] = json!(24);
    second["processed_samples"] = json!(48);
    second["update_elapsed_ns"]["value"] = json!(3);
    v["endpoints"] = json!([first, second]);
    assert!(
        parse(&v.to_string())
            .unwrap_err()
            .contains("TimingMismatch")
    );
}
#[test]
fn malformed_unicode_and_path_sensitive_string_exceptions_cannot_escape_limits() {
    for token in [r#""\uD800""#, r#""\uDC00""#, r#""\q""#] {
        let s = golden::REFERENCE.replace("\"ADVERSARIAL_ONLY\"", token);
        assert!(parse_reference_document(s.as_bytes()).is_err());
    }
    let mut v: Value = serde_json::from_str(golden::CAPTURED_REFERENCE).unwrap();
    v["provenance"]["value"]["build"]["compiler_vv"] = json!("x".repeat(16385));
    assert!(parse(&v.to_string()).unwrap_err().contains("TokenLimit"));
    reject(
        golden::CAPTURED_REFERENCE,
        "/provenance/value/invocation/argv/0/bytes",
        json!("00".repeat(4097)),
        "TokenLimit",
    );
    // A key substring at another path never authorizes a giant identity token.
    let s = format!(r#"{{"canonical_bytes":"{}"}}"#, "0".repeat(4097));
    assert!(
        format!("{:?}", parse_failure_document(s.as_bytes()).unwrap_err()).contains("TokenLimit")
    );
    let s = format!("{}{}", golden::FAILURE_NULL, " ".repeat(2097152));
    assert!(
        format!("{:?}", parse_failure_document(s.as_bytes()).unwrap_err()).contains("TooLarge")
    );
}
#[test]
fn absolute_raw_path_claims_reject_empty_relative_but_argv_is_generic() {
    for path in [
        "/provenance/value/source/root/bytes",
        "/provenance/value/executable/path/bytes",
        "/provenance/value/invocation/cwd/bytes",
    ] {
        for bytes in ["", "61", "ff"] {
            reject(golden::CAPTURED_REFERENCE, path, json!(bytes), "OutOfRange");
        }
    }
    // The independent '/\xff' captured-reference golden is already a positive case.
    // An empty argv is syntactically valid and reaches the expected stale digest check.
    reject(
        golden::CAPTURED_REFERENCE,
        "/provenance/value/invocation/argv/0/bytes",
        json!(""),
        "DigestMismatch",
    );
}
#[test]
fn all_failure_scope_pairs_and_unknown_fixed_tags_fail_before_integrity() {
    for (dtype, scope, allowed) in [
        ("f32", "full_fp32", true),
        ("f32", "quick", true),
        ("f32", "full_f64_diagnostic", false),
        ("f64", "full_fp32", false),
        ("f64", "quick", true),
        ("f64", "full_f64_diagnostic", true),
    ] {
        let mut v: Value = serde_json::from_str(golden::FAILURE_NULL).unwrap();
        v["requested_scope"] = json!(scope);
        if dtype == "f64" {
            v["arithmetic"]["dtype"] = json!("f64");
            v["arithmetic"]["accumulation"] = json!("scalar_f64");
            v["arithmetic"]["cast"] = json!("canonical_f64");
        }
        let result = parse(&v.to_string());
        if allowed {
            assert!(result.is_ok() || result.unwrap_err().contains("DigestMismatch"));
        } else {
            assert!(result.unwrap_err().contains("UnsupportedTag"));
        }
    }
    for path in [
        "/arithmetic/dtype",
        "/arithmetic/accumulation",
        "/arithmetic/cast",
        "/status",
        "/timing/wall_source",
        "/timing/monotonic_source",
        "/timing/scope",
        "/provenance/status",
        "/provenance/value/source/tracked",
        "/provenance/value/source/untracked",
        "/provenance/value/source/root/encoding",
    ] {
        reject(
            golden::CAPTURED_REFERENCE,
            path,
            json!("future"),
            "UnsupportedTag",
        );
    }
    reject(
        golden::FAILURE_NULL,
        "/phase",
        json!("future"),
        "UnsupportedTag",
    );
    reject(
        golden::FAILURE_NULL,
        "/requested_scope",
        json!("future"),
        "UnsupportedTag",
    );
}

// FIX1: allocation authority depends on actual JSON structure, never dotted display text.
#[test]
fn fix1_flattened_and_escaped_keys_cannot_select_any_large_token_exception() {
    use ckks_train_bench::study_reference::DocumentErrorCode;
    let paths = [
        "identity.contract.canonical_bytes",
        "identity.seed_receipt.canonical_bytes",
        "provenance.value.build.compiler_vv",
        "provenance.value.source.root.bytes",
        "provenance.value.executable.path.bytes",
        "provenance.value.invocation.cwd.bytes",
        "provenance.value.invocation.argv.*.bytes",
    ];
    let payload = "00".repeat(2049);
    let mut unexpected = Vec::new();
    for path in paths {
        for key in [path.to_owned(), path.replace('.', "\\u002e")] {
            let input = format!(r#"{{"{key}":"{payload}"}}"#);
            let error = parse_failure_document(input.as_bytes()).unwrap_err();
            if error.code() != DocumentErrorCode::TokenLimit {
                unexpected.push((key, error.code()));
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "ordinary-token gate bypassed: {unexpected:?}"
    );
}
#[test]
fn fix1_partially_nested_collisions_keep_ordinary_token_limits() {
    use ckks_train_bench::study_reference::DocumentErrorCode;
    let paths = [
        ["identity", "contract", "canonical_bytes"].as_slice(),
        ["identity", "seed_receipt", "canonical_bytes"].as_slice(),
        ["provenance", "value", "build", "compiler_vv"].as_slice(),
        ["provenance", "value", "source", "root", "bytes"].as_slice(),
        ["provenance", "value", "executable", "path", "bytes"].as_slice(),
        ["provenance", "value", "invocation", "cwd", "bytes"].as_slice(),
        ["provenance", "value", "invocation", "argv", "*", "bytes"].as_slice(),
    ];
    let mut unexpected = Vec::new();
    for parts in paths {
        for split in 1..parts.len() - 1 {
            let tail = parts[split..].join(".");
            let mut value = json!({tail: "00".repeat(2049)});
            for key in parts[..split].iter().rev() {
                value = json!({*key: value});
            }
            let error = parse_failure_document(value.to_string().as_bytes()).unwrap_err();
            if error.code() != DocumentErrorCode::TokenLimit {
                unexpected.push((parts.join("."), split, error.code()));
            }
        }
    }
    // Also collapse an internal pair or a root prefix, leaving the later keys nested.
    for parts in paths {
        for pair in 0..parts.len() - 1 {
            let mut keys: Vec<String> = parts.iter().map(|key| (*key).to_owned()).collect();
            keys[pair] = format!("{}.{}", keys[pair], keys[pair + 1]);
            keys.remove(pair + 1);
            let mut value = json!("00".repeat(2049));
            for key in keys.iter().rev() {
                value = json!({key: value});
            }
            for input in [value.to_string(), value.to_string().replace('.', "\\u002e")] {
                let error = parse_failure_document(input.as_bytes()).unwrap_err();
                if error.code() != DocumentErrorCode::TokenLimit {
                    unexpected.push((parts.join("."), pair, error.code()));
                }
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "partial-path token gate bypassed: {unexpected:?}"
    );
}
#[test]
fn fix1_literal_star_object_is_not_an_argv_array_element() {
    use ckks_train_bench::study_reference::DocumentErrorCode;
    let input =
        json!({"provenance":{"value":{"invocation":{"argv":{"*":{"bytes":"00".repeat(2049)}}}}}});
    let error = parse_failure_document(input.to_string().as_bytes()).unwrap_err();
    assert_eq!(error.code(), DocumentErrorCode::TokenLimit);
    let escaped = input.to_string().replace("\"*\"", "\"\\u002a\"");
    assert_eq!(
        parse_failure_document(escaped.as_bytes())
            .unwrap_err()
            .code(),
        DocumentErrorCode::TokenLimit
    );
}
#[test]
fn fix1_raw_outer_budget_counts_flattened_identity_string_contents() {
    use ckks_train_bench::study_reference::DocumentErrorCode;
    let payload = "00".repeat(2048); // Ordinary-token legal; only aggregate A must refuse.
    let mut unexpected = Vec::new();
    for identity in ["contract", "seed_receipt"] {
        for escaped in [false, true] {
            let key = format!("identity.{identity}.canonical_bytes");
            let key = if escaped {
                key.replace('.', "\\u002e")
            } else {
                key
            };
            let member = format!(r#""{key}":"{payload}""#);
            let input = format!("{{{}}}", vec![member; 513].join(","));
            assert!(input.len() > 2_097_152);
            let error = parse_failure_document(input.as_bytes()).unwrap_err();
            if error.code() != DocumentErrorCode::TooLarge {
                unexpected.push((key, error.code()));
            }
        }
    }
    assert!(
        unexpected.is_empty(),
        "raw-outer budget bypassed: {unexpected:?}"
    );
    // The review's single giant dotted token must fail at the earlier token gate.
    let input = format!(
        r#"{{"identity.contract.canonical_bytes":"{}"}}"#,
        "00".repeat(1_048_577)
    );
    assert_eq!(
        parse_failure_document(input.as_bytes()).unwrap_err().code(),
        DocumentErrorCode::TokenLimit
    );
}
