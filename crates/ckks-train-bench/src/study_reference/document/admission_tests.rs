//! Frozen wire-only claims. No registry, materializer, context or native construction.
use super::super::*;
use super::*;
use serde_json::{Value, json};
const INITIAL: &[u8] = include_bytes!("../target_goldens/initial.json");
const RETRY_NONE: &[u8] = include_bytes!("../target_goldens/retry-no-start.json");
const RETRY_SOME: &[u8] = include_bytes!("../target_goldens/retry-with-start.json");
const START_INITIAL: &[u8] = include_bytes!("../target_goldens/start-initial.json");
const START_RETRY: &[u8] = include_bytes!("../target_goldens/start-retry.json");
const LOCK: &[u8] = include_bytes!("../target_goldens/lock.json");
const REFERENCE_BYTES: &[u8] = include_bytes!("../target_goldens/reference.json");
fn unhex(s: &str) -> Vec<u8> {
    s.trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
fn value(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap()
}
fn parse_value(v: &Value) -> Result<SyntheticAdmission> {
    SyntheticAdmission::parse(&serde_json::to_vec(v).unwrap())
}
#[test]
fn d7_admission_frozen_documents_complete_bytes_and_hashes() {
    let manifest = value(include_bytes!("../target_goldens/manifest.json"));
    for (name, bytes, hex) in [
        (
            "initial",
            INITIAL,
            include_str!("../target_goldens/initial.canonical.hex"),
        ),
        (
            "retry-no-start",
            RETRY_NONE,
            include_str!("../target_goldens/retry-no-start.canonical.hex"),
        ),
        (
            "retry-with-start",
            RETRY_SOME,
            include_str!("../target_goldens/retry-with-start.canonical.hex"),
        ),
    ] {
        let a = SyntheticAdmission::parse(bytes).unwrap();
        assert_eq!(a.0.canonical().unwrap(), unhex(hex));
        assert_eq!(a.0.file_sha256, manifest["documents"][name]["file_sha256"]);
        assert_eq!(
            a.0.integrity.payload_sha256.0,
            manifest["documents"][name]["payload_sha256"]
        );
        assert!(parse_admission(bytes).is_err());
    }
    for (name, bytes, hex) in [
        (
            "start-initial",
            START_INITIAL,
            include_str!("../target_goldens/start-initial.canonical.hex"),
        ),
        (
            "start-retry",
            START_RETRY,
            include_str!("../target_goldens/start-retry.canonical.hex"),
        ),
    ] {
        let s = SyntheticStart::parse(bytes).unwrap();
        assert_eq!(s.0.canonical().unwrap(), unhex(hex));
        assert_eq!(s.0.file_sha256, manifest["documents"][name]["file_sha256"]);
        assert_eq!(
            s.0.integrity.payload_sha256.0,
            manifest["documents"][name]["payload_sha256"]
        );
        assert!(parse_start(bytes).is_err());
    }
}
#[test]
fn d7_admission_six_frozen_branches_and_domain_components() {
    let manifest = value(include_bytes!("../target_goldens/manifest.json"));
    for (branch, bytes, full, case, execution, gates) in [
        (
            0,
            include_bytes!("../target_goldens/branch-0.json").as_slice(),
            include_str!("../target_goldens/branch-0.canonical.hex"),
            include_str!("../target_goldens/case-0.canonical.hex"),
            include_str!("../target_goldens/execution-0.canonical.hex"),
            include_str!("../target_goldens/gates-0.canonical.hex"),
        ),
        (
            1,
            include_bytes!("../target_goldens/branch-1.json").as_slice(),
            include_str!("../target_goldens/branch-1.canonical.hex"),
            include_str!("../target_goldens/case-1.canonical.hex"),
            include_str!("../target_goldens/execution-1.canonical.hex"),
            include_str!("../target_goldens/gates-1.canonical.hex"),
        ),
        (
            2,
            include_bytes!("../target_goldens/branch-2.json").as_slice(),
            include_str!("../target_goldens/branch-2.canonical.hex"),
            include_str!("../target_goldens/case-2.canonical.hex"),
            include_str!("../target_goldens/execution-2.canonical.hex"),
            include_str!("../target_goldens/gates-2.canonical.hex"),
        ),
        (
            3,
            include_bytes!("../target_goldens/branch-3.json").as_slice(),
            include_str!("../target_goldens/branch-3.canonical.hex"),
            include_str!("../target_goldens/case-3.canonical.hex"),
            include_str!("../target_goldens/execution-3.canonical.hex"),
            include_str!("../target_goldens/gates-3.canonical.hex"),
        ),
        (
            4,
            include_bytes!("../target_goldens/branch-4.json").as_slice(),
            include_str!("../target_goldens/branch-4.canonical.hex"),
            include_str!("../target_goldens/case-4.canonical.hex"),
            include_str!("../target_goldens/execution-4.canonical.hex"),
            include_str!("../target_goldens/gates-4.canonical.hex"),
        ),
        (
            5,
            include_bytes!("../target_goldens/branch-5.json").as_slice(),
            include_str!("../target_goldens/branch-5.canonical.hex"),
            include_str!("../target_goldens/case-5.canonical.hex"),
            include_str!("../target_goldens/execution-5.canonical.hex"),
            include_str!("../target_goldens/gates-5.canonical.hex"),
        ),
    ] {
        let a = SyntheticAdmission::parse(bytes).unwrap().0;
        assert_eq!(a.canonical().unwrap(), unhex(full), "branch {branch}");
        assert_eq!(
            a.file_sha256,
            manifest["documents"][format!("branch-{branch}")]["file_sha256"]
        );
        assert_eq!(
            a.integrity.payload_sha256.0,
            manifest["documents"][format!("branch-{branch}")]["payload_sha256"]
        );
        assert_eq!(component(None, &a.case).unwrap(), unhex(case));
        assert_eq!(component(None, &a.execution).unwrap(), unhex(execution));
        assert_eq!(component(None, &a.gates).unwrap(), unhex(gates));
        assert_eq!(
            sha256_hex(&component(Some("hekit-study-registered-case-v1"), &a.case).unwrap()),
            a.case_id.0
        );
        assert_eq!(
            sha256_hex(&a.execution.fields().binding_bytes().unwrap()),
            a.execution_binding_sha256.0
        );
        assert_eq!(
            sha256_hex(&component(Some("hekit-study-admission-gates-v1"), &a.gates).unwrap()),
            a.gates_sha256.0
        );
        assert!(parse_admission(bytes).is_err());
    }
    let a = SyntheticAdmission::parse(INITIAL).unwrap().0;
    assert_eq!(
        component(Some("hekit-study-registered-case-v1"), &a.case).unwrap(),
        unhex(include_str!(
            "../target_goldens/registered-case.canonical.hex"
        ))
    );
    assert_eq!(
        a.execution.fields().binding_bytes().unwrap(),
        unhex(include_str!("../target_goldens/case-binding.canonical.hex"))
    );
    assert_eq!(
        component(Some("hekit-study-admission-gates-v1"), &a.gates).unwrap(),
        unhex(include_str!(
            "../target_goldens/admission-gates.canonical.hex"
        ))
    );
    for (bytes, hex) in [
        (
            INITIAL,
            include_str!("../target_goldens/predecessor-null.canonical.hex"),
        ),
        (
            RETRY_NONE,
            include_str!("../target_goldens/predecessor-no-start.canonical.hex"),
        ),
        (
            RETRY_SOME,
            include_str!("../target_goldens/predecessor-with-start.canonical.hex"),
        ),
    ] {
        let a = SyntheticAdmission::parse(bytes).unwrap().0;
        assert_eq!(component(None, &a.predecessor).unwrap(), unhex(hex));
    }
}
#[test]
fn d7_admission_target_start_and_explicit_predecessor_claim_links() {
    let lock = super::super::target::SyntheticTargetLock::parse(LOCK).unwrap();
    let reference = synthetic_reference(REFERENCE_BYTES);
    let initial = SyntheticAdmission::parse(INITIAL).unwrap();
    let retry_none = SyntheticAdmission::parse(RETRY_NONE).unwrap();
    let retry_some = SyntheticAdmission::parse(RETRY_SOME).unwrap();
    let start_initial = SyntheticStart::parse(START_INITIAL).unwrap();
    let start_retry = SyntheticStart::parse(START_RETRY).unwrap();
    for a in [&initial, &retry_none, &retry_some] {
        a.check_target(&lock, REFERENCE_BYTES, &reference).unwrap();
    }
    check_start_claims(&start_initial.0, &initial.0).unwrap();
    check_start_claims(&start_retry.0, &retry_some.0).unwrap();
    check_predecessor_claims(&retry_none.0, &initial.0, None).unwrap();
    check_predecessor_claims(&retry_some.0, &initial.0, Some(&start_initial.0)).unwrap();
    assert!(check_predecessor_claims(&retry_none.0, &initial.0, Some(&start_initial.0)).is_err());
    assert!(check_predecessor_claims(&retry_some.0, &initial.0, None).is_err());
    assert!(check_predecessor_claims(&initial.0, &initial.0, None).is_err());
    assert!(check_start_claims(&start_retry.0, &initial.0).is_err());
    let bad = SyntheticAdmission::parse(include_bytes!(
        "../target_goldens/prior-start-link-mismatch.negative.json"
    ))
    .unwrap();
    assert!(check_predecessor_claims(&bad.0, &initial.0, Some(&start_initial.0)).is_err());
    assert!(
        initial
            .check_target(&lock, &[REFERENCE_BYTES, b" "].concat(), &reference)
            .is_err()
    );
}
#[test]
fn d7_admission_required_nested_fields_unknowns_types_and_link_mutations() {
    fn objects(v: &Value, p: String, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                out.push(p.clone());
                for (k, v) in m {
                    objects(v, format!("{p}/{k}"), out)
                }
            }
            Value::Array(a) => {
                for (i, v) in a.iter().enumerate() {
                    objects(v, format!("{p}/{i}"), out)
                }
            }
            _ => (),
        }
    }
    for bytes in [INITIAL, RETRY_SOME, START_INITIAL] {
        let v = value(bytes);
        let mut paths = Vec::new();
        objects(&v, String::new(), &mut paths);
        for p in paths {
            for key in v.pointer(&p).unwrap().as_object().unwrap().keys() {
                let mut bad = v.clone();
                bad.pointer_mut(&p)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .remove(key);
                assert!(
                    if bytes == START_INITIAL {
                        SyntheticStart::parse(&serde_json::to_vec(&bad).unwrap()).is_err()
                    } else {
                        parse_value(&bad).is_err()
                    },
                    "missing {p}/{key}"
                );
            }
            let mut bad = v.clone();
            bad.pointer_mut(&p)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unexpected".into(), json!(true));
            assert!(
                if bytes == START_INITIAL {
                    SyntheticStart::parse(&serde_json::to_vec(&bad).unwrap()).is_err()
                } else {
                    parse_value(&bad).is_err()
                },
                "unknown {p}"
            );
        }
    }
    for (pointer, replacement) in [
        ("/selected_input/seed", json!(10)),
        ("/case/seed", json!(6)),
        ("/case/profile_id", json!("OTHER")),
        ("/case/logical_batch", json!(0)),
        ("/case/scope", json!("quick")),
        ("/case/backend", json!("gpu")),
        ("/case_id", json!("0".repeat(64))),
        ("/execution_binding_sha256", json!("0".repeat(64))),
        ("/gates_sha256", json!("0".repeat(64))),
        ("/contract/sha256", json!("0".repeat(64))),
        ("/execution/bootstrap_engine", json!("auto")),
        ("/execution/eval_config/hoist", json!(1)),
        ("/integrity/hash", json!("sha512")),
        ("/schema", json!("hekit-test-he-admission-v999")),
        ("/lock_id", json!("0".repeat(64))),
        ("/policy/design_documents", json!([])),
        ("/predecessor", json!({})),
        ("/attempt_id", json!("bad/id")),
        ("/provenance/value/host/execution_threads", json!(0)),
    ] {
        let mut bad = value(INITIAL);
        *bad.pointer_mut(pointer).unwrap() = replacement;
        assert!(parse_value(&bad).is_err(), "{pointer}");
    }
    let raw = std::str::from_utf8(INITIAL).unwrap();
    for bad in [raw.replacen("\"seed\": 7","\"seed\": 18446744073709551616",1),raw.replacen("\"seed\": 7","\"seed\": 1e0",1),raw.replacen("\"seed\": 7","\"seed\": -0",1),raw.replacen("\"seed\": 7","\"seed\": 1.0",1),raw.replacen("\"case_id\":", "\"case_id\": \"0000000000000000000000000000000000000000000000000000000000000000\", \"\\u0063ase_id\":",1),format!("{raw} {{}}"),raw.replacen("46494354494f4e414c2d434f4e545241435400ff","\\u00346494354494f4e414c2d434f4e545241435400ff",1)] {assert!(SyntheticAdmission::parse(bad.as_bytes()).is_err());}
}
#[test]
fn d7_admission_closed_layout_arithmetic_and_small_lexical_boundaries() {
    assert_eq!(DocumentLayout::Admission.file_cap(), 70_492_488);
    assert_eq!(DocumentLayout::Admission.canonical_cap(), 36_294_820);
    assert_eq!(DocumentLayout::Start.file_cap(), A_MAX);
    assert_eq!(DocumentLayout::Start.canonical_cap(), A_MAX);
    assert!(bounded_add(70_492_488, 1, 70_492_488, "").is_err());
    assert!(bounded_add(usize::MAX, 1, usize::MAX, "").is_err());
    assert_eq!(hex_len(2 * C_MAX, C_MAX, "").unwrap(), C_MAX);
    assert!(hex_len(2 * C_MAX + 2, C_MAX, "").is_err());
    assert_eq!(hex_len(2 * R_MAX, R_MAX, "").unwrap(), R_MAX);
    assert!(hex_len(2 * R_MAX + 2, R_MAX, "").is_err());
    assert!(hex_len(usize::MAX, R_MAX, "").is_err());
    let text = "00".repeat(4097);
    for (raw, allowed) in [
        (
            format!("{{\"contract\":{{\"canonical_bytes\":\"{text}\"}}}}"),
            true,
        ),
        (
            format!(
                "{{\"selected_input\":{{\"seed_receipt\":{{\"canonical_bytes\":\"{text}\"}}}}}}"
            ),
            true,
        ),
        (
            format!("{{\"selected_input.seed_receipt.canonical_bytes\":\"{text}\"}}"),
            false,
        ),
        (
            format!(
                "{{\"selected_input\":[{{\"seed_receipt\":{{\"canonical_bytes\":\"{text}\"}}}}]}}"
            ),
            false,
        ),
        (
            format!("{{\"identity\":{{\"contract\":{{\"canonical_bytes\":\"{text}\"}}}}}}"),
            false,
        ),
    ] {
        assert_eq!(
            preflight_layout(raw.as_bytes(), DocumentLayout::Admission).is_ok(),
            allowed
        );
        assert!(preflight_layout(raw.as_bytes(), DocumentLayout::Start).is_err());
    }
    for layout in [DocumentLayout::Admission, DocumentLayout::Start] {
        let deep = format!("{}0{}", "[".repeat(17), "]".repeat(17));
        assert!(preflight_layout(deep.as_bytes(), layout).is_err());
        let long = format!("{{\"x\":\"{}\"}}", "\\u0061".repeat(4097));
        assert!(preflight_layout(long.as_bytes(), layout).is_err());
        let mut e = Encoder::new(layout);
        e.outer = A_MAX;
        assert!(e.byte(0).is_err());
        let mut e = Encoder::new(layout);
        e.outer = A_MAX - 1;
        e.byte(0).unwrap();
        assert!(e.byte(0).is_err());
    }
}

// Separate private synthetic wrapper construction. It uses inherited fixed
// records/Fields and canonical digest verification, never the production parser
// with a synthetic flag, never a VerifiedReference or approved context.
fn synthetic_reference(bytes: &[u8]) -> ParsedReferenceDocument {
    preflight_layout(bytes, DocumentLayout::Reference).unwrap();
    let raw = serde_json::from_slice::<&RawValue>(bytes).unwrap();
    let mut f = Fields::new(
        raw,
        &[
            "schema",
            "kind",
            "status",
            "run_id",
            "writer",
            "synthetic",
            "identity",
            "arithmetic",
            "completion",
            "endpoints",
            "final_observation",
            "timing",
            "provenance",
            "integrity",
        ],
        "",
    )
    .unwrap();
    let mut c = Common {
        schema: f.field("schema", "").unwrap(),
        kind: f.field("kind", "").unwrap(),
        status: f.field("status", "").unwrap(),
        run_id: f.field("run_id", "").unwrap(),
        writer: f.field("writer", "").unwrap(),
        synthetic: f.field("synthetic", "").unwrap(),
        arithmetic: f.field("arithmetic", "").unwrap(),
        provenance: f.field("provenance", "").unwrap(),
        integrity: f.field("integrity", "").unwrap(),
        file_sha256: sha256_hex(bytes),
    };
    assert_eq!(c.schema.0, "hekit-test-reference-document-v1");
    assert!(c.synthetic);
    assert_eq!(c.kind, Kind::Reference);
    assert_eq!(c.status, Status::Complete);
    let b = ReferenceBody {
        identity: f.field("identity", "").unwrap(),
        completion: f.field("completion", "").unwrap(),
        endpoints: f.field("endpoints", "").unwrap(),
        final_observation: f.field("final_observation", "").unwrap(),
        timing: f.field("timing", "").unwrap(),
    };
    provenance(&mut c, true).unwrap();
    timing(&b.timing).unwrap();
    endpoints(&b.endpoints.0, &b.timing).unwrap();
    verify_digest(&c, reference_encoding(&c, &b).unwrap()).unwrap();
    ParsedReferenceDocument { common: c, body: b }
}

fn rehashed_admission(mut v: Value) -> Vec<u8> {
    let a = decode_admission(&serde_json::to_vec(&v).unwrap()).unwrap();
    v["integrity"]["payload_sha256"] = json!(sha256_hex(&a.canonical().unwrap()));
    serde_json::to_vec(&v).unwrap()
}
fn rehashed_start(mut v: Value) -> Vec<u8> {
    let s = decode_start(&serde_json::to_vec(&v).unwrap()).unwrap();
    v["integrity"]["payload_sha256"] = json!(sha256_hex(&s.canonical().unwrap()));
    serde_json::to_vec(&v).unwrap()
}
#[test]
fn d7_admission_rehashed_cross_record_fields_and_clock_ordering() {
    let lock = super::super::target::SyntheticTargetLock::parse(LOCK).unwrap();
    let reference = synthetic_reference(REFERENCE_BYTES);
    let initial = SyntheticAdmission::parse(INITIAL).unwrap();
    for pointer in [
        "/lock/file_sha256",
        "/lock/payload_sha256",
        "/reference/file_sha256",
        "/reference/payload_sha256",
        "/policy/registry_sha256",
        "/policy/portable_lock_sha256",
        "/policy/approval_sha256",
        "/policy/design_documents/0/sha256",
    ] {
        let mut bad = value(INITIAL);
        *bad.pointer_mut(pointer).unwrap() = json!("0".repeat(64));
        if pointer == "/policy/registry_sha256" {
            bad["case"]["registry_sha256"] = json!("0".repeat(64));
            let a = decode_admission(&serde_json::to_vec(&bad).unwrap()).unwrap();
            bad["case_id"] = json!(sha256_hex(&component(Some(CASE_DOMAIN), &a.case).unwrap()));
        }
        let a = SyntheticAdmission::parse(&rehashed_admission(bad)).unwrap();
        assert!(
            a.check_target(&lock, REFERENCE_BYTES, &reference).is_err(),
            "{pointer}"
        );
    }
    for pointer in [
        "/case_id",
        "/execution_binding_sha256",
        "/seed_receipt_sha256",
        "/materialized_sha256",
        "/admission/file_sha256",
        "/admission/payload_sha256",
        "/lock/file_sha256",
        "/lock/payload_sha256",
        "/reference/file_sha256",
        "/reference/payload_sha256",
        "/policy/approval_sha256",
    ] {
        let mut bad = value(START_INITIAL);
        *bad.pointer_mut(pointer).unwrap() = json!("0".repeat(64));
        let s = SyntheticStart::parse(&rehashed_start(bad)).unwrap();
        assert!(check_start_claims(&s.0, &initial.0).is_err(), "{pointer}");
    }
    for (pointer, replacement) in [
        ("/seed", json!(8)),
        ("/attempt_id", json!("OTHER")),
        ("/started/unix_ns", json!(39)),
    ] {
        let mut bad = value(START_INITIAL);
        *bad.pointer_mut(pointer).unwrap() = replacement;
        let s = SyntheticStart::parse(&rehashed_start(bad)).unwrap();
        assert!(check_start_claims(&s.0, &initial.0).is_err());
    }
    let mut equal_time = value(START_INITIAL);
    equal_time["started"]["unix_ns"] = json!(40);
    check_start_claims(
        &SyntheticStart::parse(&rehashed_start(equal_time))
            .unwrap()
            .0,
        &initial.0,
    )
    .unwrap();
    let mut before_lock = value(INITIAL);
    before_lock["admitted"]["unix_ns"] = json!(29);
    assert!(
        SyntheticAdmission::parse(&rehashed_admission(before_lock))
            .unwrap()
            .check_target(&lock, REFERENCE_BYTES, &reference)
            .is_err()
    );
    let mut p = value(RETRY_SOME);
    p["predecessor"]["attempt_id"] = p["attempt_id"].clone();
    assert!(SyntheticAdmission::parse(&rehashed_admission(p)).is_err());
    let mut reference = synthetic_reference(REFERENCE_BYTES);
    reference.common.integrity.payload_sha256 = Hash("0".repeat(64));
    assert!(
        initial
            .check_target(&lock, REFERENCE_BYTES, &reference)
            .is_err()
    );
}
#[test]
fn d7_admission_production_rejects_rehashed_synthetic_markers() {
    for bytes in [INITIAL, RETRY_NONE, RETRY_SOME] {
        for (schema, synthetic) in [
            ("hekit-test-he-admission-v1", true),
            ("hekit-test-he-admission-v1", false),
            (ADMISSION, true),
        ] {
            let mut v = value(bytes);
            v["schema"] = json!(schema);
            v["synthetic"] = json!(synthetic);
            assert!(parse_admission(&rehashed_admission(v)).is_err());
        }
    }
    for bytes in [START_INITIAL, START_RETRY] {
        for (schema, synthetic) in [
            ("hekit-test-he-start-v1", true),
            ("hekit-test-he-start-v1", false),
            (START, true),
        ] {
            let mut v = value(bytes);
            v["schema"] = json!(schema);
            v["synthetic"] = json!(synthetic);
            assert!(parse_start(&rehashed_start(v)).is_err());
        }
    }
}
#[test]
fn d7_admission_nested_aliases_hex_tokens_and_unsupported_schema() {
    let raw = std::str::from_utf8(INITIAL).unwrap();
    for (key, alias) in [
        ("seed", "\\u0073eed"),
        ("implementation", "\\u0069mplementation"),
        ("inputs", "\\u0069nputs"),
        ("file_sha256", "\\u0066ile_sha256"),
        ("wall_source", "\\u0077all_source"),
    ] {
        // A duplicate decoded key is rejected by Fields before either supplied
        // raw value can be decoded; no substring parser is involved.
        let old = format!("\"{key}\":");
        let new = format!("\"{alias}\":null, \"{key}\":");
        let bad = raw.replacen(&old, &new, 1);
        assert_ne!(bad, raw);
        assert_eq!(
            SyntheticAdmission::parse(bad.as_bytes())
                .err()
                .unwrap()
                .code(),
            DocumentErrorCode::DuplicateField
        );
    }
    for pointer in [
        "/contract/canonical_bytes",
        "/selected_input/seed_receipt/canonical_bytes",
    ] {
        for text in ["0", "GG", "FF", "\\u0030"] {
            let mut v = value(INITIAL);
            *v.pointer_mut(pointer).unwrap() = json!(text);
            assert!(parse_value(&v).is_err());
        }
    }
    for pointer in [
        "/contract/schema",
        "/selected_input/seed_receipt/schema",
        "/selected_input/seed_receipt/encoding",
        "/provenance/value/schema",
        "/writer/api_version",
        "/writer/algorithm_revision",
        "/integrity/encoding",
    ] {
        let mut v = value(INITIAL);
        *v.pointer_mut(pointer).unwrap() = json!("unsupported-v999");
        assert!(parse_value(&v).is_err());
    }
    let mut bad = value(INITIAL);
    bad["integrity"]["payload_sha256"] = json!("0".repeat(64));
    assert_eq!(
        parse_value(&bad).err().unwrap().code(),
        DocumentErrorCode::DigestMismatch
    );
    let escaped = raw.replacen(
        "46494354494f4e414c2d524543454950542d534545443700fd",
        "\\u00346494354494f4e414c2d524543454950542d534545443700fd",
        1,
    );
    assert_ne!(escaped, raw);
    assert_eq!(
        SyntheticAdmission::parse(escaped.as_bytes())
            .err()
            .unwrap()
            .code(),
        DocumentErrorCode::BadHex
    );
    let mut malformed = INITIAL.to_vec();
    malformed.push(0xff);
    assert!(SyntheticAdmission::parse(&malformed).is_err());
}
#[test]
fn d7_admission_inherited_caps_and_small_oversize_witnesses() {
    for layout in [DocumentLayout::Admission, DocumentLayout::Start] {
        check_file_size(layout.file_cap(), layout).unwrap();
        assert!(check_file_size(layout.file_cap() + 1, layout).is_err());
        assert_eq!(
            bounded_add(layout.canonical_cap(), 0, layout.canonical_cap(), "").unwrap(),
            layout.canonical_cap()
        );
        assert!(bounded_add(layout.canonical_cap(), 1, layout.canonical_cap(), "").is_err());
    }
    check_outer_size(2 * C_MAX + 2 * R_MAX + A_MAX, 2 * C_MAX + 2 * R_MAX).unwrap();
    assert!(check_outer_size(2 * C_MAX + 2 * R_MAX + A_MAX + 1, 2 * C_MAX + 2 * R_MAX).is_err());
    check_outer_size(A_MAX, 0).unwrap();
    assert!(check_outer_size(A_MAX + 1, 0).is_err());
    assert!(check_outer_size(0, usize::MAX).is_err());
    for (pointer, n) in [
        ("/provenance/value/build/features", 65),
        ("/provenance/value/source/manifest", 513),
        ("/provenance/value/invocation/argv", 257),
        ("/provenance/value/thread_environment", 33),
    ] {
        let mut v = value(INITIAL);
        let item = v.pointer(pointer).unwrap()[0].clone();
        *v.pointer_mut(pointer).unwrap() = json!(vec![item; n]);
        assert!(parse_value(&v).is_err());
    }
    let mut v = value(INITIAL);
    v["provenance"]["value"]["build"]["compiler_vv"] = json!("a".repeat(16385));
    assert!(parse_value(&v).is_err());
    let long_key = format!("{{\"{}\":null}}", "a".repeat(65));
    assert!(preflight_layout(long_key.as_bytes(), DocumentLayout::Admission).is_err());
    let overlong_numeric = format!("{{\"seed\":{}}}", "1".repeat(65));
    assert!(preflight_layout(overlong_numeric.as_bytes(), DocumentLayout::Admission).is_err());
}

// Explicit synthetic representation harness only. Never constructs live owners.
fn linked_fixture(name: &str) -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/validation/task54-d7-full-linked-oracle/task-54-d7-full-linked-oracle").join(name)).unwrap()
}
#[test]
fn retained_c1_current_and_prior_share_one_verified_target_representation() {
    let lock =
        super::super::target::SyntheticTargetLock::parse(&linked_fixture("lock.json")).unwrap();
    let bytes = linked_fixture("reference.json");
    let reference = synthetic_reference(&bytes);
    let a = SyntheticAdmission::parse(&linked_fixture("A/admission.json"))
        .unwrap()
        .0;
    let b = SyntheticAdmission::parse(&linked_fixture("B/admission.json"))
        .unwrap()
        .0;
    let c = SyntheticAdmission::parse(&linked_fixture("C/admission.json"))
        .unwrap()
        .0;
    let start = SyntheticStart::parse(&linked_fixture("A/start.json"))
        .unwrap()
        .0;
    let lock = lock.claims_lock();
    check_target_history_claims(&a, None, lock, &bytes, &reference).unwrap();
    check_target_history_claims(&b, Some((&a, None)), lock, &bytes, &reference).unwrap();
    check_target_history_claims(&c, Some((&a, Some(&start))), lock, &bytes, &reference).unwrap();
    assert!(
        check_target_history_claims(&b, Some((&a, Some(&start))), lock, &bytes, &reference)
            .is_err()
    );
    assert!(check_target_history_claims(&c, Some((&a, None)), lock, &bytes, &reference).is_err());
    for (name, path) in [
        ("lock", "target.lock"),
        ("reference", "target.reference"),
        ("policy", "target.policy"),
    ] {
        let dir = format!("negative/c1-prior-{name}");
        // Parsing verifies both rehashed self-integrities before comparisons.
        let prior = SyntheticAdmission::parse(&linked_fixture(&format!("{dir}/A/admission.json")))
            .unwrap()
            .0;
        let current =
            SyntheticAdmission::parse(&linked_fixture(&format!("{dir}/B/admission.json")))
                .unwrap()
                .0;
        check_predecessor_claims(&current, &prior, None).unwrap();
        check_target_claims(&current, lock, &bytes, &reference).unwrap();
        let error =
            check_target_history_claims(&current, Some((&prior, None)), lock, &bytes, &reference)
                .unwrap_err();
        assert_eq!(error.path(), path, "{name}");
    }
}

#[test]
fn retained_c1_exact_file_payload_and_stale_self_are_distinct() {
    let initial = SyntheticAdmission::parse(&linked_fixture("A/admission.json"))
        .unwrap()
        .0;
    let whitespace =
        SyntheticAdmission::parse(&linked_fixture("negative/admission-whitespace.json"))
            .unwrap()
            .0;
    assert_eq!(
        initial.integrity.payload_sha256,
        whitespace.integrity.payload_sha256
    );
    assert_ne!(initial.file_sha256, whitespace.file_sha256);
    let retry = SyntheticAdmission::parse(&linked_fixture("B/admission.json"))
        .unwrap()
        .0;
    assert_eq!(
        check_predecessor_claims(&retry, &whitespace, None)
            .unwrap_err()
            .path(),
        "predecessor.admission"
    );
    let start = SyntheticStart::parse(&linked_fixture("negative/start-wrong-payload-link.json"))
        .unwrap()
        .0;
    assert_eq!(start.admission.file_sha256.0, retry.file_sha256);
    assert_eq!(
        check_start_claims(&start, &retry).unwrap_err().path(),
        "start.admission.link"
    );
    assert_eq!(
        SyntheticAdmission::parse(&linked_fixture("negative/admission-stale-self-hash.json"))
            .err()
            .unwrap()
            .code(),
        DocumentErrorCode::DigestMismatch
    );
}
