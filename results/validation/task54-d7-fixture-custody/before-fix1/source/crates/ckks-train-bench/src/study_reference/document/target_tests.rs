use super::fixture_literals::target as target_literal;
use super::target::*;
use super::*;
fn unhex(s: &str) -> Vec<u8> {
    s.trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
fn literal_lock() -> &'static [u8] {
    target_literal("lock.json").as_bytes()
}
#[test]
fn independent_literal_lock_canonical_and_file_identity() {
    let s = SyntheticTargetLock::parse(literal_lock()).unwrap();
    assert_eq!(
        s.canonical().unwrap(),
        unhex(target_literal("lock.canonical.hex"))
    );
    assert_eq!(
        sha256_hex(literal_lock()),
        "b71e6e056dcf7faeb8322a68b73b6189a12386e898d306549372a9f285ac26e7"
    );
    assert_eq!(
        s.payload_sha256(),
        "62dabb2e01360ed99e79590b869ca38616fee2bbb8ed7572c240257688c87812"
    );
    assert!(parse_lock(literal_lock()).is_err());
}
#[test]
fn binary64_separate_operations_exact_vectors() {
    let vectors = [
        (0, 1, 0x0000000000000000, 0x0000000000000000),
        (1, 201, 0x3f7460cbc7f5cf9a, 0x0000000000000000),
        (1, 200, 0x3f747ae147ae147b, 0x0000000000000000),
        (1, 199, 0x3f749539e3b2d067, 0x3efa589c04bbec00),
        (1, 3, 0x3fd5555555555555, 0x3fd50369d0369d03),
        (199, 200, 0x3fefd70a3d70a3d7, 0x3fefae147ae147ae),
        (45035996273704, 9007199254740992, 0x3f747ae147ae1400, 0),
        (
            45035996273705,
            9007199254740992,
            0x3f747ae147ae1480,
            0x3c54000000000000,
        ),
        (
            9007199254740991,
            9007199254740992,
            0x3fefffffffffffff,
            0x3fefd70a3d70a3d6,
        ),
        (
            9007199254740992,
            9007199254740992,
            0x3ff0000000000000,
            0x3fefd70a3d70a3d7,
        ),
    ];
    for (c, n, a, t) in vectors {
        assert_eq!(target_bits(c, n).unwrap(), (a, t));
    }
    for (c, n) in [(0, 0), (2, 1), (1, 9007199254740993), (u64::MAX, u64::MAX)] {
        assert!(target_bits(c, n).is_err());
    }
    assert_ne!(target_bits(1, 199).unwrap().1, 0x3efa589c04bbec09);
}
#[test]
fn auxiliary_ids_are_domain_separated_exact_canonical_bytes() {
    let p = auxiliary("hekit-study-profile-slot-v1", &["FICTIONAL-PROFILE"]).unwrap();
    let l = auxiliary(
        "hekit-study-target-lock-id-v1",
        &["FICTIONAL-CAMPAIGN", "FICTIONAL-PROFILE"],
    )
    .unwrap();
    assert_eq!(p, unhex(target_literal("profile-slot.canonical.hex")));
    assert_eq!(l, unhex(target_literal("lock-id.canonical.hex")));
    assert_eq!(
        sha256_hex(&l),
        "bc1c2add0a5e7deac620b2fe9606f1288d1db80d165bd85700b3864142d06aea"
    );
}
fn value() -> serde_json::Value {
    serde_json::from_slice(literal_lock()).unwrap()
}
fn parse_value(v: &serde_json::Value) -> Result<SyntheticTargetLock> {
    SyntheticTargetLock::parse(&serde_json::to_vec(v).unwrap())
}
#[test]
fn every_required_lock_field_and_unknown_decoded_duplicate_keys_reject() {
    let v = value();
    for key in v.as_object().unwrap().keys() {
        let mut bad = v.clone();
        bad.as_object_mut().unwrap().remove(key);
        assert!(parse_value(&bad).is_err(), "missing {key}");
    }
    for raw in [
        target_literal("duplicate-escaped-key.negative.json").as_bytes(),
        target_literal("unknown-field.negative.json").as_bytes(),
        target_literal("escaped-large-hex.negative.json").as_bytes(),
        target_literal("bits-value-disagree.negative.json").as_bytes(),
        target_literal("target-one-ulp-up.negative.json").as_bytes(),
        target_literal("unsupported-domain.negative.json").as_bytes(),
        target_literal("unsupported-hash.negative.json").as_bytes(),
        target_literal("wrong-identity-hash.negative.json").as_bytes(),
        target_literal("wrong-payload-hash.negative.json").as_bytes(),
        target_literal("mismatched-source-schema.negative.json").as_bytes(),
        target_literal("production-header-false.negative.json").as_bytes(),
        target_literal("production-header-true.negative.json").as_bytes(),
        target_literal("test-header-false.negative.json").as_bytes(),
    ] {
        assert!(SyntheticTargetLock::parse(raw).is_err());
        assert!(parse_lock(raw).is_err());
    }
}
#[test]
fn target_rejects_one_ulp_neighbors_negative_zero_nonfinite_and_f32() {
    use serde_json::json;
    for bits in [
        0x3fd50369d0369d02u64,
        0x3fd50369d0369d04,
        0x8000000000000000,
        (0.3283333333333333f64 as f32 as f64).to_bits(),
    ] {
        let mut v = value();
        v["target"] = json!({"value":f64::from_bits(bits),"bits":format!("{bits:016x}")});
        assert_eq!(
            parse_value(&v).err().unwrap().code(),
            DocumentErrorCode::ObservationMismatch
        );
    }
    for number in ["1e999", "-1e999", "NaN", "Infinity", "-0.0"] {
        let raw = format!("{{\"value\":{number},\"bits\":\"0000000000000000\"}}");
        if let Ok(raw) = serde_json::from_str::<&RawValue>(&raw) {
            assert!(Target::decode(raw, "target").is_err());
        }
    }
}
#[test]
fn target_layout_exempts_only_exact_contract_path_and_charges_all_outer_bytes() {
    let allowed = format!(
        "{{\"contract\":{{\"canonical_bytes\":\"{}\"}}}}",
        "00".repeat(C_MAX)
    );
    preflight_layout(allowed.as_bytes(), DocumentLayout::TargetLock).unwrap();
    let at_limit = format!(
        "{allowed}{}",
        " ".repeat(A_MAX - (allowed.len() - 2 * C_MAX))
    );
    assert_eq!(at_limit.len(), 7_324_204);
    preflight_layout(at_limit.as_bytes(), DocumentLayout::TargetLock).unwrap();
    assert!(
        preflight_layout(
            format!("{at_limit} ").as_bytes(),
            DocumentLayout::TargetLock
        )
        .is_err()
    );
    for raw in [
        format!(
            "{{\"contract\":{{\"canonical_bytes\":\"{}\"}}}}",
            "00".repeat(C_MAX + 1)
        ),
        format!(
            "{{\"identity\":{{\"contract\":{{\"canonical_bytes\":\"{}\"}}}}}}",
            "00".repeat(4097)
        ),
        format!("{{\"contract.canonical_bytes\":\"{}\"}}", "00".repeat(4097)),
        format!(
            "{{\"contract\":[{{\"canonical_bytes\":\"{}\"}}]}}",
            "00".repeat(4097)
        ),
        format!(
            "{{\"x\":{{\"contract\":{{\"canonical_bytes\":\"{}\"}}}}}}",
            "00".repeat(4097)
        ),
    ] {
        assert!(preflight_layout(raw.as_bytes(), DocumentLayout::TargetLock).is_err());
    }
    let outer = format!("{{}}{}", " ".repeat(A_MAX - 2));
    preflight_layout(outer.as_bytes(), DocumentLayout::TargetLock).unwrap();
    assert!(preflight_layout(format!("{outer} ").as_bytes(), DocumentLayout::TargetLock).is_err());
    let mut encoder = Encoder::new(DocumentLayout::TargetLock);
    encoder.append(&vec![0; C_MAX], true).unwrap();
    encoder.append(&vec![0; A_MAX], false).unwrap();
    assert_eq!(encoder.bytes.len(), 4_710_678);
    assert!(encoder.byte(0).is_err());
}
#[test]
fn maximal_individual_manifest_already_overflows_aggregate_outer_budget() {
    use serde_json::json;
    let mut v = value();
    v["provenance"]["value"]["source"]["manifest"] = json!(
        (0..512)
            .map(|i| json!({"path":format!("{i:03}{}","x".repeat(4093)),"sha256":"0".repeat(64)}))
            .collect::<Vec<_>>()
    );
    let raw = serde_json::to_vec(&v).unwrap();
    assert_eq!(
        preflight_layout(&raw, DocumentLayout::TargetLock)
            .unwrap_err()
            .code(),
        DocumentErrorCode::TooLarge
    );
}
#[test]
fn bounded_integer_list_depth_and_decoded_string_limits() {
    for n in ["-0", "1.0", "1e0", "18446744073709551616"] {
        let raw: &RawValue = serde_json::from_str(n).unwrap();
        assert!(u64::decode(raw, "count").is_err());
    }
    let sixteen = format!("{}0{}", "[".repeat(16), "]".repeat(16));
    preflight_layout(sixteen.as_bytes(), DocumentLayout::TargetLock).unwrap();
    assert!(
        preflight_layout(
            format!("[{sixteen}]").as_bytes(),
            DocumentLayout::TargetLock
        )
        .is_err()
    );
    let mut v = value();
    v["reference"]["completion"]["final_endpoint"]["crossed_logical_boundaries"] =
        serde_json::json!(vec![1u64; 401]);
    assert!(parse_value(&v).is_err());
    let long = format!("{{\"x\":\"{}\"}}", "\\u0061".repeat(4096));
    preflight_layout(long.as_bytes(), DocumentLayout::TargetLock).unwrap();
    assert!(
        preflight_layout(
            long.replace("\"}", "a\"}").as_bytes(),
            DocumentLayout::TargetLock
        )
        .is_err()
    );
    for (field, n) in [("features", 65), ("argv", 257), ("thread_environment", 33)] {
        let mut v = value();
        let old = match field {
            "features" => v["provenance"]["value"]["build"][field][0].clone(),
            "argv" => v["provenance"]["value"]["invocation"][field][0].clone(),
            _ => v["provenance"]["value"][field][0].clone(),
        };
        let repeated = serde_json::json!(vec![old; n]);
        match field {
            "features" => v["provenance"]["value"]["build"][field] = repeated,
            "argv" => v["provenance"]["value"]["invocation"][field] = repeated,
            _ => v["provenance"]["value"][field] = repeated,
        }
        assert!(parse_value(&v).is_err());
    }
}

#[test]
fn exact_rational_substitution_fails_observation_before_payload_integrity() {
    let mut v = value();
    let accuracy = serde_json::json!({"correct":1,"test_count":199,"value":f64::from_bits(0x3f749539e3b2d067),"bits":"3f749539e3b2d067"});
    v["reference"]["final_observation"] = accuracy.clone();
    v["reference"]["completion"]["final_endpoint"]["observation"] = accuracy;
    v["target"] =
        serde_json::json!({"value":f64::from_bits(0x3efa589c04bbec09),"bits":"3efa589c04bbec09"});
    assert_eq!(
        parse_value(&v).err().unwrap().code(),
        DocumentErrorCode::ObservationMismatch
    );
}
