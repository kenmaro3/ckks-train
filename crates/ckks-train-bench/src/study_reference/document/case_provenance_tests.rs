use super::*;
use serde_json::{Value, json};
fn fixture(name: &str) -> Vec<u8> {
    super::fixture_literals::case(name).as_bytes().to_vec()
}
fn component(input: &[u8]) -> Result<Vec<u8>> {
    // Reuse the existing exact-path transport, without inventing an artifact.
    let mut wrapper = b"{\"provenance\":".to_vec();
    wrapper.extend_from_slice(input);
    wrapper.push(b'}');
    preflight(&wrapper)?;
    let raw = serde_json::from_slice::<&RawValue>(&wrapper).map_err(|e| json_err(e, ""))?;
    let mut fields = Fields::new(raw, &["provenance"], "")?;
    let claim =
        case_provenance::ParsedCaseProvenance::decode(fields.get("provenance", "")?, "provenance")?;
    let mut encoder = Encoder::new(DocumentLayout::Reference);
    claim.encode(&mut encoder)?;
    Ok(encoder.bytes)
}
fn decode_hex(input: &[u8]) -> Vec<u8> {
    let text = std::str::from_utf8(input).unwrap().trim();
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}
#[test]
fn case_provenance_all_six_independent_components_exact() {
    let manifest: Value = serde_json::from_slice(&fixture("manifest.json")).unwrap();
    assert_eq!(manifest["positive"].as_object().unwrap().len(), 6);
    for (name, expected) in manifest["positive"].as_object().unwrap() {
        let actual =
            component(&fixture(&format!("{name}.json"))).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            actual,
            decode_hex(&fixture(&format!("{name}.canonical.hex"))),
            "{name}"
        );
        assert_eq!(sha256_hex(&actual), expected["canonical_sha256"], "{name}");
        assert_eq!(actual[39], 1);
    }
    assert_eq!(
        &component(&fixture("threads-u64-max.json")).unwrap()[1226..1234],
        &[255; 8]
    );
    let one = component(&fixture("threads-1.json")).unwrap();
    let two = component(&fixture("threads-2.json")).unwrap();
    assert_eq!(
        one.iter()
            .zip(&two)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        [1226]
    );
}
#[test]
fn case_provenance_all_67_actual_negative_literals_reject() {
    let manifest: Value = serde_json::from_slice(&fixture("manifest.json")).unwrap();
    assert_eq!(manifest["negative"].as_object().unwrap().len(), 67);
    for (name, _) in manifest["negative"].as_object().unwrap() {
        assert!(
            component(&fixture(&format!("negative/{name}.json"))).is_err(),
            "accepted {name}"
        );
    }
}
#[test]
fn case_provenance_member_order_does_not_choose_canonical_order() {
    let input: Value = serde_json::from_slice(&fixture("threads-2.json")).unwrap();
    // serde_json's sorted object order differs from the literal's published order.
    assert_eq!(
        component(&serde_json::to_vec(&input).unwrap()).unwrap(),
        decode_hex(&fixture("threads-2.canonical.hex"))
    );
}
#[test]
fn case_provenance_reference_stays_scalar_and_unavailable() {
    let mut v: Value = serde_json::from_slice(&fixture("threads-1.json")).unwrap();
    v["value"].as_object_mut().unwrap().remove("role");
    v["value"]["schema"] = json!("hekit-study-provenance-v1");
    let bytes = serde_json::to_vec(&v).unwrap();
    let raw = serde_json::from_slice::<&RawValue>(&bytes).unwrap();
    let mut claim = ProvenanceClaim::decode(raw, "provenance").unwrap();
    validate_provenance(&mut claim, true).unwrap();
    let mut e = Encoder::new(DocumentLayout::Reference);
    claim.encode(&mut e).unwrap();
    assert_eq!(
        e.bytes,
        decode_hex(&fixture("reference-preserved.canonical.hex"))
    );
    v["value"]["host"]["execution_threads"] = json!(2);
    let b = serde_json::to_vec(&v).unwrap();
    let mut claim = ProvenanceClaim::decode(
        serde_json::from_slice::<&RawValue>(&b).unwrap(),
        "provenance",
    )
    .unwrap();
    assert!(validate_provenance(&mut claim, true).is_err());
    let raw =
        serde_json::from_str::<&RawValue>(r#"{"status":"unavailable","reason":"unset"}"#).unwrap();
    let mut absent = ProvenanceClaim::decode(raw, "provenance").unwrap();
    validate_provenance(&mut absent, true).unwrap();
}
#[test]
fn case_provenance_preflight_exact_paths_and_aggregate_budget() {
    let mut v: Value = serde_json::from_slice(&fixture("threads-2.json")).unwrap();
    v["value"]["build"]["compiler_vv"] = json!("x".repeat(16384));
    assert!(component(&serde_json::to_vec(&v).unwrap()).is_ok());
    let wrong = json!({"elsewhere":v});
    assert!(preflight(&serde_json::to_vec(&wrong).unwrap()).is_err());
    v["value"]["invocation"]["argv"] = json!([{"encoding":"unix_bytes","bytes":"ab".repeat(4096)}]);
    assert!(component(&serde_json::to_vec(&v).unwrap()).is_ok());
    let mut wrapper = b"{\"provenance\":".to_vec();
    wrapper.extend(serde_json::to_vec(&v).unwrap());
    wrapper.extend(vec![b' '; A_MAX]);
    wrapper.push(b'}');
    assert_eq!(
        preflight(&wrapper).unwrap_err().code(),
        DocumentErrorCode::TooLarge
    );
    assert!(
        preflight(br#"{"x":12345678901234567890123456789012345678901234567890123456789012345}"#)
            .is_err()
    );
    let deep = format!("{}0{}", "[".repeat(17), "]".repeat(17));
    assert!(preflight(deep.as_bytes()).is_err());
    assert!(
        preflight(
            serde_json::to_string(&json!({"x".repeat(65):0}))
                .unwrap()
                .as_bytes()
        )
        .is_err()
    );
}

#[test]
fn case_provenance_canonical_bytes_cannot_escape_outer_budget() {
    let mut v: Value = serde_json::from_slice(&fixture("threads-2.json")).unwrap();
    let hash = "a".repeat(64);
    v["value"]["source"]["manifest"] = Value::Array(
        (0..512)
            .map(|n| json!({"path":format!("f{n:03}{}","x".repeat(4092)),"sha256":hash}))
            .collect(),
    );
    // All individual fields fit, but they cannot fit the shared auxiliary budget.
    let bytes = serde_json::to_vec(&v).unwrap();
    let raw = serde_json::from_slice::<&RawValue>(&bytes).unwrap();
    let claim = case_provenance::ParsedCaseProvenance::decode(raw, "provenance").unwrap();
    let mut encoder = Encoder::new(DocumentLayout::Reference);
    assert_eq!(
        claim.encode(&mut encoder).unwrap_err().code(),
        DocumentErrorCode::TooLarge
    );
    assert!(encoder.bytes.len() <= A_MAX);
}

#[test]
fn case_provenance_producer_codec_preserves_independent_components_synthetic() {
    // Synthetic JSON fixtures exercise the production codec only, never capture authority.
    let manifest: Value = serde_json::from_slice(&fixture("manifest.json")).unwrap();
    for (name, expected) in manifest["positive"].as_object().unwrap() {
        let value: Value = serde_json::from_slice(&fixture(&format!("{name}.json"))).unwrap();
        let actual = case_provenance::encode_captured_value(value).unwrap();
        assert_eq!(
            actual.canonical_bytes(),
            decode_hex(&fixture(&format!("{name}.canonical.hex")))
        );
        assert_eq!(actual.sha256(), expected["canonical_sha256"]);
    }
    for name in manifest["negative"].as_object().unwrap().keys() {
        let bytes = fixture(&format!("negative/{name}.json"));
        // Several transport negatives intentionally cannot be represented by Value.
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            if component(&serde_json::to_vec(&value).unwrap()).is_err() {
                assert!(
                    case_provenance::encode_captured_value(value).is_err(),
                    "{name}"
                );
            }
        }
    }
}
