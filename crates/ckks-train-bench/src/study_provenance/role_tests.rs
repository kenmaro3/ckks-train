//! Frozen independent bytes; no approved context is constructed.
use super::{codec::*, role::Recipe};
use crate::data::sha256_hex;
use serde_json::Value;

pub(super) fn literal(name: &str) -> Vec<u8> {
    let bundle: Value = serde_json::from_str(include_str!("fixtures/roles/literals.json")).unwrap();
    let text = bundle[name]["text"].as_str().unwrap();
    text.split_whitespace()
        .collect::<String>()
        .as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
pub(super) fn index() -> Value {
    serde_json::from_str(include_str!("fixtures/roles/index.json")).unwrap()
}
fn decode(name: &str, kind: &str, bytes: &[u8]) -> super::Result<Vec<u8>> {
    match kind {
        "capsule" => Capsule::decode(bytes)?.encode(),
        "receipt" => Receipt::decode(bytes)?.encode(),
        "approval_v1" => Approval::decode(bytes)?.encode(),
        "approval_v2" => ApprovalV2::decode(bytes)?.encode(),
        "files" => Manifest::decode(bytes)?.encode(),
        "environment" => Environment::decode(bytes)?.encode(),
        "historical_transport" => decode(
            name,
            if name.ends_with("capsule") {
                "capsule"
            } else if name.ends_with("approval") {
                "approval_v1"
            } else {
                "receipt"
            },
            bytes,
        ),
        _ => panic!("unmapped literal {name}"),
    }
}
#[test]
fn role_literal_hashes_roundtrips_and_all_29_parser_negatives() {
    let bundle: Value = serde_json::from_str(include_str!("fixtures/roles/literals.json")).unwrap();
    let index = index();
    let mut rejected = 0;
    assert_eq!(index.as_object().unwrap().len(), 141);
    for (name, item) in index.as_object().unwrap() {
        let bytes = literal(name);
        assert_eq!(
            sha256_hex(bundle[name]["text"].as_str().unwrap().as_bytes()),
            bundle[name]["file_sha256"]
        );
        assert_eq!(sha256_hex(&bytes), item["sha256"]);
        assert_eq!(bytes.len() as u64, item["length"].as_u64().unwrap());
        let expected = item["expected"].as_str().unwrap();
        if expected == "parser_reject" {
            assert!(
                decode(name, item["kind"].as_str().unwrap(), &bytes).is_err(),
                "{name}"
            );
            rejected += 1;
        } else if !expected.contains("obligation") {
            assert_eq!(
                decode(name, item["kind"].as_str().unwrap(), &bytes).unwrap(),
                bytes,
                "{name}"
            );
        }
    }
    assert_eq!(rejected, 29);
}
#[test]
fn role_four_closed_recipes_keep_reference_entrypoint_closed() {
    for name in ["reference", "cpu_serial", "cpu_rayon", "cuda_host"] {
        let c = Capsule::decode(&literal(&format!("{name}_capsule"))).unwrap();
        Recipe::capsule_policy(&c).unwrap();
        let r = Receipt::decode(&literal(&format!("{name}_receipt"))).unwrap();
        assert_eq!(r.closed_policy().unwrap(), c);
        assert_eq!(c.policy().is_ok(), name == "reference");
        assert_eq!(r.policy().is_ok(), name == "reference");
    }
}
#[test]
fn role_exact_five_approval_memberships() {
    for (group, names) in [
        ("reference_only", vec!["reference"]),
        ("reference_cpu_serial", vec!["reference", "cpu_serial"]),
        ("reference_cpu_rayon", vec!["reference", "cpu_rayon"]),
        ("reference_cuda_host", vec!["reference", "cuda_host"]),
        (
            "multiple_builds",
            vec![
                "reference",
                "reference_second",
                "cpu_serial",
                "cpu_rayon",
                "cuda_host",
            ],
        ),
    ] {
        let a = ApprovalV2::decode(&literal(&format!("approval_v2_{group}"))).unwrap();
        let mut members = Vec::new();
        for name in names {
            let bytes = literal(&format!("{name}_receipt"));
            let r = Receipt::decode(&bytes).unwrap();
            let c = Capsule::decode(&r.capsule).unwrap();
            let recipe = Recipe::capsule_policy(&c).unwrap();
            members.push(ApprovalBuild {
                role: u8::from(recipe != Recipe::Reference),
                recipe: c.recipe,
                algorithm: c.algorithm,
                receipt: sha256_hex(&bytes),
                capsule: sha256_hex(&r.capsule),
                executable: r.executable_hash,
            });
        }
        members.sort_by(|a, b| a.receipt.cmp(&b.receipt));
        assert_eq!(a.builds, members, "{group}");
    }
}
