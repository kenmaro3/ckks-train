use super::super::fixture_literals::target as target_literal;
use super::*;
fn literal_lock() -> &'static [u8] {
    target_literal("lock.json").as_bytes()
}
fn literal_reference_bytes() -> &'static [u8] {
    target_literal("reference.json").as_bytes()
}
struct SyntheticReference {
    reference: ReferenceLink,
    contract: Vec<u8>,
}
fn synthetic_reference() -> SyntheticReference {
    let v: Value = serde_json::from_slice(literal_reference_bytes()).unwrap();
    let field = |key: &str| serde_json::value::to_raw_value(&v[key]).unwrap();
    let identity = ClaimedIdentity::decode(&field("identity"), "identity").unwrap();
    let completion = Completion::decode(&field("completion"), "completion").unwrap();
    let endpoints = List::<Endpoint, 400>::decode(&field("endpoints"), "endpoints").unwrap();
    let time = Timing::decode(&field("timing"), "timing").unwrap();
    SyntheticReference {
        contract: identity.contract.bytes,
        reference: ReferenceLink {
            run_id: Text::decode(&field("run_id"), "run_id").unwrap(),
            schema: Text::decode(&field("schema"), "schema").unwrap(),
            kind: Kind::decode(&field("kind"), "kind").unwrap(),
            link: Link {
                file_sha256: Hash(sha256_hex(literal_reference_bytes())),
                payload_sha256: Hash(
                    "26c0d842f9e02f547e8fb6678f5d7550aa9326d05c91f568ba157bf652e1762d".into(),
                ),
            },
            seed: identity.seed,
            seed_receipt_sha256: identity.seed_receipt.sha256,
            materialized_sha256: identity.materialized_sha256,
            writer_capsule_sha256: Hash(
                "5f592e4f72425d38e4b8368bfafc08796bffe9af689280a186b00c8cefcf0ebe".into(),
            ),
            start_unix_ns: time.start_unix_ns,
            completion_unix_ns: time.end_unix_ns,
            completion: CompletionProjection {
                origin_completed_updates: completion.origin_completed_updates,
                origin_processed_samples: completion.origin_processed_samples,
                planned_updates: completion.planned_updates,
                actual_updates: completion.actual_updates,
                planned_samples: completion.planned_samples,
                actual_samples: completion.actual_samples,
                planned_dropped_samples: completion.planned_dropped_samples,
                actual_dropped_samples: completion.actual_dropped_samples,
                full_plan_sha256: completion.full_plan_sha256,
                consumed_plan_sha256: completion.consumed_plan_sha256,
                final_endpoint_index: completion.final_endpoint_index,
                final_endpoint: endpoints.0.last().unwrap().clone(),
                final_state_sha256: completion.final_state_sha256,
            },
            final_observation: ClaimedAccuracy::decode(
                &field("final_observation"),
                "final_observation",
            )
            .unwrap(),
        },
    }
}
#[test]
fn literal_synthetic_reference_links_every_projection_field() {
    let source = synthetic_reference();
    let lock = SyntheticTargetLock::parse(literal_lock()).unwrap();
    source_matches(&lock.0, &source.reference, &source.contract).unwrap();
    let raw: Value = serde_json::from_slice(literal_lock()).unwrap();
    fn leaf_paths(v: &Value, p: String, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    leaf_paths(v, format!("{p}/{k}"), out)
                }
            }
            Value::Array(a) => {
                for (i, v) in a.iter().enumerate() {
                    leaf_paths(v, format!("{p}/{i}"), out)
                }
            }
            _ => out.push(p),
        }
    }
    let mut leaves = Vec::new();
    leaf_paths(&raw["reference"], "/reference".into(), &mut leaves);
    for path in leaves {
        let mut changed = raw.clone();
        let leaf = changed.pointer_mut(&path).unwrap();
        *leaf = match leaf {
            Value::String(s) => json!(if s.len() == 64 {
                "0".repeat(64)
            } else {
                format!("{s}x")
            }),
            Value::Bool(b) => json!(!*b),
            Value::Number(n) => json!(n.as_u64().unwrap_or(1).saturating_add(1)),
            _ => continue,
        };
        let bytes = serde_json::to_vec(&changed).unwrap();
        // Schema failures reject before linkage. If typed decode succeeds, full source equality must reject even with rehashed claims.
        if let Ok(mut bad) = decode_lock(&bytes) {
            bad.integrity.payload_sha256 = Hash(sha256_hex(&bad.canonical().unwrap()));
            assert!(
                source_matches(&bad, &source.reference, &source.contract).is_err(),
                "{path}"
            );
        }
    }
    for i in 0..source.contract.len() {
        let mut bad = decode_lock(literal_lock()).unwrap();
        bad.contract.bytes[i] ^= 255;
        bad.contract.sha256 = Hash(sha256_hex(&bad.contract.bytes));
        bad.integrity.payload_sha256 = Hash(sha256_hex(&bad.canonical().unwrap()));
        assert!(source_matches(&bad, &source.reference, &source.contract).is_err());
    }
}
#[test]
fn digest_confusion_whitespace_source_clock_and_endpoint_substitutions_reject() {
    let source = synthetic_reference();
    for which in 0..8 {
        let mut bad = decode_lock(literal_lock()).unwrap();
        match which {
            0 => std::mem::swap(
                &mut bad.reference.link.file_sha256,
                &mut bad.reference.link.payload_sha256,
            ),
            1 => {
                bad.reference.link.file_sha256 =
                    Hash(sha256_hex(&[literal_reference_bytes(), b" "].concat()))
            }
            2 => bad.reference.seed = 7,
            3 => bad.reference.completion.final_endpoint_index += 1,
            4 => bad
                .reference
                .completion
                .final_endpoint
                .crossed_logical_boundaries
                .0
                .reverse(),
            5 => {
                bad.reference.completion.final_endpoint.update_elapsed_ns =
                    Observation::Available(999)
            }
            6 => bad.reference.start_unix_ns += 1,
            7 => {
                bad.reference
                    .completion
                    .final_endpoint
                    .exact_logical_boundary = false
            }
            _ => unreachable!(),
        }
        bad.integrity.payload_sha256 = Hash(sha256_hex(&bad.canonical().unwrap()));
        assert!(source_matches(&bad, &source.reference, &source.contract).is_err());
    }
}
#[test]
fn every_policy_binding_rejects_rehashed_substitution() {
    let baseline = decode_lock(literal_lock()).unwrap();
    let mut values = Vec::new();
    let v: Value = serde_json::from_slice(literal_lock()).unwrap();
    for field in [
        "scientific_revision",
        "scientific_sha256",
        "registry_sha256",
        "portable_lock_sha256",
        "approval_sha256",
    ] {
        let mut bad = v.clone();
        let size = bad["policy"][field].as_str().unwrap().len();
        bad["policy"][field] = json!("0".repeat(size));
        values.push(bad);
    }
    for index in 0..5 {
        for field in ["path", "sha256"] {
            let mut bad = v.clone();
            bad["policy"]["design_documents"][index][field] = json!(if field == "path" {
                "docs/other.md".to_string()
            } else {
                "0".repeat(64)
            });
            values.push(bad);
        }
    }
    for value in values {
        let mut bad = decode_lock(&serde_json::to_vec(&value).unwrap()).unwrap();
        bad.integrity.payload_sha256 = Hash(sha256_hex(&bad.canonical().unwrap()));
        assert!(policy_matches(&bad.policy, &baseline.policy).is_err());
    }
}
#[test]
fn independent_literal_reference_uses_unchanged_inherited_record_encoder_only() {
    // Private syntax-only synthetic type. No VerifiedReference is constructed.
    struct SyntheticReferenceDocument {
        common: Common,
        body: ReferenceBody,
    }
    let v: Value = serde_json::from_slice(literal_reference_bytes()).unwrap();
    let field = |key: &str| serde_json::value::to_raw_value(&v[key]).unwrap();
    let mut reference = SyntheticReferenceDocument {
        common: Common {
            schema: Text::decode(&field("schema"), "schema").unwrap(),
            kind: Kind::decode(&field("kind"), "kind").unwrap(),
            status: Status::decode(&field("status"), "status").unwrap(),
            run_id: Text::decode(&field("run_id"), "run_id").unwrap(),
            writer: Writer::decode(&field("writer"), "writer").unwrap(),
            synthetic: bool::decode(&field("synthetic"), "synthetic").unwrap(),
            arithmetic: Arithmetic::decode(&field("arithmetic"), "arithmetic").unwrap(),
            provenance: ProvenanceClaim::decode(&field("provenance"), "provenance").unwrap(),
            integrity: Integrity::decode(&field("integrity"), "integrity").unwrap(),
            file_sha256: sha256_hex(literal_reference_bytes()),
        },
        body: ReferenceBody {
            identity: ClaimedIdentity::decode(&field("identity"), "identity").unwrap(),
            completion: Completion::decode(&field("completion"), "completion").unwrap(),
            endpoints: List::decode(&field("endpoints"), "endpoints").unwrap(),
            final_observation: ClaimedAccuracy::decode(
                &field("final_observation"),
                "final_observation",
            )
            .unwrap(),
            timing: Timing::decode(&field("timing"), "timing").unwrap(),
        },
    };
    assert!(reference.common.synthetic);
    assert_eq!(
        reference.common.schema.0,
        "hekit-test-reference-document-v1"
    );
    provenance(&mut reference.common, true).unwrap();
    let canonical = reference_encoding(&reference.common, &reference.body)
        .unwrap()
        .bytes;
    let expected = target_literal("reference.canonical.hex")
        .trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(canonical, expected);
    assert_eq!(
        sha256_hex(&canonical),
        "26c0d842f9e02f547e8fb6678f5d7550aa9326d05c91f568ba157bf652e1762d"
    );
    assert!(parse_reference_document(literal_reference_bytes()).is_err());
}
