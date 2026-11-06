use super::*;
#[test]
fn no_canonical_identity_comparison_is_hash_only() {
    fn blob(bytes: &[u8], version: &str) -> ClaimedIdentityBytes {
        ClaimedIdentityBytes {
            schema: Text(version.into()),
            encoding: Text(ENCODING_VERSION.into()),
            bytes: bytes.to_vec(),
            sha256: Hash(sha256_hex(bytes)),
        }
    }
    let c = b"SYNTHETIC full contract tags floats graph";
    let r = b"SYNTHETIC full receipt order params links";
    let base = ClaimedIdentity {
        profile_id: Text("SYNTHETIC".into()),
        seed: 0,
        materialized_sha256: Hash("1".repeat(64)),
        contract: blob(c, CONTRACT_VERSION),
        seed_receipt: blob(r, RECEIPT_VERSION),
    };
    identity_matches(&base, "SYNTHETIC", 0, &"1".repeat(64), c, r).unwrap();
    for which in 0..2 {
        let len = if which == 0 { c.len() } else { r.len() };
        for i in 0..=len {
            let mut bad = base.clone();
            let v = if which == 0 {
                &mut bad.contract
            } else {
                &mut bad.seed_receipt
            };
            if i == len {
                v.bytes.push(0)
            } else {
                v.bytes[i] ^= 255;
            }
            v.sha256 = Hash(sha256_hex(&v.bytes));
            assert!(identity_matches(&bad, "SYNTHETIC", 0, &"1".repeat(64), c, r).is_err());
        }
    }
}
// Adversarial parse fixtures only. They never enter a Store or capability constructor.
mod golden {
    pub const REFERENCE: &str = include_str!("../session_goldens/adversarial_reference.json");
    pub const CAPTURED_REFERENCE: &str =
        include_str!("../session_goldens/adversarial_captured_reference.json");
}
use serde_json::{Value, json};
fn claim() -> Value {
    serde_json::from_str(golden::REFERENCE).unwrap()
}
fn rehash(mut value: Value) -> Vec<u8> {
    for field in ["contract", "seed_receipt"] {
        let h = value["identity"][field]["canonical_bytes"]
            .as_str()
            .unwrap();
        let b: Vec<_> = h
            .as_bytes()
            .chunks_exact(2)
            .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
            .collect();
        value["identity"][field]["sha256"] = json!(sha256_hex(&b));
    }
    let initial = serde_json::to_vec(&value).unwrap();
    let raw: &RawValue = serde_json::from_slice(&initial).unwrap();
    let mut fields = Fields::new(
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
    let mut c = common(&mut fields, String::new(), REFERENCE).unwrap();
    provenance(&mut c, true).unwrap();
    let b = ReferenceBody {
        identity: fields.field("identity", "").unwrap(),
        completion: fields.field("completion", "").unwrap(),
        endpoints: fields.field("endpoints", "").unwrap(),
        final_observation: fields.field("final_observation", "").unwrap(),
        timing: fields.field("timing", "").unwrap(),
    };
    value["integrity"]["payload_sha256"] =
        json!(sha256_hex(&reference_encoding(&c, &b).unwrap().bytes));
    serde_json::to_vec(&value).unwrap()
}
fn with_plan(v: &mut Value, plan: &BatchPlan, denominator: u64) {
    let p = &mut v["completion"];
    p["planned_updates"] = json!(plan.updates.len());
    p["actual_updates"] = json!(plan.updates.len());
    p["planned_samples"] = json!(plan.processed_samples);
    p["actual_samples"] = json!(plan.processed_samples);
    p["planned_dropped_samples"] = json!(plan.dropped_samples);
    p["actual_dropped_samples"] = json!(plan.dropped_samples);
    p["final_endpoint_index"] = json!(plan.checkpoints.len() - 1);
    let a = json!({"correct":0,"test_count":denominator,"value":0.0,"bits":"0000000000000000"});
    v["endpoints"]=json!(plan.checkpoints.iter().map(|c|json!({"completed_updates":c.completed_updates,"processed_samples":c.processed_samples,"crossed_logical_boundaries":c.crossed_logical_boundaries,"exact_logical_boundary":c.exact_logical_boundary,"interval_updates":c.interval_updates,"interval_samples":c.interval_samples,"observation":a,"update_elapsed_ns":{"status":"available","value":0},"evaluation_elapsed_ns":{"status":"available","value":0}})).collect::<Vec<_>>());
    v["final_observation"] = a;
}
#[test]
fn canonical_iris_identity_regions_rehashed_do_not_become_trusted() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(root).unwrap();
    let p = materialize(&registry.resolve("iris__mlp2020", 0).unwrap()).unwrap();
    let (c, r) = derive_identities(&p).unwrap();
    let mut v = claim();
    v["identity"]["profile_id"] = json!(p.metadata().id);
    v["identity"]["seed"] = json!(0);
    v["identity"]["materialized_sha256"] = json!(p.identity().digest());
    v["identity"]["contract"]["canonical_bytes"] = json!(producer::hex(c.canonical_bytes()));
    v["identity"]["seed_receipt"]["canonical_bytes"] = json!(producer::hex(r.canonical_bytes()));
    let check_claim = |v: Value| {
        let parsed = parse_reference_document(&rehash(v)).unwrap();
        identity_matches(
            &parsed.body.identity,
            &p.metadata().id,
            0,
            p.identity().digest(),
            c.canonical_bytes(),
            r.canonical_bytes(),
        )
    };
    check_claim(v.clone()).unwrap(); // Only the identity comparison component, never verification/completion.
    for (field, bytes) in [
        ("contract", c.canonical_bytes()),
        ("seed_receipt", r.canonical_bytes()),
    ] {
        let mut cuts = vec![
            0,
            8,
            40,
            bytes.len() / 4,
            bytes.len() / 2,
            bytes.len() * 3 / 4,
            bytes.len() - 1,
        ];
        // Named text regions: policy/profile, selector and source paths. No inner decoder.
        for needle in [
            b"study-plain".as_slice(),
            b"iris",
            b"sha256",
            b"registry",
            b"chacha",
            b"scripts/",
            b"crates/",
            b"forward",
            b"weights",
        ] {
            if let Some(offset) = bytes.windows(needle.len()).position(|w| w == needle) {
                cuts.push(offset);
            }
        }
        for offset in cuts {
            let mut mutated = bytes.to_vec();
            mutated[offset] ^= 255;
            let mut bad = v.clone();
            bad["identity"][field]["canonical_bytes"] = json!(producer::hex(&mutated));
            assert!(check_claim(bad).is_err(), "{field}/{offset}");
        }
        for replacement in [
            f32::NAN.to_bits().to_le_bytes().to_vec(),
            f64::INFINITY.to_bits().to_le_bytes().to_vec(),
        ] {
            let mut mutated = bytes.to_vec();
            let offset = mutated.len() - replacement.len();
            mutated[offset..].copy_from_slice(&replacement);
            let mut bad = v.clone();
            bad["identity"][field]["canonical_bytes"] = json!(producer::hex(&mutated));
            assert!(check_claim(bad).is_err());
        }
        for extra in [
            vec![0xff],
            f32::NAN.to_bits().to_le_bytes().to_vec(),
            f64::INFINITY.to_bits().to_le_bytes().to_vec(),
        ] {
            let mut bad = v.clone();
            let mut mutated = bytes.to_vec();
            mutated.extend(extra);
            bad["identity"][field]["canonical_bytes"] = json!(producer::hex(&mutated));
            assert!(check_claim(bad).is_err());
        }
    }
    for (key, value) in [
        ("seed", json!(7)),
        ("profile_id", json!("iris__emlp1")),
        ("profile_id", json!("mnist_released__emlp1__b128_candidate")),
    ] {
        let mut bad = v.clone();
        bad["identity"][key] = value;
        assert!(check_claim(bad).is_err());
    }
}
#[test]
fn full_plan_denominator_endpoint_and_b128_boundary_forgery_reject() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(root).unwrap();
    let resolved = registry
        .resolve("mnist_released__emlp1__b128_candidate", 0)
        .unwrap();
    let trusted = resolved.preflight(None).unwrap().plan;
    assert_eq!(trusted.updates.len(), 375);
    assert_eq!(trusted.checkpoints[0].interval_updates, 38);
    assert_eq!(trusted.checkpoints[1].interval_updates, 37);
    assert_eq!(trusted.checkpoints[0].interval_samples, 4864);
    assert_eq!(trusted.checkpoints[1].interval_samples, 4736);
    let mut base = claim();
    with_plan(&mut base, &trusted, 1200);
    let full_hash = "0".repeat(64);
    let accepts = |v: Value| {
        parse_reference_document(&rehash(v))
            .and_then(|p| science(&p.body, &trusted, &full_hash, 1200))
    };
    accepts(base.clone()).unwrap();
    let mut shortened = trusted.clone();
    shortened
        .updates
        .truncate(trusted.checkpoints[0].completed_updates);
    shortened.checkpoints.truncate(1);
    shortened.processed_samples = 4864;
    let mut bad = base.clone();
    with_plan(&mut bad, &shortened, 1200);
    assert!(accepts(bad).is_err());
    for mutation in 0..9 {
        let mut bad = base.clone();
        match mutation {
            0 => {
                bad["completion"]["full_plan_sha256"] = json!("f".repeat(64));
                bad["completion"]["consumed_plan_sha256"] = json!("f".repeat(64));
            }
            1 => {
                with_plan(&mut bad, &trusted, 1201);
            }
            2 => {
                bad["endpoints"].as_array_mut().unwrap().remove(0);
            }
            3 => {
                let x = bad["endpoints"][0].clone();
                bad["endpoints"].as_array_mut().unwrap().insert(1, x);
            }
            4 => bad["endpoints"][0]["exact_logical_boundary"] = json!(true),
            5 => bad["endpoints"][0]["interval_updates"] = json!(37),
            6 => bad["endpoints"][0]["processed_samples"] = json!(4800),
            7 => bad["endpoints"][1]["interval_samples"] = json!(4864),
            _ => bad["completion"]["actual_dropped_samples"] = json!(700),
        }
        assert!(accepts(bad).is_err(), "mutation {mutation}");
    }
}
#[test]
fn unknown_writer_policy_and_unavailable_required_observations_reject() {
    let mut v: Value = serde_json::from_str(golden::CAPTURED_REFERENCE).unwrap();
    v["writer"]["api_version"] = json!(API);
    v["writer"]["algorithm_revision"] = json!(ALGORITHM);
    v["arithmetic"]["reduction"] = json!(REDUCTION);
    v["arithmetic"]["classification"] = json!(CLASSIFICATION);
    v["timing"]["finalization_ns"] = json!({"status":"available","value":0});
    let parsed = parse_reference_document(&rehash(v.clone())).unwrap();
    required_policy(&parsed.common, &parsed.body.timing).unwrap();
    for mutation in 0..6 {
        let mut bad = v.clone();
        match mutation {
            0 => bad["writer"]["api_version"] = json!("unknown"),
            1 => bad["writer"]["algorithm_revision"] = json!("unknown"),
            2 => bad["arithmetic"]["reduction"] = json!("alias"),
            3 => bad["arithmetic"]["classification"] = json!("alias"),
            4 => bad["timing"]["setup_ns"] = json!({"status":"unavailable","reason":"missing"}),
            _ => bad["provenance"] = json!({"status":"unavailable","reason":"missing"}),
        };
        let p = parse_reference_document(&rehash(bad)).unwrap();
        assert!(required_policy(&p.common, &p.body.timing).is_err());
    }
    let ProvenanceClaim::Captured(mut p) = parsed.common.provenance else {
        panic!()
    };
    p.thread_environment.0.clear();
    assert!(runtime_inventory(&p).is_err());
    // Approved build comparisons are exact typed field bytes, not claimed hashes alone.
    let expected = p.build.clone();
    for mutation in 0..4 {
        let mut bad = expected.clone();
        match mutation {
            0 => bad.recipe_version = Text("unknown".into()),
            1 => bad.receipt_file_sha256 = Hash("f".repeat(64)),
            2 => bad.compiler_vv = Text("PATH rustc".into()),
            _ => bad.source_manifest_sha256 = Hash("e".repeat(64)),
        };
        assert!(approved_build_matches(&bad, &expected).is_err());
    }
}
