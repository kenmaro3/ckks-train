//! Pure registered membership and independently frozen identities; no dataset or HE work.
use ckks_train_bench::study_profile::*;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fmt::Write, path::Path};

fn registry() -> StudyRegistry {
    StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
}
fn identity(c: &RegisteredCase) -> Value {
    let s = c.selector();
    json!({"registry_sha256": c.registry_sha256(), "origin": s.origin.as_str(),
        "registration_id": s.registration_id, "profile_id": s.profile_id, "seed": s.seed,
        "scope": s.scope.as_str(), "role": c.role().as_str(), "backend": s.backend.as_str(),
        "mode": s.mode.as_str(), "placement": s.placement.as_str(), "packing": s.packing.as_str(),
        "security_policy": c.security_policy().as_str(), "logical_batch": c.logical_batch(),
        "eval_config": c.eval_config().map(|f| json!({"hoist":f.hoist, "bsgs":f.bsgs, "lazy_relin":f.lazy_relin}))})
}
fn selector(v: &Value) -> CaseSelector {
    CaseSelector {
        origin: v["origin"].as_str().unwrap().parse().unwrap(),
        registration_id: v["registration_id"].as_str().unwrap().into(),
        profile_id: v["profile_id"].as_str().unwrap().into(),
        seed: v["seed"].as_u64().unwrap(),
        scope: v["scope"].as_str().unwrap().parse().unwrap(),
        backend: v["backend"].as_str().unwrap().parse().unwrap(),
        mode: v["mode"].as_str().unwrap().parse().unwrap(),
        placement: v["placement"].as_str().unwrap().parse().unwrap(),
        packing: v["packing"].as_str().unwrap().parse().unwrap(),
    }
}

#[test]
fn every_family_tuple_matches_independent_oracle_without_input_resolution() {
    let r = registry();
    let cases = r.registered_cases().unwrap();
    let fixture = include_str!("fixtures/registered-case-v1/family-tuples.tsv");
    let mut lines = fixture.lines();
    let fields: Vec<_> = lines.next().unwrap().split('\t').collect();
    let expected: BTreeSet<_> = lines.map(str::to_owned).collect();
    let actual: BTreeSet<_> = cases
        .family_cases()
        .iter()
        .map(|c| {
            let v = identity(c);
            fields
                .iter()
                .map(|k| {
                    v[*k]
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v[*k].to_string())
                })
                .collect::<Vec<_>>()
                .join("\t")
        })
        .collect();
    assert_eq!(actual, expected); // all axes/role/security/batch/flags, not only counts
    assert_eq!(actual.len(), 2680);
    assert_eq!(
        cases
            .family_cases()
            .iter()
            .map(|c| &c.selector().profile_id)
            .collect::<BTreeSet<_>>()
            .len(),
        21
    );
    assert!(
        cases
            .family_cases()
            .iter()
            .all(|c| c.eval_config().is_none())
    );
    let quick = cases.quick_family_selectors();
    assert_eq!(quick.len(), 268);
    for q in quick {
        assert_eq!(q.seed, 0);
        assert_eq!(q.scope, CaseScope::Quick);
        assert!(cases.select(&q).is_err());
    }
    for c in cases.family_cases() {
        assert!(std::ptr::eq(cases.select(c.selector()).unwrap(), c));
    }
    assert!(cases.family_cases().iter().any(|c| c.selector().seed == 9
        && c.logical_batch() == 128
        && c.selector().backend == CaseBackend::ExternalCpu));
}

#[test]
fn all_twenty_exact_iris_cases_and_seven_complete_descriptors_survive() {
    let cases = registry().registered_cases().unwrap();
    let expected: Vec<Value> = serde_json::from_str(include_str!(
        "fixtures/registered-case-v1/exact-iris-tuples.json"
    ))
    .unwrap();
    assert_eq!(
        cases
            .exact_iris_cases()
            .iter()
            .map(identity)
            .collect::<Vec<_>>(),
        expected
    );
    for c in cases.exact_iris_cases() {
        assert!(cases.select(c.selector()).is_ok());
    }
    let raw: Vec<Value> = serde_json::from_str(include_str!(
        "fixtures/registered-case-v1/supplements-preserved.json"
    ))
    .unwrap();
    assert_eq!(
        cases
            .supplements()
            .iter()
            .map(|s| s.raw().clone())
            .collect::<Vec<_>>(),
        raw
    );
    let rq3 = cases
        .supplements()
        .iter()
        .find(|s| s.id() == "rq3_all8")
        .unwrap();
    let notes: Value = serde_json::from_str(include_str!(
        "fixtures/registered-case-v1/supplement-inspection-notes.json"
    ))
    .unwrap();
    let flags: Vec<_> = rq3
        .ablation_flags()
        .iter()
        .map(|f| json!({"hoist":f.hoist,"bsgs":f.bsgs,"lazy_relin":f.lazy_relin}))
        .collect();
    assert_eq!(json!(flags), notes["rq3_all_eight_declared_flag_tuples"]);
    assert_eq!(cases.supplements().len(), 7);
}

#[test]
fn canonical_header_fields_and_digests_match_nineteen_independent_vectors() {
    let cases = registry().registered_cases().unwrap();
    let fixtures: Value = serde_json::from_str(include_str!(
        "fixtures/registered-case-v1/representative-vectors.json"
    ))
    .unwrap();
    for v in fixtures["vectors"].as_array().unwrap() {
        let c = cases.select(&selector(&v["identity"])).unwrap();
        assert_eq!(identity(c), v["identity"]);
        let hex = c
            .canonical_bytes()
            .iter()
            .fold(String::new(), |mut hex, b| {
                write!(hex, "{b:02x}").unwrap();
                hex
            });
        assert_eq!(hex, v["canonical_hex"].as_str().unwrap(), "{}", v["name"]);
        assert_eq!(c.digest(), v["case_sha256"].as_str().unwrap());
    }
    let ids: BTreeSet<_> = cases
        .family_cases()
        .iter()
        .chain(cases.exact_iris_cases())
        .map(RegisteredCase::digest)
        .collect();
    assert_eq!(ids.len(), 2700); // uniqueness across these two concrete sets only
}

#[test]
fn selectors_cannot_upgrade_quick_archival_unknown_or_unresolved_requests() {
    let cases = registry().registered_cases().unwrap();
    let base = cases.family_cases()[0].selector().clone();
    let mut invalid = Vec::new();
    for seed in [10, 2020] {
        let mut q = base.clone();
        q.seed = seed;
        invalid.push(q);
    }
    for scope in [CaseScope::Quick, CaseScope::ArchivalFull] {
        let mut q = base.clone();
        q.scope = scope;
        invalid.push(q);
    }
    for id in ["unknown", "FULL-1", "FULL-2"] {
        let mut q = base.clone();
        q.registration_id = id.into();
        invalid.push(q);
    }
    for id in ["unknown", "mnist_released__emlp1__b128_candidate"] {
        let mut q = base.clone();
        q.profile_id = id.into();
        invalid.push(q);
    }
    let mut q = base.clone();
    q.origin = CaseOrigin::Supplement;
    invalid.push(q);
    let mut q = base.clone();
    q.backend = CaseBackend::ExternalCpu;
    invalid.push(q);
    let mut q = base.clone();
    q.mode = CaseMode::Noninteractive;
    invalid.push(q);
    let mut q = base.clone();
    q.placement = CasePlacement::AfterUpdate;
    invalid.push(q);
    let mut q = base.clone();
    q.packing = CasePacking::Row2020;
    invalid.push(q);
    for s in cases
        .supplements()
        .iter()
        .filter(|s| s.id() != "iris_2020_exact_crypto")
    {
        let mut q = base.clone();
        q.origin = CaseOrigin::Supplement;
        q.registration_id = s.id().into();
        invalid.push(q);
    }
    for q in invalid {
        assert!(cases.select(&q).is_err(), "{q:?}");
    }
    assert!("gpu".parse::<CaseBackend>().is_err());
    assert!("typo".parse::<CaseScope>().is_err());
}

#[test]
fn registry_and_lock_alone_suffice_even_when_every_dataset_file_is_absent() {
    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!("hekit-case-projection-{}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    // Own cleanup only after successfully creating our directory.
    let temp = Temp(path);
    let configs = "crates/ckks-train-bench/configs";
    std::fs::create_dir_all(temp.0.join(configs)).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for name in ["experiment_matrix.json", "experiment_input_locks.json"] {
        std::fs::copy(
            root.join(configs).join(name),
            temp.0.join(configs).join(name),
        )
        .unwrap();
    }
    let cases = StudyRegistry::load(&temp.0)
        .unwrap()
        .registered_cases()
        .unwrap();
    assert_eq!(cases.family_cases().len(), 2680);
    assert_eq!(cases.exact_iris_cases().len(), 20);
    assert!(
        cases
            .family_cases()
            .iter()
            .any(|c| c.selector().profile_id == "fashion_mnist__emlp3" && c.selector().seed == 9)
    );
    assert!(cases.family_cases().iter().any(|c| c.selector().profile_id
        == "mnist_released__emlp1__b128_candidate"
        && c.selector().seed == 9
        && c.selector().backend == CaseBackend::ExternalCpu));
    // Changing even one byte cannot grant arbitrary registry authority.
    let registry = temp.0.join(configs).join("experiment_matrix.json");
    let mut bytes = std::fs::read(&registry).unwrap();
    bytes.push(b' ');
    std::fs::write(registry, bytes).unwrap();
    assert!(StudyRegistry::load(&temp.0).is_err());
}

#[test]
fn per_family_profile_seed_and_batch_coverage_matches_frozen_inventory() {
    let cases = registry().registered_cases().unwrap();
    let expected: Value = serde_json::from_str(include_str!(
        "fixtures/registered-case-v1/membership-summary.json"
    ))
    .unwrap();
    for family in expected["all_families"].as_array().unwrap() {
        let id = family["registration_id"].as_str().unwrap();
        let rows: Vec<_> = cases
            .family_cases()
            .iter()
            .filter(|c| c.selector().registration_id == id)
            .collect();
        assert_eq!(
            rows.len() as u64,
            family["full_count"].as_u64().unwrap(),
            "{id}"
        );
        assert_eq!(
            rows.iter().filter(|c| c.selector().seed == 0).count() as u64,
            family["seed0_projection_count"].as_u64().unwrap(),
            "{id}"
        );
        let profiles: BTreeSet<_> = rows
            .iter()
            .map(|c| c.selector().profile_id.as_str())
            .collect();
        let expected_profiles: BTreeSet<_> = family["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(profiles, expected_profiles, "{id}");
        for row in rows {
            assert_eq!(
                row.logical_batch(),
                expected["profile_batches"][&row.selector().profile_id]
                    .as_u64()
                    .unwrap()
            );
        }
    }
    for (suffix, count) in [("base", 2040), ("b64", 510), ("b128", 130)] {
        let selected = cases
            .family_cases()
            .iter()
            .filter(|c| match suffix {
                "b64" => c.selector().registration_id.ends_with("_mnist_b64"),
                "b128" => c
                    .selector()
                    .registration_id
                    .ends_with("_mnist_b128_candidate"),
                _ => !c.selector().registration_id.contains("_mnist_"),
            })
            .count();
        assert_eq!(selected, count);
    }
}
