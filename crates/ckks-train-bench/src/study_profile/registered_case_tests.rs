//! Private semantic fixtures never become a public registry parser or constructor.
use super::*;
use serde_json::json;

fn source() -> Value {
    serde_json::from_str(include_str!("../../configs/experiment_matrix.json")).unwrap()
}
fn rejects(change: impl FnOnce(&mut Value)) {
    let mut v = source();
    change(&mut v);
    assert!(project(&v).is_err());
}

#[test]
fn registered_case_semantics_reject_duplicate_and_unknown_axes() {
    rejects(|v| v["families"][0]["backends"] = json!(["cpu", "cpu"]));
    rejects(|v| v["families"][0]["profile_ids"] = json!(["iris__mlp2020", "iris__mlp2020"]));
    rejects(|v| v["families"][0]["modes"] = json!(["interactive", "interactive"]));
    rejects(|v| v["families"][0]["placements"] = json!(["client_refresh", "client_refresh"]));
    rejects(|v| v["families"][2]["packing_variants"] = json!(["row_2020", "row_2020"]));
    rejects(|v| v["families"][0]["backends"] = json!([]));
    rejects(|v| v["families"][0]["backends"] = json!(["gpu"]));
    rejects(|v| v["families"][0]["modes"] = json!(["unknown"]));
    rejects(|v| v["families"][0]["placements"] = json!(["unknown"]));
    rejects(|v| v["families"][0]["packing"] = json!("unknown"));
    rejects(|v| v["families"][0]["profile_ids"] = json!(["unregistered"]));
    rejects(|v| v["families"][0]["unknown_axis"] = json!([true]));
    rejects(|v| {
        v["families"][0]["eval_config"] = json!({"hoist":true,"bsgs":true,"lazy_relin":true})
    });
}
#[test]
fn registered_case_semantics_reject_schema_registration_profile_and_seed_conflicts() {
    rejects(|v| v["schema_version"] = json!(2));
    rejects(|v| v["artifact_type"] = json!("execution"));
    rejects(|v| v["scope"]["full_seeds"] = json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 10]));
    rejects(|v| v["scope"]["full_seeds"] = json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 2020]));
    rejects(|v| v["scope"]["full_seeds"] = json!([0, 1, 2, 3, 4, 5, 6, 7, 8, 8]));
    rejects(|v| v["scope"]["quick_seeds"] = json!([1]));
    rejects(|v| v["families"][0]["seeds"] = json!("seed2020"));
    rejects(|v| v["families"][0]["id"] = json!("unknown"));
    rejects(|v| v["families"][0]["id"] = json!("FULL-1"));
    rejects(|v| v["families"][1] = v["families"][0].clone());
    rejects(|v| v["profiles"][1] = v["profiles"][0].clone());
    rejects(|v| v["profiles"][0]["batch"] = json!(0));
    rejects(|v| v["profiles"][0]["batch"] = json!(-1));
    rejects(|v| v["profiles"][0]["id"] = json!("x".repeat(129)));
    rejects(|v| v["families"][0]["packing_variants"] = json!(["batched_diagonal"]));
    rejects(|v| {
        v["families"][0].as_object_mut().unwrap().remove("packing");
    });
}
#[test]
fn registered_case_semantics_reject_role_crypto_and_placement_substitutions() {
    rejects(|v| v["families"][0]["role"] = json!("baseline_a"));
    rejects(|v| v["families"][0]["role"] = json!("unknown"));
    rejects(|v| v["families"][0]["modes"] = json!(["noninteractive"]));
    rejects(|v| v["families"][0]["placements"] = json!(["after_update"]));
    rejects(|v| v["families"][1]["placements"] = json!(["client_refresh"]));
    rejects(|v| v["families"][0]["packing"] = json!("row_2020"));
    rejects(|v| v["families"][2]["backends"] = json!(["cuda"]));
    rejects(|v| v["families"][2]["profile_ids"] = json!(["iris__emlp1"]));
    rejects(|v| v["families"][3]["backends"] = json!(["cpu"]));
    rejects(|v| v["families"][5]["profile_ids"] = json!(["iris__emlp2"]));
    rejects(|v| v["families"][5]["placements"] = json!(["when_needed"]));
    rejects(|v| v["families"][2]["crypto_override"] = json!("seal_2020"));
    rejects(|v| v["families"][0]["crypto_override"] = json!("n16_train"));
    for key in [
        "main_interactive",
        "main_noninteractive",
        "packed_c_interactive",
        "packed_c_noninteractive",
        "baseline_a_matched",
        "baseline_a_exact_iris",
        "official_b",
    ] {
        rejects(|v| v["crypto"][key] = json!("wrong_crypto"));
    }
}
#[test]
fn registered_case_semantics_reject_supplement_authority_injection() {
    rejects(|v| v["supplements"][0]["profile"] = json!("iris__emlp1"));
    rejects(|v| v["supplements"][0]["role"] = json!("main"));
    rejects(|v| v["supplements"][0]["preset"] = json!("n16_train"));
    rejects(|v| v["supplements"][0]["seeds"] = json!("2020"));
    rejects(|v| v["supplements"][0]["packing_variants"] = json!(["row_2020", "row_2020"]));
    rejects(|v| v["supplements"][0]["placements"] = json!(["after_update"]));
    rejects(|v| v["supplements"][1]["id"] = json!("unknown"));
    rejects(|v| v["supplements"][1] = v["supplements"][0].clone());
    rejects(|v| v["supplements"][4]["configs"] = json!("caller flags"));
}

#[test]
fn registered_case_semantics_reject_unknown_concrete_profile_model() {
    // All concrete profile metadata must stay in the approved closed model vocabulary.
    rejects(|v| v["profiles"][2]["model"] = json!("unregistered_model"));
}

#[test]
fn registered_case_unresolved_descriptor_flags_cannot_create_identity() {
    let mut v = source();
    v["supplements"][4]["profile"] = json!("iris__mlp2020");
    v["supplements"][4]["packing"] = json!("batched_diagonal");
    v["supplements"][4]["eval_config"] = json!({"hoist":true,"bsgs":true,"lazy_relin":true});
    let cases = project(&v).unwrap();
    let mut request = cases.family_cases()[0].selector().clone();
    request.origin = CaseOrigin::Supplement;
    request.registration_id = "rq3_all8".into();
    assert!(cases.select(&request).is_err());
    assert_eq!(cases.supplements()[4].raw(), &v["supplements"][4]);
}
