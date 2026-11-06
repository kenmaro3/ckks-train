use super::*;
#[test]
fn runtime_environment_absent_empty_and_non_utf8_are_distinct() {
    let absent = thread_environment(|_| None).unwrap();
    assert_eq!(absent.len(), 7);
    assert!(
        absent
            .iter()
            .all(|x| x.1 == Observation::Unavailable("unset"))
    );
    assert!(
        thread_environment(|k| if k == "OMP_NUM_THREADS" {
            Some(OsString::from(""))
        } else {
            None
        })
        .is_err()
    );
    assert!(
        thread_environment(|k| if k == "OMP_NUM_THREADS" {
            Some(OsString::from_vec(vec![255]))
        } else {
            None
        })
        .is_err()
    );
    let set = thread_environment(|_| Some(OsString::from("8"))).unwrap();
    assert!(
        set.iter()
            .all(|x| x.1 == Observation::Available("8".into()))
    );
}
#[test]
fn nonrecipe_embedded_capsule_is_never_approved() {
    assert!(EMBEDDED_CAPSULE.is_none());
    let request = ContextRequest::new(
        PathBuf::from("/synthetic"),
        PathBuf::from("/synthetic/approval"),
        "0".repeat(64),
        vec![],
    );
    assert_eq!(
        VerifiedStudyContext::load(request).err().unwrap().0,
        "build has no production compile capsule"
    );
}
#[test]
fn runtime_override_and_argument_bounds_fail_closed() {
    assert!(reject_runtime_keys([OsString::from("GIT_DIR")].into_iter()).is_err());
    assert!(reject_runtime_keys([OsString::from("LD_PRELOAD")].into_iter()).is_err());
    assert!(reject_runtime_keys([OsString::from("RAYON_NUM_THREADS")].into_iter()).is_ok());
    assert!(argv(vec![OsString::from_vec(vec![b'x'; 4097])]).is_err());
    assert!(argv(vec![OsString::from("x"); 257]).is_err());
    assert_eq!(
        argv(vec![OsString::from(""), OsString::from_vec(vec![255])]).unwrap(),
        vec![vec![], vec![255]]
    );
}
#[test]
fn pinned_approval_and_current_binary_capsule_receipt_mismatches_reject() {
    fn fixture(s: &str) -> Vec<u8> {
        s.split_whitespace()
            .collect::<String>()
            .as_bytes()
            .chunks_exact(2)
            .map(|x| u8::from_str_radix(std::str::from_utf8(x).unwrap(), 16).unwrap())
            .collect()
    }
    let a = fixture(include_str!("fixtures/approval.hex"));
    let r = fixture(include_str!("fixtures/receipt.hex"));
    let second = fixture(include_str!("fixtures/receipt_second_build.hex"));
    let approval = decode_pinned_approval(&a, &sha256_hex(&a)).unwrap();
    assert!(decode_pinned_approval(&a, &"0".repeat(64)).is_err());
    let receipt = Receipt::decode(&r).unwrap();
    let capsule = Capsule::decode(&receipt.capsule).unwrap();
    check_approved_receipt(&approval, &r, &receipt, &capsule).unwrap();
    let receipt2 = Receipt::decode(&second).unwrap();
    let capsule2 = Capsule::decode(&receipt2.capsule).unwrap();
    check_approved_receipt(&approval, &second, &receipt2, &capsule2).unwrap();
    let mut changed = receipt.clone();
    changed.executable_hash = "0".repeat(64);
    assert!(check_approved_receipt(&approval, &r, &changed, &capsule).is_err());
    let mut changed = capsule.clone();
    changed.commit = "0".repeat(40);
    assert!(check_approved_receipt(&approval, &r, &receipt, &changed).is_err());
    let mut changed = capsule.clone();
    changed.manifests[0] = "0".repeat(64);
    assert!(check_approved_receipt(&approval, &r, &receipt, &changed).is_err());
    assert!(check_approved_receipt(&approval, b"changed bytes", &receipt, &capsule).is_err());
    check_current_binary(
        &receipt,
        &receipt.capsule,
        &receipt.executable,
        &receipt.executable_hash,
    )
    .unwrap();
    assert!(
        check_current_binary(
            &receipt,
            b"stale capsule",
            &receipt.executable,
            &receipt.executable_hash
        )
        .is_err()
    );
    assert!(
        check_current_binary(
            &receipt,
            &receipt.capsule,
            b"/wrong/path",
            &receipt.executable_hash
        )
        .is_err()
    );
    assert!(
        check_current_binary(
            &receipt,
            &receipt.capsule,
            &receipt.executable,
            &"0".repeat(64)
        )
        .is_err()
    );
}

fn role_receipt_predicates(name: &str) -> Result<(Receipt, Capsule)> {
    let r = Receipt::decode(&super::super::role_tests::literal(name))?;
    let c = r.closed_policy()?;
    validate_recipe_vectors(
        &r,
        &c,
        &Environment::decode(&r.environment)?,
        Recipe::capsule_policy(&c)?,
    )?;
    Ok((r, c))
}
fn role_obligation(name: &str, kind: &str, expected: &str) -> Result<()> {
    use super::super::role_tests::literal;
    if kind == "capsule" {
        let c = Capsule::decode(&literal(name))?;
        if name.contains("capsule_source_digest") {
            let r = Receipt::decode(&literal("cpu_serial_receipt"))?;
            r.check_manifest_links(&c)?;
        } else if name.contains("metadata_feature") || name.contains("build_bin") {
            let r = Receipt::decode(&literal(if name.contains("metadata") {
                "cpu_rayon_receipt"
            } else {
                "cpu_serial_receipt"
            }))?;
            validate_recipe_vectors(
                &r,
                &c,
                &Environment::decode(&r.environment)?,
                Recipe::capsule_policy(&c)?,
            )?;
        } else {
            Recipe::capsule_policy(&c)?;
        }
    } else if kind == "receipt" {
        role_receipt_predicates(name)?;
    } else {
        let a = LocalApproval::decode(&literal(name))?;
        a.policy(&DESIGNS)?;
        if expected == "linked_bundle_obligation" {
            let rname = format!("{}_receipt", name.strip_suffix("_approval").unwrap());
            let (r, c) = role_receipt_predicates(&rname)?;
            check_approved_receipt(&a, &literal(&rname), &r, &c)?;
        } else if name == "negative_design_digest" {
            let good = LocalApproval::decode(&literal("approval_v2_reference_cpu_serial"))?;
            for ((_, expected), (_, actual)) in a.designs.iter().zip(&good.designs) {
                check_design_digest(expected, actual)?;
            }
        } else {
            let r = Receipt::decode(&literal("cpu_serial_receipt"))?;
            let c = Capsule::decode(&r.capsule)?;
            check_approved_receipt(&a, &literal("cpu_serial_receipt"), &r, &c)?;
        }
    }
    Ok(())
}
#[test]
fn role_all_74_named_obligations_reach_owned_predicates() {
    let index = super::super::role_tests::index();
    let mut count = 0;
    for (name, item) in index.as_object().unwrap() {
        let expected = item["expected"].as_str().unwrap();
        if !expected.contains("obligation") {
            continue;
        }
        count += 1;
        let error =
            role_obligation(name, item["kind"].as_str().unwrap(), expected).expect_err(name);
        assert_eq!(error.0, role_expected_error(name), "{name}");
        eprintln!("POLICY {name}: {error}");
    }
    assert_eq!(count, 74);
}
#[test]
fn role_vectors_pass_but_partial_fixture_never_issues_checked_role() {
    for name in ["reference", "cpu_serial", "cpu_rayon", "cuda_host"] {
        let (r, c) = role_receipt_predicates(&format!("{name}_receipt")).unwrap();
        assert!(validate_linkage(&r, &c, &Environment::decode(&r.environment).unwrap()).is_err());
    }
}

#[test]
fn role_current_and_artifact_reference_predicates_are_independent() {
    use super::super::role_tests::literal;
    let a = LocalApproval::decode(&literal("approval_v2_multiple_builds")).unwrap();
    a.policy(&DESIGNS).unwrap();
    for name in ["reference", "cpu_serial", "cpu_rayon", "cuda_host"] {
        let bytes = literal(&format!("{name}_receipt"));
        let (r, c) = role_receipt_predicates(&format!("{name}_receipt")).unwrap();
        check_approved_receipt(&a, &bytes, &r, &c).unwrap();
        let recipe = Recipe::capsule_policy(&c).unwrap();
        // The same private predicate used by load refuses Train even when
        // embedded/path claims match exactly; no context or checked role minted.
        assert_eq!(
            reference_current_candidate(recipe, &r, &r.capsule, &r.executable),
            name == "reference"
        );
        assert_eq!(reference_recipe(recipe), name == "reference");
        assert!(!reference_current_candidate(
            recipe,
            &r,
            b"stale",
            &r.executable
        ));
        assert!(!reference_current_candidate(
            recipe, &r, &r.capsule, b"wrong"
        ));
    }
    let a = LocalApproval::decode(&literal("approval_v2_reference_only")).unwrap();
    let r = Receipt::decode(&literal("reference_second_receipt")).unwrap();
    let c = Capsule::decode(&r.capsule).unwrap();
    assert!(check_approved_receipt(&a, &literal("reference_second_receipt"), &r, &c).is_err());
}

fn role_expected_error(name: &str) -> &'static str {
    if name.contains("unknown_recipe") || name.contains("unknown_approval_recipe") {
        "unsupported recipe"
    } else if name.contains("unknown_algorithm") || name.contains("unknown_approval_algorithm") {
        "unsupported algorithm"
    } else if name.contains("feature_order") && !name.contains("metadata") {
        "noncanonical order/duplicate"
    } else if name.contains("serial_parallel")
        || name.contains("missing_feature")
        || name.contains("missing_parallel")
        || name.contains("unknown_feature")
        || name.contains("wrong_target")
    {
        "unsupported compile facts"
    } else if name.contains("compiler_host") {
        "compiler host"
    } else if name.contains("wrong_source") || name.contains("wrong_emitted_target") {
        "unique writer binary linkage"
    } else if name.contains("wrong_output") {
        "receipt executable path"
    } else if name.contains("metadata_feature") {
        "metadata argv"
    } else if name.contains("helper_extra_features") {
        "unsupported helper option"
    } else if name.contains("helper_script") {
        "fixed helper dispatch"
    } else if name.contains("build_bin") {
        "build argv"
    } else if name.contains("post_") {
        "pre/post mismatch"
    } else if name.contains("capsule_source_digest") {
        "embedded manifest linkage"
    } else if name == "negative_role_recipe_mismatch" {
        "approval role/recipe mismatch"
    } else if name.contains("triple_") || name == "negative_v1_hidden_train" {
        "receipt/capsule/executable not approved"
    } else if name == "negative_empty_id" {
        "approval policy"
    } else if name == "negative_wrong_design_path" {
        "approved design set"
    } else if name == "negative_design_digest" {
        "design bytes changed"
    } else if name == "negative_commit" || name == "negative_approval_source" {
        "approved source/version mismatch"
    } else if name.starts_with("negative_v1_") {
        "approval v1 Reference only"
    } else {
        panic!("unmapped obligation {name}")
    }
}
