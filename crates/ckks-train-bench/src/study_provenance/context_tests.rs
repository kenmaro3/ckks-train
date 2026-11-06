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
