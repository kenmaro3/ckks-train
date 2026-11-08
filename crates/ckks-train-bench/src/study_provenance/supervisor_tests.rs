//! Independent origin literals. These tests never obtain a live owner.
use super::resource_operational::approval::scan_approval;
use std::io::{Cursor, Read};
#[test]
fn operational_probe_origin_literal_and_receipt_copy_path() {
    use super::resource_operational::probe::parse_probe;
    fn text(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    fn n(b: &mut Vec<u8>, n: u64) {
        b.extend_from_slice(&n.to_le_bytes());
    }
    let mut raw = Vec::new();
    text(&mut raw, "hekit-study-reference-encoding-v1");
    text(&mut raw, "hekit-test-resource-supervisor-probe-origin-v1");
    raw.push(1);
    for s in [
        "a".repeat(64),
        "b".repeat(40),
        "c".repeat(64),
        "study-train-cpu-serial-linux-clean-v2".into(),
    ] {
        text(&mut raw, &s);
    }
    let receipt_at = raw.len();
    for name in ["build-receipt.bin", "host-origin.bin"] {
        raw.push(3);
        text(&mut raw, &format!("resource/linux-cpu-observed-v1/{name}"));
        n(&mut raw, 123);
        text(&mut raw, &"d".repeat(64));
    }
    n(&mut raw, 1000);
    for (path, inode) in [
        ("/home/unit", 21),
        ("/home/unit/policy", 22),
        ("/repo/results", 23),
    ] {
        text(&mut raw, path);
        n(&mut raw, 20);
        n(&mut raw, inode);
    }
    text(&mut raw, "probe0");
    let p = parse_probe(&raw, true).unwrap();
    assert_eq!(p.account_uid, 1000);
    assert_eq!(p.receipt.root, 3);
    assert_eq!(p.probe_id.as_bytes(), b"probe0");
    assert!(parse_probe(&raw, false).is_err());
    for end in 0..raw.len() {
        assert!(parse_probe(&raw[..end], true).is_err());
    }
    let mut bad = raw.clone();
    bad[receipt_at] = 4;
    assert!(parse_probe(&bad, true).is_err());
    let at = raw
        .windows(17)
        .position(|v| v == b"build-receipt.bin")
        .unwrap();
    let mut bad = raw.clone();
    bad[at] = b'x';
    assert!(parse_probe(&bad, true).is_err());
    let mut extra = raw;
    extra.push(0);
    assert!(parse_probe(&extra, true).is_err());
}

fn literal() -> (Vec<u8>, String) {
    let value: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/cpu_supervisor.json")).unwrap();
    let hex = value["hex"].as_str().unwrap();
    let raw = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    (raw, value["sha256"].as_str().unwrap().to_owned())
}

#[test]
fn d7_supervisor_origin_projection_literal() {
    let (raw, hash) = literal();
    assert_eq!(crate::data::sha256_hex(&raw), hash);
    let parsed = scan_approval(&mut Cursor::new(&raw), &hash).unwrap();
    assert_eq!(parsed.origin.account_uid, 1000);
    assert_eq!(parsed.origin.account_anchor.as_bytes(), b"/home/unit");
    assert_eq!(parsed.origin.policy_origin.as_bytes(), b"/home/unit/policy");
    assert_eq!(
        parsed.origin.observation_origin.as_bytes(),
        b"/repo/results"
    );
    assert_eq!(parsed.origin.account_identity, (21, 22));
    assert_eq!(parsed.origin.policy_identity, (21, 23));
    assert_eq!(parsed.origin.observation_identity, (21, 24));
    assert_eq!(
        parsed.origin.bootstrap.as_bytes(),
        b"7777777777777777777777777777777777777777777777777777777777777777"
    );
    assert_eq!(parsed.origin.host_origin, parsed.origin.namespace_origin);
    assert_eq!(
        parsed.source.as_bytes(),
        b"3333333333333333333333333333333333333333333333333333333333333333"
    );
    assert_eq!(parsed.build_count, 1);
}

#[test]
fn d7_supervisor_origin_reject_mutations() {
    let (raw, hash) = literal();
    assert!(scan_approval(&mut Cursor::new(&raw), &"0".repeat(64)).is_err());
    for end in 0..raw.len() {
        let truncated = &raw[..end];
        let pin = crate::data::sha256_hex(truncated);
        assert!(
            scan_approval(&mut Cursor::new(truncated), &pin).is_err(),
            "cut {end}"
        );
    }
    let mut extra = raw.clone();
    extra.push(0);
    let pin = crate::data::sha256_hex(&extra);
    assert!(scan_approval(&mut Cursor::new(&extra), &pin).is_err());
    for from in [b"local-approval-v3".as_slice(), b"/home/unit".as_slice()] {
        let at = raw.windows(from.len()).position(|v| v == from).unwrap();
        let mut changed = raw.clone();
        changed[at + from.len() - 1] = if from[0] == b'l' { b'2' } else { 0 };
        let pin = crate::data::sha256_hex(&changed);
        assert!(scan_approval(&mut Cursor::new(&changed), &pin).is_err());
    }
    assert!(scan_approval(&mut Cursor::new(&raw), &hash[..63]).is_err());
}

#[test]
fn d7_supervisor_origin_stream_fragments() {
    struct One<'a>(&'a [u8]);
    impl Read for One<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let count = self.0.len().min(out.len()).min(1);
            out[..count].copy_from_slice(&self.0[..count]);
            self.0 = &self.0[count..];
            Ok(count)
        }
    }
    let (raw, hash) = literal();
    let value = scan_approval(&mut One(&raw), &hash).unwrap();
    assert_eq!(value.origin.account_uid, 1000);
}

#[test]
fn d7_supervisor_v3_full_decoder_and_recipe() {
    let (raw, _) = literal();
    assert!(super::codec::LocalApproval::decode(&raw).is_ok());
    assert!(
        super::role::Recipe::classify(
            "study-train-cpu-serial-linux-clean-v2",
            "study-train-session-v1"
        )
        .is_ok()
    );
    assert!(
        super::role::Recipe::classify(
            "study-reference-linux-clean-v2",
            "study-reference-session-v1"
        )
        .is_ok()
    );
}
#[test]
fn d7_supervisor_host_projection_literal() {
    use super::resource_operational::host::parse_host;
    fn text(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn n(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }
    fn reference(out: &mut Vec<u8>, path: &str) {
        out.push(3);
        text(out, path);
        n(out, 100);
        text(out, &"a".repeat(64));
    }
    let mut b = Vec::new();
    text(&mut b, "hekit-study-reference-encoding-v1");
    text(&mut b, "hekit-test-resource-observed-host-origin-v1");
    b.push(1);
    text(&mut b, &"8".repeat(64));
    text(&mut b, "01234567-89ab-cdef-0123-456789abcdef");
    for value in 1..=8 {
        n(&mut b, value);
    }
    reference(&mut b, "resource/install.bin");
    for value in [1000, 21, 22] {
        n(&mut b, value);
    }
    for (path, values) in [
        ("/proc", [1, 2, 3, 0x9fa0]),
        ("/sys/fs/cgroup", [4, 5, 6, 0x63677270]),
    ] {
        text(&mut b, path);
        for value in values {
            n(&mut b, value);
        }
    }
    text(&mut b, "/");
    for path in [
        "raw/mountinfo",
        "raw/membership",
        "raw/controllers",
        "raw/machine",
        "resource/linux-cpu-observed-v1/filesystems.json",
    ] {
        reference(&mut b, path);
    }
    let host = parse_host(&b, true).unwrap();
    assert_eq!(host.account_uid, 1000);
    assert_eq!(host.namespaces[3], (7, 8));
    assert_eq!(host.proc_root.path.as_bytes(), b"/proc");
    assert!(parse_host(&b, false).is_err());
    for end in 0..b.len() {
        assert!(parse_host(&b[..end], true).is_err());
    }
    let mut extra = b.clone();
    extra.push(0);
    assert!(parse_host(&extra, true).is_err());
}
fn filesystem_pair(account_only: bool) -> (Vec<u8>, Vec<u8>) {
    let reference = serde_json::json!({"root":3,"path":"raw/a","length":1,"sha256":"a".repeat(64)});
    let row = |device, roots: Vec<u8>| serde_json::json!({"device":device,"mount_id":2,"type":61267,"roots":roots,"mount_evidence":reference,"quota_evidence":reference,"backing_evidence":reference});
    let rows = if account_only {
        vec![row(1, vec![]), row(2, vec![1, 3, 5])]
    } else {
        vec![row(1, vec![1, 3, 5])]
    };
    let mut value = serde_json::json!({"schema":"hekit-test-resource-observed-filesystems-v1","synthetic":true,"physical_host":"8".repeat(64),"boot_id":"01234567-89ab-cdef-0123-456789abcdef","source_manifest":"3".repeat(64),"source_revision":"2".repeat(40),"executable":"6".repeat(64),"account_filesystem":{"device":1,"mount_id":2,"type":61267},"filesystems":rows});
    let mut review = value.clone();
    review["schema"] = serde_json::json!("hekit-test-resource-observed-filesystem-review-v1");
    value["review"] = serde_json::json!({"root":3,"path":"resource/linux-cpu-observed-v1/filesystem-review.json","length":serde_json::to_vec(&review).unwrap().len(),"sha256":crate::data::sha256_hex(&serde_json::to_vec(&review).unwrap())});
    (
        serde_json::to_vec(&value).unwrap(),
        serde_json::to_vec(&review).unwrap(),
    )
}
#[test]
fn d7_supervisor_filesystem_source_exact_joins() {
    for separate in [false, true] {
        let (source, review) = filesystem_pair(separate);
        let rows = crate::study_reference::operational::validate_filesystem_sources(
            &source,
            &review,
            &[1, 3, 5],
            true,
        )
        .unwrap();
        assert_eq!(rows.len(), if separate { 2 } else { 1 });
    }
}
#[test]
fn d7_supervisor_filesystem_source_negative_joins() {
    let (source, review) = filesystem_pair(false);
    for replacement in [
        serde_json::json!([]),
        serde_json::json!([1, 3]),
        serde_json::json!([1, 3, 5, 6]),
    ] {
        let mut value: serde_json::Value = serde_json::from_slice(&source).unwrap();
        value["filesystems"][0]["roots"] = replacement;
        assert!(
            crate::study_reference::operational::validate_filesystem_sources(
                &serde_json::to_vec(&value).unwrap(),
                &review,
                &[1, 3, 5],
                true
            )
            .is_err()
        );
    }
    let mut value: serde_json::Value = serde_json::from_slice(&source).unwrap();
    value["account_filesystem"]["device"] = serde_json::json!(9);
    assert!(
        crate::study_reference::operational::validate_filesystem_sources(
            &serde_json::to_vec(&value).unwrap(),
            &review,
            &[1, 3, 5],
            true
        )
        .is_err()
    );
    let mut value: serde_json::Value = serde_json::from_slice(&source).unwrap();
    let row = value["filesystems"][0].clone();
    value["filesystems"].as_array_mut().unwrap().push(row);
    assert!(
        crate::study_reference::operational::validate_filesystem_sources(
            &serde_json::to_vec(&value).unwrap(),
            &review,
            &[1, 3, 5],
            true
        )
        .is_err()
    );
    let raw = String::from_utf8(source.clone()).unwrap();
    let duplicate = raw.replacen('{', "{\"synthetic\":true,", 1);
    assert!(
        crate::study_reference::operational::validate_filesystem_sources(
            duplicate.as_bytes(),
            &review,
            &[1, 3, 5],
            true
        )
        .is_err()
    );
    assert!(
        crate::study_reference::operational::validate_filesystem_sources(
            &source,
            &review,
            &[1, 3, 5],
            false
        )
        .is_err()
    );
}
#[test]
fn d7_supervisor_native_source_review_joins() {
    let ids = [
        "pidfd-owned-unreaped",
        "subreaper-known-unknown",
        "lock-ofd-inheritance",
        "worker-handshake-exec",
        "python-git-direct-broker",
        "git-no-descendants",
        "parent-eof-custody",
        "nice-ten",
        "monotonic-common-origin",
        "rss-ten-second-deadline",
        "rss-term-grace-kill",
        "proc-namespace-cgroup",
        "storage-mount-quota",
        "publisher-no-clobber-fsync",
        "crash-pending-block",
        "resolved-closure",
    ];
    let r = serde_json::json!({"root":3,"path":"raw/probe","length":1,"sha256":"a".repeat(64)});
    let host = serde_json::json!({"root":3,"path":"resource/linux-cpu-observed-v1/host-origin.bin","length":1,"sha256":"b".repeat(64)});
    let clock = serde_json::json!({"wall_source":0,"unix_ns":100,"monotonic_source":0,"elapsed_since_operation_start_ns":10});
    let hex = |s: &str| {
        s.as_bytes()
            .iter()
            .flat_map(|b| {
                let h = b"0123456789abcdef";
                [
                    char::from(h[(b >> 4) as usize]),
                    char::from(h[(b & 15) as usize]),
                ]
            })
            .collect::<String>()
    };
    let mut source = serde_json::json!({"schema":"hekit-test-resource-observed-native-validation-v1","synthetic":true,"source_manifest":"3".repeat(64),"source_revision":"2".repeat(40),"executable":"6".repeat(64),"recipe":"study-train-cpu-serial-linux-clean-v2","build_receipt":r,"host_origin":host,"checks":[]});
    let mut review = serde_json::json!({"schema":"hekit-test-resource-observed-native-review-v1","synthetic":true,"source_manifest":"3".repeat(64),"source_revision":"2".repeat(40),"executable":"6".repeat(64),"recipe":"study-train-cpu-serial-linux-clean-v2","host_origin":host,"checks":[]});
    for id in ids {
        let argv = [
            "/unit/exe",
            "--native-supervisor-check",
            id,
            "--probe-origin",
            "/unit/probe",
            "--probe-origin-sha256",
            &"c".repeat(64),
        ]
        .map(hex);
        source["checks"].as_array_mut().unwrap().push(serde_json::json!({"id":id,"argv":argv,"started":clock,"finished":clock,"exit_code":0,"command":r,"stdout":r,"stderr":r,"observations":r}));
        review["checks"].as_array_mut().unwrap().push(serde_json::json!({"id":id,"command":r,"stdout":r,"stderr":r,"observations":r,"threshold_event_injected":id=="rss-term-grace-kill"}));
    }
    let raw_review = serde_json::to_vec(&review).unwrap();
    source["review"] = serde_json::json!({"root":3,"path":"resource/linux-cpu-observed-v1/native-review.json","length":raw_review.len(),"sha256":crate::data::sha256_hex(&raw_review)});
    let raw = serde_json::to_vec(&source).unwrap();
    assert!(
        crate::study_reference::operational::validate_native_sources(
            &raw,
            &raw_review,
            b"/unit/exe",
            true
        )
        .is_ok()
    );
    source["checks"].as_array_mut().unwrap().swap(0, 1);
    assert!(
        crate::study_reference::operational::validate_native_sources(
            &serde_json::to_vec(&source).unwrap(),
            &raw_review,
            b"/unit/exe",
            true
        )
        .is_err()
    );
    assert!(
        crate::study_reference::operational::validate_native_sources(
            &raw,
            &raw_review,
            b"/unit/other",
            true
        )
        .is_err()
    );
    review["checks"][0]["threshold_event_injected"] = serde_json::json!(true);
    assert!(
        crate::study_reference::operational::validate_native_sources(
            &raw,
            &serde_json::to_vec(&review).unwrap(),
            b"/unit/exe",
            true
        )
        .is_err()
    );
}
#[test]
fn operational_package_roots_and_current_receipt_are_closed() {
    use super::resource_operational::package::{package_path, receipt_join};
    use crate::study_reference::operational::FileReference;
    let make = || FileReference {
        root: 3,
        path: "resource/linux-cpu-observed-v1/raw/native.stdout".into(),
        length: 3,
        hash: "a".repeat(64),
    };
    assert!(package_path(&make()).is_ok());
    for role in [0, 1, 2, 4, 5] {
        let mut r = make();
        r.root = role;
        assert!(package_path(&r).is_err());
    }
    for path in [
        "resource/linux-cpu-observed-v1/../x",
        "resource/linux-cpu-observed-v10/x",
        "/resource/linux-cpu-observed-v1/x",
        "resource/linux-cpu-observed-v1//x",
    ] {
        let mut r = make();
        r.path = path.into();
        assert!(package_path(&r).is_err());
    }
    let expected = FileReference {
        root: 4,
        path: "selected.receipt.bin".into(),
        length: 123,
        hash: "b".repeat(64),
    };
    assert!(receipt_join(&expected, &expected, (2, 3), (2, 3)).is_ok());
    for what in 0..4 {
        let mut bad = expected.clone();
        match what {
            0 => bad.root = 3,
            1 => bad.path = "sibling.receipt.bin".into(),
            2 => bad.length += 1,
            _ => bad.hash = "c".repeat(64),
        };
        assert!(receipt_join(&bad, &expected, (2, 3), (2, 3)).is_err());
    }
    assert!(receipt_join(&expected, &expected, (2, 3), (2, 4)).is_err());
}

#[test]
fn operational_probe_receipt_copy_matches_original_bytes_and_tuple() {
    use super::{
        codec::{Capsule, Receipt},
        resource_operational::probe::{receipt_copy, receipt_tuple},
    };
    let raw = super::role_tests::literal("cpu_serial_receipt");
    let receipt = Receipt::decode(&raw).unwrap();
    let capsule = Capsule::decode(&receipt.capsule).unwrap();
    receipt_copy(&raw, &raw).unwrap();
    receipt_tuple(
        &raw,
        &receipt.capsule,
        &capsule.manifests[0],
        &capsule.commit,
        &receipt.executable_hash,
        &capsule.recipe,
        &receipt.executable,
    )
    .unwrap();
    let mut bad = raw.clone();
    bad.push(0);
    assert!(receipt_copy(&raw, &bad).is_err());
    for field in 0..6 {
        let mut expected = [
            receipt.capsule.clone(),
            capsule.manifests[0].as_bytes().to_vec(),
            capsule.commit.as_bytes().to_vec(),
            receipt.executable_hash.as_bytes().to_vec(),
            capsule.recipe.as_bytes().to_vec(),
            receipt.executable.clone(),
        ];
        expected[field].push(b'x');
        assert!(
            receipt_tuple(
                &raw,
                &expected[0],
                std::str::from_utf8(&expected[1]).unwrap(),
                std::str::from_utf8(&expected[2]).unwrap(),
                std::str::from_utf8(&expected[3]).unwrap(),
                std::str::from_utf8(&expected[4]).unwrap(),
                &expected[5]
            )
            .is_err()
        );
    }
}
