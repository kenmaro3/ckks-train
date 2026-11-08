use super::*;
fn n(b: &mut Vec<u8>, v: u64) {
    b.extend(v.to_le_bytes());
}
fn text(b: &mut Vec<u8>, v: &[u8]) {
    n(b, v.len() as u64);
    b.extend(v);
}
fn pin(b: &mut Vec<u8>, name: &str) {
    b.push(3);
    text(
        b,
        format!("resource/linux-cpu-observed-v1/{name}").as_bytes(),
    );
    n(b, 123);
    text(b, "d".repeat(64).as_bytes());
}
fn origin() -> Vec<u8> {
    let mut b = Vec::new();
    text(&mut b, b"hekit-study-reference-encoding-v1");
    text(&mut b, b"hekit-study-resource-supervisor-probe-origin-v1");
    b.push(0);
    for s in [
        "a".repeat(64),
        "b".repeat(40),
        "c".repeat(64),
        "study-train-cpu-serial-linux-clean-v2".into(),
    ] {
        text(&mut b, s.as_bytes());
    }
    pin(&mut b, "build-receipt.bin");
    pin(&mut b, "host-origin.bin");
    n(&mut b, 1000);
    for (p, i) in [
        ("/home/unit", 21),
        ("/home/unit/policy", 22),
        ("/repo/results", 23),
    ] {
        text(&mut b, p.as_bytes());
        n(&mut b, 20);
        n(&mut b, i);
    }
    text(&mut b, b"probe0");
    b
}
fn literal() -> Vec<u8> {
    let origin = origin();
    let mut b = Vec::new();
    text(&mut b, b"hekit-study-reference-encoding-v1");
    text(&mut b, b"hekit-study-supervisor-probe-observations-v1");
    b.push(0);
    text(&mut b, b"pidfd-owned-unreaped");
    text(&mut b, b"/unit/probe");
    text(&mut b, sha256_hex(&origin).as_bytes());
    text(&mut b, &origin);
    for s in [
        "a".repeat(64),
        "b".repeat(40),
        "c".repeat(64),
        "study-train-cpu-serial-linux-clean-v2".into(),
    ] {
        text(&mut b, s.as_bytes());
    }
    pin(&mut b, "build-receipt.bin");
    pin(&mut b, "host-origin.bin");
    text(&mut b, b"/repo/results");
    n(&mut b, 20);
    n(&mut b, 23);
    text(
        &mut b,
        b"native-probes/probe0/pidfd-owned-unreaped/observations.bin",
    );
    for (wall, elapsed) in [(100, 1), (200, 20)] {
        b.push(0);
        n(&mut b, wall);
        b.push(0);
        n(&mut b, elapsed);
    }
    b.push(0);
    n(&mut b, 1);
    text(&mut b, b"common-clock-conversion-loss-injected");
    n(&mut b, 1);
    text(&mut b, b"common-clock-conversion-loss-injected");
    b.push(0);
    n(&mut b, 1);
    n(&mut b, 9);
    text(&mut b, b"expected refusal");
    b
}
#[test]
fn operational_native_raw_envelope_original_copy_and_negatives() {
    use super::probe_output::{ProbeObservation, check_stdout, command};
    let raw = literal();
    let parsed = ProbeObservation::parse(&raw).unwrap();
    assert_eq!(parsed.encode().unwrap(), raw);
    assert!(parsed.events[0].elapsed.is_none());
    for end in 0..raw.len() {
        assert!(ProbeObservation::parse(&raw[..end]).is_err());
    }
    let mut changed = raw.clone();
    changed.push(0);
    assert!(ProbeObservation::parse(&changed).is_err());
    let mut huge = vec![0; 8 * 1024 * 1024 + 1];
    huge[..raw.len()].copy_from_slice(&raw);
    assert!(ProbeObservation::parse(&huge).is_err());
    let hex = |s: &[u8]| {
        s.iter()
            .flat_map(|b| {
                let h = b"0123456789abcdef";
                [
                    char::from(h[(b >> 4) as usize]),
                    char::from(h[(b & 15) as usize]),
                ]
            })
            .collect::<String>()
    };
    let argv = [
        b"/unit/exe".to_vec(),
        b"--native-supervisor-check".to_vec(),
        b"pidfd-owned-unreaped".to_vec(),
        b"--probe-origin".to_vec(),
        b"/unit/probe".to_vec(),
        b"--probe-origin-sha256".to_vec(),
        sha256_hex(&origin()).into_bytes(),
    ];
    let cmd=serde_json::to_vec(&serde_json::json!({"schema":"hekit-study-supervisor-probe-command-v1","argv":argv.iter().map(|v|hex(v)).collect::<Vec<_>>(),"cwd":hex(b"/repo")})).unwrap();
    assert!(command(&cmd, &argv, b"/repo").is_ok());
    assert!(command(&cmd, &argv, b"/other").is_err());
    let duplicate =
        String::from_utf8(cmd.clone())
            .unwrap()
            .replacen('{', "{\"schema\":\"bad\",", 1);
    assert!(command(duplicate.as_bytes(), &argv, b"/repo").is_err());
    let out=serde_json::to_vec(&serde_json::json!({"schema":"hekit-study-supervisor-probe-output-v1","check_id":"pidfd-owned-unreaped","observations":{"root":5,"path":"native-probes/probe0/pidfd-owned-unreaped/observations.bin","length":raw.len(),"sha256":sha256_hex(&raw)}})).unwrap();
    let mut line = out.clone();
    line.push(b'\n');
    check_stdout(&line, &parsed, &raw).unwrap();
    assert!(check_stdout(&out, &parsed, &raw).is_err());
    let mut p = ProbeObservation::parse(&raw).unwrap();
    p.threshold = true;
    assert!(p.encode().is_err());
    let mut p = ProbeObservation::parse(&raw).unwrap();
    p.events[0].elapsed = Some(3);
    assert!(p.encode().is_err());
    let mut p = ProbeObservation::parse(&raw).unwrap();
    p.events.clear();
    assert!(p.encode().is_err());
    for field in 0..11 {
        let mut p = ProbeObservation::parse(&raw).unwrap();
        match field {
            0 => p.source = "e",
            1 => p.revision = "e",
            2 => p.executable = "e",
            3 => p.recipe = "e",
            4 => p.receipt.root = 4,
            5 => p.host.path = "wrong".into(),
            6 => p.root_identity.0 += 1,
            7 => p.observation_root = b"/wrong",
            8 => p.origin_hash = "e",
            9 => p.origin_bytes = b"incorrect",
            _ => p.output = "wrong",
        };
        assert!(p.encode().is_err());
    }
    let mut wrong = argv.clone();
    wrong[4] = b"/other".to_vec();
    assert!(parsed.argv_matches(&wrong).is_err());
    for modification in 0..3 {
        let mut c: serde_json::Value = serde_json::from_slice(&cmd).unwrap();
        match modification {
            0 => c["unknown"] = serde_json::json!(0),
            1 => c["argv"][0] = serde_json::json!("xx"),
            _ => {
                c.as_object_mut().unwrap().remove("cwd");
            }
        }
        assert!(command(&serde_json::to_vec(&c).unwrap(), &argv, b"/repo").is_err());
    }
    for field in ["root", "path", "length", "sha256"] {
        let mut out: serde_json::Value = serde_json::from_slice(&out).unwrap();
        out["observations"][field] = match field {
            "root" => serde_json::json!(3),
            "path" => serde_json::json!("copied/observations.bin"),
            "length" => serde_json::json!(1),
            _ => serde_json::json!("f".repeat(64)),
        };
        let mut b = serde_json::to_vec(&out).unwrap();
        b.push(b'\n');
        assert!(check_stdout(&b, &parsed, &raw).is_err());
    }
    let details = vec![0; 4096];
    let mut p = ProbeObservation::parse(&raw).unwrap();
    p.events = (0..4096)
        .map(|_| super::probe_output::ProbeEvent {
            name: "bounded",
            elapsed: Some(3),
            values: vec![1; 16],
            detail: &details,
        })
        .collect();
    assert!(p.encode().is_err());
}

#[test]
fn operational_probe_encoder_has_exact_eight_mib_cap() {
    use super::probe_output::{ProbeEvent, ProbeObservation};
    let raw = literal();
    let initial = ProbeObservation::parse(&raw).unwrap();
    let e = &initial.events[0];
    let original_event = 8
        + e.name.len()
        + 1
        + if e.elapsed.is_some() { 8 } else { 0 }
        + 8
        + 8 * e.values.len()
        + 8
        + e.detail.len();
    let base = raw.len() - original_event;
    let details = vec![7; 4096];
    for size in [2 * 1024 * 1024 + 1, 8 * 1024 * 1024, 8 * 1024 * 1024 + 1] {
        let mut p = ProbeObservation::parse(&raw).unwrap();
        let mut payload = size - base - 2100 * 48;
        p.events.clear();
        for _ in 0..2100 {
            let n = payload.min(4096);
            payload -= n;
            p.events.push(ProbeEvent {
                name: "bounded",
                elapsed: Some(3),
                values: vec![1],
                detail: &details[..n],
            });
        }
        assert_eq!(payload, 0);
        if size <= 8 * 1024 * 1024 {
            let bytes = p.encode().unwrap();
            assert_eq!(bytes.len(), size);
            assert_eq!(ProbeObservation::parse(&bytes).unwrap().events.len(), 2100);
        } else {
            assert!(p.encode().is_err());
        }
    }
    let mut old = Encoder::new(DocumentLayout::Start);
    assert!(old.append(&vec![0; 2 * 1024 * 1024 + 1], false).is_err());
}

#[test]
fn operational_native_account_index_pair_binds_original_session() {
    use super::probe_output::index_join;
    let raw = literal();
    let hex = |s: &[u8]| {
        s.iter()
            .flat_map(|b| {
                let h = b"0123456789abcdef";
                [
                    char::from(h[(b >> 4) as usize]),
                    char::from(h[(b & 15) as usize]),
                ]
            })
            .collect::<String>()
    };
    let mut argv = vec![
        b"/unit/exe".to_vec(),
        b"--native-supervisor-check".to_vec(),
        b"pidfd-owned-unreaped".to_vec(),
        b"--probe-origin".to_vec(),
        b"/unit/probe".to_vec(),
        b"--probe-origin-sha256".to_vec(),
        sha256_hex(&origin()).into_bytes(),
    ];
    let command = |a: &Vec<Vec<u8>>, cwd: &[u8]| {
        serde_json::to_vec(&serde_json::json!({"schema":"hekit-study-supervisor-probe-command-v1","argv":a.iter().map(|v|hex(v)).collect::<Vec<_>>(),"cwd":hex(cwd)})).unwrap()
    };
    let cmd = command(&argv, b"/repo");
    assert!(index_join(&cmd, &raw, "probe0", "pidfd-owned-unreaped").is_ok());
    for (p, c) in [("other", "pidfd-owned-unreaped"), ("probe0", "nice-ten")] {
        assert!(index_join(&cmd, &raw, p, c).is_err());
    }
    assert!(index_join(&cmd, &[], "probe0", "pidfd-owned-unreaped").is_err());
    assert!(
        index_join(
            &command(&argv, b"/other"),
            &raw,
            "probe0",
            "pidfd-owned-unreaped"
        )
        .is_err()
    );
    for i in [1, 3, 5, 6] {
        let old = argv[i].clone();
        argv[i] = b"wrong".to_vec();
        assert!(
            index_join(
                &command(&argv, b"/repo"),
                &raw,
                "probe0",
                "pidfd-owned-unreaped"
            )
            .is_err()
        );
        argv[i] = old;
    }
}
