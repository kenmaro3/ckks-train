//! Independent fictional byte expectations. No registry/input/HE/native work.
use super::observed::*;
use super::*;
fn literal(name: &str) -> Vec<u8> {
    let map: serde_json::Value =
        serde_json::from_str(include_str!("observed/literals.json")).unwrap();
    let hex = map[name].as_str().unwrap();
    hex.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn d7_cpu_wire_requested_literals() {
    for i in 1..=10 {
        let b = literal(&format!("W{i:02}"));
        let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
        assert_eq!(w.encode_fixed().unwrap().as_slice(), b);
        assert!(RequestedWork::decode(&b, Mode::Production).is_err());
    }
    assert_eq!(literal("W01").len(), 251);
}
#[test]
fn d7_cpu_wire_requested_rejects() {
    let b = literal("W02");
    for end in 0..b.len() {
        assert!(RequestedWork::decode(&b[..end], Mode::Oracle).is_err());
    }
    let mut v = b.clone();
    v.push(0);
    assert!(RequestedWork::decode(&v, Mode::Oracle).is_err());
    let pos = b
        .windows(62)
        .position(|s| s[..6] == [1, 0, 0, 1, 1, 1])
        .unwrap();
    let mut v = b.clone();
    v[pos + 38..pos + 46].copy_from_slice(&3u64.to_le_bytes());
    assert!(RequestedWork::decode(&v, Mode::Oracle).is_err());
    let mut v = b.clone();
    let p = v.windows(11).position(|x| x == b"iris__emlp1").unwrap();
    v[p] = b'x';
    assert!(RequestedWork::decode(&v, Mode::Oracle).is_err());
    let base = literal("W05");
    let kind = base.windows(11).position(|x| x == b"iris__emlp1").unwrap() - 10;
    let mut v = base.clone();
    v[kind] = 2;
    assert!(RequestedWork::decode(&v, Mode::Oracle).is_err());
    for seeds in [vec![], vec![0, 0], vec![1, 0], vec![10], vec![0; 11]] {
        let mut v = base[..base.len() - 16].to_vec();
        v.extend_from_slice(&(seeds.len() as u64).to_le_bytes());
        for n in seeds {
            v.extend_from_slice(&(n as u64).to_le_bytes());
        }
        assert!(RequestedWork::decode(&v, Mode::Oracle).is_err());
    }
    assert!(RequestedWork::decode(&[0; 4097], Mode::Oracle).is_err());
}
#[test]
fn d7_cpu_wire_request_case_binding() {
    for (work, case) in [
        ("W01", "case-i"),
        ("W02", "case-i"),
        ("W03", "case-ni"),
        ("W04", "case-ni"),
    ] {
        let b = literal(work);
        let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
        let c = literal(case);
        let req = w.train_request().unwrap();
        w.bind_train(&c, TRAIN_RECIPE, req, None).unwrap();
        assert!(
            w.bind_train(
                &literal(if case == "case-i" {
                    "case-ni"
                } else {
                    "case-i"
                }),
                TRAIN_RECIPE,
                req,
                None
            )
            .is_err()
        );
        assert!(w.bind_train(&c, "foreign", req, None).is_err());
        assert!(w.bind_train(&c, TRAIN_RECIPE, req, Some(1)).is_err());
        let other = literal(if work == "W01" || work == "W03" {
            "W02"
        } else {
            "W01"
        });
        assert!(
            w.bind_train(
                &c,
                TRAIN_RECIPE,
                RequestedWork::decode(&other, Mode::Oracle)
                    .unwrap()
                    .train_request()
                    .unwrap(),
                None
            )
            .is_err()
        );
    }
}
#[test]
fn d7_cpu_wire_root_child_graph() {
    for op in 0..6 {
        let b = literal(&format!("W{:02}", op + 5));
        let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
        w.bind_preparation_root(op, "iris__emlp1", &[0]).unwrap();
        assert!(
            w.bind_preparation_root((op + 1) % 6, "iris__emlp1", &[0])
                .is_err()
        );
        assert!(w.bind_preparation_root(op, "iris__emlp1", &[0, 1]).is_err());
        for child in 0..6 {
            assert_eq!(
                w.check_child(op, child, "iris__emlp1", 0).is_ok(),
                op >= 3 && child == 2
            );
        }
        assert!(w.check_child(op, 2, "foreign", 0).is_err());
    }
    for name in ["W01", "W03"] {
        let b = literal(name);
        let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
        w.check_child(6, 3, "iris__emlp1", 0).unwrap();
        assert!(w.check_child(6, 3, "iris__emlp1", 1).is_err());
        w.check_child(3, 2, "iris__emlp1", 0).unwrap();
        assert!(w.check_child(6, 5, "iris__emlp1", 0).is_err());
    }
    let b = literal("W04");
    assert!(
        RequestedWork::decode(&b, Mode::Oracle)
            .unwrap()
            .check_child(6, 3, "iris__emlp1", 0)
            .is_err()
    );
}
#[test]
fn d7_cpu_wire_records_literals() {
    for (name, kind) in [
        ("bootstrap", RecordKind::Bootstrap),
        ("pending", RecordKind::Pending),
        ("claim", RecordKind::Claim),
        ("host-pending", RecordKind::HostPending),
        ("host-closed", RecordKind::HostClosed),
        ("supervisor-end", RecordKind::SupervisorEnd),
        ("full-end", RecordKind::FullEnd),
        ("result", RecordKind::Result),
    ] {
        let b = literal(name);
        let r = ObservedRecord::decode(kind, &b, Mode::Oracle).unwrap();
        assert_eq!(r.canonical().unwrap(), b);
        assert!(ObservedRecord::decode(kind, &b, Mode::Production).is_err());
        let mut extra = b.clone();
        extra.push(0);
        assert!(ObservedRecord::decode(kind, &extra, Mode::Oracle).is_err());
        assert!(ObservedRecord::decode(kind, &b[..b.len() - 1], Mode::Oracle).is_err());
    }
}
#[test]
fn d7_cpu_wire_resource_arithmetic() {
    let b = literal("demand");
    let mut c = Cursor::new(&b);
    let d = SourceDemand::read(&mut c).unwrap();
    c.finish().unwrap();
    d.validate().unwrap();
    assert_eq!(d.additional_payload_bytes, 16);
    assert_eq!(d.retained_payload_bytes, 24);
    let mut d = d.clone();
    d.additional_payload_bytes = 15;
    assert!(d.validate().is_err());
    let mut d = d.clone();
    d.allocations[0].count = u64::MAX;
    assert!(d.validate().is_err());
    let b = literal("supervisor-binding");
    let mut c = Cursor::new(&b);
    let s = SupervisorBinding::read(&mut c).unwrap();
    s.validate().unwrap();
    let mut s = s.clone();
    s.max_processes = 5;
    assert!(s.validate().is_err());
}
fn with_current(prefix: &str, check: impl FnOnce(&CurrentEvidenceBundle<'_, '_>)) {
    let bytes: [Vec<u8>; 6] = [
        "inputs",
        "implementation",
        "layout",
        "user",
        "resource",
        "claim",
    ]
    .map(|n| literal(&format!("{prefix}-{n}")));
    let records: Vec<_> = bytes[..4]
        .iter()
        .map(|b| Evidence::parse_oracle(b).unwrap())
        .collect();
    let resource = OperationalResourceEvidence::decode(&bytes[4], Mode::Oracle).unwrap();
    let claim = ObservedRecord::decode(RecordKind::Claim, &bytes[5], Mode::Oracle).unwrap();
    let bundle = CurrentEvidenceBundle {
        non_resource: [&records[0], &records[1], &records[2], &records[3]],
        resource: &resource,
        claim: &claim,
    };
    check(&bundle);
}
#[test]
fn d7_cpu_wire_current_bundle() {
    for p in ["0-0", "0-1", "1-0", "1-1"] {
        with_current(p, |b| {
            b.check().unwrap();
            assert_eq!(
                b.resource.canonical().unwrap(),
                literal(&format!("{p}-resource"))
            );
        });
    }
    for p in ["0-0", "0-1"] {
        with_current(p, |bundle| {
            let mut user = bundle.non_resource[3].clone();
            let Tail::User(u) = &mut user.tail else {
                unreachable!()
            };
            let mut scope =
                Auxiliary::decode(AuxKind::UserScope, u.authority_bytes.0, Mode::Oracle).unwrap();
            let AuxiliaryBody::UserScope(s) = &mut scope.body else {
                unreachable!()
            };
            s.scopes[0].overlap = Tag(1);
            let scope = scope.canonical().unwrap();
            let sh = sha256_hex(&scope);
            let mut current =
                Auxiliary::decode(AuxKind::UserCurrent, u.current_bytes.0, Mode::Oracle).unwrap();
            let AuxiliaryBody::UserCurrent(c) = &mut current.body else {
                unreachable!()
            };
            c.authority_sha256 = H(&sh);
            let current = current.canonical().unwrap();
            let ch = sha256_hex(&current);
            u.authority_bytes = Bytes(&scope);
            u.authority_sha256 = H(&sh);
            u.authority_file.length = scope.len() as u64;
            u.authority_file.sha256 = H(&sh);
            u.current_bytes = Bytes(&current);
            u.current_sha256 = H(&ch);
            u.current_file.length = current.len() as u64;
            u.current_file.sha256 = H(&ch);
            let raw = user.canonical().unwrap();
            let user = Evidence::parse_oracle(&raw).unwrap();
            let mut non_resource = bundle.non_resource;
            non_resource[3] = &user;
            let wrong = CurrentEvidenceBundle {
                non_resource,
                resource: bundle.resource,
                claim: bundle.claim,
            };
            assert!(wrong.check().is_err());
        });
    }
}
#[test]
fn d7_cpu_wire_quick_v2() {
    for p in ["0-1", "1-1"] {
        for kind in [EventKind::Preflight, EventKind::Start, EventKind::End] {
            let b = literal(&format!("{p}-{}", kind.name()));
            let event = QuickEventV2::decode(kind, &b, Mode::Oracle).unwrap();
            assert_eq!(event.canonical().unwrap(), b);
            assert!(QuickEventV2::decode(kind, &b, Mode::Production).is_err());
            let raw = literal(&format!("{p}-{}-json", kind.name()));
            assert_eq!(decode_quick_v2_json(&raw, Mode::Oracle).unwrap(), b);
            assert_eq!(
                decode_quick_v2_json(&event.json().unwrap(), Mode::Oracle).unwrap(),
                b
            );
            let mut wrong = b.clone();
            let offset = wrong.windows(3).position(|v| v == b"-v2").unwrap();
            wrong[offset + 2] = b'1';
            assert!(QuickEventV2::decode(kind, &wrong, Mode::Oracle).is_err());
        }
        with_current(p, |b| {
            check_quick_v2_chain(
                &literal(&format!("{p}-preflight-json")),
                &literal(&format!("{p}-start-json")),
                &literal(&format!("{p}-end-json")),
                b,
            )
            .unwrap()
        });
    }
}
#[test]
fn d7_cpu_wire_result_json() {
    for p in ["0-1", "1-1"] {
        let bin = literal(&format!("{p}-result"));
        let raw = literal(&format!("{p}-result-json"));
        check_result_json(&raw, &bin, Mode::Oracle).unwrap();
        check_three_outputs(&bin, &raw, &raw, Mode::Oracle).unwrap();
        assert!(check_result_json(&raw, &bin, Mode::Production).is_err());
        let mut extra = raw.clone();
        extra.extend_from_slice(b" true");
        assert!(check_result_json(&extra, &bin, Mode::Oracle).is_err());
        let text = std::str::from_utf8(&raw).unwrap();
        for change in [
            text.replacen('{', "{\"extra\":0,", 1),
            text.replacen('{', "{\"schema\":\"duplicate\",", 1),
            text.replace("\"completed_updates\":2", "\"completed_updates\":2.0"),
            text.replace("\"completed_updates\":2", "\"completed_updates\":-1"),
            text.replace("\"completed_updates\":2", "\"completed_updates\":\"2\""),
        ] {
            assert!(check_result_json(change.as_bytes(), &bin, Mode::Oracle).is_err());
        }
        let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["payload"]["checkpoints"][0]["weight_max_abs_diff_bits"] =
            serde_json::json!(f64::NAN.to_bits());
        assert!(
            check_result_json(&serde_json::to_vec(&value).unwrap(), &bin, Mode::Oracle).is_err()
        );
        let rec = ObservedRecord::decode(RecordKind::Result, &bin, Mode::Oracle).unwrap();
        let hash = sha256_hex(&bin);
        let link = FileRef {
            root: Tag(2),
            path: Str("result.bin"),
            length: bin.len() as u64,
            sha256: H(&hash),
        };
        let authored = result_json(&rec, &link).unwrap();
        check_result_json(&authored, &bin, Mode::Oracle).unwrap();
        let mut changed = authored.clone();
        changed.push(b'\n');
        check_result_json(&changed, &bin, Mode::Oracle).unwrap();
        assert!(check_three_outputs(&bin, &raw, &changed, Mode::Oracle).is_err());
    }
    let b = literal("initial-sample");
    assert_eq!(
        ObservedRecord::decode(RecordKind::InitialSample, &b, Mode::Oracle)
            .unwrap()
            .canonical()
            .unwrap(),
        b
    );
    check_sample_stream(&literal("samples"), Mode::Oracle).unwrap();
    assert!(check_sample_stream(&b, Mode::Oracle).is_err());
    assert!(
        ObservedRecord::decode(RecordKind::InitialSample, &literal("samples"), Mode::Oracle)
            .is_err()
    );
    let b = literal("policy-json");
    ObservedPolicy::decode(&b, Mode::Oracle).unwrap();
    assert!(ObservedPolicy::decode(&b, Mode::Production).is_err());
}
#[test]
fn d7_cpu_wire_terminal_links() {
    for p in ["0-1", "1-1"] {
        with_current(p, |bundle| {
            let result = literal(&format!("{p}-result"));
            let json = literal(&format!("{p}-result-json"));
            let preflight = literal(&format!("{p}-preflight-json"));
            let start = literal(&format!("{p}-start-json"));
            let end = literal(&format!("{p}-end-json"));
            let supervisor = literal(&format!("{p}-supervisor-end"));
            let terminal = QuickTerminal {
                result: &result,
                result_json: &json,
                result_export: &json,
                preflight: &preflight,
                start: &start,
                end: &end,
                supervisor_end: &supervisor,
            };
            terminal.check(bundle).unwrap();
            let mut wrong = supervisor.clone();
            let n = wrong.len();
            wrong[n - 19] = 0;
            assert!(
                QuickTerminal {
                    supervisor_end: &wrong,
                    ..terminal
                }
                .check(bundle)
                .is_err()
            );
        });
    }
}
#[test]
fn d7_cpu_wire_target_coordinate_and_writer_order() {
    let b = literal("0-1-result");
    let mut rec = ObservedRecord::decode(RecordKind::Result, &b, Mode::Oracle).unwrap();
    let hash = sha256_hex(&b);
    let link = FileRef {
        root: Tag(2),
        path: Str("result.bin"),
        length: b.len() as u64,
        sha256: H(&hash),
    };
    assert_eq!(
        result_json(&rec, &link).unwrap(),
        literal("0-1-result-json"),
        "R-WIRE-3 requires listed JSON writer order"
    );
    let RecordBody::Result(r) = &mut rec.body else {
        unreachable!()
    };
    r.scope = Tag(0);
    r.status = Tag(2);
    r.completed_updates = 6;
    r.processed_samples = 120;
    r.checkpoints.truncate(1);
    r.checkpoints[0].updates = 6;
    r.checkpoints[0].samples = 120;
    r.checkpoints[0].crossed = vec![1];
    r.final_endpoint = None;
    r.model_digest = None;
    r.target = Some(TargetOutcome {
        lock: link,
        threshold_bits: 0,
        reached: true,
        first_checkpoint: Some(6),
        time_basis: Id("cumulative_update_ns"),
        time_ns: Some(10),
        censor_ns: 10,
    });
    r.validate(Mode::Oracle).unwrap();
    for ordinal in [0, 1] {
        r.target.as_mut().unwrap().first_checkpoint = Some(ordinal);
        assert!(r.validate(Mode::Oracle).is_err());
    }
}
#[test]
fn d7_cpu_wire_closed_domains_and_caps() {
    for (name, kind) in [
        ("bootstrap", RecordKind::Bootstrap),
        ("pending", RecordKind::Pending),
        ("claim", RecordKind::Claim),
        ("host-pending", RecordKind::HostPending),
        ("host-closed", RecordKind::HostClosed),
        ("initial-sample", RecordKind::InitialSample),
        ("supervisor-end", RecordKind::SupervisorEnd),
        ("full-end", RecordKind::FullEnd),
        ("result", RecordKind::Result),
    ] {
        let b = literal(name);
        for n in 0..b.len() {
            assert!(
                ObservedRecord::decode(kind, &b[..n], Mode::Oracle).is_err(),
                "{name}/{n}"
            );
        }
        let mut wrong = b.clone();
        wrong[49] = b'x';
        assert!(ObservedRecord::decode(kind, &wrong, Mode::Oracle).is_err());
    }
    for kind in [
        RecordKind::Bootstrap,
        RecordKind::HostPending,
        RecordKind::HostClosed,
    ] {
        assert_eq!(
            ObservedRecord::decode(kind, &[0; 4097], Mode::Oracle)
                .unwrap_err()
                .code(),
            DocumentErrorCode::TooLarge
        );
    }
    let mut result = literal("0-1-result");
    let marker = [2u64, 1, 20, 0]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect::<Vec<_>>();
    let offset = result
        .windows(marker.len())
        .position(|v| v == marker)
        .unwrap();
    result[offset..offset + 8].copy_from_slice(&401u64.to_le_bytes());
    assert_eq!(
        ObservedRecord::decode(RecordKind::Result, &result, Mode::Oracle)
            .unwrap_err()
            .code(),
        DocumentErrorCode::TooLarge
    );
    let mut b = literal("demand");
    b[17..25].copy_from_slice(&513u64.to_le_bytes());
    assert!(SourceDemand::read(&mut Cursor::new(&b)).is_err());
    let b = literal("demand");
    let mut d = SourceDemand::read(&mut Cursor::new(&b)).unwrap();
    d.allocations[0].site = Id("raw_18446744073709551616");
    assert!(d.validate().is_err());
    for site in ["raw_00", "raw_-1", "raw_x", "foreign", "decoded_f64_01"] {
        d.allocations[0].site = Id(site);
        assert!(d.validate().is_err());
    }
    let b = literal("policy-json");
    let text = std::str::from_utf8(&b).unwrap();
    for wrong in [
        text.replacen('{', "{\"unknown\":0,", 1),
        text.replacen('{', "{\"schema\":\"duplicate\",", 1),
        text.replace("94371840", "94371840.0"),
        format!("{text} true"),
    ] {
        assert!(ObservedPolicy::decode(wrong.as_bytes(), Mode::Oracle).is_err());
    }
    for p in ["0-1", "1-1"] {
        let b = literal(&format!("{p}-start"));
        let mut e = QuickEventV2::decode(EventKind::Start, &b, Mode::Oracle).unwrap();
        let QuickV2Body::Start(s) = &mut e.body else {
            unreachable!()
        };
        s.gates.resource.policy_revision = Text("test-d7-resource-v1".into());
        assert!(e.canonical().is_err());
        assert!(QuickEvent::parse_binary_oracle(EventKind::Start, &b).is_err());
    }
    let b = super::tests::fixture("quick.start.bin");
    assert!(QuickEventV2::decode(EventKind::Start, &b, Mode::Oracle).is_err());
    let b = literal("0-1-resource");
    assert!(Evidence::parse_oracle(&b).is_err());
    let b = super::tests::fixture("quick.resource.evidence.bin");
    assert!(OperationalResourceEvidence::decode(&b, Mode::Oracle).is_err());
    assert_eq!(
        export_id(1, &"d".repeat(64), Mode::Oracle).unwrap(),
        sha256_hex(&literal("export-id"))
    );
    assert!(check_transport_length(A_MAX + 1, false).is_err());
    assert!(check_transport_length(QUICK_K_MAX + 1, true).is_err());
}
#[test]
fn d7_cpu_wire_full_terminal_records() {
    let b = literal("full-end");
    let mut end = ObservedRecord::decode(RecordKind::FullEnd, &b, Mode::Oracle).unwrap();
    let RecordBody::FullEnd(e) = &mut end.body else {
        unreachable!()
    };
    e.result_json = None;
    assert!(end.validate().is_err());
    let mut end = ObservedRecord::decode(RecordKind::FullEnd, &b, Mode::Oracle).unwrap();
    let RecordBody::FullEnd(e) = &mut end.body else {
        unreachable!()
    };
    e.processed_samples -= 1;
    assert!(end.validate().is_err());
    let b = literal("0-1-end");
    let mut end = QuickEventV2::decode(EventKind::End, &b, Mode::Oracle).unwrap();
    let QuickV2Body::End(e) = &mut end.body else {
        unreachable!()
    };
    e.result_export = None;
    assert!(end.canonical().is_err());
    let b = literal("0-1-result");
    let mut result = ObservedRecord::decode(RecordKind::Result, &b, Mode::Oracle).unwrap();
    let RecordBody::Result(r) = &mut result.body else {
        unreachable!()
    };
    r.poisoned = true;
    assert!(r.validate(Mode::Oracle).is_err());
    r.status = Tag(1);
    assert!(r.validate(Mode::Oracle).is_err());
    r.final_endpoint = None;
    r.model_digest = None;
    r.validate(Mode::Oracle).unwrap();
    r.checkpoints[0].weight_max_abs_diff_bits = f64::NAN.to_bits();
    assert!(r.validate(Mode::Oracle).is_err());
    r.checkpoints[0].weight_max_abs_diff_bits = 0;
    for endpoint in &mut r.checkpoints {
        endpoint.counter.values[0] = u64::MAX;
    }
    assert!(r.validate(Mode::Oracle).is_err());
}
