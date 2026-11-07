//! Independent literal wire tests only. No registry, planning, P1, HE or native calls.
use super::super::fixture_literals::target as target_literal;
use super::*;
use serde_json::Value;
const ORACLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../results/validation/task54-d7-gate-schema-design/task-54-d7-gate-schema-vectors/"
);
pub(super) fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{ORACLE}{name}")).unwrap()
}
fn refusal<T>(result: Result<T>, code: DocumentErrorCode) {
    match result {
        Ok(_) => panic!("unexpected acceptance"),
        Err(e) => assert_eq!(e.code(), code, "{e}"),
    }
}
#[test]
fn d7_evidence_independent_all_ten_branches() {
    for scope in ["full", "quick"] {
        for gate in ["inputs", "implementation", "layout", "resource", "user"] {
            let bytes = fixture(&format!("{scope}.{gate}.evidence.bin"));
            let parsed = Evidence::parse_oracle(&bytes).unwrap();
            assert_eq!(parsed.canonical().unwrap(), bytes);
            refusal(
                Evidence::parse(&bytes),
                DocumentErrorCode::UnsupportedSchema,
            );
        }
    }
}
#[test]
fn d7_evidence_auxiliary_and_quick_exact_json() {
    for scope in ["full", "quick"] {
        for kind in [
            AuxKind::Request,
            AuxKind::Consumption,
            AuxKind::UserScope,
            AuxKind::UserCurrent,
        ] {
            let bytes = fixture(&format!("{scope}.{}.bin", kind.fixture_name()));
            let parsed = Auxiliary::parse_oracle(kind, &bytes).unwrap();
            assert_eq!(parsed.canonical().unwrap(), bytes);
            refusal(
                Auxiliary::parse(kind, &bytes),
                DocumentErrorCode::UnsupportedSchema,
            );
        }
    }
    let identity = fixture("quick.identity.bin");
    assert_eq!(
        QuickIdentity::parse_oracle(&identity)
            .unwrap()
            .canonical()
            .unwrap(),
        identity
    );
    refusal(
        QuickIdentity::parse(&identity),
        DocumentErrorCode::UnsupportedSchema,
    );
    for kind in [EventKind::Preflight, EventKind::Start, EventKind::End] {
        let raw = fixture(&format!("quick.{}.json", kind.name()));
        let parsed = QuickEvent::parse_json_oracle(&raw).unwrap();
        assert_eq!(
            parsed.canonical().unwrap(),
            fixture(&format!("quick.{}.bin", kind.name()))
        );
        assert_eq!(parsed.json().unwrap(), raw);
        assert_eq!(parsed.file_sha256(), sha256_hex(&raw));
        refusal(
            QuickEvent::parse_json(&raw),
            DocumentErrorCode::UnsupportedSchema,
        );
    }
}
#[test]
fn d7_evidence_concrete_representation_mutations() {
    use DocumentErrorCode::*;
    for (name, code) in [
        ("unknown-tag", UnsupportedTag),
        ("synthetic-bit", UnsupportedSchema),
        ("identity", DigestMismatch),
        ("input-receipt", CrossLinkMismatch),
        ("partition", CrossLinkMismatch),
        ("resource-peak", CrossLinkMismatch),
    ] {
        let bytes = fixture(&format!("negative/{name}.bin"));
        let result = if name == "partition" {
            Auxiliary::parse_oracle(AuxKind::Consumption, &bytes).map(|_| ())
        } else {
            Evidence::parse_oracle(&bytes).map(|_| ())
        };
        refusal(result, code);
    }
    for name in ["quick-seed", "quick-one-update", "quick-three-updates"] {
        refusal(
            Auxiliary::parse_oracle(
                AuxKind::Consumption,
                &fixture(&format!("negative/{name}.bin")),
            ),
            CrossLinkMismatch,
        );
    }
}
#[test]
fn d7_evidence_full_quick_exact_claims_and_nonpromotion() {
    for scope in ["full", "quick"] {
        let bytes: Vec<_> = ["inputs", "implementation", "layout", "resource", "user"]
            .map(|gate| fixture(&format!("{scope}.{gate}.evidence.bin")))
            .into();
        let records: Vec<_> = bytes
            .iter()
            .map(|b| Evidence::parse_oracle(b).unwrap())
            .collect();
        check_evidence_bundle(&records).unwrap();
        for i in 0..5 {
            let other = fixture(&format!(
                "{}.{}.evidence.bin",
                if scope == "full" { "quick" } else { "full" },
                ["inputs", "implementation", "layout", "resource", "user"][i]
            ));
            let mut mixed: Vec<_> = records.iter().collect();
            let replacement = Evidence::parse_oracle(&other).unwrap();
            mixed[i] = &replacement;
            refusal(
                check_evidence_refs(&mixed),
                DocumentErrorCode::CrossLinkMismatch,
            );
        }
    }
    let identity_bytes = fixture("quick.identity.bin");
    let mut identity = QuickIdentity::parse_oracle(&identity_bytes).unwrap();
    identity.body.operational_attempt = Id("SYNTHETIC-ATTEMPX");
    let changed = identity.canonical().unwrap();
    let hash = sha256_hex(&changed);
    let sidecar = fixture("quick.inputs.evidence.bin");
    let mut forged = Evidence::parse_oracle(&sidecar).unwrap();
    if let Branch::Quick(q) = &mut forged.branch {
        q.operational_identity = Bytes(&changed);
        q.operational_id = H(&hash);
    }
    refusal(forged.validate(), DocumentErrorCode::CrossLinkMismatch);
}
#[test]
fn d7_evidence_quick_event_links_policies_argv_and_context() {
    let raw: Vec<_> = ["preflight", "start", "end"]
        .map(|k| fixture(&format!("quick.{k}.json")))
        .into();
    let events: Vec<_> = raw
        .iter()
        .map(|r| QuickEvent::parse_json_oracle(r).unwrap())
        .collect();
    let bytes: Vec<_> = ["inputs", "implementation", "layout", "resource", "user"]
        .map(|g| fixture(&format!("quick.{g}.evidence.bin")))
        .into();
    let sidecars: Vec<_> = bytes
        .iter()
        .map(|b| Evidence::parse_oracle(b).unwrap())
        .collect();
    let argv = fixture("captured-argv.json");
    let v: Value = serde_json::from_slice(&argv).unwrap();
    // Independently captured synthetic byte argv; never obtained from parsed event.
    let captured: Vec<Vec<u8>> = v["argv_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            let h = h.as_str().unwrap();
            (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    check_quick_events(&events[0], &events[1], &events[2], &sidecars, &captured).unwrap();
    let mut tampered = captured.clone();
    tampered[2][32] ^= 1;
    refusal(
        check_quick_events(&events[0], &events[1], &events[2], &sidecars, &tampered),
        DocumentErrorCode::CrossLinkMismatch,
    );
    for (name, index) in [
        ("event-evidence", 0),
        ("event-link", 1),
        ("end-link", 2),
        ("argv-byte-tampering", 0),
    ] {
        let canonical = fixture(&format!("negative/{name}.bin"));
        let original = &events[index];
        let changed = original.with_canonical_oracle(&canonical).unwrap();
        let mut refs: [&QuickEvent; 3] = [&events[0], &events[1], &events[2]];
        refs[index] = &changed;
        refusal(
            check_quick_events(refs[0], refs[1], refs[2], &sidecars, &captured),
            DocumentErrorCode::CrossLinkMismatch,
        );
    }
    let base = fixture("quick.preflight.bin");
    for field in ["source_manifest_sha256", "source_revision"] {
        let mut changed = base.clone();
        let start = offset("quick.preflight", field);
        let length = if field == "source_revision" { 40 } else { 64 };
        changed[start + 8..start + 8 + length].fill(b'0');
        refusal(
            QuickEvent::parse_binary_oracle(EventKind::Preflight, &changed),
            DocumentErrorCode::CrossLinkMismatch,
        );
    }
    for field in [
        "approval_sha256",
        "train_receipt_sha256",
        "executable_sha256",
    ] {
        let mut changed = base.clone();
        let start = offset("quick.preflight", field);
        changed[start + 8..start + 72].fill(b'0');
        let changed = events[0].with_canonical_oracle(&changed).unwrap();
        refusal(
            check_quick_events(&changed, &events[1], &events[2], &sidecars, &captured),
            DocumentErrorCode::CrossLinkMismatch,
        );
    }
    let mut argv0 = captured.clone();
    argv0[0][0] ^= 1;
    refusal(
        check_quick_events(&events[0], &events[1], &events[2], &sidecars, &argv0),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let mut reordered = captured.clone();
    reordered.swap(0, 1);
    refusal(
        check_quick_events(&events[0], &events[1], &events[2], &sidecars, &reordered),
        DocumentErrorCode::CrossLinkMismatch,
    );
    refusal(
        QuickEvent::parse_binary_oracle(EventKind::End, &fixture("negative/end-incomplete.bin")),
        DocumentErrorCode::CrossLinkMismatch,
    );
}
#[test]
fn d7_evidence_strict_json_and_caps() {
    use DocumentErrorCode::*;
    let raw = fixture("quick.start.json");
    let text = String::from_utf8(raw.clone()).unwrap();
    let duplicate = text.replacen("{", "{\"\\u0073chema\":\"hekit-test-quick-start-v1\",", 1);
    refusal(
        QuickEvent::parse_json_oracle(duplicate.as_bytes()),
        DuplicateField,
    );
    let unknown = text.replacen("{", "{\"extra\":0,", 1);
    refusal(
        QuickEvent::parse_json_oracle(unknown.as_bytes()),
        UnknownField,
    );
    let escaped = text.replacen("\"canonical_bytes\":\"", "\"canonical_bytes\":\"\\u0030", 1);
    refusal(QuickEvent::parse_json_oracle(escaped.as_bytes()), BadHex);
    let wrong_type = text.replacen("\"synthetic\":true", "\"synthetic\":\"true\"", 1);
    refusal(
        QuickEvent::parse_json_oracle(wrong_type.as_bytes()),
        BadJson,
    );
    let mut trailing = raw;
    trailing.extend_from_slice(b"false");
    refusal(QuickEvent::parse_json_oracle(&trailing), BadJson);
    check_transport_length(A_MAX, false).unwrap();
    refusal(check_transport_length(A_MAX + 1, false), TooLarge);
    check_transport_length(QUICK_K_MAX, true).unwrap();
    refusal(check_transport_length(QUICK_K_MAX + 1, true), TooLarge);
    let b = fixture("quick.inputs.evidence.bin");
    let mut extra = b.clone();
    extra.push(0);
    refusal(Evidence::parse_oracle(&extra), OutOfRange);
    for n in [0, 1, 40, b.len() - 1] {
        refusal(Evidence::parse_oracle(&b[..n]), OutOfRange);
    }
}
#[test]
fn d7_evidence_request_consumption_retained_comparisons() {
    let request = fixture("quick.request.bin");
    let consumption = fixture("quick.consumption.bin");
    let r = Auxiliary::parse_oracle(AuxKind::Request, &request).unwrap();
    let c = Auxiliary::parse_oracle(AuxKind::Consumption, &consumption).unwrap();
    check_request_consumption(&r, &c).unwrap();
    let changed = fixture("negative/request.bin");
    let bad = Auxiliary::parse_oracle(AuxKind::Request, &changed).unwrap();
    refusal(
        check_request_consumption(&bad, &c),
        DocumentErrorCode::CrossLinkMismatch,
    );
    for choice in 0..3 {
        let mut request = r.clone();
        if let AuxiliaryBody::Request(body) = &mut request.body {
            match choice {
                0 => body.work[2] = 1,
                1 => body.limits[1] = 1,
                _ => body.physical_lanes = Some(1),
            }
        }
        let raw = request.canonical().unwrap();
        let request = Auxiliary::parse_oracle(AuxKind::Request, &raw).unwrap();
        let digest = sha256_hex(&raw);
        let mut consumed = c.clone();
        if let AuxiliaryBody::Consumption(body) = &mut consumed.body {
            body.prefix.request_sha256 = H(&digest);
        }
        let raw_consumed = consumed.canonical().unwrap();
        let consumed = Auxiliary::parse_oracle(AuxKind::Consumption, &raw_consumed).unwrap();
        let error = check_request_consumption(&request, &consumed).unwrap_err();
        assert_eq!(error.code(), DocumentErrorCode::CrossLinkMismatch);
        assert_eq!(
            error.path(),
            if choice == 2 {
                "consumption.partition"
            } else {
                "consumption.work_limit"
            }
        );
    }
    let mut swapped = c.clone();
    if let AuxiliaryBody::Consumption(body) = &mut swapped.body {
        if let ConsumptionBody::Quick(updates) = &mut body.body {
            updates.swap(0, 1);
        }
    }
    refusal(swapped.canonical(), DocumentErrorCode::CrossLinkMismatch);
    let changed = fixture("negative/order.bin");
    let bad = Auxiliary::parse_oracle(AuxKind::Consumption, &changed).unwrap();
    refusal(
        check_consumption_exact(&bad, &c),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let changed = fixture("negative/full-consumption.bin");
    refusal(
        Auxiliary::parse_oracle(AuxKind::Consumption, &changed),
        DocumentErrorCode::OutOfRange,
    );
}
#[test]
fn d7_evidence_authority_resource_and_closed_branches() {
    exercise_authority_resource_branches();
    exercise_overlap_and_retry_refusals();
}
fn exercise_authority_resource_branches() {
    use DocumentErrorCode::*;
    let sbytes = fixture("quick.authority.scope.bin");
    let cbytes = fixture("quick.authority.current.bin");
    let scope = Auxiliary::parse_oracle(AuxKind::UserScope, &sbytes).unwrap();
    let current = Auxiliary::parse_oracle(AuxKind::UserCurrent, &cbytes).unwrap();
    check_authority(&scope, &current, 0).unwrap();
    for name in ["authority-revoked", "authority-generation"] {
        let changed = fixture(&format!("negative/{name}.bin"));
        let c = Auxiliary::parse_oracle(AuxKind::UserCurrent, &changed).unwrap();
        refusal(check_authority(&scope, &c, 0), CrossLinkMismatch);
    }
    for revoked in [vec![0], vec![1], vec![0, 0], vec![1, 0]] {
        let mut c = current.clone();
        if let AuxiliaryBody::UserCurrent(v) = &mut c.body {
            v.revoked_scopes = revoked;
        }
        refusal(check_authority(&scope, &c, 0), CrossLinkMismatch);
    }
    let mut held = scope.clone();
    if let AuxiliaryBody::UserScope(s) = &mut held.body {
        s.holds.push(Hold {
            branch: s.scopes[0].branch,
            registration_id: s.scopes[0].registration_id,
            reason: Tag(1),
        });
    }
    // Original bytes digest is unchanged here deliberately: supplied record
    // consistency, not a claim that the capture has legitimate source origin.
    refusal(check_authority(&held, &current, 0), CrossLinkMismatch);
    if let AuxiliaryBody::UserScope(s) = &mut held.body {
        s.holds.last_mut().unwrap().reason = Tag(0);
    }
    refusal(check_authority(&held, &current, 0), CrossLinkMismatch);
    let bytes = fixture("quick.resource.evidence.bin");
    let evidence = Evidence::parse_oracle(&bytes).unwrap();
    let r = match &evidence.tail {
        Tail::Resource(r) => r,
        _ => panic!("resource fixture"),
    };
    r.check_supplied_capacity().unwrap();
    let mut partial = r.requirements.clone();
    for c in &mut partial.components {
        c.phase_mask = Tag(0);
        c.resident_bytes = 0;
        c.additional_bytes = 0;
    }
    partial.components[0].phase_mask = Tag(1);
    partial.components[0].resident_bytes = 7;
    partial.components[0].additional_bytes = 3;
    partial.components[1].phase_mask = Tag(2);
    partial.components[1].resident_bytes = 2;
    partial.components[1].additional_bytes = 4;
    partial.peak_host_bytes = 10;
    partial.incremental_host_bytes = 4;
    partial.validate().unwrap();
    partial.components[1].kind = Tag(0);
    refusal(partial.validate(), CrossLinkMismatch);
    let x = admission::parse_execution_binary(evidence.common.execution.0).unwrap();
    let case = admission::parse_case_binary(evidence.branch.registration()).unwrap();
    let mut no_model = r.clone();
    no_model.model.path = Str("");
    refusal(
        no_model.validate(&evidence, &x.fields(), &case),
        CrossLinkMismatch,
    );
    let mut missing = r.requirements.clone();
    missing.components.pop();
    refusal(missing.validate(), CrossLinkMismatch);
    let mut phase_overflow = r.requirements.clone();
    phase_overflow.components[0].resident_bytes = u64::MAX;
    phase_overflow.components[0].additional_bytes = 0;
    refusal(phase_overflow.validate(), TooLarge);
    let mut overflow = r.requirements.clone();
    overflow.components[0].resident_bytes = u64::MAX;
    overflow.components[0].additional_bytes = 1;
    refusal(overflow.validate(), TooLarge);
    let mut inactive = r.requirements.clone();
    inactive.components[4].additional_bytes = 1;
    refusal(inactive.validate(), CrossLinkMismatch);
    let mut increment = r.requirements.clone();
    increment.incremental_host_bytes += 1;
    refusal(increment.validate(), CrossLinkMismatch);
    let mut capacity = r.clone();
    capacity.observation.host_mem_available = 0;
    refusal(capacity.check_supplied_capacity(), CrossLinkMismatch);
    let mut capacity = r.clone();
    capacity.claim.other_claimed_host_bytes = u64::MAX;
    refusal(capacity.check_supplied_capacity(), TooLarge);
    let mut capacity = r.clone();
    capacity.observation.cgroups[0].limit = Some(0);
    refusal(capacity.check_supplied_capacity(), CrossLinkMismatch);
    let mut capacity = r.clone();
    capacity.observation.disk_available = 0;
    refusal(capacity.check_supplied_capacity(), CrossLinkMismatch);
    let mut capacity = r.clone();
    capacity.observation.cgroups[0].limit = None;
    capacity.check_supplied_capacity().unwrap();
    // Resident bytes are already reflected in capacity: compare incremental,
    // not total peak, without claiming this arithmetic proves actual residency.
    let mut capacity = r.clone();
    capacity.observation.host_mem_available = capacity.requirements.incremental_host_bytes;
    capacity.observation.cgroups.clear();
    capacity.check_supplied_capacity().unwrap();
}
// Encode an inert negative after updating BOTH administrative payloads, their
// hashes/lengths, and both FileRefs. This bypasses only the encoder's optional
// eager validation; the production parsing checks under test still run below.
fn rebind_user_scope_for_negative(evidence: &Evidence<'_>, scope: &Auxiliary<'_>) -> Vec<u8> {
    let authority_bytes = scope.canonical().unwrap();
    let authority_hash = sha256_hex(&authority_bytes);
    let Tail::User(original) = &evidence.tail else {
        panic!("user fixture")
    };
    let mut current =
        Auxiliary::parse_oracle(AuxKind::UserCurrent, original.current_bytes.0).unwrap();
    if let AuxiliaryBody::UserCurrent(c) = &mut current.body {
        c.authority_sha256 = H(&authority_hash);
    }
    let current_bytes = current.canonical().unwrap();
    let current_hash = sha256_hex(&current_bytes);
    let mut user = original.clone();
    user.authority_bytes = Bytes(&authority_bytes);
    user.authority_sha256 = H(&authority_hash);
    user.current_bytes = Bytes(&current_bytes);
    user.current_sha256 = H(&current_hash);
    user.authority_file.length = authority_bytes.len() as u64;
    user.authority_file.sha256 = H(&authority_hash);
    user.current_file.length = current_bytes.len() as u64;
    user.current_file.sha256 = H(&current_hash);
    let mut encoded = record_encoder("gate-user-evidence", Mode::Oracle).unwrap();
    evidence.common.write(&mut encoded).unwrap();
    evidence.branch.write(&mut encoded).unwrap();
    user.write(&mut encoded).unwrap();
    encoded.bytes
}
fn exercise_overlap_and_retry_refusals() {
    let bytes: Vec<_> = GATE_NAMES
        .iter()
        .map(|g| fixture(&format!("quick.{g}.evidence.bin")))
        .collect();
    let records: Vec<_> = bytes
        .iter()
        .map(|b| Evidence::parse_oracle(b).unwrap())
        .collect();
    check_evidence_bundle(&records).unwrap();
    let Tail::User(user) = &records[4].tail else {
        panic!("user fixture")
    };
    for (selected_overlap, claimed_overlap) in [(1, 0), (0, 1)] {
        let mut scope =
            Auxiliary::parse_oracle(AuxKind::UserScope, user.authority_bytes.0).unwrap();
        if let AuxiliaryBody::UserScope(s) = &mut scope.body {
            s.scopes[0].overlap = Tag(selected_overlap);
        }
        let user_bytes = rebind_user_scope_for_negative(&records[4], &scope);
        // Valid selected scope/current/digests; overlap is a legal tag in both records.
        let selected = Evidence::parse_oracle(&user_bytes).unwrap();
        let mut resource = records[3].clone();
        if let Tail::Resource(r) = &mut resource.tail {
            r.claim.overlap = Tag(claimed_overlap);
        }
        let resource_bytes = resource.canonical().unwrap();
        let claim = Evidence::parse_oracle(&resource_bytes).unwrap();
        let mut mixed: Vec<_> = records.iter().collect();
        mixed[3] = &claim;
        mixed[4] = &selected;
        let error = check_evidence_refs(&mixed).unwrap_err();
        assert_eq!(error.code(), DocumentErrorCode::CrossLinkMismatch);
        assert_eq!(error.path(), "evidence.bundle.overlap");
    }
    let full_bytes = fixture("full.user.evidence.bin");
    let full = Evidence::parse_oracle(&full_bytes).unwrap();
    let Tail::User(full_user) = &full.tail else {
        panic!("full user fixture")
    };
    let Branch::Full(full_body) = &full.branch else {
        panic!("full branch")
    };
    // A syntactically valid fictional predecessor. Its Link is a claim only;
    // no retained prior admission or actual Retry authority is asserted here.
    let predecessor = admission::Predecessor {
        attempt_id: Text("SYNTHETIC-PRIOR".into()),
        admission: full_body.lock.clone(),
        start: None,
    };
    let mut scope =
        Auxiliary::parse_oracle(AuxKind::UserScope, full_user.authority_bytes.0).unwrap();
    if let AuxiliaryBody::UserScope(s) = &mut scope.body {
        s.scopes[0].predecessor = Some(predecessor.clone());
    }
    let mismatched = rebind_user_scope_for_negative(&full, &scope);
    let error = Evidence::parse_oracle(&mismatched).unwrap_err();
    assert_eq!(error.code(), DocumentErrorCode::CrossLinkMismatch);
    assert_eq!(error.path(), "user.retry.predecessor");
    let mut quick = Auxiliary::parse_oracle(AuxKind::UserScope, user.authority_bytes.0).unwrap();
    if let AuxiliaryBody::UserScope(s) = &mut quick.body {
        s.scopes[0].predecessor = Some(predecessor);
    }
    let mut encoded = record_encoder("user-scope", Mode::Oracle).unwrap();
    if let AuxiliaryBody::UserScope(scope) = &quick.body {
        scope.write(&mut encoded).unwrap();
    }
    let error = Auxiliary::parse_oracle(AuxKind::UserScope, &encoded.bytes).unwrap_err();
    assert_eq!(error.code(), DocumentErrorCode::CrossLinkMismatch);
    assert_eq!(error.path(), "authority.quick.retry");
}

fn literal_manifest() -> Value {
    serde_json::from_slice(&fixture("literal-inputs.json")).unwrap()
}
fn offset(name: &str, path: &str) -> usize {
    literal_manifest()["records"][name]["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["path"] == path)
        .unwrap()["offset"]
        .as_u64()
        .unwrap() as usize
}
fn parse_literal(name: &str, bytes: &[u8]) -> Result<Vec<u8>> {
    if name.contains(".evidence") {
        return Evidence::parse_oracle(bytes)?.canonical();
    }
    for (suffix, kind) in [
        ("request", AuxKind::Request),
        ("consumption", AuxKind::Consumption),
        ("authority.scope", AuxKind::UserScope),
        ("authority.current", AuxKind::UserCurrent),
    ] {
        if name.ends_with(suffix) {
            return Auxiliary::parse_oracle(kind, bytes)?.canonical();
        }
    }
    if name == "quick.identity" {
        return QuickIdentity::parse_oracle(bytes)?.canonical();
    }
    for kind in [EventKind::Preflight, EventKind::Start, EventKind::End] {
        if name == format!("quick.{}", kind.name()) {
            return QuickEvent::parse_binary_oracle(kind, bytes)?.canonical();
        }
    }
    if name == "registration" {
        let case = admission::parse_case_binary(bytes)?;
        return admission::component(Some("hekit-study-registered-case-v1"), &case);
    }
    let x = admission::parse_execution_binary(bytes)?;
    x.fields().binding_bytes()
}
#[test]
fn d7_evidence_all_literals_closed_tags_and_trailing() {
    let m = literal_manifest();
    for (name, record) in m["records"].as_object().unwrap() {
        let bytes = fixture(&format!("{name}.bin"));
        assert_eq!(parse_literal(name, &bytes).unwrap(), bytes);
        for field in record["fields"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|f| f["type"] == "tag")
        {
            let mut changed = bytes.clone();
            changed[field["offset"].as_u64().unwrap() as usize] = 255;
            refusal(
                parse_literal(name, &changed),
                DocumentErrorCode::UnsupportedTag,
            );
        }
        for field in record["fields"].as_array().unwrap() {
            let end = field["offset"].as_u64().unwrap() as usize;
            refusal(
                parse_literal(name, &bytes[..end]),
                DocumentErrorCode::OutOfRange,
            );
        }
        let mut changed = bytes.clone();
        changed.push(0);
        refusal(parse_literal(name, &changed), DocumentErrorCode::OutOfRange);
    }
}
#[test]
fn d7_evidence_target_expected_claims_and_file_identity() {
    let full = fixture("full.inputs.evidence.bin");
    let expected = Evidence::parse_oracle(&full).unwrap();
    for name in ["target-link", "threshold-placeholder"] {
        let bytes = fixture(&format!("negative/{name}.bin"));
        let other = Evidence::parse_oracle(&bytes).unwrap();
        refusal(
            check_full_target_claims(&other, &expected),
            DocumentErrorCode::CrossLinkMismatch,
        );
    }
    check_full_target_claims(&expected, &expected).unwrap();
    let quick = fixture("quick.inputs.evidence.bin");
    let quick = Evidence::parse_oracle(&quick).unwrap();
    refusal(
        check_full_target_claims(&quick, &expected),
        DocumentErrorCode::CrossLinkMismatch,
    );
    if let Tail::Inputs(inputs) = &expected.tail {
        let file = &inputs.source_files[0];
        let raw = fixture(&format!("retained/{}", file.path.0));
        file.check_bytes(&raw).unwrap();
        let mut changed = raw;
        changed[0] ^= 1;
        refusal(
            file.check_bytes(&changed),
            DocumentErrorCode::CrossLinkMismatch,
        );
        let mut identity = inputs.selected_receipt.clone();
        identity.canonical_length += 1;
        refusal(
            identity.equals(&inputs.selected_receipt),
            DocumentErrorCode::CrossLinkMismatch,
        );
    } else {
        panic!("inputs")
    }
}
#[test]
fn d7_evidence_implementation_ni_mandatory_and_missing() {
    let bytes = fixture("full.implementation.evidence.bin");
    let e = Evidence::parse_oracle(&bytes).unwrap();
    let original = match &e.tail {
        Tail::Implementation(i) => i,
        _ => panic!("implementation"),
    };
    let mut case = admission::parse_case_binary(e.branch.registration()).unwrap();
    let execution = admission::parse_execution_binary(e.common.execution.0).unwrap();
    let mut x = execution.fields();
    let mut missing = original.clone();
    missing.correctness.clear();
    refusal(
        missing.validate(&e, &x, &case),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let mut missing = original.clone();
    missing.security.estimator_sources.clear();
    refusal(
        missing.validate(&e, &x, &case),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let mut missing = original.clone();
    missing.security.instances.clear();
    refusal(
        missing.validate(&e, &x, &case),
        DocumentErrorCode::CrossLinkMismatch,
    );
    case.mode = crate::study_profile::CaseMode::Noninteractive;
    x.bootstrap = Some(x.parameter);
    let mut ni = original.clone();
    ni.applicability.bootstrap = Some(H(x.parameter));
    ni.applicability.mode = Tag(1);
    ni.correctness[0].applicability = ni.applicability.clone();
    let mut ni_report = ni.correctness[0].clone();
    ni_report.kind = Tag(1);
    ni.correctness.push(ni_report);
    let mut ephemeral = ni.security.instances[0].clone();
    ephemeral.role = Tag(1);
    ni.security.instances.push(ephemeral);
    ni.construction_obligations.push(Tag(4));
    ni.validate(&e, &x, &case).unwrap();
    ni.correctness.swap(0, 1);
    refusal(
        ni.validate(&e, &x, &case),
        DocumentErrorCode::CrossLinkMismatch,
    );
}
#[test]
fn d7_evidence_quick_failed_outcomes_and_binary_argv() {
    let original = fixture("quick.end.bin");
    let outcome = offset("quick.end", "outcome");
    let completed = offset("quick.end", "completed_updates");
    let failure = offset("quick.end", "failure.present");
    for phase in 0..4 {
        for code in 0..7 {
            for updates in 0u64..=2 {
                let mut failed = original.clone();
                failed[outcome] = 1;
                failed[completed..completed + 8].copy_from_slice(&updates.to_le_bytes());
                failed[failure] = 1;
                failed.splice(failure + 1..failure + 1, [phase, code]);
                let parsed = QuickEvent::parse_binary_oracle(EventKind::End, &failed).unwrap();
                assert_eq!(parsed.canonical().unwrap(), failed);
            }
        }
    }
    for invalid_tags in [[4, 0], [0, 7]] {
        let mut invalid = original.clone();
        invalid[outcome] = 1;
        invalid[failure] = 1;
        invalid.splice(failure + 1..failure + 1, invalid_tags);
        let error = QuickEvent::parse_binary_oracle(EventKind::End, &invalid).unwrap_err();
        assert_eq!(error.code(), DocumentErrorCode::UnsupportedTag);
        assert_eq!(error.path(), "execution.tag");
    }

    let mut failed = original.clone();
    failed[outcome] = 1;
    refusal(
        QuickEvent::parse_binary_oracle(EventKind::End, &failed),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let mut invalid = original.clone();
    invalid[outcome] = 0;
    invalid[failure] = 1;
    invalid.splice(failure + 1..failure + 1, [0, 0]);
    refusal(
        QuickEvent::parse_binary_oracle(EventKind::End, &invalid),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let p = fixture("quick.preflight.bin");
    let start = offset("quick.preflight", "argv[2]");
    let mut nul = p.clone();
    nul[start + 8] = 0;
    refusal(
        QuickEvent::parse_binary_oracle(EventKind::Preflight, &nul),
        DocumentErrorCode::CrossLinkMismatch,
    );
    let mut empty = p.clone();
    let len = u64::from_le_bytes(empty[start..start + 8].try_into().unwrap()) as usize;
    empty[start..start + 8].copy_from_slice(&0u64.to_le_bytes());
    empty.drain(start + 8..start + 8 + len);
    QuickEvent::parse_binary_oracle(EventKind::Preflight, &empty).unwrap();
}
#[test]
fn d7_evidence_request_shared_p3_traversal_and_all_original_limits() {
    use crate::study_admission::{
        Consumption as RequestedConsumption, DiagonalRequest, PlanningBudget, PlanningUsage,
        ReceiptLimits,
    };
    use ckks_train_nn::dense::TransposeKeys;
    let bytes = fixture("quick.request.bin");
    let parsed = Auxiliary::parse_oracle(AuxKind::Request, &bytes).unwrap();
    let original = |limits: [usize; 2], work: [usize; 5]| DiagonalRequest {
        consumption: RequestedConsumption::Quick,
        physical_lanes: Some(2),
        transpose: TransposeKeys::Dedicated,
        eval_config: ckks_train_ckks::EvalConfig {
            hoist: false,
            bsgs: false,
            lazy_relin: false,
        },
        partitions: None,
        limits: ReceiptLimits {
            max_bytes: limits[0],
            max_records: limits[1],
        },
        work_budget: std::sync::Arc::new(PlanningBudget::new(PlanningUsage {
            geometries: work[0],
            initial_states: work[1],
            updates: work[2],
            handoffs: work[3],
            wrapper_calls: work[4],
        })),
    };
    check_request_retained(&parsed, &original([65536, 64], [1, 1, 4, 0, 3])).unwrap();
    for index in 0..7 {
        let mut limits = [65536, 64];
        let mut work = [1, 1, 4, 0, 3];
        if index < 2 {
            limits[index] += 1;
        } else {
            work[index - 2] += 1;
        }
        refusal(
            check_request_retained(&parsed, &original(limits, work)),
            DocumentErrorCode::CrossLinkMismatch,
        );
    }
    let transpose = offset("quick.request", "transpose");
    let mut reuse = bytes.clone();
    reuse[transpose] = 1;
    let mut values = 2u64.to_le_bytes().to_vec();
    values.extend_from_slice(&(-3i32).to_le_bytes());
    values.extend_from_slice(&7i32.to_le_bytes());
    reuse.splice(transpose + 1..transpose + 1, values);
    assert_eq!(
        Auxiliary::parse_oracle(AuxKind::Request, &reuse)
            .unwrap()
            .canonical()
            .unwrap(),
        reuse
    );
    let physical = offset("quick.request", "physical_lanes.present");
    let mut default = bytes.clone();
    default[physical] = 0;
    default.drain(physical + 1..physical + 9);
    assert_eq!(
        Auxiliary::parse_oracle(AuxKind::Request, &default)
            .unwrap()
            .canonical()
            .unwrap(),
        default
    );
}
#[test]
fn d7_evidence_production_mixed_nested_domains_tail_and_length_refusals() {
    use DocumentErrorCode::*;
    let bytes = fixture("quick.inputs.evidence.bin");
    let e = Evidence::parse_oracle(&bytes).unwrap();
    let mut common = e.common.clone();
    let policy = Mode::Production.policy(0);
    common.policy_revision = Id(&policy);
    let mut production = record_encoder("gate-inputs-evidence", Mode::Production).unwrap();
    common.write(&mut production).unwrap();
    e.branch.write(&mut production).unwrap();
    if let Tail::Inputs(t) = &e.tail {
        t.write(&mut production).unwrap();
    }
    // Production outer bytes, with a test identity nested inside: mixed mode rejects.
    refusal(Evidence::parse(&production.bytes), UnsupportedSchema);
    let mut true_marker = production.bytes.clone();
    let synthetic =
        8 + ENCODING_VERSION.len() + 8 + Mode::Production.domain("gate-inputs-evidence").len();
    true_marker[synthetic] = 1;
    refusal(Evidence::parse(&true_marker), UnsupportedSchema);
    let policy = Mode::Oracle.policy(1);
    common.policy_revision = Id(&policy);
    let mut wrong_tail = record_encoder("gate-implementation-evidence", Mode::Oracle).unwrap();
    common.write(&mut wrong_tail).unwrap();
    e.branch.write(&mut wrong_tail).unwrap();
    if let Tail::Inputs(t) = &e.tail {
        t.write(&mut wrong_tail).unwrap();
    }
    refusal(Evidence::parse_oracle(&wrong_tail.bytes), BadHex);
    let mut too_long = bytes.clone();
    let execution = offset("quick.inputs.evidence", "common.execution");
    too_long[execution..execution + 8].copy_from_slice(&u64::MAX.to_le_bytes());
    refusal(Evidence::parse_oracle(&too_long), TooLarge);
    refusal(check_file_size(A_MAX + 1, DocumentLayout::Quick), TooLarge);
    refusal(
        hex_len(2 * (QUICK_K_MAX + 1), QUICK_K_MAX, "quick.canonical_bytes"),
        TooLarge,
    );
    refusal(
        Evidence::parse_oracle(&fixture("negative/branch-tag.bin")),
        UnsupportedTag,
    );
}
#[test]
fn d7_evidence_shared_predecessors_and_clock_claims() {
    for hex in [
        target_literal("predecessor-null.canonical.hex"),
        target_literal("predecessor-no-start.canonical.hex"),
        target_literal("predecessor-with-start.canonical.hex"),
    ] {
        let bytes: Vec<_> = hex
            .trim()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let mut cursor = Cursor::new(&bytes);
        let predecessor = Option::<admission::Predecessor>::read(&mut cursor).unwrap();
        cursor.finish().unwrap();
        let mut output = Encoder::new(DocumentLayout::Start);
        predecessor.write(&mut output).unwrap();
        assert_eq!(output.bytes, bytes);
    }
    let clock = Clock {
        wall_source: WallSource::Unix,
        unix_ns: 4,
        monotonic_source: MonotonicSource::Instant,
        elapsed_since_operation_start_ns: 4,
    };
    clock_order(&clock, &clock).unwrap();
    let mut earlier = clock.clone();
    earlier.unix_ns = 3;
    refusal(
        clock_order(&clock, &earlier),
        DocumentErrorCode::TimingMismatch,
    );
    earlier.unix_ns = 4;
    earlier.elapsed_since_operation_start_ns = 3;
    refusal(
        clock_order(&clock, &earlier),
        DocumentErrorCode::TimingMismatch,
    );
}
