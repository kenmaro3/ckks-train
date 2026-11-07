//! FIX1 contract-derived counterexamples, no live or numerical constructors.
use super::observed::*;
use super::*;
fn old(name: &str) -> Vec<u8> {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("observed/literals.json")).unwrap();
    unhex(v[name].as_str().unwrap())
}
fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn cw03_preparation_child_selected_seed() {
    let mut accepted = Vec::new();
    for (op, id) in [(4, "W09"), (5, "W10")] {
        let b = old(id);
        let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
        w.bind_preparation_root(op, "iris__emlp1", &[0]).unwrap();
        if w.check_child(op, 2, "iris__emlp1", 1).is_ok() {
            accepted.push(op);
        }
    }
    let b = old("W08");
    let mut b = b[..b.len() - 16].to_vec();
    for n in [2u64, 0, 1] {
        b.extend_from_slice(&n.to_le_bytes());
    }
    let w = RequestedWork::decode(&b, Mode::Oracle).unwrap();
    w.bind_preparation_root(3, "iris__emlp1", &[0, 1]).unwrap();
    w.check_child(3, 2, "iris__emlp1", 1).unwrap();
    assert!(
        accepted.is_empty(),
        "seed0-only roots accepted selected1: {accepted:?}"
    );
}
#[test]
fn cw04_registered_full_checkpoint_prefix() {
    let b = old("0-1-result");
    let rec = ObservedRecord::decode(RecordKind::Result, &b, Mode::Oracle).unwrap();
    let RecordBody::Result(mut r) = rec.body else {
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
        lock: r.start.clone(),
        threshold_bits: 0,
        reached: true,
        first_checkpoint: Some(6),
        time_basis: Id("cumulative_update_ns"),
        time_ns: Some(10),
        censor_ns: 10,
    });
    r.validate(Mode::Oracle).unwrap();
    let mut accepted = Vec::new();
    for (updates, crossed) in [(1, vec![1]), (12, vec![2]), (6, vec![2]), (6, vec![])] {
        let mut wrong = r.clone();
        wrong.completed_updates = updates;
        wrong.processed_samples = updates * 20;
        wrong.checkpoints[0].updates = updates;
        wrong.checkpoints[0].samples = updates * 20;
        wrong.checkpoints[0].crossed = crossed;
        wrong.target.as_mut().unwrap().first_checkpoint = Some(updates);
        if wrong.validate(Mode::Oracle).is_ok() {
            accepted.push((updates, wrong.checkpoints[0].crossed.clone()));
        }
    }
    assert!(
        accepted.is_empty(),
        "invalid prefixes accepted: {accepted:?}"
    );
}
fn fix(name: &str) -> Vec<u8> {
    static VALUES: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
    let values = VALUES
        .get_or_init(|| serde_json::from_str(include_str!("observed/fix1_literals.json")).unwrap());
    let v = &values[name];
    if let Some(s) = v.as_str() {
        return unhex(s);
    }
    let base = v["base"].as_str().unwrap();
    let mut b = if let Some(name) = base.strip_prefix("old:") {
        old(name)
    } else {
        fix(base)
    };
    for p in v["patches"].as_array().unwrap().iter().rev() {
        let start = p[0].as_u64().unwrap() as usize;
        let remove = p[1].as_u64().unwrap() as usize;
        let replacement = unhex(p[2].as_str().unwrap());
        assert!(start + remove <= b.len());
        b.splice(start..start + remove, replacement);
        assert!(b.len() <= 524288);
    }
    b
}
fn with_old_bundle(check: impl FnOnce(&CurrentEvidenceBundle<'_, '_>)) {
    let b: [Vec<u8>; 6] = [
        "inputs",
        "implementation",
        "layout",
        "user",
        "resource",
        "claim",
    ]
    .map(|n| old(&format!("0-1-{n}")));
    with_bundle_bytes(&b, check);
}
fn with_full_bundle(mode: &str, check: impl FnOnce(&CurrentEvidenceBundle<'_, '_>)) {
    let b: [Vec<u8>; 6] = [
        "inputs",
        "implementation",
        "layout",
        "user",
        "resource",
        "claim",
    ]
    .map(|n| fix(&format!("full-{mode}-{n}")));
    with_bundle_bytes(&b, check);
}
fn with_bundle_bytes(b: &[Vec<u8>; 6], check: impl FnOnce(&CurrentEvidenceBundle<'_, '_>)) {
    let records: Vec<_> = b[..4]
        .iter()
        .map(|v| Evidence::parse_oracle(v).unwrap())
        .collect();
    let resource = OperationalResourceEvidence::decode(&b[4], Mode::Oracle).unwrap();
    let claim = ObservedRecord::decode(RecordKind::Claim, &b[5], Mode::Oracle).unwrap();
    let bundle = CurrentEvidenceBundle {
        non_resource: [&records[0], &records[1], &records[2], &records[3]],
        resource: &resource,
        claim: &claim,
    };
    bundle.check().unwrap();
    check(&bundle);
}
#[test]
fn cw02_quick_exact_preflight_bundle_binding() {
    with_old_bundle(|bundle| {
        let mut accepted = Vec::new();
        for variant in ["attempt", "approval", "receipt", "executable"] {
            let pre = fix(&format!("quick-{variant}-preflight-json"));
            let start = fix(&format!("quick-{variant}-start-json"));
            let end = fix(&format!("quick-{variant}-end-json"));
            if check_quick_v2_chain(&pre, &start, &end, bundle).is_ok() {
                accepted.push(variant);
            }
        }
        assert!(
            accepted.is_empty(),
            "consistently rehashed mismatches accepted: {accepted:?}"
        );
    });
}
#[test]
fn cw05_quick_start_before_result_clock() {
    with_old_bundle(|bundle| {
        let r = fix("quick-clock-result");
        let j = fix("quick-clock-result-json");
        let x = fix("quick-clock-result-export");
        let p = fix("quick-clock-preflight-json");
        let s = fix("quick-clock-start-json");
        let e = fix("quick-clock-end-json");
        let u = fix("quick-clock-supervisor-end");
        check_three_outputs(&r, &j, &x, Mode::Oracle).unwrap();
        check_quick_v2_chain(&p, &s, &e, bundle).unwrap();
        let result = ObservedRecord::decode(RecordKind::Result, &r, Mode::Oracle).unwrap();
        let RecordBody::Result(result) = result.body else {
            unreachable!()
        };
        result.check_bundle(bundle).unwrap();
        assert_eq!(result.ended.unix_ns, 0);
        assert!(
            QuickTerminal {
                result: &r,
                result_json: &j,
                result_export: &x,
                preflight: &p,
                start: &s,
                end: &e,
                supervisor_end: &u
            }
            .check(bundle)
            .is_err(),
            "result-ended0/0 precedes linked start104/4"
        );
    });
}
#[test]
fn cw01_frozen_full_generic_components() {
    for mode in ["i", "ni"] {
        let a = fix(&format!("full-{mode}-admission"));
        let s = fix(&format!("full-{mode}-start"));
        let a = admission::SyntheticAdmission::parse(&a).unwrap();
        let s = admission::SyntheticStart::parse(&s).unwrap();
        assert_eq!(
            a.0.canonical().unwrap(),
            fix(&format!("full-{mode}-admission-canonical"))
        );
        assert_eq!(
            s.0.canonical().unwrap(),
            fix(&format!("full-{mode}-start-canonical"))
        );
        admission::check_start_claims(&s.0, &a.0).unwrap();
        with_full_bundle(mode, |b| b.check().unwrap());
    }
}
fn full_terminal_members(prefix: &str) -> [Vec<u8>; 7] {
    [
        "admission",
        "start",
        "result",
        "result-json",
        "result-export",
        "end",
        "supervisor-end",
    ]
    .map(|name| fix(&format!("full-{prefix}-{name}")))
}
fn check_full_members(b: &[Vec<u8>; 7], bundle: &CurrentEvidenceBundle<'_, '_>) -> Result<()> {
    FullTerminal {
        admission: &b[0],
        start: &b[1],
        result: &b[2],
        result_json: &b[3],
        result_export: &b[4],
        end: &b[5],
        supervisor_end: &b[6],
    }
    .check(bundle)
}
#[test]
fn cw01_full_supplied_terminal_chain() {
    for mode in ["i", "ni"] {
        let b = full_terminal_members(mode);
        with_full_bundle(mode, |bundle| check_full_members(&b, bundle).unwrap());
    }
    with_full_bundle("i", |bundle| {
        for (variant, expected) in [
            ("tuple", "current.policy.tuple"),
            ("attempt", "full.bundle.branch"),
            ("admission-start", "start.admission.link"),
            ("target-lock", "result.bundle.target"),
            ("target-threshold", "result.bundle.target"),
            ("source", "result.bundle.provenance"),
            ("case", "result.bundle.provenance"),
            ("output", "file.retained_bytes"),
            ("evidence", "file.retained_bytes"),
            ("closure", "terminal.supervisor.closed"),
            ("clock", "quick.clock_order"),
            ("scope", "terminal.full.result"),
        ] {
            let b = full_terminal_members(variant);
            check_three_outputs(&b[2], &b[3], &b[4], Mode::Oracle).unwrap();
            let error = check_full_members(&b, bundle).expect_err(variant);
            assert_eq!(
                error.path(),
                expected,
                "wrong negative frontier for {variant}: {error}"
            );
        }
    });
    let b = full_terminal_members("i");
    with_old_bundle(|bundle| assert!(check_full_members(&b, bundle).is_err()));
}
