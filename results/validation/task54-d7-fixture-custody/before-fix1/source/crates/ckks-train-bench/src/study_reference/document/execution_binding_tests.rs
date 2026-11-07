//! Independent frozen outer fixtures; fictional claims never supply actual owners.
use super::execution_binding::ExecutionFields;
use ckks_train_ckks::EvalConfig;
use serde_json::Value;
fn fixture(name: &str) -> String {
    super::fixture_literals::target(name).to_owned()
}
fn literal(s: &str) -> Vec<u8> {
    s.trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn d7_outer_six_frozen_independent_vectors() {
    for branch in 0..6 {
        let v: Value = serde_json::from_str(&fixture(&format!("branch-{branch}.json"))).unwrap();
        let e = &v["execution"];
        let s = |k: &str| e[k].as_str().unwrap();
        let f = ExecutionFields {
            implementation: match s("implementation") {
                "main_diagonal" => 0,
                "matched_row2020" => 1,
                "matched_diagonal2020" => 2,
                "packed_c" => 3,
                "external_b" => 4,
                _ => panic!(),
            },
            revision: s("implementation_revision"),
            manifest: s("implementation_manifest_sha256"),
            parameter: s("parameter_identity_sha256"),
            bootstrap: e["bootstrap_descriptor_sha256"].as_str(),
            layout: s("layout_plan_sha256"),
            trainer: s("trainer_plan_sha256"),
            eval: EvalConfig {
                hoist: e["eval_config"]["hoist"].as_bool().unwrap(),
                bsgs: e["eval_config"]["bsgs"].as_bool().unwrap(),
                lazy_relin: e["eval_config"]["lazy_relin"].as_bool().unwrap(),
            },
            engine: match s("bootstrap_engine") {
                "none" => 0,
                "host" => 1,
                "native_cuda" => 2,
                "upstream" => 3,
                _ => panic!(),
            },
            keys: s("key_policy_sha256"),
            security: s("security_policy_sha256"),
            physical: s("physical_binding_sha256"),
            runtime: s("runtime_policy_sha256"),
        };
        let (n, body) = f.traverse(true).unwrap();
        assert_eq!(n, f.traverse(false).unwrap().0);
        assert_eq!(
            body,
            literal(&fixture(&format!("execution-{branch}.canonical.hex")))
        );
        let binding = f.binding_bytes().unwrap();
        assert_eq!(
            crate::data::sha256_hex(&binding),
            v["execution_binding_sha256"].as_str().unwrap()
        );
        if branch == 0 {
            assert_eq!(binding, literal(&fixture("case-binding.canonical.hex")));
        }
    }
}

#[test]
fn d7_outer_payload_checked_live_intervals() {
    use super::execution_binding::binding_payload;
    assert_eq!(binding_payload(209, 680, 101, 1000).unwrap(), (1017, 2017));
    assert_eq!(binding_payload(209, 680, 101, 0).unwrap(), (1017, 1145));
    assert!(binding_payload(usize::MAX, 680, 101, 1000).is_err());
    assert!(binding_payload(209, 680, usize::MAX, 1000).is_err());
    assert!(binding_payload(209, 680, 101, usize::MAX).is_err());
}

#[test]
fn d7_outer_transport_bounds_refuse_without_route_authority() {
    const H: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let fields = || ExecutionFields {
        implementation: 0,
        revision: "fixture",
        manifest: H,
        parameter: H,
        bootstrap: None,
        layout: H,
        trainer: H,
        eval: EvalConfig::OPTIMIZED,
        engine: 0,
        keys: H,
        security: H,
        physical: H,
        runtime: H,
    };
    let mut f = fields();
    f.implementation = 5;
    assert!(f.binding_bytes().is_err());
    let mut f = fields();
    f.engine = 4;
    assert!(f.binding_bytes().is_err());
    let revision = "x".repeat(129);
    let mut f = fields();
    f.revision = &revision;
    assert!(f.binding_bytes().is_err());
    for bad in [
        "a",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    ] {
        let mut f = fields();
        f.parameter = bad;
        assert!(f.binding_bytes().is_err());
        let mut f = fields();
        f.bootstrap = Some(bad);
        assert!(f.binding_bytes().is_err());
    }
}
