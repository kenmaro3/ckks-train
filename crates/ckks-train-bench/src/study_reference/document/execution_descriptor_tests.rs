//! Pure literal transport tests. No parameters, planners, crypto, native or file I/O.
use super::execution_descriptors::*;
use super::{Result, sha256_hex};
use crate::study_admission::{GeometryBody, MicroPart};
use ckks_train_ckks::EvalConfig;
use ckks_train_nn::{
    compact::CompactPlan,
    diagonal_plan::MaskRecipe,
    layout::{LayerShape, Layout},
    trainer_ni::{BiasSlot, CompactLayout, ContainerSpec},
};
use serde_json::Value;
const INPUTS: &str = include_str!("execution_descriptor_fixtures/inputs.json");
const NEGATIVES: &str = include_str!("execution_descriptor_fixtures/negative-vectors.json");
const MANIFEST: &str = include_str!("execution_descriptor_fixtures/manifest.json");
const LITERALS: [(&str, &str); 14] = [
    (
        "parameter_dense",
        include_str!("execution_descriptor_fixtures/parameter_dense.hex"),
    ),
    (
        "parameter_fixed",
        include_str!("execution_descriptor_fixtures/parameter_fixed.hex"),
    ),
    (
        "layout_i_single",
        include_str!("execution_descriptor_fixtures/layout_i_single.hex"),
    ),
    (
        "layout_ni_two",
        include_str!("execution_descriptor_fixtures/layout_ni_two.hex"),
    ),
    (
        "layout_i_three",
        include_str!("execution_descriptor_fixtures/layout_i_three.hex"),
    ),
    (
        "keys_i_dedicated",
        include_str!("execution_descriptor_fixtures/keys_i_dedicated.hex"),
    ),
    (
        "keys_i_reuse",
        include_str!("execution_descriptor_fixtures/keys_i_reuse.hex"),
    ),
    (
        "keys_ni_dedicated",
        include_str!("execution_descriptor_fixtures/keys_ni_dedicated.hex"),
    ),
    (
        "keys_ni_reuse",
        include_str!("execution_descriptor_fixtures/keys_ni_reuse.hex"),
    ),
    (
        "security_i_dense",
        include_str!("execution_descriptor_fixtures/security_i_dense.hex"),
    ),
    (
        "security_i_fixed",
        include_str!("execution_descriptor_fixtures/security_i_fixed.hex"),
    ),
    (
        "security_ni_absent",
        include_str!("execution_descriptor_fixtures/security_ni_absent.hex"),
    ),
    (
        "security_ni_fixed",
        include_str!("execution_descriptor_fixtures/security_ni_fixed.hex"),
    ),
    (
        "security_ni_dense",
        include_str!("execution_descriptor_fixtures/security_ni_dense.hex"),
    ),
];
fn bytes(s: &str) -> Vec<u8> {
    s.trim()
        .as_bytes()
        .chunks_exact(2)
        .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
        .collect()
}
fn literal(name: &str) -> Vec<u8> {
    bytes(LITERALS.iter().find(|(n, _)| *n == name).unwrap().1)
}
fn fixtures() -> Vec<Value> {
    serde_json::from_str::<Value>(INPUTS).unwrap()["vectors"]
        .as_array()
        .unwrap()
        .clone()
}
fn num(v: &Value) -> usize {
    usize::try_from(v.as_u64().unwrap()).unwrap()
}
fn flag(v: &Value) -> bool {
    v.as_u64().unwrap() == 1
}
fn nums(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap())
        .collect()
}
fn steps(v: &Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| i32::try_from(n.as_i64().unwrap()).unwrap())
        .collect()
}
fn opt_num(v: &Value) -> Option<usize> {
    (!v.is_null()).then(|| num(v))
}
fn hash(v: &Value) -> Option<&str> {
    v.as_str()
}
fn compact(v: &Value) -> CompactPlan {
    CompactPlan {
        d_out: num(&v["d_out"]),
        d_in: num(&v["d_in"]),
        batch: num(&v["batch"]),
        diags_per_ct: num(&v["diags_per_ct"]),
        lane_shift: num(&v["lane_shift"]),
    }
}
struct LiteralStage {
    dims: Vec<usize>,
    layout: Layout,
    shapes: Vec<LayerShape>,
    masks: Vec<MaskRecipe>,
    transpose: Vec<Vec<(usize, Vec<i32>)>>,
    rotations: Vec<i32>,
    compact: Option<CompactLayout>,
}
impl LiteralStage {
    fn new(v: &Value) -> Self {
        Self {
            dims: nums(&v["dims"]).into_iter().map(|n| n as usize).collect(),
            layout: Layout {
                slots: num(&v["slots"]),
                d: num(&v["d"]),
                batch: num(&v["batch"]),
            },
            shapes: v["shapes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| LayerShape {
                    d_out: num(&s[0]),
                    d_in: num(&s[1]),
                    n_out: num(&s[2]),
                    n_in: num(&s[3]),
                    l: num(&s[4]),
                    m: num(&s[5]),
                    baby: num(&s[6]),
                    giant: num(&s[7]),
                })
                .collect(),
            masks: v["masks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| MaskRecipe {
                    canonical_period: num(&m["canonical_period"]),
                    gradient_period: num(&m["gradient_period"]),
                    bias_period: num(&m["bias_period"]),
                    lanes: num(&m["lanes"]),
                    reduce_steps: steps(&m["reduce_steps"]),
                    replication_steps: steps(&m["replication_steps"]),
                    compact_first_period: flag(&m["compact_first_period"]),
                })
                .collect(),
            transpose: v["transpose"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array()
                        .unwrap()
                        .iter()
                        .map(|s| (num(&s["source"]), steps(&s["steps"])))
                        .collect()
                })
                .collect(),
            rotations: steps(&v["rotations"]),
            compact: (!v["compact"].is_null()).then(|| {
                let c = &v["compact"];
                CompactLayout {
                    plans: c["plans"].as_array().unwrap().iter().map(compact).collect(),
                    containers: c["containers"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| ContainerSpec {
                            layer: opt_num(&x["layer"]),
                            index: num(&x["index"]),
                            plan: compact(&x["plan"]),
                        })
                        .collect(),
                    biases: c["biases"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| BiasSlot {
                            container: num(&x["container"]),
                            position: num(&x["position"]),
                        })
                        .collect(),
                }
            }),
        }
    }
}
impl GeometryStages for Vec<LiteralStage> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn stage(&self, i: usize) -> StageFields<'_> {
        let s = &self[i];
        StageFields {
            dims: &s.dims,
            geometry: GeometryBody {
                layout: s.layout,
                shapes: &s.shapes,
                masks: &s.masks,
                transpose: &s.transpose,
                rotations: &s.rotations,
                compact: s.compact.as_ref(),
            },
        }
    }
}
fn encode_fixture(kind: &str, v: &Value) -> Result<Vec<u8>> {
    match kind {
        "parameters" => {
            let (q, p, qb, pb) = (
                nums(&v["q"]),
                nums(&v["p"]),
                nums(&v["q_bits"]),
                nums(&v["p_bits"]),
            );
            ParameterFields {
                q: &q,
                p: &p,
                q_bits: BitSizes::U64(&qb),
                p_bits: BitSizes::U64(&pb),
                log_n: v["log_n"].as_u64().unwrap(),
                scale_bits: v["scale_bits"].as_u64().unwrap(),
                dnum: v["dnum"].as_u64().unwrap(),
                main_weight: v["main_weight"].as_u64(),
            }
            .encode()
        }
        "layout" => {
            let stages = v["stages"]
                .as_array()
                .unwrap()
                .iter()
                .map(LiteralStage::new)
                .collect::<Vec<_>>();
            let parts = v["parts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| {
                    p.as_array()
                        .unwrap()
                        .iter()
                        .map(|p| MicroPart {
                            start: num(&p["start"]),
                            valid: num(&p["valid"]),
                        })
                        .collect()
                })
                .collect::<Vec<_>>();
            LayoutFields {
                parameter_identity: v["parameter_identity"].as_str().unwrap(),
                consumption: num(&v["consumption"]) as u8,
                mode: num(&v["mode"]) as u8,
                config: EvalConfig {
                    hoist: flag(&v["eval"]["hoist"]),
                    bsgs: flag(&v["eval"]["bsgs"]),
                    lazy_relin: flag(&v["eval"]["lazy_relin"]),
                },
                stages: &stages,
                parts: &parts,
            }
            .encode()
        }
        "keys" => {
            let pool = (!v["transpose"].is_null()).then(|| steps(&v["transpose"]));
            let (training, generated, execution) = (
                steps(&v["training_rotations"]),
                steps(&v["bootstrap_generated"]),
                steps(&v["bootstrap_execution"]),
            );
            KeyFields {
                parameter_identity: v["parameter_identity"].as_str().unwrap(),
                bootstrap_descriptor: hash(&v["bootstrap_descriptor"]),
                transpose: pool.as_deref(),
                training_level: v["training_level"].as_u64().unwrap(),
                training_rotations: &training,
                training_relin: flag(&v["training_relin"]),
                training_conjugation: flag(&v["training_conjugation"]),
                training_sse: flag(&v["training_sse"]),
                bootstrap_generated: &generated,
                bootstrap_execution: &execution,
                bootstrap_conjugation: flag(&v["bootstrap_conjugation"]),
                bootstrap_sse: flag(&v["bootstrap_sse"]),
                separate_bootstrap_bundle: flag(&v["separate_bootstrap_bundle"]),
                owner_policy: num(&v["owner_policy"]) as u8,
            }
            .encode()
        }
        "security" => {
            let requirements = nums(&v["bootstrap_unvalidated"])
                .into_iter()
                .map(|n| n as u8)
                .collect::<Vec<_>>();
            SecurityFields {
                registered_policy: num(&v["registered_policy"]) as u8,
                parameter_identity: v["parameter_identity"].as_str().unwrap(),
                bootstrap_descriptor: hash(&v["bootstrap_descriptor"]),
                main: v["main"].as_u64(),
                ephemeral: (!v["ephemeral"].is_null()).then(|| v["ephemeral"]["weight"].as_u64()),
                assurance: num(&v["assurance"]) as u8,
                bootstrap_scope: v["bootstrap_scope"].as_u64().map(|n| n as u8),
                bootstrap_unvalidated: Requirements::Tags(&requirements),
            }
            .encode()
        }
        _ => panic!("unknown fixture kind"),
    }
}
fn kind(name: &str) -> Kind {
    if name.starts_with("parameter") {
        Kind::Parameters
    } else if name.starts_with("layout") {
        Kind::Layout
    } else if name.starts_with("keys") {
        Kind::Keys
    } else {
        Kind::Security
    }
}
fn positives(k: &str) {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    for f in fixtures().iter().filter(|f| f["kind"] == k) {
        let name = f["name"].as_str().unwrap();
        let expected = literal(name);
        let actual = encode_fixture(k, &f["input"]).unwrap();
        assert_eq!(actual, expected, "{name}");
        let view = WireDescriptor::parse(kind(name), &expected).unwrap();
        assert_eq!(view.canonical_bytes(), actual);
        assert!(view.matches_bytes(&actual).is_ok());
        // The digest is an additional frozen check, never the sole byte oracle.
        let rows = manifest["vectors"].as_array().unwrap();
        let row = rows.iter().find(|r| r["name"] == name).unwrap();
        assert_eq!(sha256_hex(&actual), row["sha256"].as_str().unwrap());
    }
}
#[test]
fn execution_parameter_literal_bytes() {
    positives("parameters");
}
#[test]
fn execution_layout_literal_bytes() {
    positives("layout");
}
#[test]
fn execution_key_literal_bytes() {
    positives("keys");
}
#[test]
fn execution_security_literal_bytes() {
    positives("security");
}
#[test]
fn execution_complete_negative_images_distinguish_wire_and_binding() {
    const GRAMMAR_REFUSALS: [&str; 30] = [
        "parameter_dense-wrong-domain",
        "layout_i_single-wrong-domain",
        "keys_i_dedicated-wrong-domain",
        "security_i_dense-wrong-domain",
        "parameter-count-u64-overflow",
        "parameter-count-exceeds-A_MAX",
        "parameter-wrong-text-length",
        "parameter-option-tag2",
        "layout-mode-tag2",
        "layout-consumption-tag2",
        "layout-bool2",
        "layout-count-zero",
        "layout-count-four",
        "layout-compact-option2",
        "layout-container-layer-option2",
        "keys-bootstrap-option2",
        "keys-transpose-tag2",
        "keys-owner-tag1",
        "keys-uppercase-hash",
        "security-policy-tag2",
        "security-main-secret-tag2",
        "security-ephemeral-option2",
        "security-ephemeral-secret-tag2",
        "security-assurance1",
        "security-scope-option2",
        "security-scope-value1",
        "parameter_dense-trailing-byte",
        "layout_i_single-trailing-byte",
        "keys_i_dedicated-trailing-byte",
        "security_i_dense-trailing-byte",
    ];
    const PARAMETER_REFUSALS: [&str; 9] = [
        "parameter-empty-chain",
        "parameter-bit-vector-count",
        "parameter-dnum-zero",
        "parameter-dnum-not-divisor",
        "parameter-logN-overflow",
        "parameter-zero-scale",
        "parameter-main-weight-zero",
        "parameter-main-weight-overN",
        "parameter-swapped-bit-vectors",
    ];
    const SEMANTIC_MISMATCHES: [&str; 17] = [
        "layout-valid-zero",
        "layout-valid-over-lanes",
        "layout-start-discontinuous",
        "layout-ordered-chain-swap",
        "keys-zero-rotation",
        "keys-duplicate-rotation",
        "keys-superset-not-equal",
        "keys-generated-executed-swap",
        "keys-bootstrap-absent-with-bundle",
        "keys-reuse-pool-reorder",
        "security-weight-zero",
        "security-weight-overN",
        "security-weight-change",
        "security-limitations-order",
        "security-limitations-count",
        "security-NI-no-bootstrap",
        "security-absent-not-dense",
    ];
    let values: Value = serde_json::from_str(NEGATIVES).unwrap();
    let mut seen = [false; 56];
    let mut counts = [0; 3];
    for v in values["vectors"].as_array().unwrap() {
        let name = v["name"].as_str().unwrap();
        let b = bytes(v["complete_hex"].as_str().unwrap());
        let base = v["base"].as_str().unwrap();
        let parsed = WireDescriptor::parse(kind(base), &b);
        let (class, index) = if let Some(i) = GRAMMAR_REFUSALS.iter().position(|n| *n == name) {
            assert!(
                v["expected"]
                    .as_str()
                    .unwrap()
                    .starts_with("Reject malformed"),
                "{name}"
            );
            assert!(parsed.is_err(), "grammar refusal must not parse: {name}");
            (0, i)
        } else if let Some(i) = PARAMETER_REFUSALS.iter().position(|n| *n == name) {
            assert!(
                v["expected"].as_str().unwrap().starts_with("Wire may"),
                "{name}"
            );
            assert!(kind(base) == Kind::Parameters, "{name}");
            assert!(
                parsed.is_err(),
                "constructor-invalid parameter must not parse: {name}"
            );
            (1, GRAMMAR_REFUSALS.len() + i)
        } else if let Some(i) = SEMANTIC_MISMATCHES.iter().position(|n| *n == name) {
            assert!(
                v["expected"].as_str().unwrap().starts_with("Wire may"),
                "{name}"
            );
            let wire = parsed.unwrap_or_else(|e| panic!("semantic image must parse: {name}: {e}"));
            // The unchanged expectation is synthetic bytes, never checked preparation facts.
            assert!(wire.matches_bytes(&literal(base)).is_err(), "{name}");
            match name {
                "security-weight-zero" | "security-weight-overN" => {
                    assert!(wire.validate_security_declarations(16).is_err(), "{name}");
                }
                "security-weight-change" => {
                    assert!(wire.validate_security_declarations(16).is_ok(), "{name}");
                }
                _ => {}
            }
            (2, GRAMMAR_REFUSALS.len() + PARAMETER_REFUSALS.len() + i)
        } else {
            panic!("unclassified frozen negative image: {name}");
        };
        assert!(!seen[index], "duplicate frozen negative image: {name}");
        seen[index] = true;
        counts[class] += 1;
    }
    assert!(
        seen.into_iter().all(|present| present),
        "missing frozen negative image"
    );
    assert_eq!(counts, [30, 9, 17]);
}

#[test]
fn execution_every_proper_prefix_rejected() {
    for (name, hex) in LITERALS {
        let b = bytes(hex);
        for n in 0..b.len() {
            assert!(
                WireDescriptor::parse(kind(name), &b[..n]).is_err(),
                "{name}:{n}"
            );
        }
    }
}
#[test]
fn execution_checked_size_and_arithmetic_guards() {
    for n in [2_097_151, 2_097_152] {
        assert!(checked_size(n).is_ok());
    }
    for n in [2_097_153, u64::MAX] {
        assert!(checked_size(n).is_err());
    }
    assert!(checked_add(u64::MAX, 1).is_err());
    assert!(checked_mul(u64::MAX, 8).is_err());
    assert!(checked_usize(4_294_967_296, 32).is_err());
    assert!(checked_usize(4_294_967_295, 32).is_ok());
}
#[test]
fn execution_synthetic_wire_has_no_registered_policy_authority() {
    for name in [
        "security_i_fixed",
        "security_ni_absent",
        "security_ni_fixed",
        "security_ni_dense",
    ] {
        let b = literal(name);
        let wire = WireDescriptor::parse(Kind::Security, &b).unwrap();
        assert!(wire.validate_security_declarations(16).is_ok());
        assert!(
            wire.validate_registered_security_declarations(1 << 16)
                .is_err(),
            "{name}"
        );
    }
}
#[test]
fn execution_parameter_constructor_invariants_without_construction() {
    let f = fixtures()
        .into_iter()
        .find(|f| f["name"] == "parameter_dense")
        .unwrap();
    for (field, value) in [
        ("log_n", 0),
        ("log_n", 18),
        ("dnum", 0),
        ("dnum", 3),
        ("scale_bits", 0),
        ("main_weight", 0),
        ("main_weight", 17),
    ] {
        let mut input = f["input"].clone();
        input[field] = Value::from(value);
        assert!(
            encode_fixture("parameters", &input).is_err(),
            "{field}:{value}"
        );
    }
    for bits in [0, 1, 61] {
        let mut input = f["input"].clone();
        input["q_bits"][0] = Value::from(bits);
        assert!(encode_fixture("parameters", &input).is_err());
    }
    for exponent in [0, 61] {
        let mut input = f["input"].clone();
        input["scale_bits"] = Value::from((2f64.powi(exponent)).to_bits());
        assert!(encode_fixture("parameters", &input).is_err());
    }
    let mut input = f["input"].clone();
    input["scale_bits"] = Value::from((3f64).to_bits());
    assert!(encode_fixture("parameters", &input).is_err());
}

// Catches any changed P3 geometry primitive/order, without a geometry planner.
#[test]
fn execution_shared_p3_geometry_matches_frozen_layout_slices() {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    for f in fixtures().iter().filter(|f| f["kind"] == "layout") {
        let name = f["name"].as_str().unwrap();
        let b = literal(name);
        let rows = manifest["vectors"].as_array().unwrap();
        let offsets = &rows.iter().find(|r| r["name"] == name).unwrap()["offsets"];
        let stages = f["input"]["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(LiteralStage::new)
            .collect::<Vec<_>>();
        for i in 0..stages.len() {
            let start = num(&offsets[format!("stages.{i}.slots")]);
            let end = if i + 1 < stages.len() {
                num(&offsets[format!("stages.{}.dims.count", i + 1)])
            } else {
                num(&offsets["parts.count"])
            };
            let bytes =
                crate::study_admission::test_geometry_transcript(stages.stage(i).geometry).unwrap();
            assert_eq!(bytes, &b[start..end], "{name}:{i}");
        }
    }
}
