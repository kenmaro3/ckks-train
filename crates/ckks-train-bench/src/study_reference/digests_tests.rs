use super::*;
use crate::study_profile::{Checkpoint, PlannedUpdate};
fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("session_goldens/fixture_inputs.json")).unwrap()
}
fn hex(bytes: &[u8]) -> String {
    {
        use std::fmt::Write;
        bytes.iter().fold(String::new(), |mut out, x| {
            write!(&mut out, "{x:02x}").unwrap();
            out
        })
    }
}
#[test]
fn independent_prefix_zero_one_two_penultimate_and_full() {
    let v = fixture();
    let p = &v["prefix"];
    let plan = BatchPlan {
        updates: p["valid_samples"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, n)| PlannedUpdate {
                index,
                processed_start: 0,
                valid_samples: n.as_u64().unwrap() as usize,
                source_schedule_index: None,
                source_coefficient: 0.,
                mean_gradient_rate: 0.,
                delta_multiplier: 0.,
            })
            .collect(),
        checkpoints: p["checkpoint_completed_updates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| Checkpoint {
                completed_updates: n.as_u64().unwrap() as usize,
                processed_samples: 0,
                crossed_logical_boundaries: vec![],
                exact_logical_boundary: false,
                interval_updates: 0,
                interval_samples: 0,
            })
            .collect(),
        processed_samples: 0,
        dropped_samples: 0,
    };
    let hash = p["trusted_full_plan_sha256"].as_str().unwrap();
    for (n, expected) in [
        (0, include_str!("session_goldens/prefix_n0.hex")),
        (1, include_str!("session_goldens/prefix_n1.hex")),
        (2, include_str!("session_goldens/prefix_n2.hex")),
        (4, include_str!("session_goldens/prefix_n4.hex")),
    ] {
        let bytes = prefix_bytes(&plan, hash, n).unwrap();
        assert_eq!(hex(&bytes), expected.split_whitespace().collect::<String>());
        assert_eq!(consumed(&plan, hash, n).unwrap(), sha256_hex(&bytes));
    }
    assert_eq!(consumed(&plan, hash, 5).unwrap(), hash);
    assert!(consumed(&plan, hash, 6).is_err());
}
fn check_state<T: Scalar + std::str::FromStr>(expected: &str, positive_zero: bool)
where
    T::Err: std::fmt::Debug,
{
    let mut v = fixture();
    if positive_zero {
        v["state"]["layers"][0]["tensors"][0]["values"][0] = serde_json::json!("0.0");
    }
    let p = &v["state"];
    let arrays: Vec<Vec<ndarray::Array2<T>>> = p["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| {
            l["tensors"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    ndarray::Array2::from_shape_vec(
                        (
                            t["rows"].as_u64().unwrap() as usize,
                            t["cols"].as_u64().unwrap() as usize,
                        ),
                        t["values"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|x| {
                                let s = x.as_str().unwrap();
                                s.parse().unwrap()
                            })
                            .collect(),
                    )
                    .unwrap()
                })
                .collect()
        })
        .collect();
    let layers: Vec<_> = arrays
        .iter()
        .enumerate()
        .map(|(i, a)| LayerView {
            name: p["layers"][i]["name"].as_str().unwrap(),
            tensors: [a[0].view(), a[1].view(), a[2].view(), a[3].view()],
        })
        .collect();
    assert_eq!(
        hex(&state_bytes(
            p["materialized_input_sha256"].as_str().unwrap(),
            p["selected_receipt_sha256"].as_str().unwrap(),
            p["completed_updates"].as_u64().unwrap() as usize,
            p["processed_samples"].as_u64().unwrap() as usize,
            &layers
        )
        .unwrap()),
        expected.split_whitespace().collect::<String>()
    );
}
#[test]
fn independent_f32_f64_all_four_roles_signed_zero() {
    check_state::<f32>(include_str!("session_goldens/state_f32.hex"), false);
    check_state::<f64>(include_str!("session_goldens/state_f64.hex"), false);
    check_state::<f32>(
        include_str!("session_goldens/state_f32_positive_zero.hex"),
        true,
    );
    check_state::<f64>(
        include_str!("session_goldens/state_f64_positive_zero.hex"),
        true,
    );
}
#[test]
fn tensor_encoding_checks_shapes_nonfinite_and_logical_strides() {
    use ndarray::array;
    let canonical = array![[1.0f32, 2.0], [3.0, 4.0]];
    let transposed_storage = array![[1.0f32, 3.0], [2.0, 4.0]];
    let bias = array![[0.0f32, -0.0]];
    let zero = ndarray::Array2::<f32>::zeros((2, 2));
    let a = LayerView {
        name: "head",
        tensors: [canonical.view(), bias.view(), zero.view(), bias.view()],
    };
    let b = LayerView {
        name: "head",
        tensors: [
            transposed_storage.t(),
            bias.view(),
            zero.view(),
            bias.view(),
        ],
    };
    assert_eq!(
        state_bytes(&"1".repeat(64), &"2".repeat(64), 0, 0, &[a]).unwrap(),
        state_bytes(&"1".repeat(64), &"2".repeat(64), 0, 0, &[b]).unwrap()
    );
    for role in 0..4 {
        let mut tensors = [canonical.clone(), bias.clone(), zero.clone(), bias.clone()];
        tensors[role][[0, 0]] = f32::NAN;
        let v = LayerView {
            name: "head",
            tensors: [
                tensors[0].view(),
                tensors[1].view(),
                tensors[2].view(),
                tensors[3].view(),
            ],
        };
        assert!(state_bytes(&"1".repeat(64), &"2".repeat(64), 0, 0, &[v]).is_err());
    }
    let bad = LayerView {
        name: "head",
        tensors: [canonical.view(), canonical.view(), zero.view(), bias.view()],
    };
    assert!(state_bytes(&"1".repeat(64), &"2".repeat(64), 0, 0, &[bad]).is_err());
}
