use super::*;
#[path = "document_goldens.rs"]
mod golden;
fn unhex(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn independent_python_exact_canonical_bytes_cover_all_domains_and_timing_options() {
    for (s, h, d) in [
        (
            golden::REFERENCE,
            golden::REFERENCE_HEX,
            golden::REFERENCE_SHA,
        ),
        (
            golden::CAPTURED_REFERENCE,
            golden::CAPTURED_REFERENCE_HEX,
            golden::CAPTURED_REFERENCE_SHA,
        ),
    ] {
        let x = parse_reference_document(s.as_bytes()).unwrap();
        assert_eq!(
            reference_encoding(&x.common, &x.body).unwrap().bytes,
            unhex(h)
        );
        assert_eq!(x.payload_sha256(), d);
    }
    for (s, h, d) in [
        (golden::QUICK, golden::QUICK_HEX, golden::QUICK_SHA),
        (
            golden::QUICK_PRIOR,
            golden::QUICK_PRIOR_HEX,
            golden::QUICK_PRIOR_SHA,
        ),
        (
            golden::QUICK_F64,
            golden::QUICK_F64_HEX,
            golden::QUICK_F64_SHA,
        ),
        (
            golden::DIAGNOSTIC_FULL,
            golden::DIAGNOSTIC_FULL_HEX,
            golden::DIAGNOSTIC_FULL_SHA,
        ),
    ] {
        let x = parse_diagnostic_document(s.as_bytes()).unwrap();
        assert_eq!(
            diagnostic_encoding(&x.common, &x.body).unwrap().bytes,
            unhex(h)
        );
        assert_eq!(x.payload_sha256(), d);
    }
    for (s, h, d) in [
        (
            golden::FAILURE_NULL,
            golden::FAILURE_NULL_HEX,
            golden::FAILURE_NULL_SHA,
        ),
        (
            golden::FAILURE_PRESENT,
            golden::FAILURE_PRESENT_HEX,
            golden::FAILURE_PRESENT_SHA,
        ),
        (
            golden::FAILURE_ENDPOINT_NULL,
            golden::FAILURE_ENDPOINT_NULL_HEX,
            golden::FAILURE_ENDPOINT_NULL_SHA,
        ),
        (
            golden::FAILURE_ENDPOINT_PRESENT,
            golden::FAILURE_ENDPOINT_PRESENT_HEX,
            golden::FAILURE_ENDPOINT_PRESENT_SHA,
        ),
    ] {
        let x = parse_failure_document(s.as_bytes()).unwrap();
        assert_eq!(
            failure_encoding(&x.common, &x.body).unwrap().bytes,
            unhex(h)
        );
        assert_eq!(x.payload_sha256(), d);
    }
}
#[test]
fn bounded_counters_reject_overflow_without_large_buffers() {
    assert_eq!(
        bounded_add(0, ENVELOPE_MAX, ENVELOPE_MAX, "x").unwrap(),
        ENVELOPE_MAX
    );
    assert!(bounded_add(1, ENVELOPE_MAX, ENVELOPE_MAX, "x").is_err());
    assert!(bounded_add(usize::MAX, 1, usize::MAX, "x").is_err());
    assert_eq!(hex_len(C_MAX * 2, C_MAX, "x").unwrap(), C_MAX);
    assert!(hex_len(C_MAX * 2 + 2, C_MAX, "x").is_err());
    assert_eq!(hex_len(R_MAX * 2, R_MAX, "x").unwrap(), R_MAX);
    assert!(hex_len(R_MAX * 2 + 2, R_MAX, "x").is_err());
    assert!(hex_len(usize::MAX, C_MAX, "x").is_err());
    let mut e = Encoder {
        bytes: Vec::new(),
        outer: A_MAX,
    };
    assert!(e.byte(0).is_err());
    assert!(e.bytes.is_empty());
}
#[test]
fn reporting_binary64_literal_ratios_preserve_rounding_and_positive_zero() {
    // Python struct.pack literals, independent of the document encoder.
    let cases = [
        (1, 49, "0.02040816326530612", "3f94e5e0a72f0539"),
        (1, 10, "0.1", "3fb999999999999a"),
        (1, 7, "0.14285714285714285", "3fc2492492492492"),
        (
            9007199254740991u64,
            9007199254740992u64,
            "0.9999999999999999",
            "3fefffffffffffff",
        ),
        (0, 9007199254740992u64, "0", "0000000000000000"),
        (
            9007199254740992u64,
            9007199254740992u64,
            "1",
            "3ff0000000000000",
        ),
    ];
    let mut distinguishes_legacy = false;
    for (correct, count, value, bits) in cases {
        let s = format!(
            r#"{{"correct":{correct},"test_count":{count},"value":{value},"bits":"{bits}"}}"#
        );
        let raw = serde_json::from_str::<&RawValue>(&s).unwrap();
        let a = ClaimedAccuracy::decode(raw, "accuracy").unwrap();
        assert_eq!(a.bits, u64::from_str_radix(bits, 16).unwrap());
        let legacy: f64 = serde_json::from_str(value).unwrap();
        distinguishes_legacy |= legacy.to_bits() != a.bits;
    }
    assert!(
        distinguishes_legacy,
        "must exercise a document-local rounding boundary distinct from unchanged legacy serde conversion"
    );
    let s = r#"{"correct":0,"test_count":1,"value":-0.0,"bits":"8000000000000000"}"#;
    assert_eq!(
        ClaimedAccuracy::decode(serde_json::from_str(s).unwrap(), "accuracy")
            .unwrap_err()
            .code(),
        DocumentErrorCode::ObservationMismatch
    );
}
#[test]
fn all_21_profiles_all_ten_seeds_match_independent_size_table_and_metadata_inventory() {
    use crate::study_profile::{Initialization, OrderPolicy, StudyRegistry};
    let registry =
        StudyRegistry::load(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .unwrap();
    // Literal R/U/E/P/C/R/J table from the ratified independently reviewed source inventory.
    let table = [
        (
            "iris__mlp2020",
            48000,
            2400,
            400,
            83,
            1128176,
            1689180,
            7731864,
        ),
        (
            "iris__emlp1",
            48000,
            2400,
            400,
            292,
            1128176,
            1691688,
            7736880,
        ),
        (
            "iris__emlp2",
            48000,
            2400,
            400,
            2792,
            1128176,
            1721688,
            7796880,
        ),
        (
            "iris__emlp3",
            48000,
            2400,
            400,
            11884,
            1128176,
            1830792,
            8015088,
        ),
        (
            "breast_cancer__mlp2020",
            44800,
            2800,
            100,
            332,
            1123476,
            1688268,
            7720640,
        ),
        (
            "breast_cancer__emlp1",
            44800,
            2800,
            100,
            1058,
            1123476,
            1696980,
            7738064,
        ),
        (
            "breast_cancer__emlp2",
            44800,
            2800,
            100,
            4260,
            1123476,
            1735404,
            7814912,
        ),
        (
            "breast_cancer__emlp3",
            44800,
            2800,
            100,
            14758,
            1123476,
            1861380,
            8066864,
        ),
        (
            "mnist_released__mlp2020",
            48000,
            1000,
            10,
            2080,
            1082066,
            1563114,
            7387512,
        ),
        (
            "mnist_released__emlp1",
            48000,
            1000,
            10,
            6832,
            1082066,
            1620138,
            7501560,
        ),
        (
            "mnist_released__emlp2",
            48000,
            1000,
            10,
            16256,
            1082066,
            1733226,
            7727736,
        ),
        (
            "mnist_released__emlp3",
            48000,
            1000,
            10,
            39184,
            1082066,
            2008362,
            8278008,
        ),
        (
            "fashion_mnist__mlp2020",
            3000000,
            62500,
            50,
            2080,
            2613526,
            31138894,
            69601992,
        ),
        (
            "fashion_mnist__emlp1",
            3000000,
            62500,
            50,
            6832,
            2613526,
            31195918,
            69716040,
        ),
        (
            "fashion_mnist__emlp2",
            3000000,
            62500,
            50,
            16256,
            2613526,
            31309006,
            69942216,
        ),
        (
            "fashion_mnist__emlp3",
            3000000,
            62500,
            50,
            39184,
            2613526,
            31584142,
            70492488,
        ),
        (
            "mnist_released__mlp2020__b64",
            48000,
            750,
            10,
            2080,
            1073816,
            1536864,
            7318512,
        ),
        (
            "mnist_released__emlp1__b64",
            48000,
            750,
            10,
            6832,
            1073816,
            1593888,
            7432560,
        ),
        (
            "mnist_released__emlp2__b64",
            48000,
            750,
            10,
            16256,
            1073816,
            1706976,
            7658736,
        ),
        (
            "mnist_released__emlp3__b64",
            48000,
            750,
            10,
            39184,
            1073816,
            1982112,
            8209008,
        ),
        (
            "mnist_released__emlp1__b128_candidate",
            48000,
            375,
            10,
            6832,
            1061441,
            1554513,
            7329060,
        ),
    ];
    assert_eq!(registry.profile_ids().len(), table.len());
    let mut max_string = 0;
    let mut max_source = 0;
    let mut missing = 0;
    for (id, rows, updates, epochs, params, contract, receipt, json) in table {
        for seed in 0..10 {
            let p = registry.resolve(id, seed).unwrap();
            let m = p.metadata();
            let d = p.data_request();
            let pop = d.train_samples.unwrap();
            let flat = matches!(m.order, OrderPolicy::Captured);
            let kept = if flat || !m.drop_last {
                pop
            } else {
                pop / m.batch * m.batch
            };
            let r = kept * m.logical_epochs;
            let u = if flat {
                r.div_ceil(m.batch)
            } else {
                kept.div_ceil(m.batch) * m.logical_epochs
            };
            let e = m.logical_epochs;
            let parameters: usize = m.layers.iter().map(|l| l.input * l.output + l.output).sum();
            assert_eq!(
                (r, u, e, parameters),
                (rows, updates, epochs, params),
                "{id}/{seed}"
            );
            let schedule = usize::from(m.schedule.is_some());
            let c = (25 + 8 * schedule) * u + 49 * e + 1048576;
            let rr = (97 + 8 * schedule) * u + 8 * r + 57 * e + 8 + 12 * parameters + 1048576;
            let j = 2 * (c + rr) + 2097152;
            assert_eq!((c, rr, j), (contract, receipt, json), "{id}/{seed}");
            assert!(c <= C_MAX && rr <= R_MAX && j <= J_MAX);
            if m.dataset == "breast_cancer" {
                assert_eq!((pop - kept) * e, 700);
            }
            if m.batch == 128 {
                let ends: Vec<usize> = (1..=e)
                    .map(|epoch| (epoch * pop).div_ceil(m.batch))
                    .collect();
                let diffs: Vec<usize> = ends
                    .iter()
                    .scan(0, |prev, &end| {
                        let n = end - *prev;
                        *prev = end;
                        Some(n)
                    })
                    .collect();
                assert_eq!(diffs, vec![38, 37, 38, 37, 38, 37, 38, 37, 38, 37]);
                assert_eq!(
                    diffs.iter().map(|n| n * 128).take(2).collect::<Vec<_>>(),
                    vec![4864, 4736]
                );
            }
            let shared = p.shared_selector().sources();
            let selected = p.selected_sources();
            max_source = max_source.max(shared.len() + selected.len());
            assert!(
                shared.len() <= 64
                    && selected.len() <= 7
                    && m.identity.dataset_files.len() <= 4
                    && m.layers.len() <= 6
            );
            let mut strings = vec![
                m.id.as_str(),
                m.dataset.as_str(),
                m.rate_description.as_str(),
                m.identity.preprocessing.as_str(),
                m.identity.split_policy.as_str(),
            ];
            for l in &m.layers {
                strings.push(&l.name);
            }
            for s in shared.iter().chain(selected) {
                strings.push(s.path());
                strings.push(s.sha256());
            }
            if let Some(c) = p.shared_selector().captured() {
                strings.extend([
                    c.family(),
                    c.model(),
                    c.order_template(),
                    c.schedule_template(),
                ]);
                if let Some(s) = c.initialization_template() {
                    strings.push(s);
                }
            }
            if let Some(s) = &m.schedule {
                strings.push(&s.json_pointer);
            }
            let sum: usize = strings.iter().map(|s| 8 + s.len()).sum();
            assert!(sum + 4096 < 1048576);
            for s in strings {
                max_string = max_string.max(s.len());
                assert!(s.len() <= 4096);
            }
            if matches!(m.initialization, Initialization::MissingCaptured) {
                missing += 1;
                assert!(seed >= 3);
            }
        }
    }
    assert!(missing > 0);
    assert!(max_source <= 64);
    assert!(max_string <= 493);
    eprintln!(
        "metadata-only inventory: profiles=21 seed_rows=210 max_combined_sources={max_source} max_utf8_string_bytes={max_string} missing_captured_seed_rows={missing}"
    );
}
#[test]
fn actual_iris_seed_zero_and_seven_all_graphs_roundtrip_only_input_identities() {
    use crate::{
        study_inputs::materialize, study_profile::StudyRegistry, study_reference::derive_identities,
    };
    let registry =
        StudyRegistry::load(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .unwrap();
    fn hex(b: &[u8]) -> String {
        use std::fmt::Write;
        let mut output = String::with_capacity(b.len() * 2);
        for byte in b {
            write!(output, "{byte:02x}").unwrap();
        }
        output
    }
    for model in ["mlp2020", "emlp1", "emlp2", "emlp3"] {
        let mut prior = None;
        for seed in [0, 7] {
            let m =
                materialize(&registry.resolve(&format!("iris__{model}"), seed).unwrap()).unwrap();
            let (c, r) = derive_identities(&m).unwrap();
            let mut document = parse_failure_document(golden::FAILURE_NULL.as_bytes()).unwrap();
            let identity = serde_json::json!({"profile_id":format!("iris__{model}"),"seed":seed,"materialized_sha256":r.materialized_digest(),"contract":{"schema":CONTRACT_VERSION,"encoding":ENCODING_VERSION,"canonical_bytes":hex(c.canonical_bytes()),"sha256":c.digest()},"seed_receipt":{"schema":RECEIPT_VERSION,"encoding":ENCODING_VERSION,"canonical_bytes":hex(r.canonical_bytes()),"sha256":r.digest()}});
            let text = identity.to_string();
            document.body.identity = Some(
                ClaimedIdentity::decode(serde_json::from_str(&text).unwrap(), "identity").unwrap(),
            );
            let mut json: serde_json::Value = serde_json::from_str(golden::FAILURE_NULL).unwrap();
            json["identity"] = identity;
            json["integrity"]["payload_sha256"] = serde_json::json!(sha256_hex(
                &failure_encoding(&document.common, &document.body)
                    .unwrap()
                    .bytes
            ));
            // A no-progress adversarial failure envelope exercises transport, never fake completion.
            let parsed = parse_failure_document(json.to_string().as_bytes()).unwrap();
            let got = parsed.claimed_identity().unwrap();
            assert_eq!(got.contract().canonical_bytes(), c.canonical_bytes());
            assert_eq!(got.seed_receipt().canonical_bytes(), r.canonical_bytes());
            assert_eq!(got.seed_receipt().sha256(), r.digest());
            if let Some((contract, receipt)) = &prior {
                assert_eq!(contract, c.digest());
                assert_ne!(receipt, r.digest());
            }
            prior = Some((c.digest().to_owned(), r.digest().to_owned()));
        }
    }
}

#[test]
fn fix1_genuine_structural_and_escaped_keys_keep_every_string_exception() {
    fn nested(parts: &[&str], payload: &str) -> serde_json::Value {
        let mut value = serde_json::json!(payload);
        for part in parts.iter().rev() {
            value = serde_json::json!({*part: value});
        }
        value
    }
    let cases = [
        (["identity", "contract", "canonical_bytes"].as_slice(), 4098),
        (
            ["identity", "seed_receipt", "canonical_bytes"].as_slice(),
            4098,
        ),
        (
            ["provenance", "value", "build", "compiler_vv"].as_slice(),
            16384,
        ),
        (
            ["provenance", "value", "source", "root", "bytes"].as_slice(),
            8192,
        ),
        (
            ["provenance", "value", "executable", "path", "bytes"].as_slice(),
            8192,
        ),
        (
            ["provenance", "value", "invocation", "cwd", "bytes"].as_slice(),
            8192,
        ),
    ];
    for (parts, len) in cases {
        let input = nested(parts, &"0".repeat(len)).to_string();
        preflight(input.as_bytes()).unwrap();
        let mut escaped = input;
        for key in parts {
            let replacement = format!("\"\\u{:04x}{}\"", key.as_bytes()[0], &key[1..]);
            escaped = escaped.replace(&format!("\"{key}\""), &replacement);
        }
        preflight(escaped.as_bytes()).unwrap();
    }
}
#[test]
fn fix1_genuine_argv_array_elements_keep_os_byte_exception() {
    // Multiple large siblings exercise push/pop restoration across real array elements
    // and every object-only exception in the same document.
    let input = serde_json::json!({
        "identity": {
            "contract": {"canonical_bytes": "00".repeat(2049)},
            "seed_receipt": {"canonical_bytes": "ff".repeat(2049)}
        },
        "provenance": {"value": {
            "build": {"compiler_vv": "x".repeat(16384)},
            "source": {"root": {"bytes": "00".repeat(4096)}},
            "executable": {"path": {"bytes": "00".repeat(4096)}},
            "invocation": {
                "cwd": {"bytes": "00".repeat(4096)},
                "argv": [{"bytes":"00".repeat(4096)}, {"bytes":"ff".repeat(4096)}, {"bytes":""}]
            }
        }}
    })
    .to_string();
    preflight(input.as_bytes()).unwrap();
    let escaped = input
        .replace("\"argv\"", "\"\\u0061rgv\"")
        .replace("\"bytes\"", "\"\\u0062ytes\"");
    preflight(escaped.as_bytes()).unwrap();
}
