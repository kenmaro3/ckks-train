use super::*;
use std::path::PathBuf;
fn corpus() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../results/validation/task54-d7-resource-oracle/corpus")
}
fn bytes(name: &str) -> Vec<u8> {
    std::fs::read(corpus().join(name)).unwrap()
}
fn metadata(name: &str) -> serde_json::Value {
    let raw = bytes(name);
    if name == "manifest.json" {
        assert_eq!(
            sha256_hex(&raw),
            "3cdd988685c9ee3564d8af2a58072c29d5e23a9ee833a28167dfe76180164bd4"
        );
    }
    serde_json::from_slice(&raw).unwrap()
}
fn parse_named(name: &str, mode: Mode) -> Source<'static> {
    let b: &'static [u8] = match name {
        "inert-production/claim-reference-read.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-reference-read.json"
        )),
        "inert-production/claim-reference-recover.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-reference-recover.json"
        )),
        "inert-production/claim-reference-snapshot.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-reference-snapshot.json"
        )),
        "inert-production/claim-target-lock.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-target-lock.json"
        )),
        "inert-production/claim-target-read-future.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-target-read-future.json"
        )),
        "inert-production/claim-target-read-supplied.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-target-read-supplied.json"
        )),
        "inert-production/claim-train-full.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-train-full.json"
        )),
        "inert-production/claim-train-quick.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/claim-train-quick.json"
        )),
        "inert-production/disk-observation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/disk-observation.json"
        )),
        "inert-production/host-observation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/host-observation.json"
        )),
        "inert-production/host-policy.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/host-policy.json"
        )),
        "inert-production/model.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/model.json"
        )),
        "inert-production/validation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/inert-production/validation.json"
        )),
        "synthetic/claim-reference-read.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-reference-read.json"
        )),
        "synthetic/claim-reference-recover.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-reference-recover.json"
        )),
        "synthetic/claim-reference-snapshot.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-reference-snapshot.json"
        )),
        "synthetic/claim-target-lock.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-target-lock.json"
        )),
        "synthetic/claim-target-read-future.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-target-read-future.json"
        )),
        "synthetic/claim-target-read-supplied.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-target-read-supplied.json"
        )),
        "synthetic/claim-train-full.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-train-full.json"
        )),
        "synthetic/claim-train-quick.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/claim-train-quick.json"
        )),
        "synthetic/disk-observation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/disk-observation.json"
        )),
        "synthetic/host-observation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/host-observation.json"
        )),
        "synthetic/host-policy.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/host-policy.json"
        )),
        "synthetic/model.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/model.json"
        )),
        "synthetic/validation.json" => include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../results/validation/task54-d7-resource-oracle/corpus/synthetic/validation.json"
        )),
        _ => panic!("not an inventoried source: {name}"),
    };
    Source::parse(b, mode).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}
#[derive(Debug)]
struct NegativeCase {
    id: String,
    base: String,
    expect: String,
}
fn negative_cases() -> Vec<NegativeCase> {
    // One descriptor contains an intentionally unpaired Unicode surrogate as
    // its mutation value. Keep that value raw: only the literal id/base/expect
    // fields are test instructions; never repair or reserialize the corpus.
    let bytes = bytes("negative-cases.json");
    assert_eq!(
        sha256_hex(&bytes),
        "1cadd3ac6e7ddf0089b3653dac86c3d2e3f0edc4d4010d36dda4bfe1eb0c0f66"
    );
    let values: Vec<&RawValue> = serde_json::from_slice(&bytes).unwrap();
    values
        .into_iter()
        .map(|r| {
            let mut f = Fields::new(
                r,
                &[
                    "id", "base", "op", "path", "value", "expect", "from", "hash_of", "offset",
                ],
                "descriptor",
            )
            .unwrap();
            NegativeCase {
                id: f.field::<Text<128>>("id", "descriptor").unwrap().0,
                base: f.field::<Text<128>>("base", "descriptor").unwrap().0,
                expect: f.field::<Text<128>>("expect", "descriptor").unwrap().0,
            }
        })
        .collect()
}
fn baseline(mode: Mode) -> Baseline<'static> {
    let prefix = if mode == Mode::Production {
        "inert-production"
    } else {
        "synthetic"
    };
    Baseline::new(
        parse_named(&format!("{prefix}/model.json"), mode),
        parse_named(&format!("{prefix}/validation.json"), mode),
        parse_named(&format!("{prefix}/host-policy.json"), mode),
        parse_named(&format!("{prefix}/claim-target-read-supplied.json"), mode),
        parse_named(&format!("{prefix}/host-observation.json"), mode),
        parse_named(&format!("{prefix}/disk-observation.json"), mode),
    )
    .unwrap()
}
#[test]
fn frozen_positive_sources_preserve_whole_file_identity() {
    let manifest = metadata("manifest.json");
    for (name, pin) in manifest.as_object().unwrap() {
        if (name.starts_with("synthetic/") || name.starts_with("inert-production/"))
            && name.ends_with(".json")
        {
            let mode = if name.starts_with("synthetic/") {
                Mode::Oracle
            } else {
                Mode::Production
            };
            let s = parse_named(name, mode);
            assert_eq!(
                s.raw.len() as u64,
                pin["length"].as_u64().unwrap(),
                "{name}"
            );
            assert_eq!(s.sha256(), pin["sha256"].as_str().unwrap(), "{name}");
            baseline(mode)
                .check(&s, supplied(), physical(), 1)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }
}
fn supplied() -> &'static str {
    "bbf9372f3e913d631e8c0a911747532611cb60ebe5ef84b2c328375e84224439"
}
fn physical() -> &'static str {
    "2711bf73d369cfecd6687dccce4d57a21bd616e1d388af6e8e8002dec6b63abe"
}
#[test]
fn frozen_negative_sources_reach_independent_refusals() {
    let base = baseline(Mode::Oracle);
    let manifest = metadata("manifest.json");
    let cases = negative_cases();
    let mut count = 0;
    let mut failures = Vec::new();
    for c in &cases {
        if c.base == "witness" || c.base == "bootstrap" {
            continue;
        }
        count += 1;
        let name = c.id.as_str();
        let b = bytes(&format!("negative/{name}.json"));
        assert_eq!(
            b.len() as u64,
            manifest[format!("negative/{name}.json")]["length"]
                .as_u64()
                .unwrap()
        );
        assert_eq!(
            sha256_hex(&b),
            manifest[format!("negative/{name}.json")]["sha256"]
                .as_str()
                .unwrap()
        );
        let error = match Source::parse(&b, Mode::Oracle)
            .and_then(|s| base.check(&s, supplied(), physical(), 1))
        {
            Ok(()) => {
                failures.push(format!("{name}: unexpected acceptance"));
                continue;
            }
            Err(e) => e,
        };
        let expected = c.expect.as_str();
        let expected = match expected {
            "unicode" | "utf8" => "json",
            x => x,
        };
        if error.reason != expected {
            failures.push(format!("{name}: {error:?}, expected {expected}"));
        }
        let field = expected_field(name);
        if error.field != field {
            failures.push(format!("{name}: field {}, expected {field}", error.field));
        }
    }
    assert_eq!(count, 67);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
#[test]
fn production_refuses_all_synthetic_sources_and_anchor() {
    let manifest = metadata("manifest.json");
    let mut count = 0;
    for name in manifest
        .as_object()
        .unwrap()
        .keys()
        .filter(|s| s.starts_with("synthetic/"))
    {
        let b = bytes(name);
        if name.ends_with(".json") {
            assert_eq!(
                Source::parse(&b, Mode::Production).unwrap_err().reason,
                "domain"
            );
        } else {
            assert!(Bootstrap::parse(&b, Mode::Production).is_err());
        }
        count += 1;
    }
    assert_eq!(count, 14);
}
#[test]
fn frozen_bootstrap_literal_bytes_and_projections() {
    let manifest = metadata("manifest.json");
    for (prefix, mode, len) in [
        ("synthetic", Mode::Oracle, 703),
        ("inert-production", Mode::Production, 704),
    ] {
        let b = bytes(&format!("{prefix}/bootstrap.bin"));
        let a = Bootstrap::parse(&b, mode).unwrap();
        assert_eq!(
            sha256_hex(&b),
            manifest[format!("{prefix}/bootstrap.bin")]["sha256"]
                .as_str()
                .unwrap()
        );
        assert_eq!(b.len(), len);
        assert_eq!(a.generation, 3);
        assert_eq!(a.account_uid, 1000);
        assert_eq!(a.namespaces, [[11, 21], [12, 22], [13, 23], [14, 24]]);
        assert_eq!(a.proc_root, [31, 41, 51, 40864]);
        assert_eq!(a.cgroup_root, [32, 42, 52, 1667723888]);
        assert_eq!(a.boot_uuid, "00000000-0000-0000-0000-000000000001");
        assert_eq!(a.encode_fixed().unwrap().as_slice(), b);
        baseline(mode)
            .check_bootstrap(&a, &Bootstrap::parse(&b, mode).unwrap())
            .unwrap();
    }
}
#[test]
fn frozen_bootstrap_negatives_reach_independent_refusals() {
    let base = baseline(Mode::Oracle);
    let manifest = metadata("manifest.json");
    let good = bytes("synthetic/bootstrap.bin");
    let expected = Bootstrap::parse(&good, Mode::Oracle).unwrap();
    let cases = negative_cases();
    let mut count = 0;
    for c in cases.iter().filter(|v| v.base == "bootstrap") {
        count += 1;
        let name = c.id.as_str();
        let b = bytes(&format!("negative/{name}.bin"));
        assert_eq!(
            sha256_hex(&b),
            manifest[format!("negative/{name}.bin")]["sha256"]
                .as_str()
                .unwrap()
        );
        let e = Bootstrap::parse(&b, Mode::Oracle)
            .and_then(|a| base.check_bootstrap(&a, &expected))
            .unwrap_err();
        assert_eq!(e.reason, c.expect.as_str(), "{name}: {e:?}");
        assert_eq!(
            e.field,
            match &name[..3] {
                "n69" | "n76" => "bootstrap.length",
                "n70" => "bootstrap.policy",
                "n71" => "bootstrap.namespaces",
                "n72" => "bootstrap.origin_evidence",
                "n73" => "bootstrap.kernel_profile",
                "n74" => "bootstrap.domain",
                "n75" => "bootstrap.synthetic",
                "n77" => "bootstrap.hash",
                _ => panic!("unregistered anchor case"),
            }
        );
    }
    assert_eq!(count, 9);
}
#[test]
fn small_checked_bounds_without_cap_sized_allocation() {
    assert!(check_source_length(8_388_608).is_ok());
    assert_eq!(
        check_source_length(8_388_609).unwrap_err().reason,
        "raw-cap"
    );
    assert_eq!(
        checked_disk_sum([u64::MAX, 1]).unwrap_err().reason,
        "arithmetic-overflow"
    );
    assert!(checked_disk_sum([u64::MAX]).is_ok());
}
#[test]
fn raw_identity_preserves_whitespace_and_member_order() {
    let b = bytes("synthetic/disk-observation.json");
    let mut other = b.clone();
    other.push(b' ');
    let a = Source::parse(&b, Mode::Oracle).unwrap();
    let text = std::str::from_utf8(&b)
        .unwrap()
        .trim()
        .strip_prefix('{')
        .unwrap()
        .strip_suffix('}')
        .unwrap();
    let (first, rest) = text.split_once(',').unwrap();
    let reordered = format!("{{{rest},{first}}}\n");
    let order = Source::parse(reordered.as_bytes(), Mode::Oracle).unwrap();
    assert_ne!(a.sha256(), order.sha256());
    assert_eq!(a.body, order.body);
    let z = Source::parse(&other, Mode::Oracle).unwrap();
    assert_ne!(a.sha256(), z.sha256());
    assert_eq!(a.body, z.body);
}
#[test]
fn primitive_and_closed_nested_controls() {
    let source = String::from_utf8(bytes("synthetic/disk-observation.json")).unwrap();
    for (from, to, reason) in [
        ("\"frsize\":4", "\"frsize\":0", "disk-frsize"),
        ("\"roots\":[0,1,2,3,4,5]", "\"roots\":[]", "root-order"),
        ("\"required_files\":2", "\"required_files\":2.0", "integer"),
    ] {
        let changed = source.replace(from, to);
        assert_ne!(source, changed);
        assert_eq!(
            Source::parse(changed.as_bytes(), Mode::Oracle)
                .unwrap_err()
                .reason,
            reason
        );
    }
}

// Exact logical first-refusal fields selected from each frozen mutation and
// prerequisite relation; this table is literal, never emitted by the decoder.
fn expected_field(id: &str) -> &'static str {
    match &id[..3] {
        "n01" | "n02" | "n03" => "model.components",
        "n04" => "model.components.rule",
        "n05" => "model.rule_revision",
        "n06" | "n16" | "n17" | "n18" => "source",
        "n07" => "schema",
        "n08" => "model",
        "n09" => "model.revision",
        "n10" | "n11" | "n12" | "n13" | "n14" | "n15" => "host.sequence",
        "n19" => "revision",
        "n20" => "validation.source_manifest",
        "n21" => "validation.source_revision",
        "n22" => "validation.environment.build_receipt",
        "n23" => "validation.runs.metadata.environment",
        "n24" | "n25" => "policy.validation",
        "n26" => "validation.model",
        "n27" => "schema",
        "n28" => "synthetic",
        "n29" => "model.source_files.path",
        "n30" | "n33" => "validation.runs",
        "n31" | "n32" => "validation.coverage.validation_runs",
        "n34" | "n35" => "model.source_files",
        "n36" | "n37" => "model.supported_work",
        "n38" | "n39" | "n40" | "n41" => "claim.work.expected_file",
        "n42" => "claim.work.seed",
        "n43" | "n44" => "claim.work.target",
        "n45" | "n46" | "n47" | "n48" | "n49" => "claim.work.supplied_input",
        "n53" => "claim.work.request",
        "n54" => "claim.work",
        "n55" => "host.claim_id",
        "n56" => "disk.sequence",
        "n57" => "host.physical",
        "n58" => "host.topology_after.namespaces",
        "n59" => "host.structural_root",
        "n60" => "host.controllers.path",
        "n62" => "disk.filesystems.roots",
        "n63" => "disk.filesystems.bavail",
        "n65" => "claim.modeled_envelope.components",
        "n66" => "claim.modeled_envelope.peak_host_bytes",
        "n67" => "claim.modeled_envelope.incremental_host_bytes",
        "n78" => "work.registration",
        "n79" => "validation.runs.metadata.exit_code",
        "n80" => "validation.runs.metadata.argv",
        "n81" => "disk.filesystems.required_files",
        "n85" => "model.registry",
        _ => panic!("unregistered source case {id}"),
    }
}

#[test]
fn legacy_outer_full_quick_resource_bytes_unchanged() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../results/validation/task54-d7-gate-schema-design/task-54-d7-gate-schema-vectors",
    );
    for scope in ["full", "quick"] {
        let raw = std::fs::read(root.join(format!("{scope}.resource.evidence.bin"))).unwrap();
        let value = evidence::Evidence::parse_oracle(&raw).unwrap();
        assert_eq!(value.canonical().unwrap(), raw);
        assert_eq!(
            evidence::Evidence::parse(&raw).unwrap_err().code(),
            DocumentErrorCode::UnsupportedSchema
        );
    }
    let raw = std::fs::read(root.join("negative/resource-peak.bin")).unwrap();
    assert_eq!(
        evidence::Evidence::parse_oracle(&raw).unwrap_err().code(),
        DocumentErrorCode::CrossLinkMismatch
    );
}
#[test]
fn new_resource_byte_paths_and_integer_lexemes() {
    let source = String::from_utf8(bytes("synthetic/host-observation.json")).unwrap();
    for token in [
        "-1",
        "-0",
        "1.0",
        "1e0",
        "\"1\"",
        "true",
        "18446744073709551616",
    ] {
        let changed = source.replace("\"sequence\":1", &format!("\"sequence\":{token}"));
        assert_eq!(
            Source::parse(changed.as_bytes(), Mode::Oracle)
                .unwrap_err()
                .reason,
            "integer",
            "{token}"
        );
    }
    let validation = String::from_utf8(bytes("synthetic/validation.json")).unwrap();
    for token in ["-2147483648", "2147483647", "null", "-1"] {
        let changed = validation.replacen("\"exit_code\":0", &format!("\"exit_code\":{token}"), 1);
        Source::parse(changed.as_bytes(), Mode::Oracle).unwrap();
    }
    for token in [
        "-2147483649",
        "2147483648",
        "-0",
        "1e0",
        "1.0",
        "true",
        "\"0\"",
    ] {
        let changed = validation.replacen("\"exit_code\":0", &format!("\"exit_code\":{token}"), 1);
        assert_eq!(
            Source::parse(changed.as_bytes(), Mode::Oracle)
                .unwrap_err()
                .reason,
            "integer",
            "{token}"
        );
    }
    // Actual argv binary spelling beyond the old4096-character text path.
    let old = "[\"66696374696f6e616c2d76616c6964617465\",\"ff\",\"\"]";
    let changed = validation.replacen(old, &format!("[\"{}\",\"ff\",\"\"]", "ff".repeat(4096)), 1);
    assert_ne!(changed, validation);
    let s = Source::parse(changed.as_bytes(), Mode::Oracle).unwrap();
    let Body::Validation(v) = s.body else {
        panic!()
    };
    assert_eq!(v.runs.0[0].metadata.argv.0[0].0.len(), 4096);
    let changed = validation.replacen(old, "[\"\\u0066f\"]", 1);
    assert_eq!(
        Source::parse(changed.as_bytes(), Mode::Oracle)
            .unwrap_err()
            .reason,
        "bytes-spelling"
    );
    let train = String::from_utf8(bytes("synthetic/claim-train-full.json")).unwrap();
    let registration = train
        .split("\"registration\":\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    let changed = train.replacen(
        registration,
        &format!("{registration}{}", "00".repeat(2100)),
        1,
    );
    preflight_layout(changed.as_bytes(), DocumentLayout::Resource).unwrap();
    assert_eq!(
        Source::parse(changed.as_bytes(), Mode::Oracle)
            .unwrap_err()
            .reason,
        "registration"
    );
    // Same lexeme at a legacy path receives its original text cap.
    assert!(preflight_layout(changed.as_bytes(), DocumentLayout::Start).is_err());
}
#[test]
fn closed_nested_paths_unicode_and_production_flags() {
    let disk = String::from_utf8(bytes("inert-production/disk-observation.json")).unwrap();
    let changed = disk.replace("\"synthetic\":false", "\"synthetic\":true");
    assert_eq!(
        Source::parse(changed.as_bytes(), Mode::Production)
            .unwrap_err()
            .reason,
        "domain"
    );
    for file in [
        "/absolute",
        "a//b",
        "a/../b",
        "a/./b",
        "a/",
        "",
        "a\\b",
        "a\0b",
    ] {
        let changed = disk.replace(
            "\"fictional/facts.txt\"",
            &serde_json::to_string(file).unwrap(),
        );
        assert_eq!(
            Source::parse(changed.as_bytes(), Mode::Production)
                .unwrap_err()
                .reason,
            "fileref-path",
            "{file:?}"
        );
    }
    // Unicode byte counts are after decoding; member/string escaping is legal.
    let changed = disk.replace("fictional/facts.txt", &"é".repeat(2048));
    Source::parse(changed.as_bytes(), Mode::Production).unwrap();
    let changed = disk.replace("fictional/facts.txt", &"é".repeat(2049));
    assert!(Source::parse(changed.as_bytes(), Mode::Production).is_err());
    let changed = disk.replace("fictional/facts.txt", "\\u0066ictional/facts.txt");
    Source::parse(changed.as_bytes(), Mode::Production).unwrap();
    let changed = disk.replace("\"device\":3", "\"device\":3,\"extra\":true");
    assert_eq!(
        Source::parse(changed.as_bytes(), Mode::Production)
            .unwrap_err()
            .reason,
        "keys"
    );
    let changed = disk.replace("\"type\":5", "\"type\":5,\"\\u0074ype\":5");
    assert_eq!(
        Source::parse(changed.as_bytes(), Mode::Production)
            .unwrap_err()
            .reason,
        "duplicate-key"
    );
    let model = String::from_utf8(bytes("synthetic/model.json")).unwrap();
    for token in ["3", "-1", "true", "1e0"] {
        let changed = model.replacen(
            "\"supported_work\":[{\"kind\":0",
            &format!("\"supported_work\":[{{\"kind\":{token}"),
            1,
        );
        assert_ne!(model, changed);
        let error = Source::parse(changed.as_bytes(), Mode::Oracle).unwrap_err();
        assert_eq!(error.reason, if token == "3" { "tag" } else { "integer" });
    }
    // Exact primitive/count limits without allocating an8MiB or65536-item fixture.
    for (count, max) in [(512usize, 512usize), (256, 256), (65536, 65536)] {
        assert!(bounded_add(count, 0, max, "list").is_ok());
        assert!(bounded_add(count, 1, max, "list").is_err());
    }
    assert!(hex_len(2 * A_MAX, A_MAX, "bytes").is_ok());
    assert!(hex_len(2 * A_MAX + 2, A_MAX, "bytes").is_err());
}
#[test]
fn quick_train_preserves_registered_seed_zero_scope() {
    // Existing frozen CaseIdentity grammar puts seed at byte207 for this exact
    // independently pinned231-byte registration. This mutates source input only.
    for scope in ["full", "quick"] {
        let source =
            String::from_utf8(bytes(&format!("synthetic/claim-train-{scope}.json"))).unwrap();
        let registration = source
            .split("\"registration\":\"")
            .nth(1)
            .unwrap()
            .split('"')
            .next()
            .unwrap();
        assert_eq!(registration.len(), 462);
        assert_eq!(&registration[414..430], "0000000000000000");
        let mut mutated = registration.to_owned();
        mutated.replace_range(414..416, "07");
        let changed = source.replacen(registration, &mutated, 1);
        if scope == "quick" {
            assert_eq!(
                Source::parse(changed.as_bytes(), Mode::Oracle)
                    .unwrap_err()
                    .reason,
                "quick-seed"
            );
        } else {
            // Syntax alone cannot enroll a new case in the selected Model.
            let candidate = Source::parse(changed.as_bytes(), Mode::Oracle).unwrap();
            assert_eq!(
                baseline(Mode::Oracle)
                    .check(&candidate, supplied(), physical(), 1)
                    .unwrap_err()
                    .reason,
                "work-membership"
            );
        }
    }
}
#[test]
fn closed_inventory_counts_refuse_during_preflight() {
    for (file, needle, extra, field) in [
        (
            "synthetic/model.json",
            "],\"bootstrap_derivation\"",
            ",{}",
            "components",
        ),
        ("synthetic/validation.json", "],\"runs\"", ",{}", "coverage"),
        (
            "synthetic/claim-target-lock.json",
            "],\"peak_host_bytes\"",
            ",{}",
            "modeled_envelope.components",
        ),
        (
            "synthetic/disk-observation.json",
            "],\"frsize\"",
            ",0",
            "filesystems.*.roots",
        ),
    ] {
        let mut source = String::from_utf8(bytes(file)).unwrap();
        let at = source.find(needle).unwrap();
        source.insert_str(at, extra);
        let e = preflight_layout(source.as_bytes(), DocumentLayout::Resource).unwrap_err();
        assert_eq!(e.code(), DocumentErrorCode::TooLarge, "{file}");
        assert_eq!(e.path(), field, "{file}");
    }
    let mut source = String::from_utf8(bytes("synthetic/validation.json")).unwrap();
    let prefix = format!(
        "\"covered_components\":[{}",
        (0..22).map(|n| n.to_string()).collect::<Vec<_>>().join(",")
    );
    let at = source.find(&prefix).unwrap() + prefix.len();
    source.insert_str(at, ",0");
    let e = preflight_layout(source.as_bytes(), DocumentLayout::Resource).unwrap_err();
    assert_eq!(e.code(), DocumentErrorCode::TooLarge);
    assert_eq!(e.path(), "runs.*.covered_components");
}

#[test]
fn observed_cpu_bootstrap_mode_matches_static_baseline() {
    let baseline = baseline(Mode::Production);
    let raw = bytes("inert-production/bootstrap.bin");
    let mut anchor = Bootstrap::parse(&raw, Mode::Production).unwrap();
    anchor.mode = Mode::Oracle;
    let encoded = anchor.encode_fixed().unwrap();
    let oracle = Bootstrap::parse(encoded.as_slice(), Mode::Oracle).unwrap();
    assert!(
        baseline.check_bootstrap(&oracle, &oracle).is_err(),
        "Q1 cross-baseline mode must refuse even in cfg(test)"
    );
}
