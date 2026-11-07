//! Test-only exact UTF8 custody for independently frozen literal documents.
use serde_json::Value;
use std::sync::OnceLock;

fn bundles() -> &'static [Value; 2] {
    static BUNDLES: OnceLock<[Value; 2]> = OnceLock::new();
    BUNDLES.get_or_init(|| {
        [
            (
                include_str!("../target_goldens/literals.json"),
                "60f2292de60d2409884bf5c7d00f56f482b4c3245eda5eddac935611f0a87cd3",
            ),
            (
                include_str!("case_provenance_goldens/literals.json"),
                "3c300767ab0d47364e5f49843766d62734b4d0688c992f6b4fd22ff72b5e337a",
            ),
        ]
        .map(|(text, expected_hash)| {
            // Pin the outer transport, including its complete member-name set.
            assert_eq!(crate::data::sha256_hex(text.as_bytes()), expected_hash);
            serde_json::from_str(text).expect("frozen literal bundle")
        })
    })
}

// Borrow the decoded outer string only. Inner JSON/hex/trailing bytes stay untouched.
pub(super) fn target(name: &str) -> &'static str {
    bundles()[0]["files"][name]["text"]
        .as_str()
        .expect("known target literal")
}

pub(super) fn case(name: &str) -> &'static str {
    bundles()[1]["files"][name]["text"]
        .as_str()
        .expect("known Case provenance literal")
}

#[test]
fn all_158_logical_members_retain_lengths_hashes_and_counts() {
    let bundles = bundles();
    for (bundle, count, bytes, literal) in [
        (&bundles[0], 77, 553493, target as fn(&str) -> &'static str),
        (&bundles[1], 81, 489492, case as fn(&str) -> &'static str),
    ] {
        let files = bundle["files"].as_object().unwrap();
        assert_eq!(files.len(), count);
        let mut total = 0;
        for (name, member) in files {
            let text = literal(name);
            assert_eq!(text, member["text"].as_str().unwrap());
            assert_eq!(text.len() as u64, member["byte_length"].as_u64().unwrap());
            assert_eq!(crate::data::sha256_hex(text.as_bytes()), member["sha256"]);
            total += text.len();
        }
        assert_eq!(total, bytes);
    }
}
