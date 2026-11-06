use super::*;
use std::collections::VecDeque;
struct SyntheticClock {
    wall: VecDeque<Result<u64>>,
    monotonic: VecDeque<Result<u64>>,
}
impl Clock for SyntheticClock {
    fn wall(&mut self) -> Result<u64> {
        self.wall.pop_front().expect("wall sequence")
    }
    fn monotonic(&mut self) -> Result<u64> {
        self.monotonic.pop_front().expect("monotonic sequence")
    }
}
#[test]
fn observed_clock_reversal_or_unavailability_is_sticky() {
    for wall_fault in [true, false] {
        let mut clock = SyntheticClock {
            wall: [Ok(10), Ok(if wall_fault { 9 } else { 11 }), Ok(12)].into(),
            monotonic: [Ok(4), Ok(if wall_fault { 5 } else { 3 }), Ok(6)].into(),
        };
        let mut tracker = ClockTracker::begin(&mut clock).unwrap();
        assert!(tracker.sample(&mut clock).is_err());
        assert!(tracker.sample(&mut clock).is_err());
    }
    let mut clock = SyntheticClock {
        wall: [Ok(10), Err(TargetError::ClockInvalid), Ok(12)].into(),
        monotonic: [Ok(0), Ok(2)].into(),
    };
    let mut tracker = ClockTracker::begin(&mut clock).unwrap();
    assert!(tracker.sample(&mut clock).is_err());
    assert!(tracker.sample(&mut clock).is_err());
}
#[test]
fn clock_equalities_allowed_without_interpolation_or_tolerance() {
    let mut clock = SyntheticClock {
        wall: [Ok(10), Ok(10), Ok(u64::MAX)].into(),
        monotonic: [Ok(4), Ok(4), Ok(9)].into(),
    };
    let mut tracker = ClockTracker::begin(&mut clock).unwrap();
    assert_eq!(tracker.sample(&mut clock).unwrap(), (10, 0));
    assert_eq!(tracker.sample(&mut clock).unwrap(), (u64::MAX, 5));
}
#[test]
fn campaign_validation_prevents_path_selection() {
    for id in ["", ".", "..", "a/b", "a\\b", "é", "a\0b", &"a".repeat(129)] {
        assert!(CampaignId::parse(id).is_err());
    }
    for id in ["A_9-", "a", &"z".repeat(128)] {
        assert!(CampaignId::parse(id).is_ok());
    }
}
#[test]
fn iris_selected_seeds_keep_shared_bytes_and_distinct_actual_receipts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(&root).unwrap();
    let p0 = materialize(&registry.resolve("iris__emlp1", 0).unwrap()).unwrap();
    let p7 = materialize(&registry.resolve("iris__emlp1", 7).unwrap()).unwrap();
    let (c0, r0) = derive_identities(&p0).unwrap();
    let (c7, r7) = derive_identities(&p7).unwrap();
    assert_eq!(c0.canonical_bytes(), c7.canonical_bytes());
    assert_ne!(r0.canonical_bytes(), r7.canonical_bytes());
    assert_eq!((r0.seed(), r7.seed()), (0, 7));
}
#[test]
fn all_twenty_one_profiles_fit_reviewed_identity_caps_without_data_allocation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(&root).unwrap();
    let expected = [
        ("iris__mlp2020", 48000, 2400, 400, 83, 1128176, 1689180),
        ("iris__emlp1", 48000, 2400, 400, 292, 1128176, 1691688),
        ("iris__emlp2", 48000, 2400, 400, 2792, 1128176, 1721688),
        ("iris__emlp3", 48000, 2400, 400, 11884, 1128176, 1830792),
        (
            "breast_cancer__mlp2020",
            44800,
            2800,
            100,
            332,
            1123476,
            1688268,
        ),
        (
            "breast_cancer__emlp1",
            44800,
            2800,
            100,
            1058,
            1123476,
            1696980,
        ),
        (
            "breast_cancer__emlp2",
            44800,
            2800,
            100,
            4260,
            1123476,
            1735404,
        ),
        (
            "breast_cancer__emlp3",
            44800,
            2800,
            100,
            14758,
            1123476,
            1861380,
        ),
        (
            "mnist_released__mlp2020",
            48000,
            1000,
            10,
            2080,
            1082066,
            1563114,
        ),
        (
            "mnist_released__emlp1",
            48000,
            1000,
            10,
            6832,
            1082066,
            1620138,
        ),
        (
            "mnist_released__emlp2",
            48000,
            1000,
            10,
            16256,
            1082066,
            1733226,
        ),
        (
            "mnist_released__emlp3",
            48000,
            1000,
            10,
            39184,
            1082066,
            2008362,
        ),
        (
            "fashion_mnist__mlp2020",
            3000000,
            62500,
            50,
            2080,
            2613526,
            31138894,
        ),
        (
            "fashion_mnist__emlp1",
            3000000,
            62500,
            50,
            6832,
            2613526,
            31195918,
        ),
        (
            "fashion_mnist__emlp2",
            3000000,
            62500,
            50,
            16256,
            2613526,
            31309006,
        ),
        (
            "fashion_mnist__emlp3",
            3000000,
            62500,
            50,
            39184,
            2613526,
            31584142,
        ),
        (
            "mnist_released__mlp2020__b64",
            48000,
            750,
            10,
            2080,
            1073816,
            1536864,
        ),
        (
            "mnist_released__emlp1__b64",
            48000,
            750,
            10,
            6832,
            1073816,
            1593888,
        ),
        (
            "mnist_released__emlp2__b64",
            48000,
            750,
            10,
            16256,
            1073816,
            1706976,
        ),
        (
            "mnist_released__emlp3__b64",
            48000,
            750,
            10,
            39184,
            1073816,
            1982112,
        ),
        (
            "mnist_released__emlp1__b128_candidate",
            48000,
            375,
            10,
            6832,
            1061441,
            1554513,
        ),
    ];
    assert_eq!(registry.profile_ids().len(), expected.len());
    for (id, rows, updates, endpoints, parameters, cb, rb) in expected {
        let resolved = registry.resolve(id, 0).unwrap();
        let m = resolved.metadata();
        let train = resolved.data_request().train_samples.unwrap();
        let actual_rows = if let Some(order) = &m.captured_order {
            order.population * order.passes
        } else if m.drop_last {
            train / m.batch * m.batch * m.logical_epochs
        } else {
            train * m.logical_epochs
        };
        let actual_updates = if m.schedule.is_some() {
            actual_rows.div_ceil(m.batch)
        } else {
            (if m.drop_last {
                train / m.batch
            } else {
                train.div_ceil(m.batch)
            }) * m.logical_epochs
        };
        let params = m
            .layers
            .iter()
            .map(|l| l.input * l.output + l.output)
            .sum::<usize>();
        assert_eq!(
            (actual_rows, actual_updates, m.logical_epochs, params),
            (rows, updates, endpoints, parameters),
            "{id}"
        );
        let schedule = usize::from(m.schedule.is_some());
        assert_eq!(
            (25 + 8 * schedule) * updates + 49 * endpoints + 1048576,
            cb,
            "{id}"
        );
        assert_eq!(
            (97 + 8 * schedule) * updates
                + 8 * rows
                + 57 * endpoints
                + 8
                + 12 * parameters
                + 1048576,
            rb,
            "{id}"
        );
        assert!(cb <= 2613526 && rb <= 31584142 && 2 * cb + 2097152 <= 7324204);
        let sources =
            resolved.shared_selector().sources().len() + resolved.selected_sources().len();
        assert!(sources <= 64);
        assert!(format!("{m:?}").len() < 1048576);
        for source in resolved
            .shared_selector()
            .sources()
            .iter()
            .chain(resolved.selected_sources())
        {
            assert!(source.path().len() <= 4096);
        }
        println!(
            "{id}: rows={rows} updates={updates} endpoints={endpoints} parameters={parameters} C={cb} R={rb} lock_J={} sources={sources}",
            2 * cb + 2097152
        );
    }
    let registry_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("crates/ckks-train-bench/configs/experiment_matrix.json"))
            .unwrap(),
    )
    .unwrap();
    fn strings(v: &serde_json::Value) {
        match v {
            serde_json::Value::String(s) => assert!(s.len() <= 4096),
            serde_json::Value::Array(a) => a.iter().for_each(strings),
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    assert!(k.len() <= 4096);
                    strings(v)
                }
            }
            _ => {}
        }
    }
    strings(&registry_json);
}
#[test]
fn b128_actual_endpoint_topology_and_missing_block_seed_are_preserved() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(&root).unwrap();
    let resolved = registry
        .resolve("mnist_released__emlp1__b128_candidate", 0)
        .unwrap();
    let prepared = resolved.preflight(None).unwrap();
    assert_eq!(
        prepared
            .plan
            .checkpoints
            .iter()
            .map(|c| c.completed_updates)
            .collect::<Vec<_>>(),
        [38, 75, 113, 150, 188, 225, 263, 300, 338, 375]
    );
    assert_eq!(
        prepared
            .plan
            .checkpoints
            .iter()
            .map(|c| c.processed_samples)
            .collect::<Vec<_>>(),
        [
            4864, 9600, 14464, 19200, 24064, 28800, 33664, 38400, 43264, 48000
        ]
    );
    assert_eq!(
        prepared
            .plan
            .checkpoints
            .iter()
            .map(|c| c.interval_updates)
            .sum::<usize>(),
        375
    );
    assert_eq!(
        prepared
            .plan
            .checkpoints
            .iter()
            .map(|c| c.interval_samples)
            .sum::<usize>(),
        48000
    );
    assert!(prepared.plan.checkpoints.iter().all(
        |c| matches!(c.interval_updates, 37 | 38) && matches!(c.interval_samples, 4736 | 4864)
    ));
    let absent = registry.resolve("mnist_released__emlp2", 7).unwrap();
    assert!(matches!(
        absent.preflight(None),
        Err(crate::study_profile::ProfileError::MissingInput { .. })
    ));
}
#[test]
fn postlink_clock_or_context_failure_never_returns_even_synthetic_success() {
    struct SyntheticDurability;
    for error in [
        TargetError::ClockInvalid,
        TargetError::ContextChanged,
        TargetError::ByteOrPayloadMismatch,
    ] {
        let result = confirm_publication(
            SyntheticDurability,
            || Err(error),
            || TargetError::PublishedButDurabilityUnconfirmed {
                path: PathBuf::from("SYNTHETIC/target.lock.json"),
                file_sha256: "1".repeat(64),
                payload_sha256: "2".repeat(64),
            },
        );
        assert!(
            matches!(result,Err(TargetError::PublishedButDurabilityUnconfirmed{file_sha256,payload_sha256,..}) if file_sha256=="1".repeat(64)&&payload_sha256=="2".repeat(64))
        );
    }
    assert!(
        confirm_publication(SyntheticDurability, || Ok(()), || TargetError::ClockInvalid).is_ok()
    );
}
#[test]
fn synthetic_selected_seed_linkage_compares_shared_bytes_and_actual_seed0_receipt() {
    struct SyntheticSource {
        profile: String,
        contract: Vec<u8>,
        seed0_receipt: Vec<u8>,
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(&root).unwrap();
    let p0 = materialize(&registry.resolve("iris__emlp1", 0).unwrap()).unwrap();
    let p7 = materialize(&registry.resolve("iris__emlp1", 7).unwrap()).unwrap();
    let (contract, receipt) = derive_identities(&p0).unwrap();
    let source = SyntheticSource {
        profile: p0.metadata().id.clone(),
        contract: contract.canonical_bytes().into(),
        seed0_receipt: receipt.canonical_bytes().into(),
    };
    let r = selected_identity_matches(
        &source.profile,
        &source.contract,
        &source.seed0_receipt,
        &p7,
    )
    .unwrap();
    assert_eq!(r.seed(), 7);
    assert_ne!(r.canonical_bytes(), source.seed0_receipt);
    selected_identity_matches(
        &source.profile,
        &source.contract,
        &source.seed0_receipt,
        &p0,
    )
    .unwrap();
    let mut changed = source.contract.clone();
    changed[0] ^= 255;
    assert!(
        selected_identity_matches(&source.profile, &changed, &source.seed0_receipt, &p7).is_err()
    );
    let mut changed = source.seed0_receipt.clone();
    changed[0] ^= 255;
    assert!(selected_identity_matches(&source.profile, &source.contract, &changed, &p0).is_err());
    assert!(
        selected_identity_matches(
            "other-profile",
            &source.contract,
            &source.seed0_receipt,
            &p7
        )
        .is_err()
    );
}
