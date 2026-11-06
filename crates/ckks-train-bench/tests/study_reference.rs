//! Bounded identity checks: no training, HE state or reference artifacts.
use ckks_train_bench::{
    study_inputs::materialize,
    study_profile::{Initialization, StudyRegistry, StudySourceRole},
    study_reference::derive_identities,
};
use std::path::Path;
fn registry() -> StudyRegistry {
    StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap()
}
// A seed value, initialized array, selected order or numeric rate leaking into the
// shared projection breaks compatibility, while omitting them from receipts aliases seeds.
#[test]
fn actual_iris_all_four_graphs_share_contract_but_keep_own_seed_receipts() {
    let r = registry();
    for model in ["mlp2020", "emlp1", "emlp2", "emlp3"] {
        let a = materialize(&r.resolve(&format!("iris__{model}"), 0).unwrap()).unwrap();
        let b = materialize(&r.resolve(&format!("iris__{model}"), 7).unwrap()).unwrap();
        let (ac, ar) = derive_identities(&a).unwrap();
        let (bc, br) = derive_identities(&b).unwrap();
        assert_eq!(ac.canonical_bytes(), bc.canonical_bytes(), "{model}");
        assert_eq!(ac.digest(), bc.digest());
        assert_ne!(ar.digest(), br.digest());
        assert_eq!(ar.seed(), 0);
        assert_eq!(br.seed(), 7);
        assert_eq!(ar.materialized_digest(), a.identity().digest());
        assert_eq!(br.materialized_digest(), b.identity().digest());
        assert_ne!(a.identity().weights_sha256(), b.identity().weights_sha256());
        assert_ne!(a.identity().order_sha256(), b.identity().order_sha256());
        assert_eq!(ar.contract_digest(), ac.digest());
    }
}
// Descriptor derivation must not load feature arrays or require missing captured exports.
#[test]
fn all_21_profiles_all_ten_seeds_have_immutable_shared_source_roles() {
    let r = registry();
    let mut count = 0;
    for id in r.profile_ids() {
        let a = r.resolve(&id, 0).unwrap();
        for seed in 0..10 {
            let b = r.resolve(&id, seed).unwrap();
            assert_eq!(a.shared_selector(), b.shared_selector(), "{id}/{seed}");
            let sources = b.shared_selector().sources();
            assert!(!sources.is_empty());
            assert!(sources.iter().all(|s| !s.path().starts_with('/')));
            let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let m = b.metadata();
            for expected in std::iter::once(&m.identity.registry)
                .chain(std::iter::once(&m.identity.integrity_lock))
                .chain(&m.identity.sources)
                .chain(&m.identity.dataset_files)
            {
                assert!(
                    sources
                        .iter()
                        .any(|s| root.join(s.path()) == expected.path
                            && s.sha256() == expected.sha256),
                    "omitted shared source {id}/{seed}"
                );
            }
            assert_eq!(
                sources
                    .iter()
                    .filter(|s| s.role() == StudySourceRole::Registry)
                    .count(),
                1
            );
            assert_eq!(
                sources
                    .iter()
                    .filter(|s| s.role() == StudySourceRole::PortableLock)
                    .count(),
                1
            );
            assert_eq!(
                sources
                    .iter()
                    .filter(|s| s.role() == StudySourceRole::LocalRecipeCode)
                    .count(),
                2
            );
            assert!(sources.iter().all(|s| !matches!(
                s.role(),
                StudySourceRole::SelectedInitialization | StudySourceRole::SelectedOrder
            )));
            if let Some(schedule) = &m.schedule {
                assert!(
                    sources
                        .iter()
                        .any(|s| s.role() == StudySourceRole::ScheduleContainer
                            && root.join(s.path()) == schedule.source.path)
                );
                assert!(b.shared_selector().captured().is_some());
            } else {
                assert!(b.shared_selector().captured().is_none());
            }
            if let Initialization::Captured(arrays) = &m.initialization {
                for a in arrays {
                    assert!(sources.iter().all(|s| root.join(s.path()) != a.file.path));
                }
            }
            if let Some(order) = &m.captured_order {
                assert!(sources.iter().all(|s| root.join(s.path()) != order.path));
            }
        }
        count += 1;
    }
    assert_eq!(count, 21);
}
#[test]
fn missing_captured_seed_is_still_missing_before_identity_construction() {
    let r = registry();
    assert!(materialize(&r.resolve("mnist_released__emlp3", 7).unwrap()).is_err());
}
