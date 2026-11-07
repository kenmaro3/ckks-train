//! P3 binding tests only: secure metadata and one bundled canonical Iris materialization.
use ckks_train_bench::{
    study_admission::*,
    study_inputs::{MaterializedProfile, materialize},
    study_profile::*,
};
use ckks_train_ckks::{
    CkksParams,
    bootstrap::{
        BootParams,
        metadata::{BootstrapMetadata, DftSupportProof},
    },
};
use std::{
    collections::BTreeSet,
    path::Path,
    sync::{Arc, OnceLock},
};

struct Fixtures {
    registry: StudyRegistry,
    cases: RegisteredCases,
    interactive: CkksParams,
    noninteractive: CkksParams,
    bp: BootParams,
    boot: BootstrapMetadata,
    iris: OnceLock<MaterializedProfile>,
    budget: Arc<PlanningBudget>,
}
fn fixtures() -> &'static Fixtures {
    static FIXTURES: OnceLock<Fixtures> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let registry =
            StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let cases = registry.registered_cases().unwrap();
        let interactive = CkksParams::n16_train();
        let noninteractive = CkksParams::n16_boot();
        let bp = BootParams::default_n16();
        let proof = DftSupportProof::verify_current().unwrap();
        let boot =
            BootstrapMetadata::new_with_verified_support(&noninteractive, &bp, &proof).unwrap();
        eprintln!("P3 integration fixture: params=2 proof=1 bootstrap_metadata=1; HE=0");
        Fixtures {
            registry,
            cases,
            interactive,
            noninteractive,
            bp,
            boot,
            iris: OnceLock::new(),
            budget: Arc::new(PlanningBudget::new(PlanningUsage {
                geometries: 148,
                initial_states: 148,
                updates: 8,
                handoffs: 0,
                wrapper_calls: 8,
            })),
        }
    })
}
fn request() -> DiagonalRequest {
    let mut request = DiagonalRequest::quick();
    request.work_budget = fixtures().budget.clone();
    request
}
fn main_case(id: &str, placement: CasePlacement) -> &'static RegisteredCase {
    fixtures()
        .cases
        .family_cases()
        .iter()
        .find(|c| {
            c.role() == CaseRole::Main
                && c.selector().backend == CaseBackend::Cpu
                && c.selector().seed == 0
                && c.selector().profile_id == id
                && c.selector().placement == placement
        })
        .unwrap()
}
fn parameters(placement: CasePlacement) -> PlanningParameters<'static> {
    let f = fixtures();
    if placement == CasePlacement::ClientRefresh {
        PlanningParameters {
            params: &f.interactive,
            bootstrap: None,
            boot_params: None,
        }
    } else {
        PlanningParameters {
            params: &f.noninteractive,
            bootstrap: Some(&f.boot),
            boot_params: Some(&f.bp),
        }
    }
}

// Regression trigger: dropping a failed/missing candidate, seed-dependent geometry,
// or losing NI physical-key roles must fail this inventory binding test.
#[test]
fn p3_inventory_and_candidate_separation() {
    let f = fixtures();
    let before = f.budget.usage();
    assert_eq!(
        (
            f.cases.family_cases().len(),
            f.cases.exact_iris_cases().len(),
            f.cases.supplements().len()
        ),
        (2680, 20, 7)
    );
    let ids: BTreeSet<_> = f
        .cases
        .family_cases()
        .iter()
        .map(|c| c.selector().profile_id.as_str())
        .collect();
    assert_eq!(ids.len(), 21);
    let mut rows = 0;
    let mut stages = 0;
    for id in ids {
        let metadata = f.registry.resolve(id, 0).unwrap();
        for placement in [
            CasePlacement::ClientRefresh,
            CasePlacement::WhenNeeded,
            CasePlacement::AfterUpdate,
            CasePlacement::AfterActivation,
        ] {
            let row = describe_candidate(
                metadata.metadata(),
                main_case(id, placement),
                parameters(placement),
                &request(),
            );
            assert!(
                row.issue().is_none(),
                "{id}/{placement:?}: {:?}",
                row.issue()
            );
            assert_eq!(row.input_status(), InputStatus::Unverified);
            assert!(row.unproved().contains(&UnprovedGate::Security));
            assert!(row.unproved().contains(&UnprovedGate::Resource));
            assert!(!row.geometry_digest().is_empty());
            for state in row.initial_states() {
                assert_eq!(
                    state.parameter_identity().q(),
                    parameters(placement).params.q()
                );
            }
            if placement != CasePlacement::ClientRefresh {
                let keys = row.keys().unwrap();
                assert!(keys.bootstrap_conjugation());
                assert!(keys.bootstrap_sse());
                assert!(keys.separate_bootstrap_bundle());
            }
            stages += row.geometries().len();
            rows += 1;
        }
    }
    assert_eq!((rows, stages), (84, 144));
    // Metadata comparison below does not repeat geometry construction.
    let c = main_case("mnist_released__emlp2", CasePlacement::WhenNeeded);
    let a = f.registry.resolve(&c.selector().profile_id, 0).unwrap();
    let b = f.registry.resolve(&c.selector().profile_id, 9).unwrap();
    assert_eq!(
        geometry_identity(a.metadata()).unwrap(),
        geometry_identity(b.metadata()).unwrap()
    );
    assert_eq!(
        metadata_input_status(b.metadata()),
        InputStatus::MissingCaptured
    );
    let stats = f.budget.usage();
    assert_eq!(
        (
            stats.geometries - before.geometries,
            stats.initial_states - before.initial_states
        ),
        (144, 144)
    );
    eprintln!("P3 inventory usage: {stats:?}");
}

#[test]
fn p3_canonical_quick_full_identity_and_revalidation() {
    let f = fixtures();
    let p = f.iris.get_or_init(|| {
        eprintln!("P3 canonical bundled Iris materializations=1; plaintext_training=0");
        materialize(&f.registry.resolve("iris__mlp2020", 0).unwrap()).unwrap()
    });
    assert!(p.identity().canonical_prepared());
    assert_eq!(p.plan().updates.len(), 2400);
    let case = main_case("iris__mlp2020", CasePlacement::ClientRefresh);
    let request = request();
    let plan = plan_diagonal(p, case, parameters(CasePlacement::ClientRefresh), &request).unwrap();
    assert_eq!(plan.full_plan_sha256(), p.identity().plan_sha256());
    assert_eq!(
        (plan.full_update_count(), plan.consumed_updates()),
        (2400, 2)
    );
    assert_eq!(plan.full_rates().len(), 2400);
    assert_eq!(
        plan.reference_bits(),
        p.reference_kernel_rate_f64().to_bits()
    );
    assert_ne!(plan.full_plan_sha256(), plan.consumed_sha256());
    assert_eq!(case.selector().scope, CaseScope::Full);
    assert_eq!(plan.consumption(), Consumption::Quick);
    assert_eq!(plan.transitions().len(), 2);
    assert!(
        plan.transitions()
            .iter()
            .all(|r| r.next_full().is_none() || r.update() == 0)
    );
    assert_eq!(plan.transitions()[1].next_full(), None);
    assert!(!plan.receipt_bytes(0).unwrap().is_empty());
    assert!(plan.receipt_bytes(100).is_none());
    assert_eq!(
        ckks_train_bench::data::sha256_hex(plan.canonical_bytes()),
        plan.digest()
    );
    validate_plan_binding(
        p,
        case,
        &plan,
        parameters(CasePlacement::ClientRefresh),
        &request,
    )
    .unwrap();
    let mut bad = request.clone();
    bad.partitions = Some(vec![
        vec![MicroPart {
            start: 0,
            valid: 19
        }];
        2
    ]);
    assert!(
        validate_plan_binding(
            p,
            case,
            &plan,
            parameters(CasePlacement::ClientRefresh),
            &bad
        )
        .is_err()
    );
    assert!(
        validate_plan_binding(
            p,
            case,
            &plan,
            parameters(CasePlacement::WhenNeeded),
            &request
        )
        .is_err()
    );
    // Second production wrapper invocation, rejects before a new geometry/update.
    let wrong_case = main_case("iris__emlp1", CasePlacement::ClientRefresh);
    assert!(
        plan_diagonal(
            p,
            wrong_case,
            parameters(CasePlacement::ClientRefresh),
            &request
        )
        .is_err()
    );
    // Third wrapper invocation: bounded receipt failure, never a successful prefix plan.
    bad = request.clone();
    bad.limits.max_bytes = 16;
    let error = plan_diagonal(p, case, parameters(CasePlacement::ClientRefresh), &bad).unwrap_err();
    assert!(matches!(error.kind(), IssueKind::PlanningBudgetExceeded));
    assert_eq!(error.completed_updates(), 0);
    assert!(
        plan.unproved()
            .contains(&UnprovedGate::PreKeyFactoryIntegration)
    );
    let stats = f.budget.usage();
    assert!(stats.wrapper_calls <= 4 && stats.updates <= 8 && stats.handoffs == 0);
    eprintln!("P3 canonical usage: {stats:?}");
}

#[test]
fn p3_case_projection_and_scope() {
    let f = fixtures();
    let quick = f.cases.quick_family_selectors();
    assert_eq!(quick.len(), 268);
    assert!(f.cases.select(&quick[0]).is_err());
    let mut archival = quick[0].clone();
    archival.scope = CaseScope::ArchivalFull;
    assert!(f.cases.select(&archival).is_err());
    let case = main_case("iris__mlp2020", CasePlacement::ClientRefresh);
    let mut m = f
        .registry
        .resolve("iris__mlp2020", 0)
        .unwrap()
        .metadata()
        .clone();
    m.layers[0].input = 0;
    assert!(geometry_identity(&m).is_err());
    // Candidate axis failure keeps a row and never calls P2.
    m.seed = 9;
    let row = describe_candidate(
        &m,
        case,
        parameters(CasePlacement::ClientRefresh),
        &request(),
    );
    assert!(matches!(
        row.issue().unwrap().kind(),
        IssueKind::InvalidAxis
    ));
}

// This separately filtered amendment fixture avoids rematerializing canonical Iris.
#[test]
fn p3_bootstrap_configuration_binding() {
    let params = CkksParams::n16_boot();
    let proof = DftSupportProof::verify_current().unwrap();
    let mut caller = BootParams::default_n16();
    let descriptor =
        BootstrapMetadata::new_with_verified_support(&params, &caller, &proof).unwrap();
    let identity = descriptor.parameter_identity_bytes().to_vec();
    assert_eq!(descriptor.boot_params(), &caller);
    caller.k_intervals = 14;
    assert_eq!(descriptor.boot_params(), &BootParams::default_n16());
    assert_eq!(descriptor.parameter_identity_bytes(), identity);
    let incompatible =
        BootstrapMetadata::new_with_verified_support(&params, &caller, &proof).unwrap();
    assert_eq!(
        incompatible.generated_keys().q(),
        descriptor.generated_keys().q()
    );
    assert_eq!(incompatible.output(), descriptor.output());
    assert_eq!(incompatible.boot_params(), &caller);
    let registry =
        StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
    let cases = registry.registered_cases().unwrap();
    let case = cases
        .family_cases()
        .iter()
        .find(|c| {
            c.role() == CaseRole::Main
                && c.selector().profile_id == "iris__emlp1"
                && c.selector().seed == 0
                && c.selector().backend == CaseBackend::Cpu
                && c.selector().placement == CasePlacement::WhenNeeded
        })
        .unwrap();
    let metadata = registry.resolve("iris__emlp1", 0).unwrap();
    let mut request = DiagonalRequest::quick();
    request.work_budget = Arc::new(PlanningBudget::new(PlanningUsage {
        geometries: 1,
        initial_states: 1,
        ..PlanningUsage::default()
    }));
    let expected = BootParams::default_n16();
    let valid = describe_candidate(
        metadata.metadata(),
        case,
        PlanningParameters {
            params: &params,
            bootstrap: Some(&descriptor),
            boot_params: Some(&expected),
        },
        &request,
    );
    assert!(valid.issue().is_none(), "{:?}", valid.issue());
    assert_eq!(
        (
            request.work_budget.usage().geometries,
            request.work_budget.usage().initial_states
        ),
        (1, 1)
    );
    let row = describe_candidate(
        metadata.metadata(),
        case,
        PlanningParameters {
            params: &params,
            bootstrap: Some(&incompatible),
            boot_params: Some(&expected),
        },
        &request,
    );
    assert_eq!(row.issue().unwrap().kind(), IssueKind::IdentityMismatch);
    assert_eq!(
        request.work_budget.usage(),
        PlanningUsage {
            geometries: 1,
            initial_states: 1,
            ..PlanningUsage::default()
        }
    );
    eprintln!(
        "P3 filtered getter fixture: params=1 proof=1 metadata=2 geometry=1 initial=1; update/materialization/HE=0"
    );
}

#[test]
fn p3_fix1_late_finalization_retains_complete_prefix() {
    let f=fixtures();
    let p=f.iris.get_or_init(|| {
        eprintln!("P3 fix1 canonical Iris materializations=1");
        materialize(&f.registry.resolve("iris__mlp2020",0).unwrap()).unwrap()
    });
    let case=main_case("iris__mlp2020",CasePlacement::ClientRefresh);
    let mut request=request();
    let params=parameters(CasePlacement::ClientRefresh);
    let plan=plan_diagonal(p,case,params,&request).unwrap();
    let final_bytes=std::mem::size_of_val(plan.full_rates())+plan.full_plan_sha256().len()+64
        +std::mem::size_of_val(plan.keys().training_rotations());
    request.limits.max_bytes=plan.charges().combined()-final_bytes;
    let error=plan_diagonal(p,case,params,&request).unwrap_err();
    eprintln!("P3 fix1 late-finalization usage {:?}; cap={} completed={} prefix_records={}",f.budget.usage(),request.limits.max_bytes,error.completed_updates(),error.completed_prefix_transitions().len());
    assert_eq!(error.completed_updates(),2);
    assert_eq!(error.phase(),IssuePhase::Finalization);
    assert_eq!(error.stage(),None);assert_eq!(error.update(),None);
    assert_eq!(error.completed_prefix_transitions().len(),2);
    assert_eq!(error.kind(),IssueKind::PlanningBudgetExceeded);
    let required=request.limits.max_bytes+std::mem::size_of_val(plan.full_rates())+plan.full_plan_sha256().len()+64;
    assert_eq!(error.budget_requirement(),Some((required,request.limits.max_bytes)));
    for (i,r) in error.completed_prefix_transitions().iter().enumerate() {
        assert_eq!(r.update(),i);assert_eq!(r.stage(),0);
        assert_eq!(r.update_metadata().unwrap().post_state().weights(),plan.transitions()[i].update_metadata().unwrap().post_state().weights());
    }
    let start=plan.canonical_bytes().len()-plan.transitions().iter().enumerate().map(|(i,_)|plan.receipt_bytes(i).unwrap().len()).sum::<usize>();
    assert_eq!(&error.completed_prefix_bytes()[start..],&plan.canonical_bytes()[start..]);
}
