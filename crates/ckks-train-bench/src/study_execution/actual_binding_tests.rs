//! Metadata only: no keys, HE, native context, Linux stamp or readiness token.
use super::*;
use crate::study_admission::{Consumption, PlanningBudget, PlanningUsage};
use crate::study_profile::{CaseBackend, CasePacking, CasePlacement, CaseRole, StudyRegistry};
use ckks_train_ckks::EvalConfig;
use ckks_train_nn::dense::TransposeKeys;
use std::path::Path;
#[test]
fn d7_actual_route_six_sources_original_budget_and_owner_refusals() {
    let registry =
        StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
    let cases = registry.registered_cases().unwrap();
    for stages in 1..=3 {
        let id = format!("iris__emlp{stages}");
        let materialized =
            crate::study_inputs::materialize(&registry.resolve(&id, 0).unwrap()).unwrap();
        let other_materialized =
            crate::study_inputs::materialize(&registry.resolve(&id, 0).unwrap()).unwrap();
        for ni in [false, true] {
            let case = cases
                .family_cases()
                .iter()
                .find(|c| {
                    c.role() == CaseRole::Main
                        && c.selector().backend == CaseBackend::Cpu
                        && c.selector().profile_id == id
                        && c.selector().seed == 0
                        && c.selector().packing == CasePacking::BatchedDiagonal
                        && c.selector().placement
                            == if ni {
                                CasePlacement::WhenNeeded
                            } else {
                                CasePlacement::ClientRefresh
                            }
                })
                .unwrap();
            let parameters = RegisteredCpuParameters::for_case(case).unwrap().unwrap();
            let actual = parameters.actual();
            let request = DiagonalRequest {
                consumption: Consumption::Quick,
                physical_lanes: None,
                transpose: TransposeKeys::Dedicated,
                eval_config: EvalConfig::OPTIMIZED,
                partitions: None,
                limits: ReceiptLimits {
                    max_bytes: 64 << 20,
                    max_records: 1000000,
                },
                work_budget: Arc::new(PlanningBudget::new(PlanningUsage {
                    geometries: 2 * stages,
                    initial_states: 2 * stages,
                    updates: 2 * stages,
                    handoffs: 2 * (stages - 1),
                    wrapper_calls: 1,
                })),
            };
            let budget = request.work_budget.clone();
            let plan = Arc::new(
                crate::study_admission::plan_diagonal(&materialized, case, actual, &request)
                    .unwrap(),
            );
            let route = PreparedCpuRoute::prepare(CanonicalPreparationInput {
                materialized: &materialized,
                registered: case,
                plan: plan.clone(),
                request: &request,
                actual,
            })
            .unwrap();
            assert_eq!(route.is_block(), stages > 1);
            assert_eq!(route.bootstrap_engine(), u8::from(ni));
            assert!(std::ptr::eq(route.facts().parameters(), actual.params));
            assert!(std::ptr::eq(route.facts().plan(), plan.as_ref()));
            route.require_owners(&materialized, case, &request).unwrap();
            let copy = request.clone();
            assert!(route.require_owners(&materialized, case, &copy).is_err());
            assert!(
                route
                    .require_owners(&other_materialized, case, &request)
                    .is_err()
            );
            let other_case = cases
                .family_cases()
                .iter()
                .find(|other| other.selector().profile_id == id && other.selector().seed == 1)
                .unwrap();
            assert!(
                route
                    .require_owners(&materialized, other_case, &request)
                    .is_err()
            );
            let mut wrong_actual = actual;
            wrong_actual.bootstrap = if ni { None } else { actual.bootstrap };
            if ni {
                let wrong = PreparedCpuRoute::prepare(CanonicalPreparationInput {
                    materialized: &materialized,
                    registered: case,
                    plan: plan.clone(),
                    request: &request,
                    actual: wrong_actual,
                });
                assert!(wrong.is_err());
            }
            assert!(Arc::ptr_eq(&budget, &request.work_budget));
            assert_eq!(
                budget.usage(),
                PlanningUsage {
                    geometries: 2 * stages,
                    initial_states: 2 * stages,
                    updates: 2 * stages,
                    handoffs: 2 * (stages - 1),
                    wrapper_calls: 1,
                }
            );
            let again = PreparedCpuRoute::prepare(CanonicalPreparationInput {
                materialized: &materialized,
                registered: case,
                plan: plan.clone(),
                request: &request,
                actual,
            });
            assert!(again.is_err());
            assert_eq!(request.limits.max_bytes, 64 << 20);
            assert_eq!(request.limits.max_records, 1000000);
            assert_eq!(route.accounting_live_temporary(), 0);
        }
    }
    for case in cases.family_cases().iter().filter(|c| {
        c.role() != CaseRole::Main
            || c.selector().backend != CaseBackend::Cpu
            || c.selector().packing != CasePacking::BatchedDiagonal
    }) {
        assert!(RegisteredCpuParameters::for_case(case).unwrap().is_none());
    }
}
