//! Fixed, disconnected CPU bodies; no live gate, start capability or runtime claim.
use super::*;
use crate::{
    study_admission,
    study_blocks::{CpuBlockEvaluation, CpuBlockSession, EncryptedBlockInitial},
    study_client::{CpuBootstrapDescription, StudyClient, blocks::BlockEncryptionBinding},
    study_diagonal::{CpuDiagonalSession, CpuEvaluationContext, EncryptedInitialState},
};
use ckks_train_ckks::{CkksParams, OpCounter};
use ckks_train_core::sampler::Rng;
use std::{marker::PhantomData, rc::Rc};

// This tail owns no permission. Only an already-retained bundle can supply it.
// Move these SAME fields into the future genuine frame; never reset accounting.
// Rust drops fields in declaration order: descriptors before accounting.
struct RetainedCpuInputs<'i, 'p> {
    descriptors: CanonicalDescriptors,
    materialized: &'i MaterializedProfile,
    registered: &'i RegisteredCase,
    request: &'i DiagonalRequest,
    plan: Arc<DiagonalPlan>,
    actual: PlanningParameters<'p>,
    not_send_sync: PhantomData<Rc<()>>,
    accounting: PreparationAccounting,
}
fn retain_single_bundle<'i, 'p>(
    bundle: PreparedCanonicalBundle<'i, CheckedPreparation<'p>>,
) -> (CheckedPreparation<'p>, RetainedCpuInputs<'i, 'p>) {
    let PreparedCanonicalBundle {
        descriptors,
        preparation,
    } = bundle;
    let RetainedPreparation {
        checked,
        materialized,
        registered,
        request,
        accounting,
    } = preparation;
    let actual = checked.actual_parameters();
    let plan = match &checked.plan {
        crate::study_diagonal::PlanOwner::Canonical(p) => p.clone(),
        #[cfg(test)]
        crate::study_diagonal::PlanOwner::Synthetic(_) => unreachable!("canonical bundle"),
    };
    (
        checked,
        RetainedCpuInputs {
            descriptors,
            materialized,
            registered,
            request,
            plan,
            actual,
            not_send_sync: PhantomData,
            accounting,
        },
    )
}
fn retain_block_bundle<'i, 'p>(
    bundle: PreparedCanonicalBundle<'i, CheckedBlockPreparation<'p>>,
) -> (CheckedBlockPreparation<'p>, RetainedCpuInputs<'i, 'p>) {
    let PreparedCanonicalBundle {
        descriptors,
        preparation,
    } = bundle;
    let RetainedPreparation {
        checked,
        materialized,
        registered,
        request,
        accounting,
    } = preparation;
    let actual = checked.actual_parameters();
    let plan = match &checked.plan {
        crate::study_blocks::BlockPlanOwner::Canonical(p) => p.clone(),
        #[cfg(test)]
        crate::study_blocks::BlockPlanOwner::Synthetic(_) => unreachable!("canonical bundle"),
    };
    (
        checked,
        RetainedCpuInputs {
            descriptors,
            materialized,
            registered,
            request,
            plan,
            actual,
            not_send_sync: PhantomData,
            accounting,
        },
    )
}
pub(crate) fn reserve_setup_copies<'a>(
    accounting: &'a mut PreparationAccounting,
    materialized: &MaterializedProfile,
    block: bool,
    binding_bytes: usize,
) -> AccountingResult<Reservation<'a>> {
    accounting.reserve(sum([
        study_client::canonical_array_payload(materialized, block)?,
        binding_bytes,
    ])?)
}

// Keep fixed owners inline: boxing would add an unrelated application allocation.
#[allow(clippy::large_enum_variant)]
enum CpuOwnedSetup {
    Single {
        client: StudyClient,
        evaluation: CpuEvaluationContext,
        initial: EncryptedInitialState,
    },
    Block {
        client: StudyClient,
        evaluation: CpuEvaluationContext,
        initial: EncryptedBlockInitial,
        binding: BlockEncryptionBinding,
    },
}
// NON-authorizing intermediate owner. Its only constructors consume the retained
// bundle, never caller hashes. An unused value drops crypto BEFORE its input tail.
pub(crate) struct CpuContinuation<'i, 'p> {
    setup: CpuOwnedSetup,
    retained: RetainedCpuInputs<'i, 'p>,
}
impl<'i, 'p> CpuContinuation<'i, 'p> {
    pub(crate) fn single(
        bundle: PreparedCanonicalBundle<'i, CheckedPreparation<'p>>,
        parameters: Arc<CkksParams>,
        boot: Option<CpuBootstrapDescription>,
        key_rng: Rng,
        encryption_rng: Rng,
    ) -> CpuPreparationResult<Self> {
        let (checked, mut retained) = retain_single_bundle(bundle);
        let (client, evaluation, initial) = study_client::make_retained_owned(
            checked,
            retained.materialized,
            parameters,
            boot,
            key_rng,
            encryption_rng,
            &mut retained.accounting,
        )?;
        Ok(Self {
            setup: CpuOwnedSetup::Single {
                client,
                evaluation,
                initial,
            },
            retained,
        })
    }
    pub(crate) fn block(
        bundle: PreparedCanonicalBundle<'i, CheckedBlockPreparation<'p>>,
        parameters: Arc<CkksParams>,
        boot: Option<CpuBootstrapDescription>,
        key_rng: Rng,
        encryption_rng: Rng,
    ) -> CpuPreparationResult<Self> {
        let (checked, mut retained) = retain_block_bundle(bundle);
        let (client, evaluation, initial, binding) =
            study_client::blocks::make_retained_blocks_owned(
                checked,
                retained.materialized,
                parameters,
                boot,
                key_rng,
                encryption_rng,
                &mut retained.accounting,
            )?;
        Ok(Self {
            setup: CpuOwnedSetup::Block {
                client,
                evaluation,
                initial,
                binding,
            },
            retained,
        })
    }
    // Deliberately not called by run_owned. Genuine live gate/resource rechecks and
    // post-durable-start accounted binding validation still need their real frame.
    // Failure retains progress inline; no error-path heap allocation is needed.
    #[allow(clippy::result_large_err)]
    pub(crate) fn execute_cpu(self, refresh_rng: Rng) -> Result<CpuRunProgress, CpuRunFailure> {
        let Self { setup, retained } = self;
        // Declare retained FIRST so every setup/session local dies before it on
        // all returns, including a failed session constructor or encrypted update.
        let mut retained = retained;
        let setup = setup;
        match setup {
            CpuOwnedSetup::Single {
                client,
                evaluation,
                initial,
            } => {
                let origin = evaluation.counter();
                let mut progress = CpuRunProgress::default();
                let result = {
                    let refresh = retained
                        .actual
                        .bootstrap
                        .is_none()
                        .then(|| client.refresh_service(refresh_rng));
                    let mut session = CpuDiagonalSession::new(
                        &evaluation,
                        retained.plan.clone(),
                        initial,
                        refresh.as_ref(),
                    )
                    .map_err(|error| {
                        progress.measured = &evaluation.counter() - &origin;
                        CpuRunFailure {
                            error: error.into(),
                            progress: progress.clone(),
                        }
                    })?;
                    loop {
                        let index = progress.completed_updates;
                        if index == retained.plan.consumed_updates() {
                            break Ok(progress);
                        }
                        let step = retained
                            .encrypt_single_update(&client, &evaluation, index)
                            .and_then(|update| {
                                session.step(&update).map_err(CpuPreparationError::He)
                            });
                        match step {
                            Ok(done) => {
                                progress.completed_updates = done.completed_updates;
                                progress.processed_samples = done.processed_samples;
                                progress.measured = &evaluation.counter() - &origin;
                            }
                            Err(error) => {
                                progress.measured = &evaluation.counter() - &origin;
                                progress.poisoned = session.is_poisoned();
                                progress.failed_prefix =
                                    session.failed_update().map(|failed| CpuFailurePrefix {
                                        update: failed.update,
                                        phase: failed.phase,
                                        measured: failed.measured.clone(),
                                    });
                                break Err(CpuRunFailure { error, progress });
                            }
                        }
                    }
                    // Session drops here, before its same-client refresher.
                };
                // client/evaluation drop before retained on leaving this arm.
                result
            }
            CpuOwnedSetup::Block {
                client,
                evaluation,
                initial,
                binding,
            } => {
                let origin = evaluation.counter();
                let mut progress = CpuRunProgress::default();
                let result = {
                    let refresh = retained
                        .actual
                        .bootstrap
                        .is_none()
                        .then(|| client.refresh_service(refresh_rng));
                    let facade = CpuBlockEvaluation::new(&evaluation);
                    let mut session = CpuBlockSession::new(
                        &facade,
                        retained.plan.clone(),
                        initial,
                        refresh.as_ref(),
                    )
                    .map_err(|error| {
                        progress.measured = &evaluation.counter() - &origin;
                        CpuRunFailure {
                            error: error.into(),
                            progress: progress.clone(),
                        }
                    })?;
                    loop {
                        let index = progress.completed_updates;
                        if index == retained.plan.consumed_updates() {
                            break Ok(progress);
                        }
                        let step = retained
                            .encrypt_block_update(&client, &evaluation, &binding, index)
                            .and_then(|update| {
                                session.step(&update).map_err(CpuPreparationError::He)
                            });
                        match step {
                            Ok(done) => {
                                progress.completed_updates = done.completed_updates;
                                progress.processed_samples = done.processed_samples;
                                progress.measured = &evaluation.counter() - &origin;
                            }
                            Err(error) => {
                                progress.measured = &evaluation.counter() - &origin;
                                progress.poisoned = session.is_poisoned();
                                progress.failed_prefix =
                                    session.failed_update().map(|failed| CpuFailurePrefix {
                                        update: failed.update,
                                        phase: failed.phase,
                                        measured: failed.measured.clone(),
                                    });
                                break Err(CpuRunFailure { error, progress });
                            }
                        }
                    }
                    // session, facade, refresher drop in that order.
                };
                result
            }
        }
    }
}
impl RetainedCpuInputs<'_, '_> {
    fn binding_workspace(&self) -> CpuPreparationResult<usize> {
        study_admission::binding_workspace_payload(
            self.materialized,
            self.registered,
            self.actual,
            self.request,
        )
        .map_err(|e| CpuPreparationError::Admission(Box::new(e)))
    }
    fn encrypt_single_update(
        &mut self,
        client: &StudyClient,
        evaluation: &CpuEvaluationContext,
        index: usize,
    ) -> CpuPreparationResult<crate::study_diagonal::EncryptedLogicalUpdate> {
        let bytes = self.binding_workspace()?;
        let _reservation = self.accounting.reserve(bytes)?;
        client.encrypt_update(
            self.materialized,
            self.registered,
            self.plan.clone(),
            self.request,
            index,
            evaluation,
        )
    }
    fn encrypt_block_update(
        &mut self,
        client: &StudyClient,
        evaluation: &CpuEvaluationContext,
        binding: &BlockEncryptionBinding,
        index: usize,
    ) -> CpuPreparationResult<crate::study_blocks::EncryptedBlockUpdate> {
        let bytes = self.binding_workspace()?;
        let _reservation = self.accounting.reserve(bytes)?;
        client.encrypt_block_update(
            binding,
            self.materialized,
            self.registered,
            self.plan.clone(),
            self.request,
            index,
            evaluation,
        )
    }
}
// Operational partial progress only. No ciphertext snapshots, scientific JSON,
// endpoint metrics, gate bit or readiness identity are produced by these bodies.
#[derive(Clone, Debug, Default)]
pub(crate) struct CpuRunProgress {
    completed_updates: usize,
    processed_samples: usize,
    poisoned: bool,
    measured: OpCounter,
    failed_prefix: Option<CpuFailurePrefix>,
}
#[derive(Clone, Debug)]
struct CpuFailurePrefix {
    update: usize,
    phase: &'static str,
    measured: OpCounter,
}
#[derive(Debug)]
pub(crate) struct CpuRunFailure {
    error: CpuPreparationError,
    progress: CpuRunProgress,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study_admission::{PlanningBudget, PlanningUsage};
    use crate::study_profile::{CaseBackend, CasePlacement, CaseRole, StudyRegistry};
    use std::path::Path;

    #[test]
    fn d7_fixed_retained_association_and_copy_refusal() {
        let registry =
            StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let cases = registry.registered_cases().unwrap();
        let parameters = Arc::new(CkksParams::n16_train());
        let equal_parameters = Arc::new(CkksParams::n16_train());
        for stages in [1, 2] {
            let id = format!("iris__emlp{stages}");
            let materialized =
                crate::study_inputs::materialize(&registry.resolve(&id, 0).unwrap()).unwrap();
            let case = cases
                .family_cases()
                .iter()
                .find(|c| {
                    c.role() == CaseRole::Main
                        && c.selector().profile_id == id
                        && c.selector().seed == 0
                        && c.selector().backend == CaseBackend::Cpu
                        && c.selector().placement == CasePlacement::ClientRefresh
                })
                .unwrap();
            let mut request = DiagonalRequest::quick();
            request.work_budget = Arc::new(PlanningBudget::new(PlanningUsage {
                geometries: 2 * stages,
                initial_states: 2 * stages,
                updates: 2 * stages,
                handoffs: 2 * (stages - 1),
                wrapper_calls: 1,
            }));
            let actual = PlanningParameters {
                params: &parameters,
                bootstrap: None,
                boot_params: None,
            };
            let plan = Arc::new(
                study_admission::plan_diagonal(&materialized, case, actual, &request).unwrap(),
            );
            let input = CanonicalPreparationInput {
                materialized: &materialized,
                registered: case,
                plan: plan.clone(),
                request: &request,
                actual,
            };
            let (retained, binding_bytes) = if stages == 1 {
                let bundle =
                    collect_single(prepare_canonical(input, plan.partitions()).unwrap()).unwrap();
                let descriptor_pointer = bundle.descriptors.layout.as_ptr();
                let before = bundle.preparation.accounting.retained;
                let (checked, mut retained) = retain_single_bundle(bundle);
                assert_eq!(retained.descriptors.layout.as_ptr(), descriptor_pointer);
                assert_eq!(retained.accounting.retained, before);
                assert!(std::ptr::eq(
                    checked.actual_parameters().params,
                    retained.actual.params
                ));
                let refusal = study_client::make_retained_owned(
                    checked,
                    retained.materialized,
                    equal_parameters.clone(),
                    None,
                    Rng::from_seed([41; 32]),
                    Rng::from_seed([42; 32]),
                    &mut retained.accounting,
                );
                assert!(matches!(refusal, Err(CpuPreparationError::He(_))));
                assert_eq!(retained.accounting.retained, before);
                (retained, 0)
            } else {
                let bundle =
                    collect_blocks(prepare_blocks_canonical(input, plan.partitions()).unwrap())
                        .unwrap();
                let descriptor_pointer = bundle.descriptors.layout.as_ptr();
                let before = bundle.preparation.accounting.retained;
                let (checked, mut retained) = retain_block_bundle(bundle);
                assert_eq!(retained.descriptors.layout.as_ptr(), descriptor_pointer);
                assert_eq!(retained.accounting.retained, before);
                let binding_bytes = checked.binding_copy_payload().unwrap();
                let refusal = study_client::blocks::make_retained_blocks_owned(
                    checked,
                    retained.materialized,
                    equal_parameters.clone(),
                    None,
                    Rng::from_seed([43; 32]),
                    Rng::from_seed([44; 32]),
                    &mut retained.accounting,
                );
                assert!(matches!(refusal, Err(CpuPreparationError::He(_))));
                assert_eq!(retained.accounting.retained, before);
                (retained, binding_bytes)
            };
            assert!(Arc::ptr_eq(&retained.plan, &plan));
            assert!(std::ptr::eq(retained.materialized, &materialized));
            assert!(std::ptr::eq(retained.registered, case));
            assert!(std::ptr::eq(retained.request, &request));
            assert!(std::ptr::eq(retained.actual.params, parameters.as_ref()));
            let copies = sum([
                study_client::canonical_array_payload(&materialized, stages > 1).unwrap(),
                binding_bytes,
            ])
            .unwrap();
            // This is a fresh pure refusal probe for the SAME production reservation
            // helper, not a mutated request/plan and never a crypto constructor.
            let mut too_small = PreparationAccounting::from_seed(
                0,
                0,
                ReceiptLimits {
                    max_bytes: copies - 1,
                    max_records: 0,
                },
            )
            .unwrap();
            assert!(
                reserve_setup_copies(&mut too_small, &materialized, stages > 1, binding_bytes)
                    .is_err()
            );
            assert_eq!((too_small.retained, too_small.live_temporary), (0, 0));
            let mut exact = PreparationAccounting::from_seed(
                0,
                0,
                ReceiptLimits {
                    max_bytes: copies,
                    max_records: 0,
                },
            )
            .unwrap();
            {
                let mut reservation =
                    reserve_setup_copies(&mut exact, &materialized, stages > 1, binding_bytes)
                        .unwrap();
                reservation.retain(binding_bytes).unwrap();
                // Remaining canonical arrays are temporary and release on scope exit.
            }
            assert_eq!((exact.retained, exact.live_temporary), (binding_bytes, 0));
            assert_eq!(retained.accounting.live_temporary, 0);
            assert_eq!(
                request.work_budget.usage(),
                PlanningUsage {
                    geometries: 2 * stages,
                    initial_states: 2 * stages,
                    updates: 2 * stages,
                    handoffs: 2 * (stages - 1),
                    wrapper_calls: 1,
                }
            );
            drop(retained);
        }
    }
}
