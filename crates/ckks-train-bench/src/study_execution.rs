//! Retained canonical CPU preparation. No start or execution capability.
#![allow(dead_code)] // Intentionally disconnected from production until fixed phase3 release.
use crate::{
    study_admission::{
        DiagonalPlan, DiagonalRequest, MicroPart, PlanningParameters, ReceiptLimits,
    },
    study_client::{self, CheckedPreparation, CpuPreparationError, CpuPreparationResult},
    study_inputs::MaterializedProfile,
    study_profile::RegisteredCase,
};
use std::sync::Arc;
use study_client::blocks::CheckedBlockPreparation;

#[derive(Debug, PartialEq, Eq)]
/// A failure of the additive retained preparation requested-payload boundary.
pub enum AccountingError {
    /// Checked payload arithmetic or reservation transfer overflowed.
    Overflow,
    /// The combined plan, retained and temporary payload exceeds the original cap.
    Bytes {
        /// Actual requested payload bytes.
        required: usize,
        /// Original configured byte cap.
        configured: usize,
    },
    /// The original plan's indexed-record count exceeds its original cap.
    Records {
        /// Actual existing indexed records.
        required: usize,
        /// Original configured record cap.
        configured: usize,
    },
    /// A fallible output allocation could not be reserved.
    Allocation,
}
impl std::fmt::Display for AccountingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "preparation requested-payload accounting: {self:?}")
    }
}
impl std::error::Error for AccountingError {}
impl From<AccountingError> for CpuPreparationError {
    fn from(e: AccountingError) -> Self {
        Self::Accounting(e)
    }
}
pub(crate) type AccountingResult<T> = std::result::Result<T, AccountingError>;
pub(crate) fn payload<T>(n: usize) -> AccountingResult<usize> {
    n.checked_mul(std::mem::size_of::<T>())
        .ok_or(AccountingError::Overflow)
}
pub(crate) fn sum(values: impl IntoIterator<Item = usize>) -> AccountingResult<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b).ok_or(AccountingError::Overflow)
    })
}
pub(crate) fn allocated<T>(n: usize) -> AccountingResult<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| AccountingError::Allocation)?;
    Ok(v)
}
pub(crate) struct PreparationAccounting {
    limits: ReceiptLimits,
    plan_bytes: usize,
    retained: usize,
    live_temporary: usize,
    peak_temporary: usize,
}
impl PreparationAccounting {
    fn from_seed(
        plan_bytes: usize,
        records: usize,
        limits: ReceiptLimits,
    ) -> AccountingResult<Self> {
        if plan_bytes > limits.max_bytes {
            return Err(AccountingError::Bytes {
                required: plan_bytes,
                configured: limits.max_bytes,
            });
        }
        if records > limits.max_records {
            return Err(AccountingError::Records {
                required: records,
                configured: limits.max_records,
            });
        }
        Ok(Self {
            limits,
            plan_bytes,
            retained: 0,
            live_temporary: 0,
            peak_temporary: 0,
        })
    }
    fn seed(plan: &DiagonalPlan, limits: ReceiptLimits) -> AccountingResult<Self> {
        Self::from_seed(plan.charges().combined(), plan.transitions().len(), limits)
    }
    pub(crate) fn reserve(&mut self, bytes: usize) -> AccountingResult<Reservation<'_>> {
        let next = sum([self.live_temporary, bytes])?;
        let required = sum([self.plan_bytes, self.retained, next])?;
        if required > self.limits.max_bytes {
            return Err(AccountingError::Bytes {
                required,
                configured: self.limits.max_bytes,
            });
        }
        self.live_temporary = next;
        self.peak_temporary = self.peak_temporary.max(next);
        Ok(Reservation {
            accounting: self,
            remaining: bytes,
        })
    }
}
pub(crate) struct Reservation<'a> {
    accounting: &'a mut PreparationAccounting,
    remaining: usize,
}
impl Reservation<'_> {
    pub(crate) fn retain(&mut self, bytes: usize) -> AccountingResult<()> {
        let remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or(AccountingError::Overflow)?;
        let retained = sum([self.accounting.retained, bytes])?;
        self.remaining = remaining;
        self.accounting.live_temporary -= bytes;
        self.accounting.retained = retained;
        Ok(())
    }
    pub(crate) fn release(&mut self, bytes: usize) -> AccountingResult<()> {
        let remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or(AccountingError::Overflow)?;
        self.remaining = remaining;
        self.accounting.live_temporary -= bytes;
        Ok(())
    }
    pub(crate) fn retain_all(&mut self) -> AccountingResult<()> {
        self.retain(self.remaining)
    }
}
impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.accounting.live_temporary -= self.remaining;
    }
}
pub(crate) struct CanonicalPreparationInput<'i, 'p> {
    pub(crate) materialized: &'i MaterializedProfile,
    pub(crate) registered: &'i RegisteredCase,
    pub(crate) plan: Arc<DiagonalPlan>,
    pub(crate) request: &'i DiagonalRequest,
    pub(crate) actual: PlanningParameters<'p>,
}
// Drop order is part of the contract: checked geometry before its accounting owner.
pub(crate) struct RetainedPreparation<'i, T> {
    checked: T,
    materialized: &'i MaterializedProfile,
    registered: &'i RegisteredCase,
    request: &'i DiagonalRequest,
    accounting: PreparationAccounting,
}
pub(crate) struct CanonicalDescriptors {
    pub(crate) interactive_parameters: Option<Vec<u8>>,
    pub(crate) layout: Vec<u8>,
    pub(crate) keys: Vec<u8>,
    pub(crate) security: Vec<u8>,
    pub(crate) parameter_digest: String,
    pub(crate) bootstrap_digest: Option<String>,
    pub(crate) layout_digest: String,
    pub(crate) keys_digest: String,
    pub(crate) security_digest: String,
}
// Descriptors drop before preparation; neither checked state nor accounting is extracted.
pub(crate) struct PreparedCanonicalBundle<'i, T> {
    descriptors: CanonicalDescriptors,
    preparation: RetainedPreparation<'i, T>,
}
pub(crate) fn prepare_canonical<'i, 'p>(
    input: CanonicalPreparationInput<'i, 'p>,
    parts: &[Vec<MicroPart>],
) -> CpuPreparationResult<RetainedPreparation<'i, CheckedPreparation<'p>>> {
    let mut accounting = PreparationAccounting::seed(&input.plan, input.request.limits)?;
    let checked = study_client::check_preparation(
        study_client::PreparationInput::Canonical {
            materialized: input.materialized,
            registered: input.registered,
            plan: input.plan,
            request: input.request,
        },
        input.actual,
        parts,
        Some(&mut accounting),
    )?;
    Ok(RetainedPreparation {
        checked,
        materialized: input.materialized,
        registered: input.registered,
        request: input.request,
        accounting,
    })
}
pub(crate) fn prepare_blocks_canonical<'i, 'p>(
    input: CanonicalPreparationInput<'i, 'p>,
    parts: &[Vec<MicroPart>],
) -> CpuPreparationResult<RetainedPreparation<'i, CheckedBlockPreparation<'p>>> {
    let mut accounting = PreparationAccounting::seed(&input.plan, input.request.limits)?;
    let checked = study_client::blocks::check_block_preparation(
        study_client::blocks::BlockPreparationInput::Canonical {
            materialized: input.materialized,
            registered: input.registered,
            plan: input.plan,
            request: input.request,
        },
        input.actual,
        parts,
        Some(&mut accounting),
    )?;
    Ok(RetainedPreparation {
        checked,
        materialized: input.materialized,
        registered: input.registered,
        request: input.request,
        accounting,
    })
}
impl RetainedPreparation<'_, CheckedPreparation<'_>> {
    pub(crate) fn descriptor_parts(
        &mut self,
    ) -> (
        study_client::PreparationDescriptorFacts<'_>,
        &mut PreparationAccounting,
    ) {
        (
            self.checked
                .descriptor_facts()
                .expect("canonical retained owner"),
            &mut self.accounting,
        )
    }
}
impl RetainedPreparation<'_, CheckedBlockPreparation<'_>> {
    pub(crate) fn descriptor_parts(
        &mut self,
    ) -> (
        study_client::PreparationDescriptorFacts<'_>,
        &mut PreparationAccounting,
    ) {
        (
            self.checked
                .descriptor_facts()
                .expect("canonical retained owner"),
            &mut self.accounting,
        )
    }
}
impl<'i, T> PreparedCanonicalBundle<'i, T> {
    pub(crate) fn new(
        descriptors: CanonicalDescriptors,
        preparation: RetainedPreparation<'i, T>,
    ) -> Self {
        Self {
            descriptors,
            preparation,
        }
    }
}
pub(crate) fn collect_single<'i, 'p>(
    preparation: RetainedPreparation<'i, CheckedPreparation<'p>>,
) -> CpuPreparationResult<PreparedCanonicalBundle<'i, CheckedPreparation<'p>>> {
    crate::study_reference::collect_single_preparation_descriptors(preparation)
}
pub(crate) fn collect_blocks<'i, 'p>(
    preparation: RetainedPreparation<'i, CheckedBlockPreparation<'p>>,
) -> CpuPreparationResult<PreparedCanonicalBundle<'i, CheckedBlockPreparation<'p>>> {
    crate::study_reference::collect_block_preparation_descriptors(preparation)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::study_admission::{Consumption, PlanningBudget, PlanningUsage, ReceiptLimits};
    use crate::study_profile::{CaseBackend, CasePlacement, CaseRole, StudyRegistry};
    use ckks_train_ckks::{
        CkksParams,
        bootstrap::{
            BootParams,
            metadata::{BootstrapMetadata, DftSupportProof},
        },
    };
    use std::{path::Path, sync::Arc};

    #[test]
    fn d7_accounting_pure_boundaries() {
        let limits = ReceiptLimits {
            max_bytes: 100,
            max_records: 2,
        };
        assert!(PreparationAccounting::from_seed(101, 0, limits).is_err());
        assert!(PreparationAccounting::from_seed(0, 3, limits).is_err());
        let mut a = PreparationAccounting::from_seed(20, 2, limits).unwrap();
        {
            let mut r = a.reserve(80).unwrap();
            r.retain(30).unwrap();
        }
        assert_eq!(
            (a.retained, a.live_temporary, a.peak_temporary),
            (30, 0, 80)
        );
        assert!(a.reserve(51).is_err());
        assert_eq!(a.retained, 30);
        {
            let r = a.reserve(50).unwrap();
            drop(r);
        }
        assert_eq!(a.live_temporary, 0);
        {
            let mut r = a.reserve(50).unwrap();
            assert!(r.retain(51).is_err());
            r.release(10).unwrap();
            r.retain(40).unwrap();
        }
        assert_eq!((a.retained, a.live_temporary), (70, 0));
        let mut failure = PreparationAccounting::from_seed(0, 0, limits).unwrap();
        {
            let _r = failure.reserve(100).unwrap();
            assert!(matches!(
                allocated::<u8>(usize::MAX),
                Err(AccountingError::Allocation)
            ));
        }
        assert_eq!((failure.retained, failure.live_temporary), (0, 0));
        let mut a = PreparationAccounting::from_seed(
            usize::MAX,
            0,
            ReceiptLimits {
                max_bytes: usize::MAX,
                max_records: 0,
            },
        )
        .unwrap();
        assert!(a.reserve(1).is_err());
    }

    // An input borrow may end independently of the longer parameter borrow.
    fn independent_lifetimes<'i, 'p>(
        input: CanonicalPreparationInput<'i, 'p>,
        parts: &[Vec<crate::study_admission::MicroPart>],
    ) -> crate::study_client::CpuPreparationResult<
        RetainedPreparation<'i, crate::study_client::CheckedPreparation<'p>>,
    > {
        prepare_canonical(input, parts)
    }

    #[test]
    fn d7_canonical_six_retained_sources() {
        let registry =
            StudyRegistry::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")).unwrap();
        let cases = registry.registered_cases().unwrap();
        let interactive = Arc::new(CkksParams::n16_train());
        let noninteractive = Arc::new(CkksParams::n16_boot());
        let bp = BootParams::default_n16();
        let proof = DftSupportProof::verify_current().unwrap();
        let boot =
            BootstrapMetadata::new_with_verified_support(&noninteractive, &bp, &proof).unwrap();
        for stages in 1..=3 {
            let id = format!("iris__emlp{stages}");
            let materialized =
                crate::study_inputs::materialize(&registry.resolve(&id, 0).unwrap()).unwrap();
            for ni in [false, true] {
                let case = cases
                    .family_cases()
                    .iter()
                    .find(|c| {
                        c.role() == CaseRole::Main
                            && c.selector().profile_id == id
                            && c.selector().seed == 0
                            && c.selector().backend == CaseBackend::Cpu
                            && c.selector().placement
                                == if ni {
                                    CasePlacement::WhenNeeded
                                } else {
                                    CasePlacement::ClientRefresh
                                }
                    })
                    .unwrap();
                let actual = crate::study_admission::PlanningParameters {
                    params: if ni {
                        noninteractive.as_ref()
                    } else {
                        interactive.as_ref()
                    },
                    bootstrap: ni.then_some(&boot),
                    boot_params: ni.then_some(&bp),
                };
                let mut request = crate::study_admission::DiagonalRequest::quick();
                request.work_budget = Arc::new(PlanningBudget::new(PlanningUsage {
                    geometries: 2 * stages,
                    initial_states: 2 * stages,
                    updates: 2 * stages,
                    handoffs: 2 * (stages - 1),
                    wrapper_calls: 1,
                }));
                let plan = Arc::new(
                    crate::study_admission::plan_diagonal(&materialized, case, actual, &request)
                        .unwrap(),
                );
                let input = CanonicalPreparationInput {
                    materialized: &materialized,
                    registered: case,
                    plan: plan.clone(),
                    request: &request,
                    actual,
                };
                if stages == 1 {
                    let p = independent_lifetimes(input, plan.partitions()).unwrap();
                    assert!(std::ptr::eq(
                        plan.as_ref(),
                        p.checked.descriptor_facts().unwrap().plan()
                    ));
                    let b = collect_single(p).unwrap();
                    assert_eq!(
                        b.preparation
                            .checked
                            .descriptor_facts()
                            .unwrap()
                            .plan()
                            .consumption(),
                        Consumption::Quick
                    );
                    assert!(std::ptr::eq(b.preparation.materialized, &materialized));
                    assert!(std::ptr::eq(b.preparation.registered, case));
                    assert!(std::ptr::eq(b.preparation.request, &request));
                    assert!(std::ptr::eq(
                        b.preparation
                            .checked
                            .descriptor_facts()
                            .unwrap()
                            .parameters(),
                        actual.params
                    ));
                    assert_eq!(b.preparation.accounting.live_temporary, 0);
                    assert_bundle(
                        &b.descriptors,
                        b.preparation.checked.descriptor_facts().unwrap(),
                    );
                } else {
                    let p = prepare_blocks_canonical(input, plan.partitions()).unwrap();
                    let b = collect_blocks(p).unwrap();
                    assert!(std::ptr::eq(
                        plan.as_ref(),
                        b.preparation.checked.descriptor_facts().unwrap().plan()
                    ));
                    assert!(std::ptr::eq(b.preparation.materialized, &materialized));
                    assert!(std::ptr::eq(b.preparation.registered, case));
                    assert!(std::ptr::eq(b.preparation.request, &request));
                    assert!(std::ptr::eq(
                        b.preparation
                            .checked
                            .descriptor_facts()
                            .unwrap()
                            .parameters(),
                        actual.params
                    ));
                    assert_eq!(b.preparation.accounting.live_temporary, 0);
                    assert_bundle(
                        &b.descriptors,
                        b.preparation.checked.descriptor_facts().unwrap(),
                    );
                }
                if stages == 1 && !ni {
                    // Two extra successful binding output traversals; no extra P2 calls.
                    let make_input = || CanonicalPreparationInput {
                        materialized: &materialized,
                        registered: case,
                        plan: plan.clone(),
                        request: &request,
                        actual,
                    };
                    let bad = prepare_canonical(make_input(), &[]);
                    assert!(matches!(bad, Err(CpuPreparationError::He(_))));
                    // The original shared budget is now exhausted. Preserve candidate diagnostics.
                    let bad = prepare_canonical(make_input(), plan.partitions());
                    match bad {
                        Err(CpuPreparationError::Candidate(c)) => {
                            assert_eq!(
                                c.issue().unwrap().kind(),
                                crate::study_admission::IssueKind::PlanningBudgetExceeded
                            );
                            assert!(c.geometries().is_empty());
                        }
                        _ => panic!("expected original candidate diagnostic"),
                    }
                    // Legacy callback remains independently scoped and cannot be reached on refusal.
                    let calls = std::cell::Cell::new(0);
                    let bad = study_client::prepare_then_make(
                        study_client::PreparationInput::Canonical {
                            materialized: &materialized,
                            registered: case,
                            plan: plan.clone(),
                            request: &request,
                        },
                        actual,
                        &[],
                        |_| {
                            calls.set(calls.get() + 1);
                            Ok(())
                        },
                    );
                    assert!(bad.is_err());
                    assert_eq!(calls.get(), 0);
                    // Allocation-free count traversal and actual output workspace must agree.
                    let sizes =
                        crate::study_reference::identity_payload_sizes(&materialized).unwrap();
                    let identities =
                        crate::study_reference::derive_identities(&materialized).unwrap();
                    assert_eq!(
                        sizes,
                        (
                            identities.0.canonical_bytes().len(),
                            identities.1.canonical_bytes().len()
                        )
                    );
                }
                assert_eq!(
                    request.work_budget.usage(),
                    PlanningUsage {
                        geometries: 2 * stages,
                        initial_states: 2 * stages,
                        updates: 2 * stages,
                        handoffs: 2 * (stages - 1),
                        wrapper_calls: 1
                    }
                );
            }
        }
    }
    fn assert_bundle(
        d: &CanonicalDescriptors,
        f: crate::study_client::PreparationDescriptorFacts<'_>,
    ) {
        let s = f.descriptor_sizes().unwrap();
        assert_eq!(s.layout, d.layout.len());
        assert_eq!(s.keys, d.keys.len());
        assert_eq!(s.security, d.security.len());
        assert_eq!(d.layout, f.physical_layout_bytes().unwrap());
        assert_eq!(d.keys, f.cpu_key_policy_bytes().unwrap());
        assert_eq!(d.security, f.security_declaration_bytes().unwrap());
        assert_eq!(
            s.interactive,
            d.interactive_parameters.as_ref().map(Vec::len)
        );
        assert_eq!(
            d.parameter_digest,
            crate::data::sha256_hex(
                d.interactive_parameters
                    .as_deref()
                    .unwrap_or_else(|| f.bootstrap().unwrap().parameter_identity_bytes())
            )
        );
        assert_eq!(
            d.bootstrap_digest,
            f.bootstrap()
                .map(|b| crate::data::sha256_hex(b.descriptor_identity_bytes()))
        );
        assert_eq!(d.layout_digest, crate::data::sha256_hex(&d.layout));
        assert_eq!(d.keys_digest, crate::data::sha256_hex(&d.keys));
        assert_eq!(d.security_digest, crate::data::sha256_hex(&d.security));
    }
}
