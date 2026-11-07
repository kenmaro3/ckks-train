//! Pure predicate coverage only; these tests never forge a verified Train context.
use super::train_capture::{check_case_materialized, check_train_role, require_capture_owners};
use super::*;
use crate::study_profile::{CaseBackend, CaseRole, StudyRegistry};

// Compile-only wiring, deliberately not a runtime #[test]. Actual approved Linux
// context creation and the installed run_owned path remain separate evidence.
#[allow(dead_code)]
fn train_capture_actual_api_compile_only<'a>(
    context: &'a VerifiedStudyTrainContext,
    case: &'a crate::study_profile::RegisteredCase,
    input: &'a MaterializedProfile,
) -> Result<()> {
    let capture = context.capture_train(case, input)?;
    let checked = context.finish_train_capture(case, input, capture)?;
    let component = checked
        .case_provenance()
        .map_err(|_| ProvenanceError("case codec"))?;
    assert_eq!(sha256_hex(component.canonical_bytes()), component.sha256());
    Ok(())
}

// Compile-only negative trait assertions: adding any of these impls makes type
// inference ambiguous. These operate on the real private capture types, without
// manufacturing a context or relying only on external import privacy errors.
const _: fn() = || {
    macro_rules! not_impl {
        ($ty:ty, $bound:path) => {{
            trait AmbiguousIfImpl<A> {
                fn marker() {}
            }
            impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
            impl<T: ?Sized + $bound> AmbiguousIfImpl<u8> for T {}
            let _ = <$ty as AmbiguousIfImpl<_>>::marker;
        }};
    }
    not_impl!(TrainCapture<'static>, Send);
    not_impl!(TrainCapture<'static>, Sync);
    not_impl!(TrainCapture<'static>, Clone);
    not_impl!(TrainCapture<'static>, Default);
    not_impl!(RecheckedTrainCapture<'static>, Send);
    not_impl!(RecheckedTrainCapture<'static>, Sync);
    not_impl!(RecheckedTrainCapture<'static>, Clone);
    not_impl!(RecheckedTrainCapture<'static>, Default);
    not_impl!(TrainCapture<'static>, serde::Deserialize<'static>);
    not_impl!(RecheckedTrainCapture<'static>, serde::Deserialize<'static>);
};

#[test]
fn train_capture_canonical_iris_0_7_case_input_and_owner_binding() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let registry = StudyRegistry::load(root).unwrap();
    let cases = registry.registered_cases().unwrap();
    // Exactly two materializations, one each for canonical Iris seeds0/7; no learning.
    let p0 =
        crate::study_inputs::materialize(&registry.resolve("iris__emlp1", 0).unwrap()).unwrap();
    let p7 =
        crate::study_inputs::materialize(&registry.resolve("iris__emlp1", 7).unwrap()).unwrap();
    let case = |seed| {
        cases
            .family_cases()
            .iter()
            .find(|c| {
                c.selector().profile_id == "iris__emlp1"
                    && c.selector().seed == seed
                    && c.selector().backend == CaseBackend::Cpu
                    && c.role() == CaseRole::Main
            })
            .unwrap()
    };
    let c0 = case(0);
    let c7 = case(7);
    check_case_materialized(c0, &p0, Recipe::TrainCpuSerial).unwrap();
    check_case_materialized(c7, &p7, Recipe::TrainCpuRayon).unwrap();
    assert!(check_case_materialized(c0, &p7, Recipe::TrainCpuSerial).is_err());
    assert!(check_case_materialized(c7, &p0, Recipe::TrainCpuSerial).is_err());
    assert!(check_case_materialized(c0, &p0, Recipe::Reference).is_err());
    assert!(check_case_materialized(c0, &p0, Recipe::TrainCudaHost).is_err());
    let other_profile = cases
        .family_cases()
        .iter()
        .find(|c| {
            c.selector().profile_id != "iris__emlp1"
                && c.selector().seed == 0
                && c.selector().backend == CaseBackend::Cpu
        })
        .unwrap();
    assert!(check_case_materialized(other_profile, &p0, Recipe::TrainCpuSerial).is_err());
    let other_case = cases
        .family_cases()
        .iter()
        .find(|c| {
            c.selector().profile_id == "iris__emlp1"
                && c.selector().seed == 0
                && c.selector().backend == CaseBackend::Cpu
                && !std::ptr::eq(*c, c0)
        })
        .unwrap();
    // Plain owner sentinels exercise the same exact pointer predicate as production.
    let owner = Box::new(3u8);
    let other_owner = Box::new(3u8);
    require_capture_owners(&*owner, &*owner, c0, c0, &p0, &p0).unwrap();
    assert!(require_capture_owners(&*owner, &*other_owner, c0, c0, &p0, &p0).is_err());
    assert!(require_capture_owners(&*owner, &*owner, other_case, c0, &p0, &p0).is_err());
    assert!(require_capture_owners(&*owner, &*owner, c7, c0, &p7, &p0).is_err());
    assert!(require_capture_owners(&*owner, &*owner, c0, c0, &p7, &p0).is_err());
    let facts =
        |p: &MaterializedProfile| checked_materialized_facts(p, "receipt", "source", |_| Ok(()));
    let start = facts(&p0).unwrap();
    assert_eq!(start, facts(&p0).unwrap());
    assert_eq!(start.contract_hash, facts(&p7).unwrap().contract_hash);
    assert_ne!(
        start.input_receipt_hash,
        facts(&p7).unwrap().input_receipt_hash
    );
    assert_ne!(start, facts(&p7).unwrap());
    assert_ne!(
        start,
        checked_materialized_facts(&p0, "changed receipt", "source", |_| Ok(())).unwrap()
    );
    assert_ne!(
        start,
        checked_materialized_facts(&p0, "receipt", "changed source", |_| Ok(())).unwrap()
    );
    assert!(
        checked_materialized_facts(&p0, "receipt", "source", |_| Err(ProvenanceError(
            "source changed"
        )))
        .is_err()
    );
}

#[test]
fn train_capture_role_is_distinct_from_reference_authority() {
    assert!(check_train_role(ExecutableRole::Reference).is_err());
    check_train_role(ExecutableRole::StudyTrain).unwrap();
    assert!(require_reference_role(ExecutableRole::StudyTrain).is_err());
}
