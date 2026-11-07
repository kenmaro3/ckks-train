use super::*;
#[test]
fn runtime_worker_bounds_and_compile_role_match() {
    assert!(checked_workers(0).is_err());
    assert_eq!(checked_workers(1).unwrap(), 1);
    assert_eq!(checked_workers(2).unwrap(), 2);
    let expected = if cfg!(feature = "cuda") {
        Recipe::TrainCudaHost
    } else if cfg!(feature = "parallel") {
        Recipe::TrainCpuRayon
    } else {
        Recipe::TrainCpuSerial
    };
    for recipe in [
        Recipe::Reference,
        Recipe::TrainCpuSerial,
        Recipe::TrainCpuRayon,
        Recipe::TrainCudaHost,
    ] {
        assert_eq!(check_compiled_recipe(recipe).is_ok(), recipe == expected);
    }
}
#[test]
fn runtime_serial_owner_thread_is_not_transferable() {
    let host = HostExecutor::serial();
    assert_eq!(host.workers(), 1);
    host.recheck().unwrap();
    let error = std::thread::spawn(move || host.recheck().unwrap_err().0)
        .join()
        .unwrap();
    assert_eq!(error, "runtime owner thread changed");
}
#[cfg(feature = "parallel")]
#[test]
fn runtime_exact_pool_rejects_foreign_same_size_and_outside() {
    for count in [1, 2] {
        let pool = dedicated_pool(count).unwrap();
        let retained = pool.clone();
        let host = pool.install(move || {
            let host = HostExecutor::parallel(retained).unwrap();
            assert_eq!(host.workers(), count);
            host.recheck().unwrap();
            host
        });
        assert_eq!(host.recheck().unwrap_err().0, "outside retained Rayon pool");
        let foreign = dedicated_pool(count).unwrap();
        assert_eq!(
            foreign.install(|| host.recheck().unwrap_err().0),
            "outside retained Rayon pool"
        );
        assert!(pool.install(|| dedicated_pool(1)).is_err());
    }
}
#[cfg(feature = "parallel")]
#[test]
fn runtime_nested_join_iteration_and_synchronous_host_route_stay_owned() {
    use rayon::prelude::*;
    let pool = dedicated_pool(2).unwrap();
    let retained = pool.clone();
    pool.install(move || {
        let host = HostExecutor::parallel(retained).unwrap();
        let check = || {
            assert!(host.pool_index().is_some());
        };
        // Stand-in for the reviewed synchronous download/bootstrap/upload route.
        let synchronous_host = || {
            rayon::join(|| (0..32).into_par_iter().for_each(|_| check()), || check());
        };
        host.recheck().unwrap();
        synchronous_host();
        host.recheck().unwrap();
    });
}
#[cfg(feature = "parallel")]
#[test]
fn runtime_owned_pool_ignores_environment_in_isolated_child() {
    const KEY: &str = "CKKS_RUNTIME_POOL_TEST_CHILD";
    if std::env::var_os(KEY).is_some() {
        assert_eq!(std::env::var("RAYON_NUM_THREADS").unwrap(), "99");
        let pool = dedicated_pool(2).unwrap();
        let retained = pool.clone();
        pool.install(move || {
            let host = HostExecutor::parallel(retained).unwrap();
            assert_eq!(host.workers(), 2);
            host.recheck().unwrap();
        });
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "study_runtime::tests::runtime_owned_pool_ignores_environment_in_isolated_child",
            "--test-threads=1",
        ])
        .env(KEY, "1")
        .env("RAYON_NUM_THREADS", "99")
        .status()
        .unwrap();
    assert!(status.success());
}
#[cfg(not(target_os = "linux"))]
#[test]
fn runtime_mac_never_fabricates_production_process_stamp() {
    assert!(ProcessStamp::capture().is_err());
}

#[cfg(feature = "parallel")]
#[test]
fn runtime_retained_count_drift_fails_even_inside_exact_pool() {
    let pool = dedicated_pool(2).unwrap();
    let retained = pool.clone();
    pool.install(move || {
        let mut host = HostExecutor::parallel(retained).unwrap();
        if let HostPool::Rayon { count, .. } = &mut host.pool {
            *count = 1;
        }
        assert_eq!(
            host.recheck().unwrap_err().0,
            "retained worker count changed"
        );
    });
}

#[test]
fn runtime_owned_selector_refuses_backend_before_context_or_pool() {
    use crate::study_profile::{CaseMode, CaseOrigin, CasePacking, CasePlacement, CaseScope};
    let request = StudyRunRequest {
        locations: ContextRequest::new(
            "/nonexistent".into(),
            "/nonexistent/approval".into(),
            "a".repeat(64),
            vec!["/nonexistent/receipt".into()],
        ),
        runtime: if cfg!(feature = "cuda") {
            RuntimeRequest::CudaHost {
                workers: 2,
                ordinal: 0,
            }
        } else if cfg!(feature = "parallel") {
            RuntimeRequest::RayonCpu { workers: 2 }
        } else {
            RuntimeRequest::SerialCpu
        },
        case: CaseSelector {
            origin: CaseOrigin::Family,
            registration_id: "untrusted".into(),
            profile_id: "untrusted".into(),
            seed: 0,
            scope: CaseScope::Full,
            backend: CaseBackend::ExternalCpu,
            mode: CaseMode::Interactive,
            placement: CasePlacement::ClientRefresh,
            packing: CasePacking::BatchedDiagonal,
        },
    };
    assert_eq!(
        dispatch(request).unwrap_err().0,
        "case/runtime backend mismatch"
    );
}
