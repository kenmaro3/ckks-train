//! Concrete retained study executor. This is not D7 admission or a complete runner.
#![allow(dead_code)] // Consumed by the separately owned concrete frontend / D7 boundary.
use crate::study_profile::{CaseBackend, CaseSelector, StudyRegistry};
use crate::study_provenance::{
    ContextRequest, Host, Observation, ProvenanceError, Result, VerifiedStudyTrainContext,
    check_runtime_environment, check_train_preflight, process::ProcessStamp, role::Recipe,
};
#[cfg(feature = "parallel")]
use std::sync::Arc;
use std::{marker::PhantomData, rc::Rc, thread::ThreadId};
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub(crate) use cuda::CudaRuntimeFacts;
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests;
#[cfg(test)]
mod tests;

fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(ProvenanceError(why))
    }
}
// Only requests cross install. None of these values asserts readiness.
pub(crate) enum RuntimeRequest {
    SerialCpu,
    RayonCpu { workers: u64 },
    CudaHost { workers: u64, ordinal: usize },
}
pub(crate) struct StudyRunRequest {
    pub(crate) locations: ContextRequest,
    pub(crate) runtime: RuntimeRequest,
    pub(crate) case: CaseSelector,
}
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RunOutcome {
    MissingExecutionDependency,
}
pub(crate) struct StudyTrainContextRequest {
    locations: ContextRequest,
    runtime: CaseRuntimeLease,
}
impl StudyTrainContextRequest {
    pub(crate) fn into_parts(self) -> (ContextRequest, CaseRuntimeLease) {
        (self.locations, self.runtime)
    }
}
fn checked_workers(request: u64) -> Result<usize> {
    let count = usize::try_from(request).map_err(|_| ProvenanceError("runtime worker overflow"))?;
    require(count > 0, "runtime workers must be positive")?;
    Ok(count)
}
fn check_compiled_recipe(recipe: Recipe) -> Result<()> {
    let compiled = if cfg!(feature = "cuda") {
        Recipe::TrainCudaHost
    } else if cfg!(feature = "parallel") {
        Recipe::TrainCpuRayon
    } else {
        Recipe::TrainCpuSerial
    };
    require(
        recipe == compiled,
        "runtime form/compiled features mismatch",
    )
}
impl RuntimeRequest {
    fn recipe(&self) -> Recipe {
        match self {
            Self::SerialCpu => Recipe::TrainCpuSerial,
            Self::RayonCpu { .. } => Recipe::TrainCpuRayon,
            Self::CudaHost { .. } => Recipe::TrainCudaHost,
        }
    }
    fn backend(&self) -> CaseBackend {
        if matches!(self, Self::CudaHost { .. }) {
            CaseBackend::Cuda
        } else {
            CaseBackend::Cpu
        }
    }
}
// This private primitive has no process/approval authority. Tests may move it to
// inspect refused threads; the actual lease below is non-Send/non-Sync.
struct HostExecutor {
    owner: ThreadId,
    pool: HostPool,
}
enum HostPool {
    Serial,
    #[cfg(feature = "parallel")]
    Rayon {
        pool: Arc<rayon::ThreadPool>,
        count: usize,
    },
}
impl HostExecutor {
    fn facts<'a>(&self, host: &'a Host) -> Result<CaseRuntimeFacts<'a>> {
        self.recheck()?;
        Ok(CaseRuntimeFacts {
            host,
            execution_threads: u64::try_from(self.workers())
                .map_err(|_| ProvenanceError("worker count representation"))?,
        })
    }
    fn serial() -> Self {
        Self {
            owner: std::thread::current().id(),
            pool: HostPool::Serial,
        }
    }
    #[cfg(feature = "parallel")]
    fn parallel(pool: Arc<rayon::ThreadPool>) -> Result<Self> {
        let count = pool.current_num_threads();
        require(
            count > 0 && pool.current_thread_index().is_some(),
            "outside retained Rayon pool",
        )?;
        Ok(Self {
            owner: std::thread::current().id(),
            pool: HostPool::Rayon { pool, count },
        })
    }
    fn workers(&self) -> usize {
        match &self.pool {
            HostPool::Serial => 1,
            #[cfg(feature = "parallel")]
            HostPool::Rayon { count, .. } => *count,
        }
    }
    #[cfg(feature = "parallel")]
    fn pool_index(&self) -> Option<usize> {
        match &self.pool {
            HostPool::Serial => None,
            HostPool::Rayon { pool, .. } => pool.current_thread_index(),
        }
    }
    fn recheck(&self) -> Result<()> {
        #[cfg(feature = "parallel")]
        if let HostPool::Rayon { pool, count } = &self.pool {
            require(
                pool.current_thread_index().is_some(),
                "outside retained Rayon pool",
            )?;
            require(
                *count > 0 && pool.current_num_threads() == *count,
                "retained worker count changed",
            )?;
        }
        require(
            std::thread::current().id() == self.owner,
            "runtime owner thread changed",
        )
    }
}
#[cfg(feature = "parallel")]
fn dedicated_pool(count: usize) -> Result<Arc<rayon::ThreadPool>> {
    // TLS-only observation; never initialize or discover the global pool.
    require(
        rayon::current_thread_index().is_none(),
        "dispatcher already in Rayon pool",
    )?;
    require(count > 0, "runtime workers must be positive")?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(count)
        .build()
        .map_err(|_| ProvenanceError("dedicated Rayon pool unavailable"))?;
    require(
        pool.current_num_threads() == count,
        "requested worker count not retained",
    )?;
    Ok(Arc::new(pool))
}
pub(crate) struct CaseRuntimeLease {
    recipe: Recipe,
    executor: HostExecutor,
    process: ProcessStamp,
    host: Box<Host>,
    #[cfg(feature = "cuda")]
    cuda: Option<Box<cuda::CudaBinding>>,
    _not_send_sync: PhantomData<Rc<()>>,
}
// Immutable observations, never a runtime lease or readiness capability. Only
// the retained executor supplies the worker count; Host's scalar helper is not used.
pub(crate) struct CaseRuntimeFacts<'a> {
    host: &'a Host,
    execution_threads: u64,
}
impl CaseRuntimeFacts<'_> {
    pub(crate) fn host(&self) -> &Host {
        self.host
    }
    pub(crate) fn execution_threads(&self) -> u64 {
        self.execution_threads
    }
}
// This view observes an actual retained runtime. It supplies no implementation
// route, sequential-dispatch, stream-construction or admission authority.
pub(crate) struct RuntimeDescriptorFacts<'a> {
    runtime: CaseRuntimeFacts<'a>,
    recipe: Recipe,
    #[cfg(feature = "cuda")]
    cuda: Option<CudaRuntimeFacts<'a>>,
}
impl RuntimeDescriptorFacts<'_> {
    pub(crate) fn host(&self) -> &Host {
        self.runtime.host()
    }
    pub(crate) fn execution_threads(&self) -> u64 {
        self.runtime.execution_threads()
    }
    pub(crate) fn recipe(&self) -> Recipe {
        self.recipe
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda(&self) -> Option<&CudaRuntimeFacts<'_>> {
        self.cuda.as_ref()
    }
}
fn host_identity(host: &Host) -> Result<(&str, &str, &str)> {
    match (host.cpu_model(), host.os(), host.hostname()) {
        (Observation::Available(cpu), Observation::Available(os), Observation::Available(name)) => {
            Ok((cpu, os, name))
        }
        _ => Err(ProvenanceError("runtime host identity unavailable")),
    }
}
impl CaseRuntimeLease {
    pub(crate) fn descriptor_facts(&self) -> Result<RuntimeDescriptorFacts<'_>> {
        self.recheck()?;
        let facts = RuntimeDescriptorFacts {
            runtime: self.executor.facts(&self.host)?,
            recipe: self.recipe,
            #[cfg(feature = "cuda")]
            cuda: self
                .cuda
                .as_ref()
                .map(|binding| binding.facts())
                .transpose()?,
        };
        self.recheck()?;
        Ok(facts)
    }

    pub(crate) fn facts(&self) -> Result<CaseRuntimeFacts<'_>> {
        self.recheck()?;
        let facts = self.executor.facts(&self.host)?;
        self.recheck()?;
        Ok(facts)
    }
    // Only the concrete installed dispatcher calls this; no evidence injection.
    fn capture(recipe: Recipe, executor: HostExecutor, ordinal: Option<usize>) -> Result<Self> {
        check_compiled_recipe(recipe)?;
        executor.recheck()?;
        check_runtime_environment()?;
        let process = ProcessStamp::capture()?;
        let host = Box::new(Host::observe());
        host_identity(&host)?;
        #[cfg(feature = "cuda")]
        let cuda = match (recipe, ordinal) {
            (Recipe::TrainCudaHost, Some(ordinal)) => {
                Some(Box::new(cuda::CudaBinding::observe(ordinal)?))
            }
            _ => return Err(ProvenanceError("CUDA runtime selection mismatch")),
        };
        #[cfg(not(feature = "cuda"))]
        require(ordinal.is_none(), "unexpected CUDA ordinal")?;
        let lease = Self {
            recipe,
            executor,
            process,
            host,
            #[cfg(feature = "cuda")]
            cuda,
            _not_send_sync: PhantomData,
        };
        lease.recheck()?;
        Ok(lease)
    }
    pub(crate) fn recipe(&self) -> Recipe {
        self.recipe
    }
    pub(crate) fn execution_threads(&self) -> Result<u64> {
        self.recheck()?;
        u64::try_from(self.executor.workers())
            .map_err(|_| ProvenanceError("worker count representation"))
    }
    pub(crate) fn recheck(&self) -> Result<()> {
        self.executor.recheck()?;
        self.process.recheck()?;
        check_runtime_environment()?;
        require(
            host_identity(&self.host)? == host_identity(&Host::observe())?,
            "runtime host identity changed",
        )?;
        #[cfg(feature = "cuda")]
        if let Some(cuda) = &self.cuda {
            cuda.recheck()?;
        }
        self.executor.recheck()?;
        self.process.recheck()
    }
}
// All authority remains in this one synchronous owner frame. Later D7 extends
// this function, never a caller callback or a per-operation pool.install.
fn run_owned(request: StudyRunRequest, executor: HostExecutor) -> Result<RunOutcome> {
    let recipe = request.runtime.recipe();
    let ordinal = match request.runtime {
        RuntimeRequest::CudaHost { ordinal, .. } => Some(ordinal),
        _ => None,
    };
    let runtime = CaseRuntimeLease::capture(recipe, executor, ordinal)?;
    let context = VerifiedStudyTrainContext::load(StudyTrainContextRequest {
        locations: request.locations,
        runtime,
    })?;
    context.artifact_context().recheck()?;
    // Select only after loading the actual pinned current context, inside this
    // owner frame. Membership is expected registration, never checked readiness.
    let registry = StudyRegistry::load(context.artifact_context().root())
        .map_err(|_| ProvenanceError("registered study source unavailable"))?;
    let cases = registry
        .registered_cases()
        .map_err(|_| ProvenanceError("registered study projection unavailable"))?;
    let case = cases
        .select(&request.case)
        .map_err(|_| ProvenanceError("unregistered study selector"))?;
    let seed = u8::try_from(case.selector().seed)
        .map_err(|_| ProvenanceError("registered study seed representation"))?;
    let resolved = registry
        .resolve(&case.selector().profile_id, seed)
        .map_err(|_| ProvenanceError("selected study input resolution failed"))?;
    let input = crate::study_inputs::materialize(&resolved)
        .map_err(|_| ProvenanceError("selected study input materialization failed"))?;
    let capture = context.capture_train(case, &input)?;
    let rechecked = context.finish_train_capture(case, &input, capture)?;
    let _component = rechecked
        .case_provenance()
        .map_err(|_| ProvenanceError("captured study Case component encoding failed"))?;
    let _physical = rechecked
        .physical_descriptor()
        .map_err(|_| ProvenanceError("captured study physical descriptor encoding failed"))?;
    // Captured Case/physical bytes are inspection only. Checked runtime route,
    // remaining canonical D7 descriptors,
    // live gates/admission/start, scientific output and HE continuation are absent.
    Ok(RunOutcome::MissingExecutionDependency)
}
pub(crate) fn dispatch(request: StudyRunRequest) -> Result<RunOutcome> {
    let recipe = request.runtime.recipe();
    check_compiled_recipe(recipe)?;
    require(
        request.case.backend == request.runtime.backend(),
        "case/runtime backend mismatch",
    )?;
    check_train_preflight(recipe)?;
    match request.runtime {
        RuntimeRequest::SerialCpu => run_owned(request, HostExecutor::serial()),
        RuntimeRequest::RayonCpu { workers } | RuntimeRequest::CudaHost { workers, .. } => {
            let count = checked_workers(workers)?;
            #[cfg(feature = "parallel")]
            {
                let pool = dedicated_pool(count)?;
                let retained = Arc::clone(&pool);
                pool.install(move || run_owned(request, HostExecutor::parallel(retained)?))
            }
            #[cfg(not(feature = "parallel"))]
            {
                let _ = count;
                Err(ProvenanceError("parallel runtime not compiled"))
            }
        }
    }
}
