//! Train-only live capture. Parsed provenance and inspection bytes cannot enter here.
use super::*;
use crate::study_profile::{CaseBackend, RegisteredCase};
use crate::study_runtime::CaseRuntimeFacts;

// These exact borrows keep the owner, registered selection and immutable input
// alive through recheck and serialization. No constructor/Clone/serde is exposed.
pub(crate) struct TrainCapture<'a> {
    context: &'a VerifiedStudyTrainContext,
    case: &'a RegisteredCase,
    input: &'a MaterializedProfile,
    checked: CheckedMaterializedInput<'a>,
    runtime: CaseRuntimeFacts<'a>,
    invocation: TrainInvocation,
    thread_environment: Vec<(String, Observation<String>)>,
    executable_start: String,
}
pub(crate) struct RecheckedTrainCapture<'a> {
    start: TrainCapture<'a>,
    end: RuntimeRecheck,
}
#[derive(PartialEq, Eq)]
struct TrainInvocation {
    argv: Vec<Vec<u8>>,
    cwd: Vec<u8>,
}
impl TrainInvocation {
    fn observe(root: &Path) -> Result<Self> {
        let cwd = std::env::current_dir()?;
        canonical(&cwd)?;
        ensure(cwd == root, "runtime cwd/source root mismatch")?;
        Ok(Self {
            argv: argv(std::env::args_os())?,
            cwd: cwd.as_os_str().as_bytes().to_vec(),
        })
    }
}
pub(super) fn check_train_role(role: ExecutableRole) -> Result<()> {
    ensure(
        role == ExecutableRole::StudyTrain,
        "current StudyTrain required",
    )
}
pub(super) fn check_case_materialized(
    case: &RegisteredCase,
    input: &MaterializedProfile,
    recipe: Recipe,
) -> Result<()> {
    check_train_role(recipe.role())?;
    ensure(
        input.identity().canonical_prepared() && input.shared_selector().is_some(),
        "canonical Train input required",
    )?;
    let selected = case.selector();
    ensure(
        selected.profile_id == input.metadata().id
            && selected.seed == u64::from(input.metadata().seed)
            && Some(case.logical_batch()) == u64::try_from(input.metadata().batch).ok(),
        "Train case/input selection mismatch",
    )?;
    let backend = match recipe {
        Recipe::TrainCpuSerial | Recipe::TrainCpuRayon => CaseBackend::Cpu,
        Recipe::TrainCudaHost => CaseBackend::Cuda,
        Recipe::Reference => return Err(ProvenanceError("current StudyTrain required")),
    };
    ensure(selected.backend == backend, "Train case/runtime mismatch")?;
    ensure(
        sha256_hex(case.canonical_bytes()) == case.digest(),
        "Train registered case binding changed",
    )
}
pub(super) fn require_capture_owners<C>(
    context: &C,
    previous_context: &C,
    case: &RegisteredCase,
    previous_case: &RegisteredCase,
    input: &MaterializedProfile,
    previous_input: &MaterializedProfile,
) -> Result<()> {
    ensure(
        std::ptr::eq(context, previous_context),
        "Train capture context changed",
    )?;
    ensure(
        std::ptr::eq(case, previous_case),
        "Train capture registered case changed",
    )?;
    ensure(
        std::ptr::eq(input, previous_input),
        "Train capture materialization changed",
    )
}
impl VerifiedStudyTrainContext {
    fn retained_train_runtime(&self) -> Result<&CaseRuntimeLease> {
        check_train_role(self.artifact.current_role())?;
        match &self.artifact.core.runtime {
            CurrentRuntime::Train(runtime) => {
                ensure(
                    runtime.recipe()
                        == self.artifact.core.receipts[self.artifact.core.current]
                            .role
                            .0,
                    "Train current receipt/runtime mismatch",
                )?;
                Ok(runtime)
            }
            CurrentRuntime::Reference => Err(ProvenanceError("retained Train runtime required")),
        }
    }
    pub(crate) fn capture_train<'a>(
        &'a self,
        case: &'a RegisteredCase,
        input: &'a MaterializedProfile,
    ) -> Result<TrainCapture<'a>> {
        let runtime = self.retained_train_runtime()?;
        check_case_materialized(case, input, runtime.recipe())?;
        let checked = self.artifact.check_materialized_input_now(input)?;
        let invocation = TrainInvocation::observe(self.artifact.root())?;
        let thread_environment = thread_environment(|k| std::env::var_os(k))?;
        let executable_start = self
            .artifact
            .current_build_facts()
            .executable_sha256()
            .into();
        let facts = runtime.facts()?;
        self.artifact.recheck()?;
        Ok(TrainCapture {
            context: self,
            case,
            input,
            checked,
            runtime: facts,
            invocation,
            thread_environment,
            executable_start,
        })
    }
    pub(crate) fn finish_train_capture<'a>(
        &'a self,
        case: &'a RegisteredCase,
        input: &'a MaterializedProfile,
        start: TrainCapture<'a>,
    ) -> Result<RecheckedTrainCapture<'a>> {
        require_capture_owners(self, start.context, case, start.case, input, start.input)?;
        let runtime = self.retained_train_runtime()?;
        check_case_materialized(case, input, runtime.recipe())?;
        self.artifact
            .recheck_materialized_input_now(input, &start.checked)?;
        let facts = runtime.facts()?;
        ensure(
            std::ptr::eq(facts.host(), start.runtime.host())
                && facts.execution_threads() == start.runtime.execution_threads(),
            "Train retained executor binding changed",
        )?;
        ensure(
            TrainInvocation::observe(self.artifact.root())? == start.invocation,
            "Train invocation changed",
        )?;
        let build = self.artifact.current_build_facts();
        ensure(
            build.executable_sha256() == start.executable_start,
            "Train captured executable changed",
        )?;
        let end = RuntimeRecheck {
            source: self.artifact.core.approval.source.clone(),
            receipt_file: self.artifact.current_receipt_sha256().into(),
            executable_end: build.executable_sha256().into(),
        };
        self.artifact.recheck()?;
        Ok(RecheckedTrainCapture { start, end })
    }
}
impl RecheckedTrainCapture<'_> {
    // Producer-only immutable inspection. Build access remains Train-role checked,
    // and comes from this exact context, not from receipt IDs supplied by callers.
    pub(crate) fn train_build_facts(&self) -> Result<BuildFacts<'_>> {
        self.start.context.retained_train_runtime()?.recheck()?;
        let artifact = &self.start.context.artifact;
        let build = artifact.current_build_facts();
        ensure(
            build.receipt_file_sha256() == self.start.checked.facts.receipt_file,
            "Train producer receipt changed",
        )?;
        Ok(build)
    }
    pub(crate) fn artifact_context(&self) -> &VerifiedStudyContext {
        &self.start.context.artifact
    }
    pub(crate) fn argv(&self) -> &[Vec<u8>] {
        &self.start.invocation.argv
    }
    pub(crate) fn cwd(&self) -> &[u8] {
        &self.start.invocation.cwd
    }
    pub(crate) fn thread_environment(&self) -> &[(String, Observation<String>)] {
        &self.start.thread_environment
    }
    pub(crate) fn host(&self) -> &Host {
        self.start.runtime.host()
    }
    pub(crate) fn execution_threads(&self) -> u64 {
        self.start.runtime.execution_threads()
    }
    pub(crate) fn executable_start_sha256(&self) -> &str {
        &self.start.executable_start
    }
    pub(crate) fn recheck(&self) -> &RuntimeRecheck {
        &self.end
    }
}
