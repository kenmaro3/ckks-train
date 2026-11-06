//! Private sequential sessions. Public results have no caller-data constructors.
use super::{
    SeedInputReceiptV1, StudyProfileContractV1, derive_identities,
    digests::{self, Scalar},
};
use crate::{
    data::sha256_hex,
    study_inputs::{CastPolicy, Dtype, MaterializedProfile},
    study_plain::{PlainEvaluation, PlainState, PlainStep, evaluate_plain, step_plain},
    study_profile::BatchPlan,
    study_provenance::{RuntimeCapture, RuntimeRecheck, VerifiedStudyContext},
};
use std::{
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub(super) type Result<T> = std::result::Result<T, SessionError>;
/// Stable sanitized error; never includes arbitrary source paths or caller data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionError(pub(super) &'static str);
impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for SessionError {}
fn require(ok: bool, why: &'static str) -> Result<()> {
    if ok { Ok(()) } else { Err(SessionError(why)) }
}
pub(super) fn valid_id(id: &str) -> Result<()> {
    require(
        !id.is_empty()
            && id.len() <= 128
            && id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-'),
        "invalid_run_id",
    )
}
/// Successful progress, not a completion capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionProgress {
    updates: usize,
    samples: usize,
}
impl SessionProgress {
    #[doc = "Number of actual successfully completed updates."]
    pub fn completed_updates(&self) -> usize {
        self.updates
    }
    #[doc = "Number of actual successfully processed samples."]
    pub fn processed_samples(&self) -> usize {
        self.samples
    }
}
#[derive(Debug, Clone)]
pub(super) struct EndpointFact {
    pub checkpoint: crate::study_profile::Checkpoint,
    pub evaluation: PlainEvaluation,
    pub update_ns: u64,
    pub evaluation_ns: u64,
}
#[derive(Debug, Clone)]
pub(super) struct TimingFact {
    pub start: u64,
    pub end: u64,
    pub total: u64,
    pub setup: u64,
    pub update: u64,
    pub evaluation: u64,
    pub finalization: u64,
}
trait Clock {
    fn now(&self) -> Result<u64>;
    fn wall(&self) -> Result<u64>;
}
struct NativeClock(Instant);
impl Clock for NativeClock {
    fn now(&self) -> Result<u64> {
        u64::try_from(self.0.elapsed().as_nanos()).map_err(|_| SessionError("monotonic_range"))
    }
    fn wall(&self) -> Result<u64> {
        u64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|_| SessionError("wall_before_epoch"))?
                .as_nanos(),
        )
        .map_err(|_| SessionError("wall_range"))
    }
}
struct Timer<C> {
    clock: C,
    start: u64,
    start_wall: u64,
    totals: [u64; 4],
    final_start: Option<u64>,
    invalid: bool,
}
impl<C: Clock> Timer<C> {
    fn new(clock: C) -> Result<Self> {
        let start = clock.now()?;
        let start_wall = clock.wall()?;
        Ok(Self {
            clock,
            start,
            start_wall,
            totals: [0; 4],
            final_start: None,
            invalid: false,
        })
    }
    fn now(&mut self) -> Result<u64> {
        let r = self.clock.now();
        if r.is_err() {
            self.invalid = true;
        }
        r
    }
    fn charge(&mut self, category: usize, start: u64) -> Result<()> {
        let end = self.now()?;
        let result = end
            .checked_sub(start)
            .and_then(|d| self.totals[category].checked_add(d));
        match result {
            Some(v) => {
                self.totals[category] = v;
                Ok(())
            }
            None => {
                self.invalid = true;
                Err(SessionError("timer_overflow_or_reversal"))
            }
        }
    }
    fn begin_final(&mut self) -> Result<()> {
        require(self.final_start.is_none(), "overlapping_finalization")?;
        self.final_start = Some(self.now()?);
        Ok(())
    }
    fn pause_final(&mut self) -> Result<()> {
        if let Some(start) = self.final_start.take() {
            self.charge(3, start)?;
        }
        Ok(())
    }
    fn end(&mut self) -> Result<TimingFact> {
        let result = self.end_observed();
        if result.is_err() {
            self.invalid = true;
        }
        result
    }
    fn end_observed(&mut self) -> Result<TimingFact> {
        self.pause_final()?;
        require(!self.invalid, "invalid_timing")?;
        let end = self.now()?;
        let wall = self.clock.wall()?;
        let start_wall = self.start_wall;
        require(wall >= start_wall, "wall_reversal")?;
        let total = end
            .checked_sub(self.start)
            .ok_or(SessionError("monotonic_reversal"))?;
        let sum = self.totals.iter().try_fold(0u64, |a, &b| {
            a.checked_add(b).ok_or(SessionError("timer_sum_overflow"))
        })?;
        require(sum <= total, "overlapping_timing")?;
        Ok(TimingFact {
            start: start_wall,
            end: wall,
            total,
            setup: self.totals[0],
            update: self.totals[1],
            evaluation: self.totals[2],
            finalization: self.totals[3],
        })
    }
}
fn finish_setup_failure<C: Clock>(timer: &mut Timer<C>) -> Option<TimingFact> {
    timer.charge(0, timer.start).ok()?;
    timer.end().ok()
}
// A shared private gate keeps the initial clock observation before every setup effect.
// Tests supply only a clock and synthetic closure; no production context is fabricated.
fn prepare_session<C: Clock, R>(
    clock: C,
    run_id: &str,
    dtype: u8,
    scope: &'static str,
    work: impl FnOnce() -> Result<R>,
) -> std::result::Result<(Timer<C>, R), SessionFailure> {
    let mut timer = Timer::new(clock).map_err(|e| early_failure(run_id, dtype, scope, e))?;
    match work() {
        Ok(prepared) => Ok((timer, prepared)),
        Err(error) => {
            let mut failure = early_failure(run_id, dtype, scope, error);
            failure.detail.timing = finish_setup_failure(&mut timer);
            Err(failure)
        }
    }
}
trait Engine {
    fn progress(&self) -> SessionProgress;
    fn poisoned(&self) -> bool;
    fn step(&mut self) -> Result<PlainStep>;
    fn evaluate(&self) -> Result<PlainEvaluation>;
}
struct Real<'p, T: Scalar>(PlainState<'p, T>);
impl<T: Scalar> Engine for Real<'_, T> {
    fn progress(&self) -> SessionProgress {
        SessionProgress {
            updates: self.0.completed_updates(),
            samples: self.0.processed_samples(),
        }
    }
    fn poisoned(&self) -> bool {
        self.0.is_poisoned()
    }
    fn step(&mut self) -> Result<PlainStep> {
        step_plain(&mut self.0).map_err(|_| SessionError("plain_update_failed"))
    }
    fn evaluate(&self) -> Result<PlainEvaluation> {
        evaluate_plain(&self.0).map_err(|_| SessionError("plain_evaluation_failed"))
    }
}
#[derive(Debug, Clone, Copy)]
struct Failed {
    phase: &'static str,
    error: SessionError,
}
struct Driver<'p, E, C> {
    engine: E,
    plan: &'p BatchPlan,
    denominator: usize,
    limit: usize,
    expected: SessionProgress,
    endpoints: Vec<EndpointFact>,
    timer: Timer<C>,
    failed: Option<Failed>,
    failure_timing: Option<TimingFact>,
}
impl<'p, E: Engine, C: Clock> Driver<'p, E, C> {
    fn new(
        engine: E,
        plan: &'p BatchPlan,
        denominator: usize,
        limit: usize,
        mut timer: Timer<C>,
    ) -> Result<Self> {
        require(
            engine.progress()
                == SessionProgress {
                    updates: 0,
                    samples: 0,
                }
                && !engine.poisoned(),
            "nonzero_origin",
        )?;
        require(
            denominator > 0 && denominator as u128 <= 1u128 << 53 && limit <= plan.updates.len(),
            "invalid_session_plan",
        )?;
        timer.charge(0, timer.start)?;
        Ok(Self {
            engine,
            plan,
            denominator,
            limit,
            expected: SessionProgress {
                updates: 0,
                samples: 0,
            },
            endpoints: Vec::new(),
            timer,
            failed: None,
            failure_timing: None,
        })
    }
    fn fail(&mut self, phase: &'static str, error: SessionError) -> SessionError {
        if self.failed.is_none() {
            self.failed = Some(Failed { phase, error });
            self.failure_timing = self.timer.end().ok();
        }
        error
    }
    fn running(&self) -> Result<()> {
        require(self.failed.is_none(), "session_failed")?;
        require(!self.engine.poisoned(), "engine_poisoned")?;
        require(self.engine.progress() == self.expected, "cursor_mismatch")
    }
    fn evaluate_call(&mut self) -> Result<PlainEvaluation> {
        let start = self.timer.now()?;
        let result = self.engine.evaluate();
        self.timer.charge(2, start)?;
        result
    }
    fn validate_evaluation(&self, observation: PlainEvaluation) -> Result<PlainEvaluation> {
        require(
            observation.test_count == self.denominator
                && observation.correct <= observation.test_count,
            "evaluation_denominator",
        )?;
        Ok(observation)
    }
    fn advance(&mut self) -> Result<SessionProgress> {
        let mut phase = "update";
        let result = (|| {
            self.running()?;
            require(self.expected.updates < self.limit, "session_exhausted")?;
            let u = &self.plan.updates[self.expected.updates];
            require(
                u.index == self.expected.updates && u.processed_start == self.expected.samples,
                "plan_cursor",
            )?;
            let samples = u
                .processed_start
                .checked_add(u.valid_samples)
                .ok_or(SessionError("sample_overflow"))?;
            let start = self.timer.now()?;
            let step = self.engine.step();
            self.timer.charge(1, start)?;
            let step = step?;
            // The real update has happened even if subsequent accounting/evaluation fails.
            self.expected = SessionProgress {
                updates: u.index + 1,
                samples,
            };
            require(
                step.update_index == u.index
                    && step.valid_samples == u.valid_samples
                    && step.processed_samples == samples,
                "poststep_accounting",
            )?;
            self.running()?;
            if let Some(c) = self
                .plan
                .checkpoints
                .get(self.endpoints.len())
                .filter(|c| c.completed_updates == self.expected.updates)
                .cloned()
            {
                phase = "checkpoint";
                require(c.processed_samples == samples, "checkpoint_samples")?;
                let observation = self.evaluate_call()?;
                let observation = self.validate_evaluation(observation)?;
                self.endpoints.push(EndpointFact {
                    checkpoint: c,
                    evaluation: observation,
                    update_ns: self.timer.totals[1],
                    evaluation_ns: self.timer.totals[2],
                });
            }
            Ok(self.expected)
        })();
        result.map_err(|e| self.fail(phase, e))
    }
    fn prepare_finish(&mut self, full: bool) -> Result<PlainEvaluation> {
        let result = (|| {
            require(self.failed.is_none(), "session_failed")?;
            self.timer.begin_final()?;
            self.running()?;
            require(
                self.expected.updates == if full { self.plan.updates.len() } else { 2 },
                "incomplete_session",
            )?;
            let scheduled =
                self.plan.updates[..self.expected.updates]
                    .iter()
                    .try_fold(0usize, |a, u| {
                        a.checked_add(u.valid_samples)
                            .ok_or(SessionError("sample_overflow"))
                    })?;
            require(self.expected.samples == scheduled, "incomplete_samples")?;
            let required = self
                .plan
                .checkpoints
                .iter()
                .filter(|c| c.completed_updates <= self.expected.updates)
                .count();
            require(self.endpoints.len() == required, "incomplete_checkpoints")?;
            if full {
                require(
                    self.expected.samples == self.plan.processed_samples
                        && self.endpoints.last().is_some_and(|e| {
                            e.checkpoint.completed_updates == self.expected.updates
                                && e.checkpoint.processed_samples == self.expected.samples
                        }),
                    "missing_final_checkpoint",
                )?;
            }
            if let Some(last) = self.endpoints.last().filter(|e| {
                e.checkpoint.completed_updates == self.expected.updates
                    && e.checkpoint.processed_samples == self.expected.samples
            }) {
                return Ok(last.evaluation);
            }
            self.timer.pause_final()?;
            let evaluated = self.evaluate_call();
            // No finalization interval is open during evaluation, including failure.
            let observation = evaluated?;
            self.timer.begin_final()?;
            self.validate_evaluation(observation)
        })();
        result.map_err(|e| self.fail("finalize", e))
    }
    fn complete(&mut self) -> Result<TimingFact> {
        let result = self.timer.end();
        result.map_err(|e| self.fail("finalize", e))
    }
}
#[derive(Debug)]
pub(super) struct IdentityFact {
    pub profile: String,
    pub contract: StudyProfileContractV1,
    pub receipt: SeedInputReceiptV1,
}
#[derive(Debug)]
pub(super) struct Record {
    pub run_id: String,
    pub identity: IdentityFact,
    pub dtype: u8,
    pub full: bool,
    pub plan: BatchPlan,
    pub progress: SessionProgress,
    pub endpoints: Vec<EndpointFact>,
    pub observation: PlainEvaluation,
    pub state: String,
    pub consumed: String,
    pub full_hash: String,
    pub timing: TimingFact,
}
pub(super) struct CaptureFact<'a> {
    pub context: &'a VerifiedStudyContext,
    pub start: RuntimeCapture,
    pub end: RuntimeRecheck,
}
struct Session<'p, T: Scalar> {
    profile: &'p MaterializedProfile,
    identity: IdentityFact,
    run_id: String,
    driver: Driver<'p, Real<'p, T>, NativeClock>,
    scope: &'static str,
}
impl<'p, T: Scalar> Session<'p, T> {
    fn start_with<R>(
        profile: &'p MaterializedProfile,
        run_id: &str,
        quick: bool,
        capture: impl FnOnce(&IdentityFact) -> Result<R>,
    ) -> std::result::Result<(Self, R), SessionFailure> {
        let scope = if quick {
            "quick"
        } else if T::WIDTH == 32 {
            "full_fp32"
        } else {
            "full_f64_diagnostic"
        };
        let (timer, (identity, captured, state, denominator, limit)) =
            prepare_session(NativeClock(Instant::now()), run_id, T::WIDTH, scope, || {
                valid_id(run_id)?;
                let (contract, receipt) =
                    derive_identities(profile).map_err(|_| SessionError("canonical_identity"))?;
                let identity = IdentityFact {
                    profile: profile.metadata().id.clone(),
                    contract,
                    receipt,
                };
                let captured = capture(&identity)?;
                let state = PlainState::<T>::new(profile)
                    .map_err(|_| SessionError("plain_setup_failed"))?;
                require(
                    state.input_identity() == identity.receipt.materialized_digest()
                        && identity.receipt.contract_digest() == identity.contract.digest(),
                    "state_identity",
                )?;
                require(
                    matches!(
                        (T::WIDTH, state.dtype(), state.cast_policy()),
                        (32, Dtype::F32, CastPolicy::RoundToF32Once)
                            | (64, Dtype::F64, CastPolicy::CanonicalF64)
                    ),
                    "state_arithmetic",
                )?;
                let inputs = T::inputs(profile);
                let denominator = inputs.test_y().len();
                let limit = if quick {
                    2
                } else {
                    profile.plan().updates.len()
                };
                require(
                    inputs.test_x().nrows() == denominator
                        && denominator > 0
                        && denominator as u128 <= 1u128 << 53
                        && limit <= profile.plan().updates.len(),
                    "invalid_session_plan",
                )?;
                require(
                    state.completed_updates() == 0
                        && state.processed_samples() == 0
                        && !state.is_poisoned(),
                    "nonzero_origin",
                )?;
                Ok((identity, captured, state, denominator, limit))
            })?;
        // All semantic setup predicates were checked above while the timer is retained.
        // The only remaining failure is invalid monotonic timing, which stays unavailable.
        let driver = Driver::new(Real(state), profile.plan(), denominator, limit, timer)
            .map_err(|e| early_failure(run_id, T::WIDTH, scope, e))?;
        Ok((
            Self {
                profile,
                identity,
                run_id: run_id.into(),
                driver,
                scope,
            },
            captured,
        ))
    }
    fn failure(mut self, error: SessionError) -> SessionFailure {
        self.driver.fail("finalize", error);
        let f = self.driver.failed.unwrap();
        SessionFailure {
            detail: Box::new(FailureDetail {
                run_id: self.run_id,
                dtype: T::WIDTH,
                scope: self.scope,
                identity: Some(self.identity),
                progress: Some(self.driver.engine.progress()),
                last: self.driver.endpoints.pop(),
                timing: self.driver.failure_timing,
                phase: f.phase,
                error: f.error,
            }),
        }
    }
    fn record(
        mut self,
        full: bool,
        observation: PlainEvaluation,
        state: String,
    ) -> std::result::Result<Record, SessionFailure> {
        let consumed = match digests::consumed(
            self.profile.plan(),
            self.profile.identity().plan_sha256(),
            self.driver.expected.updates,
        ) {
            Ok(v) => v,
            Err(e) => return Err(self.failure(e)),
        };
        let timing = match self.driver.complete() {
            Ok(v) => v,
            Err(e) => return Err(self.failure(e)),
        };
        Ok(Record {
            run_id: self.run_id,
            identity: self.identity,
            dtype: T::WIDTH,
            full,
            plan: self.profile.plan().clone(),
            progress: self.driver.expected,
            endpoints: self.driver.endpoints,
            observation,
            state,
            consumed,
            full_hash: self.profile.identity().plan_sha256().into(),
            timing,
        })
    }
}
/// Canonical local source root for diagnostic input checks only. No build approval.
pub struct DiagnosticContext {
    root: PathBuf,
}
impl DiagnosticContext {
    #[doc = "Validate a canonical absolute local root for diagnostic input checks."]
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        require(
            root.is_absolute()
                && root
                    .canonicalize()
                    .map_err(|_| SessionError("diagnostic_root"))?
                    == root,
            "diagnostic_root",
        )?;
        Ok(Self {
            root: root.to_owned(),
        })
    }
    fn check(&self, p: &MaterializedProfile) -> Result<()> {
        require(
            p.identity().canonical_prepared(),
            "canonical_input_required",
        )?;
        for s in p.identity().sources() {
            let rel = Path::new(&s.path);
            require(
                !rel.is_absolute()
                    && rel
                        .components()
                        .all(|c| matches!(c, std::path::Component::Normal(_))),
                "input_source_path",
            )?;
            let path = self.root.join(rel);
            require(
                path.canonicalize()
                    .map_err(|_| SessionError("input_source_missing"))?
                    == path,
                "input_source_symlink",
            )?;
            let bytes = std::fs::read(path).map_err(|_| SessionError("input_source_missing"))?;
            require(
                s.expected_sha256 == s.actual_sha256 && sha256_hex(&bytes) == s.actual_sha256,
                "input_source_changed",
            )?;
        }
        Ok(())
    }
}
/// Failed execution, permanently incapable of becoming a completed reference.
#[derive(Debug)]
pub struct SessionFailure {
    detail: Box<FailureDetail>,
}
#[derive(Debug)]
pub(super) struct FailureDetail {
    pub(super) run_id: String,
    pub(super) dtype: u8,
    pub(super) scope: &'static str,
    pub(super) identity: Option<IdentityFact>,
    pub(super) progress: Option<SessionProgress>,
    pub(super) last: Option<EndpointFact>,
    pub(super) timing: Option<TimingFact>,
    pub(super) phase: &'static str,
    pub(super) error: SessionError,
}
impl SessionFailure {
    pub(super) fn detail(&self) -> &FailureDetail {
        &self.detail
    }
    #[doc = "Number of actual successfully completed updates."]
    pub fn completed_updates(&self) -> Option<usize> {
        self.detail.progress.map(|p| p.updates)
    }
    #[doc = "Stable bounded failure code."]
    pub fn error(&self) -> SessionError {
        self.detail.error
    }
    #[doc = "Serialize this immutable session result and reload its bounded schema for integrity."]
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        super::document::serialize_failure(self)
    }
}
impl std::fmt::Display for SessionFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail.phase, self.detail.error)
    }
}
impl std::error::Error for SessionFailure {}
fn early_failure(
    run_id: &str,
    dtype: u8,
    scope: &'static str,
    error: SessionError,
) -> SessionFailure {
    SessionFailure {
        detail: Box::new(FailureDetail {
            run_id: if valid_id(run_id).is_ok() {
                run_id.into()
            } else {
                "invalid_run_id".into()
            },
            dtype,
            scope,
            identity: None,
            progress: None,
            last: None,
            timing: None,
            phase: "start",
            error,
        }),
    }
}
/// Real local two-update result, permanently diagnostic and with no disk writer.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{LocalQuickDiagnostic,CompletedFp32Reference};
/// fn forge(d: LocalQuickDiagnostic) -> CompletedFp32Reference<'static> { d.into() }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::{LocalQuickDiagnostic,ReferenceStore};
/// fn publish(s: &ReferenceStore<'_>, d: &LocalQuickDiagnostic) { s.publish_reference(d); }
/// ```
#[derive(Debug)]
pub struct LocalQuickDiagnostic {
    record: Record,
}
impl LocalQuickDiagnostic {
    #[doc = "Actual internally evaluated correct prediction count."]
    pub fn correct(&self) -> u64 {
        self.record.observation.correct as u64
    }
    #[doc = "Actual held-out denominator, never a caller-supplied count."]
    pub fn test_count(&self) -> u64 {
        self.record.observation.test_count as u64
    }
    #[doc = "Serialize this immutable session result and reload its bounded schema for integrity."]
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        super::document::serialize_local_quick(self)
    }
    pub(super) fn record(&self) -> &Record {
        &self.record
    }
}
/// Real F64 two-update diagnostic; distinct from both F32 results and production completion.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{LocalQuickF64Diagnostic,CompletedFp32Reference};
/// fn forge(d: LocalQuickF64Diagnostic) -> CompletedFp32Reference<'static> { d.into() }
/// ```
#[derive(Debug)]
pub struct LocalQuickF64Diagnostic {
    record: Record,
}
impl LocalQuickF64Diagnostic {
    /// Actual internally evaluated correct prediction count.
    pub fn correct(&self) -> u64 {
        self.record.observation.correct as u64
    }
    /// Actual bound held-out denominator.
    pub fn test_count(&self) -> u64 {
        self.record.observation.test_count as u64
    }
    /// Serialize only an in-memory, unavailable-production-provenance diagnostic.
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        super::document::serialize_local_quick_f64(self)
    }
    pub(super) fn record(&self) -> &Record {
        &self.record
    }
}
macro_rules! quick_session {
    ($name:ident, $ty:ty, $result:ident) => {
        /// Exactly two real updates; a third call terminally fails before arithmetic.
        #[doc = concat!("```compile_fail\nuse ckks_train_bench::study_reference::", stringify!($name), ";\nfn forge(s: ", stringify!($name), "<'_>) { s.finish_full(); }\n```")]
        pub struct $name<'p> {
            session: Session<'p, $ty>,
            context: &'p DiagnosticContext,
        }
        impl<'p> $name<'p> {
            /// Start from fresh canonical inputs at update and sample origin zero.
            pub fn start(
                p: &'p MaterializedProfile,
                context: &'p DiagnosticContext,
                id: &str,
            ) -> std::result::Result<Self, SessionFailure> {
                Session::start_with(p, id, true, |_| context.check(p))
                    .map(|(session, ())| Self { session, context })
            }
            /// Consume exactly the next update; any error terminally fails the session.
            pub fn advance(&mut self) -> Result<SessionProgress> {
                self.session.driver.advance()
            }
            /// Consume exactly two successful updates as a permanently ineligible diagnostic.
            pub fn finish_quick(mut self) -> std::result::Result<$result, SessionFailure> {
                let result = (|| {
                    let observation = self.session.driver.prepare_finish(false)?;
                    let state = digests::state_digest(
                        self.session.profile,
                        &self.session.driver.engine.0,
                        self.session.identity.receipt.digest(),
                    )?;
                    self.context.check(self.session.profile)?;
                    Ok((observation, state))
                })();
                match result {
                    Ok((observation, state)) => self
                        .session
                        .record(false, observation, state)
                        .map(|record| $result { record }),
                    Err(e) => Err(self.session.failure(e)),
                }
            }
        }
    };
}
quick_session!(QuickFp32DiagnosticSession, f32, LocalQuickDiagnostic);
quick_session!(QuickFp64DiagnosticSession, f64, LocalQuickF64Diagnostic);

/// Completion can only be constructed by consuming an origin-to-full F32 session.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{ParsedReferenceDocument,CompletedFp32Reference};
/// fn forge(p: ParsedReferenceDocument) -> CompletedFp32Reference<'static> { p.into() }
/// ```
pub struct CompletedFp32Reference<'p> {
    record: Record,
    capture: CaptureFact<'p>,
    profile: &'p MaterializedProfile,
}
impl CompletedFp32Reference<'_> {
    #[doc = "Serialize this immutable session result and reload its bounded schema for integrity."]
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        super::document::serialize_reference(self)
    }
    pub(super) fn record(&self) -> &Record {
        &self.record
    }
    pub(super) fn capture(&self) -> &CaptureFact<'_> {
        &self.capture
    }
    pub(super) fn recheck(&self) -> Result<()> {
        self.capture
            .context
            .finish_capture(self.profile, &self.capture.start)
            .map(|_| ())
            .map_err(|_| SessionError("publication_provenance_changed"))
    }
}
/// Production-captured diagnostic. F64 and quick never yield scientific completion.
pub struct CapturedDiagnostic<'p> {
    record: Record,
    capture: CaptureFact<'p>,
    profile: &'p MaterializedProfile,
}
impl CapturedDiagnostic<'_> {
    #[doc = "Serialize this immutable session result and reload its bounded schema for integrity."]
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        super::document::serialize_diagnostic(self)
    }
    pub(super) fn record(&self) -> &Record {
        &self.record
    }
    pub(super) fn capture(&self) -> &CaptureFact<'_> {
        &self.capture
    }
    pub(super) fn recheck(&self) -> Result<()> {
        self.capture
            .context
            .finish_capture(self.profile, &self.capture.start)
            .map(|_| ())
            .map_err(|_| SessionError("publication_provenance_changed"))
    }
}
/// Production-captured F64 result, permanently separate from F32 diagnostics/completion.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{CapturedF64Diagnostic,CompletedFp32Reference};
/// fn forge(d: CapturedF64Diagnostic<'static>) -> CompletedFp32Reference<'static> { d.into() }
/// ```
pub struct CapturedF64Diagnostic<'p> {
    inner: CapturedDiagnostic<'p>,
}
impl CapturedF64Diagnostic<'_> {
    /// Serialize a permanently ineligible F64 diagnostic.
    pub fn document_bytes(&self) -> std::result::Result<Vec<u8>, super::DocumentError> {
        self.inner.document_bytes()
    }
    pub(super) fn inner(&self) -> &CapturedDiagnostic<'_> {
        &self.inner
    }
}
struct Production<'p, T: Scalar> {
    session: Session<'p, T>,
    context: &'p VerifiedStudyContext,
    capture: RuntimeCapture,
}
impl<'p, T: Scalar> Production<'p, T> {
    fn start(
        p: &'p MaterializedProfile,
        context: &'p VerifiedStudyContext,
        id: &str,
    ) -> std::result::Result<Self, SessionFailure> {
        let (session, capture) = Session::start_with(p, id, false, |identity| {
            let captured = context
                .capture(p)
                .map_err(|_| SessionError("production_capture_failed"))?;
            require(
                captured.contract_sha256() == identity.contract.digest()
                    && captured.input_receipt_sha256() == identity.receipt.digest()
                    && captured.materialized_sha256() == p.identity().digest(),
                "capture_identity",
            )?;
            Ok(captured)
        })?;
        Ok(Self {
            session,
            context,
            capture,
        })
    }

    fn finish(
        mut self,
        full: bool,
    ) -> std::result::Result<(Record, CaptureFact<'p>, &'p MaterializedProfile), SessionFailure>
    {
        if !full {
            self.session.scope = "quick";
        }
        let result = (|| {
            let observation = self.session.driver.prepare_finish(full)?;
            let state = digests::state_digest(
                self.session.profile,
                &self.session.driver.engine.0,
                self.session.identity.receipt.digest(),
            )?;
            let end = self
                .context
                .finish_capture(self.session.profile, &self.capture)
                .map_err(|_| SessionError("final_provenance_changed"))?;
            Ok((observation, state, end))
        })();
        match result {
            Ok((observation, state, end)) => {
                let p = self.session.profile;
                let record = self.session.record(full, observation, state)?;
                Ok((
                    record,
                    CaptureFact {
                        context: self.context,
                        start: self.capture,
                        end,
                    },
                    p,
                ))
            }
            Err(e) => Err(self.session.failure(e)),
        }
    }
}
/// Approved F32 session owns its engine; imported cursors/evaluations are impossible.
/// ```compile_fail
/// use ckks_train_bench::study_reference::Fp32Session;
/// use ckks_train_bench::study_plain::PlainEvaluation;
/// fn forge(s: Fp32Session<'_>, e: PlainEvaluation) { s.finish_full(e); }
/// ```
/// ```compile_fail
/// use ckks_train_bench::study_reference::Fp32Session;
/// fn forge(mut s: Fp32Session<'_>) { s.set_cursor(1); }
/// ```
pub struct Fp32Session<'p> {
    inner: Production<'p, f32>,
}
impl<'p> Fp32Session<'p> {
    #[doc = "Start from fresh canonical inputs at update and sample origin zero."]
    pub fn start(
        p: &'p MaterializedProfile,
        c: &'p VerifiedStudyContext,
        id: &str,
    ) -> std::result::Result<Self, SessionFailure> {
        Production::start(p, c, id).map(|inner| Self { inner })
    }
    #[doc = "Consume exactly the next planned update; any error terminally fails this session."]
    pub fn advance(&mut self) -> Result<SessionProgress> {
        self.inner.session.driver.advance()
    }
    #[doc = "Consume complete sequential coverage and all checkpoints; F64 remains diagnostic."]
    pub fn finish_full(self) -> std::result::Result<CompletedFp32Reference<'p>, SessionFailure> {
        self.inner
            .finish(true)
            .map(|(record, capture, profile)| CompletedFp32Reference {
                record,
                capture,
                profile,
            })
    }
    #[doc = "Consume exactly two successful updates as a permanently ineligible diagnostic."]
    pub fn finish_quick(self) -> std::result::Result<CapturedDiagnostic<'p>, SessionFailure> {
        self.inner
            .finish(false)
            .map(|(record, capture, profile)| CapturedDiagnostic {
                record,
                capture,
                profile,
            })
    }
}
/// Even full-length F64 remains diagnostic, without a conversion to F32 completion.
/// ```compile_fail
/// use ckks_train_bench::study_reference::{CapturedDiagnostic,CompletedFp32Reference};
/// fn forge(d: CapturedDiagnostic<'static>) -> CompletedFp32Reference<'static> { d.into() }
/// ```
pub struct F64DiagnosticSession<'p> {
    inner: Production<'p, f64>,
}
impl<'p> F64DiagnosticSession<'p> {
    #[doc = "Start from fresh canonical inputs at update and sample origin zero."]
    pub fn start(
        p: &'p MaterializedProfile,
        c: &'p VerifiedStudyContext,
        id: &str,
    ) -> std::result::Result<Self, SessionFailure> {
        Production::start(p, c, id).map(|inner| Self { inner })
    }
    #[doc = "Consume exactly the next planned update; any error terminally fails this session."]
    pub fn advance(&mut self) -> Result<SessionProgress> {
        self.inner.session.driver.advance()
    }
    #[doc = "Consume complete sequential coverage and all checkpoints; F64 remains diagnostic."]
    pub fn finish_full(self) -> std::result::Result<CapturedF64Diagnostic<'p>, SessionFailure> {
        self.inner
            .finish(true)
            .map(|(record, capture, profile)| CapturedF64Diagnostic {
                inner: CapturedDiagnostic {
                    record,
                    capture,
                    profile,
                },
            })
    }
    #[doc = "Consume exactly two successful updates as a permanently ineligible diagnostic."]
    pub fn finish_quick(self) -> std::result::Result<CapturedF64Diagnostic<'p>, SessionFailure> {
        self.inner
            .finish(false)
            .map(|(record, capture, profile)| CapturedF64Diagnostic {
                inner: CapturedDiagnostic {
                    record,
                    capture,
                    profile,
                },
            })
    }
}
#[cfg(test)]
#[path = "session_tests.rs"]
mod tests;
