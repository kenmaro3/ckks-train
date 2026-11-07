//! Client-owned encryption and explicit diagnostic endpoints for the CPU bridge.
//! Canonical preparation revalidates trusted materialization; it is not a D7 start gate.
use crate::{
    profile_data::padded_targets,
    study_admission::{self, DiagonalPlan, DiagonalRequest, MicroPart, PlanningParameters},
    study_diagonal::{
        ClientRefreshService, CpuBootstrap, CpuEvaluationContext, Description,
        EncryptedInitialState, EncryptedLogicalUpdate, EncryptedMicro, EncryptedModelSnapshot,
        PlanOwner, activations, ensure, mismatch,
    },
    study_inputs::MaterializedProfile,
    study_profile::{Activation as ProfileActivation, RegisteredCase},
};
use ckks_train_ckks::{
    CkksParams, Decryptor, Encoder, Encryptor, EvalConfig, Evaluator, HeOps, KeyGenerator,
    bootstrap::{BootParams, Bootstrapper, metadata::BootstrapMetadata},
};
use ckks_train_core::{error::Result, sampler::Rng};
use ckks_train_nn::{
    diagonal_plan::{DiagonalActivation, DiagonalMode},
    layout::unpack_activations,
    refresh::ClientRefresher,
    trainer::{BatchSpec, Mlp},
    trainer_ni::NonInteractiveTrainer,
};
use ndarray::{Array2, s};
use std::{cell::RefCell, sync::Arc};

/// Preparation failure retaining the exact canonical admission/candidate diagnostic.
#[derive(Debug)]
pub enum CpuPreparationError {
    /// Existing binding error, including typed budget/partial-receipt details.
    Admission(Box<study_admission::AdmissionIssue>),
    /// Existing candidate description, retaining its exact borrowed issue without cloning it.
    Candidate(Box<study_admission::CandidateDescription>),
    /// Bridge association, shape, encryption or evaluation error.
    He(ckks_train_core::error::HekitError),
}
impl CpuPreparationError {
    /// Exact P3 issue, including configured/required work budgets when present.
    pub fn admission_issue(&self) -> Option<&study_admission::AdmissionIssue> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Candidate(c) => c.issue(),
            Self::He(_) => None,
        }
    }
}
impl std::fmt::Display for CpuPreparationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admission(e) => e.fmt(f),
            Self::Candidate(c) => match c.issue() {
                Some(e) => e.fmt(f),
                None => write!(f, "candidate preparation failed without issue"),
            },
            Self::He(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for CpuPreparationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(e) => Some(e),
            Self::Candidate(c) => c.issue().map(|e| e as _),
            Self::He(e) => Some(e),
        }
    }
}
impl From<ckks_train_core::error::HekitError> for CpuPreparationError {
    fn from(e: ckks_train_core::error::HekitError) -> Self {
        Self::He(e)
    }
}
#[cfg(test)]
impl PartialEq<ckks_train_core::error::HekitError> for CpuPreparationError {
    fn eq(&self, other: &ckks_train_core::error::HekitError) -> bool {
        matches!(self,Self::He(e) if e==other)
    }
}
/// Result of checked canonical/synthetic preparation; never a start/admission capability.
pub type CpuPreparationResult<T> = std::result::Result<T, CpuPreparationError>;

pub(crate) enum PreparationInput<'a> {
    Canonical {
        materialized: &'a MaterializedProfile,
        registered: &'a RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &'a DiagonalRequest,
    },
    #[cfg(test)]
    Synthetic {
        plan: Arc<crate::study_admission::p4_fixtures::SyntheticCpuPlan>,
        claim: &'a crate::study_admission::p4_fixtures::SyntheticClaim,
    },
}
pub(crate) struct CheckedPreparation<'a> {
    pub(crate) plan: PlanOwner,
    pub(crate) description: Description,
    parameters: &'a CkksParams,
    bootstrap_identity: Option<&'a [u8]>,
    bootstrap_metadata: Option<&'a BootstrapMetadata>,
    boot_parameters: Option<&'a BootParams>,
}
// Immutable source projections. Only checked canonical preparation can provide this view.
// A synthetic plan is deliberately refused even when its bytes happen to match.
#[allow(dead_code)] // The next retained-route adapter consumes these non-authorizing views.
pub(crate) struct PreparationDescriptorFacts<'a> {
    plan: &'a DiagonalPlan,
    descriptions: &'a [Description],
    parameters: &'a CkksParams,
    bootstrap: Option<&'a BootstrapMetadata>,
}
#[allow(dead_code)]
impl<'a> PreparationDescriptorFacts<'a> {
    pub(crate) fn plan(&self) -> &'a DiagonalPlan {
        self.plan
    }
    pub(crate) fn parameters(&self) -> &'a CkksParams {
        self.parameters
    }
    pub(crate) fn bootstrap(&self) -> Option<&'a BootstrapMetadata> {
        self.bootstrap
    }
    pub(crate) fn stage_count(&self) -> usize {
        self.descriptions.len()
    }
    pub(crate) fn geometry(&self, index: usize) -> &'a ckks_train_nn::diagonal_plan::StageGeometry {
        &self.descriptions[index].geometry
    }
    pub(crate) fn transpose(&self) -> &'a ckks_train_nn::dense::TransposeKeys {
        &self.descriptions[0].transpose
    }
}
impl CheckedPreparation<'_> {
    #[allow(dead_code)]
    pub(crate) fn descriptor_facts(&self) -> Option<PreparationDescriptorFacts<'_>> {
        let plan = match &self.plan {
            PlanOwner::Canonical(p) => p.as_ref(),
            #[cfg(test)]
            PlanOwner::Synthetic(_) => return None,
        };
        Some(PreparationDescriptorFacts {
            plan,
            descriptions: std::slice::from_ref(&self.description),
            parameters: self.parameters,
            bootstrap: self.bootstrap_metadata,
        })
    }
}
// A single private validation-then-factory invocation, shared by real setup and terminal spy.
// No callback or digest capability is exported by the public client API.
pub(crate) fn prepare_then_make<'a, T>(
    input: PreparationInput<'_>,
    actual: PlanningParameters<'a>,
    actual_partition: &[Vec<MicroPart>],
    factory: impl FnOnce(CheckedPreparation<'a>) -> Result<T>,
) -> CpuPreparationResult<T> {
    let checked = match input {
        PreparationInput::Canonical {
            materialized,
            registered,
            plan,
            request,
        } => {
            let owner = PlanOwner::Canonical(plan.clone());
            owner.check_shape()?;
            study_admission::validate_plan_binding(
                materialized,
                registered,
                &plan,
                actual,
                request,
            )
            .map_err(|e| CpuPreparationError::Admission(Box::new(e)))?;
            ensure(actual_partition == plan.partitions(), "partition")?;
            // Reserve geometry/initial allowance in the original request BEFORE hashing.
            let candidate = study_admission::describe_candidate(
                materialized.metadata(),
                registered,
                actual,
                request,
            );
            if candidate.issue().is_some() {
                return Err(CpuPreparationError::Candidate(Box::new(candidate)));
            }
            ensure(
                candidate.geometries().len() == 1,
                "missing block-stage execution capability",
            )?;
            let keys = candidate
                .keys()
                .ok_or_else(|| mismatch("candidate key manifest"))?;
            ensure(
                keys.training_rotations() == plan.keys().training_rotations()
                    && keys.bootstrap_generated_rotations()
                        == plan.keys().bootstrap_generated_rotations()
                    && keys.bootstrap_execution_rotations()
                        == plan.keys().bootstrap_execution_rotations()
                    && keys.bootstrap_conjugation() == plan.keys().bootstrap_conjugation()
                    && keys.bootstrap_sse() == plan.keys().bootstrap_sse(),
                "candidate/plan key manifest",
            )?;
            let activations = materialized
                .metadata()
                .layers
                .iter()
                .map(|l| match l.activation {
                    ProfileActivation::Square => DiagonalActivation::Square,
                    ProfileActivation::SquarePlusLinear => DiagonalActivation::PolyRelu2,
                    ProfileActivation::Identity => DiagonalActivation::Identity,
                })
                .collect();
            let frozen = materialized.metadata().layers.iter().all(|l| !l.bias);
            ensure(
                frozen || materialized.metadata().layers.iter().all(|l| l.bias),
                "mixed bias policy unsupported",
            )?;
            // Clone only this one geometry, never the candidate's placeholder initial states.
            CheckedPreparation {
                parameters: actual.params,
                bootstrap_identity: actual.bootstrap.map(|b| b.descriptor_identity_bytes()),
                bootstrap_metadata: actual.bootstrap,
                boot_parameters: actual.boot_params,
                plan: owner,
                description: Description {
                    geometry: candidate.geometries()[0].clone(),
                    activations,
                    transpose: request.transpose.clone(),
                    frozen,
                },
            }
        }
        #[cfg(test)]
        PreparationInput::Synthetic { plan, claim } => {
            plan.validate(actual, actual_partition, claim)
                .map_err(mismatch)?;
            let owner = PlanOwner::Synthetic(plan.clone());
            owner.check_shape()?;
            let activations = if plan.frozen {
                vec![DiagonalActivation::PolyRelu2, DiagonalActivation::Identity]
            } else {
                vec![DiagonalActivation::Square; 2]
            };
            CheckedPreparation {
                parameters: actual.params,
                bootstrap_identity: actual.bootstrap.map(|b| b.descriptor_identity_bytes()),
                bootstrap_metadata: actual.bootstrap,
                boot_parameters: actual.boot_params,
                plan: owner,
                description: Description {
                    geometry: plan.geometry.clone(),
                    activations,
                    transpose: plan.transpose.clone(),
                    frozen: plan.frozen,
                },
            }
        }
    };
    factory(checked).map_err(CpuPreparationError::He)
}
/// Secret-bearing client; sessions never receive this value or a materialized profile.
pub struct StudyClient {
    owner: Arc<()>,
    encrypt: Encryptor,
    decrypt: Decryptor,
    encoder: Encoder,
    rng: RefCell<Rng>,
    prepared: RefCell<Vec<(PlanOwner, ckks_train_nn::layout::Layout)>>,
}
/// Plaintext arrays returned only through an explicit client diagnostic endpoint.
pub struct ClientModelSnapshot {
    layers: Vec<(Array2<f64>, Vec<f64>)>,
}
impl ClientModelSnapshot {
    /// Every logical weight and bias array, in bound layer order.
    pub fn layers(&self) -> &[(Array2<f64>, Vec<f64>)] {
        &self.layers
    }
}
/// Caller-owned NI setup values; the descriptor alone is not precision/security admission.
#[derive(Clone)]
pub struct CpuBootstrapDescription {
    /// Exact parameters used by the concrete bootstrapper.
    pub parameters: BootParams,
    /// Existing immutable P1 structural descriptor.
    pub metadata: Arc<BootstrapMetadata>,
}
impl StudyClient {
    /// Associates an already-created caller key owner after canonical binding checks.
    /// This path cannot claim rejection preceded the caller's external key allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_key_generator(
        materialized: &MaterializedProfile,
        registered: &RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &DiagonalRequest,
        mut keys: KeyGenerator,
        boot: Option<CpuBootstrapDescription>,
        rng: Rng,
    ) -> CpuPreparationResult<(Self, CpuEvaluationContext, EncryptedInitialState)> {
        let params = keys.params().clone();
        let bound_boot = boot.clone();
        let actual = PlanningParameters {
            params: &params,
            bootstrap: bound_boot.as_ref().map(|b| b.metadata.as_ref()),
            boot_params: bound_boot.as_ref().map(|b| &b.parameters),
        };
        let parts = plan.partitions();
        prepare_then_make(
            PreparationInput::Canonical {
                materialized,
                registered,
                plan: plan.clone(),
                request,
            },
            actual,
            parts,
            |checked| {
                let (client, context) = Self::make_material(
                    &mut keys,
                    checked.description.geometry.rotation_steps(),
                    request.eval_config,
                    boot,
                    rng,
                    #[cfg(test)]
                    None,
                )?;
                let arrays = materialized
                    .initial()
                    .layers()
                    .iter()
                    .map(|l| (l.weights().clone(), l.biases().to_vec()))
                    .collect::<Vec<_>>();
                let initial = client.encrypt_initial(checked, &context, &arrays)?;
                Ok((client, context, initial))
            },
        )
    }
    pub(crate) fn make_material(
        keys: &mut KeyGenerator,
        rotations: &[i32],
        config: EvalConfig,
        boot: Option<CpuBootstrapDescription>,
        rng: Rng,
        #[cfg(test)] budget: Option<std::rc::Rc<crate::study_diagonal::p4_tests::RuntimeBudget>>,
    ) -> Result<(Self, CpuEvaluationContext)> {
        let params = keys.params().clone();
        #[cfg(test)]
        crate::study_diagonal::p4_tests::reserve_encoder();
        #[cfg(test)]
        crate::study_blocks::p5_tests::reserve_encoder();
        let encoder = Encoder::new(params.clone());
        let (bootstrap, metadata, boot_params) = if let Some(b) = boot {
            #[cfg(test)]
            if let Some(budget) = &budget {
                budget.reserve_bootstrapper();
            }
            #[cfg(test)]
            crate::study_blocks::p5_tests::reserve_bootstrapper();
            let real = Arc::new(Bootstrapper::new(
                params.clone(),
                b.parameters.clone(),
                &encoder,
                keys,
            )?);
            (
                Some(CpuBootstrap {
                    real,
                    #[cfg(test)]
                    budget,
                    #[cfg(test)]
                    fail_after: None,
                    #[cfg(test)]
                    successful: std::cell::Cell::new(0),
                }),
                Some(b.metadata),
                Some(b.parameters),
            )
        } else {
            (None, None, None)
        };
        let encrypt = Encryptor::new(params.clone(), keys.public_key());
        let decrypt = Decryptor::new(params.clone(), keys.secret_key());
        let evaluator = Evaluator::with_config(
            params,
            Some(keys.relin_key()),
            Some(keys.galois_keys(rotations)),
            config,
        );
        let owner = Arc::new(());
        let client = Self {
            owner: owner.clone(),
            encrypt,
            decrypt,
            encoder: encoder.clone(),
            rng: RefCell::new(rng),
            prepared: RefCell::new(vec![]),
        };
        let context = CpuEvaluationContext {
            owner,
            evaluator,
            encoder,
            bootstrap,
            metadata,
            boot_params,
        };
        Ok((client, context))
    }
    /// Revalidates canonical trusted inputs and encrypts their exact initial arrays.
    pub fn prepare_initial(
        &self,
        materialized: &MaterializedProfile,
        registered: &RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &DiagonalRequest,
        context: &CpuEvaluationContext,
    ) -> CpuPreparationResult<EncryptedInitialState> {
        let checked = prepare_then_make(
            PreparationInput::Canonical {
                materialized,
                registered,
                plan: plan.clone(),
                request,
            },
            context_parameters(context),
            plan.partitions(),
            Ok,
        )?;
        let arrays = materialized
            .initial()
            .layers()
            .iter()
            .map(|l| (l.weights().clone(), l.biases().to_vec()))
            .collect::<Vec<_>>();
        self.encrypt_initial(checked, context, &arrays)
            .map_err(CpuPreparationError::He)
    }
    pub(crate) fn encrypt_initial(
        &self,
        checked: CheckedPreparation<'_>,
        context: &CpuEvaluationContext,
        arrays: &[(Array2<f64>, Vec<f64>)],
    ) -> Result<EncryptedInitialState> {
        self.check_context(context)?;
        ensure(
            std::ptr::eq(checked.parameters, context.parameters().as_ref()),
            "initial actual parameters",
        )?;
        context.check_geometry(&checked.description.geometry)?;
        let initial = &checked.plan.initial()[0];
        let layout = checked.description.geometry.layout();
        let before = context.counter();
        let diagnostic = context.evaluator.fork();
        let mut rng = self.rng.borrow_mut();
        let (mlp, compact) = match checked.description.geometry.mode() {
            DiagonalMode::Interactive => (
                Mlp::encrypt_at(
                    &diagnostic,
                    arrays,
                    activations(&checked.description.activations),
                    &layout,
                    initial.batch().x.level(),
                    &self.encrypt,
                    &self.encoder,
                    &mut rng,
                )?,
                None,
            ),
            DiagonalMode::NonInteractive(placement) => {
                let (m, c) = NonInteractiveTrainer::encrypt_network(
                    &diagnostic,
                    context
                        .bootstrap
                        .as_ref()
                        .ok_or_else(|| mismatch("bootstrap dependency"))?,
                    arrays,
                    activations(&checked.description.activations),
                    &layout,
                    placement,
                    &self.encrypt,
                    &self.encoder,
                    &mut rng,
                )?;
                (m, Some(c))
            }
        };
        ensure(
            context.counter() == before,
            "initial probe counter isolation",
        )?;
        let m = initial.batch();
        let batch = BatchSpec {
            x_level: m.x.level(),
            x_scale: m.x.scale(),
            y_level: m.y.level(),
            y_scale: m.y.scale(),
        };
        self.prepared
            .borrow_mut()
            .push((checked.plan.clone(), layout));
        Ok(EncryptedInitialState {
            owner: self.owner.clone(),
            plan: checked.plan,
            description: checked.description,
            mlp,
            compact,
            batch,
        })
    }
    /// Gathers the materializer's exact logical order, then slices it into bound micros.
    /// C/P-aware padded targets come from the existing helper; no loss fold is repeated.
    pub fn encrypt_update(
        &self,
        materialized: &MaterializedProfile,
        registered: &RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &DiagonalRequest,
        index: usize,
        context: &CpuEvaluationContext,
    ) -> CpuPreparationResult<EncryptedLogicalUpdate> {
        // Binding revalidation does not reconstruct geometry or rerun the schedule here.
        study_admission::validate_plan_binding(
            materialized,
            registered,
            &plan,
            context_parameters(context),
            request,
        )
        .map_err(|e| CpuPreparationError::Admission(Box::new(e)))?;
        let owner = PlanOwner::Canonical(plan);
        owner.check_shape()?;
        let rows = materialized
            .update_rows(index)
            .map_err(|e| mismatch(&format!("update rows: {e}")))?;
        let numeric = materialized.numeric_f64();
        let x = Array2::from_shape_fn((rows.rows().len(), numeric.train_x().ncols()), |(r, j)| {
            numeric.train_x()[[rows.rows()[r] as usize, j]]
        });
        let labels = rows
            .rows()
            .iter()
            .map(|&r| numeric.train_y()[r as usize])
            .collect::<Vec<_>>();
        self.encrypt_rows(
            owner,
            index,
            context,
            &x,
            &labels,
            materialized.metadata().classes,
            materialized.metadata().output_width,
        )
        .map_err(CpuPreparationError::He)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encrypt_rows(
        &self,
        plan: PlanOwner,
        index: usize,
        context: &CpuEvaluationContext,
        x: &Array2<f64>,
        labels: &[u8],
        classes: usize,
        width: usize,
    ) -> Result<EncryptedLogicalUpdate> {
        self.check_context(context)?;
        plan.check_shape()?;
        let parts = plan
            .parts()
            .get(index)
            .ok_or_else(|| mismatch("client update index"))?;
        let update = plan.transitions()[index]
            .update_metadata()
            .ok_or_else(|| mismatch("client transition"))?;
        let total = parts
            .iter()
            .try_fold(0usize, |s, p| s.checked_add(p.valid))
            .ok_or_else(|| mismatch("client row count overflow"))?;
        ensure(
            x.nrows() == total && labels.len() == total,
            "client label/row cardinality",
        )?;
        let targets = padded_targets(labels, classes, width)
            .map_err(|_| mismatch("client labels/target shape"))?;
        let layout = self
            .prepared
            .borrow()
            .iter()
            .find(|(p, _)| p.same(&plan))
            .map(|(_, layout)| *layout)
            .ok_or_else(|| mismatch("client prepared layout dependency"))?;
        let mut micros = Vec::with_capacity(parts.len());
        let mut rng = self.rng.borrow_mut();
        ensure(
            parts.len() == update.micros().len(),
            "client receipt micros",
        )?;
        for (part, m) in parts.iter().zip(update.micros()) {
            let end = part
                .start
                .checked_add(part.valid)
                .ok_or_else(|| mismatch("client partition overflow"))?;
            ensure(end <= total, "client ordered partition bounds")?;
            let spec = BatchSpec {
                x_level: m.x.level(),
                x_scale: m.x.scale(),
                y_level: m.y.level(),
                y_scale: m.y.scale(),
            };
            let (a, b) = spec.encrypt(
                &self.encrypt,
                &self.encoder,
                &layout,
                &x.slice(s![part.start..end, ..]).to_owned(),
                &targets.slice(s![part.start..end, ..]).to_owned(),
                &mut rng,
            )?;
            micros.push(EncryptedMicro {
                part: *part,
                x: a,
                targets: vec![b],
            });
        }
        Ok(EncryptedLogicalUpdate {
            owner: self.owner.clone(),
            plan,
            index,
            total,
            input_width: x.ncols(),
            target_width: width,
            micros,
        })
    }
    /// Creates a secret-bearing service that stays client-owned; sessions borrow its handle.
    pub fn refresh_service(&self, rng: Rng) -> ClientRefreshService {
        ClientRefreshService {
            owner: self.owner.clone(),
            inner: RefCell::new(ClientRefresher::new(
                self.decrypt.clone(),
                self.encrypt.clone(),
                self.encoder.clone(),
                rng,
            )),
            #[cfg(test)]
            fail_after: None,
            #[cfg(test)]
            successful: std::cell::Cell::new(0),
        }
    }
    /// Decrypts an explicitly exported encrypted snapshot in the client role.
    /// All logical W/b entries must be finite; training counters remain untouched.
    pub fn inspect_snapshot(
        &self,
        snapshot: &EncryptedModelSnapshot,
        context: &CpuEvaluationContext,
    ) -> Result<ClientModelSnapshot> {
        self.check_context(context)?;
        ensure(
            Arc::ptr_eq(&self.owner, &snapshot.owner),
            "snapshot context identity",
        )?;
        let before = context.counter();
        let _diagnostic = context.evaluator.fork();
        let layers = snapshot
            .layers
            .iter()
            .map(|d| {
                let w = d.w.decrypt(&self.decrypt, &self.encoder, &snapshot.layout);
                let b = unpack_activations(
                    &self.encoder.decode(&self.decrypt.decrypt(&d.b)),
                    &snapshot.layout,
                    d.w.d_out(),
                )
                .row(0)
                .to_vec();
                ensure(
                    w.iter().chain(b.iter()).all(|x| x.is_finite()),
                    "nonfinite client snapshot",
                )?;
                Ok((w, b))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure(context.counter() == before, "snapshot counter isolation")?;
        Ok(ClientModelSnapshot { layers })
    }
    fn check_context(&self, context: &CpuEvaluationContext) -> Result<()> {
        ensure(
            Arc::ptr_eq(&self.owner, &context.owner)
                && Arc::ptr_eq(self.encoder.params(), context.parameters()),
            "client context identity",
        )
    }
}
pub(crate) fn context_parameters(c: &CpuEvaluationContext) -> PlanningParameters<'_> {
    PlanningParameters {
        params: c.parameters(),
        bootstrap: c.metadata.as_deref(),
        boot_params: c.boot_params.as_ref(),
    }
}

// Reserved private owned factory, intentionally disconnected from run_owned until D7
// supplies its genuine owned gates/start frame. Synthetic tests reach this exact factory.
#[allow(dead_code)]
pub(crate) fn make_owned(
    checked: &CheckedPreparation<'_>,
    parameters: Arc<CkksParams>,
    rotations: &[i32],
    boot: Option<CpuBootstrapDescription>,
    key_rng: Rng,
    encryption_rng: Rng,
    #[cfg(test)] budget: Option<std::rc::Rc<crate::study_diagonal::p4_tests::RuntimeBudget>>,
) -> Result<(StudyClient, CpuEvaluationContext)> {
    make_owned_fields(
        SetupFields {
            parameters: checked.parameters,
            bootstrap_identity: checked.bootstrap_identity,
            boot_parameters: checked.boot_parameters,
            rotations: checked.description.geometry.rotation_steps(),
            config: checked.description.geometry.eval_config(),
        },
        parameters,
        rotations,
        boot,
        key_rng,
        encryption_rng,
        #[cfg(test)]
        budget,
    )
}
/// Borrowed validated setup fields; cannot be publicly constructed or used as a permit.
pub(crate) struct SetupFields<'a> {
    pub(crate) parameters: &'a CkksParams,
    pub(crate) bootstrap_identity: Option<&'a [u8]>,
    pub(crate) boot_parameters: Option<&'a BootParams>,
    pub(crate) rotations: &'a [i32],
    pub(crate) config: EvalConfig,
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_owned_fields(
    checked: SetupFields<'_>,
    parameters: Arc<CkksParams>,
    rotations: &[i32],
    boot: Option<CpuBootstrapDescription>,
    key_rng: Rng,
    encryption_rng: Rng,
    #[cfg(test)] budget: Option<std::rc::Rc<crate::study_diagonal::p4_tests::RuntimeBudget>>,
) -> Result<(StudyClient, CpuEvaluationContext)> {
    ensure(
        std::ptr::eq(checked.parameters, parameters.as_ref()),
        "owned actual parameters",
    )?;
    ensure(
        checked.bootstrap_identity
            == boot
                .as_ref()
                .map(|b| b.metadata.descriptor_identity_bytes())
            && checked.boot_parameters == boot.as_ref().map(|b| &b.parameters),
        "owned bootstrap identity",
    )?;
    ensure(
        checked.rotations.iter().all(|s| rotations.contains(s)),
        "owned union key manifest",
    )?;
    #[cfg(test)]
    if let Some(budget) = &budget {
        budget.reserve_key();
    }
    // First secret/key action. Validation has already consumed the closed input in
    // prepare_then_make; the checked frame cannot be publicly constructed.
    #[cfg(test)]
    crate::study_blocks::p5_tests::reserve_key();
    let mut keys = KeyGenerator::new(parameters, key_rng)?;
    StudyClient::make_material(
        &mut keys,
        rotations,
        checked.config,
        boot,
        encryption_rng,
        #[cfg(test)]
        budget,
    )
}

// Compile-visible canonical owned continuation. It cannot be called by public code or
// dispatch and grants no start authority; D7 must supply the eventual owned callsite.
#[allow(dead_code, clippy::too_many_arguments)]
fn prepare_owned_canonical(
    materialized: &MaterializedProfile,
    registered: &RegisteredCase,
    plan: Arc<DiagonalPlan>,
    request: &DiagonalRequest,
    parameters: Arc<CkksParams>,
    boot: Option<CpuBootstrapDescription>,
    key_rng: Rng,
    encryption_rng: Rng,
) -> CpuPreparationResult<(StudyClient, CpuEvaluationContext, EncryptedInitialState)> {
    let bound = boot.clone();
    let actual = PlanningParameters {
        params: &parameters,
        bootstrap: bound.as_ref().map(|b| b.metadata.as_ref()),
        boot_params: bound.as_ref().map(|b| &b.parameters),
    };
    prepare_then_make(
        PreparationInput::Canonical {
            materialized,
            registered,
            plan: plan.clone(),
            request,
        },
        actual,
        plan.partitions(),
        |checked| {
            let rotations = checked.description.geometry.rotation_steps().to_vec();
            let (client, context) = make_owned(
                &checked,
                parameters.clone(),
                &rotations,
                boot,
                key_rng,
                encryption_rng,
                #[cfg(test)]
                None,
            )?;
            let arrays = materialized
                .initial()
                .layers()
                .iter()
                .map(|l| (l.weights().clone(), l.biases().to_vec()))
                .collect::<Vec<_>>();
            let initial = client.encrypt_initial(checked, &context, &arrays)?;
            Ok((client, context, initial))
        },
    )
}

/// Closed block preparation and client endpoints, never D7 authority.
pub(crate) mod blocks;
