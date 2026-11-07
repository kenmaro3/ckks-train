//! Block endpoints in the existing secret-bearing client's child module.
use super::*;
use crate::study_blocks::{
    BlockPlanOwner, EncryptedBlockInitial, EncryptedBlockSnapshot, EncryptedBlockUpdate,
};
use ckks_train_nn::{block::BlockMlp, layout::pack_activations};

pub(crate) enum BlockPreparationInput<'a> {
    Canonical {
        materialized: &'a MaterializedProfile,
        registered: &'a RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &'a DiagonalRequest,
    },
    #[cfg(test)]
    Synthetic {
        plan: Arc<crate::study_admission::p5_fixtures::SyntheticBlockPlan>,
        claim: &'a crate::study_admission::p5_fixtures::SyntheticBlockClaim,
    },
}
pub(crate) struct CheckedBlockPreparation<'a> {
    pub(crate) plan: BlockPlanOwner,
    descriptions: Vec<Description>,
    classes: Vec<usize>,
    parameters: &'a CkksParams,
    bootstrap_identity: Option<&'a [u8]>,
    boot_parameters: Option<&'a BootParams>,
    rotations: Vec<i32>,
}
/// Closed validation before the first key-material factory; never an admission capability.
pub(crate) fn prepare_blocks_then_make<'a, T>(
    input: BlockPreparationInput<'_>,
    actual: PlanningParameters<'a>,
    parts: &[Vec<MicroPart>],
    factory: impl FnOnce(CheckedBlockPreparation<'a>) -> Result<T>,
) -> CpuPreparationResult<T> {
    let (plan, descriptions, classes, rotations) = match input {
        BlockPreparationInput::Canonical {
            materialized,
            registered,
            plan,
            request,
        } => {
            let owner = BlockPlanOwner::Canonical(plan.clone());
            owner.check_shape()?;
            study_admission::validate_plan_binding(
                materialized,
                registered,
                &plan,
                actual,
                request,
            )
            .map_err(|e| CpuPreparationError::Admission(Box::new(e)))?;
            ensure(parts == plan.partitions(), "block partition")?;
            // Original request only: reserve the additional S geometry/initial allowance
            // before its hash. No cloned replacement budget or post-binding mutation.
            let candidate = study_admission::describe_candidate(
                materialized.metadata(),
                registered,
                actual,
                request,
            );
            if candidate.issue().is_some() {
                return Err(CpuPreparationError::Candidate(Box::new(candidate)));
            }
            let graph = materialized.graph().stages();
            ensure(
                candidate.geometries().len() == graph.len() && graph.len() == owner.initial().len(),
                "block candidate stages",
            )?;
            let keys = candidate
                .keys()
                .ok_or_else(|| mismatch("block candidate manifest"))?;
            ensure(
                keys.training_rotations() == plan.keys().training_rotations()
                    && keys.bootstrap_generated_rotations()
                        == plan.keys().bootstrap_generated_rotations()
                    && keys.bootstrap_execution_rotations()
                        == plan.keys().bootstrap_execution_rotations()
                    && keys.bootstrap_conjugation() == plan.keys().bootstrap_conjugation()
                    && keys.bootstrap_sse() == plan.keys().bootstrap_sse(),
                "block candidate/plan manifest",
            )?;
            let mut descriptions = Vec::with_capacity(graph.len());
            for (stage, (topology, geometry)) in
                graph.iter().zip(candidate.geometries()).enumerate()
            {
                let range = topology.layers();
                let specs = &materialized.metadata().layers[range.clone()];
                ensure(
                    specs.len() == 2
                        && topology.representation_layer()
                            == (stage + 1 < graph.len()).then_some(range.start),
                    "block actual representation/local-head graph",
                )?;
                let acts = specs
                    .iter()
                    .map(|l| match l.activation {
                        ProfileActivation::Square => DiagonalActivation::Square,
                        ProfileActivation::SquarePlusLinear => DiagonalActivation::PolyRelu2,
                        ProfileActivation::Identity => DiagonalActivation::Identity,
                    })
                    .collect();
                let frozen = specs.iter().all(|l| !l.bias);
                ensure(
                    frozen || specs.iter().all(|l| l.bias),
                    "block mixed bias policy",
                )?;
                descriptions.push(Description {
                    geometry: geometry.clone(),
                    activations: acts,
                    transpose: request.transpose.clone(),
                    frozen,
                });
            }
            (
                owner,
                descriptions,
                vec![materialized.metadata().classes; graph.len()],
                keys.training_rotations().to_vec(),
            )
        }
        #[cfg(test)]
        BlockPreparationInput::Synthetic { plan, claim } => {
            plan.validate(actual, parts, claim).map_err(mismatch)?;
            let owner = BlockPlanOwner::Synthetic(plan.clone());
            owner.check_shape()?;
            let descriptions = plan.descriptions.clone();
            let mut rotations = descriptions
                .iter()
                .flat_map(|d| d.geometry.rotation_steps().iter().copied())
                .collect::<Vec<_>>();
            rotations.sort_unstable();
            rotations.dedup();
            (owner, descriptions, plan.classes.clone(), rotations)
        }
    };
    factory(CheckedBlockPreparation {
        plan,
        descriptions,
        classes,
        parameters: actual.params,
        bootstrap_identity: actual.bootstrap.map(|b| b.descriptor_identity_bytes()),
        boot_parameters: actual.boot_params,
        rotations,
    })
    .map_err(CpuPreparationError::He)
}
pub(crate) fn make_blocks_owned(
    checked: &CheckedBlockPreparation<'_>,
    parameters: Arc<CkksParams>,
    rotations: &[i32],
    boot: Option<CpuBootstrapDescription>,
    key_rng: Rng,
    encryption_rng: Rng,
) -> Result<(StudyClient, CpuEvaluationContext)> {
    make_owned_fields(
        SetupFields {
            parameters: checked.parameters,
            bootstrap_identity: checked.bootstrap_identity,
            boot_parameters: checked.boot_parameters,
            rotations: &checked.rotations,
            config: checked.descriptions[0].geometry.eval_config(),
        },
        parameters,
        rotations,
        boot,
        key_rng,
        encryption_rng,
        #[cfg(test)]
        None,
    )
}
/// Immutable client-side encryption binding, containing neither secrets nor plaintext data.
pub struct BlockEncryptionBinding {
    owner: Arc<()>,
    plan: BlockPlanOwner,
    descriptions: Vec<Description>,
    classes: Vec<usize>,
}
/// Finite all-stage logical W/b and old-hidden values, disclosed only to the client.
pub struct ClientBlockSnapshot {
    stages: Vec<ClientModelSnapshot>,
    handoffs: Vec<Vec<Array2<f64>>>,
}
impl ClientBlockSnapshot {
    /// Per-stage model inspection in graph order.
    pub fn stages(&self) -> &[ClientModelSnapshot] {
        &self.stages
    }
    /// `[boundary][micro]` old-hidden values, truncated to valid rows.
    pub fn handoffs(&self) -> &[Vec<Array2<f64>>] {
        &self.handoffs
    }
}
impl StudyClient {
    /// Validates actual canonical inputs and encrypts their exact stage arrays.
    /// Reserve S extra candidate geometry/initial calls in the original request before
    /// creating the bound plan. Typed budget errors remain unchanged.
    pub fn prepare_block_initial(
        &self,
        materialized: &MaterializedProfile,
        registered: &RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &DiagonalRequest,
        context: &CpuEvaluationContext,
    ) -> CpuPreparationResult<(EncryptedBlockInitial, BlockEncryptionBinding)> {
        prepare_blocks_then_make(
            BlockPreparationInput::Canonical {
                materialized,
                registered,
                plan: plan.clone(),
                request,
            },
            context_parameters(context),
            plan.partitions(),
            |checked| {
                let arrays = canonical_arrays(materialized);
                self.encrypt_block_initial(checked, context, &arrays)
            },
        )
    }
    pub(crate) fn encrypt_block_initial(
        &self,
        checked: CheckedBlockPreparation<'_>,
        context: &CpuEvaluationContext,
        arrays: &[Vec<(Array2<f64>, Vec<f64>)>],
    ) -> Result<(EncryptedBlockInitial, BlockEncryptionBinding)> {
        self.check_context(context)?;
        ensure(
            std::ptr::eq(checked.parameters, context.parameters().as_ref()),
            "block initial actual parameters",
        )?;
        checked.plan.check_shape()?;
        let n = checked.descriptions.len();
        ensure(arrays.len() == n, "block initial array stages")?;
        for (s, d) in checked.descriptions.iter().enumerate() {
            context.check_geometry(&d.geometry)?;
            ensure(
                d.geometry.mode() == checked.descriptions[0].geometry.mode()
                    && d.geometry.layout() == checked.descriptions[0].geometry.layout(),
                "block initial common configuration",
            )?;
            ensure(
                arrays[s].len() == d.geometry.dims().len() - 1,
                "block initial array layers",
            )?;
            for ((w, b), dims) in arrays[s].iter().zip(d.geometry.dims().windows(2)) {
                ensure(
                    w.nrows() == dims[1] && w.ncols() == dims[0] && b.len() == dims[1],
                    "block initial array dimensions",
                )?;
            }
        }
        let before = context.counter();
        let diagnostic = context.evaluator.fork();
        let layout = checked.descriptions[0].geometry.layout();
        let mut rng = self.rng.borrow_mut();
        let mut models = Vec::with_capacity(n);
        let mut compact = Vec::with_capacity(n);
        let mut batches = Vec::with_capacity(n);
        for (stage, d) in checked.descriptions.iter().enumerate() {
            let m = checked.plan.initial()[stage].batch();
            batches.push(BatchSpec {
                x_level: m.x.level(),
                x_scale: m.x.scale(),
                y_level: m.y.level(),
                y_scale: m.y.scale(),
            });
            let model = match d.geometry.mode() {
                DiagonalMode::Interactive => Mlp::encrypt_at(
                    &diagnostic,
                    &arrays[stage],
                    activations(&d.activations),
                    &layout,
                    m.x.level(),
                    &self.encrypt,
                    &self.encoder,
                    &mut rng,
                )?,
                DiagonalMode::NonInteractive(placement) => {
                    let (model, state) = NonInteractiveTrainer::encrypt_network(
                        &diagnostic,
                        context
                            .bootstrap
                            .as_ref()
                            .ok_or_else(|| mismatch("block initial bootstrap dependency"))?,
                        &arrays[stage],
                        activations(&d.activations),
                        &layout,
                        placement,
                        &self.encrypt,
                        &self.encoder,
                        &mut rng,
                    )?;
                    compact.push(state);
                    model
                }
            };
            models.push(model);
        }
        ensure(
            before == context.counter(),
            "block initial probe counter isolation",
        )?;
        let output = models
            .pop()
            .ok_or_else(|| mismatch("block initial empty models"))?;
        let model = BlockMlp::new(models, output)?;
        let binding = BlockEncryptionBinding {
            owner: self.owner.clone(),
            plan: checked.plan.clone(),
            descriptions: checked.descriptions.clone(),
            classes: checked.classes.clone(),
        };
        Ok((
            EncryptedBlockInitial {
                owner: self.owner.clone(),
                plan: checked.plan,
                descriptions: checked.descriptions,
                classes: checked.classes,
                model,
                compact: (!compact.is_empty()).then_some(compact),
                batches,
            },
            binding,
        ))
    }
    /// Gathers canonical D6a rows once, then creates separate stage targets.
    #[allow(clippy::too_many_arguments)]
    pub fn encrypt_block_update(
        &self,
        binding: &BlockEncryptionBinding,
        materialized: &MaterializedProfile,
        registered: &RegisteredCase,
        plan: Arc<DiagonalPlan>,
        request: &DiagonalRequest,
        index: usize,
        context: &CpuEvaluationContext,
    ) -> CpuPreparationResult<EncryptedBlockUpdate> {
        study_admission::validate_plan_binding(
            materialized,
            registered,
            &plan,
            context_parameters(context),
            request,
        )
        .map_err(|e| CpuPreparationError::Admission(Box::new(e)))?;
        ensure(
            binding.plan.same(&BlockPlanOwner::Canonical(plan)),
            "block client canonical plan association",
        )?;
        let rows = materialized
            .update_rows(index)
            .map_err(|e| mismatch(&format!("block update rows: {e}")))?;
        let numeric = materialized.numeric_f64();
        let x = Array2::from_shape_fn((rows.rows().len(), numeric.train_x().ncols()), |(r, j)| {
            numeric.train_x()[[rows.rows()[r] as usize, j]]
        });
        let labels = rows
            .rows()
            .iter()
            .map(|&r| numeric.train_y()[r as usize])
            .collect::<Vec<_>>();
        let labels = vec![labels; binding.descriptions.len()];
        self.encrypt_block_rows(binding, index, context, &x, &labels)
            .map_err(CpuPreparationError::He)
    }
    pub(crate) fn encrypt_block_rows(
        &self,
        binding: &BlockEncryptionBinding,
        index: usize,
        context: &CpuEvaluationContext,
        x: &Array2<f64>,
        labels: &[Vec<u8>],
    ) -> Result<EncryptedBlockUpdate> {
        self.check_context(context)?;
        ensure(
            Arc::ptr_eq(&binding.owner, &self.owner),
            "block client prepared owner",
        )?;
        binding.plan.check_shape()?;
        let parts = binding
            .plan
            .parts()
            .get(index)
            .ok_or_else(|| mismatch("block client update index"))?;
        let n = binding.descriptions.len();
        let layout = binding.descriptions[0].geometry.layout();
        let total = parts
            .iter()
            .try_fold(0usize, |sum, p| sum.checked_add(p.valid))
            .ok_or_else(|| mismatch("block client row overflow"))?;
        ensure(
            x.nrows() == total
                && x.ncols() == binding.descriptions[0].geometry.dims()[0]
                && labels.len() == n
                && labels.iter().all(|l| l.len() == total),
            "block client label/row cardinality",
        )?;
        let widths = binding
            .descriptions
            .iter()
            .map(|d| *d.geometry.dims().last().unwrap())
            .collect::<Vec<_>>();
        let targets = labels
            .iter()
            .enumerate()
            .map(|(s, l)| {
                padded_targets(l, binding.classes[s], widths[s])
                    .map_err(|_| mismatch("block client stage labels"))
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(test)]
        crate::study_blocks::p5_tests::reserve_packet();
        let mut micros = Vec::with_capacity(parts.len());
        let mut rng = self.rng.borrow_mut();
        let mut offset = 0;
        for (micro, part) in parts.iter().enumerate() {
            let end = part
                .start
                .checked_add(part.valid)
                .ok_or_else(|| mismatch("block client partition overflow"))?;
            ensure(
                part.start == offset
                    && part.valid > 0
                    && part.valid <= layout.batch
                    && end <= total,
                "block client ordered partition",
            )?;
            offset = end;
            let m = binding
                .plan
                .transition(index, 0)
                .update_metadata()
                .unwrap()
                .micros()[micro];
            let xp = self.encoder.encode(
                &pack_activations(&x.slice(s![part.start..end, ..]).to_owned(), &layout),
                m.x.scale(),
                m.x.level(),
            )?;
            #[cfg(test)]
            crate::study_blocks::p5_tests::reserve_x();
            let encrypted_x = self.encrypt.encrypt(&xp, &mut rng);
            let mut ys = Vec::with_capacity(n);
            for (stage, target) in targets.iter().enumerate() {
                let m = binding
                    .plan
                    .transition(index, stage)
                    .update_metadata()
                    .unwrap()
                    .micros()[micro];
                let yp = self.encoder.encode(
                    &pack_activations(&target.slice(s![part.start..end, ..]).to_owned(), &layout),
                    m.y.scale(),
                    m.y.level(),
                )?;
                #[cfg(test)]
                crate::study_blocks::p5_tests::reserve_y();
                ys.push(self.encrypt.encrypt(&yp, &mut rng));
            }
            micros.push(EncryptedMicro {
                part: *part,
                x: encrypted_x,
                targets: ys,
            });
        }
        Ok(EncryptedBlockUpdate {
            owner: self.owner.clone(),
            plan: binding.plan.clone(),
            index,
            total,
            input_width: x.ncols(),
            target_widths: widths,
            micros,
        })
    }
    /// One client endpoint for all stage W/b and every successful old-hidden handoff.
    pub fn inspect_block_snapshot(
        &self,
        snapshot: &EncryptedBlockSnapshot,
        context: &CpuEvaluationContext,
    ) -> Result<ClientBlockSnapshot> {
        self.check_context(context)?;
        let before = context.counter();
        let _diagnostic = context.evaluator.fork();
        #[cfg(test)]
        crate::study_blocks::p5_tests::reserve_endpoint();
        let stages = snapshot
            .stages
            .iter()
            .map(|s| {
                #[cfg(test)]
                crate::study_blocks::p5_tests::reserve_inspection();
                self.inspect_snapshot(s, context)
            })
            .collect::<Result<Vec<_>>>()?;
        ensure(
            snapshot.handoffs.len() == snapshot.widths.len(),
            "block diagnostic boundaries",
        )?;
        let mut handoffs = Vec::with_capacity(snapshot.widths.len());
        for (boundary, cts) in snapshot.handoffs.iter().enumerate() {
            ensure(cts.len() == snapshot.valid.len(), "block diagnostic micros")?;
            let mut parts = Vec::with_capacity(cts.len());
            for (ct, &valid) in cts.iter().zip(&snapshot.valid) {
                #[cfg(test)]
                crate::study_blocks::p5_tests::reserve_handoff_inspection();
                let values = unpack_activations(
                    &self.encoder.decode(&self.decrypt.decrypt(ct)),
                    &snapshot.stages[boundary].layout,
                    snapshot.widths[boundary],
                );
                let values = values.slice(s![..valid, ..]).to_owned();
                ensure(
                    values.iter().all(|v| v.is_finite()),
                    "nonfinite block hidden snapshot",
                )?;
                parts.push(values);
            }
            handoffs.push(parts);
        }
        ensure(
            before == context.counter(),
            "block combined diagnostic counter isolation",
        )?;
        Ok(ClientBlockSnapshot { stages, handoffs })
    }
}
#[allow(clippy::type_complexity)]
fn canonical_arrays(m: &MaterializedProfile) -> Vec<Vec<(Array2<f64>, Vec<f64>)>> {
    m.graph()
        .stages()
        .iter()
        .map(|stage| {
            m.initial().layers()[stage.layers()]
                .iter()
                .map(|l| (l.weights().clone(), l.biases().to_vec()))
                .collect()
        })
        .collect()
}
// Real checked continuation; deliberately disconnected from production D7 dispatch/start.
#[allow(dead_code, clippy::too_many_arguments)]
fn prepare_block_owned_canonical(
    materialized: &MaterializedProfile,
    registered: &RegisteredCase,
    plan: Arc<DiagonalPlan>,
    request: &DiagonalRequest,
    parameters: Arc<CkksParams>,
    boot: Option<CpuBootstrapDescription>,
    key_rng: Rng,
    encryption_rng: Rng,
) -> CpuPreparationResult<(
    StudyClient,
    CpuEvaluationContext,
    EncryptedBlockInitial,
    BlockEncryptionBinding,
)> {
    let bound = boot.clone();
    let actual = PlanningParameters {
        params: &parameters,
        bootstrap: bound.as_ref().map(|b| b.metadata.as_ref()),
        boot_params: bound.as_ref().map(|b| &b.parameters),
    };
    prepare_blocks_then_make(
        BlockPreparationInput::Canonical {
            materialized,
            registered,
            plan: plan.clone(),
            request,
        },
        actual,
        plan.partitions(),
        |checked| {
            let steps = checked.rotations.clone();
            let (client, context) = make_blocks_owned(
                &checked,
                parameters.clone(),
                &steps,
                boot,
                key_rng,
                encryption_rng,
            )?;
            let (initial, binding) =
                client.encrypt_block_initial(checked, &context, &canonical_arrays(materialized))?;
            Ok((client, context, initial, binding))
        },
    )
}
