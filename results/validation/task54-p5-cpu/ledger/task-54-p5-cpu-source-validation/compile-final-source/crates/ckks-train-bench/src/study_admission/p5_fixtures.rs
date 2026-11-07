//! Separate synthetic block identity. Never a canonical plan or start capability.
use super::*;
use crate::{study_diagonal::Description, study_plain::p5_fixtures::BlockRecipe};

pub(crate) struct SyntheticBlockPlan {
    pub(crate) descriptions: Vec<Description>,
    pub(crate) initial: Vec<StateMetadata>,
    pub(crate) index: Vec<TransitionIndex>,
    pub(crate) rates: Vec<RateCoefficients<f64>>,
    pub(crate) reference: f64,
    pub(crate) parts: Vec<Vec<MicroPart>>,
    pub(crate) classes: Vec<usize>,
    identity: String,
    parameters: Vec<u8>,
    binding: Vec<u8>,
}
impl SyntheticBlockPlan {
    pub(crate) fn build(
        recipe: &BlockRecipe,
        mode: DiagonalMode,
        p: PlanningParameters<'_>,
        budget: Arc<PlanningBudget>,
        poison: bool,
    ) -> Result<Self> {
        let mut request = DiagonalRequest::quick();
        request.consumption = Consumption::Full;
        request.physical_lanes = Some(8);
        request.work_budget = budget;
        let count = if poison { 1 } else { recipe.rates().len() };
        let rates = recipe.rates()[..count]
            .iter()
            .map(|&rate| RateCoefficients {
                source: rate,
                mean: rate,
                delta: 1.,
                kernel: rate,
            })
            .collect::<Vec<_>>();
        let reference = recipe.reference();
        let mut descriptions = Vec::new();
        let mut states = Vec::new();
        for (stage, specs) in recipe.layer_specs().chunks_exact(2).enumerate() {
            let acts = specs
                .iter()
                .map(|l| match l.activation {
                    crate::study_profile::Activation::Square => DiagonalActivation::Square,
                    crate::study_profile::Activation::SquarePlusLinear => {
                        DiagonalActivation::PolyRelu2
                    }
                    crate::study_profile::Activation::Identity => DiagonalActivation::Identity,
                })
                .collect::<Vec<_>>();
            let dims = [specs[0].input, specs[0].output, specs[1].output];
            request.work_budget.charge(Work::Geometry)?;
            let geometry = plan_stage_geometry(
                &dims,
                &Layout::new(512, 16, 8)
                    .map_err(|_| issue(IssueKind::IdentityMismatch, "synthetic layout"))?,
                &acts,
                &request.transpose,
                mode,
                request.eval_config,
            )
            .map_err(|e| p2(e, stage))?;
            request.work_budget.charge(Work::Initial)?;
            let state = initial_stage_metadata(
                &geometry,
                p.params,
                p.bootstrap,
                DiagonalOptimizer::Nag {
                    lr: rates[0].kernel,
                    reference,
                    momentum: 0.9,
                },
                BiasPolicy::Frozen,
            )
            .map_err(|e| p2(e, stage))?;
            states.push(state);
            descriptions.push(Description {
                geometry,
                activations: acts,
                transpose: request.transpose.clone(),
                frozen: true,
            });
        }
        let initial = states
            .iter()
            .map(StateMetadata::from_state)
            .collect::<Result<Vec<_>>>()?;
        let counts = recipe.partitions()[..count]
            .iter()
            .map(|p| p.iter().sum())
            .collect::<Vec<_>>();
        request.partitions = Some(
            recipe.partitions()[..count]
                .iter()
                .map(|p| {
                    let mut start = 0;
                    p.iter()
                        .map(|&valid| {
                            let part = MicroPart { start, valid };
                            start += valid;
                            part
                        })
                        .collect()
                })
                .collect(),
        );
        let parts = partitions(&counts, 8, &request)?;
        let mut transcript = Transcript::new(request.limits);
        transcript.text("hekit-p5-distinct-cpu-block-v1")?;
        encode_params(&mut transcript, p)?;
        // Shared codec with explicit reference; deliberately no canonical maximum policy.
        encode_rate_schedule_fields(&mut transcript, &rates, reference)?;
        for (description, state) in descriptions.iter().zip(&states) {
            encode_geometry(&mut transcript, &description.geometry)?;
            encode_state(&mut transcript, state)?;
        }
        bound_partitions(&mut transcript, &parts, 8)?;
        let binding = transcript.bytes.clone();
        let identity = sha256_hex(&binding);
        let geometry = descriptions
            .iter()
            .map(|d| d.geometry.clone())
            .collect::<Vec<_>>();
        let index = compose(
            &geometry,
            &mut states,
            p,
            &parts,
            &rates,
            &request,
            &mut transcript,
        )?;
        let mut parameters = Transcript::new(request.limits);
        encode_params(&mut parameters, p)?;
        eprintln!(
            "P5_PLAN identity={identity} usage={:?} records={}",
            request.work_budget.usage(),
            index.len()
        );
        Ok(Self {
            descriptions,
            initial,
            index,
            rates,
            reference,
            parts,
            classes: recipe.stage_classes(),
            identity,
            parameters: parameters.bytes,
            binding,
        })
    }
    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }
    pub(crate) fn validate(
        &self,
        p: PlanningParameters<'_>,
        parts: &[Vec<MicroPart>],
        claim: &SyntheticBlockClaim,
    ) -> std::result::Result<(), &'static str> {
        if claim.identity != self.identity || claim.binding != self.binding {
            return Err("identity");
        }
        let mut actual = Transcript::new(ReceiptLimits::default());
        encode_params(&mut actual, p).map_err(|_| "parameters")?;
        if actual.bytes != self.parameters || claim.parameters != actual.bytes {
            return Err("parameters");
        }
        if parts != self.parts {
            return Err("partition");
        }
        Ok(())
    }
    pub(crate) fn claim(&self) -> SyntheticBlockClaim {
        SyntheticBlockClaim {
            identity: self.identity.clone(),
            binding: self.binding.clone(),
            parameters: self.parameters.clone(),
        }
    }
    /// Keep the original plan binding but make the claim agree with the actual supplied
    /// parameter description. This isolates actual-vs-bound rejection from malformed claims.
    pub(crate) fn claim_for_actual_parameters(
        &self,
        p: PlanningParameters<'_>,
    ) -> SyntheticBlockClaim {
        let mut actual = Transcript::new(ReceiptLimits::default());
        encode_params(&mut actual, p).unwrap();
        let mut claim = self.claim();
        claim.parameters = actual.bytes;
        claim
    }
    pub(crate) fn bad_parameter_claim(
        &self,
        p: PlanningParameters<'_>,
        which: usize,
    ) -> SyntheticBlockClaim {
        // Exercise the existing parameter codec without allocating any new CkksParams.
        let mut q = p.params.q().to_vec();
        let mut special = p.params.p().to_vec();
        let mut fields = ParameterFields::from_params(p.params);
        match which {
            0 => {
                q[0] ^= 2;
                fields.q = &q;
            }
            1 => fields.scale *= 2.,
            2 => fields.dnum += 1,
            3 => fields.secret = Some(1),
            4 => {
                special[0] ^= 2;
                fields.p = &special;
            }
            5 => fields.n *= 2,
            _ => {}
        }
        let mut original = Transcript::new(ReceiptLimits::default());
        encode_parameter_fields(&mut original, ParameterFields::from_params(p.params)).unwrap();
        let mut changed = Transcript::new(ReceiptLimits::default());
        encode_parameter_fields(&mut changed, fields).unwrap();
        let mut claim = self.claim();
        claim
            .parameters
            .splice(..original.bytes.len(), changed.bytes);
        if which == 6 {
            let identity = p
                .bootstrap
                .expect("NI spy descriptor")
                .descriptor_identity_bytes();
            let start = claim
                .parameters
                .windows(identity.len())
                .position(|w| w == identity)
                .unwrap();
            claim.parameters[start] ^= 1;
        }
        claim
    }
}
pub(crate) struct SyntheticBlockClaim {
    pub(crate) identity: String,
    pub(crate) binding: Vec<u8>,
    parameters: Vec<u8>,
}
