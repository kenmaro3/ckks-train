//! Distinct NONSCIENTIFIC_TEST_EVIDENCE plans, never canonical admission values.
use super::*;
use crate::study_plain::p4_fixtures::SignalRecipe;

pub(crate) struct SyntheticCpuPlan {
    pub(crate) geometry: StageGeometry,
    pub(crate) initial: StateMetadata,
    pub(crate) index: Vec<TransitionIndex>,
    pub(crate) rates: Vec<RateCoefficients<f64>>,
    pub(crate) reference: f64,
    pub(crate) parts: Vec<Vec<MicroPart>>,
    pub(crate) frozen: bool,
    pub(crate) transpose: TransposeKeys,
    identity: String,
    parameters: Vec<u8>,
    binding: Vec<u8>,
}
impl SyntheticCpuPlan {
    pub(crate) fn build(
        recipe: &SignalRecipe,
        mode: DiagonalMode,
        p: PlanningParameters<'_>,
        budget: Arc<PlanningBudget>,
    ) -> Result<Self> {
        let mut request = DiagonalRequest::quick();
        request.consumption = Consumption::Full;
        request.physical_lanes = Some(8);
        request.work_budget = budget;
        let rates = recipe.rates();
        let reference = recipe.reference();
        let square = recipe.square();
        let acts = if square {
            [DiagonalActivation::Square; 2]
        } else {
            [DiagonalActivation::PolyRelu2, DiagonalActivation::Identity]
        };
        request.work_budget.charge(Work::Geometry)?;
        let geometry = plan_stage_geometry(
            &[2, 4, recipe.width()],
            &Layout::new(512, 16, 8)
                .map_err(|_| issue(IssueKind::IdentityMismatch, "synthetic layout"))?,
            &acts,
            &request.transpose,
            mode,
            request.eval_config,
        )
        .map_err(|e| p2(e, 0))?;
        let optimizer = if square {
            DiagonalOptimizer::Sgd {
                lr: rates[0].kernel,
            }
        } else {
            DiagonalOptimizer::Nag {
                lr: rates[0].kernel,
                reference,
                momentum: 0.9,
            }
        };
        request.work_budget.charge(Work::Initial)?;
        let state = initial_stage_metadata(
            &geometry,
            p.params,
            p.bootstrap,
            optimizer,
            if square {
                BiasPolicy::Active
            } else {
                BiasPolicy::Frozen
            },
        )
        .map_err(|e| p2(e, 0))?;
        let initial = StateMetadata::from_state(&state)?;
        let counts: Vec<_> = recipe.partitions().iter().map(|p| p.iter().sum()).collect();
        request.partitions = Some(
            recipe
                .partitions()
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
        transcript.text("hekit-p4-distinct-cpu-synthetic-v1")?;
        encode_params(&mut transcript, p)?;
        super::rates(&mut transcript, &rates, reference)?;
        encode_geometry(&mut transcript, &geometry)?;
        encode_state(&mut transcript, &state)?;
        bound_partitions(&mut transcript, &parts, 8)?;
        let binding = transcript.bytes.clone();
        let identity = sha256_hex(&binding);
        let index = compose(
            std::slice::from_ref(&geometry),
            &mut [state],
            p,
            &parts,
            &rates,
            &request,
            &mut transcript,
        )?;
        let mut parameters = Transcript::new(request.limits);
        encode_params(&mut parameters, p)?;
        eprintln!(
            "P4_PLAN usage={:?} records={}",
            request.work_budget.usage(),
            index.len()
        );
        Ok(Self {
            geometry,
            initial,
            index,
            rates,
            reference,
            parts,
            frozen: !square,
            transpose: request.transpose,
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
        claim: &SyntheticClaim,
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
    pub(crate) fn claim(&self) -> SyntheticClaim {
        SyntheticClaim {
            identity: self.identity.clone(),
            binding: self.binding.clone(),
            parameters: self.parameters.clone(),
        }
    }
    /// Keep the original plan binding but make the claim agree with the actual supplied
    /// parameter description. This isolates actual-vs-bound rejection from malformed claims.
    pub(crate) fn claim_for_actual_parameters(&self, p: PlanningParameters<'_>) -> SyntheticClaim {
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
    ) -> SyntheticClaim {
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
pub(crate) struct SyntheticClaim {
    pub(crate) identity: String,
    pub(crate) binding: Vec<u8>,
    parameters: Vec<u8>,
}
