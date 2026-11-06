//! Typed client refresh and NI input projection. No bootstrap, NI session or secret key.
//! Packet identity is process-local, not a cryptographically authenticated transport.
use super::ops::mask_plain;
use super::types::check_ct;
use super::{
    LayerGeometry, PackedRecord, PackedStage, PackedStepReceipt, PackedVector, PackedWeights,
    VectorSpec, invalid,
};
use crate::layout_reboot::GradRelin;
use ckks_train_ckks::{CkksParams, DiagonalEntry, Encoder, HeCiphertext, HeOps, OpCounter};
use ckks_train_core::error::{HekitError, Result};
use std::{sync::Arc, time::Duration};

/// Exact state role. Layer ordinals belong to the complete geometry graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedRefreshRole {
    /// Model matrix.
    Weight(usize),
    /// Fixed-reference optimizer U; never unscaled V.
    Velocity(usize),
    /// Full model-output bias, including trained padded classes.
    Bias(usize),
    /// Canonical old hidden at one caller-supplied valid-record identity.
    Hidden {
        /// Stage ordinal within the geometry graph.
        stage: usize,
        /// Caller-supplied valid-record identity.
        record: u64,
    },
}
/// Immutable packing contract; content support cannot be proved from ciphertext metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedRefreshShape {
    /// Exact logical matrix and orientation.
    Matrix(LayerGeometry),
    /// Canonical vector with repeated logical copies.
    Vector(VectorSpec),
}
impl PackedRefreshShape {
    fn slots(&self) -> usize {
        match self {
            Self::Matrix(l) => l.layout().slots(),
            Self::Vector(v) => v.layout().slots(),
        }
    }
    fn mask(&self) -> Vec<f64> {
        match self {
            Self::Matrix(l) => l.matrix_mask(),
            Self::Vector(v) => v.mask(),
        }
    }
}
/// Client output context, exact parameter Delta and declared level.
#[derive(Clone, Debug)]
pub struct PackedRefreshContract {
    params: Arc<CkksParams>,
    level: usize,
}
impl PackedRefreshContract {
    /// Declares a client output level in this exact parameter context. Level zero is legal.
    pub fn new<E: HeOps>(ev: &E, level: usize) -> Result<Self> {
        if level > ev.params().max_level() {
            return invalid("packed refresh output level out of range");
        }
        Ok(Self {
            params: ev.params().clone(),
            level,
        })
    }
    /// Immutable parameter context (identity, not merely numerically equal primes).
    pub fn params(&self) -> &Arc<CkksParams> {
        &self.params
    }
    /// Exact output level.
    pub fn level(&self) -> usize {
        self.level
    }
    /// Exact output Delta.
    pub fn delta(&self) -> f64 {
        self.params.scale()
    }
    fn validate<E: HeOps>(&self, ev: &E) -> Result<()> {
        if !Arc::ptr_eq(&self.params, ev.params()) || self.level > ev.params().max_level() {
            return invalid("packed refresh output contract context or level mismatch");
        }
        Ok(())
    }
}
/// Private validated packet. There is no public arbitrary-state or gradient constructor.
#[derive(Clone, Debug)]
pub struct PackedRefreshRequest<C> {
    identity: Arc<()>,
    role: PackedRefreshRole,
    shape: PackedRefreshShape,
    ct: C,
    contract: PackedRefreshContract,
}
impl<C: HeCiphertext + Clone> PackedRefreshRequest<C> {
    fn new<E: HeOps<Ct = C>>(
        ev: &E,
        role: PackedRefreshRole,
        shape: PackedRefreshShape,
        ct: C,
        contract: PackedRefreshContract,
    ) -> Result<Self> {
        contract.validate(ev)?;
        check_exact(ev, &ct, &shape)?;
        Ok(Self {
            identity: Arc::new(()),
            role,
            shape,
            ct,
            contract,
        })
    }
    /// Bound role.
    pub fn role(&self) -> PackedRefreshRole {
        self.role
    }
    /// Bound exact packing.
    pub fn shape(&self) -> &PackedRefreshShape {
        &self.shape
    }
    /// Encrypted state only.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
    /// Required client output.
    pub fn output_contract(&self) -> &PackedRefreshContract {
        &self.contract
    }
    fn validate_response<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        response: &PackedRefreshResponse<C>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &response.identity) {
            return invalid("packed refresh response belongs to another transaction or role");
        }
        self.check_output(ev, &response.ct)
    }
    fn check_output<E: HeOps<Ct = C>>(&self, ev: &E, ct: &C) -> Result<()> {
        self.contract.validate(ev)?;
        check_exact(ev, ct, &self.shape)?;
        if ct.level() != self.contract.level {
            return Err(HekitError::LevelMismatch);
        }
        Ok(())
    }
}
/// A checked ciphertext bound to one request; cannot supply optimizer state metadata.
#[derive(Clone, Debug)]
pub struct PackedRefreshResponse<C> {
    identity: Arc<()>,
    ct: C,
}
impl<C: HeCiphertext + Clone> PackedRefreshResponse<C> {
    /// Validates context, two components, scale and output level against the request.
    pub fn new<E: HeOps<Ct = C>>(ev: &E, request: &PackedRefreshRequest<C>, ct: C) -> Result<Self> {
        request.check_output(ev, &ct)?;
        Ok(Self {
            identity: request.identity.clone(),
            ct,
        })
    }
    /// Read-only returned ciphertext.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
}
fn check_exact<E: HeOps>(ev: &E, ct: &E::Ct, shape: &PackedRefreshShape) -> Result<()> {
    check_ct(ev, ct, shape.slots())?;
    if ct.scale() != ev.params().scale() {
        return Err(HekitError::ScaleMismatch);
    }
    Ok(())
}
/// Ordered all-state transaction borrowing the immutable old or updated stage.
#[derive(Debug)]
pub struct PackedRefreshBatch<'a, C> {
    old: &'a PackedStage<C>,
    requests: Vec<PackedRefreshRequest<C>>,
}
impl<'a, C: HeCiphertext + Clone> PackedRefreshBatch<'a, C> {
    /// Prepares W0,U0?,b0?,W1,U1?,b1?; no HE work or model mutation.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        old: &'a PackedStage<C>,
        contract: PackedRefreshContract,
    ) -> Result<Self> {
        contract.validate(ev)?;
        let mut requests = Vec::new();
        for (k, layer) in old.layers().iter().enumerate() {
            let l = layer.weights().layer();
            for (role, state) in
                std::iter::once((PackedRefreshRole::Weight(l.ordinal()), layer.weights())).chain(
                    layer
                        .velocity()
                        .map(|u| (PackedRefreshRole::Velocity(l.ordinal()), u)),
                )
            {
                requests.push(PackedRefreshRequest::new(
                    ev,
                    role,
                    PackedRefreshShape::Matrix(l.clone()),
                    state.ciphertext().clone(),
                    contract.clone(),
                )?);
            }
            if let Some(b) = old.biases() {
                requests.push(PackedRefreshRequest::new(
                    ev,
                    PackedRefreshRole::Bias(l.ordinal()),
                    PackedRefreshShape::Vector(l.output()),
                    b[k].ciphertext().clone(),
                    contract.clone(),
                )?);
            }
        }
        Ok(Self { old, requests })
    }
    /// Exact ordered packets; clients cannot add U or change bias domain.
    pub fn requests(&self) -> &[PackedRefreshRequest<C>] {
        &self.requests
    }
    /// Validates ALL response identities/contracts before fresh-generation reconstruction.
    pub fn complete<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        responses: Vec<PackedRefreshResponse<C>>,
    ) -> Result<PackedRefreshCandidate<C>> {
        if responses.len() != self.requests.len() {
            return invalid("packed refresh missing or extra response");
        }
        for (req, res) in self.requests.iter().zip(&responses) {
            req.validate_response(ev, res)?;
        }
        let mut it = responses.into_iter();
        let mut weights = Vec::new();
        let mut velocities = Vec::new();
        let mut biases = Vec::new();
        for layer in self.old.layers() {
            let l = layer.weights().layer();
            weights.push(PackedWeights::new(
                ev,
                l.clone(),
                it.next().expect("validated count").ct,
            )?);
            velocities.push(if layer.velocity().is_some() {
                Some(PackedWeights::new(
                    ev,
                    l.clone(),
                    it.next().expect("validated count").ct,
                )?)
            } else {
                None
            });
            if self.old.biases().is_some() {
                biases.push(PackedVector::new(
                    ev,
                    l.output(),
                    it.next().expect("validated count").ct,
                )?);
            }
        }
        let mut w = weights.into_iter();
        let mut u = velocities.into_iter();
        let mut b = biases.into_iter();
        let stage = self.old.with_refreshed_parts(
            ev,
            [w.next().unwrap(), w.next().unwrap()],
            [u.next().unwrap(), u.next().unwrap()],
            self.old
                .biases()
                .map(|_| [b.next().unwrap(), b.next().unwrap()]),
        )?;
        Ok(PackedRefreshCandidate { stage })
    }
}
/// Complete refreshed stage, with no partial commit operation.
#[derive(Debug)]
pub struct PackedRefreshCandidate<C> {
    stage: PackedStage<C>,
}
impl<C> PackedRefreshCandidate<C> {
    /// Inspect without changing state.
    pub fn stage(&self) -> &PackedStage<C> {
        &self.stage
    }
    /// Infallible all-parts state transfer.
    pub fn into_stage(self) -> PackedStage<C> {
        self.stage
    }
}
/// Observed client-owned phases. Unavailable transport/device measurements stay absent.
#[derive(Clone, Debug, Default)]
pub struct ClientRefreshTiming {
    /// Measured transfer to the client, unavailable for an in-process fixture.
    pub download: Option<Duration>,
    /// Client decrypt/decode time from a monotonic clock.
    pub decrypt_decode: Option<Duration>,
    /// Client canonical extraction, pack, encode and encryption time.
    pub canonical_repack_encrypt: Option<Duration>,
    /// Measured return transfer, unavailable for an in-process fixture.
    pub upload: Option<Duration>,
    /// Actual downloaded bytes, if transport reports them.
    pub downloaded_bytes: Option<u64>,
    /// Actual uploaded bytes, if transport reports them.
    pub uploaded_bytes: Option<u64>,
}
/// Client-owned capability, separate from the generic server and any future NI session.
/// Implementations canonicalize logical state only, without optimizer/loss/gradient work.
pub trait PackedClientRefresh<E: HeOps> {
    /// Stable context/Delta/level promised for the entire transaction.
    fn output_contract(&self) -> &PackedRefreshContract;
    /// Decrypt, canonicalize structural padding/repeated copies, repack and encrypt.
    fn refresh(
        &mut self,
        request: &PackedRefreshRequest<E::Ct>,
    ) -> Result<(PackedRefreshResponse<E::Ct>, ClientRefreshTiming)>;
}
/// Successful client event; these are observations within the outer counter interval.
#[derive(Clone, Debug)]
pub struct PackedRefreshEvent {
    role: PackedRefreshRole,
    output_level: usize,
    output_scale: f64,
    timing: ClientRefreshTiming,
    delta: Option<OpCounter>,
}
impl PackedRefreshEvent {
    /// Completed role, in transaction order.
    pub fn role(&self) -> PackedRefreshRole {
        self.role
    }
    /// Observed output level.
    pub fn output_level(&self) -> usize {
        self.output_level
    }
    /// Observed output scale.
    pub fn output_scale(&self) -> f64 {
        self.output_scale
    }
    /// Nested evaluator observation for this call, never added again to the outer count.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.delta.as_ref()
    }
    /// Client-reported phases, not inferred GPU timing.
    pub fn timing(&self) -> &ClientRefreshTiming {
        &self.timing
    }
}
/// Failure boundary; only previously validated events appear in the failure receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedRefreshPhase {
    /// Invalid context or output contract before encrypted work.
    Preflight,
    /// The single encrypted prepare_step failed.
    Update,
    /// Updated-state packet preparation failed.
    Requests,
    /// Client service failed at the indicated role.
    Client,
    /// Client returned a response violating its bound contract/identity.
    Response,
    /// Whole response validation/reconstruction failed.
    Reconstruction,
}
/// Failure preserves caller state; already executed operations/client calls are not undone.
#[derive(Debug)]
pub struct PackedRefreshFailure {
    phase: PackedRefreshPhase,
    role: Option<PackedRefreshRole>,
    events: Vec<PackedRefreshEvent>,
    delta: Option<Box<OpCounter>>,
    error: HekitError,
}
impl PackedRefreshFailure {
    /// Failure boundary.
    pub fn phase(&self) -> PackedRefreshPhase {
        self.phase
    }
    /// Role in progress, if any.
    pub fn role(&self) -> Option<PackedRefreshRole> {
        self.role
    }
    /// Completed validated events only.
    pub fn events(&self) -> &[PackedRefreshEvent] {
        &self.events
    }
    /// Counter observation; absent if an external reset made subtraction unavailable.
    pub fn counter_delta(&self) -> Option<&OpCounter> {
        self.delta.as_deref()
    }
    /// Original backend/client/validation error.
    pub fn error(&self) -> &HekitError {
        &self.error
    }
}
impl std::fmt::Display for PackedRefreshFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "packed refresh {:?} {:?}: {}",
            self.phase, self.role, self.error
        )
    }
}
impl std::error::Error for PackedRefreshFailure {}
/// Immutable old-forward, encrypted-update and client-refresh evidence for one step.
#[derive(Debug)]
pub struct PackedInteractiveReceipt<C> {
    step: PackedStepReceipt<C>,
    updated: PackedStage<C>,
    events: Vec<PackedRefreshEvent>,
    actual_counts: OpCounter,
}
impl<C> PackedInteractiveReceipt<C> {
    /// Exactly the original prepare_step receipt, including old canonical hidden.
    pub fn step(&self) -> &PackedStepReceipt<C> {
        &self.step
    }
    /// Prepared encrypted state before client refresh; no mutable state API.
    pub fn updated_stage(&self) -> &PackedStage<C> {
        &self.updated
    }
    /// Validated refresh events in W/U/bias order.
    pub fn events(&self) -> &[PackedRefreshEvent] {
        &self.events
    }
    /// Single inclusive evaluator interval. Client refresh never adds fake bootstrap counts.
    pub fn actual_counts(&self) -> &OpCounter {
        &self.actual_counts
    }
}
/// One interactive step result; consuming the candidate commits every part together.
pub type PackedInteractiveResult<C> = std::result::Result<
    (PackedRefreshCandidate<C>, PackedInteractiveReceipt<C>),
    PackedRefreshFailure,
>;
/// Public-key server orchestrator. Client capability is supplied separately per call.
pub struct PackedInteractiveStage<'a, E: HeOps> {
    ev: &'a E,
    enc: &'a Encoder,
}
impl<'a, E: HeOps> PackedInteractiveStage<'a, E> {
    /// Validates public encoder parameter identity, without touching HE counters.
    pub fn new(ev: &'a E, enc: &'a Encoder) -> Result<Self> {
        if !Arc::ptr_eq(ev.params(), enc.params()) {
            return invalid("packed interactive encoder context mismatch");
        }
        Ok(Self { ev, enc })
    }
    /// Prepares exactly one real encrypted-label update, then refreshes ALL updated state.
    /// Failure retains the borrowed old state; retry is the caller's explicit new attempt.
    pub fn prepare_step<R: PackedClientRefresh<E>>(
        &self,
        old: &PackedStage<E::Ct>,
        records: &[PackedRecord<E::Ct>],
        rate: f64,
        relin: GradRelin,
        client: &mut R,
    ) -> PackedInteractiveResult<E::Ct> {
        let before = self.ev.counter();
        let mut events = Vec::new();
        let failure = |phase, role, events, error| PackedRefreshFailure {
            phase,
            role,
            events,
            delta: counter_delta(&self.ev.counter(), &before).map(Box::new),
            error,
        };
        let contract = client.output_contract().clone();
        contract
            .validate(self.ev)
            .map_err(|e| failure(PackedRefreshPhase::Preflight, None, vec![], e))?;
        let (candidate, step) = old
            .prepare_step(self.ev, self.enc, records, rate, relin)
            .map_err(|e| failure(PackedRefreshPhase::Update, None, vec![], e))?;
        let updated = candidate.into_stage();
        let batch = PackedRefreshBatch::new(self.ev, &updated, contract)
            .map_err(|e| failure(PackedRefreshPhase::Requests, None, vec![], e))?;
        let mut responses = Vec::new();
        for req in batch.requests() {
            let event_before = self.ev.counter();
            let (res, timing) = match client.refresh(req) {
                Ok(x) => x,
                Err(e) => {
                    return Err(failure(
                        PackedRefreshPhase::Client,
                        Some(req.role),
                        events,
                        e,
                    ));
                }
            };
            if let Err(e) = req.validate_response(self.ev, &res) {
                return Err(failure(
                    PackedRefreshPhase::Response,
                    Some(req.role),
                    events,
                    e,
                ));
            }
            events.push(PackedRefreshEvent {
                role: req.role,
                output_level: res.ct.level(),
                output_scale: res.ct.scale(),
                timing,
                delta: counter_delta(&self.ev.counter(), &event_before),
            });
            responses.push(res);
        }
        let refreshed = match batch.complete(self.ev, responses) {
            Ok(x) => x,
            Err(e) => return Err(failure(PackedRefreshPhase::Reconstruction, None, events, e)),
        };
        let Some(actual_counts) = counter_delta(&self.ev.counter(), &before) else {
            return Err(failure(
                PackedRefreshPhase::Reconstruction,
                None,
                events,
                HekitError::InvalidParams(
                    "packed refresh evaluator counter reset during transaction".into(),
                ),
            ));
        };
        Ok((
            refreshed,
            PackedInteractiveReceipt {
                step,
                updated,
                events,
                actual_counts,
            },
        ))
    }
}
fn counter_delta(a: &OpCounter, b: &OpCounter) -> Option<OpCounter> {
    Some(OpCounter {
        rotations: a.rotations.checked_sub(b.rotations)?,
        relins: a.relins.checked_sub(b.relins)?,
        ct_ct_tensor: a.ct_ct_tensor.checked_sub(b.ct_ct_tensor)?,
        ct_pt_mults: a.ct_pt_mults.checked_sub(b.ct_pt_mults)?,
        rescales: a.rescales.checked_sub(b.rescales)?,
        sse_switches: a.sse_switches.checked_sub(b.sse_switches)?,
        bootstraps: a.bootstraps.checked_sub(b.bootstraps)?,
    })
}
/// Validated level-zero exact-Delta input, carrying an UNVERIFIED range obligation.
#[derive(Debug)]
pub struct PackedBootInput<C> {
    ct: C,
    role: PackedRefreshRole,
    shape: PackedRefreshShape,
}
impl<C> PackedBootInput<C> {
    /// Projected state or already canonical hidden, at level zero.
    pub fn ciphertext(&self) -> &C {
        &self.ct
    }
    /// Role defining the numerical obligation.
    pub fn role(&self) -> PackedRefreshRole {
        self.role
    }
    /// Geometry/representation retained through lowering.
    pub fn shape(&self) -> &PackedRefreshShape {
        &self.shape
    }
    /// Required |slot| <= 1 across ALL slots. Metadata does NOT verify this bound.
    pub fn required_all_slot_absolute_bound(&self) -> f64 {
        1.0
    }
}
/// NI state input: one support multiplication/rescale, then free mod-switch to zero.
/// Rejects source level zero and hidden roles before counted work; performs no bootstrap.
pub fn project_state<E: HeOps>(
    ev: &E,
    enc: &Encoder,
    request: &PackedRefreshRequest<E::Ct>,
) -> Result<PackedBootInput<E::Ct>> {
    if !Arc::ptr_eq(ev.params(), enc.params()) {
        return invalid("packed projection encoder context mismatch");
    }
    request.contract.validate(ev)?;
    check_exact(ev, &request.ct, &request.shape)?;
    if matches!(request.role, PackedRefreshRole::Hidden { .. }) {
        return invalid("canonical hidden does not require state projection");
    }
    if request.ct.level() == 0 {
        return Err(HekitError::NoLevelsLeft);
    }
    let pt = mask_plain(
        ev,
        enc,
        &request.shape.mask(),
        request.ct.level(),
        request.ct.scale(),
    )?;
    let pt = ev.upload_pt_owned(pt)?;
    let ct = ev.rescale(&ev.mul_plain_to_scale(&request.ct, &pt, ev.params().scale())?)?;
    check_exact(ev, &ct, &request.shape)?;
    if ct.level() + 1 != request.ct.level() {
        return Err(HekitError::LevelMismatch);
    }
    let ct = ev.mod_switch_to_level(&ct, 0)?;
    check_exact(ev, &ct, &request.shape)?;
    if ct.level() != 0 {
        return Err(HekitError::LevelMismatch);
    }
    Ok(PackedBootInput {
        ct,
        role: request.role,
        shape: request.shape.clone(),
    })
}
/// Canonical hidden from the old stage activation mask. Metadata cannot attest contents.
#[derive(Debug)]
pub struct CanonicalHidden<C> {
    ct: C,
    role: PackedRefreshRole,
    spec: VectorSpec,
}
impl<C: HeCiphertext + Clone> CanonicalHidden<C> {
    /// Binds a backbone's exact output spec and record identity; caller asserts masked content.
    pub fn new<E: HeOps<Ct = C>>(
        ev: &E,
        backbone: &LayerGeometry,
        record: u64,
        hidden: &PackedVector<C>,
    ) -> Result<Self> {
        if backbone.ordinal() % 2 != 0 || backbone.output() != hidden.spec() {
            return invalid("canonical hidden must match a backbone output");
        }
        check_exact(
            ev,
            hidden.ciphertext(),
            &PackedRefreshShape::Vector(hidden.spec()),
        )?;
        Ok(Self {
            ct: hidden.ciphertext().clone(),
            role: PackedRefreshRole::Hidden {
                stage: backbone.ordinal() / 2,
                record,
            },
            spec: hidden.spec(),
        })
    }
    /// Typed client packet for later hidden-boundary orchestration; no handoff is run here.
    pub fn request<E: HeOps<Ct = C>>(
        &self,
        ev: &E,
        contract: PackedRefreshContract,
    ) -> Result<PackedRefreshRequest<C>> {
        PackedRefreshRequest::new(
            ev,
            self.role,
            PackedRefreshShape::Vector(self.spec),
            self.ct.clone(),
            contract,
        )
    }
    /// Existing hidden mask already paid normalization: only mod-switch, no second mask.
    pub fn boot_input<E: HeOps<Ct = C>>(&self, ev: &E) -> Result<PackedBootInput<C>> {
        let shape = PackedRefreshShape::Vector(self.spec);
        check_exact(ev, &self.ct, &shape)?;
        let ct = ev.mod_switch_to_level(&self.ct, 0)?;
        check_exact(ev, &ct, &shape)?;
        if ct.level() != 0 {
            return Err(HekitError::LevelMismatch);
        }
        Ok(PackedBootInput {
            ct,
            role: self.role,
            shape,
        })
    }
}
