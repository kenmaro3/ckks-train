//! Pre-key structural bootstrap receipts. Not full numerical/security/runtime admission.
//!
//! First obtain a process-local [`DftSupportProof`], then pass it explicitly to a named
//! constructor. Constructors never generate DFT roots, slot vectors, encoders or HE keys.
//! The proof assumes ordinary binary64 rounding and no subsequent external mutation of
//! floating-point control state or executable/library code. Transcendental correctness
//! is not universal: finite root checks and actual per-descriptor folded checks are used.

use super::linear::{Shape, folded_factors, shape_from_offsets, sparse_transform_cost};
use super::{BootParams, Direction};
use crate::metadata::{CtMeta, exact_product_scale_at, target_plain_scale_at};
use crate::{CkksParams, OpCounter};
use ckks_train_core::error::HekitError;
use num_complex::Complex64;
use std::fmt;
use std::mem::size_of;

mod configured {
    include!(concat!(env!("OUT_DIR"), "/bootstrap_diagnostics.rs"));
}

/// Canonical byte schema version.
pub const SCHEMA_VERSION: u32 = 1;
/// Shared structural/scale algorithm revision.
pub const ALGORITHM_REVISION: u32 = 1;
/// Finite support premise revision (num-complex 0.4.6 ordinary multiplication/division).
pub const DFT_SUPPORT_PROOF_REVISION: u32 = 1;
/// Actual coefficient generation and zero-sensitive splitting revision.
pub const COEFFICIENT_ALGORITHM_REVISION: u32 = 1;

/// Resource counted by the operational metadata budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataResource {
    /// Actual scalar coefficient count.
    Coefficients,
    /// Quadratic interpolation summands.
    InterpolationTerms,
    /// Polynomial recursion, scans and cache work.
    PolynomialWork,
    /// Conservative requested heap payload bytes.
    RequestedBytes,
}
/// A missing proof, distinct from invalid CKKS parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingRequirement {
    /// Actual finite-F64 nonzero DFT support premise.
    DftKeptSupportProof,
}
/// Why the current numerical environment did not establish the premise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DftProofFailure {
    /// The legacy modular powers require a verified 64-bit usize environment.
    UnsupportedPointerWidth,
    /// The target family has not received the source/finite-domain proof checks.
    UnsupportedTargetEnvironment,
    /// Required configured-compiler or target identification is unavailable.
    ConfiguredCompilerDiagnosticsUnavailable,
    /// An actual root failed the conservative finite norm bounds.
    RootBoundFailure {
        /// Butterfly length.
        length: usize,
        /// Index within the butterfly.
        index: usize,
    },
    /// An actual inverse root failed the conservative finite norm bounds.
    InverseBoundFailure {
        /// Butterfly length.
        length: usize,
        /// Index within the butterfly.
        index: usize,
    },
    /// An ordinary binary64 or actual complex arithmetic probe failed.
    ArithmeticPremiseFailure,
    /// An actual folded scalar is outside the proved finite range.
    FoldedFactorBoundFailure,
}
/// Metadata-specific contextual outcomes. No error path allocates diagnostic strings here.
#[derive(Debug)]
pub enum BootstrapMetadataError {
    /// Malformed or unsupported metadata-domain parameters.
    InvalidParams {
        /// Affected parameter field.
        field: &'static str,
        /// Bounded diagnostic reason.
        reason: &'static str,
    },
    /// A checked size or count operation overflowed.
    ArithmeticOverflow {
        /// Affected checked operation.
        operation: &'static str,
    },
    /// An actual intermediate became nonfinite.
    NonFiniteIntermediate {
        /// Affected checked operation.
        operation: &'static str,
        /// Scalar root or coefficient index at failure.
        index: Option<usize>,
    },
    /// Operational work/storage cap exceeded, not mathematical invalidity.
    MetadataBudgetExceeded {
        /// Operational resource that exceeded its cap.
        resource: MetadataResource,
        /// Required resource amount.
        required: u64,
        /// Configured resource cap.
        limit: u64,
    },
    /// A fallible vector reservation failed.
    AllocationFailed {
        /// Name of the requested storage buffer.
        allocation: &'static str,
        /// Requested element capacity.
        elements: usize,
    },
    /// The shared exact scale search failed at a contextual boundary.
    ScalePlanning {
        /// DFT direction when applicable.
        direction: Option<Direction>,
        /// Stage ordinal when applicable.
        stage: Option<usize>,
        /// Ciphertext boundary level.
        level: usize,
        /// Original allocation-free shared scale error.
        source: HekitError,
    },
    /// A mandatory proof premise has not been established.
    MissingCode {
        /// Missing mandatory proof requirement.
        requirement: MissingRequirement,
        /// Bounded diagnostic reason.
        reason: DftProofFailure,
    },
}
impl fmt::Display for BootstrapMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bootstrap metadata: {self:?}")
    }
}
impl std::error::Error for BootstrapMetadataError {}
pub(crate) type MetaResult<T> = Result<T, BootstrapMetadataError>;
fn invalid(field: &'static str, reason: &'static str) -> BootstrapMetadataError {
    BootstrapMetadataError::InvalidParams { field, reason }
}
pub(crate) fn overflow(operation: &'static str) -> BootstrapMetadataError {
    BootstrapMetadataError::ArithmeticOverflow { operation }
}
fn missing(reason: DftProofFailure) -> BootstrapMetadataError {
    BootstrapMetadataError::MissingCode {
        requirement: MissingRequirement::DftKeptSupportProof,
        reason,
    }
}
pub(crate) fn finite(v: f64, operation: &'static str, index: Option<usize>) -> MetaResult<f64> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(BootstrapMetadataError::NonFiniteIntermediate { operation, index })
    }
}
pub(crate) fn increment(v: &mut u64, n: u64) -> MetaResult<()> {
    *v = v
        .checked_add(n)
        .ok_or_else(|| overflow("operation counter addition"))?;
    Ok(())
}
pub(crate) fn checked_merge(a: &mut OpCounter, b: &OpCounter) -> MetaResult<()> {
    increment(&mut a.rotations, b.rotations)?;
    increment(&mut a.relins, b.relins)?;
    increment(&mut a.ct_ct_tensor, b.ct_ct_tensor)?;
    increment(&mut a.ct_pt_mults, b.ct_pt_mults)?;
    increment(&mut a.rescales, b.rescales)?;
    increment(&mut a.sse_switches, b.sse_switches)?;
    increment(&mut a.bootstraps, b.bootstraps)
}
fn limit(resource: MetadataResource, required: u64, cap: u64) -> MetaResult<()> {
    if required > cap {
        Err(BootstrapMetadataError::MetadataBudgetExceeded {
            resource,
            required,
            limit: cap,
        })
    } else {
        Ok(())
    }
}

/// Metadata-only operational limits. Raising them does not alter mathematical validity.
#[derive(Debug, Clone, Copy)]
pub struct MetadataLimits {
    /// Maximum scalar coefficient count (default4096).
    pub max_coefficients: u64,
    /// Maximum interpolation summands (default2^24).
    pub max_interpolation_terms: u64,
    /// Maximum conservative polynomial work (default2^24).
    pub max_polynomial_work: u64,
    /// Maximum conservatively requested heap payload (default64MiB).
    pub max_peak_bytes: u64,
}
impl Default for MetadataLimits {
    fn default() -> Self {
        Self {
            max_coefficients: 4096,
            max_interpolation_terms: 1 << 24,
            max_polynomial_work: 1 << 24,
            max_peak_bytes: 64 << 20,
        }
    }
}
/// Conservative requested heap storage/work, not allocator RSS or runtime key memory.
#[derive(Debug, Clone, Copy)]
pub struct MetadataResourceBounds {
    coefficients: u64,
    interpolation_terms: u64,
    polynomial_work: u64,
    requested_bytes: u64,
    recursion_depth: usize,
}
impl MetadataResourceBounds {
    /// Number of scalar coefficients budgeted.
    pub fn coefficients(&self) -> u64 {
        self.coefficients
    }
    /// Checked quadratic interpolation term count.
    pub fn interpolation_terms(&self) -> u64 {
        self.interpolation_terms
    }
    /// Conservative polynomial traversal and cache work bound.
    pub fn polynomial_work(&self) -> u64 {
        self.polynomial_work
    }
    /// Cumulative requested capacities conservatively bound simultaneous live heap payload.
    pub fn requested_bytes(&self) -> u64 {
        self.requested_bytes
    }
    /// Maximum polynomial recursion frame depth bound.
    pub fn recursion_depth(&self) -> usize {
        self.recursion_depth
    }
}
/// Charges every vector's requested capacity, including temporary and canonical buffers.
/// Cumulative charging deliberately overbounds peak live storage, avoiding hidden releases.
pub(crate) struct Budget {
    cap: u64,
    pub(crate) bytes: u64,
    work: u64,
    work_limit: u64,
}
impl Budget {
    pub(crate) fn new(cap: u64, work_limit: u64) -> Self {
        Self {
            cap,
            bytes: 0,
            work: 0,
            work_limit,
        }
    }
    pub(crate) fn vec<T>(&mut self, n: usize, name: &'static str) -> MetaResult<Vec<T>> {
        let bytes = n
            .checked_mul(size_of::<T>())
            .ok_or_else(|| overflow("allocation layout"))?;
        let next = self
            .bytes
            .checked_add(u64::try_from(bytes).map_err(|_| overflow("allocation u64"))?)
            .ok_or_else(|| overflow("allocation sum"))?;
        limit(MetadataResource::RequestedBytes, next, self.cap)?;
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| BootstrapMetadataError::AllocationFailed {
                allocation: name,
                elements: n,
            })?;
        self.bytes = next;
        Ok(v)
    }
    pub(crate) fn zeros(&mut self, n: usize) -> MetaResult<Vec<f64>> {
        let mut v = self.vec(n, "polynomial scalars")?;
        v.resize(n, 0.0);
        Ok(v)
    }
    pub(crate) fn fuel(&mut self, n: usize) -> MetaResult<()> {
        let next = self
            .work
            .checked_add(u64::try_from(n).map_err(|_| overflow("work conversion"))?)
            .ok_or_else(|| overflow("polynomial work"))?;
        limit(MetadataResource::PolynomialWork, next, self.work_limit)?;
        self.work = next;
        Ok(())
    }
}

/// Immutable, non-deserializable proof token for this running process. Diagnostics are
/// borrowed static strings and allocate nothing. Not an executable/source attestation.
#[derive(Debug, Clone, Copy)]
pub struct DftSupportProof {
    roots: u64,
    root_min: f64,
    root_max: f64,
    inverse_min: f64,
    inverse_max: f64,
}
impl DftSupportProof {
    /// Streams all 65,535 scalar roots through the actual runtime root/inverse arithmetic.
    /// Exact Apple64 and GNU x86_64 Linux64 candidates; Linux validation is pending.
    /// Target eligibility alone is not tested support; actual scalar checks are mandatory.
    /// Requires unchanged ordinary FP environment thereafter. No DFT vectors or HE.
    pub fn verify_current() -> MetaResult<Self> {
        check_proof_environment(
            usize::BITS,
            configured::TARGET,
            configured::CONFIGURED_COMPILER,
            configured::HOST,
        )?;
        if f64::RADIX != 2
            || f64::MANTISSA_DIGITS != 53
            || (std::hint::black_box(1.0) + f64::EPSILON / 2.0) != 1.0
            || std::hint::black_box(f64::MIN_POSITIVE) / 2.0 == 0.0
        {
            return Err(missing(DftProofFailure::ArithmeticPremiseFailure));
        }
        let mut p = Self {
            roots: 0,
            root_min: f64::INFINITY,
            root_max: 0.0,
            inverse_min: f64::INFINITY,
            inverse_max: 0.0,
        };
        for s in 1..=16 {
            let length = 1usize << s;
            let m = 4 * length;
            for j in 0..length / 2 {
                let w = super::linear::root(super::linear::pow_mod(m - 3, j, m), m);
                // Deliberately stricter squared-norm gates leave rounding margin for
                // mathematical norm bounds [1/2,2] and inverse bounds [1/8,2].
                let norm = w.norm_sqr();
                if !(w.re.is_finite() && w.im.is_finite() && (0.5..=2.0).contains(&norm)) {
                    return Err(missing(DftProofFailure::RootBoundFailure {
                        length,
                        index: j,
                    }));
                }
                let inverse = Complex64::new(0.5, 0.0) / w;
                let invnorm = inverse.norm_sqr();
                if !(inverse.re.is_finite()
                    && inverse.im.is_finite()
                    && (1.0 / 32.0..=2.0).contains(&invnorm))
                {
                    return Err(missing(DftProofFailure::InverseBoundFailure {
                        length,
                        index: j,
                    }));
                }
                // Actual product/addition order, inverse residual, zero, conjugation, i.
                let residual = w * inverse - Complex64::new(0.5, 0.0);
                if !residual.norm_sqr().is_finite()
                    || residual.norm_sqr() > 1e-28
                    || w * Complex64::new(0.0, 0.0) != Complex64::new(0.0, 0.0)
                    || w.conj().norm_sqr() != norm
                    || (w * Complex64::i()).norm_sqr() != norm
                {
                    return Err(missing(DftProofFailure::ArithmeticPremiseFailure));
                }
                p.root_min = p.root_min.min(norm);
                p.root_max = p.root_max.max(norm);
                p.inverse_min = p.inverse_min.min(invnorm);
                p.inverse_max = p.inverse_max.max(invnorm);
                p.roots += 1;
            }
        }
        Ok(p)
    }
    /// Number of actual scalar roots verified in this process.
    pub fn root_count(&self) -> u64 {
        self.roots
    }
    /// Cargo-configured compiler diagnostics; wrappers and loaded libm are not attested.
    pub fn configured_compiler(&self) -> Option<&'static str> {
        configured::CONFIGURED_COMPILER
    }
    /// Cargo target triple, or explicit absence of diagnostics.
    pub fn target(&self) -> Option<&'static str> {
        configured::TARGET
    }
    /// Cargo build host triple, or explicit absence of diagnostics.
    pub fn host(&self) -> Option<&'static str> {
        configured::HOST
    }
    /// Observed extrema of squared root norms in the finite domain.
    pub fn root_squared_norm_bounds(&self) -> (f64, f64) {
        (self.root_min, self.root_max)
    }
    /// Observed extrema of squared inverse-root norms in the finite domain.
    pub fn inverse_squared_norm_bounds(&self) -> (f64, f64) {
        (self.inverse_min, self.inverse_max)
    }
}

// Pure diagnostic guards do not mint tokens; only verify_current performs the scalar gate.
fn check_proof_environment(
    bits: u32,
    target: Option<&str>,
    compiler: Option<&str>,
    host: Option<&str>,
) -> MetaResult<()> {
    if bits != 64 {
        return Err(missing(DftProofFailure::UnsupportedPointerWidth));
    }
    if [target, compiler, host]
        .iter()
        .any(|v| v.is_none_or(str::is_empty))
    {
        return Err(missing(
            DftProofFailure::ConfiguredCompilerDiagnosticsUnavailable,
        ));
    }
    if !matches!(
        target,
        Some("aarch64-apple-darwin" | "x86_64-unknown-linux-gnu")
    ) {
        return Err(missing(DftProofFailure::UnsupportedTargetEnvironment));
    }
    Ok(())
}

/// Key-generator declaration, not an attestation of a sampled secret.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretDeclaration {
    /// Declared dense ternary sampling; realized weight is unknown.
    DenseTernary,
    /// Declared fixed Hamming weight.
    FixedWeight(usize),
}
fn secret(weight: Option<usize>) -> SecretDeclaration {
    weight.map_or(
        SecretDeclaration::DenseTernary,
        SecretDeclaration::FixedWeight,
    )
}
/// Successful construction establishes only this conditional scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofScope {
    /// Exact structural counts and boundary labels conditional on runtime success.
    StructuralCostAndBoundaryScales,
}
/// Admission conditions intentionally not established by structural metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnvalidatedRequirement {
    /// Encoded DFT coefficients fitting Q/2 is unvalidated.
    EncodingQOverTwoFeasibility,
    /// Internal Chebyshev scales and 2^126 constants are unvalidated.
    EvalModInternalScaleUnvalidated,
    /// Encrypted message and overflow range is unvalidated.
    EncryptedInputRange,
    /// Approximation and decryption precision is unvalidated.
    ApproximationPrecision,
    /// Actual sampled-secret declaration compliance is unvalidated.
    ActualSecretContract,
    /// Actual key availability, parameter binding and provenance is unvalidated.
    KeyBinding,
    /// Cryptographic security is unvalidated.
    SecurityEstimate,
    /// Runtime key/table/device memory fit is unvalidated.
    RuntimeMemoryFit,
    /// Runtime backend admission is unvalidated.
    BackendSupport,
}
const UNVALIDATED: [UnvalidatedRequirement; 9] = [
    UnvalidatedRequirement::EncodingQOverTwoFeasibility,
    UnvalidatedRequirement::EvalModInternalScaleUnvalidated,
    UnvalidatedRequirement::EncryptedInputRange,
    UnvalidatedRequirement::ApproximationPrecision,
    UnvalidatedRequirement::ActualSecretContract,
    UnvalidatedRequirement::KeyBinding,
    UnvalidatedRequirement::SecurityEstimate,
    UnvalidatedRequirement::RuntimeMemoryFit,
    UnvalidatedRequirement::BackendSupport,
];

/// Compact stored-offset support S(a,r); duplicates at the wrap are normalized by iteration.
#[derive(Debug, Clone, Copy)]
pub struct StoredOffsetSupport {
    slots: usize,
    stride: usize,
    radius: usize,
}
impl StoredOffsetSupport {
    /// Full packing slot count.
    pub fn slots(&self) -> usize {
        self.slots
    }
    /// Slot distance between consecutive window positions.
    pub fn stride(&self) -> usize {
        self.stride
    }
    /// Radius of the signed integer interval defining stored offsets.
    pub fn signed_radius(&self) -> usize {
        self.radius
    }
    /// Whether a normalized offset is stored (before parity support pruning).
    pub fn contains(&self, offset: usize) -> bool {
        offset < self.slots && offset % self.stride == 0 && {
            let t = offset / self.stride;
            let m = self.slots / self.stride;
            t <= self.radius || m - t <= self.radius
        }
    }
    /// Sorted distinct normalized stored offsets, with no heap allocation.
    pub fn offsets(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        (0..self.slots)
            .step_by(self.stride)
            .filter(|&o| self.contains(o))
    }
}
/// One stage's stored shape, pre-encoding kept support, execution multiplicity and labels.
#[derive(Debug)]
pub struct DftStageMetadata {
    direction: Direction,
    low: u32,
    high: u32,
    ordinal: usize,
    support: StoredOffsetSupport,
    shape: Shape,
    main: Vec<usize>,
    second: Option<Vec<usize>>,
    baby_steps: [Vec<i32>; 2],
    giant_steps: Vec<i32>,
    input: CtMeta,
    output: CtMeta,
    label: u64,
    plain: u64,
    mag: u64,
    phase: [u64; 2],
    folded: [u64; 2],
    cost: OpCounter,
}
impl DftStageMetadata {
    /// Direction in which this stage is applied.
    pub fn direction(&self) -> Direction {
        self.direction
    }
    /// Original lowest zero-based butterfly distance bit.
    pub fn low_bit(&self) -> u32 {
        self.low
    }
    /// Original highest zero-based butterfly distance bit.
    pub fn high_bit(&self) -> u32 {
        self.high
    }
    /// Zero-based stage ordinal in application order.
    pub fn apply_ordinal(&self) -> usize {
        self.ordinal
    }
    /// Compact stored-offset descriptor before pruning.
    pub fn stored_support(&self) -> StoredOffsetSupport {
        self.support
    }
    /// Full BSGS window length.
    pub fn d(&self) -> usize {
        self.shape.d
    }
    /// Slot distance between consecutive window positions.
    pub fn stride(&self) -> usize {
        self.shape.stride
    }
    /// Signed BSGS window offset in stride units.
    pub fn offset(&self) -> i64 {
        self.shape.offset
    }
    /// Number of BSGS baby steps including the trivial step.
    pub fn baby(&self) -> usize {
        crate::linalg::bsgs_split(self.shape.d).0
    }
    /// Number of BSGS giant blocks.
    pub fn giant(&self) -> usize {
        crate::linalg::bsgs_split(self.shape.d).1
    }
    /// Window indices whose unencoded main diagonal is nonzero.
    pub fn main_kept_indices(&self) -> &[usize] {
        &self.main
    }
    /// Second-term kept indices; absence differs from an empty second term.
    pub fn second_kept_indices(&self) -> Option<&[usize]> {
        self.second.as_deref()
    }
    /// Executed nontrivial baby rotations for the main term.
    pub fn main_baby_steps(&self) -> &[i32] {
        &self.baby_steps[0]
    }
    /// Executed baby rotations for the optional second term.
    pub fn second_baby_steps(&self) -> Option<&[i32]> {
        self.second.as_ref().map(|_| self.baby_steps[1].as_slice())
    }
    /// Executed nontrivial giant rotations shared across terms.
    pub fn shared_giant_steps(&self) -> &[i32] {
        &self.giant_steps
    }
    /// Exact input level and scale label.
    pub fn input(&self) -> CtMeta {
        self.input
    }
    /// Exact output level and scale label.
    pub fn output(&self) -> CtMeta {
        self.output
    }
    /// Exact searched pre-rescale product label bits.
    pub fn product_label_bits(&self) -> u64 {
        self.label
    }
    /// Actual finite plaintext diagonal scale bits.
    pub fn diagonal_scale_bits(&self) -> u64 {
        self.plain
    }
    /// Actual powf-derived per-stage magnitude bits.
    pub fn folded_magnitude_bits(&self) -> u64 {
        self.mag
    }
    /// Actual complex phase bits, real then imaginary.
    pub fn phase_bits(&self) -> [u64; 2] {
        self.phase
    }
    /// Actual complex folded scalar bits, real then imaginary.
    pub fn folded_factor_bits(&self) -> [u64; 2] {
        self.folded
    }
    /// Checked successful-execution operation counts.
    pub fn cost(&self) -> &OpCounter {
        &self.cost
    }
}
/// Actual bounded scalar coefficient computation and declared-length depth.
#[derive(Debug)]
pub struct EvalModMetadata {
    degree: usize,
    coefficients: Vec<u64>,
    support: Vec<usize>,
    depth: usize,
    input: CtMeta,
    polynomial: CtMeta,
    targets: Vec<CtMeta>,
    output: CtMeta,
    cost: OpCounter,
}
impl EvalModMetadata {
    /// Declared polynomial degree, retained even with trailing zero coefficients.
    pub fn declared_degree(&self) -> usize {
        self.degree
    }
    /// Actual interpolation coefficient bits including signed zero.
    pub fn coefficient_bits(&self) -> &[u64] {
        &self.coefficients
    }
    /// Indices of actual coefficients unequal to zero.
    pub fn effective_coefficient_support(&self) -> &[usize] {
        &self.support
    }
    /// Declared-length polynomial depth plus double-angle count.
    pub fn depth(&self) -> usize {
        self.depth
    }
    /// Exact input level and scale label.
    pub fn input(&self) -> CtMeta {
        self.input
    }
    /// Exact output metadata of the polynomial boundary.
    pub fn polynomial_target(&self) -> CtMeta {
        self.polynomial
    }
    /// Ordered exact output metadata of each double-angle boundary.
    pub fn double_angle_targets(&self) -> &[CtMeta] {
        &self.targets
    }
    /// Exact output level and scale label.
    pub fn output(&self) -> CtMeta {
        self.output
    }
    /// Checked successful-execution operation counts.
    pub fn cost(&self) -> &OpCounter {
        &self.cost
    }
}
/// Basis prefixes index the ordered primes in the enclosing requirements receipt.
#[derive(Debug, Clone, Copy)]
pub struct KeyRoleMetadata {
    n: usize,
    q_len: usize,
    p_len: usize,
    level: usize,
    digits: usize,
    alpha: usize,
    source: SecretDeclaration,
    destination: SecretDeclaration,
}
impl KeyRoleMetadata {
    /// Ring degree required by this key role.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Required prefix length of the enclosing ordered Q list.
    pub fn q_prefix_len(&self) -> usize {
        self.q_len
    }
    /// Required length of the enclosing ordered P list.
    pub fn p_len(&self) -> usize {
        self.p_len
    }
    /// Maximum required key basis level, not every execution level.
    pub fn level(&self) -> usize {
        self.level
    }
    /// Required decomposition digit count.
    pub fn digits(&self) -> usize {
        self.digits
    }
    /// Number of Q limbs in each full decomposition digit.
    pub fn alpha(&self) -> usize {
        self.alpha
    }
    /// Declared source secret of this key role.
    pub fn source(&self) -> SecretDeclaration {
        self.source
    }
    /// Declared destination secret of this key role.
    pub fn destination(&self) -> SecretDeclaration {
        self.destination
    }
}
/// Required maximum key basis/roles, not resident memory or generated key availability.
#[derive(Debug)]
pub struct BootstrapKeyRequirements {
    parameter_id: Vec<u8>,
    rotations: Vec<i32>,
    q: Vec<u64>,
    p: Vec<u64>,
    galois: KeyRoleMetadata,
    relin: KeyRoleMetadata,
    to_sparse: Option<KeyRoleMetadata>,
    to_main: Option<KeyRoleMetadata>,
}
impl BootstrapKeyRequirements {
    /// Content identity binding every key role to the exact enclosing parameters.
    pub fn parameter_identity_bytes(&self) -> &[u8] {
        &self.parameter_id
    }
    /// Sorted distinct normalized full-shape rotation key requirements.
    pub fn rotation_steps(&self) -> &[i32] {
        &self.rotations
    }
    /// Ordered maximum Q basis shared by the key roles.
    pub fn q(&self) -> &[u64] {
        &self.q
    }
    /// Ordered special-prime basis shared by the key roles.
    pub fn p(&self) -> &[u64] {
        &self.p
    }
    /// Full-basis Galois key role required for each generated rotation.
    pub fn galois(&self) -> &KeyRoleMetadata {
        &self.galois
    }
    /// The full-basis conjugation key role, used twice in CtS.
    pub fn conjugation(&self) -> Option<&KeyRoleMetadata> {
        Some(&self.galois)
    }
    /// Relinearization key is caller-provided; other roles are bootstrap-owned.
    pub fn relin(&self) -> &KeyRoleMetadata {
        &self.relin
    }
    /// Optional main-to-ephemeral key at q0*P, one digit.
    pub fn sse_to_sparse(&self) -> Option<&KeyRoleMetadata> {
        self.to_sparse.as_ref()
    }
    /// Optional ephemeral-to-main key at full Q*P.
    pub fn sse_to_main(&self) -> Option<&KeyRoleMetadata> {
        self.to_main.as_ref()
    }
}
/// Immutable root-free structural metadata. Success is conditional on runtime execution.
#[derive(Debug)]
pub struct BootstrapMetadata {
    boot_params: BootParams,
    metadata_limits: MetadataLimits,
    parameter_id: Vec<u8>,
    descriptor_id: Vec<u8>,
    main: SecretDeclaration,
    ephemeral: Option<SecretDeclaration>,
    input: CtMeta,
    output: CtMeta,
    ctos: Vec<DftStageMetadata>,
    stoc: Vec<DftStageMetadata>,
    evalmod: EvalModMetadata,
    keys: BootstrapKeyRequirements,
    execution: Vec<i32>,
    ctos_cost: OpCounter,
    stoc_cost: OpCounter,
    cost: OpCounter,
    switches: u64,
    proof: DftSupportProof,
    resources: MetadataResourceBounds,
}
impl BootstrapMetadata {
    /// Constructs a root-free conditional receipt using the default operational limits.
    pub fn new_with_verified_support(
        p: &CkksParams,
        bp: &BootParams,
        proof: &DftSupportProof,
    ) -> MetaResult<Self> {
        Self::with_limits_and_verified_support(p, bp, MetadataLimits::default(), proof)
    }
    /// Constructs a root-free conditional receipt with explicit operational limits.
    pub fn with_limits_and_verified_support(
        p: &CkksParams,
        bp: &BootParams,
        limits: MetadataLimits,
        proof: &DftSupportProof,
    ) -> MetaResult<Self> {
        let (after, c, depth, mut resources) = preflight(p, bp, limits)?;
        let mut budget = Budget::new(limits.max_peak_bytes, limits.max_polynomial_work);
        let (ctos_spec, stoc_spec) = super::bootstrapper::stage_specs(p, bp);
        let (ctos, ctos_cost) = direction_stages(p, bp, ctos_spec, bp.ctos_levels, &mut budget)?;
        let (stoc, stoc_cost) = direction_stages(p, bp, stoc_spec, bp.stoc_levels, &mut budget)?;
        let em_input = ctos.last().expect("positive band").output;
        let coefficients = super::evalmod::evalmod_coeffs_checked(bp, &mut budget)?;
        let mut em_cost =
            super::poly::chebyshev_cost_checked(&coefficients, -1.0, 1.0, &mut budget)?;
        let r = u64::try_from(bp.double_angle).map_err(|_| overflow("double angles"))?;
        increment(&mut em_cost.ct_ct_tensor, r)?;
        increment(&mut em_cost.relins, r)?;
        increment(&mut em_cost.rescales, r)?;
        let mut targets = budget.zeros(bp.double_angle + 1)?;
        super::evalmod::plan_double_angle_targets(
            em_input.level(),
            depth - bp.double_angle,
            em_input.scale(),
            &mut targets,
            |level, target| label_at(p, level, target, None, None),
            true,
        )
        .map_err(|error| match error {
            super::evalmod::TargetPlanError::Source(source) => source,
            super::evalmod::TargetPlanError::NonFiniteSeed { level } => {
                BootstrapMetadataError::NonFiniteIntermediate {
                    operation: "double angle sqrt(2 sigma)",
                    index: Some(level),
                }
            }
            super::evalmod::TargetPlanError::Unreachable { level } => {
                BootstrapMetadataError::ScalePlanning {
                    direction: None,
                    stage: None,
                    level,
                    source: HekitError::ScaleMismatch,
                }
            }
        })?;
        let mut double_targets = budget.vec(bp.double_angle, "double angle metadata")?;
        let cheb_out = em_input.level() - (depth - bp.double_angle);
        for i in 1..targets.len() {
            let level = cheb_out - (i - 1);
            let sigma = label_at(p, level, targets[i], None, Some(i))?;
            let square = finite(
                targets[i - 1] * targets[i - 1],
                "double angle square",
                Some(i),
            )?;
            if ((square / 2.0 - sigma) / sigma).abs() > 2f64.powi(-40) {
                return Err(BootstrapMetadataError::ScalePlanning {
                    direction: None,
                    stage: Some(i),
                    level,
                    source: HekitError::ScaleMismatch,
                });
            }
            double_targets.push(meta(p, level - 1, targets[i])?);
        }
        let mut coefficient_bits = budget.vec(c, "coefficient bits")?;
        let mut coefficient_support = budget.vec(c, "coefficient support")?;
        for (i, v) in coefficients.into_iter().enumerate() {
            coefficient_bits.push(v.to_bits());
            if v != 0.0 {
                coefficient_support.push(i);
            }
        }
        let evalmod = EvalModMetadata {
            degree: bp.cheb_degree,
            coefficients: coefficient_bits,
            support: coefficient_support,
            depth,
            input: em_input,
            polynomial: meta(p, cheb_out, targets[0])?,
            targets: double_targets,
            output: meta(p, em_input.level() - depth, em_input.scale())?,
            cost: em_cost,
        };
        let mut generated = budget.vec(p.slots(), "full-shape key union")?;
        let mut executed = budget.vec(p.slots(), "execution key union")?;
        let mut flags = budget.vec(p.slots(), "rotation flags")?;
        flags.resize(p.slots(), false);
        for s in ctos.iter().chain(&stoc) {
            for step in
                crate::linalg::bsgs_rotation_steps_with_offset_iter(s.d(), s.stride(), s.offset())
            {
                let r = i64::from(step).rem_euclid(p.slots() as i64) as usize;
                if r != 0 {
                    flags[r] = true;
                }
            }
        }
        for (i, yes) in flags.iter_mut().enumerate() {
            if *yes {
                generated.push(i as i32);
            }
            *yes = false;
        }
        for s in ctos.iter().chain(&stoc) {
            for &step in s.baby_steps.iter().flatten().chain(&s.giant_steps) {
                flags[step as usize] = true;
            }
        }
        for (i, &yes) in flags.iter().enumerate() {
            if yes {
                executed.push(i as i32);
            }
        }
        let main = secret(p.hamming_weight());
        let ephemeral = bp.ephemeral_hw.map(|h| secret(Some(h)));
        let role = KeyRoleMetadata {
            n: p.n(),
            q_len: p.q().len(),
            p_len: p.p().len(),
            level: p.max_level(),
            digits: p.dnum(),
            alpha: p.alpha(),
            source: main,
            destination: main,
        };
        let mut q = budget.vec(p.q().len(), "key Q basis")?;
        q.extend_from_slice(p.q());
        let mut special = budget.vec(p.p().len(), "key P basis")?;
        special.extend_from_slice(p.p());
        let parameter_id = identity(&mut budget, |w| write_parameters(w, p, bp))?;
        let mut key_parameter_id = budget.vec(parameter_id.len(), "key parameter identity")?;
        key_parameter_id.extend_from_slice(&parameter_id);
        let keys = BootstrapKeyRequirements {
            parameter_id: key_parameter_id,
            rotations: generated,
            q,
            p: special,
            galois: role,
            relin: role,
            to_sparse: ephemeral.map(|e| KeyRoleMetadata {
                q_len: 1,
                level: 0,
                digits: 1,
                alpha: 1,
                destination: e,
                ..role
            }),
            to_main: ephemeral.map(|e| KeyRoleMetadata { source: e, ..role }),
        };
        let mut cost = OpCounter::default();
        checked_merge(&mut cost, &ctos_cost)?;
        checked_merge(&mut cost, &stoc_cost)?;
        checked_merge(&mut cost, &evalmod.cost)?;
        checked_merge(&mut cost, &evalmod.cost)?;
        cost.sse_switches = if ephemeral.is_some() { 2 } else { 0 };
        cost.bootstraps = 1;
        let switches = cost
            .rotations
            .checked_add(cost.relins)
            .and_then(|v| v.checked_add(cost.sse_switches))
            .ok_or_else(|| overflow("total key switches"))?;
        let mut result = Self {
            boot_params: bp.clone(),
            metadata_limits: limits,
            parameter_id,
            descriptor_id: Vec::new(),
            main,
            ephemeral,
            input: meta(p, 0, p.scale())?,
            output: meta(p, after, p.scale())?,
            ctos,
            stoc,
            evalmod,
            keys,
            execution: executed,
            ctos_cost,
            stoc_cost,
            cost,
            switches,
            proof: *proof,
            resources,
        };
        result.descriptor_id = identity(&mut budget, |w| write_descriptor(w, &result))?;
        resources.requested_bytes = budget.bytes;
        result.resources = resources;
        Ok(result)
    }
    /// Exact original bootstrap configuration, retained immutably after construction.
    /// The existing parameter identity already binds these same fields.
    pub fn boot_params(&self) -> &BootParams {
        &self.boot_params
    }
    /// Exact original operational construction ceilings, distinct from charged resource use.
    /// These ceilings do not alter the existing cryptographic descriptor identity.
    pub fn metadata_limits(&self) -> MetadataLimits {
        self.metadata_limits
    }
    /// Canonical typed-byte schema version.
    pub fn schema_version(&self) -> u32 {
        SCHEMA_VERSION
    }
    /// Structural metadata algorithm revision.
    pub fn algorithm_revision(&self) -> u32 {
        ALGORITHM_REVISION
    }
    /// Versioned content identity of all CKKS and bootstrap parameter fields.
    pub fn parameter_identity_bytes(&self) -> &[u8] {
        &self.parameter_id
    }
    /// Versioned content identity of the derived receipt and proof diagnostics.
    pub fn descriptor_identity_bytes(&self) -> &[u8] {
        &self.descriptor_id
    }
    /// Main key-generator declaration, not sampled-secret evidence.
    pub fn declared_main_secret(&self) -> SecretDeclaration {
        self.main
    }
    /// Optional ephemeral key-generator declaration.
    pub fn declared_ephemeral_secret(&self) -> Option<SecretDeclaration> {
        self.ephemeral
    }
    /// Exact input level and scale label.
    pub fn input(&self) -> CtMeta {
        self.input
    }
    /// Exact output level and scale label.
    pub fn output(&self) -> CtMeta {
        self.output
    }
    /// CoeffToSlot stages in actual application order.
    pub fn ctos_stages(&self) -> &[DftStageMetadata] {
        &self.ctos
    }
    /// SlotToCoeff stages in actual application order.
    pub fn stoc_stages(&self) -> &[DftStageMetadata] {
        &self.stoc
    }
    /// Actual scalar polynomial support, cost and boundary targets.
    pub fn evalmod(&self) -> &EvalModMetadata {
        &self.evalmod
    }
    /// Required full-shape key roles; does not certify keys exist.
    pub fn generated_keys(&self) -> &BootstrapKeyRequirements {
        &self.keys
    }
    /// Unique normalized executed rotations; stage getters retain multiplicity.
    pub fn execution_rotation_steps(&self) -> &[i32] {
        &self.execution
    }
    /// Checked total CoeffToSlot operations including both conjugations.
    pub fn ctos_cost(&self) -> &OpCounter {
        &self.ctos_cost
    }
    /// Checked total SlotToCoeff operations.
    pub fn stoc_cost(&self) -> &OpCounter {
        &self.stoc_cost
    }
    /// Checked full seven-field successful-execution bootstrap count.
    pub fn cost_per_boot(&self) -> &OpCounter {
        &self.cost
    }
    /// Checked sum of rotations, relinearizations and SSE switches.
    pub fn key_switches_per_boot(&self) -> u64 {
        self.switches
    }
    /// Conditional structural and boundary-scale scope only.
    pub fn proof_scope(&self) -> ProofScope {
        ProofScope::StructuralCostAndBoundaryScales
    }
    /// Immutable diagnostics of the current-process finite support premise.
    pub fn dft_support_proof(&self) -> &DftSupportProof {
        &self.proof
    }
    /// Admission requirements deliberately left unvalidated.
    pub fn unvalidated_requirements(&self) -> &[UnvalidatedRequirement] {
        &UNVALIDATED
    }
    /// Requested-storage and work bounds for this construction.
    pub fn resource_bounds(&self) -> &MetadataResourceBounds {
        &self.resources
    }
}

fn preflight(
    p: &CkksParams,
    bp: &BootParams,
    limits: MetadataLimits,
) -> MetaResult<(usize, usize, usize, MetadataResourceBounds)> {
    if !(1..=16).contains(&bp.log_slots) || bp.log_slots.checked_add(1) != Some(p.log_n()) {
        return Err(invalid(
            "log_slots",
            "requires full packing and log_slots in 1..=16",
        ));
    }
    for (name, band) in [
        ("ctos_levels", bp.ctos_levels),
        ("stoc_levels", bp.stoc_levels),
    ] {
        if band == 0 || band > bp.log_slots as usize {
            return Err(invalid(name, "DFT band outside 1..=log_slots"));
        }
    }
    if bp.evalmod_levels == 0 || bp.k_intervals == 0 || bp.cheb_degree == 0 {
        return Err(invalid("EvalMod", "band, K and degree must be positive"));
    }
    if bp.double_angle >= 64 {
        return Err(invalid("double_angle", "must be below 64"));
    }
    let sum = bp
        .ctos_levels
        .checked_add(bp.evalmod_levels)
        .and_then(|v| v.checked_add(bp.stoc_levels))
        .ok_or_else(|| overflow("band sum"))?;
    let after = p
        .max_level()
        .checked_sub(sum)
        .filter(|&v| v > 0)
        .ok_or_else(|| invalid("bands", "must leave at least one compute level"))?;
    let c = bp
        .cheb_degree
        .checked_add(1)
        .ok_or_else(|| overflow("degree + 1"))?;
    let pow = c
        .checked_next_power_of_two()
        .ok_or_else(|| overflow("degree next power of two"))?;
    let d = pow.trailing_zeros() as usize;
    // Legacy optimal_split uses signed shifts/sums; the bounded scalar budget is checked first.
    let depth = d
        .checked_add(bp.double_angle)
        .ok_or_else(|| overflow("EvalMod depth"))?;
    if d >= 62 || depth > bp.evalmod_levels {
        return Err(invalid(
            "degree",
            "unsupported split shift or EvalMod depth exceeds band",
        ));
    }
    for weight in [p.hamming_weight(), bp.sparse_hw, bp.ephemeral_hw]
        .into_iter()
        .flatten()
    {
        if weight == 0 || weight > p.n() {
            return Err(invalid("weight", "outside 1..=N"));
        }
    }
    if bp.sparse_hw != p.hamming_weight() {
        return Err(invalid("sparse_hw", "does not match parameter declaration"));
    }
    if bp
        .ephemeral_hw
        .is_some_and(|h| h >= p.hamming_weight().unwrap_or(p.n() / 2))
    {
        return Err(invalid(
            "ephemeral_hw",
            "must be below declared/expected main weight",
        ));
    }
    if bp.k_intervals < bp.min_k_intervals(p) {
        return Err(invalid("k_intervals", "below existing minimum-K contract"));
    }
    // All branches allocating error text in the unchanged validator were excluded above.
    bp.validate(p)
        .map_err(|_| invalid("BootParams", "semantic validation invariant"))?;
    let c64 = u64::try_from(c).map_err(|_| overflow("coefficient conversion"))?;
    limit(MetadataResource::Coefficients, c64, limits.max_coefficients)?;
    let terms = c64
        .checked_mul(c64)
        .ok_or_else(|| overflow("interpolation terms"))?;
    limit(
        MetadataResource::InterpolationTerms,
        terms,
        limits.max_interpolation_terms,
    )?;
    let work = c64
        .checked_mul((d + 1) as u64)
        .and_then(|v| v.checked_mul(32))
        .ok_or_else(|| overflow("polynomial work bound"))?;
    limit(
        MetadataResource::PolynomialWork,
        work,
        limits.max_polynomial_work,
    )?;
    let scratch = c64
        .checked_mul((d + 1) as u64)
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| c64.checked_mul(24).and_then(|extra| v.checked_add(extra)))
        .ok_or_else(|| overflow("scalar storage bound"))?;
    limit(
        MetadataResource::RequestedBytes,
        scratch,
        limits.max_peak_bytes,
    )?;
    Ok((
        after,
        c,
        depth,
        MetadataResourceBounds {
            coefficients: c64,
            interpolation_terms: terms,
            polynomial_work: work,
            requested_bytes: 0,
            recursion_depth: d + 1,
        },
    ))
}

/// Guard all allocating P1a error branches before calling the exact shared search.
fn label_at(
    p: &CkksParams,
    level: usize,
    target: f64,
    direction: Option<Direction>,
    stage: Option<usize>,
) -> MetaResult<f64> {
    if level > p.max_level() {
        return Err(invalid("level", "above modulus chain"));
    }
    if !(target.is_finite() && target > 0.0) {
        return Err(invalid("target_scale", "must be positive finite"));
    }
    if level == 0 {
        return Err(BootstrapMetadataError::ScalePlanning {
            direction,
            stage,
            level,
            source: HekitError::NoLevelsLeft,
        });
    }
    let product = finite(target * p.q()[level] as f64, "target times modulus", stage)?;
    if product.to_bits() < 8 {
        return Err(invalid(
            "target_scale",
            "product too small for legacy ULP search",
        ));
    }
    exact_product_scale_at(p, level, target).map_err(|source| {
        BootstrapMetadataError::ScalePlanning {
            direction,
            stage,
            level,
            source,
        }
    })
}
fn meta(p: &CkksParams, level: usize, scale: f64) -> MetaResult<CtMeta> {
    if level > p.max_level() || !(scale.is_finite() && scale > 0.0) {
        return Err(invalid("CtMeta", "invalid level or positive finite scale"));
    }
    CtMeta::new(p, level, scale).map_err(|_| invalid("CtMeta", "guarded invariant"))
}
fn direction_stages(
    p: &CkksParams,
    bp: &BootParams,
    spec: super::linear::StageSpec,
    levels: usize,
    budget: &mut Budget,
) -> MetaResult<(Vec<DftStageMetadata>, OpCounter)> {
    // Bootstrap factors are positive. Do not impose this on the generic runtime helper.
    if !(spec.factor.re.is_finite() && spec.factor.re > 0.0 && spec.factor.im == 0.0) {
        return Err(missing(DftProofFailure::FoldedFactorBoundFailure));
    }
    let (mag, phase) = folded_factors(spec.factor, levels)
        .map_err(|_| missing(DftProofFailure::FoldedFactorBoundFailure))?;
    let split_factor = if spec.direction == Direction::CoeffToSlot {
        phase * mag * 0.5
    } else {
        phase * mag
    };
    for v in [mag, phase, split_factor] {
        if !(v.im == 0.0 && v.re.is_finite() && (2f64.powi(-66)..=2.0).contains(&v.re)) {
            return Err(missing(DftProofFailure::FoldedFactorBoundFailure));
        }
    }
    let mut groups = [(0u32, 0u32); 16];
    let mut lo = 0;
    for (i, r) in super::linear::stage_radices_iter(bp.log_slots, levels).enumerate() {
        groups[i] = (lo, lo + r - 1);
        lo += r;
    }
    let mut out = budget.vec(levels, "DFT stage receipts")?;
    let mut total = OpCounter::default();
    let mut source = spec.input_scale;
    for (ordinal, target) in
        super::linear::scale_path(spec.input_scale, spec.output_scale, levels).enumerate()
    {
        let index = if spec.direction == Direction::CoeffToSlot {
            levels - 1 - ordinal
        } else {
            ordinal
        };
        let (low, high) = groups[index];
        let support = StoredOffsetSupport {
            slots: p.slots(),
            stride: 1usize << low,
            radius: (1usize << (high - low + 1)) - 1,
        };
        let shape = shape_from_offsets(p.slots(), support.offsets());
        // Exclude every panic/overflow precondition of shared BSGS helpers explicitly.
        let block = shape
            .d
            .checked_mul(shape.stride)
            .ok_or_else(|| overflow("DFT shape block"))?;
        if !shape.d.is_power_of_two()
            || shape.stride == 0
            || block == 0
            || block > p.slots()
            || block > i32::MAX as usize
            || p.slots() % block != 0
            || shape.offset > 0
            || shape.offset <= -(shape.d as i64)
        {
            return Err(invalid(
                "DFT shape",
                "outside bounded full-slot BSGS domain",
            ));
        }
        let split = low == 0;
        let kept = |term: usize, i: usize| {
            let offset = ((shape.offset + i as i64) * shape.stride as i64)
                .rem_euclid(p.slots() as i64) as usize;
            support.contains(offset)
                && (!split || spec.direction == Direction::SlotToCoeff || offset % 2 == term)
        };
        let mut main = budget.vec(shape.d, "main kept support")?;
        let mut second = if split {
            Some(budget.vec(shape.d, "second kept support")?)
        } else {
            None
        };
        for i in 0..shape.d {
            if kept(0, i) {
                main.push(i);
            }
            if let Some(v) = &mut second {
                if kept(1, i) {
                    v.push(i);
                }
            }
        }
        let (baby, giant) = crate::linalg::bsgs_split(shape.d);
        let mut baby_steps = [
            budget.vec(baby, "main baby rotations")?,
            budget.vec(if split { baby } else { 0 }, "second baby rotations")?,
        ];
        let mut giant_steps = budget.vec(giant, "shared giant rotations")?;
        let mut cost = sparse_transform_cost(
            shape,
            p.slots(),
            1 + usize::from(split),
            kept,
            |term, step| {
                if let Some(t) = term {
                    baby_steps[t].push(step);
                } else {
                    giant_steps.push(step);
                }
            },
        )?;
        if spec.direction == Direction::CoeffToSlot && split {
            increment(&mut cost.rotations, 1)?;
        }
        checked_merge(&mut total, &cost)?;
        let level = spec.start_level - ordinal;
        let input = meta(p, level, source)?;
        let output = meta(p, level - 1, target)?;
        let label = label_at(p, level, target, Some(spec.direction), Some(ordinal))?;
        let plain = finite(label / source, "diagonal scale", Some(ordinal))?;
        if plain < crate::evaluator::MIN_PLAIN_SCALE {
            return Err(invalid("diagonal_scale", "below 2^20"));
        }
        // Successful shared helper only: every allocating error branch is guarded above.
        let plain = target_plain_scale_at(p, level, source, target).map_err(|source| {
            BootstrapMetadataError::ScalePlanning {
                direction: Some(spec.direction),
                stage: Some(ordinal),
                level,
                source,
            }
        })?;
        let folded = if split { split_factor } else { mag };
        out.push(DftStageMetadata {
            direction: spec.direction,
            low,
            high,
            ordinal,
            support,
            shape,
            main,
            second,
            baby_steps,
            giant_steps,
            input,
            output,
            label: label.to_bits(),
            plain: plain.to_bits(),
            mag: mag.re.to_bits(),
            phase: [phase.re.to_bits(), phase.im.to_bits()],
            folded: [folded.re.to_bits(), folded.im.to_bits()],
            cost,
        });
        source = target;
    }
    // The final conjugation belongs to the direction, after all individual stages.
    if spec.direction == Direction::CoeffToSlot {
        increment(&mut total.rotations, 1)?;
    }
    Ok((out, total))
}

/// Two-pass typed serialization: count first, fallibly reserve once, then write without growth.
struct Writer<'a> {
    out: Option<&'a mut Vec<u8>>,
    len: usize,
}
impl Writer<'_> {
    fn bytes(&mut self, bytes: &[u8]) -> MetaResult<()> {
        self.len = self
            .len
            .checked_add(bytes.len())
            .ok_or_else(|| overflow("identity length"))?;
        if let Some(out) = &mut self.out {
            if self.len > out.capacity() {
                return Err(overflow("identity reserve invariant"));
            }
            out.extend_from_slice(bytes);
        }
        Ok(())
    }
    fn u8(&mut self, v: u8) -> MetaResult<()> {
        self.bytes(&[v])
    }
    fn u32(&mut self, v: u32) -> MetaResult<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn u64(&mut self, v: u64) -> MetaResult<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn usize(&mut self, v: usize) -> MetaResult<()> {
        self.u64(u64::try_from(v).map_err(|_| overflow("identity usize"))?)
    }
    fn i64(&mut self, v: i64) -> MetaResult<()> {
        self.bytes(&v.to_le_bytes())
    }
    fn blob(&mut self, v: &[u8]) -> MetaResult<()> {
        self.usize(v.len())?;
        self.bytes(v)
    }
    fn option(&mut self, v: Option<usize>) -> MetaResult<()> {
        match v {
            None => self.u8(0),
            Some(v) => {
                self.u8(1)?;
                self.usize(v)
            }
        }
    }
    fn secret(&mut self, v: SecretDeclaration) -> MetaResult<()> {
        match v {
            SecretDeclaration::DenseTernary => self.u8(0),
            SecretDeclaration::FixedWeight(h) => {
                self.u8(1)?;
                self.usize(h)
            }
        }
    }
    fn meta(&mut self, v: CtMeta) -> MetaResult<()> {
        self.usize(v.level())?;
        self.u64(v.scale_bits())
    }
    fn indices(&mut self, v: &[usize]) -> MetaResult<()> {
        self.usize(v.len())?;
        for &x in v {
            self.usize(x)?;
        }
        Ok(())
    }
    fn steps(&mut self, v: &[i32]) -> MetaResult<()> {
        self.usize(v.len())?;
        for &x in v {
            self.i64(i64::from(x))?;
        }
        Ok(())
    }
    fn counter(&mut self, c: &OpCounter) -> MetaResult<()> {
        for x in [
            c.rotations,
            c.relins,
            c.ct_ct_tensor,
            c.ct_pt_mults,
            c.rescales,
            c.sse_switches,
            c.bootstraps,
        ] {
            self.u64(x)?;
        }
        Ok(())
    }
    fn role(&mut self, r: KeyRoleMetadata) -> MetaResult<()> {
        for x in [r.n, r.q_len, r.p_len, r.level, r.digits, r.alpha] {
            self.usize(x)?;
        }
        self.secret(r.source)?;
        self.secret(r.destination)
    }
}
fn identity(
    budget: &mut Budget,
    write: impl Fn(&mut Writer<'_>) -> MetaResult<()>,
) -> MetaResult<Vec<u8>> {
    let mut count = Writer { out: None, len: 0 };
    write(&mut count)?;
    let mut v = budget.vec(count.len, "canonical identity")?;
    write(&mut Writer {
        out: Some(&mut v),
        len: 0,
    })?;
    Ok(v)
}
fn write_parameters(w: &mut Writer<'_>, p: &CkksParams, bp: &BootParams) -> MetaResult<()> {
    w.bytes(b"hekit/bootstrap-parameters\0")?;
    w.u32(SCHEMA_VERSION)?;
    w.u32(ALGORITHM_REVISION)?;
    for values in [p.q(), p.p()] {
        w.usize(values.len())?;
        for &v in values {
            w.u64(v)?;
        }
    }
    for values in [p.q_bits(), p.p_bits()] {
        w.usize(values.len())?;
        for &v in values {
            w.u64(u64::from(v))?;
        }
    }
    w.u64(u64::from(p.log_n()))?;
    w.u64(p.scale().to_bits())?;
    w.usize(p.dnum())?;
    w.option(p.hamming_weight())?;
    w.u64(u64::from(bp.log_slots))?;
    for v in [
        bp.ctos_levels,
        bp.evalmod_levels,
        bp.stoc_levels,
        bp.k_intervals,
        bp.cheb_degree,
        bp.double_angle,
    ] {
        w.usize(v)?;
    }
    w.option(bp.sparse_hw)?;
    w.option(bp.ephemeral_hw)
}
fn write_descriptor(w: &mut Writer<'_>, m: &BootstrapMetadata) -> MetaResult<()> {
    w.bytes(b"hekit/bootstrap-descriptor\0")?;
    w.u32(SCHEMA_VERSION)?;
    w.u32(ALGORITHM_REVISION)?;
    w.blob(&m.parameter_id)?;
    w.meta(m.input)?;
    w.meta(m.output)?;
    for stages in [&m.ctos, &m.stoc] {
        w.usize(stages.len())?;
        for s in stages {
            w.u8(if s.direction == Direction::CoeffToSlot {
                0
            } else {
                1
            })?;
            w.u32(s.low)?;
            w.u32(s.high)?;
            w.usize(s.ordinal)?;
            for x in [
                s.support.slots,
                s.support.stride,
                s.support.radius,
                s.d(),
                s.stride(),
                s.baby(),
                s.giant(),
            ] {
                w.usize(x)?;
            }
            w.i64(s.offset())?;
            w.indices(&s.main)?;
            match &s.second {
                None => w.u8(0)?,
                Some(v) => {
                    w.u8(1)?;
                    w.indices(v)?;
                }
            }
            for steps in &s.baby_steps {
                w.steps(steps)?;
            }
            w.steps(&s.giant_steps)?;
            w.meta(s.input)?;
            w.meta(s.output)?;
            for x in [
                s.label,
                s.plain,
                s.mag,
                s.phase[0],
                s.phase[1],
                s.folded[0],
                s.folded[1],
            ] {
                w.u64(x)?;
            }
            w.counter(&s.cost)?;
        }
    }
    let e = &m.evalmod;
    w.u32(COEFFICIENT_ALGORITHM_REVISION)?;
    w.usize(e.degree)?;
    w.usize(e.depth)?;
    w.usize(e.coefficients.len())?;
    for &v in &e.coefficients {
        w.u64(v)?;
    }
    w.indices(&e.support)?;
    w.meta(e.input)?;
    w.meta(e.polynomial)?;
    w.usize(e.targets.len())?;
    for &t in &e.targets {
        w.meta(t)?;
    }
    w.meta(e.output)?;
    w.counter(&e.cost)?;
    w.blob(&m.keys.parameter_id)?;
    w.steps(&m.keys.rotations)?;
    w.role(m.keys.galois)?;
    w.role(m.keys.relin)?;
    for role in [m.keys.to_sparse, m.keys.to_main] {
        match role {
            None => w.u8(0)?,
            Some(r) => {
                w.u8(1)?;
                w.role(r)?;
            }
        }
    }
    w.steps(&m.execution)?;
    w.counter(&m.ctos_cost)?;
    w.counter(&m.stoc_cost)?;
    w.counter(&m.cost)?;
    w.u64(m.switches)?;
    w.u8(0)?;
    w.usize(UNVALIDATED.len())?;
    for issue in UNVALIDATED {
        w.u8(issue as u8)?;
    }
    w.bytes(b"hekit/dft-support-proof\0")?;
    w.u32(DFT_SUPPORT_PROOF_REVISION)?;
    for value in [
        m.proof.configured_compiler(),
        m.proof.target(),
        m.proof.host(),
    ] {
        match value {
            None => w.u8(0)?,
            Some(s) => {
                w.u8(1)?;
                w.blob(s.as_bytes())?;
            }
        }
    }
    w.u64(m.proof.roots)?;
    for value in [
        m.proof.root_min,
        m.proof.root_max,
        m.proof.inverse_min,
        m.proof.inverse_max,
    ] {
        w.u64(value.to_bits())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_counter_overflow_covers_all_seven_fields() {
        for field in 0..7 {
            let mut base = OpCounter::default();
            let mut add = OpCounter::default();
            fn fields(c: &mut OpCounter) -> [&mut u64; 7] {
                [
                    &mut c.rotations,
                    &mut c.relins,
                    &mut c.ct_ct_tensor,
                    &mut c.ct_pt_mults,
                    &mut c.rescales,
                    &mut c.sse_switches,
                    &mut c.bootstraps,
                ]
            }
            *fields(&mut base)[field] = u64::MAX;
            *fields(&mut add)[field] = 1;
            assert!(matches!(
                checked_merge(&mut base, &add),
                Err(BootstrapMetadataError::ArithmeticOverflow { .. })
            ));
        }
    }
    #[test]
    fn scale_error_guards_precede_legacy_allocating_errors() {
        let p = CkksParams::test_tiny();
        for (level, target) in [
            (usize::MAX, 1.0),
            (1, 0.0),
            (1, -1.0),
            (1, f64::NAN),
            (1, f64::INFINITY),
            (1, f64::MAX),
        ] {
            assert!(label_at(&p, level, target, None, None).is_err());
        }
        assert!(matches!(
            label_at(&p, 0, 1.0, None, None),
            Err(BootstrapMetadataError::ScalePlanning {
                source: HekitError::NoLevelsLeft,
                ..
            })
        ));
        assert!(meta(&p, 0, f64::INFINITY).is_err());
        assert!(meta(&p, usize::MAX, 1.0).is_err());
        // Independently fixed P1a counterexample: no label in +/-32 ULP reaches it.
        let p = CkksParams::new(12, &[60, 20, 20], &[60], 20, 3).unwrap();
        let target = f64::from_bits(0x4135_eb85_1eb8_aeb9);
        let center = (target * p.q()[1] as f64).to_bits();
        assert!(
            (center - 32..=center + 32)
                .all(|bits| f64::from_bits(bits) / p.q()[1] as f64 != target)
        );
        assert!(matches!(
            label_at(&p, 1, target, None, None),
            Err(BootstrapMetadataError::ScalePlanning {
                source: HekitError::ScaleMismatch,
                ..
            })
        ));
    }
    #[cfg(any(
        all(
            target_arch = "aarch64",
            target_os = "macos",
            target_pointer_width = "64"
        ),
        all(
            target_arch = "x86_64",
            target_vendor = "unknown",
            target_os = "linux",
            target_env = "gnu",
            target_pointer_width = "64"
        )
    ))]
    #[test]
    fn finite_scalar_proof_on_supported_target() {
        let proof = DftSupportProof::verify_current().unwrap();
        assert_eq!(proof.root_count(), 65535);
        assert!(proof.configured_compiler().unwrap().contains("rustc "));
        let expected_target = if cfg!(all(
            target_arch = "aarch64",
            target_os = "macos",
            target_pointer_width = "64"
        )) {
            "aarch64-apple-darwin"
        } else {
            "x86_64-unknown-linux-gnu"
        };
        assert_eq!(proof.target(), Some(expected_target));
        eprintln!(
            "current-process DFT support proof: TARGET={:?}, HOST(build)={:?}, roots={}, root_squared_norm={:?}, inverse_squared_norm={:?}\nconfigured compiler -vV:\n{}",
            proof.target(),
            proof.host(),
            proof.root_count(),
            proof.root_squared_norm_bounds(),
            proof.inverse_squared_norm_bounds(),
            proof.configured_compiler().unwrap()
        );
    }
    #[test]
    fn folded_extremes_and_complex_support_operations() {
        for k in [1usize, 2, 13, usize::MAX] {
            for levels in 1..=16 {
                for factor in [1.0 / k as f64, 1.0 / (2.0 * std::f64::consts::PI)] {
                    let (mag, phase) = folded_factors(Complex64::new(factor, 0.0), levels).unwrap();
                    for z in [mag, phase * mag, phase * mag * 0.5] {
                        assert!(z.im == 0.0 && (2f64.powi(-66)..=2.0).contains(&z.re));
                    }
                }
            }
        }
        // Cancellation in one Cartesian component does not mean zero complex norm.
        let z = Complex64::new(1.0, 1.0) * Complex64::new(1.0, 1.0);
        assert_eq!(z.re, 0.0);
        assert_eq!(z.im, 2.0);
        let mut z = Complex64::new(1.0, 0.0);
        for _ in 0..16 {
            z *= Complex64::new(0.125, 0.0);
        }
        z *= 2f64.powi(-66);
        assert!(z.norm() > 2f64.powi(-120));
        // Shared runtime helper must retain its generic complex/negative factor domain.
        assert!(folded_factors(Complex64::new(-1.0, 0.0), 2).is_ok());
        assert!(folded_factors(Complex64::new(0.0, 1.0), 2).is_ok());
    }
    #[test]
    fn parameter_bytes_bind_every_field_even_if_changed_value_is_invalid() {
        let p = CkksParams::n16_boot();
        let bp = BootParams::default_n16();
        let id = |p: &CkksParams, bp: &BootParams| {
            identity(&mut Budget::new(1 << 20, 1 << 20), |w| {
                write_parameters(w, p, bp)
            })
            .unwrap()
        };
        let base = id(&p, &bp);
        let mut variants = Vec::new();
        let mut v = bp.clone();
        v.log_slots += 1;
        variants.push(v);
        for field in 0..6 {
            let mut v = bp.clone();
            let fields = [
                &mut v.ctos_levels,
                &mut v.evalmod_levels,
                &mut v.stoc_levels,
                &mut v.k_intervals,
                &mut v.cheb_degree,
                &mut v.double_angle,
            ];
            *fields.into_iter().nth(field).unwrap() += 1;
            variants.push(v);
        }
        let mut v = bp.clone();
        v.sparse_hw = Some(1);
        variants.push(v);
        let mut v = bp.clone();
        v.ephemeral_hw = None;
        variants.push(v);
        for changed in variants {
            assert_ne!(id(&p, &changed), base);
        }
        for (bits, scale, dnum) in [
            (p.q_bits().to_vec(), 41, p.dnum()),
            (p.q_bits().to_vec(), 42, 16),
        ] {
            let changed = CkksParams::new(p.log_n(), &bits, p.p_bits(), scale, dnum).unwrap();
            assert_ne!(id(&changed, &bp), base);
        }
        let mut q = p.q_bits().to_vec();
        q.swap(0, 1);
        let changed = CkksParams::new(p.log_n(), &q, p.p_bits(), 42, p.dnum()).unwrap();
        assert_ne!(id(&changed, &bp), base);
    }
}

#[cfg(test)]
mod proof_failure_tests {
    use super::*;
    // These tables exercise eligibility policy only; no proof token is constructed.
    #[test]
    fn exact_targets_are_eligible_for_current_process_verification() {
        for target in ["aarch64-apple-darwin", "x86_64-unknown-linux-gnu"] {
            assert!(
                check_proof_environment(64, Some(target), Some("rustc test"), Some("host")).is_ok(),
                "exact target must be eligible: {target}"
            );
        }
    }
    #[test]
    fn ineligible_diagnostics_return_exact_missing_requirement_and_reason() {
        let good = Some("aarch64-apple-darwin");
        let compiler = Some("rustc test");
        let host = Some("host");
        let mut cases = vec![
            (
                32,
                good,
                compiler,
                host,
                DftProofFailure::UnsupportedPointerWidth,
            ),
            // Pointer-width rejection must precede missing diagnostic rejection.
            (
                32,
                None,
                None,
                None,
                DftProofFailure::UnsupportedPointerWidth,
            ),
        ];
        for target in [
            "x86_64-unknown-linux-musl",
            "aarch64-unknown-linux-gnu",
            "x86_64-pc-windows-gnu",
            "unverified-target",
        ] {
            cases.push((
                64,
                Some(target),
                compiler,
                host,
                DftProofFailure::UnsupportedTargetEnvironment,
            ));
        }
        for absent in [None, Some("")] {
            for (target, compiler, host) in [
                (absent, compiler, host),
                (good, absent, host),
                (good, compiler, absent),
            ] {
                cases.push((
                    64,
                    target,
                    compiler,
                    host,
                    DftProofFailure::ConfiguredCompilerDiagnosticsUnavailable,
                ));
            }
        }
        for (bits, target, compiler, host, expected) in cases {
            assert!(
                matches!(
                    check_proof_environment(bits, target, compiler, host),
                    Err(BootstrapMetadataError::MissingCode {
                        requirement: MissingRequirement::DftKeptSupportProof,
                        reason,
                    }) if reason == expected
                ),
                "bits={bits}, target={target:?}, compiler={compiler:?}, host={host:?}, expected={expected:?}"
            );
        }
    }
    #[test]
    fn checked_stage_scales_and_folded_values_reject_before_encoding() {
        let p = CkksParams::test_tiny();
        let bp = BootParams {
            log_slots: 4,
            ctos_levels: 4,
            stoc_levels: 4,
            evalmod_levels: 2,
            k_intervals: 6,
            cheb_degree: 3,
            double_angle: 0,
            sparse_hw: None,
            ephemeral_hw: Some(4),
        };
        let (spec, _) = super::super::bootstrapper::stage_specs(&p, &bp);
        for bad in [
            super::super::linear::StageSpec {
                input_scale: f64::from_bits(1),
                ..spec
            },
            super::super::linear::StageSpec {
                output_scale: 1.0,
                input_scale: 1e300,
                ..spec
            },
            super::super::linear::StageSpec {
                output_scale: f64::MAX,
                ..spec
            },
            super::super::linear::StageSpec {
                factor: Complex64::new(f64::INFINITY, 0.0),
                ..spec
            },
        ] {
            assert!(direction_stages(&p, &bp, bad, 1, &mut Budget::new(1 << 20, 1 << 20)).is_err());
        }
    }
}
